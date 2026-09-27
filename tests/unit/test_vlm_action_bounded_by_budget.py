"""The VLM loop's time budget bounds the action in flight, not only the gaps.

MEASURED LIVE 2026-09-27 (installed build Nunba 53927f85 / HARTOS 62649bc1,
daemon goal b18bba6f, prompt 88555124130, task #125), frozen_debug.log:

    15:34:28.921  VLM action: open_file_gui   (path ...\\coding\\feedback_collector.py)
    17:30:43.816  VLM loop hit ETA limit (1800s) at iteration 1
    17:30:43.818  VLM loop finished: 1 actions in 6979.1s (exit_reason=timeout)

and the safety audit JSONL wrote that action's record at 17:30:43 with no
error, i.e. execute_action RETURNED after ~6974 s.  The daemon's thread
(Thread-42) logged nothing in between.  The 1800 s ETA was checked only at
the top of each iteration, so one action that did not return held the
daemon for 1 h 56 min.  On Windows open_file_gui is os.startfile, an
in-process ShellExecute call that no subprocess timeout can reach; `.py` on
that machine is associated with pycharm64.exe.

These tests drive the REAL loop with a tool that blocks longer than the
budget and assert on the wall clock and on the result the caller gets.
"""
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from integrations.vlm.local_loop import run_local_agentic_loop  # noqa: E402

#: Budget given to the loop in these tests, seconds.
BUDGET_S = 1.0
#: How long the stuck tool would block if nothing bounded it.  Long enough
#: that an unbounded loop is unmistakable on the clock, short enough that a
#: mutant run still finishes.
STUCK_S = 8.0
#: Allowed overrun past the budget: the screenshot/VLM mocks, thread start,
#: and the bookkeeping after the timeout.  Far below STUCK_S - BUDGET_S.
MARGIN_S = 2.0

_OPEN_FILE = ('{"Next Action": "open_file_gui", "path": "C:\\\\x\\\\feedback_collector.py", '
              '"Reasoning": "open the file", "Status": "IN_PROGRESS"}')


def _backend(response, call_api=None):
    b = MagicMock()
    b.route_task.return_value = 'multi_step'
    if call_api is not None:
        b._call_api.side_effect = call_api
    else:
        b._call_api.return_value = response
    b.try_taskbar_pre_check.return_value = None
    b.detect_grounding_bias.return_value = None
    b.retry_with_elimination.return_value = None
    return b


def _run_loop(execute_action, *, budget=BUDGET_S, call_api=None,
              message_extra=None):
    lct = MagicMock()
    lct.take_screenshot.return_value = 'base64'
    lct.execute_action.side_effect = execute_action
    lct.VLM_IMG_W = 1280
    lct.VLM_IMG_H = 720
    message = {'instruction_to_vlm_agent': 'Open feedback_collector.py',
               'max_ETA_in_seconds': budget}
    message.update(message_extra or {})
    with patch.dict('sys.modules', {'integrations.vlm.local_computer_tool': lct}):
        with patch('integrations.vlm.qwen3vl_backend.get_qwen3vl_backend',
                   return_value=_backend(_OPEN_FILE, call_api)),                 patch('integrations.vlm.activity_stream.resolve_steering_agent_id',
                      return_value=''):
            # ^ a DB lookup; its first import alone costs over a second on
            # a loaded box, which would spend the whole test budget before
            # iteration 1 and leave nothing under test.
            t0 = time.monotonic()
            result = run_local_agentic_loop(message, tier='inprocess',
                                            max_iterations=5)
            elapsed = time.monotonic() - t0
    return result, elapsed, lct


@pytest.fixture
def stuck_tool():
    """An action that blocks like the live os.startfile did, until released."""
    release = threading.Event()
    entered = threading.Event()

    def _execute(action, tier, **_kw):
        entered.set()
        release.wait(STUCK_S)
        return {'output': 'Opened %s' % action.get('path')}

    _execute.entered = entered
    yield _execute
    release.set()


@pytest.mark.usefixtures('computer_control_granted')
class TestTheBudgetBoundsTheActionInFlight:

    def test_a_stuck_action_returns_within_the_budget(self, stuck_tool):
        result, elapsed, _ = _run_loop(stuck_tool)
        assert stuck_tool.entered.is_set(), 'the action never started'
        assert elapsed < BUDGET_S + MARGIN_S, (
            'the loop held the caller %.1fs on a %.1fs budget' % (elapsed, BUDGET_S))

    def test_the_timeout_is_reported_honestly(self, stuck_tool):
        result, _, _ = _run_loop(stuck_tool)
        assert result['status'] == 'incomplete'
        assert result['exit_reason'] == 'timeout'
        last = result['extracted_responses'][-1]
        assert last['type'] == 'action'
        assert last['content']['action'] == 'open_file_gui'
        assert last['content']['ok'] is False
        assert 'FAILED' in last['content']['result']
        assert 'budget' in last['content']['result']

    def test_the_caller_is_told_it_ran_out_of_time(self, stuck_tool):
        from integrations.vlm.response_view import outcome_summary
        result, _, _ = _run_loop(stuck_tool)
        assert outcome_summary(result).startswith('Ran out of time')

    def test_the_cause_is_logged(self, stuck_tool, caplog):
        with caplog.at_level('WARNING', logger='hevolve.vlm.local_loop'):
            _run_loop(stuck_tool)
        lines = [r.getMessage() for r in caplog.records
                 if r.name == 'hevolve.vlm.local_loop']
        assert any('open_file_gui' in m and 'budget' in m for m in lines), lines

    def test_no_budget_left_means_the_action_never_starts(self):
        """The VLM call itself can spend the rest of the budget; the action
        must then not fire on the owner's machine after the deadline."""
        started = threading.Event()

        def _execute(action, tier, **_kw):
            started.set()
            return {'output': 'Opened'}

        def _slow_vlm(_messages):
            time.sleep(0.6)
            return _OPEN_FILE

        result, _, _ = _run_loop(_execute, budget=0.3, call_api=_slow_vlm)
        assert not started.is_set()
        assert result['exit_reason'] == 'timeout'
        assert result['extracted_responses'][-1]['content']['ok'] is False


@pytest.mark.usefixtures('computer_control_granted')
class TestAFastActionIsUnchanged:

    def test_its_result_and_thread_context_reach_the_loop(self):
        """The action runs on a worker now; the shell tool inside it reads
        prompt_id and the activity run from hartos.threadlocal, so both must
        still be the loop's."""
        from hartos.threadlocal import thread_local_data
        seen = {}

        def _execute(action, tier, **_kw):
            seen['prompt_id'] = thread_local_data.get_prompt_id()
            seen['run'] = thread_local_data.get_activity_run()
            seen['thread'] = threading.current_thread()
            return {'output': 'Opened it'}

        # No user_id: record_activity then stops before the ledger, so the
        # test writes nothing to the real task store.
        with patch('integrations.vlm.local_loop.time.sleep'):
            result, _, _ = _run_loop(
                _execute, budget=30, message_extra={'prompt_id': 'p-77'})
        assert seen['prompt_id'] == 'p-77'
        assert seen['run'] and seen['run']['prompt_id'] == 'p-77'
        first = result['extracted_responses'][0]
        assert first['content']['ok'] is True
        assert first['content']['result'] == 'Opened it'

    def test_an_action_that_raises_is_still_an_iteration_error(self):
        def _execute(action, tier, **_kw):
            raise RuntimeError('pyautogui exploded')

        with patch('integrations.vlm.local_loop.time.sleep'):
            result, _, _ = _run_loop(_execute, budget=30)
        assert result['exit_reason'] == 'action_error'
        assert result['extracted_responses'][0]['type'] == 'error'
        assert 'pyautogui exploded' in result['extracted_responses'][0]['content']
