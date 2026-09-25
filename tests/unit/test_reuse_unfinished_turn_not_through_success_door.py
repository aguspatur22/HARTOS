"""An action that GAVE UP must not reach the user through the success door.

MEASURED LIVE 2026-09-24 on the installed Nunba (HARTOS 20bfca03d), agent
88094979291 ("summarize into exactly three bullet points"), driven as
livetest_reuse_verify_1790351204 through POST /custom_gpt:

    21:17:52,083  Action 1: in_progress -> gave_up
    (74 ms later)
    [SYNTHESIS] the action already wrote the answer -- recovered 175 chars

The reply was a promise with no bullets, and nothing in it said action 1 had
given up.  Actions 2..4 had never been reached.

WHY.  ``_reuse_synthesis_turn`` picks between the honest-incomplete steer and
the success doors (written-answer recovery, the no-data report,
``_REUSE_SYNTHESIS_STEER``) by ONE question: ``_reuse_outstanding_tools``.
That adapter returns [] for a prose action that names no tool, so for that
action it always answered "nothing outstanding".  The lifecycle had already
recorded the truth -- GAVE_UP from ``force_state_through_valid_path`` in
``_advance_reuse_action``, which also leaves the action pointer ON the
failed action -- and the synthesis turn never read it.

These tests drive the REAL ``_reuse_synthesis_turn`` with the REAL lifecycle
state store (``lifecycle_hooks.action_states``), mocking only the steering
seat (``chat_instructor``) and the tool-evidence adapter, exactly as the
sibling suites do.

    python -m pytest tests/unit/test_reuse_unfinished_turn_not_through_success_door.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

SESSION = 'livetest_reuse_verify_1790351204_88094979291'

# The live shape: the dispatch that opened action 1, the model's promise (a
# user-readable sentence, so the recovery would take it), and the verdict tail.
DISPATCH = {
    'role': 'user', 'name': 'ChatInstructor',
    'content': ("Perform this action -> Action #1:Receive the input text from "
                "the user.\n follow these steps: [{'Read the text': "
                "{'tool_name': '', 'code': ''}}]"),
}
PROMISE = ("Sure! I'll summarize the Quenfar bridge text into exactly three "
           "bullet points for you as soon as it is ready.")
VERDICT = ('{"status": "completed", "action_id": 1, "message": "Input text '
           'received."}')


def _history():
    return [dict(DISPATCH),
            {'role': 'assistant', 'name': 'Assistant', 'content': PROMISE},
            {'role': 'user', 'name': 'StatusVerifier', 'content': VERDICT}]


class _Task:
    """The pipeline's per-session task: an action list and a pointer."""

    def __init__(self, current_action, n_actions=4):
        self.current_action = current_action
        self.actions = [{'action': 'step %d' % i} for i in range(1, n_actions + 1)]
        self.evidence_vacuous_action = None

    def get_action(self, i):
        return self.actions[i]


class _Chat:
    def __init__(self, messages):
        self.agents = []
        self.messages = list(messages)


class _Manager:
    def __init__(self):
        self._oai_messages = {}


class _Recorder:
    """Stands in for chat_instructor; captures the steer instead of sending."""

    def __init__(self):
        self.messages = []

    def initiate_chat(self, recipient=None, message=None, **kw):
        self.messages.append(message)


@pytest.fixture
def rr():
    import hartos.reuse_recipe as mod          # a skip here would be vacuous
    return mod


@pytest.fixture
def lh():
    import hartos.lifecycle_hooks as mod
    return mod


def _run(rr, lh, monkeypatch, task, state, outstanding=()):
    monkeypatch.setattr(rr, '_reuse_outstanding_tools',
                        lambda *a, **k: list(outstanding), raising=True)
    monkeypatch.setitem(rr.user_tasks, SESSION, task)
    if state is not None:
        monkeypatch.setitem(lh.action_states, SESSION,
                            {task.current_action: state})
    chat = _Chat(_history())
    rec = _Recorder()
    posted = rr._reuse_synthesis_turn(SESSION, chat, _Manager(), rec)
    return posted, chat, rec


class TestAGaveUpActionIsReportedAsUnfinished:

    def test_the_promise_is_not_recovered_as_the_answer(
            self, rr, lh, monkeypatch):
        """THE DEFECT.  RED before the fix: the recovery appended PROMISE as
        the tail and the extractor delivered it."""
        _posted, chat, _rec = _run(rr, lh, monkeypatch, _Task(1),
                                   lh.ActionState.GAVE_UP)
        assert (chat.messages[-1].get('content') or '') != PROMISE, (
            'action 1 GAVE_UP and the synthesis turn still recovered the '
            "model's promise as the finished answer (live 21:17:52)")

    def test_the_honest_incomplete_steer_is_posted(self, rr, lh, monkeypatch):
        posted, _chat, rec = _run(rr, lh, monkeypatch, _Task(1),
                                  lh.ActionState.GAVE_UP)
        assert posted is True and len(rec.messages) == 1
        steer = rec.messages[0]
        assert steer.startswith(rr._REUSE_SYNTHESIS_STEER_INCOMPLETE.split(
            '{unrun}')[0]), 'the steer posted is not the incomplete steer'
        assert 'their tools have already run' not in steer.lower()

    def test_the_steer_names_the_gave_up_action_and_the_unreached_ones(
            self, rr, lh, monkeypatch):
        """The model cannot report what it is not told."""
        _posted, _chat, rec = _run(rr, lh, monkeypatch, _Task(1),
                                   lh.ActionState.GAVE_UP)
        steer = rec.messages[0]
        assert 'action 1 gave up' in steer, steer
        assert 'actions 2 to 4 were not reached' in steer, steer

    def test_the_no_data_report_cannot_speak_for_a_gave_up_action(
            self, rr, lh, monkeypatch):
        """The vacuity stamp names an EARLIER finished action (0 = the one
        before action 1); even when it lines up it must not turn a give-up
        into "there is nothing recorded"."""
        task = _Task(1)
        task.evidence_vacuous_action = 0
        _posted, chat, rec = _run(rr, lh, monkeypatch, task,
                                  lh.ActionState.GAVE_UP)
        assert rec.messages, 'a give-up was answered by the no-data report'
        assert (chat.messages[-1].get('content') or '') != rr._REUSE_NO_DATA_REPORT


class TestAnUnfinishedActionIsNotTheWholeAnswer:

    def test_a_mid_recipe_stop_is_reported_as_unfinished(
            self, rr, lh, monkeypatch):
        """Turn ended at action 2 of 4 while it was still IN_PROGRESS (round
        budget spent): the text written so far is not the finished answer."""
        _posted, chat, rec = _run(rr, lh, monkeypatch, _Task(2),
                                  lh.ActionState.IN_PROGRESS)
        assert rec.messages, 'a half-walked recipe went out as the answer'
        assert 'action 2 did not finish' in rec.messages[0]
        assert 'actions 3 to 4 were not reached' in rec.messages[0]
        assert (chat.messages[-1].get('content') or '') != PROMISE

    def test_the_last_action_gave_up_names_no_unreached_actions(
            self, rr, lh, monkeypatch):
        _posted, _chat, rec = _run(rr, lh, monkeypatch, _Task(4),
                                   lh.ActionState.GAVE_UP)
        assert 'action 4 gave up' in rec.messages[0]
        assert 'not reached' not in rec.messages[0]

    def test_unrun_tools_and_the_give_up_are_both_named(
            self, rr, lh, monkeypatch):
        _posted, _chat, rec = _run(rr, lh, monkeypatch, _Task(1),
                                   lh.ActionState.GAVE_UP,
                                   outstanding=['google_search'])
        assert 'google_search' in rec.messages[0]
        assert 'action 1 gave up' in rec.messages[0]


class TestAFinishedTurnStillUsesTheSuccessDoor:
    """ANTI-VACUITY: a fix that always hedges would destroy the honest case
    (test_reuse_recovers_the_written_answer's three bullet points)."""

    def test_every_action_finished_recovers_the_written_answer(
            self, rr, lh, monkeypatch):
        # Pointer past the end: _advance_reuse_action moved it there only
        # after the last action's evidence-backed TERMINATED.
        posted, chat, rec = _run(rr, lh, monkeypatch, _Task(5),
                                 lh.ActionState.TERMINATED)
        assert rec.messages == [] and posted is False
        assert chat.messages[-1].get('content') == PROMISE

    def test_every_action_finished_with_nothing_written_asks_for_the_answer(
            self, rr, lh, monkeypatch):
        monkeypatch.setattr(rr, '_reuse_outstanding_tools',
                            lambda *a, **k: [], raising=True)
        monkeypatch.setitem(rr.user_tasks, SESSION, _Task(5))
        chat = _Chat([dict(DISPATCH),
                      {'role': 'user', 'name': 'StatusVerifier',
                       'content': VERDICT}])
        rec = _Recorder()
        rr._reuse_synthesis_turn(SESSION, chat, _Manager(), rec)
        assert rec.messages == [rr._REUSE_SYNTHESIS_STEER]
