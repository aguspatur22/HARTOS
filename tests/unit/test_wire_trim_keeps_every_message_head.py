"""A message the wire trim shortens keeps its head: the middle is elided.

Live 2026-09-27, installed build (HARTOS 62649bc1), REUSE probe
liveprobe_reuse_1, llm_outbound.jsonl: the dispatch turn reads
``Perform this action -> Action #1:get_self_build_status\\n\\nWhat is the
current self-build status?\\n follow these steps: [...]`` -- marker, then the
user's words, then the steps.  ``_truncate_msg_content`` cut from the HEAD,
so the first call went out as ``...[truncated head]...\\n: ''}}, {'get_data_
by_key(...`` : the marker and the words were the part removed, the steps the
part kept, and the reply was off-topic.  6 of the 77 calls carried the turn
head-cut this way.

The rule pinned here: whatever the trim shortens -- a user turn, a tool
result, the system message -- keeps its head and its tail, and the
``WIRE_TRIM_MARKER`` sits where the middle was.  Behavioural: the real
``_trim_to_budget`` with only the budget pinned.
"""
import core.llm_outbound_logger as lol
from core.constants import WIRE_TRIM_SAFETY_MARGIN_TOKENS, WIRE_TRIM_MARKER

_MAX_TOKENS = 64
_HEAD = ('Perform this action -> Action #1:get_self_build_status\n\n'
         'What is the current self-build status?\n follow these steps: ')
_STEPS = ''.join("{'get_data_by_key({\"key\":\"os.builds.k%d\"})': "
                 "{'tool_name': 'get_data_by_key', 'code': ''}}, " % i
                 for i in range(90))
_TAIL = "{'save_data_in_memory': 'LAST STEP'}]"


def _trim(msgs, budget, monkeypatch):
    per_slot = budget + _MAX_TOKENS + WIRE_TRIM_SAFETY_MARGIN_TOKENS
    monkeypatch.setattr(lol, '_get_budget_per_slot', lambda: per_slot)
    out, _, n_cut, _, est_after, got = lol._trim_to_budget(
        {'model': 'llama', 'messages': msgs, 'max_tokens': _MAX_TOKENS})
    assert n_cut > 0, 'fixture failed to force a cut'
    return out['messages'], est_after, got


def _assert_head_and_tail(text, head, tail):
    assert text.startswith(head), 'the head was cut: %r' % text[:120]
    assert text.endswith(tail), 'the tail was cut: %r' % text[-120:]
    assert WIRE_TRIM_MARKER in text


def test_the_live_dispatch_turn_keeps_the_marker_and_the_words(monkeypatch):
    sys_m = {'role': 'system', 'content': 'You are the reuse assistant. ' * 40}
    turn = {'role': 'user', 'name': 'User', 'content': _HEAD + _STEPS + _TAIL}
    out, est_after, budget = _trim([sys_m, turn], 900, monkeypatch)
    kept = next(m for m in out if m.get('name') == 'User')
    _assert_head_and_tail(kept['content'], _HEAD, _TAIL)
    assert est_after <= budget


def test_a_cut_tool_result_keeps_its_head(monkeypatch):
    task = {'role': 'user', 'name': 'User', 'content': 'crawl it'}
    call = {'role': 'assistant', 'content': None,
            'tool_calls': [{'id': 'c1', 'type': 'function',
                            'function': {'name': 'crawl', 'arguments': '{}'}}]}
    result = {'role': 'tool', 'tool_call_id': 'c1',
              'content': 'PAGEHEAD ' + 'row ' * 3000 + ' PAGETAIL'}
    out, est_after, budget = _trim(
        [{'role': 'system', 'content': 'sys'}, task, call, result],
        700, monkeypatch)
    kept = next(m for m in out if m.get('role') == 'tool')
    _assert_head_and_tail(kept['content'], 'PAGEHEAD', 'PAGETAIL')
    assert est_after <= budget


def test_a_cut_system_message_keeps_its_head(monkeypatch):
    sys_m = {'role': 'system',
             'content': 'PERSONA HEAD. ' + 'wisdom ' * 4000 + ' RECIPE TAIL.'}
    out, est_after, budget = _trim(
        [sys_m, {'role': 'user', 'content': 'go'}], 700, monkeypatch)
    _assert_head_and_tail(out[0]['content'], 'PERSONA HEAD.', 'RECIPE TAIL.')
    assert est_after <= budget
