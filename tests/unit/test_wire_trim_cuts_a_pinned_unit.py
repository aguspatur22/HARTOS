"""What the drop keeps for a protected tool result can still be cut.

be96f2510 protected the newest tool result, and 05806ae09 keeps a tool call
and its results together.  Together they pinned the whole unit -- the
assistant message carrying the tool_calls and every sibling result -- while
the cut pass shrank only the protected messages and the newest one.  The
pinned mass could be neither dropped nor cut.  Review of be96f2510
(hartos-5e scratchpad be_probe.py), real _trim_to_budget at per_slot 12288,
budget 7424:

  * [sys, task, call with 40k-char arguments, result, verdict]:
    the parent commit sends 721 tokens; be96f2510 sends 20,243 (over).
  * [sys, task, call, 3 parallel ~16k-char results, verdict]:
    the parent sends 721; be96f2510 sends 8,307 (over).

Both are llama-server context overflows.  The rule pinned here: every
message the drop keeps only because its unit holds a protected message is a
cut candidate too, cut before the protected ones; a call's arguments are cut
inside a strict JSON object (llama.cpp 500s on arguments that are not JSON);
the unit stays paired.
"""
import json

import core.llm_outbound_logger as lol
from core.constants import WIRE_TRIM_MARKER
from hartos.helper import is_wire_json

_PER_SLOT = 12288


def _trim(msgs, monkeypatch, max_tokens=2048):
    monkeypatch.setattr(lol, '_get_budget_per_slot', lambda: _PER_SLOT)
    out, _, _, _, est_after, budget = lol._trim_to_budget(
        {'model': 'llama', 'messages': msgs, 'max_tokens': max_tokens})
    return out['messages'], est_after, budget


def _paired(msgs):
    announced = set()
    for m in msgs:
        for tc in (m.get('tool_calls') or []):
            announced.add(tc['id'])
        if m.get('role') == 'tool' and m.get('tool_call_id') not in announced:
            return False
    return True


def test_a_call_with_huge_arguments_is_cut_to_fit(monkeypatch):
    sys_m = {'role': 'system', 'content': 'sys ' * 500}
    task = {'role': 'user', 'name': 'User', 'content': 'task ' * 200}
    call = {'role': 'assistant', 'content': None,
            'tool_calls': [{'id': 'c1', 'type': 'function',
                            'function': {'name': 'write_file',
                                         'arguments': json.dumps(
                                             {'body': 'x ' * 20000})}}]}
    result = {'role': 'tool', 'tool_call_id': 'c1', 'content': 'row ' * 500}
    verdict = {'role': 'user', 'name': 'StatusVerifier',
               'content': '{"status":"pending"}'}
    out, est_after, budget = _trim([sys_m, task, call, result, verdict],
                                   monkeypatch)
    assert est_after <= budget, (est_after, budget)
    assert _paired(out)
    kept_call = next(m for m in out if m.get('tool_calls'))
    args = kept_call['tool_calls'][0]['function']['arguments']
    assert is_wire_json(args) and isinstance(json.loads(args), dict), args[:200]
    kept_text = json.loads(args)['trimmed_arguments']
    assert WIRE_TRIM_MARKER in kept_text
    assert kept_text.startswith('{"body": "x x')
    assert kept_call['tool_calls'][0]['function']['name'] == 'write_file'


def test_parallel_results_are_cut_to_fit(monkeypatch):
    sys_m = {'role': 'system', 'content': 'sys ' * 500}
    task = {'role': 'user', 'name': 'User', 'content': 'task ' * 200}
    call = {'role': 'assistant', 'content': None,
            'tool_calls': [{'id': 'c%d' % i, 'type': 'function',
                            'function': {'name': 'crawl', 'arguments': '{}'}}
                           for i in range(3)]}
    results = [{'role': 'tool', 'tool_call_id': 'c%d' % i,
                'content': ('R%d ' % i) + 'row ' * 4000} for i in range(3)]
    verdict = {'role': 'user', 'name': 'StatusVerifier',
               'content': '{"status":"pending"}'}
    out, est_after, budget = _trim([sys_m, task, call] + results + [verdict],
                                   monkeypatch)
    assert est_after <= budget, (est_after, budget)
    assert _paired(out)
    kept = [m for m in out if m.get('role') == 'tool']
    assert [m['tool_call_id'] for m in kept] == ['c0', 'c1', 'c2']
    for m in kept:
        assert m['content'].startswith('R'), m['content'][:40]
    # The newest (protected) result keeps at least as much as its siblings.
    assert len(kept[-1]['content']) >= max(len(m['content']) for m in kept[:-1])
