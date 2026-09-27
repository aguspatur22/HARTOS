"""Whatever the wire trim elides is saved, and the wire carries a pointer to it.

Owner direction (relayed 2026-09-27): "tool results shd be saved with
pointers and whatever is trimmed needs pointers to memory", and "the pointer
design shd be explicitly understood by the LLM ... and a pointer shd not
influence the context".  The rules pinned here, on the real
``_trim_to_budget`` and the real ``get_data_by_key`` tool:

  * a message the trim shortens carries ``[elided:<id> <n> chars of <kind>]``
    where its middle was; a dropped tool result is listed the same way;
  * the original is saved whole in the agent-data store (namespace
    ``elided``, the store behind get_data_by_key), and
    ``get_data_by_key(key="elided:<id>")`` returns it exactly, a page at a
    time -- also after the process forgets everything (a REUSE replay);
  * the system message the model reads says what a pointer is and how to
    expand it, only when the body carries one;
  * a pointer is inert: the caller's messages -- the history the banker,
    the completion gate and the verifier judge -- are never changed, and a
    pointer fits inside the trim's 64-token floor.
"""
import json
import re

import pytest

import core.llm_outbound_logger as lol
from core.constants import WIRE_TRIM_SAFETY_MARGIN_TOKENS
from core.token_utils import count_tokens_for_text

_MAX_TOKENS = 64
_POINTER = re.compile(r'\[elided:([0-9a-f]{12}) (\d+) chars of ([a-z ]+)\]')


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The agent-data store in a temp dir, as cache_loaders resolves it."""
    import core.cache_loaders as cl
    monkeypatch.setattr(cl, 'AGENT_DATA_DIR', str(tmp_path))
    return tmp_path


def _trim(msgs, budget, monkeypatch):
    per_slot = budget + _MAX_TOKENS + WIRE_TRIM_SAFETY_MARGIN_TOKENS
    monkeypatch.setattr(lol, '_get_budget_per_slot', lambda: per_slot)
    out, _, _, _, est_after, got = lol._trim_to_budget(
        {'model': 'llama', 'messages': msgs, 'max_tokens': _MAX_TOKENS})
    return out['messages'], est_after, got


def _shape():
    sys_m = {'role': 'system', 'content': 'You are the reuse assistant.'}
    task = {'role': 'user', 'name': 'User', 'content': 'Summarise the page.'}
    call = {'role': 'assistant', 'content': None,
            'tool_calls': [{'id': 'c1', 'type': 'function',
                            'function': {'name': 'crawl', 'arguments': '{}'}}]}
    page = 'PAGEHEAD ' + ' '.join('row%d' % i for i in range(3000)) + ' PAGETAIL'
    result = {'role': 'tool', 'tool_call_id': 'c1', 'content': page}
    return sys_m, task, call, result, page


def _get_data_by_key(prompt_id='p1'):
    """The real tool closure, built the way the pipelines build it."""
    from unittest import mock
    from core.agent_tools import build_core_tool_closures
    ctx = {'user_id': 'u1', 'prompt_id': prompt_id, 'agent_data': {prompt_id: {}},
           'helper_fun': mock.MagicMock(), 'user_prompt': 'u1_p1',
           'request_id_list': {'u1_p1': 'r1'}, 'recent_file_id': {},
           'scheduler': mock.MagicMock(), 'send_message_to_user1': mock.MagicMock(),
           'retrieve_json': json.loads, 'strip_json_values': lambda x: x,
           'save_conversation_db': mock.MagicMock()}
    tools = {name: fn for name, _, fn in build_core_tool_closures(ctx)}
    return tools['get_data_by_key']


def _read_all(tool, key):
    """Follow the page notes to the end, the way a model would."""
    text, offset = '', 0
    for _ in range(1000):
        page = tool(key=key, offset=offset)
        m = re.search(r'\n\.\.\.\[chars (\d+)-(\d+) of (\d+); call get_data_by_key '
                      r'with offset=(\d+) for the rest\]$', page)
        if not m:
            return text + page
        text += page[:m.start()]
        offset = int(m.group(4))
    raise AssertionError('pages never ended')


def test_a_cut_tool_result_carries_a_pointer_to_its_original(store, monkeypatch):
    sys_m, task, call, result, page = _shape()
    out, est_after, budget = _trim([sys_m, task, call, result], 900, monkeypatch)
    kept = next(m for m in out if m.get('role') == 'tool')
    m = _POINTER.search(kept['content'])
    assert m, kept['content'][:300]
    assert m.group(3) == 'a tool result'
    assert int(m.group(2)) == len(page)
    assert est_after <= budget
    # The pointer fits inside the 64-token floor with room to spare.
    assert count_tokens_for_text(m.group(0), 'llama') <= 32


def test_the_real_tool_returns_the_exact_original_paged(store, monkeypatch):
    sys_m, task, call, result, page = _shape()
    out, _, _ = _trim([sys_m, task, call, result], 900, monkeypatch)
    pid = _POINTER.search(next(m for m in out if m.get('role') == 'tool')
                          ['content']).group(1)
    tool = _get_data_by_key()
    first = tool(key='elided:' + pid)
    assert first.startswith('PAGEHEAD') and 'offset=' in first, first[-200:]
    assert _read_all(tool, 'elided:' + pid) == page


def test_a_pointer_survives_the_process_forgetting(store, monkeypatch):
    """REUSE replays in a later turn, often a later process: the original is
    read back from the store on disk, not from anything held in memory."""
    sys_m, task, call, result, page = _shape()
    out, _, _ = _trim([sys_m, task, call, result], 900, monkeypatch)
    pid = _POINTER.search(next(m for m in out if m.get('role') == 'tool')
                          ['content']).group(1)
    assert list(store.glob('elided_agent_data.json'))
    import importlib
    importlib.reload(lol)
    assert lol.read_elided(pid) == page


def test_the_system_message_explains_a_pointer_only_when_one_is_sent(
        store, monkeypatch):
    sys_m, task, call, result, page = _shape()
    out, _, _ = _trim([sys_m, task, call, result], 900, monkeypatch)
    system = out[0]['content']
    assert system.startswith(sys_m['content'])
    assert 'get_data_by_key' in system and 'elided:' in system
    assert 'not the content' in system
    small = [dict(sys_m), {'role': 'user', 'content': 'hi'}]
    out2, _, _ = _trim(small, 900, monkeypatch)
    assert out2[0]['content'] == sys_m['content']


def test_a_dropped_tool_pair_is_listed_with_a_pointer(store, monkeypatch):
    sys_m, task, call, result, page = _shape()
    old_call = {'role': 'assistant', 'content': 'first try',
                'tool_calls': [{'id': 'c0', 'type': 'function',
                                'function': {'name': 'crawl', 'arguments': '{}'}}]}
    old_result = {'role': 'tool', 'tool_call_id': 'c0',
                  'content': 'OLD ' + 'old ' * 2000}
    newer = {'role': 'tool', 'tool_call_id': 'c1', 'content': 'the new page'}
    msgs = [sys_m, old_call, old_result, task, call, newer]
    out, est_after, budget = _trim(msgs, 900, monkeypatch)
    assert old_result not in out
    pointers = _POINTER.findall(out[0]['content'])
    assert any(kind == 'a tool result' and int(n) == len(old_result['content'])
               for _, n, kind in pointers), out[0]['content'][-400:]
    pid = next(p for p, n, k in pointers if int(n) == len(old_result['content']))
    assert lol.read_elided(pid) == old_result['content']
    assert est_after <= budget


def test_the_callers_history_is_never_changed(store, monkeypatch):
    """Inert: the banker, the gate and the verifier's evidence all read the
    conversation the caller holds, which keeps the full result."""
    import copy
    sys_m, task, call, result, page = _shape()
    msgs = [sys_m, task, call, result]
    before = copy.deepcopy(msgs)
    _trim(msgs, 900, monkeypatch)
    assert msgs == before
    assert not any(_POINTER.search(str(m.get('content'))) for m in msgs)


def test_an_unstorable_original_still_cuts_with_the_plain_marker(
        store, monkeypatch):
    """A store that cannot be written must never fail the LLM call: the cut
    falls back to the marker with no pointer."""
    from core.constants import WIRE_TRIM_MARKER
    monkeypatch.setattr(lol, '_save_elided', lambda records: False)
    sys_m, task, call, result, page = _shape()
    out, est_after, budget = _trim([sys_m, task, call, result], 900, monkeypatch)
    kept = next(m for m in out if m.get('role') == 'tool')
    assert WIRE_TRIM_MARKER in kept['content']
    assert not _POINTER.search(kept['content'])
    assert 'get_data_by_key' not in out[0]['content']
    assert est_after <= budget


def test_the_user_seed_warns_and_is_counted(caplog):
    before = lol.user_seed_count()
    msgs = [{'role': 'system', 'content': 's'},
            {'role': 'assistant', 'content': 'a'}]
    with caplog.at_level('WARNING', logger=lol.logger.name):
        assert lol.ensure_user_turn(msgs)
    assert lol.user_seed_count() == before + 1
    assert any('seed' in r.getMessage().lower() for r in caplog.records
               if r.levelname == 'WARNING')
