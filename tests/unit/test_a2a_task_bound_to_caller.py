"""An A2A task answers only the caller that started it, and the task table
stays bounded.

Review finding M5 (2026-09-26): A2AMessageHandler.tasks held every task
forever (one entry per message/send, never removed), and message/get /
task/cancel served any admitted caller that named a task id: one verified
peer could read another's result or cancel its turn.  Each task is now
bound to the identity that was admitted for its message/send (the peer's
node_id, or the /chat gate's user / key / address) and get and cancel from
anyone else answer "not found" (existence is not disclosed).  Finished
tasks are evicted after HEVOLVE_A2A_TASK_TTL_S, and past
HEVOLVE_A2A_TASK_MAX the oldest finished go first; a running task is never
evicted.

The route tests drive the REAL Flask app, gate and jsonrpc view with two
real node keys (tests/unit/test_a2a_admitted_peer_runs_shared_agent.py's
harness); the handler tests call the real A2AMessageHandler.
"""
import asyncio
import threading
import time
import uuid

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import integrations.google_a2a.google_a2a_integration as gai
import security.node_integrity as ni
from integrations.google_a2a.google_a2a_integration import (
    A2AMessageHandler, TaskState)
from integrations.social.sync_engine import SyncEngine
from tests.unit.test_a2a_admitted_peer_runs_shared_agent import (  # noqa: F401
    AGENT, _admit, _env, _post, _signed, invoker, node)


def _as_second_node(monkeypatch):
    """Sign as a DIFFERENT verified node from here on."""
    priv = Ed25519PrivateKey.generate()
    monkeypatch.setattr(ni, '_private_key', priv)
    monkeypatch.setattr(ni, '_public_key', priv.public_key())
    other = f'livetest-b-{uuid.uuid4().hex[:8]}'
    from tests.unit import test_a2a_admitted_peer_runs_shared_agent as h
    monkeypatch.setattr(SyncEngine, 'canonical_node_id', staticmethod(
        lambda: {'invoker': other, 'server': h.SERVER_ID}[h._ROLE['who']]))
    _admit(other, ni.get_public_key_hex())
    return other


def _send(node, **params):
    code, body = _post(node, _signed(params={
        'message': {'messageId': uuid.uuid4().hex,
                    'parts': [{'kind': 'text', 'text': 'x'}]}, **params}))
    assert code == 200, body
    return body['result']['id']


def test_another_peer_cannot_read_or_cancel_the_task(node, invoker,
                                                     monkeypatch):
    _admit(invoker.node_id, invoker.public_key)
    task_id = _send(node)
    code, mine = _post(node, _signed(method='message/get',
                                     params={'taskId': task_id}))
    assert mine['result']['state'] == 'completed', mine
    _as_second_node(monkeypatch)
    code, theirs = _post(node, _signed(method='message/get',
                                       params={'taskId': task_id}))
    assert 'not found' in theirs['result']['error']['message'], theirs
    assert 'content' not in theirs['result']
    code, cancel = _post(node, _signed(method='task/cancel',
                                       params={'taskId': task_id}))
    assert 'not found' in cancel['result']['error']['message'], cancel


# ── the handler itself ──────────────────────────────────────────────────

async def _ok(text, ctx):
    return {'role': 'model', 'parts': [{'text': 'ok'}]}


def _run(coro):
    return asyncio.run(coro)


def _send_as(h, caller, blocking=True):
    params = {'message': {'messageId': uuid.uuid4().hex,
                          'parts': [{'kind': 'text', 'text': 'x'}]}}
    if not blocking:
        params['configuration'] = {'blocking': False}
    return _run(h.handle_message_send(params, caller=caller))['id']


def test_a_user_caller_is_bound_too():
    h = A2AMessageHandler(_ok)
    tid = _send_as(h, 'user:alice')
    assert _run(h.handle_message_get({'taskId': tid}, caller='user:alice')
                )['state'] == 'completed'
    assert 'error' in _run(h.handle_message_get({'taskId': tid},
                                                caller='user:bob'))
    assert 'error' in _run(h.handle_task_cancel({'taskId': tid},
                                                caller='user:bob'))


def test_finished_tasks_expire(monkeypatch):
    monkeypatch.setattr(gai, '_TASK_TTL_S', 0.2)
    h = A2AMessageHandler(_ok)
    old = _send_as(h, 'user:a')
    time.sleep(0.3)
    _send_as(h, 'user:a')
    assert old not in h.tasks


def test_the_table_is_capped_and_a_running_task_survives(monkeypatch):
    monkeypatch.setattr(gai, '_TASK_MAX', 5)
    release = threading.Event()
    started = threading.Event()

    async def slow(text, ctx):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.02)
        return {'role': 'model', 'parts': [{'text': 'late'}]}
    h = A2AMessageHandler(slow)
    running = _send_as(h, 'user:a', blocking=False)
    assert started.wait(5)
    h.agent_executor = _ok
    for _ in range(20):
        _send_as(h, 'user:a')
    assert len(h.tasks) <= 5, len(h.tasks)
    assert running in h.tasks
    assert h.tasks[running].state in (TaskState.SUBMITTED, TaskState.WORKING)
    release.set()


def test_another_callers_message_id_does_not_replace_the_task():
    """The messageId is chosen by the caller; reusing someone else's must
    not overwrite (and so read or hijack) their task."""
    h = A2AMessageHandler(_ok)
    mid = uuid.uuid4().hex
    params = {'message': {'messageId': mid,
                          'parts': [{'kind': 'text', 'text': 'x'}]}}
    _run(h.handle_message_send(params, caller='user:alice'))
    theirs = _run(h.handle_message_send(params, caller='user:mallory'))
    assert theirs['id'] != mid
    assert h.tasks[mid].owner == 'user:alice'
