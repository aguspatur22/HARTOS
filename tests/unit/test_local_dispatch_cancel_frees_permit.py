"""A cancelled remote turn gives the LLM permit back before the turn starts.

Measured before choosing (review finding M2, 2026-09-26): once
dispatch.local_chat_dispatch has called the in-process /chat, nothing can
stop THAT turn alone.  The only mid-flight abort is core.foreground's cancel
registry, which closes the shared background LLM client and so drops every
background call on the node, not one turn.  What a cancel CAN do without
touching anyone else is act while the turn waits for the one permit (up to
HEVOLVE_LOCAL_LLM_WAIT_S, 30 s) and at the gate right before the turn runs.
So: a cancel_event is honoured while waiting and after the acquire, and the
permit is released; a turn already running finishes.

Driven through the REAL local_chat_dispatch and the REAL semaphore; the
in-process /chat callable and the user-activity gate are the boundary.
"""
import threading
import time

import pytest

from integrations.agent_engine import dispatch


@pytest.fixture
def chat(monkeypatch):
    ran = []

    def fake_chat(**kw):
        ran.append(kw['text'])
        return {'text': 'done'}
    monkeypatch.setattr(dispatch, '_in_process_chat', lambda *a, **k: fake_chat)
    monkeypatch.setattr(dispatch, 'is_user_recently_active', lambda: False)
    monkeypatch.setattr(dispatch, 'local_dispatch_provider_breaker_open',
                        lambda *a, **k: '')
    return ran


def _permit_free():
    ok = dispatch._local_llm_semaphore.acquire(timeout=0.2)
    if ok:
        dispatch._local_llm_semaphore.release()
    return ok


def test_a_cancel_while_waiting_for_the_permit_returns_at_once(chat):
    cancel = threading.Event()
    assert dispatch._local_llm_semaphore.acquire(timeout=1)
    out = {}
    try:
        t = threading.Thread(target=lambda: out.update(r=dispatch.local_chat_dispatch(
            'p', 'u', 'pid', daemon_id='a2a_x', cancel_event=cancel)))
        t.start()
        time.sleep(0.3)
        started = time.monotonic()
        cancel.set()
        t.join(timeout=5)
        waited = time.monotonic() - started
    finally:
        dispatch._local_llm_semaphore.release()
    assert out['r'] == ('cancelled', None)
    assert waited < 2, waited
    assert chat == []
    assert _permit_free()


def test_a_cancel_before_the_turn_releases_the_permit_it_took(chat):
    cancel = threading.Event()
    cancel.set()
    assert dispatch.local_chat_dispatch('p', 'u', 'pid', daemon_id='a2a_x',
                                        cancel_event=cancel) == ('cancelled', None)
    assert chat == []
    assert _permit_free()


def test_no_cancel_event_runs_the_turn_as_before(chat):
    assert dispatch.local_chat_dispatch('p', 'u', 'pid',
                                        daemon_id='a2a_x') == ('ok', 'done')
    assert chat == ['p']
    assert _permit_free()


def test_a_busy_permit_still_defers_after_the_wait(chat, monkeypatch):
    monkeypatch.setattr(dispatch, '_LOCAL_LLM_WAIT_S', 0.3)
    assert dispatch._local_llm_semaphore.acquire(timeout=1)
    try:
        r = dispatch.local_chat_dispatch('p', 'u', 'pid', daemon_id='a2a_x',
                                         cancel_event=threading.Event())
    finally:
        dispatch._local_llm_semaphore.release()
    assert r == ('deferred', None)


def test_the_dynamic_executor_hands_the_cancel_to_the_dispatch(monkeypatch):
    """The A2A executor for a trained agent passes the task's cancel_event
    through, and a cancelled dispatch fails the task, never completes it."""
    import asyncio
    import types
    from integrations.google_a2a import dynamic_agent_registry as dar
    seen = {}

    def spy(message, user_id, prompt_id, daemon_id=None, cancel_event=None):
        seen['cancel_event'] = cancel_event
        return 'cancelled', None
    monkeypatch.setattr(dispatch, 'local_chat_dispatch', spy)
    ex = dar.DynamicAgentExecutor.__new__(dar.DynamicAgentExecutor)
    agent = types.SimpleNamespace(persona='p', prompt_id='42', metadata={})
    ex.discovery = types.SimpleNamespace(get_agent_by_id=lambda a: agent)
    ev = threading.Event()
    with pytest.raises(RuntimeError, match='cancel'):
        asyncio.run(ex.execute_agent_task('42_0', 'hi', 'ctx',
                                          cancel_event=ev))
    assert seen['cancel_event'] is ev


def test_a_cancel_that_lands_as_the_permit_is_taken_gives_it_back(
        chat, monkeypatch):
    """The window between the acquire and the turn: the permit was taken,
    the cancel arrived, the turn must not start and the permit comes back."""
    real = threading.Semaphore(1)
    cancel = threading.Event()

    class _CancelOnAcquire:
        def acquire(self, timeout=None):
            got = real.acquire(timeout=timeout)
            if got:
                cancel.set()
            return got

        def release(self):
            real.release()
    monkeypatch.setattr(dispatch, '_local_llm_semaphore', _CancelOnAcquire())
    assert dispatch.local_chat_dispatch('p', 'u', 'pid', daemon_id='a2a_x',
                                        cancel_event=cancel) == ('cancelled', None)
    assert chat == []
    assert real.acquire(timeout=0.2), 'the permit was not given back'
