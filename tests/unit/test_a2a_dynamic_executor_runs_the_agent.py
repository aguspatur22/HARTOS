"""A peer's A2A message/send to a dynamic agent must RUN the agent, and a
turn that did not run must come back FAILED, never COMPLETED.

THE DEFECT, traced in the review of 3a32d8e4b (the fix that made
/a2a/<id>/jsonrpc answer JSON-RPC instead of an HTML 500):

  * Production registers only dynamic agents, whose executor
    (DynamicAgentExecutor.execute_agent_task) called
    ``chat_agent(message, user_id=..., prompt_id=...)`` and
    ``recipe(user_id=..., message=..., prompt_id=...)``.  Both real
    signatures are ``(user_id, text, prompt_id, file_id, request_id)``, so
    every call raised TypeError.
  * The executor, and the wrapper around it, turned the exception into
    model text, so handle_message_send marked the task COMPLETED.
    peer_reuse.try_peer_recipe_reuse then recorded the error string as a
    successful remote outcome and agent_daemon skipped its local CREATE for
    the goal: a loud failure became a silent false success.

The executor now goes through dispatch.local_chat_dispatch, the ONE
in-process call to this node's own /chat, and any turn that did not run
raises.  These tests drive the real handle_message_send with the real
executor factory; only the discovery store and the /chat boundary are
stubbed.
"""
import asyncio

import pytest

import integrations.agent_engine.dispatch as dispatch
import integrations.google_a2a.dynamic_agent_registry as dar
from integrations.google_a2a.dynamic_agent_registry import (
    DynamicAgentExecutor, TrainedAgent)
from integrations.google_a2a.google_a2a_integration import A2AMessageHandler
from integrations.google_a2a.register_dynamic_agents import (
    create_dynamic_executor_function)

_AGENT = TrainedAgent(
    agent_id='7700000123_0', prompt_id=7700000123, flow_id=0,
    persona='livetest_persona', action='summarise', recipe=[],
    status='completed', can_perform_without_user_input='yes',
    fallback_action='', metadata={'user_id': 'livetest_owner'},
    recipe_file='')


class _Discovery:
    def get_agent_by_id(self, agent_id):
        return _AGENT if agent_id == _AGENT.agent_id else None


@pytest.fixture
def calls(monkeypatch):
    executor = DynamicAgentExecutor.__new__(DynamicAgentExecutor)
    executor.discovery = _Discovery()
    monkeypatch.setattr(dar, '_dynamic_executor', executor)
    class _Seen(list):
        pass
    seen = _Seen()

    def set_reply(status, text):
        def local_chat_dispatch(prompt, user_id, prompt_id, daemon_id=None,
                                **kw):
            seen.append((prompt, user_id, prompt_id, daemon_id))
            return status, text
        monkeypatch.setattr(dispatch, 'local_chat_dispatch',
                            local_chat_dispatch)
    seen.set_reply = set_reply
    return seen


def _send(agent, text='summarise the ferry text'):
    handler = A2AMessageHandler(create_dynamic_executor_function(agent))
    return asyncio.run(handler.handle_message_send({'message': {
        'messageId': 'livetest_m1', 'contextId': 'livetest_c1',
        'parts': [{'kind': 'text', 'text': text}]}}))


def test_a_turn_that_ran_completes_with_its_answer(calls):
    calls.set_reply('ok', 'The ferry carries 83 cyclists.')
    task = _send(_AGENT)
    assert task['state'] == 'completed', task
    assert task['content']['parts'][0]['text'] == 'The ferry carries 83 cyclists.'
    # The one /chat door, called with the agent's owner and prompt_id, and
    # marked as background (a peer is not this node's human).
    assert calls == [('summarise the ferry text', 'livetest_owner',
                      7700000123, 'a2a_livetest_c1')]


def test_a_deferred_turn_fails_the_task(calls):
    calls.set_reply('deferred', None)
    task = _send(_AGENT)
    assert task['state'] == 'failed', task
    assert 'deferred' in task['error']


def test_a_failed_turn_is_not_reported_as_an_answer(calls):
    """The pipeline's own failure sentence is not a result."""
    from core.agent_tools import _SNAG_REPLY
    calls.set_reply('ok', _SNAG_REPLY)
    task = _send(_AGENT)
    assert task['state'] == 'failed', task


def test_an_empty_reply_fails_the_task(calls):
    calls.set_reply('ok', '')
    assert _send(_AGENT)['state'] == 'failed'


def test_an_unknown_agent_fails_the_task(calls):
    calls.set_reply('ok', 'should not be reached')
    ghost = TrainedAgent(**dict(_AGENT.__dict__, agent_id='livetest_ghost_0'))
    task = _send(ghost)
    assert task['state'] == 'failed', task
    assert calls == []
