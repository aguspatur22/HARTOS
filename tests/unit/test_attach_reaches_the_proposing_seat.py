"""A tool attached for the model to call NOW must reach the seat that speaks.

THE DEFECT, measured live 2026-09-25 on the installed Nunba (HARTOS 20bfca03d),
two agent-to-agent surfaces:

  A2A-3  POST /chat create_agent + autonomous, an action that says "call
         delegate_to_specialist ... if it is not on your list, call
         request_tools first".  The Assistant answered "the
         delegate_to_specialist tool isn't available in my current toolset",
         saved delegation_id null, and the verifier marked it completed.
  A2A-7  the same shape for share_context_with_agents / get_shared_context.

WHY.  ``register_core_tools(..., executor_proposes=True)`` (create_recipe.py
and reuse_recipe.py, main leg) makes the ASSISTANT a proposing seat: it carries
its own LLM schema and emits its own tool_calls.  But every other registration
goes through ``register_dual(helper, assistant, ...)``, which puts the schema on
the Helper ONLY and gives the Assistant execution only.  That covers the
escape itself (``request_tools``) and every on-demand attach path
(``discover_and_attach``, ``attach_for_names``, ``attach_for_tags``).  The
Helper speaks only on an ``@Helper`` mention, which a tool-bearing Assistant
never writes, so for the Assistant a deferral was a permanent exclusion and
request_tools' "Attached and ready to call NOW" was true only for the Helper.

Real autogen agents, not recorders: the claim is about what lands in
``llm_config['tools']``, which is what the model is sent.

    python -m pytest tests/unit/test_attach_reaches_the_proposing_seat.py -q
"""
import ast
import json
import os
import warnings
from types import SimpleNamespace
from typing import Annotated

import pytest

autogen = pytest.importorskip('autogen')

from core.agent_tools import (  # noqa: E402
    attach_for_names,
    attach_for_tags,
    defer_helper_schema,
    discover_and_attach,
    helper_tool_names,
    register_core_tools,
    register_dual,
)

_HARTOS = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# Never contacted: registration builds the client, it does not call it.
_CFG = {'config_list': [{'model': 'x', 'api_key': 'x',
                         'base_url': 'http://127.0.0.1:1/v1'}]}


def _send_message_to_user(text: Annotated[str, 'text']) -> str:
    """Send a message to the user."""
    return 'sent'


def _share_context_with_agents(
        context_key: Annotated[str, 'key'],
        context_value: Annotated[str, 'value']) -> str:
    """Share context information with other agents."""
    return json.dumps({'success': True})


def _get_shared_context(context_key: Annotated[str, 'key']) -> str:
    """Retrieve context information shared by other agents"""
    return json.dumps({'success': True})


def _delegate_to_specialist(task: Annotated[str, 'task']) -> str:
    """Delegate a task to a specialist agent based on required skills"""
    return json.dumps({'success': True, 'delegation_id': 'd1'})


def _main_leg():
    """Helper / Assistant / Executor wired the way create_agents wires them."""
    helper = autogen.AssistantAgent('Helper', llm_config=dict(_CFG))
    assistant = autogen.AssistantAgent('Assistant', llm_config=dict(_CFG))
    executor = autogen.UserProxyAgent('Executor', code_execution_config=False,
                                      human_input_mode='NEVER')
    register_core_tools([('send_message_to_user', 'send',
                          _send_message_to_user)],
                        helper, assistant,
                        executor_proposes=True, second_executor=executor)
    return helper, assistant, executor


def _a2a_registered_at_construction(helper, assistant):
    """create_recipe.py delegate/share/get registration, then the CREATE
    helper deferral that drops them from the Helper's schema too."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        register_dual(helper, assistant, _share_context_with_agents,
                      'share_context_with_agents', 'Share context')
        register_dual(helper, assistant, _get_shared_context,
                      'get_shared_context', 'Retrieve context')
        register_dual(helper, assistant, _delegate_to_specialist,
                      'delegate_to_specialist', 'Delegate to a specialist')
    defer_helper_schema(helper, {'share_context_with_agents',
                                 'get_shared_context',
                                 'delegate_to_specialist'})


class _EmptyRegistry:
    _tools = {}

    def create_endpoint_function(self, tool_name, ep_name):  # pragma: no cover
        return None


class TestOnDemandAttachReachesTheProposer:

    def test_request_tools_reattach_reaches_the_assistant(self):
        """A2A-7 exactly: the tools come back on the seat that will call them."""
        helper, assistant, _ = _main_leg()
        _a2a_registered_at_construction(helper, assistant)
        assert 'share_context_with_agents' not in helper_tool_names(assistant)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = discover_and_attach(
                'share_context_with_agents get_shared_context',
                helper, assistant, _EmptyRegistry(), set(), core_tools=[])

        assert 'Attached and ready' in out
        on_assistant = helper_tool_names(assistant)
        assert {'share_context_with_agents',
                'get_shared_context'} <= on_assistant, (
            'request_tools said "Attached and ready to call NOW" but the '
            'Assistant -- the seat that speaks -- still carries only %r'
            % sorted(on_assistant))
        # and it can run what it now proposes
        assert assistant.can_execute_function('share_context_with_agents')

    def test_named_attach_reaches_the_assistant(self):
        """An action that names delegate_to_specialist gets it on the speaker."""
        helper, assistant, _ = _main_leg()
        attached = set()
        n = attach_for_names(
            ['delegate_to_specialist'], helper, assistant, _EmptyRegistry(),
            attached, core_tools=[('delegate_to_specialist', 'Delegate',
                                   _delegate_to_specialist)])
        assert n == 1
        assert 'delegate_to_specialist' in helper_tool_names(assistant)
        assert 'delegate_to_specialist' in helper_tool_names(helper)

    def test_tag_attach_reaches_the_assistant(self):
        helper, assistant, _ = _main_leg()

        def _synth(text: Annotated[str, 'text']) -> str:
            """Text to speech."""
            return 'ok'

        reg = SimpleNamespace(
            _tools={'pocket_tts': SimpleNamespace(
                tags=['tts'],
                endpoints={'synthesize': {'description': 'Text to speech'}})},
            create_endpoint_function=lambda t, e: _synth)
        n = attach_for_tags({'tts'}, helper, assistant, reg, set())
        assert n == 1
        assert 'pocket_tts_synthesize' in helper_tool_names(assistant)


class TestTheBudgetIsNotWidened:
    """Only ATTACHED tools reach the Assistant; construction stays bounded.

    The Assistant is the seat whose bodies fit (measured 2026-09-12: all 43
    fitting CREATE bodies carried the bounded core set; the 54-tool Helper
    bodies were the ones that could not fit).  Construction-time register_dual
    must keep the rest OFF it, reachable through request_tools.
    """

    def test_construction_registration_leaves_the_assistant_schema_alone(self):
        helper, assistant, _ = _main_leg()
        before = helper_tool_names(assistant)
        _a2a_registered_at_construction(helper, assistant)
        assert helper_tool_names(assistant) == before

    def test_non_proposing_executor_gets_no_schema_and_does_not_raise(self):
        """The time / visual legs' executors carry no schema; attach must not
        turn them into proposers, and a UserProxy (llm_config False) must not
        make the attach raise."""
        helper = autogen.AssistantAgent('Helper', llm_config=dict(_CFG))
        for executor in (
                autogen.UserProxyAgent('Exec', code_execution_config=False,
                                       human_input_mode='NEVER'),
                autogen.AssistantAgent('TimeAgent', llm_config=dict(_CFG))):
            attach_for_names(
                ['delegate_to_specialist'], helper, executor,
                _EmptyRegistry(), set(),
                core_tools=[('delegate_to_specialist', 'Delegate',
                             _delegate_to_specialist)])
            assert helper_tool_names(executor) == set()
            assert executor.can_execute_function('delegate_to_specialist')


class TestTheEscapeIsOnTheProposer:

    def test_request_tools_is_on_the_assistant_and_rearms_onto_it(self):
        """End to end as the model drives it: the Assistant can SEE
        request_tools, calling it attaches onto the Assistant."""
        from core.agent_tools import register_request_tools
        helper, assistant, _ = _main_leg()
        _a2a_registered_at_construction(helper, assistant)
        assistant._hart_core_tools = []

        register_request_tools(helper, assistant, _EmptyRegistry(), set())

        assert 'request_tools' in helper_tool_names(assistant), (
            'the never-say-unavailable escape is not on the Assistant, so the '
            'seat that speaks cannot ask for a deferred tool')
        assert 'request_tools' in helper_tool_names(helper)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = assistant._function_map['request_tools'](
                'delegate a task to a specialist')
        assert 'delegate_to_specialist' in out
        assert 'delegate_to_specialist' in helper_tool_names(assistant)


def _calls_and_defs(rel):
    with open(os.path.join(_HARTOS, rel), encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    calls = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    defs = [n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return calls, defs


@pytest.mark.parametrize('rel', [os.path.join('hartos', 'create_recipe.py'),
                                 os.path.join('hartos', 'reuse_recipe.py')])
def test_source_guard_one_request_tools(rel):
    """Collapse guard: both legs register the escape through the ONE helper.

    The two inline copies are how the escape stayed Helper-only on both legs;
    a second inline ``def request_tools`` would reintroduce a copy the
    behavioural tests above cannot see.
    """
    calls, defs = _calls_and_defs(rel)
    assert 'request_tools' not in defs, (
        '%s defines request_tools inline again; register it with '
        'core.agent_tools.register_request_tools' % rel)
    assert calls.count('register_request_tools') == 1, rel
