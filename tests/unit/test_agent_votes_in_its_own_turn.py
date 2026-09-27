"""A /chat agent can vote on a thought experiment during its own turn.

Review of 924b8e9dc: cast_experiment_vote had become uncallable.  Its only
caller was the MCP bridge, which (rightly) runs with no caller, and no /chat
agent carried the tool, so no agent could vote at all.

The tool now goes through the ONE existing registration path for in-process
tools: a native ServiceToolInfo registered on service_tool_registry (the
GhPrTool / SeoAuditTool shape), attached to create_recipe's and
reuse_recipe's agents by the Tier-1 gate (core.agent_tools.
filter_service_tools) when the turn is about a thought experiment
(marketing_tools.detect_goal_tags -> 'thought_experiment', which
goal_manager already maps to the 'thought_experiment' tool tag).  It runs
on the turn's thread, where /chat has put the agent's prompt_id
(hart_intelligence_entry sets thread_local_data before the agents run), so
the vote is the calling agent's.

End to end here: the real registry shim, the real gate, real autogen
agents registered by the real register_dual, the call executed by autogen's
own execute_function, a real SQLite database.  The only stand-in is the
LLM: the tool call is handed to the executor as the model would send it.
"""
import json
import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault('HEVOLVE_DB_PATH', ':memory:')
os.environ.setdefault('SOCIAL_DB_PATH', ':memory:')

from hartos.threadlocal import thread_local_data  # noqa: E402
from integrations.social import models as social_models  # noqa: E402
from integrations.social.models import (  # noqa: E402
    Base, ExperimentVote, ThoughtExperiment, User)
from integrations.social.thought_experiment_service import (  # noqa: E402
    ThoughtExperimentService)

TURN = 'Vote on the thought experiment about cache warmup'

# autogen registers a tool schema only on an agent with an llm_config; no
# model is ever called here.
_NO_CALL_LLM = {'config_list': [{'model': 'none', 'api_key': 'none',
                                 'base_url': 'http://127.0.0.1:9'}]}


@pytest.fixture
def db(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{tmp_path / 'turnvote.db'}")
    Base.metadata.create_all(eng)
    f = sessionmaker(bind=eng, expire_on_commit=False)
    monkeypatch.setattr(social_models, '_SessionLocal', f)
    s = f()
    yield s
    s.close()
    eng.dispose()


@pytest.fixture
def registry(monkeypatch):
    """A fresh registry, so the test sees exactly what registration adds."""
    from integrations.service_tools import registry as reg_mod
    fresh = reg_mod.ServiceToolRegistry(config_file='__none__.json')
    monkeypatch.setattr(reg_mod, 'service_tool_registry', fresh)
    import integrations.agent_engine.thought_experiment_tools as tet
    monkeypatch.setattr(tet, 'service_tool_registry', fresh, raising=False)
    return fresh


@pytest.fixture
def turn():
    """The /chat turn's thread-local state, restored afterwards."""
    saved = thread_local_data.snapshot()
    yield thread_local_data
    for key in list(vars(thread_local_data._local)):
        delattr(thread_local_data._local, key)
    thread_local_data.adopt(saved)


def _user(db, user_type='human', owner_id=None, agent_id=None):
    u = User(username=f'livetest_tv_{uuid.uuid4().hex[:8]}',
             user_type=user_type, owner_id=owner_id, agent_id=agent_id)
    db.add(u)
    db.commit()
    return u


def _experiment(db):
    e = ThoughtExperiment(
        id=str(uuid.uuid4()), creator_id=_user(db).id, title='Cache warmup',
        hypothesis='Faster cache warmup lowers latency', expected_outcome='o',
        status='voting')
    db.add(e)
    db.commit()
    return e


def _gated_tools(registry, text):
    """What the Tier-1 gate attaches for a turn about ``text``."""
    from core.agent_tools import filter_service_tools
    from integrations.agent_engine.marketing_tools import resolve_goal_tags
    tags = resolve_goal_tags(None, text)
    return filter_service_tools(tags, registry.get_all_tool_functions(),
                                registry.get_tool_definitions(), registry)


def test_the_vote_tool_is_registered_on_the_one_path(registry):
    from integrations.agent_engine.thought_experiment_tools import (
        ExperimentVoteTool)
    assert ExperimentVoteTool.register() is True
    assert 'cast_experiment_vote' in registry.get_all_tool_functions()


def test_a_thought_experiment_turn_unlocks_it_and_others_do_not(registry):
    from integrations.agent_engine.thought_experiment_tools import (
        ExperimentVoteTool)
    ExperimentVoteTool.register()
    assert 'cast_experiment_vote' in _gated_tools(registry, TURN)
    assert 'cast_experiment_vote' not in _gated_tools(
        registry, 'What is the weather in Chennai today?')


def test_an_agent_votes_in_its_own_turn_end_to_end(db, registry, turn):
    autogen = pytest.importorskip('autogen')
    from core.agent_tools import register_dual
    from integrations.agent_engine.thought_experiment_tools import (
        ExperimentVoteTool)

    e = _experiment(db)
    owner = _user(db)
    agent = _user(db, 'agent', owner_id=owner.id, agent_id='55501')

    ExperimentVoteTool.register()
    tools = _gated_tools(registry, TURN)
    helper = autogen.ConversableAgent('helper', llm_config=_NO_CALL_LLM)
    executor = autogen.ConversableAgent('executor', llm_config=False,
                                        human_input_mode='NEVER')
    for name, fn in tools.items():
        register_dual(helper, executor, fn, name, fn.__doc__ or name)

    # What /chat sets before the agents run (hart_intelligence_entry).
    turn.set_user_id(owner.id)
    turn.set_prompt_id('55501')
    ok, result = executor.execute_function({
        'name': 'cast_experiment_vote',
        'arguments': json.dumps({'experiment_id': e.id, 'vote_value': 2,
                                 'reasoning': 'warm caches help'}),
    })
    assert ok, result
    body = json.loads(result['content'])
    assert body['success'] is True, body

    db.expire_all()
    rows = [(v.voter_id, v.voter_type, v.vote_value)
            for v in db.query(ExperimentVote).filter_by(experiment_id=e.id)]
    assert rows == [(agent.id, 'agent', 2)]
    tally = ThoughtExperimentService.tally_votes(db, e.id)
    assert tally['distinct_voters'] == 1          # counted as its owner


def test_in_a_turn_the_model_cannot_name_someone_else(db, registry, turn):
    autogen = pytest.importorskip('autogen')
    from core.agent_tools import register_dual
    from integrations.agent_engine.thought_experiment_tools import (
        ExperimentVoteTool)

    e = _experiment(db)
    _user(db, 'agent', owner_id=_user(db).id, agent_id='55502')
    person = _user(db)
    ExperimentVoteTool.register()
    fn = _gated_tools(registry, TURN)['cast_experiment_vote']
    helper = autogen.ConversableAgent('helper', llm_config=_NO_CALL_LLM)
    executor = autogen.ConversableAgent('executor', llm_config=False,
                                        human_input_mode='NEVER')
    register_dual(helper, executor, fn, 'cast_experiment_vote', 'vote')

    turn.set_prompt_id('55502')
    ok, result = executor.execute_function({
        'name': 'cast_experiment_vote',
        'arguments': json.dumps({'experiment_id': e.id, 'vote_value': 2,
                                 'voter_id': person.id}),
    })
    assert json.loads(result['content'])['success'] is False
    db.expire_all()
    assert db.query(ExperimentVote).filter_by(experiment_id=e.id).count() == 0


# ── Review of d99b1aa88: reach the tool from how people ask ───────────
# It was attached only for the literal phrase "thought experiment", and in
# CREATE only at agent build time.  Now a turn that pairs a vote word
# (vote / voting / ballot) with an experiment word (experiment / proposal)
# or an experiment id unlocks it, in the per-turn attach both CREATE and
# REUSE run (core.agent_tool_menu.attach_for_turn).

_EXP_ID = '3f2a9c1e-7b4d-4e2a-9c1e-7b4d4e2a9c1e'


@pytest.mark.parametrize('turn_text', [
    'cast your vote on experiment abc',
    'please vote on the experiment about latency',
    'Voting on proposal 12 closes tonight, add yours',
    'ballot for the thought experiment',
    f'vote 2 on {_EXP_ID}',
    'Vote on the thought experiment about cache warmup',
])
def test_a_vote_on_an_experiment_unlocks_the_tool(turn_text):
    from integrations.agent_engine.marketing_tools import detect_goal_tags
    assert 'thought_experiment' in detect_goal_tags(turn_text)


@pytest.mark.parametrize('turn_text', [
    'vote for the best pizza place tonight',
    'run an experiment on the cache and tell me the latency',
    f'what is the status of {_EXP_ID}?',
    'What is the weather in Chennai today?',
    'the devotee was experimenting',   # no word starts with vote
])
def test_other_turns_do_not(turn_text):
    from integrations.agent_engine.marketing_tools import detect_goal_tags
    assert 'thought_experiment' not in detect_goal_tags(turn_text)


def _agents_built_for(registry, goal_text):
    """Agents as create/reuse build them: Tier-1 gate on the build-time goal,
    register_dual, and the per-conversation ledger the turn attach reads."""
    autogen = pytest.importorskip('autogen')
    from core.agent_tools import register_dual
    from integrations.agent_engine.marketing_tools import resolve_goal_tags
    helper = autogen.ConversableAgent('helper', llm_config=_NO_CALL_LLM)
    executor = autogen.ConversableAgent('executor', llm_config=False,
                                        human_input_mode='NEVER')
    tools = _gated_tools(registry, goal_text)
    for name, fn in tools.items():
        register_dual(helper, executor, fn, name, fn.__doc__ or name)
    executor._hart_attached_tools = set(tools)
    executor._hart_unlocked_tags = set(resolve_goal_tags(None, goal_text))
    return helper, executor


def _tool_reply(executor, helper, experiment_id, value=2):
    """The executor answering a model's tool call through autogen's own
    reply machinery (generate_reply -> tool-call reply -> the function)."""
    return executor.generate_reply(messages=[{
        'role': 'assistant', 'content': None,
        'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {
            'name': 'cast_experiment_vote',
            'arguments': json.dumps({'experiment_id': experiment_id,
                                     'vote_value': value})}}],
    }], sender=helper)


def test_a_later_turn_attaches_the_tool_the_build_goal_did_not(
        db, registry, turn):
    """An agent built for something else is asked, mid-conversation, to vote:
    the turn attach gives it the tool before the model sees the turn."""
    from core.agent_tool_menu import attach_for_turn
    from integrations.agent_engine.thought_experiment_tools import (
        ExperimentVoteTool)
    from integrations.service_tools import registry as reg_mod

    ExperimentVoteTool.register()
    helper, executor = _agents_built_for(registry, 'summarise my inbox')
    assert 'cast_experiment_vote' not in executor._hart_attached_tools

    new, n = attach_for_turn('please vote on the experiment about latency',
                             helper, executor, reg_mod.service_tool_registry)
    assert 'thought_experiment' in new and n >= 1
    assert 'cast_experiment_vote' in executor._hart_attached_tools
    # Idempotent: the same turn again attaches nothing more.
    assert attach_for_turn('vote on the experiment again', helper, executor,
                           reg_mod.service_tool_registry) == ([], 0)

    e = _experiment(db)
    agent = _user(db, 'agent', owner_id=_user(db).id, agent_id='55503')
    turn.set_prompt_id('55503')
    reply = _tool_reply(executor, helper, e.id)
    assert json.loads(reply['tool_responses'][0]['content'])['success'] is True
    db.expire_all()
    assert [(v.voter_id, v.voter_type) for v in
            db.query(ExperimentVote).filter_by(experiment_id=e.id)] == [
                (agent.id, 'agent')]


def test_a_real_autogen_tool_call_runs_on_the_turns_thread(db, registry):
    """Measured, not assumed: autogen executes the tool on the thread that
    drives the reply, so it sees that thread's prompt_id.  A thread that
    carries the agent's prompt_id votes as it; another thread with none is
    refused."""
    import threading
    from integrations.agent_engine.thought_experiment_tools import (
        ExperimentVoteTool)

    ExperimentVoteTool.register()
    helper, executor = _agents_built_for(registry, TURN)
    e = _experiment(db)
    agent = _user(db, 'agent', owner_id=_user(db).id, agent_id='55504')
    out = {}

    def _turn(key, prompt_id):
        thread_local_data.set_prompt_id(prompt_id)
        reply = _tool_reply(executor, helper, e.id)
        out[key] = json.loads(reply['tool_responses'][0]['content'])

    for key, pid in (('with', '55504'), ('without', None)):
        t = threading.Thread(target=_turn, args=(key, pid))
        t.start()
        t.join(60)
    assert out['with']['success'] is True, out
    assert out['without']['success'] is False, out
    db.expire_all()
    assert [v.voter_id for v in
            db.query(ExperimentVote).filter_by(experiment_id=e.id)] == [agent.id]


def test_source_guard_create_and_reuse_turns_both_attach_per_turn():
    """Wiring guard (the behaviour is pinned above on the shared helper):
    CREATE's turn (get_response_group) and REUSE's (get_agent_response)
    both call attach_for_turn, and both builders set the ledger it reads.
    create/reuse cannot be imported in a bare pytest env."""
    import ast
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    for rel, turn_fn in (('hartos/create_recipe.py', 'get_response_group'),
                         ('hartos/reuse_recipe.py', 'get_agent_response')):
        src = open(os.path.join(root, rel), encoding='utf-8').read()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == turn_fn)
        calls = {c.func.id for c in ast.walk(fn)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        assert 'attach_for_turn' in calls, f'{rel}:{turn_fn}'
        assert '_hart_unlocked_tags = set(goal_tags)' in src, rel
