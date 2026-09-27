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
    helper = autogen.ConversableAgent('helper', llm_config=False)
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
    helper = autogen.ConversableAgent('helper', llm_config=False)
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
