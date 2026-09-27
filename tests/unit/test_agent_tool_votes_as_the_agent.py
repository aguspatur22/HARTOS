"""The agent vote tool casts an AGENT's vote, counted as its owner; it can
never cast a human's.

cast_experiment_vote is the agent entry (autogen / MCP).  It took voter_id
and voter_type as given, so an agent could name any person's id with
voter_type 'human' and cast that person's full-weight vote: one agent could
manufacture the distinct-identity quorum (>= 3 identities, >= 2 FOR) the
owner ruled must come from different people (measured at HEAD with this
file).  84c650a5d refused only the steward's id.

One rule in the tool, which replaces that steward-only refusal: the vote is
cast by the AGENT the argument resolves to, canonically -- a users row with
user_type 'agent', found by its id or by its prompt id (User.agent_id) --
and is recorded under that row's id as an agent vote, whatever voter_type
the caller passed.  tally_votes then counts it as its owner's identity (an
agent is its owner).  An argument that resolves to no agent -- a human, the
literal 'steward', an unknown id -- is refused and nothing is written.

Real SQLite; the tool runs through the real db_session.
"""
import json
import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault('HEVOLVE_DB_PATH', ':memory:')
os.environ.setdefault('SOCIAL_DB_PATH', ':memory:')

from integrations.social import models as social_models  # noqa: E402
from integrations.social.models import (  # noqa: E402
    Base, ExperimentVote, ThoughtExperiment, User)
from integrations.social.thought_experiment_service import (  # noqa: E402
    ThoughtExperimentService)


@pytest.fixture
def db(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{tmp_path / 'agentvote.db'}")
    Base.metadata.create_all(eng)
    f = sessionmaker(bind=eng, expire_on_commit=False)
    monkeypatch.setattr(social_models, '_SessionLocal', f)
    s = f()
    yield s
    s.close()
    eng.dispose()


def _user(db, user_type='human', owner_id=None, agent_id=None):
    u = User(username=f'livetest_av_{uuid.uuid4().hex[:8]}',
             user_type=user_type, owner_id=owner_id, agent_id=agent_id)
    db.add(u)
    db.commit()
    return u


def _experiment(db):
    # technical_improvement: agents may vote here.
    e = ThoughtExperiment(
        id=str(uuid.uuid4()), creator_id=_user(db).id, title='Cache warmup',
        hypothesis='Faster cache warmup lowers latency', expected_outcome='o',
        status='voting')
    db.add(e)
    db.commit()
    return e


def _cast(e, voter_id, voter_type='agent', value=2):
    from integrations.agent_engine.thought_experiment_tools import (
        cast_experiment_vote)
    return json.loads(cast_experiment_vote(e.id, voter_id, vote_value=value,
                                           voter_type=voter_type,
                                           confidence=1.0))


def _rows(db, e):
    db.expire_all()
    return [(v.voter_id, v.voter_type)
            for v in db.query(ExperimentVote).filter_by(experiment_id=e.id)]


@pytest.mark.parametrize('voter_type', ['human', 'agent'])
def test_naming_another_person_is_refused_and_writes_nothing(db, voter_type):
    e = _experiment(db)
    person = _user(db)
    out = _cast(e, person.id, voter_type=voter_type)
    assert out['success'] is False
    assert _rows(db, e) == []


@pytest.mark.parametrize('name', ['steward', 'no-such-id'])
def test_a_name_that_is_no_agent_is_refused(db, name):
    e = _experiment(db)
    assert _cast(e, name)['success'] is False
    assert _rows(db, e) == []


def test_one_agent_cannot_build_a_quorum_of_people(db):
    """The attack: three people's ids from one agent.  None is written, so
    no quorum exists."""
    e = _experiment(db)
    for _ in range(3):
        _cast(e, _user(db).id, voter_type='human')
    tally = ThoughtExperimentService.tally_votes(db, e.id)
    assert tally['distinct_voters'] == 0 and tally['quorum_met'] is False


def test_an_agents_vote_is_an_agent_vote_counted_as_its_owner(db):
    e = _experiment(db)
    owner = _user(db)
    agent = _user(db, user_type='agent', owner_id=owner.id)
    out = _cast(e, agent.id, voter_type='human')   # the claim is ignored
    assert out['success'] is True, out
    assert _rows(db, e) == [(agent.id, 'agent')]
    tally = ThoughtExperimentService.tally_votes(db, e.id)
    assert tally['agent_votes'] == 1 and tally['human_votes'] == 0
    assert tally['distinct_voters'] == 1


def test_two_agents_of_one_owner_are_one_identity(db):
    e = _experiment(db)
    owner = _user(db)
    a1 = _user(db, user_type='agent', owner_id=owner.id)
    a2 = _user(db, user_type='agent', owner_id=owner.id)
    assert _cast(e, a1.id)['success'] is True
    assert _cast(e, a2.id)['success'] is True
    tally = ThoughtExperimentService.tally_votes(db, e.id)
    assert tally['agent_votes'] == 2
    assert tally['distinct_voters'] == 1
    assert tally['distinct_supporters'] == 1


def test_an_agent_named_by_its_prompt_id_is_resolved_to_its_row(db):
    """Agents know themselves by prompt id; the vote is recorded under the
    agent's users row, so the tally can find its owner."""
    e = _experiment(db)
    owner = _user(db)
    agent = _user(db, user_type='agent', owner_id=owner.id,
                  agent_id='8865956')
    assert _cast(e, '8865956')['success'] is True
    assert _rows(db, e) == [(agent.id, 'agent')]
    assert ThoughtExperimentService.tally_votes(db, e.id)['distinct_voters'] == 1
