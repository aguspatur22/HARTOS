"""A vote is the steward's only when a signed-in human who holds the
steward role cast it.

058c01b05 made the steward's FOR vote part of the one approval rule, but it
recognised the steward by voter_id == 'steward': a string.  The agent tool
cast_experiment_vote passes any voter_id it is given, so any agent could
vote as the steward and approve a security_guardrail experiment (measured at
HEAD with this file: the goal started).

The steward is the account that holds the central role, the one
auth.require_central admits (User.role == 'central', or the is_admin flag
UserService.set_user_role keeps in step with it); no new role store.  Now:
  - only a registered HUMAN account holding that role is the steward; an
    agent never is, whoever owns it and whatever its row says (an agent
    counts as its owner for the quorum, never as the steward);
  - the literal voter_id 'steward' carries no weight of its own;
  - the agent tool cannot vote as the steward: it is not a signed-in
    human, so it refuses a voter_id that resolves to the steward.  The
    steward votes through the signed-in route (voter_id from the JWT).
  - decide()'s steward gate reads the same tally, so there is one steward
    check.

Real SQLite; the agent tool runs through the real db_session.
Each check: {what, check, expected, tolerance 0 (exact invariant)}.
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
    AgentGoal, Base, ExperimentVote, ThoughtExperiment, User)
from integrations.social.thought_experiment_service import (  # noqa: E402
    ThoughtExperimentService)

SECURITY = ('Tighten the security guardrail',
            'A stricter guardrail blocks the vulnerability')


@pytest.fixture
def factory(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{tmp_path / 'steward.db'}")
    Base.metadata.create_all(eng)
    f = sessionmaker(bind=eng, expire_on_commit=False)
    # db_session (the agent tool's session) uses this factory.
    monkeypatch.setattr(social_models, '_SessionLocal', f)
    yield f
    eng.dispose()


@pytest.fixture
def db(factory):
    s = factory()
    yield s
    s.close()


def _user(db, user_type='human', role='flat', owner_id=None, is_admin=False):
    u = User(username=f'livetest_stw_{uuid.uuid4().hex[:8]}',
             user_type=user_type, role=role, owner_id=owner_id,
             is_admin=is_admin)
    db.add(u)
    db.commit()
    return u


def _experiment(db):
    e = ThoughtExperiment(
        id=str(uuid.uuid4()), creator_id=_user(db).id, title=SECURITY[0],
        hypothesis=SECURITY[1], expected_outcome='o', status='voting')
    db.add(e)
    db.commit()
    return e


def _vote(db, exp_id, voter_id, value, voter_type='human'):
    db.add(ExperimentVote(experiment_id=exp_id, voter_id=voter_id,
                          voter_type=voter_type, vote_value=value,
                          confidence=1.0))
    db.commit()


def _people_approve(db, exp_id):
    """Quorum and 0.8 met by people alone: 5 FOR / 1 AGAINST."""
    for v in (2, 2, 2, 2, 2, -1):
        _vote(db, exp_id, _user(db).id, v)


def _evaluate(db, e):
    result = ThoughtExperimentService.request_agent_evaluation(db, e.id)
    db.commit()
    return result, db.query(AgentGoal).count()


def test_an_agent_voting_as_steward_through_the_tool_is_not_the_steward(db):
    from integrations.agent_engine.thought_experiment_tools import (
        cast_experiment_vote)

    e = _experiment(db)
    _people_approve(db, e.id)
    cast_experiment_vote(e.id, 'steward', vote_value=2,
                         voter_type='human', confidence=1.0)
    db.expire_all()

    assert ThoughtExperimentService.tally_votes(db, e.id)['steward_vote'] is None
    result, goals = _evaluate(db, e)
    assert result['success'] is False
    assert result['verdict']['reason'] == 'steward_required'
    assert goals == 0


def test_a_steward_role_human_satisfies_steward_required(db):
    e = _experiment(db)
    _people_approve(db, e.id)
    steward = _user(db, role='central', is_admin=True)
    _vote(db, e.id, steward.id, 2)

    result, goals = _evaluate(db, e)
    assert result['success'] is True and result['goal_id']
    assert goals == 1


def test_a_steward_against_blocks(db):
    e = _experiment(db)
    _people_approve(db, e.id)
    _vote(db, e.id, _user(db, role='central', is_admin=True).id, -2)
    result, goals = _evaluate(db, e)
    assert result['success'] is False and goals == 0


def test_one_steward_against_is_not_outvoted_by_another_steward_for(db):
    e = _experiment(db)
    _people_approve(db, e.id)
    for _ in range(3):   # 9 FOR / 2 AGAINST overall = 0.82: over 0.8
        _vote(db, e.id, _user(db).id, 2)
    _vote(db, e.id, _user(db, role='central', is_admin=True).id, 2)
    _vote(db, e.id, _user(db, role='central', is_admin=True).id, -1)
    result, goals = _evaluate(db, e)
    assert result['success'] is False and goals == 0
    assert result['verdict']['reason'] == 'steward_required'


def test_an_agent_the_steward_owns_is_not_the_steward(db):
    """It counts as its owner for the quorum, never as the steward."""
    e = _experiment(db)
    _people_approve(db, e.id)
    steward = _user(db, role='central', is_admin=True)
    agent = _user(db, user_type='agent', owner_id=steward.id)
    _vote(db, e.id, agent.id, 2, voter_type='agent')

    result, goals = _evaluate(db, e)
    assert result['verdict']['reason'] == 'steward_required'
    assert goals == 0


def test_an_agent_row_carrying_the_role_is_not_the_steward(db):
    e = _experiment(db)
    _people_approve(db, e.id)
    agent = _user(db, user_type='agent', role='central', is_admin=True)
    _vote(db, e.id, agent.id, 2)

    result, goals = _evaluate(db, e)
    assert result['verdict']['reason'] == 'steward_required'
    assert goals == 0


def test_an_ordinary_human_is_not_the_steward(db):
    e = _experiment(db)
    _people_approve(db, e.id)
    _vote(db, e.id, _user(db, role='regional').id, 2)
    result, goals = _evaluate(db, e)
    assert result['verdict']['reason'] == 'steward_required'
    assert goals == 0


def test_the_tool_refuses_to_vote_as_the_real_steward(db):
    """The tool is no signed-in human: naming the steward's own id must not
    let an agent cast the steward's vote."""
    from integrations.agent_engine.thought_experiment_tools import (
        cast_experiment_vote)

    e = _experiment(db)
    steward = _user(db, role='central', is_admin=True)
    out = json.loads(cast_experiment_vote(e.id, steward.id, vote_value=2,
                                          voter_type='human'))
    db.expire_all()

    assert out['success'] is False
    assert db.query(ExperimentVote).filter_by(voter_id=steward.id).count() == 0


def test_the_tool_still_casts_an_ordinary_vote(db):
    """Control: the refusal is only for the steward."""
    from integrations.agent_engine.thought_experiment_tools import (
        cast_experiment_vote)

    # security_guardrail takes no agent votes (VOTER_RULES), so a person.
    e = _experiment(db)
    person = _user(db)
    out = json.loads(cast_experiment_vote(e.id, person.id, vote_value=1,
                                          voter_type='human'))
    db.expire_all()
    assert out['success'] is True, out
    assert db.query(ExperimentVote).filter_by(voter_id=person.id).count() == 1


def test_decide_asks_the_same_steward_rule(db):
    e = _experiment(db)
    _vote(db, e.id, 'steward', 2)
    out = ThoughtExperimentService.decide(db, e.id, 'go')
    assert out.get('error') == 'steward_vote_required'

    _vote(db, e.id, _user(db, role='central', is_admin=True).id, 2)
    out = ThoughtExperimentService.decide(db, e.id, 'go')
    assert out.get('status') == 'decided'
