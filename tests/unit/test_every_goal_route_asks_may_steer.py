"""Every route that acts on, or reads, ONE agent goal asks dashboard_service.may_steer.

Review of 924b8e9dc (2026-09-27, REJECTED): "one may_steer for every steering
verb" held only on /api/social/dashboard/agents/<id>/{inject,pause,resume,
cancel}.  Beside it:

* PATCH /api/goals/<id>/status and DELETE /api/goals/<id> had a second rule
  (``goal.created_by and created_by != g.user.id``): any signed-in user could
  pause or archive a goal whose created_by was NULL; an archived goal could
  be revived; the owner and an admin were refused when created_by was a
  label; 404 vs 403 said which ids exist.
* /tracker/experiments/<post_id>/inject and /interview checked only that a
  token existed: user B wrote into A's agent's memory and ran A's agent as A.
  /tracker/dual-context cloned A's goal into new goals OWNED BY A.
* /dashboard/agents/<id>/snapshot, /chat and /a2a had no auth at all: a
  remote caller with no token read another user's live GroupChat.

ROUTES below is the vocabulary: every goal-scoped rule on these blueprints
must be in it (test_every_goal_scoped_route_is_classified reads the app's
url_map), and every entry is driven through the real app: a non-owner is
refused with the SAME answer as an unknown id, the owner is admitted.  A new
goal route without the rule fails here.

Behavioural: real blueprints, real dashboard_service / GoalManager, SQLite.
Patched boundaries: get_db, the token store, the audit log, /chat's HTTP
call, the memory graph's disk.
"""
import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault('HEVOLVE_DB_PATH', ':memory:')
os.environ.setdefault('SOCIAL_DB_PATH', ':memory:')

from hartos.lifecycle_hooks import register_groupchat_for_session  # noqa: E402
from integrations.agent_engine.api import agent_engine_bp  # noqa: E402
from integrations.social.api_dashboard import dashboard_bp  # noqa: E402
from integrations.social.api_tracker import tracker_bp  # noqa: E402
from integrations.social.models import AgentGoal, Base, User  # noqa: E402

REMOTE = {'REMOTE_ADDR': '203.0.113.7'}
TOKEN = {'Authorization': 'Bearer t'}

# (method, rule) -> how the route is called for goal ``gid`` / post ``pid``.
ROUTES = {
    ('POST', '/api/social/dashboard/agents/<agent_id>/inject'):
        lambda gid, pid: ('post', f'/api/social/dashboard/agents/{gid}/inject', {'instruction': 'go'}),
    ('POST', '/api/social/dashboard/agents/<agent_id>/pause'):
        lambda gid, pid: ('post', f'/api/social/dashboard/agents/{gid}/pause', {}),
    ('POST', '/api/social/dashboard/agents/<agent_id>/resume'):
        lambda gid, pid: ('post', f'/api/social/dashboard/agents/{gid}/resume', {}),
    ('POST', '/api/social/dashboard/agents/<agent_id>/cancel'):
        lambda gid, pid: ('post', f'/api/social/dashboard/agents/{gid}/cancel', {}),
    ('GET', '/api/social/dashboard/agents/<agent_id>/snapshot'):
        lambda gid, pid: ('get', f'/api/social/dashboard/agents/{gid}/snapshot', None),
    ('GET', '/api/social/dashboard/agents/<agent_id>/chat'):
        lambda gid, pid: ('get', f'/api/social/dashboard/agents/{gid}/chat', None),
    ('GET', '/api/social/dashboard/agents/<agent_id>/a2a'):
        lambda gid, pid: ('get', f'/api/social/dashboard/agents/{gid}/a2a', None),
    ('GET', '/api/goals/<goal_id>'):
        lambda gid, pid: ('get', f'/api/goals/{gid}', None),
    ('PATCH', '/api/goals/<goal_id>/status'):
        lambda gid, pid: ('patch', f'/api/goals/{gid}/status', {'status': 'paused'}),
    ('DELETE', '/api/goals/<goal_id>'):
        lambda gid, pid: ('delete', f'/api/goals/{gid}', None),
    ('POST', '/api/social/tracker/experiments/<post_id>/inject'):
        lambda gid, pid: ('post', f'/api/social/tracker/experiments/{pid}/inject', {'variable': 'v'}),
    ('POST', '/api/social/tracker/experiments/<post_id>/interview'):
        lambda gid, pid: ('post', f'/api/social/tracker/experiments/{pid}/interview', {'question': 'q'}),
    ('POST', '/api/social/tracker/dual-context'):
        lambda gid, pid: ('post', '/api/social/tracker/dual-context',
                          {'post_id': pid, 'contexts': [{'label': 'a'}, {'label': 'b'}]}),
}

# Tracker rules that act on a post, not on its agent: classified here so a
# NEW tracker rule has to be put in one list or the other.
TRACKER_NOT_AGENT = {
    ('GET', '/api/social/tracker/experiments'),
    ('GET', '/api/social/tracker/experiments/<post_id>'),
    ('GET', '/api/social/tracker/experiments/<post_id>/conversations'),
    ('POST', '/api/social/tracker/experiments/<post_id>/approve'),
    ('POST', '/api/social/tracker/experiments/<post_id>/reject'),
    ('GET', '/api/social/tracker/notifications'),
    ('GET', '/api/social/tracker/experiments/<post_id>/pledges'),
    ('GET', '/api/social/tracker/experiments/<post_id>/pledge-summary'),
    ('POST', '/api/social/tracker/experiments/<post_id>/pledge'),
    ('DELETE', '/api/social/tracker/experiments/<post_id>/pledge/<int:escrow_id>'),
    ('POST', '/api/social/tracker/experiments/<post_id>/consume'),
    ('GET', '/api/social/tracker/experiments/<post_id>/insights'),
    ('GET', '/api/social/tracker/pledges/mine'),
    ('GET', '/api/social/tracker/pledges/all'),
    ('POST', '/api/social/tracker/pledges/<int:escrow_id>/verify'),
    ('GET', '/api/social/tracker/encounters'),
}


@pytest.fixture(scope='module')
def sf():
    eng = create_engine('sqlite://', connect_args={'check_same_thread': False},
                        poolclass=StaticPool)
    Base.metadata.create_all(eng)
    return sessionmaker(bind=eng)


@pytest.fixture
def app(sf, monkeypatch, tmp_path):
    for var in ('TRUSTED_PROXY', 'NUNBA_CI', 'HEVOLVE_TRUST_KONG',
                'HEVOLVE_CLOUD_MODE'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('HEVOLVE_OWNER_USER_ID', 'owner-1')
    a = Flask(__name__)
    a.config['TESTING'] = True
    for bp in (dashboard_bp, agent_engine_bp, tracker_bp):
        a.register_blueprint(bp)
    chat = SimpleNamespace(status_code=200, json=lambda: {'response': 'ok'})
    with patch('integrations.social.models.get_db', side_effect=lambda: sf()), \
         patch('security.immutable_audit_log.get_audit_log'), \
         patch('core.http_pool.pooled_post', return_value=chat) as posted, \
         patch('integrations.channels.memory.memory_graph.MemoryGraph') as graph, \
         patch('core.platform_paths.get_memory_graph_dir', return_value=str(tmp_path)), \
         patch('integrations.social.realtime.publish_event'):
        graph.return_value.register.return_value = 'm1'
        a.chat_post, a.memory_graph = posted, graph
        yield a


@pytest.fixture
def client(app):
    return app.test_client()


def _user(sf, user_type='human', owner_id=None):
    db = sf()
    u = User(username=f'u_{uuid.uuid4().hex[:10]}', user_type=user_type,
             owner_id=owner_id)
    db.add(u)
    db.commit()
    uid = str(u.id)
    db.close()
    return uid


def _goal(sf, owner_id=None, created_by=None, status='active'):
    db = sf()
    gid, pid = uuid.uuid4().hex, uuid.uuid4().hex[:12]
    prompt = str(uuid.uuid4().int % 10**9)
    db.add(AgentGoal(id=gid, owner_id=owner_id, created_by=created_by,
                     goal_type='thought_experiment', title='secret title',
                     prompt_id=prompt, status=status,
                     config_json={'post_id': pid}))
    db.commit()
    db.close()
    gc = SimpleNamespace(messages=[{'role': 'user', 'content': 'PRIVATE'}])
    register_groupchat_for_session(f'{owner_id or "system"}_{prompt}', gc)
    return gid, pid, gc


def _status(sf, gid):
    db = sf()
    try:
        return db.query(AgentGoal).filter(AgentGoal.id == gid).first().status
    finally:
        db.close()


def _goal_count(sf):
    db = sf()
    try:
        return db.query(AgentGoal).count()
    finally:
        db.close()


def _as(uid, is_admin=False, role='flat', is_banned=False):
    user = SimpleNamespace(id=uid, is_admin=is_admin, role=role,
                           is_banned=is_banned)
    return patch('integrations.social.auth._get_user_from_token',
                 return_value=(user, MagicMock()))


def _call(client, key, gid, pid, environ=None, headers=None):
    method, url, body = ROUTES[key](gid, pid)
    kw = {'environ_base': environ or {}, 'headers': headers or {}}
    if body is not None:
        kw['json'] = body
    return getattr(client, method)(url, **kw)


def _goal_scoped_rules(app):
    out = set()
    for r in app.url_map.iter_rules():
        methods = r.methods - {'HEAD', 'OPTIONS'}
        for m in methods:
            key = (m, r.rule)
            if ('<agent_id>' in r.rule or '<goal_id>' in r.rule
                    or r.rule.startswith('/api/social/tracker/')):
                out.add(key)
    return out


# ── the vocabulary guard ────────────────────────────────────────────────

def test_every_goal_scoped_route_is_classified(app):
    rules = _goal_scoped_rules(app)
    assert rules, 'enumeration found nothing -- it is broken'
    unclassified = rules - set(ROUTES) - TRACKER_NOT_AGENT
    assert not unclassified, (
        f'goal-scoped routes nobody decided about: {sorted(unclassified)}; '
        'add each to ROUTES (it must ask may_steer) or, for a tracker route '
        'that acts on a post and not its agent, to TRACKER_NOT_AGENT')
    stale = (set(ROUTES) | TRACKER_NOT_AGENT) - rules
    assert not stale, f'classified routes that no longer exist: {sorted(stale)}'


@pytest.mark.parametrize('key', sorted(ROUTES), ids=lambda k: f'{k[0]} {k[1]}')
def test_a_stranger_gets_the_unknown_id_answer(app, client, sf, key):
    """Another user's goal and a goal that does not exist answer alike, and
    nothing about the goal changes."""
    owner = _user(sf)
    gid, pid, gc = _goal(sf, owner_id=owner, created_by=None)
    before, goals_before = _status(sf, gid), _goal_count(sf)
    with _as(_user(sf)):
        theirs = _call(client, key, gid, pid, REMOTE, TOKEN)
        missing = _call(client, key, uuid.uuid4().hex, uuid.uuid4().hex[:12],
                        REMOTE, TOKEN)
    assert theirs.status_code == 403, (key, theirs.status_code, theirs.get_json())
    assert missing.status_code == theirs.status_code, key
    assert missing.get_json() == theirs.get_json(), key
    assert 'secret title' not in theirs.get_data(as_text=True)
    assert 'PRIVATE' not in theirs.get_data(as_text=True)
    assert _status(sf, gid) == before
    assert _goal_count(sf) == goals_before
    assert gc.messages == [{'role': 'user', 'content': 'PRIVATE'}]
    app.memory_graph.return_value.register.assert_not_called()
    app.chat_post.assert_not_called()


@pytest.mark.parametrize('key', sorted(ROUTES), ids=lambda k: f'{k[0]} {k[1]}')
def test_the_owner_is_admitted(app, client, sf, key):
    owner = _user(sf)
    gid, pid, _ = _goal(sf, owner_id=owner, created_by='agent_daemon')
    with _as(owner):
        r = _call(client, key, gid, pid, REMOTE, TOKEN)
    assert r.status_code in (200, 201), (key, r.status_code, r.get_json())


@pytest.mark.parametrize('key', sorted(ROUTES), ids=lambda k: f'{k[0]} {k[1]}')
def test_no_token_from_another_machine_is_refused(client, sf, key):
    gid, pid, _ = _goal(sf, owner_id=_user(sf))
    r = _call(client, key, gid, pid, REMOTE)
    assert r.status_code == 401, (key, r.status_code)


# ── the specific findings ───────────────────────────────────────────────

def test_a_goal_with_no_created_by_is_not_anyones_to_archive(client, sf):
    owner = _user(sf)
    gid, pid, _ = _goal(sf, owner_id=owner, created_by=None)
    with _as(_user(sf)):
        assert client.delete(f'/api/goals/{gid}', headers=TOKEN,
                             environ_base=REMOTE).status_code == 403
    assert _status(sf, gid) == 'active'


def test_a_cancelled_goal_cannot_be_revived(client, sf):
    owner = _user(sf)
    gid, pid, _ = _goal(sf, owner_id=owner, status='archived')
    with _as(owner):
        r = client.patch(f'/api/goals/{gid}/status', json={'status': 'active'},
                         headers=TOKEN, environ_base=REMOTE)
    assert r.status_code == 400, r.get_json()
    assert _status(sf, gid) == 'archived'


def test_an_admin_manages_any_goal(client, sf):
    gid, pid, _ = _goal(sf, owner_id=_user(sf), created_by='agent_daemon')
    with _as(_user(sf), is_admin=True):
        r = client.patch(f'/api/goals/{gid}/status', json={'status': 'paused'},
                         headers=TOKEN, environ_base=REMOTE)
    assert r.status_code == 200, r.get_json()
    assert _status(sf, gid) == 'paused'


def test_patch_status_takes_only_a_steering_verbs_status(client, sf):
    owner = _user(sf)
    gid, pid, _ = _goal(sf, owner_id=owner)
    with _as(owner):
        r = client.patch(f'/api/goals/{gid}/status', json={'status': 'completed'},
                         headers=TOKEN, environ_base=REMOTE)
    assert r.status_code == 400
    assert _status(sf, gid) == 'active'


def test_interview_runs_the_agent_as_its_owner_only_for_the_owner(app, client, sf):
    owner = _user(sf)
    gid, pid, _ = _goal(sf, owner_id=owner)
    with _as(owner):
        r = client.post(f'/api/social/tracker/experiments/{pid}/interview',
                        json={'question': 'why?'}, headers=TOKEN, environ_base=REMOTE)
    assert r.status_code == 200, r.get_json()
    assert app.chat_post.call_args.kwargs['json']['user_id'] == owner


def test_a_banned_token_on_this_machine_is_not_that_user(client, sf):
    """Review M5 (survived): a banned user's token on a loopback request must
    not stand for that user; the local owner answers instead."""
    banned = _user(sf)
    gid, pid, gc = _goal(sf, owner_id=banned)
    with _as(banned, is_banned=True):
        r = client.post(f'/api/social/dashboard/agents/{gid}/inject',
                        json={'instruction': 'x'}, headers=TOKEN)
    assert r.status_code == 403
    assert len(gc.messages) == 1
