"""A node the hive admitted may run a SHARED agent on this node over A2A,
proving who it is with the gossip key it already has.  No API key, no new
credential.

OWNER RULING 2026-09-26 (verbatim): "we had trust created in same network and
when auto mode the consent is implicit only a hash verified node is enough
what other creds are we talking about? torrents is the analogy for our
design".

THE DEFECT: 05641511d/01b1b2f65 admitted message/send only when
security.middleware's /chat gate admitted the caller.  peer_reuse.
invoke_peer_agent sends no credential, so on a bundled desktop (NUNBA_BUNDLED)
or a central / keyed node every peer invoke became 401: the remote-invoke
half of cross-node REUSE stopped working for the peers it exists for.

THE FIX: the invoker signs the JSON-RPC body with this node's Ed25519 key
(sender {node_id, public_key} + timestamp inside the signed body), and the
server also admits a request signed by a peer that holds a PeerNode row
(the gossip admission gate: guardrail hash + Ed25519), is not banned, whose
key on file is the key that signed, and whose timestamp is fresh.  The one
rule lives in integrations.social.discovery.admitted_peer_sender.  The
sharing rule (peer_reuse.export_allowed) still applies.  message/get and
task/cancel get the same admission, and a /chat-gate verdict other than 401
(a phone's consent_pending 403) is passed through, not reported as 401.

Everything here drives the REAL Flask app, the REAL API gate, the REAL
jsonrpc view and the REAL invoker; the DB is the suite's real SQLite with
real PeerNode rows.  Only the network (pooled_post -> test_client), the
agent executor (a spy) and the invoker's identity (a fresh key standing for
another node) are stand-ins.
"""
import sys
import time
import types
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from flask import Flask, jsonify

import integrations.google_a2a.peer_reuse as peer_reuse
import security.node_integrity as ni
from integrations.google_a2a.google_a2a_integration import A2AProtocolServer
from integrations.social import discovery
from integrations.social.integrity_service import WITNESS_TIMESTAMP_MAX_AGE
from integrations.social.models import Base, PeerNode, db_session, get_engine
from integrations.social.sync_engine import SyncEngine
from security.middleware import _apply_api_auth

AGENT = 'livetest_shared_0'
OTHER_AGENT = 'livetest_other_0'
PEER_URL = 'http://node-b:5000'
REMOTE = {'REMOTE_ADDR': '198.51.100.23'}
LOCAL = {'REMOTE_ADDR': '127.0.0.1'}


class _Resp:
    def __init__(self, flask_resp):
        self.status_code = flask_resp.status_code
        self._json = flask_resp.get_json(silent=True)

    def json(self):
        if self._json is None:
            raise ValueError('no json')
        return self._json


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ('NUNBA_BUNDLED', 'HEVOLVE_NODE_TIER', 'HEVOLVE_API_KEY',
              'HEVOLVE_OWNER_USER_ID', 'NUNBA_CI', 'TRUSTED_PROXY'):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv('NUNBA_BUNDLED', '1')
    Base.metadata.create_all(get_engine())
    getattr(discovery, '_seen_peer_signatures', {}).clear()
    yield
    getattr(discovery, '_seen_peer_signatures', {}).clear()
    with db_session() as db:
        db.query(PeerNode).filter(
            PeerNode.node_id.like('livetest-%')).delete(
                synchronize_session=False)


@pytest.fixture
def invoker(monkeypatch):
    """This process signs as ANOTHER node: a fresh Ed25519 key and node_id.
    The server side never uses the process key, only the PeerNode rows."""
    priv = Ed25519PrivateKey.generate()
    monkeypatch.setattr(ni, '_private_key', priv)
    monkeypatch.setattr(ni, '_public_key', priv.public_key())
    node_id = f'livetest-{uuid.uuid4().hex[:12]}'
    monkeypatch.setattr(SyncEngine, 'canonical_node_id',
                        staticmethod(lambda: node_id))
    return types.SimpleNamespace(node_id=node_id,
                                 public_key=ni.get_public_key_hex())


def _other_key():
    return Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()


def _admit(node_id, public_key, **over):
    row = dict(node_id=node_id, url=PEER_URL, public_key=public_key,
               status='active', integrity_status='unverified')
    row.update(over)
    with db_session() as db:
        db.add(PeerNode(**row))


@pytest.fixture
def node(monkeypatch):
    """The serving node: its own gate, its own jsonrpc view, two agents."""
    monkeypatch.setattr(peer_reuse, 'export_allowed',
                        lambda pid: str(pid) != 'livetest_other')
    app = Flask('livetest_serving_node')
    _apply_api_auth(app)
    srv = A2AProtocolServer(app, 'http://node-a')
    ran = []

    async def spy(text, ctx):
        ran.append(text)
        return {'role': 'model', 'parts': [{'text': f'ran:{text}'}]}
    srv.register_agent(AGENT, 'shared', 'd', [{'id': 's'}], spy)
    srv.register_agent(OTHER_AGENT, 'private', 'd', [{'id': 's'}], spy)
    srv.setup_routes()
    client = app.test_client()
    secrets = {'security.secrets_manager': types.SimpleNamespace(
        get_secret=lambda name: __import__('os').environ.get(name, ''))}
    seen = []

    def routed_post(url, json=None, timeout=None, **kw):
        r = client.post(urlsplit(url).path, json=json, environ_base=REMOTE)
        seen.append((r.status_code, r.get_json(silent=True)))
        return _Resp(r)
    monkeypatch.setattr(peer_reuse, 'pooled_post', routed_post)
    with patch.dict(sys.modules, secrets):
        yield types.SimpleNamespace(client=client, ran=ran, seen=seen)


def _post(node, body, agent=AGENT, environ=REMOTE):
    r = node.client.post(f'/a2a/{agent}/jsonrpc', json=body,
                         environ_base=environ)
    return r.status_code, r.get_json()


def _signed(method='message/send', agent=AGENT, text='summarise', **over):
    body = {'jsonrpc': '2.0', 'id': uuid.uuid4().hex, 'method': method,
            'params': {'message': {'messageId': uuid.uuid4().hex,
                                   'parts': [{'kind': 'text', 'text': text}]}},
            'agent_id': agent}
    body = discovery.signed_peer_request(body)
    body.update(over)
    return body


# ── the ruling: an admitted node runs a shared agent ─────────────────────

@pytest.mark.parametrize('env', [{'NUNBA_BUNDLED': '1'},
                                 {'HEVOLVE_NODE_TIER': 'central'},
                                 {'HEVOLVE_API_KEY': 'livetest-key'}])
def test_an_admitted_peer_runs_a_shared_agent_through_the_real_invoker(
        node, invoker, monkeypatch, env):
    monkeypatch.delenv('NUNBA_BUNDLED', raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _admit(invoker.node_id, invoker.public_key)
    result = peer_reuse.invoke_peer_agent(PEER_URL, AGENT, 'collect metrics')
    assert node.seen[-1][0] == 200, node.seen
    assert result['state'] == 'completed', result
    assert node.ran == ['collect metrics']


def test_the_invoker_signs_with_the_nodes_gossip_identity(node, invoker):
    _admit(invoker.node_id, invoker.public_key)
    captured = {}
    real = peer_reuse.pooled_post

    def capture(url, json=None, timeout=None, **kw):
        captured['body'] = json
        return real(url, json=json, timeout=timeout, **kw)
    with patch.object(peer_reuse, 'pooled_post', capture):
        peer_reuse.invoke_peer_agent(PEER_URL, AGENT, 'x')
    body = captured['body']
    assert body['sender'] == {'node_id': invoker.node_id,
                              'public_key': invoker.public_key}
    assert body['agent_id'] == AGENT
    assert abs(body['timestamp'] - time.time()) < 5
    assert ni.verify_json_signature(invoker.public_key, body,
                                    body['signature'])


def test_a_suspicious_peer_is_still_admitted(node, invoker):
    """'suspicious' is where a served-out ban and a fraud score over 40 land;
    no peer path refuses it (peer_reuse.admitted_peers, witness and auditor
    selection all filter only 'banned').  Refusing on it here would make the
    invoke stricter than the recipe pull it backs up."""
    _admit(invoker.node_id, invoker.public_key, integrity_status='suspicious')
    assert _post(node, _signed())[0] == 200
    assert node.ran == ['summarise']


# ── what is refused ─────────────────────────────────────────────────────

def test_an_unknown_node_is_refused(node, invoker):
    status, body = _post(node, _signed())
    assert status == 401, body
    assert 'unknown' in body['error']['message']
    assert node.ran == []


@pytest.mark.parametrize('ban_until', [timedelta(hours=1), None])
def test_a_banned_node_is_refused(node, invoker, ban_until):
    _admit(invoker.node_id, invoker.public_key, integrity_status='banned',
           ban_until=ban_until and datetime.utcnow() + ban_until)
    status, body = _post(node, _signed())
    assert status == 401, body
    assert 'banned' in body['error']['message']
    assert node.ran == []


def test_a_ban_that_has_not_expired_is_refused_whatever_the_status(
        node, invoker):
    _admit(invoker.node_id, invoker.public_key,
           integrity_status='suspicious',
           ban_until=datetime.utcnow() + timedelta(hours=1))
    assert _post(node, _signed())[0] == 401
    assert node.ran == []


def test_a_ban_that_has_expired_no_longer_refuses(node, invoker):
    _admit(invoker.node_id, invoker.public_key,
           integrity_status='suspicious',
           ban_until=datetime.utcnow() - timedelta(hours=1))
    assert _post(node, _signed())[0] == 200


def test_a_key_other_than_the_one_on_file_is_refused(node, invoker):
    other = _other_key()
    _admit(invoker.node_id, other)
    status, body = _post(node, _signed())
    assert status == 401, body
    assert node.ran == []


def test_a_sender_that_names_the_key_on_file_but_signs_with_another_is_refused(
        node, invoker):
    other = _other_key()
    _admit(invoker.node_id, other)
    body = _signed()
    body['sender'] = {'node_id': invoker.node_id, 'public_key': other}
    assert _post(node, body)[0] == 401
    assert node.ran == []


def test_a_sender_naming_another_key_is_refused_even_signed_by_the_key_on_file(
        node, invoker):
    """The key the sender names must BE the key on file, not merely be
    outvoted by it."""
    _admit(invoker.node_id, invoker.public_key)
    body = _signed()
    body['sender'] = {'node_id': invoker.node_id, 'public_key': _other_key()}
    body['signature'] = ni.sign_json_payload(body)
    status, resp = _post(node, body)
    assert status == 401, resp
    assert 'key on file' in resp['error']['message']
    assert node.ran == []


def test_a_body_without_a_timestamp_is_refused(node, invoker):
    _admit(invoker.node_id, invoker.public_key)
    body = _signed()
    del body['timestamp']
    body['signature'] = ni.sign_json_payload(body)
    status, resp = _post(node, body)
    assert status == 401, resp
    assert 'timestamp' in resp['error']['message']


def test_the_replay_record_forgets_what_has_expired(node, invoker):
    _admit(invoker.node_id, invoker.public_key)
    discovery._seen_peer_signatures['livetest-old'] = time.time() - 1
    assert _post(node, _signed())[0] == 200
    assert 'livetest-old' not in discovery._seen_peer_signatures
    assert len(discovery._seen_peer_signatures) == 1


def test_a_tampered_body_is_refused(node, invoker):
    _admit(invoker.node_id, invoker.public_key)
    body = _signed(text='summarise')
    body['params']['message']['parts'][0]['text'] = 'delete everything'
    status, _ = _post(node, body)
    assert status == 401
    assert node.ran == []


def test_an_unsigned_body_naming_an_admitted_sender_is_refused(node, invoker):
    _admit(invoker.node_id, invoker.public_key)
    body = _signed()
    del body['signature']
    assert _post(node, body)[0] == 401
    assert node.ran == []


def _resigned_at(body, skew):
    """The invoker's own body, validly signed, but dated ``skew`` s off now."""
    body['timestamp'] += skew
    body['signature'] = ni.sign_json_payload(body)
    return body


@pytest.mark.parametrize('sign', [-1, 1])
def test_a_stale_or_future_timestamp_is_refused(node, invoker, sign):
    _admit(invoker.node_id, invoker.public_key)
    body = _resigned_at(_signed(), sign * (WITNESS_TIMESTAMP_MAX_AGE + 30))
    status, resp = _post(node, body)
    assert status == 401, resp
    assert 'timestamp' in resp['error']['message']
    assert node.ran == []


def test_a_timestamp_inside_the_window_is_admitted(node, invoker):
    _admit(invoker.node_id, invoker.public_key)
    body = _resigned_at(_signed(), -(WITNESS_TIMESTAMP_MAX_AGE - 10))
    assert _post(node, body)[0] == 200


def test_a_replayed_request_is_refused(node, invoker):
    _admit(invoker.node_id, invoker.public_key)
    body = _signed()
    assert _post(node, body)[0] == 200
    status, resp = _post(node, body)
    assert status == 401, resp
    assert node.ran == ['summarise']


def test_a_request_signed_for_one_agent_does_not_run_another(node, invoker):
    """agent_id is in the signed body; the URL is not.  A captured request
    re-posted to another agent's path must not run that agent."""
    with patch.object(peer_reuse, 'export_allowed', lambda pid: True):
        _admit(invoker.node_id, invoker.public_key)
        status, _ = _post(node, _signed(agent=AGENT), agent=OTHER_AGENT)
    assert status == 401
    assert node.ran == []


def test_an_unsigned_remote_caller_on_a_bundled_node_is_refused(node):
    body = {'jsonrpc': '2.0', 'id': 1, 'method': 'message/send', 'params': {
        'message': {'parts': [{'kind': 'text', 'text': 'x'}]}}}
    status, resp = _post(node, body)
    assert status == 401, resp
    assert node.ran == []


def test_the_desktops_own_caller_still_runs_unsigned(node):
    body = {'jsonrpc': '2.0', 'id': 1, 'method': 'message/send', 'params': {
        'message': {'parts': [{'kind': 'text', 'text': 'local'}]}}}
    assert _post(node, body, environ=LOCAL)[0] == 200
    assert node.ran == ['local']


def test_an_admitted_peer_cannot_run_an_agent_this_node_does_not_share(
        node, invoker):
    _admit(invoker.node_id, invoker.public_key)
    status, resp = _post(node, _signed(agent=OTHER_AGENT), agent=OTHER_AGENT)
    assert status == 403, resp
    assert node.ran == []


def test_a_db_failure_refuses(node, invoker, monkeypatch):
    _admit(invoker.node_id, invoker.public_key)

    def broken(*a, **k):
        raise RuntimeError('db down')
    monkeypatch.setattr(discovery, 'admitted_peer_sender', broken)
    status, _ = _post(node, _signed())
    assert status == 503
    assert node.ran == []


# ── message/get and task/cancel: same admission ─────────────────────────

def _run_one_locally(node):
    body = {'jsonrpc': '2.0', 'id': 1, 'method': 'message/send', 'params': {
        'message': {'messageId': 'livetest_task_1',
                    'parts': [{'kind': 'text', 'text': 'x'}]}}}
    assert _post(node, body, environ=LOCAL)[0] == 200
    return 'livetest_task_1'


@pytest.mark.parametrize('method', ['message/get', 'task/cancel'])
def test_task_reads_and_cancels_need_the_same_admission(node, method):
    task_id = _run_one_locally(node)
    body = {'jsonrpc': '2.0', 'id': 2, 'method': method,
            'params': {'taskId': task_id}}
    status, resp = _post(node, body)
    assert status == 401, resp
    assert 'result' not in resp


def test_an_admitted_peer_reads_a_task(node, invoker):
    task_id = _run_one_locally(node)
    _admit(invoker.node_id, invoker.public_key)
    body = discovery.signed_peer_request({
        'jsonrpc': '2.0', 'id': 3, 'method': 'message/get',
        'params': {'taskId': task_id}, 'agent_id': AGENT})
    status, resp = _post(node, body)
    assert status == 200, resp
    assert resp['result']['state'] == 'completed'


def test_the_local_caller_reads_a_task(node):
    task_id = _run_one_locally(node)
    status, resp = _post(node, {'jsonrpc': '2.0', 'id': 4,
                                'method': 'message/get',
                                'params': {'taskId': task_id}},
                         environ=LOCAL)
    assert status == 200
    assert resp['result']['id'] == task_id


# ── the gate's own verdict passes through ───────────────────────────────

def test_a_phones_consent_pending_is_reported_as_403_not_401(
        node, monkeypatch):
    from integrations.social import consent_service as cs
    from tests.unit.test_device_access_gate import Phone
    monkeypatch.setenv('HEVOLVE_OWNER_USER_ID', 'livetest-owner')
    phone = Phone()
    body = {'jsonrpc': '2.0', 'id': 5, 'method': 'message/send',
            'params': {'message': {'parts': [{'kind': 'text', 'text': 'x'}]}}}
    with patch.object(cs, '_emit'):
        r = node.client.post(
            f'/a2a/{AGENT}/jsonrpc', json=body, environ_base=REMOTE,
            headers={'Authorization': f'Bearer {phone.token()}'})
    try:
        assert r.status_code == 403, r.get_json()
        assert r.get_json()['error']['message'] == 'consent_pending'
        assert node.ran == []
    finally:
        from integrations.social.models import UserConsent
        with db_session() as db:
            db.query(UserConsent).filter_by(user_id='livetest-owner').delete(
                synchronize_session=False)
