"""Egress has one rule, and every leg that leaves the node asks it.

Owner rulings 2026-09-26: egress is a message that goes to OTHER people's
nodes, and it is scrubbed only there; local records stay raw; security must
not partition the hive.  Review of 19d4c5b02 and bd92deac4 measured four
places where that did not hold:

1. ``MessageBus._route_local`` emits ``bus.<topic>`` on the EventBus, and the
   EventBus WAMP bridge published it verbatim to ``com.hartos.event.bus.<topic>``
   -- a URI naming no user -- while the MessageBus's own Crossbar leg carried
   the scrubbed copy.  (Reviewer probe: the Crossbar leg redacted, the
   EventBus WAMP leg carried the raw email address; a chat.response
   'SECRET-CHAT' left the same way.)
2. The scrub covered a fixed list of eight field names: 'reply', 'caption',
   a nested 'body_text', a tuple and publish_async's {'raw': ...} wrapper
   all went out raw.
3. The "what may leave, and to where" logic had a third copy beside
   security.edge_privacy and secret_redactor.
4. ``hart_intelligence_entry.publish_async`` published straight to Crossbar,
   past the bus scrub.

Plus bd92deac4 review F2 (the ownership rule answered False for every
concrete per-user URI) and F3 (two answers to "who is this event for").

Real MessageBus, real EventBus, real edge_privacy / DLP / secret redactor,
and the real publish_async source.  Mocked boundaries: the two WAMP
sessions (recording fakes), the Crossbar HTTP client, the SSE broker.
"""
import ast
import asyncio
import json
import logging
import os
import sys
import threading
import types
from unittest.mock import MagicMock, patch

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.peer_link.message_bus import MessageBus, reset_message_bus  # noqa: E402
from core.platform import events as ev  # noqa: E402
from core.platform.events import EventBus  # noqa: E402
from core.platform.registry import get_registry, reset_registry  # noqa: E402

EMAIL = 'a.person@example.com'
PHONE = '415-555-0199'
PROMPT_ID = '9876543210'            # 10 digits: the DLP phone pattern eats it
PEER_URL = 'http://203.0.113.7:6777/api'   # the DLP ip pattern eats its host
API_KEY = 'sk-ant-' + 'A1b2C3d4E5' * 5     # secret_redactor anthropic_key


# ── fakes for the two WAMP legs ────────────────────────────────────────────

class _EventBusSession:
    """The EventBus bridge's WAMP session (async publish on its own loop)."""

    def __init__(self):
        self.published = []

    async def publish(self, uri, payload):
        self.published.append((uri, payload))


class _CrossbarSession:
    """hartos.crossbar_server.wamp_session: MessageBus._route_crossbar hands
    publish()'s return value to asyncio.ensure_future."""

    def __init__(self):
        self.published = []
        self._loop = asyncio.new_event_loop()

    def publish(self, uri, payload):
        self.published.append((uri, payload))
        done = self._loop.create_future()
        done.set_result(None)
        return done


@pytest.fixture()
def legs(monkeypatch, tmp_path):
    """Both WAMP legs recording, the EventBus registered, emits synchronous."""
    monkeypatch.setenv('HEVOLVE_DATA_DIR', str(tmp_path))
    monkeypatch.delenv('HEVOLVE_USER_ID', raising=False)
    reset_registry()
    reset_message_bus()
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    ebus_session = _EventBusSession()
    ebus = EventBus()
    ebus._wamp_session = ebus_session
    ebus._wamp_loop = loop
    ebus._wamp_connected = True
    get_registry().register('events', lambda: ebus, singleton=True)
    cb_session = _CrossbarSession()
    monkeypatch.setitem(sys.modules, 'hartos.crossbar_server',
                        types.SimpleNamespace(wamp_session=cb_session))
    monkeypatch.setattr(ev, 'emit_event',
                        lambda t, d=None, async_=True: ebus.emit(t, d))
    monkeypatch.setattr(ev, 'broadcast_sse_safe', lambda *a, **k: False)
    yield types.SimpleNamespace(ebus=ebus, ebus_session=ebus_session,
                                cb=cb_session, loop=loop)
    loop.call_soon_threadsafe(loop.stop)
    reset_registry()
    reset_message_bus()


def _drain(loop):
    for _ in range(3):
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(5)


def _no_link_manager():
    return patch('core.peer_link.link_manager.get_link_manager',
                 side_effect=RuntimeError('no peerlink in this test'))


# ── 1. the EventBus WAMP bridge never carries the bus echo ─────────────────

def test_a_bus_publish_leaves_only_on_its_own_scrubbed_crossbar_leg(legs):
    local = []
    legs.ebus.on('bus.community.message', lambda t, d: local.append(dict(d)))
    with _no_link_manager():
        MessageBus().publish(
            'community.message',
            {'community_id': 'c1', 'text': f'mail me at {EMAIL} or {PHONE}'},
            user_id='u1')
        MessageBus().publish('chat.response', {'text': 'SECRET-CHAT'},
                             user_id='u-1', skip_crossbar=True)
    _drain(legs.loop)

    bridged = [u for u, _ in legs.ebus_session.published]
    assert not [u for u in bridged if u.startswith('com.hartos.event.bus.')], bridged
    assert 'SECRET-CHAT' not in json.dumps(legs.ebus_session.published)
    assert EMAIL not in json.dumps(legs.ebus_session.published)

    assert len(legs.cb.published) == 1
    uri, payload = legs.cb.published[0]
    assert uri == 'com.hertzai.hevolve.community.c1'
    sent = json.loads(payload)
    assert sent['text'] == 'mail me at [EMAIL_REDACTED] or [PHONE_REDACTED]'
    assert sent['community_id'] == 'c1' and sent['user_id'] == 'u1'
    # the in-process echo is the local record: raw
    assert local and local[0]['text'] == f'mail me at {EMAIL} or {PHONE}'


def test_node_internal_topics_never_bridge_and_everything_else_still_does(legs):
    """bus.* and channel.* are node-internal; peer gossip and theme keep their
    WAMP leg (other HARTOS nodes subscribe to com.hartos.event.peer.*)."""
    legs.ebus.emit('channel.registered', {'name': 'discord'})
    legs.ebus.emit('bus.task.confirmation', {'user_id': 'u1', 'text': EMAIL})
    legs.ebus.emit('peer.capability.announce',
                   {'peer_id': 'p1', 'endpoint': PEER_URL})
    legs.ebus.emit('theme.changed', {'theme': 'aurora'})
    _drain(legs.loop)
    uris = [u for u, _ in legs.ebus_session.published]
    assert uris == ['com.hartos.event.peer.capability.announce',
                    'com.hartos.event.theme.changed']
    # gossip is not scrubbed: the endpoint reaches peers intact
    assert legs.ebus_session.published[0][1]['endpoint'] == PEER_URL


# ── 2. the scrub is structural ─────────────────────────────────────────────

def test_every_content_leaf_is_scrubbed_whatever_its_key():
    from security.edge_privacy import scrub_for_egress
    original = {
        'reply': f'call {PHONE}',
        'caption': EMAIL,
        'message': {'author': 'x', 'body_text': EMAIL},
        'text': (EMAIL,),
        'raw': f'{EMAIL} {API_KEY}',
        'items': [{'note': PHONE}, 7, None, True],
    }
    before = json.dumps(original, sort_keys=True)
    out = scrub_for_egress(original)
    assert out['reply'] == 'call [PHONE_REDACTED]'
    assert out['caption'] == '[EMAIL_REDACTED]'
    assert out['message'] == {'author': 'x', 'body_text': '[EMAIL_REDACTED]'}
    assert out['text'] == ('[EMAIL_REDACTED]',) and isinstance(out['text'], tuple)
    assert EMAIL not in out['raw'] and API_KEY not in out['raw']
    assert out['items'] == [{'note': '[PHONE_REDACTED]'}, 7, None, True]
    assert json.dumps(original, sort_keys=True) == before   # never mutated
    assert scrub_for_egress(f'bare {EMAIL}') == 'bare [EMAIL_REDACTED]'


def test_identifier_and_routing_keys_travel_byte_identical():
    from security.edge_privacy import scrub_for_egress
    ids = {
        'user_id': 'u1', 'prompt_id': PROMPT_ID, 'request_id': PROMPT_ID,
        'uid': PROMPT_ID, 'msg_id': PROMPT_ID, 'peer_url': PEER_URL,
        'endpoint': PEER_URL, 'url': PEER_URL, 'signature': 'ab' * 32,
        'relay_path': ['203.0.113.7'], 'task_type': 'async',
        'timestamp': 1.5, 'created_at': '2026-09-26T10:00:00',
        'nested': {'id': PROMPT_ID, 'text': PHONE},
    }
    out = scrub_for_egress(ids)
    assert out['nested']['text'] == '[PHONE_REDACTED]'
    out['nested']['text'] = PHONE
    assert out == ids


def test_unlisted_keys_are_scrubbed_on_the_bus_crossbar_leg(legs):
    body = {'community_id': 'c1', 'prompt_id': PROMPT_ID,
            'reply': f'call {PHONE}', 'caption': EMAIL,
            'message': {'body_text': EMAIL}}
    with _no_link_manager():
        MessageBus().publish('community.message', body, user_id='u1')
    sent = json.loads(legs.cb.published[0][1])
    assert sent['reply'] == 'call [PHONE_REDACTED]'
    assert sent['caption'] == '[EMAIL_REDACTED]'
    assert sent['message'] == {'body_text': '[EMAIL_REDACTED]'}
    assert sent['prompt_id'] == PROMPT_ID


# ── 4. publish_async goes through the same rule ────────────────────────────

class _Client:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload):
        self.published.append((topic, payload))


class _Inline:
    def submit(self, fn, *a, **k):
        fn(*a, **k)


def _publish_async(client):
    """The real publish_async, extracted from hart_intelligence_entry.py
    (that module cannot be imported in a unit test) and bound to fakes for
    its module globals: the Crossbar HTTP client and the executor."""
    src = open(os.path.join(_ROOT, 'hart_intelligence_entry.py'),
               encoding='utf-8').read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == 'publish_async')
    ns = {'json': json, 'os': os, 'client': client,
          'crossbar_executor': _Inline(), 'app': MagicMock()}
    exec(compile(ast.Module(body=[fn], type_ignores=[]),
                 'hart_intelligence_entry.py', 'exec'), ns)
    return ns['publish_async']


def test_publish_async_scrubs_a_topic_other_people_subscribe_to(legs):
    client = _Client()
    with _no_link_manager():
        _publish_async(client)('com.hertzai.longrunning.log', {
            'user_id': 'u1', 'uid': PROMPT_ID, 'status': 'INITIALIZED',
            'text': f'mail {EMAIL}'})
    assert len(client.published) == 1
    topic, payload = client.published[0]
    sent = json.loads(payload)
    assert topic == 'com.hertzai.longrunning.log'
    assert sent['text'] == 'mail [EMAIL_REDACTED]'
    assert sent['uid'] == PROMPT_ID and sent['status'] == 'INITIALIZED'


def test_publish_async_keeps_a_non_json_message_a_string(legs):
    client = _Client()
    with _no_link_manager():
        _publish_async(client)('com.hertzai.hevolve.pupitpublish',
                               f'say call {PHONE}')
    assert client.published == [('com.hertzai.hevolve.pupitpublish',
                                  'say call [PHONE_REDACTED]')]


def test_publish_async_sends_the_users_own_topic_byte_identical(legs):
    client = _Client()
    raw = json.dumps({'user_id': 'u-1', 'text': f'mail {EMAIL}'})
    with _no_link_manager():
        _publish_async(client)('com.hertzai.hevolve.chat.u-1', raw)
    assert client.published == [('com.hertzai.hevolve.chat.u-1', raw)]


def test_publish_async_withholds_what_it_could_not_scrub(legs, caplog):
    client = _Client()
    with _no_link_manager(), \
            patch('security.edge_privacy.scrub_for_egress',
                  side_effect=RuntimeError('dlp broken')), \
            caplog.at_level(logging.WARNING, logger='hevolve_security'):
        _publish_async(client)('com.hertzai.longrunning.log',
                               {'user_id': 'u1', 'text': EMAIL})
    assert client.published == []
    assert any('dlp broken' in r.getMessage() for r in caplog.records)


# ── bd92 F2: one ownership rule, templates and concrete URIs ───────────────

def test_the_ownership_rule_answers_for_concrete_uris():
    from security.edge_privacy import crossbar_uri_is_per_user as own
    assert own('com.hertzai.hevolve.chat.{user_id}') is True
    assert own('com.hertzai.hevolve.community.{community_id}') is False
    assert own('com.hertzai.hevolve.chat.u-1', 'u-1') is True
    assert own('com.hertzai.hevolve.chat.new.u-1', 'u-1') is True
    assert own('com.hertzai.hevolve.chat.u-1', 'u-2') is False
    assert own('com.hertzai.hevolve.chat.u-11', 'u-1') is False
    assert own('com.hertzai.hevolve.community.c1', 'u-1') is False
    assert own('com.hertzai.hevolve.chat.u-1', '') is False
    assert own('com.hartos.event.agent.ui.update', 'u-1') is False


def test_a_one_person_event_bridges_only_onto_its_owners_uri(legs):
    """Unpatched: the bridge publishes com.hartos.event.<topic>, so a card
    whose topic ends in its user's id is on that user's own URI."""
    legs.ebus.emit('agent.ui.update.u-1', {'user_id': 'u-1', 'code': 'K7Q2'})
    legs.ebus.emit('agent.ui.update.u-1', {'user_id': 'u-2', 'code': 'K7Q2'})
    legs.ebus.emit('agent.ui.update', {'user_id': 'u-1', 'code': 'K7Q2'})
    _drain(legs.loop)
    assert [u for u, _ in legs.ebus_session.published] == [
        'com.hartos.event.agent.ui.update.u-1']


# ── bd92 F3: one answer to "who is this event for" ─────────────────────────

def test_one_table_says_who_an_event_is_for():
    assert ev.topic_audience('bus.chat.response') == ev.AUDIENCE_NODE
    assert ev.topic_audience('channel.registered') == ev.AUDIENCE_NODE
    assert ev.topic_audience('agent.ui.update') == ev.AUDIENCE_ONE_PERSON
    assert ev.topic_audience('community.feed') == ev.AUDIENCE_EVERYONE
    assert ev.topic_audience('agent.action.completed') == ev.AUDIENCE_ADDRESSED
    classes = (ev._NODE_INTERNAL_TOPIC_PREFIXES, ev._ONE_PERSON_TOPIC_PREFIXES,
               ev._SSE_GLOBAL_PREFIXES)
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            clash = [(x, y) for x in a for y in b
                     if x.startswith(y) or y.startswith(x)]
            assert clash == [], 'a prefix answers two ways: %r' % clash


def test_realtime_never_treats_a_one_person_topic_as_public():
    """realtime listed 'agent.' as public; the one table says 'agent.ui.' is
    one person's, so it must name its publisher like any per-user topic."""
    from integrations.social.realtime import _authorize_topic_for_user_id as ok
    assert ok('agent.ui.update', 'u-1') is False
    assert ok('agent.ui.update.u-1', 'u-1') is True
    assert ok('agent.lifecycle.started', '') is True      # still public
    assert ok('com.hertzai.hevolve.social.u-1', 'u-1') is True
    assert ok('com.hertzai.hevolve.social.u-1', 'u-2') is False


# ── the transit policy lives in one place ──────────────────────────────────

def test_every_crossbar_leg_asks_the_one_policy(legs):
    """Owner delegation 2026-09-27: a per-user topic transiting a router is
    NOT egress.  Were that policy ever flipped, crossbar_leg_is_users_own is
    the one place; all three Crossbar legs follow it."""
    client = _Client()
    with patch('security.edge_privacy.crossbar_leg_is_users_own',
               return_value=False), _no_link_manager():
        MessageBus().publish('chat.social', {'text': EMAIL}, user_id='u-1',
                             skip_peerlink=True)
        _publish_async(client)('com.hertzai.hevolve.chat.u-1',
                               {'user_id': 'u-1', 'text': EMAIL})
        legs.ebus.emit('agent.ui.update.u-1', {'user_id': 'u-1', 'code': 'K'})
    _drain(legs.loop)
    assert json.loads(legs.cb.published[0][1])['text'] == '[EMAIL_REDACTED]'
    assert json.loads(client.published[0][1])['text'] == '[EMAIL_REDACTED]'
    assert legs.ebus_session.published == []


def test_the_users_own_topic_goes_raw_through_the_router(legs):
    with _no_link_manager():
        MessageBus().publish('chat.social', {'text': EMAIL}, user_id='u-1',
                             skip_peerlink=True)
    uri, payload = legs.cb.published[0]
    assert uri == 'com.hertzai.hevolve.social.u-1'
    assert json.loads(payload)['text'] == EMAIL


# ── 3. the other egress sites route through the same scrub ─────────────────

def test_redact_for_scope_scrubs_nested_content_and_keeps_ids():
    from security.edge_privacy import PrivacyScope, ScopeGuard
    data = {'_privacy_scope': PrivacyScope.FEDERATED, 'prompt_id': PROMPT_ID,
            'peer_url': PEER_URL, 'meta': {'note': f'mail {EMAIL}'}}
    out = ScopeGuard().redact_for_scope(data, PrivacyScope.FEDERATED)
    assert out['prompt_id'] == PROMPT_ID and out['peer_url'] == PEER_URL
    assert out['meta'] == {'note': 'mail [EMAIL_REDACTED]'}


def test_redact_experience_redacts_secrets_in_every_content_leaf():
    from security.secret_redactor import redact_experience
    exp = {'prompt': 'short', 'response': '', 'model_id': 'm',
           'attribution_chain': [{'observation': f'used key {API_KEY}'}],
           'escalation_reason': f'leaked {API_KEY}'}
    out = redact_experience(exp)
    assert API_KEY not in json.dumps(out)
    assert out['model_id'] == 'm'


# ── the guard that keeps it at one ─────────────────────────────────────────

_SCAN_DIRS = ('core', 'security', 'integrations', 'hartos')
_CANONICAL = os.path.join('security', 'edge_privacy.py')
_REDACTORS = {'redact', 'redact_secrets', 'scrub_text', 'scrub_for_egress',
              'redact_fields', 'map_content'}


def _sources():
    yield os.path.join(_ROOT, 'hart_intelligence_entry.py')
    for d in _SCAN_DIRS:
        for base, dirs, files in os.walk(os.path.join(_ROOT, d)):
            dirs[:] = [x for x in dirs if x != '__pycache__']
            for f in files:
                if f.endswith('.py'):
                    yield os.path.join(base, f)


def _called_names(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            yield f.id if isinstance(f, ast.Name) else getattr(f, 'attr', '')


def test_source_guard_the_egress_rule_has_one_home():
    """A second copy of 'what may leave, and to where' fails here:
      * a membership test on the literal '{user_id}' (the template rule),
      * an .endswith(f'.{user_id}') suffix test (the concrete rule),
      * a recursive payload walker that calls a redactor,
      * a module-level *CONTENT_FIELDS / *IDENTIFIER_KEY* table,
    anywhere but security/edge_privacy.py."""
    found = []
    for path in _sources():
        rel = os.path.relpath(path, _ROOT)
        if rel == _CANONICAL:
            continue
        try:
            tree = ast.parse(open(path, encoding='utf-8').read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Compare)
                    and isinstance(node.left, ast.Constant)
                    and node.left.value == '{user_id}'
                    and any(isinstance(o, ast.In) for o in node.ops)):
                found.append((rel, node.lineno, "'{user_id}' in"))
            if (isinstance(node, ast.Call)
                    and getattr(node.func, 'attr', '') == 'endswith'
                    and node.args and isinstance(node.args[0], ast.JoinedStr)):
                parts = node.args[0].values
                if (len(parts) == 2 and isinstance(parts[0], ast.Constant)
                        and parts[0].value in ('.', '/')
                        and isinstance(parts[1], ast.FormattedValue)
                        and 'user' in ast.unparse(parts[1].value)):
                    found.append((rel, node.lineno, 'user-suffix rule'))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls = set(_called_names(node))
                walks_dicts = any(
                    isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'isinstance'
                    and len(n.args) == 2 and 'dict' in ast.unparse(n.args[1])
                    for n in ast.walk(node))
                if node.name in calls and walks_dicts and calls & _REDACTORS:
                    found.append((rel, node.lineno, 'redacting walker ' + node.name))
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    name = getattr(t, 'id', '')
                    if 'CONTENT_FIELDS' in name or 'IDENTIFIER_KEY' in name:
                        found.append((rel, node.lineno, name))
    assert found == [], 'a second egress rule: %r' % found
