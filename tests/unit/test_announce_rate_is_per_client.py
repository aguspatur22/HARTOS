"""The gossip rate limit must be per CLIENT, not per socket peer.

Behind Kong every request reaches central from the gateway address, so keying
the limiter on request.remote_addr put every node in the network into ONE
budget of 10 calls a minute. With each node announcing every gossip interval,
nodes were refused (429) and aged out of the peer table. The limiter now keys
on _observed_ip(): the last X-Forwarded-For hop, which Kong appends.

Runs the real routes through a Flask test client; the gossip handler behind
the limiter is stubbed so only the limiter's decision is under test.
"""
import os
import sys
from unittest.mock import patch

import pytest
from flask import Flask

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrations.social import discovery  # noqa: E402

GATEWAY = '172.21.0.1'


@pytest.fixture
def client():
    discovery._ANNOUNCE_RATE.clear()
    app = Flask(__name__)
    app.register_blueprint(discovery.discovery_bp)
    yield app.test_client()
    discovery._ANNOUNCE_RATE.clear()


def _announce(client, xff):
    # A body with no node_id: the route answers 400 AFTER the limiter, so a
    # 400 means "allowed through", a 429 means "rate limited".
    return client.post('/api/social/peers/announce', json={},
                       headers={'X-Forwarded-For': xff},
                       environ_base={'REMOTE_ADDR': GATEWAY})


def test_distinct_clients_behind_one_gateway_are_not_limited_together(client):
    codes = [_announce(client, '203.0.113.%d' % i).status_code for i in range(1, 12)]
    assert 429 not in codes, codes


def test_one_client_is_still_limited(client):
    codes = [_announce(client, '203.0.113.7').status_code for _ in range(12)]
    assert codes[:10] == [400] * 10 and codes[10:] == [429, 429], codes


def test_broadcast_and_embedding_delta_routes_are_per_client_too(client):
    for path in ('/api/social/peers/broadcast', '/api/social/peers/embedding-delta',
                 '/api/social/peers/exchange'):
        with patch.object(discovery, '_check_announce_rate', return_value=False) as rate:
            r = client.post(path, json={}, headers={'X-Forwarded-For': '198.51.100.4'},
                            environ_base={'REMOTE_ADDR': GATEWAY})
        assert r.status_code == 429, path
        assert rate.call_args[0][0] == '198.51.100.4', (path, rate.call_args)


# A PUBLIC socket peer is a direct client, not our gateway: its
# X-Forwarded-For is whatever it chose to write (review of 55d8b9152 measured
# 500 of 500 requests through, 500 keys in the table, with a fresh header each).
# Routable on purpose: Python's ipaddress counts the RFC 5737 documentation
# ranges (198.51.100.0/24 etc.) as private, which would make it a forwarder.
DIRECT = '93.184.216.34'


def test_a_direct_client_cannot_escape_the_limit_by_forging_the_header(client):
    codes = [client.post('/api/social/peers/announce', json={},
                         headers={'X-Forwarded-For': '203.0.113.%d' % i},
                         environ_base={'REMOTE_ADDR': DIRECT}).status_code
             for i in range(1, 30)]
    assert codes[:10] == [400] * 10, codes
    assert set(codes[10:]) == {429}, codes
    assert list(discovery._ANNOUNCE_RATE) == [DIRECT]


def test_the_configured_trusted_proxy_is_believed_even_when_public(client, monkeypatch):
    monkeypatch.setenv('TRUSTED_PROXY', DIRECT)
    codes = [client.post('/api/social/peers/announce', json={},
                         headers={'X-Forwarded-For': '203.0.113.%d' % i},
                         environ_base={'REMOTE_ADDR': DIRECT}).status_code
             for i in range(1, 15)]
    assert 429 not in codes, codes


def test_check_client_rate_charges_the_socket_peer_of_a_direct_client():
    """check_client_rate is what security/middleware.py's device-ask limiter
    calls too, so both limiters charge the same client."""
    app = Flask(__name__)
    with patch.object(discovery, '_check_announce_rate', return_value=True) as rate:
        with app.test_request_context(headers={'X-Forwarded-For': '203.0.113.5'},
                                      environ_base={'REMOTE_ADDR': DIRECT}):
            discovery.check_client_rate()
        with app.test_request_context(headers={'X-Forwarded-For': '203.0.113.5'},
                                      environ_base={'REMOTE_ADDR': GATEWAY}):
            discovery.check_client_rate()
    assert [c[0][0] for c in rate.call_args_list] == [DIRECT, '203.0.113.5']


def test_idle_clients_are_swept_so_the_table_stays_bounded(client, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(discovery._time, 'time', lambda: now[0])
    discovery._rate_last_sweep[0] = 0.0
    for i in range(50):
        discovery._check_announce_rate('203.0.113.%d' % i)
    assert len(discovery._ANNOUNCE_RATE) == 50
    now[0] += discovery._RATE_WINDOW  # a window passes with no requests
    discovery._check_announce_rate('198.51.100.1')
    assert list(discovery._ANNOUNCE_RATE) == ['198.51.100.1']
