"""A correction counts only when HevolveAI says it was learned (log defect M19-B).

Measured on the live node, api_server_20260922.log: all 5 corrections failed
inside HevolveAI ("No sensor encoding available"), the server still answered
HTTP 200, and WorldModelBridge.submit_correction counted every 200 as
total_corrections += 1. That counter feeds benchmark_registry,
federated_aggregator and ip_service.

These tests call the real submit_correction; only the HTTP post and the
in-process send_expert_correction are stubbed.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from integrations.agent_engine.world_model_bridge import WorldModelBridge


def _http_bridge():
    bridge = WorldModelBridge()
    bridge._in_process = False
    bridge._http_disabled = False
    bridge._api_url = 'http://localhost:8000'   # local: no consent gate
    return bridge


def _reply(body):
    resp = MagicMock(status_code=200)
    resp.json.return_value = body
    return resp


# The reply HevolveAI sends since it reports its own verdict.
FAILED = {'status': 'failed', 'success': False,
          'message': 'Correction received but not learned: No sensor encoding '
                     'available. Pass in context.',
          'statistics': {'success': False, 'error': 'No sensor encoding'}}
LEARNED = {'status': 'success', 'success': True,
           'message': 'Correction received and learned',
           'statistics': {'success': True, 'correction_id': 4}}
# The reply an older HevolveAI sends: always "success"; only the provider
# result in statistics tells the truth.
OLD_FAILED = {'status': 'success', 'message': 'Correction received and learned',
              'statistics': {'success': False, 'error': 'No sensor encoding'}}
OLD_LEARNED = {'status': 'success', 'message': 'Correction received and learned',
               'statistics': {'success': True, 'correction_id': 4}}


@pytest.mark.parametrize('body,learned', [
    (FAILED, False), (LEARNED, True), (OLD_FAILED, False), (OLD_LEARNED, True),
    ({'status': 'success'}, False),
    # only a literal True is a verdict of learned
    ({'status': 'success', 'success': None,
      'statistics': {'success': True}}, False),
])
def test_http_correction_counts_only_a_learned_reply(body, learned):
    bridge = _http_bridge()
    with patch('integrations.agent_engine.world_model_bridge.pooled_post',
               return_value=_reply(dict(body))):
        result = bridge.submit_correction('London', 'Paris')
    assert bridge._stats['total_corrections'] == (1 if learned else 0)
    assert result['success'] is learned


@pytest.fixture
def in_process_send(monkeypatch):
    """A stand-in for hevolveai.embodied_ai.rl_ef.send_expert_correction."""
    holder = {}
    mod = types.ModuleType('hevolveai.embodied_ai.rl_ef')
    mod.send_expert_correction = lambda **kw: holder['result']
    for name in ('hevolveai', 'hevolveai.embodied_ai'):
        monkeypatch.setitem(sys.modules, name,
                            sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, 'hevolveai.embodied_ai.rl_ef', mod)
    return holder


@pytest.mark.parametrize('result,learned', [
    ({'success': False, 'error': 'No sensor encoding'}, False),
    ({'success': True, 'correction_id': 1}, True),
])
def test_in_process_correction_counts_only_a_learned_result(
        in_process_send, result, learned):
    in_process_send['result'] = result
    bridge = WorldModelBridge()
    bridge._in_process = True
    bridge._provider = object()
    out = bridge.submit_correction('London', 'Paris')
    assert bridge._stats['total_corrections'] == (1 if learned else 0)
    assert out['success'] is learned
