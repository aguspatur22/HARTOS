"""Remote compute is paid for by the requester and earned by the operator.

Owner rulings 2026-09-26:
  (a) the metered boundary is work that goes to "hive nodes ... that's not
      their node", tracked inside HARTOS;
  (b) "proportinal to compute spent and earned": ONE measured quantity is
      debited from the requester and credited to the serving operator,
      spend == earn before the 90/9/1 split;
  (c) "for local person'a work zero spark earned": own node, a SAME_USER
      node, or a local model costs 0 and earns 0.

Every test runs the real code against a real SQLite schema: the wallet, the
MeteredAPIUsage row and the settlement are observed, never a mock's call args
standing in for a balance.
"""
import os
import sys
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrations.social import models as _models  # noqa: E402
from integrations.social.models import (  # noqa: E402
    Base, User, PeerNode, ResonanceWallet, ResonanceTransaction,
    MeteredAPIUsage,
)

REQUESTER = 'user-requester'
OPERATOR = 'user-operator'
SERVING_NODE = 'node-of-operator'


@pytest.fixture
def db_factory(monkeypatch):
    """A fresh in-memory schema; every get_db() in the code under test and
    every db the test opens share it."""
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(_models, 'get_db', lambda: factory())
    # No PeerLink manager is running in a unit test: no link proves anything
    # unless a test installs one.
    fake_mgr = MagicMock()
    fake_mgr.get_link.return_value = None
    monkeypatch.setattr('core.peer_link.link_manager.get_link_manager',
                        lambda: fake_mgr)
    monkeypatch.setenv('HEVOLVE_NODE_ID', 'node-of-requester')
    yield factory
    engine.dispose()


@contextmanager
def _session(factory):
    db = factory()
    try:
        yield db
        db.commit()
    finally:
        db.close()


def _seed(factory, requester_spark=1000, operator_id=OPERATOR,
          node_id=SERVING_NODE):
    with _session(factory) as db:
        for uid in {REQUESTER, OPERATOR, operator_id} - {None}:
            db.add(User(id=uid, username=uid, user_type='human'))
        db.add(ResonanceWallet(user_id=REQUESTER, spark=requester_spark,
                               spark_lifetime=requester_spark))
        db.add(PeerNode(node_id=node_id, url='http://peer.example:6777',
                        node_operator_id=operator_id))


def _spark(factory, user_id):
    with _session(factory) as db:
        w = db.query(ResonanceWallet).filter_by(user_id=user_id).first()
        return w.spark if w else 0


def _rows(factory):
    with _session(factory) as db:
        return [(r.requester_user_id, r.operator_id, r.node_id,
                 r.tokens_in, r.tokens_out, r.estimated_spark_cost,
                 r.settlement_status, r.task_source)
                for r in db.query(MeteredAPIUsage).all()]


def _tokens_for_spark(n):
    """Measured tokens that price at exactly n Spark under the canonical rate."""
    from integrations.agent_engine.budget_gate import spark_per_1k_compute_tokens
    return int(round(n * 1000 / spark_per_1k_compute_tokens()))


def _settle(factory):
    from integrations.agent_engine.revenue_aggregator import settle_metered_api_costs
    with _session(factory) as db:
        return settle_metered_api_costs(db)


# ─── the rate: an EXISTING compute-to-Spark conversion, not a new number ───

class TestCanonicalRate:

    def test_rate_is_composed_from_the_two_existing_conversions(self):
        from integrations.agent_engine.budget_gate import spark_per_1k_compute_tokens
        from integrations.social.hosting_reward_service import GPU_SECONDS_PER_1K_TOKENS
        from integrations.social.resonance_engine import AWARD_TABLE
        expected = (GPU_SECONDS_PER_1K_TOKENS / 3600.0
                    * AWARD_TABLE['compute_hour']['spark'])
        assert spark_per_1k_compute_tokens() == pytest.approx(expected)
        assert expected > 0

    def test_rate_reads_the_award_table_live(self, monkeypatch):
        from integrations.agent_engine.budget_gate import spark_per_1k_compute_tokens
        from integrations.social import resonance_engine
        base = spark_per_1k_compute_tokens()
        monkeypatch.setitem(resonance_engine.AWARD_TABLE, 'compute_hour',
                            {'spark': resonance_engine.AWARD_TABLE['compute_hour']['spark'] * 2})
        assert spark_per_1k_compute_tokens() == pytest.approx(base * 2)

    def test_compute_stats_use_the_same_tokens_to_gpu_seconds_constant(self):
        from integrations.social.hosting_reward_service import (
            HostingRewardService, GPU_SECONDS_PER_1K_TOKENS)
        peer = MagicMock(gpu_hours_served=0, total_inferences=0,
                         energy_kwh_contributed=0, metered_api_costs_absorbed=0)
        usage = MagicMock(tokens_in=3000, tokens_out=600, actual_usd_cost=0)
        db = MagicMock()
        db.query.return_value.filter_by.return_value.first.return_value = peer
        db.query.return_value.filter.return_value.all.return_value = [usage]
        with patch('integrations.agent_engine.model_registry.model_registry') as reg:
            reg.get_total_energy_kwh.return_value = 0.0
            out = HostingRewardService.aggregate_compute_stats(db, 'n1')
        assert out['gpu_hours_added'] == pytest.approx(
            round(3.6 * GPU_SECONDS_PER_1K_TOKENS / 3600.0, 4))


# ─── charge_remote_compute: the ONE function ───

class TestChargeRemoteCompute:

    def test_other_persons_node_debits_requester_by_measured_units(self, db_factory):
        from integrations.agent_engine.budget_gate import charge_remote_compute
        _seed(db_factory, requester_spark=100)
        tin, tout = _tokens_for_spark(2), _tokens_for_spark(1)
        moved = charge_remote_compute(REQUESTER, SERVING_NODE, tin, tout,
                                      source='test', ref_id='r1', model_id='m')
        assert moved == 3
        assert _spark(db_factory, REQUESTER) == 97
        rows = _rows(db_factory)
        assert len(rows) == 1
        req, op, node, rin, rout, spark, status, src = rows[0]
        assert (req, op, node, rin, rout, spark, status) == (
            REQUESTER, OPERATOR, SERVING_NODE, tin, tout, 3, 'pending')
        # The operator has earned nothing yet: the credit is the settlement's.
        assert _spark(db_factory, OPERATOR) == 0

    def test_settlement_credits_operator_the_same_quantity(self, db_factory):
        from integrations.agent_engine.budget_gate import charge_remote_compute
        _seed(db_factory, requester_spark=100)
        moved = charge_remote_compute(REQUESTER, SERVING_NODE,
                                      _tokens_for_spark(4), 0, source='test')
        result = _settle(db_factory)
        assert _spark(db_factory, OPERATOR) == moved == 4
        assert _spark(db_factory, REQUESTER) == 96
        assert result['settled_count'] == 1
        assert _rows(db_factory)[0][6] == 'settled'
        # settling twice pays once
        _settle(db_factory)
        assert _spark(db_factory, OPERATOR) == 4
        with _session(db_factory) as db:
            earned = db.query(ResonanceTransaction).filter_by(
                user_id=OPERATOR, source_type='hive_compute_earned').all()
            spent = db.query(ResonanceTransaction).filter_by(
                user_id=REQUESTER, source_type='hive_compute_spent').all()
        assert [t.amount for t in earned] == [4]
        assert [t.amount for t in spent] == [-4]

    def test_fractions_carry_until_a_whole_spark_is_owed(self, db_factory):
        from integrations.agent_engine.budget_gate import charge_remote_compute
        _seed(db_factory, requester_spark=100)
        sixty_percent = int(_tokens_for_spark(1) * 0.6)
        assert charge_remote_compute(REQUESTER, SERVING_NODE, sixty_percent, 0,
                                     source='t') == 0
        assert _spark(db_factory, REQUESTER) == 100
        assert charge_remote_compute(REQUESTER, SERVING_NODE, sixty_percent, 0,
                                     source='t') == 1
        assert _spark(db_factory, REQUESTER) == 99
        _settle(db_factory)
        assert _spark(db_factory, OPERATOR) == 1

    def test_own_node_costs_and_earns_nothing(self, db_factory):
        from integrations.agent_engine.budget_gate import charge_remote_compute
        _seed(db_factory, requester_spark=100, operator_id=REQUESTER)
        assert charge_remote_compute(REQUESTER, SERVING_NODE,
                                     _tokens_for_spark(5), 0, source='t') == 0
        _settle(db_factory)
        assert _spark(db_factory, REQUESTER) == 100
        assert _rows(db_factory) == []

    def test_same_user_link_costs_and_earns_nothing(self, db_factory, monkeypatch):
        from core.peer_link.link import PeerLink, TrustLevel
        from integrations.agent_engine.budget_gate import charge_remote_compute
        _seed(db_factory, requester_spark=100)
        monkeypatch.setenv('HEVOLVE_USER_ID', REQUESTER)
        link = PeerLink(SERVING_NODE, 'peer.example:6777', TrustLevel.SAME_USER)
        mgr = MagicMock()
        mgr.get_link.side_effect = lambda pid: link if pid == SERVING_NODE else None
        monkeypatch.setattr('core.peer_link.link_manager.get_link_manager',
                            lambda: mgr)
        assert charge_remote_compute(REQUESTER, SERVING_NODE,
                                     _tokens_for_spark(5), 0, source='t') == 0
        _settle(db_factory)
        assert _spark(db_factory, REQUESTER) == 100
        assert _spark(db_factory, OPERATOR) == 0
        assert _rows(db_factory) == []

    def test_peer_link_is_not_ownership(self, db_factory, monkeypatch):
        """A PEER link to someone else's node proves nothing: charged."""
        from core.peer_link.link import PeerLink, TrustLevel
        from integrations.agent_engine.budget_gate import charge_remote_compute
        _seed(db_factory, requester_spark=100)
        monkeypatch.setenv('HEVOLVE_USER_ID', REQUESTER)
        link = PeerLink(SERVING_NODE, 'peer.example:6777', TrustLevel.PEER)
        mgr = MagicMock()
        mgr.get_link.return_value = link
        monkeypatch.setattr('core.peer_link.link_manager.get_link_manager',
                            lambda: mgr)
        assert charge_remote_compute(REQUESTER, SERVING_NODE,
                                     _tokens_for_spark(2), 0, source='t') == 2

    def test_unknown_operator_moves_nothing(self, db_factory):
        """No operator to credit means no charge: spend must equal earn."""
        from integrations.agent_engine.budget_gate import charge_remote_compute
        _seed(db_factory, requester_spark=100)
        assert charge_remote_compute(REQUESTER, 'node-nobody-knows',
                                     _tokens_for_spark(3), 0, source='t') == 0
        assert _spark(db_factory, REQUESTER) == 100
        assert _rows(db_factory) == []

    def test_insufficient_spark_refuses_debit_and_credit(self, db_factory):
        """Wallet semantics (ResonanceService.spend_spark): all or nothing.
        The row stays as 'unfunded' so the operator's served work is visible,
        and settlement credits nothing for it."""
        from integrations.agent_engine.budget_gate import charge_remote_compute
        _seed(db_factory, requester_spark=2)
        assert charge_remote_compute(REQUESTER, SERVING_NODE,
                                     _tokens_for_spark(5), 0, source='t') == 0
        assert _spark(db_factory, REQUESTER) == 2
        rows = _rows(db_factory)
        assert len(rows) == 1 and rows[0][6] == 'unfunded'
        _settle(db_factory)
        assert _spark(db_factory, OPERATOR) == 0
        # the refused Spark is written off, not re-billed on the next call
        assert charge_remote_compute(REQUESTER, SERVING_NODE,
                                     _tokens_for_spark(1), 0, source='t') == 1
        assert _spark(db_factory, REQUESTER) == 1

    def test_nothing_measured_moves_nothing(self, db_factory):
        from integrations.agent_engine.budget_gate import charge_remote_compute
        _seed(db_factory)
        assert charge_remote_compute(REQUESTER, SERVING_NODE, 0, 0, source='t') == 0
        assert charge_remote_compute('', SERVING_NODE, 10 ** 6, 0, source='t') == 0
        assert _rows(db_factory) == []


class TestRemoteCallTokens:

    def test_usage_block_wins(self):
        from integrations.agent_engine.budget_gate import remote_call_tokens
        assert remote_call_tokens('a b c', 'd e', {'prompt_tokens': 11,
                                                    'completion_tokens': 7}) == (11, 7)

    def test_counted_when_no_usage(self):
        from core.token_utils import count_tokens_for_text
        from integrations.agent_engine.budget_gate import remote_call_tokens
        p, r = 'hello there ' * 50, 'general kenobi ' * 20
        assert remote_call_tokens(p, r, None) == (
            count_tokens_for_text(p), count_tokens_for_text(r))


# ─── the exits: hive expert call ───

def _hive_expert(peer_id=SERVING_NODE, is_local=False):
    from integrations.agent_engine.model_registry import ModelBackend, ModelTier
    return ModelBackend(
        model_id=f'hive-{peer_id}-big', display_name='Hive: big',
        tier=ModelTier.EXPERT,
        config_list_entry={'model': 'big', 'api_key': 'tok',
                           'base_url': 'https://peer.example/v1',
                           'price': [0, 0], 'peer_id': peer_id},
        is_local=is_local)


def _dispatcher():
    from integrations.agent_engine.model_registry import ModelRegistry
    from integrations.agent_engine.speculative_dispatcher import SpeculativeDispatcher
    return SpeculativeDispatcher(model_registry=ModelRegistry())


def _resp(status, body):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    return r


class TestHiveExpertExit:

    def test_completed_hive_call_charges_real_usage_tokens(self, db_factory):
        _seed(db_factory, requester_spark=100)
        tin, tout = _tokens_for_spark(1), _tokens_for_spark(2)
        body = {'choices': [{'message': {'content': 'the answer'}}],
                'usage': {'prompt_tokens': tin, 'completion_tokens': tout}}
        with patch('requests.post', return_value=_resp(200, body)):
            out = _dispatcher()._dispatch_expert_langchain(
                _hive_expert(), 'q', REQUESTER, None, 'general', None)
        assert out == 'the answer'
        assert _spark(db_factory, REQUESTER) == 97
        rows = _rows(db_factory)
        assert [(r[0], r[1], r[3], r[4]) for r in rows] == [
            (REQUESTER, OPERATOR, tin, tout)]

    def test_completed_hive_call_without_usage_charges_counted_tokens(self, db_factory):
        from core.token_utils import count_tokens_for_text
        _seed(db_factory, requester_spark=100)
        prompt, answer = 'why ' * 40, 'because ' * 30
        body = {'choices': [{'message': {'content': answer}}]}
        with patch('requests.post', return_value=_resp(200, body)):
            _dispatcher()._dispatch_expert_langchain(
                _hive_expert(), prompt, REQUESTER, None, 'general', None)
        rows = _rows(db_factory)
        assert [(r[3], r[4]) for r in rows] == [
            (count_tokens_for_text(prompt), count_tokens_for_text(answer))]

    @pytest.mark.parametrize('resp', [
        _resp(500, {}),
        _resp(200, {'choices': []}),
        _resp(200, {'choices': [{'message': {'content': ''}}],
                    'usage': {'prompt_tokens': 10 ** 7, 'completion_tokens': 0}}),
    ])
    def test_failed_hive_call_charges_nothing(self, db_factory, resp):
        _seed(db_factory, requester_spark=100)
        with patch('requests.post', return_value=resp):
            out = _dispatcher()._dispatch_expert_langchain(
                _hive_expert(), 'q', REQUESTER, None, 'general', None)
        assert out == ''
        assert _spark(db_factory, REQUESTER) == 100
        assert _rows(db_factory) == []

    def test_local_model_charges_nothing(self, db_factory, monkeypatch):
        _seed(db_factory, requester_spark=100)
        monkeypatch.delenv('NUNBA_BUNDLED', raising=False)
        with patch('requests.post', return_value=_resp(200, {'response': 'local answer'})), \
                patch('integrations.agent_engine.dispatch._internal_auth_headers',
                      return_value={}):
            out = _dispatcher()._dispatch_expert_langchain(
                _hive_expert(is_local=True), 'q', REQUESTER, None, 'general', None)
        assert out == 'local answer'
        assert _spark(db_factory, REQUESTER) == 100
        assert _rows(db_factory) == []

    def test_discovery_records_the_serving_peer_on_the_backend(self):
        from integrations.agent_engine.hive_expert_discovery import HiveExpertDiscovery
        from integrations.agent_engine.model_registry import ModelRegistry
        reg = ModelRegistry()
        disc = HiveExpertDiscovery(registry=reg)
        with patch.object(HiveExpertDiscovery, '_verify_peer_trust', return_value=True), \
                patch.object(HiveExpertDiscovery, '_ping_latency', return_value=12.0):
            n = disc.on_peer_announce({
                'peer_id': 'peer-xyz', 'endpoint': 'https://peer.example',
                'models': [{'model_id': 'big', 'tier': 'expert',
                            'verified_baseline': 0.9}]})
        assert n == 1
        backend = reg.get_model('hive-peer-xyz-big')
        assert backend.config_list_entry['peer_id'] == 'peer-xyz'


# ─── the exits: compute mesh ───

class TestComputeMeshExit:

    def _mesh(self, peer_id=SERVING_NODE):
        from integrations.agent_engine.compute_mesh_service import (
            ComputeMeshService, MeshPeer)
        mesh = ComputeMeshService.__new__(ComputeMeshService)
        import threading
        mesh._lock = threading.Lock()
        mesh._peers = {peer_id: MeshPeer(peer_id, '10.0.0.5', 'k' * 32)}
        mesh._device_id = 'dev-requester'
        mesh.task_relay_port = 6796
        return mesh

    def test_completed_offload_charges_counted_tokens(self, db_factory):
        from core.token_utils import count_tokens_for_text
        _seed(db_factory, requester_spark=100)
        prompt, answer = 'describe ' * 30, 'a cat ' * 40
        with patch('core.http_pool.pooled_post',
                   return_value=_resp(200, {'response': answer, 'model': 'q'})):
            out = self._mesh().offload_inference(
                SERVING_NODE, 'llm', prompt, {'user_id': REQUESTER})
        assert out['response'] == answer
        rows = _rows(db_factory)
        assert [(r[0], r[1], r[3], r[4]) for r in rows] == [
            (REQUESTER, OPERATOR, count_tokens_for_text(prompt),
             count_tokens_for_text(answer))]

    def test_completed_offload_charges_usage_when_the_peer_reports_it(self, db_factory):
        _seed(db_factory, requester_spark=100)
        body = {'response': 'x', 'usage': {'prompt_tokens': _tokens_for_spark(2),
                                           'completion_tokens': 0}}
        with patch('core.http_pool.pooled_post', return_value=_resp(200, body)):
            self._mesh().offload_inference(SERVING_NODE, 'llm', 'p',
                                           {'user_id': REQUESTER})
        assert _spark(db_factory, REQUESTER) == 98

    def test_failed_offload_charges_nothing(self, db_factory):
        _seed(db_factory, requester_spark=100)
        with patch('core.http_pool.pooled_post', return_value=_resp(502, {})):
            out = self._mesh().offload_inference(
                SERVING_NODE, 'llm', 'p' * 4000, {'user_id': REQUESTER})
        assert 'error' in out
        assert _rows(db_factory) == []

    def test_error_body_charges_nothing(self, db_factory):
        _seed(db_factory, requester_spark=100)
        with patch('core.http_pool.pooled_post',
                   return_value=_resp(200, {'error': 'Local inference failed'})):
            self._mesh().offload_inference(
                SERVING_NODE, 'llm', 'p' * 4000, {'user_id': REQUESTER})
        assert _rows(db_factory) == []


# ─── settlement has a scheduled caller ───

class TestScheduledSettlement:

    def test_daemon_tick_settles_pending_compute(self, db_factory, monkeypatch):
        from integrations.agent_engine.agent_daemon import AgentDaemon
        from integrations.agent_engine.budget_gate import charge_remote_compute
        _seed(db_factory, requester_spark=100)
        charge_remote_compute(REQUESTER, SERVING_NODE, _tokens_for_spark(3), 0,
                              source='t')
        monkeypatch.setattr('integrations.agent_engine.dispatch.should_yield_to_user',
                            lambda: False)
        d = AgentDaemon()
        d._tick_count = d._remediate_every - 1   # this tick is a settlement tick
        d._tick()                                 # no goals: returns after settling
        assert _spark(db_factory, OPERATOR) == 3

    def test_off_cadence_tick_does_not_settle(self, db_factory, monkeypatch):
        from integrations.agent_engine.agent_daemon import AgentDaemon
        from integrations.agent_engine.budget_gate import charge_remote_compute
        _seed(db_factory, requester_spark=100)
        charge_remote_compute(REQUESTER, SERVING_NODE, _tokens_for_spark(3), 0,
                              source='t')
        monkeypatch.setattr('integrations.agent_engine.dispatch.should_yield_to_user',
                            lambda: False)
        d = AgentDaemon()
        d._tick_count = 0
        d._tick()
        assert _spark(db_factory, OPERATOR) == 0


# ─── /api/gateway/metering reads columns that exist ───

class TestMeteringByModel:

    def test_groups_real_columns(self, db_factory):
        from integrations.agent_engine.budget_gate import metered_usage_by_model
        with _session(db_factory) as db:
            for m, tin, tout in [('a', 10, 5), ('a', 1, 1), ('b', 7, 0)]:
                db.add(MeteredAPIUsage(node_id='n', model_id=m, task_source='hive',
                                       tokens_in=tin, tokens_out=tout))
        with _session(db_factory) as db:
            got = sorted(metered_usage_by_model(db), key=lambda r: r['provider'])
        assert got == [
            {'provider': 'a', 'total_tokens': 17, 'calls': 2},
            {'provider': 'b', 'total_tokens': 7, 'calls': 1},
        ]
