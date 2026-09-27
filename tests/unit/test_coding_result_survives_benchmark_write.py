"""A coding result is never thrown away because a benchmark row could not be written.

Measured live (gui_app.log.2, 2026-09-25 21:05:52): execute_coding_task ran
`claude -p` to completion, then BenchmarkTracker.record() raised
"attempt to write a readonly database" because the benchmark DB sat in the
install tree (Program Files).  The exception went up through
orchestrator._execute_local into execute_coding_task's catch-all, which
returned only that string, with nothing logged.  The agent retried 43 times
and the action ended GAVE_UP.

These tests pin three things:
  1. the default DB lives under core.platform_paths.get_agent_data_dir(),
     not beside the source files;
  2. a failed benchmark write never replaces the backend's result, on the
     local path and on the two hive paths (where it used to re-run the task);
     that includes a tracker that cannot be BUILT (its data dir cannot be
     created), since the DB path is now resolved and created at build time;
  3. each record site's fields reach the row unchanged (success and
     completion time order get_best_tool's routing).
execute_coding_task's own logging is pinned in
test_execute_coding_task_logs_its_failure.py.
"""
import logging
import os
import sqlite3
import stat
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _read_only_tracker(tmp_path):
    """A real BenchmarkTracker whose DB file the process cannot write."""
    from integrations.coding_agent.benchmark_tracker import BenchmarkTracker
    db = tmp_path / 'coding_benchmarks.db'
    tracker = BenchmarkTracker(db_path=str(db))
    os.chmod(db, stat.S_IREAD)
    # Prove the boundary is real before relying on it.
    with pytest.raises(sqlite3.OperationalError):
        tracker.record('feature', 'claude_code', 1.0, True)
    return tracker, db


def _backend(output='sum is 2870'):
    backend = MagicMock()
    backend.name = 'claude_code'
    backend.execute.return_value = {
        'success': True, 'output': output, 'tool': 'claude_code',
        'execution_time_s': 20.0,
    }
    return backend


def _rows(db):
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            'SELECT task_type, tool_name, model_name, user_id, '
            'completion_time_s, success, offloaded FROM benchmarks').fetchall()
    finally:
        conn.close()


def _benchmark_warnings(caplog, needle):
    """WARNING records from the benchmark guard that carry a traceback."""
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and r.exc_info
            and 'Benchmark row not recorded' in r.getMessage()
            and needle in r.getMessage()]


def _peer_mocks():
    peer = {'node_id': 'peer-1', 'x25519_public_hex': 'ab' * 32,
            'url': 'http://peer', 'trust_level': 'SAME_USER'}
    mesh = MagicMock()
    mesh.get_available_peers.return_value = [peer]
    mesh.score.return_value = 1.0
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {'encrypted': {'blob': 1}}
    return mesh, resp


@pytest.fixture
def restore_perms():
    paths = []
    yield paths
    for p in paths:
        try:
            os.chmod(p, stat.S_IREAD | stat.S_IWRITE)
        except OSError:
            pass


class TestDefaultDbPath:
    def test_default_db_is_under_the_agent_data_dir(self, tmp_path, monkeypatch):
        import core.platform_paths as pp
        from integrations.coding_agent.benchmark_tracker import BenchmarkTracker
        data_dir = tmp_path / 'user_data' / 'agent_data'   # does not exist yet
        monkeypatch.setattr(pp, 'get_agent_data_dir', lambda: str(data_dir))

        tracker = BenchmarkTracker()
        tracker.record('feature', 'claude_code', 2.0, True)

        db = data_dir / 'coding_benchmarks.db'
        assert db.is_file()
        conn = sqlite3.connect(str(db))
        try:
            rows = conn.execute('SELECT tool_name FROM benchmarks').fetchall()
        finally:
            conn.close()
        assert rows == [('claude_code',)]

    def test_default_is_resolved_when_built_not_at_import(self, tmp_path, monkeypatch):
        import core.platform_paths as pp
        from integrations.coding_agent.benchmark_tracker import BenchmarkTracker
        first = tmp_path / 'a'
        second = tmp_path / 'b'
        monkeypatch.setattr(pp, 'get_agent_data_dir', lambda: str(first))
        BenchmarkTracker().record('feature', 'x', 1.0, True)
        monkeypatch.setattr(pp, 'get_agent_data_dir', lambda: str(second))
        BenchmarkTracker().record('feature', 'x', 1.0, True)
        assert (first / 'coding_benchmarks.db').is_file()
        assert (second / 'coding_benchmarks.db').is_file()


class TestExecuteLocalKeepsResult:
    def test_result_returned_when_benchmark_write_fails(self, tmp_path, caplog, restore_perms):
        tracker, db = _read_only_tracker(tmp_path)
        restore_perms.append(db)
        backend = _backend()
        from integrations.coding_agent.orchestrator import CodingAgentOrchestrator
        with patch('integrations.coding_agent.tool_router.CodingToolRouter.route',
                   return_value=backend), \
             patch('integrations.coding_agent.benchmark_tracker.get_benchmark_tracker',
                   return_value=tracker), \
             caplog.at_level(logging.WARNING, logger='hevolve.coding_agent'):
            result = CodingAgentOrchestrator()._execute_local(
                'sum of squares 1..20', 'feature', '', 'u1', '', '')

        assert result['success'] is True
        assert result['output'] == 'sum is 2870'
        assert result['task_type'] == 'feature'
        backend.execute.assert_called_once()
        warned = _benchmark_warnings(caplog, 'readonly')
        assert warned, [(r.getMessage(), r.exc_info) for r in caplog.records]
        assert isinstance(warned[0].exc_info[1], sqlite3.OperationalError)

    def test_row_still_written_when_db_is_writable(self, tmp_path):
        from integrations.coding_agent.benchmark_tracker import BenchmarkTracker
        tracker = BenchmarkTracker(db_path=str(tmp_path / 'b.db'))
        from integrations.coding_agent.orchestrator import CodingAgentOrchestrator
        with patch('integrations.coding_agent.tool_router.CodingToolRouter.route',
                   return_value=_backend()), \
             patch('integrations.coding_agent.benchmark_tracker.get_benchmark_tracker',
                   return_value=tracker):
            CodingAgentOrchestrator()._execute_local('t', 'bug_fix', '', 'u1', 'm', '')
        assert _rows(tmp_path / 'b.db') == [
            ('bug_fix', 'claude_code', 'm', 'u1', 20.0, 1, 0)]

    def test_failed_run_is_recorded_as_a_failure(self, tmp_path):
        from integrations.coding_agent.benchmark_tracker import BenchmarkTracker
        tracker = BenchmarkTracker(db_path=str(tmp_path / 'b.db'))
        backend = _backend()
        backend.execute.return_value = {
            'success': False, 'output': '', 'tool': 'claude_code',
            'execution_time_s': 7.5}
        from integrations.coding_agent.orchestrator import CodingAgentOrchestrator
        with patch('integrations.coding_agent.tool_router.CodingToolRouter.route',
                   return_value=backend), \
             patch('integrations.coding_agent.benchmark_tracker.get_benchmark_tracker',
                   return_value=tracker):
            CodingAgentOrchestrator()._execute_local('t', 'bug_fix', '', 'u2', '', '')
        assert _rows(tmp_path / 'b.db') == [
            ('bug_fix', 'claude_code', '', 'u2', 7.5, 0, 0)]


class TestTrackerThatCannotBeBuilt:
    """The DB dir is created when the tracker is built, so building can fail."""

    def test_result_returned_when_data_dir_cannot_be_created(
            self, tmp_path, monkeypatch, caplog):
        import core.platform_paths as pp
        import integrations.coding_agent.benchmark_tracker as bt
        blocker = tmp_path / 'not_a_dir'
        blocker.write_text('a file where the data dir should be')
        monkeypatch.setattr(pp, 'get_agent_data_dir',
                            lambda: str(blocker / 'agent_data'))
        monkeypatch.setattr(bt, '_tracker', None)   # force a fresh build
        # Prove the boundary: building the tracker really raises, and it is
        # an OSError from makedirs, not a sqlite error.
        with pytest.raises(OSError):
            bt.BenchmarkTracker()

        backend = _backend()
        from integrations.coding_agent.orchestrator import CodingAgentOrchestrator
        with patch('integrations.coding_agent.tool_router.CodingToolRouter.route',
                   return_value=backend), \
             caplog.at_level(logging.WARNING, logger='hevolve.coding_agent'):
            result = CodingAgentOrchestrator()._execute_local(
                'sum of squares 1..20', 'feature', '', 'u1', '', '')

        assert result['success'] is True
        assert result['output'] == 'sum is 2870'
        backend.execute.assert_called_once()
        warned = _benchmark_warnings(caplog, 'feature/claude_code')
        assert warned, [(r.getMessage(), r.exc_info) for r in caplog.records]
        assert isinstance(warned[0].exc_info[1], OSError)
        assert bt._tracker is None   # nothing half-built was cached


class TestOffloadKeepsPeerResult:
    def test_peer_result_returned_not_rerun_locally(self, tmp_path, caplog, restore_perms):
        tracker, db = _read_only_tracker(tmp_path)
        restore_perms.append(db)
        peer = {'node_id': 'peer-1', 'x25519_public_hex': 'ab' * 32,
                'url': 'http://peer', 'trust_level': 'SAME_USER'}
        mesh = MagicMock()
        mesh.get_available_peers.return_value = [peer]
        mesh.score.return_value = 1.0
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {'encrypted': {'blob': 1}}
        peer_result = {'success': True, 'output': 'peer did it',
                       'tool': 'claude_code', 'execution_time_s': 3.0}

        from integrations.coding_agent.orchestrator import CodingAgentOrchestrator
        orch = CodingAgentOrchestrator()
        with patch('integrations.agent_engine.compute_mesh_service.get_compute_mesh',
                   return_value=mesh), \
             patch('security.channel_encryption.encrypt_json_for_peer',
                   return_value={'env': 1}), \
             patch('security.channel_encryption.decrypt_json_from_peer',
                   return_value=dict(peer_result)), \
             patch('core.http_pool.pooled_post', return_value=resp), \
             patch('integrations.coding_agent.benchmark_tracker.get_benchmark_tracker',
                   return_value=tracker), \
             patch.object(CodingAgentOrchestrator, '_record_peer_trust'), \
             patch.object(CodingAgentOrchestrator, '_execute_local') as local, \
             caplog.at_level(logging.WARNING, logger='hevolve.coding_agent'):
            result = orch._offload_to_hive('t', 'feature', '', 'u1', '', '')

        local.assert_not_called()
        assert result['output'] == 'peer did it'
        assert result['offloaded'] is True
        assert result['peer_id'] == 'peer-1'
        assert _benchmark_warnings(caplog, 'readonly'), \
            [(r.getMessage(), r.exc_info) for r in caplog.records]

    def test_offloaded_row_carries_the_peer_fields(self, tmp_path):
        from integrations.coding_agent.benchmark_tracker import BenchmarkTracker
        tracker = BenchmarkTracker(db_path=str(tmp_path / 'b.db'))
        mesh, resp = _peer_mocks()
        peer_result = {'success': True, 'output': 'peer did it',
                       'tool': 'aider', 'execution_time_s': 3.0}
        from integrations.coding_agent.orchestrator import CodingAgentOrchestrator
        with patch('integrations.agent_engine.compute_mesh_service.get_compute_mesh',
                   return_value=mesh), \
             patch('security.channel_encryption.encrypt_json_for_peer',
                   return_value={'env': 1}), \
             patch('security.channel_encryption.decrypt_json_from_peer',
                   return_value=dict(peer_result)), \
             patch('core.http_pool.pooled_post', return_value=resp), \
             patch('integrations.coding_agent.benchmark_tracker.get_benchmark_tracker',
                   return_value=tracker), \
             patch.object(CodingAgentOrchestrator, '_record_peer_trust'), \
             patch.object(CodingAgentOrchestrator, '_execute_local') as local:
            CodingAgentOrchestrator()._offload_to_hive(
                't', 'refactor', '', 'u3', 'm3', '')
        local.assert_not_called()
        assert _rows(tmp_path / 'b.db') == [
            ('refactor', 'aider', 'm3', 'u3', 3.0, 1, 1)]


class TestDistributeKeepsMergedResult:
    def test_merged_result_returned_not_reoffloaded(self, tmp_path, caplog, restore_perms):
        tracker, db = _read_only_tracker(tmp_path)
        restore_perms.append(db)
        shard = MagicMock()
        shard.task_description = 'shard task'
        shard.full_content = {'a.py': 'x = 1'}
        shard.interface_specs = []
        shard.scope.value = 'full_file'
        shard.target_files = ['a.py']
        engine = MagicMock()
        engine.decompose_task.return_value = [shard]
        guard = MagicMock()
        guard.check_egress.return_value = (True, '')
        peer = {'node_id': 'peer-1', 'x25519_public_hex': 'ab' * 32,
                'url': 'http://peer', 'trust_level': 'SAME_USER'}
        mesh = MagicMock()
        mesh.get_available_peers.return_value = [peer]
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {'encrypted': {'blob': 1}}

        from integrations.coding_agent.orchestrator import CodingAgentOrchestrator
        orch = CodingAgentOrchestrator()
        with patch('integrations.agent_engine.shard_engine.ShardEngine',
                   return_value=engine), \
             patch('security.edge_privacy.ScopeGuard', return_value=guard), \
             patch('integrations.agent_engine.compute_mesh_service.get_compute_mesh',
                   return_value=mesh), \
             patch('security.channel_encryption.encrypt_json_for_peer',
                   return_value={'env': 1}), \
             patch('security.channel_encryption.decrypt_json_from_peer',
                   return_value={'success': True, 'output': 'shard done',
                                 'diffs': {'a.py': '+y'}}), \
             patch('security.channel_encryption.get_x25519_public_hex',
                   return_value='cd' * 32), \
             patch('core.http_pool.pooled_post', return_value=resp), \
             patch('integrations.coding_agent.benchmark_tracker.get_benchmark_tracker',
                   return_value=tracker), \
             patch.object(CodingAgentOrchestrator, '_record_peer_trust'), \
             patch.object(CodingAgentOrchestrator, '_offload_to_hive') as offload, \
             patch.object(CodingAgentOrchestrator, '_execute_local') as local, \
             caplog.at_level(logging.WARNING, logger='hevolve.coding_agent'):
            result = orch._distribute_to_hive('t', 'feature', '', 'u1', '',
                                              str(tmp_path), 'trusted_peer')

        offload.assert_not_called()
        local.assert_not_called()
        assert result['tool'] == 'distributed'
        assert result['output'] == 'shard done'
        assert result['diffs'] == {'a.py': '+y'}
        assert _benchmark_warnings(caplog, 'readonly'), \
            [(r.getMessage(), r.exc_info) for r in caplog.records]
