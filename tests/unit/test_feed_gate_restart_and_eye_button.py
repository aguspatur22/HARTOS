"""The camera/screen gate after a restart, and the eye button's camera cut.

1. core.ai_sensing._stop_vision imported get_vision_service from
   integrations.vision.vision_service, which does not define it.  The import
   raised inside a bare `except: pass`, so the eye button's camera cut never
   stopped a VisionService, and status()'s camera_service_running proof was
   always False.  Now every running VisionService stops, whichever of its
   three owners made it (vision_service.running_vision_services).

2. The capture gate is process memory and started open on every boot, so
   after a restart camera frames reached the store until the owner answered
   again.  Now the first VisionService constructed restores the owner's
   saved answers (vision_service.restore_feed_answers): a standing No closes
   the feed, a read error closes both, and an answer this process already
   holds is never overwritten by the (possibly uncommitted) row.

Real VisionService, real FrameStore, real ConsentService on a file SQLite.

    python -m pytest tests/unit/test_feed_gate_restart_and_eye_button.py -q
"""
import contextlib
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OWNER = 'owner-restart'


@pytest.fixture
def gate():
    """The gate as a fresh process has it: no answers held."""
    from core import ai_sensing
    with ai_sensing._lock:
        for sensor in ai_sensing._withheld:
            ai_sensing._withheld[sensor] = False
        getattr(ai_sensing, '_answered', set()).clear()
    yield ai_sensing
    ai_sensing.set_sense('camera', False)
    ai_sensing.set_sense('screen', False)
    with ai_sensing._lock:
        for sensor in ai_sensing._withheld:
            ai_sensing._withheld[sensor] = False
        getattr(ai_sensing, '_answered', set()).clear()


@pytest.fixture
def saved_consent(tmp_path, monkeypatch):
    """The owner's consent rows on a real file DB, as the restore reads them."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import integrations.social.consent_service as cs
    import integrations.social.models as models
    from integrations.social.models import UserConsent

    engine = create_engine(f"sqlite:///{tmp_path / 'saved.db'}")
    UserConsent.__table__.create(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(cs, '_emit', lambda *a, **k: None)
    monkeypatch.setattr(cs, '_copilot_switch_from_consent', lambda *a, **k: None)
    # Writing the rows must not drive the feed: this is the state on disk
    # from a previous run.
    monkeypatch.setattr(cs, '_embodied_feed_from_consent', lambda *a, **k: None)

    @contextlib.contextmanager
    def _session(commit=True):
        s = factory()
        try:
            yield s
            if commit:
                s.commit()
        finally:
            s.close()

    monkeypatch.setattr(models, 'db_session', _session)
    monkeypatch.setenv('HEVOLVE_OWNER_USER_ID', OWNER)

    def _answer(consent_type, granted):
        from integrations.social.consent_service import ConsentService
        with _session() as s:
            if granted:
                ConsentService.grant_consent(s, OWNER, consent_type)
            else:
                ConsentService.revoke_consent(s, OWNER, consent_type)

    yield _answer
    engine.dispose()


def _new_service():
    from integrations.vision.frame_store import FrameStore
    from integrations.vision.vision_service import VisionService
    return VisionService(frame_store=FrameStore())


# ── 1. the eye button stops the camera ──────────────────────────────────────

def test_the_eye_buttons_camera_cut_stops_the_running_service(gate):
    vs = _new_service()
    vs._running = True          # stands in for start()'s hardware
    try:
        assert gate.status()['proof']['camera_service_running'] is True
        gate.set_sense('camera', True)
        assert vs.is_running() is False, 'the camera cut left VisionService running'
        assert gate.status()['proof']['camera_service_running'] is False
    finally:
        vs._running = False


def test_disable_all_stops_every_running_service(gate):
    """Nunba's boot instance and the integrations.vision singleton can both
    exist: the cut stops each one that runs."""
    a, b = _new_service(), _new_service()
    a._running = b._running = True
    try:
        gate.disable_all()
        assert (a.is_running(), b.is_running()) == (False, False)
    finally:
        gate.enable_all()
        a._running = b._running = False


# ── 2. a restart keeps the owner's No ───────────────────────────────────────

def test_a_saved_no_closes_the_feed_before_any_frame(gate, saved_consent):
    saved_consent('camera_capture', True)
    saved_consent('camera_capture', False)
    saved_consent('screen_capture', True)

    vs = _new_service()           # the boot: first VisionService
    vs.store.put_frame(OWNER, b'after-restart')
    assert vs.store.get_frame(OWNER) is None, (
        'a camera frame reached the store after a restart despite the saved No')
    assert gate.allowed('camera') is False
    assert gate.allowed('screen') is True
    vs.store.put_screen_frame(OWNER, b'screen')
    assert vs.store.get_screen_frame(OWNER) == b'screen'


def test_a_yes_given_after_a_no_is_what_the_restart_keeps(gate, saved_consent):
    """The revoked row stays on file as history; the later grant is the
    answer."""
    saved_consent('camera_capture', True)
    saved_consent('camera_capture', False)
    saved_consent('camera_capture', True)
    _new_service()
    assert gate.allowed('camera') is True


def test_no_answer_on_file_leaves_the_feeds_open(gate, saved_consent):
    vs = _new_service()
    vs.store.put_frame(OWNER, b'frame')
    assert vs.store.get_frame(OWNER) == b'frame'
    assert (gate.allowed('camera'), gate.allowed('screen')) == (True, True)


def test_a_consent_that_cannot_be_read_closes_both(gate, saved_consent,
                                                   monkeypatch):
    import integrations.social.models as models

    def _broken(commit=True):
        raise RuntimeError('database is locked')

    monkeypatch.setattr(models, 'db_session', _broken)
    _new_service()
    assert (gate.allowed('camera'), gate.allowed('screen')) == (False, False)


def test_an_answer_given_in_this_process_is_not_overwritten(gate, saved_consent):
    """The worker may construct the first VisionService while a No is still
    uncommitted, so the row reads Yes: the No must stand."""
    saved_consent('camera_capture', True)
    gate.withhold('camera', True)
    _new_service()
    assert gate.allowed('camera') is False


def test_the_restore_reads_once(gate, saved_consent):
    """A later VisionService does not re-read: a Yes given since the boot
    stands even though the row said No at boot."""
    saved_consent('camera_capture', True)
    saved_consent('camera_capture', False)
    _new_service()
    assert gate.allowed('camera') is False
    gate.withhold('camera', False)
    _new_service()
    assert gate.allowed('camera') is True


# ── source guard: one writer each ───────────────────────────────────────────

def test_source_guard_one_restore_writer():
    from tests.unit.test_feed_no_takes_effect_while_the_pool_is_busy import (
        _callers)
    found = _callers({'restore_withheld'})
    assert found and {(f, o) for f, o, _ in found} == {
        ('integrations/vision/vision_service.py', 'restore_feed_answers')}, found
    restore = _callers({'restore_feed_answers'})
    assert {(f, o) for f, o, _ in restore} == {
        ('integrations/vision/vision_service.py', 'VisionService')}, restore
