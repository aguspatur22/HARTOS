"""A notification's live push goes out only for a row that committed.

NotificationService.create / mark_read / mark_all_read / mark_dismissed each
used to hang an ``event.listen(db, 'after_commit', fn, once=True)`` on the
session.  A once-listener stays on the session until SOME commit fires it, so
after a rollback (or a close without a commit) the next, unrelated commit of
the same session pushed a notification whose row was never written.  Measured
at 66a0648ee: create, rollback, commit -> 0 rows on disk, 1 push.

They now go through the one after-commit queue beside db_session
(integrations.social.models.after_commit), which the consent grant uses too
and which drops its queue when the outer transaction ends uncommitted.

    python -m pytest tests/unit/test_notification_push_after_commit.py -q
"""
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture
def factory(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from integrations.social.models import Notification

    engine = create_engine(f"sqlite:///{tmp_path / 'notif.db'}")
    Notification.__table__.create(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def pushed(monkeypatch):
    """What reached the live pipe (realtime is the boundary)."""
    import integrations.social.realtime as rt

    sent = []
    monkeypatch.setattr(rt, 'on_notification',
                        lambda uid, d: sent.append(('new', uid, d['id'])))
    monkeypatch.setattr(rt, 'on_notification_read',
                        lambda uid, ids: sent.append(('read', uid, list(ids))))
    return sent


def _rows(db):
    from integrations.social.models import Notification
    return db.query(Notification).count()


def test_a_committed_notification_is_pushed_once(factory, pushed):
    from integrations.social.services import NotificationService

    db = factory()
    try:
        n = NotificationService.create(db, 'u1', 'comment', message='hi')
        assert pushed == [], 'pushed before the row committed'
        db.commit()
        db.commit()
    finally:
        db.close()

    assert pushed == [('new', 'u1', n.id)]


def test_a_rolled_back_notification_is_never_pushed(factory, pushed):
    from integrations.social.services import NotificationService

    db = factory()
    try:
        NotificationService.create(db, 'u1', 'comment', message='hi')
        db.rollback()
        db.commit()
        assert _rows(db) == 0
    finally:
        db.close()

    assert pushed == []


def test_a_notification_closed_without_commit_is_never_pushed(factory, pushed):
    from integrations.social.services import NotificationService

    db = factory()
    NotificationService.create(db, 'u1', 'comment', message='hi')
    db.close()
    try:
        db.commit()
    finally:
        db.close()

    assert pushed == []


def test_read_and_dismiss_fan_out_only_after_commit(factory, pushed):
    from integrations.social.services import NotificationService

    db = factory()
    try:
        a = NotificationService.create(db, 'u1', 'comment', message='a')
        b = NotificationService.create(db, 'u1', 'comment', message='b')
        db.commit()
        pushed.clear()

        NotificationService.mark_read(db, [a.id], 'u1')
        NotificationService.mark_dismissed(db, [b.id], 'u1')
        db.rollback()
        db.commit()
        assert pushed == [], 'a rolled-back read state was fanned out'

        NotificationService.mark_read(db, [a.id], 'u1')
        NotificationService.mark_dismissed(db, [b.id], 'u1')
        assert pushed == []
        db.commit()
        assert pushed == [('read', 'u1', [a.id]), ('read', 'u1', [b.id])]

        pushed.clear()
        c = NotificationService.create(db, 'u1', 'comment', message='c')
        db.commit()
        pushed.clear()
        NotificationService.mark_all_read(db, 'u1')
        db.rollback()
        db.commit()
        assert pushed == []
        NotificationService.mark_all_read(db, 'u1')
        db.commit()
        # b was dismissed, not read, so it is still unread here.
        assert [(k, u, sorted(ids)) for k, u, ids in pushed] == [
            ('read', 'u1', sorted([b.id, c.id]))]
    finally:
        db.close()
