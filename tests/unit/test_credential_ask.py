"""A credential the agent needs is asked for on the consent card, and the
agent only ever gets its alias back.

Owner requirement (2026-09-25): "when need to ask a credential ... our consent
overlay shd do that ... and the llm shd respond back with the pseudo alias
used so that actual data is encrypted and used only deterministically", and
"the consent card shd have way to enter the pass and then click accept".

Before this, Request_Resource (both copies: the LangChain tool in
hart_intelligence_entry and request_resource in core.agent_tools) returned a
RESOURCE_REQUEST:{json} marker that only the Demopage chat page turned into a
modal.  No consent row, nothing on the floating companion, and the model was
told nothing it could use in place of the value.

Now both copies call ai_key_vault.request_credential, which files ONE
'credential' ask through ConsentService (scope 'secret:<NAME>') and answers
with the {{secret:NAME}} alias.  These tests run the real ConsentService on an
in-memory database, the real vault and the real request_resource closure.
"""
import os
import sys
from unittest.mock import MagicMock

os.environ.setdefault('HEVOLVE_DB_PATH', ':memory:')
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest  # noqa: E402

from integrations.social.models import Base, UserConsent, db_session, get_engine  # noqa: E402

OWNER = 'owner-cred'
SECRET = 'Tr0ub4dor&3-horse'
ASK = '{"key_name": "site_password", "label": "Site password", ' \
      '"used_by": "the login step", "description": "Needed to sign in."}'


@pytest.fixture(autouse=True)
def world(monkeypatch):
    engine = get_engine()
    Base.metadata.create_all(engine)
    monkeypatch.setenv('HEVOLVE_OWNER_USER_ID', OWNER)
    monkeypatch.delenv('HEVOLVE_MASTER_KEY', raising=False)
    monkeypatch.delenv('SITE_PASSWORD', raising=False)
    from security.secrets_manager import SecretsManager
    from hartos import ai_key_vault
    SecretsManager.reset()
    ai_key_vault.AIKeyVault.reset()
    # get_ai_key_vault keeps its own module-level instance beside the class one.
    monkeypatch.setattr(ai_key_vault, '_instance', None)
    emitted = []
    from integrations.social import consent_service
    monkeypatch.setattr(consent_service, '_emit',
                        lambda topic, data, msg_id=None: emitted.append((topic, dict(data))))
    yield emitted
    ai_key_vault.AIKeyVault.reset()
    SecretsManager.reset()
    os.environ.pop('SITE_PASSWORD', None)
    Base.metadata.drop_all(engine)


def _asks():
    with db_session() as db:
        return [(r.consent_type, r.scope, r.agent_id, bool(r.granted))
                for r in db.query(UserConsent).filter_by(user_id=OWNER).all()]


def test_credential_is_a_consent_type():
    from integrations.social.consent_service import CONSENT_TYPES
    assert 'credential' in CONSENT_TYPES


def test_a_missing_credential_is_asked_on_the_card_and_answered_with_the_alias(world):
    from hartos.ai_key_vault import request_credential
    out = request_credential(ASK, agent_id='42')

    assert '{{secret:SITE_PASSWORD}}' in out
    assert 'RESOURCE_REQUEST' not in out, 'no second surface: the card is the ask'
    assert _asks() == [('credential', 'secret:SITE_PASSWORD', '42', False)]
    requests = [d for t, d in world if t == 'consent.request']
    assert len(requests) == 1
    assert requests[0]['scope'] == 'secret:SITE_PASSWORD'
    assert 'Site password' in requests[0]['reason']


def test_asking_again_files_no_second_row(world):
    from hartos.ai_key_vault import request_credential
    request_credential(ASK, agent_id='42')
    request_credential(ASK, agent_id='42')
    assert len(_asks()) == 1


def test_a_stored_credential_is_answered_with_the_alias_never_the_value(world):
    from hartos.ai_key_vault import get_ai_key_vault, request_credential
    get_ai_key_vault().store_credential('site_password', SECRET)

    out = request_credential(ASK, agent_id='42')
    assert SECRET not in out
    assert '{{secret:SITE_PASSWORD}}' in out
    assert _asks() == [], 'nothing to ask for'


def test_a_declined_ask_says_so_and_is_not_asked_again(world):
    from hartos.ai_key_vault import request_credential
    from integrations.social.consent_service import ConsentService
    request_credential(ASK, agent_id='42')
    with db_session(commit=True) as db:
        ConsentService.revoke_consent(db, OWNER, 'credential',
                                      'secret:SITE_PASSWORD', '42')
    world.clear()

    out = request_credential(ASK, agent_id='42')
    assert 'said no' in out
    assert [t for t, _ in world if t == 'consent.request'] == []


@pytest.mark.parametrize('agent', ['42', None])
def test_a_rejected_credential_is_asked_for_again(world, agent):
    """Owner 2026-09-25: "agent shd ask user when login attempts fails".
    The stored value was rejected by the site, so the card comes back even
    though the owner already answered it once (the card's grant writes a
    row for no agent, the same one consent_api writes)."""
    from hartos.ai_key_vault import get_ai_key_vault, request_credential
    from integrations.social.consent_service import ConsentService
    request_credential(ASK, agent_id=agent)
    get_ai_key_vault().store_credential('site_password', SECRET)
    with db_session(commit=True) as db:
        ConsentService.grant_consent(db, OWNER, 'credential', 'secret:SITE_PASSWORD')
    world.clear()

    out = request_credential(ASK[:-1] + ', "rejected": true}', agent_id=agent)
    assert SECRET not in out
    assert '{{secret:SITE_PASSWORD}}' in out
    assert 'Asked the owner' in out
    requests = [d for t, d in world if t == 'consent.request']
    assert len(requests) == 1
    assert requests[0]['scope'] == 'secret:SITE_PASSWORD'
    assert 'rejected' in requests[0]['reason']


def test_the_alias_answer_says_how_to_report_a_rejection(world):
    from hartos.ai_key_vault import get_ai_key_vault, request_credential
    get_ai_key_vault().store_credential('site_password', SECRET)
    assert '"rejected": true' in request_credential(ASK, agent_id='42')


def test_a_rejection_after_the_owner_said_no_is_not_asked_again(world):
    from hartos.ai_key_vault import request_credential
    from integrations.social.consent_service import ConsentService
    request_credential(ASK, agent_id='42')
    with db_session(commit=True) as db:
        ConsentService.revoke_consent(db, OWNER, 'credential',
                                      'secret:SITE_PASSWORD', '42')
    world.clear()
    out = request_credential(ASK[:-1] + ', "rejected": true}', agent_id='42')
    assert 'said no' in out
    assert [t for t, _ in world if t == 'consent.request'] == []


def test_with_no_owner_nothing_is_filed(world, monkeypatch):
    from hartos.ai_key_vault import request_credential
    monkeypatch.delenv('HEVOLVE_OWNER_USER_ID')
    out = request_credential(ASK, agent_id='42')
    assert _asks() == []
    assert '{{secret:' not in out


def test_a_value_the_card_stored_resolves_and_is_masked_once_granted(world, monkeypatch):
    """The card's value lands in os.environ (Nunba /api/vault/store); the
    grant that follows is what makes it the owner's credential.  An env var
    with no grant is not one (tests/unit/test_secret_owner_entered.py)."""
    from hartos.ai_key_vault import get_ai_key_vault
    from integrations.social.consent_service import ConsentService
    monkeypatch.setenv('SITE_PASSWORD', SECRET)
    with db_session(commit=True) as db:
        ConsentService.grant_consent(db, OWNER, 'credential', 'secret:SITE_PASSWORD')
    vault = get_ai_key_vault()
    assert vault.resolve_aliases('pass {{secret:SITE_PASSWORD}}') == f'pass {SECRET}'
    assert vault.mask_secrets(f'echo {SECRET}') == 'echo {{secret:SITE_PASSWORD}}'


def test_the_autogen_request_resource_tool_files_the_same_ask(world):
    from core import agent_tools
    ctx = {
        'user_id': 'someone-remote', 'prompt_id': '42', 'agent_data': {},
        'helper_fun': MagicMock(), 'user_prompt': 's-1', 'request_id_list': [],
        'recent_file_id': {}, 'scheduler': MagicMock(),
        'log_tool_execution': lambda f: f,
        'send_message_to_user1': MagicMock(), 'retrieve_json': lambda v: v,
        'strip_json_values': lambda v: v, 'save_conversation_db': MagicMock(),
    }
    tools = {n: f for n, _d, f in agent_tools.build_core_tool_closures(ctx)}

    out = tools['request_resource'](ASK)
    assert '{{secret:SITE_PASSWORD}}' in out
    assert 'RESOURCE_REQUEST' not in out
    # The machine's owner is asked, not the remote caller.
    assert _asks() == [('credential', 'secret:SITE_PASSWORD', '42', False)]
