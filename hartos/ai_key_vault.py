"""
AIKeyVault — Thin coordination layer for agent credential management.

PRIVACY RULE: Secrets NEVER leave the user's device.
  - Stored only on the local device (encrypted vault at secrets.enc)
  - NEVER transmitted over network, federation, gossip, PeerLink, or WAMP
  - NEVER included in hive deltas, task payloads, or agent responses
  - Credential endpoints accept connections from localhost ONLY
  - Hive/idle tasks get API keys STRIPPED (see tool_backends.py)
  - Output is redacted by secret_redactor.py (3-layer defense)
  - Even trusted hive nodes never receive the actual secret value

The user's secrets belong to the user, on the user's device, period.

Delegates ALL storage to SecretsManager (security/secrets_manager.py).
Adds:
  - Channel-specific key namespacing (discord + BOT_TOKEN → DISCORD_BOT_TOKEN)
  - Pending credential request tracking (what agents are waiting for)
  - Boot-time env preloading (config_cache.py contract)
  - store_credential() that persists + injects into os.environ
  - is_local_request() gate for credential endpoints

Usage:
    from hartos.ai_key_vault import AIKeyVault
    vault = AIKeyVault.get_instance()
    key = vault.get_tool_key('OPENAI_API_KEY')
    token = vault.get_channel_secret('discord', 'BOT_TOKEN')
"""

import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

logger = logging.getLogger('hevolve_security')

#: How a stored credential appears to the model: {{secret:NAME}}.  The model
#: only ever sees and writes this alias; resolve_aliases puts the real value
#: in where a tool runs and mask_secrets turns it back into the alias on the
#: way out, so the value never enters the model's context, logs or chat.
SECRET_ALIAS_RE = re.compile(r'\{\{secret:([A-Z0-9_]+)\}\}')

#: A stored value shorter than this is not masked: masking every occurrence
#: of "abc" in tool output would mangle ordinary text.
MIN_MASKED_SECRET_LEN = 4

#: Longest credential name: 'secret:' + the name is a consent scope, and
#: UserConsent.scope holds 100 characters.
MAX_CREDENTIAL_NAME_LEN = 90

#: A rejected login asks the owner again (7f3d5468f), but only this many
#: values per credential within CREDENTIAL_ENTRY_WINDOW_S.  After that the
#: agent is told to stop and tell the user: three wrong entries in a day is
#: a problem for the person, not something another card will fix.
MAX_CREDENTIAL_ENTRIES = 3
CREDENTIAL_ENTRY_WINDOW_S = 24 * 3600

# ═══════════════════════════════════════════════════════════════════════
# Pending credential request tracking
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PendingCredentialRequest:
    """A credential an agent is blocked waiting for."""
    request_id: str
    key_name: str
    resource_type: str   # api_key | channel_secret | token | config
    channel_type: str
    label: str
    description: str
    used_by: str
    requested_at: float


# ═══════════════════════════════════════════════════════════════════════
# AIKeyVault
# ═══════════════════════════════════════════════════════════════════════

class AIKeyVault:
    """Agent credential vault — delegates storage to SecretsManager."""

    _instance: Optional['AIKeyVault'] = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._pending: Dict[str, PendingCredentialRequest] = {}
        self._lock = threading.Lock()
        self._sm = None  # Lazy — loaded on first use
        # Names stored through store_credential this process.  With no
        # HEVOLVE_MASTER_KEY the value lives only in os.environ, so the
        # vault cache alone cannot say which env vars are user credentials.
        self._stored: set = set()

    @classmethod
    def get_instance(cls) -> 'AIKeyVault':
        """Thread-safe singleton (matches hart_intelligence:1796 call)."""
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Clear singleton (testing)."""
        cls._instance = None

    # ── Internal ───────────────────────────────────────────────────

    def _secrets_manager(self):
        """Lazy import to avoid circular imports at boot."""
        if self._sm is None:
            from security.secrets_manager import SecretsManager
            self._sm = SecretsManager.get_instance()
        return self._sm

    @staticmethod
    def _resolve_channel_key(channel_type: str, key_name: str) -> str:
        """Resolve channel + key into env-var name.

        ('discord', 'BOT_TOKEN') → 'DISCORD_BOT_TOKEN'
        ('', 'API_KEY')          → 'API_KEY'
        ('slack', 'SLACK_TOKEN') → 'SLACK_TOKEN'  (no double prefix)
        """
        key_upper = key_name.upper()
        if not channel_type:
            return key_upper
        prefix = channel_type.upper() + '_'
        if key_upper.startswith(prefix):
            return key_upper
        return prefix + key_upper

    # ── Retrieval ──────────────────────────────────────────────────

    def get_tool_key(self, key_name: str) -> str:
        """Get a tool/API key. Delegates to SecretsManager."""
        return self._secrets_manager().get_secret(key_name)

    def get_channel_secret(self, channel_type: str, key_name: str) -> str:
        """Get a channel-specific secret.

        Resolves name then delegates to SecretsManager.
        """
        resolved = self._resolve_channel_key(channel_type, key_name)
        return self._secrets_manager().get_secret(resolved)

    # ── Storage ────────────────────────────────────────────────────

    def store_credential(self, key_name: str, value: str,
                         channel_type: str = '') -> str:
        """Store a credential in vault + inject into os.environ.

        Returns the resolved key name.
        """
        resolved = self._resolve_channel_key(channel_type, key_name) \
            if channel_type else key_name.upper()

        with self._lock:
            # Persist to encrypted vault
            try:
                self._secrets_manager().set_secret(resolved, value)
            except RuntimeError:
                # HEVOLVE_MASTER_KEY not set — env-only fallback
                logger.warning(
                    f"Vault unavailable, storing {resolved} in env only "
                    "(will not persist across restarts)"
                )

            # Inject into current process
            os.environ[resolved] = value
            self._stored.add(resolved)

            # Clear pending request
            self._pending.pop(resolved, None)

        # Audit log (key name only, never value)
        try:
            from security.immutable_audit_log import get_audit_log
            get_audit_log().log_event(
                'credential_stored',
                actor_id='ai_key_vault',
                action=f'stored credential {resolved}',
            )
        except Exception:
            pass

        logger.info(f"Credential stored: {resolved}")
        return resolved

    # ── Alias (the only form the model sees) ───────────────────────

    def credential_name(self, key_name: str, channel_type: str = '') -> str:
        """The one name a credential goes by: its vault/env key, its alias and
        its consent scope.  Anything outside [A-Z0-9_] becomes '_' so the
        alias always matches SECRET_ALIAS_RE."""
        resolved = self._resolve_channel_key(channel_type, key_name) \
            if channel_type else str(key_name).upper()
        return re.sub(r'[^A-Z0-9_]', '_', resolved)[:MAX_CREDENTIAL_NAME_LEN]

    def alias_for(self, key_name: str, channel_type: str = '') -> str:
        """The {{secret:NAME}} alias the model uses for a stored credential."""
        return '{{secret:' + self.credential_name(key_name, channel_type) + '}}'

    @classmethod
    def _map_strings(cls, value, fn):
        """Apply ``fn`` to every string inside lists, tuples and dicts."""
        if isinstance(value, str):
            return fn(value)
        if isinstance(value, dict):
            return {k: cls._map_strings(v, fn) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cls._map_strings(v, fn) for v in value)
        return value

    def owner_credential_names(self) -> set:
        """The names of the credentials the owner of this computer entered:
        the only names an alias resolves and "is stored" answers for.

        - what store_credential stored this process (/api/credentials/submit,
          localhost only), and
        - what the owner gave on the consent card: a granted, unrevoked
          'credential' row with scope 'secret:NAME' for this computer's owner
          (HEVOLVE_OWNER_USER_ID).  The card puts the value in Nunba's
          desktop vault (/api/vault/store, which exports it to os.environ)
          and only then grants the row.  The row is the durable,
          cross-process record of "the owner entered NAME": it survives a
          restart, the privacy page lists it, and revoking it stops the
          alias resolving.  The value stores stay where they are; this is
          the one answer to WHICH names are the owner's.

        Never "any environment variable": os.environ also holds API keys,
        database URLs and key material nobody entered for an agent, and an
        alias that reached them let a model (or a page it read) put them in
        a tool URL.  A consent lookup that fails counts no grants.
        """
        names = set(self._stored)
        owner = os.environ.get('HEVOLVE_OWNER_USER_ID')
        if not owner:
            return names
        try:
            from integrations.social.consent_service import (
                CREDENTIAL_SCOPE_PREFIX, ConsentService)
            from integrations.social.models import db_session
            with db_session(commit=False) as db:
                for row in ConsentService.list_consents(db, owner, 'credential'):
                    scope = row.scope or ''
                    if (row.granted and row.revoked_at is None
                            and scope.startswith(CREDENTIAL_SCOPE_PREFIX)):
                        names.add(scope[len(CREDENTIAL_SCOPE_PREFIX):])
        except Exception:
            logger.warning("credential grants could not be read; only "
                           "credentials stored this session resolve",
                           exc_info=True)
        return names

    def owner_credential(self, name: str, names=None) -> str:
        """The value of an owner-entered credential, '' for any other name.

        ``names`` is owner_credential_names(), passed by a caller that asks
        about several names at once.
        """
        if name not in (self.owner_credential_names() if names is None
                        else names):
            return ''
        return self.get_tool_key(name)

    def resolve_aliases(self, value):
        """Replace every alias of an owner-entered credential in ``value``
        with the real value.

        Any other alias (a missing credential, or a name that is only an
        environment variable) is left as written, never as an empty string.
        """
        names = None

        def _real(match):
            nonlocal names
            if names is None:
                names = self.owner_credential_names()
            real = self.owner_credential(match.group(1), names)
            return real or match.group(0)
        return self._map_strings(value, lambda s: SECRET_ALIAS_RE.sub(_real, s))

    def mask_secrets(self, value):
        """Replace every credential value in ``value`` with its alias.

        Covers every value resolve_aliases can hand a tool (the owner-entered
        credentials) and the encrypted vault's own.  A value is matched both
        as written and as json.dumps spells it (a quote, backslash or
        non-ASCII character is escaped there), longest first so a credential
        that contains another is masked whole.
        """
        names = self.owner_credential_names() | set(self._secrets_manager()._cache)
        pairs = []
        for name in names:
            real = self.get_tool_key(name)
            if real and len(real) >= MIN_MASKED_SECRET_LEN:
                alias = '{{secret:' + name + '}}'
                pairs.append((real, alias))
                escaped = json.dumps(real)[1:-1]
                if escaped != real:
                    pairs.append((escaped, alias))
        pairs.sort(key=lambda p: -len(p[0]))

        def _mask(text):
            for real, alias in pairs:
                text = text.replace(real, alias)
            return text
        return self._map_strings(value, _mask) if pairs else value

    # ── Boot Preload ───────────────────────────────────────────────

    def preload_env(self) -> int:
        """Load all vault secrets into os.environ.

        Called at boot BEFORE config_cache runs.
        Skips keys already present in os.environ.
        Returns count of keys loaded.
        """
        sm = self._secrets_manager()
        loaded = 0

        # Load from encrypted vault cache
        for key, value in sm._cache.items():
            if key not in os.environ and value:
                os.environ[key] = value
                loaded += 1

        # Bundled mode: also check langchain_config.json next to exe
        if getattr(sys, 'frozen', False):
            exe_dir = os.path.dirname(sys.executable)
            config_path = os.path.join(exe_dir, 'langchain_config.json')
            if os.path.exists(config_path):
                try:
                    import json
                    with open(config_path, 'r') as f:
                        bundled = json.load(f)
                    for key, value in bundled.items():
                        if isinstance(value, str) and key not in os.environ and value:
                            os.environ[key] = value
                            loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load bundled config: {e}")

        if loaded:
            logger.info(f"AIKeyVault preloaded {loaded} secrets into env")
        return loaded

    # ── Pending Request Tracking ───────────────────────────────────

    def add_pending_request(self, key_name: str,
                            resource_type: str = 'api_key',
                            channel_type: str = '',
                            label: str = '',
                            description: str = '',
                            used_by: str = '') -> str:
        """Register a credential as needed-but-missing.

        Deduplicates by resolved key name. Returns request_id.
        """
        resolved = self._resolve_channel_key(channel_type, key_name) \
            if channel_type else key_name.upper()

        with self._lock:
            existing = self._pending.get(resolved)
            if existing:
                existing.requested_at = time.time()
                return existing.request_id

            req = PendingCredentialRequest(
                request_id=uuid.uuid4().hex,
                key_name=resolved,
                resource_type=resource_type,
                channel_type=channel_type,
                label=label or resolved,
                description=description,
                used_by=used_by,
                requested_at=time.time(),
            )
            self._pending[resolved] = req
            return req.request_id

    def get_pending_requests(self) -> List[dict]:
        """Return all pending credential requests as dicts."""
        with self._lock:
            return [asdict(r) for r in self._pending.values()]

    def clear_pending(self, key_name: str):
        """Remove a pending request by key name."""
        resolved = key_name.upper()
        with self._lock:
            self._pending.pop(resolved, None)

    def has_pending(self, key_name: str) -> bool:
        """Check if a key has a pending request."""
        resolved = key_name.upper()
        with self._lock:
            return resolved in self._pending


# ── Module-level singleton (HARTOS convention) ─────────────────────

_instance: Optional[AIKeyVault] = None
_lock = threading.Lock()


def get_ai_key_vault() -> AIKeyVault:
    """Module-level singleton accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AIKeyVault.get_instance()
    return _instance


# ── Asking the owner for a credential ──────────────────────────────

def _parse_resource_request(text) -> dict:
    """The tool's input: JSON {key_name, label, used_by, description,
    channel_type}, or plain text describing what is needed."""
    import json
    try:
        req = json.loads(text)
    except (ValueError, TypeError):
        req = None
    if isinstance(req, dict):
        return req
    text = str(text or '')
    return {'label': text[:100], 'description': text}


def _recent_entries(consent_service, db, owner, scope) -> int:
    """How many values the owner typed for this credential in the last
    CREDENTIAL_ENTRY_WINDOW_S: the card's Accept writes one granted
    'credential' row per value (consent_api.grant_consent, append-only), so
    those rows are the count; no second counter is kept."""
    from datetime import datetime, timedelta
    since = datetime.utcnow() - timedelta(seconds=CREDENTIAL_ENTRY_WINDOW_S)
    return sum(1 for row in consent_service.list_consents(db, owner, 'credential')
               if row.scope == scope and row.granted_at is not None
               and row.granted_at >= since)


def request_credential(resource_description, agent_id=None) -> str:
    """What Request_Resource (both the LangChain tool and core.agent_tools
    request_resource) does: get the agent a credential without the agent
    ever seeing it.

    A stored credential is answered with its {{secret:NAME}} alias.  A
    missing one is asked for through ConsentService, like every other ask:
    one 'credential' row per credential (scope 'secret:NAME') for the owner
    of this computer, whose vault it goes into, and a consent.request the
    consent card shows with a password field.  Accept stores the value and
    grants the row.  The agent is told the alias either way; resolve_aliases
    puts the value in where a tool runs.
    """
    req = _parse_resource_request(resource_description)
    label = str(req.get('label') or req.get('key_name') or 'a credential')[:100]
    used_by = str(req.get('used_by') or 'a tool')
    vault = get_ai_key_vault()
    name = vault.credential_name(req.get('key_name') or label,
                                 req.get('channel_type') or '')
    alias = '{{secret:' + name + '}}'
    use = (f"use {alias} wherever the value is needed: it is filled in only "
           f"when a tool runs, and you never see it. Do not ask for it in chat. "
           f"If signing in with it fails, call this tool again with the same "
           f'key_name and "rejected": true, and the owner is asked for it again.')
    # The site refused the stored value: never hand it back, ask again.
    rejected = req.get('rejected') is True

    # Only an owner-entered credential is "stored": answering for any
    # environment variable told the agent which ones exist.
    if vault.owner_credential(name) and not rejected:
        return f"'{label}' is stored on this computer. To use it, {use}"

    # The pending list behind /api/credentials/pending.
    vault.add_pending_request(
        key_name=name, resource_type=req.get('resource_type') or 'api_key',
        label=label, description=str(req.get('description') or ''),
        used_by=used_by)

    # Whose vault the value goes into: this computer's owner, as for every
    # other ask of this machine (vlm.safety, capability_setup).
    owner = os.environ.get('HEVOLVE_OWNER_USER_ID')
    if not owner:
        logger.warning("credential %s not asked: HEVOLVE_OWNER_USER_ID is not set",
                       name)
        return (f"Could not ask for '{label}': nobody is signed in on this "
                f"computer who could provide it.")

    reason = ' '.join(filter(None, [
        (f"{label} was rejected when {used_by} tried it; enter it again."
         if rejected else f"{label} is needed for {used_by}."),
        str(req.get('description') or '').strip(),
        f"It is kept encrypted on this computer; the agent only sees {alias}.",
    ]))
    try:
        from integrations.social.consent_service import (
            CREDENTIAL_SCOPE_PREFIX, ConsentService, known_agent_id)
        from integrations.social.models import db_session
        agent = known_agent_id(agent_id)
        scope = CREDENTIAL_SCOPE_PREFIX + name
        # Only reached with no usable value (missing, or rejected), so an
        # earlier Accept does not settle it: ask again unless the owner said
        # no.  request_consent keeps one row, so the card is still one card.
        with db_session(commit=True) as db:
            declined = ConsentService.declined(db, owner, 'credential',
                                               scope=scope, agent_id=agent)
            # A rejection re-asks, but not for ever: every value the owner
            # typed is a grant row, so the rows say how often they tried.
            entries = (_recent_entries(ConsentService, db, owner, scope)
                       if rejected and not declined else 0)
            exhausted = entries >= MAX_CREDENTIAL_ENTRIES
            if not declined and not exhausted:
                ConsentService.request_consent(db, owner, 'credential',
                                               scope=scope, agent_id=agent,
                                               reason=reason, reask=True)
    except Exception:
        logger.exception("credential %s could not be asked for", name)
        return (f"Could not ask for '{label}': the permission system is "
                f"unavailable.")

    if declined:
        return f"The owner of this computer said no to providing '{label}'."
    if exhausted:
        logger.info("credential %s: rejected after %d entries, not asked again",
                    name, entries)
        return (f"'{label}' was rejected every time: the owner has entered it "
                f"{entries} times in the last day and {used_by} refused each "
                f"one. It will not be asked for again today. Stop trying to "
                f"sign in and tell the user that {label} keeps being "
                f"rejected, so they can check it themselves.")
    return (f"Asked the owner of this computer for '{label}' on the consent "
            f"card. Once they enter it, {use}")


# ── Localhost enforcement ──────────────────────────────────────────

# Addresses that are "this machine" — secrets never leave the device
_LOCAL_ADDRS = frozenset({
    '127.0.0.1', '::1', 'localhost', '0.0.0.0',
})


def is_local_request(remote_addr: str) -> bool:
    """Check if a request originates from the local device.

    Credential endpoints MUST reject non-local requests.
    Secrets never leave the user's device — no exceptions.

    CI bypass: when ``NUNBA_CI=1`` (set by docker-compose.staging.yml)
    the function trusts ALL request origins.  This is necessary
    because the e2e probe runs from the GitHub runner host and hits
    the container via docker NAT — the request appears to come from
    the docker bridge IP (e.g. 172.17.0.1), not 127.0.0.1, so the
    strict check would reject it.  Production builds NEVER set
    NUNBA_CI=1; the env var is opt-in and explicitly gated to the
    staging compose file.
    """
    import os as _os
    if _os.environ.get('NUNBA_CI', '') == '1':
        return True
    if not remote_addr:
        return False
    return remote_addr in _LOCAL_ADDRS
