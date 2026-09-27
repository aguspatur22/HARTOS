""""Is this action autonomous?" has ONE rule: lifecycle_hooks.action_is_autonomous.

The question is asked of the recipe field ``can_perform_without_user_input``
and was answered with a private ``== 'yes'`` at every reader, each with its
own normalisation: a raw subscript compare in REUSE's state_transition and
timer state_transition1 and in CREATE's time_based_execution and timer
state_transition1, a strip().lower() compare in REUSE's session reader
(_reuse_action_is_autonomous), and five raw attribute compares on the A2A
TrainedAgent.  The opposite question already had one rule beside which this
one now lives: lifecycle_hooks.autonomy_needs_user (a leading 'no').

Behaviour is pinned for every value the banked recipes hold.  Census
2026-09-26 of ~/Documents/Nunba/data/prompts (1830 recipe files, action and
flow dicts): 'yes' 2991, missing 191, 'no' 162, None 12,
'no - requires specific dish constraints...' 3.  The review's census of the
same question (1994 'yes', 148 'no', 52 missing, 2 'no - ...') holds the same
four shapes.  For all of them the raw compare and the canonical rule agree,
so no reader changes its answer on data that exists.

The source guard at the bottom fails CI on a new private copy.
"""
import ast
import json
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_MISSING = object()

# (stored value, today's answer).  Every shape in the census above.
CENSUS = [
    ('yes', True),
    ('no', False),
    (_MISSING, False),
    (None, False),
    ('no - requires specific dish constraints, which the user must give', False),
]
_IDS = ['yes', 'no', 'missing', 'None', 'no-with-reason']


def _action(value):
    a = {'action_id': 1, 'action': 'do the thing', 'recipe': []}
    if value is not _MISSING:
        a['can_perform_without_user_input'] = value
    return a


# ── the rule itself ─────────────────────────────────────────────────────

@pytest.mark.parametrize('value,autonomous', CENSUS, ids=_IDS)
def test_the_rule_answers_every_census_value_as_today(value, autonomous):
    from hartos.lifecycle_hooks import action_is_autonomous
    assert action_is_autonomous(
        _action(value).get('can_perform_without_user_input')) is autonomous


@pytest.mark.parametrize('value', [' Yes ', 'YES', 'yes\n'])
def test_the_rule_reads_yes_as_the_session_reader_always_did(value):
    """REUSE's session reader already stripped and lower-cased before its
    compare; the rule keeps that, so no reader got stricter.  (No banked
    recipe holds such a value today; the raw-compare sites would have said
    False for it.)"""
    from hartos.lifecycle_hooks import action_is_autonomous
    assert action_is_autonomous(value) is True


@pytest.mark.parametrize('value,autonomous', CENSUS, ids=_IDS)
def test_autonomous_and_needs_user_never_both_hold(value, autonomous):
    from hartos.lifecycle_hooks import action_is_autonomous, autonomy_needs_user
    v = _action(value).get('can_perform_without_user_input')
    assert not (action_is_autonomous(v) and autonomy_needs_user(v))


# ── REUSE's session reader ──────────────────────────────────────────────

@pytest.mark.parametrize('value,autonomous', CENSUS, ids=_IDS)
def test_reuse_session_reader_answers_every_census_value_as_today(
        monkeypatch, value, autonomous):
    pytest.importorskip('autogen', reason='autogen not installed')
    from hartos import reuse_recipe as rr
    monkeypatch.setattr(rr, 'user_tasks', {'u_1': rr.Action([_action(value)])})
    assert rr._reuse_action_is_autonomous('u_1', 1) is autonomous


# ── the A2A registry (five readers of the flow-level field) ────────────

@pytest.mark.parametrize('value,autonomous', CENSUS, ids=_IDS)
def test_a2a_agent_card_answers_every_census_value_as_today(
        tmp_path, monkeypatch, value, autonomous):
    from integrations.google_a2a.dynamic_agent_registry import (
        DynamicAgentDiscovery)
    recipe = {'persona': 'clerk', 'action': 'file it', 'recipe': [],
              'status': 'done'}
    if value is not _MISSING:
        recipe['can_perform_without_user_input'] = value
    (tmp_path / '71_0_recipe.json').write_text(json.dumps(recipe),
                                               encoding='utf-8')
    disc = DynamicAgentDiscovery(prompts_dir=str(tmp_path))
    assert disc.discover_all_agents() == 1
    agent = disc.get_agent_by_id('71_0')

    assert disc.get_agent_skills(agent)[0]['metadata']['autonomous'] is autonomous
    assert ('Can operate autonomously.'
            in disc.get_agent_description(agent)) is autonomous

    from integrations.google_a2a import register_dynamic_agents as reg
    monkeypatch.setattr(reg, 'get_dynamic_discovery', lambda: disc)
    info = reg.get_registered_agent_info()
    assert (info['autonomous_agents'] == ['71_0']) is autonomous


# ── source guard: no sixth private copy ────────────────────────────────

_VOCAB = ('can_perform_without_user_input', 'autonom')
_SKIP_DIRS = {'venv', '.venv', 'venv311', 'node_modules', '.git', 'tests',
              'build', 'dist', '__pycache__', '.claude', 'python-embed',
              'site-packages', '.cache'}
_OWNER = ('hartos/lifecycle_hooks.py', 'action_is_autonomous')


def _is_yes(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip().lower() == 'yes'
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_is_yes(e) for e in node.elts)
    return False


def _names_the_field(expr, assigned):
    src = ast.unparse(expr)
    if any(v in src for v in _VOCAB):
        return True
    # One hop of indirection: `v = a['can_perform...']; v == 'yes'`.
    return any(isinstance(n, ast.Name) and any(
        v in assigned.get(n.id, '') for v in _VOCAB) for n in ast.walk(expr))


def private_autonomy_checks(src, filename='<src>'):
    """Every `<the field> == 'yes'`-shaped test in *src*, as (line, func)."""
    tree = ast.parse(src, filename)
    hits = []

    def visit(node, func, assigned):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            func = getattr(node, 'name', '<lambda>')
            assigned = dict(assigned)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned[t.id] = ast.unparse(node.value)
        if isinstance(node, ast.Compare):
            operands = [node.left] + list(node.comparators)
            if any(_is_yes(o) for o in operands) and any(
                    _names_the_field(o, assigned)
                    for o in operands if not _is_yes(o)):
                hits.append((node.lineno, func))
        for child in ast.iter_child_nodes(node):
            visit(child, func, assigned)

    visit(tree, '<module>', {})
    return hits


def _repo_checks():
    found = []
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(dirpath, fn)
            try:
                src = open(path, encoding='utf-8').read()
            except (OSError, UnicodeDecodeError):
                continue
            if not any(v in src for v in _VOCAB):
                continue
            rel = os.path.relpath(path, _ROOT).replace(os.sep, '/')
            try:
                for line, func in private_autonomy_checks(src, rel):
                    if (rel, func) != _OWNER:
                        found.append(f'{rel}:{line} in {func}')
            except SyntaxError:
                continue
    return found


def test_source_guard_is_autonomous_has_one_rule():
    offenders = _repo_checks()
    assert offenders == [], (
        "a private `can_perform_without_user_input == 'yes'` test; ask "
        "hartos.lifecycle_hooks.action_is_autonomous instead:\n  "
        + '\n  '.join(offenders))


@pytest.mark.parametrize('snippet', [
    "def f(a):\n    return a['can_perform_without_user_input'] == 'yes'\n",
    "def f(a):\n    return a.get('can_perform_without_user_input') == 'Yes'\n",
    "def f(agent):\n    return agent.can_perform_without_user_input == \"yes\"\n",
    "def f(a):\n    v = a.get('can_perform_without_user_input')\n"
    "    return v in ('yes', 'true')\n",
    "def f(p, i):\n    return _reuse_action_autonomy(p, i) != 'yes'\n",
])
def test_source_guard_sees_every_shape_of_a_copy(snippet):
    """Anti-vacuity: the guard above can fail."""
    assert private_autonomy_checks(snippet) == [(2 if 'v =' not in snippet
                                                 else 3, 'f')]


def test_source_guard_leaves_unrelated_yes_alone():
    src = ("def f(parts, v):\n"
           "    ntp = v == 'yes'\n"
           "    return parts[0] == 'yes' and ntp\n")
    assert private_autonomy_checks(src) == []
