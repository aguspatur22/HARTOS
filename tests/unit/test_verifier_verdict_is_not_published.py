"""A group member's message2userfinal reaches the user only when it IS an answer.

Review of 31ea54045: REUSE's speaker selectors (state_transition,
state_transition1 for the timer group, state_transition2 for the visual
group) publish any message2userfinal they see through send_message_to_user1,
which on the desktop is the user's chat topic.  That included the
StatusVerifier's own verdict when it carried the key, and an unfilled
'<your answer here>' template: internal plumbing delivered to the user.

The question "can the user read this?" already has one answer,
_reuse_message_is_user_answer (it refuses the verifier seat, this module's
own steers and a <placeholder> value).  The three selectors now send through
_reuse_speaker_says_to_user, which asks it first.  Pre-existing; fixed in the
same review.
"""
import json
import os
import sys
from unittest.mock import patch

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture()
def rr():
    pytest.importorskip('autogen', reason='autogen not installed')
    from hartos import reuse_recipe
    return reuse_recipe


def _msg(name, value):
    return {'role': 'assistant', 'name': name,
            'content': json.dumps({'message2userfinal': value})}


def _say(rr, message):
    with patch.object(rr, 'send_message_to_user1') as send:
        sent = rr._reuse_speaker_says_to_user('u7', message, 42)
    return sent, send


def test_the_agents_real_message_is_sent(rr):
    sent, send = _say(rr, _msg('Assistant', 'Which month should I check?'))
    assert sent is True
    send.assert_called_once_with('u7', 'Which month should I check?', '', 42)


@pytest.mark.parametrize('message', [
    _msg('StatusVerifier', 'There are no tool results in this conversation.'),
    _msg('Assistant', '<your answer here>'),
    _msg('Assistant', ''),
    {'role': 'user', 'name': 'ChatInstructor', 'content':
     'Perform this action -> Action #2: {"message2userfinal": "x"}'},
], ids=['verifier-verdict', 'placeholder', 'empty', 'own-steer'])
def test_plumbing_is_never_sent(rr, message):
    sent, send = _say(rr, message)
    assert sent is False
    send.assert_not_called()


def test_source_guard_selectors_send_only_through_the_gate(rr):
    """The three speaker selectors are closures inside the agent factories
    and cannot be driven without building a group; the behaviour is pinned
    above on the helper, and this guard pins that each selector calls it and
    none sends message2userfinal on its own."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(rr))
    selectors = [n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)
                 and n.name in ('state_transition', 'state_transition1',
                                'state_transition2')]
    assert sorted(n.name for n in selectors) == [
        'state_transition', 'state_transition1', 'state_transition2']
    for fn in selectors:
        calls = [c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call)
                 and isinstance(c.func, ast.Name)]
        assert '_reuse_speaker_says_to_user' in calls, fn.name
        assert 'send_message_to_user1' not in calls, fn.name
