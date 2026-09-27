"""An elided-text pointer never reaches the user.

Owner ruling (relayed 2026-09-27): a pointer "is never echoed to the user as
an answer".  The wire trim puts ``[elided:<id> <n> chars of <kind>]`` in the
copy of a conversation it sends a model; a model can copy one into its own
reply.  Decision (coordinator, sensible default): strip it where a message
leaves for the user -- core.peer_link.crossbar_publish.publish_agent_message
(every send_message_to_user1, CREATE and REUSE, on the desktop) and the /chat
reply (hart_intelligence_entry._chat_reply, the one builder every /chat
return goes through).  Expansion stays mid-turn, through get_data_by_key.

Behavioural: a model reply that echoes a pointer (the stubbed model) goes
through the real publisher and the real _chat_reply (extracted and run with
its collaborators stubbed, as test_consent_fanout_p2 does); what reaches the
user carries no pointer and keeps the rest of the text.
"""
import os
import re
import sys
import textwrap
from unittest.mock import MagicMock, patch

import core.llm_outbound_logger as lol

HARTOS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_ECHO = ('The page lists three items [elided:0123456789ab 11240 chars of a '
         'tool result] and the total is 42.')


def test_the_parser_and_the_stripper_agree():
    assert lol.parse_elided_pointers(_ECHO)
    stripped = lol.strip_elided_pointers(_ECHO)
    assert not lol.parse_elided_pointers(stripped)
    assert stripped == 'The page lists three items and the total is 42.'


def test_a_published_message_carries_no_pointer():
    from core.peer_link import crossbar_publish
    sent = []
    with patch('core.safe_hartos_attr.safe_hartos_attr',
               return_value=lambda topic, payload: sent.append(payload)):
        assert crossbar_publish.publish_agent_message(
            text=_ECHO, user_id='u1', request_id='r1', prompt_id='p1')
    assert sent, 'nothing was published'
    text = sent[0]['text'][0]
    assert not lol.parse_elided_pointers(text), text
    assert 'the total is 42' in text


def _extract_function(src, name):
    m = re.search(r'^def ' + name + r'\(.*?(?=^def |\Z)', src, re.S | re.M)
    return textwrap.dedent(m.group(0)) if m else None


def test_the_chat_reply_carries_no_pointer():
    src = open(os.path.join(HARTOS_ROOT, 'hart_intelligence_entry.py'),
               encoding='utf-8').read()
    fn_src = _extract_function(src, '_chat_reply')
    assert fn_src
    tts = MagicMock()
    ns = {'_tts_synthesize_and_publish': tts,
          'get_memory': MagicMock(return_value=None),
          'app': MagicMock(),
          'jsonify': lambda x: ('JSONIFIED', x)}
    with patch.dict(sys.modules, {
            'integrations.social': MagicMock(),
            'integrations.social.chat_messages': MagicMock(),
            'core.user_lang': MagicMock(get_preferred_lang=lambda: 'en'),
            'flask': MagicMock(has_request_context=lambda: False)}):
        exec(compile(fn_src, '<isolated:_chat_reply>', 'exec'), ns)
        out = ns['_chat_reply']('u1', 'r1', _ECHO, preferred_lang='en')
    body = out[1]
    assert not lol.parse_elided_pointers(body['response']), body['response']
    assert 'the total is 42' in body['response']
    for call in tts.call_args_list:
        for arg in list(call.args) + list(call.kwargs.values()):
            assert not lol.parse_elided_pointers(str(arg)), arg
