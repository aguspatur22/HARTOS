"""A call the executor refused as broken JSON is marked refused in the
conversation, never shown to the model as the split dict json_repair made.

Log RCA defect 14, leftover after 68377afd2: the patched executor refuses
``{"text": Financial Dashboard ... - Consulting: $5,000 ...}`` (a string
value left unquoted) because json_repair splits it into
``{"text": "...$10", "Consulting": "5,000", ...}``, which does not bind to
the tool.  But the TOOL-ARGS-GUARD (ensure_tool_call_arguments_json), which
rebuilds every later request's history, repaired the same text on its own and
wrote that split dict back as the call's arguments.  So the model was shown
its broken call as if it had been a well-formed one, next to a reply saying
its arguments were not valid JSON.

The executor is the one place that knows whether the call ran (it has the
tool's signature; the guard does not).  autogen hands it the very function
dict stored in the conversation (generate_tool_calls_reply:
``tool_call.get("function")`` of ``messages[-1]``), so a refusal of
arguments that were not valid JSON now writes the refused stand-in
(refused_arguments_json: what the model wrote, marked refused, and why) into
that dict.  The guard keeps a strict JSON object as it is, so every later
request carries the stand-in.

Real autogen ConversableAgents; the tool is wrapped the way live tools are
(core.tool_logging.log_tool_execution).  Boundaries mocked: the credential
vault only.
"""
import asyncio
import copy
import json
import unittest
from unittest import mock

from autogen.agentchat.conversable_agent import ConversableAgent
from flask import Flask

from core.tool_logging import log_tool_execution
from hartos.helper import (REFUSED_ARGUMENTS_KEY, REFUSED_BECAUSE_KEY,
                           ToolMessageHandler, force_apply_autogen_json_fix)

UNQUOTED = ('{"text": Financial Dashboard Revenue (Monthly): - Trading: '
            '$10,000\n- Consulting: $5,000\n- Education: $2,500\n- TOTAL: '
            '$17,500 Net Profit Margin: 28.6%}')


class _Chat(unittest.TestCase):

    def setUp(self):
        orig = (ConversableAgent.execute_function,
                ConversableAgent.a_execute_function)

        def restore():
            (ConversableAgent.execute_function,
             ConversableAgent.a_execute_function) = orig
        self.addCleanup(restore)
        self.assertTrue(force_apply_autogen_json_fix())
        vault = mock.patch('core.tool_logging.credential_vault',
                           return_value=None)
        vault.start()
        self.addCleanup(vault.stop)

        self.calls = []

        @log_tool_execution
        def send_message_to_user(text: str, avatar_id: str = '',
                                 response_type: str = 'Neutral') -> str:
            self.calls.append(text)
            return 'sent'

        @log_tool_execution
        async def text_2_image(text: str) -> str:
            self.calls.append(text)
            return 'drawn'

        self.assistant = ConversableAgent('assistant', llm_config=False,
                                          human_input_mode='NEVER')
        self.executor = ConversableAgent('executor', llm_config=False,
                                         human_input_mode='NEVER')
        self.executor.register_function({
            'send_message_to_user':
                self.executor._wrap_function(send_message_to_user),
            'text_2_image': self.executor._wrap_function(text_2_image),
        })

    def call(self, name, arguments, run_async=False):
        """The assistant sends one tool call; the executor answers it.
        Returns the executor's reply."""
        self.assistant.send(
            {'role': 'assistant', 'content': None,
             'tool_calls': [{'id': 'call_1', 'type': 'function',
                             'function': {'name': name,
                                          'arguments': arguments}}]},
            self.executor, request_reply=False, silent=True)
        if run_async:
            ok, reply = asyncio.run(self.executor.a_generate_tool_calls_reply(
                sender=self.assistant))
        else:
            ok, reply = self.executor.generate_tool_calls_reply(
                sender=self.assistant)
        self.assertTrue(ok)
        return reply

    def next_request_arguments(self):
        """The call's arguments as the assistant's next request sends them:
        its own history through the ToolMessageHandler guard."""
        history = copy.deepcopy(self.assistant._oai_messages[self.executor])
        with Flask(__name__).app_context():
            out = ToolMessageHandler().validate_messages(history)
        sent = [tc['function']['arguments'] for m in out
                for tc in (m.get('tool_calls') or [])]
        self.assertEqual(len(sent), 1)
        return sent[0]


class RefusedBrokenJsonIsMarked(_Chat):

    def _assert_marked(self, arguments, original):
        parsed = json.loads(arguments)
        self.assertEqual(set(parsed), {REFUSED_ARGUMENTS_KEY,
                                       REFUSED_BECAUSE_KEY})
        self.assertEqual(parsed[REFUSED_ARGUMENTS_KEY], original)
        self.assertIn('not run', parsed[REFUSED_BECAUSE_KEY])
        self.assertIn('not valid JSON', parsed[REFUSED_BECAUSE_KEY])

    def test_split_call_is_sent_back_marked_refused(self):
        reply = self.call('send_message_to_user', UNQUOTED)
        self.assertEqual(self.calls, [])
        self.assertIn('not valid JSON', reply['content'])
        sent = self.next_request_arguments()
        self.assertNotIn('Consulting', json.loads(sent))
        self._assert_marked(sent, UNQUOTED)

    def test_split_call_async_is_marked(self):
        self.call('text_2_image', UNQUOTED, run_async=True)
        self.assertEqual(self.calls, [])
        self._assert_marked(self.next_request_arguments(), UNQUOTED)

    def test_emptied_required_value_is_marked(self):
        self.call('send_message_to_user', '{"text":')
        self.assertEqual(self.calls, [])
        self._assert_marked(self.next_request_arguments(), '{"text":')

    def test_arguments_nothing_could_parse_are_marked(self):
        # When even the fallback parse raises, the call is refused as not
        # JSON; it is marked the same way.  The text must be one the one
        # reader (parse_tool_arguments) refuses, or the fallback never runs:
        # since 3a7abe540 '{"text": <<>>' repairs to {"text": ""} and is
        # refused as an emptied value instead.  A repair that would invent
        # Infinity is refused by the reader.
        broken = '{"text": 1e999e}'
        with mock.patch('hartos.helper.retrieve_json',
                        side_effect=ValueError('unparseable')):
            reply = self.call('send_message_to_user', broken)
        self.assertEqual(self.calls, [])
        self.assertIn('must be in JSON format', reply['content'])
        self._assert_marked(self.next_request_arguments(), broken)

    def test_the_marked_call_is_never_run_again(self):
        self.call('send_message_to_user', UNQUOTED)
        marked = self.assistant._oai_messages[self.executor][-1][
            'tool_calls'][0]['function']['arguments']
        reply = self.call('send_message_to_user', marked)
        self.assertEqual(self.calls, [])
        self.assertIn('refused earlier', reply['content'])


class CallsThatWerePossibleStayAsWritten(_Chat):

    def test_a_repaired_call_that_ran_is_not_marked(self):
        # Trailing comma: repaired, bound, run.  Nothing to mark.
        self.call('send_message_to_user', '{"text": "hi",}')
        self.assertEqual(self.calls, ['hi'])
        stored = self.assistant._oai_messages[self.executor][-1][
            'tool_calls'][0]['function']['arguments']
        self.assertEqual(stored, '{"text": "hi",}')
        self.assertEqual(json.loads(self.next_request_arguments()),
                         {'text': 'hi'})

    def test_valid_json_with_a_wrong_name_stays_as_written(self):
        # Well-formed JSON: the model's own words, refused by name.  The
        # history shows exactly what it sent, which the reply names.
        text = '{"text": "hi", "status": "done"}'
        reply = self.call('send_message_to_user', text)
        self.assertEqual(self.calls, [])
        self.assertIn('Unknown argument(s): status', reply['content'])
        self.assertEqual(self.next_request_arguments(), text)



class GroupChatSeesTheMark(unittest.TestCase):
    """The mark reaches every seat of a real autogen GroupChat: the speaker
    that wrote the call, the executor, and groupchat.messages (speaker
    selection), because the manager broadcasts the one message object."""

    def test_every_seat_holds_the_marked_call(self):
        from autogen import GroupChat, GroupChatManager
        orig = (ConversableAgent.execute_function,
                ConversableAgent.a_execute_function)

        def restore():
            (ConversableAgent.execute_function,
             ConversableAgent.a_execute_function) = orig
        self.addCleanup(restore)
        self.assertTrue(force_apply_autogen_json_fix())
        calls = []

        def send_message_to_user(text: str) -> str:
            calls.append(text)
            return 'sent'

        user = ConversableAgent('user', llm_config=False,
                                human_input_mode='NEVER', default_auto_reply='')
        assistant = ConversableAgent('assistant', llm_config=False,
                                     human_input_mode='NEVER')
        turns = []

        def reply(recipient, messages=None, sender=None, config=None):
            turns.append(1)
            if len(turns) == 1:
                return True, {'role': 'assistant', 'content': None,
                              'tool_calls': [{
                                  'id': 'call_1', 'type': 'function',
                                  'function': {'name': 'send_message_to_user',
                                               'arguments': UNQUOTED}}]}
            return True, 'TERMINATE'
        assistant.register_reply([ConversableAgent, None], reply, position=0)
        executor = ConversableAgent('executor', llm_config=False,
                                    human_input_mode='NEVER')
        executor.register_function({'send_message_to_user':
                                    executor._wrap_function(send_message_to_user)})
        chat = GroupChat(agents=[user, assistant, executor], messages=[],
                         max_round=4, speaker_selection_method='round_robin')
        manager = GroupChatManager(chat, llm_config=False)
        user.initiate_chat(manager, message='report', silent=True)

        self.assertEqual(calls, [])
        views = {'assistant': assistant._oai_messages[manager],
                 'executor': executor._oai_messages[manager],
                 'groupchat': chat.messages}
        for seat, history in views.items():
            sent = [tc['function']['arguments'] for m in history
                    for tc in (m.get('tool_calls') or [])]
            self.assertEqual(len(sent), 1, seat)
            self.assertEqual(json.loads(sent[0])[REFUSED_ARGUMENTS_KEY],
                             UNQUOTED, seat)


def _production_transforms(peers):
    """The chain every production pipeline attaches to its seats
    (create_recipe / reuse_recipe: history_limiter + token_limiter +
    ToolMessageHandler).  autogen's TransformMessages hook deep-copies the
    history before any reply function runs, so the executor is handed a COPY
    of the stored call, never the stored dict itself."""
    from autogen.agentchat.contrib.capabilities import transform_messages
    from core.constants import (AUTOGEN_HISTORY_LIMIT,
                                AUTOGEN_MESSAGE_TOKEN_BUDGET,
                                AUTOGEN_MESSAGE_TOKENS_PER_MESSAGE)
    from hartos.helper import history_limiter, token_limiter
    return transform_messages.TransformMessages(transforms=[
        history_limiter(max_messages=AUTOGEN_HISTORY_LIMIT,
                        keep_first_message=True),
        token_limiter(max_tokens=AUTOGEN_MESSAGE_TOKEN_BUDGET,
                      max_tokens_per_message=AUTOGEN_MESSAGE_TOKENS_PER_MESSAGE,
                      min_tokens=0),
        ToolMessageHandler(peer_agents=peers),
    ], verbose=False)


class ProductionTransformsKeepTheMark(unittest.TestCase):
    """Review of 7d07c0a2d: with the production TransformMessages attached,
    the executor gets a deep copy (transform_messages.py:64), so marking the
    dict it was handed marked nothing the conversation keeps, and every later
    request showed json_repair's split dict again.  The mark has to land on
    the records the conversation keeps."""

    def setUp(self):
        orig = (ConversableAgent.execute_function,
                ConversableAgent.a_execute_function)

        def restore():
            (ConversableAgent.execute_function,
             ConversableAgent.a_execute_function) = orig
        self.addCleanup(restore)
        self.assertTrue(force_apply_autogen_json_fix())
        ctx = Flask(__name__).app_context()
        ctx.push()
        self.addCleanup(ctx.pop)
        self.calls = []

        def send_message_to_user(text: str) -> str:
            self.calls.append(text)
            return 'sent'
        self.tool = send_message_to_user

    def _group_chat(self, transform_seats):
        from autogen import GroupChat, GroupChatManager
        user = ConversableAgent('user', llm_config=False,
                                human_input_mode='NEVER', default_auto_reply='')
        assistant = ConversableAgent('assistant', llm_config=False,
                                     human_input_mode='NEVER')
        turns = []

        def reply(recipient, messages=None, sender=None, config=None):
            turns.append(1)
            if len(turns) == 1:
                return True, {'role': 'assistant', 'content': None,
                              'tool_calls': [{
                                  'id': 'call_1', 'type': 'function',
                                  'function': {'name': 'send_message_to_user',
                                               'arguments': UNQUOTED}}]}
            return True, 'TERMINATE'
        assistant.register_reply([ConversableAgent, None], reply, position=0)
        executor = ConversableAgent('executor', llm_config=False,
                                    human_input_mode='NEVER')
        executor.register_function({'send_message_to_user':
                                    executor._wrap_function(self.tool)})
        seats = {'assistant': assistant, 'executor': executor}
        chain = _production_transforms([assistant, executor])
        for name in transform_seats:
            chain.add_to_agent(seats[name])
        chat = GroupChat(agents=[user, assistant, executor], messages=[],
                         max_round=4, speaker_selection_method='round_robin')
        manager = GroupChatManager(chat, llm_config=False)
        user.initiate_chat(manager, message='report', silent=True)
        return {'assistant': assistant._oai_messages[manager],
                'executor': executor._oai_messages[manager],
                'groupchat': chat.messages}

    def _assert_every_seat_marked(self, views):
        self.assertEqual(self.calls, [])
        for seat, history in views.items():
            sent = [tc['function']['arguments'] for m in history
                    for tc in (m.get('tool_calls') or [])]
            self.assertEqual(len(sent), 1, seat)
            parsed = json.loads(sent[0])
            self.assertEqual(parsed.get(REFUSED_ARGUMENTS_KEY), UNQUOTED, seat)
            self.assertNotIn('Consulting', parsed, seat)

    def test_group_chat_with_production_transforms_on_every_seat(self):
        self._assert_every_seat_marked(
            self._group_chat(('assistant', 'executor')))

    def test_group_chat_with_the_transform_on_the_executor_only(self):
        self._assert_every_seat_marked(self._group_chat(('executor',)))

    def _pairwise(self, run_async, arguments=UNQUOTED, expect='not valid JSON'):
        assistant = ConversableAgent('assistant', llm_config=False,
                                     human_input_mode='NEVER')
        executor = ConversableAgent('executor', llm_config=False,
                                    human_input_mode='NEVER')
        executor.register_function({'send_message_to_user':
                                    executor._wrap_function(self.tool)})
        _production_transforms([assistant, executor]).add_to_agent(executor)
        assistant.send(
            {'role': 'assistant', 'content': None,
             'tool_calls': [{'id': 'call_1', 'type': 'function',
                             'function': {'name': 'send_message_to_user',
                                          'arguments': arguments}}]},
            executor, request_reply=False, silent=True)
        if run_async:
            reply = asyncio.run(executor.a_generate_reply(sender=assistant))
        else:
            reply = executor.generate_reply(sender=assistant)
        self.assertIn(expect, str(reply))
        self.reply = str(reply)
        return {'assistant': assistant._oai_messages[executor],
                'executor': executor._oai_messages[assistant]}

    def test_split_call_reads_as_broken_json_not_a_naming_mistake(self):
        # Before: the executor was handed the guard's split dict as strict
        # JSON and answered "Unknown argument(s): Consulting, ...".
        self._pairwise(run_async=False)
        self.assertNotIn('Unknown argument', self.reply)

    def test_emptied_required_value_does_not_run_through_the_hook(self):
        # Before: the guard turned '{"text":' into {"text": ""}, strict JSON,
        # and the tool ran with text ''.
        for cut in ('{"text":', '{"text": }'):
            views = self._pairwise(run_async=False, arguments=cut,
                                   expect='left empty: text')
            self.assertEqual(self.calls, [], cut)
            sent = views['assistant'][-1]['tool_calls'][0]['function'][
                'arguments']
            self.assertEqual(json.loads(sent)[REFUSED_ARGUMENTS_KEY], cut)

    def test_a_benign_repair_still_runs_through_the_hook(self):
        # A trailing comma: the guard repairs it, the executor finds the
        # record, repairs it the same way, binds it and runs it.  Nothing
        # is marked.
        views = self._pairwise(run_async=False, arguments='{"text": "hi",}',
                               expect='sent')
        self.assertEqual(self.calls, ['hi'])
        self.assertEqual(views['assistant'][-1]['tool_calls'][0]['function'][
            'arguments'], '{"text": "hi",}')

    def test_pairwise_reply_through_the_transform_hook(self):
        self._assert_every_seat_marked(self._pairwise(run_async=False))

    def test_pairwise_async_reply_through_the_transform_hook(self):
        self._assert_every_seat_marked(self._pairwise(run_async=True))

    def _handed_a_guarded_copy(self, run_async):
        """The executor is handed the guard's copy of a stored call, the way
        any reply hook that copies the history hands it (the sync
        TransformMessages hook today; an async one would do the same)."""
        assistant = ConversableAgent('assistant', llm_config=False,
                                     human_input_mode='NEVER')
        executor = ConversableAgent('executor', llm_config=False,
                                    human_input_mode='NEVER')
        executor.register_function({'send_message_to_user':
                                    executor._wrap_function(self.tool)})
        assistant.send(
            {'role': 'assistant', 'content': None,
             'tool_calls': [{'id': 'call_1', 'type': 'function',
                             'function': {'name': 'send_message_to_user',
                                          'arguments': UNQUOTED}}]},
            executor, request_reply=False, silent=True)
        copy_ = ToolMessageHandler().validate_messages(
            copy.deepcopy(executor._oai_messages[assistant]))
        handed = copy_[-1]['tool_calls'][0]['function']
        self.assertIn('Consulting', json.loads(handed['arguments']))
        if run_async:
            _, reply = asyncio.run(executor.a_execute_function(handed))
        else:
            _, reply = executor.execute_function(handed)
        self.assertEqual(self.calls, [])
        self.assertIn('not valid JSON', reply['content'])
        self.assertNotIn('Unknown argument', reply['content'])
        stored = assistant._oai_messages[executor][-1]['tool_calls'][0][
            'function']['arguments']
        self.assertEqual(json.loads(stored)[REFUSED_ARGUMENTS_KEY], UNQUOTED)

    def test_sync_executor_reads_the_stored_record_not_the_copy(self):
        self._handed_a_guarded_copy(run_async=False)

    def test_async_executor_reads_the_stored_record_not_the_copy(self):
        self._handed_a_guarded_copy(run_async=True)

    def test_another_tools_call_with_the_same_text_is_not_marked(self):
        # The executor answers `assistant`; `other` sent the same text to a
        # tool this executor does not run.  That record is not this call.
        assistant = ConversableAgent('assistant', llm_config=False,
                                     human_input_mode='NEVER')
        other = ConversableAgent('other', llm_config=False,
                                 human_input_mode='NEVER')
        executor = ConversableAgent('executor', llm_config=False,
                                    human_input_mode='NEVER')
        executor.register_function({'send_message_to_user':
                                    executor._wrap_function(self.tool)})
        _production_transforms([]).add_to_agent(executor)
        assistant.send(
            {'role': 'assistant', 'content': None,
             'tool_calls': [{'id': 'call_1', 'type': 'function',
                             'function': {'name': 'send_message_to_user',
                                          'arguments': UNQUOTED}}]},
            executor, request_reply=False, silent=True)
        other.send(
            {'role': 'assistant', 'content': None,
             'tool_calls': [{'id': 'call_2', 'type': 'function',
                             'function': {'name': 'post_to_feed',
                                          'arguments': UNQUOTED}}]},
            executor, request_reply=False, silent=True)
        reply = executor.generate_reply(sender=assistant)
        self.assertIn('not valid JSON', str(reply))
        mine = assistant._oai_messages[executor][-1]['tool_calls'][0][
            'function']['arguments']
        self.assertEqual(json.loads(mine)[REFUSED_ARGUMENTS_KEY], UNQUOTED)
        theirs = other._oai_messages[executor][-1]['tool_calls'][0][
            'function']['arguments']
        self.assertEqual(theirs, UNQUOTED)

    def test_the_next_request_carries_the_stand_in(self):
        # What the assistant's next LLM request is built from: its history
        # through the same production chain.
        views = self._pairwise(run_async=False)
        out = _production_transforms([])._transform_messages(
            views['assistant'])
        sent = [tc['function']['arguments'] for m in out
                for tc in (m.get('tool_calls') or [])]
        self.assertEqual(len(sent), 1)
        self.assertEqual(json.loads(sent[0])[REFUSED_ARGUMENTS_KEY], UNQUOTED)


if __name__ == '__main__':
    unittest.main()


class ArgumentsThatAreNotText(_Chat):
    """Review of the defect-14 fix: arguments handed as a dict or None (not
    text) reached the executor's parser as they were: _format_json_str
    raised on them and the fallback read them as ``str(...)``, so a dict
    call and a None call to a zero-parameter tool were refused and marked
    with the Python repr, while the history guard (wire_tool_arguments)
    sent the dict serialized and None as {}.  Both now read them by the
    guard's own rule, so executor and history agree."""

    def setUp(self):
        super().setUp()
        self.pinged = []

        @log_tool_execution
        def ping() -> str:
            self.pinged.append(1)
            return 'pong'
        self.executor.register_function({
            'ping': self.executor._wrap_function(ping)})

    def test_a_dict_runs_and_the_history_shows_it_serialized(self):
        reply = self.call('send_message_to_user', {'text': 'hi'})
        self.assertEqual(self.calls, ['hi'])
        self.assertIn('sent', reply['content'])
        self.assertEqual(json.loads(self.next_request_arguments()),
                         {'text': 'hi'})

    def test_none_runs_a_zero_parameter_tool_and_the_history_agrees(self):
        reply = self.call('ping', None)
        self.assertEqual(self.pinged, [1])
        self.assertIn('pong', reply['content'])
        self.assertEqual(self.next_request_arguments(), '{}')

    def test_none_for_a_tool_with_a_required_value_names_it_unmarked(self):
        reply = self.call('send_message_to_user', None)
        self.assertEqual(self.calls, [])
        self.assertIn('Missing required argument(s): text', reply['content'])
        self.assertEqual(self.next_request_arguments(), '{}')


class FallbackThatFailsIsMarked(_Chat):

    def test_a_python_string_literal_is_refused_and_marked(self):
        # "'abc'": the reader refuses it, retrieve_json reads it as the
        # Python string 'abc', and the reader refuses that too -- the
        # fallback fails with no mock.  The call is marked like any other
        # refused broken JSON.
        reply = self.call('send_message_to_user', "'abc'")
        self.assertEqual(self.calls, [])
        self.assertIn('must be in JSON format', reply['content'])
        # The conversation's own record is marked, not only the guard's
        # view of it: every seat reads that record (speaker selection reads
        # groupchat.messages with no guard).
        stored = self.assistant._oai_messages[self.executor][-1][
            'tool_calls'][0]['function']['arguments']
        for text in (stored, self.next_request_arguments()):
            parsed = json.loads(text)
            self.assertEqual(parsed[REFUSED_ARGUMENTS_KEY], "'abc'")
            self.assertIn('not valid JSON', parsed[REFUSED_BECAUSE_KEY])


class KwargsToolsGetNoInventedKeys(_Chat):
    """Review of the defect-14 fix: a tool that takes **kwargs binds any
    name, so json_repair's split of an unquoted value (a bare
    ``status: ok`` read as a key) reached it with the value cut short.  The
    MCP tool_executor and service_tools endpoint_executor keep a **kwargs
    signature when a tool has no schema.  After a repair, a key the model
    did not write in quotes and the tool does not declare is refused, as it
    is for a fixed signature."""

    def setUp(self):
        super().setUp()
        self.ran = []

        @log_tool_execution
        def run_command(command: str = '', **kwargs) -> str:
            self.ran.append(dict(command=command, **kwargs))
            return 'done'
        self.executor.register_function({
            'run_command': self.executor._wrap_function(run_command)})

    def _assert_refused_and_marked(self, text):
        reply = self.call('run_command', text)
        self.assertEqual(self.ran, [])
        self.assertIn('not valid JSON', reply['content'])
        parsed = json.loads(self.next_request_arguments())
        self.assertEqual(parsed[REFUSED_ARGUMENTS_KEY], text)

    def test_an_unquoted_value_split_into_a_key_is_refused(self):
        self._assert_refused_and_marked(
            '{"command": deploy the app, then report status: ok}')

    def test_the_financial_dashboard_split_is_refused(self):
        self._assert_refused_and_marked(
            UNQUOTED.replace('"text"', '"command"'))

    def test_a_quoted_extra_key_still_runs_after_a_benign_repair(self):
        self.call('run_command', '{"command": "deploy", "status": "ok",}')
        self.assertEqual(self.ran, [{'command': 'deploy', 'status': 'ok'}])

    def test_an_unquoted_declared_key_still_runs(self):
        self.call('run_command', '{command: "deploy"}')
        self.assertEqual(self.ran, [{'command': 'deploy'}])

    def test_strict_json_extra_keys_run_as_written(self):
        self.call('run_command', '{"command": "deploy", "status": "ok"}')
        self.assertEqual(self.ran, [{'command': 'deploy', 'status': 'ok'}])


class KwargsRefusalsNameTheRealFix(KwargsToolsGetNoInventedKeys):
    """Review of 30042a2b6: an unquoted extra key on a **kwargs tool was
    refused with "Expected parameters: command ... using these names",
    which steers the model into dropping the key; a repair that emptied a
    value the model wrote ran the tool with ''; and the quoted-key scan ran
    once per extra key, even on strict JSON."""

    def test_an_unquoted_extra_key_is_told_to_quote_it_not_to_drop_it(self):
        reply = self.call('run_command', '{"command": "ls", cwd: "/tmp"}')
        self.assertEqual(self.ran, [])
        content = reply['content']
        self.assertIn('every key', content)
        self.assertIn('double quotes', content)
        self.assertNotIn('using these names', content)
        self.assertIn('also takes other named values', content)

    def test_a_repair_that_empties_a_written_value_is_refused(self):
        reply = self.call('run_command', '{"command": "ls", "cwd": /tmp}')
        self.assertEqual(self.ran, [], reply['content'])
        self.assertIn('cwd', reply['content'])

    def test_an_empty_value_the_model_wrote_still_runs_after_a_repair(self):
        self.call('run_command', '{"command": "ls", "cwd": "",}')
        self.assertEqual(self.ran, [{'command': 'ls', 'cwd': ''}])

    def test_strict_json_with_an_empty_value_runs(self):
        self.call('run_command', '{"command": "ls", "cwd": ""}')
        self.assertEqual(self.ran, [{'command': 'ls', 'cwd': ''}])


class QuotedKeysAreReadOnce(unittest.TestCase):
    """The quoted-key scan is a full read of the text: once per call after
    a repair, never on strict JSON (review of 30042a2b6: 0.42 s on a 20-key
    100 KB call, once per extra key)."""

    def _count(self, arguments, repaired, as_written):
        import hartos.helper as h

        def run_command(command: str = '', **kwargs):
            return 'done'
        with mock.patch.object(h, '_quoted_top_level_keys',
                               wraps=h._quoted_top_level_keys) as scan:
            h.tool_argument_error(run_command, 'run_command', arguments,
                                  repaired, as_written=as_written)
        return scan.call_count

    def test_strict_json_is_not_scanned(self):
        args = {'command': 'ls', **{'k%d' % i: 'v' * 5000 for i in range(20)}}
        self.assertEqual(self._count(args, False, json.dumps(args)), 0)

    def test_a_repaired_call_is_scanned_once(self):
        text = '{"command": "ls", a: "1", b: "2", c: "3"}'
        args = {'command': 'ls', 'a': '1', 'b': '2', 'c': '3'}
        self.assertEqual(self._count(args, True, text), 1)


class QuotedTopLevelKeys(unittest.TestCase):
    """The keys the rule above credits: quoted, in the outermost object,
    followed by ':' -- never a nested key, a quoted value, or a bare word."""

    def test_only_quoted_outermost_keys(self):
        from hartos.helper import _quoted_top_level_keys
        text = ('{"a": "z", d: "x", "b": {"c": 2}, "e" : "f", '
                "'g': [\"h\"], \"i\\u006a\": 0}")
        self.assertEqual(_quoted_top_level_keys(text),
                         {'a', 'b', 'e', 'g', 'ij'})

    def test_a_split_word_is_not_a_quoted_key(self):
        from hartos.helper import _quoted_top_level_keys
        self.assertEqual(_quoted_top_level_keys(
            '{"command": deploy the app, then report status: ok}'),
            {'command'})


class KeysWrittenEmpty(unittest.TestCase):
    """The keys the empty-value rule credits: outermost keys, quoted or
    bare, whose value the model left empty -- never a nested one, and never
    one whose value it wrote (review of e9daad6c5: a count named every
    empty key, "a" included, when only "cwd" was emptied)."""

    def test_only_outermost_keys_left_empty(self):
        from hartos.helper import _keys_written_empty
        self.assertEqual(_keys_written_empty(
            '{"a": "", "b": null, "c": , d: " ", "e": {"f": ""}, '
            '"g": /tmp, "h": "x", "i":'), {'a', 'b', 'c', 'd', 'i'})


class OnlyTheEmptiedKeyIsNamed(KwargsToolsGetNoInventedKeys):

    def test_a_key_written_empty_is_not_named(self):
        reply = self.call('run_command', '{"a": "", "cwd": /tmp}')
        self.assertEqual(self.ran, [])
        tail = reply['content'].split('came out empty:', 1)[1]
        named = tail.split('.', 1)[0]
        self.assertEqual(named.strip(), 'cwd')
