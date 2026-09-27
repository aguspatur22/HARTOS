"""The executor runs a tool with the arguments the model wrote, or not at all.

Review of c21b5e6e2 / 7d07c0a2d (rvc21_agent_probe.py): the history guard
(ensure_tool_call_arguments_json) reads arguments with load_wire_json, but
the executor (bind_tool_call_arguments) parsed with plain json.loads and fell
back to retrieve_json -> json.loads(repair_json(...)).  Both read an
overflowing number as inf.  So for ``{"id": 620e51403072992921}`` -- valid
JSON -- the tool ran with id=inf and answered "item inf", tool_reply_failed
called that a success (CREATE would bank it), and the next request showed the
model the id it wrote.

The rules pinned here, through a real autogen generate_tool_calls_reply with
the patched executor and a log_tool_execution-wrapped tool (the credential
vault is the only boundary mocked):

  * the executor reads arguments the way the history guard does
    (hartos.helper.parse_tool_arguments), so the tool gets the token the
    model wrote, as a string, and the next request shows the same thing;
  * a call whose arguments would carry Infinity / NaN the model never wrote
    does not run;
  * no second json.loads of tool arguments exists in the executor path
    (source guard).
"""
import ast
import copy
import inspect
import textwrap
import unittest
from unittest import mock

from autogen.agentchat.conversable_agent import ConversableAgent
from flask import Flask

from core.constants import tool_reply_failed
from core.tool_logging import log_tool_execution
from hartos.helper import ToolMessageHandler, force_apply_autogen_json_fix


class ExecutorReadsWhatTheModelWrote(unittest.TestCase):

    def setUp(self):
        self._orig = (ConversableAgent.execute_function,
                      ConversableAgent.a_execute_function)
        self.addCleanup(self._restore)
        self.assertTrue(force_apply_autogen_json_fix())
        vault = mock.patch('core.tool_logging.credential_vault',
                           return_value=None)
        vault.start()
        self.addCleanup(vault.stop)
        self.calls = []

        @log_tool_execution
        def get_item(id: str) -> str:
            self.calls.append(id)
            return 'item ' + str(id)

        self.get_item = get_item

    def _restore(self):
        (ConversableAgent.execute_function,
         ConversableAgent.a_execute_function) = self._orig

    def run_call(self, arguments):
        a = ConversableAgent('a', llm_config=False, human_input_mode='NEVER')
        ex = ConversableAgent('ex', llm_config=False, human_input_mode='NEVER')
        ex.register_function({'get_item': ex._wrap_function(self.get_item)})
        a.send({'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'c1', 'type': 'function',
                                'function': {'name': 'get_item',
                                             'arguments': arguments}}]},
               ex, request_reply=False, silent=True)
        _, reply = ex.generate_tool_calls_reply(sender=a)
        content = reply['tool_responses'][0]['content']
        history = copy.deepcopy(a._oai_messages[ex])
        with Flask(__name__).app_context():
            out = ToolMessageHandler().validate_messages(history)
        sent = [tc['function']['arguments'] for m in out
                for tc in (m.get('tool_calls') or [])][0]
        return content, sent

    def test_a_valid_overflowing_id_reaches_the_tool_as_written(self):
        content, sent = self.run_call('{"id": 620e51403072992921}')
        self.assertEqual(self.calls, ['620e51403072992921'])
        self.assertEqual(content, 'item 620e51403072992921')
        self.assertIn('620e51403072992921', sent)

    def test_repaired_shapes_never_run_with_inf(self):
        for text in ('{"id": 620e51403072992921"}', '{"id": 1e999)}',
                     '{"id": 1e999e}', '{"id": +1e999}'):
            with self.subTest(text=text):
                self.calls.clear()
                content, _ = self.run_call(text)
                for got in self.calls:
                    self.assertNotIn(str(got),
                                     ('inf', '-inf', 'nan', 'Infinity', 'NaN'))
                self.assertNotIn('item inf', content)
                if not self.calls:
                    self.assertTrue(tool_reply_failed(content), content)

    def test_an_invented_infinity_is_refused(self):
        content, _ = self.run_call('{"id": 1e999e}')
        self.assertEqual(self.calls, [])
        self.assertTrue(tool_reply_failed(content), content)


class SourceGuardOneArgumentParser(unittest.TestCase):
    """test_source_guard_: tool arguments are parsed in ONE place
    (parse_tool_arguments).  A json.loads of them anywhere in the executor's
    parse-and-bind step is the second path that ran tools with inf."""

    def test_source_guard_bind_does_not_json_loads_arguments(self):
        from hartos import helper
        tree = ast.parse(textwrap.dedent(
            inspect.getsource(helper.bind_tool_call_arguments)))
        called = {(c.func.attr if isinstance(c.func, ast.Attribute)
                   else getattr(c.func, 'id', None))
                  for c in ast.walk(tree) if isinstance(c, ast.Call)}
        self.assertIn('parse_tool_arguments', called)
        for parser in ('loads', 'repair_json', 'literal_eval'):
            self.assertNotIn(parser, called, parser)


if __name__ == '__main__':
    unittest.main()


class ReviewOf3a7abe540(ExecutorReadsWhatTheModelWrote):
    """Review of 3a7abe540 (rv3a7/cases2.py)."""

    def setUp(self):
        super().setUp()

        @log_tool_execution
        def search(q: str, id: str = '') -> str:
            self.calls.append((q, id))
            return 'found ' + q

        @log_tool_execution
        def get_two(id: str, n: int = 0) -> str:
            self.calls.append((id, n))
            return 'item %s %s' % (id, n)

        self.tools = {'search': search, 'get_two': get_two}

    def run_tool(self, name, arguments):
        a = ConversableAgent('a', llm_config=False, human_input_mode='NEVER')
        ex = ConversableAgent('ex', llm_config=False, human_input_mode='NEVER')
        ex.register_function({name: ex._wrap_function(self.tools[name])})
        a.send({'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'c1', 'type': 'function',
                                'function': {'name': name,
                                             'arguments': arguments}}]},
               ex, request_reply=False, silent=True)
        _, reply = ex.generate_tool_calls_reply(sender=a)
        return reply['tool_responses'][0]['content']

    def test_infinity_inside_a_string_is_not_the_model_writing_infinity(self):
        # "Infinity" appeared in the text (inside "Infinity war"), so the
        # invented Infinity for 1e999e was let through: search('Infinity
        # war', 'Infinity') ran and was banked as a success.
        for text in ('{"q": "Infinity war", "id": 1e999e}',
                     '{"q": "NaN", "id": +1e999}'):
            with self.subTest(text=text):
                self.calls.clear()
                content = self.run_tool('search', text)
                self.assertEqual(self.calls, [], content)
                self.assertTrue(tool_reply_failed(content), content)

    def test_a_value_the_model_wrote_as_infinity_still_runs(self):
        self.calls.clear()
        content = self.run_tool('search', '{"q": "Infinity"}')
        self.assertEqual(self.calls, [('Infinity', '')], content)

    def test_a_line_comment_does_not_swallow_the_rest(self):
        # The repair read format_json_str's output, which has no newlines,
        # so '// the id' swallowed '"n": 2' and the call was refused.  The
        # parent ran it as ('x', 2).
        nl = chr(10)
        for text in ('{' + nl + '  "id": "x", // the id' + nl + '  "n": 2' + nl + '}',
                     '{' + nl + '  "id": "x" // the id' + nl + '  , "n": 2' + nl + '}'):
            with self.subTest(text=text):
                self.calls.clear()
                content = self.run_tool('get_two', text)
                self.assertEqual(self.calls, [('x', 2)], content)
