"""TOOL-ARGS-GUARD must hand llama.cpp arguments ITS parser accepts.

Live 2026-09-25 (defect index 3 of the log RCA): a banked recipe step carried
``save_data_in_memory({"key":"user.id","value":620e51403072992921})`` -- an
unquoted hex user id.  Python's ``json.loads`` reads that number as ``inf``
(and accepts ``NaN`` / ``Infinity``), so the guard judged the arguments
"already valid JSON" and left them alone.  llama.cpp's nlohmann parser
refuses the same text with ``out_of_range.406 number overflow``, and returned
HTTP 500 on every later request of the group chat (1,266 of them in two
llama_server_8080 logs).

These tests feed the exact live payload, NaN, Infinity and an inf held in an
already-parsed dict through the guard and ``validate_messages``, then check
the outgoing text with a strict reader written here (not the helper's own):
no NaN/Infinity literals, no number that overflows a double, no lone UTF-16
surrogate escape.  The last was measured by the independent review of
bb809af28: llama.cpp :8080 answered 500 "invalid string: surrogate
U+D800..U+DBFF must be followed by U+DC00..U+DFFF" for a prior tool_call whose
arguments were {"v": "\\ud800"}, which the guard had passed unchanged.
"""
import json
import math
import unittest

from flask import Flask

from hartos.helper import ToolMessageHandler, ensure_tool_call_arguments_json

LIVE_ARGS = '{"key":"user.id","value":620e51403072992921}'


def _strict_loads(text):
    """What nlohmann accepts: no bare constants, no number that is not finite."""
    def refuse(token):
        raise ValueError(f'constant {token!r}')

    def finite(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f'number overflow {token!r}')
        return value

    def no_lone_surrogate(value):
        # nlohmann: "surrogate U+D800..U+DBFF must be followed by U+DC00..U+DFFF".
        # json.loads joins an escaped PAIR into one character, so a surrogate
        # left in a parsed string was a lone one, and utf-8 refuses it.
        if isinstance(value, str):
            value.encode('utf-8')
        elif isinstance(value, list):
            for v in value:
                no_lone_surrogate(v)
        elif isinstance(value, dict):
            for k, v in value.items():
                no_lone_surrogate(k)
                no_lone_surrogate(v)
        return value

    return no_lone_surrogate(json.loads(
        text, parse_constant=refuse, parse_float=finite,
        parse_int=lambda t: (finite(t), int(t))[1]))


def _call(args):
    return [{
        'role': 'assistant', 'content': '',
        'tool_calls': [{'id': 'c1', 'type': 'function',
                        'function': {'name': 'save_data_in_memory',
                                     'arguments': args}}],
    }]


def _out_args(msgs):
    return msgs[0]['tool_calls'][0]['function']['arguments']


class StrictNumbers(unittest.TestCase):

    def test_live_overflowing_id_becomes_its_own_string(self):
        out = _out_args(ensure_tool_call_arguments_json(_call(LIVE_ARGS)))
        parsed = _strict_loads(out)  # was the llama 500
        # The id is kept, as the token the model wrote, not turned into inf
        # or dropped to '{}'.
        self.assertEqual(parsed, {'key': 'user.id', 'value': '620e51403072992921'})

    def test_negative_overflowing_number_becomes_its_own_string(self):
        # json.loads reads it as -inf; nlohmann refuses it like +inf.
        out = _out_args(ensure_tool_call_arguments_json(
            _call('{"v": -620e51403072992921}')))
        self.assertEqual(_strict_loads(out), {'v': '-620e51403072992921'})

    def test_nan_and_infinity_literals_are_quoted(self):
        for literal in ('NaN', 'Infinity', '-Infinity'):
            with self.subTest(literal=literal):
                out = _out_args(ensure_tool_call_arguments_json(
                    _call('{"v": %s, "k": 1}' % literal)))
                self.assertEqual(_strict_loads(out), {'v': literal, 'k': 1})

    def test_huge_integer_is_quoted(self):
        big = '9' * 400  # nlohmann reads it as a double and overflows
        out = _out_args(ensure_tool_call_arguments_json(_call('{"n": %s}' % big)))
        self.assertEqual(_strict_loads(out), {'n': big})

    def test_inf_inside_a_dict_is_not_serialised_as_infinity(self):
        out = _out_args(ensure_tool_call_arguments_json(
            _call({'key': 'user.id', 'value': float('inf')})))
        self.assertEqual(_strict_loads(out), {'key': 'user.id', 'value': 'Infinity'})

    def test_overflow_inside_otherwise_broken_json_is_repaired_strictly(self):
        # Needs repair_json (single quotes) AND carries an overflow.
        out = _out_args(ensure_tool_call_arguments_json(
            _call("{'key': 'user.id', 'value': 620e51403072992921}")))
        parsed = _strict_loads(out)
        self.assertEqual(parsed['key'], 'user.id')

    def test_refused_non_object_falls_back_to_empty_object(self):
        # arguments must be an object; a list carrying NaN is neither
        # sendable as is nor a call, so it becomes a well-formed empty call.
        out = _out_args(ensure_tool_call_arguments_json(_call('[NaN, 1]')))
        self.assertEqual(out, '{}')

    def test_finite_numbers_are_left_byte_identical(self):
        text = '{"a": 1.5e10, "b": -3, "c": 0.25, "d": 12345678901234567890}'
        out = _out_args(ensure_tool_call_arguments_json(_call(text)))
        self.assertEqual(out, text)

    def test_a_coerced_call_is_counted_in_the_guard_log(self):
        app = Flask(__name__)
        with app.app_context(), self.assertLogs(app.logger, 'INFO') as logs:
            ensure_tool_call_arguments_json(_call(LIVE_ARGS))
        self.assertTrue(any('[TOOL-ARGS-GUARD] coerced 1 ' in line
                            for line in logs.output), logs.output)

    def test_validate_messages_sends_strict_arguments(self):
        app = Flask(__name__)
        with app.app_context():
            msgs = [{'role': 'user', 'content': 'remember my id'}] + _call(LIVE_ARGS)
            out = ToolMessageHandler().validate_messages(msgs)
        sent = [tc['function']['arguments'] for m in out
                for tc in (m.get('tool_calls') or [])]
        self.assertEqual(len(sent), 1)
        self.assertEqual(_strict_loads(sent[0])['value'], '620e51403072992921')


class LoneSurrogates(unittest.TestCase):
    """A lone surrogate escape is valid to json.loads and fatal to llama.cpp."""

    def test_lone_high_surrogate_value_becomes_replacement_char(self):
        out = _out_args(ensure_tool_call_arguments_json(
            _call('{"v": "a\\ud800b", "k": 1}')))
        self.assertEqual(_strict_loads(out), {'v': 'a\ufffdb', 'k': 1})

    def test_every_lone_surrogate_in_one_value_is_replaced(self):
        # Two lone surrogates, not a pair (the low one follows 'b', not the
        # high one): both must go, or the second still 500s.
        out = _out_args(ensure_tool_call_arguments_json(
            _call('{"v": "a\\ud800b\\udc00c"}')))
        self.assertEqual(_strict_loads(out), {'v': 'a�b�c'})

    def test_lone_low_surrogate_in_a_key_becomes_replacement_char(self):
        out = _out_args(ensure_tool_call_arguments_json(_call('{"\\udc00x": 1}')))
        self.assertEqual(_strict_loads(out), {'\ufffdx': 1})

    def test_lone_surrogate_inside_a_nested_list_is_replaced(self):
        out = _out_args(ensure_tool_call_arguments_json(
            _call('{"a": ["ok", {"b": "\\udbff"}]}')))
        self.assertEqual(_strict_loads(out), {'a': ['ok', {'b': '\ufffd'}]})

    def test_lone_surrogate_in_a_dict_argument_is_replaced(self):
        out = _out_args(ensure_tool_call_arguments_json(_call({'v': '\ud800'})))
        self.assertEqual(_strict_loads(out), {'v': '\ufffd'})

    def test_two_lone_surrogate_keys_are_not_merged_into_one(self):
        # Review of b0fa4989e, probed: both keys became U+FFFD, so the dict
        # held one of them, {"�": 2}, and the other argument was lost
        # without a trace.  Two arguments that cannot be told apart are not
        # a call that can be repaired: refused like unrecoverable arguments.
        out = _out_args(ensure_tool_call_arguments_json(
            _call('{"\\ud800": 1, "\\udc00": 2}')))
        self.assertEqual(out, '{}')

    def test_a_lone_surrogate_key_colliding_with_a_real_key_is_refused(self):
        # U+FFFD is also a character a key may really hold.
        out = _out_args(ensure_tool_call_arguments_json(
            _call('{"\\ud800": 1, "\\ufffd": 2}')))
        self.assertEqual(out, '{}')

    def test_colliding_keys_in_a_dict_argument_are_refused(self):
        out = _out_args(ensure_tool_call_arguments_json(
            _call({'\ud800': 1, '\udc00': 2})))
        self.assertEqual(out, '{}')

    def test_colliding_keys_in_a_nested_object_are_refused(self):
        out = _out_args(ensure_tool_call_arguments_json(
            _call('{"a": {"x\\ud800": 1, "x\\udbff": 2}}')))
        self.assertEqual(out, '{}')

    def test_escaped_surrogate_pair_is_left_byte_identical(self):
        text = '{"e": "\\ud83d\\ude00"}'  # a valid pair: one emoji
        out = _out_args(ensure_tool_call_arguments_json(_call(text)))
        self.assertEqual(out, text)

    def test_raw_surrogate_pair_characters_are_not_replaced(self):
        # Two raw (unescaped) surrogate characters that form a pair: json.loads
        # keeps them as two characters, and the request encoder (json.dumps,
        # ensure_ascii) sends them as a valid escaped pair.  Not lone.
        text = '{"e": "\ud83d\ude00"}'
        out = _out_args(ensure_tool_call_arguments_json(_call(text)))
        self.assertEqual(out, text)


if __name__ == '__main__':
    unittest.main()
