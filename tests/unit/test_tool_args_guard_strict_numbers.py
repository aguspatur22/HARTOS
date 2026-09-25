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
no NaN/Infinity literals, no number that overflows a double.
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

    return json.loads(text, parse_constant=refuse, parse_float=finite,
                      parse_int=lambda t: (finite(t), int(t))[1])


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

    def test_validate_messages_sends_strict_arguments(self):
        app = Flask(__name__)
        with app.app_context():
            msgs = [{'role': 'user', 'content': 'remember my id'}] + _call(LIVE_ARGS)
            out = ToolMessageHandler().validate_messages(msgs)
        sent = [tc['function']['arguments'] for m in out
                for tc in (m.get('tool_calls') or [])]
        self.assertEqual(len(sent), 1)
        self.assertEqual(_strict_loads(sent[0])['value'], '620e51403072992921')


if __name__ == '__main__':
    unittest.main()
