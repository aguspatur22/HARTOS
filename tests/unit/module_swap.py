"""swap_modules: put sys.modules entries in place, and restore ONLY those.

The one way for a test to stand a fake module in for a real one.  Do not
use patch.dict('sys.modules', ...): its exit restores the WHOLE dict, so
every module first imported inside it (qwen3vl_backend, activity_stream,
...) is dropped from sys.modules while its stale object stays on its
package.  A later `from integrations.vlm import qwen3vl_backend` then
returns the stale object, a test patches it, and the code under test --
whose call-time import reloads a fresh, unpatched copy -- runs for real.
Measured 2026-09-27: after tests/unit/test_vlm_local_loop.py,
test_vlm_loop_feeds_back_action_output sent real requests to
127.0.0.1:8080, and a test patching a stale subprocess_safe passed with
the code it guarded disabled.
"""
import contextlib
import sys

_MISSING = object()


@contextlib.contextmanager
def swap_modules(replacements):
    """``replacements``: {module name: module object, or None to make the
    import fail}.  Only those keys are changed and only those restored."""
    saved = {k: sys.modules.get(k, _MISSING) for k in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is _MISSING:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
