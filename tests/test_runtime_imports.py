"""Criterion: the machine-type CNN is absent from the runtime path.

The classifier remains available for offline training/eval
(engine.train_classifier, engine.eval.run_eval) but must never be pulled in by
the live-inference stack: streaming, serving, or the desktop UI helpers.
"""

import subprocess
import sys

_FORBIDDEN = ("engine.models.classifier", "engine.train_classifier")

_RUNTIME_MODULES = ("engine.stream", "engine.serve", "engine.baseline", "desktop.ui_helpers")


def test_runtime_path_never_imports_classifier():
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in _RUNTIME_MODULES)
        + f"bad = [m for m in {_FORBIDDEN!r} if m in sys.modules]\n"
        + "assert not bad, f'classifier leaked into runtime imports: {bad}'\n"
        + "print('runtime import graph clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "runtime import graph clean" in result.stdout
