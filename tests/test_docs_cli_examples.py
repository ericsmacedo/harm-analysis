# MIT License
#
# Copyright (c) 2025-2026 ericsmacedo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The CLI outputs shown in the documentation match what the CLI prints."""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "docs" / "cli_examples"


def _generator():
    spec = importlib.util.spec_from_file_location("generate", EXAMPLES_DIR / "generate.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _generator()


# Numbers in the output (e.g. -3.010296203662002, 1e-05)
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _split(text):
    """Return the text with numbers replaced by a placeholder, and the numbers.

    Whitespace runs and table rules ("|-----|") are normalized too: table column widths
    follow the length of the numbers, so they can change with their last digits.
    """
    numbers = np.array([float(n) for n in NUMBER.findall(text)])
    skeleton = NUMBER.sub("<n>", text)
    skeleton = re.sub(r"-{2,}", "--", skeleton)
    return re.sub(r"[ \t]+", " ", skeleton), numbers


@pytest.mark.parametrize(("name", "command", "args"), GENERATOR.EXAMPLES, ids=[e[0] for e in GENERATOR.EXAMPLES])
def test_documented_cli_output_is_current(name, command, args):
    """Regenerate with: uv run python docs/cli_examples/generate.py.

    The text must match exactly; the numbers to 1e-9 (relative), because the last digits of
    full-precision floats differ between platforms and numerical libraries.
    """
    actual_text, actual_numbers = _split(GENERATOR.run(command, args))
    expected_text, expected_numbers = _split((EXAMPLES_DIR / name).read_text())

    assert actual_text == expected_text
    np.testing.assert_allclose(actual_numbers, expected_numbers, rtol=1e-9, atol=1e-12)
