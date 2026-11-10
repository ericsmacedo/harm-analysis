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
from pathlib import Path

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


@pytest.mark.parametrize(("name", "command", "args"), GENERATOR.EXAMPLES, ids=[e[0] for e in GENERATOR.EXAMPLES])
def test_documented_cli_output_is_current(name, command, args):
    """Regenerate with: uv run python docs/cli_examples/generate.py."""
    assert GENERATOR.run(command, args) == (EXAMPLES_DIR / name).read_text()
