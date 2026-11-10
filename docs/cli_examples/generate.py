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
"""Generate the example data and the CLI outputs shown in docs/cli.md.

Run from the repository root:  uv run python docs/cli_examples/generate.py
tests/test_docs_cli_examples.py checks that the committed outputs are up to date.
"""

from pathlib import Path

import numpy as np
from click.testing import CliRunner

from harm_analysis.cli import harm_analysis_cmd, spec_analysis_cmd

HERE = Path(__file__).parent

# (output file, command, arguments); paths are relative to docs/cli_examples
EXAMPLES = [
    ("harm.txt", harm_analysis_cmd, ["signal.txt", "--fs", "1000"]),
    ("harm_json.txt", harm_analysis_cmd, ["signal.txt", "--fs", "1000", "--json"]),
    ("harm_fund_freq.txt", harm_analysis_cmd, ["interferer.txt", "--fs", "1000", "--fund-freq", "100.3", "--json"]),
    ("spec.txt", spec_analysis_cmd, ["signal.txt", "--fs", "1000"]),
    (
        "adc_codes.txt",
        harm_analysis_cmd,
        ["adc_codes.csv", "--fs", "1e6", "--sep", ",", "--sfactor", "1/32768", "--json"],
    ),
]


def write_data():
    """Write the example input files (seeded, so they are reproducible)."""
    fs, n = 1000, 4096
    t = np.arange(n) / fs
    noise = np.random.default_rng(0).normal(0, 1e-4, n)
    tone = (
        0.1
        + np.cos(2 * np.pi * 100.3 * t)
        + 0.01 * np.cos(2 * np.pi * 200.6 * t)
        + 0.003 * np.cos(2 * np.pi * 300.9 * t)
    )
    np.savetxt(HERE / "signal.txt", tone + noise, fmt="%.9f")
    interferer = np.cos(2 * np.pi * 100.3 * t) + 2 * np.cos(2 * np.pi * 150.7 * t)
    np.savetxt(HERE / "interferer.txt", interferer + noise, fmt="%.9f")
    # 16-bit ADC codes of a 0.9 full-scale tone at 1 MHz sampling, comma separated
    t_adc = np.arange(n) / 1e6
    codes = np.round(0.9 * 32767 * np.sin(2 * np.pi * 10.3e3 * t_adc) + np.random.default_rng(1).normal(0, 2, n))
    (HERE / "adc_codes.csv").write_text(",".join(str(int(c)) for c in codes) + "\n")


def run(command, args):
    """Run a CLI command from docs/cli_examples and return its output."""
    import os  # noqa: PLC0415

    cwd = Path.cwd()
    os.chdir(HERE)
    try:
        result = CliRunner().invoke(command, args, terminal_width=88)
    finally:
        os.chdir(cwd)
    if result.exit_code != 0:
        raise RuntimeError(f"{args} failed: {result.output}")
    return result.output


def main():
    """Regenerate the data files and the outputs."""
    write_data()
    for name, command, args in EXAMPLES:
        (HERE / name).write_text(run(command, args))


if __name__ == "__main__":
    main()
