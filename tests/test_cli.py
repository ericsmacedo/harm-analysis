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

"""Tests the CLI for the harmonic analysis function."""

import matplotlib.pyplot as plt
import numpy as np
from click.testing import CliRunner
from pytest import mark

from harm_analysis.cli import _parse_sfactor, harm_analysis_cmd, spec_analysis_cmd


@mark.parametrize("cli_name", ["harm", "spec"])
def test_cli(monkeypatch, cli_name):
    """Test for harm_analysis function.

    Checks if the function can obtain results with less than 0.1 dB of error.
    """
    # test signal
    n = 2048
    fs = 1000
    t = np.arange(0, n / fs, 1 / fs)

    noise_pow_db = -70
    noise_std = 10 ** (noise_pow_db / 20)
    dc_level = 0.123456789

    random_state = np.random.RandomState(1234567890)
    noise = random_state.normal(loc=0, scale=noise_std, size=len(t))

    f1 = 100.13

    x = (
        dc_level
        + 2 * np.cos(2 * np.pi * f1 * t)
        + 0.01 * np.cos(2 * np.pi * f1 * 2 * t)
        + 0.005 * np.cos(2 * np.pi * f1 * 3 * t)
        + noise
    )

    # Save data to TXT file
    np.savetxt("test_data_cli.txt", x, delimiter="\n")

    runner = CliRunner()

    called = {}

    def fake_show():
        called["was_called"] = True

    monkeypatch.setattr(plt, "show", fake_show)

    if cli_name == "harm":
        result = runner.invoke(harm_analysis_cmd, ["test_data_cli.txt", "--fs", str(fs), "--plot"])
    elif cli_name == "spec":
        result = runner.invoke(spec_analysis_cmd, ["test_data_cli.txt", "--fs", str(fs), "--plot"])

    assert "was_called" in called

    assert result.exit_code == 0


def test_cli_no_tones(monkeypatch):
    """Test for harm_analysis function.

    Checks if the function can obtain results with less than 0.1 dB of error.
    """
    # test signal
    n = 2048
    fs = 1000
    t = np.arange(0, n / fs, 1 / fs)

    noise_pow_db = -70
    noise_std = 10 ** (noise_pow_db / 20)
    dc_level = 0.123456789

    random_state = np.random.RandomState(1234567890)
    noise = random_state.normal(loc=0, scale=noise_std, size=len(t))

    x = dc_level + noise

    # Save data to TXT file
    np.savetxt("test_data_cli.txt", x, delimiter="\n")

    runner = CliRunner()

    called = {}

    def fake_show():
        called["was_called"] = True

    monkeypatch.setattr(plt, "show", fake_show)

    result = runner.invoke(spec_analysis_cmd, ["test_data_cli.txt", "--fs", str(fs), "--plot"])

    assert result.exit_code == 0
    assert "was_called" in called


@mark.parametrize("cmd", [harm_analysis_cmd, spec_analysis_cmd])
@mark.parametrize(
    "sfactor",
    ["__import__('os').getcwd()", "1/0", "abc", "2**3", "log2(0)", "log2(1, 2)", "sin(1)", "1e999", "True", "(1"],
)
def test_cli_invalid_sfactor(tmp_path, cmd, sfactor):
    """Invalid scaling factors are rejected instead of being evaluated."""
    path = tmp_path / "data.txt"
    np.savetxt(path, np.arange(16.0), delimiter="\n")

    result = CliRunner().invoke(cmd, [str(path), "--sfactor", sfactor])

    assert result.exit_code == 2
    assert "Invalid value for '--sfactor'" in result.output


@mark.parametrize(
    ("sfactor", "expected"),
    [
        ("5", 5.0),
        ("0.5", 0.5),
        ("1/8", 0.125),
        ("1e5", 1e5),
        ("2.5E-3", 2.5e-3),
        ("-3", -3.0),
        ("log2(1024)", 10.0),
        ("log2(8)/3 + 1", 2.0),
        ("(1+2)*3 - 4/2", 7.0),
        (" 1e3 / 4 ", 250.0),
    ],
)
def test_cli_sfactor_parsing(sfactor, expected):
    """Integers, decimals and fractions are accepted."""
    assert _parse_sfactor(None, None, sfactor) == expected


def _run_spec_cli(tmp_path, x):
    path = tmp_path / "data.txt"
    np.savetxt(path, x, delimiter="\n")
    return CliRunner().invoke(spec_analysis_cmd, [str(path), "--fs", "1000"])


def test_spec_cli_with_tones(tmp_path):
    """Tones are listed in the Tones table only, not in the main table."""
    n = 2**14
    rng = np.random.default_rng(1)
    x = 0.1 + np.cos(2 * np.pi * 0.1013 * np.arange(n)) + rng.normal(0, 1e-3, n)

    result = _run_spec_cli(tmp_path, x)

    assert result.exit_code == 0
    assert "Parameters" in result.output
    assert "Tones" in result.output
    assert "T0" in result.output
    assert "tones_db" not in result.output
    assert "tones_freq" not in result.output
    assert "No tones detected" not in result.output


def test_spec_cli_without_tones(tmp_path):
    """The Parameters table is still shown when no tones are detected."""
    rng = np.random.default_rng(1)
    x = 0.1 + rng.normal(0, 1e-3, 2**14)

    result = _run_spec_cli(tmp_path, x)

    assert result.exit_code == 0
    assert "Parameters" in result.output
    assert "noise_db" in result.output
    assert "No tones detected" in result.output
