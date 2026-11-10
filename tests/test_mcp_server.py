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

"""Tests for the MCP server, using an in-process MCP client."""

import asyncio

import numpy as np
import pytest
from mcp import Client

from harm_analysis import harm_analysis, spec_analysis
from harm_analysis.mcp_server import mcp

FS = 1000.0
N = 4096
T = np.arange(N) / FS
SIGNAL = (
    0.1
    + np.cos(2 * np.pi * 100.3 * T)
    + 0.01 * np.cos(2 * np.pi * 200.6 * T)
    + np.random.default_rng(0).normal(0, 1e-4, N)
)


def _call(tool, arguments):
    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(run())


def _list_tools():
    async def run():
        async with Client(mcp) as client:
            return (await client.list_tools()).tools

    return asyncio.run(run())


def test_tools_are_listed():
    """Both tools are exposed with their parameters and descriptions."""
    tools = {tool.name: tool for tool in _list_tools()}

    assert set(tools) == {"run_harm_analysis", "run_spec_analysis"}
    assert {"fs", "file_path", "samples", "n_harm", "bw"} <= set(tools["run_harm_analysis"].input_schema["properties"])
    assert {"fs", "file_path", "samples"} <= set(tools["run_spec_analysis"].input_schema["properties"])
    assert all(tool.description for tool in tools.values())


def test_harm_analysis_from_file(tmp_path):
    """run_harm_analysis reads a text file and returns the same metrics as the library."""
    path = tmp_path / "signal.txt"
    np.savetxt(path, SIGNAL)

    result = _call("run_harm_analysis", {"fs": FS, "file_path": str(path)})

    assert not result.is_error
    data = result.structured_content
    expected = harm_analysis(SIGNAL, fs=FS)
    assert np.isclose(data["fund_freq"], expected.fund_freq)
    assert np.isclose(data["snr_db"], expected.snr_db)
    assert np.isclose(data["thd_db"], expected.thd_db)
    assert np.allclose(data["harm_freq"], expected.harm_freq)


def test_spec_analysis_from_samples():
    """run_spec_analysis accepts the samples inline."""
    result = _call("run_spec_analysis", {"fs": FS, "samples": SIGNAL.tolist()})

    assert not result.is_error
    data = result.structured_content
    expected = spec_analysis(SIGNAL, fs=FS)
    assert np.isclose(data["dc"], expected.dc)
    assert np.allclose(data["tones_freq"], expected.tones_freq)


def test_non_finite_values_are_null():
    """NaN (THD without harmonics) is returned as null."""
    result = _call("run_harm_analysis", {"fs": FS, "samples": SIGNAL.tolist(), "n_harm": 0})

    assert not result.is_error
    assert result.structured_content["thd_db"] is None


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"fs": FS}, "exactly one of `file_path` or `samples`"),
        ({"fs": FS, "file_path": "x.txt", "samples": [1.0]}, "exactly one of `file_path` or `samples`"),
        ({"fs": FS, "samples": [0.0] * 10}, "at least 32 samples"),
        ({"fs": -1, "samples": SIGNAL.tolist()}, "`fs` must be a positive"),
        ({"fs": FS, "file_path": "does-not-exist.txt"}, "not found"),
    ],
)
def test_errors_reach_the_assistant(arguments, message):
    """Invalid input is reported as a tool error with a readable message."""
    result = _call("run_spec_analysis", arguments)

    assert result.is_error
    assert message in result.content[0].text


def test_harm_analysis_fund_freq():
    """run_harm_analysis passes fund_freq on, so a stronger interferer is not the fundamental."""
    x = SIGNAL + 2 * np.cos(2 * np.pi * 150.7 * T)

    default = _call("run_harm_analysis", {"fs": FS, "samples": x.tolist()}).structured_content
    result = _call("run_harm_analysis", {"fs": FS, "samples": x.tolist(), "fund_freq": 100.3})

    assert not result.is_error
    assert np.isclose(default["fund_freq"], 150.7, atol=0.05)
    assert np.isclose(result.structured_content["fund_freq"], 100.3, atol=0.05)
