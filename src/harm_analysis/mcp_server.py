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
"""MCP server exposing harm_analysis and spec_analysis as tools for AI assistants.

Requires the ``mcp`` extra: ``pip install "harm-analysis[mcp]"``. Run it with the
``harm-analysis-mcp`` command, which serves over stdio, and add that command to your
assistant's MCP configuration (see the documentation).
"""

import math
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np

from ._harmonic import harm_analysis
from ._spectral import spec_analysis

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError as exc:  # pragma: no cover - depends on the optional extra
    raise ImportError(
        'The MCP server needs the "mcp" extra. Install it with: pip install "harm-analysis[mcp]"'
    ) from exc

INSTRUCTIONS = """\
Tools to analyze sampled signals with the harm-analysis package.

- Use run_harm_analysis when the signal contains a known injected tone (e.g. a circuit
  driven by a sine wave). It returns the fundamental, harmonics, DC and noise power and
  THD, SNR, SINAD and THD+N.
- Use run_spec_analysis when there is no known tone; it detects DC, tones and noise.

Pass either file_path (a text file with one sample per line or whitespace-separated
samples) or samples (a list of numbers), and the sampling frequency fs. Powers are in dB;
frequencies are in the units of fs. null means the value is not finite: NaN (e.g. thd_db
when no harmonics are in the band) or -inf dB (a power of zero).

The fundamental is the highest spectral component between DC and bw. If a stronger
interferer is present, run run_spec_analysis first to find the tones and pass a bw below
the interferer.
"""

mcp = MCPServer("harm-analysis", instructions=INSTRUCTIONS)


def _load_signal(file_path: str | None, samples: list[float] | None, delimiter: str | None) -> np.ndarray:
    """Return the signal from exactly one of file_path or samples."""
    if file_path is not None and samples is None:
        return np.loadtxt(file_path, delimiter=delimiter, dtype=np.float64)
    if samples is not None and file_path is None:
        return np.asarray(samples, dtype=np.float64)
    raise ValueError("Pass exactly one of `file_path` or `samples`.")


@contextmanager
def _input_errors_as_tool_errors() -> Generator[None]:
    """Report invalid input and unreadable files to the assistant with their message.

    The MCP server hides the message of unexpected exceptions; ToolError messages are
    passed on, so the assistant can see what was wrong and retry.
    """
    try:
        yield
    except (ValueError, TypeError, OSError) as exc:
        raise ToolError(str(exc)) from exc


def _jsonable(value: Any) -> Any:
    """Convert a result value to plain JSON types; non-finite numbers become None."""
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    value = float(value)
    return value if math.isfinite(value) else None


@mcp.tool()
def run_harm_analysis(  # noqa: PLR0913
    fs: float,
    file_path: str | None = None,
    samples: list[float] | None = None,
    n_harm: int = 5,
    bw: float | None = None,
    delimiter: str | None = None,
) -> dict[str, Any]:
    """Analyze a signal that contains an injected tone.

    Returns the fundamental (fund_db, fund_freq), harmonics (harm_db, harm_freq), DC and
    noise power and THD, SNR, SINAD and THD+N, all in dB, with frequencies in the units of
    fs. Pass exactly one of file_path (text file of samples) or samples.

    Args:
        fs: Sampling frequency, e.g. 1e6 for 1 MHz.
        file_path: Path to a text file with the time-domain samples.
        samples: The time-domain samples, as an alternative to file_path.
        n_harm: Number of harmonics used for THD (default 5).
        bw: Bandwidth for the metrics, in the units of fs (default fs / 2). Use it to exclude
            a stronger tone above the fundamental.
        delimiter: Column delimiter in the file (default: any whitespace).
    """
    with _input_errors_as_tool_errors():
        x = _load_signal(file_path, samples, delimiter)
        results = harm_analysis(x, fs=fs, n_harm=n_harm, bw=bw)
    return {key: _jsonable(value) for key, value in results.to_dict().items()}


@mcp.tool()
def run_spec_analysis(
    fs: float,
    file_path: str | None = None,
    samples: list[float] | None = None,
    delimiter: str | None = None,
) -> dict[str, Any]:
    """Detect DC, tones and noise in a signal without a known injected tone.

    Returns dc (DC level), dc_db, noise_db, and the power (tones_db) and frequency
    (tones_freq) of each detected tone, with frequencies in the units of fs. Pass exactly
    one of file_path (text file of samples) or samples.

    Args:
        fs: Sampling frequency, e.g. 1e6 for 1 MHz.
        file_path: Path to a text file with the time-domain samples.
        samples: The time-domain samples, as an alternative to file_path.
        delimiter: Column delimiter in the file (default: any whitespace).
    """
    with _input_errors_as_tool_errors():
        x = _load_signal(file_path, samples, delimiter)
        results = spec_analysis(x, fs=fs)
    return {key: _jsonable(value) for key, value in results.to_dict().items()}


def main() -> None:
    """Run the MCP server over stdio (entry point of ``harm-analysis-mcp``)."""
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
