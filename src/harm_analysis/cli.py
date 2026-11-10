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
"""CLI for harm_analysis."""

import ast
import json
import math
import operator

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter
from rich.console import Console
from rich.table import Table, box

from . import harm_analysis, spec_analysis

console = Console()


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"log2": math.log2}


def _eval_node(node):
    """Evaluate a restricted arithmetic AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ValueError("unsupported constant")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and len(node.args) == 1
        and not node.keywords
    ):
        return _FUNCS[node.func.id](_eval_node(node.args[0]))
    raise ValueError("unsupported expression")


def _parse_sfactor(ctx, param, value):  # noqa: ARG001
    """Safely evaluate a scaling factor expression, without using ``eval``.

    Supports numbers (including scientific notation such as ``1e5``), ``+``, ``-``, ``*``,
    ``/``, parentheses and ``log2``. Examples: ``5``, ``1/8``, ``1e5``, ``log2(1024)/2``.
    """
    try:
        result = _eval_node(ast.parse(value.strip(), mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        raise click.BadParameter(
            f"{value!r} is not a valid expression (numbers, 1e5, + - * /, parentheses and log2 are supported)"
        ) from exc
    if not math.isfinite(result):
        raise click.BadParameter(f"{value!r} does not evaluate to a finite number")
    return result


arg_filepath = click.argument("filename", type=click.Path(exists=True, readable=True))
opt_fs = click.option("--fs", default=1.0, help="Sampling frequency.")
opt_sep = click.option("--sep", default=" ", help="Separator between items.")
opt_plot = click.option("--plot", is_flag=True, help="Plot the power spectrum of the data")
opt_fund_freq = click.option(
    "--fund-freq",
    type=float,
    default=None,
    help=(
        "Expected frequency of the fundamental, in the units of --fs. Default: the highest "
        "spectral component. Use it if the data may contain a stronger tone."
    ),
)
opt_json = click.option(
    "--json",
    "as_json",
    is_flag=True,
    help=(
        "Print the results as JSON instead of tables. Values that are not finite "
        "(NaN, or -inf dB for a power of zero) are written as null."
    ),
)
opt_sfactor = click.option(
    "--sfactor",
    default="1",
    callback=_parse_sfactor,
    help=(
        "Scaling factor to apply to the data. Supports + - * /, parentheses, 1e5 and log2. "
        "Examples: 1/8, 1e-3, log2(256)/2"
    ),
)


def _load_data(filename, sep, sfactor):
    """Read time-domain samples from a text file and apply the scaling factor."""
    return np.fromfile(filename, sep=sep) * sfactor


def _plot_axes():
    """Create the figure: time-domain data on the left, power spectrum on the right."""
    _, (data_ax, spec_ax) = plt.subplots(1, 2, figsize=(15, 5))
    return data_ax, spec_ax


def _show_plot(data_ax, spec_ax, data):
    """Plot the time-domain data, format the spectrum axes and show the figure."""
    spec_ax.grid(True, which="both")
    spec_ax.set_title("Power spectrum")
    spec_ax.set_xscale("log")
    spec_ax.xaxis.set_major_formatter(EngFormatter(unit="Hz"))

    data_ax.set_title("Data")
    data_ax.plot(data)
    data_ax.grid(True, which="both", linestyle="-")
    data_ax.set_xlabel("[n]")

    plt.tight_layout()
    plt.show()


def _json_value(value):
    """Convert a result value to a JSON-compatible value (non-finite numbers become None)."""
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    value = float(value)
    return value if math.isfinite(value) else None


def _print_json(results):
    """Print the results as a JSON object."""
    data = {key: _json_value(value) for key, value in results.to_dict().items()}
    click.echo(json.dumps(data, indent=2, allow_nan=False))


def _params_table(params):
    """Build the table of result parameters from ``(name, value)`` pairs."""
    table = Table(box=box.MARKDOWN, title="Parameters")
    table.add_column("Parameter", justify="right")
    table.add_column("value", style="magenta")
    for key, value in params:
        table.add_row(key, str(value))
    return table


@click.command(name="harm-analysis")
@arg_filepath
@opt_fs
@opt_sep
@opt_plot
@opt_sfactor
@opt_fund_freq
@opt_json
def harm_analysis_cmd(filename, fs, plot, sep, sfactor, fund_freq, as_json):  # noqa: PLR0913
    """Runs the harm_analysis function for a file containing time domain data."""
    data = _load_data(filename, sep, sfactor)

    if plot:
        data_ax, spec_ax = _plot_axes()
        results = harm_analysis(data, fs=fs, plot=True, ax=spec_ax, fund_freq=fund_freq)
    else:
        results = harm_analysis(data, fs=fs, fund_freq=fund_freq)

    if as_json:
        _print_json(results)
    else:
        print("Function results:")
        console.print(_params_table(results.to_dict().items()))

    if plot:
        _show_plot(data_ax, spec_ax, data)


@click.command(name="spec-analysis")
@arg_filepath
@opt_fs
@opt_plot
@opt_sep
@opt_sfactor
@opt_json
def spec_analysis_cmd(filename, fs, plot, sep, sfactor, as_json):  # noqa: PLR0913
    """Runs the spec_analysis function for a file containing time domain data."""
    data = _load_data(filename, sep, sfactor)

    if plot:
        data_ax, spec_ax = _plot_axes()
        results = spec_analysis(data, fs=fs, plot=True, ax=spec_ax)
    else:
        results = spec_analysis(data, fs=fs)

    if as_json:
        _print_json(results)
        if plot:
            _show_plot(data_ax, spec_ax, data)
        return

    params = ((key, value) for key, value in results.to_dict().items() if key not in {"tones_db", "tones_freq"})
    console.print(_params_table(params))

    if results.tones_db.size == 0:
        console.print("No tones detected")
    else:
        tones_table = Table(box=box.MARKDOWN, title="Tones")
        tones_table.add_column("Tone")
        tones_table.add_column("Amplitude [dB]")
        tones_table.add_column("Frequency [Hz]")

        for i, (amp, freq) in enumerate(zip(results.tones_db, results.tones_freq, strict=True)):
            tones_table.add_row(f"T{i}", str(amp), str(freq))

        console.print(tones_table)

    if plot:
        _show_plot(data_ax, spec_ax, data)
