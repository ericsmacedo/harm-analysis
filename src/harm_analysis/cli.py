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
opt_sfactor = click.option(
    "--sfactor",
    default="1",
    callback=_parse_sfactor,
    help="Scaling factor to apply to the data. Supports + - * /, parentheses, 1e5 and log2. Examples: 1/8, 1e-3, log2(256)/2",
)


@click.command(name="harm-analysis")
@arg_filepath
@opt_fs
@opt_sep
@opt_plot
@opt_sfactor
def harm_analysis_cmd(filename, fs, plot, sep, sfactor):
    """Runs the harm_analysis function for a file containing time domain data."""
    # scaling factor
    file_data = np.fromfile(filename, sep=sep) * sfactor

    if plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        results, ax[1] = harm_analysis(file_data, fs=fs, plot=True, ax=ax[1])
    else:
        results = harm_analysis(file_data, fs=fs, plot=False)

    print("Function results:")
    table = Table(box=box.MARKDOWN, title="Parameters")
    table.add_column("Parameter", justify="right")
    table.add_column("value", style="magenta")
    for key, value in results.items():
        table.add_row(key, str(value))

    console.print(table)

    if plot is True:
        ax[1].grid(True, which="both")
        ax[1].set_title("Power spectrum")
        ax[1].set_xscale("log")
        ax[1].xaxis.set_major_formatter(EngFormatter(unit="Hz"))

        ax[0].set_title("Data")
        ax[0].plot(file_data)
        ax[0].grid(True, which="both", linestyle="-")
        ax[0].set_xlabel("[n]")

        plt.tight_layout()
        plt.show()


@click.command(name="spec-analysis")
@arg_filepath
@opt_fs
@opt_plot
@opt_sep
@opt_sfactor
def spec_analysis_cmd(filename, fs, plot, sep, sfactor):
    """Runs the spec_analysis function for a file containing time domain data."""
    # scaling factor
    file_data = np.fromfile(filename, sep=sep) * sfactor

    if plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        results, ax[1] = spec_analysis(file_data, fs=fs, plot=True, ax=ax[1])
    else:
        results = spec_analysis(file_data, fs=fs)

    tones_db = results["tones_db"]
    tones_freq = results["tones_freq"]

    table = Table(box=box.MARKDOWN, title="Parameters")
    table.add_column("Parameter", justify="right")
    table.add_column("value", style="magenta")

    for key, value in results.items():
        if key in {"tones_amp_db", "tones_freq"}:
            continue
        table.add_row(key, str(value))

    if tones_db.size == 0:
        print("No tones detected")
    else:
        tones_table = Table(box=box.MARKDOWN, title="Tones")
        tones_table.add_column("Tone")
        tones_table.add_column("Amplitude [dB]")
        tones_table.add_column("Frequency [Hz]")

        for i, amp in enumerate(tones_db):
            tones_table.add_row(f"T{i}", str(amp), str(tones_freq[i]))

        console.print(table)
        console.print(tones_table)

    if plot is True:
        ax[1].grid(True, which="both")
        ax[1].set_title("Power spectrum")
        ax[1].set_xscale("log")
        ax[1].xaxis.set_major_formatter(EngFormatter(unit="Hz"))

        ax[0].set_title("Data")
        ax[0].plot(file_data)
        ax[0].grid(True, which="both", linestyle="-")
        ax[0].set_xlabel("[n]")

        plt.tight_layout()
        plt.show()
