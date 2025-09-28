# MIT License
#
# Copyright (c) 2025 ericsmacedo
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

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter
from rich.console import Console
from rich.table import Table, box

from ._harm_analysis import harm_analysis, spec_analysis

console = Console()

arg_filepath = click.argument("filename", type=click.Path(exists=True, readable=True))
opt_fs = click.option("--fs", default=1.0, help="Sampling frequency.")
opt_sep = click.option("--sep", default=" ", help="Separator between items.")
opt_plot = click.option("--plot", is_flag=True, help="Plot the power spectrum of the data")
opt_sfactor = click.option(
    "--sfactor",
    default="1",
    help="Scaling factor to apply to the data.  Examples: 1/8, 5, etc",
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
    file_data = np.fromfile(filename, sep=sep) * eval(sfactor)

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
    """Runs the harm_analysis function for a file containing time domain data."""
    # scaling factor
    file_data = np.fromfile(filename, sep=sep) * eval(sfactor)

    if plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        results, ax[1] = spec_analysis(file_data, fs=fs, plot=True, ax=ax[1])
    else:
        results = spec_analysis(file_data, fs=fs)

    tones_amp_db = results["tones_amp_db"]
    tones_freq = results["tones_freq"]

    table = Table(box=box.MARKDOWN, title="Parameters")
    table.add_column("Parameter", justify="right")
    table.add_column("value", style="magenta")

    for key, value in results.items():
        if key in {"tones_amp_db", "tones_freq"}:
            continue
        table.add_row(key, str(value))

    tones_table = Table(box=box.MARKDOWN, title="Tones")
    tones_table.add_column("Tone")
    tones_table.add_column("Amplitude [dB]")
    tones_table.add_column("Frequency [Hz]")

    for i, amp in enumerate(tones_amp_db):
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
