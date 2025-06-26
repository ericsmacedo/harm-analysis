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

from .harm_analysis import dc_measurement, harm_analysis


@click.command()
@click.argument("filename", type=click.Path(exists=True, readable=True))
@click.option("--fs", default=1.0, help="Sampling frequency.")
@click.option("--plot", is_flag=True, help="Plot the power spectrum of the data")
@click.option("--dc", is_flag=True, help="Run only DC measurement")
@click.option("--sep", default=" ", help="Separator between items.")
@click.option(
    "--sfactor",
    default="1",
    help="Scaling factor to apply to the data.  Examples: 1/8, 5, etc",
)
def cli(filename, fs, plot, sep, sfactor, dc):  # noqa: PLR0913
    """Runs the harm_analysis function for a file containing time domain data."""
    # scaling factor
    file_data = np.fromfile(filename, sep=sep) * eval(sfactor)

    if plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        if dc:
            results, ax[1] = dc_measurement(file_data, fs=fs, plot=True, ax=ax[1])
        else:
            results, ax[1] = harm_analysis(file_data, fs=fs, plot=True, ax=ax[1])
    elif dc:
        results = dc_measurement(file_data, fs=fs)
    else:
        results = harm_analysis(file_data, fs=fs, plot=False)

    print("Function results:")
    for key, value in results.items():
        click.echo(f"{key.ljust(10)}: {value}")

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
