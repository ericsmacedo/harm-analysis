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
"""Harm Analysis core functions."""

import numpy as np
from numpy.typing import NDArray

from ._spectrum import _mask_array


def _annotate(ax, x, y, text):
    ax.annotate(
        str(text),
        xy=(x, y),
        xytext=(0, 15),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "edgecolor": "brown", "facecolor": "white"},
        arrowprops={"arrowstyle": "-", "color": "brown"},
    )


def _plot_spec(  # noqa: PLR0913
    x: NDArray[np.float64],
    freq_array: NDArray[np.float64],
    dc_bins: NDArray[np.float64],
    noise_bins: NDArray[np.float64],
    int_noise: NDArray[np.float64],
    bw_bins: int,
    ax,
    tones_freq: NDArray[np.float64],
    tones_bins: NDArray[np.float64],
    tones_db: NDArray[np.float64],
):
    x_db = 10 * np.log10(x)

    dc_noise_array = _mask_array(x_db, np.concatenate((dc_bins, noise_bins)))

    if tones_bins.size > 0:
        tones_array = _mask_array(x_db, tones_bins)
        ax.plot(freq_array, tones_array, label="Tones")

        for i in range(tones_db.size):
            x_marker = tones_freq[i]
            y_marker = tones_db[i]
            _annotate(ax, x_marker, y_marker, f"T{i}")

    ax.plot(freq_array, dc_noise_array, label="DC and Noise", color="black")
    ax.plot(freq_array, int_noise, label="Integrated noise", color="green")

    if bw_bins != len(freq_array) - 1:
        ax.axvline(freq_array[bw_bins], color="black", alpha=0.3, label="bw", linestyle="--")

    ax.legend()
    ax.grid()

    ax.set_ylabel("[dB]")
    ax.set_xlabel("[Hz]")

    return ax


def _plot_harm(  # noqa: PLR0913
    x: NDArray[np.float64],
    freq_array: NDArray[np.float64],
    dc_bins: NDArray[np.float64],
    noise_bins: NDArray[np.float64],
    int_noise: NDArray[np.float64],
    int_noise_p_harm: NDArray[np.float64],
    bw_bins: int,
    ax,
    fund_pow_db: float,
    fund_freq: float,
    fund_bins: NDArray[np.float64],
    harm_bins: NDArray[np.float64],
    harm_freq: NDArray[np.float64],
    harm_pow_db: NDArray[np.float64],
):
    x_db = 10 * np.log10(x)

    fund_array = _mask_array(x_db, fund_bins)
    dc_noise_array = _mask_array(x_db, np.concatenate((dc_bins, noise_bins)))

    if fund_bins.size > 0:
        ax.plot(freq_array, fund_array, label="Fundamental")

        x_marker = fund_freq
        y_marker = fund_pow_db
        _annotate(ax, x_marker, y_marker, "F")

    if harm_bins.size > 0:
        harm_array = _mask_array(x_db, harm_bins)
        ax.plot(freq_array, harm_array, label="Harmonics")

        for i in range(harm_pow_db.size):
            x_marker = harm_freq[i]
            y_marker = harm_pow_db[i]
            _annotate(ax, x_marker, y_marker, f"H{i + 1}")

    ax.plot(freq_array, dc_noise_array, label="DC and Noise", color="black")
    ax.plot(freq_array, int_noise_p_harm, label="Integrated noise (inc. harmonics)", color="green")
    ax.plot(freq_array, int_noise, label="Integrated noise", color="purple")

    if bw_bins != len(freq_array) - 1:
        ax.axvline(freq_array[bw_bins], color="black", alpha=0.3, label="bw", linestyle="--")

    ax.legend()
    ax.grid()

    ax.set_ylabel("[dB]")
    ax.set_xlabel("[Hz]")

    return ax
