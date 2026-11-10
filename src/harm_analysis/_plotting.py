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
from matplotlib import patheffects
from numpy.typing import NDArray

from ._spectrum import _mask_array, _to_db

# Series colors (validated for color-vision deficiency and contrast on a light surface)
_TONE = "#2a78d6"  # fundamental / tones
_HARM = "#eb6834"  # harmonics
_INT_NOISE = "#1baf7a"  # integrated noise curves
_NOISE = "#9a9994"  # DC and noise spectrum (recessive)
_GRID = "#e4e3df"

# Space above the highest peak, in dB, so that peak labels don't reach the title
_HEADROOM_DB = 18


def _halo(ax):
    """Outline in the axes background color, so labels stay readable over lines."""
    return [patheffects.withStroke(linewidth=3, foreground=ax.get_facecolor())]


def _label_peak(ax, x, y, name):
    """Write the name and level of a peak just above it (direct label, no box)."""
    if not np.isfinite(y):
        return
    ax.annotate(
        f"{name}\n{y:.1f} dB",
        xy=(x, y),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        linespacing=1.1,
        path_effects=_halo(ax),
        zorder=6,
    )


def _label_curve_end(ax, freq_array, curve, text, end_bin):
    """Write the value of an integrated curve at the end of the analysis band (`end_bin`).

    That is the value the analysis reports (e.g. ``noise_db`` with a `bw`), not the end of
    the curve, which keeps integrating up to fs/2.
    """
    curve = np.ma.masked_invalid(np.ma.asarray(curve)[: end_bin + 1])
    last = np.flatnonzero(~np.ma.getmaskarray(curve))
    if last.size == 0:
        return
    k = int(last[-1])
    in_band_only = end_bin < len(freq_array) - 1
    if in_band_only:
        # The curve continues past bw: mark the reported value and label it on the right
        ax.plot(freq_array[k], curve[k], "o", ms=5, color=_INT_NOISE, mec=ax.get_facecolor(), mew=1.5, zorder=6)
    ax.annotate(
        f"{text} {curve[k]:.1f} dB",
        xy=(freq_array[k], curve[k]),
        xytext=(6, -3) if in_band_only else (-2, 4),
        textcoords="offset points",
        ha="left" if in_band_only else "right",
        va="top" if in_band_only else "bottom",
        fontsize=8,
        color=_INT_NOISE,
        path_effects=_halo(ax),
        zorder=6,
    )


def _style_axes(ax, x_db, freq_array, bw_bins):
    """Light grid, no top/right frame, axis labels, headroom for labels and the bw marker."""
    if bw_bins != len(freq_array) - 1:
        ax.axvline(freq_array[bw_bins], color="#52514e", lw=1, ls=(0, (4, 3)), label="bw", zorder=1)

    finite = x_db[np.isfinite(x_db)]
    if finite.size:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, max(top, float(finite.max()) + _HEADROOM_DB))

    ax.grid(True, which="major", color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b3")
    ax.tick_params(colors="#52514e", which="both")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_ylabel("Power [dB]")
    ax.set_xlabel("Frequency [Hz]")


def _plot_spec(  # noqa: PLR0913
    x: NDArray[np.float64],
    freq_array: NDArray[np.float64],
    dc_bins: NDArray[np.integer],
    noise_bins: NDArray[np.integer],
    int_noise: NDArray[np.float64],
    bw_bins: int,
    ax,
    tones_freq: NDArray[np.float64],
    tones_bins: NDArray[np.integer],
    tones_db: NDArray[np.float64],
):
    x_db = _to_db(x)

    dc_noise_array = _mask_array(x_db, np.concatenate((dc_bins, noise_bins)))
    ax.plot(freq_array, dc_noise_array, label="DC and noise", color=_NOISE, lw=0.8, zorder=2)

    if tones_bins.size > 0:
        tones_array = _mask_array(x_db, tones_bins)
        ax.plot(freq_array, tones_array, label="Tones", color=_TONE, lw=1.5, zorder=3)

        for i in range(tones_db.size):
            _label_peak(ax, tones_freq[i], tones_db[i], f"T{i}")

    ax.plot(freq_array, int_noise, label="Integrated noise", color=_INT_NOISE, lw=2, zorder=4)
    _label_curve_end(ax, freq_array, int_noise, "Noise", bw_bins)

    _style_axes(ax, x_db, freq_array, bw_bins)

    return ax


def _plot_harm(  # noqa: PLR0913
    x: NDArray[np.float64],
    freq_array: NDArray[np.float64],
    dc_bins: NDArray[np.integer],
    noise_bins: NDArray[np.integer],
    int_noise: NDArray[np.float64],
    int_noise_p_harm: NDArray[np.float64],
    bw_bins: int,
    ax,
    fund_pow_db: float,
    fund_freq: float,
    fund_bins: NDArray[np.integer],
    harm_bins: NDArray[np.integer],
    harm_freq: NDArray[np.float64],
    harm_pow_db: NDArray[np.float64],
):
    x_db = _to_db(x)

    dc_noise_array = _mask_array(x_db, np.concatenate((dc_bins, noise_bins)))
    ax.plot(freq_array, dc_noise_array, label="DC and noise", color=_NOISE, lw=0.8, zorder=2)

    if fund_bins.size > 0:
        fund_array = _mask_array(x_db, fund_bins)
        ax.plot(freq_array, fund_array, label="Fundamental", color=_TONE, lw=1.5, zorder=3)
        _label_peak(ax, fund_freq, fund_pow_db, "F")

    if harm_bins.size > 0:
        harm_array = _mask_array(x_db, harm_bins)
        ax.plot(freq_array, harm_array, label="Harmonics", color=_HARM, lw=1.5, zorder=3)

        # H2 is the 2nd harmonic (twice the fundamental frequency), H3 the 3rd, ...
        for i in range(harm_pow_db.size):
            _label_peak(ax, harm_freq[i], harm_pow_db[i], f"H{i + 2}")

        ax.plot(
            freq_array,
            int_noise_p_harm,
            label="Integrated noise + harmonics",
            color=_INT_NOISE,
            lw=2,
            ls=(0, (5, 2)),
            zorder=4,
        )
        _label_curve_end(ax, freq_array, int_noise_p_harm, "Noise + harmonics", bw_bins)

    ax.plot(freq_array, int_noise, label="Integrated noise", color=_INT_NOISE, lw=2, zorder=4)
    _label_curve_end(ax, freq_array, int_noise, "Noise", bw_bins)

    _style_axes(ax, x_db, freq_array, bw_bins)

    return ax
