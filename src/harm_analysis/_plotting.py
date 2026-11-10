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


# Labels closer than this fraction of the axes width are staggered vertically
_MIN_LABEL_SPACING = 0.07
_STAGGER_POINTS = 22


def _axes_fraction(ax, x):
    """Horizontal position of x as a fraction of the axes width, for the current x scale."""
    ax.autoscale_view()
    lo, hi = ax.get_xlim()
    if ax.get_xscale() == "log":
        x, lo, hi = (np.log10(max(v, 1e-300)) for v in (x, lo, hi))
    return (x - lo) / (hi - lo) if hi != lo else 0.0


def _label_peaks(ax, peaks):
    """Label peaks given as (x, y, name), staggering labels that would overlap.

    A label closer than ``_MIN_LABEL_SPACING`` (of the axes width) to the previous one goes
    one level higher, cycling through three levels; a label with room goes back to level 0.
    """
    peaks = sorted((p for p in peaks if np.isfinite(p[1])), key=lambda p: p[0])
    previous = None
    level = 0
    for x, y, name in peaks:
        position = _axes_fraction(ax, x)
        close = previous is not None and position - previous < _MIN_LABEL_SPACING
        level = (level + 1) % 3 if close else 0
        _label_peak(ax, x, y, name, offset=5 + level * _STAGGER_POINTS)
        previous = position


def _label_peak(ax, x, y, name, offset=5):
    """Write the name and level of a peak just above it (direct label, no box)."""
    ax.annotate(
        f"{name}\n{y:.1f} dB",
        xy=(x, y),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        linespacing=1.1,
        path_effects=_halo(ax),
        zorder=6,
    )


def _band_value(ax, freq_array, curve, end_bin):
    """Value of an integrated curve at the end of the analysis band (`end_bin`), or None.

    That is the value the analysis reports (e.g. ``noise_db`` with a `bw`), not the end of
    the curve, which keeps integrating up to fs/2. When the band ends before fs/2, the point
    is marked on the curve.
    """
    curve = np.ma.masked_invalid(np.ma.asarray(curve)[: end_bin + 1])
    valid = np.flatnonzero(~np.ma.getmaskarray(curve))
    if valid.size == 0:
        return None
    k = int(valid[-1])
    if end_bin < len(freq_array) - 1:
        ax.plot(freq_array[k], curve[k], "o", ms=5, color=_INT_NOISE, mec=ax.get_facecolor(), mew=1.5, zorder=6)
    return float(curve[k])


def _with_value(label, value):
    return label if value is None else f"{label} ({value:.1f} dB)"


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

        _label_peaks(ax, [(tones_freq[i], tones_db[i], f"T{i}") for i in range(tones_db.size)])

    noise = _band_value(ax, freq_array, int_noise, bw_bins)
    ax.plot(freq_array, int_noise, label=_with_value("Integrated noise", noise), color=_INT_NOISE, lw=2, zorder=4)

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

    if harm_bins.size > 0:
        harm_array = _mask_array(x_db, harm_bins)
        ax.plot(freq_array, harm_array, label="Harmonics", color=_HARM, lw=1.5, zorder=3)

        noise_harm = _band_value(ax, freq_array, int_noise_p_harm, bw_bins)
        ax.plot(
            freq_array,
            int_noise_p_harm,
            label=_with_value("Integrated noise + harmonics", noise_harm),
            color=_INT_NOISE,
            lw=2,
            ls=(0, (5, 2)),
            zorder=4,
        )

    noise = _band_value(ax, freq_array, int_noise, bw_bins)
    ax.plot(freq_array, int_noise, label=_with_value("Integrated noise", noise), color=_INT_NOISE, lw=2, zorder=4)

    peaks = [(fund_freq, fund_pow_db, "F")] if fund_bins.size > 0 else []
    if harm_bins.size > 0:
        # H2 is the 2nd harmonic (twice the fundamental frequency), H3 the 3rd, ...
        peaks += [(harm_freq[i], harm_pow_db[i], f"H{i + 2}") for i in range(harm_pow_db.size)]
    _label_peaks(ax, peaks)

    _style_axes(ax, x_db, freq_array, bw_bins)

    return ax
