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

import bottleneck as bn
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy import signal

from ._plotting import _plot_spec
from ._results import SpecResult
from ._spectrum import (
    _find_dc_bins,
    _find_freq_bins,
    _int_noise_curve,
    _power_from_bins,
    _power_spectrum,
    _to_db,
)


def _rolling_median_bn(x, window_size):
    if window_size % 2 == 0 or window_size < 1:
        window_size += 1
    k = window_size // 2

    # reflect padding (bn.move_median doesn't center by itself)
    x_pad = np.pad(x, k, mode="reflect")

    # bn.move_median uses a centered window when you pad manually
    out = bn.move_median(x_pad, window=window_size, min_count=window_size)

    return out[window_size - 1 :]


def _find_tones(x, enbw_bins, bw_bins, distance, prominence, height):  # noqa: PLR0913
    x_size = x.size
    x_db = _to_db(x)

    if height is None:
        height = _rolling_median_bn(x_db, 51) + 16
    else:
        height = np.ones(x_size) * height

    peaks, _ = signal.find_peaks(
        x_db,
        distance=distance,
        height=height,
        prominence=prominence,
    )

    fs_2_bin = x_size - 1

    if x_db[-1] > height[-1]:
        if peaks.size == 0:
            peaks = np.array([fs_2_bin])
        elif np.all(np.abs(peaks - fs_2_bin) > distance):
            peaks = np.append(peaks, fs_2_bin)

    if peaks.size == 0:
        tones_pow = np.array([])
        tones_loc = np.array([])
        tones_bins = np.array([])

    else:
        tones_pow_list = []
        tones_loc_list = []
        tones_bins_list = []
        for loc in peaks:
            tone_pow, tone_loc, tone_bins = _find_freq_bins(x, int(loc), bw_bins=bw_bins, enbw_bins=enbw_bins)
            tones_pow_list.append(tone_pow)
            tones_loc_list.append(tone_loc)
            tones_bins_list.append(tone_bins)

        tones_pow = np.asarray(tones_pow_list)
        tones_loc = np.asarray(tones_loc_list)
        tones_bins = np.hstack(tones_bins_list)

    return tones_pow, tones_loc, tones_bins


def spec_analysis(  # noqa: PLR0913
    x: NDArray[np.float64],
    fs: float = 1,
    window: None | NDArray[np.float64] = None,
    plot: bool = False,
    ax: None | Axes = None,
    distance: int = 6,
    prominence: int = 10,
    height: int | None = None,
) -> SpecResult:
    """Spectral Analysis.

    Auto-detects DC, tones, and noise from the spectrum.

    Parameters
    ----------
    x : array_like
        Input signal, containing a tone.
    fs : float, optional
        Sampling frequency.
    window : array_like, optional
        Window that will be multiplied with the signal. Default is
        Hann window.
    plot : bool, optional
        If True, the power spectrum is plotted on `ax`. The caller keeps the
        reference to `ax` and can keep customizing it after the call (title,
        scale, limits, etc.).
    distance: number, optional
        Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
        Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
    prominence: number or ndarray, optional
        Required prominence of peaks in dB. Either a number, None, or an array matching x
    height: number or ndarray or sequence, optional
        Required height of peaks in dB. Either a number, None, an array matching x.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. Required if `plot` is True.

    Returns
    -------
    SpecResult
        The analysis results (DC, noise and the detected tones). See
        :class:`SpecResult` for the fields. Use :meth:`SpecResult.to_dict` to get
        them as a dictionary.

    Raises
    ------
    ValueError
        If `plot` is True and no `ax` is given.

    """  # noqa: D416
    if plot and ax is None:
        raise ValueError("An `ax` must be provided when `plot` is True.")

    x_fft_pow, f_array, enbw_bins, bw_bins = _power_spectrum(x, fs=fs, bw=None, window=window)
    sig_len = len(x)

    dc_end = _find_dc_bins(x_fft_pow)
    dc_bins = np.arange(dc_end)

    tones_pow, tones_loc, tones_bins = _find_tones(
        x_fft_pow,
        enbw_bins=enbw_bins,
        bw_bins=bw_bins,
        distance=distance,
        prominence=prominence,
        height=height,
    )

    tones_freq = tones_loc * fs / sig_len
    tones_db = _to_db(tones_pow)

    # Obtain noise bins, by removing the DC bins from the bin list
    if tones_bins.size > 0:
        noise_bins = np.setdiff1d(np.arange(len(x_fft_pow)), np.concatenate((dc_bins, tones_bins)))
    else:
        noise_bins = np.setdiff1d(np.arange(len(x_fft_pow)), dc_bins)

    dc_power = _power_from_bins(x_fft_pow, dc_bins, enbw_bins, bw_bins)
    noise_power = _power_from_bins(x_fft_pow, noise_bins, enbw_bins, bw_bins)

    # total integrated noise curve
    int_noise = _int_noise_curve(x=x_fft_pow / enbw_bins, noise_bins=noise_bins)

    # Calculate THD, Signal Power and N metrics in dB
    dc_db = _to_db(dc_power)
    dc = 10 ** (dc_db / 20)
    noise_db = _to_db(noise_power)

    results = SpecResult(
        dc=float(dc),
        dc_db=float(dc_db),
        noise_db=float(noise_db),
        tones_db=np.asarray(tones_db, dtype=np.float64),
        tones_freq=np.asarray(tones_freq, dtype=np.float64),
    )

    if not plot:
        return results

    _plot_spec(
        x=x_fft_pow,
        freq_array=f_array,
        dc_bins=dc_bins,
        noise_bins=noise_bins,
        ax=ax,
        int_noise=int_noise,
        bw_bins=bw_bins,
        tones_freq=tones_freq,
        tones_bins=tones_bins,
        tones_db=tones_db,
    )

    return results
