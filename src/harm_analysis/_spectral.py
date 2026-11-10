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
    _validate_inputs,
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


# Default tone detection threshold (see _detection_threshold_db).
# The noise floor is estimated with a rolling median over _MEDIAN_WINDOW bins. That estimate
# is itself noisy, so _MEDIAN_MARGIN_DB is added on top of the theoretical threshold. Both were
# calibrated on white Gaussian noise: for spectra of 513 to 131073 bins and false_alarm_rate
# from 0.1 down to 1e-3, the measured rate of spectra with a false tone was between about 0.4x
# and 1.5x of false_alarm_rate (4000 to 20000 noise realizations per case at 1e-3). It is
# closest to false_alarm_rate for long spectra and low rates, and conservative otherwise.
_MEDIAN_WINDOW = 101
_MEDIAN_MARGIN_DB = 0.75


def _detection_threshold_db(n_bins: int, false_alarm_rate: float) -> float:
    """Return how far above the local median (in dB) a peak must be to count as a tone.

    For white Gaussian noise, the power of each FFT bin is exponentially distributed, so
    a bin exceeds the median by T dB with probability ``exp(-ln(2) * 10**(T / 10))``.
    Requiring ``n_bins`` times that probability to equal `false_alarm_rate` gives
    ``T = 10 * log10(ln(n_bins / false_alarm_rate) / ln(2))``, plus a small margin for the
    uncertainty of the median estimate.
    """
    return 10 * np.log10(np.log(n_bins / false_alarm_rate) / np.log(2)) + _MEDIAN_MARGIN_DB


def _find_tones(x, enbw_bins, bw_bins, distance, prominence, height, false_alarm_rate):  # noqa: PLR0913
    x_size = x.size
    x_db = _to_db(x)

    if height is None:
        height = _rolling_median_bn(x_db, _MEDIAN_WINDOW) + _detection_threshold_db(x_size, false_alarm_rate)
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
    prominence: float = 10,
    height: float | None = None,
    false_alarm_rate: float = 1e-3,
) -> SpecResult:
    """Spectral Analysis.

    Auto-detects DC, tones, and noise from the spectrum. Intended for signals
    without a known injected tone.

    A tone is a peak in the power spectrum (in dB) that meets the `distance`,
    `prominence` and `height` criteria (see :func:`scipy.signal.find_peaks`). All
    bins that are not DC or part of a tone are counted as noise.

    By default a peak must stand out from the local noise floor (the median of the
    surrounding bins) by a margin derived from `false_alarm_rate`: noise alone
    produces a false tone in at most about `false_alarm_rate` of the analyzed spectra
    (white Gaussian noise; measured between 0.4x and 1.5x of the rate). The
    margin grows slowly with the signal length (about 13.5 dB for 1024 samples and 15 dB
    for 262144 samples with the default rate), since a longer spectrum has more noise
    bins that could exceed it.

    Parameters
    ----------
    x : array_like
        Input signal.
    fs : float, optional
        Sampling frequency. Frequencies in the results are in the same units.
        Default is 1.
    window : array_like, optional
        Window that will be multiplied with the signal, with the same length as `x`.
        Default is a Hann window.
    plot : bool, optional
        If True, the power spectrum is plotted on `ax`. The caller keeps the
        reference to `ax` and can keep customizing it after the call (title,
        scale, limits, etc.).
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. Required if `plot` is True.
    distance : int, optional
        Required minimal distance (>= 1), in frequency bins, between neighbouring
        peaks. Smaller peaks are removed first until the condition is fulfilled for
        all remaining peaks. Default is 6.
    prominence : float, optional
        Required prominence of peaks, in dB. Default is 10.
    height : float, optional
        Required height of peaks, in dB (absolute). Default is None, which uses a
        threshold above the local noise floor set by `false_alarm_rate`.
    false_alarm_rate : float, optional
        Target probability that noise alone produces a false tone in a spectrum, used
        for the default threshold (ignored if `height` is given). Must be in (0, 1).
        Lower values detect fewer spurious tones but need tones to stand out more from
        the noise. Default is 1e-3.

    Returns
    -------
    SpecResult
        The analysis results (DC, noise and the detected tones). See
        :class:`SpecResult` for the fields. Use :meth:`SpecResult.to_dict` to get
        them as a dictionary.

    Raises
    ------
    TypeError
        If `x` or `window` is complex or not numeric.
    ValueError
        If `plot` is True and no `ax` is given, or if an input is invalid: `x` not
        1-D, shorter than 32 samples or containing NaN/inf, `fs` not positive,
        `window` not the same length as `x`, or `false_alarm_rate` not in (0, 1).

    Notes
    -----
    A component exactly at fs/2 (e.g. the 2nd harmonic of a fundamental at fs/4) is
    sampled as ``A * cos(phase) * (-1)**n``, so its measured power is
    ``A**2 * cos(phase)**2`` and depends on its phase. Components within about one
    bin of fs/2 (or of DC) overlap their mirror image and are also phase dependent.

    Examples
    --------
    A signal with a DC offset of 0.5 and two tones, sampled at 1 kHz:

    >>> import numpy as np
    >>> from harm_analysis import spec_analysis
    >>> fs = 1000
    >>> t = np.arange(4096) / fs
    >>> rng = np.random.default_rng(0)
    >>> x = 0.5 + np.cos(2 * np.pi * 100.3 * t) + 0.1 * np.cos(2 * np.pi * 250.7 * t)
    >>> x = x + rng.normal(0, 1e-4, t.size)
    >>> results = spec_analysis(x, fs=fs)
    >>> round(results.dc, 3)
    0.5
    >>> results.tones_freq.round(1).tolist()
    [100.3, 250.7]
    >>> results.tones_db.round(1).tolist()
    [-3.0, -23.0]
    """  # noqa: D416
    if plot and ax is None:
        raise ValueError("An `ax` must be provided when `plot` is True.")

    x = _validate_inputs(x, fs, window=window)
    if not 0 < false_alarm_rate < 1:
        raise ValueError(f"`false_alarm_rate` must be in (0, 1), got {false_alarm_rate}.")

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
        false_alarm_rate=false_alarm_rate,
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
