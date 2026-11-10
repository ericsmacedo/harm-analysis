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
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ._plotting import _plot_harm
from ._results import HarmResult
from ._spectral import _MEDIAN_WINDOW, _detection_threshold_db
from ._spectrum import (
    _find_dc_bins,
    _find_freq_bins,
    _int_noise_curve,
    _power_from_bins,
    _power_spectrum,
    _to_db,
    _validate_inputs,
)

# Half-width, in bins, of the search for the fundamental around `fund_freq`
_FUND_SEARCH_BINS = 3
# Probability that noise alone passes as the fundamental at `fund_freq`
_FUND_FALSE_ALARM_RATE = 1e-3


def _peak_near(x, fund_freq, fund_bin, dc_end, bw_bins, freq_array):  # noqa: PLR0913
    """Return the bin of the tone within ``_FUND_SEARCH_BINS`` of `fund_bin`.

    Raises a ValueError if `fund_bin` is inside the DC bins, if the spectrum keeps rising
    past the search range, or if the peak does not stand out from the local noise floor
    (so noise is never analyzed as the fundamental).
    """
    if fund_bin < dc_end:
        raise ValueError(
            f"`fund_freq` ({fund_freq:.6g}) is inside the DC component of the spectrum, which extends to "
            f"{freq_array[dc_end]:.6g} (in the units of `fs`)."
        )
    lo = max(dc_end, fund_bin - _FUND_SEARCH_BINS)
    hi = min(bw_bins, fund_bin + _FUND_SEARCH_BINS)
    peak = lo + int(np.argmax(x[lo : hi + 1]))
    rising_below = peak == lo and peak > dc_end and x[peak - 1] > x[peak]
    rising_above = peak == hi and peak + 1 < x.size and x[peak + 1] > x[peak]
    if rising_below or rising_above:
        raise ValueError(
            f"No spectral peak found within {_FUND_SEARCH_BINS} bins of `fund_freq` ({fund_freq:.6g}); "
            "check `fund_freq` and `fs`."
        )

    # The peak must be a tone, not noise: compare it with the local noise floor using the
    # tone-detection threshold of spec_analysis for a single, known location.
    half = _MEDIAN_WINDOW // 2
    floor_db = np.median(_to_db(x[max(0, peak - half) : peak + half + 1]))
    excess_db = _to_db(x[peak]) - floor_db
    if excess_db < _detection_threshold_db(1, _FUND_FALSE_ALARM_RATE):
        raise ValueError(
            f"No tone found near `fund_freq` ({fund_freq:.6g}): the largest peak within {_FUND_SEARCH_BINS} "
            f"bins is only {excess_db:.1f} dB above the local noise floor; check `fund_freq` and `fs`."
        )
    return peak


def _find_bins(x, n_harm, bw_bins, enbw_bins, freq_array, fund_freq=None, fund_bin=None):  # noqa: PLR0913
    """Find all bins that belong to the fundamental, harmonics, DC, and noise.

    Parameters
    ----------
    x : ndarray
        Input signal. The input should be the absolute value of the right-sided power
        spectrum.
    bw_bins : int
        Max bin to look for the fundamental tone. The function will only try to find
        the fundamental up to this point.
    n_harm : int
        Number of harmonics to find
    freq_array : ndarray
        Frequency of each bin of `x`, used in error messages.
    fund_freq : float or None
        Expected frequency of the fundamental (for error messages).
    fund_bin : int or None
        Expected bin of the fundamental. If None, the highest bin between DC and
        `bw_bins` is used; otherwise the peak within ``_FUND_SEARCH_BINS`` of it.

    Raises:
    ------
    ValueError
        If the analysis band (`bw_bins`) ends inside the DC bins, or if `fund_bin` is
        inside the DC bins or has no tone near it.

    Notes:
    -----
    This function only works for the right-sided power spectrum (positive frequencies
    only).
    """
    # Index of last dc bin
    dc_end = int(_find_dc_bins(x))
    dc_bins = np.arange(dc_end)
    dc_pow = _power_from_bins(x, dc_bins, enbw_bins=enbw_bins, bw_bins=bw_bins)

    if bw_bins <= dc_end:
        raise ValueError(
            f"`bw` is inside the DC component of the spectrum, which extends to {freq_array[dc_end]:.6g} "
            "(in the units of `fs`). Use a larger `bw`."
        )

    if fund_bin is None:
        # the fundamental frequency is found by searching for the bin with the
        # maximum value, excluding DC
        max_loc = int(np.argmax(x[dc_end:bw_bins])) + dc_end
    else:
        max_loc = _peak_near(x, fund_freq, fund_bin, dc_end, bw_bins, freq_array)

    # list containing bins of fundamental
    fund_pow, fund_loc, fund_bins = _find_freq_bins(x, max_loc, bw_bins=bw_bins, enbw_bins=enbw_bins, min_bin=dc_end)

    # THD+N bins (all bins excluding DC and the fundamental)
    thdn_bins = np.setdiff1d(np.arange(len(x)), np.concatenate((fund_bins, dc_bins)))

    harm_pow, harm_loc, harm_bins = _find_harm(
        x, fund_loc, n_harm, bw_bins, enbw_bins=enbw_bins, min_bin=dc_end, fund_bins=fund_bins
    )

    # Remaining bins are considered noise.
    if harm_bins.size > 0:
        noise_bins = np.setdiff1d(thdn_bins, harm_bins)
    else:
        noise_bins = thdn_bins

    noise_pow = _power_from_bins(x, noise_bins, enbw_bins=enbw_bins, bw_bins=bw_bins)

    # According to wikipedia, THD+N in dB is equal to
    # 10*log10(sum(harmonics power + Noise power)/fundamental power).
    # THD+N is recriprocal to SINAD (SINAD_dB = -THD+N_dB)
    thdn_pow = _power_from_bins(x, thdn_bins, enbw_bins=enbw_bins, bw_bins=bw_bins) / fund_pow

    return (
        (fund_pow, fund_loc, fund_bins),
        (harm_pow, harm_loc, harm_bins),
        (dc_pow, dc_bins),
        (noise_pow, noise_bins),
        (thdn_pow, thdn_bins),
    )


def _find_harm(x, fund_loc, n_harm, bw_bins, enbw_bins, min_bin=0, fund_bins=None):  # noqa: PLR0913
    """Find the power, location and bins of the harmonics inside the analysis band.

    Returns empty arrays if `n_harm` is 0 or if all harmonics are above `bw_bins`
    (the `bw` limit or fs/2).

    Bins that belong to the fundamental (`fund_bins`) are never counted as harmonic: without
    a noise floor, the fundamental's leakage can reach the harmonic frequencies, and that
    power must not be counted twice. A harmonic left with no bins has zero power.
    """
    # Expected bin of each harmonic; harmonics above the band are ignored
    harm_locs = fund_loc * np.arange(2, n_harm + 2)
    harm_locs = harm_locs[harm_locs <= bw_bins]

    if harm_locs.size == 0:  # n_harm == 0, or all harmonics outside the band
        return np.array([]), np.array([]), np.array([], dtype=int)

    exclude = np.array([], dtype=int) if fund_bins is None else fund_bins
    harm_pow, harm_loc, harm_bins = [], [], []
    for expected in harm_locs:
        _, loc, bins = _find_freq_bins(x, int(expected), bw_bins=bw_bins, enbw_bins=enbw_bins, min_bin=min_bin)
        bins = np.setdiff1d(bins, exclude)
        harm_pow.append(_power_from_bins(x, bins, enbw_bins=enbw_bins, bw_bins=bw_bins))
        harm_loc.append(loc if bins.size else expected)
        harm_bins.append(bins)

    return np.array(harm_pow), np.array(harm_loc), np.concatenate(harm_bins).astype(int)


def _last_value(curve) -> float:
    """Return the last unmasked value of an integrated (masked) curve, or -inf if all are masked.

    The integrated noise curve is masked at bins that are not noise, so its last element is
    masked whenever the last bin (fs/2) belongs to a tone or harmonic.
    """
    values = np.ma.compressed(curve)
    return float(values[-1]) if values.size else -np.inf


def harm_analysis(  # noqa: PLR0913
    x: NDArray[np.float64],
    fs: float = 1,
    bw: float | None = None,
    n_harm: int = 5,
    window: None | NDArray[np.float64] = None,
    plot: bool = False,
    ax: None | Axes = None,
    fund_freq: float | None = None,
) -> HarmResult:
    """Harmonic Analysis.

    Calculates the fundamental, harmonics, DC and noise power of the input signal x,
    and from them the THD, SNR, SINAD and THD+N.

    Intended for signals with an injected tone (e.g. simulations or measurements of
    a circuit driven by a sine wave). The THD is computed from the fundamental and
    the first `n_harm` harmonics, using a power spectrum of the same length as the
    input signal. A Hann window is applied to the signal before the power spectrum
    is obtained, unless another `window` is given.

    Parameters
    ----------
    x : array_like
        Input signal, containing a tone.
    fs : float, optional
        Sampling frequency. Frequencies in the results are in the same units.
        Default is 1.
    bw : float, optional
        Bandwidth to use for the calculation of the metrics, in the same units as
        `fs`. Default is ``fs / 2``. Also useful to exclude another tone (or noise)
        with amplitude greater than the fundamental and located above a certain
        frequency (see the shaped noise example).
    n_harm : int, optional
        Number of harmonics used in the THD calculation. Default is 5. Harmonics
        above `bw` are ignored.
    window : array_like, optional
        Window that will be multiplied with the signal, with the same length as `x`.
        Default is a Hann window.
    plot : bool, optional
        If True, the power spectrum is plotted on `ax`. The caller keeps the
        reference to `ax` and can keep customizing it after the call (title,
        scale, limits, etc.).
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. Required if `plot` is True.
    fund_freq : float, optional
        Expected frequency of the fundamental, in the same units as `fs`. The
        fundamental is the spectral peak nearest to it (within 3 bins), which must
        stand out from the local noise floor. Use it when
        the signal may contain a component stronger than the injected tone. Default
        is None: the highest spectral component between DC and `bw`.

    Returns
    -------
    HarmResult
        The analysis results (fundamental, harmonics, DC, noise, THD, SNR, SINAD and
        THD+N). See :class:`HarmResult` for the fields. Use
        :meth:`HarmResult.to_dict` to get them as a dictionary.

        Powers of zero are reported as ``-inf`` dB (no warning is raised).

    Raises
    ------
    TypeError
        If `x` or `window` is complex or not numeric.
    ValueError
        If the input signal is constant (all zeros or DC only), as there is no tone to
        analyze; if `bw` is so low that it falls inside the DC component of the
        spectrum; if `fund_freq` is outside the band, inside the DC component or has
        no tone near it (a peak that does not stand out from the noise floor); if
        `plot` is True and no `ax` is given; or if an input is invalid:
        `x` not 1-D, shorter than 32 samples or containing NaN/inf, `fs` not positive,
        `window` not the same length as `x`, `bw` not in ``(0, fs / 2]``, or `n_harm`
        negative.

    Notes
    -----
    Without `fund_freq`, the fundamental is taken to be the highest spectral
    component between DC and `bw`. If the signal contains a stronger tone, that tone
    is reported as the fundamental. Pass `fund_freq` (or use `bw` to exclude a
    stronger tone above the fundamental) to analyze the intended tone; any other tone
    in the band is then counted as noise and lowers the SNR.

    Ensure that the frequency components are far enough apart to accommodate for the
    sidelobe width of the Hann window. If this is not feasible, you can use a different
    window by using the "window" input.

    A component exactly at fs/2 (e.g. the 2nd harmonic of a fundamental at fs/4) is
    sampled as ``A * cos(phase) * (-1)**n``, so its measured power is
    ``A**2 * cos(phase)**2`` and depends on its phase. Components within about one
    bin of fs/2 (or of DC) overlap their mirror image and are also phase dependent.

    References
    ----------
    - [1] Harris, Fredric J. "On the use of windows for harmonic analysis
           with the discrete Fourier transform." Proceedings of the
           IEEE 66.1 (1978): 51-83.
    - [2] Cerna, Michael, and Audrey F. Harvey. The fundamentals of
           FFT-based signal analysis and measurement. Application Note
           041, National Instruments, 2000.

    Examples
    --------
    A 100.3 Hz tone with a 2nd harmonic 40 dB below it, sampled at 1 kHz:

    >>> import numpy as np
    >>> from harm_analysis import harm_analysis
    >>> fs = 1000
    >>> t = np.arange(4096) / fs
    >>> rng = np.random.default_rng(0)
    >>> x = np.cos(2 * np.pi * 100.3 * t) + 0.01 * np.cos(2 * np.pi * 200.6 * t)
    >>> x = x + rng.normal(0, 1e-4, t.size)
    >>> results = harm_analysis(x, fs=fs)
    >>> round(results.fund_freq, 1)
    100.3
    >>> round(results.fund_db, 2)
    -3.01
    >>> round(results.thd_db, 1)
    -40.0
    >>> round(float(results.harm_freq[0]), 1)
    200.6

    All metrics as a dictionary:

    >>> list(results.to_dict())[:4]
    ['fund_db', 'fund_freq', 'harm_freq', 'harm_db']

    Plot the spectrum on your own axes, then keep customizing them:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> results = harm_analysis(x, fs=fs, plot=True, ax=ax)
    >>> ax.set_xscale("log")
    >>> plt.close(fig)
    """  # noqa: D416
    if plot and ax is None:
        raise ValueError("An `ax` must be provided when `plot` is True.")

    x = _validate_inputs(x, fs, window=window, bw=bw, n_harm=n_harm)
    band = fs / 2 if bw is None else bw
    if fund_freq is not None and not (np.isfinite(fund_freq) and 0 < fund_freq <= band):
        raise ValueError(
            f"`fund_freq` must be in (0, {'fs / 2' if bw is None else 'bw'}] = (0, {band}], got {fund_freq}."
        )

    if np.ptp(x) == 0:
        raise ValueError("No fundamental tone found: the input signal is constant (all zeros or DC only).")

    x_fft_pow, f_array, enbw_bins, bw_bins = _power_spectrum(x, fs=fs, bw=bw, window=window)

    sig_len = len(x)

    fund_info, harm_info, dc_info, noise_info, thdn_info = _find_bins(
        x=x_fft_pow,
        n_harm=n_harm,
        bw_bins=bw_bins,
        enbw_bins=enbw_bins,
        freq_array=f_array,
        fund_freq=fund_freq,
        fund_bin=None if fund_freq is None else round(fund_freq * sig_len / fs),
    )

    fund_pow, fund_loc, fund_bins = fund_info
    harm_pow, harm_loc, harm_bins = harm_info
    dc_pow, dc_bins = dc_info
    noise_pow, noise_bins = noise_info
    thdn_pow, thdn_bins = thdn_info

    # total integrated noise curve
    int_noise = _int_noise_curve(x=x_fft_pow / enbw_bins, noise_bins=noise_bins)
    int_noise_p_harm = _int_noise_curve(x=x_fft_pow / enbw_bins, noise_bins=thdn_bins)

    # Calculate THD, Signal Power and N metrics in dB
    sig_freq = fund_loc * fs / sig_len
    dc_db = _to_db(dc_pow)
    sig_db = _to_db(fund_pow)
    noise_db = _to_db(noise_pow)
    harm_db = _to_db(harm_pow)
    thdn_db = _to_db(thdn_pow)
    snr_db = sig_db - noise_db

    # THD in dB is equal to 10*log10(sum(harmonics power)/fundamental power)
    harm_freq = np.asarray(harm_loc, dtype=np.float64) * fs / sig_len
    thd_db = _to_db(np.sum(harm_pow) / fund_pow) if harm_pow.size > 0 else np.nan

    results = HarmResult(
        fund_db=float(sig_db),
        fund_freq=float(sig_freq),
        harm_db=np.asarray(harm_db, dtype=np.float64),
        harm_freq=harm_freq,
        dc_db=float(dc_db),
        noise_db=float(noise_db),
        thd_db=float(thd_db),
        snr_db=float(snr_db),
        sinad_db=float(-thdn_db),
        thdn_db=float(thdn_db),
        total_noise_db=_last_value(int_noise),
    )

    if not plot:
        return results

    _plot_harm(
        x=x_fft_pow,
        fund_freq=sig_freq,
        freq_array=f_array,
        dc_bins=dc_bins,
        fund_bins=fund_bins,
        fund_pow_db=sig_db,
        harm_bins=harm_bins,
        harm_freq=harm_freq,
        harm_pow_db=harm_db,
        noise_bins=noise_bins,
        ax=ax,
        int_noise=int_noise,
        int_noise_p_harm=int_noise_p_harm,
        bw_bins=bw_bins,
    )

    return results
