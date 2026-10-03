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
"""Harm Analysis core functions."""

import sys

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ._plotting import _plot_harm
from ._spectrum import (
    _find_dc_bins,
    _find_freq_bins,
    _int_noise_curve,
    _power_from_bins,
    _power_spectrum,
)


def _find_bins(x, n_harm, bw_bins, enbw_bins):
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

    Notes:
    -----
    This function only works for the right-sided power spectrum (positive frequencies
    only).
    """
    # Index of last dc bin
    dc_end = _find_dc_bins(x)
    dc_bins = np.arange(dc_end)
    dc_pow = np.sum(x[dc_bins])

    if bw_bins <= dc_end:
        sys.exit("Error: max bandwidth is too low and is inside the detected dc bins.")

    # the fundamental frequency is found by searching for the bin with the
    # maximum value, excluding DC
    max_loc = np.argmax(x[dc_end:bw_bins]) + dc_end

    # list containing bins of fundamental
    fund_pow, fund_loc, fund_bins = _find_freq_bins(x, max_loc, bw_bins=bw_bins, enbw_bins=enbw_bins)

    # THD+N bins (all bins excluding DC and the fundamental)
    thdn_bins = np.setdiff1d(np.arange(len(x)), np.concatenate((fund_bins, dc_bins)))

    harm_pow, harm_loc, harm_bins = _find_harm(x, fund_loc, n_harm, bw_bins, enbw_bins=enbw_bins)

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

def _find_harm(x, fund_loc, n_harm, bw_bins, enbw_bins):
    if n_harm <= 0:
        harm_loc = np.array([])
        harm_pow = np.array([])
        harm_bins = np.array([])
    else:
        # calculate the frequency of the harmonics.
        # frequencies > fs/2 are ignored
        harm_loc = fund_loc * np.arange(2, n_harm + 2)
        harm_loc = harm_loc[harm_loc <= bw_bins]

        harm_loc_list = []
        harm_pow_list = []
        harm_bins_list = []
        if harm_loc.size > 0:
            for loc in harm_loc:
                harm_pow, harm_loc, harm_bins = _find_freq_bins(x, int(loc), bw_bins=bw_bins, enbw_bins=enbw_bins)
                harm_loc_list.append(harm_loc)
                harm_pow_list.append(harm_pow)
                harm_bins_list.append(harm_bins)

        harm_loc = np.asarray(harm_loc_list)
        harm_pow = np.asarray(harm_pow_list)
        harm_bins = np.hstack(harm_bins_list)

    return harm_pow, harm_loc, harm_bins

def harm_analysis(  # noqa: PLR0913
    x: NDArray[np.float64],
    fs: float = 1,
    bw: float | None = None,
    n_harm: int = 5,
    window: None | NDArray[np.float64] = None,
    plot: bool = False,
    ax: None | Axes = None,
):
    """Harmonic Analysis.

    Calculates SNR, THD, Fundamental power, and Noise power of the input signal x.

    The total harmonic distortion is determined from the fundamental frequency and the
    first five harmonics using a power spectrum of the same length as the input signal.
    A hann window is applied to the signal, before the power spectrum is obtained.

    For simulations with an injected tone.


    Parameters
    ----------
    x : array_like
        Input signal, containing a tone.
    fs : float, optional
        Sampling frequency.
    n_harm : int, optional
         Number of harmonics used in the THD calculation.
    window : array_like, optional
         Window that will be multiplied with the signal. Default is Hann window.
    bw : float, optional
        Bandwidth to use for the calculation of the metrics, in same units as fs.
        Also useful to filter another tone (or noise) with amplitude greater than the
        fundamental and located above a certain frequency (see shaped noise example).
    plot : bool or None, optional
        If True, the power spectrum result is plotted. If specified,
        an `ax` must be provided, and the function returns a dictionary
        with the results and the specified axes (`ax`). If plot is not set,
        only the results are returned.
    ax : plt.Axes or None, optional
        Axes to be used for plotting. Required if plot is set to True.


    Returns
    -------
    properties: dict

        Dictionary containing the analysis results

        - `fund_db`: Fundamental power in decibels.
        - `fund_freq` Frequency of the fundamental tone.
        - `dc_db`: DC power in decibels.
        - `noise_db`: Noise power in decibels.
        - `harm_db`: Harmonics power in decibels.
        - `harm_freq`: Harmonics frequency in Hz.
        - `thd_db`: Total harmonic distortion in decibels. Returns `numpy.nan` if
        `n_harm` is 0 or if all harmonics are outside the bandwidth.
        - `snr_db`: Signal-to-noise ratio in decibels.
        - `sinad_db`: Signal-to-noise-and-distortion ratio in decibels.
        - `thdn_db`: Total harmonic distortion plus noise in decibels.
        - `total_noise_and_dist_db`: Total noise and distortion in decibels.

    ax : matplotlib axes
        If plot is set to True, the Axes used for plotting is returned.

    Notes:
    -----
    The function fails if the fundamental is not the highest spectral component in the
    signal.

    Ensure that the frequency components are far enough apart to accommodate for the
    sidelobe width of the Hann window. If this is not feasible, you can use a different
    window by using the "window" input.

    References:
    ----------
    - [1] Harris, Fredric J. "On the use of windows for harmonic analysis
           with the discrete Fourier transform." Proceedings of the
           IEEE 66.1 (1978): 51-83.
    - [2] Cerna, Michael, and Audrey F. Harvey. The fundamentals of
           FFT-based signal analysis and measurement. Application Note
           041, National Instruments, 2000.
    """  # noqa: D416
    x_fft_pow, f_array, enbw_bins, bw_bins = _power_spectrum(x, fs=fs, bw=bw, window=window)

    sig_len = len(x)

    fund_info, harm_info, dc_info, noise_info, thdn_info = _find_bins(
        x=x_fft_pow, n_harm=n_harm, bw_bins=bw_bins, enbw_bins=enbw_bins
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
    harm_freq = harm_loc * fs / sig_len
    dc_db = 10 * np.log10(dc_pow)
    sig_db = 10 * np.log10(fund_pow)
    noise_db = 10 * np.log10(noise_pow)
    harm_db = 10 * np.log10(harm_pow)
    thdn_db = 10 * np.log10(thdn_pow)
    snr_db = sig_db - noise_db

    # THD in dB is equal to 10*log10(sum(harmonics power)/fundamental power)
    if harm_bins.size > 0:
        thd_db = 10 * np.log10(np.sum(harm_pow) / fund_pow)
        harm_freq = harm_loc * fs / sig_len
    else:
        thd_db = np.nan
        harm_freq = None

    results = {
        "fund_db": sig_db,
        "fund_freq": sig_freq,
        "harm_freq": harm_freq,
        "harm_db": harm_db,
        "dc_db": dc_db,
        "noise_db": noise_db,
        "thd_db": thd_db,
        "snr_db": snr_db,
        "sinad_db": -thdn_db,
        "thdn_db": thdn_db,
        "total_noise_and_dist_db": int_noise[-1],
    }

    if plot is False:
        return results
    ax = _plot_harm(
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

    return results, ax
