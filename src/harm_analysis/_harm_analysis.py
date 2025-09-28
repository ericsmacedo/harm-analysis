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
from numpy.lib.stride_tricks import as_strided
from numpy.typing import NDArray
from scipy import signal


def _rolling_median_vec(x, window_size):
    """Compute rolling median of a 1D array using vectorized striding.

    Parameters
    ----------
    x : ndarray
        Input 1D array.
    window_size : int
        Size of the sliding window (must be odd).

    Returns:
    -------
    medians : ndarray
        Array of the same length as x with rolling medians.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    k = window_size // 2
    # Reflect padding at both ends
    x_pad = np.r_[x[k:0:-1], x, x[-2 : -k - 2 : -1]]

    # Build a sliding window view
    shape = (len(x), window_size)
    strides = (x_pad.strides[0], x_pad.strides[0])
    windows = as_strided(x_pad, shape=shape, strides=strides)

    # Median along the window dimension
    return np.median(windows, axis=1)


def _arg_x_as_expected(value):
    """Ensure argument `x` is a 1-D C-contiguous array of dtype('float64').

    Used in `find_peaks`, `peak_prominences` and `peak_widths` to make `x`
    compatible with the signature of the wrapped Cython functions.

    Returns:
    -------
    value : ndarray
        A 1-D C-contiguous array with dtype('float64').
    """
    value = np.asarray(value, order="C", dtype=np.float64)
    if value.ndim != 1:
        raise ValueError("`x` must be a 1-D array")
    return value


def _rfft_length(n):
    """Compute the length of the real FFT (Fast Fourier Transform) result for a given
    signal length.

    Parameters
    ----------
    n : int
        The length of the input signal.

    Returns:
    -------
    int
        The length of the real FFT result for the input signal.

    Notes:
    -----
    The length of the real FFT result is determined based on the input signal length:
    - If `n` is even, the length is calculated as (n/2) + 1.
    - If `n` is odd, the length is calculated as (n+1)/2.

    Examples:
    --------
    >>> signal_length = 7
    >>> result_length = rfft_length(signal_length)
    >>> print(f"The length of np.fft.rfft(x) for a signal of length {signal_length} is: {result_length}")
    The length of np.fft.rfft(x) for a signal of length 7 is: 4.
    """  # noqa: D205
    if n % 2 == 0:
        return (n // 2) + 1
    return (n + 1) // 2


def _win_metrics(x):
    """Compute the coherent gain and the equivalent noise bandwidth of a window.

    Parameters
    ----------
    x : array_like
        Input window. The window should be normalized so that its DC value is 1.

    Returns:
    -------
    coherent_gain : float
        Gain added by the window. Equal to its DC value.
    eq_noise_bw : float
        Window equivalent noise bandwidth, normalized by noise power per bin (N0/T).
    """
    # Gain added by the window. Equal to its DC value
    coherent_gain = np.sum(x)

    # Equivalent noise bandwidth of the window, in number of FFT bins
    eq_noise_bw = np.sum(x**2) / coherent_gain**2

    return coherent_gain, eq_noise_bw


def _fft_pow(x: NDArray[np.float64], win: NDArray[np.float64], n_fft: int, fs: float = 1, coherent_gain: float = 1):
    """Calculate the single-sided power spectrum of the input signal.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal.
    win : numpy.ndarray
        Window function to be applied to the input signal `x` before
        computing the FFT.
    n_fft : int
        Number of points to use in the FFT (Fast Fourier Transform).
    fs : float, optional
        Sampling frequency of the input signal `x`. Defaults to 1.
    coherent_gain : float, optional
        Coherent gain factor applied to the FFT result. Defaults to 1.

    Returns:
    -------
    x_fft_pow : numpy.ndarray
        Single-sided power spectrum of the input signal `x`.
    f_array : numpy.ndarray
        Array of positive frequencies corresponding to the single-sided power spectrum.
    """
    # Positive frequencies of the FFT(x*win)
    x_fft = np.fft.rfft(x * win, n_fft)
    f_array = np.fft.rfftfreq(n_fft, 1 / fs)

    # Obtain absolute value and remove gain added by the window used
    x_fft_abs = np.abs(x_fft) / coherent_gain

    # Single-sided power spectrum
    x_fft_pow = x_fft_abs**2
    x_fft_pow[1:] *= 2

    return x_fft_pow, f_array


def _find_freq_bins(x_fft: NDArray[np.float64], freq: NDArray[np.float64]) -> NDArray[np.float64]:
    """Find frequency Bins of fundamental and harmonics.

    Finds the frequency bins of frequencies. The end/start of a frequency
    is found by comparing the amplitude of the bin on the right/left.
    The frequency harmonic ends when the next bin is greater than the
    current bin.

    Arguments:
    ---------
    x_fft:
        Absolute value of FFT from DC to Fs/2

    freq:
        Frequency array

    frequencies:
        List of frequencies to look for.

    Returns:
        list of bin bins index
    """
    fft_length = len(x_fft)

    # find local maximum near bin
    dist = 3
    idx0 = np.max(freq - dist, 0)
    idx1 = freq + dist + 1

    max_idx = np.argmax(x_fft[idx0:idx1])
    freq = max_idx + freq - dist

    start = freq
    end = freq
    max_n_smp = 30
    # find end of peak (right side)
    for i in range(fft_length - freq - 1):
        if x_fft[freq + i] - x_fft[freq + i + 1] <= 0 or (i > max_n_smp):
            end = freq + i + 1
            break

    # find end of peak (left side)
    for i in range(fft_length - freq - 1):
        if (x_fft[freq - i] - x_fft[freq - (i + 1)] <= 0) or (i > max_n_smp):
            start = freq - i
            break

    return np.arange(start, end).astype(int)


def _find_dc_bins(x_fft: NDArray[np.float64]) -> int | np.signedinteger:
    """Find DC bins of FFT output.

    Finds the DC bins. The end of DC is found by checking the amplitude of
    the consecutive bins. DC ends when the next bin is greater than
    the current bin.

    Parameters
    ---------
    x_fft : array_like
            Absolute value of the positive frequencies of FFT

    Returns:
        List of bins corresponding to DC

    """
    # Stop if DC is not found after 50 samples
    return np.argmax(np.diff(x_fft[:50]) > 0) + 1


def _find_bins(x, n_harm, bw_bins):
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

    Returns:
    -------
    fund_bins : ndarray
        Bins of fundamental frequency
    harm_loc : ndarray
        Locations of harmonics
    harm_bins : ndarray
        Bins of harmonics
    dc_bins : ndarray
        Bins of DC component
    noise_bins : ndarray
        Bins of noise

    Notes:
    -----
    This function only works for the right-sided power spectrum (positive frequencies
    only).
    """
    # Index of last dc bin
    dc_end = _find_dc_bins(x)
    dc_bins = np.arange(dc_end)

    if bw_bins <= dc_end:
        sys.exit("Error: max bandwidth is too low and is inside the detected dc bins.")

    # the fundamental frequency is found by searching for the bin with the
    # maximum value, excluding DC
    max_loc = np.argmax(x[dc_end:bw_bins]) + dc_end

    # list containing bins of fundamental
    fund_bins = _find_freq_bins(x, max_loc)

    # Estimate precise location using a weighted average
    fund_loc = np.average(fund_bins, weights=x[fund_bins])

    # THD+N bins (all bins excluding DC and the fundamental)
    thdn_bins = np.setdiff1d(np.arange(len(x)), np.concatenate((fund_bins, dc_bins)))

    harm_loc, harm_bins = _find_harm(x, fund_loc, n_harm, bw_bins)

    # Remaining bins are considered noise.
    if harm_bins is None:
        noise_bins = thdn_bins
    else:
        noise_bins = np.setdiff1d(thdn_bins, harm_bins)

    return fund_loc, fund_bins, harm_loc, harm_bins, dc_bins, noise_bins, thdn_bins


def _find_harm(x, fund_loc, n_harm, bw_bins):
    if n_harm <= 0:
        harm_bins = None
        harm_loc = None
    else:
        # calculate the frequency of the harmonics.
        # frequencies > fs/2 are ignored
        harm_loc = fund_loc * np.arange(2, n_harm + 2)
        harm_loc = harm_loc[harm_loc <= bw_bins]

        if harm_loc.size != 0:
            harm_bins = np.concatenate([_find_freq_bins(x, int(loc)) for loc in harm_loc])
        else:
            harm_bins = None
            harm_loc = None

    return harm_loc, harm_bins


def _power_from_bins(x_fft_pow, bins, enbw_bins, bw_bins):
    """Calculate the power given the power spectrum and an array of bins.

    Parameters
    ----------
    x_fft_pow : numpy.ndarray
        Power spectrum of the signal.
    bins : numpy.ndarray
        Frequency bins to consider for power calculation.
    enbw_bins : float
        Equivalent noise bandwidth of the system in number of bins.
    bw_bins : float
        Bandwidth in bins.

    Returns:
    -------
    float or None
        The normalized power within the specified frequency bins, or None if no valid
        bins are found.

    Notes:
    -----
    This function filters frequency bins outside the specified bandwidth (`bw_bins`)
    and calculates the power by summing the power values within the valid bins and
    dividing by the equivalent noise bandwidth (`enbw_bins`).

    If no valid bins are found within the specified bandwidth, the function returns
    None.
    """
    # Filter bins that are outside the specified bandwidth
    bins = bins[bins <= bw_bins]

    if bins.size == 0:
        return None
    return np.sum(x_fft_pow[bins]) / enbw_bins


def _mask_array(x, idx_list):
    """Mask an array so that only the values at the specified indices are valid.

    Parameters
    ----------
    x : array_like
        Input array.
    idx_list : list of int
        List of indices to keep.

    Returns:
    -------
    masked_array : MaskedArray
        A masked array with the same shape as `x`, where only the values at the
        indices in `idx_list` are valid.

    Examples:
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> idx_list = [0, 2, 4]
    >>> mask_array(x, idx_list)
    masked_array(data=[1, --, 3, --, 5],
                 mask=[False,  True, False,  True, False],
           fill_value=999999)
    """
    mask = np.zeros_like(x, dtype=bool)
    mask[idx_list] = True
    return np.ma.masked_array(x, mask=~mask)


def _int_noise_curve(x: NDArray[np.float64], noise_bins: NDArray[np.float64]):
    total_noise_array = _mask_array(x, noise_bins)
    total_int_noise = np.cumsum(total_noise_array)

    # The with statement removes warnings about divide-by-zero in the log10
    # calculation
    with np.errstate(divide="ignore"):
        return 10 * np.log10(total_int_noise)


def _find_tones(x, enbw_bins, bw_bins):
    x_db = 10 * np.log10(x)

    # TODO: decide which number is appropriate for the median window
    rol_median = _rolling_median_vec(x_db, 101) + 16

    peaks, _ = signal.find_peaks(
        x_db,
        distance=6,
        height=rol_median,
        prominence=10,
    )

    if len(peaks) > 0:
        tones_bins = np.concatenate([_find_freq_bins(x_db, int(loc)) for loc in peaks])

        breaks = np.where(np.diff(tones_bins) > 1)[0] + 1
        bin_groups = np.split(tones_bins, breaks)

        tones_loc = np.empty(len(peaks))
        tones_amp = np.empty(len(peaks))
        for i, bins in enumerate(bin_groups):
            tones_loc[i] = np.average(bins, weights=x_db[bins])
            tones_amp[i] = _power_from_bins(x, bins, enbw_bins, bw_bins)
    else:
        tones_bins = None
        tones_loc = None
        tones_amp = None

    return tones_loc, tones_amp, tones_bins


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
    enbw_bins: float,
    bw_bins: int,
    ax,
    tones_freq: None | NDArray[np.float64] = None,
    tones_bins: None | NDArray[np.float64] = None,
):
    x_db = 10 * np.log10(x)

    dc_noise_array = _mask_array(x_db, np.concatenate((dc_bins, noise_bins)))

    if tones_bins is not None:
        tones_array = _mask_array(x_db, tones_bins)
        ax.plot(freq_array, tones_array, label="Tones")

        if tones_freq is not None:
            breaks = np.where(np.diff(tones_bins) > 1)[0] + 1
            bin_groups = np.split(tones_bins, breaks)

            for i, bins in enumerate(bin_groups):
                x_marker = tones_freq[i]
                y_marker = 10 * np.log10(np.sum(x[bins] / enbw_bins))
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
    enbw_bins: float,
    bw_bins: int,
    ax,
    fund_freq: None | float = None,
    harm_freq: None | NDArray[np.float64] = None,
    fund_bins: None | NDArray[np.float64] = None,
    harm_bins: None | NDArray[np.float64] = None,
):
    x_db = 10 * np.log10(x)

    fund_array = _mask_array(x_db, fund_bins)
    dc_noise_array = _mask_array(x_db, np.concatenate((dc_bins, noise_bins)))

    if fund_bins is not None:
        ax.plot(freq_array, fund_array, label="Fundamental")

    if fund_freq is not None:
        x_marker = fund_freq
        y_marker = 10 * np.log10(np.sum(x[fund_bins] / enbw_bins))
        _annotate(ax, x_marker, y_marker, "F")

    if harm_bins is not None:
        harm_array = _mask_array(x_db, harm_bins)
        ax.plot(freq_array, harm_array, label="Harmonics")

        if harm_freq is not None:
            breaks = np.where(np.diff(harm_bins) > 1)[0] + 1
            bin_groups = np.split(harm_bins, breaks)

            for i, bins in enumerate(bin_groups):
                x_marker = harm_freq[i]
                y_marker = 10 * np.log10(np.sum(x[bins] / enbw_bins))
                _annotate(ax, x_marker, y_marker, f"{i + 1}")

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


def _harm_analysis(
    x: NDArray[np.float64],
    fs: float = 1,
    bw: float | None = None,
    window=None,
):
    sig_len = len(x)
    # length of the array returned by the np.fft.rfft function
    rfft_len = _rfft_length(sig_len)

    if window is None:
        window = signal.windows.hann(sig_len, sym=False)

    # window metrics
    coherent_gain, enbw = _win_metrics(window)
    enbw_bins = enbw * sig_len

    # Obtain the single-sided power spectrum
    x_fft_pow, f_array = _fft_pow(x=x, win=window, n_fft=sig_len, fs=fs, coherent_gain=coherent_gain)

    # Convert bw to number of bins
    if bw is None:
        bw_bins = rfft_len - 1
    else:
        bw_bins = np.argmin(np.abs(f_array - bw))

    return x_fft_pow, f_array, enbw_bins, bw_bins


def harm_analysis(  # noqa: PLR0913
    x: NDArray[np.float64],
    fs: float = 1,
    bw: float | None = None,
    n_harm: int = 5,
    window=None,
    plot=False,
    ax=None,
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
        - `thd_db`: Total harmonic distortion in decibels. Returns `numpy.nan` if
        `n_harm` is 0 or if all harmonics are outside the bandwidth.
        - `snr_db`: Signal-to-noise ratio in decibels.
        - `sinad_db`: Signal-to-noise-and-distortion ratio in decibels.
        - `thdn_db`: Total harmonic distortion plus noise in decibels.
        - `total_noise_and_dist`: Total noise and distortion in decibels.

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
    x_fft_pow, f_array, enbw_bins, bw_bins = _harm_analysis(x, fs=fs, bw=bw, window=window)

    sig_len = len(x)

    fund_loc, fund_bins, harm_loc, harm_bins, dc_bins, noise_bins, thdn_bins = _find_bins(
        x=x_fft_pow, n_harm=n_harm, bw_bins=bw_bins
    )

    fund_power = _power_from_bins(x_fft_pow, fund_bins, enbw_bins, bw_bins)
    dc_power = _power_from_bins(x_fft_pow, dc_bins, enbw_bins, bw_bins)
    noise_power = _power_from_bins(x_fft_pow, noise_bins, enbw_bins, bw_bins)

    # According to wikipedia, THD+N in dB is equal to
    # 10*log10(sum(harmonics power + Noise power)/fundamental power).
    # THD+N is recriprocal to SINAD (SINAD_dB = -THD+N_dB)
    thdn_power = _power_from_bins(x_fft_pow, thdn_bins, enbw_bins, bw_bins) / fund_power

    # total integrated noise curve
    int_noise = _int_noise_curve(x=x_fft_pow / enbw_bins, noise_bins=noise_bins)
    int_noise_p_harm = _int_noise_curve(x=x_fft_pow / enbw_bins, noise_bins=thdn_bins)

    # Calculate THD, Signal Power and N metrics in dB
    sig_freq = fund_loc * fs / sig_len
    dc_db = 10 * np.log10(dc_power)
    sig_pow_db = 10 * np.log10(fund_power)
    noise_pow_db = 10 * np.log10(noise_power)
    thdn_db = 10 * np.log10(thdn_power)
    snr_db = sig_pow_db - noise_pow_db

    # THD in dB is equal to 10*log10(sum(harmonics power)/fundamental power)
    if harm_bins is not None:
        harm_power = _power_from_bins(x_fft_pow, harm_bins, enbw_bins, bw_bins)
        thd_db = 10 * np.log10(harm_power / fund_power)
        harm_freq = harm_loc * fs / sig_len
    else:
        harm_power = np.nan
        thd_db = np.nan
        harm_freq = None

    results = {
        "fund_db": sig_pow_db,
        "fund_freq": sig_freq,
        "dc_db": dc_db,
        "noise_db": noise_pow_db,
        "thd_db": thd_db,
        "snr_db": snr_db,
        "sinad_db": -thdn_db,
        "thdn_db": thdn_db,
        "total_noise_and_dist": int_noise[-1],
    }

    if plot is False:
        return results
    ax = _plot_harm(
        x=x_fft_pow,
        fund_freq=sig_freq,
        harm_freq=harm_freq,
        freq_array=f_array,
        dc_bins=dc_bins,
        fund_bins=fund_bins,
        harm_bins=harm_bins,
        noise_bins=noise_bins,
        ax=ax,
        int_noise=int_noise,
        int_noise_p_harm=int_noise_p_harm,
        enbw_bins=enbw_bins,
        bw_bins=bw_bins,
    )

    return results, ax


def spec_analysis(  # noqa: PLR0913
    x: NDArray[np.float64],
    fs: float = 1,
    bw: float | None = None,
    window: None | NDArray[np.float64] = None,
    plot=False,
    ax: None | Axes = None,
):
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
    bw : float, optional
        Bandwidth to use for the calculation of the metrics, in same units as fs.
    plot : bool or None, optional
        If True, the power spectrum result is plotted. If specified,
        an `ax` must be provided, and the function returns a dictionary
        with the results and the specified axes (`ax`). If plot is not set,
        only the results are returned.
    ax : plt.Axes or None, optional
        Axes to be used for plotting. Required if plot is set to True.

    Returns
    -------
    properties : dict
        A dictionary containing the analysis results

        - `dc_db`: DC power in decibels (dc_db),
        - `noise_db`: Noise power in decibels (noise_pow_db),

    plt.axes
        If plot is set to True, the Axes used for plotting is returned.

    """  # noqa: D416
    x_fft_pow, f_array, enbw_bins, bw_bins = _harm_analysis(x, fs=fs, bw=bw, window=window)
    sig_len = len(x)

    dc_end = _find_dc_bins(x_fft_pow)
    dc_bins = np.arange(dc_end)

    tones_loc, tones_amp, tones_bins = _find_tones(x_fft_pow, enbw_bins=enbw_bins, bw_bins=bw_bins)

    if tones_loc is not None:
        tones_freq = tones_loc * fs / sig_len
        tones_amp_db = 10 * np.log10(tones_amp)
    else:
        tones_freq = None
        tones_amp_db = None

    # Obtain noise bins, by removing the DC bins from the bin list
    if tones_bins is not None:
        noise_bins = np.setdiff1d(np.arange(len(x_fft_pow)), np.concatenate((dc_bins, tones_bins)))
    else:
        noise_bins = np.setdiff1d(np.arange(len(x_fft_pow)), dc_bins)

    dc_power = _power_from_bins(x_fft_pow, dc_bins, enbw_bins, bw_bins)
    noise_power = _power_from_bins(x_fft_pow, noise_bins, enbw_bins, bw_bins)

    # total integrated noise curve
    int_noise = _int_noise_curve(x=x_fft_pow / enbw_bins, noise_bins=noise_bins)

    # Calculate THD, Signal Power and N metrics in dB
    dc_db = 10 * np.log10(dc_power)
    dc = 10 ** (dc_db / 20)
    noise_pow_db = 10 * np.log10(noise_power)

    results = {
        "dc": dc,
        "dc_db": dc_db,
        "noise_db": noise_pow_db,
        "tones_amp_db": tones_amp_db,
        "tones_freq": tones_freq,
    }

    if plot is False:
        return results
    ax = _plot_spec(
        x=x_fft_pow,
        freq_array=f_array,
        dc_bins=dc_bins,
        noise_bins=noise_bins,
        ax=ax,
        int_noise=int_noise,
        enbw_bins=enbw_bins,
        bw_bins=bw_bins,
        tones_freq=tones_freq,
        tones_bins=tones_bins,
    )

    return results, ax
