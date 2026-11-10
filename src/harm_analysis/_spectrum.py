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
from scipy import signal

MIN_SIGNAL_LENGTH = 32


def _validate_inputs(x, fs, window=None, bw=None, n_harm=None) -> NDArray[np.float64]:
    """Check the inputs of the analysis functions and return `x` as a float array.

    Raises:
    ------
    TypeError
        If `x` (or `window`) is complex or not numeric.
    ValueError
        If `x` is not 1-D, is shorter than ``MIN_SIGNAL_LENGTH`` samples or contains
        NaN or infinite values; if `fs` is not a positive finite number; if `window`
        does not have the same length as `x` or is not finite; if `bw` is not in
        ``(0, fs / 2]``; or if `n_harm` is negative.
    """
    x = _as_real_array(x, "x")
    if x.ndim != 1:
        raise ValueError(f"`x` must be a 1-D array, got an array with shape {x.shape}.")
    if x.size < MIN_SIGNAL_LENGTH:
        raise ValueError(f"`x` must have at least {MIN_SIGNAL_LENGTH} samples, got {x.size}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("`x` contains NaN or infinite values.")

    if not (np.isfinite(fs) and fs > 0):
        raise ValueError(f"`fs` must be a positive finite number, got {fs}.")

    if window is not None:
        window = _as_real_array(window, "window")
        if window.shape != x.shape:
            raise ValueError(f"`window` must have the same length as `x` ({x.size}), got shape {window.shape}.")
        if not np.all(np.isfinite(window)):
            raise ValueError("`window` contains NaN or infinite values.")

    if bw is not None and not (np.isfinite(bw) and 0 < bw <= fs / 2):
        raise ValueError(f"`bw` must be in (0, fs / 2] = (0, {fs / 2}], got {bw}.")

    if n_harm is not None and n_harm < 0:
        raise ValueError(f"`n_harm` must be 0 or positive, got {n_harm}.")

    return x


def _as_real_array(value, name: str) -> NDArray[np.float64]:
    """Convert to a float array, rejecting complex and non-numeric input."""
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise TypeError(f"`{name}` must be real; complex input is not supported.")
    try:
        return arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"`{name}` must be numeric, got dtype {arr.dtype}.") from exc


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


def _find_freq_bins(x: NDArray[np.float64], idx: int, bw_bins: int, enbw_bins: float):
    """Find frequency Bins of fundamental and harmonics.

    Finds the frequency bins of frequencies. The end/start of a frequency
    is found by comparing the amplitude of the bin on the right/left.
    The frequency harmonic ends when the next bin is greater than the
    current bin.

    Arguments:
    ---------
    x: NDArray
        Input signal

    idx: int
        index indicating where to look for local maxima

    bw_bins: int
        Bandwidth to consider when calculating parameters

    enbw_bins: float
        Equivalent noise bandwidth of the window used, in bins.

    Returns:
        bins: NDArray
            list of bins of peak

        peak_loc: float
            Location of the bin. Calculated using a weighted average of the peak bins

        peak_val: float
            power of the peak.

    """
    x_size = x.size

    # find local max near bin
    width = 3
    search_start = max(0, idx - width)
    search_end = min(x_size, idx + width + 1)
    local_max_idx = int(np.argmax(x[search_start:search_end]) + search_start)

    # Search for start/end of the peak
    start_idx = max(0, local_max_idx - 100)
    end_idx = min(x_size, local_max_idx + 100)

    if local_max_idx == 0:
        peak_start = 0
    else:
        d_start = np.diff(x[start_idx:local_max_idx])
        starts = np.where(d_start < 0)[0]
        if starts.size == 0:
            peak_start = start_idx
        else:
            peak_start = starts[-1] + start_idx + 1

    if local_max_idx == x_size - 1:
        peak_end = x_size
    else:
        d_end = np.diff(x[local_max_idx:end_idx])
        ends = np.where(d_end > 0)[0]
        if ends.size == 0:
            peak_end = end_idx
        else:
            peak_end = ends[0] + local_max_idx + 1

    bins = np.arange(peak_start, peak_end)

    # Estimate precise location using a weighted average
    peak_loc = np.average(bins, weights=x[bins])
    peak_val = _power_from_bins(x, bins, enbw_bins=enbw_bins, bw_bins=bw_bins)

    return peak_val, peak_loc, bins


def _find_dc_bins(x_fft: NDArray[np.float64]) -> int | np.signedinteger:
    """Find DC bins of FFT output.

    Finds the DC bins. The end of DC is found by checking the amplitude of
    the consecutive bins. DC ends when the next bin is greater than
    the current bin.

    The single-sided power spectrum doubles every bin except DC (bin 0). That
    doubling is undone before comparing the bins: otherwise windows with a flat
    main lobe (e.g. flat-top) look like they rise from bin 0 to bin 1, and only the
    first DC bin would be found.

    Parameters
    ---------
    x_fft : array_like
            Absolute value of the positive frequencies of FFT

    Returns:
        List of bins corresponding to DC

    """
    # Stop if DC is not found after 50 samples
    two_sided = np.array(x_fft[:50], dtype=np.float64)
    two_sided[1:] /= 2
    return np.argmax(np.diff(two_sided) > 0) + 1


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
    >>> _mask_array(x, idx_list)
    masked_array(data=[1, --, 3, --, 5],
                 mask=[False,  True, False,  True, False],
           fill_value=999999)
    """
    mask = np.zeros_like(x, dtype=bool)
    mask[idx_list] = True
    return np.ma.masked_array(x, mask=~mask)


def _to_db(power):
    """Convert power to decibels, returning ``-inf`` (without warnings) for a power of 0."""
    with np.errstate(divide="ignore"):
        return 10 * np.log10(power)


def _int_noise_curve(x: NDArray[np.float64], noise_bins: NDArray[np.integer]):
    total_noise_array = _mask_array(x, noise_bins)
    total_int_noise = np.cumsum(total_noise_array)

    return _to_db(total_int_noise)


def _power_spectrum(
    x: NDArray[np.float64],
    fs: float = 1,
    bw: float | None = None,
    window: None | NDArray[np.float64] = None,
):
    sig_len = len(x)

    if window is None:
        window = signal.windows.hann(sig_len, sym=False)

    # window metrics
    coherent_gain, enbw = _win_metrics(window)
    enbw_bins = enbw * sig_len

    # Obtain the single-sided power spectrum
    x_fft_pow, f_array = _fft_pow(x=x, win=window, n_fft=sig_len, fs=fs, coherent_gain=coherent_gain)

    # Convert bw to number of bins
    if bw is None:
        bw_bins = x_fft_pow.size - 1
    else:
        bw_bins = np.argmin(np.abs(f_array - bw))

    return x_fft_pow, f_array, enbw_bins, bw_bins
