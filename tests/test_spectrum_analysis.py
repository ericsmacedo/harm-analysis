"""Tests spectrum analysis function."""
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

import logging
import warnings
from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
from pytest import mark, raises

from harm_analysis import SpecResult, spec_analysis
from harm_analysis._spectrum import _to_db

rng = np.random.default_rng(1234567890)


def test_spec_analysis():
    """Test for spec_analysis function."""
    # test signal
    n = 2**18
    fs = 1000
    t = np.arange(0, n / fs, 1 / fs)

    noise_pow_db = -70
    harm_pow = 0.01**2 / 2
    thdn = 10 ** (noise_pow_db / 10) + harm_pow
    thdn_db = 10 * np.log10(thdn)

    noise_std = 10 ** (noise_pow_db / 20)

    dc_level = 0.123456789
    dc_power_db = 20 * np.log10(dc_level)
    noise = rng.normal(loc=0, scale=noise_std, size=len(t))

    f1 = 100.13

    x = (
        dc_level
        + 2 * np.cos(2 * np.pi * f1 * t)
        + 0.01 * np.cos(2 * np.pi * f1 * 2 * t)
        + 3 * np.cos(2 * np.pi * f1 * 3 * t)
        + 3 * np.cos(2 * np.pi * fs / 2 * t)
        + noise
    )

    results = spec_analysis(x, fs=fs)

    print("Function results:")
    for key, value in results.to_dict().items():
        print(f"{key.ljust(10)} [dB]: {value}")

    logging.info(
        "\n"
        "Expected values\n"
        f"    Total noise (dB): {thdn_db}\n"
        f"    DC power [dB]: {dc_power_db}\n"
        f"    DC level: {dc_level}\n"
    )

    tolerance = 0.3

    amp_arr = np.asarray([2, 0.01, 3])

    amp_arr_db = 10 * np.log10((amp_arr**2) / 2)

    sig_fs2 = 3 * np.cos(2 * np.pi * fs / 2 * t)
    pow_fs2 = np.mean(sig_fs2**2)
    amp_arr_db = np.append(amp_arr_db, 10 * np.log10(pow_fs2))

    assert np.isclose(results.dc, dc_level, rtol=tolerance)
    assert np.isclose(results.dc_db, dc_power_db, rtol=tolerance)
    assert np.isclose(results.noise_db, noise_pow_db, rtol=tolerance)
    assert np.allclose(results.tones_freq, np.asarray([f1, 2 * f1, 3 * f1, fs / 2]), rtol=tolerance)
    assert np.allclose(results.tones_db, amp_arr_db, rtol=tolerance)


@mark.parametrize("plot_en", [True, False])
def test_spec_analysis_plot(plot_en):
    """Test for spec_analysis function.

    Checks if the function can obtain results with less than 0.1 dB of error.
    """
    # test signal
    n = 4096
    fs = 1000
    t = np.arange(0, n / fs, 1 / fs)
    f1 = 100.13

    noise = rng.normal(loc=0, scale=10 ** (-70 / 20), size=len(t))

    # Test signal
    # Tone with harmonics, DC and white gaussian noise
    x = (
        noise
        + 0.1234
        + 2 * np.cos(2 * np.pi * f1 * t)
        + 0.01 * np.cos(2 * np.pi * f1 * 2 * t)
        + 0.005 * np.cos(2 * np.pi * f1 * 3 * t)
        + 0.01 * np.cos(2 * np.pi * f1 * 0.43 * t)
    )

    # Use the harm_analysis function
    if plot_en:
        fig, ax = plt.subplots()
        spec_analysis(x, fs=fs, plot=True, ax=ax)
    else:
        spec_analysis(x, fs=fs)


@mark.filterwarnings("error")
def test_spec_analysis_zero_signal():
    """An all-zero signal reports -inf dB instead of raising warnings."""
    results = spec_analysis(np.zeros(2048))

    assert results.dc_db == -np.inf
    assert results.noise_db == -np.inf
    assert results.tones_db.size == 0


def test_to_db_zero():
    """_to_db returns -inf for zero power without raising warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _to_db(0.0) == -np.inf
        assert _to_db(1.0) == 0.0
        assert _to_db(np.array([0.0, 10.0])).tolist() == [-np.inf, 10.0]


def test_spec_analysis_returns_spec_result():
    """spec_analysis returns a SpecResult whether or not it plots."""
    n = 2048
    x = 0.1 + np.cos(2 * np.pi * 101.3 * np.arange(n) / n) + rng.normal(0, 1e-4, n)

    results = spec_analysis(x)
    assert isinstance(results, SpecResult)
    assert list(results.to_dict()) == [f.name for f in fields(SpecResult)]
    assert isinstance(results.dc_db, float)
    assert isinstance(results.tones_freq, np.ndarray)

    fig, ax = plt.subplots()
    assert isinstance(spec_analysis(x, plot=True, ax=ax), SpecResult)
    assert ax.lines
    plt.close(fig)


def test_spec_analysis_plot_requires_ax():
    """plot=True without an ax raises a clear error."""
    with raises(ValueError, match="`ax` must be provided"):
        spec_analysis(np.zeros(2048) + 1, plot=True)
