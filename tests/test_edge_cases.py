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

"""Edge-case tests for harm_analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import signal

from harm_analysis import harm_analysis

N = 4096
T = np.arange(N)
H2_AMP = 0.01
H3_AMP = 0.005
# THD of a unit tone with 2nd and 3rd harmonics of H2_AMP and H3_AMP
THD_DB = 10 * np.log10(H2_AMP**2 + H3_AMP**2)


def _signal(f, amp=1.0, noise_std=1e-5):
    """Tone at normalized frequency f with 2nd and 3rd harmonics and white noise."""
    rng = np.random.default_rng(0)
    x = amp * (
        np.cos(2 * np.pi * f * T) + H2_AMP * np.cos(2 * np.pi * 2 * f * T) + H3_AMP * np.cos(2 * np.pi * 3 * f * T)
    )
    return x + rng.normal(0, amp * noise_std, N)


def test_harmonics_above_nyquist_are_ignored():
    """Harmonics above fs / 2 are not analyzed (no aliasing back into the band)."""
    results = harm_analysis(_signal(0.15), n_harm=5)

    # 2nd (0.30) and 3rd (0.45) harmonics are in band; 4th to 6th (0.60 to 0.90) are not
    assert np.allclose(results.harm_freq, [0.30, 0.45], atol=1e-3)
    assert results.harm_db.size == 2


def test_harmonics_above_bw_are_ignored():
    """Only harmonics up to bw contribute to THD."""
    results = harm_analysis(_signal(0.0513), n_harm=5, bw=0.11)

    assert np.allclose(results.harm_freq, [2 * 0.0513], atol=1e-3)
    assert np.isclose(results.thd_db, 20 * np.log10(H2_AMP), atol=0.1)


@pytest.mark.parametrize("window_name", ["hann", "blackmanharris", "flattop"])
def test_custom_windows(window_name):
    """Different windows give the same fundamental power and THD."""
    window = signal.windows.get_window(window_name, N, fftbins=True)
    results = harm_analysis(_signal(0.0513), window=window)

    assert np.isclose(results.fund_db, 10 * np.log10(0.5), atol=0.05)
    assert np.isclose(results.thd_db, THD_DB, atol=0.1)


@pytest.mark.parametrize("amp", [1e-3, 1.0, 1e3])
def test_amplitude_scaling(amp):
    """Absolute powers follow the amplitude; ratios (THD, SNR) do not change."""
    reference = harm_analysis(_signal(0.0513))
    results = harm_analysis(_signal(0.0513, amp=amp))

    assert np.isclose(results.fund_db, 20 * np.log10(amp / np.sqrt(2)), atol=0.05)
    assert np.isclose(results.thd_db, reference.thd_db, atol=1e-6)
    assert np.isclose(results.snr_db, reference.snr_db, atol=1e-6)


def test_sampling_frequency_scaling():
    """Fs scales the reported frequencies and leaves the powers unchanged."""
    x = _signal(0.0513)
    fs = 48e3
    normalized = harm_analysis(x)
    scaled = harm_analysis(x, fs=fs)

    assert np.isclose(scaled.fund_freq, normalized.fund_freq * fs)
    assert np.allclose(scaled.harm_freq, normalized.harm_freq * fs)
    assert np.isclose(scaled.fund_db, normalized.fund_db)
    assert np.isclose(scaled.snr_db, normalized.snr_db)
    assert np.isclose(scaled.thd_db, normalized.thd_db)


@pytest.mark.parametrize(
    ("f", "bw"),
    [(0.0513, 0.08), (0.3007, None)],
    ids=["harmonics_above_bw", "harmonics_above_nyquist"],
)
def test_all_harmonics_outside_band(f, bw):
    """With no harmonic in the band, THD is NaN and the harmonic arrays are empty (#39)."""
    results = harm_analysis(_signal(f), bw=bw)

    assert np.isnan(results.thd_db)
    assert results.harm_freq.size == 0
    assert results.harm_db.size == 0
    assert np.isclose(results.fund_freq, f, atol=1e-3)


def test_all_harmonics_outside_band_plot():
    """Plotting works when there is no harmonic in the band (#39)."""
    fig, ax = plt.subplots()
    harm_analysis(_signal(0.3007), plot=True, ax=ax)
    assert ax.lines
    plt.close(fig)


def _with_interferer(f_tone, f_interferer, n=N, seed=0):
    """Unit tone plus an unrelated tone twice as large, and white noise."""
    t = np.arange(n)
    x = np.cos(2 * np.pi * f_tone * t) + 2 * np.cos(2 * np.pi * f_interferer * t)
    return x + np.random.default_rng(seed).normal(0, 1e-5, n)


def test_stronger_interferer_is_the_default_fundamental():
    """Without fund_freq, the highest component is taken as the fundamental (unchanged behavior)."""
    results = harm_analysis(_with_interferer(0.1003, 0.1507))

    assert np.isclose(results.fund_freq, 0.1507, atol=1e-4)


@pytest.mark.parametrize(
    ("f_tone", "f_interferer"),
    [(0.1003, 0.1507), (0.2003, 0.1207)],
    ids=["interferer_above", "interferer_below"],
)
def test_fund_freq_selects_the_injected_tone(f_tone, f_interferer):
    """fund_freq analyzes the injected tone even next to a stronger interferer (#42)."""
    results = harm_analysis(_with_interferer(f_tone, f_interferer), fund_freq=f_tone)

    assert np.isclose(results.fund_freq, f_tone, atol=1e-4)
    assert np.isclose(results.fund_db, 10 * np.log10(0.5), atol=0.05)
    # the interferer (4x the power) is counted as noise
    assert np.isclose(results.snr_db, -10 * np.log10(4), atol=0.1)


def test_fund_freq_tolerates_a_slightly_off_value():
    """fund_freq only needs to be within a few bins of the tone; the result uses the measured frequency."""
    results = harm_analysis(_with_interferer(0.1003, 0.1507), fund_freq=0.1003 + 2 / N)

    assert np.isclose(results.fund_freq, 0.1003, atol=1e-5)


def test_fund_freq_with_bw_excludes_the_interferer():
    """fund_freq and bw together: the interferer is outside the band, so it does not reduce the SNR."""
    results = harm_analysis(_with_interferer(0.1003, 0.1507), fund_freq=0.1003, bw=0.125)

    assert np.isclose(results.fund_freq, 0.1003, atol=1e-4)
    assert results.snr_db > 70


@pytest.mark.parametrize(
    ("fund_freq", "kwargs", "match"),
    [
        (0, {}, r"must be in \(0, fs / 2\]"),
        (0.6, {}, r"must be in \(0, fs / 2\]"),
        (np.nan, {}, r"must be in \(0, fs / 2\]"),
        (0.13, {"bw": 0.12}, r"must be in \(0, bw\]"),
        (0.3, {}, "No tone found near `fund_freq`"),
        (0.1003 + 5 / N, {}, "No spectral peak found within 3 bins"),
    ],
    ids=["zero", "above_nyquist", "nan", "above_bw", "only_noise", "too_far_from_tone"],
)
def test_fund_freq_errors(fund_freq, kwargs, match):
    """Invalid or wrong fund_freq values raise a clear ValueError instead of analyzing noise."""
    with pytest.raises(ValueError, match=match):
        harm_analysis(_with_interferer(0.1003, 0.1507), fund_freq=fund_freq, **kwargs)


def test_fund_freq_inside_dc():
    """fund_freq inside the DC component raises a ValueError."""
    x = 0.5 + _signal(0.0513)
    with pytest.raises(ValueError, match=r"`fund_freq` .* is inside the DC component"):
        harm_analysis(x, fund_freq=1 / N)
