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

"""Tests for even and odd signal lengths, and for components at fs/2 (#49)."""

import numpy as np
import pytest

from harm_analysis import harm_analysis, spec_analysis
from harm_analysis._spectrum import _power_spectrum

EVEN_ODD = [64, 65, 1000, 1001, 4096, 4095]


def _parity(n):
    return f"n{n}_{'even' if n % 2 == 0 else 'odd'}"


@pytest.mark.parametrize("n", EVEN_ODD, ids=_parity)
def test_single_sided_spectrum_parseval(n):
    """The single-sided power spectrum preserves the signal power exactly (Parseval).

    With a rectangular window, the sum of the single-sided power spectrum divided by the
    equivalent noise bandwidth equals the mean square of the signal, for even lengths (which
    have a bin at fs/2 that must not be doubled) and odd lengths (which don't).
    """
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, n) + np.cos(np.pi * np.arange(n))  # includes a component at fs/2

    power, _, enbw_bins, _ = _power_spectrum(x, window=np.ones(n))

    assert np.isclose(np.sum(power) / enbw_bins, np.mean(x**2), rtol=1e-12)


@pytest.mark.parametrize("phase", [0, np.pi / 4, np.pi / 3])
def test_tone_at_nyquist(phase):
    """A tone exactly at fs/2 has the power of its samples, A^2 cos^2(phase), at exactly fs/2."""
    n = 4096
    t = np.arange(n)
    noise = np.random.default_rng(0).normal(0, 1e-4, n)
    x = 0.5 * np.cos(2 * np.pi * 0.1003 * t) + np.cos(np.pi * t + phase) + noise

    results = spec_analysis(x)

    assert results.tones_freq[-1] == 0.5
    assert np.isclose(results.tones_db[-1], 10 * np.log10(np.cos(phase) ** 2), atol=0.01)


def test_harmonic_at_nyquist():
    """A harmonic exactly at fs/2 (fundamental at fs/4) is measured correctly."""
    n = 4096
    t = np.arange(n)
    noise = np.random.default_rng(0).normal(0, 1e-4, n)
    x = np.cos(2 * np.pi * 0.25 * t) + 0.01 * np.cos(2 * np.pi * 0.5 * t) + noise

    # filterwarnings = error: also checks that no "masked element to nan" warning is raised
    results = harm_analysis(x, n_harm=1)

    assert results.harm_freq[0] == 0.5
    assert np.isclose(results.harm_db[0], -40, atol=0.01)  # 0.01**2 at fs/2 (cosine phase)
    assert np.isclose(results.thd_db, -40 - 10 * np.log10(0.5), atol=0.01)  # fundamental power 1/2
    assert np.isfinite(results.total_noise_and_dist_db)


@pytest.mark.parametrize("n", [1000, 1001, 4096, 4095], ids=_parity)
def test_harm_analysis_even_and_odd_lengths(n):
    """Tone, harmonic and THD are correct for even and odd lengths."""
    t = np.arange(n)
    # Low noise, so it neither shifts the harmonic level nor hides behind window leakage
    x = (
        np.cos(2 * np.pi * 0.1003 * t)
        + 0.01 * np.cos(2 * np.pi * 0.2006 * t)
        + np.random.default_rng(0).normal(0, 1e-5, n)
    )

    results = harm_analysis(x, n_harm=1)

    assert np.isclose(results.fund_db, 10 * np.log10(0.5), atol=0.01)
    assert np.isclose(results.harm_db[0], 10 * np.log10(0.01**2 / 2), atol=0.01)
    assert np.isclose(results.thd_db, -40, atol=0.01)


@pytest.mark.parametrize("n", [1000, 1001, 4096, 4095], ids=_parity)
def test_noise_power_even_and_odd_lengths(n):
    """White noise power is correct for even and odd lengths."""
    t = np.arange(n)
    # Noise well above the Hann window's leakage of the tone
    x = np.cos(2 * np.pi * 0.1003 * t) + np.random.default_rng(0).normal(0, 1e-2, n)

    results = harm_analysis(x, n_harm=0)

    # estimate from n samples: statistical spread of about 0.15 dB
    assert np.isclose(results.noise_db, -40, atol=0.25)


@pytest.mark.parametrize("n", [1000, 1001, 4096, 4095], ids=_parity)
def test_spec_analysis_even_and_odd_lengths(n):
    """Tone levels and frequencies are correct for even and odd lengths."""
    t = np.arange(n)
    x = (
        0.1
        + np.cos(2 * np.pi * 0.1003 * t)
        + 0.1 * np.cos(2 * np.pi * 0.3507 * t)
        + np.random.default_rng(0).normal(0, 1e-4, n)
    )

    results = spec_analysis(x)

    assert np.allclose(results.tones_freq, [0.1003, 0.3507], atol=0.2 / n)
    assert np.allclose(results.tones_db, 10 * np.log10([0.5, 0.1**2 / 2]), atol=0.05)
    assert np.isclose(results.dc_db, 20 * np.log10(0.1), atol=0.01)
