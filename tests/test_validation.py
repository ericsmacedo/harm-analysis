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

"""Tests for the input validation of harm_analysis and spec_analysis."""

import numpy as np
import pytest

from harm_analysis import harm_analysis, spec_analysis

N = 1024
TONE = np.cos(2 * np.pi * 0.1013 * np.arange(N)) + np.random.default_rng(0).normal(0, 1e-4, N)

FUNCS = [harm_analysis, spec_analysis]


def _with_sample(value):
    x = TONE.copy()
    x[5] = value
    return x


@pytest.mark.parametrize("func", FUNCS)
@pytest.mark.parametrize(
    ("x", "kwargs", "match"),
    [
        (TONE.reshape(2, -1), {}, "1-D"),
        (TONE[:16], {}, "at least 32 samples"),
        (_with_sample(np.nan), {}, "NaN or infinite"),
        (_with_sample(np.inf), {}, "NaN or infinite"),
        (TONE, {"fs": 0}, "positive finite"),
        (TONE, {"fs": -1}, "positive finite"),
        (TONE, {"fs": np.inf}, "positive finite"),
        (TONE, {"window": np.ones(10)}, "same length"),
        (TONE, {"window": np.full(N, np.nan)}, "window` contains NaN"),
    ],
    ids=["2d", "too_short", "nan", "inf", "fs_zero", "fs_negative", "fs_inf", "window_length", "window_nan"],
)
def test_invalid_inputs(func, x, kwargs, match):
    """Invalid inputs raise a ValueError with a clear message."""
    with pytest.raises(ValueError, match=match):
        func(x, **kwargs)


@pytest.mark.parametrize("func", FUNCS)
@pytest.mark.parametrize("x", [TONE + 0j, np.array(["a"] * N)], ids=["complex", "strings"])
def test_non_real_input(func, x):
    """Complex and non-numeric inputs raise a TypeError."""
    with pytest.raises(TypeError, match="must be"):
        func(x)


@pytest.mark.parametrize(("bw", "match"), [(0, "bw"), (-1, "bw"), (0.6, "bw"), (np.nan, "bw")])
def test_harm_analysis_invalid_bw(bw, match):
    """Bw must be in (0, fs / 2]."""
    with pytest.raises(ValueError, match=match):
        harm_analysis(TONE, bw=bw)


def test_harm_analysis_negative_n_harm():
    """n_harm must not be negative."""
    with pytest.raises(ValueError, match="n_harm"):
        harm_analysis(TONE, n_harm=-1)


@pytest.mark.parametrize("func", FUNCS)
def test_valid_inputs_still_accepted(func):
    """Lists, integer arrays, a custom window and bw = fs / 2 are accepted."""
    func(list(TONE))
    func(np.round(TONE * 1000).astype(int))
    func(TONE, window=np.hanning(N))
    func(TONE, fs=1000.0)
    if func is harm_analysis:
        func(TONE, fs=1000.0, bw=500.0)
