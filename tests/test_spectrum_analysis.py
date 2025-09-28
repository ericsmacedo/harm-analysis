"""Tests spectrum analysis function."""
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

import matplotlib.pyplot as plt
import numpy as np
from pytest import mark

from harm_analysis import dc_measurement

rng = np.random.default_rng()


@mark.parametrize("plot_en", [True, False])
def test_harm_analysis(plot_en):
    """Test for harm_analysis function.

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
        dc_measurement(x, fs=fs, plot=True, ax=ax)
    else:
        dc_measurement(x, fs=fs)

    # TODO: add assertions
    # assert np.isclose(results["fund_db"], fund_pow_db, rtol=tolerance)
    # assert np.isclose(results["fund_freq"], f1, rtol=tolerance)
    # assert np.isclose(results["dc_db"], dc_power_db, rtol=tolerance)
    # assert np.isclose(results["noise_db"], noise_pow_db, rtol=tolerance)
    # assert np.isclose(results["thd_db"], thd_db, rtol=tolerance)
    # assert np.isclose(results["thdn_db"], thdn_db, rtol=tolerance)
