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

"""Tests the harmonic analysis function."""

import logging

import numpy as np

from harm_analysis import dc_measurement, harm_analysis

rng = np.random.default_rng()


def test_harm_analysis():
    """Test for harm_analysis function.

    Checks if the function can obtain results with less than 0.1 dB of error.
    """
    # test signal
    n = 2048
    fs = 1000
    t = np.arange(0, n / fs, 1 / fs)

    noise_pow_db = -70
    noise_std = 10 ** (noise_pow_db / 20)
    dc_level = 0.123456789
    dc_power_db = 20 * np.log10(dc_level)
    noise = rng.normal(loc=0, scale=noise_std, size=len(t))

    f1 = 100.13

    x = (
        dc_level
        + 2 * np.cos(2 * np.pi * f1 * t)
        + 0.01 * np.cos(2 * np.pi * f1 * 2 * t)
        + 0.005 * np.cos(2 * np.pi * f1 * 3 * t)
        + noise
    )

    fund_pow_db = 10 * np.log10(2**2 / 2)
    harm_power = (0.01**2) / 2 + (0.005**2) / 2
    thd_db = 10 * np.log10(harm_power) - fund_pow_db
    snr_db = fund_pow_db - noise_pow_db
    thdn_db = 10 * np.log10(10 ** (noise_pow_db / 10) + 10 ** (thd_db / 10))

    results = harm_analysis(x, fs=fs)

    print("Function results:")
    for key, value in results.items():
        print(f"{key.ljust(10)} [dB]: {value}")

    logging.info(
        "\n"
        "Expected values\n"
        f"    Fundamental power (dB): {fund_pow_db}\n"
        f"    Fundamental Freq [Hz]: {f1}\n"
        f"    Noise power (dB): {noise_pow_db}\n"
        f"    DC power [dB]: {dc_power_db}\n"
        f"    DC level: {dc_level}\n"
        f"    THD (dB): {thd_db}\n"
        f"    SNR (dB): {snr_db}\n"
        f"    THD+N (dB): {thdn_db}"
    )

    tolerance = 0.3

    assert np.isclose(results["fund_db"], fund_pow_db, rtol=tolerance)
    assert np.isclose(results["fund_freq"], f1, rtol=tolerance)
    assert np.isclose(results["dc_db"], dc_power_db, rtol=tolerance)
    assert np.isclose(results["noise_db"], noise_pow_db, rtol=tolerance)
    assert np.isclose(results["thd_db"], thd_db, rtol=tolerance)
    assert np.isclose(results["thdn_db"], thdn_db, rtol=tolerance)


def test_harm_analysis_dc():
    """Test for harm_analysis function.

    Checks if the function can operate when DC frequency component is highest than the
    fundamental.
    """
    # test signal
    n = 2048
    fs = 1000
    t = np.arange(0, n / fs, 1 / fs)

    noise_pow_db = -70
    noise_std = 10 ** (noise_pow_db / 20)
    dc_level = 2.123456789
    dc_power_db = 20 * np.log10(dc_level)
    noise = rng.normal(loc=0, scale=noise_std, size=len(t))

    f1 = 100.13

    x = (
        dc_level
        + 2 * np.cos(2 * np.pi * f1 * t)
        + 0.01 * np.cos(2 * np.pi * f1 * 2 * t)
        + 0.005 * np.cos(2 * np.pi * f1 * 3 * t)
        + noise
    )

    fund_pow_db = 10 * np.log10(2**2 / 2)
    harm_power = (0.01**2) / 2 + (0.005**2) / 2
    thd_db = 10 * np.log10(harm_power) - fund_pow_db
    snr_db = fund_pow_db - noise_pow_db
    thdn_db = 10 * np.log10(10 ** (noise_pow_db / 10) + 10 ** (thd_db / 10))

    results = harm_analysis(x, fs=fs)

    print("Function results:")
    for key, value in results.items():
        print(f"{key.ljust(10)} [dB]: {value}")

    logging.info(
        "\n"
        "Expected values\n"
        f"    Fundamental power (dB): {fund_pow_db}\n"
        f"    Fundamental Freq [Hz]: {f1}\n"
        f"    Noise power (dB): {noise_pow_db}\n"
        f"    DC power [dB]: {dc_power_db}\n"
        f"    DC level: {dc_level}\n"
        f"    THD (dB): {thd_db}\n"
        f"    SNR (dB): {snr_db}\n"
        f"    THD+N (dB): {thdn_db}"
    )

    tolerance = 0.3

    assert np.isclose(results["fund_db"], fund_pow_db, rtol=tolerance)
    assert np.isclose(results["fund_freq"], f1, rtol=tolerance)
    assert np.isclose(results["dc_db"], dc_power_db, rtol=tolerance)
    assert np.isclose(results["noise_db"], noise_pow_db, rtol=tolerance)
    assert np.isclose(results["thd_db"], thd_db, rtol=tolerance)
    assert np.isclose(results["thdn_db"], thdn_db, rtol=tolerance)


def test_harm_analysis_bw():
    """Test for harm_analysis function."""
    # test signal
    n = 2**18
    fs = 1000
    t = np.arange(0, n / fs, 1 / fs)
    bw = 250

    noise_pow_db = -70
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
        + noise
    )

    fund_pow_db = 10 * np.log10(2**2 / 2)
    harm_power = (0.01**2) / 2
    thd_db = 10 * np.log10(harm_power) - fund_pow_db
    snr_db = fund_pow_db - (noise_pow_db - 3)
    thdn_db = 10 * np.log10(10 ** ((noise_pow_db - 3) / 10) + 10 ** (thd_db / 10))

    results = harm_analysis(x, bw=bw, fs=fs)

    print("Function results:")
    for key, value in results.items():
        print(f"{key.ljust(10)} [dB]: {value}")

    logging.info(
        "\n"
        "Expected values\n"
        f"    Fundamental power (dB): {fund_pow_db}\n"
        f"    Fundamental Freq [Hz]: {f1}\n"
        f"    Noise power (dB): {noise_pow_db}\n"
        f"    DC power [dB]: {dc_power_db}\n"
        f"    DC level: {dc_level}\n"
        f"    THD (dB): {thd_db}\n"
        f"    SNR (dB): {snr_db}\n"
        f"    THD+N (dB): {thdn_db}"
    )

    tolerance = 0.3

    assert np.isclose(results["fund_db"], fund_pow_db, rtol=tolerance)
    assert np.isclose(results["fund_freq"], f1, rtol=tolerance)
    assert np.isclose(results["dc_db"], dc_power_db, rtol=tolerance)
    assert np.isclose(results["noise_db"], noise_pow_db - 3, rtol=tolerance)
    assert np.isclose(results["thd_db"], thd_db, rtol=tolerance)
    assert np.isclose(results["thdn_db"], thdn_db, rtol=tolerance)


def test_harm_analysis_harm_zero():
    """Tests n_harm set to 0."""
    # test signal
    n = 2**18
    fs = 1000
    t = np.arange(0, n / fs, 1 / fs)
    bw = 250

    noise_pow_db = -70
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
        + noise
    )

    fund_pow_db = 10 * np.log10(2**2 / 2)
    harm_power = (0.01**2) / 2

    thd_db = 10 * np.log10(harm_power) - fund_pow_db
    snr_db = fund_pow_db - (noise_pow_db - 3)
    thdn_db = 10 * np.log10(10 ** ((noise_pow_db - 3) / 10) + 10 ** (thd_db / 10))
    noise_pow_db = thdn_db + fund_pow_db

    results = harm_analysis(x, bw=bw, n_harm=0, fs=fs)

    print("Function results:")
    for key, value in results.items():
        print(f"{key.ljust(10)} [dB]: {value}")

    logging.info(
        "\n"
        "Expected values\n"
        f"    Fundamental power (dB): {fund_pow_db}\n"
        f"    Fundamental Freq [Hz]: {f1}\n"
        f"    Noise power (dB): {noise_pow_db}\n"
        f"    DC power [dB]: {dc_power_db}\n"
        f"    DC level: {dc_level}\n"
        f"    THD (dB): {thd_db}\n"
        f"    SNR (dB): {snr_db}\n"
        f"    THD+N (dB): {thdn_db}"
    )

    tolerance = 0.3

    assert np.isclose(results["fund_db"], fund_pow_db, rtol=tolerance)
    assert np.isclose(results["fund_freq"], f1, rtol=tolerance)
    assert np.isclose(results["dc_db"], dc_power_db, rtol=tolerance)
    assert np.isclose(results["noise_db"], noise_pow_db - 3, rtol=tolerance)
    assert results["thd_db"] is np.nan
    assert np.isclose(results["thdn_db"], thdn_db, rtol=tolerance)


def test_dc_measurement():
    """Test for harm_analysis function."""
    # test signal
    n = 2**18
    fs = 1000
    t = np.arange(0, n / fs, 1 / fs)
    bw = 250

    noise_pow_db = -70
    harm_pow = 0.01**2 / 2
    thdn = 10 ** (noise_pow_db / 10) + harm_pow
    thdn_db = 10 * np.log10(thdn)

    noise_std = 10 ** (noise_pow_db / 20)

    dc_level = 0.123456789
    dc_power_db = 20 * np.log10(dc_level)
    noise = rng.normal(loc=0, scale=noise_std, size=len(t))

    f1 = 100.13

    x = dc_level + 0.01 * np.cos(2 * np.pi * f1 * 2 * t) + noise

    results = dc_measurement(x, bw=bw, fs=fs)

    print("Function results:")
    for key, value in results.items():
        print(f"{key.ljust(10)} [dB]: {value}")

    logging.info(
        "\n"
        "Expected values\n"
        f"    Total noise (dB): {thdn_db}\n"
        f"    DC power [dB]: {dc_power_db}\n"
        f"    DC level: {dc_level}\n"
    )

    tolerance = 0.3

    assert np.isclose(results["dc"], dc_level, rtol=tolerance)
    assert np.isclose(results["dc_db"], dc_power_db, rtol=tolerance)
    assert np.isclose(results["noise_db"], thdn_db - 3, rtol=tolerance)


if __name__ == "__main__":
    test_harm_analysis()
    test_harm_analysis_dc()
    test_harm_analysis_bw()
    test_harm_analysis_harm_zero()
    test_dc_measurement()
