'''Tests the harmonic analysis function
'''

import numpy as np
import logging
from harm_analysis._harm_analysis import harm_analysis


def test_harm_analysis():
    '''Test for harm_analysis function

    Checks if the function can obtain results with less than 0.1 dB of error.
    '''

    # test signal
    N = 2048
    FS = 1000
    t = np.arange(0, N/FS, 1/FS)

    noise_pow_db = -70
    noise_std = 10**(noise_pow_db/20)
    dc_level = 0.123456789
    dc_power_db = 20*np.log10(dc_level)
    noise = np.random.normal(loc=0, scale=noise_std, size=len(t))

    F1 = 100.13

    x = dc_level + 2*np.cos(2*np.pi*F1*t) +\
        0.01*np.cos(2*np.pi*F1*2*t) +\
        0.005*np.cos(2*np.pi*F1*3*t) +\
        noise

    fund_pow_db = 10*np.log10(2**2/2)
    harm_power = (0.01**2)/2 + (0.005**2)/2
    thd_db = 10*np.log10(harm_power) - fund_pow_db
    snr_db = fund_pow_db - noise_pow_db
    thdn_db = 10*np.log10(10**(noise_pow_db/10) + 10**(thd_db/10))

    results = harm_analysis(x, FS=FS)

    print("Function results:")
    for key, value in results.items():
        print(f"{key.ljust(10)} [dB]: {value}")

    logging.info('\n' +
                 'Expected values\n' +
                 f'    Fundamental power (dB): {fund_pow_db}\n' +
                 f'    Fundamental Freq [Hz]: {F1}\n' +
                 f'    Noise power (dB): {noise_pow_db}\n' +
                 f'    DC power [dB]: {dc_power_db}\n' +
                 f'    DC level: {dc_level}\n' +
                 f'    THD (dB): {thd_db}\n' +
                 f'    SNR (dB): {snr_db}\n' +
                 f'    THD+N (dB): {thdn_db}')

    tolerance = 0.3

    assert np.isclose(results['fund_db'], fund_pow_db, rtol=tolerance)
    assert np.isclose(results['fund_freq'], F1, rtol=tolerance)
    assert np.isclose(results['dc_db'], dc_power_db, rtol=tolerance)
    assert np.isclose(results['noise_db'], noise_pow_db, rtol=tolerance)
    assert np.isclose(results['thd_db'], thd_db, rtol=tolerance)
    assert np.isclose(results['thdn_db'], thdn_db, rtol=tolerance)


if __name__ == "__main__":
    test_harm_analysis()
