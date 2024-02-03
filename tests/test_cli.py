'''Tests the CLI for the harmonic analysis function
'''
from click.testing import CliRunner
import numpy as np
from harm_analysis.cli import cli


def test_harm_analysis_cli():
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

    random_state = np.random.RandomState(1234567890)
    noise = random_state.normal(loc=0, scale=noise_std, size=len(t))

    F1 = 100.13

    x = dc_level + 2*np.cos(2*np.pi*F1*t) +\
        0.01*np.cos(2*np.pi*F1*2*t) +\
        0.005*np.cos(2*np.pi*F1*3*t) +\
        noise

    # Save data to TXT file
    np.savetxt("test_data_cli.txt", x, delimiter="\n")

    runner = CliRunner()

    result = runner.invoke(cli, ["test_data_cli.txt", "--fs", FS])

    assert result.exit_code == 0


if __name__ == "__main__":
    test_harm_analysis_cli()
