import click
import numpy as np
import matplotlib.pyplot as plt
from harm_analysis import harm_analysis
from matplotlib.ticker import EngFormatter


@click.command()
@click.argument("filename", type=click.Path(exists=True, readable=True))
@click.option("--fs", default=1.0, help="Sampling frequency.")
@click.option("--plot", is_flag=True, help="Plot the power spectrum of the data")
@click.option("--sep", default=" ", help='Separator between items.')
@click.option("--sfactor", default="1", help='Scaling factor. The data will be multiplied by this number, before the function is called. Examples: 1/8, 5, etc')
def cli(filename, fs, plot, sep, sfactor):
    '''Runs the harm_analysis function for a file containing time domain data'''

    # scaling factor
    file_data = np.fromfile(filename, sep=sep)*eval(sfactor)

    if plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        results, ax[1] = harm_analysis(file_data, FS=fs, plot=True, ax=ax[1])
    else:
        results = harm_analysis(file_data, FS=fs, plot=False)

    print("Function results:")
    for key, value in results.items():
        click.echo(f"{key.ljust(10)}: {value}")

    if plot is True:
        ax[1].grid(True, which='both')
        ax[1].set_title('Power spectrum')
        ax[1].set_xscale('log')
        ax[1].xaxis.set_major_formatter(EngFormatter(unit='Hz'))

        ax[0].set_title('Data')
        ax[0].plot(file_data)
        ax[0].grid(True, which='both', linestyle='-')
        ax[0].set_xlabel("[n]")

        plt.tight_layout()
        plt.show()
