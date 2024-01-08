import click
import numpy as np
import matplotlib.pyplot as plt
from harm_analysis import harm_analysis
import ast


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
        fig, ax = plt.subplots()
        results, ax = harm_analysis(file_data, FS=fs, plot=True, ax=ax)
    else:
        results = harm_analysis(file_data, FS=fs, plot=False)

    print("Function results:")
    for key, value in results.items():
        click.echo(f"{key.ljust(10)} [dB]: {value}")

    if plot is True:
        ax.set_title('Harmonic analysis example')
        plt.show()
