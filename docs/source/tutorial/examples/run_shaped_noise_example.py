'''Example using the harm_analysis function in a signal with noise shaping'''
import numpy as np
from scipy import signal
from harm_analysis import harm_analysis
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


# Sampling frequency
FS = 10e6
N = 2**18

t = np.arange(N) / FS

tone = 0.001 * np.sin(2 * np.pi * 1e3 * t)

quant_noise = np.random.normal(0, scale=1, size=N)
shaped_noise = signal.lfilter(b=[1, -1], a=[1, 0], x=quant_noise)

modulator_output = tone + shaped_noise

fig, ax = plt.subplots()
_, plt.axes = harm_analysis(x=modulator_output, n_harm=0, bw=5e3, FS=FS, plot=True,
                            ax=ax)
ax.set_xscale('log')
ax.xaxis.set_major_formatter(EngFormatter(unit='Hz'))
plt.show()
