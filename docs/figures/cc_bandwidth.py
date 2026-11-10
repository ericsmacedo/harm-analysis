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
"""Corner case: shaped noise outside the band of interest (sigma-delta style)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter
from scipy import signal

from harm_analysis import harm_analysis

fs, n = 10e6, 2**18
t = np.arange(n) / fs
x = 0.001 * np.sin(2 * np.pi * 1e3 * t) + signal.lfilter([1, -1], [1, 0], np.random.default_rng(1).normal(0, 1, n))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
r1 = harm_analysis(x, fs=fs, n_harm=0, plot=True, ax=ax1)
ax1.set_title(
    f"Default bw = fs/2: SNR {r1.snr_db:.1f} dB, fundamental {r1.fund_freq / 1e6:.2f} MHz", loc="left", fontsize=10
)
r2 = harm_analysis(x, fs=fs, n_harm=0, bw=5e3, plot=True, ax=ax2)
ax2.set_title(f"bw = 5 kHz: SNR {r2.snr_db:.1f} dB, fundamental {r2.fund_freq / 1e3:.2f} kHz", loc="left", fontsize=10)
for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(EngFormatter(unit="Hz"))
fig.tight_layout()
