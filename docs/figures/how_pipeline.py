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
"""How it works: from samples to classified spectrum."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from harm_analysis import harm_analysis

fs, n = 1000, 1024
t = np.arange(n) / fs
x = (
    0.1
    + np.cos(2 * np.pi * 100.3 * t)
    + 0.01 * np.cos(2 * np.pi * 200.6 * t)
    + np.random.default_rng(0).normal(0, 1e-4, n)
)
window = signal.windows.hann(n, sym=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
ax1.plot(t * 1e3, x * window, color="#2a78d6", lw=0.5, label="x · w (windowed signal)")
ax1.plot(t * 1e3, 1.1 * window, color="#eb6834", lw=1.5, ls=(0, (5, 2)), label="Hann window w")
ax1.set_ylim(-1.2, 1.6)
ax1.set_xlabel("Time [ms]")
ax1.set_title("1. Window the samples", loc="left")
ax1.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="none")
for side in ("top", "right"):
    ax1.spines[side].set_visible(False)

harm_analysis(x, fs=fs, n_harm=1, plot=True, ax=ax2)
ax2.set_title("2-5. Power spectrum, split into DC, fundamental, harmonics and noise", loc="left")
fig.tight_layout()
