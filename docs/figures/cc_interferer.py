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
"""Corner case: a stronger interferer than the injected tone."""

import matplotlib.pyplot as plt
import numpy as np

from harm_analysis import harm_analysis

fs, n = 1000, 4096
t = np.arange(n) / fs
x = (
    np.cos(2 * np.pi * 100.3 * t)  # injected tone
    + 2 * np.cos(2 * np.pi * 150.7 * t)  # stronger interferer
    + np.random.default_rng(0).normal(0, 1e-4, n)
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
r1 = harm_analysis(x, fs=fs, n_harm=0, plot=True, ax=ax1)
ax1.set_title(f"Default: fundamental = {r1.fund_freq:.1f} Hz (the interferer)", loc="left", fontsize=10)
r2 = harm_analysis(x, fs=fs, n_harm=0, fund_freq=100.3, plot=True, ax=ax2)
ax2.set_title(f"fund_freq=100.3: fundamental = {r2.fund_freq:.1f} Hz", loc="left", fontsize=10)
fig.tight_layout()
