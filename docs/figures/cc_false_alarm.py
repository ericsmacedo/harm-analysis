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
"""Corner case: false tones from noise in spec_analysis."""

import matplotlib.pyplot as plt
import numpy as np

from harm_analysis import spec_analysis

runs, n = 500, 1024
rates = [0.3, 0.1, 0.03, 0.01]
measured = []
for rate in rates:
    false = sum(
        spec_analysis(np.random.default_rng(seed).normal(0, 1, n), false_alarm_rate=rate).tones_freq.size > 0
        for seed in range(runs)
    )
    measured.append(false / runs)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot([0.005, 0.5], [0.005, 0.5], color="#9a9994", lw=1.5, ls=(0, (4, 3)), label="measured = requested")
ax.plot(
    rates,
    np.maximum(measured, 1 / runs),
    "o",
    ms=8,
    color="#2a78d6",
    label=f"pure white noise, {runs} spectra per point",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("false_alarm_rate (requested)")
ax.set_ylabel("Fraction of spectra with a false tone")
ax.set_title("spec_analysis on noise only: false tones at (or below) the requested rate", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(color="#e4e3df", which="both")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
