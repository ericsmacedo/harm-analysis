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
"""Overview for the home page: what harm_analysis and spec_analysis measure.

Run ``python docs/figures/overview.py --save`` to update the static copy in
``docs/images/overview.png`` shown by the README on GitHub and PyPI.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from harm_analysis import harm_analysis, spec_analysis

fs, n = 48000, 8192
t = np.arange(n) / fs
rng = np.random.default_rng(1)

# A 3 kHz test tone with distortion and noise, as from an amplifier or a converter.
x = (
    np.sin(2 * np.pi * 3000 * t)
    + 10 ** (-50 / 20) * np.sin(2 * np.pi * 6000 * t)
    + 10 ** (-60 / 20) * np.sin(2 * np.pi * 9000 * t)
    + rng.normal(0, 3e-4, n)
)
# A recording without a known tone: DC offset, hum, two tones and noise.
y = (
    0.05
    + 0.02 * np.sin(2 * np.pi * 60 * t)
    + 0.5 * np.sin(2 * np.pi * 5300 * t)
    + 0.1 * np.sin(2 * np.pi * 12700 * t)
    + rng.normal(0, 1e-3, n)
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5))
r = harm_analysis(x, fs=fs, n_harm=3, plot=True, ax=ax1)
ax1.set_title(
    f"harm_analysis: {r.fund_freq / 1000:.0f} kHz tone, "
    f"SNR {r.snr_db:.1f} dB, THD {r.thd_db:.1f} dB, SINAD {r.sinad_db:.1f} dB",
    loc="left",
    fontsize=10,
)
s = spec_analysis(y, fs=fs, plot=True, ax=ax2)
khz = 1000
tones = ", ".join(f"{f / khz:.1f} kHz" if f >= khz else f"{f:.0f} Hz" for f in s.tones_freq)
ax2.set_title(f"spec_analysis: DC, tones at {tones}, noise", loc="left", fontsize=10)
fig.tight_layout()

if "--save" in sys.argv:
    fig.savefig(Path(__file__).parents[1] / "images" / "overview.png", dpi=110)
