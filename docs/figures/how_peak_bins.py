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
"""How it works: which bins belong to the fundamental."""

import matplotlib.pyplot as plt
import numpy as np

from harm_analysis._spectrum import _find_freq_bins, _power_spectrum, _to_db

n = 1024
x = np.cos(2 * np.pi * 0.1003 * np.arange(n)) + np.random.default_rng(0).normal(0, 1e-4, n)
power, freq, enbw, bw = _power_spectrum(x)
peak = int(np.argmax(power))
fund_power, fund_loc, fund_bins = _find_freq_bins(power, peak, bw_bins=bw, enbw_bins=enbw, min_bin=1)

db = _to_db(power / power.max())
bins = np.arange(peak - 25, peak + 26)
is_fund = np.isin(bins, fund_bins)

fig, ax = plt.subplots(figsize=(10, 3.6))
ax.axvspan(fund_bins[0] - 0.5, fund_bins[-1] + 0.5, color="#2a78d6", alpha=0.08, lw=0)
ax.plot(bins, db[bins], color="#d8d7d2", lw=1, zorder=1)
ax.plot(bins[is_fund], db[bins][is_fund], "o", ms=5, color="#2a78d6", label="fundamental bins", zorder=3)
ax.plot(bins[~is_fund], db[bins][~is_fund], "o", ms=5, color="#9a9994", label="noise bins", zorder=3)
ax.annotate(
    "main lobe\n(±2 bins for Hann)",
    xy=(peak, 0),
    xytext=(peak + 6, -12),
    fontsize=9,
    arrowprops={"arrowstyle": "->", "color": "#52514e"},
)
edge = fund_bins[-1]
ax.annotate(
    "the peak ends where the next\nbin is higher: leakage has\nreached the noise floor",
    xy=(edge, db[edge]),
    xytext=(edge + 3, db[edge] + 35),
    fontsize=9,
    arrowprops={"arrowstyle": "->", "color": "#52514e"},
)
ax.set_xlabel("FFT bin")
ax.set_ylabel("dB relative to peak")
ax.set_title(
    f"The fundamental is {fund_bins.size} bins: its main lobe plus the leakage above the noise floor", loc="left"
)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(color="#e4e3df")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
