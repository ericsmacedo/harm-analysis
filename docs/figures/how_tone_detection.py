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
"""How it works: tone detection in spec_analysis."""

import matplotlib.pyplot as plt
import numpy as np

from harm_analysis import spec_analysis
from harm_analysis._spectral import _MEDIAN_WINDOW, _detection_threshold_db, _rolling_median_bn
from harm_analysis._spectrum import _power_spectrum, _to_db

fs, n = 1000, 4096
t = np.arange(n) / fs
x = (
    np.cos(2 * np.pi * 100.3 * t)
    + 3e-4 * np.cos(2 * np.pi * 253.1 * t)  # weak spur, still detected
    + np.random.default_rng(3).normal(0, 1e-3, n)
)
power, freq, *_ = _power_spectrum(x, fs=fs)
db = _to_db(power)
floor = _rolling_median_bn(db, _MEDIAN_WINDOW)
margin = _detection_threshold_db(power.size, 1e-3)
results = spec_analysis(x, fs=fs)

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(freq, db, color="#9a9994", lw=0.7, label="power spectrum")
ax.plot(freq, floor, color="#52514e", lw=1.2, label=f"local noise floor (median of {_MEDIAN_WINDOW} bins)")
ax.plot(
    freq,
    floor + margin,
    color="#eb6834",
    lw=1.5,
    ls=(0, (5, 2)),
    label=f"detection threshold: floor + {margin:.1f} dB (false_alarm_rate = 1e-3)",
)
for f0, level in zip(results.tones_freq, results.tones_db, strict=True):
    ax.plot(f0, db[np.argmin(np.abs(freq - f0))], "o", ms=7, color="#2a78d6", zorder=5)
    ax.annotate(
        f"tone {f0:.1f} Hz\n{level:.1f} dB",
        xy=(f0, db[np.argmin(np.abs(freq - f0))]),
        xytext=(8, 2),
        textcoords="offset points",
        fontsize=8,
    )
ax.set_ylim(top=25)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Power [dB]")
ax.set_title("A peak is a tone if it stands out from the local noise floor by more than the threshold", loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.grid(color="#e4e3df")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
