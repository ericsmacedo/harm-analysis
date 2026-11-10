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
"""Corner case: noise-free tones, on a bin and between bins, and the window's leakage."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from harm_analysis import harm_analysis
from harm_analysis._spectrum import _power_spectrum, _to_db

n = 1024
k = np.arange(n)
cases = [
    ("on a bin (coherent), Hann", 205 / n, "hann"),
    ("between bins, Hann", 0.2, "hann"),
    ("between bins, Blackman-Harris", 0.2, "blackmanharris"),
]
colors = ["#9a9994", "#2a78d6", "#eb6834"]

fig, ax = plt.subplots(figsize=(10, 4))
for (label, f0, window_name), color in zip(cases, colors, strict=True):
    window = signal.windows.get_window(window_name, n, fftbins=True)
    x = np.cos(2 * np.pi * f0 * k)
    power, freq, *_ = _power_spectrum(x, window=window)
    snr = harm_analysis(x, window=window, n_harm=0).snr_db
    ax.plot(
        freq,
        np.maximum(_to_db(power / power.max()), -320),
        color=color,
        lw=1.3,
        label=f"{label}: reported SNR {snr:.0f} dB",
    )
ax.set_xlim(0.1, 0.3)
ax.set_ylim(-330, 10)
ax.set_xlabel("Frequency [fraction of fs]")
ax.set_ylabel("dB relative to the tone")
ax.set_title(
    "A noise-free tone between bins leaks: the leakage that is far from the tone counts as noise",
    loc="left",
    fontsize=10,
)
ax.legend(frameon=False, fontsize=8, loc="center right")
ax.grid(color="#e4e3df")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
