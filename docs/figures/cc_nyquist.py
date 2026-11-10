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
"""Corner case: components at fs/2."""

import matplotlib.pyplot as plt
import numpy as np

from harm_analysis import harm_analysis, spec_analysis

n = 4096
k = np.arange(n)
noise = np.random.default_rng(0).normal(0, 1e-4, n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.5, 1]})
x = np.cos(2 * np.pi * 0.25 * k) + 0.01 * np.cos(2 * np.pi * 0.5 * k) + noise
r = harm_analysis(x, n_harm=1, plot=True, ax=ax1)
ax1.set_xlabel("Frequency [fraction of fs]")
ax1.set_title(f"Fundamental at fs/4: H2 lands exactly on fs/2 ({r.harm_db[0]:.2f} dB)", loc="left", fontsize=10)

phases = np.linspace(0, np.pi / 2 * 0.97, 25)
measured = [
    spec_analysis(0.5 * np.cos(2 * np.pi * 0.1003 * k) + np.cos(np.pi * k + p) + noise).tones_db[-1] for p in phases
]
fine = np.linspace(0, np.pi / 2 * 0.97, 200)
ax2.plot(np.degrees(fine), 10 * np.log10(np.cos(fine) ** 2), color="#9a9994", lw=2, label="A² cos²φ (expected)")
ax2.plot(np.degrees(phases), measured, "o", ms=5, color="#2a78d6", label="measured by spec_analysis")
ax2.plot(
    [0, 90],
    [10 * np.log10(0.5)] * 2,
    color="#eb6834",
    lw=1,
    ls=(0, (4, 3)),
    label="A²/2: a tone of the same amplitude elsewhere",
)
ax2.set_xlabel("Phase φ of the fs/2 tone [degrees]")
ax2.set_ylabel("Measured power [dB]")
ax2.set_title("A tone at fs/2: its power depends on its phase", loc="left", fontsize=10)
ax2.legend(frameon=False, fontsize=8, loc="lower left")
ax2.grid(color="#e4e3df")
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)
fig.tight_layout()
