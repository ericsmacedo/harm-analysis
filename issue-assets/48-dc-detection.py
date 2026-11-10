"""Visualize why DC-bin detection found only 1 bin for the flat-top window (#48, second bug)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from harm_analysis._spectrum import _power_spectrum

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"hann": "#2a78d6", "flattop": "#eb6834"}
NAMES = {"hann": "Hann (default)", "flattop": "Flat-top"}
FLOOR = -45  # dB; lower values are drawn at the floor with a down-pointing marker

n = 4096
k = np.arange(n)
x = 0.25 + np.cos(2 * np.pi * 0.1003 * k) + np.random.default_rng(0).normal(0, 1e-6, n)

def dc_end(values):
    """Same rule as _find_dc_bins: the first bin that is higher than the previous one ends DC."""
    return int(np.argmax(np.diff(values[:50]) > 0) + 1)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey=True, facecolor=SURFACE)
bins = np.arange(8)
for row, win_name in enumerate(("hann", "flattop")):
    w = signal.windows.get_window(win_name, n, fftbins=True)
    P, *_ = _power_spectrum(x, window=w)
    one_sided = P[:50].copy()
    two_sided = one_sided.copy(); two_sided[1:] /= 2
    for col, (values, title) in enumerate(((one_sided, "Before: compares the single-sided spectrum\n(every bin except DC doubled)"),
                                          (two_sided, "After the fix: doubling undone before comparing"))):
        ax = axes[row, col]
        ax.set_facecolor(SURFACE)
        db = 10 * np.log10(values / values[0])
        end = dc_end(values)
        # bins counted as DC
        ax.axvspan(-0.5, end - 0.5, color=COLORS[win_name], alpha=0.12, lw=0)
        ax.axvline(end - 0.5, color=COLORS[win_name], lw=1, ls=(0, (3, 3)))
        shown = np.maximum(db[: bins.size], FLOOR)
        ax.plot(bins, shown, color=COLORS[win_name], lw=2, zorder=3)
        clipped = db[: bins.size] < FLOOR
        ax.plot(bins[~clipped], shown[~clipped], "o", ms=8, color=COLORS[win_name], mec=SURFACE, mew=2, zorder=4)
        ax.plot(bins[clipped], shown[clipped], "v", ms=8, color=COLORS[win_name], mec=SURFACE, mew=2, zorder=4)
        ax.axhline(0, color=GRID, lw=1, zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.text(end - 0.4, FLOOR + 3, f"{end} DC bin{'s' if end > 1 else ''}", color=INK, fontsize=10, va="bottom")
        if row == 0:
            ax.set_title(title, color=INK, fontsize=11, loc="left")
        if col == 0:
            ax.set_ylabel(f"{NAMES[win_name]}\n\npower relative to bin 0 [dB]")
        if row == 1:
            ax.set_xlabel("FFT bin")
        # explain the stop in the broken case
        if win_name == "flattop" and col == 0:
            ax.annotate("bin 1 is 2.7 dB above bin 0\n(only because it was doubled)\n→ detection stops after bin 0",
                        xy=(1, db[1]), xytext=(3.6, 0), color=INK, fontsize=10, va="top",
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
        if win_name == "hann" and col == 0:
            ax.annotate("bin 1 stays below bin 0\neven when doubled\n→ DC lobe found\n(doubling is harmless here)",
                        xy=(1, db[1]), xytext=(3.75, -4), color=INK, fontsize=10, va="top",
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
axes[0, 0].set_ylim(FLOOR - 1, 8)
axes[0, 0].set_xticks(bins)
fig.suptitle("Why DC detection failed for the flat-top window (#48, second bug)", x=0.06, ha="left", color=INK, fontsize=13, fontweight="bold")
fig.text(0.06, 0.012,
         "▼ = below −45 dB (drawn at the axis floor).   Shaded = bins summed as DC power.\n"
         "DC power error before → after the fix:  Hann in harm_analysis +1.76 → 0.00 dB (first bug);  "
         "flat-top in spec_analysis −5.76 → 0.00 dB (this bug).",
         color=INK2, fontsize=9)
fig.tight_layout(rect=(0.03, 0.05, 1, 0.95))
out = "/tmp/claude-0/-home-user-harm-analysis/7e4818d3-ae0c-5001-bd80-b483cb7cab54/scratchpad/dc_detection_bug.png"
fig.savefig(out, dpi=200, facecolor=SURFACE)
print(out)
