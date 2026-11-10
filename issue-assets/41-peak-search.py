"""Visualize issue #41: the fundamental's peak search runs to its +/-100-bin limit on a noise-free tone."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from harm_analysis._spectrum import _power_spectrum, _find_freq_bins, _find_dc_bins, _to_db

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FUND, NOISE, DC = "#2a78d6", "#eb6834", "#1baf7a"
SEARCH = 100  # half-width of the peak-boundary search in _find_freq_bins

cases = [
    (256, 0.0, "256 samples, no noise", "fundamental claims every bin (incl. DC)\n→ no noise bins left → TypeError"),
    (1024, 0.0, "1024 samples, no noise", "fundamental claims ±100 bins (search limit)\n→ runs, but 200 bins of leakage are 'the tone'"),
    (256, 1e-5, "256 samples, white noise σ = 1e-5", "peak ends where the leakage meets the noise floor\n→ analysis works"),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(3, 1, figsize=(11, 10), facecolor=SURFACE)
for ax, (n, noise_std, title, outcome) in zip(axes, cases):
    x = np.cos(2 * np.pi * 0.2 * np.arange(n)) + np.random.default_rng(0).normal(0, noise_std, n) if noise_std else np.cos(2 * np.pi * 0.2 * np.arange(n))
    P, f, enbw, bw = _power_spectrum(x)
    db = _to_db(P / P.max())
    k = int(np.argmax(P))
    _, _, fund_bins = _find_freq_bins(P, k, bw_bins=bw, enbw_bins=enbw)
    dc_end = int(_find_dc_bins(P))
    bins = np.arange(P.size)
    is_fund = np.isin(bins, fund_bins)
    is_dc = (bins < dc_end) & ~is_fund
    is_noise = ~is_fund & ~is_dc

    ax.set_facecolor(SURFACE)
    ax.axvspan(fund_bins[0] - 0.5, fund_bins[-1] + 0.5, color=FUND, alpha=0.10, lw=0)
    ax.plot(bins, db, color=GRID, lw=1, zorder=1)
    ax.plot(bins[is_fund], db[is_fund], "o", ms=4, color=FUND, zorder=3)
    ax.plot(bins[is_noise], db[is_noise], "o", ms=4, color=NOISE, zorder=3)
    ax.plot(bins[is_dc], db[is_dc], "s", ms=6, color=DC, zorder=3)
    for edge, side in ((k - SEARCH, -1), (k + SEARCH, 1)):
        if 0 <= edge < P.size:
            ax.axvline(edge, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.text(edge + side * 0.008 * P.size, -170, "search limit\n(peak ± 100 bins)", color=INK2, fontsize=8,
                    ha="left" if side > 0 else "right", va="bottom")
    # the tone's real main lobe (Hann: about ±2 bins)
    ax.annotate("Hann main lobe\n(the actual tone, ±2 bins)", xy=(k, -2), xytext=(k + 0.06 * P.size, -30),
                color=INK, fontsize=9, arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.set_ylim(-175, 25)
    ax.set_xlim(-0.02 * P.size, P.size * 1.02)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_ylabel("dB relative to peak")
    ax.set_title(f"{title}:  {is_fund.sum()} fundamental, {is_noise.sum()} noise, {is_dc.sum()} DC bins",
                 loc="left", color=INK, fontsize=11)
    ax.text(0.99, 0.96, outcome, transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))
axes[-1].set_xlabel("FFT bin")
# legend (identity never by color alone: markers + labels)
handles = [plt.Line2D([], [], ls="", marker="o", ms=6, color=FUND, label="bins assigned to the fundamental"),
           plt.Line2D([], [], ls="", marker="o", ms=6, color=NOISE, label="bins counted as noise"),
           plt.Line2D([], [], ls="", marker="s", ms=6, color=DC, label="DC bins")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.955), ncol=3, frameon=False, labelcolor=INK)
fig.suptitle("Issue #41: tone at 0.2·fs (not on a bin), no noise", x=0.06, y=0.995, ha="left",
             color=INK, fontsize=13, fontweight="bold")
fig.text(0.06, 0.01, "Without a noise floor the Hann leakage skirt keeps falling, so the rule "
         "'the peak ends where the next bin is higher' never fires and the peak grows to the search limit.",
         color=INK2, fontsize=9)
fig.tight_layout(rect=(0.03, 0.025, 1, 0.94))
out = "/tmp/claude-0/-home-user-harm-analysis/7e4818d3-ae0c-5001-bd80-b483cb7cab54/scratchpad/issue41.png"
fig.savefig(out, dpi=200, facecolor=SURFACE)
print(out)
