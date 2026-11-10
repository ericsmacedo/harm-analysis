# How it works

Both functions follow the same steps: window the samples, compute a power
spectrum, split its bins into DC, tones and noise, and add up the power of each
group. This page goes through those steps with the actual figures the library
draws. Powers are in dB relative to the signal's units squared (a sine of
amplitude 1 is −3.01 dB) and frequencies are in the units of `fs`.

```{eval-rst}
.. plot:: figures/how_pipeline.py
```

## 1. Window and power spectrum

1. The samples `x` are multiplied by a window `w`: a periodic Hann window by
   default, or the `window` you pass (same length as `x`). The window tapers
   the ends of the record so that a tone that doesn't complete a whole number of
   periods doesn't smear across the whole spectrum.
2. The FFT of `x · w` is normalized by the window's coherent gain (the sum of
   `w`), and squared into a power per bin.
3. It is folded into a **single-sided** spectrum (DC to fs/2): every bin except
   DC is doubled, because it also stands for its negative frequency. For an
   even number of samples, the last bin is fs/2 itself, which has no negative
   counterpart and is not doubled.

The power of a component (a tone, the noise, DC) is the **sum of its bins
divided by the window's equivalent noise bandwidth** (ENBW, in bins):

$$P = \frac{1}{\text{ENBW}} \sum_{k \,\in\, \text{bins}} S[k]$$

The window spreads a tone over several bins, and the ENBW is exactly the factor
that makes the sum come out right. A sine of amplitude $A$ therefore always
reads $A^2/2$, whatever the window and wherever the tone falls between two
bins.

| Window | ENBW (bins) | Main lobe (null to null) |
|---|---|---|
| Hann (default) | 1.50 | 4 bins |
| Blackman-Harris | 2.00 | 8 bins |
| Flat-top | 3.77 | 10 bins |
| Rectangular | 1.00 | 2 bins |

## 2. DC

DC is the run of bins at the start of the spectrum that keeps decreasing: it
ends at the first bin that is higher than the previous one. (The comparison
undoes the doubling of step 1, otherwise windows with a flat top, like
flat-top, would look as if the spectrum rose from bin 0 to bin 1.)

## 3. The fundamental

**Where it is.** By default, the fundamental is the highest bin between DC and
`bw` (`fs / 2` if not given). If you pass `fund_freq`, it is the highest bin
within ±3 bins of that frequency instead, and that peak must stand out from the
local noise floor; otherwise `harm_analysis` raises a `ValueError` rather than
analyze a noise bin. See
[A stronger tone than the one you injected](corner-cases.md#a-stronger-tone-than-the-one-you-injected).

**Which bins belong to it.** Starting at the peak, the tone keeps every bin
while the spectrum keeps falling. It stops at the first bin that is higher
than the previous one, which is where the window's leakage meets the noise
floor. The search goes at most 100 bins each way, and never into the DC bins.

```{eval-rst}
.. plot:: figures/how_peak_bins.py
```

The leakage around the main lobe is the tone's power, spread out by the
window, so it is counted as the tone and not as noise. Keeping only the main
lobe would count the rest as noise: with a Hann window, 39 to 42 dB below the
tone's power lies outside its main lobe, which would cap the measured SNR at
about 40 dB.

**Its frequency** is the power-weighted average of its bins, so it is resolved
to a fraction of a bin (a peak at fs/2 is reported at fs/2).

## 4. Harmonics

The harmonics are expected at $k \cdot f_0$ for $k = 2 \ldots$ `n_harm` + 1
(default `n_harm=5`, so the 2nd to the 6th harmonic). Each one that is at or
below `bw` (or fs/2) is found like the fundamental: the highest bin within
±3 bins of its expected frequency, and the bins around it while the spectrum
falls. Bins that already belong to the fundamental are never counted again.
Harmonics above `bw` or fs/2 are ignored; they are not folded back below fs/2.

In the plots, harmonics are numbered by their order, as in MATLAB: **H2** is
the 2nd harmonic ($2 f_0$).

## 5. Noise

Every remaining bin up to `bw` (not DC, not the fundamental, not a harmonic) is
noise.

## 6. The metrics

With $P_F$ the fundamental's power, $P_{H_k}$ the power of each harmonic,
$P_N$ the noise power and $P_{DC}$ the DC power (all up to `bw`):

| Result field | Definition |
|---|---|
| `fund_db` | $10 \log_{10} P_F$ |
| `fund_freq` | power-weighted center of the fundamental's bins |
| `harm_db`, `harm_freq` | $10 \log_{10} P_{H_k}$ and the center of each harmonic |
| `dc_db` | $10 \log_{10} P_{DC}$ |
| `noise_db` | $10 \log_{10} P_N$ |
| `thd_db` | $10 \log_{10} \left( \sum_k P_{H_k} / P_F \right)$; `nan` if no harmonic is in the band (e.g. `n_harm=0`) |
| `snr_db` | `fund_db` − `noise_db` |
| `thdn_db` | $10 \log_{10} \left( P_{\text{rest}} / P_F \right)$, where $P_{\text{rest}}$ is every bin up to `bw` except DC and the fundamental: harmonics, noise and any other spur |
| `sinad_db` | − `thdn_db` |
| `total_noise_and_dist_db` | the noise integrated over the whole spectrum, from DC to fs/2, ignoring `bw` (harmonics excluded). Equal to `noise_db` when `bw` is not set |

A power of exactly zero is reported as `-inf` dB (for example the noise of a
perfectly clean simulated tone), without warnings.

The integrated-noise curves in the plots are the running sums of these powers
from DC upward: their value at `bw` is `noise_db`, and with harmonics included
(dashed curve), the THD+N power in dB.

(tone-detection)=
## 7. `spec_analysis`: finding tones without knowing them

`spec_analysis` has no fundamental to start from. It finds DC like
`harm_analysis` and then looks for **tones**: peaks in the dB spectrum that stand
out from the local noise floor.

- The **local noise floor** is the median of the 101 bins around each bin. A
  median ignores the tones themselves, so it follows the noise even next to a
  strong tone or with shaped noise.
- A peak is a tone if it is more than **T dB above that floor**, where T is
  chosen so that, on average, noise alone gives a false tone in only
  `false_alarm_rate` of the analyzed spectra (default `1e-3`):

  $$T = 10 \log_{10} \frac{\ln(N_\text{bins} / \alpha)}{\ln 2} + 0.75\ \text{dB}$$

  For Gaussian noise each bin's power is exponentially distributed, so a bin
  exceeds the median by T dB with probability $\exp(-\ln 2 \cdot 10^{T/10})$;
  with $N_\text{bins}$ bins to check, that probability is set to
  $\alpha / N_\text{bins}$. The 0.75 dB accounts for the median itself being
  an estimate (calibrated on white noise).
- Peaks must also be at least `distance` bins apart (default 6) and have a
  `prominence` of at least 10 dB. You can replace the whole rule with a fixed
  `height` in dB.

| Signal length | Threshold above the local floor (α = 1e-3) |
|---|---|
| 1 024 samples | 13.5 dB |
| 4 096 samples | 14.0 dB |
| 65 536 samples | 14.7 dB |
| 262 144 samples | 15.1 dB |

```{eval-rst}
.. plot:: figures/how_tone_detection.py
```

Each tone's power and frequency are then measured with the same peak rule as
the fundamental (step 3), and everything else apart from DC is noise.

Next: [Corner cases](corner-cases.md) shows what happens in the situations
where this simple procedure needs your help.
