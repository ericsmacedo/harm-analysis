---
name: harm-analysis
description: Measure SNR, THD, SINAD, THD+N, DC, noise and tones of sampled signals with the harm-analysis Python package. Use when asked to analyze an ADC/DAC/amplifier/simulation output, a captured waveform or a file of samples, or to compute spectral metrics from time-domain data.
---

# Signal analysis with harm-analysis

`harm-analysis` estimates spectral metrics from a 1-D real signal using a windowed
FFT. Prefer it over writing your own FFT code: windowing, bin grouping and noise
integration are already handled.

## Choose the analysis

- **The signal has a known injected tone** (a circuit driven by a sine wave, a
  simulation with a test tone): use `harm_analysis` / the `run_harm_analysis` tool.
  It returns the fundamental, harmonics, DC and noise power, THD, SNR, SINAD and
  THD+N.
- **No known tone**, or you first need to find out what is in the signal: use
  `spec_analysis` / the `run_spec_analysis` tool. It detects DC, tones and noise.

## How to run it

If the `harm-analysis` MCP server is connected, call its tools
(`run_harm_analysis`, `run_spec_analysis`) with `fs` and either `file_path` (a text
file of samples) or `samples` (a list of numbers).

Otherwise use the Python package (`pip install harm-analysis`):

```python
import numpy as np
from harm_analysis import harm_analysis, spec_analysis

x = np.loadtxt("capture.txt")          # 1-D real samples
results = harm_analysis(x, fs=1e6)      # fs in Hz -> frequencies in Hz
print(results.snr_db, results.thd_db, results.fund_freq)
print(results.to_dict())                # all metrics
```

Or the command line: `harm-analysis capture.txt --fs 1e6 --json`
(`spec-analysis` for the other analysis).

## Inputs

- `fs` is the sampling frequency; result frequencies use the same units. Ask for
  it if unknown; don't assume 1.
- The signal must be real, 1-D, finite and at least 32 samples. Longer captures
  (thousands of samples) give better estimates.
- `bw` (harm_analysis) limits the band used for the metrics, up to `fs / 2`.
- `n_harm` (harm_analysis) is the number of harmonics in THD (default 5).

## Interpreting results

- Powers are in dB relative to the signal's units squared (`fund_db` of a unit
  amplitude sine is about -3.01 dB). SNR, THD, SINAD and THD+N are ratios in dB.
- `-inf` dB (or `null` in JSON) means a power of zero; `thd_db` is NaN/`null`
  when no harmonics are in the band.
- SINAD = -THD+N.

## Pitfalls

- The fundamental is taken to be the **highest** spectral component between DC
  and `bw`. If a stronger interferer is present, the results describe the wrong
  tone. Check `fund_freq` against the expected tone frequency. If it is wrong, run
  `spec_analysis` to locate the tones, then rerun `harm_analysis` with `bw` below
  the interferer.
- Tones must be far enough apart for the Hann window's sidelobes; otherwise pass
  another `window` (same length as the signal).
- `spec_analysis` reports a tone when it stands out from the local noise floor by a
  margin set by `false_alarm_rate` (default 1e-3: noise alone gives a false tone in
  about 1 of 1000 spectra; the margin is about 13.5-15 dB). Lower it (e.g. 1e-6) to
  be stricter, or raise it to find weaker tones. Treat tones just above the noise
  floor with care.
