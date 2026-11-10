# Corner cases

Spectral analysis has a few situations where the numbers can surprise you. This
page shows each one with real figures from the library: what happens, why, and
what to do. Every figure is generated from the current code when the
documentation is built, and each case is covered by the test suite.

## A stronger tone than the one you injected

**What happens.** Without help, `harm_analysis` takes the *highest* component as
the fundamental. If the signal contains something stronger than your test tone
(an interferer, a clock, a spur), that is what gets analyzed, with plausible but
wrong results. Here a 100.3 Hz tone is next to an interferer of twice its
amplitude at 150.7 Hz:

```{eval-rst}
.. plot:: figures/cc_interferer.py
```

**What to do.** Pass the frequency of the tone you injected; you usually know
it. It only needs to be within about 3 bins (3 · fs / N):

```python
harm_analysis(x, fs=fs, fund_freq=100.3)
```

The interferer is then counted as noise, which is what it is from the point of
view of the tone (here it gives an SNR of −6 dB). If the interferer is *above*
the tone, `bw` can exclude it from the band as well. If `fund_freq` is wrong and
there is no tone there, you get an error instead of a noise bin analyzed as the
fundamental:

```
ValueError: No tone found near `fund_freq` (0.3): the largest peak within 3 bins
is only 10.5 dB above the local noise floor; check `fund_freq` and `fs`.
```

## Shaped noise outside your band

**What happens.** Sigma-delta modulators and other noise-shaped systems push
their noise to high frequencies on purpose. Over the whole spectrum, that noise
is much larger than the tone, and by default it even becomes "the
fundamental":

```{eval-rst}
.. plot:: figures/cc_bandwidth.py
```

**What to do.** Set `bw` to your signal band. Only bins up to `bw` are used to
find the fundamental and to compute every metric; the curve beyond `bw` is
shown for context, and the plot marks the value at `bw`, which is the reported
`noise_db`.

## Noise-free tones in simulations

**What happens.** A simulated tone often has no noise at all. If the tone falls
exactly on an FFT bin (coherent sampling: a whole number of periods in the
record), its spectrum is clean and the reported SNR is limited only by
floating-point precision. If it falls between bins, the window's leakage never
meets a noise floor: the leakage close to the tone is counted as the tone (step 3
in [How it works](how-it-works.md#3-the-fundamental)), and the far tails, more
than 100 bins away or past the window's first sidelobe null, are counted as
noise:

```{eval-rst}
.. plot:: figures/cc_leakage.py
```

The reported SNR of a perfect off-bin tone is therefore the window's dynamic
range: about 120 dB with Hann and 92 dB with Blackman-Harris for 1024 samples.
Real measurements, whose noise is above that floor, are not affected. For very
short noise-free records, the tone's leakage can cover every bin; the noise is
then reported as zero power (`noise_db = -inf`, `snr_db = inf`).

**What to do.** In simulations, use coherent sampling (choose the tone frequency
as a whole number of cycles over the record length) when you need the SNR of a
noise-free design, or add a realistic noise level.

## Components exactly at fs/2

**What happens.** A tone at exactly half the sampling rate is sampled as
$A \cos(\varphi) \cdot (-1)^n$: its sine part is lost, so its measured power is
$A^2 \cos^2 \varphi$ and depends on its phase. This also happens to the 2nd
harmonic of a fundamental at fs/4 (or the 3rd harmonic of one at fs/6), which is
common with coherent test frequencies:

```{eval-rst}
.. plot:: figures/cc_nyquist.py
```

The library measures what is in the samples (left: the −40 dB harmonic reads
−40.00 dB with a cosine phase; right: measured power against phase, on the
expected curve). Components within about one bin of fs/2 (or of DC) overlap
their mirror image and are phase-dependent too.

**What to do.** For THD measurements, choose a fundamental whose harmonics don't
land exactly on fs/2.

## Harmonics above fs/2 or outside `bw`

Harmonics above `bw` or fs/2 are not analyzed. With a 300.7 Hz tone sampled at
1 kHz, even the 2nd harmonic (601.4 Hz) is above fs/2: `harm_freq` and `harm_db`
are empty arrays and `thd_db` is `nan`. In a real converter those harmonics fold
back (alias) below fs/2, where they show up as spurs that are counted as noise
and in THD+N, but not in THD.

## False tones from noise in `spec_analysis`

**What happens.** `spec_analysis` decides what is a tone by how far a peak stands
out from the noise (see {ref}`how tone detection works <tone-detection>`).
Noise always has peaks, so occasionally one crosses the threshold. The
`false_alarm_rate` parameter sets how often that may happen; on pure noise, the
measured rate is at or below the requested one:

```{eval-rst}
.. plot:: figures/cc_false_alarm.py
```

**What to do.** Lower `false_alarm_rate` (e.g. `1e-6`) to be stricter, at the
cost of missing tones that are only slightly above the noise; raise it to find
weaker tones. Tests that use random noise should seed the generator so their
results are reproducible.

## DC with any window

The DC level is correct for any window and for even and odd lengths (the tests
check Hann, Blackman-Harris and flat-top windows at 1000, 1001, 4095 and 4096
samples to 0.01 dB). With a rectangular window, a tone that is not on a bin leaks
into DC through the window's high sidelobes, so DC reads a few hundredths of a dB
off; use a tapered window when DC matters.

## Invalid input

Bad input fails immediately with a message that names the parameter, instead of
an obscure NumPy error or a wrong result:

| Call | Error |
|---|---|
| `harm_analysis(x[:16])` | `` ValueError: `x` must have at least 32 samples, got 16. `` |
| `harm_analysis(x.reshape(2, -1))` | `` ValueError: `x` must be a 1-D array, got an array with shape (2, 512). `` |
| `harm_analysis(x + 0j)` | `` TypeError: `x` must be real; complex input is not supported. `` |
| x containing NaN | `` ValueError: `x` contains NaN or infinite values. `` |
| `harm_analysis(np.zeros(n))` | `` ValueError: No fundamental tone found: the input signal is constant (all zeros or DC only). `` |
| `fs=0` | `` ValueError: `fs` must be a positive finite number, got 0. `` |
| bw=0.7 (with fs=1) | `` ValueError: `bw` must be in (0, fs / 2] = (0, 0.5], got 0.7. `` |
| bw inside the DC bins | `` ValueError: `bw` is inside the DC component of the spectrum, which extends to 0.00292969 (in the units of `fs`). Use a larger `bw`. `` |
| `window=np.ones(10)` | `` ValueError: `window` must have the same length as `x` (1024), got shape (10,). `` |
| `n_harm=-1` | `` ValueError: `n_harm` must be 0 or positive, got -1. `` |
| plot=True without ax | `` ValueError: An `ax` must be provided when `plot` is True. `` |
| `spec_analysis(x, false_alarm_rate=0)` | `` ValueError: `false_alarm_rate` must be in (0, 1), got 0. `` |
