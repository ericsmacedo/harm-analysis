
There are two main functions:

- `harm_analysis`: for simulations with an injected tone, returning SNR, THDN, Noise, etc.
- `spec_analysis`: for cases without an injected tone, that auto-detects DC, tones, and noise from the spectrum.


## `harm_analysis` example usage

This example generates a tone with noise, harmonics and a DC value.

```python
--8<-- "examples/example_usage.py"
```

outputs:

```
Parameters:
fund_db: 3.0103285457895366
fund_freq: 100.1299984247558
dc_db: -18.17423020970596
noise_db: -69.92423670970405
thd_db: -45.053400519370584
snr_db: 72.93456525549358
sinad_db: 45.04633222911549
thdn_db: -45.04633222911549
total_noise_and_dist: -42.036003683325944
```

![example_usage.py output plot](./images/example_usage.png)

## `spec_analysis` example usage

```python
--8<-- "examples/example_spectrum_analysis.py"
```

outputs:
```
Function results:
dc         [dB]: 0.12340127616098903
dc_db      [dB]: -18.173606979992375
noise_db   [dB]: -69.88787931686679
tones_amp_db [dB]: [-43.01019642   3.01027093 -43.01618807 -49.04094948]
tones_freq [dB]: [ 43.10254    100.08616524 200.3323352  300.11378463]
```

![example_spec_analysis.py output plot](./images/example_spec_analysis.png)
