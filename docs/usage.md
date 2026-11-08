# Usage


There are two main functions:

- `harm_analysis`: for simulations with an injected tone, returning SNR, THDN, Noise, etc.
- `spec_analysis`: for cases without an injected tone, that auto-detects DC, tones, and noise from the spectrum.

## Results

`harm_analysis` returns a `HarmResult` and `spec_analysis` returns a `SpecResult`
(see the [API](api.md)). Each metric is an attribute, and `to_dict()` returns all
of them as a dictionary:

```python
results = harm_analysis(x, fs=fs)

results.snr_db     # signal-to-noise ratio in dB
results.thd_db     # total harmonic distortion in dB
results.harm_freq  # frequencies of the harmonics (empty array if none)

for name, value in results.to_dict().items():
    print(f"{name}: {value}")
```

## Plotting

Pass `plot=True` and a Matplotlib `ax` to draw the power spectrum. The function
draws on the `ax` you pass in, so you can keep customizing it afterwards:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
results = harm_analysis(x, fs=fs, plot=True, ax=ax)

ax.set_title("My measurement")
ax.set_xscale("log")
fig.savefig("spectrum.png")
```


## `harm_analysis` example usage

This example generates a tone with noise, harmonics and a DC value.

```{literalinclude} ../examples/example_usage.py
:language: python
```

outputs:

```{literalinclude} ../examples/example_usage.md
:language: text
```

![example_usage.py output plot](./images/example_usage.png)

## `spec_analysis` example usage

```{literalinclude} ../examples/example_spec_analysis.py
:language: python
```

outputs:

```{literalinclude} ../examples/example_spec_analysis.md
:language: text
```

![example_spec_analysis.py output plot](images/example_spec_analysis.png)
