# Usage


There are two main functions:

- `harm_analysis`: for simulations with an injected tone, returning SNR, THDN, Noise, etc.
- `spec_analysis`: for cases without an injected tone, that auto-detects DC, tones, and noise from the spectrum.


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
