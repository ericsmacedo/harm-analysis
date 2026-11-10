# API reference

## At a glance

```python
from harm_analysis import harm_analysis, spec_analysis, HarmResult, SpecResult
```

| Name | What it does |
|---|---|
| {func}`~harm_analysis.harm_analysis` | Analyze a signal with an injected tone: fundamental, harmonics, DC, noise, THD, SNR, SINAD and THD+N. Returns a {class}`~harm_analysis.HarmResult`. |
| {func}`~harm_analysis.spec_analysis` | Detect DC, tones and noise in a signal without a known tone. Returns a {class}`~harm_analysis.SpecResult`. |
| {class}`~harm_analysis.HarmResult` | Results of `harm_analysis`: one attribute per metric, `to_dict()` for all of them. |
| {class}`~harm_analysis.SpecResult` | Results of `spec_analysis`. |

Conventions shared by both functions:

- **Input**: a 1-D, real, finite array of at least 32 samples, and the sampling
  frequency `fs`. Frequencies in the results are in the units of `fs`.
- **Output**: powers in dB relative to the signal's units squared (a sine of
  amplitude 1 is −3.01 dB). A power of zero is `-inf` dB. Arrays (harmonics,
  tones) are NumPy arrays, empty when there is nothing.
- **Plotting**: pass `plot=True` and a Matplotlib `ax`; the spectrum is drawn on
  that `ax`, which you can keep customizing.
- **Errors**: invalid input raises `ValueError` or `TypeError` with a message that
  names the parameter (see [Invalid input](corner-cases.md#invalid-input)).

How each metric is computed is explained in [How it works](how-it-works.md).

## Functions

```{eval-rst}
.. autofunction:: harm_analysis.harm_analysis

.. autofunction:: harm_analysis.spec_analysis
```

## Results

Both functions return a result object with one attribute per metric. Use
`to_dict()` to get the values as a dictionary, for example to print or serialize
them.

```{eval-rst}
.. autoclass:: harm_analysis.HarmResult
   :members: to_dict

.. autoclass:: harm_analysis.SpecResult
   :members: to_dict
```
