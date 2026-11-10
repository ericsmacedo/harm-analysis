# Command line interface

Two commands run the analyses on a text file of samples, without writing any
Python:

| Command | Runs | Use it for |
|---|---|---|
| `harm-analysis` | {func}`harm_analysis.harm_analysis` | a signal with an injected tone: SNR, THD, SINAD, THD+N |
| `spec-analysis` | {func}`harm_analysis.spec_analysis` | a signal without a known tone: DC, tones and noise |

Both are installed with the package (`pip install harm-analysis`).

## Input file

A text file of time-domain samples, by default separated by any whitespace (one
sample per line works). Use `--sep` for other separators, e.g. `--sep ","` for
comma-separated values. The examples on this page use
[`signal.txt`](https://github.com/ericsmacedo/harm-analysis/blob/main/docs/cli_examples/signal.txt):
4096 samples at 1 kHz of a 100.3 Hz tone with a 2nd harmonic at −40 dB, a 3rd at
−50 dB, a DC offset of 0.1 and a little noise.

## Basic use

Pass the file and the sampling frequency. Frequencies in the results are in the
units of `--fs`:

```bash
harm-analysis signal.txt --fs 1000
```

```{literalinclude} cli_examples/harm.txt
:language: text
```

`spec-analysis` lists the tones it detects:

```bash
spec-analysis signal.txt --fs 1000
```

```{literalinclude} cli_examples/spec.txt
:language: text
```

## Machine-readable output: `--json`

`--json` prints the results as one JSON object, for scripts and AI tools. Values
that are not finite (`NaN`, or `-inf` dB for a power of zero) are written as
`null`:

```bash
harm-analysis signal.txt --fs 1000 --json
```

```{literalinclude} cli_examples/harm_json.txt
:language: json
```

## The tone you injected: `--fund-freq`

If the data may contain a component stronger than your test tone, give its
frequency (see
[A stronger tone than the one you injected](corner-cases.md#a-stronger-tone-than-the-one-you-injected)).
Here [`interferer.txt`](https://github.com/ericsmacedo/harm-analysis/blob/main/docs/cli_examples/interferer.txt)
has a 100.3 Hz tone next to an interferer twice as large at 150.7 Hz:

```bash
harm-analysis interferer.txt --fs 1000 --fund-freq 100.3 --json
```

```{literalinclude} cli_examples/harm_fund_freq.txt
:language: json
```

## ADC codes and scaling: `--sfactor`

`--sfactor` multiplies the data before the analysis, for example to turn ADC
codes into volts or full-scale units. It accepts numbers (including `1e5`),
`+ - * /`, parentheses and `log2`, e.g. `1/32768`, `3.3/4096`, `1/2048/2`. Here
[`adc_codes.csv`](https://github.com/ericsmacedo/harm-analysis/blob/main/docs/cli_examples/adc_codes.csv)
holds comma-separated 16-bit codes of a 10.3 kHz tone at 0.9 of full scale,
sampled at 1 MHz:

```bash
harm-analysis adc_codes.csv --fs 1e6 --sep "," --sfactor "1/32768" --json
```

```{literalinclude} cli_examples/adc_codes.txt
:language: json
```

`fund_db` ≈ −3.93 dB is what a sine of amplitude 0.9 gives in full-scale units
($20 \log_{10}(0.9/\sqrt{2})$).

## Plot: `--plot`

`--plot` opens a window with the data and its power spectrum:

```bash
harm-analysis signal.txt --fs 1000 --plot
```

```{eval-rst}
.. plot:: figures/cli_plot.py
```

## Reference

```{eval-rst}
.. click:: harm_analysis.cli:harm_analysis_cmd
   :prog: harm-analysis
   :nested: full

.. click:: harm_analysis.cli:spec_analysis_cmd
   :prog: spec-analysis
   :nested: full
```
