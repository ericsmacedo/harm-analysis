# AGENTS.md

Guidance for AI coding agents working on this repository. Human contributors:
see [CONTRIBUTING.md](CONTRIBUTING.md).

## What this package is

`harm-analysis` estimates signal metrics from the FFT power spectrum of a 1-D
real signal:

- `harm_analysis(x, fs, ...) -> HarmResult`: signals with an injected tone.
  Finds the fundamental and harmonics; returns fundamental/harmonic/DC/noise
  power, THD, SNR, SINAD and THD+N.
- `spec_analysis(x, fs, ...) -> SpecResult`: signals without a known tone.
  Detects DC, tones and noise.
- CLI commands `harm-analysis` and `spec-analysis` wrap them for text files.

## Layout

| Path | Contents |
|---|---|
| `src/harm_analysis/__init__.py` | Public API: `harm_analysis`, `spec_analysis`, `HarmResult`, `SpecResult` |
| `src/harm_analysis/_harmonic.py` | `harm_analysis` and fundamental/harmonic bin search |
| `src/harm_analysis/_spectral.py` | `spec_analysis` and tone detection |
| `src/harm_analysis/_spectrum.py` | Shared helpers: input validation, windowed power spectrum, bin/peak search, dB conversion |
| `src/harm_analysis/_results.py` | `HarmResult` / `SpecResult` frozen dataclasses |
| `src/harm_analysis/_plotting.py` | Matplotlib plotting of the spectra |
| `src/harm_analysis/cli.py` | Click CLI |
| `src/harm_analysis/_harm_analysis.py` | Legacy re-export module, not used by the package |
| `tests/` | pytest tests |
| `examples/` | Example scripts and their captured output (`*.md`), included in the docs |
| `docs/` | Sphinx docs written in Markdown (MyST); `docs/conf.py` |
| `CHANGELOG.md` | Keep a Changelog format |

## Commands

The project uses [uv](https://docs.astral.sh/uv/) and invoke tasks.

```bash
uv run inv all          # pre-commit, tests, type check, docs: what CI runs
uv run inv test         # pytest (with coverage)
uv run inv checktypes   # ty check
uv run inv doc          # sphinx-build -W into site/
uv run inv pre-commit   # ruff check/format, codespell, license headers, ...
uv run pytest tests/test_harm_analysis.py -q   # a single test file
```

Run `uv run inv all` (or at least `pre-commit`, `test` and `checktypes`)
before committing. CI runs `inv all` on Linux, macOS and Windows with Python
3.10 to 3.13.

`uv run inv update-examples` re-runs the examples and overwrites
`examples/*.md` and `docs/images/*.png` with new random output. Only commit
those when the examples changed on purpose.

## Conventions

- **Public API**: everything public is exported from `harm_analysis/__init__.py`;
  modules starting with `_` are private.
- **Results**: analysis functions return `HarmResult` / `SpecResult`, never
  dicts or tuples. Scalars are Python `float`, arrays are `numpy.ndarray`
  (empty, not `None`, when there is nothing). Use `to_dict()` for
  printing/serialization.
- **Plotting** is a side effect on a caller-supplied `ax` (`plot=True, ax=ax`);
  the axes are not returned.
- **Errors**: invalid input raises `ValueError` / `TypeError` with a message
  naming the parameter. Never call `sys.exit` in library code.
- **dB values**: use `_to_db`; a power of zero is `-inf` dB without numpy warnings.
- **Docstrings**: numpy style, rendered by Sphinx (napoleon). Section headers
  are written without a colon (`Returns`, not `Returns:`); public docstrings
  end with `"""  # noqa: D416` so ruff does not add the colon.
- **Types**: annotate public functions; `ty` must pass. The package ships
  `py.typed`.
- **Style**: ruff (line length 120), enforced by pre-commit. New files need the
  MIT license header (the pre-commit hook inserts it).
- **Tests**: `filterwarnings = error` is set, so any warning fails a test. Seed
  random generators (`np.random.default_rng(seed)`) and write temporary files
  to pytest's `tmp_path`.
- **Changelog**: add an entry under *Unreleased* in `CHANGELOG.md` for
  user-visible changes.
- **Pull requests**: one fix or feature per PR.

## Known pitfalls

- Without `fund_freq`, the fundamental is the highest bin between DC and `bw`, so
  a stronger interferer is taken as the fundamental. With `fund_freq`, it is the
  tone within 3 bins of that frequency (`_peak_near` in `_harmonic.py`), which must
  stand out from the noise floor; other tones in the band count as noise.
- `spec_analysis` tone detection compares peaks with a 101-bin rolling median of
  the spectrum; the margin comes from `false_alarm_rate` (see
  `_detection_threshold_db` in `_spectral.py`, calibrated on white noise). Noise
  can still produce a false tone at about that rate, so tests that use random
  noise must be seeded.
- Signals must have at least 32 samples.
