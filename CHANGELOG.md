# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.0.0] - 2026-11-10

This release contains breaking changes; see *Changed (breaking)* below.

### Changed (breaking)

- `harm_analysis` and `spec_analysis` now always return a `HarmResult` /
  `SpecResult` object instead of a `dict`, or a `(dict, Axes)` tuple when
  `plot=True`. Access results as attributes (`results.snr_db`) or convert them
  with `results.to_dict()`. With `plot=True` the spectrum is drawn on the `ax`
  you pass in, which is no longer returned.
  ([#27](https://github.com/ericsmacedo/harm-analysis/pull/27))
- `HarmResult.harm_freq` is an empty array instead of `None` when there are no
  harmonics. ([#27](https://github.com/ericsmacedo/harm-analysis/pull/27))
- Invalid inputs now raise `ValueError` / `TypeError` with a clear message
  instead of failing inside numpy, being silently accepted, or exiting the
  process: `x` must be real, 1-D, at least 32 samples long and finite; `fs` must
  be positive; `window` must match the length of `x`; `bw` must be in
  `(0, fs / 2]`; `n_harm` must not be negative. Complex input is not supported.
  ([#30](https://github.com/ericsmacedo/harm-analysis/pull/30))
- `plot=True` without an `ax` raises a `ValueError`. ([#27](https://github.com/ericsmacedo/harm-analysis/pull/27))

### Changed

- Restyled the spectrum plots (`plot=True` and the CLI's `--plot`): the noise is a thin
  gray background trace, the fundamental/tones and harmonics stand out in colors that
  are distinguishable with color-vision deficiencies, peaks are labeled with their level
  (e.g. "H2 -43.0 dB"), and the legend shows the integrated noise values (at `bw` when it
  is given, matching `noise_db`). Only the given `ax` is styled.
- Harmonics in the plots are numbered by their order, as in MATLAB: the 2nd harmonic is
  "H2" (it was labeled "H1").

### Added

- Documentation: "How it works" (the algorithm and the definition of every metric),
  "Corner cases" (interferers, shaped noise, noise-free tones, fs/2, false tones,
  invalid input) with figures generated from the code at build time, CLI examples with
  real output, an API overview and a table of contents on the home page.
- `fund_freq` parameter of `harm_analysis` (and `--fund-freq` for the `harm-analysis`
  command, `fund_freq` for the MCP tool): the expected frequency of the injected
  tone, so a stronger interferer is no longer taken as the fundamental. It raises a
  `ValueError` if there is no tone near that frequency.
  ([#42](https://github.com/ericsmacedo/harm-analysis/issues/42))
- `false_alarm_rate` parameter of `spec_analysis` (default `1e-3`): the target
  probability that noise alone produces a false tone in a spectrum.
- `HarmResult` and `SpecResult` are exported from `harm_analysis`.
  ([#27](https://github.com/ericsmacedo/harm-analysis/pull/27))
- `--sfactor` accepts arithmetic expressions: numbers (including `1e5`),
  `+ - * /`, parentheses and `log2`. ([#22](https://github.com/ericsmacedo/harm-analysis/pull/22))
- `py.typed` marker, so type checkers use the package's type hints.
- `--json` option for `harm-analysis` and `spec-analysis` to print the results as
  JSON (non-finite values as `null`).
- `llms.txt` and `llms-full.txt` (see [llmstxt.org](https://llmstxt.org)) at the root of
  the documentation site, so LLMs and AI tools can read the docs, including the
  API reference, as plain text.
- Runnable examples in the `harm_analysis` and `spec_analysis` docstrings, tested
  as doctests.
- MCP server for AI assistants (`harm-analysis-mcp`, optional extra
  `harm-analysis[mcp]`) with `run_harm_analysis` and `run_spec_analysis` tools, and a
  skill file (`skills/harm-analysis/SKILL.md`) describing how to use the package.

### Fixed

- The `spec-analysis` tones table header showed "Amplitude" instead of "Amplitude [dB]"
  (rich read "[dB]" as a style tag); it is now "Power [dB]", which is what the column holds.
- `harm_analysis` and `spec_analysis` crashed with a `TypeError` on a noise-free tone
  that is not on an FFT bin, for signals of up to 256 samples: the tone's peak grew
  over the whole spectrum (DC included) and left no noise bins. Peaks no longer extend
  into the DC bins, and when no noise bins are left the noise power is 0 (`-inf` dB,
  infinite SNR). Harmonics no longer reuse bins of the fundamental, which counted its
  leakage as distortion in short signals (THD of −20 dB for a pure tone of 32 samples).
  ([#41](https://github.com/ericsmacedo/harm-analysis/issues/41))
- Components exactly at fs/2 (a tone in `spec_analysis`, or a harmonic such as the 2nd
  harmonic of a fundamental at fs/4 in `harm_analysis`) were measured 2.2 dB too high,
  because the fs/2 bin of even-length signals was doubled like the other bins. In that
  case `total_noise_and_dist_db` was also `nan` with a warning. A tone at fs/2 is now
  reported at exactly fs/2.
  ([#49](https://github.com/ericsmacedo/harm-analysis/issues/49))
- `harm_analysis` reported `dc_db` too high by the window's equivalent noise bandwidth
  (+1.76 dB with the default Hann window, +3.02 dB with Blackman-Harris). DC detection
  also stopped after the first bin for windows with a flat main lobe, which made
  `spec_analysis` report DC 5.76 dB too low with the flat-top window. DC power is now
  correct for any window and for even and odd lengths.
  ([#48](https://github.com/ericsmacedo/harm-analysis/issues/48))
- `spec_analysis` occasionally reported a noise peak as a tone (about 1 in 300
  spectra of 2^18 samples). The default detection threshold is now derived from
  `false_alarm_rate` and the number of bins (about 13.5 dB above the local noise
  floor for 1024 samples, 15 dB for 2^18), and the noise floor is estimated over
  101 bins instead of 51. Weak tones are detected about 1.5-2 dB closer to the
  noise floor than with the previous fixed 16 dB threshold.
  ([#43](https://github.com/ericsmacedo/harm-analysis/issues/43))
- `harm_analysis` crashed with `ValueError: need at least one array to concatenate`
  when all harmonics were outside the band (above `bw` or `fs / 2`); it now returns
  `thd_db = nan` and empty harmonic arrays.
  ([#39](https://github.com/ericsmacedo/harm-analysis/issues/39))
- `harm_analysis` called `sys.exit` when `bw` fell inside the DC component of the
  spectrum, terminating the whole process; it now raises a `ValueError`.
  ([#40](https://github.com/ericsmacedo/harm-analysis/issues/40))
- `--sfactor` was evaluated with `eval()`, allowing arbitrary code execution.
  ([#22](https://github.com/ericsmacedo/harm-analysis/pull/22))
- `harm_analysis` crashed with unhelpful errors on all-zero or DC-only input; it
  now raises a clear `ValueError`. Powers of zero are reported as `-inf` dB
  without numpy warnings. ([#23](https://github.com/ericsmacedo/harm-analysis/pull/23))
- `spec-analysis` printed the raw tone arrays in the parameters table and showed
  no results when no tones were detected. ([#24](https://github.com/ericsmacedo/harm-analysis/pull/24))
- Docstrings: THD uses `n_harm` harmonics (not five), parameter defaults and
  units, `spec_analysis` peak-detection parameters. ([#28](https://github.com/ericsmacedo/harm-analysis/pull/28))

### Internal

- Type checking with ty instead of mypy. ([#25](https://github.com/ericsmacedo/harm-analysis/pull/25))
- Documentation built with Sphinx and the Furo theme, still written in Markdown.
  ([#26](https://github.com/ericsmacedo/harm-analysis/pull/26))
- CLI code de-duplicated; more edge-case tests; CLI tests no longer write files
  to the repository. ([#29](https://github.com/ericsmacedo/harm-analysis/pull/29), [#31](https://github.com/ericsmacedo/harm-analysis/pull/31),
  [#32](https://github.com/ericsmacedo/harm-analysis/pull/32))

## [1.4.1] - 2025-10-31

### Fixed

- Missing parameters in docstrings. ([#19](https://github.com/ericsmacedo/harm-analysis/pull/19))
- Start/end detection of spectral peaks. ([#20](https://github.com/ericsmacedo/harm-analysis/pull/20))

## [1.4.0] - 2025-10-04

### Changed

- Harmonic markers in plots start with `H`. ([#14](https://github.com/ericsmacedo/harm-analysis/pull/14))

### Fixed

- Return an empty array instead of `None`. ([#16](https://github.com/ericsmacedo/harm-analysis/pull/16))
- Detection of a tone at fs/2. ([#17](https://github.com/ericsmacedo/harm-analysis/pull/17))
- Documentation typos and examples. ([#11](https://github.com/ericsmacedo/harm-analysis/pull/11),
  [#13](https://github.com/ericsmacedo/harm-analysis/pull/13), [#18](https://github.com/ericsmacedo/harm-analysis/pull/18))

## [1.3.0] - 2025-09-28

### Added

- `spec_analysis` for signals without an injected tone, with tone detection.
  ([#7](https://github.com/ericsmacedo/harm-analysis/pull/7), [#10](https://github.com/ericsmacedo/harm-analysis/pull/10))
- Markers on the fundamental and harmonics in plots. ([#6](https://github.com/ericsmacedo/harm-analysis/pull/6))
- Examples. ([#5](https://github.com/ericsmacedo/harm-analysis/pull/5))

## Earlier releases

See the [GitHub releases](https://github.com/ericsmacedo/harm-analysis/releases).

[Unreleased]: https://github.com/ericsmacedo/harm-analysis/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/ericsmacedo/harm-analysis/compare/v1.4.1...v2.0.0
[1.4.1]: https://github.com/ericsmacedo/harm-analysis/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/ericsmacedo/harm-analysis/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/ericsmacedo/harm-analysis/compare/v1.2.1...v1.3.0
