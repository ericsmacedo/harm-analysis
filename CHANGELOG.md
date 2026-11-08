# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

This release contains breaking changes and will be released as 2.0.0.

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

### Added

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

[Unreleased]: https://github.com/ericsmacedo/harm-analysis/compare/v1.4.1...HEAD
[1.4.1]: https://github.com/ericsmacedo/harm-analysis/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/ericsmacedo/harm-analysis/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/ericsmacedo/harm-analysis/compare/v1.2.1...v1.3.0
