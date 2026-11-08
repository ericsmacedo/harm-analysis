# Improvements

Tracking list of planned improvements. Check items off as they are completed.

## Bugs and safety

- [x] **1. Remove `eval(sfactor)` in `cli.py`.** It executes arbitrary code from a CLI argument. Now parsed by a restricted evaluator (numbers, `1e5`, `+ - * /`, parentheses, `log2`). [#22](https://github.com/ericsmacedo/harm-analysis/pull/22)
- [x] **2. Fix the spec-analysis CLI output.** [#24](https://github.com/ericsmacedo/harm-analysis/pull/24)
  - It skips `"tones_amp_db"` but the results key is `tones_db`, so the skip never matches and the arrays are printed in the main table.
  - The results table is only printed when tones exist, so a signal with no tones shows only "No tones detected".
- [x] **3. Handle `log10(0)` in `harm_analysis`.** DC, noise or harmonic power of zero (e.g. all-zero or DC-only input) gives `-inf` and a numpy warning. [#23](https://github.com/ericsmacedo/harm-analysis/pull/23)
- [ ] **4. Fail loudly when the fundamental is not the highest component.** Currently documented as "fails"; raise a clear error or warn instead.

## Design

- [ ] **5. Consistent return type.** `harm_analysis` returns a dict or a `(dict, ax)` tuple depending on `plot`. Use a typed result (dataclass or `TypedDict`), or always return the dict and put the axes under a separate key.
- [ ] **6. Fix docstring mismatches.**
  - "first five harmonics" vs configurable `n_harm`.
  - `harm_freq` is computed twice.
  - `harm_freq` documented as Hz but is in units of `fs`.
  - Type the `ax` return.
- [ ] **7. Remove duplication in `cli.py`.** Extract helpers for plot setup (title, grid, formatter, `plt.show`) and results-table building shared by both commands.
- [ ] **8. Input validation.** Check that `x` is 1-D, real, long enough, and that `bw`/`fs` are valid. Decide and document complex-input support.

## Quality and tooling

- [ ] **9. Tests and CI.**
  - Edge cases: `n_harm=0`, harmonics aliasing or outside `bw`, custom windows, DC-only input, invalid `sfactor`.
  - ~~Verify CI runs mypy.~~ Done: the project now uses `ty`, run by `inv checktypes` as part of `inv all` in CI.
- [ ] **10. Repo hygiene.** Remove the duplicate `test_data_cli.txt` at the repo root (kept in `tests/`).
- [ ] **11. Packaging.** Add `py.typed` and a changelog.

## Found later

- [ ] **12. `harm_analysis` crashes when all harmonics are outside the bandwidth.** `_find_harm` calls `np.hstack([])` on an empty list and raises `ValueError: need at least one array to concatenate` (e.g. 300 Hz tone at fs=1000, or `bw=120` with a 100 Hz fundamental). The docstring says `thd_db` should be `nan` in this case, and `harm_analysis` already handles `harm_bins.size == 0`, but that code is never reached.
- [ ] **13. Replace `sys.exit(...)` in `_find_bins` with a `ValueError`.** When `bw` is inside the DC bins the library call kills the whole process.
- [ ] **14. Spurious tones in `spec_analysis`.** The detector occasionally reports an extra tone from noise (about 1 in 300 runs on the `test_spec_analysis` signal, which made CI flaky; the test is now seeded). Investigate the threshold (rolling median + 16 dB) and add a more robust criterion.

## Tooling and docs (done outside the original list)

- [x] **Switch the type checker from mypy to ty.** [#25](https://github.com/ericsmacedo/harm-analysis/pull/25)
- [ ] **Migrate the docs from MkDocs to Sphinx + Furo, keeping Markdown (MyST).** PR open: [#26](https://github.com/ericsmacedo/harm-analysis/pull/26). Check that CI passes: the intersphinx inventories could not be fetched in the sandbox, so that part was not verified locally.

## AI-friendly package

Ordered by value for effort. See also items 4, 5, 8, 11, 12 and 13, which improve error messages, return types and annotations.

- [ ] **15. `AGENTS.md` / `CLAUDE.md`** at the repo root: layout, commands (`uv run inv all`, `inv doc`, `inv checktypes`), conventions (numpy docstrings, ruff, one fix per PR).
- [ ] **16. `llms.txt` / `llms-full.txt`** generated from the Sphinx docs (e.g. with `sphinx-llms-txt`).
- [ ] **17. Runnable docstring `Examples`** for `harm_analysis` and `spec_analysis`, tested as doctests.
- [ ] **18. `--json` flag for the CLI commands** so the output is machine-readable.
- [ ] **19. Optional MCP server** exposing `harm_analysis` and `spec_analysis` as tools, shipped as an extra (`harm-analysis[mcp]`) with a `harm-analysis-mcp` entry point, plus a skill file explaining when and how to use it. Prototype (tested): a `mcp_server.py` using `MCPServer` from `mcp.server.mcpserver` (in `mcp` 2.x `FastMCP` was renamed, so pin the SDK range) and a helper to convert numpy results to plain Python for JSON.

## Notes

- Item 4: the fundamental is simply the highest bin between DC and `bw`, so a stronger interferer is silently reported as the fundamental. Options: warn when another non-harmonic peak has similar power, and/or add an optional `fund_freq` argument.
- Item 10: `tests/test_cli.py` writes `test_data_cli.txt` to the working directory, which is why the root copy exists. Use `tmp_path` in the test first.

## Suggested order

Items 1–3 are done, and so are the `ty` switch (#25) and, once merged, the Sphinx migration (#26). Next: 12 and 13 together (small, concrete), then 4, then 5 and 8 (they underpin the AI-friendly work), then 6, 7, 9, 10, 11, 14, and finally 15–19.
