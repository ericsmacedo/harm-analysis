# Improvements

Tracking list of planned improvements. Check items off as they are completed.

## Bugs and safety

- [x] **1. Remove `eval(sfactor)` in `cli.py`.** It executes arbitrary code from a CLI argument. Parse with `fractions.Fraction` or `float` instead.
- [x] **2. Fix the spec-analysis CLI output.**
  - It skips `"tones_amp_db"` but the results key is `tones_db`, so the skip never matches and the arrays are printed in the main table.
  - The results table is only printed when tones exist, so a signal with no tones shows only "No tones detected".
- [x] **3. Handle `log10(0)` in `harm_analysis`.** DC, noise or harmonic power of zero (e.g. all-zero or DC-only input) gives `-inf` and a numpy warning.
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
  - Verify CI runs mypy (it is a dev dependency).
- [ ] **10. Repo hygiene.** Remove the duplicate `test_data_cli.txt` at the repo root (kept in `tests/`).
- [ ] **11. Packaging.** Add `py.typed` and a changelog.

## Found later

- [ ] **12. `harm_analysis` crashes when all harmonics are outside the bandwidth.** `_find_harm` calls `np.hstack([])` on an empty list and raises `ValueError: need at least one array to concatenate` (e.g. 300 Hz tone at fs=1000, or `bw=120` with a 100 Hz fundamental). The docstring says `thd_db` should be `nan` in this case, and `harm_analysis` already handles `harm_bins.size == 0`, but that code is never reached.
- [ ] **13. Replace `sys.exit(...)` in `_find_bins` with a `ValueError`.** When `bw` is inside the DC bins the library call kills the whole process.
- [ ] **14. Spurious tones in `spec_analysis`.** The detector occasionally reports an extra tone from noise (about 1 in 300 runs on the `test_spec_analysis` signal, which made CI flaky; the test is now seeded). Investigate the threshold (rolling median + 16 dB) and add a more robust criterion.

## Notes

- Item 4: the fundamental is simply the highest bin between DC and `bw`, so a stronger interferer is silently reported as the fundamental. Options: warn when another non-harmonic peak has similar power, and/or add an optional `fund_freq` argument.
- Item 10: `tests/test_cli.py` writes `test_data_cli.txt` to the working directory, which is why the root copy exists. Use `tmp_path` in the test first.

## Suggested order

Items 1–3 are done. Next: 4, then 12 and 13 (small, concrete), then the rest.
