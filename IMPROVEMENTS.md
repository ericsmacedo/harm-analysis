# Improvements

Tracking list of planned improvements. Check items off as they are completed.

## Bugs and safety

- [ ] **1. Remove `eval(sfactor)` in `cli.py`.** It executes arbitrary code from a CLI argument. Parse with `fractions.Fraction` or `float` instead.
- [ ] **2. Fix the spec-analysis CLI output.**
  - It skips `"tones_amp_db"` but the results key is `tones_db`, so the skip never matches and the arrays are printed in the main table.
  - The results table is only printed when tones exist, so a signal with no tones shows only "No tones detected".
- [ ] **3. Handle `log10(0)` in `harm_analysis`.** DC, noise or harmonic power of zero (e.g. pure synthetic tone) gives `-inf` and a numpy warning.
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

## Suggested order

Start with 1, 2 and 3: small, concrete fixes.
