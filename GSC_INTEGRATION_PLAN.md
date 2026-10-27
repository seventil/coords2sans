# Integration plan: coords2sans into gsc and shapespyer

Status: 2026-10-27. Draft for review.

- gsc: `GenericScatteringCalculator`, SANS-fitter branch `gsc`,
  `src/sans_fitter/structure/`, commit `2e9674e` (the newest on the branch).
- coords2sans: this repo, not committed yet. The exact average (Debye sum through a
  pair-distance histogram) is now its default.
- SasView: commit `02363ba`, the one `examples/compare.ipynb` downloads.

## 1. Decision

gsc becomes the core calculator. coords2sans features move into gsc. Multiprocessing
and project glue go into a wrapper in shapespyer.

- **Why gsc:** its physics is exact, and it already has provenance (`get_settings()`),
  progress and cancel, and `to_data1d()`. `to_data1d()` feeds SANSFitter fits, which a
  future Experiments tab in shapes-app can use.
- **Cost:** `import sans_fitter.structure` loads the whole `sans_fitter` package:
  sasmodels, bumps, sasdata, plotly and scipy. shapespyer would take on this dependency.
  **To check:** is this acceptable for shapespyer?

## 2. Where coords2sans stands

The earlier plan (`ALES_GSC_PLAN.md`) listed work for coords2sans "before implementing":

| Item | Now |
|---|---|
| Exact Debye sum, or measure the error for rods | **done:** the default; the old ⟨A⟩² is the opt-in `average_amplitude = True` |
| State the atomic-volume problem | **done** in the README (section 2); not corrected |
| Check solvent SLD units | not done |
| Unwrap PDB files with `CRYST1` | not done (`.gro` only) |
| Points view and Rg | not done |

Three of that plan's "weak spots" are handled:

- `sld()` raises an error for an unknown `deuterated` name.
- `structure.centre` is public.
- The settings are plain attributes, so you can change one and calculate again.

## 3. Measured facts

All numbers are for the CTA178 eq5 micelle in `examples/data`: 178 CTA molecules,
11,036 atoms. It is a short rod, 99 × 64 × 56 Å, with principal radii of gyration of
22, 12 and 11 Å.

### 3.1 The averaged amplitude is wrong for this rod

Two ways average over the directions of the micelles in a sample:

- **exact average:** average the intensity. This is Debye's sum over all atom pairs,
  Σ w_j·w_k·sinc(q r_jk), with w = (SLD − solvent SLD)·V. SasView `GenSAS` with
  `is_avg = False`, and gsc.
- **averaged amplitude, ⟨A⟩²:** average the amplitude, then square it. SasView `GenSAS`
  with `is_avg = True`, and the polyJD script. Exact only for round objects.

| q (Å⁻¹) | 0.02 | 0.04 | 0.06 | 0.095 | 0.144 |
|---|---|---|---|---|---|
| ⟨A⟩² / exact | 0.999 | 0.991 | 0.953 | 0.647 | 0.009 |

The ratio is the same in H2O and D2O. So curves from polyJD and from `GenSAS` with
`is_avg = True` are right only below about q = 0.05 Å⁻¹ for this micelle. Above that,
⟨A⟩² shows a deep minimum near 0.14 Å⁻¹ that the exact curve does not have.

### 3.2 The histogram makes the exact average fast

coords2sans computes the exact average in two steps (`SansCalculator._pair_histogram`,
then `_sinc_sum`):

1. One pass over all atom pairs. Each pair adds w_j·w_k to a bin at its distance,
   rounded to `BIN_WIDTH` = 0.01 Å. The pass works on blocks of about a million pairs,
   with distances from |a|² + |b|² − 2a·b (a matrix product).
2. For each q: Σ bin · sinc(q r) over the bins. For this micelle that is about 13,000
   bins instead of 60 million pairs.

A size f stretches every distance by f and every amplitude by f³, so it gives
f⁶·I₁(f q). All sizes of a spread use the same bins.

| Kernel | One size | 35 sizes | Time grows with |
|---|---|---|---|
| every pair at every q: gsc `debye_pairs` (40 q) | 48 s | about 28 min | atoms² × q |
| every pair at every q: SasView `sasview_sans_debye` (32 q) | 45 s | about 26 min | atoms² × q |
| coords2sans histogram (256 q) | 2.3 s | 4.6 s | atoms² for the pass, then bins × q per size |
| coords2sans ⟨A⟩² (256 q) | 0.07 s | 1.9 s | atoms × q |

Accuracy, from `examples/compare.ipynb`:

- the histogram against SasView's pair sum: 1e-4 (the rounding of the distances);
- ⟨A⟩² against `GenSAS` with `is_avg = True`: 1e-13;
- a spread of sizes against the sasmodels `sphere` model: 4e-6.

**The limit:** the pass still grows as atoms². It takes about 6 min for 130,000 atoms
and about 50 min for a 400,000-atom vesicle. For such sizes the options are ⟨A⟩² (if
the object is round), or gsc's lattice (FFT) kernel. gsc uses that kernel only for
pixel data; atoms would first have to be put on a grid. This is not tested.

**A variant, for many contrasts:** w = b − ρ_solvent·v (scattering length minus the
displaced solvent). So one histogram of w·w can be split into three, of b·b, b·v and
v·v. Then any solvent SLD costs almost nothing. A prototype took 7 s once, then 0.01 s
per contrast, with an error of 3e-5 in D2O and 6e-4 in H2O. coords2sans keeps one
histogram (2.3 s per contrast), which is simpler. A change of deuteration changes b,
so it needs new histograms either way.

### 3.3 The atom volumes are too big

One CTA gets 1194 Å³ from the elements' bulk volumes. Most of it is hydrogen:
42 × 23.6 Å³ = 993 Å³. A realistic CTA volume is about 560 Å³ (an estimate). So
I(0) is about 2× too high in D2O and 10× too high in H2O, and the match point moves.

coords2sans, gsc, SasView and polyJD all have this problem. gsc at least warns about
it. Nothing corrects it yet (see G9).

## 4. SasView findings

- **`GenSAS` with `is_avg = False`** calls
  `sas.sascalc.calculator.ausaxs.ausaxs_sans_debye.evaluate_sans_debye`. That uses the
  AUSAXS library (`pyausaxs`) when it is installed. AUSAXS also counts pairs by
  distance. Without it, SasView falls back to `sasview_sans_debye.py`: plain numpy,
  every pair at every q. `examples/compare.ipynb` uses this fallback as its reference.
- **A bug in `geni.Iq`** (`src/sas/sascalc/calculator/geni.py`, line 43). The function
  drops atoms whose contrast is exactly 0, but the names are in the wrong order:

  ```python
  coords, sld, vol = (v[index] for v in (sld, coords, vol))
  ```

  So `coords` gets the SLDs and `sld` gets the positions, and the call fails. It
  happens only when an atom's SLD equals the solvent SLD, so no parity test uses such
  an atom. **To decide:** report it to SasView?
- **`PDBReader`** reads only `ATOM` records, not `HETATM`. An unknown element gets
  SLD 0 but no volume, so the volume array is shorter than the positions. Parity tests
  with PDB files must avoid both.

## 5. Into gsc

These features are generic: useful for anyone with MD or PDB files.

| # | Feature | Source | Note |
|---|---|---|---|
| G1 | `.gro` reader, and residue arrays (`resids`, `resnames`, `names`) on `Structure` | coords2sans `Structure.read` | the base for G2–G5 and G9 |
| G2 | residue selection (`residues=`) | coords2sans | replaces the fixed water list (`HOH`, `DOD`, `WAT`); today gsc keeps `SOL` and `TIP3` |
| G3 | periodic unwrapping: the `.gro` box, and PDB `CRYST1` | coords2sans, extended | coords2sans does `.gro` only |
| G4 | elements from masses or element lists per residue, `element_sources`, a warning on a guess from the atom name | coords2sans | fixes ` NA ` read as nitrogen, and `C21` read as `C2` (double volume) |
| G5 | per-residue deuteration, as `structure.deuterated([...])`, which returns a new frozen `Structure` | coords2sans `Structure.sld(deuterated)` | fits gsc's frozen style |
| G6 | size polydispersity: Gaussian, ±3σ, 35 sizes, the sasmodels rule; through `set_params(polydispersity=…, polydispersity_n=…)` | coords2sans `SansCalculator._sizes` | needs G7: then 35 sizes take 4.6 s, not about 28 min |
| G7 | pair-histogram Debye kernel: a third method in `_debye`, next to `lattice` and `pairs`, with the same progress callback per block | coords2sans `_pair_histogram`, `_sinc_sum` | to agree: `BIN_WIDTH`, and one histogram or three (section 3.2) |
| G8 | ⟨A⟩² kernel as an opt-in (for example `kernel='average'`), marked approximate | coords2sans `average_amplitude` | for polyJD and SasView parity, and for round objects above 100,000 atoms |
| G9 | `molecular_volumes={'CTA': 560}`: scales the atom volumes inside each residue and keeps b | new | fixes the contrast and I(0) problem that gsc only warns about |
| G10 | parity tests and the compare notebook: the SasView pair sum, `GenSAS`, the sasmodels `sphere`, the polyJD loops | coords2sans `examples/compare.ipynb` | tests avoid the SasView traps of section 4 |

Not ported to gsc:

- `Curve`: gsc's `ScatteringProfile` already covers it (`save_csv`, `to_data1d`,
  `plot`).
- The solvent SLD in Å⁻². gsc uses 10⁻⁶ Å⁻² (the sasmodels unit) and warns about values
  that look like Å⁻².
- The normalisation by the volume of all atoms. gsc normalises by the volume of the
  atoms that scatter ("active volume"), which is fine.

## 6. Into the shapespyer wrapper (`shapes.project.analysis`)

| # | Feature | Why here |
|---|---|---|
| S1 | a picklable task (file paths and settings, no arrays) that builds a gsc calculator inside the child process. It runs through Job or `ProcessPoolExecutor`, over frames × contrasts × stages. coords2sans `SansTask` is the model | process handling belongs to the app, not the library |
| S2 | progress and cancel: gsc `calculate(progress=…)` reports to Job progress; returning True raises `CalculationInterrupted`, which cancels the Job | gsc already has the hook; it matters once the Job hang bug is fixed |
| S3 | masses from shapespyer's ITP reader (`Species.masses`), passed to gsc as `masses=` | gsc stays free of topology readers |
| S4 | the aggregate's residues and the deuteration choices, from the project's species | shapespyer knows the composition |
| S5 | the curves stored in the project, with `get_settings()` as their provenance | project format |
| S6 | the default kernel by atom count (histogram up to about 100,000 atoms, about 3 min; above that ⟨A⟩² or lattice), and the SLD unit conversion at the boundary (Å⁻² in shapespyer, 10⁻⁶ Å⁻² in gsc) | app policy |

## 7. Order

1. **Agree on the scope with the gsc maintainer** (Piotr Rozyczko, author of the gsc
   commit): residue arrays on `Structure`; the new `set_params` keys; how to choose the
   kernel; one histogram or three; where molecular volumes come from.
2. **gsc PRs, small ones first:** G1+G2, then G3, G4, G5, G7, G6, G8, G9, G10. Each PR
   brings its own unit tests; G10 adds the notebook.
3. **shapespyer:** S3+S4, then S1+S6, then S2, then S5.
4. **shapes-app GUI:** a separate part, later.

**coords2sans afterwards:** it stays the staging ground for the gsc PRs and the parity
notebook. Once gsc has the features, archive it, with a README pointer to gsc.

## 8. Open questions

- Is the `sans_fitter` dependency acceptable for shapespyer (section 1)?
- One histogram per contrast, or three for any contrast (section 3.2)?
- Report the `geni.Iq` bug to SasView (section 4)?
- Where do the molecular volumes for G9 come from: a table per residue, a user value,
  or a calculation?
