# Integration plan: coords2sans into gsc and shapespyer

Status: 2026-10-29. Draft for review.

- gsc: `GenericScatteringCalculator`, today in SANS-fitter branch `gsc`,
  `src/sans_fitter/structure/`, commit `2e9674e`.
- coords2sans: this repo, not committed yet.
- SasView: commit `02363ba`, the one `examples/compare.ipynb` downloads.

## 1. Decision

gsc becomes a separate library, next to sans_fitter: sans_fitter fits measured curves,
gsc computes curves from structures. The coords2sans features move into gsc.
Multiprocessing and the project glue go into a wrapper in shapespyer.

- **Why gsc:** its physics is exact, and it already has provenance (`get_settings()`),
  progress and cancel, and `to_data1d()`, which hands a curve to a sans_fitter fit.
- **Dependencies:** the calculation needs only numpy and periodictable, which
  shapespyer already has. plotly (the plots) and sasdata (`to_data1d`) can be optional
  extras. Today the gsc code still uses sans_fitter in three places, which the split
  must cut:
  - `persistence._package_version` (it imports sasmodels);
  - `data.loader.validate_q_grid` and `normalize_sans_data` (they import sasdata);
  - `plotting.error_bars` and `resolve_show` (sans_fitter's plotting module).

## 2. What coords2sans and gsc have now

| | coords2sans | gsc | How they compare |
|---|---|---|---|
| Exact average over the directions (Debye sum) | yes, the default: the pairs are first counted by distance (`_pair_histogram`) | yes: every pair at every q (`debye_pairs`) | the same curve to 1e-4; coords2sans is 20× faster at 40 q values, about 130× at 256 (section 3) |
| Averaged amplitude ⟨A⟩² (SasView `is_avg`) | yes, opt-in (`average_amplitude = True`) | no | the fastest, but right only for round objects (section 3) |
| Lattice kernel (FFT) | no | yes, for pixel files (`.sld`) | gsc only; not used for atoms |
| Size polydispersity | yes (the sasmodels rule) | no | coords2sans only |
| Atom volumes | from the elements' bulk densities | the same | the same flaw; gsc warns about it (section 8) |

## 3. Which kernel to use

The micelles of a sample point in all directions, so the curve is an average over the
directions. ⟨A⟩² is exact only for **round** (spherically symmetric) objects, such as
spherical micelles and spherical vesicles. Many micelles are not round. The CTAB
micelle in `examples/data` is a short rod (99 × 64 × 56 Å, 11,000 atoms). For it ⟨A⟩²
is 5 % too low at q = 0.06 Å⁻¹, 35 % at 0.095 and 99 % at 0.144.

| | Histogram (exact) | ⟨A⟩² | gsc `debye_pairs` (exact) |
|---|---|---|---|
| Right for | any shape | round objects only | any shape |
| Error | 1e-4 | none for a sphere; 35 % at q = 0.095 for the CTAB rod | none |
| 11,000 atoms, one size | 2.3 s | 0.07 s | 48 s at 40 q; about 5 min at 256 q |
| 11,000 atoms, 35 sizes | 4.6 s | 1.9 s | about 28 min at 40 q |
| 130,000 atoms, one size | about 6 min | about 1 s | about 2 h at 40 q |
| Time grows with | atoms² once, then bins × q | atoms × q | atoms² × q |

Times at 256 q values unless noted; the 130,000-atom times are scaled from 11,000.

- **⟨A⟩² is the fastest, but right only for round objects.** For a nearly round MD
  snapshot the error is small at low q and grows with q; not measured for a vesicle.
- **The histogram is exact for any shape.** Against gsc's current exact kernel it wins
  on every point except the last digits: the same curve to 1e-4, the same growth with
  the atoms, 20–130× faster, and bounded memory. It is slower than ⟨A⟩²: 30× for one
  size, 2.4× for a spread of 35 sizes.
- Above about 100,000 atoms, no kernel is both fast and exact (section 8.2).

### 3.1 How the histogram kernel works

1. One pass over all atom pairs, in blocks of about a million pairs (a few tens of MB
   of memory). The distances come from |a|² + |b|² − 2a·b, a matrix product. Each pair
   adds w_j·w_k to the bin at its distance, rounded to `BIN_WIDTH` = 0.01 Å, with
   w = (SLD − solvent SLD)·V.
2. For each q: Σ bin · sinc(q r) over the bins. For the CTAB micelle that is about
   13,000 bins instead of 60 million pairs. gsc's own `_sinc_sum` already does this
   step.
3. A size f stretches every distance by f and every amplitude by f³, so it gives
   f⁶·I₁(f q). All sizes of a spread reuse the same bins.

- **Accuracy:** the rounding moves each distance by at most 0.005 Å. Up to
  q = 0.2 Å⁻¹ the curve changes by about 1e-4 (checked against SasView's plain pair sum,
  and in the unit tests). Above q = 0.2 it is not tested; `BIN_WIDTH` may need to be
  smaller there.
- **Progress and cancel:** the pass runs in blocks, so gsc's progress callback fits
  after each block, as in `debye_pairs`.
- **Size:** about 25 lines of numpy.

### 3.2 During the merge

1. Add the histogram kernel to gsc, next to `debye_pairs` (G6). Which kernel is the
   default for atoms: to agree (section 10).
2. Add ⟨A⟩² as an opt-in, labelled "round objects only" (G7).
3. If the histogram becomes the default, raise gsc's run-time warning (today above
   20,000 atoms, `LARGE_PAIR_LOOP`) to about 100,000 atoms.
4. Keep the lattice kernel for pixel files as it is.

## 4. SasView findings, and why they matter

| Finding | Why it matters for the merge |
|---|---|
| `GenSAS` with `is_avg = False` uses the AUSAXS library (`pyausaxs`) when it is installed. AUSAXS also counts pairs by distance. Without it, SasView falls back to plain numpy, every pair at every q (`sasview_sans_debye.py`, 45 s per curve at 32 q for the CTAB micelle) | Counting pairs by distance is an established method, not our invention. gsc could call `pyausaxs` instead of carrying its own kernel (G6), but that adds a compiled dependency, and its speed is not measured here. Our kernel is 25 lines of numpy and is measured. The validation (section 7) uses the slow fallback as its reference, so it runs at few q values |
| `geni.Iq` fails when an atom's contrast is exactly 0: line 43, `coords, sld, vol = (v[index] for v in (sld, coords, vol))`, swaps the names | The validation (section 7) must not use atoms that match the solvent. For that case gsc is the reference: it already leaves zero-contrast atoms out and records them. Reporting the bug to SasView is outside the merge |
| `PDBReader` reads only `ATOM` records. An unknown element gets SLD 0 and no volume | The validation files must use `ATOM` records and known elements. gsc keeps its own PDB reader: it reads `HETATM` on request and refuses unknown elements |

## 5. Into gsc

These features are generic: useful for anyone with MD or PDB files. The numbers give
the order of the PRs. Each source names the coords2sans code to port.

| # | Feature | Change in gsc |
|---|---|---|
| G1 | `.gro` reader, and residue data on `Structure` | add |
| G2 | read only the aggregate's residues | extend the readers |
| G3 | periodic unwrapping | add |
| G4 | elements from the topology | extend the readers |
| G5 | per-residue deuteration | add |
| G6 | histogram kernel | add, next to `debye_pairs` |
| G7 | ⟨A⟩² kernel | add |
| G8 | size polydispersity | add |

### G1: `.gro` reader, and residue data on `Structure`

- **What for:** GROMACS writes its frames as `.gro` files. G2–G5 and the later volume
  fix (section 8.1) need to know which residue each atom belongs to.
- **gsc now:** readers for PDB, `.sld` and `.vtk`, chosen by `load()` from the file
  extension. `read_pdb` reads the residue name (columns 18–20, three letters) only to
  drop water, then forgets it. `Structure` has `symbols` and `bonds`, but no residue
  data.
- **Change (add):** `read_gro`, chosen by `load()` for `.gro` files. New optional
  `Structure` fields `resids`, `resnames` and `names` (default None), filled by
  `read_gro` and `read_pdb`. The PDB residue name takes columns 18–21, where GROMACS,
  CHARMM and shapespyer write four-letter names such as `POPC`. Nothing that exists
  changes.
- **Source:** `Structure.read`, `Structure._from_gro`, `Structure._from_pdb`.

### G2: read only the aggregate's residues

- **What for:** an MD frame holds the aggregate and its solvent: water, ions, maybe
  other molecules. The calculation needs only the aggregate's atoms; the solvent
  enters as one number, its SLD. Water and ions left in the structure are computed as
  part of the aggregate, so the curve is no longer the aggregate's, and the pair pass
  gets much bigger (100,000 water atoms next to an 11,000-atom micelle).
- **gsc now:** `read_pdb` drops the atoms whose residue name is `HOH`, `DOD` or `WAT`
  (the water names of protein PDB files), and drops `HETATM` records unless
  `include_hetatm=True`. MD programs name water differently: GROMACS `SOL`, CHARMM
  `TIP3`. So in a GROMACS or CHARMM file the water is read into the structure, and so
  are the ions written as `ATOM` records.
- **Change (extend):** a `residues=` argument for `load()`, `read_pdb`, `read_gro` and
  `GenericScatteringCalculator.load_structure`: a list of residue names, such as
  `['CTA']`. Only the atoms of these residues are read into the `Structure`. Without
  it, the readers do what they do now.
- **Source:** `Structure.read(residues=)`.

### G3: periodic unwrapping

- **What for:** MD keeps every atom inside a periodic box. A molecule that crosses a
  box face is written partly on the far side, so the aggregate is split. The curve is
  then that of the split pieces: in the coords2sans README example, the radius of
  gyration is 6.1 nm instead of 2.7 nm.
- **gsc now:** none; no reader knows the box.
- **Change (add):** a reader step after G2. Each molecule is made whole, then moved by
  whole box lengths next to the aggregate's centre. The box comes from the last line
  of a `.gro` file; only rectangular boxes. A PDB file carries the box in its `CRYST1`
  record, but many programs write a placeholder box of 1 Å there, so coords2sans does
  not unwrap PDB files. PDB: to decide (section 10).
- **Source:** `Structure._fix_periodic_wrapping`.

### G4: elements from the topology

- **What for:** each atom's element sets its SLD and its volume. MD files often do not
  name it: a `.gro` file has no element column, and the PDB element column (77–78) is
  often empty. Then the element is guessed from the atom name, and MD atom names break
  the guess: ` NA ` (a sodium ion) reads as nitrogen; `C21` (carbon 21 of a lipid
  tail) as `C2`, two carbons, with double volume; `CA` may be an alpha carbon or
  calcium. The topology (`.itp`) lists the mass of each atom in order, and a mass names
  its element without doubt.
- **gsc now:** `read_pdb` takes columns 77–78, else SasView's guess from the atom name
  (`_element_from_name`). It does not record that it guessed, and does not warn. It
  refuses elements it does not know.
- **Change (extend):** `masses=` and `elements=` arguments for the readers: by residue
  name, the masses or the element symbols of its atoms, in order. Each residue takes
  its elements from the first source that has them: masses, element lists, the PDB
  column, the guess. A mass picks the element of nearest mass, within 0.01 u. A new
  `Structure.element_sources` records which source each residue used, and a guess adds
  a note to `Structure.notes`. Needs G1.
- **Source:** `Structure.read(masses=, elements=)`, `Structure._assign_elements`,
  `Structure.element_sources`.

### G5: per-residue deuteration

- **What for:** contrast variation. Deuterating one component (the surfactant, or one
  lipid of a mixture) changes its SLD a lot and leaves the others as they are. A
  series of samples then shows the parts of an aggregate one by one. The structure is
  the same; only the SLD of some hydrogens changes.
- **gsc now:** the SLD comes from the element symbol when the file is read
  (`_element_constants`). A hydrogen is D only if the file says `D`.
- **Change (add):** `Structure.deuterated(['CTA'])` returns a new `Structure` in which
  the hydrogens of these residues have the SLD of D; their volume stays that of H. It
  fits gsc's frozen `Structure`. An unknown residue name is an error. Needs G1.
- **Source:** `Structure.sld(deuterated)`.

### G6: histogram kernel

- **What it is:** a new way to compute the exact Debye sum (section 3.1). It was
  written with AI help in coords2sans, and checked against SasView's pair sum and by
  unit tests, but no person has reviewed it yet.
- **gsc now:** `debye_pairs` (every pair at every q) for atoms, `debye_lattice` for
  pixels; `_debye` picks one by itself.
- **Change (add):** a third kernel next to `debye_pairs`, chosen with a `kernel=`
  setting (the name to agree). It reuses gsc's `_sinc_sum`. `ScatteringProfile.method`
  gets `'histogram'`. Which kernel is the default: to agree (section 10).
- **Pros:** exact for any shape, to 1e-4. 20× faster than `debye_pairs` at 40 q
  values, about 130× at 256. A spread of sizes costs little more than one size. numpy
  only, about 25 lines. Progress and cancel after each block.
- **Cons:** new, not reviewed. The distances are rounded to 0.01 Å, tested only up to
  q = 0.2 Å⁻¹. The time still grows as atoms²: about 6 min for 130,000 atoms. 30×
  slower than ⟨A⟩² for one size.
- **When to choose it:** an aggregate that is not round (rods, worms, discs, bilayer
  patches) of up to about 100,000 atoms, and any spread of sizes. `debye_pairs` stays
  the choice when the last digits matter, for example in tests.
- **Source:** `SansCalculator._pair_histogram`, then `SansCalculator._sinc_sum`.

### G7: ⟨A⟩² kernel

- **What it is:** the averaged amplitude: the amplitude is averaged over the
  directions, then squared. This is SasView's `is_avg = True`, and what the polyJD
  script uses.
- **gsc now:** none.
- **Change (add):** one more kernel, for example `kernel='average'`, with the centre at
  the middle of the bounding box, as SasView takes it.
- **Pros:** the fastest: the time grows with the atoms, not with their pairs. 0.07 s
  for 11,000 atoms, about 1 s for 130,000. The same curves as SasView's
  `is_avg = True` and polyJD (to 1e-13), so older results can be reproduced.
- **Cons:** right only for round objects. For the CTAB rod it is 5 % too low at
  q = 0.06 Å⁻¹, 35 % at 0.095 and 99 % at 0.144, with a deep minimum that the micelle
  does not have. The result depends on the chosen centre. Nothing in the result shows
  when it is wrong.
- **When to choose it:** round aggregates (spherical micelles, spherical vesicles),
  above all large ones of over 100,000 atoms; a quick first look; reproducing polyJD or
  SasView `is_avg` curves.
- **Source:** `SansCalculator.calculate` with `average_amplitude = True`,
  `Structure.centre`, `SansCalculator._sinc_sum`.

### G8: size polydispersity

- **What for:** the micelles of a real sample differ in size, but one MD aggregate has
  one size. A spread of sizes smooths the minima of the curve and raises it at low q.
- **gsc now:** none. `set_params` has `scale`, `background`, `sld_solvent` and
  `normalisation_volume`.
- **Change (add):** new `set_params` keys, for example `polydispersity=` (the relative
  standard deviation of the size) and `polydispersity_n=` (the number of sizes,
  default 35). The sizes come from a Gaussian over ±3σ and are added up by the
  sasmodels rule: a size f gives f⁶·I₁(f q). Fast only with G6 or G7: with
  `debye_pairs`, 35 sizes take about 28 min. To agree: with a spread, `V_norm` becomes
  the mean volume of the sizes.
- **Source:** `SansCalculator._sizes`, and the sum over the sizes in
  `SansCalculator.calculate`.

### Not ported to gsc

- `Curve`: gsc's `ScatteringProfile` already covers it (`save_csv`, `to_data1d`,
  `plot`).
- The solvent SLD in Å⁻². gsc uses 10⁻⁶ Å⁻² (the sasmodels unit) and warns about values
  that look like Å⁻².
- The normalisation by the volume of all atoms. gsc normalises by the volume of the
  atoms that scatter ("active volume"), which is fine.
- `SansTask`: it goes into shapespyer (S1).

## 6. Into the shapespyer wrapper (`shapes.project.analysis`)

shapespyer then depends on the gsc library: numpy and periodictable, which it has.

| # | Feature | Why here |
|---|---|---|
| S1 | a picklable task (file paths and settings, no arrays) that builds a gsc calculator inside the child process. It runs through Job or `ProcessPoolExecutor`, over frames × contrasts × stages. coords2sans `SansTask` is the model | process handling belongs to the app, not the library |
| S2 | progress and cancel: gsc `calculate(progress=…)` reports to Job progress; returning True raises `CalculationInterrupted`, which cancels the Job | gsc already has the hook; it matters once the Job hang bug is fixed |
| S3 | masses from shapespyer's ITP reader (`Species.masses`), passed to gsc as `masses=` | gsc stays free of topology readers |
| S4 | the aggregate's residues and the deuteration choices, from the project's species | shapespyer knows the composition |
| S5 | the curves stored in the project, with `get_settings()` as their provenance | project format |
| S6 | the kernel by shape and atom count (histogram up to about 100,000 atoms; ⟨A⟩² for round aggregates), and the SLD unit conversion at the boundary (Å⁻² in shapespyer, 10⁻⁶ Å⁻² in gsc) | app policy |

## 7. Validation after the merge

The notebooks in coords2sans were made to check that its new code works, and to learn
how coords2sans and SasView compute. After the merge they are run again: once, by
hand, to see that gsc gives the same curves, and as example runs to learn the new gsc
code. They are not an automated test suite; gsc's unit tests come with each PR.

1. Rewrite `examples/compare.ipynb` and `examples/usage.ipynb` for gsc, and the
   parallel part of `usage.ipynb` for the shapespyer task (S1).
2. Run them on the CTAB micelle in `examples/data`.
3. Compare with what coords2sans gives now:

| Check | Reference | coords2sans now |
|---|---|---|
| the histogram kernel (G6) | SasView's plain pair sum (`sasview_sans_debye`) | agrees to 1e-4 |
| the ⟨A⟩² kernel (G7) | SasView `GenSAS` with `is_avg = True` | agrees to 1e-13 |
| a spread of sizes (G8) on a uniform sphere | the sasmodels `sphere` model | agrees to 4e-6 |
| the polyJD script | its own loops on SasView | shows how its recipe differs (README section 10) |
| exact average against ⟨A⟩² on the CTAB rod | none | the ratio table of section 3 |

The limits of section 4 apply: few q values for the SasView reference (45 s per curve
at 32 q), no atoms that match the solvent, and PDB files with `ATOM` records and known
elements only.

## 8. Not solved by the merge

### 8.1 The atom volumes are too big

One CTA gets 1194 Å³ from the elements' bulk densities. Most of it is hydrogen:
42 × 23.6 Å³ = 993 Å³. A realistic CTA volume is about 560 Å³ (an estimate). So I(0) is
about 2× too high in D2O and 10× too high in H2O, and the match point moves.

coords2sans, gsc, SasView and polyJD all have this problem. gsc warns about it;
coords2sans states it in its README.

**Later fix:** `molecular_volumes={'CTA': 560}` scales the atom volumes inside each
residue and keeps the scattering lengths. It needs the residue data of G1. **To
decide:** where the volumes come from: a table per residue, a user value, or a
calculation.

### 8.2 Large structures that are not round

Above about 100,000 atoms the histogram gets slow: about 6 min for 130,000 atoms,
about 50 min for a 400,000-atom vesicle. ⟨A⟩² stays fast but is right only for round
objects. **Option:** put the atoms on a grid and use gsc's lattice kernel. Not tested.

### 8.3 No hydration shell, no H/D exchange

Neither tool models the water bound to the head groups, or labile hydrogens that swap
with the solvent's. gsc's warning says so.

## 9. Order

1. **Split gsc out of sans_fitter** into its own library (section 1).
2. **Agree on the scope with the gsc maintainer** (Piotr Rozyczko, author of the gsc
   commit): residue data on `Structure`; the new `set_params` keys; the kernels and
   their default; the questions of section 10.
3. **gsc PRs, in the order of their numbers:** G1+G2, then G3 to G8. Each PR brings its
   own unit tests.
4. **Validation** (section 7).
5. **shapespyer:** S3+S4, then S1+S6, then S2, then S5.
6. **shapes-app GUI:** a separate part, later.

**coords2sans afterwards:** it stays the staging ground for the gsc PRs and the
validation notebooks. Once the validation matches, archive it, with a README pointer to
gsc.

## 10. Open questions

- Which kernel is the default for atoms: `debye_pairs`, the histogram, or one chosen
  by atom count (G6)?
- One histogram per contrast, or three (of b·b, b·v and v·v, with w = b − ρ_solvent·v)?
  Three make every further solvent SLD almost free; a prototype took 7 s once. A change
  of deuteration needs new histograms either way.
- G3: unwrap PDB files from `CRYST1` too?
- Report the `geni.Iq` bug to SasView?
