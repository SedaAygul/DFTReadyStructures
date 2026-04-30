# Fe₃O₄@Au Core–Shell Nanoparticle Construction Pipeline

A reproducible atomistic workflow for building, relaxing, and validating
magnetite-core / gold-shell core–shell nanoparticles (CSNPs) using the
**CHGNet** universal machine-learning interatomic potential.

This repository contains the code (`pipeline_v32.py`) accompanying the
manuscript:

> **[Manuscript title placeholder — to be completed upon acceptance].**
> *Authors:* Seda Aygül Akyüz, Zeliha Cansu Canbek Özdil
> *Journal:* ACS Engineering Au, *under review.*
> *DOI:* `10.xxxx/xxxxxxx` *(to be inserted upon publication).*

---

## Table of Contents

1. [Why This Project](#why-this-project)
2. [How to Cite](#how-to-cite)
3. [Pipeline Overview](#pipeline-overview)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [What You Must Edit Before Running](#what-you-must-edit-before-running)
8. [Configuration Parameters](#configuration-parameters)
9. [Output Files Generated Per Run](#output-files-generated-per-run)
10. [Quality-Control Gates](#quality-control-gates)
11. [Cautions and Known Issues](#cautions-and-known-issues)
12. [Customisation Examples](#customisation-examples)
13. [Reproducibility](#reproducibility)
14. [Hardware Notes](#hardware-notes)
15. [Repository Structure](#repository-structure)
16. [Suggested Figures / Screenshots](#suggested-figures--screenshots)
17. [License](#license)
18. [Contact](#contact)
19. [Acknowledgements](#acknowledgements)

---

## Why This Project

Atomistic models of bimetallic core–shell nanoparticles are usually built
either by ad-hoc geometric overlap of two crystals — which produces
unphysical interfaces — or by full *ab initio* relaxation, which is
computationally prohibitive beyond a few hundred atoms.
This pipeline targets the regime in between: **realistic
nanometre-scale Fe₃O₄@Au CSNPs (10³–10⁵ atoms)** that are:

- geometrically consistent with experiment (Wulff-shaped Au shell, spherical
  spinel core),
- chemically clean at the interface (no fused or undercoordinated atoms),
- locally relaxed under a charge- and magnetic-moment-aware ML potential
  (CHGNet), and
- fully auditable through stage-by-stage XYZ snapshots and a battery of
  quantitative quality-control gates.

The output structures are suitable as starting geometries for downstream
electrodynamic (e.g. COMSOL EWFD) or thermal-transport simulations, or as
inputs for further DFT refinement on smaller motifs.

---

## How to Cite

If you use this pipeline in your research, please cite the manuscript above
**and** this repository. A BibTeX entry is provided once the article is
published; for now, please use:

```bibtex
@article{Akyuz_FeAu_CHGNet_2026,
  author  = {Akyüz, Seda Aygül and Canbek Özdil, Zeliha Cansu },
  title   = {[Automated Atomistic Construction and Interface Stabilization of Fe3O4@Au Core-Shell Nanostructures via Machine Learning]},
  journal = {ACS Engineering Au},
  year    = {2026},
  note    = {Manuscript under review. Code archive: https://github.com/<user>/<repo>}
}
```

The DOI, volume, pages and full author list will be filled in here upon
acceptance.

---

## Pipeline Overview

`pipeline_v32.py` is organised as one configuration dataclass
(`SimulationConfig`), one quality-control helper class (`QualityControl`),
and one master pipeline class (`ThesisPipelineV32`). For each
(diameter, shell-thickness) pair listed in the configuration, the pipeline
executes the following stages:

| # | Stage | What happens | Output file |
|---|---|---|---|
| A1 | **Spherical core cut** | A sphere of radius *D*/2 is carved from the bulk Fe₃O₄ supercell. | `01_core_raw.xyz` |
| A2 | **Surface cleaning** | Surface metal atoms with fewer than `min_oxygen_neighbors` (default 3) O neighbours within `fe_o_cutoff` (2.4 Å) are removed; the cleaned core is re-centred. | `02_core_cleaned.xyz` |
| B1 | **Wulff Au shell** | A gold polyhedron is generated via `ase.cluster.wulff_construction` using the (111)/(100)/(110) surface energies and Au lattice constant. | `03_shell_raw.xyz` |
| B2 | **Hole + clash filtering** | A spherical cavity (`hole_radius = core_radius + gap_size`) is carved out, then a two-tree (Au–O / Au–Fe) safety filter removes any shell atom too close to a core atom. Connectivity ≥ 95 % is enforced. | `04_shell_filtered.xyz` |
| C  | **Merge** | Core and shell are concatenated. | `05_merged_prerelax.xyz` |
| D1 | **Hybrid freeze** | Inner core atoms within `core_r_max − freeze_buffer` are frozen via `FixAtoms`; surface and Au atoms remain mobile. |  |
| D2 | **Phase I — FIRE** | 50 steps of FIRE relaxation (`fmax = 2.0`) to declash. | `06_after_fire.xyz` |
| D3 | **Phase II — Langevin anneal** | 5 ps Langevin MD at 500 K (default), `friction = 0.02`, 1 fs timestep, with Maxwell–Boltzmann initialisation. | `07_after_md.xyz` |
| D4 | **Phase IIb — Cooling** | 0.5 ps Langevin cooling at 10 K. |  |
| D5 | **Phase III — BFGS** | Final BFGS relaxation to `fmax = 0.1` eV/Å. | `08_final_relaxed.xyz` |
| E  | **QC and reporting** | Element-resolved minimum-distance gates, connectivity, mass-conservation, NaN watchdog, distance-distribution histograms, energy/force snapshot. | `quality_report.json`, `hist_*.png` |

Each (D, T) run produces its own subfolder
`<base_dir>/D<D>_T<T>/` containing the eight XYZ snapshots above plus log
files and JSON reports. A top-level `summary_log.csv` aggregates results
across all runs.

---

## Requirements

The pipeline is tested with:

- **Python** ≥ 3.10
- **PyTorch** (CPU build is sufficient; see *Hardware Notes*)
- **CHGNet** ≥ 0.3
- **ASE** ≥ 3.22 (for `ase.cluster.wulff_construction`)
- **NumPy**, **SciPy**, **pandas**, **matplotlib**

A minimal `requirements.txt` looks like:

```
chgnet>=0.3
ase>=3.22
torch
numpy
scipy
pandas
matplotlib
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/<user>/<repo>.git
cd <repo>

# 2. (Recommended) create a clean environment
conda create -n chgnet-csnp python=3.11 -y
conda activate chgnet-csnp

# 3. Install dependencies
pip install -r requirements.txt
```

The first time `CHGNet.load()` is called, the pre-trained model weights
will be downloaded automatically into the CHGNet cache directory.

---

## Quick Start

A minimal end-to-end run — single core diameter, two shell thicknesses —
using the defaults at the bottom of `pipeline_v32.py`:

```bash
python pipeline_v32.py
```

Programmatic use from your own driver script:

```python
from pipeline_v32 import SimulationConfig, ThesisPipelineV32

cfg = SimulationConfig(
    base_dir   = r"C:\path\to\your\output\folder",
    bulk_path  = r"C:\path\to\your\AB2O4_4x4x4.xyz",
    diameters  = (30.0, 35.0, 40.0),     # Å
    thicknesses= (1.5, 2.0, 2.5),         # Å
    anneal_steps = 5000,                   # 5 ps anneal
    seed = 2025,
)

ThesisPipelineV32(cfg).run()
```

---

## What You Must Edit Before Running

Two paths in `SimulationConfig` are environment-specific and **must** be
changed before the pipeline can run on your machine. They are at the very
top of the dataclass (≈ lines 62–63 of `pipeline_v32.py`):

```python
base_dir : str = r"C:\Users\<you>\...\Thesis_Dataset_Results"
bulk_path: str = r"C:\Users\<you>\...\AB2O4_4x4x4.xyz"
```

| Variable | What it is | What to put |
|---|---|---|
| `base_dir`  | Root folder where all per-run outputs are written. Created automatically if missing. | An absolute path to an empty folder you own. |
| `bulk_path` | An ASE-readable XYZ file containing a centred Fe₃O₄ (or any AB₂O₄ spinel) supercell large enough to contain the requested sphere. The 4×4×4 supercell used in the manuscript is the recommended starting point. | Absolute path to that file. |

> **Note.** The script forces CPU execution at line 12
> (`os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`) as a temporary workaround
> for the RTX 5070 / current-PyTorch incompatibility. If you have a
> CUDA-supported GPU and a PyTorch build that recognises it, comment out
> that line and the matching block in `_load_resources` to enable GPU
> acceleration.

---

## Configuration Parameters

All adjustable parameters live in `SimulationConfig` (≈ lines 56–131).
Below they are grouped by intent.

### Geometry — what gets built

| Parameter | Default | Meaning |
|---|---|---|
| `diameters`    | `(35,)` | Tuple of core diameters in Å. The pipeline iterates over every diameter. |
| `thicknesses`  | `(2.0, 2.5)` | Tuple of nominal shell thicknesses in Å. |
| `gap_size`     | `2.3` Å | Initial Au–core radial gap. Increase to reduce interface clashing. |
| `wulff_buffer` | `3.0` Å | Extra radius added when generating the Wulff polyhedron, so that even after the spherical hole is carved a continuous shell remains. |

### Wulff construction (Au)

| Parameter | Default | Meaning |
|---|---|---|
| `surf_energies` | `(1.0, 1.14, 1.25)` | Surface energies for {(111), (100), (110)} in J m⁻². Defaults reflect literature values for Au; edit if you rebuild on another metal. |
| `lattice_au`    | `4.08` Å | Au FCC lattice constant. |

### Surface cleaning of the magnetite core

| Parameter | Default | Meaning |
|---|---|---|
| `min_oxygen_neighbors` | `3`    | Surface Fe (or Rh) atoms with fewer than this many O neighbours are deleted. |
| `fe_o_cutoff`          | `2.4` Å | Fe–O nearest-neighbour cutoff used in the cleaning step. |
| `max_removal_fraction` | `0.20` | If more than 20 % of surface metals would be removed, a warning is printed (the run still proceeds). |

### Interface safety (pre-relax filtering)

| Parameter | Default | Meaning |
|---|---|---|
| `safe_dist_au_o`  | `1.9` Å | Any Au atom closer than this to a core O is removed before relaxation. |
| `safe_dist_au_fe` | `2.3` Å | Any Au atom closer than this to a core Fe/Rh is removed before relaxation. |

### Quality-control gates (post-relax)

| Parameter | Default | Meaning |
|---|---|---|
| `qc_min_dist_global`     | `1.2` Å | Hard global minimum-distance fuse. Below this, a run is failed. |
| `qc_cutoffs`             | `{Au-Au: 2.5, Metal-O: 1.6, O-O: 2.0, Au-Metal: 2.0, Au-O: 1.8}` | Element-pair minimum distances (Å). Tunable per system. |
| `min_shell_connectivity` | `0.95` | Largest Au connected component must contain ≥ 95 % of all Au atoms. |
| `connectivity_cutoff`    | `3.2` Å | Au–Au bond cutoff used to build the connectivity graph. |

### Relaxation physics

| Parameter | Default | Meaning |
|---|---|---|
| `anneal_temp`    | `500.0` K | Langevin temperature during Phase II. |
| `anneal_steps`   | `5000`    | Number of MD steps (×`timestep_fs` fs) — default is 5 ps. |
| `timestep_fs`    | `1.0` fs | MD timestep. |
| `freeze_buffer`  | `5.0` Å  | Width of the *unfrozen* outer shell of the core (atoms within `core_r_max − freeze_buffer` are frozen). |

### Safety / memory

| Parameter | Default | Meaning |
|---|---|---|
| `max_shell_atoms_est`  | `300 000` | Hard ceiling on the size of the Wulff polyhedron requested from ASE. |
| `max_shell_atoms_real` | `200 000` | Hard ceiling on the *final* shell size after carving and clash filtering. |

### Reproducibility

| Parameter | Default | Meaning |
|---|---|---|
| `seed` | `2025` | Master seed for NumPy and PyTorch. |

---

## Output Files Generated Per Run

For each `(D, T)` pair the pipeline creates `<base_dir>/D<D>_T<T>/`
containing:

```
01_core_raw.xyz           # spherical cut, before cleaning
02_core_cleaned.xyz       # after surface-O-neighbour filter
03_shell_raw.xyz          # full Wulff Au cluster (before hole)
04_shell_filtered.xyz     # after hole + Au-O / Au-Fe safety filter
05_merged_prerelax.xyz    # core + shell, pre-relaxation
06_after_fire.xyz         # after Phase I (FIRE)
07_after_md.xyz           # after Phase II (Langevin anneal)
08_final_relaxed.xyz      # final structure after BFGS — the deliverable
fire.log                  # FIRE optimiser log
bfgs.log                  # BFGS optimiser log
quality_report.json       # all QC metrics + final energy / fmax
run_metadata.json         # surrogate-substitution audit (Rh→Fe etc.)
hist_Au-Au_nn.png         # Au–Au nearest-neighbour distance histogram
hist_Au-O_nn.png          # Au–O nearest-neighbour distance histogram
hist_Au-Metal_nn.png      # Au–Fe/Rh nearest-neighbour distance histogram
failure_report.json       # only created if the run fails (with traceback)
```

At the root level:

```
global_config.json        # full SimulationConfig + library versions
summary_log.csv           # one row per (D, T) run with status + QC stats
```

---

## Quality-Control Gates

A run is reported as `Success` only if **every** check below passes
(`QualityControl.validate_run`):

1. `min_dist_global ≥ qc_min_dist_global`
2. `Au_connectivity ≥ min_shell_connectivity`
3. `min_dist_Au-Au   ≥ qc_cutoffs["Au-Au"]`
4. `min_dist_Metal-O ≥ qc_cutoffs["Metal-O"]`
5. `min_dist_O-O     ≥ qc_cutoffs["O-O"]`
6. `min_dist_Au-Metal≥ qc_cutoffs["Au-Metal"]`
7. `min_dist_Au-O    ≥ qc_cutoffs["Au-O"]`

Additionally:

- `_assert_finite()` is called after every stage to catch NaN / Inf
  coordinates.
- `_assert_atom_count()` is called after every relaxation phase to catch
  silent atom loss.
- A pre-MD O–O distance check fails fast before any expensive integration
  begins.

---

## Cautions and Known Issues

- **Rh → Fe surrogate.** CHGNet's training set does not include Rh, so any
  Rh atoms in the core are *temporarily* renamed to Fe before the calculator
  is attached, and renamed back afterwards. This is logged in
  `run_metadata.json` (`surrogate_active_during_relax`). Treat any Rh
  energetics from this pipeline as qualitative.
- **CHGNet training-set caveats.** The Materials Project corpus that
  underlies CHGNet does **not** contain ternary Au–Fe–O configurations.
  Local relaxations are reliable; quantitative interface adhesion energies
  are not.
- **GPU disabled by default.** Line 12 (`CUDA_VISIBLE_DEVICES = -1`) and the
  CPU override in `_load_resources` exist because the development machine
  uses an RTX 5070, which is not yet supported by stock PyTorch. Remove
  both on a supported GPU.
- **Hard-coded Windows paths.** The defaults under `SimulationConfig`
  point at the developer's local machine and contain a non-ASCII path
  (`Masaüstü`). Always pass your own `base_dir` and `bulk_path` when
  constructing the config.
- **Atom-count ceilings.** `max_shell_atoms_est` and `max_shell_atoms_real`
  are emergency brakes; if you need shells larger than ~200 k atoms, raise
  them deliberately and ensure you have the RAM.
- **`wulff_construction` import location.** ASE moved this function in
  v3.22; the script tries both import paths but very old ASE versions will
  still fail with a clear message.
- **Anneal length is a research parameter.** The default 5 ps is a
  compromise between throughput and equilibration; production-quality runs
  in the manuscript use the same value, but other systems may need longer.
- **Connectivity threshold is strict.** A 95 % Au connectivity requirement
  fails runs that produce a small detached island. Lower
  `min_shell_connectivity` only if you understand why.

---

## Customisation Examples

### Sweep larger CSNPs

```python
cfg = SimulationConfig(
    base_dir = r"D:\runs\large_sweep",
    bulk_path = r"D:\inputs\AB2O4_6x6x6.xyz",   # bigger supercell
    diameters = (40.0, 50.0, 60.0),
    thicknesses = (2.0, 3.0, 4.0),
    max_shell_atoms_real = 350_000,             # raise ceiling
)
ThesisPipelineV32(cfg).run()
```

### Use a different B-site cation (e.g. Mn, Co)

The construction logic itself is element-agnostic on the core side as long
as the input bulk file contains the desired B-site cation. If you swap to
a CHGNet-supported cation, you can disable the Rh surrogate logic by
making sure no Rh symbols are present in the bulk; nothing else needs to
change. For elements outside CHGNet's training set, expect the same
caveats described above.

### Tighter / looser interface

```python
cfg = SimulationConfig(
    gap_size = 3.0,            # start the Au shell further out
    safe_dist_au_o = 2.1,      # remove more Au at the interface
    safe_dist_au_fe = 2.5,
)
```

### Longer annealing

```python
cfg = SimulationConfig(
    anneal_temp = 600.0,
    anneal_steps = 20_000,     # 20 ps
)
```

---

## Reproducibility

- All inputs (every value of `SimulationConfig`) are serialised to
  `global_config.json` together with the exact `ase`, `torch`, `chgnet`
  and `python` versions used at runtime.
- A single master `seed` (default `2025`) controls NumPy and PyTorch.
- Eight XYZ snapshots per run document every state transition; nothing in
  the pipeline silently overwrites or discards intermediate geometries.
- A `failure_report.json` with the full Python traceback is written for
  every failed run, indexed by the stage name.

These artifacts collectively allow exact byte-for-byte reproduction of any
result reported in the manuscript when the same library versions are
installed.

---

## Hardware Notes

The reference development machine is a Lenovo LOQ 17IRX10 laptop with an
NVIDIA RTX 5070 GPU. As of this release, the stock PyTorch CUDA build
does not yet recognise the RTX 50-series; the script therefore forces CPU
execution. On CPU, a representative run with `D = 35 Å`, `T = 2 Å`,
5 ps anneal and ~10⁴ atoms takes on the order of one to two hours.

On supported NVIDIA GPUs (RTX 30/40 series, A100, H100, etc.) you can:

1. Comment out line 12 (`os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`).
2. In `_load_resources`, replace the forced `self.device = 'cpu'` with the
   commented-out auto-detection block immediately below it.

---

## Repository Structure

A typical layout for the public release:

```
.
├── pipeline_v32.py           # the pipeline (this file)
├── inputs/
│   └── AB2O4_4x4x4.xyz       # example bulk supercell (provide your own)
├── examples/
│   └── run_minimal.py        # minimal driver script
├── docs/
│   ├── workflow.png          # pipeline schematic (suggested)
│   └── csnp_render.png       # rendered final structure (suggested)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Figures / Screenshots

To make the README and the GitHub landing page more readable, the
following images would help. None of these are auto-generated; you would
need to provide them.

1. **Workflow schematic**
   
<img width="4400" height="2475" alt="Picture1" src="https://github.com/user-attachments/assets/5edff7aa-0577-4393-bdf7-5ea1aa7effa8" />
2. **Evolution of core stability with increasing diameter. (a) D=5.0 Å (b) D=6.0 Å (c) D=7.0 Å **

   <img width="1248" height="415" alt="Picture2" src="https://github.com/user-attachments/assets/a2c5b19e-a593-418a-b178-c7eb2984871e" />

3. Geometric percolation threshold of the Au shell for D=35 Å. (a) T = 0.05 Å, connectivity ≈6.4% (b) T = 0.5 Å, connectivity ≈6.3% (c) T = 0.8 Å, connectivity= 100% (d) T = 1.0 Å, connectivity=100%

<img width="1316" height="352" alt="Picture4" src="https://github.com/user-attachments/assets/0d5b7596-a721-420f-b247-6e2c57083c69" />

4.Energy evaluations throughout the relaxations

<img width="768" height="405" alt="Picture5" src="https://github.com/user-attachments/assets/cb3d0f01-efd2-49a3-a05b-b311d2c9caa9" />

5.Nearest Neighbors distances for a) Au-O b) Au-Fe

<img width="1333" height="496" alt="Picture7" src="https://github.com/user-attachments/assets/771c1549-f44c-45ef-92d7-ea99ab17940f" />

---

## Contact

- **Seda Aygül Akyüz** — *corresponding author for code questions*
  ✉ `sedaaygul99@gmail.com`

For bug reports and feature requests, please open a GitHub *Issue* rather
than emailing , it keeps the discussion searchable for other users.

---

## Acknowledgements

This work uses:

- **CHGNet** — Deng, B. *et al.* *Nat. Mach. Intell.* **5**, 1031–1041
  (2023).
- **ASE** — Larsen, A. H. *et al.* *J. Phys.: Condens. Matter* **29**,
  273002 (2017).
- **The Materials Project** — Jain, A. *et al.* *APL Materials* **1**,
  011002 (2013).

Computational resources and supervision are gratefully acknowledged in the
manuscript itself.
