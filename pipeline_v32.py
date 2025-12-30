# =============================================================================
# MODULE: THESIS PIPELINE V32 (LOCAL VERSION - GOLD STANDARD)
# =============================================================================
# Adapted for Local Environment: Lenovo LOQ 17IRX10 (RTX 5070)
# Path: C:\Users\Seda\Documents\Masaüstü\Atomic_Design\LATEST_DESIGN
# =============================================================================
import os

# --- CRITICAL FIX FOR RTX 5070 ---
# This hides the GPU from Python entirely, forcing CPU mode.
# It must run BEFORE 'import torch'.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import json
import matplotlib.pyplot as plt
import time
import warnings
import traceback
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, asdict
from typing import Sequence, Tuple, Dict
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# ASE Imports
import ase
from ase import Atoms, units
from ase.io import read, write
from ase.neighborlist import NeighborList
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

# CHGNet Check
try:
    import chgnet
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
except ImportError:
    raise ImportError("CHGNet is not installed. Please pip install chgnet.")

# Wulff Check
try:
    from ase.cluster.wulff import wulff_construction
except ImportError:
    try:
        from ase.cluster import wulff_construction
    except ImportError:
        raise ImportError("Critical: Could not import wulff_construction from ase.cluster.")


# --- 1. CONFIGURATION ---
@dataclass
class SimulationConfig:
    """Central Configuration (V32 - Local)."""
    # File Paths (Windows Local)
    # Using raw strings (r"...") to handle backslashes correctly
    base_dir: str = r"C:\Users\Seda\Documents\Masaüstü\Atomic_Design\LATEST_DESIGN\Thesis_Dataset_Results"
    bulk_path: str = r"C:\Users\Seda\Documents\Masaüstü\Atomic_Design\LATEST_DESIGN\AB2O4_4x4x4.xyz"

    # Dimensions
    diameters: Sequence[float] = (35,)
    thicknesses: Sequence[float] = (2.0, 2.5)

    # Generation Logic
    gap_size: float = 2.3
    wulff_buffer: float = 3.0
    # Emergency Brakes
    max_shell_atoms_est: int = 300000
    max_shell_atoms_real: int = 200000

    # Surface Cleaning (Core)
    min_oxygen_neighbors: int = 3
    fe_o_cutoff: float = 2.4
    max_removal_fraction: float = 0.20

    # Interface Safety (Pre-Relax Filtering)
    safe_dist_au_o: float = 1.9
    safe_dist_au_fe: float = 2.3

    # QC: Gates (Post-Relax)
    qc_min_dist_global: float = 1.2
    qc_cutoffs: Dict[str, float] = None

    # QC: Connectivity
    min_shell_connectivity: float = 0.95
    connectivity_cutoff: float = 3.2

    # Relaxation / Physics
    anneal_temp: float = 500.0
    anneal_steps: int = 5000  # 5ps
    timestep_fs: float = 1.0
    freeze_buffer: float = 5.0

    # Wulff Parameters (Au)
    surf_energies: Sequence[float] = (1.0, 1.14, 1.25)
    lattice_au: float = 4.08

    # Reproducibility
    seed: int = 2025

    def __post_init__(self):
        if self.qc_cutoffs is None:
            self.qc_cutoffs = {
                "Au-Au": 2.5,
                "Metal-O": 1.6,
                "O-O": 2.0,
                "Au-Metal": 2.0,
                "Au-O": 1.8
            }

    def to_json(self):
        data = asdict(self)
        data['_versions'] = {
            'ase': ase.__version__,
            'torch': torch.__version__,
            'chgnet': chgnet.__version__,
            'python': sys.version
        }
        data['_methodology'] = {
            'alignment': 'Always-Centered (Origin)',
            'sampling': 'Emergency Brake (No Random Deletion)',
            'connectivity': 'Symmetrized Graph Validation (>95%)',
            'equilibration': 'Maxwell-Boltzmann + 5ps Anneal-Assisted Relaxation',
            'integrity': 'Mass Conservation Checks + Stage Tracking'
        }
        return json.dumps(data, indent=4)


# --- 2. QUALITY CONTROL HELPER ---
class QualityControl:
    @staticmethod
    def analyze_structure(atoms, connectivity_cutoff=3.2, name="structure"):
        stats = {"name": name, "n_atoms": len(atoms)}

        syms = atoms.get_chemical_symbols()
        stats["n_Au"] = syms.count("Au")
        stats["n_Fe"] = syms.count("Fe")
        stats["n_Rh"] = syms.count("Rh")
        stats["n_O"] = syms.count("O")

        n_metal_core = stats["n_Fe"] + stats["n_Rh"]
        stats["ratio_Metal_O"] = round(n_metal_core / stats["n_O"], 3) if stats["n_O"] > 0 else 0

        pos = atoms.get_positions()

        idx_au = [i for i, s in enumerate(syms) if s == 'Au']
        idx_metal = [i for i, s in enumerate(syms) if s in ['Fe', 'Rh']]
        idx_o = [i for i, s in enumerate(syms) if s == 'O']

        def get_inter_dist(indices_a, indices_b):
            if len(indices_a) == 0 or len(indices_b) == 0: return 99.9
            tree_a = cKDTree(pos[indices_a])
            dists, _ = tree_a.query(pos[indices_b], k=1)
            return round(float(np.min(dists)), 4)

        def get_self_dist(indices):
            if len(indices) < 2: return 99.9
            pts = pos[indices]
            tree = cKDTree(pts)
            dists, _ = tree.query(pts, k=2)
            return round(float(np.min(dists[:, 1])), 4)

        stats["min_dist_Au_Au"] = get_self_dist(idx_au)
        stats["min_dist_Metal_O"] = get_inter_dist(idx_metal, idx_o)
        stats["min_dist_O_O"] = get_self_dist(idx_o)
        stats["min_dist_Au_Metal"] = get_inter_dist(idx_au, idx_metal)
        stats["min_dist_Au_O"] = get_inter_dist(idx_au, idx_o)

        full_tree = cKDTree(pos)
        d_glob, _ = full_tree.query(pos, k=2)
        stats["min_dist_global"] = round(float(np.min(d_glob[:, 1])), 4)

        if len(idx_au) > 0:
            stats["Au_connectivity"] = QualityControl.check_connectivity(pos[idx_au], cutoff=connectivity_cutoff)
        else:
            stats["Au_connectivity"] = 0.0

        return stats

    @staticmethod
    def check_connectivity(positions, cutoff=3.2):
        n = len(positions)
        if n < 10: return 0.0

        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=cutoff)

        if len(pairs) == 0:
            return round(1.0 / n, 4)

        row = [p[0] for p in pairs]
        col = [p[1] for p in pairs]
        data = np.ones(len(pairs))

        adj = csr_matrix((data, (row, col)), shape=(n, n))
        adj = adj + adj.T

        n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
        if n_components == 0: return 0.0

        _, counts = np.unique(labels, return_counts=True)
        largest_fraction = np.max(counts) / n
        return round(float(largest_fraction), 4)

    @staticmethod
    def validate_run(metrics: dict, cfg: SimulationConfig) -> Tuple[bool, str]:
        # 1. Global Fuse
        if metrics['min_dist_global'] < cfg.qc_min_dist_global:
            return False, f"Global Clash ({metrics['min_dist_global']} < {cfg.qc_min_dist_global})"

        # 2. Connectivity
        if metrics.get('Au_connectivity', 0) < cfg.min_shell_connectivity:
            return False, f"Shell Fragmented ({metrics['Au_connectivity']:.1%} connected)"

        # 3. Element Gates
        cut = cfg.qc_cutoffs
        if metrics['min_dist_Au_Au'] < cut['Au-Au']: return False, f"Au-Au Fusion < {cut['Au-Au']}"
        if metrics['min_dist_Metal_O'] < cut['Metal-O']: return False, f"Metal-O Fusion < {cut['Metal-O']}"
        if metrics['min_dist_O_O'] < cut['O-O']: return False, f"O-O Crash < {cut['O-O']}"
        if metrics['min_dist_Au_Metal'] < cut['Au-Metal']: return False, f"Au-Metal Crash < {cut['Au-Metal']}"
        if metrics['min_dist_Au_O'] < cut['Au-O']: return False, f"Au-O Crash < {cut['Au-O']}"

        return True, "Passed"

    @staticmethod
    def distance_distributions(atoms):
        """
        Returns nearest-neighbor distance distributions (Å) for:
          - Au-Au (nearest Au neighbor)
          - Au-O (nearest O to each Au)
          - Au-Metal (nearest Fe/Rh to each Au)
        """
        syms = np.array(atoms.get_chemical_symbols())
        pos = atoms.get_positions()

        idx_au = np.where(syms == "Au")[0]
        idx_o = np.where(syms == "O")[0]
        idx_m = np.where((syms == "Fe") | (syms == "Rh"))[0]

        dists = {}

        # Au-Au nearest neighbor distances
        if len(idx_au) >= 2:
            pts = pos[idx_au]
            tree = cKDTree(pts)
            nn, _ = tree.query(pts, k=2)  # k=2: self + nearest neighbor
            dists["Au-Au_nn"] = nn[:, 1]  # exclude self distance
        else:
            dists["Au-Au_nn"] = np.array([])

        # Au-O nearest neighbor distances (for each Au, nearest O)
        if len(idx_au) >= 1 and len(idx_o) >= 1:
            tree_o = cKDTree(pos[idx_o])
            nn_au_o, _ = tree_o.query(pos[idx_au], k=1)
            dists["Au-O_nn"] = nn_au_o
        else:
            dists["Au-O_nn"] = np.array([])

        # Au-Metal nearest neighbor distances (for each Au, nearest Fe/Rh)
        if len(idx_au) >= 1 and len(idx_m) >= 1:
            tree_m = cKDTree(pos[idx_m])
            nn_au_m, _ = tree_m.query(pos[idx_au], k=1)
            dists["Au-Metal_nn"] = nn_au_m
        else:
            dists["Au-Metal_nn"] = np.array([])

        return dists

    # ============================
    # ADD: Save histograms
    # ============================
    @staticmethod
    def save_distance_histograms(dist_dict, run_dir, run_name, bins=60):
        """
        Saves PNG histograms in run_dir. This is thesis-friendly evidence
        (not only minima).
        """
        for key, arr in dist_dict.items():
            if arr is None or len(arr) == 0:
                continue

            plt.figure()
            plt.hist(arr, bins=bins)
            plt.xlabel("Distance (Å)")
            plt.ylabel("Count")
            plt.title(f"{run_name} | {key}")
            out = os.path.join(run_dir, f"hist_{key}.png")
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()


# --- 3. THE PIPELINE ---
class ThesisPipelineV32:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self._setup_reproducibility()
        self._setup_directories()
        self._load_resources()
        self.stage = "Init"

    def _setup_reproducibility(self):
        # Local Windows optimization usually doesn't need this env var strictly,
        # but good to keep for determinism.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _setup_directories(self):
        self.root = self.cfg.base_dir
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        with open(os.path.join(self.root, "global_config.json"), "w") as f:
            f.write(self.cfg.to_json())

    def _load_resources(self):
        print("--- Loading Resources ---")

        # --- FORCE CPU MODE (Temporary fix for RTX 5070) ---
        self.device = 'cpu'
        print(f"   [System] Using device: {self.device} (CPU Fallback Mode)")

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # if self.device == 'cuda':
        #     print(f"   [Info] GPU: {torch.cuda.get_device_name(0)}")

        # Load CHGNet (It might warn about CUDA, but we will force calculations to CPU later)
        self.chgnet = CHGNet.load()

        # --- CRITICAL: Do not delete this file loading block ---
        if not os.path.exists(self.cfg.bulk_path):
            raise FileNotFoundError(f"Bulk XYZ not found at: {self.cfg.bulk_path}")
        self.bulk = read(self.cfg.bulk_path)
        self.bulk.translate(-self.bulk.get_center_of_mass())

    def _assert_finite(self, atoms, stage_name):
        """Watchdog for NaN/Inf coordinates."""
        if not np.isfinite(atoms.get_positions()).all():
            raise ValueError(f"CRITICAL: Non-finite coordinates detected at stage '{stage_name}'.")

    def _assert_atom_count(self, atoms, expected_count, stage_name):
        """Verify Mass Conservation."""
        if len(atoms) != expected_count:
            raise ValueError(
                f"MASS LOSS: Atom count changed at '{stage_name}'. Expected {expected_count}, got {len(atoms)}.")

    # --- MODULE A: CORE ---
    def generate_core(self, diameter, run_dir):
        radius = diameter / 2.0
        atoms = self.bulk.copy()

        dists = np.linalg.norm(atoms.get_positions(), axis=1)
        atoms = atoms[dists <= radius]
        self._assert_finite(atoms, "Core Spherical Cut")
        write(os.path.join(run_dir, "01_core_raw.xyz"), atoms)

        nl = NeighborList([self.cfg.fe_o_cutoff / 2.0] * len(atoms),
                          self_interaction=False, bothways=True)
        nl.update(atoms)

        valid_indices = []
        metal_count_initial = 0
        metal_count_removed = 0

        for i, atom in enumerate(atoms):
            if atom.symbol in ['Fe', 'Rh']:
                metal_count_initial += 1
                indices, _ = nl.get_neighbors(i)
                n_ox = sum(1 for n in indices if atoms[n].symbol == 'O')

                if n_ox < self.cfg.min_oxygen_neighbors:
                    metal_count_removed += 1
                    continue
            valid_indices.append(i)

        atoms = atoms[valid_indices]
        self._assert_finite(atoms, "Core Surface-Cleaning Filter")

        if len(atoms) == 0:
            raise ValueError(f"Core vanished after surface cleaning (Diameter {diameter}A).")

        if metal_count_initial > 0:
            fraction = metal_count_removed / metal_count_initial
            if fraction > self.cfg.max_removal_fraction:
                print(f"   [Warning] Aggressive cleaning! Removed {fraction:.1%} of surface metals.")

        # RE-CENTERING
        centroid = np.mean(atoms.get_positions(), axis=0)
        atoms.translate(-centroid)
        self._assert_finite(atoms, "Core Centering")

        write(os.path.join(run_dir, "02_core_cleaned.xyz"), atoms)
        return atoms

    # --- MODULE B: SHELL ---
    def generate_shell(self, core, thickness, run_dir):
        core_pos = core.get_positions()
        core_radius = np.max(np.linalg.norm(core_pos, axis=1))

        hole_radius = core_radius + self.cfg.gap_size
        target_outer = hole_radius + thickness
        buffer_radius = target_outer + self.cfg.wulff_buffer

        # Wulff Gen
        vol_sphere = (4 / 3) * np.pi * buffer_radius ** 3
        vol_atom = (self.cfg.lattice_au ** 3) / 4.0
        n_atoms_ideal = int(vol_sphere / vol_atom)

        # Safety Clamp
        if n_atoms_ideal > self.cfg.max_shell_atoms_est:
            print(
                f"   [Info] Clamping Wulff generation to {self.cfg.max_shell_atoms_est} atoms (Ideal: {n_atoms_ideal})")
            n_atoms_est = self.cfg.max_shell_atoms_est
        else:
            n_atoms_est = n_atoms_ideal

        try:
            raw = wulff_construction('Au', [(1, 1, 1), (1, 0, 0), (1, 1, 0)],
                                     self.cfg.surf_energies, size=n_atoms_est,
                                     structure='fcc', latticeconstant=self.cfg.lattice_au)
        except (TypeError, ValueError):
            raw = wulff_construction('Au', [(1, 1, 1), (1, 0, 0), (1, 1, 0)],
                                     self.cfg.surf_energies, size=n_atoms_est,
                                     latticeconstant=self.cfg.lattice_au)

        shell = Atoms(raw)
        shell.translate(-shell.get_center_of_mass())
        self._assert_finite(shell, "Shell Wulff Centered")

        # VERIFICATION
        r_actual = np.max(np.linalg.norm(shell.get_positions(), axis=1))

        if r_actual < buffer_radius - 0.5:
            msg = f"Wulff generation undersized! Actual R={r_actual:.1f} < Buffer R={buffer_radius:.1f}."
            if n_atoms_est == self.cfg.max_shell_atoms_est:
                msg += f" Clamp ({self.cfg.max_shell_atoms_est}) prevented reaching size."
            else:
                msg += " Increase wulff_buffer or apply a safety multiplier (e.g. 1.2x atoms)."
            raise ValueError(msg)

        # Cut Hole
        dists = np.linalg.norm(shell.get_positions(), axis=1)
        mask = (dists > hole_radius) & (dists <= target_outer + 1.0)
        shell = shell[mask]
        self._assert_finite(shell, "Shell Hole Cut")

        if len(shell) > self.cfg.max_shell_atoms_real:
            raise ValueError(f"Final shell size {len(shell)} > limit {self.cfg.max_shell_atoms_real}.")

        if len(shell) < 10:
            raise ValueError("Shell too small (<10 atoms) to evaluate.")

        if set(shell.get_chemical_symbols()) != {'Au'}:
            raise ValueError("Shell contains non-Au atoms! Logic error.")

        write(os.path.join(run_dir, "03_shell_raw.xyz"), shell)

        # Two-Tree Clash Removal
        shell_pos = shell.get_positions()
        core_pos = core.get_positions()
        core_syms = np.array(core.get_chemical_symbols())

        idx_o = [i for i, s in enumerate(core_syms) if s == 'O']
        idx_m = [i for i, s in enumerate(core_syms) if s in ['Fe', 'Rh']]

        mask_o = np.ones(len(shell), dtype=bool)
        if len(idx_o) > 0:
            tree_o = cKDTree(core_pos[idx_o])
            d_o, _ = tree_o.query(shell_pos, k=1)
            mask_o = d_o > self.cfg.safe_dist_au_o

        mask_m = np.ones(len(shell), dtype=bool)
        if len(idx_m) > 0:
            tree_m = cKDTree(core_pos[idx_m])
            d_m, _ = tree_m.query(shell_pos, k=1)
            mask_m = d_m > self.cfg.safe_dist_au_fe

        valid_mask = mask_o & mask_m
        n_clash = len(shell) - np.sum(valid_mask)
        shell = shell[valid_mask]
        self._assert_finite(shell, "Shell Filtered")

        # Early Abort
        if len(shell) < 10:
            raise ValueError("Shell depleted after clash filtering.")

        connectivity = QualityControl.check_connectivity(shell.get_positions(), cutoff=self.cfg.connectivity_cutoff)
        print(f"   [Shell] Connectivity Check: {connectivity:.1%} connected.")
        if connectivity < self.cfg.min_shell_connectivity:
            raise ValueError(
                f"Shell fragmented after filtering ({connectivity:.1%} < {self.cfg.min_shell_connectivity:.0%}).")

        write(os.path.join(run_dir, "04_shell_filtered.xyz"), shell)
        print(f"   [Shell] Valid & Filtered. Removed {n_clash} atoms.")
        return shell

    # --- MODULE C: PHASED RELAXATION ---
    def relax_structure(self, atoms, run_dir):
        n_atoms_start = len(atoms)

        # 0. Initial Integrity Checks
        self._assert_finite(atoms, "Pre-Relaxation Start")
        atoms.set_cell([150.0, 150.0, 150.0])
        atoms.center()
        self._assert_finite(atoms, "Pre-Relaxation Center")
        atoms.pbc = False

        # Fail-Fast: Global Pre-MD Fuse
        pos = atoms.get_positions()
        if len(pos) < 2:
            raise ValueError("Pre-MD Critical Failure: Not enough atoms (<2) for distance check.")

        full_tree = cKDTree(pos)
        d_glob, _ = full_tree.query(pos, k=2)
        min_global = np.min(d_glob[:, 1])

        if min_global < self.cfg.qc_min_dist_global:
            raise ValueError(
                f"Pre-MD Critical Failure: Global min {min_global:.2f}A < Fuse {self.cfg.qc_min_dist_global}A.")

        # Fail-Fast: Specific O-O Check
        syms = atoms.get_chemical_symbols()
        idx_o = [i for i, s in enumerate(syms) if s == 'O']
        if len(idx_o) > 1:
            pos_o = atoms.get_positions()[idx_o]
            tree_o = cKDTree(pos_o)
            d_oo, _ = tree_o.query(pos_o, k=2)
            min_oo = np.min(d_oo[:, 1])
            threshold = self.cfg.qc_cutoffs["O-O"]
            if min_oo < threshold:
                raise ValueError(f"Pre-MD Critical Failure: O-O atoms at {min_oo:.2f}A < QC Gate {threshold}A.")

        # 1. Vectorized Hybrid Freeze
        core_indices = np.array([a.index for a in atoms if a.symbol != 'Au'])

        if len(core_indices) == 0:
            raise ValueError("No core atoms detected (symbol != 'Au'). Cannot apply freeze.")

        core_pos = atoms.get_positions()[core_indices]
        core_centroid = np.mean(core_pos, axis=0)
        core_r_max = np.max(np.linalg.norm(core_pos - core_centroid, axis=1))

        freeze_r = core_r_max - self.cfg.freeze_buffer
        if freeze_r < 1.0: freeze_r = 1.0

        all_dists = np.linalg.norm(atoms.get_positions() - core_centroid, axis=1)
        mask = np.zeros(len(atoms), dtype=bool)
        mask[core_indices] = all_dists[core_indices] < freeze_r
        atoms.set_constraint(FixAtoms(mask=mask))

        # 2. Physics Init
        rh_idx = [a.index for a in atoms if a.symbol == 'Rh']
        n_rh_original = len(rh_idx)

        fe_idx = [a.index for a in atoms if a.symbol == 'Fe']
        magmoms = np.zeros(len(atoms))
        magmoms[rh_idx] = 5.0
        magmoms[fe_idx] = -5.0
        atoms.set_initial_magnetic_moments(magmoms)

        syms = atoms.get_chemical_symbols()
        calc_syms = ['Fe' if s == 'Rh' else s for s in syms]
        atoms.set_chemical_symbols(calc_syms)

        atoms.calc = CHGNetCalculator(model=self.chgnet, use_device=self.device)

        # --- PHASE I: FIRE ---
        print("   [Relax] Phase I: FIRE (De-clash)...")
        try:
            # CHANGED: Added logfile for FIRE
            opt = FIRE(atoms, logfile=os.path.join(run_dir, "fire.log"))
            opt.run(fmax=2.0, steps=50)
            self._assert_finite(atoms, "After FIRE")
            self._assert_atom_count(atoms, n_atoms_start, "After FIRE")
            write(os.path.join(run_dir, "06_after_fire.xyz"), atoms)
        except Exception as e:
            print(f"   [Warning] FIRE non-convergence: {e}")

        # --- PHASE II: Langevin ---
        print(f"   [Relax] Phase II: Langevin Heating ({self.cfg.anneal_temp}K, 5ps)...")

        MaxwellBoltzmannDistribution(atoms, temperature_K=self.cfg.anneal_temp)
        Stationary(atoms, preserve_temperature=True)

        dyn = Langevin(atoms, timestep=self.cfg.timestep_fs * units.fs,
                       temperature_K=self.cfg.anneal_temp, friction=0.02)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dyn.run(self.cfg.anneal_steps)
        self._assert_finite(atoms, "After Langevin")
        self._assert_atom_count(atoms, n_atoms_start, "After Langevin")
        write(os.path.join(run_dir, "07_after_md.xyz"), atoms)

        # --- PHASE IIb: Cooling ---
        print("   [Relax] Phase IIb: Langevin Cooling (10K)...")
        dyn_cool = Langevin(atoms, timestep=self.cfg.timestep_fs * units.fs,
                            temperature_K=10.0, friction=0.02)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dyn_cool.run(500)

        # --- PHASE III: BFGS ---
        print("   [Relax] Phase III: BFGS (Final)...")
        try:
            opt = BFGS(atoms, logfile=os.path.join(run_dir, "bfgs.log"))
            opt.run(fmax=0.1)
            self._assert_finite(atoms, "After BFGS")
            self._assert_atom_count(atoms, n_atoms_start, "After BFGS")
        except Exception:
            pass

        # Restore Symbols & VERIFY INTEGRITY
        final_syms = np.array(atoms.get_chemical_symbols())
        final_syms[rh_idx] = 'Rh'
        atoms.set_chemical_symbols(final_syms)

        # 1. Count Check
        n_rh_final = atoms.get_chemical_symbols().count('Rh')
        if n_rh_final != n_rh_original:
            raise ValueError(f"Integrity Loss: Started with {n_rh_original} Rh, restored {n_rh_final}.")

        # 2. Strict Index Check
        current_syms = np.array(atoms.get_chemical_symbols())
        if not np.all(current_syms[rh_idx] == 'Rh'):
            raise ValueError("Integrity Loss: Rh indices not restored exactly.")

        write(os.path.join(run_dir, "08_final_relaxed.xyz"), atoms)
        return atoms

    # --- MASTER RUNNER ---
    def run(self):
        print(f"=== THESIS PIPELINE V32 (LOCAL GOLD STANDARD) ===")
        print(f"Device: {self.device}")

        summary_log = []

        for D in self.cfg.diameters:
            for T in self.cfg.thicknesses:
                run_name = f"D{int(D)}_T{int(T)}"
                print(f"\n>>> PROCESSING: {run_name}")

                run_dir = os.path.join(self.root, run_name)
                os.makedirs(run_dir, exist_ok=True)

                t0 = time.time()
                status = "Success"
                error = ""
                qc_metrics = {}
                qc_passed = False
                qc_reason = ""
                self.stage = "Init"

                try:
                    self.stage = "Core Generation"
                    core = self.generate_core(diameter=D, run_dir=run_dir)
                    if len(core) < 10: raise ValueError("Core too small")

                    self.stage = "Shell Generation"
                    shell = self.generate_shell(core=core, thickness=T, run_dir=run_dir)
                    if len(shell) < 10: raise ValueError("Shell too small")

                    self.stage = "Merging"
                    merged = core + shell
                    self._assert_finite(merged, "After Merge")
                    write(os.path.join(run_dir, "05_merged_prerelax.xyz"), merged)

                    # Audit Logging
                    core_syms = core.get_chemical_symbols()
                    merged_syms = merged.get_chemical_symbols()
                    has_rh = "Rh" in merged_syms

                    run_meta = {
                        "has_Rh_core": "Rh" in core_syms,
                        "has_Rh_merged": has_rh,
                        "unexpected_Rh_in_shell": "Rh" in shell.get_chemical_symbols(),
                        "surrogate_active_during_relax": has_rh,
                        "surrogate_details": "Rh->Fe during CHGNet" if has_rh else "None (No Rh)"
                    }
                    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
                        json.dump(run_meta, f, indent=4)

                    self.stage = "Relaxation (FIRE/MD/BFGS)"
                    final = self.relax_structure(atoms=merged, run_dir=run_dir)
                    # ============================
                    # ADD: Convergence reporting
                    # ============================
                    try:
                        final_energy_eV = float(final.get_potential_energy())  # eV (CHGNet)
                        forces = final.get_forces()  # eV/Å
                        final_fmax_eV_per_A = float(np.linalg.norm(forces, axis=1).max())
                    except Exception as _:
                        final_energy_eV = None
                        final_fmax_eV_per_A = None

                    self.stage = "QC Validation"
                    qc_metrics = QualityControl.analyze_structure(
                        final,
                        connectivity_cutoff=self.cfg.connectivity_cutoff,
                        name=run_name
                    )
                    qc_metrics["final_energy_eV"] = final_energy_eV
                    qc_metrics["final_fmax_eV_per_A"] = final_fmax_eV_per_A

                    # ============================
                    # ADD: Distance distributions + hist plots + stats
                    # ============================
                    dist_dict = QualityControl.distance_distributions(final)
                    QualityControl.save_distance_histograms(dist_dict, run_dir=run_dir, run_name=run_name)

                    # Add stats to JSON (keeps JSON small but informative)
                    for k, arr in dist_dict.items():
                        if arr is not None and len(arr) > 0:
                            qc_metrics[f"{k}_mean"] = float(np.mean(arr))
                            qc_metrics[f"{k}_p05"] = float(np.percentile(arr, 5))
                            qc_metrics[f"{k}_p50"] = float(np.percentile(arr, 50))
                            qc_metrics[f"{k}_p95"] = float(np.percentile(arr, 95))

                    qc_passed, qc_reason = QualityControl.validate_run(qc_metrics, self.cfg)

                    if not qc_passed:
                        status = "QC_Failed"
                        error = qc_reason
                        print(f"   [QC] Failed: {qc_reason}")
                    else:
                        print(f"   [QC] Passed.")

                    with open(os.path.join(run_dir, "quality_report.json"), "w") as f:
                        json.dump(qc_metrics, f, indent=4)

                except Exception as e:
                    status = "Run_Failed"
                    error = str(e)
                    print(f"   [Error] {e}")
                    # Traceback + Stage Logging
                    with open(os.path.join(run_dir, "failure_report.json"), "w") as f:
                        json.dump({
                            "run": run_name,
                            "stage": self.stage,
                            "error": error,
                            "traceback": traceback.format_exc()
                        }, f, indent=4)

                entry = {
                    "Diameter": D, "Thickness": T, "Status": status,
                    "Time": round(time.time() - t0, 2), "Error": error
                }
                if qc_metrics: entry.update(qc_metrics)
                summary_log.append(entry)

        df = pd.DataFrame(summary_log)
        df.to_csv(os.path.join(self.root, "summary_log.csv"), index=False)
        print("\n=== PIPELINE FINISHED ===")


# --- EXECUTION ---
if __name__ == "__main__":
    config = SimulationConfig(
        # Note: Paths updated for local environment in dataclass defaults
        diameters=(35.0,),
        thicknesses=(1.0, 2.0),
        connectivity_cutoff=3.2,
        # Production Settings
        anneal_steps=5000,
        qc_cutoffs={
            "Au-Au": 2.5,
            "Metal-O": 1.6,
            "O-O": 2.0,
            "Au-Metal": 2.0,
            "Au-O": 1.8
        }
    )

    pipeline = ThesisPipelineV32(config)
    pipeline.run()