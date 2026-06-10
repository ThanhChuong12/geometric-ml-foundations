#!/usr/bin/env python3
"""
prepare_qm9_subsets.py
======================
Data preparation pipeline for the NequIP ablation study on QM9.

Downloads the QM9 dataset via PyTorch Geometric, converts each molecule
into an ASE Atoms object with energy and forces labels, then samples
reproducible subsets (100 and 1000 molecules) and exports them as .extxyz
files compatible with NequIP's ASEDataModule.

Design Principles:
    - Single Responsibility: download, conversion, sampling, and I/O are
      each handled by distinct, testable functions.
    - Reproducibility: all random operations use a fixed seed.
    - Robustness: comprehensive logging, type hints, and validation.

Usage:
    python prepare_qm9_subsets.py \\
        --root ./data/qm9_raw \\
        --output-dir ../data/qm9 \\
        --seed 42 \\
        --subset-sizes 100 1000

Author: AI/ML Engineering Pipeline
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("prepare_qm9_subsets")


# ===========================================================================
# 1. Download
# ===========================================================================

def download_qm9(root: str) -> "torch_geometric.datasets.QM9":
    """Download (or load from cache) the QM9 dataset via PyTorch Geometric.

    Parameters
    ----------
    root : str
        Directory where the raw / processed QM9 data will be stored.

    Returns
    -------
    torch_geometric.datasets.QM9
        The full QM9 dataset object (≈130 k molecules).
    """
    from torch_geometric.datasets import QM9

    logger.info("Loading QM9 dataset from '%s' (will download if needed)...", root)
    dataset = QM9(root=root)
    logger.info(
        "QM9 loaded successfully: %d molecules, %d properties per molecule.",
        len(dataset),
        dataset[0].y.shape[1] if dataset[0].y.dim() > 1 else 1,
    )
    return dataset


# ===========================================================================
# 2. Conversion: PyG Data → ASE Atoms
# ===========================================================================

# QM9 target index mapping (PyTorch Geometric convention):
#   0: mu (Debye), 1: alpha (Bohr^3), 2: HOMO (Ha), 3: LUMO (Ha),
#   4: gap (Ha),   5: R2 (Bohr^2),    6: ZPVE (Ha), 7: U0 (Ha),
#   8: U (Ha),     9: H (Ha),        10: G (Ha),   11: Cv (cal/mol·K)
# We use U0 (index 7) as the "energy" label — the internal energy at 0 K.
_QM9_ENERGY_INDEX: int = 7

# Mapping from atomic number to chemical symbol (elements present in QM9)
_Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def _pyg_data_to_ase_atoms(
    data,  # torch_geometric.data.Data
    energy_index: int = _QM9_ENERGY_INDEX,
) -> "ase.Atoms":
    """Convert a single PyG QM9 Data object to an ASE Atoms object.

    The resulting Atoms object stores the energy (U0 in Hartree)
    in its atoms.info dict. No force labels are attached since QM9
    only contains scalar molecular properties.

    Parameters
    ----------
    data : torch_geometric.data.Data
        A single QM9 molecule from PyG.
    energy_index : int
        Column index in ``data.y`` to use as the total energy label.

    Returns
    -------
    ase.Atoms
        Atoms object with positions, atomic numbers, and energy in info.
    """
    import ase

    atomic_numbers: np.ndarray = data.z.cpu().numpy().astype(int)
    positions: np.ndarray = data.pos.cpu().numpy().astype(np.float64)

    # Extract energy (scalar)
    if data.y.dim() > 1:
        energy: float = float(data.y[0, energy_index].cpu().item())
    else:
        energy = float(data.y[energy_index].cpu().item())

    atoms = ase.Atoms(
        numbers=atomic_numbers,
        positions=positions,
        pbc=False,
    )
    atoms.info["energy"] = energy

    return atoms


def extract_atoms_list(
    dataset,  # torch_geometric.datasets.QM9
    energy_index: int = _QM9_ENERGY_INDEX,
    max_molecules: Optional[int] = None,
) -> List["ase.Atoms"]:
    """Convert the entire PyG QM9 dataset into a list of ASE Atoms.

    Parameters
    ----------
    dataset : torch_geometric.datasets.QM9
        The full QM9 dataset.
    energy_index : int
        Column index for energy in ``data.y``.
    max_molecules : int or None
        If set, only convert the first ``max_molecules`` entries (useful for
        debugging).

    Returns
    -------
    list[ase.Atoms]
        List of ASE Atoms objects with energy attached in atoms.info.
    """
    n = len(dataset) if max_molecules is None else min(max_molecules, len(dataset))
    logger.info("Converting %d PyG Data objects → ASE Atoms...", n)

    atoms_list: List["ase.Atoms"] = []
    skipped = 0
    for i in range(n):
        try:
            atoms = _pyg_data_to_ase_atoms(dataset[i], energy_index=energy_index)
            atoms_list.append(atoms)
        except Exception as exc:
            logger.warning("Skipped molecule index %d due to error: %s", i, exc)
            skipped += 1

    logger.info(
        "Conversion complete: %d molecules converted, %d skipped.",
        len(atoms_list),
        skipped,
    )
    return atoms_list


# ===========================================================================
# 3. Subset Sampling (Pure Function)
# ===========================================================================

def sample_subset(
    atoms_list: List["ase.Atoms"],
    n: int,
    seed: int,
) -> List["ase.Atoms"]:
    """Randomly sample ``n`` molecules from ``atoms_list`` (reproducibly).

    This is a *pure function* with respect to the random seed — calling it
    with the same inputs always produces the same output.

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        The source pool of molecules.
    n : int
        Number of molecules to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[ase.Atoms]
        The sampled subset (unordered).

    Raises
    ------
    ValueError
        If ``n`` exceeds the pool size.
    """
    if n > len(atoms_list):
        raise ValueError(
            f"Requested subset size ({n}) exceeds pool size ({len(atoms_list)})."
        )
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(atoms_list), size=n, replace=False)
    indices.sort()  # deterministic ordering
    subset = [atoms_list[i] for i in indices]
    logger.info(
        "Sampled subset of %d molecules (seed=%d). Index range: [%d, %d].",
        n,
        seed,
        int(indices[0]),
        int(indices[-1]),
    )
    return subset


# ===========================================================================
# 4. I/O — Save to .extxyz
# ===========================================================================

def save_extxyz(atoms_list: List["ase.Atoms"], path: str | Path) -> None:
    """Write a list of ASE Atoms to an extended XYZ file.

    The .extxyz format is directly readable by NequIP's ``ASEDataModule``.

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        Molecules to write.
    path : str or Path
        Output file path.
    """
    from ase.io import write

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write(str(path), atoms_list, format="extxyz")
    file_size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(
        "Saved %d molecules → '%s' (%.2f MB).",
        len(atoms_list),
        path,
        file_size_mb,
    )

def validate_extxyz(path: str | Path, expected_count: int) -> bool:
    """Quick sanity check: re-read the file and verify molecule count + fields.

    Parameters
    ----------
    path : str or Path
        Path to the .extxyz file.
    expected_count : int
        Expected number of frames.

    Returns
    -------
    bool
        True if validation passes.
    """
    from ase.io import read

    frames = read(str(path), index=":", format="extxyz")
    if len(frames) != expected_count:
        logger.error(
            "Validation FAILED for '%s': expected %d frames, got %d.",
            path,
            expected_count,
            len(frames),
        )
        return False

    # Check that energy is present in the first frame (either in info or calc.results)
    sample = frames[0]
    has_energy = ("energy" in sample.info) or (sample.calc is not None and "energy" in sample.calc.results)

    if not has_energy:
        logger.error("Validation FAILED: 'energy' missing from atoms.info and calculator results.")
        return False

    energy_val = sample.info.get("energy", None)
    if energy_val is None and sample.calc is not None:
        energy_val = sample.calc.results.get("energy", None)

    if not isinstance(energy_val, (float, int, np.floating, np.integer)):
        logger.error("Validation FAILED: 'energy' is not a numeric type.")
        return False

    logger.info(
        "Validation PASSED for '%s': %d frames, energy ✓.",
        path,
        len(frames),
    )
    return True



# ===========================================================================
# 5. Main Orchestration
# ===========================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare QM9 subsets for NequIP ablation study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data/qm9_raw",
        help="Root directory for caching the raw QM9 download.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/qm9",
        help="Directory to save the output .extxyz subsets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subsetting.",
    )
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        default=[100, 1000],
        help="List of subset sizes to generate.",
    )
    parser.add_argument(
        "--energy-index",
        type=int,
        default=_QM9_ENERGY_INDEX,
        help="Column index in QM9 data.y to use as energy (7 = U0).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip re-reading and validating written .extxyz files.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point for the QM9 data preparation pipeline."""
    args = parse_args(argv)

    logger.info("=" * 60)
    logger.info("QM9 Data Preparation Pipeline")
    logger.info("=" * 60)
    logger.info("Arguments: %s", vars(args))

    # Step 1: Download / load QM9
    dataset = download_qm9(args.root)

    # Step 2: Convert to ASE Atoms
    atoms_list = extract_atoms_list(dataset, energy_index=args.energy_index)

    # Step 3 & 4: For each requested subset size, sample and save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for size in args.subset_sizes:
        logger.info("-" * 40)
        logger.info("Generating subset of size %d...", size)

        subset = sample_subset(atoms_list, n=size, seed=args.seed)

        out_path = output_dir / f"qm9_subset_{size}.extxyz"
        save_extxyz(subset, out_path)

        if not args.skip_validation:
            ok = validate_extxyz(out_path, expected_count=size)
            if not ok:
                logger.error("Aborting due to validation failure.")
                sys.exit(1)

    logger.info("=" * 60)
    logger.info("All subsets generated successfully.")
    logger.info("Output directory: %s", output_dir.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
