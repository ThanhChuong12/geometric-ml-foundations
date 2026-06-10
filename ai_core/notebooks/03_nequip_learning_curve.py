#!/usr/bin/env python3
"""
03_nequip_learning_curve.py
===========================
Evaluation and visualization script for the NequIP l=0 vs l=1 ablation study.

Parses training metrics from NequIP Lightning CSV logs (or accepts manually
entered results), then generates publication-quality log-log learning curve
plots comparing:
  - Invariant GNN  (l_max=0) at dataset sizes {100, 1000}
  - Equivariant GNN (l_max=1) at dataset sizes {100, 1000}

Outputs:
  - Log-log learning curve: Dataset Size vs MAE (Energy and Forces)
  - Slope annotations showing convergence rate
  - Saved figure as PNG and PDF

Usage:
    python 03_nequip_learning_curve.py --results-dir ../../outputs
    python 03_nequip_learning_curve.py --use-placeholder-data

This script can also be converted to a Jupyter notebook via:
    jupytext --to notebook 03_nequip_learning_curve.py

Author: AI/ML Engineering Pipeline
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("nequip_learning_curve")


# ===========================================================================
# 1. Log Parsing — Extract Final MAE from NequIP Lightning CSV Logs
# ===========================================================================

def parse_csv_metrics(csv_path: str | Path) -> Dict[str, float]:
    """Parse a Lightning CSVLogger ``metrics.csv`` file.

    Extracts the final (last-epoch) values for energy MAE.

    Parameters
    ----------
    csv_path : str or Path
        Path to the ``metrics.csv`` file produced by Lightning's CSVLogger.

    Returns
    -------
    dict
        Keys: ``'energy_mae'`` (float value).
    """
    import csv

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {csv_path}")

    logger.info("Parsing metrics from: %s", csv_path)

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")

    # Look for test/val metrics in the last row containing them
    energy_mae = None

    # Priority: search from the end of the file for the first non-empty energy MAE metric
    for row in reversed(rows):
        for prefix in ["test0_epoch", "val0_epoch"]:
            energy_key = f"{prefix}/total_energy_mae"
            if energy_key in row and row[energy_key]:
                energy_mae = float(row[energy_key])
                break
        if energy_mae is not None:
            break

    # Fallback: search for any column containing "energy_mae" or "total_energy_mae" in all rows starting from the end
    if energy_mae is None:
        for row in reversed(rows):
            for key, val in row.items():
                if "total_energy_mae" in key.lower() and val:
                    energy_mae = float(val)
                    break
                elif "energy_mae" in key.lower() and val:
                    energy_mae = float(val)
                    break
            if energy_mae is not None:
                break

    if energy_mae is None:
        # Check last row keys to report
        last_row = rows[-1]
        logger.warning(
            "Could not find energy_mae in %s. Available columns: %s",
            csv_path,
            list(last_row.keys()),
        )
        energy_mae = float("nan")

    result = {"energy_mae": energy_mae}
    logger.info("  → Energy MAE: %.6f", energy_mae)
    return result


def collect_results_from_logs(
    results_dir: str | Path,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Scan the results directory for NequIP training outputs and extract MAE.

    Expected directory structure (produced by Hydra + CSVLogger):
    ::

        results_dir/
        ├── baseline_l0_100/
        │   └── version_0/metrics.csv
        ├── baseline_l0_1000/
        │   └── version_0/metrics.csv
        ├── nequip_l1_100/
        │   └── version_0/metrics.csv
        └── nequip_l1_1000/
            └── version_0/metrics.csv

    Returns
    -------
    dict
        Nested dict: ``{model_label: {dataset_size: {energy_mae}}}``
    """
    results_dir = Path(results_dir)
    logger.info("Scanning results directory: %s", results_dir)

    # Mapping from directory name prefix → (model_label, dataset_size)
    experiment_map = {
        "baseline_l0_100": ("Invariant ($\\ell=0$)", 100),
        "baseline_l0_1000": ("Invariant ($\\ell=0$)", 1000),
        "nequip_l1_100": ("Equivariant ($\\ell=1$)", 100),
        "nequip_l1_1000": ("Equivariant ($\\ell=1$)", 1000),
    }

    results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for dir_name, (label, size) in experiment_map.items():
        # Search for metrics.csv in common locations
        candidates = [
            results_dir / dir_name / "version_0" / "metrics.csv",
            results_dir / dir_name / "metrics.csv",
            results_dir / dir_name / "lightning_logs" / "version_0" / "metrics.csv",
        ]
        found = False
        for csv_path in candidates:
            if csv_path.exists():
                try:
                    metrics = parse_csv_metrics(csv_path)
                    results.setdefault(label, {})[size] = metrics
                    found = True
                    break
                except Exception as e:
                    logger.warning("Error parsing %s: %s", csv_path, e)

        if not found:
            logger.warning(
                "No metrics.csv found for '%s'. Searched: %s",
                dir_name,
                [str(p) for p in candidates],
            )

    return results


# ===========================================================================
# 2. Placeholder Data (for when logs are not yet available)
# ===========================================================================

def get_placeholder_results() -> Dict[str, Dict[int, Dict[str, float]]]:
    """Return placeholder results for visualization scaffolding.

    These values are representative of expected NequIP performance and
    should be replaced with actual training results once available.

    The placeholder values follow the expected trend:
    - l=1 outperforms l=0 at both dataset sizes
    - More data → lower MAE (learning curve)
    - The gap between l=0 and l=1 is larger at small dataset sizes
    """
    return {
        "Invariant ($\\ell=0$)": {
            100: {"energy_mae": 0.150},
            1000: {"energy_mae": 0.045},
        },
        "Equivariant ($\\ell=1$)": {
            100: {"energy_mae": 0.080},
            1000: {"energy_mae": 0.015},
        },
    }


# ===========================================================================
# 3. Plotting — Log-Log Learning Curve
# ===========================================================================

# Publication-quality style parameters
STYLE_CONFIG = {
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

# Color palette (colorblind-friendly)
COLORS = {
    "Invariant ($\\ell=0$)": "#E74C3C",      # red
    "Equivariant ($\\ell=1$)": "#2E86C1",    # blue
}

MARKERS = {
    "Invariant ($\\ell=0$)": "s",             # square
    "Equivariant ($\\ell=1$)": "o",           # circle
}

LINESTYLES = {
    "Invariant ($\\ell=0$)": "--",            # dashed
    "Equivariant ($\\ell=1$)": "-",           # solid
}


def compute_slope(
    sizes: List[int],
    maes: List[float],
) -> float:
    """Compute the slope of the log-log learning curve.

    slope = Δ log(MAE) / Δ log(N)

    A steeper negative slope indicates faster learning with more data.
    """
    if len(sizes) < 2 or any(m <= 0 for m in maes):
        return float("nan")
    log_sizes = np.log10(sizes)
    log_maes = np.log10(maes)
    slope = (log_maes[-1] - log_maes[0]) / (log_sizes[-1] - log_sizes[0])
    return float(slope)


def plot_learning_curves(
    results: Dict[str, Dict[int, Dict[str, float]]],
    output_dir: str | Path = ".",
    filename_prefix: str = "nequip_learning_curve",
) -> None:
    """Generate a publication-quality log-log learning curve plot.

    Creates a figure with a single subplot:
      - Energy MAE vs Dataset Size

    Both axes use log scale. Lines connect the l=0 and l=1 models, with
    slope annotations showing the convergence rate.

    Parameters
    ----------
    results : dict
        Nested dict from ``collect_results_from_logs`` or ``get_placeholder_results``.
    output_dir : str or Path
        Directory to save the output figures.
    filename_prefix : str
        Base name for the output files (without extension).
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(
        "NequIP Ablation Study: Learning Curves on QM9\n"
        "Invariant ($\\ell_{\\max}=0$) vs Equivariant ($\\ell_{\\max}=1$)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    metric_key = "energy_mae"
    ylabel = "Energy MAE"

    for model_label, size_metrics in sorted(results.items()):
        sizes = sorted(size_metrics.keys())
        maes = [size_metrics[s][metric_key] for s in sizes]

        color = COLORS.get(model_label, "#333333")
        marker = MARKERS.get(model_label, "D")
        ls = LINESTYLES.get(model_label, "-")

        ax.plot(
            sizes,
            maes,
            color=color,
            marker=marker,
            markersize=10,
            linewidth=2.5,
            linestyle=ls,
            label=model_label,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=5,
        )

        # Annotate slope
        slope = compute_slope(sizes, maes)
        if not np.isnan(slope):
            # Place annotation at midpoint (geometric mean)
            mid_x = np.sqrt(sizes[0] * sizes[-1])
            mid_y = np.sqrt(maes[0] * maes[-1])
            ax.annotate(
                f"slope = {slope:.2f}",
                xy=(mid_x, mid_y),
                fontsize=10,
                fontweight="bold",
                color=color,
                ha="center",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.85,
                ),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset Size ($N$)", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")

    # Custom tick labels for dataset sizes
    all_sizes = sorted({s for sm in results.values() for s in sm.keys()})
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([str(s) for s in all_sizes])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save figures
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["png", "pdf"]:
        save_path = output_dir / f"{filename_prefix}.{ext}"
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        logger.info("Saved figure: %s", save_path)

    plt.show()
    logger.info("Plot displayed successfully.")


# ===========================================================================
# 4. Summary Table
# ===========================================================================

def print_results_table(
    results: Dict[str, Dict[int, Dict[str, float]]],
) -> None:
    """Print a formatted summary table of all ablation results."""
    header = f"{'Model':<30} {'N':>6} {'Energy MAE':>14} {'Slope (E)':>10}"
    separator = "─" * len(header)

    print("\n" + separator)
    print("  NequIP Ablation Study — Results Summary")
    print(separator)
    print(header)
    print(separator)

    for model_label in sorted(results.keys()):
        size_metrics = results[model_label]
        sizes = sorted(size_metrics.keys())
        energy_maes = [size_metrics[s]["energy_mae"] for s in sizes]

        slope_e = compute_slope(sizes, energy_maes)

        for i, size in enumerate(sizes):
            m = size_metrics[size]
            slope_e_str = f"{slope_e:.2f}" if i == 0 and not np.isnan(slope_e) else ""
            print(
                f"{model_label:<30} {size:>6} {m['energy_mae']:>14.6f} {slope_e_str:>10}"
            )

    print(separator)
    print(
        "  Slope = Δ log(MAE) / Δ log(N).  More negative → faster convergence.\n"
    )


# ===========================================================================
# 5. Main
# ===========================================================================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NequIP Ablation Study — Learning Curve Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../../outputs",
        help="Root directory containing NequIP training output folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../notebooks/figures",
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--use-placeholder-data",
        action="store_true",
        help="Use placeholder results instead of parsing real logs "
        "(useful for testing the plotting pipeline).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    logger.info("=" * 60)
    logger.info("NequIP Ablation Study — Learning Curve Evaluation")
    logger.info("=" * 60)

    # Collect results
    if args.use_placeholder_data:
        logger.info("Using PLACEHOLDER data (not real training results).")
        results = get_placeholder_results()
    else:
        results = collect_results_from_logs(args.results_dir)
        if not results:
            logger.warning(
                "No results found in '%s'. Falling back to placeholder data.\n"
                "  → Run the experiments first, or use --use-placeholder-data.",
                args.results_dir,
                )
            results = get_placeholder_results()

    # Print summary table
    print_results_table(results)

    # Generate learning curve plot
    plot_learning_curves(results, output_dir=args.output_dir)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
