#!/bin/bash
# =============================================================================
# NequIP Ablation Study — Run All Experiments
# =============================================================================
#
# Orchestrates the full 2×2 ablation grid:
#   1. baseline_l0_100   — Invariant GNN, 100 molecules
#   2. baseline_l0_1000  — Invariant GNN, 1000 molecules
#   3. nequip_l1_100     — Equivariant GNN, 100 molecules
#   4. nequip_l1_1000    — Equivariant GNN, 1000 molecules
#
# Prerequisites:
#   - QM9 subsets prepared: python scripts/prepare_qm9_subsets.py
#   - NequIP installed: pip install nequip
#   - Configs in ai_core/configs/
#
# Usage:
#   cd ai_core
#   bash scripts/run_all_experiments.sh
#
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_CORE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${AI_CORE_DIR}/configs"
OUTPUT_DIR="${AI_CORE_DIR}/outputs"

# List of experiment configs (order matters for sequential execution)
EXPERIMENTS=(
    "baseline_l0_100"
    "baseline_l0_1000"
    "nequip_l1_100"
    "nequip_l1_1000"
)

# ---------------------------------------------------------------------------
# Step 0: Data Preparation (if not already done)
# ---------------------------------------------------------------------------
DATA_DIR="${AI_CORE_DIR}/data/qm9"
if [ ! -f "${DATA_DIR}/qm9_subset_100.extxyz" ] || [ ! -f "${DATA_DIR}/qm9_subset_1000.extxyz" ]; then
    echo "============================================================"
    echo "  Step 0: Preparing QM9 Subsets"
    echo "============================================================"
    python "${SCRIPT_DIR}/prepare_qm9_subsets.py" \
        --root "${AI_CORE_DIR}/data/qm9_raw" \
        --output-dir "${DATA_DIR}" \
        --seed 42 \
        --subset-sizes 100 1000
    echo ""
else
    echo "QM9 subsets already exist in ${DATA_DIR}. Skipping data preparation."
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 1: Run NequIP Training for Each Configuration
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  NequIP Ablation Study — Starting All Experiments"
echo "============================================================"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Configs:          ${CONFIG_DIR}"
echo "  Experiments:      ${EXPERIMENTS[*]}"
echo "============================================================"
echo ""

FAILED=()
SUCCEEDED=()

for exp_name in "${EXPERIMENTS[@]}"; do
    config_file="${CONFIG_DIR}/${exp_name}.yaml"
    exp_output="${OUTPUT_DIR}/${exp_name}"

    echo "------------------------------------------------------------"
    echo "  Running: ${exp_name}"
    echo "  Config:  ${config_file}"
    echo "  Output:  ${exp_output}"
    echo "------------------------------------------------------------"

    if [ ! -f "$config_file" ]; then
        echo "  ERROR: Config file not found: ${config_file}"
        FAILED+=("$exp_name")
        continue
    fi

    # Create output directory
    mkdir -p "$exp_output"

    # Run nequip-train with Hydra
    # The --config-path and --config-name flags tell Hydra where to find the config
    if nequip-train \
        --config-path "${CONFIG_DIR}" \
        --config-name "${exp_name}" \
        hydra.run.dir="${exp_output}" 2>&1 | tee "${exp_output}/train.log"; then
        echo "  ✓ ${exp_name} completed successfully."
        SUCCEEDED+=("$exp_name")
    else
        echo "  ✗ ${exp_name} FAILED. Check log: ${exp_output}/train.log"
        FAILED+=("$exp_name")
    fi

    echo ""
done

# ---------------------------------------------------------------------------
# Step 2: Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Experiment Summary"
echo "============================================================"
echo "  Succeeded: ${#SUCCEEDED[@]} / ${#EXPERIMENTS[@]}"
for exp in "${SUCCEEDED[@]}"; do
    echo "    ✓ ${exp}"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "  Failed: ${#FAILED[@]} / ${#EXPERIMENTS[@]}"
    for exp in "${FAILED[@]}"; do
        echo "    ✗ ${exp}"
    done
    echo ""
    echo "  Check individual train.log files in ${OUTPUT_DIR}/ for details."
fi

echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Generate Learning Curves (optional, only if all succeeded)
# ---------------------------------------------------------------------------
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All experiments completed. Generating learning curves..."
    python "${AI_CORE_DIR}/notebooks/03_nequip_learning_curve.py" \
        --results-dir "${OUTPUT_DIR}" \
        --output-dir "${AI_CORE_DIR}/notebooks/figures"
    echo "Learning curve plots saved to: ${AI_CORE_DIR}/notebooks/figures/"
else
    echo "Skipping learning curve generation due to failed experiments."
    echo "You can still generate plots with placeholder data:"
    echo "  python notebooks/03_nequip_learning_curve.py --use-placeholder-data"
fi
