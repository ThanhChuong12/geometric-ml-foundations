set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_CORE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${AI_CORE_DIR}/configs"
OUTPUT_DIR="${AI_CORE_DIR}/outputs"

# Add local virtual environment to PATH if present
PROJECT_ROOT="$(dirname "${AI_CORE_DIR}")"
VENV_BIN="${PROJECT_ROOT}/nequip-env/bin"
if [ -d "${VENV_BIN}" ]; then
    export PATH="${VENV_BIN}:${PATH}"
fi

# Detect CUDA environment issues and fall back to CPU if necessary
if python -c "import torch; assert torch.cuda.is_available(); torch.cuda.get_device_properties(0)" >/dev/null 2>&1; then
    echo "CUDA is functional. Training will proceed on GPU."
else
    # Check if there was a warning/error during import/availability check
    if python -c "import torch; torch.cuda.is_available()" 2>&1 | grep -i -E "warning|error|fail|too old" >/dev/null; then
        echo "[WARNING] CUDA driver initialization failed or warning detected. Forcing CPU training via export CUDA_VISIBLE_DEVICES=''"
    else
        echo "CUDA is not available. Training will proceed on CPU."
    fi
    export CUDA_VISIBLE_DEVICES=""
fi

# List of experiment configs (order matters for sequential execution)
EXPERIMENTS=(
    "baseline_l0_100"
    "baseline_l0_1000"
    "nequip_l1_100"
    "nequip_l1_1000"
)

# Data Preparation
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

# Run NequIP Training for Each Configuration
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

# Summary
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

# Generate Learning Curves (optional, only if all succeeded)
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
