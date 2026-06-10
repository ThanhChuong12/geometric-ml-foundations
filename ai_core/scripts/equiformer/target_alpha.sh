#!/bin/bash
# =============================================================================
# Equiformer — Train on QM9 Target: Alpha (Isotropic Polarizability)
# =============================================================================
#
# This script trains the Equiformer model on the QM9 dataset for the
# isotropic polarizability (alpha) prediction task.
#
# QM9 Target Index Mapping (Equiformer convention):
#   0: mu         — Dipole moment (D)
#   1: alpha      — Isotropic polarizability (Bohr^3)  ← THIS SCRIPT
#   2: HOMO       — Highest occupied molecular orbital energy (Ha)
#   3: LUMO       — Lowest unoccupied molecular orbital energy (Ha)
#   4: gap        — HOMO-LUMO gap (Ha)
#   5: R2         — Electronic spatial extent (Bohr^2)
#   6: ZPVE       — Zero point vibrational energy (Ha)
#   7: U0         — Internal energy at 0K (Ha)
#   8: U          — Internal energy at 298.15K (Ha)
#   9: H          — Enthalpy at 298.15K (Ha)
#  10: G          — Free energy at 298.15K (Ha)
#  11: Cv         — Heat capacity at 298.15K (cal/mol·K)
#
# NOTE: QM9 dataset auto-downloads on the first run. The dataset will be
#       cached at the path specified by --data-path. Subsequent runs will
#       reuse the cached data without re-downloading.
#
# Model Variants:
#   1. graph_attention_transformer_nonlinear_l2  (default, below)
#      — Uses nonlinear messages with SE(3)/E(3) equivariant attention.
#      — Higher expressivity; generally more accurate.
#
#   2. graph_attention_transformer_l2  (commented-out alternative)
#      — Uses linear messages with dot-product attention.
#      — Faster training; useful for ablation / comparison.
#
# Usage:
#   bash scripts/equiformer/target_alpha.sh
#
# =============================================================================

# --- Environment Setup (adjust for your system) ---
# Uncomment the following lines if running on a cluster with module system:
# source /etc/profile
# module load anaconda/2021a
# export PYTHONNOUSERSITE=True    # prevent using packages from base env
# source activate th102_cu113_tgconda

# =============================================================================
# Option 1: Nonlinear Messages + SE(3) Attention (Recommended — Higher Accuracy)
# =============================================================================
python main_qm9.py \
    --output-dir 'models/qm9/equiformer/se_l2/target@alpha/' \
    --model-name 'graph_attention_transformer_nonlinear_l2' \
    --input-irreps '5x0e' \
    --target 1 \
    --data-path 'datasets/qm9' \
    --feature-type 'one_hot' \
    --batch-size 128 \
    --radius 5.0 \
    --num-basis 128 \
    --drop-path 0.0 \
    --weight-decay 5e-3 \
    --lr 5e-4 \
    --min-lr 1e-6 \
    --no-model-ema \
    --no-amp

# =============================================================================
# Option 2: Linear Messages + Dot Product Attention (Alternative — Faster)
# =============================================================================
# Uncomment the block below and comment out Option 1 above to switch to the
# DP (Dot Product) Equiformer variant. This uses linear message passing and
# dot-product attention, which is computationally cheaper but may sacrifice
# some accuracy compared to the nonlinear variant.
#
# python main_qm9.py \
#     --output-dir 'models/qm9/dp_equiformer/se_l2/target@alpha/' \
#     --model-name 'graph_attention_transformer_l2' \
#     --input-irreps '5x0e' \
#     --target 1 \
#     --data-path 'datasets/qm9' \
#     --feature-type 'one_hot' \
#     --batch-size 128 \
#     --radius 5.0 \
#     --num-basis 128 \
#     --drop-path 0.0 \
#     --weight-decay 5e-3 \
#     --lr 5e-4 \
#     --min-lr 1e-6 \
#     --no-model-ema \
#     --no-amp
