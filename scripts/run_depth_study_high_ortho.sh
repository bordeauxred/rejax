#!/bin/bash
# Depth Study: High Ortho Coefficient Comparison
#
# Tests whether stronger orthogonality enforcement helps deep networks.
# Compares AdaMO and Scale-AdaMO with high ortho_coeff (0.5, 1.0)
#
# Depths:        16, 32, 64 layers (focus on where AdaMO failed)
# Methods:       adamo_0.5, adamo_1.0, scale_adamo_0.5, scale_adamo_1.0
# Seeds:         2
#
# Logs to same WandB project as original depth study for easy comparison.
#
# NOTE: Uses 45% GPU memory to allow running 2 experiments in parallel
#
# Usage:
#   ./scripts/run_depth_study_high_ortho.sh                    # Full run (10M steps)
#   ./scripts/run_depth_study_high_ortho.sh 1000000 1          # Quick test (1M, 1 seed)

set -e

# Limit GPU memory to 45% (allows running 2 in parallel)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

TIMESTEPS=${1:-10000000}
NUM_SEEDS=${2:-2}
EVAL_FREQ=${3:-500000}
NUM_ENVS=${4:-2048}

# Use same folder as original depth study for easy comparison
OUTPUT_DIR="results/depth_study_breakout_ppo"
WANDB_PROJECT="rejax-depth-study-ppo-mlp"  # Same project as original depth study

echo "=============================================================="
echo "DEPTH STUDY: High Ortho Coefficient Comparison"
echo "=============================================================="
echo "GPU memory: 45% (XLA_PYTHON_CLIENT_MEM_FRACTION=0.45)"
echo "Timesteps: $TIMESTEPS"
echo "Seeds: $NUM_SEEDS"
echo "Eval freq: $EVAL_FREQ"
echo "Num envs: $NUM_ENVS"
echo "Output: $OUTPUT_DIR"
echo "W&B Project: $WANDB_PROJECT (same as original depth study)"
echo "=============================================================="
echo "Depths: 16, 32, 64"
echo "Configs (4 total, skipping baseline and adamo_0.1 - already ran):"
echo "  1. adamo_0.5         - GroupSort, ortho_coeff=0.5"
echo "  2. adamo_1.0         - GroupSort, ortho_coeff=1.0"
echo "  3. scale_adamo_0.5   - GroupSort, ortho_coeff=0.5, learnable scale"
echo "  4. scale_adamo_1.0   - GroupSort, ortho_coeff=1.0, learnable scale"
echo "=============================================================="

cd "$(dirname "$0")/.."

mkdir -p "$OUTPUT_DIR"

# Log start time
echo "Started at: $(date)" | tee "$OUTPUT_DIR/run.log"

uv run python scripts/bench_depth_study_high_ortho.py \
    --timesteps $TIMESTEPS \
    --num-seeds $NUM_SEEDS \
    --eval-freq $EVAL_FREQ \
    --num-envs $NUM_ENVS \
    --depths 16 32 64 \
    --output-dir "$OUTPUT_DIR" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

echo ""
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "Results saved to: $OUTPUT_DIR"
