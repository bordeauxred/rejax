#!/bin/bash
# Depth Study: PPO MLP Network Depth Ablation
#
# Goal: Study the effect of network depth on PPO performance with MLP
#
# Depths:        2, 4, 8, 16, 32, 64 layers (all 256-wide)
# Methods:       Baseline (ReLU) vs AdaMO (GroupSort + orthogonal optimizer)
# Game:          Breakout-MinAtar (Gymnax)
# Seeds:         3
#
# HYPOTHESIS: Deeper networks may suffer from optimization issues (vanishing gradients,
#             dead neurons). AdaMO's orthogonal regularization should help maintain
#             gradient flow in deeper networks.
#
# RUNTIME: 25M steps × 6 depths × 2 methods × 3 seeds = 900M steps
#          → ~4-6h on A100 (depends on depth, deeper = slower)
#
# Usage:
#   ./scripts/run_depth_study_ppo_mlp.sh                           # Full run (25M steps)
#   ./scripts/run_depth_study_ppo_mlp.sh 1000000 1                 # Quick test (1M, 1 seed)

set -e

TIMESTEPS=${1:-25000000}
NUM_SEEDS=${2:-3}
EVAL_FREQ=${3:-500000}
NUM_ENVS=${4:-2048}

EXPERIMENT_NAME="depth_study_ppo_mlp_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
WANDB_PROJECT="rejax-depth-study-ppo-mlp"

echo "=============================================================="
echo "DEPTH STUDY: PPO MLP Network Depth Ablation"
echo "=============================================================="
echo "Timesteps: $TIMESTEPS (25M default)"
echo "Seeds: $NUM_SEEDS"
echo "Eval freq: $EVAL_FREQ"
echo "Num envs: $NUM_ENVS"
echo "Output: $OUTPUT_DIR"
echo "W&B Project: $WANDB_PROJECT"
echo "=============================================================="
echo "Depths: 2, 4, 8, 16, 32, 64 layers (all 256-wide)"
echo "Methods:"
echo "  1. Baseline (ReLU, standard Adam)"
echo "  2. AdaMO (GroupSort, orthogonal optimizer)"
echo "Game: Breakout-MinAtar"
echo "=============================================================="

cd "$(dirname "$0")/.."

mkdir -p "$OUTPUT_DIR"

# Log start time
echo "Started at: $(date)" | tee "$OUTPUT_DIR/run.log"

uv run python scripts/bench_depth_study.py \
    --timesteps $TIMESTEPS \
    --num-seeds $NUM_SEEDS \
    --eval-freq $EVAL_FREQ \
    --num-envs $NUM_ENVS \
    --depths 2 4 8 16 32 64 \
    --game Breakout-MinAtar \
    --output-dir "$OUTPUT_DIR" \
    --plot \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

echo ""
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "Results saved to: $OUTPUT_DIR"
