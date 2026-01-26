#!/bin/bash
# Experiment 6: Scale-AdaMO (Per-Layer Learnable Scaling) with Permutations
#
# Goal: Test Scale-AdaMO vs AdaMO vs Baseline for continual plasticity
#
# Scale-AdaMO addresses the "scaling bottleneck" in orthonormal networks:
# - Hidden layers are orthonormalized (W@W.T ≈ I), preserving signal magnitude
# - Output layer is the ONLY layer that can scale outputs
# - Different games have different return magnitudes (2-150x range)
# - Per-layer learnable α allows scaling to be distributed across layers
#   while maintaining orthonormality benefits (stable gradients, feature preservation)
#
# Network:      64-64-64-64 MLP (small, same as exp5)
# Steps/game:   15M (matching other 64x4 experiments)
# Cycles:       20 (longer to see degradation)
# Methods:      Baseline, AdaMO, Scale-AdaMO
# Seeds:        2
# Games:        4 (excludes Seaquest - reward scale outlier)
# Permutation:  YES - all 10 channels shuffled + game order
#
# HYPOTHESIS: Scale-AdaMO will outperform standard AdaMO by allowing the network
#             to adapt scaling across layers without losing orthonormality benefits.
#             This should be especially visible in continual learning where the
#             network must adapt to different reward scales across games.
#
# RUNTIME: 15M x 4 games x 20 cycles x 3 methods x 2 seeds = 7.2B steps
#          -> ~12h on H100 (small network is fast)
#
# Usage:
#   ./scripts/run_continual_exp6_scale_adamo.sh                    # Full run
#   ./scripts/run_continual_exp6_scale_adamo.sh 1000000 2 1        # Quick test

set -e

STEPS_PER_GAME=${1:-15000000}
NUM_CYCLES=${2:-20}
NUM_SEEDS=${3:-2}
NUM_ENVS=${4:-2048}
EVAL_FREQ=${5:-500000}

EXPERIMENT_NAME="continual_exp6_scale_adamo_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
WANDB_PROJECT="continual_minatar_small_permuted"  # Same as exp5 and other 64x4 experiments

echo "=============================================================="
echo "EXPERIMENT 6: SCALE-ADAMO (PER-LAYER LEARNABLE SCALING)"
echo "=============================================================="
echo "Steps per game: $STEPS_PER_GAME"
echo "Cycles: $NUM_CYCLES"
echo "Seeds: $NUM_SEEDS"
echo "Envs: $NUM_ENVS"
echo "Eval freq: $EVAL_FREQ"
echo "Output: $OUTPUT_DIR"
echo "W&B Project: $WANDB_PROJECT"
echo "=============================================================="
echo "Configs:"
echo "  1. mlp_baseline_small - MLP 64x4, ReLU, no regularization"
echo "  2. mlp_adamo_small    - MLP 64x4, GroupSort, AdaMO ortho"
echo "  3. mlp_scale_adamo_small - MLP 64x4, GroupSort, AdaMO + learnable α"
echo ""
echo "Permutations ENABLED:"
echo "  - Channel permutation: all 10 channels shuffled per game/cycle"
echo "  - Game order shuffle: random per cycle"
echo ""
echo "Excluded: Seaquest-MinAtar (reward scale outlier)"
echo "Games: Breakout, Asterix, SpaceInvaders, Freeway"
echo ""
echo "Scale-AdaMO features:"
echo "  - Per-layer learnable α initialized to 1.0"
echo "  - log(α)² regularization (coeff=0.01) to keep α near 1"
echo "  - Allows network to learn appropriate scaling per layer"
echo "  - Maintains orthonormality (W@W.T ≈ I) while having flexible output scale"
echo "=============================================================="

cd "$(dirname "$0")/.."

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Log start time
echo "Started at: $(date)" | tee "$OUTPUT_DIR/run.log"

uv run python scripts/bench_continual.py \
    --steps-per-game $STEPS_PER_GAME \
    --num-cycles $NUM_CYCLES \
    --num-seeds $NUM_SEEDS \
    --num-envs $NUM_ENVS \
    --eval-freq $EVAL_FREQ \
    --configs mlp_scale_adamo_small \
    --permute-channels \
    --random-game-order \
    --exclude-games Seaquest-MinAtar \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

echo ""
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "Results saved to: $OUTPUT_DIR"
