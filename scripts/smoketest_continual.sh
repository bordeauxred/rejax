#!/bin/bash
# Smoke test for continual learning benchmark
# Run on remote GPU with wandb logging

set -e

echo "=============================================="
echo "Continual Learning Benchmark - Smoke Test"
echo "=============================================="

# Test 1: Network use_bias verification
echo ""
echo "Test 1: Verifying network use_bias parameter..."
uv run python -c "
import jax
import jax.numpy as jnp
from rejax.networks import MLP, VNetwork, DiscretePolicy, DiscreteQNetwork

# Test MLP with use_bias=False
mlp = MLP((256, 256, 256, 256), jax.nn.tanh, use_bias=False)
params = mlp.init(jax.random.PRNGKey(0), jnp.zeros((1, 100)))
flat = jax.tree_util.tree_leaves_with_path(params)
bias_found = any('bias' in str(k) for k, _ in flat)
assert not bias_found, 'MLP use_bias=False should have no bias'

# Test with use_bias=True
mlp_bias = MLP((256, 256, 256, 256), jax.nn.tanh, use_bias=True)
params_bias = mlp_bias.init(jax.random.PRNGKey(0), jnp.zeros((1, 100)))
flat_bias = jax.tree_util.tree_leaves_with_path(params_bias)
bias_found_bias = any('bias' in str(k) for k, _ in flat_bias)
assert bias_found_bias, 'MLP use_bias=True should have bias'

# Test VNetwork with 4 layers
vnet = VNetwork((256, 256, 256, 256), jax.nn.tanh, use_bias=False)
params_v = vnet.init(jax.random.PRNGKey(0), jnp.zeros((1, 100)))
flat_v = jax.tree_util.tree_leaves_with_path(params_v)
bias_found_v = any('bias' in str(k) for k, _ in flat_v)
assert not bias_found_v, 'VNetwork use_bias=False should have no bias'

print('  [PASS] Network use_bias parameter works correctly')
print('  [PASS] 4-layer networks initialize correctly')
"

# Test 2: Run smoke test with all configs
echo ""
echo "Test 2: Running continual benchmark smoke test..."
echo "  - 100k steps per game"
echo "  - 1 cycle"
echo "  - 2 seeds"
echo "  - All 3 configs: baseline, ortho_adamo, ortho_adamo_lyle_lr"
echo "  - With wandb logging"
echo ""

uv run python scripts/bench_continual.py \
    --steps-per-game 50000 \
    --num-cycles 2 \
    --num-seeds 2 \
    --configs baseline ortho_adamo ortho_adamo_lyle_lr \
    --use-wandb \
    --wandb-project rejax-ppo-continual-minatar-smoketest

echo ""
echo "=============================================="
echo "Smoke test completed successfully!"
echo "=============================================="
