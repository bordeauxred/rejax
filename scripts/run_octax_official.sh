#!/bin/bash
# Run Octax paper's OFFICIAL PPO to get ground truth returns
# This tells us what returns we SHOULD expect

set -e

cd /tmp

# Clone if not exists
if [ ! -d "octax" ]; then
    git clone https://github.com/riiswa/octax.git
fi

cd octax
pip install . --quiet

echo "=============================================="
echo "Running OFFICIAL Octax PPO (ground truth)"
echo "=============================================="

# Run 3 games with 2 seeds each to get reliable baselines
# Paper default: 5M steps, 512 envs, 4 epochs, 32 minibatches

echo ""
echo "=== BRIX (2 seeds) ==="
python train.py env=brix num_seeds=2 total_timesteps=5000000

echo ""
echo "=== TETRIS (2 seeds) ==="
python train.py env=tetris num_seeds=2 total_timesteps=5000000

echo ""
echo "=== TANK (2 seeds) ==="
python train.py env=tank num_seeds=2 total_timesteps=5000000

echo ""
echo "=============================================="
echo "DONE - Check results/ folder for learning curves"
echo "=============================================="
