#!/bin/bash
# Run Octax paper's OFFICIAL PPO to get ground truth returns
# NOTE: Octax depends on rejax! They use our PPO.

set -e

cd /tmp

rm -rf octax
git clone https://github.com/riiswa/octax.git
cd octax

# Install training deps (rejax, gymnax, hydra)
pip install hydra-core gymnax
pip install .

echo "=============================================="
echo "Running OFFICIAL Octax PPO (ground truth)"
echo "NOTE: Octax uses rejax PPO internally!"
echo "=============================================="

echo ""
echo "=== BRIX (2 seeds) ==="
python train.py env=brix num_seeds=2 total_timesteps=5000000

echo ""
echo "=== TETRIS (2 seeds) ==="
python train.py env=tetris num_seeds=2 total_timesteps=5000000

echo ""
echo "DONE - Results in /tmp/octax/results/"
