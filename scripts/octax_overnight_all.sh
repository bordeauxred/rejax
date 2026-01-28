#!/bin/bash
# Full overnight run: All games + key games comparison
# Time: ~4.5 hours with 2 seeds
# 48 runs total

set -e

echo "=== Octax Full Overnight Run ==="
echo "Started at $(date)"
echo "Total runs: 48"
echo "Estimated time: ~4.5 hours"
echo ""

./scripts/octax_overnight_part1.sh
./scripts/octax_overnight_part2.sh

echo ""
echo "=== ALL DONE at $(date) ==="
