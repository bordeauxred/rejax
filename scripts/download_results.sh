#!/bin/bash
# Download benchmark results from cluster
# Run this ON YOUR LOCAL MACHINE, not on cluster

# Usage: bash scripts/download_results.sh user@cluster-ip

if [ -z "$1" ]; then
    echo "Usage: bash scripts/download_results.sh user@cluster-ip"
    echo "Example: bash scripts/download_results.sh ubuntu@209-20-156-252"
    exit 1
fi

REMOTE=$1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_DIR="results_ortho_scaling_${TIMESTAMP}"

mkdir -p "$LOCAL_DIR"

echo "Downloading benchmark_results/ ..."
scp -r "${REMOTE}:~/rejax/benchmark_results/" "${LOCAL_DIR}/"

echo "Downloading wandb/ (optional, already synced to cloud)..."
scp -r "${REMOTE}:~/rejax/wandb/" "${LOCAL_DIR}/" 2>/dev/null || echo "No local wandb dir or skipped"

echo ""
echo "Results downloaded to: ${LOCAL_DIR}/"
echo "JSON files: ${LOCAL_DIR}/benchmark_results/"
ls -la "${LOCAL_DIR}/benchmark_results/"
