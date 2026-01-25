#!/bin/bash
# Wait for experiment 3 to finish, then start experiment 4
#
# Usage: nohup ./scripts/run_exp4_after_exp3.sh &

set -e

EXP3_PID=${1:-4510}

echo "Waiting for experiment 3 (PID $EXP3_PID) to finish..."
echo "Started waiting at: $(date)"

# Wait for the process to finish
while kill -0 $EXP3_PID 2>/dev/null; do
    sleep 60  # Check every minute
done

echo "Experiment 3 finished at: $(date)"
echo "Starting experiment 4 in 60 seconds..."
sleep 60

# Start experiment 4
cd "$(dirname "$0")/.."
bash scripts/run_continual_exp4_l2_init.sh
