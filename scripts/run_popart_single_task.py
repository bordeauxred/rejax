#!/usr/bin/env python
"""
Single-task PopArt PPO benchmark on Octax games.

Tests: 7 games × 3 architectures × 2 seeds = 42 runs

Usage:
    # Sequential (single GPU)
    python scripts/run_popart_single_task.py

    # Parallel on multiple GPUs (recommended)
    python scripts/run_popart_single_task.py --parallel 4

    # Run single job (for job arrays / SLURM)
    python scripts/run_popart_single_task.py --job-id 0
    python scripts/run_popart_single_task.py --job-id 1
    ...

    # Generate SLURM array script
    python scripts/run_popart_single_task.py --slurm > submit.sh
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Configuration
GAMES = ["blinky", "brix", "pong", "tank", "flight_runner", "tetris", "worm"]
ARCHS = ["256x1", "64x4", "256x4"]
NUM_SEEDS = 2
DEFAULT_STEPS = 5_000_000
DEFAULT_NUM_ENVS = 512
DEFAULT_POPART_BETA = 0.0001

# Hardware presets (num_envs optimized for GPU memory/throughput)
PRESETS = {
    "h100": {"num_envs": 2048, "desc": "H100 40-80GB"},
    "a100": {"num_envs": 1024, "desc": "A100 40GB"},
    "rtx4090": {"num_envs": 512, "desc": "RTX 4090 24GB"},
    "cpu": {"num_envs": 64, "desc": "CPU only"},
}


def get_all_jobs():
    """Generate all (game, arch) combinations."""
    jobs = []
    for game in GAMES:
        for arch in ARCHS:
            jobs.append((game, arch))
    return jobs


def run_single_job(
    game: str,
    arch: str,
    steps: int,
    num_envs: int,
    num_seeds: int,
    popart_beta: float,
    output_dir: str,
    eval_freq: int = None,
    gpu_id: int = None,
):
    """Run a single (game, arch) experiment with full learning curves."""
    output_file = f"{output_dir}/{game}_{arch}.json"

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "scripts/bench_octax_popart.py",
        "--game", game,
        "--arch", arch,
        "--steps", str(steps),
        "--num-envs", str(num_envs),
        "--num-seeds", str(num_seeds),
        "--popart-beta", str(popart_beta),
        "--output", output_file,
    ]

    if eval_freq:
        cmd.extend(["--eval-freq", str(eval_freq)])

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed = time.time() - start

        with open(output_file) as f:
            data = json.load(f)

        return {
            "game": game,
            "arch": arch,
            "success": True,
            "elapsed": elapsed,
            "result": data,
        }

    except subprocess.CalledProcessError as e:
        return {
            "game": game,
            "arch": arch,
            "success": False,
            "error": e.stderr,
        }


def run_sequential(args):
    """Run all jobs sequentially."""
    jobs = get_all_jobs()
    results = {}

    print(f"Running {len(jobs)} jobs sequentially...")
    print(f"Games: {GAMES}")
    print(f"Archs: {ARCHS}")
    print(f"Seeds: {args.num_seeds}")
    print()

    for i, (game, arch) in enumerate(jobs):
        print(f"[{i+1}/{len(jobs)}] {game} + {arch}...", flush=True)

        result = run_single_job(
            game, arch,
            args.steps, args.num_envs, args.num_seeds, args.popart_beta,
            args.output_dir, args.eval_freq,
        )

        if result["success"]:
            r = result["result"]
            print(f"  Return: {r['final_return_mean']:.1f} +/- {r['final_return_std']:.1f} "
                  f"({result['elapsed']:.1f}s)")
        else:
            print(f"  FAILED: {result.get('error', 'unknown')[:100]}")

        if game not in results:
            results[game] = {}
        results[game][arch] = result

    return results


def run_parallel(args):
    """Run jobs in parallel across multiple GPUs."""
    jobs = get_all_jobs()
    num_gpus = args.parallel
    results = {}

    print(f"Running {len(jobs)} jobs on {num_gpus} GPUs in parallel...")
    print()

    def worker(job_idx):
        game, arch = jobs[job_idx]
        gpu_id = job_idx % num_gpus
        return run_single_job(
            game, arch,
            args.steps, args.num_envs, args.num_seeds, args.popart_beta,
            args.output_dir, args.eval_freq, gpu_id,
        )

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(worker, i): i for i in range(len(jobs))}

        for future in as_completed(futures):
            job_idx = futures[future]
            game, arch = jobs[job_idx]
            result = future.result()

            if result["success"]:
                r = result["result"]
                print(f"[{job_idx+1}/{len(jobs)}] {game} + {arch}: "
                      f"{r['final_return_mean']:.1f} ({result['elapsed']:.1f}s)")
            else:
                print(f"[{job_idx+1}/{len(jobs)}] {game} + {arch}: FAILED")

            if game not in results:
                results[game] = {}
            results[game][arch] = result

    return results


def run_single_job_by_id(args):
    """Run a single job by its ID (for SLURM arrays)."""
    jobs = get_all_jobs()
    if args.job_id >= len(jobs):
        print(f"Job ID {args.job_id} out of range (max: {len(jobs)-1})")
        sys.exit(1)

    game, arch = jobs[args.job_id]
    print(f"Job {args.job_id}: {game} + {arch}")

    result = run_single_job(
        game, arch,
        args.steps, args.num_envs, args.num_seeds, args.popart_beta,
        args.output_dir, args.eval_freq,
    )

    if result["success"]:
        r = result["result"]
        print(f"Return: {r['final_return_mean']:.1f} +/- {r['final_return_std']:.1f}")
        print(f"Learning curve points: {len(r['eval_steps'])}")
    else:
        print(f"FAILED: {result.get('error', 'unknown')}")
        sys.exit(1)


def generate_slurm_script(args):
    """Generate SLURM array job script."""
    jobs = get_all_jobs()
    eval_freq_arg = f"    --eval-freq {args.eval_freq} \\\n" if args.eval_freq else ""
    script = f"""#!/bin/bash
#SBATCH --job-name=popart_single
#SBATCH --array=0-{len(jobs)-1}
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/popart_%A_%a.out

# PopArt single-task benchmark
# {len(jobs)} jobs: {len(GAMES)} games × {len(ARCHS)} archs

cd $SLURM_SUBMIT_DIR
mkdir -p logs {args.output_dir}

python scripts/run_popart_single_task.py \\
    --job-id $SLURM_ARRAY_TASK_ID \\
    --steps {args.steps} \\
    --num-envs {args.num_envs} \\
    --num-seeds {args.num_seeds} \\
    --popart-beta {args.popart_beta} \\
{eval_freq_arg}    --output-dir {args.output_dir}
"""
    print(script)


def print_summary(results, output_file):
    """Print and save summary."""
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Print table
    print(f"\n{'Game':<15} {'256x1':>12} {'64x4':>12} {'256x4':>12}")
    print("-" * 55)
    for game in GAMES:
        row = f"{game:<15}"
        for arch in ARCHS:
            if game in results and arch in results[game]:
                r = results[game][arch]
                if r["success"]:
                    row += f" {r['result']['final_return_mean']:>12.1f}"
                else:
                    row += f" {'ERR':>12}"
            else:
                row += f" {'-':>12}"
        print(row)

    # Save
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="PopArt single-task benchmark")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--num-envs", type=int, default=None,
                        help=f"Parallel envs (default: {DEFAULT_NUM_ENVS}, see --preset)")
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--popart-beta", type=float, default=DEFAULT_POPART_BETA)
    parser.add_argument("--output-dir", type=str, default="results/popart_single_task")
    parser.add_argument("--eval-freq", type=int, default=None,
                        help="Eval frequency for learning curves (default: steps/20)")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()),
                        help="Hardware preset: " + ", ".join(f"{k}={v['num_envs']}" for k,v in PRESETS.items()))

    # Execution mode
    parser.add_argument("--parallel", type=int, default=None,
                        help="Number of GPUs for parallel execution")
    parser.add_argument("--job-id", type=int, default=None,
                        help="Single job ID (for SLURM arrays)")
    parser.add_argument("--slurm", action="store_true",
                        help="Generate SLURM array script")
    parser.add_argument("--list-jobs", action="store_true",
                        help="List all job IDs")

    args = parser.parse_args()

    # Apply preset
    if args.preset:
        preset = PRESETS[args.preset]
        if args.num_envs is None:
            args.num_envs = preset["num_envs"]
        print(f"Using preset '{args.preset}': {preset['desc']}, num_envs={args.num_envs}")
    if args.num_envs is None:
        args.num_envs = DEFAULT_NUM_ENVS

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.list_jobs:
        jobs = get_all_jobs()
        for i, (game, arch) in enumerate(jobs):
            print(f"{i:2d}: {game} + {arch}")
        return

    if args.slurm:
        generate_slurm_script(args)
        return

    if args.job_id is not None:
        run_single_job_by_id(args)
        return

    # Full run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_dir}/results_{timestamp}.json"

    if args.parallel:
        results = run_parallel(args)
    else:
        results = run_sequential(args)

    print_summary(results, output_file)


if __name__ == "__main__":
    main()
