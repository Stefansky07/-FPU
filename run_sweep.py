#!/usr/bin/env python
"""
Parameter sweep script for FPU operator benchmarking.

Crosses: slot_capacity × quant_bits × reuse_buffers × client_weight × physical_quantized_output
Outputs JSON + CSV to results/ directory.

Usage:
    python run_sweep.py --param-count 1000000
    python run_sweep.py --param-count 3000000000 --iterations 10 --warmup 5
    python run_sweep.py --param-count 1000000 --lock-clocks
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="FPU Parameter Sweep")
    parser.add_argument(
        "--param-count",
        type=int,
        required=True,
        help="Number of synthetic parameters",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations (default: 20)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Measurement iterations (default: 100)",
    )
    parser.add_argument(
        "--backend",
        choices=["triton", "triton_v2", "torch_ref"],
        default="triton_v2",
        help="Backend (default: triton_v2)",
    )
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=1.0,
        help="Clipping norm (default: 1.0)",
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=0.0,
        help="DP noise multiplier (default: 0.0)",
    )
    parser.add_argument(
        "--lock-clocks",
        action="store_true",
        help="Attempt GPU clock locking (needs admin)",
    )
    parser.add_argument(
        "--slot-capacities",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
        help="Slot capacities to sweep (default: 1024 2048 4096 8192)",
    )
    parser.add_argument(
        "--quant-bits",
        type=int,
        nargs="+",
        default=[4, 8],
        help="Quant bits to sweep (default: 4 8)",
    )
    parser.add_argument(
        "--client-weights",
        type=float,
        nargs="+",
        default=[1.0, 0.5],
        help="Client weights to sweep (default: 1.0 0.5)",
    )
    args = parser.parse_args()

    # Check dependencies
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    try:
        import triton
        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("ERROR: Triton not installed")
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    from fpu.benchmark import (
        benchmark_param_sweep,
        save_sweep_results,
        get_gpu_environment,
    )

    # Print GPU environment
    gpu_env = get_gpu_environment()
    print("GPU Environment:")
    for k, v in gpu_env.items():
        print(f"  {k}: {v}")
    print()

    # Run sweep
    results = benchmark_param_sweep(
        num_params=args.param_count,
        slot_capacities=args.slot_capacities,
        quant_bits_list=args.quant_bits,
        reuse_buffers_list=[True, False],
        client_weights=args.client_weights,
        physical_quant_list=[True, False],
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        backend=args.backend,
        lock_clocks=args.lock_clocks,
        verbose=True,
    )

    # Save
    prefix = f"sweep_{args.param_count}"
    json_path, csv_path = save_sweep_results(results, args.output, prefix=prefix)
    print(f"\n{'='*60}")
    print(f"Sweep complete: {len(results)} configurations")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = (
        f"{'slot':>6} {'qb':>3} {'reuse':>5} {'cw':>4} {'phys':>5} "
        f"{'mean_ms':>8} {'p90_ms':>8} {'cv%':>5} "
        f"{'out_dtype':>10} {'actual_MB':>10} {'logic_MB':>10} {'BW_GB/s':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            continue
        print(
            f"{r.get('slot_capacity', r.get('sweep_slot_capacity', '')):>6} "
            f"{r.get('quant_bits', r.get('sweep_quant_bits', '')):>3} "
            f"{str(r.get('reuse_buffers', r.get('sweep_reuse_buffers', ''))):>5} "
            f"{r.get('client_weight', r.get('sweep_client_weight', '')):>4} "
            f"{str(r.get('physical_quantized_output', r.get('sweep_physical_quant', ''))):>5} "
            f"{r.get('cuda_mean_ms', 0):>8.3f} "
            f"{r.get('cuda_p90_ms', 0):>8.3f} "
            f"{r.get('cuda_cv_pct', 0):>5.1f} "
            f"{r.get('output_dtype', ''):>10} "
            f"{r.get('actual_output_mb', 0):>10.1f} "
            f"{r.get('logical_output_mb', 0):>10.1f} "
            f"{r.get('bandwidth_gbps', 0):>8.1f}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
