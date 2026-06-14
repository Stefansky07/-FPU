#!/usr/bin/env python
"""
Main script for running benchmarks and validation.

Usage:
    python run_benchmark.py --mode benchmark --output results/
    python run_benchmark.py --mode validate
    python run_benchmark.py --mode both --output results/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="FPU Benchmark & Validation")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "validate", "both"],
        default="both",
        help="Run mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        choices=["tiny_cnn", "tiny_mlp", "resnet18"],
        default=None,
        help="Specific model to benchmark (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
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

    # Import FPU
    from fpu.validate import run_validation_suite
    from fpu.benchmark import run_full_benchmark, benchmark_scaling

    # Run validation
    if args.mode in ["validate", "both"]:
        print("="*60)
        print("VALIDATION")
        print("="*60)
        passed = run_validation_suite(verbose=args.verbose)
        if not passed:
            print("\nVALIDATION FAILED!")
            sys.exit(1)
        print("\nVALIDATION PASSED!")

    # Run benchmarks
    if args.mode in ["benchmark", "both"]:
        print("\n" + "="*60)
        print("BENCHMARK")
        print("="*60)

        if args.model:
            # Single model benchmark
            from fpu.validate import create_test_state_dict
            from fpu.benchmark import benchmark_kernel
            from fpu.types import FusedUpdateConfig

            state_dict = create_test_state_dict(args.model)
            num_params = sum(t.numel() for t in state_dict.values())
            bundle_count = (num_params + 4095) // 4096

            config = FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.1)
            results = benchmark_kernel(
                state_dict, config, bundle_count,
                num_warmup=10, num_iterations=100,
                backend="triton", verbose=True,
            )

            # Save results
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"benchmark_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_dir / filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_dir / filename}")

        else:
            # Full benchmark suite
            results = run_full_benchmark(output_dir=args.output, verbose=args.verbose)

    print("\nDone!")


if __name__ == "__main__":
    main()
