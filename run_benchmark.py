#!/usr/bin/env python
"""
Main script for running benchmarks and validation.

Usage:
    python run_benchmark.py --mode benchmark --output results/
    python run_benchmark.py --mode validate
    python run_benchmark.py --mode both --output results/
    python run_benchmark.py --mode benchmark --synthetic-params 1000000 --physical-quant-output
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
        "--synthetic-params",
        type=int,
        default=None,
        help="Benchmark a synthetic flat update with this many parameters",
    )
    parser.add_argument(
        "--backend",
        choices=["triton", "triton_v2", "torch_ref"],
        default="triton_v2",
        help="Backend for single-model benchmark mode",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations for benchmark mode (default: 20)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Measurement iterations for benchmark mode (default: 100)",
    )
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=1.0,
        help="Clipping norm for single-model benchmark mode",
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=0.1,
        help="DP noise multiplier for single-model benchmark mode",
    )
    parser.add_argument(
        "--quant-bits",
        type=int,
        default=8,
        help="Quantization bits for single-model benchmark mode (0 disables quantization)",
    )
    parser.add_argument(
        "--client-weight",
        type=float,
        default=1.0,
        help="Client weight for single-model benchmark mode",
    )
    parser.add_argument(
        "--physical-quant-output",
        action="store_true",
        help="Enable physical int8 quantized output (requires --quant-bits 8)",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="Seed for DP noise generation",
    )
    parser.add_argument(
        "--stateless-noise",
        action="store_true",
        help="Generate DP noise inside Triton kernels with tl.randn(seed, offset)",
    )
    parser.add_argument(
        "--fixed-quant-scale",
        type=float,
        default=None,
        help="Use a fixed quantization scale and skip transformed abs-max stats",
    )
    parser.add_argument(
        "--fixed-clip-scale",
        type=float,
        default=None,
        help="Use a fixed clipping scale and skip L2 reduction (for graph/pack-only tests)",
    )
    parser.add_argument(
        "--calibration-json",
        type=str,
        default=None,
        help="Read quant_scale/clip_scale from an exact benchmark JSON when fixed scales are omitted",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Launch fused pack kernels in chunks of this many slots",
    )
    parser.add_argument(
        "--chunk-quant-scale",
        action="store_true",
        help="Compute a separate quantization scale per chunk",
    )
    parser.add_argument(
        "--pack-block-size",
        type=int,
        default=1024,
        help="Triton BLOCK_SIZE for fused pack kernels (default: 1024)",
    )
    parser.add_argument(
        "--reuse-buffers",
        action="store_true",
        help="Reuse triton_v2 output/noise buffers across benchmark iterations",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Replay graph-captured triton_v2 pack-only benchmark when requirements are met",
    )
    parser.add_argument(
        "--stream-output-dir",
        type=str,
        default=None,
        help="Stream fixed-scale physical int8 chunks to this directory instead of running timing loop",
    )
    parser.add_argument(
        "--stream-prefix",
        type=str,
        default="fpu_chunk",
        help="Filename prefix for --stream-output-dir chunks",
    )
    parser.add_argument(
        "--measure-output-quality",
        action="store_true",
        help="Compute extra int8 payload quality metrics such as saturation rate",
    )
    parser.add_argument(
        "--lock-clocks",
        action="store_true",
        help="Attempt to lock GPU clocks for stable benchmark (needs admin)",
    )
    parser.add_argument(
        "--slot-capacity",
        type=int,
        default=4096,
        help="Slot capacity (default: 4096)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )
    args = parser.parse_args()

    if args.calibration_json:
        calibration_path = Path(args.calibration_json)
        with open(calibration_path, "r", encoding="utf-8") as f:
            calibration = json.load(f)
        if args.fixed_quant_scale is None:
            if "quant_scale" not in calibration:
                raise ValueError(f"{calibration_path} does not contain quant_scale")
            args.fixed_quant_scale = float(calibration["quant_scale"])
        if args.fixed_clip_scale is None:
            clip_scale = calibration.get("clip_scale")
            if clip_scale is None and isinstance(calibration.get("metadata"), dict):
                clip_scale = calibration["metadata"].get("clip_scale")
            if clip_scale is None:
                raise ValueError(f"{calibration_path} does not contain clip_scale")
            args.fixed_clip_scale = float(clip_scale)

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
    from fpu.benchmark import (
        run_full_benchmark,
        benchmark_kernel,
        get_gpu_environment,
    )

    # Print GPU environment
    gpu_env = get_gpu_environment()
    print("GPU Environment:")
    for k, v in gpu_env.items():
        print(f"  {k}: {v}")
    print()

    # Run validation
    if args.mode in ["validate", "both"]:
        print("=" * 60)
        print("VALIDATION")
        print("=" * 60)
        passed = run_validation_suite(verbose=args.verbose)
        if not passed:
            print("\nVALIDATION FAILED!")
            sys.exit(1)
        print("\nVALIDATION PASSED!")

    # Run benchmarks
    if args.mode in ["benchmark", "both"]:
        print("\n" + "=" * 60)
        print("BENCHMARK")
        print("=" * 60)

        if args.model or args.synthetic_params is not None:
            # Single model benchmark
            from fpu.validate import create_test_state_dict
            from fpu.types import FusedUpdateConfig

            if args.synthetic_params is not None:
                if args.synthetic_params < 0:
                    raise ValueError("--synthetic-params must be non-negative")
                gen = torch.Generator()
                gen.manual_seed(2026)
                state_dict = {"flat": torch.randn(args.synthetic_params, generator=gen)}
                model_name = f"synthetic_{args.synthetic_params}"
            else:
                state_dict = create_test_state_dict(args.model)
                model_name = args.model

            num_params = sum(t.numel() for t in state_dict.values())
            bundle_count = (num_params + args.slot_capacity - 1) // args.slot_capacity

            config = FusedUpdateConfig(
                clip_norm=args.clip_norm,
                noise_multiplier=args.noise_multiplier,
                quant_bits=args.quant_bits,
                client_weight=args.client_weight,
                physical_quantized_output=args.physical_quant_output,
                noise_seed=args.noise_seed,
                stateless_noise=args.stateless_noise,
                fixed_quant_scale=args.fixed_quant_scale,
                fixed_clip_scale=args.fixed_clip_scale,
                chunk_size=args.chunk_size,
                chunk_quant_scale=args.chunk_quant_scale,
                pack_block_size=args.pack_block_size,
            )
            if args.stream_output_dir:
                from fpu.torch_ref import flatten_state_dict
                from fpu.triton_kernel import fused_private_update_triton_v2_stream_to_cpu

                flat_gpu = flatten_state_dict(state_dict).to("cuda")
                stream_meta = fused_private_update_triton_v2_stream_to_cpu(
                    flat_gpu,
                    config,
                    bundle_count,
                    slot_capacity=args.slot_capacity,
                    output_dir=args.stream_output_dir,
                    chunk_size=args.chunk_size,
                    prefix=args.stream_prefix,
                    keep_cpu_chunks=False,
                )
                stream_meta["calibration_json"] = args.calibration_json or ""

                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = (
                    f"stream_{model_name}_{args.backend}_q{args.quant_bits}"
                    f"_{time.strftime('%Y%m%d_%H%M%S')}.json"
                )
                filepath = output_dir / filename
                serializable = dict(stream_meta)
                serializable.pop("cpu_chunks", None)
                with open(filepath, "w") as f:
                    json.dump(serializable, f, indent=2)
                print("\nStreaming export complete:")
                print(f"  Chunks: {stream_meta['chunk_count']}")
                print(f"  Output bytes: {stream_meta['output_bytes']:,}")
                print(f"  CUDA total ms: {stream_meta['cuda_total_ms']:.3f}")
                print(f"  Write total ms: {stream_meta['write_total_ms']:.3f}")
                print(f"  Metadata saved to: {filepath}")
                return

            results = benchmark_kernel(
                state_dict, config, bundle_count,
                slot_capacity=args.slot_capacity,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
                backend=args.backend,
                reuse_buffers=args.reuse_buffers,
                use_cuda_graph=args.cuda_graph,
                measure_output_quality=args.measure_output_quality,
                lock_clocks=args.lock_clocks,
                verbose=True,
            )
            results["calibration_json"] = args.calibration_json or ""

            # Save results
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            phys = "_physint8" if args.physical_quant_output else ""
            stateless = "_stateless" if args.stateless_noise else ""
            fixed = "_fixedscale" if args.fixed_quant_scale is not None else ""
            chunk = f"_chunk{args.chunk_size}" if args.chunk_size else ""
            graph = "_cudagraph" if args.cuda_graph else ""
            filename = (
                f"benchmark_{model_name}_{args.backend}"
                f"_q{args.quant_bits}{phys}{stateless}{fixed}{chunk}{graph}"
                f"_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            filepath = output_dir / filename
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {filepath}")

        else:
            # Full benchmark suite
            results = run_full_benchmark(
                output_dir=args.output,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
                verbose=args.verbose,
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
