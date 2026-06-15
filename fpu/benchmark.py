"""
Benchmarking utilities for the fused private update operator.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .types import FusedUpdateConfig, FusedUpdateOutput, KernelMetrics
from .torch_ref import fused_private_update_ref, flatten_state_dict
from .triton_kernel import fused_private_update_triton, fused_private_update_triton_v2
from .validate import check_triton_available, create_test_state_dict


def benchmark_kernel(
    state_dict: Dict[str, torch.Tensor],
    config: FusedUpdateConfig,
    bundle_count: int,
    slot_capacity: int = 4096,
    num_warmup: int = 10,
    num_iterations: int = 100,
    backend: str = "triton",
    reuse_buffers: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark kernel execution time.

    Args:
        state_dict: Model state dict
        config: Fused update configuration
        bundle_count: Number of bundles
        slot_capacity: Slots per bundle
        num_warmup: Warmup iterations
        num_iterations: Measurement iterations
        backend: "triton", "triton_v2", or "torch_ref"
        reuse_buffers: Reuse v2 output/noise buffers across iterations
        verbose: Print results

    Returns:
        results: Benchmark results
    """
    if not check_triton_available():
        return {"error": "Triton/CUDA not available"}

    device = torch.device("cuda")

    # Select implementation
    flat_gpu = None
    if backend == "triton":
        def run_kernel(measure_time: bool = False) -> FusedUpdateOutput:
            return fused_private_update_triton(
                state_dict, config, bundle_count, slot_capacity, device, measure_time=measure_time
            )
    elif backend == "triton_v2":
        flat_gpu = flatten_state_dict(state_dict).to(device)
        packed_out = None
        noise_buffer = None
        if reuse_buffers:
            total_slots = bundle_count * slot_capacity
            packed_out = torch.empty(total_slots, dtype=torch.float32, device=device)
            if config.noise_multiplier > 0:
                if config.noise_seed is not None:
                    gen = torch.Generator(device=device)
                    gen.manual_seed(config.noise_seed)
                    noise_buffer = torch.randn(
                        flat_gpu.numel(), generator=gen, device=device, dtype=torch.float32
                    )
                else:
                    noise_buffer = torch.randn(flat_gpu.numel(), device=device, dtype=torch.float32)

        def run_kernel(measure_time: bool = False) -> FusedUpdateOutput:
            return fused_private_update_triton_v2(
                flat_gpu,
                config,
                bundle_count,
                slot_capacity,
                measure_time=measure_time,
                noise=noise_buffer,
                packed_out=packed_out,
            )
    elif backend == "torch_ref":
        def run_kernel(measure_time: bool = False) -> FusedUpdateOutput:
            return fused_private_update_ref(
                state_dict, config, bundle_count, slot_capacity, device, measure_time=measure_time
            )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Warmup
    for _ in range(num_warmup):
        _ = run_kernel(measure_time=False)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = run_kernel(measure_time=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    measured_output = run_kernel(measure_time=True)

    # Compute statistics
    times_tensor = torch.tensor(times)
    num_params = sum(t.numel() for t in state_dict.values())

    results = {
        "backend": backend,
        "clip_norm": config.clip_norm,
        "noise_multiplier": config.noise_multiplier,
        "quant_bits": config.quant_bits,
        "client_weight": config.client_weight,
        "reuse_buffers": reuse_buffers,
        "num_params": num_params,
        "bundle_count": bundle_count,
        "slot_capacity": slot_capacity,
        "num_iterations": num_iterations,
        "mean_ms": float(times_tensor.mean().item()),
        "std_ms": float(times_tensor.std().item()),
        "min_ms": float(times_tensor.min().item()),
        "max_ms": float(times_tensor.max().item()),
        "median_ms": float(times_tensor.median().item()),
        "p95_ms": float(torch.quantile(times_tensor, 0.95).item()),
        "p99_ms": float(torch.quantile(times_tensor, 0.99).item()),
        "kernel_launch_count": measured_output.metadata.get("kernel_launch_count", 0),
        "fused_operator_ms": measured_output.metadata.get("fused_operator_ms", 0.0),
        "quant_stat_ms": measured_output.metadata.get("quant_stat_ms", 0.0),
        "noise_bytes": measured_output.metadata.get("noise_bytes", 0),
        "noise_source": measured_output.metadata.get("noise_source", ""),
    }

    # Compute bandwidth
    input_bytes = num_params * 4  # float32
    output_bytes = bundle_count * slot_capacity * 4
    total_bytes = input_bytes + output_bytes
    results["input_mb"] = input_bytes / 1e6
    results["output_mb"] = output_bytes / 1e6
    results["bandwidth_gbps"] = (total_bytes / 1e9) / (results["mean_ms"] / 1e3)

    if verbose:
        print(f"\nBenchmark ({backend}):")
        print(f"  Params: {num_params:,}")
        print(f"  Bundles: {bundle_count}")
        print(f"  Mean: {results['mean_ms']:.3f} ms")
        print(f"  Std: {results['std_ms']:.3f} ms")
        print(f"  Min: {results['min_ms']:.3f} ms")
        print(f"  P95: {results['p95_ms']:.3f} ms")
        print(f"  Bandwidth: {results['bandwidth_gbps']:.1f} GB/s")

    return results


def benchmark_scaling(
    model_sizes: List[str] = None,
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.1,
    num_warmup: int = 10,
    num_iterations: int = 100,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Benchmark scaling across different model sizes.

    Args:
        model_sizes: List of model types to test
        clip_norm: Clipping norm
        noise_multiplier: Noise multiplier
        num_warmup: Warmup iterations
        num_iterations: Measurement iterations
        verbose: Print results

    Returns:
        results: List of benchmark results
    """
    if model_sizes is None:
        model_sizes = ["tiny_cnn", "tiny_mlp", "resnet18"]

    config = FusedUpdateConfig(
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
    )

    results = []
    for model_type in model_sizes:
        state_dict = create_test_state_dict(model_type)
        num_params = sum(t.numel() for t in state_dict.values())
        bundle_count = (num_params + 4095) // 4096

        if verbose:
            print(f"\n{'='*60}")
            print(f"Model: {model_type} ({num_params:,} params)")

        # Benchmark Triton
        triton_result = benchmark_kernel(
            state_dict, config, bundle_count,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            backend="triton",
            verbose=verbose,
        )

        # Benchmark fused Triton
        triton_v2_result = benchmark_kernel(
            state_dict, config, bundle_count,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            backend="triton_v2",
            verbose=verbose,
        )

        # Benchmark PyTorch reference
        ref_result = benchmark_kernel(
            state_dict, config, bundle_count,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            backend="torch_ref",
            verbose=verbose,
        )

        # Compute speedup
        if ref_result.get("mean_ms", 0) > 0:
            speedup = ref_result["mean_ms"] / triton_result["mean_ms"]
            v2_speedup = ref_result["mean_ms"] / triton_v2_result["mean_ms"]
        else:
            speedup = 0.0
            v2_speedup = 0.0

        result = {
            "model": model_type,
            "num_params": num_params,
            "triton": triton_result,
            "triton_v2": triton_v2_result,
            "torch_ref": ref_result,
            "speedup": speedup,
            "v2_speedup": v2_speedup,
        }
        results.append(result)

        if verbose:
            print(f"\n  Speedup: {speedup:.2f}x")
            print(f"  V2 speedup: {v2_speedup:.2f}x")

    return results


def benchmark_configurations(
    state_dict: Dict[str, torch.Tensor],
    configs: List[FusedUpdateConfig],
    bundle_count: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Benchmark different configurations on the same model.

    Args:
        state_dict: Model state dict
        configs: List of configurations to test
        bundle_count: Number of bundles
        num_warmup: Warmup iterations
        num_iterations: Measurement iterations
        verbose: Print results

    Returns:
        results: List of benchmark results
    """
    results = []
    for i, config in enumerate(configs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Config {i}: clip={config.clip_norm}, noise={config.noise_multiplier}, quant={config.quant_bits}")

        triton_result = benchmark_kernel(
            state_dict, config, bundle_count,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            backend="triton",
            verbose=verbose,
        )
        triton_v2_result = benchmark_kernel(
            state_dict, config, bundle_count,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            backend="triton_v2",
            verbose=verbose,
        )

        result = {
            "config_index": i,
            "clip_norm": config.clip_norm,
            "noise_multiplier": config.noise_multiplier,
            "quant_bits": config.quant_bits,
            "triton": triton_result,
            "triton_v2": triton_v2_result,
        }
        results.append(result)

    return results


def run_full_benchmark(
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark suite.

    Args:
        output_dir: Directory to save results
        verbose: Print results

    Returns:
        results: All benchmark results
    """
    if not check_triton_available():
        return {"error": "Triton/CUDA not available"}

    results = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    # Scaling benchmark
    if verbose:
        print("\n" + "="*60)
        print("SCALING BENCHMARK")
        print("="*60)

    results["scaling"] = benchmark_scaling(verbose=verbose)

    # Configuration benchmark
    if verbose:
        print("\n" + "="*60)
        print("CONFIGURATION BENCHMARK")
        print("="*60)

    state_dict = create_test_state_dict("tiny_cnn")
    num_params = sum(t.numel() for t in state_dict.values())
    bundle_count = (num_params + 4095) // 4096

    configs = [
        FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.0, quant_bits=0),
        FusedUpdateConfig(clip_norm=0.5, noise_multiplier=0.0, quant_bits=0),
        FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.1, quant_bits=0),
        FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.5, quant_bits=0),
        FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.0, quant_bits=8),
        FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.1, quant_bits=8),
    ]

    results["configurations"] = benchmark_configurations(
        state_dict, configs, bundle_count, verbose=verbose
    )

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"benchmark_{results['timestamp']}.json"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        if verbose:
            print(f"\nResults saved to: {filepath}")

    return results


if __name__ == "__main__":
    results = run_full_benchmark(output_dir="results", verbose=True)
