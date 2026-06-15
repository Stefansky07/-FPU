"""
Benchmarking utilities for the fused private update operator.

Key improvements over initial version:
  - CUDA event timing (GPU-accurate) alongside wall-clock
  - 20 warmup + 100 measurement iterations by default
  - GPU environment logging (model, driver, clocks, power)
  - Best-effort GPU clock locking via nvidia-smi
  - Peak memory tracking
  - actual_output_mb vs logical_output_mb to catch dtype misreporting
  - CV (coefficient of variation) warning when > 5%
  - quant_stat hotspot flag when > 15% of fused time
  - Parameter sweep across slot_capacity / quant_bits / reuse / weight / physical_quant
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .types import FusedUpdateConfig, FusedUpdateOutput, KernelMetrics
from .torch_ref import fused_private_update_ref, flatten_state_dict
from .triton_kernel import fused_private_update_triton, fused_private_update_triton_v2
from .validate import check_triton_available, create_test_state_dict


# ============================================================================
# GPU Environment Helpers
# ============================================================================

def get_gpu_environment() -> Dict[str, Any]:
    """Collect GPU runtime environment information (best-effort)."""
    info: Dict[str, Any] = {}
    info["torch_version"] = torch.__version__
    info["cuda_version"] = getattr(torch.version, "cuda", None) or "unknown"

    try:
        import triton
        info["triton_version"] = triton.__version__
    except ImportError:
        info["triton_version"] = "not installed"

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,clocks.current.graphics,"
                    "clocks.max.graphics,power.limit,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 5:
                    info["driver_version"] = parts[0]
                    info["current_clock_mhz"] = parts[1]
                    info["max_clock_mhz"] = parts[2]
                    info["power_limit_w"] = parts[3]
                    info["memory_total_mib"] = parts[4]
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            info["nvidia_smi"] = "unavailable"
    else:
        info["gpu_name"] = "N/A (CPU only)"

    return info


def try_lock_gpu_clocks(device_id: int = 0) -> Tuple[bool, str]:
    """Best-effort GPU clock locking. Requires admin/root privileges."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "-i", str(device_id),
                "--query-gpu=clocks.max.graphics",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False, f"query failed: {result.stderr.strip()}"
        max_clock = result.stdout.strip()

        result = subprocess.run(
            ["nvidia-smi", "-i", str(device_id), "-lgc", f"{max_clock},{max_clock}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return True, f"locked to {max_clock} MHz"
        else:
            return False, f"lock failed (need admin?): {result.stderr.strip()}"
    except FileNotFoundError:
        return False, "nvidia-smi not found"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timed out"
    except Exception as e:
        return False, f"exception: {e}"


def try_unlock_gpu_clocks(device_id: int = 0) -> None:
    """Unlock GPU clocks (best-effort, silent on failure)."""
    try:
        subprocess.run(
            ["nvidia-smi", "-i", str(device_id), "-rgc"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        pass


# ============================================================================
# Core Benchmark
# ============================================================================

def benchmark_kernel(
    state_dict: Dict[str, torch.Tensor],
    config: FusedUpdateConfig,
    bundle_count: int,
    slot_capacity: int = 4096,
    num_warmup: int = 20,
    num_iterations: int = 100,
    backend: str = "triton",
    reuse_buffers: bool = False,
    lock_clocks: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark kernel execution time with CUDA event timing.

    Args:
        state_dict: Model state dict
        config: Fused update configuration
        bundle_count: Number of bundles
        slot_capacity: Slots per bundle
        num_warmup: Warmup iterations (default 20)
        num_iterations: Measurement iterations (default 100)
        backend: "triton", "triton_v2", or "torch_ref"
        reuse_buffers: Reuse v2 output/noise buffers across iterations
        lock_clocks: Attempt GPU clock locking for stable results
        verbose: Print results

    Returns:
        results: Comprehensive benchmark results dict
    """
    if not check_triton_available():
        return {"error": "Triton/CUDA not available"}

    device = torch.device("cuda")

    # --- GPU environment ---
    gpu_env = get_gpu_environment()
    clock_locked = False
    clock_msg = "not attempted"
    if lock_clocks:
        clock_locked, clock_msg = try_lock_gpu_clocks()
        if verbose:
            status = "✓" if clock_locked else "✗"
            print(f"  Clock lock: {status} {clock_msg}")

    try:
        # --- Select implementation ---
        flat_gpu = None
        if backend == "triton":
            def run_kernel(measure_time: bool = False) -> FusedUpdateOutput:
                return fused_private_update_triton(
                    state_dict, config, bundle_count, slot_capacity, device,
                    measure_time=measure_time,
                )
        elif backend == "triton_v2":
            flat_gpu = flatten_state_dict(state_dict).to(device)
            packed_out = None
            noise_buffer = None
            if reuse_buffers:
                total_slots = bundle_count * slot_capacity
                use_physical_int8 = (
                    config.physical_quantized_output and config.quant_bits == 8
                )
                out_dtype = torch.int8 if use_physical_int8 else torch.float32
                packed_out = torch.empty(total_slots, dtype=out_dtype, device=device)
                if config.noise_multiplier > 0:
                    if config.noise_seed is not None:
                        gen = torch.Generator(device=device)
                        gen.manual_seed(config.noise_seed)
                        noise_buffer = torch.randn(
                            flat_gpu.numel(), generator=gen,
                            device=device, dtype=torch.float32,
                        )
                    else:
                        noise_buffer = torch.randn(
                            flat_gpu.numel(), device=device, dtype=torch.float32,
                        )

            def run_kernel(measure_time: bool = False) -> FusedUpdateOutput:
                return fused_private_update_triton_v2(
                    flat_gpu, config, bundle_count, slot_capacity,
                    measure_time=measure_time,
                    noise=noise_buffer, packed_out=packed_out,
                )
        elif backend == "torch_ref":
            def run_kernel(measure_time: bool = False) -> FusedUpdateOutput:
                return fused_private_update_ref(
                    state_dict, config, bundle_count, slot_capacity, device,
                    measure_time=measure_time,
                )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # --- Warmup ---
        for _ in range(num_warmup):
            _ = run_kernel(measure_time=False)
        torch.cuda.synchronize()

        # --- Peak memory (single run after warmup) ---
        torch.cuda.reset_peak_memory_stats(device)
        _ = run_kernel(measure_time=False)
        torch.cuda.synchronize()
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)

        # --- Measurement loop: CUDA events + wall clock ---
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        cuda_times: List[float] = []
        wall_times: List[float] = []

        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_event.record()
            t0 = time.perf_counter()
            _ = run_kernel(measure_time=False)
            end_event.record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            cuda_times.append(start_event.elapsed_time(end_event))  # ms
            wall_times.append((t1 - t0) * 1000)  # ms

        # --- Measured run for detailed metadata ---
        measured_output = run_kernel(measure_time=True)

        # --- Compute statistics ---
        cuda_t = torch.tensor(cuda_times)
        wall_t = torch.tensor(wall_times)
        num_params = sum(t.numel() for t in state_dict.values())
        meta = measured_output.metadata

        input_bytes = meta.get("input_bytes", num_params * 4)
        logical_quant_payload_bytes = meta.get("logical_quantized_payload_bytes", 0)

        # Inspect actual output tensor
        packed = measured_output.packed_slots
        actual_output_bytes = packed.numel() * packed.element_size()

        # Logical output bytes: what the compressed payload would be
        total_slots = bundle_count * slot_capacity
        if config.quant_bits > 0:
            logical_output_bytes = (total_slots * config.quant_bits + 7) // 8
        else:
            logical_output_bytes = total_slots * 4  # float32

        # CV (coefficient of variation)
        cuda_mean = float(cuda_t.mean().item())
        cuda_std = float(cuda_t.std().item())
        cv_pct = (cuda_std / cuda_mean * 100) if cuda_mean > 0 else 0.0

        # Timing breakdown from measured run
        fused_op_ms = meta.get("fused_operator_ms", 0.0)
        quant_stat_ms = meta.get("quant_stat_ms", 0.0)
        pack_ms = meta.get("pack_ms", 0.0)

        results: Dict[str, Any] = {
            # Configuration
            "backend": backend,
            "clip_norm": config.clip_norm,
            "noise_multiplier": config.noise_multiplier,
            "quant_bits": config.quant_bits,
            "physical_quantized_output": config.physical_quantized_output,
            "client_weight": config.client_weight,
            "reuse_buffers": reuse_buffers,
            "num_params": num_params,
            "bundle_count": bundle_count,
            "slot_capacity": slot_capacity,
            "num_warmup": num_warmup,
            "num_iterations": num_iterations,
            # CUDA event timing (GPU-accurate)
            "cuda_mean_ms": cuda_mean,
            "cuda_std_ms": cuda_std,
            "cuda_min_ms": float(cuda_t.min().item()),
            "cuda_max_ms": float(cuda_t.max().item()),
            "cuda_median_ms": float(cuda_t.median().item()),
            "cuda_p90_ms": float(torch.quantile(cuda_t, 0.90).item()),
            "cuda_p95_ms": float(torch.quantile(cuda_t, 0.95).item()),
            "cuda_p99_ms": float(torch.quantile(cuda_t, 0.99).item()),
            "cuda_cv_pct": round(cv_pct, 2),
            # Wall clock timing (secondary)
            "wall_mean_ms": float(wall_t.mean().item()),
            "wall_std_ms": float(wall_t.std().item()),
            "wall_median_ms": float(wall_t.median().item()),
            # Kernel metadata (from single measured run)
            "kernel_launch_count": meta.get("kernel_launch_count", 0),
            "fused_operator_ms": fused_op_ms,
            "quant_stat_ms": quant_stat_ms,
            "quant_stat_source": meta.get("quant_stat_source", "none"),
            "quant_abs_max": meta.get("quant_abs_max", 0.0),
            "quant_scale": meta.get("quant_scale", 1.0),
            "packing_ms": pack_ms,
            "noise_ms": meta.get("noise_ms", 0.0),
            "noise_bytes": meta.get("noise_bytes", 0),
            "noise_source": meta.get("noise_source", ""),
            # Output tensor inspection
            "output_dtype": str(packed.dtype).replace("torch.", ""),
            "output_format": meta.get("output_format", "float_slots"),
            "packed_shape": list(packed.shape),
            "packed_numel": packed.numel(),
            "packed_element_size": packed.element_size(),
            # Byte-accurate I/O stats
            "input_bytes": input_bytes,
            "input_mb": round(input_bytes / 1e6, 3),
            "actual_output_bytes": actual_output_bytes,
            "actual_output_mb": round(actual_output_bytes / 1e6, 3),
            "logical_output_bytes": logical_output_bytes,
            "logical_output_mb": round(logical_output_bytes / 1e6, 3),
            "logical_quantized_payload_bytes": logical_quant_payload_bytes,
            "logical_quantized_payload_mb": round(logical_quant_payload_bytes / 1e6, 3),
            # Compression
            "compression_ratio": round(
                input_bytes / max(actual_output_bytes, 1), 3
            ),
            # Bandwidth (based on actual physical bytes transferred)
            "bandwidth_gbps": round(
                ((input_bytes + actual_output_bytes) / 1e9) / (cuda_mean / 1e3), 1
            ) if cuda_mean > 0 else 0.0,
            # Memory
            "peak_memory_mb": round(peak_memory_bytes / 1e6, 1),
            # GPU environment
            "gpu_clock_locked": clock_locked,
            "gpu_clock_msg": clock_msg,
        }

        # --- Flags / warnings ---
        if cv_pct > 5.0:
            results["_warning_high_cv"] = (
                f"CV={cv_pct:.1f}% > 5%, results may be noisy. "
                f"Consider --lock-clocks or increasing iterations."
            )

        if fused_op_ms > 0 and quant_stat_ms / fused_op_ms > 0.15:
            results["_flag_quant_stat_hot"] = (
                f"quant_stat_ms={quant_stat_ms:.2f} ms is "
                f"{quant_stat_ms / fused_op_ms * 100:.1f}% of fused_operator_ms "
                f"— candidate for fusion optimization"
            )

        if verbose:
            _print_benchmark_summary(results)

        return results

    finally:
        if lock_clocks:
            try_unlock_gpu_clocks()


def _print_benchmark_summary(r: Dict[str, Any]) -> None:
    """Pretty-print benchmark results."""
    print(f"\nBenchmark ({r['backend']}):")
    print(f"  Params: {r['num_params']:,}")
    print(f"  Bundles: {r['bundle_count']}, Slot capacity: {r['slot_capacity']}")
    print(f"  Physical quant output: {r['physical_quantized_output']}")
    print(f"  CUDA event timing ({r['num_iterations']} iters, {r['num_warmup']} warmup):")
    print(f"    Mean:   {r['cuda_mean_ms']:.3f} ms")
    print(f"    Median: {r['cuda_median_ms']:.3f} ms")
    print(f"    Std:    {r['cuda_std_ms']:.3f} ms")
    print(f"    P90:    {r['cuda_p90_ms']:.3f} ms")
    print(f"    P95:    {r['cuda_p95_ms']:.3f} ms")
    print(f"    CV:     {r['cuda_cv_pct']:.2f}%")
    print(f"  Wall clock mean: {r['wall_mean_ms']:.3f} ms")
    print(f"  Kernel launches: {r['kernel_launch_count']}")
    print(f"  Timing breakdown (single measured run):")
    print(f"    fused_operator_ms: {r['fused_operator_ms']:.3f}")
    print(f"    quant_stat_ms:     {r['quant_stat_ms']:.3f}")
    print(f"    packing_ms:        {r['packing_ms']:.3f}")
    print(f"    noise_ms:          {r['noise_ms']:.3f}")
    print(f"  Output tensor:")
    print(f"    dtype={r['output_dtype']}, format={r['output_format']}")
    print(f"    shape={r['packed_shape']}, numel={r['packed_numel']}")
    print(f"    element_size={r['packed_element_size']} bytes")
    print(f"  I/O bytes (actual physical):")
    print(f"    input_mb:          {r['input_mb']:.3f}")
    print(f"    actual_output_mb:  {r['actual_output_mb']:.3f}")
    print(f"    logical_output_mb: {r['logical_output_mb']:.3f}")
    print(f"    compression_ratio: {r['compression_ratio']:.2f}x")
    print(f"  Bandwidth: {r['bandwidth_gbps']:.1f} GB/s")
    print(f"  Peak memory: {r['peak_memory_mb']:.1f} MB")
    if r.get("_warning_high_cv"):
        print(f"  [WARN] {r['_warning_high_cv']}")
    if r.get("_flag_quant_stat_hot"):
        print(f"  [HOT]  {r['_flag_quant_stat_hot']}")


# ============================================================================
# Scaling Benchmark
# ============================================================================

def benchmark_scaling(
    model_sizes: List[str] = None,
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.1,
    num_warmup: int = 20,
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
        ref_ms = ref_result.get("cuda_mean_ms", 0)
        if ref_ms > 0:
            speedup = ref_ms / triton_result.get("cuda_mean_ms", 1)
            v2_speedup = ref_ms / triton_v2_result.get("cuda_mean_ms", 1)
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


# ============================================================================
# Configuration Benchmark
# ============================================================================

def benchmark_configurations(
    state_dict: Dict[str, torch.Tensor],
    configs: List[FusedUpdateConfig],
    bundle_count: int,
    num_warmup: int = 20,
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
            print(
                f"Config {i}: clip={config.clip_norm}, noise={config.noise_multiplier}, "
                f"quant={config.quant_bits}, phys_quant={config.physical_quantized_output}"
            )

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
            "physical_quantized_output": config.physical_quantized_output,
            "triton": triton_result,
            "triton_v2": triton_v2_result,
        }
        results.append(result)

    return results


# ============================================================================
# Parameter Sweep
# ============================================================================

def benchmark_param_sweep(
    num_params: int,
    slot_capacities: Optional[List[int]] = None,
    quant_bits_list: Optional[List[int]] = None,
    reuse_buffers_list: Optional[List[bool]] = None,
    client_weights: Optional[List[float]] = None,
    physical_quant_list: Optional[List[bool]] = None,
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.0,
    num_warmup: int = 20,
    num_iterations: int = 100,
    backend: str = "triton_v2",
    lock_clocks: bool = False,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Parameter sweep across configurations.

    Args:
        num_params: Number of synthetic parameters
        slot_capacities: List of slot capacities to test
        quant_bits_list: List of quant bits to test
        reuse_buffers_list: List of reuse_buffers values
        client_weights: List of client weights
        physical_quant_list: List of physical_quantized_output values
        clip_norm: Clipping norm
        noise_multiplier: Noise multiplier
        num_warmup: Warmup iterations
        num_iterations: Measurement iterations
        backend: Backend to benchmark
        lock_clocks: Attempt GPU clock locking
        verbose: Print results

    Returns:
        results: List of benchmark results per configuration
    """
    if slot_capacities is None:
        slot_capacities = [1024, 2048, 4096, 8192]
    if quant_bits_list is None:
        quant_bits_list = [4, 8]
    if reuse_buffers_list is None:
        reuse_buffers_list = [True, False]
    if client_weights is None:
        client_weights = [1.0, 0.5]
    if physical_quant_list is None:
        physical_quant_list = [True, False]

    # Generate synthetic state dict
    gen = torch.Generator()
    gen.manual_seed(2026)
    state_dict = {"flat": torch.randn(num_params, generator=gen)}

    # Count total combinations
    combos = []
    for sc in slot_capacities:
        for qb in quant_bits_list:
            for rb in reuse_buffers_list:
                for cw in client_weights:
                    for pq in physical_quant_list:
                        # physical_quantized_output only valid with quant_bits=8
                        if pq and qb != 8:
                            continue
                        combos.append((sc, qb, rb, cw, pq))

    total = len(combos)
    if verbose:
        print(f"\nParameter sweep: {total} combinations, {num_params:,} params")
        print(f"  slot_capacities: {slot_capacities}")
        print(f"  quant_bits: {quant_bits_list}")
        print(f"  reuse_buffers: {reuse_buffers_list}")
        print(f"  client_weights: {client_weights}")
        print(f"  physical_quant: {physical_quant_list}")

    results = []
    for idx, (sc, qb, rb, cw, pq) in enumerate(combos):
        bundle_count = (num_params + sc - 1) // sc

        config = FusedUpdateConfig(
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            quant_bits=qb,
            client_weight=cw,
            physical_quantized_output=pq,
        )

        if verbose:
            print(
                f"\n[{idx+1}/{total}] slot={sc} qb={qb} reuse={rb} "
                f"cw={cw} phys_quant={pq}"
            )

        result = benchmark_kernel(
            state_dict, config, bundle_count,
            slot_capacity=sc,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            backend=backend,
            reuse_buffers=rb,
            lock_clocks=lock_clocks,
            verbose=verbose,
        )

        # Tag sweep parameters for easy filtering
        result["sweep_slot_capacity"] = sc
        result["sweep_quant_bits"] = qb
        result["sweep_reuse_buffers"] = rb
        result["sweep_client_weight"] = cw
        result["sweep_physical_quant"] = pq
        results.append(result)

    return results


def save_sweep_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    prefix: str = "sweep",
) -> Tuple[Path, Path]:
    """
    Save sweep results as JSON and summary CSV.

    Returns:
        (json_path, csv_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"{prefix}_{timestamp}.json"
    csv_path = output_path / f"{prefix}_{timestamp}.csv"

    # JSON (full data)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV (summary columns)
    csv_columns = [
        "sweep_slot_capacity", "sweep_quant_bits", "sweep_reuse_buffers",
        "sweep_client_weight", "sweep_physical_quant",
        "cuda_mean_ms", "cuda_median_ms", "cuda_std_ms", "cuda_p90_ms",
        "cuda_cv_pct", "kernel_launch_count",
        "fused_operator_ms", "quant_stat_ms", "packing_ms",
        "input_mb", "actual_output_mb", "logical_output_mb",
        "output_dtype", "output_format", "compression_ratio",
        "bandwidth_gbps", "peak_memory_mb",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    return json_path, csv_path


# ============================================================================
# Full Benchmark Suite
# ============================================================================

def run_full_benchmark(
    output_dir: Optional[str] = None,
    num_warmup: int = 20,
    num_iterations: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark suite.

    Args:
        output_dir: Directory to save results
        num_warmup: Warmup iterations
        num_iterations: Measurement iterations
        verbose: Print results

    Returns:
        results: All benchmark results
    """
    if not check_triton_available():
        return {"error": "Triton/CUDA not available"}

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "gpu_environment": get_gpu_environment(),
    }

    # Scaling benchmark
    if verbose:
        print("\n" + "=" * 60)
        print("SCALING BENCHMARK")
        print("=" * 60)

    results["scaling"] = benchmark_scaling(
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        verbose=verbose,
    )

    # Configuration benchmark
    if verbose:
        print("\n" + "=" * 60)
        print("CONFIGURATION BENCHMARK")
        print("=" * 60)

    state_dict = create_test_state_dict("tiny_cnn")
    num_params = sum(t.numel() for t in state_dict.values())
    bundle_count = (num_params + 4095) // 4096

    configs = [
        FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.0, quant_bits=0),
        FusedUpdateConfig(clip_norm=0.5, noise_multiplier=0.0, quant_bits=0),
        FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.1, quant_bits=0),
        FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.5, quant_bits=0),
        FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.0, quant_bits=8),
        FusedUpdateConfig(
            clip_norm=1.0, noise_multiplier=0.0, quant_bits=8,
            physical_quantized_output=True,
        ),
        FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.1, quant_bits=8),
    ]

    results["configurations"] = benchmark_configurations(
        state_dict, configs, bundle_count,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        verbose=verbose,
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
