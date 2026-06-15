"""
Validation utilities for verifying kernel correctness.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .types import FusedUpdateConfig, FusedUpdateOutput
from .torch_ref import fused_private_update_ref, flatten_state_dict

try:
    import triton
    from .triton_kernel import (
        compute_transformed_abs_max_triton,
        fused_private_update_triton,
        fused_private_update_triton_v2,
    )
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def check_triton_available() -> bool:
    """Check if Triton is available and CUDA is accessible."""
    if not _HAS_TRITON:
        print("WARNING: Triton not installed, skipping GPU tests")
        return False
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping GPU tests")
        return False
    return True


def _compute_error_metrics(
    ref_slots: torch.Tensor,
    test_slots: torch.Tensor,
) -> Dict[str, float]:
    """Compute absolute and relative error metrics for validation."""
    if ref_slots.numel() == 0:
        return {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_rel_error": 0.0,
            "mean_rel_error": 0.0,
        }

    abs_diff = torch.abs(ref_slots - test_slots)
    ref_abs = torch.abs(ref_slots) + 1e-8
    rel_diff = abs_diff / ref_abs
    return {
        "max_abs_error": float(torch.max(abs_diff).item()),
        "mean_abs_error": float(torch.mean(abs_diff).item()),
        "max_rel_error": float(torch.max(rel_diff).item()),
        "mean_rel_error": float(torch.mean(rel_diff).item()),
    }


def validate_against_reference(
    state_dict: Dict[str, torch.Tensor],
    config: FusedUpdateConfig,
    bundle_count: int,
    slot_capacity: int = 4096,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    backend: str = "triton",
    verbose: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate Triton implementation against PyTorch reference.

    Args:
        state_dict: Model update state dict
        config: Fused update configuration
        bundle_count: Number of bundles
        slot_capacity: Slots per bundle
        rtol: Relative tolerance
        atol: Absolute tolerance
        backend: "triton" for the staged Triton path or "triton_v2" for fused path
        verbose: Print details

    Returns:
        passed: Whether validation passed
        metrics: Error metrics
    """
    if not check_triton_available():
        return True, {"skipped": True}

    device = torch.device("cuda")

    # Run reference implementation
    ref_output = fused_private_update_ref(
        state_dict, config, bundle_count, slot_capacity, device, measure_time=False
    )

    # Run Triton implementation
    if backend == "triton":
        triton_output = fused_private_update_triton(
            state_dict, config, bundle_count, slot_capacity, device, measure_time=False
        )
    elif backend == "triton_v2":
        flat_gpu = flatten_state_dict(state_dict).to(device)
        triton_output = fused_private_update_triton_v2(
            flat_gpu, config, bundle_count, slot_capacity, measure_time=False
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Compare outputs
    ref_slots = ref_output.packed_slots
    triton_slots = triton_output.packed_slots

    metrics = _compute_error_metrics(ref_slots, triton_slots)
    metrics["backend"] = backend

    # Check tolerances
    passed = metrics["max_abs_error"] < atol or metrics["max_rel_error"] < rtol

    if verbose:
        print(f"Validation ({backend}) {'PASSED' if passed else 'FAILED'}:")
        print(f"  Max absolute error: {metrics['max_abs_error']:.2e}")
        print(f"  Mean absolute error: {metrics['mean_abs_error']:.2e}")
        print(f"  Max relative error: {metrics['max_rel_error']:.2e}")
        print(f"  Mean relative error: {metrics['mean_rel_error']:.2e}")

    return passed, metrics


def validate_slot_packing(
    flat: torch.Tensor,
    bundle_count: int,
    slot_capacity: int,
    rtol: float = 1e-5,
    verbose: bool = True,
) -> bool:
    """
    Validate that slot packing preserves values correctly.

    Args:
        flat: Input flat tensor
        bundle_count: Number of bundles
        slot_capacity: Slots per bundle
        rtol: Relative tolerance
        verbose: Print details

    Returns:
        passed: Whether validation passed
    """
    if not check_triton_available():
        return True

    device = torch.device("cuda")
    flat_gpu = flat.to(device)
    num_params = flat_gpu.numel()
    total_slots = bundle_count * slot_capacity

    # Manual packing
    padded = torch.zeros(total_slots, dtype=torch.float32, device=device)
    padded[:num_params] = flat_gpu
    expected = padded.reshape(bundle_count, slot_capacity)

    # Triton packing
    from .triton_kernel import _pack_to_slots_kernel
    packed = torch.zeros(total_slots, dtype=torch.float32, device=device)

    BLOCK_SIZE = 1024
    if num_params > 0:
        grid = (triton.cdiv(num_params, BLOCK_SIZE),)
        _pack_to_slots_kernel[grid](
            flat_gpu,
            packed,
            num_params,
            slot_capacity,
            num_params,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    packed = packed.reshape(bundle_count, slot_capacity)

    # Compare
    max_error = float(torch.max(torch.abs(expected - packed)).item())
    passed = max_error < rtol

    if verbose:
        print(f"Slot packing validation {'PASSED' if passed else 'FAILED'}:")
        print(f"  Max error: {max_error:.2e}")

    return passed


def validate_l2_norm(
    flat: torch.Tensor,
    rtol: float = 1e-5,
    verbose: bool = True,
) -> bool:
    """
    Validate L2 norm computation.

    Args:
        flat: Input tensor
        rtol: Relative tolerance
        verbose: Print details

    Returns:
        passed: Whether validation passed
    """
    if not check_triton_available():
        return True

    device = torch.device("cuda")
    flat_gpu = flat.to(device)

    # Reference
    ref_norm = float(torch.linalg.vector_norm(flat_gpu).item())

    # Triton
    from .triton_kernel import compute_l2_norm_triton
    triton_norm = compute_l2_norm_triton(flat_gpu)

    # Compare
    error = abs(ref_norm - triton_norm) / (ref_norm + 1e-8)
    passed = error < rtol

    if verbose:
        print(f"L2 norm validation {'PASSED' if passed else 'FAILED'}:")
        print(f"  Reference: {ref_norm:.6f}")
        print(f"  Triton: {triton_norm:.6f}")
        print(f"  Relative error: {error:.2e}")

    return passed


def validate_edge_cases(verbose: bool = True) -> bool:
    """Validate empty updates and argument checks that should fail fast."""
    if not check_triton_available():
        return True

    device = torch.device("cuda")
    all_passed = True

    if verbose:
        print("\n--- Edge Case Test ---")

    empty_state: Dict[str, torch.Tensor] = {}
    empty_config = FusedUpdateConfig()
    ref_empty = fused_private_update_ref(empty_state, empty_config, 1, 4096, device)
    triton_empty = fused_private_update_triton(empty_state, empty_config, 1, 4096, device)
    flat_empty = torch.empty(0, dtype=torch.float32, device=device)
    v2_empty = fused_private_update_triton_v2(flat_empty, empty_config, 1, 4096)

    for name, output in (("triton", triton_empty), ("triton_v2", v2_empty)):
        metrics = _compute_error_metrics(ref_empty.packed_slots, output.packed_slots)
        passed = metrics["max_abs_error"] == 0.0
        all_passed = all_passed and passed
        if verbose:
            print(f"  Empty update {name}: {'PASSED' if passed else 'FAILED'}")

    state_dict = create_test_state_dict("tiny_mlp")
    flat_gpu = flatten_state_dict(state_dict).to(device)
    too_small_bundle_count = 1
    bad_config = FusedUpdateConfig(quant_bits=1)

    expected_errors = [
        (
            "capacity check",
            lambda: fused_private_update_triton(
                state_dict, empty_config, too_small_bundle_count, 4096, device
            ),
        ),
        (
            "v2 capacity check",
            lambda: fused_private_update_triton_v2(
                flat_gpu, empty_config, too_small_bundle_count, 4096
            ),
        ),
        (
            "invalid quant_bits",
            lambda: fused_private_update_triton(
                state_dict, bad_config, 13, 4096, device
            ),
        ),
    ]

    for label, fn in expected_errors:
        try:
            fn()
        except (TypeError, ValueError):
            passed = True
        else:
            passed = False
        all_passed = all_passed and passed
        if verbose:
            print(f"  {label}: {'PASSED' if passed else 'FAILED'}")

    return all_passed


def validate_large_scale_readiness(
    num_params: int = 1_000_000,
    slot_capacity: int = 4096,
    verbose: bool = True,
) -> bool:
    """Smoke-test the fused v2 path used by large-scale callers."""
    if not check_triton_available():
        return True

    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(2026)

    flat = torch.randn(num_params, generator=gen, device=device, dtype=torch.float32)
    noise = torch.randn(num_params, generator=gen, device=device, dtype=torch.float32)
    bundle_count = (num_params + slot_capacity - 1) // slot_capacity
    config = FusedUpdateConfig(
        clip_norm=1.0,
        noise_multiplier=0.1,
        quant_bits=8,
        client_weight=0.25,
    )

    packed_out = torch.empty(bundle_count, slot_capacity, dtype=torch.float32, device=device)
    output = fused_private_update_triton_v2(
        flat,
        config,
        bundle_count,
        slot_capacity,
        measure_time=True,
        noise=noise,
        packed_out=packed_out,
    )

    clip_scale = float(output.metadata["clip_scale"])
    noise_std = float(output.metadata["noise_std"])
    quant_scale = float(output.metadata["quant_scale"])
    transformed = flat * clip_scale * config.client_weight + noise * noise_std
    qmin = -(2 ** (config.quant_bits - 1))
    qmax = 2 ** (config.quant_bits - 1) - 1
    expected_flat = torch.clamp(torch.round(transformed / quant_scale), qmin, qmax) * quant_scale

    expected = torch.zeros(bundle_count * slot_capacity, dtype=torch.float32, device=device)
    expected[:num_params] = expected_flat
    expected = expected.reshape(bundle_count, slot_capacity)

    metrics = _compute_error_metrics(expected, output.packed_slots)
    output_reused = output.packed_slots.data_ptr() == packed_out.data_ptr()
    stat_abs_max = compute_transformed_abs_max_triton(
        flat,
        noise,
        num_params,
        clip_scale,
        config.client_weight,
        noise_std,
    )
    ref_abs_max = float(torch.max(torch.abs(transformed)).item())
    stat_rel_error = abs(stat_abs_max - ref_abs_max) / (ref_abs_max + 1e-8)
    quant_lsb_ok = metrics["max_abs_error"] <= quant_scale * 1.01

    passed = (
        quant_lsb_ok
        and output_reused
        and stat_rel_error < 1e-5
        and output.metadata.get("noise_source") == "external"
    )

    if verbose:
        print("\n--- Large-Scale Readiness Test ---")
        print(f"  Params: {num_params:,}")
        print(f"  Bundles: {bundle_count}")
        print(f"  Output buffer reused: {'yes' if output_reused else 'no'}")
        print(f"  Max absolute error: {metrics['max_abs_error']:.2e}")
        print(f"  Max relative error: {metrics['max_rel_error']:.2e}")
        print(f"  Quant scale: {quant_scale:.2e}")
        print(f"  Abs-max stat relative error: {stat_rel_error:.2e}")
        print(f"  Kernel launches: {output.metadata.get('kernel_launch_count', 0)}")
        print(f"  Noise bytes: {output.metadata.get('noise_bytes', 0):,}")
        print(f"  Result: {'PASSED' if passed else 'FAILED'}")

    return passed


def create_test_state_dict(
    model_type: str = "tiny_cnn",
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Create a test state dict for validation.

    Args:
        model_type: Type of model (tiny_cnn, tiny_mlp, resnet18)
        seed: Random seed

    Returns:
        state_dict: Test state dict
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    if model_type == "tiny_cnn":
        return {
            "conv1.weight": torch.randn(8, 3, 3, 3, generator=gen),
            "conv1.bias": torch.randn(8, generator=gen),
            "conv2.weight": torch.randn(16, 8, 3, 3, generator=gen),
            "conv2.bias": torch.randn(16, generator=gen),
            "fc1.weight": torch.randn(128, 16 * 6 * 6, generator=gen),
            "fc1.bias": torch.randn(128, generator=gen),
            "fc2.weight": torch.randn(10, 128, generator=gen),
            "fc2.bias": torch.randn(10, generator=gen),
        }
    elif model_type == "tiny_mlp":
        return {
            "fc1.weight": torch.randn(64, 784, generator=gen),
            "fc1.bias": torch.randn(64, generator=gen),
            "fc2.weight": torch.randn(32, 64, generator=gen),
            "fc2.bias": torch.randn(32, generator=gen),
            "fc3.weight": torch.randn(10, 32, generator=gen),
            "fc3.bias": torch.randn(10, generator=gen),
        }
    elif model_type == "resnet18":
        # Simplified ResNet18-like structure
        return {
            "conv1.weight": torch.randn(64, 3, 7, 7, generator=gen),
            "bn1.weight": torch.randn(64, generator=gen),
            "bn1.bias": torch.randn(64, generator=gen),
            "layer1.0.conv1.weight": torch.randn(64, 64, 3, 3, generator=gen),
            "layer1.0.bn1.weight": torch.randn(64, generator=gen),
            "layer1.0.conv2.weight": torch.randn(64, 64, 3, 3, generator=gen),
            "layer1.0.bn2.weight": torch.randn(64, generator=gen),
            "fc.weight": torch.randn(10, 512, generator=gen),
            "fc.bias": torch.randn(10, generator=gen),
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_validation_suite(
    verbose: bool = True,
) -> bool:
    """
    Run complete validation suite.

    Args:
        verbose: Print details

    Returns:
        all_passed: Whether all tests passed
    """
    if not check_triton_available():
        print("Skipping validation (Triton/CUDA not available)")
        return True

    all_passed = True

    # Test configurations
    configs = [
        ("baseline", FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.0, quant_bits=0)),
        ("noise_only", FusedUpdateConfig(clip_norm=0.5, noise_multiplier=0.1, quant_bits=0, noise_seed=42)),
        ("quant_only", FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.0, quant_bits=8)),
        ("noise_quant", FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.1, quant_bits=8, noise_seed=42)),
        (
            "weighted_noise_quant",
            FusedUpdateConfig(
                clip_norm=1.0,
                noise_multiplier=0.1,
                quant_bits=8,
                client_weight=0.25,
                noise_seed=42,
            ),
        ),
    ]

    models = ["tiny_cnn", "tiny_mlp"]
    backends = ["triton", "triton_v2"]

    for model_type in models:
        state_dict = create_test_state_dict(model_type)
        num_params = sum(t.numel() for t in state_dict.values())
        bundle_count = (num_params + 4095) // 4096  # Ceiling division

        for i, (config_name, config) in enumerate(configs):
            for backend in backends:
                if verbose:
                    print(f"\n--- Test {model_type} config {i} ({config_name}) backend={backend} ---")

                passed, metrics = validate_against_reference(
                    state_dict, config, bundle_count, backend=backend, verbose=verbose
                )
                if not passed:
                    all_passed = False

    # Test L2 norm
    if verbose:
        print("\n--- L2 Norm Test ---")
    flat = torch.randn(10000)
    if not validate_l2_norm(flat, verbose=verbose):
        all_passed = False

    # Test slot packing
    if verbose:
        print("\n--- Slot Packing Test ---")
    flat = torch.randn(5000)
    if not validate_slot_packing(flat, 2, 4096, verbose=verbose):
        all_passed = False

    if not validate_edge_cases(verbose=verbose):
        all_passed = False

    if not validate_large_scale_readiness(verbose=verbose):
        all_passed = False

    if verbose:
        print(f"\n{'='*50}")
        print(f"Validation {'PASSED' if all_passed else 'FAILED'}")

    return all_passed


if __name__ == "__main__":
    success = run_validation_suite(verbose=True)
    sys.exit(0 if success else 1)
