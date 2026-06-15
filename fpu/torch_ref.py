"""
PyTorch reference implementation of the fused private update operator.

This serves as the ground truth for validating the Triton kernel.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch


from .types import (
    FusedUpdateConfig,
    FusedUpdateOutput,
    KernelMetrics,
    validate_fused_update_args,
)


def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state dict into a single 1D tensor."""
    parts = []
    for _, tensor in state_dict.items():
        parts.append(tensor.detach().reshape(-1).float())
    if not parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(parts)


def compute_l2_norm(flat: torch.Tensor) -> torch.Tensor:
    """Compute L2 norm of a flat tensor."""
    return torch.linalg.vector_norm(flat)


def clip_gradients(flat: torch.Tensor, clip_norm: float) -> Tuple[torch.Tensor, float, float]:
    """
    Apply L2 clipping to gradients.

    Returns:
        clipped: Clipped gradient tensor
        l2_norm: Original L2 norm
        clip_scale: Scale factor applied
    """
    l2_norm = compute_l2_norm(flat)
    clip_scale = torch.clamp(clip_norm / (l2_norm + 1e-8), max=1.0)
    clipped = flat * clip_scale
    return clipped, float(l2_norm.item()), float(clip_scale.item())


def add_dp_noise(
    flat: torch.Tensor,
    noise_multiplier: float,
    clip_norm: float,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Add calibrated Gaussian noise for differential privacy.

    noise ~ N(0, (noise_multiplier * clip_norm)^2)

    Returns:
        noisy: Noisy gradient tensor
        noise_std: Standard deviation of noise added
    """
    if noise_multiplier <= 0:
        return flat, 0.0

    noise_std = noise_multiplier * clip_norm

    if seed is not None:
        gen = torch.Generator(device=flat.device)
        gen.manual_seed(seed)
        noise = torch.randn(flat.shape, generator=gen, device=flat.device, dtype=flat.dtype) * noise_std
    else:
        noise = torch.randn_like(flat) * noise_std

    return flat + noise, noise_std


def quantize_tensor(flat: torch.Tensor, quant_bits: int) -> torch.Tensor:
    """
    Symmetric quantization to specified bit width.

    Maps float values to [-2^(bits-1), 2^(bits-1)-1] range.
    """
    if quant_bits <= 0:
        return flat
    if quant_bits == 1 or quant_bits >= 32:
        raise ValueError("quant_bits must be 0 or an integer in [2, 31]")
    if flat.numel() == 0:
        return flat

    qmin = -(2 ** (quant_bits - 1))
    qmax = 2 ** (quant_bits - 1) - 1

    # Compute scale per tensor (symmetric)
    abs_max = torch.max(torch.abs(flat))
    if abs_max < 1e-8:
        return flat

    scale = abs_max / qmax

    # Quantize and dequantize
    quantized = torch.clamp(torch.round(flat / scale), qmin, qmax)
    dequantized = quantized * scale

    return dequantized


def pack_to_slots(
    flat: torch.Tensor,
    bundle_count: int,
    slot_capacity: int,
    num_params: int,
) -> torch.Tensor:
    """
    Pack flat gradient into slot buffers according to LayoutPlan.

    Args:
        flat: 1D gradient tensor [num_params]
        bundle_count: Number of ciphertext bundles
        slot_capacity: Slots per bundle (e.g., 4096 for CKKS)
        num_params: Actual number of parameters (before padding)

    Returns:
        packed: [bundle_count, slot_capacity] tensor
    """
    device = flat.device
    total_slots = validate_fused_update_args(
        FusedUpdateConfig(), bundle_count, slot_capacity, num_params
    )

    # Create padded tensor
    padded = torch.zeros(total_slots, dtype=torch.float32, device=device)

    # Copy parameters into padded buffer
    copy_len = min(flat.numel(), num_params)
    padded[:copy_len] = flat[:copy_len]

    # Reshape into bundles
    packed = padded.reshape(bundle_count, slot_capacity)

    return packed


def fused_private_update_ref(
    state_dict: Dict[str, torch.Tensor],
    config: FusedUpdateConfig,
    bundle_count: int,
    slot_capacity: int = 4096,
    device: torch.device = torch.device("cuda"),
    measure_time: bool = False,
) -> FusedUpdateOutput:
    """
    Reference implementation of fused private update operator.

    This function performs all operations sequentially using PyTorch ops.
    Use this as ground truth for validating the Triton kernel.

    Args:
        state_dict: Model update as state dict
        config: Fused update configuration
        bundle_count: Number of ciphertext bundles
        slot_capacity: Slots per bundle
        device: Target device
        measure_time: Whether to measure timing

    Returns:
        FusedUpdateOutput with packed slots and metadata
    """
    metrics = KernelMetrics()
    metrics.backend = "torch_ref"

    if measure_time:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

    # Step 1: Flatten
    flat = flatten_state_dict(state_dict).to(device)
    num_params = flat.numel()
    validate_fused_update_args(config, bundle_count, slot_capacity, num_params)
    metrics.input_bytes = flat.numel() * flat.element_size()

    if measure_time:
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        metrics.flatten_ms = (t1 - t0) * 1000

    # Step 2: L2 clipping
    flat, l2_norm, clip_scale = clip_gradients(flat, config.clip_norm)
    metrics.l2_norm = l2_norm
    metrics.clip_scale = clip_scale

    if measure_time:
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        metrics.clip_ms = (t2 - t1) * 1000

    # Step 3: Client weight
    flat = flat * config.client_weight

    # Step 4: DP noise
    flat, noise_std = add_dp_noise(
        flat,
        config.noise_multiplier,
        config.clip_norm,
        config.noise_seed,
    )
    metrics.noise_std = noise_std

    if measure_time:
        torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        metrics.noise_ms = (t3 - t2) * 1000

    # Step 5: Quantization
    if config.quant_bits > 0:
        flat = quantize_tensor(flat, config.quant_bits)

    if measure_time:
        torch.cuda.synchronize(device)
        t4 = time.perf_counter()
        metrics.quant_ms = (t4 - t3) * 1000

    # Step 6: Pack to slots
    packed = pack_to_slots(flat, bundle_count, slot_capacity, num_params)
    metrics.output_bytes = packed.numel() * packed.element_size()

    if measure_time:
        torch.cuda.synchronize(device)
        t5 = time.perf_counter()
        metrics.pack_ms = (t5 - t4) * 1000
        metrics.total_ms = (t5 - t0) * 1000

    return FusedUpdateOutput(
        packed_slots=packed,
        bundle_count=bundle_count,
        slot_capacity=slot_capacity,
        num_params=num_params,
        metadata=metrics.to_dict() if measure_time else {
            "l2_norm": l2_norm,
            "clip_scale": clip_scale,
            "noise_std": noise_std,
        },
    )


def fused_private_update_batch_ref(
    updates: List[Dict[str, torch.Tensor]],
    weights: List[float],
    config: FusedUpdateConfig,
    bundle_count: int,
    slot_capacity: int = 4096,
    device: torch.device = torch.device("cuda"),
) -> FusedUpdateOutput:
    """
    Batch reference implementation: process multiple client updates.

    Applies clipping, noise, and weighting to each client, then aggregates.

    Args:
        updates: List of client update state dicts
        weights: Client weights
        config: Fused update configuration
        bundle_count: Number of ciphertext bundles
        slot_capacity: Slots per bundle
        device: Target device

    Returns:
        Aggregated FusedUpdateOutput
    """
    if len(updates) != len(weights):
        raise ValueError(f"updates and weights must have same length: {len(updates)} vs {len(weights)}")

    # Process each client update
    packed_list = []
    first_num_params = None
    for update, weight in zip(updates, weights):
        client_config = FusedUpdateConfig(
            clip_norm=config.clip_norm,
            noise_multiplier=config.noise_multiplier,
            quant_bits=config.quant_bits,
            client_weight=weight,
            ckks_scale=config.ckks_scale,
            noise_seed=config.noise_seed,
        )
        output = fused_private_update_ref(
            update, client_config, bundle_count, slot_capacity, device
        )
        packed_list.append(output.packed_slots)
        if first_num_params is None:
            first_num_params = output.num_params

    # Aggregate (sum) all client packed slots
    aggregated = torch.stack(packed_list).sum(dim=0)

    return FusedUpdateOutput(
        packed_slots=aggregated,
        bundle_count=bundle_count,
        slot_capacity=slot_capacity,
        num_params=first_num_params,
        metadata={
            "num_clients": len(updates),
            "weights": weights,
        },
    )
