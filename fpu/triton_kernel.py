"""
Triton implementation of the fused private update operator.

This kernel fuses multiple operations into a single GPU pass:
1. L2 norm computation (reduction)
2. Gradient clipping
3. Client weight scaling
4. DP noise injection
5. Quantization
6. Slot packing
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from .types import FusedUpdateConfig, FusedUpdateOutput, KernelMetrics


# ============================================================================
# Triton Kernel: L2 Norm Reduction (Two-Pass for Numerical Stability)
# ============================================================================

@triton.jit
def _l2_norm_partial_kernel(
    x_ptr,
    partial_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Pass 1: Compute partial sum of squares per block.
    Each program instance writes one partial sum to avoid atomic_add
    non-determinism and float32 accumulation precision loss.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load and compute squares
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    squared = x * x

    # Sum within this block and write to partial sums buffer
    block_sum = tl.sum(squared, axis=0)
    tl.store(partial_sums_ptr + pid, block_sum)


@triton.jit
def _l2_norm_reduce_kernel(
    partial_sums_ptr,
    out_ptr,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Pass 2: Reduce partial sums into a single total sum of squares.
    Single program instance performs the final reduction.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks

    partial = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partial, axis=0)
    tl.store(out_ptr, total)


# ============================================================================
# Triton Kernel: Scale and Clip (scalar parameters, not pointers — fix #10)
# ============================================================================

@triton.jit
def _scale_and_clip_kernel(
    x_ptr,
    out_ptr,
    clip_scale,
    client_weight,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply pre-computed clipping scale and client weight.
    out = x * clip_scale * client_weight

    clip_scale is pre-computed on the host as min(1.0, clip_norm / (l2_norm + eps)).
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply combined scaling
    combined_scale = clip_scale * client_weight
    out = x * combined_scale

    # Store
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _add_noise_kernel(
    x_ptr,
    noise_ptr,
    out_ptr,
    n_elements,
    noise_std,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Add pre-generated Gaussian noise to gradient.
    out = x + noise * noise_std
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    noise = tl.load(noise_ptr + offsets, mask=mask, other=0.0)

    # Add scaled noise
    out = x + noise * noise_std

    # Store
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================================
# Triton Kernel: Quantize (fix #4 — added scale/inv_scale params)
# ============================================================================

@triton.jit
def _quantize_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    inv_scale,
    scale,
    qmin,
    qmax,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Symmetric quantization with proper scale handling.
    quantized = clamp(round(x / scale), qmin, qmax)
    dequantized = quantized * scale

    Args:
        inv_scale: 1.0 / scale, pre-computed for efficiency (= qmax / abs_max)
        scale: abs_max / qmax, used for dequantization
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Quantize: divide by scale, round, clamp
    scaled = x * inv_scale
    quantized = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.round(scaled), qmin), qmax)

    # Dequantize: multiply back by scale
    dequantized = quantized * scale

    # Store
    tl.store(out_ptr + offsets, dequantized, mask=mask)


@triton.jit
def _pack_to_slots_kernel(
    flat_ptr,
    packed_ptr,
    num_params,
    slot_capacity,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Pack flat gradient into slot buffers.
    Maps flat index -> (bundle_id, slot_offset) and writes to packed buffer.

    Note: For the current linear LayoutPlan, dest == offsets (identity mapping).
    The bundle_id / slot_offset decomposition is kept for future non-linear
    LayoutPlan support where layers may be reordered across bundles.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Only process up to num_params
    valid = mask & (offsets < num_params)

    # Load flat values
    x = tl.load(flat_ptr + offsets, mask=valid, other=0.0)

    # Compute bundle_id and slot_offset (identity for linear layout,
    # but kept for extensibility to non-contiguous LayoutPlans)
    bundle_ids = offsets // slot_capacity
    slot_offsets = offsets % slot_capacity

    # Compute destination in packed buffer [bundles, slots]
    dest = bundle_ids * slot_capacity + slot_offsets

    # Store to packed buffer
    tl.store(packed_ptr + dest, x, mask=valid)


# ============================================================================
# Combined Kernel: Full Pipeline (fix #1 — clip_scale now a parameter)
# ============================================================================

@triton.jit
def _fused_update_kernel(
    # Input
    flat_ptr,
    noise_ptr,
    # Output
    packed_ptr,
    # Pre-computed scalars
    clip_scale,
    client_weight,
    noise_std,
    # Quantization params (pre-computed on host)
    quant_inv_scale,
    quant_scale,
    qmin,
    qmax,
    use_quant: tl.constexpr,
    # Dimensions
    num_params,
    total_slots,
    slot_capacity,
    # Kernel config
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully fused kernel combining clip + scale + noise + quant + pack.

    Note: L2 norm must be computed separately (reduction operation).
    clip_scale = min(1.0, clip_norm / (l2_norm + eps)) is pre-computed
    on the host and passed as a scalar parameter.

    Pipeline per element:
    1. x = flat[i] * clip_scale * client_weight
    2. x = x + noise[i] * noise_std  (if noise_std > 0)
    3. x = dequant(quant(x))          (if use_quant)
    4. packed[bundle_id, slot_offset] = x
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Process up to total_slots (includes padding)
    mask = offsets < total_slots

    # Only load values for actual parameters
    param_mask = offsets < num_params

    # Load flat values (0 for padding)
    x = tl.load(flat_ptr + offsets, mask=param_mask, other=0.0)

    # Apply clip_scale * client_weight
    combined_scale = clip_scale * client_weight
    x = x * combined_scale

    # Add noise if enabled
    if noise_std > 0:
        noise = tl.load(noise_ptr + offsets, mask=param_mask, other=0.0)
        x = x + noise * noise_std

    # Quantize if enabled
    if use_quant:
        scaled = x * quant_inv_scale
        quantized = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.round(scaled), qmin), qmax)
        x = quantized * quant_scale

    # Compute slot positions (identity for linear layout)
    bundle_ids = offsets // slot_capacity
    slot_offsets = offsets % slot_capacity
    dest = bundle_ids * slot_capacity + slot_offsets

    # Store to packed buffer
    tl.store(packed_ptr + dest, x, mask=mask)


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def compute_l2_norm_triton(flat: torch.Tensor) -> float:
    """Compute L2 norm using two-pass Triton reduction for numerical stability.

    Pass 1: Each block computes a partial sum of squares.
    Pass 2: A single block reduces all partial sums.
    This avoids atomic_add non-determinism and float32 accumulation errors.
    """
    n_elements = flat.numel()
    if n_elements == 0:
        return 0.0

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    # Allocate partial sums buffer (one per block)
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=flat.device)

    # Pass 1: compute partial sums
    grid = (num_blocks,)
    _l2_norm_partial_kernel[grid](
        flat,
        partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Pass 2: reduce partial sums
    # Find the smallest power-of-2 >= num_blocks for the reduction kernel
    reduce_block = 1
    while reduce_block < num_blocks:
        reduce_block *= 2
    # Cap at a reasonable Triton block size
    reduce_block = min(reduce_block, 65536)

    sum_sq = torch.zeros(1, dtype=torch.float32, device=flat.device)
    _l2_norm_reduce_kernel[(1,)](
        partial_sums,
        sum_sq,
        num_blocks,
        BLOCK_SIZE=reduce_block,
    )

    # Compute final norm on CPU
    return math.sqrt(float(sum_sq.item()))


def fused_private_update_triton(
    state_dict: Dict[str, torch.Tensor],
    config: FusedUpdateConfig,
    bundle_count: int,
    slot_capacity: int = 4096,
    device: torch.device = torch.device("cuda"),
    measure_time: bool = False,
) -> FusedUpdateOutput:
    """
    Triton implementation of fused private update operator.

    Uses dedicated Triton kernels for each pipeline stage:
    1. L2 norm (two-pass reduction)
    2. Scale + clip (Triton kernel)
    3. Noise injection (Triton kernel)
    4. Quantization (Triton kernel)
    5. Slot packing (Triton kernel)

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
    BLOCK_SIZE = 1024

    if measure_time:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

    # Step 1: Flatten state dict
    parts = []
    for _, tensor in state_dict.items():
        parts.append(tensor.detach().reshape(-1).float())
    if not parts:
        flat = torch.empty(0, dtype=torch.float32, device=device)
    else:
        flat = torch.cat(parts).to(device)

    num_params = flat.numel()
    total_slots = bundle_count * slot_capacity
    metrics.input_bytes = flat.numel() * flat.element_size()

    if measure_time:
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        metrics.flatten_ms = (t1 - t0) * 1000

    # Step 2: Compute L2 norm (Triton two-pass reduction)
    l2_norm = compute_l2_norm_triton(flat)
    metrics.l2_norm = l2_norm

    # Compute clip scale on host
    clip_scale = min(1.0, config.clip_norm / (l2_norm + 1e-8))
    metrics.clip_scale = clip_scale

    if measure_time:
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        metrics.clip_ms = (t2 - t1) * 1000

    # Step 3: Apply clipping and client weight (Triton kernel)
    flat_scaled = torch.empty_like(flat)
    grid = (triton.cdiv(num_params, BLOCK_SIZE),)
    _scale_and_clip_kernel[grid](
        flat,
        flat_scaled,
        clip_scale,
        config.client_weight,
        num_params,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if measure_time:
        torch.cuda.synchronize(device)
        t3 = time.perf_counter()

    # Step 4: Add DP noise if enabled (Triton kernel)
    if config.noise_multiplier > 0:
        noise_std = config.noise_multiplier * config.clip_norm
        metrics.noise_std = noise_std

        if config.noise_seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(config.noise_seed)
            noise = torch.randn(flat_scaled.shape, generator=gen, device=device, dtype=torch.float32)
        else:
            noise = torch.randn_like(flat_scaled)

        noisy_out = torch.empty_like(flat_scaled)
        _add_noise_kernel[grid](
            flat_scaled,
            noise,
            noisy_out,
            num_params,
            noise_std,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        flat_scaled = noisy_out

    if measure_time:
        torch.cuda.synchronize(device)
        t4 = time.perf_counter()
        metrics.noise_ms = (t4 - t3) * 1000

    # Step 5: Quantization (Triton kernel)
    if config.quant_bits > 0:
        qmin = -(2 ** (config.quant_bits - 1))
        qmax_val = 2 ** (config.quant_bits - 1) - 1
        abs_max = torch.max(torch.abs(flat_scaled))
        if abs_max > 1e-8:
            scale = float(abs_max.item()) / qmax_val
            inv_scale = 1.0 / scale

            quant_out = torch.empty_like(flat_scaled)
            _quantize_kernel[grid](
                flat_scaled,
                quant_out,
                num_params,
                inv_scale,
                scale,
                qmin,
                qmax_val,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            flat_scaled = quant_out

    if measure_time:
        torch.cuda.synchronize(device)
        t5 = time.perf_counter()
        metrics.quant_ms = (t5 - t4) * 1000

    # Step 6: Pack to slot buffers (Triton kernel)
    packed = torch.zeros(total_slots, dtype=torch.float32, device=device)
    pack_grid = (triton.cdiv(total_slots, BLOCK_SIZE),)
    _pack_to_slots_kernel[pack_grid](
        flat_scaled,
        packed,
        num_params,
        slot_capacity,
        total_slots,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    packed = packed.reshape(bundle_count, slot_capacity)
    metrics.output_bytes = packed.numel() * packed.element_size()

    if measure_time:
        torch.cuda.synchronize(device)
        t6 = time.perf_counter()
        metrics.pack_ms = (t6 - t5) * 1000
        metrics.total_ms = (t6 - t0) * 1000

    return FusedUpdateOutput(
        packed_slots=packed,
        bundle_count=bundle_count,
        slot_capacity=slot_capacity,
        num_params=num_params,
        metadata=metrics.to_dict() if measure_time else {
            "l2_norm": l2_norm,
            "clip_scale": clip_scale,
            "noise_std": metrics.noise_std,
        },
    )


def fused_private_update_triton_v2(
    flat: torch.Tensor,
    config: FusedUpdateConfig,
    bundle_count: int,
    slot_capacity: int = 4096,
    measure_time: bool = False,
) -> FusedUpdateOutput:
    """
    Optimized Triton implementation using the fully-fused kernel.

    This version uses _fused_update_kernel to combine clip + scale + noise +
    quant + pack into a single kernel launch (after the L2 norm reduction).

    Args:
        flat: Pre-flattened gradient tensor on GPU [num_params]
        config: Fused update configuration
        bundle_count: Number of ciphertext bundles
        slot_capacity: Slots per bundle
        measure_time: Whether to measure timing

    Returns:
        FusedUpdateOutput with packed slots and metadata
    """
    metrics = KernelMetrics()
    device = flat.device
    BLOCK_SIZE = 1024

    if measure_time:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

    num_params = flat.numel()
    total_slots = bundle_count * slot_capacity
    metrics.input_bytes = flat.numel() * flat.element_size()

    # Step 1: Compute L2 norm (two-pass Triton reduction)
    l2_norm = compute_l2_norm_triton(flat)
    metrics.l2_norm = l2_norm

    # Compute clip scale on host
    clip_scale = min(1.0, config.clip_norm / (l2_norm + 1e-8))
    metrics.clip_scale = clip_scale

    if measure_time:
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        metrics.clip_ms = (t1 - t0) * 1000

    # Prepare noise buffer
    noise_std = 0.0
    if config.noise_multiplier > 0:
        noise_std = config.noise_multiplier * config.clip_norm
        metrics.noise_std = noise_std

        if config.noise_seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(config.noise_seed)
            noise = torch.randn(total_slots, generator=gen, device=device, dtype=torch.float32)
        else:
            noise = torch.randn(total_slots, device=device, dtype=torch.float32)
    else:
        # Dummy buffer (not read when noise_std == 0)
        noise = torch.empty(1, device=device, dtype=torch.float32)

    # Prepare quantization params
    use_quant = config.quant_bits > 0
    quant_scale = 1.0
    quant_inv_scale = 1.0
    qmin = 0
    qmax_val = 0
    if use_quant:
        qmin = -(2 ** (config.quant_bits - 1))
        qmax_val = 2 ** (config.quant_bits - 1) - 1
        # Need abs_max for scale — requires a quick pass
        # Apply clip_scale first to get the right magnitude
        abs_max_val = float(torch.max(torch.abs(flat)).item()) * clip_scale * abs(config.client_weight)
        if abs_max_val > 1e-8:
            quant_scale = abs_max_val / qmax_val
            quant_inv_scale = 1.0 / quant_scale

    if measure_time:
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()

    # Step 2: Fused kernel — clip + scale + noise + quant + pack
    packed = torch.zeros(total_slots, dtype=torch.float32, device=device)

    grid = (triton.cdiv(total_slots, BLOCK_SIZE),)
    _fused_update_kernel[grid](
        flat,
        noise,
        packed,
        clip_scale,
        config.client_weight,
        noise_std,
        quant_inv_scale,
        quant_scale,
        qmin,
        qmax_val,
        use_quant,
        num_params,
        total_slots,
        slot_capacity,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    packed = packed.reshape(bundle_count, slot_capacity)
    metrics.output_bytes = packed.numel() * packed.element_size()

    if measure_time:
        torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        metrics.pack_ms = (t3 - t2) * 1000
        metrics.total_ms = (t3 - t0) * 1000

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
