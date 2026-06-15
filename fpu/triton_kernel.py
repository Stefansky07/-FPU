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

from .types import (
    FusedUpdateConfig,
    FusedUpdateOutput,
    KernelMetrics,
    validate_fused_update_args,
)


L2_BLOCK_SIZE = 1024
MAX_REDUCE_BLOCK_SIZE = 65536
STAT_BLOCK_SIZE = 1024

# Large synthetic runs can exceed the signed int32 index range. Keep tensor
# offsets int64 inside Triton kernels so pointer arithmetic stays valid.


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
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_elements

    # Load and compute squares
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    squared = x * x

    # Sum within this block and write to partial sums buffer
    block_sum = tl.sum(squared, axis=0)
    tl.store(partial_sums_ptr + pid, block_sum)


@triton.jit
def _l2_norm_abs_max_partial_kernel(
    x_ptr,
    partial_sums_ptr,
    partial_max_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Pass 1 variant for quantized no-noise runs: compute both L2 partial sums
    and raw abs-max partials while the input is already being streamed.
    """
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(x * x, axis=0)
    block_max = tl.max(tl.abs(x), axis=0)

    tl.store(partial_sums_ptr + pid, block_sum)
    tl.store(partial_max_ptr + pid, block_max)


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
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < num_blocks

    partial = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partial, axis=0)
    tl.store(out_ptr, total)


@triton.jit
def _sum_max_reduce_blocks_kernel(
    partial_sums_ptr,
    partial_max_ptr,
    next_sums_ptr,
    next_max_ptr,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce sum and max partial buffers together."""
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < num_blocks

    partial_sum = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    partial_max = tl.load(partial_max_ptr + offsets, mask=mask, other=0.0)

    tl.store(next_sums_ptr + pid, tl.sum(partial_sum, axis=0))
    tl.store(next_max_ptr + pid, tl.max(partial_max, axis=0))


@triton.jit
def _l2_norm_abs_max_reduce_kernel(
    partial_sums_ptr,
    partial_max_ptr,
    sum_out_ptr,
    max_out_ptr,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Final single-program reduction for L2 sum and raw abs max."""
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < num_blocks

    partial_sum = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    partial_max = tl.load(partial_max_ptr + offsets, mask=mask, other=0.0)

    tl.store(sum_out_ptr, tl.sum(partial_sum, axis=0))
    tl.store(max_out_ptr, tl.max(partial_max, axis=0))


@triton.jit
def _sum_reduce_blocks_kernel(
    partial_sums_ptr,
    out_ptr,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce a partial-sum buffer into another partial-sum buffer."""
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < num_blocks

    partial = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partial, axis=0)
    tl.store(out_ptr + pid, total)


@triton.jit
def _abs_max_transform_partial_kernel(
    x_ptr,
    noise_ptr,
    partial_max_ptr,
    n_elements,
    clip_scale,
    client_weight,
    noise_std,
    USE_NOISE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute per-block abs max after clip/weight/noise without materializing
    the transformed vector.
    """
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x * (clip_scale * client_weight)
    if USE_NOISE:
        noise = tl.load(noise_ptr + offsets, mask=mask, other=0.0)
        x = x + noise * noise_std

    block_max = tl.max(tl.abs(x), axis=0)
    tl.store(partial_max_ptr + pid, block_max)


@triton.jit
def _max_reduce_blocks_kernel(
    partial_max_ptr,
    out_ptr,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce a partial max buffer into another partial max buffer."""
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < num_blocks

    partial = tl.load(partial_max_ptr + offsets, mask=mask, other=0.0)
    block_max = tl.max(partial, axis=0)
    tl.store(out_ptr + pid, block_max)


@triton.jit
def _max_reduce_kernel(
    partial_max_ptr,
    out_ptr,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Final single-program max reduction."""
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < num_blocks

    partial = tl.load(partial_max_ptr + offsets, mask=mask, other=0.0)
    total = tl.max(partial, axis=0)
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
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
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
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
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
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
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
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
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
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)

    param_mask = offsets < num_params
    store_mask = offsets < total_slots
    safe_offsets = tl.minimum(offsets, num_params - 1)

    # Load flat values and synthesize zero-valued padding lanes. This lets the
    # kernel initialize the tail slots without a separate full-buffer zero pass.
    x = tl.load(flat_ptr + safe_offsets, mask=param_mask, other=0.0)

    # Apply clip_scale * client_weight
    combined_scale = clip_scale * client_weight
    x = x * combined_scale

    # Add noise if enabled
    if noise_std > 0:
        noise = tl.load(noise_ptr + safe_offsets, mask=param_mask, other=0.0)
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
    tl.store(packed_ptr + dest, x, mask=store_mask)


@triton.jit
def _fused_update_int8_kernel(
    # Input
    flat_ptr,
    noise_ptr,
    # Output
    packed_ptr,
    # Pre-computed scalars
    clip_scale,
    client_weight,
    noise_std,
    # Quantization params
    quant_inv_scale,
    qmin,
    qmax,
    # Dimensions
    num_params,
    total_slots,
    slot_capacity,
    # Kernel config
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused clip + scale + noise + int8 quant + pack.

    Unlike _fused_update_kernel, this stores the quantized integer payload
    directly. Padding slots are written as zero.
    """
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)

    param_mask = offsets < num_params
    store_mask = offsets < total_slots
    safe_offsets = tl.minimum(offsets, num_params - 1)

    x = tl.load(flat_ptr + safe_offsets, mask=param_mask, other=0.0)
    x = x * (clip_scale * client_weight)

    if noise_std > 0:
        noise = tl.load(noise_ptr + safe_offsets, mask=param_mask, other=0.0)
        x = x + noise * noise_std

    scaled = x * quant_inv_scale
    quantized = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.round(scaled), qmin), qmax)
    quantized = tl.where(param_mask, quantized, 0.0)

    bundle_ids = offsets // slot_capacity
    slot_offsets = offsets % slot_capacity
    dest = bundle_ids * slot_capacity + slot_offsets

    # Cast to physical int8 for compressed output
    quantized_i8 = quantized.to(tl.int8)
    tl.store(packed_ptr + dest, quantized_i8, mask=store_mask)


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

    num_blocks = triton.cdiv(n_elements, L2_BLOCK_SIZE)

    # Allocate partial sums buffer (one per block)
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=flat.device)

    # Pass 1: compute partial sums
    grid = (num_blocks,)
    _l2_norm_partial_kernel[grid](
        flat,
        partial_sums,
        n_elements,
        BLOCK_SIZE=L2_BLOCK_SIZE,
    )

    # Pass 2+: reduce partial sums until one block can cover the whole buffer.
    current = partial_sums
    current_blocks = num_blocks
    while current_blocks > MAX_REDUCE_BLOCK_SIZE:
        next_blocks = triton.cdiv(current_blocks, MAX_REDUCE_BLOCK_SIZE)
        next_sums = torch.empty(next_blocks, dtype=torch.float32, device=flat.device)
        _sum_reduce_blocks_kernel[(next_blocks,)](
            current,
            next_sums,
            current_blocks,
            BLOCK_SIZE=MAX_REDUCE_BLOCK_SIZE,
        )
        current = next_sums
        current_blocks = next_blocks

    # Final reduction in a single program. Triton block sizes are kept power-of-2.
    reduce_block = 1
    while reduce_block < current_blocks:
        reduce_block *= 2

    sum_sq = torch.zeros(1, dtype=torch.float32, device=flat.device)
    _l2_norm_reduce_kernel[(1,)](
        current,
        sum_sq,
        current_blocks,
        BLOCK_SIZE=reduce_block,
    )

    # Compute final norm on CPU
    return math.sqrt(float(sum_sq.item()))


def compute_l2_norm_and_abs_max_triton(flat: torch.Tensor) -> Tuple[float, float]:
    """Compute L2 norm and raw abs max in one streamed pass.

    This is useful for quantized runs without DP noise. The quantization scale
    needs abs(max(flat * scalar)); because clipping and client weighting are
    scalar transforms in that case, the raw abs max can be reused after the L2
    norm determines the clipping scale.
    """
    n_elements = flat.numel()
    if n_elements == 0:
        return 0.0, 0.0

    num_blocks = triton.cdiv(n_elements, L2_BLOCK_SIZE)
    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device=flat.device)
    partial_max = torch.empty(num_blocks, dtype=torch.float32, device=flat.device)

    _l2_norm_abs_max_partial_kernel[(num_blocks,)](
        flat,
        partial_sums,
        partial_max,
        n_elements,
        BLOCK_SIZE=L2_BLOCK_SIZE,
    )

    current_sums = partial_sums
    current_max = partial_max
    current_blocks = num_blocks
    while current_blocks > MAX_REDUCE_BLOCK_SIZE:
        next_blocks = triton.cdiv(current_blocks, MAX_REDUCE_BLOCK_SIZE)
        next_sums = torch.empty(next_blocks, dtype=torch.float32, device=flat.device)
        next_max = torch.empty(next_blocks, dtype=torch.float32, device=flat.device)
        _sum_max_reduce_blocks_kernel[(next_blocks,)](
            current_sums,
            current_max,
            next_sums,
            next_max,
            current_blocks,
            BLOCK_SIZE=MAX_REDUCE_BLOCK_SIZE,
        )
        current_sums = next_sums
        current_max = next_max
        current_blocks = next_blocks

    reduce_block = 1
    while reduce_block < current_blocks:
        reduce_block *= 2

    sum_sq = torch.empty(1, dtype=torch.float32, device=flat.device)
    raw_abs_max = torch.empty(1, dtype=torch.float32, device=flat.device)
    _l2_norm_abs_max_reduce_kernel[(1,)](
        current_sums,
        current_max,
        sum_sq,
        raw_abs_max,
        current_blocks,
        BLOCK_SIZE=reduce_block,
    )

    return math.sqrt(float(sum_sq.item())), float(raw_abs_max.item())


def _l2_norm_launch_count(n_elements: int) -> int:
    """Return the number of Triton launches used by compute_l2_norm_triton."""
    if n_elements == 0:
        return 0
    current_blocks = triton.cdiv(n_elements, L2_BLOCK_SIZE)
    launches = 1
    while current_blocks > MAX_REDUCE_BLOCK_SIZE:
        current_blocks = triton.cdiv(current_blocks, MAX_REDUCE_BLOCK_SIZE)
        launches += 1
    return launches + 1


def compute_transformed_abs_max_triton(
    flat: torch.Tensor,
    noise: torch.Tensor,
    n_elements: int,
    clip_scale: float,
    client_weight: float,
    noise_std: float,
) -> float:
    """
    Compute max(abs(flat * clip_scale * client_weight + noise * noise_std)).

    This is used by the fused v2 quantization path so correctness does not
    require materializing a full transformed temporary tensor.
    """
    if n_elements == 0:
        return 0.0
    if n_elements > flat.numel():
        raise ValueError(f"n_elements={n_elements} exceeds flat.numel()={flat.numel()}")
    use_noise = noise_std > 0
    if use_noise and noise.numel() < n_elements:
        raise ValueError(f"noise has {noise.numel()} elements but {n_elements} are required")

    num_blocks = triton.cdiv(n_elements, STAT_BLOCK_SIZE)
    partial_max = torch.empty(num_blocks, dtype=torch.float32, device=flat.device)
    _abs_max_transform_partial_kernel[(num_blocks,)](
        flat,
        noise,
        partial_max,
        n_elements,
        clip_scale,
        client_weight,
        noise_std,
        USE_NOISE=use_noise,
        BLOCK_SIZE=STAT_BLOCK_SIZE,
    )

    current = partial_max
    current_blocks = num_blocks
    while current_blocks > MAX_REDUCE_BLOCK_SIZE:
        next_blocks = triton.cdiv(current_blocks, MAX_REDUCE_BLOCK_SIZE)
        next_max = torch.empty(next_blocks, dtype=torch.float32, device=flat.device)
        _max_reduce_blocks_kernel[(next_blocks,)](
            current,
            next_max,
            current_blocks,
            BLOCK_SIZE=MAX_REDUCE_BLOCK_SIZE,
        )
        current = next_max
        current_blocks = next_blocks

    reduce_block = 1
    while reduce_block < current_blocks:
        reduce_block *= 2

    out = torch.empty(1, dtype=torch.float32, device=flat.device)
    _max_reduce_kernel[(1,)](
        current,
        out,
        current_blocks,
        BLOCK_SIZE=reduce_block,
    )
    return float(out.item())


def _stat_reduce_launch_count(n_elements: int) -> int:
    """Return Triton launch count for compute_transformed_abs_max_triton."""
    if n_elements == 0:
        return 0
    current_blocks = triton.cdiv(n_elements, STAT_BLOCK_SIZE)
    launches = 1
    while current_blocks > MAX_REDUCE_BLOCK_SIZE:
        current_blocks = triton.cdiv(current_blocks, MAX_REDUCE_BLOCK_SIZE)
        launches += 1
    return launches + 1


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
    metrics.backend = "triton"
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
    total_slots = validate_fused_update_args(config, bundle_count, slot_capacity, num_params)
    metrics.input_bytes = flat.numel() * flat.element_size()
    metrics.logical_quantized_payload_bytes = (
        (num_params * config.quant_bits + 7) // 8 if config.quant_bits > 0 else 0
    )
    metrics.kernel_launch_count = _l2_norm_launch_count(num_params)

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
    if num_params > 0:
        grid = (triton.cdiv(num_params, BLOCK_SIZE),)
        _scale_and_clip_kernel[grid](
            flat,
            flat_scaled,
            clip_scale,
            config.client_weight,
            num_params,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        metrics.kernel_launch_count += 1
    else:
        grid = (0,)

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
        if num_params > 0:
            _add_noise_kernel[grid](
                flat_scaled,
                noise,
                noisy_out,
                num_params,
                noise_std,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            metrics.kernel_launch_count += 1
        flat_scaled = noisy_out

    if measure_time:
        torch.cuda.synchronize(device)
        t4 = time.perf_counter()
        metrics.noise_ms = (t4 - t3) * 1000

    # Step 5: Quantization (Triton kernel)
    if config.quant_bits > 0 and num_params > 0:
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
            metrics.kernel_launch_count += 1
            flat_scaled = quant_out

    if measure_time:
        torch.cuda.synchronize(device)
        t5 = time.perf_counter()
        metrics.quant_ms = (t5 - t4) * 1000

    # Step 6: Pack to slot buffers (Triton kernel)
    packed = torch.zeros(total_slots, dtype=torch.float32, device=device)
    if total_slots > 0 and num_params > 0:
        pack_grid = (triton.cdiv(num_params, BLOCK_SIZE),)
        _pack_to_slots_kernel[pack_grid](
            flat_scaled,
            packed,
            num_params,
            slot_capacity,
            num_params,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        metrics.kernel_launch_count += 1

    packed = packed.reshape(bundle_count, slot_capacity)
    metrics.output_bytes = packed.numel() * packed.element_size()
    metrics.output_dtype = str(packed.dtype).replace("torch.", "")

    if measure_time:
        torch.cuda.synchronize(device)
        t6 = time.perf_counter()
        metrics.pack_ms = (t6 - t5) * 1000
        metrics.fused_operator_ms = (t6 - t2) * 1000
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
            "kernel_launch_count": metrics.kernel_launch_count,
            "backend": metrics.backend,
        },
    )


def fused_private_update_triton_v2(
    flat: torch.Tensor,
    config: FusedUpdateConfig,
    bundle_count: int,
    slot_capacity: int = 4096,
    measure_time: bool = False,
    noise: Optional[torch.Tensor] = None,
    packed_out: Optional[torch.Tensor] = None,
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
        noise: Optional pre-generated standard normal noise buffer with at least
            num_params elements. Supplying this lets large-scale callers reuse
            memory and control RNG outside the operator.
        packed_out: Optional preallocated contiguous float32 CUDA output buffer
            with bundle_count * slot_capacity elements.

    Returns:
        FusedUpdateOutput with packed slots and metadata
    """
    if not flat.is_cuda:
        raise ValueError("fused_private_update_triton_v2 expects a CUDA tensor")
    flat = flat.detach().reshape(-1).contiguous()
    if flat.dtype != torch.float32:
        flat = flat.float()

    metrics = KernelMetrics()
    metrics.backend = "triton_v2"
    device = flat.device
    BLOCK_SIZE = 1024

    if measure_time:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

    num_params = flat.numel()
    total_slots = validate_fused_update_args(config, bundle_count, slot_capacity, num_params)
    metrics.input_bytes = flat.numel() * flat.element_size()
    metrics.logical_quantized_payload_bytes = (
        (num_params * config.quant_bits + 7) // 8 if config.quant_bits > 0 else 0
    )
    metrics.kernel_launch_count = _l2_norm_launch_count(num_params)

    use_quant = config.quant_bits > 0
    can_fold_quant_stat = use_quant and config.noise_multiplier <= 0

    # Step 1: Compute L2 norm. Quantized no-noise runs also need abs-max; fold
    # that stat into the L2 pass so the input is not streamed twice.
    if can_fold_quant_stat:
        l2_norm, raw_abs_max = compute_l2_norm_and_abs_max_triton(flat)
    else:
        l2_norm = compute_l2_norm_triton(flat)
        raw_abs_max = 0.0
    metrics.l2_norm = l2_norm

    # Compute clip scale on host
    clip_scale = min(1.0, config.clip_norm / (l2_norm + 1e-8))
    metrics.clip_scale = clip_scale

    if measure_time:
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        metrics.clip_ms = (t1 - t0) * 1000

    # Prepare noise buffer. Only actual parameter slots need noise; padding stays zero.
    noise_std = 0.0
    noise_source = "none"
    if measure_time:
        torch.cuda.synchronize(device)
        t_noise0 = time.perf_counter()

    if config.noise_multiplier > 0:
        noise_std = config.noise_multiplier * config.clip_norm
        metrics.noise_std = noise_std

        if noise is not None:
            noise_buffer = noise.detach().reshape(-1)
            if not noise_buffer.is_cuda or noise_buffer.device != device:
                raise ValueError("noise must be a CUDA tensor on the same device as flat")
            if noise_buffer.numel() < num_params:
                raise ValueError(
                    f"noise has {noise_buffer.numel()} elements but {num_params} are required"
                )
            if noise_buffer.dtype != torch.float32:
                noise_buffer = noise_buffer.float()
            if not noise_buffer.is_contiguous():
                noise_buffer = noise_buffer.contiguous()
            noise_source = "external"
        elif config.noise_seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(config.noise_seed)
            noise_buffer = torch.randn(num_params, generator=gen, device=device, dtype=torch.float32)
            noise_source = "seeded"
        else:
            noise_buffer = torch.randn(num_params, device=device, dtype=torch.float32)
            noise_source = "random"
    else:
        # Dummy buffer (not read when noise_std == 0)
        noise_buffer = torch.empty(1, device=device, dtype=torch.float32)

    metrics.noise_bytes = noise_buffer.numel() * noise_buffer.element_size()
    metrics.noise_source = noise_source
    if measure_time:
        torch.cuda.synchronize(device)
        t_noise1 = time.perf_counter()
        metrics.noise_ms = (t_noise1 - t_noise0) * 1000

    # Prepare quantization params
    quant_scale = 1.0
    quant_inv_scale = 1.0
    qmin = 0
    qmax_val = 0
    if use_quant:
        qmin = -(2 ** (config.quant_bits - 1))
        qmax_val = 2 ** (config.quant_bits - 1) - 1
        if num_params > 0:
            if can_fold_quant_stat:
                abs_max_val = raw_abs_max * abs(clip_scale * config.client_weight)
                metrics.quant_stat_source = "l2_abs_max"
            else:
                if measure_time:
                    torch.cuda.synchronize(device)
                    t_stat0 = time.perf_counter()
                abs_max_val = compute_transformed_abs_max_triton(
                    flat,
                    noise_buffer,
                    num_params,
                    clip_scale,
                    config.client_weight,
                    noise_std,
                )
                metrics.kernel_launch_count += _stat_reduce_launch_count(num_params)
                metrics.quant_stat_source = "transformed_abs_max"
                if measure_time:
                    torch.cuda.synchronize(device)
                    t_stat1 = time.perf_counter()
                    metrics.quant_stat_ms = (t_stat1 - t_stat0) * 1000
            metrics.quant_abs_max = abs_max_val
            if abs_max_val > 1e-8:
                quant_scale = abs_max_val / qmax_val
                quant_inv_scale = 1.0 / quant_scale
                metrics.quant_scale = quant_scale
        else:
            metrics.quant_stat_source = "empty"

    if measure_time:
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()

    # Step 2: Fused kernel — clip + scale + noise + quant + pack. The kernel
    # writes zero padding lanes, so callers do not pay for a full packed.zero_().
    use_physical_int8 = config.physical_quantized_output and config.quant_bits == 8

    if use_physical_int8:
        # Physical int8 output: allocate int8 buffer, use int8 kernel
        if packed_out is None:
            packed = torch.empty(total_slots, dtype=torch.int8, device=device)
        else:
            if not packed_out.is_cuda or packed_out.device != device:
                raise ValueError("packed_out must be a CUDA tensor on the same device as flat")
            if packed_out.dtype != torch.int8:
                raise ValueError(
                    f"packed_out must be int8 for physical_quantized_output, got {packed_out.dtype}"
                )
            if not packed_out.is_contiguous():
                raise ValueError("packed_out must be contiguous")
            if packed_out.numel() != total_slots:
                raise ValueError(
                    f"packed_out has {packed_out.numel()} elements but {total_slots} are required"
                )
            packed = packed_out.reshape(-1)
    else:
        # Standard float32 CKKS-ready slot output
        if packed_out is None:
            packed = torch.empty(total_slots, dtype=torch.float32, device=device)
        else:
            if not packed_out.is_cuda or packed_out.device != device:
                raise ValueError("packed_out must be a CUDA tensor on the same device as flat")
            if packed_out.dtype != torch.float32:
                raise ValueError(f"packed_out must be float32, got {packed_out.dtype}")
            if not packed_out.is_contiguous():
                raise ValueError("packed_out must be contiguous")
            if packed_out.numel() != total_slots:
                raise ValueError(
                    f"packed_out has {packed_out.numel()} elements but {total_slots} are required"
                )
            packed = packed_out.reshape(-1)

    if total_slots > 0:
        if num_params > 0:
            grid = (triton.cdiv(total_slots, BLOCK_SIZE),)
            if use_physical_int8:
                # Int8 kernel stores quantized integers directly as int8
                _fused_update_int8_kernel[grid](
                    flat,
                    noise_buffer,
                    packed,
                    clip_scale,
                    config.client_weight,
                    noise_std,
                    quant_inv_scale,
                    qmin,
                    qmax_val,
                    num_params,
                    total_slots,
                    slot_capacity,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            else:
                _fused_update_kernel[grid](
                    flat,
                    noise_buffer,
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
            metrics.kernel_launch_count += 1
        else:
            packed.zero_()

    packed = packed.reshape(bundle_count, slot_capacity)
    metrics.output_bytes = packed.numel() * packed.element_size()
    metrics.output_dtype = str(packed.dtype).replace("torch.", "")
    if use_physical_int8:
        metrics.output_format = "int8_quantized"

    if measure_time:
        torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        metrics.pack_ms = (t3 - t2) * 1000
        metrics.fused_operator_ms = metrics.quant_stat_ms + metrics.pack_ms
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
            "noise_bytes": metrics.noise_bytes,
            "noise_source": noise_source,
            "quant_abs_max": metrics.quant_abs_max,
            "quant_scale": metrics.quant_scale,
            "quant_stat_source": metrics.quant_stat_source,
            "kernel_launch_count": metrics.kernel_launch_count,
            "backend": metrics.backend,
            "input_bytes": metrics.input_bytes,
            "output_bytes": metrics.output_bytes,
            "output_dtype": metrics.output_dtype,
            "output_format": metrics.output_format,
            "logical_quantized_payload_bytes": metrics.logical_quantized_payload_bytes,
        },
    )
