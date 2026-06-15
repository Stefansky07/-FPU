"""
Type definitions for the Fused Private Update operator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class FusedUpdateConfig:
    """Configuration for the fused private update operator."""

    # L2 clipping threshold for DP-SGD
    clip_norm: float = 1.0

    # DP noise multiplier (0 = no noise)
    # noise_std = noise_multiplier * clip_norm
    noise_multiplier: float = 0.0

    # Quantization bits (0 = no quantization, 8 = int8, 16 = int16)
    quant_bits: int = 0

    # Emit physical int8 slots instead of dequantized float32 CKKS slots.
    # This is an opt-in benchmark/transport path; the default remains float32
    # for direct CKKS encoding compatibility.
    physical_quantized_output: bool = False

    # Client weight for aggregation
    client_weight: float = 1.0

    # CKKS encoding scale (for metadata only)
    ckks_scale: float = 2**40

    # Random seed for noise generation (None = random)
    noise_seed: Optional[int] = None

    # Generate DP noise inside Triton kernels as randn(seed, global_offset)
    # instead of materializing a full float32 noise tensor.
    stateless_noise: bool = False

    # Skip abs-max quantization statistics and use this quantization scale
    # directly. This is an experimental fast path for calibrated/fixed-scale
    # benchmarks.
    fixed_quant_scale: Optional[float] = None

    # Skip L2 reduction and use this clip scale directly. This is intended for
    # CUDA Graph / pack-only experiments after the scale has been calibrated.
    fixed_clip_scale: Optional[float] = None

    # Process the pack kernel in chunks. This is useful for large-scale
    # experiments and for later CPU/GPU streaming integration.
    chunk_size: Optional[int] = None

    # Compute an independent quantization scale per chunk. Requires chunk_size.
    chunk_quant_scale: bool = False

    # Triton block size used by the fused pack kernels.
    pack_block_size: int = 1024

    def validate(self) -> None:
        """Validate scalar configuration before launching GPU work."""
        if not math.isfinite(self.clip_norm) or self.clip_norm <= 0:
            raise ValueError(f"clip_norm must be a positive finite value, got {self.clip_norm}")
        if not math.isfinite(self.noise_multiplier) or self.noise_multiplier < 0:
            raise ValueError(
                f"noise_multiplier must be a non-negative finite value, got {self.noise_multiplier}"
            )
        if isinstance(self.quant_bits, bool) or not isinstance(self.quant_bits, int):
            raise TypeError(f"quant_bits must be an integer, got {type(self.quant_bits).__name__}")
        if self.quant_bits < 0 or self.quant_bits == 1 or self.quant_bits >= 32:
            raise ValueError("quant_bits must be 0 or an integer in [2, 31]")
        if not isinstance(self.physical_quantized_output, bool):
            raise TypeError(
                "physical_quantized_output must be a boolean, "
                f"got {type(self.physical_quantized_output).__name__}"
            )
        if self.physical_quantized_output and self.quant_bits != 8:
            raise ValueError("physical_quantized_output currently requires quant_bits=8")
        if not math.isfinite(self.client_weight):
            raise ValueError(f"client_weight must be finite, got {self.client_weight}")
        if not math.isfinite(self.ckks_scale) or self.ckks_scale <= 0:
            raise ValueError(f"ckks_scale must be a positive finite value, got {self.ckks_scale}")
        if not isinstance(self.stateless_noise, bool):
            raise TypeError(
                f"stateless_noise must be a boolean, got {type(self.stateless_noise).__name__}"
            )
        if self.stateless_noise and self.noise_multiplier > 0 and self.noise_seed is None:
            raise ValueError("stateless_noise with DP noise requires noise_seed for reproducibility")
        if self.fixed_quant_scale is not None:
            if (
                not isinstance(self.fixed_quant_scale, (float, int))
                or not math.isfinite(float(self.fixed_quant_scale))
                or float(self.fixed_quant_scale) <= 0
            ):
                raise ValueError(
                    "fixed_quant_scale must be a positive finite value when provided"
                )
        if self.fixed_clip_scale is not None:
            if (
                not isinstance(self.fixed_clip_scale, (float, int))
                or not math.isfinite(float(self.fixed_clip_scale))
                or float(self.fixed_clip_scale) <= 0
            ):
                raise ValueError(
                    "fixed_clip_scale must be a positive finite value when provided"
                )
        if self.chunk_size is not None:
            if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int):
                raise TypeError(
                    f"chunk_size must be an integer or None, got {type(self.chunk_size).__name__}"
                )
            if self.chunk_size <= 0:
                raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if not isinstance(self.chunk_quant_scale, bool):
            raise TypeError(
                f"chunk_quant_scale must be a boolean, got {type(self.chunk_quant_scale).__name__}"
            )
        if self.chunk_quant_scale and self.chunk_size is None:
            raise ValueError("chunk_quant_scale requires chunk_size")
        if self.chunk_quant_scale and self.fixed_quant_scale is not None:
            raise ValueError("chunk_quant_scale and fixed_quant_scale are mutually exclusive")
        if isinstance(self.pack_block_size, bool) or not isinstance(self.pack_block_size, int):
            raise TypeError(
                f"pack_block_size must be an integer, got {type(self.pack_block_size).__name__}"
            )
        if self.pack_block_size <= 0 or self.pack_block_size > 8192:
            raise ValueError("pack_block_size must be in the range [1, 8192]")
        if self.pack_block_size & (self.pack_block_size - 1):
            raise ValueError("pack_block_size must be a power of two")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_multiplier,
            "quant_bits": self.quant_bits,
            "physical_quantized_output": self.physical_quantized_output,
            "client_weight": self.client_weight,
            "ckks_scale": self.ckks_scale,
            "noise_seed": self.noise_seed,
            "stateless_noise": self.stateless_noise,
            "fixed_quant_scale": self.fixed_quant_scale,
            "fixed_clip_scale": self.fixed_clip_scale,
            "chunk_size": self.chunk_size,
            "chunk_quant_scale": self.chunk_quant_scale,
            "pack_block_size": self.pack_block_size,
        }


@dataclass
class FusedUpdateOutput:
    """Output from the fused private update operator."""

    # Packed slot buffers [bundles, slot_capacity]
    packed_slots: torch.Tensor

    # Layout plan used for packing
    bundle_count: int
    slot_capacity: int
    num_params: int

    # Metadata for debugging and analysis
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_slots(self) -> int:
        return self.bundle_count * self.slot_capacity

    @property
    def padding_slots(self) -> int:
        return self.total_slots - self.num_params

    @property
    def utilization(self) -> float:
        return self.num_params / max(self.total_slots, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_count": self.bundle_count,
            "slot_capacity": self.slot_capacity,
            "num_params": self.num_params,
            "total_slots": self.total_slots,
            "padding_slots": self.padding_slots,
            "utilization": self.utilization,
            "metadata": self.metadata,
        }


@dataclass
class KernelMetrics:
    """Performance metrics for kernel execution."""

    # Timing in milliseconds
    total_ms: float = 0.0
    flatten_ms: float = 0.0
    clip_ms: float = 0.0
    noise_ms: float = 0.0
    quant_stat_ms: float = 0.0
    quant_ms: float = 0.0
    pack_ms: float = 0.0
    fused_operator_ms: float = 0.0

    # Computed values
    l2_norm: float = 0.0
    clip_scale: float = 1.0
    noise_std: float = 0.0
    quant_abs_max: float = 0.0
    quant_scale: float = 1.0
    quant_stat_source: str = "none"
    kernel_launch_count: int = 0
    backend: str = ""
    noise_source: str = "none"

    # Memory
    input_bytes: int = 0
    output_bytes: int = 0
    logical_quantized_payload_bytes: int = 0
    output_dtype: str = ""
    output_format: str = "float_slots"
    noise_bytes: int = 0

    @property
    def bandwidth_gbps(self) -> float:
        total_bytes = self.input_bytes + self.output_bytes
        if self.total_ms <= 0:
            return 0.0
        return (total_bytes / 1e9) / (self.total_ms / 1e3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ms": self.total_ms,
            "flatten_ms": self.flatten_ms,
            "clip_ms": self.clip_ms,
            "noise_ms": self.noise_ms,
            "quant_stat_ms": self.quant_stat_ms,
            "quant_ms": self.quant_ms,
            "pack_ms": self.pack_ms,
            "fused_operator_ms": self.fused_operator_ms,
            "l2_norm": self.l2_norm,
            "clip_scale": self.clip_scale,
            "noise_std": self.noise_std,
            "quant_abs_max": self.quant_abs_max,
            "quant_scale": self.quant_scale,
            "quant_stat_source": self.quant_stat_source,
            "kernel_launch_count": self.kernel_launch_count,
            "backend": self.backend,
            "noise_source": self.noise_source,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "logical_quantized_payload_bytes": self.logical_quantized_payload_bytes,
            "output_dtype": self.output_dtype,
            "output_format": self.output_format,
            "noise_bytes": self.noise_bytes,
            "bandwidth_gbps": self.bandwidth_gbps,
        }


def validate_fused_update_args(
    config: FusedUpdateConfig,
    bundle_count: int,
    slot_capacity: int,
    num_params: int,
) -> int:
    """Validate shared layout/config arguments and return total slot count."""
    config.validate()

    for name, value in (
        ("bundle_count", bundle_count),
        ("slot_capacity", slot_capacity),
        ("num_params", num_params),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")

    if bundle_count < 0:
        raise ValueError(f"bundle_count must be non-negative, got {bundle_count}")
    if slot_capacity <= 0:
        raise ValueError(f"slot_capacity must be positive, got {slot_capacity}")
    if num_params < 0:
        raise ValueError(f"num_params must be non-negative, got {num_params}")

    total_slots = bundle_count * slot_capacity
    if total_slots < num_params:
        raise ValueError(
            f"slot capacity is insufficient: total_slots={total_slots}, num_params={num_params}"
        )
    return total_slots
