"""
Type definitions for the Fused Private Update operator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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

    # Client weight for aggregation
    client_weight: float = 1.0

    # CKKS encoding scale (for metadata only)
    ckks_scale: float = 2**40

    # Random seed for noise generation (None = random)
    noise_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_multiplier,
            "quant_bits": self.quant_bits,
            "client_weight": self.client_weight,
            "ckks_scale": self.ckks_scale,
            "noise_seed": self.noise_seed,
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
    quant_ms: float = 0.0
    pack_ms: float = 0.0

    # Computed values
    l2_norm: float = 0.0
    clip_scale: float = 1.0
    noise_std: float = 0.0

    # Memory
    input_bytes: int = 0
    output_bytes: int = 0

    @property
    def bandwidth_gbps(self) -> float:
        total_bytes = self.input_bytes + self.output_bytes
        if self.total_ms <= 0:
            return 0.0
        return (total_bytes / 1e9) / (self.total_ms / 1e3)

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_ms": self.total_ms,
            "flatten_ms": self.flatten_ms,
            "clip_ms": self.clip_ms,
            "noise_ms": self.noise_ms,
            "quant_ms": self.quant_ms,
            "pack_ms": self.pack_ms,
            "l2_norm": self.l2_norm,
            "clip_scale": self.clip_scale,
            "noise_std": self.noise_std,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "bandwidth_gbps": self.bandwidth_gbps,
        }
