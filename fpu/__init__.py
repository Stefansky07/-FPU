"""
Fused Private Update (FPU) Operator
====================================

A GPU-accelerated fused kernel for privacy-preserving federated learning updates.

Operations fused into a single kernel pass:
1. Flatten model update (state_dict → flat vector)
2. L2 norm clipping (DP-SGD)
3. Client weight absorption
4. Optional DP Gaussian noise injection
5. Optional quantization
6. LayoutPlan-guided slot packing
7. Padding and metadata output
"""

from .types import FusedUpdateConfig, FusedUpdateOutput
from .torch_ref import fused_private_update_ref

# Triton backend is optional — gracefully degrade on platforms without Triton
try:
    from .triton_kernel import (
        fused_private_update_triton,
        fused_private_update_triton_v2,
        fused_private_update_triton_v2_stream_to_cpu,
    )
    _HAS_TRITON = True
except ImportError:
    fused_private_update_triton = None
    fused_private_update_triton_v2 = None
    fused_private_update_triton_v2_stream_to_cpu = None
    _HAS_TRITON = False

__version__ = "0.1.1"
__all__ = [
    "FusedUpdateConfig",
    "FusedUpdateOutput",
    "fused_private_update_ref",
    "fused_private_update_triton",
    "fused_private_update_triton_v2",
    "fused_private_update_triton_v2_stream_to_cpu",
    "_HAS_TRITON",
]
