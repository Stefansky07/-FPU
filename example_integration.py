#!/usr/bin/env python
"""
Example: Integrating FPU with hefl_runtime.

This script demonstrates how to use the fused private update operator
with the existing LayoutPlan system from hefl_runtime.
"""

import sys
from pathlib import Path

# Add hefl_runtime to path
sys.path.insert(0, str(Path(__file__).parent.parent / "hefl_runtime"))

import torch

from fpu import FusedUpdateConfig, fused_private_update_triton, fused_private_update_ref
from fpu.validate import create_test_state_dict

# Import hefl_runtime components
try:
    from hefl.types import CryptoParams
    from hefl.layout import HeuristicLayoutPlanner
    HAS_HEFL = True
except ImportError:
    print("WARNING: hefl_runtime not found, using standalone mode")
    HAS_HEFL = False


def example_standalone():
    """Example: Standalone usage without hefl_runtime."""
    print("="*60)
    print("Example: Standalone Usage")
    print("="*60)

    # Create a model update
    state_dict = {
        "layer1.weight": torch.randn(64, 32),
        "layer1.bias": torch.randn(64),
        "layer2.weight": torch.randn(10, 64),
        "layer2.bias": torch.randn(10),
    }

    num_params = sum(t.numel() for t in state_dict.values())
    print(f"Model parameters: {num_params:,}")

    # Configure
    config = FusedUpdateConfig(
        clip_norm=1.0,
        noise_multiplier=0.1,
        quant_bits=0,
        client_weight=1.0,
    )

    # Compute layout
    slot_capacity = 4096  # CKKS slots
    bundle_count = (num_params + slot_capacity - 1) // slot_capacity
    print(f"Bundle count: {bundle_count}")
    print(f"Slot capacity: {slot_capacity}")

    # Run on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output = fused_private_update_triton(
        state_dict, config, bundle_count, slot_capacity, device,
        measure_time=True,
    )

    print(f"\nOutput shape: {output.packed_slots.shape}")
    print(f"Utilization: {output.utilization:.1%}")
    print(f"Metadata: {output.metadata}")

    return output


def example_with_layout_plan():
    """Example: Using with hefl_runtime LayoutPlan."""
    print("\n" + "="*60)
    print("Example: Integration with LayoutPlan")
    print("="*60)

    if not HAS_HEFL:
        print("Skipping (hefl_runtime not available)")
        return

    # Create crypto parameters
    crypto = CryptoParams(
        scheme="CKKS",
        poly_modulus_degree=8192,
        slots=4096,
        coeff_mod_bit_sizes=(60, 40, 40, 60),
    )

    # Create layout planner
    planner = HeuristicLayoutPlanner(crypto)

    # Create a model state dict
    state_dict = create_test_state_dict("tiny_cnn")
    num_params = sum(t.numel() for t in state_dict.values())
    print(f"Model parameters: {num_params:,}")

    # Generate layout plan
    layout = planner.plan(
        state_dict,
        strategy="layer_order",
        template_policy="tight",
    )

    print(f"Bundle count: {layout.bundle_count}")
    print(f"Slot utilization: {layout.slot_utilization:.1%}")

    # Configure FPU
    config = FusedUpdateConfig(
        clip_norm=1.0,
        noise_multiplier=0.1,
        quant_bits=0,
        client_weight=1.0,
    )

    # Run fused operation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output = fused_private_update_triton(
        state_dict,
        config,
        layout.bundle_count,
        layout.slot_capacity,
        device,
        measure_time=True,
    )

    print(f"\nOutput shape: {output.packed_slots.shape}")
    print(f"Utilization: {output.utilization:.1%}")

    return output


def example_multi_client():
    """Example: Processing multiple client updates."""
    print("\n" + "="*60)
    print("Example: Multi-Client Aggregation")
    print("="*60)

    # Simulate 4 client updates
    num_clients = 4
    state_dicts = [create_test_state_dict("tiny_mlp") for _ in range(num_clients)]
    weights = [0.3, 0.25, 0.25, 0.2]  # Client importance weights

    num_params = sum(t.numel() for t in state_dicts[0].values())
    slot_capacity = 4096
    bundle_count = (num_params + slot_capacity - 1) // slot_capacity

    print(f"Clients: {num_clients}")
    print(f"Parameters per client: {num_params:,}")
    print(f"Weights: {weights}")

    # Configure
    config = FusedUpdateConfig(
        clip_norm=1.0,
        noise_multiplier=0.1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process each client and aggregate
    aggregated = torch.zeros(
        bundle_count * slot_capacity, dtype=torch.float32, device=device
    )

    for i, (state_dict, weight) in enumerate(zip(state_dicts, weights)):
        output = fused_private_update_triton(
            state_dict,
            FusedUpdateConfig(
                clip_norm=config.clip_norm,
                noise_multiplier=config.noise_multiplier,
                client_weight=weight,
            ),
            bundle_count,
            slot_capacity,
            device,
        )
        aggregated += output.packed_slots.reshape(-1)
        print(f"  Client {i}: weight={weight}, norm={output.metadata.get('l2_norm', 0):.4f}")

    aggregated = aggregated.reshape(bundle_count, slot_capacity)

    print(f"\nAggregated shape: {aggregated.shape}")
    print(f"Aggregated norm: {torch.linalg.vector_norm(aggregated.float()).item():.4f}")

    return aggregated


def main():
    print("FPU Integration Examples")
    print("="*60)

    # Run examples
    example_standalone()
    example_with_layout_plan()
    example_multi_client()

    print("\n" + "="*60)
    print("All examples completed!")


if __name__ == "__main__":
    main()
