#!/usr/bin/env python3
"""
Debug intermediate CROWN bounds to compare γ-CROWN vs auto_LiRPA.

This script runs both verifiers and logs intermediate bounds at each layer
to identify where the gap originates.
"""
import sys
import json
import subprocess
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add auto_LiRPA to path
sys.path.insert(0, str(Path(__file__).parent.parent / "research/repos/auto_LiRPA"))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def load_resnet_model():
    """Load ResNet model and extract parameters."""
    import onnx
    from onnx import numpy_helper

    model_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
    model = onnx.load(str(model_path))

    # Extract initializers (weights and biases)
    initializers = {}
    for init in model.graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    return model, initializers


def get_auto_lirpa_bounds_detailed():
    """Get auto_LiRPA bounds with detailed logging enabled."""
    import logging
    logging.basicConfig(level=logging.DEBUG)

    model_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
    vnnlib_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/vnnlib/prop_0_eps_0.008.vnnlib"

    # Parse vnnlib to get input bounds
    with open(vnnlib_path) as f:
        content = f.read()

    # Extract input bounds from vnnlib
    import re
    lower_bounds = []
    upper_bounds = []

    for line in content.split('\n'):
        if line.startswith('(assert (>='):
            match = re.search(r'X_\d+ ([-\d.e]+)\)', line)
            if match:
                lower_bounds.append(float(match.group(1)))
        elif line.startswith('(assert (<='):
            match = re.search(r'X_\d+ ([-\d.e]+)\)', line)
            if match:
                upper_bounds.append(float(match.group(1)))

    x_L = torch.tensor(lower_bounds, dtype=torch.float32).reshape(1, 3, 32, 32)
    x_U = torch.tensor(upper_bounds, dtype=torch.float32).reshape(1, 3, 32, 32)
    x_center = (x_L + x_U) / 2

    print(f"Input bounds: L=[{x_L.min():.6f}, {x_L.max():.6f}], U=[{x_U.min():.6f}, {x_U.max():.6f}]")

    # Load ONNX model
    import onnx
    import onnx2pytorch

    onnx_model = onnx.load(str(model_path))
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True)
    pytorch_model.eval()

    # Create bounded module with verbose logging
    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu', verbose=True)

    # Create perturbation
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    # Compute bounds with CROWN - this should trigger intermediate bound logging
    print("\n=== Running auto_LiRPA CROWN (IBP=False) ===")
    lb_crown, ub_crown = lirpa_model.compute_bounds(
        x=(bounded_x,),
        method='backward',
        IBP=False,
        bound_upper=True,
    )

    crown_width = (ub_crown - lb_crown).sum().item()
    print(f"\nCROWN output bounds:")
    print(f"  Lower: {lb_crown.flatten().tolist()}")
    print(f"  Upper: {ub_crown.flatten().tolist()}")
    print(f"  Total width: {crown_width:.4f}")

    # Now print intermediate bounds for all nodes
    print("\n=== Intermediate bounds after CROWN ===")
    for name, node in lirpa_model.bound_ops.items():
        if hasattr(node, 'lower') and node.lower is not None:
            if hasattr(node.lower, 'shape'):
                l = node.lower.flatten()
                u = node.upper.flatten()
                width = (u - l).sum().item()
                max_width = (u - l).max().item()
                print(f"  {name}: width={width:.4f}, max={max_width:.4f}, shape={list(node.lower.shape)}")

    return lirpa_model


def run_gamma_crown_debug():
    """Run γ-CROWN with debug logging for intermediate bounds."""
    model_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
    vnnlib_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/vnnlib/prop_0_eps_0.008.vnnlib"
    gamma_binary = Path(__file__).parent.parent / "target/release/gamma"

    print("\n=== Running γ-CROWN with debug logging ===")

    cmd = [
        str(gamma_binary),
        "verify",
        str(model_path),
        "--property", str(vnnlib_path),
        "--method", "crown",
        "--json",
    ]

    # Run with debug logging
    env = {"RUST_LOG": "gamma_propagate=debug"}
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        env={**dict(__builtins__.__dict__.get('__import__')('os').environ), **env}
    )

    # Print debug output
    if result.stderr:
        print("Debug output (first 5000 chars):")
        print(result.stderr[:5000])

    if result.returncode == 0:
        output = json.loads(result.stdout)
        bounds = output.get('output_bounds', [])
        if bounds:
            lower = [b['lower'] for b in bounds]
            upper = [b['upper'] for b in bounds]
            width = sum(u - l for l, u in zip(lower, upper))
            print(f"\nγ-CROWN CROWN output bounds:")
            print(f"  Lower: {lower}")
            print(f"  Upper: {upper}")
            print(f"  Total width: {width:.4f}")
    else:
        print(f"γ-CROWN failed: {result.stderr}")


def main():
    print("=" * 70)
    print("Debugging intermediate CROWN bounds")
    print("=" * 70)

    # Run auto_LiRPA with detailed logging
    lirpa_model = get_auto_lirpa_bounds_detailed()

    # Run γ-CROWN with debug logging
    run_gamma_crown_debug()


if __name__ == '__main__':
    main()
