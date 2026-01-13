#!/usr/bin/env python3
"""
Compare intermediate layer bounds between γ-CROWN and auto_LiRPA.

This script helps identify WHERE the bound gap originates by comparing
layer-by-layer bounds through the ResNet.

Usage:
    .venv/bin/python scripts/compare_intermediate_bounds.py
"""
import sys
import json
import subprocess
import numpy as np
import torch
import onnx
from pathlib import Path

# Add auto_LiRPA to path
sys.path.insert(0, str(Path(__file__).parent.parent / "research/repos/auto_LiRPA"))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def load_onnx_as_pytorch(onnx_path):
    """Convert ONNX model to PyTorch for auto_LiRPA."""
    import onnx2torch
    onnx_model = onnx.load(onnx_path)
    pytorch_model = onnx2torch.convert(onnx_model)
    return pytorch_model


def parse_vnnlib_bounds(vnnlib_path):
    """Parse input bounds from a vnnlib file."""
    lower = []
    upper = []

    with open(vnnlib_path, 'r') as f:
        content = f.read()

    import re
    lower_pattern = re.compile(r'\(assert \(>= X_(\d+) ([-\d.e+]+)\)\)')
    upper_pattern = re.compile(r'\(assert \(<= X_(\d+) ([-\d.e+]+)\)\)')

    lower_matches = lower_pattern.findall(content)
    upper_matches = upper_pattern.findall(content)

    lower_dict = {int(idx): float(val) for idx, val in lower_matches}
    upper_dict = {int(idx): float(val) for idx, val in upper_matches}

    num_inputs = max(max(lower_dict.keys()), max(upper_dict.keys())) + 1

    for i in range(num_inputs):
        lower.append(lower_dict.get(i, 0.0))
        upper.append(upper_dict.get(i, 0.0))

    return np.array(lower), np.array(upper)


def get_auto_lirpa_layer_bounds(model, x_L, x_U):
    """Get intermediate layer bounds from auto_LiRPA using CROWN."""
    x_center = (x_L + x_U) / 2

    # Create bounded model
    lirpa_model = BoundedModule(model, x_center, device='cpu')

    # Set up perturbation
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    # Get CROWN bounds first
    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')

    # Extract intermediate bounds from the model
    # auto_LiRPA stores bounds in the nodes
    layer_bounds = {}

    for name, node in lirpa_model.named_modules():
        if hasattr(node, 'lower') and node.lower is not None:
            lower = node.lower.detach().numpy()
            upper = node.upper.detach().numpy() if hasattr(node, 'upper') and node.upper is not None else lower
            width = (upper - lower).sum()

            layer_bounds[name] = {
                'shape': list(lower.shape),
                'total_width': float(width),
                'mean_width': float((upper - lower).mean()) if upper.size > 0 else 0,
                'max_width': float((upper - lower).max()) if upper.size > 0 else 0,
                'op_type': type(node).__name__,
            }

    return layer_bounds, lb.numpy(), ub.numpy()


def get_gamma_crown_layer_bounds(model_path, vnnlib_path):
    """Get intermediate layer bounds from γ-CROWN."""
    gamma_binary = Path(__file__).parent.parent / "target/release/gamma"

    # Use verify with --dump-bounds to get intermediate bounds
    cmd = [
        str(gamma_binary),
        "verify",
        str(model_path),
        "--property", str(vnnlib_path),
        "--method", "crown",
        "--json",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Parse JSON output
        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
            except json.JSONDecodeError:
                # Find JSON in output
                for line in result.stdout.split('\n'):
                    if line.startswith('{'):
                        try:
                            output = json.loads(line)
                            break
                        except:
                            pass
                else:
                    return {'error': 'No JSON found in output'}

            return output
        else:
            return {'error': result.stderr}

    except Exception as e:
        return {'error': str(e)}


def main():
    torch.set_grad_enabled(False)

    # Paths
    model_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
    vnnlib_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"

    print("=" * 70)
    print("Intermediate Bounds Comparison: γ-CROWN vs auto_LiRPA")
    print("=" * 70)

    # Parse input bounds
    x_L_np, x_U_np = parse_vnnlib_bounds(vnnlib_path)
    x_L = torch.tensor(x_L_np, dtype=torch.float32).reshape(1, 3, 32, 32)
    x_U = torch.tensor(x_U_np, dtype=torch.float32).reshape(1, 3, 32, 32)

    input_width = float((x_U - x_L).sum())
    print(f"\nInput total width: {input_width:.4f}")

    # Load model
    print("\nLoading model for auto_LiRPA...")
    pytorch_model = load_onnx_as_pytorch(str(model_path))
    pytorch_model.eval()

    # Get auto_LiRPA bounds
    print("\nComputing auto_LiRPA CROWN intermediate bounds...")
    lirpa_bounds, lb, ub = get_auto_lirpa_layer_bounds(pytorch_model, x_L, x_U)

    # Print auto_LiRPA layer bounds
    print("\n" + "-" * 70)
    print("auto_LiRPA Intermediate Bounds (CROWN):")
    print("-" * 70)

    # Sort by total width descending
    sorted_layers = sorted(lirpa_bounds.items(), key=lambda x: -x[1]['total_width'])

    for name, info in sorted_layers[:20]:  # Top 20 layers
        print(f"  {name:40s} | width: {info['total_width']:12.2f} | shape: {info['shape']} | {info['op_type']}")

    print(f"\nOutput bounds (CROWN):")
    print(f"  Lower: {[f'{x:.4f}' for x in lb.flatten()[:10]]}")
    print(f"  Upper: {[f'{x:.4f}' for x in ub.flatten()[:10]]}")
    print(f"  Total width: {(ub - lb).sum():.4f}")

    # Get γ-CROWN bounds
    print("\n" + "-" * 70)
    print("γ-CROWN Output Bounds:")
    print("-" * 70)

    gamma_result = get_gamma_crown_layer_bounds(model_path, vnnlib_path)

    if 'error' in gamma_result:
        print(f"Error: {gamma_result['error']}")
    else:
        bounds = gamma_result.get('output_bounds', [])
        if bounds:
            gamma_lower = [b['lower'] for b in bounds]
            gamma_upper = [b['upper'] for b in bounds]
            gamma_width = sum(u - l for l, u in zip(gamma_lower, gamma_upper))

            print(f"  Lower: {[f'{x:.4f}' for x in gamma_lower[:10]]}")
            print(f"  Upper: {[f'{x:.4f}' for x in gamma_upper[:10]]}")
            print(f"  Total width: {gamma_width:.4f}")

            # Compare per-output
            print("\n" + "-" * 70)
            print("Per-Output Comparison:")
            print("-" * 70)
            print(f"{'Output':>6s} | {'auto_LiRPA':>12s} | {'γ-CROWN':>12s} | {'Ratio':>8s}")
            print("-" * 50)

            lirpa_lower = lb.flatten()
            lirpa_upper = ub.flatten()

            for i in range(min(10, len(gamma_lower))):
                lirpa_w = lirpa_upper[i] - lirpa_lower[i]
                gamma_w = gamma_upper[i] - gamma_lower[i]
                ratio = gamma_w / lirpa_w if lirpa_w > 0 else float('inf')
                print(f"{i:>6d} | {lirpa_w:>12.4f} | {gamma_w:>12.4f} | {ratio:>8.2f}x")

            total_lirpa = (lirpa_upper - lirpa_lower).sum()
            total_gamma = gamma_width
            print("-" * 50)
            print(f"{'TOTAL':>6s} | {total_lirpa:>12.4f} | {total_gamma:>12.4f} | {total_gamma/total_lirpa:>8.2f}x")


if __name__ == '__main__':
    main()
