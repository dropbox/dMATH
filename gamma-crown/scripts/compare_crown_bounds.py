#!/usr/bin/env python3
"""
Compare γ-CROWN and auto_LiRPA CROWN bounds on cifar10_resnet.

This script helps diagnose why γ-CROWN produces looser bounds than auto_LiRPA.

Usage:
    PYTHONPATH=research/repos/auto_LiRPA .venv/bin/python scripts/compare_crown_bounds.py
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
    # Find all assertions for input bounds
    # Format: (assert (>= X_i val)) for lower, (assert (<= X_i val)) for upper
    lower_pattern = re.compile(r'\(assert \(>= X_(\d+) ([-\d.e+]+)\)\)')
    upper_pattern = re.compile(r'\(assert \(<= X_(\d+) ([-\d.e+]+)\)\)')

    lower_matches = lower_pattern.findall(content)
    upper_matches = upper_pattern.findall(content)

    # Convert to dict for proper ordering
    lower_dict = {int(idx): float(val) for idx, val in lower_matches}
    upper_dict = {int(idx): float(val) for idx, val in upper_matches}

    num_inputs = max(max(lower_dict.keys()), max(upper_dict.keys())) + 1

    for i in range(num_inputs):
        lower.append(lower_dict.get(i, 0.0))
        upper.append(upper_dict.get(i, 0.0))

    return np.array(lower), np.array(upper)


def compute_auto_lirpa_bounds_with_explicit_bounds(model, x_L, x_U, methods=['IBP', 'CROWN', 'CROWN-IBP']):
    """Compute bounds using auto_LiRPA with explicit input bounds."""
    results = {}

    # Center point
    x_center = (x_L + x_U) / 2

    # Wrap model
    lirpa_model = BoundedModule(model, x_center, device='cpu')

    # Compute forward pass
    with torch.no_grad():
        pred = lirpa_model(x_center)
    results['prediction'] = pred.numpy().flatten().tolist()

    # Set up perturbation with explicit bounds
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    for method in methods:
        try:
            lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method=method)
            lb_np = lb.detach().numpy().flatten()
            ub_np = ub.detach().numpy().flatten()
            width = float((ub_np - lb_np).sum())
            max_width = float((ub_np - lb_np).max())

            results[method] = {
                'lower': lb_np.tolist(),
                'upper': ub_np.tolist(),
                'total_width': width,
                'max_width': max_width,
            }
        except Exception as e:
            results[method] = {'error': str(e)}

    return results


def compute_gamma_bounds_json(model_path, vnnlib_path):
    """Compute bounds using gamma CLI and parse JSON output."""
    gamma_binary = Path(__file__).parent.parent / "target/release/gamma"
    cmd = [
        str(gamma_binary),
        "verify",
        str(model_path),
        "--property", str(vnnlib_path),
        "--method", "crown",
        "--json"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            output = json.loads(result.stdout)
            bounds = output.get('output_bounds', [])
            if bounds:
                lower = [b['lower'] for b in bounds]
                upper = [b['upper'] for b in bounds]
                widths = [u - l for l, u in zip(lower, upper)]
                return {
                    'lower': lower,
                    'upper': upper,
                    'total_width': sum(widths),
                    'max_width': max(widths),
                    'status': output.get('status'),
                    'property_status': output.get('property_status'),
                }
        return {'error': result.stderr or 'Unknown error'}
    except Exception as e:
        return {'error': str(e)}


def main():
    torch.set_grad_enabled(False)

    # Paths
    model_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
    vnnlib_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"

    print("=" * 70)
    print("CROWN Bounds Comparison: γ-CROWN vs auto_LiRPA")
    print("=" * 70)
    print(f"Model: {model_path.name}")
    print(f"Property: {vnnlib_path.name}")

    # Parse input bounds from vnnlib
    print("\nParsing vnnlib property...")
    x_L_np, x_U_np = parse_vnnlib_bounds(vnnlib_path)

    # Reshape to CIFAR10 shape [1, 3, 32, 32]
    x_L = torch.tensor(x_L_np, dtype=torch.float32).reshape(1, 3, 32, 32)
    x_U = torch.tensor(x_U_np, dtype=torch.float32).reshape(1, 3, 32, 32)

    input_width = float((x_U - x_L).sum())
    avg_eps = float((x_U - x_L).mean()) / 2

    print(f"\nInput bounds from vnnlib:")
    print(f"  Shape: {list(x_L.shape)}")
    print(f"  Total input width: {input_width:.4f}")
    print(f"  Average per-input epsilon: {avg_eps:.6f}")
    print(f"  Sample lower[0:5]: {x_L_np[:5]}")
    print(f"  Sample upper[0:5]: {x_U_np[:5]}")

    # Load model for auto_LiRPA
    print("\nLoading model...")
    try:
        pytorch_model = load_onnx_as_pytorch(str(model_path))
        pytorch_model.eval()
        print("  Model loaded successfully")
    except Exception as e:
        print(f"  Failed to load model: {e}")
        print("  Note: You may need to install onnx2torch: pip install onnx2torch")
        return

    # Compute auto_LiRPA bounds
    print("\n" + "-" * 50)
    print("auto_LiRPA Bounds (same input as vnnlib):")
    print("-" * 50)

    lirpa_results = compute_auto_lirpa_bounds_with_explicit_bounds(pytorch_model, x_L, x_U)

    print(f"\nPrediction at center: {[f'{x:.2f}' for x in lirpa_results['prediction']]}")

    for method in ['IBP', 'CROWN-IBP', 'CROWN']:
        if method in lirpa_results:
            r = lirpa_results[method]
            if 'error' in r:
                print(f"\n{method}: ERROR - {r['error']}")
            else:
                print(f"\n{method}:")
                print(f"  Lower: {[f'{x:.2f}' for x in r['lower']]}")
                print(f"  Upper: {[f'{x:.2f}' for x in r['upper']]}")
                print(f"  Total width: {r['total_width']:.2f}")
                print(f"  Max single output width: {r['max_width']:.2f}")

    # Compute gamma bounds
    print("\n" + "-" * 50)
    print("γ-CROWN Bounds (same input as vnnlib):")
    print("-" * 50)

    gamma_results = compute_gamma_bounds_json(model_path, vnnlib_path)

    if 'error' in gamma_results:
        print(f"\nγ-CROWN: ERROR - {gamma_results['error']}")
    else:
        print(f"\nCROWN:")
        print(f"  Lower: {[f'{x:.2f}' for x in gamma_results['lower']]}")
        print(f"  Upper: {[f'{x:.2f}' for x in gamma_results['upper']]}")
        print(f"  Total width: {gamma_results['total_width']:.2f}")
        print(f"  Max single output width: {gamma_results['max_width']:.2f}")
        print(f"  Status: {gamma_results['status']}, Property: {gamma_results['property_status']}")

    # Report summary
    print("\n" + "=" * 70)
    print("SUMMARY - CROWN COMPARISON")
    print("=" * 70)

    if 'CROWN' in lirpa_results and 'error' not in lirpa_results['CROWN']:
        lirpa_crown = lirpa_results['CROWN']
        print(f"\nauto_LiRPA CROWN:")
        print(f"  Max output width: {lirpa_crown['max_width']:.4f}")
        print(f"  Total output width: {lirpa_crown['total_width']:.4f}")

    if 'error' not in gamma_results:
        print(f"\nγ-CROWN CROWN:")
        print(f"  Max output width: {gamma_results['max_width']:.4f}")
        print(f"  Total output width: {gamma_results['total_width']:.4f}")

        if 'CROWN' in lirpa_results and 'error' not in lirpa_results['CROWN']:
            ratio = gamma_results['total_width'] / lirpa_results['CROWN']['total_width']
            print(f"\n*** γ-CROWN is {ratio:.1f}x LOOSER than auto_LiRPA ***")
            if ratio > 10:
                print("*** THIS IS A SIGNIFICANT BUG - CROWN bounds should be similar ***")

    # Save results
    output_path = Path(__file__).parent.parent / "benchmarks/crown_comparison_results.json"
    results = {
        'auto_lirpa': lirpa_results,
        'gamma_crown': gamma_results,
        'input_bounds': {
            'total_width': input_width,
            'avg_epsilon': avg_eps,
        }
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
