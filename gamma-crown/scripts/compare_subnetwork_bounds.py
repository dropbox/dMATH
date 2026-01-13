#!/usr/bin/env python3
"""
Compare bounds on subnetworks of the resnet model to find where the gap originates.
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


def compare_ibp_and_crown(lirpa_model, bounded_x):
    """Compare IBP and CROWN bounds."""
    # IBP bounds
    lb_ibp, ub_ibp = lirpa_model.compute_bounds(x=(bounded_x,), method='IBP')
    ibp_lb = lb_ibp.detach().numpy().flatten()
    ibp_ub = ub_ibp.detach().numpy().flatten()
    ibp_width = (ibp_ub - ibp_lb).sum()

    # CROWN-IBP bounds
    lb_ci, ub_ci = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN-IBP')
    ci_lb = lb_ci.detach().numpy().flatten()
    ci_ub = ub_ci.detach().numpy().flatten()
    ci_width = (ci_ub - ci_lb).sum()

    # CROWN bounds
    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
    crown_lb = lb.detach().numpy().flatten()
    crown_ub = ub.detach().numpy().flatten()
    crown_width = (crown_ub - crown_lb).sum()

    return {
        'IBP': {'width': ibp_width, 'lower': ibp_lb, 'upper': ibp_ub},
        'CROWN-IBP': {'width': ci_width, 'lower': ci_lb, 'upper': ci_ub},
        'CROWN': {'width': crown_width, 'lower': crown_lb, 'upper': crown_ub},
    }


def get_gamma_bounds(model_path, vnnlib_path, method='crown'):
    """Get γ-CROWN bounds."""
    gamma_binary = Path(__file__).parent.parent / "target/release/gamma"
    cmd = [
        str(gamma_binary),
        "verify",
        str(model_path),
        "--property", str(vnnlib_path),
        "--method", method,
        "--json",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            output = json.loads(result.stdout)
            bounds = output.get('output_bounds', [])
            if bounds:
                lower = np.array([b['lower'] for b in bounds])
                upper = np.array([b['upper'] for b in bounds])
                width = (upper - lower).sum()
                return {'width': width, 'lower': lower, 'upper': upper}
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    torch.set_grad_enabled(False)

    # Paths
    model_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
    vnnlib_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"

    print("=" * 70)
    print("Detailed Bound Comparison: γ-CROWN vs auto_LiRPA")
    print("=" * 70)

    # Parse input bounds
    x_L_np, x_U_np = parse_vnnlib_bounds(vnnlib_path)
    x_L = torch.tensor(x_L_np, dtype=torch.float32).reshape(1, 3, 32, 32)
    x_U = torch.tensor(x_U_np, dtype=torch.float32).reshape(1, 3, 32, 32)
    input_width = float((x_U - x_L).sum())

    print(f"\nInput total width: {input_width:.4f}")

    # Load model
    print("\nLoading model...")
    pytorch_model = load_onnx_as_pytorch(str(model_path))
    pytorch_model.eval()

    # Get auto_LiRPA bounds
    x_center = (x_L + x_U) / 2
    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    print("\n" + "-"*70)
    print("auto_LiRPA Bounds:")
    print("-"*70)

    lirpa_results = compare_ibp_and_crown(lirpa_model, bounded_x)
    for method, res in lirpa_results.items():
        print(f"\n{method}:")
        print(f"  Total width: {res['width']:.4f}")
        print(f"  Max width: {(res['upper'] - res['lower']).max():.4f}")

    # Get γ-CROWN bounds
    print("\n" + "-"*70)
    print("γ-CROWN Bounds:")
    print("-"*70)

    for method in ['ibp', 'crown-ibp', 'crown']:
        gamma_res = get_gamma_bounds(model_path, vnnlib_path, method)
        if gamma_res:
            print(f"\n{method.upper()}:")
            print(f"  Total width: {gamma_res['width']:.4f}")
            print(f"  Max width: {(gamma_res['upper'] - gamma_res['lower']).max():.4f}")

            # Compare with auto_LiRPA
            lirpa_key = method.upper()
            if lirpa_key in lirpa_results:
                ratio = gamma_res['width'] / lirpa_results[lirpa_key]['width']
                print(f"  Ratio vs auto_LiRPA: {ratio:.2f}x")

    # Per-output comparison for CROWN
    print("\n" + "-"*70)
    print("Per-Output CROWN Comparison:")
    print("-"*70)

    gamma_crown = get_gamma_bounds(model_path, vnnlib_path, 'crown')
    if gamma_crown:
        lirpa_crown = lirpa_results['CROWN']
        print(f"{'Output':>6s} | {'auto_LiRPA':>12s} | {'γ-CROWN':>12s} | {'Ratio':>8s}")
        print("-" * 50)

        for i in range(len(gamma_crown['lower'])):
            lirpa_w = lirpa_crown['upper'][i] - lirpa_crown['lower'][i]
            gamma_w = gamma_crown['upper'][i] - gamma_crown['lower'][i]
            ratio = gamma_w / lirpa_w if lirpa_w > 0 else float('inf')
            print(f"{i:>6d} | {lirpa_w:>12.4f} | {gamma_w:>12.4f} | {ratio:>8.2f}x")

        total_lirpa = lirpa_crown['width']
        total_gamma = gamma_crown['width']
        print("-" * 50)
        print(f"{'TOTAL':>6s} | {total_lirpa:>12.4f} | {total_gamma:>12.4f} | {total_gamma/total_lirpa:>8.2f}x")

    # Extract intermediate bounds from auto_LiRPA
    print("\n" + "-"*70)
    print("auto_LiRPA Intermediate CROWN Bounds:")
    print("-"*70)

    # Re-run CROWN to populate intermediate bounds
    lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')

    intermediates = []
    for name, node in lirpa_model.named_modules():
        if hasattr(node, 'lower') and node.lower is not None:
            lower = node.lower.detach().numpy()
            upper = node.upper.detach().numpy() if hasattr(node, 'upper') and node.upper is not None else lower
            width = (upper - lower).sum()
            op_type = type(node).__name__
            if width > 0:
                intermediates.append({
                    'name': name,
                    'op': op_type,
                    'shape': list(lower.shape),
                    'width': width,
                    'max_width': (upper - lower).max(),
                })

    # Sort by width descending
    intermediates.sort(key=lambda x: -x['width'])
    for info in intermediates[:15]:
        print(f"  {info['name']:40s} | {info['op']:15s} | width: {info['width']:>10.2f} | max: {info['max_width']:.4f}")


if __name__ == '__main__':
    main()
