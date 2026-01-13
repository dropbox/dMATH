#!/usr/bin/env python3
"""
Debug what intermediate bounds auto_LiRPA uses during CROWN backward.
"""
import sys
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


def main():
    torch.set_grad_enabled(False)

    # Paths
    model_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
    vnnlib_path = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"

    print("=" * 70)
    print("Debug auto_LiRPA Intermediate Bounds")
    print("=" * 70)

    # Parse input bounds
    x_L_np, x_U_np = parse_vnnlib_bounds(vnnlib_path)
    x_L = torch.tensor(x_L_np, dtype=torch.float32).reshape(1, 3, 32, 32)
    x_U = torch.tensor(x_U_np, dtype=torch.float32).reshape(1, 3, 32, 32)

    # Load model
    print("\nLoading model...")
    pytorch_model = load_onnx_as_pytorch(str(model_path))
    pytorch_model.eval()

    x_center = (x_L + x_U) / 2
    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    # Check what bounds are before any computation
    print("\n" + "-"*70)
    print("Before computing bounds:")
    print("-"*70)
    for name, node in lirpa_model.named_modules():
        if hasattr(node, 'lower') and node.lower is not None:
            lower = node.lower.detach().numpy()
            upper = node.upper.detach().numpy() if hasattr(node, 'upper') and node.upper is not None else lower
            width = (upper - lower).sum()
            print(f"  {name:40s} | width: {width:.4f}")

    # Compute IBP bounds
    print("\n" + "-"*70)
    print("After IBP:")
    print("-"*70)
    lb_ibp, ub_ibp = lirpa_model.compute_bounds(x=(bounded_x,), method='IBP')
    print(f"Output width: {(ub_ibp - lb_ibp).sum().item():.4f}")

    print("\nIntermediate bounds:")
    for name, node in lirpa_model.named_modules():
        if hasattr(node, 'lower') and node.lower is not None:
            lower = node.lower.detach().numpy()
            upper = node.upper.detach().numpy() if hasattr(node, 'upper') and node.upper is not None else lower
            width = (upper - lower).sum()
            op_type = type(node).__name__
            if width > 0:
                print(f"  {name:40s} | {op_type:15s} | width: {width:>10.2f}")

    # Now compute CROWN-IBP bounds
    print("\n" + "-"*70)
    print("After CROWN-IBP:")
    print("-"*70)
    # Reset model to clear previous bounds
    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    lb_ci, ub_ci = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN-IBP')
    print(f"Output width: {(ub_ci - lb_ci).sum().item():.4f}")

    print("\nIntermediate bounds:")
    for name, node in lirpa_model.named_modules():
        if hasattr(node, 'lower') and node.lower is not None:
            lower = node.lower.detach().numpy()
            upper = node.upper.detach().numpy() if hasattr(node, 'upper') and node.upper is not None else lower
            width = (upper - lower).sum()
            op_type = type(node).__name__
            if width > 0:
                print(f"  {name:40s} | {op_type:15s} | width: {width:>10.2f}")

    # Now compute CROWN bounds
    print("\n" + "-"*70)
    print("After CROWN:")
    print("-"*70)
    # Reset model
    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
    print(f"Output width: {(ub - lb).sum().item():.4f}")

    print("\nIntermediate bounds:")
    intermediate_info = []
    for name, node in lirpa_model.named_modules():
        if hasattr(node, 'lower') and node.lower is not None:
            lower = node.lower.detach().numpy()
            upper = node.upper.detach().numpy() if hasattr(node, 'upper') and node.upper is not None else lower
            width = (upper - lower).sum()
            max_width = (upper - lower).max()
            op_type = type(node).__name__
            if width > 0:
                intermediate_info.append({
                    'name': name,
                    'op': op_type,
                    'width': width,
                    'max_width': max_width,
                    'shape': list(lower.shape)
                })

    # Sort by execution order (approximately by name)
    intermediate_info.sort(key=lambda x: x['name'])
    for info in intermediate_info:
        print(f"  {info['name']:40s} | {info['op']:15s} | width: {info['width']:>10.2f} | max: {info['max_width']:.4f}")

    # Check what method CROWN actually uses for intermediate bounds
    print("\n" + "-"*70)
    print("Checking CROWN intermediate bound method...")
    print("-"*70)

    # In auto_LiRPA, check what the IBP attribute does
    # From documentation: IBP=False means "do not use IBP for intermediate bounds"
    # When method='CROWN' and IBP=False, it uses CROWN for everything

    # Let's see if there's a difference between different settings
    print("\nComparing CROWN with different settings:")

    # Default CROWN (should use CROWN for intermediates)
    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)
    lb1, ub1 = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
    w1 = (ub1 - lb1).sum().item()
    print(f"  CROWN (default): {w1:.4f}")

    # CROWN with explicit IBP=False
    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)
    lb2, ub2 = lirpa_model.compute_bounds(x=(bounded_x,), method='backward', IBP=False)
    w2 = (ub2 - lb2).sum().item()
    print(f"  backward, IBP=False: {w2:.4f}")

    # backward with IBP=True (should be CROWN-IBP)
    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)
    lb3, ub3 = lirpa_model.compute_bounds(x=(bounded_x,), method='backward', IBP=True)
    w3 = (ub3 - lb3).sum().item()
    print(f"  backward, IBP=True: {w3:.4f}")


if __name__ == '__main__':
    main()
