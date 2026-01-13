#!/usr/bin/env python3
"""
Compare CROWN bounds on a simple 2-layer MLP.
This helps isolate where the bound gap originates.
"""
import sys
import json
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add auto_LiRPA to path
sys.path.insert(0, str(Path(__file__).parent.parent / "research/repos/auto_LiRPA"))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def create_simple_mlp():
    """Create a simple 2-layer MLP with known weights."""
    model = nn.Sequential(
        nn.Linear(4, 8, bias=True),
        nn.ReLU(),
        nn.Linear(8, 2, bias=True),
    )

    # Set reproducible weights
    torch.manual_seed(42)
    with torch.no_grad():
        model[0].weight.fill_(0.5)
        model[0].bias.fill_(0.1)
        model[2].weight.fill_(0.3)
        model[2].bias.fill_(-0.1)

    return model


def export_to_onnx(model, input_shape, path):
    """Export PyTorch model to ONNX."""
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=['input'],
        output_names=['output'],
        opset_version=13,
    )


def create_vnnlib(input_shape, x_L, x_U, output_size, path):
    """Create a VNN-LIB file."""
    num_inputs = np.prod(input_shape)
    with open(path, 'w') as f:
        for i in range(num_inputs):
            f.write(f"(declare-const X_{i} Real)\n")
        for i in range(output_size):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        x_L_flat = x_L.flatten()
        x_U_flat = x_U.flatten()
        for i in range(num_inputs):
            f.write(f"(assert (>= X_{i} {x_L_flat[i]:.10f}))\n")
            f.write(f"(assert (<= X_{i} {x_U_flat[i]:.10f}))\n")

        # Trivial property
        f.write("\n(assert (or ")
        for i in range(output_size):
            f.write(f"(>= Y_{i} -1e10) ")
        f.write("))\n")


def run_gamma_crown(onnx_path, vnnlib_path):
    """Run γ-CROWN verification."""
    gamma_binary = Path(__file__).parent.parent / "target/release/gamma"
    cmd = [
        str(gamma_binary),
        "verify",
        str(onnx_path),
        "--property", str(vnnlib_path),
        "--method", "crown",
        "--json",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            output = json.loads(result.stdout)
            bounds = output.get('output_bounds', [])
            if bounds:
                lower = np.array([b['lower'] for b in bounds])
                upper = np.array([b['upper'] for b in bounds])
                return lower, upper
        else:
            print(f"γ-CROWN error: {result.stderr}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(42)

    model = create_simple_mlp()
    model.eval()

    input_shape = (4,)
    eps = 0.5

    # Input bounds
    x_center = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    x_L = x_center - eps
    x_U = x_center + eps

    print("=" * 70)
    print("Comparing CROWN bounds on 2-layer MLP")
    print("=" * 70)
    print(f"Input: {x_center.numpy()}")
    print(f"Epsilon: {eps}")

    # Compute forward pass
    with torch.no_grad():
        output = model(x_center)
    print(f"Forward output: {output.numpy()}")

    # auto_LiRPA CROWN bounds
    print("\n--- auto_LiRPA CROWN (IBP=False) ---")
    lirpa_model = BoundedModule(model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    lb_crown, ub_crown = lirpa_model.compute_bounds(
        x=(bounded_x,),
        method='backward',
        IBP=False,
    )
    lirpa_lb = lb_crown.detach().numpy().flatten()
    lirpa_ub = ub_crown.detach().numpy().flatten()
    lirpa_width = (lirpa_ub - lirpa_lb).sum()

    print(f"Lower: {lirpa_lb}")
    print(f"Upper: {lirpa_ub}")
    print(f"Width: {lirpa_width:.6f}")

    # Print intermediate bounds from auto_LiRPA
    print("\nIntermediate bounds:")
    for node in lirpa_model._modules.values():
        if hasattr(node, 'lower') and node.lower is not None:
            l = node.lower.flatten()
            u = node.upper.flatten()
            width = (u - l).sum().item()
            print(f"  {node.name}: L=[{l.min().item():.4f}, {l.max().item():.4f}], "
                  f"U=[{u.min().item():.4f}, {u.max().item():.4f}], width={width:.4f}")

    # auto_LiRPA IBP bounds
    print("\n--- auto_LiRPA IBP ---")
    lb_ibp, ub_ibp = lirpa_model.compute_bounds(
        x=(bounded_x,),
        method='IBP',
    )
    ibp_lb = lb_ibp.detach().numpy().flatten()
    ibp_ub = ub_ibp.detach().numpy().flatten()
    ibp_width = (ibp_ub - ibp_lb).sum()

    print(f"Lower: {ibp_lb}")
    print(f"Upper: {ibp_ub}")
    print(f"Width: {ibp_width:.6f}")

    # auto_LiRPA CROWN-IBP bounds
    print("\n--- auto_LiRPA CROWN-IBP ---")
    # Reset model state
    lirpa_model = BoundedModule(model, x_center, device='cpu')
    lb_crown_ibp, ub_crown_ibp = lirpa_model.compute_bounds(
        x=(bounded_x,),
        method='backward',
        IBP=True,
    )
    crown_ibp_lb = lb_crown_ibp.detach().numpy().flatten()
    crown_ibp_ub = ub_crown_ibp.detach().numpy().flatten()
    crown_ibp_width = (crown_ibp_ub - crown_ibp_lb).sum()

    print(f"Lower: {crown_ibp_lb}")
    print(f"Upper: {crown_ibp_ub}")
    print(f"Width: {crown_ibp_width:.6f}")

    # γ-CROWN bounds
    print("\n--- γ-CROWN ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"
        vnnlib_path = Path(tmpdir) / "property.vnnlib"

        export_to_onnx(model, input_shape, onnx_path)
        create_vnnlib(input_shape, x_L.numpy(), x_U.numpy(), 2, vnnlib_path)

        gamma_lb, gamma_ub = run_gamma_crown(onnx_path, vnnlib_path)

        if gamma_lb is not None:
            gamma_width = (gamma_ub - gamma_lb).sum()
            print(f"Lower: {gamma_lb}")
            print(f"Upper: {gamma_ub}")
            print(f"Width: {gamma_width:.6f}")

            print("\n" + "=" * 70)
            print("COMPARISON")
            print("=" * 70)
            print(f"auto_LiRPA CROWN width:     {lirpa_width:.6f}")
            print(f"auto_LiRPA CROWN-IBP width: {crown_ibp_width:.6f}")
            print(f"auto_LiRPA IBP width:       {ibp_width:.6f}")
            print(f"γ-CROWN width:              {gamma_width:.6f}")
            print()
            print(f"Ratio γ-CROWN / CROWN:      {gamma_width / lirpa_width:.4f}x")
            print(f"Ratio γ-CROWN / CROWN-IBP:  {gamma_width / crown_ibp_width:.4f}x")

            # Check element-wise differences
            print("\nElement-wise lower bound diff (γ-CROWN - auto_LiRPA CROWN):")
            print(f"  {gamma_lb - lirpa_lb}")
            print("Element-wise upper bound diff (γ-CROWN - auto_LiRPA CROWN):")
            print(f"  {gamma_ub - lirpa_ub}")
        else:
            print("γ-CROWN failed")


if __name__ == '__main__':
    main()
