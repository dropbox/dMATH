#!/usr/bin/env python3
"""
Isolate the bound gap by testing individual layer types and combinations.

This script exports simple models to ONNX and compares γ-CROWN vs auto_LiRPA bounds.
"""
import sys
import json
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn
import onnx
from pathlib import Path

# Add auto_LiRPA to path
sys.path.insert(0, str(Path(__file__).parent.parent / "research/repos/auto_LiRPA"))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


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


def create_vnnlib(input_shape, input_lower, input_upper, output_size, path):
    """Create a simple VNN-LIB file for verification."""
    num_inputs = np.prod(input_shape)
    with open(path, 'w') as f:
        # Declare inputs
        for i in range(num_inputs):
            f.write(f"(declare-const X_{i} Real)\n")
        # Declare outputs
        for i in range(output_size):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Input bounds
        for i in range(num_inputs):
            f.write(f"(assert (>= X_{i} {input_lower.flatten()[i]:.10f}))\n")
            f.write(f"(assert (<= X_{i} {input_upper.flatten()[i]:.10f}))\n")

        # Trivial property (all outputs can be anything)
        f.write("\n(assert (or ")
        for i in range(output_size):
            f.write(f"(>= Y_{i} -1e10) ")
        f.write("))\n")


def run_gamma_crown(onnx_path, vnnlib_path):
    """Run γ-CROWN verification and return output bounds."""
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
                lower = [b['lower'] for b in bounds]
                upper = [b['upper'] for b in bounds]
                return np.array(lower), np.array(upper)
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def compare_model(name, model, input_shape, eps=0.1):
    """Compare bounds for a model between γ-CROWN and auto_LiRPA."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")

    model.eval()

    # Create input bounds
    x_center = torch.randn(1, *input_shape)
    x_L = x_center - eps
    x_U = x_center + eps

    # Get output shape
    with torch.no_grad():
        output = model(x_center)
    output_size = output.numel()

    # auto_LiRPA bounds
    lirpa_model = BoundedModule(model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    lirpa_lb, lirpa_ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
    lirpa_lb = lirpa_lb.detach().numpy().flatten()
    lirpa_ub = lirpa_ub.detach().numpy().flatten()
    lirpa_width = (lirpa_ub - lirpa_lb).sum()

    print(f"\nauto_LiRPA CROWN:")
    print(f"  Total width: {lirpa_width:.4f}")
    print(f"  Max width: {(lirpa_ub - lirpa_lb).max():.4f}")

    # Export to ONNX and create VNN-LIB
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"
        vnnlib_path = Path(tmpdir) / "property.vnnlib"

        export_to_onnx(model, input_shape, onnx_path)
        create_vnnlib(input_shape, x_L.numpy(), x_U.numpy(), output_size, vnnlib_path)

        # γ-CROWN bounds
        gamma_lb, gamma_ub = run_gamma_crown(onnx_path, vnnlib_path)

        if gamma_lb is not None:
            gamma_width = (gamma_ub - gamma_lb).sum()
            print(f"\nγ-CROWN CROWN:")
            print(f"  Total width: {gamma_width:.4f}")
            print(f"  Max width: {(gamma_ub - gamma_lb).max():.4f}")

            ratio = gamma_width / lirpa_width if lirpa_width > 0 else float('inf')
            print(f"\n*** Ratio (γ-CROWN / auto_LiRPA): {ratio:.2f}x ***")

            if ratio > 1.5:
                print("*** SIGNIFICANT GAP DETECTED ***")

            return ratio
        else:
            print("γ-CROWN failed to compute bounds")
            return None


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(42)

    results = {}

    # Test 1: Single Linear layer
    class SingleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 16, bias=True)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

    model = SingleLinear()
    nn.init.xavier_uniform_(model.fc.weight)
    results['Linear'] = compare_model("Single Linear", model, (64,), eps=0.1)

    # Test 2: Single Conv2d layer
    class SingleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)

        def forward(self, x):
            return self.conv(x)

    model = SingleConv()
    nn.init.xavier_uniform_(model.conv.weight)
    nn.init.zeros_(model.conv.bias)
    results['Conv2d'] = compare_model("Single Conv2d", model, (3, 8, 8), eps=0.1)

    # Test 3: Conv2d + ReLU
    class ConvRelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    model = ConvRelu()
    nn.init.xavier_uniform_(model.conv.weight)
    nn.init.zeros_(model.conv.bias)
    results['Conv2d+ReLU'] = compare_model("Conv2d + ReLU", model, (3, 8, 8), eps=0.1)

    # Test 4: Two Conv2d + ReLU layers (like ResNet blocks)
    class TwoConvRelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=True)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=True)
            self.relu2 = nn.ReLU()

        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            return x

    model = TwoConvRelu()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    results['TwoConvRelu'] = compare_model("Two Conv2d+ReLU", model, (3, 16, 16), eps=0.1)

    # Test 5: Simple residual block
    class ResBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)

        def forward(self, x):
            identity = x
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
            out = out + identity
            return out

    model = ResBlock()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
    results['ResBlock'] = compare_model("Residual Block", model, (8, 8, 8), eps=0.1)

    # Test 6: Linear -> ReLU -> Linear (simple MLP)
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 32, bias=True)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 16, bias=True)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleMLP()
    nn.init.xavier_uniform_(model.fc1.weight)
    nn.init.xavier_uniform_(model.fc2.weight)
    results['MLP'] = compare_model("Simple MLP (Linear+ReLU+Linear)", model, (64,), eps=0.1)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, ratio in results.items():
        if ratio is not None:
            status = "OK" if ratio < 1.5 else "GAP!"
            print(f"  {name:30s}: {ratio:.2f}x {status}")
        else:
            print(f"  {name:30s}: FAILED")


if __name__ == '__main__':
    main()
