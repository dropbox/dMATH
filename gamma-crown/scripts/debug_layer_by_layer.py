#!/usr/bin/env python3
"""
Debug layer-by-layer bound differences between γ-CROWN and auto_LiRPA.

This script isolates each layer type to identify where the bound gap originates.
"""
import sys
import json
import subprocess
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add auto_LiRPA to path
sys.path.insert(0, str(Path(__file__).parent.parent / "research/repos/auto_LiRPA"))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def test_single_conv():
    """Test a single Conv2d layer."""
    print("=" * 70)
    print("Testing single Conv2d layer")
    print("=" * 70)

    # Create a simple Conv2d layer
    conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)

    # Fixed weights for reproducibility
    torch.manual_seed(42)
    nn.init.xavier_uniform_(conv.weight)
    nn.init.zeros_(conv.bias)

    conv.eval()

    # Input bounds
    x_center = torch.randn(1, 3, 8, 8)
    eps = 0.1
    x_L = x_center - eps
    x_U = x_center + eps

    # auto_LiRPA bounds
    lirpa_model = BoundedModule(conv, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
    lb_np = lb.detach().numpy().flatten()
    ub_np = ub.detach().numpy().flatten()
    lirpa_width = (ub_np - lb_np).sum()

    print(f"\nauto_LiRPA CROWN:")
    print(f"  Total width: {lirpa_width:.4f}")
    print(f"  Max width: {(ub_np - lb_np).max():.4f}")
    print(f"  Output shape: {lb.shape}")

    # Save layer info for γ-CROWN
    layer_info = {
        'type': 'conv2d',
        'weight': conv.weight.detach().numpy().tolist(),
        'bias': conv.bias.detach().numpy().tolist() if conv.bias is not None else None,
        'stride': list(conv.stride),
        'padding': list(conv.padding),
        'input_shape': list(x_center.shape),
        'input_lower': x_L.numpy().flatten().tolist(),
        'input_upper': x_U.numpy().flatten().tolist(),
    }

    with open('/tmp/conv_test.json', 'w') as f:
        json.dump(layer_info, f)

    print(f"\nSaved layer config to /tmp/conv_test.json")
    print("Run the Rust test to compare...")

    return lirpa_width, (ub_np - lb_np).max()


def test_single_relu():
    """Test a single ReLU layer."""
    print("\n" + "=" * 70)
    print("Testing single ReLU layer")
    print("=" * 70)

    # Create a simple ReLU
    relu = nn.ReLU()

    # Input bounds (some positive, some negative, some crossing)
    x_center = torch.tensor([[-0.5, 0.5, 0.0, 1.0, -1.0]], dtype=torch.float32)
    eps = 0.3

    x_L = x_center - eps
    x_U = x_center + eps

    # auto_LiRPA bounds
    lirpa_model = BoundedModule(relu, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
    lb_np = lb.detach().numpy().flatten()
    ub_np = ub.detach().numpy().flatten()

    print(f"\nInput bounds:")
    print(f"  Lower: {x_L.numpy().flatten()}")
    print(f"  Upper: {x_U.numpy().flatten()}")

    print(f"\nauto_LiRPA CROWN output bounds:")
    print(f"  Lower: {lb_np}")
    print(f"  Upper: {ub_np}")
    print(f"  Width: {ub_np - lb_np}")


def test_conv_relu_sequence():
    """Test Conv2d -> ReLU sequence."""
    print("\n" + "=" * 70)
    print("Testing Conv2d -> ReLU sequence")
    print("=" * 70)

    class ConvRelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    model = ConvRelu()
    torch.manual_seed(42)
    nn.init.xavier_uniform_(model.conv.weight)
    nn.init.zeros_(model.conv.bias)
    model.eval()

    # Input bounds
    x_center = torch.randn(1, 3, 8, 8)
    eps = 0.1
    x_L = x_center - eps
    x_U = x_center + eps

    # auto_LiRPA bounds
    lirpa_model = BoundedModule(model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
    lb_np = lb.detach().numpy().flatten()
    ub_np = ub.detach().numpy().flatten()
    lirpa_width = (ub_np - lb_np).sum()

    print(f"\nauto_LiRPA CROWN:")
    print(f"  Total width: {lirpa_width:.4f}")
    print(f"  Max width: {(ub_np - lb_np).max():.4f}")

    return lirpa_width


def test_two_conv_relu():
    """Test Conv2d -> ReLU -> Conv2d -> ReLU sequence (like first part of ResNet)."""
    print("\n" + "=" * 70)
    print("Testing Conv2d -> ReLU -> Conv2d -> ReLU")
    print("=" * 70)

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
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    model.eval()

    # CIFAR-10 like input
    x_center = torch.randn(1, 3, 32, 32)
    eps = 0.008  # Same epsilon as cifar10_resnet
    x_L = x_center - eps
    x_U = x_center + eps

    # auto_LiRPA bounds
    lirpa_model = BoundedModule(model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    # IBP bounds
    lb_ibp, ub_ibp = lirpa_model.compute_bounds(x=(bounded_x,), method='IBP')
    ibp_width = (ub_ibp - lb_ibp).sum().item()

    # CROWN bounds
    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
    lb_np = lb.detach().numpy().flatten()
    ub_np = ub.detach().numpy().flatten()
    lirpa_width = (ub_np - lb_np).sum()

    print(f"\nInput total width: {(x_U - x_L).sum().item():.4f}")
    print(f"\nauto_LiRPA IBP:")
    print(f"  Total width: {ibp_width:.4f}")
    print(f"\nauto_LiRPA CROWN:")
    print(f"  Total width: {lirpa_width:.4f}")
    print(f"  Max width: {(ub_np - lb_np).max():.4f}")
    print(f"\nCROWN tightening: {ibp_width / lirpa_width:.2f}x")

    # Check intermediate bounds
    print("\nIntermediate bounds:")
    for name, node in lirpa_model.named_modules():
        if hasattr(node, 'lower') and node.lower is not None:
            lower = node.lower.detach().numpy()
            upper = node.upper.detach().numpy() if hasattr(node, 'upper') and node.upper is not None else lower
            width = (upper - lower).sum()
            if width > 0:
                print(f"  {name:30s} | width: {width:12.4f} | shape: {list(lower.shape)}")

    return lirpa_width


def test_resnet_block():
    """Test a minimal ResNet block with skip connection."""
    print("\n" + "=" * 70)
    print("Testing ResNet block (Conv -> ReLU -> Conv + skip)")
    print("=" * 70)

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
            out = out + identity  # Skip connection
            return out

    model = ResBlock()
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
    model.eval()

    x_center = torch.randn(1, 8, 8, 8)
    eps = 0.1
    x_L = x_center - eps
    x_U = x_center + eps

    # auto_LiRPA bounds
    lirpa_model = BoundedModule(model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    lb_ibp, ub_ibp = lirpa_model.compute_bounds(x=(bounded_x,), method='IBP')
    ibp_width = (ub_ibp - lb_ibp).sum().item()

    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
    lb_np = lb.detach().numpy().flatten()
    ub_np = ub.detach().numpy().flatten()
    crown_width = (ub_np - lb_np).sum()

    print(f"\nInput total width: {(x_U - x_L).sum().item():.4f}")
    print(f"\nauto_LiRPA IBP:")
    print(f"  Total width: {ibp_width:.4f}")
    print(f"\nauto_LiRPA CROWN:")
    print(f"  Total width: {crown_width:.4f}")
    print(f"\nCROWN tightening: {ibp_width / crown_width:.2f}x")

    # Check intermediate bounds
    print("\nIntermediate bounds from CROWN:")
    for name, node in lirpa_model.named_modules():
        if hasattr(node, 'lower') and node.lower is not None:
            lower = node.lower.detach().numpy()
            upper = node.upper.detach().numpy() if hasattr(node, 'upper') and node.upper is not None else lower
            width = (upper - lower).sum()
            if width > 0:
                print(f"  {name:30s} | width: {width:12.4f} | shape: {list(lower.shape)}")


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    test_single_conv()
    test_single_relu()
    test_conv_relu_sequence()
    test_two_conv_relu()
    test_resnet_block()
