#!/usr/bin/env python3
"""
Compare γ-CROWN vs Auto-LiRPA bounds on ACAS-Xu 1_1 model.

Run: source .venv/bin/activate && python benchmarks/test_autolirpa_acasxu.py
"""

import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


def load_nnet(path):
    """Load NNet format model."""
    with open(path, 'r') as f:
        lines = [l for l in f.readlines() if not l.startswith('//')]

    header = lines[0].strip().strip(',').split(',')
    num_layers = int(header[0])
    layer_sizes = [int(x.strip()) for x in lines[1].strip().strip(',').split(',') if x.strip()]

    line_idx = 7  # Weight data starts at line 7
    weights = []
    biases = []
    for i in range(num_layers):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]

        W = np.zeros((out_dim, in_dim))
        for j in range(out_dim):
            row = [float(x) for x in lines[line_idx].strip().strip(',').split(',') if x.strip()]
            W[j] = row
            line_idx += 1

        b = np.zeros(out_dim)
        for j in range(out_dim):
            b[j] = float(lines[line_idx].strip().strip(',').split(',')[0])
            line_idx += 1

        weights.append(W)
        biases.append(b)

    return weights, biases, layer_sizes


def main():
    # Load model
    weights, biases, layer_sizes = load_nnet('tests/models/acasxu_1_1.nnet')
    print(f'Loaded ACAS-Xu 1_1: {len(weights)} layers, sizes: {layer_sizes}')

    # Create PyTorch model
    layers = []
    for i, (W, b) in enumerate(zip(weights, biases)):
        linear = nn.Linear(W.shape[1], W.shape[0])
        linear.weight.data = torch.tensor(W, dtype=torch.float32)
        linear.bias.data = torch.tensor(b, dtype=torch.float32)
        layers.append(linear)
        if i < len(weights) - 1:
            layers.append(nn.ReLU())

    model = nn.Sequential(*layers)
    model.eval()

    # VNNLIB Property 1 bounds (normalized space)
    lower = torch.tensor([[0.6, -0.5, -0.5, 0.45, -0.5]])
    upper = torch.tensor([[0.679857769, 0.5, 0.5, 0.5, -0.45]])
    center = (lower + upper) / 2

    print(f'\nInput bounds:')
    for i in range(5):
        print(f'  X_{i}: [{lower[0, i].item():.6f}, {upper[0, i].item():.6f}]')

    # Create bounded model
    bounded_model = BoundedModule(model, center)
    ptb = PerturbationLpNorm(x_L=lower, x_U=upper)
    bounded_input = BoundedTensor(center, ptb)

    # Compute bounds
    print('\n=== Auto-LiRPA Bounds ===')

    methods = [
        ('IBP', 'IBP'),
        ('CROWN (backward)', 'backward'),
        ('CROWN-Optimized (alpha)', 'CROWN-Optimized'),
    ]

    for name, method in methods:
        try:
            lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method=method)
            print(f'\n{name}:')
            for i in range(5):
                print(f'  Y_{i}: [{lb[0, i].item():.4f}, {ub[0, i].item():.4f}]')
            print(f'  Total width: {(ub - lb).sum().item():.4f}')
        except Exception as e:
            print(f'\n{name}: Error - {e}')

    # Property threshold
    threshold = 3.991125645861615
    print(f'\n=== Property Status ===')
    print(f'Threshold: {threshold:.6f}')
    print(f'Property: Y_0 >= {threshold} is UNSAFE')
    print(f'To verify safety: need upper_bound[0] < {threshold}')


if __name__ == '__main__':
    main()
