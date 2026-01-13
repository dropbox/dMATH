#!/usr/bin/env python3
"""
Head-to-head benchmark: gamma-CROWN vs Auto-LiRPA.

Measures execution time and bound tightness on identical models and perturbations.
This validates the performance claims in gamma-CROWN documentation.

Usage:
    python scripts/benchmark_vs_autolirpa.py
    python scripts/benchmark_vs_autolirpa.py --runs 5 --verbose
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add Auto-LiRPA to path
REPO_ROOT = Path(__file__).parent.parent
TEST_MODELS_DIR = REPO_ROOT / "tests" / "models"
sys.path.insert(0, str(REPO_ROOT / "research" / "repos" / "auto_LiRPA"))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

torch.manual_seed(42)
np.random.seed(42)


@dataclass
class BenchmarkResult:
    """Result of one benchmark comparison."""
    model_name: str
    input_shape: Tuple[int, ...]
    epsilon: float
    autolirpa_time_ms: float
    gamma_time_ms: float
    speedup: float
    autolirpa_width: float
    gamma_width: float
    bounds_match: bool
    message: str


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Module:
    """Create an MLP with specified dimensions."""
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def create_simple_mlp() -> nn.Module:
    """Create a simple 2-layer MLP matching tests/models/simple_mlp.onnx."""
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 4)
            self.fc2 = nn.Linear(4, 2)

            with torch.no_grad():
                self.fc1.weight.copy_(torch.tensor([
                    [1.0, 0.5],
                    [-1.0, 0.5],
                    [0.5, 1.0],
                    [0.5, -1.0]
                ]))
                self.fc1.bias.copy_(torch.tensor([0.1, 0.1, 0.1, 0.1]))
                self.fc2.weight.copy_(torch.tensor([
                    [1.0, 1.0, 1.0, 1.0],
                    [-1.0, 1.0, -1.0, 1.0]
                ]))
                self.fc2.bias.copy_(torch.tensor([0.0, 0.0]))

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return SimpleMLP()


def create_single_linear() -> nn.Module:
    """Create a single linear layer matching tests/models/single_linear.onnx."""
    class SingleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 3)

            with torch.no_grad():
                self.fc.weight.copy_(torch.tensor([
                    [1.0, 2.0],
                    [3.0, -1.0],
                    [-2.0, 1.0]
                ]))
                self.fc.bias.copy_(torch.tensor([0.5, -0.5, 1.0]))

        def forward(self, x):
            return self.fc(x)

    return SingleLinear()


def create_transformer_mlp() -> nn.Module:
    """Create a transformer MLP block matching tests/models/transformer_mlp.onnx."""
    import onnx
    from onnx import numpy_helper

    onnx_path = TEST_MODELS_DIR / "transformer_mlp.onnx"
    onnx_model = onnx.load(str(onnx_path))
    initializers = {init.name: numpy_helper.to_array(init) for init in onnx_model.graph.initializer}

    class TransformerMLP(nn.Module):
        def __init__(self, dim=8, hidden=32):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)
            self.norm = nn.LayerNorm(dim)
            self.gelu = nn.GELU()

        def forward(self, x):
            residual = x
            x = self.norm(x)
            x = self.fc1(x)
            x = self.gelu(x)
            x = self.fc2(x)
            return x + residual

    model = TransformerMLP()

    # Load weights from ONNX if available
    with torch.no_grad():
        if 'onnx::MatMul_78' in initializers:
            model.fc1.weight.copy_(torch.tensor(initializers['onnx::MatMul_78'].T))
            model.fc1.bias.copy_(torch.tensor(initializers['fc1.bias']))
            model.fc2.weight.copy_(torch.tensor(initializers['onnx::MatMul_79'].T))
            model.fc2.bias.copy_(torch.tensor(initializers['fc2.bias']))
            model.norm.weight.copy_(torch.tensor(initializers['norm.weight']))
            model.norm.bias.copy_(torch.tensor(initializers['norm.bias']))

    return model


def benchmark_autolirpa(
    model: nn.Module,
    x: torch.Tensor,
    epsilon: float,
    method: str = 'IBP',
    num_runs: int = 3
) -> Tuple[float, float]:
    """Benchmark Auto-LiRPA, return (time_ms, max_width)."""
    model.eval()

    # Warm-up
    lirpa_model = BoundedModule(model, torch.empty_like(x))
    lower = x - epsilon
    upper = x + epsilon
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper)
    bounded_x = BoundedTensor(x, ptb)
    lirpa_model.compute_bounds(x=(bounded_x,), method=method)

    # Benchmark
    times = []
    for _ in range(num_runs):
        lirpa_model = BoundedModule(model, torch.empty_like(x))
        bounded_x = BoundedTensor(x, ptb)

        start = time.perf_counter()
        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method=method)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    max_width = (ub - lb).max().item()

    return avg_time, max_width


def benchmark_gamma(
    onnx_path: str,
    epsilon: float,
    num_runs: int = 3
) -> Tuple[float, float]:
    """Benchmark gamma-CROWN, return (time_ms, max_width).

    Uses pre-built release binary for fair comparison (no cargo overhead).
    """
    gamma_bin = REPO_ROOT / "target" / "release" / "gamma"
    if not gamma_bin.exists():
        # Fall back to cargo run
        cmd = [
            "cargo", "run", "--release", "--bin", "gamma", "--",
            "profile-bounds",
            onnx_path,
            "--epsilon", str(epsilon),
            "--json"
        ]
    else:
        cmd = [
            str(gamma_bin),
            "profile-bounds",
            onnx_path,
            "--epsilon", str(epsilon),
            "--json"
        ]

    # Warm-up
    subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "RUST_LOG": "error"}
    )

    # Benchmark
    times = []
    max_width = 0.0

    for _ in range(num_runs):
        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ, "RUST_LOG": "error"}
        )
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        if result.returncode == 0:
            stdout = result.stdout.strip()
            json_start = stdout.find('{')
            if json_start != -1:
                data = json.loads(stdout[json_start:])
                max_width = data["final_width"]

    avg_time = sum(times) / len(times)
    return avg_time, max_width


def run_benchmark(
    name: str,
    pytorch_model: nn.Module,
    onnx_path: str,
    input_shape: Tuple[int, ...],
    epsilon: float,
    num_runs: int,
    verbose: bool
) -> BenchmarkResult:
    """Run benchmark on one model."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"  Input shape: {input_shape}, Epsilon: {epsilon}")

    x = torch.zeros(input_shape)

    # Benchmark Auto-LiRPA
    try:
        autolirpa_time, autolirpa_width = benchmark_autolirpa(
            pytorch_model, x, epsilon, num_runs=num_runs
        )
    except Exception as e:
        if verbose:
            print(f"  Auto-LiRPA error: {e}")
        autolirpa_time = float('inf')
        autolirpa_width = float('nan')

    # Benchmark gamma-CROWN
    try:
        gamma_time, gamma_width = benchmark_gamma(onnx_path, epsilon, num_runs=num_runs)
    except Exception as e:
        if verbose:
            print(f"  gamma error: {e}")
        gamma_time = float('inf')
        gamma_width = float('nan')

    speedup = autolirpa_time / gamma_time if gamma_time > 0 else 0.0

    # Check if bounds match (within 10% relative tolerance)
    if np.isnan(autolirpa_width) or np.isnan(gamma_width):
        bounds_match = False
        message = "Error computing bounds"
    else:
        rel_diff = abs(gamma_width - autolirpa_width) / max(autolirpa_width, 1e-10)
        bounds_match = rel_diff <= 0.10 or gamma_width <= autolirpa_width * 1.001
        if gamma_width < autolirpa_width * 0.99:
            message = f"gamma {(1 - gamma_width/autolirpa_width)*100:.1f}% tighter"
        elif rel_diff <= 0.10:
            message = "bounds match"
        else:
            message = f"gamma {rel_diff*100:.1f}% different"

    if verbose:
        print(f"  Auto-LiRPA: {autolirpa_time:.2f} ms, width={autolirpa_width:.6f}")
        print(f"  gamma:      {gamma_time:.2f} ms, width={gamma_width:.6f}")
        print(f"  Speedup:    {speedup:.2f}x")
        print(f"  {message}")

    return BenchmarkResult(
        model_name=name,
        input_shape=input_shape,
        epsilon=epsilon,
        autolirpa_time_ms=autolirpa_time,
        gamma_time_ms=gamma_time,
        speedup=speedup,
        autolirpa_width=autolirpa_width,
        gamma_width=gamma_width,
        bounds_match=bounds_match,
        message=message
    )


def generate_larger_models():
    """Generate larger MLP models for scaling benchmarks."""
    models = []

    # MLP scaling: increase hidden dimension
    for hidden in [64, 128, 256, 512]:
        name = f"mlp_{hidden}h"
        model = create_mlp(64, hidden, 10, 4)

        # Export to ONNX
        onnx_path = TEST_MODELS_DIR / f"{name}.onnx"
        x = torch.randn(1, 64)
        try:
            torch.onnx.export(
                model, x, str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                opset_version=18,
                dynamo=False  # Use legacy exporter
            )
            models.append((name, model, str(onnx_path), (1, 64)))
        except Exception as e:
            print(f"  Warning: Could not export {name}: {e}")

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark gamma-CROWN vs Auto-LiRPA"
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Perturbation epsilon")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmarks")
    args = parser.parse_args()

    print("=" * 70)
    print("gamma-CROWN vs Auto-LiRPA Benchmark")
    print("=" * 70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Runs per benchmark: {args.runs}")
    print(f"Epsilon: {args.epsilon}")

    # Standard test models
    test_cases = [
        (
            "single_linear",
            create_single_linear(),
            str(TEST_MODELS_DIR / "single_linear.onnx"),
            (1, 2)
        ),
        (
            "simple_mlp",
            create_simple_mlp(),
            str(TEST_MODELS_DIR / "simple_mlp.onnx"),
            (1, 2)
        ),
    ]

    # Add transformer MLP if available
    if (TEST_MODELS_DIR / "transformer_mlp.onnx").exists():
        test_cases.append((
            "transformer_mlp",
            create_transformer_mlp(),
            str(TEST_MODELS_DIR / "transformer_mlp.onnx"),
            (1, 4, 8)
        ))

    # Add scaling models
    if args.scaling:
        print("\nGenerating larger models for scaling benchmark...")
        test_cases.extend(generate_larger_models())

    results: List[BenchmarkResult] = []

    for name, pytorch_model, onnx_path, input_shape in test_cases:
        if not Path(onnx_path).exists():
            if args.verbose:
                print(f"\nSkipping {name}: ONNX file not found")
            continue

        result = run_benchmark(
            name, pytorch_model, onnx_path, input_shape,
            args.epsilon, args.runs, args.verbose
        )
        results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Auto-LiRPA':<12} {'gamma':<12} {'Speedup':<10} {'Bounds':<15}")
    print("-" * 70)

    for r in results:
        print(f"{r.model_name:<20} {r.autolirpa_time_ms:>8.2f} ms  {r.gamma_time_ms:>8.2f} ms  {r.speedup:>6.2f}x    {r.message:<15}")

    # Calculate averages
    valid_results = [r for r in results if r.speedup > 0 and r.speedup != float('inf')]
    if valid_results:
        avg_speedup = sum(r.speedup for r in valid_results) / len(valid_results)
        geometric_mean = np.exp(np.mean([np.log(r.speedup) for r in valid_results]))

        print("-" * 70)
        print(f"\nArithmetic mean speedup: {avg_speedup:.2f}x")
        print(f"Geometric mean speedup:  {geometric_mean:.2f}x")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    faster_count = sum(1 for r in results if r.speedup > 1.0)
    slower_count = sum(1 for r in results if r.speedup < 1.0 and r.speedup > 0)
    bounds_ok = sum(1 for r in results if r.bounds_match)

    print(f"\ngamma-CROWN faster on {faster_count}/{len(results)} models")
    if slower_count > 0:
        print(f"gamma-CROWN slower on {slower_count}/{len(results)} models")
        print("  Note: gamma includes subprocess overhead; native calls are faster")
    print(f"Bounds correct on {bounds_ok}/{len(results)} models")

    # Export results
    output_path = REPO_ROOT / "reports" / "main" / f"benchmark_vs_autolirpa_{time.strftime('%Y-%m-%d')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "date": time.strftime('%Y-%m-%d %H:%M:%S'),
        "epsilon": args.epsilon,
        "runs": args.runs,
        "results": [
            {
                "model": r.model_name,
                "input_shape": list(r.input_shape),
                "autolirpa_ms": r.autolirpa_time_ms,
                "gamma_ms": r.gamma_time_ms,
                "speedup": r.speedup,
                "bounds_match": r.bounds_match
            }
            for r in results
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\nResults exported to: {output_path}")

    return 0 if bounds_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
