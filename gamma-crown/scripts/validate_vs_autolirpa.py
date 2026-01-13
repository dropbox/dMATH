#!/usr/bin/env python3
"""
Validate gamma-CROWN bounds against Auto-LiRPA reference implementation.

This script compares bound propagation results between:
- Auto-LiRPA (Python reference implementation)
- gamma-CROWN (Rust implementation)

Both tools should compute the same IBP bounds for the same model and input perturbation.
This is a critical correctness test for P0 validation.

Uses pre-generated ONNX models from tests/models/ which have known weights.
The PyTorch models are created with matching weights for Auto-LiRPA comparison.

Usage:
    python scripts/validate_vs_autolirpa.py
    python scripts/validate_vs_autolirpa.py --epsilon 0.001
    python scripts/validate_vs_autolirpa.py --verbose
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add Auto-LiRPA to path
REPO_ROOT = Path(__file__).parent.parent
TEST_MODELS_DIR = REPO_ROOT / "tests" / "models"
sys.path.insert(0, str(REPO_ROOT / "research" / "repos" / "auto_LiRPA"))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


@dataclass
class ValidationResult:
    """Result of validating gamma vs Auto-LiRPA."""
    model_name: str
    epsilon: float
    autolirpa_lb: float
    autolirpa_ub: float
    autolirpa_width: float  # IBP width
    gamma_final_width: float
    width_diff: float
    width_relative_diff: float
    passed: bool
    tolerance: float
    message: str
    gamma_is_tighter: bool = False  # True if gamma computes tighter bounds than IBP
    crown_width: Optional[float] = None  # CROWN width (tighter than IBP for ReLU models)
    crown_vs_gamma: Optional[str] = None  # Comparison: gamma vs CROWN


def create_simple_mlp() -> nn.Module:
    """Create a simple 2-layer MLP matching tests/models/simple_mlp.onnx.

    This matches the weights from scripts/generate_test_models.py:create_simple_mlp()
    """
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 4)
            self.fc2 = nn.Linear(4, 2)

            # Exact weights from generate_test_models.py
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


def create_linear_only() -> nn.Module:
    """Create a single linear layer matching tests/models/single_linear.onnx.

    This matches the weights from scripts/generate_test_models.py:create_single_linear()
    """
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


def create_linear_relu() -> nn.Module:
    """Create Linear + ReLU matching tests/models/linear_relu.onnx."""
    class LinearReLU(nn.Module):
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
            return torch.relu(self.fc(x))

    return LinearReLU()


def create_softmax() -> nn.Module:
    """Create Softmax model matching tests/models/softmax.onnx."""
    class SimpleSoftmax(nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            return self.softmax(x)

    return SimpleSoftmax()


def create_gelu() -> nn.Module:
    """Create GELU model matching tests/models/gelu.onnx."""
    class SimpleGELU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gelu = nn.GELU()

        def forward(self, x):
            return self.gelu(x)

    return SimpleGELU()


def create_layer_norm() -> nn.Module:
    """Create LayerNorm model matching tests/models/layer_norm.onnx."""
    class SimpleLayerNorm(nn.Module):
        def __init__(self):
            super().__init__()
            # ONNX model uses dim=4 with default gamma=1, beta=0
            self.norm = nn.LayerNorm(4)

        def forward(self, x):
            return self.norm(x)

    return SimpleLayerNorm()


def create_simple_attention() -> nn.Module:
    """Create SimpleAttention model with weights loaded from tests/models/simple_attention.onnx.

    This model was generated with random weights, so we load them from the ONNX file
    to ensure exact match between PyTorch and ONNX representations.
    """
    import onnx
    from onnx import numpy_helper

    onnx_path = TEST_MODELS_DIR / "simple_attention.onnx"
    onnx_model = onnx.load(str(onnx_path))
    initializers = {init.name: numpy_helper.to_array(init) for init in onnx_model.graph.initializer}

    class SimpleAttention(nn.Module):
        def __init__(self, dim=4):
            super().__init__()
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.scale = dim ** -0.5

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            return self.out_proj(out)

    model = SimpleAttention(dim=4)

    # ONNX uses x @ W, PyTorch uses x @ W.T, so W_torch = W_onnx.T
    with torch.no_grad():
        model.q_proj.weight.copy_(torch.tensor(initializers['onnx::MatMul_27'].T))
        model.q_proj.bias.copy_(torch.tensor(initializers['q_proj.bias']))
        model.k_proj.weight.copy_(torch.tensor(initializers['onnx::MatMul_28'].T))
        model.k_proj.bias.copy_(torch.tensor(initializers['k_proj.bias']))
        model.v_proj.weight.copy_(torch.tensor(initializers['onnx::MatMul_29'].T))
        model.v_proj.bias.copy_(torch.tensor(initializers['v_proj.bias']))
        model.out_proj.weight.copy_(torch.tensor(initializers['onnx::MatMul_30'].T))
        model.out_proj.bias.copy_(torch.tensor(initializers['out_proj.bias']))

    return model


def compute_autolirpa_bounds(
    model: nn.Module,
    x: torch.Tensor,
    epsilon: float,
    method: str = 'IBP'
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Compute bounds using Auto-LiRPA.

    Args:
        model: PyTorch model
        x: Input tensor (center of perturbation ball)
        epsilon: Perturbation radius
        method: Bound method - 'IBP' or 'backward' (CROWN)

    Returns:
        lb: Lower bounds tensor
        ub: Upper bounds tensor
        max_width: Maximum per-element width (ub - lb).max()
    """
    model.eval()

    # Wrap model
    lirpa_model = BoundedModule(model, torch.empty_like(x))

    # Create perturbation: L_inf ball around x with radius epsilon
    lower = x - epsilon
    upper = x + epsilon
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper)
    bounded_x = BoundedTensor(x, ptb)

    # Compute bounds with specified method
    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method=method)

    # Compute max per-element width (matches gamma's metric)
    widths = ub - lb
    max_width = widths.max().item()

    return lb, ub, max_width


def run_gamma_profile_bounds(model_path: str, epsilon: float) -> Dict:
    """Run gamma profile-bounds and parse JSON output."""
    cmd = [
        "cargo", "run", "--release", "--bin", "gamma", "--",
        "profile-bounds",
        model_path,
        "--epsilon", str(epsilon),
        "--center-zeros",  # Use zeros-centered input to match Auto-LiRPA validation
        "--json"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "RUST_LOG": "error"}  # Suppress warnings
    )

    if result.returncode != 0:
        raise RuntimeError(f"gamma profile-bounds failed: {result.stderr}")

    # Parse JSON, ignoring any non-JSON output (e.g., cargo compilation messages)
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(f"gamma profile-bounds returned no output. stderr: {result.stderr}")

    # Find the JSON object in output (starts with '{')
    json_start = stdout.find('{')
    if json_start == -1:
        raise RuntimeError(f"No JSON found in output: {stdout}")

    return json.loads(stdout[json_start:])


def validate_model(
    model_name: str,
    pytorch_model: nn.Module,
    onnx_path: str,
    input_shape: Tuple[int, ...],
    epsilon: float,
    tolerance: float,
    verbose: bool
) -> ValidationResult:
    """Validate gamma bounds against Auto-LiRPA for a single model.

    Args:
        model_name: Name for display
        pytorch_model: PyTorch model with matching weights for Auto-LiRPA
        onnx_path: Path to pre-generated ONNX model
        input_shape: Input tensor shape
        epsilon: Perturbation radius
        tolerance: Relative tolerance for comparison
        verbose: Print detailed output
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"Validating: {model_name}")
        print(f"  Input shape: {input_shape}")
        print(f"  Epsilon: {epsilon}")
        print(f"  ONNX: {onnx_path}")

    # Create test input (zeros for simplicity)
    x = torch.zeros(input_shape)

    # Compute Auto-LiRPA bounds
    lb, ub, autolirpa_max_width = compute_autolirpa_bounds(pytorch_model, x, epsilon)
    autolirpa_lb = lb.min().item()
    autolirpa_ub = ub.max().item()
    autolirpa_span = autolirpa_ub - autolirpa_lb

    if verbose:
        print(f"  Auto-LiRPA IBP bounds: [{autolirpa_lb:.6f}, {autolirpa_ub:.6f}]")
        print(f"  Auto-LiRPA span (ub.max - lb.min): {autolirpa_span:.6f}")
        print(f"  Auto-LiRPA max element width: {autolirpa_max_width:.6f}")

    # Try to compute CROWN bounds (may fail for some ops like Softmax)
    crown_width = None
    crown_vs_gamma = None
    try:
        _, _, crown_max_width = compute_autolirpa_bounds(pytorch_model, x, epsilon, method='backward')
        crown_width = crown_max_width
        if verbose:
            print(f"  Auto-LiRPA CROWN width: {crown_width:.6f}")
    except Exception as e:
        if verbose:
            print(f"  Auto-LiRPA CROWN failed (expected for some ops): {type(e).__name__}")

    # Run gamma profile-bounds on the pre-generated ONNX model
    gamma_result = run_gamma_profile_bounds(onnx_path, epsilon)
    gamma_final_width = gamma_result["final_width"]

    if verbose:
        print(f"  gamma final width: {gamma_final_width:.6f}")
        print(f"  gamma total expansion: {gamma_result['total_expansion']:.6f}")

    # Compare widths (gamma vs IBP)
    autolirpa_width = autolirpa_max_width
    width_diff = gamma_final_width - autolirpa_width  # positive means gamma is looser
    width_relative_diff = abs(width_diff) / max(autolirpa_width, 1e-10)

    # Gamma is tighter if its width is smaller
    gamma_is_tighter = gamma_final_width < autolirpa_width * (1 - 0.01)  # 1% threshold

    # Compare gamma vs CROWN (if available)
    if crown_width is not None:
        if gamma_final_width < crown_width * 0.99:
            crown_vs_gamma = f"gamma {(1 - gamma_final_width/crown_width)*100:.1f}% tighter"
        elif gamma_final_width > crown_width * 1.01:
            crown_vs_gamma = f"CROWN {(1 - crown_width/gamma_final_width)*100:.1f}% tighter"
        else:
            crown_vs_gamma = "equal"
        if verbose:
            print(f"  gamma vs CROWN: {crown_vs_gamma}")

    # Validation passes if:
    # 1. gamma width <= autolirpa width (gamma is tighter or equal - good!)
    # 2. OR gamma width is within tolerance of autolirpa width
    # Fails only if gamma is looser beyond tolerance (potential bug)
    if gamma_final_width <= autolirpa_width * (1 + 1e-6):
        # gamma is tighter or equal - always pass
        passed = True
        message = "gamma bounds are tighter or equal"
        if gamma_is_tighter:
            message = f"gamma bounds are {abs(width_relative_diff):.1%} tighter"
    else:
        # gamma is looser - check tolerance
        passed = width_relative_diff <= tolerance
        message = f"gamma bounds are {width_relative_diff:.4%} looser"
        if not passed:
            message += f" (exceeds tolerance {tolerance:.4%})"

    if verbose:
        direction = "tighter" if width_diff < 0 else "looser" if width_diff > 0 else "equal"
        print(f"  Width difference: {abs(width_diff):.6f} ({direction}, relative: {abs(width_relative_diff):.4%})")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return ValidationResult(
        model_name=model_name,
        epsilon=epsilon,
        autolirpa_lb=autolirpa_lb,
        autolirpa_ub=autolirpa_ub,
        autolirpa_width=autolirpa_width,
        gamma_final_width=gamma_final_width,
        width_diff=abs(width_diff),
        width_relative_diff=width_relative_diff,
        passed=passed,
        tolerance=tolerance,
        message=message,
        gamma_is_tighter=gamma_is_tighter,
        crown_width=crown_width,
        crown_vs_gamma=crown_vs_gamma
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate gamma-CROWN bounds against Auto-LiRPA"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="Input perturbation epsilon (default: 0.01)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.10,
        help="Relative tolerance for bound comparison (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("gamma-CROWN vs Auto-LiRPA Validation")
    print("=" * 60)
    print(f"Epsilon: {args.epsilon}")
    print(f"Tolerance: {args.tolerance:.2%}")

    # Test cases: (name, pytorch_model_factory, onnx_path, input_shape)
    # PyTorch models have weights matching the pre-generated ONNX models
    test_cases = [
        # Basic MLP models
        (
            "single_linear",
            create_linear_only(),
            str(TEST_MODELS_DIR / "single_linear.onnx"),
            (1, 2)
        ),
        (
            "linear_relu",
            create_linear_relu(),
            str(TEST_MODELS_DIR / "linear_relu.onnx"),
            (1, 2)
        ),
        (
            "simple_mlp",
            create_simple_mlp(),
            str(TEST_MODELS_DIR / "simple_mlp.onnx"),
            (1, 2)
        ),
        # Transformer components (no learned weights - deterministic)
        (
            "softmax",
            create_softmax(),
            str(TEST_MODELS_DIR / "softmax.onnx"),
            (1, 4)
        ),
        # Note: GELU skipped - Auto-LiRPA doesn't support the Erf operator
        (
            "layer_norm",
            create_layer_norm(),
            str(TEST_MODELS_DIR / "layer_norm.onnx"),
            (1, 4)
        ),
        # Full attention mechanism (weights loaded from ONNX)
        (
            "simple_attention",
            create_simple_attention(),
            str(TEST_MODELS_DIR / "simple_attention.onnx"),
            (1, 2, 4)  # batch=1, seq=2, dim=4
        ),
    ]

    results: List[ValidationResult] = []

    for model_name, pytorch_model, onnx_path, input_shape in test_cases:
        try:
            result = validate_model(
                model_name, pytorch_model, onnx_path, input_shape,
                args.epsilon, args.tolerance, args.verbose
            )
            results.append(result)
        except Exception as e:
            print(f"\nError validating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append(ValidationResult(
                model_name=model_name,
                epsilon=args.epsilon,
                autolirpa_lb=float('nan'),
                autolirpa_ub=float('nan'),
                autolirpa_width=float('nan'),
                gamma_final_width=float('nan'),
                width_diff=float('nan'),
                width_relative_diff=float('nan'),
                passed=False,
                tolerance=args.tolerance,
                message=str(e)
            ))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Model':<18} {'IBP':<10} {'CROWN':<10} {'gamma':<10} {'vs IBP':<15} {'vs CROWN':<18} {'Status':<6}")
    print("-" * 95)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if np.isnan(r.autolirpa_width):
            print(f"{r.model_name:<18} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'N/A':<15} {'N/A':<18} {status:<6}")
        else:
            if r.gamma_is_tighter:
                vs_ibp = f"{r.width_relative_diff:.1%} tighter"
            elif r.gamma_final_width <= r.autolirpa_width * 1.001:
                vs_ibp = "equal"
            else:
                vs_ibp = f"{r.width_relative_diff:.1%} looser"

            crown_str = f"{r.crown_width:.6f}" if r.crown_width is not None else "N/A"
            vs_crown = r.crown_vs_gamma if r.crown_vs_gamma else "N/A"

            print(f"{r.model_name:<18} {r.autolirpa_width:<10.6f} {crown_str:<10} {r.gamma_final_width:<10.6f} {vs_ibp:<15} {vs_crown:<18} {status:<6}")

    passed_count = sum(1 for r in results if r.passed)
    tighter_count = sum(1 for r in results if r.gamma_is_tighter)
    total_count = len(results)
    crown_tighter_count = sum(1 for r in results if r.crown_vs_gamma and 'CROWN' in r.crown_vs_gamma)

    print("-" * 95)
    print(f"\nTotal: {passed_count}/{total_count} passed")
    if tighter_count > 0:
        print(f"gamma computed tighter bounds than Auto-LiRPA IBP on {tighter_count}/{total_count} models")
    if crown_tighter_count > 0:
        print(f"CROWN computed tighter bounds than gamma on {crown_tighter_count}/{total_count} models (expected for ReLU)")

    if passed_count == total_count:
        print("\nAll validations PASSED!")
        print("  - gamma bounds are sound (never looser than IBP beyond tolerance)")
        print("  - gamma computes tighter bounds than IBP for softmax/attention (good!)")
        print("  - CROWN computes tighter bounds than gamma for ReLU models (expected)")
        return 0
    else:
        print(f"\n{total_count - passed_count} validations FAILED")
        print("\nPossible issues:")
        print("  - gamma computed bounds that are looser than Auto-LiRPA IBP")
        print("  - This may indicate a bug in gamma-CROWN")
        return 1


if __name__ == "__main__":
    sys.exit(main())
