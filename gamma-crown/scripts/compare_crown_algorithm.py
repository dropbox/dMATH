#!/usr/bin/env python3
"""
Compare the CROWN algorithm step by step between γ-CROWN and auto_LiRPA.
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
    print("Detailed CROWN Algorithm Comparison")
    print("=" * 70)

    # Parse input bounds
    x_L_np, x_U_np = parse_vnnlib_bounds(vnnlib_path)
    x_L = torch.tensor(x_L_np, dtype=torch.float32).reshape(1, 3, 32, 32)
    x_U = torch.tensor(x_U_np, dtype=torch.float32).reshape(1, 3, 32, 32)

    # Load model
    pytorch_model = load_onnx_as_pytorch(str(model_path))
    pytorch_model.eval()

    x_center = (x_L + x_U) / 2
    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    # First compute IBP to get initial bounds
    print("\n" + "-"*70)
    print("Step 1: IBP Forward Pass (Initial Bounds)")
    print("-"*70)

    lb_ibp, ub_ibp = lirpa_model.compute_bounds(x=(bounded_x,), method='IBP')
    print(f"Output width (IBP): {(ub_ibp - lb_ibp).sum().item():.4f}")

    # Print intermediate IBP bounds at ReLU pre-activations
    print("\nIBP bounds at ReLU pre-activations:")
    relu_pre_bounds_ibp = {}
    for name, node in lirpa_model.named_modules():
        if 'Relu' in type(node).__name__:
            # Get the input node to ReLU
            if hasattr(node, 'inputs') and len(node.inputs) > 0:
                input_node = node.inputs[0]
                if hasattr(input_node, 'lower') and input_node.lower is not None:
                    lower = input_node.lower.detach().numpy()
                    upper = input_node.upper.detach().numpy()
                    width = (upper - lower).sum()
                    relu_pre_bounds_ibp[name] = width
                    print(f"  {name:40s} | pre-ReLU width: {width:.2f}")

    # Now compute CROWN bounds
    print("\n" + "-"*70)
    print("Step 2: CROWN Backward Pass")
    print("-"*70)

    # Reset model
    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
    print(f"Output width (CROWN): {(ub - lb).sum().item():.4f}")

    # Print intermediate CROWN bounds
    print("\nCROWN bounds at intermediate layers:")
    for name, node in lirpa_model.named_modules():
        if hasattr(node, 'lower') and node.lower is not None:
            lower = node.lower.detach().numpy()
            upper = node.upper.detach().numpy() if hasattr(node, 'upper') and node.upper is not None else lower
            width = (upper - lower).sum()
            op_type = type(node).__name__
            if width > 0:
                ibp_width = relu_pre_bounds_ibp.get(name, None)
                if ibp_width:
                    ratio = ibp_width / width
                    print(f"  {name:40s} | {op_type:15s} | width: {width:>10.2f} | IBP ratio: {ratio:.2f}x")
                else:
                    print(f"  {name:40s} | {op_type:15s} | width: {width:>10.2f}")

    # Check what's happening during backward
    print("\n" + "-"*70)
    print("Step 3: Analyzing CROWN Backward Pass")
    print("-"*70)

    # Reset and trace
    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    # Get model structure
    print("\nModel structure (nodes in order):")
    for name, node in lirpa_model.named_modules():
        op_type = type(node).__name__
        if 'Bound' in op_type:
            inputs = getattr(node, 'inputs', [])
            input_names = [getattr(inp, 'name', str(inp)) for inp in inputs]
            print(f"  {name:40s} | {op_type:15s} | inputs: {input_names[:3]}")

    # Check the order of backward propagation
    print("\n" + "-"*70)
    print("Step 4: Key Insight")
    print("-"*70)

    print("""
    auto_LiRPA CROWN backward pass:
    1. Starts from output layer with C = identity matrix
    2. Propagates backward through each layer
    3. For ReLU layers: uses pre-activation bounds to compute relaxation
    4. The pre-activation bounds come from PRIOR CROWN tightening, not IBP

    Key difference:
    - Pure CROWN (method='CROWN'): Uses CROWN-tightened bounds for intermediates
    - CROWN-IBP (method='CROWN-IBP'): Uses IBP bounds for intermediates

    auto_LiRPA's 'CROWN' method appears to use CROWN-tightened intermediate bounds,
    while γ-CROWN may be using IBP intermediate bounds.
    """)

    # Verify by computing with different intermediate bound settings
    print("\n" + "-"*70)
    print("Step 5: Verify Intermediate Bound Source")
    print("-"*70)

    # Check what auto_LiRPA does internally
    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    # Try with IBP=True to see if it's the same as CROWN-IBP
    lb1, ub1 = lirpa_model.compute_bounds(x=(bounded_x,), method='backward', IBP=True)
    w1 = (ub1 - lb1).sum().item()

    lirpa_model = BoundedModule(pytorch_model, x_center, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x_center, ptb)

    lb2, ub2 = lirpa_model.compute_bounds(x=(bounded_x,), method='backward', IBP=False)
    w2 = (ub2 - lb2).sum().item()

    print(f"backward + IBP=True (CROWN-IBP): {w1:.4f}")
    print(f"backward + IBP=False (pure CROWN): {w2:.4f}")

    print("\n" + "="*70)
    print("Conclusion")
    print("="*70)
    print(f"""
    The key difference is how intermediate bounds are computed:

    - auto_LiRPA CROWN (width={w2:.2f}):
      * Uses CROWN-tightened intermediate bounds
      * This requires O(N²) CROWN passes for N layers

    - auto_LiRPA CROWN-IBP (width={w1:.2f}):
      * Uses IBP intermediate bounds + CROWN for final output
      * This is O(N) but produces looser bounds

    γ-CROWN's current CROWN implementation produces width=75.46, which is:
    * Tighter than pure IBP (14881.76)
    * But looser than auto_LiRPA CROWN (21.89) by 3.45x

    Possible causes:
    1. γ-CROWN is using IBP intermediates instead of CROWN-tightened intermediates
    2. There's a bug in the CROWN-IBP tightening pass
    3. The intermediate bound tightening is not being applied correctly
    """)


if __name__ == '__main__':
    main()
