#!/usr/bin/env python3
"""
Benchmark comparison: γ-CROWN vs Auto-LiRPA

Compares IBP verification time and memory usage on Whisper models.
"""

import time
import sys
import traceback

import torch
import numpy as np
from transformers import WhisperModel
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


class WhisperEncoderWrapper(torch.nn.Module):
    """Wrapper to return tensor instead of BaseModelOutput."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        output = self.encoder(x)
        return output.last_hidden_state


def benchmark_autolirpa(model_name: str, seq_len: int = 16, epsilon: float = 0.001) -> dict:
    """Run Auto-LiRPA IBP on Whisper model."""
    print(f"\n[Auto-LiRPA] Loading {model_name}...")

    # Load model
    model = WhisperModel.from_pretrained(model_name)
    encoder = WhisperEncoderWrapper(model.encoder)
    encoder.eval()

    # Count parameters (from original encoder)
    param_count = sum(p.numel() for p in model.encoder.parameters())
    print(f"  Encoder parameters: {param_count / 1e6:.1f}M")

    # Create input (mel spectrogram format: [batch, channels=80, time])
    # Whisper expects 80 mel channels and 3000 time steps for 30 seconds
    batch_size = 1
    mel_channels = 80
    time_steps = 3000  # Whisper requires exactly 3000

    x = torch.randn(batch_size, mel_channels, time_steps)

    # Create bounded model
    print(f"  Creating BoundedModule...")
    try:
        start_bound = time.perf_counter()
        bounded_model = BoundedModule(encoder, (x,), device='cpu')
        bound_creation_time = time.perf_counter() - start_bound
        print(f"  BoundedModule creation: {bound_creation_time:.2f}s")
    except Exception as e:
        return {"model": model_name, "params_m": param_count / 1e6, "error": f"BoundedModule creation failed: {e}"}

    # Define perturbation
    ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    bounded_input = BoundedTensor(x, ptb)

    # Run IBP
    print(f"  Running IBP verification...")
    try:
        start_ibp = time.perf_counter()
        lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method='ibp')
        ibp_time = time.perf_counter() - start_ibp

        return {
            "model": model_name,
            "params_m": param_count / 1e6,
            "bound_creation_s": bound_creation_time,
            "ibp_time_s": ibp_time,
            "total_time_s": bound_creation_time + ibp_time,
            "lb_sample": float(lb.flatten()[0]),
            "ub_sample": float(ub.flatten()[0]),
            "lb_min": float(lb.min()),
            "ub_max": float(ub.max()),
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "model": model_name,
            "params_m": param_count / 1e6,
            "error": str(e),
        }


def main():
    models = [
        ("openai/whisper-tiny", 39),
        ("openai/whisper-small", 244),
        # ("openai/whisper-medium", 769),  # Too slow for Auto-LiRPA
    ]

    print("=" * 60)
    print("Auto-LiRPA IBP Benchmark on Whisper Models")
    print("=" * 60)

    results = []
    for model_name, expected_params in models:
        result = benchmark_autolirpa(model_name)
        results.append(result)

        if "error" in result:
            print(f"\n  ERROR: {result['error']}")
        else:
            print(f"\n  Results for {model_name}:")
            print(f"    Params: {result['params_m']:.1f}M")
            print(f"    BoundedModule creation: {result['bound_creation_s']:.2f}s")
            print(f"    IBP computation: {result['ibp_time_s']:.2f}s")
            print(f"    Total time: {result['total_time_s']:.2f}s")
            print(f"    Bounds: [{result['lb_min']:.4f}, {result['ub_max']:.4f}]")

    print("\n" + "=" * 60)
    print("Summary (IBP computation time only)")
    print("=" * 60)
    print(f"{'Model':<25} {'Params':<10} {'Auto-LiRPA':<15}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<25} {r.get('params_m', '?'):<10.1f}M ERROR")
        else:
            print(f"{r['model']:<25} {r['params_m']:<10.1f}M {r['ibp_time_s']:<15.2f}s")


if __name__ == "__main__":
    main()
