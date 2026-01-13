#!/usr/bin/env python3
"""Benchmark CPU vs wgpu vs MLX backends on Whisper sequential verification.

This uses `gamma whisper-seq` and parses:
- `total_time_ms`
- `final_output_width`
- per-block widths and `gpu` column

It runs multiple repeats per configuration and reports median timings.
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass

DEFAULT_MODEL = "tests/models/whisper_tiny_encoder.onnx"
DEFAULT_GAMMA = "./target/release/gamma"


@dataclass(frozen=True)
class WhisperSeqResult:
    time_ms: int
    final_output_width: float
    gpu_enabled: bool
    gpu_used_any: bool
    fell_back_to_cpu: bool


_RE_TIME_MS = re.compile(r"total_time_ms=(\d+)")
_RE_FINAL_WIDTH = re.compile(r"final_output_width=([0-9eE+.\-]+)")
_RE_BACKEND = re.compile(r"^Backend:\s+(\S+)\s+\(GPU:\s+(enabled|disabled)\)", re.MULTILINE)
_RE_PER_BLOCK_ROW = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(yes|no)\s+(\d+)\s*$",
    re.MULTILINE,
)
_RE_MLX_FALLBACK = re.compile(r"MLX backend not available: .*Using CPU\.", re.MULTILINE)


def _run_whisper_seq(
    gamma: str,
    model: str,
    backend: str,
    seq_len: int,
    blocks: int,
    epsilon: float,
    timeout_s: int,
) -> WhisperSeqResult:
    cmd = [
        gamma,
        "whisper-seq",
        model,
        "--start-block",
        "0",
        "--end-block",
        str(blocks),
        "--epsilon",
        str(epsilon),
        "--seq-len",
        str(seq_len),
        "--backend",
        backend,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    output = completed.stdout + completed.stderr

    if completed.returncode != 0:
        raise RuntimeError(f"gamma whisper-seq failed (backend={backend}):\n{output}")

    match_time = _RE_TIME_MS.search(output)
    if match_time is None:
        raise RuntimeError(f"Failed to parse total_time_ms (backend={backend}):\n{output}")
    time_ms = int(match_time.group(1))

    match_width = _RE_FINAL_WIDTH.search(output)
    if match_width is None:
        raise RuntimeError(f"Failed to parse final_output_width (backend={backend}):\n{output}")
    final_output_width = float(match_width.group(1))

    match_backend = _RE_BACKEND.search(output)
    gpu_enabled = False
    if match_backend is not None:
        gpu_enabled = match_backend.group(2) == "enabled"

    gpu_used_any = any(m.group(6) == "yes" for m in _RE_PER_BLOCK_ROW.finditer(output))
    fell_back_to_cpu = _RE_MLX_FALLBACK.search(output) is not None

    return WhisperSeqResult(
        time_ms=time_ms,
        final_output_width=final_output_width,
        gpu_enabled=gpu_enabled,
        gpu_used_any=gpu_used_any,
        fell_back_to_cpu=fell_back_to_cpu,
    )


def _median_ms(samples: list[int]) -> int:
    # Use the lower-median to avoid averaging in the even-N case.
    # For small-N (1/2/3) this is typically more robust to a single outlier.
    return int(statistics.median_low(samples))


def _rel_diff(a: float, b: float) -> float:
    denom = max(1.0, abs(a), abs(b))
    return abs(a - b) / denom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", default=DEFAULT_GAMMA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller config set (faster, less coverage).",
    )
    args = parser.parse_args()

    if args.quick:
        configs: list[tuple[int, int]] = [
            (64, 2),
            (128, 2),
            (256, 2),
            (128, 4),
        ]
    else:
        configs = [
            (64, 2),
            (64, 4),
            (128, 2),
            (128, 4),
            (256, 2),
            (256, 4),
        ]

    print("=" * 70)
    print("CPU vs wgpu vs MLX Backend Benchmark (whisper-seq)")
    print("=" * 70)
    print()
    print(f"Binary: {args.gamma}")
    print(f"Model:  {args.model}")
    print(f"Config: repeats={args.repeats}, epsilon={args.epsilon}")
    print()
    print("| seq_len | blocks | CPU med (ms) | wgpu med (ms) | MLX med (ms) | wgpu speedup | MLX speedup |")
    print("|---------|--------|--------------|---------------|--------------|--------------|-------------|")

    correctness_failures: list[str] = []
    speedups_wgpu: list[float] = []
    speedups_mlx: list[float] = []
    mlx_unavailable_seen = False

    for seq_len, blocks in configs:
        per_backend_times: dict[str, list[int]] = {"cpu": [], "wgpu": [], "mlx": []}
        per_backend_widths: dict[str, list[float]] = {"cpu": [], "wgpu": [], "mlx": []}
        per_backend_gpu_used_any: dict[str, bool] = {"cpu": False, "wgpu": False, "mlx": False}
        mlx_available = True

        for backend in ("cpu", "wgpu", "mlx"):
            warmup = _run_whisper_seq(
                args.gamma, args.model, backend, seq_len, blocks, args.epsilon, args.timeout_s
            )
            if backend == "mlx" and warmup.fell_back_to_cpu:
                mlx_available = False
                mlx_unavailable_seen = True
                break

            for _ in range(args.repeats):
                r = _run_whisper_seq(
                    args.gamma,
                    args.model,
                    backend,
                    seq_len,
                    blocks,
                    args.epsilon,
                    args.timeout_s,
                )
                per_backend_times[backend].append(r.time_ms)
                per_backend_widths[backend].append(r.final_output_width)
                per_backend_gpu_used_any[backend] = per_backend_gpu_used_any[backend] or r.gpu_used_any

        cpu_ms = _median_ms(per_backend_times["cpu"])
        wgpu_ms = _median_ms(per_backend_times["wgpu"])
        mlx_ms = _median_ms(per_backend_times["mlx"]) if mlx_available else 0

        wgpu_speedup = (cpu_ms / wgpu_ms) if wgpu_ms > 0 else 0.0
        mlx_speedup = (cpu_ms / mlx_ms) if mlx_ms > 0 else 0.0

        speedups_wgpu.append(wgpu_speedup)
        if mlx_available:
            speedups_mlx.append(mlx_speedup)

        mlx_ms_cell = f"{mlx_ms:12d}" if mlx_available else f"{'N/A':>12s}"
        mlx_speedup_cell = f"{mlx_speedup:10.2f}x" if mlx_available else f"{'N/A':>10s}"
        print(
            f"| {seq_len:7d} | {blocks:6d} | {cpu_ms:12d} | {wgpu_ms:13d} | {mlx_ms_cell} | {wgpu_speedup:11.2f}x | {mlx_speedup_cell} |"
        )

        cpu_width = statistics.median_low(per_backend_widths["cpu"])
        for backend in ("wgpu", "mlx"):
            if backend == "mlx" and not mlx_available:
                continue
            backend_width = statistics.median_low(per_backend_widths[backend])
            diff = _rel_diff(cpu_width, backend_width)
            if diff > args.tolerance:
                correctness_failures.append(
                    f"seq_len={seq_len} blocks={blocks} backend={backend} rel_diff={diff:.3e} cpu={cpu_width:.6e} backend={backend_width:.6e}"
                )

        if seq_len >= 64 and not per_backend_gpu_used_any["wgpu"]:
            correctness_failures.append(
                f"seq_len={seq_len} blocks={blocks} backend=wgpu did not report any GPU usage"
            )

    print()
    print("Summary:")
    print(f"  Average wgpu speedup (median-of-medians): {statistics.mean(speedups_wgpu):.2f}x")
    if speedups_mlx:
        print(f"  Average MLX speedup (median-of-medians):  {statistics.mean(speedups_mlx):.2f}x")
    else:
        print("  Average MLX speedup (median-of-medians):  N/A (mlx unavailable)")
    print()
    if mlx_unavailable_seen:
        print("Note: mlx backend unavailable; `gamma` fell back to CPU for mlx runs.")
        print("      Rebuild with `cargo build --release --features mlx` to enable MLX.")
        print()

    if correctness_failures:
        print("Correctness check: FAILED")
        for line in correctness_failures:
            print(f"  - {line}")
        return 1

    print(f"Correctness check: OK (tolerance={args.tolerance:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
