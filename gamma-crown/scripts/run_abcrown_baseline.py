#!/usr/bin/env python3
"""
Run alpha-beta-CROWN baseline on cifar10_resnet benchmark instances.
This script runs the official alpha-beta-CROWN verifier on the first N instances
of the cifar10_resnet benchmark to establish baseline verification rates.
"""

import argparse
import subprocess
import os
import sys
import time
import json
from pathlib import Path

# Configuration
ABCROWN_DIR = Path("/Users/ayates/gamma-crown/research/repos/alpha-beta-CROWN")
VENV_PYTHON = ABCROWN_DIR / "venv312/bin/python"
BENCHMARK_DIR = Path("/Users/ayates/gamma-crown/benchmarks/vnncomp2021/benchmarks/cifar10_resnet")
ONNX_2B = BENCHMARK_DIR / "onnx/resnet_2b.onnx"
VNNLIB_DIR = BENCHMARK_DIR / "vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered"


def run_instance(prop_idx: int, timeout: int = 60) -> dict:
    """Run a single verification instance."""
    vnnlib_path = VNNLIB_DIR / f"prop_{prop_idx}_eps_0.008.vnnlib"
    results_file = f"/tmp/abcrown_result_{prop_idx}.txt"

    if not vnnlib_path.exists():
        return {"idx": prop_idx, "status": "error", "message": f"VNNLib file not found: {vnnlib_path}"}

    cmd = [
        str(VENV_PYTHON),
        str(ABCROWN_DIR / "complete_verifier/abcrown.py"),
        "--config", str(ABCROWN_DIR / "complete_verifier/exp_configs/vnncomp21/cifar10-resnet.yaml"),
        "--onnx_path", str(ONNX_2B),
        "--vnnlib_path", str(vnnlib_path),
        "--timeout", str(timeout),
        "--device", "cpu",
        "--results_file", results_file,
    ]

    start_time = time.time()
    try:
        # Use subprocess with a timeout slightly longer than the verification timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60,  # Extra buffer for setup/teardown
            cwd=str(ABCROWN_DIR / "complete_verifier"),
        )
        elapsed = time.time() - start_time

        # Parse result from file
        status = "unknown"
        if os.path.exists(results_file):
            with open(results_file) as f:
                content = f.read().strip().lower()
                # Check unsat FIRST (unsat contains "sat" as substring)
                if content == "unsat" or "holds" in content:
                    status = "unsat"
                elif content == "sat" or "violated" in content:
                    status = "sat"
                elif "timeout" in content:
                    status = "timeout"
                else:
                    status = content[:50] if content else "empty"

        # Also check stdout/stderr for result
        stdout_lower = result.stdout.lower()
        if status == "unknown":
            if "result: sat" in stdout_lower or "violated" in stdout_lower:
                status = "sat"
            elif "result: unsat" in stdout_lower or "verified" in stdout_lower:
                status = "unsat"
            elif "timeout" in stdout_lower:
                status = "timeout"

        return {
            "idx": prop_idx,
            "status": status,
            "time": elapsed,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "idx": prop_idx,
            "status": "timeout",
            "time": elapsed,
        }
    except Exception as e:
        return {
            "idx": prop_idx,
            "status": "error",
            "message": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run alpha-beta-CROWN baseline benchmark")
    parser.add_argument("--max-instances", type=int, default=20, help="Max instances to run")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per instance in seconds")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting property index")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    args = parser.parse_args()

    print(f"Running alpha-beta-CROWN baseline on cifar10_resnet (2b)")
    print(f"Instances: {args.start_idx} to {args.start_idx + args.max_instances - 1}")
    print(f"Timeout per instance: {args.timeout}s")
    print()

    results = []
    verified = 0
    falsified = 0
    timeouts = 0

    total_start = time.time()

    for i in range(args.start_idx, args.start_idx + args.max_instances):
        print(f"[{i-args.start_idx+1}/{args.max_instances}] Running prop_{i}... ", end="", flush=True)
        result = run_instance(i, args.timeout)
        results.append(result)

        if result["status"] == "unsat":
            verified += 1
            print(f"✓ verified ({result.get('time', 0):.1f}s)")
        elif result["status"] == "sat":
            falsified += 1
            print(f"✗ falsified ({result.get('time', 0):.1f}s)")
        elif result["status"] == "timeout":
            timeouts += 1
            print(f"⏱ timeout ({result.get('time', 0):.1f}s)")
        else:
            print(f"? {result['status']} ({result.get('time', 0):.1f}s)")

    total_elapsed = time.time() - total_start

    print()
    print("=" * 60)
    print(f"Summary:")
    print(f"  Verified:  {verified}/{args.max_instances} ({100*verified/args.max_instances:.1f}%)")
    print(f"  Falsified: {falsified}/{args.max_instances}")
    print(f"  Timeout:   {timeouts}/{args.max_instances}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print("=" * 60)

    # Save results
    output_data = {
        "config": {
            "model": "resnet_2b",
            "timeout": args.timeout,
            "instances": args.max_instances,
            "start_idx": args.start_idx,
        },
        "summary": {
            "verified": verified,
            "falsified": falsified,
            "timeout": timeouts,
            "total_time": total_elapsed,
        },
        "results": results,
    }

    output_file = args.output or f"reports/main/abcrown_baseline_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path = Path("/Users/ayates/gamma-crown") / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
