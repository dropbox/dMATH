#!/usr/bin/env python3
"""Benchmark Z4 vs Z3 on SMT-LIB files."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class Result:
    file: str
    z4_result: str
    z4_time: float
    z3_result: str
    z3_time: float
    agree: bool
    speedup: float  # z3_time / z4_time (>1 = Z4 faster)

def run_solver(solver_cmd: list, file_path: str, timeout: float = 30.0) -> tuple:
    """Run a solver and return (result, time)."""
    try:
        start = time.perf_counter()
        proc = subprocess.run(
            solver_cmd + [file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.perf_counter() - start
        output = proc.stdout.strip()

        # Parse result
        if "sat" in output.lower():
            # Check for unsat first (contains "sat")
            if "unsat" in output.lower():
                result = "unsat"
            else:
                result = "sat"
        elif "unknown" in output.lower():
            result = "unknown"
        else:
            result = "error"

        return result, elapsed
    except subprocess.TimeoutExpired:
        return "timeout", timeout
    except Exception as e:
        return f"error: {e}", 0.0

def benchmark_file(z4_cmd: list, z3_cmd: list, file_path: str) -> Result:
    """Benchmark a single file."""
    z4_result, z4_time = run_solver(z4_cmd, file_path)
    z3_result, z3_time = run_solver(z3_cmd, file_path)

    # Check agreement
    agree = z4_result == z3_result or z4_result == "unknown" or z3_result == "unknown"

    # Calculate speedup
    if z4_time > 0.001:
        speedup = z3_time / z4_time
    else:
        speedup = float('inf')

    return Result(
        file=os.path.basename(file_path),
        z4_result=z4_result,
        z4_time=z4_time,
        z3_result=z3_result,
        z3_time=z3_time,
        agree=agree,
        speedup=speedup
    )

def main():
    parser = argparse.ArgumentParser(description="Benchmark Z4 vs Z3")
    parser.add_argument("benchmark_dir", help="Directory containing .smt2 files")
    parser.add_argument("--z4", default="target/release/z4", help="Path to Z4 binary")
    parser.add_argument("--z3", default="z3", help="Path to Z3 binary")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout per file")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs")
    args = parser.parse_args()

    # Find all .smt2 files
    files = sorted(Path(args.benchmark_dir).rglob("*.smt2"))
    if not files:
        print(f"No .smt2 files found in {args.benchmark_dir}")
        sys.exit(1)

    print(f"Benchmarking {len(files)} files in {args.benchmark_dir}")
    print(f"Z4: {args.z4}")
    print(f"Z3: {args.z3}")
    print("-" * 80)

    z4_cmd = [args.z4]
    z3_cmd = [args.z3]

    results = []
    z4_total = 0.0
    z3_total = 0.0
    agreements = 0
    disagreements = []

    for f in files:
        result = benchmark_file(z4_cmd, z3_cmd, str(f))
        results.append(result)

        z4_total += result.z4_time
        z3_total += result.z3_time

        if result.agree:
            agreements += 1
        else:
            disagreements.append(result)

        # Print row
        status = "OK" if result.agree else "DISAGREE"
        speedup_str = f"{result.speedup:.2f}x" if result.speedup < 100 else ">>>"
        print(f"{result.file:40} Z4:{result.z4_result:8} {result.z4_time:7.3f}s  "
              f"Z3:{result.z3_result:8} {result.z3_time:7.3f}s  {speedup_str:>8} {status}")

    # Summary
    print("-" * 80)
    print(f"Total: Z4={z4_total:.3f}s, Z3={z3_total:.3f}s")
    if z4_total > 0.001:
        overall_speedup = z3_total / z4_total
        print(f"Overall ratio: {overall_speedup:.2f}x {'(Z4 faster)' if overall_speedup > 1 else '(Z3 faster)'}")
    print(f"Agreement: {agreements}/{len(results)} ({100*agreements/len(results):.1f}%)")

    if disagreements:
        print("\nDISAGREEMENTS:")
        for d in disagreements:
            print(f"  {d.file}: Z4={d.z4_result}, Z3={d.z3_result}")

    # Return non-zero if any disagreements
    sys.exit(0 if not disagreements else 1)

if __name__ == "__main__":
    main()
