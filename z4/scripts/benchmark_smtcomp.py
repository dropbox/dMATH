#!/usr/bin/env python3
"""Benchmark Z4 vs Z3 on SMT-COMP benchmarks."""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import os

@dataclass
class Result:
    file: str
    z4_result: str
    z4_time: float
    z3_result: str
    z3_time: float
    agree: bool
    speedup: float  # z3_time / z4_time (>1 = Z4 faster)
    expected: str = ""  # from status annotation

@dataclass
class TheoryStats:
    theory: str
    total: int = 0
    z4_sat: int = 0
    z4_unsat: int = 0
    z4_unknown: int = 0
    z4_timeout: int = 0
    z4_error: int = 0
    z3_sat: int = 0
    z3_unsat: int = 0
    z3_unknown: int = 0
    z3_timeout: int = 0
    z3_error: int = 0
    agreements: int = 0
    z4_wins: int = 0  # z4 faster by >10%
    z3_wins: int = 0  # z3 faster by >10%
    z4_total_time: float = 0.0
    z3_total_time: float = 0.0
    results: List[Result] = field(default_factory=list)

def get_expected_status(file_path: str) -> str:
    """Parse expected status from smt2 file."""
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if ':status' in line:
                    if 'sat' in line and 'unsat' not in line:
                        return 'sat'
                    elif 'unsat' in line:
                        return 'unsat'
                    elif 'unknown' in line:
                        return 'unknown'
    except:
        pass
    return ''

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
        output = proc.stdout.strip().lower()

        # Parse result
        if 'unsat' in output:
            result = 'unsat'
        elif 'sat' in output:
            result = 'sat'
        elif 'unknown' in output:
            result = 'unknown'
        else:
            result = 'error'

        return result, elapsed
    except subprocess.TimeoutExpired:
        return 'timeout', timeout
    except Exception as e:
        return f'error', 0.0

def benchmark_file(z4_cmd: list, z3_cmd: list, file_path: str, timeout: float) -> Result:
    """Benchmark a single file."""
    z4_result, z4_time = run_solver(z4_cmd, file_path, timeout)
    z3_result, z3_time = run_solver(z3_cmd, file_path, timeout)
    expected = get_expected_status(file_path)

    # Check agreement (ignore unknown/timeout)
    definite_z4 = z4_result in ('sat', 'unsat')
    definite_z3 = z3_result in ('sat', 'unsat')

    if definite_z4 and definite_z3:
        agree = z4_result == z3_result
    else:
        agree = True  # Don't count as disagreement if one is unknown/timeout

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
        speedup=speedup,
        expected=expected
    )

def run_benchmarks(theory: str, files: List[str], z4_cmd: list, z3_cmd: list, timeout: float) -> TheoryStats:
    """Run benchmarks for a theory."""
    stats = TheoryStats(theory=theory)
    stats.total = len(files)

    for f in files:
        result = benchmark_file(z4_cmd, z3_cmd, f, timeout)
        stats.results.append(result)

        # Update Z4 stats
        stats.z4_total_time += result.z4_time
        if result.z4_result == 'sat': stats.z4_sat += 1
        elif result.z4_result == 'unsat': stats.z4_unsat += 1
        elif result.z4_result == 'unknown': stats.z4_unknown += 1
        elif result.z4_result == 'timeout': stats.z4_timeout += 1
        else: stats.z4_error += 1

        # Update Z3 stats
        stats.z3_total_time += result.z3_time
        if result.z3_result == 'sat': stats.z3_sat += 1
        elif result.z3_result == 'unsat': stats.z3_unsat += 1
        elif result.z3_result == 'unknown': stats.z3_unknown += 1
        elif result.z3_result == 'timeout': stats.z3_timeout += 1
        else: stats.z3_error += 1

        if result.agree:
            stats.agreements += 1

        # Win/loss (>10% difference)
        if result.speedup > 1.1:
            stats.z4_wins += 1
        elif result.speedup < 0.9:
            stats.z3_wins += 1

        # Progress indicator
        status = "OK" if result.agree else "DISAGREE"
        speedup_str = f"{result.speedup:.2f}x" if result.speedup < 100 else ">>>"
        print(f"  {result.file[:40]:40} Z4:{result.z4_result:8} {result.z4_time:6.3f}s  "
              f"Z3:{result.z3_result:8} {result.z3_time:6.3f}s  {speedup_str:>8} {status}")

    return stats

def main():
    parser = argparse.ArgumentParser(description="Benchmark Z4 vs Z3 on SMT-COMP")
    parser.add_argument("--z4", default="target/release/z4", help="Path to Z4 binary")
    parser.add_argument("--z3", default="z3", help="Path to Z3 binary")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout per file")
    parser.add_argument("--sample-dir", default="benchmarks/samples", help="Directory with sample files")
    args = parser.parse_args()

    z4_cmd = [args.z4]
    z3_cmd = [args.z3]

    # Find sample files
    sample_dir = Path(args.sample_dir)
    theories = []

    for sample_file in sorted(sample_dir.glob("*.txt")):
        theory_name = sample_file.stem.upper().replace("_ALL", "").replace("_100", "")
        files = [line.strip() for line in sample_file.read_text().splitlines() if line.strip()]
        if files:
            theories.append((theory_name, files))

    if not theories:
        print(f"No sample files found in {sample_dir}")
        sys.exit(1)

    print(f"Z4: {args.z4}")
    print(f"Z3: {args.z3}")
    print(f"Timeout: {args.timeout}s")
    print("=" * 100)

    all_stats = []
    total_disagreements = []

    for theory_name, files in theories:
        print(f"\n### {theory_name} ({len(files)} benchmarks) ###")
        stats = run_benchmarks(theory_name, files, z4_cmd, z3_cmd, args.timeout)
        all_stats.append(stats)

        # Collect disagreements
        for r in stats.results:
            if not r.agree:
                total_disagreements.append((theory_name, r))

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Theory':<10} {'Total':>6} {'Z4time':>8} {'Z3time':>8} {'Ratio':>8} "
          f"{'Agree':>8} {'Z4wins':>7} {'Z3wins':>7} {'Z4err':>6} {'Z3err':>6}")
    print("-" * 100)

    total_z4_time = 0
    total_z3_time = 0
    total_files = 0
    total_agree = 0
    total_z4_wins = 0
    total_z3_wins = 0

    for stats in all_stats:
        ratio = stats.z3_total_time / stats.z4_total_time if stats.z4_total_time > 0.001 else 0
        agree_pct = 100 * stats.agreements / stats.total if stats.total > 0 else 0
        z4_err = stats.z4_error + stats.z4_timeout
        z3_err = stats.z3_error + stats.z3_timeout

        print(f"{stats.theory:<10} {stats.total:>6} {stats.z4_total_time:>7.2f}s {stats.z3_total_time:>7.2f}s "
              f"{ratio:>7.2f}x {agree_pct:>7.1f}% {stats.z4_wins:>7} {stats.z3_wins:>7} "
              f"{z4_err:>6} {z3_err:>6}")

        total_z4_time += stats.z4_total_time
        total_z3_time += stats.z3_total_time
        total_files += stats.total
        total_agree += stats.agreements
        total_z4_wins += stats.z4_wins
        total_z3_wins += stats.z3_wins

    print("-" * 100)
    overall_ratio = total_z3_time / total_z4_time if total_z4_time > 0.001 else 0
    overall_agree = 100 * total_agree / total_files if total_files > 0 else 0
    print(f"{'TOTAL':<10} {total_files:>6} {total_z4_time:>7.2f}s {total_z3_time:>7.2f}s "
          f"{overall_ratio:>7.2f}x {overall_agree:>7.1f}% {total_z4_wins:>7} {total_z3_wins:>7}")

    if overall_ratio > 1:
        print(f"\nZ4 is {overall_ratio:.2f}x FASTER than Z3 overall")
    else:
        print(f"\nZ3 is {1/overall_ratio:.2f}x faster than Z4 overall")

    # Disagreements detail
    if total_disagreements:
        print(f"\n### DISAGREEMENTS ({len(total_disagreements)}) ###")
        for theory, r in total_disagreements:
            print(f"  [{theory}] {r.file}: Z4={r.z4_result}, Z3={r.z3_result}, expected={r.expected}")
        sys.exit(1)
    else:
        print("\nNo disagreements detected.")

    sys.exit(0)

if __name__ == "__main__":
    main()
