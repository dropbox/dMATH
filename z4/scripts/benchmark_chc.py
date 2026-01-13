#!/usr/bin/env python3
"""Benchmark Z4 CHC solver vs Z3 Spacer on CHC-COMP benchmarks."""

import argparse
import subprocess
import sys
import time
import os
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class Result:
    file: str
    z4_result: str
    z4_time: float
    z3_result: str
    z3_time: float
    agree: bool
    speedup: float  # z3_time / z4_time (>1 = Z4 faster)
    expected: str   # from yml file

@dataclass
class TrackStats:
    track: str
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
    disagreements: int = 0
    z4_correct: int = 0  # matches expected
    z4_wrong: int = 0    # contradicts expected
    z3_correct: int = 0
    z3_wrong: int = 0
    z4_wins: int = 0     # z4 faster by >10%
    z3_wins: int = 0     # z3 faster by >10%
    z4_total_time: float = 0.0
    z3_total_time: float = 0.0
    results: List[Result] = field(default_factory=list)


def get_expected_status(smt2_path: str) -> str:
    """Parse expected status from yml file."""
    if not HAS_YAML:
        return ''
    yml_path = smt2_path.replace('.smt2', '.yml')
    if not os.path.exists(yml_path):
        return ''
    try:
        with open(yml_path, 'r') as f:
            data = yaml.safe_load(f)
        props = data.get('properties', [])
        if props:
            verdict = props[0].get('expected_verdict')
            # true = sat (safe), false = unsat (unsafe)
            if verdict is True:
                return 'sat'
            elif verdict is False:
                return 'unsat'
    except Exception:
        pass
    return ''


def run_solver(solver_cmd: list, file_path: str, timeout: float = 30.0) -> Tuple[str, float]:
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
        stderr = proc.stderr.strip().lower()

        # Parse result - CHC solvers output sat/unsat/unknown
        if 'unsat' in output:
            result = 'unsat'
        elif 'sat' in output:
            result = 'sat'
        elif 'unknown' in output:
            result = 'unknown'
        elif 'error' in output or 'error' in stderr:
            result = 'error'
        else:
            result = 'unknown'

        return result, elapsed
    except subprocess.TimeoutExpired:
        return 'timeout', timeout
    except Exception as e:
        return 'error', 0.0


def benchmark_file(z4_cmd: list, z3_cmd: list, file_path: str, timeout: float) -> Result:
    """Benchmark a single file."""
    z4_result, z4_time = run_solver(z4_cmd, file_path, timeout)
    z3_result, z3_time = run_solver(z3_cmd, file_path, timeout)
    expected = get_expected_status(file_path)

    # Check agreement (ignore unknown/timeout/error)
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
        speedup = float('inf') if z3_time > 0.001 else 1.0

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


def run_benchmarks(track: str, files: List[str], z4_cmd: list, z3_cmd: list, timeout: float, verbose: bool = False) -> TrackStats:
    """Run benchmarks for a track."""
    stats = TrackStats(track=track)
    stats.total = len(files)

    for i, f in enumerate(files):
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

        # Agreement/correctness
        if result.agree:
            stats.agreements += 1
        else:
            stats.disagreements += 1

        # Check against expected
        if result.expected:
            if result.z4_result == result.expected:
                stats.z4_correct += 1
            elif result.z4_result in ('sat', 'unsat') and result.z4_result != result.expected:
                stats.z4_wrong += 1
            if result.z3_result == result.expected:
                stats.z3_correct += 1
            elif result.z3_result in ('sat', 'unsat') and result.z3_result != result.expected:
                stats.z3_wrong += 1

        # Win/loss (>10% difference, only if both solved)
        if result.z4_result in ('sat', 'unsat') and result.z3_result in ('sat', 'unsat'):
            if result.speedup > 1.1:
                stats.z4_wins += 1
            elif result.speedup < 0.9:
                stats.z3_wins += 1

        # Progress indicator
        if verbose or stats.total <= 50:
            status = "OK" if result.agree else "DISAGREE"
            z4_correct_str = "!" if result.z4_result in ('sat', 'unsat') and result.expected and result.z4_result != result.expected else ""
            speedup_str = f"{result.speedup:.2f}x" if result.speedup < 100 else ">>>"
            print(f"  [{i+1:3}/{stats.total}] {result.file[:35]:35} Z4:{result.z4_result:8}{z4_correct_str:1} {result.z4_time:6.3f}s  "
                  f"Z3:{result.z3_result:8} {result.z3_time:6.3f}s  {speedup_str:>8} {status}")
        else:
            # Compact progress
            if (i + 1) % 10 == 0 or i == stats.total - 1:
                print(f"  Progress: {i+1}/{stats.total}", end='\r')

    if not verbose and stats.total > 50:
        print()  # newline after progress

    return stats


def find_benchmarks(base_dir: str, track: str, limit: Optional[int] = None) -> List[str]:
    """Find benchmark files for a track."""
    files = []

    # Track directories vary - find by pattern
    track_dirs = {
        'LIA-Lin': ['extra-small-lia'],
        'LIA': ['extra-small-lia', 'aeval-benchmarks'],
        'small': ['extra-small-lia'],
    }

    search_dirs = track_dirs.get(track, [track])

    for dir_name in search_dirs:
        track_path = Path(base_dir) / dir_name
        if track_path.exists():
            for smt2 in sorted(track_path.glob('**/*.smt2')):
                files.append(str(smt2))
                if limit and len(files) >= limit:
                    return files

    return files


def main():
    parser = argparse.ArgumentParser(description="Benchmark Z4 CHC vs Z3 Spacer")
    parser.add_argument("--z4", default="target/release/z4", help="Path to Z4 binary")
    parser.add_argument("--z3", default="z3", help="Path to Z3 binary")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout per file (seconds)")
    parser.add_argument("--benchmark-dir", default="benchmarks/chc/chc-comp25-benchmarks",
                        help="Directory with CHC-COMP benchmarks")
    parser.add_argument("--track", default="small", help="Track to benchmark (small, LIA-Lin, LIA)")
    parser.add_argument("--limit", type=int, default=50, help="Max files per track (0 = all)")
    parser.add_argument("--verbose", action="store_true", help="Show all individual results")
    parser.add_argument("--files", nargs="*", help="Specific files to benchmark")
    args = parser.parse_args()

    z4_cmd = [args.z4, "--chc"]
    z3_cmd = [args.z3]

    # Check binaries exist
    if not os.path.exists(args.z4):
        print(f"Error: Z4 binary not found at {args.z4}")
        print("Run: cargo build --release")
        sys.exit(1)

    print(f"Z4: {args.z4} --chc")
    print(f"Z3: {args.z3}")
    print(f"Timeout: {args.timeout}s")
    print("=" * 100)

    # Find files
    if args.files:
        files = args.files
        track_name = "custom"
    else:
        files = find_benchmarks(args.benchmark_dir, args.track, args.limit if args.limit > 0 else None)
        track_name = args.track

    if not files:
        print(f"No benchmark files found for track '{args.track}' in {args.benchmark_dir}")
        sys.exit(1)

    print(f"\n### {track_name} ({len(files)} benchmarks) ###")
    stats = run_benchmarks(track_name, files, z4_cmd, z3_cmd, args.timeout, args.verbose)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Results breakdown
    print(f"\nZ4: {stats.z4_sat} sat, {stats.z4_unsat} unsat, {stats.z4_unknown} unknown, "
          f"{stats.z4_timeout} timeout, {stats.z4_error} error")
    print(f"Z3: {stats.z3_sat} sat, {stats.z3_unsat} unsat, {stats.z3_unknown} unknown, "
          f"{stats.z3_timeout} timeout, {stats.z3_error} error")

    # Correctness (vs expected)
    if stats.z4_correct > 0 or stats.z4_wrong > 0:
        print(f"\nZ4 vs expected: {stats.z4_correct} correct, {stats.z4_wrong} WRONG")
    if stats.z3_correct > 0 or stats.z3_wrong > 0:
        print(f"Z3 vs expected: {stats.z3_correct} correct, {stats.z3_wrong} WRONG")

    # Agreement
    print(f"\nAgreement: {stats.agreements}/{stats.total}")
    if stats.disagreements > 0:
        print(f"DISAGREEMENTS: {stats.disagreements}")

    # Performance
    z4_solved = stats.z4_sat + stats.z4_unsat
    z3_solved = stats.z3_sat + stats.z3_unsat
    print(f"\nSolved: Z4={z4_solved}, Z3={z3_solved}")
    print(f"Total time: Z4={stats.z4_total_time:.2f}s, Z3={stats.z3_total_time:.2f}s")

    if stats.z4_total_time > 0.001:
        ratio = stats.z3_total_time / stats.z4_total_time
        if ratio > 1:
            print(f"Z4 is {ratio:.2f}x faster overall")
        else:
            print(f"Z3 is {1/ratio:.2f}x faster overall")

    print(f"Wins: Z4={stats.z4_wins}, Z3={stats.z3_wins}")

    # Disagreement details
    if stats.disagreements > 0:
        print(f"\n### DISAGREEMENTS ({stats.disagreements}) ###")
        for r in stats.results:
            if not r.agree:
                print(f"  {r.file}: Z4={r.z4_result}, Z3={r.z3_result}, expected={r.expected}")

    # Wrong answers
    wrong_z4 = [r for r in stats.results if r.expected and r.z4_result in ('sat', 'unsat') and r.z4_result != r.expected]
    if wrong_z4:
        print(f"\n### Z4 WRONG ANSWERS ({len(wrong_z4)}) ###")
        for r in wrong_z4:
            print(f"  {r.file}: Z4={r.z4_result}, expected={r.expected}")

    sys.exit(1 if stats.disagreements > 0 or stats.z4_wrong > 0 else 0)


if __name__ == "__main__":
    main()
