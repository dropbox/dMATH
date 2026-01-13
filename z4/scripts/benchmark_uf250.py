#!/usr/bin/env python3
"""
Benchmark Z4 vs CaDiCaL on uf250 instances.

This script provides consistent performance measurement for tracking Z4's
progress against CaDiCaL on 250-variable random 3-SAT instances.

Usage:
    python3 scripts/benchmark_uf250.py [--verbose] [--limit N]

Notes:
    - Z4 timing comes from `bench_dimacs` output.
    - CaDiCaL timing is parsed from its output (either "solve" or "total process time").
"""

import subprocess
import os
import glob
import tempfile
import sys
import argparse

# Configuration
Z4_BIN = os.path.join(os.path.dirname(__file__), "../target/release/examples/bench_dimacs")
CADICAL = os.path.join(os.path.dirname(__file__), "../reference/cadical/build/cadical")
BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "../benchmarks/satlib/uf250")


def get_z4_time_ms(filepath, metric):
    """Run Z4 and return timing in milliseconds (from bench_dimacs output)."""
    result = subprocess.run([Z4_BIN, filepath], capture_output=True, text=True)
    output = result.stdout + result.stderr
    for line in output.split('\n'):
        if line.strip().startswith(('SAT', 'UNSAT')):
            parts = line.split()
            if metric == 'solve':
                if len(parts) >= 2:
                    time_str = parts[1]
                    if time_str.endswith('ms'):
                        return float(time_str[:-2])
            elif metric == 'total':
                for i, token in enumerate(parts):
                    if token.startswith('total:'):
                        suffix = token[len('total:') :]
                        if suffix:
                            if suffix.endswith('ms'):
                                return float(suffix[:-2])
                        elif i + 1 < len(parts):
                            time_str = parts[i + 1]
                            if time_str.endswith('ms'):
                                return float(time_str[:-2])
            else:
                raise ValueError(f'unknown metric: {metric}')
    return None


def get_cadical_time_s(filepath, metric):
    """Run CaDiCaL and return timing in seconds."""
    result = subprocess.run([CADICAL, filepath], capture_output=True, text=True)
    output = result.stdout + result.stderr
    for line in output.split('\n'):
        stripped = line.strip()
        if metric == 'solve':
            # Example:
            #   c         0.12   99.69% solve
            if stripped.startswith('c') and stripped.endswith(' solve'):
                parts = stripped.split()
                if len(parts) >= 4 and parts[-1] == 'solve':
                    try:
                        return float(parts[1])
                    except ValueError:
                        pass
        elif metric == 'total':
            if 'total process time since initialization' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    words = parts[1].strip().split()
                    if words:
                        try:
                            return float(words[0])
                        except ValueError:
                            pass
        else:
            raise ValueError(f'unknown metric: {metric}')
    return None


def clean_satlib_file(src, dst):
    """Remove % terminator from SATLIB files for CaDiCaL compatibility."""
    with open(src, 'r') as f:
        with open(dst, 'w') as out:
            for line in f:
                if line.startswith('%'):
                    break
                out.write(line)


def main():
    parser = argparse.ArgumentParser(description='Benchmark Z4 vs CaDiCaL on uf250')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show per-file results')
    parser.add_argument('--limit', type=int, default=None, help='Only run the first N files')
    parser.add_argument(
        '--metric',
        choices=['solve', 'total'],
        default='solve',
        help='Compare solver-only time ("solve") or end-to-end time ("total")',
    )
    args = parser.parse_args()

    # Build Z4 first
    print("Building Z4...")
    build = subprocess.run(
        ['cargo', 'build', '--release', '--example', 'bench_dimacs', '-p', 'z4-sat'],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stdout)
        sys.stderr.write(build.stderr)
        print("ERROR: Failed to build Z4 bench binary")
        sys.exit(build.returncode)

    # Verify binaries exist
    if not os.path.exists(Z4_BIN):
        print(f"ERROR: Z4 binary not found at {Z4_BIN}")
        sys.exit(1)
    if not os.path.exists(CADICAL):
        print(f"ERROR: CaDiCaL not found at {CADICAL}")
        print("Run: cd reference/cadical && ./configure && make")
        sys.exit(1)

    # Get benchmark files
    files = sorted(glob.glob(os.path.join(BENCHMARK_DIR, "*.cnf")))
    if not files:
        print(f"ERROR: No benchmark files found in {BENCHMARK_DIR}")
        sys.exit(1)
    if args.limit is not None:
        files = files[: args.limit]

    print(f"Benchmarking on {len(files)} uf250 instances (metric={args.metric})...\n")

    if args.verbose:
        label = 'Z4 (ms)' if args.metric == 'solve' else 'Z4 total (ms)'
        cad_label = 'CaDiCaL (ms)' if args.metric == 'solve' else 'CaDiCaL total (ms)'
        print(f"{'File':<20} {label:>14} {cad_label:>16} {'Ratio':>8}")
        print("-" * 55)

    z4_total_ms = 0.0
    cadical_total_s = 0.0
    z4_wins = 0
    errors = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, f in enumerate(files):
            fname = os.path.basename(f)
            if not args.verbose:
                sys.stdout.write(f"\rProcessing {i+1}/{len(files)}...")
                sys.stdout.flush()

            # Get Z4 time
            z4_ms = get_z4_time_ms(f, args.metric)
            if z4_ms is None:
                if args.verbose:
                    print(f"{fname:<20} {'ERROR':>14} {'-':>16} {'-':>8}  (Z4 parse)")
                errors += 1
                continue

            # Clean file for CaDiCaL
            dst = os.path.join(tmpdir, fname)
            clean_satlib_file(f, dst)

            # Get CaDiCaL time
            cadical_s = get_cadical_time_s(dst, args.metric)
            if cadical_s is None:
                if args.verbose:
                    print(f"{fname:<20} {z4_ms:>14.1f} {'ERROR':>16} {'-':>8}  (CaDiCaL parse)")
                errors += 1
                continue

            z4_total_ms += z4_ms
            cadical_total_s += cadical_s
            cadical_ms = cadical_s * 1000

            if z4_ms / 1000 < cadical_s:
                z4_wins += 1

            if args.verbose:
                ratio = z4_ms / cadical_ms if cadical_ms > 0 else float('inf')
                print(f"{fname:<20} {z4_ms:>10.1f} {cadical_ms:>12.1f} {ratio:>8.2f}x")

    if not args.verbose:
        print("\n")

    # Summary
    z4_total_s = z4_total_ms / 1000.0
    count = len(files) - errors
    ratio = z4_total_s / cadical_total_s if cadical_total_s > 0 else float('inf')

    print("=" * 55)
    print("Results:")
    print(f"  Benchmarks:    {count} (errors: {errors})")
    print(f"  Z4 total:      {z4_total_s:.2f}s")
    print(f"  CaDiCaL total: {cadical_total_s:.2f}s")
    print(f"  Ratio:         {ratio:.2f}x (lower is better)")
    print(f"  Wins:          Z4={z4_wins}, CaDiCaL={count-z4_wins}")
    print("=" * 55)


if __name__ == '__main__':
    main()
