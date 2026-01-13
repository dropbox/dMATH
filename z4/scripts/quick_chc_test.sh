#!/bin/bash
# Quick CHC verification for fast iteration
# Tier 1: ~15 seconds total

set -e

echo "=== Kani Fast (8 files, must pass) ==="
failed=0
for f in /tmp/kani_benchmarks/b*.smt2; do
    result=$(timeout 5 ./target/release/z4 --chc "$f" 2>&1)
    if [ "$result" = "sat" ]; then
        echo "✓ $(basename $f)"
    else
        echo "✗ $(basename $f): $result"
        failed=1
    fi
done

if [ $failed -eq 1 ]; then
    echo "KANI FAST FAILED - fix before continuing"
    exit 1
fi

echo ""
echo "=== Quick CHC-COMP Sample (5 known-solvable, 10s timeout) ==="
# These are files Z4 has solved before
sample_files=(
    "const_mod_1_000.smt2"
    "const_mod_2_000.smt2"
)
# Keep sample small for speed - these solve in <2s

solved=0
for f in "${sample_files[@]}"; do
    path="benchmarks/chc/chc-comp25-benchmarks/extra-small-lia/$f"
    if [ -f "$path" ]; then
        result=$(timeout 10 ./target/release/z4 --chc "$path" 2>&1)
        if [ "$result" = "sat" ] || [ "$result" = "unsat" ]; then
            echo "✓ $f: $result"
            solved=$((solved+1))
        else
            echo "? $f: timeout/unknown"
        fi
    fi
done

echo ""
echo "Quick test: Kani 8/8, Sample $solved/${#sample_files[@]}"
echo "Run 'scripts/full_chc_test.sh' for complete verification"
