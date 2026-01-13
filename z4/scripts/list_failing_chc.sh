#!/bin/bash
# List failing CHC benchmarks

set -e

TIMEOUT=${1:-30}
JOBS=${2:-8}

bench_dir="benchmarks/chc/chc-comp25-benchmarks/extra-small-lia"

if [ ! -d "$bench_dir" ]; then
    echo "Missing benchmarks directory: $bench_dir" >&2
    exit 1
fi

echo "=== Failing CHC-COMP extra-small-lia (${TIMEOUT}s timeout) ==="

if command -v parallel &> /dev/null; then
    ls "$bench_dir"/*.smt2 | parallel -j"$JOBS" --line-buffer \
        "f={}; r=\$(timeout $TIMEOUT ./target/release/z4 --chc \"\$f\" 2>&1 | head -1); if [ \"\$r\" != sat ] && [ \"\$r\" != unsat ]; then echo \"\$(basename \"\$f\"): \$r\"; fi" | \
        sort
else
    for f in "$bench_dir"/*.smt2; do
        r=$(timeout "$TIMEOUT" ./target/release/z4 --chc "$f" 2>&1 | head -1)
        if [ "$r" != "sat" ] && [ "$r" != "unsat" ]; then
            echo "$(basename "$f"): $r"
        fi
    done | sort
fi
