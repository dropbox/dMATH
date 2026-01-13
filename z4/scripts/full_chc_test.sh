#!/bin/bash
# Full CHC-COMP verification (parallel execution)
# Takes ~3-5 min with parallelization

set -e

TIMEOUT=${1:-30}
JOBS=${2:-8}

echo "=== Full CHC-COMP extra-small-lia (${TIMEOUT}s timeout, $JOBS parallel) ==="

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel..."
    results=$(ls benchmarks/chc/chc-comp25-benchmarks/extra-small-lia/*.smt2 | \
        parallel -j$JOBS "timeout $TIMEOUT ./target/release/z4 --chc {} 2>&1 | head -1" | \
        sort | uniq -c)
    
    sat=$(echo "$results" | grep -w sat | awk '{print $1}' || echo 0)
    unsat=$(echo "$results" | grep -w unsat | awk '{print $1}' || echo 0)
    total=$((sat + unsat))
    echo "Solved: $total/55 (sat=$sat, unsat=$unsat)"
else
    echo "GNU parallel not found, running sequentially..."
    sat=0; unsat=0
    for f in benchmarks/chc/chc-comp25-benchmarks/extra-small-lia/*.smt2; do
        r=$(timeout $TIMEOUT ./target/release/z4 --chc "$f" 2>&1 | head -1)
        case "$r" in
            sat) sat=$((sat+1));;
            unsat) unsat=$((unsat+1));;
        esac
    done
    echo "Solved: $((sat+unsat))/55 (sat=$sat, unsat=$unsat)"
fi
