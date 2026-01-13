#!/bin/bash
# Compare Z3 and Z4 on CHC benchmarks with 30s timeout

TIMEOUT=${1:-30}
BENCH_DIR="/Users/ayates/z4/benchmarks/chc/chc-comp25-benchmarks/extra-small-lia"

echo "=== Z3 vs Z4 CHC comparison (${TIMEOUT}s timeout) ==="

z4_solved=0
z3_solved=0
both_solved=0

for f in "$BENCH_DIR"/*.smt2; do
    name=$(basename "$f" .smt2)
    z4_result=$(timeout "$TIMEOUT" ./target/release/z4 --chc "$f" 2>/dev/null | head -1)
    z3_result=$(timeout "$TIMEOUT" z3 "$f" 2>/dev/null | head -1)

    z4_ok=0
    z3_ok=0

    if [[ "$z4_result" =~ ^(sat|unsat)$ ]]; then
        z4_ok=1
        ((z4_solved++))
    fi

    if [[ "$z3_result" =~ ^(sat|unsat)$ ]]; then
        z3_ok=1
        ((z3_solved++))
    fi

    if [[ $z4_ok -eq 1 ]] && [[ $z3_ok -eq 1 ]]; then
        ((both_solved++))
    fi

    # Show differences
    if [[ "$z4_result" != "$z3_result" ]]; then
        echo "$name: Z4='$z4_result' Z3='$z3_result'"
    fi
done

echo ""
echo "Summary:"
echo "  Z4 solved: $z4_solved"
echo "  Z3 solved: $z3_solved"
echo "  Both solved: $both_solved"
