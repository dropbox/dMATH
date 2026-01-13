#!/bin/bash
# Benchmark Z4 against all competitors on CHC-COMP
# Usage: ./scripts/benchmark_all.sh [timeout] [parallel_jobs]

TIMEOUT=${1:-30}
JOBS=${2:-8}
BENCH_DIR="benchmarks/chc/chc-comp25-benchmarks/extra-small-lia"

echo "=== CHC-COMP Benchmark: Z4 vs All Competitors ==="
echo "Timeout: ${TIMEOUT}s, Parallel jobs: $JOBS"
echo ""

# Function to test a solver
test_solver() {
    local solver_name=$1
    local solver_cmd=$2
    local sat=0
    local unsat=0
    
    for f in "$BENCH_DIR"/*.smt2; do
        result=$(timeout "$TIMEOUT" $solver_cmd "$f" 2>&1 | head -1)
        case "$result" in
            sat) sat=$((sat+1));;
            unsat) unsat=$((unsat+1));;
        esac
    done
    
    echo "$solver_name: $((sat+unsat))/55 (sat=$sat, unsat=$unsat)"
}

# Test Z4
echo "Testing Z4..."
test_solver "Z4" "./target/release/z4 --chc"

# Test Z3 if available
if command -v z3 &> /dev/null; then
    echo "Testing Z3..."
    test_solver "Z3" "z3"
fi

# Test Golem if available
if command -v golem &> /dev/null; then
    echo "Testing Golem..."
    test_solver "Golem" "golem"
fi

# Test CVC5 if available
if command -v cvc5 &> /dev/null; then
    echo "Testing CVC5..."
    test_solver "CVC5" "cvc5 --lang=smt2"
fi

echo ""
echo "=== Done ==="
