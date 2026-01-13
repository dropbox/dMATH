#!/bin/bash
# Compare z4-sat against CaDiCaL on DIMACS benchmarks
#
# Usage: ./scripts/compare_cadical.sh [num_files] [timeout_per_file]
#
# Default: First 50 files, 10 second timeout each

set -e

NUM_FILES=${1:-50}
TIMEOUT=${2:-10}

CADICAL="/Users/ayates/z4/reference/cadical/build/cadical"
BENCHMARK_DIR="/Users/ayates/z4/benchmarks/dimacs"

# Build z4-sat example in release mode
echo "Building z4-sat..."
cargo build --release --example bench_dimacs -p z4-sat 2>/dev/null

Z4_BIN="/Users/ayates/z4/target/release/examples/bench_dimacs"

# Check CaDiCaL exists
if [ ! -f "$CADICAL" ]; then
    echo "CaDiCaL not found at $CADICAL"
    echo "Run: cd reference/cadical && ./configure && make"
    exit 1
fi

# Get first N benchmark files
FILES=($(ls "$BENCHMARK_DIR"/*.cnf | head -n "$NUM_FILES"))

echo "Comparing z4-sat vs CaDiCaL on ${#FILES[@]} benchmarks (${TIMEOUT}s timeout)"
echo "==========================================================================="
echo ""

z4_total=0
cadical_total=0
z4_solved=0
cadical_solved=0

printf "%-40s %12s %12s %8s\n" "File" "Z4 (ms)" "CaDiCaL (ms)" "Ratio"
printf "%-40s %12s %12s %8s\n" "----" "-------" "-----------" "-----"

# Create temp dir for cleaned files (CaDiCaL doesn't handle % terminator)
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

for file in "${FILES[@]}"; do
    name=$(basename "$file")

    # Run z4
    z4_start=$(perl -MTime::HiRes=time -e 'printf "%.6f\n", time')
    z4_output=$(timeout "$TIMEOUT"s "$Z4_BIN" "$file" 2>&1 || true)
    z4_end=$(perl -MTime::HiRes=time -e 'printf "%.6f\n", time')
    z4_time=$(echo "($z4_end - $z4_start) * 1000" | bc)

    if echo "$z4_output" | grep -qE "^(SAT|UNSAT)"; then
        z4_solved=$((z4_solved + 1))
        z4_total=$(echo "$z4_total + $z4_time" | bc)
        z4_ms=$(printf "%.3f" "$z4_time")
    else
        z4_ms="TIMEOUT"
        z4_time=0
    fi

    # Create cleaned file for CaDiCaL (strip everything from % terminator onwards)
    clean_file="$TMPDIR/$name"
    sed '/^%/,$d' "$file" > "$clean_file"

    # Run CaDiCaL
    cadical_start=$(perl -MTime::HiRes=time -e 'printf "%.6f\n", time')
    cadical_output=$(timeout "$TIMEOUT"s "$CADICAL" "$clean_file" 2>&1 || true)
    cadical_end=$(perl -MTime::HiRes=time -e 'printf "%.6f\n", time')
    cadical_time=$(echo "($cadical_end - $cadical_start) * 1000" | bc)

    if echo "$cadical_output" | grep -qE "^s (SATISFIABLE|UNSATISFIABLE)"; then
        cadical_solved=$((cadical_solved + 1))
        cadical_total=$(echo "$cadical_total + $cadical_time" | bc)
        cadical_ms=$(printf "%.3f" "$cadical_time")
    else
        cadical_ms="TIMEOUT"
        cadical_time=0
    fi

    # Calculate ratio (z4/cadical)
    if [ "$z4_ms" != "TIMEOUT" ] && [ "$cadical_ms" != "TIMEOUT" ] && [ "$(echo "$cadical_time > 0" | bc)" -eq 1 ]; then
        ratio=$(echo "scale=2; $z4_time / $cadical_time" | bc)
        ratio_str="${ratio}x"
    else
        ratio_str="-"
    fi

    printf "%-40s %12s %12s %8s\n" "$name" "$z4_ms" "$cadical_ms" "$ratio_str"
done

echo ""
echo "==========================================================================="
echo "Summary:"
echo "  Z4:      $z4_solved/$NUM_FILES solved, total ${z4_total}ms"
echo "  CaDiCaL: $cadical_solved/$NUM_FILES solved, total ${cadical_total}ms"

if [ "$(echo "$cadical_total > 0" | bc)" -eq 1 ]; then
    overall_ratio=$(echo "scale=2; $z4_total / $cadical_total" | bc)
    echo "  Overall ratio: ${overall_ratio}x (lower is better)"
fi
