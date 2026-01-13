#!/bin/bash
# Benchmark script comparing Z4 vs Z3 Spacer performance
#
# Usage: ./scripts/benchmark_solvers.sh [test_filter]
#
# Runs a subset of tests with both Z4-only and hybrid (Z4+Z3) modes,
# generating a comparison report.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DRIVER_DIR="$REPO_ROOT/crates/kani-fast-compiler"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TEST_FILTER=${1:-"simple_arithmetic|boolean_logic|function_inlining|closure_basic|kani_any|bitwise"}

echo -e "${YELLOW}=== Kani Fast Solver Benchmark ===${NC}"
echo "Filter: $TEST_FILTER"
echo ""

cd "$DRIVER_DIR"

# Build first
echo -e "${BLUE}Building driver...${NC}"
cargo build --quiet 2>&1

# Set up environment
TOOLCHAIN="nightly-2025-11-20"
SYSROOT=$(rustup run $TOOLCHAIN rustc --print sysroot)
HOST_TRIPLE=$(rustup run $TOOLCHAIN rustc -vV | grep host | cut -d' ' -f2)
export DYLD_LIBRARY_PATH="$SYSROOT/lib:$SYSROOT/lib/rustlib/$HOST_TRIPLE/lib"
export LD_LIBRARY_PATH="$SYSROOT/lib:$SYSROOT/lib/rustlib/$HOST_TRIPLE/lib"

# Run tests with Z4-only (no fallback)
echo ""
echo -e "${YELLOW}=== Running with Z4 only (no fallback) ===${NC}"
KANI_FAST_NO_FALLBACK=1 KANI_FAST_FILTER="$TEST_FILTER" KANI_FAST_PARALLEL=0 KANI_FAST_QUICK=1 ./test_driver.sh 2>&1 | tee /tmp/z4_only.log

# Run tests with hybrid mode (Z4 + Z3 fallback)
echo ""
echo -e "${YELLOW}=== Running with hybrid mode (Z4 + Z3 fallback) ===${NC}"
KANI_FAST_FILTER="$TEST_FILTER" KANI_FAST_PARALLEL=0 KANI_FAST_QUICK=1 ./test_driver.sh 2>&1 | tee /tmp/hybrid.log

# Parse results and generate report
echo ""
echo -e "${YELLOW}=== Generating comparison report ===${NC}"

z4_pass=$(grep -c "PASS" /tmp/z4_only.log 2>/dev/null || true)
z4_pass=${z4_pass:-0}
z4_fail=$(grep -c -E "FAIL|TIMEOUT" /tmp/z4_only.log 2>/dev/null || true)
z4_fail=${z4_fail:-0}

hybrid_pass=$(grep -c "PASS" /tmp/hybrid.log 2>/dev/null || true)
hybrid_pass=${hybrid_pass:-0}
hybrid_fail=$(grep -c -E "FAIL|TIMEOUT" /tmp/hybrid.log 2>/dev/null || true)
hybrid_fail=${hybrid_fail:-0}

REPORT_DIR="$REPO_ROOT/reports"
mkdir -p "$REPORT_DIR"
REPORT_FILE="$REPORT_DIR/solver_comparison_$(date +%Y%m%d).md"

cat > "$REPORT_FILE" << EOF
# Solver Comparison Report

**Date:** $(date +%Y-%m-%d)
**Filter:** $TEST_FILTER

## Summary

| Mode | Pass | Fail/Timeout | Total |
|------|------|--------------|-------|
| Z4 Only | $z4_pass | $z4_fail | $((z4_pass + z4_fail)) |
| Hybrid (Z4+Z3) | $hybrid_pass | $hybrid_fail | $((hybrid_pass + hybrid_fail)) |

## Analysis

- **Z4 Performance:** Z4 is fast for simple proofs but may return "unknown" for complex loops
- **Hybrid Benefit:** Fallback to Z3 can solve some cases Z4 cannot

## Detailed Results

### Z4 Only
\`\`\`
$(grep -E "Testing.*PASS|Testing.*FAIL|Testing.*TIMEOUT" /tmp/z4_only.log | head -50)
\`\`\`

### Hybrid Mode
\`\`\`
$(grep -E "Testing.*PASS|Testing.*FAIL|Testing.*TIMEOUT" /tmp/hybrid.log | head -50)
\`\`\`
EOF

echo -e "${GREEN}Report written to: $REPORT_FILE${NC}"
echo ""
cat "$REPORT_FILE"
