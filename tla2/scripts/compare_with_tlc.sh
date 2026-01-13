#!/bin/bash
# TLC Comparison Test Suite
# Runs both TLC and TLA2 on specs, compares results for semantic equivalence
#
# Usage:
#   ./scripts/compare_with_tlc.sh                    # Run all comparison tests
#   ./scripts/compare_with_tlc.sh spec.tla spec.cfg  # Compare single spec

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TLA2="$REPO_ROOT/target/release/tla"
TLC_JAR="$HOME/tlaplus/tla2tools.jar"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
SKIP=0

# Build TLA2 if needed
if [ ! -f "$TLA2" ]; then
    echo "Building TLA2..."
    cargo build --release -p tla-cli 2>/dev/null
fi

# Check TLC is available
if [ ! -f "$TLC_JAR" ]; then
    echo "ERROR: TLC not found at $TLC_JAR"
    echo "Please build TLC: cd ~/tlaplus && ant"
    exit 1
fi

# Parse TLC output to extract state count
parse_tlc_states() {
    local output="$1"
    # Match "N distinct states found" and extract N
    local match=$(echo "$output" | grep -oE '[0-9]+ distinct states found' | head -1)
    if [ -n "$match" ]; then
        echo "$match" | grep -oE '^[0-9]+'
    else
        echo "0"
    fi
}

# Parse TLA2 output to extract state count
parse_tla2_states() {
    local output="$1"
    local match=$(echo "$output" | grep -oE 'States found: [0-9]+' | head -1)
    if [ -n "$match" ]; then
        echo "$match" | grep -oE '[0-9]+$'
    else
        echo "0"
    fi
}

# Check if TLC found an error
tlc_has_error() {
    local output="$1"
    echo "$output" | grep -qE 'Error:|Invariant .* is violated|Deadlock reached'
}

# Check if TLA2 found an error
tla2_has_error() {
    local output="$1"
    echo "$output" | grep -qE 'Error:|Invariant.*violated|Deadlock'
}

# Compare single spec
compare_spec() {
    local name="$1"
    local spec="$2"
    local config="$3"

    if [ ! -f "$spec" ]; then
        echo -e "${YELLOW}[ SKIP ]${NC} $name - spec not found: $spec"
        SKIP=$((SKIP + 1))
        return
    fi

    if [ -n "$config" ] && [ ! -f "$config" ]; then
        echo -e "${YELLOW}[ SKIP ]${NC} $name - config not found: $config"
        SKIP=$((SKIP + 1))
        return
    fi

    local spec_dir=$(dirname "$spec")

    # Run TLC
    local tlc_cmd="java -XX:+UseParallelGC -Xmx4g -jar $TLC_JAR"
    if [ -n "$config" ]; then
        tlc_cmd="$tlc_cmd -config $config"
    fi
    tlc_cmd="$tlc_cmd -workers 1 $spec"

    local tlc_output
    tlc_output=$(cd "$spec_dir" && $tlc_cmd 2>&1) || true
    local tlc_states=$(parse_tlc_states "$tlc_output")

    # Run TLA2
    local tla2_cmd="$TLA2 check --workers 1"
    if [ -n "$config" ]; then
        tla2_cmd="$tla2_cmd --config $config"
    fi
    tla2_cmd="$tla2_cmd $spec"

    local tla2_output
    tla2_output=$($tla2_cmd 2>&1) || true
    local tla2_states=$(parse_tla2_states "$tla2_output")

    # Compare results
    local result="PASS"
    local details=""

    # Check state count match
    if [ "$tlc_states" != "$tla2_states" ]; then
        result="FAIL"
        details="states: TLC=$tlc_states, TLA2=$tla2_states"
    fi

    # Check error detection consistency
    local tlc_error=0
    local tla2_error=0
    tlc_has_error "$tlc_output" && tlc_error=1
    tla2_has_error "$tla2_output" && tla2_error=1

    if [ "$tlc_error" != "$tla2_error" ]; then
        result="FAIL"
        details="$details error_detection: TLC=$tlc_error, TLA2=$tla2_error"
    fi

    # Output result
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}[ PASS ]${NC} $name: $tla2_states states"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}[ FAIL ]${NC} $name: $details"
        FAIL=$((FAIL + 1))

        # Show detailed comparison on failure
        echo "  TLC output (last 10 lines):"
        echo "$tlc_output" | tail -10 | sed 's/^/    /'
        echo "  TLA2 output (last 10 lines):"
        echo "$tla2_output" | tail -10 | sed 's/^/    /'
    fi
}

# Main test suite
run_test_suite() {
    echo "=== TLC vs TLA2 Comparison Test Suite ==="
    echo "Date: $(date)"
    echo ""

    local examples_dir="$HOME/tlaplus-examples/specifications"
    local test_specs="$REPO_ROOT/test_specs"

    # Core algorithm specs
    echo "--- Core Algorithms ---"
    compare_spec "DieHard" "$examples_dir/DieHard/DieHard.tla" "$examples_dir/DieHard/DieHard.cfg"
    compare_spec "TwoPhase" "$examples_dir/transaction_commit/TwoPhase.tla" "$examples_dir/transaction_commit/TwoPhase.cfg"
    compare_spec "TCommit" "$examples_dir/transaction_commit/TCommit.tla" "$examples_dir/transaction_commit/TCommit.cfg"

    # Mutual exclusion
    echo ""
    echo "--- Mutual Exclusion ---"
    compare_spec "Peterson" "$examples_dir/locks_auxiliary_vars/Peterson.tla" "$examples_dir/locks_auxiliary_vars/Peterson.cfg"
    compare_spec "Lock" "$examples_dir/locks_auxiliary_vars/Lock.tla" "$examples_dir/locks_auxiliary_vars/Lock.cfg"
    compare_spec "MCBakery" "$examples_dir/Bakery-Boulangerie/MCBakery.tla" "$test_specs/MCBakery.cfg"

    # Distributed systems
    echo ""
    echo "--- Distributed Systems ---"
    compare_spec "TokenRing" "$examples_dir/ewd426/TokenRing.tla" "$examples_dir/ewd426/TokenRing.cfg"
    compare_spec "EWD840" "$examples_dir/ewd840/EWD840.tla" "$examples_dir/ewd840/EWD840.cfg"
    compare_spec "MCChangRoberts" "$examples_dir/chang_roberts/MCChangRoberts.tla" "$examples_dir/chang_roberts/MCChangRoberts.cfg"
    compare_spec "Huang" "$examples_dir/Huang/Huang.tla" "$examples_dir/Huang/Huang.cfg"

    # Resource allocation
    echo ""
    echo "--- Resource Allocation ---"
    compare_spec "SimpleAllocator" "$examples_dir/allocator/SimpleAllocator.tla" "$examples_dir/allocator/SimpleAllocator.cfg"
    compare_spec "SchedulingAllocator" "$examples_dir/allocator/SchedulingAllocator.tla" "$examples_dir/allocator/SchedulingAllocator.cfg"

    # Classic problems
    echo ""
    echo "--- Classic Problems ---"
    compare_spec "DiningPhilosophers" "$examples_dir/DiningPhilosophers/DiningPhilosophers.tla" "$examples_dir/DiningPhilosophers/DiningPhilosophers.cfg"
    compare_spec "MissionariesAndCannibals" "$examples_dir/MissionariesAndCannibals/MissionariesAndCannibals.tla" "$examples_dir/MissionariesAndCannibals/MissionariesAndCannibals.cfg"

    # Summary
    echo ""
    echo "=== Summary ==="
    echo -e "PASS: ${GREEN}$PASS${NC}"
    echo -e "FAIL: ${RED}$FAIL${NC}"
    echo -e "SKIP: ${YELLOW}$SKIP${NC}"

    if [ $FAIL -gt 0 ]; then
        echo ""
        echo -e "${RED}COMPARISON FAILED${NC}: TLA2 differs from TLC on $FAIL spec(s)"
        exit 1
    else
        echo ""
        echo -e "${GREEN}ALL COMPARISONS PASSED${NC}"
        exit 0
    fi
}

# Entry point
if [ $# -eq 2 ]; then
    # Single spec comparison
    compare_spec "$(basename "$1" .tla)" "$1" "$2"
elif [ $# -eq 1 ]; then
    # Single spec without config
    compare_spec "$(basename "$1" .tla)" "$1" ""
else
    # Full test suite
    run_test_suite
fi
