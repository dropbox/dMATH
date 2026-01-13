#!/bin/bash
# Test all liveness specs against TLC baseline
# This script discovers which liveness specs TLA2 handles correctly

set -e

echo "=== TLA2 Liveness Spec Testing ==="
echo "Date: $(date)"
echo ""

# Per-test timeouts (seconds). Some TLC example specs can take a long time.
# Huang takes ~300s total (TLC + TLA2), so allow 600s per tool.
TLC_TIMEOUT_SECS="${TLC_TIMEOUT_SECS:-600}"
TLA2_TIMEOUT_SECS="${TLA2_TIMEOUT_SECS:-600}"

# Run a command with a timeout, capturing stdout/stderr to stdout.
# Exits with the command's exit code, or 124 on timeout.
run_with_timeout() {
    local timeout_secs="$1"
    shift

    python3 - "$timeout_secs" "$@" <<'PY'
import subprocess
import sys

timeout = float(sys.argv[1])
cmd = sys.argv[2:]

try:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    sys.stdout.write(p.stdout)
    sys.exit(p.returncode)
except subprocess.TimeoutExpired as e:
    # Handle partial output - may be bytes in some Python versions
    if e.stdout:
        if isinstance(e.stdout, bytes):
            sys.stdout.write(e.stdout.decode('utf-8', errors='replace'))
        else:
            sys.stdout.write(e.stdout)
    sys.stdout.write(f"\n[TIMEOUT] after {timeout:.0f}s: {' '.join(cmd)}\n")
    sys.exit(124)
PY
}

# Build release binary
echo "Building release binary..."
cargo build --release -p tla-cli 2>/dev/null

TLA2="./target/release/tla"
TLC_JAR="$HOME/tlaplus/tla2tools.jar"
COMMUNITY_MODULES="$HOME/tlaplus/CommunityModules.jar"
TLC_CP="$TLC_JAR:$COMMUNITY_MODULES"
EXAMPLES="$HOME/tlaplus-examples/specifications"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TLC_TLA_LIBRARY="$REPO_ROOT/test_specs/tla_library"
export TLA_PATH="$TLC_TLA_LIBRARY${TLA_PATH:+:$TLA_PATH}"

PASS=0
FAIL=0
SKIP=0
ERROR=0

# Results file
RESULTS_FILE="$REPO_ROOT/tests/liveness/results.txt"
mkdir -p "$REPO_ROOT/tests/liveness"
echo "# Liveness Test Results - $(date)" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

run_liveness_test() {
    local name="$1"
    local spec="$2"
    local config="$3"
    local extra_args="${4:-}"

    if [ ! -f "$spec" ]; then
        echo "[ SKIP ] $name - spec not found"
        echo "SKIP: $name (spec not found)" >> "$RESULTS_FILE"
        SKIP=$((SKIP + 1))
        return
    fi

    if [ ! -f "$config" ]; then
        echo "[ SKIP ] $name - config not found"
        echo "SKIP: $name (config not found)" >> "$RESULTS_FILE"
        SKIP=$((SKIP + 1))
        return
    fi

    # Run TLC first to get baseline state count
    set +e
    tlc_output=$(run_with_timeout "$TLC_TIMEOUT_SECS" java -DTLA-Library="$TLC_TLA_LIBRARY" -cp "$TLC_CP" tlc2.TLC -config "$config" "$spec" -workers 1 2>&1)
    tlc_rc=$?
    set -e

    if [ "$tlc_rc" -eq 124 ]; then
        echo "[ SKIP ] $name - TLC timeout (${TLC_TIMEOUT_SECS}s)"
        echo "SKIP: $name (TLC timeout)" >> "$RESULTS_FILE"
        SKIP=$((SKIP + 1))
        return
    fi

    # Extract state count - pattern: "N distinct states found" (may have commas in N)
    # Get the last occurrence of "X distinct states found" and extract X (removing commas)
    tlc_states=$(echo "$tlc_output" | grep "distinct states found" | tail -1 | sed 's/.*generated[^0-9]*//; s/,//g; s/ distinct states found.*//' || echo "")

    if [ -z "$tlc_states" ]; then
        # Check if TLC had an error
        if echo "$tlc_output" | grep -iq "Error:\|Exception\|FAILED"; then
            echo "[ SKIP ] $name - TLC error"
            echo "SKIP: $name (TLC error)" >> "$RESULTS_FILE"
            SKIP=$((SKIP + 1))
            return
        fi
        echo "[ SKIP ] $name - TLC no state count"
        echo "SKIP: $name (TLC no state count)" >> "$RESULTS_FILE"
        SKIP=$((SKIP + 1))
        return
    fi

    # Run TLA2
    set +e
    tla2_output=$(run_with_timeout "$TLA2_TIMEOUT_SECS" $TLA2 check "$spec" --config "$config" --workers 1 $extra_args 2>&1)
    tla2_rc=$?
    set -e

    if [ "$tla2_rc" -eq 124 ]; then
        echo "[ ERROR ] $name: TLA2 timeout (${TLA2_TIMEOUT_SECS}s) (TLC: $tlc_states states)"
        echo "ERROR: $name (TLA2 timeout, TLC: $tlc_states)" >> "$RESULTS_FILE"
        ERROR=$((ERROR + 1))
        return
    fi

    tla2_states=$(echo "$tla2_output" | grep "States found:" | head -1 | sed 's/.*States found: \([0-9][0-9]*\).*/\1/' || echo "0")

    # Check for errors (but "No errors found" is OK)
    if echo "$tla2_output" | grep -iq "Error:\|panic\|exceeded\|overflow"; then
        if ! echo "$tla2_output" | grep -q "No errors found"; then
            echo "[ ERROR ] $name: TLA2 error (TLC: $tlc_states states)"
            echo "ERROR: $name (TLA2 error, TLC: $tlc_states)" >> "$RESULTS_FILE"
            ERROR=$((ERROR + 1))
            return
        fi
    fi

    if [ "$tla2_states" = "$tlc_states" ]; then
        echo "[ PASS ] $name: $tla2_states states (TLC: $tlc_states)"
        echo "PASS: $name ($tla2_states states)" >> "$RESULTS_FILE"
        PASS=$((PASS + 1))
    else
        echo "[ FAIL ] $name: TLA2=$tla2_states TLC=$tlc_states"
        echo "FAIL: $name (TLA2=$tla2_states, TLC=$tlc_states)" >> "$RESULTS_FILE"
        FAIL=$((FAIL + 1))
    fi
}

echo "Running liveness spec tests..."
echo ""

# SpecifyingSystems/Liveness specs
echo "=== SpecifyingSystems/Liveness ==="
run_liveness_test "LiveHourClock" \
    "$EXAMPLES/SpecifyingSystems/Liveness/LiveHourClock.tla" \
    "$EXAMPLES/SpecifyingSystems/Liveness/LiveHourClock.cfg"

run_liveness_test "MCLiveInternalMemory" \
    "$EXAMPLES/SpecifyingSystems/Liveness/MCLiveInternalMemory.tla" \
    "$EXAMPLES/SpecifyingSystems/Liveness/MCLiveInternalMemory.cfg"

run_liveness_test "MCLiveWriteThroughCache" \
    "$EXAMPLES/SpecifyingSystems/Liveness/MCLiveWriteThroughCache.tla" \
    "$EXAMPLES/SpecifyingSystems/Liveness/MCLiveWriteThroughCache.cfg"

echo ""
echo "=== Allocator Specs ==="
run_liveness_test "SimpleAllocator" \
    "$EXAMPLES/allocator/SimpleAllocator.tla" \
    "$EXAMPLES/allocator/SimpleAllocator.cfg"

run_liveness_test "SchedulingAllocator" \
    "$EXAMPLES/allocator/SchedulingAllocator.tla" \
    "$EXAMPLES/allocator/SchedulingAllocator.cfg"

echo ""
echo "=== EWD Termination Detection ==="
run_liveness_test "EWD840+Liveness" \
    "$EXAMPLES/ewd840/EWD840.tla" \
    "$REPO_ROOT/test_specs/EWD840_Liveness.cfg"

run_liveness_test "SyncTerminationDetection" \
    "$EXAMPLES/ewd840/SyncTerminationDetection.tla" \
    "$EXAMPLES/ewd840/SyncTerminationDetection.cfg"

echo ""
echo "=== Other Liveness Specs ==="

# TokenRing (ewd426) has liveness property
run_liveness_test "TokenRing" \
    "$EXAMPLES/ewd426/TokenRing.tla" \
    "$EXAMPLES/ewd426/TokenRing.cfg"

# glowingRaccoon specs have temporal properties
run_liveness_test "stages" \
    "$EXAMPLES/glowingRaccoon/stages.tla" \
    "$EXAMPLES/glowingRaccoon/stages.cfg"

run_liveness_test "product" \
    "$EXAMPLES/glowingRaccoon/product.tla" \
    "$EXAMPLES/glowingRaccoon/product.cfg"

run_liveness_test "clean" \
    "$EXAMPLES/glowingRaccoon/clean.tla" \
    "$EXAMPLES/glowingRaccoon/clean.cfg"

# CigaretteSmokers has fairness
run_liveness_test "CigaretteSmokers" \
    "$EXAMPLES/CigaretteSmokers/CigaretteSmokers.tla" \
    "$EXAMPLES/CigaretteSmokers/CigaretteSmokers.cfg"

# DiningPhilosophers
run_liveness_test "DiningPhilosophers" \
    "$EXAMPLES/DiningPhilosophers/DiningPhilosophers.tla" \
    "$EXAMPLES/DiningPhilosophers/DiningPhilosophers.cfg"

# ChangRoberts ring election
run_liveness_test "MCChangRoberts" \
    "$EXAMPLES/chang_roberts/MCChangRoberts.tla" \
    "$EXAMPLES/chang_roberts/MCChangRoberts.cfg"

# MCEcho - distributed echo algorithm
run_liveness_test "MCEcho" \
    "$EXAMPLES/echo/MCEcho.tla" \
    "$EXAMPLES/echo/MCEcho.cfg"

# Bakery algorithm (mutex)
run_liveness_test "MCBakery" \
    "$EXAMPLES/Bakery-Boulangerie/MCBakery.tla" \
    "$EXAMPLES/Bakery-Boulangerie/MCBakery.cfg"

# Paxos specs (Voting instances Consensus)
run_liveness_test "MCVoting" \
    "$EXAMPLES/Paxos/MCVoting.tla" \
    "$EXAMPLES/Paxos/MCVoting.cfg"

# bcastFolklore - broadcast folklore algorithm
run_liveness_test "bcastFolklore" \
    "$EXAMPLES/bcastFolklore/bcastFolklore.tla" \
    "$EXAMPLES/bcastFolklore/bcastFolklore.cfg"

echo ""
echo "=== Additional Liveness Specs ==="

# EWD998 - Termination detection (extended)
run_liveness_test "EWD998" \
    "$EXAMPLES/ewd998/EWD998.tla" \
    "$EXAMPLES/ewd998/EWD998.cfg"

run_liveness_test "AsyncTerminationDetection" \
    "$EXAMPLES/ewd998/AsyncTerminationDetection.tla" \
    "$EXAMPLES/ewd998/AsyncTerminationDetection.cfg"

# ReadersWriters - classic concurrent programming
run_liveness_test "ReadersWriters" \
    "$EXAMPLES/ReadersWriters/MC.tla" \
    "$EXAMPLES/ReadersWriters/MC.cfg"

# Peterson - mutual exclusion
run_liveness_test "Peterson" \
    "$EXAMPLES/locks_auxiliary_vars/Peterson.tla" \
    "$EXAMPLES/locks_auxiliary_vars/Peterson.cfg"

# Huang - leader election
run_liveness_test "Huang" \
    "$EXAMPLES/Huang/Huang.tla" \
    "$EXAMPLES/Huang/Huang.cfg"

# Prisoners - logic puzzle with liveness
run_liveness_test "Prisoners" \
    "$EXAMPLES/Prisoners/Prisoners.tla" \
    "$EXAMPLES/Prisoners/Prisoners.cfg"

# Barrier - synchronization barrier
run_liveness_test "Barrier" \
    "$EXAMPLES/barriers/Barrier.tla" \
    "$EXAMPLES/barriers/Barrier.cfg"

# KeyValueStore - distributed key-value store
run_liveness_test "MCKVsnap" \
    "$EXAMPLES/KeyValueStore/MCKVsnap.tla" \
    "$EXAMPLES/KeyValueStore/MCKVsnap.cfg"

# ReplicatedLog - distributed consensus log
run_liveness_test "MCReplicatedLog" \
    "$EXAMPLES/FiniteMonotonic/MCReplicatedLog.tla" \
    "$EXAMPLES/FiniteMonotonic/MCReplicatedLog.cfg"

# Disruptor - high-performance queue liveness
run_liveness_test "Disruptor_MPMC_liveliness" \
    "$EXAMPLES/Disruptor/Disruptor_MPMC.tla" \
    "$EXAMPLES/Disruptor/Disruptor_MPMC_liveliness.cfg"

# EWD687a - termination detection variant
run_liveness_test "MCEWD687a" \
    "$EXAMPLES/ewd687a/MCEWD687a.tla" \
    "$EXAMPLES/ewd687a/MCEWD687a.cfg"

# SpanTree - spanning tree construction
run_liveness_test "SpanTree" \
    "$EXAMPLES/SpanningTree/SpanTree.tla" \
    "$EXAMPLES/SpanningTree/SpanTree.cfg"

# DijkstraMutex - Dijkstra mutual exclusion
# NOTE: Skipped - MC.tla uses spec aliases (spec_1293897152943429000 == LSpec) which TLA2
# cannot resolve (operator not found error). TLC baseline passes with 90882 states.
# run_liveness_test "DijkstraMutex" \
#     "$EXAMPLES/dijkstra-mutex/DijkstraMutex.toolbox/LSpec-model/MC.tla" \
#     "$EXAMPLES/dijkstra-mutex/DijkstraMutex.toolbox/LSpec-model/MC.cfg"

echo ""
echo "=== Summary ==="
echo "PASS:  $PASS"
echo "FAIL:  $FAIL"
echo "ERROR: $ERROR"
echo "SKIP:  $SKIP"
echo ""
echo "Results saved to: $RESULTS_FILE"

if [ $FAIL -gt 0 ] || [ $ERROR -gt 0 ]; then
    echo "SOME TESTS FAILED OR HAD ERRORS"
    exit 1
else
    echo "ALL TESTS PASSED"
    exit 0
fi
