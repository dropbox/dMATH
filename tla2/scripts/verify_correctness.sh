#!/bin/bash
# MANDATORY VERIFICATION SCRIPT
# Run this BEFORE and AFTER any performance changes
# If ANY spec changes state count, the change is REJECTED

set -e

echo "=== TLA2 Correctness Verification ==="
echo "Date: $(date)"
echo ""

# Build release binary
echo "Building release binary..."
cargo build --release -p tla-cli 2>/dev/null

TLA2="./target/release/tla"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

PASS=0
FAIL=0
SKIP=0
XFAIL=0  # Expected failures (known bugs)
EVAL=0   # Evaluator-only tests (1 state, no transitions)

config_has_property() {
    local cfg="${1:-}"
    [ -n "$cfg" ] && [ -f "$cfg" ] && grep -Eq '^[[:space:]]*PROPERTY\b' "$cfg"
}

# W10: Run a negative test with trace comparison between TLA2 and TLC
# Verifies both tools find the same violation with matching trace signatures
run_negative() {
    local name="$1"
    local spec="$2"
    local config="${3:-}"
    local expected_error="${4:-invariant}"  # invariant, deadlock, or liveness
    local extra_args="${5:-}"

    if [ ! -f "$spec" ]; then
        echo "[ SKIP ] $name (negative) - spec not found"
        SKIP=$((SKIP + 1))
        return
    fi

    # Run TLA2
    local tla2_args="--workers 1"
    if [ -n "$config" ] && [ -f "$config" ]; then
        tla2_args="$tla2_args --config $config"
    fi
    tla2_output=$($TLA2 check "$spec" $tla2_args $extra_args 2>&1) || true

    # Run TLC
    local tlc_jar="$HOME/tlaplus/tla2tools.jar"
    if [ ! -f "$tlc_jar" ]; then
        echo "[ SKIP ] $name (negative) - TLC jar not found"
        SKIP=$((SKIP + 1))
        return
    fi
    local tlc_args="-workers 1"
    if [ -n "$config" ] && [ -f "$config" ]; then
        tlc_args="$tlc_args -config $config"
    fi
    if [ "$expected_error" != "deadlock" ]; then
        tlc_args="$tlc_args -deadlock"  # Disable deadlock checking unless testing for it
    fi
    tlc_output=$(cd "$(dirname "$spec")" && java -jar "$tlc_jar" $tlc_args "$(basename "$spec")" 2>&1) || true

    # Extract TLA2 trace info (use extended regex for multi-digit state numbers)
    local tla2_trace_len=$(echo "$tla2_output" | grep -cE "^State [0-9]+ " || echo "0")
    local tla2_final_state=$(echo "$tla2_output" | awk '/^State [0-9]+ \(/{latest=$0; next} /^  [a-z]/{if(latest!="") vals=vals" "$0} END{print vals}' | tail -1)

    # Extract TLA2 error type
    local tla2_error=""
    if echo "$tla2_output" | grep -q "Error: Invariant"; then
        tla2_error="invariant"
    elif echo "$tla2_output" | grep -q "Error: Deadlock"; then
        tla2_error="deadlock"
    elif echo "$tla2_output" | grep -q "Error:.*liveness\|Error:.*temporal"; then
        tla2_error="liveness"
    fi

    # Extract TLC trace info (use extended regex for multi-digit state numbers)
    local tlc_trace_len=$(echo "$tlc_output" | grep -cE "^State [0-9]+:" || echo "0")
    local tlc_final_state=$(echo "$tlc_output" | awk '/^State [0-9]+:/{latest=$0; next} /^\/ \\/{if(latest!="") vals=vals" "$0} END{print vals}' | tail -1)

    # Extract TLC error type
    local tlc_error=""
    if echo "$tlc_output" | grep -q "Invariant.*violated"; then
        tlc_error="invariant"
    elif echo "$tlc_output" | grep -q "Deadlock reached"; then
        tlc_error="deadlock"
    elif echo "$tlc_output" | grep -q "liveness\|temporal"; then
        tlc_error="liveness"
    fi

    # Compare results
    local passed=true
    local failures=""

    # Check both found errors
    if [ -z "$tla2_error" ]; then
        passed=false
        failures="$failures TLA2 found no error;"
    fi
    if [ -z "$tlc_error" ]; then
        passed=false
        failures="$failures TLC found no error;"
    fi

    # Check error types match
    if [ -n "$tla2_error" ] && [ -n "$tlc_error" ] && [ "$tla2_error" != "$tlc_error" ]; then
        passed=false
        failures="$failures error type mismatch (TLA2:$tla2_error vs TLC:$tlc_error);"
    fi

    # Check trace lengths match
    if [ "$tla2_trace_len" != "$tlc_trace_len" ]; then
        passed=false
        failures="$failures trace length mismatch (TLA2:$tla2_trace_len vs TLC:$tlc_trace_len);"
    fi

    if [ "$passed" = "true" ]; then
        echo "[ PASS ] $name (negative): $tla2_error, trace=$tla2_trace_len states"
        PASS=$((PASS + 1))
    else
        echo "[ FAIL ] $name (negative):$failures"
        echo "TLA2 output: $tla2_output" | head -50
        echo "---"
        echo "TLC output: $tlc_output" | head -50
        FAIL=$((FAIL + 1))
    fi
}

run_check() {
    local name="$1"
    local spec="$2"
    local expected="$3"
    local config="${4:-}"
    local skip_liveness="${5:-0}"
    local extra_args="${6:-}"
    local expected_error="${7:-}"  # W1: Optional expected error type (invariant/deadlock/liveness)

    if [ ! -f "$spec" ]; then
        echo "[ SKIP ] $name - spec not found"
        SKIP=$((SKIP + 1))
        return
    fi

    # W5: If TLC config has PROPERTY, liveness must be enabled.
    local effective_skip_liveness="$skip_liveness"
    if config_has_property "$config"; then
        effective_skip_liveness="0"
    fi

    # Run TLA2 (optionally skip liveness for specs that don't need it)
    if [ "$effective_skip_liveness" = "1" ]; then
        export TLA2_SKIP_LIVENESS=1
    else
        unset TLA2_SKIP_LIVENESS
    fi

    if [ -n "$config" ] && [ -f "$config" ]; then
        if [ "$skip_liveness" = "1" ] && [ "$effective_skip_liveness" = "0" ]; then
            echo "[ INFO ] $name: enabling liveness (PROPERTY in config)"
        fi
        output=$($TLA2 check "$spec" --config "$config" --workers 1 $extra_args 2>&1) || true
    else
        output=$($TLA2 check "$spec" --workers 1 $extra_args 2>&1) || true
    fi

    # Extract state count
    states=$(echo "$output" | grep -oE "States found: [0-9]+" | grep -oE "[0-9]+" || echo "0")

    # W1: Detect errors in output
    error_found=""
    if echo "$output" | grep -q "Error: Invariant"; then
        error_found="invariant"
    elif echo "$output" | grep -q "Error: Deadlock"; then
        error_found="deadlock"
    elif echo "$output" | grep -q "Error:.*liveness\|Error:.*temporal\|Error:.*stuttering"; then
        error_found="liveness"
    elif echo "$output" | grep -q "Error:"; then
        error_found="other"
    fi

    # W1: Verify error detection matches expectation
    local error_ok=true
    local error_msg=""

    if [ -n "$expected_error" ]; then
        # Error expected - verify it was found
        if [ -z "$error_found" ]; then
            error_ok=false
            error_msg="Expected $expected_error error, but TLA2 found no error"
        elif [ "$error_found" != "$expected_error" ]; then
            # Allow some type flexibility (invariant=safety, liveness=temporal)
            case "$expected_error-$error_found" in
                invariant-safety|safety-invariant|liveness-temporal|temporal-liveness)
                    error_ok=true  # Acceptable mismatch
                    ;;
                *)
                    error_ok=false
                    error_msg="Expected $expected_error error, but TLA2 found $error_found"
                    ;;
            esac
        fi
    else
        # No error expected - verify none was found
        if [ -n "$error_found" ]; then
            error_ok=false
            error_msg="TLA2 found unexpected $error_found error"
        fi
    fi

    # Final pass/fail decision
    if [ "$states" = "$expected" ] && [ "$error_ok" = "true" ]; then
        echo "[ PASS ] $name: $states states (expected $expected)"
        PASS=$((PASS + 1))
    else
        if [ "$states" != "$expected" ]; then
            echo "[ FAIL ] $name: $states states (expected $expected)"
        else
            echo "[ FAIL ] $name: $error_msg"
        fi
        echo "Output: $output"
        FAIL=$((FAIL + 1))
    fi
}

# run_eval: Run an evaluator-only test (1 state, no transitions)
# These test expression evaluation via ASSUME/invariants, NOT model checking.
# Output uses [EVAL ] prefix to distinguish from model checking tests.
run_eval() {
    local name="$1"
    local spec="$2"
    local expected="$3"
    local config="${4:-}"
    local extra_args="${5:-}"

    if [ ! -f "$spec" ]; then
        echo "[ SKIP ] $name - spec not found"
        SKIP=$((SKIP + 1))
        return
    fi

    # Evaluator tests always skip liveness (no transitions)
    export TLA2_SKIP_LIVENESS=1

    if [ -n "$config" ] && [ -f "$config" ]; then
        output=$($TLA2 check "$spec" --config "$config" --workers 1 $extra_args 2>&1) || true
    else
        output=$($TLA2 check "$spec" --workers 1 $extra_args 2>&1) || true
    fi

    # Extract state count
    states=$(echo "$output" | grep -oE "States found: [0-9]+" | grep -oE "[0-9]+" || echo "0")

    # Check for unexpected errors
    if echo "$output" | grep -q "Error:"; then
        echo "[ FAIL ] $name (eval): unexpected error in evaluator test"
        echo "Output: $output"
        FAIL=$((FAIL + 1))
        return
    fi

    if [ "$states" = "$expected" ]; then
        echo "[ EVAL ] $name: $states states (evaluator-only)"
        EVAL=$((EVAL + 1))
    else
        echo "[ FAIL ] $name (eval): $states states (expected $expected)"
        echo "Output: $output"
        FAIL=$((FAIL + 1))
    fi
}

# run_xfail: Run a test that is expected to fail (known bug)
# Reports XFAIL if test fails as expected, XPASS if it unexpectedly passes
run_xfail() {
    local name="$1"
    local spec="$2"
    local expected="$3"
    local config="${4:-}"
    local skip_liveness="${5:-0}"
    local extra_args="${6:-}"
    local bug_desc="${7:-Known bug}"  # Description of the bug

    if [ ! -f "$spec" ]; then
        echo "[ SKIP ] $name - spec not found"
        SKIP=$((SKIP + 1))
        return
    fi

    # W5: If TLC config has PROPERTY, liveness must be enabled.
    local effective_skip_liveness="$skip_liveness"
    if config_has_property "$config"; then
        effective_skip_liveness="0"
    fi

    # Run TLA2 (optionally skip liveness for specs that don't need it)
    if [ "$effective_skip_liveness" = "1" ]; then
        export TLA2_SKIP_LIVENESS=1
    else
        unset TLA2_SKIP_LIVENESS
    fi

    if [ -n "$config" ] && [ -f "$config" ]; then
        if [ "$skip_liveness" = "1" ] && [ "$effective_skip_liveness" = "0" ]; then
            echo "[ INFO ] $name: enabling liveness (PROPERTY in config)"
        fi
        output=$($TLA2 check "$spec" --config "$config" --workers 1 $extra_args 2>&1) || true
    else
        output=$($TLA2 check "$spec" --workers 1 $extra_args 2>&1) || true
    fi

    # Extract state count
    states=$(echo "$output" | grep -oE "States found: [0-9]+" | grep -oE "[0-9]+" || echo "0")

    # Detect errors in output
    error_found=""
    if echo "$output" | grep -q "Error: Invariant"; then
        error_found="invariant"
    elif echo "$output" | grep -q "Error: Deadlock"; then
        error_found="deadlock"
    elif echo "$output" | grep -q "Error:"; then
        error_found="other"
    fi

    # For XFAIL, we expect failure (either wrong state count or unexpected error)
    local test_failed=false
    if [ "$states" != "$expected" ] || [ -n "$error_found" ]; then
        test_failed=true
    fi

    if [ "$test_failed" = "true" ]; then
        echo "[XFAIL ] $name: $bug_desc"
        XFAIL=$((XFAIL + 1))
    else
        echo "[XPASS ] $name: bug appears to be fixed! ($states states, no error)"
        PASS=$((PASS + 1))  # Unexpected pass is good - bug might be fixed
    fi
}

echo "Running correctness checks..."
echo ""

# Core specs with expected state counts from TLC baseline
# Most specs skip liveness (5th argument=1) because they don't have liveness properties
# DieHard: NotSolved invariant violation is the point (finding big=4 solution)
run_check "DieHard" "$HOME/tlaplus-examples/specifications/DieHard/DieHard.tla" 14 "" 1 "" "invariant"
# Counter: Deadlock when x reaches MAX (5) - Next guard fails
run_check "Counter" "examples/Counter.tla" 6 "" 1 "" "deadlock"
run_check "Barrier" "$HOME/tlaplus-examples/specifications/barriers/Barrier.tla" 64 "" 1
run_check "DiningPhilosophers" "$HOME/tlaplus-examples/specifications/DiningPhilosophers/DiningPhilosophers.tla" 67 "" 1
# MissionariesAndCannibals: Solution invariant violation is the point (all reach other bank)
run_check "MissionariesAndCannibals" "$HOME/tlaplus-examples/specifications/MissionariesAndCannibals/MissionariesAndCannibals.tla" 61 "" 1 "" "invariant"
run_check "TCommit" "$HOME/tlaplus-examples/specifications/transaction_commit/TCommit.tla" 34 "$HOME/tlaplus-examples/specifications/transaction_commit/TCommit.cfg" 1
run_check "MCChangRoberts" "$HOME/tlaplus-examples/specifications/chang_roberts/MCChangRoberts.tla" 137 "$HOME/tlaplus-examples/specifications/chang_roberts/MCChangRoberts.cfg" 1

# MCLamportMutex - smoke test only (full run is ~724K states and slow in sequential mode)
run_check "MCLamportMutex-smoke" "$HOME/tlaplus-examples/specifications/lamport_mutex/MCLamportMutex.tla" 1000 "$HOME/tlaplus-examples/specifications/lamport_mutex/MCLamportMutex.cfg" 1 "--max-states 1000"

# SimpleAllocator - resource allocator with quantified fairness constraints
# Tests \A c \in Clients: WF_vars(Action(c)) style fairness
run_check "SimpleAllocator" "$HOME/tlaplus-examples/specifications/allocator/SimpleAllocator.tla" 400 "$HOME/tlaplus-examples/specifications/allocator/SimpleAllocator.cfg" 1

# SchedulingAllocator - recursive PermSeqs + quantified fairness extracted from Liveness operator
# Tests recursive LET function definitions and fairness extraction from conjuncted operator references.
run_check "SchedulingAllocator" "$HOME/tlaplus-examples/specifications/allocator/SchedulingAllocator.tla" 1690 "$HOME/tlaplus-examples/specifications/allocator/SchedulingAllocator.cfg" 0

# EWD840 with liveness checking enabled (tests the Liveness property)
# Uses custom config that excludes TDSpec (has module reference issues)
run_check "EWD840+Liveness" "$HOME/tlaplus-examples/specifications/ewd840/EWD840.tla" 302 "$REPO_ROOT/test_specs/EWD840_Liveness.cfg" 0

# EnabledFairness tests ENABLED semantics with WF_vars(Next) disjunctive actions
# Tests that []<>(x = MAX) and []<>(x = 0) hold with weak fairness
run_check "EnabledFairness+Liveness" "$REPO_ROOT/test_specs/EnabledFairness.tla" 4 "$REPO_ROOT/test_specs/EnabledFairness.cfg" 0

# EnabledInAction tests ENABLED operator used inside action guards
# Regression test for bug: TLA2 wasn't evaluating ENABLED in action guards
# IncIfDecEnabled == /\ ENABLED Dec /\ Inc - only enabled when x > 0 (Dec can fire)
# Expected: 6 states (x = 5..10), matches TLC exactly
run_check "EnabledInAction" "$REPO_ROOT/test_specs/EnabledInAction.tla" 6 "$REPO_ROOT/test_specs/EnabledInAction.cfg" 1 "--no-deadlock"

# BidirectionalTransitions from TLC baseline (test-model/)
# Tests WF with disjunctive actions and modular arithmetic
# Test1: A \/ B with mod 3 arithmetic (3 states)
run_check "BidirectionalTransitions1" "$REPO_ROOT/test_specs/BidirectionalTransitions.tla" 3 "$REPO_ROOT/test_specs/BidirectionalTransitions1.cfg" 1

# Test2: C \/ D with mod 4 arithmetic (4 states)
run_check "BidirectionalTransitions2" "$REPO_ROOT/test_specs/BidirectionalTransitions.tla" 4 "$REPO_ROOT/test_specs/BidirectionalTransitions2.cfg" 1

# ============================================================================
# NEGATIVE TESTS (W10: Trace comparison between TLA2 and TLC)
# These specs are EXPECTED to find violations. We verify both tools:
# 1. Find the same type of error (invariant/deadlock/liveness)
# 2. Produce traces of the same length
# ============================================================================
echo ""
echo "=== Negative Tests (error trace comparison) ==="

# DieHard: NotSolved invariant violation expected - finding big=4 is the solution
run_negative "DieHard-trace" "$HOME/tlaplus-examples/specifications/DieHard/DieHard.tla" "" "invariant"

# Counter: Deadlock expected when x reaches MAX (5) - Next guard fails
run_negative "Counter-trace" "$REPO_ROOT/examples/Counter.tla" "" "deadlock"

# MissionariesAndCannibals: Solution invariant violation - all reach other bank
run_negative "MissionariesAndCannibals-trace" "$HOME/tlaplus-examples/specifications/MissionariesAndCannibals/MissionariesAndCannibals.tla" "" "invariant"

# ============================================================================
# EVALUATOR-ONLY TESTS (1 state, Next=UNCHANGED)
# These test expression evaluation via ASSUME/invariants, NOT model checking.
# They have no state transitions - only test the evaluator on the Init state.
# WARNING: These do NOT test the model checker's successor generation!
# ============================================================================
echo ""
echo "=== Evaluator Tests (expression evaluation, not model checking) ==="

# BagsTest - Tests Bags module operators via 126 ASSUME statements
run_eval "BagsTest" "$REPO_ROOT/test_specs/BagsTest.tla" 1 "$REPO_ROOT/test_specs/BagsTest.cfg" "--no-deadlock"

# TLCExtTest - tests TLCModelValue creation and equality via ASSUME statements
run_eval "TLCExtTest" "$REPO_ROOT/test_specs/TLCExtTest.tla" 1 "$REPO_ROOT/test_specs/TLCExtTest.cfg" "--no-deadlock"

# EmptyExistentialQuantifier - tests that \E i \in {}: P is FALSE
run_eval "EmptyExistentialQuantifier" "$REPO_ROOT/test_specs/EmptyExistentialQuantifier.tla" 1 "$REPO_ROOT/test_specs/EmptyExistentialQuantifier.cfg" "--no-deadlock"

# SubSeqExceptTest - tests SubSeq/EXCEPT behavior on seq-like functions
run_eval "SubSeqExceptTest" "$REPO_ROOT/test_specs/SubSeqExceptTest.tla" 1 "$REPO_ROOT/test_specs/SubSeqExceptTest.cfg" "--no-deadlock"

# FunctionOverrideTest - tests @@ (function override) across Records/Seqs/Funcs
run_eval "FunctionOverrideTest" "$REPO_ROOT/test_specs/FunctionOverrideTest.tla" 1 "$REPO_ROOT/test_specs/FunctionOverrideTest.cfg" "--no-deadlock"

# ConstLevelInvariant - constant-level invariant C \subseteq Nat
# Tests that invariants referencing only constants work correctly with infinite sets
# Bug fixed in #528: compiled_guard.rs now handles ModelValue (Nat/Int/Real) in Subseteq
run_check "ConstLevelInvariant" "$REPO_ROOT/test_specs/ConstLevelInvariant.tla" 2 "$REPO_ROOT/test_specs/ConstLevelInvariant.cfg"

# test30 - operator replacement via config
run_eval "test30" "$REPO_ROOT/test_specs/test30.tla" 1 "$REPO_ROOT/test_specs/test30.cfg"

# test7 - predicate logic (\E, \A, CHOOSE)
run_eval "test7" "$REPO_ROOT/test_specs/test7.tla" 1 "$REPO_ROOT/test_specs/test7.cfg"

# test14 - IF/THEN/ELSE, CASE/OTHER, nested junctions
run_eval "test14" "$REPO_ROOT/test_specs/test14.tla" 1 "$REPO_ROOT/test_specs/test14.cfg"

# test21 - priming semantics: primed operator calls, LET/IN with primes
# Note: This has 2 states so it's a mini model-checking test, not pure evaluator
run_check "test21" "$REPO_ROOT/test_specs/test21.tla" 2 "$REPO_ROOT/test_specs/test21.cfg" 1

# test24 - UNCHANGED semantics with quantifiers, CHOOSE, DOMAIN
# Note: This has 2 states so it's a mini model-checking test, not pure evaluator
run_check "test24" "$REPO_ROOT/test_specs/test24.tla" 2 "$REPO_ROOT/test_specs/test24.cfg" 1

# test1 - set equality: literals, comprehensions, SUBSET, UNION, intervals
run_eval "test1" "$REPO_ROOT/test_specs/test1.tla" 1 "$REPO_ROOT/test_specs/test1.cfg"

# test2 - function equality: funcs vs tuples, funcs vs records, EXCEPT
run_eval "test2" "$REPO_ROOT/test_specs/test2.tla" 1 "$REPO_ROOT/test_specs/test2.cfg"

# test3 - function application, EXCEPT with @[idx] syntax
run_eval "test3" "$REPO_ROOT/test_specs/test3.tla" 1 "$REPO_ROOT/test_specs/test3.cfg"

# test4 - fingerprinting of sets: different representations produce same state
run_eval "test4" "$REPO_ROOT/test_specs/test4.tla" 1 "$REPO_ROOT/test_specs/test4.cfg"

# test5 - cartesian product (\X) semantics over finite/infinite sets
run_eval "test5" "$REPO_ROOT/test_specs/test5.tla" 1 "$REPO_ROOT/test_specs/test5.cfg"

# test6 - propositional logic: /\, \/, ~, =>, <=>, \equiv, BOOLEAN
run_eval "test6" "$REPO_ROOT/test_specs/test6.tla" 1 "$REPO_ROOT/test_specs/test6.cfg"

# test8 - set operators: \cup, \cap, \subseteq, \, Seq operations
run_eval "test8" "$REPO_ROOT/test_specs/test8.tla" 1 "$REPO_ROOT/test_specs/test8.cfg"

# test15 - empty set handling: SUBSET {}, UNION {}, quantifiers over {}, 1..0
run_eval "test15" "$REPO_ROOT/test_specs/test15.tla" 1 "$REPO_ROOT/test_specs/test15.cfg"

# test9 - set constructors: {x \in S : P(x)}, {e(x) : x \in S}
run_eval "test9" "$REPO_ROOT/test_specs/test9.tla" 1 "$REPO_ROOT/test_specs/test9.cfg"

# test10 - function definition/application: multi-arg funcs, EXCEPT
run_eval "test10" "$REPO_ROOT/test_specs/test10.tla" 1 "$REPO_ROOT/test_specs/test10.cfg"

# ===== TLA+ Examples repository specs (TLC equivalence validated) =====

# CigaretteSmokers from tlaplus/Examples - classic concurrency problem
# Tests resource sharing with multiple agents
run_check "CigaretteSmokers" "$HOME/tlaplus-examples/specifications/CigaretteSmokers/CigaretteSmokers.tla" 6 "$HOME/tlaplus-examples/specifications/CigaretteSmokers/CigaretteSmokers.cfg" 1

# TokenRing from tlaplus/Examples (ewd426) - Dijkstra's token ring mutual exclusion
# Tests large state space (46656 states) with modular arithmetic
run_check "TokenRing" "$HOME/tlaplus-examples/specifications/ewd426/TokenRing.tla" 46656 "$HOME/tlaplus-examples/specifications/ewd426/TokenRing.cfg" 1

# MCEcho from tlaplus/Examples - Echo algorithm for distributed systems
# Tests spanning tree construction via message passing
run_check "MCEcho" "$HOME/tlaplus-examples/specifications/echo/MCEcho.tla" 75 "$HOME/tlaplus-examples/specifications/echo/MCEcho.cfg" 1

# GameOfLife from tlaplus/Examples - Conway's Game of Life
# Tests large state space (65536 states) with cellular automata rules
run_check "GameOfLife" "$HOME/tlaplus-examples/specifications/GameOfLife/GameOfLife.tla" 65536 "" 1

# HourClock from Specifying Systems - Lamport's classic example
# Tests simple temporal behavior with cyclic state space
run_check "HourClock" "$HOME/tlaplus-examples/specifications/SpecifyingSystems/HourClock/HourClock.tla" 12 "$HOME/tlaplus-examples/specifications/SpecifyingSystems/HourClock/HourClock.cfg" 1

# HourClock2 from Specifying Systems - variation with different Init
run_check "HourClock2" "$HOME/tlaplus-examples/specifications/SpecifyingSystems/HourClock/HourClock2.tla" 12 "$HOME/tlaplus-examples/specifications/SpecifyingSystems/HourClock/HourClock2.cfg" 1

# SKIPPED: SimTokenRing - designed for TLC's -generate mode, not exhaustive model checking
# The spec has ASSUME TLCGet("config").mode = "generate" which fails in bfs/check mode.
# Use `tla simulate` instead if you want to run this spec (it sets mode="generate").
# SimTokenRing is a statistics collection spec, not a model checking correctness test.
echo "[ SKIP ] SimTokenRing - requires 'generate' mode (use 'tla simulate' instead)"
SKIP=$((SKIP + 1))

# clean from tlaplus/Examples (glowingRaccoon) - pipeline cleaning
# Tests simple concurrent state machine
run_check "clean" "$HOME/tlaplus-examples/specifications/glowingRaccoon/clean.tla" 63 "$HOME/tlaplus-examples/specifications/glowingRaccoon/clean.cfg" 1

# product from tlaplus/Examples (glowingRaccoon) - product pipeline
run_check "product" "$HOME/tlaplus-examples/specifications/glowingRaccoon/product.tla" 305 "$HOME/tlaplus-examples/specifications/glowingRaccoon/product.cfg" 1

# stages from tlaplus/Examples (glowingRaccoon) - staged pipeline
run_check "stages" "$HOME/tlaplus-examples/specifications/glowingRaccoon/stages.tla" 83 "$HOME/tlaplus-examples/specifications/glowingRaccoon/stages.cfg" 1

# TwoPhase from tlaplus/Examples (transaction_commit) - Two-phase commit protocol
# Tests distributed transaction commit with resource managers
run_check "TwoPhase" "$HOME/tlaplus-examples/specifications/transaction_commit/TwoPhase.tla" 288 "$HOME/tlaplus-examples/specifications/transaction_commit/TwoPhase.cfg" 1

# AsynchInterface from SpecifyingSystems - Asynchronous interface
# Tests simple send/receive with type invariant
run_check "AsynchInterface" "$HOME/tlaplus-examples/specifications/SpecifyingSystems/AsynchronousInterface/AsynchInterface.tla" 12 "$HOME/tlaplus-examples/specifications/SpecifyingSystems/AsynchronousInterface/AsynchInterface.cfg" 1

# CoffeeCan100Beans from tlaplus/Examples - Coffee can problem with 100 beans
# Tests combinatorial state space with loop invariant (skip liveness)
run_check "CoffeeCan100Beans" "$HOME/tlaplus-examples/specifications/CoffeeCan/CoffeeCan.tla" 5150 "$HOME/tlaplus-examples/specifications/CoffeeCan/CoffeeCan100Beans.cfg" 1

# ReadersWriters from tlaplus/Examples - Classic readers-writers concurrency problem
# Tests mutual exclusion with multiple readers/writers (skip liveness - complex temporal)
run_check "ReadersWriters" "$HOME/tlaplus-examples/specifications/ReadersWriters/MC.tla" 21527 "$HOME/tlaplus-examples/specifications/ReadersWriters/MC.cfg" 1

# MCMajority from tlaplus/Examples - Boyer-Moore majority vote algorithm
# Tests voting correctness invariants with 1092 initial states
run_check "MCMajority" "$HOME/tlaplus-examples/specifications/Majority/MCMajority.tla" 2733 "$HOME/tlaplus-examples/specifications/Majority/MCMajority.cfg" 1

# MCInternalMemory from SpecifyingSystems - Internal memory abstraction
# Tests request/response memory interface with 4408 states
run_check "MCInternalMemory" "$HOME/tlaplus-examples/specifications/SpecifyingSystems/CachingMemory/MCInternalMemory.tla" 4408 "$HOME/tlaplus-examples/specifications/SpecifyingSystems/CachingMemory/MCInternalMemory.cfg" 1

# LiveHourClock from SpecifyingSystems - Hour clock with liveness
# Tests simple temporal behavior with liveness properties (skip liveness for safety check)
run_check "LiveHourClock" "$HOME/tlaplus-examples/specifications/SpecifyingSystems/Liveness/LiveHourClock.tla" 12 "$HOME/tlaplus-examples/specifications/SpecifyingSystems/Liveness/LiveHourClock.cfg" 1

# MCLiveInternalMemory from SpecifyingSystems - Internal memory with liveness
# Tests liveness properties with 4408 states (skip liveness for safety check)
run_check "MCLiveInternalMemory" "$HOME/tlaplus-examples/specifications/SpecifyingSystems/Liveness/MCLiveInternalMemory.tla" 4408 "$HOME/tlaplus-examples/specifications/SpecifyingSystems/Liveness/MCLiveInternalMemory.cfg" 1

# nbacg_guer01 from tlaplus/Examples - Non-blocking atomic commitment
# Tests distributed atomic commitment with 24922 states (skip liveness)
run_check "nbacg_guer01" "$HOME/tlaplus-examples/specifications/nbacg_guer01/nbacg_guer01.tla" 24922 "$HOME/tlaplus-examples/specifications/nbacg_guer01/nbacg_guer01.cfg" 1

# test16 - one-tuples: tuple literals, quantifiers, comprehensions
run_eval "test16" "$REPO_ROOT/test_specs/test16.tla" 1 "$REPO_ROOT/test_specs/test16.cfg"

# test19 - large initial state space (400K states - actual model checking test!)
run_check "test19" "$REPO_ROOT/test_specs/test19.tla" 400000 "$REPO_ROOT/test_specs/test19.cfg" 1

# test35 - Sequences module: Seq, \o, Len, Append, Head, Tail, SubSeq, SelectSeq
run_eval "test35" "$REPO_ROOT/test_specs/test35.tla" 1 "$REPO_ROOT/test_specs/test35.cfg"

# test37 - FiniteSets module: IsFiniteSet, Cardinality
run_eval "test37" "$REPO_ROOT/test_specs/test37.tla" 1 "$REPO_ROOT/test_specs/test37.cfg"

# TLAPSTest - TLAPS module operators (7 states - tests state transitions too)
run_check "TLAPSTest" "$REPO_ROOT/test_specs/TLAPSTest.tla" 7 "$REPO_ROOT/test_specs/TLAPSTest.cfg" 1

# AddTwoTest - TLAPS operators (5 states - tests state transitions too)
run_check "AddTwoTest" "$REPO_ROOT/test_specs/AddTwoTest.tla" 5 "$REPO_ROOT/test_specs/AddTwoTest.cfg" 1

# test11 - DOMAIN and Function Sets
run_eval "test11" "$REPO_ROOT/test_specs/test11.tla" 1 "$REPO_ROOT/test_specs/test11.cfg"

# Github407 - existential quantifier with set union (4 states - model checking)
run_check "Github407" "$REPO_ROOT/test_specs/Github407.tla" 4 "$REPO_ROOT/test_specs/Github407.cfg" 1

# Github1145 - SubSeq/EXCEPT regression
run_eval "Github1145" "$REPO_ROOT/test_specs/Github1145.tla" 1 "$REPO_ROOT/test_specs/Github1145.cfg"

# test40 - Naturals module: +, -, *, ^, \div, %
run_eval "test40" "$REPO_ROOT/test_specs/test40.tla" 1 "$REPO_ROOT/test_specs/test40.cfg" "--no-deadlock"

# test41 - Integers module: negative numbers, Int membership
run_eval "test41" "$REPO_ROOT/test_specs/test41.tla" 1 "$REPO_ROOT/test_specs/test41.cfg" "--no-deadlock"

# test46 - \prec operator, UNION comprehension (6 states - model checking)
run_check "test46" "$REPO_ROOT/test_specs/test46.tla" 6 "$REPO_ROOT/test_specs/test46.cfg" 1 "--no-deadlock"

# test47 - EXTENDS with multiple modules (3 states - model checking)
run_check "test47" "$REPO_ROOT/test_specs/test47.tla" 3 "$REPO_ROOT/test_specs/test47.cfg" 1

# test32 - [A]_e and <<A>>_e stuttering-step expressions
run_eval "test32" "$REPO_ROOT/test_specs/test32.tla" 1 "$REPO_ROOT/test_specs/test32.cfg"

# Huang - Termination detection algorithm using dyadic rationals
# Tests DyadicRationals community module (Zero, One, Add, Half, IsDyadicRational)
# Requires passing Add as higher-order argument to FoldFunction
run_check "Huang" "$HOME/tlaplus-examples/specifications/Huang/Huang.tla" 81256 "$HOME/tlaplus-examples/specifications/Huang/Huang.cfg" 0

# SyncTerminationDetection - Termination detection with synchronous rounds
# Tests INSTANCE with parameter substitution and state space enumeration
run_check "SyncTerminationDetection" "$HOME/tlaplus-examples/specifications/ewd840/SyncTerminationDetection.tla" 129 "$HOME/tlaplus-examples/specifications/ewd840/SyncTerminationDetection.cfg" 0

# MCFindHighest - Find highest element in array (uses TLAPS proof stub)
# Tests TLAPS module stub support and loop invariants
run_check "MCFindHighest" "$HOME/tlaplus-examples/specifications/LearnProofs/MCFindHighest.tla" 742 "$HOME/tlaplus-examples/specifications/LearnProofs/MCFindHighest.cfg" 1

# Lock - Simple lock with auxiliary variables (uses TLAPS proof stub)
# Tests TLAPS stub support and lock semantics
run_check "Lock" "$HOME/tlaplus-examples/specifications/locks_auxiliary_vars/Lock.tla" 12 "$HOME/tlaplus-examples/specifications/locks_auxiliary_vars/Lock.cfg" 1

# Peterson - Peterson's mutual exclusion algorithm (uses TLAPS proof stub)
# Tests TLAPS stub support and refinement proofs
run_check "Peterson" "$HOME/tlaplus-examples/specifications/locks_auxiliary_vars/Peterson.tla" 42 "$HOME/tlaplus-examples/specifications/locks_auxiliary_vars/Peterson.cfg" 0

# MCBakery - Lamport's bakery algorithm with TLAPS proof annotations
# Tests TLAPS module support with proof-annotated spec (uses custom config for regular Spec)
run_check "MCBakery" "$HOME/tlaplus-examples/specifications/Bakery-Boulangerie/MCBakery.tla" 2303 "$REPO_ROOT/test_specs/MCBakery.cfg" 1

# MCPaxos - Paxos consensus algorithm model checking spec
# Tests nested INSTANCE with implicit substitution (Value from outer module)
run_check "MCPaxos" "$HOME/tlaplus-examples/specifications/Paxos/MCPaxos.tla" 25 "$HOME/tlaplus-examples/specifications/Paxos/MCPaxos.cfg" 0

# MCVoting - Voting layer of Paxos
# Tests INSTANCE with implicit substitution: Consensus!chosen uses Voting's chosen operator
# This was previously broken - fixed in #510 by handling implicit VARIABLE substitution
run_check "MCVoting" "$HOME/tlaplus-examples/specifications/Paxos/MCVoting.tla" 77 "$HOME/tlaplus-examples/specifications/Paxos/MCVoting.cfg" 0

# MCYoYoPruning - YoYo leader election with pruning optimization
# Tests ENABLED operator in invariants (FinishIffTerminated uses ENABLED Next)
# Fixed in #512 - ENABLED invariants now route through liveness checker
run_check "MCYoYoPruning" "$HOME/tlaplus-examples/specifications/YoYo/MCYoYoPruning.tla" 102 "$HOME/tlaplus-examples/specifications/YoYo/MCYoYoPruning.cfg" 0

# SingleLaneBridge - Cars crossing a single-lane bridge
# Tests liveness properties with 3605 states and fair progress
run_check "SingleLaneBridge" "$HOME/tlaplus-examples/specifications/SingleLaneBridge/MC.tla" 3605 "$HOME/tlaplus-examples/specifications/SingleLaneBridge/MC.cfg" 0

# SimplifiedFastPaxos/Paxos - Simplified Paxos from Lamport's tutorial
# Tests consensus with symmetry reduction (174 permutations)
run_check "SimplifiedPaxos" "$HOME/tlaplus-examples/specifications/SimplifiedFastPaxos/Paxos.tla" 1207 "$HOME/tlaplus-examples/specifications/SimplifiedFastPaxos/Paxos.cfg" 0

# SimplifiedFastPaxos/FastPaxos - Fast Paxos variant
# Tests consensus with fast path optimization (25617 states, 70s runtime)
run_check "SimplifiedFastPaxos" "$HOME/tlaplus-examples/specifications/SimplifiedFastPaxos/FastPaxos.tla" 25617 "$HOME/tlaplus-examples/specifications/SimplifiedFastPaxos/FastPaxos.cfg" 0

# MCKVsnap - Key-Value store with snapshot isolation
# Tests snapshot isolation invariant and termination (32293 states)
run_check "MCKVsnap" "$HOME/tlaplus-examples/specifications/KeyValueStore/MCKVsnap.tla" 32293 "$HOME/tlaplus-examples/specifications/KeyValueStore/MCKVsnap.cfg" 0

# EnvironmentController-smoke - Eventually perfect failure detector (large)
# Smoke test (limit) to ensure INSTANCE inlining preserves instance-local ops (Age_Channel!Unpack).
run_check "EnvironmentController-smoke" "$HOME/tlaplus-examples/specifications/detector_chan96/EnvironmentController.tla" 1001 "$HOME/tlaplus-examples/specifications/detector_chan96/EnvironmentController.cfg" 1 "--no-trace --max-states 1000"

# MCInnerFIFO from SpecifyingSystems - Inner FIFO queue implementation
# Tests buffered channel with send/receive semantics (3864 states)
run_check "MCInnerFIFO" "$HOME/tlaplus-examples/specifications/SpecifyingSystems/FIFO/MCInnerFIFO.tla" 3864 "$HOME/tlaplus-examples/specifications/SpecifyingSystems/FIFO/MCInnerFIFO.cfg" 1

# MCtcp - TCP connection state machine
# Tests TCP three-way handshake and state transitions (1182 states)
run_check "MCtcp" "$HOME/tlaplus-examples/specifications/tcp/MCtcp.tla" 1182 "$HOME/tlaplus-examples/specifications/tcp/MCtcp.cfg" 0

# nbacc_ray97 - Non-blocking atomic commitment with raynal97
# Tests distributed commit protocol (3016 states)
run_check "nbacc_ray97" "$HOME/tlaplus-examples/specifications/nbacc_ray97/nbacc_ray97.tla" 3016 "$HOME/tlaplus-examples/specifications/nbacc_ray97/nbacc_ray97.cfg" 1

# MCInnerSequential from SpecifyingSystems - Sequential execution model
# Tests linearizable operations with refinement (3528 states)
run_check "MCInnerSequential" "$HOME/tlaplus-examples/specifications/SpecifyingSystems/AdvancedExamples/MCInnerSequential.tla" 3528 "$HOME/tlaplus-examples/specifications/SpecifyingSystems/AdvancedExamples/MCInnerSequential.cfg" 0

# MCWriteThroughCache from SpecifyingSystems - Write-through cache with memory queue
# Tests LET binding error handling: LET r == Head(memQ)[2] IN /\ guard /\ ...
# Previously failed with IndexOutOfBounds when memQ empty (5196 states, TLC verified)
run_check "MCWriteThroughCache" "$HOME/tlaplus-examples/specifications/SpecifyingSystems/CachingMemory/MCWriteThroughCache.tla" 5196 "$HOME/tlaplus-examples/specifications/SpecifyingSystems/CachingMemory/MCWriteThroughCache.cfg" 1

# SpanTree from SpanningTree - Distributed spanning tree construction
# Tests broadcast-based spanning tree with liveness properties (1236 states, TLC verified)
run_check "SpanTree" "$HOME/tlaplus-examples/specifications/SpanningTree/SpanTree.tla" 1236 "$HOME/tlaplus-examples/specifications/SpanningTree/SpanTree.cfg" 0

# Prisoners - 100 prisoners lightbulb problem
# Tests counting strategy with safety and liveness (214 states, TLC verified)
run_check "Prisoners" "$HOME/tlaplus-examples/specifications/Prisoners/Prisoners.tla" 214 "$HOME/tlaplus-examples/specifications/Prisoners/Prisoners.cfg" 0

# Prisoner - Single-switch prisoner puzzle variant
# Tests Init with negated constants (~Light_Unknown) in disjunctions (16 states, TLC verified)
run_check "Prisoner" "$HOME/tlaplus-examples/specifications/Prisoners_Single_Switch/Prisoner.tla" 16 "$HOME/tlaplus-examples/specifications/Prisoners_Single_Switch/Prisoner.cfg" 0

# Chameneos - Concurrency benchmark from Erlang shootout
# Tests rendezvous-style meeting with color changes (34534 states, TLC verified)
run_check "Chameneos" "$HOME/tlaplus-examples/specifications/Chameneos/Chameneos.tla" 34534 "$HOME/tlaplus-examples/specifications/Chameneos/Chameneos.cfg" 1

# MCEWD687a - Dijkstra's distributed termination detection
# Tests tree-based termination detection (18028 states, TLC verified)
run_check "MCEWD687a" "$HOME/tlaplus-examples/specifications/ewd687a/MCEWD687a.tla" 18028 "$HOME/tlaplus-examples/specifications/ewd687a/MCEWD687a.cfg" 0

# === AUDIT 2026-01-05: Specs that were incorrectly skipped ===

# MCYoYoNoPruning - YoYo leader election (was marked as missing IsUndirectedGraph)
run_check "MCYoYoNoPruning" "$HOME/tlaplus-examples/specifications/YoYo/MCYoYoNoPruning.tla" 60 "$HOME/tlaplus-examples/specifications/YoYo/MCYoYoNoPruning.cfg" 0

# Barriers - Multi-barrier synchronization (29279 states, TLC verified)
run_check "Barriers" "$HOME/tlaplus-examples/specifications/barriers/Barriers.tla" 29279 "$HOME/tlaplus-examples/specifications/barriers/Barriers.cfg" 0

# === Python tests (W7: Integrate Python tests into verify_correctness.sh) ===
# Run a subset of Python tests for quick validation
# Full test suite can be run separately with: pytest tests/tlc_comparison/ -m fast
echo ""
echo "=== Running Python TLC Comparison Tests (fast subset) ==="
if [ -d "$REPO_ROOT/venv" ] && [ -f "$REPO_ROOT/venv/bin/pytest" ]; then
    # Run only fast tests to keep verification quick
    # -m fast: only fast-marked tests
    # -x: stop on first failure
    # --tb=short: short traceback
    source "$REPO_ROOT/venv/bin/activate"
    if pytest tests/tlc_comparison/test_tlaplus_examples.py -m fast -x --tb=short -q 2>/dev/null; then
        echo "[ PASS ] Python TLC comparison tests"
        PASS=$((PASS + 1))
    else
        echo "[ FAIL ] Python TLC comparison tests"
        FAIL=$((FAIL + 1))
    fi
    deactivate 2>/dev/null || true
else
    echo "[ SKIP ] Python tests - venv/pytest not found"
    echo "         To enable: python3 -m venv venv && venv/bin/pip install pytest"
    SKIP=$((SKIP + 1))
fi

echo ""
echo "=== Summary ==="
echo "PASS:  $PASS (model checking tests)"
echo "EVAL:  $EVAL (evaluator-only tests - no state transitions)"
echo "FAIL:  $FAIL"
echo "XFAIL: $XFAIL (expected failures - known bugs)"
echo "SKIP:  $SKIP"
echo ""

TOTAL=$((PASS + EVAL))
if [ $FAIL -gt 0 ]; then
    echo "VERIFICATION FAILED - DO NOT COMMIT"
    exit 1
else
    if [ $XFAIL -gt 0 ]; then
        echo "VERIFICATION PASSED ($TOTAL tests, with $XFAIL known bugs)"
    else
        echo "VERIFICATION PASSED ($TOTAL tests)"
    fi
    exit 0
fi
