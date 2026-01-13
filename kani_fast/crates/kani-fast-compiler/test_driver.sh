#!/bin/bash
# End-to-end test script for kani-fast-driver
# This script tests the native rustc driver with CHC verification
#
# TEST TIERS:
#   - Default: Run fast tests only (<5s each, ~10 min total)
#   - KANI_FAST_FULL=1: Include medium tests (<30s each)
#   - KANI_FAST_BENCHMARK=1: Full suite including slow tests (>30s each)
#
# PARALLEL MODE: Tests run with 2 parallel jobs by default for stability.
#   Set KANI_FAST_PARALLEL=1 for sequential execution (debugging)
#   Set KANI_FAST_PARALLEL=4 for faster but may have flaky timeouts
#
# QUICK MODE (legacy): Set KANI_FAST_QUICK=1 to skip slow tests and use shorter timeout
#   Example: KANI_FAST_QUICK=1 ./test_driver.sh
#
# FILTER: Set KANI_FAST_FILTER=pattern to only run tests matching pattern
#   Example: KANI_FAST_FILTER=closure ./test_driver.sh
# CATEGORY: Set KANI_FAST_CATEGORY=features|soundness|regression to run a single category
# JSON OUTPUT: Override KANI_FAST_JSON=/tmp/results.json to change where machine-readable
#              output is written (defaults to target/kani-fast-driver/results.json).

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
# Default to 2 parallel jobs for stability (Z4 is fast, but solver contention still possible).
# Set KANI_FAST_PARALLEL=1 for sequential (debugging), KANI_FAST_PARALLEL=4 for faster runs.
PARALLEL_JOBS=${KANI_FAST_PARALLEL:-2}
QUICK_MODE=${KANI_FAST_QUICK:-0}
FULL_MODE=${KANI_FAST_FULL:-0}
BENCHMARK_MODE=${KANI_FAST_BENCHMARK:-0}
TEST_FILTER=${KANI_FAST_FILTER:-}
CATEGORY_FILTER=${KANI_FAST_CATEGORY:-}
JSON_OUTPUT=${KANI_FAST_JSON:-}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Set up environment (detect a usable nightly toolchain for sysroot)
# Note: This function outputs "toolchain:sysroot" to avoid subshell variable scoping issues
detect_sysroot() {
    local sysroot toolchain
    local toolchains=("nightly-2025-11-20" "nightly")

    if command -v rustup >/dev/null 2>&1; then
        for toolchain in "${toolchains[@]}"; do
            if rustup which --toolchain "$toolchain" rustc >/dev/null 2>&1; then
                if sysroot=$(rustup run "$toolchain" rustc --print sysroot 2>/dev/null); then
                    echo "$toolchain:$sysroot"
                    return 0
                fi
            fi
        done
    fi

    if sysroot=$(rustc --print sysroot 2>/dev/null); then
        echo "system:$sysroot"
        return 0
    fi

    return 1
}

TOOLCHAIN_INFO=""
if ! TOOLCHAIN_INFO=$(detect_sysroot); then
    echo -e "${RED}Unable to detect a working Rust toolchain for sysroot lookup.${NC}"
    echo -e "${YELLOW}Install the recommended nightly with: rustup toolchain install nightly-2025-11-20${NC}"
    exit 1
fi

# Parse toolchain:sysroot output
SELECTED_TOOLCHAIN="${TOOLCHAIN_INFO%%:*}"
SYSROOT="${TOOLCHAIN_INFO#*:}"

if [ "$SELECTED_TOOLCHAIN" != "system" ]; then
    export RUSTUP_TOOLCHAIN="$SELECTED_TOOLCHAIN"
    echo -e "${BLUE}Using rustup toolchain: $SELECTED_TOOLCHAIN${NC}"
else
    echo -e "${YELLOW}Using system rustc for sysroot; nightly toolchain not found${NC}"
fi

# Detect host triple for library path
HOST_TRIPLE=$(rustc -vV | grep host | cut -d' ' -f2)

# Set library path based on platform
case "$(uname)" in
    Darwin)
        export DYLD_LIBRARY_PATH="$SYSROOT/lib:$SYSROOT/lib/rustlib/$HOST_TRIPLE/lib"
        ;;
    Linux)
        export LD_LIBRARY_PATH="$SYSROOT/lib:$SYSROOT/lib/rustlib/$HOST_TRIPLE/lib"
        ;;
esac

# Z4 is the sole SMT backend (Z3 was removed in #568)
# The KANI_FAST_CHC_BACKEND env var is no longer used

# Build the driver
echo -e "${YELLOW}Building kani-fast-driver...${NC}"
cargo build --quiet --release

DRIVER="$SCRIPT_DIR/target/release/kani-fast-driver"
TMPDIR=$(mktemp -d)
RESULTS_DIR=$(mktemp -d)
trap "rm -rf $TMPDIR $RESULTS_DIR" EXIT

RESULT_LOG="$RESULTS_DIR/results.jsonl"
: > "$RESULT_LOG"
if [ -z "$JSON_OUTPUT" ]; then
    JSON_OUTPUT="$REPO_ROOT/target/kani-fast-driver/results.json"
fi
mkdir -p "$(dirname "$JSON_OUTPUT")"

PASSED=0
FAILED=0
SKIPPED=0
TEST_COUNT=0
VERIFY_SUCCESS=0     # Tests that actually verified (expect_pass="pass" and passed)
EXPECTED_FAIL=0      # Known limitations (expect_pass="fail" and failed as expected)
UNSOUND_PASS=0       # Soundness bugs (expect_pass="unsound" - incorrectly passes)

# Test tier patterns
# SLOW: Tests known to take >30s (nonlinear math, complex loops, timeouts)
# Includes explicit names from tests/slow.sh: gcd, sum_squares, for_loop_sum, checked_mul_nonlinear
SLOW_PATTERNS="nonlinear|timeout|complex_loop|fibonacci|nested_loops|count_down_exit|^gcd$|sum_squares|for_loop_sum"
# MEDIUM: Tests that take 5-30s (division, some loops, complex closures)
MEDIUM_PATTERNS="division|minmax_update|popcount|while_loop|for_loop|iterator|generic_function|chained_function"

# Timeout settings based on mode
if [ "$BENCHMARK_MODE" = "1" ]; then
    TIMEOUT_SECONDS=180
    MODE_LABEL="benchmark"
    echo -e "${BLUE}Benchmark mode: timeout=${TIMEOUT_SECONDS}s, running ALL tests${NC}"
elif [ "$FULL_MODE" = "1" ]; then
    TIMEOUT_SECONDS=60
    MODE_LABEL="full"
    echo -e "${BLUE}Full mode: timeout=${TIMEOUT_SECONDS}s, running fast+medium tests${NC}"
elif [ "$QUICK_MODE" = "1" ]; then
    TIMEOUT_SECONDS=15
    MODE_LABEL="quick"
    echo -e "${BLUE}Quick mode (legacy): timeout=${TIMEOUT_SECONDS}s${NC}"
elif [ "$PARALLEL_JOBS" -gt 2 ]; then
    TIMEOUT_SECONDS=30  # Fast tests with some buffer for parallelism
    MODE_LABEL="fast"
elif [ "$PARALLEL_JOBS" -gt 0 ]; then
    TIMEOUT_SECONDS=20  # Fast tests default
    MODE_LABEL="fast"
else
    TIMEOUT_SECONDS=15  # Sequential fast tests
    MODE_LABEL="fast"
fi

# Detect timeout command (gtimeout on macOS with coreutils, timeout on Linux)
if command -v gtimeout &> /dev/null; then
    TIMEOUT_CMD="gtimeout"
elif command -v timeout &> /dev/null; then
    TIMEOUT_CMD="timeout"
else
    TIMEOUT_CMD=""
fi

# Kani library linking for transforms and intrinsics
# Allow overrides via KANI_LIB_DIR to avoid hard-coded paths (manager directive).
if [ -n "${KANI_LIB_DIR:-}" ] && [ ! -d "$KANI_LIB_DIR" ]; then
    echo -e "${RED}KANI_LIB_DIR is set but does not exist: $KANI_LIB_DIR${NC}"
    exit 1
fi

if [ -z "${KANI_LIB_DIR:-}" ]; then
    # Prefer the newest installed Kani runtime under ~/.kani/kani-*/lib
    latest_kani_lib=$(ls -1dt "$HOME"/.kani/kani-*/lib 2>/dev/null | head -n 1)
    if [ -n "$latest_kani_lib" ] && [ -d "$latest_kani_lib" ]; then
        KANI_LIB_DIR="$latest_kani_lib"
    else
        # Backward-compatible fallbacks for older installs
        for candidate in "$HOME/.kani/kani-0.66.0-fast/lib" "$HOME/.kani/kani-0.66.0/lib"; do
            if [ -d "$candidate" ]; then
                KANI_LIB_DIR="$candidate"
                break
            fi
        done
    fi
fi

if [ -z "${KANI_LIB_DIR:-}" ] || [ ! -d "$KANI_LIB_DIR" ] || [ ! -f "$KANI_LIB_DIR/libkani.rlib" ]; then
    echo -e "${RED}Missing Kani library; set KANI_LIB_DIR or install with: cargo install --locked kani-verifier${NC}"
    echo -e "${YELLOW}Checked for ~/.kani/kani-*/lib and libkani.rlib presence${NC}"
    exit 1
fi

echo -e "${BLUE}Using Kani library at: $KANI_LIB_DIR${NC}"
KANI_LIB_FLAGS=(-L "$KANI_LIB_DIR" --extern "kani=$KANI_LIB_DIR/libkani.rlib")
KANI_LIB_FLAGS_STR=$(printf "%q " "${KANI_LIB_FLAGS[@]}")

record_result() {
    local name=$1
    local category=$2
    local expect=$3
    local status=$4

    echo "$status" > "$RESULTS_DIR/$name.result"
    local json="{\"name\":\"$name\",\"category\":\"$category\",\"expect\":\"$expect\",\"status\":\"$status\"}"
    echo "$json" >> "$RESULT_LOG"
}

run_test() {
    local name=$1
    local file=$2
    local expect_pass=$3  # "pass" or "fail"
    local category=${4:-}

    if [ -z "$category" ]; then
        case "$name" in
            simple_arithmetic|enum_with_data)
                category="regression"
                ;;
        esac
    fi

    if [ -z "$category" ]; then
        if [[ "$name" =~ soundness ]]; then
            category="soundness"
        elif [[ "$name" =~ regression ]]; then
            category="regression"
        elif [ "$expect_pass" = "fail" ]; then
            category="soundness"
        else
            category="features"
        fi
    fi

    if [ -n "$CATEGORY_FILTER" ] && [ "$category" != "$CATEGORY_FILTER" ]; then
        if [ "$PARALLEL_JOBS" -le 0 ]; then
            echo -e "  Testing $name... ${BLUE}SKIP (category filter: $CATEGORY_FILTER)${NC}"
        fi
        record_result "$name" "$category" "$expect_pass" "SKIP"
        return 0
    fi

    # Apply test filter
    if [ -n "$TEST_FILTER" ] && [[ ! "$name" =~ $TEST_FILTER ]]; then
        record_result "$name" "$category" "$expect_pass" "SKIP"
        return 0
    fi

    # Test tier filtering (default: fast tests only)
    if [ "$BENCHMARK_MODE" != "1" ]; then
        # Skip slow tests unless in benchmark mode
        if [[ "$name" =~ $SLOW_PATTERNS ]]; then
            if [ "$PARALLEL_JOBS" -le 0 ]; then
                echo -e "  Testing $name... ${BLUE}SKIP (slow, use KANI_FAST_BENCHMARK=1)${NC}"
            fi
            record_result "$name" "$category" "$expect_pass" "SKIP"
            return 0
        fi

        # Skip medium tests unless in full or benchmark mode
        if [ "$FULL_MODE" != "1" ] && [[ "$name" =~ $MEDIUM_PATTERNS ]]; then
            if [ "$PARALLEL_JOBS" -le 0 ]; then
                echo -e "  Testing $name... ${BLUE}SKIP (medium, use KANI_FAST_FULL=1)${NC}"
            fi
            record_result "$name" "$category" "$expect_pass" "SKIP"
            return 0
        fi
    fi

    # Legacy quick mode skip (for backward compatibility)
    if [ "$QUICK_MODE" = "1" ] && [[ "$name" =~ $SLOW_PATTERNS ]]; then
        if [ "$PARALLEL_JOBS" -le 0 ]; then
            echo -e "  Testing $name... ${BLUE}SKIP (quick mode)${NC}"
        fi
        record_result "$name" "$category" "$expect_pass" "SKIP"
        return 0
    fi

    TEST_COUNT=$((TEST_COUNT + 1))

    # Parallel mode: run in background using xargs-style batching
    if [ "$PARALLEL_JOBS" -gt 0 ]; then
        # Store test info for batch processing
        echo "$name|$file|$expect_pass|$category" >> "$RESULTS_DIR/test_queue.txt"
        return 0
    else
        # Sequential mode: original behavior
        echo -n "  Testing $name... "

        # Use timeout wrapper if available
        local cmd_prefix=""
        if [ -n "$TIMEOUT_CMD" ]; then
            cmd_prefix="$TIMEOUT_CMD $TIMEOUT_SECONDS"
        fi

        local status=""
        # Run and check for success message in output
        if $cmd_prefix $DRIVER "$file" --crate-type=lib "${KANI_LIB_FLAGS[@]}" 2>&1 | grep -q "All harnesses verified successfully"; then
            if [ "$expect_pass" = "pass" ]; then
                echo -e "${GREEN}PASS${NC}"
                status="PASS"
            elif [ "$expect_pass" = "unsound" ]; then
                # Soundness bug: verification passes but SHOULD fail
                echo -e "${YELLOW}UNSOUND (incorrectly passes)${NC}"
                status="UNSOUND_PASS"
            else
                echo -e "${RED}FAIL (expected failure)${NC}"
                status="FAIL_UNEXPECTED_PASS"
            fi
        else
            # Check if we timed out
            if [ ${PIPESTATUS[0]} -eq 124 ] 2>/dev/null; then
                if [ "$expect_pass" = "fail" ]; then
                    echo -e "${GREEN}PASS (timeout, expected failure)${NC}"
                    status="PASS_EXPECTED_TIMEOUT"
                elif [ "$expect_pass" = "unsound" ]; then
                    # Soundness bug resolved via timeout - still treat as known issue
                    echo -e "${YELLOW}UNSOUND (timeout)${NC}"
                    status="UNSOUND_TIMEOUT"
                else
                    echo -e "${YELLOW}TIMEOUT${NC}"
                    status="TIMEOUT"
                fi
            elif [ "$expect_pass" = "fail" ]; then
                echo -e "${GREEN}PASS (expected failure)${NC}"
                status="PASS_EXPECTED_FAIL"
            elif [ "$expect_pass" = "unsound" ]; then
                # Soundness was fixed! Verification now correctly fails
                echo -e "${GREEN}FIXED (soundness resolved)${NC}"
                status="UNSOUND_FIXED"
            else
                echo -e "${RED}FAIL${NC}"
                status="FAIL"
            fi
        fi

        record_result "$name" "$category" "$expect_pass" "$status"
    fi
}

echo ""
echo -e "${YELLOW}=== Kani Fast Driver End-to-End Tests ===${NC}"
if [ "$PARALLEL_JOBS" -gt 0 ]; then
    echo -e "Running with ${BLUE}$PARALLEL_JOBS parallel jobs${NC}"
fi
if [ -n "$TEST_FILTER" ]; then
    echo -e "Filter: ${BLUE}$TEST_FILTER${NC}"
fi
if [ -n "$CATEGORY_FILTER" ]; then
    echo -e "Category: ${BLUE}$CATEGORY_FILTER${NC}"
fi
echo ""


# Load tests by category to keep this driver modular
TEST_CATEGORY_DIR="$SCRIPT_DIR/tests"
ALL_TEST_CATEGORIES=(regression features soundness limitation unsound slow)

# Get category file path (bash 3.2 compatible - no associative arrays)
get_category_file() {
    local category="$1"
    case "$category" in
        regression)  echo "$TEST_CATEGORY_DIR/regression.sh" ;;
        features)    echo "$TEST_CATEGORY_DIR/features.sh" ;;
        soundness)   echo "$TEST_CATEGORY_DIR/soundness.sh" ;;
        limitation)  echo "$TEST_CATEGORY_DIR/limitation.sh" ;;
        unsound)     echo "$TEST_CATEGORY_DIR/unsound.sh" ;;
        slow)        echo "$TEST_CATEGORY_DIR/slow.sh" ;;
        *)           echo "" ;;
    esac
}

load_test_categories() {
    local categories=()
    if [ -n "$CATEGORY_FILTER" ]; then
        categories=($(printf "%s" "$CATEGORY_FILTER"))
    else
        categories=("${ALL_TEST_CATEGORIES[@]}")
    fi

    for category in "${categories[@]}"; do
        local file
        file=$(get_category_file "$category")
        if [ -n "$file" ] && [ -f "$file" ]; then
            # shellcheck source=/dev/null
            source "$file"
        else
            echo -e "${YELLOW}Skipping missing category file: $category${NC}"
        fi
    done
}

load_test_categories

# Process parallel tests using xargs
if [ "$PARALLEL_JOBS" -gt 0 ]; then
    QUEUE_FILE="$RESULTS_DIR/test_queue.txt"
    if [ -f "$QUEUE_FILE" ]; then
        TOTAL_TESTS=$(wc -l < "$QUEUE_FILE")
        echo -e "\n${YELLOW}Running $TOTAL_TESTS tests with $PARALLEL_JOBS parallel jobs...${NC}"

        # Create a runner script that xargs will call
        RUNNER_SCRIPT="$RESULTS_DIR/run_test.sh"
        cat > "$RUNNER_SCRIPT" << RUNNER_EOF
#!/bin/bash
IFS='|' read -r name file expect_pass category <<< "\$1"
RESULTS_DIR="\$2"
DRIVER="\$3"
TIMEOUT_CMD="\$4"
TIMEOUT_SECONDS="\$5"
KANI_LIB_FLAGS_STR="\$6"
RESULT_LOG="\$7"

# Set up library path and toolchain for rustc driver
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH"
export RUSTUP_TOOLCHAIN="$RUSTUP_TOOLCHAIN"

# Thread safety: disable incremental compilation and use unique Z3 seed
export CARGO_INCREMENTAL=0
export RUSTC_INCREMENTAL=""
export Z3_SEED="\$\$"  # Use PID as unique seed for Z3 solver

# Create unique temp dir for this test to avoid race conditions on shared temp files
TEST_TMPDIR="\$RESULTS_DIR/tmp_\$name"
mkdir -p "\$TEST_TMPDIR"

# Copy test file to unique location
TEST_FILE="\$TEST_TMPDIR/test.rs"
cp "\$file" "\$TEST_FILE"

cmd_prefix=""
if [ -n "\$TIMEOUT_CMD" ]; then
    cmd_prefix="\$TIMEOUT_CMD \$TIMEOUT_SECONDS"
fi

if [ -n "\$KANI_LIB_FLAGS_STR" ]; then
    eval "KANI_LIB_FLAGS=(\$KANI_LIB_FLAGS_STR)"
else
    KANI_LIB_FLAGS=()
fi

if \$cmd_prefix \$DRIVER "\$TEST_FILE" --crate-type=lib --crate-name="\$name" --out-dir="\$TEST_TMPDIR" "\${KANI_LIB_FLAGS[@]}" 2>&1 | grep -q "All harnesses verified successfully"; then
    if [ "\$expect_pass" = "pass" ]; then
        status="PASS"
    elif [ "\$expect_pass" = "unsound" ]; then
        # Soundness bug: verification passes but SHOULD fail
        status="UNSOUND_PASS"
    else
        status="FAIL_UNEXPECTED_PASS"
    fi
else
    exit_code=\${PIPESTATUS[0]}
    if [ "\$exit_code" -eq 124 ] 2>/dev/null; then
        if [ "\$expect_pass" = "fail" ]; then
            status="PASS_EXPECTED_TIMEOUT"
        elif [ "\$expect_pass" = "unsound" ]; then
            status="UNSOUND_TIMEOUT"
        else
            status="TIMEOUT"
        fi
    elif [ "\$expect_pass" = "fail" ]; then
        status="PASS_EXPECTED_FAIL"
    elif [ "\$expect_pass" = "unsound" ]; then
        # Soundness was fixed! Verification now correctly fails
        status="UNSOUND_FIXED"
    else
        status="FAIL"
    fi
fi
echo "\$status" > "\$RESULTS_DIR/\$name.result"
json="{\"name\":\"\$name\",\"category\":\"\$category\",\"expect\":\"\$expect_pass\",\"status\":\"\$status\"}"
echo "\$json" >> "\$RESULT_LOG"
rm -rf "\$TEST_TMPDIR"
RUNNER_EOF
        chmod +x "$RUNNER_SCRIPT"

        # Run tests in parallel using xargs
        cat "$QUEUE_FILE" | xargs -P "$PARALLEL_JOBS" -I{} "$RUNNER_SCRIPT" "{}" "$RESULTS_DIR" "$DRIVER" "$TIMEOUT_CMD" "$TIMEOUT_SECONDS" "$KANI_LIB_FLAGS_STR" "$RESULT_LOG"

        echo -e "${YELLOW}All tests completed.${NC}"
    fi

fi

PASSED=0
FAILED=0
SKIPPED=0
VERIFY_SUCCESS=0     # Tests that actually verified (PASS)
EXPECTED_FAIL=0      # Known limitations (PASS_EXPECTED_FAIL, PASS_EXPECTED_TIMEOUT)
UNSOUND_PASS=0       # Soundness bugs (UNSOUND_PASS, UNSOUND_TIMEOUT)
UNSOUND_FIXED=0      # Soundness bugs that were fixed (UNSOUND_FIXED)
FAILED_TESTS=()

if compgen -G "$RESULTS_DIR"/*.result > /dev/null; then
    for result_file in "$RESULTS_DIR"/*.result; do
        test_name=$(basename "$result_file" .result)
        status=$(cat "$result_file")
        case "$status" in
            PASS)
                PASSED=$((PASSED + 1))
                VERIFY_SUCCESS=$((VERIFY_SUCCESS + 1))
                ;;
            PASS_EXPECTED_FAIL|PASS_EXPECTED_TIMEOUT)
                PASSED=$((PASSED + 1))
                EXPECTED_FAIL=$((EXPECTED_FAIL + 1))
                ;;
            UNSOUND_PASS|UNSOUND_TIMEOUT)
                # Soundness bugs are tracked separately, not counted as failures
                UNSOUND_PASS=$((UNSOUND_PASS + 1))
                ;;
            UNSOUND_FIXED)
                # Soundness was fixed! Count as a pass
                PASSED=$((PASSED + 1))
                UNSOUND_FIXED=$((UNSOUND_FIXED + 1))
                ;;
            SKIP)
                SKIPPED=$((SKIPPED + 1))
                ;;
            FAIL|FAIL_UNEXPECTED_PASS|TIMEOUT)
                FAILED=$((FAILED + 1))
                FAILED_TESTS+=("$test_name: $status")
                ;;
        esac
    done
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed tests:${NC}"
    for failed in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}$failed${NC}"
    done
fi

if [ -n "$JSON_OUTPUT" ]; then
    if command -v python3 > /dev/null 2>&1; then
        MODE_LABEL="$MODE_LABEL" PARALLEL_JOBS="$PARALLEL_JOBS" TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
        CATEGORY_FILTER="$CATEGORY_FILTER" TEST_FILTER="$TEST_FILTER" RESULT_LOG="$RESULT_LOG" JSON_OUTPUT="$JSON_OUTPUT" \
        python3 - <<'PY'
import json
import os
import pathlib

result_log = os.environ["RESULT_LOG"]
output_path = os.environ["JSON_OUTPUT"]
mode = os.environ.get("MODE_LABEL", "fast")
parallel = int(os.environ.get("PARALLEL_JOBS", "0"))
timeout = int(os.environ.get("TIMEOUT_SECONDS", "0"))
category_filter = os.environ.get("CATEGORY_FILTER") or None
name_filter = os.environ.get("TEST_FILTER") or None

results = []
try:
    with open(result_log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
except FileNotFoundError:
    results = []

summary = {
    "verification_successes": 0,
    "expected_failures": 0,
    "soundness_bugs": 0,
    "soundness_fixed": 0,
    "failures": 0,
    "skipped": 0,
    "total": 0,
}

for entry in results:
    status = entry.get("status", "")
    if status == "PASS":
        summary["verification_successes"] += 1
    elif status in {"PASS_EXPECTED_FAIL", "PASS_EXPECTED_TIMEOUT"}:
        summary["expected_failures"] += 1
    elif status in {"UNSOUND_PASS", "UNSOUND_TIMEOUT"}:
        summary["soundness_bugs"] += 1
    elif status == "UNSOUND_FIXED":
        summary["soundness_fixed"] += 1
    elif status == "SKIP":
        summary["skipped"] += 1
    else:
        summary["failures"] += 1

summary["total"] = len(results)

payload = {
    "mode": mode,
    "parallel_jobs": parallel,
    "timeout_seconds": timeout,
    "category_filter": category_filter,
    "name_filter": name_filter,
    "results": results,
    "summary": summary,
}

pathlib.Path(output_path).write_text(json.dumps(payload, indent=2))
print(f"JSON results written to {output_path}")
PY
    else
        echo -e "${YELLOW}JSON output skipped: python3 not available${NC}"
    fi
fi

echo ""
echo -e "${YELLOW}=== Summary ===${NC}"

# Show test tier mode
if [ "$BENCHMARK_MODE" = "1" ]; then
    echo -e "  Mode: ${BLUE}Benchmark (all tests, timeout=${TIMEOUT_SECONDS}s)${NC}"
elif [ "$FULL_MODE" = "1" ]; then
    echo -e "  Mode: ${BLUE}Full (fast+medium, timeout=${TIMEOUT_SECONDS}s)${NC}"
elif [ "$QUICK_MODE" = "1" ]; then
    echo -e "  Mode: ${BLUE}Quick (legacy, timeout=${TIMEOUT_SECONDS}s)${NC}"
else
    echo -e "  Mode: ${BLUE}Fast only (timeout=${TIMEOUT_SECONDS}s)${NC}"
fi
if [ "$PARALLEL_JOBS" -gt 0 ]; then
    echo -e "  Parallelism: ${BLUE}$PARALLEL_JOBS jobs${NC}"
fi
if [ -n "$JSON_OUTPUT" ]; then
    echo -e "  JSON output: ${BLUE}$JSON_OUTPUT${NC}"
fi
echo -e "  ${GREEN}Verification successes: $VERIFY_SUCCESS${NC}"
echo -e "  ${BLUE}Expected failures (known limitations): $EXPECTED_FAIL${NC}"
if [ "$UNSOUND_PASS" -gt 0 ]; then
    echo -e "  ${YELLOW}Soundness bugs (tracked): $UNSOUND_PASS${NC}"
fi
if [ "$UNSOUND_FIXED" -gt 0 ]; then
    echo -e "  ${GREEN}Soundness bugs fixed: $UNSOUND_FIXED${NC}"
fi
echo -e "  ${RED}Unexpected failures: $FAILED${NC}"
if [ "$SKIPPED" -gt 0 ]; then
    echo -e "  ${BLUE}Skipped: $SKIPPED${NC}"
fi
TOTAL=$((VERIFY_SUCCESS + EXPECTED_FAIL + UNSOUND_PASS + UNSOUND_FIXED + FAILED + SKIPPED))
echo -e "  Total: $TOTAL tests"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests behaved as expected!${NC}"
    exit 0
else
    echo -e "${RED}Some tests had unexpected failures.${NC}"
    exit 1
fi
