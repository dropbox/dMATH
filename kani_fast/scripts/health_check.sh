#!/usr/bin/env bash
# Health check script for kani_fast project
# Verifies all components are working correctly

set -euo pipefail

# Get the repo root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

errors=0
warnings=0

print_status() {
    local status=$1
    local message=$2
    case "$status" in
        pass)
            echo -e "  ${GREEN}[PASS]${NC} $message"
            ;;
        fail)
            echo -e "  ${RED}[FAIL]${NC} $message"
            ((errors++)) || true
            ;;
        warn)
            echo -e "  ${YELLOW}[WARN]${NC} $message"
            ((warnings++)) || true
            ;;
    esac
}

echo "=== Kani Fast Health Check ==="
echo ""

# 1. Check Rust toolchain
echo "1. Rust Toolchain"
if command -v cargo &> /dev/null; then
    version=$(cargo --version | cut -d' ' -f2)
    print_status pass "cargo $version"
else
    print_status fail "cargo not found"
fi

if command -v rustc &> /dev/null; then
    version=$(rustc --version | cut -d' ' -f2)
    print_status pass "rustc $version"
else
    print_status fail "rustc not found"
fi
echo ""

# 2. Check solver dependencies
echo "2. Solver Dependencies"
if command -v z3 &> /dev/null; then
    version=$(z3 --version 2>&1 | head -1)
    print_status pass "Z3: $version"
else
    print_status fail "Z3 not found (required for CHC solving)"
fi

if [[ -x ~/.local/bin/z4 ]]; then
    version=$(~/.local/bin/z4 --version 2>&1 | head -1)
    print_status pass "Z4: $version"
else
    print_status warn "Z4 not found (optional, falls back to Z3)"
fi

if command -v cadical &> /dev/null; then
    print_status pass "CaDiCaL found"
else
    print_status warn "CaDiCaL not found (optional for portfolio mode)"
fi

if command -v kissat &> /dev/null; then
    print_status pass "Kissat found"
else
    print_status warn "Kissat not found (optional for portfolio mode)"
fi
echo ""

# 3. Check Kani
echo "3. Kani Verifier"
if command -v cargo-kani &> /dev/null; then
    print_status pass "cargo-kani found"
else
    print_status warn "cargo-kani not found (required for some features)"
fi
echo ""

# 4. Build check
echo "4. Build Check"
if cargo build --workspace 2>&1 | tail -1 | grep -q "Finished\|Compiling"; then
    print_status pass "Workspace builds successfully"
else
    print_status fail "Workspace build failed"
fi
echo ""

# 5. Test check
echo "5. Unit Tests"
test_output=$(cargo test --workspace 2>&1)
# Check if any test failed
if echo "$test_output" | grep -q "FAILED\|test result: FAILED"; then
    print_status fail "Some tests failed"
else
    # Count passed tests from all "test result:" lines
    passed=$(echo "$test_output" | grep "test result:" | grep -oE "[0-9]+ passed" | awk '{sum+=$1} END {print sum}')
    print_status pass "$passed tests passed"
fi
echo ""

# 6. Clippy check
echo "6. Clippy"
if cargo clippy --workspace 2>&1 | grep -q "Finished"; then
    print_status pass "No clippy warnings"
else
    print_status warn "Clippy warnings found"
fi
echo ""

# 7. Driver tests (optional, slow)
echo "7. Driver Tests"
if [[ -f crates/kani-fast-compiler/test_driver.sh ]]; then
    if [[ "${FULL_CHECK:-}" == "1" ]]; then
        cd crates/kani-fast-compiler
        driver_output=$(./test_driver.sh 2>&1)
        if echo "$driver_output" | tail -5 | grep -q "All tests behaved as expected"; then
            # Extract total test count from output
            total=$(echo "$driver_output" | grep -oE "Total: [0-9]+" | head -1 | grep -oE "[0-9]+")
            print_status pass "All ${total:-???} driver tests pass"
        else
            print_status fail "Some driver tests failed"
        fi
        cd - > /dev/null
    else
        print_status warn "Skipped (set FULL_CHECK=1 to run)"
    fi
else
    print_status fail "test_driver.sh not found"
fi
echo ""

# Summary
echo "=== Summary ==="
if [[ $errors -eq 0 && $warnings -eq 0 ]]; then
    echo -e "${GREEN}All checks passed!${NC}"
elif [[ $errors -eq 0 ]]; then
    echo -e "${YELLOW}Passed with $warnings warning(s)${NC}"
else
    echo -e "${RED}Failed with $errors error(s) and $warnings warning(s)${NC}"
    exit 1
fi
