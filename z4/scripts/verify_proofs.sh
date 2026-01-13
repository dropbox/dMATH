#!/usr/bin/env bash
# Verify DRAT proofs for UNSAT results from z4-sat
#
# This script runs the Rust integration tests which include DRAT verification.
# The tests use the z4-sat library directly with DIMACS files.
#
# Usage: ./scripts/verify_proofs.sh
#
# To install drat-trim manually:
#   git clone https://github.com/marijnheule/drat-trim /tmp/drat-trim
#   cd /tmp/drat-trim && make

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== DRAT Proof Verification ==="
echo ""

# Check for drat-trim and install if needed
DRAT_TRIM="/tmp/drat-trim/drat-trim"
if [ ! -x "$DRAT_TRIM" ]; then
    if ! command -v drat-trim &> /dev/null; then
        echo -e "${YELLOW}Installing drat-trim...${NC}"
        if [ ! -d /tmp/drat-trim ]; then
            git clone https://github.com/marijnheule/drat-trim /tmp/drat-trim
        fi
        (cd /tmp/drat-trim && make -j4)
        echo ""
    fi
fi

echo "Running DRAT proof generation tests..."
cargo test -p z4-sat --test integration test_drat -- --nocapture
if [ $? -eq 0 ]; then
    echo -e "${GREEN}DRAT proof tests passed${NC}"
else
    echo -e "${RED}DRAT proof tests failed${NC}"
    exit 1
fi

echo ""
echo "Running differential tests with model verification..."
cargo test -p z4-sat --test integration differential -- --nocapture
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Differential tests passed${NC}"
else
    echo -e "${RED}Differential tests failed${NC}"
    exit 1
fi

echo ""
echo "Running all z4-sat tests..."
cargo test -p z4-sat
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed${NC}"
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== All proofs verified ===${NC}"
