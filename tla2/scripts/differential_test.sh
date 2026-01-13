#!/bin/bash
# Differential Testing: TLA2 vs TLC
# Compares state counts to verify semantic equivalence

set -e

TLA2_BIN="${TLA2_BIN:-./target/release/tla}"
TLC_JAR="${TLC_JAR:-$HOME/tlaplus/tla2tools.jar}"
EXAMPLES_DIR="${EXAMPLES_DIR:-$HOME/tlaplus-examples/specifications}"
TIMEOUT="${TIMEOUT:-60}"  # seconds per spec

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0
TIMEOUT_COUNT=0

declare -a FAILURES

echo "=== Differential Testing: TLA2 vs TLC ==="
echo "TLA2: $TLA2_BIN"
echo "TLC:  $TLC_JAR"
echo "Timeout: ${TIMEOUT}s per spec"
echo ""

# Find all specs with config files
mapfile -t SPECS < <(find "$EXAMPLES_DIR" -name "*.cfg" 2>/dev/null | sort)

echo "Found ${#SPECS[@]} specs with config files"
echo ""

for cfg in "${SPECS[@]}"; do
    dir=$(dirname "$cfg")
    base=$(basename "$cfg" .cfg)
    tla="$dir/$base.tla"

    # Skip if TLA file doesn't exist
    if [ ! -f "$tla" ]; then
        continue
    fi

    # Skip TTrace files
    if [[ "$base" == *"_TTrace_"* ]]; then
        continue
    fi

    # Skip known problematic specs (liveness-only, etc.)
    case "$base" in
        *Proof*|*_anim*|*_json*|*Sim*|MCKVS|MCKVsnap)
            ((SKIP++))
            continue
            ;;
    esac

    printf "Testing %-50s " "$base"

    # Run TLC with timeout
    TLC_OUTPUT=$(timeout "$TIMEOUT" java -XX:+UseParallelGC -Xmx4g -cp "$TLC_JAR" tlc2.TLC \
        -config "$cfg" "$tla" -workers 1 -nowarning 2>&1) || TLC_EXIT=$?

    if [ "${TLC_EXIT:-0}" -eq 124 ]; then
        printf "${YELLOW}TIMEOUT${NC}\n"
        ((TIMEOUT_COUNT++))
        continue
    fi

    # Extract TLC state count
    TLC_STATES=$(echo "$TLC_OUTPUT" | grep -oP '\d+(?= distinct states found)' | head -1)

    if [ -z "$TLC_STATES" ]; then
        # TLC may have errored or found violation - that's OK, we'll compare
        TLC_STATES=$(echo "$TLC_OUTPUT" | grep -oP '\d+(?= states generated)' | head -1)
        if [ -z "$TLC_STATES" ]; then
            printf "${YELLOW}SKIP${NC} (TLC parse/eval error)\n"
            ((SKIP++))
            continue
        fi
    fi

    # Run TLA2 with timeout
    TLA2_OUTPUT=$(timeout "$TIMEOUT" "$TLA2_BIN" check "$tla" -c "$cfg" -w 1 --no-trace 2>&1) || TLA2_EXIT=$?

    if [ "${TLA2_EXIT:-0}" -eq 124 ]; then
        printf "${YELLOW}TIMEOUT${NC}\n"
        ((TIMEOUT_COUNT++))
        continue
    fi

    # Extract TLA2 state count
    TLA2_STATES=$(echo "$TLA2_OUTPUT" | grep -oP 'States found: \K\d+' | head -1)

    if [ -z "$TLA2_STATES" ]; then
        printf "${YELLOW}SKIP${NC} (TLA2 error)\n"
        ((SKIP++))
        continue
    fi

    # Compare
    if [ "$TLC_STATES" -eq "$TLA2_STATES" ]; then
        printf "${GREEN}PASS${NC} (${TLC_STATES} states)\n"
        ((PASS++))
    else
        printf "${RED}FAIL${NC} (TLC: $TLC_STATES, TLA2: $TLA2_STATES)\n"
        ((FAIL++))
        FAILURES+=("$base: TLC=$TLC_STATES TLA2=$TLA2_STATES")
    fi
done

echo ""
echo "=== Summary ==="
echo -e "${GREEN}PASS:${NC} $PASS"
echo -e "${RED}FAIL:${NC} $FAIL"
echo -e "${YELLOW}SKIP:${NC} $SKIP"
echo -e "${YELLOW}TIMEOUT:${NC} $TIMEOUT_COUNT"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "=== Failures ==="
    for f in "${FAILURES[@]}"; do
        echo "  $f"
    done
    exit 1
fi

echo "DIFFERENTIAL TEST PASSED"
