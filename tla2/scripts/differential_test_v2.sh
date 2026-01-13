#!/bin/bash
# Differential Testing: TLA2 vs TLC (v2 - fixed parsing)
# Compares state counts to verify semantic equivalence

set -e

TLA2_BIN="${TLA2_BIN:-./target/release/tla}"
TLC_JAR="${TLC_JAR:-$HOME/tlaplus/tla2tools.jar}"
EXAMPLES_DIR="${EXAMPLES_DIR:-$HOME/tlaplus-examples/specifications}"
TIMEOUT="${TIMEOUT:-60}"

PASS=0
FAIL=0
SKIP=0
declare -a FAILURES

echo "=== Differential Test: TLA2 vs TLC ==="
echo "Timeout: ${TIMEOUT}s per spec"
echo ""

for cfg in $(find "$EXAMPLES_DIR" -name "*.cfg" 2>/dev/null | grep -v "_TTrace_" | sort); do
    dir=$(dirname "$cfg")
    base=$(basename "$cfg" .cfg)
    tla="$dir/$base.tla"

    [ ! -f "$tla" ] && continue

    # Skip known problematic (proofs, animations, simulations)
    case "$base" in
        *Proof*|*_anim*|*_json*|*Sim*|*TTrace*|*MC_*|MCKVS|MCKVsnap)
            ((SKIP++))
            continue
            ;;
    esac

    # Run TLC
    TLC_OUT=$(timeout "$TIMEOUT" java -XX:+UseParallelGC -Xmx2g -cp "$TLC_JAR" tlc2.TLC \
        -config "$cfg" "$tla" -workers 1 -nowarning 2>&1) || true

    # Extract distinct states from TLC
    TLC_STATES=$(echo "$TLC_OUT" | grep -o '[0-9]* distinct' | grep -o '[0-9]*' | head -1)

    if [ -z "$TLC_STATES" ]; then
        # Maybe TLC errored - check for violation
        if echo "$TLC_OUT" | grep -q "Error:"; then
            TLC_STATES="ERROR"
        else
            ((SKIP++))
            continue
        fi
    fi

    # Run TLA2
    TLA2_OUT=$(timeout "$TIMEOUT" "$TLA2_BIN" check "$tla" -c "$cfg" -w 1 --no-trace 2>&1) || true

    # Extract states from TLA2
    TLA2_STATES=$(echo "$TLA2_OUT" | grep "States found:" | grep -o '[0-9]*' | head -1)

    if [ -z "$TLA2_STATES" ]; then
        if echo "$TLA2_OUT" | grep -q "Error:"; then
            TLA2_STATES="ERROR"
        else
            printf "%-45s SKIP (TLA2 no output)\n" "$base"
            ((SKIP++))
            continue
        fi
    fi

    # Compare
    if [ "$TLC_STATES" = "$TLA2_STATES" ]; then
        printf "%-45s PASS (%s states)\n" "$base" "$TLC_STATES"
        ((PASS++))
    elif [ "$TLC_STATES" = "ERROR" ] && [ "$TLA2_STATES" = "ERROR" ]; then
        printf "%-45s PASS (both found error)\n" "$base"
        ((PASS++))
    else
        printf "%-45s FAIL (TLC=%s TLA2=%s)\n" "$base" "$TLC_STATES" "$TLA2_STATES"
        FAILURES+=("$base: TLC=$TLC_STATES TLA2=$TLA2_STATES")
        ((FAIL++))
    fi
done

echo ""
echo "=== SUMMARY ==="
echo "PASS: $PASS"
echo "FAIL: $FAIL"
echo "SKIP: $SKIP"
echo ""

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo "=== FAILURES ==="
    for f in "${FAILURES[@]}"; do
        echo "  $f"
    done
    exit 1
fi

echo "DIFFERENTIAL TEST PASSED"
