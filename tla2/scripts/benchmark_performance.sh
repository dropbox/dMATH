#!/bin/bash
# PERFORMANCE BENCHMARK SCRIPT
# Run this BEFORE and AFTER performance changes
# Records timing for comparison

set -e

echo "=== TLA2 Performance Benchmark ==="
echo "Date: $(date)"
echo "Git: $(git rev-parse --short HEAD)"
echo ""

# Build release binary
echo "Building release binary..."
cargo build --release -p tla-cli 2>/dev/null

TLA2="./target/release/tla"

run_benchmark() {
    local name="$1"
    local spec="$2"
    local config="${3:-}"

    if [ ! -f "$spec" ]; then
        echo "| $name | SKIP | - | - |"
        return
    fi

    # Run with timing
    start=$(python3 -c 'import time; print(time.time())')
    if [ -n "$config" ] && [ -f "$config" ]; then
        output=$($TLA2 check "$spec" --config "$config" --workers 1 2>&1) || true
    else
        output=$($TLA2 check "$spec" --workers 1 2>&1) || true
    fi
    end=$(python3 -c 'import time; print(time.time())')

    # Calculate time
    time=$(python3 -c "print(f'{$end - $start:.3f}')")

    # Extract state count
    states=$(echo "$output" | grep -oE "States found: [0-9]+" | grep -oE "[0-9]+" || echo "0")

    # Calculate rate
    if [ "$states" != "0" ]; then
        rate=$(python3 -c "print(int($states / ($end - $start)))")
    else
        rate="N/A"
    fi

    echo "| $name | $states | ${time}s | $rate |"
}

echo "| Spec | States | Time | States/sec |"
echo "|------|--------|------|------------|"

run_benchmark "DieHard" "$HOME/tlaplus-examples/specifications/DieHard/DieHard.tla"
run_benchmark "DiningPhilosophers" "$HOME/tlaplus-examples/specifications/DiningPhilosophers/DiningPhilosophers.tla"
run_benchmark "bcastFolklore" "$HOME/tlaplus-examples/specifications/bcastFolklore/bcastFolklore.tla" "$HOME/tlaplus-examples/specifications/bcastFolklore/bcastFolklore.cfg"

echo ""
echo "Benchmark complete. Compare with previous runs to verify improvement."
