#!/usr/bin/env bash
# Build TLA2 with Profile-Guided Optimization (PGO)
#
# PGO improves performance by ~11-15% by optimizing for actual workload patterns.
# This script:
# 1. Builds an instrumented binary to collect profile data
# 2. Runs benchmarks to collect profile data
# 3. Builds an optimized binary using the collected profile
#
# Requirements:
# - llvm-profdata (via Xcode Command Line Tools on macOS)
# - Z3 installed via homebrew

set -euo pipefail

PGO_DIR="/tmp/pgo-data"
Z3_PREFIX="${Z3_PREFIX:-/opt/homebrew/opt/z3}"

echo "=== Step 1: Clean previous profile data ==="
rm -rf "$PGO_DIR"
mkdir -p "$PGO_DIR"

echo "=== Step 2: Build with profile generation ==="
cargo clean -p tla-check -p tla-core -p tla-cli 2>/dev/null || true

Z3_SYS_Z3_HEADER="$Z3_PREFIX/include/z3.h" \
LIBRARY_PATH="$Z3_PREFIX/lib" \
RUSTFLAGS="-Cprofile-generate=$PGO_DIR" \
cargo build --release

echo "=== Step 3: Collect profile data ==="
echo "Running MCLamportMutex..."
./target/release/tla check \
  ~/tlaplus-examples/specifications/lamport_mutex/MCLamportMutex.tla \
  --config ~/tlaplus-examples/specifications/lamport_mutex/MCLamportMutex.cfg \
  --workers 1 > /dev/null 2>&1

# MCLamportMutex provides good profile data for typical model checking workloads

echo "=== Step 4: Merge profile data ==="
# Find llvm-profdata (macOS uses xcrun, Linux uses direct path)
if command -v xcrun &> /dev/null; then
    LLVM_PROFDATA="xcrun llvm-profdata"
else
    LLVM_PROFDATA="llvm-profdata"
fi

$LLVM_PROFDATA merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR"/*.profraw

echo "=== Step 5: Build with profile optimization ==="
cargo clean -p tla-check -p tla-core -p tla-cli 2>/dev/null || true

Z3_SYS_Z3_HEADER="$Z3_PREFIX/include/z3.h" \
LIBRARY_PATH="$Z3_PREFIX/lib" \
RUSTFLAGS="-Cprofile-use=$PGO_DIR/merged.profdata" \
cargo build --release

echo "=== Done! ==="
echo "PGO-optimized binary available at: ./target/release/tla"
echo "Expected improvement: ~11-15% faster execution"
