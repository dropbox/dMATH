#!/bin/bash
# Run the kani-fast-driver with proper library paths
#
# Usage: ./scripts/run-driver.sh <rust-file>
#
# Example:
#   ./scripts/run-driver.sh tests/compiler/simple_proof.rs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPILER_DIR="$PROJECT_ROOT/crates/kani-fast-compiler"

# Build if needed
if [ ! -f "$COMPILER_DIR/target/debug/kani-fast-driver" ]; then
    echo "Building kani-fast-driver..."
    (cd "$COMPILER_DIR" && cargo build)
fi

# Get rustc sysroot and host triple for library path
TOOLCHAIN="nightly-2025-11-20"
RUSTC=$(rustup +$TOOLCHAIN which rustc)
SYSROOT=$(dirname "$(dirname "$RUSTC")")
HOST_TRIPLE=$(rustup run $TOOLCHAIN rustc -vV | grep host | cut -d' ' -f2)

# Run the driver with proper library paths
export DYLD_LIBRARY_PATH="$SYSROOT/lib:$SYSROOT/lib/rustlib/$HOST_TRIPLE/lib:$DYLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$SYSROOT/lib:$SYSROOT/lib/rustlib/$HOST_TRIPLE/lib:$LD_LIBRARY_PATH"

exec "$COMPILER_DIR/target/debug/kani-fast-driver" "$@"
