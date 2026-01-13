#!/bin/bash
# Category-specific wrapper for soundness and expected-failure tests
# Usage: ./test_soundness.sh [env overrides]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KANI_FAST_CATEGORY=${KANI_FAST_CATEGORY:-soundness}

exec "$SCRIPT_DIR/test_driver.sh" "$@"
