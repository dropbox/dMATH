#!/bin/bash
# Category-specific wrapper for regression-focused tests
# Usage: ./test_regression.sh [env overrides]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KANI_FAST_CATEGORY=${KANI_FAST_CATEGORY:-regression}

exec "$SCRIPT_DIR/test_driver.sh" "$@"
