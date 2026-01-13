#!/bin/bash
# Category-specific wrapper for slow (timeout-prone) tests
# Usage: ./test_slow.sh [env overrides]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KANI_FAST_CATEGORY=${KANI_FAST_CATEGORY:-slow}

exec "$SCRIPT_DIR/test_driver.sh" "$@"
