#!/bin/bash
# Category-specific wrapper for limitation (known unsupported) tests
# Usage: ./test_limitation.sh [env overrides]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KANI_FAST_CATEGORY=${KANI_FAST_CATEGORY:-limitation}

exec "$SCRIPT_DIR/test_driver.sh" "$@"
