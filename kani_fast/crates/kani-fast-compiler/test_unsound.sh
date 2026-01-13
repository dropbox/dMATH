#!/bin/bash
# Category-specific wrapper for soundness bug tests (tracked issues)
# Usage: ./test_unsound.sh [env overrides]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KANI_FAST_CATEGORY=${KANI_FAST_CATEGORY:-unsound}

exec "$SCRIPT_DIR/test_driver.sh" "$@"
