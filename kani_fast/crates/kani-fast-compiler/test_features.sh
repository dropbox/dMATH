#!/bin/bash
# Category-specific wrapper for feature-focused tests
# Usage: ./test_features.sh [env overrides]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KANI_FAST_CATEGORY=${KANI_FAST_CATEGORY:-features}

exec "$SCRIPT_DIR/test_driver.sh" "$@"
