#!/bin/bash
# Install git hooks for DashProve
# Run this after cloning the repository

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Installing git hooks for DashProve..."

# Copy hooks from .githooks to .git/hooks
cp "$REPO_ROOT/.githooks/pre-commit" "$REPO_ROOT/.git/hooks/pre-commit"
chmod +x "$REPO_ROOT/.git/hooks/pre-commit"

echo "✓ Pre-commit hook installed"

# Configure git to use .githooks directory (alternative method)
# git config core.hooksPath .githooks

echo ""
echo "✅ Git hooks installed successfully!"
echo ""
echo "The pre-commit hook will run:"
echo "  1. cargo fmt --check (formatting)"
echo "  2. cargo clippy -- -D warnings (linting)"
echo ""
echo "To bypass hooks temporarily: git commit --no-verify"
