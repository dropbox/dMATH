#!/usr/bin/env bash
set -euo pipefail

Z4_BIN="${Z4_BIN:-./target/release/z4}"
TIMEOUT_SECS="${TIMEOUT_SECS:-10}"

if [[ ! -x "$Z4_BIN" ]]; then
  echo "Z4 binary not found/executable at: $Z4_BIN"
  echo "Building: cargo build --release"
  cargo build --release
fi

echo "=== Z4 CHC Verification ==="
echo "Z4: $Z4_BIN --chc"
echo "Timeout: ${TIMEOUT_SECS}s"
echo "Git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""

echo "=== Kani Fast (/tmp/kani_benchmarks) ==="
if [[ -d /tmp/kani_benchmarks ]]; then
  found=0
  for f in /tmp/kani_benchmarks/b*.smt2; do
    [[ -e "$f" ]] || continue
    found=1
    echo -n "$(basename "$f"): "
    timeout "$TIMEOUT_SECS" "$Z4_BIN" --chc "$f" 2>/dev/null | head -1 | tr -d '\r' | xargs || echo timeout
  done
  if [[ "$found" -eq 0 ]]; then
    echo "No files matching /tmp/kani_benchmarks/b*.smt2"
  fi
else
  echo "Missing /tmp/kani_benchmarks (skipping)"
fi
echo ""

echo "=== CHC Examples (crates/z4-chc/examples) ==="
if [[ -d crates/z4-chc/examples ]]; then
  for f in crates/z4-chc/examples/*.smt2; do
    [[ -e "$f" ]] || continue
    echo -n "$(basename "$f"): "
    timeout "$TIMEOUT_SECS" "$Z4_BIN" --chc "$f" 2>/dev/null | head -1 | tr -d '\r' | xargs || echo timeout
  done
else
  echo "Missing crates/z4-chc/examples (skipping)"
fi
echo ""

echo "=== CHC-COMP extra-small-lia ==="
sat=0
unsat=0
other=0
total=0

for f in benchmarks/chc/chc-comp25-benchmarks/extra-small-lia/*.smt2; do
  [[ -e "$f" ]] || continue
  total=$((total+1))
  r="$(timeout "$TIMEOUT_SECS" "$Z4_BIN" --chc "$f" 2>/dev/null | head -1 | tr -d '\r' | xargs || true)"
  case "$r" in
    sat) sat=$((sat+1)) ;;
    unsat) unsat=$((unsat+1)) ;;
    *) other=$((other+1)) ;;
  esac
done

echo "Solved: $((sat+unsat))/$total (sat=$sat, unsat=$unsat, other=$other)"
