#!/bin/bash
# Compare Z4 and Z3 on CHC benchmarks

for f in benchmarks/chc/chc-comp25-benchmarks/extra-small-lia/*.smt2; do
  name=$(basename "$f" .smt2)
  z4_result=$(timeout 3 ./target/release/z4 --chc "$f" 2>/dev/null | head -1)
  z3_result=$(timeout 3 z3 "$f" 2>/dev/null | head -1)

  # Only show cases where Z3 solves but Z4 doesn't
  if [[ "$z3_result" =~ ^(sat|unsat)$ ]] && [[ ! "$z4_result" =~ ^(sat|unsat)$ ]]; then
    echo "$name: Z3=$z3_result, Z4=$z4_result"
  fi
done
