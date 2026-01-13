#!/bin/bash
# Test CHC benchmarks

for f in benchmarks/chc/chc-comp25-benchmarks/extra-small-lia/*.smt2; do
  name=$(basename "$f" .smt2)
  z4_result=$(timeout 5 ./target/release/z4 --chc "$f" 2>/dev/null | head -1)
  echo "$name: $z4_result"
done
