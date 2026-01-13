# TLA2 Testing Plan

## Overview

TLA2 correctness is validated by comparison with TLC (the Java reference implementation).
A model checker that produces wrong results is worse than useless - it gives false confidence.

## Test Categories

### 1. TLC Equivalence Tests (Critical)
**Location:** `tests/tlc_comparison/`

Compare TLA2 output against TLC for semantic equivalence:
- State count must match exactly
- Error detection must match (invariant violations, deadlocks)
- Counterexample traces should be valid (not necessarily identical)

**Specs tested:**
| Category | Specs | Purpose |
|----------|-------|---------|
| Core | DieHard, TCommit, TwoPhase | Basic BFS, safety |
| Mutual Exclusion | Peterson, Lock, MCBakery | TLAPS support, locks |
| Distributed | TokenRing, Huang, MCChangRoberts | Termination detection |
| Allocation | SimpleAllocator, SchedulingAllocator | Resource management |
| Classic | DiningPhilosophers, MissionariesAndCannibals | Classic problems |

### 2. Unit Tests
**Location:** `crates/*/tests/`

Test individual components:
- Parser (tla-core)
- Evaluator (tla-check)
- Liveness checker (tla-check)
- SMT translation (tla-smt)

### 3. Integration Tests
**Location:** `test_specs/`

TLA2-specific test specs for:
- Standard library operators
- Edge cases
- Regression tests

### 4. Performance Benchmarks
**Location:** `benches/`

Track performance vs TLC:
- TokenRing (46K states)
- Huang (81K states)
- MCBakery (655K states with ISpec)

## Running Tests

```bash
# All TLC comparison tests
pytest tests/tlc_comparison/ -v

# Quick smoke test
pytest tests/tlc_comparison/ -v -k smoke

# Specific category
pytest tests/tlc_comparison/test_tlc_equivalence.py::TestDistributedSystems -v

# Unit tests
cargo test --release

# Pre-commit verification
./scripts/verify_correctness.sh
```

## Test Output Interpretation

### PASS
TLA2 and TLC produce identical state counts and error detection.

### FAIL
- **State count mismatch:** Bug in state enumeration, action evaluation, or fingerprinting
- **Error detection mismatch:** Bug in invariant checking or deadlock detection
- **Timeout:** Performance regression or infinite loop

### SKIP
- TLC missing required modules (e.g., TLAPS)
- Spec file not found

## Adding New Tests

1. Add spec to `test_specs/` or use from `tlaplus-examples`
2. Verify with TLC first: `java -jar ~/tlaplus/tla2tools.jar -config X.cfg X.tla`
3. Add test to appropriate class in `test_tlc_equivalence.py`
4. Run: `pytest tests/tlc_comparison/test_tlc_equivalence.py::TestClass::test_name -v`

## CI Integration

Pre-commit hook runs:
1. `cargo test --release` - Unit tests
2. `./scripts/verify_correctness.sh` - Known state counts
3. `pytest tests/tlc_comparison/ -x` - TLC comparison (fail fast)

## Known Limitations

1. **ISpec pattern:** TLA2 can't enumerate from invariant predicates (Inv as Init)
2. **Some temporal formulas:** Complex liveness may differ from TLC
3. **Model values:** Symmetry reduction not implemented

## Failure Response

If TLC comparison fails:
1. **Do not ship.** This is a correctness bug.
2. Capture both outputs for debugging
3. Create minimal reproduction spec
4. Fix before any other work
