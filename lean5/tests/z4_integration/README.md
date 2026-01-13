# Z4 Integration Test Suite

Test files for Z4 SMT solver integration with Lean5.

## Files

| File | Description | Z4 Logic |
|------|-------------|----------|
| `basic_sat.lean` | Propositional SAT/UNSAT | QF_SAT |
| `linear_arith.lean` | Linear integer arithmetic | QF_LIA |
| `bitvector.lean` | Fixed-width bitvector ops | QF_BV |
| `arrays.lean` | Array theory with LIA | QF_AUFLIA |
| `proof_import.lean` | DRAT/LRAT proof verification | - |
| `performance.lean` | Performance benchmarks | Mixed |

## Status

**Phase 1**: Test files created, awaiting Z4 FFI implementation.

These files define the **interface contract** between Lean5 and Z4. They will
not compile until:
1. Z4 implements the FFI layer per `Z4_INTEGRATION_RESPONSE.md`
2. Lean5 implements the tactic stubs in `lean5-elab`

## Running Tests

Once Z4 integration is complete:

```bash
# Run all Z4 integration tests
lake test z4_integration

# Run specific test file
lake test z4_integration/basic_sat

# Run with verbose output (shows SMT-LIB2)
lake test z4_integration -v
```

## Performance Targets

See `performance.lean` for benchmark definitions. Targets from spec:

| Operation | Target |
|-----------|--------|
| FFI overhead | < 10 μs |
| Simple SAT | < 1 ms |
| QF_LIA | < 10 ms |
| QF_BV | < 10 ms |
| DRAT verify | < 100 ms |

## Contact

- Lean5: github.com/dropbox/dMATH/lean5
- Z4: github.com/dropbox/z4
