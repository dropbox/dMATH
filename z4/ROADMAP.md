# Roadmap: z4

> High-Performance SMT Solver in Rust

## Current Focus

CHC solving improvements and soundness hardening.

## Active Issues

| # | Priority | Title |
|---|----------|-------|
| #7 | P0 | CRITICAL: Z4 PDR Soundness Bug - False Safe Results |
| #11 | P2 | [RESEARCHER] Create system diagrams |
| #10 | P2 | TLAPS backend requirements for TLA2 |
| #9 | P2 | Native compilation for solver expressions via tRust |
| #2 | P2 | Add XOR Gauss-Jordan theory |
| #1 | P3 | Constraint diagnosis (UNSAT cores + MUS/MCS) |

## Performance Status

| Theory | Z4 vs Z3 | Status |
|--------|----------|--------|
| QF_LIA | 1.50x faster | Verified |
| QF_UF | 1.79x faster | Verified |
| QF_BV | 1.19x faster | Verified |
| QF_LRA | 1.37x faster | Verified |
| CHC | 53/55 vs 16/55 | Z4 dominates |

## Phases

1. **SAT Core** - Complete (beats CaDiCaL on uf250)
2. **SMT Infrastructure** - Complete
3. **Core Theories** - Complete (QF_LIA, QF_UF, QF_BV, QF_LRA)
4. **CHC Solving** - Active (53/55 solved)
5. **Advanced** - Planned (Strings, FP, Quantifiers)

## Integrations

- **kani_fast** - Primary SMT/CHC backend
- **lean5** - Proof certificate verification
- **tla2** - TLA+ model checking backend
- **tRust** - Verified Rust compilation
