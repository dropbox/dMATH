# Roadmap: kani_fast

> Fast Bounded Model Checking for Rust

## Current Focus

Float parsing fix and CHC integration improvements.

## Active Issues

| # | Priority | Title |
|---|----------|-------|
| #4 | P1 | Float MIR parsing: f32/f64 literals converted to 0 |
| #3 | P1 | Float literal parsing in MIR (duplicate of #4) |
| #9 | P2 | [RESEARCHER] Create system diagrams |
| #8 | P2 | Native compilation for verification conditions via tRust |
| #7 | P2 | TLA2 requirements from Kani Fast |
| #5 | P2 | Verification-Guided Optimization for Rust BMC |
| #1 | P3 | Failure diagnosis (minimal failing VCs) |

## Features

| Feature | Status |
|---------|--------|
| Bounded model checking | Complete |
| Incremental verification | Complete |
| k-induction | Complete |
| CHC solving | Complete |
| Portfolio solving | Complete |
| AI-assisted invariants | In Progress |
| Lean5 proof certificates | Planned |

## Test Status

- 3625 tests passing
- 81 compiler tests
- 672 e2e tests
- 32 documented soundness limitations (see docs/SOUNDNESS_LIMITATIONS.md)

## Dependencies

- **z4** - Primary SMT/CHC solver
- **lean5** - Proof certificate verification
- **Kani** - Upstream verifier (wrapped)
