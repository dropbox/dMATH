# TLA2 Project Plan

This document outlines the implementation phases for TLA2.

## Phase Overview

| Phase | Component | Goal | Status |
|-------|-----------|------|--------|
| 1 | tla-core | Parse TLA+ | ✓ Complete |
| 2 | tla-core | Evaluate expressions | ✓ Complete |
| 3 | tla-check | Basic model checking | ✓ Complete |
| 4 | tla-check | Parallel exploration | ✓ Complete |
| 5 | tla-smt | Z3 integration | ✓ Complete |
| 6 | tla-prove | Proof manager | ✓ Complete |
| 7 | tla-zenon | Tableau prover | ✓ Complete |
| 8 | tla-codegen | Rust generation | ✓ Complete |
| 9 | tla-lsp | Editor support | ✓ Complete |
| 10 | All | Production polish | ✓ Complete |

## Phase 1: Parser Foundation ✓ COMPLETE

**Goal**: Parse TLA+ specifications into AST

### Deliverables
- [x] Lexer with all TLA+ tokens (~100 token types)
- [x] Parser for module structure
- [x] Parser for operators and expressions
- [x] AST types with source spans
- [x] Pretty-printer for roundtrip testing
- [x] Module resolution (EXTENDS)

### Validation
- Parse all specs in https://github.com/tlaplus/Examples
- Roundtrip: `parse(pretty_print(parse(src))) == parse(src)`

### Key Design Decisions
- Use **logos** for lexer generation
- Use **rowan** for lossless syntax tree (IDE-friendly)
- String interning for identifiers

## Phase 2: Evaluation Engine ✓ COMPLETE

**Goal**: Evaluate TLA+ expressions

### Deliverables
- [x] Value types (Bool, Int, Set, Function, Sequence, Record, Tuple)
- [x] Built-in operator implementations (~150 operators)
- [x] Expression evaluator
- [x] Standard library modules (Naturals, Integers, Sequences, FiniteSets, TLC)

### Validation
- Unit tests for all operators
- Property tests for algebraic laws (e.g., set union associativity)
- Comparison with TLC evaluation on sample expressions

## Phase 3: Basic Model Checker ✓ COMPLETE

**Goal**: Single-threaded model checking

### Deliverables
- [x] State representation
- [x] State fingerprinting (must match TLC for validation)
- [x] BFS state exploration
- [x] Invariant checking
- [x] Counterexample trace generation
- [x] Configuration file parsing (.cfg)

### Validation
- Find same bugs as TLC on known buggy specs
- Same state counts as TLC
- Same counterexamples (modulo state ordering)

## Phase 4: Parallel Model Checker ✓ COMPLETE

**Goal**: Multi-threaded exploration

### Deliverables
- [x] Work-stealing state queue (crossbeam)
- [x] Lock-free seen set (DashMap)
- [x] Worker thread pool
- [x] Load balancing

### Validation
- Linear speedup up to core count on large specs
- Same results as single-threaded
- No race conditions (ThreadSanitizer clean)

## Phase 5: SMT Integration ✓ COMPLETE

**Goal**: Connect to Z3

### Deliverables
- [x] TLA+ to SMT-LIB translation
- [x] Z3 backend implementation
- [x] Timeout handling
- [x] Model extraction for counterexamples
- [x] Bounded quantifier support in translation

### Validation
- Prove simple obligations
- Compare with TLAPM SMT results on test suite

## Phase 6: Proof Manager ✓ COMPLETE

**Goal**: Basic proof support

### Deliverables
- [x] Proof syntax parsing (BY, SUFFICES, HAVE, TAKE, etc.)
- [x] Proof obligation extraction
- [x] Backend orchestration (SMT and Zenon backends)
- [x] Proof result caching with fingerprints

### Validation
- Prove theorems from TLAPM test suite

## Phase 7: Zenon Port ✓ COMPLETE

**Goal**: First-order tableau prover in Rust

### Deliverables
- [x] Tableau construction (formula.rs, proof.rs)
- [x] Alpha/beta rule application (rules.rs)
- [x] Gamma/delta rules for quantifiers
- [x] Closure detection
- [x] Backtracking with depth/node limits
- [x] Equality reasoning (reflexivity, symmetry, substitution placeholders)
- [ ] Proof certificate generation (future work)

### Validation
- Match Zenon on its test suite

### Fallback
If port too complex, use FFI to OCaml Zenon

## Phase 8: Code Generation ✓ COMPLETE

**Goal**: Generate Rust from TLA+ specs

### Deliverables
- [x] Type inference for code generation (tla-codegen/types.rs)
- [x] Rust code emitter (tla-codegen/emit.rs)
- [x] StateMachine trait implementation (tla-runtime)
- [x] Runtime types (TlaSet, TlaFunc, TlaRecord) with full operations
- [x] Full expression-to-code translation (quantifiers, CHOOSE, LET, CASE, etc.)
- [x] Init predicate -> initial states enumeration
- [x] Next action -> successor state computation
- [x] Kani harness generation
- [x] Property-based test generation (proptest)
- [x] End-to-end integration testing (tests/integration.rs, 44 tests)
- [x] End-to-end execution testing (generated code runs correctly)
- [x] CLI codegen command (tla codegen)
- [x] Runtime model checker helper (model_check, collect_states)

### Validation
- Generated code compiles ✓
- Generated code executes correctly ✓
- Kani verifies invariants
- Property tests pass

## Phase 9: LSP Support ✓ COMPLETE

**Goal**: Editor integration

### Deliverables
- [x] Language server (tower-lsp)
- [x] Diagnostics (errors, warnings)
- [x] Completion (keywords, stdlib modules, operators, local symbols)
- [x] Hover information
- [x] Document symbols
- [x] Go to definition
- [x] Find references

### Validation
- Works in VS Code, Neovim

## Phase 10: Production Polish ✓ COMPLETE

**Goal**: Release-ready quality

### Deliverables
- [x] Beautiful error messages (ariadne diagnostic rendering in tla-core)
- [x] Performance benchmarking (sequential vs parallel, scaling analysis)
- [x] Adaptive parallelism (automatic sequential/parallel selection based on spec characteristics)
- [x] Evaluation caching - Investigated, deferred (see notes below)
- [x] Documentation (README performance section, benchmark report)
- [x] Release builds (optimized profiles, strip symbols, RELEASE.md)
- [x] CI/CD pipeline

### Evaluation Caching Decision
Investigated in iteration #64. Not implemented because:
- Most expressions depend on state variables; each state has unique bindings
- Cache keys would require expensive (expression, environment) hashing
- Existing optimizations (short-circuit evaluation, structural sharing) sufficient
- Parallel model checker provides better scaling for large specs

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Fingerprint mismatch with TLC | Byte-level comparison, extensive testing |
| SMT translation errors | Formal translation validation |
| Performance regression | Continuous benchmarking |
| Memory explosion | Streaming state representation |

## Validation Checkpoints

1. **Parser**: All TLA+ Examples parse
2. **Evaluator**: All built-in operators match TLC
3. **Model checker**: Same state counts as TLC
4. **Prover**: TLAPM test suite passes
5. **Codegen**: Generated code verifies with Kani

## Future Applications

Novel applications of TLA2's fast model checking engine beyond traditional TLA+ verification.

### Connectivity Prover (Backlog)

**Status**: Proposed | **Issue**: [#2](https://github.com/dropbox/dMATH/tla2/issues/2) | **Design**: `docs/proposals/PROPOSAL_connectivity_prover_2026-01-06.md`

Compile-time proof system that mathematically verifies software integration completeness. Solves the problem of AI-generated code that passes unit tests but fails at runtime due to disconnected code paths.

**Key Features**:
- Call graph extraction from rustc/tRust MIR
- Auto-generated TLA+ models with service availability as state variables
- Reachability proofs for all public APIs
- Build fails if connectivity proof fails
- Service dependency analysis ("what works when MCP is down?")

**Why TLA2**: Requires a fast model checker that can explore all service availability combinations and prove reachability theorems. TLA2's speed makes this practical as a compile-time tool.

### tRust Integration (Backlog)

Integration with the tRust proven Rust compiler project. TLA2 would provide:
- Connectivity proving as a compilation phase
- Formal verification of compiler transformations
- Model checking of generated code properties

See: https://github.com/dropbox/tRust
