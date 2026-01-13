# Implementation Phases - Detailed Reference

This file contains detailed implementation information for each phase. For a summary, see [CLAUDE.md](../CLAUDE.md).

---

## Phase 1: Foundation (COMPLETE)

- [x] Workspace setup with all crate stubs
- [x] Wrap `cargo kani` execution (KaniWrapper)
- [x] Port counterexample parsing from DashProve
- [x] CLI with `kani-fast verify` command
- [x] Integration tests with example Kani projects
- [x] Basic benchmarks

## Phase 2: Portfolio Solving (COMPLETE)

- [x] Solver trait abstraction (SolverResult, SolverConfig, SolverStats)
- [x] CaDiCaL wrapper with auto-detection
- [x] Parallel portfolio executor with cancellation
- [x] Kissat SAT solver wrapper with auto-detection
- [x] Z3 SMT solver wrapper with auto-detection (SAT + SMT-LIB2)
- [x] Adaptive strategy selection (start fast solvers, add more after delay)
- [x] Portfolio Kani verification (run multiple solvers in parallel via CLI)

## Phase 3: K-Induction (COMPLETE)

- [x] K-induction engine with base case and induction step checking
- [x] SMT-LIB2 formula generation for both checks
- [x] Simple path constraint for induction strengthening
- [x] Loop bound analysis (counter loops, bounded iteration patterns)
- [x] Template-based invariant synthesis (12 standard templates)
- [x] ICE-style learning from counterexample models
- [x] Integration with portfolio solver

## Phase 4: Incremental BMC (COMPLETE)

- [x] Clause database with persistence (SQLite + BLAKE3 hashing)
- [x] Diff analysis for code changes (function-level tracking)
- [x] Content-addressable storage for clauses
- [x] Verification result caching
- [x] Incremental BMC engine skeleton
- [x] Integration with SAT solver learned clause export (DRAT proof parsing)
- [x] Watch mode for continuous verification (debounced file watching)

## Phase 5: CHC Engine (COMPLETE)

- [x] CHC clause representation (Horn clauses, predicates, variables)
- [x] CHC system builder with init/transition/property clauses
- [x] SMT-LIB2 HORN logic output generation
- [x] Z3 Spacer integration for CHC solving
- [x] Invariant model extraction from Z3 output
- [x] Transition system to CHC encoding
- [x] Rust MIR to CHC encoding (parser + end-to-end verification)
- [x] Invariant-to-Lean5 translation (kani-fast-lean5 crate)
- [x] CLI integration with --lean5 flag for proof obligation generation
- [x] Z4 CHC/PDR integration (Phase 18: works for basic CHC, Z3 fallback for complex cases)

## Phase 6: AI Invariants (COMPLETE)

- [x] ICE learning framework (kani-fast-ai crate)
- [x] LLM integration for invariant suggestions (OpenAI, Anthropic, Ollama)
- [x] Invariant corpus from successful proofs (SQLite + BLAKE3)
- [x] Generate Lean5 proof sketches from learned invariants
- [x] CLI integration with --ai flag (--ai, --ai-only, --ai-max-attempts)

## Phase 7: Lean5 Proof Generation (COMPLETE)

- [x] Lean5 backend integration (invoke Lean compiler for proof checking)
- [x] Tactic generation module (auto-generate proof tactics for simple cases)
- [x] Proof certificate generation (JSON metadata + Lean file)
- [x] CLI: `lean5-check` command for verifying Lean files
- [x] CLI: `--certificate` flag for generating proof certificates
- [x] CLI: `--verify-lean` flag for verifying generated proofs with Lean
- [x] K-induction proofs generate certificates with --lean5/--certificate flags
- [x] Simple proofs (trivial arithmetic) now complete without sorry
- [x] Pattern-based tactic selection for common invariant forms
- [x] Multi-variable arithmetic pattern detection and tactic generation
- [x] Nested conjunction handling for k-induction consecution proofs
- [x] simp_arith and linarith tactic support for complex linear arithmetic
- [x] Non-linear arithmetic support (nlinarith, ring_nf, polyrith tactics)
- [x] Modular arithmetic support (mod_cast, norm_cast tactics)
- [x] Bitvector operation support (bv_decide, bv_omega tactics)

## Phase 8: Beautiful Counterexamples (COMPLETE)

- [x] Counterexample minimization via delta debugging
- [x] Natural language explanations with failure categorization
- [x] Repair suggestions with code snippets and confidence scores
- [x] CLI integration with --explain flag
- [x] Integration with Kani harness verification (tested with real harnesses)

## Phase 9: DashProve Integration (COMPLETE)

- [x] Add `BackendId::KaniFast` to DashProve (traits.rs, backend_ids.rs)
- [x] Implement `VerificationBackend` trait (kanifast backend module)
- [x] Add KaniFast to all backend match statements (AI, dispatcher, knowledge, client)
- [x] Integration tests for kanifast backend (11 tests passing)
- [ ] USL contract compilation (future: requires contract→harness translation)

## Phase 10: tRust Library API (COMPLETE)

- [x] `VerificationEngine` - Main engine with automatic escalation pipeline
- [x] `MirInput` - Input format with MIR program + specs + source info
- [x] `VerificationConfig` - Configurable timeouts, strictness, feature flags
- [x] `EngineResult` - Proven/Disproven/Unknown with proof certificates
- [x] `IncrementalState` - Hash-based change detection for caching
- [x] `FunctionSpecs` - requires/ensures/invariant clauses
- [x] Re-exported MIR types from kani-fast-chc for tRust consumption
- [x] Integration tests for the library API

## Phase 11: Kani Source Fork (COMPLETE)

**Motivation:** tRust needs full Rust support, not just simple functions.

- [x] Copy Kani's `kani_middle/` - battle-tested MIR handling (15K lines)
- [x] Copy `kani_queries/`, `intrinsics.rs` (200+ intrinsics)
- [x] Create `codegen_chc/` - replace CBMC with CHC/SMT backend
- [x] Wire kani_middle analysis → CHC generation
- [x] End-to-end: real Rust code → native CHC verification (kani-fast-driver)

**Testing:** `cd crates/kani-fast-compiler && ./test_driver.sh`

## Phase 12: tRust Performance Requirements (COMPLETE)

**Source:** tRust feature request `TRUST_FEATURE_REQUEST.md`

**Critical (blocks tRust adoption):**

- [x] **10x Faster**: Simple function <1s, loop bound 100 <30s (#143: achieved <100ms for all)
- [x] **Incremental Analysis**: Only re-verify changed functions + dependents (#145: achieved <100ms)
  - ChcVerificationCache: SQLite-based function-level result caching
  - Cache key: function name + source hash (invalidates on source change)
  - Dependency tracking for transitive invalidation
  - CLI: `--cache`, `--cache-path`, `--no-cache`, `--cache-stats` flags

**High Priority:**

- [x] **Z4 Handoff Protocol**: Accept partial proofs from Z4, focus on hard subproblems (#147)
  - `Z4Result` enum with `Proven`, `Counterexample`, `Unknown` variants
  - `Z4UnknownReason`: Timeout, MemoryLimit, QuantifierInstantiation, HeapAliasing, etc.
  - `Z4Subproblem`: Predicate + context for focused re-verification
  - `Z4PartialProof`: Partial invariants + certificate from upstream solver
  - `apply_z4_handoff()`: Integrates Z4 partial proofs into k-induction/CHC pipeline
  - `MirInput.with_z4_result()`: Attach Z4 handoff to verification input
- [x] **Parallel Exploration**: Function-level parallelism (#146)
  - CLI: `--parallel N` or `-j N` for `chc --all-proofs` command
  - Default: uses all CPU cores when --parallel is specified without value
  - Uses async streams with `buffer_unordered` for concurrent Z3 invocations
  - Thread-safe atomic counters for result aggregation
  - Speedup depends on proof complexity (Z3 startup time dominates simple proofs)

**Medium Priority:**

- [x] **Bounds Inference**: Infer unwind bounds from loop variants + preconditions (#149)
  - `FunctionSpecs::new().require("(< n 100)").variant("n - i")` → suggested_unwind = 100
  - Supports SMT-LIB and infix precondition formats
  - Per-loop bounds for nested loops via `variant_for_loop(loop_id, expr)`
  - `BoundsInference` engine with configurable default/max bounds
- [x] **Result Classification**: Clear reporting of what was proven (#148)
  - `KaniResult` enum with Verified/Counterexample/BoundedCheck/ResourceExhausted
  - `Coverage` enum for bounded check coverage info
  - Auto-generated Rust test cases from counterexamples
  - `EngineResult::to_kani_result()` conversion method

## Phase 13: Shared Proof Format (COMPLETE)

**Source:** tRust backend requirements (derived from DashProve integration needs)

All backends (Z4, Lean5, TLA2, Kani Fast) must emit proofs in shared format:

- [x] Universal proof representation (`kani-fast-proof` crate)
  - `UniversalProof`: Core proof type with VC, steps, dependencies
  - `ProofStep`: DRAT, SMT (Alethe-style), CHC, Lean proof steps
  - `BackendId`: KaniFast, Z3, Z4, Lean5, TLA2, CaDiCaL, Kissat
  - `ProofFormat`: DRAT, SMT, CHC, Lean, Mixed
- [x] Content-addressable: BLAKE3 hash of VC + proof + backend version
  - `ContentHash`: 32-byte BLAKE3 hash with hex encoding
  - `ContentAddressable` trait for proof types
  - Hash includes: VC, backend, format, backend_version, steps
- [x] Composable: dependency DAG for combining proofs
  - `UniversalProof.dependencies: Vec<ContentHash>`
  - Trust steps for referencing external proofs
  - Proof integrity verification via `verify_integrity()`
- [x] Proof storage with O(1) lookup
  - `MemoryStorage`: In-memory cache
  - `FileStorage`: Persistent with 2-char sharding
  - `LayeredStorage`: Cache promotion pattern
- [x] Integration with existing CHC and K-induction outputs
  - `generate_chc_proof()` in kani-fast-chc::proof
  - `generate_kinduction_proof()` in kani-fast-kinduction::proof
  - CLI: `--proof` and `--proof-output` flags for CHC and kinduction commands
- [x] Fast checking: O(proof size) verification for CHC proofs
  - `ProofChecker`: Z3-based verification of CHC proof steps (#157)
  - `UniversalProof::verify()`: Convenience method for proof verification
  - Verifies initiation, consecution, and property steps via SMT queries
  - 20 checker tests covering valid/invalid proofs

## Phase 14: Path Condition Tracking (COMPLETE)

**Source:** tRust feedback on false positives (N=55, 56, 59)

Problem: Calls inside conditionals or loops may report false positives when:
- Loop invariant not included in call-site precondition antecedent
- Branch conditions not tracked for guarded calls (`if x > 0 { f(x) }`)

Implemented fixes:

- [x] **Path condition collection**: DFS backwards from call site via `collect_path_conditions()` in mir.rs
- [x] **Loop invariant in call-site checks**: Invariant included as antecedent in call clause building
- [x] **CondGoto semantics audit**: Verified `then_target` = exit (guard failed), `else_target` = body
- [x] **Guarded call tests**: `tests/compiler/guarded_call_proofs.rs` - 4 tests all passing

See: `reports/main/MANAGER_DIRECTIVE_PATH_CONDITIONS_2025-12-31.md`

## Phase 15: Testing Infrastructure (COMPLETE)

**Source:** Lean5 practices + soundness bug retrospective

The struct soundness bug (#173) survived 50+ commits. Testing improvements:

- [x] **Differential testing**: `crates/kani-fast-chc/tests/differential.rs`
  - 3 tests: safe_add, buggy_multiply, overflow_behavior_documented
  - Tests skip gracefully if kani-fast-driver not built
- [x] **Property-based testing**: `crates/kani-fast-chc/tests/proptest_soundness.rs`
  - 7 proptest tests for structural invariants
- [x] **Soundness test suite**: `tests/soundness/` and `tests/compiler/`
  - End-to-end tests in `crates/kani-fast-compiler/test_driver.sh` (672 tests)
- [x] **Mutation testing**: `cargo mutants` on mir.rs (#182, #216)
  - mir.rs: 191 mutants, 91.1% score (174 caught, 2 missed, 12 unviable, 3 timeout)
- [x] **Kani self-verification**: 28 harnesses in kani-fast-chc (#192)
  - Run with: `cargo kani -p kani-fast-chc`

## Phase 16: Bitwise Strategy (COMPLETE)

**Status:** TEMPORARY - Superseded by Phase 17 Z4 BitVec integration

- [x] `algebraic_rewrite.rs` - `x & 0xFF` → `x mod 256`, `x << n` → `x * 2^n` (#189)
- [x] `proof_relevance.rs` - Only apply rewrites when bitwise affects proof (#189)
- [x] `delegation.rs` - Complex cases → `cargo kani` (#189)
- [x] Integration in `mir.rs` - Strategy selection (#190)

## Phase 17: Z4 Integration (COMPLETE)

Native QF_BV (bitvector) encoding with auto-detection:

- [x] Add BitVec encoding option to mir.rs (`--bitvec` flag, `KANI_FAST_BITVEC` env)
- [x] Encode i32 as `(_ BitVec 32)` when bitwise ops detected
- [x] Use `bvand`, `bvor`, `bvxor`, `bvshl`, `bvlshr` instead of uninterpreted functions
- [x] Auto-detection via `program_needs_bitvec_encoding()` in mir.rs and codegen_chc
- [x] CLI flags: `--bitvec`, `--bitvec-width` in kani-fast-cli (#235)
- [x] Convert 3 "expected failure" tests to actual PASS (#236)

**Result:** All 672 end-to-end tests pass

## Phase 18: Z4 CHC Integration (COMPLETE - partial)

- [x] Z4 HORN logic via stdin: `echo "(set-logic HORN)..." | z4`
- [x] Z4 auto-detected as CHC backend (prefers Z4, falls back to Z3)
- [x] `ChcSolverConfig::with_backend(ChcBackend::Z4)` for explicit Z4 use
- [x] Z4-specific CLI arguments (no `-smt2 -in fp.engine=spacer` needed)
- [x] `KANI_FAST_CHC_BACKEND` env var for backend selection (#290)

**Known Z4 Limitations:**
- HORN logic (CHC) hangs on some problems - Z3 Spacer more mature
- Some bounded counter problems return `unknown` (Z3 returns `sat`)
- No detailed statistics output (unlike Z3 Spacer)

## Phase 19: Hybrid Mode (COMPLETE)

CHC verification excels at unbounded proofs but returns "unknown" for unsupported features. Hybrid mode falls back to Kani BMC.

- [x] `HybridResult` enum: ChcVerified, ChcViolated, BmcVerified, BmcViolated, Unknown
- [x] `HybridConfig`: chc_timeout, kani_timeout, kani_unwind
- [x] `hybrid_verify_with_chc_result()`: Takes CHC result, falls back to Kani if unknown
- [x] CLI: `--hybrid` flag enables BMC fallback
- [x] CLI: `--hybrid-kani-timeout` configures Kani timeout (default 60s)

**Usage:**
```bash
kani-fast chc myfile.rs --hybrid
kani-fast chc myfile.rs --hybrid --hybrid-kani-timeout 120
```
