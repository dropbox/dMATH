# Kani Fast + tRust Integration Plan

**Date**: 2025-12-29
**Status**: Architecture Specification

## Overview

Kani Fast is the verification engine library. tRust is the compiler that integrates it.

```
┌─────────────────────────────────────────────────────────────────────┐
│                           tRust                                     │
│                    (rustc fork - the compiler)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Parse → HIR → MIR → Borrow Check → ┌──────────────┐ → Codegen     │
│                                     │ VERIFICATION │                │
│                                     │    PASS      │                │
│                                     └──────┬───────┘                │
│                                            │                        │
└────────────────────────────────────────────┼────────────────────────┘
                                             │ calls as library
                                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Kani Fast                                    │
│                  (verification engine library)                      │
├─────────────────────────────────────────────────────────────────────┤
│  MIR input → BMC → K-ind → CHC → AI Synth → Lean5 Gen              │
│                         ↓                                           │
│                        Z4                                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Division of Responsibility

### tRust (Compiler) Owns

| Component | Description |
|-----------|-------------|
| **Spec Parsing** | `#[requires]`, `#[ensures]`, `#[invariant]` attribute parsing |
| **HIR/MIR Extensions** | Carry specifications through lowering |
| **Verification Pass** | Compiler pass that invokes Kani Fast |
| **Error Reporting** | Emit verification failures as compiler errors |
| **Incremental Compilation** | Track what needs re-verification |
| **IDE Integration** | rust-analyzer shows verification errors inline |
| **Trust Levels** | `#![trust_level(verified/assumed/audited)]` |
| **Escape Hatches** | `#[unsafe_unverified]`, `#[may_diverge]` |

### Kani Fast (Engine) Owns

| Component | Description |
|-----------|-------------|
| **BMC Engine** | Bounded model checking via CBMC/SAT |
| **K-Induction** | Unbounded verification for loops |
| **CHC Solver** | Constrained Horn Clauses via Z4 Spacer |
| **AI Synthesis** | LLM-assisted invariant discovery |
| **Lean5 Generation** | Synthesize proofs for complex properties |
| **Z4 Backend** | Unified SAT/SMT/CHC solving |
| **Counterexamples** | Structured counterexample extraction |
| **Clause Caching** | Learned clause persistence for incrementality |

## Integration API

### Kani Fast as a Library

```rust
// tRust calls Kani Fast like this:

use kani_fast::{
    VerificationEngine,
    MirInput,
    VerificationResult,
    VerificationConfig,
};

/// Called by tRust's verification pass
pub fn verify_function(
    mir: &rustc_middle::mir::Body,
    specs: &FunctionSpecs,
    config: &VerificationConfig,
) -> VerificationResult {
    let engine = VerificationEngine::new(config);

    // Convert rustc MIR to Kani Fast's input format
    let input = MirInput::from_rustc_mir(mir, specs);

    // Run verification pipeline
    engine.verify(input)
}
```

### Result Types

```rust
pub enum VerificationResult {
    /// Property proven for all inputs
    Proven {
        method: ProofMethod,  // BMC, KInduction, CHC, Lean5
        duration: Duration,
        proof_certificate: Option<ProofCertificate>,
    },

    /// Counterexample found
    Disproven {
        counterexample: StructuredCounterexample,
        duration: Duration,
    },

    /// Could not determine (timeout, resource limit)
    Unknown {
        reason: String,
        partial_result: Option<PartialResult>,
        suggested_action: SuggestedAction,
    },
}

pub enum ProofMethod {
    BoundedModelChecking { bound: u32 },
    KInduction { k: u32 },
    CHC { invariant: String },
    Lean5 { proof_term: String },
}

pub enum SuggestedAction {
    IncreaseTimeout,
    AddInvariant(String),      // AI-suggested invariant
    AddPrecondition(String),   // AI-suggested requires clause
    UseManualProof,            // Escalate to Lean5
}
```

### Incremental Verification

```rust
/// Kani Fast maintains clause database for incrementality
pub struct IncrementalState {
    /// Learned clauses from previous verifications
    clause_db: ClauseDatabase,

    /// Hash of verified functions (for change detection)
    verified_hashes: HashMap<DefId, u64>,
}

impl VerificationEngine {
    /// Create engine with previous state
    pub fn with_state(config: &Config, state: IncrementalState) -> Self;

    /// Check if function needs re-verification
    pub fn needs_verification(&self, mir: &MirInput) -> bool;

    /// Export state for persistence
    pub fn export_state(&self) -> IncrementalState;
}
```

## Verification Pipeline

### Escalation Chain

```
1. BMC (fastest, bounded)
   └─ Success → Proven(BMC)
   └─ Counterexample → Disproven
   └─ Needs unbounded → Continue

2. K-Induction (fast, unbounded for some)
   └─ Success → Proven(KInduction)
   └─ Counterexample → Disproven
   └─ Can't find k → Continue

3. CHC/Spacer (slower, finds invariants)
   └─ Success → Proven(CHC) + extracted invariant
   └─ Counterexample → Disproven
   └─ Timeout → Continue

4. AI Invariant Synthesis (LLM-assisted)
   └─ Suggest invariant → Retry CHC
   └─ No suggestion → Continue

5. Lean5 Proof Generation (slowest, most powerful)
   └─ Generate proof → Proven(Lean5)
   └─ Can't generate → Unknown(UseManualProof)
```

### Configurable Strategies

```rust
pub struct VerificationConfig {
    /// Maximum time per function
    pub timeout: Duration,

    /// BMC bound (if unbounded proof fails)
    pub bmc_bound: u32,

    /// Enable AI invariant synthesis
    pub ai_synthesis: bool,

    /// Enable Lean5 proof generation
    pub lean5_backend: bool,

    /// Parallelism for portfolio solving
    pub num_threads: usize,

    /// Strictness level
    pub strictness: Strictness,
}

pub enum Strictness {
    /// Fail compilation if any verification fails
    Strict,

    /// Warn but continue if verification times out
    BestEffort,

    /// Only verify functions with explicit specs
    OptIn,
}
```

## Error Integration

### Verification Errors as Compiler Errors

```
error[E0XXX]: verification failed: precondition may not hold
  --> src/lib.rs:42:5
   |
42 |     arr[idx]
   |     ^^^^^^^^ index may be out of bounds
   |
   = note: counterexample: idx = 100, arr.len() = 50
   = help: add `#[requires(idx < arr.len())]` or bounds check

error[E0XXX]: verification failed: postcondition not proven
  --> src/lib.rs:50:1
   |
50 | #[ensures(result > 0)]
   | ^^^^^^^^^^^^^^^^^^^^^^ could not prove result > 0
   |
   = note: verification method: k-induction (k=10)
   = help: consider adding loop invariant: `#[invariant(x > 0)]`
```

### JSON Output for AI

```json
{
  "type": "verification_error",
  "location": {"file": "src/lib.rs", "line": 42, "column": 5},
  "message": "index may be out of bounds",
  "counterexample": {
    "inputs": {"idx": 100, "arr_len": 50},
    "trace": [...]
  },
  "suggestions": [
    {"type": "add_requires", "clause": "idx < arr.len()"},
    {"type": "add_bounds_check", "code": "if idx >= arr.len() { return None; }"}
  ],
  "verification_method": "bmc",
  "duration_ms": 150
}
```

## File Locations

### Kani Fast

```
~/kani_fast/
├── crates/
│   ├── kani-fast/              # Public API (what tRust imports)
│   ├── kani-fast-core/         # Verification engine
│   ├── kani-fast-kinduction/   # K-induction
│   ├── kani-fast-chc/          # CHC solver (Z4 Spacer)
│   ├── kani-fast-ai/           # AI invariant synthesis
│   ├── kani-fast-lean5/        # Lean5 proof generation
│   └── kani-fast-counterexample/
└── docs/
    └── TRUST_INTEGRATION.md    # This file
```

### tRust

```
~/trust/
├── compiler/
│   └── rustc_verification/     # New: verification pass
├── backends/
│   └── kani_fast/              # Kani Fast integration
└── docs/
    └── KANI_FAST_INTEGRATION.md  # Mirror of this file
```

## Related Projects

| Project | Repo | Role |
|---------|------|------|
| **tRust** | github.com/dropbox/tRust | Compiler (uses Kani Fast) |
| **Kani Fast** | github.com/dropbox/dMATH/kani_fast | Verification engine |
| **Z4** | github.com/dropbox/z4 | SAT/SMT/CHC backend |
| **Lean5** | github.com/dropbox/lean5 | Theorem prover backend |
| **TLA2** | github.com/dropbox/tla2 | Temporal logic (Phase 6) |
| **DashProve** | github.com/dropbox/dashprove | Unified verification platform |

## Timeline Alignment

| tRust Phase | Kani Fast Phase | Integration Point |
|-------------|-----------------|-------------------|
| Phase 2: Core Verification | Phase 1-2: Foundation + Portfolio | Basic BMC integration |
| Phase 4: Termination/Loops | Phase 3-4: K-Induction + Incremental | Unbounded verification |
| Phase 4.3: Kani Integration | Phase 5: CHC Engine | Full Kani Fast as backend |
| Phase 8: Lean5 Integration | Phase 7: Lean5 Proof Gen | Proof synthesis |

## Success Criteria

### For Integration

1. **API Stability**: Kani Fast public API is stable for tRust consumption
2. **Performance**: Verification adds <1s to typical function compilation
3. **Incrementality**: Re-verification after small changes is <100ms
4. **Error Quality**: Counterexamples are actionable by AI

### For Vision

1. **Zero Annotations**: Most code verifies without explicit specs
2. **Compile = Verify**: `trust build` proves and compiles in one step
3. **AI Invisible**: AI writes Rust, never sees Z4/Lean5/CHC directly

---

**See also:**
- [tRust ROADMAP.md](https://github.com/dropbox/tRust/blob/main/ROADMAP.md)
- [Z4 Requirements](https://github.com/dropbox/z4/blob/main/docs/KANI_FAST_REQUIREMENTS.md)
