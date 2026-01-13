# Z4 Feature Request from tRust

**From:** tRust (verified Rust compiler)
**To:** Z4 team (git@github.com:dropbox/dMATH/z4.git)
**Date:** 2024-12-31

---

## What is tRust?

tRust is a Rust compiler that integrates formal verification. We generate verification conditions (VCs) from Rust code with specifications, then dispatch them to proof backends. Z4 is our primary backend—the workhorse that handles ~90% of VCs.

**Our thesis:** Proofs are context compression for AI coding at scale. When code is proven correct, AI loads the spec (3 lines) instead of the implementation (300 lines). Math remembers, so AI doesn't have to.

---

## Why Z4 Matters to Us

Z4 is the fast path. Most verification conditions are:
- Arithmetic bounds checking
- Array index validity
- Simple logical implications
- Pre/postcondition checking

These should verify in milliseconds. Z4's speed determines whether verification feels instant (good) or annoying (adoption killer).

**Current state:** We use z3 Rust bindings. It works. But we restart the solver for each VC, counterexamples are ugly, and there's no proof certificate.

---

## Feature Requests (Priority Order)

### 1. Incremental Solving (CRITICAL)

**Current behavior:**
```rust
for vc in verification_conditions {
    let solver = Z4::new();  // Fresh solver each time
    solver.assert(vc);
    solver.check();          // No shared learning
}
// 1000 VCs = 1000 solver startups
```

**Requested behavior:**
```rust
let mut solver = Z4::incremental();
for vc in verification_conditions {
    solver.push();
    solver.assert(vc);
    solver.check();  // Reuses learned lemmas from previous VCs
    solver.pop();
}
// 1000 VCs = 1 solver, shared context
```

**Why:**
- 10-100x speedup on large codebases
- VCs in same function share context (same variables, similar structure)
- Learned lemmas from VC #1 often help VC #2

**Acceptance criteria:**
- `push()`/`pop()` for context management
- Learned clauses persist across push/pop
- No correctness regression (incremental = non-incremental results)

---

### 2. Minimal Counterexamples (HIGH)

**Current behavior:**
```
Counterexample: x = 847293847, y = -12847392, z = 99281
```

**Requested behavior:**
```
Counterexample: x = 0
(y and z unconstrained, any value works)
```

**Why:**
- Users need to understand why proof failed
- `x = 0` is actionable; `x = 847293847` is noise
- AI agents iterate on counterexamples—simpler = faster convergence

**Minimization heuristics:**
1. Prefer 0, 1, -1, MIN, MAX over arbitrary values
2. Minimize number of constrained variables
3. For arrays, prefer shortest length that triggers failure

**API sketch:**
```rust
enum CounterexampleStyle {
    Any,           // Current behavior (fast)
    Minimal,       // Minimize values (slower)
    Readable,      // Prefer round numbers (slowest)
}

solver.set_counterexample_style(CounterexampleStyle::Minimal);
```

---

### 3. Proof Certificates (HIGH)

**Current behavior:**
```
Result: Valid
// Trust me bro
```

**Requested behavior:**
```
Result: Valid
Certificate: ProofCertificate {
    steps: [...],
    checksum: 0x8f3a2b...,
    verifiable_by: independent_checker
}
```

**Why:**
- Verify the verifier (catch Z4 bugs)
- Cache proofs across runs (don't re-prove unchanged code)
- Distributed proof checking (verify on different machine than generated)
- Audit trail ("this binary was verified, here's the proof")

**Requirements:**
- Certificate checkable in O(proof_size) time
- Independent checker (not Z4 itself)
- Deterministic (same VC + same Z4 version = same certificate)

---

### 4. Native Rust API (MEDIUM)

**Current:** C FFI to libz3, unsafe blocks, awkward lifetime management

**Requested:** Pure Rust, idiomatic API

```rust
// Current (awkward)
let ctx = z3::Context::new(&z3::Config::new());
let solver = z3::Solver::new(&ctx);
let x = z3::ast::Int::new_const(&ctx, "x");
// Everything tied to ctx lifetime, lots of &

// Requested (clean)
let mut solver = z4::Solver::new();
let x = solver.int_var("x");
let y = solver.int_var("y");
solver.assert(x.gt(0));
solver.assert(y.eq(x.add(1)));
// Normal Rust ownership, no lifetime gymnastics
```

**Why:**
- tRust is Rust-native; backend should match
- Easier to maintain, fewer unsafe blocks
- Better error messages (Rust compiler helps)

---

### 5. Custom Theory Plugins (MEDIUM)

**Want:** Extend Z4 with Rust-specific theories

```rust
// Rust ownership theory
solver.register_theory(RustOwnership::new());

// Now Z4 understands:
// - &mut T is exclusive
// - &T is shared
// - Moving invalidates source
```

**Use cases:**
- Slice bounds (`arr[i]` valid iff `i < arr.len()`)
- Option semantics (`x.unwrap()` valid iff `x.is_some()`)
- Reference validity

**Why:** Rust semantics aren't in SMT-LIB. Either we encode them (verbose, slow) or Z4 understands them natively (clean, fast).

---

## Performance Targets

| Metric | Current Z3 | Z4 Target |
|--------|------------|-----------|
| Simple VC (arithmetic) | 5-10ms | <1ms |
| Medium VC (quantifiers) | 50-200ms | <20ms |
| Incremental overhead | N/A | <0.1ms per push/pop |
| Counterexample minimization | N/A | <10ms additional |
| Proof certificate generation | N/A | <5ms additional |

---

## Integration Protocol

### Input: Verification Condition
```rust
struct VC {
    id: VcId,
    name: String,
    condition: Predicate,  // What to prove
    context: Vec<Predicate>,  // Assumptions

    // Hints
    theories_needed: Vec<Theory>,  // LIA, LRA, BV, Arrays, UF
    estimated_difficulty: Difficulty,
}
```

### Output: Result
```rust
enum Z4Result {
    Valid { certificate: Option<ProofCertificate> },
    Invalid { counterexample: Counterexample },
    Unknown { reason: String, timeout: bool },
}

struct Counterexample {
    assignments: Vec<(String, Value)>,
    minimized: bool,
}
```

---

## Questions for Z4 Team

1. Is incremental solving on your roadmap? What's the timeline?
2. What's the overhead of proof certificate generation? Acceptable for all VCs or opt-in?
3. Custom theories: plugin API or compile-time registration?
4. Are you tracking Z3 upstream or diverging? (Affects our expectations)

---

## Contact

tRust issues: https://github.com/dropbox/tRust/issues
This request: `reports/main/feature_requests/Z4_FEATURE_REQUEST.md`

---

## Z4 Team Response (2025-12-31)

All feature requests have been reviewed and added to the roadmap.

**See:** `docs/ARCHIMEDES_ROADMAP.md` - "tRust Integration Priorities" section

### Summary

| Request | Status | Roadmap Position |
|---------|--------|------------------|
| Incremental Solving | Planned | CRITICAL PATH #4 |
| Minimal Counterexamples | Planned | CRITICAL PATH #5 |
| Proof Certificates | Planned | CRITICAL PATH #6 (z4-proof crate) |
| Native Rust API | COMPLETE | Already pure Rust |
| Custom Theory Plugins | Planned | Medium-term #11 |
| Floating Point Theory | Planned | Medium-term #10 (z4-fp crate) |

### Answers to Questions

1. **Incremental solving timeline:** On critical path, next major feature after current work.
2. **Proof certificate overhead:** Opt-in. DRAT exists for SAT, Alethe planned for SMT.
3. **Custom theories:** Plugin API (runtime registration via trait).
4. **Z3 tracking:** Diverging. Z4 is ground-up Rust implementation, not Z3 port.

### Performance Targets Accepted

Z4 commits to meeting the tRust performance targets:
- Simple VC: <1ms (vs Z3 5-10ms)
- Push/pop overhead: <0.1ms
- Counterexample minimization: <10ms additional
