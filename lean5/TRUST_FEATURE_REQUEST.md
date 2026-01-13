# Lean 5 Feature Request from tRust

**From:** tRust (verified Rust compiler)
**To:** Lean 5 team (git@github.com:dropbox/dMATH/lean5.git)
**Date:** 2024-12-31

---

## What is tRust?

tRust is a Rust compiler that integrates formal verification. We generate verification conditions (VCs) from Rust code, then dispatch them to proof backends. Lean 5 is our escape hatch for proofs that SMT solvers can't handle automatically.

**Our thesis:** Proofs are context compression for AI coding at scale. Verified code becomes a black box with a trusted spec. AI loads specs, not implementations.

---

## Why Lean 5 Matters to Us

Z4 (our SMT solver) handles ~90% of VCs automatically. But some proofs need:
- Induction
- Complex invariants
- Custom lemmas
- Mathematical reasoning beyond decidable theories

That's where Lean 5 comes in. When Z4 says "unknown," we escalate to Lean 5.

**The challenge:** Most tRust users aren't theorem proving experts. Lean 5 needs to be as automatic as possible, with graceful fallback to interactive proving when necessary.

---

## Feature Requests (Priority Order)

### 1. Blast-Style Automation (CRITICAL)

**Current Lean 4 behavior:**
```lean
-- User must choose tactics
theorem vc_47 : ∀ x : Int, x > 0 → x + 1 > x := by
  intro x hx
  linarith  -- User must know linarith exists
```

**Requested behavior:**
```lean
theorem vc_47 : ∀ x : Int, x > 0 → x + 1 > x := by
  blast  -- Just prove it
-- Works for arithmetic, simple logic, basic properties
```

**Why:**
- tRust generates VCs automatically; tactics should be automatic too
- Users shouldn't need to know `linarith` vs `omega` vs `simp` vs `decide`
- AI agents can't guess the right tactic

**Scope of `blast`:**
- Linear arithmetic
- Boolean logic
- Simple quantifier instantiation
- Basic set/list properties
- Common patterns from VC generation (weakest precondition structure)

**Success metric:** 95% of tRust-generated VCs prove with just `blast`

---

### 2. Rust Code Extraction (HIGH)

**Current:** Lean 4 extracts to C

**Requested:** Lean 5 extracts to idiomatic Rust

```lean
def mergeSort (xs : List Int) : List Int := ...

-- Extract to:
fn merge_sort(xs: Vec<i32>) -> Vec<i32> { ... }
```

**Why:**
- Verified algorithms directly usable in tRust code
- No FFI overhead, no C intermediate
- Rust type system alignment (Option, Result, ownership)

**Requirements:**
- Handle Lean's dependent types → Rust types where possible
- Monadic code → Rust Result/Option idiomatically
- Generate `#[no_panic]` code where provable
- Clear error when extraction impossible (explain why)

---

### 3. VC-Specific Tactics (HIGH)

tRust VCs have predictable structure (weakest precondition calculus). Custom tactics can exploit this.

**Requested tactics:**
```lean
-- Discharge precondition obligation
wp_precondition

-- Prove postcondition holds given path condition
wp_postcondition

-- Prove loop invariant preserved
wp_invariant_preserved

-- Prove loop invariant holds initially
wp_invariant_init

-- Prove loop variant decreases
wp_variant_decreases

-- Apply function contract at call site
wp_call [function_name]
```

**Example:**
```lean
-- VC from tRust: loop invariant preserved
theorem loop_inv_preserved :
  ∀ i n sum,
    (i ≥ 0 ∧ i ≤ n ∧ sum ≥ 0) →  -- invariant holds
    i < n →                        -- loop continues
    (i + 1 ≥ 0 ∧ i + 1 ≤ n ∧ sum + i ≥ 0)  -- invariant preserved
:= by
  wp_invariant_preserved  -- Specialized tactic
```

---

### 4. Bidirectional Rust Interop (MEDIUM)

**Rust → Lean (prove hard thing at compile time):**
```rust
#[prove_with(lean)]
#[requires(n >= 0)]
#[ensures(result == fibonacci(n))]  // fibonacci defined in Lean
fn fib(n: u64) -> u64 {
    // implementation
}
```

tRust generates VC, sends to Lean 5, Lean proves it, tRust continues compilation.

**Lean → Rust (use verified Rust in proofs):**
```lean
-- Import Rust function's verified spec
import Rust.MyModule.verified_sort

-- Use in proof
theorem my_theorem : ... := by
  have h := verified_sort_spec input
  ...
```

**Why:** Seamless escalation. User doesn't care which backend proved it.

---

### 5. Fast Proof Checking (MEDIUM)

**Requirement:** Check proof certificates in milliseconds

**Why:**
- CI re-checks proofs without re-running prover
- Distributed verification (prove on powerful machine, check everywhere)
- Proof caching validation

**Target:** <10ms for typical tRust VC proof

---

## Integration Protocol

### Input: Verification Condition
```
VC from tRust:
- Name: "loop_invariant_preserved_line_47"
- Statement: ∀ i n, (0 ≤ i ∧ i ≤ n) → i < n → (0 ≤ i+1 ∧ i+1 ≤ n)
- Hints: ["arithmetic", "linear", "loop_invariant"]
- Context: [definitions of relevant functions]
```

### Output: Result
```
Success:
- Proof term (for certificate)
- Tactics used (for user education)
- Time taken

Failure:
- Stuck goals (what couldn't be proven)
- Suggested lemmas (what might help)
- Counterexample if found
```

### Escalation Protocol
```
1. tRust generates VC
2. Z4 attempts proof (fast path)
3. Z4 returns Unknown
4. tRust sends VC to Lean 5 with hints
5. Lean 5 attempts blast
6. If blast fails, return stuck goals to user
7. User provides lemmas/tactics
8. Lean 5 completes proof
9. Proof certificate returned to tRust
```

---

## Shared Proof Format

Lean 5 proofs should emit certificates in tRust's shared format:

```
ProofCertificate {
    vc_hash: Hash,
    backend: "lean5",
    backend_version: "5.x.y",
    proof: LeanProofTerm,  // Or serialized representation
    checkable: true,
}
```

This allows:
- Z4 and Lean 5 proofs in same cache
- Unified trust model
- Cross-backend proof composition

---

## Questions for Lean 5 Team

1. How aggressive can `blast` be without becoming slow? What's the timeout?
2. Rust extraction: which Lean constructs are problematic?
3. Can Lean 5 import Rust type definitions directly? Or manual translation?
4. Proof certificate format: Lean's native or something portable?
5. Interactive mode: can tRust invoke Lean 5 as library, or separate process?

---

## Contact

tRust issues: https://github.com/dropbox/tRust/issues
This request: `reports/main/feature_requests/LEAN5_FEATURE_REQUEST.md`
