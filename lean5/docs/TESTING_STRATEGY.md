# Lean5 Testing Strategy

## Overview

Lean5 uses a multi-layered testing approach to ensure kernel correctness. The goal is **zero doubt** - if Lean5 accepts a proof, it must be correct.

---

## Testing Layers

| Layer | Purpose | Tools |
|-------|---------|-------|
| Unit Tests | Catch specific bugs | `cargo test` |
| Property Tests | Verify invariants hold for all inputs | `proptest` |
| Differential Tests | Match Lean 4 behavior exactly | Custom harness |
| Mutation Tests | Prove tests are comprehensive | `cargo-mutants` |
| Cross-Validation | Independent verification | Micro-checker |

---

## Layer 1: Unit Tests

Standard Rust tests covering specific functionality.

```rust
#[test]
fn test_whnf_beta_reduction() {
    let env = Environment::new();
    let tc = TypeChecker::new(&env);
    
    // (λ x. x) y → y
    let id = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::bvar(0));
    let app = Expr::app(id, Expr::prop());
    
    assert_eq!(tc.whnf(&app), Expr::prop());
}
```

**Target**: Every public function has at least one test.

---

## Layer 2: Property-Based Tests

Using `proptest` to verify properties hold for randomly generated inputs.

```rust
proptest! {
    #[test]
    fn whnf_is_idempotent(e in expr_strategy()) {
        let tc = TypeChecker::new(&env);
        let once = tc.whnf(&e);
        let twice = tc.whnf(&once);
        assert_eq!(once, twice);
    }
    
    #[test]
    fn is_def_eq_is_reflexive(e in expr_strategy()) {
        let tc = TypeChecker::new(&env);
        assert!(tc.is_def_eq(&e, &e));
    }
}
```

**Target**: All kernel invariants have property tests.

---

## Layer 3: Differential Testing

Compare Lean5 output against Lean 4 for identical inputs.

```bash
# For each test expression
lean4_type=$(lean --print-type input.lean)
lean5_type=$(lean5 --print-type input.lean)

if [ "$lean4_type" != "$lean5_type" ]; then
    echo "MISMATCH on input.lean"
    exit 1
fi
```

**Target**: 1000+ expressions tested, zero mismatches.

**Current**: 1044 expressions passing.

---

## Layer 4: Mutation Testing

Mutation testing verifies that our tests actually catch bugs.

### How It Works

1. `cargo-mutants` creates "mutants" - small code changes (bugs)
2. Tests run against each mutant
3. If tests **fail** → mutant "killed" (good)
4. If tests **pass** → mutant "survives" (bad - tests missed a bug)

### Kill Rate Target: 100%

| Status | Meaning |
|--------|---------|
| 100% killed | Tests catch every possible single-point mutation |
| <100% killed | Test gaps exist |

### Survivor Types

**Type 1: Regular Survivors**
- Tests don't cover this code path
- **Fix**: Add targeted test

**Type 2: Equivalent Mutants**
- Mutation doesn't change behavior due to code structure
- **Fix**: Refactor code (see below)

---

## Equivalent Mutants = Code Smells

### The Problem

An "equivalent mutant" survives because the mutation doesn't change behavior:

```rust
// Original
if depth == target { return X }
if depth > target { return Y }

// Mutant: > changed to >=
if depth == target { return X }
if depth >= target { return Y }  // SURVIVES - same behavior!
```

The `>=` mutant survives because `==` is already handled. The code is **ambiguous**.

### The Solution: Refactor

Equivalent mutants are NOT acceptable. They indicate code that should be cleaner:

```rust
// REFACTORED - no equivalent mutants possible
match depth.cmp(&target) {
    Ordering::Equal => X,
    Ordering::Greater => Y,
    Ordering::Less => Z,
}
```

Now ANY mutation to the match arms will change behavior and be caught by tests.

### Rule

> **Zero equivalent mutants = zero ambiguous code paths**

For each equivalent mutant:
1. Find the code location
2. Ask: WHY is this mutation equivalent?
3. Refactor to eliminate ambiguity
4. Re-run mutation testing

---

## Layer 5: Cross-Validation (Micro-Checker)

The micro-checker is a minimal (~1000 line) independent type checker.

```rust
// In debug builds, every type inference is cross-validated
pub fn infer_type(&mut self, e: &Expr) -> Result<Expr, TypeError> {
    let ty = self.infer_type_impl(e)?;
    
    #[cfg(debug_assertions)]
    {
        let cert = self.generate_cert(e, &ty);
        let micro_result = MicroChecker::verify(&cert, e);
        assert_eq!(Ok(ty.clone()), micro_result);
    }
    
    Ok(ty)
}
```

If the main kernel and micro-checker disagree, something is wrong.

**Target**: 100% of inferences cross-validated in debug builds.

---

## Metrics Dashboard

| Metric | Current | Target |
|--------|---------|--------|
| Unit test count | 1100+ | Comprehensive |
| Property tests | 30+ | All invariants |
| Differential tests | 1044 | 1000+ ✓ |
| Mutation kill rate | 96%+ | 100% |
| Micro-checker coverage | 100% | 100% ✓ |

---

## Running Tests

```bash
# All tests
cargo test

# Property tests only
cargo test proptest

# Mutation testing
cargo mutants -p lean5-kernel --release

# Differential tests
./scripts/differential_test.sh
```

---

## Summary

| Survivor Type | Problem | Fix |
|---------------|---------|-----|
| Regular mutant | Missing test | Add test |
| Equivalent mutant | Ambiguous code | Refactor code |

Both are failures. 100% kill rate requires comprehensive tests AND clean code.
