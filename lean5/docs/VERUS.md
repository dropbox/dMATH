# Verus Setup for Lean5 Formal Verification

**Status**: Phase 6 (Bootstrap Verification) - In Progress
**Last Updated**: 2025-12-30 (Iteration 43)

## Overview

Lean5 uses Verus for deductive verification of the kernel implementation during the bootstrap phase. This is temporary scaffolding - eventually Lean5 will verify itself (Phase 7).

## Installation

Verus is installed locally in `tools/verus-arm64-macos/`.

### Prerequisites

1. Rust toolchain 1.91.0 (automatically installed by Verus)
2. macOS arm64 (other platforms: download appropriate binary)

### Installation Steps (Already Done)

```bash
# Download Verus binary release
mkdir -p tools && cd tools
curl -LO "https://github.com/verus-lang/verus/releases/download/release/0.2025.12.23.ab8296c/verus-0.2025.12.23.ab8296c-arm64-macos.zip"
unzip verus-0.2025.12.23.ab8296c-arm64-macos.zip

# Install required Rust toolchain
rustup install 1.91.0-aarch64-apple-darwin

# Verify installation
./verus-arm64-macos/verus --version
```

### Running Verus

```bash
# From lean5 root
./tools/verus-arm64-macos/verus verus-proofs/level_spec.rs
```

## Proof Structure

Proofs are organized in `verus-proofs/`:

```
verus-proofs/
├── test_setup.rs      # Basic verification test (3 proofs)
├── level_spec.rs      # Level type specification and proofs (17 proofs)
└── expr_spec.rs       # Expression type specification and proofs (81 proofs)
```

### level_spec.rs

Specification model of `lean5-kernel`'s `Level` type with proofs of:

| Property | Corresponds to proptest |
|----------|------------------------|
| `lemma_level_def_eq_reflexive` | `prop_level_def_eq_reflexive` |
| `lemma_level_def_eq_symmetric` | `prop_level_def_eq_symmetric` |
| `lemma_max_zero_left_identity` | `prop_level_max_zero_left_identity` |
| `lemma_max_idempotent` | - |
| `lemma_imax_zero_right` | `prop_level_imax_zero_right` |
| `lemma_level_geq_reflexive` | `prop_level_geq_reflexive` |
| `lemma_succ_geq` | `prop_level_succ_geq` |
| `lemma_geq_zero` | - |

**Verification results**: 17 verified, 0 errors

### expr_spec.rs

Specification model of `lean5-kernel`'s `Expr` type with proofs of:

| Category | Properties |
|----------|-----------|
| Expression equality | reflexive, symmetric |
| De Bruijn operations | lift(0) identity, lift_at specification |
| Substitution | instantiate_bvar_zero, instantiate_closed (sketch) |
| Sort typing | sort_has_type, prop_is_sort, is_prop_correct |
| Application | get_app_fn idempotent, get_app_fn non-app |
| WHNF simple | sort/bvar/fvar/lam/pi are WHNF, in_whnf with environment |
| Type inference | sort inference succeeds, Pi/Lam/App soundness |
| Determinism | infer deterministic for sort |
| Beta reduction | deterministic |
| Def eq | reflexive, symmetric, transitivity (atomic, Sort, composite) |
| Typing rules | sort, pi, lambda, application (specifications) |
| Environment | constant lookup, empty env, get_type, unfold |
| Local context | variable lookup, bound-variable lookup, empty ctx, push, push_let |
| infer_type soundness | Sort, FVar, Const, Lit, BVar, Pi, Lam, App cases |
| WHNF termination | expr_size measure, beta decreases body |
| Type preservation | axiom specification |
| Pi typing | well-formed Pi, Pi into Prop, Pi is type |
| Lambda typing | well-typed with Pi type, has_pi_type |
| App typing | well-typed with instantiated result, type from Pi |
| has_type | Full specification for Sort, BVar, FVar, Const, Lit, Pi, Lam, App, Let |
| is_type | Direct predicate for Sort, Pi, Const, FVar, BVar |
| WHNF fuel-based | whnf_fuel spec, whnf result type, termination proofs |
| WHNF properties | zero fuel timeout, atom/lam/pi immediate, deterministic, fuel monotonic |
| Well-foundedness | const_count, env_ordered, env_well_founded predicates |
| Sufficient fuel | termination measure, sufficient_fuel predicate, whnf_terminates_well_typed |

**Verification results**: 81 verified, 0 errors

**Total verification results**: 101 verified, 0 errors

## Verification Strategy

### Phase 6: Bootstrap Verification (Current)

Use Verus to verify key kernel functions:

1. **Level operations** (complete - 17 proofs)
   - [x] is_zero, is_nonzero
   - [x] normalize
   - [x] is_def_eq (reflexivity, symmetry)
   - [x] is_geq (reflexivity, succ property)
   - [x] make_max, make_imax

2. **Expression operations** (complete - 81 proofs)
   - [x] Expr structural equality (reflexive, symmetric)
   - [x] De Bruijn lift(0) identity
   - [x] Instantiate basic properties
   - [x] is_sort, is_prop predicates
   - [x] get_app_fn properties
   - [x] WHNF simple predicates
   - [x] in_whnf with environment model
   - [x] expr_size termination measure
   - [x] WHNF fuel-based termination (whnf_fuel spec, termination proofs)
   - [x] Well-foundedness predicates (const_count, env_well_founded)
   - [x] Sufficient fuel theorem (whnf_terminates_well_typed)

3. **Type checker** (complete - 67 proofs)
   - [x] Sort typing rule specification
   - [x] Pi, Lambda, App typing rule specifications
   - [x] is_def_eq_simple (reflexive, symmetric)
   - [x] is_def_eq transitivity (atomic, Sort, composite)
   - [x] Beta reduction determinism
   - [x] infer_type deterministic for Sort
   - [x] infer_type soundness for Sort, FVar, Const, Lit
   - [x] infer_type soundness for BVar
   - [x] infer_type soundness for Pi (well-formed, into Prop)
   - [x] infer_type soundness for Lambda
   - [x] infer_type soundness for App
   - [x] Type preservation axiom specification
   - [x] has_type full specification (Sort, BVar, FVar, Const, Lit, Pi, Lam, App, Let)
   - [x] is_type direct predicate
   - [x] Full is_def_eq transitivity for compound expressions

4. **Environment** (complete - concrete model)
   - [x] Env struct with Seq<ConstantInfo>
   - [x] LocalCtx struct with Seq<LocalDecl>
   - [x] env_get_const, env_get_type, env_unfold
   - [x] ctx_get, ctx_get_type, ctx_push, ctx_push_let
   - [x] empty_env, empty_ctx
   - [x] Proofs: lookup in empty returns None
   - [x] Proofs: push makes variable retrievable

### Phase 7: Self-Verification (Future)

Replace Verus proofs with Lean5 self-verification:
- Formalize Rust semantics in Lean5
- Re-prove kernel correctness using Lean5
- Delete Verus dependency

## Writing Verus Proofs

### Basic Pattern

```rust
use vstd::prelude::*;

verus! {

// Specification function (ghost code)
pub open spec fn my_spec(x: nat) -> bool {
    x > 0
}

// Proof (checked by Verus)
proof fn my_lemma(x: nat)
    requires x > 0
    ensures my_spec(x)
{
    // Proof body - assertions help Verus
}

// Executable function with verification
fn my_function(x: u64) -> (result: u64)
    requires x < u64::MAX
    ensures result == x + 1
{
    x + 1
}

} // verus!
```

### Key Concepts

1. **spec fn**: Ghost functions for specifications (not compiled)
2. **proof fn**: Lemmas checked by Verus SMT solver
3. **exec fn**: Regular functions with pre/postconditions
4. **requires/ensures**: Pre/postconditions
5. **decreases**: Termination measure for recursive functions

### Tips

1. Use `assert()` statements to help Verus find proofs
2. Break complex proofs into helper lemmas
3. Use `decreases` clauses for recursive spec functions
4. Check Verus guide: https://verus-lang.github.io/verus/guide/

## Resources

- [Verus GitHub](https://github.com/verus-lang/verus)
- [Verus Guide](https://verus-lang.github.io/verus/guide/)
- [Verus Zulip](https://verus-lang.zulipchat.com/)
- [vstd API](https://verus-lang.github.io/verus/verusdoc/vstd/)
