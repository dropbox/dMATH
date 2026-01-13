# Archive

This directory contains archived code that is no longer actively used but is preserved for historical reference.

## verus-proofs/

Original Verus proofs for Lean5 kernel correctness. These proofs were developed in iterations #43-#48 and subsequently converted to Lean5 self-verification in iterations #49-#52.

**Status**: Superseded by `crates/lean5-verify/` which contains the same proofs formalized in Lean5's own type theory.

**Files**:
- `expr_spec.rs` - Expression type specifications and proofs (WHNF, beta reduction)
- `level_spec.rs` - Universe level specifications and proofs
- `test_setup.rs` - Verus test setup

**Why archived**: Lean5 is now self-verifying. The lean5-verify crate contains:
- All Verus lemmas converted to Lean5 specifications
- Proof witnesses that type-check in the kernel
- Cross-validation between specification and implementation
- Micro-checker formalization and soundness proofs

The Verus binary tools (`tools/verus-arm64-macos/`) were deleted to reduce repository size (~80MB).
