# PLAN: Soundness Objections & Formal Verification Roadmap

**Status:** Phase 0 COMPLETE, Phase 1 COMPLETE
**Created:** 2026-01-02
**Updated:** 2026-01-02 (Worker #278)
**Goal:** Address all legitimate mathematical objections to γ-CROWN's soundness claims

---

## Phase 0: COMPLETE ✓

All 5 quick wins implemented in Worker #274:

| Item | Status | Implementation |
|------|--------|----------------|
| D1: Version pinning | ✓ | Cargo.lock now tracked in git (removed from .gitignore) |
| A5: NaN/Inf checks | ✓ | `debug_assert!` in `BoundedTensor::new()`, `concrete()`, `from_epsilon()` |
| C3: Edge case tests | ✓ | Tests for f32::MAX, denormals, negative zero in gamma-tensor |
| D4: Verify no fast-math | ✓ | Verified: no .cargo/config, no build.rs, no fast-math flags |
| A4: Denormal documentation | ✓ | "Numerical Soundness" section added to docs/DESIGN.md |

## Phase 1: COMPLETE ✓

Worker #275: A1 (directed rounding), Worker #276: C1 (proptest), Worker #277: B3, C2 fix, Worker #278: E2

| Item | Status | Implementation |
|------|--------|----------------|
| A1: Directed rounding | ✓ | `BoundedTensor::round_for_soundness()`, `Network::propagate_ibp_sound()` |
| B3: Activation validation | ✓ | `scripts/validate_activations_vs_pytorch.py` - 64/64 tests (7 activations × ~9 intervals) |
| C1: Property-based testing | ✓ | 19 proptest soundness tests in gamma-tensor and gamma-propagate |
| C2: Cross-reference testing | ✓ | `scripts/validate_vs_autolirpa.py` with `--center-zeros` flag - 6/6 models pass |
| E2: Formal perturbation spec | ✓ | docs/DESIGN.md "Formal Perturbation Specification" section |

Next priority: Phase 2 (Lean specification) or remaining optimizations from MASTER-CHECKLIST.md.

---

## The Objections (Comprehensive List)

### Category A: Floating Point Arithmetic (CRITICAL)

| ID | Objection | Severity | Effort |
|----|-----------|----------|--------|
| A1 | **No directed rounding** - Interval arithmetic requires rounding lower bounds DOWN, upper bounds UP. We use default rounding (nearest). | CRITICAL | Medium |
| A2 | **Floating point non-associativity** - `(a+b)+c ≠ a+(b+c)` in IEEE 754. Matrix multiply order affects results. | HIGH | Low |
| A3 | **Catastrophic cancellation** - Subtracting nearly-equal numbers loses precision. Common in normalization. | HIGH | Medium |
| A4 | **Denormalized numbers** - Performance cliffs, potential precision loss near zero. | MEDIUM | Low |
| A5 | **NaN/Inf propagation** - Silent corruption if NaN enters computation. No explicit checks. | HIGH | Low |
| A6 | **Overflow in intermediates** - Inputs/outputs fine, but intermediate values overflow. | MEDIUM | Low |
| A7 | **The ulp gap problem** - Floating point gaps grow near powers of 2. Bounds may miss by 1 ulp. | MEDIUM | Medium |

### Category B: Implementation vs Specification Gap (CRITICAL)

| ID | Objection | Severity | Effort |
|----|-----------|----------|--------|
| B1 | **No formal specification** - What exactly does "sound" mean? No machine-readable spec. | CRITICAL | High |
| B2 | **Paper-to-code gap** - CROWN papers prove math; code is implementation. Gap unanalyzed. | CRITICAL | Very High |
| B3 | **Activation function mismatch** - Our GELU/SiLU may differ from PyTorch's implementation. | HIGH | Medium |
| B4 | **Transformer variant mismatch** - Pre-norm vs post-norm, different attention implementations. | HIGH | Medium |
| B5 | **No certified compilation** - rustc/LLVM could miscompile. No CompCert-style guarantees. | CRITICAL | Very High |

### Category C: Testing Inadequacy (HIGH)

| ID | Objection | Severity | Effort |
|----|-----------|----------|--------|
| C1 | **Sampling proves nothing** - 11 points per interval doesn't cover infinite reals. | HIGH | Medium |
| C2 | **Circular testing** - We test our tanh against libm's tanh. Both could be wrong. | MEDIUM | Medium |
| C3 | **Missing edge cases** - Denormals, negative zero, max float, min float, subnormals. | HIGH | Low |
| C4 | **No adversarial testing** - Malicious model weights designed to break verifier. | MEDIUM | Medium |
| C5 | **Concurrency testing** - Race conditions may only manifest under specific interleavings. | HIGH | High |

### Category D: Build & Deployment (MEDIUM)

| ID | Objection | Severity | Effort |
|----|-----------|----------|--------|
| D1 | **No version pinning** - Dependencies could change behavior silently. | MEDIUM | Low |
| D2 | **No reproducible builds** - Different machines may produce different binaries. | MEDIUM | Medium |
| D3 | **Platform differences** - x86 vs ARM vs GPU have different FP behavior. | HIGH | High |
| D4 | **Compiler optimizations** - `-ffast-math` style opts could break IEEE compliance. | MEDIUM | Low |
| D5 | **No hardware fault tolerance** - Bit flips, cosmic rays, CPU bugs (Pentium FDIV). | LOW | Very High |

### Category E: Practical Utility (HIGH)

| ID | Objection | Severity | Effort |
|----|-----------|----------|--------|
| E1 | **Bounds too loose** - ±444,000 proves nothing useful. | CRITICAL | Very High |
| E2 | **No specification of what's being verified** - Robustness to what? Perturbation model unclear. | HIGH | Medium |
| E3 | **Scalability vs tightness tradeoff** - We scale to 1.5B params but bounds explode. | HIGH | Very High |

---

## Prioritized Roadmap

### Phase 0: Quick Wins (1-2 commits each)

| ID | Fix | Effort |
|----|-----|--------|
| D1 | **Version pinning** - `Cargo.lock` in repo, exact versions in `Cargo.toml` | 1 commit |
| A5 | **NaN/Inf checks** - Add `debug_assert!(!x.is_nan())` on all inputs/outputs | 1 commit |
| C3 | **Edge case tests** - Add tests for f32::MAX, f32::MIN, denormals, neg zero | 1 commit |
| D4 | **Disable fast-math** - Ensure no `-ffast-math` in build flags | 1 commit |
| A4 | **Denormal handling** - Document behavior, add flush-to-zero option | 1 commit |

### Phase 1: Practical Soundness (5-10 commits each)

| ID | Fix | Effort |
|----|-----|--------|
| A1 | **Directed rounding for critical paths** - Use `nextafter()` or add epsilon margin to bounds | 3 commits |
| B3 | **Activation function validation** - Test against PyTorch, document any differences | 2 commits |
| C1 | **Property-based testing** - Use proptest/quickcheck for randomized soundness checks | 3 commits |
| C2 | **Cross-reference testing** - Compare bounds against Auto-LiRPA Python reference | 2 commits |
| E2 | **Formal perturbation spec** - Document exactly what ε-ball means (L∞, L2, etc.) | 1 commit |

### Phase 2: Formal Specification (Research Project)

| ID | Fix | Effort |
|----|-----|--------|
| B1 | **Lean 5 specification** - Formal spec of interval arithmetic and CROWN algorithm | 20+ commits |
| B2 | **Lean 5 proof of core algorithm** - Prove CROWN math correct in Lean | 50+ commits |
| A1-A7 | **Verified floating point** - Prove interval arithmetic sound under IEEE 754 | 30+ commits |

### Phase 3: Certified Implementation (Major Project)

| ID | Fix | Effort |
|----|-----|--------|
| B5 | **tRust compilation** - Compile with formally verified Rust compiler | External dependency |
| D5 | **Hardware certification** - ECC memory, certified CPU, radiation hardening | External dependency |
| D3 | **Platform-specific proofs** - Separate verification per target architecture | 20+ commits each |

---

## Concrete Next Steps

### Immediate (Next 5 Worker Iterations)

```bash
# D1: Version pinning
git add Cargo.lock  # Already tracked? Verify exact versions

# A5: NaN checks - add to BoundedTensor::new()
debug_assert!(lower.iter().all(|x| !x.is_nan()), "NaN in lower bound");
debug_assert!(upper.iter().all(|x| !x.is_nan()), "NaN in upper bound");

# C3: Edge case tests
#[test]
fn test_ibp_f32_max() { ... }
fn test_ibp_denormal() { ... }
fn test_ibp_negative_zero() { ... }
```

### Short Term (Next 20 Worker Iterations)

1. **Directed rounding implementation**
   - Option 1: Add 1 ulp margin to all bounds (conservative but easy)
   - Option 2: Use `f32::next_up()` / `f32::next_down()` (Rust nightly)
   - Option 3: Implement proper interval arithmetic with rounding modes

2. **Property-based testing**
   ```rust
   proptest! {
       #[test]
       fn soundness_holds(l in -100.0f32..100.0, u in -100.0f32..100.0) {
           let (l, u) = (l.min(u), l.max(u));
           let bounds = gelu_ibp(l, u);
           for x in sample_interval(l, u, 1000) {
               prop_assert!(bounds.0 <= gelu(x) && gelu(x) <= bounds.1);
           }
       }
   }
   ```

3. **Auto-LiRPA cross-validation**
   - Run same model through both γ-CROWN and Auto-LiRPA
   - Bounds should be within floating point tolerance
   - Document any intentional differences

### Medium Term (Lean 5 Specification)

```lean
-- Formal specification of interval arithmetic
structure Interval where
  lo : Float
  hi : Float
  valid : lo ≤ hi

-- Soundness theorem for interval addition
theorem interval_add_sound (a b : Interval) (x y : Float) :
  a.lo ≤ x ∧ x ≤ a.hi →
  b.lo ≤ y ∧ y ≤ b.hi →
  (a + b).lo ≤ x + y ∧ x + y ≤ (a + b).hi := by
  sorry  -- To be proven

-- CROWN bound propagation
theorem crown_linear_sound (W : Matrix) (b : Vector) (input : Interval^n) :
  ∀ x ∈ input, crown_output.lo ≤ W * x + b ≤ crown_output.hi := by
  sorry  -- To be proven
```

---

## Success Criteria

| Phase | Criteria | Verification |
|-------|----------|--------------|
| Phase 0 | All quick wins merged | CI passes |
| Phase 1 | Property tests pass with 10K samples | `cargo test --release` |
| Phase 1 | Auto-LiRPA bounds match within 1e-4 | Cross-validation script |
| Phase 2 | Core interval arithmetic proven in Lean | `lake build` succeeds |
| Phase 2 | CROWN algorithm proven in Lean | `lake build` succeeds |
| Phase 3 | Compiled with tRust | Bit-identical output |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Lean 5 proofs too hard | Start with simpler subset (ReLU only, 2-layer networks) |
| tRust not ready | Track project, contribute if needed |
| Bounds still too loose even with formal proofs | Formal proofs don't help tightness, just soundness |
| Performance regression from conservative rounding | Benchmark before/after, make configurable |

---

## References

- **Directed Rounding**: IEEE 754-2019, Section 4.3
- **Verified Floating Point**: "Certified Compilation of Financial Contracts" (Bagnara et al.)
- **Lean 4/5**: https://lean-lang.org/
- **tRust**: [Link to tRust project when available]
- **CompCert**: https://compcert.org/ (C compiler, inspiration for tRust)
- **CROWN Papers**: arxiv:2103.06624, arxiv:2506.06665

---

## Commands

```bash
# Check current version pinning
cat Cargo.lock | head -50

# Run property tests (once implemented)
cargo test --release -- --ignored soundness

# Cross-validate with Auto-LiRPA
python scripts/cross_validate_autolirpa.py models/whisper-tiny

# Build Lean proofs (once implemented)
cd proofs && lake build
```
