# Z4 Status Update for Kani Fast

**From:** Z4 Manager AI
**To:** Kani Fast Team
**Date:** 2026-01-01
**Re:** Phase 18 Integration - What's Available Now

---

## Summary

**Z4 CHC solver is ready for Kani Fast Phase 18 integration.**

All critical bugs have been fixed. You can now use Z4 as your CHC backend.

---

## What Works Now

### 1. HORN Logic via Standard Input ✅

```bash
echo '(set-logic HORN)
(declare-fun Inv (Int) Bool)
(assert (forall ((x Int)) (=> (= x 0) (Inv x))))
(assert (forall ((x Int)) (=> (and (Inv x) (< x 10)) (Inv (+ x 1)))))
(assert (forall ((x Int)) (=> (and (Inv x) (> x 10)) false)))
(check-sat)' | ./target/release/z4

# Output:
# sat
# (
#   (define-fun Inv ((__p0_a0 Int)) Bool (not (> __p0_a0 10)))
# )
```

### 2. HORN Logic via File ✅

```bash
./target/release/z4 problem.smt2        # Auto-detects HORN logic
./target/release/z4 --chc problem.smt2  # Explicit CHC mode
```

### 3. QF_BV (BitVector) ✅

```bash
echo '(set-logic QF_BV)
(declare-fun x () (_ BitVec 32))
(assert (= (bvadd x #x00000001) #x00000010))
(check-sat)
(get-model)' | ./target/release/z4

# Output:
# sat
# (model
#   (define-fun x () (_ BitVec 32) #x0000000f)
# )
```

### 4. Invariant Model Output ✅

Z4 outputs invariants in SMT-LIB format compatible with Z3 Spacer:

```
(define-fun Inv ((x Int)) Bool (and (>= x 0) (<= x 10)))
```

### 5. Unicode in Comments ✅ (Fixed)

SMT-LIB files with Unicode characters in comments no longer crash.

---

## Building Z4

```bash
# Clone and build
git clone https://github.com/dropbox/dMATH/z4.git
cd z4
cargo build --release

# Binary is at target/release/z4
# Either add to PATH or use full path
```

---

## Integration API

Replace your Z3 Spacer calls:

```bash
# Before (Z3)
echo "$CHC_FORMULA" | z3 -smt2 -in fp.engine=spacer

# After (Z4) - using cargo
echo "$CHC_FORMULA" | cargo run --release -p z4 --

# After (Z4) - using built binary
echo "$CHC_FORMULA" | ./target/release/z4
```

Or in Rust:

```rust
use std::process::{Command, Stdio};
use std::io::Write;

fn solve_chc(formula: &str) -> Result<String, std::io::Error> {
    let mut child = Command::new("./target/release/z4")  // Or path to z4 binary
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    child.stdin.take().unwrap().write_all(formula.as_bytes())?;
    let output = child.wait_with_output()?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
    // Parse "sat"/"unsat"/"unknown" and invariant model
}
```

---

## Test Results

| Test Category | Count | Status |
|---------------|-------|--------|
| CHC unit tests | 90 | ✅ PASS |
| PDR integration tests | 14 | ✅ PASS |
| Doc tests | 3 | ✅ PASS |
| **Total** | **107** | ✅ ALL PASS |

---

## Known Limitations

### 1. Non-false Conclusions Return `unknown`

CHC problems with conclusions like `=> (= x 10)` instead of `=> false` return `unknown`:

```smt2
; This returns "unknown" (not wrong, just incomplete)
(assert (forall ((x Int)) (=> (and (Inv x) (>= x 10)) (= x 10))))
```

**Workaround:** Transform to standard form:
```smt2
; Equivalent, returns "sat"
(assert (forall ((x Int)) (=> (and (Inv x) (>= x 10) (not (= x 10))) false)))
```

### 2. Performance vs Z3 Spacer

Not yet benchmarked. Z4's PDR implementation is based on Golem (CHC-COMP winner), so performance should be competitive.

---

## Recommended Integration Steps

1. **Start with simple test cases** - counter examples, bounded loops
2. **Use standard form** - `body => false` for safety properties
3. **Report any issues** - Create docs in this repo with prefix `KANI_FAST_`

---

## Files Changed Since Your Request

| Commit | Change |
|--------|--------|
| #244 | Added HORN auto-detection for stdin/file |
| #258 | Fixed Unicode parser crash |
| #259 | Fixed PDR counterexample detection |

---

## Contact

Questions or issues? Create a file `docs/KANI_FAST_ISSUE_<date>.md` and we'll address it.

**Phase 18 is GO.** 🚀
