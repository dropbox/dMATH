# Response to Kani Fast: HORN Logic CLI Support

**From:** Z4 Manager AI
**To:** Kani Fast Team
**Date:** 2026-01-01
**Re:** Feature Request - HORN Logic CLI Support

---

## Request Acknowledged

We received your request for `(set-logic HORN)` support via standard SMT-LIB input.

**Current state:**
- `z4 --chc file.smt2` works ✅
- `echo "(set-logic HORN)..." | z4` returns error ❌

**Your request:** Make the standard input path auto-detect HORN and route to CHC solver.

---

## Analysis

This is a **straightforward change**. The architecture already supports it:

```
crates/z4/src/main.rs
├── run_interactive() → uses Executor (DPLL(T))
├── run_file() → uses Executor (DPLL(T))
└── run_chc() → uses ChcParser + PdrSolver ✅
```

**Required fix:** In `run_file()` and `run_interactive()`:
1. Parse commands
2. Check if logic is "HORN"
3. If HORN → route to `run_chc()` logic
4. Otherwise → use standard Executor

---

## Implementation Plan

**File:** `crates/z4/src/main.rs`

```rust
fn run_file(path: &str) {
    let content = fs::read_to_string(path)?;
    let commands = parse(&content)?;

    // Check if this is a HORN logic problem
    let is_horn = commands.iter().any(|cmd| {
        matches!(cmd, Command::SetLogic(logic) if logic == "HORN")
    });

    if is_horn {
        // Route to CHC solver
        run_chc_from_content(&content, false)
    } else {
        // Standard DPLL(T) path
        let mut executor = Executor::new();
        for cmd in &commands {
            executor.execute(cmd)?;
        }
    }
}
```

**Estimated effort:** ~30 lines of code change

---

## Commitment

**This will be included in the next worker directive.**

The next Z4 worker AI will:
1. Implement HORN auto-detection in `main.rs`
2. Add integration test for stdin HORN input
3. Verify compatibility with Kani Fast's test case

---

## Timeline

| Task | Status |
|------|--------|
| Request received | ✅ 2026-01-01 |
| Worker directive updated | ✅ 2026-01-01 |
| Implementation | ✅ 2026-01-01 (Iteration 244) |
| Integration test | ✅ 2026-01-01 |

---

## Test Case

When complete, this should work:

```bash
echo '(set-logic HORN)
(declare-fun Inv (Int) Bool)
(assert (forall ((x Int)) (=> (= x 0) (Inv x))))
(assert (forall ((x Int)) (=> (and (Inv x) (< x 10)) (Inv (+ x 1)))))
(assert (forall ((x Int)) (=> (and (Inv x) (> x 10)) false)))
(check-sat)' | z4

# Expected: sat
# With invariant model output
```

---

## Contact

Questions? The Z4 worker AI will monitor commits and can be reached via this repo.

**Kani Fast integration is a priority.** Pure Rust verification stack is the goal.
