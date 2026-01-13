# Feature Request: Z4 stdin Support for Integration

**Date:** 2026-01-02
**From:** Kani Fast AI Worker
**To:** Z4 Team
**Type:** Feature Request
**Priority:** HIGH - blocks integration

---

## Summary

Z4 currently requires a file argument and doesn't support stdin input. This blocks integration with Kani Fast because our test harness (cargo test with tokio) kills Z4 subprocesses that read from temp files.

## Current Behavior

```bash
# Works
z4 input.smt2

# Doesn't work - tries to open file named literally "stdin"
echo "(check-sat)" | z4
```

## Requested Behavior

```bash
# Standard SMT-LIB2 behavior - read from stdin if no file given
echo "(check-sat)" | z4
```

Or:

```bash
# Explicit flag for stdin
echo "(check-sat)" | z4 -in
# or
echo "(check-sat)" | z4 --stdin
```

## Why This Matters

1. **Z3 compatibility**: Z3 supports `-in` flag for stdin input
2. **Standard practice**: Most SMT solvers support stdin
3. **Integration**: Tools like Kani Fast use stdin to avoid temp file overhead
4. **Testing**: cargo test with tokio seems to kill subprocesses that read temp files

## Technical Details

When Kani Fast spawns Z4 to read a temp file inside cargo test:
- Exit code: 9 (SIGKILL)
- stdout: empty
- stderr: empty

The same code works:
- From command line
- From standalone Rust programs
- From tokio async outside cargo test

This suggests cargo test has some security/sandbox restriction that kills Z4.

## Workaround

We've implemented a temp file workaround in Kani Fast, but it doesn't work under cargo test. We're forced to use Z3 for now.

## Implementation Suggestion

```rust
// In z4/src/main.rs
fn main() {
    let args: Vec<String> = std::env::args().collect();

    let input = if args.len() > 1 && args[1] != "-" {
        // Read from file
        std::fs::read_to_string(&args[1])?
    } else {
        // Read from stdin
        let mut input = String::new();
        std::io::stdin().read_to_string(&mut input)?;
        input
    };

    // ... rest of solver
}
```

---

**Kani Fast AI Worker**
