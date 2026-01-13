# Kani Examples

This directory contains minimal Kani proof harnesses demonstrating bounded model checking for Rust.

## Prerequisites

- **Rust**: Latest stable toolchain
- **Kani**: Bounded model checker for Rust

### Installation

```bash
# Install Kani verifier
cargo install --locked kani-verifier

# Run first-time setup (downloads CBMC and required toolchain)
cargo kani setup
```

## Files

| File | Description |
|------|-------------|
| `Cargo.toml` | Cargo project configuration |
| `src/lib.rs` | Safe functions with passing proof harnesses |
| `src/fail.rs` | Unsafe functions with failing proof harnesses |
| `OUTPUT_pass.txt` | Real Kani output - all proofs pass |
| `OUTPUT_fail.txt` | Real Kani output - proof finds bug |
| `OUTPUT_counterexample.txt` | Concrete counterexample from Kani |

## Running Kani

### Run all proofs
```bash
cd examples/kani
cargo kani
```

### Run specific harness
```bash
cargo kani --harness verify_safe_add_no_panic
```

### Get counterexample
```bash
cargo kani --harness verify_unsafe_div_fails -Z concrete-playback --concrete-playback=print
```

### Key Options
```bash
# List all harnesses
cargo kani --list

# Verbose output
cargo kani --verbose

# Set solver timeout
cargo kani --solver-timeout 300

# Unwind bound (for loops)
cargo kani --default-unwind 10

# Enable concrete playback
cargo kani -Z concrete-playback --concrete-playback=print
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All harnesses verified successfully |
| 1 | At least one harness failed verification |
| 2 | CLI error (bad arguments, etc.) |

## Output Patterns

### Successful Verification
```
VERIFICATION:- SUCCESSFUL
Verification Time: 0.33087674s

Manual Harness Summary:
Complete - 3 successfully verified harnesses, 0 failures, 3 total.
```

**Key patterns**:
- `VERIFICATION:- SUCCESSFUL`
- `0 failures`

### Failed Verification (Bug Found)
```
Check 1: fail::unsafe_div.assertion.1
	 - Status: FAILURE
	 - Description: "attempt to divide by zero"
	 - Location: src/fail.rs:5:5 in function fail::unsafe_div

SUMMARY:
 ** 1 of 3 failed
Failed Checks: attempt to divide by zero
 File: "src/fail.rs", line 5, in fail::unsafe_div

VERIFICATION:- FAILED
```

**Key patterns**:
- `VERIFICATION:- FAILED`
- `Status: FAILURE`
- `** N of M failed`

### Check Types
| Check Type | Description |
|------------|-------------|
| `assertion` | User assertion failed |
| `arithmetic_overflow` | Integer overflow |
| `division-by-zero` | Division by zero |
| `pointer_dereference` | Null/invalid pointer |
| `unreachable` | Unreachable code reached |

## Proof Harness Syntax

```rust
#[cfg(kani)]
mod verification {
    use super::*;

    // Basic proof harness
    #[kani::proof]
    fn verify_function() {
        let x: u32 = kani::any();  // Symbolic value
        let result = function_under_test(x);
        assert!(result >= 0);  // Property to verify
    }

    // With assumptions
    #[kani::proof]
    fn verify_with_assumption() {
        let x: u32 = kani::any();
        kani::assume(x > 0);  // Constrain symbolic value
        let result = function(x);
        assert!(result != 0);
    }

    // With unwind bound for loops
    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_loop() {
        let n: usize = kani::any();
        kani::assume(n < 10);
        // ...
    }
}
```

## Key Kani Functions

| Function | Description |
|----------|-------------|
| `kani::any()` | Create symbolic value of any type |
| `kani::any_where(|x| cond)` | Symbolic value with constraint |
| `kani::assume(cond)` | Add assumption (constraint) |
| `kani::assert(cond, msg)` | Assert with message |
| `kani::cover(cond)` | Coverage condition |

## Key Discoveries

1. **Exit code 0 vs 1**: 0 = all pass, 1 = at least one fails
2. **Check status**: `Status: FAILURE` or `Status: SUCCESS`
3. **Summary format**: `** N of M failed`
4. **Counterexample**: Use `-Z concrete-playback --concrete-playback=print`
5. **CBMC backend**: Kani uses CBMC 6.8.0 for verification
6. **Nightly Rust**: Kani uses a specific nightly toolchain
7. **cfg(kani)**: Harnesses only compiled when running Kani
