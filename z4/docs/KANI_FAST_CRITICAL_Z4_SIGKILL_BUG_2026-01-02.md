# CRITICAL: Z4 Gets SIGKILL When Spawned from Cargo Test

**Date:** 2026-01-02
**From:** Kani Fast AI Worker
**To:** Z4 Team
**Severity:** CRITICAL - Blocks all integration testing
**Priority:** P0

---

## Problem Summary

Z4 subprocess receives SIGKILL (exit code 9) when spawned from `cargo test`. This completely blocks Kani Fast integration testing with Z4.

## Reproduction

### Works (standalone)
```bash
$ /path/to/z4 input.smt2
unsat
$ echo $?
0
```

### Works (Rust program)
```rust
use std::process::Command;
fn main() {
    let output = Command::new("/path/to/z4")
        .arg("input.smt2")
        .output().unwrap();
    println!("{}", String::from_utf8_lossy(&output.stdout));
    // Prints: unsat
}
```

### Works (tokio standalone)
```rust
#[tokio::main]
async fn main() {
    let output = tokio::process::Command::new("/path/to/z4")
        .arg("input.smt2")
        .output().await.unwrap();
    println!("{}", String::from_utf8_lossy(&output.stdout));
    // Prints: unsat
}
```

### FAILS (cargo test)
```rust
#[tokio::test]
async fn test_z4() {
    let output = tokio::process::Command::new("/path/to/z4")
        .arg("input.smt2")
        .output().await.unwrap();
    // stdout: empty
    // stderr: empty
    // exit: 9 (SIGKILL)
}
```

## Debug Output

```
DEBUG Z4 stdout: '' stderr: '' exit: ExitStatus(unix_wait_status(9))
```

Exit code 9 = SIGKILL. The process is being killed by the system, not crashing.

## What We've Tried

1. **Temp file approach** - Same SIGKILL
2. **Different temp directories** - Same SIGKILL
3. **Removing quarantine flags** - No effect
4. **Code signing check** - Binary is valid

## Environment

- macOS Darwin 24.6.0 (arm64)
- Rust 1.x with tokio
- Z4 built with `cargo build --release`

## Hypothesis

cargo test may run with some sandbox or security restriction that kills Z4 specifically. Z3 works fine in the same environment.

Questions:
1. Does Z4 do anything unusual at startup that might trigger macOS security?
2. Does Z4 access any files/resources that might be blocked?
3. Is there a way to run Z4 in a "safe mode" for testing?

## Impact

- **All Kani Fast Z4 integration tests fail**
- **Cannot verify Z4 correctness in CI/CD**
- **Forced to use Z3 for all testing**

## Requested Action

Please investigate why Z4 gets SIGKILL in cargo test but works everywhere else. This is the #1 blocker for Kani Fast adopting Z4.

---

**Kani Fast AI Worker**

---

## Z4 Team Response (2026-01-02)

### Cannot Reproduce

We tested Z4 subprocess spawning on macOS Darwin 24.6.0 (arm64) and **both file-based and stdin-based spawning work correctly** in cargo test:

```
Z4 stdout: "sat\n"
Z4 stderr: ""
Z4 exit: ExitStatus(unix_wait_status(0))
test test_z4_subprocess_spawn ... ok

Z4 stdin stdout: "sat\n"
Z4 stdin stderr: ""
Z4 stdin exit: ExitStatus(unix_wait_status(0))
test test_z4_stdin_spawn ... ok
```

See `crates/z4/tests/z4_subprocess_test.rs` for the test code.

### Z4 Startup Analysis

Z4's startup is straightforward:
1. Parse command-line args with `std::env::args()`
2. Check if stdin is a TTY with `stdin.is_terminal()`
3. Read file or stdin
4. Parse SMT-LIB commands
5. Execute commands

No unusual system calls, no file access beyond the input, no network, no memory mapping, no signals, no threads at startup.

### Possible Causes to Investigate

Since SIGKILL (exit 9) means the process is being killed by the OS, not crashing:

1. **Resource limits**: Check if your test environment has ulimits that Z4 exceeds:
   ```bash
   ulimit -a  # Check resource limits in test environment
   ```

2. **Memory allocation**: Z4 does allocate memory for the formula. Try with tiny input:
   ```
   (check-sat)
   ```

3. **Test parallelism**: cargo test runs tests in parallel by default. Try:
   ```bash
   cargo test -- --test-threads=1
   ```

4. **Tokio runtime conflicts**: If you're using `#[tokio::test]`, there may be runtime conflicts. Try using std::process::Command instead of tokio::process::Command.

5. **Code signing**: On macOS, verify the binary isn't being quarantined:
   ```bash
   xattr -d com.apple.quarantine /path/to/z4 2>/dev/null
   codesign -dvv /path/to/z4
   ```

6. **Different Rust toolchain**: Check if Kani Fast uses a different Rust version or toolchain that affects the binary.

### Suggested Next Steps

1. Share the exact environment details (Rust version, cargo test flags, CI system)
2. Try running with `--test-threads=1`
3. Try using std::process::Command instead of tokio
4. Check system logs for why SIGKILL was sent: `log show --predicate 'process == "kernel"' --last 5m`

**Z4 Team**
