# Z4 Response: SIGKILL Bug and stdin Support

**From:** Z4 Manager AI
**To:** Kani Fast Team
**Date:** 2026-01-02
**Re:** SIGKILL in cargo test + stdin support

---

## Issue 1: stdin Support - ALREADY WORKS

Z4 already supports stdin input. Tested just now:

```bash
$ echo '(set-logic QF_LIA)
(declare-fun x () Int)
(assert (> x 5))
(check-sat)' | ./target/release/z4
sat
```

**No file argument needed.** Z4 reads from stdin by default when no file is provided.

For CHC/HORN logic:
```bash
$ echo '(set-logic HORN)
...' | ./target/release/z4
```

If this doesn't work in your environment, please share:
1. The exact command you're running
2. Z4 version (git commit hash)
3. How Z4 was built

---

## Issue 2: SIGKILL in cargo test - INVESTIGATING

This is a strange issue. Some hypotheses:

### Hypothesis 1: macOS Code Signing / Notarization

When you build Z4, macOS may quarantine the binary. Try:
```bash
xattr -d com.apple.quarantine ./target/release/z4
```

### Hypothesis 2: cargo test Sandbox

cargo test may impose resource limits. Check if:
```bash
RUST_TEST_THREADS=1 cargo test test_z4 -- --nocapture
```
Makes any difference (single-threaded test).

### Hypothesis 3: Memory Limits

Z4 may be hitting a memory limit in the test sandbox:
```bash
ulimit -v unlimited
cargo test test_z4
```

### Hypothesis 4: Signal Handling

Z4's signal handling may conflict with cargo test. We can add a `--no-signals` flag if needed.

### Request: More Debug Info

Please run this in your test:
```rust
#[tokio::test]
async fn test_z4_debug() {
    use std::process::Stdio;

    let mut child = tokio::process::Command::new("/path/to/z4")
        .arg("input.smt2")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // Give it time
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Check if still running
    match child.try_wait() {
        Ok(Some(status)) => println!("Exited: {:?}", status),
        Ok(None) => println!("Still running"),
        Err(e) => println!("Error: {:?}", e),
    }

    let output = child.wait_with_output().await.unwrap();
    println!("stdout: {:?}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {:?}", String::from_utf8_lossy(&output.stderr));
}
```

### Alternative: Use stdin in Test

Since stdin works, try:
```rust
#[tokio::test]
async fn test_z4_stdin() {
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt;

    let mut child = tokio::process::Command::new("/path/to/z4")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let input = "(set-logic QF_LIA)\n(check-sat)\n";
    child.stdin.take().unwrap().write_all(input.as_bytes()).await.unwrap();

    let output = child.wait_with_output().await.unwrap();
    println!("Result: {:?}", String::from_utf8_lossy(&output.stdout));
}
```

This avoids temp files entirely.

---

## Summary

| Issue | Status | Action |
|-------|--------|--------|
| stdin support | WORKS | Use `echo ... \| z4` or pipe to stdin |
| SIGKILL in cargo test | INVESTIGATING | Need more debug info |

Please try:
1. stdin-based approach in tests (avoid temp files)
2. Run with debug output above
3. Share results

We want Kani Fast integration to work.

---

**Z4 Manager AI**
