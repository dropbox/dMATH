# Kani Fast

| Director | Status |
|:--------:|:------:|
| MATH | ACTIVE |

**Next-generation Rust verification: 10-100x faster with unbounded proofs**

Kani Fast dramatically improves on [Kani](https://github.com/model-checking/kani) with:

- **10-100x faster** incremental verification via clause learning and caching
- **Unbounded verification** via k-induction and CHC solving
- **Portfolio solving** with parallel SAT/SMT solvers (Z4, CaDiCaL, Kissat)
- **AI-assisted invariant synthesis** for automatic loop invariant discovery
- **Lean5 proof generation** for machine-checkable verification certificates
- **Beautiful counterexamples** with natural language explanations and repair suggestions

## Quick Start

```bash
# Install
cargo install --path crates/kani-fast-cli

# Check Kani installation
kani-fast check

# Verify a project (bounded model checking)
kani-fast verify /path/to/rust/project

# Verify with portfolio solving (multiple solvers in parallel)
kani-fast verify --portfolio /path/to/project

# Unbounded verification via k-induction + CHC
kani-fast unbounded src/lib.rs --function my_function

# AI-assisted invariant synthesis
kani-fast unbounded src/lib.rs --ai

# Generate Lean5 proof certificates
kani-fast unbounded src/lib.rs --certificate --verify-lean

# Watch mode for continuous verification
kani-fast watch /path/to/project
```

## Requirements

- Rust 1.75+
- [Kani Verifier](https://model-checking.github.io/kani/install-guide.html)

```bash
# Install Kani
cargo install --locked kani-verifier
cargo kani setup
```

### Optional Dependencies (for advanced features)

- **Z4** - Primary SMT/CHC solver (auto-detected, [install from source](https://github.com/dropbox/z4))
- **CaDiCaL** - SAT solver for portfolio mode
- **Kissat** - SAT solver for portfolio mode
- **Lean** - Theorem prover for proof certificate verification

```bash
# macOS (via Homebrew)
brew install cadical kissat

# Z4 (install to ~/.local/bin)
# See https://github.com/dropbox/z4 for build instructions

# Lean (via elan)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

## Library Usage

```rust
use kani_fast::{verify, KaniConfig, KaniWrapper};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Simple verification
    let result = kani_fast::verify("my-project").await?;
    println!("{}", result);

    // With custom configuration
    let config = KaniConfig {
        timeout: std::time::Duration::from_secs(600),
        default_unwind: Some(20),
        ..Default::default()
    };
    let wrapper = KaniWrapper::new(config)?;
    let result = wrapper.verify(Path::new("my-project")).await?;

    if result.status.is_success() {
        println!("Verified!");
    } else if let Some(ce) = result.counterexample {
        println!("Counterexample:\n{}", ce.format_detailed());
    }

    Ok(())
}
```

## Architecture

```
crates/
├── kani-fast              # Main library - public API with tRust integration
├── kani-fast-core         # Verification pipeline orchestration
├── kani-fast-abstract-interp # Abstract interpretation for bounds/nullability analysis
├── kani-fast-counterexample  # Beautiful counterexample generation
├── kani-fast-cli          # Command-line interface
├── kani-fast-portfolio    # Parallel solver management (Z4, CaDiCaL, Kissat)
├── kani-fast-kinduction   # K-induction engine for unbounded verification
├── kani-fast-incremental  # Incremental BMC with clause learning
├── kani-fast-chc          # CHC encoding + Z4 PDR solver
├── kani-fast-ai           # AI-assisted invariant synthesis (LLM integration)
├── kani-fast-lean5        # Lean5 proof generation backend
├── kani-fast-proof        # Universal proof format with O(1) storage
└── kani-fast-compiler     # Native rustc driver (nightly, built separately)
```

## Environment Variables

Configure advanced features via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `KANI_FAST_CHC_BACKEND` | CHC solver backend (only `z4` supported) | z4 |
| `KANI_FAST_BITVEC` | Enable BitVec encoding for precise bitwise operations | disabled |
| `KANI_FAST_BITVEC_WIDTH` | Bit width for BitVec encoding (e.g., 32, 64) | 32 |
| `KANI_FAST_OVERFLOW_CHECKS` | Enable overflow detection in integer arithmetic | disabled |
| `KANI_FAST_DEBUG` | Enable debug output (SMT generation, MIR dumps) | disabled |
| `KANI_FAST_NO_INLINE` | Disable automatic function inlining | disabled (inlining enabled) |
| `KANI_FAST_TIMEOUT_MS` | Z4 solver timeout in milliseconds | 30000 |
| `KANI_FAST_AI_HINT` | Show AI synthesis hints when CHC returns "unknown" | disabled |
| `KANI_FAST_DRY_RUN` | Parse and encode without invoking solver | disabled |

Example usage:
```bash
# Enable bitvec encoding for bitwise-heavy code
KANI_FAST_BITVEC=1 kani-fast chc src/crypto.rs

# Debug mode with 64-bit bitvectors
KANI_FAST_DEBUG=1 KANI_FAST_BITVEC_WIDTH=64 kani-fast chc src/lib.rs

```

## Development

```bash
# Build
cargo build

# Test
cargo test

# Test with verbose output
cargo test --workspace -- --nocapture

# Run CLI in dev mode
cargo run -p kani-fast-cli --bin kani-fast -- --help

# Build native rustc driver (uses pinned nightly from rust-toolchain.toml)
cd crates/kani-fast-compiler && cargo build

# Test native driver
cd crates/kani-fast-compiler && ./test_driver.sh
```

## Known Limitations

For the full, up-to-date list see `docs/SOUNDNESS_LIMITATIONS.md` (mirrors the summary in `CLAUDE.md`). Key CHC-mode gaps to watch for:
- Array reference parameters (`fn foo(arr: &[T; N])`) yield unconstrained index reads; pass arrays by value instead.
- `matches!`/match-to-bool on enum references returns unconstrained booleans; match on owned enums.
- `overflowing_*` tuple extraction leaves the overflow flag unconstrained; prefer `wrapping_*` or `saturating_*`.

### Integer Overflow Detection

The native rustc driver (`kani-fast-driver`) uses unbounded mathematical integers (SMT Int sort) by default, which means arithmetic overflow is not detected. For overflow detection, use:

1. **Kani mode** (`kani-fast verify`): Full overflow checking via bounded model checking
2. **BitVec mode** (`KANI_FAST_BITVEC=1`): Precise fixed-width arithmetic (detects wrap-around, not explicit overflow checks yet)

### Mutable Reference Function Calls

Mutable reference writes across function boundaries are now supported (fixed in #293). When a function takes `&mut T` and writes `*x = value`, the deref assignment is tracked and propagated back to the caller's variable after the call returns.

### MIR Text Parser Limitations

The CLI's CHC mode (`kani-fast chc`) parses MIR text output, which has limitations:
- Enum constructors not parsed (use `kani-fast-driver` instead)
- Some complex MIR patterns may not be supported

For full Rust support, use the native driver:
```bash
cd crates/kani-fast-compiler && cargo build
./target/debug/kani-fast-driver your_file.rs
```

### For Loops / Iterators

Range iterators (`for i in 0..n`) are fully supported with semantic modeling. The driver detects `Range<T>::next()` calls and encodes them with proper iterator semantics.

Complex summation loops may timeout due to invariant synthesis challenges:

```rust
// Fully supported:
for i in 0..5 { count += 1; }  // Works

// May timeout (requires complex invariant):
for i in 0..n { sum += i; }  // Summation with symbolic bound
```

### Trait Method Dispatch

Trait method calls (`x.trait_method()`) are not fully inlined and become uninterpreted functions in the CHC encoding. The solver cannot reason about trait method implementations. Use direct function calls or struct methods instead:

```rust
// Instead of:
trait Doubler { fn double(&self) -> i32; }
impl Doubler for i32 { fn double(&self) -> i32 { *self * 2 } }
let result = x.double();  // Not supported

// Use:
fn double(x: i32) -> i32 { x * 2 }
let result = double(x);  // Works
```

## Test Status

| Category | Result | Notes |
|----------|--------|-------|
| Unit tests | 3625 pass | Workspace crates (kani-fast, kani-fast-chc, etc.; unit + doc tests) |
| Compiler unit tests | 81 pass | kani-fast-compiler (excluded from workspace, test separately) |
| Native driver | 672 defined | 328 pass, 287 expected-fail, 32 soundness bugs (fast mode skips ~25 medium/slow cases) |
| Mutation score | 91.1% | mir.rs (2 equivalent mutants remain) |
| Integration | Pass | CHC, k-induction, portfolio |

Run the full test suite:
```bash
cargo test --workspace                            # 3625 pass (workspace crates)
cd crates/kani-fast-compiler && cargo test        # 81 unit tests (compiler crate)
cd crates/kani-fast-compiler && ./test_driver.sh  # 672 end-to-end tests
```

Static test definitions: 337 pass, 303 expected-fail, 32 soundness bugs (672 total). The default fast mode of `./test_driver.sh` skips ~25 medium/slow tests, so runtime shows ~328 pass, ~287 expected-fail, 25 skipped, 32 soundness.

Test count verified 2026-01-06 (iteration #560).

## License

MIT OR Apache-2.0

## Related Projects

- [Kani](https://github.com/model-checking/kani) - The bit-precise model checker we wrap
- [DashProve](https://github.com/dropbox/dashprove) - Unified verification platform (will integrate Kani Fast)
