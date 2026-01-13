# Cargo - The Rust Package Manager

Cargo is the Rust package manager. Cargo downloads your Rust package's dependencies, compiles your packages, makes distributable packages, and uploads them to crates.io.

## Getting Started

### Installation

Cargo is installed automatically when you install Rust via `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Creating a New Project

```bash
cargo new hello_world
cd hello_world
```

This creates:
```
hello_world/
├── Cargo.toml
└── src
    └── main.rs
```

### Building and Running

```bash
cargo build          # Compile the project
cargo run            # Compile and run
cargo build --release  # Build with optimizations
```

### Testing

```bash
cargo test           # Run all tests
cargo test test_name # Run specific test
```

## Cargo.toml

The manifest file for a Rust project:

```toml
[package]
name = "hello_world"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
tokio = { version = "1", features = ["full"] }

[dev-dependencies]
proptest = "1.0"

[build-dependencies]
cc = "1.0"
```

## Common Commands

| Command | Description |
|---------|-------------|
| `cargo new` | Create a new Cargo package |
| `cargo init` | Initialize Cargo in existing directory |
| `cargo build` | Compile the current package |
| `cargo run` | Build and execute src/main.rs |
| `cargo test` | Run the tests |
| `cargo bench` | Run benchmarks |
| `cargo doc` | Build documentation |
| `cargo publish` | Package and upload to crates.io |
| `cargo install` | Install a Rust binary |
| `cargo search` | Search for crates |
| `cargo update` | Update dependencies |
| `cargo clean` | Remove build artifacts |
| `cargo check` | Check for errors without building |
| `cargo clippy` | Run the Clippy linter |
| `cargo fmt` | Format code with rustfmt |
| `cargo tree` | Display dependency tree |
| `cargo vendor` | Vendor all dependencies |
| `cargo fix` | Automatically fix warnings |

## Features

Enable or disable optional features:

```toml
[dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }

[features]
default = ["std"]
std = []
unstable = []
```

## Workspaces

Manage multiple related packages:

```toml
# root Cargo.toml
[workspace]
members = ["crate1", "crate2", "crate3"]

[workspace.dependencies]
serde = "1.0"
```

## Build Scripts

Custom build logic via `build.rs`:

```rust
// build.rs
fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rustc-link-lib=foo");
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CARGO_HOME` | Cargo home directory |
| `CARGO_TARGET_DIR` | Output directory for build artifacts |
| `RUSTFLAGS` | Additional rustc flags |
| `CARGO_INCREMENTAL` | Enable incremental compilation |
| `CARGO_BUILD_JOBS` | Number of parallel jobs |

## Profiles

Configure compilation settings:

```toml
[profile.dev]
opt-level = 0
debug = true

[profile.release]
opt-level = 3
lto = true
panic = "abort"

[profile.bench]
opt-level = 3
debug = false
```

## Documentation

- Official Book: https://doc.rust-lang.org/cargo/
- Reference: https://doc.rust-lang.org/cargo/reference/
- Crates.io: https://crates.io/
