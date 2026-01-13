# rustdoc - The Rust Documentation Generator

rustdoc is the documentation compiler for Rust. It generates HTML documentation from Rust source code, including doc comments and examples.

## Basic Usage

```bash
# Generate documentation
cargo doc

# Open docs in browser
cargo doc --open

# Include private items
cargo doc --document-private-items

# Generate without dependencies
cargo doc --no-deps
```

## Doc Comments

### Outer Documentation

```rust
/// This documents the following item
///
/// # Examples
///
/// ```
/// let x = 5;
/// assert_eq!(x, 5);
/// ```
pub fn my_function() {}

/// A struct with documentation
pub struct MyStruct {
    /// Field documentation
    pub field: i32,
}
```

### Inner Documentation

```rust
//! This documents the enclosing item (usually a module or crate)
//!
//! # Crate-level documentation
//!
//! This appears on the crate root page.

mod inner {
    //! Module-level documentation
}
```

## Markdown Features

### Headers

```rust
/// # Main Header
/// ## Subheader
/// ### Sub-subheader
```

### Code Blocks

```rust
/// ```rust
/// let x = 5;
/// ```
///
/// ```ignore
/// // Not compiled or run
/// ```
///
/// ```no_run
/// // Compiled but not executed
/// std::process::exit(0);
/// ```
///
/// ```should_panic
/// panic!("This should panic");
/// ```
///
/// ```compile_fail
/// let x: i32 = "string"; // Won't compile
/// ```
```

### Links

```rust
/// See [`OtherStruct`] for more info.
///
/// External link: [Rust website](https://rust-lang.org)
///
/// Link to method: [`MyStruct::my_method`]
///
/// Link to module: [`crate::my_module`]
```

### Lists

```rust
/// - Item one
/// - Item two
///   - Nested item
///
/// 1. First
/// 2. Second
```

## Special Sections

### Examples

```rust
/// # Examples
///
/// ```
/// use my_crate::MyStruct;
///
/// let s = MyStruct::new();
/// assert!(s.is_valid());
/// ```
```

### Panics

```rust
/// # Panics
///
/// Panics if `n` is zero.
pub fn divide(a: i32, n: i32) -> i32 {
    if n == 0 { panic!("Division by zero"); }
    a / n
}
```

### Errors

```rust
/// # Errors
///
/// Returns `Err` if the file cannot be read.
pub fn read_config() -> Result<Config, std::io::Error> {
    // ...
}
```

### Safety

```rust
/// # Safety
///
/// The pointer must be valid and properly aligned.
pub unsafe fn dangerous_operation(ptr: *const i32) {
    // ...
}
```

## Doc Attributes

### Hide from Documentation

```rust
#[doc(hidden)]
pub fn internal_function() {}
```

### Inline Module Documentation

```rust
#[doc(inline)]
pub use other_crate::SomeType;
```

### Alias for Search

```rust
#[doc(alias = "append")]
pub fn push(&mut self, item: T) {}
```

### Custom Documentation

```rust
#[doc = include_str!("../README.md")]
pub struct Crate;
```

## Configuration

### Cargo.toml

```toml
[package]
name = "my_crate"
documentation = "https://docs.rs/my_crate"

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

### Feature Flags

```rust
/// This only appears when `my_feature` is enabled.
#[cfg(feature = "my_feature")]
pub fn feature_function() {}

/// Show feature gate in docs
#[cfg_attr(docsrs, doc(cfg(feature = "my_feature")))]
pub fn documented_feature() {}
```

## Command Line

```bash
# Basic generation
cargo doc

# With private items
cargo doc --document-private-items

# Specific package
cargo doc -p my_crate

# All features
cargo doc --all-features

# Custom output directory
cargo doc --target-dir custom_docs

# Generate JSON
cargo rustdoc -- -Z unstable-options --output-format json

# Lint warnings
RUSTDOCFLAGS="-D warnings" cargo doc
```

## Doc Tests

Doc tests are run with:
```bash
cargo test --doc
```

### Hiding Lines

```rust
/// ```
/// # // Lines starting with # are hidden but compiled
/// # fn setup() {}
/// # setup();
/// let visible = true;
/// ```
```

### Test Attributes

```rust
/// ```no_run
/// loop {} // Compiles but doesn't run
/// ```
///
/// ```ignore
/// not_rust_code
/// ```
```

## docs.rs

docs.rs automatically builds documentation for crates on crates.io:

```toml
# Cargo.toml
[package.metadata.docs.rs]
features = ["full"]
all-features = true
targets = ["x86_64-unknown-linux-gnu"]
rustdoc-args = ["--cfg", "docsrs"]
```

## Best Practices

1. **Document Public Items**: All public APIs should have documentation
2. **Include Examples**: Provide runnable examples for complex functions
3. **Use Sections**: Organize with # Examples, # Panics, # Errors, # Safety
4. **Test Examples**: Ensure doc tests pass
5. **Link Related Items**: Use intra-doc links liberally
6. **Crate Documentation**: Add comprehensive //! docs at crate root

## Documentation

- Official: https://doc.rust-lang.org/rustdoc/
- The Rust Book: https://doc.rust-lang.org/book/ch14-02-publishing-to-crates-io.html
- docs.rs: https://docs.rs/
