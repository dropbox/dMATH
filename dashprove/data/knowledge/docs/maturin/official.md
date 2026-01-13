# Maturin - Build and Publish Rust-based Python Packages

Maturin is a build system for building and publishing Rust-based Python packages with minimal configuration. It supports pyo3, rust-cpython, cffi, and uniffi bindings.

## Installation

```bash
# Using pip
pip install maturin

# Using pipx
pipx install maturin

# Using cargo
cargo install maturin

# Using uv
uv tool install maturin
```

## Basic Usage

### Create New Project

```bash
# Create new project with pyo3 bindings
maturin init

# Create with specific bindings
maturin init --bindings pyo3
maturin init --bindings cffi
maturin init --bindings uniffi

# Create mixed Rust/Python project
maturin init --mixed
```

### Development Workflow

```bash
# Build and install in current virtualenv
maturin develop

# Build with release optimizations
maturin develop --release

# Build with specific features
maturin develop --features feature1,feature2
```

### Building Wheels

```bash
# Build wheel for current platform
maturin build

# Build release wheel
maturin build --release

# Build for specific Python version
maturin build -i python3.12

# Build for all Python versions
maturin build --find-interpreter

# Build source distribution
maturin sdist
```

### Publishing

```bash
# Publish to PyPI
maturin publish

# Publish to test PyPI
maturin publish --repository testpypi

# Build and upload in one step
maturin upload target/wheels/*.whl
```

## Project Structure

### Standard Layout

```
my-project/
├── Cargo.toml
├── pyproject.toml
├── src/
│   └── lib.rs
└── python/
    └── my_project/
        └── __init__.py
```

### Cargo.toml

```toml
[package]
name = "my-project"
version = "0.1.0"
edition = "2021"

[lib]
name = "my_project"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
```

### pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "my-project"
version = "0.1.0"
description = "My Rust-Python project"
requires-python = ">=3.9"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy",
]

[tool.maturin]
python-source = "python"
features = ["pyo3/extension-module"]
```

## PyO3 Bindings

### Basic Module

```rust
use pyo3::prelude::*;

#[pyfunction]
fn sum_as_string(a: i64, b: i64) -> String {
    (a + b).to_string()
}

#[pymodule]
fn my_project(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
```

### Classes

```rust
use pyo3::prelude::*;

#[pyclass]
struct Counter {
    count: i64,
}

#[pymethods]
impl Counter {
    #[new]
    fn new() -> Self {
        Counter { count: 0 }
    }

    fn increment(&mut self) {
        self.count += 1;
    }

    fn get_count(&self) -> i64 {
        self.count
    }
}

#[pymodule]
fn my_project(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Counter>()?;
    Ok(())
}
```

## Mixed Rust/Python Projects

### Structure

```
my-project/
├── Cargo.toml
├── pyproject.toml
├── src/
│   └── lib.rs        # Rust code
└── python/
    └── my_project/
        ├── __init__.py
        └── utils.py   # Pure Python code
```

### Configuration

```toml
# pyproject.toml
[tool.maturin]
python-source = "python"
module-name = "my_project._internal"
```

```python
# python/my_project/__init__.py
from .utils import helper_function
from ._internal import rust_function
```

## Cross-Compilation

```bash
# Build for Linux on macOS
maturin build --target x86_64-unknown-linux-gnu

# Build for Windows
maturin build --target x86_64-pc-windows-msvc

# Build for multiple platforms
maturin build --target x86_64-unknown-linux-gnu \
              --target aarch64-unknown-linux-gnu
```

### Using zig for cross-compilation

```bash
# Install zig linker
pip install ziglang

# Build with zig
maturin build --zig --target x86_64-unknown-linux-gnu
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Build wheels

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4

      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release --out dist

      - name: Upload wheels
        uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: dist

  publish:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4

      - name: Publish to PyPI
        uses: PyO3/maturin-action@v1
        env:
          MATURIN_PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
        with:
          command: upload
          args: --skip-existing wheels-*/*
```

### Linux Wheels (manylinux)

```yaml
- name: Build manylinux wheels
  uses: PyO3/maturin-action@v1
  with:
    manylinux: auto
    command: build
    args: --release
```

## Features and Conditional Compilation

```toml
# Cargo.toml
[features]
default = []
extension-module = ["pyo3/extension-module"]
numpy = ["numpy"]

[dependencies]
numpy = { version = "0.22", optional = true }
```

```bash
# Build with features
maturin build --features numpy
```

## Configuration Options

```toml
# pyproject.toml
[tool.maturin]
# Python source directory
python-source = "python"

# Module name (if different from package name)
module-name = "my_project._core"

# Bindings type
bindings = "pyo3"

# Cargo features
features = ["pyo3/extension-module"]

# Strip symbols for smaller binary
strip = true

# Compatibility tag
compatibility = "manylinux2014"

# Excluded files from sdist
exclude = ["tests/*", "benches/*"]
```

## Common Commands

```bash
# Development
maturin develop          # Build and install locally
maturin develop --release  # With optimizations

# Building
maturin build            # Build wheel
maturin build --release  # Release build
maturin sdist            # Source distribution

# Publishing
maturin publish          # Publish to PyPI
maturin upload dist/*    # Upload built wheels

# Cross-compilation
maturin build --target aarch64-apple-darwin
maturin build --zig --target x86_64-unknown-linux-gnu
```

## Best Practices

1. **Use --release for distribution** - Debug builds are slow
2. **Test with maturin develop** - Fast iteration cycle
3. **Use maturin-action in CI** - Handles cross-compilation
4. **Enable strip** - Smaller wheel sizes
5. **Use mixed projects** - Combine Rust speed with Python flexibility

## Links

- Documentation: https://www.maturin.rs/
- GitHub: https://github.com/PyO3/maturin
- PyPI: https://pypi.org/project/maturin/
- PyO3: https://pyo3.rs/
