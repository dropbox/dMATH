# cargo-llvm-cov - LLVM Source-Based Code Coverage

cargo-llvm-cov is a cargo subcommand for LLVM source-based code coverage. It provides accurate coverage information using LLVM's native coverage instrumentation.

## Features

- **Source-Based Coverage**: Uses LLVM instrumentation for accurate results
- **Multiple Output Formats**: lcov, HTML, JSON, text
- **Workspace Support**: Coverage for entire workspaces
- **Doc Test Coverage**: Includes documentation tests
- **Branch Coverage**: Track branch/decision coverage

## Installation

```bash
cargo install cargo-llvm-cov
```

Requires LLVM tools (installed via rustup):
```bash
rustup component add llvm-tools-preview
```

## Basic Usage

```bash
# Run tests with coverage
cargo llvm-cov

# Generate HTML report
cargo llvm-cov --html

# Open HTML report
cargo llvm-cov --open

# Generate lcov report
cargo llvm-cov --lcov --output-path lcov.info

# JSON output
cargo llvm-cov --json --output-path coverage.json
```

## Output Formats

### Text (Default)

```bash
cargo llvm-cov
# Output:
# Filename                      Regions    Missed  Cover   Lines  Missed  Cover
# src/lib.rs                         45        10  77.78%    120      25  79.17%
# src/parser.rs                      30         5  83.33%     85      10  88.24%
# TOTAL                              75        15  80.00%    205      35  82.93%
```

### HTML Report

```bash
cargo llvm-cov --html
cargo llvm-cov --html --output-dir coverage_html

# Open in browser
cargo llvm-cov --open
```

### lcov Format

```bash
cargo llvm-cov --lcov --output-path lcov.info

# For Codecov, Coveralls, etc.
cargo llvm-cov --lcov > coverage.lcov
```

### JSON Format

```bash
cargo llvm-cov --json --output-path coverage.json

# Pretty JSON
cargo llvm-cov --json --pretty
```

### Cobertura XML

```bash
cargo llvm-cov --cobertura --output-path coverage.xml
```

## Test Selection

```bash
# Run specific tests
cargo llvm-cov -- specific_test

# Run only lib tests
cargo llvm-cov --lib

# Run only bin tests
cargo llvm-cov --bin my_binary

# Run all tests including ignored
cargo llvm-cov -- --include-ignored

# Include doc tests
cargo llvm-cov --doctests
```

## Filtering

### Include/Exclude Files

```bash
# Ignore specific files
cargo llvm-cov --ignore-filename-regex "tests/.*"

# Only show specific files
cargo llvm-cov --include-fns "my_module::.*"
```

### Branch Coverage

```bash
# Enable branch coverage
cargo llvm-cov --branch
```

## Workspace Coverage

```bash
# Coverage for entire workspace
cargo llvm-cov --workspace

# Coverage for specific packages
cargo llvm-cov -p crate1 -p crate2

# Exclude packages
cargo llvm-cov --workspace --exclude test-utils
```

## CI Integration

### GitHub Actions

```yaml
- name: Install cargo-llvm-cov
  uses: taiki-e/install-action@cargo-llvm-cov

- name: Generate coverage
  run: cargo llvm-cov --lcov --output-path lcov.info

- name: Upload to Codecov
  uses: codecov/codecov-action@v3
  with:
    files: lcov.info
```

### With Coverage Threshold

```yaml
- name: Check coverage threshold
  run: |
    cargo llvm-cov --json | jq '.data[0].totals.lines.percent' | \
    awk '{ if ($1 < 80) exit 1 }'
```

## Configuration

### Environment Variables

```bash
# Set coverage output directory
CARGO_LLVM_COV_TARGET_DIR=/path/to/dir cargo llvm-cov
```

### With Cargo Features

```bash
# Enable specific features
cargo llvm-cov --features "feature1,feature2"

# All features
cargo llvm-cov --all-features

# No default features
cargo llvm-cov --no-default-features
```

## Advanced Usage

### Clean Previous Data

```bash
# Clean before run
cargo llvm-cov clean --workspace
cargo llvm-cov
```

### Run Without Report

```bash
# Just run tests (for merging later)
cargo llvm-cov --no-report

# Generate report from existing data
cargo llvm-cov --no-run --lcov
```

### Merge Coverage Data

```bash
# Run unit tests
cargo llvm-cov --no-report

# Run integration tests
cargo llvm-cov --no-report -- --test integration

# Generate merged report
cargo llvm-cov --no-run --lcov
```

### Profile Override

```bash
# Use release profile
cargo llvm-cov --release

# Use custom profile
cargo llvm-cov --profile custom-profile
```

## Comparison with tarpaulin

| Feature | cargo-llvm-cov | tarpaulin |
|---------|----------------|-----------|
| Method | LLVM instrumentation | ptrace |
| Accuracy | Higher | Good |
| Speed | Faster | Slower |
| Platform | All | Linux mainly |
| Branch coverage | Yes | Yes |
| Doc tests | Yes | Limited |

## Troubleshooting

### Missing Coverage

```bash
# Ensure LLVM tools are installed
rustup component add llvm-tools-preview

# Clean and rebuild
cargo llvm-cov clean
cargo llvm-cov
```

### Inconsistent Results

```bash
# Use clean build
cargo llvm-cov clean --workspace
cargo llvm-cov
```

## Documentation

- GitHub: https://github.com/taiki-e/cargo-llvm-cov
- crates.io: https://crates.io/crates/cargo-llvm-cov
