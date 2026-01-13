# cargo-nextest - A Next-Generation Test Runner for Rust

cargo-nextest is a next-generation test runner for Rust projects. It's designed to be fast, reliable, and provide better output than `cargo test`.

## Features

- **Parallel Test Execution**: Runs tests in parallel with better resource utilization
- **Better Output**: Clean, human-readable output with progress bars
- **Retries**: Automatically retry flaky tests
- **Partitioning**: Split tests across multiple CI jobs
- **JUnit XML Output**: Native support for CI integration
- **Test Archives**: Create and run test archives for reproducibility

## Installation

```bash
cargo install cargo-nextest --locked
```

Or with cargo-binstall:
```bash
cargo binstall cargo-nextest
```

## Basic Usage

```bash
# Run all tests
cargo nextest run

# Run specific tests
cargo nextest run test_name

# Run tests matching a filter
cargo nextest run -E 'test(my_test)'

# Run tests in a specific package
cargo nextest run -p my_package
```

## Configuration

Create `.config/nextest.toml` in your project:

```toml
[profile.default]
retries = 2
test-threads = "num-cpus"
fail-fast = false
slow-timeout = { period = "60s", terminate-after = 2 }

[profile.ci]
retries = 3
fail-fast = true

[profile.default.junit]
path = "target/nextest/default/junit.xml"
```

## Filtering Tests

Use the `-E` flag with filter expressions:

```bash
# Run tests containing "parse"
cargo nextest run -E 'test(parse)'

# Run tests in package "core"
cargo nextest run -E 'package(core)'

# Run slow tests only
cargo nextest run -E 'test(/slow/)'

# Combine filters
cargo nextest run -E 'test(parse) & package(core)'
```

## Filter Expression Syntax

| Expression | Description |
|------------|-------------|
| `test(name)` | Match test names |
| `package(name)` | Match package names |
| `binary(name)` | Match binary names |
| `platform(host)` | Match host platform |
| `all()` | Match all tests |
| `none()` | Match no tests |
| `&` | AND |
| `\|` | OR |
| `!` | NOT |

## CI Integration

### GitHub Actions

```yaml
- name: Install nextest
  uses: taiki-e/install-action@nextest

- name: Run tests
  run: cargo nextest run --profile ci
```

### Test Partitioning

Split tests across multiple CI jobs:

```bash
# Job 1 of 3
cargo nextest run --partition count:1/3

# Job 2 of 3
cargo nextest run --partition count:2/3

# Job 3 of 3
cargo nextest run --partition count:3/3
```

## Test Archives

Create a test archive for reproducible runs:

```bash
# Create archive
cargo nextest archive --archive-file tests.tar.zst

# Run from archive
cargo nextest run --archive-file tests.tar.zst
```

## Output Formats

```bash
# Default output
cargo nextest run

# JSON output
cargo nextest run --message-format json

# JUnit XML
cargo nextest run --profile ci  # if junit configured
```

## Retries for Flaky Tests

```toml
# .config/nextest.toml
[profile.default]
retries = 2

# Or mark specific tests
[profile.default.overrides]
"test(flaky_test)" = { retries = 5 }
```

## Performance Tips

1. Use `--no-fail-fast` in CI for complete results
2. Configure `test-threads` appropriately
3. Use test archives for distributed testing
4. Enable `--cargo-profile release-with-debug` for faster tests

## Comparison with cargo test

| Feature | cargo test | cargo nextest |
|---------|------------|---------------|
| Parallel execution | Per-binary | Per-test |
| Progress output | Basic | Rich |
| Retries | No | Yes |
| Partitioning | No | Yes |
| JUnit output | No | Yes |
| Archives | No | Yes |

## Documentation

- Official: https://nexte.st/
- GitHub: https://github.com/nextest-rs/nextest
- Configuration: https://nexte.st/book/configuration.html
