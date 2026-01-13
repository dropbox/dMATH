# cargo-watch - Watch for Changes and Run Cargo Commands

cargo-watch watches your Rust project for file changes and runs specified cargo commands automatically. Essential for fast development feedback loops.

## Installation

```bash
cargo install cargo-watch
```

## Basic Usage

```bash
# Run cargo check on changes
cargo watch

# Run specific command
cargo watch -x check
cargo watch -x test
cargo watch -x run

# Chain multiple commands
cargo watch -x check -x test -x run

# Custom command
cargo watch -- cargo build --release
```

## Common Workflows

### Development

```bash
# Check for errors (fastest)
cargo watch -x check

# Check and test
cargo watch -x check -x test

# Check, test, and run
cargo watch -x check -x test -x "run -- --some-arg"
```

### Testing

```bash
# Run all tests
cargo watch -x test

# Run specific test
cargo watch -x "test my_test_name"

# Run tests with output
cargo watch -x "test -- --nocapture"

# Test with nextest
cargo watch -x "nextest run"
```

### Building

```bash
# Debug build
cargo watch -x build

# Release build
cargo watch -x "build --release"

# Build specific binary
cargo watch -x "build --bin my_binary"
```

## Options

### Watch Paths

```bash
# Watch specific directory
cargo watch -w src -x check

# Watch multiple paths
cargo watch -w src -w tests -x test

# Ignore paths
cargo watch -i "*.log" -i "target/*" -x check
```

### Clear Screen

```bash
# Clear screen before each run
cargo watch -c -x check
```

### Delay

```bash
# Wait 500ms after changes before running
cargo watch -d 0.5 -x check

# No delay (immediate)
cargo watch -d 0 -x check
```

### Shell

```bash
# Use specific shell
cargo watch -s "echo hello && cargo check"

# Run arbitrary shell command
cargo watch -- sh -c "cargo build && ./target/debug/my_app"
```

### Quiet Mode

```bash
# Suppress cargo-watch output
cargo watch -q -x check
```

## Filtering

### File Extensions

```bash
# Only watch Rust files
cargo watch -w src --why -x check
```

### Ignore Patterns

```bash
# Ignore generated files
cargo watch -i "src/generated/*" -x check

# Ignore test output
cargo watch -i "*.log" -i "*.tmp" -x check

# Ignore by pattern
cargo watch -i "**/tests/**" -x check
```

## Environment Variables

```bash
# Set environment variable
RUST_LOG=debug cargo watch -x run

# With env flag
cargo watch -E RUST_LOG=debug -x run
```

## Notification

```bash
# Desktop notification on finish (requires notify-send or similar)
cargo watch -x check -s "notify-send 'Build complete'"
```

## Working with Workspaces

```bash
# Watch entire workspace
cargo watch -x "check --workspace"

# Watch specific package
cargo watch -p my_crate -x check

# Watch and test all
cargo watch -x "test --workspace"
```

## Integration with Other Tools

### With clippy

```bash
cargo watch -x clippy
cargo watch -x "clippy -- -D warnings"
```

### With rustfmt

```bash
cargo watch -x "fmt -- --check"
```

### With nextest

```bash
cargo watch -x "nextest run"
```

### With miri (unsafe code checking)

```bash
cargo watch -x "miri test"
```

## Poll Mode

For environments where filesystem events don't work:

```bash
# Enable polling (useful in Docker, VMs)
cargo watch --poll -x check
```

## Performance Tips

1. **Use check instead of build**: `cargo check` is faster
2. **Add ignores**: Ignore target/, logs, generated files
3. **Watch specific paths**: Don't watch entire project if unnecessary
4. **Use delay**: Prevent multiple runs for batched saves
5. **Clear screen**: Easier to see fresh output

## Example .cargo/watch.toml

```toml
# Not officially supported, but you can create shell aliases

# In your shell config:
alias cw="cargo watch -c -x check"
alias cwt="cargo watch -c -x check -x test"
alias cwr="cargo watch -c -x run"
```

## Full Command Reference

```bash
cargo watch [OPTIONS] [-- <command>...]

OPTIONS:
    -x, --exec <cmd>       Cargo command to run
    -s, --shell <cmd>      Shell command to run
    -w, --watch <path>     Watch specific path
    -i, --ignore <pattern> Ignore pattern
    -c, --clear            Clear screen before run
    -q, --quiet            Suppress output
    -d, --delay <secs>     Delay before running
    --poll                 Use polling instead of events
    --why                  Show changed files
    -E, --env <KEY=VALUE>  Set environment variable
```

## Documentation

- GitHub: https://github.com/watchexec/cargo-watch
- crates.io: https://crates.io/crates/cargo-watch
