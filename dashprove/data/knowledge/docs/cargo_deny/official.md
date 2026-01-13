# cargo-deny - Cargo Plugin for Linting Dependencies

cargo-deny is a cargo plugin that lets you lint your project's dependency graph to ensure all dependencies conform to your requirements.

## Features

- **License Compliance**: Check that dependencies use allowed licenses
- **Security Vulnerabilities**: Detect crates with known vulnerabilities
- **Bans**: Prevent specific crates or versions
- **Sources**: Restrict which registries/sources are allowed
- **Duplicate Detection**: Find multiple versions of the same crate

## Installation

```bash
cargo install cargo-deny --locked
```

Or with cargo-binstall:
```bash
cargo binstall cargo-deny
```

## Basic Usage

```bash
# Check all policies
cargo deny check

# Check specific advisories
cargo deny check advisories

# Check licenses only
cargo deny check licenses

# Check bans only
cargo deny check bans

# Check sources only
cargo deny check sources
```

## Configuration

Create `deny.toml` in your project root:

```toml
# General configuration
[graph]
targets = [
    "x86_64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc",
]
all-features = true

# License configuration
[licenses]
version = 2
allow = [
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Zlib",
    "MPL-2.0",
]
confidence-threshold = 0.8

# Allowed exceptions
[[licenses.clarify]]
name = "ring"
expression = "ISC AND MIT AND OpenSSL"
license-files = [{ path = "LICENSE", hash = 0xbd0eed23 }]

# Security advisories
[advisories]
version = 2
db-path = "~/.cargo/advisory-db"
db-urls = ["https://github.com/rustsec/advisory-db"]
vulnerability = "deny"
unmaintained = "warn"
yanked = "warn"
notice = "warn"
ignore = []

# Banned crates
[bans]
multiple-versions = "warn"
wildcards = "allow"
highlight = "all"

# Deny specific crates
[[bans.deny]]
name = "openssl"
reason = "Use rustls instead"

# Allow specific duplicates
[[bans.skip]]
name = "windows-sys"
version = "*"

# Sources configuration
[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
```

## License Checking

### Allowed Licenses

```toml
[licenses]
allow = [
    "MIT",
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Zlib",
    "CC0-1.0",
    "Unlicense",
]
```

### SPDX Expressions

```toml
[licenses]
allow = ["MIT OR Apache-2.0"]

# Clarify complex licenses
[[licenses.clarify]]
name = "unicode-ident"
expression = "(MIT OR Apache-2.0) AND Unicode-DFS-2016"
```

### License Exceptions

```toml
[[licenses.exceptions]]
name = "some-crate"
allow = ["LGPL-3.0"]
```

## Ban Configuration

### Deny Specific Crates

```toml
[[bans.deny]]
name = "openssl"
reason = "Security: Use rustls instead"

[[bans.deny]]
name = "chrono"
version = "<0.4.20"
reason = "Known vulnerability in older versions"
```

### Allow Specific Duplicates

```toml
[[bans.skip]]
name = "windows-sys"
version = "*"
reason = "Different ecosystem crates use different versions"

[[bans.skip-tree]]
name = "windows"
reason = "Skip entire windows dependency tree"
```

### Wrappers

```toml
[[bans.deny]]
name = "openssl-sys"
wrappers = ["native-tls"]  # Only allow through native-tls
```

## Advisory Configuration

```toml
[advisories]
vulnerability = "deny"      # Fail on vulnerabilities
unmaintained = "warn"       # Warn on unmaintained crates
yanked = "deny"            # Fail on yanked crates
notice = "warn"            # Warn on informational notices

# Ignore specific advisories
ignore = [
    "RUSTSEC-2020-0071",   # Reason documented here
]
```

## CI Integration

### GitHub Actions

```yaml
- name: Install cargo-deny
  uses: taiki-e/install-action@cargo-deny

- name: Check dependencies
  run: cargo deny check
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All checks passed |
| 1 | One or more checks failed |
| 2 | Error occurred |

## Commands

```bash
# Initialize deny.toml
cargo deny init

# Check all policies
cargo deny check

# Check specific policy
cargo deny check licenses
cargo deny check advisories
cargo deny check bans
cargo deny check sources

# List all licenses
cargo deny list

# Fetch advisory database
cargo deny fetch
```

## Output Formats

```bash
# Default terminal output
cargo deny check

# JSON output
cargo deny check --format json

# Specify log level
cargo deny -L debug check
```

## Best Practices

1. **Start Permissive**: Begin with warnings, then tighten to deny
2. **Document Exceptions**: Always include reasons for exceptions
3. **Regular Updates**: Keep advisory database fresh
4. **CI Integration**: Run on every PR
5. **Version Control**: Commit deny.toml

## Documentation

- Official: https://embarkstudios.github.io/cargo-deny/
- GitHub: https://github.com/EmbarkStudios/cargo-deny
- Configuration: https://embarkstudios.github.io/cargo-deny/checks/index.html
