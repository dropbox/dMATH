# cargo-audit - Security Vulnerability Scanner for Rust

cargo-audit is a tool that audits `Cargo.lock` files for crates with security vulnerabilities reported to the RustSec Advisory Database.

## Features

- **Vulnerability Detection**: Scans for known security vulnerabilities
- **Advisory Database**: Uses the RustSec Advisory Database
- **CI Integration**: Exit codes for CI/CD pipelines
- **Multiple Output Formats**: Human-readable, JSON, and SARIF
- **Ignore Advisories**: Skip specific advisories when needed

## Installation

```bash
cargo install cargo-audit --locked
```

Or with cargo-binstall:
```bash
cargo binstall cargo-audit
```

## Basic Usage

```bash
# Audit current project
cargo audit

# Audit specific Cargo.lock
cargo audit -f /path/to/Cargo.lock

# Update advisory database
cargo audit fetch

# Show version
cargo audit --version
```

## Output

```
    Fetching advisory database from `https://github.com/RustSec/advisory-db`
      Loaded 615 security advisories (from ~/.cargo/advisory-db)
    Scanning Cargo.lock for vulnerabilities (234 crate dependencies)
Crate:    smallvec
Version:  0.6.13
Warning:  unsound
Title:    Buffer overflow in SmallVec::insert_many
Date:     2021-01-08
ID:       RUSTSEC-2021-0003
URL:      https://rustsec.org/advisories/RUSTSEC-2021-0003
Solution: Upgrade to >=1.6.1

error: 1 vulnerability found!
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | No vulnerabilities found |
| 1 | Vulnerabilities detected |
| 2 | Error occurred |

## Configuration

Create `.cargo/audit.toml`:

```toml
[advisories]
# Ignore specific advisories
ignore = [
    "RUSTSEC-2020-0071",  # Reason: We don't use the affected feature
]
# Treat warnings as errors
informational_warnings = ["unmaintained"]

[database]
# Path to advisory database
path = "~/.cargo/advisory-db"
# URL to fetch from
url = "https://github.com/RustSec/advisory-db"

[output]
# Output format: terminal, json, markdown
format = "terminal"
# Show all advisories, not just vulnerabilities
quiet = false
```

## Ignoring Advisories

```toml
# .cargo/audit.toml
[advisories]
ignore = [
    "RUSTSEC-2020-0071",  # smallvec: We pin to safe version
    "RUSTSEC-2021-0124",  # chrono: Awaiting upstream fix
]
```

Or inline ignore:
```bash
cargo audit --ignore RUSTSEC-2020-0071
```

## CI Integration

### GitHub Actions

```yaml
- name: Install cargo-audit
  run: cargo install cargo-audit --locked

- name: Security audit
  run: cargo audit
```

### Pre-commit Hook

```bash
#!/bin/bash
cargo audit || exit 1
```

## Output Formats

```bash
# Default human-readable
cargo audit

# JSON output
cargo audit --json

# Markdown output
cargo audit --output markdown

# SARIF output (for GitHub Security)
cargo audit --output sarif
```

## Advanced Usage

### Check Yanked Crates

```bash
cargo audit --deny yanked
```

### Database Management

```bash
# Fetch latest advisories
cargo audit fetch

# Use local database
cargo audit --db /path/to/advisory-db

# Stale database check
cargo audit --stale
```

### Fixing Vulnerabilities

```bash
# Show upgrade paths
cargo audit fix --dry-run

# Automatically upgrade (experimental)
cargo audit fix
```

## Advisory Types

| Type | Description |
|------|-------------|
| `vulnerability` | Security vulnerability |
| `unmaintained` | Crate no longer maintained |
| `unsound` | API allows undefined behavior |
| `yanked` | Crate version was yanked |

## Integration with cargo-deny

cargo-audit focuses on security vulnerabilities, while cargo-deny provides broader policy enforcement. They complement each other:

```bash
# Security vulnerabilities
cargo audit

# License compliance + security + more
cargo deny check
```

## RustSec Advisory Database

The advisories come from https://rustsec.org/, a community-driven security advisory database for the Rust ecosystem.

## Documentation

- Official: https://rustsec.org/
- GitHub: https://github.com/rustsec/rustsec
- Advisory Database: https://github.com/RustSec/advisory-db
