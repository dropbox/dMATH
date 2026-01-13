# A vulnerability database for the Rust ecosystem

or [browse advisories][1]

## Tooling

### `cargo-audit`

Audit `Cargo.lock` files for crates with security vulnerabilities.

[Get started][2]

> cargo audit
    Scanning Cargo.lock for vulnerabilities (4 crate dependencies)
Crate:     lz4-sys
Version:   1.9.3
Title:     Memory corruption in liblz4
Date:      2022-08-25
ID:        RUSTSEC-2022-0051
URL:       https://rustsec.org/advisories/RUSTSEC-2022-0051
Solution:  Upgrade to >=1.9.4
Dependency tree:
lz4-sys 1.9.3
└── crate 0.1.0

error: 1 vulnerability found!
    

### `cargo-deny`

Audit `Cargo.lock` files for crates with security vulnerabilities, limit the usage of particular
dependencies, their licenses, sources to download from, detect multiple versions of same packages in
the dependency tree and more.

[Get started][3]

### `cargo-auditable`

Embed the dependency tree into compiled executables, to make production Rust binaries auditable by
cargo-audit.

[Get started][4]

### `cargo-audit` Github action

Audit changes, schedule dependencies audits and open issues for found vulnerabilities using
cargo-audit with the `rust-audit-check` Github action.

[Get started][5]

### `cargo-deny` Github action

Audit changes and schedule dependencies audits using cargo-deny with the `cargo-deny-action` Github
action.

[Get started][6]

## Data Interchange

We export all our data to [Open Source Vulnerabilities][7] in real time. This enables many other
tools, such as [Trivy][8], to access RustSec advisories.

You can access RustSec advisories in the OSV format either directly as a [zip archive][9] or using
the [OSV API][10].

The [Github Advisory Database][11] imports our advisories and makes them available in its [public
API][12].

This allows [dependabot][13] to fix vulnerable dependencies for you by raising pull requests with
security updates.

## About

The [RustSec Advisory Database][14] is a repository of security advisories filed against Rust crates
published via [crates.io][15] maintained by the [Rust Secure Code Working Group][16].

[1]: /advisories/
[2]: https://github.com/rustsec/rustsec/blob/main/cargo-audit/README.md
[3]: https://embarkstudios.github.io/cargo-deny/
[4]: https://github.com/rust-secure-code/cargo-auditable
[5]: https://github.com/rustsec/audit-check
[6]: https://github.com/marketplace/actions/cargo-deny
[7]: https://osv.dev/
[8]: https://aquasecurity.github.io/trivy/
[9]: https://codeload.github.com/rustsec/advisory-db/zip/refs/heads/osv
[10]: https://osv.dev/docs/
[11]: https://github.com/advisories
[12]: https://docs.github.com/en/graphql/reference/objects#securityadvisory
[13]: https://docs.github.com/en/code-security/dependabot/dependabot-security-updates/about-dependab
ot-security-updates
[14]: https://github.com/RustSec/advisory-db
[15]: https://crates.io
[16]: https://www.rust-lang.org/governance/wgs/wg-secure-code
