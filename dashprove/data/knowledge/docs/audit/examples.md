# RustSec Crates 🦀🛡️📦

The RustSec Advisory Database is a repository of security advisories filed against Rust crates
published via [crates.io][1].

The advisory database itself can be found at:

[https://github.com/RustSec/advisory-db][2]

## About this repository

This repository contains a Cargo Workspace with all of the crates maintained by the RustSec project:

─────────────────┬─────────────────────────────────────┬───────────────┬───────────────────┬────────
Name             │Description                          │Crate          │Documentation      │Build   
─────────────────┼─────────────────────────────────────┼───────────────┼───────────────────┼────────
[`cargo‑audit`][3│Audit Cargo.lock against the advisory│[[crates.io]][4│[[Documentation]][5│[[CI]][6
]                │DB                                   │]              │]                  │]       
─────────────────┼─────────────────────────────────────┼───────────────┼───────────────────┼────────
[`cargo‑lock`][7]│Self-contained Cargo.lock parser     │[[crates.io]][8│[[Documentation]][9│[[CI]][1
                 │                                     │]              │]                  │0]      
─────────────────┼─────────────────────────────────────┼───────────────┼───────────────────┼────────
[`cvss`][11]     │Common Vulnerability Scoring System  │[[crates.io]][1│[[Documentation]][1│[[CI]][1
                 │                                     │2]             │3]                 │4]      
─────────────────┼─────────────────────────────────────┼───────────────┼───────────────────┼────────
[`platforms`][15]│Rust platform registry               │[[crates.io]][1│[[Documentation]][1│[[CI]][1
                 │                                     │6]             │7]                 │8]      
─────────────────┼─────────────────────────────────────┼───────────────┼───────────────────┼────────
[`rustsec`][19]  │Advisory DB client library           │[[crates.io]][2│[[Documentation]][2│[[CI]][2
                 │                                     │0]             │1]                 │2]      
─────────────────┼─────────────────────────────────────┼───────────────┼───────────────────┼────────
[`rustsec‑admin`]│Linter and web site generator        │[[crates.io]][2│[[Documentation]][2│[[CI]][2
[23]             │                                     │4]             │5]                 │6]      
─────────────────┴─────────────────────────────────────┴───────────────┴───────────────────┴────────

## License

All crates licensed under either of

* [Apache License, Version 2.0][27]
* [MIT license][28]

at your option.

[1]: https://crates.io
[2]: https://github.com/RustSec/advisory-db
[3]: https://github.com/RustSec/rustsec/tree/main/cargo-audit
[4]: https://crates.io/crates/cargo-audit
[5]: https://docs.rs/cargo-audit
[6]: https://github.com/RustSec/rustsec/actions/workflows/cargo-audit.yml
[7]: https://github.com/RustSec/rustsec/tree/main/cargo-lock
[8]: https://crates.io/crates/cargo-lock
[9]: https://docs.rs/cargo-lock
[10]: https://github.com/RustSec/rustsec/actions/workflows/cargo-lock.yml
[11]: https://github.com/RustSec/rustsec/tree/main/cvss
[12]: https://crates.io/crates/cvss
[13]: https://docs.rs/cvss
[14]: https://github.com/RustSec/rustsec/actions/workflows/cvss.yml
[15]: https://github.com/RustSec/rustsec/tree/main/platforms
[16]: https://crates.io/crates/platforms
[17]: https://docs.rs/platforms
[18]: https://github.com/RustSec/rustsec/actions/workflows/platforms.yml
[19]: https://github.com/RustSec/rustsec/tree/main/rustsec
[20]: https://crates.io/crates/rustsec
[21]: https://docs.rs/rustsec
[22]: https://github.com/RustSec/rustsec/actions/workflows/rustsec.yml
[23]: https://github.com/RustSec/rustsec/tree/main/admin
[24]: https://crates.io/crates/rustsec-admin
[25]: https://docs.rs/rustsec-admin
[26]: https://github.com/RustSec/rustsec/actions/workflows/admin.yml
[27]: http://www.apache.org/licenses/LICENSE-2.0
[28]: http://opensource.org/licenses/MIT
