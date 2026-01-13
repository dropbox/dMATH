# cargo-nextest[¶][1]

Welcome to the home page for **cargo-nextest**, a next-generation test runner for Rust projects.

## Features[¶][2]

* **Clean, beautiful user interface**
  
  
  See which tests passed and failed at a glance.
  
  [ Running tests][3]

* **Up to 3x as fast as cargo test**
  
  Nextest uses a modern [execution model][4] for faster, more reliable test runs.
  
  [ Benchmarks][5]

* **Powerful test selection**
  
  Use a sophisticated [expression language][6] to select exactly the tests you need. Filter by name,
  binary, platform, or any combination.
  
  [ Filtersets][7]

* **Identify misbehaving tests**
  
  Treat tests as cattle, not pets. Detect and terminate [slow tests][8]. Loop over tests many times
  with [stress testing][9].
  
  [ Slow tests and timeouts][10]

* **Customize settings by test**
  
  Automatically [retry][11] some tests, mark them as [heavy][12], run them [serially][13], and much
  more.
  
  [ Per-test settings][14]

* **Designed for CI**
  
  [Archive][15] and [partition][16] tests across multiple workers, export [JUnit XML][17], and use
  [profiles][18] for different environments.
  
  [ Configuration profiles][19]

* **An ecosystem of tools**
  
  Collect [test coverage][20]. Do [mutation testing][21]. Spin up [debuggers][22]. Observe system
  behavior with [DTrace and bpftrace probes][23].
  
  [ Integrations][24]

* **Cross-platform**
  
  Runs on Linux, Mac, Windows, and other Unix-like systems. Download binaries or build it [from
  source][25].
  
  [ Pre-built binaries][26]

* **Open source, widely trusted**
  
  Powers Rust development at every scale, from independent open source projects to the world's
  largest tech companies.
  
  [ License (Apache 2.0)][27]

* **State-of-the-art, made with love**
  
  Nextest brings [infrastructure-grade reliability][28] to test runners, [with *care*][29] about
  getting the details right.
  
  [ Sponsor on GitHub][30]

## Quick start[¶][31]

Install cargo-nextest for your platform using the [pre-built binaries][32].

For most Rust projects, nextest works out of the box. To run all tests in a workspace:

`cargo nextest run
`

> **Note:** Doctests are currently [not supported][33] because of limitations in stable Rust. For
> now, run doctests in a separate step with `cargo test --doc`.

## Crates in this project[¶][34]

─────────────────────────────────────┬──────────────────────┬──────────────────────┬────────────────
Crate                                │crates.io             │rustdoc (latest       │rustdoc (main)  
                                     │                      │version)              │                
─────────────────────────────────────┼──────────────────────┼──────────────────────┼────────────────
**cargo-nextest,** the main test     │[[cargo-nextest on    │[[Documentation       │[[Documentation 
binary                               │crates.io]][35]       │(latest release)]][36]│(main)]][37]    
─────────────────────────────────────┼──────────────────────┼──────────────────────┼────────────────
**nextest-runner,** core nextest     │[[nextest-runner on   │[[Documentation       │[[Documentation 
logic                                │crates.io]][38]       │(latest release)]][39]│(main)]][40]    
─────────────────────────────────────┼──────────────────────┼──────────────────────┼────────────────
**nextest-metadata,** parsers for    │[[nextest-metadata on │[[Documentation       │[[Documentation 
machine-readable output              │crates.io]][41]       │(latest release)]][42]│(main)]][43]    
─────────────────────────────────────┼──────────────────────┼──────────────────────┼────────────────
**nextest-filtering,** parser and    │[[nextest-filtering on│[[Documentation       │[[Documentation 
evaluator for [filtersets][44]       │crates.io]][45]       │(latest release)]][46]│(main)]][47]    
─────────────────────────────────────┼──────────────────────┼──────────────────────┼────────────────
**quick-junit,** JUnit XML serializer│[[quick-junit on      │[[Documentation       │[[Documentation 
                                     │crates.io]][48]       │(latest release)]][49]│(main)]][50]    
─────────────────────────────────────┼──────────────────────┼──────────────────────┼────────────────
**datatest-stable,** [custom test    │[[datatest-stable on  │[[Documentation       │[[Documentation 
harness][51] for data-driven tests   │crates.io]][52]       │(latest release)]][53]│(main)]][54]    
─────────────────────────────────────┼──────────────────────┼──────────────────────┼────────────────
**future-queue,** run queued futures │[[future-queue on     │[[Documentation       │[[Documentation 
with global and group limits         │crates.io]][55]       │(latest release)]][56]│(main)]][57]    
─────────────────────────────────────┴──────────────────────┴──────────────────────┴────────────────

## Contributing[¶][58]

The source code for nextest and this site are hosted on GitHub, at
[https://github.com/nextest-rs/nextest][59].

Contributions are welcome! Please see the [CONTRIBUTING file][60] for how to help out.

## License[¶][61]

The source code for nextest is licensed under the [MIT][62] and [Apache 2.0][63] licenses.

For information about code signing, see [*Code signing policy*][64].

This document is licensed under [CC BY 4.0][65]. This means that you are welcome to share, adapt or
modify this material as long as you give appropriate credit.

November 19, 2025 February 9, 2022

[1]: #cargo-nextest
[2]: #features
[3]: docs/running/
[4]: docs/design/how-it-works/
[5]: docs/benchmarks/
[6]: docs/filtersets/
[7]: docs/filtersets/
[8]: docs/features/slow-tests/
[9]: docs/features/stress-tests/
[10]: docs/features/slow-tests/
[11]: docs/features/retries/
[12]: docs/configuration/threads-required/
[13]: docs/configuration/test-groups/
[14]: docs/configuration/per-test-overrides/
[15]: docs/ci-features/archiving/
[16]: docs/ci-features/partitioning/
[17]: docs/machine-readable/junit/
[18]: docs/configuration/#profiles
[19]: docs/configuration/#profiles
[20]: docs/integrations/test-coverage/
[21]: docs/integrations/cargo-mutants/
[22]: docs/integrations/debuggers-tracers/
[23]: docs/integrations/usdt/
[24]: docs/integrations/
[25]: docs/installation/from-source/
[26]: docs/installation/pre-built-binaries/
[27]: https://github.com/nextest-rs/nextest/blob/main/LICENSE-APACHE
[28]: docs/design/why-process-per-test/
[29]: docs/design/architecture/runner-loop/
[30]: https://github.com/sponsors/sunshowers
[31]: #quick-start
[32]: docs/installation/pre-built-binaries/
[33]: https://github.com/nextest-rs/nextest/issues/16
[34]: #crates-in-this-project
[35]: https://crates.io/crates/cargo-nextest
[36]: https://docs.rs/cargo-nextest
[37]: https://nexte.st/rustdoc/cargo_nextest
[38]: https://crates.io/crates/nextest-runner
[39]: https://docs.rs/nextest-runner
[40]: https://nexte.st/rustdoc/nextest_runner
[41]: https://crates.io/crates/nextest-metadata
[42]: https://docs.rs/nextest-metadata
[43]: https://nexte.st/rustdoc/nextest_metadata
[44]: docs/filtersets/
[45]: https://crates.io/crates/nextest-filtering
[46]: https://docs.rs/nextest-filtering
[47]: https://nexte.st/rustdoc/nextest_filtering
[48]: https://crates.io/crates/quick-junit
[49]: https://docs.rs/quick-junit
[50]: https://quick-junit.nexte.st
[51]: docs/design/custom-test-harnesses/
[52]: https://crates.io/crates/datatest-stable
[53]: https://docs.rs/datatest-stable
[54]: https://datatest-stable.nexte.st
[55]: https://crates.io/crates/future-queue
[56]: https://docs.rs/future-queue
[57]: https://nextest-rs.github.io/future-queue/rustdoc/future_queue/
[58]: #contributing
[59]: https://github.com/nextest-rs/nextest
[60]: https://github.com/nextest-rs/nextest/blob/main/CONTRIBUTING.md
[61]: #license
[62]: https://github.com/nextest-rs/nextest/blob/main/LICENSE-MIT
[63]: https://github.com/nextest-rs/nextest/blob/main/LICENSE-APACHE
[64]: docs/installation/pre-built-binaries/#code-signing-policy
[65]: https://creativecommons.org/licenses/by/4.0/
