# Nextest

Nextest is a next-generation test runner for Rust. For more information, **check out [the
website][1]**.

This repository contains the source code for:

* [**cargo-nextest**][2]: a new, faster Cargo test runner [[cargo-nextest on crates.io]][3]
  [[Documentation (website)]][4]
* libraries used by cargo-nextest:
  
  * [**nextest-runner**][5]: core logic for cargo-nextest [[nextest-runner on crates.io]][6]
    [[Documentation (latest release)]][7] [[Documentation (main)]][8]
  * [**nextest-metadata**][9]: library for calling cargo-nextest over the command line
    [[nextest-metadata on crates.io]][10] [[Documentation (latest release)]][11] [[Documentation
    (main)]][12]
  * [**nextest-filtering**][13]: parser and evaluator for [filtersets][14] [[nextest-filtering on
    crates.io]][15] [[Documentation (latest release)]][16] [[Documentation (main)]][17]

## Minimum supported Rust version

The minimum supported Rust version to *run* nextest with is **Rust 1.41.** Nextest is not tested
against versions that are that old, but it should work with any version of Rust released in the past
year. (Please report a bug if not!)

The minimum supported Rust version to *build* nextest with is **Rust 1.89.** For building, at least
the last 3 versions of stable Rust are supported at any given time.

See the [stability policy][18] for more details.

While a crate is pre-release status (0.x.x) it may have its MSRV bumped in a patch release. Once a
crate has reached 1.x, any MSRV bump will be accompanied with a new minor version.

## Contributing

See the [CONTRIBUTING][19] file for how to help out.

*Looking to contribute to nextest and don't know where to get started?* Check out the list of [good
first issues][20].

## License

**Nextest is Free Software.** This project is available under the terms of either the [Apache 2.0
license][21] or the [MIT license][22].

**Like all Free Software, nextest is a gift.** Nextest is provided on an "AS IS" basis and there is
NO WARRANTY attached to it. As a user, please treat the authors and contributors to this project as
if you were treating the giver of a gift. In particular, you're asked to follow the [code of
conduct][23].

This project is derived from [diem-devtools][24]. Upstream source code is used under the terms of
the [Apache 2.0 license][25] and the [MIT license][26].

## macOS support

macOS is supported through the MacStadium Open Source Developer Program.

[[image]][27]

[1]: https://nexte.st/
[2]: /nextest-rs/nextest/blob/main/cargo-nextest
[3]: https://crates.io/crates/cargo-nextest
[4]: https://nexte.st
[5]: /nextest-rs/nextest/blob/main/nextest-runner
[6]: https://crates.io/crates/nextest-runner
[7]: https://docs.rs/nextest-runner
[8]: https://nexte.st/rustdoc/nextest_runner/
[9]: /nextest-rs/nextest/blob/main/nextest-metadata
[10]: https://crates.io/crates/nextest-metadata
[11]: https://docs.rs/nextest-metadata
[12]: https://nexte.st/rustdoc/nextest_metadata
[13]: /nextest-rs/nextest/blob/main/nextest-filtering
[14]: https://nexte.st/docs/filtersets
[15]: https://crates.io/crates/nextest-filtering
[16]: https://docs.rs/nextest-filtering
[17]: https://nexte.st/rustdoc/nextest_filtering
[18]: https://nexte.st/docs/stability/
[19]: /nextest-rs/nextest/blob/main/CONTRIBUTING.md
[20]: https://github.com/nextest-rs/nextest/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%
3A%22good+first+issue%22
[21]: /nextest-rs/nextest/blob/main/LICENSE-APACHE
[22]: /nextest-rs/nextest/blob/main/LICENSE-MIT
[23]: /nextest-rs/nextest/blob/main/CODE_OF_CONDUCT.md
[24]: https://github.com/diem/diem-devtools/
[25]: https://github.com/diem/diem-devtools/blob/main/LICENSE-APACHE
[26]: https://github.com/diem/diem-devtools/blob/main/LICENSE-MIT
[27]: https://camo.githubusercontent.com/b20304c7ddfd6e1fdc24357be1a4bbe0b367fd7f533d003a805cdde7784
13246/68747470733a2f2f75706c6f6164732d73736c2e776562666c6f772e636f6d2f356163336330343663383237323439
3730666336303931382f3563303139643931376262613331326166373535336234395f4d61635374616469756d2d64657665
6c6f7065726c6f676f2e706e67
