# `❌ cargo-deny`

**Cargo plugin for linting your dependencies**

[[Embark Opensource]][1] [[Embark Discord]][2] [[Crates.io]][3] [[API Docs]][4] [[Docs]][5]
[[Minimum Stable Rust Version]][6] [[SPDX Version]][7] [[dependency status]][8] [[Build Status]][9]

See the [book 📕][10] for in-depth documentation.

To run on CI as a GitHub Action, see [`cargo-deny-action`][11].

*Please Note: This is a tool that we use (and like!) and it makes sense to us to release it as open
source. However, we can’t take any responsibility for your use of the tool, if it will function
correctly or fulfil your needs. No functionality in - or information provided by - cargo-deny
constitutes legal advice.*

## [Quickstart][12]

cargo install --locked cargo-deny && cargo deny init && cargo deny check

## Usage

[[Packaging status]][13]

### [Install][14] cargo-deny

If you want to use `cargo-deny` without having `cargo` installed, build `cargo-deny` with the
`standalone` feature. This can be useful in Docker Images.

cargo install --locked cargo-deny

# Or, if you're an Arch user
pacman -S cargo-deny

### [Initialize][15] your project

cargo deny init

### [Check][16] your crates

cargo deny check

#### [Licenses][17]

The licenses check is used to verify that every crate you use has license terms you find acceptable.

cargo deny check licenses

[[licenses output]][18]

#### [Bans][19]

The bans check is used to deny (or allow) specific crates, as well as detect and handle multiple
versions of the same crate.

cargo deny check bans

[[bans output]][20]

#### [Advisories][21]

The advisories check is used to detect issues for crates by looking in an advisory database.

cargo deny check advisories

[[advisories output]][22]

#### [Sources][23]

The sources check ensures crates only come from sources you trust.

cargo deny check sources

[[sources output]][24]

### Pre-commit hook

You can use `cargo-deny` with [pre-commit][25]. Add it to your local `.pre-commit-config.yaml` as
follows:

- repo: https://github.com/EmbarkStudios/cargo-deny
  rev: 0.14.16 # choose your preferred tag
  hooks:
    - id: cargo-deny
      args: ["--all-features", "check"] # optionally modify the arguments for cargo-deny (default ar
guments shown here)

## Contributing

[[Contributor Covenant]][26]

We welcome community contributions to this project.

Please read our [Contributor Guide][27] for more information on how to get started.

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE][28] or
  [http://www.apache.org/licenses/LICENSE-2.0][29])
* MIT license ([LICENSE-MIT][30] or [http://opensource.org/licenses/MIT][31])

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

[1]: https://embark.dev
[2]: https://discord.gg/Fg4u4VX
[3]: https://crates.io/crates/cargo-deny
[4]: https://docs.rs/cargo-deny
[5]: https://embarkstudios.github.io/cargo-deny/
[6]: https://blog.rust-lang.org/2025/06/26/Rust-1.88.0/
[7]: https://spdx.org/licenses/
[8]: https://deps.rs/repo/github/EmbarkStudios/cargo-deny
[9]: https://github.com/EmbarkStudios/cargo-deny/actions?workflow=CI
[10]: https://embarkstudios.github.io/cargo-deny/
[11]: https://github.com/EmbarkStudios/cargo-deny-action
[12]: https://embarkstudios.github.io/cargo-deny/
[13]: https://repology.org/project/cargo-deny/versions
[14]: https://embarkstudios.github.io/cargo-deny/cli/index.html
[15]: https://embarkstudios.github.io/cargo-deny/cli/init.html
[16]: https://embarkstudios.github.io/cargo-deny/cli/check.html
[17]: https://embarkstudios.github.io/cargo-deny/checks/licenses/index.html
[18]: /EmbarkStudios/cargo-deny/blob/main/docs/src/output/licenses.svg
[19]: https://embarkstudios.github.io/cargo-deny/checks/bans/index.html
[20]: /EmbarkStudios/cargo-deny/blob/main/docs/src/output/bans.svg
[21]: https://embarkstudios.github.io/cargo-deny/checks/advisories/index.html
[22]: /EmbarkStudios/cargo-deny/blob/main/docs/src/output/advisories.svg
[23]: https://embarkstudios.github.io/cargo-deny/checks/sources/index.html
[24]: /EmbarkStudios/cargo-deny/blob/main/docs/src/output/sources.svg
[25]: https://pre-commit.com
[26]: /EmbarkStudios/cargo-deny/blob/main/CODE_OF_CONDUCT.md
[27]: /EmbarkStudios/cargo-deny/blob/main/CONTRIBUTING.md
[28]: /EmbarkStudios/cargo-deny/blob/main/LICENSE-APACHE
[29]: http://www.apache.org/licenses/LICENSE-2.0
[30]: /EmbarkStudios/cargo-deny/blob/main/LICENSE-MIT
[31]: http://opensource.org/licenses/MIT
