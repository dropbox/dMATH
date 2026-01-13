# cargo-geiger ☢️

[[CI]][1] [[unsafe forbidden]][2] [[crates.io]][3] [[Crates.io]][4]

A tool that lists statistics related to the usage of unsafe Rust code in a Rust crate and all its
dependencies.

This cargo plugin was originally based on the code from two other projects:

* [https://github.com/icefoxen/cargo-osha][5] and
* [https://github.com/sfackler/cargo-tree][6]

## Installation

Try to find and use a system-wide installed OpenSSL library:

cargo install --locked cargo-geiger

Or, build and statically link OpenSSL as part of the cargo-geiger executable:

cargo install --locked cargo-geiger --features vendored-openssl

Alternatively pre-built binary releases are available from [GitHub releases][7].

## Usage

1. Navigate to the same directory as the `Cargo.toml` you want to analyze.
2. `cargo geiger`

## Intended Use

This tool is not meant to advise directly whether the code ultimately is truly insecure or not.

The purpose of cargo-geiger is to provide statistical input to auditing e.g. with:

* [cargo-crev][8]
* [safety-dance][9]

The use of unsafe is nuanced and necessary in some cases and any motivation to use it is outside the
scope of cargo-geiger.

It is important that any reporting is handled with care:

* [Reddit: The Stigma around Unsafe][10]
* [YouTube: Rust NYC: Jon Gjengset - Demystifying unsafe code][11]
* [Rust-lang: WG Unsafe Code Guidelines][12]

## Output example

[[Example output]][13]

## Known issues

* See the [issue tracker][14].

## Libraries

Cargo Geiger exposes three libraries:

* `cargo-geiger` - Unversioned and highly unstable library exposing the internals of the
  `cargo-geiger` binary. As such, any function contained within this library may be subject to
  change.
* `cargo-geiger-serde` - A library containing the serializable report types
* `geiger` - A library containing a few decoupled [cargo][15] components used by [cargo-geiger][16]

## Changelog

See the [changelog][17].

## Why the name?

[https://en.wikipedia.org/wiki/Geiger_counter][18]

Unsafe code, like ionizing radiation, is unavoidable in some situations and should be safely
contained!

[1]: https://github.com/geiger-rs/cargo-geiger/actions/workflows/ci.yml
[2]: https://github.com/rust-secure-code/safety-dance/
[3]: https://crates.io/crates/cargo-geiger
[4]: https://crates.io/crates/cargo-geiger
[5]: https://github.com/icefoxen/cargo-osha
[6]: https://github.com/sfackler/cargo-tree
[7]: https://github.com/geiger-rs/cargo-geiger/releases
[8]: https://crates.io/crates/cargo-crev
[9]: https://github.com/rust-secure-code/safety-dance
[10]: https://www.reddit.com/r/rust/comments/y1u068/the_stigma_around_unsafe/
[11]: https://youtu.be/QAz-maaH0KM
[12]: https://github.com/rust-lang/unsafe-code-guidelines
[13]: https://user-images.githubusercontent.com/3704611/53132247-845f7080-356f-11e9-9c76-a9498d4a744
b.png
[14]: https://github.com/rust-secure-code/cargo-geiger/issues
[15]: https://crates.io/crates/cargo
[16]: https://crates.io/crates/cargo-geiger
[17]: https://github.com/rust-secure-code/cargo-geiger/blob/master/CHANGELOG.md
[18]: https://en.wikipedia.org/wiki/Geiger_counter
