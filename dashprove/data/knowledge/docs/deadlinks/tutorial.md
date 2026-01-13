# cargo-deadlinks • [[Crates.io]][1] [[License]][2]

Check your `cargo doc` documentation for broken links!

Useful if you just refactored the structure of your crate or want to ensure that your documentation
is readable offline.

This crate pairs well with [intra-doc links][3] and [`cargo-intraconv`][4], which make it easier to
write links without having to know the exact HTML page rustdoc will generate.

`deadlinks` can also be used on projects that aren't Rust crates.

## Installation

Install cargo-deadlinks via:

cargo install cargo-deadlinks

Alternatively, install pre-built binaries from the [releases page][5].

## Usage

From your packages directory run:

# this will automatically run `cargo doc` for you
# any broken links will show up in the output
cargo deadlinks
# if you also want to check http and https links
cargo deadlinks --check-http

By default `cargo deadlinks` will check only the offline (`file://`) links of your package.

If you want to check the documentation in another directory e.g. to check all your dependencies, you
can provide the `--dir` argument:

cargo deadlinks --dir target/doc

To check a project that isn't a crate, use `deadlinks` instead:

# this requires that you already have a static site in build/html
deadlinks build/html

For information about other arguments run `cargo deadlinks --help`.

## Minimum Supported Rust Version (MSRV)

The current MSRV is **1.45.0**. This may be increased in minor versions, but will never increase in
a patch version.

## Contributing

We are happy about any contributions!

To get started you can take a look at our [Github issues][6].

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you, as defined in the Apache-2.0 license, shall be dual licensed as below, without any
additional terms or conditions.

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE][7] or
  [http://www.apache.org/licenses/LICENSE-2.0][8])
* MIT license ([LICENSE-MIT][9] or [http://opensource.org/licenses/MIT][10])

at your option.

[1]: https://crates.io/crates/cargo-deadlinks
[2]: https://camo.githubusercontent.com/91d4d0284c6d45490bcffd58049302441216eba7ebc4dd65e91b858dc917
8693/68747470733a2f2f696d672e736869656c64732e696f2f6372617465732f6c2f636172676f2d646561646c696e6b732
e737667
[3]: https://doc.rust-lang.org/rustdoc/linking-to-items-by-name.html
[4]: https://github.com/poliorcetics/cargo-intraconv
[5]: https://github.com/deadlinks/cargo-deadlinks/releases
[6]: https://github.com/deadlinks/cargo-deadlinks/issues
[7]: /deadlinks/cargo-deadlinks/blob/master/LICENSE-APACHE
[8]: http://www.apache.org/licenses/LICENSE-2.0
[9]: /deadlinks/cargo-deadlinks/blob/master/LICENSE-MIT
[10]: http://opensource.org/licenses/MIT
