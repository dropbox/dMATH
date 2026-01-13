# Crate insta Copy item path

[Source][1]
Expand description

**insta: a snapshot testing library for Rust**

## [§][2]What are snapshot tests

Snapshots tests (also sometimes called approval tests) are tests that assert values against a
reference value (the snapshot). This is similar to how [`assert_eq!`][3] lets you compare a value
against a reference value but unlike simple string assertions, snapshot tests let you test against
complex values and come with comprehensive tools to review changes.

Snapshot tests are particularly useful if your reference values are very large or change often.

## [§][4]What it looks like:

`#[test]
fn test_hello_world() {
    insta::assert_debug_snapshot!(vec![1, 2, 3]);
}`

Where are the snapshots stored? Right next to your test in a folder called `snapshots` as individual
[`.snap` files][5].

Got curious?

* [Read the introduction][6]
* [Read the main documentation][7] which does not just cover the API of the crate but also many of
  the details of how it works.
* There is a screencast that shows the entire workflow: [watch the insta introduction
  screencast][8].

## [§][9]Writing Tests

`use insta::assert_debug_snapshot;

#[test]
fn test_snapshots() {
    assert_debug_snapshot!(vec![1, 2, 3]);
}`

The recommended flow is to run the tests once, have them fail and check if the result is okay. By
default, the new snapshots are stored next to the old ones with the extra `.new` extension. Once you
are satisfied move the new files over. To simplify this workflow you can use `cargo insta review`
(requires [`cargo-insta`][10]) which will let you interactively review them:

`$ cargo test
$ cargo insta review`

## [§][11]Use Without `cargo-insta`

Note that `cargo-insta` is entirely optional. You can also just use insta directly from `cargo test`
and control it via the `INSTA_UPDATE` environment variable — see [Updating snapshots][12] for
details.

You can for instance first run the tests and not write any new snapshots, and if you like them run
the tests again and update them:

`INSTA_UPDATE=no cargo test
INSTA_UPDATE=always cargo test`

## [§][13]Assertion Macros

This crate exports multiple macros for snapshot testing:

* [`assert_snapshot!`][14] for comparing basic snapshots of [`Display`][15] outputs, often strings.
* [`assert_debug_snapshot!`][16] for comparing [`Debug`][17] outputs of values.

The following macros require the use of [`serde::Serialize`][18]:

* [`assert_csv_snapshot!`][19] for comparing CSV serialized output. (requires the `csv` feature)
* [`assert_toml_snapshot!`][20] for comparing TOML serialized output. (requires the `toml` feature)
* [`assert_yaml_snapshot!`][21] for comparing YAML serialized output. (requires the `yaml` feature)
* [`assert_ron_snapshot!`][22] for comparing RON serialized output. (requires the `ron` feature)
* [`assert_json_snapshot!`][23] for comparing JSON serialized output. (requires the `json` feature)
* [`assert_compact_json_snapshot!`][24] for comparing JSON serialized output while preferring
  single-line formatting. (requires the `json` feature)

For macros that work with [`serde`][25] this crate also permits redacting of partial values. See
[redactions in the documentation][26] for more information.

## [§][27]Updating snapshots

During test runs snapshots will be updated according to the `INSTA_UPDATE` environment variable. The
default is `auto` which will write snapshots for any failing tests into `.snap.new` files (if no CI
is detected) so that [`cargo-insta`][28] can pick them up for review. Normally you don’t have to
change this variable.

`INSTA_UPDATE` modes:

* `auto`: the default. `no` for CI environments or `new` otherwise
* `new`: writes snapshots for any failing tests into `.snap.new` files, pending review
* `always`: writes snapshots for any failing tests into `.snap` files, bypassing review
* `unseen`: `always` for previously unseen snapshots or `new` for existing snapshots
* `no`: does not write to snapshot files at all; just runs tests
* `force`: forcibly updates snapshot files, even if assertions pass

When `new`, `auto` or `unseen` is used, the [`cargo-insta`][29] command can be used to review the
snapshots conveniently:

`$ cargo insta review`

“enter” or “a” accepts a new snapshot, “escape” or “r” rejects, “space” or “s” skips the snapshot
for now.

For more information [read the cargo insta docs][30].

## [§][31]Inline Snapshots

Additionally snapshots can also be stored inline. In that case the format for the snapshot macros is
`assert_snapshot!(reference_value, @"snapshot")`. The leading at sign (`@`) indicates that the
following string is the reference value. On review, `cargo-insta` will update the string with the
new value.

Example:

`assert_snapshot!(2 + 2, @"");`

Like with normal snapshots, an initial test failure will write the proposed value into a draft file
(note that inline snapshots use `.pending-snap` files rather than `.snap.new` files). Running `cargo
insta review` will review the proposed changes and update the source files on acceptance
automatically.

## [§][32]Features

The following features exist:

* `csv`: enables CSV support (via [`serde`][33])
* `json`: enables JSON support (via [`serde`][34])
* `ron`: enables RON support (via [`serde`][35])
* `toml`: enables TOML support (via [`serde`][36])
* `yaml`: enables YAML support (via [`serde`][37])
* `redactions`: enables support for redactions
* `filters`: enables support for filters
* `glob`: enables support for globbing ([`glob!`][38])
* `colors`: enables color output (enabled by default)

For legacy reasons the `json` and `yaml` features are enabled by default in limited capacity. You
will receive a deprecation warning if you are not opting into them but for now the macros will
continue to function.

Enabling any of the [`serde`][39] based formats enables the hidden `serde` feature which gates some
[`serde`][40] specific APIs such as [`Settings::set_info`][41].

## [§][42]Dependencies

[`insta`][43] tries to be light in dependencies but this is tricky to accomplish given what it tries
to do. By default, it currently depends on [`serde`][44] for the [`assert_toml_snapshot!`][45] and
[`assert_yaml_snapshot!`][46] macros. In the future this default dependencies will be removed. To
already benefit from this optimization you can disable the default features and manually opt into
what you want.

## [§][47]Settings

There are some settings that can be changed on a per-thread (and thus per-test) basis. For more
information see [Settings][48].

Additionally, Insta will load a YAML config file with settings that change the behavior of insta
between runs. It’s loaded from any of the following locations: `.config/insta.yaml`, `insta.yaml`
and `.insta.yaml` from the workspace root. The following config options exist:

`behavior:
  # also set by INSTA_REQUIRE_FULL_MATCH
  require_full_match: true/false
  # also set by INSTA_FORCE_PASS
  force_pass: true/false
  # also set by INSTA_OUTPUT
  output: "diff" | "summary" | "minimal" | "none"
  # also set by INSTA_UPDATE
  update: "auto" | "new" | "always" | "no" | "unseen" | "force"
  # also set by INSTA_GLOB_FAIL_FAST
  glob_fail_fast: true/false

# these are used by cargo insta test
test:
  # also set by INSTA_TEST_RUNNER
  # cargo-nextest binary path can be explicitly set by INSTA_CARGO_NEXTEST_BIN
  runner: "auto" | "cargo-test" | "nextest"
  # whether to fallback to `cargo-test` if `nextest` is not available,
  # also set by INSTA_TEST_RUNNER_FALLBACK, default false
  test_runner_fallback: true/false
  # disable running doctests separately when using nextest
  disable_nextest_doctest: true/false
  # automatically assume --review was passed to cargo insta test
  auto_review: true/false
  # automatically assume --accept-unseen was passed to cargo insta test
  auto_accept_unseen: true/false

# these are used by cargo insta review
review:
  # also look for snapshots in ignored folders
  include_ignored: true / false
  # also look for snapshots in hidden folders
  include_hidden: true / false
  # show a warning if undiscovered (ignored or hidden) snapshots are found.
  # defaults to true but creates a performance hit.
  warn_undiscovered: true / false`

## [§][49]External Diff Tools

By default, insta displays diffs inline in unified format. You can configure an external diff tool
via the `INSTA_DIFF_TOOL` environment variable. When set, insta writes the old and new snapshot
contents to temporary files and invokes your diff tool with those files as arguments.

`# Use delta for syntax-highlighted diffs
export INSTA_DIFF_TOOL=delta

# With arguments
export INSTA_DIFF_TOOL="delta --side-by-side"

# Or any other diff tool
export INSTA_DIFF_TOOL=difftastic`

This is a user-level setting (not project-level) since diff tool preference varies by developer. The
tool is invoked as `<tool> [args...] <old_file> <new_file>`. If the tool fails to run, insta falls
back to the built-in diff.

## [§][50]Optional: Faster Runs

Insta benefits from being compiled in release mode, even as dev dependency. It will compile slightly
slower once, but use less memory, have faster diffs and just generally be more fun to use. To
achieve that, opt [`insta`][51] and [`similar`][52] (the diffing library) into higher optimization
in your `Cargo.toml`:

`[profile.dev.package.insta]
opt-level = 3

[profile.dev.package.similar]
opt-level = 3`

You can also disable the default features of [`insta`][53] which will cut down on the compile time a
bit by removing some quality of life features.

## Modules[§][54]

*[internals][55]*
  Exposes some library internals.

## Macros[§][56]

*[allow_duplicates][57]*
  Utility macro to permit a multi-snapshot run where all snapshots match.
*[assert_binary_snapshot][58]*
  (Experimental) Asserts a binary snapshot in the form of a [`Vec<u8>`][59].
*[assert_compact_debug_snapshot][60]*
  Asserts a [`Debug`][61] snapshot in compact format.
*[assert_compact_json_snapshot][62]`json`*
  Asserts a [`serde::Serialize`][63] snapshot in compact JSON format.
*[assert_csv_snapshot][64]`csv`*
  Asserts a [`serde::Serialize`][65] snapshot in CSV format.
*[assert_debug_snapshot][66]*
  Asserts a [`Debug`][67] snapshot.
*[assert_display_snapshot][68]Deprecated*
  Asserts a [`Display`][69] snapshot.
*[assert_json_snapshot][70]`json`*
  Asserts a [`serde::Serialize`][71] snapshot in JSON format.
*[assert_ron_snapshot][72]`ron`*
  Asserts a [`serde::Serialize`][73] snapshot in RON format.
*[assert_snapshot][74]*
  Asserts a [`String`][75] snapshot.
*[assert_toml_snapshot][76]`toml`*
  Asserts a [`serde::Serialize`][77] snapshot in TOML format.
*[assert_yaml_snapshot][78]`yaml`*
  Asserts a [`serde::Serialize`][79] snapshot in YAML format.
*[elog][80]*
*[glob][81]`glob`*
  Executes a closure for all input files matching a glob.
*[with_settings][82]*
  Settings configuration macro.

## Structs[§][83]

*[MetaData][84]*
  Snapshot metadata information.
*[Settings][85]*
  Configures how insta operates at test time.
*[Snapshot][86]*
  A helper to work with file snapshots.

## Enums[§][87]

*[TextSnapshotKind][88]*

## Functions[§][89]

*[dynamic_redaction][90]`redactions`*
  Creates a dynamic redaction.
*[rounded_redaction][91]`redactions`*
  Creates a redaction that rounds floating point numbers to a given number of decimal places.
*[sorted_redaction][92]`redactions`*
  Creates a dynamic redaction that sorts the value at the selector.

[1]: ../src/insta/lib.rs.html#1-393
[2]: #what-are-snapshot-tests
[3]: https://doc.rust-lang.org/nightly/core/macro.assert_eq.html
[4]: #what-it-looks-like
[5]: https://insta.rs/docs/snapshot-files/
[6]: https://insta.rs/docs/quickstart/
[7]: https://insta.rs/docs/
[8]: https://www.youtube.com/watch?v=rCHrMqE4JOY&feature=youtu.be
[9]: #writing-tests
[10]: https://crates.io/crates/cargo-insta
[11]: #use-without-cargo-insta
[12]: #updating-snapshots
[13]: #assertion-macros
[14]: macro.assert_snapshot.html
[15]: https://doc.rust-lang.org/nightly/core/fmt/trait.Display.html
[16]: macro.assert_debug_snapshot.html
[17]: https://doc.rust-lang.org/nightly/core/fmt/macros/derive.Debug.html
[18]: https://docs.rs/serde_core/1.0.228/x86_64-unknown-linux-gnu/serde_core/ser/trait.Serialize.htm
l
[19]: macro.assert_csv_snapshot.html
[20]: macro.assert_toml_snapshot.html
[21]: macro.assert_yaml_snapshot.html
[22]: macro.assert_ron_snapshot.html
[23]: macro.assert_json_snapshot.html
[24]: macro.assert_compact_json_snapshot.html
[25]: https://docs.rs/serde/1.0.228/x86_64-unknown-linux-gnu/serde/index.html
[26]: https://insta.rs/docs/redactions/
[27]: #updating-snapshots
[28]: https://crates.io/crates/cargo-insta
[29]: https://crates.io/crates/cargo-insta
[30]: https://insta.rs/docs/cli/
[31]: #inline-snapshots
[32]: #features
[33]: https://docs.rs/serde/1.0.228/x86_64-unknown-linux-gnu/serde/index.html
[34]: https://docs.rs/serde/1.0.228/x86_64-unknown-linux-gnu/serde/index.html
[35]: https://docs.rs/serde/1.0.228/x86_64-unknown-linux-gnu/serde/index.html
[36]: https://docs.rs/serde/1.0.228/x86_64-unknown-linux-gnu/serde/index.html
[37]: https://docs.rs/serde/1.0.228/x86_64-unknown-linux-gnu/serde/index.html
[38]: macro.glob.html
[39]: https://docs.rs/serde/1.0.228/x86_64-unknown-linux-gnu/serde/index.html
[40]: https://docs.rs/serde/1.0.228/x86_64-unknown-linux-gnu/serde/index.html
[41]: struct.Settings.html#method.set_info
[42]: #dependencies
[43]: https://docs.rs/insta
[44]: https://docs.rs/serde/1.0.228/x86_64-unknown-linux-gnu/serde/index.html
[45]: macro.assert_toml_snapshot.html
[46]: macro.assert_yaml_snapshot.html
[47]: #settings-1
[48]: struct.Settings.html
[49]: #external-diff-tools
[50]: #optional-faster-runs
[51]: https://docs.rs/insta
[52]: https://docs.rs/similar/2.7.0/x86_64-unknown-linux-gnu/similar/index.html
[53]: https://docs.rs/insta
[54]: #modules
[55]: internals/index.html
[56]: #macros
[57]: macro.allow_duplicates.html
[58]: macro.assert_binary_snapshot.html
[59]: https://doc.rust-lang.org/nightly/alloc/vec/struct.Vec.html
[60]: macro.assert_compact_debug_snapshot.html
[61]: https://doc.rust-lang.org/nightly/core/fmt/macros/derive.Debug.html
[62]: macro.assert_compact_json_snapshot.html
[63]: https://docs.rs/serde_core/1.0.228/x86_64-unknown-linux-gnu/serde_core/ser/trait.Serialize.htm
l
[64]: macro.assert_csv_snapshot.html
[65]: https://docs.rs/serde_core/1.0.228/x86_64-unknown-linux-gnu/serde_core/ser/trait.Serialize.htm
l
[66]: macro.assert_debug_snapshot.html
[67]: https://doc.rust-lang.org/nightly/core/fmt/macros/derive.Debug.html
[68]: macro.assert_display_snapshot.html
[69]: https://doc.rust-lang.org/nightly/core/fmt/trait.Display.html
[70]: macro.assert_json_snapshot.html
[71]: https://docs.rs/serde_core/1.0.228/x86_64-unknown-linux-gnu/serde_core/ser/trait.Serialize.htm
l
[72]: macro.assert_ron_snapshot.html
[73]: https://docs.rs/serde_core/1.0.228/x86_64-unknown-linux-gnu/serde_core/ser/trait.Serialize.htm
l
[74]: macro.assert_snapshot.html
[75]: https://doc.rust-lang.org/nightly/alloc/string/struct.String.html
[76]: macro.assert_toml_snapshot.html
[77]: https://docs.rs/serde_core/1.0.228/x86_64-unknown-linux-gnu/serde_core/ser/trait.Serialize.htm
l
[78]: macro.assert_yaml_snapshot.html
[79]: https://docs.rs/serde_core/1.0.228/x86_64-unknown-linux-gnu/serde_core/ser/trait.Serialize.htm
l
[80]: macro.elog.html
[81]: macro.glob.html
[82]: macro.with_settings.html
[83]: #structs
[84]: struct.MetaData.html
[85]: struct.Settings.html
[86]: struct.Snapshot.html
[87]: #enums
[88]: enum.TextSnapshotKind.html
[89]: #functions
[90]: fn.dynamic_redaction.html
[91]: fn.rounded_redaction.html
[92]: fn.sorted_redaction.html
