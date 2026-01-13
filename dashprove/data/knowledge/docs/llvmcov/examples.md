# cargo-llvm-cov

[[crates.io]][1] [[license]][2] [[github actions]][3]

Cargo subcommand to easily use LLVM source-based code coverage.

This is a wrapper around rustc [`-C instrument-coverage`][4] and provides:

* Generate very precise coverage data. (line, region, and branch coverage. branch coverage is
  currently optional and requires nightly, see [#8][5] for more)
* Support `cargo test`, `cargo run`, and [`cargo nextest`][6] with command-line interface compatible
  with cargo.
* Support for proc-macro, including coverage of UI tests.
* Support for doc tests. (this is currently optional and requires nightly, see [#2][7] for more)
* Fast because it does not introduce extra layers between rustc, cargo, and llvm-tools.

**Table of Contents:**

* [Usage][8]
  
  * [Basic usage][9]
  * [Merge coverages generated under different test conditions][10]
  * [Get coverage of C/C++ code linked to Rust library/binary][11]
  * [Get coverage of external tests][12]
  * [Get coverage of AFL fuzzers][13]
  * [Exclude file from coverage][14]
  * [Exclude code from coverage][15]
  * [Continuous Integration][16]
    
    * [GitHub Actions and Codecov][17]
    * [GitLab CI][18]
  * [Display coverage in VS Code][19]
  * [Environment variables][20]
  * [Additional JSON information][21]
* [Installation][22]
* [Known limitations][23]
* [Related Projects][24]
* [License][25]

## Usage

### Basic usage

Complete list of options (click to show)

(See [docs][26] directory for options of subcommands)

$ cargo llvm-cov --help
cargo-llvm-cov
Cargo subcommand to easily use LLVM source-based code coverage (-C instrument-coverage).

USAGE:
    cargo llvm-cov [SUBCOMMAND] [OPTIONS] [-- <args>...]

ARGS:
    <args>...
            Arguments for the test binary

OPTIONS:
        --json
            Export coverage data in "json" format

            If --output-path is not specified, the report will be printed to stdout.

            This internally calls `llvm-cov export -format=text`. See
            <https://llvm.org/docs/CommandGuide/llvm-cov.html#llvm-cov-export> for more.

        --lcov
            Export coverage data in "lcov" format

            If --output-path is not specified, the report will be printed to stdout.

            This internally calls `llvm-cov export -format=lcov`. See
            <https://llvm.org/docs/CommandGuide/llvm-cov.html#llvm-cov-export> for more.

        --cobertura
            Export coverage data in "cobertura" XML format

            If --output-path is not specified, the report will be printed to stdout.

            This internally calls `llvm-cov export -format=lcov` and then converts to cobertura.xml.
            See <https://llvm.org/docs/CommandGuide/llvm-cov.html#llvm-cov-export> for more.

        --codecov
            Export coverage data in "Codecov Custom Coverage" format

            If --output-path is not specified, the report will be printed to stdout.

            This internally calls `llvm-cov export -format=json` and then converts to codecov.json.
            See <https://llvm.org/docs/CommandGuide/llvm-cov.html#llvm-cov-export> for more.

        --text
            Generate coverage report in "text" format

            If --output-path or --output-dir is not specified, the report will be printed to stdout.

            This internally calls `llvm-cov show -format=text`. See
            <https://llvm.org/docs/CommandGuide/llvm-cov.html#llvm-cov-show> for more.

        --html
            Generate coverage report in "html" format

            If --output-dir is not specified, the report will be generated in `target/llvm-cov/html`
            directory.

            This internally calls `llvm-cov show -format=html`. See
            <https://llvm.org/docs/CommandGuide/llvm-cov.html#llvm-cov-show> for more.

        --open
            Generate coverage reports in "html" format and open them in a browser after the
            operation.

            See --html for more.

        --summary-only
            Export only summary information for each file in the coverage data

            This flag can only be used together with --json, --lcov, or --cobertura.

        --output-path <PATH>
            Specify a file to write coverage data into.

            This flag can only be used together with --json, --lcov, --cobertura, or --text.
            See --output-dir for --html and --open.

        --output-dir <DIRECTORY>
            Specify a directory to write coverage report into (default to `target/llvm-cov`).

            This flag can only be used together with --text, --html, or --open. See also
            --output-path.

        --failure-mode <any|all>
            Fail if `any` or `all` profiles cannot be merged (default to `any`)

        --ignore-filename-regex <PATTERN>
            Skip source code files with file paths that match the given regular expression

        --show-instantiations
            Show instantiations in report

        --no-cfg-coverage
            Unset cfg(coverage), which is enabled when code is built using cargo-llvm-cov

        --no-cfg-coverage-nightly
            Unset cfg(coverage_nightly), which is enabled when code is built using cargo-llvm-cov
            and nightly compiler

        --no-report
            Run tests, but don't generate coverage report

        --no-clean
            Build without cleaning any old build artifacts

        --fail-under-functions <MIN>
            Exit with a status of 1 if the total function coverage is less than MIN percent

        --fail-under-lines <MIN>
            Exit with a status of 1 if the total line coverage is less than MIN percent

        --fail-under-regions <MIN>
            Exit with a status of 1 if the total region coverage is less than MIN percent

        --fail-uncovered-lines <MAX>
            Exit with a status of 1 if the uncovered lines are greater than MAX

        --fail-uncovered-regions <MAX>
            Exit with a status of 1 if the uncovered regions are greater than MAX

        --fail-uncovered-functions <MAX>
            Exit with a status of 1 if the uncovered functions are greater than MAX

        --show-missing-lines
            Show lines with no coverage

        --include-build-script
            Include build script in coverage report

        --dep-coverage <NAME>
            Show coverage of the specified dependency instead of the crates in the current workspace
. (unstable)

        --skip-functions
            Skip exporting per-function coverage data.

            This flag can only be used together with --json, --lcov, or --cobertura.

        --branch
            Enable branch coverage. (unstable)

        --mcdc
            Enable mcdc coverage. (unstable)

        --doctests
            Including doc tests (unstable)

            This flag is unstable. See <https://github.com/taiki-e/cargo-llvm-cov/issues/2> for
            more.

        --no-run
            Generate coverage report without running tests

        --no-fail-fast
            Run all tests regardless of failure

        --ignore-run-fail
            Run all tests regardless of failure and generate report

            If tests failed but report generation succeeded, exit with a status of 0.

    -q, --quiet
            Display one character per test instead of one line

        --lib
            Test only this package's library unit tests

        --bin <NAME>
            Test only the specified binary

        --bins
            Test all binaries

        --example <NAME>
            Test only the specified example

        --examples
            Test all examples

        --test <NAME>
            Test only the specified test target

        --tests
            Test all tests

        --bench <NAME>
            Test only the specified bench target

        --benches
            Test all benches

        --all-targets
            Test all targets

        --doc
            Test only this library's documentation (unstable)

            This flag is unstable because it automatically enables --doctests flag. See
            <https://github.com/taiki-e/cargo-llvm-cov/issues/2> for more.

    -p, --package <SPEC>
            Package to run tests for

        --workspace
            Test all packages in the workspace

        --all
            Alias for --workspace (deprecated)

        --exclude <SPEC>
            Exclude packages from both the test and report

        --exclude-from-test <SPEC>
            Exclude packages from the test (but not from the report)

        --exclude-from-report <SPEC>
            Exclude packages from the report (but not from the test)

    -j, --jobs <N>
            Number of parallel jobs, defaults to # of CPUs

    -r, --release
            Build artifacts in release mode, with optimizations

        --profile <PROFILE-NAME>
            Build artifacts with the specified profile

    -F, --features <FEATURES>
            Space or comma separated list of features to activate

        --all-features
            Activate all available features

        --no-default-features
            Do not activate the `default` feature

        --target <TRIPLE>
            Build for the target triple

            When this option is used, coverage for proc-macro and build script will not be displayed
            because cargo does not pass RUSTFLAGS to them.

        --coverage-target-only
            Activate coverage reporting only for the target triple

            Activate coverage reporting only for the target triple specified via `--target`. This is
            important, if the project uses multiple targets via the cargo bindeps feature, and not
            all targets can use `instrument-coverage`, e.g. a microkernel, or an embedded binary.

    -v, --verbose
            Use verbose output

            Use -vv (-vvv) to propagate verbosity to cargo.

        --color <WHEN>
            Coloring: auto, always, never

        --remap-path-prefix
            Use --remap-path-prefix for workspace root

            Note that this does not fully compatible with doctest.

        --include-ffi
            Include coverage of C/C++ code linked to Rust library/binary

            Note that `CC`/`CXX`/`LLVM_COV`/`LLVM_PROFDATA` environment variables must be set to
            Clang/LLVM compatible with the LLVM version used in rustc.

        --keep-going
            Do not abort the build as soon as there is an error (unstable)

        --ignore-rust-version
            Ignore `rust-version` specification in packages

        --manifest-path <PATH>
            Path to Cargo.toml

        --frozen
            Require Cargo.lock and cache are up to date

        --locked
            Require Cargo.lock is up to date

        --offline
            Run without accessing the network

    -Z <FLAG>
            Unstable (nightly-only) flags to Cargo, see 'cargo -Z help' for
            details

    -h, --help
            Print help information

    -V, --version
            Print version information

SUBCOMMANDS:
    test
            Run tests and generate coverage report
            This is equivalent to `cargo llvm-cov` without subcommand,
            except that test name filtering is supported.
    run
            Run a binary or example and generate coverage report
    report
            Generate coverage report
    show-env
            Output the environment set by cargo-llvm-cov to build Rust projects
    clean
            Remove artifacts that cargo-llvm-cov has generated in the past
    nextest
            Run tests with cargo nextest
            This internally calls `cargo nextest run`.

By default, run tests (via `cargo test`), and print the coverage summary to stdout.

cargo llvm-cov

Currently, doc tests are disabled by default because nightly-only features are required to make
coverage work for doc tests. see [#2][27] for more.

To run `cargo run` instead of `cargo test`, use `run` subcommand.

cargo llvm-cov run

With html report (the report will be generated to `target/llvm-cov/html` directory):

cargo llvm-cov --html
open target/llvm-cov/html/index.html

or

cargo llvm-cov --open

With plain text report (if `--output-path` is not specified, the report will be printed to stdout):

cargo llvm-cov --text | less -R

With json report (if `--output-path` is not specified, the report will be printed to stdout):

cargo llvm-cov --json --output-path cov.json

With lcov report (if `--output-path` is not specified, the report will be printed to stdout):

cargo llvm-cov --lcov --output-path lcov.info

You can get a coverage report in a different format based on the results of a previous run by using
`cargo llvm-cov report`.

cargo llvm-cov --html          # run tests and generate html report
cargo llvm-cov report --lcov # generate lcov report

`cargo llvm-cov`/`cargo llvm-cov run`/`cargo llvm-cov nextest` cleans some build artifacts by
default to avoid false positives/false negatives due to old build artifacts. This behavior is
disabled when `--no-clean`, `--no-report`, or `--no-run` is passed, and old build artifacts are
retained. When using these flags, it is recommended to first run `cargo llvm-cov clean --workspace`
to remove artifacts that may affect the coverage results.

cargo llvm-cov clean --workspace # remove artifacts that may affect the coverage results
cargo llvm-cov --no-clean

### Merge coverages generated under different test conditions

You can merge the coverages generated under different test conditions by using `--no-report` and
`cargo llvm-cov report`.

cargo llvm-cov clean --workspace # remove artifacts that may affect the coverage results
cargo llvm-cov --no-report --features a
cargo llvm-cov --no-report --features b
cargo llvm-cov report --lcov # generate report without tests

Tip

To include coverage for doctests you also need to pass `--doctests` to `cargo llvm-cov report`.

### Get coverage of C/C++ code linked to Rust library/binary

Set `CC`, `CXX`, `LLVM_COV`, and `LLVM_PROFDATA` environment variables to Clang/LLVM compatible with
the LLVM version used in rustc, and run cargo-llvm-cov with `--include-ffi` flag.

CC=<clang-path> \
CXX=<clang++-path> \
LLVM_COV=<llvm-cov-path> \
LLVM_PROFDATA=<llvm-profdata-path> \
  cargo llvm-cov --lcov --include-ffi

Known compatible Rust (installed via rustup) and LLVM versions:

──────────┬──────────────┬──────────────┬──────────────
          │Rust 1.60-1.77│Rust 1.78-1.81│Rust 1.82-1.92
──────────┼──────────────┼──────────────┼──────────────
LLVM 14-17│**✓**         │              │              
──────────┼──────────────┼──────────────┼──────────────
LLVM 18   │              │**✓**         │              
──────────┼──────────────┼──────────────┼──────────────
LLVM 19-21│              │              │**✓**         
──────────┴──────────────┴──────────────┴──────────────

### Get coverage of external tests

`cargo test`, `cargo run`, and [`cargo nextest`][28] are available as builtin, but cargo-llvm-cov
can also be used for arbitrary binaries built using cargo (including other cargo subcommands or
external tests that use make, [xtask][29], etc.)

# Set the environment variables needed to get coverage.
source <(cargo llvm-cov show-env --export-prefix)
# Remove artifacts that may affect the coverage results.
# This command should be called after show-env.
cargo llvm-cov clean --workspace
# Above two commands should be called before build binaries.

cargo build # Build rust binaries with instrumentation.
# Commands using binaries in target/debug/*, including `cargo test` and other cargo subcommands.
# ...

cargo llvm-cov report --lcov # Generate report without tests.

Caution

cargo-llvm-cov subcommands other than `report` and `clean` may not work correctly in the context
where environment variables are set by `show-env`; consider using normal
`cargo`/`cargo-nextest`/etc. commands.

Tip

To include coverage for doctests you also need to pass `--doctests` to both `cargo llvm-cov
show-env` and `cargo llvm-cov report`.

Tip

The same thing can be achieved in PowerShell 6+ by substituting the source command with:

Invoke-Expression (cargo llvm-cov show-env --with-pwsh-env-prefix | Out-String)

### Get coverage of AFL fuzzers

Cargo-llvm-cov can be used with [AFL.rs][30] similar to the way external tests are done, but with a
few caveats.

# Set environment variables and clean workspace
source <(cargo llvm-cov show-env --export-prefix)
cargo llvm-cov clean --workspace
# Build the fuzz target
cargo afl build
# Run the fuzzer, the AFL_FUZZER_LOOPCOUNT is needed, because otherwise .profraw files aren't emitte
d
# To get coverage of current corpus, minimize it and set it as input, then run the fuzzer until it p
rocesses the corpus
AFL_FUZZER_LOOPCOUNT=20 cargo afl fuzz -c - -i in -o out target/debug/fuzz-target
# Generate report
# If you pass `--release` to `cargo afl build`, you also need to pass `--release` to `cargo llvm-cov
 report`
cargo llvm-cov report --lcov

### Exclude file from coverage

To exclude specific file patterns from the report, use the `--ignore-filename-regex` option.

cargo llvm-cov --open --ignore-filename-regex build

By default, [vendored sources][31] will not be included in the report. Also, the following patterns
and crates specified by `--exclude` or `--exclude-from-report` are excluded:

`{SEPARATOR}rustc{SEPARATOR}([0-9a-f]+|[0-9]+\.[0-9]+\.[0-9]+){SEPARATOR}
^{WORKSPACE_ROOT}({SEPARATOR}.*)?{SEPARATOR}(tests|examples|benches){SEPARATOR}
^{TARGET_DIR}($|{SEPARATOR})
^{CARGO_HOME}{SEPARATOR}(registry|git){SEPARATOR}
^{RUSTUP_HOME}{SEPARATOR}toolchains($|{SEPARATOR})
`

You can use `--disable-default-ignore-filename-regex` to disable default exclude setting.

### Exclude code from coverage

To exclude the specific function or module from coverage, use the [`#[coverage(off)]`
attribute][32].

Since `#[coverage(off)]` is unstable, it is recommended to use it together with `cfg(coverage)` or
`cfg(coverage_nightly)` set by cargo-llvm-cov.

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

// function
#[cfg_attr(coverage_nightly, coverage(off))]
fn exclude_fn_from_coverage() {
    // ...
}

// module
#[cfg_attr(coverage_nightly, coverage(off))]
mod exclude_mod_from_coverage {
    // ...
}

cfgs are set under the following conditions:

* `cfg(coverage)` is always set when using cargo-llvm-cov (unless `--no-cfg-coverage` flag passed)
* `cfg(coverage_nightly)` is set when using cargo-llvm-cov with nightly toolchain (unless
  `--no-cfg-coverage-nightly` flag passed)

Rust 1.80+ warns the above cfgs as `unexpected_cfgs`. The recommended way to address this is to add
a [`lints` table][33] to `Cargo.toml`.

[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(coverage,coverage_nightly)'] }

If you want to ignore all `#[test]`-related code, you can use module-level `#[coverage(off)]`
attribute:

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    // ...
}

cargo-llvm-cov excludes code contained in the directory named `tests` from the report by default, so
you can also use it instead of `#[coverage(off)]` attribute.

### Continuous Integration

#### GitHub Actions and Codecov

Here is an example of GitHub Actions workflow that uploads coverage to [Codecov][34].

name: Coverage

on: [pull_request, push]

jobs:
  coverage:
    runs-on: ubuntu-latest
    env:
      CARGO_TERM_COLOR: always
    steps:
      - uses: actions/checkout@v5
      - name: Install Rust
        run: rustup update stable
      - name: Install cargo-llvm-cov
        uses: taiki-e/install-action@cargo-llvm-cov
      - name: Generate code coverage
        run: cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v5
        with:
          token: ${{ secrets.CODECOV_TOKEN }} # required for private repos or protected branches
          files: lcov.info
          fail_ci_if_error: true

Currently, when using `--lcov` flag, [only line coverage is available on Codecov][35].

By using `--codecov` flag instead of `--lcov` flag, you can use region coverage on Codecov:

- name: Generate code coverage
  run: cargo llvm-cov --all-features --workspace --codecov --output-path codecov.json
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v5
  with:
    token: ${{ secrets.CODECOV_TOKEN }} # required for private repos or protected branches
    files: codecov.json
    fail_ci_if_error: true

Note that [the way Codecov shows region/branch coverage is not very good][36].

#### GitLab CI

First of all, when running the CI you need to make sure `cargo-llvm-cov` is available in the
execution script. Whether you add it to a custom image, or run `cargo install` as part of your
pipeline, it should be available and in `PATH`. Once done, it's simple:

unit_tests:
  artifacts:
    reports:
      junit: target/nextest/default/junit.xml
      coverage_report:
        coverage_format: cobertura
        path: target/llvm-cov-target/cobertura.xml
  # this uses region for coverage summary
  coverage: '/TOTAL\s+(\d+\s+)+(\d+\.\d+\%)/'
  script:
    - cargo llvm-cov nextest
    - cargo llvm-cov report --cobertura --output-path target/llvm-cov-target/cobertura.xml

Caution

GitLab has certain [limits for Cobertura reports][37] make sure you obey them.

Note

Note that this example uses [`cargo-nextest`][38] to run the tests (which must similarly be
available), with the following `.config/nextest.toml`:

[profile.default.junit]
path = "junit.xml"

### Display coverage in VS Code

You can display coverage in VS Code using [Coverage Gutters][39].

Coverage Gutters supports lcov style coverage file and detects `lcov.info` files at the top level or
in the `coverage` directory. Below is an example command to generate the coverage file.

cargo llvm-cov --lcov --output-path lcov.info

You may need to click the "Watch" label in the bottom bar of VS Code to display coverage.

### Environment variables

You can override these environment variables to change cargo-llvm-cov's behavior on your system:

* `CARGO_LLVM_COV_TARGET_DIR` -- Location of where to place all generated artifacts, relative to the
  current working directory. Default to `<cargo_target_dir>/llvm-cov-target`.
* `CARGO_LLVM_COV_BUILD_DIR` -- Location of where intermediate build artifacts will be stored,
  relative to the current working directory. Default to `<cargo_target_dir>/llvm-cov-target`.
* `CARGO_LLVM_COV_SETUP` -- Control behavior if `llvm-tools-preview` component is not installed. See
  [#219][40] for more.
* `LLVM_COV` -- Override the path to `llvm-cov`. You may need to specify both this and
  `LLVM_PROFDATA` environment variables if you are using [`--include-ffi` flag][41] or if you are
  using a toolchain installed without via rustup. `llvm-cov` version must be compatible with the
  LLVM version used in rustc.
* `LLVM_PROFDATA` -- Override the path to `llvm-profdata`. See `LLVM_COV` environment variable for
  more.
* `LLVM_COV_FLAGS` -- A space-separated list of additional flags to pass to all `llvm-cov`
  invocations that cargo-llvm-cov performs. See [LLVM documentation][42] for available options.
* `LLVM_PROFDATA_FLAGS` -- A space-separated list of additional flags to pass to all `llvm-profdata`
  invocations that cargo-llvm-cov performs. See [LLVM documentation][43] for available options.
* `LLVM_PROFILE_FILE_NAME` -- Override the file name (the final component of the path) of the
  `LLVM_PROFILE_FILE`. See [LLVM documentation][44] for available syntax.

See also [environment variables that Cargo reads][45]. cargo-llvm-cov respects many of them.

### Additional JSON information

If **JSON** is selected as output format (with the `--json` flag), then cargo-llvm-cov will add
additional contextual information at the root of the llvm-cov data. This can be helpful for programs
that rely on the output of cargo-llvm-cov.

{
  // Other regular llvm-cov fields ...
  "cargo_llvm_cov": {
    "version": "0.0.0",
    "manifest_path": "/path/to/your/project/Cargo.toml"
  }
}

* `version` specifies the version of cargo-llvm-cov that was used. This allows other programs to
  verify a certain version of it was used and make assertions of its behavior.
* `manifest_path` defines the absolute path to the Rust project's Cargo.toml that cargo-llvm-cov was
  executed on. It can help to avoid repeating the same option on both programs.

For example, when forwarding the JSON output directly to another program:

cargo-llvm-cov --json | some-program

## Installation

### From source

cargo +stable install cargo-llvm-cov --locked

Currently, installing cargo-llvm-cov requires rustc 1.87+.

cargo-llvm-cov is usually runnable with Cargo versions older than the Rust version required for
installation (e.g., `cargo +1.60 llvm-cov`). Currently, to run cargo-llvm-cov requires Cargo 1.60+.

### From prebuilt binaries

You can download prebuilt binaries from the [Release page][46]. Prebuilt binaries are available for
Linux (x86_64 gnu/musl, aarch64 gnu/musl, powerpc64le gnu/musl, riscv64gc gnu/musl, and s390x gnu,
musl binaries are static executable), macOS (x86_64, aarch64, and universal), Windows (x86_64,
static executable), FreeBSD (x86_64).

Example of script to install from the Release page (click to show)
# Get host target
host=$(rustc -vV | grep '^host:' | cut -d' ' -f2)
# Download binary and install to $HOME/.cargo/bin
curl --proto '=https' --tlsv1.2 -fsSL "https://github.com/taiki-e/cargo-llvm-cov/releases/latest/dow
nload/cargo-llvm-cov-$host.tar.gz" \
  | tar xzf - -C "$HOME/.cargo/bin"

### On GitHub Actions

You can use [taiki-e/install-action][47] to install prebuilt binaries on Linux, macOS, and Windows.
This makes the installation faster and may avoid the impact of [problems caused by upstream
changes][48].

- uses: taiki-e/install-action@cargo-llvm-cov

When used with [nextest][49]:

- uses: taiki-e/install-action@cargo-llvm-cov
- uses: taiki-e/install-action@nextest

### Via Homebrew

You can install cargo-llvm-cov from [homebrew-core][50] (x86_64/AArch64 macOS, x86_64/AArch64
Linux):

brew install cargo-llvm-cov

Alternatively, you can also install from the [Homebrew tap maintained by us][51] (x86_64/AArch64
macOS, x86_64/AArch64 Linux):

brew install taiki-e/tap/cargo-llvm-cov

### Via Scoop (Windows)

You can install cargo-llvm-cov from the [Scoop bucket maintained by us][52]:

scoop bucket add taiki-e https://github.com/taiki-e/scoop-bucket
scoop install cargo-llvm-cov

### Via cargo-binstall

You can install cargo-llvm-cov using [cargo-binstall][53]:

cargo binstall cargo-llvm-cov

### Via pacman (Arch Linux)

You can install cargo-llvm-cov from the [extra repository][54]:

pacman -S cargo-llvm-cov

### Via ports (FreeBSD)

You can install cargo-llvm-cov from the [official ports][55]:

pkg install cargo-llvm-cov

### Via other package managers

[[Packaging status]][56]

## Known limitations

* Support for branch coverage is unstable. See [#8][57] and [rust-lang/rust#79649][58] for more.
* Support for doc tests is unstable and has known issues. See [#2][59] and
  [rust-lang/rust#79417][60] for more.

See also [the code-coverage-related issues reported in rust-lang/rust][61].

## Related Projects

* [cargo-config2][62]: Library to load and resolve Cargo configuration. cargo-llvm-cov uses this
  library.
* [cargo-hack][63]: Cargo subcommand to provide various options useful for testing and continuous
  integration.
* [cargo-minimal-versions][64]: Cargo subcommand for proper use of `-Z minimal-versions`.

## License

Licensed under either of [Apache License, Version 2.0][65] or [MIT license][66] at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

[1]: https://crates.io/crates/cargo-llvm-cov
[2]: #license
[3]: https://github.com/taiki-e/cargo-llvm-cov/actions
[4]: https://doc.rust-lang.org/rustc/instrument-coverage.html
[5]: https://github.com/taiki-e/cargo-llvm-cov/issues/8
[6]: https://nexte.st/book/test-coverage.html
[7]: https://github.com/taiki-e/cargo-llvm-cov/issues/2
[8]: #usage
[9]: #basic-usage
[10]: #merge-coverages-generated-under-different-test-conditions
[11]: #get-coverage-of-cc-code-linked-to-rust-librarybinary
[12]: #get-coverage-of-external-tests
[13]: #get-coverage-of-afl-fuzzers
[14]: #exclude-file-from-coverage
[15]: #exclude-code-from-coverage
[16]: #continuous-integration
[17]: #github-actions-and-codecov
[18]: #gitlab-ci
[19]: #display-coverage-in-vs-code
[20]: #environment-variables
[21]: #additional-json-information
[22]: #installation
[23]: #known-limitations
[24]: #related-projects
[25]: #license
[26]: /taiki-e/cargo-llvm-cov/blob/main/docs
[27]: https://github.com/taiki-e/cargo-llvm-cov/issues/2
[28]: https://nexte.st/book/test-coverage.html
[29]: https://github.com/matklad/cargo-xtask
[30]: https://github.com/rust-fuzz/afl.rs
[31]: https://doc.rust-lang.org/cargo/commands/cargo-vendor.html
[32]: https://github.com/rust-lang/rust/issues/84605
[33]: https://doc.rust-lang.org/nightly/rustc/check-cfg/cargo-specifics.html#check-cfg-in-lintsrust-
table
[34]: https://codecov.io
[35]: https://github.com/taiki-e/cargo-llvm-cov/issues/20
[36]: https://github.com/taiki-e/cargo-llvm-cov/pull/255#issuecomment-1513318191
[37]: https://docs.gitlab.com/ee/ci/testing/test_coverage_visualization/cobertura.html#limits
[38]: https://nexte.st/
[39]: https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters
[40]: https://github.com/taiki-e/cargo-llvm-cov/issues/219
[41]: #get-coverage-of-cc-code-linked-to-rust-librarybinary
[42]: https://llvm.org/docs/CommandGuide/llvm-cov.html
[43]: https://llvm.org/docs/CommandGuide/llvm-profdata.html
[44]: https://clang.llvm.org/docs/SourceBasedCodeCoverage.html#running-the-instrumented-program
[45]: https://doc.rust-lang.org/nightly/cargo/reference/environment-variables.html#environment-varia
bles-cargo-reads
[46]: https://github.com/taiki-e/cargo-llvm-cov/releases
[47]: https://github.com/taiki-e/install-action
[48]: https://github.com/tokio-rs/bytes/issues/506
[49]: https://nexte.st/book/test-coverage.html
[50]: https://formulae.brew.sh/formula/cargo-llvm-cov
[51]: https://github.com/taiki-e/homebrew-tap/blob/HEAD/Formula/cargo-llvm-cov.rb
[52]: https://github.com/taiki-e/scoop-bucket/blob/HEAD/bucket/cargo-llvm-cov.json
[53]: https://github.com/cargo-bins/cargo-binstall
[54]: https://archlinux.org/packages/extra/x86_64/cargo-llvm-cov
[55]: https://www.freshports.org/devel/cargo-llvm-cov
[56]: https://repology.org/project/cargo-llvm-cov/versions
[57]: https://github.com/taiki-e/cargo-llvm-cov/issues/8
[58]: https://github.com/rust-lang/rust/issues/79649
[59]: https://github.com/taiki-e/cargo-llvm-cov/issues/2
[60]: https://github.com/rust-lang/rust/issues/79417
[61]: https://github.com/rust-lang/rust/labels/A-code-coverage
[62]: https://github.com/taiki-e/cargo-config2
[63]: https://github.com/taiki-e/cargo-hack
[64]: https://github.com/taiki-e/cargo-minimal-versions
[65]: /taiki-e/cargo-llvm-cov/blob/main/LICENSE-APACHE
[66]: /taiki-e/cargo-llvm-cov/blob/main/LICENSE-MIT
