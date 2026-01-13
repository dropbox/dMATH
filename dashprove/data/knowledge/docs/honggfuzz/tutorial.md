# honggfuzz-rs [[Build Status]][1] [[Crates.io]][2] [[Documentation]][3]

Fuzz your Rust code with Google-developed Honggfuzz !

## [Documentation][4]

[[asciicast]][5]

## About Honggfuzz

Honggfuzz is a security oriented fuzzer with powerful analysis options. Supports evolutionary,
feedback-driven fuzzing based on code coverage (software- and hardware-based).

* project homepage [honggfuzz.com][6]
* project repository [github.com/google/honggfuzz][7]
* this upstream project is maintained by Google, but ...
* this is NOT an official Google product

## Compatibility

* **Rust**: stable, beta, nightly
* **OS**: GNU/Linux, macOS, FreeBSD, NetBSD, Android, WSL (Windows Subsystem for Linux)
* **Arch**: x86_64, x86, arm64-v8a, armeabi-v7a, armeabi
* **Sanitizer**: none, address, thread, leak

## Dependencies

### Linux

* C compiler: `cc`
* GNU Make: `make`
* GNU Binutils development files for the BFD library: `libbfd.h`
* libunwind development files: `libunwind.h`
* Blocks runtime library (when compiling with clang)
* liblzma development files

For example on Debian and its derivatives:

sudo apt install build-essential binutils-dev libunwind-dev libblocksruntime-dev liblzma-dev

## How to use this crate

Install honggfuzz commands to build with instrumentation and fuzz

# installs hfuzz and honggfuzz subcommands in cargo
cargo install honggfuzz

Add to your dependencies

[dependencies]
honggfuzz = "0.5"

Create a target to fuzz

use honggfuzz::fuzz;

fn main() {
    // Here you can parse `std::env::args and
    // setup / initialize your project

    // You have full control over the loop but
    // you're supposed to call `fuzz` ad vitam aeternam
    loop {
        // The fuzz macro gives an arbitrary object (see `arbitrary crate`)
        // to a closure-like block of code.
        // For performance reasons, it is recommended that you use the native type
        // `&[u8]` when possible.
        // Here, this slice will contain a "random" quantity of "random" data.
        fuzz!(|data: &[u8]| {
            if data.len() != 3 {return}
            if data[0] != b'h' {return}
            if data[1] != b'e' {return}
            if data[2] != b'y' {return}
            panic!("BOOM")
        });
    }
}

Fuzz for fun and profit !

# builds with fuzzing instrumentation and then fuzz the "example" target
cargo hfuzz run example

Once you got a crash, replay it easily in a debug environment

# builds the target in debug mode and replays automatically the crash in rust-lldb
cargo hfuzz run-debug example hfuzz_workspace/*/*.fuzz

You can also build and run your project without compile-time software instrumentation (LLVM's SanCov
passes)

This allows you for example to try hardware-only feedback driven fuzzing:

# builds without fuzzing instrumentation and then fuzz the "example" target using hardware-based fee
dback
HFUZZ_RUN_ARGS="--linux_perf_ipt_block --linux_perf_instr --linux_perf_branch" cargo hfuzz run-no-in
str example

Clean

# a wrapper on "cargo clean" which cleans the fuzzing_target directory
cargo hfuzz clean

Version

cargo hfuzz version

### Environment variables

#### `RUSTFLAGS`

You can use `RUSTFLAGS` to send additional arguments to `rustc`.

For instance, you can enable the use of LLVM's [sanitizers][8]. This is a recommended option if you
want to test your `unsafe` rust code but it will have an impact on performance.

RUSTFLAGS="-Z sanitizer=address" cargo hfuzz run example

#### `HFUZZ_BUILD_ARGS`

You can use `HFUZZ_BUILD_ARGS` to send additional arguments to `cargo build`.

#### `HFUZZ_RUN_ARGS`

You can use `HFUZZ_RUN_ARGS` to send additional arguments to `honggfuzz`. See [USAGE][9] for the
list of those.

For example:

# 1 second of timeout
# use 12 fuzzing thread
# be verbose
# stop after 1000000 fuzzing iteration
# exit upon crash
HFUZZ_RUN_ARGS="-t 1 -n 12 -v -N 1000000 --exit_upon_crash" cargo hfuzz run example

#### `HFUZZ_DEBUGGER`

By default we use `rust-lldb` but you can change it to `rust-gdb`, `gdb`, `/usr/bin/lldb-7` ...

#### `CARGO_TARGET_DIR`

Target compilation directory, defaults to `hfuzz_target` to not clash with `cargo build`'s default
`target` directory.

#### `HFUZZ_WORKSPACE`

Honggfuzz working directory, defaults to `hfuzz_workspace`.

#### `HFUZZ_INPUT`

Honggfuzz input files (also called "corpus"), defaults to `$HFUZZ_WORKSPACE/{TARGET}/input`.

## Conditional compilation

Sometimes, it is necessary to make some specific adaptation to your code to yield a better fuzzing
efficiency.

For instance:

* Make you software behavior as much as possible deterministic on the fuzzing input
  
  * [PRNG][10]s must be seeded with a constant or the fuzzer input
  * Behavior shouldn't change based on the computer's clock.
  * Avoid potential undeterministic behavior from racing threads.
  * ...
* Never ever call `std::process::exit()`.
* Disable logging and other unnecessary functionalities.
* Try to avoid modifying global state when possible.
* Do not set up your own panic hook when run with `cfg(fuzzing)`

When building with `cargo hfuzz`, the argument `--cfg fuzzing` is passed to `rustc` to allow you to
condition the compilation of those adaptations thanks to the `cfg` macro like so:

#[cfg(fuzzing)]
let mut rng = rand_chacha::ChaCha8Rng::from_seed(&[0]);
#[cfg(not(fuzzing))]
let mut rng = rand::thread_rng();

Also, when building in debug mode, the `fuzzing_debug` argument is added in addition to `fuzzing`.

For more information about conditional compilation, please see the [reference][11].

## Relevant documentation about honggfuzz

* [USAGE][12]
* [FeedbackDrivenFuzzing][13]
* [PersistentFuzzing][14]

## About Rust fuzzing

There is other projects providing Rust fuzzing support at [github.com/rust-fuzz][15].

You'll find support for [AFL][16] and LLVM's [LibFuzzer][17] and there is also a [trophy case][18]
;-) .

This crate was inspired by those projects!

[1]: https://github.com/rust-fuzz/honggfuzz-rs/actions/workflows/rust.yml
[2]: https://crates.io/crates/honggfuzz
[3]: https://docs.rs/honggfuzz
[4]: https://docs.rs/honggfuzz
[5]: https://asciinema.org/a/43MLo5Xl8ukHxgwDLArKqS9xc
[6]: http://honggfuzz.com/
[7]: https://github.com/google/honggfuzz
[8]: https://github.com/japaric/rust-san
[9]: https://github.com/google/honggfuzz/blob/master/docs/USAGE.md
[10]: https://en.wikipedia.org/wiki/Pseudorandom_number_generator
[11]: https://doc.rust-lang.org/reference/attributes.html#conditional-compilation
[12]: https://github.com/google/honggfuzz/blob/master/docs/USAGE.md
[13]: https://github.com/google/honggfuzz/blob/master/docs/FeedbackDrivenFuzzing.md
[14]: https://github.com/google/honggfuzz/blob/master/docs/PersistentFuzzing.md
[15]: https://github.com/rust-fuzz
[16]: https://github.com/rust-fuzz/afl.rs
[17]: https://github.com/rust-fuzz/cargo-fuzz
[18]: https://github.com/rust-fuzz/trophy-case
