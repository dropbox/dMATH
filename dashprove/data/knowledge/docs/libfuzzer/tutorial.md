# [Fuzzing with cargo-fuzz][1]

[cargo-fuzz][2] is the recommended tool for fuzz testing Rust code.

cargo-fuzz is itself not a fuzzer, but a tool to invoke a fuzzer. Currently, the only fuzzer it
supports is [libFuzzer][3] (through the [libfuzzer-sys][4] crate), but it could be extended to
[support other fuzzers in the future][5].

[1]: #fuzzing-with-cargo-fuzz
[2]: https://github.com/rust-fuzz/cargo-fuzz
[3]: http://llvm.org/docs/LibFuzzer.html
[4]: https://github.com/rust-fuzz/libfuzzer
[5]: https://github.com/rust-fuzz/cargo-fuzz/issues/1
