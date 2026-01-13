# [[afl.rs logo]][1]
# afl.rs

#### Fuzzing [Rust][2] code with [AFLplusplus][3]

## What is it?

[Fuzz testing][4] is a software testing technique used to find security and stability issues by
providing pseudo-random data as input to the software. [AFLplusplus][5] is a popular, effective, and
modern fuzz testing tool based on [AFL][6]. This library, afl.rs, allows one to run AFLplusplus on
code written in [the Rust programming language][7].

## Documentation

Documentation can be found in the [Rust Fuzz Book][8].

## What does it look like?

[[Screen recording of afl]][9]

Screen recording of AFL running on Rust code.

## Hints

Before starting to fuzz, you should reconfigure your system for optimal performance and better crash
detection. This can be done with `cargo afl system-config`. But this subcommand requires root, so it
uses sudo internally. Hence, you might need to enter your password.

By default, the AFL++ [CMPLOG][10] feature is activated, which helps to achieve good code coverage.
However, it is not beneficial to activate CMPLOG on more than two instances. So if you run multiple
AFL++ instances on your fuzzing target, you can disable CMPLOG by specifying the command line
parameter '-c -'.

This [document][11] will familiarize you with AFL++ features to help in running a successful fuzzing
campaign.

By default, the `fuzzing` config is set when `cargo-afl` is used to build. If you want to prevent
this, just set the environment variable `AFL_NO_CFG_FUZZING` to `1` when building.

## IJON

If you want to use [IJON][12] - helping fuzzer coverage through code annotation - then have a look
at the [maze example][13].

Note that the IJON macros have been rustyfied to lowercase - hence `IJON_MAX(x)` is `ijon_max(x)` in
Rust.

You will need to the following imports from `afl`, in addition to any macros that you use (e.g.,
`afl::ijon_max`):

`use afl::ijon_hashint;
use afl::ijon_hashstr;
`

[1]: https://github.com/frewsxcv/afl.rs/issues/66
[2]: https://www.rust-lang.org
[3]: https://aflplus.plus/
[4]: https://en.wikipedia.org/wiki/Fuzz_testing
[5]: https://aflplus.plus/
[6]: http://lcamtuf.coredump.cx/afl/
[7]: https://www.rust-lang.org
[8]: https://rust-fuzz.github.io/book/afl.html
[9]: https://raw.githubusercontent.com/rust-fuzz/afl.rs/refs/heads/master/etc/screencap.gif
[10]: https://github.com/AFLplusplus/AFLplusplus/blob/stable/instrumentation/README.cmplog.md
[11]: https://github.com/AFLplusplus/AFLplusplus/blob/stable/docs/fuzzing_in_depth.md
[12]: https://github.com/AFLplusplus/AFLplusplus/blob/stable/docs/IJON.md
[13]: /rust-fuzz/afl.rs/blob/master/afl/examples/maze.rs
