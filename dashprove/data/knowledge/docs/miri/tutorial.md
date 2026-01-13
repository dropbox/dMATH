# Miri

Miri is an [Undefined Behavior][1] detection tool for Rust. It can run binaries and test suites of
cargo projects and detect unsafe code that fails to uphold its safety requirements. For instance:

* Out-of-bounds memory accesses and use-after-free
* Invalid use of uninitialized data
* Violation of intrinsic preconditions (an [`unreachable_unchecked`][2] being reached, calling
  [`copy_nonoverlapping`][3] with overlapping ranges, ...)
* Not sufficiently aligned memory accesses and references
* Violation of basic type invariants (a `bool` that is not 0 or 1, for example, or an invalid enum
  discriminant)
* Data races and emulation of *some* weak memory effects, i.e., atomic reads can return outdated
  values
* **Experimental**: Violations of the [Stacked Borrows][4] rules governing aliasing for reference
  types
* **Experimental**: Violations of the [Tree Borrows][5] aliasing rules, as an optional alternative
  to [Stacked Borrows][6]

On top of that, Miri will also tell you about memory leaks: when there is memory still allocated at
the end of the execution, and that memory is not reachable from a global `static`, Miri will raise
an error.

You can use Miri to emulate programs on other targets, e.g. to ensure that byte-level data
manipulation works correctly both on little-endian and big-endian systems. See
[cross-interpretation][7] below.

Miri has already discovered many [real-world bugs][8]. If you found a bug with Miri, we'd appreciate
if you tell us and we'll add it to the list!

By default, Miri ensures a fully deterministic execution and isolates the program from the host
system. Some APIs that would usually access the host, such as gathering entropy for random number
generators, environment variables, and clocks, are replaced by deterministic "fake" implementations.
Set `MIRIFLAGS="-Zmiri-disable-isolation"` to access the real system APIs instead. (In particular,
the "fake" system RNG APIs make Miri **not suited for cryptographic use**! Do not generate keys
using Miri.)

All that said, be aware that Miri does **not catch every violation of the Rust specification** in
your program, not least because there is no such specification. Miri uses its own approximation of
what is and is not Undefined Behavior in Rust. To the best of our knowledge, all Undefined Behavior
that has the potential to affect a program's correctness *is* being detected by Miri (modulo
[bugs][9]), but you should consult [the Reference][10] for the official definition of Undefined
Behavior. Miri will be updated with the Rust compiler to protect against UB as it is understood by
the current compiler, but it makes no promises about future versions of rustc.

Further caveats that Miri users should be aware of:

* If the program relies on unspecified details of how data is laid out, it will still run fine in
  Miri -- but might break (including causing UB) on different compiler versions or different
  platforms. (You can use `-Zrandomize-layout` to detect some of these cases.)
* Program execution is non-deterministic when it depends, for example, on where exactly in memory
  allocations end up, or on the exact interleaving of concurrent threads. Miri tests one of many
  possible executions of your program, but it will miss bugs that only occur in a different possible
  execution. You can alleviate this to some extent by running Miri with different values for
  `-Zmiri-seed`, but that will still by far not explore all possible executions.
* Miri runs the program as a platform-independent interpreter, so the program has no access to most
  platform-specific APIs or FFI. A few APIs have been implemented (such as printing to stdout,
  accessing environment variables, and basic file system access) but most have not: for example,
  Miri currently does not support networking. System API support varies between targets; if you run
  on Windows it is a good idea to use `--target x86_64-unknown-linux-gnu` to get better support.
* Weak memory emulation is not complete: there are legal behaviors that Miri will never produce.
  However, Miri produces many behaviors that are hard to observe on real hardware, so it can help
  quite a bit in finding weak memory concurrency bugs. To be really sure about complicated atomic
  code, use specialized tools such as [loom][11].

Moreover, Miri fundamentally cannot ensure that your code is *sound*. [Soundness][12] is the
property of never causing undefined behavior when invoked from arbitrary safe code, even in
combination with other sound code. In contrast, Miri can just tell you if *a particular way of
interacting with your code* (e.g., a test suite) causes any undefined behavior *in a particular
execution* (of which there may be many, e.g. when concurrency or other forms of non-determinism are
involved). When Miri finds UB, your code is definitely unsound, but when Miri does not find UB, then
you may just have to test more inputs or more possible non-deterministic choices.

## Using Miri

Install Miri on Rust nightly via `rustup`:

rustup +nightly component add miri

All the following commands assume the nightly toolchain is pinned via `rustup override set nightly`.
Alternatively, use `cargo +nightly` for each of the following commands.

Now you can run your project in Miri:

* To run all tests in your project through Miri, use `cargo miri test`.
* If you have a binary project, you can run it through Miri using `cargo miri run`.

The first time you run Miri, it will perform some extra setup and install some dependencies. It will
ask you for confirmation before installing anything.

`cargo miri run/test` supports the exact same flags as `cargo run/test`. For example, `cargo miri
test filter` only runs the tests containing `filter` in their name.

You can pass [flags][13] to Miri via `MIRIFLAGS`. For example,
`MIRIFLAGS="-Zmiri-disable-stacked-borrows" cargo miri run` runs the program without checking the
aliasing of references.

When compiling code via `cargo miri`, the `cfg(miri)` config flag is set for code that will be
interpreted under Miri. You can use this to ignore test cases that fail under Miri because they do
things Miri does not support:

#[test]
#[cfg_attr(miri, ignore)]
fn does_not_work_on_miri() {
    tokio::run(futures::future::ok::<_, ()>(()));
}

There is no way to list all the infinite things Miri cannot do, but the interpreter will explicitly
tell you when it finds something unsupported:

`error: unsupported operation: can't call foreign function: bind
    ...
    = help: this is likely not a bug in the program; it indicates that the program \
            performed an operation that Miri does not support
`

### Cross-interpretation: running for different targets

Miri can not only run a binary or test suite for your host target, it can also perform
cross-interpretation for arbitrary foreign targets: `cargo miri run --target
x86_64-unknown-linux-gnu` will run your program as if it was a Linux program, no matter your host
OS. This is particularly useful if you are using Windows, as the Linux target is much better
supported than Windows targets.

You can also use this to test platforms with different properties than your host platform. For
example `cargo miri test --target s390x-unknown-linux-gnu` will run your test suite on a big-endian
target, which is useful for testing endian-sensitive code.

### Testing multiple different executions

Certain parts of the execution are picked randomly by Miri, such as the exact base address
allocations are stored at and the interleaving of concurrently executing threads. Sometimes, it can
be useful to explore multiple different execution, e.g. to make sure that your code does not depend
on incidental "super-alignment" of new allocations and to test different thread interleavings. This
can be done with the `-Zmiri-many-seeds` flag:

`MIRIFLAGS="-Zmiri-many-seeds" cargo miri test # tries the seeds in 0..64
MIRIFLAGS="-Zmiri-many-seeds=0..16" cargo miri test
`

The default of 64 different seeds can be quite slow, so you often want to specify a smaller range.

### Running Miri on CI

When running Miri on CI, use the following snippet to install a nightly toolchain with the Miri
component:

rustup toolchain install nightly --component miri
rustup override set nightly

cargo miri test

Here is an example job for GitHub Actions:

  miri:
    name: "Miri"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Miri
        run: |
          rustup toolchain install nightly --component miri
          rustup override set nightly
          cargo miri setup
      - name: Test with Miri
        run: cargo miri test

The explicit `cargo miri setup` helps to keep the output of the actual test step clean.

### Supported targets

Miri does not support all targets supported by Rust. The good news, however, is that no matter your
host OS/platform, it is easy to run code for *any* target using `--target`!

The following targets are tested on CI and thus should always work (to the degree documented below):

* All Rust [Tier 1 targets][14] are supported by Miri. They are all checked on Miri's CI, and some
  (at least one per OS) are even checked on every Rust PR, so the shipped Miri should always work on
  these targets.
* `s390x-unknown-linux-gnu` is supported as our "big-endian target of choice".
* For every other target with OS `linux`, `macos`, or `windows`, Miri should generally work, but we
  make no promises and we don't run tests for such targets.
* We have unofficial support (not maintained by the Miri team itself) for some further operating
  systems.
  
  * `solaris` / `illumos`: maintained by @devnexen. Supports the entire test suite.
  * `freebsd`: maintained by @YohDeadfall and @LorrensP-2158466. Supports the entire test suite.
  * `android`: **maintainer wanted**. Basic OS APIs and concurrency work, but file system access is
    not supported.
* For targets on other operating systems, Miri might fail before even reaching the `main` function.

However, even for targets that we do support, the degree of support for accessing platform APIs
(such as the file system) differs between targets: generally, Linux targets have the best support,
and macOS targets are usually on par. Windows is supported less well.

### Running tests in parallel

Though it implements Rust threading, Miri itself is a single-threaded interpreter. This means that
when running `cargo miri test`, you will probably see a dramatic increase in the amount of time it
takes to run your whole test suite due to the inherent interpreter slowdown and a loss of
parallelism.

You can get your test suite's parallelism back by running `cargo miri nextest run -jN` (note that
you will need [`cargo-nextest`][15] installed). This works because `cargo-nextest` collects a list
of all tests then launches a separate `cargo miri run` for each test. For more information about
nextest, see the [`cargo-nextest` Miri documentation][16].

Note: This one-test-per-process model means that `cargo miri test` is able to detect data races
where two tests race on a shared resource, but `cargo miri nextest run` will not detect such races.

Note: `cargo-nextest` does not support doctests, see [nextest-rs/nextest#16][17]

### Directly invoking the `miri` driver

The recommended way to invoke Miri is via `cargo miri`. Directly invoking the underlying `miri`
driver is not supported, which is why that binary is not even installed into the PATH. However, if
you need to run Miri on many small tests and want to invoke it directly like you would invoke
`rustc`, that is still possible with a bit of extra effort:

# one-time setup
cargo +nightly miri setup
SYSROOT=$(cargo +nightly miri setup --print-sysroot)
# per file
~/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/miri --sysroot "$SYSROOT" file.rs

### Common Problems

When using the above instructions, you may encounter a number of confusing compiler errors.

#### "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace"

You may see this when trying to get Miri to display a backtrace. By default, Miri doesn't expose any
environment to the program, so running `RUST_BACKTRACE=1 cargo miri test` will not do what you
expect.

To get a backtrace, you need to disable isolation [using `-Zmiri-disable-isolation`][18]:

RUST_BACKTRACE=1 MIRIFLAGS="-Zmiri-disable-isolation" cargo miri test

#### "found crate `std` compiled by an incompatible version of rustc"

You may be running `cargo miri` with a different compiler version than the one used to build the
custom libstd that Miri uses, and Miri failed to detect that. Try running `cargo miri clean`.

## Miri `-Z` flags and environment variables

Miri adds its own set of `-Z` flags, which are usually set via the `MIRIFLAGS` environment variable.
We first document the most relevant and most commonly used flags:

* `-Zmiri-backtrace=<0|1|full>` configures how Miri prints backtraces: `1` is the default, where
  backtraces are printed in pruned form; `full` prints backtraces without pruning, and `0` disables
  backtraces entirely.
* `-Zmiri-deterministic-concurrency` makes Miri's concurrency-related behavior fully deterministic.
  Strictly speaking, Miri is always fully deterministic when isolation is enabled (the default
  mode), but this determinism is achieved by using an RNG with a fixed seed. Seemingly harmless
  changes to the program, or just running it for a different target architecture, can thus lead to
  completely different program behavior down the line. This flag disables the use of an RNG for
  concurrency-related decisions. Therefore, Miri cannot find bugs that only occur under some
  specific circumstances, but Miri's behavior will also be more stable across versions and targets.
  This is equivalent to `-Zmiri-fixed-schedule -Zmiri-compare-exchange-weak-failure-rate=0.0
  -Zmiri-address-reuse-cross-thread-rate=0.0 -Zmiri-disable-weak-memory-emulation`.
* `-Zmiri-disable-isolation` disables host isolation. As a consequence, the program has access to
  host resources such as environment variables, file systems, and randomness. This overwrites a
  previous `-Zmiri-isolation-error`.
* `-Zmiri-disable-leak-backtraces` disables backtraces reports for memory leaks. By default, a
  backtrace is captured for every allocation when it is created, just in case it leaks. This incurs
  some memory overhead to store data that is almost never used. This flag is implied by
  `-Zmiri-ignore-leaks`.
* `-Zmiri-env-forward=<var>` forwards the `var` environment variable to the interpreted program. Can
  be used multiple times to forward several variables. Execution will still be deterministic if the
  value of forwarded variables stays the same. Has no effect if `-Zmiri-disable-isolation` is set.
* `-Zmiri-env-set=<var>=<value>` sets the `var` environment variable to `value` in the interpreted
  program. It can be used to pass environment variables without needing to alter the host
  environment. It can be used multiple times to set several variables. If `-Zmiri-disable-isolation`
  or `-Zmiri-env-forward` is set, values set with this option will have priority over values from
  the host environment.
* `-Zmiri-ignore-leaks` disables the memory leak checker, and also allows some remaining threads to
  exist when the main thread exits.
* `-Zmiri-isolation-error=<action>` configures Miri's response to operations requiring host access
  while isolation is enabled. `abort`, `hide`, `warn`, and `warn-nobacktrace` are the supported
  actions. The default is to `abort`, which halts the machine. Some (but not all) operations also
  support continuing execution with a "permission denied" error being returned to the program.
  `warn` prints a full backtrace each time that happens; `warn-nobacktrace` is less verbose and
  shown at most once per operation. `hide` hides the warning entirely. This overwrites a previous
  `-Zmiri-disable-isolation`.
* `-Zmiri-many-seeds=[<from>]..<to>` runs the program multiple times with different seeds for Miri's
  RNG. With different seeds, Miri will make different choices to resolve non-determinism such as the
  order in which concurrent threads are scheduled, or the exact addresses assigned to allocations.
  This is useful to find bugs that only occur under particular interleavings of concurrent threads,
  or that otherwise depend on non-determinism. If the `<from>` part is skipped, it defaults to `0`.
  Can be used without a value; in that case the range defaults to `0..64`.
* `-Zmiri-many-seeds-keep-going` tells Miri to really try all the seeds in the given range, even if
  a failing seed has already been found. This is useful to determine which fraction of seeds fails.
* `-Zmiri-max-extra-rounding-error` tells Miri to always apply the maximum error to float operations
  that do not have a guaranteed precision. The sign of the error is still non-deterministic.
* `-Zmiri-no-extra-rounding-error` stops Miri from adding extra rounding errors to float operations
  that do not have a guaranteed precision.
* `-Zmiri-no-short-fd-operations` stops Miri from artificially forcing `read`/`write` operations to
  only process a part of their buffer. Note that whenever Miri uses host operations to implement
  `read`/`write` (e.g. for file-backed file descriptors), the host system can still introduce short
  reads/writes.
* `-Zmiri-num-cpus` states the number of available CPUs to be reported by miri. By default, the
  number of available CPUs is `1`. Note that this flag does not affect how miri handles threads in
  any way.
* `-Zmiri-permissive-provenance` disables the warning for integer-to-pointer casts and
  [`ptr::with_exposed_provenance`][19]. This will necessarily miss some bugs as those operations are
  not efficiently and accurately implementable in a sanitizer, but it will only miss bugs that
  concern memory/pointers which is subject to these operations.
* `-Zmiri-report-progress` makes Miri print the current stacktrace every now and then, so you can
  tell what it is doing when a program just keeps running. You can customize how frequently the
  report is printed via `-Zmiri-report-progress=<blocks>`, which prints the report every N basic
  blocks.
* `-Zmiri-seed=<num>` configures the seed of the RNG that Miri uses to resolve non-determinism. This
  RNG is used to pick base addresses for allocations, to determine preemption and failure of
  `compare_exchange_weak`, and to control store buffering for weak memory emulation. When isolation
  is enabled (the default), this is also used to emulate system entropy. The default seed is 0. You
  can increase test coverage by running Miri multiple times with different seeds.
* `-Zmiri-strict-provenance` enables [strict provenance][20] checking in Miri. This means that
  casting an integer to a pointer will stop execution because the provenance of the pointer cannot
  be determined.
* `-Zmiri-symbolic-alignment-check` makes the alignment check more strict. By default, alignment is
  checked by casting the pointer to an integer, and making sure that is a multiple of the alignment.
  This can lead to cases where a program passes the alignment check by pure chance, because things
  "happened to be" sufficiently aligned -- there is no UB in this execution but there would be UB in
  others. To avoid such cases, the symbolic alignment check only takes into account the requested
  alignment of the relevant allocation, and the offset into that allocation. This avoids missing
  such bugs, but it also incurs some false positives when the code does manual integer arithmetic to
  ensure alignment. (The standard library `align_to` method works fine in both modes; under symbolic
  alignment it only fills the middle slice when the allocation guarantees sufficient alignment.)
* `-Zmiri-user-relevant-crates=<crate>,<crate>,...` extends the list of crates that Miri considers
  "user-relevant". This affects the rendering of backtraces (for user-relevant crates, Miri shows
  not just the function name but the actual code) and it affects the spans collected for data races
  and aliasing violations (where Miri will show the span of the topmost non-`#[track_caller]` frame
  in a user-relevant crate). When using `cargo miri`, the crates in the local workspace are always
  considered user-relevant.

The remaining flags are for advanced use only, and more likely to change or be removed. Some of
these are **unsound**, which means they can lead to Miri failing to detect cases of undefined
behavior in a program.

* `-Zmiri-address-reuse-rate=<rate>` changes the probability that a freed *non-stack* allocation
  will be added to the pool for address reuse, and the probability that a new *non-stack* allocation
  will be taken from the pool. Stack allocations never get added to or taken from the pool. The
  default is `0.5`.
* `-Zmiri-address-reuse-cross-thread-rate=<rate>` changes the probability that an allocation which
  attempts to reuse a previously freed block of memory will also consider blocks freed by *other
  threads*. The default is `0.1`, which means by default, in 90% of the cases where an address reuse
  attempt is made, only addresses from the same thread will be considered. Reusing an address from
  another thread induces synchronization between those threads, which can mask data races and weak
  memory bugs.
* `-Zmiri-compare-exchange-weak-failure-rate=<rate>` changes the failure rate of
  `compare_exchange_weak` operations. The default is `0.8` (so 4 out of 5 weak ops will fail). You
  can change it to any value between `0.0` and `1.0`, where `1.0` means it will always fail and
  `0.0` means it will never fail. Note that setting it to `1.0` will likely cause hangs, since it
  means programs using `compare_exchange_weak` cannot make progress.
* `-Zmiri-deterministic-floats` makes Miri's floating-point behavior fully deterministic. This means
  that operations will always return the preferred NaN, imprecise operations will not have any
  random error applied to them, and `min`/`max` and "maybe fused" multiply-add all behave
  deterministically. Note that Miri still uses host floats for some operations, so behavior can
  still differ depending on the host target and setup. See `-Zmiri-no-extra-rounding-error` for a
  flag that specifically only disables the random error.
* `-Zmiri-disable-alignment-check` disables checking pointer alignment, so you can focus on other
  failures, but it means Miri can miss bugs in your program. Using this flag is **unsound**.
* `-Zmiri-disable-data-race-detector` disables checking for data races. Using this flag is
  **unsound**. This implies `-Zmiri-disable-weak-memory-emulation`.
* `-Zmiri-disable-stacked-borrows` disables checking the experimental aliasing rules to track
  borrows ([Stacked Borrows][21] and [Tree Borrows][22]). This can make Miri run faster, but it also
  means no aliasing violations will be detected. Using this flag is **unsound** (but the affected
  soundness rules are experimental). Later flags take precedence: borrow tracking can be reactivated
  by `-Zmiri-tree-borrows`.
* `-Zmiri-disable-validation` disables enforcing validity invariants, which are enforced by default.
  This is mostly useful to focus on other failures (such as out-of-bounds accesses) first. Setting
  this flag means Miri can miss bugs in your program. However, this can also help to make Miri run
  faster. Using this flag is **unsound**.
* `-Zmiri-disable-weak-memory-emulation` disables the emulation of some C++11 weak memory effects.
* `-Zmiri-fixed-schedule` disables preemption (like `-Zmiri-preemption-rate=0.0`) and furthermore
  disables the randomization of the next thread to be picked, instead fixing a round-robin schedule.
  Note however that other aspects of Miri's concurrency behavior are still randomize; use
  `-Zmiri-deterministic-concurrency` to disable them all.
* `-Zmiri-force-intrinsic-fallback` forces the use of the "fallback" body for all intrinsics that
  have one. This is useful to test the fallback bodies, but should not be used otherwise. It is
  **unsound** since the fallback body might not be checking for all UB.
* `-Zmiri-native-lib=<path to a shared object file or folder>` is an experimental flag for providing
  support for calling native functions from inside the interpreter via FFI. The flag is supported
  only on Unix systems. Functions not provided by that file are still executed via the usual Miri
  shims. If a path to a directory is specified, all files in that directory are included
  non-recursively. This flag can be passed multiple times to specify multiple files and/or
  directories. **WARNING**: If an invalid/incorrect `.so` file is specified, this can cause
  Undefined Behavior in Miri itself! And of course, Miri often cannot do any checks on the actions
  taken by the native code. Note that Miri has its own handling of file descriptors, so if you want
  to replace *some* functions working on file descriptors, you will have to replace *all* of them,
  or the two kinds of file descriptors will be mixed up. This is **work in progress**; currently,
  only integer and pointers arguments and return values are supported and memory allocated by the
  native code cannot be accessed from Rust (only the other way around). Native code must not spawn
  threads that keep running in the background after the call has returned to Rust and that access
  Rust-allocated memory. Finally, the flag is **unsound** in the sense that Miri stops tracking
  details such as initialization and provenance on memory shared with native code, so it is easily
  possible to write code that has UB which is missed by Miri.
* `-Zmiri-native-lib-enable-tracing` enables the WIP detailed tracing mode for invoking native code.
  Note that this flag is only meaningful on Linux systems; other Unixes (currently) do not support
  tracing mode.
* `-Zmiri-measureme=<name>` enables `measureme` profiling for the interpreted program. This can be
  used to find which parts of your program are executing slowly under Miri. The profile is written
  out to a file inside a directory called `<name>`, and can be processed using the tools in the
  repository [https://github.com/rust-lang/measureme][23].
* `-Zmiri-mute-stdout-stderr` silently ignores all writes to stdout and stderr, but reports to the
  program that it did actually write. This is useful when you are not interested in the actual
  program's output, but only want to see Miri's errors and warnings.
* `-Zmiri-recursive-validation` is a *highly experimental* flag that makes validity checking recurse
  below references.
* `-Zmiri-preemption-rate` configures the probability that at the end of a basic block, the active
  thread will be preempted. The default is `0.01` (i.e., 1%). Setting this to `0` disables
  preemption. Note that even without preemption, the schedule is still non-deterministic: if a
  thread blocks or yields, the next thread is chosen randomly.
* `-Zmiri-provenance-gc=<blocks>` configures how often the pointer provenance garbage collector
  runs. The default is to search for and remove unreachable provenance once every `10000` basic
  blocks. Setting this to `0` disables the garbage collector, which causes some programs to have
  explosive memory usage and/or super-linear runtime.
* `-Zmiri-track-alloc-accesses` show not only allocation and free events for tracked allocations,
  but also reads and writes.
* `-Zmiri-track-alloc-id=<id1>,<id2>,...` shows a backtrace when the given allocations are being
  allocated or freed. This helps in debugging memory leaks and use after free bugs. Specifying this
  argument multiple times does not overwrite the previous values, instead it appends its values to
  the list. Listing an ID multiple times has no effect. You can also add IDs at runtime using
  `miri_track_alloc`.
* `-Zmiri-track-pointer-tag=<tag1>,<tag2>,...` shows a backtrace when a given pointer tag is created
  and when (if ever) it is popped from a borrow stack (which is where the tag becomes invalid and
  any future use of it will error). This helps you in finding out why UB is happening and where in
  your code would be a good place to look for it. Specifying this argument multiple times does not
  overwrite the previous values, instead it appends its values to the list. Listing a tag multiple
  times has no effect.
* `-Zmiri-track-weak-memory-loads` shows a backtrace when weak memory emulation returns an outdated
  value from a load. This can help diagnose problems that disappear under
  `-Zmiri-disable-weak-memory-emulation`.
* `-Zmiri-tree-borrows` replaces [Stacked Borrows][24] with the [Tree Borrows][25] rules. Tree
  Borrows is even more experimental than Stacked Borrows. While Tree Borrows is still sound in the
  sense of catching all aliasing violations that current versions of the compiler might exploit, it
  is likely that the eventual final aliasing model of Rust will be stricter than Tree Borrows. In
  other words, if you use Tree Borrows, even if your code is accepted today, it might be declared UB
  in the future. This is much less likely with Stacked Borrows.
* `-Zmiri-tree-borrows-no-precise-interior-mut` makes Tree Borrows track interior mutable data on
  the level of references instead of on the byte-level as is done by default. Therefore, with this
  flag, Tree Borrows will be more permissive.
* `-Zmiri-force-page-size=<num>` overrides the default page size for an architecture, in multiples
  of 1k. `4` is default for most targets. This value should always be a power of 2 and nonzero.

Some native rustc `-Z` flags are also very relevant for Miri:

* `-Zmir-opt-level` controls how many MIR optimizations are performed. Miri overrides the default to
  be `0`; be advised that using any higher level can make Miri miss bugs in your program because
  they got optimized away.
* `-Zalways-encode-mir` makes rustc dump MIR even for completely monomorphic functions. This is
  needed so that Miri can execute such functions, so Miri sets this flag per default.
* `-Zmir-emit-retag` controls whether `Retag` statements are emitted. Miri enables this per default
  because it is needed for [Stacked Borrows][26] and [Tree Borrows][27].

Moreover, Miri recognizes some environment variables:

* `MIRIFLAGS` defines extra flags to be passed to Miri.
* `MIRI_LIB_SRC` defines the directory where Miri expects the sources of the standard library that
  it will build and use for interpretation. This directory must point to the `library` subdirectory
  of a `rust-lang/rust` repository checkout.
* `MIRI_SYSROOT` indicates the sysroot to use. When using `cargo miri test`/`cargo miri run`, this
  skips the automatic setup -- only set this if you do not want to use the automatically created
  sysroot. When invoking `cargo miri setup`, this indicates where the sysroot will be put.
* `MIRI_NO_STD` makes sure that the target's sysroot is built without libstd. This allows testing
  and running no_std programs. This should *not usually be used*; Miri has a heuristic to detect
  no-std targets based on the target name. Setting this on a target that does support libstd can
  lead to confusing results.

## Miri `extern` functions

Miri provides some `extern` functions that programs can import to access Miri-specific
functionality. They are declared in [/tests/utils/miri_extern.rs][28].

## Entry point for no-std binaries

Binaries that do not use the standard library are expected to declare a function like this so that
Miri knows where it is supposed to start execution:

#[cfg(miri)]
#[unsafe(no_mangle)]
fn miri_start(argc: isize, argv: *const *const u8) -> isize {
    // Call the actual start function that your project implements, based on your target's conventio
ns.
}

## Contributing and getting help

If you want to contribute to Miri, great! Please check out our [contribution guide][29].

For help with running Miri, you can open an issue here on GitHub or use the [Miri stream on the Rust
Zulip][30].

## History

This project began as part of an undergraduate research course in 2015 by @solson at the [University
of Saskatchewan][31]. There are [slides][32] and a [report][33] available from that project. In
2016, @oli-obk joined to prepare Miri for eventually being used as const evaluator in the Rust
compiler itself (basically, for `const` and `static` stuff), replacing the old evaluator that worked
directly on the AST. In 2017, @RalfJung did an internship with Mozilla and began developing Miri
towards a tool for detecting undefined behavior, and also using Miri as a way to explore the
consequences of various possible definitions for undefined behavior in Rust. @oli-obk's move of the
Miri engine into the compiler finally came to completion in early 2018. Meanwhile, later that year,
@RalfJung did a second internship, developing Miri further with support for checking basic type
invariants and verifying that references are used according to their aliasing restrictions.

## Bugs found by Miri

Miri has already found a number of bugs in the Rust standard library and beyond, some of which we
collect here. If Miri helped you find a subtle UB bug in your code, we'd appreciate a PR adding it
to the list!

Definite bugs found:

* [`Debug for vec_deque::Iter` accessing uninitialized memory][34]
* [`Vec::into_iter` doing an unaligned ZST read][35]
* [`From<&[T]> for Rc` creating a not sufficiently aligned reference][36]
* [`BTreeMap` creating a shared reference pointing to a too small allocation][37]
* [`Vec::append` creating a dangling reference][38]
* [Futures turning a shared reference into a mutable one][39]
* [`str` turning a shared reference into a mutable one][40]
* [`rand` performing unaligned reads][41]
* [The Unix allocator calling `posix_memalign` in an invalid way][42]
* [`getrandom` calling the `getrandom` syscall in an invalid way][43]
* [`Vec`][44] and [`BTreeMap`][45] leaking memory under some (panicky) conditions
* [`beef` leaking memory][46]
* [`EbrCell` using uninitialized memory incorrectly][47]
* [TiKV performing an unaligned pointer access][48]
* [`servo_arc` creating a dangling shared reference][49]
* [TiKV constructing out-of-bounds pointers (and overlapping mutable references)][50]
* [`encoding_rs` doing out-of-bounds pointer arithmetic][51]
* [TiKV using `Vec::from_raw_parts` incorrectly][52]
* Incorrect doctests for [`AtomicPtr`][53] and [`Box::from_raw_in`][54]
* [Insufficient alignment in `ThinVec`][55]
* [`crossbeam-epoch` calling `assume_init` on a partly-initialized `MaybeUninit`][56]
* [`integer-encoding` dereferencing a misaligned pointer][57]
* [`rkyv` constructing a `Box<[u8]>` from an overaligned allocation][58]
* [Data race in `arc-swap`][59]
* [Data race in `thread::scope`][60]
* [`regex` incorrectly handling unaligned `Vec<u8>` buffers][61]
* [Incorrect use of `compare_exchange_weak` in `once_cell`][62]
* [Dropping with unaligned pointers in `vec::IntoIter`][63]
* [Deallocating with the wrong layout in new specializations for in-place `Iterator::collect`][64]
* [Incorrect offset computation for highly-aligned types in `portable-atomic-util`][65]
* [Occasional memory leak in `std::mpsc` channels][66] (original code in [crossbeam][67])
* [Weak-memory-induced memory leak in Windows thread-local storage][68]
* [A bug in the new `RwLock::downgrade` implementation][69] (caught by Miri before it landed in the
  Rust repo)
* [Mockall reading uninitialized memory when mocking `std::io::Read::read`, even if all expectations
  are satisfied][70] (caught by Miri running Tokio's test suite)
* [`ReentrantLock` not correctly dealing with reuse of addresses for TLS storage of different
  threads][71]
* [Rare Deadlock in the thread (un)parking example code][72]
* [`winit` registering a global constructor with the wrong ABI on Windows][73]

Violations of [Stacked Borrows][74] found that are likely bugs (but Stacked Borrows is currently
just an experiment):

* [`VecDeque::drain` creating overlapping mutable references][75]
* Various `BTreeMap` problems
  
  * [`BTreeMap` iterators creating mutable references that overlap with shared references][76]
  * [`BTreeMap::iter_mut` creating overlapping mutable references][77]
  * [`BTreeMap` node insertion using raw pointers outside their valid memory area][78]
* [`LinkedList` cursor insertion creating overlapping mutable references][79]
* [`Vec::push` invalidating existing references into the vector][80]
* [`align_to_mut` violating uniqueness of mutable references][81]
* [`sized-chunks` creating aliasing mutable references][82]
* [`String::push_str` invalidating existing references into the string][83]
* [`ryu` using raw pointers outside their valid memory area][84]
* [ink! creating overlapping mutable references][85]
* [TiKV creating overlapping mutable reference and raw pointer][86]
* [Windows `Env` iterator using a raw pointer outside its valid memory area][87]
* [`VecDeque::iter_mut` creating overlapping mutable references][88]
* [Various standard library aliasing issues involving raw pointers][89]
* [`<[T]>::copy_within` using a loan after invalidating it][90]

## Scientific papers employing Miri

* [Stacked Borrows: An Aliasing Model for Rust][91]
* [Using Lightweight Formal Methods to Validate a Key-Value Storage Node in Amazon S3][92]
* [SyRust: Automatic Testing of Rust Libraries with Semantic-Aware Program Synthesis][93]
* [Crabtree: Rust API Test Synthesis Guided by Coverage and Type][94]
* [Rustlantis: Randomized Differential Testing of the Rust Compiler][95]
* [A Study of Undefined Behavior Across Foreign Function Boundaries in Rust Libraries][96]
* [Tree Borrows][97]
* [Miri: Practical Undefined Behavior Detection for Rust][98] **(this paper describes Miri itself)**

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE][99] or
  [http://www.apache.org/licenses/LICENSE-2.0][100])
* MIT license ([LICENSE-MIT][101] or [http://opensource.org/licenses/MIT][102])

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you shall be dual licensed as above, without any additional terms or conditions.

[1]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
[2]: https://doc.rust-lang.org/stable/std/hint/fn.unreachable_unchecked.html
[3]: https://doc.rust-lang.org/stable/std/ptr/fn.copy_nonoverlapping.html
[4]: https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md
[5]: https://perso.crans.org/vanille/treebor/
[6]: https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md
[7]: #cross-interpretation-running-for-different-targets
[8]: #bugs-found-by-miri
[9]: https://github.com/rust-lang/miri/labels/I-misses-UB
[10]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
[11]: https://github.com/tokio-rs/loom
[12]: https://rust-lang.github.io/unsafe-code-guidelines/glossary.html#soundness-of-code--of-a-libra
ry
[13]: #miri--z-flags-and-environment-variables
[14]: https://doc.rust-lang.org/rustc/platform-support.html
[15]: https://nexte.st
[16]: https://nexte.st/book/miri.html
[17]: https://github.com/nextest-rs/nextest/issues/16
[18]: #miri--z-flags-and-environment-variables
[19]: https://doc.rust-lang.org/nightly/std/ptr/fn.with_exposed_provenance.html
[20]: https://doc.rust-lang.org/nightly/std/ptr/index.html#strict-provenance
[21]: https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md
[22]: https://perso.crans.org/vanille/treebor/
[23]: https://github.com/rust-lang/measureme
[24]: https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md
[25]: https://perso.crans.org/vanille/treebor/
[26]: https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md
[27]: https://perso.crans.org/vanille/treebor/
[28]: /rust-lang/miri/blob/master/tests/utils/miri_extern.rs
[29]: /rust-lang/miri/blob/master/CONTRIBUTING.md
[30]: https://rust-lang.zulipchat.com/#narrow/stream/269128-miri
[31]: https://www.usask.ca/
[32]: https://solson.me/miri-slides.pdf
[33]: https://solson.me/miri-report.pdf
[34]: https://github.com/rust-lang/rust/issues/53566
[35]: https://github.com/rust-lang/rust/pull/53804
[36]: https://github.com/rust-lang/rust/issues/54908
[37]: https://github.com/rust-lang/rust/issues/54957
[38]: https://github.com/rust-lang/rust/pull/61082
[39]: https://github.com/rust-lang/rust/pull/56319
[40]: https://github.com/rust-lang/rust/pull/58200
[41]: https://github.com/rust-random/rand/issues/779
[42]: https://github.com/rust-lang/rust/issues/62251
[43]: https://github.com/rust-random/getrandom/pull/73
[44]: https://github.com/rust-lang/rust/issues/69770
[45]: https://github.com/rust-lang/rust/issues/69769
[46]: https://github.com/maciejhirsz/beef/issues/12
[47]: https://github.com/Firstyear/concread/commit/b15be53b6ec076acb295a5c0483cdb4bf9be838f#diff-628
2b2fc8e98bd089a1f0c86f648157cR229
[48]: https://github.com/tikv/tikv/issues/7613
[49]: https://github.com/servo/servo/issues/26357
[50]: https://github.com/tikv/tikv/pull/7751
[51]: https://github.com/hsivonen/encoding_rs/pull/53
[52]: https://github.com/tikv/agatedb/pull/24
[53]: https://github.com/rust-lang/rust/pull/84052
[54]: https://github.com/rust-lang/rust/pull/84053
[55]: https://github.com/Gankra/thin-vec/pull/27
[56]: https://github.com/crossbeam-rs/crossbeam/pull/779
[57]: https://github.com/dermesser/integer-encoding-rs/pull/23
[58]: https://github.com/rkyv/rkyv/commit/a9417193a34757e12e24263178be8b2eebb72456
[59]: https://github.com/vorner/arc-swap/issues/76
[60]: https://github.com/rust-lang/rust/issues/98498
[61]: https://www.reddit.com/r/rust/comments/vq3mmu/comment/ienc7t0?context=3
[62]: https://github.com/matklad/once_cell/issues/186
[63]: https://github.com/rust-lang/rust/pull/106084
[64]: https://github.com/rust-lang/rust/pull/118460
[65]: https://github.com/taiki-e/portable-atomic/pull/138
[66]: https://github.com/rust-lang/rust/issues/121582
[67]: https://github.com/crossbeam-rs/crossbeam/pull/1084
[68]: https://github.com/rust-lang/rust/pull/124281
[69]: https://rust-lang.zulipchat.com/#narrow/channel/269128-miri/topic/Miri.20error.20library.20tes
t
[70]: https://github.com/asomers/mockall/issues/647
[71]: https://github.com/rust-lang/rust/pull/141248
[72]: https://github.com/rust-lang/rust/issues/145816
[73]: https://github.com/rust-windowing/winit/issues/4435
[74]: https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md
[75]: https://github.com/rust-lang/rust/pull/56161
[76]: https://github.com/rust-lang/rust/pull/58431
[77]: https://github.com/rust-lang/rust/issues/73915
[78]: https://github.com/rust-lang/rust/issues/78477
[79]: https://github.com/rust-lang/rust/pull/60072
[80]: https://github.com/rust-lang/rust/issues/60847
[81]: https://github.com/rust-lang/rust/issues/68549
[82]: https://github.com/bodil/sized-chunks/issues/8
[83]: https://github.com/rust-lang/rust/issues/70301
[84]: https://github.com/dtolnay/ryu/issues/24
[85]: https://github.com/rust-lang/miri/issues/1364
[86]: https://github.com/tikv/tikv/pull/7709
[87]: https://github.com/rust-lang/rust/pull/70479
[88]: https://github.com/rust-lang/rust/issues/74029
[89]: https://github.com/rust-lang/rust/pull/78602
[90]: https://github.com/rust-lang/rust/pull/85610
[91]: https://plv.mpi-sws.org/rustbelt/stacked-borrows/
[92]: https://www.amazon.science/publications/using-lightweight-formal-methods-to-validate-a-key-val
ue-storage-node-in-amazon-s3
[93]: https://dl.acm.org/doi/10.1145/3453483.3454084
[94]: https://dl.acm.org/doi/10.1145/3689733
[95]: https://dl.acm.org/doi/10.1145/3689780
[96]: https://arxiv.org/abs/2404.11671
[97]: https://plf.inf.ethz.ch/research/pldi25-tree-borrows.html
[98]: https://plf.inf.ethz.ch/research/popl26-miri.html
[99]: /rust-lang/miri/blob/master/LICENSE-APACHE
[100]: http://www.apache.org/licenses/LICENSE-2.0
[101]: /rust-lang/miri/blob/master/LICENSE-MIT
[102]: http://opensource.org/licenses/MIT
