[ [LLVM Logo]][1]

### Navigation

* [index][2]
* [next][3] |
* [previous][4] |
* [LLVM Home][5] |
* [Documentation][6]»
* [Reference][7] »
* [libFuzzer – a library for coverage-guided fuzz testing.][8]

### Documentation

* [Getting Started/Tutorials][9]
* [User Guides][10]
* [Reference][11]

### Getting Involved

* [Contributing to LLVM][12]
* [Submitting Bug Reports][13]
* [Mailing Lists][14]
* [Discord][15]
* [Meetups and Social Events][16]

### Additional Links

* [FAQ][17]
* [Glossary][18]
* [Publications][19]
* [Github Repository][20]

### This Page

* [Show Source][21]

### Quick search

# libFuzzer – a library for coverage-guided fuzz testing.[¶][22]

* [Introduction][23]
* [Status][24]
* [Versions][25]
* [Getting Started][26]
* [Options][27]
* [Output][28]
* [Examples][29]
* [Advanced features][30]
* [Developing libFuzzer][31]
* [FAQ][32]
* [Trophies][33]

## [Introduction][34][¶][35]

LibFuzzer is an in-process, coverage-guided, evolutionary fuzzing engine.

LibFuzzer is linked with the library under test, and feeds fuzzed inputs to the library via a
specific fuzzing entrypoint (aka “target function”); the fuzzer then tracks which areas of the code
are reached, and generates mutations on the corpus of input data in order to maximize the code
coverage. The code coverage information for libFuzzer is provided by LLVM’s [SanitizerCoverage][36]
instrumentation.

Contact: libfuzzer(#)googlegroups.com

## [Status][37][¶][38]

The original authors of libFuzzer have stopped active work on it and switched to working on another
fuzzing engine, [Centipede][39]. LibFuzzer is still fully supported in that important bugs will get
fixed. However, please do not expect major new features or code reviews, other than for bug fixes.

## [Versions][40][¶][41]

LibFuzzer requires a matching version of Clang.

## [Getting Started][42][¶][43]

* [Fuzz Target][44]
* [Fuzzer Usage][45]
* [Corpus][46]
* [Running][47]
* [Parallel Fuzzing][48]
* [Fork mode][49]
* [Resuming merge][50]

### [Fuzz Target][51][¶][52]

The first step in using libFuzzer on a library is to implement a *fuzz target* – a function that
accepts an array of bytes and does something interesting with these bytes using the API under test.
Like this:

// fuzz_target.cc
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  DoSomethingInterestingWithMyAPI(Data, Size);
  return 0;  // Values other than 0 and -1 are reserved for future use.
}

Note that this fuzz target does not depend on libFuzzer in any way and so it is possible and even
desirable to use it with other fuzzing engines e.g. [AFL][53] and/or [Radamsa][54].

Some important things to remember about fuzz targets:

* The fuzzing engine will execute the fuzz target many times with different inputs in the same
  process.
* It must tolerate any kind of input (empty, huge, malformed, etc).
* It must not exit() on any input.
* It may use threads but ideally all threads should be joined at the end of the function.
* It must be as deterministic as possible. Non-determinism (e.g. random decisions not based on the
  input bytes) will make fuzzing inefficient.
* It must be fast. Try avoiding cubic or greater complexity, logging, or excessive memory
  consumption.
* Ideally, it should not modify any global state (although that’s not strict).
* Usually, the narrower the target the better. E.g. if your target can parse several data formats,
  split it into several targets, one per format.

### [Fuzzer Usage][55][¶][56]

Recent versions of Clang (starting from 6.0) include libFuzzer, and no extra installation is
necessary.

In order to build your fuzzer binary, use the -fsanitize=fuzzer flag during the compilation and
linking. In most cases you may want to combine libFuzzer with [AddressSanitizer][57] (ASAN),
[UndefinedBehaviorSanitizer][58] (UBSAN), or both. You can also build with [MemorySanitizer][59]
(MSAN), but support is experimental:

clang -g -O1 -fsanitize=fuzzer                         mytarget.c # Builds the fuzz target w/o sanit
izers
clang -g -O1 -fsanitize=fuzzer,address                 mytarget.c # Builds the fuzz target with ASAN
clang -g -O1 -fsanitize=fuzzer,signed-integer-overflow mytarget.c # Builds the fuzz target with a pa
rt of UBSAN
clang -g -O1 -fsanitize=fuzzer,memory                  mytarget.c # Builds the fuzz target with MSAN

This will perform the necessary instrumentation, as well as linking with the libFuzzer library. Note
that `-fsanitize=fuzzer` links in the libFuzzer’s `main()` symbol.

If modifying `CFLAGS` of a large project, which also compiles executables requiring their own `main`
symbol, it may be desirable to request just the instrumentation without linking:

clang -fsanitize=fuzzer-no-link mytarget.c

Then libFuzzer can be linked to the desired driver by passing in `-fsanitize=fuzzer` during the
linking stage.

### [Corpus][60][¶][61]

Coverage-guided fuzzers like libFuzzer rely on a corpus of sample inputs for the code under test.
This corpus should ideally be seeded with a varied collection of valid and invalid inputs for the
code under test; for example, for a graphics library the initial corpus might hold a variety of
different small PNG/JPG/GIF files. The fuzzer generates random mutations based around the sample
inputs in the current corpus. If a mutation triggers execution of a previously-uncovered path in the
code under test, then that mutation is saved to the corpus for future variations.

LibFuzzer will work without any initial seeds, but will be less efficient if the library under test
accepts complex, structured inputs.

The corpus can also act as a sanity/regression check, to confirm that the fuzzing entrypoint still
works and that all of the sample inputs run through the code under test without problems.

If you have a large corpus (either generated by fuzzing or acquired by other means) you may want to
minimize it while still preserving the full coverage. One way to do that is to use the -merge=1
flag:

mkdir NEW_CORPUS_DIR  # Store minimized corpus here.
./my_fuzzer -merge=1 NEW_CORPUS_DIR FULL_CORPUS_DIR

You may use the same flag to add more interesting items to an existing corpus. Only the inputs that
trigger new coverage will be added to the first corpus.

./my_fuzzer -merge=1 CURRENT_CORPUS_DIR NEW_POTENTIALLY_INTERESTING_INPUTS_DIR

### [Running][62][¶][63]

To run the fuzzer, first create a [Corpus][64] directory that holds the initial “seed” sample
inputs:

mkdir CORPUS_DIR
cp /some/input/samples/* CORPUS_DIR

Then run the fuzzer on the corpus directory:

./my_fuzzer CORPUS_DIR  # -max_len=1000 -jobs=20 ...

As the fuzzer discovers new interesting test cases (i.e. test cases that trigger coverage of new
paths through the code under test), those test cases will be added to the corpus directory.

By default, the fuzzing process will continue indefinitely – at least until a bug is found. Any
crashes or sanitizer failures will be reported as usual, stopping the fuzzing process, and the
particular input that triggered the bug will be written to disk (typically as `crash-<sha1>`,
`leak-<sha1>`, or `timeout-<sha1>`).

### [Parallel Fuzzing][65][¶][66]

Each libFuzzer process is single-threaded, unless the library under test starts its own threads.
However, it is possible to run multiple libFuzzer processes in parallel with a shared corpus
directory; this has the advantage that any new inputs found by one fuzzer process will be available
to the other fuzzer processes (unless you disable this with the `-reload=0` option).

This is primarily controlled by the `-jobs=N` option, which indicates that that N fuzzing jobs
should be run to completion (i.e. until a bug is found or time/iteration limits are reached). These
jobs will be run across a set of worker processes, by default using half of the available CPU cores;
the count of worker processes can be overridden by the `-workers=N` option. For example, running
with `-jobs=30` on a 12-core machine would run 6 workers by default, with each worker averaging 5
bugs by completion of the entire process.

### [Fork mode][67][¶][68]

**Experimental** mode `-fork=N` (where `N` is the number of parallel jobs) enables oom-, timeout-,
and crash-resistant fuzzing with separate processes (using `fork-exec`, not just `fork`).

The top libFuzzer process will not do any fuzzing itself, but will spawn up to `N` concurrent child
processes providing them small random subsets of the corpus. After a child exits, the top process
merges the corpus generated by the child back to the main corpus.

Related flags:

*`-ignore_ooms`*
  True by default. If an OOM happens during fuzzing in one of the child processes, the reproducer is
  saved on disk, and fuzzing continues.
*`-ignore_timeouts`*
  True by default, same as `-ignore_ooms`, but for timeouts.
*`-ignore_crashes`*
  False by default, same as `-ignore_ooms`, but for all other crashes.

The plan is to eventually replace `-jobs=N` and `-workers=N` with `-fork=N`.

### [Resuming merge][69][¶][70]

Merging large corpora may be time consuming, and it is often desirable to do it on preemptable VMs,
where the process may be killed at any time. In order to seamlessly resume the merge, use the
`-merge_control_file` flag and use `killall -SIGUSR1 /path/to/fuzzer/binary` to stop the merge
gracefully. Example:

% rm -f SomeLocalPath
% ./my_fuzzer CORPUS1 CORPUS2 -merge=1 -merge_control_file=SomeLocalPath
...
MERGE-INNER: using the control file 'SomeLocalPath'
...
# While this is running, do `killall -SIGUSR1 my_fuzzer` in another console
==9015== INFO: libFuzzer: exiting as requested

# This will leave the file SomeLocalPath with the partial state of the merge.
# Now, you can continue the merge by executing the same command. The merge
# will continue from where it has been interrupted.
% ./my_fuzzer CORPUS1 CORPUS2 -merge=1 -merge_control_file=SomeLocalPath
...
MERGE-OUTER: non-empty control file provided: 'SomeLocalPath'
MERGE-OUTER: control file ok, 32 files total, first not processed file 20
...

## [Options][71][¶][72]

To run the fuzzer, pass zero or more corpus directories as command line arguments. The fuzzer will
read test inputs from each of these corpus directories, and any new test inputs that are generated
will be written back to the first corpus directory:

./fuzzer [-flag1=val1 [-flag2=val2 ...] ] [dir1 [dir2 ...] ]

If a list of files (rather than directories) are passed to the fuzzer program, then it will re-run
those files as test inputs but will not perform any fuzzing. In this mode the fuzzer binary can be
used as a regression test (e.g. on a continuous integration system) to check the target function and
saved inputs still work.

The most important command line options are:

*`-help`*
  Print help message (`-help=1`).
*`-seed`*
  Random seed. If 0 (the default), the seed is generated.
*`-runs`*
  Number of individual test runs, -1 (the default) to run indefinitely.
*`-max_len`*
  Maximum length of a test input. If 0 (the default), libFuzzer tries to guess a good value based on
  the corpus (and reports it).
*`-len_control`*
  Try generating small inputs first, then try larger inputs over time. Specifies the rate at which
  the length limit is increased (smaller == faster). Default is 100. If 0, immediately try inputs
  with size up to max_len.
*`-timeout`*
  Timeout in seconds, default 1200. If an input takes longer than this timeout, the process is
  treated as a failure case.
*`-rss_limit_mb`*
  Memory usage limit in Mb, default 2048. Use 0 to disable the limit. If an input requires more than
  this amount of RSS memory to execute, the process is treated as a failure case. The limit is
  checked in a separate thread every second. If running w/o ASAN/MSAN, you may use ‘ulimit -v’
  instead.
*`-malloc_limit_mb`*
  If non-zero, the fuzzer will exit if the target tries to allocate this number of Mb with one
  malloc call. If zero (default) same limit as rss_limit_mb is applied.
*`-timeout_exitcode`*
  Exit code (default 77) used if libFuzzer reports a timeout.
*`-error_exitcode`*
  Exit code (default 77) used if libFuzzer itself (not a sanitizer) reports a bug (leak, OOM, etc).
*`-max_total_time`*
  If positive, indicates the maximum total time in seconds to run the fuzzer. If 0 (the default),
  run indefinitely.
*`-merge`*
  If set to 1, any corpus inputs from the 2nd, 3rd etc. corpus directories that trigger new code
  coverage will be merged into the first corpus directory. Defaults to 0. This flag can be used to
  minimize a corpus.
*`-merge_control_file`*
  Specify a control file used for the merge process. If a merge process gets killed it tries to
  leave this file in a state suitable for resuming the merge. By default a temporary file will be
  used.
*`-minimize_crash`*
  If 1, minimizes the provided crash input. Use with -runs=N or -max_total_time=N to limit the
  number of attempts.
*`-reload`*
  If set to 1 (the default), the corpus directory is re-read periodically to check for new inputs;
  this allows detection of new inputs that were discovered by other fuzzing processes.
*`-jobs`*
  Number of fuzzing jobs to run to completion. Default value is 0, which runs a single fuzzing
  process until completion. If the value is >= 1, then this number of jobs performing fuzzing are
  run, in a collection of parallel separate worker processes; each such worker process has its
  `stdout`/`stderr` redirected to `fuzz-<JOB>.log`.
*`-workers`*
  Number of simultaneous worker processes to run the fuzzing jobs to completion in. If 0 (the
  default), `min(jobs, NumberOfCpuCores()/2)` is used.
*`-dict`*
  Provide a dictionary of input keywords; see [Dictionaries][73].
*`-use_counters`*
  Use [coverage counters][74] to generate approximate counts of how often code blocks are hit;
  defaults to 1.
*`-reduce_inputs`*
  Try to reduce the size of inputs while preserving their full feature sets; defaults to 1.
*`-use_value_profile`*
  Use [value profile][75] to guide corpus expansion; defaults to 0.
*`-only_ascii`*
  If 1, generate only ASCII (`isprint``+``isspace`) inputs. Defaults to 0.
*`-artifact_prefix`*
  Provide a prefix to use when saving fuzzing artifacts (crash, timeout, or slow inputs) as
  `$(artifact_prefix)file`. Defaults to empty.
*`-exact_artifact_path`*
  Ignored if empty (the default). If non-empty, write the single artifact on failure (crash,
  timeout) as `$(exact_artifact_path)`. This overrides `-artifact_prefix` and will not use checksum
  in the file name. Do not use the same path for several parallel processes.
*`-print_pcs`*
  If 1, print out newly covered PCs. Defaults to 0.
*`-print_final_stats`*
  If 1, print statistics at exit. Defaults to 0.
*`-detect_leaks`*
  If 1 (default) and if LeakSanitizer is enabled try to detect memory leaks during fuzzing (i.e. not
  only at shut down).
*`-close_fd_mask`*
  Indicate output streams to close at startup. Be careful, this will remove diagnostic output from
  target code (e.g. messages on assert failure).
  
  > * 0 (default): close neither `stdout` nor `stderr`
  > * 1 : close `stdout`
  > * 2 : close `stderr`
  > * 3 : close both `stdout` and `stderr`.

For the full list of flags run the fuzzer binary with `-help=1`.

## [Output][76][¶][77]

During operation the fuzzer prints information to `stderr`, for example:

INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 1434179311
INFO: Loaded 1 modules   (8 inline 8-bit counters): 8 [0x5f03d189be90, 0x5f03d189be98),
INFO: Loaded 1 PC tables (8 PCs): 8 [0x5f03d189be98,0x5f03d189bf18),
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: A corpus is not provided, starting from an empty corpus
#2      INITED cov: 2 ft: 2 corp: 1/1b exec/s: 0 rss: 31Mb
#144    NEW    cov: 3 ft: 3 corp: 2/2b lim: 4 exec/s: 0 rss: 31Mb L: 1/1 MS: 2 ChangeByte-ChangeByte
-
#157    NEW    cov: 4 ft: 4 corp: 3/4b lim: 4 exec/s: 0 rss: 31Mb L: 2/2 MS: 3 CrossOver-ChangeBit-C
rossOver-
#1345   NEW    cov: 5 ft: 5 corp: 4/8b lim: 14 exec/s: 0 rss: 32Mb L: 4/4 MS: 3 InsertByte-ChangeBit
-CrossOver-
#1696   NEW    cov: 6 ft: 6 corp: 5/10b lim: 17 exec/s: 0 rss: 32Mb L: 2/4 MS: 1 EraseBytes-
#1832   REDUCE cov: 6 ft: 6 corp: 5/9b lim: 17 exec/s: 0 rss: 32Mb L: 3/3 MS: 1 EraseBytes-
...

The early parts of the output include information about the fuzzer options and configuration,
including the current random seed (in the `Seed:` line; this can be overridden with the `-seed=N`
flag).

Further output lines have the form of an event code and statistics. The possible event codes are:

*`READ`*
  The fuzzer has read in all of the provided input samples from the corpus directories.
*`INITED`*
  The fuzzer has completed initialization, which includes running each of the initial input samples
  through the code under test.
*`NEW`*
  The fuzzer has created a test input that covers new areas of the code under test. This input will
  be saved to the primary corpus directory.
*`REDUCE`*
  The fuzzer has found a better (smaller) input that triggers previously discovered features (set
  `-reduce_inputs=0` to disable).
*`pulse`*
  The fuzzer has generated 2^{n} inputs (generated periodically to reassure the user that the fuzzer
  is still working).
*`DONE`*
  The fuzzer has completed operation because it has reached the specified iteration limit (`-runs`)
  or time limit (`-max_total_time`).
*`RELOAD`*
  The fuzzer is performing a periodic reload of inputs from the corpus directory; this allows it to
  discover any inputs discovered by other fuzzer processes (see [Parallel Fuzzing][78]).

Each output line also reports the following statistics (when non-zero):

*`cov:`*
  Total number of code blocks or edges covered by executing the current corpus.
*`ft:`*
  libFuzzer uses different signals to evaluate the code coverage: edge coverage, edge counters,
  value profiles, indirect caller/callee pairs, etc. These signals combined are called *features*
  (ft:).
*`corp:`*
  Number of entries in the current in-memory test corpus and its size in bytes.
*`lim:`*
  Current limit on the length of new entries in the corpus. Increases over time until the max length
  (`-max_len`) is reached.
*`exec/s:`*
  Number of fuzzer iterations per second.
*`rss:`*
  Current memory consumption.

For `NEW` and `REDUCE` events, the output line also includes information about the mutation
operation that produced the new input:

*`L:`*
  Size of the new/reduced input in bytes and the size of the largest input in current in-memory test
  corpus.
*`MS: <n> <operations>`*
  Count and list of the mutation operations used to generate the input.

## [Examples][79][¶][80]

* [Toy example][81]
* [More examples][82]

### [Toy example][83][¶][84]

A simple function that does something interesting if it receives the input “HI!”:

cat << EOF > test_fuzzer.cc
#include <stdint.h>
#include <stddef.h>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size > 0 && data[0] == 'H')
    if (size > 1 && data[1] == 'I')
       if (size > 2 && data[2] == '!')
       __builtin_trap();
  return 0;
}
EOF
# Build test_fuzzer.cc with asan and link against libFuzzer.
clang++ -fsanitize=address,fuzzer test_fuzzer.cc
# Run the fuzzer with no corpus.
./a.out

You should get an error pretty quickly:

INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 1434179311
INFO: Loaded 1 modules   (8 inline 8-bit counters): 8 [0x5f03d189be90, 0x5f03d189be98),
INFO: Loaded 1 PC tables (8 PCs): 8 [0x5f03d189be98,0x5f03d189bf18),
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: A corpus is not provided, starting from an empty corpus
#2      INITED cov: 2 ft: 2 corp: 1/1b exec/s: 0 rss: 31Mb
#144    NEW    cov: 3 ft: 3 corp: 2/2b lim: 4 exec/s: 0 rss: 31Mb L: 1/1 MS: 2 ChangeByte-ChangeByte
-
#157    NEW    cov: 4 ft: 4 corp: 3/4b lim: 4 exec/s: 0 rss: 31Mb L: 2/2 MS: 3 CrossOver-ChangeBit-C
rossOver-
#1345   NEW    cov: 5 ft: 5 corp: 4/8b lim: 14 exec/s: 0 rss: 32Mb L: 4/4 MS: 3 InsertByte-ChangeBit
-CrossOver-
#1696   NEW    cov: 6 ft: 6 corp: 5/10b lim: 17 exec/s: 0 rss: 32Mb L: 2/4 MS: 1 EraseBytes-
#1832   REDUCE cov: 6 ft: 6 corp: 5/9b lim: 17 exec/s: 0 rss: 32Mb L: 3/3 MS: 1 EraseBytes-
==840148== ERROR: libFuzzer: deadly signal
...
SUMMARY: libFuzzer: deadly signal
MS: 2 CopyPart-ChangeByte-; base unit: dbee5f8c7a5da845446e75b4a5708e74428b520a
0x48,0x49,0x21,
HI!
artifact_prefix='./'; Test unit written to ./crash-7a8dc3985d2a90fb6e62e94910fc11d31949c348
Base64: SEkh

### [More examples][85][¶][86]

Examples of real-life fuzz targets and the bugs they find can be found at
[http://tutorial.libfuzzer.info][87]. Among other things you can learn how to detect
[Heartbleed][88] in one second.

## [Advanced features][89][¶][90]

* [Dictionaries][91]
* [Tracing CMP instructions][92]
* [Value Profile][93]
* [Fuzzer-friendly build mode][94]
* [AFL compatibility][95]
* [How good is my fuzzer?][96]
* [User-supplied mutators][97]
* [Startup initialization][98]
* [Using libFuzzer as a library][99]
* [Rejecting unwanted inputs][100]
* [Leaks][101]

### [Dictionaries][102][¶][103]

LibFuzzer supports user-supplied dictionaries with input language keywords or other interesting byte
sequences (e.g. multi-byte magic values). Use `-dict=DICTIONARY_FILE`. For some input languages
using a dictionary may significantly improve the search speed. The dictionary syntax is similar to
that used by [AFL][104] for its `-x` option:

# Lines starting with '#' and empty lines are ignored.

# Adds "blah" (w/o quotes) to the dictionary.
kw1="blah"
# Use \\ for backslash and \" for quotes.
kw2="\"ac\\dc\""
# Use \xAB for hex values
kw3="\xF7\xF8"
# the name of the keyword followed by '=' may be omitted:
"foo\x0Abar"

### [Tracing CMP instructions][105][¶][106]

With an additional compiler flag `-fsanitize-coverage=trace-cmp` (on by default as part of
`-fsanitize=fuzzer`, see [SanitizerCoverageTraceDataFlow][107]) libFuzzer will intercept CMP
instructions and guide mutations based on the arguments of intercepted CMP instructions. This may
slow down the fuzzing but is very likely to improve the results.

### [Value Profile][108][¶][109]

With `-fsanitize-coverage=trace-cmp` (default with `-fsanitize=fuzzer`) and extra run-time flag
`-use_value_profile=1` the fuzzer will collect value profiles for the parameters of compare
instructions and treat some new values as new coverage.

The current implementation does roughly the following:

* The compiler instruments all CMP instructions with a callback that receives both CMP arguments.
* The callback computes (caller_pc&4095) | (popcnt(Arg1 ^ Arg2) << 12) and uses this value to set a
  bit in a bitset.
* Every new observed bit in the bitset is treated as new coverage.

This feature has a potential to discover many interesting inputs, but there are two downsides.
First, the extra instrumentation may bring up to 2x additional slowdown. Second, the corpus may grow
by several times.

### [Fuzzer-friendly build mode][110][¶][111]

Sometimes the code under test is not fuzzing-friendly. Examples:

> * The target code uses a PRNG seeded e.g. by system time and thus two consequent invocations may
>   potentially execute different code paths even if the end result will be the same. This will
>   cause a fuzzer to treat two similar inputs as significantly different and it will blow up the
>   test corpus. E.g. libxml uses `rand()` inside its hash table.
> * The target code uses checksums to protect from invalid inputs. E.g. png checks CRC for every
>   chunk.

In many cases it makes sense to build a special fuzzing-friendly build with certain
fuzzing-unfriendly features disabled. We propose to use a common build macro for all such cases for
consistency: `FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION`.

void MyInitPRNG() {
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  // In fuzzing mode the behavior of the code should be deterministic.
  srand(0);
#else
  srand(time(0));
#endif
}

### [AFL compatibility][112][¶][113]

LibFuzzer can be used together with [AFL][114] on the same test corpus. Both fuzzers expect the test
corpus to reside in a directory, one file per input. You can run both fuzzers on the same corpus,
one after another:

./afl-fuzz -i testcase_dir -o findings_dir /path/to/program @@
./llvm-fuzz testcase_dir findings_dir  # Will write new tests to testcase_dir

Periodically restart both fuzzers so that they can use each other’s findings. Currently, there is no
simple way to run both fuzzing engines in parallel while sharing the same corpus dir.

You may also use AFL on your target function `LLVMFuzzerTestOneInput`: see an example [here][115].

### [How good is my fuzzer?][116][¶][117]

Once you implement your target function `LLVMFuzzerTestOneInput` and fuzz it to death, you will want
to know whether the function or the corpus can be improved further. One easy to use metric is, of
course, code coverage.

We recommend to use [Clang Coverage][118], to visualize and study your code coverage
([example][119]).

### [User-supplied mutators][120][¶][121]

LibFuzzer allows to use custom (user-supplied) mutators, see [Structure-Aware Fuzzing][122] for more
details.

### [Startup initialization][123][¶][124]

If the library being tested needs to be initialized, there are several options.

The simplest way is to have a statically initialized global object inside LLVMFuzzerTestOneInput (or
in global scope if that works for you):

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  static bool Initialized = DoInitialization();
  ...

Alternatively, you may define an optional init function and it will receive the program arguments
that you can read and modify. Do this **only** if you really need to access `argv`/`argc`.

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
 ReadAndMaybeModify(argc, argv);
 return 0;
}

### [Using libFuzzer as a library][125][¶][126]

If the code being fuzzed must provide its own main, it’s possible to invoke libFuzzer as a library.
Be sure to pass `-fsanitize=fuzzer-no-link` during compilation, and link your binary against the
no-main version of libFuzzer. On Linux installations, this is typically located at:

/usr/lib/<llvm-version>/lib/clang/<clang-version>/lib/linux/libclang_rt.fuzzer_no_main-<architecture
>.a

If building libFuzzer from source, this is located at the following path in the build output
directory:

lib/linux/libclang_rt.fuzzer_no_main-<architecture>.a

From here, the code can do whatever setup it requires, and when it’s ready to start fuzzing, it can
call LLVMFuzzerRunDriver, passing in the program arguments and a callback. This callback is invoked
just like LLVMFuzzerTestOneInput, and has the same signature.

extern "C" int LLVMFuzzerRunDriver(int *argc, char ***argv,
                  int (*UserCb)(const uint8_t *Data, size_t Size));

### [Rejecting unwanted inputs][127][¶][128]

It may be desirable to reject some inputs, i.e. to not add them to the corpus.

For example, when fuzzing an API consisting of parsing and other logic, one may want to allow only
those inputs into the corpus that parse successfully.

If the fuzz target returns -1 on a given input, libFuzzer will not add that input top the corpus,
regardless of what coverage it triggers.

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (auto *Obj = ParseMe(Data, Size)) {
    Obj->DoSomethingInteresting();
    return 0;  // Accept. The input may be added to the corpus.
  }
  return -1;  // Reject; The input will not be added to the corpus.
}

### [Leaks][129][¶][130]

Binaries built with [AddressSanitizer][131] or [LeakSanitizer][132] will try to detect memory leaks
at the process shutdown. For in-process fuzzing this is inconvenient since the fuzzer needs to
report a leak with a reproducer as soon as the leaky mutation is found. However, running full leak
detection after every mutation is expensive.

By default (`-detect_leaks=1`) libFuzzer will count the number of `malloc` and `free` calls when
executing every mutation. If the numbers don’t match (which by itself doesn’t mean there is a leak)
libFuzzer will invoke the more expensive [LeakSanitizer][133] pass and if the actual leak is found,
it will be reported with the reproducer and the process will exit.

If your target has massive leaks and the leak detection is disabled you will eventually run out of
RAM (see the `-rss_limit_mb` flag).

## [Developing libFuzzer][134][¶][135]

LibFuzzer is built as a part of LLVM project by default on macos and Linux. Users of other operating
systems can explicitly request compilation using `-DCOMPILER_RT_BUILD_LIBFUZZER=ON` flag. Tests are
run using `check-fuzzer` target from the build directory which was configured with
`-DCOMPILER_RT_INCLUDE_TESTS=ON` flag.

ninja check-fuzzer

## [FAQ][136][¶][137]

### Q. Why doesn’t libFuzzer use any of the LLVM support?[¶][138]

There are two reasons.

First, we want this library to be used outside of the LLVM without users having to build the rest of
LLVM. This may sound unconvincing for many LLVM folks, but in practice the need for building the
whole LLVM frightens many potential users – and we want more users to use this code.

Second, there is a subtle technical reason not to rely on the rest of LLVM, or any other large body
of code (maybe not even STL). When coverage instrumentation is enabled, it will also instrument the
LLVM support code which will blow up the coverage set of the process (since the fuzzer is
in-process). In other words, by using more external dependencies we will slow down the fuzzer while
the main reason for it to exist is extreme speed.

### Q. Does libFuzzer Support Windows?[¶][139]

Yes, libFuzzer now supports Windows. Initial support was added in r341082. Any build of Clang 9
supports it. You can download a build of Clang for Windows that has libFuzzer from [LLVM Snapshot
Builds][140].

Using libFuzzer on Windows without ASAN is unsupported. Building fuzzers with the `/MD` (dynamic
runtime library) compile option is unsupported. Support for these may be added in the future.
Linking fuzzers with the `/INCREMENTAL` link option (or the `/DEBUG` option which implies it) is
also unsupported.

Send any questions or comments to the mailing list: libfuzzer(#)googlegroups.com

### Q. When libFuzzer is not a good solution for a problem?[¶][141]

* If the test inputs are validated by the target library and the validator asserts/crashes on
  invalid inputs, in-process fuzzing is not applicable.
* Bugs in the target library may accumulate without being detected. E.g. a memory corruption that
  goes undetected at first and then leads to a crash while testing another input. This is why it is
  highly recommended to run this in-process fuzzer with all sanitizers to detect most bugs on the
  spot.
* It is harder to protect the in-process fuzzer from excessive memory consumption and infinite loops
  in the target library (still possible).
* The target library should not have significant global state that is not reset between the runs.
* Many interesting target libraries are not designed in a way that supports the in-process fuzzer
  interface (e.g. require a file path instead of a byte array).
* If a single test run takes a considerable fraction of a second (or more) the speed benefit from
  the in-process fuzzer is negligible.
* If the target library runs persistent threads (that outlive execution of one test) the fuzzing
  results will be unreliable.

### Q. So, what exactly this Fuzzer is good for?[¶][142]

This Fuzzer might be a good choice for testing libraries that have relatively small inputs, each
input takes < 10ms to run, and the library code is not expected to crash on invalid inputs.
Examples: regular expression matchers, text or binary format parsers, compression, network, crypto.

### Q. LibFuzzer crashes on my complicated fuzz target (but works fine for me on smaller
### targets).[¶][143]

Check if your fuzz target uses `dlclose`. Currently, libFuzzer doesn’t support targets that call
`dlclose`, this may be fixed in future.

## [Trophies][144][¶][145]

* Thousands of bugs found on OSS-Fuzz:
  [https://opensource.googleblog.com/2017/05/oss-fuzz-five-months-later-and.html][146]
* GLIBC: [https://sourceware.org/glibc/wiki/FuzzingLibc][147]
* MUSL LIBC: [[1]][148] [[2]][149]
* [pugixml][150]
* PCRE: Search for “LLVM fuzzer” in
  [http://vcs.pcre.org/pcre2/code/trunk/ChangeLog?view=markup][151]; also in [bugzilla][152]
* [ICU][153]
* [Freetype][154]
* [Harfbuzz][155]
* [SQLite][156]
* [Python][157]
* OpenSSL/BoringSSL: [[1]][158] [[2]][159] [[3]][160] [[4]][161] [[5]][162] [[6]][163]
* [Libxml2][164] and [[HT206167]][165] (CVE-2015-5312, CVE-2015-7500, CVE-2015-7942)
* [Linux Kernel’s BPF verifier][166]
* [Linux Kernel’s Crypto code][167]
* Capstone: [[1]][168] [[2]][169]
* file:[[1]][170] [[2]][171] [[3]][172] [[4]][173]
* Radare2: [[1]][174]
* gRPC: [[1]][175] [[2]][176] [[3]][177] [[4]][178] [[5]][179] [[6]][180]
* WOFF2: [[1]][181]
* LLVM: [Clang][182], [Clang-format][183], [libc++][184], [llvm-as][185], [Demangler][186],
  Disassembler: [http://reviews.llvm.org/rL247405][187], [http://reviews.llvm.org/rL247414][188],
  [http://reviews.llvm.org/rL247416][189], [http://reviews.llvm.org/rL247417][190],
  [http://reviews.llvm.org/rL247420][191], [http://reviews.llvm.org/rL247422][192].
* Tensorflow: [[1]][193]
* Ffmpeg: [[1]][194] [[2]][195] [[3]][196] [[4]][197]
* [Wireshark][198]
* [QEMU][199]

### Navigation

* [index][200]
* [next][201] |
* [previous][202] |
* [LLVM Home][203] |
* [Documentation][204]»
* [Reference][205] »
* [libFuzzer – a library for coverage-guided fuzz testing.][206]
© Copyright 2003-2025, LLVM Project. Last updated on 2025-12-22. Created using [Sphinx][207] 7.2.6.

[1]: index.html
[2]: genindex.html
[3]: MarkedUpDisassembly.html
[4]: LangRef.html
[5]: https://llvm.org/
[6]: index.html
[7]: Reference.html
[8]: 
[9]: https://llvm.org/docs/GettingStartedTutorials.html
[10]: https://llvm.org/docs/UserGuides.html
[11]: https://llvm.org/docs/Reference.html
[12]: https://llvm.org/docs/Contributing.html
[13]: https://llvm.org/docs/HowToSubmitABug.html
[14]: https://llvm.org/docs/GettingInvolved.html#mailing-lists
[15]: https://llvm.org/docs/GettingInvolved.html#discord
[16]: https://llvm.org/docs/GettingInvolved.html#meetups-and-social-events
[17]: https://llvm.org/docs/FAQ.html
[18]: https://llvm.org/docs/Lexicon.html
[19]: https://llvm.org/pubs
[20]: https://github.com/llvm/llvm-project/
[21]: _sources/LibFuzzer.rst.txt
[22]: #libfuzzer-a-library-for-coverage-guided-fuzz-testing
[23]: #introduction
[24]: #status
[25]: #versions
[26]: #getting-started
[27]: #options
[28]: #output
[29]: #examples
[30]: #advanced-features
[31]: #developing-libfuzzer
[32]: #faq
[33]: #trophies
[34]: #id12
[35]: #introduction
[36]: https://clang.llvm.org/docs/SanitizerCoverage.html
[37]: #id13
[38]: #status
[39]: https://github.com/google/centipede
[40]: #id14
[41]: #versions
[42]: #id15
[43]: #getting-started
[44]: #fuzz-target
[45]: #fuzzer-usage
[46]: #corpus
[47]: #running
[48]: #parallel-fuzzing
[49]: #fork-mode
[50]: #resuming-merge
[51]: #id23
[52]: #fuzz-target
[53]: http://lcamtuf.coredump.cx/afl/
[54]: https://github.com/aoh/radamsa
[55]: #id24
[56]: #fuzzer-usage
[57]: https://clang.llvm.org/docs/AddressSanitizer.html
[58]: https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
[59]: https://clang.llvm.org/docs/MemorySanitizer.html
[60]: #id25
[61]: #corpus
[62]: #id26
[63]: #running
[64]: #corpus
[65]: #id27
[66]: #parallel-fuzzing
[67]: #id28
[68]: #fork-mode
[69]: #id29
[70]: #resuming-merge
[71]: #id16
[72]: #options
[73]: #dictionaries
[74]: https://clang.llvm.org/docs/SanitizerCoverage.html#coverage-counters
[75]: #value-profile
[76]: #id17
[77]: #output
[78]: #parallel-fuzzing
[79]: #id18
[80]: #examples
[81]: #toy-example
[82]: #more-examples
[83]: #id30
[84]: #toy-example
[85]: #id31
[86]: #more-examples
[87]: http://tutorial.libfuzzer.info
[88]: http://en.wikipedia.org/wiki/Heartbleed
[89]: #id19
[90]: #advanced-features
[91]: #dictionaries
[92]: #tracing-cmp-instructions
[93]: #value-profile
[94]: #fuzzer-friendly-build-mode
[95]: #afl-compatibility
[96]: #how-good-is-my-fuzzer
[97]: #user-supplied-mutators
[98]: #startup-initialization
[99]: #using-libfuzzer-as-a-library
[100]: #rejecting-unwanted-inputs
[101]: #leaks
[102]: #id32
[103]: #dictionaries
[104]: http://lcamtuf.coredump.cx/afl/
[105]: #id33
[106]: #tracing-cmp-instructions
[107]: https://clang.llvm.org/docs/SanitizerCoverage.html#tracing-data-flow
[108]: #id34
[109]: #value-profile
[110]: #id35
[111]: #fuzzer-friendly-build-mode
[112]: #id36
[113]: #afl-compatibility
[114]: http://lcamtuf.coredump.cx/afl/
[115]: https://github.com/llvm/llvm-project/tree/main/compiler-rt/lib/fuzzer/afl
[116]: #id37
[117]: #how-good-is-my-fuzzer
[118]: https://clang.llvm.org/docs/SourceBasedCodeCoverage.html
[119]: https://github.com/google/fuzzer-test-suite/blob/master/tutorial/libFuzzerTutorial.md#visuali
zing-coverage
[120]: #id38
[121]: #user-supplied-mutators
[122]: https://github.com/google/fuzzing/blob/master/docs/structure-aware-fuzzing.md
[123]: #id39
[124]: #startup-initialization
[125]: #id40
[126]: #using-libfuzzer-as-a-library
[127]: #id41
[128]: #rejecting-unwanted-inputs
[129]: #id42
[130]: #leaks
[131]: https://clang.llvm.org/docs/AddressSanitizer.html
[132]: https://clang.llvm.org/docs/LeakSanitizer.html
[133]: https://clang.llvm.org/docs/LeakSanitizer.html
[134]: #id20
[135]: #developing-libfuzzer
[136]: #id21
[137]: #faq
[138]: #q-why-doesn-t-libfuzzer-use-any-of-the-llvm-support
[139]: #q-does-libfuzzer-support-windows
[140]: https://llvm.org/builds/
[141]: #q-when-libfuzzer-is-not-a-good-solution-for-a-problem
[142]: #q-so-what-exactly-this-fuzzer-is-good-for
[143]: #q-libfuzzer-crashes-on-my-complicated-fuzz-target-but-works-fine-for-me-on-smaller-targets
[144]: #id22
[145]: #trophies
[146]: https://opensource.googleblog.com/2017/05/oss-fuzz-five-months-later-and.html
[147]: https://sourceware.org/glibc/wiki/FuzzingLibc
[148]: http://git.musl-libc.org/cgit/musl/commit/?id=39dfd58417ef642307d90306e1c7e50aaec5a35c
[149]: http://www.openwall.com/lists/oss-security/2015/03/30/3
[150]: https://github.com/zeux/pugixml/issues/39
[151]: http://vcs.pcre.org/pcre2/code/trunk/ChangeLog?view=markup
[152]: https://bugs.exim.org/buglist.cgi?bug_status=__all__&content=libfuzzer&no_redirect=1&order=Im
portance&product=PCRE&query_format=specific
[153]: http://bugs.icu-project.org/trac/ticket/11838
[154]: https://savannah.nongnu.org/search/?words=LibFuzzer&type_of_search=bugs&Search=Search&exact=1
#options
[155]: https://github.com/behdad/harfbuzz/issues/139
[156]: http://www3.sqlite.org/cgi/src/info/088009efdd56160b
[157]: http://bugs.python.org/issue25388
[158]: https://boringssl.googlesource.com/boringssl/+/cb852981cd61733a7a1ae4fd8755b7ff950e857d
[159]: https://openssl.org/news/secadv/20160301.txt
[160]: https://boringssl.googlesource.com/boringssl/+/2b07fa4b22198ac02e0cee8f37f3337c3dba91bc
[161]: https://boringssl.googlesource.com/boringssl/+/6b6e0b20893e2be0e68af605a60ffa2cbb0ffa64
[162]: https://github.com/openssl/openssl/pull/931/commits/dd5ac557f052cc2b7f718ac44a8cb7ac6f77dca8
[163]: https://github.com/openssl/openssl/pull/931/commits/19b5b9194071d1d84e38ac9a952e715afbc85a81
[164]: https://bugzilla.gnome.org/buglist.cgi?bug_status=__all__&content=libFuzzer&list_id=68957&ord
er=Importance&product=libxml2&query_format=specific
[165]: https://support.apple.com/en-gb/HT206167
[166]: https://github.com/iovisor/bpf-fuzzer
[167]: https://www.spinics.net/lists/stable/msg199712.html
[168]: https://github.com/aquynh/capstone/issues/600
[169]: https://github.com/aquynh/capstone/commit/6b88d1d51eadf7175a8f8a11b690684443b11359
[170]: http://bugs.gw.com/view.php?id=550
[171]: http://bugs.gw.com/view.php?id=551
[172]: http://bugs.gw.com/view.php?id=553
[173]: http://bugs.gw.com/view.php?id=554
[174]: https://github.com/revskills?tab=contributions&from=2016-04-09
[175]: https://github.com/grpc/grpc/pull/6071/commits/df04c1f7f6aec6e95722ec0b023a6b29b6ea871c
[176]: https://github.com/grpc/grpc/pull/6071/commits/22a3dfd95468daa0db7245a4e8e6679a52847579
[177]: https://github.com/grpc/grpc/pull/6071/commits/9cac2a12d9e181d130841092e9d40fa3309d7aa7
[178]: https://github.com/grpc/grpc/pull/6012/commits/82a91c91d01ce9b999c8821ed13515883468e203
[179]: https://github.com/grpc/grpc/pull/6202/commits/2e3e0039b30edaf89fb93bfb2c1d0909098519fa
[180]: https://github.com/grpc/grpc/pull/6106/files
[181]: https://github.com/google/woff2/commit/a15a8ab
[182]: https://bugs.llvm.org/show_bug.cgi?id=23057
[183]: https://bugs.llvm.org/show_bug.cgi?id=23052
[184]: https://bugs.llvm.org/show_bug.cgi?id=24411
[185]: https://bugs.llvm.org/show_bug.cgi?id=24639
[186]: https://bugs.chromium.org/p/chromium/issues/detail?id=606626
[187]: http://reviews.llvm.org/rL247405
[188]: http://reviews.llvm.org/rL247414
[189]: http://reviews.llvm.org/rL247416
[190]: http://reviews.llvm.org/rL247417
[191]: http://reviews.llvm.org/rL247420
[192]: http://reviews.llvm.org/rL247422
[193]: https://da-data.blogspot.com/2017/01/finding-bugs-in-tensorflow-with.html
[194]: https://github.com/FFmpeg/FFmpeg/commit/c92f55847a3d9cd12db60bfcd0831ff7f089c37c
[195]: https://github.com/FFmpeg/FFmpeg/commit/25ab1a65f3acb5ec67b53fb7a2463a7368f1ad16
[196]: https://github.com/FFmpeg/FFmpeg/commit/85d23e5cbc9ad6835eef870a5b4247de78febe56
[197]: https://github.com/FFmpeg/FFmpeg/commit/04bd1b38ee6b8df410d0ab8d4949546b6c4af26a
[198]: https://bugs.wireshark.org/bugzilla/buglist.cgi?bug_status=UNCONFIRMED&bug_status=CONFIRMED&b
ug_status=IN_PROGRESS&bug_status=INCOMPLETE&bug_status=RESOLVED&bug_status=VERIFIED&f0=OP&f1=OP&f2=p
roduct&f3=component&f4=alias&f5=short_desc&f7=content&f8=CP&f9=CP&j1=OR&o2=substring&o3=substring&o4
=substring&o5=substring&o6=substring&o7=matches&order=bug_id%20DESC&query_format=advanced&v2=libfuzz
er&v3=libfuzzer&v4=libfuzzer&v5=libfuzzer&v6=libfuzzer&v7=%22libfuzzer%22
[199]: https://researchcenter.paloaltonetworks.com/2017/09/unit42-palo-alto-networks-discovers-new-q
emu-vulnerability/
[200]: genindex.html
[201]: MarkedUpDisassembly.html
[202]: LangRef.html
[203]: https://llvm.org/
[204]: index.html
[205]: Reference.html
[206]: 
[207]: https://www.sphinx-doc.org/
