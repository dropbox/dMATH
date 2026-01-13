# MemorySanitizer[¶][1]

* [Introduction][2]
* [How to build][3]
* [Usage][4]
  
  * [`__has_feature(memory_sanitizer)`][5]
  * [`__attribute__((no_sanitize("memory")))`][6]
  * [`__attribute__((disable_sanitizer_instrumentation))`][7]
  * [Ignorelist][8]
* [Report symbolization][9]
* [Origin Tracking][10]
* [Use-after-destruction detection][11]
* [Handling external code][12]
* [Security Considerations][13]
* [Supported Platforms][14]
* [Limitations][15]
* [Current Status][16]
* [More Information][17]

## [Introduction][18][¶][19]

MemorySanitizer is a detector of uninitialized memory use. It consists of a compiler instrumentation
module and a run-time library.

Typical slowdown introduced by MemorySanitizer is **3x**.

Here is a not comprehensive of list cases when MemorySanitizer will report an error:

* Uninitialized value was used in a conditional branch.
* Uninitialized pointer was used for memory accesses.
* Uninitialized value was passed or returned from a function call, which is considered an undefined
  behavior. The check can be disabled with `-fno-sanitize-memory-param-retval`.
* Uninitialized data was passed into some libc calls.

## [How to build][20][¶][21]

Build LLVM/Clang with [CMake][22].

## [Usage][23][¶][24]

Simply compile and link your program with `-fsanitize=memory` flag. The MemorySanitizer run-time
library should be linked to the final executable, so make sure to use `clang` (not `ld`) for the
final link step. When linking shared libraries, the MemorySanitizer run-time is not linked, so
`-Wl,-z,defs` may cause link errors (don’t use it with MemorySanitizer). To get a reasonable
performance add `-O1` or higher. To get meaningful stack traces in error messages add
`-fno-omit-frame-pointer`. To get perfect stack traces you may need to disable inlining (just use
`-O1`) and tail call elimination (`-fno-optimize-sibling-calls`).

% cat umr.cc
#include <stdio.h>

int main(int argc, char** argv) {
  int* a = new int[10];
  a[5] = 0;
  if (a[argc])
    printf("xx\n");
  return 0;
}

% clang -fsanitize=memory -fno-omit-frame-pointer -g -O2 umr.cc

If a bug is detected, the program will print an error message to stderr and exit with a non-zero
exit code.

% ./a.out
WARNING: MemorySanitizer: use-of-uninitialized-value
    #0 0x7f45944b418a in main umr.cc:6
    #1 0x7f45938b676c in __libc_start_main libc-start.c:226

By default, MemorySanitizer exits on the first detected error. If you find the error report hard to
understand, try enabling [origin tracking][25].

### [`__has_feature(memory_sanitizer)`][26][¶][27]

In some cases one may need to execute different code depending on whether MemorySanitizer is
enabled. [__has_feature][28] can be used for this purpose.

#if defined(__has_feature)
#  if __has_feature(memory_sanitizer)
// code that builds only under MemorySanitizer
#  endif
#endif

### [`__attribute__((no_sanitize("memory")))`][29][¶][30]

Some code should not be checked by MemorySanitizer. One may use the function attribute
`no_sanitize("memory")` to disable uninitialized checks in a particular function. MemorySanitizer
may still instrument such functions to avoid false positives. This attribute may not be supported by
other compilers, so we suggest to use it together with `__has_feature(memory_sanitizer)`.

### [`__attribute__((disable_sanitizer_instrumentation))`][31][¶][32]

The `disable_sanitizer_instrumentation` attribute can be applied to functions to prevent all kinds
of instrumentation. As a result, it may introduce false positives and therefore should be used with
care, and only if absolutely required; for example for certain code that cannot tolerate any
instrumentation and resulting side-effects. This attribute overrides `no_sanitize("memory")`.

### [Ignorelist][33][¶][34]

MemorySanitizer supports `src` and `fun` entity types in [Sanitizer special case list][35], that can
be used to relax MemorySanitizer checks for certain source files and functions. All “Use of
uninitialized value” warnings will be suppressed and all values loaded from memory will be
considered fully initialized.

## [Report symbolization][36][¶][37]

MemorySanitizer uses an external symbolizer to print files and line numbers in reports. Make sure
that `llvm-symbolizer` binary is in `PATH`, or set environment variable `MSAN_SYMBOLIZER_PATH` to
point to it.

## [Origin Tracking][38][¶][39]

MemorySanitizer can track origins of uninitialized values, similar to Valgrind’s –track-origins
option. This feature is enabled by `-fsanitize-memory-track-origins=2` (or simply
`-fsanitize-memory-track-origins`) Clang option. With the code from the example above,

% cat umr2.cc
#include <stdio.h>

int main(int argc, char** argv) {
  int* a = new int[10];
  a[5] = 0;
  volatile int b = a[argc];
  if (b)
    printf("xx\n");
  return 0;
}

% clang -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer -g -O2 umr2.cc
% ./a.out
WARNING: MemorySanitizer: use-of-uninitialized-value
    #0 0x7f7893912f0b in main umr2.cc:7
    #1 0x7f789249b76c in __libc_start_main libc-start.c:226

  Uninitialized value was stored to memory at
    #0 0x7f78938b5c25 in __msan_chain_origin msan.cc:484
    #1 0x7f7893912ecd in main umr2.cc:6

  Uninitialized value was created by a heap allocation
    #0 0x7f7893901cbd in operator new[](unsigned long) msan_new_delete.cc:44
    #1 0x7f7893912e06 in main umr2.cc:4

By default, MemorySanitizer collects both allocation points and all intermediate stores the
uninitialized value went through. Origin tracking has proved to be very useful for debugging
MemorySanitizer reports. It slows down program execution by a factor of 1.5x-2x on top of the usual
MemorySanitizer slowdown and increases memory overhead.

Clang option `-fsanitize-memory-track-origins=1` enables a slightly faster mode when MemorySanitizer
collects only allocation points but not intermediate stores.

## [Use-after-destruction detection][40][¶][41]

MemorySanitizer includes use-after-destruction detection. After invocation of the destructor, the
object will be considered no longer readable, and using underlying memory will lead to error reports
in runtime. Refer to the standard for [lifetime][42] definition.

This feature can be disabled with either:

1. Pass additional Clang option `-fno-sanitize-memory-use-after-dtor` during compilation.
2. Set environment variable MSAN_OPTIONS=poison_in_dtor=0 before running the program.

## [Handling external code][43][¶][44]

MemorySanitizer requires that all program code is instrumented. This also includes any libraries
that the program depends on, even libc. Failing to achieve this may result in false reports. For the
same reason you may need to replace all inline assembly code that writes to memory with a pure C/C++
code.

Full MemorySanitizer instrumentation is very difficult to achieve. To make it easier,
MemorySanitizer runtime library includes 70+ interceptors for the most common libc functions. They
make it possible to run MemorySanitizer-instrumented programs linked with uninstrumented libc. For
example, the authors were able to bootstrap MemorySanitizer-instrumented Clang compiler by linking
it with self-built instrumented libc++ (as a replacement for libstdc++).

## [Security Considerations][45][¶][46]

MemorySanitizer is a bug detection tool and its runtime is not meant to be linked against production
executables. While it may be useful for testing, MemorySanitizer’s runtime was not developed with
security-sensitive constraints in mind and may compromise the security of the resulting executable.

## [Supported Platforms][47][¶][48]

MemorySanitizer is supported on the following OS:

* Linux
* NetBSD
* FreeBSD

## [Limitations][49][¶][50]

* MemorySanitizer uses 2x more real memory than a native run, 3x with origin tracking.
* MemorySanitizer maps (but not reserves) 64 Terabytes of virtual address space. This means that
  tools like `ulimit` may not work as usually expected.
* Static linking is not supported.
* Older versions of MSan (LLVM 3.7 and older) didn’t work with non-position-independent executables,
  and could fail on some Linux kernel versions with disabled ASLR. Refer to documentation for older
  versions for more details.
* MemorySanitizer might be incompatible with position-independent executables from FreeBSD 13 but
  there is a check done at runtime and throws a warning in this case.

## [Current Status][51][¶][52]

MemorySanitizer is known to work on large real-world programs (like Clang/LLVM itself) that can be
recompiled from source, including all dependent libraries.

## [More Information][53][¶][54]

[https://github.com/google/sanitizers/wiki/MemorySanitizer][55]

[1]: #memorysanitizer
[2]: #introduction
[3]: #how-to-build
[4]: #usage
[5]: #has-feature-memory-sanitizer
[6]: #attribute-no-sanitize-memory
[7]: #attribute-disable-sanitizer-instrumentation
[8]: #ignorelist
[9]: #report-symbolization
[10]: #origin-tracking
[11]: #use-after-destruction-detection
[12]: #handling-external-code
[13]: #security-considerations
[14]: #supported-platforms
[15]: #limitations
[16]: #current-status
[17]: #more-information
[18]: #id1
[19]: #introduction
[20]: #id2
[21]: #how-to-build
[22]: https://llvm.org/docs/CMake.html
[23]: #id3
[24]: #usage
[25]: #msan-origins
[26]: #id4
[27]: #has-feature-memory-sanitizer
[28]: LanguageExtensions.html#langext-has-feature-has-extension
[29]: #id5
[30]: #attribute-no-sanitize-memory
[31]: #id6
[32]: #attribute-disable-sanitizer-instrumentation
[33]: #id7
[34]: #ignorelist
[35]: SanitizerSpecialCaseList.html
[36]: #id8
[37]: #report-symbolization
[38]: #id9
[39]: #origin-tracking
[40]: #id10
[41]: #use-after-destruction-detection
[42]: https://eel.is/c++draft/basic.life#1
[43]: #id11
[44]: #handling-external-code
[45]: #id12
[46]: #security-considerations
[47]: #id13
[48]: #supported-platforms
[49]: #id14
[50]: #limitations
[51]: #id15
[52]: #current-status
[53]: #id16
[54]: #more-information
[55]: https://github.com/google/sanitizers/wiki/MemorySanitizer
