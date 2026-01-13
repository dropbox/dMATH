# LeakSanitizer[¶][1]

* [Introduction][2]
* [Usage][3]
* [Security Considerations][4]
* [Supported Platforms][5]
* [More Information][6]

## [Introduction][7][¶][8]

LeakSanitizer is a run-time memory leak detector. It can be combined with [AddressSanitizer][9] to
get both memory error and leak detection, or used in a stand-alone mode. LSan adds almost no
performance overhead until the very end of the process, at which point there is an extra leak
detection phase.

## [Usage][10][¶][11]

[AddressSanitizer][12]: integrates LeakSanitizer and enables it by default on supported platforms.

$ cat memory-leak.c
#include <stdlib.h>
void *p;
int main() {
  p = malloc(7);
  p = 0; // The memory is leaked here.
  return 0;
}
% clang -fsanitize=address -g memory-leak.c ; ASAN_OPTIONS=detect_leaks=1 ./a.out
==23646==ERROR: LeakSanitizer: detected memory leaks
Direct leak of 7 byte(s) in 1 object(s) allocated from:
    #0 0x4af01b in __interceptor_malloc /projects/compiler-rt/lib/asan/asan_malloc_linux.cc:52:3
    #1 0x4da26a in main memory-leak.c:4:7
    #2 0x7f076fd9cec4 in __libc_start_main libc-start.c:287
SUMMARY: AddressSanitizer: 7 byte(s) leaked in 1 allocation(s).

To use LeakSanitizer in stand-alone mode, link your program with `-fsanitize=leak` flag. Make sure
to use `clang` (not `ld`) for the link step, so that it would link in proper LeakSanitizer run-time
library into the final executable.

## [Security Considerations][13][¶][14]

LeakSanitizer is a bug detection tool and its runtime is not meant to be linked against production
executables. While it may be useful for testing, LeakSanitizer’s runtime was not developed with
security-sensitive constraints in mind and may compromise the security of the resulting executable.

## [Supported Platforms][15][¶][16]

* Android
* Fuchsia
* Linux
* macOS
* NetBSD

## [More Information][17][¶][18]

[https://github.com/google/sanitizers/wiki/AddressSanitizerLeakSanitizer][19]

[1]: #leaksanitizer
[2]: #introduction
[3]: #usage
[4]: #security-considerations
[5]: #supported-platforms
[6]: #more-information
[7]: #id1
[8]: #introduction
[9]: AddressSanitizer.html
[10]: #id2
[11]: #usage
[12]: AddressSanitizer.html
[13]: #id3
[14]: #security-considerations
[15]: #id4
[16]: #supported-platforms
[17]: #id5
[18]: #more-information
[19]: https://github.com/google/sanitizers/wiki/AddressSanitizerLeakSanitizer
