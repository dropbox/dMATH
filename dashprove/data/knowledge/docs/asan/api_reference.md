[ google ][1] / ** [sanitizers][2] ** Public

* [ Notifications ][3] You must be signed in to change notification settings
* [ Fork 1.1k ][4]
* [ Star 12.3k ][5]

* [ Code ][6]
* [ Issues 550 ][7]
* [ Pull requests 0 ][8]
* [ Actions ][9]
* [ Projects 0 ][10]
* [ Wiki ][11]
* [ Security ][12]
  [
  
  ### Uh oh!
  
  
  ][13]
* [ Insights ][14]
Additional navigation options

* [ Code ][15]
* [ Issues ][16]
* [ Pull requests ][17]
* [ Actions ][18]
* [ Projects ][19]
* [ Wiki ][20]
* [ Security ][21]
* [ Insights ][22]

# AddressSanitizer

[Jump to bottom][23]
chefmax edited this page May 15, 2019 · [29 revisions][24]

# Introduction

[AddressSanitizer][25] (aka ASan) is a memory error detector for C/C++. It finds:

* [Use after free][26] (dangling pointer dereference)
* [Heap buffer overflow][27]
* [Stack buffer overflow][28]
* [Global buffer overflow][29]
* [Use after return][30]
* [Use after scope][31]
* [Initialization order bugs][32]
* [Memory leaks][33]

This tool is very fast. The average slowdown of the instrumented program is ~2x (see
[AddressSanitizerPerformanceNumbers][34]).

The tool consists of a compiler instrumentation module (currently, an LLVM pass) and a run-time
library which replaces the `malloc` function.

The tool works on x86, ARM, MIPS (both 32- and 64-bit versions of all architectures), PowerPC64. The
supported operation systems are Linux, Darwin (OS X and iOS Simulator), FreeBSD, Android:

─────────────┬───┬──────┬───┬─────┬────┬──────┬───────┬─────────
OS           │x86│x86_64│ARM│ARM64│MIPS│MIPS64│PowerPC│PowerPC64
─────────────┼───┼──────┼───┼─────┼────┼──────┼───────┼─────────
Linux        │yes│yes   │   │     │yes │yes   │yes    │yes      
─────────────┼───┼──────┼───┼─────┼────┼──────┼───────┼─────────
OS X         │yes│yes   │   │     │    │      │       │         
─────────────┼───┼──────┼───┼─────┼────┼──────┼───────┼─────────
iOS Simulator│yes│yes   │   │     │    │      │       │         
─────────────┼───┼──────┼───┼─────┼────┼──────┼───────┼─────────
FreeBSD      │yes│yes   │   │     │    │      │       │         
─────────────┼───┼──────┼───┼─────┼────┼──────┼───────┼─────────
Android      │yes│yes   │yes│yes  │    │      │       │         
─────────────┴───┴──────┴───┴─────┴────┴──────┴───────┴─────────

Other OS/arch combinations may work as well, but aren't actively developed/tested.

See also:

* [AddressSanitizerAlgorithm][35] -- if you are curious how it works.
* [AddressSanitizerComparisonOfMemoryTools][36]

# Getting AddressSanitizer

[AddressSanitizer][37] is a part of [LLVM][38] starting with version 3.1 and a part of [GCC][39]
starting with version 4.8 If you prefer to build from source, see [AddressSanitizerHowToBuild][40].

So far, [AddressSanitizer][41] has been tested only on Linux Ubuntu 12.04, 64-bit (it can run both
64- and 32-bit programs), Mac 10.6, 10.7 and 10.8, and [AddressSanitizerOnAndroid][42] 4.2+.

# Using AddressSanitizer

In order to use [AddressSanitizer][43] you will need to compile and link your program using `clang`
with the `-fsanitize=address` switch. To get a reasonable performance add `-O1` or higher. To get
nicer stack traces in error messages add `-fno-omit-frame-pointer`. Note: [Clang 3.1 release uses
another flag syntax][44].

`% cat tests/use-after-free.c
#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char*));
  free(x);
  return x[5];
}
% ../clang_build_Linux/Release+Asserts/bin/clang -fsanitize=address -O1 -fno-omit-frame-pointer -g  
 tests/use-after-free.c
`

Now, run the executable. [AddressSanitizerCallStack][45] page describes how to obtain symbolized
stack traces.

`% ./a.out
==9901==ERROR: AddressSanitizer: heap-use-after-free on address 0x60700000dfb5 at pc 0x45917b bp 0x7
fff4490c700 sp 0x7fff4490c6f8
READ of size 1 at 0x60700000dfb5 thread T0
    #0 0x45917a in main use-after-free.c:5
    #1 0x7fce9f25e76c in __libc_start_main /build/buildd/eglibc-2.15/csu/libc-start.c:226
    #2 0x459074 in _start (a.out+0x459074)
0x60700000dfb5 is located 5 bytes inside of 80-byte region [0x60700000dfb0,0x60700000e000)
freed by thread T0 here:
    #0 0x4441ee in __interceptor_free projects/compiler-rt/lib/asan/asan_malloc_linux.cc:64
    #1 0x45914a in main use-after-free.c:4
    #2 0x7fce9f25e76c in __libc_start_main /build/buildd/eglibc-2.15/csu/libc-start.c:226
previously allocated by thread T0 here:
    #0 0x44436e in __interceptor_malloc projects/compiler-rt/lib/asan/asan_malloc_linux.cc:74
    #1 0x45913f in main use-after-free.c:3
    #2 0x7fce9f25e76c in __libc_start_main /build/buildd/eglibc-2.15/csu/libc-start.c:226
SUMMARY: AddressSanitizer: heap-use-after-free use-after-free.c:5 main
`

# Interaction with other tools

## gdb

See [AddressSanitizerAndDebugger][46]

## ulimit -v

The `ulimit -v` command makes little sense with ASan-ified binaries because ASan consumes 20
terabytes of virtual memory (plus a bit).

You may try more sophisticated tools to limit your memory consumption, e.g.
[https://en.wikipedia.org/wiki/Cgroups][47]

# Flags

See the separate [AddressSanitizerFlags][48] page.

# Call stack

See the separate [AddressSanitizerCallStack][49] page.

# Incompatibility

Sometimes an [AddressSanitizer][50] build may behave differently than the regular one. See
[AddressSanitizerIncompatiblity][51] for details.

# Turning off instrumentation

In some cases a particular function should be ignored (not instrumented) by [AddressSanitizer][52]:

* Ignore a very hot function known to be correct to speedup the app.
* Ignore a function that does some low-level magic (e.g. walking through the thread's stack
  bypassing the frame boundaries).
* Don't report a known problem. In either case, be **very careful**.

To ignore certain functions, one can use the **no_sanitize_address** attribute supported by Clang
(3.3+) and GCC (4.8+). You can define the following macro:

`#if defined(__clang__) || defined (__GNUC__)
# define ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
# define ATTRIBUTE_NO_SANITIZE_ADDRESS
#endif
...
ATTRIBUTE_NO_SANITIZE_ADDRESS
void ThisFunctionWillNotBeInstrumented() {...}
`

Clang 3.1 and 3.2 supported `__attribute__((no_address_safety_analysis))` instead.

You may also ignore certain functions using a blacklist: create a file `my_ignores.txt` and pass it
to [AddressSanitizer][53] at compile time using `-fsanitize-blacklist=my_ignores.txt` (This flag is
new and is only supported by Clang now):

`# Ignore exactly this function (the names are mangled)
fun:MyFooBar
# Ignore MyFooBar(void) if it is in C++:
fun:_Z8MyFooBarv
# Ignore all function containing MyFooBar
fun:*MyFooBar*
`

# FAQ

* Q: Can [AddressSanitizer][54] continue running after reporting first error?
* A: Yes it can, AddressSanitizer has recently got continue-after-error mode. This is somewhat
  experimental so may not yet be as reliable as default setting (and not as timely supported). Also
  keep in mind that errors after the first one may actually be spurious. To enable
  continue-after-error, compile with `-fsanitize-recover=address` and then run your code with
  `ASAN_OPTIONS=halt_on_error=0`.
* Q: Why didn't ASan report an obviously invalid memory access in my code?
* A1: If your errors is too obvious, compiler might have already optimized it out by the time Asan
  runs.
* A2: Another, C-only option is accesses to global common symbols which are not protected by Asan
  (you can use -fno-common to disable generation of common symbols and hopefully detect more bugs).
* A3: If _FORTIFY_SOURCE is enabled, ASan may have false positives, see next question.
* Q: I've compiled my code with -D_FORTIFY_SOURCE flag and ASan, or -D_FORTIFY_SOURCE is enabled by
  default in my distribution (most modern distributions). Now ASan misbehaves (either produces false
  warnings, or does not find some bugs).
* A: Currently ASan (and other sanitizers) doesn't support source fortification, see
  [https://github.com/google/sanitizers/issues/247][55]. The fix should most likely be on the glibc
  side, see the (stalled) discussion [here][56].
* Q: When I link my shared library with -fsanitize=address, it fails due to some undefined ASan
  symbols (e.g. asan_init_v4)?
* A: Most probably you link with -Wl,-z,defs or -Wl,--no-undefined. These flags don't work with ASan
  unless you also use -shared-libasan (which is the default mode for GCC, but not for Clang).
* Q: My malloc stacktraces are too short or do not make sense?
* A: Try to compile your code with -fno-omit-frame-pointer or set
  ASAN_OPTIONS=fast_unwind_on_malloc=0 (the latter would be a performance killer though unless you
  also specify malloc_context_size=2 or lower). Note that frame-pointer-based unwinding does not
  work on Thumb.
* Q: My new() and delete() stacktraces are too short or do not make sense?
* A: This may happen when the C++ standard library is linked statically. Prebuilt libstdc++/libc++
  often do not use frame pointers, and it breaks fast (frame-pointer-based) unwinding. Either switch
  to the shared library with the -shared-libstdc++ flag, or use
  ASAN_OPTIONS=fast_unwind_on_malloc=0. The latter could be very slow.
* Q: I'm using dynamic ASan runtime and my program crashes at start with "Shadow memory range
  interleaves with an existing memory mapping. ASan cannot proceed correctly.".
* A1: If you are using shared ASan DSO, try LD_PRELOAD'ing Asan runtime into your program.
* A2: Otherwise you are probably hitting a known limitation of dynamic runtime. Libasan is
  initialized at the end of program startup so if some preceeding library initializer did lots of
  memory allocations memory region required for ASan shadow memory could be occupied by unrelated
  mappings.
* Q: The PC printed in ASan stack traces is consistently off by 1?
* A: This is not a bug but rather a design choice. It is hard to compute exact size of preceding
  instruction on CISC platforms. So ASan just decrements 1 which is enough for tools like addr2line
  or readelf to symbolize addresses.
* Q: I've ran with ASAN_OPTIONS=verbosity=1 and ASan tells something like
` ==30654== Parsed ASAN_OPTIONS: verbosity=1
 ==30654== AddressSanitizer: failed to intercept 'memcpy'
`

* A: This warning is false (see [https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58680][57] for
  details).
* Q: I've built my main executable with ASan. Do I also need to build shared libraries?
* A: ASan will work even if you rebuild just part of your program. But you'll have to rebuild all
  components to detect all errors.
* Q: I've built my shared library with ASan. Can I run it with unsanitized executable?
* A: Yes! You'll need to build your library with [dynamic version of ASan][58] and then run
  executable with LD_PRELOAD=path/to/asan/runtime/lib.
* Q: On Linux I am seeings a crash at startup with something like this
`ERROR: AddressSanitizer failed to allocate 0x400000000 (17179869184) bytes at address 67fff8000 (er
rno: 12)
`

* A: Make sure you don't have `2` in `/proc/sys/vm/overcommit_memory`
* Q: I'm working on a project that uses bare-metal OS with no pthread (TLS) support and no POSIX
  syscalls and want to use ASan, but its code depends on some stuff (e.g. **dlsym**) that is
  unavailable on my platform. Does ASan support bare-metal targets?
* A: Out of the box we don't have support for your use case. The easiest for you would be to rip off
  everything you don't have and rebuild the ASan run-time. However, there have been many attempts in
  applying ASan to bare-metal and at least some were successful. E.g.
  [http://events.linuxfoundation.org/sites/events/files/slides/Alexander_Popov-KASan_in_a_Bare-Metal
  _Hypervisor_0.pdf][59] and also grep for "bare-metal" and similar stuff in
  [https://groups.google.com/forum/#!forum/address-sanitizer][60] group.
* Q: Can I run AddressSanitizer with more aggressive diagnostics enabled?
* A: Yes! In particular you may want to enable
` CFLAGS += -fsanitize-address-use-after-scope
 ASAN_OPTIONS=strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:st
rict_init_order=1
`

check [Flags wiki] ([https://github.com/google/sanitizers/wiki/AddressSanitizerFlags][61] ) for more
details on this.

* Q: My library crashes with SIGABRT while calling `free`. What's going wrong?
* A: Most probably you are dlopening your library with **RTLD_DEEPBIND** flag. ASan doesn't support
  **RTLD_DEEPBIND**, see [issue #611][62] for details.
* Q: I'm getting following error when building my code: `gcc: error: cannot specify -static with
  -fsanitize=address`.
* A: ASan doesn't work with static linkage. You need to disable static linkage in order to use ASan
  on your code.
* Q: What do `0xbebebebebebebebe` and `0xbebebebe` mean?
* A: ASan, by default, writes `0xbe` to newly allocated memory (see [malloc_fill_byte][63]).
  `0xbebebebebebebebe` is (possibly) a 64-bit value that was allocated but not initialized.
  Similarly for `0xbebebebe`.
* Q: Can I mix code instrumented by Clang with code instrumented by GCC? Or can I at least compile
  code with Clang and link against GCC ASan runtime and vice versa?
* A: No, you cannot, Clang and GCC have completely incompatible ASan implementations, you cannot mix
  them in any way.

# Talks and papers

* Watch the presentation from the [LLVM Developer's meeting][64] (Nov 18, 2011): [Video][65],
  [slides][66].
* Read the [USENIX ATC '2012 paper][67].

# Comments?

Send comments to [address-sanitizer@googlegroups.com][68] or [in Google+][69].

## Toggle table of contents Pages 78

* Loading [
  Home
  ][70]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][71].
* Loading [
  AddressSanitizer
  ][72]
  
  * [Introduction][73]
  * [Getting AddressSanitizer][74]
  * [Using AddressSanitizer][75]
  * [Interaction with other tools][76]
  * [gdb][77]
  * [ulimit -v][78]
  * [Flags][79]
  * [Call stack][80]
  * [Incompatibility][81]
  * [Turning off instrumentation][82]
  * [FAQ][83]
  * [Talks and papers][84]
  * [Comments?][85]
* Loading [
  AddressSanitizerAlgorithm
  ][86]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][87].
* Loading [
  AddressSanitizerAndDebugger
  ][88]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][89].
* Loading [
  AddressSanitizerAndroidPlatform
  ][90]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][91].
* Loading [
  AddressSanitizerAsDso
  ][92]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][93].
* Loading [
  AddressSanitizerBasicBlockTracing
  ][94]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][95].
* Loading [
  AddressSanitizerCallStack
  ][96]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][97].
* Loading [
  AddressSanitizerClangVsGCC (3.8 vs 6.0)
  ][98]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][99].
* Loading [
  AddressSanitizerClangVsGCC (5.0 vs 7.1)
  ][100]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][101].
* Loading [
  AddressSanitizerClangVsGCC (6.0 vs 8.1)
  ][102]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][103].
* Loading [
  AddressSanitizerComparisonOfMemoryTools
  ][104]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][105].
* Loading [
  AddressSanitizerCompileTimeOptimizations
  ][106]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][107].
* Loading [
  AddressSanitizerContainerOverflow
  ][108]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][109].
* Loading [
  AddressSanitizerExampleGlobalOutOfBounds
  ][110]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][111].
* Loading [
  AddressSanitizerExampleHeapOutOfBounds
  ][112]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][113].
* Loading [
  AddressSanitizerExampleInitOrderFiasco
  ][114]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][115].
* Loading [
  AddressSanitizerExampleStackOutOfBounds
  ][116]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][117].
* Loading [
  AddressSanitizerExampleUseAfterFree
  ][118]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][119].
* Loading [
  AddressSanitizerExampleUseAfterReturn
  ][120]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][121].
* Loading [
  AddressSanitizerExampleUseAfterScope
  ][122]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][123].
* Loading [
  AddressSanitizerFlags
  ][124]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][125].
* Loading [
  AddressSanitizerForKernel
  ][126]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][127].
* Loading [
  AddressSanitizerFoundBugs
  ][128]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][129].
* Loading [
  AddressSanitizerHowToBuild
  ][130]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][131].
* Loading [
  AddressSanitizerHowToContribute
  ][132]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][133].
* Loading [
  AddressSanitizerIncompatiblity
  ][134]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][135].
* Loading [
  AddressSanitizerInHardware
  ][136]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][137].
* Loading [
  AddressSanitizerInitializationOrderFiasco
  ][138]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][139].
* Loading [
  AddressSanitizerIntelMemoryProtectionExtensions
  ][140]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][141].
* Loading [
  AddressSanitizerIntraObjectOverflow
  ][142]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][143].
* Loading [
  AddressSanitizerLeakSanitizer
  ][144]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][145].
* Loading [
  AddressSanitizerLeakSanitizerDesignDocument
  ][146]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][147].
* Loading [
  AddressSanitizerLeakSanitizerVsHeapChecker
  ][148]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][149].
* Loading [
  AddressSanitizerLogo
  ][150]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][151].
* Loading [
  AddressSanitizerManualPoisoning
  ][152]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][153].
* Loading [
  AddressSanitizerOnAndroid
  ][154]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][155].
* Loading [
  AddressSanitizerOnAndroidO
  ][156]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][157].
* Loading [
  AddressSanitizerOneDefinitionRuleViolation
  ][158]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][159].
* Loading [
  AddressSanitizerPerformanceNumbers
  ][160]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][161].
* Loading [
  AddressSanitizerRunningSpecBenchmarks
  ][162]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][163].
* Loading [
  AddressSanitizerSupportedPlatforms
  ][164]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][165].
* Loading [
  AddressSanitizerTestSuite
  ][166]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][167].
* Loading [
  AddressSanitizerUseAfterReturn
  ][168]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][169].
* Loading [
  AddressSanitizerUseAfterScope
  ][170]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][171].
* Loading [
  AddressSanitizerWindowsPort
  ][172]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][173].
* Loading [
  AddressSanitizerZeroBasedShadow
  ][174]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][175].
* Loading [
  MemorySanitizer
  ][176]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][177].
* Loading [
  MemorySanitizerBootstrappingClang
  ][178]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][179].
* Loading [
  MemorySanitizerBuildingClangOnOlderSystems
  ][180]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][181].
* Loading [
  MemorySanitizerFAQ
  ][182]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][183].
* Loading [
  MemorySanitizerFoundBugs
  ][184]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][185].
* Loading [
  MemorySanitizerInstrumentingLibstdcxx
  ][186]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][187].
* Loading [
  MemorySanitizerJIT
  ][188]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][189].
* Loading [
  MemorySanitizerLibcxxHowTo
  ][190]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][191].
* Loading [
  MemorySanitizerRunning
  ][192]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][193].
* Loading [
  MemorySanitizerUsingGit
  ][194]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][195].
* Loading [
  SanitizerBotReproduceBuild
  ][196]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][197].
* Loading [
  SanitizerCommonFlags
  ][198]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][199].
* Loading [
  Stack instrumentation with ARM Memory Tagging Extension (MTE)
  ][200]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][201].
* Loading [
  ThreadSanitizerAboutRaces
  ][202]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][203].
* Loading [
  ThreadSanitizerAlgorithm
  ][204]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][205].
* Loading [
  ThreadSanitizerAtomicOperations
  ][206]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][207].
* Loading [
  ThreadSanitizerBenchmarkingApache
  ][208]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][209].
* Loading [
  ThreadSanitizerCppManual
  ][210]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][211].
* Loading [
  ThreadSanitizerDeadlockDetector
  ][212]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][213].
* Loading [
  ThreadSanitizerDetectableBugs
  ][214]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][215].
* Loading [
  ThreadSanitizerDevelopment
  ][216]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][217].
* Loading [
  ThreadSanitizerFlags
  ][218]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][219].
* Loading [
  ThreadSanitizerForKernel
  ][220]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][221].
* Loading [
  ThreadSanitizerForKernelReports
  ][222]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][223].
* Loading [
  ThreadSanitizerFoundBugs
  ][224]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][225].
* Loading [
  ThreadSanitizerGoManual
  ][226]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][227].
* Loading [
  ThreadSanitizerOnAndroid
  ][228]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][229].
* Loading [
  ThreadSanitizerPopularDataRaces
  ][230]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][231].
* Loading [
  ThreadSanitizerReportFormat
  ][232]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][233].
* Loading [
  ThreadSanitizerSuppressions
  ][234]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][235].
* Loading [
  ThreadSanitizerVolatileRanges
  ][236]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][237].
* Show 63 more pages…

### Clone this wiki locally

[1]: /google
[2]: /google/sanitizers
[3]: /login?return_to=%2Fgoogle%2Fsanitizers
[4]: /login?return_to=%2Fgoogle%2Fsanitizers
[5]: /login?return_to=%2Fgoogle%2Fsanitizers
[6]: /google/sanitizers
[7]: /google/sanitizers/issues
[8]: /google/sanitizers/pulls
[9]: /google/sanitizers/actions
[10]: /google/sanitizers/projects
[11]: /google/sanitizers/wiki
[12]: /google/sanitizers/security
[13]: /google/sanitizers/security
[14]: /google/sanitizers/pulse
[15]: /google/sanitizers
[16]: /google/sanitizers/issues
[17]: /google/sanitizers/pulls
[18]: /google/sanitizers/actions
[19]: /google/sanitizers/projects
[20]: /google/sanitizers/wiki
[21]: /google/sanitizers/security
[22]: /google/sanitizers/pulse
[23]: #wiki-pages-box
[24]: /google/sanitizers/wiki/AddressSanitizer/_history
[25]: AddressSanitizer
[26]: AddressSanitizerExampleUseAfterFree
[27]: AddressSanitizerExampleHeapOutOfBounds
[28]: AddressSanitizerExampleStackOutOfBounds
[29]: AddressSanitizerExampleGlobalOutOfBounds
[30]: AddressSanitizerExampleUseAfterReturn
[31]: AddressSanitizerExampleUseAfterScope
[32]: AddressSanitizerInitializationOrderFiasco
[33]: AddressSanitizerLeakSanitizer
[34]: AddressSanitizerPerformanceNumbers
[35]: AddressSanitizerAlgorithm
[36]: AddressSanitizerComparisonOfMemoryTools
[37]: AddressSanitizer
[38]: http://llvm.org
[39]: http://gcc.gnu.org
[40]: AddressSanitizerHowToBuild
[41]: AddressSanitizer
[42]: AddressSanitizerOnAndroid
[43]: AddressSanitizer
[44]: http://llvm.org/releases/3.1/tools/clang/docs/AddressSanitizer.html
[45]: AddressSanitizerCallStack
[46]: AddressSanitizerAndDebugger
[47]: https://en.wikipedia.org/wiki/Cgroups
[48]: AddressSanitizerFlags
[49]: AddressSanitizerCallStack
[50]: AddressSanitizer
[51]: AddressSanitizerIncompatiblity
[52]: AddressSanitizer
[53]: AddressSanitizer
[54]: AddressSanitizer
[55]: https://github.com/google/sanitizers/issues/247
[56]: http://www.sourceware.org/ml/libc-alpha/2016-09/msg00080.html
[57]: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58680
[58]: AddressSanitizerAsDso
[59]: http://events.linuxfoundation.org/sites/events/files/slides/Alexander_Popov-KASan_in_a_Bare-Me
tal_Hypervisor_0.pdf
[60]: https://groups.google.com/forum/#!forum/address-sanitizer
[61]: https://github.com/google/sanitizers/wiki/AddressSanitizerFlags
[62]: https://github.com/google/sanitizers/issues/611
[63]: https://github.com/google/sanitizers/wiki/AddressSanitizerFlags#run-time-flags
[64]: http://llvm.org/devmtg/2011-11/
[65]: http://www.youtube.com/watch?v=CPnRS1nv3_s
[66]: http://llvm.org/devmtg/2011-11/Serebryany_FindingRacesMemoryErrors.pdf
[67]: http://research.google.com/pubs/pub37752.html
[68]: mailto:address-sanitizer@googlegroups.com
[69]: https://plus.google.com/117014197169958493500
[70]: /google/sanitizers/wiki
[71]: 
[72]: /google/sanitizers/wiki/AddressSanitizer
[73]: /google/sanitizers/wiki/AddressSanitizer#introduction
[74]: /google/sanitizers/wiki/AddressSanitizer#getting-addresssanitizer
[75]: /google/sanitizers/wiki/AddressSanitizer#using-addresssanitizer
[76]: /google/sanitizers/wiki/AddressSanitizer#interaction-with-other-tools
[77]: /google/sanitizers/wiki/AddressSanitizer#gdb
[78]: /google/sanitizers/wiki/AddressSanitizer#ulimit--v
[79]: /google/sanitizers/wiki/AddressSanitizer#flags
[80]: /google/sanitizers/wiki/AddressSanitizer#call-stack
[81]: /google/sanitizers/wiki/AddressSanitizer#incompatibility
[82]: /google/sanitizers/wiki/AddressSanitizer#turning-off-instrumentation
[83]: /google/sanitizers/wiki/AddressSanitizer#faq
[84]: /google/sanitizers/wiki/AddressSanitizer#talks-and-papers
[85]: /google/sanitizers/wiki/AddressSanitizer#comments
[86]: /google/sanitizers/wiki/AddressSanitizerAlgorithm
[87]: 
[88]: /google/sanitizers/wiki/AddressSanitizerAndDebugger
[89]: 
[90]: /google/sanitizers/wiki/AddressSanitizerAndroidPlatform
[91]: 
[92]: /google/sanitizers/wiki/AddressSanitizerAsDso
[93]: 
[94]: /google/sanitizers/wiki/AddressSanitizerBasicBlockTracing
[95]: 
[96]: /google/sanitizers/wiki/AddressSanitizerCallStack
[97]: 
[98]: /google/sanitizers/wiki/AddressSanitizerClangVsGCC-(3.8-vs-6.0)
[99]: 
[100]: /google/sanitizers/wiki/AddressSanitizerClangVsGCC-(5.0-vs-7.1)
[101]: 
[102]: /google/sanitizers/wiki/AddressSanitizerClangVsGCC-(6.0-vs-8.1)
[103]: 
[104]: /google/sanitizers/wiki/AddressSanitizerComparisonOfMemoryTools
[105]: 
[106]: /google/sanitizers/wiki/AddressSanitizerCompileTimeOptimizations
[107]: 
[108]: /google/sanitizers/wiki/AddressSanitizerContainerOverflow
[109]: 
[110]: /google/sanitizers/wiki/AddressSanitizerExampleGlobalOutOfBounds
[111]: 
[112]: /google/sanitizers/wiki/AddressSanitizerExampleHeapOutOfBounds
[113]: 
[114]: /google/sanitizers/wiki/AddressSanitizerExampleInitOrderFiasco
[115]: 
[116]: /google/sanitizers/wiki/AddressSanitizerExampleStackOutOfBounds
[117]: 
[118]: /google/sanitizers/wiki/AddressSanitizerExampleUseAfterFree
[119]: 
[120]: /google/sanitizers/wiki/AddressSanitizerExampleUseAfterReturn
[121]: 
[122]: /google/sanitizers/wiki/AddressSanitizerExampleUseAfterScope
[123]: 
[124]: /google/sanitizers/wiki/AddressSanitizerFlags
[125]: 
[126]: /google/sanitizers/wiki/AddressSanitizerForKernel
[127]: 
[128]: /google/sanitizers/wiki/AddressSanitizerFoundBugs
[129]: 
[130]: /google/sanitizers/wiki/AddressSanitizerHowToBuild
[131]: 
[132]: /google/sanitizers/wiki/AddressSanitizerHowToContribute
[133]: 
[134]: /google/sanitizers/wiki/AddressSanitizerIncompatiblity
[135]: 
[136]: /google/sanitizers/wiki/AddressSanitizerInHardware
[137]: 
[138]: /google/sanitizers/wiki/AddressSanitizerInitializationOrderFiasco
[139]: 
[140]: /google/sanitizers/wiki/AddressSanitizerIntelMemoryProtectionExtensions
[141]: 
[142]: /google/sanitizers/wiki/AddressSanitizerIntraObjectOverflow
[143]: 
[144]: /google/sanitizers/wiki/AddressSanitizerLeakSanitizer
[145]: 
[146]: /google/sanitizers/wiki/AddressSanitizerLeakSanitizerDesignDocument
[147]: 
[148]: /google/sanitizers/wiki/AddressSanitizerLeakSanitizerVsHeapChecker
[149]: 
[150]: /google/sanitizers/wiki/AddressSanitizerLogo
[151]: 
[152]: /google/sanitizers/wiki/AddressSanitizerManualPoisoning
[153]: 
[154]: /google/sanitizers/wiki/AddressSanitizerOnAndroid
[155]: 
[156]: /google/sanitizers/wiki/AddressSanitizerOnAndroidO
[157]: 
[158]: /google/sanitizers/wiki/AddressSanitizerOneDefinitionRuleViolation
[159]: 
[160]: /google/sanitizers/wiki/AddressSanitizerPerformanceNumbers
[161]: 
[162]: /google/sanitizers/wiki/AddressSanitizerRunningSpecBenchmarks
[163]: 
[164]: /google/sanitizers/wiki/AddressSanitizerSupportedPlatforms
[165]: 
[166]: /google/sanitizers/wiki/AddressSanitizerTestSuite
[167]: 
[168]: /google/sanitizers/wiki/AddressSanitizerUseAfterReturn
[169]: 
[170]: /google/sanitizers/wiki/AddressSanitizerUseAfterScope
[171]: 
[172]: /google/sanitizers/wiki/AddressSanitizerWindowsPort
[173]: 
[174]: /google/sanitizers/wiki/AddressSanitizerZeroBasedShadow
[175]: 
[176]: /google/sanitizers/wiki/MemorySanitizer
[177]: 
[178]: /google/sanitizers/wiki/MemorySanitizerBootstrappingClang
[179]: 
[180]: /google/sanitizers/wiki/MemorySanitizerBuildingClangOnOlderSystems
[181]: 
[182]: /google/sanitizers/wiki/MemorySanitizerFAQ
[183]: 
[184]: /google/sanitizers/wiki/MemorySanitizerFoundBugs
[185]: 
[186]: /google/sanitizers/wiki/MemorySanitizerInstrumentingLibstdcxx
[187]: 
[188]: /google/sanitizers/wiki/MemorySanitizerJIT
[189]: 
[190]: /google/sanitizers/wiki/MemorySanitizerLibcxxHowTo
[191]: 
[192]: /google/sanitizers/wiki/MemorySanitizerRunning
[193]: 
[194]: /google/sanitizers/wiki/MemorySanitizerUsingGit
[195]: 
[196]: /google/sanitizers/wiki/SanitizerBotReproduceBuild
[197]: 
[198]: /google/sanitizers/wiki/SanitizerCommonFlags
[199]: 
[200]: /google/sanitizers/wiki/Stack-instrumentation-with-ARM-Memory-Tagging-Extension-(MTE)
[201]: 
[202]: /google/sanitizers/wiki/ThreadSanitizerAboutRaces
[203]: 
[204]: /google/sanitizers/wiki/ThreadSanitizerAlgorithm
[205]: 
[206]: /google/sanitizers/wiki/ThreadSanitizerAtomicOperations
[207]: 
[208]: /google/sanitizers/wiki/ThreadSanitizerBenchmarkingApache
[209]: 
[210]: /google/sanitizers/wiki/ThreadSanitizerCppManual
[211]: 
[212]: /google/sanitizers/wiki/ThreadSanitizerDeadlockDetector
[213]: 
[214]: /google/sanitizers/wiki/ThreadSanitizerDetectableBugs
[215]: 
[216]: /google/sanitizers/wiki/ThreadSanitizerDevelopment
[217]: 
[218]: /google/sanitizers/wiki/ThreadSanitizerFlags
[219]: 
[220]: /google/sanitizers/wiki/ThreadSanitizerForKernel
[221]: 
[222]: /google/sanitizers/wiki/ThreadSanitizerForKernelReports
[223]: 
[224]: /google/sanitizers/wiki/ThreadSanitizerFoundBugs
[225]: 
[226]: /google/sanitizers/wiki/ThreadSanitizerGoManual
[227]: 
[228]: /google/sanitizers/wiki/ThreadSanitizerOnAndroid
[229]: 
[230]: /google/sanitizers/wiki/ThreadSanitizerPopularDataRaces
[231]: 
[232]: /google/sanitizers/wiki/ThreadSanitizerReportFormat
[233]: 
[234]: /google/sanitizers/wiki/ThreadSanitizerSuppressions
[235]: 
[236]: /google/sanitizers/wiki/ThreadSanitizerVolatileRanges
[237]: 
