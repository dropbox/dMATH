* [1 Introduction][1]
* [2 Installation][2]
  
  * [2.1 Prerequisites][3]
  * [2.2 Building & Installing][4]
* [3 An Introduction to Using DIVINE][5]
  
  * [3.1 Basics of Program Analysis][6]
  * [3.2 Debugging Counterexamples with the Interactive Simulator][7]
  * [3.3 Controlling the Execution Environment][8]
  * [3.4 Compilation Options and Compilation of Multiple Files][9]
* [4 Commandline Interface][10]
  
  * [4.1 Synopsis][11]
  * [4.2 Input Options][12]
  * [4.3 State Space Visualisation & Simulation][13]
  * [4.4 Model Checking][14]
* [5 Model Checking C and C++ Code via LLVM Bitcode][15]
  
  * [5.1 Compiling Programs][16]
  * [5.2 Limitations][17]
  * [5.3 State Space of a Program][18]
  * [5.4 Non-Deterministic Choice][19]
  * [5.5 ω-Regular Properties and LTL][20]
  * [5.6 Symbolic verification][21]
* [6 Interactive Debugging][22]
  
  * [6.1 Tutorial][23]
  * [6.2 Collecting Information][24]
* [7 DiVM: A Virtual Machine for Verification][25]
  
  * [7.1 Activation Frames][26]
  * [7.2 Control Registers][27]
  * [7.3 Heap][28]
  * [7.4 The Hypercall Interface][29]
  * [7.5 Scheduling][30]
  * [7.6 Faults][31]
  * [7.7 Boot Sequence][32]
  * [7.8 Memory Management Hypercalls][33]
  * [7.9 Non-deterministic Choice and Counterexamples][34]
  * [7.10 Debug Mode][35]
* [8 DiOS, A Small, Verification-Oriented Operating System][36]
  
  * [8.1 DiOS Compared to Traditional Operating Systems][37]
  * [8.2 Fault Handling And Error Traces][38]
  * [8.3 DiOS Configuration][39]
  * [8.4 Virtual File System][40]

# 1 Introduction

The DIVINE project aims to develop a general-purpose, fast, reliable and easy-to-use model checker.
The roots of the project go back to a special-purpose, explicit-state, asynchronous system model
checking tool for LTL properties. However, rigorous development processes are in a steady decline,
being displaced by more agile, flexible and dynamic methods. In the agile world, there is little
place for large-scale, long-term planning and pondering on “paper only” designs, which would favour
the use of a traditional model checker.

The current version of DIVINE strives to keep up with this dynamic world, bringing “heavy-duty”
model checking technology much closer to daily programming routine. Our major goal is to express
model checking problems in a language which every developer is fluent with: the programming language
of their own project. Even if you don’t apply model checking to your resulting program directly,
writing throwaway models makes much more sense in a language you understand well and use daily.

Current versions of DIVINE provide out-of-the-box support for the C (C99) and C++ (C++14)
programming languages, including their respective standard libraries. Additional libraries may be
rebuilt for use with DIVINE by the user.

# 2 Installation

This section is only relevant when you are installing from source. We will assume that you have
obtained a source tarball from [http://divine.fi.muni.cz][41], e.g. `divine-4.4.2.tar.gz`.

DIVINE can be built on Linux and possibly on other POSIX-compatible systems including macOS (not
tested). There is currently no support for DIVINE on Windows. If you do not want to build DIVINE
from sources, you can download a virtual machine image with pre-built DIVINE.

## 2.1 Prerequisites

If you use recent Ubuntu, Fedora or Arch Linux (or possibly another distribution which uses
`apt-get`, `yum` or `pacman` as a package manager) the easiest way to get dependencies of DIVINE is
to run `make prerequisites` in the directory with the sources (you will need to have `make`
installed):

`$ tar xvzf divine-4.4.2.tar.gz
$ cd divine-4.4.2
$ make prerequisites`

Otherwise, to build DIVINE, you will need the following:

* A POSIX-compatible operating system,
* `make` (tested with BSD and GNU),
* GNU C++ (4.9 or newer) or clang (3.2 or newer),
* CMake [[www.cmake.org][42]] 3.2 or newer,
* `libedit` [[thrysoee.dk/editline][43]],
* about 12GB of disk space and 4GB of RAM (18GB for both release and debug builds),

Additionally, DIVINE can make use of the following optional components:

* ninja build system [[ninja-build.org][44]] for faster builds,
* pandoc [[pandoc.org][45]] for formatting the manual (HTML and PDF with pdflatex).

## 2.2 Building & Installing

First, unzip the distribution tarball and enter the newly created directory

`$ tar xvzf divine-4.4.2.tar.gz
$ cd divine-4.4.2`

The build is driven by a Makefile, and should be fully automatic. You only need to run:

`$ make`

This will first build a C++14 toolchain and a runtime required to build DIVINE itself, then proceed
to compile DIVINE. After a while, you should obtain the main DIVINE binary. You can check that this
is indeed the case by running:

`$ ./_build.release/tools/divine help`

You can now run DIVINE from its build directory, or you can optionally install it by issuing

`$ make install`

This will install DIVINE and its version of LLVM into `/opt/divine/`.

You can also run the test-suite if you like:

`$ make check`

# 3 An Introduction to Using DIVINE

In this section, we will give a short example on how to invoke DIVINE and its various functions. We
will use a small C program (consisting of a single compilation unit) as an example, along with a few
simple properties.

## 3.1 Basics of Program Analysis

We will start with the following simple C program:

`#include <stdio.h>
#include <assert.h>

void init( int *array )
{
    for ( int i = 0; i <= 4; ++i )
    {
        printf( "writing at index %d\n", i );
        array[i] = i;
    }
}

int main()
{
    int x[4];
    init( x );
    assert( x[3] == 3 );
}`

The above code contains a bug: an out of bounds access to `array` at index `i == 4`; we will see how
this is presented by DIVINE.

The program can be compiled by your system’s C compiler and executed. If you do so, it will probably
run OK despite the out-of-bounds access (this is an example of a stack buffer overflow –- the
program will incorrectly overwrite an adjacent value on the stack which, in most cases, does not
interfere with its execution). We can proceed to check the program using DIVINE:

`$ divine check program.c`

DIVINE will compile your program and run the verifier on the compiled code. After a short while, it
will produce the following output:

`compiling program.c
loading bitcode … LART … RR … constants … done
booting … done
states per second: 419.192
state count: 83

error found: yes
error trace: |
  [0] writing at index 0
  [0] writing at index 1
  [0] writing at index 2
  [0] writing at index 3
  [0] writing at index 4
  FAULT: access of size 4 at [heap* 295cbfc3 10h ddp] is 4 bytes out of bounds
  [0] FATAL: memory error in userspace

active stack:
  - symbol: void {Fault}::handler<{Context} >(_VM_Fault, _VM_Frame*, void (*)(), ...)
    location: /divine/include/dios/core/fault.hpp:189
  - symbol: init
    location: program.c:9
  - symbol: main
    location: program.c:16
  - symbol: _start
    location: /divine/src/libc/functions/sys/start.cpp:77
a report was written to program.report.mtjmlg`

The output begins with compile- and load-time report messages, followed by some statistics. Starting
with `error found: yes`, the detected error is introduced. The error-related information contains:

* `error trace` – shows the output that the program printed during its execution, until the point of
  the error; a description of the error concludes this section,
* `active stack` – this field contains the stack trace of the thread responsible for the error (with
  a fault handler at the top)

In our case, the most important information is `FAULT: access of size 4 at [heap* 295cbfc3 10h ddp]
is 4 bytes out of bounds` which indicates the error was caused by an invalid memory access. The
other crucial information is the line of code which caused the error:

`  - symbol: init
    location: program.c:9`

Hence we can see that the problem is caused by an invalid memory access on line 9 in our program.

*Note:* one might notice that the addresses in DIVINE are printed in the form `[heap* 295cbfc3 10h
ddp]`; the meaning of which is: the pointer in question is a heap pointer (other types of pointers
are constant pointers and global pointers; stack pointers are not distinguished from heap pointers);
the object identifier (in hexadecimal, assigned by the VM) is `295cbfc3`; the offset (again in
hexadecimal) is `10` and the value is a defined pointer (`ddp`, i.e. it is not an uninitialised
value).

## 3.2 Debugging Counterexamples with the Interactive Simulator

Now that we have found a bug in our program, it might be useful to inspect the error in more detail.
For this, we can use DIVINE’s simulator.

`$ divine sim program.c`

After compilation, we will land in an interactive debugger environment:

`Welcome to 'divine sim', an interactive debugger. Type 'help' to get started.
# executing __boot at /divine/src/dios/core/dios.cpp:315
>`

There are a few commands we could use in this situation. For instance, the `start` command brings
the program to the beginning of the `main` function (fast-forwarding through the internal program
initialisation process). Another alternative is to invoke `sim` with `--load-report` (where the name
of the report is printed at the very end of the `check` output), like this:

`$ divine sim --load-report program.report.mtjmlg`

The simulator now prints identifiers of program states along the violating execution, together with
the output of the program (prefixed with `T:`). The replay stops at the error that was found by
`divine check`.

One can use the `up` and `down` commands to move through the active stack, to examine the context of
the error. To examine local variables, we can use the `show`, including the value of `i`:

`.i$1:
    type:    int
    value:   [i32 4 d]`

The entry suggests that `i` is an `int` variable and its value is represented as `[i32 4 d]`:
meaning it is a 32 bit integer with value 4 and it is fully defined (`d`). If we go one frame `up`
and use `show` again, we can see the entry for `x`:

`.x:
    type:    int[]
    .[0]:
        type: int
        value: [i32 42 d]
    .[1]:
        type: int
        value: [i32 42 d]
    .[2]:
        type: int
        value: [i32 42 d]
    .[3]:
        type: int
        value: [i32 42 d]`

We see that `x` is an `int` array and that it contains 4 values: the access at `x[4]` is clearly out
of bounds.

More details about the simulator can be found in the section on [interactive debugging][46].

## 3.3 Controlling the Execution Environment

Programs in DIVINE run in an environment provided by [DiOS][47], DIVINE’s operating system, and by
runtime libraries (including C and C++ standard libraries and `pthreads`). The behaviour of this
runtime can be configured using the `-o` option. To get the list of options, run `divine info
program.c`:

`$ divine info program.c
compiling program.c

DIVINE 4.0.22

Available options for test/c/2.trivial.c are:
- [force-]{ignore|report|abort}: configure the fault handler
  arguments:
   - assert
   - arithmetic
   - memory
[...]
- config: run DiOS in a given configuration
  arguments:
   - default: async threads, processes, vfs
   - passthrough: pass syscalls to the host OS
   - replay: re-use a trace recorded in passthrough mode
   - synchronous: for use with synchronous systems
[...]
use -o {option}:{value} to pass these options to the program`

It is often convenient to assume that malloc never fails: this can be achieved by passing the `-o
nofail:malloc` option to DiOS. Other important options are those controlling the fatalness of errors
(the default option is `abort` – if an error of type `abort` is encountered, the verification ends
with an error report; on the other hand, the verifier will attempt to continue when it detects an
error that was marked as `ignore`).

Furthermore, it is possible to pass arguments to the `main` function of the program by appending
them after the name of the source file (e.g. `divine verify program.c main-arg-1 main-arg-2`), and
to add environment variables for the program using the `-DVAR=VALUE` option.

## 3.4 Compilation Options and Compilation of Multiple Files

Supposing you wish to verify something that is not just a simple C program, you may have to pass
compilation options to DIVINE. In some cases, it is sufficient to pass the following options to
`divine verify`:

* `-std=<version>`, (e.g. `-std=c++14`) option sets the standard of C/C++ to be used and is directly
  forwarded to the compiler;
* other options can be passed using the `-C` option, i.e. `-C,-O3` to enable optimisation, or
  `-C,-I,include-path` to add `include-path` to the directories in which compiler looks for header
  files.

However, if you need to pass more options or if your program consists of more than one source file,
it might be more practical to compile it to LLVM bitcode first and pass this bitcode to `divine
verify`:

`$ divine cc -Iinclude-path -O1 -std=c++14 -DFOO=bar program.c other.c
$ divine verify program.bc`

`divine cc` is a wrapper for the clang compiler and it is possible to pass most of clang’s options
to it directly.

# 4 Commandline Interface

The main interface to DIVINE is the command-line tool simply called `divine`. Basic information
about the binary can be obtained by issuing `divine --version`:

`version: 4.0.12+8fbbb7005640
source sha: 37282b999cc10a027cabbb9669365f9e630fb685
runtime sha: b8939c99ca81161cbe69076e31073807f69dda72
build date: 2017-09-19, 14:02 UTC
build type: Debug`

## 4.1 Synopsis

`divine cc [compiler flags] <sources>
divine verify [...] <input file>
divine draw [...] <input file>
divine run [...] <input file>
divine sim [...] <input file>`

## 4.2 Input Options

All commands that work with inputs share a number of flags that influence the input program.

`divine {...} [-D {string}]
             [--autotrace {tracepoint}]
             [--sequential]
             [--disable-static-reduction]
             [--relaxed-memory {string}]
             [--lart {string}]`

## 4.3 State Space Visualisation & Simulation

`divine draw [--distance {int}]
            [--render {string}]
            {input options}`

To visualise (a part of) the state space, you can use `divine draw`, which creates a
graphviz-compatible representation. By default, it will run “`dot -Tx11`” to display the resulting
graph. You can override the drawing command to run by using the `--render` switch. The command will
get the dot-formatted graph description on its standard input. Out of common input options, the
`--autotrace` option is often quite useful with `draw`.

`divine sim [--batch] [--load-report {file}] [--skip-init]
           {input options}`

The `sim` sub-command is used to interactively explore a program using a terminal-based interface.
Use `help` in the interactive prompt to obtain help on available commands. See also [Interactive
Debugging][48].

## 4.4 Model Checking

`divine {check|verify} [model checking options]
                      [exploration options]
                      {input options}`

These two commands are the main workhorse of model checking. The `verify` command performs full
model checking under conservative assumptions. The `check` command is more liberal, and will assume,
for instance, that `malloc` will not fail. While `check` is likely to cover most scenarios, it omits
cases that are either very expensive to check or that appear in programs very often and make the
tool cumbersome to use.

The algorithms DIVINE uses are often resource-intensive and some are parallel (multi-threaded). The
`verify` and `check` commands will, in common cases, use a parallel algorithm to explore the state
space of the system. By default, parallel algorithms will use available cores, 4 at most. A few
switches control resource use:

`divine {...} [--threads {int}]
             [--max-memory {mem}]
             [--max-time {int}]`

*`--threads {int} | -T {int}`*
  The number of threads to use for verification. The default is 4 or the number of cores if less
  than 4. For optimal performance, each thread should get one otherwise mostly idle CPU core. Your
  mileage may vary with hyper-threading (it is best to run a few benchmarks on your system to find
  the best configuration).
*`--max-memory {mem}`*
  Limit the amount of memory `divine` is allowed to allocate. This is mainly useful to limit
  swapping. When the verification exceeds available RAM, it will usually take extremely long time to
  finish and put a lot of strain on the IO subsystem. It is recommended that you do not allow
  `divine` to swap excessively, either using this option or by some other means.
*`--max-time {int}`*
  Put a limit of `{int}` seconds on the maximal running time.

Verification results can be written in a few forms, and resource use can also be logged for
benchmarking purposes:

`divine {...} [--report {fmt}]
             [--no-report-file]
             [--report-filename {string}]
             [--num-callers {int}]`

*`--report {fmt}`*
  At the end of a verification run, produce a comprehensive, machine-readable report. Currently the
  available formats are: `none` – disables the report entirely, `yaml` – prints a concise,
  yaml-fomatted summary of the results (without memory statistics or machine-readable counterexample
  data) and `yaml-long` which prints everything as yaml.
*`--no-report-file`*
  By default, the long-form verification results (equivalent to `--report yaml-long`) are stored in
  a file. This switch suppresses that behaviour.
*`--report-filename {string}`*
  Store the long-form verification results in a file with the given name. If this option is not
  used, a unique filename is derived from the name of the input file.

# 5 Model Checking C and C++ Code via LLVM Bitcode

The traditional “explicit-state” model checking practice is not widely adopted in the programming
community, and vice-versa, use of mainstream programming languages is not a common practice in the
model checking community. Hence, model checking of systems expressed as C programs comes with some
novelties for both kinds of users.

First of all, the current main application of DIVINE is verification of safety (and some liveness)
properties of asynchronous, shared-memory programs. The typical realisation of such asynchronous
parallelism is programming with threads and shared memory. Often for performance and/or familiarity
reasons, programming with threads, shared memory and locking is the only viable alternative, even
though the approach is fraught with difficulties and presents many pitfalls that can catch even
expert programmers unaware – not to say novices. Sadly, resource locking is inherently
non-compositional, hence there is virtually no way to provide a reliable yet powerful abstraction,
all that while retaining speed and scalability.

Despite all its shortcomings, lock-based programming (or alternatively, lock-free shared memory
programming, which is yet more difficult) is becoming more prevalent. Model checking provides a
powerful tool to ascertain correctness of programs written with locks around concurrent access to
shared memory. Most programmers will agree that bugs that show up rarely and are nearly impossible
to reproduce are the worst kind to deal with. Sadly, most concurrency bugs are of this nature, since
they arise from subtle interactions of nondeterministically scheduled threads of execution. A
test-case may work 99 times out of 100, yet the 100th time die due to an unfathomable invalid memory
access. Even sophisticated modern debugging tools like `valgrind` are often powerless in this
situation.

This is where DIVINE can help, since it systematically and efficiently explores all relevant
execution orderings, discovering even the subtlest race conditions, restoring a crucially important
property of bugs: reproducibility. If you have ever changed a program and watched the test-case run
in a loop for hundreds of iterations, wondering if the bug is really fixed, or it just stubbornly
refuses to crash the program… well, DIVINE is the tool you have been waiting for.

Of course, there is a catch. Model checking is computationally intensive and memory-hungry. While
this is usually not a problem with comparably small unit tests, applying model checking to large
programs may not be feasible – depending on your use-case, and on the amount of memory and time that
you have.

With the universal LLVM backend, DIVINE can support a wide range of compiled programming languages.
However, out of the box, language-specific support is only provided for C and C++. A fairly complete
ISO C runtime library is provided as part of DIVINE, with appropriate hooks into DIVINE, as well as
an implementation of the `pthread` specification, i.e. the POSIX threading interface. Additionally,
an implementation of the standard C++ library is bundled with DIVINE. Besides the standard library,
DIVINE also provides an adapted version of the *runtime* library required by C++ programs to
implement runtime type identification and exception handling; both are fully supported by DIVINE.

If your language interfaces with the C library, the `libc` part of language support can be re-used
transparently. However, currently no other platform glue is provided for other languages. Your
mileage may vary. Data structure and algorithmic code can be very likely processed with at most
trivial additions to the support code, in any language that can be compiled into LLVM bitcode.

## 5.1 Compiling Programs

The first step to take when you want to use DIVINE for C/C++ verification is to compile your program
into LLVM bitcode and link it against the DIVINE-provided runtime libraries. When you issue `divine
cc program.c`, `divine` will compile the runtime support and your program and link them together
automatically, using a builtin clang-based compiler. The `cc` subcommand accepts a wide array of
traditional C compiler flags like `-I`, `-std`, `-W` and so on.

Alternatively, you can pass a single-file C or C++ program directly to other `divine` commands, like
`verify` or `sim`, in which case the program will be compiled and linked transparently.

## 5.2 Limitations

When DIVINE interprets your program, it does so in a very strictly controlled environment, so that
every step is fully reproducible. Hence, no real IO is allowed; the program can only see its own
memory.

While the `pthread` interface is fully supported, the `fork` system call is not: DIVINE cannot
verify multi-process systems at this time. Likewise, C++ exceptions are well-supported but the C
functions `setjmp` and `longjmp` are not yet implemented.

The interpreter simulates an “in order” CPU architecture, hence any possible detrimental effects of
instruction and memory access reordering won’t be detected. Again, an improvement in this area is
planned for a future release.

## 5.3 State Space of a Program

Internally, DIVINE constructs the state space of your program – an oriented graph, whose vertices
represent various states your program can reach, i.e. the values of all its mapped memory locations,
the values in all machine registers, and so on. DIVINE normally doesn’t store (or construct) all the
possible states – only a relevant subset. In a multi-threaded program, it often happens that more
than one thread can run at once, i.e. if your program is in a given state, the next state it will
get into is determined by chance – it all hinges on which thread gets to run. Hence, some (in fact,
most) states in a parallel program can proceed to multiple different configurations, depending on a
scheduling choice. In such cases, DIVINE has to explore both such “variant” successors to determine
the behaviour of the program in all possible scenarios.

You can visualise the state space DIVINE explores by using `divine draw` – this will show how the
“future” of your program branches through various configurations, and how it converges back into a
common point – or, if its behaviour had changed depending on scheduling, diverges into multiple
different outcomes.

## 5.4 Non-Deterministic Choice

Scheduling is not the only source of non-determinism in typical programs. Interactions with the
environment can have different outcomes, even if the internal state of the program is identical. A
typical example would be memory allocation: the same call in the same context could either succeed
or fail, depending on the conditions outside the control of the program. In addition to failures,
the normal input of the program (files, network, user input on the terminal) are all instances of
non-determinism due to external influences. The sum of all such external behaviours that affect the
outcome of the program is called the environment (this includes the scheduling of threads done by
the operating system and/or CPU). When testing, the environment is controlled to some degree: the
inputs are usually fixed, but eg. thread scheduling is (basically) random – it changes with every
test execution. Resource exhaustion can be simulated in a test environment to some degree. In a
model checker, the environment is controlled much more strictly: when a test case is run through a
model checker, it will come out the same every time.

## 5.5 ω-Regular Properties and LTL

## 5.6 Symbolic verification

In DIVINE, we support verification of programs with inputs, using a symbolic representation of data.
You can denote a symbolic variable with an annotation when you declare it in the code:

`_SYM int x;`

To do so, you need to include a header file `abstract/domains.h`. The annotated value does not need
to be initialized since it is inherently considered to have an arbitrary value of given type. A
verification of program with symbolic variables requires a special exploration algorithm. You can
turn it on with `--symbolic` option when you verify a program:

`$ divine check --symbolic program.cpp`

Besides annotations, DIVINE supports [SV-COMP][49] intrinsics as defined in competition rules.
Implementation of intrinsics using symbolic domain is compiled and linked with a verified program
and can be directly used with verified program without any includes. DIVINE provides following
intrinsics:

* `__VERIFIER_nondet_X()` to model nondeterministic values of type`X` (`bool, char, int, uint,
  short, ushort, long, ulong`),
* `__VERIFIER_assume(int expression)` restricts a program behaviour according to the expression,
* `__VERIFIER_assert(int condition)`,
* `__VERIFIER_error()`,
* `__VERIFIER_atomic_begin()`,
* `__VERIFIER_atomic_end()`.

The implementation of the intrinsics can be found in the file `runtime/abstract/svcomp.cpp`.

# 6 Interactive Debugging

DIVINE comes with an interactive debugger, available as `divine sim`. The debugger loads programs
the same way `verify` and other DIVINE commands do, so you can run it on standalone C or C++ source
files directly, or you can compile larger programs into bitcode and load that.

## 6.1 Tutorial

Let’s say you have a small C program which you wish to debug. We will refer to this program as
`program.c`. To load the program into the debugger, simply execute

`$ divine sim program.c`

and DIVINE will take care of compiling and linking your program. It will load the resulting bitcode
but will not execute it immediately: instead, `sim` will present a prompt to you, looking like this:

`# executing __boot at /divine/src/dios/dios.cpp:79
>`

The `__boot` function is common to all DIVINE-compiled programs and belongs to DiOS, our minimalist
operating system. When debugging user-mode programs, the good first command to run is

`> start`

which will start executing the program until it enters the `main` function:

`# a new program state was stored as #1
# active threads: [0:0]
# a new program state was stored as #2
# active threads: [0:0]
# executing main at program.c:14
>`

We can already see a few DIVINE-specific features of the debugger. First, program states are stored
and retained for future reference. Second, thread switching is quite explicit – every time a
scheduling decision is made, `sim` informs you of this fact. We will look at how to influence these
decisions later.

Now is a good time to familiarise ourselves with how to inspect the program. There are two commands
for listing the program itself: `source` and `bitcode` and each will print the currently executing
function in the appropriate form (the original C source code and LLVM bitcode, respectively).
Additionally, when printing bitcode, the current values of all LLVM registers are shown as inline
comments:

`  label %entry:
>>  %01 = alloca [i32 1 d]                                  # [global* 0 0 uun]
    dbg.declare 
    %03 = getelementptr %01 [i32 0 d] [i32 0 d]             # [global* 0 0 uun]
    call @init %03 `

Besides printing the currently executing function, both `source` and `bitcode` can print code
corresponding to other functions; in fact, by default they print whatever function the debugger
variable `$frame` refers to. To print the source code of the current function’s caller, you can
issue

`> source caller
     97 void _start( int l, int argc, char **argv, char **envp ) {
>>   98     int res = __execute_main( l, argc, argv, envp );
     99     exit( res );
    100 }`

To inspect data, we can instead use `show` and `inspect`. We have mentioned `$frame` earlier: there
is, in fact, a number of variables set by `sim`. The most interesting of those is `$_` which
normally refers to the most interesting object at hand, typically the executing frame. By default,
`show` will print the content of `$_` (like many other commands). However, when we pass an explicit
parameter to these commands, the difference between `show` and `inspect` becomes apparent: the
latter also sets `$_` to its parameter. This makes `inspect` more suitable for exploring more
complex data, while `show` is useful for quickly glancing at nearby values:

`> step --count 2
> show
attributes:
    address: heap* 9cd25662 0+0
    shared:  0
    pc:      code* 2 2
    insn:    dbg.declare 
    location: program.c:16
    symbol:  main
.x:
    type:    int[]
    .[0]:
        type: int
        value: [i32 0 u]
    .[1]:
        type: int
        value: [i32 0 u]
    .[2]:
        type: int
        value: [i32 0 u]
    .[3]:
        type: int
        value: [i32 0 u]
related:     [ caller ]`

This is how a frame is presented when we look at it with `show`.

## 6.2 Collecting Information

Apart from `show` and `inspect` which simply print structured program data to the screen, there are
additional commands for data extraction. First, `backtrace` will print the entire stack trace in one
go, by default starting with the currently executing frame. It is also possible to obtain *all*
stack traces reachable from a given heap variable, e.g.

`> backtrace $state
# backtrace 1:
  __dios::_InterruptMask<true>::Without::Without(__dios::_InterruptMask<true>&)
      at /divine/include/libc/include/sys/interrupt.h:42
  _pthread_join(__dios::_InterruptMask<true>&, _DiOS_TLS*, void**)
      at /divine/include/libc/include/sys/interrupt.h:77
  pthread_join at /divine/src/libc/functions/sys/pthread.cpp:539
  main at test/pthread/2.mutex-good.c:22
  _start at /divine/src/libc/functions/sys/start.cpp:76
# backtrace 2:
  __pthread_entry at /divine/src/libc/functions/sys/pthread.cpp:447`

Another command to gather data is `call`, which allows you to call a function defined in the
program. The function must not take any parameters and will be executed in *debug mode* (see Section
[7.10][50] – the important caveat is that any `dbg.call` instructions in your info function will be
ignored). Execution of the function will have no effect on the state of the simulated program. If
you have a program like this:

`#include <sys/divm.h>

void print_hello()
{
    __vm_trace( _VM_T_Text, "hello world" );
}

int main() {}`

The `call` command works like this:

`> call print_hello
  hello world`

Finally, the `info` command serves as a universal information gathering alias – you can set up your
own commands that then become available as `info` sub-commands:

`> info --setup "call print_hello" hello
> info hello
  hello world`

The `info` command also provides a built-in sub-command `registers` which prints the current values
of machine control registers (see also Section [7.2][51]):

`> info registers
Constants:    220000000
Globals:      120000000
Frame:        9cd2566220000000
PC:           340000000
Scheduler:    eb40000000
State:        4d7b876d20000000
IntFrame:     1020000000
Flags:        0
FaultHandler: b940000000
ObjIdShuffle: faa6693f
User1:        0
User2:        201879b120000000
User3:        6ca5bc2260000000
User4:        0`

# 7 DiVM: A Virtual Machine for Verification

Programs in DIVINE are executed by a virtual machine (called DiVM). The machine code executed by
this virtual machine is an extension of the LLVM bitcode. The LLVM part of this “machine language”
is described in detail in the [LLVM Documentation][52]. The extensions of the language and the
semantics specific to DiVM are the subject of this chapter.

## 7.1 Activation Frames

Unlike in a ‘traditional’ implementations of C, there is no continuous stack in DiVM. Instead, each
activation record (frame) is allocated from the heap and its size is fixed. Allocations that are
normally done at runtime from the stack are instead done from the heap, using the `alloca` LLVM
instruction. Additionally, since LLVM bitcode is in partial SSA form, what LLVM calls ‘registers’
are objects quite different from traditional machine registers. The registers used by a given
function are bound to the frame of that function (they cannot be used to pass values around and they
don’t need to be explicitly saved across calls). In the VM, this is realized by storing registers
(statically allocated) in the activation record itself, along with DiVM-specific “program counter”
register (this is an actual register, but is saved across calls automatically by the VM, see also
[Control Registers][53] below) and a pointer to the caller’s activation frame. The header of the
activation record has a C-compatible representation, available as `struct _VM_Frame` in
`sys/divm.h`.

## 7.2 Control Registers

The state of the VM consists of two parts, a set of *control registers* and the *heap* (structured
memory). All available control registers are described by `enum _VM_ControlRegister` defined in
`sys/divm.h` and can be manipulated through the `__vm_control` hypercall (see also Section
**¿sec:hypercalls?**] below). Please note that control registers and LLVM registers (SSA values) are
two different things. The control registers are of two types, holding either an integer or a
pointer. There only integer register, `_VM_CR_Flags`, is used as a bitfield.

Four control registers govern address translation and normal execution:

* `_VM_CR_Constants` contains the base address of the heap object (see [Heap][54] below) used by the
  VM to hold constants
* `_VM_CR_Globals` is the base address of the heap object where global variables are stored
* `_VM_CR_Frame` points to the currently executing activation frame
* `_VM_CR_PC` is the program counter

Additional 4 registers are concerned with scheduling and interrupt control (for details, see Section
[7.5][55] below):

* `_VM_CR_Scheduler` is the entry address of the scheduler
* `_VM_CR_State` is the object holding persistent state of the scheduler
* `_VM_CR_IntFrame` the address of the interrupted frame (see also Section [7.5][56])
* `_VM_CR_Flags` is described in more detail below

Finally, there’s `_VM_CR_FaultHandler` (the address of the fault handler, see Section [7.6][57]) and
4 user registers (`_VM_CR_User1` through `_VM_CR_User4`) of unspecified types: they can hold either
a 64 bit integer or a pointer. The VM itself never looks at the content of those registers.

The flags register (`_VM_CR_Flags`) is further broken down into individual bits, described by `enum
_VM_ControlFlags`, again defined in `sys/divm.h`. These are:

* `_VM_CF_Mask`, if set, blocks *all* interrupts
* `_VM_CF_Interrupted`, if set, causes an immediate interrupt (unless `_VM_CF_Mask` is also set, in
  which case the interrupt happens as soon as `_VM_CF_Mask` is lifted).
* `_VM_CF_KernelMode` which indicates whether the VM is running in user or kernel mode; this bit is
  set by the VM when `__boot` or the scheduler is entered and whenever an interrupt happens; the bit
  can be cleared (but not set) via `__vm_control`

The remaining 3 flags indicate properties of the resulting edge in the state space (see also [State
Space of a Program][58]). These may be set by the program itself or by a special monitor automaton,
a feature of DiOS which enables modular specification of non-trivial (sequence-dependent)
properties. These 3 flags are reset by the VM upon entering the scheduler (see Section [7.5][59]).
The edge-specific flags are:

* `_VM_CF_Error` indicates that an error – a safety violation – ought to be reported (a good place
  to set this is the *fault handler*, see Section [7.6][60]),
* `_VM_CF_Accepting` indicates that the edge is accepting, under a Büchi acceptance condition (see
  also [ω-Regular Properties and LTL][61]).
* `_VM_CF_Cancel` indicates that this edge should be abandoned (it will not become a part of the
  state space and neither will its target state, unless also reachable some other way)

## 7.3 Heap

The entire *persistent* state of the VM is stored in the heap. The heap is represented as a directed
graph of objects, where pointers stored in those objects act as the edges of the graph. For each
object, in addition to the memory corresponding to that object, a supplemental information area is
allocated transparently by the VM for tracking metadata, like which bytes in the object are
initialised (defined) and a list of addresses in the object where pointers are stored.

Activation frames, global variables and even constants are all stored in the heap. The heap is also
stored in a way that makes it quite efficient (both time- and memory-wise) for the VM to take
snapshots and store them. This is how model checking and reversible debugging is realized in DIVINE.

## 7.4 The Hypercall Interface

The interface between the program and the VM is based on a small set of *hypercalls* (a list is
provided in tbl. [1][62]). This way, unlike pure LLVM, the DiVM language is capable of encoding an
operating system, along with a syscall interface and all the usual functionality included in system
libraries.

─────────────┬────────────────────────────────────────────────────────
Hypercall    │Description                                             
─────────────┼────────────────────────────────────────────────────────
`obj_make`   │Create a new object in the memory graph of the program  
─────────────┼────────────────────────────────────────────────────────
`obj_free`   │Explicitly destroys an object in the memory graph       
─────────────┼────────────────────────────────────────────────────────
`obj_size`   │Obtain the current size of an object                    
─────────────┼────────────────────────────────────────────────────────
`obj_resize` │Efficiently resize an object (optional)                 
─────────────┼────────────────────────────────────────────────────────
`obj_shared` │Mark an object as *shared* for τ reduction (optional)   
─────────────┼────────────────────────────────────────────────────────
`trace`      │Attach a piece of data to an edge in the execution graph
─────────────┼────────────────────────────────────────────────────────
`interrupt_me│Mark a memory-access-related interrupt point            
m`           │                                                        
─────────────┼────────────────────────────────────────────────────────
`interrupt_cf│Mark a control-flow-related interrupt point             
l`           │                                                        
─────────────┼────────────────────────────────────────────────────────
`choose`     │Non-deterministic choice (a fork in the execution graph)
─────────────┼────────────────────────────────────────────────────────
`control`    │Read or manipulate machine control registers            
─────────────┴────────────────────────────────────────────────────────

## 7.5 Scheduling

The DIVINE VM has no intrinsic concept of threads or processes. Instead, it relies on an “operating
system” to implement such abstractions and the VM itself only provides the minimum support
necessary. Unlike with “real” computers, a system required to operate DiVM can be extremely simple,
consisting of just 2 C functions (one of them is `__boot`, see [Boot Sequence][63] below). The
latter of those is the scheduler, the responsibility of which is to organize interleaving of threads
in the program to be verified. However, the program may not use threads but some other form of
concurrency – it is up to the scheduler, which may be provided by the user, to implement the correct
abstractions.

From the point of view of the state space (cf. [State Space of a Program][64]), the scheduler
decides what the successors of a given state are. When DIVINE needs to construct successors to a
particular state, it executes the scheduler in that state; the scheduler decides which thread to run
(usually with the help of the non-deterministic choice operator) and transfers control to that
thread (by changing the value of the `_VM_CR_Frame` control register, i.e. by instructing DIVINE to
execute a particular activation frame). The VM then continues execution in the activation frame that
the scheduler has chosen, until it encounters an *interrupt*. When DIVINE loads a program, it
annotates the bitcode with *interrupt points*, that is, locations in the program where threads may
need to be re-scheduled. When such a point is encountered, the VM sets the `_VM_CF_Interrupted` bit
in `_VM_CR_Flags` and unless `_VM_CF_Mask` is in effect, an interrupt is raised immediately.

Upon an interrupt, the values of `_VM_CR_IntFrame` and `_VM_CR_Frame` are swapped, which usually
means that the control is transferred back to the scheduler, which can then read the address of the
interrupted frame from `_VM_CR_IntFrame` (this may be a descendant or a parent of the frame that the
scheduler originally transferred control to, or may be a null pointer if the activation stack became
empty).

Of course, the scheduler needs to store its state – for this purpose, it must use the `_VM_CR_State`
register, which is set by `__boot` to point to a particular heap object. This heap object can be
resized by calling `__vm_obj_resize` if needed, but the register itself is read-only after `__boot`
returns. The object can be used to, for example, store pointers to activation frames corresponding
to individual threads (but of course, those may also be stored indirectly, behind a pointer to
another heap object). In other words, the object pointed to by `_VM_CR_State` serves as the *root*
of the heap.

## 7.6 Faults

An important role of DIVINE is to detect errors – various types of safety violations – in the
program. For this reason, it needs to interpret the bitcode as strictly as possible and report any
problems back to the user. Specifically, any dangerous operations that would normally lead to a
crash (or worse, a security vulnerability) are caught and reported as *faults* by the VM. The fault
types that can arise are the following (enumerated in `enum _VM_Fault` in `divine.h`):

* `_VM_F_Arithmetic` is raised when the program attempts to divide by zero
* `_VM_F_Memory` is raised on attempts at illegal memory access and related errors (out-of-bounds
  loads or writes, double free, attempts to dereference undefined pointers)
* `_VM_F_Control` is raised on control flow errors – undefined conditional jumps, invalid call of a
  null or invalid function pointer, wrong number of arguments in a `call` instruction, `select` or
  `switch` on an undefined value or attempt to execute the `unreachable` LLVM instruction
* `_VM_F_Hypercall` is raised when an invalid hypercall is attempted (wrong number or type of
  parameters, undefined parameter values)

When a fault is raised, control is transferred to a user-defined *fault handler* (a function the
address of which is held in the `_VM_CR_FaultHandler` control register). Out of the box, DiOS (see
[DiOS, DIVINE’s Operating System]) provides a configurable fault handler. If a fault handler is set,
faults are not fatal (the only exception is a double fault, that is, a fault that occurs while the
fault handler itself is active). The fault handler, possibly with cooperation from the scheduler
(see Section [7.5][65]), can terminate the program, or raise the `_VM_CF_Error` flag, or take other
appropriate actions.

The handler can also choose to continue with execution despite the fault, by transferring control to
the activation frame and program counter value that are provided by the VM for this purpose. (Note:
this is necessary, because the fault might occur in the middle of evaluating a control flow
instruction, in which case, the VM could not finish its evaluation. The continuation passed to the
fault handler is the best estimate by the VM on where the execution should resume. The fault handler
is free to choose a different location.)

## 7.7 Boot Sequence

The virtual machine explicitly recognizes two modes of execution: privileged (kernel) mode and
normal, unprivileged user mode. When the VM is started, it looks up a function named `__boot` in the
bitcode file and starts executing this function, in kernel mode. The responsibility of this function
is to set up the operating system and set up the VM state for execution of the user program. There
are only two mandatory steps in the boot process: set the `_VM_CR_Scheduler` and the `_VM_CR_State`
control registers (see above). An optional (but recommended) step is to inform the VM (or more
specifically, any debugging or verification tools outside the VM) about the C/C++ *type* (or DWARF
type, to be precise, as this is also possible for non-C languages) associated with the OS state.
This is accomplished by an appropriate `__vm_trace` call (see also below).

## 7.8 Memory Management Hypercalls

Since LLVM bitcode is not tied to a memory representation, its apparatus for memory management is
quite limited. Just like in C, `malloc`, `free`, and related functions are provided by libraries,
but ultimately based on some lower-level mechanism, like, for example, the `mmap` system call. This
is often the case in POSIX systems targeting machines with a flat-addressed virtual memory system:
`mmap` is tailored to allocate comparatively large, contiguous chunks of memory (the requested size
must be an integer multiple of hardware page size) and management of individual objects is done
entirely in user-level code. Lack of any per-object protections is also a source of many common
programming errors, which are often hard to detect and debug.

It is therefore highly desirable that a single object obtained from `malloc` corresponds to a single
VM-managed and properly isolated object. This way, object boundaries can easily be enforced by the
model checker, and any violations reported back to the user. This means that, instead of subdividing
memory obtained from `mmap`, the `libc` running in DiVM uses `obj_make` to create a separate object
for each memory allocation. The `obj_make` hypercall obtains the object size as a parameter and
writes the address of the newly created object into the corresponding LLVM register (LLVM registers
are stored in memory, and therefore participate in the graph structure; this is described in more
detail in Section **¿sec:frames?**). Therefore, the newly created object is immediately and
atomically connected to the rest of the memory graph.

The standard counterpart to `malloc` is `free`, which returns memory, which is no longer needed by
the program, into the pool used by `malloc`. Again, in DiVM, there is a hypercall – `obj_free` –
with a role similar to that of standard `free`. In particular, `obj_free` takes a pointer as an
argument, and marks the corresponding object as *invalid*. Any further access to this object is a
*fault* (faults are described in more detail in Section [7.6][66]). The remaining hypercalls in the
`obj_` family exist to simplify bookkeeping and are not particularly important to the semantics of
the language.

## 7.9 Non-deterministic Choice and Counterexamples

It is often the case that the behaviour of a program depends on outside influences, which cannot be
reasonably described in a deterministic fashion and wired into the SUT. Such influences are
collectively known as the *environment*, and the effects of the environment translate into
non-deterministic behaviour. A major source of this non-determinism is thread interleaving – or,
equivalently, the choice of which thread should run next after an interrupt.

In our design, all non-determinism in the program (and the operating system) is derived from uses of
the `choose` hypercall (which non-deterministically returns an integer between 0 and a given
number). Since everything else in the SUT is completely deterministic, the succession of values
produced by calls to `choose` specifies an execution trace unambiguously. This trait makes it quite
simple to store counterexamples and other traces in a tool-neutral, machine-readable fashion.
Additionally, hints about which interrupts fired can be included in case the counterexample consumer
does not wish to reproduce the exact interrupt semantics of the given VM implementation.

Finally, the `trace` hypercall serves to attach additional information to transitions in the
execution graph. In particular, this information then becomes part of the counterexample when it is
presented to the user. For example, the `libc` provided by DIVINE uses the `trace` hypercall in the
implementation of standard IO functions. This way, if a program prints something to its standard
output during the violating run, this output becomes visible in the counterexample.

## 7.10 Debug Mode

The virtual machine has a special *debug* mode which allows instrumentation of the program under
test with additional tracing or other information gathering functionality. This is achieved by a
special `dbg.call` instruction, which is emitted by the bitcode loader whenever it encounters an
LLVM `call` instruction that targets a function annotated with `divine.debugfn`. For instance, the
DiOS tracing functions (`__dios_trace*`) carry this annotation. The virtual machine has two
operation modes differentiated by how they treat `dbg.call` instructions. In the *debug allowed*
mode, the instruction is executed and for the duration of the call, the VM enters a *debug mode*. In
the other mode (debug forbidden), the instruction is simply ignored.

Typically, verification (state space generation) would be done in the *debug forbidden* operation
mode, while the counter-example trace would be obtained or replayed in the *debug allowed* mode. To
make this approach feasible, there are certain limitations on the behaviour of a function called
using `dbg.call`:

* when in *debug mode* (i.e. when already executing a `dbg.call` instruction), all further
  `dbg.call` instructions are *ignored*
* faults in debug mode always cause the execution of `dbg.call` to be abandoned and the program
  continues executing as if the `dbg.call` returned normally
* interrupts and the `vm_choose` hypercall are forbidden in `dbg.call` and both will cause a fault
  (and hence abandonment of the call)

# 8 DiOS, A Small, Verification-Oriented Operating System

Programs traditionally rely on a wide array of services provided by their runtime environment (that
is, a combination of libraries, the operating system kernel, the hardware architecture and its
peripherals and so on). When DIVINE executes a program, it needs to provide these services to the
program. Some of those, especially library functions, can be obtained the same way they are in a
traditional (real) execution environment: the libraries can be compiled just like other programs and
the resulting bitcode files can be linked just as easily.

The remaining services, though, must be somehow supplied by DIVINE, since they are not readily
available as libraries. Some of them are part of the virtual machine, like memory management and
interrupt control (cf. [The DIVINE Virtual Machine]). The rest is provided by an “operating system”.
In principle, you can write your own small operating system for use with DIVINE; however, to make
common verification tasks easier, DIVINE ships with a small OS that implements a subset of POSIX
interfaces that should cover the requirements of a typical user-level program.

## 8.1 DiOS Compared to Traditional Operating Systems

The main goal of DiOS is to provide a runtime environment for programs under inspection, which
should not be distinguishable from the real environment. To achieve this goal, it presents an API
for thread (and process in the future) handling, faults and virtual machine configuration. DiOS API
is then used to implement the POSIX interface and supports the implementation of standard C and C++
library.

However, there are a few differences to the real POSIX-compatible OSs the user should be aware of:

* First of all, DiOS is trapped inside DIVINE VM, and therefore, it cannot perform any I/O
  operations with the outside world. All I/O has to be emulated.
* Consequently, DiOS cannot provide an access to a real file system, but supplies tools for
  capturing a file system snapshot, which can be used for emulation of file operations. See the file
  system section of this manual for further information.
* As the goal of the verification is to show that a program is safe no matter what scheduling
  choices are made, the thread scheduling differs from that of standard OSs. The user should not
  make any assumptions about it.
* DiOS currently does not cover the entire POSIX standard; however the support is commonly growing.

## 8.2 Fault Handling And Error Traces

When DIVINE 4 VM performs an illegal operation (e.g. division by zero or null pointer dereference),
it performs a so-called fault and a user supplied function, the fault handler, is called. This
function can react to the error - collect additional information or decide how the error should be
handled. DiOS provides its own fault handler, so that verified programs do not have to.

The DiOS fault handler prints a simple human readable stack trace together with a short summary of
the error. When a fault is triggered, it can either abort the verification or continue execution –
depending on its configuration for a given fault. Please see the next section for configuration
details.

Consider the following simple C++ program:

`int main()
{
    int *a = 0;
    *a = 42;
    return 0;
}`

This program does nothing interesting, it just triggers a fault. If we execute it using `divine run
test.cpp`, we obtain the following output:

`FAULT: null pointer dereference: [global* 0 0 ddp]
[0] FATAL: memory error in userspace`

To make debugging the problem easier, DiOS can print a backtrace when a fault is encountered (this
is disabled by default, since the `verify` command prints a more detailed backtrace without the help
of DiOS – see below):

`$ divine run -o debug:faultbt test.cpp`

The output then is:

`FAULT: null pointer dereference: [global* 0 0 ddp]
[0] FATAL: memory error in userspace
[0] Backtrace:
[0]   1: main
[0]   2: _start`

By inspecting the trace, we can see that a fault was triggered. When the VM triggers a fault, it
prints the reason – here a null pointer dereference caused the problem. The error caused the DiOS
fault handler to be called. The fault handler first communicates what the problem was and whether
the fault occurred in the DiOS kernel or in userspace. This is followed by a simple backtrace. Note
that `_start` is a DiOS function and is always at the bottom of a backtrace. It calls all global
constructors, initialises the standard libraries and calls `main` with the right arguments.

As mentioned above, `divine verify` will give us more information about the problem:

`$ divine verify test.cpp`

produces the following:

`error found: yes
error trace: |
  FAULT: null pointer dereference: [global* 0 0 ddp]
  [0] FATAL: memory error in userspace

active stack:
  - symbol: void {Fault}::handler<{Context} >(_VM_Fault, _VM_Frame*, void (*)(), ...)
    location: /divine/include/dios/core/fault.hpp:184
  - symbol: main
    location: test.cpp:3
  - symbol: _start
    location: /divine/src/libc/functions/sys/start.cpp:76
a report was written to test.report`

The error trace is the same as in the previous case, the ‘active stack’ section contains backtraces
for all the active threads. In this example, we only see one backtrace, since this is a single
threaded program. In addition to the earlier output provided by DiOS, the fault handler is also
visible.

## 8.3 DiOS Configuration

DIVINE supports passing of boot parameters to the OS. Some of these parameters are automatically
generated by DIVINE (e.g. the name of the program, program parameters or a snapshot of a file
system), others can be supplied by the user. These parameters can be specified using the command
line option `-o {argument}`.

DiOS expects `{argument}` to be in the form `{command}:{argument}`. DiOS help can be also invoked
with the shortcut `-o help`. All arguments are processed during the boot phase: if an invalid
command or argument is passed, DiOS fails to boot. DIVINE handles this state as unsuccessful
verification and the output contains a description of the error. DiOS supports the following
commands:

* `debug`: print debug information during the boot. By default no information is printed during the
  boot. Supported arguments:
  
  * `help`: print help and abort the boot,
  * `machineparams`: print user specified machine parameters, e.g. the number of cpus,
  * `mainargs`: print `argv` and `envp`, which will be passed to the `main` function,
  * `faultcfg`: print fault and simfail configuration, which will be used for verification; note
    that if the configuration is not forced, the program under inspection can change the
    configuration.
* `trace` and `notrace`: report/do not report state of its argument back to VM; supported arguments:
  
  * `threads`: report all active threads back to the VM, so it can e.g. allow user to choose which
    thread to run. By default, threads are not traced.
* `ignore`, `report`, `abort` with variants prefixed with `force-`: configure handling of a given
  fault type. When `abort` is specified, DiOS passes the fault as an error back to the VM and the
  verification is terminated. Faults marked as `report` are reported back to the VM, but are not
  treated as errors – the verification process may continue past the fault. When a fault is ignored,
  it is not reported back to the VM at all. If the execution after a fault continues, the
  instruction causing the fault is ignored or produces an undefined value. The following fault
  categories are present in DIVINE and these faults can be passed to the command:
  
  * `assert`: an `assert` call fails,
  * `arithmetic`: arithmetic errors – e.g. division by 0,
  * `memory`: access to uninitialised memory or an invalid pointer dereference,
  * `control`: check the return value of a function and jump on undefined values,
  * `hypercall`: an invalid parameter to a VM hypercall was passed,
  * `notimplemented`: attempted to perform an unimplemented operation,
  * `diosassert`: an internal DiOS assertion was violated.
* `simfail` and `nofail`: simulate a possible failure of the given feature. E.g. `malloc` can fail
  in a real system and therefore, when set to `simfail`, both success and failure of `malloc` are
  tested. Supported arguments:
  
  * `malloc`: simulate failure of memory allocation.
* `ncpus:<num>`: specify number of machine physical cores – this has no direct impact on
  verification and affects only library calls which can return number of cores.
* `stdout`: specify how to treat the standard output of the program; the following options are
  supported:
  
  * `notrace`: the output is ignored.
  * `unbuffered`: each write is printed on a separate line,
  * `buffered`: each line of the output is printed (default)
* `stderr`: specify how to treat the standard error output; see also `stdout`.

## 8.4 Virtual File System

DiOS provides a POSIX-compatible filesystem implementation. Since no real I/O operations are
allowed, the *Virtual File System* (VFS) operates on top of a filesystem snapshot and the effects of
operations performed on the VFS are not propagated back to the host. The snapshots are created by
DIVINE just before DiOS boots; to create a snapshot of a directory, use the option `--capture
{vfsdir}`, where `{vfsdir}` consists of up to three `:`-separated components:

* path to a directory (mandatory),
* `follow` or `nofollow` – specifies whether symlink targets should or should not be captured
  (optional, defaults to `follow`),
* the “mount point” in the VFS – if not specified, it is the same as the capture path

DIVINE can capture files, directories, symlinks and hardlinks. Additionally, DiOS can also create
pipes and UNIX sockets but these cannot be captured from the host system by DIVINE.

The size of the snapshot is by default limited to 16 MiB. This prevents accidental capture of a
large snapshot. Example:

`divine verify --capture testdir:follow:/home/test/ --vfslimit 1kB test.cpp`

Additionally, the stdin of the program can be provided in a file and used using the DIVINE switch
`--stdin {file}`. Finally, the standard and error output can be handled in one of several ways:

* it can be completely ignored
* it is traced and becomes part of transition labels
  
  * in a line-buffered fashion (this is the default behaviour)
  * in an unbuffered way, where each `write` is printed as a single line of the trace

All of the above options can be specified via DiOS boot parameters. See Section [8.3][67] for more
details.

The VFS implements basic POSIX syscalls – `write`, `read`, etc. and standard C functions like
`printf`, `scanf` or C++ streams are implemented on top of these. All functions which operate on the
filesystem only modify the internal filesystem snapshot.

[1]: #introduction
[2]: #installation
[3]: #prerequisites
[4]: #building-installing
[5]: #an-introduction-to-using-divine
[6]: #basics-of-program-analysis
[7]: #debugging-counterexamples-with-the-interactive-simulator
[8]: #controlling-the-execution-environment
[9]: #compilation-options-and-compilation-of-multiple-files
[10]: #commandline-interface
[11]: #synopsis
[12]: #input-options
[13]: #state-space-visualisation-simulation
[14]: #model-checking
[15]: #model-checking-c-and-c-code-via-llvm-bitcode
[16]: #compiling-programs
[17]: #limitations
[18]: #state-space-of-a-program
[19]: #non-deterministic-choice
[20]: #ω-regular-properties-and-ltl
[21]: #symbolic-verification
[22]: #sim
[23]: #tutorial
[24]: #collecting-information
[25]: #divm-a-virtual-machine-for-verification
[26]: #activation-frames
[27]: #sec:control
[28]: #heap
[29]: #the-hypercall-interface
[30]: #sec:scheduling
[31]: #sec:faults
[32]: #boot-sequence
[33]: #memory-management-hypercalls
[34]: #sec:nondeterminism
[35]: #sec:debugmode
[36]: #dios
[37]: #dios-compared-to-traditional-operating-systems
[38]: #fault-handling-and-error-traces
[39]: #sec:dios.config
[40]: #virtual-file-system
[41]: http://divine.fi.muni.cz
[42]: http://www.cmake.org
[43]: http://thrysoee.dk/editline/
[44]: https://ninja-build.org
[45]: http://pandoc.org
[46]: #sim
[47]: #dios
[48]: #sim
[49]: https://sv-comp.sosy-lab.org/
[50]: #sec:debugmode
[51]: #sec:control
[52]: http://llvm.org/docs/LangRef.html
[53]: #sec:control
[54]: #heap
[55]: #sec:scheduling
[56]: #sec:scheduling
[57]: #sec:faults
[58]: #state-space-of-a-program
[59]: #sec:scheduling
[60]: #sec:faults
[61]: #ω-regular-properties-and-ltl
[62]: #tbl:hypercalls
[63]: #boot-sequence
[64]: #state-space-of-a-program
[65]: #sec:scheduling
[66]: #sec:faults
[67]: #sec:dios.config
