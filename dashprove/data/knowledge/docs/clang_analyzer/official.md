────────────────────────────────────────────────────────────────────────────────────────┬───────────
# Clang Static Analyzer                                                                 │[[analyzer 
                                                                                        │in         
The Clang Static Analyzer is a source code analysis tool that finds bugs in C, C++, and │xcode]][7] 
Objective-C programs.                                                                   │Viewing    
                                                                                        │static     
The analyzer is 100% open source and is part of the [Clang][1] project. Like the rest of│analyzer   
Clang, the analyzer is implemented as a C++ library that can be used by other tools and │results in 
applications.                                                                           │Xcode      
                                                                                        │[[analyzer 
To get started with the Clang Static Analyzer, visit the [LLVM releases page][2] for    │in         
download and installation instructions. The official releases include both the analyzer │browser]][8
and [scan-build][3], a command-line tool for running the analyzer on your codebase.     │]          
                                                                                        │Viewing    
If you're installing Clang from a different source, such as a Linux package repository, │static     
then scan-build may be packaged separately as an individual package, or as part of a    │analyzer   
"clang tools" package.                                                                  │results in 
                                                                                        │a web      
If your IDE is using Clang, it may natively integrate the static analyzer. On macOS, the│browser    
easiest way to use the static analyzer is to invoke it [directly from Xcode][4].        │           
                                                                                        │           
Additionally, if you're using [clang-tidy][5], you can naturally make the static        │           
analyzer run alongside clang-tidy by enabling the [clang-analyzer][6] checks.           │           
────────────────────────────────────────────────────────────────────────────────────────┴───────────

## What is Static Analysis?

The term "static analysis" is conflated, but here we use it to mean a collection of algorithms and
techniques used to analyze source code in order to automatically find bugs. The idea is similar in
spirit to compiler warnings (which can be useful for finding coding errors) but to take that idea a
step further and find bugs that are traditionally found using run-time debugging techniques such as
testing.

Static analysis bug-finding tools have evolved over the last several decades from basic syntactic
checkers to those that find deep bugs by reasoning about the semantics of code. The goal of the
Clang Static Analyzer is to provide a industrial-quality static analysis framework for analyzing C,
C++, and Objective-C programs that is freely available, extensible, and has a high quality of
implementation.

### Part of Clang and LLVM

As its name implies, the Clang Static Analyzer is built on top of [Clang][9] and [LLVM][10].
Strictly speaking, the analyzer is part of Clang, as Clang consists of a set of reusable C++
libraries for building powerful source-level tools. The static analysis engine used by the Clang
Static Analyzer is a Clang library, and has the capability to be reused in different contexts and by
different clients.

## Important Points to Consider

While we believe that the static analyzer is already very useful for finding bugs, we ask you to
bear in mind a few points when using it.

### Work-in-Progress

The analyzer is a continuous work-in-progress. There are many planned enhancements to improve both
the precision and scope of its analysis algorithms as well as the kinds of bugs it will find. While
there are fundamental limitations to what static analysis can do, we have a long way to go before
hitting that wall.

### Slower than Compilation

Operationally, using static analysis to automatically find deep program bugs is about trading CPU
time for the hardening of code. Because of the deep analysis performed by state-of-the-art static
analysis tools, static analysis can be much slower than compilation.

While the Clang Static Analyzer is being designed to be as fast and light-weight as possible, please
do not expect it to be as fast as compiling a program (even with optimizations enabled). Some of the
algorithms needed to find bugs require in the worst case exponential time.

The Clang Static Analyzer runs in a reasonable amount of time by both bounding the amount of
checking work it will do as well as using clever algorithms to reduce the amount of work it must do
to find bugs.

### False Positives

Static analysis is not perfect. It can falsely flag bugs in a program where the code behaves
correctly. Because some code checks require more analysis precision than others, the frequency of
false positives can vary widely between different checks. Our long-term goal is to have the analyzer
have a low false positive rate for most code on all checks.

Please help us in this endeavor by [reporting false positives][11]. False positives cannot be
addressed unless we know about them.

### More Checks

Static analysis is not magic; a static analyzer can only find bugs that it has been specifically
engineered to find. If there are specific kinds of bugs you would like the Clang Static Analyzer to
find, please feel free to file [feature requests][12] or contribute your own patches.

[1]: https://clang.llvm.org
[2]: https://releases.llvm.org/download.html
[3]: scan-build.html
[4]: https://clang.llvm.org/docs/analyzer/user-docs/UsingWithXCode.html
[5]: https://clang.llvm.org/extra/clang-tidy/
[6]: https://clang.llvm.org/extra/clang-tidy/checks/list.html
[7]: images/analyzer_xcode.png
[8]: images/analyzer_html.png
[9]: https://clang.llvm.org
[10]: https://llvm.org
[11]: filing_bugs.html
[12]: filing_bugs.html
