# CBMC

## About CBMC

[circuit] CBMC is a Bounded Model Checker for C and C++ programs. It sup­ports C89, C99, most of
C11/C17 and most compi­ler exten­sions pro­vided by gcc, clang, and Visual Studio. A variant of CBMC
that analyses Java bytecode is available as [JBMC][1]. For Rust, get [Kani][2].

CBMC verifies memory safety (which includes array bounds checks and checks for the safe use of
pointers), checks for various further variants of undefined behavior, and user-specified as­ser­tions.
Further­more, it can check C and C++ for I/O equivalence with other languages, such as Verilog. The
verification is performed by unwinding the loops in the program and passing the re­sul­ting equation
to a decision procedure.

CBMC is available for most flavours of Linux (pre-packaged for Debian/Ubuntu), Windows and MacOS.
You should also read the [CBMC license][3] (BSD 4-clause).

CBMC comes with a built-in solver for bit-vector formulas that is based on MiniSat. As an
alternative, CBMC has featured support for external SMT solvers since version 3.3. The solvers we
recommend are (in no particular order) [Boolector][4], [CVC5][5] and [Z3][6]. Note that these
solvers need to be installed separately and have different licensing conditions.

For questions about CBMC, contact [Daniel Kroening][7].

## CBMC News

CBMC version 6 has been released!

Major changes:

* Checks for undefined behavior are now on by default. Use `--no-standard-checks` to restore the
  previous default.
* The default behavior of `malloc` now enables returning `NULL`. Use `--no-malloc-may-fail` to
  restore the previous default.
* Calls to functions without body now trigger a verification error.
* The default verbosity is now 6. Use `--verbosity 8` to restore the previous default.

## CBMC Documentation

* The [CPROVER Manual][8] contains a tutorial from a user's point of view and describes what
  properties are checked.
* A set of slides on CBMC: [PDF][9], [2x3 handouts][10].
  The sources are available [here][11].
* The primary reference for CBMC is [A Tool for Checking ANSI-C Programs][12] (ca. 2000 citations).

We also have a [list of interesting applications of CBMC][13].

## CBMC Download

We are maintaining CBMC for x86 Linux, Windows and MacOS.

───────┬────────────────────────────────────────────────────────────────────────────────────────────
Windows│Download [cbmc-6.8.0-win64.msi][14], and then double-click to install. This is an x64 binary
       │for the command line (there is no GUI). You will need to run CBMC from the *Visual Studio   
       │Command Prompt*.                                                                            
       │                                                                                            
       │We recommend you install the free [ Visual Studio Community][15].                           
───────┼────────────────────────────────────────────────────────────────────────────────────────────
Ubuntu │Do                                                                                          
25.04  │`apt-get install cbmc`                                                                      
and    │to install. This will give you cbmc 6.6.0 on Debian Trixie (stable) and cbmc 6.4.1 on       
Debian │Ubuntu.                                                                                     
trixie │                                                                                            
───────┼────────────────────────────────────────────────────────────────────────────────────────────
Ubuntu │While Ubuntu 20/22.04 do have a cbmc package, this package is at version 5.12, which is more
22.04  │than a decade old. Download one of the following binary packages:                           
and    │64-bit Ubuntu 22.04/x64: [ubuntu-22.04-cbmc-6.8.0-Linux.deb][16]                            
24.04  │64-bit Ubuntu 24.04/x64: [ubuntu-24.04-cbmc-6.8.0-Linux.deb][17]                            
       │Do                                                                                          
       │`dpkg -i ubuntu-24.04-cbmc-6.8.0-Linux.deb`                                                 
       │to install.                                                                                 
───────┼────────────────────────────────────────────────────────────────────────────────────────────
Fedora │Do                                                                                          
       │`dnf install cbmc`                                                                          
       │On a recent version of Fedora (43), this will give you CBMC version 6.7.1. On Fedora 42, you
       │get 6.4.1.                                                                                  
───────┼────────────────────────────────────────────────────────────────────────────────────────────
MacOS  │To get a recent version of CBMC, consider using [Homebrew][18], and then do                 
       │`brew install cbmc`                                                                         
       │                                                                                            
       │This is a command-line tool only, there is no GUI. You need to have the Command Line Tools  
       │for Xcode, which can be downloaded by running the Xcode application or from [here][19].     
───────┼────────────────────────────────────────────────────────────────────────────────────────────
Source │Source code is available [here][20].                                                        
Code   │                                                                                            
───────┴────────────────────────────────────────────────────────────────────────────────────────────

If you need a Model Checker for Verilog or SystemVerilog, consider [EBMC][21].

This research was sponsored by the Semiconductor Research Corporation (SRC) under contract no.
99-TJ-684, the National Science Foundation (NSF) under grant no. CCR-9803774, the Office of Naval
Research (ONR), the Naval Research Laboratory (NRL) under contract no. N00014-01-1-0796, and by the
Defense Advanced Research Projects Agency, and the Army Research Office (ARO) under contract no.
DAAD19-01-1-0485, and the General Motors Collaborative Research Lab at CMU. The views and
conclusions contained in this document are those of the author and should not be interpreted as
representing the official policies, either expressed or implied, of SRC, NSF, ONR, NRL, DOD, ARO, or
the U.S. government.

[1]: /jbmc
[2]: https://model-checking.github.io/kani/
[3]: LICENSE.txt
[4]: https://boolector.github.io/
[5]: https://cvc5.github.io/
[6]: https://github.com/Z3Prover/z3/wiki
[7]: http://www.kroening.com/
[8]: http://www.cprover.org/cprover-manual/
[9]: doc/cbmc-slides.pdf
[10]: doc/cbmc-slides-2x3.pdf
[11]: https://github.com/diffblue/cbmc/tree/master/doc/slides/cbmc-latex-beamer
[12]: http://www-2.cs.cmu.edu/~svc/papers/view-publications-ckl2004.html
[13]: applications/
[14]: https://github.com/diffblue/cbmc/releases/download/cbmc-6.8.0/cbmc-6.8.0-win64.msi
[15]: https://visualstudio.microsoft.com/vs/community/
[16]: https://github.com/diffblue/cbmc/releases/download/cbmc-6.8.0/ubuntu-22.04-cbmc-6.8.0-Linux.de
b
[17]: https://github.com/diffblue/cbmc/releases/download/cbmc-6.8.0/ubuntu-24.04-cbmc-6.8.0-Linux.de
b
[18]: https://brew.sh/
[19]: https://developer.apple.com/download/all/?q=command%20line%20tools
[20]: https://github.com/diffblue/cbmc/
[21]: http://www.cprover.org/ebmc/
