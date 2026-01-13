OCLint is a [static code analysis][1] tool for improving quality and reducing defects by inspecting
C, C++ and Objective-C code and looking for potential problems like:

* Possible bugs - empty if/else/try/catch/finally statements
* Unused code - unused local variables and parameters
* Complicated code - high cyclomatic complexity, NPath complexity and high NCSS
* Redundant code - redundant if statement and useless parentheses
* Code smells - long method and long parameter list
* Bad practices - inverted logic and parameter reassignment
* ...

Static code analysis is a critical technique to detect defects that aren't visible to compilers.
OCLint automates this inspection process with advanced features:

* Relying on the [abstract syntax tree][2] of the source code for better accuracy and efficiency;
  False positives are mostly reduced to avoid useful results sinking in them.
* Dynamically loading rules into system, even in the runtime.
* Flexible and extensible configurations ensure users in customizing the behaviors of the tool.
* Command line invocation facilitates [continuous integration][3] and continuous inspection of the
  code while being developed, so that technical debts can be fixed early on to reduce the
  maintenance cost.

### Get OCLint

OCLint is a standalone tool that runs on Linux and Mac OS X platforms. You can get the latest
version by [building from source code][4] or [downloading pre-compiled binaries][5]. The OCLint
source code and binaries are distributed under [Modified BSD License][6].

### Documentation

* [Installation][7]
* [Tutorial][8]
* [Manual][9]
* [Guide][10]
* [Full index][11]

### News

* [Releasing OCLint 21.10][12] *Oct 26, 2021*
* [Releasing OCLint 21.03][13] *Mar 06, 2021*
* [Releasing OCLint 20.11][14] *Nov 18, 2020*
* [Releasing OCLint 0.13][15] *Sep 30, 2017*
* [Releasing OCLint 0.12][16] *Mar 30, 2017*

### Inspirations and Acknowledgment

We're inspired by the great static analysis tools like [P.M.D.][17] for Java and [CodeNarc][18] for
Groovy, as well as the smart code inspections performed by Jetbrains [IntelliJ IDEA][19] and
[AppCode][20].

OCLint is based on [Clang Tooling][21], it's a handy library with great support writing standalone
tools. We would like to appreciate the entire [LLVM community][22] for their hard work.

[1]: https://en.wikipedia.org/wiki/Static_program_analysis
[2]: https://en.wikipedia.org/wiki/Abstract_syntax_tree
[3]: http://martinfowler.com/articles/continuousIntegration.html
[4]: https://docs.oclint.org/en/stable/intro/build.html
[5]: https://github.com/oclint/oclint/releases
[6]: https://github.com/oclint/oclint/blob/master/LICENSE
[7]: https://docs.oclint.org/en/stable/intro/installation.html
[8]: https://docs.oclint.org/en/stable/intro/tutorial.html
[9]: https://docs.oclint.org/en/stable/manual/index.html
[10]: https://docs.oclint.org/en/stable/guide/index.html
[11]: https://docs.oclint.org/en/stable/
[12]: /news/2021/10/26/release-21-10.html
[13]: /news/2021/03/06/release-21-03.html
[14]: /news/2020/11/18/release-20-11.html
[15]: /news/2017/09/30/release-0-13.html
[16]: /news/2017/03/30/release-0-12.html
[17]: http://pmd.sourceforge.net/
[18]: http://codenarc.sourceforge.net/
[19]: http://www.jetbrains.com/idea/
[20]: http://www.jetbrains.com/objc/
[21]: http://clang.llvm.org/docs/LibTooling.html
[22]: http://llvm.org/
