**Cppcheck** is a [static analysis tool][1] for C/C++ code. It provides [unique code analysis][2] to
detect bugs and focuses on detecting undefined behaviour and dangerous coding constructs. The goal
is to have very few false positives. Cppcheck is designed to be able to analyze your C/C++ code even
if it has non-standard syntax (common in embedded projects).

Cppcheck is available both as open-source (this page) and as Cppcheck Premium with extended
functionality and support. Please visit [www.cppcheck.com][3] for more information and purchase
options for the commercial version.

## Download

### Cppcheck 2.19 (open source)

──────────────────────────────┬──────────────
Platform                      │File          
──────────────────────────────┼──────────────
Windows 64-bit (No XP support)│[Installer][4]
──────────────────────────────┼──────────────
Source code (.zip)            │[Archive][5]  
──────────────────────────────┼──────────────
Source code (.tar.gz)         │[Archive][6]  
──────────────────────────────┴──────────────

### Packages

Cppcheck can also be installed from various package managers; however, you might get an outdated
version then.

#### Debian:

sudo apt-get install cppcheck

#### Fedora:

sudo yum install cppcheck

#### Mac:

brew install cppcheck

## Features

Unique code analysis that detect various kinds of bugs in your code.

Both command line interface and graphical user interface are available.

Cppcheck has a strong focus on detecting undefined behaviour.

### Unique analysis

Using several static analysis tools can be a good idea. There are unique features in each tool. This
has been established in many studies.

So what is unique in Cppcheck.

Cppcheck uses unsound flow sensitive analysis. Several other analyzers use path sensitive analysis
based on abstract interpretation, that is also great however that has both advantages and
disadvantages. In theory by definition, it is better with path sensitive analysis than flow
sensitive analysis. But in practice, it means Cppcheck will detect bugs that the other tools do not
detect.

In Cppcheck the data flow analysis is not only "forward" but "bi-directional". Most analyzers will
diagnose this:

void foo(int x)
{
    int buf[10];
    if (x == 1000)
        buf[x] = 0; // <- ERROR
}

Most tools can determine that the array index will be 1000 and there will be overflow.

Cppcheck will also diagnose this:

void foo(int x)
{
    int buf[10];
    buf[x] = 0; // <- ERROR
    if (x == 1000) {}
}

### Undefined behaviour

* Dead pointers
* Division by zero
* Integer overflows
* Invalid bit shift operands
* Invalid conversions
* Invalid usage of STL
* Memory management
* Null pointer dereferences
* Out of bounds checking
* Uninitialized variables
* Writing const data

### Security

The most common types of security vulnerabilities in 2017 (CVE count) was:

───────────────────────────────────────────┬─────────┬────────────────────────────
Category                                   │Amount   │Detected by Cppcheck        
───────────────────────────────────────────┼─────────┼────────────────────────────
Buffer Errors                              │[2530][7]│A few                       
───────────────────────────────────────────┼─────────┼────────────────────────────
Improper Access Control                    │[1366][8]│A few (unintended backdoors)
───────────────────────────────────────────┼─────────┼────────────────────────────
Information Leak                           │[1426][9]│A few (unintended backdoors)
───────────────────────────────────────────┼─────────┼────────────────────────────
Permissions, Privileges, and Access Control│[1196][10│A few (unintended backdoors)
                                           │]        │                            
───────────────────────────────────────────┼─────────┼────────────────────────────
Input Validation                           │[968][11]│No                          
───────────────────────────────────────────┴─────────┴────────────────────────────

CVEs that was found using Cppcheck:

* [CVE-2017-1000249][12] : file : stack based buffer overflow
  This was found by Thomas Jarosch using Cppcheck. The cause is a mistake in a condition.
* [CVE-2013-6462][13] : 23 year old stack overflow in X.org that was found with Cppcheck.
  This has been described in a few articles ([link][14]).
* [CVE-2012-1147][15] : readfilemap.c in expat before 2.1.0 allows context-dependent attackers to
  cause a denial of service (file descriptor consumption) via a large number of crafted XML files..

These CVEs are shown when you google "cppcheck CVE". Feel free to compare the search results with
other static analysis tools.

Security experts recommend that static analysis is used. And using several tools is the best
approach from a security perspective.

### Coding standards

─────────────────────────────────┬──────────────┬─────────────
Coding standard                  │Open Source   │Premium      
─────────────────────────────────┼──────────────┼─────────────
Misra C 2012 - original rules    │Partial       │Yes          
─────────────────────────────────┼──────────────┼─────────────
Misra C 2012 - amendment #1      │Partial       │Yes          
─────────────────────────────────┼──────────────┼─────────────
Misra C 2012 - amendment #2      │Partial       │Yes          
─────────────────────────────────┼──────────────┼─────────────
Misra C 2012 - amendment #3      │              │Yes          
─────────────────────────────────┼──────────────┼─────────────
Misra C 2012 - amendment #4      │              │Yes          
─────────────────────────────────┼──────────────┼─────────────
Misra C 2012 - Compliance report │              │Yes          
─────────────────────────────────┼──────────────┼─────────────
Misra C 2012 - Rule texts        │User provided │Yes          
─────────────────────────────────┼──────────────┼─────────────
Misra C 2023                     │              │Yes          
─────────────────────────────────┼──────────────┼─────────────
Misra C++ 2008                   │              │Yes          
─────────────────────────────────┼──────────────┼─────────────
Misra C++ 2023                   │              │Yes          
─────────────────────────────────┼──────────────┼─────────────
Cert C                           │              │Yes          
─────────────────────────────────┼──────────────┼─────────────
Cert C++                         │              │Yes          
─────────────────────────────────┼──────────────┼─────────────
Autosar                          │              │[Partial][16]
─────────────────────────────────┴──────────────┴─────────────

### All checks

For a list of all checks in Cppcheck see: [http://sourceforge.net/p/cppcheck/wiki/ListOfChecks][17].

## Clients and plugins

Cppcheck is integrated with many popular development tools. For instance:

* **Buildbot** - [integrated][18]
* **CLion** - [Cppcheck plugin][19]
* **Code::Blocks** - *integrated*
* **CodeDX** (software assurance tool) - [integrated][20]
* **CodeLite** - *integrated*
* **CppDepend 5** - [integrated][21]
* **Eclipse** - [Cppcheclipse][22]
* **gedit** - [gedit plugin][23]
* **github** - [Codacy][24], [Codety][25] and [SoftaCheck][26]
* **Hudson** - [Cppcheck Plugin][27]
* **Jenkins** - [Cppcheck Plugin][28]
* **KDevelop** - [integrated since v5.1][29]
* **Mercurial (Linux)** - [pre-commit hook][30] - Check for new errors on commit (requires
  interactive terminal)
* **QtCreator** - [Qt Project Tool (qpt)][31]
* **Tortoise SVN** - [Adding a pre-commit hook script][32]
* **Vim** - [Vim Compiler][33]
* **Visual Studio** - [Visual Studio plugin][34]
* **VScode** - [VScode plugin][35]

## Other static analysis tools

Using a battery of tools is better than using one tool. Each tool has unique code analysis and
therefore we recommend that you also use other tools.

Cppcheck focus on bugs instead of stylistic issues. Therefore a tool that focus on stylistic issues
could be a good addition.

Cppcheck tries very hard to avoid false positives. Sometimes people want to detect all bugs even if
there will be many false warnings, for instance when they are working on a release and want to
verify that there are no bugs. A tool that is much more noisy than Cppcheck might be a good
addition.

Even tools that have the same design goals as Cppcheck will probably be good additions. Static
analysis is such a big field, Cppcheck only covers a small fraction of it. No tool covers the whole
field. The day when all manual testing will be obsolete because of some tool is very far away.

## News


[View all news…][36]

## Documentation

You can read the [manual][37] or download some [articles][38].

## Support

* Use [Trac][39] to report bugs and feature requests
* Ask questions at the IRC channel [#cppcheck][40]

## Donate CPU

The Cppcheck project is a hobby project with limited resources. You can help us by donating CPU (1
core or as many as you like). It is simple:

1. Download (and extract) Cppcheck source code
2. Run script: python cppcheck/tools/donate-cpu.py

The script will analyse debian source code and upload the results to a cppcheck server. We need
these results both to improve Cppcheck and to detect regressions.

You can stop the script whenever you like with Ctrl C.

## Contribute

You are welcome to contribute. Help is needed.

A presentation that might be interesting: [Contribute to open source static analysis][41]

*Testing*
  Pick a project and test its source with the latest version of Cppcheck. Submit tickets to
  [Trac][42] about the issues you find in Cppcheck.
*Developing*
  Pick a ticket from [Trac][43], write a test case for it (and write a comment to the ticket for
  which that test case has been created). Alternatively, pick a test case that fails and try to fix
  it. Make a patch and submit it to Trac either inline, if it is small, or otherwise - attach it as
  a file.
*Marketing*
  Write articles, reviews or tell your friends about us. The more users we have, the more people we
  have testing and the better we can become.
*Design*
  Come up with some new good checks, and create tickets in [the Trac instance][44] about them.
*Integration*
  Write a plugin for your favorite IDE or create a package for your distribution or operating
  system.
*Technical Writing*
  Write better documentation for the bugs we find. Currently only a few bugs have any documentation
  at all.

[1]: http://en.wikipedia.org/wiki/Static_analysis_tool
[2]: #unique
[3]: https://www.cppcheck.com?utm_source=sourceforge&utm_medium=opensource&utm_campaign=websitelink
[4]: https://github.com/danmar/cppcheck/releases/download/2.19.0/cppcheck-2.19.0-x64-Setup.msi
[5]: https://github.com/danmar/cppcheck/archive/2.19.0.zip
[6]: https://github.com/danmar/cppcheck/archive/2.19.0.tar.gz
[7]: https://nvd.nist.gov/vuln/search/statistics?results_type=statistics&cwe_id=CWE-119
[8]: https://nvd.nist.gov/vuln/search/statistics?results_type=statistics&cwe_id=CWE-284
[9]: https://nvd.nist.gov/vuln/search/statistics?results_type=statistics&cwe_id=CWE-200
[10]: https://nvd.nist.gov/vuln/search/statistics?results_type=statistics&cwe_id=CWE-264
[11]: https://nvd.nist.gov/vuln/search/statistics?results_type=statistics&cwe_id=CWE-20
[12]: https://nvd.nist.gov/vuln/detail/CVE-2017-1000249
[13]: https://nvd.nist.gov/vuln/detail/CVE-2013-6462
[14]: https://www.theregister.co.uk/2014/01/09/x11_has_privilege_escalation_bug/
[15]: https://nvd.nist.gov/vuln/detail/CVE-2012-1147
[16]: https://files.cppchecksolutions.com/autosar.html
[17]: http://sourceforge.net/p/cppcheck/wiki/ListOfChecks
[18]: https://docs.buildbot.net/latest/manual/configuration/steps/cppcheck.html
[19]: https://plugins.jetbrains.com/plugin/8143
[20]: http://codedx.com/code-dx-standard/
[21]: http://www.cppdepend.com/CppDependV5.aspx
[22]: https://github.com/cppchecksolutions/cppcheclipse/wiki/Installation
[23]: http://github.com/odamite/gedit-cppcheck
[24]: https://www.codacy.com/
[25]: https://www.codety.io/
[26]: http://www.softacheck.com/
[27]: http://wiki.hudson-ci.org/display/HUDSON/Cppcheck+Plugin
[28]: http://wiki.jenkins-ci.org/display/JENKINS/Cppcheck+Plugin
[29]: https://kdevelop.org/
[30]: http://sourceforge.net/p/cppcheck/wiki/mercurialhook/
[31]: https://sourceforge.net/projects/qtprojecttool/files
[32]: http://omerez.com/automatic-static-code-analysis/
[33]: https://vimhelp.org/quickfix.txt.html#compiler-cppcheck
[34]: https://github.com/VioletGiraffe/cppcheck-vs-addin/releases/latest
[35]: https://marketplace.visualstudio.com/items?itemName=NathanJ.cppcheck-plugin
[36]: https://sourceforge.net/p/cppcheck/news/
[37]: manual.pdf
[38]: http://sourceforge.net/projects/cppcheck/files/Articles/
[39]: http://trac.cppcheck.net
[40]: irc://irc.libera.chat/#cppcheck
[41]: https://www.youtube.com/watch?v=Cc_U1Hil0S4
[42]: http://trac.cppcheck.net
[43]: http://trac.cppcheck.net
[44]: http://trac.cppcheck.net
