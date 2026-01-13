As a part of [**research on decision procedures**][1], our Lab develops a tool, OpenSMT, that is
used as a framework to perform all our experiments.

OpenSMT is a compact and open-source SMT-solver written in C++, with the main goal of making
SMT-Solvers easy to understand and use as a computational engine for formal verification. OpenSMT is
built on top of [MiniSAT][2].

The git repository for [OpenSMT2][3] is available at

**git clone [https://github.com/usi-verification-and-security/opensmt.git][4]**

The new [OpenSMT2][5] is currently under active development, built on the solid foundations of the
previous OpenSMT generations but providing native support for the SMTLIB version 2 and more
efficient implementation for most internal data structures. OpenSMT2 is released under the MIT
license. For now, old [OpenSMT1][6] still provides support for a wider variety of theories and is
released under the General Public Licence version 3 license.

[1]: /FVBTR
[2]: http://minisat.se
[3]: /opensmt2
[4]: https://github.com/usi-verification-and-security/opensmt.git
[5]: /opensmt2
[6]: /opensmt1
