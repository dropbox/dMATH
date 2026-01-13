[[License: MIT]][1]

# CaDiCaL Simplified Satisfiability Solver

The goal of the development of CaDiCaL was to obtain a CDCL solver, which is easy to understand and
change, while at the same time not being much slower than other state-of-the-art CDCL solvers.

Originally we wanted to also radically simplify the design and internal data structures, but that
goal was only achieved partially, at least for instance compared to Lingeling.

However, the code is much better documented and CaDiCaL actually became in general faster than
Lingeling even though it is missing some preprocessors (mostly parity and cardinality constraint
reasoning), which would be crucial to solve certain instances.

Use `./configure && make` to configure and build `cadical` and the library `libcadical.a` in the
default `build` sub-directory. The header file of the library is [`src/cadical.hpp`][2] and includes
an example for API usage.

See [`BUILD.md`][3] for options and more details related to the build process and
[`test/README.md`][4] for testing the library and the solver. Since release 1.5.1 we have a
[`NEWS.md`][5] file. You might also want to check out [`CONTRIBUTING.md`][6] on if you want to
contribute.

The solver has the following usage `cadical [ dimacs [ proof ] ]`. See `cadical -h` for more
options.

If you want to cite CaDiCaL please use as reference our CaDiCaL 2.0 tool paper which appeared at
CAV'24:

[CaDiCaL 2.0][7]
[Armin Biere][8], [Tobias Faller][9], [Katalin Fazekas][10], [Mathias Fleury][11], [Nils
Froleyks][12] and [Florian Pollitt][13]
* Proc. Computer Aidded Verification - 26th Intl. Conf. (CAV'24)*
Lecture Notes in Computer Science (LNCS)
vol. 14681, pages 133-152, Springer 2024
[ [paper][14] | [bibtex][15] | [official][16] | [artifact][17] | [github][18] | [doi][19] ]

[1]: https://opensource.org/licenses/MIT
[2]: /arminbiere/cadical/blob/master/src/cadical.hpp
[3]: /arminbiere/cadical/blob/master/BUILD.md
[4]: /arminbiere/cadical/blob/master/test/README.md
[5]: /arminbiere/cadical/blob/master/NEWS.md
[6]: /arminbiere/cadical/blob/master/CONTRIBUTING.md
[7]: https://cca.informatik.uni-freiburg.de/papers/BiereFallerFazekasFleuryFroleyksPollitt-CAV24.pdf
[8]: https://cca.informatik.uni-freiburg.de/biere
[9]: https://cca.informatik.uni-freiburg.de/fallert
[10]: https://kfazekas.github.io
[11]: https://cca.informatik.uni-freiburg.de/fleury
[12]: https://fmv.jku.at/froleyks
[13]: https://cca.informatik.uni-freiburg.de/pollittf
[14]: https://cca.informatik.uni-freiburg.de/papers/BiereFallerFazekasFleuryFroleyksPollitt-CAV24.pd
f
[15]: https://cca.informatik.uni-freiburg.de/papers/BiereFallerFazekasFleuryFroleyksPollitt-CAV24.bi
b
[16]: https://cca.informatik.uni-freiburg.de/papers/BiereFallerFazekasFleuryFroleyksPollitt-CAV24-Sp
ringer.pdf
[17]: https://zenodo.org/records/10943125
[18]: https://github.com/arminbiere/cadical
[19]: https://doi.org/10.1007/978-3-031-37703-7
