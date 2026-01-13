[[License: MIT]][1]

# The Kissat SAT Solver

Kissat is a "keep it simple and clean bare metal SAT solver" written in C. It is a port of CaDiCaL
back to C with improved data structures, better scheduling of inprocessing and optimized algorithms
and implementation.

Coincidentally "kissat" also means "cats" in Finnish.

Run `./configure && make test` to configure, build and test in `build`.

Binaries are provided with each major [release][2].

You can get more information about Kissat in the last solver description for the SAT Competition
2024:

[Armin Biere][3], [Tobias Faller][4], Katalin Fazekas, [Mathias Fleury][5], Nils Froleyks and
[Florian Pollitt][6]
[CaDiCaL, Gimsatul, IsaSAT and Kissat Entering the SAT Competition 2024][7]
*Proc. SAT Competition 2024: Solver, Benchmark and Proof Checker Descriptions*
Marijn Heule, Markus Iser, Matti Järvisalo, Martin Suda (editors)
Department of Computer Science Report Series B
vol. B-2024-1, pages 8-10, University of Helsinki 2024
[ [paper][8] | [bibtex][9] | [cadical][10] | [kissat][11] | [gimsatul][12] | [medals][13] ]

See [NEWS.md][14] for feature updates.

[1]: https://opensource.org/licenses/MIT
[2]: https://github.com/arminbiere/kissat/releases/
[3]: https://cca.informatik.uni-freiburg.de/biere/index.html#publications
[4]: /arminbiere/kissat/blob/master/biere/index.html
[5]: https://cca.informatik.uni-freiburg.de/fleury/index.html
[6]: https://cca.informatik.uni-freiburg.de/pollittf.html
[7]: https://cca.informatik.uni-freiburg.de/papers/BiereFallerFazekasFleuryFroleyksPollitt-SAT-Compe
tition-2024-solvers.pdf
[8]: https://cca.informatik.uni-freiburg.de/papers/BiereFallerFazekasFleuryFroleyksPollitt-SAT-Compe
tition-2024-solvers.pdf
[9]: https://cca.informatik.uni-freiburg.de/papers/BiereFallerFazekasFleuryFroleyksPollitt-SAT-Compe
tition-2024-solvers.bib
[10]: https://github.com/arminbiere/cadical
[11]: https://github.com/arminbiere/kissat
[12]: https://github.com/arminbiere/gimsatul
[13]: https://cca.informatik.uni-freiburg.de/sat24medals
[14]: /arminbiere/kissat/blob/master/NEWS.md
