[MiniSat logo]by Niklas Eén, Niklas Sörensson
[Main][1] [MiniSat][2] [MiniSat+][3] [SatELite][4] [Papers][5] [Authors][6] [Links][7]

# Introduction

[[Trophies]][8] MiniSat is a minimalistic, open-source SAT solver, developed to help researchers and
developers alike to get started on SAT. It is released under the MIT licence, and is currently used
in a number of projects (see "Links"). On this page you will find binaries, sources, documentation
and projects related to MiniSat, including the Pseudo-boolean solver MiniSat+ and the CNF
minimizer/preprocessor SatELite. Together with SatELite, MiniSat was recently awarded in the three
[industrial][9] categories and one of the "crafted" categories of the SAT 2005 competition (see
picture).

Some key features of MiniSat:

* Easy to modify. MiniSat is small and well-documented, and possibly also well-designed, making it
  an ideal starting point for adapting SAT based techniques to domain specific problems.
* Highly efficient. Winning all the industrial categories of the SAT 2005 competition, MiniSat is a
  good starting point both for future research in SAT, and for applications using SAT.
* Designed for integration. MiniSat supports incremental SAT and has mechanisms for adding
  non-clausal constraints. By virtue of being easy to modify, it is a good choice for integrating as
  a backend to another tool, such as a model checker or a more generic constraint solver.

We would like to start a community to help distribute knowledge about how to use and modify MiniSat.
Any questions, comments, bugreports, bugfixes etc. should be directed to
[minisat@googlegroups.com][10]. The archives can be browsed [ here][11]. The source code repository
for MiniSat 2.2 can be found at [github][12].

— Niklas & Niklas

# News

From newest to oldest...

* A [paper][13] on how to compute localization abstractions using the incremental interface of
  MiniSat. View this as an example of a non-trivial incremental SAT application. A key feature of
  the algorithm is the use of *analyzeFinal()* to get the subset of assumptions used in the UNSAT
  proof.
* Finally! A new release of MiniSat, [downloads/minisat-2.2.0.tar.gz][14]. For more info, see the
  [release notes][15].
* Paper on [Cut-sweeping][16] added, a light-weight alternative to SAT-sweeping useful for
  preprocessing AIGs before applying SAT.
* Added slides for invited talk given by Niklas Een at FMCAD
  ([Linux][17]/[Cygwin][18]/[Windows][19])
* Paper on [efficient CNF generation][20] added ([slides][21] also available).
* Bug-fix in MiniSat+ for trivially unsatisfiable instances.
* MiniSat v.2.0 has been released.
* Bug-fix to the proof-logging version of MiniSat (thanks to Georg Weißenbacher for reporting the
  problem!).
* Bug-fix to the C-version of MiniSat.
* Paper on MiniSat+ added.
* Removed version 1.13 of MiniSat. It was an intermediate version that apperently was buggy.
* Patch for Visual Studio users added.
* Buggy v1.13 Cygwin binary replaced with working v1.14 binary. The bug did not affect the Linux
  version.
* Fixed the output of MiniSat+ so that it is flushed properly when stdout is redirected to a file.
* Fixed a silly bug in MiniSat that caused it to crash on the empty SAT problem if a random variable
  was picked by the branching heuristic.
* Added slides for the invited talk given by Niklas Sörensson at ESCAR (under Papers).
* A C version of MiniSat v1.14 released.
* The cleaned up version v1.14 of the competing solver released, now including proof logging.
* Source code for MiniSat v1.13 and SatELite 1.0, competing in SAT 2005 released.
* The PB solver MiniSat+ v0.99 released.
* MiniSat and SatELite won the industrial categories of the SAT 2005 competition.
* All MiniSat related information finally collected into a web page.

[1]: Main.html
[2]: MiniSat.html
[3]: MiniSat+.html
[4]: SatELite.html
[5]: Papers.html
[6]: Authors.html
[7]: Links.html
[8]: trophies-big.jpg
[9]: 2nd_stage_all.png
[10]: mailto:minisat@googlegroups.com
[11]: http://groups.google.com/group/minisat
[12]: http://github.com/niklasso/minisat
[13]: http://arxiv.org/abs/1008.2021
[14]: downloads/minisat-2.2.0.tar.gz
[15]: downloads/ReleaseNotes-2.2.0.txt
[16]: downloads/CutSweeping.pdf
[17]: downloads/PracticalSAT_linux.zip
[18]: downloads/PracticalSAT_cygwin.zip
[19]: downloads/PracticalSAT_windows.exe
[20]: downloads/synth_in_sat.pdf
[21]: downloads/SAT07-Een.pdf
