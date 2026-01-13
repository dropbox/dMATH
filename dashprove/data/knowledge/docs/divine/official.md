# synopsis

`divine verify program.c [program arguments]
`

# download

The most recent available release (as a source tarball) is *[divine-4.4.4.tar.gz][1]*. Binaries,
nightly snapshots, and older versions are *[also available][2]*.

# description

The `divine` toolset (to be written)…
You can read a lot more about `divine` in our *[academic papers][3]* but also on our *[blog][4]*.

# what's new

We are currently working on version 5, which will bring considerable internal changes, support for
LLVM 13+, streamlined build, and many improvements in performance and reliability. The timeline is
not yet finalized, but preview releases are expected in 2022. Stay tuned.
Previously: [old news][5].

# see also

[valgrind][6](1), [asan][7](7), [klee][8](1), [symbiotic][9](1), …

# history

The history of `divine` dates back to circa *2000*. In its initial incarnation, it was an
explicit-state *distributed model checker* with a dedicated, non-embedded domain-specific language:
think SPIN, but in distributed memory (i.e. using resources of an entire compute cluster to tackle
larger problems than a single computer could handle). The name stood for ‘distributed verification
environment’ (it is not a coincidence that it was created in paradise: a ‘parallel and distributed
systems laboratory’).
By *2007*, the focus was shifting towards *shared-memory parallelism*, reflecting the trends in
hardware development. In the following years, there were brief detours into GPU-based acceleration
to speed up computation and into using external memories (hard drives, flash storage) to tackle
state spaces that would not fit into RAM of contemporary machines.
In *2011*, generating state spaces of *LLVM bitcode* programs became our dominant research
direction. Even though distributed computation was still supported at this time, it was clearly not
a priority, and was entirely removed by *2017* in the 4.0 release, along with support for half a
dozen input languages. Model checking of C and C++ programs, via LLVM, was very much the main focus
at this point. Verification of liveness properties, specified using LTL, got likewise sidelined – it
was first dropped early in the 4.x series, though unlike support for distributed memory, limited LTL
support was soon re-introduced.
Around *2016*, automatic *abstraction* bubbled up to the top of our priority list, and has remained
there since. Our method of choice was to implement abstraction as an LLVM → LLVM program
transformation, and by *2019*, we had a reasonably solid implementation. The semantics of abstract
domains were implemented in C++ and substituted into the program by *LART* (LLVM Abstraction and
Refinement Tool). Unfortunately, we have encountered a number of ‘impedance mismatches’ around LLVM,
and both the discovery of instructions that need to be substituted, as well as the substitution
itself, turned out to be fragile and complicated.
At the same time, a combination of factors (with the COVID-19 pandemic playing a prominent role)
essentially ground the development effort to a halt. At the moment (spring *2022*), we are slowly
picking up the pieces and the pace to chart a new course and restart development.

# academic use

When you refer to `divine` in an academic paper, we would appreciate if you could use the following
reference:
`@InProceedings{BBK+17,
  author =    {Zuzana Baranová and Jiří Barnat and Katarína Kejstová and
               Tadeáš Kučera and Henrich Lauko and Jan Mrázek and Petr Ročkai
               and Vladimír Štill},
  title =     {Model Checking of {C} and {C}++ with {DIVINE} 4},
  booktitle = {Automated Technology for Verification and Analysis},
  pages =     {201-207},
  volume =    10482,
  series =    {LNCS},
  year =      2017,
  publisher = {Springer},
  doi =       {10.1007/978-3-319-68167-2_14}
}
`

# authors

If you have comments or questions about `divine`, please send an email to `divine at fi.muni.cz`.

[1]: ./download/divine-4.4.4.tar.gz
[2]: ./get.html
[3]: ./papers.html
[4]: ./blog.html
[5]: previously.html
[6]: https://valgrind.org/
[7]: https://github.com/google/sanitizers/wiki/AddressSanitizer
[8]: https://klee.github.io
[9]: https://staticafi.github.io/symbiotic/
