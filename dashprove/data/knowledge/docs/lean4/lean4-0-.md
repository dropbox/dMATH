# Learn Lean

Lean is a functional programming language and theorem prover built for
formalizing math and for formal verification, but is flexible enough for general
coding. If you’re a beginner, we recommend the Natural Number Game. If you feel
ready to dive deeper, there are great textbooks, tutorials and interactive games
to be found on this page.

[API Reference][1][Install][2][Core Docs][3]

## Core documentation

[

**Functional Programming in Lean (FPIL)** is the main resource for programmers
who want to learn Lean. It assumes a background in programming, but no prior
knowledge of functional programming is needed.

READ NOW
][4][

**Theorem Proving in Lean (TPIL)** is designed to teach you to develop and
verify proofs in Lean and covers dependent type theory, automated proof methods,
and Lean-specific features for interactive theorem proving.

READ NOW
][5][

**Mathematics in Lean (MIL)** is the main resource for mathematicians who want
to learn mathematical formalization through interactive, tactic-based theorem
proving using Lean's Mathlib library.

READ NOW
][6][

**The Lean Language Reference** is a comprehensive, precise description of Lean:
a reference work in which all aspects of Lean are clearly specified, and
demonstrated through succinct examples.

READ NOW
][7]

## Further reading

[

**Mathlib API Reference** Includes reference information for Lean core, the Lean
standard library, Mathlib, and other critical Lean packages.

READ NOW
][8][

**The Hitchhiker's Guide to Logical Verification** Originally designed as a
companion text for a graduate-level course on interactive theorem proving at
Vrije Universiteit Amsterdam.

READ NOW
][9][

**Logic and Proof** A textbook teaching the basics of classical logic, such as
propositional logic, first order logic, natural deduction and axiomatic
reasoning, using Lean.

READ NOW
][10][

**The Mechanics of Proof** Originally written as a companion text to the course
Math2001 at Fordham University, it teaches the basics of mathematical reasoning
to students of mathematics using Lean.

READ NOW
][11]

## Interactive Games and Tutorials

[

**The Natural Number Game** A gamified introduction to mathematical proof that
introduces Lean 4 concepts through a purpose-built Lean 4 dialect.

PLAY NOW
][12][

**The Lean Game Server** A collection of games similar to the Natural Number
Game.

PLAY NOW
][13]

## Tools for working with Lean

[

**Lean 4 VS Code Extension Manual** Describes how to interact with Lean 4 using
the VS Code extension.

READ NOW
][14][

**Semantic Highlighting** Configuring Lean's semantic highlighing (enabled in
VSCode by selecting "Editor > Semantic Highlighting"

READ NOW
][15][

**LaTeX** Best practices for highlighting Lean code in LaTeX documents

READ NOW
][16]

## Resources

[

**Loogle!** A Lean and Mathlib search tool for finding definitions and lemmas,
available on the web, via CLI or through IDE extensions.

OPEN
][17][

**LeanExplore** A natural language search engine for Lean declarations, indexing
commonly used Lean libraries.

OPEN
][18][

**LeanSearch** A Mathlib search engine for finding tactics and theorems via
natural language queries.

OPEN
][19][

**LeanDojo** A tool for data extraction and interacting with Lean
programmatically.

OPEN
][20][

**REPL** An interactive Read-Eval-Print Loop (REPL) for Lean intended for
machine-to-machine interaction and AI applications.

OPEN
][21][

**Pantograph** A machine-to-machine interaction system that provides interfaces
to execute proofs, construct expressions, and examine the symbol list of a Lean
project for machine learning.

OPEN
][22][

**Lean4Web** A web-based version of Lean that allows you to run Lean code
directly in your browser.

OPEN
][23]

## How to cite Lean

### Lean 4

To cite Lean 4 please reference [The Lean 4 Theorem Prover and Programming
Language][24] published in CADE-28 - The 28th International Conference on
Automated Deduction. [PDF][25]

#### Bibtex

@inproceedings{10.1007/978-3-030-79876-5_37,
  title = {The Lean 4 Theorem Prover and Programming Language},
  author = {Moura, Leonardo de and Ullrich, Sebastian},
  year = {2021},
  isbn = {978-3-030-79875-8},
  publisher = {Springer-Verlag},
  address = {Berlin, Heidelberg},
  url = {https://doi.org/10.1007/978-3-030-79876-5_37},
  doi = {10.1007/978-3-030-79876-5_37},
  abstract = {Lean 4 is a reimplementation of the Lean interactive theorem prove
r (ITP) in Lean itself. It addresses many shortcomings of the previous versions 
and contains many new features. Lean 4 is fully extensible: users can modify and
 extend the parser, elaborator, tactics, decision procedures, pretty printer, an
d code generator. The new system has a hygienic macro system custom-built for IT
Ps. It contains a new typeclass resolution procedure based on tabled resolution,
 addressing significant performance problems reported by the growing user base. 
Lean 4 is also an efficient functional programming language based on a novel pro
gramming paradigm called functional but in-place. Efficient code generation is c
rucial for Lean users because many write custom proof automation procedures in L
ean itself.},
  booktitle = {Automated Deduction – CADE 28: 28th International Conference on A
utomated Deduction, Virtual Event, July 12–15, 2021, Proceedings},
  pages = {625–635},
  numpages = {11}
}

### Lean 3

To cite Lean 3 please reference [The Lean Theorem Prover (System
Description)][26] published in *Lecture Notes in Computer Science*. [PDF][27].

#### Bibtex

@inproceedings{inproceedings,
  author = {de Moura, Leonardo and Kong, Soonho and Doorn, Floris and Raumer, Ja
kob},
  year = {2015},
  month = {08},
  pages = {378-388},
  title = {The Lean Theorem Prover (System Description)},
  volume = {9195},
  isbn = {978-3-319-21400-9},
  doi = {10.1007/978-3-319-21401-6_26}
}

[1]: /doc/api/
[2]: /install
[3]: /learn#core-documentation
[4]: https://lean-lang.org/functional_programming_in_lean/
[5]: https://lean-lang.org/theorem_proving_in_lean4/
[6]: https://leanprover-community.github.io/mathematics_in_lean/
[7]: https://lean-lang.org/doc/reference/latest/
[8]: https://leanprover-community.github.io/mathlib4_docs/
[9]: https://github.com/lean-forward/logical_verification_2025
[10]: https://leanprover.github.io/logic_and_proof/
[11]: https://hrmacbeth.github.io/math2001/
[12]: https://adam.math.hhu.de/#/g/leanprover-community/NNG4
[13]: https://adam.math.hhu.de/
[14]: https://github.com/leanprover/vscode-lean4/blob/master/vscode-lean4/manual
/manual.md
[15]: https://lean-lang.org/documentation/semantic-tokens/
[16]: https://lean-lang.org/documentation/latex-syntax-highlighting/
[17]: https://loogle.lean-lang.org/
[18]: https://www.leanexplore.com/
[19]: https://leansearch.net/
[20]: https://leandojo.org/
[21]: https://github.com/leanprover-community/repl
[22]: https://github.com/leanprover/Pantograph
[23]: https://github.com/leanprover-community/lean4web
[24]: https://dl.acm.org/doi/10.1007/978-3-030-79876-5_37
[25]: https://lean-lang.org/papers/lean4.pdf
[26]: https://www.researchgate.net/publication/300636103_The_Lean_Theorem_Prover
_System_Description
[27]: https://lean-lang.org/papers/system.pdf
