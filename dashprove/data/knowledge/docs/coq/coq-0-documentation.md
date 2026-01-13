* [ Learn ][1]
* Overview Platform Docs Supporting tools Exercises Books Papers ▾
[ Overview ][2] [ Platform Docs ][3] [ Supporting tools ][4] [ Exercises ][5] [
Books ][6] [ Papers ][7]

# Explore the Rocq Documentation

You can explore the Rocq & Rocq Platform packages documentation, guides,
exercises, and books to enhance your knowledge.
Search inside the Rocq documentation
[

Install Rocq Platform 2025.01.0

][8] [

Rocq Reference Manual

][9] [

Rocq Corelib Theories

][10] [

Rocq OCaml API

][11] [

Rocq Stdlib Manual

][12] [

Rocq Stdlib Theories

][13]
[ Jump to Beginner Section ][14] [ Jump to Intermediate Section ][15] [ Jump to
Advanced Section ][16]

## FOUNDATIONS FOR BEGINNERS

GET STARTED

## Introduction To Rocq

Install Rocq and gain a high-level understanding of the language

* [ Installing the Rocq Prover ][17]
* [ A Tour of Rocq ][18]
Documentation

## The Rocq Prover

An in-depth explanation of language features and data structures from the
Standard Library

* [ Rocq Platform Docs ][19]
RECOMMENDED BOOK

## Software Foundations: Logical Foundations (Volume 1)

[
FREE
][20]
Beginner

For newcomers interested in programming languages, the Software Foundations
series is an excellent introduction to using the Rocq Prover for formal
verification.

[Read Online ][21]
[ See More Books ][22]
RECOMMENDED BOOK

## Mathematical Components

[
FREE
][23]
Beginner

For newcomers with a background in mathematics rather than computer science,
this book is an introduction to the Rocq Prover and to the SSReflect proof
language used in the Mathematical Components library. Accustomed Rocq users will
find a substantial account of the formalization style that made the Mathematical
Components library possible.

[Read Online ][24]
[ See More Books ][25]
RECOMMENDED BOOK

## Programs and Proofs: Mechanizing Mathematics with Dependent Types

[
FREE
][26]
Beginner

The primary audience of the manuscript are the readers with expertise in
software development and programming and knowledge of discrete mathematic
disciplines on the level of an undergraduate university program. The high-level
goal of the course is, therefore, to demonstrate how much the rigorous
mathematical reasoning and development of robust and intellectually manageable
programs have in common, and how understanding of common programming language
concepts provides a solid background for building mathematical abstractions and
proving theorems formally. The low-level goal of this course is to provide an
overview of the Rocq Prover, taken in its both incarnations: as an expressive
functional programming language with dependent types and as a proof assistant
providing support for mechanized interactive theorem proving.

[Read Online ][27]
[ See More Books ][28]
RECOMMENDED BOOK

## Coq'Art

[
][29]
Beginner

This textbook is a foundational book that introduces readers to the Rocq Prover
and its underlying formal system, the Calculus of Inductive Constructions (CIC).
The book provides a comprehensive guide to interactive theorem proving,
combining theory with practical applications. It covers key concepts such as
inductive types, recursive functions, and proof construction, while offering
numerous examples and case studies, including the formalization of algorithms
and mathematical proofs. Coq'Art serves as both an introduction for beginners
and a reference for experts, making it an essential resource for anyone working
with Rocq or exploring formal methods.

[Publisher's Book Website ][30] [Examples and Exercises ][31]
[ See More Books ][32]
PLATFORM TOOLS

## The Rocq Platform

[

### opam

OCaml's package manager for managing Rocq and libraries

Get Started
][33]
[Go to Platform Tools Documentation][34]

## EXPLORING THE INTERMEDIATE LEVEL

RECOMMENDED BOOK

## Software Foundations: Verified Functional Algorithms (Volume 3)

[
FREE
][35]
Intermediate

Verified Functional Algorithms shows how a variety of fundamental data
structures can be specified and mechanically verified.

[Read Online ][36]
[ See More Books ][37]
RECOMMENDED BOOK

## Formal Reasoning About Programs

[
FREE
][38]
Intermediate

Formal Reasoning About Programs (FRAP) is a book that provides a general
introduction to formal logical reasoning about the correctness of programs and
to using Rocq for this purpose. The main narrative of FRAP presents standard
program-proof ideas, without rigorous proofs. Matching Rocq code then show how
to make it rigorous. Interleaved with that narrative, there are also other
lectures' worth of material, for building up more practical background on Rocq
itself.

[Read Online ][39] [Source Code ][40]
[ See More Books ][41]
[
GUIDE

## Reference Manual

The reference manual is the authoritative source of documentation for the Rocq
Prover. It contains a changelog describing updates to Rocq, which we recommend
you read when you upgrade.

Take Me There
][42]

## ADVANCED LEVEL

RECOMMENDED BOOK

## Certified Programming with Dependent Types

[
FREE
][43]
Advanced

Certified Programming with Dependent Types (CPDT) is a hands-on guide to
advanced programming and proving using the Rocq Prover. The book focuses on
leveraging Roqc’s dependent type system to write certified programs, where
correctness is formally verified alongside implementation. It introduces key
concepts such as inductive types, type theory, proof automation, and advanced
tactics, providing a blend of theoretical foundations and practical examples.
CPDT is particularly known for its focus on proof engineering and building
large-scale verified systems efficiently. Aimed at readers with some prior
exposure to Rocq, it serves as an invaluable resource for those looking to
master the interplay between programming and formal verification.

[Read Online ][44]
[ See More Books ][45]
RECOMMENDED BOOK

## Computer Arithmetic and Formal Proofs: Verifying Floating-point Algorithms
## with the Coq System

[
][46]
Advanced

Floating-point arithmetic is ubiquitous in modern computing, as it is the tool
of choice to approximate real numbers. Due to its limited range and precision,
its use can become quite involved and potentially lead to numerous failures. One
way to greatly increase confidence in floating-point software is by
computer-assisted verification of its correctness proofs. This book provides a
comprehensive view of how to formally specify and verify tricky floating-point
algorithms with the Coq proof assistant. It describes the Flocq formalization of
floating-point arithmetic and some methods to automate theorem proofs. It then
presents the specification and verification of various algorithms, from
error-free transformations to a numerical scheme for a partial differential
equation. The examples cover not only mathematical algorithms but also C
programs as well as issues related to compilation.

[Publisher's Book Website ][47]
[ See More Books ][48]

### Papers

Aspiring towards greater understanding of Rocq and its underlying theory? Check
out papers written by Rocq researchers:
[
Correct and Complete Type Checking and Certified Erasure for Coq, in Coq
Coq is built around a well-delimited kernel that performs type checking for
definitions in a variant of the Calculus of Inductive Constructions (CIC).
Although the metatheory of CIC is very stable and reliable, the correctness of
its implementation in Coq is less clear. Indeed, implementing an efficient type
checker for CIC is a rather complex task, and many parts of the code rely on
implicit invariants which can easily be broken by further evolution of the code.
Therefore, on average, one critical bug has been found every year in Coq. This
paper presents the first implementation of a type checker for the kernel of Coq
(without the module system, template polymorphism and η-conversion), which is
proven sound and complete in Coq with respect to its formal specification. Note
that because of Gödel’s second incompleteness theorem, there is no hope to prove
completely the soundness of the specification of Coq inside Coq (in particular
strong normalization), but it is possible to prove the correctness and
completeness of the implementation assuming soundness of the specification, thus
moving from a trusted code base (TCB) to a trusted theory base (TTB) paradigm.
Our work is based on the MetaCoq project which provides meta-programming
facilities to work with terms and declarations at the level of the kernel. We
verify a relatively efficient type checker based on the specification of the
typing relation of the Polymorphic, Cumulative Calculus of Inductive
Constructions (PCUIC) at the basis of Coq. It is worth mentioning that during
the verification process, we have found a source of incompleteness in Coq’s
official type checker, which has then been fixed in Coq 8.14 thanks to our work.
In addition to the kernel implementation, another essential feature of Coq is
the so-called extraction mechanism: the production of executable code in
functional languages from Coq definitions. We present a verified version of this
subtle type and proof erasure step, therefore enabling the verified extraction
of a safe type checker for Coq in the future.
Matthieu Sozeau, Yannick Forster, Meven Lennon-Bertrand, Jakob Botsch Nielsen,
Nicolas Tabareau, Théo Winterhalter
Theory and Implementation of Rocq
JACM
type-checker
][49] [
The Coq Proof Assistant Version 8.20.0
Coq version 8.20 adds a new rewrite rule mechanism along with a few new
features, a host of improvements to the virtual machine, the notation system,
Ltac2 and the standard library.
Yves Bertot, Frédéric Besson, Ana Borges, Ali Caglayan, Tej Chajed, Cyril Cohen,
Pierre Corbineau, Pierre Courtieu, Andres Erbsen, Jim Fehrle, Julien Forest,
Emilio Jesús Gallego Arias, Gaëtan Gilbert, Georges Gonthier, Benjamin Grégoire,
Jason Gross, Hugo Herbelin, Vincent Laporte, Olivier Laurent, Assia Mahboubi,
Kenji Maillard, Erik Martin-Dorel, Guillaume Melquiond, Pierre-Marie Pédrot,
Clément Pit-Claudel, Pierre Roux, Kazuhiko Sakaguchi, Vincent Semeria, Michael
Soegtrop, Matthieu Sozeau, Arnaud Spiwack, Nicolas Tabareau, Enrico Tassi,
Laurent Théry, Anton Trunov, Xia Li-yao, Théo Zimmermann
Theory and Implementation of Rocq
Release
][50] [
Trocq: Proof Transfer for Free, With or Without Univalence
In interactive theorem proving, a range of different representations may be
available for a single mathematical concept, and some proofs may rely on several
representations. Without automated support such as proof transfer, theorems
available with different representations cannot be combined, without light to
major manual input from the user. Tools with such a purpose exist, but in proof
assistants based on dependent type theory, it still requires human effort to
prove transfer, whereas it is obvious and often left implicit on paper. This
paper presents Trocq, a new proof transfer framework, based on a generalization
of the univalent parametricity translation, thanks to a new formulation of type
equivalence. This translation takes care to avoid dependency on the axiom of
univalence for transfers in a delimited class of statements, and may be used
with relations that are not necessarily isomorphisms. We motivate and apply our
framework on a set of examples designed to show that it unifies several existing
proof transfer tools. The article also discusses an implementation of this
translation for the Coq proof assistant, in the Coq-Elpi metalanguage.
Cyril Cohen, Enzo Crance, Assia Mahboubi
Theory and Implementation of Rocq
ESOP
Proof Transfer
][51] [
Verified Extraction from Coq to OCaml
One of the central claims of fame of the Coq proof assistant is extraction,
i.e., the ability to obtain efficient programs in industrial programming
languages such as OCaml, Haskell, or Scheme from programs written in Coq's
expressive dependent type theory. Extraction is of great practical usefulness,
used crucially e.g., in the CompCert project. However, for such executables
obtained by extraction, the extraction process is part of the trusted code base
(TCB), as are Coq's kernel and the compiler used to compile the extracted code.
The extraction process contains intricate semantic transformation of programs
that rely on subtle operational features of both the source and target language.
Its code has also evolved since the last theoretical exposition in the seminal
PhD thesis of Pierre Letouzey. Furthermore, while the exact correctness
statements for the execution of extracted code are described clearly in academic
literature, the interoperability with unverified code has never been
investigated formally, and yet is used in virtually every project relying on
extraction. In this paper, we describe the development of a novel extraction
pipeline from Coq to OCaml, implemented and verified in Coq itself, with a clear
correctness theorem and guarantees for safe interoperability. We build our work
on the MetaCoq project, which aims at decreasing the TCB of Coq's kernel by
reimplementing it in Coq itself and proving it correct w.r.t. a formal
specification of Coq's type theory in Coq. Since OCaml does not have a formal
specification, we make use of the Malfunction project specifying the semantics
of the intermediate language of the OCaml compiler. Our work fills some gaps in
the literature and highlights important differences between the operational
semantics of Coq programs and their extracted variants. In particular, we focus
on the guarantees that can be provided for interoperability with unverified
code, identify guarantees that are infeasible to provide, and raise interesting
open question regarding semantic guarantees that could be provided. As central
result, we prove that extracted programs of first-order data type are correct
and can safely interoperate, whereas for higher-order programs already simple
interoperations can lead to incorrect behaviour and even outright segfaults.
Yannick Forster, Matthieu Sozeau, Nicolas Tabareau
Theory and Implementation of Rocq
PLDI
extraction
][52]
[ View all papers ][53]

[1]: /docs
[2]: /docs
[3]: /docs/platform-docs
[4]: /docs/tools
[5]: /exercises
[6]: /books
[7]: /papers
[8]: /install
[9]: /refman
[10]: /corelib
[11]: /api
[12]: /refman-stdlib
[13]: /stdlib
[14]: #beginner_section
[15]: #intermediate_section
[16]: #advanced_section
[17]: /install
[18]: /docs/tour-of-rocq
[19]: /docs/platform-docs
[20]: https://softwarefoundations.cis.upenn.edu/
[21]: https://softwarefoundations.cis.upenn.edu/
[22]: /books
[23]: https://math-comp.github.io/mcb/
[24]: https://math-comp.github.io/mcb/
[25]: /books
[26]: https://ilyasergey.net/pnp/
[27]: https://ilyasergey.net/pnp/
[28]: /books
[29]: https://doi.org/10.1007/978-3-662-07964-5
[30]: https://doi.org/10.1007/978-3-662-07964-5
[31]: https://rocq-community.org/coq-art
[32]: /books
[33]: https://opam.ocaml.org
[34]: /docs/platform-docs
[35]: https://softwarefoundations.cis.upenn.edu/
[36]: https://softwarefoundations.cis.upenn.edu/
[37]: /books
[38]: http://adam.chlipala.net/frap/
[39]: http://adam.chlipala.net/frap/
[40]: https://github.com/achlipala/frap
[41]: /books
[42]: /refman
[43]: http://adam.chlipala.net/cpdt/
[44]: http://adam.chlipala.net/cpdt/
[45]: /books
[46]: https://www.sciencedirect.com/book/9781785481123/computer-arithmetic-and-f
ormal-proofs
[47]: https://www.sciencedirect.com/book/9781785481123/computer-arithmetic-and-f
ormal-proofs
[48]: /books
[49]: https://dl.acm.org/doi/10.1145/3706056
[50]: https://zenodo.org/records/14542673
[51]: https://inria.hal.science/hal-04177913v5
[52]: https://dl.acm.org/doi/10.1145/3656379
[53]: /papers
