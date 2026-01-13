─────────┬─┬────────────────┬─┬──────────────────┬─┬──────────────┬─┬────────────┬─┬─────────────
[Home][1]│•│[Description][2]│•│[Documentation][3]│•│[Downloads][4]│•│[Support][5]│•│[FM Tools][6]
─────────┴─┴────────────────┴─┴──────────────────┴─┴──────────────┴─┴────────────┴─┴─────────────

[[PVS]][7]

#### Download 7.1

* [ Linux Allegro][8]
* [ Mac OS X Allegro][9]
* [ Linux SBCL][10]
* [ Mac OS X SBCL][11]
* [Github sources][12]
* [All Downloads...][13]

#### [Install 8.0 Beta][14]

#### NASA Additions

* [NASA Library][15]
* [VSCode PVS Plugin][16]

#### Manuals

* [Release Notes][17]
* [System Guide][18]
* [Language Reference][19]
* [Prover Guide][20]
* [Datatypes][21]
* [Theory Interpretations][22]

#### Patches

* [List of available patches][23]
* [How to install patches][24]

# PVS Documentation

PVS has had many developments since it was first officially introduced in 1996, and the manuals have
not kept up. For the most part, the system is backward-compatible, so the documentation is still
accurate. However, many important new features of PVS are only mentioned in [the Release Notes][25].

If you're new to PVS, we suggest you start with the [the system guide tour][26]. This will give you
a quick feel for the system and it's basic capabilities. After that it depends on your goals and
interests.

There is extensive PVS documentation, divided below into sections:

* [PVS Release Notes][27]
* [PVS System][28]
* [PVS Language][29]
* [PVS Prover][30]
* [PVS Code Generation and Evaluation][31]
* [Specialized Tools][32]
* [Tutorials][33]
* [Examples][34]

### PVS Release Notes

[Release Notes][35] cover versions of PVS from 3.0 to 7.1, they describe new features, bug fixes,
etc. It is worth skimming, as many useful new features of PVS are only described in the release
notes.

### PVS System

[ System Guide][36] has a tour of the most used features of PVS, along with the commands, files, and
operating system interaction. PVS was originally written with an Emacs GUI, and this is described
here.

[vscode-pvs][37] is a newer GUI for PVS, based on Microsoft Visual Studio code, that is being
actively developed by NASA as an alternative to Emacs. Both GUIs will be supported in PVS in the
future, so choose either or both.

[PVS: A Prototype Verification System][38] by Sam Owre, Natarajan Shankar, and John Rushby, from
CADE 11, Saratoga Springs, NY, June 1992. One of the earliest descriptions of the PVS system.

[PVS: An Experience Report][39] by Sam Owre, John Rushby, N. Shankar and David Stringer-Calvert,
from Applied Formal Methods---FM-Trends 98, Boppard, Germany, October 1998. A brief description of
PVS and a few of its applications.

### PVS Language

[ Language Reference][40] The PVS specification language is based on higher-order logic, with
predicate subtypes. Some parts of the language are described in separate manuals.

[ Datatypes][41] This describes the datatype mechanism in PVS, used to define recursive types, such
as lists and trees.

[ Theory Interpretations][42] This describes how to map uninterpreted types and constants from one
theory into another. Useful for checking that your theory is consistent, and for evaluating
instances of it.

[ PVS Semantics][43] This gives the formal semantics for PVS.

[Subtypes for Specifications: Predicate Subtyping in PVS][44], by John Rushby, Sam Owre, and N.
Shankar, Appears in IEEE Transactions on Software Engineering, Volume 24, Number 9. September, 1998.
Describes how predicate subtyping works in PVS, with many illustrative examples.

[Principles and Pragmatics of Subtyping in PVS][45]. Presented at the 14th International Workshop on
Algebraic Development Techniques [WADT'99][46], September, 1999, Toulouse, France. Further
discussion of predicate subtyping in PVS.

[Integration in PVS: Tables, Types, and Model Checking][47], by Sam Owre, John Rushby, and N.
Shankar, Appears in Tools and Algorithms for the Construction and Analysis of Systems TACAS '97.
Enschede, The Netherlands. April, 1997. The title says what it's about.

### PVS Prover

[ Prover Guide][48] Describes the proof checker in detail, though the release notes should be
consulted for newer commands as well as some new strategy writing information.

[ ProofLite][49] Allows batch proving and proof scripting, enabling a semi-literate proving style,
where specification and proof scripts can be kept in the same file. This is intended for non-expert
users of PVS.

[ Real Automation in the Field][50] Describes a built-in package of PVS strategies that supports
automation of non-linear arithmetic. In particular, it includes a semi-decision procedure for the
field of real numbers and a strategy for cancellation of common terms.

[ Manip Guide][51] Provides an approach to tactic-based proving for improved interactive deduction.
The built-in Manip package includes a set of strategies (tactics) and support functions. Although it
was designed originally to reduce the tedium of low-level arithmetic manipulation, many of its
features are suitable as general-purpose prover utilities. Besides strategies aimed at algebraic
simplification of real-valued expressions, Manip includes term-access techniques applicable in
arbitrary settings.

[Writing PVS Proof Strategies][52] ([Slides][53]) by Sam Owre and N. Shankar, from [STRATA
2003][54], Rome, Italy, September 2003. Gives details of the PVS strategies mechanism, as well as
many pragmatic examples.

### PVS Code Generation and Evaluation

Much of the PVS language is executable, and this is taken advantage of by translations to Lisp, C,
and Clean. The Lisp translation was the first, and is also referred to as the *Ground Evaluator*.
See [the ground evaluator][55] for more details.

[PVSio Animation][56] PVSio extends the ground evaluator with a predefined library of imperative
programming language features such as side effects, unbounded loops, input/output operations,
floating point arithmetic, exception handling, pretty printing, and parsing. The PVSio input/output
library is implemented via semantic attachments.

[ Evaluating, Testing, and Animating PVS Specifications][57] by Judy Crow, Sam Owre, John Rushby, N.
Shankar, and Dave Stringer-Calvert, CSL Technical Report, March 30, 2001. Describes many examples
using the ground evaluator.

[Random Testing in PVS][58] ([Slides][59]) by Sam Owre, from [AFM 2006][60], Seattle, WA, August
2006. Similar to QuickCheck for Haskell, random testing in PVS allows one to look for
counterexamples. This can help, for example, to refine hypotheses, before wasting time trying to
prove an invalid formula.

### PVS Specialized Tools

The following are a part of PVS, but of rather specialized interest.

[Predicate Abstraction in PVS][61]

[WS1S][62] - PVS with Mona

## PVS Tutorials

In general, the specs for these tutorials are included in the latest PVS installations, under the
Examples directory.

[ The standard introductory tutorial to PVS (WIFT)][63] [[ Examples/wift-tutorial/][64]]: is a good
place to start. It provides an introductory example and a tutorial. It is intended to provide enough
information to get you started using PVS, and to help you appreciate the capabilities of the system
and the purposes for which it is suitable.

[A less elementary version of Ricky Butler's "Elementary Tutorial"][65] [[
Examples/elementary-tutorial/][66]] introduces some of the more powerful strategies provided by the
PVS theorem prover. It consists of two parts: the first extends a previous tutorial by Ricky Butler
demonstrating how his proofs may be performed in a more automated manner; the second uses the
*unwinding theorem* from the noninterference formulation of security to introduce theorem-proving
strategies for induction. This tutorial also shows how specifications and proofs may be better
presented using the LaTeX and PostScript generating facilities of PVS.

[A tutorial on some advanced features of PVS][67] [[ Examples/fme96/][68]] given at FME 96. This
covers predicate subtypes and judgements in detail.

[Tutorial on Specification, Proof Checking, and Model Checking for Protocols and Distributed Systems
with PVS][69] [[ Examples/forte97-tutorial/][70]] given at FORTE/PSTV 97. Analyzes a simple
cache-coherence protocol using a combination of theorem proving and model checking.

[PVS Tutorial, FM99][71] [[ Examples/fm99/][72]] given at FM 99. A simple tutorial illustrating
recursion, higher-order features, and abstract datatypes of PVS.

## PVS Examples

There are many sources of PVS Examples. The PVS prelude offers foundational theories, and can easily
be viewed using M-x view-prelude-theory. Another useful source of examples is in the [PVS
Library][73] maintained by NASA.

[ Hardware Verification Using PVS][74] [[ Examples/HWVbookchap/][75]] includes examples that appear
in the chapter Hardware Verification Using PVS in the book Formal Hardware Verification Methods and
Systems in Comparison, Edited by Thomas Kropf, Springer, LNCS 1287. The examples are: Arbiter,
Blackjack, Detect110, Fir_filter5, Pipeline, Singlepulser, and Tamarack.

[Tabular specifications and model-checking SCR transition relations][76] [[
Examples/pvs-tables/][77]] describes PVS's capabilities for representing tabular specifications of
the kind advocated by Parnas and others, and shows how PVS's Type Correctness Conditions (TCCs) are
used to ensure certain well-formedness properties.

[Byzantine Agreement][78] [[ Examples/byzantine/][79]] describes the formal specification and
verification of an algorithm for Interactive Consistency based on the Oral Messages algorithm for
Byzantine Agreement.

[Ag in PVS][80] [[ Examples/AgExample/][81]] Ag is a specification language presented as a syntactic
sugaring of the First-Order Dynamic Logic of Fork Algebras. We show how the higher-order logic
theorem prover PVS can be used to build a semantic framework for reasoning about specifications
written in Ag by encoding the semantics within the PVS language. We then illustrate how this
semantic embedding can be used by means of a case study of a cache system specification.

Last modified: | Webmaster: [Sam Owre][82] • [N. Shankar][83]

[1]: index.html
[2]: description.html
[3]: documentation.html
[4]: downloads.html
[5]: support.html
[6]: http://fm.csl.sri.com
[7]: #
[8]: license.html?tgzfile=pvs7.1.0-ix86_64-Linux-allegro.tgz
[9]: license.html?tgzfile=pvs7.1.0-ix86-MacOSX-allegro.tgz
[10]: downloads/pvs7.1.0-ix86_64-Linux-sbclisp.tgz
[11]: downloads/pvs7.1.0-ix86-MacOSX-sbclisp.tgz
[12]: https://github.com/SRI-CSL/PVS
[13]: downloads.html
[14]: downloads.html
[15]: https://shemesh.larc.nasa.gov/fm/pvs/PVS-library
[16]: https://github.com/nasa/vscode-pvs
[17]: doc/pvs-release-notes.pdf
[18]: doc/pvs-system-guide.pdf
[19]: doc/pvs-language-reference.pdf
[20]: doc/pvs-prover-guide.pdf
[21]: doc/datatypes.pdf
[22]: doc/interpretations.pdf
[23]: pvs-patches/
[24]: pvs-patches/
[25]: doc/pvs-release-notes.pdf
[26]: doc/pvs-system-guide.pdf#Chapter2
[27]: #release-notes
[28]: #system
[29]: #language
[30]: #prover
[31]: #eval
[32]: #special
[33]: #tutorials
[34]: #examples
[35]: doc/pvs-release-notes.pdf
[36]: doc/pvs-system-guide.pdf
[37]: http://github.com/nasa/vscode-pvs
[38]: doc/cade92-pvs.ps
[39]: doc/fmtrends98-pvs.ps
[40]: doc/pvs-language-reference.pdf
[41]: doc/datatypes.pdf
[42]: doc/interpretations.pdf
[43]: doc/semantics.pdf
[44]: doc/tse98.pdf
[45]: doc/wadt99.ps
[46]: http://www-lsr.imag.fr/WADT99/
[47]: doc/tacas97.pdf
[48]: doc/pvs-prover-guide.pdf
[49]: doc/ProofLite-4.2.pdf
[50]: doc/extrategies.pdf
[51]: doc/manip-guide.pdf
[52]: doc/strata03.pdf
[53]: doc/strata03-slides.pdf
[54]: http://research.nianet.org/fm-at-nia/STRATA2003/
[55]: eval.html
[56]: https://shemesh.larc.nasa.gov/fm/pvs/PVSio/
[57]: doc/eval-report.pdf
[58]: doc/5-Owre.pdf
[59]: doc/Owre-randomtest.pdf
[60]: http://fm.csl.sri.com/AFM06/
[61]: abstract.html
[62]: ws1s.html
[63]: doc/wift-tutorial.pdf
[64]: https://github.com/SRI-CSL/PVS/tree/master/Examples/wift-tutorial/
[65]: doc/csl-95-10.ps
[66]: https://github.com/SRI-CSL/PVS/tree/master/Examples/elementary-tutorial/
[67]: doc/fme96-tutorial.ps
[68]: https://github.com/SRI-CSL/PVS/tree/master/Examples/fme96/
[69]: doc/forte97.ps
[70]: https://github.com/SRI-CSL/PVS/tree/master/Examples/forte97-tutorial/
[71]: doc/fm99tut.pdf
[72]: https://github.com/SRI-CSL/PVS/tree/master/Examples/fm99/
[73]: https://github.com/nasa/pvslib
[74]: doc/HWVerifinPVSdocument.pdf
[75]: https://github.com/SRI-CSL/PVS/tree/master/Examples/HWVbookchap/
[76]: doc/csl-95-12.ps
[77]: https://github.com/SRI-CSL/PVS/tree/master/Examples/pvs-tables/
[78]: doc/csl-92-1.ps
[79]: https://github.com/SRI-CSL/PVS/tree/master/Examples/byzantine/
[80]: doc/Ag-report.pdf
[81]: https://github.com/SRI-CSL/PVS/tree/master/Examples/AgExample/
[82]: mailto:owre@csl.sri.com
[83]: mailto:shankar@csl.sri.com
