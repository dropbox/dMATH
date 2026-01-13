Toggle navigation [ [Alt-Ergo] Alt-Ergo by ][1][ [OCamlPro] ][2]

* [News][3]
* [About][4]
* [Try Online][5]
* [Install][6]
* [Documentation][7]
* [The Club][8]
* [Services][9]
* [Publications][10]
* [History][11]

# An SMT Solver for Software Verification

Alt-Ergo is an automatic prover of mathematical formulas used behind software verification tools
such as Frama-C, SPARK, Why3, Atelier-B and Caveat.

[Try Online][12] [Install][13] [Documentation][14] [The Alt-Ergo Users' Club][15]
[The DéCySif joint project][16] [Services][17] [Publications][18]

# Latest News

* May 20, 2025 Annual meeting of the Alt-Ergo users's club.
* May 6, 2025 Release of Alt-Ergo 2.6.2.
* Apr 15, 2025 Release of Alt-Ergo 2.6.1.
* Sep 24, 2024 Release of Alt-Ergo 2.6.0.
* Oct 18, 2023 Release of minor version 2.5.2.
* Sep 14, 2023 Release of minor version 2.5.1.
* Sep 6, 2023 Release of version 2.5.0.
* Apr 27, 2023 Release of minor version 2.4.3.
* Nov, 2022 Launch of the the DéCySif R&D joint project.
* Aug 01, 2022 Release of minor version 2.4.2.
* May 20, 2022 Version 2.3.3 becomes Alt-Ergo-Free 2.3.3.
* Jul 27, 2021 Release of minor version 2.4.1.
* Jan 22, 2021 Release of version 2.4.0.
* Aug 19, 2020 Release of minor version 2.3.3.
* Jun 2, 2020 Version 2.2.0 becomes Alt-Ergo-Free 2.2.0.
* Mar 25, 2020 Release of minor version 2.3.2.
* Feb 19, 2020 Release of minor version 2.3.1.
* Feb 14, 2020 Second annual meeting of the Alt-Ergo users's club.
* Jul 7, 2019 Participation to the SMT-COMP : [results][19].
* Feb 13, 2019 First annual meeting of the Alt-Ergo users's club.
* More news [Find out more][20].

# About

## What is Alt-Ergo ?

Alt-Ergo is an open-source automatic solver of mathematical formulas designed for program
verification. It is based on Satisfiability Modulo Theories (SMT). Solvers of this family have made
impressive advances and became very popular during the last decade. They are now used is various
domains such as hardware design, software verification and formal testing.

## What is Alt-Ergo Good for ?

Alt-Ergo is very successful for proving formulas generated in the context of deductive program
verification. It was originally designed and tuned to be used by the [Why platform][21]. Currently,
it is used as a back-end of different tools and in various settings, in particular via the [Why3
platform][22]. For instance, the [Frama-C suite][23] relies on it to prove formulas generated from C
code, and the [SPARK toolset][24] uses it to check formulas produced from Ada programs. In addition,
Alt-Ergo is used to prove formulas issued from [B modelizations][25] and from [cryptographic
protocols verification][26]. The figure given below shows the main tools that rely on Alt-Ergo to
prove the formulas they generate.

[Alt-Ergo Spider Web]


You are using Alt-Ergo in another context/tool not cited above ? [Let us know][27] !

## Under the Hood

Alt-Ergo's native input language is a polymorphic first-order logic "*à la ML*" modulo theories.
This logic is very suitable for expressing formulas generated in the context of program
verification. It also provides a support of the SMT-LIB v2 standard input format. Currently,
Alt-Ergo is capable of reasoning in the combination of the following built-in theories:

* the free theory of equality with uninterpreted symbols,
* linear arithmetic over integers and rationals,
* fragments of non-linear arithmetic,
* polymorphic functional arrays with extensionality,
* enumerated datatypes,
* record datatypes,
* associative and commutative (AC) symbols,
* fixed-size bit-vectors with concatenation and extraction operators.


## Origins

Alt-Ergo results from academic researches conducted conjointly at [Laboratoire de Recherche en
Informatique][28], [Inria Saclay Ile-de-France][29] and [CNRS][30] since 2006. Publications and
theoretical foundations are available on its [academic web page][31]. Since September 2013, Alt-Ergo
is maintained and distributed by the [OCamlPro][32] company. Academic researches are now conducted
in collaboration with the VALS team of LRI.

[[Université Paris-Sud]][33] [[LRI]][34] [[INRIA Saclay Ile-de-France]][35] [[CNRS]][36]
[ See publications on Alt-Ergo back then at lri.fr, now Laboratoire Méthodes Formelles (CNRS-Inria).
][37]


## The DéCySif joint research project

DéCySif is a regional R&D project that aims at improving the safety and security of critical systems
using formal verification tools. The project is funded by an i-Démo call and the Ile-de-France
French Region, gathering 4 partners: AdaCore, Inria, OCamlPro and TrustInSoft.

[[Le projet DéCySif]][38]
[ See the DéCySif project's website ][39]

# Online Version

Try-Alt-Ergo is a Javascript version of Alt-Ergo that runs directly in your browser. You don't need
to install anything to start using it.

[New Try-Alt-Ergo][40] [Blogpost][41] [Alt-Ergo Syntax Documentation][42]


The Javascript version is also used as a backend prover in [TryWhy3][43].

## Old Version

[Try-Alt-Ergo][44] [Small Tutorial][45]

# Releases

Alt-Ergo is released either under *OCamlPro Non-Commercial License* (`alt-ergo` on Opam), either
under an *Open-Source License* with a few years' delay (`alt-ergo-free` on Opam).

## Latest Release

The latest release of Alt-Ergo is version 2.6.0. It was released in September 24, 2024. It is
available under the terms of the following [license][46].

Please, follow the links below to download Alt-Ergo, to report a bug, or to ask a question. You may
also want to read the [CHANGES][47] or see our [documentation][48].

### Sources & Binaries

(follow instructions in [here][49] to install Alt-Ergo)

[github.com/OCamlPro/alt-ergo/tree/v2.6.0][50]

[alt-ergo-v2.6.0.tar.gz][51]

### Miscellaneous

[Commercial Support][52]

[Bugs tracker][53]


### OPAM package(s)

$ opam install [alt-ergo][54]




## Latest Free Release

Alt-Ergo-Free version 2.3.3 was released in May 20, 2022. It is based on Alt-Ergo version 2.3.3, and
is available under the terms of the Apache Software License version 2.0.

### Sources & Binaries

(follow instructions in INSTALL.md)
[github.com/OCamlPro/alt-ergo/tree/2.3.3-free][55]
[alt-ergo-free-2.3.3.tar.gz][56]

### OPAM package(s)

$ opam install [alt-ergo-free.2.3.3][57]




## Releases History

────────────────────────────────────────┬──────────────────┬───────────────────────────
version 2.6.0 released                  │Sep 24, 2024      │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.5.2 released                  │Oct 18, 2023      │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.5.1 released                  │Sep 14, 2023      │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.5.0 released                  │Sep 6, 2023       │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.4.3 released                  │Apr 27, 2023      │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.4.2 released                  │Aug 01, 2022      │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.4.1 released                  │July 27, 2021     │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
                                        │July 27, 2021     │free version 2.3.0 released
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.4.0 released                  │January 21, 2021  │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.3.3 released                  │August 19, 2020   │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
                                        │June 2, 2020      │free version 2.2.0 released
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.3.2 released                  │March 25, 2020    │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.3.1 released                  │February 19, 2020 │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
                                        │February 13, 2019 │free version 2.0.0 released
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.3.0 released                  │February 11, 2019 │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.2.0 released                  │April 21, 2018    │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.1.0 released                  │March 14, 2018    │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 2.0.0 released                  │November 14, 2017 │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
version 1.30 released                   │November 21, 2016 │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
public release 1.01                     │February 16, 2016 │                           
(based on v. 1.00)                      │                  │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
                                        │February 09, 2016 │private release 1.20       
────────────────────────────────────────┼──────────────────┼───────────────────────────
                                        │October 19, 2015  │private release 1.10       
────────────────────────────────────────┼──────────────────┼───────────────────────────
                                        │January 29, 2015  │private release 1.00       
────────────────────────────────────────┼──────────────────┼───────────────────────────
public release 0.99.1                   │December 30, 2014 │                           
(based on v. 0.99)                      │                  │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
                                        │January 01, 2014  │private release 0.99       
────────────────────────────────────────┼──────────────────┼───────────────────────────
public release 0.95.2                   │September 20, 2013│                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
                                        │Alt-Ergo@OCamlPro │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
public release 0.95.1                   │March 05, 2013    │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
public release 0.95                     │January 11, 2013  │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
public release 0.94                     │December 02, 2011 │                           
────────────────────────────────────────┼──────────────────┼───────────────────────────
                                        │. . .             │                           
────────────────────────────────────────┴──────────────────┴───────────────────────────

# The Alt-Ergo Users' Club

## What is the Alt-Ergo Users' Club?

The Alt-Ergo Users' Club was launched in 2019, as a way for the Alt-Ergo team to get closer to their
users, collect their needs, integrate them in the Alt-Ergo roadmap, and ensure sustainable funding
for this project's long-term development.

We are proud to thank the members of the Club for their support: AdaCore, CEA List, Thalès, and
MERCE (Mitsubishi Electric R&D Centre Europe), as well as the Why3 team.

[adacore logo] [thales logo] [mitsubishi logo] [cea list logo]

### Roadmap of the 2.6.0 release

* Support for optimization in models of LIA and LRA theories;
* Ground model checking;
* Support for the float theory of the SMT-LIB standard;
* Improve constants propagation for partially interpreted functions.

### Join the Club

* Custom licensing
* Dedicated support
* Priority given to your needs
[ Join The Club! ][58]

# Services

## Alt-Ergo @ OCamlPro

OCamlPro is investing a lot of time to develop and maintain the Alt-Ergo theorem prover. The aims of
this effort are, among others:

* delivering worldwide and academia users an efficient open source SMT solver for software
  verification within a basic support to learn how to use it;
* providing a quality industrial support for developers and industrial users in order to get the
  best performances from the solver.


Our clients will have access to the sources of the latest private releases. They will also benefit
from our services such as extended developer/production support, dev-on-demand, and re-licensing.
Feel free to [contact us][59] for more details.

### World-Wide Users

* sources of latest releases
* mailing list
* bugs tracker
* basic support

### Industrial Support

* dedicated releases
* developers discussion list
* developers / production support
* re-licensing
* front-line interaction with retailers
* freezing / maintaining versions
* qualification kit
* dev-on-demand



## Formal Methods

You have a problem related to formal methods ? You don't know if Alt-Ergo is suitable for your
situation ? [We can help you][60] to determine the better technology to use for your need.

You are involved in a promising R&D project and you are looking for partners in the field of formal
methods ? Our experienced R&D engineers will be happy to contribute to the success of your project.
Feel free to [contact us][61] to see how we can collaborate.

# History

* Nov, 2022 Launch of the DéCySif Project.
* Jan 21, 2021 Release of version 2.4.0.
* Feb 13, 2019 First annual meeting of the Alt-Ergo users's club.
* Feb 13, 2019 Version 2.0.0 becomes Alt-Ergo-Free 2.0.0.
* Feb 11, 2019 A small recap of [what we have done in 2018][62].
* Feb 11, 2019 Release of version 2.3.0. See the main [CHANGES][63].
* April 21, 2018 New release (version 2.2.0) with an experimental support for (a polymorphic
  extension of) SMT-LIB 2.
* Mar 14, 2018 New release (v. 2.1.0) with the CDCL solver as a default SAT engine.
* Nov 14, 2017 New release (v. 2.0.0) with support for floating-point arithmetic.
* Nov 21, 2016 New release (v. 1.30) with [experimental support for models generation][64].
* Feb 16, 2016 New public release: private version 1.00 becomes [public release 1.01][65].
* Feb 09, 2016 A new private version (1.20) and its [Javascript version][66] are released.
* Jan 29, 2015 A new major private release (version 1.00) of Alt-Ergo is [released][67].
* Jul 15, 2014 [Here is a small tutorial][68] about "try Alt-Ergo".
* Feb 10, 2014 Discover our [development/release process][69] scheme !
* Feb 07, 2014 The latest public release is now [available][70] on a Github repository.

# Related Publications

* ### Arithmetic
* [ A Three-Tier Strategy for Reasoning About Floating-Point Numbers in SMT. ][71]
  Sylvain Conchon, Mohamed Iguernelala, Kailiang Ji, Guillaume Melquiond and Clément Fumex. CAV.
  2017 [[bib]][72]
* [ A Simplex-based extension of Fourier-Motzkin for solving linear integer arithmetic. ][73]
  François Bobot, Sylvain Conchon, Evelyne Contejean, Mohamed Iguernelala, Assia Mahboubi, Alain
  Mebsout, and Guillaume Melquiond. IJCAR, 2012. [[bib]][74]
* [ Built-in treatment of an axiomatic floating-point theory for SMT solvers. ][75]
  Sylvain Conchon, Guillaume Melquiond, Cody Roux, and Mohamed Iguernelala. SMT, 2012. [[bib]][76]
* 
* ### Shostak-like theories combination
* [ Canonized rewriting and ground AC completion modulo Shostak theories: Design and implementation.
  ][77]
  Sylvain Conchon, Évelyne Contejean, and Mohamed Iguernelala. LMCS, 2012. [[bib]][78]
* [ Canonized Rewriting and Ground AC Completion Modulo Shostak Theories. ][79]
  Sylvain Conchon, Évelyne Contejean, and Mohamed Iguernelala. TACAS, 2011. [[bib]][80]
* [ CC(X): Semantical combination of congruence closure with solvable theories. ][81]
  Sylvain Conchon, Évelyne Contejean, Johannes Kanig, and Stéphane Lescuyer. ENTCS, 2007.
  [[bib]][82]
* CC(X): Efficiently combining equality and solvable theories without canonizers.
  Sylvain Conchon, Évelyne Contejean, and Johannes Kanig. SMT Workshop, 2007. [[bib]][83]
* 
* ### Quantifiers
* [ Adding Decision Procedures to SMT Solvers Using Axioms with Triggers ][84]
  Claire Dross, Sylvain Conchon, Johannes Kanig and Andrei Paskevich. Journal of Automated
  Reasoning. 2016 [[bib]][85]
* [ Reasoning with triggers. ][86]
  Claire Dross, Sylvain Conchon, Johannes Kanig, and Andrei Paskevich. SMT workshop, 2012.
  [[bib]][87]
* [ Implementing Polymorphism in SMT solvers. ][88]
  François Bobot, Sylvain Conchon, Évelyne Contejean, and Stéphane Lescuyer. SMT Workshop, 2008.
  [[bib]][89]
* 
* ### Certification
* [ Improving Coq propositional reasoning using a lazy CNF conversion scheme. ][90]
  Stéphane Lescuyer and Sylvain Conchon. FroCoS, 2009. [[bib]][91]
* [ A reflexive formalization of a SAT solver in Coq. ][92]
  Stéphane Lescuyer and Sylvain Conchon. TPHOLs, 2008. [[bib]][93]
* [ Lightweight Integration of the Ergo Theorem Prover inside a Proof Assistant. ][94]
  Sylvain Conchon, Évelyne Contejean, Johannes Kanig, and Stéphane Lescuyer. AFM, 2007. [[bib]][95]
* 
* ### Thesis
* [ Improving performance of the SMT solver Alt-Ergo with a better integration of efficient SAT
  solver [FR]. ][96]
  Albin Coquereau. PhD thesis, 2019. [[bib]][97]
* [ Strengthening the Heart of an SMT-Solver: Design and Implementation of Efficient Decision
  Procedures. ][98]
  Mohamed Iguernelala. PhD thesis, 2013. [[bib]][99]
* [ SMT Techniques and their Applications: from Alt-Ergo to Cubicle. ][100]
  Sylvain Conchon. Habilitation thesis, 2012. [[bib]][101]
* [ Formalizing and Implementing a Reflexive Tactic for Automated Deduction in Coq ][102]
  Stéphane Lescuyer. PhD thesis, 2011. [[bib]][103]
* 
* ### Others
* [ SMT en pratique : le démonstrateur Alt-Ergo [Fr] ][104]
  Sylvain Conchon. College de France. 2016
* [ Alt-Ergo 2.2 ][105]
  Sylvain Conchon, Albin Coquereau, Mohamed Iguernlala and Alain Mebsout. SMT. 2018 [[bib]][106]
* [ Altgr-ergo, a graphical user interface for the SMT solver alt-ergo. ][107]
  Sylvain Conchon, Mohamed Iguernelala, and Alain Mebsout. UITP 2016 [[bib]][108]
* [ Alt-Ergo-Fuzz: A fuzzer for the Alt-Ergo SMT solver. ][109]
  Hichem Rami Ait El Hara, Guillaume Bury, and Steven de Oliveira. JFLA 2022 [[bib]][110]
* Also check our [blog posts about Alt-Ergo ][111]
This website uses the [creative template][112] of startbootstrap.com © 2013 — 2024 OCamlPro SAS, All
rights reserved. [Contact an administrator][113] [Legal][114]

[1]: #page-top
[2]: https://www.ocamlpro.com/
[3]: #news
[4]: #about
[5]: #tryit
[6]: #releases
[7]: https://ocamlpro.github.io/alt-ergo/latest/
[8]: #club
[9]: #services
[10]: #publications
[11]: #history
[12]: #tryit
[13]: #releases
[14]: https://ocamlpro.github.io/alt-ergo/latest/
[15]: #club
[16]: #decysif
[17]: #services
[18]: #publications
[19]: https://www.ocamlpro.com/2019/07/09/alt-ergo-participation-to-the-smt-comp-2019/
[20]: #history
[21]: https://why.lri.fr
[22]: https://why3.org
[23]: https://frama-c.com/
[24]: https://www.spark-2014.org
[25]: https://www.atelierb.eu
[26]: https://www.easycrypt.info/
[27]: mailto:alt-ergo@ocamlpro.com
[28]: https://www.lri.fr/index_en.php%3Flang=EN.html
[29]: https://www.inria.fr/centre/saclay
[30]: https://www.cnrs.fr/index.php
[31]: https://www.lri.fr/index_en.php%3Flang=EN.html
[32]: https://www.ocamlpro.com
[33]: https://www.u-psud.fr/en/
[34]: https://usr.lmf.cnrs.fr/ergo/
[35]: https://www.inria.fr/en/centre/saclay/
[36]: https://www.cnrs.fr/index.php
[37]: https://usr.lmf.cnrs.fr/ergo/
[38]: https://www.decysif.fr
[39]: https://decysif.fr/en/
[40]: https://try-alt-ergo.ocamlpro.com/
[41]: https://www.ocamlpro.com/2021/03/29/new-try-alt-ergo/
[42]: https://ocamlpro.github.io/alt-ergo/latest/Alt_ergo_native/index.html
[43]: https://why3.lri.fr/try
[44]: try.html
[45]: https://www.ocamlpro.com/2014/07/15/try-alt-ergo-in-your-browser/
[46]: https://ocamlpro.github.io/alt-ergo/latest/About/license.html
[47]: https://ocamlpro.github.io/alt-ergo/latest/About/changes.html
[48]: https://ocamlpro.github.io/alt-ergo/latest/
[49]: https://ocamlpro.github.io/alt-ergo/latest/Install/index.html
[50]: https://github.com/OCamlPro/alt-ergo/tree/v2.6.0
[51]: https://github.com/OCamlPro/alt-ergo/archive/v2.6.0.tar.gz
[52]: mailto:alt-ergo@ocamlpro.com?subject=Commercial%20Support
[53]: https://github.com/OCamlPro/alt-ergo/issues
[54]: https://ocaml.org/p/alt-ergo/
[55]: https://github.com/OCamlPro/alt-ergo/tree/2.3.3-free
[56]: http/alt-ergo-free-2.3.3/alt-ergo-free-2.3.3.tar.gz
[57]: https://opam.ocaml.org/packages/alt-ergo-free/alt-ergo-free.2.3.3/
[58]: mailto:alt-ergo@ocamlpro.com?subject=Join%20The%20Alt-Ergo%20Club
[59]: mailto:alt-ergo@ocamlpro.com?subject=Commercial%20Support
[60]: mailto:alt-ergo@ocamlpro.com
[61]: mailto:alt-ergo@ocamlpro.com
[62]: https://ocamlpro.com/blog/2019_02_11_whats_new_for_alt_ergo_in_2018_here_is_a_recap/
[63]: http/alt-ergo-2.3.0/CHANGES
[64]: https://www.ocamlpro.com/2016/11/21/release-of-alt-ergo-1-30-with-experimental-support-for-mod
els-generation/
[65]: #releases
[66]: try.html
[67]: https://www.ocamlpro.com/2015/01/29/private-release-of-alt-ergo-1-00/
[68]: https://www.ocamlpro.com/2014/07/15/try-alt-ergo-in-your-browser/
[69]: #services
[70]: https://github.com/OCamlPro/alt-ergo
[71]: https://hal.inria.fr/hal-01522770/document
[72]: https://www.lri.fr/~conchon/bib/conchon_bib.html#ConchonIJMF17
[73]: http://hal.inria.fr/docs/00/68/81/54/PDF/main.pdf
[74]: http://toccata.lri.fr/publications/conchon_bib.html#bobot12ijcar
[75]: https://www.lri.fr/~melquion/doc/12-smt-article.pdf
[76]: http://toccata.lri.fr/publications/conchon_bib.html#conchon12smt
[77]: http://arxiv.org/pdf/1207.3262
[78]: http://toccata.lri.fr/publications/conchon_bib.html#conchon12lmcs
[79]: http://hal.inria.fr/docs/00/77/76/63/PDF/conchon11tacas.pdf
[80]: http://toccata.lri.fr/publications/conchon_bib.html#conchon11tacas
[81]: http://www.sciencedirect.com/science/article/pii/S1571066108002958/pdf?md5=40cffa97d665d9b4e4c
87e0feaa31ab0&pid=1-s2.0-S1571066108002958-main.pdf
[82]: http://toccata.lri.fr/publications/conchon_bib.html#conchon08entcs
[83]: http://toccata.lri.fr/publications/conchon_bib.html#conchon07smt
[84]: https://hal.archives-ouvertes.fr/hal-01221066
[85]: https://www.lri.fr/~conchon/bib/conchon_bib.html#DrossCKP16
[86]: https://inria.hal.science/hal-00703207/PDF/RR-7986.pdf
[87]: http://toccata.lri.fr/publications/conchon_bib.html#dross12smt
[88]: https://www.researchgate.net/publication/234789107_Implementing_polymorphism_in_SMT_solvers
[89]: http://toccata.lri.fr/publications/conchon_bib.html#conchon08smt
[90]: https://link.springer.com/chapter/10.1007/978-3-642-04222-5_18
[91]: http://toccata.lri.fr/publications/conchon_bib.html#LescuyerConchon09frocos
[92]: https://users.encs.concordia.ca/~tphols08/TPHOLs2008/ET/64-75.pdf
[93]: http://toccata.lri.fr/publications/conchon_bib.html#lescuyer08tpholset
[94]: https://dl.acm.org/doi/abs/10.1145/1345169.1345176
[95]: http://toccata.lri.fr/publications/conchon_bib.html#conchon07afm
[96]: https://pastel.archives-ouvertes.fr/tel-02504894/document
[97]: https://pastel.archives-ouvertes.fr/tel-02504894v1/bibtex
[98]: http://tel.archives-ouvertes.fr/docs/00/84/25/55/PDF/VD2_IGUERNELALA_MOHAMED_10062013.pdf
[99]: http://toccata.lri.fr/publications/complete-phd_bib.html#iguer13phd
[100]: http://www.lri.fr/~conchon/publis/conchonHDR.pdf
[101]: http://toccata.lri.fr/publications/complete-phd_bib.html#conchon12hdr
[102]: http://tel.archives-ouvertes.fr/docs/00/71/36/68/PDF/VA2_LESCUYER_STEPHANE_04012011.pdf
[103]: http://toccata.lri.fr/publications/complete-phd_bib.html#lescuyer11these
[104]: https://www.college-de-france.fr/fr/agenda/seminaire/structures-de-donnees-et-algorithmes-pou
r-la-verification-formelle/smt-en-pratique-le-demonstrateur-alt-ergo
[105]: https://hal.inria.fr/hal-01960203/document
[106]: https://hal.inria.fr/hal-01960203v1/bibtex
[107]: https://arxiv.org/pdf/1701.07124.pdf
[108]: https://www.lri.fr/~conchon/bib/conchon_bib.html#ConchonIM17
[109]: https://hal.inria.fr/hal-03626861/document
[110]: https://hal.inria.fr/hal-03626861/bibtex
[111]: https://ocamlpro.com/blog/tag/altergo/
[112]: https://startbootstrap.com/template-overviews/creative/
[113]: mailto:contact@ocamlpro.com
[114]: https://ocamlpro.com/legal-notice/
