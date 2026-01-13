[Frama-C][1]

* [Features][2]
* [Documentation][3]
* [Publications][4]
* [Blog][5]
* [Jobs][6]
* [Contact][7]
* [Download][8]
[******][9]

## A platform to make your C code safer and more secure

Frama-C is an open-source extensible and collaborative platform dedicated to source-code analysis of
C software. The Frama-C analyzers assist you in various source-code-related activities, from the
navigation through unfamiliar projects up to the certification of critical software.

[ Download Frama-C ** ** ** ][10]
[All versions][11]
[Frama-C on GitLab][12]

## Latest News [ [RSS Feeds] ][13]

──────────┬──────────────────────────────────────────────────
2025-12-03│[Release of Frama-C 32.0 (Germanium)][14]         
──────────┼──────────────────────────────────────────────────
2025-11-24│[Internship Position at CEA List - LSL][15]       
──────────┼──────────────────────────────────────────────────
2025-11-04│[Beta release of Frama-C 32.0~beta                
          │(Germanium)][16]                                  
──────────┼──────────────────────────────────────────────────
2025-10-13│[Internship Position at CEA LIST - LSL][17]       
──────────┼──────────────────────────────────────────────────
2025-10-13│[Internship Position at CEA LIST - LSL][18]       
──────────┼──────────────────────────────────────────────────
2025-10-13│[Internship Position at CEA LIST - LSL][19]       
──────────┼──────────────────────────────────────────────────
2025-07-09│[LUncov v0.2.4 for Frama-C 31.0 Gallium][20]      
──────────┼──────────────────────────────────────────────────
2025-07-07│[LAnnotate v0.2.4 for Frama-C 31.0 Gallium][21]   
──────────┼──────────────────────────────────────────────────
2025-06-25│[MetAcsl v0.9 for Frama-C 31.0 Gallium][22]       
──────────┴──────────────────────────────────────────────────

[More news ...][23]

## Highlights of Frama-C's capabilities

### Overview

### Eva

### WP

### E-ACSL

Since no single technique will ever be able to fit all software verification needs, **Frama-C** aims
at combining program analysis technics, provided as **plug-ins**, to **guarantee the absence of
bugs** in C programs.

#### An analysis framework fueled by formal methods

The main particularity of Frama-C is that it embeds tools based on **formal methods**, that are
mathematical technics to reason about programs. Thus, most of the analyzers in Frama-C are
**sound**: they never remain silent when a bug might happen.

#### Absence of runtime errors and beyond

**Undefined behaviors** in C programs can cause safety and security issues. Many tools can be used
to find these **runtime errors**, but most of them provide heuristic bug finding that can miss bugs,
whereas Frama-C is meant to **guarantee that no bug can happen**. Moreover, Frama-C provides a
**formal specification** language, [ACSL][24], which gives the opportunity not only to prove that no
runtime error can happen, but also **conformance to a functional specification**.

#### Widely used, from experimental research to industry

Frama-C is largely used for **teaching**, **experimental research**, and **industrial
applications**. It has been used successfully for **certification** purposes for DO-178, IEC 60880,
Common Criteria EAL 6-7 … You may now want to go on to [the description of Frama-C’s features][25]
or to [a page with more details about its modular, extensible architecture][26]

The **Eva** plug-in aims at **proving the absence of runtime errors** caused by C undefined
behaviors, such as invalid memory accesses, reads of uninitialized memory, integer overflows,
divisions by zero, dangling pointers…

#### Based on abstract interpretation

Eva relies on abstract interpretation to perform a **sound static analysis** of the entire program
and capture all possible behaviors of its executions. It is thus able to **report all errors that
might happen** — in the class of undefined behaviors supported by the analysis.

During its analysis, Eva infers many properties about the analyzed program, including an
over-approximation of **the possible values for each variable** at each program point. The Frama-C
**graphical user interface** can then be used to browse the analyzed code, review the list of
emitted alarms, display the inferred ranges for any variable, highlight dead code, and more.

#### Highly configurable analysis

Although the Eva analysis is **automatic**, many parameters are available to finely configure its
behavior, impacting its results accuracy and analysis time. More information on this plug-in and how
to use it are available on the [dedicated page][27] and in the [Eva user manual][28].

[Eva overview]

The **WP** plug-in aims at proving **functional correctness** of a program. However, while runtime
errors can be deduced from the rules of the programming language, logic errors are related to the
intention of the developer, that must be described to be verified.

#### Functional specification

[ACSL][29] (ANSI/ISO C Specification Language) is the language used in Frama-C to **provide
specifications**, and in particular **function contracts**, that is the **precondition** of the
function (what it *requires* from the input) and the **postcondition** (what it *ensures* in
output), provided as annotations.

#### A deductive verification tool

The [WP plugin of Frama-C][30] uses (a variant of) **weakest precondition calculus**. Compared to
the Eva plug-in, it requires more work from the user: one must provide **the contracts of the
functions**, but also additional annotations, for example loop invariants and assertions, to guide
the proof process. For each annotation, WP produces a verification condition, a formula that must
hold to **guarantee that the annotation is verified**.

#### Maximizing automation

WP relies on **SMT solvers** (like Alt-Ergp, CVC5 or Z3) that can **prove automatically up to 98%**
of the verification conditions on real world case studies. Furthermore, it provides interactive
mechanisms to describe user-defined proof strategies or to build proof scripts when automated
solvers fail to prove.

More information on this plug-in, and in particular tutorials, are available [on the dedicated
page][31]

[WP overview]

The **E-ACSL** plug-in provides **Runtime Annotation Checking (RAC)**, a lightweight formal method
consisting in checking code annotations during the program execution. While static formal methods
aim for guarantees that hold for any execution, RAC only **provides guarantees about the particular
execution** it monitors. This allows RAC-based tools to be used with **minimum intervention from the
user**.

#### Runtime assertion checking with the E-ACSL plug-in

E-ACSL is able to translate (Executable-)ACSL annotations into C code, so that they can be verified
at runtime. E-ACSL is often **used in combination** with other plug-ins of Frama-C, *e.g.* Eva and
WP. It can be used to understand **why a proof with WP fails**, or to **monitor alarms after Eva
analysis**. E-ACSL builds upon the results from the other plug-ins: it does not instrument
annotations that have already been proved valid by a static analyzer.

#### Formally correct monitoring

Translating ACSL is not simple as it may seem. The generated C code must **not introduce any new bug
in the code**. Thus, E-ACSL relies on the [RTE][32] plug-in, on GMP integers and on a **high
performance shadow memory** to capture **all possible runtime error**. Since all of this can be
costly, E-ACSL optimizes these constructs when possible, this guarantees that **E-ACSL has a
reasonable runtime overhead**.

Not all ACSL constructs can be translated into C code (for example quantifiers must be bounded), one
has to restrict to the *executable* fragment of the language. More details can be found on [the page
dedicated to E-ACSL][33].

* The actual transformation is slightly more complex than this illustration.

[E-ACSL overview]
Copyright © 2007-2025 Frama-C. All Rights Reserved.

* [Terms Of Use][34]
* [Authors][35]
* [Acknowledgements][36]

[1]: /index.html
[2]: /html/kernel-plugin.html
[3]: /html/documentation.html
[4]: /html/publications.html
[5]: /blog/index.html
[6]: /html/jobs.html
[7]: /html/contact.html
[8]: /html/get-frama-c.html
[9]: /html/get-frama-c.html
[10]: /html/get-frama-c.html
[11]: /html/framac-versions.html
[12]: https://git.frama-c.com/pub/frama-c
[13]: /html/feeds.html
[14]: /html/news.html#2025-12-03
[15]: /html/news.html#2025-11-24
[16]: /html/news.html#2025-11-04
[17]: /html/news.html#2025-10-13
[18]: /html/news.html#2025-10-13
[19]: /html/news.html#2025-10-13
[20]: /html/news.html#2025-07-09
[21]: /html/news.html#2025-07-07
[22]: /html/news.html#2025-06-25
[23]: /html/news.html#2025-06-25
[24]: /html/acsl.html
[25]: /html/kernel-plugin.html
[26]: /html/kernel.html
[27]: /fc-plugins/eva.html
[28]: 
[29]: /html/acsl.html
[30]: /fc-plugins/wp.html
[31]: /fc-plugins/wp.html
[32]: /fc-plugins/rte.html
[33]: /fc-plugins/e-acsl.html
[34]: /html/terms-of-use.html
[35]: /html/authors.html
[36]: /html/acknowledgement.html
