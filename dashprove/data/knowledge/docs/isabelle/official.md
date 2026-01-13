## What is Isabelle?

Isabelle is a generic proof assistant. It allows mathematical formulas to be expressed in a formal
language and provides tools for proving those formulas in a logical calculus. Isabelle was
originally developed at the [University of Cambridge][1] and [Technische Universität München][2],
but now includes numerous contributions from institutions and individuals worldwide. See the
[Isabelle overview][3] for a brief introduction.

## Now available: Isabelle2025-1 (December 2025)

* [[ ] [ ] Download for Linux (Intel)][4]
* [[ ] [ ] Download for Linux (ARM)][5]
* [[ ] [ ] Download for Windows][6]
* [[ ] [ ] Download for macOS][7]

**Hardware requirements:**

* *Small experiments:* 4 GB memory, 2 CPU cores
* *Medium applications:* 8 GB memory, 4 CPU cores
* *Large projects:* 16 GB memory, 8 CPU cores
* *Extra-large projects:* 64 GB memory, 16 CPU cores

**Some notable changes:**

* PIDE: load markup from background session image (e.g. theory `HOL.Nat`).
* Isabelle/jEdit: support for command-line system options (`-o`).
* Isabelle/jEdit: support for dark mode and screen readers.
* Isabelle/jEdit: built-in navigator, without requiring plugins.
* Isabelle/jEdit: more reactive mouse handling, notably on macOS.
* Isabelle/jEdit: updated FlatLaf GUI L&F with scalable SVG icons.
* Isabelle/jEdit: improved appearance on Linux (GUI scaling) and macOS (L&F).
* Isabelle/VSCode: more robust session build process on console.
* Isabelle/VSCode: GUI panels for Documentation, Symbols, Sledgehammer.
* Isabelle/VSCode: update of underlying VSCodium distribution.
* HOL: various improvements of theory libraries.
* HOL: various improvements to code generation.
* HOL: updates and improvements of Sledgehammer and external provers.
* HOL: generation of running time functions for HOL functions.
* System: ML settings may depend on command-line system options (`-o`).
* System: command-line tool to process theories within adhoc session.
* System: more detailed build progress.
* System: more robust build cluster management.

See also the cumulative [NEWS][8].

## Distribution & Support

Isabelle is distributed for free under a conglomerate of open-source licenses, but the main
code-base is subject to BSD-style regulations. The application bundles include source and binary
packages and documentation, see the detailed [installation instructions][9]. A vast collection of
Isabelle examples and applications is available from the [Archive of Formal Proofs][10].

Support is available by the official [documentation][11] and mailing lists:

* The **[isabelle-users mailing list][12]** provides a forum for Isabelle users to discuss problems,
  exchange information, and make announcements. Users of official Isabelle releases should
  [subscribe][13] or see the [archive][14].
* The **[isabelle-dev mailing list][15]** covers the Isabelle development process, including
  intermediate repository versions, and administrative issues concerning the website or testing
  infrastructure. Early adopters of [development snapshots][16] or [repository versions][17] (with
  alternative [repository clone][18]) should [subscribe][19] or see the [archive][20].

[Zulip Chat][21] is a real-time discussion platform to exchange ideas, ask questions, and
collaborate on Isabelle projects, with minimalistic [public archive][22].

[Stack Overflow][23] and [Stack Exchange][24] are a question-and-answer platform, with complex
review process but limited discussion facilities.

[1]: https://www.cl.cam.ac.uk/research/hvg/Isabelle/Cambridge
[2]: https://www21.in.tum.de
[3]: overview.html
[4]: dist/Isabelle2025-1_linux.tar.gz
[5]: dist/Isabelle2025-1_linux_arm.tar.gz
[6]: dist/Isabelle2025-1.exe
[7]: dist/Isabelle2025-1_macos.tar.gz
[8]: dist/Isabelle2025-1/doc/NEWS.html
[9]: installation.html
[10]: https://www.isa-afp.org/
[11]: documentation.html
[12]: https://lists.cam.ac.uk/sympa/subscribe/cl-isabelle-users
[13]: https://lists.cam.ac.uk/sympa/subscribe/cl-isabelle-users?previous_action=info
[14]: https://lists.cam.ac.uk/sympa/arc/cl-isabelle-users
[15]: https://mailman46.in.tum.de/pipermail/isabelle-dev
[16]: https://isatest.sketis.net/devel/release_snapshot
[17]: https://isabelle.in.tum.de/repos/isabelle
[18]: https://isabelle.sketis.net/repos/isabelle
[19]: https://mailman46.in.tum.de/mailman/listinfo/isabelle-dev
[20]: https://mailman46.in.tum.de/pipermail/isabelle-dev
[21]: https://isabelle.zulipchat.com
[22]: https://isabelle.systems/zulip-archive
[23]: https://stackoverflow.com/questions/tagged/isabelle
[24]: https://proofassistants.stackexchange.com/questions/tagged/isabelle
