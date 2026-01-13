* Welcome to Alt-Ergo’s documentation!

# Welcome to Alt-Ergo’s documentation![][1]

[Alt-Ergo][2] is an open-source automatic solver of mathematical formulas designed for program
verification. It is based on [Satisfiability Modulo Theories][3] (SMT). Solvers of this family have
made impressive advances and became very popular during the last decade. They are now used is
various domains such as hardware design, software verification and formal testing.

It was developed at [LRI][4], and is now improved and maintained at [OCamlPro][5], and friendly
collaboration is maintained with the [Why3][6] development team.

You can [try Alt-Ergo][7] online. You can also learn more about our partners with the [Alt-Ergo
Users’ Club][8].

If you are using Alt-Ergo as a library, see the [API documentation][9] (also available [on
ocaml.org][10]).

## Input file formats[][11]

Alt-ergo supports different input languages:

* Alt-ergo supports the SMT-LIB language v2.6. **This is Alt-Ergo’s preferred and recommended input
  format.**
* The original input language is its native language, based on the language of the Why3 platform.
  Since the version 2.6.0, this language is deprecated.
* It also (partially) supports the input language of Why3 through the [AB-Why3 plugin][12].

Contents

* [Install][13]
  
  * [From a package manager][14]
  * [From GitHub releases (Linux and macOS)][15]
  * [From sources][16]
* [Usage][17]
  
  * [Command-line][18]
  * [Library][19]
  * [Javascript][20]
* [SMT-LIB language][21]
  
  * [Bit-vectors][22]
  * [Floating-Point Arithmetic][23]
* [Alt-Ergo's native language][24]
  
  * [ Summary][25]
  * [ Declaration of symbols][26]
  * [ Types][27]
  * [ Declaration of axioms][28]
  * [ Setting goals][29]
  * [ Theories][30]
  * [ Control Flow][31]
  * [ Syntax of declarations and expressions][32]
* [Model generation][33]
  
  * [Activation][34]
  * [Correctness of model generation][35]
  * [Examples][36]
* [Optimization][37]
  
  * [MaxSMT syntax][38]
  * [Activation][39]
  * [Examples][40]
* [API documentation][41]
* [Plugins][42]
  
  * [Inequality plugins][43]
  * [AB why3 plugin (**deprecated**)][44]
* [Developer's documentation][45]
  
  * [Project Architecture][46]
  * [Contributing guidelines][47]
* [About][48]
  
  * [Licenses][49]
  * [Changes][50]
  * [Try-Alt-Ergo][51]
  * [Alt-Ergo's website][52]
  * [Scientific publications][53]
[Next ][54]

© Copyright 2020 - 2023, Alt-Ergo developers.

Built with [Sphinx][55] using a [theme][56] provided by [Read the Docs][57].

[1]: #welcome-to-alt-ergo-s-documentation
[2]: https://alt-ergo.ocamlpro.com
[3]: https://en.wikipedia.org/wiki/Satisfiability_modulo_theories
[4]: https://www.lri.fr
[5]: https://www.ocamlpro.com
[6]: http://why3.lri.fr/
[7]: https://alt-ergo.ocamlpro.com/try.html
[8]: https://alt-ergo.ocamlpro.com/#club
[9]: API/index.html
[10]: https://ocaml.org/p/alt-ergo-lib/latest/doc/index.html
[11]: #input-file-formats
[12]: Plugins/ab_why3.html
[13]: Install/index.html
[14]: Install/index.html#from-a-package-manager
[15]: Install/index.html#from-github-releases-linux-and-macos
[16]: Install/index.html#from-sources
[17]: Usage/index.html
[18]: Usage/index.html#command-line
[19]: Usage/index.html#library
[20]: Usage/index.html#javascript
[21]: SMT-LIB_language/index.html
[22]: SMT-LIB_language/index.html#bit-vectors
[23]: SMT-LIB_language/index.html#floating-point-arithmetic
[24]: Alt_ergo_native/index.html
[25]: Alt_ergo_native/00_summary.html
[26]: Alt_ergo_native/01_declaration_of_symbols.html
[27]: Alt_ergo_native/02_types/index.html
[28]: Alt_ergo_native/03_declaration_of_axioms.html
[29]: Alt_ergo_native/04_setting_goals.html
[30]: Alt_ergo_native/05_theories.html
[31]: Alt_ergo_native/06_control_flow.html
[32]: Alt_ergo_native/07_syntax_of_declarations_and_expressions.html
[33]: Model_generation.html
[34]: Model_generation.html#activation
[35]: Model_generation.html#correctness-of-model-generation
[36]: Model_generation.html#examples
[37]: Optimization.html
[38]: Optimization.html#maxsmt-syntax
[39]: Optimization.html#activation
[40]: Optimization.html#examples
[41]: API/index.html
[42]: Plugins/index.html
[43]: Plugins/index.html#inequality-plugins
[44]: Plugins/index.html#ab-why3-plugin-deprecated
[45]: Dev/index.html
[46]: Dev/architecture.html
[47]: Dev/contributing.html
[48]: About/index.html
[49]: About/license.html
[50]: About/changes.html
[51]: https://alt-ergo.ocamlpro.com/try.html
[52]: https://alt-ergo.ocamlpro.com/
[53]: https://alt-ergo.ocamlpro.com/#publications
[54]: Install/index.html
[55]: https://www.sphinx-doc.org/
[56]: https://github.com/readthedocs/sphinx_rtd_theme
[57]: https://readthedocs.org
