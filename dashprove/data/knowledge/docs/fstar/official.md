# Introduction

F* (pronounced F star) is a general-purpose proof-oriented programming language, supporting both
purely functional and effectful programming. It combines the expressive power of dependent types
with proof automation based on SMT solving and tactic-based interactive theorem proving.

F* programs compile, by default, to OCaml. Various fragments of F* can also be extracted to F#, to C
or Wasm by a tool called [KaRaMeL][1], or to assembly using the [Vale][2] toolchain. F* is
implemented in F* and bootstrapped using OCaml.

F* is open source on [GitHub][3] and is under active development by [Microsoft Research][4],
[Inria][5], and by the community.

[1]: https://github.com/FStarLang/karamel
[2]: https://github.com/project-everest/vale
[3]: https://github.com/FStarLang/FStar
[4]: https://research.microsoft.com
[5]: https://team.inria.fr/prosecco/
