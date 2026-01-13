# [Clippy][1]

[[License: MIT OR Apache-2.0]][2]

A collection of lints to catch common mistakes and improve your [Rust][3] code.

[There are over 750 lints included in this crate!][4]

Lints are divided into categories, each with a default [lint level][5]. You can choose how much
Clippy is supposed to a̶n̶n̶o̶y̶ help you by changing the lint level by category.

────────────────┬───────────────────────────────────────────────────────────────────────┬───────────
Category        │Description                                                            │Default    
                │                                                                       │level      
────────────────┼───────────────────────────────────────────────────────────────────────┼───────────
`clippy::all`   │all lints that are on by default (correctness, suspicious, style,      │**warn/deny
                │complexity, perf)                                                      │**         
────────────────┼───────────────────────────────────────────────────────────────────────┼───────────
`clippy::correct│code that is outright wrong or useless                                 │**deny**   
ness`           │                                                                       │           
────────────────┼───────────────────────────────────────────────────────────────────────┼───────────
`clippy::suspici│code that is most likely wrong or useless                              │**warn**   
ous`            │                                                                       │           
────────────────┼───────────────────────────────────────────────────────────────────────┼───────────
`clippy::style` │code that should be written in a more idiomatic way                    │**warn**   
────────────────┼───────────────────────────────────────────────────────────────────────┼───────────
`clippy::complex│code that does something simple but in a complex way                   │**warn**   
ity`            │                                                                       │           
────────────────┼───────────────────────────────────────────────────────────────────────┼───────────
`clippy::perf`  │code that can be written to run faster                                 │**warn**   
────────────────┼───────────────────────────────────────────────────────────────────────┼───────────
`clippy::pedanti│lints which are rather strict or have occasional false positives       │allow      
c`              │                                                                       │           
────────────────┼───────────────────────────────────────────────────────────────────────┼───────────
`clippy::restric│lints which prevent the use of language and library features^{[1][6]}  │allow      
tion`           │                                                                       │           
────────────────┼───────────────────────────────────────────────────────────────────────┼───────────
`clippy::nursery│new lints that are still under development                             │allow      
`               │                                                                       │           
────────────────┼───────────────────────────────────────────────────────────────────────┼───────────
`clippy::cargo` │lints for the cargo manifest                                           │allow      
────────────────┴───────────────────────────────────────────────────────────────────────┴───────────

More to come, please [file an issue][7] if you have ideas!

The `restriction` category should, *emphatically*, not be enabled as a whole. The contained lints
may lint against perfectly reasonable code, may not have an alternative suggestion, and may
contradict any other lints (including other categories). Lints should be considered on a
case-by-case basis before enabling.

1. Some use cases for `restriction` lints include:
   
   * Strict coding styles (e.g. [`clippy::else_if_without_else`][8]).
   * Additional restrictions on CI (e.g. [`clippy::todo`][9]).
   * Preventing panicking in certain functions (e.g. [`clippy::unwrap_used`][10]).
   * Running a lint only on a subset of code (e.g. `#[forbid(clippy::float_arithmetic)]` on a
     module).
   [↩][11]

[1]: #clippy
[2]: https://github.com/rust-lang/rust-clippy#license
[3]: https://github.com/rust-lang/rust
[4]: https://rust-lang.github.io/rust-clippy/master/index.html
[5]: https://doc.rust-lang.org/rustc/lints/levels.html
[6]: #footnote-restrict
[7]: https://github.com/rust-lang/rust-clippy/issues
[8]: https://rust-lang.github.io/rust-clippy/master/index.html#else_if_without_else
[9]: https://rust-lang.github.io/rust-clippy/master/index.html#todo
[10]: https://rust-lang.github.io/rust-clippy/master/index.html#unwrap_used
[11]: #fr-restrict-1
