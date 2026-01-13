# cargo-mutants

[https://github.com/sourcefrog/cargo-mutants][1]

[[Tests]][2] [[crates.io]][3] [[libs.rs]][4] [[GitHub Sponsors]][5] [[Donate]][6]

cargo-mutants helps you improve your program's quality by finding places where bugs could be
inserted without causing any tests to fail.

Coverage measurements can be helpful, but they really tell you what code is *reached* by a test, and
not whether the test really *checks* anything about the behavior of the code. Mutation tests give
different information, about whether the tests really check the code's behavior.

The goal of cargo-mutants is to be *easy* to run on any Rust source tree, and to tell you something
*interesting* about areas where bugs might be lurking or the tests might be insufficient.

**For more background, see the [slides][7] and [video][8] from my Rustconf 2024 talk.**

**The main documentation is the user guide at [https://mutants.rs/][9].**

## Prerequisites

cargo-mutants can help on trees with non-flaky tests that run under `cargo test` or [`cargo nextest
run`][10].

## Install

cargo install --locked cargo-mutants

You can also install using [cargo-binstall][11] or from binaries attached to GitHub releases.

## Quick start

From within a Rust source directory, just run

cargo mutants

To generate mutants in only one file:

cargo mutants -f src/something.rs

## Integration with CI

The [manual includes instructions and examples for automatically testing mutants in CI][12],
including incremental testing of pull requests and full testing of the development branch.

## Help advance cargo-mutants

If you use cargo-mutants or just like the idea you can help it get better:

* [Post an experience report in GitHub discussions][13], saying whether it worked, failed, found
  interesting results, etc.
* [Sponsor development][14]

## Project status

As of August 2025 this is an actively-maintained spare time project. I expect to make [releases][15]
about every one or two months.

It's very usable at it is and there's room for lots more future improvement, especially in adding
new types of mutation.

If you try it out on your project, [I'd love to hear back in a github discussion][16] whether it
worked well or what could be better:

* Did it work on your tree? Did you need to set any options or do any debugging to get it working?
* Did it find meaningful gaps in testing? Where there too many false positives?
* What do you think would make it better or easier?

This software is provided as-is with no warranty of any kind.

## Further reading

See also:

* [cargo-mutants manual][17]
* [How cargo-mutants compares to other techniques and tools][18].
* [Design notes][19]
* [Contributing][20]
* [Release notes][21]
* [Discussions][22]

[1]: https://github.com/sourcefrog/cargo-mutants
[2]: https://github.com/sourcefrog/cargo-mutants/actions/workflows/tests.yml?query=branch%3Amain
[3]: https://crates.io/crates/cargo-mutants
[4]: https://lib.rs/crates/cargo-mutants
[5]: https://github.com/sponsors/sourcefrog
[6]: https://donate.stripe.com/fZu6oH6ry9I6epVfF7dfG00
[7]: https://docs.google.com/presentation/d/1YDwHz6ysRRNYRDtv80EMRAs4FQu2KKQ-IbGu2jrqswY/edit?pli=1&
slide=id.g2876539b71f_0_0
[8]: https://www.youtube.com/watch?v=PjDHe-PkOy8&pp=ygUNY2FyZ28tbXV0YW50cw%3D%3D
[9]: https://mutants.rs/
[10]: https://nexte.st/
[11]: https://github.com/cargo-bins/cargo-binstall
[12]: https://mutants.rs/ci.html
[13]: https://github.com/sourcefrog/cargo-mutants/discussions
[14]: https://github.com/sponsors/sourcefrog
[15]: https://github.com/sourcefrog/cargo-mutants/releases
[16]: https://github.com/sourcefrog/cargo-mutants/discussions/categories/general
[17]: https://mutants.rs/
[18]: https://github.com/sourcefrog/cargo-mutants/wiki/Compared
[19]: /sourcefrog/cargo-mutants/blob/main/DESIGN.md
[20]: /sourcefrog/cargo-mutants/blob/main/CONTRIBUTING.md
[21]: /sourcefrog/cargo-mutants/blob/main/NEWS.md
[22]: https://github.com/sourcefrog/cargo-mutants/discussions
