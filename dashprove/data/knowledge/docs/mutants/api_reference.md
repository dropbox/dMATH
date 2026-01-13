[ Docs.rs ][1]

* [ cargo-mutants-26.0.0 ][2]

* [docs.rs][3]
  
  * [ About docs.rs][4]
  * [ Badges][5]
  * [ Builds][6]
  * [ Metadata][7]
  * [ Shorthand URLs][8]
  * [ Download][9]
  * [ Rustdoc JSON][10]
  * [ Build queue][11]
  * [ Privacy policy][12]

* [Rust][13]
  
  * [Rust website][14]
  * [The Book][15]
  * [Standard Library API Reference][16]
  * [Rust by Example][17]
  * [The Cargo Guide][18]
  * [Clippy Documentation][19]

# cargo-mutants 26.0.0

Inject bugs and see if your tests catch them

* [ Crate ][20]
* [ Source ][21]
* [ Builds ][22]
* [ Feature flags ][23]

* Size
* Source code size: 1.14 MB This is the summed size of all the files inside the crates.io package
  for this release.
* Links
* [ Homepage ][24]
* [ sourcefrog/cargo-mutants
  1028 33 82 ][25]
* [ crates.io ][26]
* Dependencies
* * [ anyhow ^1.0.86 *normal* ][27]
  * [ camino ^1.1.9 *normal* ][28]
  * [ cargo_metadata ^0.19 *normal* ][29]
  * [ clap ^4.5 *normal* ][30]
  * [ clap_complete ^4 *normal* ][31]
  * [ color-print ^0.3 *normal* ][32]
  * [ console ^0.15 *normal* ][33]
  * [ ctrlc ^3.4 *normal* ][34]
  * [ fastrand ^2 *normal* ][35]
  * [ fs2 ^0.4 *normal* ][36]
  * [ globset ^0.4.16 *normal* ][37]
  * [ ignore ^0.4.23 *normal* ][38]
  * [ indoc ^2.0.0 *normal* ][39]
  * [ itertools ^0.13 *normal* ][40]
  * [ jiff ^0.1 *normal* ][41]
  * [ jobserver ^0.1 *normal* ][42]
  * [ mutants ^0.0.3 *normal* ][43]
  * [ nextest-metadata ^0.12.1 *normal* ][44]
  * [ num_cpus ^1.16 *normal* ][45]
  * [ nutmeg ^0.1.5 *normal* ][46]
  * [ patch ^0.7 *normal* ][47]
  * [ path-slash ^0.2 *normal* ][48]
  * [ proc-macro2 ^1.0.91 *normal* ][49]
  * [ quote ^1.0.35 *normal* ][50]
  * [ reflink ^0.1 *normal* ][51]
  * [ regex ^1.10 *normal* ][52]
  * [ schemars ^0.9 *normal* ][53]
  * [ serde ^1.0.209 *normal* ][54]
  * [ serde_json ^1.0.128 *normal* ][55]
  * [ similar ^2.1 *normal* ][56]
  * [ strum ^0.26 *normal* ][57]
  * [ syn ^2.0.104 *normal* ][58]
  * [ tempfile ^3.20 *normal* ][59]
  * [ test-log ^0.2.16 *normal* ][60]
  * [ toml ^0.8 *normal* ][61]
  * [ tracing ^0.1.41 *normal* ][62]
  * [ tracing-subscriber ^0.3.20 *normal* ][63]
  * [ whoami ^1.5 *normal* ][64]
  * [ assert_cmd ^2.0 *dev* ][65]
  * [ assert_matches ^1.5 *dev* ][66]
  * [ cp_r ^0.5.2 *dev* ][67]
  * [ insta ^1.12 *dev* ][68]
  * [ lazy_static ^1.4 *dev* ][69]
  * [ predicates ^3 *dev* ][70]
  * [ pretty_assertions ^1 *dev* ][71]
  * [ rusty-fork ^0.3.1 *dev* ][72]
  * [ walkdir ^2.5 *dev* ][73]
  * [ nix ^0.30 *normal* ][74]
* Versions
* * [26.0.0 (2025-12-08)][75]
  * [25.3.1 (2025-08-10)][76]
  * [25.3.0 (2025-08-10)][77]
  * [25.2.2 (2025-07-19)][78]
  * [25.2.2-pre2 (2025-07-19)][79]
  * [25.2.2-pre0 (2025-07-19)][80]
  * [25.2.1 (2025-07-10)][81]
  * [25.2.0 (2025-06-30)][82]
  * [25.1.0 (2025-06-05)][83]
  * [25.0.1 (2025-04-21)][84]
  * [25.0.1-pre3 (2025-01-05)][85]
  * [25.0.1-pre2 (2025-01-05)][86]
  * [25.0.0 (2025-01-04)][87]
  * [24.11.2 (2024-11-24)][88]
  * [24.11.1 (2024-11-23)][89]
  * [24.11.0 (2024-11-12)][90]
  * [24.9.0 (2024-09-12)][91]
  * [24.7.1 (2024-07-25)][92]
  * [24.7.0 (2024-07-05)][93]
  * [24.5.0 (2024-05-14)][94]
  * [24.4.0 (2024-04-22)][95]
  * [24.3.0 (2024-03-24)][96]
  * [24.2.1 (2024-02-25)][97]
  * [24.2.0 (2024-02-04)][98]
  * [24.1.2 (2024-01-24)][99]
  * [24.1.1 (2024-01-16)][100]
  * [24.1.0 (2024-01-11)][101]
  * [23.12.2 (2023-12-23)][102]
  * [23.12.0 (2023-12-04)][103]
  * [23.11.2 (2023-11-26)][104]
  * [23.11.1 (2023-11-13)][105]
  * [23.11.0 (2023-11-12)][106]
  * [23.10.0 (2023-10-04)][107]
  * [23.9.1 (2023-09-17)][108]
  * [23.9.0 (2023-09-14)][109]
  * [23.6.0 (2023-06-11)][110]
  * [23.5.0 (2023-05-27)][111]
  * [1.2.3 (2023-05-05)][112]
  * [1.2.2 (2023-04-01)][113]
  * [1.2.1 (2023-01-06)][114]
  * [1.2.0 (2022-11-25)][115]
  * [1.1.1 (2022-11-01)][116]
  * [1.1.1-pre (2022-10-31)][117]
  * [1.1.0 (2022-10-30)][118]
  * [1.0.3 (2022-09-29)][119]
  * [1.0.2 (2022-09-24)][120]
  * [1.0.1 (2022-09-12)][121]
  * [1.0.0 (2022-08-21)][122]
  * [0.2.11 (2022-08-20)][123]
  * [0.2.10 (2022-08-08)][124]
  * [0.2.9 (2022-07-30)][125]
  * [0.2.8 (2022-07-18)][126]
  * [0.2.7 (2022-07-12)][127]
  * [0.2.6 (2022-04-17)][128]
  * [0.2.5 (2022-04-14)][129]
  * [0.2.4 (2022-03-26)][130]
  * [0.2.3 (2022-03-24)][131]
  * [0.2.2 (2022-02-16)][132]
  * [0.2.1 (2022-02-11)][133]
  * [0.2.0 (2022-02-07)][134]
  * [0.1.0 (2021-12-01)][135]
  * [0.0.4 (2021-11-11)][136]
  * [0.0.3 (2021-11-06)][137]
  * [0.0.2 (2021-10-25)][138]
  * [0.0.1 (2021-10-25)][139]
* Owners
* [ [sourcefrog] ][140]
cargo-mutants-26.0.0 is not a library.

# cargo-mutants

[https://github.com/sourcefrog/cargo-mutants][141]

[[Tests]][142] [[crates.io]][143] [[libs.rs]][144] [[GitHub Sponsors]][145] [[Donate]][146]

cargo-mutants helps you improve your program's quality by finding places where bugs could be
inserted without causing any tests to fail.

Coverage measurements can be helpful, but they really tell you what code is *reached* by a test, and
not whether the test really *checks* anything about the behavior of the code. Mutation tests give
different information, about whether the tests really check the code's behavior.

The goal of cargo-mutants is to be *easy* to run on any Rust source tree, and to tell you something
*interesting* about areas where bugs might be lurking or the tests might be insufficient.

**For more background, see the [slides][147] and [video][148] from my Rustconf 2024 talk.**

**The main documentation is the user guide at [https://mutants.rs/][149].**

## Prerequisites

cargo-mutants can help on trees with non-flaky tests that run under `cargo test` or [`cargo nextest
run`][150].

## Install

`cargo install --locked cargo-mutants
`

You can also install using [cargo-binstall][151] or from binaries attached to GitHub releases.

## Quick start

From within a Rust source directory, just run

`cargo mutants
`

To generate mutants in only one file:

`cargo mutants -f src/something.rs
`

## Integration with CI

The [manual includes instructions and examples for automatically testing mutants in CI][152],
including incremental testing of pull requests and full testing of the development branch.

## Help advance cargo-mutants

If you use cargo-mutants or just like the idea you can help it get better:

* [Post an experience report in GitHub discussions][153], saying whether it worked, failed, found
  interesting results, etc.
* [Sponsor development][154]

## Project status

As of August 2025 this is an actively-maintained spare time project. I expect to make
[releases][155] about every one or two months.

It's very usable at it is and there's room for lots more future improvement, especially in adding
new types of mutation.

If you try it out on your project, [I'd love to hear back in a github discussion][156] whether it
worked well or what could be better:

* Did it work on your tree? Did you need to set any options or do any debugging to get it working?
* Did it find meaningful gaps in testing? Where there too many false positives?
* What do you think would make it better or easier?

This software is provided as-is with no warranty of any kind.

## Further reading

See also:

* [cargo-mutants manual][157]
* [How cargo-mutants compares to other techniques and tools][158].
* [Design notes][159]
* [Contributing][160]
* [Release notes][161]
* [Discussions][162]

[1]: /
[2]: /crate/cargo-mutants/latest
[3]: #
[4]: /about
[5]: /about/badges
[6]: /about/builds
[7]: /about/metadata
[8]: /about/redirections
[9]: /about/download
[10]: /about/rustdoc-json
[11]: /releases/queue
[12]: https://foundation.rust-lang.org/policies/privacy-policy/#docs.rs
[13]: #
[14]: https://www.rust-lang.org/
[15]: https://doc.rust-lang.org/book/
[16]: https://doc.rust-lang.org/std/
[17]: https://doc.rust-lang.org/rust-by-example/
[18]: https://doc.rust-lang.org/cargo/guide/
[19]: https://doc.rust-lang.org/nightly/clippy
[20]: /crate/cargo-mutants/latest
[21]: /crate/cargo-mutants/latest/source/
[22]: /crate/cargo-mutants/latest/builds
[23]: /crate/cargo-mutants/latest/features
[24]: https://mutants.rs/
[25]: https://github.com/sourcefrog/cargo-mutants
[26]: https://crates.io/crates/cargo-mutants
[27]: /crate/anyhow/^1.0.86
[28]: /crate/camino/^1.1.9
[29]: /crate/cargo_metadata/^0.19
[30]: /crate/clap/^4.5
[31]: /crate/clap_complete/^4
[32]: /crate/color-print/^0.3
[33]: /crate/console/^0.15
[34]: /crate/ctrlc/^3.4
[35]: /crate/fastrand/^2
[36]: /crate/fs2/^0.4
[37]: /crate/globset/^0.4.16
[38]: /crate/ignore/^0.4.23
[39]: /crate/indoc/^2.0.0
[40]: /crate/itertools/^0.13
[41]: /crate/jiff/^0.1
[42]: /crate/jobserver/^0.1
[43]: /crate/mutants/^0.0.3
[44]: /crate/nextest-metadata/^0.12.1
[45]: /crate/num_cpus/^1.16
[46]: /crate/nutmeg/^0.1.5
[47]: /crate/patch/^0.7
[48]: /crate/path-slash/^0.2
[49]: /crate/proc-macro2/^1.0.91
[50]: /crate/quote/^1.0.35
[51]: /crate/reflink/^0.1
[52]: /crate/regex/^1.10
[53]: /crate/schemars/^0.9
[54]: /crate/serde/^1.0.209
[55]: /crate/serde_json/^1.0.128
[56]: /crate/similar/^2.1
[57]: /crate/strum/^0.26
[58]: /crate/syn/^2.0.104
[59]: /crate/tempfile/^3.20
[60]: /crate/test-log/^0.2.16
[61]: /crate/toml/^0.8
[62]: /crate/tracing/^0.1.41
[63]: /crate/tracing-subscriber/^0.3.20
[64]: /crate/whoami/^1.5
[65]: /crate/assert_cmd/^2.0
[66]: /crate/assert_matches/^1.5
[67]: /crate/cp_r/^0.5.2
[68]: /crate/insta/^1.12
[69]: /crate/lazy_static/^1.4
[70]: /crate/predicates/^3
[71]: /crate/pretty_assertions/^1
[72]: /crate/rusty-fork/^0.3.1
[73]: /crate/walkdir/^2.5
[74]: /crate/nix/^0.30
[75]: /crate/cargo-mutants/26.0.0
[76]: /crate/cargo-mutants/25.3.1
[77]: /crate/cargo-mutants/25.3.0
[78]: /crate/cargo-mutants/25.2.2
[79]: /crate/cargo-mutants/25.2.2-pre2
[80]: /crate/cargo-mutants/25.2.2-pre0
[81]: /crate/cargo-mutants/25.2.1
[82]: /crate/cargo-mutants/25.2.0
[83]: /crate/cargo-mutants/25.1.0
[84]: /crate/cargo-mutants/25.0.1
[85]: /crate/cargo-mutants/25.0.1-pre3
[86]: /crate/cargo-mutants/25.0.1-pre2
[87]: /crate/cargo-mutants/25.0.0
[88]: /crate/cargo-mutants/24.11.2
[89]: /crate/cargo-mutants/24.11.1
[90]: /crate/cargo-mutants/24.11.0
[91]: /crate/cargo-mutants/24.9.0
[92]: /crate/cargo-mutants/24.7.1
[93]: /crate/cargo-mutants/24.7.0
[94]: /crate/cargo-mutants/24.5.0
[95]: /crate/cargo-mutants/24.4.0
[96]: /crate/cargo-mutants/24.3.0
[97]: /crate/cargo-mutants/24.2.1
[98]: /crate/cargo-mutants/24.2.0
[99]: /crate/cargo-mutants/24.1.2
[100]: /crate/cargo-mutants/24.1.1
[101]: /crate/cargo-mutants/24.1.0
[102]: /crate/cargo-mutants/23.12.2
[103]: /crate/cargo-mutants/23.12.0
[104]: /crate/cargo-mutants/23.11.2
[105]: /crate/cargo-mutants/23.11.1
[106]: /crate/cargo-mutants/23.11.0
[107]: /crate/cargo-mutants/23.10.0
[108]: /crate/cargo-mutants/23.9.1
[109]: /crate/cargo-mutants/23.9.0
[110]: /crate/cargo-mutants/23.6.0
[111]: /crate/cargo-mutants/23.5.0
[112]: /crate/cargo-mutants/1.2.3
[113]: /crate/cargo-mutants/1.2.2
[114]: /crate/cargo-mutants/1.2.1
[115]: /crate/cargo-mutants/1.2.0
[116]: /crate/cargo-mutants/1.1.1
[117]: /crate/cargo-mutants/1.1.1-pre
[118]: /crate/cargo-mutants/1.1.0
[119]: /crate/cargo-mutants/1.0.3
[120]: /crate/cargo-mutants/1.0.2
[121]: /crate/cargo-mutants/1.0.1
[122]: /crate/cargo-mutants/1.0.0
[123]: /crate/cargo-mutants/0.2.11
[124]: /crate/cargo-mutants/0.2.10
[125]: /crate/cargo-mutants/0.2.9
[126]: /crate/cargo-mutants/0.2.8
[127]: /crate/cargo-mutants/0.2.7
[128]: /crate/cargo-mutants/0.2.6
[129]: /crate/cargo-mutants/0.2.5
[130]: /crate/cargo-mutants/0.2.4
[131]: /crate/cargo-mutants/0.2.3
[132]: /crate/cargo-mutants/0.2.2
[133]: /crate/cargo-mutants/0.2.1
[134]: /crate/cargo-mutants/0.2.0
[135]: /crate/cargo-mutants/0.1.0
[136]: /crate/cargo-mutants/0.0.4
[137]: /crate/cargo-mutants/0.0.3
[138]: /crate/cargo-mutants/0.0.2
[139]: /crate/cargo-mutants/0.0.1
[140]: https://crates.io/users/sourcefrog
[141]: https://github.com/sourcefrog/cargo-mutants
[142]: https://github.com/sourcefrog/cargo-mutants/actions/workflows/tests.yml?query=branch%3Amain
[143]: https://crates.io/crates/cargo-mutants
[144]: https://lib.rs/crates/cargo-mutants
[145]: https://github.com/sponsors/sourcefrog
[146]: https://donate.stripe.com/fZu6oH6ry9I6epVfF7dfG00
[147]: https://docs.google.com/presentation/d/1YDwHz6ysRRNYRDtv80EMRAs4FQu2KKQ-IbGu2jrqswY/edit?pli=
1&slide=id.g2876539b71f_0_0
[148]: https://www.youtube.com/watch?v=PjDHe-PkOy8&pp=ygUNY2FyZ28tbXV0YW50cw%3D%3D
[149]: https://mutants.rs/
[150]: https://nexte.st/
[151]: https://github.com/cargo-bins/cargo-binstall
[152]: https://mutants.rs/ci.html
[153]: https://github.com/sourcefrog/cargo-mutants/discussions
[154]: https://github.com/sponsors/sourcefrog
[155]: https://github.com/sourcefrog/cargo-mutants/releases
[156]: https://github.com/sourcefrog/cargo-mutants/discussions/categories/general
[157]: https://mutants.rs/
[158]: https://github.com/sourcefrog/cargo-mutants/wiki/Compared
[159]: DESIGN.md
[160]: CONTRIBUTING.md
[161]: NEWS.md
[162]: https://github.com/sourcefrog/cargo-mutants/discussions
