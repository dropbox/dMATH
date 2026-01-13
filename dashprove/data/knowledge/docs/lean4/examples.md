# mathlib4

[[GitHub CI]][1] [[Bors enabled]][2] [[project chat]][3] [[Gitpod Ready-to-Code]][4]

[Mathlib][5] is a user maintained library for the [Lean theorem prover][6]. It contains both
programming infrastructure and mathematics, as well as tactics that use the former and allow to
develop the latter.

## Installation

You can find detailed instructions to install Lean, mathlib, and supporting tools on [our
website][7]. Alternatively, click on one of the buttons below to open a GitHub Codespace or a Gitpod
workspace containing the project.

[[Open in GitHub Codespaces]][8]

[[Open in Gitpod]][9]

## Using `mathlib4` as a dependency

Please refer to
[https://github.com/leanprover-community/mathlib4/wiki/Using-mathlib4-as-a-dependency][10]

## Experimenting

Got everything installed? Why not start with the [tutorial project][11]?

For more pointers, see [Learning Lean][12].

## Documentation

Besides the installation guides above and [Lean's general documentation][13], the documentation of
mathlib consists of:

* [The mathlib4 docs][14]: documentation [generated automatically][15] from the source `.lean`
  files.
* A description of [currently covered theories][16], as well as an [overview][17] for
  mathematicians.
* Some [extra Lean documentation][18] not specific to mathlib (see "Miscellaneous topics")
* Documentation for people who would like to [contribute to mathlib][19]

Much of the discussion surrounding mathlib occurs in a [Zulip chat room][20], and you are welcome to
join, or read along without signing up. Questions from users at all levels of expertise are welcome!
We also provide an [archive of the public discussions][21], which is useful for quick reference.

## Contributing

The complete documentation for contributing to `mathlib` is located [on the community guide
contribute to mathlib][22]

You may want to subscribe to the `mathlib4` channel on [Zulip][23] to introduce yourself and your
plan to the community. Often you can find community members willing to help you get started and
advise you on the fit and feasibility of your project.

* To obtain precompiled `olean` files, run `lake exe cache get`. (Skipping this step means the next
  step will be very slow.)
* To build `mathlib4` run `lake build`.
* To build and run all tests, run `lake test`.
* You can use `lake build Mathlib.Import.Path` to build a particular file, e.g. `lake build
  Mathlib.Algebra.Group.Defs`.
* If you added a new file, run the following command to update `Mathlib.lean`
  
  lake exe mk_all

### Guidelines

Mathlib has the following guidelines and conventions that must be followed

* The [style guide][24]
* A guide on the [naming convention][25]
* The [documentation style][26]

### Downloading cached build files

You can run `lake exe cache get` to download cached build files that are computed by `mathlib4`'s
automated workflow.

If something goes mysteriously wrong, you can try one of `lake clean` or `rm -rf .lake` before
trying `lake exe cache get` again. In some circumstances you might try `lake exe cache get!` which
re-downloads cached build files even if they are available locally.

Call `lake exe cache` to see its help menu.

### Building HTML documentation

The [mathlib4_docs repository][27] is responsible for generating and publishing the [mathlib4
docs][28].

That repo can be used to build the docs locally:

git clone https://github.com/leanprover-community/mathlib4_docs.git
cd mathlib4_docs
cp ../mathlib4/lean-toolchain .
lake exe cache get
lake build Mathlib:docs

The last step may take a while (>20 minutes). The HTML files can then be found in `.lake/build/doc`.

## Transitioning from Lean 3

For users familiar with Lean 3 who want to get up to speed in Lean 4 and migrate their existing Lean
3 code we have:

* A [survival guide][29] for Lean 3 users
* [Instructions to run `mathport`][30] on a project other than mathlib. `mathport` is the tool the
  community used to port the entirety of `mathlib` from Lean 3 to Lean 4.

### Dependencies

If you are a mathlib contributor and want to update dependencies, use `lake update`, or `lake update
batteries aesop` (or similar) to update a subset of the dependencies. This will update the
`lake-manifest.json` file correctly. You will need to make a PR after committing the changes to this
file.

Please do not run `lake update -Kdoc=on` as previously advised, as the documentation related
dependencies should only be included when CI is building documentation.

## Maintainers:

For a list containing more detailed information, see
[https://leanprover-community.github.io/teams/maintainers.html][31]

* Anne Baanen (@Vierkantor): algebra, number theory, tactics
* Matthew Robert Ballard (@mattrobball): algebra, algebraic geometry, category theory
* Riccardo Brasca (@riccardobrasca): algebra, number theory, algebraic geometry, category theory
* Kevin Buzzard (@kbuzzard): algebra, number theory, algebraic geometry, category theory
* Mario Carneiro (@digama0): lean formalization, tactics, type theory, proof engineering
* Bryan Gin-ge Chen (@bryangingechen): documentation, infrastructure
* Johan Commelin (@jcommelin): algebra, number theory, category theory, algebraic geometry
* Anatole Dedecker (@ADedecker): topology, functional analysis, calculus
* Rémy Degenne (@RemyDegenne): probability, measure theory, analysis
* Floris van Doorn (@fpvandoorn): measure theory, model theory, tactics
* Frédéric Dupuis (@dupuisf): linear algebra, functional analysis
* Sébastien Gouëzel (@sgouezel): topology, calculus, geometry, analysis, measure theory
* Markus Himmel (@TwoFX): category theory
* Yury G. Kudryashov (@urkud): analysis, topology, measure theory
* Robert Y. Lewis (@robertylewis): tactics, documentation
* Jireh Loreaux (@j-loreaux): analysis, topology, operator algebras
* Heather Macbeth (@hrmacbeth): geometry, analysis
* Patrick Massot (@patrickmassot): documentation, topology, geometry
* Bhavik Mehta (@b-mehta): category theory, combinatorics
* Kyle Miller (@kmill): combinatorics, tactics, metaprogramming
* Kim Morrison (@kim-em): category theory, tactics
* Oliver Nash (@ocfnash): algebra, geometry, topology
* Joël Riou (@joelriou): category theory, homology, algebraic geometry
* Michael Rothgang (@grunweg): differential geometry, analysis, topology, linters
* Damiano Testa (@adomani): algebra, algebraic geometry, number theory, tactics, linter
* Adam Topaz (@adamtopaz): algebra, category theory, algebraic geometry
* Eric Wieser (@eric-wieser): algebra, infrastructure

## Past maintainers:

* Jeremy Avigad (@avigad): analysis
* Reid Barton (@rwbarton): category theory, topology
* Gabriel Ebner (@gebner): tactics, infrastructure, core, formal languages
* Johannes Hölzl (@johoelzl): measure theory, topology
* Simon Hudon (@cipher1024): tactics
* Chris Hughes (@ChrisHughes24): algebra

[1]: https://github.com/leanprover-community/mathlib4/actions/workflows/build.yml/badge.svg?branch=m
aster
[2]: https://mathlib-bors-ca18eefec4cb.herokuapp.com/repositories/16
[3]: https://leanprover.zulipchat.com
[4]: https://gitpod.io/#https://github.com/leanprover-community/mathlib4
[5]: https://leanprover-community.github.io
[6]: https://leanprover.github.io
[7]: https://leanprover-community.github.io/get_started.html
[8]: https://codespaces.new/leanprover-community/mathlib4
[9]: https://gitpod.io/#https://github.com/leanprover-community/mathlib4
[10]: https://github.com/leanprover-community/mathlib4/wiki/Using-mathlib4-as-a-dependency
[11]: https://leanprover-community.github.io/install/project.html
[12]: https://leanprover-community.github.io/learn.html
[13]: https://docs.lean-lang.org/lean4/doc/
[14]: https://leanprover-community.github.io/mathlib4_docs/index.html
[15]: https://github.com/leanprover/doc-gen4
[16]: https://leanprover-community.github.io/theories.html
[17]: https://leanprover-community.github.io/mathlib-overview.html
[18]: https://leanprover-community.github.io/learn.html
[19]: https://leanprover-community.github.io/contribute/index.html
[20]: https://leanprover.zulipchat.com/
[21]: https://leanprover-community.github.io/archive/
[22]: https://leanprover-community.github.io/contribute/index.html
[23]: https://leanprover.zulipchat.com/
[24]: https://leanprover-community.github.io/contribute/style.html
[25]: https://leanprover-community.github.io/contribute/naming.html
[26]: https://leanprover-community.github.io/contribute/doc.html
[27]: https://github.com/leanprover-community/mathlib4_docs
[28]: https://leanprover-community.github.io/mathlib4_docs/index.html
[29]: https://github.com/leanprover-community/mathlib4/wiki/Lean-4-survival-guide-for-Lean-3-users
[30]: https://github.com/leanprover-community/mathport#running-on-a-project-other-than-mathlib
[31]: https://leanprover-community.github.io/teams/maintainers.html
