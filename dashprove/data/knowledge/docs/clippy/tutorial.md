# [Usage][1]

This chapter describes how to use Clippy to get the most out of it. Clippy can be used as a `cargo`
subcommand or, like `rustc`, directly with the `clippy-driver` binary.

> *Note:* This chapter assumes that you have Clippy installed already. If you're not sure, take a
> look at the [Installation][2] chapter.

## [Cargo subcommand][3]

The easiest and most common way to run Clippy is through `cargo`. To do that, just run

`cargo clippy
`

### [Lint configuration][4]

The above command will run the default set of lints, which are included in the lint group
`clippy::all`. You might want to use even more lints, or you may not agree with every Clippy lint,
and for that there are ways to configure lint levels.

> *Note:* Clippy is meant to be used with a generous sprinkling of `#[allow(..)]`s through your
> code. So if you disagree with a lint, don't feel bad disabling them for parts of your code or the
> whole project.

#### [Command line][5]

You can configure lint levels on the command line by adding `-A/W/D clippy::lint_name` like this:

`cargo clippy -- -Aclippy::style -Wclippy::box_default -Dclippy::perf
`

For [CI][6] all warnings can be elevated to errors which will in turn fail the build and cause
Clippy to exit with a code other than `0`.

`cargo clippy -- -Dwarnings
`

> *Note:* Adding `-D warnings` will cause your build to fail if **any** warnings are found in your
> code. That includes warnings found by rustc (e.g. `dead_code`, etc.).

For more information on configuring lint levels, see the [rustc documentation][7].

#### [Even more lints][8]

Clippy has lint groups which are allow-by-default. This means, that you will have to enable the
lints in those groups manually.

For a full list of all lints with their description and examples, please refer to [Clippy's lint
list][9]. The two most important allow-by-default groups are described below:

[`clippy::pedantic`][10]

The first group is the `pedantic` group. This group contains really opinionated lints, that may have
some intentional false positives in order to prevent false negatives. So while this group is ready
to be used in production, you can expect to sprinkle multiple `#[allow(..)]`s in your code. If you
find any false positives, you're still welcome to report them to us for future improvements.

> FYI: Clippy uses the whole group to lint itself.

[`clippy::restriction`][11]

The second group is the `restriction` group. This group contains lints that "restrict" the language
in some way. For example the `clippy::unwrap` lint from this group won't allow you to use
`.unwrap()` in your code. You may want to look through the lints in this group and enable the ones
that fit your need.

> *Note:* You shouldn't enable the whole lint group, but cherry-pick lints from this group. Some
> lints in this group will even contradict other Clippy lints!

#### [Too many lints][12]

The most opinionated warn-by-default group of Clippy is the `clippy::style` group. Some people
prefer to disable this group completely and then cherry-pick some lints they like from this group.
The same is of course possible with every other of Clippy's lint groups.

> *Note:* We try to keep the warn-by-default groups free from false positives (FP). If you find that
> a lint wrongly triggers, please report it in an issue (if there isn't an issue for that FP
> already)

#### [Source Code][13]

You can configure lint levels in source code the same way you can configure `rustc` lints:

`#![allow(clippy::style)]

#[warn(clippy::box_default)]
fn main() {
    let _ = Box::<String>::new(Default::default());
    // ^ warning: `Box::new(_)` of default value
}`

### [Automatically applying Clippy suggestions][14]

Clippy can automatically apply some lint suggestions, just like the compiler. Note that `--fix`
implies `--all-targets`, so it can fix as much code as it can.

`cargo clippy --fix
`

### [Workspaces][15]

All the usual workspace options should work with Clippy. For example the following command will run
Clippy on the `example` crate in your workspace:

`cargo clippy -p example
`

As with `cargo check`, this includes dependencies that are members of the workspace, like path
dependencies. If you want to run Clippy **only** on the given crate, use the `--no-deps` option like
this:

`cargo clippy -p example -- --no-deps
`

## [Using Clippy without `cargo`: `clippy-driver`][16]

Clippy can also be used in projects that do not use cargo. To do so, run `clippy-driver` with the
same arguments you use for `rustc`. For example:

`clippy-driver --edition 2018 -Cpanic=abort foo.rs
`

> *Note:* `clippy-driver` is designed for running Clippy and should not be used as a general
> replacement for `rustc`. `clippy-driver` may produce artifacts that are not optimized as expected,
> for example.

[1]: #usage
[2]: installation.html
[3]: #cargo-subcommand
[4]: #lint-configuration
[5]: #command-line
[6]: continuous_integration/index.html
[7]: https://doc.rust-lang.org/rustc/lints/levels.html#configuring-warning-levels
[8]: #even-more-lints
[9]: https://rust-lang.github.io/rust-clippy/master/index.html
[10]: #clippypedantic
[11]: #clippyrestriction
[12]: #too-many-lints
[13]: #source-code
[14]: #automatically-applying-clippy-suggestions
[15]: #workspaces
[16]: #using-clippy-without-cargo-clippy-driver
