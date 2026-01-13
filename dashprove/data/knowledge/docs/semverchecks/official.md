[[semver-checks]][1]

# cargo-semver-checks

Lint your crate API changes for semver violations.

* [Quick Start][2]
* [FAQ][3]
* [Configuration][4]
* [Troubleshooting][5]
* [Contributing][6]

## Quick Start

# If you already use `cargo-binstall` for faster tool installations:
$ cargo binstall cargo-semver-checks

# Otherwise:
$ cargo install cargo-semver-checks --locked

# Lint a new release for SemVer breakage before `cargo publish`:
$ cargo semver-checks

Or use as a [GitHub Action][7] (used in `.github/workflows/ci.yml` in this repo):

- name: Check semver
  uses: obi1kenobi/cargo-semver-checks-action@v2

[[image]][8]

Each failing check references specific items in the [Cargo SemVer reference][9] or other reference
pages, as appropriate. It also includes the item name and file location that are the cause of the
problem, as well as a link to the implementation of that query in the current version of the tool.

## FAQ

* [What Rust versions does `cargo-semver-checks` support?][10]
* [Can I use `cargo-semver-checks` with `nightly` Rust?][11]
* [What if my project needs stronger guarantees around supported Rust versions?][12]
* [Does the crate I'm checking have to be published on crates.io?][13]
  
  * [Git repository detection and configuration][14]
  * [Use with jujutsu][15]
* [What features does `cargo-semver-checks` enable in the tested crates?][16]
* [My crate uses `--cfg` conditional compilation. Can `cargo-semver-checks` scan it?][17]
* [Does `cargo-semver-checks` have false positives?][18]
* [Will `cargo-semver-checks` catch every semver violation?][19]
* [Can I configure individual lints?][20]
* [If I really want a new feature to be implemented, can I sponsor its development?][21]
* [How is `cargo-semver-checks` similar to and different from other tools?][22]
* [Why is it sometimes `cargo-semver-check` and `cargo-semver-checks`?][23]
* [What is the MSRV policy with respect to semver?][24]

### What Rust versions does `cargo-semver-checks` support?

`cargo-semver-checks` uses the rustdoc tool to analyze the crate's API. Rustdoc's JSON output format
isn't stable, and can have breaking changes in new Rust versions.

When each `cargo-semver-checks` version is released, it will at minimum include support for the
then-current stable and beta Rust versions. It may, but is not guaranteed to, additionally support
some nightly Rust versions.

[The GitHub Action][25] by default uses the most recent versions of both `cargo-semver-checks` and
stable Rust, so it should be unaffected. Users using `cargo-semver-checks` in other ways are
encouraged to update `cargo-semver-checks` when updating Rust versions to ensure continued
compatibility.

### Can I use `cargo-semver-checks` with `nightly` Rust?

Support for `nightly` Rust versions is on a best-effort basis. It will work *often, but not always*.
If you *must* use `nightly`, it's strongly recommended to pin to a specific `nightly` version to
avoid broken workflows.

`cargo-semver-checks` relies on the rustdoc JSON format, which is unstable and changes often. After
a new rustdoc JSON format version gets shipped in `nightly`, it usually takes several days to
several weeks for it to be supported in a new `cargo-semver-checks`, during which time it is not
possible to use `cargo-semver-checks` with those `nightly` versions.

It's also possible that support for some `nightly` versions may be dropped even while older stable
versions are still supported. This usually happens when a rustdoc format gets superseded by a newer
version before becoming part of any stable Rust. In that case, we may drop support for that format
to conserve maintenance bandwidth and speed up compile times. For example, `cargo-semver-checks`
v0.24 [supported][26] rustdoc formats v24, v26, and v27, but did not support the nightly-only v25
format.

### What if my project needs stronger guarantees around supported Rust versions?

If you'd like extended support for older Rust versions, or an SLA on supporting new `nightly`
releases, we're happy to offer those *on a commercial basis*. It could be in the form of a formal
support contract, or something as simple as discussing expectations over email and setting up a
recurring GitHub sponsorship for an agreed-upon amount.

Please reach out at the email in the [Cargo.toml][27] and let us know about what projects this is
for and what their needs are.

### Does the crate I'm checking have to be published on crates.io?

No, it does not have to be published anywhere. You'll just need to use a flag to help
`cargo-semver-checks` locate the version to use as a baseline for semver-checking.

By default, `cargo-semver-checks` uses crates.io to look up the previous version of the crate, which
is used as the baseline for semver-checking the current version of the crate. The following flags
can be used to explicitly specify a baseline instead:

`--baseline-version <X.Y.Z>
    Version from registry to lookup for a baseline

--baseline-rev <REV>
    Git revision to lookup for a baseline

--baseline-root <MANIFEST_ROOT>
    Directory containing baseline crate source

--baseline-rustdoc <JSON_PATH>
    The rustdoc json file to use as a semver baseline
`

Custom registries are not currently supported ([#160][28]), so crates published on registries other
than crates.io should use one of the other approaches of generating the baseline.

#### Git repository detection and configuration

When looking up a git revision with `--baseline-rev`, `cargo-semver-checks` will walk up the current
directory until it finds a `.git/` directory to resolve the revision and extract the corresponding
worktree. You can use the following environment variables to influence git repository detection:

* [`GIT_DIR`][29]: explicitly set the location of the `.git/` directory. If the value is a relative
  path, it is resolved from the current working directory. If this is not set, `cargo-semver-checks`
  will search the current working directory and any of its parents for the `.git/` directory.
* [`GIT_CEILING_DIRECTORIES`][30]: a colon (`:`) separated list of absolute paths which
  `cargo-semver-checks` should not search for the `.git/` directory. This can be used to prevent
  `cargo-semver-checks` from searching slow network mounted directories. This environment variable
  cannot be used to prevent searching the current working directory, or a directory explicitly set
  with [`GIT_DIR`][31]. If this is not set, `cargo-semver-checks` will search the current working
  directory and any of its parents (up to the root directory).
* [`GIT_DISCOVERY_ACROSS_FILESYSTEM`][32]: enable git repository detection across filesystems. By
  default, `cargo-semver-checks` will not cross filesystem boundaries when searching for the `.git/`
  directory. Set this environment variable to `true` to enable cross-filesystem detection. This
  environment variable has no effect when setting [`GIT_DIR`][33] explicitly.

#### Use with jujutsu

Setting the `GIT_DIR` environment variable allows using `cargo-semver-checks` with a non-colocated
[jujutsu][34] repository.

GIT_DIR="$(jj root)/.jj/repo/store/git" cargo semver-checks --package "foo" --baseline-rev "foo/v0.1
.0"

### What features does `cargo-semver-checks` enable in the tested crates?

By default, checking is done on all features except features named `unstable`, `nightly`, `bench`,
`no_std`, or ones with prefix `_`, `unstable-`, or `unstable_`, as such names are commonly used for
private or unstable features.

This behaviour can be overriden. Checked feature set can be changed to:

* *all* the features, selected with `--all-features`,
* only the crate's default features, selected with `--default-features`,
* empty set, selected with `--only-explicit-features`.

Additionally, features can be enabled one-by-one, using flags `--features`, `--baseline-features`
and `--current-features`.

For example, consider crate [serde][35], with the following features (per v1.0.163):

* `std` - the crate's only default feature,
* `alloc`, `derive`, `rc` - optional features,
* `unstable` - a feature that possibly breaks semver.

────────────────────────────────┬────────────────────┬──────────────────────────────────────────────
used flags                      │selected feature set│explanation                                   
────────────────────────────────┼────────────────────┼──────────────────────────────────────────────
none                            │`std`, `alloc`,     │Feature `unstable` is excluded by the default 
                                │`derive`, `rc`      │heuristic.                                    
────────────────────────────────┼────────────────────┼──────────────────────────────────────────────
`--features unstable`           │`std`, `alloc`,     │The flag explicitly adds `unstable` to the    
                                │`derive`, `rc`,     │heuristic's selections.                       
                                │`unstable`          │                                              
────────────────────────────────┼────────────────────┼──────────────────────────────────────────────
`--all-features`                │`std`, `alloc`,     │All the features are used, disabling the      
                                │`derive`, `rc`,     │default heuristic.                            
                                │`unstable`          │                                              
────────────────────────────────┼────────────────────┼──────────────────────────────────────────────
`--default-features`            │`std`               │The crate has only one default feature.       
────────────────────────────────┼────────────────────┼──────────────────────────────────────────────
`--default-features --features  │`std`, `derive`     │Feature `derive` is used along with crate's   
derive`                         │                    │default features.                             
────────────────────────────────┼────────────────────┼──────────────────────────────────────────────
`--only-explicit-features`      │none                │No explicit features are passed.              
────────────────────────────────┼────────────────────┼──────────────────────────────────────────────
`--only-explicit-features       │`unstable`          │All features can be added explicitly,         
--features unstable`            │                    │regardless of their name.                     
────────────────────────────────┴────────────────────┴──────────────────────────────────────────────

### My crate uses `--cfg` conditional compilation. Can `cargo-semver-checks` scan it?

Yes! You can configure the `--cfg` options that `cargo-semver-checks` will use when scanning your
crate by setting them in the `RUSTDOCFLAGS` environment variable.

For example, you can ask `cargo-semver-checks` to enable the `some-option` config by invoking it as:

`RUSTDOCFLAGS="--cfg some-option" cargo semver-checks
`

### Does `cargo-semver-checks` have false positives?

"False positive" means that `cargo-semver-checks` reported a semver violation incorrectly. A design
goal of `cargo-semver-checks` is to not have false positives. If they do occur, they are considered
bugs.

When `cargo-semver-checks` reports a semver violation, it should always point to a specific file and
approximate line number where the specified issue occurs; failure to specify a file and line number
is also considered a bug.

If you think `cargo-semver-checks` might have a false-positive but you aren't sure, please [open an
issue][36]. Semver in Rust has [many non-obvious and tricky edge cases][37], especially [in the
presence of macros][38]. We'd be happy to look into it together with you to determine if it's a
false positive or not.

### Will `cargo-semver-checks` catch every semver violation?

No, it will not — not yet! There are many ways to break semver, and `cargo-semver-checks` [doesn't
yet have lints for all of them][39]. New lints are added frequently, and [we'd be happy to mentor
you][40] if you'd like to contribute new lints!

Append `--verbose` when semver-checking your crate to see the full list of performed semver checks.

Here are some example areas where `cargo-semver-checks` currently will not catch semver violations:

* breaking type changes, for example in the type of a field or function parameter
* breaking changes in generics or lifetimes
* breaking changes that exist when only a subset of all crate features are activated

### Can I configure individual lints?

Yes! See [lint-level configuration][41].

### If I really want a new feature to be implemented, can I sponsor its development?

Depending on the feature, possibly yes!

Please reach out to us *ahead of time* [over email][42] to discuss the scope of the feature and
sponsorship.

It's possible that the feature might be deemed out of scope, too complex to build or maintain, or
otherwise unsuitable. In such cases we'd like to let you know that *before* you've sent us money,
since [there are no refunds on GitHub Sponsors][43].

If the feature is viable and the work involved in building is commensurate to the sponsorship
amount, we'd be happy to build it. At your option, we'd also be happy to give you a shout-out for
sponsoring the feature when it is announced in the release notes.

### How is `cargo-semver-checks` similar to and different from other tools?

[rust semverver][44] builds on top of rustc internals to build rlib's and compare their metadata.
This strips the code down to the basics for identifying changes. However, is tightly coupled to
specific nightly compiler versions and [takes work to stay in sync][45]. As of April 17, 2023, it
appears to be [deprecated and no longer maintained][46].

[cargo breaking][47] effectively runs `cargo expand` and re-parses the code using [`syn`][48] which
requires re-implementing large swaths of rust's semantics to then lint the API for changes. This is
limited to the feature and target the crate was compiled for. As of November 22, 2022, it appears to
be [archived and no longer maintained][49].

`cargo-semver-checks` sources its data from rustdoc's json output. While the json output format is
unstable, the rate of change is fairly low, reducing the churn in keeping up. The lints are also
written as queries for `trustfall` ["query everything" engine][50], reducing the work for creating
and maintaining them. Because of the extra data that rustdoc includes, some level of feature/target
awareness might be able to be introduced.

There is interest in [hosting rustdoc JSON][51] on `docs.rs` meaning that semver-checking could one
day download the baseline rustdoc JSON file instead of generating it. Also, generally speaking,
inspecting JSON data is likely going to be faster than full compilation.

[`cargo-public-api`][52] uses rustdoc, like `cargo-semver-checks`, but focuses more on API diffing
(showing which items has changed) and not API linting (explaining why they have changed and
providing control over what counts).

### Why is it sometimes `cargo-semver-check` and `cargo-semver-checks`?

This crate was intended to be published under the name `cargo-semver-check`, and may indeed one day
be published under that name. Due to [an unfortunate mishap][53], it remains `cargo-semver-checks`
for the time being.

The `cargo_semver_check` name is reserved on crates.io but all its versions are intentionally
yanked. Please use the `cargo-semver-checks` crate instead.

### What is the MSRV policy with respect to semver?

MSRV bumps are *not* considered major changes.

Please go to our [Releases page][54] to find the MSRV of the `cargo-semver-checks` version you are
using.

`cargo-semver-checks` has two Rust version bounds, since it depends on Rust both at compile-time and
at runtime. These versions are *usually but not always* the same:

* The MSRV for *compiling* `cargo-semver-checks` (*"compile MSRV"*) is listed in our `Cargo.toml`
  file. This version governs which Rust version you can `cargo install` the project with, and is
  primarily determined by our dependencies' MSRVs.
* The MSRV for *checking crates* (*"runtime MSRV"*) is the minimum Rust version with which running
  `cargo-semver-checks` will succeed. This is determined based on the rustdoc JSON format versions
  and known bugs in older rustdoc versions.

As much as practically possible, changes to the *runtime MSRV* will come in bumps of the *middle*
number in the version, e.g. `0.24.1 -> 0.25.0` or `1.2.3 -> 1.3.0`.

Changes to the *compile MSRV* may happen in any kind of version bump. As much as practically
possible, we'll aim to make them simultaneously with *runtime MSRV* bumps.

## Configuration

### Lint-level configuration

`cargo-semver-checks` offers the ability to customize which lints are enforced, what SemVer versions
they require, and whether violations of that lint produce `deny / warn / allow` behavior.

As a reminder, a "lint" or "check" is a rule in `cargo-semver-checks` that looks for a specific kind
of issue. For example, the `function_missing` lint looks for functions that no longer exist in a
crate's public API. `cargo-semver-checks` has many dozens of such lints.

Lints may be configured in two ways:

* **lint level:** When a lint finds issues, the lint level controls how `cargo-semver-checks`
  responds. The `deny` level makes the lint a hard error: `cargo-semver-checks` will exit with an
  error and will require a version bump to resolve. The `warn` level will print a warning describing
  the issue, but will not cause `cargo-semver-checks` to exit with an error code — meaning it *will
  not* block CI runs. The `allow` level means the findings of the lint should be silently ignored,
  so the check doesn't even need to be run.
* **required update:** This sets the kind of version bump this check should require (for
  `deny`-level) or suggest (for `warn`) when it spots an issue. For example, the `function_missing`
  lint is `major` by default, so if a public function is removed between versions, the version needs
  a major version bump (e.g., 1.2.3 to 2.0.0 or 0.5.2 to 0.6.0). This can be configured to `major`
  or `minor` (1.2.3 to 1.3.0 or 0.5.2 to 0.5.3). There is no "patch" setting since that is
  equivalent to setting an `allow` lint level.

To configure the level and/or required update for a lint, first find its name. This will be in
`snake_case` and is reported in the CLI on errors/warnings, and can also be found as the file name
in the [lints folder][55].

* [Example: Downgrading an error-level lint to a warning][56]
* [Example: Changing the SemVer requirement for a lint][57]
* [Example: Configuring lints for an entire workspace][58]
* [Example: Overriding workspace configuration][59]
* [Common configurations: Make `#[must_use]` lints warn-only][60]
* [Common configurations: Disable `#[must_use]` lints entirely][61]
* [Implementation details & limitations][62]

#### Example: Downgrading an error-level lint to a warning

`cargo-semver-checks` by default considers adding `#[must_use]` on an existing function to require a
`minor` version bump, and has level `deny`. Let's say our package considers that too strict, and
would prefer adding `#[must_use]` to produce a warning instead of an error.

We can accomplish that by adding the following to the `Cargo.toml` manifest for the package:

[package.metadata.cargo-semver-checks.lints]
function_must_use_added = "warn"

That option is shorthand notation for:

[package.metadata.cargo-semver-checks.lints]
function_must_use_added = { level = "warn" }

If we wanted to configure other lints simultaneously, we could add their configuration there as
well: one lint per line.

#### Example: Changing the SemVer requirement for a lint

Say we instead wanted to mandate a major version bump if `#[must_use]` is added.

We'd add the following to our package's `Cargo.toml` file:

[package.metadata.cargo-semver-checks.lints]
function_must_use_added = { required-update = "major" }

Of course, it's possible to combine `level` and `required-update` settings. For example, the
following will make the `function_must_use_added` cause warnings (instead of errors) whenever
`#[must_use]` is added without a major version bump:

[package.metadata.cargo-semver-checks.lints]
function_must_use_added = { level = "warn", required-update = "major" }

#### Example: Configuring lints for an entire workspace

`cargo-semver-checks` allows defining your lint configuration at the workspace level, and reusing it
in each of your crates.

First, add your configuration to the workspace root `Cargo.toml`, noting the `workspace.metadata`
prefix instead of `package.metadata`:

[workspace.metadata.cargo-semver-checks.lints]
function_must_use_added = { level = "warn" }

Then, to opt into the workspace configuration in individual packages, add either one of these keys
to that package's `Cargo.toml`:

[package.metadata.cargo-semver-checks.lints]
workspace = true

or, if your workspace already configures workspace-level lints (e.g. for clippy) at
`[workspace.lints]`,

[lints]
workspace = true

Using the `lints.workspace` key will cause a cargo error if it is set and there is no
`[workspace.lints]` table in the workspace Cargo.toml. We support the `lints.workspace` key to ease
our transition toward [merging `cargo-semver-checks` into `cargo` itself][63].

Setting `workspace = false` is not valid configuration for either of these keys. To have a package
opt-out from the workspace's lints configuration, omit both keys entirely from the package's
`Cargo.toml`.

#### Example: Overriding workspace configuration

When `workspace = true` is set, it is still possible to override individual lint settings in a
package.

Rule of thumb: for each config option of each lint,

* the workspace configuration overrides the lint's defaults, and
* the package's configuration overrides both of the above.

For example, if we have in the *workspace* `Cargo.toml`:

[workspace.metadata.cargo-semver-checks.lints]
function_missing = { level = "warn", required-update = "minor" }
trait_now_doc_hidden = "warn"
function_must_use_added = "warn"

and in the *package* `Cargo.toml`:

[package.metadata.cargo-semver-checks.lints]
workspace = true
function_missing = "deny"
trait_now_doc_hidden = { required-update = "minor" }

Here's the final configuration for that package:

* For `function_missing`:
  
  * The [built-in defaults][64] are `level = "deny", required-update = "major"`.
  * The workspace overrides both to `level = "warn", required-update = "minor"`.
  * The package opts into the workspace's configuration with `workspace = true`, then overrides to
    `level = "deny"`.
  * The final config: `level = "deny", required-update = "minor"`.
* For `trait_now_doc_hidden`:
  
  * The [built-in defaults][65] are `level = "deny", required-update = "major"`.
  * The workspace overrides `level = warn"`, leaving `required-update` unchanged.
  * The package opts into the workspace's configuration with `workspace = true`, then sets
    `required-update = "minor"`.
  * The final config: `level = "deny", required-update = "minor"`.
* For `function_must_use_added`:
  
  * The [built-in defaults][66] are `level = "deny", required-update = "minor"`.
  * The workspace overrides `level = "warn"`, leaving `required-update` unchanged.
  * The package opts into the workspace's configuration with `workspace = true`.
  * The final config: `level = "warn", required-update = "minor"`.

#### Common configurations: Make `#[must_use]` lints warn-only

In the default configuration, `cargo-semver-checks` considers it an error to add `#[must_use]`
attributes in patch versions. The rationale is that such an addition risks introducing new lints in
downstream projects, and many projects consider lints as errors and may be broken as a result.
(Whether that setting is good practice for widespread use or not is outside the scope of
`cargo-semver-checks`.)

To downgrade all lints related to `#[must_use]` from error to warnings, add the following lines to
the `cargo-semver-checks` configuration in your package or workspace:

function_must_use_added = "warn"
inherent_method_must_use_added = "warn"
struct_must_use_added = "warn"
enum_must_use_added = "warn"
trait_must_use_added = "warn"
union_must_use_added = "warn"

#### Common configurations: Disable `#[must_use]` lints entirely

To skip checking `#[must_use]`-related lints entirely, apply the following configuration to your
package or workspace:

function_must_use_added = "allow"
inherent_method_must_use_added = "allow"
struct_must_use_added = "allow"
enum_must_use_added = "allow"
trait_must_use_added = "allow"
union_must_use_added = "allow"

#### Implementation details & limitations

When checking a package, `cargo-semver-checks` uses the manifest of the current subject's
`Cargo.toml` file (plus its workspace configuration, if opted in).

Configuration set in the *baseline version* of the package (the version being compared *against*,
such as an existing version published on crates.io) is not read and has no effect. This is because
when publishing a new version, it makes sense to use the new version's configuration instead of any
prior configuration.

When the `--manifest-path` option is used to specify the subject package's `Cargo.toml` file, that's
also the file from which configuration is loaded. If that CLI flag is not specified,
`cargo-semver-checks` will by default attempt to find and use a `Cargo.toml` file that belongs to
the current directory.

If `cargo-semver-checks` is executed in a way that skips reading the current manifest (such as with
the `--current-rustdoc` flag), it is currently not possible to configure lints. Interest in, and
progress toward resolving this limitation is tracked in [this issue][67].

## Troubleshooting

This section documents common issues and the best ways to resolve them.

### Running `cargo install cargo-semver-checks --locked` produces an error

**Recommendation**: use [`cargo-binstall`][68] to download a prebuilt binary if one is available for
your platform, rather than compiling from scratch.

Specific errors and their resolutions:

* ["error: failed to run custom build command for `libz-ng-sys vX.Y.Z`"][69]
  
  * **Preferred resolution**: use a prebuilt binary to avoid this problem entirely.
  * This error is caused by missing `cmake` on your system, which is required by a transitive
    dependency of `cargo-semver-checks`. `cargo` does not currently offer a mechanism for such a
    binary dependency to be declared as required, nor automatically installed when needed.
  * You can usually install it via a command like `apt install cmake` or `brew install cmake`,
    depending on your platform.

### Visual Design

Logo by [NUMI][70]:

[[NUMI Logo]][71]

### License

Available under the Apache License (Version 2.0) or the MIT license, at your option.

Copyright 2022-present Predrag Gruevski and Project Contributors. The present date is determined by
the timestamp of the most recent commit in the repository. Project Contributors are all authors and
committers of commits in the repository.

[1]: /obi1kenobi/cargo-semver-checks/blob/main/brand/banner.png
[2]: #quick-start
[3]: #faq
[4]: #configuration
[5]: #troubleshooting
[6]: https://github.com/obi1kenobi/cargo-semver-checks/blob/main/CONTRIBUTING.md
[7]: https://github.com/obi1kenobi/cargo-semver-checks-action
[8]: https://user-images.githubusercontent.com/2348618/180127698-240e4bed-5581-4cbd-9f47-038affbc4a3
e.png
[9]: https://doc.rust-lang.org/cargo/reference/semver.html
[10]: #what-rust-versions-does-cargo-semver-checks-support
[11]: #can-i-use-cargo-semver-checks-with-nightly-rust
[12]: #what-if-my-project-needs-stronger-guarantees-around-supported-rust-versions
[13]: #does-the-crate-im-checking-have-to-be-published-on-cratesio
[14]: #git-repository-detection-and-configuration
[15]: #use-with-jujutsu
[16]: #what-features-does-cargo-semver-checks-enable-in-the-tested-crates
[17]: #my-crate-uses---cfg-conditional-compilation-can-cargo-semver-checks-scan-it
[18]: #does-cargo-semver-checks-have-false-positives
[19]: #will-cargo-semver-checks-catch-every-semver-violation
[20]: #can-i-configure-individual-lints
[21]: #if-i-really-want-a-new-feature-to-be-implemented-can-i-sponsor-its-development
[22]: #how-is-cargo-semver-checks-similar-to-and-different-from-other-tools
[23]: #why-is-it-sometimes-cargo-semver-check-and-cargo-semver-checks
[24]: #what-is-the-msrv-policy-with-respect-to-semver
[25]: https://github.com/obi1kenobi/cargo-semver-checks-action
[26]: https://github.com/obi1kenobi/cargo-semver-checks/blob/93baddc2a65a5055117ca670fe156dc761403fa
5/Cargo.toml#L18
[27]: https://github.com/obi1kenobi/cargo-semver-checks/blob/93baddc2a65a5055117ca670fe156dc761403fa
5/Cargo.toml#L5
[28]: https://github.com/obi1kenobi/cargo-semver-checks/issues/160
[29]: https://git-scm.com/docs/git.html#Documentation/git.txt-codeGITDIRcode
[30]: https://git-scm.com/docs/git.html#Documentation/git.txt-codeGITCEILINGDIRECTORIEScode
[31]: https://git-scm.com/docs/git.html#Documentation/git.txt-codeGITDIRcode
[32]: https://git-scm.com/docs/git.html#Documentation/git.txt-codeGITDISCOVERYACROSSFILESYSTEMcode
[33]: https://git-scm.com/docs/git.html#Documentation/git.txt-codeGITDIRcode
[34]: https://jj-vcs.github.io/jj/latest/
[35]: https://github.com/serde-rs/serde
[36]: https://github.com/obi1kenobi/cargo-semver-checks/issues/new?assignees=&labels=C-bug&template=
bug_report.yml
[37]: https://predr.ag/blog/toward-fearless-cargo-update/
[38]: https://github.com/obi1kenobi/cargo-semver-checks/issues/167
[39]: https://github.com/obi1kenobi/cargo-semver-checks/issues/5
[40]: https://github.com/obi1kenobi/cargo-semver-checks/blob/main/CONTRIBUTING.md
[41]: #lint-level-configuration
[42]: https://github.com/obi1kenobi/cargo-semver-checks/blob/93baddc2a65a5055117ca670fe156dc761403fa
5/Cargo.toml#L5
[43]: https://docs.github.com/en/billing/managing-billing-for-github-sponsors/downgrading-a-sponsors
hip#:~:text=When%20you%20downgrade%20or%20cancel,for%20payments%20for%20GitHub%20Sponsors.
[44]: https://github.com/rust-lang/rust-semverver
[45]: https://github.com/rust-lang/rust-semverver/search?q=Rustup+to&type=commits
[46]: https://github.com/rust-lang/rust-semverver/pull/390
[47]: https://github.com/iomentum/cargo-breaking
[48]: https://crates.io/crates/syn
[49]: https://github.com/iomentum/cargo-breaking
[50]: https://github.com/obi1kenobi/trustfall
[51]: https://github.com/rust-lang/docs.rs/issues/1285
[52]: https://crates.io/crates/cargo-public-api
[53]: https://github.com/rust-lang/crates.io/issues/728#issuecomment-1182760954
[54]: https://github.com/obi1kenobi/cargo-semver-checks/releases
[55]: https://github.com/obi1kenobi/cargo-semver-checks/tree/main/src/lints
[56]: #example-changing-the-semver-requirement-for-a-lint
[57]: #example-downgrading-an-error-level-lint-to-a-warning
[58]: #example-configuring-lints-for-an-entire-workspace
[59]: #example-overriding-workspace-configuration
[60]: #common-configurations-make-must_use-lints-warn-only
[61]: #common-configurations-disable-must_use-lints-entirely
[62]: #implementation-details--limitations
[63]: https://github.com/obi1kenobi/cargo-semver-checks/issues/61
[64]: https://github.com/obi1kenobi/cargo-semver-checks/blob/1e0c82331cf26cf5ea52f3787d07eb583556cf7
9/src/lints/function_missing.ron#L5-L6
[65]: https://github.com/obi1kenobi/cargo-semver-checks/blob/1e0c82331cf26cf5ea52f3787d07eb583556cf7
9/src/lints/trait_now_doc_hidden.ron#L5-L6
[66]: https://github.com/obi1kenobi/cargo-semver-checks/blob/1e0c82331cf26cf5ea52f3787d07eb583556cf7
9/src/lints/function_must_use_added.ron#L5-L6
[67]: https://github.com/obi1kenobi/cargo-semver-checks/issues/827
[68]: https://github.com/cargo-bins/cargo-binstall
[69]: https://github.com/obi1kenobi/cargo-semver-checks/issues/841
[70]: https://github.com/numi-hq/open-design
[71]: https://numi.tech/?ref=semver-checks
