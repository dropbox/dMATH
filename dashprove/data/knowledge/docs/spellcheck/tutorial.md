# cargo-spellcheck

[[crates.io]][1] [[CI]][2] [[commits-since]][3] [[Crates.io MSRV]][4]

Check your spelling with `hunspell` and/or `nlprule`.

## Use Cases

Run `cargo spellcheck --fix` or `cargo spellcheck fix` to fix all your documentation comments in
order to avoid nasty typos all over your source tree. Meant as a helper simplifying review as well
as improving CI checks after a learning phase for custom/topic specific lingo.

`cargo-spellcheck` is also a valuable tool to run from git commit hooks or CI/CD systems.

### Check For Spelling and/or Grammar Mistakes

cargo spellcheck check
`error: spellcheck
   --> src/main.rs:44
    |
 44 | Fun facets shalld cause some erroris.
    |            ^^^^^^
    | - shall or shall d
    |`

### Apply Suggestions Interactively

cargo spellcheck fix
`error: spellcheck(Hunspell)
    --> /media/supersonic1t/projects/cargo-spellcheck/src/literalset.rs:291
     |
 291 |  Returns literl within the Err variant if not adjacent
     |          ^^^^^^

(13/14) Apply this suggestion [y,n,q,a,d,j,e,?]?

   lite
   litter
   litterer
   liter l
   liters
   literal
   liter
 » a custom replacement literal`

## Installation

`cargo install --locked cargo-spellcheck`

The `--locked` flag is the preferred way of installing to get the tested set of dependencies.

on OS X, you need to ensure that `libclang.dylib` can be found by the linker

which can be achieved by setting `DYLB_FALLBACK_LIBRARY_PATH`:

`export DYLD_FALLBACK_LIBRARY_PATH= \
    "$(xcode-select --print-path)/Toolchains/XcodeDefault.xctoolchain/usr/lib/"
`

In Linux, the file is `libclang.so` which can be installed via:

`apt-get install libclang-dev
`

Afterwards, you can set the variable `LIBCLANG_PATH` via:

`export LIBCLANG_PATH=/usr/lib/llvm-14/lib/
`

## Completions

`cargo spellcheck completions` for autodetection of your current shell via `$SHELL`,

or

`cargo spellcheck completions --shell zsh`

to explicitly specify your shell type.

Commonly it's use like this from your shell's `.rc*` file:

`source <(cargo spellcheck completions)`

Note: There is a [relevant clap issue (#3508)][5] that makes this fail in some cases.

## 🎈 Contribute!

Contributions are very welcome!

Generally the preferred way of doing so, is to comment in an issue that you would like to tackle the
implementation/fix.

This is usually followed by an initial PR where the implementation is then discussed and iteratively
refined. No need to get it all correct the first time!

## Documentation

* [Features and Roadmap][6]
* [Remedies for common issues][7]
* [Configuration][8]
* [Available Checkers][9]
* [Automation of `cargo-spellcheck`][10]

[1]: https://crates.io/crates/cargo-spellcheck
[2]: https://ci.fff.rs/teams/main/pipelines/cargo-spellcheck/jobs/master-validate
[3]: https://camo.githubusercontent.com/5d18fffb01d0a70e836b6044094c70e0c0a7021f05a9347720dd7bebee2b
c5fa/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f636f6d6d6974732d73696e63652f6472616
86e722f636172676f2d7370656c6c636865636b2f6c61746573742e737667
[4]: https://camo.githubusercontent.com/6187e752584f14461ca1cb9de8b4a79b28e9a1612aadeed999c4eff15bc3
82be/68747470733a2f2f696d672e736869656c64732e696f2f6372617465732f6d7372762f636172676f2d7370656c6c636
865636b
[5]: https://github.com/clap-rs/clap/issues/3508
[6]: /drahnr/cargo-spellcheck/blob/master/docs/features.md
[7]: /drahnr/cargo-spellcheck/blob/master/docs/remedy.md
[8]: /drahnr/cargo-spellcheck/blob/master/docs/configuration.md
[9]: /drahnr/cargo-spellcheck/blob/master/docs/checkers.md
[10]: /drahnr/cargo-spellcheck/blob/master/docs/automation.md
