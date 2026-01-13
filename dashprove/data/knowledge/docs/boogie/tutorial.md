# Boogie

[[License]][1] [[NuGet package]][2]

Boogie is a modeling language, intended as a layer on which to build program verifiers for other
languages. Several program verifiers have been built in this way, including the [VCC][3] and
[HAVOC][4] verifiers for C and the verifiers for [Dafny][5], [Chalice][6], [Spec#][7], and
[Move][8]. For a sample verifier for a toy language built on top of Boogie, see [Forro][9].

Boogie is also the name of a tool. The tool accepts the Boogie language as input, optionally infers
some invariants in the given Boogie program, and then generates verification conditions that are
passed to an SMT solver. The default SMT solver is [Z3][10].

A tutorial for Boogie is available [here][11]. The documentation in this tutorial, although slightly
out-of-date, captures the essence of the Boogie language and tool.

Boogie has long been used for modeling and verifying sequential programs. Recently, through its
[Civl][12] extension, Boogie has become capable of modeling concurrent and distributed systems.

## Getting help

You can ask questions and report issues on our [issue tracker][13].

## Contribute

We are happy to receive contributions via [pull requests][14].

## Dependencies

Boogie requires [.NET Core][15] and a supported SMT solver (see [below][16]).

## Installation

Boogie releases are packaged as a .NET Core global tool available at [nuget.org][17]. To install
Boogie simply run:

`$ dotnet tool install --global boogie
`

## Building

To build Boogie run:

`$ dotnet build Source/Boogie.sln
`

The compiled Boogie binary is `Source/BoogieDriver/bin/${CONFIGURATION}/${FRAMEWORK}/BoogieDriver`.

## Backend SMT Solver

The default SMT solver for Boogie is [Z3][18]. Support for [CVC5][19] and [Yices2][20] is
experimental.

By default, Boogie looks for an executable called `z3|cvc5|yices2[.exe]` in your `PATH` environment
variable. If the solver executable is called differently on your system, use
`/proverOpt:PROVER_NAME=<exeName>`. Alternatively, an explicit path can be given using
`/proverOpt:PROVER_PATH=<path>`.

To learn how custom options can be supplied to the SMT solver (and more), call Boogie with
`/proverHelp`.

### Z3

The current test suite assumes version 4.11.2, but earlier and newer versions may also work.

### CVC5 (experimental)

Call Boogie with `/proverOpt:SOLVER=CVC5`.

### Yices2 (experimental)

Call Boogie with `/proverOpt:SOLVER=Yices2`.

Works for unquantified fragments, e.g. arrays + arithmetic + bitvectors. Does not work for
quantifiers, generalized arrays, datatypes.

## Testing

Boogie has two forms of tests. Driver tests and unit tests

### Driver tests

See the [Driver test documentation][21]

### Unit tests

See the [Unit test documentation][22]

## Versioning and Release

The current version of Boogie is noted in a [build property][23]. To push a new version to nuget,
perform the following steps:

* Update the version (e.g., x.y.z) in `Source/Directory.Build.props`
* `git add Source/Directory.Build.props; git commit -m "Release version x.y.z"` to commit the change
  locally
* `git push origin master:release-vx.y.z` to push your change in a separate branch, where `origin`
  normally denotes the remote on github.com
* [Create a pull request][24] and wait for it to be approved and merged
* `git tag vx.y.z` to create a local tag for the version
* `git push origin vx.y.z` to push the tag once the pull request is merged

The [CI workflow][25] will build and push the packages.

## License

Boogie is licensed under the MIT License (see [LICENSE.txt][26]).

[1]: /boogie-org/boogie/blob/master/LICENSE.txt
[2]: https://www.nuget.org/packages/Boogie
[3]: https://github.com/microsoft/vcc
[4]: https://www.microsoft.com/en-us/research/project/havoc
[5]: https://github.com/dafny-lang/dafny
[6]: https://www.microsoft.com/en-us/research/project/chalice
[7]: https://www.microsoft.com/en-us/research/project/spec
[8]: https://github.com/Move/move
[9]: https://github.com/boogie-org/forro
[10]: https://github.com/Z3Prover/z3
[11]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/krml178.pdf
[12]: https://civl-verifier.github.io/
[13]: https://github.com/boogie-org/boogie/issues
[14]: https://github.com/boogie-org/boogie/pulls
[15]: https://dotnet.microsoft.com
[16]: #backend-smt-solver
[17]: https://www.nuget.org/packages/Boogie
[18]: https://github.com/Z3Prover/z3
[19]: https://cvc5.github.io/
[20]: https://yices.csl.sri.com/
[21]: /boogie-org/boogie/blob/master/Test/README.md
[22]: /boogie-org/boogie/blob/master/Source/UnitTests/README.md
[23]: /boogie-org/boogie/blob/master/Source/Directory.Build.props
[24]: https://github.com/boogie-org/boogie/compare
[25]: /boogie-org/boogie/blob/master/.github/workflows/test.yml
[26]: /boogie-org/boogie/blob/master/LICENSE.txt
