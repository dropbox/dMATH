# [cargo-deny][1]

cargo-deny is a cargo plugin that lets you lint your project's dependency graph to ensure all your
dependencies conform to your expectations and requirements.

## [Quickstart][2]

Installs cargo-deny, initializes your project with a default configuration, then runs all of the
checks against your project.

`cargo install --locked cargo-deny && cargo deny init && cargo deny check
`

## [Command Line Interface][3]

cargo-deny is intended to be used as a [Command Line Tool][4], see the link for the available
commands and options.

## [Checks][5]

cargo-deny supports several classes of checks, see [Checks][6] for the available checks and their
configuration options.

## [API][7]

cargo-deny is primarily meant to be used as a cargo plugin, but a majority of its functionality is
within a library whose docs you may view on [docs.rs][8]

## [GitHub Action][9]

For GitHub projects, one can run cargo-deny automatically as part of continuous integration using a
GitHub Action:

`name: CI
on: [push, pull_request]
jobs:
  cargo-deny:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: EmbarkStudios/cargo-deny-action@v1
`

For more information, see [`cargo-deny-action`][10] repository.

[1]: https://github.com/EmbarkStudios/cargo-deny
[2]: #quickstart
[3]: #command-line-interface
[4]: cli/index.html
[5]: #checks
[6]: checks/index.html
[7]: #api
[8]: https://docs.rs/cargo-deny
[9]: #github-action
[10]: https://github.com/EmbarkStudios/cargo-deny-action
