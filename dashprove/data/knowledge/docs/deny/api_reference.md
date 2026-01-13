# [Checks][1]

cargo-deny supports several different classes of checks that can be performed on your project's
crate graph. By default, `cargo deny check` will execute **all** of the supported checks, falling
back to the default configuration for that check if one is not explicitly specified.

## [licenses][2]

Checks the license information for each crate.

## [bans][3]

Checks for specific crates in your graph, as well as duplicates.

## [advisories][4]

Checks advisory databases for crates with security vulnerabilities, or that have been marked as
`Unmaintained`, or which have been yanked from their source registry.

## [sources][5]

Checks the source location for each crate.

[1]: #checks
[2]: licenses/index.html
[3]: bans/index.html
[4]: advisories/index.html
[5]: sources/index.html
