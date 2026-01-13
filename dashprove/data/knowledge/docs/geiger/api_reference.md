# Crate cargo_geiger Copy item path

[Source][1]
Expand description

These modules expose the internal workings of `cargo-geiger`. They are currently not stable, and
therefore have no associated `SemVer`. As such, any function contained within may be subject to
change.

## Modules[§][2]

*[args][3]*
  Argument parsing
*[cli][4]*
  Bootstrapping functions for structs required by the CLI
*[graph][5]*
  Construction of the dependency graph
*[mapping][6]*
  Mapping functionality from `cargo::core` to `cargo_metadata`
*[readme][7]*
  Interaction with README.md files
*[scan][8]*
  Functions for scanning projects for unsafe code

[1]: ../src/cargo_geiger/lib.rs.html#1-46
[2]: #modules
[3]: args/index.html
[4]: cli/index.html
[5]: graph/index.html
[6]: mapping/index.html
[7]: readme/index.html
[8]: scan/index.html
