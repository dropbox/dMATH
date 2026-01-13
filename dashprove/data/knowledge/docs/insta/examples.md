**insta: a snapshot testing library for Rust**

[[Crates.io]][1] [[License]][2] [[Documentation]][3] [[VSCode Extension]][4]

## Introduction

Snapshots tests (also sometimes called approval tests) are tests that assert values against a
reference value (the snapshot). This is similar to how `assert_eq!` lets you compare a value against
a reference value but unlike simple string assertions, snapshot tests let you test against complex
values and come with comprehensive tools to review changes.

Snapshot tests are particularly useful if your reference values are very large or change often.

## Example

#[test]
fn test_hello_world() {
    insta::assert_debug_snapshot!(vec![1, 2, 3]);
}

Curious? There is a screencast that shows the entire workflow: [watch the insta introduction
screencast][5]. Or if you're not into videos, read the [5 minute introduction][6].

Insta also supports inline snapshots which are stored right in your source file instead of separate
files. This is accomplished by the companion [cargo-insta][7] tool.

## Editor Support

For looking at `.snap` files there is a [vscode extension][8] which can syntax highlight snapshot
files, review snapshots and more. It can be installed from the marketplace: [view on
marketplace][9].

[[jump to definition]][10]

## Diffing

Insta uses [`similar`][11] for all its diffing operations. You can use it independently of insta.
You can use the [`similar-asserts`][12] crate to get inline diffs for the standard `assert_eq!`
macro to achieve insta like diffs for regular comparisons:

use similar_asserts::assert_eq;

fn main() {
    let reference = vec![1, 2, 3, 4];
    assert_eq!(reference, (0..4).collect::<Vec<_>>());
}

## Sponsor

If you like the project and find it useful you can [become a sponsor][13].

## License and Links

* [Project Website][14]
* [Documentation][15]
* [Issue Tracker][16]
* License: [Apache-2.0][17]

[1]: https://crates.io/crates/insta
[2]: https://github.com/mitsuhiko/insta/blob/master/LICENSE
[3]: https://docs.rs/insta
[4]: https://marketplace.visualstudio.com/items?itemName=mitsuhiko.insta
[5]: https://www.youtube.com/watch?v=rCHrMqE4JOY&feature=youtu.be
[6]: https://insta.rs/docs/quickstart/
[7]: https://github.com/mitsuhiko/insta/tree/master/cargo-insta
[8]: https://github.com/mitsuhiko/insta/tree/master/vscode-insta
[9]: https://marketplace.visualstudio.com/items?itemName=mitsuhiko.insta
[10]: https://raw.githubusercontent.com/mitsuhiko/insta/master/vscode-insta/images/jump-to-definitio
n.gif
[11]: https://github.com/mitsuhiko/similar
[12]: https://github.com/mitsuhiko/similar-asserts
[13]: https://github.com/sponsors/mitsuhiko
[14]: https://insta.rs/
[15]: https://docs.rs/insta/
[16]: https://github.com/mitsuhiko/insta/issues
[17]: https://github.com/mitsuhiko/insta/blob/master/LICENSE
