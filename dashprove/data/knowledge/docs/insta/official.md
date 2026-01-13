Toggle Navigation
[Hello Snapshot Testing][1]
[What's in the Box?][2]
[What Does This Look Like?][3]
[Test Workflow][4]
[Learn More][5]

# Hello Snapshot Testing

[[Build Status]][6] [[Crates.io]][7] [[License]][8] [[Documentation]][9]

Snapshots tests (also sometimes called approval tests) are tests that assert values against a
reference value (the snapshot). Think of it as a supercharged version of `assert_eq!`. It lets you
compare a value against a reference value but unlike simple assertions the reference value is
managed by `insta` for you.

## What's in the Box?

* **Interactive snapshot reviews**: with `cargo-insta` you can perform reviews of all changed
  snapshots conveniently.
* **Inline snapshots**: insta can store snapshots right within your source file.
* **External snapshots**: it's also possible to store snapshots as separate files.
* **Redactions**: if you have output which can change between test runs (such as random identifiers,
  timestamps or others) you can instruct insta to redact these parts.
* **Flexible formats**: you can pick between snapshoting into different formats such as JSON, YAML,
  TOML, CSV or others.
* **Editor Support**: insta also provides a [VS Code Extension][10] that lets you review snapshots
  right from within your editor.
* **Pretty Diffs**: insta renders beautiful snapshot diffs right in your terminal with the help of
  the [similar][11] crate.
* **Supports older Rust:** insta, similar and [similar-asserts][12] support Rust down to 1.51.
* **Apache-2.0 licensed**: because the best tools are Open Source under a convenient license.

## What Does This Look Like?

Pretty simple. You write a test function where you perform some sort of computation and then use one
of the `insta` provided assertion macros:

`#[test]
fn test_simple() {
    insta::assert_yaml_snapshot!(calculate_value());
}
`

Note that no reference value is provided. Instead the reference value is placed on the first test
run in a separate snapshot file. Insta will automatically manage that snapshot for you.

## Test Workflow

This is what this process looks like in an actual example:

## Learn More

Want to learn more?

* There is a 12 minute introduction screencast that walks you through how it works: [Snapshot
  Testing with Insta][13]
* Read [the quickstart][14] for more information
* Read [the API documentation][15] for a deep dive on the API
Found an issue? You can [edit this page][16] on GitHub.

[1]: https://insta.rs/#hello-snapshot-testing
[2]: https://insta.rs/#what-s-in-the-box
[3]: https://insta.rs/#what-does-this-look-like
[4]: https://insta.rs/#test-workflow
[5]: https://insta.rs/#learn-more
[6]: https://github.com/mitsuhiko/insta/actions?query=workflow%3ATests
[7]: https://crates.io/crates/insta
[8]: https://github.com/mitsuhiko/insta/blob/master/LICENSE
[9]: https://docs.rs/insta
[10]: https://marketplace.visualstudio.com/items?itemName=mitsuhiko.insta
[11]: /similar/
[12]: https://github.com/mitsuhiko/similar-asserts
[13]: https://www.youtube.com/embed/rCHrMqE4JOY
[14]: /docs/quickstart
[15]: https://docs.rs/insta
[16]: https://github.com/mitsuhiko/insta-website/edit/main/content/_index.md
