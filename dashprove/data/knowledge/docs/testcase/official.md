[[Crates.io]][1] [[Crates.io]][2] [[Docs.rs]][3] [[MIT License]][4] [[Build Status]][5]
[[Maintenance]][6]

# Test Case

## Overview

`test_case` crate provides procedural macro attribute that generates parametrized test instances.

## Getting Started

Crate has to be added as a dependency to `Cargo.toml`:

[dev-dependencies]
test-case = "*"

and imported to the scope of a block where it's being called (since attribute name collides with
rust's built-in `custom_test_frameworks`) via:

use test_case::test_case;

## Example usage:

#[cfg(test)]
mod tests {
    use test_case::test_case;

    #[test_case(-2, -4 ; "when both operands are negative")]
    #[test_case(2,  4  ; "when both operands are positive")]
    #[test_case(4,  2  ; "when operands are swapped")]
    fn multiplication_tests(x: i8, y: i8) {
        let actual = (x * y).abs();

        assert_eq!(8, actual)
    }
}

Output from `cargo test` for this example:

$ cargo test

running 4 tests
test tests::multiplication_tests::when_both_operands_are_negative ... ok
test tests::multiplication_tests::when_both_operands_are_positive ... ok
test tests::multiplication_tests::when_operands_are_swapped ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

### Test Matrix

The `#[test_matrix(...)]` macro allows generating multiple test cases from the Cartesian product of
one or more possible values for each test function argument. The number of arguments to the
`test_matrix` macro must be the same as the number of arguments to the test function. Each macro
argument can be:

`1. A list in array (`[x, y, ...]`) or tuple (`(x, y, ...)`) syntax. The values can be any
   valid [expression](https://doc.rust-lang.org/reference/expressions.html).
2. A closed numeric range expression (e.g. `0..100` or `1..=99`), which will generate
   argument values for all integers in the range.
3. A single expression, which can be used to keep one argument constant while varying the
   other test function arguments using a list or range.
`

#### Example usage:

#[cfg(test)]
mod tests {
    use test_case::test_matrix;

    #[test_matrix(
        [-2, 2],
        [-4, 4]
    )]
    fn multiplication_tests(x: i8, y: i8) {
        let actual = (x * y).abs();

        assert_eq!(8, actual)
    }
}

## MSRV Policy

Starting with version 3.0 and up `test-case` introduces policy of only supporting latest stable
Rust. These changes may happen overnight, so if your stack is lagging behind current stable release,
it may be best to consider locking `test-case` version with `=` in your `Cargo.toml`.

## Documentation

Most up to date documentation is available in our [wiki][7].

# License

Licensed under of MIT license ([LICENSE-MIT][8] or [https://opensource.org/licenses/MIT][9])

# Contributing

Project roadmap is available at [link][10]. All contributions are welcome.

Recommended tools:

* `cargo readme` - to regenerate README.md based on template and lib.rs comments
* `cargo insta` - to review test snapshots
* `cargo edit` - to add/remove dependencies
* `cargo fmt` - to format code
* `cargo clippy` - for all insights and tips
* `cargo fix` - for fixing warnings

[1]: https://crates.io/crates/test-case
[2]: https://crates.io/crates/test-case
[3]: https://docs.rs/test-case
[4]: https://raw.githubusercontent.com/rust-lang/docs.rs/master/LICENSE
[5]: https://github.com/frondeus/test-case/actions
[6]: https://camo.githubusercontent.com/6b64898228508112ee4f35f8422cc3473d61b1acca407e73b284061cd81a
35fe/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6d61696e74656e616e63652d61637469766c7
92d2d646576656c6f7065642d627269676874677265656e2e737667
[7]: https://github.com/frondeus/test-case/wiki
[8]: /frondeus/test-case/blob/master/LICENSE-MIT
[9]: https://opensource.org/licenses/MIT
[10]: https://github.com/frondeus/test-case/issues/74
