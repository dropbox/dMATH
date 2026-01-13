# Crate test_case Copy item path

[Source][1]
Expand description

## [§][2]Overview

`test_case` crate provides procedural macro attribute that generates parametrized test instances.

## [§][3]Getting Started

Crate has to be added as a dependency to `Cargo.toml`:

`[dev-dependencies]
test-case = "*"`

and imported to the scope of a block where it’s being called (since attribute name collides with
rust’s built-in `custom_test_frameworks`) via:

`use test_case::test_case;`

## [§][4]Example usage:

`#[cfg(test)]
mod tests {
    use test_case::test_case;

    #[test_case(-2, -4 ; "when both operands are negative")]
    #[test_case(2,  4  ; "when both operands are positive")]
    #[test_case(4,  2  ; "when operands are swapped")]
    fn multiplication_tests(x: i8, y: i8) {
        let actual = (x * y).abs();

        assert_eq!(8, actual)
    }
}`

Output from `cargo test` for this example:

`$ cargo test

running 4 tests
test tests::multiplication_tests::when_both_operands_are_negative ... ok
test tests::multiplication_tests::when_both_operands_are_positive ... ok
test tests::multiplication_tests::when_operands_are_swapped ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`

### [§][5]Test Matrix

The `#[test_matrix(...)]` macro allows generating multiple test cases from the Cartesian product of
one or more possible values for each test function argument. The number of arguments to the
`test_matrix` macro must be the same as the number of arguments to the test function. Each macro
argument can be:

`1. A list in array (`[x, y, ...]`) or tuple (`(x, y, ...)`) syntax. The values can be any
   valid [expression](https://doc.rust-lang.org/reference/expressions.html).
2. A closed numeric range expression (e.g. `0..100` or `1..=99`), which will generate
   argument values for all integers in the range.
3. A single expression, which can be used to keep one argument constant while varying the
   other test function arguments using a list or range.`

#### [§][6]Example usage:

`#[cfg(test)]
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
}`

## [§][7]MSRV Policy

Starting with version 3.0 and up `test-case` introduces policy of only supporting latest stable
Rust. These changes may happen overnight, so if your stack is lagging behind current stable release,
it may be best to consider locking `test-case` version with `=` in your `Cargo.toml`.

## [§][8]Documentation

Most up to date documentation is available in our [wiki][9].

## Attribute Macros[§][10]

*[case][11]*
  Generates tests for given set of data
*[test_case][12]*
  Generates tests for given set of data
*[test_matrix][13]*
  Generates tests for the cartesian product of a given set of data

[1]: ../src/test_case/lib.rs.html#1-98
[2]: #overview
[3]: #getting-started
[4]: #example-usage
[5]: #test-matrix
[6]: #example-usage-1
[7]: #msrv-policy
[8]: #documentation
[9]: https://github.com/frondeus/test-case/wiki
[10]: #attributes
[11]: attr.case.html
[12]: attr.test_case.html
[13]: attr.test_matrix.html
