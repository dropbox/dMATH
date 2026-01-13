# Crate proptest Copy item path

[Source][1]
Expand description

## [§][2]Proptest Reference Documentation

This is the reference documentation for the proptest API.

For documentation on how to get started with proptest and general usage advice, please refer to the
[Proptest Book][3].

## Modules[§][4]

*[arbitrary][5]*
  Defines the `Arbitrary` trait and related free functions and type aliases.
*[array][6]*
  Support for strategies producing fixed-length arrays.
*[bits][7]*
  Strategies for working with bit sets.
*[bool][8]*
  Strategies for generating `bool` values.
*[char][9]*
  Strategies for generating `char` values.
*[collection][10]*
  Strategies for generating `std::collections` of values.
*[num][11]*
  Strategies to generate numeric values (as opposed to integers used as bit fields).
*[option][12]*
  Strategies for generating `std::Option` values.
*[path][13]`std`*
  Strategies for generating [`PathBuf`] and related path types.
*[prelude][14]*
  Re-exports the most commonly-needed APIs of proptest.
*[result][15]*
  Strategies for combining delegate strategies into `std::Result`s.
*[sample][16]*
  Strategies for generating values by taking samples of collections.
*[strategy][17]*
  Defines the core traits used by Proptest.
*[string][18]`std`*
  Strategies for generating strings and byte strings from regular expressions.
*[test_runner][19]*
  State and functions for running proptest tests.
*[tuple][20]*
  Support for combining strategies into tuples.

## Macros[§][21]

*[prop_assert][22]*
  Similar to `assert!` from std, but returns a test failure instead of panicking if the condition
  fails.
*[prop_assert_eq][23]*
  Similar to `assert_eq!` from std, but returns a test failure instead of panicking if the condition
  fails.
*[prop_assert_ne][24]*
  Similar to `assert_ne!` from std, but returns a test failure instead of panicking if the condition
  fails.
*[prop_assume][25]*
  Rejects the test input if assumptions are not met.
*[prop_compose][26]*
  Convenience to define functions which produce new strategies.
*[prop_oneof][27]*
  Produce a strategy which picks one of the listed choices.
*[proptest][28]*
  Easily define `proptest` tests.

## Attribute Macros[§][29]

*[property_test][30]`attr-macro`*
  The `property_test` procedural macro simplifies the creation of property-based tests using the
  `proptest` crate. This macro provides a more concise syntax for writing tests that automatically
  generate test cases based on properties.

[1]: ../src/proptest/lib.rs.html#10-95
[2]: #proptest-reference-documentation
[3]: https://proptest-rs.github.io/proptest/intro.html
[4]: #modules
[5]: arbitrary/index.html
[6]: array/index.html
[7]: bits/index.html
[8]: bool/index.html
[9]: char/index.html
[10]: collection/index.html
[11]: num/index.html
[12]: option/index.html
[13]: path/index.html
[14]: prelude/index.html
[15]: result/index.html
[16]: sample/index.html
[17]: strategy/index.html
[18]: string/index.html
[19]: test_runner/index.html
[20]: tuple/index.html
[21]: #macros
[22]: macro.prop_assert.html
[23]: macro.prop_assert_eq.html
[24]: macro.prop_assert_ne.html
[25]: macro.prop_assume.html
[26]: macro.prop_compose.html
[27]: macro.prop_oneof.html
[28]: macro.proptest.html
[29]: #attributes
[30]: attr.property_test.html
