# Crate quickcheck Copy item path

[Source][1]
Expand description

This crate is a port of [Haskell’s QuickCheck][2].

For detailed examples, please see the [README][3].

## [§][4]Compatibility

In general, this crate considers the `Arbitrary` implementations provided as implementation details.
Strategies may or may not change over time, which may cause new test failures, presumably due to the
discovery of new bugs due to a new kind of witness being generated. These sorts of changes may
happen in semver compatible releases.

## Macros[§][5]

*[quickcheck][6]*
  A macro for writing quickcheck tests.

## Structs[§][7]

*[Gen][8]*
  Gen represents a PRNG.
*[QuickCheck][9]*
  The main QuickCheck type for setting configuration and running QuickCheck.
*[TestResult][10]*
  Describes the status of a single instance of a test.

## Traits[§][11]

*[Arbitrary][12]*
  `Arbitrary` describes types whose values can be randomly generated and shrunk.
*[Testable][13]*
  `Testable` describes types (e.g., a function) whose values can be tested.

## Functions[§][14]

*[empty_shrinker][15]*
  Creates a shrinker with zero elements.
*[quickcheck][16]*
  Convenience function for running QuickCheck.
*[single_shrinker][17]*
  Creates a shrinker with a single element.

[1]: ../src/quickcheck/lib.rs.html#1-93
[2]: https://hackage.haskell.org/package/QuickCheck
[3]: https://github.com/BurntSushi/quickcheck
[4]: #compatibility
[5]: #macros
[6]: macro.quickcheck.html
[7]: #structs
[8]: struct.Gen.html
[9]: struct.QuickCheck.html
[10]: struct.TestResult.html
[11]: #traits
[12]: trait.Arbitrary.html
[13]: trait.Testable.html
[14]: #functions
[15]: fn.empty_shrinker.html
[16]: fn.quickcheck.html
[17]: fn.single_shrinker.html
