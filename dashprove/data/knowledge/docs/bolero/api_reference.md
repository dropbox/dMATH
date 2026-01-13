# Crate bolero Copy item path

[Source][1]

## Modules[§][2]

*[generator][3]*
  Re-export of [`bolero_generator`][4]

## Macros[§][5]

*[check][6]*
  Execute tests for a given target
*[fuzz][7]Deprecated*

## Structs[§][8]

*[ByteSliceGenerator][9]*
  Default generator for byte slices
*[DefaultEngine][10]*
  The default engine used when defining a test target Engine implementation which mimics Rust’s
  default test harness. By default, the test inputs will include any present `corpus` and `crashes`
  files, as well as generating
*[TestTarget][11]*
  Configuration for a test target

## Enums[§][12]

*[DriverMode][13]Deprecated*
  Byte exhaustion strategy for the driver

## Traits[§][14]

*[Any][15]*
*[AnySliceExt][16]*
*[AnySliceMutExt][17]*
*[Driver][18]*
  Trait for driving the generation of a value
*[Engine][19]*
  Trait for defining an engine that executes a test
*[OneOfExt][20]*
  Extensions for picking a generator from a set of generators
*[OneValueOfExt][21]*
  Extensions for picking a value from a set of values
*[Test][22]*
  Trait for defining a test case
*[TypeGenerator][23]*
  Generate a value for a given type
*[TypeGeneratorWithParams][24]*
  Convert a type generator into the default value generator
*[ValueGenerator][25]*
  Generate a value with a parameterized generator

## Functions[§][26]

*[any][27]*
*[assume][28]*
*[constant][29]*
  Always generate the same value
*[fill][30]*
*[fill_bytes][31]*
*[gen][32]Deprecated*
  Generate a value for a given type
*[gen_with][33]Deprecated*
  Generate a value for a given type
*[one_of][34]*
  Pick a generator for the provided set of generators
*[one_value_of][35]*
  Pick a value for the provided set of values
*[pick][36]*
*[produce][37]*
  Generate a value for a given type
*[produce_with][38]*
  Generate a value for a given type with additional constraints
*[shuffle][39]*

## Derive Macros[§][40]

*[TypeGenerator][41]*
  Derive the an implementation of `TypeGenerator` for the given type.

[1]: ../src/bolero/lib.rs.html#1-495
[2]: #modules
[3]: generator/index.html
[4]: https://docs.rs/bolero-generator/0.13.5/x86_64-unknown-linux-gnu/bolero_generator/index.html
[5]: #macros
[6]: macro.check.html
[7]: macro.fuzz.html
[8]: #structs
[9]: struct.ByteSliceGenerator.html
[10]: struct.DefaultEngine.html
[11]: struct.TestTarget.html
[12]: #enums
[13]: enum.DriverMode.html
[14]: #traits
[15]: trait.Any.html
[16]: trait.AnySliceExt.html
[17]: trait.AnySliceMutExt.html
[18]: trait.Driver.html
[19]: trait.Engine.html
[20]: trait.OneOfExt.html
[21]: trait.OneValueOfExt.html
[22]: trait.Test.html
[23]: trait.TypeGenerator.html
[24]: trait.TypeGeneratorWithParams.html
[25]: trait.ValueGenerator.html
[26]: #functions
[27]: fn.any.html
[28]: fn.assume.html
[29]: fn.constant.html
[30]: fn.fill.html
[31]: fn.fill_bytes.html
[32]: fn.gen.html
[33]: fn.gen_with.html
[34]: fn.one_of.html
[35]: fn.one_value_of.html
[36]: fn.pick.html
[37]: fn.produce.html
[38]: fn.produce_with.html
[39]: fn.shuffle.html
[40]: #derives
[41]: derive.TypeGenerator.html
