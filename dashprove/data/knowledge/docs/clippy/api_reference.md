## absolute_paths [¶][1] [📋][2]
## restriction allow

### What it does

Checks for usage of items through absolute paths, like `std::env::current_dir`.

### Why restrict this?

Many codebases have their own style when it comes to importing, but one that is seldom used is using
absolute paths *everywhere*. This is generally considered unidiomatic, and you should add a `use`
statement.

The default maximum segments (2) is pretty strict, you may want to increase this in `clippy.toml`.

Note: One exception to this is code from macro expansion - this does not lint such cases, as using
absolute paths is the proper way of referencing items in one.

### Known issues

There are currently a few cases which are not caught by this lint:

* Macro calls. e.g. `path::to::macro!()`
* Derive macros. e.g. `#[derive(path::to::macro)]`
* Attribute macros. e.g. `#[path::to::macro]`

### Example

`let x = std::f64::consts::PI;
`

Use any of the below instead, or anything else:

`use std::f64;
use std::f64::consts;
use std::f64::consts::PI;
let x = f64::consts::PI;
let x = consts::PI;
let x = PI;
use std::f64::consts as f64_consts;
let x = f64_consts::PI;
`

### Configuration

* `absolute-paths-allowed-crates`: Which crates to allow absolute paths from
  
  (default: `[]`)
* `absolute-paths-max-segments`: The maximum number of segments a path can have before being linted,
  anything above this will be linted.
  
  (default: `2`)
Applicability: Unspecified[(?)][3]
Added in: 1.73.0
[Related Issues][4]
[View Source][5]

[1]: #absolute_paths
[2]: 
[3]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint_defs/enum.Applicability.html#variant
s
[4]: https://github.com/rust-lang/rust-clippy/issues?q=is%3Aissue+absolute_paths
[5]: https://github.com/rust-lang/rust-clippy/blob/master/clippy_lints/src/absolute_paths.rs#L13
