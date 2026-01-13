# [Quick start][1]

## [Installation][2]

See the [Installation][3] section.

## [Create a project][4]

Create a new project with this command:

`cargo creusot new project-name
`

> Note
> 
> If you are using the development version of Creusot (`master` branch), you
> should also point your project to your local copy of `creusot-contracts`,
> using the `--creusot-contracts` option (otherwise the default is to use the
> released version on crates.io). To avoid hard-coding local paths in your
> configuration, one approach is to set `--creusot-contracts creusot-contracts`,
> and make a symbolic link `creusot-contracts` pointing to your local
> `creusot-contracts`.

That command creates a directory `package-name` containing the basic elements of
a Rust project verified with Creusot. The file `src/lib.rs` is initialized with
an example function annotated with a contract:

`// src/lib.rs
use creusot_contracts::prelude::*;

#[requires(x@ < i64::MAX@)]
#[ensures(result@ == x@ + 1)]
pub fn add_one(x: i64) -> i64 {
    x + 1
}`

## [Compile and prove][5]

To verify this project, run this command:

`cargo creusot prove
`

A successful run gives us the certainty that functions defined in this package
satisfy their contracts: for all arguments satisfying the preconditions
(`requires` clauses), the result of the function will satisfy the postconditions
(`ensures` clauses).

The command `cargo creusot prove` does two things: compile your Rust crate to
Coma, then search for proofs of verification conditions generated from the Coma
code using Why3find. These steps can be performed separately.

1. Run only the compiler and obtain Coma code:
   
   `cargo creusot
   `
2. Run Why3find’s proof search only on a specific Coma file (by default,
   Why3find is run on all Coma files under the `verif`):
   
   `cargo creusot prove verif/[COMA_FILE]
   `
   
   Multiple files can also be specified in a single command.

When the proof fails, you can add the `-i` option to open the Coma file in Why3
IDE.

`cargo creusot prove verif/[COMA_FILE] -i
`

The `-i` option only launches the Why3 IDE if the proof fails. You can also use
`--ide-always`

`cargo creusot prove verif/[COMA_FILE] --ide-always
`

When you know that the proof is going to fail, it can be slow to update every
time you modify your code. To skip proof search and just reuse the existing
`proof.json` as is, add the option `--replay`.

`cargo creusot prove verif/[COMA_FILE] --ide-always --replay
`

The documentation for the Why3 IDE can be found [here][6].

We also recommend section 2.3 of this [thesis][7] for a brief overview of Why3
and Creusot proofs.

### [Troubleshooting][8]

If you get an error like this

`error: The `creusot_contracts` crate is loaded, but the following items are mis
sing: <a list of identifiers> Maybe your version of `creusot-contracts` is wrong
?
`

Add the following to your `Cargo.toml` file:

`[patch.crates-io]
creusot-contracts = { path = "/relative/or/absolute/path/to/creusot-contracts/in
/creusot/directory" }
`

And please notify the Creusot developers that the version of Creusot should be
bumped to `NEXT_VERSION-dev` to prevent this error.

## [Legacy workflow with Why3 IDE][9]

This workflow is intended to help projects using old versions of Creusot that
still use `why3session.xml`.

Run the Creusot compiler:

`cargo creusot
`

Launch the Why3 IDE:

`cargo creusot why3 ide [FILE]
`

You must specify a file `why3session.xml` or a Coma file.

Difference with `cargo creusot prove`:

* `cargo creusot prove` (with `-i` or `--ide-always`) runs the Creusot compiler
  and the Why3find proof search beforehand, ensuring that you’re always working
  on the latest version of your code.
* `cargo creusot why3 ide` only runs `why3 ide` with the necessary options to
  load Coma files produced by Creusot. It’s up to you to make sure that the Coma
  files are up-to-date.

[1]: #quick-start
[2]: #installation
[3]: https://creusot-rs.github.io/creusot/guide/installation.html
[4]: #create-a-project
[5]: #compile-and-prove
[6]: https://www.why3.org/doc/starting.html#getting-started-with-the-gui
[7]: https://sarsko.github.io/_pages/SarekSkot%C3%A5m_thesis.pdf
[8]: #troubleshooting
[9]: #legacy-workflow-with-why3-ide
