//! Z4 Frontend - SMT-LIB 2.6 parser and preprocessor
//!
//! Parses SMT-LIB input and converts it to the internal representation.
//!
//! # Example
//!
//! ```
//! use z4_frontend::parser::parse;
//! use z4_frontend::command::Command;
//!
//! let input = r#"
//!     (set-logic QF_LIA)
//!     (declare-const x Int)
//!     (assert (> x 0))
//!     (check-sat)
//! "#;
//!
//! let commands = parse(input).unwrap();
//! assert_eq!(commands.len(), 4);
//! assert!(matches!(&commands[0], Command::SetLogic(logic) if logic == "QF_LIA"));
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod command;
pub mod elaborate;
pub mod lexer;
pub mod parser;
pub mod sexp;

pub use command::{Command, Constant, Sort, Term};
pub use elaborate::{CommandResult, Context, ElaborateError, OptionValue, SymbolInfo};
pub use parser::{parse, Parser};
pub use sexp::{ParseError, SExpr};
