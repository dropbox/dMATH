//! tla-smt - SMT solver integration for TLA+
//!
//! This crate provides translation from TLA+ expressions to SMT-LIB format
//! and integration with the Z3 SMT solver.
//!
//! # Features
//!
//! - Translation of TLA+ expressions to Z3
//! - Support for booleans, integers, and uninterpreted sorts
//! - Quantifier support (\A, \E)
//! - Timeout handling
//! - Model extraction for counterexamples
//!
//! # Example
//!
//! ```rust,ignore
//! use tla_smt::{SmtContext, SmtResult};
//! use tla_core::ast::Expr;
//!
//! let ctx = SmtContext::new();
//! let result = ctx.check_sat(&expr)?;
//! match result {
//!     SmtResult::Sat(model) => println!("Satisfiable: {:?}", model),
//!     SmtResult::Unsat => println!("Unsatisfiable"),
//!     SmtResult::Unknown => println!("Unknown"),
//! }
//! ```

mod error;
mod solver;
mod translate;

pub use error::{SmtError, SmtResult as Result};
pub use solver::{SmtCheckResult, SmtContext, SmtModel, SmtValue};
pub use translate::{SmtTranslator, Sort};
