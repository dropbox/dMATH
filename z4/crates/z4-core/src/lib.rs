//! Z4 Core - Common types and traits for the Z4 SMT solver
//!
//! This crate provides the foundational types shared across all Z4 components:
//! - Term representation (hash-consed DAG)
//! - Sort system (type checking)
//! - Model types (satisfying assignments)
//! - Theory trait (interface for theory solvers)
//! - Proof types (resolution proofs, theory lemmas)
//! - Tseitin transformation (Boolean to CNF)

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod model;
pub mod proof;
pub mod sort;
pub mod term;
pub mod theory;
pub mod tseitin;

pub use model::Model;
pub use proof::{AletheRule, Proof, ProofId, ProofStep};
pub use sort::Sort;
pub use term::{Constant, RationalWrapper, Symbol, Term, TermData, TermId, TermStore};
pub use theory::{
    DisequlitySplitRequest, ExpressionSplitRequest, SplitRequest, TheoryLit, TheoryPropagation,
    TheoryResult, TheorySolver,
};
pub use tseitin::{CnfClause, CnfLit, Tseitin, TseitinResult, TseitinState};
