//! Z4 QBF - Quantified Boolean Formula Solver
//!
//! A QCDCL (Quantified Conflict-Driven Clause Learning) solver for QBF.
//!
//! ## Features
//! - QDIMACS parser for standard QBF benchmarks
//! - QCDCL algorithm based on DepQBF and Zhang & Malik
//! - Universal reduction for clause simplification
//! - Quantifier-aware unit propagation
//! - Certificate generation (Skolem functions for SAT, Herbrand for UNSAT)
//!
//! ## Algorithm Overview
//!
//! QCDCL extends CDCL to handle quantified formulas:
//! - Variables are partitioned into existential (∃) and universal (∀) quantifier blocks
//! - Universal reduction removes universal literals that cannot affect satisfiability
//! - Conflict analysis respects quantifier dependencies
//! - Learned clauses must be "dependency-valid"
//!
//! ## References
//! - Lonsing & Biere, "DepQBF: A Dependency-Aware QBF Solver"
//! - Zhang & Malik, "Conflict Driven Learning in a Quantified Boolean Satisfiability Solver"

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod formula;
pub mod parser;
pub mod solver;

pub use formula::{QbfFormula, Quantifier, QuantifierBlock};
pub use parser::{parse_qdimacs, QdimacsError};
pub use solver::{Certificate, QbfResult, QbfSolver};
