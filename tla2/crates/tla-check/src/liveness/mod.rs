//! Liveness checking module
//!
//! This module implements liveness checking for TLA+ specifications using
//! the tableau construction method from Manna & Pnueli.
//!
//! # Overview
//!
//! Liveness properties are temporal properties that assert something "eventually"
//! happens. Examples:
//! - `<>P` - Eventually P holds
//! - `[]<>P` - Infinitely often P holds
//! - `P ~> Q` - P leads to Q
//! - `WF_vars(A)` - Weak fairness for action A
//! - `SF_vars(A)` - Strong fairness for action A
//!
//! # Algorithm
//!
//! 1. Parse temporal formula from AST using [`AstToLive`]
//! 2. Convert to positive normal form (negation pushed to atoms)
//! 3. Build tableau graph from the formula
//! 4. Build behavior graph × tableau product during state exploration
//! 5. Find strongly connected components (SCCs) using Tarjan's algorithm
//! 6. Check each SCC for accepting cycles (liveness violations)
//! 7. Extract counterexample trace if violation found
//!
//! # Architecture
//!
//! The liveness checking implementation consists of:
//!
//! - [`LiveExpr`]: Internal representation for temporal formulas (positive normal form)
//! - [`Tableau`]: Automaton graph built from temporal formula negation
//! - [`BehaviorGraph`]: Product graph of (state × tableau) pairs
//! - [`is_state_consistent`]: Check if a state satisfies tableau node predicates
//!
//! # References
//!
//! - Manna & Pnueli, "Temporal Verification of Reactive Systems: Safety", Ch 5
//! - TLC implementation in tlc2.tool.liveness package

mod ast_to_live;
mod behavior_graph;
mod checker;
mod consistency;
mod live_expr;
mod tableau;
mod tarjan;

pub use ast_to_live::{AstToLive, ConvertError};
pub use behavior_graph::{BehaviorGraph, BehaviorGraphNode, NodeInfo};
pub use checker::{LivenessChecker, LivenessConstraints, LivenessResult, LivenessStats};
pub use consistency::{eval_live_expr, is_state_consistent, is_transition_consistent};
pub use live_expr::{ExprLevel, LiveExpr};
pub use tableau::{Particle, Tableau, TableauNode};
pub use tarjan::{
    find_accepting_sccs, find_cycle_in_scc, find_cycles, find_sccs, Scc, TarjanResult, TarjanStats,
};
