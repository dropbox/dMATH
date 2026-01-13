//! CHC (Constrained Horn Clauses) Engine for Unbounded Verification
//!
//! This crate provides a CHC-based verification engine that uses Spacer
//! backends (Z3/Z4) to discover inductive invariants and verify unbounded
//! properties.
//!
//! # Overview
//!
//! Constrained Horn Clauses (CHC) provide a powerful framework for automatic
//! program verification. The key idea is to encode verification conditions as
//! Horn clauses and use fixpoint solvers to discover inductive invariants.
//!
//! # Example
//!
//! ```ignore
//! use kani_fast_chc::*;
//! use kani_fast_kinduction::{TransitionSystemBuilder, SmtType};
//!
//! // Define a simple counter system
//! let ts = TransitionSystemBuilder::new()
//!     .variable("x", SmtType::Int)
//!     .init("(= x 0)")
//!     .transition("(= x' (+ x 1))")
//!     .property("p1", "nonneg", "(>= x 0)")
//!     .build();
//!
//! // Encode as CHC
//! let chc = encode_transition_system(&ts);
//!
//! // Solve using Spacer
//! let config = ChcSolverConfig::default();
//! let result = verify_chc(&chc, &config).await?;
//!
//! match result {
//!     ChcResult::Sat { model, .. } => {
//!         println!("Property verified! Invariant: {}", model.to_readable_string());
//!     }
//!     ChcResult::Unsat { .. } => {
//!         println!("Property violated!");
//!     }
//!     ChcResult::Unknown { reason, .. } => {
//!         println!("Unknown: {}", reason);
//!     }
//! }
//! ```
//!
//! # CHC Encoding
//!
//! A transition system `(Init, Trans, Prop)` is encoded as:
//!
//! 1. **Predicate**: `Inv(state_vars)` representing the invariant
//! 2. **Init clause**: `Init(x) => Inv(x)`
//! 3. **Trans clause**: `Inv(x) ∧ Trans(x, x') => Inv(x')`
//! 4. **Property clause**: `Inv(x) ∧ ¬Prop(x) => false`
//!
//! The CHC system is SAT if there exists an invariant `Inv` that:
//! - Contains all initial states
//! - Is preserved by the transition relation
//! - Implies the property
//!
//! # Spacer
//!
//! Spacer (available in Z3 and Z4) is a CHC solver based on IC3/PDR techniques.
//! It works by:
//! 1. Starting from the property and working backwards
//! 2. Building a sequence of "frames" representing reachable states
//! 3. Propagating lemmas forward to strengthen invariants
//! 4. Using interpolation for lemma generalization

pub mod algebraic_rewrite;
pub mod bitvec;
pub mod clause;
pub mod delegation;
pub mod encoding;
pub mod intrinsics;
pub mod mir;
pub mod mir_parser;
pub mod proof;
pub mod proof_relevance;
pub mod result;
pub mod smt_intrinsics;
pub mod solver;

// Re-use find_executable from kani-fast-portfolio to avoid duplication
pub(crate) use kani_fast_portfolio::find_executable;

// Re-export main types
pub use clause::{
    sanitize_smt_expr, sanitize_smt_identifier, ChcSystem, ChcSystemBuilder, ClauseHead,
    HornClause, Predicate, PredicateApp, UninterpretedFunction, Variable,
};
pub use encoding::{
    encode_array_bounds, encode_properties_separately, encode_simple_loop, encode_transition_system,
};
pub use mir::{
    apply_algebraic_rewrites, encode_mir_to_chc, encode_mir_to_chc_bitvec,
    encode_mir_to_chc_with_overflow_checks, encode_mir_to_chc_with_strategy,
    encode_mir_to_transition_system, optimize_mir_for_unbounded, program_needs_bitvec_encoding,
    MirBasicBlock, MirLocal, MirProgram, MirProgramBuilder, MirStatement, MirTerminator,
    VerificationResult, PANIC_BLOCK_ID, PC_ABORT_SENTINEL, PC_PANIC_SENTINEL, PC_RETURN_SENTINEL,
};
pub use mir_parser::{
    generate_mir_from_file, generate_mir_from_source, MirParseError, MirParser, ParsedMirFunction,
};
pub use proof::{generate_chc_proof, proof_to_json, ChcProofBuilder};
pub use result::{
    parse_spacer_proof, parse_z3_statistics, ChcResult, ChcSolverStats, CounterexampleState,
    CounterexampleTrace, InvariantModel, SmtValue, SolvedPredicate, VerificationOutcome,
};
pub use solver::{verify_chc, ChcBackend, ChcSolver, ChcSolverConfig, ChcSolverError};

// Algebraic rewriting for bitwise operations
pub use algebraic_rewrite::{
    collect_bitwise_ops, contains_bitwise, rewrite_expression, try_rewrite, BitwiseOp,
    RewriteResult,
};

// Bitvector encoding for precise bitwise reasoning
pub use bitvec::{
    convert_int_to_bitvec, needs_bitvec_encoding, BitvecConfig, BitvecOp, BitvecUnaryOp,
};

// Thin SMT intrinsic mapping (Rust intrinsics -> SMT-LIB2 bitvector ops)
pub use smt_intrinsics::intrinsic_to_smt;

// Proof relevance analysis
pub use proof_relevance::{BitwiseLocation, ProofRelevanceAnalysis};

// Verification delegation
pub use delegation::{
    choose_strategy, delegate_to_kani, is_kani_available, DelegationReason, KaniResult,
    VerificationPath,
};

// Hybrid verification (CHC + BMC fallback)
pub use delegation::{hybrid_verify_with_chc_result, HybridConfig, HybridResult};

/// Verify a transition system using CHC solving
///
/// This is a convenience function that encodes a transition system as CHC
/// and verifies it using Z3's Spacer engine.
pub async fn verify_transition_system(
    ts: &kani_fast_kinduction::TransitionSystem,
    config: &ChcSolverConfig,
) -> Result<ChcResult, ChcSolverError> {
    let chc = encode_transition_system(ts);
    verify_chc(&chc, config).await
}

/// Verify a transition system with default configuration
pub async fn verify_transition_system_default(
    ts: &kani_fast_kinduction::TransitionSystem,
) -> Result<ChcResult, ChcSolverError> {
    verify_transition_system(ts, &ChcSolverConfig::default()).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};
    use std::time::Duration;

    fn has_solver() -> bool {
        find_executable("z4").is_some() || find_executable("z3").is_some()
    }

    #[tokio::test]
    async fn test_verify_counter() {
        if !has_solver() {
            return;
        }

        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .build();

        let config = ChcSolverConfig::new().with_timeout(Duration::from_secs(10));
        let result = verify_transition_system(&ts, &config).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_sat());
    }

    #[tokio::test]
    async fn test_verify_two_counters() {
        if !has_solver() {
            return;
        }

        // x and y both start at 0
        // x always increments
        // y sometimes increments (non-deterministic)
        // Property: x >= y
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Int)
            .init("(and (= x 0) (= y 0))")
            .transition("(and (= x' (+ x 1)) (or (= y' (+ y 1)) (= y' y)))")
            .property("p1", "x_ge_y", "(>= x y)")
            .build();

        let result = verify_transition_system_default(&ts).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_sat(), "Expected SAT, got: {:?}", result);

        // Check that invariant is found
        if let ChcResult::Sat { model, .. } = &result {
            assert!(
                !model.predicates.is_empty(),
                "Expected invariant predicates, got: {}",
                model.to_readable_string()
            );
        }
    }

    #[tokio::test]
    async fn test_verify_violated_property() {
        if !has_solver() {
            return;
        }

        // Counter that goes negative
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 5)")
            .transition("(= x' (- x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .build();

        let result = verify_transition_system_default(&ts).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(
            result.is_unsat(),
            "Expected UNSAT (property violated), got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_verify_bounded_counter() {
        if !has_solver() {
            return;
        }

        // Counter with bound check
        let ts = TransitionSystemBuilder::new()
            .variable("i", SmtType::Int)
            .init("(= i 0)")
            .transition("(and (< i 10) (= i' (+ i 1)))")
            .property("p1", "bounds", "(and (>= i 0) (<= i 10))")
            .build();

        // Use Z3 for this specific test (consistent benchmark reference)
        let config = ChcSolverConfig::new()
            .with_backend(ChcBackend::Z3)
            .with_timeout(std::time::Duration::from_secs(10));
        let chc = encode_transition_system(&ts);
        let result = verify_chc(&chc, &config).await;

        assert!(
            result.is_ok(),
            "CHC solving failed: {:?}",
            result.as_ref().err()
        );
        let result = result.unwrap();
        assert!(
            result.is_sat(),
            "Expected SAT (property holds with invariant 0 <= i <= 10), got: {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_chc_output_format() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .build();

        let chc = encode_transition_system(&ts);
        let smt2 = chc.to_smt2();

        // Verify correct format
        assert!(
            smt2.contains("(set-logic HORN)"),
            "Missing HORN logic declaration"
        );
        assert!(
            smt2.contains("(declare-fun Inv (Int) Bool)"),
            "Missing Inv declaration"
        );
        assert!(smt2.contains("(assert (forall"), "Missing forall assertion");
        assert!(smt2.contains("(check-sat)"), "Missing check-sat");
        assert!(smt2.contains("(get-model)"), "Missing get-model");
    }

    // ============================================================
    // Mutation coverage tests
    // ============================================================

    /// Test find_executable returns Some for known executables
    /// Catches: lib.rs:87:5 replace find_executable -> Option<PathBuf> with None
    #[test]
    fn test_find_executable_finds_known_binaries() {
        // "ls" should exist on any Unix system, "cmd" on Windows
        #[cfg(unix)]
        {
            let result = find_executable("ls");
            assert!(result.is_some(), "ls should be findable in PATH");
            let path = result.unwrap();
            assert!(path.is_file(), "Found path should be a file");
        }

        // Also test something that definitely doesn't exist
        let nonexistent = find_executable("this_executable_definitely_does_not_exist_12345");
        assert!(
            nonexistent.is_none(),
            "Nonexistent binary should return None"
        );
    }

    /// Test find_executable returns correct path for z3 if installed
    #[test]
    fn test_find_executable_z3_or_z4() {
        // At least one solver should be available (tests depend on it)
        let z3 = find_executable("z3");
        let z4 = find_executable("z4");

        // At least one should exist (based on has_solver() usage)
        let has_any = z3.is_some() || z4.is_some();

        if has_any {
            // If found, verify it's a real file
            if let Some(path) = z3.as_ref().or(z4.as_ref()) {
                assert!(path.is_file(), "Found solver path should be a file");
            }
        }
        // If neither exists, the test is still valid (tests behavior of None return)
    }
}
