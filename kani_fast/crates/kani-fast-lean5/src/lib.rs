//! Lean5 proof generation for Kani Fast
//!
//! This crate provides translation from CHC invariants discovered by Kani Fast
//! to Lean5 proof terms and proof obligations, along with backend integration
//! for verifying generated proofs.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  Invariant-to-Lean5 Pipeline                    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  CHC Invariant  ──► Lean5 Expr  ──► Tactics  ──► Certificate   │
//! │  (SMT formula)       (AST)          (proof)      (verified)     │
//! │                                                                  │
//! │  Then optionally:                                                │
//! │                                                                  │
//! │  Certificate ──────────────────────► Lean5 Backend             │
//! │                                       (type check)              │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use kani_fast_lean5::{
//!     translate_invariant, ProofObligation, ProofCertificate,
//!     Lean5Backend, Lean5Config, certificate_from_chc
//! };
//! use kani_fast_chc::InvariantModel;
//!
//! // After CHC solving produces an invariant
//! let model: InvariantModel = /* ... */;
//!
//! // Generate Lean5 proof obligations
//! let obligations = ProofObligation::from_invariant(&model)?;
//!
//! // Create a proof certificate
//! let mut cert = certificate_from_chc("my_property", "Z3", obligations);
//!
//! // Optionally verify with Lean
//! if Lean5Backend::is_available() {
//!     cert.verify()?;
//!     println!("{}", cert.summary());
//! }
//!
//! // Save certificate to file
//! cert.write_to_file(Path::new("proof.lean"))?;
//! ```

mod backend;
mod certificate;
mod expr;
mod obligation;
mod smt_parser;
mod tactics;
mod translate;

// Backend integration
pub use backend::{check_lean_installation, Lean5Backend, Lean5Config, Lean5Error, Lean5Result};

// Certificate generation
pub use certificate::{
    certificate_from_ai, certificate_from_chc, certificate_from_kinduction,
    CertificateVerificationResult, ObligationCertificate, ProofCertificate,
    ProofCertificateBuilder, VerificationMethod,
};

// Expression types
pub use expr::{Lean5Expr, Lean5Name, Lean5Type};

// Proof obligations
pub use obligation::{
    generate_kinduction_obligation, ProofObligation, ProofObligationBuilder, ProofObligationKind,
};

// SMT parsing
pub use smt_parser::{parse_smt_formula, ParseError, SmtAst, SmtSort};

// Tactic generation
pub use tactics::{generate_tactics, Tactic, TacticBlock};

// Translation
pub use translate::{translate_ast, translate_invariant, TranslationContext, TranslationError};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    /// Integration test: CHC invariant model -> Lean5 proof obligations
    #[test]
    fn test_chc_to_lean5_integration() {
        use kani_fast_chc::result::{InvariantModel, SolvedPredicate};
        use kani_fast_kinduction::{SmtType, StateFormula};

        // Simulate a CHC solver result: an invariant for a simple counter
        // The invariant discovered for "counter starts at 0, increments" with property "x >= 0"
        // would be something like "(>= x 0)" or "(not (<= x (- 1)))"
        let model = InvariantModel {
            predicates: vec![SolvedPredicate {
                name: "Inv".to_string(),
                params: vec![("x".to_string(), SmtType::Int)],
                formula: StateFormula::new("(not (<= x (- 1)))"),
            }],
        };

        // Translate to Lean5
        let translated = translate_invariant(&model).expect("Translation should succeed");
        assert_eq!(translated.len(), 1);

        // Generate proof obligations
        let obligations =
            ProofObligation::from_invariant(&model).expect("Obligation generation should succeed");

        // Should have at least the invariant assertion and initiation obligations
        assert!(
            obligations.len() >= 2,
            "Expected at least 2 obligations, got {}",
            obligations.len()
        );

        // Generate Lean5 file
        let lean_file = ProofObligation::to_lean5_file(&obligations);

        // Verify the generated file structure
        assert!(lean_file.contains("namespace KaniFast"));
        assert!(lean_file.contains("end KaniFast"));
        assert!(lean_file.contains("theorem"));
        assert!(lean_file.contains("Inv"));

        // Verify the file contains proper Lean5 syntax
        assert!(
            lean_file.contains(":="),
            "Should contain proof term assignment"
        );
    }

    /// Test that complex invariants with multiple variables translate correctly
    #[test]
    fn test_multi_variable_invariant() {
        use kani_fast_chc::result::{InvariantModel, SolvedPredicate};
        use kani_fast_kinduction::{SmtType, StateFormula};

        // Two-counter invariant: x >= y
        let model = InvariantModel {
            predicates: vec![SolvedPredicate {
                name: "Inv".to_string(),
                params: vec![
                    ("x".to_string(), SmtType::Int),
                    ("y".to_string(), SmtType::Int),
                ],
                formula: StateFormula::new("(>= x y)"),
            }],
        };

        let translated = translate_invariant(&model).expect("Translation should succeed");
        assert_eq!(translated.len(), 1);

        // The result should be a nested forall: ∀ x, ∀ y, x ≥ y
        let expr = &translated[0];
        match expr {
            Lean5Expr::Forall(name1, _, inner) => {
                match inner.as_ref() {
                    Lean5Expr::Forall(name2, _, _) => {
                        // Parameters should appear in order
                        assert!(matches!(name1.as_str(), "x" | "y"));
                        assert!(matches!(name2.as_str(), "x" | "y"));
                        assert_ne!(name1, name2);
                    }
                    _ => panic!("Expected inner Forall, got {:?}", inner),
                }
            }
            _ => panic!("Expected outer Forall, got {:?}", expr),
        }
    }

    /// Test Z3-style variable names (with ! suffixes) are handled
    #[test]
    fn test_z3_variable_names() {
        use kani_fast_chc::result::{InvariantModel, SolvedPredicate};
        use kani_fast_kinduction::{SmtType, StateFormula};

        // Z3 often renames variables like x!0, x!1
        let model = InvariantModel {
            predicates: vec![SolvedPredicate {
                name: "Inv".to_string(),
                params: vec![("x!0".to_string(), SmtType::Int)],
                formula: StateFormula::new("(>= x!0 0)"),
            }],
        };

        let _translated = translate_invariant(&model).expect("Translation should succeed");
        let lean_file = ProofObligation::to_lean5_file(
            &ProofObligation::from_invariant(&model).expect("Should generate obligations"),
        );

        // Variable names should be cleaned (! replaced with _)
        assert!(
            lean_file.contains("x_0"),
            "Should contain cleaned variable name x_0, got:\n{}",
            lean_file
        );
        // Ensure the raw Z3 name with ! does not appear anywhere
        assert!(
            !lean_file.contains("x!0"),
            "Should not contain raw Z3 name x!0, got:\n{}",
            lean_file
        );
    }
}
