//! Lean5 Self-Verification
//!
//! This crate provides the infrastructure for Lean5 to verify its own kernel.
//! The approach is:
//!
//! 1. **Specification**: Define what kernel correctness means in Lean5's type theory
//! 2. **Proofs**: Construct proof terms that witness these properties
//! 3. **Verification**: Type-check the proofs using the kernel
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    SELF-VERIFICATION STACK                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  Level 3: Kernel Properties (verified by Level 2)               │
//! │           - Type preservation                                   │
//! │           - Progress                                            │
//! │           - Confluence                                          │
//! │                                                                  │
//! │  Level 2: Kernel Specification (Lean5 terms)                    │
//! │           - Expr, Level, Environment as inductive types         │
//! │           - has_type, is_def_eq as recursive functions          │
//! │           - Proof terms: λx. rfl, λx. congruence, etc.         │
//! │                                                                  │
//! │  Level 1: Lean5 Kernel (Rust implementation)                    │
//! │           - Type checks Level 2 and Level 3                     │
//! │           - Trust base: ~8k lines Rust                          │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Insight
//!
//! The specification at Level 2 is a *model* of the kernel written in Lean5.
//! We prove properties about this model, then verify that the Rust kernel
//! matches the model via cross-validation testing.
//!
//! ## Module Structure
//!
//! - `spec`: Core specifications (Expr, Level, typing judgment)
//! - `props`: Kernel properties (preservation, progress, confluence)
//! - `proofs`: Proof terms witnessing properties
//! - `validate`: Cross-validation with Rust kernel

pub mod proofs;
pub mod props;
pub mod spec;
pub mod validate;

pub use proofs::{ProofLibrary, ProofTerm};
pub use props::{Property, PropertyResult};
pub use spec::{SpecExpr, SpecLevel, Specification};
pub use validate::{CrossValidator, ValidationResult};

/// Result of self-verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Total specifications defined
    pub total_specs: usize,
    /// Specifications successfully verified
    pub verified_specs: usize,
    /// Failed verifications with errors
    pub failures: Vec<VerificationFailure>,
    /// Cross-validation results
    pub cross_validation: Option<CrossValidationSummary>,
}

/// A verification failure
#[derive(Debug, Clone)]
pub struct VerificationFailure {
    /// Name of the property that failed
    pub property: String,
    /// Error message
    pub error: String,
    /// Location in proof term (if applicable)
    pub location: Option<String>,
}

/// Summary of cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationSummary {
    /// Total test cases run
    pub total_cases: usize,
    /// Cases where Lean5 spec and Rust kernel agree
    pub matching: usize,
    /// Cases where they disagree (bugs!)
    pub mismatches: Vec<CrossValidationMismatch>,
}

/// A mismatch between spec and implementation
#[derive(Debug, Clone)]
pub struct CrossValidationMismatch {
    /// Input that caused mismatch
    pub input: String,
    /// Spec's result
    pub spec_result: String,
    /// Rust kernel's result
    pub impl_result: String,
}

/// Run self-verification
pub fn verify_kernel() -> VerificationResult {
    let spec = match Specification::new() {
        Ok(spec) => spec,
        Err(e) => {
            return VerificationResult {
                total_specs: 0,
                verified_specs: 0,
                failures: vec![VerificationFailure {
                    property: "specification".to_string(),
                    error: format!("failed to build specification: {e}"),
                    location: None,
                }],
                cross_validation: None,
            }
        }
    };
    let proofs = ProofLibrary::new();

    let mut result = VerificationResult {
        total_specs: spec.definitions().len(),
        verified_specs: 0,
        failures: Vec::new(),
        cross_validation: None,
    };

    // Verify each proof
    for (name, proof) in proofs.all_proofs() {
        match proof.verify(&spec) {
            Ok(()) => {
                result.verified_specs += 1;
            }
            Err(e) => {
                result.failures.push(VerificationFailure {
                    property: name.to_string(),
                    error: e.to_string(),
                    location: None,
                });
            }
        }
    }

    // Run cross-validation
    let validator = CrossValidator::new(&spec);
    result.cross_validation = Some(validator.run_validation());

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_framework() {
        let result = verify_kernel();

        println!("Verification result:");
        println!("  Total specs: {}", result.total_specs);
        println!("  Verified: {}", result.verified_specs);
        println!("  Failures: {}", result.failures.len());

        // Framework should at least run without panic and with no failures now that
        // the spec is registered inside the environment.
        assert!(result.total_specs > 0, "Should have some specs");
        assert!(
            result.failures.is_empty(),
            "Expected all proofs to verify: {:?}",
            result.failures
        );

        // Cross-validation should work
        if let Some(cv) = &result.cross_validation {
            println!("Cross-validation:");
            println!("  Total cases: {}", cv.total_cases);
            println!("  Matching: {}", cv.matching);
            println!("  Mismatches: {}", cv.mismatches.len());

            // All cross-validation cases should match
            assert!(
                cv.mismatches.is_empty(),
                "Cross-validation mismatches: {:?}",
                cv.mismatches
            );
        }
    }

    #[test]
    fn test_spec_definitions() {
        let spec = Specification::new().expect("spec should build");
        assert!(
            spec.definitions().len() >= 20,
            "Should have at least 20 definitions"
        );

        // Check key definitions exist
        assert!(spec.definitions().contains_key("Eq"));
        assert!(spec.definitions().contains_key("has_type"));
        assert!(spec.definitions().contains_key("is_def_eq"));
        assert!(spec.definitions().contains_key("TypePreservation"));
    }
}
