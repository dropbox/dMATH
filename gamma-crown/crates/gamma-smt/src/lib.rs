//! SMT solver integration for γ-CROWN neural network verification.
//!
//! This crate provides complete verification capability using the Z4 SMT solver.
//! It encodes neural networks as SMT formulas in the QF_LRA (Quantifier-Free
//! Linear Real Arithmetic) theory.
//!
//! # Architecture
//!
//! The integration follows the Reluplex/Marabou approach:
//! 1. Encode network structure as linear constraints
//! 2. Encode ReLU activations using Big-M or lazy splitting
//! 3. Query Z4 for satisfiability of property negation
//!
//! # Example
//!
//! ```ignore
//! use gamma_smt::SmtVerifier;
//! use gamma_core::{Bound, VerificationSpec};
//!
//! let verifier = SmtVerifier::new();
//! let result = verifier.verify(&network, &spec);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

mod encoder;
mod lazy_verifier;
mod model_parser;
mod propagate_integration;
mod verifier;

pub use encoder::{NetworkEncoder, SmtNetwork, SmtResult};
pub use lazy_verifier::{LazyVerifier, LazyVerifierConfig, ReluNeuron};
pub use model_parser::{parse_model, parse_model_to_map};
pub use propagate_integration::{BoundMethod, IntegratedVerifier, IntegratedVerifierConfig};
pub use verifier::{SmtVerifier, SmtVerifierConfig};

/// Error types for SMT operations.
#[derive(Debug, thiserror::Error)]
pub enum SmtError {
    /// Failed to encode network.
    #[error("encoding error: {0}")]
    EncodingError(String),
    /// SMT solver error.
    #[error("solver error: {0}")]
    SolverError(String),
    /// Unsupported layer type.
    #[error("unsupported layer for SMT encoding: {0}")]
    UnsupportedLayer(String),
    /// Invalid input bounds.
    #[error("invalid input bounds: {0}")]
    InvalidBounds(String),
}

/// Result type for SMT operations.
pub type Result<T> = std::result::Result<T, SmtError>;

#[cfg(test)]
mod tests {
    use super::*;
    use gamma_core::Bound;

    #[test]
    fn test_basic_compilation() {
        // Verify the crate compiles and exports work
        let verifier = SmtVerifier::new();
        let _config = SmtVerifierConfig::default();
        assert!(verifier
            .is_sat("(set-logic QF_LRA)(declare-const x Real)(assert (> x 0))(check-sat)")
            .unwrap());
    }

    #[test]
    fn test_relu_network_verification() {
        // Simple network with ReLU: y = ReLU(x + 1)
        // Input: x in [-2, 2]
        // After linear: z = x + 1, so z in [-1, 3]
        // After ReLU: y = max(0, z), so y in [0, 3]
        //
        // Property: y in [0, 3.5] should be VERIFIED
        let verifier = SmtVerifier::new();

        let weights = vec![
            vec![1.0], // First layer: identity
            vec![1.0], // Second layer: identity (after ReLU)
        ];
        let biases = vec![
            vec![1.0], // Add 1 to input
            vec![0.0], // No bias on output
        ];
        let layer_dims = vec![1, 1, 1];

        let input_bounds = vec![Bound::new(-2.0, 2.0)];
        let output_bounds = vec![Bound::new(0.0, 3.5)];
        // Intermediate bounds after first linear (before ReLU): [-1, 3]
        let intermediate_bounds = vec![vec![Bound::new(-1.0, 3.0)]];

        let result = verifier
            .verify_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
            )
            .unwrap();

        assert!(
            result.is_verified(),
            "Expected verification, got {:?}",
            result
        );
    }

    #[test]
    fn test_relu_network_tight_bounds() {
        // Same network but tighter bounds that should still verify
        // y = ReLU(x + 1) for x in [-2, 2] gives y in [0, 3]
        let verifier = SmtVerifier::new();

        let weights = vec![vec![1.0], vec![1.0]];
        let biases = vec![vec![1.0], vec![0.0]];
        let layer_dims = vec![1, 1, 1];

        let input_bounds = vec![Bound::new(-2.0, 2.0)];
        // Exactly the tight bounds [0, 3] - should verify
        let output_bounds = vec![Bound::new(0.0, 3.0)];
        let intermediate_bounds = vec![vec![Bound::new(-1.0, 3.0)]];

        let result = verifier
            .verify_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
            )
            .unwrap();

        assert!(
            result.is_verified(),
            "Expected verification, got {:?}",
            result
        );
    }
}
