//! Nova-style Folding Schemes for Lean5 Proof Compression
//!
//! This crate implements Nova-style folding schemes for incrementally
//! composing and compressing lean5 proof certificates.
//!
//! # Overview
//!
//! Nova-style folding enables composing proofs incrementally without
//! proof size blowup. Instead of proving `F(x₁) ∧ F(x₂)` separately,
//! instances are "folded":
//!
//! - Two instances `(u₁, w₁)` and `(u₂, w₂)` fold into one `(u, w)`
//! - Verifier work is constant per fold step
//! - Final verification is single instance check
//!
//! # Architecture
//!
//! The crate is organized into:
//!
//! - [`r1cs`]: R1CS constraint system representation
//! - [`relaxed`]: Relaxed R1CS for folding (Az ∘ Bz = u·Cz + E)
//! - [`transcript`]: Fiat-Shamir transcript for challenges
//! - [`folding`]: Core folding operation
//! - [`ivc`]: Incrementally Verifiable Computation proofs
//!
//! # Example
//!
//! ```ignore
//! use lean5_fold::{IvcProof, start_ivc, extend_ivc, verify_ivc};
//! use lean5_kernel::ProofCert;
//!
//! // Start IVC from initial certificate
//! let mut ivc = start_ivc(&cert1, &env)?;
//!
//! // Extend with additional certificates
//! extend_ivc(&mut ivc, &cert2, &env)?;
//! extend_ivc(&mut ivc, &cert3, &env)?;
//!
//! // Verify the accumulated proof
//! assert!(verify_ivc(&ivc)?);
//! ```

pub mod cert_encoding;
pub mod error;
pub mod folding;
pub mod ivc;
pub mod r1cs;
pub mod relaxed;
pub mod transcript;

pub use cert_encoding::{encode_cert_to_r1cs, verify_encoded, EncodedR1CS};
pub use error::{FoldError, IvcError};
pub use folding::fold;
pub use ivc::{
    extend_ivc, extend_ivc_with_cert, start_ivc, start_ivc_from_cert, verify_ivc, IvcProof,
};
pub use r1cs::{R1CSBuilder, R1CSInstance, R1CSShape, R1CSWitness};
pub use relaxed::{RelaxedR1CSInstance, RelaxedR1CSWitness};
pub use transcript::Transcript;

use ark_bls12_381::Fr;

/// Field element type used throughout the crate
pub type Scalar = Fr;

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_compiles() {
        // Verify types are accessible (compilation test)
        let _: super::Scalar = super::Scalar::from(0u64);
    }
}
