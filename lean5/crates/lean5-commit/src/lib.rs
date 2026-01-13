//! Polynomial Commitment Schemes for Lean5 Proof Certificates
//!
//! This crate provides polynomial commitment schemes (KZG and IPA) for
//! compressing and efficiently verifying lean5 proof certificates.
//!
//! # Overview
//!
//! Polynomial commitments allow committing to a proof certificate as a
//! single group element (commitment), then proving properties about the
//! committed data without revealing it. This enables:
//!
//! - **Smaller proofs**: Commitment is constant size regardless of proof size
//! - **Batch verification**: Multiple proofs verified with single pairing check
//! - **Incremental updates**: Modify proofs without full reconstruction
//!
//! # Commitment Schemes
//!
//! Two schemes are supported:
//!
//! - **KZG (Kate-Zaverucha-Goldberg)**: Requires trusted setup but provides
//!   constant-size proofs and efficient verification via pairings.
//!
//! - **IPA (Inner Product Argument)**: No trusted setup (transparent) but
//!   logarithmic proof size and slower verification.
//!
//! # Example
//!
//! ```ignore
//! use lean5_commit::{KzgScheme, ProofCommitmentScheme};
//! use lean5_kernel::ProofCert;
//!
//! // Setup KZG with max degree
//! let kzg = KzgScheme::setup(1 << 16)?;
//!
//! // Commit to a proof certificate
//! let commitment = kzg.commit(&cert)?;
//!
//! // Open at a challenge point
//! let (value, proof) = kzg.open(&cert, challenge)?;
//!
//! // Verify opening
//! assert!(kzg.verify(&commitment, challenge, value, &proof)?);
//! ```

pub mod encoding;
pub mod error;
pub mod ipa;
pub mod kzg;

pub use encoding::{decode_cert, encode_cert, EncodedCert};
pub use error::{CommitError, VerifyError};
pub use ipa::IpaScheme;
pub use kzg::KzgScheme;

use ark_ff::Field;
use lean5_kernel::ProofCert;

/// A batch verification item containing commitment, point, value, and proof
pub type BatchVerifyItem<C, F, P> = (C, F, F, P);

/// Trait for polynomial commitment schemes over proof certificates
pub trait ProofCommitmentScheme {
    /// The commitment type (typically a group element)
    type Commitment: Clone;

    /// Opening proof type
    type OpeningProof: Clone;

    /// Field element type
    type Fr: Field;

    /// Commit to a proof certificate
    ///
    /// Returns a succinct commitment that binds to the certificate content.
    fn commit(&self, cert: &ProofCert) -> Result<Self::Commitment, CommitError>;

    /// Open commitment at a point
    ///
    /// Returns the evaluation and a proof that the evaluation is correct.
    fn open(
        &self,
        cert: &ProofCert,
        point: Self::Fr,
    ) -> Result<(Self::Fr, Self::OpeningProof), CommitError>;

    /// Verify an opening proof
    ///
    /// Returns true if the opening proof is valid, i.e., the committed
    /// polynomial evaluates to `value` at `point`.
    fn verify(
        &self,
        commitment: &Self::Commitment,
        point: Self::Fr,
        value: Self::Fr,
        proof: &Self::OpeningProof,
    ) -> Result<bool, VerifyError>;

    /// Batch verify multiple opening proofs
    ///
    /// More efficient than verifying individually when checking many proofs.
    fn batch_verify(
        &self,
        items: &[BatchVerifyItem<Self::Commitment, Self::Fr, Self::OpeningProof>],
    ) -> Result<bool, VerifyError>;
}

/// A committed proof certificate
#[derive(Clone, Debug)]
pub struct CommittedProof<C> {
    /// The polynomial commitment
    pub commitment: C,
    /// Degree of the underlying polynomial
    pub degree: usize,
    /// Original certificate hash for identification
    pub cert_hash: [u8; 32],
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_compiles() {
        // Verify types are accessible (compilation test)
        let _: Option<super::CommittedProof<()>> = None;
    }
}
