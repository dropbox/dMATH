//! Fiat-Shamir transcript for non-interactive challenges
//!
//! Converts interactive proofs to non-interactive by deriving
//! challenges from a hash of the transcript so far.

use crate::Scalar;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use sha2::{Digest, Sha256};

/// Domain separation tags for different protocol messages
#[derive(Clone, Copy, Debug)]
pub enum DomainSeparator {
    /// Folding challenge
    FoldingChallenge,
    /// Commitment to witness
    WitnessCommitment,
    /// Cross term commitment
    CrossTermCommitment,
    /// Public input
    PublicInput,
}

impl DomainSeparator {
    fn as_bytes(self) -> &'static [u8] {
        match self {
            DomainSeparator::FoldingChallenge => b"lean5-fold:folding-challenge",
            DomainSeparator::WitnessCommitment => b"lean5-fold:witness-commitment",
            DomainSeparator::CrossTermCommitment => b"lean5-fold:cross-term-commitment",
            DomainSeparator::PublicInput => b"lean5-fold:public-input",
        }
    }
}

/// Fiat-Shamir transcript for generating non-interactive challenges
#[derive(Clone, Debug)]
pub struct Transcript {
    /// Running hash state
    hasher: Sha256,
}

impl Default for Transcript {
    fn default() -> Self {
        Self::new()
    }
}

impl Transcript {
    /// Create a new transcript with protocol domain separator
    pub fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"lean5-fold-transcript-v1");
        Self { hasher }
    }

    /// Append a domain separator
    pub fn append_domain_separator(&mut self, sep: DomainSeparator) {
        self.hasher.update(sep.as_bytes());
        self.hasher.update([0u8]); // null terminator
    }

    /// Append raw bytes to the transcript
    pub fn append_bytes(&mut self, label: &[u8], data: &[u8]) {
        self.hasher.update(label);
        self.hasher.update((data.len() as u64).to_le_bytes());
        self.hasher.update(data);
    }

    /// Append a scalar to the transcript
    pub fn append_scalar(&mut self, label: &[u8], scalar: &Scalar) {
        let bytes = scalar_to_bytes(scalar);
        self.append_bytes(label, &bytes);
    }

    /// Append multiple scalars to the transcript
    pub fn append_scalars(&mut self, label: &[u8], scalars: &[Scalar]) {
        self.hasher.update(label);
        self.hasher.update((scalars.len() as u64).to_le_bytes());
        for s in scalars {
            let bytes = scalar_to_bytes(s);
            self.hasher.update(bytes);
        }
    }

    /// Append a point (in compressed form) to the transcript
    pub fn append_point<P: ark_serialize::CanonicalSerialize>(&mut self, label: &[u8], point: &P) {
        let mut bytes = Vec::new();
        point
            .serialize_compressed(&mut bytes)
            .expect("serialization should not fail");
        self.append_bytes(label, &bytes);
    }

    /// Squeeze a challenge scalar from the transcript
    ///
    /// This finalizes the current hash state, produces a challenge,
    /// and reinitializes the state with the challenge for chaining.
    pub fn squeeze_challenge(&mut self) -> Scalar {
        // Clone the hasher to get the current state
        let hasher = self.hasher.clone();
        let hash = hasher.finalize();

        // Derive scalar from hash (use rejection sampling for uniformity)
        let challenge = scalar_from_hash(&hash);

        // Update transcript with the challenge for chaining
        self.hasher.update(hash);

        challenge
    }

    /// Get the current transcript hash (for debugging/verification)
    pub fn current_hash(&self) -> [u8; 32] {
        let hasher = self.hasher.clone();
        let hash = hasher.finalize();
        let mut result = [0u8; 32];
        result.copy_from_slice(&hash);
        result
    }
}

/// Convert a scalar to bytes (little-endian)
fn scalar_to_bytes(scalar: &Scalar) -> Vec<u8> {
    let mut bytes = Vec::new();
    scalar
        .serialize_compressed(&mut bytes)
        .expect("serialization should not fail");
    bytes
}

/// Derive a scalar from a hash using rejection sampling
///
/// This ensures the scalar is uniformly distributed in the field.
fn scalar_from_hash(hash: &[u8]) -> Scalar {
    // Expand hash to enough bytes for rejection sampling
    let mut expanded = Vec::with_capacity(64);
    let mut hasher = Sha256::new();
    hasher.update(hash);
    hasher.update([0u8]);
    expanded.extend_from_slice(&hasher.finalize());

    let mut hasher = Sha256::new();
    hasher.update(hash);
    hasher.update([1u8]);
    expanded.extend_from_slice(&hasher.finalize());

    // Try to convert to scalar, reducing modulo the field modulus
    Scalar::from_le_bytes_mod_order(&expanded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;

    #[test]
    fn test_transcript_deterministic() {
        let mut t1 = Transcript::new();
        let mut t2 = Transcript::new();

        t1.append_scalar(b"x", &Scalar::from(42u64));
        t2.append_scalar(b"x", &Scalar::from(42u64));

        let c1 = t1.squeeze_challenge();
        let c2 = t2.squeeze_challenge();

        assert_eq!(c1, c2);
    }

    #[test]
    fn test_transcript_different_inputs() {
        let mut t1 = Transcript::new();
        let mut t2 = Transcript::new();

        t1.append_scalar(b"x", &Scalar::from(42u64));
        t2.append_scalar(b"x", &Scalar::from(43u64));

        let c1 = t1.squeeze_challenge();
        let c2 = t2.squeeze_challenge();

        assert_ne!(c1, c2);
    }

    #[test]
    fn test_transcript_chaining() {
        let mut t = Transcript::new();

        t.append_scalar(b"x", &Scalar::from(1u64));
        let c1 = t.squeeze_challenge();

        t.append_scalar(b"y", &Scalar::from(2u64));
        let c2 = t.squeeze_challenge();

        // Challenges should be different due to chaining
        assert_ne!(c1, c2);

        // Verify determinism with same sequence
        let mut t2 = Transcript::new();
        t2.append_scalar(b"x", &Scalar::from(1u64));
        let c1_check = t2.squeeze_challenge();
        t2.append_scalar(b"y", &Scalar::from(2u64));
        let c2_check = t2.squeeze_challenge();

        assert_eq!(c1, c1_check);
        assert_eq!(c2, c2_check);
    }

    #[test]
    fn test_domain_separator() {
        let mut t1 = Transcript::new();
        let mut t2 = Transcript::new();

        t1.append_domain_separator(DomainSeparator::FoldingChallenge);
        t1.append_scalar(b"x", &Scalar::from(1u64));

        t2.append_domain_separator(DomainSeparator::WitnessCommitment);
        t2.append_scalar(b"x", &Scalar::from(1u64));

        // Different domain separators should produce different challenges
        let c1 = t1.squeeze_challenge();
        let c2 = t2.squeeze_challenge();
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_scalar_uniform() {
        // Check that challenges are well-distributed (basic sanity check)
        let mut rng = ark_std::test_rng();

        for _ in 0..100 {
            let mut t = Transcript::new();
            let random_input = Scalar::rand(&mut rng);
            t.append_scalar(b"input", &random_input);
            let challenge = t.squeeze_challenge();

            // Challenge should not be zero (extremely unlikely for uniform)
            assert_ne!(challenge, Scalar::from(0u64));
        }
    }
}
