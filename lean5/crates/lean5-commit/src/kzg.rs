//! KZG (Kate-Zaverucha-Goldberg) Polynomial Commitment Scheme
//!
//! KZG commitments provide constant-size proofs and efficient verification
//! using bilinear pairings. Requires a trusted setup ceremony.

use ark_bls12_381::{Bls12_381, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ec::PrimeGroup;
use ark_ff::{Field, One, PrimeField, UniformRand};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::Rng;
use ark_std::Zero;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use lean5_kernel::ProofCert;

use crate::encoding::encode_cert;
use crate::error::{CommitError, VerifyError};
use crate::ProofCommitmentScheme;

/// KZG Structured Reference String (public parameters from trusted setup)
#[derive(Clone, Debug)]
pub struct KzgSrs {
    /// Powers of tau in G1: [G1, tau*G1, tau^2*G1, ..., tau^n*G1]
    pub powers_g1: Vec<G1Affine>,
    /// tau*G2 for pairing verification
    pub tau_g2: G2Affine,
    /// G2 generator
    pub g2: G2Affine,
    /// Maximum supported degree
    pub max_degree: usize,
}

/// KZG commitment (single G1 element)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KzgCommitment(#[serde(with = "g1_serde")] pub G1Affine);

/// KZG opening proof (single G1 element)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KzgOpeningProof(#[serde(with = "g1_serde")] pub G1Affine);

/// KZG polynomial commitment scheme
pub struct KzgScheme {
    srs: KzgSrs,
}

impl KzgScheme {
    /// Create a new KZG scheme with the given SRS
    pub fn new(srs: KzgSrs) -> Self {
        Self { srs }
    }

    /// Generate SRS via trusted setup (FOR TESTING ONLY)
    ///
    /// In production, use a properly conducted ceremony.
    pub fn setup<R: Rng>(max_degree: usize, rng: &mut R) -> Result<Self, CommitError> {
        // Generate secret tau
        let tau = Fr::rand(rng);

        // Compute powers of tau in G1
        let g1 = G1Projective::generator();
        let mut powers_g1 = Vec::with_capacity(max_degree + 1);
        let mut tau_power = Fr::from(1u64);

        for _ in 0..=max_degree {
            powers_g1.push((g1 * tau_power).into());
            tau_power *= tau;
        }

        // Compute tau*G2
        let g2: G2Affine = G2Projective::generator().into();
        let tau_g2: G2Affine = (G2Projective::generator() * tau).into();

        let srs = KzgSrs {
            powers_g1,
            tau_g2,
            g2,
            max_degree,
        };

        Ok(Self { srs })
    }

    /// Commit to a polynomial directly
    pub fn commit_poly(&self, poly: &DensePolynomial<Fr>) -> Result<KzgCommitment, CommitError> {
        if poly.degree() > self.srs.max_degree {
            return Err(CommitError::DegreeTooLarge(
                poly.degree(),
                self.srs.max_degree,
            ));
        }

        // Compute commitment: sum(coeff_i * tau^i * G1)
        let commitment = poly
            .coeffs
            .iter()
            .zip(self.srs.powers_g1.iter())
            .fold(G1Projective::zero(), |acc, (coeff, power)| {
                acc + *power * coeff
            });

        Ok(KzgCommitment(commitment.into()))
    }

    /// Open polynomial at a point
    pub fn open_poly(
        &self,
        poly: &DensePolynomial<Fr>,
        point: Fr,
    ) -> Result<(Fr, KzgOpeningProof), CommitError> {
        // Evaluate polynomial at point
        let value = poly.evaluate(&point);

        // Compute quotient polynomial: q(x) = (p(x) - p(point)) / (x - point)
        let numerator = poly - &DensePolynomial::from_coefficients_vec(vec![value]);

        // Synthetic division by (x - point)
        let divisor = DensePolynomial::from_coefficients_vec(vec![-point, Fr::from(1u64)]);
        let (quotient, remainder) = divide_polys(&numerator, &divisor);

        // Remainder should be zero (or very small due to floating point)
        debug_assert!(remainder.coeffs.iter().all(ark_std::Zero::is_zero));

        // Commit to quotient polynomial
        let proof = self.commit_poly(&quotient)?;

        Ok((value, KzgOpeningProof(proof.0)))
    }

    /// Verify opening proof using pairing check
    pub fn verify_opening(
        &self,
        commitment: &KzgCommitment,
        point: Fr,
        value: Fr,
        proof: &KzgOpeningProof,
    ) -> Result<bool, VerifyError> {
        // Pairing check: e(C - v*G1, G2) = e(proof, tau*G2 - point*G2)
        let g1 = G1Projective::generator();
        let lhs = G1Projective::from(commitment.0) - g1 * value;

        let g2 = G2Projective::generator();
        let rhs_g2 = G2Projective::from(self.srs.tau_g2) - g2 * point;

        // Check e(lhs, G2) == e(proof, rhs_g2)
        let pairing_lhs = Bls12_381::pairing(lhs, self.srs.g2);
        let pairing_rhs = Bls12_381::pairing(proof.0, rhs_g2);

        Ok(pairing_lhs == pairing_rhs)
    }
}

impl ProofCommitmentScheme for KzgScheme {
    type Commitment = KzgCommitment;
    type OpeningProof = KzgOpeningProof;
    type Fr = Fr;

    fn commit(&self, cert: &ProofCert) -> Result<Self::Commitment, CommitError> {
        let encoded = encode_cert(cert)?;
        self.commit_poly(&encoded.poly)
    }

    fn open(
        &self,
        cert: &ProofCert,
        point: Self::Fr,
    ) -> Result<(Self::Fr, Self::OpeningProof), CommitError> {
        let encoded = encode_cert(cert)?;
        self.open_poly(&encoded.poly, point)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        point: Self::Fr,
        value: Self::Fr,
        proof: &Self::OpeningProof,
    ) -> Result<bool, VerifyError> {
        self.verify_opening(commitment, point, value, proof)
    }

    fn batch_verify(
        &self,
        items: &[(Self::Commitment, Self::Fr, Self::Fr, Self::OpeningProof)],
    ) -> Result<bool, VerifyError> {
        if items.is_empty() {
            return Ok(true);
        }

        let challenges = derive_challenges(items)?;

        let g1 = G1Projective::generator();
        let g2 = G2Projective::generator();

        let mut acc_lhs = G1Projective::zero();
        let mut acc_tau = G1Projective::zero();
        let mut acc_point = G1Projective::zero();

        for ((commitment, point, value, proof), challenge) in items.iter().zip(challenges.iter()) {
            let scaled_challenge = *challenge;
            let commitment_term =
                (G1Projective::from(commitment.0) - g1 * *value) * scaled_challenge;
            let proof_point_term = G1Projective::from(proof.0) * (*point * scaled_challenge);
            let proof_tau_term = G1Projective::from(proof.0) * scaled_challenge;

            acc_lhs += commitment_term;
            acc_point += proof_point_term;
            acc_tau += proof_tau_term;
        }

        // Check e(acc_lhs, g2) * e(-acc_tau, tau_g2) * e(acc_point, g2) == 1
        let g1_inputs = [
            <Bls12_381 as Pairing>::G1Prepared::from(G1Affine::from(acc_lhs)),
            <Bls12_381 as Pairing>::G1Prepared::from(G1Affine::from(-acc_tau)),
            <Bls12_381 as Pairing>::G1Prepared::from(G1Affine::from(acc_point)),
        ];
        let g2_inputs = [
            <Bls12_381 as Pairing>::G2Prepared::from(self.srs.g2),
            <Bls12_381 as Pairing>::G2Prepared::from(self.srs.tau_g2),
            <Bls12_381 as Pairing>::G2Prepared::from(G2Affine::from(g2)),
        ];

        let ml_result = Bls12_381::multi_miller_loop(g1_inputs, g2_inputs);
        let result = Bls12_381::final_exponentiation(ml_result)
            .ok_or_else(|| VerifyError::VerificationFailed("final exponentiation failed".into()))?;

        Ok(result.0.is_one())
    }
}

/// Polynomial division with remainder
fn divide_polys(
    numerator: &DensePolynomial<Fr>,
    divisor: &DensePolynomial<Fr>,
) -> (DensePolynomial<Fr>, DensePolynomial<Fr>) {
    if numerator.is_zero() {
        return (
            DensePolynomial::from_coefficients_vec(vec![]),
            DensePolynomial::from_coefficients_vec(vec![]),
        );
    }

    assert!(!divisor.is_zero(), "Division by zero polynomial");

    if numerator.degree() < divisor.degree() {
        return (
            DensePolynomial::from_coefficients_vec(vec![]),
            numerator.clone(),
        );
    }

    let mut remainder = numerator.clone();
    let mut quotient_coeffs = vec![Fr::zero(); numerator.degree() - divisor.degree() + 1];

    let divisor_lead_inv = divisor
        .coeffs
        .last()
        .expect("divisor not zero")
        .inverse()
        .expect("leading coeff invertible");

    while !remainder.is_zero() && remainder.degree() >= divisor.degree() {
        let coeff = *remainder.coeffs.last().expect("remainder not zero") * divisor_lead_inv;
        let shift = remainder.degree() - divisor.degree();

        quotient_coeffs[shift] = coeff;

        // Subtract coeff * x^shift * divisor from remainder
        for (i, &d_coeff) in divisor.coeffs.iter().enumerate() {
            remainder.coeffs[i + shift] -= coeff * d_coeff;
        }

        // Remove leading zeros
        while !remainder.coeffs.is_empty() && remainder.coeffs.last() == Some(&Fr::zero()) {
            remainder.coeffs.pop();
        }
    }

    (
        DensePolynomial::from_coefficients_vec(quotient_coeffs),
        remainder,
    )
}

/// Serde support for G1Affine
mod g1_serde {
    use ark_bls12_381::G1Affine;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(point: &G1Affine, serializer: S) -> Result<S::Ok, S::Error> {
        let mut bytes = Vec::new();
        point
            .serialize_compressed(&mut bytes)
            .map_err(serde::ser::Error::custom)?;
        bytes.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<G1Affine, D::Error> {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        G1Affine::deserialize_compressed(&bytes[..]).map_err(serde::de::Error::custom)
    }
}

/// Derive deterministic Fiat-Shamir challenges for batch verification
fn derive_challenges(
    items: &[(KzgCommitment, Fr, Fr, KzgOpeningProof)],
) -> Result<Vec<Fr>, VerifyError> {
    let mut challenges = Vec::with_capacity(items.len());

    for (idx, (commitment, point, value, proof)) in items.iter().enumerate() {
        let mut buffer = Vec::new();
        commitment
            .0
            .serialize_compressed(&mut buffer)
            .map_err(|e| VerifyError::VerificationFailed(format!("serialize commitment: {e}")))?;
        point
            .serialize_compressed(&mut buffer)
            .map_err(|e| VerifyError::VerificationFailed(format!("serialize point: {e}")))?;
        value
            .serialize_compressed(&mut buffer)
            .map_err(|e| VerifyError::VerificationFailed(format!("serialize value: {e}")))?;
        proof
            .0
            .serialize_compressed(&mut buffer)
            .map_err(|e| VerifyError::VerificationFailed(format!("serialize proof: {e}")))?;
        buffer.extend_from_slice(&idx.to_le_bytes());

        let digest = Sha256::digest(buffer);
        let mut challenge = Fr::from_le_bytes_mod_order(&digest);
        if challenge.is_zero() {
            challenge = Fr::from(1u64);
        }

        challenges.push(challenge);
    }

    Ok(challenges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::test_rng;

    #[test]
    fn test_kzg_setup() {
        let mut rng = test_rng();
        let kzg = KzgScheme::setup(16, &mut rng).expect("setup should succeed");
        assert_eq!(kzg.srs.powers_g1.len(), 17);
    }

    #[test]
    fn test_kzg_commit_verify() {
        let mut rng = test_rng();
        let kzg = KzgScheme::setup(16, &mut rng).expect("setup should succeed");

        // Create simple polynomial
        let poly = DensePolynomial::from_coefficients_vec(vec![
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
        ]);

        let commitment = kzg.commit_poly(&poly).expect("commit should succeed");

        let point = Fr::from(5u64);
        let (value, proof) = kzg.open_poly(&poly, point).expect("open should succeed");

        // Verify: p(5) = 1 + 2*5 + 3*25 = 1 + 10 + 75 = 86
        assert_eq!(value, Fr::from(86u64));

        let valid = kzg
            .verify_opening(&commitment, point, value, &proof)
            .expect("verify should succeed");
        assert!(valid);

        // Verify with wrong value should fail
        let invalid = kzg
            .verify_opening(&commitment, point, Fr::from(87u64), &proof)
            .expect("verify should succeed");
        assert!(!invalid);
    }

    #[test]
    fn test_kzg_with_cert() {
        use lean5_kernel::Level;
        use std::sync::Arc;

        let mut rng = test_rng();
        let kzg = KzgScheme::setup(1024, &mut rng).expect("setup should succeed");

        let cert = ProofCert::Sort {
            level: Level::Succ(Arc::new(Level::Zero)),
        };

        let commitment = kzg.commit(&cert).expect("commit should succeed");

        let point = Fr::rand(&mut rng);
        let (value, proof) = kzg.open(&cert, point).expect("open should succeed");

        let valid = kzg
            .verify(&commitment, point, value, &proof)
            .expect("verify should succeed");
        assert!(valid);
    }

    #[test]
    fn test_kzg_batch_verify_valid() {
        let mut rng = test_rng();
        let kzg = KzgScheme::setup(32, &mut rng).expect("setup should succeed");

        let poly1 = DensePolynomial::from_coefficients_vec(vec![Fr::from(3u64), Fr::from(2u64)]);
        let poly2 = DensePolynomial::from_coefficients_vec(vec![
            Fr::from(5u64),
            Fr::from(1u64),
            Fr::from(4u64),
        ]);

        let commitment1 = kzg.commit_poly(&poly1).expect("commitment should work");
        let commitment2 = kzg.commit_poly(&poly2).expect("commitment should work");

        let point1 = Fr::from(7u64);
        let point2 = Fr::from(11u64);

        let (value1, proof1) = kzg.open_poly(&poly1, point1).expect("open should succeed");
        let (value2, proof2) = kzg.open_poly(&poly2, point2).expect("open should succeed");

        let batch = vec![
            (commitment1, point1, value1, proof1),
            (commitment2, point2, value2, proof2),
        ];

        let valid = kzg
            .batch_verify(&batch)
            .expect("batch verify should succeed");
        assert!(valid);
    }

    #[test]
    fn test_kzg_batch_verify_detects_invalid() {
        let mut rng = test_rng();
        let kzg = KzgScheme::setup(32, &mut rng).expect("setup should succeed");

        let poly = DensePolynomial::from_coefficients_vec(vec![Fr::from(2u64), Fr::from(9u64)]);
        let commitment = kzg.commit_poly(&poly).expect("commitment should work");

        let point = Fr::from(5u64);
        let (value, proof) = kzg.open_poly(&poly, point).expect("open should succeed");

        let mut batch = vec![(commitment.clone(), point, value, proof.clone())];
        batch.push((commitment, point + Fr::from(1u64), value, proof));

        // Tamper with the second value so aggregation must catch the error
        batch[1].2 += Fr::from(1u64);

        let valid = kzg
            .batch_verify(&batch)
            .expect("batch verify should succeed");
        assert!(!valid);
    }
}
