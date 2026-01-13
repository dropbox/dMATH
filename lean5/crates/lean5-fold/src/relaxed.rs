//! Relaxed R1CS for Nova-style folding
//!
//! Standard R1CS: Az ∘ Bz = Cz
//! Relaxed R1CS: Az ∘ Bz = u·Cz + E
//!
//! The relaxation scalar `u` and error vector `E` enable folding:
//! - Standard R1CS has u=1, E=0
//! - Folding two instances produces a new relaxed instance
//! - Final verification still checks the relation holds

use crate::r1cs::{R1CSInstance, R1CSShape, R1CSWitness};
use crate::{error::FoldError, Scalar};
use ark_bls12_381::G1Projective as G1;
use ark_ff::Zero;

/// Relaxed R1CS instance (public data)
///
/// Contains commitments to the witness and error vector,
/// the relaxation scalar, and public inputs.
#[derive(Clone, Debug)]
pub struct RelaxedR1CSInstance {
    /// Commitment to the witness W
    pub commit_w: G1,
    /// Commitment to the error vector E
    pub commit_e: G1,
    /// Relaxation scalar (u=1 for standard R1CS)
    pub u: Scalar,
    /// Public inputs
    pub x: Vec<Scalar>,
}

impl RelaxedR1CSInstance {
    /// Create a default (zero) relaxed instance
    pub fn default(num_io: usize) -> Self {
        Self {
            commit_w: G1::zero(),
            commit_e: G1::zero(),
            u: Scalar::zero(),
            x: vec![Scalar::zero(); num_io],
        }
    }

    /// Create from a standard R1CS instance
    ///
    /// The error vector is zero and u=1 for a standard instance.
    pub fn from_r1cs_instance(instance: &R1CSInstance, commit_w: G1) -> Self {
        Self {
            commit_w,
            commit_e: G1::zero(),
            u: Scalar::from(1u64),
            x: instance.x.clone(),
        }
    }

    /// Linear combination of two instances: self + r * other
    #[must_use]
    pub fn fold(&self, other: &Self, r: Scalar, cross_term_commit: G1) -> Self {
        Self {
            // Commit_W = Commit_W1 + r * Commit_W2
            commit_w: self.commit_w + other.commit_w * r,
            // Commit_E = Commit_E1 + r * T + r^2 * Commit_E2
            commit_e: self.commit_e + cross_term_commit * r + other.commit_e * (r * r),
            // u = u1 + r * u2
            u: self.u + r * other.u,
            // x = x1 + r * x2
            x: self
                .x
                .iter()
                .zip(&other.x)
                .map(|(a, b)| *a + r * *b)
                .collect(),
        }
    }
}

/// Relaxed R1CS witness (private data)
#[derive(Clone, Debug)]
pub struct RelaxedR1CSWitness {
    /// Private witness elements
    pub w: Vec<Scalar>,
    /// Error vector
    pub e: Vec<Scalar>,
}

impl RelaxedR1CSWitness {
    /// Create a default (zero) relaxed witness
    pub fn default(num_vars: usize, num_constraints: usize) -> Self {
        Self {
            w: vec![Scalar::zero(); num_vars],
            e: vec![Scalar::zero(); num_constraints],
        }
    }

    /// Create from a standard R1CS witness
    ///
    /// The error vector is zero for a satisfying witness.
    pub fn from_r1cs_witness(witness: &R1CSWitness, num_constraints: usize) -> Self {
        Self {
            w: witness.w.clone(),
            e: vec![Scalar::zero(); num_constraints],
        }
    }

    /// Linear combination of two witnesses: self + r * other
    pub fn fold(&self, other: &Self, r: Scalar, cross_term: &[Scalar]) -> Result<Self, FoldError> {
        if cross_term.len() != self.e.len() {
            return Err(FoldError::DimensionMismatch(format!(
                "cross term has {} elements, expected {}",
                cross_term.len(),
                self.e.len()
            )));
        }

        Ok(Self {
            // W = W1 + r * W2
            w: self
                .w
                .iter()
                .zip(&other.w)
                .map(|(a, b)| *a + r * *b)
                .collect(),
            // E = E1 + r * T + r^2 * E2
            e: self
                .e
                .iter()
                .zip(&other.e)
                .zip(cross_term)
                .map(|((e1, e2), t)| *e1 + r * *t + r * r * *e2)
                .collect(),
        })
    }
}

/// Check if a relaxed R1CS instance/witness pair satisfies the relation
///
/// Verifies: Az ∘ Bz = u·Cz + E
pub fn is_relaxed_satisfied(
    shape: &R1CSShape,
    u: Scalar,
    x: &[Scalar],
    w: &[Scalar],
    e: &[Scalar],
) -> Result<bool, FoldError> {
    // Build z = (u, x, W) - note: in relaxed R1CS, the constant is u not 1
    let mut z = Vec::with_capacity(shape.num_z());
    z.push(u);
    z.extend_from_slice(x);
    z.extend_from_slice(w);

    let az = shape.a.mul_vec(&z)?;
    let bz = shape.b.mul_vec(&z)?;
    let cz = shape.c.mul_vec(&z)?;

    // Check Az ∘ Bz = u·Cz + E for each constraint
    for i in 0..shape.num_constraints {
        let lhs = az[i] * bz[i];
        let rhs = u * cz[i] + e[i];
        if lhs != rhs {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Compute the cross term T for folding two instances
///
/// T = Az1 ∘ Bz2 + Az2 ∘ Bz1 - u1·Cz2 - u2·Cz1
///
/// This is the "error" introduced by folding, which gets absorbed
/// into the error vector of the folded instance.
pub fn compute_cross_term(
    shape: &R1CSShape,
    u1: Scalar,
    x1: &[Scalar],
    w1: &[Scalar],
    u2: Scalar,
    x2: &[Scalar],
    w2: &[Scalar],
) -> Result<Vec<Scalar>, FoldError> {
    // Build z1 and z2
    let mut z1 = Vec::with_capacity(shape.num_z());
    z1.push(u1);
    z1.extend_from_slice(x1);
    z1.extend_from_slice(w1);

    let mut z2 = Vec::with_capacity(shape.num_z());
    z2.push(u2);
    z2.extend_from_slice(x2);
    z2.extend_from_slice(w2);

    // Compute matrix-vector products
    let az1 = shape.a.mul_vec(&z1)?;
    let bz1 = shape.b.mul_vec(&z1)?;
    let cz1 = shape.c.mul_vec(&z1)?;

    let az2 = shape.a.mul_vec(&z2)?;
    let bz2 = shape.b.mul_vec(&z2)?;
    let cz2 = shape.c.mul_vec(&z2)?;

    // T = Az1 ∘ Bz2 + Az2 ∘ Bz1 - u1·Cz2 - u2·Cz1
    let cross_term: Vec<Scalar> = (0..shape.num_constraints)
        .map(|i| az1[i] * bz2[i] + az2[i] * bz1[i] - u1 * cz2[i] - u2 * cz1[i])
        .collect();

    Ok(cross_term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs::R1CSBuilder;

    #[test]
    fn test_standard_r1cs_as_relaxed() {
        // x * y = z with x=3, y=4, z=12
        let mut builder = R1CSBuilder::new(0);
        let x = builder.alloc_var();
        let y = builder.alloc_var();
        let z = builder.alloc_var();

        builder.add_constraint(
            vec![(builder.var_idx(x), Scalar::from(1u64))],
            vec![(builder.var_idx(y), Scalar::from(1u64))],
            vec![(builder.var_idx(z), Scalar::from(1u64))],
        );

        let shape = builder.build().unwrap();

        // Standard instance: u=1, E=0
        let u = Scalar::from(1u64);
        let public_x: Vec<Scalar> = vec![];
        let w = vec![Scalar::from(3u64), Scalar::from(4u64), Scalar::from(12u64)];
        let e = vec![Scalar::zero()];

        assert!(is_relaxed_satisfied(&shape, u, &public_x, &w, &e).unwrap());
    }

    #[test]
    fn test_cross_term_computation() {
        // x * y = z
        let mut builder = R1CSBuilder::new(0);
        let x = builder.alloc_var();
        let y = builder.alloc_var();
        let z = builder.alloc_var();

        builder.add_constraint(
            vec![(builder.var_idx(x), Scalar::from(1u64))],
            vec![(builder.var_idx(y), Scalar::from(1u64))],
            vec![(builder.var_idx(z), Scalar::from(1u64))],
        );

        let shape = builder.build().unwrap();

        // Instance 1: x=2, y=3, z=6 (2*3=6)
        let u1 = Scalar::from(1u64);
        let x1: Vec<Scalar> = vec![];
        let w1 = vec![Scalar::from(2u64), Scalar::from(3u64), Scalar::from(6u64)];

        // Instance 2: x=4, y=5, z=20 (4*5=20)
        let u2 = Scalar::from(1u64);
        let x2: Vec<Scalar> = vec![];
        let w2 = vec![Scalar::from(4u64), Scalar::from(5u64), Scalar::from(20u64)];

        let cross_term = compute_cross_term(&shape, u1, &x1, &w1, u2, &x2, &w2).unwrap();

        // T = A(z1) ∘ B(z2) + A(z2) ∘ B(z1) - u1·C(z2) - u2·C(z1)
        // For x*y=z: A extracts x, B extracts y, C extracts z
        // T = x1 * y2 + x2 * y1 - z2 - z1
        // T = 2*5 + 4*3 - 20 - 6 = 10 + 12 - 26 = -4
        assert_eq!(cross_term.len(), 1);
        assert_eq!(cross_term[0], Scalar::from(0u64) - Scalar::from(4u64));
    }
}
