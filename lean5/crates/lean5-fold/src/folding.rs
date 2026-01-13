//! Core folding operation for Nova-style proof compression
//!
//! Folds two relaxed R1CS instances into one, maintaining the
//! invariant that if both input instances are satisfiable, the
//! folded instance is also satisfiable.

use crate::r1cs::R1CSShape;
use crate::relaxed::{compute_cross_term, RelaxedR1CSInstance, RelaxedR1CSWitness};
use crate::transcript::{DomainSeparator, Transcript};
use crate::{error::FoldError, Scalar};
use ark_bls12_381::G1Projective as G1;
use ark_ff::UniformRand;

/// Result of a folding operation
#[derive(Clone, Debug)]
pub struct FoldingResult {
    /// Folded instance
    pub instance: RelaxedR1CSInstance,
    /// Folded witness
    pub witness: RelaxedR1CSWitness,
    /// Cross term used in folding (for verification)
    pub cross_term: Vec<Scalar>,
    /// Challenge used in folding
    pub challenge: Scalar,
}

/// Fold two relaxed R1CS instance/witness pairs
///
/// Given two satisfying pairs (U1, W1) and (U2, W2), produces a
/// new pair (U, W) that is satisfying if and only if both inputs were.
///
/// # Arguments
/// * `shape` - The R1CS shape (constraint matrices)
/// * `instance1` - First relaxed instance
/// * `witness1` - First relaxed witness
/// * `instance2` - Second relaxed instance
/// * `witness2` - Second relaxed witness
/// * `transcript` - Fiat-Shamir transcript for generating challenge
///
/// # Returns
/// A `FoldingResult` containing the folded instance and witness.
pub fn fold(
    shape: &R1CSShape,
    instance1: &RelaxedR1CSInstance,
    witness1: &RelaxedR1CSWitness,
    instance2: &RelaxedR1CSInstance,
    witness2: &RelaxedR1CSWitness,
    transcript: &mut Transcript,
) -> Result<FoldingResult, FoldError> {
    // Validate dimensions
    if witness1.w.len() != shape.num_vars {
        return Err(FoldError::DimensionMismatch(format!(
            "witness1 has {} vars, expected {}",
            witness1.w.len(),
            shape.num_vars
        )));
    }
    if witness2.w.len() != shape.num_vars {
        return Err(FoldError::DimensionMismatch(format!(
            "witness2 has {} vars, expected {}",
            witness2.w.len(),
            shape.num_vars
        )));
    }

    // Step 1: Compute cross term T
    let cross_term = compute_cross_term(
        shape,
        instance1.u,
        &instance1.x,
        &witness1.w,
        instance2.u,
        &instance2.x,
        &witness2.w,
    )?;

    // Step 2: Commit to cross term (placeholder - using random for now)
    // In a real implementation, this would use the commitment scheme
    let mut rng = ark_std::test_rng();
    let cross_term_commit = G1::rand(&mut rng);

    // Step 3: Generate challenge via Fiat-Shamir
    transcript.append_domain_separator(DomainSeparator::FoldingChallenge);
    transcript.append_point(b"commit_w1", &instance1.commit_w);
    transcript.append_point(b"commit_w2", &instance2.commit_w);
    transcript.append_point(b"commit_e1", &instance1.commit_e);
    transcript.append_point(b"commit_e2", &instance2.commit_e);
    transcript.append_scalar(b"u1", &instance1.u);
    transcript.append_scalar(b"u2", &instance2.u);
    transcript.append_scalars(b"x1", &instance1.x);
    transcript.append_scalars(b"x2", &instance2.x);
    transcript.append_point(b"cross_term_commit", &cross_term_commit);

    let r = transcript.squeeze_challenge();

    // Step 4: Fold instances
    let folded_instance = instance1.fold(instance2, r, cross_term_commit);

    // Step 5: Fold witnesses
    let folded_witness = witness1.fold(witness2, r, &cross_term)?;

    Ok(FoldingResult {
        instance: folded_instance,
        witness: folded_witness,
        cross_term,
        challenge: r,
    })
}

/// Non-interactive folding (generates transcript internally)
pub fn fold_noninteractive(
    shape: &R1CSShape,
    instance1: &RelaxedR1CSInstance,
    witness1: &RelaxedR1CSWitness,
    instance2: &RelaxedR1CSInstance,
    witness2: &RelaxedR1CSWitness,
) -> Result<FoldingResult, FoldError> {
    let mut transcript = Transcript::new();
    fold(
        shape,
        instance1,
        witness1,
        instance2,
        witness2,
        &mut transcript,
    )
}

/// Verify that a folded instance was computed correctly
///
/// Given the two original instances, the challenge, and the claimed
/// folded instance, verify the folding was done correctly.
pub fn verify_fold(
    instance1: &RelaxedR1CSInstance,
    instance2: &RelaxedR1CSInstance,
    r: Scalar,
    cross_term_commit: G1,
    claimed_instance: &RelaxedR1CSInstance,
) -> bool {
    let expected = instance1.fold(instance2, r, cross_term_commit);

    // Check all components match
    claimed_instance.commit_w == expected.commit_w
        && claimed_instance.commit_e == expected.commit_e
        && claimed_instance.u == expected.u
        && claimed_instance.x == expected.x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs::{R1CSBuilder, R1CSInstance, R1CSWitness};
    use crate::relaxed::is_relaxed_satisfied;

    /// Create a simple multiplication constraint: x * y = z
    fn create_mul_shape() -> R1CSShape {
        let mut builder = R1CSBuilder::new(0);
        let x = builder.alloc_var();
        let y = builder.alloc_var();
        let z = builder.alloc_var();

        builder.add_constraint(
            vec![(builder.var_idx(x), Scalar::from(1u64))],
            vec![(builder.var_idx(y), Scalar::from(1u64))],
            vec![(builder.var_idx(z), Scalar::from(1u64))],
        );

        builder.build().unwrap()
    }

    #[test]
    fn test_fold_satisfying_instances() {
        let shape = create_mul_shape();

        // Instance 1: 2 * 3 = 6
        let instance1 = R1CSInstance::new(vec![]);
        let witness1 = R1CSWitness::new(vec![
            Scalar::from(2u64),
            Scalar::from(3u64),
            Scalar::from(6u64),
        ]);

        // Verify instance1 satisfies standard R1CS
        assert!(shape.is_satisfied(&instance1, &witness1).unwrap());

        // Instance 2: 4 * 5 = 20
        let instance2 = R1CSInstance::new(vec![]);
        let witness2 = R1CSWitness::new(vec![
            Scalar::from(4u64),
            Scalar::from(5u64),
            Scalar::from(20u64),
        ]);

        // Verify instance2 satisfies standard R1CS
        assert!(shape.is_satisfied(&instance2, &witness2).unwrap());

        // Convert to relaxed form
        let relaxed_instance1 = RelaxedR1CSInstance::from_r1cs_instance(&instance1, G1::default());
        let relaxed_witness1 =
            RelaxedR1CSWitness::from_r1cs_witness(&witness1, shape.num_constraints);

        let relaxed_instance2 = RelaxedR1CSInstance::from_r1cs_instance(&instance2, G1::default());
        let relaxed_witness2 =
            RelaxedR1CSWitness::from_r1cs_witness(&witness2, shape.num_constraints);

        // Fold the instances
        let result = fold_noninteractive(
            &shape,
            &relaxed_instance1,
            &relaxed_witness1,
            &relaxed_instance2,
            &relaxed_witness2,
        )
        .unwrap();

        // Verify folded instance satisfies relaxed R1CS
        let satisfied = is_relaxed_satisfied(
            &shape,
            result.instance.u,
            &result.instance.x,
            &result.witness.w,
            &result.witness.e,
        )
        .unwrap();

        assert!(satisfied, "Folded instance should satisfy relaxed R1CS");
    }

    #[test]
    fn test_fold_deterministic() {
        let shape = create_mul_shape();

        let instance1 = R1CSInstance::new(vec![]);
        let witness1 = R1CSWitness::new(vec![
            Scalar::from(2u64),
            Scalar::from(3u64),
            Scalar::from(6u64),
        ]);

        let instance2 = R1CSInstance::new(vec![]);
        let witness2 = R1CSWitness::new(vec![
            Scalar::from(4u64),
            Scalar::from(5u64),
            Scalar::from(20u64),
        ]);

        let relaxed_instance1 = RelaxedR1CSInstance::from_r1cs_instance(&instance1, G1::default());
        let relaxed_witness1 =
            RelaxedR1CSWitness::from_r1cs_witness(&witness1, shape.num_constraints);

        let relaxed_instance2 = RelaxedR1CSInstance::from_r1cs_instance(&instance2, G1::default());
        let relaxed_witness2 =
            RelaxedR1CSWitness::from_r1cs_witness(&witness2, shape.num_constraints);

        // Fold twice with same inputs
        let mut t1 = Transcript::new();
        let mut t2 = Transcript::new();

        let result1 = fold(
            &shape,
            &relaxed_instance1,
            &relaxed_witness1,
            &relaxed_instance2,
            &relaxed_witness2,
            &mut t1,
        )
        .unwrap();

        let result2 = fold(
            &shape,
            &relaxed_instance1,
            &relaxed_witness1,
            &relaxed_instance2,
            &relaxed_witness2,
            &mut t2,
        )
        .unwrap();

        // Cross terms and challenges should be identical (deterministic)
        assert_eq!(result1.cross_term, result2.cross_term);
        assert_eq!(result1.challenge, result2.challenge);
        assert_eq!(result1.instance.u, result2.instance.u);
        assert_eq!(result1.instance.x, result2.instance.x);
        assert_eq!(result1.witness.w, result2.witness.w);
        assert_eq!(result1.witness.e, result2.witness.e);
    }

    #[test]
    fn test_multiple_folds() {
        let shape = create_mul_shape();

        // Create three instances
        let instances = [
            (2u64, 3u64, 6u64),   // 2 * 3 = 6
            (4u64, 5u64, 20u64),  // 4 * 5 = 20
            (7u64, 11u64, 77u64), // 7 * 11 = 77
        ];

        let relaxed: Vec<_> = instances
            .iter()
            .map(|(x, y, z)| {
                let inst = R1CSInstance::new(vec![]);
                let wit =
                    R1CSWitness::new(vec![Scalar::from(*x), Scalar::from(*y), Scalar::from(*z)]);
                (
                    RelaxedR1CSInstance::from_r1cs_instance(&inst, G1::default()),
                    RelaxedR1CSWitness::from_r1cs_witness(&wit, shape.num_constraints),
                )
            })
            .collect();

        // Fold first two
        let fold1 = fold_noninteractive(
            &shape,
            &relaxed[0].0,
            &relaxed[0].1,
            &relaxed[1].0,
            &relaxed[1].1,
        )
        .unwrap();

        // Verify intermediate result
        assert!(is_relaxed_satisfied(
            &shape,
            fold1.instance.u,
            &fold1.instance.x,
            &fold1.witness.w,
            &fold1.witness.e,
        )
        .unwrap());

        // Fold result with third
        let fold2 = fold_noninteractive(
            &shape,
            &fold1.instance,
            &fold1.witness,
            &relaxed[2].0,
            &relaxed[2].1,
        )
        .unwrap();

        // Verify final result
        assert!(is_relaxed_satisfied(
            &shape,
            fold2.instance.u,
            &fold2.instance.x,
            &fold2.witness.w,
            &fold2.witness.e,
        )
        .unwrap());
    }
}
