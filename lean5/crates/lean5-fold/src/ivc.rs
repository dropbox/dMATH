//! Incrementally Verifiable Computation (IVC) proofs
//!
//! IVC enables proving a sequence of computations incrementally:
//! - Start with an initial proof
//! - Extend with additional proofs
//! - Verify the accumulated proof at any point
//!
//! This is built on top of the folding scheme.

use crate::cert_encoding::encode_cert_to_r1cs;
use crate::error::IvcError;
use crate::folding::fold;
use crate::r1cs::{R1CSInstance, R1CSShape, R1CSWitness};
use crate::relaxed::{is_relaxed_satisfied, RelaxedR1CSInstance, RelaxedR1CSWitness};
use crate::transcript::Transcript;
use crate::Scalar;
use ark_bls12_381::G1Projective as G1;
use lean5_kernel::{Environment, ProofCert};

/// An IVC proof accumulating multiple R1CS instances
#[derive(Clone, Debug)]
pub struct IvcProof {
    /// The R1CS shape (constraint structure)
    pub shape: R1CSShape,
    /// Running folded instance
    pub running_instance: RelaxedR1CSInstance,
    /// Running folded witness
    pub running_witness: RelaxedR1CSWitness,
    /// Fiat-Shamir transcript state
    pub transcript: Transcript,
    /// Number of instances folded
    pub step: u64,
}

/// Compressed IVC proof (for final verification)
#[derive(Clone, Debug)]
pub struct CompressedIvcProof {
    /// Relaxation scalar
    pub u: Scalar,
    /// Public inputs
    pub x: Vec<Scalar>,
    /// Witness (for the verifier to check)
    pub w: Vec<Scalar>,
    /// Error vector
    pub e: Vec<Scalar>,
    /// Number of steps
    pub step: u64,
}

/// Start an IVC proof from an initial R1CS instance
///
/// # Arguments
/// * `shape` - The R1CS constraint structure
/// * `instance` - Initial public instance
/// * `witness` - Initial private witness
///
/// # Returns
/// A new IVC proof initialized with the given instance.
pub fn start_ivc(
    shape: R1CSShape,
    instance: &R1CSInstance,
    witness: &R1CSWitness,
) -> Result<IvcProof, IvcError> {
    // Verify the initial instance satisfies the constraints
    if !shape
        .is_satisfied(instance, witness)
        .map_err(|e| IvcError::VerificationFailed(format!("initial instance unsatisfied: {e}")))?
    {
        return Err(IvcError::VerificationFailed(
            "initial instance does not satisfy R1CS".to_string(),
        ));
    }

    // Convert to relaxed form
    let running_instance = RelaxedR1CSInstance::from_r1cs_instance(instance, G1::default());
    let running_witness = RelaxedR1CSWitness::from_r1cs_witness(witness, shape.num_constraints);

    Ok(IvcProof {
        shape,
        running_instance,
        running_witness,
        transcript: Transcript::new(),
        step: 1,
    })
}

/// Extend an IVC proof with a new R1CS instance
///
/// Folds the new instance into the running accumulator.
///
/// # Arguments
/// * `ivc` - The IVC proof to extend (modified in place)
/// * `instance` - New public instance to fold in
/// * `witness` - New private witness
///
/// # Returns
/// Ok(()) on success, error if folding fails.
pub fn extend_ivc(
    ivc: &mut IvcProof,
    instance: &R1CSInstance,
    witness: &R1CSWitness,
) -> Result<(), IvcError> {
    // Verify the new instance satisfies the constraints
    if !ivc
        .shape
        .is_satisfied(instance, witness)
        .map_err(|e| IvcError::VerificationFailed(format!("new instance unsatisfied: {e}")))?
    {
        return Err(IvcError::VerificationFailed(
            "new instance does not satisfy R1CS".to_string(),
        ));
    }

    // Convert to relaxed form
    let new_instance = RelaxedR1CSInstance::from_r1cs_instance(instance, G1::default());
    let new_witness = RelaxedR1CSWitness::from_r1cs_witness(witness, ivc.shape.num_constraints);

    // Fold into running accumulator
    let result = fold(
        &ivc.shape,
        &ivc.running_instance,
        &ivc.running_witness,
        &new_instance,
        &new_witness,
        &mut ivc.transcript,
    )?;

    ivc.running_instance = result.instance;
    ivc.running_witness = result.witness;
    ivc.step += 1;

    Ok(())
}

/// Verify an IVC proof
///
/// Checks that the accumulated instance satisfies the relaxed R1CS relation.
///
/// # Arguments
/// * `ivc` - The IVC proof to verify
///
/// # Returns
/// true if the proof is valid, false otherwise.
pub fn verify_ivc(ivc: &IvcProof) -> Result<bool, IvcError> {
    is_relaxed_satisfied(
        &ivc.shape,
        ivc.running_instance.u,
        &ivc.running_instance.x,
        &ivc.running_witness.w,
        &ivc.running_witness.e,
    )
    .map_err(|e| IvcError::VerificationFailed(format!("verification error: {e}")))
}

/// Compress an IVC proof to a succinct form
///
/// The compressed proof contains only the necessary data for
/// verification without the full constraint matrices.
pub fn compress_ivc(ivc: &IvcProof) -> CompressedIvcProof {
    CompressedIvcProof {
        u: ivc.running_instance.u,
        x: ivc.running_instance.x.clone(),
        w: ivc.running_witness.w.clone(),
        e: ivc.running_witness.e.clone(),
        step: ivc.step,
    }
}

/// Verify a compressed IVC proof against a shape
pub fn verify_compressed(shape: &R1CSShape, proof: &CompressedIvcProof) -> Result<bool, IvcError> {
    is_relaxed_satisfied(shape, proof.u, &proof.x, &proof.w, &proof.e)
        .map_err(|e| IvcError::VerificationFailed(format!("verification error: {e}")))
}

/// Start an IVC proof from a proof certificate
///
/// Encodes the certificate as R1CS constraints and initializes an IVC proof.
///
/// # Arguments
/// * `cert` - The proof certificate to encode
/// * `_env` - The environment (reserved for future use in full verification)
///
/// # Returns
/// A new IVC proof initialized from the certificate.
pub fn start_ivc_from_cert(cert: &ProofCert, _env: &Environment) -> Result<IvcProof, IvcError> {
    // Encode the certificate to R1CS
    let encoded = encode_cert_to_r1cs(cert)?;

    // Start IVC with the encoded R1CS
    start_ivc(encoded.shape, &encoded.instance, &encoded.witness)
}

/// Extend an IVC proof with a proof certificate
///
/// Encodes the certificate and folds it into the running IVC accumulator.
///
/// # Arguments
/// * `ivc` - The IVC proof to extend (modified in place)
/// * `cert` - The new certificate to fold in
/// * `_env` - The environment (reserved for future use in full verification)
///
/// # Returns
/// Ok(()) on success, error if encoding or folding fails.
///
/// # Note
/// The new certificate must produce an R1CS with the same shape as the
/// initial certificate. Certificates with different structures will fail.
pub fn extend_ivc_with_cert(
    ivc: &mut IvcProof,
    cert: &ProofCert,
    _env: &Environment,
) -> Result<(), IvcError> {
    // Encode the certificate to R1CS
    let encoded = encode_cert_to_r1cs(cert)?;

    // Verify shape compatibility
    if encoded.shape.num_constraints != ivc.shape.num_constraints {
        return Err(IvcError::VerificationFailed(format!(
            "shape mismatch: IVC has {} constraints, new cert has {}",
            ivc.shape.num_constraints, encoded.shape.num_constraints
        )));
    }

    if encoded.shape.num_vars != ivc.shape.num_vars {
        return Err(IvcError::VerificationFailed(format!(
            "shape mismatch: IVC has {} vars, new cert has {}",
            ivc.shape.num_vars, encoded.shape.num_vars
        )));
    }

    // Extend IVC with the encoded instance
    extend_ivc(ivc, &encoded.instance, &encoded.witness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs::R1CSBuilder;

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
    fn test_ivc_start() {
        let shape = create_mul_shape();
        let instance = R1CSInstance::new(vec![]);
        let witness = R1CSWitness::new(vec![
            Scalar::from(2u64),
            Scalar::from(3u64),
            Scalar::from(6u64),
        ]);

        let ivc = start_ivc(shape, &instance, &witness).unwrap();

        assert_eq!(ivc.step, 1);
        assert!(verify_ivc(&ivc).unwrap());
    }

    #[test]
    fn test_ivc_extend() {
        let shape = create_mul_shape();

        // Start with first instance: 2 * 3 = 6
        let instance1 = R1CSInstance::new(vec![]);
        let witness1 = R1CSWitness::new(vec![
            Scalar::from(2u64),
            Scalar::from(3u64),
            Scalar::from(6u64),
        ]);

        let mut ivc = start_ivc(shape, &instance1, &witness1).unwrap();
        assert_eq!(ivc.step, 1);

        // Extend with second instance: 4 * 5 = 20
        let instance2 = R1CSInstance::new(vec![]);
        let witness2 = R1CSWitness::new(vec![
            Scalar::from(4u64),
            Scalar::from(5u64),
            Scalar::from(20u64),
        ]);

        extend_ivc(&mut ivc, &instance2, &witness2).unwrap();
        assert_eq!(ivc.step, 2);
        assert!(verify_ivc(&ivc).unwrap());
    }

    #[test]
    fn test_ivc_multiple_extends() {
        let shape = create_mul_shape();

        let instances = [
            (2u64, 3u64, 6u64),
            (4u64, 5u64, 20u64),
            (7u64, 11u64, 77u64),
            (13u64, 17u64, 221u64),
        ];

        let instance0 = R1CSInstance::new(vec![]);
        let witness0 = R1CSWitness::new(vec![
            Scalar::from(instances[0].0),
            Scalar::from(instances[0].1),
            Scalar::from(instances[0].2),
        ]);

        let mut ivc = start_ivc(shape, &instance0, &witness0).unwrap();

        for (i, (x, y, z)) in instances.iter().enumerate().skip(1) {
            let instance = R1CSInstance::new(vec![]);
            let witness =
                R1CSWitness::new(vec![Scalar::from(*x), Scalar::from(*y), Scalar::from(*z)]);

            extend_ivc(&mut ivc, &instance, &witness).unwrap();
            assert_eq!(ivc.step, (i + 1) as u64);
            assert!(verify_ivc(&ivc).unwrap(), "Failed at step {}", i + 1);
        }

        // Final verification
        assert_eq!(ivc.step, 4);
        assert!(verify_ivc(&ivc).unwrap());
    }

    #[test]
    fn test_ivc_reject_invalid_initial() {
        let shape = create_mul_shape();
        let instance = R1CSInstance::new(vec![]);
        // Invalid: 2 * 3 = 7 (wrong!)
        let witness = R1CSWitness::new(vec![
            Scalar::from(2u64),
            Scalar::from(3u64),
            Scalar::from(7u64),
        ]);

        let result = start_ivc(shape, &instance, &witness);
        assert!(result.is_err());
    }

    #[test]
    fn test_ivc_reject_invalid_extend() {
        let shape = create_mul_shape();
        let instance1 = R1CSInstance::new(vec![]);
        let witness1 = R1CSWitness::new(vec![
            Scalar::from(2u64),
            Scalar::from(3u64),
            Scalar::from(6u64),
        ]);

        let mut ivc = start_ivc(shape, &instance1, &witness1).unwrap();

        // Try to extend with invalid instance: 4 * 5 = 21 (wrong!)
        let instance2 = R1CSInstance::new(vec![]);
        let witness2 = R1CSWitness::new(vec![
            Scalar::from(4u64),
            Scalar::from(5u64),
            Scalar::from(21u64),
        ]);

        let result = extend_ivc(&mut ivc, &instance2, &witness2);
        assert!(result.is_err());
    }

    #[test]
    fn test_compress_and_verify() {
        let shape = create_mul_shape();
        let instance = R1CSInstance::new(vec![]);
        let witness = R1CSWitness::new(vec![
            Scalar::from(2u64),
            Scalar::from(3u64),
            Scalar::from(6u64),
        ]);

        let ivc = start_ivc(shape.clone(), &instance, &witness).unwrap();
        let compressed = compress_ivc(&ivc);

        assert_eq!(compressed.step, 1);
        assert!(verify_compressed(&shape, &compressed).unwrap());
    }

    // Tests for certificate-based IVC operations
    mod cert_ivc_tests {
        use super::*;
        use lean5_kernel::{Expr, Level};

        fn empty_env() -> Environment {
            Environment::default()
        }

        #[test]
        fn test_start_ivc_from_sort_cert() {
            let cert = ProofCert::Sort { level: Level::Zero };
            let env = empty_env();

            let ivc = start_ivc_from_cert(&cert, &env).expect("should start IVC");

            assert_eq!(ivc.step, 1);
            assert!(verify_ivc(&ivc).expect("verification should succeed"));
        }

        #[test]
        fn test_start_ivc_from_bvar_cert() {
            let cert = ProofCert::BVar {
                idx: 0,
                expected_type: Box::new(Expr::Sort(Level::Zero)),
            };
            let env = empty_env();

            let ivc = start_ivc_from_cert(&cert, &env).expect("should start IVC");

            assert_eq!(ivc.step, 1);
            assert!(verify_ivc(&ivc).expect("verification should succeed"));
        }

        #[test]
        fn test_extend_ivc_same_shape() {
            // Create two certificates with identical structure
            let cert1 = ProofCert::Sort { level: Level::Zero };
            let cert2 = ProofCert::Sort { level: Level::Zero };
            let env = empty_env();

            let mut ivc = start_ivc_from_cert(&cert1, &env).expect("should start IVC");
            assert_eq!(ivc.step, 1);

            extend_ivc_with_cert(&mut ivc, &cert2, &env).expect("should extend IVC");
            assert_eq!(ivc.step, 2);
            assert!(verify_ivc(&ivc).expect("verification should succeed"));
        }

        #[test]
        fn test_extend_ivc_shape_mismatch_rejected() {
            // Create certificates with different structures
            let cert1 = ProofCert::Sort { level: Level::Zero };
            let cert2 = ProofCert::BVar {
                idx: 0,
                expected_type: Box::new(Expr::Sort(Level::Zero)),
            };
            let env = empty_env();

            let mut ivc = start_ivc_from_cert(&cert1, &env).expect("should start IVC");

            // Should fail because structures differ
            let result = extend_ivc_with_cert(&mut ivc, &cert2, &env);
            assert!(result.is_err(), "should reject mismatched shapes");
        }

        #[test]
        fn test_multiple_extends_same_shape() {
            // Fold multiple instances of the same certificate structure
            let env = empty_env();

            let cert = ProofCert::Sort { level: Level::Zero };

            let mut ivc = start_ivc_from_cert(&cert, &env).expect("should start IVC");

            for i in 1..5 {
                let next_cert = ProofCert::Sort { level: Level::Zero };
                extend_ivc_with_cert(&mut ivc, &next_cert, &env)
                    .unwrap_or_else(|e| panic!("extend {i} failed: {e}"));
                assert_eq!(ivc.step, (i + 1) as u64);
            }

            assert!(verify_ivc(&ivc).expect("final verification should succeed"));
            assert_eq!(ivc.step, 5);
        }

        #[test]
        fn test_ivc_with_app_cert() {
            let fn_cert = ProofCert::Sort { level: Level::Zero };
            let arg_cert = ProofCert::Sort { level: Level::Zero };

            let cert = ProofCert::App {
                fn_cert: Box::new(fn_cert),
                fn_type: Box::new(Expr::Sort(Level::Zero)),
                arg_cert: Box::new(arg_cert),
                result_type: Box::new(Expr::Sort(Level::Zero)),
            };
            let env = empty_env();

            let ivc = start_ivc_from_cert(&cert, &env).expect("should start IVC");

            assert_eq!(ivc.step, 1);
            assert!(verify_ivc(&ivc).expect("verification should succeed"));
        }

        #[test]
        fn test_compress_cert_ivc() {
            let cert = ProofCert::Sort { level: Level::Zero };
            let env = empty_env();

            let ivc = start_ivc_from_cert(&cert, &env).expect("should start IVC");
            let compressed = compress_ivc(&ivc);

            assert_eq!(compressed.step, 1);
            assert!(verify_compressed(&ivc.shape, &compressed).expect("compressed verification"));
        }
    }
}
