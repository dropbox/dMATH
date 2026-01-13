//! Certificate to Polynomial Encoding
//!
//! This module implements encoding of ProofCert trees into polynomials
//! suitable for polynomial commitment schemes.
//!
//! # Encoding Strategy
//!
//! ProofCert trees are flattened to a sequence of field elements:
//! 1. Each node has a tag identifying its variant
//! 2. Node contents are recursively encoded
//! 3. The sequence is interpreted as polynomial evaluations
//! 4. Polynomial is recovered via inverse FFT

use ark_bls12_381::Fr;
use ark_ff::PrimeField;
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain,
};

use lean5_kernel::cert::ZFCSetCertKind;
use lean5_kernel::name::NameInner;
use lean5_kernel::{Expr, Level, Name, ProofCert};

use crate::error::CommitError;

/// Node tag constants for encoding
mod tags {
    // Core certificates (all modes)
    pub const SORT: u64 = 0;
    pub const BVAR: u64 = 1;
    pub const FVAR: u64 = 2;
    pub const CONST: u64 = 3;
    pub const APP: u64 = 4;
    pub const LAM: u64 = 5;
    pub const PI: u64 = 6;
    pub const LET: u64 = 7;
    pub const LIT: u64 = 8;
    pub const DEF_EQ: u64 = 9;
    pub const MDATA: u64 = 10;
    pub const PROJ: u64 = 11;

    // Cubical mode certificates
    pub const CUBICAL_INTERVAL: u64 = 12;
    pub const CUBICAL_ENDPOINT: u64 = 13;
    pub const CUBICAL_PATH: u64 = 14;
    pub const CUBICAL_PATH_LAM: u64 = 15;
    pub const CUBICAL_PATH_APP: u64 = 16;
    pub const CUBICAL_HCOMP: u64 = 17;
    pub const CUBICAL_TRANSP: u64 = 18;

    // Classical mode certificates
    pub const CLASSICAL_CHOICE: u64 = 19;
    pub const CLASSICAL_EPSILON: u64 = 20;

    // ZFC/Set-theoretic mode certificates
    pub const ZFC_SET: u64 = 21;
    pub const ZFC_MEM: u64 = 22;
    pub const ZFC_COMPREHENSION: u64 = 23;

    // Impredicative mode certificates
    pub const SPROP: u64 = 24;
    pub const SQUASH: u64 = 25;

    // ZFC set construction sub-tags
    pub mod zfc {
        pub const EMPTY: u64 = 0;
        pub const SINGLETON: u64 = 1;
        pub const PAIR: u64 = 2;
        pub const UNION: u64 = 3;
        pub const POWERSET: u64 = 4;
        pub const SEPARATION: u64 = 5;
        pub const REPLACEMENT: u64 = 6;
        pub const INFINITY: u64 = 7;
        pub const CHOICE: u64 = 8;
    }
}

/// Encoded certificate as polynomial and metadata
#[derive(Clone, Debug)]
pub struct EncodedCert {
    /// Polynomial representation
    pub poly: DensePolynomial<Fr>,
    /// Number of field elements in encoding
    pub element_count: usize,
    /// Domain size (power of 2)
    pub domain_size: usize,
}

/// Encode a ProofCert as a polynomial over Fr
///
/// The certificate is flattened to field elements, then interpreted as
/// polynomial evaluations over a suitable domain.
pub fn encode_cert(cert: &ProofCert) -> Result<EncodedCert, CommitError> {
    // Flatten certificate to field elements
    let mut elements = Vec::new();
    flatten_cert(cert, &mut elements)?;

    let element_count = elements.len();

    // Find suitable domain size (power of 2)
    let domain_size = element_count.next_power_of_two();

    // Pad to domain size
    elements.resize(domain_size, Fr::from(0u64));

    // Create evaluation domain
    let domain = GeneralEvaluationDomain::<Fr>::new(domain_size).ok_or_else(|| {
        CommitError::InvalidDegree(format!("Cannot create domain of size {domain_size}"))
    })?;

    // Interpolate polynomial (inverse FFT)
    let coeffs = domain.ifft(&elements);
    let poly = DensePolynomial::from_coefficients_vec(coeffs);

    Ok(EncodedCert {
        poly,
        element_count,
        domain_size,
    })
}

/// Decode a polynomial back to field elements (for verification)
pub fn decode_cert(encoded: &EncodedCert) -> Vec<Fr> {
    let domain = GeneralEvaluationDomain::<Fr>::new(encoded.domain_size)
        .expect("Domain was valid during encoding");

    domain.fft(&encoded.poly.coeffs)
}

/// Flatten a ProofCert into a sequence of field elements
fn flatten_cert(cert: &ProofCert, out: &mut Vec<Fr>) -> Result<(), CommitError> {
    match cert {
        ProofCert::Sort { level } => {
            out.push(Fr::from(tags::SORT));
            flatten_level(level, out);
        }

        ProofCert::BVar { idx, expected_type } => {
            out.push(Fr::from(tags::BVAR));
            out.push(Fr::from(u64::from(*idx)));
            flatten_expr(expected_type, out)?;
        }

        ProofCert::FVar { id, type_ } => {
            out.push(Fr::from(tags::FVAR));
            // Encode FVarId as its raw value
            out.push(Fr::from(id.0));
            flatten_expr(type_, out)?;
        }

        ProofCert::Const {
            name,
            levels,
            type_,
        } => {
            out.push(Fr::from(tags::CONST));
            flatten_name(name, out);
            out.push(Fr::from(levels.len() as u64));
            for level in levels {
                flatten_level(level, out);
            }
            flatten_expr(type_, out)?;
        }

        ProofCert::App {
            fn_cert,
            fn_type,
            arg_cert,
            result_type,
        } => {
            out.push(Fr::from(tags::APP));
            flatten_cert(fn_cert, out)?;
            flatten_expr(fn_type, out)?;
            flatten_cert(arg_cert, out)?;
            flatten_expr(result_type, out)?;
        }

        ProofCert::Lam {
            binder_info,
            arg_type_cert,
            body_cert,
            result_type,
        } => {
            out.push(Fr::from(tags::LAM));
            out.push(Fr::from(*binder_info as u64));
            flatten_cert(arg_type_cert, out)?;
            flatten_cert(body_cert, out)?;
            flatten_expr(result_type, out)?;
        }

        ProofCert::Pi {
            binder_info,
            arg_type_cert,
            arg_level,
            body_type_cert,
            body_level,
        } => {
            out.push(Fr::from(tags::PI));
            out.push(Fr::from(*binder_info as u64));
            flatten_cert(arg_type_cert, out)?;
            flatten_level(arg_level, out);
            flatten_cert(body_type_cert, out)?;
            flatten_level(body_level, out);
        }

        ProofCert::Let {
            type_cert,
            value_cert,
            body_cert,
            result_type,
        } => {
            out.push(Fr::from(tags::LET));
            flatten_cert(type_cert, out)?;
            flatten_cert(value_cert, out)?;
            flatten_cert(body_cert, out)?;
            flatten_expr(result_type, out)?;
        }

        ProofCert::Lit { lit, type_ } => {
            out.push(Fr::from(tags::LIT));
            // Encode literal value
            match lit {
                lean5_kernel::Literal::Nat(n) => {
                    out.push(Fr::from(0u64)); // Nat tag
                                              // Encode arbitrary precision nat
                    let bytes = n.to_le_bytes();
                    out.push(Fr::from(bytes.len() as u64));
                    for chunk in bytes.chunks(31) {
                        // 31 bytes fit in Fr
                        let mut arr = [0u8; 32];
                        arr[..chunk.len()].copy_from_slice(chunk);
                        out.push(Fr::from_le_bytes_mod_order(&arr));
                    }
                }
                lean5_kernel::Literal::String(s) => {
                    out.push(Fr::from(1u64)); // String tag
                    let bytes = s.as_bytes();
                    out.push(Fr::from(bytes.len() as u64));
                    for chunk in bytes.chunks(31) {
                        let mut arr = [0u8; 32];
                        arr[..chunk.len()].copy_from_slice(chunk);
                        out.push(Fr::from_le_bytes_mod_order(&arr));
                    }
                }
            }
            flatten_expr(type_, out)?;
        }

        ProofCert::DefEq {
            inner,
            expected_type,
            actual_type,
            eq_steps,
        } => {
            out.push(Fr::from(tags::DEF_EQ));
            flatten_cert(inner, out)?;
            flatten_expr(expected_type, out)?;
            flatten_expr(actual_type, out)?;
            // Encode eq_steps length (actual steps encoding is complex, simplify for now)
            out.push(Fr::from(eq_steps.len() as u64));
        }

        ProofCert::MData {
            metadata: _,
            inner_cert,
            result_type,
        } => {
            out.push(Fr::from(tags::MDATA));
            // Skip metadata encoding for now (can be extended)
            flatten_cert(inner_cert, out)?;
            flatten_expr(result_type, out)?;
        }

        ProofCert::Proj {
            struct_name,
            idx,
            expr_cert,
            expr_type,
            field_type,
        } => {
            out.push(Fr::from(tags::PROJ));
            flatten_name(struct_name, out);
            out.push(Fr::from(u64::from(*idx)));
            flatten_cert(expr_cert, out)?;
            flatten_expr(expr_type, out)?;
            flatten_expr(field_type, out)?;
        }

        // ════════════════════════════════════════════════════════════════════════
        // Cubical mode certificates
        // ════════════════════════════════════════════════════════════════════════

        ProofCert::CubicalInterval => {
            out.push(Fr::from(tags::CUBICAL_INTERVAL));
        }

        ProofCert::CubicalEndpoint { is_one } => {
            out.push(Fr::from(tags::CUBICAL_ENDPOINT));
            out.push(Fr::from(if *is_one { 1u64 } else { 0u64 }));
        }

        ProofCert::CubicalPath {
            ty_cert,
            ty_level,
            left_cert,
            right_cert,
        } => {
            out.push(Fr::from(tags::CUBICAL_PATH));
            flatten_cert(ty_cert, out)?;
            flatten_level(ty_level, out);
            flatten_cert(left_cert, out)?;
            flatten_cert(right_cert, out)?;
        }

        ProofCert::CubicalPathLam {
            body_cert,
            body_type,
            result_type,
        } => {
            out.push(Fr::from(tags::CUBICAL_PATH_LAM));
            flatten_cert(body_cert, out)?;
            flatten_expr(body_type, out)?;
            flatten_expr(result_type, out)?;
        }

        ProofCert::CubicalPathApp {
            path_cert,
            arg_cert,
            path_type,
            result_type,
        } => {
            out.push(Fr::from(tags::CUBICAL_PATH_APP));
            flatten_cert(path_cert, out)?;
            flatten_cert(arg_cert, out)?;
            flatten_expr(path_type, out)?;
            flatten_expr(result_type, out)?;
        }

        ProofCert::CubicalHComp {
            ty_cert,
            phi_cert,
            u_cert,
            base_cert,
            result_type,
        } => {
            out.push(Fr::from(tags::CUBICAL_HCOMP));
            flatten_cert(ty_cert, out)?;
            flatten_cert(phi_cert, out)?;
            flatten_cert(u_cert, out)?;
            flatten_cert(base_cert, out)?;
            flatten_expr(result_type, out)?;
        }

        ProofCert::CubicalTransp {
            ty_cert,
            phi_cert,
            base_cert,
            result_type,
        } => {
            out.push(Fr::from(tags::CUBICAL_TRANSP));
            flatten_cert(ty_cert, out)?;
            flatten_cert(phi_cert, out)?;
            flatten_cert(base_cert, out)?;
            flatten_expr(result_type, out)?;
        }

        // ════════════════════════════════════════════════════════════════════════
        // Classical mode certificates
        // ════════════════════════════════════════════════════════════════════════

        ProofCert::ClassicalChoice {
            ty_cert,
            pred_cert,
            proof_cert,
            result_type,
        } => {
            out.push(Fr::from(tags::CLASSICAL_CHOICE));
            flatten_cert(ty_cert, out)?;
            flatten_cert(pred_cert, out)?;
            flatten_cert(proof_cert, out)?;
            flatten_expr(result_type, out)?;
        }

        ProofCert::ClassicalEpsilon {
            ty_cert,
            pred_cert,
            result_type,
        } => {
            out.push(Fr::from(tags::CLASSICAL_EPSILON));
            flatten_cert(ty_cert, out)?;
            flatten_cert(pred_cert, out)?;
            flatten_expr(result_type, out)?;
        }

        // ════════════════════════════════════════════════════════════════════════
        // ZFC/Set-theoretic mode certificates
        // ════════════════════════════════════════════════════════════════════════

        ProofCert::ZFCSet { kind, result_type } => {
            out.push(Fr::from(tags::ZFC_SET));
            flatten_zfc_set_kind(kind, out)?;
            flatten_expr(result_type, out)?;
        }

        ProofCert::ZFCMem {
            elem_cert,
            set_cert,
        } => {
            out.push(Fr::from(tags::ZFC_MEM));
            flatten_cert(elem_cert, out)?;
            flatten_cert(set_cert, out)?;
        }

        ProofCert::ZFCComprehension {
            var_ty_cert,
            pred_cert,
            result_type,
        } => {
            out.push(Fr::from(tags::ZFC_COMPREHENSION));
            flatten_cert(var_ty_cert, out)?;
            flatten_cert(pred_cert, out)?;
            flatten_expr(result_type, out)?;
        }

        // ════════════════════════════════════════════════════════════════════════
        // Impredicative mode certificates
        // ════════════════════════════════════════════════════════════════════════

        ProofCert::SProp => {
            out.push(Fr::from(tags::SPROP));
        }

        ProofCert::Squash { inner_cert } => {
            out.push(Fr::from(tags::SQUASH));
            flatten_cert(inner_cert, out)?;
        }
    }

    Ok(())
}

/// Flatten a ZFCSetCertKind to field elements
fn flatten_zfc_set_kind(kind: &ZFCSetCertKind, out: &mut Vec<Fr>) -> Result<(), CommitError> {
    match kind {
        ZFCSetCertKind::Empty => {
            out.push(Fr::from(tags::zfc::EMPTY));
        }
        ZFCSetCertKind::Singleton(cert) => {
            out.push(Fr::from(tags::zfc::SINGLETON));
            flatten_cert(cert, out)?;
        }
        ZFCSetCertKind::Pair(cert1, cert2) => {
            out.push(Fr::from(tags::zfc::PAIR));
            flatten_cert(cert1, out)?;
            flatten_cert(cert2, out)?;
        }
        ZFCSetCertKind::Union(cert) => {
            out.push(Fr::from(tags::zfc::UNION));
            flatten_cert(cert, out)?;
        }
        ZFCSetCertKind::PowerSet(cert) => {
            out.push(Fr::from(tags::zfc::POWERSET));
            flatten_cert(cert, out)?;
        }
        ZFCSetCertKind::Separation { set_cert, pred_cert } => {
            out.push(Fr::from(tags::zfc::SEPARATION));
            flatten_cert(set_cert, out)?;
            flatten_cert(pred_cert, out)?;
        }
        ZFCSetCertKind::Replacement { set_cert, func_cert } => {
            out.push(Fr::from(tags::zfc::REPLACEMENT));
            flatten_cert(set_cert, out)?;
            flatten_cert(func_cert, out)?;
        }
        ZFCSetCertKind::Infinity => {
            out.push(Fr::from(tags::zfc::INFINITY));
        }
        ZFCSetCertKind::Choice(cert) => {
            out.push(Fr::from(tags::zfc::CHOICE));
            flatten_cert(cert, out)?;
        }
    }
    Ok(())
}

/// Flatten a Level to field elements
fn flatten_level(level: &Level, out: &mut Vec<Fr>) {
    match level {
        Level::Zero => {
            out.push(Fr::from(0u64)); // Zero tag
        }
        Level::Succ(inner) => {
            out.push(Fr::from(1u64)); // Succ tag
            flatten_level(inner, out);
        }
        Level::Max(l1, l2) => {
            out.push(Fr::from(2u64)); // Max tag
            flatten_level(l1, out);
            flatten_level(l2, out);
        }
        Level::IMax(l1, l2) => {
            out.push(Fr::from(3u64)); // IMax tag
            flatten_level(l1, out);
            flatten_level(l2, out);
        }
        Level::Param(name) => {
            out.push(Fr::from(4u64)); // Param tag
            flatten_name(name, out);
        }
    }
}

/// Flatten a Name to field elements
fn flatten_name(name: &Name, out: &mut Vec<Fr>) {
    match name.inner() {
        NameInner::Anon => {
            out.push(Fr::from(0u64)); // Anon tag
        }
        NameInner::Str(parent, s) => {
            out.push(Fr::from(1u64)); // Str tag
            flatten_name(parent, out);
            // Hash string to field element
            let hash = hash_string(s);
            out.push(hash);
        }
        NameInner::Num(parent, n) => {
            out.push(Fr::from(2u64)); // Num tag
            flatten_name(parent, out);
            out.push(Fr::from(*n));
        }
    }
}

/// Flatten an Expr to field elements (simplified encoding)
fn flatten_expr(expr: &Expr, out: &mut Vec<Fr>) -> Result<(), CommitError> {
    // For now, use a simplified hash-based encoding of expressions
    // Full encoding would mirror the certificate encoding structure
    let hash = hash_expr(expr);
    out.push(hash);
    Ok(())
}

/// Hash a string to a field element
fn hash_string(s: &str) -> Fr {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    Fr::from(hasher.finish())
}

/// Hash an expression to a field element (simplified)
fn hash_expr(expr: &Expr) -> Fr {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    // Use debug representation for hashing (not ideal but functional)
    format!("{expr:?}").hash(&mut hasher);
    Fr::from(hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_poly::Polynomial;
    use std::sync::Arc;

    #[test]
    fn test_encode_sort() {
        let cert = ProofCert::Sort { level: Level::Zero };

        let encoded = encode_cert(&cert).expect("encoding should succeed");
        assert!(encoded.poly.degree() < encoded.domain_size);
        assert!(encoded.element_count > 0);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let cert = ProofCert::Sort {
            level: Level::Succ(Arc::new(Level::Zero)),
        };

        let encoded = encode_cert(&cert).expect("encoding should succeed");
        let decoded = decode_cert(&encoded);

        // First elements should match
        assert_eq!(decoded[0], Fr::from(tags::SORT));
    }
}
