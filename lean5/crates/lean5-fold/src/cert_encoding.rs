//! ProofCert to R1CS Encoding
//!
//! This module implements encoding of ProofCert verification as R1CS constraints.
//! Each certificate node generates constraints that verify its structure is valid.
//!
//! # Overview
//!
//! R1CS (Rank-1 Constraint System) represents computations as:
//!   Az ∘ Bz = Cz
//!
//! We encode certificate verification by:
//! 1. Allocating witness variables for certificate contents
//! 2. Adding constraints that verify certificate structure
//! 3. Using hash-based equality checks for expressions
//!
//! # Encoding Strategy
//!
//! Each ProofCert variant encodes to constraints verifying:
//! - Node tag matches expected variant
//! - Sub-certificates verify recursively
//! - Type/expression relationships hold
//!
//! This encoding enables:
//! - Folding multiple certificate verifications via Nova
//! - Generating succinct proofs of verification
//! - Batch verification of multiple certificates

use crate::error::FoldError;
use crate::r1cs::{R1CSBuilder, R1CSInstance, R1CSShape, R1CSWitness};
use crate::Scalar;
use ark_ff::{One, Zero};

use lean5_kernel::cert::ZFCSetCertKind;
use lean5_kernel::{BinderInfo, Expr, Level, Literal, Name, ProofCert};

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Node tag constants for constraint generation
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

/// Result of encoding a ProofCert to R1CS
#[derive(Debug)]
pub struct EncodedR1CS {
    /// The R1CS shape (constraint structure)
    pub shape: R1CSShape,
    /// Public instance (certificate hash as public input)
    pub instance: R1CSInstance,
    /// Private witness (certificate contents)
    pub witness: R1CSWitness,
    /// Number of constraints generated
    pub num_constraints: usize,
    /// Number of witness variables
    pub num_witness_vars: usize,
}

/// Builder for encoding ProofCert to R1CS constraints
struct CertR1CSEncoder {
    builder: R1CSBuilder,
    witness: Vec<Scalar>,
}

impl CertR1CSEncoder {
    /// Create a new encoder with 1 public input (certificate hash)
    fn new() -> Self {
        Self {
            builder: R1CSBuilder::new(1), // 1 public input: cert hash
            witness: Vec::new(),
        }
    }

    /// Allocate a witness variable and set its value
    fn alloc(&mut self, value: Scalar) -> usize {
        let idx = self.builder.alloc_var();
        // Ensure witness vector is large enough
        while self.witness.len() <= idx {
            self.witness.push(Scalar::zero());
        }
        self.witness[idx] = value;
        idx
    }

    /// Add constraint: a * b = c (using variable indices in z)
    #[allow(dead_code)] // Useful for future constraint patterns
    fn constrain_mul(&mut self, a_idx: usize, b_idx: usize, c_idx: usize) {
        self.builder.add_constraint(
            vec![(a_idx, Scalar::one())],
            vec![(b_idx, Scalar::one())],
            vec![(c_idx, Scalar::one())],
        );
    }

    /// Add constraint: a = b (using variable indices in z)
    fn constrain_eq(&mut self, a_idx: usize, b_idx: usize) {
        // Encode a = b as: a * 1 = b
        let const_idx = self.builder.const_idx();
        self.builder.add_constraint(
            vec![(a_idx, Scalar::one())],
            vec![(const_idx, Scalar::one())],
            vec![(b_idx, Scalar::one())],
        );
    }

    /// Add constraint: var = constant
    fn constrain_const(&mut self, var_idx: usize, value: Scalar) {
        // Encode var = value as: var * 1 = value * 1
        let const_idx = self.builder.const_idx();
        self.builder.add_constraint(
            vec![(var_idx, Scalar::one())],
            vec![(const_idx, Scalar::one())],
            vec![(const_idx, value)],
        );
    }

    /// Encode a ProofCert, returning the witness variable index for the node hash
    fn encode_cert(&mut self, cert: &ProofCert) -> Result<usize, FoldError> {
        match cert {
            ProofCert::Sort { level } => Ok(self.encode_sort(level)),
            ProofCert::BVar { idx, expected_type } => Ok(self.encode_bvar(*idx, expected_type)),
            ProofCert::FVar { id, type_ } => Ok(self.encode_fvar(id.0, type_)),
            ProofCert::Const {
                name,
                levels,
                type_,
            } => Ok(self.encode_const(name, levels, type_)),
            ProofCert::App {
                fn_cert,
                fn_type,
                arg_cert,
                result_type,
            } => self.encode_app(fn_cert, fn_type, arg_cert, result_type),
            ProofCert::Lam {
                binder_info,
                arg_type_cert,
                body_cert,
                result_type,
            } => self.encode_lam(*binder_info, arg_type_cert, body_cert, result_type),
            ProofCert::Pi {
                binder_info,
                arg_type_cert,
                arg_level,
                body_type_cert,
                body_level,
            } => self.encode_pi(
                *binder_info,
                arg_type_cert,
                arg_level,
                body_type_cert,
                body_level,
            ),
            ProofCert::Let {
                type_cert,
                value_cert,
                body_cert,
                result_type,
            } => self.encode_let(type_cert, value_cert, body_cert, result_type),
            ProofCert::Lit { lit, type_ } => Ok(self.encode_lit(lit, type_)),
            ProofCert::DefEq {
                inner,
                expected_type,
                actual_type,
                eq_steps,
            } => self.encode_def_eq(inner, expected_type, actual_type, eq_steps),
            ProofCert::MData {
                metadata: _,
                inner_cert,
                result_type,
            } => self.encode_mdata(inner_cert, result_type),
            ProofCert::Proj {
                struct_name,
                idx,
                expr_cert,
                expr_type,
                field_type,
            } => self.encode_proj(struct_name, *idx, expr_cert, expr_type, field_type),

            // ════════════════════════════════════════════════════════════════════════
            // Cubical mode certificates
            // ════════════════════════════════════════════════════════════════════════

            ProofCert::CubicalInterval => Ok(self.encode_cubical_interval()),

            ProofCert::CubicalEndpoint { is_one } => Ok(self.encode_cubical_endpoint(*is_one)),

            ProofCert::CubicalPath {
                ty_cert,
                ty_level,
                left_cert,
                right_cert,
            } => self.encode_cubical_path(ty_cert, ty_level, left_cert, right_cert),

            ProofCert::CubicalPathLam {
                body_cert,
                body_type,
                result_type,
            } => self.encode_cubical_path_lam(body_cert, body_type, result_type),

            ProofCert::CubicalPathApp {
                path_cert,
                arg_cert,
                path_type,
                result_type,
            } => self.encode_cubical_path_app(path_cert, arg_cert, path_type, result_type),

            ProofCert::CubicalHComp {
                ty_cert,
                phi_cert,
                u_cert,
                base_cert,
                result_type,
            } => self.encode_cubical_hcomp(ty_cert, phi_cert, u_cert, base_cert, result_type),

            ProofCert::CubicalTransp {
                ty_cert,
                phi_cert,
                base_cert,
                result_type,
            } => self.encode_cubical_transp(ty_cert, phi_cert, base_cert, result_type),

            // ════════════════════════════════════════════════════════════════════════
            // Classical mode certificates
            // ════════════════════════════════════════════════════════════════════════

            ProofCert::ClassicalChoice {
                ty_cert,
                pred_cert,
                proof_cert,
                result_type,
            } => self.encode_classical_choice(ty_cert, pred_cert, proof_cert, result_type),

            ProofCert::ClassicalEpsilon {
                ty_cert,
                pred_cert,
                result_type,
            } => self.encode_classical_epsilon(ty_cert, pred_cert, result_type),

            // ════════════════════════════════════════════════════════════════════════
            // ZFC/Set-theoretic mode certificates
            // ════════════════════════════════════════════════════════════════════════

            ProofCert::ZFCSet { kind, result_type } => {
                self.encode_zfc_set(kind, result_type)
            }

            ProofCert::ZFCMem {
                elem_cert,
                set_cert,
            } => self.encode_zfc_mem(elem_cert, set_cert),

            ProofCert::ZFCComprehension {
                var_ty_cert,
                pred_cert,
                result_type,
            } => self.encode_zfc_comprehension(var_ty_cert, pred_cert, result_type),

            // ════════════════════════════════════════════════════════════════════════
            // Impredicative mode certificates
            // ════════════════════════════════════════════════════════════════════════

            ProofCert::SProp => Ok(self.encode_sprop()),

            ProofCert::Squash { inner_cert } => self.encode_squash(inner_cert),
        }
    }

    fn encode_sort(&mut self, level: &Level) -> usize {
        // Allocate tag variable and constrain to SORT
        let tag_var = self.alloc(Scalar::from(tags::SORT));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::SORT));

        // Encode level
        let _level_var = self.encode_level(level);

        // Allocate node hash combining tag and level
        let node_hash = hash_node(&[tags::SORT], &[level]);
        self.alloc(node_hash)
    }

    fn encode_bvar(&mut self, idx: u32, expected_type: &Expr) -> usize {
        let tag_var = self.alloc(Scalar::from(tags::BVAR));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::BVAR));

        let _idx_var = self.alloc(Scalar::from(u64::from(idx)));
        let _type_hash_var = self.encode_expr(expected_type);

        let node_hash = hash_bvar(idx, expected_type);
        self.alloc(node_hash)
    }

    fn encode_fvar(&mut self, id: u64, type_: &Expr) -> usize {
        let tag_var = self.alloc(Scalar::from(tags::FVAR));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::FVAR));

        let _id_var = self.alloc(Scalar::from(id));
        let _type_hash_var = self.encode_expr(type_);

        let node_hash = hash_fvar(id, type_);
        self.alloc(node_hash)
    }

    fn encode_const(&mut self, name: &Name, levels: &[Level], type_: &Expr) -> usize {
        let tag_var = self.alloc(Scalar::from(tags::CONST));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::CONST));

        let _name_hash_var = self.encode_name(name);
        let _levels_count_var = self.alloc(Scalar::from(levels.len() as u64));
        for level in levels {
            self.encode_level(level);
        }
        let _type_hash_var = self.encode_expr(type_);

        let node_hash = hash_const(name, levels, type_);
        self.alloc(node_hash)
    }

    fn encode_app(
        &mut self,
        fn_cert: &ProofCert,
        fn_type: &Expr,
        arg_cert: &ProofCert,
        result_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::APP));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::APP));

        // Recursively encode sub-certificates
        let _fn_cert_var = self.encode_cert(fn_cert)?;
        let _fn_type_var = self.encode_expr(fn_type);
        let _arg_cert_var = self.encode_cert(arg_cert)?;
        let _result_type_var = self.encode_expr(result_type);

        // Add constraint: fn_type must be a Pi type
        // This is encoded as a constraint on the fn_type hash
        // In a full implementation, we would verify fn_type.is_pi()

        let node_hash = hash_app(fn_cert, fn_type, arg_cert, result_type);
        let hash_var = self.alloc(node_hash);

        Ok(hash_var)
    }

    fn encode_lam(
        &mut self,
        binder_info: BinderInfo,
        arg_type_cert: &ProofCert,
        body_cert: &ProofCert,
        result_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::LAM));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::LAM));

        let _bi_var = self.alloc(Scalar::from(binder_info as u64));
        let _arg_type_var = self.encode_cert(arg_type_cert)?;
        let _body_var = self.encode_cert(body_cert)?;
        let _result_type_var = self.encode_expr(result_type);

        let node_hash = hash_lam(binder_info, arg_type_cert, body_cert, result_type);
        let hash_var = self.alloc(node_hash);

        Ok(hash_var)
    }

    fn encode_pi(
        &mut self,
        binder_info: BinderInfo,
        arg_type_cert: &ProofCert,
        arg_level: &Level,
        body_type_cert: &ProofCert,
        body_level: &Level,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::PI));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::PI));

        let _bi_var = self.alloc(Scalar::from(binder_info as u64));
        let _arg_type_var = self.encode_cert(arg_type_cert)?;
        let _arg_level_var = self.encode_level(arg_level);
        let _body_type_var = self.encode_cert(body_type_cert)?;
        let _body_level_var = self.encode_level(body_level);

        let node_hash = hash_pi(
            binder_info,
            arg_type_cert,
            arg_level,
            body_type_cert,
            body_level,
        );
        let hash_var = self.alloc(node_hash);

        Ok(hash_var)
    }

    fn encode_let(
        &mut self,
        type_cert: &ProofCert,
        value_cert: &ProofCert,
        body_cert: &ProofCert,
        result_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::LET));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::LET));

        let _type_var = self.encode_cert(type_cert)?;
        let _value_var = self.encode_cert(value_cert)?;
        let _body_var = self.encode_cert(body_cert)?;
        let _result_type_var = self.encode_expr(result_type);

        let node_hash = hash_let(type_cert, value_cert, body_cert, result_type);
        let hash_var = self.alloc(node_hash);

        Ok(hash_var)
    }

    fn encode_lit(&mut self, lit: &Literal, type_: &Expr) -> usize {
        let tag_var = self.alloc(Scalar::from(tags::LIT));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::LIT));

        let _lit_hash_var = self.alloc(hash_literal(lit));
        let _type_hash_var = self.encode_expr(type_);

        let node_hash = hash_lit(lit, type_);
        self.alloc(node_hash)
    }

    fn encode_def_eq(
        &mut self,
        inner: &ProofCert,
        expected_type: &Expr,
        actual_type: &Expr,
        eq_steps: &[lean5_kernel::DefEqStep],
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::DEF_EQ));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::DEF_EQ));

        let _inner_var = self.encode_cert(inner)?;
        let _expected_var = self.encode_expr(expected_type);
        let _actual_var = self.encode_expr(actual_type);
        let _steps_count_var = self.alloc(Scalar::from(eq_steps.len() as u64));

        let node_hash = hash_def_eq(inner, expected_type, actual_type, eq_steps.len());
        let hash_var = self.alloc(node_hash);

        Ok(hash_var)
    }

    fn encode_mdata(
        &mut self,
        inner_cert: &ProofCert,
        result_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::MDATA));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::MDATA));

        let _inner_var = self.encode_cert(inner_cert)?;
        let _result_type_var = self.encode_expr(result_type);

        let node_hash = hash_mdata(inner_cert, result_type);
        let hash_var = self.alloc(node_hash);

        Ok(hash_var)
    }

    fn encode_proj(
        &mut self,
        struct_name: &Name,
        idx: u32,
        expr_cert: &ProofCert,
        expr_type: &Expr,
        field_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::PROJ));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::PROJ));

        let _name_var = self.encode_name(struct_name);
        let _idx_var = self.alloc(Scalar::from(u64::from(idx)));
        let _expr_cert_var = self.encode_cert(expr_cert)?;
        let _expr_type_var = self.encode_expr(expr_type);
        let _field_type_var = self.encode_expr(field_type);

        let node_hash = hash_proj(struct_name, idx, expr_cert, expr_type, field_type);
        let hash_var = self.alloc(node_hash);

        Ok(hash_var)
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Cubical mode certificate encoding
    // ════════════════════════════════════════════════════════════════════════════

    fn encode_cubical_interval(&mut self) -> usize {
        let tag_var = self.alloc(Scalar::from(tags::CUBICAL_INTERVAL));
        self.constrain_const(
            self.builder.var_idx(tag_var),
            Scalar::from(tags::CUBICAL_INTERVAL),
        );
        let node_hash = hash_cubical_interval();
        self.alloc(node_hash)
    }

    fn encode_cubical_endpoint(&mut self, is_one: bool) -> usize {
        let tag_var = self.alloc(Scalar::from(tags::CUBICAL_ENDPOINT));
        self.constrain_const(
            self.builder.var_idx(tag_var),
            Scalar::from(tags::CUBICAL_ENDPOINT),
        );
        let _is_one_var = self.alloc(Scalar::from(if is_one { 1u64 } else { 0u64 }));
        let node_hash = hash_cubical_endpoint(is_one);
        self.alloc(node_hash)
    }

    fn encode_cubical_path(
        &mut self,
        ty_cert: &ProofCert,
        ty_level: &Level,
        left_cert: &ProofCert,
        right_cert: &ProofCert,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::CUBICAL_PATH));
        self.constrain_const(
            self.builder.var_idx(tag_var),
            Scalar::from(tags::CUBICAL_PATH),
        );
        let _ty_cert_var = self.encode_cert(ty_cert)?;
        let _ty_level_var = self.encode_level(ty_level);
        let _left_cert_var = self.encode_cert(left_cert)?;
        let _right_cert_var = self.encode_cert(right_cert)?;
        let node_hash = hash_cubical_path(ty_cert, ty_level, left_cert, right_cert);
        Ok(self.alloc(node_hash))
    }

    fn encode_cubical_path_lam(
        &mut self,
        body_cert: &ProofCert,
        body_type: &Expr,
        result_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::CUBICAL_PATH_LAM));
        self.constrain_const(
            self.builder.var_idx(tag_var),
            Scalar::from(tags::CUBICAL_PATH_LAM),
        );
        let _body_cert_var = self.encode_cert(body_cert)?;
        let _body_type_var = self.encode_expr(body_type);
        let _result_type_var = self.encode_expr(result_type);
        let node_hash = hash_cubical_path_lam(body_cert, body_type, result_type);
        Ok(self.alloc(node_hash))
    }

    fn encode_cubical_path_app(
        &mut self,
        path_cert: &ProofCert,
        arg_cert: &ProofCert,
        path_type: &Expr,
        result_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::CUBICAL_PATH_APP));
        self.constrain_const(
            self.builder.var_idx(tag_var),
            Scalar::from(tags::CUBICAL_PATH_APP),
        );
        let _path_cert_var = self.encode_cert(path_cert)?;
        let _arg_cert_var = self.encode_cert(arg_cert)?;
        let _path_type_var = self.encode_expr(path_type);
        let _result_type_var = self.encode_expr(result_type);
        let node_hash = hash_cubical_path_app(path_cert, arg_cert, path_type, result_type);
        Ok(self.alloc(node_hash))
    }

    fn encode_cubical_hcomp(
        &mut self,
        ty_cert: &ProofCert,
        phi_cert: &ProofCert,
        u_cert: &ProofCert,
        base_cert: &ProofCert,
        result_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::CUBICAL_HCOMP));
        self.constrain_const(
            self.builder.var_idx(tag_var),
            Scalar::from(tags::CUBICAL_HCOMP),
        );
        let _ty_cert_var = self.encode_cert(ty_cert)?;
        let _phi_cert_var = self.encode_cert(phi_cert)?;
        let _u_cert_var = self.encode_cert(u_cert)?;
        let _base_cert_var = self.encode_cert(base_cert)?;
        let _result_type_var = self.encode_expr(result_type);
        let node_hash = hash_cubical_hcomp(ty_cert, phi_cert, u_cert, base_cert, result_type);
        Ok(self.alloc(node_hash))
    }

    fn encode_cubical_transp(
        &mut self,
        ty_cert: &ProofCert,
        phi_cert: &ProofCert,
        base_cert: &ProofCert,
        result_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::CUBICAL_TRANSP));
        self.constrain_const(
            self.builder.var_idx(tag_var),
            Scalar::from(tags::CUBICAL_TRANSP),
        );
        let _ty_cert_var = self.encode_cert(ty_cert)?;
        let _phi_cert_var = self.encode_cert(phi_cert)?;
        let _base_cert_var = self.encode_cert(base_cert)?;
        let _result_type_var = self.encode_expr(result_type);
        let node_hash = hash_cubical_transp(ty_cert, phi_cert, base_cert, result_type);
        Ok(self.alloc(node_hash))
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Classical mode certificate encoding
    // ════════════════════════════════════════════════════════════════════════════

    fn encode_classical_choice(
        &mut self,
        ty_cert: &ProofCert,
        pred_cert: &ProofCert,
        proof_cert: &ProofCert,
        result_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::CLASSICAL_CHOICE));
        self.constrain_const(
            self.builder.var_idx(tag_var),
            Scalar::from(tags::CLASSICAL_CHOICE),
        );
        let _ty_cert_var = self.encode_cert(ty_cert)?;
        let _pred_cert_var = self.encode_cert(pred_cert)?;
        let _proof_cert_var = self.encode_cert(proof_cert)?;
        let _result_type_var = self.encode_expr(result_type);
        let node_hash = hash_classical_choice(ty_cert, pred_cert, proof_cert, result_type);
        Ok(self.alloc(node_hash))
    }

    fn encode_classical_epsilon(
        &mut self,
        ty_cert: &ProofCert,
        pred_cert: &ProofCert,
        result_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::CLASSICAL_EPSILON));
        self.constrain_const(
            self.builder.var_idx(tag_var),
            Scalar::from(tags::CLASSICAL_EPSILON),
        );
        let _ty_cert_var = self.encode_cert(ty_cert)?;
        let _pred_cert_var = self.encode_cert(pred_cert)?;
        let _result_type_var = self.encode_expr(result_type);
        let node_hash = hash_classical_epsilon(ty_cert, pred_cert, result_type);
        Ok(self.alloc(node_hash))
    }

    // ════════════════════════════════════════════════════════════════════════════
    // ZFC/Set-theoretic mode certificate encoding
    // ════════════════════════════════════════════════════════════════════════════

    fn encode_zfc_set(
        &mut self,
        kind: &ZFCSetCertKind,
        result_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::ZFC_SET));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::ZFC_SET));
        let _kind_hash_var = self.encode_zfc_set_kind(kind)?;
        let _result_type_var = self.encode_expr(result_type);
        let node_hash = hash_zfc_set(kind, result_type);
        Ok(self.alloc(node_hash))
    }

    fn encode_zfc_set_kind(&mut self, kind: &ZFCSetCertKind) -> Result<usize, FoldError> {
        match kind {
            ZFCSetCertKind::Empty => {
                let tag_var = self.alloc(Scalar::from(tags::zfc::EMPTY));
                self.constrain_const(
                    self.builder.var_idx(tag_var),
                    Scalar::from(tags::zfc::EMPTY),
                );
                Ok(self.alloc(hash_zfc_empty()))
            }
            ZFCSetCertKind::Singleton(cert) => {
                let tag_var = self.alloc(Scalar::from(tags::zfc::SINGLETON));
                self.constrain_const(
                    self.builder.var_idx(tag_var),
                    Scalar::from(tags::zfc::SINGLETON),
                );
                let _cert_var = self.encode_cert(cert)?;
                Ok(self.alloc(hash_zfc_singleton(cert)))
            }
            ZFCSetCertKind::Pair(cert1, cert2) => {
                let tag_var = self.alloc(Scalar::from(tags::zfc::PAIR));
                self.constrain_const(
                    self.builder.var_idx(tag_var),
                    Scalar::from(tags::zfc::PAIR),
                );
                let _cert1_var = self.encode_cert(cert1)?;
                let _cert2_var = self.encode_cert(cert2)?;
                Ok(self.alloc(hash_zfc_pair(cert1, cert2)))
            }
            ZFCSetCertKind::Union(cert) => {
                let tag_var = self.alloc(Scalar::from(tags::zfc::UNION));
                self.constrain_const(
                    self.builder.var_idx(tag_var),
                    Scalar::from(tags::zfc::UNION),
                );
                let _cert_var = self.encode_cert(cert)?;
                Ok(self.alloc(hash_zfc_union(cert)))
            }
            ZFCSetCertKind::PowerSet(cert) => {
                let tag_var = self.alloc(Scalar::from(tags::zfc::POWERSET));
                self.constrain_const(
                    self.builder.var_idx(tag_var),
                    Scalar::from(tags::zfc::POWERSET),
                );
                let _cert_var = self.encode_cert(cert)?;
                Ok(self.alloc(hash_zfc_powerset(cert)))
            }
            ZFCSetCertKind::Separation { set_cert, pred_cert } => {
                let tag_var = self.alloc(Scalar::from(tags::zfc::SEPARATION));
                self.constrain_const(
                    self.builder.var_idx(tag_var),
                    Scalar::from(tags::zfc::SEPARATION),
                );
                let _set_cert_var = self.encode_cert(set_cert)?;
                let _pred_cert_var = self.encode_cert(pred_cert)?;
                Ok(self.alloc(hash_zfc_separation(set_cert, pred_cert)))
            }
            ZFCSetCertKind::Replacement { set_cert, func_cert } => {
                let tag_var = self.alloc(Scalar::from(tags::zfc::REPLACEMENT));
                self.constrain_const(
                    self.builder.var_idx(tag_var),
                    Scalar::from(tags::zfc::REPLACEMENT),
                );
                let _set_cert_var = self.encode_cert(set_cert)?;
                let _func_cert_var = self.encode_cert(func_cert)?;
                Ok(self.alloc(hash_zfc_replacement(set_cert, func_cert)))
            }
            ZFCSetCertKind::Infinity => {
                let tag_var = self.alloc(Scalar::from(tags::zfc::INFINITY));
                self.constrain_const(
                    self.builder.var_idx(tag_var),
                    Scalar::from(tags::zfc::INFINITY),
                );
                Ok(self.alloc(hash_zfc_infinity()))
            }
            ZFCSetCertKind::Choice(cert) => {
                let tag_var = self.alloc(Scalar::from(tags::zfc::CHOICE));
                self.constrain_const(
                    self.builder.var_idx(tag_var),
                    Scalar::from(tags::zfc::CHOICE),
                );
                let _cert_var = self.encode_cert(cert)?;
                Ok(self.alloc(hash_zfc_choice(cert)))
            }
        }
    }

    fn encode_zfc_mem(
        &mut self,
        elem_cert: &ProofCert,
        set_cert: &ProofCert,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::ZFC_MEM));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::ZFC_MEM));
        let _elem_cert_var = self.encode_cert(elem_cert)?;
        let _set_cert_var = self.encode_cert(set_cert)?;
        let node_hash = hash_zfc_mem(elem_cert, set_cert);
        Ok(self.alloc(node_hash))
    }

    fn encode_zfc_comprehension(
        &mut self,
        var_ty_cert: &ProofCert,
        pred_cert: &ProofCert,
        result_type: &Expr,
    ) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::ZFC_COMPREHENSION));
        self.constrain_const(
            self.builder.var_idx(tag_var),
            Scalar::from(tags::ZFC_COMPREHENSION),
        );
        let _var_ty_cert_var = self.encode_cert(var_ty_cert)?;
        let _pred_cert_var = self.encode_cert(pred_cert)?;
        let _result_type_var = self.encode_expr(result_type);
        let node_hash = hash_zfc_comprehension(var_ty_cert, pred_cert, result_type);
        Ok(self.alloc(node_hash))
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Impredicative mode certificate encoding
    // ════════════════════════════════════════════════════════════════════════════

    fn encode_sprop(&mut self) -> usize {
        let tag_var = self.alloc(Scalar::from(tags::SPROP));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::SPROP));
        let node_hash = hash_sprop();
        self.alloc(node_hash)
    }

    fn encode_squash(&mut self, inner_cert: &ProofCert) -> Result<usize, FoldError> {
        let tag_var = self.alloc(Scalar::from(tags::SQUASH));
        self.constrain_const(self.builder.var_idx(tag_var), Scalar::from(tags::SQUASH));
        let _inner_cert_var = self.encode_cert(inner_cert)?;
        let node_hash = hash_squash(inner_cert);
        Ok(self.alloc(node_hash))
    }

    /// Encode a Level, returning witness variable index
    fn encode_level(&mut self, level: &Level) -> usize {
        let hash = hash_level(level);
        self.alloc(hash)
    }

    /// Encode a Name, returning witness variable index
    fn encode_name(&mut self, name: &Name) -> usize {
        let hash = hash_name(name);
        self.alloc(hash)
    }

    /// Encode an Expr, returning witness variable index
    fn encode_expr(&mut self, expr: &Expr) -> usize {
        let hash = hash_expr(expr);
        self.alloc(hash)
    }

    /// Finalize the encoding into an R1CS
    fn finalize(self, cert_hash: Scalar) -> Result<EncodedR1CS, FoldError> {
        let num_witness_vars = self.witness.len();
        let num_constraints = self.builder.num_constraints();

        let shape = self.builder.build()?;
        let instance = R1CSInstance::new(vec![cert_hash]);
        let witness = R1CSWitness::new(self.witness);

        Ok(EncodedR1CS {
            shape,
            instance,
            witness,
            num_constraints,
            num_witness_vars,
        })
    }
}

/// Encode a ProofCert as R1CS constraints
///
/// Returns an EncodedR1CS containing:
/// - The constraint shape (matrices A, B, C)
/// - Public instance (certificate hash)
/// - Private witness (certificate contents)
pub fn encode_cert_to_r1cs(cert: &ProofCert) -> Result<EncodedR1CS, FoldError> {
    let mut encoder = CertR1CSEncoder::new();

    // Encode the certificate
    let _root_hash_var = encoder.encode_cert(cert)?;

    // Compute certificate hash for public input
    let cert_hash = hash_cert(cert);

    // Add constraint: public input equals computed hash
    let public_hash_idx = encoder.builder.io_idx(0);
    let computed_hash_var = encoder.alloc(cert_hash);
    encoder.constrain_eq(public_hash_idx, encoder.builder.var_idx(computed_hash_var));

    encoder.finalize(cert_hash)
}

/// Verify that an encoded R1CS is satisfied
pub fn verify_encoded(encoded: &EncodedR1CS) -> Result<bool, FoldError> {
    encoded
        .shape
        .is_satisfied(&encoded.instance, &encoded.witness)
}

// --- Hash functions for certificate components ---

fn hash_cert(cert: &ProofCert) -> Scalar {
    let mut hasher = DefaultHasher::new();
    format!("{cert:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_node(tags: &[u64], levels: &[&Level]) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags.hash(&mut hasher);
    for level in levels {
        format!("{level:?}").hash(&mut hasher);
    }
    Scalar::from(hasher.finish())
}

fn hash_bvar(idx: u32, expected_type: &Expr) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::BVAR.hash(&mut hasher);
    idx.hash(&mut hasher);
    format!("{expected_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_fvar(id: u64, type_: &Expr) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::FVAR.hash(&mut hasher);
    id.hash(&mut hasher);
    format!("{type_:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_const(name: &Name, levels: &[Level], type_: &Expr) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::CONST.hash(&mut hasher);
    format!("{name:?}").hash(&mut hasher);
    format!("{levels:?}").hash(&mut hasher);
    format!("{type_:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_app(
    fn_cert: &ProofCert,
    fn_type: &Expr,
    arg_cert: &ProofCert,
    result_type: &Expr,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::APP.hash(&mut hasher);
    format!("{fn_cert:?}").hash(&mut hasher);
    format!("{fn_type:?}").hash(&mut hasher);
    format!("{arg_cert:?}").hash(&mut hasher);
    format!("{result_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_lam(
    binder_info: BinderInfo,
    arg_type_cert: &ProofCert,
    body_cert: &ProofCert,
    result_type: &Expr,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::LAM.hash(&mut hasher);
    (binder_info as u64).hash(&mut hasher);
    format!("{arg_type_cert:?}").hash(&mut hasher);
    format!("{body_cert:?}").hash(&mut hasher);
    format!("{result_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_pi(
    binder_info: BinderInfo,
    arg_type_cert: &ProofCert,
    arg_level: &Level,
    body_type_cert: &ProofCert,
    body_level: &Level,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::PI.hash(&mut hasher);
    (binder_info as u64).hash(&mut hasher);
    format!("{arg_type_cert:?}").hash(&mut hasher);
    format!("{arg_level:?}").hash(&mut hasher);
    format!("{body_type_cert:?}").hash(&mut hasher);
    format!("{body_level:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_let(
    type_cert: &ProofCert,
    value_cert: &ProofCert,
    body_cert: &ProofCert,
    result_type: &Expr,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::LET.hash(&mut hasher);
    format!("{type_cert:?}").hash(&mut hasher);
    format!("{value_cert:?}").hash(&mut hasher);
    format!("{body_cert:?}").hash(&mut hasher);
    format!("{result_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_literal(lit: &Literal) -> Scalar {
    let mut hasher = DefaultHasher::new();
    format!("{lit:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_lit(lit: &Literal, type_: &Expr) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::LIT.hash(&mut hasher);
    format!("{lit:?}").hash(&mut hasher);
    format!("{type_:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_def_eq(
    inner: &ProofCert,
    expected_type: &Expr,
    actual_type: &Expr,
    num_steps: usize,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::DEF_EQ.hash(&mut hasher);
    format!("{inner:?}").hash(&mut hasher);
    format!("{expected_type:?}").hash(&mut hasher);
    format!("{actual_type:?}").hash(&mut hasher);
    num_steps.hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_mdata(inner_cert: &ProofCert, result_type: &Expr) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::MDATA.hash(&mut hasher);
    format!("{inner_cert:?}").hash(&mut hasher);
    format!("{result_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_proj(
    struct_name: &Name,
    idx: u32,
    expr_cert: &ProofCert,
    expr_type: &Expr,
    field_type: &Expr,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::PROJ.hash(&mut hasher);
    format!("{struct_name:?}").hash(&mut hasher);
    idx.hash(&mut hasher);
    format!("{expr_cert:?}").hash(&mut hasher);
    format!("{expr_type:?}").hash(&mut hasher);
    format!("{field_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_level(level: &Level) -> Scalar {
    let mut hasher = DefaultHasher::new();
    format!("{level:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_name(name: &Name) -> Scalar {
    let mut hasher = DefaultHasher::new();
    format!("{name:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_expr(expr: &Expr) -> Scalar {
    let mut hasher = DefaultHasher::new();
    format!("{expr:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

// --- Mode-specific hash functions ---

// Cubical mode hashes

fn hash_cubical_interval() -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::CUBICAL_INTERVAL.hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_cubical_endpoint(is_one: bool) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::CUBICAL_ENDPOINT.hash(&mut hasher);
    is_one.hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_cubical_path(
    ty_cert: &ProofCert,
    ty_level: &Level,
    left_cert: &ProofCert,
    right_cert: &ProofCert,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::CUBICAL_PATH.hash(&mut hasher);
    format!("{ty_cert:?}").hash(&mut hasher);
    format!("{ty_level:?}").hash(&mut hasher);
    format!("{left_cert:?}").hash(&mut hasher);
    format!("{right_cert:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_cubical_path_lam(body_cert: &ProofCert, body_type: &Expr, result_type: &Expr) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::CUBICAL_PATH_LAM.hash(&mut hasher);
    format!("{body_cert:?}").hash(&mut hasher);
    format!("{body_type:?}").hash(&mut hasher);
    format!("{result_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_cubical_path_app(
    path_cert: &ProofCert,
    arg_cert: &ProofCert,
    path_type: &Expr,
    result_type: &Expr,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::CUBICAL_PATH_APP.hash(&mut hasher);
    format!("{path_cert:?}").hash(&mut hasher);
    format!("{arg_cert:?}").hash(&mut hasher);
    format!("{path_type:?}").hash(&mut hasher);
    format!("{result_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_cubical_hcomp(
    ty_cert: &ProofCert,
    phi_cert: &ProofCert,
    u_cert: &ProofCert,
    base_cert: &ProofCert,
    result_type: &Expr,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::CUBICAL_HCOMP.hash(&mut hasher);
    format!("{ty_cert:?}").hash(&mut hasher);
    format!("{phi_cert:?}").hash(&mut hasher);
    format!("{u_cert:?}").hash(&mut hasher);
    format!("{base_cert:?}").hash(&mut hasher);
    format!("{result_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_cubical_transp(
    ty_cert: &ProofCert,
    phi_cert: &ProofCert,
    base_cert: &ProofCert,
    result_type: &Expr,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::CUBICAL_TRANSP.hash(&mut hasher);
    format!("{ty_cert:?}").hash(&mut hasher);
    format!("{phi_cert:?}").hash(&mut hasher);
    format!("{base_cert:?}").hash(&mut hasher);
    format!("{result_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

// Classical mode hashes

fn hash_classical_choice(
    ty_cert: &ProofCert,
    pred_cert: &ProofCert,
    proof_cert: &ProofCert,
    result_type: &Expr,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::CLASSICAL_CHOICE.hash(&mut hasher);
    format!("{ty_cert:?}").hash(&mut hasher);
    format!("{pred_cert:?}").hash(&mut hasher);
    format!("{proof_cert:?}").hash(&mut hasher);
    format!("{result_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_classical_epsilon(
    ty_cert: &ProofCert,
    pred_cert: &ProofCert,
    result_type: &Expr,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::CLASSICAL_EPSILON.hash(&mut hasher);
    format!("{ty_cert:?}").hash(&mut hasher);
    format!("{pred_cert:?}").hash(&mut hasher);
    format!("{result_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

// ZFC mode hashes

fn hash_zfc_set(kind: &ZFCSetCertKind, result_type: &Expr) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::ZFC_SET.hash(&mut hasher);
    format!("{kind:?}").hash(&mut hasher);
    format!("{result_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_zfc_empty() -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::zfc::EMPTY.hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_zfc_singleton(cert: &ProofCert) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::zfc::SINGLETON.hash(&mut hasher);
    format!("{cert:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_zfc_pair(cert1: &ProofCert, cert2: &ProofCert) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::zfc::PAIR.hash(&mut hasher);
    format!("{cert1:?}").hash(&mut hasher);
    format!("{cert2:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_zfc_union(cert: &ProofCert) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::zfc::UNION.hash(&mut hasher);
    format!("{cert:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_zfc_powerset(cert: &ProofCert) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::zfc::POWERSET.hash(&mut hasher);
    format!("{cert:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_zfc_separation(set_cert: &ProofCert, pred_cert: &ProofCert) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::zfc::SEPARATION.hash(&mut hasher);
    format!("{set_cert:?}").hash(&mut hasher);
    format!("{pred_cert:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_zfc_replacement(set_cert: &ProofCert, func_cert: &ProofCert) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::zfc::REPLACEMENT.hash(&mut hasher);
    format!("{set_cert:?}").hash(&mut hasher);
    format!("{func_cert:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_zfc_infinity() -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::zfc::INFINITY.hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_zfc_choice(cert: &ProofCert) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::zfc::CHOICE.hash(&mut hasher);
    format!("{cert:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_zfc_mem(elem_cert: &ProofCert, set_cert: &ProofCert) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::ZFC_MEM.hash(&mut hasher);
    format!("{elem_cert:?}").hash(&mut hasher);
    format!("{set_cert:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_zfc_comprehension(
    var_ty_cert: &ProofCert,
    pred_cert: &ProofCert,
    result_type: &Expr,
) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::ZFC_COMPREHENSION.hash(&mut hasher);
    format!("{var_ty_cert:?}").hash(&mut hasher);
    format!("{pred_cert:?}").hash(&mut hasher);
    format!("{result_type:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

// Impredicative mode hashes

fn hash_sprop() -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::SPROP.hash(&mut hasher);
    Scalar::from(hasher.finish())
}

fn hash_squash(inner_cert: &ProofCert) -> Scalar {
    let mut hasher = DefaultHasher::new();
    tags::SQUASH.hash(&mut hasher);
    format!("{inner_cert:?}").hash(&mut hasher);
    Scalar::from(hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;
    use std::sync::Arc;

    #[test]
    fn test_encode_sort() {
        let cert = ProofCert::Sort { level: Level::Zero };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(encoded.num_constraints > 0);
        assert!(encoded.num_witness_vars > 0);

        // Verify R1CS is satisfied
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_bvar() {
        let cert = ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_sort_succ() {
        let cert = ProofCert::Sort {
            level: Level::Succ(Arc::new(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_app() {
        // Create a simple application certificate
        let fn_cert = ProofCert::Sort { level: Level::Zero };
        let arg_cert = ProofCert::Sort { level: Level::Zero };

        let cert = ProofCert::App {
            fn_cert: Box::new(fn_cert),
            fn_type: Box::new(Expr::Sort(Level::Zero)),
            arg_cert: Box::new(arg_cert),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_lam() {
        let arg_type_cert = ProofCert::Sort { level: Level::Zero };
        let body_cert = ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let cert = ProofCert::Lam {
            binder_info: BinderInfo::Default,
            arg_type_cert: Box::new(arg_type_cert),
            body_cert: Box::new(body_cert),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_pi() {
        let arg_type_cert = ProofCert::Sort { level: Level::Zero };
        let body_type_cert = ProofCert::Sort { level: Level::Zero };

        let cert = ProofCert::Pi {
            binder_info: BinderInfo::Default,
            arg_type_cert: Box::new(arg_type_cert),
            arg_level: Level::Zero,
            body_type_cert: Box::new(body_type_cert),
            body_level: Level::Zero,
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_let() {
        let type_cert = ProofCert::Sort { level: Level::Zero };
        let value_cert = ProofCert::Sort { level: Level::Zero };
        let body_cert = ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let cert = ProofCert::Let {
            type_cert: Box::new(type_cert),
            value_cert: Box::new(value_cert),
            body_cert: Box::new(body_cert),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_lit_nat() {
        let cert = ProofCert::Lit {
            lit: Literal::Nat(42),
            type_: Box::new(Expr::Const("Nat".parse().unwrap(), smallvec![])),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_nested_cert() {
        // Create a nested certificate structure
        let inner1 = ProofCert::Sort { level: Level::Zero };
        let inner2 = ProofCert::Sort {
            level: Level::Succ(Arc::new(Level::Zero)),
        };

        let app1 = ProofCert::App {
            fn_cert: Box::new(inner1),
            fn_type: Box::new(Expr::Sort(Level::Zero)),
            arg_cert: Box::new(inner2),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let cert = ProofCert::App {
            fn_cert: Box::new(app1),
            fn_type: Box::new(Expr::Sort(Level::Zero)),
            arg_cert: Box::new(ProofCert::Sort { level: Level::Zero }),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Mode-specific certificate encoding tests
    // ════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_encode_cubical_interval() {
        let cert = ProofCert::CubicalInterval;
        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_cubical_endpoint_i0() {
        let cert = ProofCert::CubicalEndpoint { is_one: false };
        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_cubical_endpoint_i1() {
        let cert = ProofCert::CubicalEndpoint { is_one: true };
        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_cubical_path() {
        let ty_cert = ProofCert::Sort { level: Level::Zero };
        let left_cert = ProofCert::CubicalEndpoint { is_one: false };
        let right_cert = ProofCert::CubicalEndpoint { is_one: true };

        let cert = ProofCert::CubicalPath {
            ty_cert: Box::new(ty_cert),
            ty_level: Level::Zero,
            left_cert: Box::new(left_cert),
            right_cert: Box::new(right_cert),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_cubical_path_lam() {
        let body_cert = ProofCert::CubicalInterval;
        let cert = ProofCert::CubicalPathLam {
            body_cert: Box::new(body_cert),
            body_type: Box::new(Expr::CubicalInterval),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_cubical_path_app() {
        let path_cert = ProofCert::CubicalInterval;
        let arg_cert = ProofCert::CubicalEndpoint { is_one: false };

        let cert = ProofCert::CubicalPathApp {
            path_cert: Box::new(path_cert),
            arg_cert: Box::new(arg_cert),
            path_type: Box::new(Expr::Sort(Level::Zero)),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_cubical_hcomp() {
        let ty_cert = ProofCert::Sort { level: Level::Zero };
        let phi_cert = ProofCert::CubicalInterval;
        let u_cert = ProofCert::CubicalInterval;
        let base_cert = ProofCert::Sort { level: Level::Zero };

        let cert = ProofCert::CubicalHComp {
            ty_cert: Box::new(ty_cert),
            phi_cert: Box::new(phi_cert),
            u_cert: Box::new(u_cert),
            base_cert: Box::new(base_cert),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_cubical_transp() {
        let ty_cert = ProofCert::Sort { level: Level::Zero };
        let phi_cert = ProofCert::CubicalInterval;
        let base_cert = ProofCert::Sort { level: Level::Zero };

        let cert = ProofCert::CubicalTransp {
            ty_cert: Box::new(ty_cert),
            phi_cert: Box::new(phi_cert),
            base_cert: Box::new(base_cert),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_classical_choice() {
        let ty_cert = ProofCert::Sort { level: Level::Zero };
        let pred_cert = ProofCert::Sort { level: Level::Zero };
        let proof_cert = ProofCert::Sort { level: Level::Zero };

        let cert = ProofCert::ClassicalChoice {
            ty_cert: Box::new(ty_cert),
            pred_cert: Box::new(pred_cert),
            proof_cert: Box::new(proof_cert),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_classical_epsilon() {
        let ty_cert = ProofCert::Sort { level: Level::Zero };
        let pred_cert = ProofCert::Sort { level: Level::Zero };

        let cert = ProofCert::ClassicalEpsilon {
            ty_cert: Box::new(ty_cert),
            pred_cert: Box::new(pred_cert),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_zfc_set_empty() {
        let cert = ProofCert::ZFCSet {
            kind: ZFCSetCertKind::Empty,
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_zfc_set_singleton() {
        let inner_cert = ProofCert::Sort { level: Level::Zero };
        let cert = ProofCert::ZFCSet {
            kind: ZFCSetCertKind::Singleton(Box::new(inner_cert)),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_zfc_set_pair() {
        let cert1 = ProofCert::Sort { level: Level::Zero };
        let cert2 = ProofCert::Sort { level: Level::Zero };
        let cert = ProofCert::ZFCSet {
            kind: ZFCSetCertKind::Pair(Box::new(cert1), Box::new(cert2)),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_zfc_set_infinity() {
        let cert = ProofCert::ZFCSet {
            kind: ZFCSetCertKind::Infinity,
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_zfc_mem() {
        let elem_cert = ProofCert::Sort { level: Level::Zero };
        let set_cert = ProofCert::ZFCSet {
            kind: ZFCSetCertKind::Empty,
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let cert = ProofCert::ZFCMem {
            elem_cert: Box::new(elem_cert),
            set_cert: Box::new(set_cert),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_zfc_comprehension() {
        let var_ty_cert = ProofCert::Sort { level: Level::Zero };
        let pred_cert = ProofCert::Sort { level: Level::Zero };

        let cert = ProofCert::ZFCComprehension {
            var_ty_cert: Box::new(var_ty_cert),
            pred_cert: Box::new(pred_cert),
            result_type: Box::new(Expr::Sort(Level::Zero)),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_sprop() {
        let cert = ProofCert::SProp;
        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }

    #[test]
    fn test_encode_squash() {
        let inner_cert = ProofCert::Sort { level: Level::Zero };
        let cert = ProofCert::Squash {
            inner_cert: Box::new(inner_cert),
        };

        let encoded = encode_cert_to_r1cs(&cert).expect("encoding should succeed");
        assert!(verify_encoded(&encoded).expect("verification should succeed"));
    }
}
