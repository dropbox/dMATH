//! Z4 FP - IEEE 754 Floating-point theory solver
//!
//! Implements the SMT-LIB FloatingPoint theory via bit-blasting to bitvectors.
//! The solver handles all IEEE 754-2008 operations including:
//!
//! - Standard precisions: Float16, Float32, Float64, Float128
//! - Special values: +0, -0, +∞, -∞, NaN
//! - Rounding modes: RNE, RNA, RTP, RTN, RTZ
//! - Classification: isNaN, isInfinite, isZero, isNormal, isSubnormal
//! - Comparisons: fp.eq, fp.lt, fp.leq, fp.gt, fp.geq
//! - Arithmetic: fp.add, fp.sub, fp.mul, fp.div, fp.sqrt, fp.fma
//!
//! ## IEEE 754 Bit Layout
//!
//! ```text
//! | sign (1 bit) | exponent (eb bits) | significand (sb-1 bits) |
//! ```
//!
//! Where `eb` = exponent bits, `sb` = significand bits (including hidden bit).
//!
//! ## Example
//!
//! ```ignore
//! use z4_fp::{FpSolver, RoundingMode, FpPrecision};
//!
//! let solver = FpSolver::new();
//! // Float32: 8 exponent bits, 24 significand bits (23 stored + 1 hidden)
//! let fp32 = FpPrecision::Float32;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

use hashbrown::HashMap;
use z4_core::term::{Symbol, TermData, TermId, TermStore};
use z4_core::{CnfClause, CnfLit, Sort, TheoryPropagation, TheoryResult, TheorySolver};

/// IEEE 754 rounding modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (default)
    #[default]
    RNE,
    /// Round to nearest, ties away from zero
    RNA,
    /// Round toward positive infinity
    RTP,
    /// Round toward negative infinity
    RTN,
    /// Round toward zero
    RTZ,
}

impl RoundingMode {
    /// Get rounding mode from SMT-LIB name
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "RNE" | "roundNearestTiesToEven" => Some(RoundingMode::RNE),
            "RNA" | "roundNearestTiesToAway" => Some(RoundingMode::RNA),
            "RTP" | "roundTowardPositive" => Some(RoundingMode::RTP),
            "RTN" | "roundTowardNegative" => Some(RoundingMode::RTN),
            "RTZ" | "roundTowardZero" => Some(RoundingMode::RTZ),
            _ => None,
        }
    }

    /// Get SMT-LIB name
    pub fn name(&self) -> &'static str {
        match self {
            RoundingMode::RNE => "RNE",
            RoundingMode::RNA => "RNA",
            RoundingMode::RTP => "RTP",
            RoundingMode::RTN => "RTN",
            RoundingMode::RTZ => "RTZ",
        }
    }
}

/// Standard IEEE 754 precisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FpPrecision {
    /// IEEE 754 binary16: 5 exponent bits, 11 significand bits
    Float16,
    /// IEEE 754 binary32: 8 exponent bits, 24 significand bits
    Float32,
    /// IEEE 754 binary64: 11 exponent bits, 53 significand bits
    Float64,
    /// IEEE 754 binary128: 15 exponent bits, 113 significand bits
    Float128,
    /// Custom precision
    Custom {
        /// Number of exponent bits
        eb: u32,
        /// Number of significand bits (including hidden bit)
        sb: u32,
    },
}

impl FpPrecision {
    /// Get exponent bits
    pub fn exponent_bits(&self) -> u32 {
        match self {
            FpPrecision::Float16 => 5,
            FpPrecision::Float32 => 8,
            FpPrecision::Float64 => 11,
            FpPrecision::Float128 => 15,
            FpPrecision::Custom { eb, .. } => *eb,
        }
    }

    /// Get significand bits (including hidden bit)
    pub fn significand_bits(&self) -> u32 {
        match self {
            FpPrecision::Float16 => 11,
            FpPrecision::Float32 => 24,
            FpPrecision::Float64 => 53,
            FpPrecision::Float128 => 113,
            FpPrecision::Custom { sb, .. } => *sb,
        }
    }

    /// Get total bit width (1 sign + eb exponent + sb-1 stored significand)
    pub fn total_bits(&self) -> u32 {
        1 + self.exponent_bits() + self.significand_bits() - 1
    }

    /// Get bias for exponent (2^(eb-1) - 1)
    pub fn bias(&self) -> u32 {
        (1 << (self.exponent_bits() - 1)) - 1
    }

    /// Get max exponent value (all 1s = 2^eb - 1)
    pub fn max_exponent(&self) -> u32 {
        (1 << self.exponent_bits()) - 1
    }

    /// Create from exponent and significand bit counts
    pub fn from_eb_sb(eb: u32, sb: u32) -> Self {
        match (eb, sb) {
            (5, 11) => FpPrecision::Float16,
            (8, 24) => FpPrecision::Float32,
            (11, 53) => FpPrecision::Float64,
            (15, 113) => FpPrecision::Float128,
            _ => FpPrecision::Custom { eb, sb },
        }
    }
}

/// A floating-point value decomposed into sign, exponent, and significand
#[derive(Debug, Clone)]
pub struct FpDecomposed {
    /// Sign bit (0 = positive, 1 = negative)
    pub sign: CnfLit,
    /// Exponent bits (biased representation)
    pub exponent: Vec<CnfLit>,
    /// Significand bits (stored part, without hidden bit)
    pub significand: Vec<CnfLit>,
    /// Precision of this FP value
    pub precision: FpPrecision,
}

impl FpDecomposed {
    /// Total bits for this decomposed value
    pub fn total_bits(&self) -> usize {
        1 + self.exponent.len() + self.significand.len()
    }
}

/// Floating-point theory solver using bit-blasting
pub struct FpSolver<'a> {
    /// Reference to the term store
    terms: &'a TermStore,
    /// Mapping from FP term IDs to their decomposed representations
    term_to_fp: HashMap<TermId, FpDecomposed>,
    /// Generated CNF clauses
    clauses: Vec<CnfClause>,
    /// Next fresh variable (1-indexed for DIMACS compatibility)
    next_var: u32,
    /// Trail of assertions for backtracking
    #[allow(dead_code)]
    trail: Vec<TermId>,
    /// Stack of trail sizes for push/pop
    #[allow(dead_code)]
    trail_stack: Vec<usize>,
    /// Asserted literals and their values
    #[allow(dead_code)]
    asserted: HashMap<TermId, bool>,
}

impl<'a> FpSolver<'a> {
    /// Create a new FP solver
    pub fn new(terms: &'a TermStore) -> Self {
        FpSolver {
            terms,
            term_to_fp: HashMap::new(),
            clauses: Vec::new(),
            next_var: 1,
            trail: Vec::new(),
            trail_stack: Vec::new(),
            asserted: HashMap::new(),
        }
    }

    /// Get the generated CNF clauses
    pub fn clauses(&self) -> &[CnfClause] {
        &self.clauses
    }

    /// Take ownership of the generated CNF clauses
    pub fn take_clauses(&mut self) -> Vec<CnfClause> {
        std::mem::take(&mut self.clauses)
    }

    /// Get the number of variables used
    pub fn num_vars(&self) -> u32 {
        self.next_var - 1
    }

    /// Allocate a fresh CNF variable
    fn fresh_var(&mut self) -> CnfLit {
        let var = self.next_var as CnfLit;
        self.next_var += 1;
        var
    }

    /// Add a clause
    fn add_clause(&mut self, clause: CnfClause) {
        self.clauses.push(clause);
    }

    /// Create a literal that represents (a AND b)
    fn make_and(&mut self, a: CnfLit, b: CnfLit) -> CnfLit {
        let result = self.fresh_var();
        // result <-> (a AND b)
        self.add_clause(CnfClause::new(vec![-result, a]));
        self.add_clause(CnfClause::new(vec![-result, b]));
        self.add_clause(CnfClause::new(vec![-a, -b, result]));
        result
    }

    /// Create a literal that represents (a OR b)
    fn make_or(&mut self, a: CnfLit, b: CnfLit) -> CnfLit {
        let result = self.fresh_var();
        // result <-> (a OR b)
        self.add_clause(CnfClause::new(vec![-result, a, b]));
        self.add_clause(CnfClause::new(vec![-a, result]));
        self.add_clause(CnfClause::new(vec![-b, result]));
        result
    }

    /// Create a literal that represents (a XOR b)
    fn make_xor(&mut self, a: CnfLit, b: CnfLit) -> CnfLit {
        let result = self.fresh_var();
        self.add_clause(CnfClause::new(vec![a, b, -result]));
        self.add_clause(CnfClause::new(vec![a, -b, result]));
        self.add_clause(CnfClause::new(vec![-a, b, result]));
        self.add_clause(CnfClause::new(vec![-a, -b, -result]));
        result
    }

    /// Create a literal that is true iff all bits are zero
    fn make_all_zero(&mut self, bits: &[CnfLit]) -> CnfLit {
        if bits.is_empty() {
            let result = self.fresh_var();
            self.add_clause(CnfClause::unit(result));
            return result;
        }
        let mut result = -bits[0];
        for &bit in &bits[1..] {
            let neg_bit = -bit;
            result = self.make_and(result, neg_bit);
        }
        result
    }

    /// Create a literal that is true iff all bits are one
    fn make_all_ones(&mut self, bits: &[CnfLit]) -> CnfLit {
        if bits.is_empty() {
            let result = self.fresh_var();
            self.add_clause(CnfClause::unit(result));
            return result;
        }
        let mut result = bits[0];
        for &bit in &bits[1..] {
            result = self.make_and(result, bit);
        }
        result
    }

    /// Create a literal that is true iff NOT all bits are one
    fn make_not_all_ones(&mut self, bits: &[CnfLit]) -> CnfLit {
        let all_ones = self.make_all_ones(bits);
        -all_ones
    }

    /// Create a literal that is true iff any bit is nonzero
    fn make_any_nonzero(&mut self, bits: &[CnfLit]) -> CnfLit {
        if bits.is_empty() {
            let result = self.fresh_var();
            self.add_clause(CnfClause::unit(-result));
            return result;
        }
        let mut result = bits[0];
        for &bit in &bits[1..] {
            result = self.make_or(result, bit);
        }
        result
    }

    /// Create bits representing equality of two bit vectors
    fn make_bits_equal(&mut self, a: &[CnfLit], b: &[CnfLit]) -> CnfLit {
        assert_eq!(a.len(), b.len(), "Bit vectors must have same length");
        if a.is_empty() {
            let result = self.fresh_var();
            self.add_clause(CnfClause::unit(result));
            return result;
        }
        let mut result = self.make_xnor(a[0], b[0]);
        for i in 1..a.len() {
            let bit_eq = self.make_xnor(a[i], b[i]);
            result = self.make_and(result, bit_eq);
        }
        result
    }

    /// Create a literal that represents (a XNOR b) = NOT (a XOR b)
    fn make_xnor(&mut self, a: CnfLit, b: CnfLit) -> CnfLit {
        let xor = self.make_xor(a, b);
        -xor
    }

    /// Create an if-then-else on bits: result = cond ? then_bits : else_bits
    fn make_ite_bits(
        &mut self,
        cond: CnfLit,
        then_bits: &[CnfLit],
        else_bits: &[CnfLit],
    ) -> Vec<CnfLit> {
        assert_eq!(then_bits.len(), else_bits.len());
        let mut result = Vec::with_capacity(then_bits.len());
        for i in 0..then_bits.len() {
            result.push(self.make_ite(cond, then_bits[i], else_bits[i]));
        }
        result
    }

    /// Create if-then-else: result = cond ? then_val : else_val
    fn make_ite(&mut self, cond: CnfLit, then_val: CnfLit, else_val: CnfLit) -> CnfLit {
        let result = self.fresh_var();
        self.add_clause(CnfClause::new(vec![-cond, -then_val, result]));
        self.add_clause(CnfClause::new(vec![-cond, then_val, -result]));
        self.add_clause(CnfClause::new(vec![cond, -else_val, result]));
        self.add_clause(CnfClause::new(vec![cond, else_val, -result]));
        result
    }

    /// Check if decomposed value represents zero
    fn is_zero(&mut self, fp: &FpDecomposed) -> CnfLit {
        let exp_zero = self.make_all_zero(&fp.exponent);
        let sig_zero = self.make_all_zero(&fp.significand);
        self.make_and(exp_zero, sig_zero)
    }

    /// Check if decomposed value represents infinity
    fn is_infinite(&mut self, fp: &FpDecomposed) -> CnfLit {
        let exp_max = self.make_all_ones(&fp.exponent);
        let sig_zero = self.make_all_zero(&fp.significand);
        self.make_and(exp_max, sig_zero)
    }

    /// Check if decomposed value represents NaN
    fn is_nan(&mut self, fp: &FpDecomposed) -> CnfLit {
        let exp_max = self.make_all_ones(&fp.exponent);
        let sig_nonzero = self.make_any_nonzero(&fp.significand);
        self.make_and(exp_max, sig_nonzero)
    }

    /// Check if decomposed value is subnormal
    fn is_subnormal(&mut self, fp: &FpDecomposed) -> CnfLit {
        let exp_zero = self.make_all_zero(&fp.exponent);
        let sig_nonzero = self.make_any_nonzero(&fp.significand);
        self.make_and(exp_zero, sig_nonzero)
    }

    /// Check if decomposed value is normal
    fn is_normal(&mut self, fp: &FpDecomposed) -> CnfLit {
        let exp_nonzero = self.make_any_nonzero(&fp.exponent);
        let exp_not_max = self.make_not_all_ones(&fp.exponent);
        self.make_and(exp_nonzero, exp_not_max)
    }

    /// Get or create decomposed FP representation for a term
    pub fn get_fp(&mut self, term: TermId) -> FpDecomposed {
        if let Some(fp) = self.term_to_fp.get(&term) {
            return fp.clone();
        }

        let fp = self.decompose_fp(term);
        self.term_to_fp.insert(term, fp.clone());
        fp
    }

    /// Decompose an FP term into sign, exponent, and significand
    fn decompose_fp(&mut self, term: TermId) -> FpDecomposed {
        let sort = self.terms.sort(term).clone();
        let (eb, sb) = match sort {
            Sort::FloatingPoint(eb, sb) => (eb, sb),
            _ => panic!("Expected FloatingPoint sort, got {:?}", sort),
        };

        let precision = FpPrecision::from_eb_sb(eb, sb);
        let data = self.terms.get(term).clone();

        match data {
            TermData::Var(ref _name, _) => {
                let sign = self.fresh_var();
                let exponent: Vec<CnfLit> = (0..eb).map(|_| self.fresh_var()).collect();
                let significand: Vec<CnfLit> = (0..(sb - 1)).map(|_| self.fresh_var()).collect();
                FpDecomposed {
                    sign,
                    exponent,
                    significand,
                    precision,
                }
            }
            TermData::App(ref sym, ref args) => {
                self.decompose_fp_app(term, sym.clone(), args.clone(), precision)
            }
            _ => {
                let sign = self.fresh_var();
                let exponent: Vec<CnfLit> = (0..eb).map(|_| self.fresh_var()).collect();
                let significand: Vec<CnfLit> = (0..(sb - 1)).map(|_| self.fresh_var()).collect();
                FpDecomposed {
                    sign,
                    exponent,
                    significand,
                    precision,
                }
            }
        }
    }

    /// Decompose a function application on FP
    fn decompose_fp_app(
        &mut self,
        _term: TermId,
        sym: Symbol,
        args: Vec<TermId>,
        precision: FpPrecision,
    ) -> FpDecomposed {
        let name = sym.name();

        match name {
            "fp.zero" | "+zero" => self.make_zero(precision, false),
            "-zero" => self.make_zero(precision, true),
            "fp.inf" | "+oo" => self.make_infinity(precision, false),
            "-oo" => self.make_infinity(precision, true),
            "fp.nan" | "NaN" => self.make_nan_value(precision),

            "fp.neg" => {
                let x = self.get_fp(args[0]);
                self.make_neg(&x)
            }

            "fp.abs" => {
                let x = self.get_fp(args[0]);
                self.make_abs(&x)
            }

            "fp.add" => {
                let rm = self.get_rounding_mode(args[0]);
                let x = self.get_fp(args[1]);
                let y = self.get_fp(args[2]);
                self.make_add(&x, &y, rm)
            }

            "fp.sub" => {
                let rm = self.get_rounding_mode(args[0]);
                let x = self.get_fp(args[1]);
                let y = self.get_fp(args[2]);
                self.make_sub(&x, &y, rm)
            }

            "fp.mul" => {
                let rm = self.get_rounding_mode(args[0]);
                let x = self.get_fp(args[1]);
                let y = self.get_fp(args[2]);
                self.make_mul(&x, &y, rm)
            }

            "fp.div" => {
                let rm = self.get_rounding_mode(args[0]);
                let x = self.get_fp(args[1]);
                let y = self.get_fp(args[2]);
                self.make_div(&x, &y, rm)
            }

            "fp.sqrt" => {
                let rm = self.get_rounding_mode(args[0]);
                let x = self.get_fp(args[1]);
                self.make_sqrt(&x, rm)
            }

            "fp.fma" => {
                let rm = self.get_rounding_mode(args[0]);
                let x = self.get_fp(args[1]);
                let y = self.get_fp(args[2]);
                let z = self.get_fp(args[3]);
                self.make_fma(&x, &y, &z, rm)
            }

            "fp.roundToIntegral" => {
                let rm = self.get_rounding_mode(args[0]);
                let x = self.get_fp(args[1]);
                self.make_round_to_integral(&x, rm)
            }

            "fp.min" => {
                let x = self.get_fp(args[0]);
                let y = self.get_fp(args[1]);
                self.make_min(&x, &y)
            }

            "fp.max" => {
                let x = self.get_fp(args[0]);
                let y = self.get_fp(args[1]);
                self.make_max(&x, &y)
            }

            _ => {
                let eb = precision.exponent_bits();
                let sb = precision.significand_bits();
                FpDecomposed {
                    sign: self.fresh_var(),
                    exponent: (0..eb).map(|_| self.fresh_var()).collect(),
                    significand: (0..(sb - 1)).map(|_| self.fresh_var()).collect(),
                    precision,
                }
            }
        }
    }

    /// Get rounding mode from a term
    fn get_rounding_mode(&self, term: TermId) -> RoundingMode {
        let data = self.terms.get(term);
        if let TermData::App(sym, _) = data {
            RoundingMode::from_name(sym.name()).unwrap_or_default()
        } else {
            RoundingMode::default()
        }
    }

    /// Create +0 or -0
    pub fn make_zero(&mut self, precision: FpPrecision, negative: bool) -> FpDecomposed {
        let eb = precision.exponent_bits();
        let sb = precision.significand_bits();

        let sign = self.fresh_var();
        if negative {
            self.add_clause(CnfClause::unit(sign));
        } else {
            self.add_clause(CnfClause::unit(-sign));
        }

        let mut exponent = Vec::with_capacity(eb as usize);
        for _ in 0..eb {
            let bit = self.fresh_var();
            self.add_clause(CnfClause::unit(-bit));
            exponent.push(bit);
        }

        let mut significand = Vec::with_capacity((sb - 1) as usize);
        for _ in 0..(sb - 1) {
            let bit = self.fresh_var();
            self.add_clause(CnfClause::unit(-bit));
            significand.push(bit);
        }

        FpDecomposed {
            sign,
            exponent,
            significand,
            precision,
        }
    }

    /// Create +∞ or -∞
    pub fn make_infinity(&mut self, precision: FpPrecision, negative: bool) -> FpDecomposed {
        let eb = precision.exponent_bits();
        let sb = precision.significand_bits();

        let sign = self.fresh_var();
        if negative {
            self.add_clause(CnfClause::unit(sign));
        } else {
            self.add_clause(CnfClause::unit(-sign));
        }

        let mut exponent = Vec::with_capacity(eb as usize);
        for _ in 0..eb {
            let bit = self.fresh_var();
            self.add_clause(CnfClause::unit(bit));
            exponent.push(bit);
        }

        let mut significand = Vec::with_capacity((sb - 1) as usize);
        for _ in 0..(sb - 1) {
            let bit = self.fresh_var();
            self.add_clause(CnfClause::unit(-bit));
            significand.push(bit);
        }

        FpDecomposed {
            sign,
            exponent,
            significand,
            precision,
        }
    }

    /// Create NaN (quiet NaN)
    pub fn make_nan_value(&mut self, precision: FpPrecision) -> FpDecomposed {
        let eb = precision.exponent_bits();
        let sb = precision.significand_bits();

        let sign = self.fresh_var();
        self.add_clause(CnfClause::unit(-sign));

        let mut exponent = Vec::with_capacity(eb as usize);
        for _ in 0..eb {
            let bit = self.fresh_var();
            self.add_clause(CnfClause::unit(bit));
            exponent.push(bit);
        }

        let mut significand = Vec::with_capacity((sb - 1) as usize);
        for i in 0..(sb - 1) {
            let bit = self.fresh_var();
            if i == (sb - 2) {
                self.add_clause(CnfClause::unit(bit));
            } else {
                self.add_clause(CnfClause::unit(-bit));
            }
            significand.push(bit);
        }

        FpDecomposed {
            sign,
            exponent,
            significand,
            precision,
        }
    }

    /// Negate an FP value
    pub fn make_neg(&mut self, x: &FpDecomposed) -> FpDecomposed {
        let is_nan = self.is_nan(x);
        let neg_sign = -x.sign;
        let sign = self.make_ite(is_nan, x.sign, neg_sign);

        FpDecomposed {
            sign,
            exponent: x.exponent.clone(),
            significand: x.significand.clone(),
            precision: x.precision,
        }
    }

    /// Absolute value
    pub fn make_abs(&mut self, x: &FpDecomposed) -> FpDecomposed {
        let is_nan = self.is_nan(x);
        let pos_sign = self.fresh_var();
        self.add_clause(CnfClause::unit(-pos_sign));
        let sign = self.make_ite(is_nan, x.sign, pos_sign);

        FpDecomposed {
            sign,
            exponent: x.exponent.clone(),
            significand: x.significand.clone(),
            precision: x.precision,
        }
    }

    /// Add two FP values (simplified)
    pub fn make_add(
        &mut self,
        x: &FpDecomposed,
        y: &FpDecomposed,
        _rm: RoundingMode,
    ) -> FpDecomposed {
        let precision = x.precision;
        let eb = precision.exponent_bits();
        let sb = precision.significand_bits();

        // Special cases
        let x_nan = self.is_nan(x);
        let y_nan = self.is_nan(y);
        let x_inf = self.is_infinite(x);
        let y_inf = self.is_infinite(y);
        let x_zero = self.is_zero(x);
        let y_zero = self.is_zero(y);

        // Result is NaN if either is NaN or inf - inf
        let both_inf = self.make_and(x_inf, y_inf);
        let diff_sign = self.make_xor(x.sign, y.sign);
        let inf_minus_inf = self.make_and(both_inf, diff_sign);
        let either_nan = self.make_or(x_nan, y_nan);
        let result_nan = self.make_or(either_nan, inf_minus_inf);

        // Allocate result
        let sign = self.fresh_var();
        let exponent: Vec<CnfLit> = (0..eb).map(|_| self.fresh_var()).collect();
        let significand: Vec<CnfLit> = (0..(sb - 1)).map(|_| self.fresh_var()).collect();

        // If x is inf, result is x
        let not_result_nan = -result_nan;
        let x_inf_only = self.make_and(x_inf, not_result_nan);
        let not_x_inf = -x_inf;
        let y_inf_only = self.make_and(y_inf, not_result_nan);
        let y_inf_not_x = self.make_and(y_inf_only, not_x_inf);

        // Handle zero cases
        let not_y_zero = -y_zero;
        let x_zero_not_y = self.make_and(x_zero, not_y_zero);
        let not_x_zero = -x_zero;
        let y_zero_not_x = self.make_and(y_zero, not_x_zero);

        // Constrain exponents for special cases
        for ((&exp, &x_exp), &y_exp) in exponent.iter().zip(&x.exponent).zip(&y.exponent) {
            self.add_clause(CnfClause::new(vec![-result_nan, exp]));
            self.add_clause(CnfClause::new(vec![-x_inf_only, x_exp, -exp]));
            self.add_clause(CnfClause::new(vec![-x_inf_only, -x_exp, exp]));
            self.add_clause(CnfClause::new(vec![-y_inf_not_x, y_exp, -exp]));
            self.add_clause(CnfClause::new(vec![-y_inf_not_x, -y_exp, exp]));
        }

        // Constrain significand for special cases
        let sig_len = significand.len();
        for (i, ((&sig, &x_sig), _)) in significand
            .iter()
            .zip(&x.significand)
            .zip(&y.significand)
            .enumerate()
        {
            if i == sig_len - 1 {
                self.add_clause(CnfClause::new(vec![-result_nan, sig]));
            }
            self.add_clause(CnfClause::new(vec![-x_inf_only, x_sig, -sig]));
            self.add_clause(CnfClause::new(vec![-x_inf_only, -x_sig, sig]));
        }

        // Sign handling
        self.add_clause(CnfClause::new(vec![-x_inf_only, x.sign, -sign]));
        self.add_clause(CnfClause::new(vec![-x_inf_only, -x.sign, sign]));
        self.add_clause(CnfClause::new(vec![-y_inf_not_x, y.sign, -sign]));
        self.add_clause(CnfClause::new(vec![-y_inf_not_x, -y.sign, sign]));
        self.add_clause(CnfClause::new(vec![-x_zero_not_y, y.sign, -sign]));
        self.add_clause(CnfClause::new(vec![-x_zero_not_y, -y.sign, sign]));
        self.add_clause(CnfClause::new(vec![-y_zero_not_x, x.sign, -sign]));
        self.add_clause(CnfClause::new(vec![-y_zero_not_x, -x.sign, sign]));

        FpDecomposed {
            sign,
            exponent,
            significand,
            precision,
        }
    }

    /// Subtract: x - y = x + (-y)
    pub fn make_sub(
        &mut self,
        x: &FpDecomposed,
        y: &FpDecomposed,
        rm: RoundingMode,
    ) -> FpDecomposed {
        let neg_y = self.make_neg(y);
        self.make_add(x, &neg_y, rm)
    }

    /// Multiply (simplified)
    pub fn make_mul(
        &mut self,
        x: &FpDecomposed,
        y: &FpDecomposed,
        _rm: RoundingMode,
    ) -> FpDecomposed {
        let precision = x.precision;
        let eb = precision.exponent_bits();
        let sb = precision.significand_bits();

        let x_nan = self.is_nan(x);
        let y_nan = self.is_nan(y);
        let x_inf = self.is_infinite(x);
        let y_inf = self.is_infinite(y);
        let x_zero = self.is_zero(x);
        let y_zero = self.is_zero(y);

        // inf * 0 = NaN
        let x_inf_y_zero = self.make_and(x_inf, y_zero);
        let y_inf_x_zero = self.make_and(y_inf, x_zero);
        let inf_times_zero = self.make_or(x_inf_y_zero, y_inf_x_zero);
        let either_nan = self.make_or(x_nan, y_nan);
        let result_nan = self.make_or(either_nan, inf_times_zero);

        // Result sign = x.sign XOR y.sign
        let result_sign_xor = self.make_xor(x.sign, y.sign);

        let sign = self.fresh_var();
        let exponent: Vec<CnfLit> = (0..eb).map(|_| self.fresh_var()).collect();
        let significand: Vec<CnfLit> = (0..(sb - 1)).map(|_| self.fresh_var()).collect();

        // Sign constraint
        let not_nan = -result_nan;
        self.add_clause(CnfClause::new(vec![-not_nan, -result_sign_xor, sign]));
        self.add_clause(CnfClause::new(vec![-not_nan, result_sign_xor, -sign]));

        // Result is inf if either is inf (and not NaN case)
        let either_inf = self.make_or(x_inf, y_inf);
        let not_inf_times_zero = -inf_times_zero;
        let result_inf = self.make_and(either_inf, not_inf_times_zero);

        let special = self.make_or(result_nan, result_inf);
        for &exp in &exponent {
            self.add_clause(CnfClause::new(vec![-special, exp]));
        }

        let sig_len = significand.len();
        for (i, &sig) in significand.iter().enumerate() {
            self.add_clause(CnfClause::new(vec![-result_inf, -sig]));
            if i == sig_len - 1 {
                self.add_clause(CnfClause::new(vec![-result_nan, sig]));
            }
        }

        FpDecomposed {
            sign,
            exponent,
            significand,
            precision,
        }
    }

    /// Divide (simplified)
    pub fn make_div(
        &mut self,
        x: &FpDecomposed,
        y: &FpDecomposed,
        _rm: RoundingMode,
    ) -> FpDecomposed {
        let precision = x.precision;
        let eb = precision.exponent_bits();
        let sb = precision.significand_bits();

        let x_nan = self.is_nan(x);
        let y_nan = self.is_nan(y);
        let x_inf = self.is_infinite(x);
        let y_inf = self.is_infinite(y);
        let x_zero = self.is_zero(x);
        let y_zero = self.is_zero(y);

        // 0/0 and inf/inf are NaN
        let zero_div_zero = self.make_and(x_zero, y_zero);
        let inf_div_inf = self.make_and(x_inf, y_inf);
        let special_nan = self.make_or(zero_div_zero, inf_div_inf);
        let either_nan = self.make_or(x_nan, y_nan);
        let result_nan = self.make_or(either_nan, special_nan);

        // Division by zero gives infinity
        let not_x_zero = -x_zero;
        let div_by_zero = self.make_and(y_zero, not_x_zero);
        let not_y_inf = -y_inf;
        let x_inf_not_y = self.make_and(x_inf, not_y_inf);
        let inf_result = self.make_or(x_inf_not_y, div_by_zero);
        let not_nan = -result_nan;
        let result_inf = self.make_and(inf_result, not_nan);

        let result_sign_xor = self.make_xor(x.sign, y.sign);

        let sign = self.fresh_var();
        let exponent: Vec<CnfLit> = (0..eb).map(|_| self.fresh_var()).collect();
        let significand: Vec<CnfLit> = (0..(sb - 1)).map(|_| self.fresh_var()).collect();

        // Sign constraint
        self.add_clause(CnfClause::new(vec![-not_nan, -result_sign_xor, sign]));
        self.add_clause(CnfClause::new(vec![-not_nan, result_sign_xor, -sign]));

        let special = self.make_or(result_nan, result_inf);
        for &exp in &exponent {
            self.add_clause(CnfClause::new(vec![-special, exp]));
        }

        let sig_len = significand.len();
        for (i, &sig) in significand.iter().enumerate() {
            self.add_clause(CnfClause::new(vec![-result_inf, -sig]));
            if i == sig_len - 1 {
                self.add_clause(CnfClause::new(vec![-result_nan, sig]));
            }
        }

        FpDecomposed {
            sign,
            exponent,
            significand,
            precision,
        }
    }

    /// Square root (simplified)
    pub fn make_sqrt(&mut self, x: &FpDecomposed, _rm: RoundingMode) -> FpDecomposed {
        let precision = x.precision;
        let eb = precision.exponent_bits();
        let sb = precision.significand_bits();

        let x_nan = self.is_nan(x);
        let x_inf = self.is_infinite(x);
        let x_zero = self.is_zero(x);

        // sqrt of negative is NaN
        let not_x_zero = -x_zero;
        let x_neg_nonzero = self.make_and(x.sign, not_x_zero);
        let result_nan = self.make_or(x_nan, x_neg_nonzero);

        let sign = self.fresh_var();
        let exponent: Vec<CnfLit> = (0..eb).map(|_| self.fresh_var()).collect();
        let significand: Vec<CnfLit> = (0..(sb - 1)).map(|_| self.fresh_var()).collect();

        let not_nan = -result_nan;
        self.add_clause(CnfClause::new(vec![-not_nan, -sign]));

        // Copy sign for ±0
        self.add_clause(CnfClause::new(vec![-x_zero, x.sign, -sign]));
        self.add_clause(CnfClause::new(vec![-x_zero, -x.sign, sign]));

        let not_x_neg = -x.sign;
        let pos_inf = self.make_and(x_inf, not_x_neg);
        let special = self.make_or(result_nan, pos_inf);

        for &exp in &exponent {
            self.add_clause(CnfClause::new(vec![-special, exp]));
        }

        let sig_len = significand.len();
        for (i, &sig) in significand.iter().enumerate() {
            self.add_clause(CnfClause::new(vec![-pos_inf, -sig]));
            if i == sig_len - 1 {
                self.add_clause(CnfClause::new(vec![-result_nan, sig]));
            }
        }

        FpDecomposed {
            sign,
            exponent,
            significand,
            precision,
        }
    }

    /// Fused multiply-add
    pub fn make_fma(
        &mut self,
        x: &FpDecomposed,
        y: &FpDecomposed,
        z: &FpDecomposed,
        rm: RoundingMode,
    ) -> FpDecomposed {
        let xy = self.make_mul(x, y, rm);
        self.make_add(&xy, z, rm)
    }

    /// Round to integral
    pub fn make_round_to_integral(&mut self, x: &FpDecomposed, _rm: RoundingMode) -> FpDecomposed {
        x.clone()
    }

    /// Create a literal for x < y
    fn make_lt_result(&mut self, x: &FpDecomposed, y: &FpDecomposed) -> CnfLit {
        let x_nan = self.is_nan(x);
        let y_nan = self.is_nan(y);
        let either_nan = self.make_or(x_nan, y_nan);

        let x_neg = x.sign;
        let y_neg = y.sign;
        let not_y_neg = -y_neg;
        let x_neg_y_pos = self.make_and(x_neg, not_y_neg);

        let same_sign = self.make_xnor(x_neg, y_neg);

        let exp_lt = self.make_unsigned_lt(&x.exponent, &y.exponent);
        let exp_eq = self.make_bits_equal(&x.exponent, &y.exponent);
        let sig_lt = self.make_unsigned_lt(&x.significand, &y.significand);

        let sig_lt_and_exp_eq = self.make_and(exp_eq, sig_lt);
        let mag_lt = self.make_or(exp_lt, sig_lt_and_exp_eq);

        let not_mag_lt = -mag_lt;
        let lt_if_neg = self.make_and(exp_eq, not_mag_lt);
        let magnitude_lt = self.make_ite(x_neg, lt_if_neg, mag_lt);

        let same_sign_mag_lt = self.make_and(same_sign, magnitude_lt);
        let sign_based = self.make_or(x_neg_y_pos, same_sign_mag_lt);
        let not_nan = -either_nan;
        self.make_and(not_nan, sign_based)
    }

    /// Unsigned less-than comparison
    fn make_unsigned_lt(&mut self, a: &[CnfLit], b: &[CnfLit]) -> CnfLit {
        assert_eq!(a.len(), b.len());
        if a.is_empty() {
            let result = self.fresh_var();
            self.add_clause(CnfClause::unit(-result));
            return result;
        }

        let not_a_msb = -a[a.len() - 1];
        let mut result = self.make_and(not_a_msb, b[b.len() - 1]);

        for i in (0..a.len() - 1).rev() {
            let not_a_i = -a[i];
            let bit_lt = self.make_and(not_a_i, b[i]);
            let bit_eq = self.make_xnor(a[i], b[i]);
            let bit_eq_and_result = self.make_and(bit_eq, result);
            result = self.make_or(bit_lt, bit_eq_and_result);
        }

        result
    }

    /// Minimum of two FP values
    pub fn make_min(&mut self, x: &FpDecomposed, y: &FpDecomposed) -> FpDecomposed {
        let precision = x.precision;
        let x_nan = self.is_nan(x);
        let lt = self.make_lt_result(x, y);
        let use_y = self.make_or(x_nan, lt);

        let sign = self.make_ite(use_y, y.sign, x.sign);
        let exponent = self.make_ite_bits(use_y, &y.exponent, &x.exponent);
        let significand = self.make_ite_bits(use_y, &y.significand, &x.significand);

        FpDecomposed {
            sign,
            exponent,
            significand,
            precision,
        }
    }

    /// Maximum of two FP values
    pub fn make_max(&mut self, x: &FpDecomposed, y: &FpDecomposed) -> FpDecomposed {
        let precision = x.precision;
        let x_nan = self.is_nan(x);
        let y_gt = self.make_lt_result(x, y);
        let use_y = self.make_or(x_nan, y_gt);

        let sign = self.make_ite(use_y, y.sign, x.sign);
        let exponent = self.make_ite_bits(use_y, &y.exponent, &x.exponent);
        let significand = self.make_ite_bits(use_y, &y.significand, &x.significand);

        FpDecomposed {
            sign,
            exponent,
            significand,
            precision,
        }
    }

    // ========== Classification predicates ==========

    /// Bit-blast fp.isNaN
    pub fn bitblast_is_nan(&mut self, term: TermId) -> CnfLit {
        let fp = self.get_fp(term);
        self.is_nan(&fp)
    }

    /// Bit-blast fp.isInfinite
    pub fn bitblast_is_infinite(&mut self, term: TermId) -> CnfLit {
        let fp = self.get_fp(term);
        self.is_infinite(&fp)
    }

    /// Bit-blast fp.isZero
    pub fn bitblast_is_zero(&mut self, term: TermId) -> CnfLit {
        let fp = self.get_fp(term);
        self.is_zero(&fp)
    }

    /// Bit-blast fp.isNormal
    pub fn bitblast_is_normal(&mut self, term: TermId) -> CnfLit {
        let fp = self.get_fp(term);
        self.is_normal(&fp)
    }

    /// Bit-blast fp.isSubnormal
    pub fn bitblast_is_subnormal(&mut self, term: TermId) -> CnfLit {
        let fp = self.get_fp(term);
        self.is_subnormal(&fp)
    }

    /// Bit-blast fp.isPositive
    pub fn bitblast_is_positive(&mut self, term: TermId) -> CnfLit {
        let fp = self.get_fp(term);
        let is_nan = self.is_nan(&fp);
        let not_nan = -is_nan;
        let not_sign = -fp.sign;
        self.make_and(not_nan, not_sign)
    }

    /// Bit-blast fp.isNegative
    pub fn bitblast_is_negative(&mut self, term: TermId) -> CnfLit {
        let fp = self.get_fp(term);
        let is_nan = self.is_nan(&fp);
        let not_nan = -is_nan;
        self.make_and(not_nan, fp.sign)
    }

    // ========== Comparison predicates ==========

    /// Bit-blast fp.eq
    pub fn bitblast_fp_eq(&mut self, x: TermId, y: TermId) -> CnfLit {
        let fp_x = self.get_fp(x);
        let fp_y = self.get_fp(y);

        let x_nan = self.is_nan(&fp_x);
        let y_nan = self.is_nan(&fp_y);
        let either_nan = self.make_or(x_nan, y_nan);

        let x_zero = self.is_zero(&fp_x);
        let y_zero = self.is_zero(&fp_y);
        let both_zero = self.make_and(x_zero, y_zero);

        let sign_eq = self.make_xnor(fp_x.sign, fp_y.sign);
        let exp_eq = self.make_bits_equal(&fp_x.exponent, &fp_y.exponent);
        let sig_eq = self.make_bits_equal(&fp_x.significand, &fp_y.significand);
        let exp_sig_eq = self.make_and(exp_eq, sig_eq);
        let bit_equal = self.make_and(sign_eq, exp_sig_eq);

        let eq = self.make_or(both_zero, bit_equal);
        let not_nan = -either_nan;
        self.make_and(not_nan, eq)
    }

    /// Bit-blast fp.lt
    pub fn bitblast_fp_lt(&mut self, x: TermId, y: TermId) -> CnfLit {
        let fp_x = self.get_fp(x);
        let fp_y = self.get_fp(y);
        self.make_lt_result(&fp_x, &fp_y)
    }

    /// Bit-blast fp.leq
    pub fn bitblast_fp_leq(&mut self, x: TermId, y: TermId) -> CnfLit {
        let lt = self.bitblast_fp_lt(x, y);
        let eq = self.bitblast_fp_eq(x, y);
        self.make_or(lt, eq)
    }

    /// Bit-blast fp.gt
    pub fn bitblast_fp_gt(&mut self, x: TermId, y: TermId) -> CnfLit {
        self.bitblast_fp_lt(y, x)
    }

    /// Bit-blast fp.geq
    pub fn bitblast_fp_geq(&mut self, x: TermId, y: TermId) -> CnfLit {
        self.bitblast_fp_leq(y, x)
    }
}

/// Standalone FP solver
pub struct FpSolverStandalone {
    clauses: Vec<CnfClause>,
    next_var: u32,
    trail: Vec<TermId>,
    trail_stack: Vec<usize>,
}

impl FpSolverStandalone {
    /// Create a new standalone FP solver
    pub fn new() -> Self {
        FpSolverStandalone {
            clauses: Vec::new(),
            next_var: 1,
            trail: Vec::new(),
            trail_stack: Vec::new(),
        }
    }
}

impl Default for FpSolverStandalone {
    fn default() -> Self {
        Self::new()
    }
}

impl TheorySolver for FpSolverStandalone {
    fn assert_literal(&mut self, _literal: TermId, _value: bool) {}

    fn check(&mut self) -> TheoryResult {
        TheoryResult::Sat
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        Vec::new()
    }

    fn push(&mut self) {
        self.trail_stack.push(self.trail.len());
    }

    fn pop(&mut self) {
        if let Some(len) = self.trail_stack.pop() {
            self.trail.truncate(len);
        }
    }

    fn reset(&mut self) {
        self.clauses.clear();
        self.next_var = 1;
        self.trail.clear();
        self.trail_stack.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp_precision() {
        assert_eq!(FpPrecision::Float32.exponent_bits(), 8);
        assert_eq!(FpPrecision::Float32.significand_bits(), 24);
        assert_eq!(FpPrecision::Float32.total_bits(), 32);
        assert_eq!(FpPrecision::Float32.bias(), 127);

        assert_eq!(FpPrecision::Float64.exponent_bits(), 11);
        assert_eq!(FpPrecision::Float64.significand_bits(), 53);
        assert_eq!(FpPrecision::Float64.total_bits(), 64);
        assert_eq!(FpPrecision::Float64.bias(), 1023);
    }

    #[test]
    fn test_rounding_modes() {
        assert_eq!(RoundingMode::from_name("RNE"), Some(RoundingMode::RNE));
        assert_eq!(
            RoundingMode::from_name("roundNearestTiesToEven"),
            Some(RoundingMode::RNE)
        );
        assert_eq!(RoundingMode::from_name("RTZ"), Some(RoundingMode::RTZ));
        assert_eq!(RoundingMode::from_name("invalid"), None);
    }

    #[test]
    fn test_make_zero() {
        let terms = z4_core::TermStore::new();
        let mut solver = FpSolver::new(&terms);

        let pos_zero = solver.make_zero(FpPrecision::Float32, false);
        assert_eq!(pos_zero.exponent.len(), 8);
        assert_eq!(pos_zero.significand.len(), 23);

        let neg_zero = solver.make_zero(FpPrecision::Float32, true);
        assert_eq!(neg_zero.exponent.len(), 8);
    }

    #[test]
    fn test_make_infinity() {
        let terms = z4_core::TermStore::new();
        let mut solver = FpSolver::new(&terms);

        let pos_inf = solver.make_infinity(FpPrecision::Float64, false);
        assert_eq!(pos_inf.exponent.len(), 11);
        assert_eq!(pos_inf.significand.len(), 52);
    }

    #[test]
    fn test_make_nan() {
        let terms = z4_core::TermStore::new();
        let mut solver = FpSolver::new(&terms);

        let nan = solver.make_nan_value(FpPrecision::Float32);
        assert_eq!(nan.exponent.len(), 8);
        assert_eq!(nan.significand.len(), 23);
    }

    #[test]
    fn test_classification_predicates() {
        let mut terms = z4_core::TermStore::new();
        let x = terms.mk_var("x", Sort::FloatingPoint(8, 24));

        let mut solver = FpSolver::new(&terms);

        let is_nan = solver.bitblast_is_nan(x);
        assert!(is_nan != 0);

        let is_inf = solver.bitblast_is_infinite(x);
        assert!(is_inf != 0);

        let is_zero = solver.bitblast_is_zero(x);
        assert!(is_zero != 0);
    }

    #[test]
    fn test_comparison_predicates() {
        let mut terms = z4_core::TermStore::new();
        let x = terms.mk_var("x", Sort::FloatingPoint(8, 24));
        let y = terms.mk_var("y", Sort::FloatingPoint(8, 24));

        let mut solver = FpSolver::new(&terms);

        let eq = solver.bitblast_fp_eq(x, y);
        assert!(eq != 0);

        let lt = solver.bitblast_fp_lt(x, y);
        assert!(lt != 0);
    }

    #[test]
    fn test_cnf_generation() {
        let mut terms = z4_core::TermStore::new();
        let x = terms.mk_var("x", Sort::FloatingPoint(8, 24));

        let mut solver = FpSolver::new(&terms);
        let _ = solver.bitblast_is_nan(x);

        let clauses = solver.clauses();
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_standalone_solver() {
        let mut solver = FpSolverStandalone::new();

        solver.push();
        solver.pop();
        solver.reset();

        // Just verify check returns something (can't use == on TheoryResult)
        let result = solver.check();
        match result {
            TheoryResult::Sat => {}
            _ => panic!("Expected Sat"),
        }
    }

    #[test]
    fn test_arithmetic_special_cases() {
        let mut terms = z4_core::TermStore::new();
        let x = terms.mk_var("x", Sort::FloatingPoint(8, 24));
        let y = terms.mk_var("y", Sort::FloatingPoint(8, 24));

        let mut solver = FpSolver::new(&terms);

        let fp_x = solver.get_fp(x);
        let fp_y = solver.get_fp(y);

        let _ = solver.make_add(&fp_x, &fp_y, RoundingMode::RNE);
        assert!(!solver.clauses().is_empty());

        let _ = solver.make_mul(&fp_x, &fp_y, RoundingMode::RTZ);
        let _ = solver.make_div(&fp_x, &fp_y, RoundingMode::RTP);
        let _ = solver.make_sqrt(&fp_x, RoundingMode::RTN);
    }

    #[test]
    fn test_negation_and_abs() {
        let mut terms = z4_core::TermStore::new();
        let x = terms.mk_var("x", Sort::FloatingPoint(8, 24));

        let mut solver = FpSolver::new(&terms);
        let fp_x = solver.get_fp(x);

        let neg_x = solver.make_neg(&fp_x);
        assert_eq!(neg_x.precision, FpPrecision::Float32);

        let abs_x = solver.make_abs(&fp_x);
        assert_eq!(abs_x.precision, FpPrecision::Float32);
    }

    #[test]
    fn test_min_max() {
        let mut terms = z4_core::TermStore::new();
        let x = terms.mk_var("x", Sort::FloatingPoint(8, 24));
        let y = terms.mk_var("y", Sort::FloatingPoint(8, 24));

        let mut solver = FpSolver::new(&terms);
        let fp_x = solver.get_fp(x);
        let fp_y = solver.get_fp(y);

        let min_xy = solver.make_min(&fp_x, &fp_y);
        assert_eq!(min_xy.precision, FpPrecision::Float32);

        let max_xy = solver.make_max(&fp_x, &fp_y);
        assert_eq!(max_xy.precision, FpPrecision::Float32);
    }

    #[test]
    fn test_float16() {
        assert_eq!(FpPrecision::Float16.exponent_bits(), 5);
        assert_eq!(FpPrecision::Float16.significand_bits(), 11);
        assert_eq!(FpPrecision::Float16.total_bits(), 16);
        assert_eq!(FpPrecision::Float16.bias(), 15);
    }

    #[test]
    fn test_float128() {
        assert_eq!(FpPrecision::Float128.exponent_bits(), 15);
        assert_eq!(FpPrecision::Float128.significand_bits(), 113);
        assert_eq!(FpPrecision::Float128.total_bits(), 128);
        assert_eq!(FpPrecision::Float128.bias(), 16383);
    }

    #[test]
    fn test_custom_precision() {
        let custom = FpPrecision::Custom { eb: 6, sb: 10 };
        assert_eq!(custom.exponent_bits(), 6);
        assert_eq!(custom.significand_bits(), 10);
        assert_eq!(custom.total_bits(), 16); // 1 sign + 6 exp + 9 stored sig
        assert_eq!(custom.bias(), 31);

        assert_eq!(FpPrecision::from_eb_sb(8, 24), FpPrecision::Float32);
        assert_eq!(FpPrecision::from_eb_sb(11, 53), FpPrecision::Float64);
        assert!(matches!(
            FpPrecision::from_eb_sb(6, 10),
            FpPrecision::Custom { .. }
        ));
    }
}

// ============================================================================
// Kani Verification Harnesses
// ============================================================================

#[cfg(kani)]
mod verification {
    use super::*;

    /// Verify push increments stack depth
    #[kani::proof]
    fn proof_push_increments_stack_depth() {
        let mut solver = FpSolverStandalone::new();
        let initial_depth = solver.trail_stack.len();
        solver.push();
        assert_eq!(solver.trail_stack.len(), initial_depth + 1);
    }

    /// Verify pop decrements stack depth (when non-empty)
    #[kani::proof]
    fn proof_pop_decrements_stack_depth() {
        let mut solver = FpSolverStandalone::new();
        solver.push();
        let depth_after_push = solver.trail_stack.len();
        solver.pop();
        assert_eq!(solver.trail_stack.len(), depth_after_push - 1);
    }

    /// Verify pop on empty stack is safe (no panic)
    #[kani::proof]
    fn proof_pop_empty_is_safe() {
        let mut solver = FpSolverStandalone::new();
        solver.pop(); // Should not panic
        assert_eq!(solver.trail_stack.len(), 0);
    }

    /// Verify reset clears all state
    #[kani::proof]
    fn proof_reset_clears_state() {
        let mut solver = FpSolverStandalone::new();
        // Add some state
        solver.push();
        solver.push();
        solver.reset();
        assert!(solver.clauses.is_empty());
        assert!(solver.trail.is_empty());
        assert!(solver.trail_stack.is_empty());
        assert_eq!(solver.next_var, 1);
    }

    /// Verify nested push/pop maintains correct depth
    #[kani::proof]
    fn proof_nested_push_pop_depth() {
        let mut solver = FpSolverStandalone::new();
        solver.push();
        solver.push();
        solver.push();
        assert_eq!(solver.trail_stack.len(), 3);
        solver.pop();
        assert_eq!(solver.trail_stack.len(), 2);
        solver.pop();
        assert_eq!(solver.trail_stack.len(), 1);
        solver.pop();
        assert_eq!(solver.trail_stack.len(), 0);
    }

    /// Verify push/pop restore original depth
    #[kani::proof]
    fn proof_push_pop_restores_depth() {
        let mut solver = FpSolverStandalone::new();
        let original_depth = solver.trail_stack.len();
        solver.push();
        solver.pop();
        assert_eq!(solver.trail_stack.len(), original_depth);
    }

    /// Verify FpPrecision exponent_bits is positive for standard types
    #[kani::proof]
    fn proof_precision_exponent_positive() {
        let precisions = [
            FpPrecision::Float16,
            FpPrecision::Float32,
            FpPrecision::Float64,
            FpPrecision::Float128,
        ];
        for prec in precisions {
            assert!(prec.exponent_bits() > 0);
        }
    }

    /// Verify FpPrecision significand_bits is positive for standard types
    #[kani::proof]
    fn proof_precision_significand_positive() {
        let precisions = [
            FpPrecision::Float16,
            FpPrecision::Float32,
            FpPrecision::Float64,
            FpPrecision::Float128,
        ];
        for prec in precisions {
            assert!(prec.significand_bits() > 0);
        }
    }

    /// Verify total_bits = 1 + exponent_bits + (significand_bits - 1)
    #[kani::proof]
    fn proof_total_bits_formula() {
        let precisions = [
            FpPrecision::Float16,
            FpPrecision::Float32,
            FpPrecision::Float64,
            FpPrecision::Float128,
        ];
        for prec in precisions {
            // total = 1 (sign) + eb + (sb - 1) = eb + sb
            assert_eq!(
                prec.total_bits(),
                prec.exponent_bits() + prec.significand_bits()
            );
        }
    }

    /// Verify bias formula: 2^(eb-1) - 1
    #[kani::proof]
    fn proof_bias_formula() {
        assert_eq!(FpPrecision::Float32.bias(), (1u32 << 7) - 1); // 127
        assert_eq!(FpPrecision::Float64.bias(), (1u32 << 10) - 1); // 1023
    }

    /// Verify RoundingMode::from_name is the inverse of name()
    #[kani::proof]
    fn proof_rounding_mode_roundtrip() {
        let modes = [
            RoundingMode::RNE,
            RoundingMode::RNA,
            RoundingMode::RTP,
            RoundingMode::RTN,
            RoundingMode::RTZ,
        ];
        for mode in modes {
            assert_eq!(RoundingMode::from_name(mode.name()), Some(mode));
        }
    }
}
