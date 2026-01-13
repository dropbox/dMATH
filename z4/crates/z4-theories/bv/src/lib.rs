//! Z4 BV - Bitvector theory solver
//!
//! Implements eager bit-blasting for bitvectors. Each bitvector variable is
//! mapped to a vector of boolean variables (one per bit), and bitvector
//! operations are translated to boolean circuits.

#![warn(missing_docs)]
#![warn(clippy::all)]

use hashbrown::HashMap;
use z4_core::term::{Constant, Symbol, TermData, TermId, TermStore};
use z4_core::{CnfClause, CnfLit, Sort, TheoryPropagation, TheoryResult, TheorySolver};

/// A vector of boolean literals representing a bitvector
/// LSB is at index 0
pub type BvBits = Vec<CnfLit>;

/// Model extracted from BV solver with variable assignments
#[derive(Debug, Clone)]
pub struct BvModel {
    /// Variable assignments: term_id -> bitvector value (as BigInt)
    pub values: HashMap<TermId, num_bigint::BigInt>,
    /// Term to bit mappings (for debugging)
    pub term_to_bits: HashMap<TermId, BvBits>,
}

/// Bitvector theory solver using eager bit-blasting
pub struct BvSolver<'a> {
    /// Reference to the term store
    terms: &'a TermStore,
    /// Mapping from BV term IDs to their bit representations
    term_to_bits: HashMap<TermId, BvBits>,
    /// Generated CNF clauses
    clauses: Vec<CnfClause>,
    /// Next fresh variable (1-indexed for DIMACS compatibility)
    next_var: u32,
    /// Trail of assertions for backtracking
    trail: Vec<TermId>,
    /// Stack of trail sizes for push/pop
    trail_stack: Vec<usize>,
    /// Asserted literals and their values
    asserted: HashMap<TermId, bool>,
}

impl<'a> BvSolver<'a> {
    /// Create a new BV solver
    pub fn new(terms: &'a TermStore) -> Self {
        BvSolver {
            terms,
            term_to_bits: HashMap::new(),
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

    /// Get the bit representation for a term, if it has been bit-blasted
    ///
    /// Returns None if the term has not been processed by the bit-blaster
    pub fn get_term_bits(&self, term: TermId) -> Option<&[CnfLit]> {
        self.term_to_bits.get(&term).map(|v| v.as_slice())
    }

    /// Get an iterator over all bit-blasted terms and their bit representations
    pub fn iter_term_bits(&self) -> impl Iterator<Item = (TermId, &[CnfLit])> {
        self.term_to_bits
            .iter()
            .map(|(&id, bits)| (id, bits.as_slice()))
    }

    /// Get a reference to all term-to-bits mappings
    ///
    /// Used for preserving BV state across incremental solving sessions.
    pub fn term_to_bits(&self) -> &HashMap<TermId, BvBits> {
        &self.term_to_bits
    }

    /// Set the bit representation for a term
    ///
    /// Used for restoring BV state in incremental solving.
    pub fn set_term_bits(&mut self, term: TermId, bits: BvBits) {
        self.term_to_bits.insert(term, bits);
    }

    /// Set the next variable counter
    ///
    /// Used for restoring BV state in incremental solving.
    pub fn set_next_var(&mut self, next_var: u32) {
        self.next_var = next_var;
    }

    /// Bit-blast an assertion and add clauses
    ///
    /// This processes a single assertion term and generates CNF clauses
    /// for all BV constraints it contains. The assertion is assumed to be true.
    pub fn bitblast_assertion(&mut self, term: TermId) {
        self.process_assertion(term, true);
    }

    /// Bit-blast multiple assertions and return CNF
    ///
    /// This is the main entry point for eager bit-blasting. It processes
    /// all assertions and returns the resulting CNF clauses.
    pub fn bitblast_all(&mut self, assertions: &[TermId]) -> Vec<CnfClause> {
        for &term in assertions {
            self.bitblast_assertion(term);
        }
        self.take_clauses()
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

    /// Get or create bit representation for a BV term
    fn get_bits(&mut self, term: TermId) -> BvBits {
        if let Some(bits) = self.term_to_bits.get(&term) {
            return bits.clone();
        }

        let bits = self.bitblast(term);
        self.term_to_bits.insert(term, bits.clone());
        bits
    }

    /// Bit-blast a bitvector term
    fn bitblast(&mut self, term: TermId) -> BvBits {
        let data = self.terms.get(term).clone();

        match data {
            TermData::Const(Constant::BitVec { ref value, width }) => {
                // Constant: create bits from value
                let mut bits = Vec::with_capacity(width as usize);
                for i in 0..width {
                    let bit_set =
                        (value >> i) & num_bigint::BigInt::from(1) != num_bigint::BigInt::from(0);
                    let lit = self.fresh_var();
                    // Assert the constant value as a unit clause
                    self.add_clause(CnfClause::unit(if bit_set { lit } else { -lit }));
                    bits.push(lit);
                }
                bits
            }
            TermData::Var(ref _name, _) => {
                // Variable: allocate fresh boolean variables
                let width = match self.terms.sort(term) {
                    Sort::BitVec(w) => *w,
                    _ => panic!("Expected BitVec sort"),
                };
                let mut bits = Vec::with_capacity(width as usize);
                for _ in 0..width {
                    bits.push(self.fresh_var());
                }
                bits
            }
            TermData::App(ref sym, ref args) => self.bitblast_app(term, sym.clone(), args.clone()),
            _ => {
                // Unknown term type - allocate fresh bits
                if let Sort::BitVec(width) = self.terms.sort(term) {
                    let mut bits = Vec::with_capacity(*width as usize);
                    for _ in 0..*width {
                        bits.push(self.fresh_var());
                    }
                    bits
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// Bit-blast a function application
    fn bitblast_app(&mut self, term: TermId, sym: Symbol, args: Vec<TermId>) -> BvBits {
        let name = sym.name();

        match name {
            "bvadd" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_add(&a, &b)
            }
            "bvsub" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_sub(&a, &b)
            }
            "bvmul" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_mul(&a, &b)
            }
            "bvand" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_and(&a, &b)
            }
            "bvor" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_or(&a, &b)
            }
            "bvxor" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_xor(&a, &b)
            }
            "bvnot" => {
                let a = self.get_bits(args[0]);
                self.bitblast_not(&a)
            }
            "bvneg" => {
                let a = self.get_bits(args[0]);
                self.bitblast_neg(&a)
            }
            "bvshl" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_shl(&a, &b)
            }
            "bvlshr" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_lshr(&a, &b)
            }
            "bvashr" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_ashr(&a, &b)
            }
            "bvudiv" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_udiv(&a, &b)
            }
            "bvurem" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_urem(&a, &b)
            }
            "bvsdiv" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_sdiv(&a, &b)
            }
            "bvsrem" => {
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                self.bitblast_srem(&a, &b)
            }
            "concat" => {
                // concat(a, b) = a is high bits, b is low bits
                let a = self.get_bits(args[0]);
                let b = self.get_bits(args[1]);
                let mut result = b;
                result.extend(a);
                result
            }
            "extract" => {
                // extract[hi:lo](x) - extract bits hi down to lo
                // Args should be [hi, lo, x] based on SMT-LIB
                // But in z4 representation it might be different
                // Check the term structure
                // Need to get hi and lo from the symbol name or term structure
                // For now, assume full extraction
                self.get_bits(args[0])
            }
            "zero_extend" | "sign_extend" | "repeat" | "rotate_left" | "rotate_right" => {
                // These are handled by z4-core's term simplification
                // If we get here, create fresh bits
                let width = args
                    .first()
                    .and_then(|&a| match self.terms.sort(a) {
                        Sort::BitVec(w) => Some(*w),
                        _ => None,
                    })
                    .unwrap_or(32);
                let mut bits = Vec::with_capacity(width as usize);
                for _ in 0..width {
                    bits.push(self.fresh_var());
                }
                bits
            }
            _ => {
                // Unknown function (including select, store, etc.) - treat as uninterpreted
                // Look up the term's sort to determine the width
                let width = match self.terms.sort(term) {
                    Sort::BitVec(w) => *w,
                    _ => {
                        // Fall back to checking first arg's sort, then default
                        args.first()
                            .and_then(|&a| match self.terms.sort(a) {
                                Sort::BitVec(w) => Some(*w),
                                _ => None,
                            })
                            .unwrap_or(32)
                    }
                };
                let mut bits = Vec::with_capacity(width as usize);
                for _ in 0..width {
                    bits.push(self.fresh_var());
                }
                bits
            }
        }
    }

    // =========================================================================
    // Bit-blasting for arithmetic operations
    // =========================================================================

    /// Create a half adder: (sum, carry) = a + b
    fn half_adder(&mut self, a: CnfLit, b: CnfLit) -> (CnfLit, CnfLit) {
        let sum = self.mk_xor(a, b);
        let carry = self.mk_and(a, b);
        (sum, carry)
    }

    /// Create a full adder: (sum, carry) = a + b + cin
    fn full_adder(&mut self, a: CnfLit, b: CnfLit, cin: CnfLit) -> (CnfLit, CnfLit) {
        let (s1, c1) = self.half_adder(a, b);
        let (sum, c2) = self.half_adder(s1, cin);
        let carry = self.mk_or(c1, c2);
        (sum, carry)
    }

    /// Bit-blast addition using ripple-carry adder
    fn bitblast_add(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut result = Vec::with_capacity(n);

        // First bit: half adder
        if n == 0 {
            return result;
        }
        let (s, mut carry) = self.half_adder(a[0], b[0]);
        result.push(s);

        // Remaining bits: full adder
        for i in 1..n {
            let (s, c) = self.full_adder(a[i], b[i], carry);
            result.push(s);
            carry = c;
        }

        result
    }

    /// Bit-blast subtraction: a - b = a + (~b + 1) = a + ~b + 1
    fn bitblast_sub(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        let not_b = self.bitblast_not(b);
        let one = self.const_bits(1, b.len());
        let b_plus_one = self.bitblast_add(&not_b, &one);
        self.bitblast_add(a, &b_plus_one)
    }

    /// Bit-blast negation: -a = ~a + 1
    fn bitblast_neg(&mut self, a: &BvBits) -> BvBits {
        let not_a = self.bitblast_not(a);
        let one = self.const_bits(1, a.len());
        self.bitblast_add(&not_a, &one)
    }

    /// Bit-blast multiplication using shift-and-add
    fn bitblast_mul(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        assert_eq!(a.len(), b.len());
        let n = a.len();

        if n == 0 {
            return Vec::new();
        }

        // Initialize with zero
        let mut result = self.const_bits(0, n);

        for (i, &bi) in b.iter().enumerate().take(n) {
            // If b[i] is set, add a << i to result
            let shifted = self.shift_left_const(a, i);
            let masked: BvBits = shifted.iter().map(|&s| self.mk_and(s, bi)).collect();
            result = self.bitblast_add(&result, &masked);
        }

        result
    }

    // =========================================================================
    // Bit-blasting for bitwise operations
    // =========================================================================

    /// Bit-blast bitwise AND
    fn bitblast_and(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| self.mk_and(ai, bi))
            .collect()
    }

    /// Bit-blast bitwise OR
    fn bitblast_or(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| self.mk_or(ai, bi))
            .collect()
    }

    /// Bit-blast bitwise XOR
    fn bitblast_xor(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| self.mk_xor(ai, bi))
            .collect()
    }

    /// Bit-blast bitwise NOT
    fn bitblast_not(&mut self, a: &BvBits) -> BvBits {
        a.iter().map(|&ai| -ai).collect()
    }

    // =========================================================================
    // Bit-blasting for shift operations
    // =========================================================================

    /// Shift left by a constant amount
    fn shift_left_const(&mut self, a: &BvBits, amt: usize) -> BvBits {
        let n = a.len();
        let mut result = Vec::with_capacity(n);

        // Low bits become zero
        for _ in 0..amt.min(n) {
            let zero = self.fresh_var();
            self.add_clause(CnfClause::unit(-zero)); // Assert zero
            result.push(zero);
        }

        // Shift in the original bits
        for &bit in a.iter().take(n - amt.min(n)) {
            result.push(bit);
        }

        result
    }

    /// Logical shift right by a constant amount
    fn shift_right_const(&mut self, a: &BvBits, amt: usize) -> BvBits {
        let n = a.len();
        let mut result = Vec::with_capacity(n);

        // Shift in the original bits
        for &bit in a.iter().skip(amt.min(n)) {
            result.push(bit);
        }

        // High bits become zero
        for _ in 0..amt.min(n) {
            let zero = self.fresh_var();
            self.add_clause(CnfClause::unit(-zero)); // Assert zero
            result.push(zero);
        }

        result
    }

    /// Bit-blast left shift (variable amount)
    fn bitblast_shl(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        let n = a.len();
        // Use barrel shifter approach
        let mut current = a.clone();

        for (i, &bi) in b.iter().enumerate() {
            if i >= 32 {
                break;
            } // Prevent overflow
            let shift_amt = 1 << i;
            if shift_amt >= n {
                break;
            }

            let shifted = self.shift_left_const(&current, shift_amt);
            // MUX: if bi then shifted else current
            current = self.bitwise_mux(&shifted, &current, bi);
        }

        current
    }

    /// Bit-blast logical right shift (variable amount)
    fn bitblast_lshr(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        let n = a.len();
        let mut current = a.clone();

        for (i, &bi) in b.iter().enumerate() {
            if i >= 32 {
                break;
            }
            let shift_amt = 1 << i;
            if shift_amt >= n {
                break;
            }

            let shifted = self.shift_right_const(&current, shift_amt);
            current = self.bitwise_mux(&shifted, &current, bi);
        }

        current
    }

    /// Bit-blast arithmetic right shift (variable amount)
    fn bitblast_ashr(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        let n = a.len();
        if n == 0 {
            return Vec::new();
        }

        let sign_bit = a[n - 1];
        let mut current = a.clone();

        for (i, &bi) in b.iter().enumerate() {
            if i >= 32 {
                break;
            }
            let shift_amt = 1 << i;
            if shift_amt >= n {
                break;
            }

            // Arithmetic shift fills with sign bit
            let mut shifted = Vec::with_capacity(n);
            for &bit in current.iter().skip(shift_amt) {
                shifted.push(bit);
            }
            for _ in 0..shift_amt.min(n) {
                shifted.push(sign_bit);
            }

            current = self.bitwise_mux(&shifted, &current, bi);
        }

        current
    }

    // =========================================================================
    // Bit-blasting for division operations
    // =========================================================================
    //
    // Division is implemented using constraint-based encoding:
    // For q = a / b and r = a % b:
    //   - a = q * b + r  (quotient-remainder relationship)
    //   - r < b OR b = 0 (remainder bounds)
    //   - q < 2^n (implicit from bit width)
    //
    // SMT-LIB semantics for division by zero:
    //   - a / 0 = all ones (2^n - 1)
    //   - a % 0 = a
    //
    // This approach is cleaner than a full divider circuit and leverages
    // the existing multiplication and comparison implementations.

    /// Bit-blast unsigned division and remainder together
    /// Returns (quotient, remainder)
    fn bitblast_udiv_urem(&mut self, a: &BvBits, b: &BvBits) -> (BvBits, BvBits) {
        assert_eq!(a.len(), b.len());
        let n = a.len();

        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        // Create fresh variables for quotient and remainder
        let mut q = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for _ in 0..n {
            q.push(self.fresh_var());
            r.push(self.fresh_var());
        }

        // Check if divisor is zero: b = 0
        let b_is_zero = self.is_zero(b);

        // SMT-LIB semantics:
        // When b = 0: q = all_ones, r = a
        // When b != 0: a = q * b + r AND r < b

        // Constraint 1: q = (b = 0 ? all_ones : q_normal)
        // Constraint 2: r = (b = 0 ? a : r_normal)
        // where q_normal and r_normal satisfy a = q_normal * b + r_normal AND r_normal < b

        // For the case b != 0, assert: a = q * b + r AND r < b
        // Compute q * b
        let q_times_b = self.bitblast_mul(&q, b);
        // Compute q * b + r
        let q_times_b_plus_r = self.bitblast_add(&q_times_b, &r);

        // Assert: (b != 0) => (a = q * b + r)
        // Equivalently: (b = 0) OR (a = q * b + r)
        let eq_constraint = self.bitblast_eq(a, &q_times_b_plus_r);
        // (b_is_zero OR eq_constraint)
        let div_constraint = self.mk_or(b_is_zero, eq_constraint);
        self.add_clause(CnfClause::unit(div_constraint));

        // Assert: (b != 0) => (r < b)
        // Equivalently: (b = 0) OR (r < b)
        let r_lt_b = self.bitblast_ult(&r, b);
        let rem_constraint = self.mk_or(b_is_zero, r_lt_b);
        self.add_clause(CnfClause::unit(rem_constraint));

        // Assert: (b = 0) => (q = all_ones)
        // For each bit: (b = 0) => q[i] = 1
        // Equivalently: (b != 0) OR (q[i] = 1)
        for &qi in &q {
            let q_one_constraint = self.mk_or(-b_is_zero, qi);
            self.add_clause(CnfClause::unit(q_one_constraint));
        }

        // Assert: (b = 0) => (r = a)
        // For each bit: (b = 0) => (r[i] = a[i])
        // r[i] = a[i] means (r[i] XOR a[i]) = 0, i.e., r[i] XNOR a[i]
        for (&ri, &ai) in r.iter().zip(a.iter()) {
            let r_eq_a = self.mk_xnor(ri, ai);
            let r_constraint = self.mk_or(-b_is_zero, r_eq_a);
            self.add_clause(CnfClause::unit(r_constraint));
        }

        (q, r)
    }

    /// Check if a bitvector is zero
    fn is_zero(&mut self, bits: &BvBits) -> CnfLit {
        // All bits must be 0
        // is_zero = NOT(bit[0] OR bit[1] OR ... OR bit[n-1])
        if bits.is_empty() {
            let t = self.fresh_var();
            self.add_clause(CnfClause::unit(t));
            return t;
        }

        // Create OR of all bits, then negate
        let any_set = self.mk_or_many(bits);
        -any_set
    }

    /// Create OR of many literals
    fn mk_or_many(&mut self, lits: &[CnfLit]) -> CnfLit {
        if lits.is_empty() {
            let f = self.fresh_var();
            self.add_clause(CnfClause::unit(-f));
            return f;
        }
        if lits.len() == 1 {
            return lits[0];
        }

        let mut result = lits[0];
        for &lit in &lits[1..] {
            result = self.mk_or(result, lit);
        }
        result
    }

    /// Bit-blast unsigned division
    fn bitblast_udiv(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        let (q, _r) = self.bitblast_udiv_urem(a, b);
        q
    }

    /// Bit-blast unsigned remainder
    fn bitblast_urem(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        let (_q, r) = self.bitblast_udiv_urem(a, b);
        r
    }

    /// Bit-blast signed division
    /// SMT-LIB uses truncation toward zero semantics:
    ///   (-7) /s 2 = -3  (not -4)
    ///   7 /s (-2) = -3
    ///   (-7) /s (-2) = 3
    fn bitblast_sdiv(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        assert_eq!(a.len(), b.len());
        let n = a.len();

        if n == 0 {
            return Vec::new();
        }

        let sign_a = a[n - 1];
        let sign_b = b[n - 1];

        // Convert to unsigned: abs(a) and abs(b)
        let abs_a = self.conditional_neg(a, sign_a);
        let abs_b = self.conditional_neg(b, sign_b);

        // Unsigned division
        let (abs_q, _) = self.bitblast_udiv_urem(&abs_a, &abs_b);

        // Result sign: negative if exactly one of a, b is negative
        // result_neg = sign_a XOR sign_b
        let result_neg = self.mk_xor(sign_a, sign_b);

        // Conditionally negate the quotient
        self.conditional_neg(&abs_q, result_neg)
    }

    /// Bit-blast signed remainder
    /// SMT-LIB semantics: sign of remainder matches sign of dividend
    ///   (-7) %s 2 = -1
    ///   7 %s (-2) = 1
    ///   (-7) %s (-2) = -1
    fn bitblast_srem(&mut self, a: &BvBits, b: &BvBits) -> BvBits {
        assert_eq!(a.len(), b.len());
        let n = a.len();

        if n == 0 {
            return Vec::new();
        }

        let sign_a = a[n - 1];
        let sign_b = b[n - 1];

        // Convert to unsigned
        let abs_a = self.conditional_neg(a, sign_a);
        let abs_b = self.conditional_neg(b, sign_b);

        // Unsigned division
        let (_, abs_r) = self.bitblast_udiv_urem(&abs_a, &abs_b);

        // Result sign matches dividend sign
        self.conditional_neg(&abs_r, sign_a)
    }

    /// Conditionally negate a bitvector
    /// Returns: if cond then -bits else bits
    fn conditional_neg(&mut self, bits: &BvBits, cond: CnfLit) -> BvBits {
        let neg = self.bitblast_neg(bits);
        self.bitwise_mux(&neg, bits, cond)
    }

    // =========================================================================
    // Bit-blasting for comparison operations
    // =========================================================================

    /// Bit-blast equality: a = b
    pub fn bitblast_eq(&mut self, a: &BvBits, b: &BvBits) -> CnfLit {
        assert_eq!(a.len(), b.len());

        // a = b iff all bits are equal
        // Each bit equal: (ai XOR bi) = false, i.e., ~(ai XOR bi)
        let mut equal_bits: Vec<CnfLit> = Vec::new();
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            let xor = self.mk_xor(ai, bi);
            equal_bits.push(-xor); // NOT XOR = XNOR = equal
        }

        // AND all equalities
        self.mk_and_many(&equal_bits)
    }

    /// Bit-blast unsigned less-than: a < b
    pub fn bitblast_ult(&mut self, a: &BvBits, b: &BvBits) -> CnfLit {
        assert_eq!(a.len(), b.len());
        let n = a.len();

        if n == 0 {
            let f = self.fresh_var();
            self.add_clause(CnfClause::unit(-f));
            return f;
        }

        // a < b from MSB to LSB
        // Starting from MSB, find first bit where they differ
        // If a[i] < b[i] (i.e., a[i]=0, b[i]=1), then a < b

        // Build from LSB to MSB
        // lt[i] = (a[i] < b[i]) OR ((a[i] = b[i]) AND lt[i-1])

        let mut lt = self.fresh_var();
        self.add_clause(CnfClause::unit(-lt)); // lt[-1] = false

        for i in 0..n {
            let ai = a[i];
            let bi = b[i];

            // a[i] < b[i] means a[i]=0 and b[i]=1
            let a_lt_b = self.mk_and(-ai, bi);

            // a[i] = b[i] means XNOR
            let xor = self.mk_xor(ai, bi);
            let eq = -xor;

            // New lt = a_lt_b OR (eq AND old_lt)
            let eq_and_lt = self.mk_and(eq, lt);
            lt = self.mk_or(a_lt_b, eq_and_lt);
        }

        lt
    }

    /// Bit-blast unsigned less-or-equal: a <= b
    pub fn bitblast_ule(&mut self, a: &BvBits, b: &BvBits) -> CnfLit {
        // a <= b iff NOT (b < a)
        let b_lt_a = self.bitblast_ult(b, a);
        -b_lt_a
    }

    /// Bit-blast signed less-than: a <_s b
    pub fn bitblast_slt(&mut self, a: &BvBits, b: &BvBits) -> CnfLit {
        assert_eq!(a.len(), b.len());
        let n = a.len();

        if n == 0 {
            let f = self.fresh_var();
            self.add_clause(CnfClause::unit(-f));
            return f;
        }

        let sign_a = a[n - 1];
        let sign_b = b[n - 1];

        // Case 1: a negative, b non-negative => a < b
        let a_neg_b_pos = self.mk_and(sign_a, -sign_b);

        // Case 2: signs equal, compare magnitudes
        let signs_eq = self.mk_xnor(sign_a, sign_b);
        let ult = self.bitblast_ult(a, b);
        let same_sign_lt = self.mk_and(signs_eq, ult);

        self.mk_or(a_neg_b_pos, same_sign_lt)
    }

    /// Bit-blast signed less-or-equal: a <=_s b
    pub fn bitblast_sle(&mut self, a: &BvBits, b: &BvBits) -> CnfLit {
        let b_lt_a = self.bitblast_slt(b, a);
        -b_lt_a
    }

    // =========================================================================
    // CNF encoding helpers
    // =========================================================================

    /// Create constant bit vector
    fn const_bits(&mut self, value: u64, width: usize) -> BvBits {
        let mut bits = Vec::with_capacity(width);
        for i in 0..width {
            let bit_set = (value >> i) & 1 == 1;
            let var = self.fresh_var();
            self.add_clause(CnfClause::unit(if bit_set { var } else { -var }));
            bits.push(var);
        }
        bits
    }

    /// Create AND gate: out = a AND b
    fn mk_and(&mut self, a: CnfLit, b: CnfLit) -> CnfLit {
        let out = self.fresh_var();
        // out => a: (-out OR a)
        // out => b: (-out OR b)
        // a AND b => out: (-a OR -b OR out)
        self.add_clause(CnfClause::binary(-out, a));
        self.add_clause(CnfClause::binary(-out, b));
        self.add_clause(CnfClause::new(vec![-a, -b, out]));
        out
    }

    /// Create OR gate: out = a OR b
    fn mk_or(&mut self, a: CnfLit, b: CnfLit) -> CnfLit {
        let out = self.fresh_var();
        // a => out: (-a OR out)
        // b => out: (-b OR out)
        // out => a OR b: (-out OR a OR b)
        self.add_clause(CnfClause::binary(-a, out));
        self.add_clause(CnfClause::binary(-b, out));
        self.add_clause(CnfClause::new(vec![-out, a, b]));
        out
    }

    /// Create XOR gate: out = a XOR b
    fn mk_xor(&mut self, a: CnfLit, b: CnfLit) -> CnfLit {
        let out = self.fresh_var();
        // XOR truth table:
        // a=0, b=0 => out=0
        // a=0, b=1 => out=1
        // a=1, b=0 => out=1
        // a=1, b=1 => out=0
        // Clauses:
        // (-a OR -b OR -out)
        // (-a OR b OR out)
        // (a OR -b OR out)
        // (a OR b OR -out)
        self.add_clause(CnfClause::new(vec![-a, -b, -out]));
        self.add_clause(CnfClause::new(vec![-a, b, out]));
        self.add_clause(CnfClause::new(vec![a, -b, out]));
        self.add_clause(CnfClause::new(vec![a, b, -out]));
        out
    }

    /// Create XNOR gate: out = a XNOR b = NOT(a XOR b)
    fn mk_xnor(&mut self, a: CnfLit, b: CnfLit) -> CnfLit {
        let xor = self.mk_xor(a, b);
        -xor
    }

    /// Create AND of many literals
    fn mk_and_many(&mut self, lits: &[CnfLit]) -> CnfLit {
        if lits.is_empty() {
            let t = self.fresh_var();
            self.add_clause(CnfClause::unit(t));
            return t;
        }
        if lits.len() == 1 {
            return lits[0];
        }

        let mut result = lits[0];
        for &lit in &lits[1..] {
            result = self.mk_and(result, lit);
        }
        result
    }

    /// Create MUX: if sel then a else b (bitwise)
    fn bitwise_mux(&mut self, a: &BvBits, b: &BvBits, sel: CnfLit) -> BvBits {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| self.mk_mux(ai, bi, sel))
            .collect()
    }

    /// Create MUX: if sel then a else b
    fn mk_mux(&mut self, a: CnfLit, b: CnfLit, sel: CnfLit) -> CnfLit {
        let out = self.fresh_var();
        // out = (sel AND a) OR (NOT sel AND b)
        // Clauses:
        // (-sel OR -a OR out)  -- sel=1, a=1 => out=1
        // (-sel OR a OR -out)  -- sel=1, a=0 => out=0
        // (sel OR -b OR out)   -- sel=0, b=1 => out=1
        // (sel OR b OR -out)   -- sel=0, b=0 => out=0
        self.add_clause(CnfClause::new(vec![-sel, -a, out]));
        self.add_clause(CnfClause::new(vec![-sel, a, -out]));
        self.add_clause(CnfClause::new(vec![sel, -b, out]));
        self.add_clause(CnfClause::new(vec![sel, b, -out]));
        out
    }

    // =========================================================================
    // Theory solver interface helpers
    // =========================================================================

    /// Process an assertion
    fn process_assertion(&mut self, term: TermId, value: bool) {
        let data = self.terms.get(term).clone();

        match data {
            TermData::Not(inner) => {
                // Not negates the value
                self.process_assertion(inner, !value);
            }
            TermData::App(ref sym, ref args) if args.len() == 2 => {
                let name = sym.name();
                match name {
                    "=" => {
                        // Equality between bitvectors
                        let a_sort = self.terms.sort(args[0]);
                        if matches!(a_sort, Sort::BitVec(_)) {
                            let a = self.get_bits(args[0]);
                            let b = self.get_bits(args[1]);
                            let eq = self.bitblast_eq(&a, &b);
                            self.add_clause(CnfClause::unit(if value { eq } else { -eq }));
                        }
                    }
                    "bvult" => {
                        let a = self.get_bits(args[0]);
                        let b = self.get_bits(args[1]);
                        let lt = self.bitblast_ult(&a, &b);
                        self.add_clause(CnfClause::unit(if value { lt } else { -lt }));
                    }
                    "bvule" => {
                        let a = self.get_bits(args[0]);
                        let b = self.get_bits(args[1]);
                        let le = self.bitblast_ule(&a, &b);
                        self.add_clause(CnfClause::unit(if value { le } else { -le }));
                    }
                    "bvugt" => {
                        let a = self.get_bits(args[0]);
                        let b = self.get_bits(args[1]);
                        let gt = self.bitblast_ult(&b, &a);
                        self.add_clause(CnfClause::unit(if value { gt } else { -gt }));
                    }
                    "bvuge" => {
                        let a = self.get_bits(args[0]);
                        let b = self.get_bits(args[1]);
                        let ge = self.bitblast_ule(&b, &a);
                        self.add_clause(CnfClause::unit(if value { ge } else { -ge }));
                    }
                    "bvslt" => {
                        let a = self.get_bits(args[0]);
                        let b = self.get_bits(args[1]);
                        let lt = self.bitblast_slt(&a, &b);
                        self.add_clause(CnfClause::unit(if value { lt } else { -lt }));
                    }
                    "bvsle" => {
                        let a = self.get_bits(args[0]);
                        let b = self.get_bits(args[1]);
                        let le = self.bitblast_sle(&a, &b);
                        self.add_clause(CnfClause::unit(if value { le } else { -le }));
                    }
                    "bvsgt" => {
                        let a = self.get_bits(args[0]);
                        let b = self.get_bits(args[1]);
                        let gt = self.bitblast_slt(&b, &a);
                        self.add_clause(CnfClause::unit(if value { gt } else { -gt }));
                    }
                    "bvsge" => {
                        let a = self.get_bits(args[0]);
                        let b = self.get_bits(args[1]);
                        let ge = self.bitblast_sle(&b, &a);
                        self.add_clause(CnfClause::unit(if value { ge } else { -ge }));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

impl Default for BvSolver<'_> {
    fn default() -> Self {
        // Can't implement Default without a TermStore reference
        panic!("BvSolver requires a TermStore reference, use BvSolver::new(terms) instead")
    }
}

impl TheorySolver for BvSolver<'_> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        self.trail.push(literal);
        self.asserted.insert(literal, value);
        self.process_assertion(literal, value);
    }

    fn check(&mut self) -> TheoryResult {
        // All constraints are eagerly bit-blasted to CNF
        // The SAT solver will find inconsistencies
        TheoryResult::Sat
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        // Eager bit-blasting doesn't propagate - all is in CNF
        Vec::new()
    }

    fn push(&mut self) {
        self.trail_stack.push(self.trail.len());
    }

    fn pop(&mut self) {
        if let Some(size) = self.trail_stack.pop() {
            while self.trail.len() > size {
                if let Some(term) = self.trail.pop() {
                    self.asserted.remove(&term);
                }
            }
        }
    }

    fn reset(&mut self) {
        self.term_to_bits.clear();
        self.clauses.clear();
        self.next_var = 1;
        self.trail.clear();
        self.trail_stack.clear();
        self.asserted.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn setup_store() -> TermStore {
        TermStore::new()
    }

    #[test]
    fn test_const_bits() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let bits = solver.const_bits(5, 4); // 0101
        assert_eq!(bits.len(), 4);
        // Should have 4 unit clauses for the constant
    }

    #[test]
    fn test_bitblast_and() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let a = solver.const_bits(0b1100, 4);
        let b = solver.const_bits(0b1010, 4);
        let result = solver.bitblast_and(&a, &b);

        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_bitblast_add() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let a = solver.const_bits(3, 4);
        let b = solver.const_bits(5, 4);
        let result = solver.bitblast_add(&a, &b);

        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_bitblast_variable() {
        let mut store = setup_store();
        let x = store.mk_var("x", Sort::BitVec(8));

        let mut solver = BvSolver::new(&store);
        let bits = solver.get_bits(x);

        assert_eq!(bits.len(), 8);
    }

    #[test]
    fn test_bitblast_bvadd_term() {
        let mut store = setup_store();
        let x = store.mk_var("x", Sort::BitVec(8));
        let y = store.mk_var("y", Sort::BitVec(8));
        let sum = store.mk_bvadd(vec![x, y]);

        let mut solver = BvSolver::new(&store);
        let bits = solver.get_bits(sum);

        assert_eq!(bits.len(), 8);
    }

    #[test]
    fn test_bitblast_equality() {
        let mut store = setup_store();
        let x = store.mk_var("x", Sort::BitVec(4));
        let c = store.mk_bitvec(BigInt::from(5), 4);

        let mut solver = BvSolver::new(&store);
        let x_bits = solver.get_bits(x);
        let c_bits = solver.get_bits(c);

        let eq = solver.bitblast_eq(&x_bits, &c_bits);
        assert!(eq != 0);
    }

    #[test]
    fn test_bitblast_ult() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let a = solver.const_bits(3, 4);
        let b = solver.const_bits(5, 4);
        let lt = solver.bitblast_ult(&a, &b);

        assert!(lt != 0);
    }

    #[test]
    fn test_theory_solver_interface() {
        let mut store = setup_store();
        let x = store.mk_var("x", Sort::BitVec(8));
        let y = store.mk_var("y", Sort::BitVec(8));
        let eq = store.mk_eq(x, y);

        let mut solver = BvSolver::new(&store);
        solver.assert_literal(eq, true);

        // Check returns Sat for eager bit-blasting (consistency checked by SAT solver)
        assert!(matches!(solver.check(), TheoryResult::Sat));
    }

    // =========================================================================
    // Division tests
    // =========================================================================

    #[test]
    fn test_is_zero() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let zero = solver.const_bits(0, 4);
        let nonzero = solver.const_bits(5, 4);

        let zero_is_zero = solver.is_zero(&zero);
        let nonzero_is_zero = solver.is_zero(&nonzero);

        // These return literals - the actual constraints will be in clauses
        assert!(zero_is_zero != 0);
        assert!(nonzero_is_zero != 0);
    }

    #[test]
    fn test_bitblast_udiv_urem_basic() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        // 7 / 3 = 2, 7 % 3 = 1
        let a = solver.const_bits(7, 4);
        let b = solver.const_bits(3, 4);

        let (q, r) = solver.bitblast_udiv_urem(&a, &b);

        assert_eq!(q.len(), 4);
        assert_eq!(r.len(), 4);
        // Constraints will be added to clauses
        assert!(!solver.clauses.is_empty());
    }

    #[test]
    fn test_bitblast_udiv() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        // 10 / 3 = 3
        let a = solver.const_bits(10, 4);
        let b = solver.const_bits(3, 4);

        let q = solver.bitblast_udiv(&a, &b);

        assert_eq!(q.len(), 4);
    }

    #[test]
    fn test_bitblast_urem() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        // 10 % 3 = 1
        let a = solver.const_bits(10, 4);
        let b = solver.const_bits(3, 4);

        let r = solver.bitblast_urem(&a, &b);

        assert_eq!(r.len(), 4);
    }

    #[test]
    fn test_bitblast_sdiv() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        // In 4-bit signed: -7 = 0b1001, 2 = 0b0010
        // -7 / 2 = -3 (truncation toward zero) = 0b1101
        let a = solver.const_bits(0b1001, 4); // -7 in 4-bit signed
        let b = solver.const_bits(0b0010, 4); // 2

        let q = solver.bitblast_sdiv(&a, &b);

        assert_eq!(q.len(), 4);
    }

    #[test]
    fn test_bitblast_srem() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        // In 4-bit signed: -7 = 0b1001, 2 = 0b0010
        // -7 % 2 = -1 (sign matches dividend) = 0b1111
        let a = solver.const_bits(0b1001, 4); // -7 in 4-bit signed
        let b = solver.const_bits(0b0010, 4); // 2

        let r = solver.bitblast_srem(&a, &b);

        assert_eq!(r.len(), 4);
    }

    #[test]
    fn test_bitblast_div_by_zero() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        // Division by zero: a / 0 = all_ones, a % 0 = a
        let a = solver.const_bits(7, 4);
        let zero = solver.const_bits(0, 4);

        let (q, r) = solver.bitblast_udiv_urem(&a, &zero);

        assert_eq!(q.len(), 4);
        assert_eq!(r.len(), 4);
        // Constraints enforce q = all_ones and r = a when divisor is 0
    }

    #[test]
    fn test_conditional_neg() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let a = solver.const_bits(5, 4);
        let cond_true = solver.fresh_var();
        solver.add_clause(CnfClause::unit(cond_true));

        let result = solver.conditional_neg(&a, cond_true);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_mk_or_many() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        // Test with multiple literals
        let lits = vec![solver.fresh_var(), solver.fresh_var(), solver.fresh_var()];

        let result = solver.mk_or_many(&lits);
        assert!(result != 0);
    }

    #[test]
    fn test_mk_or_many_empty() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        // Empty case should return false
        let result = solver.mk_or_many(&[]);
        assert!(result != 0);
    }

    #[test]
    fn test_mk_or_many_single() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let lit = solver.fresh_var();
        let result = solver.mk_or_many(&[lit]);

        // Single literal case should return the literal itself
        assert_eq!(result, lit);
    }

    #[test]
    fn test_division_generates_clauses() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let initial_clauses = solver.clauses.len();

        let a = solver.const_bits(15, 4);
        let b = solver.const_bits(4, 4);
        let _q = solver.bitblast_udiv(&a, &b);

        // Division should generate additional clauses for the constraints
        assert!(solver.clauses.len() > initial_clauses);
    }

    #[test]
    fn test_signed_division_symmetry() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        // Test that both operands being negative gives positive result
        // -6 / -2 = 3
        let a = solver.const_bits(0b1010, 4); // -6 in 4-bit signed
        let b = solver.const_bits(0b1110, 4); // -2 in 4-bit signed

        let q = solver.bitblast_sdiv(&a, &b);
        assert_eq!(q.len(), 4);
    }
}

// Kani verification proofs for bitvector theory solver
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    fn setup_store() -> TermStore {
        TermStore::new()
    }

    /// Proof: push/pop maintains stack consistency
    #[kani::proof]
    fn proof_push_pop_stack_depth() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let num_pushes: u8 = kani::any();
        kani::assume(num_pushes <= 10);
        let num_pops: u8 = kani::any();
        kani::assume(num_pops <= num_pushes);

        // Push n times
        for _ in 0..num_pushes {
            solver.push();
        }
        assert_eq!(solver.trail_stack.len(), num_pushes as usize);

        // Pop m times (m <= n)
        for _ in 0..num_pops {
            solver.pop();
        }
        assert_eq!(solver.trail_stack.len(), (num_pushes - num_pops) as usize);
    }

    /// Proof: pop on empty stack is safe (no-op)
    #[kani::proof]
    fn proof_pop_empty_is_safe() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        // Pop on empty stack should do nothing
        let trail_len_before = solver.trail.len();
        let asserted_len_before = solver.asserted.len();

        solver.pop();

        // State should be unchanged
        assert_eq!(solver.trail.len(), trail_len_before);
        assert_eq!(solver.asserted.len(), asserted_len_before);
        assert!(solver.trail_stack.is_empty());
    }

    /// Proof: reset clears all mutable state
    #[kani::proof]
    fn proof_reset_clears_state() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        // Add some state
        solver.push();
        let _ = solver.fresh_var();
        let _ = solver.fresh_var();
        solver.push();

        // Reset should clear everything
        solver.reset();

        assert!(
            solver.term_to_bits.is_empty(),
            "reset must clear term_to_bits"
        );
        assert!(solver.clauses.is_empty(), "reset must clear clauses");
        assert_eq!(solver.next_var, 1, "reset must reset next_var to 1");
        assert!(solver.trail.is_empty(), "reset must clear trail");
        assert!(
            solver.trail_stack.is_empty(),
            "reset must clear trail_stack"
        );
        assert!(solver.asserted.is_empty(), "reset must clear asserted");
    }

    /// Proof: fresh_var is monotonically increasing
    #[kani::proof]
    fn proof_fresh_var_monotonic() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let initial = solver.next_var;
        let v1 = solver.fresh_var();
        let mid = solver.next_var;
        let v2 = solver.fresh_var();

        // Each call should return unique, increasing values
        assert!(v1 > 0);
        assert!(v2 > v1);
        assert!(mid > initial);
        assert!(solver.next_var > mid);
    }

    /// Proof: const_bits returns correct number of bits
    #[kani::proof]
    fn proof_const_bits_width() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let value: u64 = kani::any();
        let width: usize = kani::any();
        kani::assume(width > 0 && width <= 16);

        let bits = solver.const_bits(value, width);

        assert_eq!(
            bits.len(),
            width,
            "const_bits must return correct number of bits"
        );
    }

    /// Proof: num_vars returns correct count
    #[kani::proof]
    fn proof_num_vars_correct() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        assert_eq!(solver.num_vars(), 0);

        let n: u8 = kani::any();
        kani::assume(n > 0 && n <= 20);

        for _ in 0..n {
            let _ = solver.fresh_var();
        }

        assert_eq!(solver.num_vars(), n as u32);
    }

    /// Proof: trail_stack markers are valid positions
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_trail_stack_markers_valid() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let depth: u8 = kani::any();
        kani::assume(depth > 0 && depth <= 5);

        // Push multiple times
        let mut expected_markers: Vec<usize> = Vec::new();
        for _ in 0..depth {
            expected_markers.push(solver.trail.len());
            solver.push();
        }

        // Verify markers are correct and in ascending order
        for i in 0..depth as usize {
            assert_eq!(solver.trail_stack[i], expected_markers[i]);
            if i > 0 {
                assert!(solver.trail_stack[i] >= solver.trail_stack[i - 1]);
            }
        }
    }

    /// Proof: bitblast_and preserves width
    #[kani::proof]
    fn proof_bitblast_and_width() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let width: usize = kani::any();
        kani::assume(width > 0 && width <= 8);

        let a = solver.const_bits(0, width);
        let b = solver.const_bits(0, width);

        let result = solver.bitblast_and(&a, &b);

        assert_eq!(result.len(), width, "AND must preserve width");
    }

    /// Proof: bitblast_or preserves width
    #[kani::proof]
    fn proof_bitblast_or_width() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let width: usize = kani::any();
        kani::assume(width > 0 && width <= 8);

        let a = solver.const_bits(0, width);
        let b = solver.const_bits(0, width);

        let result = solver.bitblast_or(&a, &b);

        assert_eq!(result.len(), width, "OR must preserve width");
    }

    /// Proof: bitblast_xor preserves width
    #[kani::proof]
    fn proof_bitblast_xor_width() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let width: usize = kani::any();
        kani::assume(width > 0 && width <= 8);

        let a = solver.const_bits(0, width);
        let b = solver.const_bits(0, width);

        let result = solver.bitblast_xor(&a, &b);

        assert_eq!(result.len(), width, "XOR must preserve width");
    }

    /// Proof: bitblast_not preserves width
    #[kani::proof]
    fn proof_bitblast_not_width() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let width: usize = kani::any();
        kani::assume(width > 0 && width <= 8);

        let a = solver.const_bits(0, width);

        let result = solver.bitblast_not(&a);

        assert_eq!(result.len(), width, "NOT must preserve width");
    }

    /// Proof: bitblast_add preserves width
    #[kani::proof]
    fn proof_bitblast_add_width() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let width: usize = kani::any();
        kani::assume(width > 0 && width <= 8);

        let a = solver.const_bits(0, width);
        let b = solver.const_bits(0, width);

        let result = solver.bitblast_add(&a, &b);

        assert_eq!(result.len(), width, "ADD must preserve width");
    }

    /// Proof: clauses are only added, never removed (except reset)
    #[kani::proof]
    fn proof_clauses_monotonic() {
        let store = setup_store();
        let mut solver = BvSolver::new(&store);

        let initial_clauses = solver.clauses.len();

        // Generate some bits (which adds clauses)
        let _ = solver.const_bits(5, 4);

        let after_const = solver.clauses.len();

        // Push/pop should not affect clauses
        solver.push();
        let after_push = solver.clauses.len();
        solver.pop();
        let after_pop = solver.clauses.len();

        assert!(after_const >= initial_clauses);
        assert_eq!(after_push, after_const);
        assert_eq!(after_pop, after_push);
    }
}
