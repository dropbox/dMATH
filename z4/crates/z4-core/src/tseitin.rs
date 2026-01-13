//! Tseitin transformation: convert Boolean formulas to CNF
//!
//! The Tseitin transformation converts an arbitrary Boolean formula into
//! Conjunctive Normal Form (CNF) by introducing fresh variables for subterms.
//! The resulting CNF is equisatisfiable with the original formula.
//!
//! # Example
//!
//! For a formula `(a ∧ b) ∨ c`:
//! 1. Introduce variable `t1` for `(a ∧ b)`
//! 2. Add clauses: `t1 → a`, `t1 → b`, `(a ∧ b) → t1`
//! 3. Introduce variable `t2` for `t1 ∨ c`
//! 4. Add clauses: `¬t1 → t2`, `¬c → t2`, `t2 → (t1 ∨ c)`
//! 5. Assert `t2`
//!
//! The output uses DIMACS-style signed integer literals where:
//! - Positive integer N means variable N is true
//! - Negative integer -N means variable N is false

use crate::term::{Constant, Symbol, TermData, TermId, TermStore};
use crate::Sort;
use std::collections::BTreeMap;

/// A CNF literal (signed integer, DIMACS-style)
/// Positive values represent the variable being true,
/// negative values represent the variable being false.
/// Variable numbering starts at 1.
pub type CnfLit = i32;

/// A CNF clause (disjunction of literals)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CnfClause(pub Vec<CnfLit>);

impl CnfClause {
    /// Create a new clause from literals
    pub fn new(literals: Vec<CnfLit>) -> Self {
        CnfClause(literals)
    }

    /// Create a unit clause
    pub fn unit(lit: CnfLit) -> Self {
        CnfClause(vec![lit])
    }

    /// Create a binary clause
    pub fn binary(a: CnfLit, b: CnfLit) -> Self {
        CnfClause(vec![a, b])
    }

    /// Create a ternary clause
    pub fn ternary(a: CnfLit, b: CnfLit, c: CnfLit) -> Self {
        CnfClause(vec![a, b, c])
    }

    /// Check if the clause is empty (conflict)
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the literals
    pub fn literals(&self) -> &[CnfLit] {
        &self.0
    }
}

/// Result of Tseitin transformation
#[derive(Debug)]
pub struct TseitinResult {
    /// The CNF clauses
    pub clauses: Vec<CnfClause>,
    /// Mapping from term IDs to CNF variables (1-indexed)
    pub term_to_var: BTreeMap<TermId, u32>,
    /// Mapping from CNF variables to term IDs
    pub var_to_term: BTreeMap<u32, TermId>,
    /// The root literal (represents the whole formula)
    pub root: CnfLit,
    /// Number of variables used (max variable index)
    pub num_vars: u32,
}

impl TseitinResult {
    /// Convert a CNF variable to a TermId if it exists
    pub fn term_for_var(&self, var: u32) -> Option<TermId> {
        self.var_to_term.get(&var).copied()
    }

    /// Convert a TermId to a CNF variable if it exists
    pub fn var_for_term(&self, term: TermId) -> Option<u32> {
        self.term_to_var.get(&term).copied()
    }
}

/// Saved Tseitin state for incremental use
///
/// This stores the internal state of a Tseitin transformer, allowing
/// it to be saved and restored across multiple transformation calls
/// while sharing term-to-variable mappings.
#[derive(Debug, Clone, Default)]
pub struct TseitinState {
    /// Mapping from term IDs to CNF variables
    pub term_to_var: BTreeMap<TermId, u32>,
    /// Mapping from CNF variables to term IDs
    pub var_to_term: BTreeMap<u32, TermId>,
    /// Next variable index (1-indexed, DIMACS style)
    pub next_var: u32,
    /// Cache for polarity-aware encoding
    pub encoded: BTreeMap<(TermId, bool), CnfLit>,
}

impl TseitinState {
    /// Create a new empty state
    pub fn new() -> Self {
        TseitinState {
            term_to_var: BTreeMap::new(),
            var_to_term: BTreeMap::new(),
            next_var: 1, // DIMACS variables are 1-indexed
            encoded: BTreeMap::new(),
        }
    }
}

/// Tseitin transformation context
pub struct Tseitin<'a> {
    /// The term store
    terms: &'a TermStore,
    /// Mapping from term IDs to CNF variables
    term_to_var: BTreeMap<TermId, u32>,
    /// Mapping from CNF variables to term IDs
    var_to_term: BTreeMap<u32, TermId>,
    /// Generated clauses
    clauses: Vec<CnfClause>,
    /// Next variable index (1-indexed, DIMACS style)
    next_var: u32,
    /// Cache for polarity-aware encoding
    encoded: BTreeMap<(TermId, bool), CnfLit>,
    /// Index of clauses that have been extracted (for incremental use)
    clauses_extracted: usize,
}

impl<'a> Tseitin<'a> {
    /// Create a new Tseitin transformation context
    pub fn new(terms: &'a TermStore) -> Self {
        Tseitin {
            terms,
            term_to_var: BTreeMap::new(),
            var_to_term: BTreeMap::new(),
            clauses: Vec::new(),
            next_var: 1, // DIMACS variables are 1-indexed
            encoded: BTreeMap::new(),
            clauses_extracted: 0,
        }
    }

    /// Create a Tseitin context from saved state
    ///
    /// This allows continuing from a previous transformation session,
    /// preserving term-to-variable mappings for incremental solving.
    pub fn from_state(terms: &'a TermStore, state: TseitinState) -> Self {
        Tseitin {
            terms,
            term_to_var: state.term_to_var,
            var_to_term: state.var_to_term,
            clauses: Vec::new(), // Start with empty clauses
            next_var: state.next_var,
            encoded: state.encoded,
            clauses_extracted: 0,
        }
    }

    /// Save the current state for later restoration
    ///
    /// This extracts the term-to-variable mappings and variable counter
    /// so that a new Tseitin instance can be created later that continues
    /// from this point.
    pub fn save_state(&self) -> TseitinState {
        TseitinState {
            term_to_var: self.term_to_var.clone(),
            var_to_term: self.var_to_term.clone(),
            next_var: self.next_var,
            encoded: self.encoded.clone(),
        }
    }

    /// Take ownership of state (more efficient than clone)
    pub fn into_state(self) -> TseitinState {
        TseitinState {
            term_to_var: self.term_to_var,
            var_to_term: self.var_to_term,
            next_var: self.next_var,
            encoded: self.encoded,
        }
    }

    /// Allocate a fresh CNF variable
    fn fresh_var(&mut self) -> u32 {
        let var = self.next_var;
        self.next_var += 1;
        var
    }

    /// Get or create a CNF variable for a term
    fn get_var(&mut self, term_id: TermId) -> u32 {
        if let Some(&var) = self.term_to_var.get(&term_id) {
            return var;
        }
        let var = self.fresh_var();
        self.term_to_var.insert(term_id, var);
        self.var_to_term.insert(var, term_id);
        var
    }

    /// Get a literal for a term (positive or negative based on value)
    fn get_literal(&mut self, term_id: TermId, positive: bool) -> CnfLit {
        let var = self.get_var(term_id) as i32;
        if positive {
            var
        } else {
            -var
        }
    }

    /// Negate a literal
    fn negate(lit: CnfLit) -> CnfLit {
        -lit
    }

    /// Add a clause
    fn add_clause(&mut self, clause: CnfClause) {
        if !clause.is_empty() {
            self.clauses.push(clause);
        }
    }

    /// Encode a term to CNF
    ///
    /// Returns a literal representing the term.
    /// The `positive` parameter indicates the polarity context:
    /// - true: the term appears positively (we need it to be true)
    /// - false: the term appears negatively (we need it to be false)
    pub fn encode(&mut self, term_id: TermId, positive: bool) -> CnfLit {
        // Check cache
        if let Some(&lit) = self.encoded.get(&(term_id, positive)) {
            return lit;
        }

        let result = self.encode_inner(term_id, positive);

        // Cache the result
        self.encoded.insert((term_id, positive), result);

        result
    }

    fn encode_inner(&mut self, term_id: TermId, positive: bool) -> CnfLit {
        match self.terms.get(term_id) {
            TermData::Const(Constant::Bool(true)) => {
                // True is represented by a fresh variable forced to true
                let var = self.fresh_var() as i32;
                self.add_clause(CnfClause::unit(var));
                if positive {
                    var
                } else {
                    -var
                }
            }
            TermData::Const(Constant::Bool(false)) => {
                // False is represented by a fresh variable forced to false
                let var = self.fresh_var() as i32;
                self.add_clause(CnfClause::unit(-var));
                if positive {
                    var
                } else {
                    -var
                }
            }
            TermData::Const(_) => {
                // Non-boolean constants shouldn't appear at Boolean positions
                // Create a variable for theory reasoning
                self.get_literal(term_id, positive)
            }
            TermData::Var(_, _) => {
                // Variables get direct CNF variables
                self.get_literal(term_id, positive)
            }
            TermData::Not(inner) => {
                // not(x) just flips the polarity
                self.encode(*inner, !positive)
            }
            TermData::Ite(cond, then_term, else_term) => {
                self.encode_ite(term_id, *cond, *then_term, *else_term, positive)
            }
            TermData::App(symbol, args) => {
                self.encode_app(term_id, symbol.clone(), args.clone(), positive)
            }
            TermData::Let(_, _) => {
                // Let bindings should be expanded before Tseitin
                panic!("Let binding in Tseitin transformation - should be expanded first");
            }
        }
    }

    fn encode_ite(
        &mut self,
        term_id: TermId,
        cond: TermId,
        then_term: TermId,
        else_term: TermId,
        positive: bool,
    ) -> CnfLit {
        // Polarity-aware ITE encoding (Plaisted–Greenbaum style).
        //
        // Full equivalence for v ↔ ite(c, t, e) can be encoded as 4 clauses:
        // - v -> ite: (¬v ∨ ¬c ∨ t) ∧ (¬v ∨ c ∨ e)
        // - ite -> v: (¬c ∨ ¬t ∨ v) ∧ (c ∨ ¬e ∨ v)
        //
        // But for Tseitin we only need the direction required by the context polarity:
        // - If the term appears positively (we need v=true), constrain v -> ite.
        // - If the term appears negatively (we need v=false), constrain ite -> v.
        //
        // If both polarities occur in the formula, we'll visit both and emit both halves.

        let v = self.get_var(term_id) as i32;

        let c_lit = self.encode(cond, true);
        let t_lit = self.encode(then_term, true);
        let e_lit = self.encode(else_term, true);

        if positive {
            // v -> ite(c, t, e)
            // (¬v ∨ ¬c ∨ t)
            self.add_clause(CnfClause::ternary(Self::negate(c_lit), -v, t_lit));
            // (¬v ∨ c ∨ e)
            self.add_clause(CnfClause::ternary(c_lit, -v, e_lit));
        } else {
            // ite(c, t, e) -> v
            // (¬c ∨ ¬t ∨ v)
            self.add_clause(CnfClause::ternary(
                Self::negate(c_lit),
                Self::negate(t_lit),
                v,
            ));
            // (c ∨ ¬e ∨ v)
            self.add_clause(CnfClause::ternary(c_lit, Self::negate(e_lit), v));
        }

        if positive {
            v
        } else {
            -v
        }
    }

    fn encode_app(
        &mut self,
        term_id: TermId,
        symbol: Symbol,
        args: Vec<TermId>,
        positive: bool,
    ) -> CnfLit {
        match symbol.name() {
            "and" => self.encode_and(term_id, &args, positive),
            "or" => self.encode_or(term_id, &args, positive),
            "xor" => {
                if args.len() == 2 {
                    self.encode_xor(term_id, args[0], args[1], positive)
                } else {
                    // Invalid xor - encoded as variable
                    self.get_literal(term_id, positive)
                }
            }
            "=" => {
                if args.len() == 2 {
                    self.encode_eq(term_id, args[0], args[1], positive)
                } else {
                    // Multi-way equality - encoded as variable
                    self.get_literal(term_id, positive)
                }
            }
            "distinct" => {
                // For distinct, just create a variable for theory reasoning
                self.get_literal(term_id, positive)
            }
            _ => {
                // Theory predicates - create a variable
                self.get_literal(term_id, positive)
            }
        }
    }

    fn encode_and(&mut self, term_id: TermId, args: &[TermId], positive: bool) -> CnfLit {
        if args.is_empty() {
            // and() = true
            let var = self.fresh_var() as i32;
            self.add_clause(CnfClause::unit(var));
            return if positive { var } else { -var };
        }

        if args.len() == 1 {
            return self.encode(args[0], positive);
        }

        // v ↔ (a1 ∧ a2 ∧ ... ∧ an)
        // v → (a1 ∧ a2 ∧ ... ∧ an): (¬v ∨ a1), (¬v ∨ a2), ..., (¬v ∨ an)
        // (a1 ∧ a2 ∧ ... ∧ an) → v: (¬a1 ∨ ¬a2 ∨ ... ∨ ¬an ∨ v)

        let v = self.get_var(term_id) as i32;

        let arg_lits: Vec<CnfLit> = args.iter().map(|&a| self.encode(a, true)).collect();

        // v → ai for each i
        for &a_lit in &arg_lits {
            self.add_clause(CnfClause::binary(-v, a_lit));
        }

        // (a1 ∧ ... ∧ an) → v
        let mut clause_lits: Vec<CnfLit> = arg_lits.iter().map(|&l| Self::negate(l)).collect();
        clause_lits.push(v);
        self.add_clause(CnfClause::new(clause_lits));

        if positive {
            v
        } else {
            -v
        }
    }

    fn encode_or(&mut self, term_id: TermId, args: &[TermId], positive: bool) -> CnfLit {
        if args.is_empty() {
            // or() = false
            let var = self.fresh_var() as i32;
            self.add_clause(CnfClause::unit(-var));
            return if positive { var } else { -var };
        }

        if args.len() == 1 {
            return self.encode(args[0], positive);
        }

        // v ↔ (a1 ∨ a2 ∨ ... ∨ an)
        // v → (a1 ∨ a2 ∨ ... ∨ an): (¬v ∨ a1 ∨ a2 ∨ ... ∨ an)
        // (a1 ∨ a2 ∨ ... ∨ an) → v: (¬a1 ∨ v), (¬a2 ∨ v), ..., (¬an ∨ v)

        let v = self.get_var(term_id) as i32;

        let arg_lits: Vec<CnfLit> = args.iter().map(|&a| self.encode(a, true)).collect();

        // v → (a1 ∨ ... ∨ an)
        let mut clause_lits = vec![-v];
        clause_lits.extend(arg_lits.iter().copied());
        self.add_clause(CnfClause::new(clause_lits));

        // ai → v for each i
        for &a_lit in &arg_lits {
            self.add_clause(CnfClause::binary(Self::negate(a_lit), v));
        }

        if positive {
            v
        } else {
            -v
        }
    }

    fn encode_eq(&mut self, term_id: TermId, lhs: TermId, rhs: TermId, positive: bool) -> CnfLit {
        // For Boolean equality: a = b ↔ (a ↔ b)
        // For non-Boolean: theory variable

        if matches!(self.terms.sort(lhs), Sort::Bool) {
            // Boolean equality: v ↔ (a ↔ b)
            // v → (a ↔ b): (¬v ∨ ¬a ∨ b) ∧ (¬v ∨ a ∨ ¬b)
            // (a ↔ b) → v: (a ∨ b ∨ v) ∧ (¬a ∨ ¬b ∨ v)

            let v = self.get_var(term_id) as i32;

            let a_lit = self.encode(lhs, true);
            let b_lit = self.encode(rhs, true);

            // (¬v ∨ ¬a ∨ b)
            self.add_clause(CnfClause::ternary(-v, Self::negate(a_lit), b_lit));
            // (¬v ∨ a ∨ ¬b)
            self.add_clause(CnfClause::ternary(-v, a_lit, Self::negate(b_lit)));
            // (a ∨ b ∨ v) - both false means v must be true
            self.add_clause(CnfClause::ternary(a_lit, b_lit, v));
            // (¬a ∨ ¬b ∨ v) - both true means v must be true
            self.add_clause(CnfClause::ternary(
                Self::negate(a_lit),
                Self::negate(b_lit),
                v,
            ));

            if positive {
                v
            } else {
                -v
            }
        } else {
            // Theory equality - create a variable
            self.get_literal(term_id, positive)
        }
    }

    fn encode_xor(&mut self, term_id: TermId, lhs: TermId, rhs: TermId, positive: bool) -> CnfLit {
        // XOR: v ↔ (a ⊕ b)
        // XOR is true when exactly one of a, b is true
        // v → (a ⊕ b): (¬v ∨ a ∨ b) ∧ (¬v ∨ ¬a ∨ ¬b)
        // (a ⊕ b) → v: (¬a ∨ b ∨ v) ∧ (a ∨ ¬b ∨ v)

        let v = self.get_var(term_id) as i32;

        let a_lit = self.encode(lhs, true);
        let b_lit = self.encode(rhs, true);

        // (¬v ∨ a ∨ b) - if v is true but both a,b are false, contradiction
        self.add_clause(CnfClause::ternary(-v, a_lit, b_lit));
        // (¬v ∨ ¬a ∨ ¬b) - if v is true but both a,b are true, contradiction
        self.add_clause(CnfClause::ternary(
            -v,
            Self::negate(a_lit),
            Self::negate(b_lit),
        ));
        // (¬a ∨ b ∨ v) - if a is true and b is false, v must be true
        self.add_clause(CnfClause::ternary(Self::negate(a_lit), b_lit, v));
        // (a ∨ ¬b ∨ v) - if a is false and b is true, v must be true
        self.add_clause(CnfClause::ternary(a_lit, Self::negate(b_lit), v));

        if positive {
            v
        } else {
            -v
        }
    }

    /// Transform a term to CNF
    ///
    /// Returns the result containing clauses and mappings.
    pub fn transform(mut self, root: TermId) -> TseitinResult {
        let root_lit = self.encode(root, true);

        // Assert the root
        self.add_clause(CnfClause::unit(root_lit));

        TseitinResult {
            clauses: self.clauses,
            term_to_var: self.term_to_var,
            var_to_term: self.var_to_term,
            root: root_lit,
            num_vars: self.next_var - 1,
        }
    }

    /// Transform multiple terms to CNF (conjunction)
    pub fn transform_all(mut self, terms: &[TermId]) -> TseitinResult {
        let mut roots = Vec::new();

        for &term_id in terms {
            let lit = self.encode(term_id, true);
            roots.push(lit);
            self.add_clause(CnfClause::unit(lit));
        }

        let root = if roots.len() == 1 {
            roots[0]
        } else {
            // Create a dummy root for multiple assertions
            self.fresh_var() as i32
        };

        TseitinResult {
            clauses: self.clauses,
            term_to_var: self.term_to_var,
            var_to_term: self.var_to_term,
            root,
            num_vars: self.next_var - 1,
        }
    }

    // ==================== Incremental API ====================

    /// Encode and assert a term, returning the root literal
    ///
    /// This is the incremental version that doesn't consume self.
    /// Multiple calls build up the clause database while sharing
    /// term-to-variable mappings.
    pub fn encode_and_assert(&mut self, term_id: TermId) -> CnfLit {
        let lit = self.encode(term_id, true);
        self.add_clause(CnfClause::unit(lit));
        lit
    }

    /// Encode and assert multiple terms
    pub fn encode_and_assert_all(&mut self, terms: &[TermId]) -> Vec<CnfLit> {
        terms
            .iter()
            .map(|&term_id| self.encode_and_assert(term_id))
            .collect()
    }

    /// Get new clauses since last extraction
    ///
    /// This returns clauses added since the last call to this method
    /// or since construction if never called.
    pub fn take_new_clauses(&mut self) -> Vec<CnfClause> {
        let new_clauses = self.clauses[self.clauses_extracted..].to_vec();
        self.clauses_extracted = self.clauses.len();
        new_clauses
    }

    /// Get all clauses (doesn't affect extraction marker)
    pub fn all_clauses(&self) -> &[CnfClause] {
        &self.clauses
    }

    /// Get the current number of variables
    pub fn num_vars(&self) -> u32 {
        self.next_var - 1
    }

    /// Get the term-to-variable mapping
    pub fn term_to_var(&self) -> &BTreeMap<TermId, u32> {
        &self.term_to_var
    }

    /// Get the variable-to-term mapping
    pub fn var_to_term(&self) -> &BTreeMap<u32, TermId> {
        &self.var_to_term
    }

    /// Get the CNF variable for a term if it exists
    pub fn get_var_for_term(&self, term: TermId) -> Option<u32> {
        self.term_to_var.get(&term).copied()
    }

    /// Get the term for a CNF variable if it exists
    pub fn get_term_for_var(&self, var: u32) -> Option<TermId> {
        self.var_to_term.get(&var).copied()
    }

    /// Reset clause extraction marker (but keep all mappings and clauses)
    pub fn reset_extraction_marker(&mut self) {
        self.clauses_extracted = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tseitin_simple_and() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let and_ab = store.mk_and(vec![a, b]);

        let tseitin = Tseitin::new(&store);
        let result = tseitin.transform(and_ab);

        // Should have clauses for:
        // - v_and ↔ (a ∧ b)
        // - v_and (root assertion)
        assert!(result.num_vars >= 3); // a, b, and_ab
        assert!(!result.clauses.is_empty());
    }

    #[test]
    fn test_tseitin_simple_or() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let or_ab = store.mk_or(vec![a, b]);

        let tseitin = Tseitin::new(&store);
        let result = tseitin.transform(or_ab);

        assert!(result.num_vars >= 3);
        assert!(!result.clauses.is_empty());
    }

    #[test]
    fn test_tseitin_negation() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let not_a = store.mk_not(a);

        let tseitin = Tseitin::new(&store);
        let result = tseitin.transform(not_a);

        // not(a) should just flip polarity
        // The root assertion should be ¬a
        assert!(!result.clauses.is_empty());
        // Should have a unit clause with negative literal
        let has_negative_unit = result.clauses.iter().any(|c| c.0.len() == 1 && c.0[0] < 0);
        assert!(has_negative_unit);
    }

    #[test]
    fn test_tseitin_ite() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let ite = store.mk_ite(c, a, b);

        let tseitin = Tseitin::new(&store);
        let result = tseitin.transform(ite);

        // With polarity-aware encoding, a top-level positive ITE only needs v -> ite (2 clauses)
        // plus a root assertion.
        assert!(result.clauses.len() >= 3);
        let ternary_count = result.clauses.iter().filter(|c| c.0.len() == 3).count();
        assert_eq!(ternary_count, 2);
    }

    #[test]
    fn test_tseitin_ite_both_polarities_emits_full_encoding() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let ite = store.mk_ite(c, a, b);
        let not_ite = store.mk_not(ite);

        // Force both polarities to be encoded: ite and (not ite) are both asserted.
        let tseitin = Tseitin::new(&store);
        let result = tseitin.transform_all(&[ite, not_ite]);

        // Both halves (v->ite and ite->v) are needed, totaling 4 ternary clauses.
        let ternary_count = result.clauses.iter().filter(|cl| cl.0.len() == 3).count();
        assert_eq!(ternary_count, 4);
    }

    #[test]
    fn test_tseitin_boolean_eq() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let eq_ab = store.mk_eq(a, b);

        let tseitin = Tseitin::new(&store);
        let result = tseitin.transform(eq_ab);

        // Boolean equality generates 4 clauses + root assertion
        assert!(result.clauses.len() >= 4);
    }

    #[test]
    fn test_tseitin_nested() {
        let mut store = TermStore::new();

        // (a ∧ b) ∨ (c ∧ d)
        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let c = store.mk_var("c", Sort::Bool);
        let d = store.mk_var("d", Sort::Bool);

        let and_ab = store.mk_and(vec![a, b]);
        let and_cd = store.mk_and(vec![c, d]);
        let or_both = store.mk_or(vec![and_ab, and_cd]);

        let tseitin = Tseitin::new(&store);
        let result = tseitin.transform(or_both);

        // Should have variables for a, b, c, d, and_ab, and_cd, or_both
        assert!(result.num_vars >= 7);
    }

    #[test]
    fn test_tseitin_multiple_terms() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let c = store.mk_var("c", Sort::Bool);

        let terms = vec![a, b, c];

        let tseitin = Tseitin::new(&store);
        let result = tseitin.transform_all(&terms);

        // Should have unit clauses for a, b, c
        let unit_clauses: Vec<_> = result.clauses.iter().filter(|c| c.0.len() == 1).collect();
        assert_eq!(unit_clauses.len(), 3);
    }

    #[test]
    fn test_variable_mapping() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let and_ab = store.mk_and(vec![a, b]);

        let tseitin = Tseitin::new(&store);
        let result = tseitin.transform(and_ab);

        // Check that we can map back from variables to terms
        assert!(result.var_for_term(a).is_some());
        assert!(result.var_for_term(b).is_some());

        let var_a = result.var_for_term(a).unwrap();
        let var_b = result.var_for_term(b).unwrap();

        assert_eq!(result.term_for_var(var_a), Some(a));
        assert_eq!(result.term_for_var(var_b), Some(b));
    }

    #[test]
    fn test_incremental_api() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let or_ab = store.mk_or(vec![a, b]);

        let mut tseitin = Tseitin::new(&store);

        // First batch: encode (a ∨ b)
        let _lit1 = tseitin.encode_and_assert(or_ab);
        let clauses1 = tseitin.take_new_clauses();
        assert!(!clauses1.is_empty());

        // Record variable assignment for 'a'
        let var_a = tseitin.get_var_for_term(a).unwrap();

        // Second batch: encode 'a' again - should reuse same variable
        let _lit2 = tseitin.encode_and_assert(a);
        let clauses2 = tseitin.take_new_clauses();

        // 'a' was already encoded, so we should just get a unit clause
        // with the same variable
        assert!(clauses2.len() == 1);
        let unit_clause = &clauses2[0];
        assert_eq!(unit_clause.0.len(), 1);
        assert_eq!(unit_clause.0[0].unsigned_abs(), var_a);
    }

    #[test]
    fn test_incremental_shared_variables() {
        let mut store = TermStore::new();

        // Create terms that share subterms
        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let c = store.mk_var("c", Sort::Bool);
        let and_ab = store.mk_and(vec![a, b]);
        let or_ab_c = store.mk_or(vec![and_ab, c]);

        let mut tseitin = Tseitin::new(&store);

        // First: encode (a ∧ b) ∨ c
        tseitin.encode_and_assert(or_ab_c);
        let _ = tseitin.take_new_clauses();

        // Get variable for 'a'
        let var_a_first = tseitin.get_var_for_term(a).unwrap();

        // Second: encode just 'a' - should get same variable
        tseitin.encode_and_assert(a);

        let var_a_second = tseitin.get_var_for_term(a).unwrap();

        // Critical: variable for 'a' should be the same in both encodings
        assert_eq!(var_a_first, var_a_second);
    }
}
