//! Gate Extraction for SAT solving
//!
//! Recognizes Boolean gate patterns in CNF clauses:
//! - AND gates: y ↔ (x1 ∧ x2 ∧ ... ∧ xn)
//! - XOR gates: y ↔ (x1 ⊕ x2)
//! - ITE gates: y ↔ ITE(c, t, e)
//! - Equivalences: y ↔ x
//!
//! Gate extraction enables more efficient variable elimination by restricting
//! resolutions to only those between gate and non-gate clauses.
//!
//! Reference: Eén & Biere, "Effective Preprocessing in SAT through Variable
//! and Clause Elimination", SAT 2005.

use crate::clause_db::ClauseDB;
use crate::literal::{Literal, Variable};

/// Types of gates that can be recognized
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateType {
    /// AND gate: y = x1 ∧ x2 ∧ ... ∧ xn
    And,
    /// XOR gate: y = x1 ⊕ x2 (binary XOR only for now)
    Xor,
    /// If-then-else: y = ITE(c, t, e) = (c ∧ t) ∨ (¬c ∧ e)
    Ite,
    /// Equivalence: y ↔ x
    Equiv,
}

/// A recognized gate structure
#[derive(Debug, Clone)]
pub struct Gate {
    /// The output variable (pivot)
    pub output: Variable,
    /// Type of gate
    pub gate_type: GateType,
    /// Input literals for the gate
    pub inputs: Vec<Literal>,
    /// Indices of clauses that form the gate definition
    pub defining_clauses: Vec<usize>,
}

/// Statistics for gate extraction
#[derive(Debug, Clone, Default)]
pub struct GateStats {
    /// Number of AND gates found
    pub and_gates: u64,
    /// Number of XOR gates found
    pub xor_gates: u64,
    /// Number of ITE gates found
    pub ite_gates: u64,
    /// Number of equivalences found
    pub equivalences: u64,
    /// Total gate extraction calls
    pub extraction_calls: u64,
}

impl GateStats {
    /// Total gates found
    pub fn total_gates(&self) -> u64 {
        self.and_gates + self.xor_gates + self.ite_gates + self.equivalences
    }
}

/// Gate extraction engine
pub struct GateExtractor {
    /// Temporary mark array for literals: 0 = unmarked, 1 = pos marked, -1 = neg marked
    marks: Vec<i8>,
    /// Marked literals to unmark later
    marked_lits: Vec<Literal>,
    /// Statistics
    stats: GateStats,
}

impl GateExtractor {
    /// Create a new gate extractor for n variables
    pub fn new(num_vars: usize) -> Self {
        GateExtractor {
            marks: vec![0; num_vars],
            marked_lits: Vec::new(),
            stats: GateStats::default(),
        }
    }

    /// Ensure internal buffers can handle `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        if self.marks.len() < num_vars {
            self.marks.resize(num_vars, 0);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &GateStats {
        &self.stats
    }

    /// Mark a literal
    fn mark(&mut self, lit: Literal) {
        let var_idx = lit.variable().0 as usize;
        if var_idx < self.marks.len() {
            self.marks[var_idx] = if lit.is_positive() { 1 } else { -1 };
            self.marked_lits.push(lit);
        }
    }

    /// Get mark for a variable
    fn get_mark(&self, var: Variable) -> i8 {
        let var_idx = var.0 as usize;
        if var_idx < self.marks.len() {
            self.marks[var_idx]
        } else {
            0
        }
    }

    /// Unmark all marked literals
    fn unmark_all(&mut self) {
        for &lit in &self.marked_lits {
            let var_idx = lit.variable().0 as usize;
            if var_idx < self.marks.len() {
                self.marks[var_idx] = 0;
            }
        }
        self.marked_lits.clear();
    }

    /// Try to find a gate for the given pivot variable
    ///
    /// # Arguments
    /// * `pivot` - The variable to find a gate for
    /// * `clauses` - All clauses as literal slices
    /// * `pos_occs` - Indices of clauses containing positive pivot
    /// * `neg_occs` - Indices of clauses containing negative pivot
    ///
    /// # Returns
    /// * `Some(Gate)` if a gate is found, `None` otherwise
    pub fn find_gate(
        &mut self,
        pivot: Variable,
        clauses: &[Vec<Literal>],
        pos_occs: &[usize],
        neg_occs: &[usize],
    ) -> Option<Gate> {
        self.stats.extraction_calls += 1;

        // Try each gate type in order of likelihood/simplicity
        if let Some(gate) = self.find_equivalence(pivot, clauses, pos_occs, neg_occs) {
            self.stats.equivalences += 1;
            return Some(gate);
        }

        if let Some(gate) = self.find_and_gate(pivot, clauses, pos_occs, neg_occs) {
            self.stats.and_gates += 1;
            return Some(gate);
        }

        if let Some(gate) = self.find_xor_gate(pivot, clauses, pos_occs, neg_occs) {
            self.stats.xor_gates += 1;
            return Some(gate);
        }

        if let Some(gate) = self.find_ite_gate(pivot, clauses, pos_occs) {
            self.stats.ite_gates += 1;
            return Some(gate);
        }

        None
    }

    /// Find a gate suitable for BVE restricted resolution.
    ///
    /// This variant operates directly on the clause database and intentionally
    /// skips ITE detection (which is incomplete and also requires a global scan).
    pub fn find_gate_for_bve(
        &mut self,
        pivot: Variable,
        clauses: &ClauseDB,
        pos_occs: &[usize],
        neg_occs: &[usize],
    ) -> Option<Gate> {
        self.stats.extraction_calls += 1;

        if let Some(gate) = self.find_equivalence_db(pivot, clauses, pos_occs, neg_occs) {
            self.stats.equivalences += 1;
            return Some(gate);
        }

        if let Some(gate) = self.find_and_gate_db(pivot, clauses, pos_occs, neg_occs) {
            self.stats.and_gates += 1;
            return Some(gate);
        }

        if let Some(gate) = self.find_xor_gate_db(pivot, clauses, pos_occs, neg_occs) {
            self.stats.xor_gates += 1;
            return Some(gate);
        }

        None
    }

    /// Find equivalence gate: pivot ↔ x
    /// Encoded as: (¬pivot ∨ x) ∧ (pivot ∨ ¬x)
    fn find_equivalence(
        &mut self,
        pivot: Variable,
        clauses: &[Vec<Literal>],
        pos_occs: &[usize],
        neg_occs: &[usize],
    ) -> Option<Gate> {
        let pivot_pos = Literal::positive(pivot);
        let pivot_neg = Literal::negative(pivot);

        // Mark all literals in binary clauses with positive pivot
        // (pivot ∨ x) means we're looking for ¬x in clauses with ¬pivot
        for &clause_idx in pos_occs {
            let clause = &clauses[clause_idx];
            if let Some(other) = self.get_binary_other(clause, pivot_pos) {
                self.mark(other);
            }
        }

        // Look for binary clause (¬pivot ∨ x) where ¬x is marked
        let mut result = None;
        for &clause_idx in neg_occs {
            let clause = &clauses[clause_idx];
            if let Some(other) = self.get_binary_other(clause, pivot_neg) {
                // Check if negation is marked (meaning we have pivot ∨ ¬other)
                let neg_other = other.negated();
                if self.get_mark(neg_other.variable()) != 0 {
                    let mark = self.get_mark(neg_other.variable());
                    let expected = if neg_other.is_positive() { 1 } else { -1 };
                    if mark == expected {
                        // Found equivalence: pivot ↔ other
                        // Find the matching positive clause
                        let mut pos_clause_idx = None;
                        for &ci in pos_occs {
                            let c = &clauses[ci];
                            if let Some(o) = self.get_binary_other(c, pivot_pos) {
                                if o == neg_other {
                                    pos_clause_idx = Some(ci);
                                    break;
                                }
                            }
                        }

                        if let Some(pci) = pos_clause_idx {
                            result = Some(Gate {
                                output: pivot,
                                gate_type: GateType::Equiv,
                                inputs: vec![other],
                                defining_clauses: vec![pci, clause_idx],
                            });
                            break;
                        }
                    }
                }
            }
        }

        self.unmark_all();
        result
    }

    fn find_equivalence_db(
        &mut self,
        pivot: Variable,
        clauses: &ClauseDB,
        pos_occs: &[usize],
        neg_occs: &[usize],
    ) -> Option<Gate> {
        let pivot_pos = Literal::positive(pivot);
        let pivot_neg = Literal::negative(pivot);

        for &clause_idx in pos_occs {
            if clause_idx >= clauses.len() || clauses.header(clause_idx).is_empty() {
                continue;
            }
            let clause = clauses.literals(clause_idx);
            if let Some(other) = self.get_binary_other(clause, pivot_pos) {
                self.mark(other);
            }
        }

        let mut result = None;
        for &clause_idx in neg_occs {
            if clause_idx >= clauses.len() || clauses.header(clause_idx).is_empty() {
                continue;
            }
            let clause = clauses.literals(clause_idx);
            if let Some(other) = self.get_binary_other(clause, pivot_neg) {
                let neg_other = other.negated();
                if self.get_mark(neg_other.variable()) != 0 {
                    let mark = self.get_mark(neg_other.variable());
                    let expected = if neg_other.is_positive() { 1 } else { -1 };
                    if mark == expected {
                        let mut pos_clause_idx = None;
                        for &ci in pos_occs {
                            if ci >= clauses.len() || clauses.header(ci).is_empty() {
                                continue;
                            }
                            let c = clauses.literals(ci);
                            if let Some(o) = self.get_binary_other(c, pivot_pos) {
                                if o == neg_other {
                                    pos_clause_idx = Some(ci);
                                    break;
                                }
                            }
                        }

                        if let Some(pci) = pos_clause_idx {
                            result = Some(Gate {
                                output: pivot,
                                gate_type: GateType::Equiv,
                                inputs: vec![other],
                                defining_clauses: vec![pci, clause_idx],
                            });
                            break;
                        }
                    }
                }
            }
        }

        self.unmark_all();
        result
    }

    /// Find AND gate: pivot ↔ (x1 ∧ x2 ∧ ... ∧ xn)
    /// Encoded as:
    /// - n binary clauses: (¬pivot ∨ x1), (¬pivot ∨ x2), ..., (¬pivot ∨ xn)
    /// - 1 clause: (pivot ∨ ¬x1 ∨ ¬x2 ∨ ... ∨ ¬xn)
    fn find_and_gate(
        &mut self,
        pivot: Variable,
        clauses: &[Vec<Literal>],
        pos_occs: &[usize],
        neg_occs: &[usize],
    ) -> Option<Gate> {
        let pivot_pos = Literal::positive(pivot);
        let pivot_neg = Literal::negative(pivot);

        // Mark all literals in binary clauses with ¬pivot
        // These are the (¬pivot ∨ xi) clauses
        let mut binary_clause_indices = Vec::new();
        for &clause_idx in neg_occs {
            let clause = &clauses[clause_idx];
            if let Some(other) = self.get_binary_other(clause, pivot_neg) {
                self.mark(other);
                binary_clause_indices.push(clause_idx);
            }
        }

        if binary_clause_indices.is_empty() {
            self.unmark_all();
            return None;
        }

        // Look for the long clause (pivot ∨ ¬x1 ∨ ¬x2 ∨ ... ∨ ¬xn)
        let mut result = None;
        for &clause_idx in pos_occs {
            let clause = &clauses[clause_idx];
            if clause.len() < 3 {
                continue;
            }

            // Check if all non-pivot literals are negations of marked literals
            let mut all_match = true;
            let mut inputs = Vec::new();
            let mut arity = 0;

            for &lit in clause {
                if lit == pivot_pos {
                    continue;
                }
                // lit should be ¬xi where xi is marked
                let neg_lit = lit.negated();
                let mark = self.get_mark(neg_lit.variable());
                let expected = if neg_lit.is_positive() { 1 } else { -1 };

                if mark != expected {
                    all_match = false;
                    break;
                }
                inputs.push(neg_lit);
                arity += 1;
            }

            // Check arity matches number of binary clauses
            if all_match && arity > 0 && arity == binary_clause_indices.len() {
                let mut defining = vec![clause_idx];
                defining.extend(binary_clause_indices.iter().cloned());

                result = Some(Gate {
                    output: pivot,
                    gate_type: GateType::And,
                    inputs,
                    defining_clauses: defining,
                });
                break;
            }
        }

        self.unmark_all();
        result
    }

    fn find_and_gate_db(
        &mut self,
        pivot: Variable,
        clauses: &ClauseDB,
        pos_occs: &[usize],
        neg_occs: &[usize],
    ) -> Option<Gate> {
        let pivot_pos = Literal::positive(pivot);
        let pivot_neg = Literal::negative(pivot);

        let mut binary_clause_indices = Vec::new();
        for &clause_idx in neg_occs {
            if clause_idx >= clauses.len() || clauses.header(clause_idx).is_empty() {
                continue;
            }
            let clause = clauses.literals(clause_idx);
            if let Some(other) = self.get_binary_other(clause, pivot_neg) {
                self.mark(other);
                binary_clause_indices.push(clause_idx);
            }
        }

        if binary_clause_indices.is_empty() {
            self.unmark_all();
            return None;
        }

        let mut result = None;
        for &clause_idx in pos_occs {
            if clause_idx >= clauses.len() {
                continue;
            }
            let header = clauses.header(clause_idx);
            if header.is_empty() || header.len() < 3 {
                continue;
            }
            let clause = clauses.literals(clause_idx);

            let mut all_match = true;
            let mut inputs = Vec::new();
            let mut arity = 0usize;

            for &lit in clause {
                if lit == pivot_pos {
                    continue;
                }
                let neg_lit = lit.negated();
                let mark = self.get_mark(neg_lit.variable());
                let expected = if neg_lit.is_positive() { 1 } else { -1 };

                if mark != expected {
                    all_match = false;
                    break;
                }
                inputs.push(neg_lit);
                arity += 1;
            }

            if all_match && arity > 0 && arity == binary_clause_indices.len() {
                let mut defining = vec![clause_idx];
                defining.extend(binary_clause_indices.iter().copied());

                result = Some(Gate {
                    output: pivot,
                    gate_type: GateType::And,
                    inputs,
                    defining_clauses: defining,
                });
                break;
            }
        }

        self.unmark_all();
        result
    }

    /// Find XOR gate: pivot ↔ (x1 ⊕ x2)
    /// Binary XOR encoded as 4 clauses:
    /// (pivot ∨ ¬x1 ∨ ¬x2), (pivot ∨ x1 ∨ x2),
    /// (¬pivot ∨ ¬x1 ∨ x2), (¬pivot ∨ x1 ∨ ¬x2)
    fn find_xor_gate(
        &mut self,
        pivot: Variable,
        clauses: &[Vec<Literal>],
        pos_occs: &[usize],
        neg_occs: &[usize],
    ) -> Option<Gate> {
        let pivot_pos = Literal::positive(pivot);
        let pivot_neg = Literal::negative(pivot);

        // Look for ternary clauses with pivot
        for &clause_idx in pos_occs {
            let clause = &clauses[clause_idx];
            let others = self.get_ternary_others(clause, pivot_pos);
            if others.is_none() {
                continue;
            }
            let (a, b) = others.unwrap();

            // For XOR, we need 4 specific clauses
            // This is the first clause: (pivot ∨ a ∨ b)
            // We need to find: (pivot ∨ ¬a ∨ ¬b), (¬pivot ∨ ¬a ∨ b), (¬pivot ∨ a ∨ ¬b)

            let needed_pos = [a.negated(), b.negated()]; // (pivot ∨ ¬a ∨ ¬b)
            let needed_neg1 = [a.negated(), b]; // (¬pivot ∨ ¬a ∨ b)
            let needed_neg2 = [a, b.negated()]; // (¬pivot ∨ a ∨ ¬b)

            // Find the other positive clause
            let mut pos_idx2 = None;
            for &ci in pos_occs {
                if ci == clause_idx {
                    continue;
                }
                let c = &clauses[ci];
                if let Some((x, y)) = self.get_ternary_others(c, pivot_pos) {
                    if (x == needed_pos[0] && y == needed_pos[1])
                        || (x == needed_pos[1] && y == needed_pos[0])
                    {
                        pos_idx2 = Some(ci);
                        break;
                    }
                }
            }

            if pos_idx2.is_none() {
                continue;
            }

            // Find the negative clauses
            let mut neg_idx1 = None;
            let mut neg_idx2 = None;
            for &ci in neg_occs {
                let c = &clauses[ci];
                if let Some((x, y)) = self.get_ternary_others(c, pivot_neg) {
                    if (x == needed_neg1[0] && y == needed_neg1[1])
                        || (x == needed_neg1[1] && y == needed_neg1[0])
                    {
                        neg_idx1 = Some(ci);
                    } else if (x == needed_neg2[0] && y == needed_neg2[1])
                        || (x == needed_neg2[1] && y == needed_neg2[0])
                    {
                        neg_idx2 = Some(ci);
                    }
                }
            }

            if neg_idx1.is_some() && neg_idx2.is_some() {
                // Found XOR gate
                return Some(Gate {
                    output: pivot,
                    gate_type: GateType::Xor,
                    inputs: vec![a, b],
                    defining_clauses: vec![
                        clause_idx,
                        pos_idx2.unwrap(),
                        neg_idx1.unwrap(),
                        neg_idx2.unwrap(),
                    ],
                });
            }
        }

        None
    }

    fn find_xor_gate_db(
        &mut self,
        pivot: Variable,
        clauses: &ClauseDB,
        pos_occs: &[usize],
        neg_occs: &[usize],
    ) -> Option<Gate> {
        let pivot_pos = Literal::positive(pivot);
        let pivot_neg = Literal::negative(pivot);

        for &clause_idx in pos_occs {
            if clause_idx >= clauses.len() || clauses.header(clause_idx).is_empty() {
                continue;
            }
            let clause = clauses.literals(clause_idx);
            let others = self.get_ternary_others(clause, pivot_pos);
            if others.is_none() {
                continue;
            }
            let (a, b) = others.unwrap();

            let needed_pos = [a.negated(), b.negated()];
            let needed_neg1 = [a.negated(), b];
            let needed_neg2 = [a, b.negated()];

            let mut pos_idx2 = None;
            for &ci in pos_occs {
                if ci == clause_idx || ci >= clauses.len() || clauses.header(ci).is_empty() {
                    continue;
                }
                let c = clauses.literals(ci);
                if let Some((x, y)) = self.get_ternary_others(c, pivot_pos) {
                    if (x == needed_pos[0] && y == needed_pos[1])
                        || (x == needed_pos[1] && y == needed_pos[0])
                    {
                        pos_idx2 = Some(ci);
                        break;
                    }
                }
            }

            if pos_idx2.is_none() {
                continue;
            }

            let mut neg_idx1 = None;
            let mut neg_idx2 = None;
            for &ci in neg_occs {
                if ci >= clauses.len() || clauses.header(ci).is_empty() {
                    continue;
                }
                let c = clauses.literals(ci);
                if let Some((x, y)) = self.get_ternary_others(c, pivot_neg) {
                    if (x == needed_neg1[0] && y == needed_neg1[1])
                        || (x == needed_neg1[1] && y == needed_neg1[0])
                    {
                        neg_idx1 = Some(ci);
                    } else if (x == needed_neg2[0] && y == needed_neg2[1])
                        || (x == needed_neg2[1] && y == needed_neg2[0])
                    {
                        neg_idx2 = Some(ci);
                    }
                }
            }

            if neg_idx1.is_some() && neg_idx2.is_some() {
                return Some(Gate {
                    output: pivot,
                    gate_type: GateType::Xor,
                    inputs: vec![a, b],
                    defining_clauses: vec![
                        clause_idx,
                        pos_idx2.unwrap(),
                        neg_idx1.unwrap(),
                        neg_idx2.unwrap(),
                    ],
                });
            }
        }

        None
    }

    /// Find ITE gate: pivot ↔ ITE(c, t, e)
    /// Encoded as 4 ternary clauses:
    /// (pivot ∨ c ∨ ¬e), (pivot ∨ ¬c ∨ ¬t),
    /// (¬pivot ∨ c ∨ e), (¬pivot ∨ ¬c ∨ t)
    fn find_ite_gate(
        &mut self,
        pivot: Variable,
        clauses: &[Vec<Literal>],
        pos_occs: &[usize],
    ) -> Option<Gate> {
        let pivot_pos = Literal::positive(pivot);

        // Look for pairs of ternary clauses with positive pivot
        for (i, &ci) in pos_occs.iter().enumerate() {
            let c1 = &clauses[ci];
            let others1 = self.get_ternary_others(c1, pivot_pos);
            if others1.is_none() {
                continue;
            }
            let (a1, b1) = others1.unwrap();

            for &cj in &pos_occs[i + 1..] {
                let c2 = &clauses[cj];
                let others2 = self.get_ternary_others(c2, pivot_pos);
                if others2.is_none() {
                    continue;
                }
                let (a2, b2) = others2.unwrap();

                // Check for ITE pattern: one literal should be negated between clauses
                // Pattern: (pivot ∨ c ∨ ¬e) and (pivot ∨ ¬c ∨ ¬t)
                // So a1 = c, b1 = ¬e, a2 = ¬c, b2 = ¬t (or permutations)
                let ite_check = |cond: Literal, then_neg: Literal, else_neg: Literal| {
                    // We need:
                    // (¬pivot ∨ cond ∨ else) and (¬pivot ∨ ¬cond ∨ then)
                    let pivot_neg = Literal::negative(pivot);
                    let else_lit = else_neg.negated();
                    let then_lit = then_neg.negated();

                    // Search for matching negative clauses (simplified)
                    for clause in clauses {
                        if clause.len() != 3 {
                            continue;
                        }
                        if !clause.contains(&pivot_neg) {
                            continue;
                        }
                        if let Some((x, y)) = self.get_ternary_others(clause, pivot_neg) {
                            if (x == cond && y == else_lit) || (x == else_lit && y == cond) {
                                // Found one matching negative clause
                                // Need to also find (¬pivot ∨ ¬cond ∨ then)
                                for clause2 in clauses {
                                    if clause2.len() != 3 {
                                        continue;
                                    }
                                    if !clause2.contains(&pivot_neg) {
                                        continue;
                                    }
                                    if let Some((p, q)) =
                                        self.get_ternary_others(clause2, pivot_neg)
                                    {
                                        if (p == cond.negated() && q == then_lit)
                                            || (p == then_lit && q == cond.negated())
                                        {
                                            return Some((cond, then_lit, else_lit));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    None
                };

                // Try different assignments of condition/then/else
                if a1 == a2.negated() {
                    // a1 = c, a2 = ¬c, b1 = ¬e, b2 = ¬t
                    if let Some((c, t, e)) = ite_check(a1, b2, b1) {
                        return Some(Gate {
                            output: pivot,
                            gate_type: GateType::Ite,
                            inputs: vec![c, t, e], // condition, then, else
                            defining_clauses: vec![ci, cj], // Simplified - would need all 4
                        });
                    }
                } else if b1 == b2.negated() {
                    // b1 = c, b2 = ¬c, a1 = ¬e, a2 = ¬t
                    if let Some((c, t, e)) = ite_check(b1, a2, a1) {
                        return Some(Gate {
                            output: pivot,
                            gate_type: GateType::Ite,
                            inputs: vec![c, t, e],
                            defining_clauses: vec![ci, cj],
                        });
                    }
                } else if a1 == b2.negated() {
                    // a1 = c, b2 = ¬c
                    if let Some((c, t, e)) = ite_check(a1, a2, b1) {
                        return Some(Gate {
                            output: pivot,
                            gate_type: GateType::Ite,
                            inputs: vec![c, t, e],
                            defining_clauses: vec![ci, cj],
                        });
                    }
                } else if b1 == a2.negated() {
                    // b1 = c, a2 = ¬c
                    if let Some((c, t, e)) = ite_check(b1, b2, a1) {
                        return Some(Gate {
                            output: pivot,
                            gate_type: GateType::Ite,
                            inputs: vec![c, t, e],
                            defining_clauses: vec![ci, cj],
                        });
                    }
                }
            }
        }

        None
    }

    /// Get the other literal in a binary clause (excluding the given literal)
    fn get_binary_other(&self, clause: &[Literal], exclude: Literal) -> Option<Literal> {
        if clause.len() != 2 {
            return None;
        }
        if clause[0] == exclude {
            Some(clause[1])
        } else if clause[1] == exclude {
            Some(clause[0])
        } else {
            None
        }
    }

    /// Get the other two literals in a ternary clause (excluding the given literal)
    fn get_ternary_others(
        &self,
        clause: &[Literal],
        exclude: Literal,
    ) -> Option<(Literal, Literal)> {
        if clause.len() != 3 {
            return None;
        }
        let mut others = Vec::with_capacity(2);
        for &lit in clause {
            if lit != exclude {
                others.push(lit);
            }
        }
        if others.len() == 2 {
            Some((others[0], others[1]))
        } else {
            None
        }
    }

    /// Given a gate, determine which clauses are "gate clauses" vs "non-gate clauses"
    /// for BVE purposes. Returns indices of non-gate clauses.
    pub fn get_non_gate_clauses(
        &self,
        gate: &Gate,
        pos_occs: &[usize],
        neg_occs: &[usize],
    ) -> Vec<usize> {
        let gate_set: std::collections::HashSet<usize> =
            gate.defining_clauses.iter().cloned().collect();

        let mut non_gate = Vec::new();
        for &idx in pos_occs {
            if !gate_set.contains(&idx) {
                non_gate.push(idx);
            }
        }
        for &idx in neg_occs {
            if !gate_set.contains(&idx) {
                non_gate.push(idx);
            }
        }
        non_gate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(v: u32, positive: bool) -> Literal {
        if positive {
            Literal::positive(Variable(v))
        } else {
            Literal::negative(Variable(v))
        }
    }

    #[test]
    fn test_equivalence_detection() {
        // y ↔ x encoded as (y ∨ ¬x) ∧ (¬y ∨ x)
        // Variable 0 is y, variable 1 is x
        let clauses = vec![
            vec![lit(0, true), lit(1, false)], // (y ∨ ¬x) - pos_occs for y
            vec![lit(0, false), lit(1, true)], // (¬y ∨ x) - neg_occs for y
        ];

        let mut extractor = GateExtractor::new(2);
        let pos_occs = vec![0]; // clause 0 has positive y
        let neg_occs = vec![1]; // clause 1 has negative y

        let gate = extractor.find_gate(Variable(0), &clauses, &pos_occs, &neg_occs);
        assert!(gate.is_some());
        let g = gate.unwrap();
        assert_eq!(g.gate_type, GateType::Equiv);
        assert_eq!(g.output, Variable(0));
        assert_eq!(g.inputs.len(), 1);
        assert_eq!(g.inputs[0].variable(), Variable(1));
    }

    #[test]
    fn test_and_gate_detection() {
        // y ↔ (a ∧ b) encoded as:
        // (¬y ∨ a), (¬y ∨ b), (y ∨ ¬a ∨ ¬b)
        // Variable 0 is y, variable 1 is a, variable 2 is b
        let clauses = vec![
            vec![lit(0, false), lit(1, true)],                // (¬y ∨ a)
            vec![lit(0, false), lit(2, true)],                // (¬y ∨ b)
            vec![lit(0, true), lit(1, false), lit(2, false)], // (y ∨ ¬a ∨ ¬b)
        ];

        let mut extractor = GateExtractor::new(3);
        let pos_occs = vec![2]; // clause 2 has positive y
        let neg_occs = vec![0, 1]; // clauses 0, 1 have negative y

        let gate = extractor.find_gate(Variable(0), &clauses, &pos_occs, &neg_occs);
        assert!(gate.is_some());
        let g = gate.unwrap();
        assert_eq!(g.gate_type, GateType::And);
        assert_eq!(g.output, Variable(0));
        assert_eq!(g.inputs.len(), 2);
    }

    #[test]
    fn test_xor_gate_detection() {
        // y ↔ (a ⊕ b) encoded as:
        // (y ∨ a ∨ b), (y ∨ ¬a ∨ ¬b), (¬y ∨ ¬a ∨ b), (¬y ∨ a ∨ ¬b)
        // Variable 0 is y, variable 1 is a, variable 2 is b
        let clauses = vec![
            vec![lit(0, true), lit(1, true), lit(2, true)], // (y ∨ a ∨ b)
            vec![lit(0, true), lit(1, false), lit(2, false)], // (y ∨ ¬a ∨ ¬b)
            vec![lit(0, false), lit(1, false), lit(2, true)], // (¬y ∨ ¬a ∨ b)
            vec![lit(0, false), lit(1, true), lit(2, false)], // (¬y ∨ a ∨ ¬b)
        ];

        let mut extractor = GateExtractor::new(3);
        let pos_occs = vec![0, 1]; // clauses 0, 1 have positive y
        let neg_occs = vec![2, 3]; // clauses 2, 3 have negative y

        let gate = extractor.find_gate(Variable(0), &clauses, &pos_occs, &neg_occs);
        assert!(gate.is_some());
        let g = gate.unwrap();
        assert_eq!(g.gate_type, GateType::Xor);
        assert_eq!(g.output, Variable(0));
        assert_eq!(g.inputs.len(), 2);
    }

    #[test]
    fn test_no_gate() {
        // Random clauses that don't form a gate pattern
        let clauses = vec![
            vec![lit(0, true), lit(1, true)],
            vec![lit(0, false), lit(2, true)],
            vec![lit(1, true), lit(2, false)],
        ];

        let mut extractor = GateExtractor::new(3);
        let pos_occs = vec![0];
        let neg_occs = vec![1];

        let gate = extractor.find_gate(Variable(0), &clauses, &pos_occs, &neg_occs);
        assert!(gate.is_none());
    }

    #[test]
    fn test_gate_stats() {
        let mut extractor = GateExtractor::new(3);

        // Test equivalence
        let clauses = vec![
            vec![lit(0, true), lit(1, false)],
            vec![lit(0, false), lit(1, true)],
        ];
        let _ = extractor.find_gate(Variable(0), &clauses, &[0], &[1]);

        let stats = extractor.stats();
        assert_eq!(stats.equivalences, 1);
        assert_eq!(stats.extraction_calls, 1);
    }

    #[test]
    fn test_non_gate_clauses() {
        // y ↔ x with extra non-gate clause
        let clauses = vec![
            vec![lit(0, true), lit(1, false)], // gate clause
            vec![lit(0, false), lit(1, true)], // gate clause
            vec![lit(0, true), lit(2, true)],  // non-gate clause
        ];

        let mut extractor = GateExtractor::new(3);
        let pos_occs = vec![0, 2];
        let neg_occs = vec![1];

        let gate = extractor.find_gate(Variable(0), &clauses, &pos_occs, &neg_occs);
        assert!(gate.is_some());

        let non_gate = extractor.get_non_gate_clauses(&gate.unwrap(), &pos_occs, &neg_occs);
        assert_eq!(non_gate.len(), 1);
        assert!(non_gate.contains(&2));
    }

    #[test]
    fn test_find_gate_for_bve() {
        let mut extractor = GateExtractor::new(3);

        // y ↔ x encoded as (y ∨ ¬x) ∧ (¬y ∨ x)
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, false)], false);
        clauses.add(&[lit(0, false), lit(1, true)], false);

        let pos_occs = vec![0];
        let neg_occs = vec![1];

        let gate = extractor.find_gate_for_bve(Variable(0), &clauses, &pos_occs, &neg_occs);
        assert!(gate.is_some());
        let g = gate.unwrap();
        assert_eq!(g.gate_type, GateType::Equiv);
    }
}
