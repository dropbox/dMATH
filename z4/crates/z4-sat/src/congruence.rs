//! Congruence closure for preprocessing
//!
//! Detects equivalent variables by analyzing gate structures. If two gates
//! have the same type and equivalent inputs, their outputs are equivalent.
//!
//! For example:
//! - Gate G1: y1 = AND(a, b)
//! - Gate G2: y2 = AND(a, b)
//! - Then y1 and y2 are equivalent
//!
//! More generally, if G1: y1 = AND(a, b) and G2: y2 = AND(c, d), and we know
//! a ≡ c and b ≡ d (from the union-find), then y1 ≡ y2.
//!
//! This is called "congruence closure" because it closes the equivalence
//! relation under the congruence rule: if f(x) = f(y) and x ≡ y, then
//! f(x) ≡ f(y).
//!
//! Reference: CaDiCaL's probing and equivalence detection techniques.

use crate::clause_db::ClauseDB;
use crate::gates::{Gate, GateExtractor, GateType};
use crate::literal::{Literal, Variable};
use std::collections::HashMap;

/// Statistics for congruence closure
#[derive(Debug, Clone, Default)]
pub struct CongruenceStats {
    /// Number of gates analyzed
    pub gates_analyzed: u64,
    /// Number of equivalences found via congruence
    pub equivalences_found: u64,
    /// Number of literals rewritten
    pub literals_rewritten: u64,
    /// Number of clauses modified
    pub clauses_modified: u64,
    /// Number of rounds of congruence closure
    pub rounds: u64,
}

/// Union-Find data structure for tracking equivalence classes
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }

        // Union by rank, preferring smaller index as representative
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => {
                self.parent[rx] = ry;
            }
            std::cmp::Ordering::Greater => {
                self.parent[ry] = rx;
            }
            std::cmp::Ordering::Equal => {
                // Prefer smaller index as root
                if rx < ry {
                    self.parent[ry] = rx;
                    self.rank[rx] += 1;
                } else {
                    self.parent[rx] = ry;
                    self.rank[ry] += 1;
                }
            }
        }
        true
    }
}

/// Canonical form of a gate for comparison
/// Inputs are sorted to handle commutativity
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct GateSignature {
    gate_type: GateType,
    /// Canonicalized inputs (sorted by representative literal index)
    inputs: Vec<usize>,
}

/// Congruence closure engine
pub struct CongruenceClosure {
    num_vars: usize,
    stats: CongruenceStats,
}

impl CongruenceClosure {
    /// Create a new congruence closure engine for n variables
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            stats: CongruenceStats::default(),
        }
    }

    /// Ensure internal buffers can handle `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        if num_vars > self.num_vars {
            self.num_vars = num_vars;
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &CongruenceStats {
        &self.stats
    }

    /// Run congruence closure on the clause database.
    ///
    /// Returns the literal mapping (canonical representative for each literal)
    /// and whether any equivalences were found.
    pub fn run(&mut self, clauses: &ClauseDB) -> CongruenceResult {
        self.stats.rounds += 1;

        // Step 1: Extract gates from the formula
        let mut extractor = GateExtractor::new(self.num_vars);
        let gates = self.extract_all_gates(&mut extractor, clauses);

        if gates.is_empty() {
            return CongruenceResult {
                found_equivalences: false,
                lit_map: self.identity_map(),
                clauses_to_delete: Vec::new(),
                clauses_to_replace: Vec::new(),
            };
        }

        self.stats.gates_analyzed += gates.len() as u64;

        // Step 2: Initialize union-find for literals (2 * num_vars entries)
        let num_lits = self.num_vars * 2;
        let mut uf = UnionFind::new(num_lits);

        // Step 3: Build congruence closure
        let found_new = self.compute_congruence_closure(&gates, &mut uf);

        if !found_new {
            return CongruenceResult {
                found_equivalences: false,
                lit_map: self.identity_map(),
                clauses_to_delete: Vec::new(),
                clauses_to_replace: Vec::new(),
            };
        }

        // Step 4: Build literal map from union-find
        let lit_map = self.build_lit_map(&mut uf);

        // Step 5: Compute rewrites needed
        let (clauses_to_delete, clauses_to_replace) = self.compute_rewrites(clauses, &lit_map);

        CongruenceResult {
            found_equivalences: true,
            lit_map,
            clauses_to_delete,
            clauses_to_replace,
        }
    }

    /// Extract all gates from the clause database
    fn extract_all_gates(&self, extractor: &mut GateExtractor, clauses: &ClauseDB) -> Vec<Gate> {
        let mut gates = Vec::new();

        // Build occurrence lists
        let mut pos_occs: Vec<Vec<usize>> = vec![Vec::new(); self.num_vars];
        let mut neg_occs: Vec<Vec<usize>> = vec![Vec::new(); self.num_vars];

        for idx in clauses.indices() {
            let header = clauses.header(idx);
            if header.is_empty() {
                continue;
            }
            for &lit in clauses.literals(idx) {
                let var_idx = lit.variable().0 as usize;
                if var_idx < self.num_vars {
                    if lit.is_positive() {
                        pos_occs[var_idx].push(idx);
                    } else {
                        neg_occs[var_idx].push(idx);
                    }
                }
            }
        }

        // Try to find a gate for each variable
        for var_idx in 0..self.num_vars {
            if pos_occs[var_idx].is_empty() || neg_occs[var_idx].is_empty() {
                continue;
            }

            if let Some(gate) = extractor.find_gate_for_bve(
                Variable(var_idx as u32),
                clauses,
                &pos_occs[var_idx],
                &neg_occs[var_idx],
            ) {
                gates.push(gate);
            }
        }

        gates
    }

    /// Compute congruence closure iteratively
    fn compute_congruence_closure(&mut self, gates: &[Gate], uf: &mut UnionFind) -> bool {
        let mut changed = true;
        let mut found_any = false;

        // Iterate until fixpoint
        while changed {
            changed = false;

            // Group gates by their signature (type + canonicalized inputs)
            let mut gate_groups: HashMap<GateSignature, Vec<usize>> = HashMap::new();

            for (gate_idx, gate) in gates.iter().enumerate() {
                let sig = self.gate_signature(gate, uf);
                gate_groups.entry(sig).or_default().push(gate_idx);
            }

            // For each group with multiple gates, merge their outputs
            for (_sig, group) in gate_groups.iter() {
                if group.len() < 2 {
                    continue;
                }

                // Merge all outputs in this group
                let first_output = gates[group[0]].output;
                let first_pos_idx = Literal::positive(first_output).index();

                for &gate_idx in &group[1..] {
                    let output = gates[gate_idx].output;
                    let pos_idx = Literal::positive(output).index();

                    if uf.union(first_pos_idx, pos_idx) {
                        self.stats.equivalences_found += 1;
                        changed = true;
                        found_any = true;

                        // Also union the negative literals
                        let first_neg_idx = Literal::negative(first_output).index();
                        let neg_idx = Literal::negative(output).index();
                        uf.union(first_neg_idx, neg_idx);
                    }
                }
            }
        }

        found_any
    }

    /// Create a signature for a gate based on its type and canonicalized inputs
    fn gate_signature(&self, gate: &Gate, uf: &mut UnionFind) -> GateSignature {
        // Map each input to its representative
        let mut canonical_inputs: Vec<usize> =
            gate.inputs.iter().map(|lit| uf.find(lit.index())).collect();

        // Sort for commutative gates (AND, XOR)
        match gate.gate_type {
            GateType::And | GateType::Xor => {
                canonical_inputs.sort();
            }
            GateType::Ite => {
                // ITE is not fully commutative, but then/else can be swapped
                // with condition negation. For simplicity, we don't handle this.
            }
            GateType::Equiv => {
                canonical_inputs.sort();
            }
        }

        GateSignature {
            gate_type: gate.gate_type,
            inputs: canonical_inputs,
        }
    }

    /// Build literal map from union-find
    fn build_lit_map(&self, uf: &mut UnionFind) -> Vec<Literal> {
        let num_lits = self.num_vars * 2;
        let mut lit_map = Vec::with_capacity(num_lits);

        for lit_idx in 0..num_lits {
            let rep_idx = uf.find(lit_idx);
            lit_map.push(Literal::from_index(rep_idx));
        }

        lit_map
    }

    /// Create identity literal map
    fn identity_map(&self) -> Vec<Literal> {
        let num_lits = self.num_vars * 2;
        (0..num_lits).map(Literal::from_index).collect()
    }

    /// Compute which clauses need to be deleted or replaced based on the literal map.
    fn compute_rewrites(
        &mut self,
        clauses: &ClauseDB,
        lit_map: &[Literal],
    ) -> (Vec<usize>, Vec<(usize, Vec<Literal>)>) {
        let mut clauses_to_delete = Vec::new();
        let mut clauses_to_replace = Vec::new();

        for idx in clauses.indices() {
            let header = clauses.header(idx);
            if header.is_empty() {
                continue;
            }

            let old_lits = clauses.literals(idx);
            let mut modified = false;
            let mut new_lits = Vec::with_capacity(old_lits.len());
            let mut seen = std::collections::HashSet::new();
            let mut is_tautology = false;

            for &lit in old_lits {
                let lit_idx = lit.index();
                let mapped = if lit_idx < lit_map.len() {
                    lit_map[lit_idx]
                } else {
                    lit
                };

                if mapped != lit {
                    modified = true;
                    self.stats.literals_rewritten += 1;
                }

                // Check for tautology (both literal and its negation present)
                if seen.contains(&mapped.negated()) {
                    // Tautology - mark for deletion
                    is_tautology = true;
                    break;
                }

                // Skip duplicate literals
                if !seen.contains(&mapped) {
                    seen.insert(mapped);
                    new_lits.push(mapped);
                }
            }

            if is_tautology {
                clauses_to_delete.push(idx);
                self.stats.clauses_modified += 1;
            } else if modified {
                clauses_to_replace.push((idx, new_lits));
                self.stats.clauses_modified += 1;
            }
        }

        (clauses_to_delete, clauses_to_replace)
    }
}

/// Result of running congruence closure
#[derive(Debug)]
pub struct CongruenceResult {
    /// Whether any equivalences were found
    pub found_equivalences: bool,
    /// Literal mapping (canonical representative for each literal index)
    pub lit_map: Vec<Literal>,
    /// Clauses that should be deleted (became tautologies)
    pub clauses_to_delete: Vec<usize>,
    /// Clauses that should be replaced: (clause_idx, new_literals)
    pub clauses_to_replace: Vec<(usize, Vec<Literal>)>,
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
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(10);

        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.find(5), 5);

        uf.union(0, 5);
        assert_eq!(uf.find(0), uf.find(5));

        uf.union(1, 2);
        uf.union(2, 3);
        assert_eq!(uf.find(1), uf.find(3));
    }

    #[test]
    fn test_no_gates_no_change() {
        // Simple clauses with no gate structure
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(1, false), lit(2, true)], false);

        let mut cc = CongruenceClosure::new(3);
        let result = cc.run(&clauses);

        assert!(!result.found_equivalences);
    }

    #[test]
    fn test_identical_and_gates() {
        // Two AND gates with identical inputs:
        // y1 = AND(a, b): (¬y1 ∨ a), (¬y1 ∨ b), (y1 ∨ ¬a ∨ ¬b)
        // y2 = AND(a, b): (¬y2 ∨ a), (¬y2 ∨ b), (y2 ∨ ¬a ∨ ¬b)
        // Variables: y1=0, y2=1, a=2, b=3

        let mut clauses = ClauseDB::new();
        // y1 = AND(a, b)
        clauses.add(&[lit(0, false), lit(2, true)], false); // (¬y1 ∨ a)
        clauses.add(&[lit(0, false), lit(3, true)], false); // (¬y1 ∨ b)
        clauses.add(&[lit(0, true), lit(2, false), lit(3, false)], false); // (y1 ∨ ¬a ∨ ¬b)
                                                                           // y2 = AND(a, b)
        clauses.add(&[lit(1, false), lit(2, true)], false); // (¬y2 ∨ a)
        clauses.add(&[lit(1, false), lit(3, true)], false); // (¬y2 ∨ b)
        clauses.add(&[lit(1, true), lit(2, false), lit(3, false)], false); // (y2 ∨ ¬a ∨ ¬b)

        let mut cc = CongruenceClosure::new(4);
        let result = cc.run(&clauses);

        // Should detect that y1 ≡ y2
        assert!(result.found_equivalences);
        assert!(cc.stats.equivalences_found > 0);
    }

    #[test]
    fn test_equivalence_gates() {
        // Two EQUIV gates:
        // y1 ↔ a: (y1 ∨ ¬a), (¬y1 ∨ a)
        // y2 ↔ a: (y2 ∨ ¬a), (¬y2 ∨ a)
        // Variables: y1=0, y2=1, a=2

        let mut clauses = ClauseDB::new();
        // y1 ↔ a
        clauses.add(&[lit(0, true), lit(2, false)], false); // (y1 ∨ ¬a)
        clauses.add(&[lit(0, false), lit(2, true)], false); // (¬y1 ∨ a)
                                                            // y2 ↔ a
        clauses.add(&[lit(1, true), lit(2, false)], false); // (y2 ∨ ¬a)
        clauses.add(&[lit(1, false), lit(2, true)], false); // (¬y2 ∨ a)

        let mut cc = CongruenceClosure::new(3);
        let result = cc.run(&clauses);

        // Should detect that y1 ≡ y2
        assert!(result.found_equivalences);
    }

    #[test]
    fn test_stats_tracking() {
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);

        let mut cc = CongruenceClosure::new(2);
        cc.run(&clauses);

        assert_eq!(cc.stats.rounds, 1);
    }
}
