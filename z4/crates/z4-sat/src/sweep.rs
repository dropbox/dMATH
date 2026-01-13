//! SAT sweeping (equivalence merging) for CNF formulas.
//!
//! Sweeping detects equivalent literals in the binary-clause implication graph
//! (2-SAT style SCC analysis). If two literals are mutually implied, they are
//! equivalent and can be merged by rewriting clauses to use a canonical
//! representative.
//!
//! This is an equisatisfiable simplification (not model-preserving unless a
//! reconstruction map is maintained by the caller).

use crate::clause_db::ClauseDB;
use crate::literal::{Literal, Variable};

/// Statistics for sweeping.
#[derive(Debug, Clone, Default)]
pub struct SweepStats {
    /// Number of sweep rounds executed.
    pub rounds: u64,
    /// Number of binary clauses scanned for implications.
    pub binary_clauses: u64,
    /// Number of SCC components found.
    pub components: u64,
    /// Number of variables proven contradictory (x and ¬x in same SCC).
    pub contradictions: u64,
    /// Number of literals rewritten via canonicalization.
    pub literals_rewritten: u64,
    /// Number of clauses changed (literals or length changed).
    pub clauses_rewritten: u64,
    /// Number of tautological clauses deleted after rewriting.
    pub clauses_deleted_tautology: u64,
    /// Number of clauses that became unit after rewriting.
    pub clauses_became_unit: u64,
    /// Number of clauses that became empty after rewriting.
    pub clauses_became_empty: u64,
}

/// Outcome of a sweep round.
#[derive(Debug, Clone)]
pub struct SweepOutcome {
    /// If true, the formula is UNSAT (contradiction detected during SCC).
    pub unsat: bool,
    /// Canonical literal mapping for each literal index (size = 2*num_vars).
    pub lit_map: Vec<Literal>,
    /// Unit literals created by clause rewriting.
    pub new_units: Vec<Literal>,
    /// Clauses that should be deleted (marked as tautology or empty).
    pub clauses_to_delete: Vec<usize>,
    /// Clauses that should be replaced: (clause_idx, new_literals).
    pub clauses_to_replace: Vec<(usize, Vec<Literal>)>,
}

/// Sweeping engine.
pub struct Sweeper {
    num_vars: usize,
    stats: SweepStats,
    graph: Vec<Vec<usize>>,
    rev_graph: Vec<Vec<usize>>,
    seen_mark: Vec<i8>,
    touched_vars: Vec<usize>,
}

impl Sweeper {
    /// Create a new sweeper for `num_vars` variables.
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            stats: SweepStats::default(),
            graph: vec![Vec::new(); num_vars * 2],
            rev_graph: vec![Vec::new(); num_vars * 2],
            seen_mark: vec![0; num_vars],
            touched_vars: Vec::new(),
        }
    }

    /// Ensure internal buffers can handle `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        if self.num_vars >= num_vars {
            return;
        }
        self.num_vars = num_vars;
        let num_lits = num_vars.saturating_mul(2);
        if self.graph.len() < num_lits {
            self.graph.resize_with(num_lits, Vec::new);
        }
        if self.rev_graph.len() < num_lits {
            self.rev_graph.resize_with(num_lits, Vec::new);
        }
        if self.seen_mark.len() < num_vars {
            self.seen_mark.resize(num_vars, 0);
        }
    }

    /// Get sweep statistics.
    pub fn stats(&self) -> &SweepStats {
        &self.stats
    }

    /// Run a sweep round on the given clause database.
    ///
    /// Returns an outcome containing the literal mapping, new units, and
    /// lists of clauses to delete or replace. The caller is responsible for
    /// applying these changes to the clause database.
    pub fn sweep(&mut self, clauses: &ClauseDB) -> SweepOutcome {
        self.stats.rounds += 1;

        let lit_map = self.compute_lit_map(clauses);
        let mut new_units = Vec::new();
        let mut clauses_to_delete = Vec::new();
        let mut clauses_to_replace = Vec::new();
        let mut derived_empty_clause = false;

        // If compute_lit_map found an UNSAT contradiction, it encodes that by
        // returning an empty map.
        if lit_map.is_empty() {
            return SweepOutcome {
                unsat: true,
                lit_map,
                new_units,
                clauses_to_delete,
                clauses_to_replace,
            };
        }

        for idx in clauses.indices() {
            let header = clauses.header(idx);
            if header.is_empty() {
                continue;
            }

            let old_lits: Vec<Literal> = clauses.literals(idx).to_vec();
            let old_len = old_lits.len();

            match self.rewrite_clause(&old_lits, &lit_map) {
                RewriteResult::Tautology => {
                    self.stats.clauses_deleted_tautology += 1;
                    clauses_to_delete.push(idx);
                }
                RewriteResult::Empty => {
                    self.stats.clauses_became_empty += 1;
                    clauses_to_delete.push(idx);
                    derived_empty_clause = true;
                    break;
                }
                RewriteResult::Unit(lit) => {
                    self.stats.clauses_became_unit += 1;
                    clauses_to_replace.push((idx, vec![lit]));
                    new_units.push(lit);
                }
                RewriteResult::Changed(new_lits) => {
                    if new_lits.len() != old_len || new_lits != old_lits {
                        self.stats.clauses_rewritten += 1;
                    }
                    clauses_to_replace.push((idx, new_lits));
                }
                RewriteResult::Unchanged => {}
            }
        }

        if derived_empty_clause {
            return SweepOutcome {
                unsat: true,
                lit_map,
                new_units,
                clauses_to_delete,
                clauses_to_replace,
            };
        }

        SweepOutcome {
            unsat: false,
            lit_map,
            new_units,
            clauses_to_delete,
            clauses_to_replace,
        }
    }

    fn clear_graph(&mut self) {
        for edges in &mut self.graph {
            edges.clear();
        }
        for edges in &mut self.rev_graph {
            edges.clear();
        }
    }

    fn add_edge(&mut self, from: Literal, to: Literal) {
        let from_idx = from.index();
        let to_idx = to.index();
        if from_idx >= self.graph.len() || to_idx >= self.graph.len() {
            return;
        }
        self.graph[from_idx].push(to_idx);
        self.rev_graph[to_idx].push(from_idx);
    }

    /// Build the implication graph from binary clauses and compute a canonical map.
    ///
    /// Returns an empty vector if an UNSAT SCC contradiction is detected.
    fn compute_lit_map(&mut self, clauses: &ClauseDB) -> Vec<Literal> {
        self.clear_graph();

        let num_nodes = self.num_vars * 2;

        for idx in clauses.indices() {
            let header = clauses.header(idx);
            if header.is_empty() {
                continue;
            }
            if header.len() != 2 {
                continue;
            }
            self.stats.binary_clauses += 1;

            let lits = clauses.literals(idx);
            let a = lits[0];
            let b = lits[1];

            // (a ∨ b) gives implications: ¬a → b and ¬b → a.
            self.add_edge(a.negated(), b);
            self.add_edge(b.negated(), a);
        }

        let (comp, num_comps) = kosaraju_scc(&self.graph, &self.rev_graph);
        self.stats.components += num_comps as u64;

        // Unsat check: x and ¬x in same SCC.
        for var_idx in 0..self.num_vars {
            let pos = Literal::positive(Variable(var_idx as u32)).index();
            let neg = Literal::negative(Variable(var_idx as u32)).index();
            if pos < comp.len() && neg < comp.len() && comp[pos] == comp[neg] {
                self.stats.contradictions += 1;
                return Vec::new();
            }
        }

        // Representative per component: choose minimal literal index.
        let mut rep: Vec<usize> = vec![usize::MAX; num_comps];
        for (node, &cid) in comp.iter().enumerate().take(num_nodes) {
            rep[cid] = rep[cid].min(node);
        }

        let mut lit_map = Vec::with_capacity(num_nodes);
        for node in 0..num_nodes {
            let mapped = rep[comp[node]];
            if mapped != node {
                self.stats.literals_rewritten += 1;
            }
            lit_map.push(Literal(mapped as u32));
        }

        lit_map
    }

    fn rewrite_clause(&mut self, lits: &[Literal], lit_map: &[Literal]) -> RewriteResult {
        let mut out = Vec::with_capacity(lits.len());

        for &lit in lits {
            let idx = lit.index();
            if idx >= lit_map.len() {
                out.push(lit);
                continue;
            }
            let mapped = lit_map[idx];

            let var_idx = mapped.variable().index();
            if var_idx >= self.seen_mark.len() {
                out.push(mapped);
                continue;
            }

            let sign: i8 = if mapped.is_positive() { 1 } else { -1 };
            let mark = self.seen_mark[var_idx];
            if mark == -sign {
                self.clear_seen_marks();
                return RewriteResult::Tautology;
            }
            if mark == 0 {
                self.seen_mark[var_idx] = sign;
                self.touched_vars.push(var_idx);
                out.push(mapped);
            }
        }

        self.clear_seen_marks();

        match out.len() {
            0 => RewriteResult::Empty,
            1 => RewriteResult::Unit(out[0]),
            _ => {
                if out == lits {
                    RewriteResult::Unchanged
                } else {
                    RewriteResult::Changed(out)
                }
            }
        }
    }

    fn clear_seen_marks(&mut self) {
        for &v in &self.touched_vars {
            self.seen_mark[v] = 0;
        }
        self.touched_vars.clear();
    }
}

#[derive(Debug, Clone)]
enum RewriteResult {
    Unchanged,
    Changed(Vec<Literal>),
    Unit(Literal),
    Empty,
    Tautology,
}

fn kosaraju_scc(graph: &[Vec<usize>], rev_graph: &[Vec<usize>]) -> (Vec<usize>, usize) {
    let n = graph.len();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);

    for start in 0..n {
        if visited[start] {
            continue;
        }
        // (node, next edge index)
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        visited[start] = true;
        while let Some((node, next_idx)) = stack.pop() {
            if next_idx < graph[node].len() {
                let to = graph[node][next_idx];
                stack.push((node, next_idx + 1));
                if !visited[to] {
                    visited[to] = true;
                    stack.push((to, 0));
                }
            } else {
                order.push(node);
            }
        }
    }

    let mut comp = vec![usize::MAX; n];
    let mut comp_id = 0usize;

    for &start in order.iter().rev() {
        if comp[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        comp[start] = comp_id;
        while let Some(node) = stack.pop() {
            for &to in &rev_graph[node] {
                if comp[to] == usize::MAX {
                    comp[to] = comp_id;
                    stack.push(to);
                }
            }
        }
        comp_id += 1;
    }

    (comp, comp_id)
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
    fn sweep_merges_equivalence_and_rewrites() {
        // x0 ↔ x1:
        // (x0 ∨ ¬x1) ∧ (¬x0 ∨ x1)
        // plus: (x1 ∨ x2)
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, false)], false);
        clauses.add(&[lit(0, false), lit(1, true)], false);
        clauses.add(&[lit(1, true), lit(2, true)], false);

        let mut sweeper = Sweeper::new(3);
        let outcome = sweeper.sweep(&clauses);
        assert!(!outcome.unsat);
        assert_eq!(outcome.lit_map.len(), 6);

        // The equivalence clauses should become tautological and be marked for deletion.
        assert!(outcome.clauses_to_delete.contains(&0) || outcome.clauses_to_delete.contains(&1));

        // (x1 ∨ x2) should rewrite to (x0 ∨ x2) under canonicalization.
        let replaced = outcome.clauses_to_replace.iter().find(|(idx, _)| *idx == 2);
        if let Some((_, new_lits)) = replaced {
            assert_eq!(new_lits.len(), 2);
            assert!(new_lits.contains(&Literal::positive(Variable(0))));
            assert!(new_lits.contains(&Literal::positive(Variable(2))));
        }
    }

    #[test]
    fn sweep_detects_unsat_via_scc() {
        // Unsat 2-SAT core:
        // (x ∨ y) ∧ (¬x ∨ y) ∧ (x ∨ ¬y) ∧ (¬x ∨ ¬y)
        // This forces y and ¬y (and equivalently x and ¬x) via implications.
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, false), lit(1, true)], false);
        clauses.add(&[lit(0, true), lit(1, false)], false);
        clauses.add(&[lit(0, false), lit(1, false)], false);

        let mut sweeper = Sweeper::new(2);
        let outcome = sweeper.sweep(&clauses);
        assert!(outcome.unsat);
        assert!(outcome.lit_map.is_empty());
    }
}
