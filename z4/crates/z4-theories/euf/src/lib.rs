//! Z4 EUF - Equality and Uninterpreted Functions theory solver
//!
//! Implements congruence closure for equality reasoning.

#![warn(missing_docs)]
#![warn(clippy::all)]
// Complex types for congruence closure lookup maps
#![allow(clippy::type_complexity)]

use hashbrown::{HashMap, HashSet};
use std::collections::{BTreeMap, VecDeque};
use z4_core::term::{Symbol, TermData, TermId, TermStore};
use z4_core::{Sort, TheoryLit, TheoryPropagation, TheoryResult, TheorySolver};

/// Model for uninterpreted sorts - maps sort names to element enumerations
pub type SortModel = HashMap<String, Vec<String>>;

/// Function table entry: maps argument values to result value
pub type FunctionTable = Vec<(Vec<String>, String)>;

/// Model for uninterpreted functions
#[derive(Debug, Clone, Default)]
pub struct EufModel {
    /// Element representatives for each uninterpreted sort
    /// Maps sort name -> list of distinct element names
    pub sort_elements: SortModel,
    /// Maps term IDs to their model element name
    pub term_values: HashMap<TermId, String>,
    /// Function interpretations as finite tables
    /// Maps function name -> list of (arg_values, result_value) entries
    pub function_tables: HashMap<String, FunctionTable>,
}

/// Union-Find structure for equivalence classes
pub struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u32>,
}

impl UnionFind {
    /// Create a new union-find with n elements
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // n is bounded by term count which fits in u32
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    fn reset(&mut self) {
        for (idx, p) in self.parent.iter_mut().enumerate() {
            *p = idx as u32;
        }
        self.rank.fill(0);
    }

    fn ensure_size(&mut self, n: usize) {
        if n <= self.parent.len() {
            return;
        }
        let start = self.parent.len() as u32;
        self.parent
            .extend(start..start + (n - self.parent.len()) as u32);
        self.rank.resize(n, 0);
    }

    /// Find the representative of an element (with path compression)
    pub fn find(&mut self, x: u32) -> u32 {
        if self.parent[x as usize] != x {
            self.parent[x as usize] = self.find(self.parent[x as usize]);
        }
        self.parent[x as usize]
    }

    /// Union two elements
    pub fn union(&mut self, x: u32, y: u32) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx != ry {
            match self.rank[rx as usize].cmp(&self.rank[ry as usize]) {
                std::cmp::Ordering::Less => {
                    self.parent[rx as usize] = ry;
                }
                std::cmp::Ordering::Greater => {
                    self.parent[ry as usize] = rx;
                }
                std::cmp::Ordering::Equal => {
                    self.parent[ry as usize] = rx;
                    self.rank[rx as usize] += 1;
                }
            }
        }
    }
}

/// Reason for an edge in the equality graph
#[derive(Clone, Debug)]
enum EqualityReason {
    /// Direct equality assertion: the TermId of the (= a b) term
    Direct(TermId),
    /// Congruence: f(a1,...,an) = f(b1,...,bn) because ai = bi for all i
    /// Stores (term1, term2, [(arg1_i, arg2_i), ...])
    /// Note: term1/term2 stored for future proof generation but not yet used
    #[allow(dead_code)]
    Congruence(TermId, TermId, Vec<(TermId, TermId)>),
}

/// Cached metadata for a function application term
struct FuncAppMeta {
    term_id: u32,
    /// Pre-computed hash of (symbol, result_sort) for fast signature lookup
    func_hash: u64,
    /// Argument term ids (not representatives - those change)
    args: Vec<u32>,
}

/// EUF theory solver
pub struct EufSolver<'a> {
    terms: &'a TermStore,
    uf: UnionFind,
    assigns: HashMap<TermId, bool>,
    trail: Vec<(TermId, Option<bool>)>,
    scopes: Vec<usize>,
    dirty: bool,
    /// Equality graph: maps (min(a,b), max(a,b)) -> reason why a = b
    equality_edges: HashMap<(u32, u32), EqualityReason>,
    /// Pre-computed list of function application terms with their argument ids
    /// This avoids iterating all terms and cloning during congruence closure
    func_apps: Vec<FuncAppMeta>,
    /// Whether func_apps has been initialized
    func_apps_init: bool,
    /// Hash of the set of true equalities from the last rebuild_closure
    /// Used to detect if closure needs rebuilding after soft_reset
    last_eq_hash: u64,
    /// Whether we're in "soft reset" mode where we can potentially skip rebuilding
    soft_mode: bool,
    /// Direct conflict detected when assigning term to both true and false
    /// Stores (term, positive_assignment) - the conflict is between term=true and term=false
    pending_conflict: Option<TermId>,
}

impl<'a> EufSolver<'a> {
    /// Create a new EUF solver
    #[must_use]
    pub fn new(terms: &'a TermStore) -> Self {
        EufSolver {
            terms,
            uf: UnionFind::new(terms.len()),
            assigns: HashMap::new(),
            trail: Vec::new(),
            scopes: Vec::new(),
            dirty: true,
            equality_edges: HashMap::new(),
            func_apps: Vec::new(),
            func_apps_init: false,
            last_eq_hash: 0,
            soft_mode: false,
            pending_conflict: None,
        }
    }

    /// Initialize the func_apps cache if not already done
    fn init_func_apps(&mut self) {
        if self.func_apps_init {
            return;
        }

        self.func_apps.clear();
        for idx in 0..self.terms.len() {
            let term_id = TermId(idx as u32);
            if let TermData::App(sym, args) = self.terms.get(term_id) {
                if !Self::is_builtin_symbol(sym) && !args.is_empty() {
                    // Pre-compute hash of (symbol, sort)
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    sym.hash(&mut hasher);
                    self.terms.sort(term_id).hash(&mut hasher);
                    let func_hash = hasher.finish();

                    self.func_apps.push(FuncAppMeta {
                        term_id: idx as u32,
                        func_hash,
                        args: args.iter().map(|t| t.0).collect(),
                    });
                }
            }
        }
        self.func_apps_init = true;
    }

    /// Current number of terms managed by the solver.
    #[must_use]
    pub fn num_terms(&self) -> usize {
        self.uf.parent.len()
    }

    fn record_assignment(&mut self, term: TermId, value: bool) {
        let debug = std::env::var("Z4_DEBUG_EUF").is_ok();
        match self.assigns.get(&term).copied() {
            Some(prev) if prev == value => {
                if debug {
                    eprintln!(
                        "[EUF] record_assignment: term {} = {} (unchanged)",
                        term.0, value
                    );
                }
            }
            Some(prev) => {
                // Conflicting assignment: term was prev, now assigning !prev
                if debug {
                    eprintln!(
                        "[EUF] record_assignment: CONFLICT term {} was {} now {}",
                        term.0, prev, value
                    );
                }
                // Record the conflict - don't overwrite, let check() handle it
                self.pending_conflict = Some(term);
            }
            None => {
                self.trail.push((term, None));
                self.assigns.insert(term, value);
                self.dirty = true;
                if debug && self.decode_eq(term).is_some() {
                    eprintln!("[EUF] record_assignment: eq term {} = {} (NEW, dirty=true, total_assigns={})", term.0, value, self.assigns.len());
                }
            }
        }
    }

    fn is_builtin_symbol(sym: &Symbol) -> bool {
        matches!(sym.name(), "and" | "or" | "=" | "distinct")
    }

    fn decode_eq(&self, term: TermId) -> Option<(TermId, TermId)> {
        match self.terms.get(term) {
            TermData::App(sym, args) if sym.name() == "=" && args.len() == 2 => {
                Some((args[0], args[1]))
            }
            _ => None,
        }
    }

    fn decode_distinct(&self, term: TermId) -> Option<&[TermId]> {
        match self.terms.get(term) {
            TermData::App(sym, args) if sym.name() == "distinct" => Some(args),
            _ => None,
        }
    }

    /// Decode NOT(inner) -> inner term
    fn decode_not(&self, term: TermId) -> Option<TermId> {
        match self.terms.get(term) {
            TermData::Not(inner) => Some(*inner),
            _ => None,
        }
    }

    /// Helper to create a canonical edge key (smaller id first)
    fn edge_key(a: u32, b: u32) -> (u32, u32) {
        if a < b {
            (a, b)
        } else {
            (b, a)
        }
    }

    /// Compute a hash of the current set of true equalities
    fn compute_eq_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Collect and sort equality term IDs that are true
        let mut eq_ids: Vec<u32> = self
            .assigns
            .iter()
            .filter(|(_, &v)| v)
            .filter_map(|(&term, _)| {
                if self.decode_eq(term).is_some() {
                    Some(term.0)
                } else {
                    None
                }
            })
            .collect();
        eq_ids.sort_unstable();

        for id in eq_ids {
            id.hash(&mut hasher);
        }
        hasher.finish()
    }

    fn rebuild_closure(&mut self) {
        let debug = std::env::var("Z4_DEBUG_EUF").is_ok();

        if !self.dirty {
            if debug {
                eprintln!("[EUF] rebuild_closure: not dirty, skipping!");
            }
            return;
        }

        // Check if we can skip rebuilding (same equalities as before)
        if self.soft_mode {
            let current_eq_hash = self.compute_eq_hash();
            if current_eq_hash == self.last_eq_hash && self.last_eq_hash != 0 {
                if debug {
                    eprintln!("[EUF] rebuild_closure: same equalities, skipping rebuild!");
                }
                self.dirty = false;
                return;
            }
            self.last_eq_hash = current_eq_hash;
        }

        // Initialize func_apps cache on first use
        self.init_func_apps();

        self.uf.ensure_size(self.terms.len());
        self.uf.reset();
        self.equality_edges.clear();

        // Apply asserted equalities and record edges.
        for (&lit_term, &value) in &self.assigns {
            if !value {
                continue;
            }
            let Some((lhs, rhs)) = self.decode_eq(lit_term) else {
                continue;
            };
            // Only union well-typed equalities.
            if self.terms.sort(lhs) != self.terms.sort(rhs) {
                continue;
            }
            if debug {
                eprintln!(
                    "[EUF] Union: term {} with term {} (from eq {})",
                    lhs.0, rhs.0, lit_term.0
                );
            }
            // Record the direct equality edge before union
            let key = Self::edge_key(lhs.0, rhs.0);
            self.equality_edges
                .entry(key)
                .or_insert(EqualityReason::Direct(lit_term));
            self.uf.union(lhs.0, rhs.0);
        }

        // Congruence closure using cached func_apps and efficient signature
        // Signature: (func_hash, arg_rep1, arg_rep2, ...)
        loop {
            let mut changed = false;
            // Maps (func_hash, arg_reps...) -> (term_id, arg_ids)
            let mut sig_table: HashMap<(u64, Vec<u32>), (u32, Vec<u32>)> = HashMap::new();

            for meta in &self.func_apps {
                // Compute representative-based signature
                let reps: Vec<u32> = meta.args.iter().map(|&a| self.uf.find(a)).collect();
                let sig = (meta.func_hash, reps);

                if let Some(&(other_id, ref other_args)) = sig_table.get(&sig) {
                    let r1 = self.uf.find(meta.term_id);
                    let r2 = self.uf.find(other_id);
                    if r1 != r2 {
                        // Record congruence edge
                        let arg_pairs: Vec<(TermId, TermId)> = meta
                            .args
                            .iter()
                            .zip(other_args.iter())
                            .map(|(&a, &b)| (TermId(a), TermId(b)))
                            .collect();
                        let edge_key = Self::edge_key(meta.term_id, other_id);
                        self.equality_edges
                            .entry(edge_key)
                            .or_insert(EqualityReason::Congruence(
                                TermId(meta.term_id),
                                TermId(other_id),
                                arg_pairs,
                            ));
                        self.uf.union(r1, r2);
                        changed = true;
                    }
                } else {
                    sig_table.insert(sig, (meta.term_id, meta.args.clone()));
                }
            }

            if !changed {
                break;
            }
        }

        self.dirty = false;

        if debug {
            // Show which x_i terms (2, 5, 8, 11, ...) are connected
            let mut reps: Vec<(u32, u32)> = Vec::new();
            for i in 0..31 {
                let term = (i * 3 + 2) as u32; // x_i terms start at 2 and go 2, 5, 8, 11, ...
                if (term as usize) < self.uf.parent.len() {
                    let rep = self.uf.find(term);
                    reps.push((term, rep));
                }
            }
            eprintln!("[EUF] x_i reps: {:?}", reps);
            eprintln!("[EUF] equality_edges: {} edges", self.equality_edges.len());
        }
    }

    fn all_true_equalities(&self) -> Vec<TheoryLit> {
        let mut out = Vec::new();
        for (&t, &value) in &self.assigns {
            if !value {
                continue;
            }
            if self.decode_eq(t).is_some() {
                out.push(TheoryLit::new(t, true));
            }
        }
        out
    }

    /// Explain why two terms are equal using BFS over the equality graph.
    /// Returns the minimal set of direct equality assertions needed.
    fn explain(&mut self, a: TermId, b: TermId) -> Vec<TheoryLit> {
        let debug = std::env::var("Z4_DEBUG_EUF").is_ok();

        if a == b {
            return Vec::new();
        }

        // BFS to find path from a to b in the equality graph
        // We need to find paths through the actual term graph, not just representatives
        let mut visited: HashSet<u32> = HashSet::new();
        let mut queue: VecDeque<u32> = VecDeque::new();
        let mut parent: HashMap<u32, (u32, EqualityReason)> = HashMap::new();

        queue.push_back(a.0);
        visited.insert(a.0);

        // Build adjacency from equality_edges
        let mut adj: HashMap<u32, Vec<(u32, EqualityReason)>> = HashMap::new();
        for (&(x, y), reason) in &self.equality_edges {
            adj.entry(x).or_default().push((y, reason.clone()));
            adj.entry(y).or_default().push((x, reason.clone()));
        }

        while let Some(curr) = queue.pop_front() {
            if curr == b.0 {
                break;
            }
            if let Some(neighbors) = adj.get(&curr) {
                for (next, reason) in neighbors {
                    if !visited.contains(next) {
                        visited.insert(*next);
                        parent.insert(*next, (curr, reason.clone()));
                        queue.push_back(*next);
                    }
                }
            }
        }

        // Reconstruct path and collect reasons
        let mut reasons = Vec::new();
        let mut curr = b.0;

        if !parent.contains_key(&curr) && curr != a.0 {
            // No path found - fall back to all equalities
            // This shouldn't happen if rebuild_closure was called properly
            if debug {
                eprintln!(
                    "[EUF EXPLAIN] No path from {} to {}, falling back",
                    a.0, b.0
                );
            }
            return self.all_true_equalities();
        }

        while curr != a.0 {
            if let Some((prev, reason)) = parent.get(&curr) {
                self.collect_reason_literals(reason, &mut reasons);
                curr = *prev;
            } else {
                break;
            }
        }

        if debug {
            eprintln!(
                "[EUF EXPLAIN] Path from {} to {} needs {} reasons",
                a.0,
                b.0,
                reasons.len()
            );
        }

        // Deduplicate
        reasons.sort_by_key(|l| (l.term.0, l.value));
        reasons.dedup_by_key(|l| (l.term.0, l.value));

        reasons
    }

    /// Recursively collect the direct equality literals for a reason
    fn collect_reason_literals(&mut self, reason: &EqualityReason, out: &mut Vec<TheoryLit>) {
        match reason {
            EqualityReason::Direct(eq_term) => {
                out.push(TheoryLit::new(*eq_term, true));
            }
            EqualityReason::Congruence(_, _, arg_pairs) => {
                // For each argument pair that are in the same equivalence class,
                // we need to explain why they're equal
                for &(arg_a, arg_b) in arg_pairs {
                    if self.uf.find(arg_a.0) == self.uf.find(arg_b.0) && arg_a != arg_b {
                        // Recursively explain why arg_a = arg_b
                        let sub_reasons = self.explain(arg_a, arg_b);
                        out.extend(sub_reasons);
                    }
                }
            }
        }
    }

    fn conflict_with_reasons(&self, mut reasons: Vec<TheoryLit>, lit: TheoryLit) -> TheoryResult {
        reasons.push(lit);
        // Keep the clause reasonably small by removing duplicates.
        reasons.sort_by_key(|l| (l.term, l.value));
        reasons.dedup_by_key(|l| (l.term, l.value));
        TheoryResult::Unsat(reasons)
    }

    /// Extract a model after solving (call after check() returns Sat)
    ///
    /// Returns an `EufModel` containing:
    /// - Element representatives for uninterpreted sorts
    /// - Term-to-element mappings
    /// - Function table interpretations for uninterpreted functions
    pub fn extract_model(&mut self) -> EufModel {
        // Ensure congruence closure is up-to-date
        self.rebuild_closure();

        let mut model = EufModel::default();

        // Collect equivalence class representatives per sort
        // Maps (sort_name, representative_id) -> element_name
        let mut rep_to_elem: HashMap<(String, u32), String> = HashMap::new();
        // Counter for generating element names per sort
        let mut sort_counters: HashMap<String, usize> = HashMap::new();

        // First pass: assign element names to representatives
        for idx in 0..self.terms.len() {
            let term_id = TermId(idx as u32);
            let sort = self.terms.sort(term_id);

            // Only process uninterpreted sorts
            let sort_name = match sort {
                Sort::Uninterpreted(name) => name.clone(),
                _ => continue,
            };

            let rep = self.uf.find(term_id.0);
            let key = (sort_name.clone(), rep);

            if !rep_to_elem.contains_key(&key) {
                let counter = sort_counters.entry(sort_name.clone()).or_insert(0);
                let elem_name = format!("@{}!{}", sort_name, counter);
                *counter += 1;

                rep_to_elem.insert(key.clone(), elem_name.clone());

                // Add to sort_elements
                model
                    .sort_elements
                    .entry(sort_name)
                    .or_default()
                    .push(elem_name);
            }
        }

        // Second pass: map each term to its element name
        for idx in 0..self.terms.len() {
            let term_id = TermId(idx as u32);
            let sort = self.terms.sort(term_id);

            let sort_name = match sort {
                Sort::Uninterpreted(name) => name.clone(),
                _ => continue,
            };

            let rep = self.uf.find(term_id.0);
            let key = (sort_name, rep);

            if let Some(elem_name) = rep_to_elem.get(&key) {
                model.term_values.insert(term_id, elem_name.clone());
            }
        }

        // Third pass: build function tables for uninterpreted functions
        // Use BTreeMap for deterministic ordering
        let mut fn_entries: BTreeMap<String, Vec<(Vec<String>, String, TermId)>> = BTreeMap::new();
        // Separate tracking for predicates (Bool-returning functions)
        let mut pred_entries: BTreeMap<String, Vec<(Vec<String>, String, TermId)>> =
            BTreeMap::new();

        for idx in 0..self.terms.len() {
            let term_id = TermId(idx as u32);

            // Get function applications
            let (sym, args) = match self.terms.get(term_id) {
                TermData::App(sym, args) if !Self::is_builtin_symbol(sym) => {
                    (sym.clone(), args.clone())
                }
                _ => continue,
            };

            // Skip nullary functions (constants) - handled in second pass
            if args.is_empty() {
                continue;
            }

            let result_sort = self.terms.sort(term_id);

            // Get element names for arguments
            let arg_values: Vec<String> = args
                .iter()
                .map(|&arg| {
                    model
                        .term_values
                        .get(&arg)
                        .cloned()
                        .unwrap_or_else(|| format!("@?{}", arg.0))
                })
                .collect();

            // Handle predicates (Bool-sorted functions) using assigns
            if matches!(result_sort, Sort::Bool) {
                // Get value from assigns (SAT model propagated to theory)
                let result_value = match self.assigns.get(&term_id) {
                    Some(true) => "true".to_string(),
                    Some(false) => "false".to_string(),
                    None => "false".to_string(), // Default unassigned to false
                };

                pred_entries.entry(sym.to_string()).or_default().push((
                    arg_values,
                    result_value,
                    term_id,
                ));
                continue;
            }

            // Get element name for result (non-Bool functions)
            let result_value = model
                .term_values
                .get(&term_id)
                .cloned()
                .unwrap_or_else(|| format!("@?{}", term_id.0));

            fn_entries.entry(sym.to_string()).or_default().push((
                arg_values,
                result_value,
                term_id,
            ));
        }

        // Deduplicate function table entries by representative
        for (fn_name, entries) in fn_entries {
            let mut seen: HashMap<Vec<String>, String> = HashMap::new();
            let mut table = Vec::new();

            for (args, result, _term_id) in entries {
                // Use first occurrence for each argument combination
                if !seen.contains_key(&args) {
                    seen.insert(args.clone(), result.clone());
                    table.push((args, result));
                }
            }

            if !table.is_empty() {
                model.function_tables.insert(fn_name, table);
            }
        }

        // Deduplicate predicate table entries
        for (pred_name, entries) in pred_entries {
            let mut seen: HashMap<Vec<String>, String> = HashMap::new();
            let mut table = Vec::new();

            for (args, result, _term_id) in entries {
                // Use first occurrence for each argument combination
                if !seen.contains_key(&args) {
                    seen.insert(args.clone(), result.clone());
                    table.push((args, result));
                }
            }

            if !table.is_empty() {
                model.function_tables.insert(pred_name, table);
            }
        }

        model
    }
}

impl TheorySolver for EufSolver<'_> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        let debug = std::env::var("Z4_DEBUG_EUF").is_ok();

        // Unwrap NOT: NOT(inner)=true means inner=false
        let (term, val) = if let Some(inner) = self.decode_not(literal) {
            if debug {
                eprintln!(
                    "[EUF ASSERT] NOT term {} unwrapped to inner {} with value {}",
                    literal.0, inner.0, !value
                );
            }
            (inner, !value)
        } else {
            (literal, value)
        };

        if debug {
            // Check if it's an equality
            if let Some((lhs, rhs)) = self.decode_eq(term) {
                eprintln!(
                    "[EUF ASSERT] eq term {} (terms {} == {}) = {}",
                    term.0, lhs.0, rhs.0, val
                );
            }
        }
        self.record_assignment(term, val);
    }

    fn check(&mut self) -> TheoryResult {
        let debug = std::env::var("Z4_DEBUG_EUF").is_ok();
        if debug {
            eprintln!(
                "[EUF] check() called: dirty={}, assigns={}",
                self.dirty,
                self.assigns.len()
            );
        }

        // 0) Direct conflict: term assigned both true and false
        if let Some(conflict_term) = self.pending_conflict.take() {
            if debug {
                eprintln!("[EUF CHECK] Direct conflict on term {}", conflict_term.0);
            }
            // Return conflict clause: {term=true, term=false} -> both are in conflict
            return TheoryResult::Unsat(vec![
                TheoryLit::new(conflict_term, true),
                TheoryLit::new(conflict_term, false),
            ]);
        }

        self.rebuild_closure();

        if debug {
            let eq_count = self
                .assigns
                .iter()
                .filter(|(t, &v)| v && self.decode_eq(**t).is_some())
                .count();
            eprintln!("[EUF CHECK] {} equalities asserted true", eq_count);
        }

        // 1) Conflicts from explicit disequalities (= ...)=false.
        // Collect first to avoid borrow issues
        let diseqs: Vec<(TermId, TermId, TermId)> = self
            .assigns
            .iter()
            .filter(|(_, &v)| !v)
            .filter_map(|(&lit_term, _)| {
                let (lhs, rhs) = self.decode_eq(lit_term)?;
                if self.terms.sort(lhs) != self.terms.sort(rhs) {
                    return None;
                }
                Some((lit_term, lhs, rhs))
            })
            .collect();

        for (lit_term, lhs, rhs) in diseqs {
            let lhs_rep = self.uf.find(lhs.0);
            let rhs_rep = self.uf.find(rhs.0);
            if debug {
                eprintln!(
                    "[EUF CHECK] Diseq: term {} != term {} (reps: {} vs {})",
                    lhs.0, rhs.0, lhs_rep, rhs_rep
                );
            }
            if lhs_rep == rhs_rep {
                if debug {
                    eprintln!("[EUF CHECK] CONFLICT DETECTED!");
                }
                // Use explain() to get minimal reasons instead of all_true_equalities()
                let reasons = self.explain(lhs, rhs);
                if debug {
                    eprintln!(
                        "[EUF CHECK] Conflict explained with {} reasons (vs {} all equalities)",
                        reasons.len(),
                        self.all_true_equalities().len()
                    );
                }
                return self.conflict_with_reasons(reasons, TheoryLit::new(lit_term, false));
            }
        }

        // 2) Conflicts from (distinct ...)=true.
        // Collect to avoid borrow issues
        let distincts: Vec<(TermId, Vec<TermId>)> = self
            .assigns
            .iter()
            .filter(|(_, &v)| v)
            .filter_map(|(&lit_term, _)| {
                let args = self.decode_distinct(lit_term)?;
                Some((lit_term, args.to_vec()))
            })
            .collect();

        if debug {
            eprintln!("[EUF CHECK] Found {} distinct constraints", distincts.len());
        }

        for (lit_term, args) in distincts {
            if debug {
                eprintln!(
                    "[EUF CHECK] Checking distinct term {} with {} args",
                    lit_term.0,
                    args.len()
                );
            }
            for i in 0..args.len() {
                for j in (i + 1)..args.len() {
                    let rep_i = self.uf.find(args[i].0);
                    let rep_j = self.uf.find(args[j].0);
                    if debug {
                        eprintln!(
                            "[EUF CHECK] args[{}]={} (rep={}) vs args[{}]={} (rep={})",
                            i, args[i].0, rep_i, j, args[j].0, rep_j
                        );
                    }
                    if rep_i == rep_j {
                        // Use explain() for minimal conflict clause
                        let reasons = self.explain(args[i], args[j]);
                        return self.conflict_with_reasons(reasons, TheoryLit::new(lit_term, true));
                    }
                }
            }
        }

        // 3) Congruence for Boolean-valued terms: merged terms must share truth value.
        // Collect to avoid borrow issues
        let bool_terms: Vec<(TermId, bool)> = self
            .assigns
            .iter()
            .filter_map(|(&term, &val)| {
                if self.terms.sort(term) != &Sort::Bool {
                    return None;
                }
                let enforce = match self.terms.get(term) {
                    TermData::Var(_, _) => true,
                    TermData::App(sym, _) => !Self::is_builtin_symbol(sym),
                    _ => false,
                };
                if !enforce {
                    return None;
                }
                Some((term, val))
            })
            .collect();

        let mut rep_value: HashMap<u32, (TermId, bool)> = HashMap::new();
        for (term, val) in bool_terms {
            let rep = self.uf.find(term.0);
            if let Some(&(other_term, other_val)) = rep_value.get(&rep) {
                if other_val != val {
                    // Use explain() for minimal conflict clause
                    let mut reasons = self.explain(term, other_term);
                    reasons.push(TheoryLit::new(other_term, other_val));
                    return self.conflict_with_reasons(reasons, TheoryLit::new(term, val));
                }
            } else {
                rep_value.insert(rep, (term, val));
            }
        }

        if debug {
            eprintln!("[EUF CHECK] Returning SAT");
        }

        TheoryResult::Sat
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        let debug = std::env::var("Z4_DEBUG_EUF").is_ok();

        // Rebuild closure if needed
        if self.dirty {
            self.rebuild_closure();
        }

        let mut propagations = Vec::new();

        // Check for implied equalities
        // Collect potential propagations first to avoid borrow issues
        let potential_props: Vec<(TermId, TermId, TermId)> = (0..self.terms.len())
            .filter_map(|idx| {
                let term_id = TermId(idx as u32);
                let (lhs, rhs) = self.decode_eq(term_id)?;
                if self.assigns.contains_key(&term_id) {
                    return None;
                }
                if self.terms.sort(lhs) != self.terms.sort(rhs) {
                    return None;
                }
                let lhs_rep = self.uf.find(lhs.0);
                let rhs_rep = self.uf.find(rhs.0);
                if lhs_rep == rhs_rep {
                    Some((term_id, lhs, rhs))
                } else {
                    None
                }
            })
            .collect();

        for (term_id, lhs, rhs) in potential_props {
            // Use explain() for minimal propagation reasons
            let reasons = self.explain(lhs, rhs);
            if debug {
                eprintln!(
                    "[EUF PROPAGATE] Propagating eq {} = true (terms {} == {}) with {} reasons",
                    term_id.0,
                    lhs.0,
                    rhs.0,
                    reasons.len()
                );
            }
            propagations.push(TheoryPropagation {
                literal: TheoryLit::new(term_id, true),
                reason: reasons,
            });
        }

        propagations
    }

    fn push(&mut self) {
        self.scopes.push(self.trail.len());
    }

    fn pop(&mut self) {
        let Some(mark) = self.scopes.pop() else {
            return;
        };
        while self.trail.len() > mark {
            let (term, prev) = self.trail.pop().expect("trail length checked above");
            match prev {
                Some(v) => {
                    self.assigns.insert(term, v);
                }
                None => {
                    self.assigns.remove(&term);
                }
            }
        }
        self.dirty = true;
        self.pending_conflict = None;
    }

    fn reset(&mut self) {
        let debug = std::env::var("Z4_DEBUG_EUF").is_ok();
        if debug {
            eprintln!(
                "[EUF] reset() called, clearing {} assigns",
                self.assigns.len()
            );
        }
        self.assigns.clear();
        self.trail.clear();
        self.scopes.clear();
        self.uf.ensure_size(self.terms.len());
        self.uf.reset();
        self.equality_edges.clear();
        self.last_eq_hash = 0;
        self.soft_mode = false;
        self.dirty = true;
        self.pending_conflict = None;
    }

    fn soft_reset(&mut self) {
        let debug = std::env::var("Z4_DEBUG_EUF").is_ok();
        if debug {
            eprintln!(
                "[EUF] soft_reset() called, clearing {} assigns",
                self.assigns.len()
            );
        }
        // Clear assignments but preserve closure state
        // The closure will be validated/rebuilt lazily in rebuild_closure
        self.assigns.clear();
        self.trail.clear();
        self.scopes.clear();
        self.soft_mode = true;
        self.dirty = true;
        self.pending_conflict = None;
    }
}

// ============================================================================
// Kani Verification Harnesses
// ============================================================================
//
// These proofs verify the core invariants of the Union-Find and EUF solver:
// 1. Union-Find correctness: find, union, path compression
// 2. Congruence closure soundness
// 3. Push/pop state consistency

#[cfg(kani)]
mod verification {
    use super::*;

    // ========================================================================
    // Union-Find Invariants
    // ========================================================================

    /// After union(x, y), find(x) == find(y)
    #[kani::proof]
    fn proof_union_makes_equivalent() {
        let mut uf = UnionFind::new(8);

        let x: u32 = kani::any();
        let y: u32 = kani::any();
        kani::assume(x < 8 && y < 8);

        uf.union(x, y);

        let rx = uf.find(x);
        let ry = uf.find(y);
        assert!(rx == ry, "After union, find(x) must equal find(y)");
    }

    /// find is idempotent: find(find(x)) == find(x)
    #[kani::proof]
    fn proof_find_idempotent() {
        let mut uf = UnionFind::new(8);

        let x: u32 = kani::any();
        kani::assume(x < 8);

        let r1 = uf.find(x);
        let r2 = uf.find(r1);
        assert!(r1 == r2, "find must be idempotent");
    }

    /// find returns a valid representative (within bounds)
    #[kani::proof]
    fn proof_find_in_bounds() {
        let mut uf = UnionFind::new(8);

        let x: u32 = kani::any();
        kani::assume(x < 8);

        let r = uf.find(x);
        assert!(r < 8, "find must return a valid index");
    }

    /// Transitivity: if union(x,y) and union(y,z), then find(x) == find(z)
    #[kani::proof]
    fn proof_union_transitive() {
        let mut uf = UnionFind::new(8);

        let x: u32 = kani::any();
        let y: u32 = kani::any();
        let z: u32 = kani::any();
        kani::assume(x < 8 && y < 8 && z < 8);

        uf.union(x, y);
        uf.union(y, z);

        let rx = uf.find(x);
        let rz = uf.find(z);
        assert!(rx == rz, "Union must be transitive");
    }

    /// Reset restores initial state where each element is its own representative
    #[kani::proof]
    fn proof_reset_restores_identity() {
        let mut uf = UnionFind::new(8);

        // Do some unions
        let x: u32 = kani::any();
        let y: u32 = kani::any();
        kani::assume(x < 8 && y < 8 && x != y);

        uf.union(x, y);

        // Before reset, x and y have same representative
        assert!(uf.find(x) == uf.find(y));

        // After reset, each element is its own representative
        uf.reset();

        let rx = uf.find(x);
        let ry = uf.find(y);
        assert!(rx == x, "After reset, find(x) == x");
        assert!(ry == y, "After reset, find(y) == y");
    }

    /// ensure_size extends union-find without breaking existing structure
    #[kani::proof]
    fn proof_ensure_size_preserves_structure() {
        let mut uf = UnionFind::new(4);

        let x: u32 = kani::any();
        let y: u32 = kani::any();
        kani::assume(x < 4 && y < 4);

        uf.union(x, y);
        let rep_before = uf.find(x);

        // Extend the union-find
        uf.ensure_size(8);

        // Existing structure preserved
        let rep_after = uf.find(x);
        assert!(
            rep_before == rep_after,
            "ensure_size must preserve structure"
        );

        // New elements are their own representatives
        let new_elem: u32 = kani::any();
        kani::assume(new_elem >= 4 && new_elem < 8);
        assert!(
            uf.find(new_elem) == new_elem,
            "New elements are self-representative"
        );
    }

    /// Rank bounds are maintained: rank[x] < log2(n)
    #[kani::proof]
    fn proof_rank_bounded() {
        let mut uf = UnionFind::new(8);

        // Do several unions
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        let c: u32 = kani::any();
        kani::assume(a < 8 && b < 8 && c < 8);

        uf.union(a, b);
        uf.union(b, c);

        // Check all ranks are bounded (for n=8, max rank is 3)
        for i in 0..8 {
            assert!(uf.rank[i] <= 3, "Rank must be bounded by log2(n)");
        }
    }

    // ========================================================================
    // EUF Solver State Invariants
    // ========================================================================

    /// Push/pop preserves solver consistency
    #[kani::proof]
    fn proof_push_pop_consistency() {
        // Use minimal term store to keep verification tractable
        let mut store = z4_core::term::TermStore::new();
        let u = z4_core::Sort::Uninterpreted("U".to_string());

        let a = store.mk_var("a", u.clone());
        let b = store.mk_var("b", u.clone());
        let eq_ab = store.mk_eq(a, b);

        let mut euf = EufSolver::new(&store);

        // Initial state: no assignments
        let initial_assigns = euf.assigns.len();

        // Push, assert, check
        euf.push();
        euf.assert_literal(eq_ab, true);
        assert!(
            euf.assigns.len() > initial_assigns,
            "Assignment should be recorded"
        );

        // Pop should restore
        euf.pop();
        assert!(
            euf.assigns.len() == initial_assigns,
            "Pop should restore state"
        );
    }

    /// Multiple push/pop maintains stack discipline
    #[kani::proof]
    fn proof_nested_push_pop() {
        let mut store = z4_core::term::TermStore::new();
        let u = z4_core::Sort::Uninterpreted("U".to_string());

        let a = store.mk_var("a", u.clone());
        let b = store.mk_var("b", u.clone());
        let c = store.mk_var("c", u.clone());
        let eq_ab = store.mk_eq(a, b);
        let eq_bc = store.mk_eq(b, c);

        let mut euf = EufSolver::new(&store);

        // Level 0
        let l0_assigns = euf.assigns.len();

        // Push to level 1, assert
        euf.push();
        euf.assert_literal(eq_ab, true);
        let l1_assigns = euf.assigns.len();

        // Push to level 2, assert
        euf.push();
        euf.assert_literal(eq_bc, true);

        // Pop to level 1
        euf.pop();
        assert!(euf.assigns.len() == l1_assigns, "Pop to level 1");

        // Pop to level 0
        euf.pop();
        assert!(euf.assigns.len() == l0_assigns, "Pop to level 0");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euf_conflict_from_implied_equality() {
        let mut store = TermStore::new();
        let u = Sort::Uninterpreted("U".to_string());

        let a = store.mk_var("a", u.clone());
        let b = store.mk_var("b", u.clone());
        let c = store.mk_var("c", u.clone());

        let eq_ac = store.mk_eq(a, c);
        let eq_cb = store.mk_eq(c, b);
        let eq_ab = store.mk_eq(a, b);

        let mut euf = EufSolver::new(&store);
        euf.assert_literal(eq_ac, true);
        euf.assert_literal(eq_cb, true);
        euf.assert_literal(eq_ab, false);

        assert!(matches!(euf.check(), TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_euf_distinct_conflict() {
        let mut store = TermStore::new();
        let u = Sort::Uninterpreted("U".to_string());

        let a = store.mk_var("a", u.clone());
        let b = store.mk_var("b", u.clone());

        let eq_ab = store.mk_eq(a, b);
        let distinct_ab = store.mk_distinct(vec![a, b]);

        let mut euf = EufSolver::new(&store);
        euf.assert_literal(eq_ab, true);
        euf.assert_literal(distinct_ab, true);

        assert!(matches!(euf.check(), TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_euf_predicate_congruence_conflict() {
        let mut store = TermStore::new();
        let u = Sort::Uninterpreted("U".to_string());

        let a = store.mk_var("a", u.clone());
        let b = store.mk_var("b", u.clone());

        let eq_ab = store.mk_eq(a, b);
        let p_a = store.mk_app(Symbol::named("p"), vec![a], Sort::Bool);
        let p_b = store.mk_app(Symbol::named("p"), vec![b], Sort::Bool);

        let mut euf = EufSolver::new(&store);
        euf.assert_literal(eq_ab, true);
        euf.assert_literal(p_a, true);
        euf.assert_literal(p_b, false);

        assert!(matches!(euf.check(), TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_euf_predicate_congruence_sat() {
        let mut store = TermStore::new();
        let u = Sort::Uninterpreted("U".to_string());

        let a = store.mk_var("a", u.clone());
        let b = store.mk_var("b", u.clone());

        let eq_ab = store.mk_eq(a, b);
        let p_a = store.mk_app(Symbol::named("p"), vec![a], Sort::Bool);
        let p_b = store.mk_app(Symbol::named("p"), vec![b], Sort::Bool);

        let mut euf = EufSolver::new(&store);
        euf.assert_literal(eq_ab, true);
        euf.assert_literal(p_a, true);
        euf.assert_literal(p_b, true);

        assert!(matches!(euf.check(), TheoryResult::Sat));
    }
}
