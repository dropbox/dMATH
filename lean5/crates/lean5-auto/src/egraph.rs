//! E-graph for congruence closure (ported from egg/Z3)
//!
//! An E-graph (equivalence graph) is a data structure that efficiently maintains
//! equivalence classes of terms and supports congruence closure.
//!
//! # Core Concepts
//!
//! - **E-node**: A function application `f(a, b, c, ...)` where the arguments
//!   are e-class IDs rather than terms directly.
//! - **E-class**: An equivalence class containing multiple e-nodes that are
//!   all considered equal.
//! - **Congruence**: If `f(a) = f(b)` when `a = b` (for all arguments).
//!
//! # Usage
//!
//! ```ignore
//! let mut egraph = EGraph::new();
//!
//! // Add terms: f(a, b) and f(a, c)
//! let a = egraph.add_const("a");
//! let b = egraph.add_const("b");
//! let c = egraph.add_const("c");
//! let fab = egraph.add_app("f", vec![a, b]);
//! let fac = egraph.add_app("f", vec![a, c]);
//!
//! // Assert b = c
//! egraph.union(b, c);
//!
//! // By congruence, f(a, b) = f(a, c)
//! assert!(egraph.are_equal(fab, fac));
//! ```
//!
//! # Algorithm
//!
//! The congruence closure algorithm works by:
//! 1. Maintaining a union-find for e-class membership
//! 2. Using a hashcons to deduplicate e-nodes
//! 3. Propagating equalities via congruence when e-classes merge

use std::collections::{HashMap, VecDeque};

/// An e-class ID (equivalence class identifier)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EClassId(u32);

impl EClassId {
    /// Create a new e-class ID
    #[inline]
    pub fn new(id: u32) -> Self {
        EClassId(id)
    }

    /// Get the raw ID value
    #[inline]
    pub fn id(self) -> u32 {
        self.0
    }
}

/// A symbol (function name or constant name)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Symbol(String);

impl Symbol {
    /// Create a new symbol
    pub fn new(name: impl Into<String>) -> Self {
        Symbol(name.into())
    }

    /// Get the symbol name
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Symbol {
    fn from(s: &str) -> Self {
        Symbol::new(s)
    }
}

impl From<String> for Symbol {
    fn from(s: String) -> Self {
        Symbol(s)
    }
}

/// An e-node represents a function application.
/// The children are e-class IDs, not direct terms.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ENode {
    /// The function symbol (or constant name if no children)
    pub symbol: Symbol,
    /// Child e-class IDs (empty for constants)
    pub children: Vec<EClassId>,
}

/// Reason why two e-classes were merged
#[derive(Clone, Debug)]
pub enum MergeReason {
    /// Direct union call (external assertion)
    External,
    /// Congruence: two function applications became equal
    /// because their arguments became equal
    Congruence {
        /// The function symbol
        func: Symbol,
        /// The original children of the first e-node (before canonicalization)
        children1: Vec<EClassId>,
        /// The original children of the second e-node (before canonicalization)
        children2: Vec<EClassId>,
    },
}

/// Record of a merge operation in the E-graph
#[derive(Clone, Debug)]
pub struct MergeRecord {
    /// First e-class (before merge)
    pub ec1: EClassId,
    /// Second e-class (before merge)
    pub ec2: EClassId,
    /// Reason for the merge
    pub reason: MergeReason,
}

impl ENode {
    /// Create a constant (no children)
    pub fn constant(symbol: impl Into<Symbol>) -> Self {
        ENode {
            symbol: symbol.into(),
            children: Vec::new(),
        }
    }

    /// Create a function application
    pub fn app(symbol: impl Into<Symbol>, children: Vec<EClassId>) -> Self {
        ENode {
            symbol: symbol.into(),
            children,
        }
    }

    /// Check if this is a constant (no children)
    pub fn is_constant(&self) -> bool {
        self.children.is_empty()
    }

    /// Get the arity (number of children)
    pub fn arity(&self) -> usize {
        self.children.len()
    }

    /// Create a canonical version of this e-node using the given function
    /// to map e-class IDs to their canonical representatives
    #[must_use]
    pub fn canonicalize(&self, mut find: impl FnMut(EClassId) -> EClassId) -> Self {
        ENode {
            symbol: self.symbol.clone(),
            children: self.children.iter().map(|&id| find(id)).collect(),
        }
    }
}

/// An equivalence class containing e-nodes
#[derive(Clone, Debug)]
pub struct EClass {
    /// The canonical ID of this e-class
    pub id: EClassId,
    /// All e-nodes in this e-class
    pub nodes: Vec<ENode>,
    /// Parent e-nodes that reference this e-class
    /// (used for upward merging during congruence closure)
    pub parents: Vec<(ENode, EClassId)>,
}

impl EClass {
    /// Create a new e-class with a single node
    fn new(id: EClassId, node: ENode) -> Self {
        EClass {
            id,
            nodes: vec![node],
            parents: Vec::new(),
        }
    }
}

/// Union-Find data structure for efficient equivalence class operations
#[derive(Clone, Debug)]
struct UnionFind {
    /// Parent pointers (id -> parent id).
    /// If `parent[i] == i`, then `i` is a root.
    parent: Vec<u32>,
    /// Rank for union by rank optimization
    rank: Vec<u32>,
}

impl UnionFind {
    /// Create a new empty union-find
    fn new() -> Self {
        UnionFind {
            parent: Vec::new(),
            rank: Vec::new(),
        }
    }

    /// Make a new singleton set and return its ID
    fn make_set(&mut self) -> EClassId {
        let id = u32::try_from(self.parent.len())
            .expect("e-class count exceeded u32::MAX during make_set");
        self.parent.push(id);
        self.rank.push(0);
        EClassId(id)
    }

    /// Find the canonical representative of an e-class (with path compression)
    fn find(&mut self, id: EClassId) -> EClassId {
        let idx = id.0 as usize;
        if self.parent[idx] != id.0 {
            // Path compression
            let root = self.find(EClassId(self.parent[idx]));
            self.parent[idx] = root.0;
        }
        EClassId(self.parent[idx])
    }

    /// Find without mutation (for use in contexts where we can't mutate)
    fn find_const(&self, id: EClassId) -> EClassId {
        let mut current = id.0;
        while self.parent[current as usize] != current {
            current = self.parent[current as usize];
        }
        EClassId(current)
    }

    /// Union two e-classes, returns the new root
    /// Returns (new_root, merged_root) - merged_root is now pointing to new_root
    fn union(&mut self, a: EClassId, b: EClassId) -> (EClassId, EClassId) {
        let a_root = self.find(a);
        let b_root = self.find(b);

        if a_root == b_root {
            return (a_root, a_root);
        }

        let a_idx = a_root.0 as usize;
        let b_idx = b_root.0 as usize;

        // Union by rank
        if self.rank[a_idx] < self.rank[b_idx] {
            self.parent[a_idx] = b_root.0;
            (b_root, a_root)
        } else if self.rank[a_idx] > self.rank[b_idx] {
            self.parent[b_idx] = a_root.0;
            (a_root, b_root)
        } else {
            self.parent[b_idx] = a_root.0;
            self.rank[a_idx] += 1;
            (a_root, b_root)
        }
    }
}

/// The E-graph data structure
#[derive(Clone, Debug)]
pub struct EGraph {
    /// Union-find for e-class membership
    uf: UnionFind,
    /// E-classes indexed by their canonical ID
    classes: HashMap<EClassId, EClass>,
    /// Hashcons: maps canonical e-nodes to their e-class
    hashcons: HashMap<ENode, EClassId>,
    /// Pending merges for congruence closure (includes reason)
    pending: VecDeque<(EClassId, EClassId, MergeReason)>,
    /// Whether the egraph needs rebuilding (after unions)
    dirty: bool,
    /// History of merge operations (for proof reconstruction)
    merge_history: Vec<MergeRecord>,
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl EGraph {
    /// Create a new empty e-graph
    pub fn new() -> Self {
        EGraph {
            uf: UnionFind::new(),
            classes: HashMap::new(),
            hashcons: HashMap::new(),
            pending: VecDeque::new(),
            dirty: false,
            merge_history: Vec::new(),
        }
    }

    /// Get the canonical representative of an e-class ID
    pub fn find(&mut self, id: EClassId) -> EClassId {
        self.uf.find(id)
    }

    /// Get the canonical representative without mutation
    pub fn find_const(&self, id: EClassId) -> EClassId {
        self.uf.find_const(id)
    }

    /// Check if two e-class IDs are in the same equivalence class
    pub fn are_equal(&mut self, a: EClassId, b: EClassId) -> bool {
        self.find(a) == self.find(b)
    }

    /// Add a constant to the e-graph
    pub fn add_const(&mut self, name: impl Into<Symbol>) -> EClassId {
        let node = ENode::constant(name);
        self.add_node(node)
    }

    /// Add a function application to the e-graph
    pub fn add_app(&mut self, symbol: impl Into<Symbol>, children: Vec<EClassId>) -> EClassId {
        // Canonicalize children first
        let canonical_children: Vec<EClassId> = children.iter().map(|&id| self.find(id)).collect();
        let node = ENode::app(symbol, canonical_children);
        self.add_node(node)
    }

    /// Add an e-node to the e-graph
    fn add_node(&mut self, node: ENode) -> EClassId {
        // Canonicalize the node
        let canonical = node.canonicalize(|id| self.find(id));

        // Check if we already have this e-node (hashconsing)
        if let Some(&id) = self.hashcons.get(&canonical) {
            return self.find(id);
        }

        // Create a new e-class
        let id = self.uf.make_set();
        let eclass = EClass::new(id, canonical.clone());

        // Register this e-class as a parent of its children
        for &child_id in &canonical.children {
            let child_canonical = self.find(child_id);
            if let Some(child_class) = self.classes.get_mut(&child_canonical) {
                child_class.parents.push((canonical.clone(), id));
            }
        }

        self.hashcons.insert(canonical, id);
        self.classes.insert(id, eclass);

        id
    }

    /// Merge two e-classes (assert they are equal)
    /// This triggers congruence closure.
    pub fn union(&mut self, a: EClassId, b: EClassId) -> EClassId {
        self.union_with_reason(a, b, MergeReason::External)
    }

    /// Merge two e-classes with a specified reason
    /// This triggers congruence closure.
    pub fn union_with_reason(&mut self, a: EClassId, b: EClassId, reason: MergeReason) -> EClassId {
        let a = self.find(a);
        let b = self.find(b);

        if a == b {
            return a;
        }

        self.pending.push_back((a, b, reason));
        self.dirty = true;

        // Process all pending merges (congruence closure)
        self.rebuild();

        self.find(a)
    }

    /// Rebuild the e-graph after unions (congruence closure)
    fn rebuild(&mut self) {
        while let Some((a, b, reason)) = self.pending.pop_front() {
            self.do_union(a, b, reason);
        }
        self.dirty = false;

        // Rebuild hashcons with canonical e-nodes
        self.rebuild_hashcons();
    }

    /// Perform the actual union and propagate congruences
    fn do_union(&mut self, a: EClassId, b: EClassId, reason: MergeReason) {
        let a = self.uf.find(a);
        let b = self.uf.find(b);

        if a == b {
            return;
        }

        // Record this merge in history (before the union changes canonical IDs)
        self.merge_history.push(MergeRecord {
            ec1: a,
            ec2: b,
            reason,
        });

        // Perform union in union-find
        let (new_root, merged_root) = self.uf.union(a, b);

        // Merge the e-classes
        let merged_class = self
            .classes
            .remove(&merged_root)
            .expect("merged e-class must exist after union");
        let root_class = self
            .classes
            .get_mut(&new_root)
            .expect("root e-class must exist after union");

        // Merge nodes
        for node in merged_class.nodes {
            root_class.nodes.push(node);
        }

        // Collect parents that need to be checked for congruence
        let parents_to_check: Vec<(ENode, EClassId)> = merged_class.parents;

        // Merge parents
        for parent in &parents_to_check {
            root_class.parents.push(parent.clone());
        }

        // Check for congruent parents
        // Two parents are congruent if they have the same symbol and
        // all their children (after canonicalization) are equal
        for (node, parent_id) in parents_to_check {
            let canonical = node.canonicalize(|id| self.uf.find(id));

            // Check if this canonical node already exists
            if let Some(&existing_id) = self.hashcons.get(&canonical) {
                let existing_canonical = self.uf.find(existing_id);
                let parent_canonical = self.uf.find(parent_id);

                if existing_canonical != parent_canonical {
                    // Found a congruence - find the original e-nodes to capture children
                    // The congruence reason: node and existing_node have equal children
                    let existing_node = self.find_enode_in_class(existing_id);
                    let congruence_reason = MergeReason::Congruence {
                        func: node.symbol.clone(),
                        children1: node.children.clone(),
                        children2: existing_node
                            .map(|n| n.children.clone())
                            .unwrap_or_default(),
                    };

                    // Queue for merging with congruence reason
                    self.pending.push_back((
                        existing_canonical,
                        parent_canonical,
                        congruence_reason,
                    ));
                }
            }
        }
    }

    /// Find an e-node in an e-class (for congruence reason construction)
    fn find_enode_in_class(&self, id: EClassId) -> Option<&ENode> {
        let canonical = self.uf.find_const(id);
        self.classes.get(&canonical).and_then(|c| c.nodes.first())
    }

    /// Rebuild the hashcons table with canonical e-nodes
    fn rebuild_hashcons(&mut self) {
        let mut new_hashcons = HashMap::new();

        for (&id, class) in &self.classes {
            let canonical_id = self.uf.find_const(id);
            if canonical_id != id {
                continue; // Skip non-canonical classes
            }

            for node in &class.nodes {
                let canonical_node = node.canonicalize(|id| self.uf.find_const(id));
                new_hashcons.entry(canonical_node).or_insert(canonical_id);
            }
        }

        self.hashcons = new_hashcons;
    }

    /// Get an e-class by ID (returns the canonical class)
    pub fn get_class(&mut self, id: EClassId) -> Option<&EClass> {
        let canonical = self.find(id);
        self.classes.get(&canonical)
    }

    /// Get the number of e-classes
    pub fn num_classes(&self) -> usize {
        // Count canonical classes only
        self.classes
            .keys()
            .filter(|&&id| self.uf.find_const(id) == id)
            .count()
    }

    /// Get the total number of e-nodes
    pub fn num_nodes(&self) -> usize {
        self.classes
            .values()
            .filter(|c| self.uf.find_const(c.id) == c.id)
            .map(|c| c.nodes.len())
            .sum()
    }

    /// Get all canonical e-class IDs
    pub fn classes(&self) -> impl Iterator<Item = EClassId> + '_ {
        self.classes
            .keys()
            .copied()
            .filter(move |&id| self.uf.find_const(id) == id)
    }

    /// Check if the e-graph contains an e-node
    pub fn contains(&self, node: &ENode) -> bool {
        let canonical = node.canonicalize(|id| self.uf.find_const(id));
        self.hashcons.contains_key(&canonical)
    }

    /// Lookup an e-node in the e-graph
    pub fn lookup(&mut self, node: &ENode) -> Option<EClassId> {
        let canonical = node.canonicalize(|id| self.uf.find(id));
        if let Some(&id) = self.hashcons.get(&canonical) {
            Some(self.find(id))
        } else {
            None
        }
    }

    /// Get all e-nodes in an e-class
    pub fn get_nodes(&mut self, id: EClassId) -> Vec<ENode> {
        let canonical = self.find(id);
        self.classes
            .get(&canonical)
            .map(|c| c.nodes.clone())
            .unwrap_or_default()
    }

    /// Extract a representative term from an e-class (smallest by default)
    pub fn extract(&mut self, id: EClassId) -> Option<Term> {
        let canonical = self.find(id);

        // Clone nodes to avoid borrow issues
        let nodes: Vec<ENode> = self.classes.get(&canonical)?.nodes.clone();

        // Find the smallest node (by total term size)
        let mut best: Option<(usize, Term)> = None;

        for node in &nodes {
            if let Some(term) = self.extract_term(node) {
                let size = term.size();
                match &best {
                    None => best = Some((size, term)),
                    Some((best_size, _)) if size < *best_size => best = Some((size, term)),
                    _ => {}
                }
            }
        }

        best.map(|(_, t)| t)
    }

    /// Extract a term from an e-node
    fn extract_term(&mut self, node: &ENode) -> Option<Term> {
        if node.is_constant() {
            return Some(Term::Const(node.symbol.name().to_string()));
        }

        // Clone children to avoid borrow issues
        let children_ids: Vec<EClassId> = node.children.clone();
        let mut children = Vec::new();
        for child_id in children_ids {
            children.push(self.extract(child_id)?);
        }

        Some(Term::App(node.symbol.name().to_string(), children))
    }

    /// Clear the e-graph
    pub fn clear(&mut self) {
        self.uf = UnionFind::new();
        self.classes.clear();
        self.hashcons.clear();
        self.pending.clear();
        self.dirty = false;
        self.merge_history.clear();
    }

    /// Get the merge history for proof reconstruction
    pub fn merge_history(&self) -> &[MergeRecord] {
        &self.merge_history
    }

    /// Clear the merge history (useful after extracting for proofs)
    pub fn clear_merge_history(&mut self) {
        self.merge_history.clear();
    }
}

// ============================================================================
// E-Matching for Quantifier Instantiation
// ============================================================================

/// A pattern for E-matching
///
/// Patterns are used to find substitutions in an E-graph that make a pattern
/// term match some e-class. This is the core of trigger-based quantifier
/// instantiation in SMT solvers.
///
/// # Example
///
/// Pattern `f(?x, ?y)` matches `f(a, b)` with substitution `{?x → a, ?y → b}`
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pattern {
    /// Pattern variable `?x` - matches any e-class
    Var(String),
    /// Constant/function symbol with pattern children
    App(Symbol, Vec<Pattern>),
}

impl Pattern {
    /// Create a pattern variable
    pub fn var(name: impl Into<String>) -> Self {
        Pattern::Var(name.into())
    }

    /// Create an application pattern
    pub fn app(symbol: impl Into<Symbol>, children: Vec<Pattern>) -> Self {
        Pattern::App(symbol.into(), children)
    }

    /// Create a constant pattern (0-ary application)
    pub fn constant(name: impl Into<Symbol>) -> Self {
        Pattern::App(name.into(), vec![])
    }

    /// Collect all pattern variables in this pattern
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut Vec<String>) {
        match self {
            Pattern::Var(name) => {
                if !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            Pattern::App(_, children) => {
                for child in children {
                    child.collect_vars(vars);
                }
            }
        }
    }
}

/// A substitution mapping pattern variables to e-class IDs
#[derive(Clone, Debug, Default)]
pub struct Substitution {
    /// Mapping from variable name to e-class ID
    bindings: HashMap<String, EClassId>,
}

impl Substitution {
    /// Create an empty substitution
    pub fn new() -> Self {
        Substitution {
            bindings: HashMap::new(),
        }
    }

    /// Get the binding for a variable
    pub fn get(&self, var: &str) -> Option<EClassId> {
        self.bindings.get(var).copied()
    }

    /// Set the binding for a variable
    /// Returns false if the variable is already bound to a different class
    pub fn bind(&mut self, var: &str, class: EClassId) -> bool {
        if let Some(&existing) = self.bindings.get(var) {
            existing == class
        } else {
            self.bindings.insert(var.to_string(), class);
            true
        }
    }

    /// Check if a variable is bound
    pub fn is_bound(&self, var: &str) -> bool {
        self.bindings.contains_key(var)
    }

    /// Get all bindings
    pub fn bindings(&self) -> &HashMap<String, EClassId> {
        &self.bindings
    }

    /// Get the number of bindings
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Check if the substitution is empty
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

/// E-matching engine for pattern matching in E-graphs
///
/// This implements the classic E-matching algorithm from SMT solvers like Z3.
/// Given a pattern with variables, it finds all substitutions that make the
/// pattern equal to some e-class in the graph.
pub struct EMatcher<'a> {
    egraph: &'a EGraph,
}

impl<'a> EMatcher<'a> {
    /// Create a new E-matcher for the given e-graph
    pub fn new(egraph: &'a EGraph) -> Self {
        EMatcher { egraph }
    }

    /// Find all substitutions that make the pattern match some e-class
    ///
    /// Returns an iterator of (EClassId, Substitution) pairs where each
    /// substitution makes the pattern equal to the corresponding e-class.
    pub fn find_matches(&self, pattern: &Pattern) -> Vec<(EClassId, Substitution)> {
        let mut matches = Vec::new();

        // Try matching against every e-class in the graph
        for &root_id in self.egraph.classes.keys() {
            let canonical = self.egraph.find_const(root_id);
            // Only try each canonical representative once
            if canonical == root_id {
                let mut subst = Substitution::new();
                if self.match_pattern(pattern, canonical, &mut subst) {
                    matches.push((canonical, subst));
                }
            }
        }

        matches
    }

    /// Try to match a pattern against a specific e-class
    ///
    /// Returns true and fills in the substitution if matching succeeds.
    pub fn match_against(&self, pattern: &Pattern, target: EClassId) -> Option<Substitution> {
        let mut subst = Substitution::new();
        if self.match_pattern(pattern, self.egraph.find_const(target), &mut subst) {
            Some(subst)
        } else {
            None
        }
    }

    /// Core matching algorithm
    ///
    /// Recursively matches a pattern against an e-class, extending the substitution.
    fn match_pattern(&self, pattern: &Pattern, class: EClassId, subst: &mut Substitution) -> bool {
        let class = self.egraph.find_const(class);

        match pattern {
            Pattern::Var(name) => {
                // Pattern variable: either bind it or check consistency
                subst.bind(name, class)
            }
            Pattern::App(symbol, children) => {
                // Application pattern: find a matching e-node in the class
                if let Some(eclass) = self.egraph.classes.get(&class) {
                    for node in &eclass.nodes {
                        if node.symbol == *symbol && node.children.len() == children.len() {
                            // Try to match all children
                            let mut local_subst = subst.clone();
                            let mut all_match = true;

                            for (pat_child, &node_child) in
                                children.iter().zip(node.children.iter())
                            {
                                if !self.match_pattern(pat_child, node_child, &mut local_subst) {
                                    all_match = false;
                                    break;
                                }
                            }

                            if all_match {
                                // Commit the local substitution
                                *subst = local_subst;
                                return true;
                            }
                        }
                    }
                }
                false
            }
        }
    }

    /// Find all matches for a multi-pattern (conjunction of patterns)
    ///
    /// Multi-patterns are used to reduce the number of spurious instantiations.
    /// A substitution is only returned if it makes ALL patterns match.
    pub fn find_multi_matches(&self, patterns: &[Pattern]) -> Vec<Substitution> {
        if patterns.is_empty() {
            return vec![Substitution::new()];
        }

        // Start with matches for the first pattern
        let first_matches = self.find_matches(&patterns[0]);
        if patterns.len() == 1 {
            return first_matches.into_iter().map(|(_, s)| s).collect();
        }

        // Filter and extend with remaining patterns
        let mut results = Vec::new();
        for (_, subst) in first_matches {
            if self.extend_multi_match(&patterns[1..], subst, &mut results) {
                // Continue collecting all matches
            }
        }

        results
    }

    /// Extend a partial substitution with remaining patterns
    fn extend_multi_match(
        &self,
        patterns: &[Pattern],
        subst: Substitution,
        results: &mut Vec<Substitution>,
    ) -> bool {
        if patterns.is_empty() {
            results.push(subst);
            return true;
        }

        let pattern = &patterns[0];
        let remaining = &patterns[1..];

        // Find matches for this pattern that are consistent with current substitution
        let matches = self.find_matches(pattern);
        let mut found_any = false;

        for (_, new_subst) in matches {
            // Check if new substitution is compatible with existing one
            let mut combined = subst.clone();
            let mut compatible = true;

            for (var, class) in new_subst.bindings() {
                if !combined.bind(var, *class) {
                    compatible = false;
                    break;
                }
            }

            if compatible {
                found_any = true;
                self.extend_multi_match(remaining, combined, results);
            }
        }

        found_any
    }
}

/// A trigger pattern for quantifier instantiation
///
/// Triggers are patterns that, when matched, indicate a useful instantiation
/// of a quantified formula. They are typically sub-terms of the quantified body.
#[derive(Clone, Debug)]
pub struct Trigger {
    /// The pattern(s) that must match (multi-trigger if > 1)
    pub patterns: Vec<Pattern>,
}

impl Trigger {
    /// Create a single-pattern trigger
    pub fn single(pattern: Pattern) -> Self {
        Trigger {
            patterns: vec![pattern],
        }
    }

    /// Create a multi-pattern trigger
    pub fn multi(patterns: Vec<Pattern>) -> Self {
        Trigger { patterns }
    }

    /// Get all variables bound by this trigger
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        for p in &self.patterns {
            for v in p.variables() {
                if !vars.contains(&v) {
                    vars.push(v);
                }
            }
        }
        vars
    }
}

/// A term (for extraction)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Term {
    /// A constant
    Const(String),
    /// A function application
    App(String, Vec<Term>),
}

impl Term {
    /// Get the size of this term (number of nodes)
    pub fn size(&self) -> usize {
        match self {
            Term::Const(_) => 1,
            Term::App(_, children) => 1 + children.iter().map(Term::size).sum::<usize>(),
        }
    }

    /// Pretty print the term
    pub fn to_string_pretty(&self) -> String {
        match self {
            Term::Const(name) => name.clone(),
            Term::App(name, children) if children.is_empty() => name.clone(),
            Term::App(name, children) => {
                let args: Vec<String> = children.iter().map(Term::to_string_pretty).collect();
                format!("{}({})", name, args.join(", "))
            }
        }
    }
}

/// Builder for adding complex terms to an e-graph
pub struct TermBuilder<'a> {
    egraph: &'a mut EGraph,
}

impl<'a> TermBuilder<'a> {
    /// Create a new term builder
    pub fn new(egraph: &'a mut EGraph) -> Self {
        TermBuilder { egraph }
    }

    /// Add a term to the e-graph
    pub fn add_term(&mut self, term: &Term) -> EClassId {
        match term {
            Term::Const(name) => self.egraph.add_const(name.as_str()),
            Term::App(name, children) => {
                let child_ids: Vec<EClassId> = children.iter().map(|c| self.add_term(c)).collect();
                self.egraph.add_app(name.as_str(), child_ids)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_const() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let a2 = egraph.add_const("a");

        // Same constant should return same e-class
        assert_eq!(a, a2);
        // Different constants should be different
        assert_ne!(a, b);
    }

    #[test]
    fn test_add_app() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let fa = egraph.add_app("f", vec![a]);
        let fb = egraph.add_app("f", vec![b]);
        let fa2 = egraph.add_app("f", vec![a]);

        // Same application should return same e-class
        assert_eq!(fa, fa2);
        // Different applications should be different
        assert_ne!(fa, fb);
    }

    #[test]
    fn test_union() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");

        assert!(!egraph.are_equal(a, b));

        egraph.union(a, b);

        assert!(egraph.are_equal(a, b));
    }

    #[test]
    fn test_congruence_simple() {
        let mut egraph = EGraph::new();

        // f(a) and f(b)
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let fa = egraph.add_app("f", vec![a]);
        let fb = egraph.add_app("f", vec![b]);

        // Initially different
        assert!(!egraph.are_equal(fa, fb));

        // After asserting a = b, f(a) = f(b) by congruence
        egraph.union(a, b);

        assert!(egraph.are_equal(fa, fb));
    }

    #[test]
    fn test_congruence_nested() {
        let mut egraph = EGraph::new();

        // g(f(a)) and g(f(b))
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let fa = egraph.add_app("f", vec![a]);
        let fb = egraph.add_app("f", vec![b]);
        let gfa = egraph.add_app("g", vec![fa]);
        let gfb = egraph.add_app("g", vec![fb]);

        // Initially different
        assert!(!egraph.are_equal(gfa, gfb));

        // After asserting a = b:
        // f(a) = f(b) by congruence
        // g(f(a)) = g(f(b)) by congruence
        egraph.union(a, b);

        assert!(egraph.are_equal(fa, fb));
        assert!(egraph.are_equal(gfa, gfb));
    }

    #[test]
    fn test_congruence_multiarg() {
        let mut egraph = EGraph::new();

        // f(a, c) and f(b, c)
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let c = egraph.add_const("c");
        let fac = egraph.add_app("f", vec![a, c]);
        let fbc = egraph.add_app("f", vec![b, c]);

        assert!(!egraph.are_equal(fac, fbc));

        egraph.union(a, b);

        assert!(egraph.are_equal(fac, fbc));
    }

    #[test]
    fn test_congruence_chain() {
        let mut egraph = EGraph::new();

        // a = b, b = c -> a = c (transitivity)
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let c = egraph.add_const("c");
        let fa = egraph.add_app("f", vec![a]);
        let fc = egraph.add_app("f", vec![c]);

        egraph.union(a, b);
        egraph.union(b, c);

        assert!(egraph.are_equal(a, c));
        assert!(egraph.are_equal(fa, fc));
    }

    #[test]
    fn test_hashcons_after_union() {
        let mut egraph = EGraph::new();

        let a = egraph.add_const("a");
        let b = egraph.add_const("b");

        // Union a and b
        egraph.union(a, b);

        // Now add f(a) - should be same as f(b) by hashcons
        let fa = egraph.add_app("f", vec![a]);
        let fb = egraph.add_app("f", vec![b]);

        assert!(egraph.are_equal(fa, fb));
    }

    #[test]
    fn test_extract_const() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");

        let term = egraph.extract(a).unwrap();
        assert_eq!(term, Term::Const("a".to_string()));
    }

    #[test]
    fn test_extract_app() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let fab = egraph.add_app("f", vec![a, b]);

        let term = egraph.extract(fab).unwrap();
        assert_eq!(
            term,
            Term::App(
                "f".to_string(),
                vec![Term::Const("a".to_string()), Term::Const("b".to_string())]
            )
        );
    }

    #[test]
    fn test_num_classes() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let c = egraph.add_const("c");

        assert_eq!(egraph.num_classes(), 3);

        egraph.union(a, b);
        assert_eq!(egraph.num_classes(), 2);

        egraph.union(b, c);
        assert_eq!(egraph.num_classes(), 1);
    }

    #[test]
    fn test_term_builder() {
        let mut egraph = EGraph::new();

        let term = Term::App(
            "f".to_string(),
            vec![
                Term::Const("a".to_string()),
                Term::App("g".to_string(), vec![Term::Const("b".to_string())]),
            ],
        );

        let id = {
            let mut builder = TermBuilder::new(&mut egraph);
            builder.add_term(&term)
        };

        let extracted = egraph.extract(id).unwrap();
        assert_eq!(extracted, term);
    }

    #[test]
    fn test_contains() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let _fa = egraph.add_app("f", vec![a]);

        assert!(egraph.contains(&ENode::constant("a")));
        assert!(egraph.contains(&ENode::app("f", vec![a])));
        assert!(!egraph.contains(&ENode::constant("b")));
    }

    #[test]
    fn test_lookup() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let fa = egraph.add_app("f", vec![a]);

        assert_eq!(egraph.lookup(&ENode::constant("a")), Some(a));
        assert_eq!(egraph.lookup(&ENode::app("f", vec![a])), Some(fa));
        assert_eq!(egraph.lookup(&ENode::constant("b")), None);
    }

    #[test]
    fn test_clear() {
        let mut egraph = EGraph::new();
        egraph.add_const("a");
        egraph.add_const("b");

        assert_eq!(egraph.num_classes(), 2);

        egraph.clear();

        assert_eq!(egraph.num_classes(), 0);
    }

    #[test]
    fn test_complex_congruence() {
        let mut egraph = EGraph::new();

        // Build: h(f(a), g(b)) and h(f(c), g(d))
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let c = egraph.add_const("c");
        let d = egraph.add_const("d");

        let fa = egraph.add_app("f", vec![a]);
        let fc = egraph.add_app("f", vec![c]);
        let gb = egraph.add_app("g", vec![b]);
        let gd = egraph.add_app("g", vec![d]);

        let h1 = egraph.add_app("h", vec![fa, gb]);
        let h2 = egraph.add_app("h", vec![fc, gd]);

        // Initially different
        assert!(!egraph.are_equal(h1, h2));

        // Assert a = c and b = d
        egraph.union(a, c);
        egraph.union(b, d);

        // By congruence: f(a) = f(c), g(b) = g(d)
        // Therefore: h(f(a), g(b)) = h(f(c), g(d))
        assert!(egraph.are_equal(fa, fc));
        assert!(egraph.are_equal(gb, gd));
        assert!(egraph.are_equal(h1, h2));
    }

    #[test]
    fn test_self_loop() {
        let mut egraph = EGraph::new();

        // f(f(a)) where we assert f(a) = a
        let a = egraph.add_const("a");
        let fa = egraph.add_app("f", vec![a]);
        let ffa = egraph.add_app("f", vec![fa]);

        // Assert f(a) = a
        egraph.union(fa, a);

        // Now f(f(a)) should equal f(a) (which equals a)
        assert!(egraph.are_equal(a, fa));
        assert!(egraph.are_equal(fa, ffa));
        assert!(egraph.are_equal(a, ffa));
    }

    #[test]
    fn test_term_size() {
        let t1 = Term::Const("a".to_string());
        assert_eq!(t1.size(), 1);

        let t2 = Term::App("f".to_string(), vec![Term::Const("a".to_string())]);
        assert_eq!(t2.size(), 2);

        let t3 = Term::App(
            "f".to_string(),
            vec![
                Term::Const("a".to_string()),
                Term::App("g".to_string(), vec![Term::Const("b".to_string())]),
            ],
        );
        assert_eq!(t3.size(), 4);
    }

    #[test]
    fn test_term_pretty() {
        let t = Term::App(
            "f".to_string(),
            vec![
                Term::Const("a".to_string()),
                Term::App("g".to_string(), vec![Term::Const("b".to_string())]),
            ],
        );
        assert_eq!(t.to_string_pretty(), "f(a, g(b))");
    }

    // E-matching tests
    #[test]
    fn test_ematch_constant() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let _b = egraph.add_const("b");

        let matcher = EMatcher::new(&egraph);

        // Pattern: constant "a"
        let pattern = Pattern::constant("a");
        let matches = matcher.find_matches(&pattern);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0, a);
    }

    #[test]
    fn test_ematch_variable() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");

        let matcher = EMatcher::new(&egraph);

        // Pattern: ?x (matches everything)
        let pattern = Pattern::var("x");
        let matches = matcher.find_matches(&pattern);

        // Should match both a and b
        assert_eq!(matches.len(), 2);
        let classes: Vec<_> = matches.iter().map(|(c, _)| *c).collect();
        assert!(classes.contains(&a));
        assert!(classes.contains(&b));
    }

    #[test]
    fn test_ematch_app_pattern() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let fa = egraph.add_app("f", vec![a]);
        let fb = egraph.add_app("f", vec![b]);
        let _ga = egraph.add_app("g", vec![a]);

        let matcher = EMatcher::new(&egraph);

        // Pattern: f(?x)
        let pattern = Pattern::app("f", vec![Pattern::var("x")]);
        let matches = matcher.find_matches(&pattern);

        // Should match f(a) and f(b)
        assert_eq!(matches.len(), 2);
        let classes: Vec<_> = matches.iter().map(|(c, _)| *c).collect();
        assert!(classes.contains(&fa));
        assert!(classes.contains(&fb));

        // Check substitutions
        for (class, subst) in &matches {
            let x_val = subst.get("x").unwrap();
            if *class == fa {
                assert_eq!(x_val, a);
            } else {
                assert_eq!(x_val, b);
            }
        }
    }

    #[test]
    fn test_ematch_nested_pattern() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let fa = egraph.add_app("f", vec![a]);
        let gfa = egraph.add_app("g", vec![fa]);
        let _gb = egraph.add_app("g", vec![b]);

        let matcher = EMatcher::new(&egraph);

        // Pattern: g(f(?x))
        let pattern = Pattern::app("g", vec![Pattern::app("f", vec![Pattern::var("x")])]);
        let matches = matcher.find_matches(&pattern);

        // Should match g(f(a))
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0, gfa);
        assert_eq!(matches[0].1.get("x").unwrap(), a);
    }

    #[test]
    fn test_ematch_multi_arg() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let c = egraph.add_const("c");
        let fab = egraph.add_app("f", vec![a, b]);
        let _fac = egraph.add_app("f", vec![a, c]);

        let matcher = EMatcher::new(&egraph);

        // Pattern: f(?x, ?y)
        let pattern = Pattern::app("f", vec![Pattern::var("x"), Pattern::var("y")]);
        let matches = matcher.find_matches(&pattern);

        // Should match both f(a, b) and f(a, c)
        assert_eq!(matches.len(), 2);

        // Check one specific match
        let fab_match = matches.iter().find(|(c, _)| *c == fab).unwrap();
        assert_eq!(fab_match.1.get("x").unwrap(), a);
        assert_eq!(fab_match.1.get("y").unwrap(), b);
    }

    #[test]
    fn test_ematch_repeated_var() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let faa = egraph.add_app("f", vec![a, a]);
        let _fab = egraph.add_app("f", vec![a, b]);

        let matcher = EMatcher::new(&egraph);

        // Pattern: f(?x, ?x) - same variable twice
        let pattern = Pattern::app("f", vec![Pattern::var("x"), Pattern::var("x")]);
        let matches = matcher.find_matches(&pattern);

        // Should only match f(a, a), not f(a, b)
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0, faa);
        assert_eq!(matches[0].1.get("x").unwrap(), a);
    }

    #[test]
    fn test_ematch_with_equivalence() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let fa = egraph.add_app("f", vec![a]);
        let _fb = egraph.add_app("f", vec![b]);

        // Make a = b
        egraph.union(a, b);

        let matcher = EMatcher::new(&egraph);

        // Pattern: f(?x)
        let pattern = Pattern::app("f", vec![Pattern::var("x")]);
        let matches = matcher.find_matches(&pattern);

        // After union, f(a) and f(b) are in the same class
        // Should return 1 match (canonical representative)
        assert_eq!(matches.len(), 1);
        // The canonical class should be f(a)'s canonical rep
        let canon_fa = egraph.find(fa);
        assert_eq!(matches[0].0, canon_fa);
    }

    #[test]
    fn test_ematch_multi_pattern() {
        let mut egraph = EGraph::new();
        let a = egraph.add_const("a");
        let b = egraph.add_const("b");
        let c = egraph.add_const("c");
        let fa = egraph.add_app("f", vec![a]);
        let ga = egraph.add_app("g", vec![a]);
        let _fb = egraph.add_app("f", vec![b]);
        let _gc = egraph.add_app("g", vec![c]);

        let matcher = EMatcher::new(&egraph);

        // Multi-pattern: {f(?x), g(?x)} - same ?x in both patterns
        let patterns = vec![
            Pattern::app("f", vec![Pattern::var("x")]),
            Pattern::app("g", vec![Pattern::var("x")]),
        ];
        let matches = matcher.find_multi_matches(&patterns);

        // Only ?x = a satisfies both f(?x) and g(?x)
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].get("x").unwrap(), a);

        // Verify f(a) and g(a) exist
        let _ = (fa, ga); // Just to silence warnings
    }

    #[test]
    fn test_trigger_variables() {
        // Pattern: f(?x, g(?y))
        let pattern = Pattern::app(
            "f",
            vec![
                Pattern::var("x"),
                Pattern::app("g", vec![Pattern::var("y")]),
            ],
        );
        let trigger = Trigger::single(pattern);

        let vars = trigger.variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
    }

    #[test]
    fn test_pattern_variables() {
        // Pattern with repeated var: f(?x, ?x, ?y)
        let pattern = Pattern::app(
            "f",
            vec![Pattern::var("x"), Pattern::var("x"), Pattern::var("y")],
        );
        let vars = pattern.variables();
        // Should deduplicate
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
    }
}
