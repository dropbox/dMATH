//! Tableau graph construction for liveness checking
//!
//! This module implements the tableau construction algorithm from
//! Manna & Pnueli's "Temporal Verification of Reactive Systems: Safety"
//! (Chapter 5, pages 405-452).
//!
//! # Overview
//!
//! A tableau is a graph where:
//! - Each node represents a "particle" (set of formulas that must hold)
//! - Edges represent valid transitions (what must hold in successor states)
//! - Initial nodes are particles derived from the formula being checked
//!
//! # Algorithm
//!
//! 1. Start with the formula F (already in positive normal form)
//! 2. Compute the particle closure of {F}
//! 3. For each particle, compute implied successors
//! 4. Build graph edges based on successor particles

use super::live_expr::LiveExpr;
use std::collections::HashSet;

/// A particle is a maximal consistent set of formulas from the closure
///
/// In the tableau, each particle represents a possible "state" of the
/// formula satisfaction. A behavior satisfies a temporal formula iff
/// there's a path through the tableau whose particles are consistent
/// with the behavior states.
#[derive(Debug, Clone)]
pub struct Particle {
    /// The formulas in this particle
    formulas: Vec<LiveExpr>,
}

impl Particle {
    /// Create a new particle with a single formula
    pub fn new(formula: LiveExpr) -> Self {
        Self {
            formulas: vec![formula],
        }
    }

    /// Create a particle from multiple formulas
    pub fn from_vec(formulas: Vec<LiveExpr>) -> Self {
        Self { formulas }
    }

    /// Check if this particle is empty
    pub fn is_empty(&self) -> bool {
        self.formulas.is_empty()
    }

    /// Get the formulas in this particle
    pub fn formulas(&self) -> &[LiveExpr] {
        &self.formulas
    }

    /// Add a formula to this particle
    pub fn add(&mut self, formula: LiveExpr) {
        self.formulas.push(formula);
    }

    /// Compute the implied successors for this particle
    ///
    /// For each formula in the particle:
    /// - []P implies ()[]P (always P means next state also has always P)
    /// - <>P implies P \/ ()<>P (eventually either now or later)
    /// - ()P implies P must hold in successor
    ///
    /// Returns the set of formulas that must be satisfiable in successors
    pub fn implied_successors(&self) -> Particle {
        let mut successor_formulas = Vec::new();

        for formula in &self.formulas {
            if let LiveExpr::Next(inner) = formula {
                // ()P means P must hold in successor
                successor_formulas.push((**inner).clone());
            }
        }

        Particle::from_vec(successor_formulas)
    }

    /// Check if this particle contains a formula equivalent to the given one
    fn contains_equivalent(&self, target: &LiveExpr) -> bool {
        self.formulas.iter().any(|f| f.structurally_equal(target))
    }

    /// Check if this particle contains a formula (structural equality).
    pub fn member(&self, target: &LiveExpr) -> bool {
        self.contains_equivalent(target)
    }

    /// Check if this particle fulfills a promise of the form `<>r`.
    ///
    /// A particle `A` fulfills `<>r` iff either:
    /// - `<>r ∉ A`, or
    /// - `r ∈ A`
    ///
    /// See TLC: `TBPar.isFulfilling`.
    pub fn is_fulfilling(&self, promise: &LiveExpr) -> bool {
        let LiveExpr::Eventually(body) = promise else {
            return true;
        };

        !self.member(promise) || self.member(body)
    }

    /// Extract state predicates from this particle
    ///
    /// These are the predicates that must be checked against actual states
    /// to determine if a state is consistent with this tableau node.
    pub fn state_predicates(&self) -> Vec<&LiveExpr> {
        self.formulas
            .iter()
            .filter(|f| {
                matches!(
                    f.level(),
                    super::live_expr::ExprLevel::Constant | super::live_expr::ExprLevel::State
                )
            })
            .collect()
    }

    /// Check structural equality between particles
    pub fn equals(&self, other: &Particle) -> bool {
        if self.formulas.len() != other.formulas.len() {
            return false;
        }
        // Check all formulas match (order-independent)
        self.formulas
            .iter()
            .all(|f| other.formulas.iter().any(|g| f.structurally_equal(g)))
            && other
                .formulas
                .iter()
                .all(|f| self.formulas.iter().any(|g| f.structurally_equal(g)))
    }
}

/// A node in the tableau graph
#[derive(Debug, Clone)]
pub struct TableauNode {
    /// The particle (set of formulas) for this node
    particle: Particle,
    /// Indices of successor nodes (HashSet for O(1) membership check)
    successors: HashSet<usize>,
    /// Index of this node in the tableau
    index: usize,
    /// State predicates extracted from particle for quick checking
    state_preds: Vec<LiveExpr>,
}

impl TableauNode {
    /// Create a new tableau node
    pub fn new(particle: Particle, index: usize) -> Self {
        let state_preds = particle.state_predicates().into_iter().cloned().collect();
        Self {
            particle,
            successors: HashSet::new(),
            index,
            state_preds,
        }
    }

    /// Get the particle
    pub fn particle(&self) -> &Particle {
        &self.particle
    }

    /// Get successor indices
    pub fn successors(&self) -> &HashSet<usize> {
        &self.successors
    }

    /// Add a successor (O(1) with HashSet)
    pub fn add_successor(&mut self, idx: usize) {
        self.successors.insert(idx);
    }

    /// Get the node index
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get state predicates
    pub fn state_preds(&self) -> &[LiveExpr] {
        &self.state_preds
    }

    /// Check if this is a self-loop node with empty particle (accepting)
    pub fn is_accepting(&self) -> bool {
        self.particle.is_empty()
            && self.successors.len() == 1
            && self.successors.contains(&self.index)
    }
}

/// The tableau graph for a temporal formula
#[derive(Debug, Clone)]
pub struct Tableau {
    /// The original temporal formula
    formula: LiveExpr,
    /// All nodes in the tableau
    nodes: Vec<TableauNode>,
    /// Number of initial nodes (nodes reachable from the formula)
    init_count: usize,
}

impl Tableau {
    /// Construct a tableau from a temporal formula
    ///
    /// The formula should already be in positive normal form.
    pub fn new(formula: LiveExpr) -> Self {
        // Start with the initial particle containing just the formula
        let init_particle = Particle::new(formula.clone());

        // Compute the particle closure
        let init_particles = particle_closure(init_particle);

        // Create initial nodes
        let mut nodes: Vec<TableauNode> = init_particles
            .into_iter()
            .enumerate()
            .map(|(i, p)| TableauNode::new(p, i))
            .collect();

        let init_count = nodes.len();

        // Build edges: compute successors for each node
        let mut i = 0;
        while i < nodes.len() {
            let implied = nodes[i].particle.implied_successors();
            let successor_particles = particle_closure(implied);

            for succ_particle in successor_particles {
                // Find or create node for this particle
                let succ_idx = find_or_create_node(&mut nodes, succ_particle);
                nodes[i].add_successor(succ_idx);
            }
            i += 1;
        }

        Self {
            formula,
            nodes,
            init_count,
        }
    }

    /// Get the original formula
    pub fn formula(&self) -> &LiveExpr {
        &self.formula
    }

    /// Get all nodes
    pub fn nodes(&self) -> &[TableauNode] {
        &self.nodes
    }

    /// Get a specific node
    pub fn node(&self, idx: usize) -> Option<&TableauNode> {
        self.nodes.get(idx)
    }

    /// Get number of initial nodes
    pub fn init_count(&self) -> usize {
        self.init_count
    }

    /// Check if a node is an initial node
    pub fn is_init_node(&self, idx: usize) -> bool {
        idx < self.init_count
    }

    /// Get the number of nodes
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tableau is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Find an existing node with matching particle, or create a new one
fn find_or_create_node(nodes: &mut Vec<TableauNode>, particle: Particle) -> usize {
    for (i, node) in nodes.iter().enumerate() {
        if node.particle.equals(&particle) {
            return i;
        }
    }
    let idx = nodes.len();
    nodes.push(TableauNode::new(particle, idx));
    idx
}

/// Compute the particle closure of a particle
///
/// This implements the expansion rules from Manna & Pnueli (p. 452):
/// - α-formulas (conjunctions): both conjuncts must be in particle
/// - β-formulas (disjunctions): at least one disjunct in particle
/// - []P: P and ()[]P
/// - <>P: P or ()<>P
///
/// Returns all maximal consistent particles derived from the input.
fn particle_closure(initial: Particle) -> Vec<Particle> {
    let mut results = Vec::new();
    expand_particle(initial, &mut results);
    results
}

/// Expand a particle using tableau expansion rules
fn expand_particle(particle: Particle, results: &mut Vec<Particle>) {
    // Find first expandable formula
    for formula in particle.formulas.iter() {
        match formula {
            LiveExpr::And(conjuncts) => {
                // α-rule: expand conjunction - add all conjuncts
                // KEEP the original And formula so that <>P membership checks work
                // when P is a conjunction
                let mut new_formulas = particle.formulas.clone();
                // Don't remove - just add the conjuncts that aren't already present
                let mut any_added = false;
                for conjunct in conjuncts {
                    if !particle.member(conjunct) {
                        new_formulas.push(conjunct.clone());
                        any_added = true;
                    }
                }
                if any_added {
                    expand_particle(Particle::from_vec(new_formulas), results);
                    return;
                }
                // All conjuncts already present, continue to next formula
            }

            LiveExpr::Or(disjuncts) => {
                // β-rule: expand disjunction - branch on each disjunct
                // Check if any disjunct is already present - if so, skip expansion
                let mut any_present = false;
                for disjunct in disjuncts {
                    if particle.member(disjunct) {
                        any_present = true;
                        break;
                    }
                }
                if !any_present {
                    // No disjunct present yet - branch on each
                    for disjunct in disjuncts {
                        let mut new_formulas = particle.formulas.clone();
                        new_formulas.push(disjunct.clone());
                        expand_particle(Particle::from_vec(new_formulas), results);
                    }
                    return;
                }
                // Some disjunct already present, continue to next formula
            }

            LiveExpr::Always(inner) => {
                // α-rule: []P expands to P and ()[]P, but keep []P in the particle.
                // (TLC: TBPar.particleClosure alpha expansion)
                let mut new_formulas = particle.formulas.clone();
                let p_now = (**inner).clone();
                let next_always = LiveExpr::next(LiveExpr::always((**inner).clone()));

                let mut changed = false;
                if !particle.member(&p_now) {
                    new_formulas.push(p_now);
                    changed = true;
                }
                if !particle.member(&next_always) {
                    new_formulas.push(next_always);
                    changed = true;
                }

                if changed {
                    expand_particle(Particle::from_vec(new_formulas), results);
                    return;
                }
            }

            LiveExpr::Eventually(inner) => {
                // β-rule: <>P expands to P or ()<>P, but keep <>P in the particle.
                // Only branch when neither alternative is present yet.
                // (TLC: TBPar.particleClosure beta expansion)
                let p_now = (**inner).clone();
                let next_eventually = LiveExpr::next(LiveExpr::eventually((**inner).clone()));

                if !particle.member(&p_now) && !particle.member(&next_eventually) {
                    let mut branch1 = particle.formulas.clone();
                    branch1.push(p_now);
                    expand_particle(Particle::from_vec(branch1), results);

                    let mut branch2 = particle.formulas.clone();
                    branch2.push(next_eventually);
                    expand_particle(Particle::from_vec(branch2), results);
                    return;
                }
            }

            // Atoms and Next don't expand further
            LiveExpr::Bool(_)
            | LiveExpr::StatePred { .. }
            | LiveExpr::ActionPred { .. }
            | LiveExpr::Enabled { .. }
            | LiveExpr::StateChanged { .. }
            | LiveExpr::Not(_)
            | LiveExpr::Next(_) => {}
        }
    }

    // No more expansions possible - check consistency and add to results
    if is_locally_consistent(&particle) && !particle_exists(results, &particle) {
        results.push(particle);
    }
}

/// Check if a particle is locally consistent
///
/// A particle is inconsistent if it contains both P and ~P for some atom P.
fn is_locally_consistent(particle: &Particle) -> bool {
    // Check for TRUE/FALSE contradictions
    let has_true = particle
        .formulas
        .iter()
        .any(|f| matches!(f, LiveExpr::Bool(true)));
    let has_false = particle
        .formulas
        .iter()
        .any(|f| matches!(f, LiveExpr::Bool(false)));
    if has_true && has_false {
        return false;
    }
    if has_false {
        return false; // FALSE is always inconsistent
    }

    // Check for P and ~P contradictions
    for (i, f) in particle.formulas.iter().enumerate() {
        if let LiveExpr::Not(inner) = f {
            for (j, g) in particle.formulas.iter().enumerate() {
                if i != j && inner.structurally_equal(g) {
                    return false;
                }
            }
        }
    }

    true
}

/// Check if an equivalent particle already exists in the list
fn particle_exists(particles: &[Particle], particle: &Particle) -> bool {
    particles.iter().any(|p| p.equals(particle))
}

impl std::fmt::Display for Tableau {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tableau for: {}", self.formula)?;
        writeln!(f, "Nodes: {} (init: {})", self.nodes.len(), self.init_count)?;
        for node in &self.nodes {
            write!(f, "  Node {}", node.index)?;
            if node.index < self.init_count {
                write!(f, " (init)")?;
            }
            if node.is_accepting() {
                write!(f, " (accepting)")?;
            }
            writeln!(f, ":")?;
            for formula in node.particle.formulas() {
                writeln!(f, "    {}", formula)?;
            }
            writeln!(f, "    -> {:?}", node.successors)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tla_core::ast::Expr;
    use tla_core::Spanned;

    fn state_pred(tag: u32) -> LiveExpr {
        LiveExpr::StatePred {
            expr: Arc::new(Spanned::dummy(Expr::Bool(true))),
            tag,
        }
    }

    #[test]
    fn test_particle_basic() {
        let p = state_pred(1);
        let particle = Particle::new(p);
        assert_eq!(particle.formulas().len(), 1);
        assert!(!particle.is_empty());
    }

    #[test]
    fn test_particle_closure_atom() {
        // Closure of a single atom is just that atom
        let p = state_pred(1);
        let particles = particle_closure(Particle::new(p));
        assert_eq!(particles.len(), 1);
    }

    #[test]
    fn test_particle_closure_conjunction() {
        // P /\ Q expands to single particle with P, Q, and (P /\ Q)
        // The original And is kept for membership checks during <>P expansion
        let p = state_pred(1);
        let q = state_pred(2);
        let conj = LiveExpr::and(vec![p, q]);
        let particles = particle_closure(Particle::new(conj));
        assert_eq!(particles.len(), 1);
        // Particle has: the original (P /\ Q), plus the expanded P and Q
        assert_eq!(particles[0].formulas().len(), 3);
    }

    #[test]
    fn test_particle_closure_disjunction() {
        // P \/ Q expands to two particles
        let p = state_pred(1);
        let q = state_pred(2);
        let disj = LiveExpr::or(vec![p, q]);
        let particles = particle_closure(Particle::new(disj));
        assert_eq!(particles.len(), 2);
    }

    #[test]
    fn test_particle_closure_always() {
        // []P expands to P /\ ()[]P
        let p = state_pred(1);
        let always_p = LiveExpr::always(p);
        let particles = particle_closure(Particle::new(always_p));
        assert_eq!(particles.len(), 1);
        // Should contain P and ()[]P
        let particle = &particles[0];
        assert!(particle.formulas().len() >= 2);
    }

    #[test]
    fn test_particle_closure_eventually() {
        // <>P expands to P \/ ()<>P (branching)
        let p = state_pred(1);
        let eventually_p = LiveExpr::eventually(p);
        let particles = particle_closure(Particle::new(eventually_p));
        // Two branches: one with P now, one with ()<>P
        assert_eq!(particles.len(), 2);
    }

    #[test]
    fn test_local_consistency() {
        let p = state_pred(1);

        // Consistent: just P
        let particle1 = Particle::new(p.clone());
        assert!(is_locally_consistent(&particle1));

        // Inconsistent: P and ~P
        let particle2 = Particle::from_vec(vec![p.clone(), LiveExpr::not(p.clone())]);
        assert!(!is_locally_consistent(&particle2));

        // Inconsistent: FALSE
        let particle3 = Particle::new(LiveExpr::Bool(false));
        assert!(!is_locally_consistent(&particle3));
    }

    #[test]
    fn test_tableau_construction() {
        // Simple: []P
        let p = state_pred(1);
        let always_p = LiveExpr::always(p);
        let tableau = Tableau::new(always_p);

        assert!(!tableau.is_empty());
        assert!(tableau.init_count() > 0);

        // All initial nodes should have successors
        for i in 0..tableau.init_count() {
            assert!(!tableau.nodes()[i].successors().is_empty());
        }
    }

    #[test]
    fn test_tableau_eventually_always() {
        // <>[]P (eventually always)
        let p = state_pred(1);
        let ea_p = LiveExpr::eventually(LiveExpr::always(p));
        let tableau = Tableau::new(ea_p);

        // Should have nodes for both branches
        assert!(tableau.len() >= 2);
    }
}
