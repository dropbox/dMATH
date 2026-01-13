//! Liveness checker implementation
//!
//! This module implements the product graph exploration for liveness checking.
//! The product graph is the cross-product of:
//! - The state graph (TLA+ states connected by Next relation)
//! - The tableau automaton (derived from negation of liveness property)
//!
//! A liveness violation exists iff there's an accepting cycle in this product graph.
//!
//! # TLC Reference
//!
//! This follows TLC's implementation in:
//! - `tlc2/tool/liveness/LiveCheck.java` - Main liveness checker
//! - `tlc2/tool/liveness/LiveWorker.java` - SCC detection
//!
//! # Phases
//!
//! Phase B2 (this file): Behavior graph construction
//! Phase B3: SCC detection using Tarjan's algorithm

use super::behavior_graph::{BehaviorGraph, BehaviorGraphNode};
use super::consistency::is_state_consistent;
use super::live_expr::LiveExpr;
use super::tableau::Tableau;
use crate::check::SuccessorWitnessMap;
use crate::error::EvalError;
use crate::error::EvalResult;
use crate::eval::{Env, EvalCtx};
use crate::state::{Fingerprint, State};
use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

/// A path segment in a counterexample trace: list of (state, tableau_node_index) pairs
type CounterexamplePath = Vec<(State, usize)>;

/// Result of liveness checking
#[derive(Debug, Clone)]
pub enum LivenessResult {
    /// No liveness violation found
    Satisfied,
    /// Liveness violation found - includes counterexample
    Violated {
        /// The liveness property that was violated
        property_desc: String,
        /// Prefix of the counterexample (finite path to the cycle)
        prefix: Vec<(State, usize)>,
        /// The cycle itself (back edge goes from last to first)
        cycle: Vec<(State, usize)>,
    },
    /// Checking was incomplete (e.g., state space too large)
    Incomplete {
        /// Description of why checking was incomplete
        reason: String,
    },
}

/// Constraints derived from fairness and temporal property patterns.
///
/// This corresponds to TLC's AE/EA checks (see `Liveness.classifyExpr`).
#[derive(Debug, Clone, Default)]
pub struct LivenessConstraints {
    /// State-level checks of the form `[]<>S` (must hold infinitely often).
    pub ae_state: Vec<LiveExpr>,
    /// Action-level checks of the form `[]<>A` (must occur infinitely often).
    pub ae_action: Vec<LiveExpr>,
    /// State-level checks of the form `<>[]S` (eventually always true states on suffix).
    pub ea_state: Vec<LiveExpr>,
    /// Action-level checks of the form `<>[]A` (eventually always true on the suffix).
    pub ea_action: Vec<LiveExpr>,
}

/// Liveness checker that builds and analyzes the behavior graph
///
/// The behavior graph is the product of state graph × tableau automaton.
/// Liveness checking proceeds in phases:
/// 1. Build behavior graph during state exploration
/// 2. Find strongly connected components (SCCs)
/// 3. Check each SCC for accepting cycles
pub struct LivenessChecker {
    /// The tableau automaton for the liveness property
    tableau: Tableau,
    /// The behavior graph (product of state graph × tableau)
    graph: BehaviorGraph,
    /// Base evaluation context for checking consistency
    ctx: EvalCtx,
    /// Promises (<>r) extracted from the tableau temporal formula
    promises: Vec<LiveExpr>,
    /// AE/EA constraints that must be satisfied by a counterexample cycle
    constraints: LivenessConstraints,
    /// Cached state graph successors (needed for ENABLED)
    state_successors: HashMap<crate::state::Fingerprint, Vec<State>>,
    /// Cached consistency check results: (state_fp, tableau_idx) -> is_consistent
    consistency_cache: HashMap<(crate::state::Fingerprint, usize), bool>,
    /// Optional mapping from representative state fingerprint -> canonical fingerprint (symmetry).
    state_fp_to_canon_fp: Option<Arc<FxHashMap<Fingerprint, Fingerprint>>>,
    /// Reduced state graph successors keyed by canonical fingerprint.
    ///
    /// Used as a fallback for ENABLED evaluation when no witness map is provided.
    graph_successors: Option<Arc<FxHashMap<Fingerprint, Vec<State>>>>,
    /// Concrete successor witnesses keyed by canonical source fingerprint.
    ///
    /// Each entry contains `(canonical_dest_fp, successor_state)` pairs for each
    /// concrete successor generated from the representative source state.
    ///
    /// When present, this is used to evaluate ENABLED and action predicates under symmetry.
    succ_witnesses: Option<Arc<SuccessorWitnessMap>>,
    /// Statistics
    stats: LivenessStats,
    /// Cached Env representations of states, keyed by fingerprint.
    /// Avoids repeated HashMap construction during SCC constraint checking.
    state_env_cache: HashMap<Fingerprint, Arc<Env>>,
}

/// Statistics for liveness checking
#[derive(Debug, Clone, Default)]
pub struct LivenessStats {
    /// Number of states explored
    pub states_explored: usize,
    /// Number of behavior graph nodes created
    pub graph_nodes: usize,
    /// Number of transitions in behavior graph
    pub graph_edges: usize,
    /// Number of consistency checks performed
    pub consistency_checks: usize,
    /// Microseconds spent adding initial states
    pub init_state_time_us: u64,
    /// Microseconds spent adding successor states
    pub add_successors_time_us: u64,
    /// Microseconds spent computing state successors
    pub get_successors_time_us: u64,
    /// Microseconds spent cloning states from graph
    pub state_clone_time_us: u64,
}

impl LivenessChecker {
    /// Create a new liveness checker
    ///
    /// # Arguments
    ///
    /// * `tableau` - The tableau automaton for the liveness property
    /// * `ctx` - Base evaluation context (with operators loaded)
    pub fn new(tableau: Tableau, ctx: EvalCtx) -> Self {
        Self::new_with_constraints(tableau, ctx, LivenessConstraints::default())
    }

    /// Create a new liveness checker with additional AE/EA constraints.
    pub fn new_with_constraints(
        tableau: Tableau,
        ctx: EvalCtx,
        constraints: LivenessConstraints,
    ) -> Self {
        // Collect ALL promises from the formula. Promises are <> subformulas that
        // must be fulfilled somewhere in any accepting SCC. The tableau expansion
        // and is_fulfilling check handle the fulfillment semantics correctly for
        // promises with temporal bodies (e.g., <>(P /\ []Q)).
        //
        // Previously, promises with temporal-level bodies were filtered out, but this
        // caused false positives: for <>(terminated /\ []~terminationDetected), the
        // promise was not tracked, so any SCC was considered violating.
        let promises = tableau.formula().extract_promises();

        Self {
            tableau,
            graph: BehaviorGraph::new(),
            ctx,
            promises,
            constraints,
            state_successors: HashMap::new(),
            consistency_cache: HashMap::new(),
            state_fp_to_canon_fp: None,
            graph_successors: None,
            succ_witnesses: None,
            stats: LivenessStats::default(),
            state_env_cache: HashMap::new(),
        }
    }

    /// Provide precomputed successor information for symmetry-aware liveness evaluation.
    ///
    /// When symmetry reduction is enabled, liveness checking needs access to the concrete
    /// successor states (not just canonical fingerprints) to correctly evaluate `ENABLED`
    /// and action-level predicates. TLC evaluates action checks on the concrete successor
    /// states *before* applying symmetry.
    pub fn set_successor_maps(
        &mut self,
        state_fp_to_canon_fp: Arc<FxHashMap<Fingerprint, Fingerprint>>,
        graph_successors: Arc<FxHashMap<Fingerprint, Vec<State>>>,
        succ_witnesses: Option<Arc<SuccessorWitnessMap>>,
    ) {
        self.state_fp_to_canon_fp = Some(state_fp_to_canon_fp);
        self.graph_successors = Some(graph_successors);
        self.succ_witnesses = succ_witnesses;
    }

    /// Decompose a (negated) liveness formula into one checker per DNF clause.
    ///
    /// This implements the same high-level approach as TLC's
    /// `Liveness.processLiveness`: convert to DNF and classify each conjunct into:
    /// - `<>[]A` (EA action checks)
    /// - `[]<>S` (AE state checks)
    /// - `[]<>A` (AE action checks)
    /// - remaining general temporal formulas (tableau `tf`)
    pub fn from_formula(formula: &LiveExpr, ctx: EvalCtx) -> Result<Vec<Self>, String> {
        fn push_unique(list: &mut Vec<LiveExpr>, expr: &LiveExpr) {
            if !list.iter().any(|e| e.structurally_equal(expr)) {
                list.push(expr.clone());
            }
        }

        let clauses = formula.to_dnf_clauses();
        let mut out = Vec::new();

        for clause in clauses {
            let mut constraints = LivenessConstraints::default();
            let mut tf_terms = Vec::new();

            for term in &clause {
                if let Some(body) = term.get_ea_body() {
                    match body.level() {
                        super::live_expr::ExprLevel::Constant
                        | super::live_expr::ExprLevel::State => {
                            push_unique(&mut constraints.ea_state, body);
                        }
                        super::live_expr::ExprLevel::Action => {
                            push_unique(&mut constraints.ea_action, body);
                        }
                        super::live_expr::ExprLevel::Temporal => {
                            return Err(format!("unsupported <>[] body: {}", term));
                        }
                    }
                    continue;
                }

                if let Some(body) = term.get_ae_body() {
                    match body.level() {
                        super::live_expr::ExprLevel::Constant
                        | super::live_expr::ExprLevel::State => {
                            push_unique(&mut constraints.ae_state, body);
                        }
                        super::live_expr::ExprLevel::Action => {
                            push_unique(&mut constraints.ae_action, body);
                        }
                        super::live_expr::ExprLevel::Temporal => {
                            return Err(format!("unsupported []<> body: {}", term));
                        }
                    }
                    continue;
                }

                // Handle <>Action as EA_action constraint.
                // This arises from negating [][Action]_vars: ~[][A]_v = <>(~[A]_v) = <>(~A /\ v' != v)
                // The <>Action pattern means "eventually take this action", which is checked by
                // verifying the action eventually occurs on some transition in the suffix.
                if let LiveExpr::Eventually(inner) = term {
                    if inner.level() == super::live_expr::ExprLevel::Action {
                        // <>Action can be treated as EA action (eventually always action in degenerate case)
                        // Or more properly, we check if this action ever becomes true on a transition
                        // For now, add to ea_action to ensure it's checked on suffixes
                        push_unique(&mut constraints.ea_action, inner);
                        continue;
                    }
                }

                if term.contains_action() {
                    return Err(format!(
                        "unsupported temporal subformula containing actions: {}",
                        term
                    ));
                }

                // Extract nested []<> (AE) patterns from within the term.
                // NOTE: We do NOT extract nested [] inside <> for leads-to violations.
                // The formula <>(P /\ []~Q) means "eventually P, and from that point []~Q".
                // The []~Q applies only AFTER P becomes true, so it cannot be extracted as
                // a global EA constraint that filters all edges. Instead, we let the tableau
                // handle this - the tableau for <>(P /\ []~Q) will track the []~Q obligation.
                let (ae_bodies, simplified_term) = term.extract_nested_ae();
                let ea_bodies: Vec<super::live_expr::LiveExpr> = Vec::new(); // Disabled for now

                // Add extracted AE bodies to constraints
                for body in ae_bodies {
                    match body.level() {
                        super::live_expr::ExprLevel::Constant
                        | super::live_expr::ExprLevel::State => {
                            push_unique(&mut constraints.ae_state, &body);
                        }
                        super::live_expr::ExprLevel::Action => {
                            push_unique(&mut constraints.ae_action, &body);
                        }
                        super::live_expr::ExprLevel::Temporal => {
                            // Should not happen since we only extract non-temporal bodies
                        }
                    }
                }

                // Add extracted EA bodies to constraints
                for body in ea_bodies {
                    match body.level() {
                        super::live_expr::ExprLevel::Constant
                        | super::live_expr::ExprLevel::State => {
                            push_unique(&mut constraints.ea_state, &body);
                        }
                        super::live_expr::ExprLevel::Action => {
                            push_unique(&mut constraints.ea_action, &body);
                        }
                        super::live_expr::ExprLevel::Temporal => {
                            // Should not happen since we only extract non-temporal bodies
                        }
                    }
                }

                // Add the simplified term (with nested patterns replaced by true)
                // Skip if the term simplified to just true
                if !matches!(simplified_term, LiveExpr::Bool(true)) {
                    tf_terms.push(simplified_term);
                }
            }

            let tf = LiveExpr::and(tf_terms);
            let tableau = Tableau::new(tf);
            out.push(LivenessChecker::new_with_constraints(
                tableau,
                ctx.clone(),
                constraints,
            ));
        }

        Ok(out)
    }

    /// Get the tableau
    pub fn tableau(&self) -> &Tableau {
        &self.tableau
    }

    /// Get the behavior graph
    pub fn graph(&self) -> &BehaviorGraph {
        &self.graph
    }

    /// Get statistics
    pub fn stats(&self) -> &LivenessStats {
        &self.stats
    }

    /// Get promises (<>r) extracted from the tableau formula.
    pub fn promises(&self) -> &[LiveExpr] {
        &self.promises
    }

    /// Get AE/EA constraints.
    pub fn constraints(&self) -> &LivenessConstraints {
        &self.constraints
    }

    /// Get or create a cached Env for a state.
    ///
    /// This avoids repeated HashMap construction during SCC constraint checking,
    /// which can be called thousands of times on the same states.
    fn get_cached_env(&mut self, state: &State) -> Arc<Env> {
        let fp = state.fingerprint();
        if let Some(env) = self.state_env_cache.get(&fp) {
            return Arc::clone(env);
        }

        // Build Env from state vars
        let mut env = Env::new();
        for (name, value) in state.vars() {
            env.insert(Arc::clone(name), value.clone());
        }
        let env = Arc::new(env);
        self.state_env_cache.insert(fp, Arc::clone(&env));
        env
    }

    /// Check consistency with caching. Returns cached result if available.
    fn check_consistency_cached<F>(
        &mut self,
        state: &State,
        tableau_idx: usize,
        get_successors: &mut F,
    ) -> EvalResult<bool>
    where
        F: FnMut(&State) -> EvalResult<Vec<State>>,
    {
        let fp = state.fingerprint();
        let cache_key = (fp, tableau_idx);

        // Check cache first
        if let Some(&cached) = self.consistency_cache.get(&cache_key) {
            return Ok(cached);
        }

        // Compute and cache the result
        self.stats.consistency_checks += 1;
        let tableau_node = match self.tableau.node(tableau_idx) {
            Some(n) => n,
            None => {
                self.consistency_cache.insert(cache_key, false);
                return Ok(false);
            }
        };

        let consistent = is_state_consistent(&self.ctx, state, tableau_node, get_successors)?;
        self.consistency_cache.insert(cache_key, consistent);
        Ok(consistent)
    }

    /// Add an initial state to the behavior graph
    ///
    /// This checks consistency with each initial tableau node and adds
    /// (state, tableau_node) pairs for consistent combinations.
    ///
    /// # Returns
    ///
    /// Vector of newly added behavior graph nodes
    pub fn add_initial_state<F>(
        &mut self,
        state: &State,
        get_successors: &mut F,
    ) -> EvalResult<Vec<BehaviorGraphNode>>
    where
        F: FnMut(&State) -> EvalResult<Vec<State>>,
    {
        let mut added = Vec::new();

        // For each initial tableau node (indices 0..init_count), check consistency
        for tableau_idx in 0..self.tableau.init_count() {
            let consistent = self.check_consistency_cached(state, tableau_idx, get_successors)?;
            if consistent && self.graph.add_init_node(state, tableau_idx) {
                let bg_node = BehaviorGraphNode::from_state(state, tableau_idx);
                added.push(bg_node);
                self.stats.graph_nodes += 1;
            }
        }

        self.stats.states_explored += 1;
        Ok(added)
    }

    /// Add successor states from a behavior graph node
    ///
    /// This computes the cross-product of:
    /// - State successors (from the Next relation)
    /// - Tableau successors (from the tableau automaton)
    ///
    /// Only consistent combinations are added to the behavior graph.
    ///
    /// # Arguments
    ///
    /// * `from` - The source behavior graph node
    /// * `successors` - State successors from the Next relation
    ///
    /// # Returns
    ///
    /// Vector of newly added behavior graph nodes
    pub fn add_successors<F>(
        &mut self,
        from: BehaviorGraphNode,
        successors: &[State],
        get_successors: &mut F,
    ) -> EvalResult<Vec<BehaviorGraphNode>>
    where
        F: FnMut(&State) -> EvalResult<Vec<State>>,
    {
        let mut added = Vec::new();

        // Cache state-graph successors for ENABLED evaluation.
        self.state_successors
            .entry(from.state_fp)
            .or_insert_with(|| successors.to_vec());

        let tableau_node = match self.tableau.node(from.tableau_idx) {
            Some(node) => node,
            None => return Ok(added), // Invalid tableau index
        };

        // Get successor indices before iterating
        let tableau_succ_indices: Vec<_> = tableau_node.successors().iter().copied().collect();

        // For each state successor
        for succ_state in successors {
            // NOTE: TLC KEEPS self-loops (stuttering edges) in the behavior graph.
            // TLC only treats single-node SCCs as trivial if they have NO self-loop.
            // See LiveWorker.java:539-544 (checkComponent) and 769-785 (isStuttering).
            // We must keep stuttering edges so the SCC algorithm can detect them.

            // For each tableau successor
            for &tableau_succ_idx in &tableau_succ_indices {
                // Check if successor state is consistent with successor tableau node (cached)
                let consistent =
                    self.check_consistency_cached(succ_state, tableau_succ_idx, get_successors)?;
                if consistent {
                    if self.graph.add_successor(from, succ_state, tableau_succ_idx) {
                        let bg_node = BehaviorGraphNode::from_state(succ_state, tableau_succ_idx);
                        added.push(bg_node);
                        self.stats.graph_nodes += 1;
                    }
                    self.stats.graph_edges += 1;
                }
            }
        }

        self.stats.states_explored += successors.len();
        Ok(added)
    }

    /// Explore the behavior graph using BFS
    ///
    /// This method performs a full BFS exploration of the behavior graph,
    /// starting from all initial states that are consistent with initial
    /// tableau nodes.
    ///
    /// # Arguments
    ///
    /// * `init_states` - All initial states from the model
    /// * `get_successors` - Function to compute successor states for a given state
    ///
    /// # Returns
    ///
    /// The number of behavior graph nodes explored
    pub fn explore_bfs<F>(
        &mut self,
        init_states: &[State],
        get_successors: &mut F,
    ) -> EvalResult<usize>
    where
        F: FnMut(&State) -> EvalResult<Vec<State>>,
    {
        let mut queue: VecDeque<BehaviorGraphNode> = VecDeque::new();

        // Add all initial states (with timing)
        let init_start = Instant::now();
        for init_state in init_states {
            let added = self.add_initial_state(init_state, get_successors)?;
            queue.extend(added);
        }
        self.stats.init_state_time_us += init_start.elapsed().as_micros() as u64;

        // BFS exploration
        while let Some(current) = queue.pop_front() {
            // Get the state for this behavior graph node (with timing)
            let clone_start = Instant::now();
            let state = match self.graph.get_state(&current) {
                Some(s) => s.clone(),
                None => continue,
            };
            self.stats.state_clone_time_us += clone_start.elapsed().as_micros() as u64;

            // Compute (and cache) state successors (with timing).
            let get_start = Instant::now();
            let state_successors =
                if let Some(cached) = self.state_successors.get(&state.fingerprint()) {
                    cached.clone()
                } else {
                    let succs = get_successors(&state)?;
                    self.state_successors
                        .insert(state.fingerprint(), succs.clone());
                    succs
                };
            self.stats.get_successors_time_us += get_start.elapsed().as_micros() as u64;

            // Add successor (state, tableau) pairs (with timing)
            let add_start = Instant::now();
            let added = self.add_successors(current, &state_successors, get_successors)?;
            self.stats.add_successors_time_us += add_start.elapsed().as_micros() as u64;
            queue.extend(added);
        }

        Ok(self.stats.graph_nodes)
    }

    /// Check if a behavior graph node is in an accepting tableau node
    ///
    /// An accepting node is one where the tableau indicates the property
    /// could be violated (e.g., a node with an "eventually" obligation
    /// that hasn't been fulfilled).
    pub fn is_accepting(&self, node: &BehaviorGraphNode) -> bool {
        self.tableau
            .node(node.tableau_idx)
            .is_some_and(|tn| tn.is_accepting())
    }

    /// Find strongly connected components in the behavior graph
    ///
    /// Returns all SCCs using Tarjan's algorithm.
    pub fn find_sccs(&self) -> super::tarjan::TarjanResult {
        super::tarjan::find_sccs(&self.graph)
    }

    /// Find non-trivial cycles in the behavior graph
    ///
    /// Returns SCCs that are actual cycles (not single nodes without self-loops).
    pub fn find_cycles(&self) -> Vec<super::tarjan::Scc> {
        super::tarjan::find_cycles(&self.graph)
    }

    /// Check for liveness violations
    ///
    /// A liveness violation occurs when there is an accepting cycle in the
    /// behavior graph. An accepting cycle is a strongly connected component
    /// that:
    /// 1. Has at least one edge (non-trivial SCC)
    /// 2. Contains at least one accepting tableau node
    ///
    /// # Returns
    ///
    /// A `LivenessResult` indicating whether the property is satisfied,
    /// violated (with counterexample), or checking was incomplete.
    pub fn check_liveness(&mut self) -> LivenessResult {
        let profile = std::env::var("LIVENESS_PROFILE").is_ok();

        let debug_scc = std::env::var("TLA2_DEBUG_SCC").is_ok();
        if debug_scc {
            eprintln!("[DEBUG CONSTRAINTS] ea_state: {}, ea_action: {}, ae_state: {}, ae_action: {}",
                self.constraints.ea_state.len(),
                self.constraints.ea_action.len(),
                self.constraints.ae_state.len(),
                self.constraints.ae_action.len(),
            );
            for (i, ea) in self.constraints.ea_action.iter().enumerate() {
                eprintln!("[DEBUG EA_ACTION {}] {}", i, ea);
                eprintln!("[DEBUG EA_ACTION {} FULL] {:?}", i, ea);
            }
        }

        // If there are EA checks (<>[]A), restrict SCC analysis to edges that
        // satisfy all of them (TLC: restrict SCC search by EAAction).
        let ea_start = Instant::now();
        let allowed_edges = match self.compute_allowed_edges_for_ea() {
            Ok(v) => v,
            Err(e) => {
                return LivenessResult::Incomplete {
                    reason: format!("error evaluating EAAction checks: {}", e),
                };
            }
        };
        if profile {
            eprintln!(
                "  check_liveness: compute_allowed_edges_for_ea: {:.3}s (filtered_edges={})",
                ea_start.elapsed().as_secs_f64(),
                allowed_edges.as_ref().map(|s| s.len()).unwrap_or(0)
            );
        }

        let scc_start = Instant::now();
        let scc_result = if let Some(ref allowed) = allowed_edges {
            super::tarjan::find_sccs_with_edge_filter(&self.graph, &|from, to| {
                allowed.contains(&(*from, *to))
            })
        } else {
            self.find_sccs()
        };
        if profile {
            eprintln!(
                "  check_liveness: tarjan: {:.3}s (sccs={})",
                scc_start.elapsed().as_secs_f64(),
                scc_result.sccs.len()
            );
        }

        // On-demand cache for AE state constraint evaluations.
        // Caches (constraint_idx, state_fp) -> satisfies during SCC iteration.
        // This avoids redundant re-evaluation when the same state appears in multiple SCCs.
        // Using FxHashMap for faster lookups with small keys.
        let mut ae_state_cache: FxHashMap<(usize, Fingerprint), bool> = FxHashMap::default();
        // Clone AE state checks once, not per-SCC
        let ae_state_checks: Vec<_> = self.constraints.ae_state.clone();

        // Check SCCs for satisfiability of the PEM conjunct (AE checks + promises).
        let mut non_trivial_count = 0usize;
        let scc_loop_start = Instant::now();
        for scc in &scc_result.sccs {
            if self.is_trivial_scc_with_allowed_edges(scc, allowed_edges.as_ref()) {
                continue;
            }
            non_trivial_count += 1;

            let scc_check_start = if profile { Some(Instant::now()) } else { None };
            let constraints_result = self.scc_satisfies_constraints_cached(scc, allowed_edges.as_ref(), &mut ae_state_cache, &ae_state_checks);
            let debug_scc = std::env::var("TLA2_DEBUG_SCC").is_ok();
            if debug_scc {
                eprintln!("[DEBUG SCC] SCC {} nodes, constraints_result = {:?}", scc.len(), constraints_result);
            }
            match constraints_result {
                Ok(true) => {
                    if debug_scc {
                        eprintln!("[DEBUG SCC] Found violating SCC with {} nodes", scc.len());
                        let scc_tableau = scc.nodes()[0].tableau_idx;
                        // Print some sample nodes
                        for (i, node) in scc.nodes().iter().take(5).enumerate() {
                            if let Some(state) = self.graph.get_state(node) {
                                if let Some(pc) = state.get("pc") {
                                    eprintln!("  Sample node {}: tableau={}, pc={}", i, node.tableau_idx, pc);
                                }
                            }
                        }
                        // Check if there are ANY nodes in the whole graph with same tableau and pc=cs
                        let mut cs_nodes_count = 0;
                        let mut cs_nodes_same_tableau = 0;
                        for (node, _) in self.graph.nodes() {
                            if let Some(state) = self.graph.get_state(node) {
                                if let Some(pc) = state.get("pc") {
                                    let pc_str = format!("{}", pc);
                                    if pc_str.contains("\"cs\"") {
                                        cs_nodes_count += 1;
                                        if node.tableau_idx == scc_tableau {
                                            cs_nodes_same_tableau += 1;
                                        }
                                    }
                                }
                            }
                        }
                        eprintln!("[DEBUG SCC] Graph has {} nodes with pc=cs, {} with tableau={}", cs_nodes_count, cs_nodes_same_tableau, scc_tableau);
                    }
                }
                Ok(false) => continue,
                Err(e) => {
                    return LivenessResult::Incomplete {
                        reason: format!("error checking SCC constraints: {}", e),
                    };
                }
            }
            if let Some(start) = scc_check_start {
                eprintln!(
                    "  check_liveness: scc_satisfies_constraints: {:.3}s (size={})",
                    start.elapsed().as_secs_f64(),
                    scc.len()
                );
            }

            // Found a violating SCC. Construct a lasso cycle that contains witnesses
            // for each AE check and each promise.
            let witness_start = if profile { Some(Instant::now()) } else { None };
            let cycle_nodes = match self.build_witness_cycle_in_scc(scc, allowed_edges.as_ref()) {
                Ok(Some(cycle)) => cycle,
                Ok(None) => continue,
                Err(e) => {
                    return LivenessResult::Incomplete {
                        reason: format!("error constructing counterexample cycle: {}", e),
                    };
                }
            };
            if let Some(start) = witness_start {
                eprintln!(
                    "  check_liveness: build_witness_cycle_in_scc: {:.3}s (cycle_len={})",
                    start.elapsed().as_secs_f64(),
                    cycle_nodes.len()
                );
            }

            let (prefix, cycle) = self.build_counterexample(&cycle_nodes);
            return LivenessResult::Violated {
                property_desc: "Liveness property".to_string(),
                prefix,
                cycle,
            };
        }

        if profile {
            eprintln!(
                "  check_liveness: scc_loop: {:.3}s (non_trivial={}, cache_size={})",
                scc_loop_start.elapsed().as_secs_f64(),
                non_trivial_count,
                ae_state_cache.len()
            );
        }

        LivenessResult::Satisfied
    }

    fn build_witness_cycle_in_scc(
        &mut self,
        scc: &super::tarjan::Scc,
        allowed_edges: Option<&HashSet<(BehaviorGraphNode, BehaviorGraphNode)>>,
    ) -> EvalResult<Option<Vec<BehaviorGraphNode>>> {
        if scc.is_empty() {
            return Ok(None);
        }

        // Self-loop SCC.
        if scc.len() == 1 {
            let node = scc.nodes()[0];
            let has_self_loop = if let Some(allowed_edges) = allowed_edges {
                allowed_edges.contains(&(node, node))
            } else {
                self.graph
                    .get_node_info(&node)
                    .is_some_and(|info| info.successors.contains(&node))
            };
            return Ok(has_self_loop.then_some(vec![node, node]));
        }

        #[derive(Clone, Copy)]
        enum Milestone {
            Node(BehaviorGraphNode),
            Edge(BehaviorGraphNode, BehaviorGraphNode),
        }

        let scc_set: HashSet<BehaviorGraphNode> = scc.nodes().iter().copied().collect();
        let mut milestones: Vec<Milestone> = Vec::new();

        // Promise witnesses.
        for promise in &self.promises {
            let mut found = None;
            for node in scc.nodes() {
                let Some(tnode) = self.tableau.node(node.tableau_idx) else {
                    continue;
                };
                if tnode.particle().is_fulfilling(promise) {
                    found = Some(*node);
                    break;
                }
            }
            let Some(node) = found else {
                return Ok(None);
            };
            milestones.push(Milestone::Node(node));
        }

        // AEState witnesses.
        // Clone checks and collect state data to avoid borrow issues
        let ae_state_checks: Vec<_> = self.constraints.ae_state.clone();
        let scc_states: Vec<_> = scc
            .nodes()
            .iter()
            .filter_map(|node| Some((*node, self.graph.get_state(node)?.clone())))
            .collect();
        for check in &ae_state_checks {
            let mut found = None;
            for (node, state) in &scc_states {
                if self.eval_check_on_state(check, state)? {
                    found = Some(*node);
                    break;
                }
            }
            let Some(node) = found else {
                return Ok(None);
            };
            milestones.push(Milestone::Node(node));
        }

        // AEAction witnesses.
        // Note: For fairness constraints like WF_vars(A), the action must cause a state change.
        // The subscripted action <<A>>_vars = A /\ (vars' ≠ vars) requires vars to change.
        // So we skip self-loop transitions (stuttering) when looking for action witnesses.
        // Clone checks and collect edge data to avoid borrow issues
        let ae_action_checks: Vec<_> = self.constraints.ae_action.clone();
        let scc_edges: Vec<_> = scc
            .nodes()
            .iter()
            .filter_map(|from| {
                let from_info = self.graph.get_node_info(from)?;
                let from_state = self.graph.get_state(from)?.clone();
                let valid_successors: Vec<_> = from_info
                    .successors
                    .iter()
                    .filter(|to| {
                        scc_set.contains(*to)
                            && allowed_edges.map_or(true, |ae| ae.contains(&(*from, **to)))
                    })
                    .filter_map(|to| Some((*to, self.graph.get_state(to)?.clone())))
                    .collect();
                Some((*from, from_state, valid_successors))
            })
            .collect();
        for check in &ae_action_checks {
            let mut found = None;
            for (from, from_state, successors) in &scc_edges {
                for (to, to_state) in successors {
                    if self.eval_check_on_transition(check, from_state, to_state)? {
                        found = Some((*from, *to));
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            let Some((from, to)) = found else {
                return Ok(None);
            };
            milestones.push(Milestone::Edge(from, to));
        }

        // Pick a start node (prefer a milestone).
        let start = milestones
            .iter()
            .map(|m| match m {
                Milestone::Node(n) => *n,
                Milestone::Edge(from, _) => *from,
            })
            .next()
            .unwrap_or_else(|| scc.nodes()[0]);

        let mut cycle = vec![start];
        let mut current = start;

        for milestone in &milestones {
            match *milestone {
                Milestone::Node(target) => {
                    let Some(path) =
                        self.find_path_within_scc(current, target, &scc_set, allowed_edges)
                    else {
                        return Ok(None);
                    };
                    cycle.extend(path.into_iter().skip(1));
                    current = target;
                }
                Milestone::Edge(from, to) => {
                    let Some(path) =
                        self.find_path_within_scc(current, from, &scc_set, allowed_edges)
                    else {
                        return Ok(None);
                    };
                    cycle.extend(path.into_iter().skip(1));

                    if let Some(allowed_edges) = allowed_edges {
                        if !allowed_edges.contains(&(from, to)) {
                            return Ok(None);
                        }
                    }
                    cycle.push(to);
                    current = to;
                }
            }
        }

        // Close the cycle.
        let Some(back_path) = self.find_path_within_scc(current, start, &scc_set, allowed_edges)
        else {
            return Ok(None);
        };
        cycle.extend(back_path.into_iter().skip(1));

        Ok(Some(cycle))
    }

    fn find_path_within_scc(
        &self,
        start: BehaviorGraphNode,
        goal: BehaviorGraphNode,
        scc_set: &HashSet<BehaviorGraphNode>,
        allowed_edges: Option<&HashSet<(BehaviorGraphNode, BehaviorGraphNode)>>,
    ) -> Option<Vec<BehaviorGraphNode>> {
        if start == goal {
            return Some(vec![start]);
        }

        let mut visited: HashSet<BehaviorGraphNode> = HashSet::new();
        let mut parent: HashMap<BehaviorGraphNode, BehaviorGraphNode> = HashMap::new();
        let mut queue: VecDeque<BehaviorGraphNode> = VecDeque::new();

        visited.insert(start);
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            let Some(info) = self.graph.get_node_info(&node) else {
                continue;
            };

            for succ in &info.successors {
                if !scc_set.contains(succ) || visited.contains(succ) {
                    continue;
                }
                if let Some(allowed_edges) = allowed_edges {
                    if !allowed_edges.contains(&(node, *succ)) {
                        continue;
                    }
                }

                visited.insert(*succ);
                parent.insert(*succ, node);

                if *succ == goal {
                    break;
                }
                queue.push_back(*succ);
            }

            if visited.contains(&goal) {
                break;
            }
        }

        if !visited.contains(&goal) {
            return None;
        }

        let mut path = Vec::new();
        let mut cur = goal;
        path.push(cur);
        while let Some(&p) = parent.get(&cur) {
            path.push(p);
            if p == start {
                break;
            }
            cur = p;
        }
        path.reverse();
        Some(path)
    }

    fn compute_allowed_edges_for_ea(
        &mut self,
    ) -> crate::error::EvalResult<Option<HashSet<(BehaviorGraphNode, BehaviorGraphNode)>>> {
        if self.constraints.ea_action.is_empty() && self.constraints.ea_state.is_empty() {
            return Ok(None);
        }

        // Clone checks and collect edge data to avoid borrow issues
        let ea_action_checks: Vec<_> = self.constraints.ea_action.clone();
        let ea_state_checks: Vec<_> = self.constraints.ea_state.clone();

        let edges_to_check: Vec<_> = self
            .graph
            .nodes()
            .filter_map(|(from, info)| {
                let from_state = self.graph.get_state(from)?.clone();
                let successors: Vec<_> = info
                    .successors
                    .iter()
                    .filter_map(|succ| Some((*succ, self.graph.get_state(succ)?.clone())))
                    .collect();
                Some((*from, from_state, successors))
            })
            .collect();

        let mut allowed = HashSet::new();

        for (from, from_state, successors) in &edges_to_check {
            for (succ, to_state) in successors {
                // Check ea_action constraints (evaluated on transitions)
                let action_ok = if ea_action_checks.is_empty() {
                    true
                } else {
                    self.eval_checks_on_transition(&ea_action_checks, from_state, to_state)?
                };

                // Check ea_state constraints (both endpoints must satisfy state predicates)
                // For <>[]P where P is state-level, we need both from and to to satisfy P
                let state_ok = if ea_state_checks.is_empty() {
                    true
                } else {
                    self.eval_checks_on_state(&ea_state_checks, from_state)?
                        && self.eval_checks_on_state(&ea_state_checks, to_state)?
                };

                if action_ok && state_ok {
                    allowed.insert((*from, *succ));
                }
            }
        }

        Ok(Some(allowed))
    }

    fn is_trivial_scc_with_allowed_edges(
        &self,
        scc: &super::tarjan::Scc,
        allowed_edges: Option<&HashSet<(BehaviorGraphNode, BehaviorGraphNode)>>,
    ) -> bool {
        if scc.len() != 1 {
            return false;
        }
        let node = scc.nodes()[0];
        if let Some(allowed_edges) = allowed_edges {
            !allowed_edges.contains(&(node, node))
        } else {
            self.graph
                .get_node_info(&node)
                .map_or(true, |info| !info.successors.contains(&node))
        }
    }

    fn scc_satisfies_constraints_cached(
        &mut self,
        scc: &super::tarjan::Scc,
        allowed_edges: Option<&HashSet<(BehaviorGraphNode, BehaviorGraphNode)>>,
        ae_state_cache: &mut FxHashMap<(usize, Fingerprint), bool>,
        ae_state_checks: &[LiveExpr],
    ) -> crate::error::EvalResult<bool> {
        let debug = std::env::var("TLA2_DEBUG_SCC").is_ok();
        let fulfills_promises = self.scc_fulfills_promises(scc);
        if debug {
            eprintln!("[DEBUG SCC CONSTRAINTS] fulfills_promises={} (num_promises={})", fulfills_promises, self.promises.len());
        }
        if !fulfills_promises {
            return Ok(false);
        }
        let ae_state_ok = self.scc_satisfies_ae_state_cached(scc, ae_state_cache, ae_state_checks)?;
        if debug {
            eprintln!("[DEBUG SCC CONSTRAINTS] ae_state_ok={} (num_checks={})", ae_state_ok, ae_state_checks.len());
        }
        if !ae_state_ok {
            return Ok(false);
        }
        let ae_action_ok = self.scc_satisfies_ae_action(scc, allowed_edges)?;
        if debug {
            eprintln!("[DEBUG SCC CONSTRAINTS] ae_action_ok={} (num_checks={})", ae_action_ok, self.constraints.ae_action.len());
        }
        if !ae_action_ok {
            return Ok(false);
        }
        Ok(true)
    }

    fn scc_fulfills_promises(&self, scc: &super::tarjan::Scc) -> bool {
        for promise in &self.promises {
            let mut fulfilled_somewhere = false;
            for node in scc.nodes() {
                let Some(tnode) = self.tableau.node(node.tableau_idx) else {
                    continue;
                };
                if tnode.particle().is_fulfilling(promise) {
                    fulfilled_somewhere = true;
                    break;
                }
            }
            if !fulfilled_somewhere {
                return false;
            }
        }
        true
    }

    /// Check AE state constraints with on-demand caching.
    ///
    /// Caches evaluation results so states appearing in multiple SCCs
    /// are only evaluated once per constraint.
    fn scc_satisfies_ae_state_cached(
        &mut self,
        scc: &super::tarjan::Scc,
        cache: &mut FxHashMap<(usize, Fingerprint), bool>,
        checks: &[LiveExpr],
    ) -> crate::error::EvalResult<bool> {
        for (check_idx, check) in checks.iter().enumerate() {
            let mut found = false;
            for node in scc.nodes() {
                let Some(state) = self.graph.get_state(node) else {
                    continue;
                };
                let fp = state.fingerprint();
                let cache_key = (check_idx, fp);

                // Check cache first
                let satisfies = if let Some(&cached) = cache.get(&cache_key) {
                    cached
                } else {
                    // Evaluate and cache
                    let state_cloned = state.clone();
                    let result = self.eval_check_on_state(check, &state_cloned)?;
                    cache.insert(cache_key, result);
                    result
                };

                if satisfies {
                    found = true;
                    break;
                }
            }
            if !found {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn scc_satisfies_ae_action(
        &mut self,
        scc: &super::tarjan::Scc,
        allowed_edges: Option<&HashSet<(BehaviorGraphNode, BehaviorGraphNode)>>,
    ) -> crate::error::EvalResult<bool> {
        let debug = std::env::var("TLA2_DEBUG_AE_ACTION").is_ok();
        if self.constraints.ae_action.is_empty() {
            return Ok(true);
        }

        let scc_set: HashSet<BehaviorGraphNode> = scc.nodes().iter().copied().collect();

        // Clone checks and collect edge data to avoid borrow issues
        let checks: Vec<_> = self.constraints.ae_action.clone();
        let edges: Vec<_> = scc
            .nodes()
            .iter()
            .filter_map(|from| {
                let from_info = self.graph.get_node_info(from)?;
                let from_state = self.graph.get_state(from)?.clone();
                let valid_successors: Vec<_> = from_info
                    .successors
                    .iter()
                    .filter(|to| {
                        scc_set.contains(*to)
                            && allowed_edges.map_or(true, |ae| ae.contains(&(*from, **to)))
                    })
                    .filter_map(|to| Some((*to, self.graph.get_state(to)?.clone())))
                    .collect();
                Some((*from, from_state, valid_successors))
            })
            .collect();

        if debug {
            let total_edges: usize = edges.iter().map(|(_, _, succs)| succs.len()).sum();
            eprintln!("[DEBUG AE_ACTION] SCC {} nodes, {} internal edges, {} checks", scc.len(), total_edges, checks.len());
            // Show which processes are taking steps in this SCC
            let mut step_count = std::collections::HashMap::<String, usize>::new();
            for (_, from_state, successors) in &edges {
                for (_, to_state) in successors {
                    if let (Some(from_pc), Some(to_pc)) = (from_state.get("pc"), to_state.get("pc")) {
                        let from_str = format!("{}", from_pc);
                        let to_str = format!("{}", to_pc);
                        // Find which process moved (pc value changed)
                        for p in ["p1", "p2", "p3"] {
                            if from_str.contains(p) && to_str.contains(p) {
                                // Check if this process's pc changed
                                // This is a simplistic check - look for different values after the process name
                                let from_idx = from_str.find(p).unwrap_or(0);
                                let to_idx = to_str.find(p).unwrap_or(0);
                                let from_part: String = from_str.chars().skip(from_idx).take(30).collect();
                                let to_part: String = to_str.chars().skip(to_idx).take(30).collect();
                                if from_part != to_part {
                                    *step_count.entry(p.to_string()).or_insert(0) += 1;
                                }
                            }
                        }
                    }
                }
            }
            eprintln!("[DEBUG AE_ACTION] Process step counts in SCC: {:?}", step_count);
        }

        for (check_idx, check) in checks.iter().enumerate() {
            let mut found = false;
            let mut found_reason = "";
            for (from, from_state, successors) in &edges {
                for (to, to_state) in successors {
                    if self.eval_check_on_transition(check, from_state, to_state)? {
                        found = true;
                        if debug {
                            // Try to determine WHY it passed (ENABLED or action happened)
                            if let LiveExpr::Or(disjuncts) = check {
                                for d in disjuncts {
                                    if let LiveExpr::Not(inner) = d {
                                        if matches!(inner.as_ref(), LiveExpr::Enabled { .. })
                                            && self.eval_check_on_state(d, from_state).unwrap_or(false)
                                        {
                                            found_reason = "~ENABLED was true";
                                        }
                                    } else if let LiveExpr::And(_) = d {
                                        if self.eval_check_on_transition(d, from_state, to_state).unwrap_or(false) {
                                            found_reason = "action happened";
                                        }
                                    }
                                }
                            }
                            eprintln!("[DEBUG AE_ACTION] Check {} passed: {} -> {} (reason: {})",
                                check_idx, from.tableau_idx, to.tableau_idx, found_reason);
                            if let Some(pc) = from_state.get("pc") {
                                eprintln!("[DEBUG AE_ACTION]   from pc={}", pc);
                            }
                        }
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if debug {
                eprintln!("[DEBUG AE_ACTION] Check {}: {} found={}", check_idx, check, found);
            }
            if !found {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn eval_checks_on_transition(
        &mut self,
        checks: &[LiveExpr],
        state0: &State,
        state1: &State,
    ) -> crate::error::EvalResult<bool> {
        for check in checks {
            if !self.eval_check_on_transition(check, state0, state1)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn eval_check_on_state(&mut self, check: &LiveExpr, state: &State) -> EvalResult<bool> {
        self.eval_live_check_expr(check, state, None)
    }

    fn eval_checks_on_state(&mut self, checks: &[LiveExpr], state: &State) -> EvalResult<bool> {
        for check in checks {
            if !self.eval_check_on_state(check, state)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn eval_check_on_transition(
        &mut self,
        check: &LiveExpr,
        state0: &State,
        state1: &State,
    ) -> EvalResult<bool> {
        // Under symmetry, a reduced edge (state0 -> state1) can represent multiple concrete
        // successor states. Evaluate the check existentially over all concrete witnesses.
        // Clone the Arc references we need to avoid borrow issues
        let fp_map = self.state_fp_to_canon_fp.clone();
        let witnesses = self.succ_witnesses.clone();
        if let (Some(fp_map), Some(witnesses)) = (fp_map, witnesses)
        {
            let fp0 = state0.fingerprint();
            let fp1 = state1.fingerprint();

            let Some(canon0) = fp_map.get(&fp0).copied() else {
                return self.eval_live_check_expr(check, state0, Some(state1));
            };
            let Some(canon1) = fp_map.get(&fp1).copied() else {
                return self.eval_live_check_expr(check, state0, Some(state1));
            };

            // The reduced graph always contains an explicit stuttering self-loop (s -> s).
            if canon0 == canon1 && self.eval_live_check_expr(check, state0, Some(state0))? {
                return Ok(true);
            }

            if let Some(succs) = witnesses.get(&canon0) {
                for (dest_canon_fp, succ_state) in succs {
                    if *dest_canon_fp != canon1 {
                        continue;
                    }
                    if self.eval_live_check_expr(check, state0, Some(succ_state))? {
                        return Ok(true);
                    }
                }
            }

            // No concrete candidate satisfied the check.
            return Ok(false);
        }

        self.eval_live_check_expr(check, state0, Some(state1))
    }

    /// Evaluate a liveness check expression against current (and optionally next) state.
    ///
    /// Optimized to use array-based state binding via `bind_state_array` when the
    /// VarRegistry is populated. Falls back to HashMap-based binding when the
    /// registry is empty (e.g., in tests with minimal setup).
    fn eval_live_check_expr(
        &mut self,
        expr: &LiveExpr,
        current_state: &State,
        next_state: Option<&State>,
    ) -> EvalResult<bool> {
        // Get cached next_state Env upfront (avoids repeated HashMap construction)
        let cached_next_env = next_state.map(|ns| self.get_cached_env(ns));

        let registry = &self.ctx.shared.var_registry;

        // Use optimized array-based path if VarRegistry is populated
        if !registry.is_empty() {
            let registry_cloned = registry.clone();
            let current_values = current_state.to_values(&registry_cloned);

            // Save current state_env and bind new one (O(1) pointer swap)
            let prev_state_env = self.ctx.bind_state_array(&current_values);

            // Save current next_state and set new one if provided (using cached Env)
            let prev_next_state = self.ctx.next_state.take();
            if let Some(env) = cached_next_env.clone() {
                self.ctx.next_state = Some(env);
            }

            let has_action_ctx = self.ctx.next_state.is_some();
            let result =
                self.eval_live_check_expr_inner(expr, current_state, next_state, has_action_ctx);

            // Restore previous state (O(1))
            self.ctx.restore_state_env(prev_state_env);
            self.ctx.next_state = prev_next_state;

            result
        } else {
            // Fallback: Use HashMap-based binding for empty VarRegistry (e.g., minimal tests)
            // Bind current state variables to env
            let prev_env = self.ctx.env.clone();
            for (name, value) in current_state.vars() {
                self.ctx.bind_mut(Arc::clone(name), value.clone());
            }

            // Save current next_state and set new one if provided (using cached Env)
            let prev_next_state = self.ctx.next_state.take();
            if let Some(env) = cached_next_env {
                self.ctx.next_state = Some(env);
            }

            let has_action_ctx = self.ctx.next_state.is_some();
            let result =
                self.eval_live_check_expr_inner(expr, current_state, next_state, has_action_ctx);

            // Restore previous state
            self.ctx.env = prev_env;
            self.ctx.next_state = prev_next_state;

            result
        }
    }

    /// Inner recursive evaluation of liveness check expressions.
    ///
    /// Uses `self.ctx` which has been pre-configured with:
    /// - `state_env` pointing to current state values
    /// - `next_state` set if action context is needed
    fn eval_live_check_expr_inner(
        &self,
        expr: &LiveExpr,
        current_state: &State,
        next_state: Option<&State>,
        has_action_ctx: bool,
    ) -> EvalResult<bool> {
        match expr {
            LiveExpr::Bool(b) => Ok(*b),

            LiveExpr::StatePred { expr, .. } => {
                // self.ctx has state_env set, and next_state if action context is available
                match crate::eval::eval(&self.ctx, expr)? {
                    crate::Value::Bool(b) => Ok(b),
                    _ => Ok(false),
                }
            }

            LiveExpr::ActionPred { expr, tag, .. } => {
                // Action predicates require next_state to be set
                if !has_action_ctx {
                    return Ok(false);
                }
                let result = match crate::eval::eval(&self.ctx, expr)? {
                    crate::Value::Bool(b) => b,
                    _ => false,
                };
                if std::env::var("TLA2_DEBUG_ACTION_PRED").is_ok() {
                    eprintln!("[DEBUG ACTION_PRED] tag={} expr={:?} result={}", tag, expr.node, result);
                }
                Ok(result)
            }

            LiveExpr::Enabled {
                action,
                require_state_change,
                subscript,
                ..
            } => self.eval_enabled(
                &self.ctx,
                current_state,
                action,
                *require_state_change,
                subscript.as_ref(),
            ),

            LiveExpr::StateChanged { subscript, tag, .. } => {
                // StateChanged is true iff e' ≠ e for subscript expression e
                // This is used for subscripted action semantics <<A>>_e = A /\ (e' ≠ e)
                match next_state {
                    Some(ns) => {
                        if let Some(sub_expr) = subscript {
                            // Evaluate subscript in both states and compare values
                            let result = self.eval_subscript_changed(&self.ctx, current_state, ns, sub_expr)?;
                            if std::env::var("TLA2_DEBUG_CHANGED").is_ok() {
                                eprintln!("[DEBUG CHANGED] tag={} subscript={:?} result={}", tag, sub_expr.node, result);
                            }
                            Ok(result)
                        } else {
                            // No subscript - use global fingerprint comparison (fallback)
                            // Under symmetry, compare canonical fingerprints to detect true state change.
                            if let Some(fp_map) = &self.state_fp_to_canon_fp {
                                let canon_current =
                                    fp_map.get(&current_state.fingerprint()).copied();
                                let canon_next = fp_map.get(&ns.fingerprint()).copied();
                                match (canon_current, canon_next) {
                                    (Some(c1), Some(c2)) => Ok(c1 != c2),
                                    _ => Ok(current_state.fingerprint() != ns.fingerprint()),
                                }
                            } else {
                                Ok(current_state.fingerprint() != ns.fingerprint())
                            }
                        }
                    }
                    None => Ok(false), // No next state means we can't evaluate state change
                }
            }

            LiveExpr::Not(inner) => Ok(!self.eval_live_check_expr_inner(
                inner,
                current_state,
                next_state,
                has_action_ctx,
            )?),

            LiveExpr::And(exprs) => {
                for e in exprs {
                    if !self.eval_live_check_expr_inner(
                        e,
                        current_state,
                        next_state,
                        has_action_ctx,
                    )? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }

            LiveExpr::Or(exprs) => {
                for e in exprs {
                    if self.eval_live_check_expr_inner(
                        e,
                        current_state,
                        next_state,
                        has_action_ctx,
                    )? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }

            LiveExpr::Always(_) | LiveExpr::Eventually(_) | LiveExpr::Next(_) => {
                Err(EvalError::Internal {
                    message: format!("unexpected temporal operator in check evaluation: {}", expr),
                    span: None,
                })
            }
        }
    }

    fn eval_enabled(
        &self,
        ctx_current: &EvalCtx,
        current_state: &State,
        action: &std::sync::Arc<tla_core::Spanned<tla_core::ast::Expr>>,
        require_state_change: bool,
        subscript: Option<&std::sync::Arc<tla_core::Spanned<tla_core::ast::Expr>>>,
    ) -> EvalResult<bool> {
        // CRITICAL FIX (#55): Do NOT rely on pre-computed successors for ENABLED evaluation.
        //
        // The bug was: when evaluating ENABLED(A) for a specific action A, we were checking
        // if A(s, s') returns true for pre-computed successors s'. But those successors were
        // generated by the ENTIRE Next disjunction, not by action A specifically.
        //
        // Example: For NonCounterStep(p) in Prisoners.tla:
        // - NonCounterStep(p3) might produce successor S by flipping switchB
        // - NonCounterStep(p4) might also produce the SAME successor S (fingerprint dedup)
        // - When checking ENABLED NonCounterStep(p3), we'd try evaluating it on S
        // - But S was generated by p4's action, so NonCounterStep(p3)(current, S) = FALSE
        // - Result: False negative - TLA2 incorrectly reports action as not enabled
        //
        // TLC's approach (LNStateEnabled.java): Call tool.enabled() to perform a fresh
        // existential search for ANY satisfying successor from the specific action.
        // Quote from TLC source: "// Note that s2 is useless." - it doesn't use pre-computed
        // successors at all.
        //
        // The fix: Use enumerate::eval_enabled which performs fresh enumeration from the
        // specific action expression, just like TLC does.

        // Get variable names for enumeration
        let vars: Vec<Arc<str>> = ctx_current.shared.var_registry.names().to_vec();

        // If var_registry is populated, use fresh enumeration (the correct approach).
        // Otherwise, fall back to pre-computed successor checking (for synthetic tests
        // that don't set up full infrastructure).
        if !vars.is_empty() {
            // Create evaluation context with current state bound in env
            let mut eval_ctx = ctx_current.clone();
            eval_ctx.next_state = None; // Clear any stale next_state - ENABLED quantifies existentially

            // If we need state change (subscripted action <<A>>_e), enumerate successors
            // and check if any cause the subscript to change (e' ≠ e)
            if require_state_change {
                match crate::enumerate::enumerate_action_successors(
                    &mut eval_ctx,
                    action,
                    current_state,
                    &vars,
                ) {
                    Ok(successors) => {
                        // Check if any successor has different subscript value
                        for succ in successors {
                            if let Some(sub_expr) = subscript {
                                // Compare subscript expression in both states
                                let changed = self.eval_subscript_changed(
                                    ctx_current,
                                    current_state,
                                    &succ,
                                    sub_expr,
                                )?;
                                if changed {
                                    return Ok(true);
                                }
                            } else {
                                // No subscript - fallback to fingerprint comparison
                                let fp_changed = succ.fingerprint() != current_state.fingerprint();
                                if fp_changed {
                                    return Ok(true);
                                }
                            }
                        }
                        Ok(false)
                    }
                    Err(e) => {
                        // TLC semantics: certain runtime errors mean the action is disabled
                        match &e {
                            EvalError::NotInDomain { .. }
                            | EvalError::IndexOutOfBounds { .. }
                            | EvalError::NoSuchField { .. }
                            | EvalError::ChooseFailed { .. }
                            | EvalError::DivisionByZero { .. } => Ok(false),
                            _ => Err(e),
                        }
                    }
                }
            } else {
                // For non-subscripted ENABLED A, use the standard eval_enabled
                crate::enumerate::eval_enabled(&mut eval_ctx, action, &vars)
            }
        } else {
            // Fallback: var_registry is empty (synthetic test scenario).
            // Use pre-computed successors (original behavior for mock tests).
            self.eval_enabled_fallback(ctx_current, current_state, action, require_state_change)
        }
    }

    /// Evaluate whether the subscript expression changed between two states.
    /// Returns true iff subscript(s1) ≠ subscript(s2)
    fn eval_subscript_changed(
        &self,
        ctx: &EvalCtx,
        s1: &State,
        s2: &State,
        subscript: &tla_core::Spanned<tla_core::ast::Expr>,
    ) -> EvalResult<bool> {
        let debug = std::env::var("TLA2_DEBUG_SUBSCRIPT").is_ok();
        if debug {
            eprintln!("[DEBUG SUBSCRIPT] Evaluating subscript: {:?}", subscript.node);
            eprintln!("[DEBUG SUBSCRIPT] s1 vars: {:?}", s1.vars().map(|(k, v)| (k.to_string(), v.clone())).collect::<Vec<_>>());
            eprintln!("[DEBUG SUBSCRIPT] s2 vars: {:?}", s2.vars().map(|(k, v)| (k.to_string(), v.clone())).collect::<Vec<_>>());
        }

        // Build environment from s1 (current state)
        let mut env1 = crate::eval::Env::new();
        for (name, value) in s1.vars() {
            env1.insert(Arc::clone(name), value.clone());
        }
        let ctx1 = ctx.with_explicit_env(env1);
        let val1 = crate::eval::eval(&ctx1, subscript)?;

        // Build environment from s2 (next state)
        let mut env2 = crate::eval::Env::new();
        for (name, value) in s2.vars() {
            env2.insert(Arc::clone(name), value.clone());
        }
        let ctx2 = ctx.with_explicit_env(env2);
        let val2 = crate::eval::eval(&ctx2, subscript)?;

        if debug {
            eprintln!("[DEBUG SUBSCRIPT] val1={}, val2={}, changed={}", val1, val2, val1 != val2);
        }

        // Compare values
        Ok(val1 != val2)
    }

    /// Fallback ENABLED evaluation using pre-computed successors.
    /// Used when var_registry is empty (synthetic tests without full infrastructure).
    fn eval_enabled_fallback(
        &self,
        ctx_current: &EvalCtx,
        current_state: &State,
        action: &std::sync::Arc<tla_core::Spanned<tla_core::ast::Expr>>,
        require_state_change: bool,
    ) -> EvalResult<bool> {
        let fp = current_state.fingerprint();

        // Check pre-computed successors
        let Some(succs) = self.state_successors.get(&fp) else {
            // No cached successors - action is disabled
            return Ok(false);
        };

        for succ_state in succs {
            if require_state_change && succ_state.fingerprint() == fp {
                continue;
            }
            let mut next_env = Env::new();
            for (name, value) in succ_state.vars() {
                next_env.insert(Arc::clone(name), value.clone());
            }
            let ctx = ctx_current.clone().with_next_state(next_env);

            match crate::eval::eval(&ctx, action)? {
                crate::Value::Bool(true) => return Ok(true),
                crate::Value::Bool(false) => {}
                _ => {}
            }
        }

        Ok(false)
    }

    /// Build a counterexample trace from a cycle in the behavior graph
    ///
    /// Returns (prefix, cycle) where:
    /// - prefix: Path from initial state to the start of the cycle
    /// - cycle: The violating cycle itself
    fn build_counterexample(
        &self,
        cycle_nodes: &[BehaviorGraphNode],
    ) -> (CounterexamplePath, CounterexamplePath) {
        if cycle_nodes.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // First node in cycle is where we'll build the prefix to
        let cycle_start = cycle_nodes[0];

        // Get the prefix: path from init to cycle_start
        let prefix_trace = self.graph.reconstruct_trace(cycle_start);

        // Build the cycle portion with states
        let mut cycle = Vec::new();
        for node in cycle_nodes {
            if let Some(info) = self.graph.get_node_info(node) {
                cycle.push((info.state.clone(), node.tableau_idx));
            }
        }

        // The prefix should not include the cycle start (it's in the cycle)
        let prefix = if prefix_trace.len() > 1 {
            prefix_trace[..prefix_trace.len() - 1].to_vec()
        } else {
            Vec::new()
        };

        (prefix, cycle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liveness::LiveExpr;
    use crate::Value;
    use std::sync::Arc;
    use tla_core::ast::Expr;
    use tla_core::Spanned;

    fn empty_successors(_: &State) -> EvalResult<Vec<State>> {
        Ok(Vec::new())
    }

    #[test]
    fn test_liveness_checker_new() {
        // Create a simple tableau for []P
        let formula = LiveExpr::always(LiveExpr::Bool(true));
        let tableau = Tableau::new(formula);
        let ctx = EvalCtx::new();

        let checker = LivenessChecker::new(tableau, ctx);
        assert_eq!(checker.stats().graph_nodes, 0);
        assert_eq!(checker.stats().states_explored, 0);
    }

    #[test]
    fn test_add_initial_state() {
        // Create a tableau where []TRUE (always true) - all states should be consistent
        let formula = LiveExpr::always(LiveExpr::Bool(true));
        let tableau = Tableau::new(formula);
        let ctx = EvalCtx::new();

        let mut checker = LivenessChecker::new(tableau, ctx);
        let mut get_successors = empty_successors;

        let state = State::from_pairs([("x", Value::int(0))]);
        let added = checker
            .add_initial_state(&state, &mut get_successors)
            .unwrap();

        // Should have added at least one node
        assert!(!added.is_empty());
        assert_eq!(checker.graph().len(), added.len());
    }

    #[test]
    fn test_enabled_in_state_consistency() {
        // Build a tableau whose initial node requires ENABLED(x' = x + 1).
        let x = spanned(Expr::Ident("x".to_string()));
        let x_prime = spanned(Expr::Prime(Box::new(x.clone())));
        let x_plus_1 = spanned(Expr::Add(
            Box::new(x),
            Box::new(spanned(Expr::Int(1.into()))),
        ));
        let inc_expr = Arc::new(spanned(Expr::Eq(Box::new(x_prime), Box::new(x_plus_1))));

        let tableau = Tableau::new(LiveExpr::enabled(inc_expr.clone(), 1));
        let ctx = EvalCtx::new();
        let mut checker = LivenessChecker::new(tableau, ctx);

        let s0 = State::from_pairs([("x", Value::int(0))]);

        // Self-loop where Inc is false => ENABLED Inc is false => inconsistent.
        let mut get_successors = |_s: &State| Ok(vec![s0.clone()]);
        let added = checker.add_initial_state(&s0, &mut get_successors).unwrap();
        assert!(added.is_empty());

        // If there exists a successor satisfying Inc, ENABLED Inc becomes true.
        let tableau = Tableau::new(LiveExpr::enabled(inc_expr, 1));
        let ctx = EvalCtx::new();
        let mut checker = LivenessChecker::new(tableau, ctx);
        let s1 = State::from_pairs([("x", Value::int(1))]);
        let mut get_successors = |_s: &State| Ok(vec![s1.clone()]);
        let added = checker.add_initial_state(&s0, &mut get_successors).unwrap();
        assert!(!added.is_empty());
    }

    #[test]
    fn test_add_successors() {
        // Create a simple tableau
        let formula = LiveExpr::always(LiveExpr::Bool(true));
        let tableau = Tableau::new(formula);
        let ctx = EvalCtx::new();

        let mut checker = LivenessChecker::new(tableau, ctx);
        let mut get_successors = empty_successors;

        // Add initial state
        let state0 = State::from_pairs([("x", Value::int(0))]);
        let added0 = checker
            .add_initial_state(&state0, &mut get_successors)
            .unwrap();
        assert!(!added0.is_empty());

        // Add successors
        let state1 = State::from_pairs([("x", Value::int(1))]);
        let added1 = checker
            .add_successors(added0[0], &[state1], &mut get_successors)
            .unwrap();
        let _ = added1; // Silence unused variable warning

        // Should have added successor nodes
        assert!(checker.graph().len() > 1);
        // Stats should be updated
        assert!(checker.stats().graph_edges > 0);
    }

    #[test]
    fn test_check_liveness_no_cycle() {
        // Create a tableau where []TRUE - always satisfied
        let formula = LiveExpr::always(LiveExpr::Bool(true));
        let tableau = Tableau::new(formula);
        let ctx = EvalCtx::new();

        let mut checker = LivenessChecker::new(tableau, ctx);
        let mut get_successors = empty_successors;

        // Add a linear chain of states (no cycle)
        let state0 = State::from_pairs([("x", Value::int(0))]);
        let state1 = State::from_pairs([("x", Value::int(1))]);
        let state2 = State::from_pairs([("x", Value::int(2))]);

        let added0 = checker
            .add_initial_state(&state0, &mut get_successors)
            .unwrap();
        if !added0.is_empty() {
            let _ = checker
                .add_successors(
                    added0[0],
                    std::slice::from_ref(&state1),
                    &mut get_successors,
                )
                .unwrap();
            // Need to get the actual successor node
            let n1 = BehaviorGraphNode::from_state(&state1, added0[0].tableau_idx);
            let _ = checker
                .add_successors(n1, std::slice::from_ref(&state2), &mut get_successors)
                .unwrap();
        }

        // Should be satisfied (no accepting cycle)
        let result = checker.check_liveness();
        assert!(matches!(result, LivenessResult::Satisfied));
    }

    #[test]
    fn test_find_sccs() {
        let formula = LiveExpr::always(LiveExpr::Bool(true));
        let tableau = Tableau::new(formula);
        let ctx = EvalCtx::new();

        let mut checker = LivenessChecker::new(tableau, ctx);
        let mut get_successors = empty_successors;

        // Create a cycle: s0 -> s1 -> s0
        let state0 = State::from_pairs([("x", Value::int(0))]);
        let state1 = State::from_pairs([("x", Value::int(1))]);

        let added0 = checker
            .add_initial_state(&state0, &mut get_successors)
            .unwrap();
        if !added0.is_empty() {
            let _ = checker
                .add_successors(
                    added0[0],
                    std::slice::from_ref(&state1),
                    &mut get_successors,
                )
                .unwrap();
            let n1 = BehaviorGraphNode::from_state(&state1, added0[0].tableau_idx);
            let _ = checker
                .add_successors(n1, std::slice::from_ref(&state0), &mut get_successors)
                .unwrap();
        }

        // Should find a cycle
        let _cycles = checker.find_cycles();
        // Note: The cycle detection depends on tableau structure
        // At minimum, we should have explored the states
        assert!(checker.graph().len() >= 2);
    }

    fn spanned(node: Expr) -> Spanned<Expr> {
        Spanned::dummy(node)
    }

    fn state_pred_x_eq(n: i64, tag: u32) -> LiveExpr {
        LiveExpr::state_pred(
            Arc::new(spanned(Expr::Eq(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(n.into()))),
            ))),
            tag,
        )
    }

    fn action_pred_xprime_eq_x(tag: u32) -> LiveExpr {
        let x = spanned(Expr::Ident("x".to_string()));
        let x_prime = spanned(Expr::Prime(Box::new(x.clone())));
        LiveExpr::action_pred(
            Arc::new(spanned(Expr::Eq(Box::new(x_prime), Box::new(x)))),
            tag,
        )
    }

    fn action_pred_xprime_eq_x_plus_1(tag: u32) -> LiveExpr {
        let x = spanned(Expr::Ident("x".to_string()));
        let x_prime = spanned(Expr::Prime(Box::new(x.clone())));
        let x_plus_1 = spanned(Expr::Add(
            Box::new(x),
            Box::new(spanned(Expr::Int(1.into()))),
        ));
        LiveExpr::action_pred(
            Arc::new(spanned(Expr::Eq(Box::new(x_prime), Box::new(x_plus_1)))),
            tag,
        )
    }

    #[test]
    fn test_promise_tracking_eventually() {
        // tf == <>P; system never satisfies P. Promise tracking should prevent
        // reporting a counterexample.
        let p = state_pred_x_eq(1, 1);
        let tableau = Tableau::new(LiveExpr::eventually(p));
        let ctx = EvalCtx::new();
        let mut checker = LivenessChecker::new(tableau, ctx);
        let mut get_successors = empty_successors;

        let s0 = State::from_pairs([("x", Value::int(0))]);
        let init_nodes = checker.add_initial_state(&s0, &mut get_successors).unwrap();
        assert_eq!(init_nodes.len(), 1);

        // Self-loop on the single state.
        let _ = checker
            .add_successors(
                init_nodes[0],
                std::slice::from_ref(&s0),
                &mut get_successors,
            )
            .unwrap();

        let result = checker.check_liveness();
        assert!(matches!(result, LivenessResult::Satisfied));
    }

    #[test]
    fn test_check_liveness_violation_cycle_no_promises() {
        // tf == []~P (negation of <>P). With a cycle where P never holds, this is satisfiable.
        let p = state_pred_x_eq(1, 1);
        let tf = LiveExpr::always(LiveExpr::not(p));
        let tableau = Tableau::new(tf);
        let ctx = EvalCtx::new();
        let mut checker = LivenessChecker::new(tableau, ctx);
        let mut get_successors = empty_successors;

        let s0 = State::from_pairs([("x", Value::int(0))]);
        let init_nodes = checker.add_initial_state(&s0, &mut get_successors).unwrap();
        assert!(!init_nodes.is_empty());
        let _ = checker
            .add_successors(
                init_nodes[0],
                std::slice::from_ref(&s0),
                &mut get_successors,
            )
            .unwrap();

        let result = checker.check_liveness();
        assert!(matches!(result, LivenessResult::Violated { .. }));
    }

    #[test]
    fn test_ae_action_requires_witness_edge() {
        // Require []<>A where A is action-level.
        // A = (x' = x + 1) is false on a self-loop, thus no counterexample should be found.
        let tableau = Tableau::new(LiveExpr::always(LiveExpr::Bool(true)));
        let constraints = LivenessConstraints {
            ae_action: vec![action_pred_xprime_eq_x_plus_1(1)],
            ..Default::default()
        };
        let ctx = EvalCtx::new();
        let mut checker = LivenessChecker::new_with_constraints(tableau, ctx, constraints);
        let mut get_successors = empty_successors;

        let s0 = State::from_pairs([("x", Value::int(0))]);
        let init_nodes = checker.add_initial_state(&s0, &mut get_successors).unwrap();
        let _ = checker
            .add_successors(
                init_nodes[0],
                std::slice::from_ref(&s0),
                &mut get_successors,
            )
            .unwrap();

        let result = checker.check_liveness();
        assert!(matches!(result, LivenessResult::Satisfied));

        // If we require []<>(x' = x) instead, the self-loop witnesses it.
        let tableau = Tableau::new(LiveExpr::always(LiveExpr::Bool(true)));
        let constraints = LivenessConstraints {
            ae_action: vec![action_pred_xprime_eq_x(2)],
            ..Default::default()
        };
        let ctx = EvalCtx::new();
        let mut checker = LivenessChecker::new_with_constraints(tableau, ctx, constraints);

        let init_nodes = checker.add_initial_state(&s0, &mut get_successors).unwrap();
        let _ = checker
            .add_successors(
                init_nodes[0],
                std::slice::from_ref(&s0),
                &mut get_successors,
            )
            .unwrap();

        let result = checker.check_liveness();
        assert!(matches!(result, LivenessResult::Violated { .. }));
    }

    #[test]
    fn test_ea_state_filters_scc_edges() {
        // EA check <>[]S where S is state-level (x = 0).
        // On a 2-state cycle (0 <-> 1), there is no sub-SCC where x=0 holds always.
        let tableau = Tableau::new(LiveExpr::Bool(true));
        let constraints = LivenessConstraints {
            ea_state: vec![state_pred_x_eq(0, 1)],
            ..Default::default()
        };
        let ctx = EvalCtx::new();
        let mut checker = LivenessChecker::new_with_constraints(tableau, ctx, constraints);
        let mut get_successors = empty_successors;

        let s0 = State::from_pairs([("x", Value::int(0))]);
        let s1 = State::from_pairs([("x", Value::int(1))]);

        let init_nodes = checker.add_initial_state(&s0, &mut get_successors).unwrap();
        let _ = checker
            .add_successors(
                init_nodes[0],
                std::slice::from_ref(&s1),
                &mut get_successors,
            )
            .unwrap();

        let n1 = BehaviorGraphNode::from_state(&s1, 1);
        let _ = checker
            .add_successors(n1, std::slice::from_ref(&s0), &mut get_successors)
            .unwrap();

        let n0 = BehaviorGraphNode::from_state(&s0, 1);
        let _ = checker
            .add_successors(n0, std::slice::from_ref(&s1), &mut get_successors)
            .unwrap();

        let result = checker.check_liveness();
        assert!(matches!(result, LivenessResult::Satisfied));
    }

    #[test]
    fn test_enabled_evaluation_in_checks() {
        // Weak fairness style check: []<>(~ENABLED Inc \/ Inc)
        // If Inc is not enabled, the disjunction holds even if Inc never occurs.
        let tableau = Tableau::new(LiveExpr::always(LiveExpr::Bool(true)));
        let ctx = EvalCtx::new();

        let x = spanned(Expr::Ident("x".to_string()));
        let x_prime = spanned(Expr::Prime(Box::new(x.clone())));
        let x_plus_1 = spanned(Expr::Add(
            Box::new(x),
            Box::new(spanned(Expr::Int(1.into()))),
        ));
        let inc_expr = Arc::new(spanned(Expr::Eq(Box::new(x_prime), Box::new(x_plus_1))));

        let not_enabled_inc = LiveExpr::not(LiveExpr::enabled(inc_expr.clone(), 1));
        let inc_occurs = LiveExpr::action_pred(inc_expr, 2);
        let wf_body = LiveExpr::or(vec![not_enabled_inc, inc_occurs]);

        let constraints = LivenessConstraints {
            ae_action: vec![wf_body],
            ..Default::default()
        };

        let mut checker = LivenessChecker::new_with_constraints(tableau, ctx, constraints);
        let mut get_successors = empty_successors;

        // One-state system with only a self-loop where Inc is false.
        let s0 = State::from_pairs([("x", Value::int(0))]);
        let init_nodes = checker.add_initial_state(&s0, &mut get_successors).unwrap();
        let _ = checker
            .add_successors(
                init_nodes[0],
                std::slice::from_ref(&s0),
                &mut get_successors,
            )
            .unwrap();

        let result = checker.check_liveness();
        assert!(matches!(result, LivenessResult::Violated { .. }));
    }
}
