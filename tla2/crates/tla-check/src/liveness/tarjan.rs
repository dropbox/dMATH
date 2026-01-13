//! Tarjan's algorithm for strongly connected component detection
//!
//! This module implements an iterative version of Tarjan's algorithm to find
//! strongly connected components (SCCs) in the behavior graph. SCCs are cycles
//! in the product graph that may indicate liveness violations.
//!
//! # Algorithm Overview
//!
//! Tarjan's algorithm uses depth-first search to find SCCs in O(V + E) time.
//! This implementation is iterative (rather than recursive) to handle large
//! graphs without stack overflow.
//!
//! The key insight is that an SCC is identified when we find a node that is
//! its own "low link" - meaning it can reach itself through a cycle.
//!
//! # TLC Reference
//!
//! This follows TLC's implementation in:
//! - `tlc2/tool/liveness/LiveWorker.java` - checkSccs method
//!
//! # References
//!
//! - Tarjan, R. E. (1972). "Depth-first search and linear graph algorithms"
//! - <https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm>

use super::behavior_graph::{BehaviorGraph, BehaviorGraphNode};
use std::collections::{HashMap, HashSet};

/// Optional edge filter predicate for Tarjan's algorithm
type EdgeFilter<'a> = Option<&'a dyn Fn(&BehaviorGraphNode, &BehaviorGraphNode) -> bool>;

/// A strongly connected component in the behavior graph
///
/// An SCC is a maximal set of nodes where every node can reach every other node.
/// In the context of liveness checking, an SCC represents a potential cycle in
/// the product graph (state × tableau).
#[derive(Debug, Clone)]
pub struct Scc {
    /// The nodes in this SCC
    nodes: Vec<BehaviorGraphNode>,
}

impl Scc {
    /// Create a new SCC from a list of nodes
    pub fn new(nodes: Vec<BehaviorGraphNode>) -> Self {
        Self { nodes }
    }

    /// Get the nodes in this SCC
    pub fn nodes(&self) -> &[BehaviorGraphNode] {
        &self.nodes
    }

    /// Get the number of nodes in this SCC
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if this SCC is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Check if this SCC is trivial (single node with no self-loop)
    ///
    /// A trivial SCC has exactly one node and no edge to itself.
    /// Trivial SCCs are not considered cycles for liveness checking.
    pub fn is_trivial(&self, graph: &BehaviorGraph) -> bool {
        if self.nodes.len() != 1 {
            return false;
        }

        // Check if the single node has a self-loop
        let node = &self.nodes[0];
        if let Some(info) = graph.get_node_info(node) {
            !info.successors.contains(node)
        } else {
            true // Node not in graph, consider trivial
        }
    }

    /// Check if this SCC contains an accepting node
    ///
    /// An accepting SCC contains at least one accepting tableau node,
    /// which indicates a potential liveness violation.
    pub fn contains_accepting<F>(&self, is_accepting: F) -> bool
    where
        F: Fn(&BehaviorGraphNode) -> bool,
    {
        self.nodes.iter().any(is_accepting)
    }
}

/// Statistics for SCC detection
#[derive(Debug, Clone, Default)]
pub struct TarjanStats {
    /// Number of SCCs found
    pub scc_count: usize,
    /// Number of non-trivial SCCs (actual cycles)
    pub nontrivial_sccs: usize,
    /// Largest SCC size
    pub max_scc_size: usize,
    /// Total nodes processed
    pub nodes_processed: usize,
}

/// Result of Tarjan's algorithm
#[derive(Debug, Clone)]
pub struct TarjanResult {
    /// All SCCs found, in reverse topological order
    pub sccs: Vec<Scc>,
    /// Statistics about the computation
    pub stats: TarjanStats,
}

/// State for a node during Tarjan's algorithm
#[derive(Debug, Clone)]
struct NodeState {
    /// Index when node was first visited (discovery time)
    index: usize,
    /// Lowest index reachable from this node
    low_link: usize,
    /// Whether node is currently on the stack
    on_stack: bool,
}

/// Frame for iterative DFS traversal
#[derive(Debug)]
enum DfsFrame {
    /// First visit to a node - discover and push to stack
    Visit(BehaviorGraphNode),
    /// After processing all successors - check for SCC root
    PostProcess { node: BehaviorGraphNode },
    /// Process next successor
    ProcessSuccessor {
        node: BehaviorGraphNode,
        successors: Vec<BehaviorGraphNode>,
        succ_idx: usize,
    },
}

/// Find all strongly connected components using iterative Tarjan's algorithm
///
/// This is the main entry point for SCC detection. It returns all SCCs in
/// the graph in reverse topological order (i.e., if SCC A can reach SCC B,
/// then B comes before A in the output).
///
/// # Arguments
///
/// * `graph` - The behavior graph to analyze
///
/// # Returns
///
/// A `TarjanResult` containing all SCCs and statistics
pub fn find_sccs(graph: &BehaviorGraph) -> TarjanResult {
    let finder = TarjanFinder::new(graph, None);
    finder.find_all()
}

/// Find SCCs using Tarjan's algorithm, restricting edges with a predicate.
///
/// Only edges `(from -> to)` for which `edge_filter(from, to)` returns true are
/// considered during SCC construction.
pub fn find_sccs_with_edge_filter(
    graph: &BehaviorGraph,
    edge_filter: &dyn Fn(&BehaviorGraphNode, &BehaviorGraphNode) -> bool,
) -> TarjanResult {
    let finder = TarjanFinder::new(graph, Some(edge_filter));
    finder.find_all()
}

/// Find non-trivial SCCs (actual cycles) in the behavior graph
///
/// This is a convenience function that filters out trivial SCCs
/// (single nodes without self-loops).
pub fn find_cycles(graph: &BehaviorGraph) -> Vec<Scc> {
    let result = find_sccs(graph);
    result
        .sccs
        .into_iter()
        .filter(|scc| !scc.is_trivial(graph))
        .collect()
}

/// Internal state for Tarjan's algorithm
struct TarjanFinder<'a> {
    graph: &'a BehaviorGraph,
    edge_filter: EdgeFilter<'a>,
    /// Node states (index, low_link, on_stack)
    node_states: HashMap<BehaviorGraphNode, NodeState>,
    /// Current index counter
    index: usize,
    /// The Tarjan stack (nodes potentially in current SCC)
    stack: Vec<BehaviorGraphNode>,
    /// Found SCCs
    sccs: Vec<Scc>,
    /// Statistics
    stats: TarjanStats,
}

impl<'a> TarjanFinder<'a> {
    fn new(graph: &'a BehaviorGraph, edge_filter: EdgeFilter<'a>) -> Self {
        Self {
            graph,
            edge_filter,
            node_states: HashMap::new(),
            index: 0,
            stack: Vec::new(),
            sccs: Vec::new(),
            stats: TarjanStats::default(),
        }
    }

    /// Run Tarjan's algorithm on all nodes
    fn find_all(mut self) -> TarjanResult {
        // Collect all nodes to iterate over (avoiding borrow issues)
        let all_nodes: Vec<BehaviorGraphNode> = self.graph.nodes().map(|(n, _)| *n).collect();

        // Start DFS from each unvisited node
        for node in all_nodes {
            if !self.node_states.contains_key(&node) {
                self.tarjan_iterative(node);
            }
        }

        self.stats.scc_count = self.sccs.len();

        TarjanResult {
            sccs: self.sccs,
            stats: self.stats,
        }
    }

    /// Iterative Tarjan's algorithm starting from a given node
    ///
    /// This replaces the recursive version to avoid stack overflow on large graphs.
    /// Uses an explicit DFS stack with frames that track the traversal state.
    fn tarjan_iterative(&mut self, start: BehaviorGraphNode) {
        let mut dfs_stack: Vec<DfsFrame> = vec![DfsFrame::Visit(start)];

        while let Some(frame) = dfs_stack.pop() {
            match frame {
                DfsFrame::Visit(node) => {
                    // Skip if already visited
                    if self.node_states.contains_key(&node) {
                        continue;
                    }

                    // Initialize node state
                    let node_index = self.index;
                    self.index += 1;
                    self.node_states.insert(
                        node,
                        NodeState {
                            index: node_index,
                            low_link: node_index,
                            on_stack: true,
                        },
                    );
                    self.stack.push(node);
                    self.stats.nodes_processed += 1;

                    // Get successors from the graph
                    let successors: Vec<BehaviorGraphNode> = self
                        .graph
                        .get_node_info(&node)
                        .map(|info| {
                            if let Some(edge_filter) = self.edge_filter {
                                info.successors
                                    .iter()
                                    .copied()
                                    .filter(|succ| edge_filter(&node, succ))
                                    .collect()
                            } else {
                                info.successors.iter().copied().collect()
                            }
                        })
                        .unwrap_or_default();

                    // Push post-process frame (will be executed after all successors)
                    dfs_stack.push(DfsFrame::PostProcess { node });

                    // Push frame to process successors
                    if !successors.is_empty() {
                        dfs_stack.push(DfsFrame::ProcessSuccessor {
                            node,
                            successors,
                            succ_idx: 0,
                        });
                    }
                }

                DfsFrame::ProcessSuccessor {
                    node,
                    successors,
                    succ_idx,
                } => {
                    if succ_idx >= successors.len() {
                        continue;
                    }

                    let succ = successors[succ_idx];

                    // Push frame for next successor
                    if succ_idx + 1 < successors.len() {
                        dfs_stack.push(DfsFrame::ProcessSuccessor {
                            node,
                            successors: successors.clone(),
                            succ_idx: succ_idx + 1,
                        });
                    }

                    if let Some(succ_state) = self.node_states.get(&succ) {
                        // Successor already visited
                        if succ_state.on_stack {
                            // Successor is on stack - update low_link
                            let succ_index = succ_state.index;
                            if let Some(node_state) = self.node_states.get_mut(&node) {
                                node_state.low_link = node_state.low_link.min(succ_index);
                            }
                        }
                    } else {
                        // Successor not visited - visit it first, then update low_link
                        dfs_stack.push(DfsFrame::Visit(succ));

                        // After visiting successor, update our low_link
                        // This is handled in PostProcess
                    }
                }

                DfsFrame::PostProcess { node, .. } => {
                    // Update low_link from all successors that have been visited
                    let successors: Vec<BehaviorGraphNode> = self
                        .graph
                        .get_node_info(&node)
                        .map(|info| {
                            if let Some(edge_filter) = self.edge_filter {
                                info.successors
                                    .iter()
                                    .copied()
                                    .filter(|succ| edge_filter(&node, succ))
                                    .collect()
                            } else {
                                info.successors.iter().copied().collect()
                            }
                        })
                        .unwrap_or_default();

                    // Get node's current index before iterating
                    let node_index = self.node_states.get(&node).map(|s| s.index).unwrap_or(0);

                    // Collect all lowlink updates from successors
                    let mut min_low_link = self
                        .node_states
                        .get(&node)
                        .map(|s| s.low_link)
                        .unwrap_or(usize::MAX);

                    for succ in &successors {
                        if let Some(succ_state) = self.node_states.get(succ) {
                            if succ_state.on_stack {
                                // Successor on stack - use its low_link (can reach back further)
                                min_low_link = min_low_link.min(succ_state.low_link);
                            } else {
                                // Successor processed (in an SCC) - use its low_link
                                // Only update if successor was discovered after us
                                // (i.e., it's a tree edge, not a cross edge to another SCC)
                                if succ_state.index > node_index {
                                    min_low_link = min_low_link.min(succ_state.low_link);
                                }
                            }
                        }
                    }

                    // Apply the minimum low_link
                    if let Some(node_state) = self.node_states.get_mut(&node) {
                        node_state.low_link = min_low_link;
                    }

                    // Check if this node is the root of an SCC
                    let (node_index, node_low_link) = {
                        let state = self.node_states.get(&node).unwrap();
                        (state.index, state.low_link)
                    };

                    if node_low_link == node_index {
                        // Found an SCC - pop nodes from stack until we reach this node
                        let mut scc_nodes = Vec::new();

                        loop {
                            let top = self.stack.pop().expect("Stack should not be empty");

                            if let Some(top_state) = self.node_states.get_mut(&top) {
                                top_state.on_stack = false;
                            }

                            scc_nodes.push(top);

                            if top == node {
                                break;
                            }
                        }

                        // Track statistics
                        let scc_size = scc_nodes.len();
                        if scc_size > self.stats.max_scc_size {
                            self.stats.max_scc_size = scc_size;
                        }
                        if scc_size > 1 {
                            self.stats.nontrivial_sccs += 1;
                        }

                        self.sccs.push(Scc::new(scc_nodes));
                    }
                }
            }
        }
    }
}

/// Find SCCs that contain accepting nodes (potential liveness violations)
///
/// This is the main function for liveness checking. It finds all non-trivial
/// SCCs that contain at least one accepting tableau node.
///
/// # Arguments
///
/// * `graph` - The behavior graph
/// * `is_accepting` - Function to check if a node is accepting
///
/// # Returns
///
/// Vector of SCCs that could indicate liveness violations
pub fn find_accepting_sccs<F>(graph: &BehaviorGraph, is_accepting: F) -> Vec<Scc>
where
    F: Fn(&BehaviorGraphNode) -> bool,
{
    let result = find_sccs(graph);

    result
        .sccs
        .into_iter()
        .filter(|scc| !scc.is_trivial(graph) && scc.contains_accepting(&is_accepting))
        .collect()
}

/// Check if a path from start to end exists within an SCC
///
/// This is used for counterexample construction - finding the actual cycle
/// within an SCC.
pub fn find_cycle_in_scc(graph: &BehaviorGraph, scc: &Scc) -> Option<Vec<BehaviorGraphNode>> {
    if scc.len() < 2 {
        // Check for self-loop on single node
        if scc.len() == 1 {
            let node = scc.nodes[0];
            if let Some(info) = graph.get_node_info(&node) {
                if info.successors.contains(&node) {
                    return Some(vec![node, node]);
                }
            }
        }
        return None;
    }

    // Build set of SCC nodes for fast lookup
    let scc_set: HashSet<BehaviorGraphNode> = scc.nodes.iter().copied().collect();

    // Use BFS to find a cycle within the SCC
    // Start from first node, find path back to it
    let start = scc.nodes[0];

    // BFS to find path back to start
    let mut visited: HashSet<BehaviorGraphNode> = HashSet::new();
    let mut parent: HashMap<BehaviorGraphNode, BehaviorGraphNode> = HashMap::new();
    let mut queue = std::collections::VecDeque::new();

    // Start from successors of start node
    if let Some(info) = graph.get_node_info(&start) {
        for succ in &info.successors {
            if scc_set.contains(succ) {
                visited.insert(*succ);
                parent.insert(*succ, start);
                queue.push_back(*succ);
            }
        }
    }

    while let Some(current) = queue.pop_front() {
        if current == start {
            // Found cycle! Reconstruct path by following parents back from current
            // The parent chain goes: current <- ... <- successor_of_start <- start
            // We need: start -> successor_of_start -> ... -> current (which is start)
            let mut path = Vec::new();
            let mut node = current;

            // Walk back through parents until we reach start
            while let Some(&p) = parent.get(&node) {
                path.push(node);
                node = p;
                if node == start {
                    break;
                }
            }
            path.push(start);

            // Reverse to get forward direction: start -> ... -> start
            path.reverse();
            return Some(path);
        }

        if let Some(info) = graph.get_node_info(&current) {
            for succ in &info.successors {
                if scc_set.contains(succ) && !visited.contains(succ) {
                    visited.insert(*succ);
                    parent.insert(*succ, current);
                    queue.push_back(*succ);
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::State;
    use crate::Value;

    fn make_state(x: i64) -> State {
        State::from_pairs([("x", Value::int(x))])
    }

    #[test]
    fn test_empty_graph() {
        let graph = BehaviorGraph::new();
        let result = find_sccs(&graph);

        assert_eq!(result.sccs.len(), 0);
        assert_eq!(result.stats.nodes_processed, 0);
    }

    #[test]
    fn test_single_node_no_self_loop() {
        let mut graph = BehaviorGraph::new();
        let s0 = make_state(0);
        graph.add_init_node(&s0, 0);

        let result = find_sccs(&graph);

        // Single node with no self-loop is trivial SCC
        assert_eq!(result.sccs.len(), 1);
        assert_eq!(result.sccs[0].len(), 1);
        assert!(result.sccs[0].is_trivial(&graph));

        // find_cycles should filter out trivial SCCs
        let cycles = find_cycles(&graph);
        assert_eq!(cycles.len(), 0);
    }

    #[test]
    fn test_single_node_with_self_loop() {
        let mut graph = BehaviorGraph::new();
        let s0 = make_state(0);
        graph.add_init_node(&s0, 0);

        let n0 = BehaviorGraphNode::from_state(&s0, 0);
        // Add self-loop
        graph.add_successor(n0, &s0, 0);

        let result = find_sccs(&graph);

        // Single node with self-loop is non-trivial SCC
        assert_eq!(result.sccs.len(), 1);
        assert_eq!(result.sccs[0].len(), 1);
        assert!(!result.sccs[0].is_trivial(&graph));

        let cycles = find_cycles(&graph);
        assert_eq!(cycles.len(), 1);
    }

    #[test]
    fn test_two_node_cycle() {
        let mut graph = BehaviorGraph::new();
        let s0 = make_state(0);
        let s1 = make_state(1);

        graph.add_init_node(&s0, 0);
        let n0 = BehaviorGraphNode::from_state(&s0, 0);

        graph.add_successor(n0, &s1, 0);
        let n1 = BehaviorGraphNode::from_state(&s1, 0);

        // Add back edge to form cycle
        graph.add_successor(n1, &s0, 0);

        let result = find_sccs(&graph);

        // Both nodes should be in one SCC
        assert_eq!(result.sccs.len(), 1);
        assert_eq!(result.sccs[0].len(), 2);
        assert!(!result.sccs[0].is_trivial(&graph));

        let cycles = find_cycles(&graph);
        assert_eq!(cycles.len(), 1);
    }

    #[test]
    fn test_linear_chain_no_cycle() {
        let mut graph = BehaviorGraph::new();
        let s0 = make_state(0);
        let s1 = make_state(1);
        let s2 = make_state(2);

        graph.add_init_node(&s0, 0);
        let n0 = BehaviorGraphNode::from_state(&s0, 0);

        graph.add_successor(n0, &s1, 0);
        let n1 = BehaviorGraphNode::from_state(&s1, 0);

        graph.add_successor(n1, &s2, 0);

        let result = find_sccs(&graph);

        // Three trivial SCCs (one per node)
        assert_eq!(result.sccs.len(), 3);
        for scc in &result.sccs {
            assert_eq!(scc.len(), 1);
            assert!(scc.is_trivial(&graph));
        }

        let cycles = find_cycles(&graph);
        assert_eq!(cycles.len(), 0);
    }

    #[test]
    fn test_two_separate_cycles() {
        let mut graph = BehaviorGraph::new();

        // First cycle: s0 -> s1 -> s0
        let s0 = make_state(0);
        let s1 = make_state(1);
        graph.add_init_node(&s0, 0);
        let n0 = BehaviorGraphNode::from_state(&s0, 0);
        graph.add_successor(n0, &s1, 0);
        let n1 = BehaviorGraphNode::from_state(&s1, 0);
        graph.add_successor(n1, &s0, 0);

        // Second cycle: s2 -> s3 -> s2 (different tableau index)
        let s2 = make_state(2);
        let s3 = make_state(3);
        graph.add_init_node(&s2, 1);
        let n2 = BehaviorGraphNode::from_state(&s2, 1);
        graph.add_successor(n2, &s3, 1);
        let n3 = BehaviorGraphNode::from_state(&s3, 1);
        graph.add_successor(n3, &s2, 1);

        let result = find_sccs(&graph);

        // Two non-trivial SCCs
        let cycles = find_cycles(&graph);
        assert_eq!(cycles.len(), 2);

        assert_eq!(result.stats.nontrivial_sccs, 2);
    }

    #[test]
    fn test_tableau_differentiation() {
        // Same state with different tableau indices should be different nodes
        let mut graph = BehaviorGraph::new();
        let s0 = make_state(0);

        graph.add_init_node(&s0, 0);
        graph.add_init_node(&s0, 1);

        let n0_t0 = BehaviorGraphNode::from_state(&s0, 0);
        let n0_t1 = BehaviorGraphNode::from_state(&s0, 1);

        assert_ne!(n0_t0, n0_t1);

        // Link them in a cycle
        graph.add_successor(n0_t0, &s0, 1);
        graph.add_successor(n0_t1, &s0, 0);

        let cycles = find_cycles(&graph);
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].len(), 2);
    }

    #[test]
    fn test_edge_filter_excludes_back_edge_from_scc() {
        // Base graph is a 2-node cycle. With an edge filter that removes the
        // back-edge, the filtered graph is acyclic and must not produce a
        // non-trivial SCC.
        let mut graph = BehaviorGraph::new();
        let s0 = make_state(0);
        let s1 = make_state(1);

        graph.add_init_node(&s0, 0);
        let n0 = BehaviorGraphNode::from_state(&s0, 0);
        graph.add_successor(n0, &s1, 0);
        let n1 = BehaviorGraphNode::from_state(&s1, 0);
        graph.add_successor(n1, &s0, 0);

        let result = find_sccs_with_edge_filter(&graph, &|from, to| !(*from == n1 && *to == n0));

        assert_eq!(result.sccs.len(), 2);
        assert!(result.sccs.iter().all(|scc| scc.len() == 1));
    }

    #[test]
    fn test_find_cycle_in_scc() {
        let mut graph = BehaviorGraph::new();
        let s0 = make_state(0);
        let s1 = make_state(1);
        let s2 = make_state(2);

        graph.add_init_node(&s0, 0);
        let n0 = BehaviorGraphNode::from_state(&s0, 0);

        graph.add_successor(n0, &s1, 0);
        let n1 = BehaviorGraphNode::from_state(&s1, 0);

        graph.add_successor(n1, &s2, 0);
        let n2 = BehaviorGraphNode::from_state(&s2, 0);

        graph.add_successor(n2, &s0, 0);

        let cycles = find_cycles(&graph);
        assert_eq!(cycles.len(), 1);

        let cycle_path = find_cycle_in_scc(&graph, &cycles[0]);
        assert!(cycle_path.is_some());

        let path = cycle_path.unwrap();
        // Cycle should start and end at same node
        assert_eq!(path.first(), path.last());
        // Path should have at least 2 elements (start and back to start)
        assert!(path.len() >= 2);
    }

    #[test]
    fn test_accepting_scc_detection() {
        let mut graph = BehaviorGraph::new();
        let s0 = make_state(0);
        let s1 = make_state(1);

        graph.add_init_node(&s0, 0);
        let n0 = BehaviorGraphNode::from_state(&s0, 0);

        graph.add_successor(n0, &s1, 0);
        let n1 = BehaviorGraphNode::from_state(&s1, 0);

        graph.add_successor(n1, &s0, 0);

        // All nodes accepting
        let accepting_sccs = find_accepting_sccs(&graph, |_| true);
        assert_eq!(accepting_sccs.len(), 1);

        // No nodes accepting
        let accepting_sccs = find_accepting_sccs(&graph, |_| false);
        assert_eq!(accepting_sccs.len(), 0);

        // Only tableau index 0 accepting
        let accepting_sccs = find_accepting_sccs(&graph, |n| n.tableau_idx == 0);
        assert_eq!(accepting_sccs.len(), 1);
    }

    #[test]
    fn test_complex_graph() {
        // A more complex graph:
        //     ┌─────────────┐
        //     ▼             │
        //     0 ──▶ 1 ──▶ 2─┘
        //     │     │     │
        //     ▼     ▼     ▼
        //     3 ──▶ 4 ◀── 5
        //     │           │
        //     └───────────┘
        //
        // SCCs: {0,1,2}, {3,4,5}

        let mut graph = BehaviorGraph::new();
        let states: Vec<State> = (0..6).map(make_state).collect();

        graph.add_init_node(&states[0], 0);
        let n0 = BehaviorGraphNode::from_state(&states[0], 0);

        // First cycle: 0 -> 1 -> 2 -> 0
        graph.add_successor(n0, &states[1], 0);
        let n1 = BehaviorGraphNode::from_state(&states[1], 0);
        graph.add_successor(n1, &states[2], 0);
        let n2 = BehaviorGraphNode::from_state(&states[2], 0);
        graph.add_successor(n2, &states[0], 0);

        // Cross edges down
        graph.add_successor(n0, &states[3], 0);
        graph.add_successor(n1, &states[4], 0);
        graph.add_successor(n2, &states[5], 0);

        // Second cycle: 3 -> 4 <- 5 -> 3
        let n3 = BehaviorGraphNode::from_state(&states[3], 0);
        graph.add_successor(n3, &states[4], 0);
        let n5 = BehaviorGraphNode::from_state(&states[5], 0);
        graph.add_successor(n5, &states[4], 0);
        graph.add_successor(n5, &states[3], 0);
        graph.add_successor(n3, &states[5], 0);

        let result = find_sccs(&graph);
        let cycles = find_cycles(&graph);

        // Should have 2 non-trivial SCCs
        // (node 4 might be separate since it has no outgoing edges in cycle)
        assert!(!cycles.is_empty());
        assert!(result.stats.nodes_processed == 6);
    }
}
