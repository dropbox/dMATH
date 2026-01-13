//! Behavior graph for liveness checking
//!
//! The behavior graph is the product of the state graph and the tableau automaton.
//! Each node is a `(state, tableau_node)` pair, and transitions follow both:
//! - The state graph (via the Next relation)
//! - The tableau automaton (via tableau node successors)
//!
//! A liveness violation exists iff there is a reachable accepting cycle in this
//! product graph.
//!
//! # TLC Reference
//!
//! This follows TLC's implementation in:
//! - `tlc2/tool/liveness/GraphNode.java` - Node representation
//! - `tlc2/tool/liveness/TableauNodePtrTable.java` - (fp, tidx) tracking
//! - `tlc2/tool/liveness/LiveCheck.java` - Product graph construction

use crate::state::{Fingerprint, State};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// A node in the behavior graph: (state fingerprint, tableau node index) pair
///
/// This is the fundamental unit for liveness checking. Two behavior graph nodes
/// are equal iff they have the same state fingerprint AND the same tableau index.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BehaviorGraphNode {
    /// Fingerprint of the TLA+ state
    pub state_fp: Fingerprint,
    /// Index of the tableau node
    pub tableau_idx: usize,
}

impl BehaviorGraphNode {
    /// Create a new behavior graph node
    pub fn new(state_fp: Fingerprint, tableau_idx: usize) -> Self {
        Self {
            state_fp,
            tableau_idx,
        }
    }

    /// Create from a state and tableau index
    pub fn from_state(state: &State, tableau_idx: usize) -> Self {
        Self::new(state.fingerprint(), tableau_idx)
    }
}

impl fmt::Debug for BehaviorGraphNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BG({}, t{})", self.state_fp, self.tableau_idx)
    }
}

impl fmt::Display for BehaviorGraphNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, t{})", self.state_fp, self.tableau_idx)
    }
}

/// The behavior graph: product of state graph × tableau
///
/// This structure tracks:
/// - All visited (state, tableau) pairs
/// - Transitions between pairs (for SCC detection)
/// - Parent pointers (for counterexample trace reconstruction)
#[derive(Debug, Clone)]
pub struct BehaviorGraph {
    /// All seen behavior graph nodes, with their full states for later access
    nodes: HashMap<BehaviorGraphNode, NodeInfo>,
    /// Initial nodes in the behavior graph
    init_nodes: Vec<BehaviorGraphNode>,
}

/// Information stored for each behavior graph node
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// The full state (for trace reconstruction)
    pub state: State,
    /// Successor nodes (HashSet for O(1) membership check)
    pub successors: HashSet<BehaviorGraphNode>,
    /// Parent node (for trace reconstruction)
    pub parent: Option<BehaviorGraphNode>,
    /// BFS depth
    pub depth: usize,
}

impl BehaviorGraph {
    /// Create a new empty behavior graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            init_nodes: Vec::new(),
        }
    }

    /// Add an initial node to the behavior graph
    ///
    /// Returns true if the node was newly added, false if it already existed.
    pub fn add_init_node(&mut self, state: &State, tableau_idx: usize) -> bool {
        let node = BehaviorGraphNode::from_state(state, tableau_idx);

        if self.nodes.contains_key(&node) {
            return false;
        }

        self.init_nodes.push(node);
        self.nodes.insert(
            node,
            NodeInfo {
                state: state.clone(),
                successors: HashSet::new(),
                parent: None,
                depth: 0,
            },
        );
        true
    }

    /// Add a successor node to the behavior graph
    ///
    /// Returns true if the successor was newly added, false if it already existed.
    pub fn add_successor(
        &mut self,
        from: BehaviorGraphNode,
        to_state: &State,
        to_tableau_idx: usize,
    ) -> bool {
        let to_node = BehaviorGraphNode::from_state(to_state, to_tableau_idx);

        // Record the transition from -> to (O(1) with HashSet)
        if let Some(from_info) = self.nodes.get_mut(&from) {
            from_info.successors.insert(to_node);
        }

        // If to_node is new, add it
        if self.nodes.contains_key(&to_node) {
            return false;
        }

        let from_depth = self.nodes.get(&from).map(|n| n.depth).unwrap_or(0);
        self.nodes.insert(
            to_node,
            NodeInfo {
                state: to_state.clone(),
                successors: HashSet::new(),
                parent: Some(from),
                depth: from_depth + 1,
            },
        );
        true
    }

    /// Check if a behavior graph node has been visited
    pub fn contains(&self, node: &BehaviorGraphNode) -> bool {
        self.nodes.contains_key(node)
    }

    /// Check if a (state, tableau_idx) pair has been visited
    pub fn contains_pair(&self, state_fp: Fingerprint, tableau_idx: usize) -> bool {
        let node = BehaviorGraphNode::new(state_fp, tableau_idx);
        self.nodes.contains_key(&node)
    }

    /// Get information about a node
    pub fn get_node_info(&self, node: &BehaviorGraphNode) -> Option<&NodeInfo> {
        self.nodes.get(node)
    }

    /// Get the state for a behavior graph node
    pub fn get_state(&self, node: &BehaviorGraphNode) -> Option<&State> {
        self.nodes.get(node).map(|info| &info.state)
    }

    /// Get initial nodes
    pub fn init_nodes(&self) -> &[BehaviorGraphNode] {
        &self.init_nodes
    }

    /// Get the number of nodes in the behavior graph
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the behavior graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get all nodes
    pub fn nodes(&self) -> impl Iterator<Item = (&BehaviorGraphNode, &NodeInfo)> {
        self.nodes.iter()
    }

    /// Reconstruct a trace from an initial state to the given node
    pub fn reconstruct_trace(&self, end: BehaviorGraphNode) -> Vec<(State, usize)> {
        let mut trace = Vec::new();
        let mut current = Some(end);

        while let Some(node) = current {
            if let Some(info) = self.nodes.get(&node) {
                trace.push((info.state.clone(), node.tableau_idx));
                current = info.parent;
            } else {
                break;
            }
        }

        trace.reverse();
        trace
    }
}

impl Default for BehaviorGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;

    #[test]
    fn test_behavior_graph_node_equality() {
        let s1 = State::from_pairs([("x", Value::int(1))]);
        let s2 = State::from_pairs([("x", Value::int(1))]);
        let s3 = State::from_pairs([("x", Value::int(2))]);

        // Same state, same tableau index -> equal
        let n1 = BehaviorGraphNode::from_state(&s1, 0);
        let n2 = BehaviorGraphNode::from_state(&s2, 0);
        assert_eq!(n1, n2);

        // Same state, different tableau index -> not equal
        let n3 = BehaviorGraphNode::from_state(&s1, 1);
        assert_ne!(n1, n3);

        // Different state, same tableau index -> not equal
        let n4 = BehaviorGraphNode::from_state(&s3, 0);
        assert_ne!(n1, n4);
    }

    #[test]
    fn test_behavior_graph_add_init() {
        let mut graph = BehaviorGraph::new();
        let s1 = State::from_pairs([("x", Value::int(0))]);

        // First add should succeed
        assert!(graph.add_init_node(&s1, 0));
        assert_eq!(graph.len(), 1);
        assert_eq!(graph.init_nodes().len(), 1);

        // Duplicate add should not increase size
        assert!(!graph.add_init_node(&s1, 0));
        assert_eq!(graph.len(), 1);

        // Same state, different tableau index should be added
        assert!(graph.add_init_node(&s1, 1));
        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn test_behavior_graph_add_successor() {
        let mut graph = BehaviorGraph::new();
        let s0 = State::from_pairs([("x", Value::int(0))]);
        let s1 = State::from_pairs([("x", Value::int(1))]);

        graph.add_init_node(&s0, 0);
        let init_node = BehaviorGraphNode::from_state(&s0, 0);

        // Add successor
        assert!(graph.add_successor(init_node, &s1, 0));
        assert_eq!(graph.len(), 2);

        // Check parent pointer
        let succ_node = BehaviorGraphNode::from_state(&s1, 0);
        let info = graph.get_node_info(&succ_node).unwrap();
        assert_eq!(info.parent, Some(init_node));
        assert_eq!(info.depth, 1);
    }

    #[test]
    fn test_behavior_graph_trace_reconstruction() {
        let mut graph = BehaviorGraph::new();
        let s0 = State::from_pairs([("x", Value::int(0))]);
        let s1 = State::from_pairs([("x", Value::int(1))]);
        let s2 = State::from_pairs([("x", Value::int(2))]);

        graph.add_init_node(&s0, 0);
        let n0 = BehaviorGraphNode::from_state(&s0, 0);

        graph.add_successor(n0, &s1, 0);
        let n1 = BehaviorGraphNode::from_state(&s1, 0);

        graph.add_successor(n1, &s2, 1);
        let n2 = BehaviorGraphNode::from_state(&s2, 1);

        let trace = graph.reconstruct_trace(n2);
        assert_eq!(trace.len(), 3);
        assert_eq!(trace[0].0, s0);
        assert_eq!(trace[0].1, 0);
        assert_eq!(trace[1].0, s1);
        assert_eq!(trace[1].1, 0);
        assert_eq!(trace[2].0, s2);
        assert_eq!(trace[2].1, 1);
    }

    #[test]
    fn test_behavior_graph_contains() {
        let mut graph = BehaviorGraph::new();
        let s1 = State::from_pairs([("x", Value::int(1))]);

        let node = BehaviorGraphNode::from_state(&s1, 0);
        assert!(!graph.contains(&node));

        graph.add_init_node(&s1, 0);
        assert!(graph.contains(&node));

        // Different tableau index should not be found
        let node2 = BehaviorGraphNode::from_state(&s1, 1);
        assert!(!graph.contains(&node2));
    }
}
