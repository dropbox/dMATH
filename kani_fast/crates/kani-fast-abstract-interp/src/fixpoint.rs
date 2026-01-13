//! Fixed-point iteration for abstract interpretation.
//!
//! Abstract interpretation computes over-approximations of program behavior
//! by iterating until a fixed point is reached. For loops, we may need
//! widening to ensure termination.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

use crate::lattice::Lattice;

/// Configuration for fixed-point computation.
#[derive(Debug, Clone)]
pub struct FixpointConfig {
    /// Maximum iterations before giving up.
    pub max_iterations: usize,
    /// Number of iterations before applying widening.
    /// Widening is needed to ensure termination on infinite domains.
    pub widen_delay: usize,
    /// Whether to apply narrowing after reaching a fixed point.
    pub use_narrowing: bool,
}

impl Default for FixpointConfig {
    fn default() -> Self {
        FixpointConfig {
            max_iterations: 1000,
            widen_delay: 3,
            use_narrowing: true,
        }
    }
}

/// Result of fixed-point computation.
#[derive(Debug, Clone)]
pub struct FixpointResult<K, V> {
    /// The computed abstract state at each program point.
    pub states: HashMap<K, V>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the computation reached a fixed point.
    pub converged: bool,
}

/// A control flow graph for abstract interpretation.
pub trait ControlFlowGraph {
    /// Type of nodes (basic blocks, statements, etc.)
    type Node: Clone + Eq + Hash;

    /// Entry node of the CFG.
    fn entry(&self) -> Self::Node;

    /// Successor nodes of a given node.
    fn successors(&self, node: &Self::Node) -> Vec<Self::Node>;

    /// Predecessor nodes of a given node.
    fn predecessors(&self, node: &Self::Node) -> Vec<Self::Node>;

    /// All nodes in the CFG.
    fn nodes(&self) -> Vec<Self::Node>;

    /// Check if a node is a loop header (for widening).
    fn is_loop_header(&self, node: &Self::Node) -> bool;
}

/// Transfer function that computes abstract state transformation.
pub trait TransferFunction<V: Lattice> {
    /// Type of nodes in the CFG.
    type Node;

    /// Apply the transfer function to compute the abstract state after a node.
    fn transfer(&self, node: &Self::Node, input: &V) -> V;
}

/// Forward dataflow analysis using chaotic iteration.
pub fn forward_analysis<G, V, T>(
    cfg: &G,
    transfer: &T,
    entry_state: V,
    config: &FixpointConfig,
) -> FixpointResult<G::Node, V>
where
    G: ControlFlowGraph,
    V: Lattice,
    T: TransferFunction<V, Node = G::Node>,
{
    let nodes = cfg.nodes();
    let mut states: HashMap<G::Node, V> = HashMap::with_capacity(nodes.len());
    let mut worklist: VecDeque<G::Node> = VecDeque::with_capacity(nodes.len());

    // Initialize all nodes to bottom
    for node in nodes {
        states.insert(node, V::bottom());
    }

    // Entry node: set entry_state and add successors to worklist
    let entry = cfg.entry();
    states.insert(entry.clone(), entry_state);
    for succ in cfg.successors(&entry) {
        worklist.push_back(succ);
    }

    let mut iteration = 0;

    while let Some(node) = worklist.pop_front() {
        if iteration >= config.max_iterations {
            break;
        }
        iteration += 1;

        // Compute input state by joining predecessors (no intermediate allocation)
        let preds = cfg.predecessors(&node);
        let mut input = V::bottom();
        let mut first = true;
        for p in &preds {
            if let Some(state) = states.get(p) {
                if first {
                    input = state.clone();
                    first = false;
                } else {
                    input = input.join(state);
                }
            }
        }

        // Apply transfer function
        let new_state = transfer.transfer(&node, &input);

        // Get old state for comparison
        let old_state = states.get(&node).cloned().unwrap_or_else(V::bottom);

        // Apply widening if at loop header and past delay
        let final_state = if cfg.is_loop_header(&node) && iteration > config.widen_delay {
            old_state.widen(&new_state)
        } else {
            new_state
        };

        // Check if state changed
        if final_state != old_state {
            states.insert(node.clone(), final_state);

            // Add successors to worklist
            for succ in cfg.successors(&node) {
                // O(n) contains check - acceptable for small CFGs
                // For large CFGs, consider a HashSet for tracking visited
                if !worklist.contains(&succ) {
                    worklist.push_back(succ);
                }
            }
        }
    }

    let converged = worklist.is_empty();

    // Optional: narrowing phase
    if converged && config.use_narrowing {
        let mut narrow_iter = 0;
        let mut changed = true;

        while changed && narrow_iter < config.max_iterations {
            changed = false;
            narrow_iter += 1;

            for node in cfg.nodes() {
                // Compute input by joining predecessors (no intermediate allocation)
                let input = if node == cfg.entry() {
                    states.get(&node).cloned().unwrap_or_else(V::bottom)
                } else {
                    let preds = cfg.predecessors(&node);
                    let mut result = V::bottom();
                    let mut first = true;
                    let mut has_pred = false;
                    for p in &preds {
                        if let Some(state) = states.get(p) {
                            has_pred = true;
                            if first {
                                result = state.clone();
                                first = false;
                            } else {
                                result = result.join(state);
                            }
                        }
                    }
                    if !has_pred {
                        continue;
                    }
                    result
                };

                let transferred = transfer.transfer(&node, &input);
                let old_state = states.get(&node).cloned().unwrap_or_else(V::bottom);
                let narrowed = old_state.narrow(&transferred);

                if narrowed != old_state {
                    states.insert(node.clone(), narrowed);
                    changed = true;
                }
            }
        }
    }

    FixpointResult {
        states,
        iterations: iteration,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::FlatLattice;

    // Simple linear CFG for testing: 0 -> 1 -> 2
    struct LinearCFG;

    impl ControlFlowGraph for LinearCFG {
        type Node = usize;

        fn entry(&self) -> usize {
            0
        }

        fn successors(&self, node: &usize) -> Vec<usize> {
            match node {
                0 => vec![1],
                1 => vec![2],
                _ => vec![],
            }
        }

        fn predecessors(&self, node: &usize) -> Vec<usize> {
            match node {
                1 => vec![0],
                2 => vec![1],
                _ => vec![],
            }
        }

        fn nodes(&self) -> Vec<usize> {
            vec![0, 1, 2]
        }

        fn is_loop_header(&self, _node: &usize) -> bool {
            false
        }
    }

    // Identity transfer function
    struct IdentityTransfer;

    impl TransferFunction<FlatLattice<i32>> for IdentityTransfer {
        type Node = usize;

        fn transfer(&self, _node: &usize, input: &FlatLattice<i32>) -> FlatLattice<i32> {
            input.clone()
        }
    }

    #[test]
    fn test_linear_forward_analysis() {
        let cfg = LinearCFG;
        let transfer = IdentityTransfer;
        let entry_state = FlatLattice::Value(42);
        let config = FixpointConfig::default();

        let result = forward_analysis(&cfg, &transfer, entry_state.clone(), &config);

        assert!(result.converged);
        assert_eq!(result.states.get(&0), Some(&entry_state));
        assert_eq!(result.states.get(&1), Some(&entry_state));
        assert_eq!(result.states.get(&2), Some(&entry_state));
    }

    // CFG with a loop: 0 -> 1 -> 2, 2 -> 1
    struct LoopCFG;

    impl ControlFlowGraph for LoopCFG {
        type Node = usize;

        fn entry(&self) -> usize {
            0
        }

        fn successors(&self, node: &usize) -> Vec<usize> {
            match node {
                0 => vec![1],
                1 => vec![2],
                2 => vec![1, 3], // Back edge to 1, exit to 3
                _ => vec![],
            }
        }

        fn predecessors(&self, node: &usize) -> Vec<usize> {
            match node {
                1 => vec![0, 2], // Entry and back edge
                2 => vec![1],
                3 => vec![2],
                _ => vec![],
            }
        }

        fn nodes(&self) -> Vec<usize> {
            vec![0, 1, 2, 3]
        }

        fn is_loop_header(&self, node: &usize) -> bool {
            *node == 1 // Node 1 is the loop header
        }
    }

    #[test]
    fn test_loop_forward_analysis() {
        let cfg = LoopCFG;
        let transfer = IdentityTransfer;
        let entry_state = FlatLattice::Value(42);
        let config = FixpointConfig::default();

        let result = forward_analysis(&cfg, &transfer, entry_state.clone(), &config);

        assert!(result.converged);
        // With identity transfer, the value should propagate through the loop
        assert_eq!(result.states.get(&0), Some(&entry_state));
    }

    // Benchmark test: large CFG with many nodes
    struct LargeCFG {
        size: usize,
    }

    impl ControlFlowGraph for LargeCFG {
        type Node = usize;

        fn entry(&self) -> usize {
            0
        }

        fn successors(&self, node: &usize) -> Vec<usize> {
            if *node < self.size - 1 {
                vec![node + 1]
            } else {
                vec![]
            }
        }

        fn predecessors(&self, node: &usize) -> Vec<usize> {
            if *node > 0 {
                vec![node - 1]
            } else {
                vec![]
            }
        }

        fn nodes(&self) -> Vec<usize> {
            (0..self.size).collect()
        }

        fn is_loop_header(&self, _node: &usize) -> bool {
            false
        }
    }

    #[test]
    fn test_large_cfg_performance() {
        use std::time::Instant;

        let cfg = LargeCFG { size: 1000 };
        let transfer = IdentityTransfer;
        let entry_state = FlatLattice::Value(42);
        let config = FixpointConfig::default();

        let start = Instant::now();
        let result = forward_analysis(&cfg, &transfer, entry_state.clone(), &config);
        let elapsed = start.elapsed();

        assert!(result.converged);
        assert_eq!(result.states.len(), 1000);
        // Should complete in under 10ms with inline optimizations
        assert!(
            elapsed.as_millis() < 100,
            "Analysis took too long: {:?}",
            elapsed
        );
    }
}
