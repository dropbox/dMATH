/*
 * Minimal Alloy model that should FAIL an assertion.
 * Models a simple graph where we incorrectly claim no cycles exist.
 */

sig Node {
    edges: set Node  -- Nodes can point to other nodes
}

-- Some edges must exist (not empty graph)
fact SomeEdges {
    some edges
}

-- WRONG assertion: claims no node can reach itself
-- This is FALSE - self-loops and cycles are possible
assert NoCycles {
    no n: Node | n in n.^edges
}

-- Check with small scope - will find counterexample
check NoCycles for 3

-- Run to find any instance
run {} for 3
