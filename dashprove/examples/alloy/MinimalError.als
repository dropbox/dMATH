/*
 * Minimal Alloy model with intentional syntax errors.
 */

sig Node {
    edges: set Node
}

-- Missing closing brace
fact BadFact {
    some edges

-- Unknown keyword
unknown_keyword Node

-- Type error
assert BadAssert {
    edges = 5  -- edges is a relation, not an int
}
