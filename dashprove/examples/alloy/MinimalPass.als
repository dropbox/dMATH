/*
 * Minimal Alloy model that should PASS all assertions.
 * Models a simple state machine with valid transitions.
 */

sig State {
    next: lone State  -- Each state has at most one next state
}

one sig Initial extends State {}
one sig Final extends State {}

-- Initial has a next state, Final does not
fact {
    Initial.next != none
    no Final.next
}

-- There's always a path from Initial to Final
fact reachability {
    Final in Initial.*next
}

-- Assertion: Initial is not equal to Final
assert InitialNotFinal {
    Initial != Final
}

-- Check the assertion with default scope
check InitialNotFinal for 5

-- Run command to find a valid instance
run {} for 5
