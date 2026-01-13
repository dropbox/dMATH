# Phase 1 Work Order: CDCL SAT Solver

**Status**: Ready for WORKER
**Target**: `z4-sat` crate
**Objective**: Implement a **verified** CDCL SAT solver

> **IMPORTANT**: See `docs/PHASE1_EXECUTION_ROADMAP.md` for the step-by-step execution plan.
> That document specifies the exact order of implementation with verification-first approach.

---

## Current State

The `z4-sat` crate has skeleton code:
- `literal.rs`: Complete - Variable/Literal encoding
- `clause.rs`: Complete - Basic Clause struct
- `watched.rs`: Partial - WatchedLists struct, no propagation
- `vsids.rs`: Partial - Activity bump/decay, no variable selection
- `conflict.rs`: Skeleton - No 1UIP algorithm
- `solver.rs`: Skeleton - `solve()` returns `Unknown`

**Result**: Cannot solve any SAT problem.

---

## Requirements

### Functional
1. Parse DIMACS CNF format (or accept clauses programmatically)
2. Return SAT with satisfying assignment, or UNSAT
3. Pass basic SAT tests (small CNF files)
4. Eventually: competitive with MiniSat on SAT Competition benchmarks

### Non-Functional
1. No panics on valid input
2. Memory-safe (no unsafe blocks initially)
3. Tests for every component

---

## Implementation Tasks

### Task 1: Unit Propagation

**File**: `solver.rs`
**Reference**: MiniSat paper Section 3, `papers/minisat-sat04.pdf`

Implement Boolean Constraint Propagation (BCP) with 2-watched literals.

```rust
/// Unit propagation using 2-watched literals
/// Returns: None if no conflict, Some(clause_ref) if conflict
fn propagate(&mut self) -> Option<ClauseRef> {
    while self.propagation_queue.is_not_empty() {
        let p = self.propagation_queue.pop(); // Literal that became true
        // All clauses watching ¬p must update watches
        for each watcher in watches[¬p]:
            if !propagate_clause(watcher.clause, ¬p):
                // Conflict: clause is falsified
                clear_propagation_queue();
                return Some(watcher.clause);
    }
    None
}
```

**Key invariants**:
- Two watched literals per clause (length >= 2)
- At least one watched literal is not false under current assignment
- Watches only update when watched literal becomes false

**Test**: `test_unit_propagation_simple` - single unit clause forces assignment

### Task 2: Watched Literal Updates

**File**: `watched.rs`
**Reference**: MiniSat paper Section 3.1

When literal `p` becomes false, update watches for clauses watching `p`:

```rust
fn propagate_clause(&mut self, clause_ref: ClauseRef, false_lit: Literal) -> bool {
    let clause = &mut self.clauses[clause_ref];

    // Ensure false_lit is clause[1] (swap if needed)
    if clause[0] == false_lit {
        swap(clause[0], clause[1]);
    }

    // If clause[0] is true, clause is satisfied - keep watching
    if self.value(clause[0]) == Some(true) {
        return true; // No conflict
    }

    // Look for new literal to watch
    for i in 2..clause.len() {
        if self.value(clause[i]) != Some(false) {
            swap(clause[1], clause[i]);
            // Move watch from false_lit to clause[1]
            add_watch(clause[1].negated(), clause_ref, clause[0]);
            return true;
        }
    }

    // No replacement found
    if self.value(clause[0]) == Some(false) {
        return false; // Conflict!
    } else {
        // clause[0] is unassigned - unit propagation!
        self.enqueue(clause[0], clause_ref);
        return true;
    }
}
```

**Test**: `test_watched_literal_update` - watch moves correctly

### Task 3: Trail Management

**File**: `solver.rs`

Maintain the assignment trail with decision levels and reasons:

```rust
struct TrailEntry {
    literal: Literal,
    reason: Option<ClauseRef>,  // None if decision
}

/// Assign literal with reason
fn enqueue(&mut self, lit: Literal, reason: Option<ClauseRef>) {
    let var = lit.variable();
    self.assignment[var.index()] = Some(lit.is_positive());
    self.level[var.index()] = self.decision_level;
    self.reason[var.index()] = reason;
    self.trail.push(lit);
    self.propagation_queue.push(lit);
}

/// Make a decision
fn decide(&mut self, lit: Literal) {
    self.decision_level += 1;
    self.trail_lim.push(self.trail.len());
    self.enqueue(lit, None);
}
```

### Task 4: VSIDS Variable Selection

**File**: `vsids.rs`
**Reference**: Chaff paper, `papers/chaff-dac01.pdf`

Add method to pick highest-activity unassigned variable:

```rust
/// Select next variable to branch on
pub fn pick_branching_variable(&self, assignment: &[Option<bool>]) -> Option<Variable> {
    let mut best_var = None;
    let mut best_activity = -1.0;

    for (i, &assigned) in assignment.iter().enumerate() {
        if assigned.is_none() {
            let activity = self.activities[i];
            if activity > best_activity {
                best_activity = activity;
                best_var = Some(Variable(i as u32));
            }
        }
    }
    best_var
}
```

**Optimization (later)**: Use a heap for O(log n) selection instead of O(n) scan.

### Task 5: 1UIP Conflict Analysis

**File**: `conflict.rs`
**Reference**: MiniSat paper Section 4, DPLL(T) paper Section 3

Implement First-UIP learning:

```rust
pub fn analyze(
    &mut self,
    conflict_clause: ClauseRef,
    clauses: &[Clause],
    trail: &[Literal],
    level: &[u32],
    reason: &[Option<ClauseRef>],
    current_level: u32,
) -> ConflictResult {
    self.clear();
    let mut counter = 0;
    let mut p: Option<Literal> = None;
    let mut index = trail.len();

    // Start with conflict clause
    let mut clause = &clauses[conflict_clause.0 as usize];

    loop {
        // Add literals from clause to learned clause
        for &lit in &clause.literals {
            if p.is_some() && lit == p.unwrap() {
                continue;
            }
            let var = lit.variable();
            if !self.is_seen(var.index()) {
                self.mark_seen(var.index());
                if level[var.index()] == current_level {
                    counter += 1;
                } else if level[var.index()] > 0 {
                    self.learned.push(lit.negated());
                }
            }
        }

        // Select next literal to resolve
        loop {
            index -= 1;
            p = Some(trail[index]);
            if self.is_seen(p.unwrap().variable().index()) {
                break;
            }
        }

        counter -= 1;
        if counter == 0 {
            break; // Found UIP
        }

        // Get reason clause for p
        let reason_ref = reason[p.unwrap().variable().index()].unwrap();
        clause = &clauses[reason_ref.0 as usize];
    }

    // p is the UIP - add its negation as first literal
    self.learned.insert(0, p.unwrap().negated());

    // Compute backtrack level (second highest level in learned clause)
    let backtrack_level = self.compute_backtrack_level(level);

    // Compute LBD
    let lbd = self.compute_lbd(level);

    ConflictResult {
        learned_clause: self.learned.clone(),
        backtrack_level,
        lbd,
    }
}
```

**Test**: `test_conflict_analysis_simple` - learns correct clause

### Task 6: Backtracking

**File**: `solver.rs`

```rust
fn backtrack(&mut self, target_level: u32) {
    while self.decision_level > target_level {
        while self.trail.len() > self.trail_lim[self.decision_level as usize - 1] {
            let lit = self.trail.pop().unwrap();
            let var = lit.variable();
            self.assignment[var.index()] = None;
            self.reason[var.index()] = None;
            // Note: don't reset level - not needed
        }
        self.trail_lim.pop();
        self.decision_level -= 1;
    }
}
```

### Task 7: Main CDCL Loop

**File**: `solver.rs`

```rust
pub fn solve(&mut self) -> SolveResult {
    // Initialize watches for all clauses
    self.initialize_watches();

    loop {
        // Propagate
        if let Some(conflict) = self.propagate() {
            if self.decision_level == 0 {
                return SolveResult::Unsat;
            }

            // Conflict analysis
            let result = self.conflict.analyze(
                conflict,
                &self.clauses,
                &self.trail,
                &self.level,
                &self.reason,
                self.decision_level,
            );

            // Learn clause
            let learned_ref = self.add_learned_clause(result.learned_clause, result.lbd);

            // Backtrack
            self.backtrack(result.backtrack_level);

            // Assert the learned clause's UIP literal
            let uip = self.clauses[learned_ref.0 as usize].literals[0];
            self.enqueue(uip, Some(learned_ref));

            // Bump VSIDS
            self.vsids.decay();
        } else {
            // No conflict - decide or return SAT
            if let Some(var) = self.vsids.pick_branching_variable(&self.assignment) {
                // Use phase saving (initially: positive)
                let lit = Literal::positive(var);
                self.decide(lit);
            } else {
                // All variables assigned
                return SolveResult::Sat(self.get_model());
            }
        }
    }
}
```

### Task 8: DIMACS Parser (Optional for Phase 1)

**File**: New file `dimacs.rs` or in test utilities

```rust
pub fn parse_dimacs(input: &str) -> (usize, Vec<Vec<Literal>>) {
    let mut num_vars = 0;
    let mut clauses = Vec::new();

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') {
            continue;
        }
        if line.starts_with('p') {
            // p cnf <vars> <clauses>
            let parts: Vec<&str> = line.split_whitespace().collect();
            num_vars = parts[2].parse().unwrap();
            continue;
        }
        // Clause line: "1 -2 3 0"
        let clause: Vec<Literal> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<i32>().ok())
            .take_while(|&n| n != 0)
            .map(|n| {
                if n > 0 {
                    Literal::positive(Variable((n - 1) as u32))
                } else {
                    Literal::negative(Variable((-n - 1) as u32))
                }
            })
            .collect();
        if !clause.is_empty() {
            clauses.push(clause);
        }
    }
    (num_vars, clauses)
}
```

---

## Testing Checklist

### Unit Tests (z4-sat/src/*.rs)

- [ ] `literal.rs`: positive/negative encoding, variable extraction, negation
- [ ] `clause.rs`: create, len, is_empty, is_unit
- [ ] `watched.rs`: add_watch, get_watches
- [ ] `vsids.rs`: bump, decay, pick_branching_variable
- [ ] `conflict.rs`: analyze returns correct learned clause
- [ ] `solver.rs`: propagate, backtrack, solve

### Integration Tests (z4-sat/tests/)

- [ ] `test_trivial_sat.rs`: (a ∨ b) is SAT
- [ ] `test_trivial_unsat.rs`: (a) ∧ (¬a) is UNSAT
- [ ] `test_small_cnf.rs`: 3-SAT instances with known answers
- [ ] `test_dimacs.rs`: Parse and solve DIMACS files

### Benchmark Tests (later)

- [ ] Compare against MiniSat on SAT Competition benchmarks
- [ ] Profile for performance bottlenecks

---

## Success Criteria

Phase 1 is complete when:

1. **Correctness**: `cargo test -p z4-sat` passes all tests
2. **Functionality**: Can solve basic SAT problems (SAT and UNSAT)
3. **Model Verification**: SAT results include valid satisfying assignment
4. **No panics**: Handles edge cases gracefully

---

## References

| Paper | Purpose | Location |
|-------|---------|----------|
| MiniSat SAT 2004 | CDCL architecture, 2-watched literals | `papers/minisat-sat04.pdf` |
| Chaff DAC 2001 | VSIDS heuristic | `papers/chaff-dac01.pdf` |
| DPLL(T) JACM 2006 | 1UIP conflict analysis | `papers/dpll-t-jacm2006.pdf` |
| CaDiCaL SAT 2018 | Modern optimizations (Phase 2) | `papers/cadical-sat2018.pdf` |

---

## Notes for WORKER

1. **Start simple**: Get basic CDCL working before optimizations
2. **Test incrementally**: Write tests as you implement each component
3. **Read the MiniSat paper**: It's 10 pages and explains everything
4. **Don't optimize early**: Correctness first, performance second
5. **Commit frequently**: Each task above can be a separate commit

When done, run:
```bash
cargo test -p z4-sat
cargo clippy -p z4-sat -- -W clippy::all
```

Both should pass with no warnings.

---

## Phase 1 Extensions (After Core Works)

1. **Phase saving**: Remember and prefer last polarity
2. **LBD tracking**: Compute and store Literal Block Distance
3. **Restarts**: Implement Luby sequence restarts
4. **Clause deletion**: Remove low-quality learned clauses
5. **Heap-based VSIDS**: O(log n) variable selection
6. **Blocker literals**: Store extra literal in watch for faster filtering

These are NOT required for Phase 1 completion but will improve performance.
