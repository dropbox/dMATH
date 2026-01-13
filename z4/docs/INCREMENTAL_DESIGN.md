# Incremental Solving Design Document

**Date**: 2025-12-31
**Author**: Z4 Team (Worker #155)
**Status**: Design Phase
**Priority**: CRITICAL (tRust, Kani Fast integration)

---

## Executive Summary

This document specifies the design for incremental solving in Z4, enabling efficient push/pop context management with learned clause preservation. This is the #1 feature request from tRust AI for verification condition batching.

**Goal**: Enable 10x speedup on incremental verification workflows by preserving solver state across queries.

---

## Requirements

### Functional Requirements

1. **Push/Pop Context**: Save and restore solver state
2. **Assertion Scoping**: Assertions added after `push()` are removed by `pop()`
3. **Learned Clause Preservation**: Retain valid learned clauses across pop
4. **Theory State Management**: Theory solvers maintain consistent state
5. **SMT-LIB Compliance**: Support `(push)` and `(pop)` commands

### Performance Requirements

| Metric | Target |
|--------|--------|
| Push overhead | <0.1ms |
| Pop overhead | <0.1ms |
| Learned clause retention | >80% on pop |
| Memory overhead | <10% vs non-incremental |

### Use Cases

**tRust Counterexample Minimization**:
```smt2
(push)
  (assert <VC-variant-1>)
  (check-sat)
(pop)
(push)
  (assert <VC-variant-2>)
  (check-sat)
(pop)
```

**Kani Fast Incremental Verification**:
```smt2
; Base constraints (persist)
(assert base-constraints)
(push)
  ; Function-specific constraints
  (assert func-specific)
  (check-sat)
(pop)
; Base constraints still active, learned clauses retained
```

---

## Architecture

### Current State

```
┌─────────────────────────────────────────────────────────────────┐
│                        Executor                                 │
│  - No push/pop support                                         │
│  - Rebuilds solver each check-sat                              │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                        DpllT<T>                                 │
│  - No push/pop support                                         │
│  - Holds SAT solver + theory solver                            │
└─────────────────────────────────────────────────────────────────┘
          │                                    │
┌─────────▼─────────┐              ┌───────────▼───────────────────┐
│    SatSolver      │              │       TheorySolver (trait)    │
│  - Has push/pop   │              │  - Has push/pop methods       │
│  - Selector-based │              │  - Not fully implemented      │
└───────────────────┘              └───────────────────────────────┘
```

### Target State

```
┌─────────────────────────────────────────────────────────────────┐
│                        Executor                                 │
│  + push() / pop() API                                          │
│  + Assertion scope tracking                                    │
│  + Tseitin state preservation                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                        DpllT<T>                                 │
│  + push() / pop() orchestration                                │
│  + Coordinates SAT + theory push/pop                           │
│  + Variable mapping preservation                               │
└─────────────────────────────────────────────────────────────────┘
          │                                    │
┌─────────▼─────────┐              ┌───────────▼───────────────────┐
│    SatSolver      │              │       TheorySolver            │
│  - push/pop       │              │  + Fully implemented push/pop │
│  - Clause tagging │              │  + State checkpoint/restore   │
│  + Clause retention│             └───────────────────────────────┘
└───────────────────┘
```

---

## Detailed Design

### 1. SAT Solver Incremental Enhancement

**Current Implementation** (`crates/z4-sat/src/solver.rs:846-864`):
```rust
pub fn push(&mut self) {
    let selector = self.new_var_internal();
    self.scope_selectors.push(selector);
}

pub fn pop(&mut self) -> bool {
    let selector = match self.scope_selectors.pop() {
        Some(v) => v,
        None => return false,
    };
    // Disable clauses guarded by this selector
    let _ = self.add_clause_unscoped(vec![Literal::positive(selector)], false);
    true
}
```

**Issue**: All learned clauses that depend on popped assertions become invalid but aren't tracked.

**Enhancement**: Add clause dependency tracking:

```rust
/// Track which scope level each clause was added at
struct ClauseMetadata {
    /// Scope level when clause was added
    scope_level: u32,
    /// Whether clause depends on assertions from this or higher levels
    depends_on_scope: u32,
}

pub fn pop(&mut self) -> bool {
    let selector = match self.scope_selectors.pop() {
        Some(v) => v,
        None => return false,
    };

    // Invalidate learned clauses that depend on popped assertions
    self.invalidate_dependent_clauses(current_scope);

    // Disable clauses guarded by this selector
    let _ = self.add_clause_unscoped(vec![Literal::positive(selector)], false);
    true
}
```

**Learned Clause Dependency Tracking**:

During conflict analysis, track which scope levels contributed:
```rust
fn analyze_conflict(&mut self, ...) -> (Vec<Literal>, usize, u32) {
    // Track max scope level of literals in conflict
    let mut max_scope = 0;
    for lit in &conflict_clause {
        let var = lit.variable();
        if let Some(reason) = self.reason[var.index()] {
            max_scope = max_scope.max(self.clause_scope[reason.0 as usize]);
        }
    }
    // Return scope dependency with learned clause
    (learned_clause, backtrack_level, max_scope)
}
```

### 2. DpllT Push/Pop Orchestration

Add push/pop to `DpllT<T>`:

```rust
impl<T: TheorySolver> DpllT<T> {
    /// Push a new assertion scope
    pub fn push(&mut self) {
        self.sat.push();
        self.theory.push();
    }

    /// Pop the most recent assertion scope
    pub fn pop(&mut self) -> bool {
        if !self.sat.pop() {
            return false;
        }
        self.theory.pop();
        true
    }
}
```

### 3. Theory Solver Push/Pop Implementation

**TheorySolver trait** (`crates/z4-core/src/theory.rs:81-85`):
```rust
trait TheorySolver {
    /// Push a new scope
    fn push(&mut self);

    /// Pop to previous scope
    fn pop(&mut self);
}
```

**EUF Solver Implementation**:
```rust
impl TheorySolver for EufSolver {
    fn push(&mut self) {
        self.scope_stack.push(ScopeFrame {
            union_find_snapshot: self.union_find.snapshot(),
            pending_len: self.pending.len(),
            explanation_len: self.explanations.len(),
        });
    }

    fn pop(&mut self) {
        if let Some(frame) = self.scope_stack.pop() {
            self.union_find.restore(frame.union_find_snapshot);
            self.pending.truncate(frame.pending_len);
            self.explanations.truncate(frame.explanation_len);
        }
    }
}
```

**LIA/LRA Solver Implementation**:
```rust
impl TheorySolver for LiaSolver {
    fn push(&mut self) {
        self.scope_stack.push(ScopeFrame {
            tableau_snapshot: self.tableau.snapshot(),
            bounds_len: self.bounds.len(),
            assertions_len: self.assertions.len(),
        });
    }

    fn pop(&mut self) {
        if let Some(frame) = self.scope_stack.pop() {
            self.tableau.restore(frame.tableau_snapshot);
            self.bounds.truncate(frame.bounds_len);
            self.assertions.truncate(frame.assertions_len);
        }
    }
}
```

### 4. Executor Push/Pop

Add to `Executor`:

```rust
impl Executor {
    /// Push a new assertion scope
    pub fn push(&mut self) {
        // Save Tseitin state
        let tseitin_state = self.tseitin.save_state();
        self.tseitin_stack.push(tseitin_state);

        // Save assertion indices
        self.assertion_stack.push(self.assertions.len());

        // Push to solver if it exists
        if let Some(solver) = &mut self.solver {
            solver.push();
        }
    }

    /// Pop the most recent assertion scope
    pub fn pop(&mut self) -> bool {
        // Restore Tseitin state
        let state = match self.tseitin_stack.pop() {
            Some(s) => s,
            None => return false,
        };
        self.tseitin.restore_state(&state);

        // Restore assertions
        if let Some(len) = self.assertion_stack.pop() {
            self.assertions.truncate(len);
        }

        // Pop from solver if it exists
        if let Some(solver) = &mut self.solver {
            solver.pop();
        }

        true
    }
}
```

### 5. SMT-LIB Command Handling

Update command execution in `executor.rs`:

```rust
fn execute_command(&mut self, cmd: &Command) -> Result<CommandResult> {
    match cmd {
        Command::Push(n) => {
            for _ in 0..*n {
                self.push();
            }
            Ok(CommandResult::Success)
        }
        Command::Pop(n) => {
            for _ in 0..*n {
                if !self.pop() {
                    return Err(ExecutorError::InvalidPop);
                }
            }
            Ok(CommandResult::Success)
        }
        // ... other commands
    }
}
```

---

## Data Structures

### Scope Frame

```rust
/// Checkpoint of solver state at a push point
struct ScopeFrame {
    /// SAT solver scope selector variable
    sat_selector: Variable,

    /// Number of assertions at this level
    assertion_count: usize,

    /// Tseitin transformation state
    tseitin_state: TseitinState,

    /// Theory solver checkpoint (opaque to DpllT)
    theory_checkpoint: Box<dyn Any>,
}
```

### Clause Dependency Graph

```rust
/// Tracks which clauses depend on which scope levels
struct ClauseDependency {
    /// For each clause, the minimum scope level it depends on
    /// Clauses with level > current scope are invalidated on pop
    clause_min_scope: Vec<u32>,

    /// Current scope level
    current_scope: u32,
}
```

---

## Correctness Invariants

### INV-1: Scope Consistency
After any sequence of push/pop, the solver is in a consistent state:
- All SAT clauses are either valid or disabled
- All theory assertions are consistent with scope
- Variable mappings are preserved

### INV-2: Learned Clause Validity
A learned clause is retained after pop if and only if:
- All literals in the clause have variables from scope <= current
- The clause was derived without using any popped assertions

### INV-3: Model Correctness
If `check-sat` returns `sat` after any push/pop sequence:
- The model satisfies all assertions in scope
- The model satisfies all base (level 0) assertions

### INV-4: Soundness
If `check-sat` returns `unsat`:
- The conjunction of all in-scope assertions is unsatisfiable
- The proof (if generated) is valid

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_basic_push_pop() {
    let mut exec = Executor::new();
    exec.assert("a");           // Level 0
    exec.push();
    exec.assert("b");           // Level 1
    assert_eq!(exec.check_sat(), Sat);
    exec.pop();                 // Remove b
    exec.push();
    exec.assert("(not b)");     // Level 1 (different)
    assert_eq!(exec.check_sat(), Sat);  // Should still be sat
}

#[test]
fn test_learned_clause_retention() {
    let mut exec = Executor::new();
    // Add clauses that will generate learned clauses
    exec.assert("(or a b)");
    exec.assert("(or (not a) c)");
    exec.push();
    exec.assert("(not c)");
    exec.assert("(not b)");
    assert_eq!(exec.check_sat(), Unsat);  // Learns clauses
    exec.pop();
    // After pop, learned clauses from level 0 should remain
    // Next check should be faster
}
```

### Differential Tests

```rust
#[test]
fn differential_incremental_vs_z3() {
    // Compare Z4 incremental behavior against Z3
    for benchmark in incremental_benchmarks() {
        let z3_results = run_z3_incremental(&benchmark);
        let z4_results = run_z4_incremental(&benchmark);
        assert_eq!(z3_results, z4_results);
    }
}
```

### Property Tests

```rust
proptest! {
    #[test]
    fn incremental_soundness(ops in vec(solver_op(), 1..100)) {
        let mut solver = Executor::new();
        let mut oracle = NonIncrementalOracle::new();

        for op in ops {
            match op {
                Op::Push => { solver.push(); oracle.push(); }
                Op::Pop => { solver.pop(); oracle.pop(); }
                Op::Assert(a) => { solver.assert(&a); oracle.assert(&a); }
                Op::CheckSat => {
                    let s = solver.check_sat();
                    let o = oracle.check_sat();
                    // Sat/Unsat must match, Unknown is ok
                    assert!(compatible(s, o));
                }
            }
        }
    }
}
```

---

## Implementation Plan

### Phase 1: SAT Layer (Week 1)
1. Add clause dependency tracking to SatSolver
2. Implement learned clause invalidation on pop
3. Add metrics for clause retention rate
4. Unit tests for SAT incremental

### Phase 2: Theory Layer (Week 2)
1. Implement EufSolver push/pop with union-find snapshots
2. Implement LiaSolver push/pop with tableau snapshots
3. Implement ArraySolver push/pop
4. Unit tests for each theory

### Phase 3: DpllT Integration (Week 3)
1. Add push/pop orchestration to DpllT
2. Variable mapping preservation
3. Integration tests

### Phase 4: Executor Integration (Week 4)
1. Add push/pop to Executor
2. SMT-LIB command handling
3. Tseitin state management
4. End-to-end tests

### Phase 5: Validation (Week 5)
1. Differential testing vs Z3 on incremental benchmarks
2. Performance benchmarking
3. tRust integration testing

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Learned clause tracking overhead | Performance | Use lightweight tagging, benchmark early |
| Theory snapshot memory | Memory | Implement copy-on-write for large structures |
| Variable mapping corruption | Correctness | Strong invariants, defensive checks |
| Z3 compatibility | Integration | Extensive differential testing |

---

## References

- MiniSat incremental interface (Eén & Sörensson 2003)
- CaDiCaL assumptions API (Biere 2021)
- Z3 push/pop semantics (de Moura & Bjorner 2008)
- SMT-LIB 2.6 standard (push/pop commands)

---

## Appendix: SMT-LIB Incremental Commands

```smt2
; Push n scope levels (default n=1)
(push <numeral>?)

; Pop n scope levels (default n=1)
(pop <numeral>?)

; Reset to initial state (equivalent to popping all levels)
(reset)

; Reset assertions only (keep declarations)
(reset-assertions)
```

---

**End of Design Document**
