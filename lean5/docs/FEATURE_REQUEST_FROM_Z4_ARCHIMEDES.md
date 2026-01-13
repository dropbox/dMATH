# Feature Request: Z4/Archimedes Integration

**From**: Z4 SMT Solver & Archimedes AI Math Research Platform
**To**: Lean 5 Team
**Date**: 2025-12-31
**Priority**: High

---

## Context

**Z4** is a high-performance SMT solver in Rust (github.com/dropbox/z4).

**Archimedes** is an AI-assisted mathematical research platform built on Z4, Lean 5, and LLMs. Goal: vibecode new mathematics by letting AI explore conjectures while Lean provides ground truth verification.

We need tight integration between Z4 and Lean 5.

---

## Feature Requests

### P0: External Solver Interface

#### 1. Native Rust FFI for Tactics

**Request**: First-class support for calling Rust libraries from Lean tactics.

```lean
-- Ideal: declare external Rust function
@[extern "z4_solve_sat"]
opaque z4SolveSat : @& ByteArray → IO (Option (Array Int))

-- Use in tactic
syntax "z4_decide" : tactic

macro_rules
| `(tactic| z4_decide) => do
    let goal ← getGoal
    let formula := encodeAsFormula goal
    match ← z4SolveSat formula with
    | some model => exact (proveFromModel goal model)
    | none => exact (proveFromUnsat goal)
```

**Why**: JSON/pipe interface adds latency. For proof search making thousands of calls, we need native speed.

**Acceptable alternative**: Stable C FFI that Rust can target.

#### 2. DRAT/LRAT Proof Certificate Import

**Request**: Built-in verifier for SAT solver proof certificates.

```lean
-- Z4 returns UNSAT with DRAT proof
-- Lean should verify and accept as proof

structure DratProof where
  clauses : Array (Array Int)  -- Added/deleted clauses

def verifyDrat (formula : CNF) (proof : DratProof) : Bool :=
  -- Built-in efficient DRAT checker

-- Tactic using external proof
syntax "sat_decide_with_proof" : tactic
-- Calls Z4, gets DRAT, verifies, constructs Lean proof
```

**Why**: Z4 generates machine-checkable proofs. Lean should accept them without re-proving from scratch.

**Reference**: drat-trim, LRAT format (more efficient)

---

### P1: Proof Search API

#### 3. Programmatic Goal State Access

**Request**: Export current goal state for external tools.

```lean
structure GoalState where
  target : Expr                    -- What we're proving
  hypotheses : List (Name × Expr)  -- Available assumptions
  metavars : List MVarId           -- Subgoals

@[export lean_get_goal_state]
def getGoalState : TacticM GoalState

@[export lean_try_tactic]
def tryTactic (tactic : String) : TacticM (Except String GoalState)
```

**Why**: LLM-guided proof search needs to:
1. See current goal
2. Try tactics
3. Backtrack on failure
4. Explore multiple paths

#### 4. Tactic Suggestion Hook

**Request**: Register external tactic suggesters.

```lean
-- Register suggester
register_tactic_suggester "z4_suggester" (fun goal => do
  -- Call Z4/LLM
  return ["simp", "ring", "z4_decide"]
)

-- In proof, get suggestions
example : P := by
  suggest  -- Shows: simp, ring, z4_decide
```

**Why**: AI proposes, human approves, Lean verifies.

---

### P2: Theory-Specific Bridges

#### 5. Propositional Logic Encoding

**Request**: Standard encoding from Prop goals to CNF.

```lean
-- Encode decidable Prop goal as CNF
def goalToCnf (goal : Expr) : Option CNF

-- Reconstruct proof from SAT model
def modelToProof (goal : Expr) (model : Array Int) : Expr

-- The tactic
syntax "prop_decide" : tactic
-- Works for: ∀ p q : Prop, (p → q) → (¬q → ¬p)
```

#### 6. Arithmetic Theory Bridges

```lean
-- Linear Integer Arithmetic (calls Z4 LIA solver)
syntax "lia_decide" : tactic

-- Linear Real Arithmetic (calls Z4 LRA solver)
syntax "lra_decide" : tactic

-- Bitvectors (calls Z4 BV solver)
syntax "bv_decide" : tactic
```

**Why**: Many proof obligations are decidable by theory solvers.

---

### P3: Infrastructure

#### 7. Parallel Tactic Execution

**Request**: Try multiple tactics in parallel.

```lean
syntax "race" "[" tactic,* "]" : tactic

example : P := by
  race [simp, ring, omega, z4_decide, nlinarith]
  -- Takes first success
```

**Why**: AI generates many candidates. Parallel exploration is faster.

#### 8. Proof State Checkpoint/Resume

**Request**: Save and restore proof state.

```lean
def checkpoint (state : ProofState) (path : String) : IO Unit
def resume (path : String) : IO ProofState
```

**Why**: Long-running AI proof searches need to checkpoint progress.

#### 9. Proof Trace Export

**Request**: Export successful proof traces for ML training.

```lean
set_option trace.proof.export true in
theorem example : ... := by
  intro h
  simp [h]
  -- Exports JSON: {"goal": "...", "tactics": ["intro h", "simp [h]"]}
```

**Why**: Train AI on successful proofs.

---

### P4: Mathlib/Stdlib Extensions

#### 10. Complexity Theory Foundations

**Request**: Basic complexity theory definitions.

```lean
namespace Complexity

-- Turing machines (or equivalent)
structure TuringMachine where ...

-- Time complexity
def timeComplexity (tm : TuringMachine) (input : List Bool) : ℕ

-- Complexity classes
def P : Set (List Bool → Option Bool) := ...
def NP : Set (List Bool → Option Bool) := ...

-- Circuit complexity
structure Circuit where ...
def circuitSize : Circuit → ℕ
def circuitDepth : Circuit → ℕ

end Complexity
```

**Why**: Archimedes goal is to contribute to complexity theory. Need formal definitions.

---

## Integration Architecture

```
┌─────────────────────────────────────────────────┐
│                    Lean 5                        │
│                                                  │
│  ┌────────────┐    ┌────────────┐               │
│  │   Tactic   │───→│  Goal      │               │
│  │   Engine   │    │  State API │───────┐       │
│  └────────────┘    └────────────┘       │       │
│        │                                │       │
│        ▼                                ▼       │
│  ┌────────────┐              ┌────────────────┐ │
│  │   DRAT     │              │  External FFI  │ │
│  │  Verifier  │              │  (Rust/C)      │ │
│  └────────────┘              └───────┬────────┘ │
│        ▲                             │          │
└────────┼─────────────────────────────┼──────────┘
         │                             │
         │      ┌──────────────────────┘
         │      │
         │      ▼
    ┌────┴──────────┐
    │      Z4       │
    │  SAT/SMT/QBF  │
    │  Rust Solver  │
    └───────────────┘
```

---

## Contact

**Repo**: github.com/dropbox/z4
**Docs**: z4/docs/ARCHIMEDES_ROADMAP.md

Happy to coordinate on API design. Let's make AI-assisted math real.
