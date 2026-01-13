# Lean 5 Feature Requests for Z4 Integration

**Context**: Z4 is a Rust SMT solver. We want tight integration with Lean 5 for verified mathematical research (project Archimedes).

---

## Priority 1: External Solver Interface

### 1.1 Native FFI for Rust Solvers

**Request**: First-class FFI support for calling Rust libraries from Lean tactics.

```lean
-- Ideal API
@[extern "z4_sat_solve"]
opaque z4Solve : Array Clause → IO SolveResult

-- Tactic that calls Z4
syntax "z4_decide" : tactic

macro_rules
| `(tactic| z4_decide) => do
    let goal ← getGoal
    let formula := encodeAsFormula goal
    let result := z4Solve formula
    match result with
    | .sat model => exact (proveFromModel goal model)
    | .unsat proof => exact (proveFromDrat goal proof)
```

**Why**: Current `native_decide` is limited. We want to call Z4 directly with full control.

**Alternative**: JSON/pipe interface works but adds latency.

---

### 1.2 Proof Certificate Import

**Request**: Import DRAT/LRAT proofs as Lean proofs.

```lean
-- Z4 returns UNSAT with DRAT proof
-- Lean should be able to verify and convert to native proof

def verifyDratProof (formula : CNF) (drat : DratProof) : Bool :=
  -- Built-in DRAT checker

theorem fromDrat (formula : CNF) (proof : DratProof) (h : verifyDratProof formula proof = true) :
    Unsatisfiable formula := by
  native_drat_verify  -- New tactic
```

**Why**: Z4 generates DRAT proofs for UNSAT. These should become Lean proofs without re-proving.

**Current gap**: No standard way to import external proof certificates.

---

## Priority 2: Incremental/Interactive Mode

### 2.1 REPL with Goal State Access

**Request**: Programmatic access to current goal state for external tools.

```lean
-- External tool can query:
-- 1. Current goal type
-- 2. Local context (hypotheses)
-- 3. Available lemmas
-- 4. Try tactics and get resulting goals

structure GoalState where
  goal : Expr
  hypotheses : List (Name × Expr)
  availableLemmas : List Name

-- External API
@[export]
def getCurrentGoalState : TacticM GoalState

@[export]
def tryTactic (tactic : String) : TacticM (Option (List GoalState))
```

**Why**: LLM-guided proof search needs to query goal state, try tactics, backtrack.

---

### 2.2 Tactic Suggestion Hook

**Request**: Hook for external tactic suggesters.

```lean
-- When user types `?` or `suggest`, call external suggester
register_tactic_suggester "z4_suggester" (fun goal => do
  -- Call Z4/LLM to suggest tactics
  return ["simp", "ring", "z4_decide"]
)
```

**Why**: AI can suggest proof steps, user approves, Lean verifies.

---

## Priority 3: Theory-Specific Bridges

### 3.1 SAT/Propositional

```lean
-- Encode Prop goal as CNF, solve with Z4, reconstruct proof
theorem prop_example : ∀ p q : Prop, (p ∧ q) → (q ∧ p) := by
  z4_decide  -- Should just work
```

**Current**: `decide` works for finite types. Need extension for general Prop with specific structure.

---

### 3.2 Linear Arithmetic

```lean
-- Encode linear arithmetic as SMT, solve with Z4
theorem arith_example : ∀ x y : Int, x > 0 → y > 0 → x + y > 0 := by
  z4_lia  -- Linear integer arithmetic
```

**Request**: Standard encoding from Lean Int/Nat goals to SMT-LIB LIA.

---

### 3.3 Bitvectors

```lean
-- Encode bitvector goals as QF_BV
theorem bv_example : ∀ x : BitVec 32, x &&& 0 = 0 := by
  z4_bv  -- Bitvector theory
```

**Why**: Critical for Kani-style verification of Rust code.

---

## Priority 4: Proof Search Infrastructure

### 4.1 Checkpoint/Resume

**Request**: Save and restore proof state for long-running searches.

```lean
-- Save proof state to file
def checkpoint (state : ProofState) : IO Unit

-- Resume from checkpoint
def resume (file : String) : IO ProofState
```

**Why**: AI proof search may run for hours. Need to checkpoint progress.

---

### 4.2 Parallel Tactic Execution

**Request**: Try multiple tactics in parallel, take first success.

```lean
-- Try tactics in parallel
syntax "race" "[" tactic,* "]" : tactic

example : P := by
  race [simp, ring, z4_decide, nlinarith]
```

**Why**: AI generates many candidate tactics. Parallel exploration is faster.

---

### 4.3 Proof Recording

**Request**: Record successful proof paths for learning.

```lean
-- After proof succeeds, export trace
set_option trace.proof.record true

-- Exports JSON: {"goal": "...", "tactics": ["intro", "simp", "exact h"], "time": 0.3}
```

**Why**: Train AI on successful proofs.

---

## Priority 5: Mathematical Library Integration

### 5.1 Complexity Theory Foundations

**Request**: Mathlib/stdlib definitions for complexity classes.

```lean
-- Definitions we need
def TuringMachine : Type
def Computes (tm : TuringMachine) (f : ℕ → ℕ) : Prop
def TimeComplexity (tm : TuringMachine) : ℕ → ℕ

def P : Set (ℕ → Bool) := { f | ∃ tm, Computes tm f ∧ ∀ n, TimeComplexity tm n ≤ n^c }
def NP : Set (ℕ → Bool) := { f | ∃ tm, ... }  -- Nondeterministic

-- The question
theorem p_vs_np : P = NP ∨ P ≠ NP := by sorry  -- lol
```

**Why**: To formalize complexity theory proofs in Lean.

**Status**: Some work in Mathlib but incomplete.

---

### 5.2 Circuit Complexity

```lean
-- Boolean circuits
inductive Circuit where
  | input : ℕ → Circuit
  | and : Circuit → Circuit → Circuit
  | or : Circuit → Circuit → Circuit
  | not : Circuit → Circuit

def CircuitSize : Circuit → ℕ
def CircuitDepth : Circuit → ℕ
def Computes (c : Circuit) (f : Bool^n → Bool) : Prop

-- Complexity classes
def AC0 : Set (ℕ → (Bool^n → Bool)) := ...
def P_poly : Set (ℕ → (Bool^n → Bool)) := ...
```

**Why**: Circuit complexity is the main approach to P vs NP.

---

## Priority 6: AI Integration Points

### 6.1 LLM Tactic Generation

```lean
-- Call LLM to generate tactic
syntax "llm_suggest" : tactic

-- Returns list of (tactic, confidence) pairs
-- User picks one, Lean verifies

@[llm_tactic]
partial def llmSuggest : TacticM (List (String × Float)) := do
  let goal ← getMainGoal
  let ctx ← getLCtx
  let prompt := formatForLLM goal ctx
  let suggestions ← callLLM prompt
  return suggestions
```

---

### 6.2 Conjecture Testing

```lean
-- Test if conjecture has obvious counterexamples
def testConjecture (conj : Expr) (numTests : ℕ := 1000) : IO Bool := do
  -- Generate random instances
  -- Check if any violate the conjecture
  ...

-- #test my_conjecture  -- Quick check before attempting proof
```

**Why**: Don't waste time proving false conjectures.

---

## Summary: What Z4/Archimedes Needs from Lean 5

| Feature | Priority | Use Case |
|---------|----------|----------|
| Rust FFI for tactics | P0 | Call Z4 directly |
| DRAT proof import | P0 | Convert Z4 proofs to Lean proofs |
| Goal state API | P1 | LLM-guided proof search |
| Tactic suggestion hook | P1 | AI suggests, human approves |
| SAT/SMT bridges | P2 | Automated theory reasoning |
| Checkpoint/resume | P2 | Long-running searches |
| Parallel tactics | P2 | Explore many approaches |
| Complexity theory defs | P3 | Formalize P vs NP work |

---

## Example Integration Flow

```
User: "Prove ∀ x y : Prop, (x → y) → (¬y → ¬x)"

Lean 5:
  1. Parse goal, create GoalState
  2. Call Z4 bridge with encoded formula

Z4:
  3. Solve as propositional SAT (trivially valid)
  4. Return proof certificate (empty clause derivation)

Lean 5:
  5. Verify certificate
  6. Construct Lean proof term
  7. Type-check proof

User: sees "Goals accomplished" ✓
```

---

## Contact

If building Lean 5 and want to discuss integration:
- Open issue on Z4 repo with `lean-integration` tag
- Or coordinate via Archimedes project channels

Let's make AI-assisted mathematics real.
