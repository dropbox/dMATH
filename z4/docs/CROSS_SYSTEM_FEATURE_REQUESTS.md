# Cross-System Feature Requests for Archimedes

**Project**: Archimedes - AI-assisted mathematical research platform
**Goal**: Vibecode new mathematics by connecting LLMs, solvers, and proof assistants
**Contact**: Z4 project (this repo)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ARCHIMEDES                                   │
│              AI Mathematical Research Platform                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│    LLMs     │       │     Z4      │       │   Lean 5    │
│  (Creative) │       │  (Compute)  │       │  (Verify)   │
│ Claude/GPT/ │       │  SAT/SMT/   │       │   Proofs    │
│   Gemini    │       │  QBF/etc    │       │   Ground    │
└─────────────┘       └─────────────┘       │   Truth     │
                                            └─────────────┘
```

Each system has a role:
- **LLMs**: Generate conjectures, suggest proof strategies, find analogies
- **Z4**: Fast computation - SAT solving, counterexample search, circuit analysis
- **Lean 5**: Verification - proof checking, ground truth, certified results

---

## Feature Requests by System

---

# FOR: Z4 (SAT/SMT Solver - This Project)

**Status**: In development
**Role**: Computational workhorse for Archimedes

## Must Build (Phase 1 - Current)

### 1.1 Complete CDCL SAT Solver
- [x] Basic CDCL with 2WL
- [x] VSIDS with heap
- [x] Glucose restarts
- [ ] DRAT proof generation (all paths)
- [ ] Model verification (all SAT results)

### 1.2 DRAT/LRAT Proof Generation
**Critical**: Every UNSAT must have a machine-checkable proof.

```rust
pub struct Solver {
    /// Generate DRAT proof for UNSAT results
    pub fn solve_with_proof(&mut self, formula: &Formula) -> (SolveResult, Option<DratProof>);

    /// Generate LRAT proof (includes deletion info, faster to check)
    pub fn solve_with_lrat(&mut self, formula: &Formula) -> (SolveResult, Option<LratProof>);
}
```

## Must Build (Phase 2 - Archimedes Prerequisites)

### 2.1 QBF Solver (z4-qbf)
**Why**: QBF captures PSPACE. Many complexity reductions need quantifiers.

```rust
// Core API
pub fn solve_qbf(formula: &QBF) -> QbfResult;
pub fn solve_qbf_with_certificate(formula: &QBF) -> (QbfResult, Certificate);

// Certificate types
pub enum Certificate {
    Skolem(SkolemFunctions),   // For SAT: witness functions
    Herbrand(HerbrandFunctions), // For UNSAT: counterexample functions
}
```

**Algorithm**: QCDCL (Quantified CDCL)
**Reference**: DepQBF, CAQE

### 2.2 AllSAT Solver (z4-allsat)
**Why**: Enumerate all solutions for exhaustive analysis.

```rust
// Enumerate all satisfying assignments
pub fn all_sat(formula: &Formula) -> impl Iterator<Item = Model>;

// Count solutions (using #SAT techniques)
pub fn count_sat(formula: &Formula) -> BigUint;

// Enumerate up to limit
pub fn all_sat_bounded(formula: &Formula, limit: usize) -> Vec<Model>;
```

### 2.3 Circuit Analysis (z4-circuits)
**Why**: P vs NP is about circuits. Need tools to analyze them.

```rust
// Core types
pub struct Circuit { /* DAG of gates */ }
pub struct TruthTable { /* 2^n bit vector */ }

// Analysis
pub fn circuit_size(c: &Circuit) -> usize;
pub fn circuit_depth(c: &Circuit) -> usize;
pub fn circuit_to_cnf(c: &Circuit) -> Formula;
pub fn cnf_to_circuit(f: &Formula) -> Circuit;

// Synthesis
pub fn synthesize_minimum(tt: &TruthTable, max_size: usize) -> Option<Circuit>;
pub fn enumerate_circuits(inputs: usize, size: usize) -> impl Iterator<Item = Circuit>;

// Equivalence checking (via SAT)
pub fn circuits_equivalent(c1: &Circuit, c2: &Circuit) -> bool;
```

## Should Build (Phase 3)

### 3.1 SMT Theories
- EUF (equality + uninterpreted functions)
- LIA (linear integer arithmetic)
- LRA (linear real arithmetic)
- BV (bitvectors)
- Arrays

### 3.2 Optimization
- MaxSAT (weighted, partial)
- Optimization Modulo Theories (OMT)

### 3.3 Proof Complexity Tools
- Resolution proof extraction
- Proof size analysis
- Hard formula generators (pigeonhole, Tseitin, etc.)

## API Requirements for Lean Integration

```rust
// JSON-based API for cross-language calls
pub mod api {
    #[derive(Serialize, Deserialize)]
    pub struct SolveRequest {
        pub formula: String,  // DIMACS or SMT-LIB format
        pub timeout_ms: Option<u64>,
        pub proof_format: Option<ProofFormat>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct SolveResponse {
        pub result: String,  // "sat", "unsat", "unknown"
        pub model: Option<Vec<i32>>,  // Variable assignments
        pub proof: Option<String>,  // DRAT/LRAT proof
        pub stats: SolveStats,
    }

    // For FFI
    #[no_mangle]
    pub extern "C" fn z4_solve(dimacs: *const c_char, result: *mut c_char) -> i32;
}
```

---

# FOR: Lean 5 (Proof Assistant)

**Status**: Being developed (successor to Lean 4)
**Role**: Ground truth verification for Archimedes

## Priority 0: External Solver FFI

### Request: Native Rust FFI
Allow Lean tactics to call Rust functions directly.

```lean
-- Declare external Rust function
@[extern "z4_solve_sat"]
opaque z4SolveSat : @& ByteArray → IO (Option (Array Int))

-- Use in tactic
syntax "z4_decide" : tactic
```

**Why**: JSON/pipe is too slow for tight integration. Need native FFI.

**Current gap**: Lean 4 has limited C FFI. Need Rust support or C wrapper.

### Request: DRAT/LRAT Proof Import

```lean
-- Built-in DRAT verifier
def verifyDrat (formula : CNF) (proof : DratProof) : Bool

-- Tactic that uses external SAT solver + DRAT
syntax "sat_decide" : tactic
-- 1. Encodes goal as SAT
-- 2. Calls Z4
-- 3. If UNSAT: verifies DRAT, constructs Lean proof
-- 4. If SAT: provides counterexample
```

**Why**: Z4 generates proofs. Lean should accept them as valid.

## Priority 1: Proof Search API

### Request: Goal State Access

```lean
-- External tools can query current proof state
structure GoalState where
  target : Expr           -- What we're trying to prove
  context : List LocalDecl -- Available hypotheses
  metavars : List MVarId  -- Unresolved goals

-- Export current state
@[export lean_get_goal_state]
def getGoalState : TacticM GoalState
```

**Why**: LLM needs to see what we're proving to suggest tactics.

### Request: Tactic Execution API

```lean
-- Try tactic, get resulting state or error
@[export lean_try_tactic]
def tryTactic (state : GoalState) (tactic : String) : IO (Except String GoalState)

-- Batch try multiple tactics
@[export lean_try_tactics]
def tryTactics (state : GoalState) (tactics : List String) : IO (List (String × Except String GoalState))
```

**Why**: Proof search needs to explore many tactics, backtrack on failure.

### Request: Parallel Tactic Execution

```lean
-- Try tactics in parallel, return first success
syntax "race" "[" tactic,* "]" : tactic

example : P := by
  race [simp, ring, omega, z4_decide]
```

**Why**: AI generates many candidates. Parallel exploration is faster.

## Priority 2: Theory Bridges

### Request: Propositional Logic Bridge

```lean
-- Convert Prop goal to CNF
def goalToCnf (goal : Expr) : Option CNF

-- Convert SAT model back to proof
def modelToProof (goal : Expr) (model : Array Int) : Except String Expr

-- The tactic
syntax "prop_decide" : tactic
-- Works for: ∀ p q r : Prop, (p ∧ q → r) ↔ (p → q → r)
```

### Request: Arithmetic Bridges

```lean
-- Linear Integer Arithmetic
syntax "lia_decide" : tactic  -- Calls Z4 LIA theory

-- Linear Real Arithmetic
syntax "lra_decide" : tactic  -- Calls Z4 LRA theory

-- Bitvectors
syntax "bv_decide" : tactic   -- Calls Z4 BV theory
```

## Priority 3: Mathematical Library

### Request: Complexity Theory Definitions

```lean
-- In Mathlib or stdlib
namespace Complexity

-- Turing machines
structure TuringMachine where
  states : Type
  alphabet : Type
  transition : states → alphabet → states × alphabet × Direction

-- Time/space complexity
def timeComplexity (tm : TuringMachine) (input : List alphabet) : ℕ
def spaceComplexity (tm : TuringMachine) (input : List alphabet) : ℕ

-- Complexity classes
def DTIME (t : ℕ → ℕ) : Set (List Bool → Bool)
def P : Set (List Bool → Bool) := ⋃ k, DTIME (fun n => n^k)
def NP : Set (List Bool → Bool) := ...

-- The question
axiom p_vs_np_undecided : True  -- placeholder

end Complexity
```

### Request: Circuit Complexity Definitions

```lean
namespace Circuits

inductive Gate where
  | input : ℕ → Gate
  | and : Gate → Gate → Gate
  | or : Gate → Gate → Gate
  | not : Gate → Gate

def Circuit := Gate  -- Output gate

def size : Circuit → ℕ
def depth : Circuit → ℕ
def eval : Circuit → (ℕ → Bool) → Bool

-- Complexity classes
def SIZE (s : ℕ → ℕ) : Set (ℕ → (Fin n → Bool) → Bool)
def P_poly : Set (...) := ⋃ k, SIZE (fun n => n^k)

end Circuits
```

## Priority 4: Infrastructure

### Request: Proof Recording

```lean
-- Record successful proofs for ML training
set_option trace.proof.export true

-- Exports: goal, tactics used, time, success/failure
-- Format: JSON lines for easy processing
```

### Request: Checkpoint/Resume

```lean
-- Save proof state to file (for long-running searches)
def checkpoint (state : ProofState) (path : String) : IO Unit
def resume (path : String) : IO ProofState
```

---

# FOR: TLA+ / TLA+ 2

**Status**: Considering whether needed
**Role**: Model checking for concurrent/distributed systems

## Assessment: Probably Not Needed for Math Discovery

TLA+ is great for:
- Specifying distributed algorithms
- Model checking finite state systems
- Verifying concurrent code

But for Archimedes (math discovery), we need:
- Proof verification (Lean)
- SAT/SMT solving (Z4)
- Symbolic reasoning (CAS)

**Verdict**: TLA+ is useful for verifying Z4 itself (concurrent data structures), but NOT for the math discovery pipeline.

## If We Do Use It

### Request: Better Integration with Lean
- Export TLA+ specs to Lean for proof
- Verify TLA+ models in Lean

### Request: SMT Backend Option
- Use Z4 as SMT backend for TLC
- Currently uses Z3

**Priority**: Low. Focus on Lean and Z4 first.

---

# FOR: Computer Algebra System (New Request)

**Status**: Need one, unclear which
**Role**: Symbolic manipulation for algebraic proofs

## Options

| System | License | Language | Notes |
|--------|---------|----------|-------|
| SymPy | BSD | Python | Good, but Python |
| Mathematica | Proprietary | Wolfram | Powerful, but closed |
| SageMath | GPL | Python | Comprehensive |
| GiNaC | GPL | C++ | Could wrap |
| Build our own | Apache | Rust | Full control, more work |

## What We Need

```rust
// Core symbolic types
pub enum Expr {
    Const(Rational),
    Var(String),
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    Pow(Box<Expr>, i32),
    // ...
}

// Essential operations
pub fn simplify(expr: &Expr) -> Expr;
pub fn expand(expr: &Expr) -> Expr;
pub fn factor(poly: &Polynomial) -> Vec<Polynomial>;
pub fn solve(equations: &[Expr], vars: &[String]) -> Vec<Solution>;

// Polynomial operations
pub fn gcd(p: &Polynomial, q: &Polynomial) -> Polynomial;
pub fn resultant(p: &Polynomial, q: &Polynomial, var: &str) -> Polynomial;
pub fn grobner_basis(polys: &[Polynomial]) -> Vec<Polynomial>;

// For GCT approach (ambitious)
pub fn kronecker_coefficient(lambda: &Partition, mu: &Partition, nu: &Partition) -> i64;
```

## Request: Build z4-cas

Minimal CAS in Rust, focused on what Archimedes needs:
- Polynomial arithmetic
- Grobner bases
- Basic symbolic simplification
- Group theory primitives

**Priority**: Medium. Start after z4-qbf and z4-circuits.

---

# FOR: Knowledge Graph System (New Request)

**Status**: Need one
**Role**: Store and query mathematical knowledge

## What We Need

```rust
// Store mathematical objects and their relationships
pub struct MathGraph {
    // Nodes: theorems, definitions, conjectures, techniques
    // Edges: uses, proves, generalizes, is_instance_of, etc.
}

impl MathGraph {
    // Find related theorems
    pub fn related_to(&self, theorem: &str) -> Vec<(String, Relation)>;

    // Find theorems using technique
    pub fn using_technique(&self, technique: &str) -> Vec<String>;

    // Find path between concepts
    pub fn path(&self, from: &str, to: &str) -> Option<Vec<String>>;

    // Semantic search
    pub fn search(&self, query: &str, limit: usize) -> Vec<(String, f32)>;
}
```

## Data Sources

- Mathlib (formalized theorems)
- arXiv (papers, abstracts)
- OEIS (integer sequences)
- MathOverflow (questions, conjectures)
- ProofWiki (informal proofs)

## Request: Build z4-graph

Graph database optimized for mathematical knowledge:
- Import from Mathlib, arXiv, etc.
- Embedding-based semantic search
- Relationship inference

**Priority**: Medium. Useful for conjecture generation.

---

# FOR: LLM Providers (Claude, GPT, Gemini)

**Status**: External services
**Role**: Creativity engine for Archimedes

## API Requirements

### Structured Output
Need reliable JSON output for:
- Conjecture generation
- Tactic suggestion
- Proof strategy

```json
{
  "conjectures": [
    {
      "statement": "∀ n : ℕ, prime n → ...",
      "confidence": 0.7,
      "reasoning": "Pattern from similar theorems..."
    }
  ]
}
```

### Long Context
- Need to include full proof context (10k+ tokens)
- Mathematical definitions and lemmas
- Previous attempts and failures

### Low Latency
- Proof search makes many calls
- Need <1s response for tactic suggestions

### Fine-tuning Access
- Fine-tune on Mathlib proofs
- Fine-tune on successful explorations
- Improve over time

## Request: Math-Specialized Models

Would be valuable:
- Models trained on Mathlib, arXiv math
- Reliable LaTeX/Lean syntax generation
- Understanding of proof structure

---

## Summary: What Each System Provides

| System | Provides | Needs From Others |
|--------|----------|-------------------|
| **Z4** | SAT/SMT solving, proofs, circuits | - |
| **Lean 5** | Proof verification, ground truth | Z4 proofs, FFI |
| **CAS** | Symbolic algebra | Z4 for solving |
| **Graph** | Knowledge retrieval | All: to index |
| **LLM** | Creativity, natural language | All: as tools |

## Integration Points

```
LLM ──generates──> Conjecture ──encodes──> Z4
                                            │
                                    checks counterexamples
                                            │
                                            ▼
                                   Survives? ──yes──> Lean 5
                                            │              │
                                            │         verifies proof
                                            │              │
                                            ▼              ▼
                                         Graph ◄───── New Theorem
                                     (stores result)
```

---

## Contact

This document is part of the Z4/Archimedes project.

To coordinate:
- Open issue on Z4 repo with `archimedes` tag
- Reference this document in cross-project discussions

Let's build the future of mathematical discovery.
