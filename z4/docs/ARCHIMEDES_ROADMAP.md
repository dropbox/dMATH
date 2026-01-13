# Archimedes: AI-Assisted Mathematical Research Platform

**Mission**: Vibecode new mathematics. Let AI explore the space of proofs, conjectures, and counterexamples while humans provide intuition and direction.

**Ultimate Goal**: Build infrastructure that could contribute to solving hard problems (P vs NP, Riemann Hypothesis, etc.) through systematic AI-assisted exploration.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ARCHIMEDES                                      │
│                     AI Mathematical Research Platform                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   EXPLORE     │          │    VERIFY     │          │   SEARCH      │
│  Conjecture   │          │  Prove/Check  │          │  Literature   │
│  Generation   │          │   Formally    │          │   & Analogy   │
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────────────────┐
│                            Z4 SOLVER STACK                                   │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────────────┤
│ z4-sat  │ z4-smt  │ z4-qbf  │z4-circuit│z4-proof │z4-cas   │ z4-lean-bridge │
│  CDCL   │ Theories│  QCDCL  │ Analysis │Complex  │ Algebra │  Verification  │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────────────────┐
│                           KNOWLEDGE LAYER                                    │
├──────────────────┬───────────────────┬──────────────────────────────────────┤
│   Math Graph     │   Proof Corpus    │         Barrier Database             │
│  (concepts,      │  (Mathlib, AFP,   │   (failed attempts, why they fail)   │
│   relations)     │   Z4 proofs)      │                                      │
└──────────────────┴───────────────────┴──────────────────────────────────────┘
```

---

## tRust Integration Priorities

tRust (verified Rust compiler) uses Z4 as its primary proof backend (~90% of verification conditions). These features are critical for adoption.

**Source:** `TRUST_FEATURE_REQUEST.md`, `docs/DIRECTIVE_FLOATING_POINT.md`

### Priority 1: Incremental Solving (CRITICAL)

**Current:** Fresh solver for each verification condition (1000 VCs = 1000 startups)
**Target:** Push/pop context with learned clause preservation

```rust
let mut solver = z4::Solver::incremental();
for vc in verification_conditions {
    solver.push();
    solver.assert(&vc);
    solver.check();  // Reuses learned lemmas from previous VCs
    solver.pop();
}
// 10-100x speedup on large codebases
```

**Implementation:**
- Add `push()` / `pop()` context stack to z4-dpll
- Preserve learned clauses across push/pop boundaries
- Track assertion levels for efficient backtracking
- **Crate:** z4-dpll, z4-core

### Priority 2: Minimal Counterexamples (HIGH)

**Current:** `x = 847293847` (arbitrary satisfying value)
**Target:** `x = 0` (minimal, readable counterexample)

```rust
enum CounterexampleStyle {
    Any,       // Fast, current behavior
    Minimal,   // Prefer 0, 1, -1, MIN, MAX
    Readable,  // Prefer round numbers
}
solver.set_counterexample_style(CounterexampleStyle::Minimal);
```

**Implementation:**
- Post-SAT model minimization pass
- Preference ordering: 0 > ±1 > powers of 2 > MIN/MAX > arbitrary
- Unset don't-care variables
- **Crate:** z4-dpll model extraction

### Priority 3: Proof Certificates (HIGH)

**Current:** "Result: Valid" (trust me)
**Target:** Independent verification of solver results

```rust
struct ProofCertificate {
    steps: Vec<ProofStep>,
    checksum: [u8; 32],
    format: CertificateFormat,  // DRAT, LRAT, Alethe
}
```

**Implementation:**
- DRAT already implemented for SAT (extend to SMT)
- Add Alethe format for SMT proofs (tRust integration standard)
- Independent checker library (z4-proof-checker)
- **Crate:** z4-proof (new), z4-sat (DRAT), z4-dpll (theory proofs)

### Priority 4: Native Rust API (COMPLETE)

Z4 is already pure Rust with idiomatic API. No C FFI, no lifetime gymnastics.

```rust
let mut solver = z4::Solver::new();
let x = solver.int_var("x");
let y = solver.int_var("y");
solver.assert(x.gt(0));
solver.assert(y.eq(x.add(1)));
```

### Priority 5: Custom Theory Plugins (MEDIUM)

Allow tRust to define Rust-specific theories:

```rust
trait TheoryPlugin {
    fn name(&self) -> &str;
    fn check_consistency(&mut self, model: &PartialModel) -> TheoryResult;
    fn propagate(&mut self, trail: &Trail) -> Vec<Literal>;
    fn explain(&self, lit: Literal) -> Vec<Literal>;
}

// Example: Rust ownership theory
solver.register_theory(RustOwnership::new());
```

**Use cases:**
- Slice bounds (`arr[i]` valid iff `i < arr.len()`)
- Option semantics (`x.unwrap()` valid iff `x.is_some()`)
- Reference validity (lifetime checking)
- **Crate:** z4-theories (plugin trait)

### Priority 6: Floating Point Theory (MEDIUM)

IEEE 754 support for `f32`/`f64` verification:

```rust
let x = Float64::new(&solver, "x");
let y = Float64::new(&solver, "y");
solver.assert(x.is_finite());
solver.assert(!x.add(y, RoundingMode::NearestEven).is_nan());
```

**Implementation approach:** Hybrid (real approximation + bit-precise for edge cases)
**Phases:**
1. Basic FP sort with comparisons and special values
2. Arithmetic with rounding modes
3. Transcendentals and error bound reasoning

**Crate:** z4-theories/fp (new)

### tRust Performance Targets

| Metric | Current Z3 | Z4 Target |
|--------|------------|-----------|
| Simple VC (arithmetic) | 5-10ms | <1ms |
| Medium VC (quantifiers) | 50-200ms | <20ms |
| Incremental overhead | N/A | <0.1ms per push/pop |
| Counterexample minimization | N/A | <10ms additional |
| Proof certificate generation | N/A | <5ms additional |

---

## New Z4 Crates Roadmap

### Phase A: Extended Solving (Prerequisites)

| Crate | Purpose | Dependencies | Effort |
|-------|---------|--------------|--------|
| **z4-qbf** | Quantified Boolean Formulas (∀∃ SAT) | z4-sat | 2 weeks |
| **z4-maxsat** | Optimization (MAX-SAT, weighted) | z4-sat | 1 week |
| **z4-allsat** | Enumerate all solutions | z4-sat | 1 week |

### Phase B: Complexity Analysis Tools

| Crate | Purpose | Dependencies | Effort |
|-------|---------|--------------|--------|
| **z4-circuits** | Boolean circuit analysis & synthesis | z4-sat | 3 weeks |
| **z4-proof-complexity** | Proof system lower bounds | z4-sat, z4-proof | 4 weeks |
| **z4-barriers** | Relativization/Natural/Algebrization checks | z4-circuits | 2 weeks |

### Phase C: Formal Verification Bridge

| Crate | Purpose | Dependencies | Effort |
|-------|---------|--------------|--------|
| **z4-lean-bridge** | Lean 5 FFI, tactic integration | z4-sat, z4-smt | 3 weeks |
| **z4-tla-bridge** | TLA+ model checking integration | z4-sat | 1 week |
| **z4-verified** | Verified core (Kani proofs) | z4-sat | 2 weeks |

### Phase D: Mathematical Infrastructure

| Crate | Purpose | Dependencies | Effort |
|-------|---------|--------------|--------|
| **z4-fp** | IEEE 754 Floating Point Theory (tRust) | z4-core, z4-bv | 3 weeks |
| **z4-proof** | Proof certificates (Alethe, LRAT) | z4-sat, z4-dpll | 2 weeks |
| **z4-cas** | Computer Algebra System (polynomials, groups) | num-* | 4 weeks |
| **z4-graph** | Mathematical knowledge graph | - | 2 weeks |
| **z4-search** | Semantic search over proofs/papers | z4-graph | 2 weeks |

### Phase E: AI Orchestration

| Crate | Purpose | Dependencies | Effort |
|-------|---------|--------------|--------|
| **archimedes-core** | LLM orchestration, agent coordination | all z4-* | 4 weeks |
| **archimedes-explore** | Conjecture generation & testing | archimedes-core | 3 weeks |
| **archimedes-prove** | Automated proof search | archimedes-core, z4-lean-bridge | 4 weeks |

---

## Detailed Crate Specifications

### z4-qbf: Quantified Boolean Formulas

**Why**: QBF captures PSPACE. Many complexity questions reduce to QBF. Essential for circuit lower bound proofs.

```rust
// crates/z4-qbf/src/lib.rs

/// Quantified Boolean Formula
pub enum QBF {
    Var(Variable),
    Not(Box<QBF>),
    And(Vec<QBF>),
    Or(Vec<QBF>),
    Forall(Variable, Box<QBF>),
    Exists(Variable, Box<QBF>),
}

/// QCDCL Solver (Quantified CDCL)
pub struct QbfSolver {
    sat_solver: z4_sat::Solver,
    quantifier_prefix: Vec<Quantifier>,
}

impl QbfSolver {
    /// Solve QBF instance
    pub fn solve(&mut self, formula: &QBF) -> QbfResult;

    /// Generate certificate (Skolem functions for SAT, Herbrand for UNSAT)
    pub fn certificate(&self) -> Certificate;
}
```

**Algorithm**: QCDCL (Quantified CDCL) with:
- Quantifier-aware unit propagation
- Universal reduction
- Dependency learning
- Certificate generation (Skolem/Herbrand functions)

**References**:
- DepQBF (Lonsing & Biere)
- QCDCL (Zhang & Malik)

---

### z4-circuits: Boolean Circuit Analysis

**Why**: P vs NP is fundamentally about circuit complexity. Need tools to analyze, synthesize, and prove bounds on circuits.

```rust
// crates/z4-circuits/src/lib.rs

/// Boolean circuit (DAG of gates)
pub struct Circuit {
    inputs: Vec<Variable>,
    gates: Vec<Gate>,
    outputs: Vec<GateId>,
}

pub enum Gate {
    Input(Variable),
    And(GateId, GateId),
    Or(GateId, GateId),
    Not(GateId),
    Xor(GateId, GateId),
    // Threshold gates for TC⁰
    Threshold { inputs: Vec<GateId>, threshold: usize },
    // Mod gates for ACC⁰
    Mod { inputs: Vec<GateId>, modulus: usize },
}

pub struct CircuitAnalyzer;

impl CircuitAnalyzer {
    /// Circuit size (number of gates)
    pub fn size(c: &Circuit) -> usize;

    /// Circuit depth (longest path)
    pub fn depth(c: &Circuit) -> usize;

    /// Check if circuit is in complexity class
    pub fn is_in_class(c: &Circuit, class: CircuitClass) -> bool;

    /// Synthesize minimum circuit for function (brute force, small n)
    pub fn synthesize_minimum(f: &TruthTable, bound: usize) -> Option<Circuit>;

    /// Enumerate all circuits of given size
    pub fn enumerate(inputs: usize, size: usize) -> impl Iterator<Item = Circuit>;

    /// Check if circuit computes function (via Z4 SAT)
    pub fn computes(&self, c: &Circuit, f: &TruthTable) -> bool;

    /// Encode "circuit C computes SAT" as SAT formula
    pub fn encode_circuit_sat(c: &Circuit, phi: &Formula) -> Formula;
}

pub enum CircuitClass {
    AC0,      // Constant depth, unbounded fan-in AND/OR
    ACC0,     // AC⁰ + MOD gates
    TC0,      // AC⁰ + threshold gates
    NC1,      // Log depth, bounded fan-in
    P_Poly,   // Polynomial size
}
```

**Key capabilities**:
1. **Circuit synthesis**: Find smallest circuit for a function
2. **Circuit enumeration**: Systematically explore circuit space
3. **SAT encoding**: "Does circuit C of size s solve formula φ?"
4. **Class membership**: Check if circuit is in AC⁰, TC⁰, etc.

**References**:
- Jukna, "Boolean Function Complexity"
- Williams, "Non-uniform ACC Circuit Lower Bounds"

---

### z4-proof-complexity: Proof System Analysis

**Why**: If we prove exponential lower bounds on proof size in all proof systems, we separate P from NP. Also useful for understanding SAT solver behavior.

```rust
// crates/z4-proof-complexity/src/lib.rs

/// Propositional proof systems
pub enum ProofSystem {
    Resolution,
    TreeResolution,
    RegularResolution,
    ExtendedResolution,
    Frege,
    ExtendedFrege,
    Cutting Planes,
    Polynomial Calculus,
    SOS,  // Sum of Squares
}

pub struct ProofComplexityAnalyzer {
    sat_solver: z4_sat::Solver,
}

impl ProofComplexityAnalyzer {
    /// Find proof in system (if exists within size bound)
    pub fn find_proof(
        &self,
        formula: &Formula,
        system: ProofSystem,
        size_bound: usize
    ) -> Option<Proof>;

    /// Verify proof in system
    pub fn verify_proof(&self, proof: &Proof, system: ProofSystem) -> bool;

    /// Translate proof between systems
    pub fn translate(
        &self,
        proof: &Proof,
        from: ProofSystem,
        to: ProofSystem
    ) -> Option<Proof>;

    /// Prove lower bound on proof size (via game/adversary methods)
    pub fn prove_lower_bound(
        &self,
        formula: &Formula,
        system: ProofSystem
    ) -> Option<LowerBound>;

    /// Generate hard formulas for system (pigeonhole, Tseitin, random k-CNF)
    pub fn hard_formula(system: ProofSystem, size: usize) -> Formula;
}

/// Known hard formulas
pub mod hard_formulas {
    /// Pigeonhole principle: n+1 pigeons, n holes
    pub fn pigeonhole(n: usize) -> Formula;

    /// Tseitin formulas on graph
    pub fn tseitin(graph: &Graph, parity: &[bool]) -> Formula;

    /// Random k-CNF at clause/variable ratio
    pub fn random_k_cnf(k: usize, n: usize, ratio: f64) -> Formula;

    /// Parity constraints
    pub fn parity(n: usize) -> Formula;
}
```

**References**:
- Krajíček, "Proof Complexity"
- Cook & Reckhow, "The Relative Efficiency of Propositional Proof Systems"

---

### z4-barriers: P vs NP Barrier Checker

**Why**: Automatically detect when a proof attempt hits known barriers. Save years of wasted effort.

```rust
// crates/z4-barriers/src/lib.rs

/// The three main barriers to P vs NP proofs
pub enum Barrier {
    /// Proof works relative to some oracle, fails relative to another
    Relativization {
        positive_oracle: Oracle,   // Makes P = NP
        negative_oracle: Oracle,   // Makes P ≠ NP
    },

    /// Proof uses "natural" combinatorial properties
    NaturalProof {
        largeness: bool,      // Property holds for many functions
        constructivity: bool, // Property is efficiently testable
    },

    /// Proof works in algebraic setting where P = NP
    Algebrization {
        algebraic_oracle: AlgebraicOracle,
    },
}

pub struct BarrierChecker {
    circuit_analyzer: z4_circuits::CircuitAnalyzer,
}

impl BarrierChecker {
    /// Check if proof technique relativizes
    /// Returns oracles that break the proof if it does
    pub fn check_relativization(&self, proof: &ProofSketch) -> Option<Barrier>;

    /// Check if lower bound proof is "natural"
    /// (uses largeness + constructivity)
    pub fn check_natural_proof(&self, proof: &LowerBoundProof) -> Option<Barrier>;

    /// Check if proof algebrizes
    pub fn check_algebrization(&self, proof: &ProofSketch) -> Option<Barrier>;

    /// Run all barrier checks
    pub fn full_check(&self, proof: &ProofSketch) -> Vec<Barrier>;
}

/// Proof sketch representation for barrier analysis
pub struct ProofSketch {
    /// What complexity classes are being separated?
    separation: (ComplexityClass, ComplexityClass),

    /// What properties of the hard function are used?
    properties_used: Vec<FunctionProperty>,

    /// Does proof use diagonalization?
    uses_diagonalization: bool,

    /// Does proof use circuit lower bounds?
    uses_circuit_bounds: bool,
}
```

**References**:
- Baker, Gill, Solovay, "Relativizations of the P =? NP Question"
- Razborov, Rudich, "Natural Proofs"
- Aaronson, Wigderson, "Algebrization"

---

### z4-lean-bridge: Lean 5 Integration

**Why**: Lean provides ground truth for proof correctness. Z4 provides computational power. Together they're stronger.

```rust
// crates/z4-lean-bridge/src/lib.rs

use std::process::Command;

/// Bridge to Lean 5 proof assistant
pub struct LeanBridge {
    lean_path: PathBuf,
    project_path: PathBuf,
}

impl LeanBridge {
    /// Check if Lean statement is provable (with timeout)
    pub fn check_provable(&self, statement: &str, timeout: Duration) -> LeanResult;

    /// Run Lean tactic and get result
    pub fn run_tactic(&self, goal: &str, tactic: &str) -> TacticResult;

    /// Verify Z4 SAT result in Lean
    pub fn verify_sat(&self, formula: &Formula, model: &Model) -> bool;

    /// Verify Z4 UNSAT result in Lean (via DRAT proof)
    pub fn verify_unsat(&self, formula: &Formula, proof: &DratProof) -> bool;

    /// Export Z4 formula to Lean syntax
    pub fn export_formula(&self, formula: &Formula) -> String;

    /// Import Lean goal as Z4 formula
    pub fn import_goal(&self, goal: &str) -> Option<Formula>;

    /// Suggest tactics using Z4 (SAT-based tactic selection)
    pub fn suggest_tactics(&self, goal: &str) -> Vec<String>;
}

pub enum LeanResult {
    Proved(Proof),
    Disproved(Counterexample),
    Unknown,
    Timeout,
}

/// Lean tactic that calls Z4
///
/// ```lean
/// theorem example : ∀ x y : Bool, x && y = y && x := by
///   z4_decide  -- Calls Z4 SAT solver
/// ```
pub fn generate_z4_tactic() -> String {
    include_str!("lean/Z4Tactic.lean").to_string()
}
```

**Lean side** (`lean/Z4Tactic.lean`):
```lean
import Lean
import Std.Tactic.BVDecide

/-- Call Z4 SAT solver to decide propositional goals -/
syntax "z4_decide" : tactic

/-- Call Z4 SMT solver for theory goals -/
syntax "z4_smt" : tactic

macro_rules
| `(tactic| z4_decide) => `(tactic| native_decide)  -- placeholder
```

---

### z4-cas: Computer Algebra System

**Why**: Many advanced proof techniques (GCT, algebraic methods) need symbolic algebra.

```rust
// crates/z4-cas/src/lib.rs

/// Symbolic expression
pub enum Expr {
    Const(Rational),
    Var(String),
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    Pow(Box<Expr>, i32),
    Neg(Box<Expr>),
    // Polynomial-specific
    Polynomial(Polynomial),
}

/// Multivariate polynomial
pub struct Polynomial {
    terms: Vec<(Rational, Monomial)>,
    ordering: MonomialOrdering,
}

pub struct CAS;

impl CAS {
    // Basic operations
    pub fn simplify(expr: &Expr) -> Expr;
    pub fn expand(expr: &Expr) -> Expr;
    pub fn factor(poly: &Polynomial) -> Vec<Polynomial>;

    // Polynomial operations
    pub fn grobner_basis(polys: &[Polynomial], ordering: MonomialOrdering) -> Vec<Polynomial>;
    pub fn resultant(p: &Polynomial, q: &Polynomial, var: &str) -> Polynomial;
    pub fn gcd(p: &Polynomial, q: &Polynomial) -> Polynomial;

    // Solving
    pub fn solve_system(equations: &[Expr], vars: &[String]) -> Vec<Solution>;
    pub fn find_roots(poly: &Polynomial) -> Vec<Complex>;

    // Pattern matching
    pub fn match_pattern(expr: &Expr, pattern: &Pattern) -> Option<Substitution>;
    pub fn find_pattern(examples: &[Expr]) -> Option<Expr>;

    // Group theory (for GCT approach)
    pub fn character(group: &Group, representation: &Rep) -> Polynomial;
    pub fn kronecker_coefficient(lambda: &Partition, mu: &Partition, nu: &Partition) -> i64;
}
```

---

### archimedes-core: AI Orchestration

**Why**: Coordinate LLMs, solvers, and provers into a coherent research system.

```rust
// crates/archimedes-core/src/lib.rs

/// Main orchestration for AI math research
pub struct Archimedes {
    // Solvers
    sat_solver: z4_sat::Solver,
    smt_solver: z4_smt::Solver,
    qbf_solver: z4_qbf::QbfSolver,

    // Analysis tools
    circuit_analyzer: z4_circuits::CircuitAnalyzer,
    proof_analyzer: z4_proof_complexity::ProofComplexityAnalyzer,
    barrier_checker: z4_barriers::BarrierChecker,

    // Verification
    lean_bridge: z4_lean_bridge::LeanBridge,

    // Knowledge
    math_graph: z4_graph::MathGraph,

    // LLM interface
    llm: Box<dyn LLMProvider>,

    // Persistent state
    state: ResearchState,
}

impl Archimedes {
    /// Explore a mathematical problem
    pub async fn explore(&mut self, problem: &str) -> ExplorationResult {
        // 1. Parse and understand the problem
        let parsed = self.llm.parse_problem(problem).await?;

        // 2. Search for related work
        let related = self.math_graph.find_related(&parsed);
        let failed_attempts = self.state.failed_attempts(&parsed);

        // 3. Generate conjectures
        let conjectures = self.llm.generate_conjectures(&parsed, &related).await?;

        // 4. Test conjectures
        for conjecture in conjectures {
            // Quick SAT/SMT check for obvious counterexamples
            if let Some(cex) = self.find_counterexample(&conjecture) {
                self.state.record_failure(&conjecture, cex);
                continue;
            }

            // Check barriers (for complexity conjectures)
            if let Some(barrier) = self.barrier_checker.full_check(&conjecture) {
                self.state.record_barrier(&conjecture, barrier);
                continue;
            }

            // Try to prove in Lean
            match self.lean_bridge.try_prove(&conjecture).await {
                LeanResult::Proved(proof) => {
                    self.state.record_proof(&conjecture, proof);
                    return ExplorationResult::Proved(conjecture, proof);
                }
                LeanResult::Disproved(cex) => {
                    self.state.record_failure(&conjecture, cex);
                }
                _ => {
                    // Add to open problems
                    self.state.add_open_problem(&conjecture);
                }
            }
        }

        ExplorationResult::OpenProblems(self.state.open_problems())
    }

    /// Attack a specific problem with full resources
    pub async fn attack(&mut self, problem: &Problem) -> AttackResult {
        // Spawn multiple approaches in parallel
        let approaches = vec![
            self.try_direct_proof(problem),
            self.try_circuit_lower_bound(problem),
            self.try_proof_complexity(problem),
            self.try_algebraic(problem),
        ];

        // Race approaches, share discoveries
        let (result, discoveries) = self.race_approaches(approaches).await;

        // Record what we learned
        for discovery in discoveries {
            self.state.record_discovery(discovery);
        }

        result
    }

    /// Vibecode: Let AI explore freely with minimal guidance
    pub async fn vibecode(&mut self, direction: &str) -> Stream<Discovery> {
        loop {
            // Generate interesting question
            let question = self.llm.generate_question(direction, &self.state).await?;

            // Explore it
            let result = self.explore(&question).await;

            // Yield discoveries
            for discovery in result.discoveries() {
                yield discovery;
            }

            // Update direction based on what we learned
            direction = self.llm.refine_direction(direction, &result).await?;
        }
    }
}

/// LLM provider trait (Claude, GPT, Gemini, etc.)
#[async_trait]
pub trait LLMProvider {
    async fn parse_problem(&self, problem: &str) -> Result<ParsedProblem>;
    async fn generate_conjectures(&self, problem: &ParsedProblem, context: &[RelatedWork]) -> Result<Vec<Conjecture>>;
    async fn suggest_proof_strategy(&self, conjecture: &Conjecture) -> Result<Strategy>;
    async fn formalize(&self, statement: &str) -> Result<LeanCode>;
    async fn generate_question(&self, direction: &str, state: &ResearchState) -> Result<String>;
    async fn refine_direction(&self, direction: &str, result: &ExplorationResult) -> Result<String>;
}
```

---

## Updated Workspace

```toml
# Cargo.toml additions

[workspace]
members = [
    # ... existing crates ...

    # Phase A: Extended Solving
    "crates/z4-qbf",
    "crates/z4-maxsat",
    "crates/z4-allsat",

    # Phase B: Complexity Analysis
    "crates/z4-circuits",
    "crates/z4-proof-complexity",
    "crates/z4-barriers",

    # Phase C: Verification Bridge
    "crates/z4-lean-bridge",
    "crates/z4-tla-bridge",
    "crates/z4-verified",

    # Phase D: Mathematical Infrastructure
    "crates/z4-fp",       # tRust: IEEE 754 floating point
    "crates/z4-proof",    # tRust: Proof certificates (Alethe, LRAT)
    "crates/z4-cas",
    "crates/z4-graph",
    "crates/z4-search",

    # Phase E: AI Orchestration
    "crates/archimedes-core",
    "crates/archimedes-explore",
    "crates/archimedes-prove",
]

[workspace.dependencies]
# ... existing deps ...

# New dependencies
z4-qbf = { path = "crates/z4-qbf" }
z4-maxsat = { path = "crates/z4-maxsat" }
z4-allsat = { path = "crates/z4-allsat" }
z4-circuits = { path = "crates/z4-circuits" }
z4-proof-complexity = { path = "crates/z4-proof-complexity" }
z4-barriers = { path = "crates/z4-barriers" }
z4-lean-bridge = { path = "crates/z4-lean-bridge" }
z4-tla-bridge = { path = "crates/z4-tla-bridge" }
z4-verified = { path = "crates/z4-verified" }
z4-cas = { path = "crates/z4-cas" }
z4-graph = { path = "crates/z4-graph" }
z4-search = { path = "crates/z4-search" }
z4-fp = { path = "crates/z4-fp" }           # tRust: IEEE 754 floating point
z4-proof = { path = "crates/z4-proof" }     # tRust: Proof certificates
archimedes-core = { path = "crates/archimedes-core" }
archimedes-explore = { path = "crates/archimedes-explore" }
archimedes-prove = { path = "crates/archimedes-prove" }

# LLM integration
async-trait = "0.1"
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

---

## Execution Order

### Immediate (Complete Phase 1 SAT first)
1. ✅ Finish z4-sat optimizations (heap VSIDS, etc.)
2. ✅ Complete verification (DRAT, Kani)
3. ✅ Fix LIA soundness bug (learned clause preservation)

### tRust Integration (CRITICAL PATH)
4. **Incremental solving** - push/pop context in z4-dpll
5. **Minimal counterexamples** - model minimization pass
6. **z4-proof** - Alethe certificates for SMT proofs

### Short-term (Weeks 1-4)
7. **z4-qbf** - Enables complexity reductions
8. **z4-circuits** - Core tool for P vs NP work
9. **z4-lean-bridge** - Ground truth verification

### Medium-term (Weeks 5-12)
10. **z4-fp** - IEEE 754 floating point theory (tRust)
11. **Theory plugins** - Custom theory registration API
12. **z4-proof-complexity** - Proof system analysis
13. **z4-barriers** - Automatic barrier detection
14. **z4-cas** - Symbolic algebra

### Long-term (Weeks 13+)
15. **archimedes-core** - LLM orchestration
16. **archimedes-explore** - Conjecture generation
17. **archimedes-prove** - Automated proof search

---

## Success Metrics

### tRust Integration (Critical)

| Milestone | Metric |
|-----------|--------|
| Incremental solving | <0.1ms push/pop overhead, 10x speedup on VC batches |
| Minimal counterexamples | Prefer 0, 1, -1 over arbitrary values in 90%+ of cases |
| Proof certificates | Alethe format generation, independent verification passes |
| FP theory | Prove `magnitude()` safe, find bug in `unsafe_div()` |
| Theory plugins | tRust can register custom Rust ownership theory |

### Archimedes Research Platform

| Milestone | Metric |
|-----------|--------|
| z4-qbf works | Solve QBFLIB benchmarks |
| z4-circuits works | Enumerate circuits, find minimum for 4-bit functions |
| z4-lean-bridge works | Verify Z4 SAT results in Lean |
| z4-barriers works | Correctly identify barrier violations in test cases |
| archimedes-core works | Generate and test 100 conjectures autonomously |
| **Moonshot** | Contribute to a publishable mathematical result |

---

## References

### QBF
- Lonsing & Biere, "DepQBF: A Dependency-Aware QBF Solver"
- Zhang & Malik, "Conflict Driven Learning in a Quantified Boolean Satisfiability Solver"

### Circuit Complexity
- Jukna, "Boolean Function Complexity: Advances and Frontiers"
- Williams, "Non-uniform ACC Circuit Lower Bounds"
- Agrawal & Saptharishi, "Arithmetic Circuits: A Survey"

### Proof Complexity
- Krajíček, "Proof Complexity"
- Cook & Reckhow, "The Relative Efficiency of Propositional Proof Systems"
- Beame & Pitassi, "Propositional Proof Complexity: Past, Present, and Future"

### Barriers
- Baker, Gill, Solovay, "Relativizations of the P =? NP Question" (1975)
- Razborov & Rudich, "Natural Proofs" (1997)
- Aaronson & Wigderson, "Algebrization: A New Barrier" (2009)

### AI for Math
- Polu & Sutskever, "Generative Language Modeling for Automated Theorem Proving"
- First et al., "Baldur: Whole-Proof Generation and Repair with LLMs"
- Trinh et al., "AlphaGeometry" (2024)
- AlphaProof (2024)

---

*"The only way to do great work is to love what you do. If you haven't found it yet, keep looking. Don't settle."* - Steve Jobs

*"Let's vibecode some new math."* - You, 2025
