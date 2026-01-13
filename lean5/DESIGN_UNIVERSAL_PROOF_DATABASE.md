# Universal Proof Database (UPD) Design Specification

**Version:** 1.0
**Status:** Design (DO NOT EXECUTE)
**Date:** 2026-01-07
**Scope:** ALL OF MATHEMATICS - Every formal proof from every major system

---

## Executive Summary

**Mission:** Create a unified database containing every formally verified mathematical proof from all major proof assistants, stored in a common format that enables:
- Cross-system verification
- Proof translation between systems
- Redundant validation (same theorem proved multiple ways)
- AI training on the complete corpus of formal mathematics
- Sub-microsecond lookup and verification

**Target Systems:**

| System | Type Theory | Primary Use | Library Size |
|--------|-------------|-------------|--------------|
| **Lean 4** | CIC + Quotients | Modern math (Mathlib) | ~130K theorems |
| **Coq/Rocq** | CIC (pCIC) | Verification, math | ~50K+ theorems |
| **Isabelle/HOL** | Higher-Order Logic | Verification, AFP | ~200K+ theorems |
| **Agda** | Dependent types + Cubical | HoTT, programming | ~30K theorems |
| **HOL Light** | Simple HOL | Flyspeck, analysis | ~20K theorems |
| **HOL4** | Higher-Order Logic | Hardware verification | ~30K theorems |
| **Mizar** | First-order + types | Traditional math | ~60K theorems |
| **Metamath** | Substitution-based | Foundations | ~40K theorems |
| **PVS** | Higher-Order Logic | Verification | ~15K theorems |
| **ACL2** | First-order + induction | Hardware/software | ~20K theorems |

**Total Target:** ~600K+ unique theorems, many with multiple proofs across systems.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     UNIVERSAL PROOF DATABASE                                 │
│                        "All of Mathematics"                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   SOURCE SYSTEMS                                                             │
│   ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐              │
│   │ Lean 4 │ │  Coq   │ │ Isabelle │ │  Agda  │ │ HOL Light│  ...         │
│   │ .olean │ │  .vo   │ │  .thy    │ │ .agdai │ │  .ml     │              │
│   └───┬────┘ └───┬────┘ └────┬─────┘ └───┬────┘ └────┬─────┘              │
│       │          │           │           │           │                      │
│       ▼          ▼           ▼           ▼           ▼                      │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    FRONTEND PARSERS                              │      │
│   │  lean5-olean │ coq-vo │ isabelle-thy │ agda-iface │ hol-export  │      │
│   └─────────────────────────────┬───────────────────────────────────┘      │
│                                 │                                           │
│                                 ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │              UNIVERSAL PROOF IR (UPIR)                           │      │
│   │                                                                  │      │
│   │  • Common expression language (superset of all systems)          │      │
│   │  • Universe hierarchy (unified stratification)                   │      │
│   │  • Inductive types (common representation)                       │      │
│   │  • Proof terms (preserving source semantics)                     │      │
│   │  • Source annotations (provenance tracking)                      │      │
│   └─────────────────────────────┬───────────────────────────────────┘      │
│                                 │                                           │
│                    ┌────────────┼────────────┐                             │
│                    ▼            ▼            ▼                             │
│              ┌──────────┐ ┌──────────┐ ┌──────────┐                       │
│              │ STORAGE  │ │ SEARCH   │ │TRANSLATE │                       │
│              │ lean5db  │ │  INDEX   │ │  ENGINE  │                       │
│              └──────────┘ └──────────┘ └──────────┘                       │
│                    │            │            │                             │
│                    ▼            ▼            ▼                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    UNIFIED API                                   │      │
│   │  • JSON-RPC for AI agents                                        │      │
│   │  • REST for web interfaces                                       │      │
│   │  • Native Rust API for lean5 kernel                              │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Source System Analysis

### 2.1 Lean 4 (.olean)

**Type Theory:** Calculus of Inductive Constructions with Quotient Types
**Universe Hierarchy:** `Prop`, `Type u` (cumulative, universe polymorphism)

**File Format:**
```
[56-byte header]
  - magic: 4 bytes ("olean")
  - version: varies
  - git hash: 40 bytes
  - base address: 8 bytes
[compacted region]
  - pointer-serialized data
  - requires relocation on load
```

**Unique Features:**
- Quotient types (not in standard CIC)
- Universe polymorphism with `imax`
- Structures with inheritance
- String and Nat literals
- MData (metadata annotations)

**Parser Status:** ✅ Implemented (`lean5-olean`)

---

### 2.2 Coq/Rocq (.vo)

**Type Theory:** Predicative Calculus of Inductive Constructions (pCIC)
**Universe Hierarchy:** `Prop` (impredicative), `Set`, `Type@{i}` (predicative)

**File Format:**
```
[header]
  - magic: "Coq" or version-specific
  - library digest
[body]
  - OCaml Marshal format
  - Contains: library_info, summary, statements
```

**Unique Features:**
- Impredicative `Prop` (proofs can quantify over all props)
- Predicative `Set` (optional)
- SProp (strict propositions, proof irrelevance)
- Primitive projections
- Coinductive types
- Universe polymorphism (template, algebraic)
- Module system with functors

**Key Differences from Lean:**
- `Prop` impredicativity (Lean's `Prop` is predicative)
- No quotient types (must encode)
- Different reduction strategies (lazy vs eager)
- Algebraic universes (not just max/imax)

**Parser Status:** ❌ Needs implementation

---

### 2.3 Isabelle/HOL (.thy)

**Type Theory:** Higher-Order Logic (Church's Simple Type Theory)
**Universe:** Single type universe (no dependent types)

**File Format:**
```
[.thy files - source]
  - Isar proof language
  - Declarative proofs
[.hs / heap files]
  - Serialized ML heap
  - Contains: theories, proofs, ML code
```

**Unique Features:**
- No dependent types (fundamentally different!)
- Locale system (local contexts)
- Type classes
- Code generation
- Sledgehammer (external provers)
- Proof terms optional (often erased)

**Key Differences from CIC Systems:**
- NOT dependently typed
- Proofs by tactics, terms often not preserved
- Axiom of choice built-in
- Classical logic default

**Parser Status:** ❌ Needs implementation (complex - ML heap format)

---

### 2.4 Agda (.agdai)

**Type Theory:** Dependent types, optionally Cubical Type Theory
**Universe Hierarchy:** `Set₀`, `Set₁`, ..., `Setω`

**File Format:**
```
[.agdai interface files]
  - Haskell serialization format
  - Contains: signatures, definitions, constraints
```

**Unique Features:**
- Cubical type theory (paths, hcomp, transp)
- Sized types (for productivity)
- Instance arguments
- Reflection (metaprogramming)
- --without-K (restrict elimination)
- Rewriting rules

**Key Differences from Lean:**
- Cubical features (univalence is provable!)
- No quotients needed (cubical gives them)
- Different termination checking
- No tactics (proofs are terms)

**Parser Status:** ❌ Needs implementation

---

### 2.5 HOL Light (.ml)

**Type Theory:** Simple Higher-Order Logic
**Universe:** Single type universe

**File Format:**
```
[OCaml source files]
  - Proofs constructed via ML functions
  - LCF-style kernel
[Proof logs - optional]
  - Record of inference steps
```

**Unique Features:**
- Extremely simple kernel (~400 lines)
- Every proof step explicit
- Flyspeck project (Kepler conjecture)
- Formal verification of floating-point

**Parser Status:** ❌ Needs implementation (extract from OCaml)

---

### 2.6 HOL4 (.holsig, .holsml)

**Type Theory:** Higher-Order Logic
**Similar to:** HOL Light, Isabelle/HOL

**Parser Status:** ❌ Needs implementation

---

### 2.7 Mizar (.miz, .xml)

**Type Theory:** First-order logic with soft types
**Universe:** Set-theoretic (Tarski-Grothendieck)

**File Format:**
```
[.miz source files]
  - Human-readable proof language
[.xml exports]
  - Structured representation
  - MML (Mizar Mathematical Library)
```

**Unique Features:**
- Set-theoretic foundations
- Soft typing (types are predicates)
- Natural language-like syntax
- Extensive library (MML)

**Parser Status:** ❌ Needs implementation (XML export available)

---

### 2.8 Metamath (.mm)

**Type Theory:** Substitution-based (no built-in logic)
**Foundation:** User-defined (usually set theory)

**File Format:**
```
[.mm database files]
  - Plain text
  - Axioms, theorems, proofs
  - Each proof step: apply rule + substitution
```

**Unique Features:**
- Minimal metalogic
- Every proof step explicit
- Multiple axiom systems (ZFC, intuitionistic, etc.)
- Simple verifier (~300 lines)

**Parser Status:** ❌ Needs implementation (simple format)

---

### 2.9 PVS

**Type Theory:** Higher-Order Logic with subtypes
**Unique:** Predicate subtyping, dependent types

**Parser Status:** ❌ Needs implementation

---

### 2.10 ACL2

**Type Theory:** First-order with induction
**Foundation:** Quantifier-free, computational

**Parser Status:** ❌ Needs implementation

---

## 3. Universal Proof IR (UPIR)

### 3.1 Design Principles

1. **Superset:** UPIR can represent any term from any supported system
2. **Lossless:** Source semantics preserved (with annotations)
3. **Canonical:** Equivalent terms have canonical representation
4. **Extensible:** New systems can be added
5. **Verifiable:** Each representation is checkable

### 3.2 Expression Grammar

```rust
/// Universal Expression - superset of all systems
#[derive(Clone, Debug)]
pub enum UExpr {
    // ══════════════════════════════════════════════════════════════
    // COMMON CORE (all systems)
    // ══════════════════════════════════════════════════════════════

    /// De Bruijn variable
    Var(DeBruijnIndex),

    /// Sort/Universe
    Sort(USort),

    /// Named constant with universe arguments
    Const {
        name: UName,
        universes: Vec<ULevel>,
        source: SourceSystem,
    },

    /// Function application
    App(Box<UExpr>, Box<UExpr>),

    /// Lambda abstraction
    Lambda {
        binder: UBinder,
        domain: Box<UExpr>,
        body: Box<UExpr>,
    },

    /// Dependent function type (Pi/forall)
    Pi {
        binder: UBinder,
        domain: Box<UExpr>,
        codomain: Box<UExpr>,
    },

    /// Local definition
    Let {
        binder: UBinder,
        type_: Box<UExpr>,
        value: Box<UExpr>,
        body: Box<UExpr>,
    },

    // ══════════════════════════════════════════════════════════════
    // CIC SYSTEMS (Lean, Coq, Agda)
    // ══════════════════════════════════════════════════════════════

    /// Inductive type
    Ind {
        name: UName,
        universes: Vec<ULevel>,
        source: SourceSystem,
    },

    /// Constructor
    Construct {
        ind: UName,
        idx: u32,
        universes: Vec<ULevel>,
    },

    /// Pattern match / eliminator
    Match {
        scrutinee: Box<UExpr>,
        motive: Box<UExpr>,
        branches: Vec<UExpr>,
        source_style: MatchStyle,
    },

    /// Fixpoint (recursive definition)
    Fix {
        funcs: Vec<UFixFunc>,
        focus: u32,
    },

    /// Cofixpoint (corecursive)
    CoFix {
        funcs: Vec<UFixFunc>,
        focus: u32,
    },

    /// Projection from structure/record
    Proj {
        struct_name: UName,
        field_idx: u32,
        expr: Box<UExpr>,
    },

    // ══════════════════════════════════════════════════════════════
    // LEAN-SPECIFIC
    // ══════════════════════════════════════════════════════════════

    /// Lean quotient type operations
    LeanQuot(LeanQuotOp),

    /// Lean literals (Nat, String)
    LeanLit(LeanLiteral),

    /// Lean metadata wrapper
    LeanMData {
        data: LeanMDataEntries,
        expr: Box<UExpr>,
    },

    // ══════════════════════════════════════════════════════════════
    // COQ-SPECIFIC
    // ══════════════════════════════════════════════════════════════

    /// Coq's impredicative Prop
    CoqProp,

    /// Coq's Set (when predicative)
    CoqSet,

    /// Coq's SProp (strict proposition)
    CoqSProp,

    /// Coq primitive integer operations
    CoqPrimInt(CoqPrimIntOp),

    /// Coq primitive float operations
    CoqPrimFloat(CoqPrimFloatOp),

    /// Coq primitive array
    CoqPrimArray {
        elem_type: Box<UExpr>,
        elements: Vec<UExpr>,
    },

    // ══════════════════════════════════════════════════════════════
    // AGDA-SPECIFIC (Cubical)
    // ══════════════════════════════════════════════════════════════

    /// Interval type (I)
    AgdaInterval,

    /// Interval endpoints
    AgdaIZero,
    AgdaIOne,

    /// Path type
    AgdaPathP {
        type_: Box<UExpr>,  // I → Type
        left: Box<UExpr>,
        right: Box<UExpr>,
    },

    /// Path lambda
    AgdaPathLam(Box<UExpr>),

    /// Path application
    AgdaPathApp(Box<UExpr>, Box<UExpr>),

    /// Homogeneous composition
    AgdaHComp {
        type_: Box<UExpr>,
        phi: Box<UExpr>,
        u: Box<UExpr>,
        base: Box<UExpr>,
    },

    /// Transport
    AgdaTransp {
        type_: Box<UExpr>,
        phi: Box<UExpr>,
        base: Box<UExpr>,
    },

    /// Glue types (for univalence)
    AgdaGlue {
        base: Box<UExpr>,
        phi: Box<UExpr>,
        types: Box<UExpr>,
        equivs: Box<UExpr>,
    },

    // ══════════════════════════════════════════════════════════════
    // HOL SYSTEMS (Isabelle, HOL Light, HOL4)
    // ══════════════════════════════════════════════════════════════

    /// Simple type (non-dependent)
    HolType(HolSimpleType),

    /// HOL constant (with type instantiation, not universe)
    HolConst {
        name: UName,
        type_args: Vec<HolSimpleType>,
    },

    /// HOL abstraction (non-dependent)
    HolAbs {
        var_type: HolSimpleType,
        body: Box<UExpr>,
    },

    /// HOL type variable (for polymorphism)
    HolTyVar(String),

    // ══════════════════════════════════════════════════════════════
    // MIZAR-SPECIFIC
    // ══════════════════════════════════════════════════════════════

    /// Mizar set-theoretic term
    MizarSet(MizarSetTerm),

    /// Mizar type (as predicate)
    MizarType(MizarTypeTerm),

    /// Mizar "it" (the object being defined)
    MizarIt,

    // ══════════════════════════════════════════════════════════════
    // METAMATH-SPECIFIC
    // ══════════════════════════════════════════════════════════════

    /// Metamath expression (string of symbols)
    MetamathExpr {
        symbols: Vec<String>,
        substitution: MetamathSubst,
    },

    // ══════════════════════════════════════════════════════════════
    // ANNOTATIONS
    // ══════════════════════════════════════════════════════════════

    /// Source location annotation
    SourceLoc {
        expr: Box<UExpr>,
        file: String,
        line: u32,
        col: u32,
    },

    /// Type annotation (for bidirectional typing)
    Annot {
        expr: Box<UExpr>,
        type_: Box<UExpr>,
    },

    /// Hole / metavariable (for incomplete terms)
    Hole {
        id: u64,
        type_: Option<Box<UExpr>>,
    },
}
```

### 3.3 Universal Sort Hierarchy

```rust
/// Universal sort (universe) representation
#[derive(Clone, Debug)]
pub enum USort {
    // ══════════════════════════════════════════════════════════════
    // CIC-STYLE (Lean, Coq, Agda)
    // ══════════════════════════════════════════════════════════════

    /// Prop (impredicative or predicative depending on source)
    Prop {
        impredicative: bool,  // true for Coq, false for Lean
        proof_irrelevant: bool,
    },

    /// Type at level
    Type(ULevel),

    // ══════════════════════════════════════════════════════════════
    // COQ-SPECIFIC
    // ══════════════════════════════════════════════════════════════

    /// Coq's Set (predicative Type 0 in some modes)
    CoqSet,

    /// Coq's SProp (definitionally proof-irrelevant)
    CoqSProp,

    // ══════════════════════════════════════════════════════════════
    // AGDA-SPECIFIC
    // ══════════════════════════════════════════════════════════════

    /// Agda's Setω (above all finite universes)
    AgdaSetOmega(u32),  // Setω, Setω₁, etc.

    /// Agda's SizeUniv
    AgdaSizeUniv,

    // ══════════════════════════════════════════════════════════════
    // HOL-STYLE (Isabelle, HOL Light)
    // ══════════════════════════════════════════════════════════════

    /// HOL has no universe hierarchy (single Bool type for props)
    HolBool,

    /// HOL type universe (implicit, no levels)
    HolType,
}

/// Universal universe level
#[derive(Clone, Debug)]
pub enum ULevel {
    /// Universe 0
    Zero,

    /// Successor universe
    Succ(Box<ULevel>),

    /// Maximum of two universes
    Max(Box<ULevel>, Box<ULevel>),

    /// Impredicative maximum (Lean-specific)
    /// imax u v = 0 if v = 0, max u v otherwise
    IMax(Box<ULevel>, Box<ULevel>),

    /// Universe variable (polymorphism)
    Var(UniverseVar),

    /// Coq algebraic universe expression
    CoqAlgebraic(CoqUnivExpr),

    /// Agda universe level (can be ω + n)
    AgdaLevel(AgdaLevelExpr),
}
```

### 3.4 Source System Tracking

```rust
/// Source system for provenance tracking
#[derive(Clone, Debug)]
pub enum SourceSystem {
    Lean4 { version: Version, toolchain: String },
    Coq { version: Version },
    Isabelle { version: String, logic: String },  // e.g., "HOL", "ZF"
    Agda { version: Version, flags: AgdaFlags },
    HolLight { version: String },
    Hol4 { version: String },
    Mizar { version: String, mml_version: String },
    Metamath { database: String },  // e.g., "set.mm", "iset.mm"
    Pvs { version: String },
    Acl2 { version: String },

    /// Translated from another system
    Translated {
        from: Box<SourceSystem>,
        to: TargetSystem,
        translator_version: Version,
        translation_date: DateTime,
        equivalence_proof: Option<ProofId>,
    },

    /// Multiple sources (same theorem proved in multiple systems)
    Multiple(Vec<SourceSystem>),
}

/// What Agda flags were used
#[derive(Clone, Debug)]
pub struct AgdaFlags {
    pub cubical: bool,
    pub without_k: bool,
    pub safe: bool,
    pub sized_types: bool,
}
```

---

## 4. Multi-Proof Storage

### 4.1 Same Theorem, Multiple Proofs

The UPD explicitly supports and encourages multiple proofs of the same theorem:

```rust
/// A mathematical statement with potentially multiple proofs
pub struct UniversalTheorem {
    /// Canonical name
    pub name: UName,

    /// Aliases in different systems
    pub aliases: HashMap<SourceSystem, UName>,
    // e.g., Nat.add_comm (Lean) = Nat.add_comm (Coq) = add_comm (Isabelle)

    /// The statement (in UPIR)
    pub statement: UExpr,

    /// Multiple proofs from different sources
    pub proofs: Vec<ProofVariant>,

    /// Cross-references (which proofs verify equivalently)
    pub equivalences: Vec<ProofEquivalence>,

    /// Human-readable description
    pub description: Option<String>,

    /// Mathematical classification (MSC 2020)
    pub msc_codes: Vec<String>,
}

/// One proof of a theorem
pub struct ProofVariant {
    /// Unique ID for this proof
    pub id: ProofId,

    /// Source system
    pub source: SourceSystem,

    /// Original name in source system
    pub original_name: UName,

    /// The proof term (in UPIR)
    pub proof: UExpr,

    /// Type of proof
    pub proof_style: ProofStyle,

    /// Verification status
    pub verified: VerificationStatus,

    /// Dependencies (other theorems used)
    pub dependencies: Vec<ProofId>,

    /// Metrics
    pub metrics: ProofMetrics,
}

pub enum ProofStyle {
    /// Full term (Lean, Coq, Agda)
    TermProof,

    /// Tactic script (Coq Ltac, Lean tactics)
    TacticScript { tactics: Vec<String> },

    /// Declarative (Isabelle Isar, Mizar)
    Declarative { steps: Vec<String> },

    /// Step-by-step (Metamath)
    Substitution { steps: Vec<MetamathStep> },

    /// External (Sledgehammer, SMT)
    External { oracle: String, certificate: Vec<u8> },

    /// No proof term preserved
    Axiom,
}

pub struct ProofMetrics {
    /// Size of proof term (nodes in AST)
    pub term_size: u64,

    /// Depth of proof term
    pub depth: u32,

    /// Number of lemmas used
    pub lemma_count: u32,

    /// Estimated complexity (lines if written out)
    pub complexity_estimate: u32,
}

/// Evidence that two proofs prove the same thing
pub struct ProofEquivalence {
    pub proof1: ProofId,
    pub proof2: ProofId,

    /// How equivalence was established
    pub evidence: EquivalenceEvidence,
}

pub enum EquivalenceEvidence {
    /// Types are definitionally equal in UPIR
    TypeEquality,

    /// Manual verification that statements match
    ManualReview { reviewer: String, date: DateTime },

    /// Automated matching (statement normalization)
    AutomatedMatch { algorithm: String, confidence: f64 },

    /// Formal proof of equivalence
    FormalProof { proof_id: ProofId },
}
```

### 4.2 Example: Commutativity of Addition

```yaml
theorem: "nat_add_comm"
statement: "∀ (n m : ℕ), n + m = m + n"

proofs:
  - id: "lean4:Nat.add_comm"
    source: Lean4 (Mathlib)
    style: TermProof
    verified: Kernel
    size: 847 nodes

  - id: "coq:Nat.add_comm"
    source: Coq (Stdlib)
    style: TermProof
    verified: Kernel
    size: 923 nodes

  - id: "isabelle:Nat.add_commute"
    source: Isabelle/HOL (Main)
    style: Declarative (Isar)
    verified: Kernel
    size: 12 steps

  - id: "agda:+-comm"
    source: Agda (stdlib)
    style: TermProof
    verified: Kernel
    size: 234 nodes

  - id: "metamath:addcom"
    source: Metamath (set.mm)
    style: Substitution
    verified: Metamath
    size: 89 steps

  - id: "mizar:NAT_1:4"
    source: Mizar (MML)
    style: Declarative
    verified: Mizar
    size: 15 steps

  - id: "hollight:ADD_SYM"
    source: HOL Light
    style: TermProof
    verified: Kernel
    size: 156 nodes

equivalences:
  - [lean4:Nat.add_comm, coq:Nat.add_comm, evidence: TypeEquality]
  - [lean4:Nat.add_comm, isabelle:Nat.add_commute, evidence: AutomatedMatch]
  - [metamath:addcom, coq:Nat.add_comm, evidence: FormalProof]
```

---

## 5. Human-Readable Export (Markdown)

### 5.1 Example Record

```markdown
# ℕ.add_comm

> **For all natural numbers n and m, addition is commutative: n + m = m + n**

## Statement

```
∀ (n m : ℕ), n + m = m + n
```

### In LaTeX
$$\forall (n\, m : \mathbb{N}),\; n + m = m + n$$

### In Natural Language
For any two natural numbers n and m, the sum n plus m equals the sum m plus n.
This is the commutative property of addition on natural numbers.

---

## Classification

| Field | Value |
|-------|-------|
| **MSC 2020** | 11B13 (Additive bases) |
| **Category** | Elementary number theory |
| **Difficulty** | Trivial (undergraduate) |
| **First proved** | Ancient (Euclid, informally) |

---

## Proofs (7 variants)

### Proof 1: Lean 4 (Mathlib)

**Source:** `Mathlib.Data.Nat.Basic`
**Style:** Term proof (induction)
**Size:** 847 nodes
**Verified:** ✅ Lean 4 kernel

<details>
<summary>View proof term</summary>

```lean
theorem Nat.add_comm (n m : Nat) : n + m = m + n := by
  induction n with
  | zero => simp [Nat.zero_add, Nat.add_zero]
  | succ n ih => simp [Nat.succ_add, Nat.add_succ, ih]
```

</details>

**Dependencies:**
- `Nat.zero_add`
- `Nat.add_zero`
- `Nat.succ_add`
- `Nat.add_succ`

---

### Proof 2: Coq (Stdlib)

**Source:** `Arith.PeanoNat`
**Style:** Term proof
**Size:** 923 nodes
**Verified:** ✅ Coq kernel

<details>
<summary>View proof term</summary>

```coq
Theorem add_comm : forall n m : nat, n + m = m + n.
Proof.
  intros n m. induction n as [| n' IHn'].
  - simpl. rewrite add_0_r. reflexivity.
  - simpl. rewrite IHn'. rewrite add_succ_r. reflexivity.
Qed.
```

</details>

---

### Proof 3: Isabelle/HOL

**Source:** `HOL.Nat`
**Style:** Declarative (Isar)
**Size:** 12 steps
**Verified:** ✅ Isabelle kernel

<details>
<summary>View Isar proof</summary>

```isabelle
lemma add_commute: "m + n = n + (m::nat)"
proof (induct m)
  case 0
  then show ?case by simp
next
  case (Suc m)
  then show ?case by simp
qed
```

</details>

---

### Proof 4: Agda (Standard Library)

**Source:** `Data.Nat.Properties`
**Style:** Term proof (pattern matching)
**Size:** 234 nodes
**Verified:** ✅ Agda type checker

<details>
<summary>View proof</summary>

```agda
+-comm : ∀ m n → m + n ≡ n + m
+-comm zero    n = sym (+-identityʳ n)
+-comm (suc m) n = begin
  suc m + n   ≡⟨⟩
  suc (m + n) ≡⟨ cong suc (+-comm m n) ⟩
  suc (n + m) ≡⟨ sym (+-suc n m) ⟩
  n + suc m   ∎
```

</details>

---

### Proof 5: Metamath (set.mm)

**Source:** `set.mm`
**Style:** Substitution steps
**Size:** 89 steps
**Verified:** ✅ Metamath verifier

<details>
<summary>View proof</summary>

```metamath
addcom $p |- ( A + B ) = ( B + A ) $=
  ( cc0 caddc wceq cc cn wcel wa oveq1 oveq2 addid1 eqtr3d ... )
  ABCDEFGHIJKLMNOPQRSTUVWXYZAA $.
```

</details>

---

### Proof 6: Mizar (MML)

**Source:** `NAT_1:4`
**Style:** Declarative
**Size:** 15 steps
**Verified:** ✅ Mizar checker

---

### Proof 7: HOL Light

**Source:** `arithmetic.ml`
**Style:** Term proof
**Size:** 156 nodes
**Verified:** ✅ HOL Light kernel

---

## Cross-References

### Equivalences Verified
- ✅ Lean ↔ Coq (type equality in UPIR)
- ✅ Lean ↔ Isabelle (automated matching, 99.9% confidence)
- ✅ Metamath ↔ Coq (formal proof of equivalence)
- ⚠️ Agda ↔ Lean (needs manual review - Agda uses different equality)

### Used By
| System | Count | Examples |
|--------|-------|----------|
| Lean | 1,247 | `Int.add_comm`, `Rat.add_comm`, `Ring.add_comm`, ... |
| Coq | 892 | `Z.add_comm`, `Q.add_comm`, ... |
| Isabelle | 2,103 | `int_add_commute`, `rat_add_commute`, ... |

### Uses
| Theorem | System | Count |
|---------|--------|-------|
| `Nat.succ_add` | All | 7 |
| `Nat.add_succ` | All | 7 |
| `Nat.zero_add` | All | 7 |
| `Nat.add_zero` | All | 7 |
| `Eq.refl` | All | 7 |

---

## Metadata

| Field | Value |
|-------|-------|
| **UPD ID** | `upd:nat:add_comm:v1` |
| **Content Hash** | `blake3:a7f3b2c1d4e5...` |
| **Created** | 2026-01-07 |
| **Last Updated** | 2026-01-07 |
| **Proof Count** | 7 |
| **Total Size** | 3,446 nodes |

---

## Version History

| Date | Change |
|------|--------|
| 2026-01-07 | Initial import from Lean 4, Coq, Isabelle |
| 2026-01-07 | Added Agda, Metamath, Mizar proofs |
| 2026-01-07 | Added HOL Light proof |
| 2026-01-07 | Verified cross-system equivalences |
```

---

## 6. Parser Implementation Status

| System | File Format | Parser Status | Effort Estimate |
|--------|-------------|---------------|-----------------|
| **Lean 4** | .olean | ✅ Done | - |
| **Coq** | .vo (OCaml marshal) | ❌ Needed | 4 weeks |
| **Isabelle** | .thy + heap | ❌ Needed | 6 weeks (complex) |
| **Agda** | .agdai | ❌ Needed | 3 weeks |
| **HOL Light** | .ml (extract) | ❌ Needed | 3 weeks |
| **HOL4** | .holsig | ❌ Needed | 2 weeks |
| **Mizar** | .miz, .xml | ❌ Needed | 3 weeks (XML available) |
| **Metamath** | .mm | ❌ Needed | 1 week (simple format) |
| **PVS** | varies | ❌ Needed | 3 weeks |
| **ACL2** | .lisp | ❌ Needed | 2 weeks |

**Total new parser work: ~27 weeks**

---

## 7. Identified Gaps and Flaws

### Gap 1: HOL Systems Lack Dependent Types

**Problem:** Isabelle/HOL, HOL Light, and HOL4 use Simple Type Theory without dependent types. We cannot directly translate HOL proofs to CIC systems.

**Solution:** Shallow embedding with explicit type constraints:

```rust
/// HOL term embedded in UPIR
/// We encode `∀x:α. P(x)` as `∀(x : HolTerm α). P(x)`
pub struct HolEmbedding {
    /// The HOL term/type
    pub hol_term: HolTerm,

    /// Explicit well-formedness constraint
    pub constraint: UExpr,

    /// Flag that this is HOL (not full CIC)
    pub is_hol_only: bool,
}

// HOL: `∀x. x + 0 = x`
// becomes UPIR: `∀(x : HolNat), hol_eq (hol_add x hol_zero) x`
```

---

### Gap 2: Classical vs Constructive Logic

**Problem:** Systems have different logical foundations:
- Lean/Coq/Agda: Constructive (can add classical axioms)
- Isabelle/HOL: Classical by default
- Mizar: Classical (excluded middle)
- Metamath: User-defined (can be either)

A classical proof may not translate to a constructive system.

**Solution:** Logic annotation and separation:

```rust
pub enum LogicMode {
    /// Constructive (no excluded middle, no choice)
    Constructive,

    /// Classical (excluded middle)
    Classical,

    /// Uses axiom of choice
    Choice,

    /// Uses both
    ClassicalWithChoice,

    /// Unknown/mixed
    Unknown,
}

pub struct ProofVariant {
    // ... existing fields ...

    /// What logic mode this proof requires
    pub logic_mode: LogicMode,

    /// Can this proof be made constructive?
    pub constructivizable: Option<bool>,
}
```

---

### Gap 3: Different Equality Types

**Problem:** Systems have different notions of equality:
- Lean: Inductive `Eq`, propositional
- Coq: Inductive `eq`, propositional (or SProp)
- Agda (Cubical): Path types (definitionally UIP-free)
- HOL: Primitive equality `=`
- Mizar: Set-theoretic equality

**Solution:** Track equality type in UPIR:

```rust
pub enum EqualityType {
    /// Inductive equality (Lean/Coq)
    Inductive,

    /// Path equality (Cubical Agda)
    Path,

    /// Primitive HOL equality
    HolPrimitive,

    /// Set-theoretic (Mizar)
    SetTheoretic,

    /// Definitional (judgmental)
    Definitional,
}
```

---

### Gap 4: Universe Inconsistencies

**Problem:** Universe handling differs significantly:
- Lean: `Prop`, `Type u`, cumulative, `imax`
- Coq: `Prop` (impredicative), `Set`, `Type@{i}`, algebraic
- Agda: `Set₀`, `Set₁`, ..., `Setω`, potentially non-cumulative
- HOL: No universes

**Solution:** Universe translation layer:

```rust
/// Universe normalization for cross-system comparison
pub fn normalize_universe(level: &ULevel, source: SourceSystem) -> NormalizedLevel {
    match source {
        SourceSystem::Lean4 { .. } => {
            // Lean universes map directly
            lean_to_normalized(level)
        }
        SourceSystem::Coq { .. } => {
            // Coq algebraic universes need solving
            coq_to_normalized(level)
        }
        SourceSystem::Agda { .. } => {
            // Agda Setω maps to "large"
            agda_to_normalized(level)
        }
        SourceSystem::Isabelle { .. } | SourceSystem::HolLight { .. } => {
            // HOL has no universes - use implicit
            NormalizedLevel::Implicit
        }
        // ...
    }
}
```

---

### Gap 5: Proof Irrelevance Differences

**Problem:**
- Lean `Prop`: proof-relevant (can pattern match)
- Coq `Prop`: proof-irrelevant (cannot match on proofs to produce data)
- Coq `SProp`: definitionally proof-irrelevant
- Agda: optional proof irrelevance

**Solution:** Track proof relevance:

```rust
pub enum ProofRelevance {
    /// Proofs are distinct terms
    Relevant,

    /// Proofs are propositionally equal
    Irrelevant,

    /// Proofs are definitionally equal (Coq SProp)
    DefinitionallyIrrelevant,
}
```

---

### Gap 6: Coinduction Differences

**Problem:** Coinductive types work differently:
- Coq: CoFixpoint with guardedness checking
- Agda: Coinduction with sized types
- Lean: No built-in coinduction (must encode)
- HOL: No coinduction

**Solution:** Unified coinduction representation:

```rust
pub struct CoinductiveSpec {
    pub style: CoinductionStyle,
    pub productivity_evidence: Option<ProductivityProof>,
}

pub enum CoinductionStyle {
    CoqCofix { guard_index: u32 },
    AgdaSized { size_arg: UExpr },
    Encoded { encoding: CoinductionEncoding },
}
```

---

### Gap 7: Module/Namespace Differences

**Problem:** Module systems are wildly different:
- Lean: Namespaces, open, export
- Coq: Modules with functors, sections
- Isabelle: Locales, theories
- Agda: Modules with parameters
- HOL: Theories
- Mizar: Articles

**Solution:** Flatten to fully qualified names:

```rust
pub struct UName {
    /// Fully qualified name
    pub full_path: Vec<String>,

    /// Original module structure
    pub original_module: ModuleRef,

    /// Source system
    pub source: SourceSystem,
}

// Lean: Mathlib.Data.Nat.Basic.Nat.add_comm
// Coq:  Arith.PeanoNat.Nat.add_comm
// Isabelle: HOL.Nat.add_commute
// All become: nat.add_comm (canonical)
```

---

### Gap 8: Automation/Tactic Opacity

**Problem:** Many proofs use automation that hides the proof term:
- Isabelle's Sledgehammer (calls external SMT solvers)
- Coq's `omega`, `lia`, `ring`
- Lean's `decide`, `native_decide`

The actual proof term may be huge or unavailable.

**Solution:** Certificate storage:

```rust
pub enum ProofCertificate {
    /// Full proof term available
    FullTerm(UExpr),

    /// Proof term available but large
    LargeTerm {
        size: u64,
        hash: [u8; 32],
        storage: StorageRef,
    },

    /// External certificate (SMT, etc.)
    External {
        oracle: String,
        certificate: Vec<u8>,
        verifier: String,
    },

    /// Proof not preserved (trust the system)
    Trusted {
        system: SourceSystem,
        verification_date: DateTime,
    },
}
```

---

### Gap 9: No Unified Search Across Systems

**Problem:** Current search design assumes single system. Need cross-system search.

**Solution:** Federated search with canonical statements:

```rust
/// Search across all systems
pub struct UnifiedSearch {
    /// Per-system indexes
    indexes: HashMap<SourceSystem, SearchIndex>,

    /// Canonical statement index (normalized)
    canonical_index: CanonicalStatementIndex,

    /// Cross-reference index
    equivalence_index: EquivalenceIndex,
}

impl UnifiedSearch {
    /// Find all proofs of a statement (any system)
    pub fn find_proofs(&self, statement: &UExpr) -> Vec<ProofVariant> {
        let canonical = self.canonicalize(statement);

        // Search canonical index
        let matches = self.canonical_index.search(&canonical);

        // Expand via equivalences
        let expanded = self.equivalence_index.expand(&matches);

        // Return all variants
        expanded.into_iter()
            .flat_map(|id| self.get_all_proofs(id))
            .collect()
    }
}
```

---

### Gap 10: Version Compatibility Matrix

**Problem:** Different versions of each system may have incompatible features.

**Solution:** Explicit compatibility tracking:

```rust
pub struct CompatibilityMatrix {
    /// Which system versions are supported
    pub supported_versions: HashMap<SystemType, VersionRange>,

    /// Known breaking changes
    pub breaking_changes: Vec<BreakingChange>,

    /// Feature availability by version
    pub features: HashMap<Feature, HashMap<SystemType, VersionRange>>,
}

pub struct BreakingChange {
    pub system: SystemType,
    pub version: Version,
    pub description: String,
    pub migration: Option<MigrationGuide>,
}
```

---

## 8. Implementation Phases

### Phase 0: Foundation (4 weeks)
- [ ] UPIR specification finalization
- [ ] Core data structures
- [ ] Basic serialization

### Phase 1: Lean 4 Complete (2 weeks)
- [ ] Adapt existing lean5-olean
- [ ] Export to UPIR
- [ ] Full Mathlib conversion

### Phase 2: Coq Parser (4 weeks)
- [ ] OCaml marshal format parser
- [ ] Coq term → UPIR conversion
- [ ] Universe handling
- [ ] Coq stdlib import

### Phase 3: Metamath Parser (1 week)
- [ ] .mm file parser (simple format)
- [ ] set.mm import
- [ ] Substitution proof → UPIR

### Phase 4: Agda Parser (3 weeks)
- [ ] .agdai interface parser
- [ ] Cubical features
- [ ] Standard library import

### Phase 5: HOL Systems (6 weeks)
- [ ] HOL Light proof extraction
- [ ] Isabelle export mechanism
- [ ] HOL4 parser
- [ ] HOL → UPIR embedding

### Phase 6: Mizar Parser (3 weeks)
- [ ] XML export parsing
- [ ] MML import
- [ ] Set-theoretic → UPIR

### Phase 7: Cross-System Features (4 weeks)
- [ ] Equivalence detection
- [ ] Unified search
- [ ] Translation engine
- [ ] Multi-proof storage

### Phase 8: Integration & Polish (4 weeks)
- [ ] API finalization
- [ ] Documentation
- [ ] Performance optimization
- [ ] Testing against full libraries

**Total: ~31 weeks**

---

## 9. Scale Estimates

| Library | System | Theorems | Expressions | Source Size | UPD Size |
|---------|--------|----------|-------------|-------------|----------|
| Mathlib | Lean 4 | ~130K | ~100M | ~4 GB | ~800 MB |
| Stdlib | Coq | ~15K | ~10M | ~200 MB | ~80 MB |
| MathComp | Coq | ~20K | ~20M | ~400 MB | ~120 MB |
| CoqHoTT | Coq | ~8K | ~15M | ~150 MB | ~60 MB |
| AFP | Isabelle | ~200K | ~50M | ~2 GB | ~400 MB |
| Stdlib | Agda | ~10K | ~8M | ~100 MB | ~40 MB |
| Cubical | Agda | ~5K | ~10M | ~80 MB | ~35 MB |
| set.mm | Metamath | ~40K | ~5M | ~50 MB | ~30 MB |
| MML | Mizar | ~60K | ~20M | ~300 MB | ~100 MB |
| HOL Light | - | ~20K | ~5M | ~50 MB | ~30 MB |
| HOL4 | - | ~30K | ~8M | ~80 MB | ~40 MB |
| **TOTAL** | | **~538K** | **~251M** | **~7.4 GB** | **~1.7 GB** |

With proof redundancy (same theorem, multiple systems): **~1M proof variants**

---

## 10. Success Criteria

1. **Coverage:** Import >95% of theorems from each supported system
2. **Correctness:** All imported proofs verify in source system
3. **Equivalence:** Detect >80% of cross-system equivalences automatically
4. **Performance:** <100ms lookup for any theorem
5. **Size:** Total database <2GB compressed
6. **Search:** Find related theorems across systems in <1s

---

## Appendix: File Format Summary

| System | Primary Format | Binary? | Documented? | Complexity |
|--------|---------------|---------|-------------|------------|
| Lean 4 | .olean | Yes | Partial | Medium |
| Coq | .vo | Yes (OCaml) | No | High |
| Isabelle | .thy + heap | Mixed | Partial | Very High |
| Agda | .agdai | Yes (Haskell) | No | High |
| HOL Light | .ml | No (source) | N/A | Medium |
| HOL4 | .holsig | Yes | Partial | Medium |
| Mizar | .miz + .xml | Text + XML | Yes | Medium |
| Metamath | .mm | Text | Yes | Low |
| PVS | varies | Mixed | Partial | Medium |
| ACL2 | .lisp | Text | Yes | Low |
