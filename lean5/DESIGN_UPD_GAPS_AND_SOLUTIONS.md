# Universal Proof Database: Additional Gaps and Solutions

**Version:** 1.0
**Date:** 2026-01-07
**Purpose:** Deep analysis of gaps in multi-system proof unification

---

## Gaps 11-25: Additional Critical Issues

### Gap 11: Axiom Incompatibilities

**Problem:** Different systems assume different axioms:

| System | Axiom of Choice | LEM | Prop Extensionality | Func Extensionality | Univalence |
|--------|-----------------|-----|---------------------|---------------------|------------|
| Lean 4 | Optional (`Classical`) | Optional | ✅ Built-in | ✅ Via `funext` | ❌ |
| Coq | Optional | Optional | Optional | Optional | ❌ |
| Agda | Optional | Optional | Optional | Optional | ✅ (Cubical) |
| Isabelle | ✅ Built-in | ✅ Built-in | ✅ | ✅ | ❌ |
| HOL Light | ✅ Built-in | ✅ Built-in | ✅ | ✅ | ❌ |
| Mizar | ✅ Built-in | ✅ Built-in | ✅ | ✅ | ❌ |

A proof using Choice in Isabelle cannot be translated to constructive Coq.

**Solution: Axiom Dependency Tracking**

```rust
/// Axioms used by a proof
#[derive(Clone, Debug, Default)]
pub struct AxiomUsage {
    pub choice: bool,
    pub excluded_middle: bool,
    pub prop_ext: bool,
    pub func_ext: bool,
    pub univalence: bool,
    pub quotients: bool,
    pub large_elimination: bool,

    /// Custom axioms (system-specific)
    pub custom: Vec<CustomAxiom>,
}

impl AxiomUsage {
    /// Can this proof be used in target system?
    pub fn compatible_with(&self, target: &SystemCapabilities) -> bool {
        (!self.choice || target.has_choice) &&
        (!self.excluded_middle || target.has_lem) &&
        (!self.prop_ext || target.has_prop_ext) &&
        (!self.func_ext || target.has_func_ext) &&
        (!self.univalence || target.has_univalence) &&
        (!self.quotients || target.has_quotients) &&
        (!self.large_elimination || target.has_large_elim)
    }

    /// What axioms would need to be added to make this work?
    pub fn missing_for(&self, target: &SystemCapabilities) -> Vec<RequiredAxiom> {
        let mut missing = vec![];
        if self.choice && !target.has_choice {
            missing.push(RequiredAxiom::Choice);
        }
        // ... etc
        missing
    }
}

/// Compute axiom usage for a proof term
pub fn analyze_axioms(proof: &UExpr, env: &Environment) -> AxiomUsage {
    let mut usage = AxiomUsage::default();

    visit_expr(proof, |expr| {
        if let UExpr::Const { name, .. } = expr {
            if env.is_choice_axiom(name) {
                usage.choice = true;
            }
            if env.is_lem_axiom(name) {
                usage.excluded_middle = true;
            }
            // ... detect other axioms
        }
    });

    usage
}
```

---

### Gap 12: Termination/Productivity Checking Differences

**Problem:** Systems have different termination checkers:

- **Lean:** Structural recursion + well-founded recursion
- **Coq:** Guard condition (syntactic) + Program (well-founded)
- **Agda:** Sized types + structural
- **Isabelle:** Functions package (automation)
- **HOL:** Must prove termination separately

A function that passes Lean's termination checker might fail Coq's guard condition.

**Solution: Termination Evidence Abstraction**

```rust
/// How termination/productivity is established
pub enum TerminationEvidence {
    /// Structurally recursive on argument
    Structural {
        decreasing_arg: u32,
        inductive_type: UName,
    },

    /// Well-founded recursion with measure
    WellFounded {
        measure: UExpr,
        relation: UExpr,
        proof: Box<UExpr>,
    },

    /// Sized types (Agda)
    Sized {
        size_arg: u32,
        size_type: UExpr,
    },

    /// Coq guard condition (syntactic)
    CoqGuard {
        guard_indices: Vec<u32>,
    },

    /// External proof (Isabelle function package)
    ExternalProof {
        system: SourceSystem,
        evidence_id: String,
    },

    /// Axiomatized (partial function)
    Axiomatized,
}

/// Translate termination evidence between systems
pub fn translate_termination(
    evidence: &TerminationEvidence,
    from: SourceSystem,
    to: SourceSystem,
) -> Result<TerminationEvidence, TerminationError> {
    match (evidence, from, to) {
        // Structural is usually universal
        (TerminationEvidence::Structural { .. }, _, _) => {
            Ok(evidence.clone())
        }

        // Well-founded can usually translate
        (TerminationEvidence::WellFounded { .. }, _, _) => {
            // Need to translate the measure and proof
            translate_well_founded(evidence, from, to)
        }

        // Sized types are Agda-specific
        (TerminationEvidence::Sized { .. }, SourceSystem::Agda { .. }, to) => {
            // Must convert to structural or well-founded
            sized_to_structural_or_wf(evidence, to)
        }

        _ => Err(TerminationError::CannotTranslate {
            evidence: evidence.clone(),
            from,
            to,
        })
    }
}
```

---

### Gap 13: Implicit Arguments / Elaboration Differences

**Problem:** Systems have different implicit argument inference:

```lean
-- Lean: instance arguments with []
def map [Functor f] (g : α → β) (x : f α) : f β := ...

-- Coq: type class with {}
Definition map {f : Type -> Type} `{Functor f} {α β : Type}
  (g : α -> β) (x : f α) : f β := ...

-- Agda: instance arguments with ⦃⦄
map : ⦃ Functor f ⦄ → (α → β) → f α → f β
```

The elaborated terms differ significantly.

**Solution: Normalize to Explicit + Annotations**

```rust
/// Binder information that captures all systems
pub struct UBinder {
    /// Name (for readability, not semantics)
    pub name: Option<String>,

    /// Binding mode in source
    pub mode: BinderMode,

    /// Was this inferred or explicit in source?
    pub origin: BinderOrigin,
}

pub enum BinderMode {
    /// Explicit: (x : A)
    Explicit,

    /// Implicit, inferred from usage: {x : A}
    Implicit,

    /// Implicit, inferred from scope: {{x : A}}
    StrictImplicit,

    /// Instance argument: [x : A]
    Instance,

    /// Agda's dotted pattern: .x
    Irrelevant,
}

/// In UPIR, all arguments are explicit, but we track original mode
pub fn elaborate_to_upir(term: SourceTerm) -> UExpr {
    // Fill in all implicit arguments explicitly
    // But annotate with original BinderMode for round-trip
}
```

---

### Gap 14: Notation and Pretty Printing

**Problem:** The same mathematical concept has different notation:

| Concept | Lean | Coq | Isabelle | Agda | Metamath |
|---------|------|-----|----------|------|----------|
| Addition | `a + b` | `a + b` | `a + b` | `a + b` | `( a + b )` |
| Function type | `A → B` | `A -> B` | `A ⇒ B` | `A → B` | `( A -> B )` |
| Forall | `∀ x, P x` | `forall x, P x` | `∀x. P x` | `∀ x → P x` | `A. x P x` |
| Lambda | `fun x => e` | `fun x => e` | `λx. e` | `λ x → e` | N/A |
| Nat type | `Nat` | `nat` | `nat` | `ℕ` | `NN` |

**Solution: Notation Registry**

```rust
/// Registry mapping concepts to notations
pub struct NotationRegistry {
    /// Canonical name → system-specific notations
    notations: HashMap<UName, HashMap<SourceSystem, Notation>>,

    /// Parsing rules per system
    parsers: HashMap<SourceSystem, NotationParser>,

    /// Rendering rules per system
    renderers: HashMap<SourceSystem, NotationRenderer>,
}

pub struct Notation {
    /// How to render this concept
    pub format: NotationFormat,

    /// Precedence for parsing
    pub precedence: u32,

    /// Associativity
    pub assoc: Associativity,

    /// Unicode vs ASCII variants
    pub variants: Vec<String>,
}

impl NotationRegistry {
    /// Render UPIR expression for a target system
    pub fn render(&self, expr: &UExpr, target: SourceSystem) -> String {
        match expr {
            UExpr::Const { name, .. } => {
                if let Some(notation) = self.notations.get(name)?.get(&target) {
                    notation.render()
                } else {
                    name.to_string()
                }
            }
            UExpr::App(f, args) => {
                // Check for infix notation
                if let Some(infix) = self.get_infix(f, target) {
                    infix.render_infix(args, target)
                } else {
                    // Prefix application
                    format!("{} {}", self.render(f, target), self.render(args, target))
                }
            }
            // ... etc
        }
    }
}
```

---

### Gap 15: Record/Structure Encoding Differences

**Problem:** Records are encoded differently:

- **Lean:** Structures with inheritance, projections are functions
- **Coq:** Records with primitive projections (optional)
- **Agda:** Records with eta, potentially copatterns
- **Isabelle:** Records as syntactic sugar for tuples
- **HOL:** No records (encode as tuples)

```lean
-- Lean structure
structure Point where
  x : Float
  y : Float

-- Coq record
Record Point := { x : float; y : float }.

-- Agda record
record Point : Set where
  field x y : Float
```

**Solution: Unified Record Representation**

```rust
/// Universal record representation
pub struct URecord {
    /// Record name
    pub name: UName,

    /// Type parameters
    pub params: Vec<UBinder>,

    /// Fields
    pub fields: Vec<URecordField>,

    /// Parent records (for inheritance)
    pub parents: Vec<UName>,

    /// Source-specific behavior
    pub behavior: RecordBehavior,
}

pub struct URecordField {
    pub name: String,
    pub type_: UExpr,
    pub default: Option<UExpr>,
}

pub struct RecordBehavior {
    /// Does eta-expansion hold? (Agda)
    pub eta: bool,

    /// Are projections primitive? (Coq)
    pub primitive_projections: bool,

    /// Support copatterns? (Agda)
    pub copatterns: bool,

    /// How encoded in target?
    pub encoding: RecordEncoding,
}

pub enum RecordEncoding {
    /// Native record support
    Native,

    /// Encoded as inductive with one constructor
    SingleConstructor,

    /// Encoded as nested pairs
    NestedPairs,

    /// Encoded as function record
    FunctionRecord,
}
```

---

### Gap 16: Type Class vs Module vs Locale

**Problem:** Abstraction mechanisms differ fundamentally:

- **Lean:** Type classes (instance search)
- **Coq:** Type classes OR modules with functors
- **Agda:** Instance arguments OR modules with parameters
- **Isabelle:** Locales (named contexts)
- **HOL:** Type variables (simple polymorphism)

A Lean type class can't directly translate to an Isabelle locale.

**Solution: Abstraction Mechanism Mapping**

```rust
/// Abstract over abstraction mechanisms
pub enum AbstractionMechanism {
    /// Type class with instance search
    TypeClass {
        class: TypeClassDef,
        instances: Vec<InstanceDef>,
    },

    /// Parameterized module
    Module {
        module_type: ModuleSignature,
        params: Vec<ModuleParam>,
    },

    /// Named context (Isabelle locale)
    Locale {
        name: String,
        assumes: Vec<Assumption>,
        defines: Vec<Definition>,
    },

    /// Simple parametric polymorphism
    Polymorphic {
        type_vars: Vec<TypeVar>,
    },
}

/// When importing, choose target mechanism
pub fn translate_abstraction(
    source: &AbstractionMechanism,
    target_system: SourceSystem,
) -> AbstractionMechanism {
    match (source, target_system) {
        // Type class to type class (similar systems)
        (AbstractionMechanism::TypeClass { .. }, SourceSystem::Lean4 { .. }) |
        (AbstractionMechanism::TypeClass { .. }, SourceSystem::Coq { .. }) |
        (AbstractionMechanism::TypeClass { .. }, SourceSystem::Agda { .. }) => {
            source.clone()
        }

        // Type class to locale (Isabelle)
        (AbstractionMechanism::TypeClass { class, instances }, SourceSystem::Isabelle { .. }) => {
            typeclass_to_locale(class, instances)
        }

        // Type class to polymorphism (HOL)
        (AbstractionMechanism::TypeClass { class, .. }, SourceSystem::HolLight { .. }) => {
            typeclass_to_polymorphic(class)
        }

        // ... other translations
    }
}
```

---

### Gap 17: Proof Automation Certificates

**Problem:** Automation doesn't produce portable proof terms:

| System | Automation | Output |
|--------|------------|--------|
| Lean | `simp`, `aesop`, `omega` | Proof term (sometimes huge) |
| Coq | `auto`, `lia`, `ring` | Proof term |
| Isabelle | `simp`, `auto`, `sledgehammer` | Isar proof OR external certificate |
| Agda | `auto` (limited) | Proof term |
| Metamath | N/A (all explicit) | N/A |

Sledgehammer might call Z3, CVC5, E-prover with different certificate formats.

**Solution: Certificate Normalization**

```rust
/// Normalized proof certificate
pub enum ProofCertificate {
    /// Full proof term in UPIR
    TermProof(UExpr),

    /// Resolution proof (from SAT/SMT)
    Resolution {
        clauses: Vec<Clause>,
        resolution_steps: Vec<ResolutionStep>,
        original_solver: String,
    },

    /// DRAT proof (SAT)
    Drat {
        proof: Vec<DratLine>,
    },

    /// Alethe proof (SMT-LIB)
    Alethe {
        proof: AletheProof,
    },

    /// LFSC proof (CVC)
    Lfsc {
        proof: LfscProof,
    },

    /// Reconstruction script (can replay in target)
    ReconstructionScript {
        target_system: SourceSystem,
        script: String,
    },

    /// No certificate, trust oracle
    TrustedOracle {
        oracle: String,
        query: String,
        result: bool,
    },
}

/// Convert between certificate formats
pub fn normalize_certificate(
    cert: &ProofCertificate,
    source: SourceSystem,
) -> Result<ProofCertificate, CertError> {
    match cert {
        ProofCertificate::Alethe { proof } => {
            // Alethe can be checked or converted to term
            alethe_to_term(proof)
        }
        ProofCertificate::Drat { proof } => {
            // DRAT is for SAT, harder to convert
            Err(CertError::SatCertificateNotConvertible)
        }
        ProofCertificate::TrustedOracle { .. } => {
            // Can't do anything with trusted oracles
            Ok(cert.clone())
        }
        _ => Ok(cert.clone()),
    }
}
```

---

### Gap 18: Definitional vs Propositional Equality

**Problem:** What's definitional in one system is propositional in another:

```lean
-- Lean: definitional
example : (fun x => x) 5 = 5 := rfl

-- But this might not be definitional in Coq depending on settings
```

Systems have different reduction strategies affecting what's definitional.

**Solution: Equality Classification**

```rust
/// Classification of equality
pub enum EqualityKind {
    /// Definitionally equal (reduces to same term)
    Definitional,

    /// Propositionally equal (provable but not definitional)
    Propositional,

    /// Observationally equal (same behavior)
    Observational,

    /// Unknown (depends on system settings)
    Unknown,
}

/// Check equality status
pub fn classify_equality(
    lhs: &UExpr,
    rhs: &UExpr,
    system: SourceSystem,
) -> EqualityKind {
    // Try to reduce both sides
    let lhs_nf = normalize(lhs, system);
    let rhs_nf = normalize(rhs, system);

    if alpha_equivalent(&lhs_nf, &rhs_nf) {
        EqualityKind::Definitional
    } else if can_prove_equal(&lhs, &rhs, system) {
        EqualityKind::Propositional
    } else {
        EqualityKind::Unknown
    }
}
```

---

### Gap 19: Inductive Family Differences

**Problem:** Indexed inductive families work differently:

```lean
-- Lean: indices after parameters
inductive Vec (α : Type) : Nat → Type where
  | nil : Vec α 0
  | cons : α → Vec α n → Vec α (n + 1)

-- Coq: can have dependencies in indices
Inductive Vec (A : Type) : nat -> Type :=
  | vnil : Vec A 0
  | vcons : forall n, A -> Vec A n -> Vec A (S n).

-- Agda: more flexible index handling
data Vec (A : Set) : ℕ → Set where
  []  : Vec A 0
  _∷_ : ∀ {n} → A → Vec A n → Vec A (suc n)
```

**Solution: Unified Inductive Specification**

```rust
/// Universal inductive type specification
pub struct UInductive {
    /// Name
    pub name: UName,

    /// Universe parameters
    pub universe_params: Vec<UniverseName>,

    /// Parameters (shared across constructors)
    pub params: Vec<UParam>,

    /// Indices (vary per constructor)
    pub indices: Vec<UIndex>,

    /// Constructors
    pub constructors: Vec<UConstructor>,

    /// Source-specific info
    pub source_info: InductiveSourceInfo,
}

pub struct UParam {
    pub name: String,
    pub type_: UExpr,
    pub kind: ParamKind,
}

pub enum ParamKind {
    /// True parameter (same in all constructors)
    Uniform,

    /// Non-uniform (Coq-specific)
    NonUniform,
}

pub struct UIndex {
    pub name: String,
    pub type_: UExpr,
}

pub struct UConstructor {
    pub name: UName,
    pub args: Vec<UConstructorArg>,
    pub result_indices: Vec<UExpr>,
}

pub struct UConstructorArg {
    pub name: Option<String>,
    pub type_: UExpr,
    pub recursive: bool,
}
```

---

### Gap 20: Coinductive Types

**Problem:** Coinduction is handled very differently or not at all:

| System | Coinduction | Mechanism |
|--------|-------------|-----------|
| Lean 4 | ❌ (encode via quotients) | N/A |
| Coq | ✅ CoInductive | Guardedness |
| Agda | ✅ Records + sized | Sized types |
| Isabelle | ✅ codatatype | BNF |
| HOL | ❌ | N/A |

**Solution: Coinduction Abstraction**

```rust
/// Universal coinductive specification
pub struct UCoinductive {
    pub name: UName,
    pub params: Vec<UParam>,

    /// Observations (projections)
    pub observations: Vec<UObservation>,

    /// How coinduction is implemented
    pub implementation: CoinductionImpl,
}

pub struct UObservation {
    pub name: String,
    pub result_type: UExpr,
}

pub enum CoinductionImpl {
    /// Native coinductive (Coq)
    Native {
        guard_indices: Vec<u32>,
    },

    /// Via sized types (Agda)
    SizedTypes {
        size_type: UExpr,
    },

    /// Encoded as quotient (Lean)
    QuotientEncoding {
        underlying: UExpr,
        relation: UExpr,
    },

    /// BNF-based (Isabelle)
    Bnf {
        bnf_spec: String,
    },
}
```

---

### Gap 21: Proof Irrelevance

**Problem:** Proof irrelevance behaves differently:

- **Lean:** Propositions are proof-relevant (can pattern match)
- **Coq Prop:** Proof-irrelevant for extraction, but terms exist
- **Coq SProp:** Definitionally proof-irrelevant
- **Agda:** Optional irrelevance annotations

**Solution: Irrelevance Tracking**

```rust
/// Proof relevance mode
pub struct RelevanceInfo {
    /// Source system's treatment
    pub source_mode: RelevanceMode,

    /// Does the proof actually use relevance?
    pub uses_relevance: bool,

    /// Can be made irrelevant?
    pub irrelevance_safe: bool,
}

pub enum RelevanceMode {
    /// Proofs are first-class terms
    Relevant,

    /// Proofs are equal (propositional)
    PropIrrelevant,

    /// Proofs are definitionally equal
    DefIrrelevant,

    /// Erased at runtime but present in types
    RuntimeIrrelevant,
}
```

---

### Gap 22: Setoid Hell vs Quotients

**Problem:** Not all systems have quotient types:

- **Lean:** Built-in quotients (`Quot`)
- **Coq:** No quotients (setoid hell)
- **Agda (Cubical):** Higher inductive types (better!)
- **HOL:** Quotients available

Proofs using Lean quotients can't translate to standard Coq.

**Solution: Quotient Abstraction**

```rust
/// How quotients are handled
pub enum QuotientEncoding {
    /// Native quotient types (Lean, HOL)
    Native,

    /// Higher inductive types (Cubical)
    HIT,

    /// Setoids (Coq without quotients)
    Setoid {
        carrier: UExpr,
        equiv: UExpr,
        equiv_proof: UExpr,
    },

    /// Not supported
    Unsupported,
}

/// When importing quotient-using proof to Coq
pub fn quotient_to_setoid(
    quotient_proof: &UExpr,
) -> Result<SetoidProof, TranslationError> {
    // Extract the equivalence relation
    // Build setoid structure
    // Translate proof to work with setoids
    // This is complex and not always possible!
}
```

---

### Gap 23: Large Elimination

**Problem:** Large elimination (eliminating into Type) is restricted differently:

- **Lean:** Allowed for all inductives
- **Coq:** Restricted for Prop (no large elim from Prop to Type)
- **Agda:** Configurable via --without-K

**Solution: Elimination Tracking**

```rust
/// What eliminations are used
pub struct EliminationUsage {
    /// Uses large elimination?
    pub large_elim: bool,

    /// From what sorts?
    pub elim_from: Vec<USort>,

    /// To what sorts?
    pub elim_to: Vec<USort>,

    /// Specific K-like eliminations?
    pub uses_k: bool,
}

pub fn check_elimination_compatibility(
    usage: &EliminationUsage,
    target: SourceSystem,
) -> CompatibilityResult {
    match target {
        SourceSystem::Coq { .. } if usage.large_elim => {
            CompatibilityResult::Warning(
                "Coq restricts large elimination from Prop"
            )
        }
        SourceSystem::Agda { flags } if usage.uses_k && flags.without_k => {
            CompatibilityResult::Error(
                "Proof uses K but target Agda has --without-K"
            )
        }
        _ => CompatibilityResult::Ok,
    }
}
```

---

### Gap 24: Mutual Recursion / Induction

**Problem:** Mutual definitions handled differently:

- **Lean:** `mutual` block
- **Coq:** `with` in Fixpoint, or combined inductive
- **Agda:** Mutual blocks with interleaving
- **Isabelle:** Simultaneously defined

**Solution: Unified Mutual Block**

```rust
/// Mutual definition block
pub struct MutualBlock {
    /// Type of mutual definition
    pub kind: MutualKind,

    /// All definitions in the block
    pub definitions: Vec<MutualDef>,

    /// Termination evidence (for recursive)
    pub termination: Option<MutualTermination>,
}

pub enum MutualKind {
    /// Mutual inductive types
    Inductive,

    /// Mutual recursive functions
    Recursive,

    /// Mutual coinductive
    Coinductive,

    /// Mixed (some systems allow)
    Mixed,
}

pub struct MutualDef {
    pub name: UName,
    pub type_: UExpr,
    pub value: Option<UExpr>,
}
```

---

### Gap 25: SProp and Strict Propositions

**Problem:** Coq's SProp (strict propositions) has no direct equivalent:

- **Coq SProp:** Definitionally proof-irrelevant, no elimination
- **Lean:** No SProp
- **Agda:** Can simulate with irrelevance

**Solution: SProp Layer**

```rust
/// Handle SProp in translations
pub enum PropKind {
    /// Regular propositions
    Prop,

    /// Strict propositions (Coq SProp)
    SProp,

    /// Irrelevant (Agda @0)
    Irrelevant,
}

/// When translating SProp to non-SProp system
pub fn translate_sprop(
    expr: &UExpr,
    from: SourceSystem,
    to: SourceSystem,
) -> Result<UExpr, TranslationError> {
    if contains_sprop(expr) {
        match to {
            // Lean doesn't have SProp
            SourceSystem::Lean4 { .. } => {
                // Option 1: Upgrade to regular Prop (loses definitional irrelevance)
                // Option 2: Reject translation
                Err(TranslationError::SPropNotSupported)
            }
            // Agda can use irrelevance
            SourceSystem::Agda { .. } => {
                sprop_to_irrelevance(expr)
            }
            _ => Err(TranslationError::SPropNotSupported),
        }
    } else {
        Ok(expr.clone())
    }
}
```

---

## Summary: All Identified Gaps

| # | Gap | Severity | Solution |
|---|-----|----------|----------|
| 1 | Expression pool bloat (FlatBuffers) | HIGH | Custom compact encoding |
| 2 | BigNat literals | HIGH | Variable-length encoding |
| 3 | No cross-reference validation | HIGH | Bounds-checking layer |
| 4 | FlatBuffers union overhead | HIGH | Custom encoding for expressions |
| 5 | No string deduplication | MEDIUM | String interning |
| 6 | Mmap page fault storms | MEDIUM | Prefetching + batch access |
| 7 | No embedded schema | MEDIUM | Schema hash in header |
| 8 | Level pool same issues | HIGH | Custom level encoding |
| 9 | Coq compatibility | HIGH | UPIR design |
| 10 | HOL lacks dependent types | HIGH | Shallow embedding |
| 11 | Axiom incompatibilities | HIGH | Axiom tracking |
| 12 | Termination checking differences | HIGH | Evidence abstraction |
| 13 | Implicit argument differences | MEDIUM | Normalize to explicit |
| 14 | Notation differences | LOW | Notation registry |
| 15 | Record encoding differences | MEDIUM | Unified record repr |
| 16 | Type class vs module vs locale | HIGH | Abstraction mapping |
| 17 | Proof automation certificates | HIGH | Certificate normalization |
| 18 | Definitional vs propositional equality | MEDIUM | Equality classification |
| 19 | Inductive family differences | MEDIUM | Unified inductive spec |
| 20 | Coinductive types | HIGH | Coinduction abstraction |
| 21 | Proof irrelevance | MEDIUM | Irrelevance tracking |
| 22 | Setoid hell vs quotients | HIGH | Quotient abstraction |
| 23 | Large elimination restrictions | MEDIUM | Elimination tracking |
| 24 | Mutual recursion differences | MEDIUM | Mutual block |
| 25 | SProp and strict propositions | MEDIUM | SProp layer |

**Total Gaps Identified: 25**
**Solutions Proposed: 25**

---

## Implementation Priority

### Must Address (Blockers)
- Gap 1-4: Storage format issues
- Gap 9-10: Basic Coq/HOL support
- Gap 11: Axiom tracking
- Gap 22: Quotient handling

### Should Address (v1.0)
- Gap 12: Termination evidence
- Gap 16: Abstraction mechanisms
- Gap 17: Proof certificates
- Gap 19-20: Inductive/coinductive

### Can Defer (v1.1)
- Gap 5-8: Optimizations
- Gap 13-15: Elaboration/notation
- Gap 18, 21, 23-25: Edge cases

---

*This document complements DESIGN_UNIVERSAL_PROOF_DATABASE.md*
