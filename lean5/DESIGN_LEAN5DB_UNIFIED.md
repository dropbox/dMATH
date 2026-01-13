# Lean5DB: Unified Format for Universal Proofs

**Version:** 2.0
**Status:** Design (DO NOT EXECUTE)
**Date:** 2026-01-07
**Principle:** ONE format, Lean5-native, extensible

---

## Core Design Principle

```
┌─────────────────────────────────────────────────────────────────┐
│                     ONE FORMAT: Lean5DB                          │
│                                                                  │
│   NOT: lean.db + coq.db + isabelle.db + agda.db                 │
│   YES: unified.lean5db (contains everything)                     │
│                                                                  │
│   Primary consumer: Lean5 kernel (sub-microsecond verification)  │
│   Secondary: AI training, search, cross-reference                │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight:** All proofs are stored in Lean5-native representation. Proofs from other systems are *translated* to Lean5 terms during import, not stored in their original format.

---

## 1. Why Lean5-Native?

### 1.1 Lean5 as the Universal Verifier

```
External Systems          Translation Layer           Lean5DB
                                                      (ONE FORMAT)
┌─────────┐
│ Lean 4  │ ─────── direct import ──────────────┐
│ .olean  │                                      │
└─────────┘                                      │
                                                 │
┌─────────┐                                      ▼
│   Coq   │ ─────── translate to Lean5 ────▶ ┌──────────────┐
│   .vo   │                                  │              │
└─────────┘                                  │   Lean5DB    │
                                             │              │
┌─────────┐                                  │  • Lean5     │
│ Isabelle│ ─────── embed in Lean5 ─────────▶│    native    │
│  .thy   │                                  │    terms     │
└─────────┘                                  │              │
                                             │  • Unified   │
┌─────────┐                                  │    format    │
│  Agda   │ ─────── translate to Lean5 ────▶│              │
│ .agdai  │                                  │  • One file  │
└─────────┘                                  │              │
                                             └──────────────┘
┌─────────┐                                      │
│Metamath │ ─────── reconstruct in Lean5 ───────┘
│  .mm    │
└─────────┘
```

### 1.2 Benefits of Lean5-Native Storage

| Benefit | Explanation |
|---------|-------------|
| **Single verifier** | Lean5 kernel can verify ALL proofs |
| **Uniform API** | One API for everything |
| **Sub-microsecond** | Optimized for Lean5's kernel |
| **No format proliferation** | One format to maintain |
| **Consistent types** | Everything uses Lean5's type theory |
| **AI-ready** | Consistent representation for training |

### 1.3 What About HOL/Mizar/etc?

Systems with different type theories get **embedded** or **reconstructed**:

```lean
-- HOL theorem embedded in Lean5
-- Original: ∀x. x + 0 = x (in simple HOL)
-- Lean5 version:
namespace HOL
  axiom add_zero : ∀ (x : HOL.Nat), HOL.add x HOL.zero = x
  -- Marked as coming from HOL, verified in HOL
  -- Can be used in Lean5 via the HOL embedding
end HOL

-- Or: fully reconstructed proof
theorem Nat.add_zero : ∀ (n : Nat), n + 0 = n := by
  intro n; induction n <;> simp [Nat.add]
  -- This is a NEW Lean5 proof, inspired by HOL
```

---

## 2. Lean5DB Expression Format

### 2.1 Native Lean5 Expressions

The core format stores **Lean5 expressions directly**, not an intermediate representation:

```rust
/// Lean5-native expression (not UPIR!)
/// This is what lean5 kernel understands
pub enum Lean5Expr {
    /// Bound variable (de Bruijn index)
    BVar(u32),

    /// Sort (Prop or Type u)
    Sort(Lean5Level),

    /// Constant reference
    Const {
        name: NameId,
        levels: Vec<Lean5Level>,
    },

    /// Function application
    App {
        func: ExprId,
        arg: ExprId,
    },

    /// Lambda abstraction
    Lam {
        binder_info: BinderInfo,
        domain: ExprId,
        body: ExprId,
    },

    /// Pi type (dependent function)
    Pi {
        binder_info: BinderInfo,
        domain: ExprId,
        codomain: ExprId,
    },

    /// Let binding
    Let {
        type_: ExprId,
        value: ExprId,
        body: ExprId,
    },

    /// Literal (Nat or String)
    Lit(Lean5Literal),

    /// Projection
    Proj {
        struct_name: NameId,
        idx: u32,
        struct_: ExprId,
    },

    /// Metadata (source info, etc)
    MData {
        data: MDataId,
        expr: ExprId,
    },
}

/// Lean5-native universe level
pub enum Lean5Level {
    Zero,
    Succ(LevelId),
    Max(LevelId, LevelId),
    IMax(LevelId, LevelId),
    Param(NameId),
}
```

### 2.2 Source Provenance (Metadata, Not Structure)

The format tracks WHERE a proof came from, but stores it as Lean5:

```rust
/// Every constant has provenance info
pub struct ConstantInfo {
    /// The constant itself (Lean5 native)
    pub name: Name,
    pub type_: Lean5Expr,
    pub value: Option<Lean5Expr>,
    pub kind: ConstantKind,

    /// Where did this come from? (metadata)
    pub provenance: Provenance,
}

pub struct Provenance {
    /// Original source system
    pub source: SourceSystem,

    /// Original name (may differ from Lean5 name)
    pub original_name: Option<String>,

    /// How was it obtained?
    pub method: ImportMethod,

    /// Confidence level
    pub confidence: Confidence,

    /// Cross-references to same theorem in other systems
    pub equivalents: Vec<EquivalentRef>,
}

pub enum ImportMethod {
    /// Direct import (Lean 4 → Lean5)
    Direct,

    /// Translated proof term (Coq/Agda → Lean5)
    Translated {
        translator_version: String,
    },

    /// Reconstructed (proof rebuilt in Lean5)
    Reconstructed {
        original_proof_hash: [u8; 32],
    },

    /// Embedded (HOL as axioms in Lean5)
    Embedded {
        embedding_theory: String,
    },

    /// Axiomatized (no proof, trust source)
    Axiomatized,
}

pub enum Confidence {
    /// Verified by Lean5 kernel
    Verified,

    /// Verified in source system, translated
    TranslatedVerified,

    /// Axiomatized (trusted)
    Trusted,
}
```

---

## 3. Import Pipelines

### 3.1 Lean 4 → Lean5DB (Direct)

```rust
/// Lean 4 imports directly (same type theory)
pub fn import_lean4(olean_path: &Path) -> Result<Vec<ConstantInfo>, Error> {
    let module = parse_olean(olean_path)?;

    module.constants.into_iter().map(|c| {
        Ok(ConstantInfo {
            name: c.name,
            type_: lean4_expr_to_lean5(c.type_)?,
            value: c.value.map(|v| lean4_expr_to_lean5(v)).transpose()?,
            kind: c.kind,
            provenance: Provenance {
                source: SourceSystem::Lean4,
                original_name: None,  // Same name
                method: ImportMethod::Direct,
                confidence: Confidence::Verified,
                equivalents: vec![],
            },
        })
    }).collect()
}

/// Lean 4 → Lean5 expression (nearly identical)
fn lean4_expr_to_lean5(expr: Lean4Expr) -> Result<Lean5Expr, Error> {
    // Lean 4 and Lean 5 have same expression language
    // Just a direct mapping
    match expr {
        Lean4Expr::BVar(i) => Ok(Lean5Expr::BVar(i)),
        Lean4Expr::Sort(l) => Ok(Lean5Expr::Sort(lean4_level_to_lean5(l)?)),
        Lean4Expr::Const(n, ls) => Ok(Lean5Expr::Const {
            name: n,
            levels: ls.into_iter().map(lean4_level_to_lean5).collect::<Result<_, _>>()?,
        }),
        // ... all cases map directly
    }
}
```

### 3.2 Coq → Lean5DB (Translation)

```rust
/// Coq requires translation (different type theory details)
pub fn import_coq(vo_path: &Path) -> Result<Vec<ConstantInfo>, Error> {
    let coq_module = parse_coq_vo(vo_path)?;

    coq_module.constants.into_iter().map(|c| {
        // Translate Coq term to Lean5 term
        let lean5_type = translate_coq_to_lean5(&c.type_)?;
        let lean5_value = c.value.map(|v| translate_coq_to_lean5(&v)).transpose()?;

        Ok(ConstantInfo {
            name: coq_name_to_lean5_name(&c.name),
            type_: lean5_type,
            value: lean5_value,
            kind: coq_kind_to_lean5_kind(c.kind),
            provenance: Provenance {
                source: SourceSystem::Coq,
                original_name: Some(c.name.to_string()),
                method: ImportMethod::Translated {
                    translator_version: "0.1.0".into(),
                },
                confidence: Confidence::TranslatedVerified,
                equivalents: vec![],
            },
        })
    }).collect()
}

/// Core Coq → Lean5 translation
fn translate_coq_to_lean5(coq: &CoqTerm) -> Result<Lean5Expr, TranslationError> {
    match coq {
        // Variables map directly
        CoqTerm::Rel(i) => Ok(Lean5Expr::BVar(*i)),

        // Coq Prop → Lean5 Prop (but note: Coq Prop is impredicative!)
        CoqTerm::Sort(CoqSort::Prop) => {
            // Lean5's Prop is predicative, so some Coq proofs won't work
            // We mark this and may need to lift to Type
            Ok(Lean5Expr::Sort(Lean5Level::Zero))
        }

        // Coq Set → Lean5 Type 0
        CoqTerm::Sort(CoqSort::Set) => {
            Ok(Lean5Expr::Sort(Lean5Level::Succ(Box::new(Lean5Level::Zero))))
        }

        // Coq Type@{i} → Lean5 Type i
        CoqTerm::Sort(CoqSort::Type(u)) => {
            Ok(Lean5Expr::Sort(coq_universe_to_lean5(u)?))
        }

        // Products map to Pi
        CoqTerm::Prod(_, ty, body) => {
            Ok(Lean5Expr::Pi {
                binder_info: BinderInfo::Default,
                domain: Box::new(translate_coq_to_lean5(ty)?),
                codomain: Box::new(translate_coq_to_lean5(body)?),
            })
        }

        // Lambdas map directly
        CoqTerm::Lambda(_, ty, body) => {
            Ok(Lean5Expr::Lam {
                binder_info: BinderInfo::Default,
                domain: Box::new(translate_coq_to_lean5(ty)?),
                body: Box::new(translate_coq_to_lean5(body)?),
            })
        }

        // App maps directly
        CoqTerm::App(f, args) => {
            let mut result = translate_coq_to_lean5(f)?;
            for arg in args {
                result = Lean5Expr::App {
                    func: Box::new(result),
                    arg: Box::new(translate_coq_to_lean5(arg)?),
                };
            }
            Ok(result)
        }

        // Coq Fix needs careful handling
        CoqTerm::Fix(fix_data) => {
            translate_coq_fix_to_lean5(fix_data)
        }

        // Coq Match → Lean5 match (via recursor)
        CoqTerm::Case(case_data) => {
            translate_coq_case_to_lean5(case_data)
        }

        // Constants need name mapping
        CoqTerm::Const(name, univs) => {
            Ok(Lean5Expr::Const {
                name: coq_name_to_lean5_name(name),
                levels: univs.iter().map(coq_universe_to_lean5).collect::<Result<_, _>>()?,
            })
        }

        // Features Lean5 doesn't have
        CoqTerm::SProp => {
            // No SProp in Lean5 - use regular Prop with annotation
            Err(TranslationError::FeatureNotSupported("SProp"))
        }
    }
}
```

### 3.3 Isabelle/HOL → Lean5DB (Embedding)

```rust
/// HOL lacks dependent types - must embed as a DSL in Lean5
pub fn import_isabelle(thy_path: &Path) -> Result<Vec<ConstantInfo>, Error> {
    let isabelle_thy = parse_isabelle(thy_path)?;

    // First, ensure HOL embedding is available
    let hol_embedding = get_or_create_hol_embedding()?;

    isabelle_thy.theorems.into_iter().map(|thm| {
        // Convert HOL proposition to Lean5 type
        // This uses the HOL embedding: `HOL.Prop` type in Lean5
        let lean5_type = embed_hol_prop_as_lean5_type(&thm.statement, &hol_embedding)?;

        // HOL proofs are often not available as terms
        // We axiomatize them in the HOL namespace
        let lean5_value = if let Some(proof_term) = thm.proof_term {
            Some(embed_hol_proof_as_lean5(&proof_term, &hol_embedding)?)
        } else {
            None  // Axiomatized
        };

        Ok(ConstantInfo {
            name: Name::from_str(&format!("HOL.{}", thm.name))?,
            type_: lean5_type,
            value: lean5_value,
            kind: ConstantKind::Theorem,
            provenance: Provenance {
                source: SourceSystem::Isabelle,
                original_name: Some(thm.name),
                method: if lean5_value.is_some() {
                    ImportMethod::Embedded { embedding_theory: "HOL".into() }
                } else {
                    ImportMethod::Axiomatized
                },
                confidence: Confidence::Trusted,
                equivalents: vec![],
            },
        })
    }).collect()
}

/// The HOL embedding in Lean5
/// This defines HOL types and operations as Lean5 definitions
pub struct HolEmbedding {
    // HOL.Prop : Type (not Lean's Prop!)
    // HOL.Forall : (α → HOL.Prop) → HOL.Prop
    // HOL.Exists : (α → HOL.Prop) → HOL.Prop
    // HOL.Implies : HOL.Prop → HOL.Prop → HOL.Prop
    // HOL.And : HOL.Prop → HOL.Prop → HOL.Prop
    // etc.
}

/// Convert HOL formula to Lean5 type in HOL embedding
fn embed_hol_prop_as_lean5_type(
    hol_prop: &HolProp,
    embedding: &HolEmbedding,
) -> Result<Lean5Expr, Error> {
    match hol_prop {
        HolProp::Forall(ty, body) => {
            // ∀x:τ. P(x) becomes HOL.Forall (fun x : τ => P(x))
            Ok(Lean5Expr::App {
                func: embedding.forall_const.clone(),
                arg: Lean5Expr::Lam {
                    binder_info: BinderInfo::Default,
                    domain: embed_hol_type_as_lean5(ty)?,
                    body: embed_hol_prop_as_lean5_type(body, embedding)?,
                },
            })
        }
        HolProp::Implies(a, b) => {
            // A ⟹ B becomes HOL.Implies A B
            Ok(Lean5Expr::App {
                func: Lean5Expr::App {
                    func: embedding.implies_const.clone(),
                    arg: embed_hol_prop_as_lean5_type(a, embedding)?,
                },
                arg: embed_hol_prop_as_lean5_type(b, embedding)?,
            })
        }
        // ... etc
    }
}
```

### 3.4 Metamath → Lean5DB (Reconstruction)

```rust
/// Metamath proofs can be reconstructed step-by-step in Lean5
pub fn import_metamath(mm_path: &Path) -> Result<Vec<ConstantInfo>, Error> {
    let mm_db = parse_metamath(mm_path)?;

    mm_db.theorems.into_iter().map(|thm| {
        // Parse the Metamath statement into Lean5 type
        let lean5_type = metamath_stmt_to_lean5_type(&thm.statement, &mm_db)?;

        // Reconstruct the proof in Lean5
        // Metamath proofs are explicit - each step is an axiom application
        let lean5_proof = reconstruct_metamath_proof_in_lean5(
            &thm.proof,
            &mm_db,
        )?;

        Ok(ConstantInfo {
            name: Name::from_str(&format!("Metamath.{}", thm.label))?,
            type_: lean5_type,
            value: Some(lean5_proof),
            kind: ConstantKind::Theorem,
            provenance: Provenance {
                source: SourceSystem::Metamath,
                original_name: Some(thm.label),
                method: ImportMethod::Reconstructed {
                    original_proof_hash: hash_metamath_proof(&thm.proof),
                },
                confidence: Confidence::Verified,  // Re-verified in Lean5!
                equivalents: vec![],
            },
        })
    }).collect()
}

/// Reconstruct Metamath proof as Lean5 term
fn reconstruct_metamath_proof_in_lean5(
    proof: &MetamathProof,
    db: &MetamathDb,
) -> Result<Lean5Expr, Error> {
    // Metamath proofs are RPN sequences
    // Each step is: apply an axiom/theorem with substitutions
    let mut stack: Vec<Lean5Expr> = vec![];

    for step in &proof.steps {
        match step {
            MetamathStep::Hyp(idx) => {
                // Push hypothesis
                stack.push(Lean5Expr::BVar(*idx));
            }
            MetamathStep::Apply(label) => {
                // Get the axiom/theorem
                let stmt = db.get(label)?;

                // Pop arguments from stack
                let args: Vec<_> = (0..stmt.hyp_count)
                    .map(|_| stack.pop().unwrap())
                    .collect();

                // Build application
                let mut result = Lean5Expr::Const {
                    name: Name::from_str(&format!("Metamath.{}", label))?,
                    levels: vec![],
                };
                for arg in args.into_iter().rev() {
                    result = Lean5Expr::App {
                        func: Box::new(result),
                        arg: Box::new(arg),
                    };
                }
                stack.push(result);
            }
        }
    }

    assert_eq!(stack.len(), 1);
    Ok(stack.pop().unwrap())
}
```

---

## 4. File Format (Single Format)

### 4.1 Lean5DB File Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEAN5DB FILE (v2.0)                          │
│                    ONE FORMAT FOR ALL                           │
├─────────────────────────────────────────────────────────────────┤
│  Header (256 bytes)                                             │
│    magic: "LEAN5DB2"                                            │
│    version: 2.0                                                 │
│    flags: compression, features enabled                         │
│    source_systems: bitmap of imported systems                   │
│    content_hash: BLAKE3 of all constants                        │
├─────────────────────────────────────────────────────────────────┤
│  Bloom Filter (~1MB)                                            │
│    Fast negative lookup for constant names                      │
├─────────────────────────────────────────────────────────────────┤
│  String Table (FlatBuffers)                                     │
│    All interned strings                                         │
├─────────────────────────────────────────────────────────────────┤
│  Name Table (FlatBuffers)                                       │
│    Hierarchical names (Lean5 native format)                     │
├─────────────────────────────────────────────────────────────────┤
│  Expression Pool (Custom compact)                               │
│    All Lean5 expressions (deduplicated)                         │
│    Offset table for O(1) access                                 │
├─────────────────────────────────────────────────────────────────┤
│  Level Pool (Custom compact)                                    │
│    All universe levels                                          │
├─────────────────────────────────────────────────────────────────┤
│  Constant Index (FlatBuffers)                                   │
│    name_id, kind, type_expr_id, value_expr_id                   │
│    provenance_id (links to provenance table)                    │
├─────────────────────────────────────────────────────────────────┤
│  Provenance Table (FlatBuffers)                                 │
│    source_system, original_name, import_method                  │
│    equivalents (cross-references)                               │
├─────────────────────────────────────────────────────────────────┤
│  Constant Data Chunks (postcard + zstd)                         │
│    Inductive specs, recursor rules, etc.                        │
├─────────────────────────────────────────────────────────────────┤
│  Proof Archive (optional, separate or inline)                   │
│    Large proof terms (zstd compressed)                          │
├─────────────────────────────────────────────────────────────────┤
│  Search Index (optional)                                        │
│    Keyword, semantic, type search data                          │
├─────────────────────────────────────────────────────────────────┤
│  Footer (64 bytes)                                              │
│    Section checksums                                            │
│    File checksum                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Source System Tracking in Header

```rust
/// Header tracks which systems have been imported
pub struct Lean5DbHeader {
    pub magic: [u8; 8],
    pub version_major: u16,
    pub version_minor: u16,
    pub flags: u32,

    /// Bitmap of source systems included
    /// bit 0: Lean4
    /// bit 1: Coq
    /// bit 2: Isabelle
    /// bit 3: Agda
    /// bit 4: HOL Light
    /// bit 5: HOL4
    /// bit 6: Mizar
    /// bit 7: Metamath
    /// bit 8: PVS
    /// bit 9: ACL2
    pub source_systems: u32,

    /// Counts per system
    pub lean4_count: u32,
    pub coq_count: u32,
    pub isabelle_count: u32,
    pub agda_count: u32,
    pub hol_count: u32,
    pub mizar_count: u32,
    pub metamath_count: u32,
    pub other_count: u32,

    /// Content hash (proves integrity)
    pub content_hash: [u8; 32],

    // ... section offsets
}
```

---

## 5. Lean5 Kernel Integration

### 5.1 Direct Loading

```rust
impl Lean5Kernel {
    /// Load constants from Lean5DB directly into kernel
    pub fn load_lean5db(&mut self, db: &Lean5Db) -> Result<(), KernelError> {
        for constant in db.stream_constants() {
            // Constants are already in Lean5-native format
            // No translation needed!
            self.add_constant(constant)?;
        }
        Ok(())
    }

    /// Verify a specific constant
    pub fn verify_constant(&self, name: &Name) -> Result<(), KernelError> {
        let constant = self.get_constant(name)?;

        // Type check the type
        self.check_type(&constant.type_)?;

        // If has value, verify it has the declared type
        if let Some(value) = &constant.value {
            let inferred = self.infer_type(value)?;
            self.ensure_def_eq(&inferred, &constant.type_)?;
        }

        Ok(())
    }
}
```

### 5.2 Lazy Verification

```rust
/// Lazy verifier - verify on demand
pub struct LazyVerifier {
    db: Lean5Db,
    kernel: Lean5Kernel,
    verified: HashSet<Name>,
}

impl LazyVerifier {
    /// Get a constant, verifying it first if needed
    pub fn get_verified(&mut self, name: &Name) -> Result<&ConstantInfo, Error> {
        if !self.verified.contains(name) {
            // Load dependencies first
            let deps = self.db.get_dependencies(name)?;
            for dep in deps {
                self.get_verified(&dep)?;
            }

            // Now verify this constant
            let constant = self.db.get_constant(name)?;
            self.kernel.verify_constant(constant)?;
            self.verified.insert(name.clone());
        }

        self.db.get_constant(name)
    }
}
```

---

## 6. Query API (Unified)

### 6.1 Single API for All Sources

```rust
/// Unified query interface
impl Lean5Db {
    /// Get constant by name (any original system)
    pub fn get(&self, name: &str) -> Option<&ConstantInfo> {
        // Try as Lean5 name first
        if let Some(c) = self.get_by_lean5_name(name) {
            return Some(c);
        }

        // Try as original name from any system
        self.get_by_original_name(name)
    }

    /// Find equivalents across systems
    pub fn get_equivalents(&self, name: &Name) -> Vec<&ConstantInfo> {
        let constant = self.get_constant(name)?;
        constant.provenance.equivalents.iter()
            .filter_map(|eq| self.get_constant(&eq.name))
            .collect()
    }

    /// Get all constants from a specific source system
    pub fn from_source(&self, source: SourceSystem) -> impl Iterator<Item = &ConstantInfo> {
        self.constants.iter()
            .filter(move |c| c.provenance.source == source)
    }

    /// Search across all systems
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        // Searches everything - Lean4, Coq, Isabelle, etc.
        // Returns unified results
        self.search_engine.search(query)
    }
}
```

### 6.2 Provenance Queries

```rust
impl Lean5Db {
    /// How many proofs do we have for this theorem?
    pub fn proof_count(&self, statement_hash: &[u8; 32]) -> usize {
        self.equivalence_index.get(statement_hash)
            .map(|eqs| eqs.len())
            .unwrap_or(0)
    }

    /// Get all proofs of a statement
    pub fn all_proofs_of(&self, name: &Name) -> Vec<ProofVariant> {
        let constant = self.get_constant(name)?;
        let equivalents = self.get_equivalents(name);

        let mut proofs = vec![];

        // Primary proof
        if let Some(value) = &constant.value {
            proofs.push(ProofVariant {
                source: constant.provenance.source.clone(),
                proof: value.clone(),
                confidence: constant.provenance.confidence,
            });
        }

        // Equivalent proofs from other systems
        for eq in equivalents {
            if let Some(value) = &eq.value {
                proofs.push(ProofVariant {
                    source: eq.provenance.source.clone(),
                    proof: value.clone(),
                    confidence: eq.provenance.confidence,
                });
            }
        }

        proofs
    }
}
```

---

## 7. Extension Points (Future-Proof)

### 7.1 Adding New Source Systems

```rust
/// Trait for source system importers
pub trait SourceImporter {
    /// System identifier
    fn system(&self) -> SourceSystem;

    /// Parse source files
    fn parse(&self, path: &Path) -> Result<Vec<SourceConstant>, ParseError>;

    /// Translate to Lean5
    fn to_lean5(&self, constant: &SourceConstant) -> Result<ConstantInfo, TranslationError>;
}

/// Register a new importer
pub fn register_importer(importer: Box<dyn SourceImporter>) {
    IMPORTERS.lock().insert(importer.system(), importer);
}

// Adding a new system is just implementing SourceImporter
// No file format changes needed!
```

### 7.2 Version Migration

```rust
/// File format is versioned for forward compatibility
pub struct FormatVersion {
    pub major: u16,  // Breaking changes
    pub minor: u16,  // New optional features
}

impl Lean5Db {
    pub fn can_read(file_version: FormatVersion, reader_version: FormatVersion) -> bool {
        // Same major version required
        // Reader must be >= file minor version
        file_version.major == reader_version.major &&
        reader_version.minor >= file_version.minor
    }

    pub fn migrate(data: &[u8], from: FormatVersion, to: FormatVersion) -> Result<Vec<u8>, Error> {
        // Migration path for version upgrades
    }
}
```

---

## 8. Summary: One Format Principle

| Aspect | Approach |
|--------|----------|
| **Storage format** | ONE: Lean5DB |
| **Expression language** | Lean5 native (not intermediate) |
| **Multiple systems** | Translated/embedded into Lean5 |
| **Provenance** | Metadata, not structural |
| **Verification** | Lean5 kernel verifies all |
| **API** | Unified, system-agnostic |
| **Extension** | Add importers, not formats |

**The key insight:** Instead of storing proofs in their original format and having multiple verifiers, we translate everything to Lean5 and use one verifier. This gives us:

1. **Consistency** - Same types everywhere
2. **Verifiability** - One kernel checks everything
3. **Simplicity** - One format to maintain
4. **Performance** - Optimized for Lean5
5. **Future-proof** - Add systems without format changes
