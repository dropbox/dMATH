# TLA2 Modernization Vision: Rust + Lean4

> A complete reimagining of the TLA+ toolchain with Rust as the primary implementation language and Lean4 as the proof backend.

## Table of Contents

1. [The Problem](#the-problem)
2. [The Vision](#the-vision)
3. [Why Rust](#why-rust)
4. [Why Lean4](#why-lean4)
5. [Architecture](#architecture)
6. [Component Breakdown](#component-breakdown)
7. [The Proof Stack](#the-proof-stack)
8. [Migration from Current Tools](#migration-from-current-tools)
9. [What We Keep](#what-we-keep)
10. [What We Replace](#what-we-replace)
11. [Implementation Strategy](#implementation-strategy)
12. [Risk Assessment](#risk-assessment)

---

## The Problem

### Current State: Fragmented and Heavy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CURRENT TLA+ ECOSYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐      ┌─────────────────────────┐              │
│  │     tlaplus/tlaplus     │      │     tlaplus/tlapm       │              │
│  │                         │      │                         │              │
│  │  Java: 375,000 lines    │      │  OCaml: 91,000 lines    │              │
│  │                         │      │                         │              │
│  │  • SANY parser          │      │  • TLAPM proof manager  │              │
│  │  • TLC model checker    │      │  • Zenon prover         │              │
│  │  • PlusCal translator   │      │  • LSP server           │              │
│  │  • Eclipse Toolbox      │      │                         │              │
│  └───────────┬─────────────┘      └───────────┬─────────────┘              │
│              │                                │                             │
│              │                                │                             │
│              │    NO SHARED CODE              │                             │
│              │    DIFFERENT ASTs              │                             │
│              │    DIFFERENT BUGS              │                             │
│              │                                │                             │
│              │                                ▼                             │
│              │                    ┌─────────────────────────┐              │
│              │                    │      ISABELLE           │              │
│              │                    │                         │              │
│              │                    │  2,000,000+ lines       │              │
│              │                    │  1GB download           │              │
│              │                    │  Escape hatch for       │              │
│              │                    │  hard proofs            │              │
│              │                    └─────────────────────────┘              │
│              │                                                              │
└──────────────┴──────────────────────────────────────────────────────────────┘

Problems:
• Two implementations of TLA+ semantics (Java + OCaml)
• Two parsers with different behaviors
• 1GB dependency for interactive proofs
• No code generation
• Dated IDE (Eclipse)
• No incremental checking
• Academic error messages
```

### By the Numbers

| Component | Language | Lines | Problem |
|-----------|----------|-------|---------|
| TLC | Java | 93K | No shared code with proofs |
| SANY | Java | 52K | Duplicated in TLAPM |
| TLAPM | OCaml | 67K | Different AST than TLC |
| Zenon | OCaml | 15K | Bundled, not reusable |
| Toolbox | Java | 117K | Eclipse-based, dated |
| Isabelle | ML/Scala | 2M+ | Massive overkill |

**Total: ~500K lines across 3 languages, with duplicated functionality**

---

## The Vision

### Target State: Unified and Modern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TLA2: UNIFIED RUST TOOLCHAIN                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌─────────────────────────┐                         │
│                         │       tla-core          │                         │
│                         │                         │                         │
│                         │  • Parser (tree-sitter) │                         │
│                         │  • AST                  │                         │
│                         │  • Semantic analysis    │                         │
│                         │  • Evaluation           │                         │
│                         │                         │                         │
│                         │  ONE SOURCE OF TRUTH    │                         │
│                         └───────────┬─────────────┘                         │
│                                     │                                       │
│           ┌─────────────────────────┼─────────────────────────┐             │
│           │                         │                         │             │
│           ▼                         ▼                         ▼             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   tla-check     │    │   tla-prove     │    │   tla-codegen   │         │
│  │                 │    │                 │    │                 │         │
│  │ Model checker   │    │ Proof manager   │    │ Rust generation │         │
│  │ (replaces TLC)  │    │(replaces TLAPM) │    │ + verification  │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           │              ┌───────┴───────┐              │                   │
│           │              │               │              │                   │
│           │              ▼               ▼              │                   │
│           │     ┌─────────────┐ ┌─────────────┐        │                   │
│           │     │  tla-smt    │ │  tla-zenon  │        │                   │
│           │     │             │ │             │        │                   │
│           │     │ Z3/CVC5     │ │ Tableau     │        │                   │
│           │     │ (z3 crate)  │ │ (Rust impl) │        │                   │
│           │     └──────┬──────┘ └──────┬──────┘        │                   │
│           │            │               │               │                   │
│           │            └───────┬───────┘               │                   │
│           │                    │                       │                   │
│           │                    ▼                       │                   │
│           │         ┌─────────────────┐                │                   │
│           │         │   tla-cert      │                │                   │
│           │         │                 │                │                   │
│           │         │ Certificate     │                │                   │
│           │         │ checker         │                │                   │
│           │         │ (trusted core)  │                │                   │
│           │         └────────┬────────┘                │                   │
│           │                  │                         │                   │
│           │                  ▼                         │                   │
│           │    ┌──────────────────────────┐           │                   │
│           │    │   LEAN4 (optional)       │           │                   │
│           │    │                          │           │                   │
│           │    │  • Interactive proofs    │           │                   │
│           │    │  • Hard cases only       │           │                   │
│           │    │  • ~50MB (not 1GB)       │           │                   │
│           │    │  • Export/import certs   │           │                   │
│           │    └──────────────────────────┘           │                   │
│           │                                           │                   │
│           └─────────────────────┬─────────────────────┘                   │
│                                 │                                         │
│                                 ▼                                         │
│                    ┌──────────────────────────┐                           │
│                    │   Verified Rust Code     │                           │
│                    │   + Runtime Monitoring   │                           │
│                    └──────────────────────────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Benefits:
• Single implementation of TLA+ semantics
• One parser, one AST, shared by all tools
• Lightweight: no mandatory large dependencies
• Code generation with verification
• Modern error messages
• LSP-first editor support
• Incremental everything
```

---

## Why Rust

### Technical Reasons

| Requirement | Rust Capability |
|-------------|-----------------|
| **Performance** | Zero-cost abstractions, no GC pauses |
| **Parallelism** | Fearless concurrency, rayon for easy parallelism |
| **Memory safety** | No null pointers, no data races |
| **FFI** | Easy C interop for Z3, potential Lean4 FFI |
| **Tooling** | Cargo, rustfmt, clippy, excellent IDE support |
| **WASM** | Compile to WebAssembly for browser playground |
| **Ecosystem** | z3 crate, tower-lsp, tree-sitter bindings |

### Strategic Reasons

| Factor | Benefit |
|--------|---------|
| **Growing adoption** | Rust is the fastest-growing systems language |
| **Verification ecosystem** | Kani, Verus, Creusot - active development |
| **Target language** | Generated code is also Rust - same ecosystem |
| **Community** | Large, active, willing to contribute |
| **Longevity** | Strong industry backing (Microsoft, Google, AWS) |

### Comparison with Alternatives

| Language | Pros | Cons |
|----------|------|------|
| **Rust** | Fast, safe, great ecosystem | Learning curve |
| **OCaml** | Good for compilers, TLAPM already uses it | Smaller ecosystem, less parallelism |
| **Haskell** | Pure, good for DSLs | Lazy evaluation complexity, GC |
| **Go** | Simple, fast compilation | No generics (until recently), GC |
| **C++** | Fast, mature | Memory safety nightmare |

**Rust wins on the combination of performance, safety, and ecosystem.**

---

## Why Lean4

### The Isabelle Problem

Isabelle is a fantastic proof assistant, but it's overkill for TLA+:

- **Size**: 2M+ lines, 1GB download
- **Complexity**: Full interactive proof environment
- **Learning curve**: Steep, separate tool to learn
- **Integration**: Subprocess only, clunky
- **Usage**: 95% of users never need it

### Why Not Keep Isabelle?

| Issue | Impact |
|-------|--------|
| Download size | 1GB is absurd for an "escape hatch" |
| Cold start | Minutes to load |
| UX | Different tool, different syntax, different workflow |
| Maintenance | We don't control it, ML expertise needed |

### Why Lean4?

| Factor | Lean4 | Isabelle |
|--------|-------|----------|
| **Size** | ~50MB | ~1GB |
| **Language** | Lean (modern) | ML (dated) |
| **Startup** | Fast | Slow |
| **Automation** | Good and improving | Good but stagnant |
| **FFI** | Possible from Rust | Not practical |
| **Community** | Growing fast (Mathlib) | Stable but not growing |
| **Self-hosting** | Lean4 is written in Lean4 | Complex ML/Scala stack |

### What Lean4 Provides

1. **Escape hatch for hard proofs**: When Z3/Zenon fail
2. **Certificate verification**: Alternative trusted kernel
3. **Future integration**: Potential deeper embedding of TLA+
4. **Community**: Active, growing, mathematically sophisticated

### What Lean4 Does NOT Do

- It's not mandatory - most users never need it
- It doesn't replace Z3/Zenon for automation
- It doesn't change the primary Rust architecture

---

## Architecture

### Crate Dependency Graph

```
                              tla-cli
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
                 ▼               ▼               ▼
            tla-check       tla-prove       tla-codegen
                 │               │               │
                 │       ┌───────┴───────┐       │
                 │       │               │       │
                 │       ▼               ▼       │
                 │   tla-smt        tla-zenon   │
                 │       │               │       │
                 │       └───────┬───────┘       │
                 │               │               │
                 │               ▼               │
                 │          tla-cert             │
                 │               │               │
                 └───────────────┼───────────────┘
                                 │
                                 ▼
                             tla-core
```

### Data Flow

```
TLA+ Source
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              tla-core                                        │
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  Lexer  │───▶│ Parser  │───▶│   AST   │───▶│Resolver │───▶│  Typed  │  │
│  │         │    │         │    │         │    │         │    │   AST   │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│                                                                     │       │
└─────────────────────────────────────────────────────────────────────┼───────┘
                                                                      │
                      ┌───────────────────────────────────────────────┤
                      │                       │                       │
                      ▼                       ▼                       ▼
               ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
               │  tla-check  │         │  tla-prove  │         │ tla-codegen │
               │             │         │             │         │             │
               │ State space │         │ Obligations │         │ Rust source │
               │ exploration │         │ generation  │         │ generation  │
               └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
                      │                       │                       │
                      ▼                       ▼                       ▼
               ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
               │   Result    │         │   Proof     │         │   Rust +    │
               │             │         │   Result    │         │   Kani +    │
               │ • States    │         │             │         │   Verus     │
               │ • Errors    │         │ • Proved    │         │             │
               │ • Traces    │         │ • Failed    │         │             │
               └─────────────┘         │ • Unknown   │         └─────────────┘
                                       └─────────────┘
```

---

## Component Breakdown

### tla-core (Foundation)

**Replaces**: SANY (Java) + TLAPM parser (OCaml)

```rust
// Core types shared by all tools
pub struct Module {
    pub name: Symbol,
    pub extends: Vec<Symbol>,
    pub units: Vec<Unit>,
}

pub enum Unit {
    Constant(Vec<Decl>),
    Variable(Vec<Decl>),
    Operator(OpDef),
    Theorem { name: Option<Symbol>, body: Expr, proof: Option<Proof> },
    // ...
}

pub enum Expr {
    // All TLA+ expression forms
}

pub enum TypedExpr {
    // After semantic analysis - resolved names, checked arities
}
```

**Key features**:
- Tree-sitter grammar for incremental parsing
- Rowan for syntax tree manipulation
- Ariadne for beautiful error messages
- Single AST used by checker, prover, and codegen

### tla-check (Model Checker)

**Replaces**: TLC (Java, 93K lines)

```rust
pub struct ModelChecker {
    spec: TypedModule,
    config: CheckConfig,
    seen: FingerprintSet,
    queue: StateQueue,
}

impl ModelChecker {
    pub fn check(&mut self) -> CheckResult {
        // Parallel BFS/DFS state exploration
        // Invariant checking
        // Liveness checking
        // Counterexample generation
    }
}
```

**Key features**:
- Parallel state exploration (rayon)
- Lock-free fingerprint set
- Disk-based state storage for large state spaces
- Simulation mode for huge spaces

### tla-prove (Proof Manager)

**Replaces**: TLAPM (OCaml, 67K lines)

```rust
pub struct ProofManager {
    spec: TypedModule,
    obligations: Vec<Obligation>,
    provers: Vec<Box<dyn Prover>>,
    cache: ProofCache,
}

impl ProofManager {
    pub fn prove(&mut self) -> ProofResult {
        // Generate obligations from proof structure
        // Dispatch to provers (Z3, Zenon, Lean4)
        // Collect and aggregate results
        // Cache successful proofs
    }
}
```

**Key features**:
- Smart prover selection based on obligation features
- Parallel prover invocation
- Proof caching (sled database)
- Incremental re-proving on spec changes

### tla-smt (SMT Integration)

**Replaces**: TLAPM SMT backend

```rust
pub trait SmtSolver {
    fn assert(&mut self, formula: &SmtExpr);
    fn check_sat(&mut self) -> SatResult;
    fn get_proof(&mut self) -> Option<SmtProof>;
}

pub struct Z3Solver { /* z3 crate */ }
pub struct SmtLibSolver { /* text-based for CVC5, Yices */ }
```

**Key features**:
- Z3 via `z3` crate (native bindings)
- SMT-LIB text output for other solvers
- Proof certificate extraction

### tla-zenon (Tableau Prover)

**Replaces**: Zenon (OCaml, 15K lines)

```rust
pub struct Tableau {
    branches: Vec<Branch>,
    proof: ProofTree,
}

impl Tableau {
    pub fn prove(&mut self, goal: Formula, timeout: Duration) -> ProofResult {
        // Tableau expansion
        // Branch closing
        // Proof certificate generation
    }
}
```

**Key features**:
- Pure Rust implementation (not a port)
- Better parallelism than OCaml version
- Proof certificate output

### tla-cert (Certificate Checker)

**New component - no equivalent in current tools**

```rust
/// Trusted proof certificate checker.
/// Target: <2000 lines, auditable.
pub struct CertificateChecker {
    facts: HashMap<StepId, Expr>,
}

impl CertificateChecker {
    pub fn check(&mut self, cert: &Certificate) -> Result<(), CheckError> {
        // Verify each proof step
        // Trust nothing except logical rules
    }
}
```

**Key features**:
- Minimal trusted computing base
- Accepts certificates from Z3, Zenon, Lean4
- Independent verification of proofs

### tla-codegen (Code Generation)

**New component - no equivalent in current tools**

```rust
pub struct CodeGenerator {
    spec: TypedModule,
    config: CodegenConfig,
}

impl CodeGenerator {
    pub fn generate(&self) -> GeneratedCode {
        // State struct from VARIABLES
        // Action enum from actions
        // StateMachine impl
        // Invariant checks
        // Kani harnesses
        // Verus specs (optional)
    }
}
```

**Key features**:
- Generate idiomatic Rust
- StateMachine trait implementation
- Test harness generation
- Kani proof harnesses
- Optional Verus specifications

### tla-lsp (Language Server)

**Replaces**: TLAPM LSP (OCaml, 5K lines)

```rust
#[tower_lsp::async_trait]
impl LanguageServer for TlaLanguageServer {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult>;
    async fn did_change(&self, params: DidChangeTextDocumentParams);
    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>>;
    async fn goto_definition(&self, params: GotoDefinitionParams) -> Result<Option<GotoDefinitionResponse>>;
    // ...
}
```

**Key features**:
- tower-lsp for protocol handling
- Incremental parsing via tree-sitter
- Real-time diagnostics
- Go-to-definition, hover, completions

### tla-cli (Command Line)

**Replaces**: Various CLI entry points

```rust
#[derive(Parser)]
#[command(name = "tla")]
pub enum Command {
    /// Parse and check a TLA+ specification
    Check {
        #[arg(required = true)]
        files: Vec<PathBuf>,
        #[arg(short, long)]
        workers: Option<usize>,
    },
    /// Prove properties of a specification
    Prove {
        #[arg(required = true)]
        files: Vec<PathBuf>,
    },
    /// Generate Rust code from a specification
    Codegen {
        #[arg(required = true)]
        file: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Start the language server
    Lsp,
}
```

---

## The Proof Stack

### Automation First

```
Obligation arrives
        │
        ▼
┌───────────────────┐
│ Feature Analysis  │  ← Arithmetic? Sets? Temporal? First-order?
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Strategy Selection│  ← Based on features + history
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌───────┐   ┌───────┐
│  Z3   │   │Zenon  │  ← Try in parallel
└───┬───┘   └───┬───┘
    │           │
    └─────┬─────┘
          │
          ▼
    ┌───────────┐
    │  Result?  │
    └─────┬─────┘
          │
    ┌─────┴─────┐
    │           │
Proved      Failed/Unknown
    │           │
    ▼           ▼
┌───────┐   ┌───────────────┐
│ Cache │   │ Try Lean4     │  ← Optional, user-installed
│       │   │ (if available)│
└───────┘   └───────┬───────┘
                    │
              ┌─────┴─────┐
              │           │
           Proved    Still Failed
              │           │
              ▼           ▼
          ┌───────┐   ┌───────────────┐
          │ Cache │   │ Report to user│
          └───────┘   │ with guidance │
                      └───────────────┘
```

### Lean4 Integration

Lean4 is **optional** and **on-demand**:

```rust
pub struct Lean4Prover {
    lean_path: Option<PathBuf>,  // None if not installed
}

impl Prover for Lean4Prover {
    fn prove(&self, obligation: &Obligation) -> ProofResult {
        let Some(lean) = &self.lean_path else {
            return ProofResult::Unavailable {
                reason: "Lean4 not installed. Install with: elan install lean4",
            };
        };

        // Generate Lean4 file
        let lean_source = self.generate_lean(obligation);

        // Run Lean4
        let output = Command::new(lean)
            .arg(&lean_file)
            .output()?;

        // Parse result
        self.parse_result(output)
    }
}
```

**Integration modes**:

1. **Batch mode**: Generate .lean file, run lean4, parse output
2. **Certificate mode**: Extract proof certificate, verify in tla-cert
3. **Interactive mode**: Export to Lean4, user completes proof, import certificate

### TLA+ in Lean4

For deep integration, define TLA+ semantics in Lean4:

```lean
-- TLA+ formalization in Lean4
-- This is the "18K lines of Isabelle theories" equivalent

namespace TLA

-- TLA+ values
inductive Value where
  | bool : Bool → Value
  | int : Int → Value
  | string : String → Value
  | set : Finset Value → Value
  | func : (Value → Value) → Value
  | tuple : List Value → Value

-- TLA+ state
def State := String → Value

-- TLA+ action (state predicate over primed/unprimed vars)
def Action := State → State → Prop

-- TLA+ temporal formula
inductive Temporal where
  | always : Temporal → Temporal      -- []P
  | eventually : Temporal → Temporal  -- <>P
  | until : Temporal → Temporal → Temporal
  | state : (State → Prop) → Temporal

-- Satisfaction relation
def satisfies (behavior : ℕ → State) (φ : Temporal) : Prop := ...

end TLA
```

This allows:
- Proving properties about TLA+ itself
- Verifying the model checker is sound
- Deep integration with Mathlib for mathematical reasoning

---

## Migration from Current Tools

### Compatibility Strategy

| Current | TLA2 | Compatibility |
|---------|------|---------------|
| .tla files | .tla files | 100% syntax compatible |
| TLC config | tla.toml | Migration tool provided |
| TLAPM proofs | Same syntax | Should work unchanged |
| Toolbox projects | tla.toml | Migration tool provided |

### Migration Path

```
Phase 1: Parser Compatibility
         Parse all tlaplus/Examples without error

Phase 2: Semantic Compatibility
         Same model checking results as TLC

Phase 3: Proof Compatibility
         Prove same theorems as TLAPM

Phase 4: Feature Parity
         All TLC/TLAPM features working

Phase 5: Extended Features
         Codegen, better UX, new capabilities
```

### Validation

```bash
# Clone test suite
git clone https://github.com/tlaplus/Examples

# Run comparison tests
cargo run -- test-compat \
    --reference-tlc /path/to/tla2tools.jar \
    --specs Examples/**/*.tla

# Output:
# Parsed: 847/847 (100%)
# Same state count: 842/847 (99.4%)
# Same errors found: 845/847 (99.8%)
# Differences logged to: compat-report.md
```

---

## What We Keep

### From the TLA+ Ecosystem

| Keep | Reason |
|------|--------|
| TLA+ syntax | Standard, well-designed, not broken |
| PlusCal syntax | Popular, useful |
| Proof syntax | Works well, established |
| Module system | Necessary for large specs |
| Standard modules | Integers, Sequences, etc. |

### From External Tools

| Tool | Usage |
|------|-------|
| Z3 | Via z3 crate - excellent SMT solver |
| tree-sitter | Incremental parsing |
| CVC5, Yices | Alternative SMT solvers |
| Lean4 | Optional proof backend |

---

## What We Replace

| Current | Replacement | Reason |
|---------|-------------|--------|
| SANY (Java) | tla-core (Rust) | Unify with proof system |
| TLC (Java) | tla-check (Rust) | Performance, parallelism |
| TLAPM (OCaml) | tla-prove (Rust) | Unify with model checker |
| Zenon (OCaml) | tla-zenon (Rust) | Better integration |
| Isabelle | Lean4 (optional) | Smaller, modern |
| Toolbox (Eclipse) | LSP + any editor | Modern workflow |
| tla2tools.jar | tla binary | Native, fast startup |

---

## Implementation Strategy

### Phase 1: Foundation (Months 1-3)

```
Goal: Parse TLA+, evaluate expressions

Deliverables:
├── tla-core crate
│   ├── Lexer
│   ├── Parser (tree-sitter)
│   ├── AST types
│   ├── Semantic analysis
│   └── Expression evaluator
├── tla-cli skeleton
└── Test suite from tlaplus/Examples
```

### Phase 2: Model Checker (Months 4-7)

```
Goal: Find bugs in specifications

Deliverables:
├── tla-check crate
│   ├── State representation
│   ├── State exploration (parallel)
│   ├── Invariant checking
│   ├── Liveness checking
│   └── Counterexample traces
└── Compatibility tests vs TLC
```

### Phase 3: PlusCal (Months 8-9)

```
Goal: Support algorithm notation

Deliverables:
├── PlusCal parser
├── PlusCal AST
├── Translation to TLA+
└── Label inference
```

### Phase 4: Proof System (Months 10-16)

```
Goal: Prove properties

Deliverables:
├── tla-prove crate
│   ├── Proof syntax parser
│   ├── Obligation generator
│   └── Prover orchestration
├── tla-smt crate (Z3 integration)
├── tla-zenon crate (tableau prover)
├── tla-cert crate (certificate checker)
└── Lean4 integration (optional)
```

### Phase 5: Code Generation (Months 17-20)

```
Goal: Generate verified Rust

Deliverables:
├── tla-codegen crate
│   ├── State struct generation
│   ├── Action enum generation
│   ├── StateMachine impl
│   ├── Kani harnesses
│   └── Verus specs (optional)
├── tla-runtime crate
└── Examples with full pipeline
```

### Phase 6: Polish (Months 21-24)

```
Goal: Production ready

Deliverables:
├── tla-lsp crate
├── Beautiful error messages
├── Documentation
├── WASM build for playground
└── Package distribution
```

---

## Risk Assessment

### Technical Risks

| Risk | Mitigation |
|------|------------|
| Semantic divergence from TLA+ | Extensive compatibility testing |
| Performance worse than TLC | Continuous benchmarking, optimization |
| Proof system incomplete | Start with subset, expand incrementally |
| Lean4 FFI complexity | Keep as subprocess initially |

### Project Risks

| Risk | Mitigation |
|------|------------|
| Scope too large | Clear phase boundaries, usable milestones |
| Single developer | Good docs, clean architecture |
| Funding/time | Phase 2-3 is usable standalone |

### Strategic Risks

| Risk | Mitigation |
|------|------------|
| TLA+ Foundation doesn't adopt | Build community independently |
| Competing projects emerge | Move fast, focus on UX |
| Lean4 changes incompatibly | Isolate Lean4 integration |

---

## Summary

### The Transformation

```
FROM:
  Java (375K) + OCaml (91K) + Isabelle (2M+)
  Three languages, no shared code, 1GB dependency

TO:
  Rust (~150K estimated) + Lean4 (optional, 50MB)
  One language, shared core, lightweight
```

### Key Decisions

1. **Rust everywhere** except optional Lean4 backend
2. **One AST** shared by checker, prover, codegen
3. **Lean4 not Isabelle** for interactive proofs
4. **Code generation** as first-class feature
5. **LSP-first** not IDE-first

### Success Metrics

| Metric | Target |
|--------|--------|
| Parse compatibility | 100% of tlaplus/Examples |
| Model check compatibility | 99%+ same results as TLC |
| Proof compatibility | 95%+ of TLAPM test suite |
| Binary size | <50MB (vs 1GB+ current) |
| Startup time | <100ms (vs seconds for Java) |
| User adoption | 100+ GitHub stars, 10+ production users |

---

*This vision will evolve as implementation progresses. The goal is clear: a modern, unified, lightweight TLA+ toolchain that takes formal verification from design to deployment.*
