# TLA2: A Modern TLA+ Implementation in Rust

| Director | Status |
|:--------:|:------:|
| MATH | ACTIVE |

> A ground-up reimplementation of the TLA+ formal verification toolchain in Rust, unifying model checking and theorem proving into a cohesive, modern system with verified code generation.

---

## Current State (January 2026)

**TLA2 is approximately 70% feature-complete** compared to TLC, with active development on performance optimization and correctness verification.

### What Works Today

| Capability | Status | Notes |
|------------|--------|-------|
| **Model Checking** | ✅ Production-ready | BFS/DFS, parallel workers, invariants, deadlock detection |
| **Liveness Checking** | ✅ Complete | Tableau, SCC detection, WF/SF fairness, counterexamples |
| **Standard Library** | ✅ 94% coverage | 17/18 TLC modules fully implemented |
| **Checkpoint/Resume** | ✅ Complete | Long-running model checks can be paused and resumed |
| **Scalability** | ✅ Memory-mapped | Mmap fingerprints, trace files for billion-state specs |
| **Simulation Mode** | ✅ Complete | Random trace exploration with seeds |
| **Symmetry Reduction** | ✅ Complete | SYMMETRY keyword, permutation detection |
| **LSP Support** | ✅ Complete | Completion, hover, goto-def, find-refs |

### Test Coverage

- **1,389 unit tests** across all crates
- **301 property-based tests** for evaluator correctness
- **180 Kani formal verification harnesses** (26 fully verified)
- **98 bash integration tests** + **182 Python correctness tests**
- **18+ specs verified** against TLC state counts

### Known Limitations

| Issue | Status | Impact |
|-------|--------|--------|
| Performance on large specs | 🔶 In progress | ~3x slower than TLC on bosco (1M states) |
| `MCRealTimeHourClock` liveness | ✅ Fixed #14 | TLA2 now finds 216 states, matches TLC |
| `bosco` performance | 🔶 #16 open | 3.2x slower (19min vs 6min) - target is <1x |
| Distributed checking | ❌ Not planned | Use TLC for multi-machine exploration |

### Architecture

```
tla2/
├── crates/
│   ├── tla-core/      # Parser, AST, semantic analysis, evaluation (146 tests)
│   ├── tla-check/     # Model checker - TLC replacement (996 tests)
│   ├── tla-prove/     # Proof manager - TLAPM replacement (29 tests)
│   ├── tla-zenon/     # First-order tableau prover (52 tests)
│   ├── tla-cert/      # Proof certificate checker (39 tests)
│   ├── tla-smt/       # Z3 integration (28 tests)
│   ├── tla-codegen/   # Rust code generation (59 tests)
│   ├── tla-runtime/   # Runtime support for generated code (26 tests)
│   ├── tla-lsp/       # Language Server Protocol (14 tests)
│   └── tla-cli/       # Command-line interface
└── docs/              # Design documents and proposals
```

---

## Future Ideal State

### Vision: AI Writes Proven Code

The end goal is a world where **AI writes code, it compiles, and it's mathematically proven correct**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI Writes Proven Code                           │
│                                                                         │
│   "Too hard" → AI writes the proofs                                     │
│   "Too slow" → Modern hardware + optimized tools                        │
│   "Too much friction" → Compile = verify (automatic)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────┬───────────┬───────┴───────┬───────────┬─────────────┐
    ▼           ▼           ▼               ▼           ▼             ▼
┌───────┐  ┌────────┐  ┌─────────┐  ┌────────────┐  ┌───────┐  ┌───────────┐
│ TLA2  │  │ Lean5  │  │   Z4    │  │   tRust    │  │ Kani+ │  │α-β-CROWN │
│       │  │        │  │         │  │            │  │       │  │          │
│Design │  │ Proofs │  │  SMT    │  │ Verified   │  │(temp) │  │ Neural   │
│ spec  │  │        │  │ solving │  │   Rust     │  │       │  │ Networks │
└───────┘  └────────┘  └─────────┘  └────────────┘  └───────┘  └───────────┘
```

### Target Milestones

| Milestone | Target | Metric |
|-----------|--------|--------|
| **TLC Performance Parity** | Q1 2026 | Within 2x of TLC on all benchmarks |
| **Full TLC Feature Parity** | Q2 2026 | 100% of tlaplus/Examples pass |
| **Connectivity Prover MVP** | Q2 2026 | Prove integration completeness for Rust apps |
| **tRust Integration** | Q3 2026 | Compile-time verification in rustc fork |
| **Kani Full Verification** | Q4 2026 | All 180 harnesses verified |

---

## The Connectivity Prover: Proving Software Is Actually Connected

### The Problem

AI coding assistants produce code that:
- ✅ Has thousands of passing unit tests
- ✅ Has high code coverage metrics
- ❌ **Fails immediately when launched**

Root cause: Tests verify components in isolation. Nothing verifies that components are actually connected into a working system.

```
AI: "Done! 2,847 tests passing!"
Human: ./app
App: Error: MCP not configured. Crash.

The path from main() to the feature was never wired up.
Tests can't catch this. Tests can be gamed.
```

### The Solution: Mathematical Proof of Connectivity

TLA2's **Connectivity Prover** will mathematically prove that:
1. Every public API function is reachable from entry points
2. External dependencies are explicitly declared and accounted for
3. The system works under specified service availability conditions

**Tests can lie. Proofs cannot.**

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TLA2 Connectivity Prover                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────┐     ┌────────────┐     ┌────────────┐               │
│  │  Rust     │     │   Call     │     │   TLA+     │               │
│  │  Source   │────▶│   Graph    │────▶│   Model    │               │
│  │           │     │ Extraction │     │ Generation │               │
│  └───────────┘     └────────────┘     └────────────┘               │
│                                              │                      │
│                                              ▼                      │
│                                       ┌────────────┐               │
│                                       │  tla-check │               │
│                                       │   Engine   │               │
│                                       └────────────┘               │
│                                              │                      │
│                            ┌─────────────────┼─────────────────┐   │
│                            ▼                 ▼                 ▼   │
│                       ┌────────┐       ┌──────────┐     ┌───────┐ │
│                       │ PASSED │       │  FAILED  │     │ Report│ │
│                       │ Build  │       │  Build   │     │       │ │
│                       │ continues      │  stops   │     │       │ │
│                       └────────┘       └──────────┘     └───────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**Generated TLA+ Model (from call graph):**

```tla
---- MODULE Connectivity ----
VARIABLE pc, available   \* program counter + service availability

Init == pc \in EntryPoints /\ available \in [Services -> BOOLEAN]

Next ==
    \/ (pc = "main" /\ pc' = "init_services" /\ UNCHANGED available)
    \/ (pc = "init_services" /\ available["mcp"] /\ pc' = "connect_mcp" /\ ...)
    \/ ... \* Auto-generated from call graph

\* THEOREM: All public APIs are reachable
AllGoalsReachable == \A g \in PublicAPIs: <>(pc = g)

\* THEOREM: Health check works even with services down
HealthCheckRobust == [](\A s \in Services: ~available[s]) => <>(pc = "health_check")
====
```

### Sample Output

```
══════════════════════════════════════════════════════════════════════
 TLA2 CONNECTIVITY PROOF
 Binary: target/debug/myapp
 Functions: 1,247 | Entry points: 3 | Goals: 50
══════════════════════════════════════════════════════════════════════

THEOREM: AllGoalsReachable
  Status: FAILED

  Unreachable goals:
    ✗ export_csv         src/export.rs:45
    ✗ generate_report    src/reports.rs:12

  Counterexample for 'export_csv':
    No path exists from any entry point.
    Nearest reachable: handle_request (src/api.rs:89)
    Missing: handle_request never calls route_export()

    Suggested fix:
      Add: handle_request() → route_export() for "/export" routes

──────────────────────────────────────────────────────────────────────
BUILD FAILED: 2 public APIs unreachable
══════════════════════════════════════════════════════════════════════
```

### Usage (Planned)

```bash
# Prove connectivity (build fails if proof fails)
tla prove-connectivity ./target/debug/myapp

# CI mode with threshold
tla prove-connectivity ./target/debug/myapp --ci --require 100%

# Integration with cargo (via tRust)
cargo build --features tla-connectivity
```

### Implementation Roadmap

| Phase | Work | Estimate |
|-------|------|----------|
| **Phase 1** | Call graph extraction from MIR/HIR | 2-3 commits |
| **Phase 2** | TLA+ model generation from call graph | 3-4 commits |
| **Phase 3** | Engine optimization for graph reachability | 2-3 commits |
| **Phase 4** | tRust compiler integration | 2-3 commits |

See [PROPOSAL_connectivity_prover_2026-01-06.md](docs/proposals/PROPOSAL_connectivity_prover_2026-01-06.md) for full design.

---

## Vision

TLA2 reimagines the TLA+ ecosystem as a unified, modern toolchain:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TLA+ Specification                              │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────────────────┐
        ▼                           ▼                           ▼           ▼
   ┌─────────┐               ┌─────────┐               ┌─────────┐    ┌─────────┐
   │  Check  │               │  Prove  │               │ Codegen │    │   LSP   │
   │         │               │         │               │         │    │         │
   │ Model   │               │ Theorem │               │ Rust +  │    │ Editor  │
   │ checker │               │ prover  │               │ Verify  │    │ support │
   └─────────┘               └─────────┘               └─────────┘    └─────────┘
        │                           │                       │
        │                           │                       ▼
        │                           │               ┌───────────────┐
        │                           │               │ Verified Rust │
        │                           │               │ Implementation│
        │                           │               └───────────────┘
        │                           │                       │
        └───────────────────────────┼───────────────────────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │  Production   │
                            │    System     │
                            └───────────────┘
```

## Why TLA2?

The current TLA+ ecosystem has accumulated 20+ years of technical debt:

| Problem | Current State | TLA2 Solution |
|---------|---------------|---------------|
| **Split brain** | Java (TLC) + OCaml (TLAPM) | One language: Rust |
| **Two parsers** | Different ASTs, different bugs | One parser, one AST |
| **Heavy deps** | 1GB Isabelle download | Lightweight, optional Lean4 |
| **No code gen** | Manual translation to code | Verified Rust generation |
| **Dated UX** | Eclipse IDE, batch workflows | LSP-first, incremental |

## Architectural Thesis: Best-in-Class, Not Ensembles

The traditional approach to formal verification has been **ensemble systems** - routers that dispatch to multiple partial tools, hoping one works:

```
Traditional TLAPM (ensemble/router):

                    ┌─────────────────┐
                    │     TLAPM       │
                    │   "try them all"│
                    └────────┬────────┘
                             │
    ┌──────┬──────┬──────┬───┴───┬──────┬──────┐
    ▼      ▼      ▼      ▼       ▼      ▼      ▼
  Zenon  Isa    Z3    CVC4   Yices  SPASS  LS4
    │      │      │      │       │      │      │
    └──────┴──────┴──────┴───────┴──────┴──────┘
              "guess and check"
```

**We reject this approach.** Instead, we build **unified best-in-class tools** for each capability:

```
Our approach (best-in-class per domain):

┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│      TLA2       │   │     Lean5       │   │       Z4        │
│                 │   │                 │   │                 │
│  Model Checking │   │ Theorem Proving │   │   SMT Solving   │
│  (best TLC)     │   │  (best prover)  │   │   (best SMT)    │
└─────────────────┘   └─────────────────┘   └─────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    Clean interfaces between
                    best-in-class tools
```

**Why this matters:**

| Ensemble Approach | Best-in-Class Approach |
|-------------------|------------------------|
| Route to 6 mediocre tools | One excellent tool per job |
| "Try Zenon, then Z3, then CVC..." | "Use the tool that handles this" |
| Complex dispatcher logic | Simple, direct interfaces |
| Partial coverage per tool | Complete coverage per domain |
| Hope something works | Know it will work |

**The tools:**

| Domain | Tool | Status |
|--------|------|--------|
| Model Checking | **TLA2** | This project - TLC replacement |
| Theorem Proving | **Lean5** | Companion project - next-gen Lean4 |
| SMT Solving | **Z4** | Companion project - next-gen Z3 |

**Current TLAPM compatibility** (Phases 1-5 in the roadmap) is scaffolding to pass existing TLA+ tests and validate our understanding of proof obligations. Long-term, theorem proving migrates to Lean5, and the ensemble/dispatcher complexity disappears.

## The Complete Vision: AI Writes Proven Code

The end goal is: **AI writes code. It compiles. It's proven correct.**

This requires a unified verification stack where each tool is best-in-class:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI Writes Proven Code                           │
│                                                                         │
│   "Too hard" → AI writes the proofs                                     │
│   "Too slow" → Modern hardware + optimized tools                        │
│   "Too much friction" → Compile = verify (automatic)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────┬───────────┬───────┴───────┬───────────┬─────────────┐
    ▼           ▼           ▼               ▼           ▼             ▼
┌───────┐  ┌────────┐  ┌─────────┐  ┌────────────┐  ┌───────┐  ┌───────────┐
│ TLA2  │  │ Lean5  │  │   Z4    │  │   tRust    │  │ Kani+ │  │α-β-CROWN │
│       │  │        │  │         │  │            │  │       │  │          │
│Design │  │ Proofs │  │  SMT    │  │ Verified   │  │(temp) │  │ Neural   │
│ spec  │  │        │  │ solving │  │   Rust     │  │       │  │ Networks │
└───────┘  └────────┘  └─────────┘  └────────────┘  └───────┘  └───────────┘
```

**The tools:**

| Domain | Tool | Description |
|--------|------|-------------|
| Design Verification | **TLA2** | Model checking for specifications (this project) |
| Theorem Proving | **Lean5** | Next-gen Lean4 for deep proofs |
| SMT Solving | **Z4** | Next-gen Z3 for constraint solving |
| Code Verification | **tRust** | Rust fork with built-in verification |
| Code Verification | **Kani+** | Enhanced Kani (temporary, exploring) |
| Neural Networks | **α-β-CROWN** | NN robustness and safety verification |

**tRust philosophy:**
- Minimal or no annotations in the code itself
- Proofs exist but are clean/separate, not inline macro noise
- Compilation = verification (automatic, not opt-in)
- The code stays readable; the proofs provide guarantees

**Kani+ is exploratory** - it may be a stepping stone to tRust, or the approaches may converge. The key insight is that AI can generate code satisfying whatever verification the compiler requires.

## Project Goals

1. **Unified Implementation**: One codebase, one language, one semantics
2. **Modern UX**: Beautiful errors, LSP support, fast incremental checking
3. **Verified Code Generation**: TLA+ → Rust with Kani/Verus verification
4. **Practical Refinement**: Design → Code → Deploy with machine-checked steps
5. **Lightweight**: No 1GB dependencies for basic usage
6. **Best-in-Class**: Excellence in model checking, not mediocrity across many tools

## Documentation

| Document | Description |
|----------|-------------|
| [Ecosystem Overview](docs/ecosystem-overview.md) | Understanding the current TLA+ landscape |
| [Modernization Vision](docs/modernization-vision.md) | The Rust + Lean4 architecture |
| [Project Plan](docs/project-plan.md) | Detailed implementation phases |
| [Refinement Design](docs/refinement-design.md) | Spec → Verified Code pipeline |
| [Verification Stack](docs/verification-stack.md) | Kani vs Verus: two-level verification |

## Architecture

```
tla2/
├── crates/
│   ├── tla-core/          # Parser, AST, semantic analysis, evaluation
│   ├── tla-check/         # Model checker (replaces TLC)
│   ├── tla-prove/         # Proof manager (replaces TLAPM)
│   ├── tla-codegen/       # Rust code generation
│   ├── tla-runtime/       # Runtime support for generated code
│   ├── tla-smt/           # SMT solver integration (Z3, CVC5)
│   ├── tla-zenon/         # Tableau prover (Rust implementation)
│   ├── tla-cert/          # Proof certificate checker
│   ├── tla-lsp/           # Language Server Protocol
│   └── tla-cli/           # Command-line interface
├── docs/                  # Documentation
├── examples/              # Example specifications and implementations
├── tests/                 # Integration tests
└── benches/               # Benchmarks
```

## Quick Start

```bash
# Install from source
cargo install --path crates/tla-cli

# Optional: enable the `prove` command (requires Z3)
cargo install --path crates/tla-cli --features prove

# Check a specification
tla check MySpec.tla

# Prove properties (requires building with `--features prove`)
tla prove MySpec.tla

# Generate verified Rust code
tla codegen MySpec.tla -o src/generated/

# Run all verification
cargo test && cargo kani
```

## The Pipeline

```
1. DESIGN      Write TLA+ spec, model check, prove properties
                     │
2. GENERATE    tla codegen → Rust state machine + tests + Kani harnesses
                     │
3. IMPLEMENT   Write I/O layer using generated skeleton
                     │
4. VERIFY      cargo test → cargo kani → cargo verus (optional)
                     │
5. DEPLOY      Runtime spec monitoring in production
```

## Two-Level Verification

TLA2 provides **bidirectional verification** - proving correctness at both the design and code levels:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DESIGN LEVEL                                     │
│                                                                          │
│   TLA+ Specification                                                     │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────┐                                                        │
│   │  tla-check  │  ← Model checker explores ALL states                   │
│   │  (TLA2)     │    "Does my design have bugs?"                         │
│   └─────────────┘                                                        │
│         │                                                                │
│         │  Design is correct ✓                                           │
│         │                                                                │
└─────────┼────────────────────────────────────────────────────────────────┘
          │
          │  tla-codegen
          │
┌─────────┼────────────────────────────────────────────────────────────────┐
│         ▼                        CODE LEVEL                              │
│   ┌─────────────┐                                                        │
│   │  Generated  │                                                        │
│   │  Rust Code  │                                                        │
│   └──────┬──────┘                                                        │
│          │                                                               │
│    ┌─────┴─────┬─────────────┬─────────────┐                            │
│    │           │             │             │                            │
│    ▼           ▼             ▼             ▼                            │
│ ┌───────┐ ┌─────────┐ ┌───────────┐ ┌──────────┐                       │
│ │proptest│ │  Kani   │ │  Verus    │ │ Runtime  │                       │
│ │       │ │         │ │ (optional)│ │ Monitor  │                       │
│ │Random │ │Bounded  │ │Full       │ │Production│                       │
│ │testing│ │model    │ │proofs     │ │checking  │                       │
│ │       │ │checking │ │           │ │          │                       │
│ └───────┘ └─────────┘ └───────────┘ └──────────┘                       │
│                                                                          │
│   "Does my CODE match my DESIGN?"                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Verification Tools

| Tool | Level | What It Proves |
|------|-------|----------------|
| **tla-check** | Design | "Is my TLA+ spec correct?" (explores all states) |
| **proptest** | Code | "Does code work on random inputs?" |
| **Kani** | Code | "Does code preserve invariants up to bound N?" |
| **Verus** | Code | "Does code ALWAYS preserve invariants?" (full proof) |
| **Runtime** | Production | "Is my live system following the spec?" |

### Generated Kani Harnesses

Code generation produces Kani verification harnesses automatically:

```rust
#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_invariant_inductive() {
        // Pick ANY state satisfying invariant
        let state: State = kani::any();
        kani::assume(MySpec::invariant(&state));

        // Pick ANY action
        let action: Action = kani::any();

        // If action is enabled, invariant MUST be preserved
        if let Some(next_state) = MySpec::next(&state, &action) {
            kani::assert(
                MySpec::invariant(&next_state),
                "Invariant preservation"
            );
        }
    }
}
```

This proves: **if the invariant holds before a step, it holds after ANY step** - an inductive invariant proof.

### The Closed Loop

Traditional approach:
```
Spec (paper) → Manual coding → Hope it's right → Production bugs
```

TLA2 approach:
```
TLA+ Spec → Model check → Generate code → Verify code → Monitor production
     ✓           ✓              ✓              ✓              ✓
  Design     No design      Automated     Code matches    Stays correct
  reviewed   bugs           translation   spec (proved)   at runtime
```

The gap between "spec" and "code" is **machine-checked**, not hoped.

## Beautiful Error Messages

TLA2 provides rich, colorful error diagnostics using [ariadne](https://crates.io/crates/ariadne):

```
Error: syntax error: expected expression, found Some(ModuleEnd)
   ╭─[test.tla:6:1]
   │
 6 │ ====
   │ ──┬─
   │   ╰───here
───╯
```

Errors show:
- The exact source location with line and column
- The problematic code highlighted in context
- Clear, actionable error messages

## Detailed Status

See **Current State** section above for summary. For full audit details, see [AUDIT.md](docs/AUDIT.md).

| Component | Status | Tests |
|-----------|--------|-------|
| tla-core | ✓ Parser, AST, evaluation, diagnostics | 146 tests |
| tla-check | ✓ Sequential + parallel model checking, liveness checking, checkpoint/resume | 996 tests |
| tla-smt | ✓ Z3 integration (requires Z3) | 28 tests |
| tla-prove | ✓ Proof manager with SMT/Zenon backends | 29 tests |
| tla-zenon | ✓ First-order tableau prover, certificate generation | 52 tests |
| tla-cert | ✓ Proof certificate checker with JSON serialization, alpha-equivalence | 39 tests |
| tla-lsp | ✓ Full LSP (completion, hover, goto-def, find-refs) | 14 tests |
| tla-runtime | ✓ StateMachine trait, runtime types, model checker helper | 26 tests |
| tla-codegen | ✓ Type inference, full expression codegen, e2e execution | 59 tests |

**Open Issues:** 13 ([view on GitHub](https://github.com/dropbox/dMATH/tla2/issues))

See [Project Plan](docs/project-plan.md) for detailed roadmap.

## Building

```bash
# Clone the repository
git clone https://github.com/dropbox/dMATH/tla2
cd tla2

# Build all crates
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench

# Build optimized release binary
cargo build --release -p tla-cli
```

See [RELEASE.md](RELEASE.md) for detailed release build instructions and cross-compilation.

## Development Setup

### Z3 SMT Solver

The `tla-smt` and `tla-prove` crates require the Z3 SMT solver. The `z3` Rust crate uses `z3-sys` which builds Z3 bindings at compile time and needs the Z3 headers.

#### macOS (Homebrew)

```bash
brew install z3

# Set environment variables (add to ~/.zshrc or ~/.bashrc)
export Z3_SYS_Z3_HEADER="$(brew --prefix z3)/include/z3.h"
export LIBRARY_PATH="$(brew --prefix z3)/lib"
```

#### Ubuntu/Debian

```bash
sudo apt-get install libz3-dev
```

#### Arch Linux

```bash
sudo pacman -S z3
```

#### Windows

Download Z3 from [GitHub releases](https://github.com/Z3Prover/z3/releases) and add the `bin` directory to PATH.

### Enabling Prove (Z3)

By default, `tla-cli` builds without the `prove` command (no Z3 required). To enable `prove`:

```bash
cargo build -p tla-cli --features prove
```

The default build gives you access to:
- `tla parse` - Parse TLA+ files
- `tla check` - Model check specifications
- `tla codegen` - Generate Rust code
- `tla lsp` - Start language server
- `tla fmt` - Format TLA+ files
- `tla ast` - Dump AST

You can also build individual crates independently:

```bash
# These crates don't require Z3
cargo build -p tla-core
cargo build -p tla-check
cargo build -p tla-runtime
cargo build -p tla-codegen
cargo build -p tla-lsp
cargo build -p tla-zenon
```

### Model Checking with Limits

For unbounded specifications that would run indefinitely, use exploration limits:

```bash
# Limit to 1000 states
tla check MySpec.tla --max-states 1000

# Limit BFS depth to 10
tla check MySpec.tla --max-depth 10

# Limits work with parallel mode too
tla check MySpec.tla --max-states 1000 --workers 4
```

### Progress Reporting

For long-running model checks, enable progress reporting to see real-time status:

```bash
# Show progress during model checking
tla check MySpec.tla --progress

# Progress works with all other options
tla check MySpec.tla --progress --workers 8 --max-states 100000
```

Progress output shows:
- **States found**: Number of distinct states discovered
- **Current depth**: BFS depth currently being explored
- **Transitions**: Total state transitions examined
- **States/sec**: State exploration rate
- **Elapsed time**: Time elapsed since model checking started

### Checkpoint/Resume

For long-running model checks, use checkpoints to save progress and resume later:

```bash
# Save checkpoints periodically (every 100,000 states by default)
tla check MySpec.tla --checkpoint checkpoint.json

# Customize checkpoint interval
tla check MySpec.tla --checkpoint checkpoint.json --checkpoint-interval 50000

# Resume from a checkpoint
tla check MySpec.tla --resume checkpoint.json
```

### Scalable Storage

For state spaces that exceed available RAM, use memory-mapped storage:

```bash
# Use memory-mapped fingerprint storage (capacity in number of states)
tla check MySpec.tla --mmap-fingerprints 100000000

# Specify a directory for the backing file
tla check MySpec.tla --mmap-fingerprints 100000000 --mmap-dir /tmp/tla2

# Combine with trace file mode for counterexample reconstruction
tla check MySpec.tla --mmap-fingerprints 100000000 --trace-file trace.bin
```

Memory-mapped storage provides:
- Capacity warnings at 80% and 95% of limit
- Error detection for dropped states (prevents false verifications)
- Automatic disk paging for large state spaces

### Performance

TLA2 is designed for high performance model checking with **adaptive parallelism**. By default (`--workers 0`), TLA2 automatically selects the optimal strategy based on spec characteristics:

```bash
# Auto mode (recommended) - adapts to spec size
tla check MySpec.tla

# Force sequential (for small specs or debugging)
tla check MySpec.tla --workers 1

# Force parallel with specific worker count
tla check MySpec.tla --workers 4
```

**Adaptive Strategy Selection:**
- Runs a pilot phase to analyze initial states and branching factor
- Estimates total state space size
- Selects sequential for small specs (<1,000 estimated states)
- Selects parallel with optimal worker count for larger specs

**Benchmarks** (Apple M-series machine):

| Spec | States | Sequential | Parallel (4 workers) | Speedup |
|------|--------|------------|---------------------|---------|
| SmallCounter | 10 | 16 us | 66 us | 0.2x (overhead) |
| MediumCounter | ~100 | 17 us | 70 us | 0.2x (overhead) |
| LargeCounter | ~1,000 | 775 us | 342 us | 2.3x |
| VeryLarge | ~10,000 | 3.57 ms | 1.33 ms | **2.7x** |

**Key insights:**
- For small state spaces (<1,000 states), sequential checking is faster due to parallel overhead
- For larger state spaces (>1,000 states), parallel checking provides significant speedup
- 2 workers often outperform 4+ workers for mid-size specs due to synchronization overhead
- Adaptive mode (`--workers 0`) automatically makes the right choice

Run benchmarks locally:
```bash
cargo bench -p tla-check
```

## Contributing

This project is in early development. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT OR Apache-2.0

## Acknowledgments

TLA2 builds on decades of work by:
- Leslie Lamport (TLA+ creator)
- The TLA+ Foundation
- Microsoft Research (TLC, TLAPS contributors)
- INRIA (TLAPS, Zenon)
- The Rust community (Kani, Verus, z3 crate)
- The Lean community (Lean4, Mathlib)

We aim to carry this legacy forward into a new generation of tooling.
