# TLA+ Ecosystem Overview

This document provides a comprehensive overview of the current TLA+ ecosystem that TLA2 aims to modernize and unify.

## What is TLA+?

TLA+ (Temporal Logic of Actions) is a formal specification language designed by Leslie Lamport for describing and reasoning about concurrent and distributed systems. It combines:

- **Set theory and first-order logic** for describing system states
- **Temporal logic** for describing how states change over time
- **A simple action-based model** where system behavior is a sequence of states

## Current Toolchain Components

### TLC (Model Checker)

**Language**: Java (~93K lines)
**Repository**: https://github.com/tlaplus/tlaplus

TLC is an explicit-state model checker that:
- Explores all reachable states of a specification
- Checks invariants (properties that must hold in every state)
- Checks temporal properties (liveness, fairness)
- Generates counterexample traces when violations are found

Key algorithms:
- BFS/DFS state exploration
- State fingerprinting (128-bit hash for duplicate detection)
- Symmetry reduction (permutation groups)
- Multi-threaded parallel exploration
- Liveness checking via SCC detection

### SANY (Parser)

**Language**: Java (~52K lines)
**Repository**: Part of tlaplus/tlaplus

SANY (Syntactic Analyzer) handles:
- Lexical analysis of TLA+ source
- Parsing into AST
- Semantic analysis (name resolution, type inference)
- Module system (EXTENDS, INSTANCE)
- PlusCal translation

### TLAPM (Proof Manager)

**Language**: OCaml (~67K lines)
**Repository**: https://github.com/tlaplus/tlapm

TLAPM orchestrates theorem proving:
- Parses proof syntax (BY, SUFFICES, HAVE, etc.)
- Extracts proof obligations
- Dispatches to backend provers
- Caches proof results
- Tracks proof status

### Zenon (Tableau Prover)

**Language**: OCaml (~15K lines)
**Repository**: Embedded in TLAPM

A first-order tableau prover:
- Automated proof search
- Generates proof certificates
- Used as first-line prover before SMT

## External Dependencies

### Z3 (SMT Solver)

Microsoft's SMT solver used by TLAPM for:
- Automated theorem proving
- Quantifier instantiation
- Arithmetic reasoning

### Isabelle

A massive (~2M+ lines) interactive theorem prover used as:
- Escape hatch for proofs too hard for automation
- Proof certificate validation
- Library of formalized mathematics

**Note**: TLA2 replaces Isabelle dependency with optional Lean4 integration.

## Problems with Current Ecosystem

| Problem | Impact |
|---------|--------|
| **Split codebase** | Java (TLC) vs OCaml (TLAPM) means duplicated effort |
| **Two parsers** | Different ASTs can produce different interpretations |
| **Heavy dependencies** | Isabelle is 1GB+ download |
| **Poor error messages** | Legacy tooling with dated UX |
| **No code generation** | Manual translation from spec to implementation |
| **Batch-oriented** | No incremental checking, slow feedback loops |

## TLA2 Solution

TLA2 addresses these problems with:

1. **One codebase in Rust** - Parser, checker, prover share code
2. **One AST** - Single source of truth for specification semantics
3. **Lightweight deps** - Z3 only, optional Lean4
4. **Modern UX** - Beautiful errors, LSP support
5. **Code generation** - TLA+ to verified Rust
6. **Incremental** - Fast feedback during development
