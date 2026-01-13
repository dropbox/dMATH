//! Unified Specification Language (USL) for `DashProve`
//!
//! USL is a specification language that compiles to multiple verification backends:
//! - LEAN 4 (theorems, invariants, refinements)
//! - TLA+ (temporal properties)
//! - Kani (contracts)
//! - Alloy (bounded model checking)
//!
//! See docs/DESIGN.md for the full grammar specification.
//!
//! # Example
//!
//! ```rust
//! use dashprove_usl::{parse, typecheck, compile_to_lean, compile_to_tlaplus};
//!
//! let input = r#"
//!     theorem test {
//!         forall x: Bool . x or not x
//!     }
//! "#;
//!
//! let spec = parse(input).expect("parse failed");
//! let typed_spec = typecheck(spec).expect("type check failed");
//!
//! // Compile to LEAN 4
//! let lean_output = compile_to_lean(&typed_spec);
//! assert!(lean_output.code.contains("theorem test"));
//!
//! // Compile to TLA+
//! let tla_output = compile_to_tlaplus(&typed_spec);
//! assert!(tla_output.code.contains("test =="));
//! ```

pub mod ast;
pub mod compile;
pub mod dependency;
#[allow(missing_docs)] // pest derive generates undocumented Rule enum
pub mod grammar;
pub mod typecheck;

// Re-exports
pub use ast::*;
pub use compile::{
    compile_invariant_to_closure, compile_to_alloy, compile_to_coq, compile_to_dafny,
    compile_to_isabelle, compile_to_kani, compile_to_lean, compile_to_rust_closures,
    compile_to_smtlib2, compile_to_smtlib2_with_logic, compile_to_tlaplus,
    generate_monitor_registration, suggest_tactics_for_property, AlloyCompiler, CompileError,
    CompiledRustClosure, CompiledSpec, CoqCompiler, DafnyCompiler, IsabelleCompiler, KaniCompiler,
    Lean4Compiler, RustClosureCompiler, SmtLib2Compiler, TlaPlusCompiler,
};
pub use dependency::{DependencyGraph, PropertyDependencies, SpecDiff};
pub use grammar::{parse, ParseError};
pub use typecheck::{typecheck, CheckedType, TypeChecker, TypeEnv, TypeError, TypedSpec};
