//! Compilation from USL to backend-specific formats
//!
//! This module provides compilation functions for multiple verification backends:
//! - TLA+ (temporal properties, invariants)
//! - LEAN 4 (theorems, invariants, refinements)
//! - Kani (contracts)
//! - Alloy (bounded model checking)
//! - Isabelle/HOL (theorem proving)
//! - Coq (theorem proving)
//! - Dafny (verification)
//! - SMT-LIB2 (Z3, CVC5)
//! - Platform API (static checkers for external API constraints)
//! - Rust Closures (runtime monitors for dashprove-monitor)

// Allow methods that use &self only for recursive calls - this is a valid pattern
// for maintaining consistent API and potential future state access
#![allow(clippy::only_used_in_recursion)]

mod alloy;
mod coq;
mod dafny;
mod isabelle;
mod kani;
mod lean4;
mod platform_api;
mod rust_closure;
mod smtlib2;
mod tlaplus;

use crate::ast::Property;

// Re-export all compilers and public functions
pub use alloy::{compile_to_alloy, AlloyCompiler};
pub use coq::{compile_to_coq, CoqCompiler};
pub use dafny::{compile_to_dafny, DafnyCompiler};
pub use isabelle::{compile_to_isabelle, IsabelleCompiler};
pub use kani::{compile_to_kani, KaniCompiler};
pub use lean4::{compile_to_lean, Lean4Compiler};
pub use platform_api::{compile_to_platform_api, PlatformApiCompiler};
pub use rust_closure::{
    compile_invariant_to_closure, compile_to_rust_closures, generate_monitor_registration,
    CompiledRustClosure, RustClosureCompiler,
};
pub use smtlib2::{compile_to_smtlib2, compile_to_smtlib2_with_logic, SmtLib2Compiler};
pub use tlaplus::{compile_to_tlaplus, TlaPlusCompiler};

/// Compiled output for a specific backend
#[derive(Debug, Clone)]
pub struct CompiledSpec {
    /// Backend identifier
    pub backend: String,
    /// Generated code
    pub code: String,
    /// Module name (if applicable)
    pub module_name: Option<String>,
    /// Required imports/extends
    pub imports: Vec<String>,
}

/// Compilation error
#[derive(Debug, Clone, thiserror::Error)]
pub enum CompileError {
    /// Property type not supported by target backend
    #[error("Unsupported property type {property_type} for backend {backend}")]
    UnsupportedProperty {
        /// Type of property attempted
        property_type: String,
        /// Target backend name
        backend: String,
    },
    /// Expression cannot be compiled to target backend
    #[error("Unsupported expression: {0}")]
    UnsupportedExpression(String),
    /// Type information required but not available
    #[error("Missing type information: {0}")]
    MissingTypeInfo(String),
}

/// Get suggested LEAN 4 tactics for a property based on its structure
///
/// This uses the compiler's heuristics to suggest tactics like:
/// - `decide` for simple boolean expressions
/// - `omega` for linear arithmetic
/// - `ring` for polynomial expressions
/// - `simp` for simplification
/// - `intro` for forall/implications
/// - etc.
#[must_use]
pub fn suggest_tactics_for_property(property: &Property) -> Vec<String> {
    let compiler = Lean4Compiler::new("Suggestions");

    let tactic = match property {
        Property::Theorem(thm) => compiler.suggest_tactic(&thm.body),
        Property::Invariant(inv) => compiler.suggest_tactic(&inv.body),
        Property::Refinement(ref_) => {
            let abs_tactic = compiler.suggest_tactic(&ref_.abstraction);
            let sim_tactic = compiler.suggest_tactic(&ref_.simulation);
            format!("constructor\n  · {abs_tactic}\n  · {sim_tactic}")
        }
        Property::Security(security) => compiler.suggest_tactic(&security.body),
        // Composed theorems use tactics based on their body
        Property::Composed(comp) => compiler.suggest_tactic(&comp.body),
        // Temporal, Contract, Probabilistic, PlatformApi, Bisimulation, Version, Capability don't use LEAN tactics
        Property::Temporal(_)
        | Property::Contract(_)
        | Property::Probabilistic(_)
        | Property::Semantic(_)
        | Property::PlatformApi(_)
        | Property::Bisimulation(_)
        | Property::Version(_)
        | Property::Capability(_)
        | Property::DistributedInvariant(_)
        | Property::DistributedTemporal(_)
        | Property::ImprovementProposal(_)
        | Property::VerificationGate(_)
        | Property::Rollback(_) => {
            return vec![];
        }
    };

    // Parse the tactic string into individual tactics
    tactic
        .split('\n')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(std::string::ToString::to_string)
        .collect()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, typecheck, typecheck::TypedSpec};

    fn compile_usl(input: &str) -> TypedSpec {
        let spec = parse(input).expect("parse failed");
        typecheck(spec).expect("typecheck failed")
    }

    #[test]
    fn test_tlaplus_compile_simple_invariant() {
        let input = r#"
            invariant test {
                forall x: Node . x == x
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_tlaplus(&typed);

        assert!(compiled.code.contains("---- MODULE USLSpec ----"));
        assert!(compiled.code.contains("test =="));
        assert!(compiled.code.contains("\\A x"));
        assert!(compiled.code.contains("===="));
    }

    #[test]
    fn test_tlaplus_compile_temporal() {
        let input = r#"
            temporal no_deadlock {
                always(eventually(done))
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_tlaplus(&typed);

        assert!(compiled.code.contains("no_deadlock =="));
        assert!(compiled.code.contains("[]"));
        assert!(compiled.code.contains("<>"));
    }

    #[test]
    fn test_tlaplus_compile_exists_in() {
        let input = r#"
            temporal liveness {
                always(exists agent in agents . enabled(agent))
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_tlaplus(&typed);

        assert!(compiled.code.contains("\\E agent \\in agents:"));
    }

    #[test]
    fn test_lean_compile_theorem() {
        let input = r#"
            theorem excluded_middle {
                forall p: Bool . p or not p
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        assert!(compiled.code.contains("namespace USLSpec"));
        assert!(compiled.code.contains("theorem excluded_middle"));
        assert!(compiled.code.contains("∀ p: Bool"));
        assert!(compiled.code.contains("∨"));
        assert!(compiled.code.contains("¬"));
    }

    #[test]
    fn test_lean_compile_type_def() {
        let input = r#"
            type Node = { id: String, value: Int }

            theorem test {
                forall n: Node . n == n
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        assert!(compiled.code.contains("structure Node where"));
        assert!(compiled.code.contains("id : String"));
        assert!(compiled.code.contains("value : Int"));
    }

    #[test]
    fn test_lean_compile_refinement() {
        let input = r#"
            refinement optimized refines base {
                abstraction { to_base(opt) == base }
                simulation { forall s: State . step(to_base(opt), s) == to_base(step(opt, s)) }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        assert!(compiled.code.contains("Refinement: optimized refines base"));
        assert!(compiled.code.contains("optimized_abstraction"));
        assert!(compiled.code.contains("optimized_simulation"));
    }

    #[test]
    fn test_kani_compile_contract() {
        let input = r#"
            contract add(x: Int, y: Int) -> Int {
                requires { x >= 0 }
                requires { y >= 0 }
                ensures { result >= x }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        assert!(compiled.code.contains("#[kani::proof]"));
        assert!(compiled.code.contains("fn verify_add()"));
        assert!(compiled.code.contains("kani::any()"));
        assert!(compiled.code.contains("kani::assume"));
        assert!(compiled.code.contains("kani::assert"));
    }

    #[test]
    fn test_kani_compile_contract_with_types() {
        let input = r#"
            contract Graph::add_node(self: Graph, node: Node) -> Result<()> {
                requires { not self.contains(node.id) }
                ensures { self.contains(node.id) }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        assert!(compiled.code.contains("verify_Graph_add_node"));
        assert!(compiled.code.contains("let self: Graph"));
        assert!(compiled.code.contains("let node: Node"));
    }

    #[test]
    fn test_kani_compile_contract_with_error_postconditions() {
        let input = r#"
            contract divide(x: Int, y: Int) -> Result<Int> {
                requires { y != 0 }
                ensures { result * y == x }
                ensures_err { y == 0 }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        assert!(compiled.code.contains("match result"));
        assert!(compiled.code.contains("Ok(result) => {"));
        assert!(compiled.code.contains("Err(result) => {"));
        assert!(compiled.code.contains("postcondition_0"));
        assert!(compiled.code.contains("error_postcondition_0"));
    }

    #[test]
    fn test_alloy_compile_invariant() {
        let input = r#"
            type Node = { id: String }

            invariant unique_ids {
                forall n1: Node, n2: Node .
                    n1.id == n2.id implies n1 == n2
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_alloy(&typed);

        assert!(compiled.code.contains("module USLSpec"));
        assert!(compiled.code.contains("sig Node"));
        // Invariants are now compiled as assertions with check commands
        assert!(compiled.code.contains("assert unique_ids"));
        assert!(compiled.code.contains("check unique_ids for 5"));
        assert!(compiled.code.contains("all n1: Node"));
    }

    #[test]
    fn test_tlaplus_compile_implies_and_or() {
        let input = r#"
            theorem logic_test {
                forall a: Bool, b: Bool .
                    (a implies b) implies (not a or b)
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_tlaplus(&typed);

        assert!(compiled.code.contains("=>"));
        assert!(compiled.code.contains("\\/"));
        assert!(compiled.code.contains("~"));
    }

    #[test]
    fn test_lean_compile_security() {
        let input = r#"
            security isolation {
                forall t1: Tenant, t2: Tenant .
                    t1 != t2 implies not can_observe(t1, actions(t2))
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        assert!(compiled.code.contains("theorem isolation"));
        assert!(compiled.code.contains("≠"));
    }

    #[test]
    fn test_backends_return_correct_names() {
        let input = r#"
            theorem test { true }
        "#;
        let typed = compile_usl(input);

        assert_eq!(compile_to_tlaplus(&typed).backend, "TLA+");
        assert_eq!(compile_to_lean(&typed).backend, "LEAN4");
        assert_eq!(compile_to_kani(&typed).backend, "Kani");
        assert_eq!(compile_to_alloy(&typed).backend, "Alloy");
    }

    #[test]
    fn test_tlaplus_field_access() {
        let input = r#"
            type Graph = { nodes: Set<Node> }

            invariant test {
                forall g: Graph . forall n in (g.nodes) . n == n
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_tlaplus(&typed);

        assert!(compiled.code.contains("g.nodes"));
    }

    #[test]
    fn test_kani_arithmetic() {
        let input = r#"
            contract compute(x: Int, y: Int) -> Int {
                requires { y != 0 }
                ensures { result == x + y * 2 }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        assert!(compiled.code.contains("+"));
        assert!(compiled.code.contains("*"));
    }

    // ============================================================================
    // LEAN TACTIC SUGGESTION TESTS
    // ============================================================================

    #[test]
    fn test_lean_tactic_trivial_true() {
        let input = r#"
            theorem trivial_test { true }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // true literal should generate "trivial" tactic
        assert!(
            compiled.code.contains("trivial"),
            "Expected 'trivial' tactic for true literal"
        );
    }

    #[test]
    fn test_lean_tactic_reflexivity() {
        let input = r#"
            theorem reflexivity_test {
                forall x: Int . x == x
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // Reflexivity should generate "intro" + "rfl" tactics
        assert!(
            compiled.code.contains("intro x"),
            "Expected 'intro x' for forall"
        );
        assert!(
            compiled.code.contains("rfl"),
            "Expected 'rfl' for reflexivity"
        );
    }

    #[test]
    fn test_lean_tactic_excluded_middle() {
        let input = r#"
            theorem excluded_middle_tactic {
                forall p: Bool . p or not p
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // Excluded middle should use Classical.em
        assert!(
            compiled.code.contains("intro p"),
            "Expected 'intro p' for forall"
        );
        assert!(
            compiled.code.contains("Classical.em"),
            "Expected 'Classical.em' for excluded middle"
        );

        // Should also include Classical import
        assert!(
            compiled.code.contains("import Mathlib.Logic.Classical"),
            "Expected Classical import"
        );
        assert!(
            compiled.code.contains("open Classical"),
            "Expected 'open Classical' namespace"
        );
    }

    #[test]
    fn test_lean_arithmetic_uses_omega() {
        let input = r#"
            theorem linear_arith {
                forall x: Int, y: Int . x + y == y + x
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // Linear arithmetic should use omega
        assert!(
            compiled.code.contains("intro x y"),
            "Expected 'intro x y' for nested foralls"
        );
        assert!(
            compiled.code.contains("omega"),
            "Expected 'omega' for linear arithmetic"
        );
    }

    #[test]
    fn test_lean_nested_forall_intros() {
        let input = r#"
            theorem nested_forall {
                forall a: Int, b: Int, c: Int . a == a
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // All three variables should be introduced
        assert!(
            compiled.code.contains("intro a b c"),
            "Expected 'intro a b c' for nested foralls"
        );
        assert!(
            compiled.code.contains("rfl"),
            "Expected 'rfl' for reflexivity"
        );
    }

    #[test]
    fn test_lean_implication_introduces_hypothesis() {
        let input = r#"
            theorem modus_ponens {
                forall p: Bool . p implies p
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // Should introduce both the variable and the hypothesis
        assert!(
            compiled.code.contains("intro p h"),
            "Expected 'intro p h' for forall + implication"
        );
    }

    #[test]
    fn test_lean_conjunction_uses_constructor() {
        let input = r#"
            theorem conjunction_test {
                true and true
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // Conjunction should use constructor to split
        assert!(
            compiled.code.contains("constructor"),
            "Expected 'constructor' for conjunction"
        );
    }

    #[test]
    fn test_lean_decidable_uses_decide() {
        let input = r#"
            theorem decidable_test {
                1 == 1
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // Integer equality should use decide or omega
        // (omega works for decidable integer comparisons too)
        assert!(
            compiled.code.contains("decide")
                || compiled.code.contains("omega")
                || compiled.code.contains("rfl"),
            "Expected decide, omega, or rfl for integer equality"
        );
    }

    #[test]
    fn test_lean_omega_import_included() {
        let input = r#"
            theorem arith_test { 1 + 1 == 2 }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // Omega import should always be included
        assert!(
            compiled.code.contains("import Mathlib.Tactic.Omega"),
            "Expected Omega import"
        );
    }

    #[test]
    fn test_lean_no_classical_when_not_needed() {
        let input = r#"
            theorem simple_test { true }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // Should NOT include Classical import when not needed
        assert!(
            !compiled.code.contains("import Mathlib.Logic.Classical"),
            "Should not have Classical import"
        );
        assert!(
            !compiled.code.contains("open Classical"),
            "Should not have 'open Classical'"
        );
    }

    #[test]
    fn test_lean_inequality_uses_omega() {
        let input = r#"
            theorem inequality_test {
                forall x: Int . x <= x
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // <= comparison with arithmetic should use omega
        assert!(
            compiled.code.contains("omega"),
            "Expected 'omega' for inequality"
        );
    }

    // ============================================================================
    // ALLOY CHECK COMMAND TESTS
    // ============================================================================

    #[test]
    fn test_alloy_theorem_generates_check() {
        let input = r#"
            theorem reflexive {
                forall x: Node . x == x
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_alloy(&typed);

        // Theorems should generate assert and check commands
        assert!(compiled.code.contains("assert reflexive"));
        assert!(compiled.code.contains("check reflexive for 5"));
    }

    #[test]
    fn test_alloy_multiple_assertions() {
        let input = r#"
            invariant inv1 { forall x: Node . x == x }
            invariant inv2 { forall y: Node . y == y }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_alloy(&typed);

        // Multiple invariants should generate multiple check commands
        assert!(compiled.code.contains("assert inv1"));
        assert!(compiled.code.contains("assert inv2"));
        assert!(compiled.code.contains("check inv1 for 5"));
        assert!(compiled.code.contains("check inv2 for 5"));
    }

    #[test]
    fn test_alloy_pred_stubs_for_functions() {
        // When a theorem references undefined functions, Alloy should generate pred stubs
        let input = r#"
            type Tenant = { id: String }

            security isolation {
                forall t1: Tenant, t2: Tenant .
                    t1 != t2 implies not can_observe(t1, actions(t2))
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_alloy(&typed);

        // Should generate pred stubs for can_observe and actions
        assert!(
            compiled.code.contains("pred can_observe"),
            "Expected pred stub for can_observe"
        );
        assert!(
            compiled.code.contains("pred actions"),
            "Expected pred stub for actions"
        );
        // Stubs should have correct arity
        assert!(
            compiled
                .code
                .contains("pred can_observe[x0: univ, x1: univ]"),
            "Expected can_observe with 2 params"
        );
        assert!(
            compiled.code.contains("pred actions[x0: univ]"),
            "Expected actions with 1 param"
        );
    }

    #[test]
    fn test_alloy_no_pred_stubs_when_no_functions() {
        // When no functions are referenced, no pred stubs should be generated
        let input = r#"
            type Node = { id: String }

            invariant unique_ids {
                forall n1: Node, n2: Node .
                    n1.id == n2.id implies n1 == n2
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_alloy(&typed);

        // Should not have predicate stubs section
        assert!(
            !compiled.code.contains("// Predicate stubs"),
            "Should not have pred stubs when no functions referenced"
        );
    }

    #[test]
    fn test_lean_ring_tactic_import() {
        // Non-linear arithmetic (x*y) should use ring tactic and include Ring import
        // Use x*y vs y*x which is not reflexive but needs ring to prove commutative
        let input = r#"
            theorem poly_test {
                forall x: Int, y: Int . x * y == y * x
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // ring tactic should be suggested for polynomial equality
        assert!(
            compiled.code.contains("ring"),
            "Expected 'ring' tactic for polynomial"
        );
        // Ring import should be included
        assert!(
            compiled.code.contains("import Mathlib.Tactic.Ring"),
            "Expected Ring import"
        );
    }

    #[test]
    fn test_lean_no_ring_import_when_linear() {
        // Linear arithmetic should use omega, not ring
        let input = r#"
            theorem linear_test {
                forall x: Int . x + 1 == 1 + x
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // omega tactic should be used for linear arithmetic
        assert!(
            compiled.code.contains("omega"),
            "Expected 'omega' for linear arithmetic"
        );
        // Ring import should NOT be included
        assert!(
            !compiled.code.contains("import Mathlib.Tactic.Ring"),
            "Should not have Ring import for linear arithmetic"
        );
    }

    #[test]
    fn test_lean_double_negation_uses_simp() {
        // Double negation as the body should use simp
        // Use `not not true` which is just a double negation expression
        let input = r#"
            theorem double_neg {
                not not true
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // Double negation should use simp
        assert!(
            compiled.code.contains("simp"),
            "Expected 'simp' for double negation"
        );
    }

    #[test]
    fn test_lean_method_call_length_uses_simp() {
        // List length method call should use simp when comparing with 0
        // The inner expression after intros is xs.length == 0
        let input = r#"
            theorem empty_list {
                forall xs: List<Int> . xs.length == 0
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // Method calls like length should use simp
        assert!(
            compiled.code.contains("simp"),
            "Expected 'simp' for list method calls"
        );
    }

    #[test]
    fn test_lean_linarith_with_hypotheses() {
        // When we have hypotheses from implications, linarith can use them
        // for linear arithmetic inequality goals
        let input = r#"
            theorem transitivity {
                forall x: Int . forall y: Int . forall z: Int .
                    x < y implies y < z implies x < z
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // Should use linarith when hypotheses are available for linear arithmetic
        assert!(
            compiled.code.contains("linarith"),
            "Expected 'linarith' for inequality with hypotheses"
        );
        // Should include Linarith import
        assert!(
            compiled.code.contains("import Mathlib.Tactic.Linarith"),
            "Expected Linarith import"
        );
    }

    #[test]
    fn test_lean_no_linarith_without_hypotheses() {
        // Without hypotheses (no implications), should use omega not linarith
        let input = r#"
            theorem simple_ineq {
                forall x: Int . x < x + 1
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_lean(&typed);

        // Without hypotheses, should use omega not linarith
        assert!(
            compiled.code.contains("omega"),
            "Expected 'omega' without hypotheses"
        );
        // Linarith import should NOT be included
        assert!(
            !compiled.code.contains("import Mathlib.Tactic.Linarith"),
            "Should not have Linarith import without hypotheses"
        );
    }

    // ========================================================================
    // SMT-LIB2 TESTS
    // ========================================================================

    #[test]
    fn test_smtlib2_compile_theorem() {
        let input = r#"
            theorem positive {
                forall x: Int . x > 0 implies x >= 1
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_smtlib2(&typed);

        assert_eq!(compiled.backend, "SMT-LIB2");
        assert!(compiled.code.contains("(set-logic ALL)"));
        assert!(compiled.code.contains("(forall ((x Int))"));
        assert!(compiled.code.contains("(=> (> x 0) (>= x 1))"));
        assert!(compiled.code.contains("(assert (not"));
        assert!(compiled.code.contains("(check-sat)"));
    }

    #[test]
    fn test_smtlib2_compile_invariant() {
        let input = r#"
            invariant non_negative {
                forall x: Int . x * x >= 0
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_smtlib2(&typed);

        assert!(compiled.code.contains("; Invariant: non_negative"));
        assert!(compiled.code.contains("(forall ((x Int))"));
        assert!(compiled.code.contains("(>= (* x x) 0)"));
    }

    #[test]
    fn test_smtlib2_compile_contract() {
        let input = r#"
            contract add(x: Int, y: Int) -> Int {
                requires { x >= 0 }
                ensures { result >= x }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_smtlib2(&typed);

        assert!(compiled.code.contains("; Contract: add"));
        assert!(compiled.code.contains("(assert (>= x 0)) ; precondition"));
        assert!(compiled
            .code
            .contains("(assert (not (>= result x))) ; postcondition"));
    }

    #[test]
    fn test_smtlib2_operators() {
        let input = r#"
            theorem ops {
                forall a: Int, b: Int .
                    (a + b) - (a * b) == a % b implies a / b > 0
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_smtlib2(&typed);

        assert!(compiled.code.contains("(+ a b)"));
        assert!(compiled.code.contains("(* a b)"));
        assert!(compiled.code.contains("(- (+ a b) (* a b))"));
        assert!(compiled.code.contains("(mod a b)"));
        assert!(compiled.code.contains("(div a b)"));
    }

    #[test]
    fn test_smtlib2_boolean_ops() {
        let input = r#"
            theorem bool_test {
                forall p: Bool, q: Bool .
                    (p and q) implies (p or q)
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_smtlib2(&typed);

        assert!(compiled.code.contains("(and p q)"));
        assert!(compiled.code.contains("(or p q)"));
        assert!(compiled.code.contains("(=>"));
    }

    #[test]
    fn test_smtlib2_comparison_ne() {
        let input = r#"
            theorem distinct_test {
                forall x: Int, y: Int . x != y implies not (x == y)
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_smtlib2(&typed);

        assert!(compiled.code.contains("(distinct x y)"));
        assert!(compiled.code.contains("(= x y)"));
    }

    #[test]
    fn test_smtlib2_with_logic() {
        let input = r#"
            theorem simple { true }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_smtlib2_with_logic(&typed, "QF_LIA");

        assert!(compiled.code.contains("(set-logic QF_LIA)"));
    }

    #[test]
    fn test_smtlib2_type_declarations() {
        let input = r#"
            type Node = { id: Int }
            theorem node_test { forall n: Node . n == n }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_smtlib2(&typed);

        assert!(compiled.code.contains("(declare-sort Node 0)"));
    }

    #[test]
    fn test_smtlib2_exists() {
        let input = r#"
            theorem exists_test {
                exists x: Int . x > 0
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_smtlib2(&typed);

        assert!(compiled.code.contains("(exists ((x Int))"));
        assert!(compiled.code.contains("(> x 0)"));
    }

    #[test]
    fn test_smtlib2_negation() {
        let input = r#"
            theorem neg_test {
                forall x: Int . not (x < 0) implies x >= 0
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_smtlib2(&typed);

        assert!(compiled.code.contains("(not (< x 0))"));
    }

    // ========================================================================
    // TACTIC SUGGESTION TESTS
    // ========================================================================

    #[test]
    fn test_suggest_tactics_for_theorem() {
        let input = r#"
            theorem test_bool {
                forall p: Bool . p or not p
            }
        "#;
        let typed = compile_usl(input);
        let tactics = suggest_tactics_for_property(&typed.spec.properties[0]);

        // Should return non-empty tactics for theorem
        assert!(
            !tactics.is_empty(),
            "Expected non-empty tactics for theorem"
        );
        // Tactics should be meaningful, not garbage
        for tactic in &tactics {
            assert!(
                !tactic.is_empty(),
                "Tactics should not contain empty strings"
            );
            assert_ne!(tactic, "xyzzy", "Tactics should be meaningful");
        }
    }

    #[test]
    fn test_suggest_tactics_for_invariant() {
        let input = r#"
            invariant always_positive {
                forall x: Int . x * x >= 0
            }
        "#;
        let typed = compile_usl(input);
        let tactics = suggest_tactics_for_property(&typed.spec.properties[0]);

        // Should return non-empty tactics for invariant
        assert!(
            !tactics.is_empty(),
            "Expected non-empty tactics for invariant"
        );
        // Should include omega for arithmetic
        let tactic_str = tactics.join(" ");
        assert!(
            tactic_str.contains("omega")
                || tactic_str.contains("decide")
                || tactic_str.contains("simp"),
            "Expected numeric tactic suggestion for arithmetic invariant, got: {tactics:?}"
        );
    }

    #[test]
    fn test_suggest_tactics_for_refinement() {
        let input = r#"
            refinement optimized refines base {
                abstraction { true }
                simulation { forall s: State . step(s) == step(s) }
            }
        "#;
        let typed = compile_usl(input);
        let tactics = suggest_tactics_for_property(&typed.spec.properties[0]);

        // Refinement should return tactics with constructor pattern
        assert!(
            !tactics.is_empty(),
            "Expected non-empty tactics for refinement"
        );
        // First tactic should be constructor for splitting goals
        assert!(
            tactics[0].contains("constructor"),
            "Refinement tactics should start with constructor"
        );
    }

    #[test]
    fn test_suggest_tactics_for_security() {
        let input = r#"
            security confidentiality {
                forall user: User . not (can_read(user, secret))
            }
        "#;
        let typed = compile_usl(input);
        let tactics = suggest_tactics_for_property(&typed.spec.properties[0]);

        // Should return non-empty tactics for security
        assert!(
            !tactics.is_empty(),
            "Expected non-empty tactics for security property"
        );
    }

    #[test]
    fn test_suggest_tactics_for_temporal_empty() {
        let input = r#"
            temporal liveness {
                always(eventually(done))
            }
        "#;
        let typed = compile_usl(input);
        let tactics = suggest_tactics_for_property(&typed.spec.properties[0]);

        // Temporal properties don't use LEAN tactics
        assert!(
            tactics.is_empty(),
            "Expected empty tactics for temporal property, got: {tactics:?}"
        );
    }

    #[test]
    fn test_suggest_tactics_for_contract_empty() {
        let input = r#"
            contract add(x: Int, y: Int) -> Int {
                requires { x >= 0 }
                ensures { result >= x }
            }
        "#;
        let typed = compile_usl(input);
        let tactics = suggest_tactics_for_property(&typed.spec.properties[0]);

        // Contracts don't use LEAN tactics
        assert!(
            tactics.is_empty(),
            "Expected empty tactics for contract, got: {tactics:?}"
        );
    }

    #[test]
    fn test_suggest_tactics_no_empty_strings() {
        // Test that filter(|s| !s.is_empty()) works correctly
        let input = r#"
            theorem multiline_tactic {
                forall x: Int . x == x
            }
        "#;
        let typed = compile_usl(input);
        let tactics = suggest_tactics_for_property(&typed.spec.properties[0]);

        // None of the returned tactics should be empty strings
        for tactic in &tactics {
            assert!(
                !tactic.is_empty(),
                "Found empty string in tactics - filter should remove these"
            );
        }
    }
}
