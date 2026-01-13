//! Transition system to CHC encoding
//!
//! This module converts transition systems (from kani-fast-kinduction) into
//! CHC format suitable for Z3's Spacer engine.

use crate::clause::{
    ChcSystem, ChcSystemBuilder, ClauseHead, HornClause, Predicate, PredicateApp, Variable,
};
use kani_fast_kinduction::{PropertyType, SmtType, StateFormula, TransitionSystem};

/// Transform primed variable notation (x') to _next notation (x_next)
/// Z3 doesn't support apostrophes in identifiers
///
/// Uses a single-pass approach: scan the formula once, replacing primed
/// variables as they are encountered.
///
/// NOTE: This function transforms ALL primed identifiers (ending with '),
/// not just those in var_names. This is important because dead variable
/// elimination may remove some variables from the state, but the transition
/// formula may still reference their primed versions. Transforming all primed
/// identifiers ensures consistent SMT-LIB2 output.
fn transform_primes_to_next(formula: &str, _var_names: &[String]) -> String {
    let mut result = String::with_capacity(formula.len() * 2);
    // SMT-LIB formulas are ASCII, so we can work directly with bytes
    let bytes = formula.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i];
        // Check if we're at the start of an identifier (letter or underscore)
        if c.is_ascii_alphabetic() || c == b'_' {
            // Collect the identifier
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'\'')
            {
                i += 1;
            }
            // SAFETY: We started with valid UTF-8 and only collected ASCII bytes
            let ident = &formula[start..i];

            // Transform ANY primed identifier (ending with ') to _next notation
            if let Some(base) = ident.strip_suffix('\'') {
                // Remove trailing ' and add _next
                result.push_str(base);
                result.push_str("_next");
            } else {
                result.push_str(ident);
            }
        } else {
            // SAFETY: SMT-LIB is ASCII, single byte is valid char
            result.push(c as char);
            i += 1;
        }
    }

    result
}

/// Encode a transition system as CHC
///
/// The encoding creates:
/// - One predicate Inv(state_vars) for the invariant
/// - Init clause: init_formula => Inv(state_vars)
/// - Trans clause: Inv(state_vars) ∧ trans_formula => Inv(state_vars')
/// - Property clause: Inv(state_vars) => property_formula
pub fn encode_transition_system(ts: &TransitionSystem) -> ChcSystem {
    let mut system = ChcSystem::new();

    // Create invariant predicate with state variables
    let inv_types: Vec<SmtType> = ts.variables.iter().map(|v| v.smt_type.clone()).collect();
    let inv_pred = Predicate::new("Inv", inv_types);
    system.add_predicate(inv_pred);

    // Create variable list for current state
    let current_vars: Vec<Variable> = ts
        .variables
        .iter()
        .map(|v| Variable::new(&v.name, v.smt_type.clone()))
        .collect();

    // Collect variable names once for reuse
    let var_names: Vec<String> = ts.variables.iter().map(|v| v.name.clone()).collect();

    // Create variable list for next state (using _next suffix instead of primes)
    let next_vars: Vec<Variable> = ts
        .variables
        .iter()
        .map(|v| Variable::new(format!("{}_next", v.name), v.smt_type.clone()))
        .collect();

    let next_args: Vec<String> = var_names.iter().map(|n| format!("{}_next", n)).collect();

    // All variables for transition clause
    let all_vars: Vec<Variable> = current_vars
        .iter()
        .chain(next_vars.iter())
        .cloned()
        .collect();

    // Initial state clause: init => Inv(current)
    let init_clause = HornClause::new(
        current_vars.clone(),
        vec![],
        ts.init.clone(),
        ClauseHead::Predicate(PredicateApp::new("Inv", var_names.clone())),
    )
    .with_name("initial_state");
    system.add_clause(init_clause);

    // Transition clause: Inv(current) ∧ trans => Inv(next)
    let transformed_trans = transform_primes_to_next(&ts.transition.smt_formula, &var_names);
    let trans_clause = HornClause::new(
        all_vars.clone(),
        vec![PredicateApp::new("Inv", var_names.clone())],
        StateFormula::new(transformed_trans),
        ClauseHead::Predicate(PredicateApp::new("Inv", next_args)),
    )
    .with_name("transition");
    system.add_clause(trans_clause);

    // Add auxiliary invariants as additional constraints
    for (i, inv) in ts.invariants.iter().enumerate() {
        let transformed_inv = transform_primes_to_next(&inv.smt_formula, &var_names);
        // Check if transformed_inv contains "{name}_next" for any variable name
        // Using byte-level check to avoid format! allocation per variable
        let uses_next = var_names.iter().any(|name| {
            let name_bytes = name.as_bytes();
            let inv_bytes = transformed_inv.as_bytes();
            let suffix = b"_next";
            let pattern_len = name_bytes.len() + suffix.len();
            inv_bytes
                .windows(pattern_len)
                .any(|w| &w[..name_bytes.len()] == name_bytes && &w[name_bytes.len()..] == suffix)
        });
        let quantified_vars = if uses_next {
            all_vars.clone()
        } else {
            current_vars.clone()
        };

        let inv_clause = HornClause::new(
            quantified_vars,
            vec![PredicateApp::new("Inv", var_names.clone())],
            StateFormula::new(transformed_inv).negate(),
            ClauseHead::Query,
        )
        .with_name(format!("aux_invariant_{}", i));
        system.add_clause(inv_clause);
    }

    // Property clauses
    for prop in &ts.properties {
        match prop.property_type {
            PropertyType::Safety => {
                // Safety: Inv(x) => property(x)
                // Encoded as: Inv(x) ∧ ¬property(x) => false
                let negated_prop = prop.formula.negate();
                let prop_clause = HornClause::new(
                    current_vars.clone(),
                    vec![PredicateApp::new("Inv", var_names.clone())],
                    negated_prop,
                    ClauseHead::Query,
                )
                .with_name(format!("property_{}", prop.id));
                system.add_clause(prop_clause);
            }
            PropertyType::Reachability => {
                // Reachability: check if property can be reached
                // For CHC this is trickier - skip for now
            }
            PropertyType::Liveness => {
                // Liveness requires fairness constraints - not directly supported
            }
        }
    }

    system
}

/// Encode multiple properties as separate CHC queries
///
/// This generates one CHC system per property, allowing independent verification.
pub fn encode_properties_separately(ts: &TransitionSystem) -> Vec<(String, ChcSystem)> {
    ts.properties
        .iter()
        .filter(|p| p.property_type == PropertyType::Safety)
        .map(|prop| {
            let mut single_prop_ts = ts.clone();
            single_prop_ts.properties = vec![prop.clone()];
            (prop.id.clone(), encode_transition_system(&single_prop_ts))
        })
        .collect()
}

/// Encode a simple loop as CHC
///
/// This helper creates a CHC system for a simple loop verification:
/// - Initial value x = init_value
/// - Loop body increments/updates x
/// - Property must hold for all reachable x values
///
/// Note: use `x_next` notation in loop_body for the next-state variable
pub fn encode_simple_loop(
    var_name: &str,
    var_type: SmtType,
    init_constraint: &str,
    loop_body: &str,
    property: &str,
) -> ChcSystem {
    // Transform primes to _next notation if user provided primed notation
    let transformed_body =
        loop_body.replace(&format!("{}'", var_name), &format!("{}_next", var_name));

    ChcSystemBuilder::new()
        .predicate("Inv", vec![var_type.clone()])
        .init(
            "Inv",
            vec![var_name.to_string()],
            vec![Variable::new(var_name, var_type.clone())],
            init_constraint,
        )
        .transition(
            "Inv",
            vec![var_name.to_string()],
            vec![format!("{}_next", var_name)],
            vec![
                Variable::new(var_name, var_type.clone()),
                Variable::new(format!("{}_next", var_name), var_type),
            ],
            &transformed_body,
        )
        .property(
            "Inv",
            vec![var_name.to_string()],
            vec![Variable::new(var_name, SmtType::Int)],
            property,
        )
        .build()
}

/// Encode an array bounds check as CHC
///
/// Verifies that array accesses are always within bounds.
pub fn encode_array_bounds(
    index_var: &str,
    array_len: u64,
    init_constraint: &str,
    update_expr: &str,
) -> ChcSystem {
    let bounds_property = format!("(and (>= {} 0) (< {} {}))", index_var, index_var, array_len);

    // Transform primes to _next notation
    let transformed_update =
        update_expr.replace(&format!("{}'", index_var), &format!("{}_next", index_var));

    ChcSystemBuilder::new()
        .predicate("Inv", vec![SmtType::Int])
        .init(
            "Inv",
            vec![index_var.to_string()],
            vec![Variable::new(index_var, SmtType::Int)],
            init_constraint,
        )
        .transition(
            "Inv",
            vec![index_var.to_string()],
            vec![format!("{}_next", index_var)],
            vec![
                Variable::new(index_var, SmtType::Int),
                Variable::new(format!("{}_next", index_var), SmtType::Int),
            ],
            &transformed_update,
        )
        .property(
            "Inv",
            vec![index_var.to_string()],
            vec![Variable::new(index_var, SmtType::Int)],
            &bounds_property,
        )
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kani_fast_kinduction::TransitionSystemBuilder;

    // ==================== transform_primes_to_next Tests ====================

    #[test]
    fn test_transform_primes_single_var() {
        let result = transform_primes_to_next("(= x' (+ x 1))", &["x".to_string()]);
        assert_eq!(result, "(= x_next (+ x 1))");
    }

    #[test]
    fn test_transform_primes_multiple_vars() {
        let result = transform_primes_to_next(
            "(and (= x' (+ x 1)) (= y' y))",
            &["x".to_string(), "y".to_string()],
        );
        assert_eq!(result, "(and (= x_next (+ x 1)) (= y_next y))");
    }

    #[test]
    fn test_transform_primes_no_primes() {
        let result = transform_primes_to_next("(= x (+ y 1))", &["x".to_string(), "y".to_string()]);
        assert_eq!(result, "(= x (+ y 1))");
    }

    #[test]
    fn test_transform_primes_empty_vars() {
        // Now transforms ALL primed variables, regardless of var_names list
        let result = transform_primes_to_next("(= x' (+ x 1))", &[]);
        assert_eq!(result, "(= x_next (+ x 1))");
    }

    #[test]
    fn test_transform_primes_all_primed() {
        // Transforms ALL primed variables to _next, not just those in the list
        // This is important for dead variable elimination scenarios
        let result = transform_primes_to_next("(= x' (+ y' 1))", &["x".to_string()]);
        assert_eq!(result, "(= x_next (+ y_next 1))");
    }

    // ==================== encode_transition_system Tests ====================

    #[test]
    fn test_encode_counter() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "x_nonneg", "(>= x 0)")
            .build();

        let chc = encode_transition_system(&ts);

        assert!(chc.predicates.contains_key("Inv"));
        assert_eq!(chc.clauses.len(), 3); // init, trans, property

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("(set-logic HORN)"));
        assert!(smt2.contains("(declare-fun Inv (Int) Bool)"));
    }

    #[test]
    fn test_encode_two_var_system() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Int)
            .init("(and (= x 0) (= y 0))")
            .transition("(and (= x' (+ x 1)) (or (= y' (+ y 1)) (= y' y)))")
            .property("p1", "x_ge_y", "(>= x y)")
            .build();

        let chc = encode_transition_system(&ts);

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("(declare-fun Inv (Int Int) Bool)"));
    }

    #[test]
    fn test_encode_no_properties() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        let chc = encode_transition_system(&ts);

        // Should have Inv predicate, init and trans clauses, no property clause
        assert!(chc.predicates.contains_key("Inv"));
        assert_eq!(chc.clauses.len(), 2);
        assert!(!chc.has_query());
    }

    #[test]
    fn test_encode_bool_variable() {
        let ts = TransitionSystemBuilder::new()
            .variable("flag", SmtType::Bool)
            .init("(= flag false)")
            .transition("(= flag' (not flag))")
            .property("p1", "alternates", "true")
            .build();

        let chc = encode_transition_system(&ts);

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("(declare-fun Inv (Bool) Bool)"));
    }

    #[test]
    fn test_encode_bitvec_variable() {
        let ts = TransitionSystemBuilder::new()
            .variable("bv", SmtType::BitVec(32))
            .init("(= bv #x00000000)")
            .transition("(= bv' (bvadd bv #x00000001))")
            .property("p1", "bounded", "(bvult bv #xFFFFFFFF)")
            .build();

        let chc = encode_transition_system(&ts);

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("(_ BitVec 32)"));
    }

    #[test]
    fn test_encode_multiple_properties() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .property("p2", "positive", "(> x -1)")
            .build();

        let chc = encode_transition_system(&ts);

        // Should have init, trans, and two property clauses
        assert_eq!(chc.clauses.len(), 4);
    }

    #[test]
    fn test_encode_with_invariants() {
        let mut ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .build();

        ts.add_invariant(StateFormula::new("(>= x 0)"));

        let chc = encode_transition_system(&ts);

        assert!(chc.predicates.contains_key("Inv"));
        assert_eq!(chc.clauses.len(), 4);

        let invariant_clauses: Vec<&HornClause> = chc
            .clauses
            .iter()
            .filter(|c| {
                c.name
                    .as_deref()
                    .is_some_and(|n| n.starts_with("aux_invariant_"))
            })
            .collect();

        assert_eq!(invariant_clauses.len(), 1);
        assert!(invariant_clauses[0]
            .constraint
            .smt_formula
            .contains("(not (>= x 0))"));
    }

    #[test]
    fn test_encode_invariant_with_prime_variables() {
        let mut ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .build();

        ts.add_invariant(StateFormula::new("(= x' (+ x 2))"));

        let chc = encode_transition_system(&ts);

        let invariant_clause = chc
            .clauses
            .iter()
            .find(|c| {
                c.name
                    .as_deref()
                    .is_some_and(|n| n.starts_with("aux_invariant_"))
            })
            .expect("missing auxiliary invariant clause");

        assert!(invariant_clause.constraint.smt_formula.contains("x_next"));
        assert!(invariant_clause
            .variables
            .iter()
            .any(|v| v.name == "x_next"));
    }

    // ==================== encode_properties_separately Tests ====================

    #[test]
    fn test_encode_properties_separately() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .property("p2", "bounded", "(< x 1000)")
            .build();

        let systems = encode_properties_separately(&ts);
        assert_eq!(systems.len(), 2);
        assert_eq!(systems[0].0, "p1");
        assert_eq!(systems[1].0, "p2");
    }

    #[test]
    fn test_encode_properties_separately_no_properties() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        let systems = encode_properties_separately(&ts);
        assert!(systems.is_empty());
    }

    #[test]
    fn test_encode_properties_separately_single() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .build();

        let systems = encode_properties_separately(&ts);
        assert_eq!(systems.len(), 1);
        assert_eq!(systems[0].0, "p1");

        // Each system should be complete (predicate, init, trans, property)
        let (_, system) = &systems[0];
        assert!(system.predicates.contains_key("Inv"));
        assert_eq!(system.clauses.len(), 3);
        assert!(system.has_query());
    }

    // ==================== encode_simple_loop Tests ====================

    #[test]
    fn test_encode_simple_loop() {
        let chc = encode_simple_loop("i", SmtType::Int, "(= i 0)", "(= i' (+ i 1))", "(>= i 0)");

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("(set-logic HORN)"));
        assert!(smt2.contains("(declare-fun Inv (Int) Bool)"));
        assert!(smt2.contains("(check-sat)"));
    }

    #[test]
    fn test_encode_simple_loop_with_next_notation() {
        // Test that _next notation works directly
        let chc = encode_simple_loop(
            "i",
            SmtType::Int,
            "(= i 0)",
            "(= i_next (+ i 1))",
            "(>= i 0)",
        );

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("i_next"));
    }

    #[test]
    fn test_encode_simple_loop_decrement() {
        let chc = encode_simple_loop("x", SmtType::Int, "(= x 10)", "(= x' (- x 1))", "(>= x 0)");

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("(= x 10)"));
        assert!(smt2.contains("(- x 1)"));
    }

    #[test]
    fn test_encode_simple_loop_bool_type() {
        let chc = encode_simple_loop(
            "flag",
            SmtType::Bool,
            "(= flag true)",
            "(= flag' (not flag))",
            "flag",
        );

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("(declare-fun Inv (Bool) Bool)"));
    }

    // ==================== encode_array_bounds Tests ====================

    #[test]
    fn test_encode_array_bounds() {
        let chc = encode_array_bounds("i", 10, "(= i 0)", "(and (< i 10) (= i' (+ i 1)))");

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("(< i 10)"));
        assert!(smt2.contains("(>= i 0)"));
    }

    #[test]
    fn test_encode_array_bounds_zero_length() {
        let chc = encode_array_bounds("i", 0, "(= i 0)", "(= i' i)");

        let smt2 = chc.to_smt2();
        // Property should require i >= 0 and i < 0, which is false
        assert!(smt2.contains("(< i 0)"));
    }

    #[test]
    fn test_encode_array_bounds_large_array() {
        let chc = encode_array_bounds("idx", 1000000, "(= idx 0)", "(= idx' (+ idx 1))");

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("1000000"));
    }

    #[test]
    fn test_encode_array_bounds_with_condition() {
        let chc = encode_array_bounds(
            "i",
            100,
            "(and (= i 0) (> n 0))",
            "(and (< i n) (= i' (+ i 1)))",
        );

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("(and (= i 0) (> n 0))"));
    }

    // ==================== Integration / Edge Case Tests ====================

    #[test]
    fn test_encode_system_with_reachability_property() {
        use kani_fast_kinduction::PropertyType;

        let mut ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        // Add a reachability property (should be skipped in CHC encoding)
        ts.add_property(kani_fast_kinduction::Property {
            id: "reach".to_string(),
            name: "reach state x=10".to_string(),
            formula: StateFormula::new("(= x 10)"),
            property_type: PropertyType::Reachability,
        });

        let chc = encode_transition_system(&ts);

        // Reachability properties are not supported, so no query clause
        assert!(!chc.has_query());
        assert_eq!(chc.clauses.len(), 2); // Just init and trans
    }

    #[test]
    fn test_encode_system_with_liveness_property() {
        use kani_fast_kinduction::PropertyType;

        let mut ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        // Add a liveness property (should be skipped in CHC encoding)
        ts.add_property(kani_fast_kinduction::Property {
            id: "live".to_string(),
            name: "eventually x > 100".to_string(),
            formula: StateFormula::new("(> x 100)"),
            property_type: PropertyType::Liveness,
        });

        let chc = encode_transition_system(&ts);

        // Liveness properties are not supported, so no query clause
        assert!(!chc.has_query());
    }

    #[test]
    fn test_encode_system_clause_names() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .build();

        let chc = encode_transition_system(&ts);

        // Check clause names
        assert!(chc
            .clauses
            .iter()
            .any(|c| c.name.as_deref() == Some("initial_state")));
        assert!(chc
            .clauses
            .iter()
            .any(|c| c.name.as_deref() == Some("transition")));
        assert!(chc
            .clauses
            .iter()
            .any(|c| c.name.as_ref().is_some_and(|s| s.contains("property"))));
    }

    #[test]
    fn test_encode_complex_init() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Int)
            .variable("z", SmtType::Int)
            .init("(and (= x 0) (= y 0) (= z 0))")
            .transition("(and (= x' (+ x 1)) (= y' (+ y x)) (= z' (+ z y)))")
            .property("p1", "ordered", "(and (<= x y) (<= y z))")
            .build();

        let chc = encode_transition_system(&ts);

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("(declare-fun Inv (Int Int Int) Bool)"));
        assert!(smt2.contains("z_next"));
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_encode_bitvec_type() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::BitVec(32))
            .init("(= x #x00000000)")
            .transition("(= x' (bvadd x #x00000001))")
            .property("p1", "nonzero_check", "(not (= x #xFFFFFFFF))")
            .build();

        let chc = encode_transition_system(&ts);
        let smt2 = chc.to_smt2();

        assert!(smt2.contains("(_ BitVec 32)"));
        assert!(smt2.contains("(declare-fun Inv ((_ BitVec 32)) Bool)"));
    }

    #[test]
    fn test_encode_multiple_identical_properties() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "nonneg1", "(>= x 0)")
            .property("p2", "nonneg2", "(>= x 0)") // Same formula, different name
            .build();

        let chc = encode_transition_system(&ts);

        // Should have 2 property clauses (one for each property)
        let query_count = chc
            .clauses
            .iter()
            .filter(|c| matches!(c.head, ClauseHead::Query))
            .count();
        assert_eq!(query_count, 2);
    }

    #[test]
    fn test_encode_underscore_prefixed_vars() {
        // MIR-style variable names
        let ts = TransitionSystemBuilder::new()
            .variable("_0", SmtType::Int)
            .variable("_1", SmtType::Int)
            .init("(and (= _0 0) (= _1 1))")
            .transition("(and (= _0' (+ _0 _1)) (= _1' _1))")
            .property("p1", "positive", "(>= _0 0)")
            .build();

        let chc = encode_transition_system(&ts);
        let smt2 = chc.to_smt2();

        assert!(smt2.contains("_0_next"));
        assert!(smt2.contains("_1_next"));
    }

    #[test]
    fn test_transform_primes_nested_expressions() {
        let result = transform_primes_to_next(
            "(ite (> x' 0) (and (= y' (+ x' 1)) (= z' x')) (= y' 0))",
            &["x".to_string(), "y".to_string(), "z".to_string()],
        );
        assert_eq!(
            result,
            "(ite (> x_next 0) (and (= y_next (+ x_next 1)) (= z_next x_next)) (= y_next 0))"
        );
    }

    #[test]
    fn test_transform_primes_adjacent_primed_vars() {
        // Test case where primed variables appear adjacent without spaces
        // The tokenizer includes ' as part of identifiers, so x'y' is ONE identifier
        // that ends with ' and gets transformed to x'y_next
        let result = transform_primes_to_next("(= x'y' 0)", &["x".to_string(), "y".to_string()]);
        // x'y' is parsed as single identifier "x'y'" and becomes "x'y_next"
        assert_eq!(result, "(= x'y_next 0)");
    }

    #[test]
    fn test_encode_bool_array_mixed_types() {
        let ts = TransitionSystemBuilder::new()
            .variable("i", SmtType::Int)
            .variable("found", SmtType::Bool)
            .init("(and (= i 0) (= found false))")
            .transition("(and (= i' (+ i 1)) (= found' (or found (= i 42))))")
            .property("p1", "eventual_find", "(=> (> i 100) found)")
            .build();

        let chc = encode_transition_system(&ts);
        let smt2 = chc.to_smt2();

        assert!(smt2.contains("(declare-fun Inv (Int Bool) Bool)"));
    }

    #[test]
    fn test_encode_true_init_constraint() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("true") // Unconstrained initial state
            .transition("(= x' x)")
            .property("p1", "trivial", "true")
            .build();

        let chc = encode_transition_system(&ts);
        let smt2 = chc.to_smt2();

        // Should produce valid SMT2 even with trivial constraints
        assert!(smt2.contains("(set-logic HORN)"));
        assert!(smt2.contains("(check-sat)"));
    }

    #[test]
    fn test_encode_simple_loop_bitvec() {
        let chc = encode_simple_loop(
            "counter",
            SmtType::BitVec(8),
            "(= counter #x00)",
            "(= counter' (bvadd counter #x01))",
            "(bvult counter #xff)",
        );

        let smt2 = chc.to_smt2();
        assert!(smt2.contains("(_ BitVec 8)"));
        assert!(smt2.contains("counter_next"));
    }

    #[test]
    fn test_encode_system_no_transition() {
        // System with identity transition
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 5)")
            .transition("(= x' x)") // Identity: state doesn't change
            .property("p1", "constant", "(= x 5)")
            .build();

        let chc = encode_transition_system(&ts);
        assert!(chc.has_query());

        // The property should be satisfiable since x is always 5
        let smt2 = chc.to_smt2();
        assert!(smt2.contains("(= x_next x)"));
    }

    #[test]
    fn test_transform_primes_with_numbers_in_names() {
        let result = transform_primes_to_next(
            "(and (= var123' (+ var123 1)) (= x2' x2))",
            &["var123".to_string(), "x2".to_string()],
        );
        assert_eq!(result, "(and (= var123_next (+ var123 1)) (= x2_next x2))");
    }

    #[test]
    fn test_encode_properties_separately_preserves_order() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("first", "prop1", "(>= x 0)")
            .property("second", "prop2", "(<= x 100)")
            .property("third", "prop3", "(> x (- 10))")
            .build();

        let systems = encode_properties_separately(&ts);
        assert_eq!(systems.len(), 3);

        // Properties should maintain order
        assert_eq!(systems[0].0, "first");
        assert_eq!(systems[1].0, "second");
        assert_eq!(systems[2].0, "third");
    }

    #[test]
    fn test_encode_array_bounds_negative_property_generated() {
        let chc = encode_array_bounds("idx", 5, "(= idx 0)", "(= idx' (+ idx 1))");

        let smt2 = chc.to_smt2();
        // The property should include both lower and upper bounds
        assert!(smt2.contains(">= idx 0"));
        assert!(smt2.contains("< idx 5"));
    }
}
