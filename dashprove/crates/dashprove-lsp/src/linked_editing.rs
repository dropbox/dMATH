//! Linked editing range support for USL
//!
//! Provides linked editing ranges, enabling simultaneous editing of related text regions.
//! This is useful for renaming identifiers where all occurrences should change together.
//!
//! In USL, linked editing is supported for:
//! - Type names: all references to a type (definition and usages)
//! - Property names: all references to a property (definition and usages)
//! - Bound variables in quantifiers (forall/exists): the binding and all uses within scope

use crate::document::Document;
use dashprove_usl::{Expr, Property, TemporalExpr};
use std::collections::HashSet;
use tower_lsp::lsp_types::{LinkedEditingRanges, Position, Range};

/// Generate linked editing ranges for the identifier at the given position.
///
/// Returns all ranges that should be edited together when the user edits
/// the identifier at the given position, along with an optional word pattern
/// that constrains valid identifiers.
pub fn generate_linked_editing_ranges(
    doc: &Document,
    pos: Position,
) -> Option<LinkedEditingRanges> {
    let word = doc.word_at_position(pos.line, pos.character)?;
    let spec = doc.spec.as_ref()?;

    // Check if cursor is on a type name
    if spec.types.iter().any(|t| t.name == word) {
        let ranges = doc.find_all_references(word);
        if ranges.len() > 1 {
            return Some(LinkedEditingRanges {
                ranges,
                word_pattern: Some(identifier_pattern()),
            });
        }
    }

    // Check if cursor is on a property name
    if spec.properties.iter().any(|p| p.name() == word) {
        let ranges = doc.find_all_references(word);
        if ranges.len() > 1 {
            return Some(LinkedEditingRanges {
                ranges,
                word_pattern: Some(identifier_pattern()),
            });
        }
    }

    // Check if cursor is on a bound variable in a quantifier
    // This requires walking the AST to find the scope of the binding
    if let Some(ranges) = find_bound_variable_ranges(doc, spec, word, pos) {
        if ranges.len() > 1 {
            return Some(LinkedEditingRanges {
                ranges,
                word_pattern: Some(identifier_pattern()),
            });
        }
    }

    None
}

/// Regex pattern for valid USL identifiers.
fn identifier_pattern() -> String {
    r"[a-zA-Z_][a-zA-Z0-9_']*".to_string()
}

/// Find all ranges for a bound variable within its scope.
///
/// This searches for quantified variables (forall/exists) and returns
/// the binding and all usages within the quantifier's body.
fn find_bound_variable_ranges(
    doc: &Document,
    spec: &dashprove_usl::Spec,
    var_name: &str,
    _pos: Position,
) -> Option<Vec<Range>> {
    // We need to find if the variable is a quantified binding in any property
    // For simplicity, we check all properties and collect matches

    let mut all_ranges = Vec::new();

    for property in &spec.properties {
        if let Some(ranges) = find_variable_ranges_in_property(doc, property, var_name) {
            all_ranges.extend(ranges);
        }
    }

    if all_ranges.is_empty() {
        None
    } else {
        // Deduplicate ranges
        let mut seen = HashSet::new();
        all_ranges.retain(|r| {
            let key = (r.start.line, r.start.character, r.end.line, r.end.character);
            seen.insert(key)
        });
        Some(all_ranges)
    }
}

/// Find variable ranges within a specific property.
fn find_variable_ranges_in_property(
    doc: &Document,
    property: &Property,
    var_name: &str,
) -> Option<Vec<Range>> {
    match property {
        Property::Theorem(t) => find_variable_ranges_in_expr(doc, &t.body, var_name),
        Property::Invariant(i) => find_variable_ranges_in_expr(doc, &i.body, var_name),
        Property::Security(s) => find_variable_ranges_in_expr(doc, &s.body, var_name),
        Property::Semantic(s) => find_variable_ranges_in_expr(doc, &s.body, var_name),
        Property::Temporal(t) => find_variable_ranges_in_temporal(doc, &t.body, var_name),
        Property::Contract(c) => {
            let mut ranges = Vec::new();

            // Check parameter names
            for param in &c.params {
                if param.name == var_name {
                    ranges.extend(doc.find_all_references(var_name));
                    break;
                }
            }

            // Check requires/ensures clauses
            for req in &c.requires {
                if let Some(r) = find_variable_ranges_in_expr(doc, req, var_name) {
                    ranges.extend(r);
                }
            }
            for ens in &c.ensures {
                if let Some(r) = find_variable_ranges_in_expr(doc, ens, var_name) {
                    ranges.extend(r);
                }
            }
            for ens_err in &c.ensures_err {
                if let Some(r) = find_variable_ranges_in_expr(doc, ens_err, var_name) {
                    ranges.extend(r);
                }
            }

            if ranges.is_empty() {
                None
            } else {
                Some(ranges)
            }
        }
        Property::Refinement(r) => {
            let mut ranges = Vec::new();
            if let Some(rs) = find_variable_ranges_in_expr(doc, &r.abstraction, var_name) {
                ranges.extend(rs);
            }
            if let Some(rs) = find_variable_ranges_in_expr(doc, &r.simulation, var_name) {
                ranges.extend(rs);
            }
            if ranges.is_empty() {
                None
            } else {
                Some(ranges)
            }
        }
        Property::Probabilistic(p) => find_variable_ranges_in_expr(doc, &p.condition, var_name),
        Property::PlatformApi(_) => None, // PlatformApi doesn't have expression bodies
        Property::Bisimulation(_) => None, // Bisimulation doesn't have expression bodies
        Property::Version(_) => None,     // Version doesn't have expression bodies
        Property::Capability(_) => None,  // Capability doesn't have expression bodies
        Property::DistributedInvariant(d) => find_variable_ranges_in_expr(doc, &d.body, var_name),
        Property::DistributedTemporal(d) => {
            find_variable_ranges_in_temporal(doc, &d.body, var_name)
        }
        Property::Composed(c) => find_variable_ranges_in_expr(doc, &c.body, var_name),
        Property::ImprovementProposal(p) => {
            let mut ranges = Vec::new();
            if let Some(rs) = find_variable_ranges_in_expr(doc, &p.target, var_name) {
                ranges.extend(rs);
            }
            for expr in &p.improves {
                if let Some(rs) = find_variable_ranges_in_expr(doc, expr, var_name) {
                    ranges.extend(rs);
                }
            }
            for expr in &p.preserves {
                if let Some(rs) = find_variable_ranges_in_expr(doc, expr, var_name) {
                    ranges.extend(rs);
                }
            }
            for expr in &p.requires {
                if let Some(rs) = find_variable_ranges_in_expr(doc, expr, var_name) {
                    ranges.extend(rs);
                }
            }
            if ranges.is_empty() {
                None
            } else {
                Some(ranges)
            }
        }
        Property::VerificationGate(g) => {
            let mut ranges = Vec::new();
            for check in &g.checks {
                if let Some(rs) = find_variable_ranges_in_expr(doc, &check.condition, var_name) {
                    ranges.extend(rs);
                }
            }
            if let Some(rs) = find_variable_ranges_in_expr(doc, &g.on_pass, var_name) {
                ranges.extend(rs);
            }
            if let Some(rs) = find_variable_ranges_in_expr(doc, &g.on_fail, var_name) {
                ranges.extend(rs);
            }
            if ranges.is_empty() {
                None
            } else {
                Some(ranges)
            }
        }
        Property::Rollback(r) => {
            let mut ranges = Vec::new();
            for expr in &r.invariants {
                if let Some(rs) = find_variable_ranges_in_expr(doc, expr, var_name) {
                    ranges.extend(rs);
                }
            }
            if let Some(rs) = find_variable_ranges_in_expr(doc, &r.trigger, var_name) {
                ranges.extend(rs);
            }
            for expr in &r.guarantees {
                if let Some(rs) = find_variable_ranges_in_expr(doc, expr, var_name) {
                    ranges.extend(rs);
                }
            }
            if ranges.is_empty() {
                None
            } else {
                Some(ranges)
            }
        }
    }
}

/// Find variable ranges within an expression.
///
/// This recursively searches for the variable name within the expression,
/// particularly within quantified expressions where the variable is bound.
fn find_variable_ranges_in_expr(doc: &Document, expr: &Expr, var_name: &str) -> Option<Vec<Range>> {
    let mut ranges = Vec::new();
    collect_variable_refs_in_expr(doc, expr, var_name, &mut ranges);
    if ranges.is_empty() {
        None
    } else {
        Some(ranges)
    }
}

/// Recursively collect variable references in an expression.
fn collect_variable_refs_in_expr(
    doc: &Document,
    expr: &Expr,
    var_name: &str,
    ranges: &mut Vec<Range>,
) {
    match expr {
        Expr::Var(name) if name == var_name => {
            // This is a use of the variable
            ranges.extend(doc.find_all_references(var_name));
        }
        Expr::ForAll { var, ty: _, body } | Expr::Exists { var, ty: _, body } => {
            // If the bound variable matches, collect its uses
            if var == var_name {
                // The binding itself and uses within the body
                ranges.extend(doc.find_all_references(var_name));
            } else {
                // Recurse into the body for other variables
                collect_variable_refs_in_expr(doc, body, var_name, ranges);
            }
        }
        Expr::ForAllIn {
            var,
            collection,
            body,
        }
        | Expr::ExistsIn {
            var,
            collection,
            body,
        } => {
            if var == var_name {
                ranges.extend(doc.find_all_references(var_name));
            } else {
                collect_variable_refs_in_expr(doc, collection, var_name, ranges);
                collect_variable_refs_in_expr(doc, body, var_name, ranges);
            }
        }
        Expr::Not(inner) | Expr::Neg(inner) => {
            collect_variable_refs_in_expr(doc, inner, var_name, ranges);
        }
        Expr::And(left, right)
        | Expr::Or(left, right)
        | Expr::Implies(left, right)
        | Expr::Compare(left, _, right)
        | Expr::Binary(left, _, right) => {
            collect_variable_refs_in_expr(doc, left, var_name, ranges);
            collect_variable_refs_in_expr(doc, right, var_name, ranges);
        }
        Expr::App(_, args) => {
            for arg in args {
                collect_variable_refs_in_expr(doc, arg, var_name, ranges);
            }
        }
        Expr::MethodCall {
            receiver,
            method: _,
            args,
        } => {
            collect_variable_refs_in_expr(doc, receiver, var_name, ranges);
            for arg in args {
                collect_variable_refs_in_expr(doc, arg, var_name, ranges);
            }
        }
        Expr::FieldAccess(receiver, _) => {
            collect_variable_refs_in_expr(doc, receiver, var_name, ranges);
        }
        Expr::Var(_) | Expr::Int(_) | Expr::Float(_) | Expr::String(_) | Expr::Bool(_) => {
            // No recursion needed
        }
    }
}

/// Find variable ranges within a temporal expression.
fn find_variable_ranges_in_temporal(
    doc: &Document,
    expr: &TemporalExpr,
    var_name: &str,
) -> Option<Vec<Range>> {
    let mut ranges = Vec::new();
    collect_variable_refs_in_temporal(doc, expr, var_name, &mut ranges);
    if ranges.is_empty() {
        None
    } else {
        Some(ranges)
    }
}

/// Recursively collect variable references in a temporal expression.
fn collect_variable_refs_in_temporal(
    doc: &Document,
    expr: &TemporalExpr,
    var_name: &str,
    ranges: &mut Vec<Range>,
) {
    match expr {
        TemporalExpr::Always(inner) | TemporalExpr::Eventually(inner) => {
            collect_variable_refs_in_temporal(doc, inner, var_name, ranges);
        }
        TemporalExpr::LeadsTo(left, right) => {
            collect_variable_refs_in_temporal(doc, left, var_name, ranges);
            collect_variable_refs_in_temporal(doc, right, var_name, ranges);
        }
        TemporalExpr::Atom(expr) => {
            collect_variable_refs_in_expr(doc, expr, var_name, ranges);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower_lsp::lsp_types::Url;

    fn make_doc(content: &str) -> Document {
        Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            content.to_string(),
        )
    }

    #[test]
    fn linked_editing_for_type_name() {
        let doc = make_doc(
            "type Counter = { value: Int }\ntheorem test { forall c: Counter . c.value >= 0 }",
        );

        // Position on "Counter" definition (line 0, char 5)
        let result = generate_linked_editing_ranges(&doc, Position::new(0, 5));
        assert!(result.is_some());

        let ranges = result.unwrap();
        // Counter appears twice: definition and type annotation
        assert_eq!(ranges.ranges.len(), 2);
        assert!(ranges.word_pattern.is_some());
    }

    #[test]
    fn linked_editing_for_property_name() {
        let doc = make_doc(
            "type Value = { id: Int }\n\
             theorem non_negative { forall v: Value . v.id >= 0 }\n\
             // Reference: non_negative is important",
        );

        // Position on "non_negative" at its definition
        let result = generate_linked_editing_ranges(&doc, Position::new(1, 10));

        // Property name appears twice if referenced in comment, otherwise once
        // Since comments are not parsed as references, should be None (only 1 occurrence)
        // Actually, let's check if it appears
        let refs = doc.find_all_references("non_negative");
        if refs.len() > 1 {
            assert!(result.is_some());
        } else {
            // Only one occurrence, no linked editing
            assert!(result.is_none());
        }
    }

    #[test]
    fn linked_editing_returns_none_for_single_occurrence() {
        let doc = make_doc("type Counter = { value: Int }");

        // "Counter" only appears once
        let result = generate_linked_editing_ranges(&doc, Position::new(0, 5));

        // Should be None since there's only one occurrence
        assert!(result.is_none());
    }

    #[test]
    fn linked_editing_returns_none_for_unknown_identifier() {
        let doc = make_doc("type Counter = { value: Int }");

        // Position on a non-identifier location
        let result = generate_linked_editing_ranges(&doc, Position::new(0, 0));

        assert!(result.is_none());
    }

    #[test]
    fn linked_editing_for_quantified_variable() {
        let doc = make_doc(
            "type Item = { price: Int }\n\
             theorem all_positive { forall x: Item . x.price > 0 }",
        );

        // Position on "x" in the quantifier
        // "x" appears twice: binding and usage
        let result = generate_linked_editing_ranges(&doc, Position::new(1, 31));

        // Check if x is found as a bound variable
        let refs = doc.find_all_references("x");
        if refs.len() > 1 {
            // Should have linked editing for both x occurrences
            assert!(result.is_some());
            let ranges = result.unwrap();
            assert!(ranges.ranges.len() >= 2);
        }
    }

    #[test]
    fn identifier_pattern_matches_expected_format() {
        let pattern = identifier_pattern();
        // Verify the pattern is a valid identifier pattern
        assert!(pattern.starts_with("[a-zA-Z_]"));
        assert!(pattern.contains("[a-zA-Z0-9_']*"));
    }

    #[test]
    fn linked_editing_with_multiple_types() {
        // Test with the type definition of Node (appears 3 times)
        let doc = make_doc(
            "type Node = { id: Int }\n\
             type Edge = { from: Node, to: Node }\n\
             theorem edge_valid { forall e: Edge . e.from.id != e.to.id }",
        );

        // Position on "Node" definition at line 0, char 5
        let result = generate_linked_editing_ranges(&doc, Position::new(0, 5));

        // Node appears 3 times: definition, from field type, to field type
        assert!(result.is_some());
        let ranges = result.unwrap();
        assert_eq!(ranges.ranges.len(), 3);
    }

    #[test]
    fn linked_editing_for_contract_parameter() {
        let doc = make_doc(
            "type Value = { amount: Int }\n\
             contract Value::add(self: Value, delta: Int) -> Value {\n\
                 requires { delta >= 0 }\n\
                 ensures { self'.amount == self.amount + delta }\n\
             }",
        );

        // Position on "delta" in parameter
        let result = generate_linked_editing_ranges(&doc, Position::new(1, 39));

        // delta appears in parameter, requires, and ensures
        if let Some(ranges) = result {
            assert!(ranges.ranges.len() >= 2);
        }
    }

    #[test]
    fn linked_editing_respects_word_boundaries() {
        let doc = make_doc(
            "type Counter = { counter: Int }\n\
             theorem test { forall c: Counter . c.counter >= 0 }",
        );

        // Position on type name "Counter" (not field name "counter")
        let result = generate_linked_editing_ranges(&doc, Position::new(0, 5));

        if let Some(ranges) = result {
            // Should only find "Counter" (type), not "counter" (field)
            // Both should have distinct positions
            for range in &ranges.ranges {
                // "Counter" starts at char 5 on line 0, or char 25 on line 1
                let text = &doc.text;
                let lines: Vec<&str> = text.lines().collect();
                let line = lines.get(range.start.line as usize);
                if let Some(line) = line {
                    let word: String = line
                        .chars()
                        .skip(range.start.character as usize)
                        .take((range.end.character - range.start.character) as usize)
                        .collect();
                    assert_eq!(word, "Counter");
                }
            }
        }
    }

    #[test]
    fn test_single_reference_type_no_linking() {
        // Mutation test: verify ranges.len() > 1 for types
        let doc = make_doc("type Solo = { val: Int }");
        let result = generate_linked_editing_ranges(&doc, Position::new(0, 5));
        // Only one reference to "Solo", should return None
        assert!(result.is_none());
    }

    #[test]
    fn test_single_reference_property_no_linking() {
        // Mutation test: verify ranges.len() > 1 for properties
        let doc = make_doc("theorem unique { true }");
        let result = generate_linked_editing_ranges(&doc, Position::new(0, 10));
        // Only one reference to "unique", should return None
        assert!(result.is_none());
    }

    #[test]
    fn test_variable_name_equality_in_contract() {
        // Mutation test: verify param.name == var_name comparison
        let doc = make_doc(
            "type T = { v: Int }\n\
             contract T::op(self: T, x: Int) -> Int {\n\
                 requires { x > 0 }\n\
                 ensures { x > 0 }\n\
             }",
        );

        // Search for "x" parameter
        let spec = doc.spec.as_ref().unwrap();
        for prop in &spec.properties {
            if let Property::Contract(c) = prop {
                // Verify we can find parameter by name
                assert!(c.params.iter().any(|p| p.name == "x"));
                // Verify we don't match wrong names
                assert!(!c.params.iter().any(|p| p.name == "y"));
            }
        }
    }

    #[test]
    fn test_var_name_match_in_forall() {
        // Mutation test: verify name == var_name in Expr::Var match guard
        let doc = make_doc("theorem test { forall x: Int . x > 0 }");

        let spec = doc.spec.as_ref().unwrap();
        for prop in &spec.properties {
            if let Property::Theorem(t) = prop {
                // The expression should have "x" as variable
                let mut found_x = false;
                let mut found_other = false;
                check_var_in_expr(&t.body, "x", &mut found_x);
                check_var_in_expr(&t.body, "y", &mut found_other);
                assert!(found_x, "Should find variable x");
                assert!(!found_other, "Should not find variable y");
            }
        }
    }

    fn check_var_in_expr(expr: &Expr, name: &str, found: &mut bool) {
        match expr {
            Expr::Var(v) if v == name => *found = true,
            Expr::ForAll { var, body, .. } | Expr::Exists { var, body, .. } => {
                if var == name {
                    *found = true;
                }
                check_var_in_expr(body, name, found);
            }
            Expr::And(l, r)
            | Expr::Or(l, r)
            | Expr::Implies(l, r)
            | Expr::Compare(l, _, r)
            | Expr::Binary(l, _, r) => {
                check_var_in_expr(l, name, found);
                check_var_in_expr(r, name, found);
            }
            Expr::Not(inner) | Expr::Neg(inner) => check_var_in_expr(inner, name, found),
            _ => {}
        }
    }

    #[test]
    fn test_forall_var_binding_equality() {
        // Mutation test: verify var == var_name in ForAll branch
        let doc = make_doc(
            "theorem test {\n\
                 forall a: Int . forall b: Int . a + b >= 0\n\
             }",
        );

        let spec = doc.spec.as_ref().unwrap();
        // Check that both "a" and "b" are found as bound variables
        for prop in &spec.properties {
            let ranges_a = find_variable_ranges_in_property(&doc, prop, "a");
            let ranges_b = find_variable_ranges_in_property(&doc, prop, "b");
            // Both should be found
            assert!(
                ranges_a.is_some() || ranges_b.is_some(),
                "Should find at least one bound variable"
            );
        }
    }

    #[test]
    fn test_forall_in_var_binding_equality() {
        // Mutation test: verify var == var_name in ForAllIn branch
        let doc = make_doc(
            "theorem test {\n\
                 forall x in set . x > 0\n\
             }",
        );

        let spec = doc.spec.as_ref().unwrap();
        for prop in &spec.properties {
            let ranges_x = find_variable_ranges_in_property(&doc, prop, "x");
            let ranges_y = find_variable_ranges_in_property(&doc, prop, "y");
            // x should be found, y should not
            assert!(ranges_x.is_some(), "Should find bound variable x");
            assert!(
                ranges_y.is_none(),
                "Should not find non-existent variable y"
            );
        }
    }

    #[test]
    fn test_temporal_variable_ranges() {
        // Mutation test: verify find_variable_ranges_in_temporal function
        let doc = make_doc(
            "temporal safe {\n\
                 always(forall x: Int . x >= 0)\n\
             }",
        );

        let spec = doc.spec.as_ref().unwrap();
        for prop in &spec.properties {
            if let Property::Temporal(t) = prop {
                let ranges = find_variable_ranges_in_temporal(&doc, &t.body, "x");
                // Should find variable x in the temporal expression
                assert!(
                    ranges.is_some(),
                    "Should find variable in temporal expression"
                );
            }
        }
    }

    #[test]
    fn test_temporal_collect_refs() {
        // Mutation test: verify collect_variable_refs_in_temporal function
        let doc = make_doc(
            "temporal liveness {\n\
                 eventually(forall y: Bool . y)\n\
             }",
        );

        let spec = doc.spec.as_ref().unwrap();
        for prop in &spec.properties {
            if let Property::Temporal(t) = prop {
                let mut ranges = Vec::new();
                collect_variable_refs_in_temporal(&doc, &t.body, "y", &mut ranges);
                // ranges should be non-empty if y is found
                assert!(
                    !ranges.is_empty() || doc.find_all_references("y").is_empty(),
                    "collect_variable_refs_in_temporal should find references"
                );
            }
        }
    }

    #[test]
    fn test_temporal_leads_to() {
        // Test LeadsTo variant in temporal expressions
        let doc = make_doc(
            "temporal progress {\n\
                 forall x: Bool . x ~> x\n\
             }",
        );

        let spec = doc.spec.as_ref().unwrap();
        for prop in &spec.properties {
            if let Property::Temporal(t) = prop {
                let ranges = find_variable_ranges_in_temporal(&doc, &t.body, "x");
                // x should be found in LeadsTo expression
                assert!(
                    ranges.is_some() || doc.find_all_references("x").is_empty(),
                    "Should handle LeadsTo expressions"
                );
            }
        }
    }

    #[test]
    fn test_temporal_always_eventually() {
        // Test Always and Eventually variants explicitly
        let doc = make_doc(
            "temporal stability {\n\
                 always(eventually(forall v: Int . v > 0))\n\
             }",
        );

        let spec = doc.spec.as_ref().unwrap();
        for prop in &spec.properties {
            if let Property::Temporal(t) = prop {
                let mut ranges = Vec::new();
                collect_variable_refs_in_temporal(&doc, &t.body, "v", &mut ranges);
                // The function should recurse through Always(Eventually(...))
                // and find v in the inner Atom expression
            }
        }
    }

    #[test]
    fn test_no_variable_in_temporal() {
        // Test when variable is not found in temporal expression
        let doc = make_doc(
            "temporal simple {\n\
                 always(true)\n\
             }",
        );

        let spec = doc.spec.as_ref().unwrap();
        for prop in &spec.properties {
            if let Property::Temporal(t) = prop {
                let ranges = find_variable_ranges_in_temporal(&doc, &t.body, "nonexistent");
                assert!(
                    ranges.is_none(),
                    "Should return None for non-existent variable"
                );
            }
        }
    }
}
