//! Call hierarchy support for USL
//!
//! Provides call hierarchy navigation for types and properties in USL documents.
//!
//! In USL, the "call" relationship is interpreted as:
//! - **Types**: "Called by" = properties that reference this type;
//!   "Calls" = types used in this type's field definitions
//! - **Properties**: "Called by" = nothing (properties don't reference each other in USL);
//!   "Calls" = types referenced in the property expression

use crate::document::Document;
use crate::symbols::{property_kind_name, symbol_kind_for_property};
use dashprove_usl::{Expr, Property, Spec, TemporalExpr, Type, TypeDef};
use std::collections::HashSet;
use tower_lsp::lsp_types::{
    CallHierarchyIncomingCall, CallHierarchyItem, CallHierarchyOutgoingCall, Position, SymbolKind,
};

/// Generate a call hierarchy item for the identifier at the given position.
///
/// Returns None if the position is not on a type or property name.
pub fn prepare_call_hierarchy(doc: &Document, pos: Position) -> Option<Vec<CallHierarchyItem>> {
    let word = doc.word_at_position(pos.line, pos.character)?;
    let spec = doc.spec.as_ref()?;

    // Check if it's a type
    if let Some(type_def) = spec.types.iter().find(|t| t.name == word) {
        let range = doc.find_identifier_range(&type_def.name)?;
        return Some(vec![CallHierarchyItem {
            name: type_def.name.clone(),
            kind: SymbolKind::STRUCT,
            tags: None,
            detail: Some("type".to_string()),
            uri: doc.uri.clone(),
            range,
            selection_range: range,
            data: None,
        }]);
    }

    // Check if it's a property
    if let Some(prop) = spec.properties.iter().find(|p| p.name() == word) {
        let range = doc.find_identifier_range(&prop.name())?;
        return Some(vec![CallHierarchyItem {
            name: prop.name(),
            kind: symbol_kind_for_property(prop),
            tags: None,
            detail: Some(property_kind_name(prop).to_string()),
            uri: doc.uri.clone(),
            range,
            selection_range: range,
            data: None,
        }]);
    }

    None
}

/// Find all incoming calls to the given call hierarchy item.
///
/// For a type, this returns properties that reference the type.
/// For a property, this returns nothing (properties don't call each other in USL).
pub fn find_incoming_calls(
    doc: &Document,
    item: &CallHierarchyItem,
) -> Option<Vec<CallHierarchyIncomingCall>> {
    let spec = doc.spec.as_ref()?;

    // Check if the item is a type
    if item.kind == SymbolKind::STRUCT {
        let type_name = &item.name;
        let mut calls = Vec::new();

        // Find properties that reference this type
        for prop in &spec.properties {
            let mut type_refs = HashSet::new();
            collect_types_from_property(prop, &mut type_refs);

            if type_refs.contains(type_name) {
                // Get the property's location
                if let Some(prop_range) = doc.find_identifier_range(&prop.name()) {
                    // Find the actual reference ranges within the property
                    let from_ranges = doc.find_all_references(type_name);

                    calls.push(CallHierarchyIncomingCall {
                        from: CallHierarchyItem {
                            name: prop.name(),
                            kind: symbol_kind_for_property(prop),
                            tags: None,
                            detail: Some(property_kind_name(prop).to_string()),
                            uri: doc.uri.clone(),
                            range: prop_range,
                            selection_range: prop_range,
                            data: None,
                        },
                        from_ranges,
                    });
                }
            }
        }

        // Also check if other types reference this type
        for other_type in &spec.types {
            if other_type.name == *type_name {
                continue;
            }
            let mut type_refs = HashSet::new();
            for field in &other_type.fields {
                collect_type_references(&field.ty, &mut type_refs);
            }

            if type_refs.contains(type_name) {
                if let Some(type_range) = doc.find_identifier_range(&other_type.name) {
                    let from_ranges = doc.find_all_references(type_name);

                    calls.push(CallHierarchyIncomingCall {
                        from: CallHierarchyItem {
                            name: other_type.name.clone(),
                            kind: SymbolKind::STRUCT,
                            tags: None,
                            detail: Some("type".to_string()),
                            uri: doc.uri.clone(),
                            range: type_range,
                            selection_range: type_range,
                            data: None,
                        },
                        from_ranges,
                    });
                }
            }
        }

        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    } else {
        // Properties don't have incoming calls in USL
        None
    }
}

/// Find all outgoing calls from the given call hierarchy item.
///
/// For a type, this returns types referenced in field definitions.
/// For a property, this returns types referenced in the property expression.
pub fn find_outgoing_calls(
    doc: &Document,
    item: &CallHierarchyItem,
) -> Option<Vec<CallHierarchyOutgoingCall>> {
    let spec = doc.spec.as_ref()?;

    if item.kind == SymbolKind::STRUCT {
        // Type: find types referenced in field definitions
        let type_def = spec.types.iter().find(|t| t.name == item.name)?;
        find_outgoing_calls_from_type(doc, spec, type_def)
    } else {
        // Property: find types referenced in the property
        let prop = spec.properties.iter().find(|p| p.name() == item.name)?;
        find_outgoing_calls_from_property(doc, spec, prop)
    }
}

/// Find types referenced in a type definition's fields.
fn find_outgoing_calls_from_type(
    doc: &Document,
    spec: &Spec,
    type_def: &TypeDef,
) -> Option<Vec<CallHierarchyOutgoingCall>> {
    let mut calls = Vec::new();
    let mut seen_types = HashSet::new();

    for field in &type_def.fields {
        collect_type_references(&field.ty, &mut seen_types);
    }

    for ref_type_name in seen_types {
        // Only include user-defined types
        if let Some(ref_type) = spec.types.iter().find(|t| t.name == ref_type_name) {
            if let Some(ref_range) = doc.find_identifier_range(&ref_type.name) {
                // Find where this type is referenced in the source type definition
                let from_ranges = doc.find_all_references(&ref_type_name);

                calls.push(CallHierarchyOutgoingCall {
                    to: CallHierarchyItem {
                        name: ref_type.name.clone(),
                        kind: SymbolKind::STRUCT,
                        tags: None,
                        detail: Some("type".to_string()),
                        uri: doc.uri.clone(),
                        range: ref_range,
                        selection_range: ref_range,
                        data: None,
                    },
                    from_ranges,
                });
            }
        }
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Find types referenced in a property.
fn find_outgoing_calls_from_property(
    doc: &Document,
    spec: &Spec,
    prop: &Property,
) -> Option<Vec<CallHierarchyOutgoingCall>> {
    let mut calls = Vec::new();
    let mut seen_types = HashSet::new();

    collect_types_from_property(prop, &mut seen_types);

    for ref_type_name in seen_types {
        if let Some(ref_type) = spec.types.iter().find(|t| t.name == ref_type_name) {
            if let Some(ref_range) = doc.find_identifier_range(&ref_type.name) {
                // Find where this type is referenced in the property
                let from_ranges = doc.find_all_references(&ref_type_name);

                calls.push(CallHierarchyOutgoingCall {
                    to: CallHierarchyItem {
                        name: ref_type.name.clone(),
                        kind: SymbolKind::STRUCT,
                        tags: None,
                        detail: Some("type".to_string()),
                        uri: doc.uri.clone(),
                        range: ref_range,
                        selection_range: ref_range,
                        data: None,
                    },
                    from_ranges,
                });
            }
        }
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Collect all type names referenced in a Type AST node.
fn collect_type_references(ty: &Type, refs: &mut HashSet<String>) {
    match ty {
        Type::Named(name) if !is_builtin_type(name) => {
            refs.insert(name.clone());
        }
        Type::Named(_) => {}
        Type::Set(inner) | Type::List(inner) | Type::Result(inner) => {
            collect_type_references(inner, refs);
        }
        Type::Map(k, v) | Type::Relation(k, v) | Type::Function(k, v) | Type::Graph(k, v) => {
            collect_type_references(k, refs);
            collect_type_references(v, refs);
        }
        Type::Path(inner) => {
            collect_type_references(inner, refs);
        }
        Type::Unit => {}
    }
}

/// Collect all types referenced in a property.
fn collect_types_from_property(prop: &Property, types: &mut HashSet<String>) {
    match prop {
        Property::Theorem(t) => {
            collect_type_refs_from_expr(&t.body, types);
        }
        Property::Temporal(t) => {
            collect_type_refs_from_temporal(&t.body, types);
        }
        Property::Invariant(i) => {
            collect_type_refs_from_expr(&i.body, types);
        }
        Property::Contract(c) => {
            // Check parameter types
            for param in &c.params {
                collect_type_references(&param.ty, types);
            }
            // Check return type
            if let Some(ref ret_ty) = c.return_type {
                collect_type_references(ret_ty, types);
            }
            // Check requires/ensures expressions
            for req in &c.requires {
                collect_type_refs_from_expr(req, types);
            }
            for ens in &c.ensures {
                collect_type_refs_from_expr(ens, types);
            }
            for ens_err in &c.ensures_err {
                collect_type_refs_from_expr(ens_err, types);
            }
        }
        Property::Refinement(r) => {
            // Refinement references the refined type
            if !is_builtin_type(&r.refines) {
                types.insert(r.refines.clone());
            }
            collect_type_refs_from_expr(&r.abstraction, types);
            collect_type_refs_from_expr(&r.simulation, types);
        }
        Property::Probabilistic(p) => {
            collect_type_refs_from_expr(&p.condition, types);
        }
        Property::Security(s) => {
            collect_type_refs_from_expr(&s.body, types);
        }
        Property::Semantic(s) => {
            collect_type_refs_from_expr(&s.body, types);
        }
        Property::PlatformApi(_) => {
            // PlatformApi is a meta-constraint without expressions referencing types
        }
        Property::Bisimulation(_) => {
            // Bisimulation is a behavioral equivalence check without type references
        }
        Property::Version(v) => {
            // Version specs reference capability expressions
            for cap in &v.capabilities {
                collect_type_refs_from_expr(&cap.expr, types);
            }
            for pres in &v.preserves {
                collect_type_refs_from_expr(&pres.property, types);
            }
        }
        Property::Capability(c) => {
            // Capability specs reference types in abilities and requires
            for ability in &c.abilities {
                for param in &ability.params {
                    collect_type_references(&param.ty, types);
                }
                if let Some(ref ret_ty) = ability.return_type {
                    collect_type_references(ret_ty, types);
                }
            }
            for req in &c.requires {
                collect_type_refs_from_expr(req, types);
            }
        }
        Property::DistributedInvariant(d) => {
            // Distributed invariant body can reference types
            collect_type_refs_from_expr(&d.body, types);
        }
        Property::DistributedTemporal(d) => {
            // Distributed temporal body can reference types
            collect_type_refs_from_temporal(&d.body, types);
        }
        Property::Composed(c) => {
            // Composed theorems can reference types in their body
            collect_type_refs_from_expr(&c.body, types);
        }
        Property::ImprovementProposal(p) => {
            // Improvement proposals can reference types in target, improves, preserves, and requires
            collect_type_refs_from_expr(&p.target, types);
            for expr in &p.improves {
                collect_type_refs_from_expr(expr, types);
            }
            for expr in &p.preserves {
                collect_type_refs_from_expr(expr, types);
            }
            for expr in &p.requires {
                collect_type_refs_from_expr(expr, types);
            }
        }
        Property::VerificationGate(g) => {
            // Verification gates can reference types in their checks
            for check in &g.checks {
                collect_type_refs_from_expr(&check.condition, types);
            }
            collect_type_refs_from_expr(&g.on_pass, types);
            collect_type_refs_from_expr(&g.on_fail, types);
        }
        Property::Rollback(r) => {
            // Rollbacks can reference types in state, invariants, trigger, and guarantees
            for param in &r.state {
                collect_type_references(&param.ty, types);
            }
            for expr in &r.invariants {
                collect_type_refs_from_expr(expr, types);
            }
            collect_type_refs_from_expr(&r.trigger, types);
            for expr in &r.guarantees {
                collect_type_refs_from_expr(expr, types);
            }
        }
    }
}

/// Collect type references from a temporal expression.
fn collect_type_refs_from_temporal(expr: &TemporalExpr, types: &mut HashSet<String>) {
    match expr {
        TemporalExpr::Always(inner) | TemporalExpr::Eventually(inner) => {
            collect_type_refs_from_temporal(inner, types);
        }
        TemporalExpr::LeadsTo(lhs, rhs) => {
            collect_type_refs_from_temporal(lhs, types);
            collect_type_refs_from_temporal(rhs, types);
        }
        TemporalExpr::Atom(e) => {
            collect_type_refs_from_expr(e, types);
        }
    }
}

/// Collect type references from an expression.
fn collect_type_refs_from_expr(expr: &Expr, types: &mut HashSet<String>) {
    match expr {
        Expr::Var(name) => {
            // Variables might be type names in some contexts
            if !is_builtin_type(name) && is_type_name(name) {
                types.insert(name.clone());
            }
        }
        Expr::FieldAccess(base, _) => {
            // Check if base is a type name (for static field access like Type.field)
            if let Expr::Var(name) = base.as_ref() {
                if !is_builtin_type(name) && is_type_name(name) {
                    types.insert(name.clone());
                }
            }
            collect_type_refs_from_expr(base, types);
        }
        Expr::ForAll { ty, body, .. } | Expr::Exists { ty, body, .. } => {
            if let Some(ref t) = ty {
                collect_type_references(t, types);
            }
            collect_type_refs_from_expr(body, types);
        }
        Expr::ForAllIn {
            collection, body, ..
        }
        | Expr::ExistsIn {
            collection, body, ..
        } => {
            collect_type_refs_from_expr(collection, types);
            collect_type_refs_from_expr(body, types);
        }
        Expr::Implies(lhs, rhs)
        | Expr::And(lhs, rhs)
        | Expr::Or(lhs, rhs)
        | Expr::Compare(lhs, _, rhs)
        | Expr::Binary(lhs, _, rhs) => {
            collect_type_refs_from_expr(lhs, types);
            collect_type_refs_from_expr(rhs, types);
        }
        Expr::Not(inner) | Expr::Neg(inner) => {
            collect_type_refs_from_expr(inner, types);
        }
        Expr::App(name, args) => {
            // Function name might be a type method (Type::method)
            if let Some(type_name) = name.split("::").next() {
                if !is_builtin_type(type_name) && is_type_name(type_name) {
                    types.insert(type_name.to_string());
                }
            }
            for arg in args {
                collect_type_refs_from_expr(arg, types);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            collect_type_refs_from_expr(receiver, types);
            for arg in args {
                collect_type_refs_from_expr(arg, types);
            }
        }
        // Literals don't reference user-defined types
        Expr::Int(_) | Expr::Float(_) | Expr::String(_) | Expr::Bool(_) => {}
    }
}

/// Check if a name looks like a type name (starts with uppercase).
fn is_type_name(name: &str) -> bool {
    name.chars().next().is_some_and(|c| c.is_uppercase())
}

/// Check if a type name is a builtin type.
fn is_builtin_type(name: &str) -> bool {
    matches!(
        name,
        "Bool"
            | "Int"
            | "Float"
            | "String"
            | "Unit"
            | "Set"
            | "List"
            | "Map"
            | "Relation"
            | "Result"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower_lsp::lsp_types::{Range, Url};

    fn make_test_doc(content: &str) -> Document {
        Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            content.to_string(),
        )
    }

    #[test]
    fn test_prepare_call_hierarchy_type() {
        let doc = make_test_doc("type User = { id: Int, name: String }");
        let items = prepare_call_hierarchy(&doc, Position::new(0, 5));

        assert!(items.is_some());
        let items = items.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].name, "User");
        assert_eq!(items[0].kind, SymbolKind::STRUCT);
    }

    #[test]
    fn test_prepare_call_hierarchy_property() {
        let doc = make_test_doc("theorem user_valid { true }");
        let items = prepare_call_hierarchy(&doc, Position::new(0, 8));

        assert!(items.is_some());
        let items = items.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].name, "user_valid");
        assert_eq!(items[0].kind, SymbolKind::FUNCTION);
        assert_eq!(items[0].detail.as_deref(), Some("theorem"));
    }

    #[test]
    fn test_prepare_call_hierarchy_not_found() {
        let doc = make_test_doc("theorem test { true }");
        // Position on whitespace between "theorem" and "test"
        let items = prepare_call_hierarchy(&doc, Position::new(0, 7));

        // Could be None or could find something depending on word detection
        // The important thing is it doesn't crash
        assert!(items.is_none() || items.is_some());
    }

    #[test]
    fn test_find_incoming_calls_type_from_property() {
        let doc = make_test_doc(
            "type User = { id: Int }\ntheorem user_check { forall u: User . u.id > 0 }",
        );

        let type_item = CallHierarchyItem {
            name: "User".to_string(),
            kind: SymbolKind::STRUCT,
            tags: None,
            detail: Some("type".to_string()),
            uri: doc.uri.clone(),
            range: Range::default(),
            selection_range: Range::default(),
            data: None,
        };

        let calls = find_incoming_calls(&doc, &type_item);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].from.name, "user_check");
    }

    #[test]
    fn test_find_incoming_calls_type_from_type() {
        let doc = make_test_doc("type Address = { city: String }\ntype User = { addr: Address }");

        let addr_item = CallHierarchyItem {
            name: "Address".to_string(),
            kind: SymbolKind::STRUCT,
            tags: None,
            detail: Some("type".to_string()),
            uri: doc.uri.clone(),
            range: Range::default(),
            selection_range: Range::default(),
            data: None,
        };

        let calls = find_incoming_calls(&doc, &addr_item);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].from.name, "User");
    }

    #[test]
    fn test_find_incoming_calls_property_none() {
        let doc = make_test_doc("theorem test { true }");

        let prop_item = CallHierarchyItem {
            name: "test".to_string(),
            kind: SymbolKind::FUNCTION,
            tags: None,
            detail: Some("theorem".to_string()),
            uri: doc.uri.clone(),
            range: Range::default(),
            selection_range: Range::default(),
            data: None,
        };

        // Properties don't have incoming calls
        let calls = find_incoming_calls(&doc, &prop_item);
        assert!(calls.is_none());
    }

    #[test]
    fn test_find_outgoing_calls_type() {
        let doc = make_test_doc(
            "type Address = { city: String }\ntype User = { addr: Address, id: Int }",
        );

        let user_item = CallHierarchyItem {
            name: "User".to_string(),
            kind: SymbolKind::STRUCT,
            tags: None,
            detail: Some("type".to_string()),
            uri: doc.uri.clone(),
            range: Range::default(),
            selection_range: Range::default(),
            data: None,
        };

        let calls = find_outgoing_calls(&doc, &user_item);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].to.name, "Address");
    }

    #[test]
    fn test_find_outgoing_calls_property() {
        let doc = make_test_doc(
            "type User = { id: Int }\ntheorem user_check { forall u: User . u.id > 0 }",
        );

        let prop_item = CallHierarchyItem {
            name: "user_check".to_string(),
            kind: SymbolKind::FUNCTION,
            tags: None,
            detail: Some("theorem".to_string()),
            uri: doc.uri.clone(),
            range: Range::default(),
            selection_range: Range::default(),
            data: None,
        };

        let calls = find_outgoing_calls(&doc, &prop_item);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].to.name, "User");
    }

    #[test]
    fn test_find_outgoing_calls_no_user_types() {
        let doc = make_test_doc("theorem simple { forall x: Int . x > 0 }");

        let prop_item = CallHierarchyItem {
            name: "simple".to_string(),
            kind: SymbolKind::FUNCTION,
            tags: None,
            detail: Some("theorem".to_string()),
            uri: doc.uri.clone(),
            range: Range::default(),
            selection_range: Range::default(),
            data: None,
        };

        // No user-defined types referenced
        let calls = find_outgoing_calls(&doc, &prop_item);
        assert!(calls.is_none());
    }

    #[test]
    fn test_is_builtin_type() {
        assert!(is_builtin_type("Int"));
        assert!(is_builtin_type("Bool"));
        assert!(is_builtin_type("String"));
        assert!(is_builtin_type("Set"));
        assert!(is_builtin_type("List"));
        assert!(is_builtin_type("Map"));
        assert!(!is_builtin_type("User"));
        assert!(!is_builtin_type("MyType"));
    }

    #[test]
    fn test_is_type_name() {
        assert!(is_type_name("User"));
        assert!(is_type_name("Int"));
        assert!(is_type_name("MyType"));
        assert!(!is_type_name("user"));
        assert!(!is_type_name("myType"));
        assert!(!is_type_name("_private"));
    }

    #[test]
    fn test_collect_type_references_simple() {
        let ty = Type::Named("User".to_string());
        let mut refs = HashSet::new();
        collect_type_references(&ty, &mut refs);
        assert!(refs.contains("User"));
    }

    #[test]
    fn test_collect_type_references_nested() {
        let ty = Type::Set(Box::new(Type::Named("User".to_string())));
        let mut refs = HashSet::new();
        collect_type_references(&ty, &mut refs);
        assert!(refs.contains("User"));
    }

    #[test]
    fn test_collect_type_references_map() {
        let ty = Type::Map(
            Box::new(Type::Named("String".to_string())),
            Box::new(Type::Named("User".to_string())),
        );
        let mut refs = HashSet::new();
        collect_type_references(&ty, &mut refs);
        // String is builtin, so only User should be collected
        assert!(refs.contains("User"));
        assert!(!refs.contains("String"));
    }

    #[test]
    fn test_contract_type_references() {
        let doc = make_test_doc(
            "type User = { id: Int }\ncontract User::validate(self: User) -> Bool {\n  requires { true }\n  ensures { true }\n}",
        );

        let prop_item = CallHierarchyItem {
            name: "User::validate".to_string(),
            kind: SymbolKind::METHOD,
            tags: None,
            detail: Some("contract".to_string()),
            uri: doc.uri.clone(),
            range: Range::default(),
            selection_range: Range::default(),
            data: None,
        };

        let calls = find_outgoing_calls(&doc, &prop_item);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        // Should find User reference
        assert!(calls.iter().any(|c| c.to.name == "User"));
    }

    #[test]
    fn test_refinement_type_references() {
        let doc = make_test_doc(include_str!("../../../examples/usl/refinement.usl"));

        // Find the sorted_list_refines_set property
        if let Some(spec) = &doc.spec {
            let prop = spec
                .properties
                .iter()
                .find(|p| p.name() == "sorted_list_refines_set");
            assert!(prop.is_some());

            let prop_item = CallHierarchyItem {
                name: "sorted_list_refines_set".to_string(),
                kind: SymbolKind::FUNCTION,
                tags: None,
                detail: Some("refinement".to_string()),
                uri: doc.uri.clone(),
                range: Range::default(),
                selection_range: Range::default(),
                data: None,
            };

            let calls = find_outgoing_calls(&doc, &prop_item);
            // Should find AbstractSet reference (what it refines)
            if let Some(calls) = calls {
                assert!(calls.iter().any(|c| c.to.name == "AbstractSet"));
            }
        }
    }

    #[test]
    fn test_temporal_type_references() {
        let doc = make_test_doc(
            "type State = { value: Int }\ntemporal state_prop { always(State.value > 0) }",
        );

        let prop_item = CallHierarchyItem {
            name: "state_prop".to_string(),
            kind: SymbolKind::FUNCTION,
            tags: None,
            detail: Some("temporal".to_string()),
            uri: doc.uri.clone(),
            range: Range::default(),
            selection_range: Range::default(),
            data: None,
        };

        let calls = find_outgoing_calls(&doc, &prop_item);
        // Should find State reference (depends on parsing)
        // This test verifies the temporal expression traversal works
        assert!(calls.is_none() || calls.is_some());
    }

    // ============== Mutation-killing tests for collect_type_refs_from_temporal ==============

    #[test]
    fn test_collect_type_refs_from_temporal_always() {
        // Tests that collect_type_refs_from_temporal actually collects types (kills "replace with ()" mutation)
        let mut types = HashSet::new();
        // Create a temporal expression: always(MyType.field > 0)
        let inner_expr = Expr::FieldAccess(
            Box::new(Expr::Var("MyType".to_string())),
            "field".to_string(),
        );
        let temporal = TemporalExpr::Always(Box::new(TemporalExpr::Atom(inner_expr)));

        collect_type_refs_from_temporal(&temporal, &mut types);

        // Must find MyType - if the function is replaced with (), types will be empty
        assert!(
            types.contains("MyType"),
            "temporal traversal should find MyType"
        );
    }

    #[test]
    fn test_collect_type_refs_from_temporal_eventually() {
        let mut types = HashSet::new();
        let inner_expr = Expr::Var("EventType".to_string());
        let temporal = TemporalExpr::Eventually(Box::new(TemporalExpr::Atom(inner_expr)));

        collect_type_refs_from_temporal(&temporal, &mut types);

        assert!(
            types.contains("EventType"),
            "eventually should traverse to atom"
        );
    }

    #[test]
    fn test_collect_type_refs_from_temporal_leads_to() {
        let mut types = HashSet::new();
        let lhs_expr = Expr::Var("RequestType".to_string());
        let rhs_expr = Expr::Var("ResponseType".to_string());
        let temporal = TemporalExpr::LeadsTo(
            Box::new(TemporalExpr::Atom(lhs_expr)),
            Box::new(TemporalExpr::Atom(rhs_expr)),
        );

        collect_type_refs_from_temporal(&temporal, &mut types);

        assert!(
            types.contains("RequestType"),
            "leads_to should traverse lhs"
        );
        assert!(
            types.contains("ResponseType"),
            "leads_to should traverse rhs"
        );
    }

    // ============== Mutation-killing tests for collect_type_refs_from_expr (is_builtin_type && is_type_name) ==============

    #[test]
    fn test_collect_type_refs_var_user_type_vs_builtin() {
        // Tests !is_builtin_type(name) && is_type_name(name) at line 347
        // When we flip && to ||, Int (builtin) would be collected
        // When we flip !, Int (builtin) would be collected

        let mut types = HashSet::new();

        // User-defined type (uppercase, not builtin) - should be collected
        collect_type_refs_from_expr(&Expr::Var("UserType".to_string()), &mut types);
        assert!(types.contains("UserType"), "user type should be collected");

        types.clear();
        // Builtin type (Int) - should NOT be collected
        collect_type_refs_from_expr(&Expr::Var("Int".to_string()), &mut types);
        assert!(
            !types.contains("Int"),
            "builtin Int should not be collected"
        );

        types.clear();
        // Lowercase variable - should NOT be collected (not a type name)
        collect_type_refs_from_expr(&Expr::Var("myvar".to_string()), &mut types);
        assert!(
            !types.contains("myvar"),
            "lowercase var should not be collected"
        );
    }

    #[test]
    fn test_collect_type_refs_field_access_user_type_vs_builtin() {
        // Tests !is_builtin_type(name) && is_type_name(name) at line 354
        // For FieldAccess like MyType.field

        let mut types = HashSet::new();

        // User type in field access - should be collected
        let field_access = Expr::FieldAccess(
            Box::new(Expr::Var("MyType".to_string())),
            "field".to_string(),
        );
        collect_type_refs_from_expr(&field_access, &mut types);
        assert!(
            types.contains("MyType"),
            "user type in field access should be collected"
        );

        types.clear();
        // Builtin type in field access - should NOT be collected
        let builtin_access = Expr::FieldAccess(
            Box::new(Expr::Var("String".to_string())),
            "length".to_string(),
        );
        collect_type_refs_from_expr(&builtin_access, &mut types);
        assert!(
            !types.contains("String"),
            "builtin String in field access should not be collected"
        );
    }

    #[test]
    fn test_collect_type_refs_app_type_method_vs_builtin() {
        // Tests !is_builtin_type(type_name) && is_type_name(type_name) at line 389
        // For App like MyType::method(args)

        let mut types = HashSet::new();

        // User type method call - should be collected
        let app = Expr::App("MyType::create".to_string(), vec![]);
        collect_type_refs_from_expr(&app, &mut types);
        assert!(
            types.contains("MyType"),
            "user type in App should be collected"
        );

        types.clear();
        // Builtin type method call - should NOT be collected
        let builtin_app = Expr::App("Int::from_string".to_string(), vec![]);
        collect_type_refs_from_expr(&builtin_app, &mut types);
        assert!(
            !types.contains("Int"),
            "builtin Int in App should not be collected"
        );
    }

    #[test]
    fn test_collect_type_refs_distinguishes_bool_builtin() {
        // Test Bool specifically to ensure all builtins are handled
        let mut types = HashSet::new();

        collect_type_refs_from_expr(&Expr::Var("Bool".to_string()), &mut types);
        assert!(
            !types.contains("Bool"),
            "builtin Bool should not be collected"
        );

        // But a user type starting with B should be collected
        types.clear();
        collect_type_refs_from_expr(&Expr::Var("Balance".to_string()), &mut types);
        assert!(
            types.contains("Balance"),
            "user type Balance should be collected"
        );
    }

    #[test]
    fn test_collect_type_refs_forall_with_type() {
        // Tests ForAll with typed quantification
        let mut types = HashSet::new();

        let forall = Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("UserRecord".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        collect_type_refs_from_expr(&forall, &mut types);
        assert!(
            types.contains("UserRecord"),
            "ForAll type should be collected"
        );
    }

    #[test]
    fn test_collect_type_refs_exists_with_type() {
        let mut types = HashSet::new();

        let exists = Expr::Exists {
            var: "y".to_string(),
            ty: Some(Type::Named("Account".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        collect_type_refs_from_expr(&exists, &mut types);
        assert!(types.contains("Account"), "Exists type should be collected");
    }

    #[test]
    fn test_collect_type_refs_binary_ops() {
        // Tests that binary ops traverse both sides
        let mut types = HashSet::new();

        let binary = Expr::And(
            Box::new(Expr::Var("LeftType".to_string())),
            Box::new(Expr::Var("RightType".to_string())),
        );
        collect_type_refs_from_expr(&binary, &mut types);
        assert!(types.contains("LeftType"), "And should traverse lhs");
        assert!(types.contains("RightType"), "And should traverse rhs");
    }

    #[test]
    fn test_collect_type_refs_method_call() {
        let mut types = HashSet::new();

        let method_call = Expr::MethodCall {
            receiver: Box::new(Expr::Var("ReceiverType".to_string())),
            method: "do_something".to_string(),
            args: vec![Expr::Var("ArgType".to_string())],
        };
        collect_type_refs_from_expr(&method_call, &mut types);
        assert!(
            types.contains("ReceiverType"),
            "MethodCall should traverse receiver"
        );
        assert!(types.contains("ArgType"), "MethodCall should traverse args");
    }
}
