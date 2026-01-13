//! Symbol-related helpers for USL language server.
//!
//! Provides document symbols, workspace symbols, and type/property formatting.

use crate::document::{Document, DocumentStore};
use dashprove_usl::{
    BinaryOp, ComparisonOp, Contract, Expr, Property, Refinement, Spec, Type, TypeDef,
};
use tower_lsp::lsp_types::*;

/// Helper for plural suffix (returns "s" for counts != 1).
pub fn plural(count: usize) -> &'static str {
    if count == 1 {
        ""
    } else {
        "s"
    }
}

/// Get hover info for a type or property defined in the spec.
pub fn type_or_property_info(spec: &Spec, name: &str) -> Option<String> {
    // Check types
    for type_def in &spec.types {
        if type_def.name == name {
            return Some(format_type_def(type_def));
        }
    }

    // Check properties
    for prop in &spec.properties {
        if prop.name() == name {
            return Some(format_property(prop));
        }
    }

    None
}

/// Format a type definition for display.
pub fn format_type_def(type_def: &TypeDef) -> String {
    let mut s = format!(
        "**type {}**\n\n```usl\ntype {} = {{\n",
        type_def.name, type_def.name
    );
    for field in &type_def.fields {
        s.push_str(&format!(
            "    {}: {},\n",
            field.name,
            format_type(&field.ty)
        ));
    }
    s.push_str("}\n```");
    s
}

/// Format a type for display.
pub fn format_type(ty: &Type) -> String {
    match ty {
        Type::Named(name) => name.clone(),
        Type::Set(inner) => format!("Set<{}>", format_type(inner)),
        Type::List(inner) => format!("List<{}>", format_type(inner)),
        Type::Map(k, v) => format!("Map<{}, {}>", format_type(k), format_type(v)),
        Type::Relation(a, b) => format!("Relation<{}, {}>", format_type(a), format_type(b)),
        Type::Function(from, to) => format!("{} -> {}", format_type(from), format_type(to)),
        Type::Result(inner) => format!("Result<{}>", format_type(inner)),
        Type::Graph(n, e) => format!("Graph<{}, {}>", format_type(n), format_type(e)),
        Type::Path(n) => format!("Path<{}>", format_type(n)),
        Type::Unit => "Unit".to_string(),
    }
}

/// Format a USL expression for display (simplified).
pub fn format_expr(expr: &Expr) -> String {
    match expr {
        Expr::Var(name) => name.clone(),
        Expr::Int(n) => n.to_string(),
        Expr::Float(f) => f.to_string(),
        Expr::String(s) => format!("\"{}\"", s),
        Expr::Bool(b) => b.to_string(),
        Expr::Not(e) => format!("not ({})", format_expr(e)),
        Expr::And(l, r) => format!("({}) and ({})", format_expr(l), format_expr(r)),
        Expr::Or(l, r) => format!("({}) or ({})", format_expr(l), format_expr(r)),
        Expr::Implies(l, r) => format!("({}) implies ({})", format_expr(l), format_expr(r)),
        Expr::Compare(l, op, r) => {
            let op_str = match op {
                ComparisonOp::Eq => "==",
                ComparisonOp::Ne => "!=",
                ComparisonOp::Lt => "<",
                ComparisonOp::Le => "<=",
                ComparisonOp::Gt => ">",
                ComparisonOp::Ge => ">=",
            };
            format!("({}) {} ({})", format_expr(l), op_str, format_expr(r))
        }
        Expr::Binary(l, op, r) => {
            let op_str = match op {
                BinaryOp::Add => "+",
                BinaryOp::Sub => "-",
                BinaryOp::Mul => "*",
                BinaryOp::Div => "/",
                BinaryOp::Mod => "%",
            };
            format!("({}) {} ({})", format_expr(l), op_str, format_expr(r))
        }
        Expr::Neg(e) => format!("-({})", format_expr(e)),
        Expr::ForAll { var, ty, body } => {
            let ty_str = ty
                .as_ref()
                .map(|t| format!(": {}", format_type(t)))
                .unwrap_or_default();
            format!("forall {}{} . {}", var, ty_str, format_expr(body))
        }
        Expr::Exists { var, ty, body } => {
            let ty_str = ty
                .as_ref()
                .map(|t| format!(": {}", format_type(t)))
                .unwrap_or_default();
            format!("exists {}{} . {}", var, ty_str, format_expr(body))
        }
        Expr::ForAllIn {
            var,
            collection,
            body,
        } => {
            format!(
                "forall {} in {} . {}",
                var,
                format_expr(collection),
                format_expr(body)
            )
        }
        Expr::ExistsIn {
            var,
            collection,
            body,
        } => {
            format!(
                "exists {} in {} . {}",
                var,
                format_expr(collection),
                format_expr(body)
            )
        }
        Expr::FieldAccess(base, field) => format!("{}.{}", format_expr(base), field),
        Expr::App(name, args) => {
            let a: Vec<String> = args.iter().map(format_expr).collect();
            format!("{}({})", name, a.join(", "))
        }
        Expr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let a: Vec<String> = args.iter().map(format_expr).collect();
            format!("{}.{}({})", format_expr(receiver), method, a.join(", "))
        }
    }
}

/// Format a property for display.
pub fn format_property(prop: &Property) -> String {
    match prop {
        Property::Theorem(t) => format!("**theorem {}**\n\nMathematical property", t.name),
        Property::Temporal(t) => format!("**temporal {}**\n\nTemporal logic property", t.name),
        Property::Contract(c) => format!(
            "**contract {}**\n\nFunction contract with {} preconditions and {} postconditions",
            c.type_path.join("::"),
            c.requires.len(),
            c.ensures.len()
        ),
        Property::Invariant(i) => format!("**invariant {}**\n\nState invariant", i.name),
        Property::Refinement(r) => {
            format!("**refinement {}** refines {}", r.name, r.refines)
        }
        Property::Probabilistic(p) => format!(
            "**probabilistic {}**\n\nProbability bound: {} {}",
            p.name, p.comparison, p.bound
        ),
        Property::Security(s) => format!("**security {}**\n\nSecurity property", s.name),
        Property::Semantic(s) => format!(
            "**semantic {}**\n\nSemantic/embedding-based property",
            s.name
        ),
        Property::PlatformApi(p) => {
            format!("**platform_api {}**\n\nPlatform API constraint", p.name)
        }
        Property::Bisimulation(b) => {
            format!(
                "**bisimulation {}**\n\nBehavioral equivalence check",
                b.name
            )
        }
        Property::Version(v) => {
            format!(
                "**version {}** improves {}\n\n{} capabilities, {} preserves",
                v.name,
                v.improves,
                v.capabilities.len(),
                v.preserves.len()
            )
        }
        Property::Capability(c) => {
            format!(
                "**capability {}**\n\n{} abilities, {} requires",
                c.name,
                c.abilities.len(),
                c.requires.len()
            )
        }
        Property::DistributedInvariant(d) => {
            format!(
                "**distributed invariant {}**\n\nMulti-agent coordination invariant",
                d.name
            )
        }
        Property::DistributedTemporal(d) => {
            format!(
                "**distributed temporal {}**\n\nMulti-agent temporal property with {} fairness constraints",
                d.name,
                d.fairness.len()
            )
        }
        Property::Composed(c) => {
            format!(
                "**composed {}**\n\nComposed theorem using {} dependencies",
                c.name,
                c.uses.len()
            )
        }
        Property::ImprovementProposal(p) => {
            format!(
                "**improvement_proposal {}**\n\nImprovement proposal with {} improves and {} preserves",
                p.name,
                p.improves.len(),
                p.preserves.len()
            )
        }
        Property::VerificationGate(g) => {
            format!(
                "**verification_gate {}**\n\nVerification gate with {} checks",
                g.name,
                g.checks.len()
            )
        }
        Property::Rollback(r) => {
            format!(
                "**rollback_spec {}**\n\nRollback spec with {} invariants",
                r.name,
                r.invariants.len()
            )
        }
    }
}

/// Get the kind name for a property.
pub fn property_kind_name(prop: &Property) -> &'static str {
    match prop {
        Property::Theorem(_) => "theorem",
        Property::Temporal(_) => "temporal",
        Property::Contract(_) => "contract",
        Property::Invariant(_) => "invariant",
        Property::Refinement(_) => "refinement",
        Property::Probabilistic(_) => "probabilistic",
        Property::Security(_) => "security",
        Property::Semantic(_) => "semantic",
        Property::PlatformApi(_) => "platform_api",
        Property::Bisimulation(_) => "bisimulation",
        Property::Version(_) => "version",
        Property::Capability(_) => "capability",
        Property::DistributedInvariant(_) => "distributed_invariant",
        Property::DistributedTemporal(_) => "distributed_temporal",
        Property::Composed(_) => "composed",
        Property::ImprovementProposal(_) => "improvement_proposal",
        Property::VerificationGate(_) => "verification_gate",
        Property::Rollback(_) => "rollback",
    }
}

/// Map property to LSP symbol kind.
pub fn symbol_kind_for_property(prop: &Property) -> SymbolKind {
    match prop {
        Property::Contract(_) => SymbolKind::METHOD,
        Property::Theorem(_)
        | Property::Temporal(_)
        | Property::Invariant(_)
        | Property::Refinement(_)
        | Property::Probabilistic(_)
        | Property::Security(_)
        | Property::Semantic(_)
        | Property::PlatformApi(_)
        | Property::Bisimulation(_)
        | Property::Version(_)
        | Property::Capability(_)
        | Property::DistributedInvariant(_)
        | Property::DistributedTemporal(_)
        | Property::Composed(_)
        | Property::ImprovementProposal(_)
        | Property::VerificationGate(_)
        | Property::Rollback(_) => SymbolKind::FUNCTION,
    }
}

/// Check if the given range represents a definition site for an identifier.
///
/// Returns true if this is where the type or property is defined (WRITE),
/// false if it's a reference/usage (READ).
pub fn is_definition_site(spec: &Spec, name: &str, range: &Range, text: &str) -> bool {
    // Get approximate byte offset for the start of the range
    let line = range.start.line as usize;
    let char = range.start.character as usize;

    // Simple heuristic: check the text before the identifier
    // Type definitions: "type <Name> ="
    // Property definitions: "theorem <Name> {", "temporal <Name> {", etc.

    // Get the line text
    let lines: Vec<&str> = text.lines().collect();
    if line >= lines.len() {
        return false;
    }

    let line_text = lines[line];
    let prefix_end = char.min(line_text.len());
    let prefix = &line_text[..prefix_end].trim_end();

    // Check if this is a type definition
    if spec.types.iter().any(|t| t.name == name) && prefix.ends_with("type") {
        return true;
    }

    // Check if this is a property definition
    let is_property = spec.properties.iter().any(|p| p.name() == name);
    if is_property {
        // Property keywords that precede the name
        let property_keywords = [
            "theorem",
            "temporal",
            "contract",
            "invariant",
            "refinement",
            "probabilistic",
            "security",
        ];
        for kw in property_keywords {
            if prefix.ends_with(kw) {
                return true;
            }
        }
    }

    false
}

/// Build document symbols for a USL document.
///
/// Provides hierarchical symbols with children:
/// - Types include their fields as children
/// - Contracts include requires/ensures/ensures_err clauses as children
/// - Refinements include abstraction/simulation as children
pub fn document_symbols_for_doc(doc: &Document) -> Vec<DocumentSymbol> {
    let mut symbols = Vec::new();
    let spec = match &doc.spec {
        Some(spec) => spec,
        None => return symbols,
    };

    for type_def in &spec.types {
        if let Some(range) = doc.find_identifier_range(&type_def.name) {
            // Build field children
            let children = build_type_field_children(doc, type_def);
            #[allow(deprecated)]
            symbols.push(DocumentSymbol {
                name: type_def.name.clone(),
                detail: Some(format!("type ({} fields)", type_def.fields.len())),
                kind: SymbolKind::STRUCT,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: if children.is_empty() {
                    None
                } else {
                    Some(children)
                },
            });
        }
    }

    for prop in &spec.properties {
        let name = prop.name();
        if let Some(range) = doc.find_identifier_range(&name) {
            // Build children based on property type
            let children = build_property_children(doc, prop);
            #[allow(deprecated)]
            symbols.push(DocumentSymbol {
                name,
                detail: Some(property_kind_name(prop).to_string()),
                kind: symbol_kind_for_property(prop),
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: if children.is_empty() {
                    None
                } else {
                    Some(children)
                },
            });
        }
    }

    symbols
}

/// Build children symbols for type definition fields.
fn build_type_field_children(doc: &Document, type_def: &TypeDef) -> Vec<DocumentSymbol> {
    let mut children = Vec::new();

    for field in &type_def.fields {
        if let Some(range) = doc.find_identifier_range(&field.name) {
            #[allow(deprecated)]
            children.push(DocumentSymbol {
                name: field.name.clone(),
                detail: Some(format_type(&field.ty)),
                kind: SymbolKind::FIELD,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            });
        }
    }

    children
}

/// Build children symbols for property definitions.
fn build_property_children(doc: &Document, prop: &Property) -> Vec<DocumentSymbol> {
    match prop {
        Property::Contract(contract) => build_contract_children(doc, contract),
        Property::Refinement(refinement) => build_refinement_children(doc, refinement),
        _ => Vec::new(),
    }
}

/// Build children for contract clauses (params, requires, ensures, ensures_err).
fn build_contract_children(doc: &Document, contract: &Contract) -> Vec<DocumentSymbol> {
    let mut children = Vec::new();

    // Add parameters as children
    for param in &contract.params {
        if let Some(range) = doc.find_identifier_range(&param.name) {
            #[allow(deprecated)]
            children.push(DocumentSymbol {
                name: param.name.clone(),
                detail: Some(format_type(&param.ty)),
                kind: SymbolKind::VARIABLE,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            });
        }
    }

    // Add requires clauses
    if !contract.requires.is_empty() {
        if let Some(range) = doc.find_identifier_range("requires") {
            #[allow(deprecated)]
            children.push(DocumentSymbol {
                name: "requires".to_string(),
                detail: Some(format!("{} clause(s)", contract.requires.len())),
                kind: SymbolKind::KEY,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            });
        }
    }

    // Add ensures clauses
    if !contract.ensures.is_empty() {
        if let Some(range) = doc.find_identifier_range("ensures") {
            #[allow(deprecated)]
            children.push(DocumentSymbol {
                name: "ensures".to_string(),
                detail: Some(format!("{} clause(s)", contract.ensures.len())),
                kind: SymbolKind::KEY,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            });
        }
    }

    // Add ensures_err clauses
    if !contract.ensures_err.is_empty() {
        if let Some(range) = doc.find_identifier_range("ensures_err") {
            #[allow(deprecated)]
            children.push(DocumentSymbol {
                name: "ensures_err".to_string(),
                detail: Some(format!("{} clause(s)", contract.ensures_err.len())),
                kind: SymbolKind::KEY,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            });
        }
    }

    children
}

/// Build children for refinement clauses (abstraction, simulation).
fn build_refinement_children(doc: &Document, _refinement: &Refinement) -> Vec<DocumentSymbol> {
    let mut children = Vec::new();

    // Add abstraction section
    if let Some(range) = doc.find_identifier_range("abstraction") {
        #[allow(deprecated)]
        children.push(DocumentSymbol {
            name: "abstraction".to_string(),
            detail: Some("abstraction function".to_string()),
            kind: SymbolKind::KEY,
            tags: None,
            deprecated: None,
            range,
            selection_range: range,
            children: None,
        });
    }

    // Add simulation section
    if let Some(range) = doc.find_identifier_range("simulation") {
        #[allow(deprecated)]
        children.push(DocumentSymbol {
            name: "simulation".to_string(),
            detail: Some("simulation relation".to_string()),
            kind: SymbolKind::KEY,
            tags: None,
            deprecated: None,
            range,
            selection_range: range,
            children: None,
        });
    }

    children
}

/// Collect workspace symbols across all open documents matching a query.
pub fn collect_workspace_symbols(store: &DocumentStore, query: &str) -> Vec<SymbolInformation> {
    let normalized_query = query.trim().to_lowercase();
    let mut symbols = Vec::new();

    for uri in store.all_uris() {
        if let Some(mut doc_symbols) = store.with_document(&uri, |doc| {
            workspace_symbols_for_doc(doc, &normalized_query)
        }) {
            symbols.append(&mut doc_symbols);
        }
    }

    symbols
}

/// Build workspace symbol information for a single document.
pub fn workspace_symbols_for_doc(doc: &Document, normalized_query: &str) -> Vec<SymbolInformation> {
    let mut symbols = Vec::new();
    let spec = match &doc.spec {
        Some(spec) => spec,
        None => return symbols,
    };

    let matches_query =
        |name: &str| normalized_query.is_empty() || name.to_lowercase().contains(normalized_query);

    for type_def in &spec.types {
        if !matches_query(&type_def.name) {
            continue;
        }
        if let Some(range) = doc.find_identifier_range(&type_def.name) {
            #[allow(deprecated)]
            symbols.push(SymbolInformation {
                name: type_def.name.clone(),
                kind: SymbolKind::STRUCT,
                tags: None,
                deprecated: None,
                location: Location {
                    uri: doc.uri.clone(),
                    range,
                },
                container_name: None,
            });
        }
    }

    for prop in &spec.properties {
        let name = prop.name();
        if !matches_query(&name) {
            continue;
        }
        if let Some(range) = doc.find_identifier_range(&name) {
            // Build container_name with signature info for contracts
            let container_name = property_signature_info(prop);

            #[allow(deprecated)]
            symbols.push(SymbolInformation {
                name,
                kind: symbol_kind_for_property(prop),
                tags: None,
                deprecated: None,
                location: Location {
                    uri: doc.uri.clone(),
                    range,
                },
                container_name,
            });
        }
    }

    symbols
}

/// Generate signature information for a property (used in workspace symbols).
///
/// Returns a string describing the property's signature (parameters, return type).
pub fn property_signature_info(prop: &Property) -> Option<String> {
    match prop {
        Property::Contract(contract) => {
            let params: Vec<String> = contract
                .params
                .iter()
                .map(|p| format!("{}: {}", p.name, format_type(&p.ty)))
                .collect();
            let return_str = contract
                .return_type
                .as_ref()
                .map(|t| format!(" -> {}", format_type(t)))
                .unwrap_or_default();
            Some(format!("({}){}", params.join(", "), return_str))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::{Contract, Invariant, Theorem};
    use tower_lsp::lsp_types::Url;

    #[test]
    fn test_format_type() {
        assert_eq!(format_type(&Type::Named("Foo".to_string())), "Foo");
        assert_eq!(
            format_type(&Type::Set(Box::new(Type::Named("Int".to_string())))),
            "Set<Int>"
        );
        assert_eq!(
            format_type(&Type::Map(
                Box::new(Type::Named("String".to_string())),
                Box::new(Type::Named("Int".to_string()))
            )),
            "Map<String, Int>"
        );
    }

    #[test]
    fn test_property_kind_name() {
        let theorem = Property::Theorem(Theorem {
            name: "test".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        assert_eq!(property_kind_name(&theorem), "theorem");

        let invariant = Property::Invariant(Invariant {
            name: "test".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        assert_eq!(property_kind_name(&invariant), "invariant");
    }

    #[test]
    fn test_document_symbols_extract_types_and_properties() {
        let doc = Document::new(
            Url::parse("file:///refinement.usl").unwrap(),
            1,
            include_str!("../../../examples/usl/refinement.usl").to_string(),
        );

        let symbols = document_symbols_for_doc(&doc);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();

        // Types
        assert!(names.contains(&"AbstractSet"));
        assert!(names.contains(&"SortedListSet"));
        assert!(names.contains(&"AbstractQueue"));
        assert!(names.contains(&"RingBuffer"));

        // Refinements as document symbols
        let sorted = symbols
            .iter()
            .find(|s| s.name == "sorted_list_refines_set")
            .expect("sorted_list_refines_set symbol missing");
        assert_eq!(sorted.kind, SymbolKind::FUNCTION);
        assert_eq!(sorted.detail.as_deref(), Some("refinement"));
        assert_eq!(sorted.range.start.line, 14);
        assert_eq!(sorted.range.start.character, 11);

        // Total symbols should match types + refinements
        assert_eq!(symbols.len(), 6);
    }

    #[test]
    fn test_document_symbols_type_field_children() {
        let doc = Document::new(
            Url::parse("file:///types.usl").unwrap(),
            1,
            "type User = { id: Int, name: String, active: Bool }".to_string(),
        );

        let symbols = document_symbols_for_doc(&doc);
        assert_eq!(symbols.len(), 1);

        let user_sym = &symbols[0];
        assert_eq!(user_sym.name, "User");
        assert_eq!(user_sym.kind, SymbolKind::STRUCT);
        assert_eq!(user_sym.detail.as_deref(), Some("type (3 fields)"));

        // Check children (fields)
        let children = user_sym
            .children
            .as_ref()
            .expect("User should have children");
        assert_eq!(children.len(), 3);

        let field_names: Vec<_> = children.iter().map(|c| c.name.as_str()).collect();
        assert!(field_names.contains(&"id"));
        assert!(field_names.contains(&"name"));
        assert!(field_names.contains(&"active"));

        // Check field details (types)
        let id_field = children.iter().find(|c| c.name == "id").unwrap();
        assert_eq!(id_field.kind, SymbolKind::FIELD);
        assert_eq!(id_field.detail.as_deref(), Some("Int"));
    }

    #[test]
    fn test_document_symbols_contract_children() {
        let doc = Document::new(
            Url::parse("file:///contracts.usl").unwrap(),
            1,
            include_str!("../../../examples/usl/contracts.usl").to_string(),
        );

        let symbols = document_symbols_for_doc(&doc);

        // Find the Stack::push contract
        let push_sym = symbols
            .iter()
            .find(|s| s.name == "Stack::push")
            .expect("Stack::push symbol missing");
        assert_eq!(push_sym.kind, SymbolKind::METHOD);
        assert_eq!(push_sym.detail.as_deref(), Some("contract"));

        // Check children (params and clauses)
        let children = push_sym
            .children
            .as_ref()
            .expect("contract should have children");

        let child_names: Vec<_> = children.iter().map(|c| c.name.as_str()).collect();

        // Parameters should appear as children
        assert!(child_names.contains(&"self"));
        assert!(child_names.contains(&"value"));

        // Clauses should appear as children
        assert!(child_names.contains(&"requires"));
        assert!(child_names.contains(&"ensures"));
        assert!(child_names.contains(&"ensures_err"));

        // Check parameter details
        let self_param = children.iter().find(|c| c.name == "self").unwrap();
        assert_eq!(self_param.kind, SymbolKind::VARIABLE);
        assert_eq!(self_param.detail.as_deref(), Some("Stack"));

        let value_param = children.iter().find(|c| c.name == "value").unwrap();
        assert_eq!(value_param.kind, SymbolKind::VARIABLE);
        assert_eq!(value_param.detail.as_deref(), Some("Int"));

        // Check clause details
        let requires = children.iter().find(|c| c.name == "requires").unwrap();
        assert_eq!(requires.kind, SymbolKind::KEY);
        assert_eq!(requires.detail.as_deref(), Some("1 clause(s)"));
    }

    #[test]
    fn test_document_symbols_refinement_children() {
        let doc = Document::new(
            Url::parse("file:///refinement.usl").unwrap(),
            1,
            include_str!("../../../examples/usl/refinement.usl").to_string(),
        );

        let symbols = document_symbols_for_doc(&doc);

        // Find the sorted_list_refines_set refinement
        let refine_sym = symbols
            .iter()
            .find(|s| s.name == "sorted_list_refines_set")
            .expect("sorted_list_refines_set symbol missing");

        // Check children (abstraction and simulation)
        let children = refine_sym
            .children
            .as_ref()
            .expect("refinement should have children");
        assert_eq!(children.len(), 2);

        let child_names: Vec<_> = children.iter().map(|c| c.name.as_str()).collect();
        assert!(child_names.contains(&"abstraction"));
        assert!(child_names.contains(&"simulation"));

        // Check clause details
        let abstraction = children.iter().find(|c| c.name == "abstraction").unwrap();
        assert_eq!(abstraction.kind, SymbolKind::KEY);
        assert_eq!(abstraction.detail.as_deref(), Some("abstraction function"));
    }

    #[test]
    fn test_document_symbols_contract_param_children() {
        // Contract parameters should appear as children with their types
        let doc = Document::new(
            Url::parse("file:///params.usl").unwrap(),
            1,
            r#"
contract divide(x: Int, y: Int) -> Result<Int> {
    requires { y != 0 }
    ensures { result * y == x }
}
"#
            .to_string(),
        );

        let symbols = document_symbols_for_doc(&doc);
        let divide_sym = symbols
            .iter()
            .find(|s| s.name == "divide")
            .expect("divide symbol missing");

        let children = divide_sym
            .children
            .as_ref()
            .expect("contract should have children");

        // Parameters should be first
        let x_param = children.iter().find(|c| c.name == "x").unwrap();
        assert_eq!(x_param.kind, SymbolKind::VARIABLE);
        assert_eq!(x_param.detail.as_deref(), Some("Int"));

        let y_param = children.iter().find(|c| c.name == "y").unwrap();
        assert_eq!(y_param.kind, SymbolKind::VARIABLE);
        assert_eq!(y_param.detail.as_deref(), Some("Int"));

        // Clauses should also be present
        let child_names: Vec<_> = children.iter().map(|c| c.name.as_str()).collect();
        assert!(child_names.contains(&"requires"));
        assert!(child_names.contains(&"ensures"));
    }

    #[test]
    fn test_document_symbols_empty_children() {
        // Theorems and invariants should not have children
        let doc = Document::new(
            Url::parse("file:///basic.usl").unwrap(),
            1,
            "theorem simple_theorem { true }".to_string(),
        );

        let symbols = document_symbols_for_doc(&doc);
        assert_eq!(symbols.len(), 1);

        let theorem_sym = &symbols[0];
        assert_eq!(theorem_sym.name, "simple_theorem");
        assert!(theorem_sym.children.is_none());
    }

    #[test]
    fn test_workspace_symbols_across_documents() {
        let store = DocumentStore::new();
        let uri1 = Url::parse("file:///user.usl").unwrap();
        let uri2 = Url::parse("file:///graph.usl").unwrap();

        store.open(
            uri1.clone(),
            1,
            "type User = { id: Int }\ntheorem user_valid { true }".to_string(),
        );

        store.open(
            uri2.clone(),
            1,
            "type Graph = { nodes: List<Int> }\ncontract Graph::add_node(self: Graph, id: Int) -> Bool {\n  requires { true }\n  ensures { true }\n}"
                .to_string(),
        );

        let user_symbols = collect_workspace_symbols(&store, "user");
        let user_names: Vec<_> = user_symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(user_names.contains(&"User"));
        assert!(user_names.contains(&"user_valid"));
        assert!(user_symbols.iter().all(|s| s.location.uri == uri1));

        let graph_symbols = collect_workspace_symbols(&store, "add_node");
        assert_eq!(graph_symbols.len(), 1);
        let sym = &graph_symbols[0];
        assert_eq!(sym.name, "Graph::add_node");
        assert_eq!(sym.kind, SymbolKind::METHOD);
        assert_eq!(sym.location.uri, uri2);

        let all_symbols = collect_workspace_symbols(&store, "");
        assert_eq!(all_symbols.len(), 4);
    }

    #[test]
    fn test_workspace_symbols_contract_signature() {
        // Contract symbols should have parameter signature in container_name
        let store = DocumentStore::new();
        let uri = Url::parse("file:///contracts.usl").unwrap();

        store.open(
            uri.clone(),
            1,
            r#"
contract Stack::push(self: Stack, item: Int) -> Result<Stack> {
    requires { true }
    ensures { true }
}
"#
            .to_string(),
        );

        let symbols = collect_workspace_symbols(&store, "push");
        assert_eq!(symbols.len(), 1);

        let sym = &symbols[0];
        assert_eq!(sym.name, "Stack::push");
        assert_eq!(sym.kind, SymbolKind::METHOD);

        // container_name should contain the signature
        let sig = sym
            .container_name
            .as_ref()
            .expect("contract should have signature");
        assert!(sig.contains("self: Stack"));
        assert!(sig.contains("item: Int"));
        assert!(sig.contains("-> Result<Stack>"));
    }

    #[test]
    fn test_property_signature_info_contract() {
        // Test the helper function directly
        let contract = Contract {
            type_path: vec!["Foo".to_string(), "bar".to_string()],
            params: vec![
                dashprove_usl::Param {
                    name: "x".to_string(),
                    ty: dashprove_usl::Type::Named("Int".to_string()),
                },
                dashprove_usl::Param {
                    name: "y".to_string(),
                    ty: dashprove_usl::Type::Named("Bool".to_string()),
                },
            ],
            return_type: Some(dashprove_usl::Type::Named("Unit".to_string())),
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        };

        let sig =
            property_signature_info(&Property::Contract(contract)).expect("should have signature");
        assert_eq!(sig, "(x: Int, y: Bool) -> Unit");
    }

    #[test]
    fn test_property_signature_info_theorem() {
        // Non-contract properties should not have signatures
        let theorem = Theorem {
            name: "test".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        };

        let sig = property_signature_info(&Property::Theorem(theorem));
        assert!(sig.is_none(), "theorems should not have signatures");
    }

    #[test]
    fn test_is_definition_site_type() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Value = { id: Int }\ntheorem value_check { forall v: Value . true }".to_string(),
        );

        let spec = doc.spec.as_ref().expect("should parse");

        // The type definition "type Value = " should be a definition site
        let def_range = Range {
            start: Position::new(0, 5),
            end: Position::new(0, 10),
        };
        assert!(is_definition_site(spec, "Value", &def_range, &doc.text));

        // Usage "forall v: Value" should NOT be a definition site
        let usage_range = Range {
            start: Position::new(1, 32),
            end: Position::new(1, 37),
        };
        assert!(!is_definition_site(spec, "Value", &usage_range, &doc.text));
    }

    #[test]
    fn test_is_definition_site_property() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem my_theorem { true }\ninvariant my_invariant { true }".to_string(),
        );

        let spec = doc.spec.as_ref().expect("should parse");

        // "theorem my_theorem" should be a definition site
        let theorem_range = Range {
            start: Position::new(0, 8),
            end: Position::new(0, 18),
        };
        assert!(is_definition_site(
            spec,
            "my_theorem",
            &theorem_range,
            &doc.text
        ));

        // "invariant my_invariant" should be a definition site
        let invariant_range = Range {
            start: Position::new(1, 10),
            end: Position::new(1, 22),
        };
        assert!(is_definition_site(
            spec,
            "my_invariant",
            &invariant_range,
            &doc.text
        ));
    }

    // ========== MUTATION-KILLING TESTS ==========

    // Test type_or_property_info property name matching
    #[test]
    fn test_type_or_property_info_property_name_match() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type MyType = { field: Int }\ntheorem my_theorem { true }".to_string(),
        );

        let spec = doc.spec.as_ref().expect("should parse");

        // Property name should match exactly
        let result = type_or_property_info(spec, "my_theorem");
        assert!(result.is_some(), "Should find property by name");
        let info = result.unwrap();
        assert!(
            info.contains("my_theorem"),
            "Result should contain the property name"
        );

        // Non-matching name should return None
        let no_match = type_or_property_info(spec, "other_theorem");
        assert!(no_match.is_none(), "Non-matching name should return None");
    }

    #[test]
    fn test_type_or_property_info_type_name_match() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Counter = { value: Int, max: Int }".to_string(),
        );

        let spec = doc.spec.as_ref().expect("should parse");

        // Type name should match exactly
        let result = type_or_property_info(spec, "Counter");
        assert!(result.is_some(), "Should find type by name");
        let info = result.unwrap();
        assert!(
            info.contains("Counter"),
            "Result should contain the type name"
        );

        // Non-matching name should return None
        let no_match = type_or_property_info(spec, "NonexistentType");
        assert!(
            no_match.is_none(),
            "Non-matching type name should return None"
        );
    }

    // Test format_type_def returns non-empty meaningful output
    #[test]
    fn test_format_type_def_content() {
        let type_def = dashprove_usl::TypeDef {
            name: "User".to_string(),
            fields: vec![
                dashprove_usl::Field {
                    name: "id".to_string(),
                    ty: Type::Named("Int".to_string()),
                },
                dashprove_usl::Field {
                    name: "name".to_string(),
                    ty: Type::Named("String".to_string()),
                },
            ],
        };

        let formatted = format_type_def(&type_def);

        // Should contain type name
        assert!(
            formatted.contains("User"),
            "format_type_def should include type name"
        );
        // Should contain field names
        assert!(
            formatted.contains("id"),
            "format_type_def should include field names"
        );
        assert!(
            formatted.contains("name"),
            "format_type_def should include field names"
        );
        // Should contain field types
        assert!(
            formatted.contains("Int"),
            "format_type_def should include field types"
        );
        assert!(
            formatted.contains("String"),
            "format_type_def should include field types"
        );
        // Should not be empty or just "xyzzy"
        assert!(
            formatted.len() > 20,
            "format_type_def should return substantial content"
        );
        assert!(
            !formatted.contains("xyzzy"),
            "format_type_def should not return placeholder"
        );
    }

    #[test]
    fn test_format_type_def_structure() {
        let type_def = dashprove_usl::TypeDef {
            name: "Point".to_string(),
            fields: vec![
                dashprove_usl::Field {
                    name: "x".to_string(),
                    ty: Type::Named("Int".to_string()),
                },
                dashprove_usl::Field {
                    name: "y".to_string(),
                    ty: Type::Named("Int".to_string()),
                },
            ],
        };

        let formatted = format_type_def(&type_def);

        // Should have markdown structure
        assert!(formatted.contains("**type"), "Should have bold type marker");
        assert!(formatted.contains("```usl"), "Should have code block");
        assert!(formatted.contains("```"), "Should close code block");
    }

    // Test format_expr returns meaningful expression representation
    #[test]
    fn test_format_expr_var() {
        let expr = dashprove_usl::Expr::Var("myVariable".to_string());
        let formatted = format_expr(&expr);
        assert_eq!(formatted, "myVariable");
    }

    #[test]
    fn test_format_expr_int() {
        let expr = dashprove_usl::Expr::Int(42);
        let formatted = format_expr(&expr);
        assert_eq!(formatted, "42");
    }

    #[test]
    fn test_format_expr_float() {
        let expr = dashprove_usl::Expr::Float(1.5);
        let formatted = format_expr(&expr);
        assert!(formatted.contains("1.5"));
    }

    #[test]
    fn test_format_expr_string() {
        let expr = dashprove_usl::Expr::String("hello".to_string());
        let formatted = format_expr(&expr);
        assert!(formatted.contains("hello"));
        assert!(formatted.contains("\""), "String should have quotes");
    }

    #[test]
    fn test_format_expr_bool() {
        let true_expr = dashprove_usl::Expr::Bool(true);
        let false_expr = dashprove_usl::Expr::Bool(false);
        assert_eq!(format_expr(&true_expr), "true");
        assert_eq!(format_expr(&false_expr), "false");
    }

    #[test]
    fn test_format_expr_binary_ops() {
        // Test each binary operator
        let add = dashprove_usl::Expr::Binary(
            Box::new(dashprove_usl::Expr::Int(1)),
            BinaryOp::Add,
            Box::new(dashprove_usl::Expr::Int(2)),
        );
        let formatted = format_expr(&add);
        assert!(formatted.contains("+"), "Add should use +");
        assert!(formatted.contains("1"), "Should have left operand");
        assert!(formatted.contains("2"), "Should have right operand");

        let sub = dashprove_usl::Expr::Binary(
            Box::new(dashprove_usl::Expr::Int(5)),
            BinaryOp::Sub,
            Box::new(dashprove_usl::Expr::Int(3)),
        );
        assert!(format_expr(&sub).contains("-"), "Sub should use -");

        let mul = dashprove_usl::Expr::Binary(
            Box::new(dashprove_usl::Expr::Int(2)),
            BinaryOp::Mul,
            Box::new(dashprove_usl::Expr::Int(3)),
        );
        assert!(format_expr(&mul).contains("*"), "Mul should use *");

        let div = dashprove_usl::Expr::Binary(
            Box::new(dashprove_usl::Expr::Int(10)),
            BinaryOp::Div,
            Box::new(dashprove_usl::Expr::Int(2)),
        );
        assert!(format_expr(&div).contains("/"), "Div should use /");

        let mod_op = dashprove_usl::Expr::Binary(
            Box::new(dashprove_usl::Expr::Int(10)),
            BinaryOp::Mod,
            Box::new(dashprove_usl::Expr::Int(3)),
        );
        assert!(format_expr(&mod_op).contains("%"), "Mod should use %");
    }

    #[test]
    fn test_format_expr_comparison_ops() {
        let eq = dashprove_usl::Expr::Compare(
            Box::new(dashprove_usl::Expr::Var("x".to_string())),
            ComparisonOp::Eq,
            Box::new(dashprove_usl::Expr::Int(0)),
        );
        assert!(format_expr(&eq).contains("=="), "Eq should use ==");

        let ne = dashprove_usl::Expr::Compare(
            Box::new(dashprove_usl::Expr::Var("x".to_string())),
            ComparisonOp::Ne,
            Box::new(dashprove_usl::Expr::Int(0)),
        );
        assert!(format_expr(&ne).contains("!="), "Ne should use !=");

        let lt = dashprove_usl::Expr::Compare(
            Box::new(dashprove_usl::Expr::Var("x".to_string())),
            ComparisonOp::Lt,
            Box::new(dashprove_usl::Expr::Int(0)),
        );
        let lt_str = format_expr(&lt);
        assert!(
            lt_str.contains("<") && !lt_str.contains("<="),
            "Lt should use <"
        );

        let le = dashprove_usl::Expr::Compare(
            Box::new(dashprove_usl::Expr::Var("x".to_string())),
            ComparisonOp::Le,
            Box::new(dashprove_usl::Expr::Int(0)),
        );
        assert!(format_expr(&le).contains("<="), "Le should use <=");

        let gt = dashprove_usl::Expr::Compare(
            Box::new(dashprove_usl::Expr::Var("x".to_string())),
            ComparisonOp::Gt,
            Box::new(dashprove_usl::Expr::Int(0)),
        );
        let gt_str = format_expr(&gt);
        assert!(
            gt_str.contains(">") && !gt_str.contains(">="),
            "Gt should use >"
        );

        let ge = dashprove_usl::Expr::Compare(
            Box::new(dashprove_usl::Expr::Var("x".to_string())),
            ComparisonOp::Ge,
            Box::new(dashprove_usl::Expr::Int(0)),
        );
        assert!(format_expr(&ge).contains(">="), "Ge should use >=");
    }

    #[test]
    fn test_format_expr_logical_ops() {
        let not_expr = dashprove_usl::Expr::Not(Box::new(dashprove_usl::Expr::Bool(true)));
        assert!(
            format_expr(&not_expr).contains("not"),
            "Not should use 'not'"
        );

        let and_expr = dashprove_usl::Expr::And(
            Box::new(dashprove_usl::Expr::Bool(true)),
            Box::new(dashprove_usl::Expr::Bool(false)),
        );
        assert!(
            format_expr(&and_expr).contains("and"),
            "And should use 'and'"
        );

        let or_expr = dashprove_usl::Expr::Or(
            Box::new(dashprove_usl::Expr::Bool(true)),
            Box::new(dashprove_usl::Expr::Bool(false)),
        );
        assert!(format_expr(&or_expr).contains("or"), "Or should use 'or'");

        let implies = dashprove_usl::Expr::Implies(
            Box::new(dashprove_usl::Expr::Bool(true)),
            Box::new(dashprove_usl::Expr::Bool(false)),
        );
        assert!(
            format_expr(&implies).contains("implies"),
            "Implies should use 'implies'"
        );
    }

    #[test]
    fn test_format_expr_quantifiers() {
        let forall = dashprove_usl::Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("Int".to_string())),
            body: Box::new(dashprove_usl::Expr::Bool(true)),
        };
        let forall_str = format_expr(&forall);
        assert!(
            forall_str.contains("forall"),
            "ForAll should contain 'forall'"
        );
        assert!(forall_str.contains("x"), "Should contain variable name");
        assert!(forall_str.contains("Int"), "Should contain type");

        let exists = dashprove_usl::Expr::Exists {
            var: "y".to_string(),
            ty: None,
            body: Box::new(dashprove_usl::Expr::Bool(true)),
        };
        let exists_str = format_expr(&exists);
        assert!(
            exists_str.contains("exists"),
            "Exists should contain 'exists'"
        );
        assert!(exists_str.contains("y"), "Should contain variable name");
    }

    #[test]
    fn test_format_expr_function_app() {
        let app = dashprove_usl::Expr::App(
            "myFunc".to_string(),
            vec![dashprove_usl::Expr::Int(1), dashprove_usl::Expr::Int(2)],
        );
        let formatted = format_expr(&app);
        assert!(formatted.contains("myFunc"), "Should contain function name");
        assert!(formatted.contains("1"), "Should contain first arg");
        assert!(formatted.contains("2"), "Should contain second arg");
        assert!(formatted.contains(","), "Should have comma separator");
    }

    #[test]
    fn test_format_expr_method_call() {
        let method = dashprove_usl::Expr::MethodCall {
            receiver: Box::new(dashprove_usl::Expr::Var("obj".to_string())),
            method: "doSomething".to_string(),
            args: vec![dashprove_usl::Expr::Int(42)],
        };
        let formatted = format_expr(&method);
        assert!(formatted.contains("obj"), "Should contain receiver");
        assert!(
            formatted.contains("doSomething"),
            "Should contain method name"
        );
        assert!(formatted.contains("42"), "Should contain argument");
    }

    #[test]
    fn test_format_expr_field_access() {
        let field = dashprove_usl::Expr::FieldAccess(
            Box::new(dashprove_usl::Expr::Var("struct".to_string())),
            "field".to_string(),
        );
        let formatted = format_expr(&field);
        assert!(formatted.contains("struct"), "Should contain base");
        assert!(formatted.contains("field"), "Should contain field name");
        assert!(formatted.contains("."), "Should have dot separator");
    }

    // Test format_property returns meaningful output for each property type
    #[test]
    fn test_format_property_theorem() {
        let theorem = Property::Theorem(Theorem {
            name: "my_theorem".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let formatted = format_property(&theorem);
        assert!(
            formatted.contains("my_theorem"),
            "Should contain theorem name"
        );
        assert!(
            formatted.contains("theorem"),
            "Should indicate it's a theorem"
        );
        assert!(!formatted.is_empty(), "Should not be empty");
        assert!(
            !formatted.contains("xyzzy"),
            "Should not return placeholder"
        );
    }

    #[test]
    fn test_format_property_temporal() {
        let temporal = Property::Temporal(dashprove_usl::Temporal {
            name: "liveness".to_string(),
            body: dashprove_usl::TemporalExpr::Always(Box::new(
                dashprove_usl::TemporalExpr::Eventually(Box::new(
                    dashprove_usl::TemporalExpr::Atom(dashprove_usl::Expr::Bool(true)),
                )),
            )),
            fairness: vec![],
        });
        let formatted = format_property(&temporal);
        assert!(formatted.contains("liveness"), "Should contain name");
        assert!(formatted.contains("temporal"), "Should indicate type");
    }

    #[test]
    fn test_format_property_contract() {
        let contract = Property::Contract(Contract {
            type_path: vec!["Type".to_string(), "method".to_string()],
            params: vec![dashprove_usl::Param {
                name: "self".to_string(),
                ty: Type::Named("Type".to_string()),
            }],
            return_type: Some(Type::Named("Bool".to_string())),
            requires: vec![dashprove_usl::Expr::Bool(true)],
            ensures: vec![
                dashprove_usl::Expr::Bool(true),
                dashprove_usl::Expr::Bool(true),
            ],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        });
        let formatted = format_property(&contract);
        assert!(
            formatted.contains("Type::method"),
            "Should contain full path"
        );
        assert!(formatted.contains("contract"), "Should indicate type");
        assert!(
            formatted.contains("1 preconditions"),
            "Should show precondition count"
        );
        assert!(
            formatted.contains("2 postconditions"),
            "Should show postcondition count"
        );
    }

    #[test]
    fn test_format_property_invariant() {
        let invariant = Property::Invariant(Invariant {
            name: "always_positive".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let formatted = format_property(&invariant);
        assert!(formatted.contains("always_positive"), "Should contain name");
        assert!(formatted.contains("invariant"), "Should indicate type");
    }

    #[test]
    fn test_format_property_refinement() {
        let refinement = Property::Refinement(dashprove_usl::Refinement {
            name: "impl_refines_spec".to_string(),
            refines: "AbstractSpec".to_string(),
            mappings: vec![],
            invariants: vec![],
            abstraction: dashprove_usl::Expr::Bool(true),
            simulation: dashprove_usl::Expr::Bool(true),
            actions: vec![],
        });
        let formatted = format_property(&refinement);
        assert!(
            formatted.contains("impl_refines_spec"),
            "Should contain name"
        );
        assert!(
            formatted.contains("refines"),
            "Should show refinement relationship"
        );
        assert!(
            formatted.contains("AbstractSpec"),
            "Should show what it refines"
        );
    }

    #[test]
    fn test_format_property_probabilistic() {
        let prob = Property::Probabilistic(dashprove_usl::Probabilistic {
            name: "high_probability".to_string(),
            comparison: ComparisonOp::Ge,
            bound: 0.95,
            condition: dashprove_usl::Expr::Bool(true),
        });
        let formatted = format_property(&prob);
        assert!(
            formatted.contains("high_probability"),
            "Should contain name"
        );
        assert!(formatted.contains("probabilistic"), "Should indicate type");
        assert!(formatted.contains("0.95"), "Should show bound");
    }

    #[test]
    fn test_format_property_security() {
        let security = Property::Security(dashprove_usl::Security {
            name: "no_leaks".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let formatted = format_property(&security);
        assert!(formatted.contains("no_leaks"), "Should contain name");
        assert!(formatted.contains("security"), "Should indicate type");
    }

    #[test]
    fn test_format_property_semantic() {
        let semantic = Property::Semantic(dashprove_usl::SemanticProperty {
            name: "semantic_check".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let formatted = format_property(&semantic);
        assert!(formatted.contains("semantic_check"), "Should contain name");
        assert!(formatted.contains("semantic"), "Should indicate type");
    }

    #[test]
    fn test_format_property_platform_api() {
        let api = Property::PlatformApi(dashprove_usl::PlatformApi {
            name: "metal_api".to_string(),
            states: vec![],
        });
        let formatted = format_property(&api);
        assert!(formatted.contains("metal_api"), "Should contain name");
        assert!(formatted.contains("platform_api"), "Should indicate type");
    }

    #[test]
    fn test_format_property_bisimulation() {
        let bisim = Property::Bisimulation(dashprove_usl::Bisimulation {
            name: "equiv_check".to_string(),
            oracle: "reference".to_string(),
            subject: "implementation".to_string(),
            equivalent_on: vec![],
            tolerance: None,
            property: None,
        });
        let formatted = format_property(&bisim);
        assert!(formatted.contains("equiv_check"), "Should contain name");
        assert!(formatted.contains("bisimulation"), "Should indicate type");
    }

    // Test plural helper
    #[test]
    fn test_plural_zero() {
        assert_eq!(plural(0), "s");
    }

    #[test]
    fn test_plural_one() {
        assert_eq!(plural(1), "");
    }

    #[test]
    fn test_plural_many() {
        assert_eq!(plural(2), "s");
        assert_eq!(plural(10), "s");
        assert_eq!(plural(100), "s");
    }
}
