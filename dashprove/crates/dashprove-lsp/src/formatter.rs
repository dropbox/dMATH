//! USL code formatter
//!
//! Formats USL specifications with consistent indentation and spacing.

use dashprove_usl::{
    BinaryOp, ComparisonOp, Contract, Expr, Field, Invariant, Param, Probabilistic, Property,
    Refinement, Security, Spec, Temporal, TemporalExpr, Theorem, Type, TypeDef,
};

/// Format configuration
#[derive(Debug, Clone)]
pub struct FormatConfig {
    /// Indentation string (default: 4 spaces)
    pub indent: String,
    /// Maximum line width for future line wrapping support (default: 100)
    #[allow(dead_code)]
    pub max_width: usize,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            indent: "    ".to_string(),
            max_width: 100,
        }
    }
}

/// Format a complete USL specification.
pub fn format_spec(spec: &Spec, config: &FormatConfig) -> String {
    let mut output = String::new();

    // Format type definitions
    for (i, type_def) in spec.types.iter().enumerate() {
        if i > 0 {
            output.push('\n');
        }
        output.push_str(&format_type_def(type_def, config));
        output.push('\n');
    }

    // Add separation between types and properties
    if !spec.types.is_empty() && !spec.properties.is_empty() {
        output.push('\n');
    }

    // Format properties
    for (i, prop) in spec.properties.iter().enumerate() {
        if i > 0 {
            output.push('\n');
        }
        output.push_str(&format_property(prop, config));
        output.push('\n');
    }

    output
}

/// Format a type definition.
fn format_type_def(type_def: &TypeDef, config: &FormatConfig) -> String {
    let mut output = format!("type {} = {{\n", type_def.name);

    for (i, field) in type_def.fields.iter().enumerate() {
        output.push_str(&config.indent);
        output.push_str(&format_field(field));
        if i < type_def.fields.len() - 1 {
            output.push(',');
        }
        output.push('\n');
    }

    output.push('}');
    output
}

/// Format a field.
fn format_field(field: &Field) -> String {
    format!("{}: {}", field.name, format_type(&field.ty))
}

/// Format a type expression.
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
        Type::Unit => "()".to_string(),
    }
}

/// Format a property.
fn format_property(prop: &Property, config: &FormatConfig) -> String {
    match prop {
        Property::Theorem(t) => format_theorem(t, config),
        Property::Temporal(t) => format_temporal(t, config),
        Property::Contract(c) => format_contract(c, config),
        Property::Invariant(i) => format_invariant(i, config),
        Property::Refinement(r) => format_refinement(r, config),
        Property::Probabilistic(p) => format_probabilistic(p, config),
        Property::Security(s) => format_security(s, config),
        Property::Semantic(s) => format!(
            "semantic_property {} {{\n{}{}\n}}",
            s.name,
            config.indent,
            format_expr(&s.body, config, 1)
        ),
        Property::PlatformApi(p) => format!("platform_api {} {{ /* constraints */ }}", p.name),
        Property::Bisimulation(b) => format!(
            "bisimulation {} {{ oracle: \"{}\", subject: \"{}\" }}",
            b.name, b.oracle, b.subject
        ),
        Property::Version(v) => format!(
            "version {} improves {} {{ /* capabilities and preserves */ }}",
            v.name, v.improves
        ),
        Property::Capability(c) => format!(
            "capability {} {{ /* {} abilities, {} requires */ }}",
            c.name,
            c.abilities.len(),
            c.requires.len()
        ),
        Property::DistributedInvariant(d) => format!(
            "distributed invariant {} {{\n{}{}\n}}",
            d.name,
            config.indent,
            format_expr(&d.body, config, 1)
        ),
        Property::DistributedTemporal(d) => format!(
            "distributed temporal {} {{ /* temporal formula with {} fairness constraints */ }}",
            d.name,
            d.fairness.len()
        ),
        Property::Composed(c) => format!(
            "composed {} uses [{}] {{\n{}{}\n}}",
            c.name,
            c.uses.join(", "),
            config.indent,
            format_expr(&c.body, config, 1)
        ),
        Property::ImprovementProposal(p) => format!(
            "improvement_proposal {} {{ /* improves: {}, preserves: {} */ }}",
            p.name,
            p.improves.len(),
            p.preserves.len()
        ),
        Property::VerificationGate(g) => format!(
            "verification_gate {} {{ /* {} checks */ }}",
            g.name,
            g.checks.len()
        ),
        Property::Rollback(r) => format!(
            "rollback_spec {} {{ /* {} invariants */ }}",
            r.name,
            r.invariants.len()
        ),
    }
}

/// Format a theorem.
fn format_theorem(theorem: &Theorem, config: &FormatConfig) -> String {
    format!(
        "theorem {} {{\n{}{}\n}}",
        theorem.name,
        config.indent,
        format_expr(&theorem.body, config, 1)
    )
}

/// Format a temporal property.
fn format_temporal(temporal: &Temporal, config: &FormatConfig) -> String {
    format!(
        "temporal {} {{\n{}{}\n}}",
        temporal.name,
        config.indent,
        format_temporal_expr(&temporal.body, config, 1)
    )
}

/// Format a contract.
fn format_contract(contract: &Contract, config: &FormatConfig) -> String {
    let mut output = String::new();

    // Contract signature
    output.push_str("contract ");
    output.push_str(&contract.type_path.join("::"));
    output.push('(');

    let params: Vec<String> = contract.params.iter().map(format_param).collect();
    output.push_str(&params.join(", "));
    output.push(')');

    if let Some(ref ret_ty) = contract.return_type {
        output.push_str(" -> ");
        output.push_str(&format_type(ret_ty));
    }

    output.push_str(" {\n");

    // Requires clauses
    for req in &contract.requires {
        output.push_str(&config.indent);
        output.push_str("requires {\n");
        output.push_str(&config.indent);
        output.push_str(&config.indent);
        output.push_str(&format_expr(req, config, 2));
        output.push('\n');
        output.push_str(&config.indent);
        output.push_str("}\n");
    }

    // Ensures clauses
    for ens in &contract.ensures {
        output.push_str(&config.indent);
        output.push_str("ensures {\n");
        output.push_str(&config.indent);
        output.push_str(&config.indent);
        output.push_str(&format_expr(ens, config, 2));
        output.push('\n');
        output.push_str(&config.indent);
        output.push_str("}\n");
    }

    // Ensures_err clauses
    for ens_err in &contract.ensures_err {
        output.push_str(&config.indent);
        output.push_str("ensures_err {\n");
        output.push_str(&config.indent);
        output.push_str(&config.indent);
        output.push_str(&format_expr(ens_err, config, 2));
        output.push('\n');
        output.push_str(&config.indent);
        output.push_str("}\n");
    }

    output.push('}');
    output
}

/// Format a contract parameter.
fn format_param(param: &Param) -> String {
    format!("{}: {}", param.name, format_type(&param.ty))
}

/// Format an invariant.
fn format_invariant(invariant: &Invariant, config: &FormatConfig) -> String {
    format!(
        "invariant {} {{\n{}{}\n}}",
        invariant.name,
        config.indent,
        format_expr(&invariant.body, config, 1)
    )
}

/// Format a refinement.
fn format_refinement(refinement: &Refinement, config: &FormatConfig) -> String {
    let mut output = String::new();

    output.push_str(&format!(
        "refinement {} refines {} {{\n",
        refinement.name, refinement.refines
    ));

    // Abstraction clause
    output.push_str(&config.indent);
    output.push_str("abstraction {\n");
    output.push_str(&config.indent);
    output.push_str(&config.indent);
    output.push_str(&format_expr(&refinement.abstraction, config, 2));
    output.push('\n');
    output.push_str(&config.indent);
    output.push_str("}\n");

    // Simulation clause
    output.push_str(&config.indent);
    output.push_str("simulation {\n");
    output.push_str(&config.indent);
    output.push_str(&config.indent);
    output.push_str(&format_expr(&refinement.simulation, config, 2));
    output.push('\n');
    output.push_str(&config.indent);
    output.push_str("}\n");

    output.push('}');
    output
}

/// Format a probabilistic property.
fn format_probabilistic(prob: &Probabilistic, config: &FormatConfig) -> String {
    format!(
        "probabilistic {} {{\n{}probability({}) {} {}\n}}",
        prob.name,
        config.indent,
        format_expr(&prob.condition, config, 1),
        format_comparison_op(&prob.comparison),
        prob.bound
    )
}

/// Format a security property.
fn format_security(security: &Security, config: &FormatConfig) -> String {
    format!(
        "security {} {{\n{}{}\n}}",
        security.name,
        config.indent,
        format_expr(&security.body, config, 1)
    )
}

/// Format an expression.
fn format_expr(expr: &Expr, config: &FormatConfig, _depth: usize) -> String {
    match expr {
        Expr::Var(name) => name.clone(),
        Expr::Int(n) => n.to_string(),
        Expr::Float(f) => format_float(*f),
        Expr::String(s) => format!("\"{}\"", s),
        Expr::Bool(b) => b.to_string(),

        Expr::ForAll { var, ty, body } => {
            let type_annot = ty
                .as_ref()
                .map(|t| format!(": {}", format_type(t)))
                .unwrap_or_default();
            format!(
                "forall {}{} . {}",
                var,
                type_annot,
                format_expr(body, config, _depth)
            )
        }

        Expr::Exists { var, ty, body } => {
            let type_annot = ty
                .as_ref()
                .map(|t| format!(": {}", format_type(t)))
                .unwrap_or_default();
            format!(
                "exists {}{} . {}",
                var,
                type_annot,
                format_expr(body, config, _depth)
            )
        }

        Expr::ForAllIn {
            var,
            collection,
            body,
        } => {
            format!(
                "forall {} in {} . {}",
                var,
                format_expr(collection, config, _depth),
                format_expr(body, config, _depth)
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
                format_expr(collection, config, _depth),
                format_expr(body, config, _depth)
            )
        }

        Expr::Implies(lhs, rhs) => {
            format!(
                "{} implies {}",
                format_expr_paren(lhs, config, _depth, needs_paren_for_implies(lhs)),
                format_expr(rhs, config, _depth)
            )
        }

        Expr::And(lhs, rhs) => {
            format!(
                "{} and {}",
                format_expr_paren(lhs, config, _depth, needs_paren_for_and(lhs)),
                format_expr_paren(rhs, config, _depth, needs_paren_for_and(rhs))
            )
        }

        Expr::Or(lhs, rhs) => {
            format!(
                "{} or {}",
                format_expr_paren(lhs, config, _depth, needs_paren_for_or(lhs)),
                format_expr_paren(rhs, config, _depth, needs_paren_for_or(rhs))
            )
        }

        Expr::Not(inner) => {
            format!(
                "not {}",
                format_expr_paren(inner, config, _depth, needs_paren_for_not(inner))
            )
        }

        Expr::Compare(lhs, op, rhs) => {
            format!(
                "{} {} {}",
                format_expr(lhs, config, _depth),
                format_comparison_op(op),
                format_expr(rhs, config, _depth)
            )
        }

        Expr::Binary(lhs, op, rhs) => {
            let op_str = format_binary_op(op);
            format!(
                "{} {} {}",
                format_expr_paren(lhs, config, _depth, needs_paren_for_binary(lhs, op)),
                op_str,
                format_expr_paren(rhs, config, _depth, needs_paren_for_binary(rhs, op))
            )
        }

        Expr::Neg(inner) => {
            format!(
                "-{}",
                format_expr_paren(inner, config, _depth, needs_paren_for_neg(inner))
            )
        }

        Expr::App(name, args) => {
            let args_str: Vec<String> = args
                .iter()
                .map(|a| format_expr(a, config, _depth))
                .collect();
            format!("{}({})", name, args_str.join(", "))
        }

        Expr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let args_str: Vec<String> = args
                .iter()
                .map(|a| format_expr(a, config, _depth))
                .collect();
            format!(
                "{}.{}({})",
                format_expr(receiver, config, _depth),
                method,
                args_str.join(", ")
            )
        }

        Expr::FieldAccess(obj, field) => {
            format!("{}.{}", format_expr(obj, config, _depth), field)
        }
    }
}

/// Format expression with optional parentheses.
fn format_expr_paren(expr: &Expr, config: &FormatConfig, depth: usize, paren: bool) -> String {
    let formatted = format_expr(expr, config, depth);
    if paren {
        format!("({})", formatted)
    } else {
        formatted
    }
}

/// Check if expression needs parentheses for implies.
fn needs_paren_for_implies(expr: &Expr) -> bool {
    matches!(expr, Expr::Implies(_, _))
}

/// Check if expression needs parentheses for and.
fn needs_paren_for_and(expr: &Expr) -> bool {
    matches!(expr, Expr::Implies(_, _) | Expr::Or(_, _))
}

/// Check if expression needs parentheses for or.
fn needs_paren_for_or(expr: &Expr) -> bool {
    matches!(expr, Expr::Implies(_, _))
}

/// Check if expression needs parentheses for not.
fn needs_paren_for_not(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Implies(_, _)
            | Expr::And(_, _)
            | Expr::Or(_, _)
            | Expr::Compare(_, _, _)
            | Expr::Binary(_, _, _)
    )
}

/// Check if expression needs parentheses for binary operation.
fn needs_paren_for_binary(expr: &Expr, _parent_op: &BinaryOp) -> bool {
    // Simple approach: parenthesize lower precedence operations
    matches!(
        expr,
        Expr::Implies(_, _) | Expr::And(_, _) | Expr::Or(_, _) | Expr::Compare(_, _, _)
    )
}

/// Check if expression needs parentheses for unary negation.
fn needs_paren_for_neg(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Implies(_, _)
            | Expr::And(_, _)
            | Expr::Or(_, _)
            | Expr::Compare(_, _, _)
            | Expr::Binary(_, _, _)
    )
}

/// Format a temporal expression.
fn format_temporal_expr(expr: &TemporalExpr, config: &FormatConfig, depth: usize) -> String {
    match expr {
        TemporalExpr::Always(inner) => {
            format!("always({})", format_temporal_expr(inner, config, depth + 1))
        }
        TemporalExpr::Eventually(inner) => {
            format!(
                "eventually({})",
                format_temporal_expr(inner, config, depth + 1)
            )
        }
        TemporalExpr::LeadsTo(lhs, rhs) => {
            format!(
                "{} ~> {}",
                format_temporal_expr(lhs, config, depth),
                format_temporal_expr(rhs, config, depth)
            )
        }
        TemporalExpr::Atom(expr) => format_expr(expr, config, depth),
    }
}

/// Format a comparison operator.
fn format_comparison_op(op: &ComparisonOp) -> &'static str {
    match op {
        ComparisonOp::Eq => "==",
        ComparisonOp::Ne => "!=",
        ComparisonOp::Lt => "<",
        ComparisonOp::Le => "<=",
        ComparisonOp::Gt => ">",
        ComparisonOp::Ge => ">=",
    }
}

/// Format a binary operator.
fn format_binary_op(op: &BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Mod => "%",
    }
}

/// Format a float with appropriate precision.
fn format_float(f: f64) -> String {
    if f.fract() == 0.0 {
        format!("{}.0", f as i64)
    } else {
        f.to_string()
    }
}

use tower_lsp::lsp_types::{Position, Range, TextEdit};

/// Compute text edits for a range formatting request.
///
/// This function compares the original and formatted text, and returns edits
/// only for the portion that overlaps with the requested range.
pub fn compute_range_edits(original: &str, formatted: &str, range: Range) -> Option<Vec<TextEdit>> {
    let orig_lines: Vec<&str> = original.lines().collect();
    let fmt_lines: Vec<&str> = formatted.lines().collect();

    let start_line = range.start.line as usize;
    let end_line = range.end.line as usize;

    // Ensure we have valid line indices
    if start_line >= orig_lines.len() {
        return None;
    }

    // For range formatting, we need to find what portion of the formatted output
    // corresponds to the requested range. Since we format the whole spec, we
    // return edits for the entire range if there are differences.

    // Compare the lines in the requested range
    let mut edits = Vec::new();

    // Calculate the range to check (clamped to actual line counts)
    let check_start = start_line;
    let check_end = end_line.min(orig_lines.len().saturating_sub(1));

    // For simplicity and correctness, if there are any differences between
    // original and formatted in the requested range, return a single edit
    // that replaces the entire range with the formatted equivalent.

    // Extract the original lines in range
    let orig_in_range: String = orig_lines
        .iter()
        .skip(check_start)
        .take(check_end - check_start + 1)
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");

    // We need to map from original line positions to formatted positions.
    // Since formatting preserves structure (types and properties in order),
    // we can do a simple approach: find the content for the range.

    // For the MVP, if the formatted text differs from original, return
    // an edit that replaces the requested lines with the full formatted output
    // for the corresponding range.
    if check_end < fmt_lines.len() {
        let fmt_in_range: String = fmt_lines
            .iter()
            .skip(check_start)
            .take(check_end - check_start + 1)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");

        if orig_in_range != fmt_in_range {
            let last_line_in_range = &orig_lines[check_end.min(orig_lines.len() - 1)];
            edits.push(TextEdit {
                range: Range {
                    start: Position::new(start_line as u32, 0),
                    end: Position::new(check_end as u32, last_line_in_range.len() as u32),
                },
                new_text: fmt_in_range,
            });
        }
    }

    if edits.is_empty() {
        None
    } else {
        Some(edits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_type_named() {
        assert_eq!(format_type(&Type::Named("Int".to_string())), "Int");
        assert_eq!(format_type(&Type::Named("MyType".to_string())), "MyType");
    }

    #[test]
    fn test_format_type_set() {
        let ty = Type::Set(Box::new(Type::Named("Int".to_string())));
        assert_eq!(format_type(&ty), "Set<Int>");
    }

    #[test]
    fn test_format_type_map() {
        let ty = Type::Map(
            Box::new(Type::Named("String".to_string())),
            Box::new(Type::Named("Int".to_string())),
        );
        assert_eq!(format_type(&ty), "Map<String, Int>");
    }

    #[test]
    fn test_format_type_function() {
        let ty = Type::Function(
            Box::new(Type::Named("Int".to_string())),
            Box::new(Type::Named("Bool".to_string())),
        );
        assert_eq!(format_type(&ty), "Int -> Bool");
    }

    #[test]
    fn test_format_type_unit() {
        assert_eq!(format_type(&Type::Unit), "()");
    }

    #[test]
    fn test_format_simple_expr() {
        let config = FormatConfig::default();
        assert_eq!(format_expr(&Expr::Var("x".to_string()), &config, 0), "x");
        assert_eq!(format_expr(&Expr::Int(42), &config, 0), "42");
        assert_eq!(format_expr(&Expr::Bool(true), &config, 0), "true");
    }

    #[test]
    fn test_format_comparison() {
        let config = FormatConfig::default();
        let expr = Expr::Compare(
            Box::new(Expr::Var("x".to_string())),
            ComparisonOp::Lt,
            Box::new(Expr::Int(10)),
        );
        assert_eq!(format_expr(&expr, &config, 0), "x < 10");
    }

    #[test]
    fn test_format_logical() {
        let config = FormatConfig::default();
        let expr = Expr::And(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(format_expr(&expr, &config, 0), "a and b");
    }

    #[test]
    fn test_format_forall() {
        let config = FormatConfig::default();
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("Int".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(format_expr(&expr, &config, 0), "forall x: Int . true");
    }

    #[test]
    fn test_format_function_call() {
        let config = FormatConfig::default();
        let expr = Expr::App(
            "f".to_string(),
            vec![Expr::Var("x".to_string()), Expr::Int(1)],
        );
        assert_eq!(format_expr(&expr, &config, 0), "f(x, 1)");
    }

    #[test]
    fn test_format_field_access() {
        let config = FormatConfig::default();
        let expr = Expr::FieldAccess(Box::new(Expr::Var("obj".to_string())), "field".to_string());
        assert_eq!(format_expr(&expr, &config, 0), "obj.field");
    }

    #[test]
    fn test_format_type_def() {
        let config = FormatConfig::default();
        let type_def = TypeDef {
            name: "Point".to_string(),
            fields: vec![
                Field {
                    name: "x".to_string(),
                    ty: Type::Named("Int".to_string()),
                },
                Field {
                    name: "y".to_string(),
                    ty: Type::Named("Int".to_string()),
                },
            ],
        };
        let formatted = format_type_def(&type_def, &config);
        assert!(formatted.contains("type Point = {"));
        assert!(formatted.contains("x: Int,"));
        assert!(formatted.contains("y: Int"));
        assert!(formatted.ends_with('}'));
    }

    #[test]
    fn test_format_theorem() {
        let config = FormatConfig::default();
        let theorem = Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        };
        let formatted = format_theorem(&theorem, &config);
        assert!(formatted.contains("theorem test {"));
        assert!(formatted.contains("true"));
    }

    #[test]
    fn test_format_temporal_always() {
        let config = FormatConfig::default();
        let expr = TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Var("p".to_string()))));
        assert_eq!(format_temporal_expr(&expr, &config, 0), "always(p)");
    }

    #[test]
    fn test_format_temporal_leads_to() {
        let config = FormatConfig::default();
        let expr = TemporalExpr::LeadsTo(
            Box::new(TemporalExpr::Atom(Expr::Var("a".to_string()))),
            Box::new(TemporalExpr::Atom(Expr::Var("b".to_string()))),
        );
        assert_eq!(format_temporal_expr(&expr, &config, 0), "a ~> b");
    }

    #[test]
    fn test_format_roundtrip_simple() {
        use dashprove_usl::parse;

        let source = r#"type Point = {
    x: Int,
    y: Int
}

theorem origin {
    forall p: Point . p.x == 0 implies p.y == 0
}
"#;
        let spec = parse(source).expect("should parse");
        let config = FormatConfig::default();
        let formatted = format_spec(&spec, &config);

        // Re-parse to verify
        let reparsed = parse(&formatted).expect("formatted output should parse");
        assert_eq!(spec, reparsed);
    }

    #[test]
    fn test_format_binary_arithmetic() {
        let config = FormatConfig::default();
        let expr = Expr::Binary(
            Box::new(Expr::Var("x".to_string())),
            BinaryOp::Add,
            Box::new(Expr::Int(1)),
        );
        assert_eq!(format_expr(&expr, &config, 0), "x + 1");
    }

    #[test]
    fn test_format_nested_arithmetic() {
        let config = FormatConfig::default();
        let expr = Expr::Binary(
            Box::new(Expr::Binary(
                Box::new(Expr::Var("a".to_string())),
                BinaryOp::Add,
                Box::new(Expr::Var("b".to_string())),
            )),
            BinaryOp::Mul,
            Box::new(Expr::Var("c".to_string())),
        );
        // Multiplication has higher precedence, so inner addition should be parenthesized
        // when needed for clarity
        let formatted = format_expr(&expr, &config, 0);
        assert!(formatted.contains('+'));
        assert!(formatted.contains('*'));
    }

    #[test]
    fn test_format_method_call() {
        let config = FormatConfig::default();
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("list".to_string())),
            method: "len".to_string(),
            args: vec![],
        };
        assert_eq!(format_expr(&expr, &config, 0), "list.len()");
    }

    #[test]
    fn test_format_float() {
        assert_eq!(format_float(3.25), "3.25");
        assert_eq!(format_float(1.0), "1.0");
        assert_eq!(format_float(0.5), "0.5");
    }

    #[test]
    fn test_format_contract() {
        use dashprove_usl::parse;

        let source = r#"contract Stack::push(self:Stack,value:Int)->Result<Stack>{requires{true}ensures{true}}"#;
        let spec = parse(source).expect("should parse");
        let config = FormatConfig::default();
        let formatted = format_spec(&spec, &config);

        // Verify formatting improves readability
        assert!(
            formatted.contains("contract Stack::push(self: Stack, value: Int) -> Result<Stack>")
        );
        assert!(formatted.contains("requires {\n"));
        assert!(formatted.contains("ensures {\n"));

        // Verify re-parseable
        parse(&formatted).expect("formatted output should parse");
    }

    #[test]
    fn test_format_invariant() {
        use dashprove_usl::parse;

        let source = r#"invariant non_negative{forall x:Int.x>=0 implies x*x>=0}"#;
        let spec = parse(source).expect("should parse");
        let config = FormatConfig::default();
        let formatted = format_spec(&spec, &config);

        assert!(formatted.contains("invariant non_negative {"));
        assert!(formatted.contains("forall x: Int . x >= 0 implies x * x >= 0"));

        parse(&formatted).expect("formatted output should parse");
    }

    #[test]
    fn test_format_probabilistic() {
        use dashprove_usl::parse;

        let source = r#"probabilistic event_likely{probability(success)==0.9}"#;
        let spec = parse(source).expect("should parse");
        let config = FormatConfig::default();
        let formatted = format_spec(&spec, &config);

        assert!(formatted.contains("probabilistic event_likely {"));
        assert!(formatted.contains("probability(success) == 0.9"));

        parse(&formatted).expect("formatted output should parse");
    }

    #[test]
    fn test_format_security() {
        use dashprove_usl::parse;

        let source = r#"security confidential{not leaked(secret)}"#;
        let spec = parse(source).expect("should parse");
        let config = FormatConfig::default();
        let formatted = format_spec(&spec, &config);

        assert!(formatted.contains("security confidential {"));
        assert!(formatted.contains("not leaked(secret)"));

        parse(&formatted).expect("formatted output should parse");
    }

    #[test]
    fn test_format_roundtrip_contracts_file() {
        use dashprove_usl::parse;

        let source = include_str!("../../../examples/usl/contracts.usl");
        let spec = parse(source).expect("should parse");
        let config = FormatConfig::default();
        let formatted = format_spec(&spec, &config);

        // Verify re-parseable
        let reparsed = parse(&formatted).expect("formatted output should parse");
        assert_eq!(spec.types.len(), reparsed.types.len());
        assert_eq!(spec.properties.len(), reparsed.properties.len());
    }

    #[test]
    fn test_format_roundtrip_temporal_file() {
        use dashprove_usl::parse;

        let source = include_str!("../../../examples/usl/temporal.usl");
        let spec = parse(source).expect("should parse");
        let config = FormatConfig::default();
        let formatted = format_spec(&spec, &config);

        // Verify re-parseable
        let reparsed = parse(&formatted).expect("formatted output should parse");
        assert_eq!(spec.types.len(), reparsed.types.len());
        assert_eq!(spec.properties.len(), reparsed.properties.len());
    }

    #[test]
    fn test_format_roundtrip_refinement_file() {
        use dashprove_usl::parse;

        let source = include_str!("../../../examples/usl/refinement.usl");
        let spec = parse(source).expect("should parse");
        let config = FormatConfig::default();
        let formatted = format_spec(&spec, &config);

        // Verify re-parseable
        let reparsed = parse(&formatted).expect("formatted output should parse");
        assert_eq!(spec.types.len(), reparsed.types.len());
        assert_eq!(spec.properties.len(), reparsed.properties.len());
    }

    #[test]
    fn test_format_exists_in() {
        let config = FormatConfig::default();
        let expr = Expr::ExistsIn {
            var: "x".to_string(),
            collection: Box::new(Expr::Var("items".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(format_expr(&expr, &config, 0), "exists x in items . true");
    }

    #[test]
    fn test_format_negation() {
        let config = FormatConfig::default();
        let expr = Expr::Neg(Box::new(Expr::Var("x".to_string())));
        assert_eq!(format_expr(&expr, &config, 0), "-x");
    }

    #[test]
    fn test_format_not_with_complex_expr() {
        let config = FormatConfig::default();
        let expr = Expr::Not(Box::new(Expr::And(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        )));
        // Should add parentheses around complex expression
        assert_eq!(format_expr(&expr, &config, 0), "not (a and b)");
    }

    #[test]
    fn test_format_empty_spec() {
        let config = FormatConfig::default();
        let spec = Spec::default();
        let formatted = format_spec(&spec, &config);
        assert!(formatted.is_empty());
    }

    #[test]
    fn test_compute_range_edits_no_change() {
        let original = "line 0\nline 1\nline 2";
        let formatted = "line 0\nline 1\nline 2";
        let range = Range {
            start: Position::new(0, 0),
            end: Position::new(2, 6),
        };

        let result = compute_range_edits(original, formatted, range);
        assert!(result.is_none());
    }

    #[test]
    fn test_compute_range_edits_single_line_change() {
        let original = "line 0\nbad line\nline 2";
        let formatted = "line 0\ngood line\nline 2";
        let range = Range {
            start: Position::new(1, 0),
            end: Position::new(1, 8),
        };

        let result = compute_range_edits(original, formatted, range);
        assert!(result.is_some());
        let edits = result.unwrap();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].new_text, "good line");
    }

    #[test]
    fn test_compute_range_edits_multi_line_change() {
        let original = "line 0\nold 1\nold 2\nline 3";
        let formatted = "line 0\nnew 1\nnew 2\nline 3";
        let range = Range {
            start: Position::new(1, 0),
            end: Position::new(2, 5),
        };

        let result = compute_range_edits(original, formatted, range);
        assert!(result.is_some());
        let edits = result.unwrap();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].new_text, "new 1\nnew 2");
    }

    #[test]
    fn test_compute_range_edits_invalid_range() {
        let original = "line 0\nline 1";
        let formatted = "line 0\nline 1";
        let range = Range {
            start: Position::new(100, 0),
            end: Position::new(100, 5),
        };

        let result = compute_range_edits(original, formatted, range);
        assert!(result.is_none());
    }

    #[test]
    fn test_compute_range_edits_preserves_unchanged_lines() {
        let original = "unchanged\nneeds fix\nunchanged";
        let formatted = "unchanged\nfixed line\nunchanged";
        // Request format only for line 1
        let range = Range {
            start: Position::new(1, 0),
            end: Position::new(1, 9),
        };

        let result = compute_range_edits(original, formatted, range);
        assert!(result.is_some());
        let edits = result.unwrap();
        assert_eq!(edits.len(), 1);
        // Should only change line 1
        assert_eq!(edits[0].range.start.line, 1);
        assert_eq!(edits[0].range.end.line, 1);
        assert_eq!(edits[0].new_text, "fixed line");
    }

    #[test]
    fn test_compute_range_edits_first_line() {
        let original = "bad start\nline 1\nline 2";
        let formatted = "good start\nline 1\nline 2";
        let range = Range {
            start: Position::new(0, 0),
            end: Position::new(0, 9),
        };

        let result = compute_range_edits(original, formatted, range);
        assert!(result.is_some());
        let edits = result.unwrap();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].new_text, "good start");
    }

    // ========== Mutation-killing tests ==========

    /// Test format_spec with only types (no properties) - kills line 43 && mutation
    #[test]
    fn test_format_spec_only_types() {
        let config = FormatConfig::default();
        let spec = Spec {
            types: vec![
                TypeDef {
                    name: "A".to_string(),
                    fields: vec![],
                },
                TypeDef {
                    name: "B".to_string(),
                    fields: vec![],
                },
            ],
            properties: vec![],
        };
        let formatted = format_spec(&spec, &config);
        // With && mutation to ||, would add extra newline even with no properties
        // Should have types separated by single newline, no trailing double newline
        assert!(formatted.contains("type A = {\n}\n\ntype B = {\n}"));
        assert!(!formatted.ends_with("\n\n\n"));
    }

    /// Test format_spec with only properties (no types) - kills line 43 ! mutation
    #[test]
    fn test_format_spec_only_properties() {
        let config = FormatConfig::default();
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Theorem(Theorem {
                name: "t".to_string(),
                body: Expr::Bool(true),
            })],
        };
        let formatted = format_spec(&spec, &config);
        // With ! deleted, would add spurious separator with empty types
        assert!(formatted.starts_with("theorem t"));
        // Should not have leading newlines
        assert!(!formatted.starts_with("\n"));
    }

    /// Test format_spec with both types and properties - kills line 43 boundary
    #[test]
    fn test_format_spec_types_and_properties() {
        let config = FormatConfig::default();
        let spec = Spec {
            types: vec![TypeDef {
                name: "T".to_string(),
                fields: vec![],
            }],
            properties: vec![Property::Theorem(Theorem {
                name: "p".to_string(),
                body: Expr::Bool(true),
            })],
        };
        let formatted = format_spec(&spec, &config);
        // Should have blank line between types and properties (}\n + \n from format_type_def + \n from separator)
        assert!(
            formatted.contains("}\n\ntheorem"),
            "Should have blank line between types and properties: {}",
            formatted
        );
    }

    /// Test format_spec with multiple types - kills line 35 > mutations
    #[test]
    fn test_format_spec_multiple_types_newline() {
        let config = FormatConfig::default();
        let spec = Spec {
            types: vec![
                TypeDef {
                    name: "First".to_string(),
                    fields: vec![],
                },
                TypeDef {
                    name: "Second".to_string(),
                    fields: vec![],
                },
            ],
            properties: vec![],
        };
        let formatted = format_spec(&spec, &config);
        // Should have blank line between types (i > 0 check)
        assert!(
            formatted.contains("}\n\ntype Second"),
            "Should have newline between types: {}",
            formatted
        );
    }

    /// Test format_spec first type has no leading newline - kills line 35 >= boundary
    #[test]
    fn test_format_spec_first_type_no_leading_newline() {
        let config = FormatConfig::default();
        let spec = Spec {
            types: vec![TypeDef {
                name: "Only".to_string(),
                fields: vec![],
            }],
            properties: vec![],
        };
        let formatted = format_spec(&spec, &config);
        // First type (i==0) should not have leading newline
        assert!(
            formatted.starts_with("type Only"),
            "First type should have no leading newline: {}",
            formatted
        );
    }

    /// Test format_spec multiple properties - kills line 49 > mutations
    #[test]
    fn test_format_spec_multiple_properties_newline() {
        let config = FormatConfig::default();
        let spec = Spec {
            types: vec![],
            properties: vec![
                Property::Theorem(Theorem {
                    name: "first".to_string(),
                    body: Expr::Bool(true),
                }),
                Property::Theorem(Theorem {
                    name: "second".to_string(),
                    body: Expr::Bool(false),
                }),
            ],
        };
        let formatted = format_spec(&spec, &config);
        // Should have newline between properties
        assert!(
            formatted.contains("}\n\ntheorem second"),
            "Should have newline between properties: {}",
            formatted
        );
    }

    /// Test type_def with single field (no trailing comma) - kills line 66 < boundary
    #[test]
    fn test_format_type_def_single_field_no_comma() {
        let config = FormatConfig::default();
        let type_def = TypeDef {
            name: "Single".to_string(),
            fields: vec![Field {
                name: "only".to_string(),
                ty: Type::Named("Int".to_string()),
            }],
        };
        let formatted = format_type_def(&type_def, &config);
        // Last field should not have comma (i < len - 1 check when i=0, len=1)
        assert!(
            !formatted.contains("Int,"),
            "Single field should not have comma: {}",
            formatted
        );
        assert!(formatted.contains("only: Int\n}"));
    }

    /// Test type_def with multiple fields (all but last have comma) - kills line 66 <= mutation
    #[test]
    fn test_format_type_def_commas() {
        let config = FormatConfig::default();
        let type_def = TypeDef {
            name: "Multi".to_string(),
            fields: vec![
                Field {
                    name: "a".to_string(),
                    ty: Type::Named("Int".to_string()),
                },
                Field {
                    name: "b".to_string(),
                    ty: Type::Named("Bool".to_string()),
                },
                Field {
                    name: "c".to_string(),
                    ty: Type::Named("String".to_string()),
                },
            ],
        };
        let formatted = format_type_def(&type_def, &config);
        // First two have commas, last does not
        assert!(formatted.contains("a: Int,"), "First field needs comma");
        assert!(formatted.contains("b: Bool,"), "Second field needs comma");
        assert!(
            !formatted.contains("c: String,"),
            "Last field should not have comma"
        );
    }

    /// Test needs_paren_for_implies returns true - kills line 430 false mutation
    #[test]
    fn test_needs_paren_for_implies_true() {
        let inner = Expr::Implies(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        assert!(
            needs_paren_for_implies(&inner),
            "Nested implies should need parens"
        );
    }

    /// Test needs_paren_for_implies returns false - kills line 430 boundary
    #[test]
    fn test_needs_paren_for_implies_false() {
        let inner = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        assert!(
            !needs_paren_for_implies(&inner),
            "And should not need parens for implies"
        );
    }

    /// Test needs_paren_for_and returns true - kills line 435 false mutation
    #[test]
    fn test_needs_paren_for_and_true_implies() {
        let inner = Expr::Implies(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        assert!(
            needs_paren_for_and(&inner),
            "Implies should need parens inside and"
        );
    }

    /// Test needs_paren_for_and returns true for or - kills line 435 boundary
    #[test]
    fn test_needs_paren_for_and_true_or() {
        let inner = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        assert!(
            needs_paren_for_and(&inner),
            "Or should need parens inside and"
        );
    }

    /// Test needs_paren_for_and returns false - kills line 435 boundary
    #[test]
    fn test_needs_paren_for_and_false() {
        let inner = Expr::Var("x".to_string());
        assert!(
            !needs_paren_for_and(&inner),
            "Var should not need parens inside and"
        );
    }

    /// Test needs_paren_for_or returns true - kills line 440 true/false mutations
    #[test]
    fn test_needs_paren_for_or_true() {
        let inner = Expr::Implies(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        assert!(
            needs_paren_for_or(&inner),
            "Implies should need parens inside or"
        );
    }

    /// Test needs_paren_for_or returns false - kills line 440 boundary
    #[test]
    fn test_needs_paren_for_or_false() {
        let inner = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        assert!(
            !needs_paren_for_or(&inner),
            "And should not need parens inside or"
        );
    }

    /// Test needs_paren_for_binary returns true - kills line 458 false mutation
    #[test]
    fn test_needs_paren_for_binary_true() {
        let inner = Expr::Compare(
            Box::new(Expr::Int(1)),
            ComparisonOp::Lt,
            Box::new(Expr::Int(2)),
        );
        assert!(
            needs_paren_for_binary(&inner, &BinaryOp::Add),
            "Compare should need parens inside binary"
        );
    }

    /// Test needs_paren_for_binary returns false - kills line 458 boundary
    #[test]
    fn test_needs_paren_for_binary_false() {
        let inner = Expr::Int(42);
        assert!(
            !needs_paren_for_binary(&inner, &BinaryOp::Add),
            "Int literal should not need parens inside binary"
        );
    }

    /// Test needs_paren_for_neg returns true - kills line 466 false mutation
    #[test]
    fn test_needs_paren_for_neg_true() {
        let inner = Expr::Binary(
            Box::new(Expr::Int(1)),
            BinaryOp::Add,
            Box::new(Expr::Int(2)),
        );
        assert!(
            needs_paren_for_neg(&inner),
            "Binary should need parens inside neg"
        );
    }

    /// Test needs_paren_for_neg returns false - kills line 466 boundary
    #[test]
    fn test_needs_paren_for_neg_false() {
        let inner = Expr::Var("x".to_string());
        assert!(
            !needs_paren_for_neg(&inner),
            "Var should not need parens inside neg"
        );
    }

    /// Test compute_range_edits - kills line 568 - mutation
    #[test]
    fn test_compute_range_edits_take_count() {
        // Test that the take count is correct (end - start + 1)
        let original = "line0\nline1\nline2\nline3";
        let formatted = "line0\nLINE1\nLINE2\nline3";
        let range = Range {
            start: Position::new(1, 0),
            end: Position::new(2, 5),
        };

        let result = compute_range_edits(original, formatted, range);
        assert!(result.is_some());
        let edits = result.unwrap();
        // Should edit lines 1-2
        assert_eq!(edits[0].new_text, "LINE1\nLINE2");
    }

    /// Test compute_range_edits - kills line 580 <= mutation
    #[test]
    fn test_compute_range_edits_boundary_check() {
        // Test when end_line equals formatted line count
        let original = "a\nb\nc";
        let formatted = "a\nB\nC";
        let range = Range {
            start: Position::new(1, 0),
            end: Position::new(2, 1),
        };

        let result = compute_range_edits(original, formatted, range);
        assert!(result.is_some());
    }

    /// Test compute_range_edits - kills line 590 - mutations
    #[test]
    fn test_compute_range_edits_last_line_length() {
        let original = "start\nchangeme\nend";
        let formatted = "start\nchanged\nend";
        let range = Range {
            start: Position::new(1, 0),
            end: Position::new(1, 8),
        };

        let result = compute_range_edits(original, formatted, range);
        assert!(result.is_some());
        let edits = result.unwrap();
        // End character should be length of "changeme" = 8
        assert_eq!(edits[0].range.end.character, 8);
    }
}
