//! PCTL/CSL property generation from USL specs

use super::util::to_prism_ident;
use dashprove_usl::ast::{ComparisonOp, Expr, Property};
use dashprove_usl::typecheck::TypedSpec;

/// Generate PCTL/CSL property from USL probabilistic property
pub fn generate_pctl_property(spec: &TypedSpec) -> String {
    let mut properties = Vec::new();

    for property in &spec.spec.properties {
        if let Property::Probabilistic(prob) = property {
            // Compile comparison operator
            let op_str = match prob.comparison {
                ComparisonOp::Ge => ">=",
                ComparisonOp::Gt => ">",
                ComparisonOp::Le => "<=",
                ComparisonOp::Lt => "<",
                ComparisonOp::Eq => "=",
                ComparisonOp::Ne => "!=",
            };

            // Compile condition to PCTL path formula
            let path_formula = compile_pctl_condition(&prob.condition);

            // Format: P>=bound[F condition] or P>=bound[G condition]
            let pctl = format!("P{}{}[{}]", op_str, prob.bound, path_formula);
            properties.push(pctl);
        }
    }

    if properties.is_empty() {
        // Default reachability property
        "P=? [F state=MAX_STATE]".to_string()
    } else {
        properties.join(" & ")
    }
}

/// Compile USL expression to PCTL path formula
pub fn compile_pctl_condition(expr: &Expr) -> String {
    match expr {
        // Function applications translate to path operators or state predicates
        Expr::App(name, args) => {
            let lower_name = name.to_lowercase();

            // Temporal operators
            if lower_name == "eventually" || lower_name == "finally" {
                if let Some(arg) = args.first() {
                    return format!("F {}", compile_pctl_condition(arg));
                }
            }
            if lower_name == "always" || lower_name == "globally" {
                if let Some(arg) = args.first() {
                    return format!("G {}", compile_pctl_condition(arg));
                }
            }
            if lower_name == "until" && args.len() >= 2 {
                return format!(
                    "{} U {}",
                    compile_pctl_condition(&args[0]),
                    compile_pctl_condition(&args[1])
                );
            }
            if lower_name == "next" {
                if let Some(arg) = args.first() {
                    return format!("X {}", compile_pctl_condition(arg));
                }
            }

            // State predicates - check for labels
            if lower_name.contains("goal")
                || lower_name.contains("done")
                || lower_name.contains("success")
            {
                return format!("\"{}\"", to_prism_ident(name));
            }

            // Generic function call → state formula
            format!("({})", to_prism_ident(name))
        }

        Expr::Var(name) => {
            if name == "true" {
                "true".to_string()
            } else if name == "false" {
                "false".to_string()
            } else {
                to_prism_ident(name)
            }
        }

        Expr::Compare(left, op, right) => {
            let l = compile_pctl_condition(left);
            let r = compile_pctl_condition(right);
            let op_str = match op {
                ComparisonOp::Eq => "=",
                ComparisonOp::Ne => "!=",
                ComparisonOp::Lt => "<",
                ComparisonOp::Le => "<=",
                ComparisonOp::Gt => ">",
                ComparisonOp::Ge => ">=",
            };
            format!("({} {} {})", l, op_str, r)
        }

        Expr::And(left, right) => {
            format!(
                "({} & {})",
                compile_pctl_condition(left),
                compile_pctl_condition(right)
            )
        }

        Expr::Or(left, right) => {
            format!(
                "({} | {})",
                compile_pctl_condition(left),
                compile_pctl_condition(right)
            )
        }

        Expr::Not(inner) => {
            format!("!({})", compile_pctl_condition(inner))
        }

        Expr::Implies(left, right) => {
            // P implies Q  ≡  !P | Q
            format!(
                "(!{} | {})",
                compile_pctl_condition(left),
                compile_pctl_condition(right)
            )
        }

        Expr::Int(n) => n.to_string(),
        Expr::Float(f) => format!("{:.6}", f),
        Expr::Bool(b) => b.to_string(),
        Expr::String(s) => format!("\"{}\"", s),

        Expr::FieldAccess(_, field) => to_prism_ident(field),

        Expr::Binary(left, op, right) => {
            let l = compile_pctl_condition(left);
            let r = compile_pctl_condition(right);
            let op_str = match op {
                dashprove_usl::ast::BinaryOp::Add => "+",
                dashprove_usl::ast::BinaryOp::Sub => "-",
                dashprove_usl::ast::BinaryOp::Mul => "*",
                dashprove_usl::ast::BinaryOp::Div => "/",
                dashprove_usl::ast::BinaryOp::Mod => "mod",
            };
            format!("({} {} {})", l, op_str, r)
        }

        // Quantifiers in PCTL context
        Expr::ForAll { body, .. } | Expr::Exists { body, .. } => compile_pctl_condition(body),

        Expr::ForAllIn { body, .. } | Expr::ExistsIn { body, .. } => compile_pctl_condition(body),

        Expr::MethodCall { receiver, .. } => compile_pctl_condition(receiver),

        Expr::Neg(inner) => {
            format!("-({})", compile_pctl_condition(inner))
        }
    }
}
