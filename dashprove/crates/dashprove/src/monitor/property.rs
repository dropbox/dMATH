//! Property extraction and classification utilities

use crate::monitor::config::PropertyKind;
use crate::monitor::utils::to_snake_case;
use dashprove_usl::ast::{Contract, Expr, Property, TemporalExpr};

/// Determine property kind from AST
pub fn property_kind(prop: &Property) -> PropertyKind {
    match prop {
        Property::Invariant(_) => PropertyKind::Invariant,
        Property::Temporal(_) => PropertyKind::Temporal,
        Property::Contract(c) => {
            if c.requires.is_empty() {
                PropertyKind::Postcondition
            } else {
                PropertyKind::Precondition
            }
        }
        _ => PropertyKind::Invariant,
    }
}

/// Extract expression from property for compilation
pub fn property_expr(prop: &Property) -> Option<&Expr> {
    match prop {
        Property::Theorem(t) => Some(&t.body),
        Property::Invariant(i) => Some(&i.body),
        Property::Security(s) => Some(&s.body),
        Property::Probabilistic(p) => Some(&p.condition),
        _ => None,
    }
}

/// Extract temporal expression from property
pub fn property_temporal_expr(prop: &Property) -> Option<&TemporalExpr> {
    match prop {
        Property::Temporal(t) => Some(&t.body),
        _ => None,
    }
}

/// Generate base name for contract methods
pub fn contract_base_name(contract: &Contract) -> String {
    if contract.type_path.is_empty() {
        "contract".to_string()
    } else {
        to_snake_case(&contract.type_path.join("_"))
    }
}
