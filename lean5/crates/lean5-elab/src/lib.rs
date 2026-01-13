//! Lean5 Elaborator
//!
//! Converts surface syntax to kernel terms via:
//! - Type inference with metavariables
//! - Named to de Bruijn conversion
//! - Implicit argument insertion (future)
//! - Type class instance resolution (future)
//! - Macro expansion before elaboration
//!
//! # Example
//!
//! ```
//! use lean5_elab::{elaborate, ElabCtx};
//! use lean5_kernel::Environment;
//! use lean5_parser::parse_expr;
//!
//! let env = Environment::new();
//! let mut ctx = ElabCtx::new(&env);
//! let surface = parse_expr("fun (x : Type) => x").unwrap();
//! let kernel_expr = ctx.elaborate(&surface).unwrap();
//! ```

pub mod infer;
pub mod instances;
pub mod macro_integration;
pub mod tactic;
pub mod unify;

use lean5_kernel::Name;

pub use infer::{DerivedInstance, ElabCtx, ElabResult};
pub use instances::{ClassInfo, InstanceInfo, InstanceTable, DEFAULT_PRIORITY};
pub use macro_integration::{
    expand_surface_macros, surface_to_syntax, syntax_to_surface, MacroCtx, MacroExpansionError,
};
pub use tactic::{
    apply, assumption, constructor, exact, intro, intros, rfl, Goal, LocalDecl, ProofState,
    TacticError, TacticResult,
};
pub use unify::{MetaId, MetaState, MetaVar, Unifier, UnifyResult};

/// Elaborate surface syntax to kernel expression
pub fn elaborate(
    env: &lean5_kernel::Environment,
    surface: &lean5_parser::SurfaceExpr,
) -> Result<lean5_kernel::Expr, ElabError> {
    let mut ctx = ElabCtx::new(env);
    ctx.elaborate(surface)
}

/// Elaborate a surface declaration to a kernel declaration result
pub fn elaborate_decl(
    env: &lean5_kernel::Environment,
    decl: &lean5_parser::SurfaceDecl,
) -> Result<ElabResult, ElabError> {
    let mut ctx = ElabCtx::new(env);
    ctx.elab_decl(decl)
}

#[derive(Debug, thiserror::Error)]
pub enum ElabError {
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },
    #[error("Unknown identifier: {0}")]
    UnknownIdent(String),
    #[error("Cannot infer type")]
    CannotInfer,
    #[error("Invalid projection target: {0}")]
    InvalidProjectionTarget(String),
    #[error("Unknown projection field {field} on structure {struct_name}")]
    UnknownProjectionField { struct_name: Name, field: String },
    #[error("Projection index {idx} out of bounds for {struct_name} (fields: {field_count})")]
    ProjectionIndexOutOfBounds {
        struct_name: Name,
        idx: u32,
        field_count: u32,
    },
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Macro expansion failed: {0}")]
    MacroError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use lean5_kernel::Environment;
    use lean5_parser::parse_expr;

    #[test]
    fn test_elaborate_type() {
        let env = Environment::new();
        let surface = parse_expr("Type").unwrap();
        let result = elaborate(&env, &surface).unwrap();
        assert!(matches!(result, lean5_kernel::Expr::Sort(_)));
    }

    #[test]
    fn test_elaborate_identity() {
        let env = Environment::new();
        let surface = parse_expr("fun (A : Type) (x : A) => x").unwrap();
        let result = elaborate(&env, &surface).unwrap();
        assert!(matches!(result, lean5_kernel::Expr::Lam(_, _, _)));
    }

    #[test]
    fn test_elaborate_arrow() {
        let env = Environment::new();
        let surface = parse_expr("Type -> Type").unwrap();
        let result = elaborate(&env, &surface).unwrap();
        assert!(matches!(result, lean5_kernel::Expr::Pi(_, _, _)));
    }
}
