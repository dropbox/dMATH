//! Lean5 Macro System
//!
//! This crate provides the macro system for Lean5, enabling:
//! - Syntax quotations: `` `(term) `` for constructing syntax
//! - Antiquotations: `$x` for splicing values into quotations
//! - Macro definitions: `macro` and `macro_rules`
//! - Syntax extensions: `syntax` and `declare_syntax_cat`
//! - Hygiene: fresh name generation to prevent variable capture
//! - Built-in macros: `do` notation, `if let`, `when`, `unless`, etc.
//!
//! The design follows Lean 4's macro system architecture.

pub mod builtins;
pub mod expand;
pub mod hygiene;
pub mod quotation;
pub mod registry;
pub mod syntax;

pub use builtins::{builtin_registry, register_builtins};
pub use expand::{expand_hygienic, HygienicExpander, MacroExpander, MacroResult};
pub use hygiene::{HygieneContext, HygieneState, MacroScope, ScopedName};
pub use quotation::{Antiquotation, SyntaxQuote};
pub use registry::{MacroDef, MacroRegistry};
pub use syntax::{SourceInfo, Syntax, SyntaxKind, SyntaxNode};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_syntax_construction() {
        let ident = Syntax::ident("foo");
        assert!(ident.is_ident());
        assert_eq!(ident.as_ident(), Some("foo"));
    }

    #[test]
    fn test_syntax_node_construction() {
        let node = Syntax::node(
            SyntaxKind::app("App"),
            vec![Syntax::ident("f"), Syntax::ident("x")],
        );
        assert!(node.is_node());
        assert_eq!(node.children().len(), 2);
    }

    #[test]
    fn test_atom_construction() {
        let nat = Syntax::atom("42");
        assert!(nat.is_atom());
        assert_eq!(nat.as_atom(), Some("42"));
    }
}
