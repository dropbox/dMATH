//! Lean5 Parser
//!
//! Parses Lean 4 syntax into surface syntax AST.
//!
//! # Overview
//!
//! The parser is a recursive descent parser that produces a surface syntax AST.
//! The AST uses named bindings (not de Bruijn indices) and preserves source spans.
//!
//! # Example
//!
//! ```
//! use lean5_parser::{parse_expr, parse_decl};
//!
//! let expr = parse_expr("fun x => x").unwrap();
//! let decl = parse_decl("def id (x : Type) := x").unwrap();
//! ```

pub mod grammar;
pub mod lexer;
pub mod surface;

#[cfg(test)]
mod lean4_compat;

#[cfg(test)]
mod lean4_features;

pub use grammar::Parser;
pub use surface::{
    Attribute, LevelExpr, MacroArm, NotationItem, NotationKind, PrecedenceLevel, Projection, Span,
    SurfaceArg, SurfaceBinder, SurfaceBinderInfo, SurfaceCtor, SurfaceDecl, SurfaceExpr,
    SurfaceField, SurfaceFieldAssign, SurfaceLit, SurfaceMatchArm, SurfacePattern,
    SyntaxPatternItem, UniverseExpr,
};

/// Parse a string into a surface expression
///
/// # Errors
///
/// Returns a `ParseError` if the input is not a valid expression.
pub fn parse_expr(input: &str) -> Result<SurfaceExpr, ParseError> {
    Parser::parse_expr(input)
}

/// Parse a string into a surface declaration
///
/// # Errors
///
/// Returns a `ParseError` if the input is not a valid declaration.
pub fn parse_decl(input: &str) -> Result<SurfaceDecl, ParseError> {
    Parser::parse_decl(input)
}

/// Parse a file containing multiple declarations
///
/// # Errors
///
/// Returns a `ParseError` if the input contains invalid declarations.
pub fn parse_file(input: &str) -> Result<Vec<SurfaceDecl>, ParseError> {
    Parser::parse_file(input)
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Unexpected token at {line}:{col}: {message}")]
    UnexpectedToken {
        line: usize,
        col: usize,
        message: String,
    },
    #[error("Unexpected end of input")]
    UnexpectedEof,
    #[error("Numeric literal {value} is too large (max {max})")]
    NumericOverflow { value: u64, max: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_identity() {
        let expr = parse_expr("fun x => x").unwrap();
        assert!(matches!(expr, SurfaceExpr::Lambda(_, _, _)));
    }

    #[test]
    fn test_parse_def() {
        let decl = parse_decl("def id (x : Type) := x").unwrap();
        match decl {
            SurfaceDecl::Def { name, .. } => assert_eq!(name, "id"),
            _ => panic!("expected Def"),
        }
    }

    #[test]
    fn test_parse_arrow_chain() {
        let expr = parse_expr("A -> B -> C").unwrap();
        // Right associative: A -> (B -> C)
        match expr {
            SurfaceExpr::Arrow(_, left, right) => {
                assert!(matches!(*left, SurfaceExpr::Ident(_, _)));
                assert!(matches!(*right, SurfaceExpr::Arrow(_, _, _)));
            }
            _ => panic!("expected Arrow"),
        }
    }

    #[test]
    fn test_parse_projection_index() {
        let expr = parse_expr("x.1").unwrap();
        match expr {
            SurfaceExpr::Proj(_, base, Projection::Index(1)) => {
                assert!(matches!(*base, SurfaceExpr::Ident(_, ref name) if name == "x"));
            }
            other => panic!("expected projection, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_projection_named_after_app() {
        let expr = parse_expr("(f x).field").unwrap();
        match expr {
            SurfaceExpr::Proj(_, base, Projection::Named(ref field)) => {
                assert_eq!(field, "field");
                let inner = match base.as_ref() {
                    SurfaceExpr::Paren(_, inner) => inner.as_ref(),
                    other => other,
                };
                assert!(
                    matches!(inner, SurfaceExpr::App(_, _, _)),
                    "expected application base, got {inner:?}"
                );
            }
            other => panic!("expected named projection, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_syntax_quote_preserved() {
        let expr = parse_expr("`(x)").unwrap();
        match expr {
            SurfaceExpr::SyntaxQuote(_, content) => assert!(content.contains('x')),
            other => panic!("expected syntax quote, got {other:?}"),
        }
    }
}
