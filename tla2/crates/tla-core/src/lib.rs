//! tla-core: Core TLA+ parser, AST, semantic analysis, and evaluation
//!
//! This crate provides the foundation for all TLA2 tools:
//! - Lexer and parser for TLA+ source code
//! - Abstract syntax tree (AST) with source spans
//! - Semantic analysis (name resolution, scope checking)
//! - Expression evaluation
//! - Module system support (EXTENDS, INSTANCE)
//!
//! # Architecture
//!
//! ```text
//! TLA+ Source
//!      │
//!      ▼
//! ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
//! │  Lexer  │───▶│ Parser  │───▶│   AST   │───▶│Resolver │
//! │ (logos) │    │(rowan)  │    │         │    │         │
//! └─────────┘    └─────────┘    └─────────┘    └─────────┘
//! ```

pub mod ast;
pub mod diagnostic;
pub mod error;
pub mod loader;
pub mod lower;
pub mod pretty;
pub mod resolve;
pub mod span;
pub mod stdlib;
pub mod syntax;

// Re-exports
pub use error::{Error, Result};
pub use span::{FileId, Span, Spanned};

// Syntax re-exports
pub use syntax::Token;
pub use syntax::{parse, parse_to_syntax_tree, ParseError, ParseResult};
pub use syntax::{SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken, TlaLanguage};

// Lower re-exports
pub use lower::{lower, lower_single_expr, LowerCtx, LowerError, LowerResult};

// Pretty-print re-exports
pub use pretty::{pretty_expr, pretty_module, PrettyPrinter};

// Resolve re-exports
pub use resolve::{
    inject_module_symbols, resolve, resolve_with_extends, ResolveCtx, ResolveError, ResolveResult,
    Symbol, SymbolKind,
};

// Loader re-exports
pub use loader::{LoadError, LoadedModule, ModuleLoader};

// Diagnostic re-exports
pub use diagnostic::{Diagnostic, DiagnosticLabel, DiagnosticSpan, Severity, SourceCache};
