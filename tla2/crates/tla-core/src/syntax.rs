//! Lexer and concrete syntax tree for TLA+
//!
//! This module uses:
//! - `logos` for fast lexical analysis
//! - `rowan` for lossless syntax tree (preserves whitespace, comments)

pub mod kinds;
pub mod lexer;
pub mod parser;

pub use kinds::{SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken, TlaLanguage};
pub use lexer::Token;
pub use parser::{parse, parse_to_syntax_tree, ParseError, ParseResult};
