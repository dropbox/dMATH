//! Syntax quotations and antiquotations
//!
//! Syntax quotations allow constructing syntax programmatically:
//! - `` `(term) `` - quote a term, producing a `Syntax` value
//! - `$x` - antiquotation: splice a `Syntax` value into a quotation
//! - `$[xs]*` - splice antiquotation: splice multiple values
//!
//! Example:
//! ```text
//! def mkAdd (x y : Syntax) : Syntax := `($x + $y)
//! ```

use crate::syntax::{SourceInfo, Syntax, SyntaxKind};
use std::collections::HashMap;

/// A syntax quotation with potential antiquotations
#[derive(Debug, Clone)]
pub struct SyntaxQuote {
    /// The quoted syntax (may contain antiquotation nodes)
    pub syntax: Syntax,
    /// The category of syntax being quoted (term, command, tactic, etc.)
    pub category: SyntaxKind,
}

impl SyntaxQuote {
    /// Create a new syntax quotation
    pub fn new(syntax: Syntax, category: SyntaxKind) -> Self {
        Self { syntax, category }
    }

    /// Create a term quotation
    pub fn term(syntax: Syntax) -> Self {
        Self::new(syntax, SyntaxKind::term())
    }

    /// Create a command quotation
    pub fn command(syntax: Syntax) -> Self {
        Self::new(syntax, SyntaxKind::command())
    }

    /// Create a tactic quotation
    pub fn tactic(syntax: Syntax) -> Self {
        Self::new(syntax, SyntaxKind::tactic())
    }

    /// Get all antiquotation names in this quotation
    pub fn antiquot_names(&self) -> Vec<String> {
        self.syntax
            .collect_antiquots()
            .iter()
            .filter_map(|s| {
                s.children()
                    .first()
                    .and_then(|c| c.as_ident())
                    .map(String::from)
            })
            .collect()
    }

    /// Check if this quotation has any antiquotations
    pub fn has_antiquots(&self) -> bool {
        !self.syntax.collect_antiquots().is_empty()
    }

    /// Substitute antiquotations with provided values
    pub fn substitute(&self, bindings: &HashMap<String, Syntax>) -> Syntax {
        substitute_antiquots(&self.syntax, bindings)
    }
}

/// An antiquotation inside a quotation
#[derive(Debug, Clone)]
pub struct Antiquotation {
    /// The name of the antiquotation variable
    pub name: String,
    /// Whether this is a splice antiquotation `$[x]*`
    pub is_splice: bool,
    /// Optional syntax category annotation (`$x:term`, `$x:tactic`, etc.)
    /// When set, this constrains what kind of syntax the antiquotation expects
    pub category: Option<String>,
    /// Source location
    pub info: SourceInfo,
}

impl Antiquotation {
    /// Create a simple antiquotation `$name`
    pub fn simple(name: &str) -> Self {
        Self {
            name: name.to_string(),
            is_splice: false,
            category: None,
            info: SourceInfo::dummy(),
        }
    }

    /// Create a type-annotated antiquotation `$name:category`
    pub fn typed(name: &str, category: &str) -> Self {
        Self {
            name: name.to_string(),
            is_splice: false,
            category: Some(category.to_string()),
            info: SourceInfo::dummy(),
        }
    }

    /// Create a splice antiquotation `$[name]*`
    pub fn splice(name: &str) -> Self {
        Self {
            name: name.to_string(),
            is_splice: true,
            category: None,
            info: SourceInfo::dummy(),
        }
    }

    /// Create a typed splice antiquotation `$[name:category]*`
    pub fn typed_splice(name: &str, category: &str) -> Self {
        Self {
            name: name.to_string(),
            is_splice: true,
            category: Some(category.to_string()),
            info: SourceInfo::dummy(),
        }
    }

    /// Create from a syntax node
    pub fn from_syntax(syntax: &Syntax) -> Option<Self> {
        if !syntax.is_antiquot() {
            return None;
        }

        let kind = syntax.kind()?;
        let is_splice = kind.is_antiquotation() && kind.name_str().contains("splice");

        // Check for typed antiquotation (has category child)
        let children = syntax.children();
        let name = children.first()?.as_ident()?.to_string();

        // Second child is the category annotation if present
        let category = children.get(1).and_then(|c| c.as_ident()).map(String::from);

        Some(Self {
            name,
            is_splice,
            category,
            info: syntax.source_info().clone(),
        })
    }

    /// Check if this antiquotation has a type annotation
    pub fn is_typed(&self) -> bool {
        self.category.is_some()
    }
}

/// Substitute all antiquotations in syntax with provided values
/// Handles both simple antiquotations and splice antiquotations
fn substitute_antiquots(syntax: &Syntax, bindings: &HashMap<String, Syntax>) -> Syntax {
    substitute_recursive(syntax, bindings)
}

/// Recursive substitution that handles splices in node children
fn substitute_recursive(syntax: &Syntax, bindings: &HashMap<String, Syntax>) -> Syntax {
    // Check if this is an antiquotation to substitute
    if syntax.is_antiquot() {
        if let Some(antiquot) = Antiquotation::from_syntax(syntax) {
            if let Some(replacement) = bindings.get(&antiquot.name) {
                // For splice bindings (splice_list nodes), if used in a simple antiquot position,
                // we just return the splice_list as-is. It will be expanded by the parent if needed.
                return replacement.clone();
            }
        }
        // No binding found, leave as-is
        return syntax.clone();
    }

    // For nodes, recursively process children and handle splices
    match syntax {
        Syntax::Node(node) => {
            // Check if any children are splice antiquotations that need expansion
            let has_splice_antiquot = node.children.iter().any(Syntax::is_antiquot_splice);

            if has_splice_antiquot {
                // Need to expand splice antiquotations in children
                let mut new_children = Vec::new();
                for child in &node.children {
                    if child.is_antiquot_splice() {
                        // Splice antiquotation: expand the list binding
                        if let Some(antiquot) = Antiquotation::from_syntax(child) {
                            if let Some(replacement) = bindings.get(&antiquot.name) {
                                // If it's a splice_list, expand its children
                                if let Some(kind) = replacement.kind() {
                                    if kind.name_str() == "splice_list" {
                                        for splice_child in replacement.children() {
                                            new_children
                                                .push(substitute_recursive(splice_child, bindings));
                                        }
                                        continue;
                                    }
                                }
                                // Otherwise, treat as single element
                                new_children.push(replacement.clone());
                                continue;
                            }
                        }
                        // No binding, leave as-is
                        new_children.push(child.clone());
                    } else {
                        // Regular child, recurse
                        new_children.push(substitute_recursive(child, bindings));
                    }
                }
                Syntax::node(node.kind.clone(), new_children)
            } else {
                // No splices, just recurse into children
                let new_children: Vec<Syntax> = node
                    .children
                    .iter()
                    .map(|c| substitute_recursive(c, bindings))
                    .collect();
                Syntax::node(node.kind.clone(), new_children)
            }
        }
        // Other syntax types pass through unchanged
        _ => syntax.clone(),
    }
}

/// Parse a syntax quotation from a string
///
/// This is a simplified parser for quotations. The full implementation
/// would integrate with the main parser.
pub fn parse_quotation(input: &str) -> Result<SyntaxQuote, QuotationError> {
    let trimmed = input.trim();

    // Check for quotation syntax: `(...)
    if !trimmed.starts_with('`') {
        return Err(QuotationError::NotAQuotation);
    }

    let rest = &trimmed[1..];

    // Determine category and parse content
    let (category, content) = if rest.starts_with('(') && rest.ends_with(')') {
        // Term quotation: `(term)
        (SyntaxKind::term(), &rest[1..rest.len() - 1])
    } else if rest.starts_with('[') && rest.ends_with(']') {
        // Tactic quotation: `[tactic]
        (SyntaxKind::tactic(), &rest[1..rest.len() - 1])
    } else if rest.starts_with('{') && rest.ends_with('}') {
        // Command quotation: `{command}
        (SyntaxKind::command(), &rest[1..rest.len() - 1])
    } else {
        // Simple identifier quotation
        (SyntaxKind::ident_kind(), rest)
    };

    let syntax = parse_quoted_content(content)?;

    Ok(SyntaxQuote::new(syntax, category))
}

/// Parse the content inside a quotation
fn parse_quoted_content(content: &str) -> Result<Syntax, QuotationError> {
    let trimmed = content.trim();

    if trimmed.is_empty() {
        return Ok(Syntax::missing());
    }

    // Check for antiquotation
    if let Some(rest) = trimmed.strip_prefix('$') {
        return parse_antiquotation(rest);
    }

    // Check for nested parentheses (application)
    if trimmed.contains(' ') && !trimmed.starts_with('"') {
        let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
        if parts.len() == 2 {
            let func = parse_quoted_content(parts[0])?;
            let args = parse_args(parts[1])?;
            return Ok(Syntax::mk_app(func, args));
        }
    }

    // Check for string literal
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() > 1 {
        return Ok(Syntax::mk_str(&trimmed[1..trimmed.len() - 1]));
    }

    // Check for numeric literal
    if let Ok(n) = trimmed.parse::<u64>() {
        return Ok(Syntax::mk_num(n));
    }

    // Treat as identifier
    Ok(Syntax::ident(trimmed))
}

/// Parse an antiquotation after the $
fn parse_antiquotation(content: &str) -> Result<Syntax, QuotationError> {
    let trimmed = content.trim();

    // Check for splice: $[name]* or $[name:category]*
    if trimmed.starts_with('[') {
        if let Some(end) = trimmed.find("]*") {
            let inner = &trimmed[1..end];
            // Check for type annotation within splice
            if let Some(colon_pos) = inner.find(':') {
                let name = &inner[..colon_pos];
                let category = &inner[colon_pos + 1..];
                return Ok(Syntax::mk_antiquot_splice_typed(name, category));
            }
            return Ok(Syntax::mk_antiquot_splice(inner));
        }
        return Err(QuotationError::MalformedAntiquotation);
    }

    // Check for parenthesized: $(expr) or $(expr:category)
    if trimmed.starts_with('(') {
        if let Some(end) = find_matching_paren(trimmed) {
            let inner = &trimmed[1..end];
            // Check for type annotation
            if let Some(colon_pos) = inner.rfind(':') {
                // Make sure the colon is at top level (not inside nested parens)
                let before_colon = &inner[..colon_pos];
                let depth: i32 = before_colon.chars().fold(0, |d, c| match c {
                    '(' | '[' | '{' => d + 1,
                    ')' | ']' | '}' => d - 1,
                    _ => d,
                });
                if depth == 0 {
                    let name = before_colon.trim();
                    let category = inner[colon_pos + 1..].trim();
                    return Ok(Syntax::mk_antiquot_typed(name, category));
                }
            }
            return Ok(Syntax::mk_antiquot(inner));
        }
        return Err(QuotationError::MalformedAntiquotation);
    }

    // Simple identifier antiquotation: $name or $name:category
    let name_end = trimmed
        .find(|c: char| !c.is_alphanumeric() && c != '_' && c != ':')
        .unwrap_or(trimmed.len());
    let name_and_maybe_type = &trimmed[..name_end];

    // Check for type annotation: $name:category
    if let Some(colon_pos) = name_and_maybe_type.find(':') {
        let name = &name_and_maybe_type[..colon_pos];
        let category = &name_and_maybe_type[colon_pos + 1..];
        if name.is_empty() || category.is_empty() {
            return Err(QuotationError::MalformedAntiquotation);
        }
        return Ok(Syntax::mk_antiquot_typed(name, category));
    }

    if name_and_maybe_type.is_empty() {
        return Err(QuotationError::MalformedAntiquotation);
    }

    Ok(Syntax::mk_antiquot(name_and_maybe_type))
}

/// Parse space-separated arguments
fn parse_args(content: &str) -> Result<Vec<Syntax>, QuotationError> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in content.chars() {
        match ch {
            '(' | '[' | '{' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                current.push(ch);
            }
            ' ' if depth == 0 => {
                if !current.is_empty() {
                    args.push(parse_quoted_content(&current)?);
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }

    if !current.is_empty() {
        args.push(parse_quoted_content(&current)?);
    }

    Ok(args)
}

/// Find the matching closing parenthesis
fn find_matching_paren(s: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Quotation parsing error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuotationError {
    NotAQuotation,
    MalformedAntiquotation,
    UnbalancedDelimiters,
    ParseError(String),
}

impl std::fmt::Display for QuotationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuotationError::NotAQuotation => write!(f, "not a quotation"),
            QuotationError::MalformedAntiquotation => write!(f, "malformed antiquotation"),
            QuotationError::UnbalancedDelimiters => write!(f, "unbalanced delimiters"),
            QuotationError::ParseError(s) => write!(f, "parse error: {s}"),
        }
    }
}

impl std::error::Error for QuotationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_quotation() {
        let quote = parse_quotation("`(foo)").unwrap();
        assert_eq!(quote.category, SyntaxKind::term());
        assert_eq!(quote.syntax.as_ident(), Some("foo"));
    }

    #[test]
    fn test_parse_application_quotation() {
        let quote = parse_quotation("`(f x)").unwrap();
        assert!(quote.syntax.is_node());
        assert_eq!(quote.syntax.children().len(), 2);
    }

    #[test]
    fn test_parse_antiquotation() {
        let quote = parse_quotation("`($x)").unwrap();
        assert!(quote.syntax.is_antiquot());
        let names = quote.antiquot_names();
        assert_eq!(names, vec!["x"]);
    }

    #[test]
    fn test_parse_numeric_literal() {
        let quote = parse_quotation("`(42)").unwrap();
        assert!(quote.syntax.is_node());
        assert_eq!(quote.syntax.kind(), Some(&SyntaxKind::num()));
    }

    #[test]
    fn test_antiquotation_from_syntax() {
        let syntax = Syntax::mk_antiquot("x");
        let antiquot = Antiquotation::from_syntax(&syntax).unwrap();
        assert_eq!(antiquot.name, "x");
        assert!(!antiquot.is_splice);
    }

    #[test]
    fn test_substitute_antiquots() {
        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), Syntax::ident("replaced"));

        let quote = SyntaxQuote::term(Syntax::mk_app(
            Syntax::ident("f"),
            vec![Syntax::mk_antiquot("x")],
        ));

        let result = quote.substitute(&bindings);
        let pretty = result.pretty();
        assert!(pretty.contains("replaced"));
    }

    #[test]
    fn test_has_antiquots() {
        let quote1 = SyntaxQuote::term(Syntax::ident("foo"));
        assert!(!quote1.has_antiquots());

        let quote2 = SyntaxQuote::term(Syntax::mk_antiquot("x"));
        assert!(quote2.has_antiquots());
    }

    #[test]
    fn test_parse_splice_antiquotation() {
        let syntax = parse_antiquotation("[items]*").unwrap();
        assert!(syntax.is_antiquot());
        let antiquot = Antiquotation::from_syntax(&syntax).unwrap();
        assert!(antiquot.is_splice);
        assert_eq!(antiquot.name, "items");
    }

    #[test]
    fn test_quotation_error_display() {
        assert_eq!(
            format!("{}", QuotationError::NotAQuotation),
            "not a quotation"
        );
        assert_eq!(
            format!("{}", QuotationError::MalformedAntiquotation),
            "malformed antiquotation"
        );
    }

    #[test]
    fn test_substitute_splice_antiquots() {
        // Test that splice bindings are expanded in replacement
        // Pattern: `(f $[args]*)` -> `(g $[args]*)`
        // With args = [a, b], result should be (g a b)
        let mut bindings = HashMap::new();

        // Create a splice_list binding
        let splice_list = Syntax::node(
            SyntaxKind::app("splice_list"),
            vec![Syntax::ident("a"), Syntax::ident("b")],
        );
        bindings.insert("args".to_string(), splice_list);

        // Create replacement template with splice antiquotation
        let template = SyntaxQuote::term(Syntax::node(
            SyntaxKind::app_kind(),
            vec![Syntax::ident("g"), Syntax::mk_antiquot_splice("args")],
        ));

        let result = template.substitute(&bindings);

        // Result should be (g a b)
        assert!(result.is_node());
        let children = result.children();
        assert_eq!(children.len(), 3); // g, a, b
        assert_eq!(children[0].as_ident(), Some("g"));
        assert_eq!(children[1].as_ident(), Some("a"));
        assert_eq!(children[2].as_ident(), Some("b"));
    }

    #[test]
    fn test_substitute_empty_splice() {
        // Test splice with empty list
        // Pattern: `(f $[args]*)` with args = []
        // Result should be (f)
        let mut bindings = HashMap::new();

        // Create an empty splice_list
        let splice_list = Syntax::node(SyntaxKind::app("splice_list"), vec![]);
        bindings.insert("args".to_string(), splice_list);

        // Create replacement template
        let template = SyntaxQuote::term(Syntax::node(
            SyntaxKind::app_kind(),
            vec![Syntax::ident("f"), Syntax::mk_antiquot_splice("args")],
        ));

        let result = template.substitute(&bindings);

        // Result should be (f)
        assert!(result.is_node());
        let children = result.children();
        assert_eq!(children.len(), 1); // just f
        assert_eq!(children[0].as_ident(), Some("f"));
    }

    #[test]
    fn test_parse_typed_antiquotation_simple() {
        // $x:term
        let syntax = parse_antiquotation("x:term").unwrap();
        assert!(syntax.is_antiquot());
        assert!(syntax.is_antiquot_typed());
        let antiquot = Antiquotation::from_syntax(&syntax).unwrap();
        assert_eq!(antiquot.name, "x");
        assert!(!antiquot.is_splice);
        assert_eq!(antiquot.category, Some("term".to_string()));
    }

    #[test]
    fn test_parse_typed_antiquotation_tactic() {
        // $t:tactic
        let syntax = parse_antiquotation("t:tactic").unwrap();
        assert!(syntax.is_antiquot_typed());
        let antiquot = Antiquotation::from_syntax(&syntax).unwrap();
        assert_eq!(antiquot.name, "t");
        assert_eq!(antiquot.category, Some("tactic".to_string()));
    }

    #[test]
    fn test_parse_typed_splice_antiquotation() {
        // $[args:term]*
        let syntax = parse_antiquotation("[args:term]*").unwrap();
        assert!(syntax.is_antiquot());
        assert!(syntax.is_antiquot_splice());
        assert!(syntax.is_antiquot_typed());
        let antiquot = Antiquotation::from_syntax(&syntax).unwrap();
        assert_eq!(antiquot.name, "args");
        assert!(antiquot.is_splice);
        assert_eq!(antiquot.category, Some("term".to_string()));
    }

    #[test]
    fn test_parse_parenthesized_typed_antiquotation() {
        // $(expr:term)
        let syntax = parse_antiquotation("(foo:term)").unwrap();
        assert!(syntax.is_antiquot_typed());
        let antiquot = Antiquotation::from_syntax(&syntax).unwrap();
        assert_eq!(antiquot.name, "foo");
        assert_eq!(antiquot.category, Some("term".to_string()));
    }

    #[test]
    fn test_antiquotation_typed_constructor() {
        let antiquot = Antiquotation::typed("x", "term");
        assert_eq!(antiquot.name, "x");
        assert!(!antiquot.is_splice);
        assert!(antiquot.is_typed());
        assert_eq!(antiquot.category, Some("term".to_string()));
    }

    #[test]
    fn test_antiquotation_typed_splice_constructor() {
        let antiquot = Antiquotation::typed_splice("args", "command");
        assert_eq!(antiquot.name, "args");
        assert!(antiquot.is_splice);
        assert!(antiquot.is_typed());
        assert_eq!(antiquot.category, Some("command".to_string()));
    }

    #[test]
    fn test_mk_antiquot_typed() {
        let syntax = Syntax::mk_antiquot_typed("x", "term");
        assert!(syntax.is_antiquot());
        assert!(syntax.is_antiquot_typed());
        let children = syntax.children();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].as_ident(), Some("x"));
        assert_eq!(children[1].as_ident(), Some("term"));
    }

    #[test]
    fn test_mk_antiquot_splice_typed() {
        let syntax = Syntax::mk_antiquot_splice_typed("items", "tactic");
        assert!(syntax.is_antiquot());
        assert!(syntax.is_antiquot_splice());
        assert!(syntax.is_antiquot_typed());
        let children = syntax.children();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].as_ident(), Some("items"));
        assert_eq!(children[1].as_ident(), Some("tactic"));
    }

    #[test]
    fn test_antiquotation_from_typed_syntax() {
        let syntax = Syntax::mk_antiquot_typed("expr", "term");
        let antiquot = Antiquotation::from_syntax(&syntax).unwrap();
        assert_eq!(antiquot.name, "expr");
        assert!(!antiquot.is_splice);
        assert_eq!(antiquot.category, Some("term".to_string()));
    }

    #[test]
    fn test_antiquotation_from_typed_splice_syntax() {
        let syntax = Syntax::mk_antiquot_splice_typed("stmts", "command");
        let antiquot = Antiquotation::from_syntax(&syntax).unwrap();
        assert_eq!(antiquot.name, "stmts");
        assert!(antiquot.is_splice);
        assert_eq!(antiquot.category, Some("command".to_string()));
    }

    #[test]
    fn test_untyped_antiquotation_has_no_category() {
        let syntax = Syntax::mk_antiquot("x");
        let antiquot = Antiquotation::from_syntax(&syntax).unwrap();
        assert!(!antiquot.is_typed());
        assert_eq!(antiquot.category, None);
    }
}
