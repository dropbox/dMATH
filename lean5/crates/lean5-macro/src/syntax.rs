//! Core syntax types for the macro system
//!
//! The `Syntax` type is a general AST representation that can represent any
//! syntactic construct. It's the foundation for syntax quotations and macros.
//!
//! Based on Lean 4's `Syntax` type with the following constructors:
//! - `node`: A syntax node with a kind and children
//! - `atom`: A lexical token (literal)
//! - `ident`: An identifier with optional resolution info
//! - `missing`: A placeholder for missing/erroneous syntax

use lean5_kernel::name::Name;
use std::sync::Arc;

/// Source location information for syntax
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceInfo {
    /// Starting position in source text
    pub start: usize,
    /// Ending position in source text
    pub end: usize,
    /// Optional file name
    pub file: Option<Arc<str>>,
}

impl SourceInfo {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            file: None,
        }
    }

    pub fn with_file(start: usize, end: usize, file: impl Into<Arc<str>>) -> Self {
        Self {
            start,
            end,
            file: Some(file.into()),
        }
    }

    pub fn dummy() -> Self {
        Self::default()
    }

    /// Merge two source infos to cover the combined span
    #[must_use]
    pub fn merge(&self, other: &SourceInfo) -> SourceInfo {
        SourceInfo {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
            file: self.file.clone().or_else(|| other.file.clone()),
        }
    }
}

/// Kind identifier for syntax nodes
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SyntaxKind {
    name: Name,
    /// Cached string representation for fast comparisons
    name_str: String,
}

impl SyntaxKind {
    /// Create a syntax kind from a name
    pub fn from_name(name: Name) -> Self {
        let name_str = name.to_string();
        Self { name, name_str }
    }

    /// Create a syntax kind from a string
    pub fn app(s: &str) -> Self {
        let name = Name::from_string(s);
        Self {
            name,
            name_str: s.to_string(),
        }
    }

    /// Get the name of this syntax kind
    pub fn name(&self) -> &Name {
        &self.name
    }

    /// Get the string representation of this syntax kind
    pub fn name_str(&self) -> &str {
        &self.name_str
    }

    /// Check if this is a quotation kind (ends with "quot" or "Quot")
    pub fn is_quotation(&self) -> bool {
        self.name_str.ends_with("quot") || self.name_str.ends_with("Quot")
    }

    /// Check if this is an antiquotation kind
    pub fn is_antiquotation(&self) -> bool {
        self.name_str.contains("antiquot")
    }

    // Common syntax kinds
    pub fn term() -> Self {
        Self::app("term")
    }
    pub fn command() -> Self {
        Self::app("command")
    }
    pub fn tactic() -> Self {
        Self::app("tactic")
    }
    pub fn ident_kind() -> Self {
        Self::app("ident")
    }
    pub fn num() -> Self {
        Self::app("num")
    }
    pub fn str() -> Self {
        Self::app("str")
    }
    pub fn paren() -> Self {
        Self::app("paren")
    }
    pub fn app_kind() -> Self {
        Self::app("app")
    }
    pub fn fun() -> Self {
        Self::app("fun")
    }
    pub fn forall_kind() -> Self {
        Self::app("forall")
    }
    pub fn arrow() -> Self {
        Self::app("arrow")
    }
    pub fn let_kind() -> Self {
        Self::app("let")
    }
    pub fn hole() -> Self {
        Self::app("hole")
    }
    pub fn antiquot() -> Self {
        Self::app("antiquot")
    }
    pub fn antiquot_typed() -> Self {
        Self::app("antiquot_typed")
    }
    pub fn antiquot_splice() -> Self {
        Self::app("antiquot_splice")
    }
    pub fn antiquot_splice_typed() -> Self {
        Self::app("antiquot_splice_typed")
    }
    pub fn missing() -> Self {
        Self::app("missing")
    }
}

impl std::fmt::Display for SyntaxKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name_str)
    }
}

/// A syntax node with kind and children
#[derive(Debug, Clone)]
pub struct SyntaxNode {
    pub info: SourceInfo,
    pub kind: SyntaxKind,
    pub children: Vec<Syntax>,
}

/// General syntax representation
///
/// This type can represent any syntactic construct in Lean.
/// It's designed for manipulation in macros and quotations.
#[derive(Debug, Clone)]
pub enum Syntax {
    /// A syntax node with a kind and children
    Node(Box<SyntaxNode>),

    /// A lexical atom (literal token)
    Atom(SourceInfo, String),

    /// An identifier
    Ident(SourceInfo, String),

    /// Missing/erroneous syntax (placeholder)
    Missing(SourceInfo),
}

impl Syntax {
    // Constructors

    /// Create a syntax node
    pub fn node(kind: SyntaxKind, children: Vec<Syntax>) -> Self {
        Syntax::Node(Box::new(SyntaxNode {
            info: SourceInfo::dummy(),
            kind,
            children,
        }))
    }

    /// Create a syntax node with source info
    pub fn node_with_info(info: SourceInfo, kind: SyntaxKind, children: Vec<Syntax>) -> Self {
        Syntax::Node(Box::new(SyntaxNode {
            info,
            kind,
            children,
        }))
    }

    /// Create an atom (literal)
    pub fn atom(value: &str) -> Self {
        Syntax::Atom(SourceInfo::dummy(), value.to_string())
    }

    /// Create an atom with source info
    pub fn atom_with_info(info: SourceInfo, value: &str) -> Self {
        Syntax::Atom(info, value.to_string())
    }

    /// Create an identifier
    pub fn ident(name: &str) -> Self {
        Syntax::Ident(SourceInfo::dummy(), name.to_string())
    }

    /// Create an identifier with source info
    pub fn ident_with_info(info: SourceInfo, name: &str) -> Self {
        Syntax::Ident(info, name.to_string())
    }

    /// Create missing syntax
    pub fn missing() -> Self {
        Syntax::Missing(SourceInfo::dummy())
    }

    // Predicates

    /// Check if this is a node
    pub fn is_node(&self) -> bool {
        matches!(self, Syntax::Node(_))
    }

    /// Check if this is an atom
    pub fn is_atom(&self) -> bool {
        matches!(self, Syntax::Atom(_, _))
    }

    /// Check if this is an identifier
    pub fn is_ident(&self) -> bool {
        matches!(self, Syntax::Ident(_, _))
    }

    /// Check if this is missing syntax
    pub fn is_missing(&self) -> bool {
        matches!(self, Syntax::Missing(_))
    }

    /// Check if this is an antiquotation node
    pub fn is_antiquot(&self) -> bool {
        match self {
            Syntax::Node(node) => node.kind.is_antiquotation(),
            _ => false,
        }
    }

    /// Check if this is a splice antiquotation node (`$[x]*` or `$[x:cat]*`)
    pub fn is_antiquot_splice(&self) -> bool {
        match self {
            Syntax::Node(node) => {
                let kind_str = node.kind.name_str();
                kind_str == "antiquot_splice" || kind_str == "antiquot_splice_typed"
            }
            _ => false,
        }
    }

    /// Check if this is a typed antiquotation node (`$x:term` or `$[x:term]*`)
    pub fn is_antiquot_typed(&self) -> bool {
        match self {
            Syntax::Node(node) => {
                let kind_str = node.kind.name_str();
                kind_str == "antiquot_typed" || kind_str == "antiquot_splice_typed"
            }
            _ => false,
        }
    }

    /// Check if this is a simple (non-splice) antiquotation
    pub fn is_simple_antiquot(&self) -> bool {
        match self {
            Syntax::Node(node) => {
                let kind_str = node.kind.name_str();
                node.kind.is_antiquotation()
                    && kind_str != "antiquot_splice"
                    && kind_str != "antiquot_splice_typed"
            }
            _ => false,
        }
    }

    /// Check if this is a quotation node
    pub fn is_quotation(&self) -> bool {
        match self {
            Syntax::Node(node) => node.kind.is_quotation(),
            _ => false,
        }
    }

    // Accessors

    /// Get the source info
    pub fn source_info(&self) -> &SourceInfo {
        match self {
            Syntax::Node(node) => &node.info,
            Syntax::Atom(info, _) | Syntax::Ident(info, _) | Syntax::Missing(info) => info,
        }
    }

    /// Get the syntax kind if this is a node
    pub fn kind(&self) -> Option<&SyntaxKind> {
        match self {
            Syntax::Node(node) => Some(&node.kind),
            _ => None,
        }
    }

    /// Get children if this is a node
    pub fn children(&self) -> &[Syntax] {
        match self {
            Syntax::Node(node) => &node.children,
            _ => &[],
        }
    }

    /// Get mutable children if this is a node
    pub fn children_mut(&mut self) -> Option<&mut Vec<Syntax>> {
        match self {
            Syntax::Node(node) => Some(&mut node.children),
            _ => None,
        }
    }

    /// Get the atom value if this is an atom
    pub fn as_atom(&self) -> Option<&str> {
        match self {
            Syntax::Atom(_, v) => Some(v),
            _ => None,
        }
    }

    /// Get the identifier name if this is an ident
    pub fn as_ident(&self) -> Option<&str> {
        match self {
            Syntax::Ident(_, n) => Some(n),
            _ => None,
        }
    }

    /// Get the first child
    pub fn first_child(&self) -> Option<&Syntax> {
        self.children().first()
    }

    /// Get the last child
    pub fn last_child(&self) -> Option<&Syntax> {
        self.children().last()
    }

    /// Get child at index
    pub fn child(&self, index: usize) -> Option<&Syntax> {
        self.children().get(index)
    }

    // Transformations

    /// Map a function over all nodes in the syntax tree
    #[must_use]
    pub fn map<F>(&self, f: &F) -> Syntax
    where
        F: Fn(&Syntax) -> Syntax,
    {
        let mapped = f(self);
        match &mapped {
            Syntax::Node(node) => {
                let new_children: Vec<_> = node.children.iter().map(|c| c.map(f)).collect();
                Syntax::node_with_info(node.info.clone(), node.kind.clone(), new_children)
            }
            _ => mapped,
        }
    }

    /// Replace syntax matching a predicate
    #[must_use]
    pub fn replace<P, R>(&self, pred: &P, replace: &R) -> Syntax
    where
        P: Fn(&Syntax) -> bool,
        R: Fn(&Syntax) -> Syntax,
    {
        if pred(self) {
            replace(self)
        } else {
            match self {
                Syntax::Node(node) => {
                    let new_children: Vec<_> = node
                        .children
                        .iter()
                        .map(|c| c.replace(pred, replace))
                        .collect();
                    Syntax::node_with_info(node.info.clone(), node.kind.clone(), new_children)
                }
                _ => self.clone(),
            }
        }
    }

    /// Collect all antiquotations in this syntax tree
    pub fn collect_antiquots(&self) -> Vec<&Syntax> {
        let mut result = Vec::new();
        self.collect_antiquots_into(&mut result);
        result
    }

    fn collect_antiquots_into<'a>(&'a self, result: &mut Vec<&'a Syntax>) {
        if self.is_antiquot() {
            result.push(self);
        }
        for child in self.children() {
            child.collect_antiquots_into(result);
        }
    }

    /// Pretty-print the syntax for debugging
    pub fn pretty(&self) -> String {
        match self {
            Syntax::Node(node) => {
                let children_str: Vec<_> = node.children.iter().map(Syntax::pretty).collect();
                format!(
                    "({}{}{})",
                    node.kind,
                    if children_str.is_empty() { "" } else { " " },
                    children_str.join(" ")
                )
            }
            Syntax::Atom(_, v) => format!("\"{v}\""),
            Syntax::Ident(_, n) => n.clone(),
            Syntax::Missing(_) => "_".to_string(),
        }
    }
}

impl std::fmt::Display for Syntax {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.pretty())
    }
}

// Builder helpers for common syntax patterns

impl Syntax {
    /// Create an application syntax node
    pub fn mk_app(func: Syntax, args: Vec<Syntax>) -> Self {
        let mut children = vec![func];
        children.extend(args);
        Syntax::node(SyntaxKind::app_kind(), children)
    }

    /// Create a lambda syntax node
    pub fn mk_lambda(binders: Vec<Syntax>, body: Syntax) -> Self {
        let mut children = binders;
        children.push(body);
        Syntax::node(SyntaxKind::fun(), children)
    }

    /// Create a forall syntax node
    pub fn mk_forall(binders: Vec<Syntax>, body: Syntax) -> Self {
        let mut children = binders;
        children.push(body);
        Syntax::node(SyntaxKind::forall_kind(), children)
    }

    /// Create an arrow type syntax node
    pub fn mk_arrow(from: Syntax, to: Syntax) -> Self {
        Syntax::node(SyntaxKind::arrow(), vec![from, to])
    }

    /// Create a let syntax node
    pub fn mk_let(name: Syntax, ty: Option<Syntax>, val: Syntax, body: Syntax) -> Self {
        let mut children = vec![name];
        if let Some(t) = ty {
            children.push(t);
        }
        children.push(val);
        children.push(body);
        Syntax::node(SyntaxKind::let_kind(), children)
    }

    /// Create a hole syntax node
    pub fn mk_hole() -> Self {
        Syntax::node(SyntaxKind::hole(), vec![])
    }

    /// Create a numeric literal syntax
    pub fn mk_num(n: u64) -> Self {
        Syntax::node(SyntaxKind::num(), vec![Syntax::atom(&n.to_string())])
    }

    /// Create a string literal syntax
    pub fn mk_str(s: &str) -> Self {
        Syntax::node(SyntaxKind::str(), vec![Syntax::atom(s)])
    }

    /// Create a parenthesized syntax
    pub fn mk_paren(inner: Syntax) -> Self {
        Syntax::node(SyntaxKind::paren(), vec![inner])
    }

    /// Create an antiquotation node for `$name`
    pub fn mk_antiquot(name: &str) -> Self {
        Syntax::node(SyntaxKind::antiquot(), vec![Syntax::ident(name)])
    }

    /// Create a typed antiquotation node for `$name:category`
    pub fn mk_antiquot_typed(name: &str, category: &str) -> Self {
        Syntax::node(
            SyntaxKind::antiquot_typed(),
            vec![Syntax::ident(name), Syntax::ident(category)],
        )
    }

    /// Create a splice antiquotation for `$[items]*`
    pub fn mk_antiquot_splice(name: &str) -> Self {
        Syntax::node(SyntaxKind::antiquot_splice(), vec![Syntax::ident(name)])
    }

    /// Create a typed splice antiquotation for `$[items:category]*`
    pub fn mk_antiquot_splice_typed(name: &str, category: &str) -> Self {
        Syntax::node(
            SyntaxKind::antiquot_splice_typed(),
            vec![Syntax::ident(name), Syntax::ident(category)],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntax_kind_app() {
        let kind = SyntaxKind::app("Foo.Bar");
        assert_eq!(kind.name_str(), "Foo.Bar");
    }

    #[test]
    fn test_syntax_node_children() {
        let node = Syntax::node(
            SyntaxKind::app_kind(),
            vec![Syntax::ident("f"), Syntax::ident("x"), Syntax::ident("y")],
        );
        assert_eq!(node.children().len(), 3);
        assert_eq!(node.child(0).unwrap().as_ident(), Some("f"));
        assert_eq!(node.child(1).unwrap().as_ident(), Some("x"));
        assert_eq!(node.child(2).unwrap().as_ident(), Some("y"));
    }

    #[test]
    fn test_syntax_pretty_print() {
        let app = Syntax::mk_app(
            Syntax::ident("Nat.add"),
            vec![Syntax::mk_num(1), Syntax::mk_num(2)],
        );
        let pretty = app.pretty();
        assert!(pretty.contains("app"));
        assert!(pretty.contains("Nat.add"));
    }

    #[test]
    fn test_source_info_merge() {
        let s1 = SourceInfo::new(0, 10);
        let s2 = SourceInfo::new(5, 20);
        let merged = s1.merge(&s2);
        assert_eq!(merged.start, 0);
        assert_eq!(merged.end, 20);
    }

    #[test]
    fn test_syntax_kind_predicates() {
        assert!(SyntaxKind::app("termQuot").is_quotation());
        assert!(!SyntaxKind::app("term").is_quotation());
        assert!(SyntaxKind::app("antiquot").is_antiquotation());
        assert!(!SyntaxKind::app("app").is_antiquotation());
    }

    #[test]
    fn test_antiquot_collection() {
        let syntax = Syntax::mk_app(
            Syntax::ident("f"),
            vec![Syntax::mk_antiquot("x"), Syntax::mk_antiquot("y")],
        );
        let antiquots = syntax.collect_antiquots();
        assert_eq!(antiquots.len(), 2);
    }

    #[test]
    fn test_syntax_replace() {
        let syntax = Syntax::mk_app(Syntax::ident("f"), vec![Syntax::ident("x")]);
        let replaced = syntax.replace(&|s| s.as_ident() == Some("x"), &|_| Syntax::ident("y"));
        let pretty = replaced.pretty();
        assert!(pretty.contains('y'));
        assert!(!pretty.contains(" x"));
    }

    #[test]
    fn test_mk_arrow() {
        let arr = Syntax::mk_arrow(Syntax::ident("Nat"), Syntax::ident("Bool"));
        assert_eq!(arr.kind(), Some(&SyntaxKind::arrow()));
        assert_eq!(arr.children().len(), 2);
    }

    #[test]
    fn test_mk_lambda() {
        let lam = Syntax::mk_lambda(vec![Syntax::ident("x")], Syntax::ident("x"));
        assert_eq!(lam.kind(), Some(&SyntaxKind::fun()));
        assert_eq!(lam.children().len(), 2);
    }

    #[test]
    fn test_is_antiquot_typed() {
        let typed = Syntax::mk_antiquot_typed("x", "term");
        assert!(typed.is_antiquot());
        assert!(typed.is_antiquot_typed());
        assert!(!typed.is_antiquot_splice());

        let untyped = Syntax::mk_antiquot("x");
        assert!(untyped.is_antiquot());
        assert!(!untyped.is_antiquot_typed());
    }

    #[test]
    fn test_is_antiquot_splice_typed() {
        let typed_splice = Syntax::mk_antiquot_splice_typed("xs", "term");
        assert!(typed_splice.is_antiquot());
        assert!(typed_splice.is_antiquot_typed());
        assert!(typed_splice.is_antiquot_splice());

        let untyped_splice = Syntax::mk_antiquot_splice("xs");
        assert!(untyped_splice.is_antiquot());
        assert!(!untyped_splice.is_antiquot_typed());
        assert!(untyped_splice.is_antiquot_splice());
    }

    #[test]
    fn test_syntax_kind_antiquot_typed() {
        let kind = SyntaxKind::antiquot_typed();
        assert_eq!(kind.name_str(), "antiquot_typed");
        assert!(kind.is_antiquotation());
    }

    #[test]
    fn test_syntax_kind_antiquot_splice_typed() {
        let kind = SyntaxKind::antiquot_splice_typed();
        assert_eq!(kind.name_str(), "antiquot_splice_typed");
        assert!(kind.is_antiquotation());
    }

    #[test]
    fn test_collect_typed_antiquots() {
        let syntax = Syntax::mk_app(
            Syntax::ident("f"),
            vec![
                Syntax::mk_antiquot_typed("x", "term"),
                Syntax::mk_antiquot("y"),
                Syntax::mk_antiquot_splice_typed("zs", "tactic"),
            ],
        );
        let antiquots = syntax.collect_antiquots();
        assert_eq!(antiquots.len(), 3);
    }

    #[test]
    fn test_is_simple_antiquot_excludes_typed_splices() {
        let typed_splice = Syntax::mk_antiquot_splice_typed("xs", "term");
        assert!(!typed_splice.is_simple_antiquot());

        let untyped_splice = Syntax::mk_antiquot_splice("xs");
        assert!(!untyped_splice.is_simple_antiquot());

        let typed = Syntax::mk_antiquot_typed("x", "term");
        assert!(typed.is_simple_antiquot());

        let untyped = Syntax::mk_antiquot("x");
        assert!(untyped.is_simple_antiquot());
    }
}
