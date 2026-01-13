//! Macro registry and definitions
//!
//! The registry stores macro definitions and provides lookup during expansion.
//! Macros are keyed by the syntax kind they match.

use crate::quotation::SyntaxQuote;
use crate::syntax::{Syntax, SyntaxKind};
use std::collections::HashMap;
use std::sync::Arc;

/// A macro definition
#[derive(Debug, Clone)]
pub struct MacroDef {
    /// Name of the macro (for debugging)
    pub name: String,
    /// The syntax kind this macro matches
    pub kind: SyntaxKind,
    /// Pattern to match (may contain wildcards)
    pub pattern: Syntax,
    /// Replacement template (may contain antiquotations for captured values)
    pub replacement: SyntaxQuote,
    /// Priority (higher = tried first)
    pub priority: i32,
    /// Documentation string
    pub doc: Option<String>,
}

impl MacroDef {
    /// Create a new macro definition
    pub fn new(
        name: impl Into<String>,
        kind: SyntaxKind,
        pattern: Syntax,
        replacement: SyntaxQuote,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            pattern,
            replacement,
            priority: 0,
            doc: None,
        }
    }

    /// Set priority
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set documentation
    #[must_use]
    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = Some(doc.into());
        self
    }

    /// Try to match this macro against syntax, returning bindings if successful
    pub fn try_match(&self, syntax: &Syntax) -> Option<HashMap<String, Syntax>> {
        match_pattern(&self.pattern, syntax)
    }

    /// Apply this macro to matched syntax
    pub fn apply(&self, bindings: &HashMap<String, Syntax>) -> Syntax {
        self.replacement.substitute(bindings)
    }
}

/// Match a pattern against syntax, returning captured bindings
fn match_pattern(pattern: &Syntax, syntax: &Syntax) -> Option<HashMap<String, Syntax>> {
    let mut bindings = HashMap::new();
    if match_pattern_inner(pattern, syntax, &mut bindings) {
        Some(bindings)
    } else {
        None
    }
}

fn match_pattern_inner(
    pattern: &Syntax,
    syntax: &Syntax,
    bindings: &mut HashMap<String, Syntax>,
) -> bool {
    // Simple antiquotation in pattern captures the corresponding syntax
    if pattern.is_simple_antiquot() {
        if let Some(name) = pattern.children().first().and_then(|c| c.as_ident()) {
            bindings.insert(name.to_string(), syntax.clone());
            return true;
        }
        return false;
    }

    // Splice antiquotation should be handled at the node level (see match_children_with_splice)
    // If we encounter it here directly, it matches nothing
    if pattern.is_antiquot_splice() {
        return false;
    }

    // Missing pattern matches anything
    if pattern.is_missing() {
        return true;
    }

    match (pattern, syntax) {
        (Syntax::Ident(_, p_name), Syntax::Ident(_, s_name)) => p_name == s_name,

        (Syntax::Atom(_, p_val), Syntax::Atom(_, s_val)) => p_val == s_val,

        (Syntax::Node(p_node), Syntax::Node(s_node)) => {
            // Kinds must match
            if p_node.kind != s_node.kind {
                return false;
            }
            // Check if pattern has splice antiquotations
            let has_splice = p_node.children.iter().any(Syntax::is_antiquot_splice);
            if has_splice {
                // Use splice-aware matching
                match_children_with_splice(&p_node.children, &s_node.children, bindings)
            } else {
                // Children must match exactly
                if p_node.children.len() != s_node.children.len() {
                    return false;
                }
                for (p_child, s_child) in p_node.children.iter().zip(s_node.children.iter()) {
                    if !match_pattern_inner(p_child, s_child, bindings) {
                        return false;
                    }
                }
                true
            }
        }

        (Syntax::Missing(_), _) => true,

        _ => false,
    }
}

/// Match children with splice antiquotation support
/// A splice `$[x]*` matches zero or more consecutive children and binds them as a list
fn match_children_with_splice(
    pattern_children: &[Syntax],
    syntax_children: &[Syntax],
    bindings: &mut HashMap<String, Syntax>,
) -> bool {
    let mut p_idx = 0;
    let mut s_idx = 0;

    while p_idx < pattern_children.len() {
        let p_child = &pattern_children[p_idx];

        if p_child.is_antiquot_splice() {
            // Splice antiquotation: $[name]*
            // This greedily matches zero or more children
            if let Some(name) = p_child.children().first().and_then(|c| c.as_ident()) {
                // Determine how many syntax children to consume
                // Strategy: consume until next pattern element matches, or end of syntax
                let next_pattern = pattern_children.get(p_idx + 1);
                let mut end_idx = s_idx;

                if let Some(next_p) = next_pattern {
                    // Find where next pattern element matches
                    let mut found = false;
                    for i in s_idx..=syntax_children.len() {
                        if i == syntax_children.len() {
                            // Check if next pattern can match empty or is also a splice
                            if next_p.is_antiquot_splice() || next_p.is_missing() {
                                end_idx = syntax_children.len();
                                found = true;
                                break;
                            }
                        } else {
                            // Try matching next pattern at this position
                            let mut test_bindings = HashMap::new();
                            if match_pattern_inner(next_p, &syntax_children[i], &mut test_bindings)
                            {
                                end_idx = i;
                                found = true;
                                break;
                            }
                        }
                    }
                    if !found {
                        // Try consuming all remaining - the pattern may still work
                        end_idx = syntax_children.len();
                    }
                } else {
                    // No more patterns, consume all remaining syntax
                    end_idx = syntax_children.len();
                }

                // Collect matched children into a list node
                let matched: Vec<Syntax> = syntax_children[s_idx..end_idx].to_vec();
                let splice_result = Syntax::node(SyntaxKind::app("splice_list"), matched);
                bindings.insert(name.to_string(), splice_result);

                s_idx = end_idx;
            }
            p_idx += 1;
        } else {
            // Regular pattern element
            if s_idx >= syntax_children.len() {
                return false;
            }
            if !match_pattern_inner(p_child, &syntax_children[s_idx], bindings) {
                return false;
            }
            p_idx += 1;
            s_idx += 1;
        }
    }

    // All pattern elements consumed; check if all syntax children consumed
    // (unless there was a trailing splice that consumed them)
    s_idx == syntax_children.len()
}

/// Registry for macro definitions
#[derive(Debug, Clone, Default)]
pub struct MacroRegistry {
    /// Macros indexed by their target syntax kind
    macros: HashMap<SyntaxKind, Vec<Arc<MacroDef>>>,
    /// All macros by name (for lookup)
    by_name: HashMap<String, Arc<MacroDef>>,
}

impl MacroRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a macro definition
    pub fn register(&mut self, def: MacroDef) {
        let def = Arc::new(def);
        self.by_name.insert(def.name.clone(), Arc::clone(&def));

        let macros = self.macros.entry(def.kind.clone()).or_default();
        macros.push(def);

        // Keep macros sorted by priority (highest first)
        macros.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Look up a macro by name
    pub fn get_by_name(&self, name: &str) -> Option<&Arc<MacroDef>> {
        self.by_name.get(name)
    }

    /// Get all macros that could match a given syntax kind
    pub fn get_by_kind(&self, kind: &SyntaxKind) -> &[Arc<MacroDef>] {
        self.macros.get(kind).map_or(&[], |v| v.as_slice())
    }

    /// Find and apply the first matching macro
    pub fn try_expand(&self, syntax: &Syntax) -> Option<Syntax> {
        let kind = syntax.kind()?;
        let macros = self.get_by_kind(kind);

        for macro_def in macros {
            if let Some(bindings) = macro_def.try_match(syntax) {
                return Some(macro_def.apply(&bindings));
            }
        }

        None
    }

    /// Get all registered macro names
    pub fn macro_names(&self) -> Vec<&str> {
        self.by_name.keys().map(String::as_str).collect()
    }

    /// Get count of registered macros
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

/// Syntax category registration
#[derive(Debug, Clone)]
pub struct SyntaxCategory {
    /// Category name
    pub name: String,
    /// Parser kind
    pub kind: SyntaxKind,
    /// Description
    pub description: Option<String>,
}

impl SyntaxCategory {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            kind: SyntaxKind::app(&name),
            name,
            description: None,
        }
    }

    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// Registry for syntax categories
#[derive(Debug, Clone, Default)]
pub struct SyntaxCategoryRegistry {
    categories: HashMap<String, SyntaxCategory>,
}

impl SyntaxCategoryRegistry {
    pub fn new() -> Self {
        let mut registry = Self::default();
        // Register built-in categories
        registry.register(SyntaxCategory::new("term").with_description("Term expressions"));
        registry.register(SyntaxCategory::new("command").with_description("Top-level commands"));
        registry.register(SyntaxCategory::new("tactic").with_description("Proof tactics"));
        registry.register(SyntaxCategory::new("doElem").with_description("Do notation elements"));
        registry.register(SyntaxCategory::new("attr").with_description("Attributes"));
        registry
    }

    /// Register a new syntax category
    pub fn register(&mut self, category: SyntaxCategory) {
        self.categories.insert(category.name.clone(), category);
    }

    /// Look up a category
    pub fn get(&self, name: &str) -> Option<&SyntaxCategory> {
        self.categories.get(name)
    }

    /// Check if a category exists
    pub fn exists(&self, name: &str) -> bool {
        self.categories.contains_key(name)
    }

    /// Get all category names
    pub fn names(&self) -> Vec<&str> {
        self.categories.keys().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_def_creation() {
        let def = MacroDef::new(
            "test_macro",
            SyntaxKind::term(),
            Syntax::mk_antiquot("x"),
            SyntaxQuote::term(Syntax::ident("replaced")),
        );
        assert_eq!(def.name, "test_macro");
        assert_eq!(def.priority, 0);
    }

    #[test]
    fn test_pattern_matching_ident() {
        let pattern = Syntax::ident("foo");
        let syntax = Syntax::ident("foo");
        let bindings = match_pattern(&pattern, &syntax);
        assert!(bindings.is_some());
        assert!(bindings.unwrap().is_empty());

        let syntax2 = Syntax::ident("bar");
        assert!(match_pattern(&pattern, &syntax2).is_none());
    }

    #[test]
    fn test_pattern_matching_antiquot() {
        let pattern = Syntax::mk_antiquot("x");
        let syntax = Syntax::ident("anything");
        let bindings = match_pattern(&pattern, &syntax).unwrap();
        assert_eq!(bindings.len(), 1);
        assert!(bindings.contains_key("x"));
    }

    #[test]
    fn test_pattern_matching_node() {
        let pattern = Syntax::mk_app(Syntax::ident("f"), vec![Syntax::mk_antiquot("arg")]);
        let syntax = Syntax::mk_app(Syntax::ident("f"), vec![Syntax::ident("x")]);
        let bindings = match_pattern(&pattern, &syntax).unwrap();
        assert!(bindings.contains_key("arg"));
    }

    #[test]
    fn test_registry_register_and_lookup() {
        let mut registry = MacroRegistry::new();

        let def = MacroDef::new(
            "my_macro",
            SyntaxKind::term(),
            Syntax::mk_antiquot("x"),
            SyntaxQuote::term(Syntax::ident("result")),
        );

        registry.register(def);

        assert!(registry.get_by_name("my_macro").is_some());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_try_expand() {
        let mut registry = MacroRegistry::new();

        // Register a macro that matches any term and replaces with "expanded"
        let kind = SyntaxKind::app_kind();
        let def = MacroDef::new(
            "expand_app",
            kind.clone(),
            Syntax::node(
                kind.clone(),
                vec![Syntax::ident("myMacro"), Syntax::mk_antiquot("arg")],
            ),
            SyntaxQuote::term(Syntax::mk_antiquot("arg")),
        );

        registry.register(def);

        let input = Syntax::node(kind, vec![Syntax::ident("myMacro"), Syntax::ident("hello")]);

        let result = registry.try_expand(&input);
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_ident(), Some("hello"));
    }

    #[test]
    fn test_macro_priority() {
        let mut registry = MacroRegistry::new();

        let kind = SyntaxKind::term();

        let def1 = MacroDef::new(
            "low_priority",
            kind.clone(),
            Syntax::mk_antiquot("x"),
            SyntaxQuote::term(Syntax::ident("low")),
        )
        .with_priority(0);

        let def2 = MacroDef::new(
            "high_priority",
            kind.clone(),
            Syntax::mk_antiquot("x"),
            SyntaxQuote::term(Syntax::ident("high")),
        )
        .with_priority(100);

        registry.register(def1);
        registry.register(def2);

        let macros = registry.get_by_kind(&kind);
        assert_eq!(macros.len(), 2);
        assert_eq!(macros[0].name, "high_priority"); // Should be first due to higher priority
    }

    #[test]
    fn test_syntax_category_registry() {
        let registry = SyntaxCategoryRegistry::new();
        assert!(registry.exists("term"));
        assert!(registry.exists("command"));
        assert!(registry.exists("tactic"));
        assert!(!registry.exists("nonexistent"));
    }

    #[test]
    fn test_syntax_category_custom() {
        let mut registry = SyntaxCategoryRegistry::new();
        registry.register(SyntaxCategory::new("myCategory").with_description("Custom category"));

        assert!(registry.exists("myCategory"));
        let cat = registry.get("myCategory").unwrap();
        assert_eq!(cat.description, Some("Custom category".to_string()));
    }

    #[test]
    fn test_pattern_matching_splice_empty() {
        // Pattern: (f $[args]*)
        // Syntax: (f)
        // Should match with args = []
        let kind = SyntaxKind::app_kind();
        let pattern = Syntax::node(
            kind.clone(),
            vec![Syntax::ident("f"), Syntax::mk_antiquot_splice("args")],
        );
        let syntax = Syntax::node(kind, vec![Syntax::ident("f")]);

        let bindings = match_pattern(&pattern, &syntax);
        assert!(bindings.is_some(), "Pattern should match empty args");
        let bindings = bindings.unwrap();
        assert!(bindings.contains_key("args"));
        // args should be a splice_list with 0 children
        let args = &bindings["args"];
        assert_eq!(args.children().len(), 0);
    }

    #[test]
    fn test_pattern_matching_splice_multiple() {
        // Pattern: (f $[args]*)
        // Syntax: (f a b c)
        // Should match with args = [a, b, c]
        let kind = SyntaxKind::app_kind();
        let pattern = Syntax::node(
            kind.clone(),
            vec![Syntax::ident("f"), Syntax::mk_antiquot_splice("args")],
        );
        let syntax = Syntax::node(
            kind,
            vec![
                Syntax::ident("f"),
                Syntax::ident("a"),
                Syntax::ident("b"),
                Syntax::ident("c"),
            ],
        );

        let bindings = match_pattern(&pattern, &syntax);
        assert!(bindings.is_some(), "Pattern should match multiple args");
        let bindings = bindings.unwrap();
        let args = &bindings["args"];
        assert_eq!(args.children().len(), 3);
        assert_eq!(args.children()[0].as_ident(), Some("a"));
        assert_eq!(args.children()[1].as_ident(), Some("b"));
        assert_eq!(args.children()[2].as_ident(), Some("c"));
    }

    #[test]
    fn test_pattern_matching_splice_with_prefix() {
        // Pattern: (let $name $[exprs]*)
        // Syntax: (let x a b)
        // Should match with name = x, exprs = [a, b]
        let kind = SyntaxKind::app_kind();
        let pattern = Syntax::node(
            kind.clone(),
            vec![
                Syntax::ident("let"),
                Syntax::mk_antiquot("name"),
                Syntax::mk_antiquot_splice("exprs"),
            ],
        );
        let syntax = Syntax::node(
            kind,
            vec![
                Syntax::ident("let"),
                Syntax::ident("x"),
                Syntax::ident("a"),
                Syntax::ident("b"),
            ],
        );

        let bindings = match_pattern(&pattern, &syntax);
        assert!(bindings.is_some(), "Pattern should match with prefix");
        let bindings = bindings.unwrap();
        assert_eq!(bindings["name"].as_ident(), Some("x"));
        let exprs = &bindings["exprs"];
        assert_eq!(exprs.children().len(), 2);
    }

    #[test]
    fn test_pattern_matching_splice_with_suffix() {
        // Pattern: (fn $[args]* => $body)
        // Syntax: (fn x y => z)
        // Should match with args = [x, y], body = z
        let kind = SyntaxKind::app_kind();
        let pattern = Syntax::node(
            kind.clone(),
            vec![
                Syntax::ident("fn"),
                Syntax::mk_antiquot_splice("args"),
                Syntax::ident("=>"),
                Syntax::mk_antiquot("body"),
            ],
        );
        let syntax = Syntax::node(
            kind,
            vec![
                Syntax::ident("fn"),
                Syntax::ident("x"),
                Syntax::ident("y"),
                Syntax::ident("=>"),
                Syntax::ident("z"),
            ],
        );

        let bindings = match_pattern(&pattern, &syntax);
        assert!(bindings.is_some(), "Pattern should match with suffix");
        let bindings = bindings.unwrap();
        let args = &bindings["args"];
        assert_eq!(args.children().len(), 2);
        assert_eq!(args.children()[0].as_ident(), Some("x"));
        assert_eq!(args.children()[1].as_ident(), Some("y"));
        assert_eq!(bindings["body"].as_ident(), Some("z"));
    }
}
