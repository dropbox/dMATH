//! Macro expansion
//!
//! This module implements the macro expansion algorithm. During elaboration,
//! syntax is repeatedly expanded until no more macros apply.
//!
//! Supports hygienic expansion where each macro expansion gets a unique scope,
//! preventing accidental variable capture.

use crate::hygiene::HygieneContext;
use crate::registry::MacroRegistry;
use crate::syntax::Syntax;

/// Maximum number of expansion iterations to prevent infinite loops
const MAX_EXPANSION_DEPTH: usize = 1000;

/// Result of macro expansion
pub type MacroResult<T> = Result<T, MacroError>;

/// Macro expansion error
#[derive(Debug, Clone)]
pub enum MacroError {
    /// Maximum expansion depth exceeded (likely infinite loop)
    MaxDepthExceeded,
    /// Expansion failed with a message
    ExpansionFailed(String),
    /// Cyclic macro detected
    CyclicMacro(String),
    /// Unknown macro referenced
    UnknownMacro(String),
}

impl std::fmt::Display for MacroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MacroError::MaxDepthExceeded => {
                write!(f, "maximum macro expansion depth exceeded")
            }
            MacroError::ExpansionFailed(msg) => {
                write!(f, "macro expansion failed: {msg}")
            }
            MacroError::CyclicMacro(name) => {
                write!(f, "cyclic macro detected: {name}")
            }
            MacroError::UnknownMacro(name) => {
                write!(f, "unknown macro: {name}")
            }
        }
    }
}

impl std::error::Error for MacroError {}

/// Macro expander
pub struct MacroExpander<'a> {
    /// Registry of macros
    registry: &'a MacroRegistry,
    /// Track expansion depth
    depth: usize,
    /// Statistics
    stats: ExpansionStats,
}

/// Statistics from expansion
#[derive(Debug, Clone, Default)]
pub struct ExpansionStats {
    /// Number of expansions performed
    pub expansions: usize,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Macro names that were expanded
    pub macros_used: Vec<String>,
}

impl<'a> MacroExpander<'a> {
    /// Create a new expander
    pub fn new(registry: &'a MacroRegistry) -> Self {
        Self {
            registry,
            depth: 0,
            stats: ExpansionStats::default(),
        }
    }

    /// Expand all macros in syntax, returning the fully expanded result
    pub fn expand(&mut self, syntax: Syntax) -> MacroResult<Syntax> {
        self.expand_inner(syntax)
    }

    /// Expand macros with depth tracking
    fn expand_inner(&mut self, syntax: Syntax) -> MacroResult<Syntax> {
        if self.depth > MAX_EXPANSION_DEPTH {
            return Err(MacroError::MaxDepthExceeded);
        }

        self.depth += 1;
        self.stats.max_depth = self.stats.max_depth.max(self.depth);

        // Try to expand this node
        let result = if let Some(expanded) = self.registry.try_expand(&syntax) {
            self.stats.expansions += 1;
            // Record which macro was used (if we can determine it)
            if let Some(kind) = syntax.kind() {
                for m in self.registry.get_by_kind(kind) {
                    if m.try_match(&syntax).is_some() {
                        self.stats.macros_used.push(m.name.clone());
                        break;
                    }
                }
            }
            // Recursively expand the result
            self.expand_inner(expanded)?
        } else {
            // No macro matched, expand children
            self.expand_children(syntax)?
        };

        self.depth -= 1;
        Ok(result)
    }

    /// Expand macros in children of a syntax node
    fn expand_children(&mut self, syntax: Syntax) -> MacroResult<Syntax> {
        match syntax {
            Syntax::Node(mut node) => {
                let mut new_children = Vec::with_capacity(node.children.len());
                for child in node.children.drain(..) {
                    new_children.push(self.expand_inner(child)?);
                }
                node.children = new_children;
                Ok(Syntax::Node(node))
            }
            // Atoms, identifiers, and missing don't have children
            other => Ok(other),
        }
    }

    /// Get expansion statistics
    pub fn stats(&self) -> &ExpansionStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ExpansionStats::default();
    }
}

/// Expand syntax using a registry (convenience function)
pub fn expand(registry: &MacroRegistry, syntax: Syntax) -> MacroResult<Syntax> {
    let mut expander = MacroExpander::new(registry);
    expander.expand(syntax)
}

/// Hygienic macro expander
///
/// This expander applies hygiene to macro expansions by assigning unique scopes
/// to each expansion. Identifiers introduced by macros will have these scopes
/// attached, preventing accidental capture.
pub struct HygienicExpander<'a> {
    /// Registry of macros
    registry: &'a MacroRegistry,
    /// Hygiene context for generating fresh scopes
    hygiene: HygieneContext,
    /// Track expansion depth
    depth: usize,
    /// Statistics
    stats: ExpansionStats,
}

impl<'a> HygienicExpander<'a> {
    /// Create a new hygienic expander
    pub fn new(registry: &'a MacroRegistry) -> Self {
        Self {
            registry,
            hygiene: HygieneContext::new(),
            depth: 0,
            stats: ExpansionStats::default(),
        }
    }

    /// Create with an existing hygiene context
    pub fn with_hygiene(registry: &'a MacroRegistry, hygiene: HygieneContext) -> Self {
        Self {
            registry,
            hygiene,
            depth: 0,
            stats: ExpansionStats::default(),
        }
    }

    /// Expand all macros with hygiene, returning the fully expanded result
    pub fn expand(&mut self, syntax: Syntax) -> MacroResult<Syntax> {
        self.expand_inner(syntax)
    }

    /// Expand macros with depth and hygiene tracking
    fn expand_inner(&mut self, syntax: Syntax) -> MacroResult<Syntax> {
        if self.depth > MAX_EXPANSION_DEPTH {
            return Err(MacroError::MaxDepthExceeded);
        }

        self.depth += 1;
        self.stats.max_depth = self.stats.max_depth.max(self.depth);

        // Try to expand this node
        let result = if let Some(expanded) = self.registry.try_expand(&syntax) {
            self.stats.expansions += 1;

            // Record which macro was used
            if let Some(kind) = syntax.kind() {
                for m in self.registry.get_by_kind(kind) {
                    if m.try_match(&syntax).is_some() {
                        self.stats.macros_used.push(m.name.clone());
                        break;
                    }
                }
            }

            // Push a new scope for this macro expansion
            let _scope = self.hygiene.state_mut().push_scope();

            // Apply hygiene to the expanded syntax
            let hygienic = self.apply_hygiene(expanded);

            // Recursively expand the result
            let result = self.expand_inner(hygienic)?;

            // Pop the scope
            self.hygiene.state_mut().pop_scope();

            result
        } else {
            // No macro matched, expand children
            self.expand_children(syntax)?
        };

        self.depth -= 1;
        Ok(result)
    }

    /// Apply hygiene scope to identifiers introduced by macro expansion
    ///
    /// This marks identifiers that are "introduced" by the macro (not from
    /// antiquotation splicing) with the current macro scope. The scope parameter
    /// is reserved for future use (e.g., more fine-grained scope tracking).
    fn apply_hygiene(&mut self, syntax: Syntax) -> Syntax {
        match syntax {
            // For identifiers, we add the scope as a suffix to indicate
            // this name was introduced by a macro
            Syntax::Ident(info, name) => {
                // Mark the identifier with the scope
                // In a full implementation, we'd attach scopes to SourceInfo
                // For now, we use mangled names for fresh identifiers
                if self.is_fresh_name(&name) {
                    let scoped_name = self.hygiene.state().apply_scopes(&name);
                    Syntax::Ident(info, scoped_name.mangled())
                } else {
                    Syntax::Ident(info, name)
                }
            }

            Syntax::Node(mut node) => {
                // Recursively apply to children
                let new_children: Vec<_> = node
                    .children
                    .drain(..)
                    .map(|c| self.apply_hygiene(c))
                    .collect();
                node.children = new_children;
                Syntax::Node(node)
            }

            // Atoms and missing syntax don't need hygiene
            other => other,
        }
    }

    /// Check if a name looks like a fresh/generated name
    ///
    /// Fresh names typically start with underscore or contain generated suffixes.
    /// In a full implementation, we'd track which names came from antiquotations.
    fn is_fresh_name(&self, name: &str) -> bool {
        // Names starting with underscore are typically generated/temporary
        name.starts_with('_')
    }

    /// Expand macros in children of a syntax node
    fn expand_children(&mut self, syntax: Syntax) -> MacroResult<Syntax> {
        match syntax {
            Syntax::Node(mut node) => {
                let mut new_children = Vec::with_capacity(node.children.len());
                for child in node.children.drain(..) {
                    new_children.push(self.expand_inner(child)?);
                }
                node.children = new_children;
                Ok(Syntax::Node(node))
            }
            other => Ok(other),
        }
    }

    /// Get expansion statistics
    pub fn stats(&self) -> &ExpansionStats {
        &self.stats
    }

    /// Get read access to hygiene context
    pub fn hygiene(&self) -> &HygieneContext {
        &self.hygiene
    }

    /// Get mutable access to hygiene context
    pub fn hygiene_mut(&mut self) -> &mut HygieneContext {
        &mut self.hygiene
    }

    /// Generate a fresh identifier that's hygienic (won't capture)
    pub fn fresh_ident(&mut self, prefix: &str) -> String {
        let name = self.hygiene.fresh(prefix);
        let scoped = self.hygiene.scoped_name(&name);
        scoped.mangled()
    }
}

/// Expand syntax hygienically (convenience function)
pub fn expand_hygienic(registry: &MacroRegistry, syntax: Syntax) -> MacroResult<Syntax> {
    let mut expander = HygienicExpander::new(registry);
    expander.expand(syntax)
}

/// Expand syntax once (single step)
pub fn expand_once(registry: &MacroRegistry, syntax: &Syntax) -> Option<Syntax> {
    registry.try_expand(syntax)
}

/// Check if syntax would be expanded
pub fn would_expand(registry: &MacroRegistry, syntax: &Syntax) -> bool {
    expand_once(registry, syntax).is_some()
}

/// Trace expansion steps for debugging
#[derive(Debug, Clone)]
pub struct ExpansionTrace {
    pub steps: Vec<ExpansionStep>,
}

#[derive(Debug, Clone)]
pub struct ExpansionStep {
    pub input: Syntax,
    pub output: Syntax,
    pub macro_name: Option<String>,
}

impl ExpansionTrace {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn add_step(&mut self, input: Syntax, output: Syntax, macro_name: Option<String>) {
        self.steps.push(ExpansionStep {
            input,
            output,
            macro_name,
        });
    }
}

impl Default for ExpansionTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Expand with tracing
pub fn expand_traced(
    registry: &MacroRegistry,
    syntax: Syntax,
) -> MacroResult<(Syntax, ExpansionTrace)> {
    let mut trace = ExpansionTrace::new();
    let result = expand_traced_inner(registry, syntax, &mut trace, 0)?;
    Ok((result, trace))
}

fn expand_traced_inner(
    registry: &MacroRegistry,
    syntax: Syntax,
    trace: &mut ExpansionTrace,
    depth: usize,
) -> MacroResult<Syntax> {
    if depth > MAX_EXPANSION_DEPTH {
        return Err(MacroError::MaxDepthExceeded);
    }

    if let Some(expanded) = registry.try_expand(&syntax) {
        // Find which macro was applied
        let macro_name = syntax.kind().and_then(|kind| {
            registry
                .get_by_kind(kind)
                .iter()
                .find(|m| m.try_match(&syntax).is_some())
                .map(|m| m.name.clone())
        });

        trace.add_step(syntax, expanded.clone(), macro_name);
        expand_traced_inner(registry, expanded, trace, depth + 1)
    } else {
        // Expand children
        match syntax {
            Syntax::Node(mut node) => {
                let mut new_children = Vec::with_capacity(node.children.len());
                for child in node.children.drain(..) {
                    new_children.push(expand_traced_inner(registry, child, trace, depth + 1)?);
                }
                node.children = new_children;
                Ok(Syntax::Node(node))
            }
            other => Ok(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quotation::SyntaxQuote;
    use crate::registry::MacroDef;
    use crate::syntax::SyntaxKind;

    #[test]
    fn test_expand_no_macros() {
        let registry = MacroRegistry::new();
        let syntax = Syntax::ident("foo");

        let result = expand(&registry, syntax).unwrap();
        assert_eq!(result.as_ident(), Some("foo"));
    }

    #[test]
    fn test_expand_simple_macro() {
        let mut registry = MacroRegistry::new();

        let kind = SyntaxKind::app("myMacro");
        let def = MacroDef::new(
            "myMacro",
            kind.clone(),
            Syntax::node(kind.clone(), vec![Syntax::mk_antiquot("x")]),
            SyntaxQuote::term(Syntax::mk_app(
                Syntax::ident("f"),
                vec![Syntax::mk_antiquot("x")],
            )),
        );
        registry.register(def);

        let input = Syntax::node(kind, vec![Syntax::ident("arg")]);
        let result = expand(&registry, input).unwrap();

        assert!(result.is_node());
        assert_eq!(result.kind(), Some(&SyntaxKind::app_kind()));
    }

    #[test]
    fn test_expand_nested() {
        let mut registry = MacroRegistry::new();

        let kind = SyntaxKind::app("wrap");
        let def = MacroDef::new(
            "wrap",
            kind.clone(),
            Syntax::node(kind.clone(), vec![Syntax::mk_antiquot("x")]),
            SyntaxQuote::term(Syntax::mk_paren(Syntax::mk_antiquot("x"))),
        );
        registry.register(def);

        // Test that macro is applied to nested occurrences
        let inner = Syntax::node(kind.clone(), vec![Syntax::ident("inner")]);
        let outer = Syntax::node(kind, vec![inner]);

        let result = expand(&registry, outer).unwrap();
        // Should be doubly parenthesized
        assert_eq!(result.kind(), Some(&SyntaxKind::paren()));
    }

    #[test]
    fn test_expand_max_depth() {
        let mut registry = MacroRegistry::new();

        // Create a macro that expands to itself (infinite loop)
        let kind = SyntaxKind::app("loop");
        let def = MacroDef::new(
            "loop",
            kind.clone(),
            Syntax::node(kind.clone(), vec![]),
            SyntaxQuote::term(Syntax::node(kind.clone(), vec![])),
        );
        registry.register(def);

        let input = Syntax::node(kind, vec![]);
        let result = expand(&registry, input);

        assert!(matches!(result, Err(MacroError::MaxDepthExceeded)));
    }

    #[test]
    fn test_expand_traced() {
        let mut registry = MacroRegistry::new();

        let kind = SyntaxKind::app("myMacro");
        let def = MacroDef::new(
            "myMacro",
            kind.clone(),
            Syntax::node(kind.clone(), vec![]),
            SyntaxQuote::term(Syntax::ident("expanded")),
        );
        registry.register(def);

        let input = Syntax::node(kind, vec![]);
        let (result, trace) = expand_traced(&registry, input).unwrap();

        assert_eq!(result.as_ident(), Some("expanded"));
        assert_eq!(trace.steps.len(), 1);
        assert_eq!(trace.steps[0].macro_name, Some("myMacro".to_string()));
    }

    #[test]
    fn test_would_expand() {
        let mut registry = MacroRegistry::new();

        let kind = SyntaxKind::app("myMacro");
        let def = MacroDef::new(
            "myMacro",
            kind.clone(),
            Syntax::node(kind.clone(), vec![]),
            SyntaxQuote::term(Syntax::ident("result")),
        );
        registry.register(def);

        let matching = Syntax::node(kind, vec![]);
        let non_matching = Syntax::ident("other");

        assert!(would_expand(&registry, &matching));
        assert!(!would_expand(&registry, &non_matching));
    }

    #[test]
    fn test_expansion_stats() {
        let mut registry = MacroRegistry::new();

        let kind = SyntaxKind::app("counted");
        let def = MacroDef::new(
            "counted",
            kind.clone(),
            Syntax::node(kind.clone(), vec![]),
            SyntaxQuote::term(Syntax::ident("done")),
        );
        registry.register(def);

        let mut expander = MacroExpander::new(&registry);
        let input = Syntax::node(kind, vec![]);
        let _ = expander.expand(input);

        assert_eq!(expander.stats().expansions, 1);
        assert!(expander
            .stats()
            .macros_used
            .contains(&"counted".to_string()));
    }

    #[test]
    fn test_macro_error_display() {
        assert!(format!("{}", MacroError::MaxDepthExceeded).contains("maximum"));
        assert!(format!("{}", MacroError::CyclicMacro("foo".into())).contains("cyclic"));
        assert!(format!("{}", MacroError::UnknownMacro("bar".into())).contains("unknown"));
    }

    // Hygienic expander tests

    #[test]
    fn test_hygienic_expand_basic() {
        let mut registry = MacroRegistry::new();

        let kind = SyntaxKind::app("myMacro");
        let def = MacroDef::new(
            "myMacro",
            kind.clone(),
            Syntax::node(kind.clone(), vec![Syntax::mk_antiquot("x")]),
            SyntaxQuote::term(Syntax::mk_app(
                Syntax::ident("f"),
                vec![Syntax::mk_antiquot("x")],
            )),
        );
        registry.register(def);

        let input = Syntax::node(kind, vec![Syntax::ident("arg")]);
        let result = expand_hygienic(&registry, input).unwrap();

        assert!(result.is_node());
        assert_eq!(result.kind(), Some(&SyntaxKind::app_kind()));
    }

    #[test]
    fn test_hygienic_fresh_ident() {
        let registry = MacroRegistry::new();
        let mut expander = HygienicExpander::new(&registry);

        let f1 = expander.fresh_ident("_temp");
        let f2 = expander.fresh_ident("_temp");

        // Fresh identifiers should be different
        assert_ne!(f1, f2);
        // Both should start with the prefix
        assert!(f1.starts_with("_temp"));
        assert!(f2.starts_with("_temp"));
    }

    #[test]
    fn test_hygienic_expansion_with_fresh_names() {
        let mut registry = MacroRegistry::new();

        // A macro that introduces a fresh identifier
        let kind = SyntaxKind::app("letMacro");
        let def = MacroDef::new(
            "letMacro",
            kind.clone(),
            Syntax::node(kind.clone(), vec![Syntax::mk_antiquot("body")]),
            // Expands to: let _x = 42 in body
            SyntaxQuote::term(Syntax::mk_let(
                Syntax::ident("_x"), // This should get hygienically renamed
                None,
                Syntax::mk_num(42),
                Syntax::mk_antiquot("body"),
            )),
        );
        registry.register(def);

        let input = Syntax::node(kind, vec![Syntax::ident("result")]);
        let mut expander = HygienicExpander::new(&registry);
        let result = expander.expand(input).unwrap();

        // The _x should have been scoped
        let pretty = result.pretty();
        assert!(pretty.contains("_x")); // Base name is preserved
        assert_eq!(expander.stats().expansions, 1);
    }

    #[test]
    fn test_hygienic_no_capture() {
        // Test that two expansions of the same macro get different scopes
        let mut registry = MacroRegistry::new();

        let kind = SyntaxKind::app("scopeTest");
        let def = MacroDef::new(
            "scopeTest",
            kind.clone(),
            Syntax::node(kind.clone(), vec![]),
            SyntaxQuote::term(Syntax::ident("_generated")),
        );
        registry.register(def);

        let input1 = Syntax::node(kind.clone(), vec![]);
        let input2 = Syntax::node(kind, vec![]);

        let mut expander = HygienicExpander::new(&registry);
        let result1 = expander.expand(input1).unwrap();
        let result2 = expander.expand(input2).unwrap();

        // Both results should have mangled names, but different ones
        // because they were expanded in different scopes
        let name1 = result1.as_ident().unwrap();
        let name2 = result2.as_ident().unwrap();

        // Both start with _generated but have scope suffixes
        assert!(name1.starts_with("_generated"));
        assert!(name2.starts_with("_generated"));
        // They should be different due to different expansion scopes
        assert_ne!(name1, name2);
    }

    #[test]
    fn test_hygienic_expander_stats() {
        let mut registry = MacroRegistry::new();

        let kind = SyntaxKind::app("statsMacro");
        let def = MacroDef::new(
            "statsMacro",
            kind.clone(),
            Syntax::node(kind.clone(), vec![]),
            SyntaxQuote::term(Syntax::ident("done")),
        );
        registry.register(def);

        let mut expander = HygienicExpander::new(&registry);
        let input = Syntax::node(kind, vec![]);
        let _ = expander.expand(input);

        assert_eq!(expander.stats().expansions, 1);
        assert!(expander
            .stats()
            .macros_used
            .contains(&"statsMacro".to_string()));
    }
}
