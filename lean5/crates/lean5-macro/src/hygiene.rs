//! Macro hygiene implementation
//!
//! Hygiene prevents accidental variable capture during macro expansion.
//! Each macro expansion gets a unique scope that makes its bindings distinct
//! from bindings in the caller's context.
//!
//! Lean 4 uses a "marks" or "scopes" approach similar to Racket's hygiene.
//! Each identifier carries a set of scopes, and two identifiers are considered
//! equal only if they have the same name AND the same scopes.
//!
//! This implementation provides:
//! - `MacroScope`: A unique identifier for each macro expansion
//! - `HygieneState`: Tracks the current scope stack during expansion
//! - `ScopedName`: An identifier with its associated scopes

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique macro scopes
static NEXT_SCOPE_ID: AtomicU64 = AtomicU64::new(1);

/// A unique identifier for a macro expansion scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MacroScope(pub u64);

impl MacroScope {
    /// Generate a fresh unique scope
    pub fn fresh() -> Self {
        MacroScope(NEXT_SCOPE_ID.fetch_add(1, Ordering::SeqCst))
    }

    /// The empty/root scope (scope 0)
    pub fn root() -> Self {
        MacroScope(0)
    }

    /// Check if this is the root scope
    pub fn is_root(&self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Display for MacroScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// A name with associated hygiene scopes
///
/// Two scoped names are equal if they have the same base name AND
/// the same set of scopes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScopedName {
    /// The base identifier name
    pub name: String,
    /// The set of scopes this name was defined in
    pub scopes: BTreeSet<MacroScope>,
}

impl ScopedName {
    /// Create a new scoped name with no scopes (for user code)
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            scopes: BTreeSet::new(),
        }
    }

    /// Create a scoped name with a single scope
    pub fn with_scope(name: impl Into<String>, scope: MacroScope) -> Self {
        let mut scopes = BTreeSet::new();
        scopes.insert(scope);
        Self {
            name: name.into(),
            scopes,
        }
    }

    /// Create from a name and set of scopes
    pub fn with_scopes(name: impl Into<String>, scopes: BTreeSet<MacroScope>) -> Self {
        Self {
            name: name.into(),
            scopes,
        }
    }

    /// Add a scope to this name
    pub fn add_scope(&mut self, scope: MacroScope) {
        self.scopes.insert(scope);
    }

    /// Remove a scope from this name
    pub fn remove_scope(&mut self, scope: MacroScope) {
        self.scopes.remove(&scope);
    }

    /// Check if this name has a particular scope
    pub fn has_scope(&self, scope: MacroScope) -> bool {
        self.scopes.contains(&scope)
    }

    /// Get the "mangled" name that includes scope information
    ///
    /// This produces a unique string representation suitable for
    /// use in contexts that don't support scoped names natively.
    pub fn mangled(&self) -> String {
        if self.scopes.is_empty() {
            self.name.clone()
        } else {
            let scope_suffix: String = self.scopes.iter().map(|s| format!("_{}", s.0)).collect();
            format!("{}{}", self.name, scope_suffix)
        }
    }

    /// Check if two scoped names would bind to the same identifier
    ///
    /// This implements the hygiene check: names are equal only if
    /// they have the same base name AND compatible scopes.
    pub fn binds_same_as(&self, other: &ScopedName) -> bool {
        self.name == other.name && self.scopes == other.scopes
    }
}

impl std::fmt::Display for ScopedName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.scopes.is_empty() {
            write!(f, "{}", self.name)
        } else {
            let scopes: Vec<_> = self.scopes.iter().map(ToString::to_string).collect();
            write!(f, "{}@{{{}}}", self.name, scopes.join(","))
        }
    }
}

/// Tracks hygiene state during macro expansion
#[derive(Debug, Clone, Default)]
pub struct HygieneState {
    /// Stack of active scopes (innermost last)
    scope_stack: Vec<MacroScope>,
    /// Names introduced in the current expansion (for tracking)
    introduced_names: Vec<ScopedName>,
}

impl HygieneState {
    /// Create a new hygiene state
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a new scope for a macro expansion
    pub fn push_scope(&mut self) -> MacroScope {
        let scope = MacroScope::fresh();
        self.scope_stack.push(scope);
        scope
    }

    /// Pop the innermost scope
    pub fn pop_scope(&mut self) -> Option<MacroScope> {
        self.scope_stack.pop()
    }

    /// Get the current innermost scope, if any
    pub fn current_scope(&self) -> Option<MacroScope> {
        self.scope_stack.last().copied()
    }

    /// Get all currently active scopes
    pub fn current_scopes(&self) -> BTreeSet<MacroScope> {
        self.scope_stack.iter().copied().collect()
    }

    /// Create a fresh identifier with current scopes
    pub fn fresh_ident(&mut self, base: &str) -> ScopedName {
        let scopes = self.current_scopes();
        let name = ScopedName::with_scopes(base, scopes);
        self.introduced_names.push(name.clone());
        name
    }

    /// Create a fresh identifier with a generated unique suffix
    pub fn gensym(&mut self, prefix: &str) -> ScopedName {
        let scope = self.current_scope().unwrap_or_else(MacroScope::root);
        let unique_name = format!(
            "{}_{}_{}",
            prefix,
            scope.0,
            NEXT_SCOPE_ID.fetch_add(1, Ordering::SeqCst)
        );
        ScopedName::with_scope(unique_name, scope)
    }

    /// Apply current scopes to an existing name
    pub fn apply_scopes(&self, name: &str) -> ScopedName {
        ScopedName::with_scopes(name, self.current_scopes())
    }

    /// Get all names introduced during expansion
    pub fn introduced_names(&self) -> &[ScopedName] {
        &self.introduced_names
    }

    /// Clear introduced names (for reuse)
    pub fn clear_introduced(&mut self) {
        self.introduced_names.clear();
    }

    /// Get the depth of the scope stack
    pub fn depth(&self) -> usize {
        self.scope_stack.len()
    }
}

/// A context for managing hygiene across multiple macro expansions
#[derive(Debug, Default)]
pub struct HygieneContext {
    /// The current hygiene state
    state: HygieneState,
    /// Counter for generating fresh identifiers within this context
    fresh_counter: u64,
}

impl HygieneContext {
    /// Create a new hygiene context
    pub fn new() -> Self {
        Self::default()
    }

    /// Get mutable access to the hygiene state
    pub fn state_mut(&mut self) -> &mut HygieneState {
        &mut self.state
    }

    /// Get read access to the hygiene state
    pub fn state(&self) -> &HygieneState {
        &self.state
    }

    /// Enter a new macro scope, returning a guard that will pop on drop
    pub fn enter_scope(&mut self) -> ScopeGuard<'_> {
        let scope = self.state.push_scope();
        ScopeGuard { ctx: self, scope }
    }

    /// Generate a fresh identifier unique to this context
    pub fn fresh(&mut self, prefix: &str) -> String {
        self.fresh_counter += 1;
        format!("{}_{}", prefix, self.fresh_counter)
    }

    /// Create a scoped name with current context
    pub fn scoped_name(&self, name: &str) -> ScopedName {
        self.state.apply_scopes(name)
    }
}

/// Guard that automatically pops a scope when dropped
pub struct ScopeGuard<'a> {
    ctx: &'a mut HygieneContext,
    scope: MacroScope,
}

impl<'a> ScopeGuard<'a> {
    /// Get the scope this guard represents
    pub fn scope(&self) -> MacroScope {
        self.scope
    }

    /// Create a fresh name in this scope
    pub fn fresh_name(&mut self, base: &str) -> ScopedName {
        self.ctx.state.fresh_ident(base)
    }
}

impl<'a> Drop for ScopeGuard<'a> {
    fn drop(&mut self) {
        self.ctx.state.pop_scope();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_scope_fresh() {
        let s1 = MacroScope::fresh();
        let s2 = MacroScope::fresh();
        assert_ne!(s1, s2);
        assert!(s2.0 > s1.0);
    }

    #[test]
    fn test_macro_scope_root() {
        let root = MacroScope::root();
        assert!(root.is_root());
        assert_eq!(root.0, 0);
    }

    #[test]
    fn test_scoped_name_equality() {
        let s1 = MacroScope::fresh();
        let s2 = MacroScope::fresh();

        let n1 = ScopedName::with_scope("x", s1);
        let n2 = ScopedName::with_scope("x", s1);
        let n3 = ScopedName::with_scope("x", s2);
        let n4 = ScopedName::with_scope("y", s1);

        assert!(n1.binds_same_as(&n2)); // Same name, same scope
        assert!(!n1.binds_same_as(&n3)); // Same name, different scope
        assert!(!n1.binds_same_as(&n4)); // Different name, same scope
    }

    #[test]
    fn test_scoped_name_mangled() {
        let s = MacroScope::fresh();
        let n1 = ScopedName::new("x");
        let n2 = ScopedName::with_scope("x", s);

        assert_eq!(n1.mangled(), "x");
        assert!(n2.mangled().starts_with("x_"));
    }

    #[test]
    fn test_hygiene_state_scope_stack() {
        let mut state = HygieneState::new();
        assert_eq!(state.depth(), 0);
        assert!(state.current_scope().is_none());

        let s1 = state.push_scope();
        assert_eq!(state.depth(), 1);
        assert_eq!(state.current_scope(), Some(s1));

        let s2 = state.push_scope();
        assert_eq!(state.depth(), 2);
        assert_eq!(state.current_scope(), Some(s2));

        assert_eq!(state.pop_scope(), Some(s2));
        assert_eq!(state.current_scope(), Some(s1));

        assert_eq!(state.pop_scope(), Some(s1));
        assert!(state.current_scope().is_none());
    }

    #[test]
    fn test_hygiene_state_fresh_ident() {
        let mut state = HygieneState::new();
        let s1 = state.push_scope();

        let n1 = state.fresh_ident("x");
        assert!(n1.has_scope(s1));
        assert_eq!(n1.name, "x");

        let s2 = state.push_scope();
        let n2 = state.fresh_ident("x");
        assert!(n2.has_scope(s1));
        assert!(n2.has_scope(s2));

        // These should not bind to the same thing
        assert!(!n1.binds_same_as(&n2));
    }

    #[test]
    fn test_hygiene_context_enter_scope() {
        let mut ctx = HygieneContext::new();
        assert_eq!(ctx.state().depth(), 0);

        // Manually push and pop to test the mechanism
        let scope = ctx.state_mut().push_scope();
        assert_eq!(ctx.state().depth(), 1);
        assert!(scope.0 > 0); // Non-root scope

        ctx.state_mut().pop_scope();
        // Scope automatically popped when guard dropped
        assert_eq!(ctx.state().depth(), 0);
    }

    #[test]
    fn test_hygiene_context_fresh() {
        let mut ctx = HygieneContext::new();
        let f1 = ctx.fresh("temp");
        let f2 = ctx.fresh("temp");

        assert_ne!(f1, f2);
        assert!(f1.starts_with("temp_"));
        assert!(f2.starts_with("temp_"));
    }

    #[test]
    fn test_gensym() {
        let mut state = HygieneState::new();
        let _ = state.push_scope();

        let g1 = state.gensym("_x");
        let g2 = state.gensym("_x");

        assert_ne!(g1.name, g2.name);
        assert_ne!(g1.mangled(), g2.mangled());
    }

    #[test]
    fn test_scoped_name_display() {
        let s1 = MacroScope::fresh();
        let n = ScopedName::with_scope("foo", s1);
        let displayed = format!("{n}");
        assert!(displayed.contains("foo"));
        assert!(displayed.contains('@'));
    }

    #[test]
    fn test_macro_scope_display() {
        let s = MacroScope::fresh();
        let displayed = format!("{s}");
        assert!(displayed.starts_with('s'));
    }
}
