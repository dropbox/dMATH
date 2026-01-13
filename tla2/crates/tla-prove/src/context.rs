//! Proof context management
//!
//! Tracks the facts, definitions, and assumptions available during proof checking.

use std::collections::HashMap;
use tla_core::ast::{Expr, Module, Unit};
use tla_core::span::Spanned;

/// The proof context holds all available facts and definitions
#[derive(Debug, Clone)]
pub struct ProofContext {
    /// Named facts (theorems, lemmas, assumptions) that have been proved
    facts: HashMap<String, Fact>,
    /// Operator definitions that can be expanded
    definitions: HashMap<String, Definition>,
    /// Stack of local contexts (for nested proofs)
    local_stack: Vec<LocalContext>,
}

/// A fact that can be used in proofs
#[derive(Debug, Clone)]
pub struct Fact {
    /// The name of the fact
    pub name: String,
    /// The expression representing the fact
    pub expr: Spanned<Expr>,
    /// Whether this fact has been proved or is assumed
    pub proved: bool,
}

/// An operator definition
#[derive(Debug, Clone)]
pub struct Definition {
    /// The name of the operator
    pub name: String,
    /// Parameter names
    pub params: Vec<String>,
    /// The body expression
    pub body: Spanned<Expr>,
}

/// Local context for nested proofs
#[derive(Debug, Clone)]
struct LocalContext {
    /// Local facts introduced in this scope
    facts: HashMap<String, Fact>,
    /// Local definitions introduced in this scope
    definitions: HashMap<String, Definition>,
    /// Variables introduced by TAKE
    taken_vars: Vec<(String, Option<Spanned<Expr>>)>,
}

impl ProofContext {
    /// Create a new empty proof context
    pub fn new() -> Self {
        Self {
            facts: HashMap::new(),
            definitions: HashMap::new(),
            local_stack: Vec::new(),
        }
    }

    /// Initialize context from a module
    pub fn from_module(module: &Module) -> Self {
        let mut ctx = Self::new();
        ctx.load_module(module);
        ctx
    }

    /// Load definitions and assumptions from a module
    pub fn load_module(&mut self, module: &Module) {
        for unit in &module.units {
            match &unit.node {
                Unit::Operator(op) => {
                    self.add_definition(
                        op.name.node.clone(),
                        op.params.iter().map(|p| p.name.node.clone()).collect(),
                        op.body.clone(),
                    );
                }
                Unit::Assume(assume) => {
                    let name = assume
                        .name
                        .as_ref()
                        .map(|n| n.node.clone())
                        .unwrap_or_else(|| format!("_ASSUME_{}", self.facts.len()));
                    self.add_assumed_fact(name, assume.expr.clone());
                }
                Unit::Theorem(thm) => {
                    // Theorems without proofs or with OMITTED proofs are not added
                    // Theorems with proofs will be added after they're proved
                    if let Some(name) = &thm.name {
                        // Register as a known theorem name (but not proved yet)
                        self.definitions.insert(
                            name.node.clone(),
                            Definition {
                                name: name.node.clone(),
                                params: Vec::new(),
                                body: thm.body.clone(),
                            },
                        );
                    }
                }
                _ => {}
            }
        }
    }

    /// Add a proved fact
    pub fn add_proved_fact(&mut self, name: String, expr: Spanned<Expr>) {
        let fact = Fact {
            name: name.clone(),
            expr,
            proved: true,
        };

        if self.local_stack.is_empty() {
            self.facts.insert(name, fact);
        } else {
            self.local_stack
                .last_mut()
                .unwrap()
                .facts
                .insert(name, fact);
        }
    }

    /// Add an assumed fact (like ASSUME)
    pub fn add_assumed_fact(&mut self, name: String, expr: Spanned<Expr>) {
        let fact = Fact {
            name: name.clone(),
            expr,
            proved: false,
        };

        if self.local_stack.is_empty() {
            self.facts.insert(name, fact);
        } else {
            self.local_stack
                .last_mut()
                .unwrap()
                .facts
                .insert(name, fact);
        }
    }

    /// Add an operator definition
    pub fn add_definition(&mut self, name: String, params: Vec<String>, body: Spanned<Expr>) {
        let def = Definition {
            name: name.clone(),
            params,
            body,
        };

        if self.local_stack.is_empty() {
            self.definitions.insert(name, def);
        } else {
            self.local_stack
                .last_mut()
                .unwrap()
                .definitions
                .insert(name, def);
        }
    }

    /// Look up a fact by name
    pub fn get_fact(&self, name: &str) -> Option<&Fact> {
        // Search local stacks first (innermost to outermost)
        for local in self.local_stack.iter().rev() {
            if let Some(fact) = local.facts.get(name) {
                return Some(fact);
            }
        }
        self.facts.get(name)
    }

    /// Look up a definition by name
    pub fn get_definition(&self, name: &str) -> Option<&Definition> {
        // Search local stacks first
        for local in self.local_stack.iter().rev() {
            if let Some(def) = local.definitions.get(name) {
                return Some(def);
            }
        }
        self.definitions.get(name)
    }

    /// Get all available facts
    pub fn all_facts(&self) -> Vec<&Fact> {
        let mut result: Vec<&Fact> = self.facts.values().collect();
        for local in &self.local_stack {
            result.extend(local.facts.values());
        }
        result
    }

    /// Get all available definitions
    pub fn all_definitions(&self) -> Vec<&Definition> {
        let mut result: Vec<&Definition> = self.definitions.values().collect();
        for local in &self.local_stack {
            result.extend(local.definitions.values());
        }
        result
    }

    /// Enter a new local scope (for nested proofs)
    pub fn push_scope(&mut self) {
        self.local_stack.push(LocalContext {
            facts: HashMap::new(),
            definitions: HashMap::new(),
            taken_vars: Vec::new(),
        });
    }

    /// Exit the current local scope
    pub fn pop_scope(&mut self) {
        self.local_stack.pop();
    }

    /// Add a taken variable (from TAKE)
    pub fn add_taken_var(&mut self, name: String, domain: Option<Spanned<Expr>>) {
        if let Some(local) = self.local_stack.last_mut() {
            local.taken_vars.push((name, domain));
        }
    }

    /// Get taken variables in current scope
    pub fn taken_vars(&self) -> Vec<&(String, Option<Spanned<Expr>>)> {
        self.local_stack
            .last()
            .map(|l| l.taken_vars.iter().collect())
            .unwrap_or_default()
    }

    /// Check if we're in a local scope
    pub fn in_local_scope(&self) -> bool {
        !self.local_stack.is_empty()
    }

    /// Get the depth of local scopes
    pub fn scope_depth(&self) -> usize {
        self.local_stack.len()
    }
}

impl Default for ProofContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::span::{FileId, Span};

    fn span() -> Span {
        Span::new(FileId(0), 0, 0)
    }

    fn spanned<T>(node: T) -> Spanned<T> {
        Spanned::new(node, span())
    }

    #[test]
    fn test_add_and_get_fact() {
        let mut ctx = ProofContext::new();
        let expr = spanned(Expr::Bool(true));

        ctx.add_proved_fact("P".to_string(), expr.clone());

        let fact = ctx.get_fact("P").unwrap();
        assert_eq!(fact.name, "P");
        assert!(fact.proved);
    }

    #[test]
    fn test_local_scope() {
        let mut ctx = ProofContext::new();
        let global_expr = spanned(Expr::Bool(true));
        let local_expr = spanned(Expr::Bool(false));

        ctx.add_proved_fact("global".to_string(), global_expr);

        ctx.push_scope();
        ctx.add_proved_fact("local".to_string(), local_expr);

        // Both should be visible
        assert!(ctx.get_fact("global").is_some());
        assert!(ctx.get_fact("local").is_some());

        ctx.pop_scope();

        // Only global should be visible
        assert!(ctx.get_fact("global").is_some());
        assert!(ctx.get_fact("local").is_none());
    }

    #[test]
    fn test_definition_lookup() {
        let mut ctx = ProofContext::new();
        let body = spanned(Expr::Bool(true));

        ctx.add_definition("Op".to_string(), vec!["x".to_string()], body);

        let def = ctx.get_definition("Op").unwrap();
        assert_eq!(def.name, "Op");
        assert_eq!(def.params, vec!["x".to_string()]);
    }
}
