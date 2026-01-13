//! Framework-specific wiring patterns.
//!
//! Each framework has specific patterns for how components are "wired together".
//! This module defines the `FrameworkPattern` trait and provides implementations
//! for common frameworks.

use crate::graph::WiringGraph;
use crate::report::WiringIssue;

pub mod clap;
pub mod tokio;
pub mod axum;

/// Trait for detecting and checking framework-specific wiring patterns.
pub trait FrameworkPattern: Send + Sync {
    /// The name of the framework.
    fn name(&self) -> &'static str;

    /// Detect if this framework is used in the project.
    fn detect(&self, graph: &WiringGraph, sources: &[String]) -> bool;

    /// Check for wiring issues specific to this framework.
    fn check_wiring(&self, graph: &WiringGraph, sources: &[String]) -> Vec<WiringIssue>;

    /// Get a description of what properly wired code looks like for this framework.
    fn wiring_description(&self) -> &'static str;
}

/// Registry of framework patterns.
pub struct PatternRegistry {
    patterns: Vec<Box<dyn FrameworkPattern>>,
}

impl Default for PatternRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            patterns: Vec::new(),
        };

        // Register built-in patterns
        registry.register(Box::new(clap::ClapPattern::new()));
        registry.register(Box::new(tokio::TokioPattern::new()));
        registry.register(Box::new(axum::AxumPattern::new()));

        registry
    }

    /// Register a new framework pattern.
    pub fn register(&mut self, pattern: Box<dyn FrameworkPattern>) {
        self.patterns.push(pattern);
    }

    /// Get all patterns that are detected in the project.
    pub fn detect_patterns(
        &self,
        graph: &WiringGraph,
        sources: &[String],
    ) -> Vec<&dyn FrameworkPattern> {
        self.patterns
            .iter()
            .filter(|p| p.detect(graph, sources))
            .map(|p| p.as_ref())
            .collect()
    }

    /// Check wiring for all detected patterns.
    pub fn check_all(&self, graph: &WiringGraph, sources: &[String]) -> Vec<WiringIssue> {
        let mut issues = Vec::new();

        for pattern in &self.patterns {
            if pattern.detect(graph, sources) {
                issues.extend(pattern.check_wiring(graph, sources));
            }
        }

        issues
    }
}

/// Common framework detection helpers.
pub mod detection {
    /// Check if source contains a pattern (simple substring match).
    pub fn contains_pattern(sources: &[String], pattern: &str) -> bool {
        sources.iter().any(|s| s.contains(pattern))
    }

    /// Check if source contains any of the patterns.
    pub fn contains_any(sources: &[String], patterns: &[&str]) -> bool {
        patterns.iter().any(|p| contains_pattern(sources, p))
    }

    /// Check if source contains a use/import statement.
    pub fn has_import(sources: &[String], module: &str) -> bool {
        let pattern = format!("use {}", module);
        contains_pattern(sources, &pattern)
    }

    /// Check if source contains a derive attribute.
    pub fn has_derive(sources: &[String], derive: &str) -> bool {
        let pattern = format!("#[derive({})]", derive);
        contains_pattern(sources, &pattern)
            || contains_pattern(sources, &format!("#[derive({},", derive))
            || contains_pattern(sources, &format!("#[derive({} ", derive))
    }
}
