//! Tokio async runtime pattern detection.
//!
//! Detects common wiring issues with tokio-based applications:
//! - Async functions defined but no runtime started
//! - Missing #[tokio::main] attribute
//! - Runtime created but block_on never called

use super::{detection, FrameworkPattern};
use crate::graph::{EntryKind, Node, WiringGraph};
use crate::report::{Severity, WiringIssue};

/// Pattern checker for the tokio async runtime.
pub struct TokioPattern;

impl TokioPattern {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TokioPattern {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameworkPattern for TokioPattern {
    fn name(&self) -> &'static str {
        "tokio"
    }

    fn detect(&self, _graph: &WiringGraph, sources: &[String]) -> bool {
        detection::has_import(sources, "tokio")
            || detection::contains_pattern(sources, "#[tokio::main]")
            || detection::contains_pattern(sources, "tokio::runtime")
            || detection::contains_pattern(sources, "Runtime::new")
    }

    fn check_wiring(&self, graph: &WiringGraph, sources: &[String]) -> Vec<WiringIssue> {
        let mut issues = Vec::new();

        // Count async functions
        let async_functions: Vec<_> = graph
            .nodes()
            .filter(|(_, node)| {
                matches!(
                    node,
                    Node::Function {
                        is_async: true,
                        ..
                    }
                )
            })
            .collect();

        if async_functions.is_empty() {
            return issues; // No async code, no issues
        }

        // Check 1: Async functions exist but no async runtime
        let has_async_main = graph.nodes().any(|(_, node)| {
            matches!(
                node,
                Node::EntryPoint {
                    kind: EntryKind::AsyncMain,
                    ..
                }
            )
        });

        let has_runtime_block_on = detection::contains_any(
            sources,
            &[
                "block_on",
                "#[tokio::main]",
                "#[async_std::main]",
                "Runtime::new",
            ],
        );

        if !has_async_main && !has_runtime_block_on {
            issues.push(WiringIssue {
                code: "TOKIO001".to_string(),
                message: format!(
                    "{} async functions defined but no async runtime started",
                    async_functions.len()
                ),
                detail: Some(
                    "You have async functions but main() is not async and no runtime is created. \
                     Async code will never execute."
                        .to_string(),
                ),
                severity: Severity::Error,
                location: None,
                suggestion: Some(
                    "Add #[tokio::main] to main() or use Runtime::new().block_on()".to_string(),
                ),
            });
        }

        // Check 2: Runtime created but block_on never called
        let has_runtime_new = detection::contains_pattern(sources, "Runtime::new");
        let has_block_on = detection::contains_pattern(sources, "block_on");

        if has_runtime_new && !has_block_on && !has_async_main {
            issues.push(WiringIssue {
                code: "TOKIO002".to_string(),
                message: "Tokio runtime created but block_on() never called".to_string(),
                detail: Some(
                    "You create a tokio Runtime but never call .block_on() on it. \
                     The runtime exists but no async code is executed on it."
                        .to_string(),
                ),
                severity: Severity::Error,
                location: None,
                suggestion: Some("Call runtime.block_on(async { ... })".to_string()),
            });
        }

        // Check 3: Async functions not reachable from async entry point
        if has_async_main || has_runtime_block_on {
            let reachable = graph.reachable_from_entries();

            for (id, node) in &async_functions {
                if !reachable.contains(id) {
                    if let Node::Function { name, .. } = node {
                        issues.push(WiringIssue {
                            code: "TOKIO003".to_string(),
                            message: format!("Async function '{}' is never awaited", name),
                            detail: Some(
                                "This async function is defined but never called or awaited. \
                                 It will never execute."
                                    .to_string(),
                            ),
                            severity: Severity::Warning,
                            location: node.location().cloned(),
                            suggestion: Some(format!("Call and await '{}' from your async code", name)),
                        });
                    }
                }
            }
        }

        // Check 4: spawn() without runtime
        let has_spawn = detection::contains_any(sources, &["tokio::spawn", "spawn("]);
        if has_spawn && !has_async_main && !has_runtime_block_on {
            issues.push(WiringIssue {
                code: "TOKIO004".to_string(),
                message: "tokio::spawn() called but no async runtime active".to_string(),
                detail: Some(
                    "You're calling tokio::spawn() but there's no async runtime running. \
                     This will panic at runtime."
                        .to_string(),
                ),
                severity: Severity::Error,
                location: None,
                suggestion: Some(
                    "Ensure spawn() is called within an async context with #[tokio::main]"
                        .to_string(),
                ),
            });
        }

        issues
    }

    fn wiring_description(&self) -> &'static str {
        "Tokio apps need: (1) #[tokio::main] on async fn main(), or (2) Runtime::new().block_on(). \
         All async functions must be awaited from the async entry point."
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Location;

    #[test]
    fn test_detect_tokio() {
        let pattern = TokioPattern::new();
        let graph = WiringGraph::new();

        let sources = vec!["use tokio;".to_string()];
        assert!(pattern.detect(&graph, &sources));

        let sources = vec!["#[tokio::main]".to_string()];
        assert!(pattern.detect(&graph, &sources));

        let sources = vec!["use std::io;".to_string()];
        assert!(!pattern.detect(&graph, &sources));
    }

    #[test]
    fn test_async_without_runtime() {
        let pattern = TokioPattern::new();
        let mut graph = WiringGraph::new();

        // Add a non-async main
        graph.add_node(Node::EntryPoint {
            kind: EntryKind::Main,
            loc: Location::new("src/main.rs".into(), 1, 1),
        });

        // Add an async function
        graph.add_node(Node::Function {
            name: "do_async".to_string(),
            loc: Location::new("src/main.rs".into(), 5, 1),
            is_async: true,
            is_public: false,
            params: vec![],
        });

        let sources = vec![
            r#"
use tokio;

fn main() {
    // No async runtime!
}

async fn do_async() {
    println!("async");
}
"#
            .to_string(),
        ];

        let issues = pattern.check_wiring(&graph, &sources);
        assert!(issues.iter().any(|i| i.code == "TOKIO001"));
    }

    #[test]
    fn test_proper_tokio_usage() {
        let pattern = TokioPattern::new();
        let mut graph = WiringGraph::new();

        // Add an async main
        graph.add_node(Node::EntryPoint {
            kind: EntryKind::AsyncMain,
            loc: Location::new("src/main.rs".into(), 1, 1),
        });

        let sources = vec![
            r#"
use tokio;

#[tokio::main]
async fn main() {
    println!("async main");
}
"#
            .to_string(),
        ];

        let issues = pattern.check_wiring(&graph, &sources);
        assert!(!issues.iter().any(|i| i.code == "TOKIO001"));
    }
}
