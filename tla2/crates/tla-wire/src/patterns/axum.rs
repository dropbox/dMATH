//! Axum web framework pattern detection.
//!
//! Detects common wiring issues with axum-based applications:
//! - Routes defined but not added to Router
//! - Router created but server never started
//! - Handlers defined but not registered

use super::{detection, FrameworkPattern};
use crate::graph::{Node, WiringGraph};
use crate::report::{Severity, WiringIssue};

/// Pattern checker for the axum web framework.
pub struct AxumPattern;

impl AxumPattern {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AxumPattern {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameworkPattern for AxumPattern {
    fn name(&self) -> &'static str {
        "axum"
    }

    fn detect(&self, _graph: &WiringGraph, sources: &[String]) -> bool {
        detection::has_import(sources, "axum")
            || detection::contains_pattern(sources, "axum::Router")
            || detection::contains_pattern(sources, "Router::new")
    }

    fn check_wiring(&self, graph: &WiringGraph, sources: &[String]) -> Vec<WiringIssue> {
        let mut issues = Vec::new();

        // Check 1: Router created but serve() never called
        let has_router = detection::contains_any(sources, &["Router::new", "Router::<"]);
        let has_serve = detection::contains_any(
            sources,
            &[
                "axum::serve",
                ".serve(",
                "hyper::Server",
                "Server::bind",
                "serve(",
            ],
        );

        if has_router && !has_serve {
            issues.push(WiringIssue {
                code: "AXUM001".to_string(),
                message: "Router created but server never started".to_string(),
                detail: Some(
                    "You create an axum Router but never pass it to axum::serve() or similar. \
                     The HTTP server will never start."
                        .to_string(),
                ),
                severity: Severity::Error,
                location: None,
                suggestion: Some(
                    "Add: axum::serve(listener, app).await.unwrap()".to_string(),
                ),
            });
        }

        // Check 2: Routes might be defined but not registered
        // Look for handler-like functions that aren't in the reachable set
        let reachable = graph.reachable_from_entries();

        for (id, node) in graph.nodes() {
            if let Node::Function { name, is_async, .. } = node {
                // Heuristic: async functions with HTTP-like names
                let is_handler_like = *is_async
                    && (name.starts_with("get_")
                        || name.starts_with("post_")
                        || name.starts_with("put_")
                        || name.starts_with("delete_")
                        || name.starts_with("patch_")
                        || name.starts_with("handle_")
                        || name.ends_with("_handler")
                        || name == "index"
                        || name == "root"
                        || name == "health"
                        || name == "health_check");

                if is_handler_like && !reachable.contains(&id) {
                    // Check if this handler name appears in a .route() call
                    let is_registered = detection::contains_any(
                        sources,
                        &[
                            &format!(".route(\"{}", name),
                            &format!("get({})", name),
                            &format!("post({})", name),
                            &format!("put({})", name),
                            &format!("delete({})", name),
                            &format!("patch({})", name),
                        ],
                    );

                    if !is_registered {
                        issues.push(WiringIssue {
                            code: "AXUM002".to_string(),
                            message: format!("Handler '{}' appears unregistered", name),
                            detail: Some(
                                "This function looks like an HTTP handler but doesn't appear \
                                 to be registered with .route() or similar."
                                    .to_string(),
                            ),
                            severity: Severity::Warning,
                            location: node.location().cloned(),
                            suggestion: Some(format!(
                                "Register with: .route(\"/path\", get({}))",
                                name
                            )),
                        });
                    }
                }
            }
        }

        // Check 3: Routes defined but Router not awaited/used
        let has_route_calls = detection::contains_pattern(sources, ".route(");
        let has_await_or_block =
            detection::contains_any(sources, &[".await", "block_on", "#[tokio::main]"]);

        if has_router && has_route_calls && !has_await_or_block {
            issues.push(WiringIssue {
                code: "AXUM003".to_string(),
                message: "Axum routes defined but no async runtime".to_string(),
                detail: Some(
                    "You have axum routes but no #[tokio::main] or async runtime. \
                     The server cannot start without an async runtime."
                        .to_string(),
                ),
                severity: Severity::Error,
                location: None,
                suggestion: Some("Add #[tokio::main] to main() function".to_string()),
            });
        }

        // Check 4: TcpListener::bind but not awaited
        let has_listener_bind = detection::contains_pattern(sources, "TcpListener::bind");
        let has_listener_await = detection::contains_pattern(sources, "TcpListener::bind")
            && detection::contains_pattern(sources, ".await");

        if has_listener_bind && !has_listener_await {
            issues.push(WiringIssue {
                code: "AXUM004".to_string(),
                message: "TcpListener::bind() not awaited".to_string(),
                detail: Some(
                    "TcpListener::bind() returns a Future that must be awaited. \
                     Without .await, the listener is never actually created."
                        .to_string(),
                ),
                severity: Severity::Error,
                location: None,
                suggestion: Some("Add .await after TcpListener::bind()".to_string()),
            });
        }

        issues
    }

    fn wiring_description(&self) -> &'static str {
        "Axum apps need: (1) Router::new(), (2) .route() calls to register handlers, \
         (3) TcpListener::bind().await, (4) axum::serve().await"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_axum() {
        let pattern = AxumPattern::new();
        let graph = WiringGraph::new();

        let sources = vec!["use axum::Router;".to_string()];
        assert!(pattern.detect(&graph, &sources));

        let sources = vec!["Router::new()".to_string()];
        assert!(pattern.detect(&graph, &sources));

        let sources = vec!["use actix_web;".to_string()];
        assert!(!pattern.detect(&graph, &sources));
    }

    #[test]
    fn test_router_without_serve() {
        let pattern = AxumPattern::new();
        let graph = WiringGraph::new();

        let sources = vec![
            r#"
use axum::Router;

fn main() {
    let app = Router::new();
    // Forgot to serve!
}
"#
            .to_string(),
        ];

        let issues = pattern.check_wiring(&graph, &sources);
        assert!(issues.iter().any(|i| i.code == "AXUM001"));
    }

    #[test]
    fn test_proper_axum_usage() {
        let pattern = AxumPattern::new();
        let graph = WiringGraph::new();

        let sources = vec![
            r#"
use axum::{Router, routing::get};

#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(root));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    "Hello"
}
"#
            .to_string(),
        ];

        let issues = pattern.check_wiring(&graph, &sources);
        assert!(!issues.iter().any(|i| i.code == "AXUM001"));
    }
}
