//! Rust language adapter for wiring graph extraction.
//!
//! Extracts functions, entry points, effects, and call relationships from Rust source code.

use super::{ExtractError, LanguageAdapter};
use crate::graph::{Edge, EffectKind, EntryKind, Location, Node, WiringGraph};
use std::path::Path;

/// Adapter for extracting wiring information from Rust source code.
pub struct RustAdapter {
    language: tree_sitter::Language,
}

impl RustAdapter {
    pub fn new() -> Self {
        // tree-sitter-rust 0.23 uses tree_sitter_language which wraps Language
        // Call the function pointer to get the raw language pointer
        let lang_ptr = unsafe { tree_sitter_rust::LANGUAGE.into_raw()() };
        Self {
            language: unsafe {
                tree_sitter::Language::from_raw(lang_ptr as *const tree_sitter::ffi::TSLanguage)
            },
        }
    }
}

impl Default for RustAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageAdapter for RustAdapter {
    fn language(&self) -> &'static str {
        "rust"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["rs"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        self.language.clone()
    }

    fn extract_nodes(
        &self,
        tree: &tree_sitter::Tree,
        source: &str,
        path: &Path,
    ) -> Result<Vec<Node>, ExtractError> {
        let mut nodes = Vec::new();
        let root = tree.root_node();

        self.extract_nodes_recursive(&root, source, path, &mut nodes);

        Ok(nodes)
    }

    fn extract_edges(
        &self,
        tree: &tree_sitter::Tree,
        source: &str,
        path: &Path,
        graph: &WiringGraph,
    ) -> Result<Vec<Edge>, ExtractError> {
        let mut edges = Vec::new();
        let root = tree.root_node();

        self.extract_edges_recursive(&root, source, path, graph, &mut edges);

        Ok(edges)
    }
}

impl RustAdapter {
    fn extract_nodes_recursive(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        path: &Path,
        nodes: &mut Vec<Node>,
    ) {
        match node.kind() {
            "function_item" => {
                if let Some(extracted) = self.extract_function(node, source, path) {
                    nodes.push(extracted);
                }
            }
            "macro_invocation" => {
                if let Some(extracted) = self.extract_macro_effect(node, source, path) {
                    nodes.push(extracted);
                }
            }
            "impl_item" => {
                // Extract methods from impl blocks
                for child in node.children(&mut node.walk()) {
                    if child.kind() == "declaration_list" {
                        for item in child.children(&mut child.walk()) {
                            if item.kind() == "function_item" {
                                if let Some(extracted) = self.extract_function(&item, source, path)
                                {
                                    nodes.push(extracted);
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        // Recurse into children
        for child in node.children(&mut node.walk()) {
            self.extract_nodes_recursive(&child, source, path, nodes);
        }
    }

    fn extract_function(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        path: &Path,
    ) -> Option<Node> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

        let loc = Location::new(
            path.to_path_buf(),
            node.start_position().row as u32 + 1,
            node.start_position().column as u32 + 1,
        )
        .with_end(
            node.end_position().row as u32 + 1,
            node.end_position().column as u32 + 1,
        );

        // Check for attributes
        let mut is_async = false;
        let mut is_main = false;
        let mut is_tokio_main = false;
        let mut is_test = false;
        let mut is_public = false;

        // Check visibility
        for child in node.children(&mut node.walk()) {
            if child.kind() == "visibility_modifier" {
                is_public = true;
            }
            if child.kind() == "function_modifiers" {
                let mods = child.utf8_text(source.as_bytes()).ok()?;
                if mods.contains("async") {
                    is_async = true;
                }
            }
        }

        // Check for attributes (preceding siblings)
        if let Some(parent) = node.parent() {
            let mut cursor = parent.walk();
            for sibling in parent.children(&mut cursor) {
                if sibling.end_byte() >= node.start_byte() {
                    break;
                }
                if sibling.kind() == "attribute_item" {
                    let attr_text = sibling.utf8_text(source.as_bytes()).ok()?;
                    if attr_text.contains("tokio::main") || attr_text.contains("async_std::main") {
                        is_tokio_main = true;
                        is_async = true;
                    }
                    if attr_text.contains("test") {
                        is_test = true;
                    }
                }
            }
        }

        // Check if this is main
        if name == "main" {
            is_main = true;
        }

        // Determine node type
        if is_main || is_tokio_main {
            Some(Node::EntryPoint {
                kind: if is_tokio_main {
                    EntryKind::AsyncMain
                } else {
                    EntryKind::Main
                },
                loc,
            })
        } else if is_test {
            Some(Node::EntryPoint {
                kind: EntryKind::Test { name: name.clone() },
                loc,
            })
        } else {
            // Extract parameters
            let params = self.extract_params(node, source);

            Some(Node::Function {
                name,
                loc,
                is_async,
                is_public,
                params,
            })
        }
    }

    fn extract_params(&self, node: &tree_sitter::Node, source: &str) -> Vec<String> {
        let mut params = Vec::new();

        if let Some(params_node) = node.child_by_field_name("parameters") {
            for child in params_node.children(&mut params_node.walk()) {
                if child.kind() == "parameter" {
                    if let Some(pattern) = child.child_by_field_name("pattern") {
                        if let Ok(name) = pattern.utf8_text(source.as_bytes()) {
                            params.push(name.to_string());
                        }
                    }
                }
            }
        }

        params
    }

    fn extract_macro_effect(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        path: &Path,
    ) -> Option<Node> {
        let macro_node = node.child_by_field_name("macro")?;
        let macro_name = macro_node.utf8_text(source.as_bytes()).ok()?;

        let loc = Location::new(
            path.to_path_buf(),
            node.start_position().row as u32 + 1,
            node.start_position().column as u32 + 1,
        );

        let effect_kind = match macro_name {
            "println" | "print" => Some(EffectKind::Stdout),
            "eprintln" | "eprint" => Some(EffectKind::Stderr),
            "panic" | "unreachable" | "unimplemented" | "todo" => Some(EffectKind::Panic),
            _ => None,
        };

        effect_kind.map(|kind| Node::Effect { kind, loc })
    }

    fn extract_edges_recursive(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        path: &Path,
        graph: &WiringGraph,
        edges: &mut Vec<Edge>,
    ) {
        // Look for function calls
        if node.kind() == "call_expression" {
            if let Some(edge) = self.extract_call_edge(node, source, path, graph) {
                edges.push(edge);
            }
        }

        // Look for await expressions
        if node.kind() == "await_expression" {
            if let Some(edge) = self.extract_await_edge(node, source, path, graph) {
                edges.push(edge);
            }
        }

        // Recurse into children
        for child in node.children(&mut node.walk()) {
            self.extract_edges_recursive(&child, source, path, graph, edges);
        }
    }

    fn extract_call_edge(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        path: &Path,
        graph: &WiringGraph,
    ) -> Option<Edge> {
        let function_node = node.child_by_field_name("function")?;
        let callee_name = self.extract_callee_name(&function_node, source)?;

        // Find the enclosing function (caller)
        let caller_id = self.find_enclosing_function(node, source, graph)?;

        // Find the callee in the graph
        let callee_id = graph.find_by_name(&callee_name).next()?;

        let loc = Location::new(
            path.to_path_buf(),
            node.start_position().row as u32 + 1,
            node.start_position().column as u32 + 1,
        );

        Some(Edge::Calls {
            from: caller_id,
            to: callee_id,
            loc,
        })
    }

    fn extract_await_edge(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        path: &Path,
        graph: &WiringGraph,
    ) -> Option<Edge> {
        // The awaited expression
        let expr = node.child(0)?;

        // If it's a call expression, extract the callee
        if expr.kind() == "call_expression" {
            return self.extract_call_edge(&expr, source, path, graph).map(|e| {
                if let Edge::Calls { from, to, loc } = e {
                    Edge::Awaits { from, to, loc }
                } else {
                    e
                }
            });
        }

        None
    }

    fn extract_callee_name(&self, node: &tree_sitter::Node, source: &str) -> Option<String> {
        match node.kind() {
            "identifier" => node.utf8_text(source.as_bytes()).ok().map(String::from),
            "field_expression" => {
                // For method calls like foo.bar(), extract "bar"
                node.child_by_field_name("field")
                    .and_then(|f| f.utf8_text(source.as_bytes()).ok())
                    .map(String::from)
            }
            "scoped_identifier" => {
                // For paths like module::function, extract "function"
                node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .map(String::from)
            }
            _ => node.utf8_text(source.as_bytes()).ok().map(String::from),
        }
    }

    fn find_enclosing_function(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        graph: &WiringGraph,
    ) -> Option<crate::graph::NodeId> {
        let mut current = Some(*node);

        while let Some(n) = current {
            if n.kind() == "function_item" {
                if let Some(name_node) = n.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                        // Check if this is main
                        if name == "main" {
                            // Find main entry point
                            for (id, node) in graph.nodes() {
                                if matches!(
                                    node,
                                    Node::EntryPoint {
                                        kind: EntryKind::Main | EntryKind::AsyncMain,
                                        ..
                                    }
                                ) {
                                    return Some(id);
                                }
                            }
                        }
                        return graph.find_by_name(name).next();
                    }
                }
            }
            current = n.parent();
        }

        None
    }
}

/// Recognized Rust effect-producing function calls.
pub fn is_rust_effect_call(name: &str) -> Option<EffectKind> {
    match name {
        // IO
        "println" | "print" => Some(EffectKind::Stdout),
        "eprintln" | "eprint" => Some(EffectKind::Stderr),
        "write" | "write_all" | "write_fmt" => Some(EffectKind::FileWrite { path: None }),

        // Network
        "send" | "request" | "get" | "post" | "put" | "delete" => {
            Some(EffectKind::NetworkRequest {
                kind: crate::graph::NetworkKind::Http,
            })
        }

        // Process
        "spawn" | "exec" | "Command" => Some(EffectKind::ProcessSpawn { command: None }),
        "exit" => Some(EffectKind::Exit { code: None }),

        // Panic
        "panic" | "unwrap" | "expect" => Some(EffectKind::Panic),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_main() {
        let adapter = RustAdapter::new();
        let source = r#"
fn main() {
    println!("Hello");
}
"#;

        let tree = adapter.parse(source).unwrap();
        let nodes = adapter
            .extract_nodes(&tree, source, Path::new("test.rs"))
            .unwrap();

        assert!(nodes
            .iter()
            .any(|n| matches!(n, Node::EntryPoint { kind: EntryKind::Main, .. })));
    }

    #[test]
    fn test_extract_function() {
        let adapter = RustAdapter::new();
        let source = r#"
fn foo(x: i32) -> i32 {
    x + 1
}

pub async fn bar() {
    println!("bar");
}
"#;

        let tree = adapter.parse(source).unwrap();
        let nodes = adapter
            .extract_nodes(&tree, source, Path::new("test.rs"))
            .unwrap();

        let foo = nodes.iter().find(|n| n.name() == "foo");
        assert!(foo.is_some());
        if let Some(Node::Function { is_async, .. }) = foo {
            assert!(!is_async);
        }

        let bar = nodes.iter().find(|n| n.name() == "bar");
        assert!(bar.is_some());
        if let Some(Node::Function {
            is_async,
            is_public,
            ..
        }) = bar
        {
            assert!(*is_async);
            assert!(*is_public);
        }
    }

    #[test]
    fn test_extract_effects() {
        let adapter = RustAdapter::new();
        let source = r#"
fn main() {
    println!("Hello");
    eprintln!("Error");
}
"#;

        let tree = adapter.parse(source).unwrap();
        let nodes = adapter
            .extract_nodes(&tree, source, Path::new("test.rs"))
            .unwrap();

        let stdout_effects = nodes
            .iter()
            .filter(|n| matches!(n, Node::Effect { kind: EffectKind::Stdout, .. }))
            .count();
        assert_eq!(stdout_effects, 1);

        let stderr_effects = nodes
            .iter()
            .filter(|n| matches!(n, Node::Effect { kind: EffectKind::Stderr, .. }))
            .count();
        assert_eq!(stderr_effects, 1);
    }
}
