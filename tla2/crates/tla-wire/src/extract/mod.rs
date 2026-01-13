//! Language-agnostic code extraction using tree-sitter.
//!
//! This module defines the `LanguageAdapter` trait that each supported language
//! implements to extract wiring graph nodes and edges from source code.

use crate::graph::{Edge, Node, WiringGraph};
use std::path::Path;
use thiserror::Error;

#[cfg(feature = "rust")]
pub mod rust;

#[cfg(feature = "typescript")]
pub mod typescript;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "go")]
pub mod go_lang;

/// Errors that can occur during extraction.
#[derive(Debug, Error)]
pub enum ExtractError {
    #[error("Failed to parse source: {0}")]
    ParseError(String),

    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Tree-sitter error: {0}")]
    TreeSitterError(String),
}

/// Trait for language-specific extraction adapters.
///
/// Each supported language implements this trait to extract nodes and edges
/// from source code using tree-sitter.
pub trait LanguageAdapter: Send + Sync {
    /// The language name (e.g., "rust", "typescript").
    fn language(&self) -> &'static str;

    /// File extensions this adapter handles.
    fn extensions(&self) -> &'static [&'static str];

    /// Get the tree-sitter language for parsing.
    fn tree_sitter_language(&self) -> tree_sitter::Language;

    /// Extract all nodes from a parsed source file.
    fn extract_nodes(
        &self,
        tree: &tree_sitter::Tree,
        source: &str,
        path: &Path,
    ) -> Result<Vec<Node>, ExtractError>;

    /// Extract all edges from a parsed source file.
    /// This is called after nodes have been added to the graph.
    fn extract_edges(
        &self,
        tree: &tree_sitter::Tree,
        source: &str,
        path: &Path,
        graph: &WiringGraph,
    ) -> Result<Vec<Edge>, ExtractError>;

    /// Check if this adapter can handle the given file.
    fn can_handle(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| self.extensions().contains(&e))
            .unwrap_or(false)
    }

    /// Parse source code into a tree-sitter tree.
    fn parse(&self, source: &str) -> Result<tree_sitter::Tree, ExtractError> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&self.tree_sitter_language())
            .map_err(|e| ExtractError::TreeSitterError(e.to_string()))?;

        parser
            .parse(source, None)
            .ok_or_else(|| ExtractError::ParseError("Failed to parse source".to_string()))
    }
}

/// Registry of language adapters.
pub struct AdapterRegistry {
    adapters: Vec<Box<dyn LanguageAdapter>>,
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            adapters: Vec::new(),
        };

        // Register available adapters based on features
        #[cfg(feature = "rust")]
        registry.register(Box::new(rust::RustAdapter::new()));

        #[cfg(feature = "typescript")]
        {
            registry.register(Box::new(typescript::TypeScriptAdapter::new()));
            registry.register(Box::new(typescript::JavaScriptAdapter::new()));
        }

        #[cfg(feature = "python")]
        registry.register(Box::new(python::PythonAdapter::new()));

        #[cfg(feature = "go")]
        registry.register(Box::new(go_lang::GoAdapter::new()));

        registry
    }

    /// Register a new language adapter.
    pub fn register(&mut self, adapter: Box<dyn LanguageAdapter>) {
        self.adapters.push(adapter);
    }

    /// Find an adapter for the given file path.
    pub fn adapter_for(&self, path: &Path) -> Option<&dyn LanguageAdapter> {
        self.adapters
            .iter()
            .find(|a| a.can_handle(path))
            .map(|a| a.as_ref())
    }

    /// Get all registered adapters.
    pub fn adapters(&self) -> impl Iterator<Item = &dyn LanguageAdapter> {
        self.adapters.iter().map(|a| a.as_ref())
    }

    /// Check if any adapter can handle the given file.
    pub fn can_handle(&self, path: &Path) -> bool {
        self.adapters.iter().any(|a| a.can_handle(path))
    }
}

/// Extract a wiring graph from a project directory.
pub struct ProjectExtractor {
    registry: AdapterRegistry,
}

impl Default for ProjectExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl ProjectExtractor {
    pub fn new() -> Self {
        Self {
            registry: AdapterRegistry::new(),
        }
    }

    pub fn with_registry(registry: AdapterRegistry) -> Self {
        Self { registry }
    }

    /// Extract a wiring graph from a single file.
    pub fn extract_file(&self, path: &Path) -> Result<WiringGraph, ExtractError> {
        let adapter = self
            .registry
            .adapter_for(path)
            .ok_or_else(|| ExtractError::UnsupportedLanguage(format!("{:?}", path.extension())))?;

        let source = std::fs::read_to_string(path)?;
        let tree = adapter.parse(&source)?;

        let mut graph = WiringGraph::new();

        // First pass: extract nodes
        let nodes = adapter.extract_nodes(&tree, &source, path)?;
        for node in nodes {
            graph.add_node(node);
        }

        // Second pass: extract edges (needs node IDs)
        let edges = adapter.extract_edges(&tree, &source, path, &graph)?;
        for edge in edges {
            graph.add_edge(edge);
        }

        Ok(graph)
    }

    /// Extract a wiring graph from a project directory.
    pub fn extract_project(&self, root: &Path) -> Result<WiringGraph, ExtractError> {
        let mut graph = WiringGraph::new();
        self.extract_recursive(root, &mut graph)?;
        Ok(graph)
    }

    fn extract_recursive(&self, dir: &Path, graph: &mut WiringGraph) -> Result<(), ExtractError> {
        if !dir.is_dir() {
            return Ok(());
        }

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            // Skip common non-source directories
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if matches!(
                    name,
                    "node_modules"
                        | "target"
                        | ".git"
                        | "dist"
                        | "build"
                        | "__pycache__"
                        | ".venv"
                        | "vendor"
                ) {
                    continue;
                }
                self.extract_recursive(&path, graph)?;
            } else if self.registry.can_handle(&path) {
                match self.extract_file_into(&path, graph) {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::warn!("Failed to extract {:?}: {}", path, e);
                    }
                }
            }
        }

        Ok(())
    }

    fn extract_file_into(&self, path: &Path, graph: &mut WiringGraph) -> Result<(), ExtractError> {
        let adapter = self
            .registry
            .adapter_for(path)
            .ok_or_else(|| ExtractError::UnsupportedLanguage(format!("{:?}", path.extension())))?;

        let source = std::fs::read_to_string(path)?;
        let tree = adapter.parse(&source)?;

        // Extract nodes
        let nodes = adapter.extract_nodes(&tree, &source, path)?;
        for node in nodes {
            graph.add_node(node);
        }

        // Extract edges
        let edges = adapter.extract_edges(&tree, &source, path, graph)?;
        for edge in edges {
            graph.add_edge(edge);
        }

        Ok(())
    }
}
