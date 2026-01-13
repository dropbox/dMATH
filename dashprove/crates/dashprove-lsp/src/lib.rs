//! Language Server Protocol implementation for USL
//!
//! This module provides an LSP server for the Unified Specification Language (USL),
//! enabling IDE features like:
//! - Syntax error diagnostics
//! - Type error diagnostics
//! - Hover information for properties and types
//! - Go-to-definition for type references
//! - Completion suggestions for keywords and types
//! - Document symbols for types and properties
//! - Workspace symbols for cross-file symbol search
//! - Find references for identifiers
//! - Rename refactoring for user-defined types and properties
//! - Semantic tokens (full and range) for syntax coloring
//! - Code actions for quick fixes (create missing types, fix field names)
//! - Folding ranges for collapsible blocks (types, properties, comments)
//! - Inlay hints for type annotations and return types
//! - Signature help for contract parameter hints while typing
//! - Document formatting for consistent code style
//! - Code lenses for verification actions on properties
//! - Execute command for triggering verification from code lenses
//! - Selection range for syntax-aware selection expansion
//! - Call hierarchy for type and property reference navigation
//! - Linked editing ranges for simultaneous identifier editing
//! - Moniker provider for stable symbol identity across workspaces
//!
//! # Usage
//!
//! Run the LSP server with:
//! ```bash
//! dashprove-lsp
//! ```
//!
//! Or use as a library:
//! ```rust,ignore
//! use dashprove_lsp::run_server;
//! run_server().await;
//! ```

mod backend;
mod call_hierarchy;
mod capabilities;
mod code_actions;
mod code_lens;
mod commands;
mod diagnostics;
mod document;
mod folding;
mod formatter;
mod info;
mod inlay_hints;
mod linked_editing;
mod moniker;
mod selection_range;
mod semantic_tokens;
mod signature_help;
mod symbols;

pub use backend::UslLanguageServer;
pub use capabilities::server_capabilities;
pub use code_lens::{
    generate_all_code_lenses, generate_document_stats_lenses, generate_workspace_stats_lenses,
    DocumentStats, PropertyCounts, WorkspaceStats,
};
pub use commands::{
    COMMAND_ANALYZE_WORKSPACE, COMMAND_COMPILATION_GUIDANCE, COMMAND_EXPLAIN_DIAGNOSTIC,
    COMMAND_RECOMMEND_BACKEND, COMMAND_SHOW_BACKEND_INFO, COMMAND_SUGGEST_TACTICS, COMMAND_VERIFY,
    SUPPORTED_COMMANDS,
};
pub use document::{Document, DocumentStore};

use tower_lsp::{LspService, Server};

/// Run the LSP server on stdin/stdout.
///
/// # Errors
///
/// Returns an error if the server fails to start or encounters an I/O error.
pub async fn run_server() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(UslLanguageServer::new);
    Server::new(stdin, stdout, socket).serve(service).await;

    Ok(())
}
