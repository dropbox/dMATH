//! Lean5 Language Server Protocol implementation
//!
//! This crate provides an LSP server for Lean5, enabling IDE support for:
//! - Real-time parse error diagnostics
//! - Type error diagnostics
//! - Hover information (types, documentation)
//! - Go to definition
//! - Find references
//! - Document symbols
//! - Code completion
//!
//! # Architecture
//!
//! The server uses `tower-lsp` for the LSP framework with async/await.
//! Document state is managed incrementally using `ropey` for efficient
//! text rope operations.
//!
//! # Example
//!
//! ```ignore
//! use lean5_lsp::run_server;
//!
//! #[tokio::main]
//! async fn main() {
//!     run_server().await;
//! }
//! ```

pub mod backend;
pub mod diagnostics;
pub mod document;

pub use backend::Lean5Backend;
pub use document::Document;

use tower_lsp::{LspService, Server};

/// Run the LSP server on stdin/stdout
pub async fn run_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(Lean5Backend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}

/// Run the LSP server on a TCP socket (for testing)
pub async fn run_server_tcp(addr: &str) -> Result<(), std::io::Error> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("LSP server listening on {}", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let (read, write) = tokio::io::split(stream);

        let (service, socket) = LspService::new(Lean5Backend::new);
        tokio::spawn(async move {
            Server::new(read, write, socket).serve(service).await;
        });
    }
}

// Tests are in the individual modules
