//! DashProve LSP Server
//!
//! Language Server Protocol implementation for USL (Unified Specification Language).
//!
//! This binary starts an LSP server that communicates via stdin/stdout,
//! suitable for use with any LSP-compatible editor (VS Code, Neovim, Emacs, etc.).

use dashprove_lsp::run_server;

#[tokio::main]
async fn main() {
    // Initialize tracing for logging (to stderr to not interfere with LSP)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    tracing::info!("Starting DashProve LSP server");

    if let Err(e) = run_server().await {
        tracing::error!("LSP server error: {}", e);
        std::process::exit(1);
    }
}
