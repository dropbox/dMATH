//! Lean5 LSP Server Binary
//!
//! Run with: cargo run -p lean5-lsp
//!
//! Communicates via stdin/stdout using the LSP protocol.

#[tokio::main]
async fn main() {
    lean5_lsp::run_server().await;
}
