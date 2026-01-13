//! Fetch tool documentation from web sources
//!
//! This binary fetches official documentation for verification tools
//! from the URLs stored in their JSON knowledge entries.
//!
//! Usage:
//!   cargo run -p dashprove-knowledge --bin fetch_docs -- \[OPTIONS\]
//!
//! Options:
//!   --all       Fetch docs for all tools with documentation URLs
//!   --priority  Fetch docs for priority tools only (default)
//!   --tool ID   Fetch docs for a specific tool

use dashprove_knowledge::{DocFetcher, ToolKnowledgeStore};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mode = if args.contains(&"--all".to_string()) {
        FetchMode::All
    } else if let Some(pos) = args.iter().position(|a| a == "--tool") {
        if let Some(tool_id) = args.get(pos + 1) {
            FetchMode::Single(tool_id.clone())
        } else {
            eprintln!("Error: --tool requires a tool ID");
            std::process::exit(1);
        }
    } else {
        FetchMode::Priority
    };

    // Load tool knowledge
    let tools_dir = PathBuf::from("data/knowledge/tools");
    if !tools_dir.exists() {
        eprintln!("Error: {} does not exist", tools_dir.display());
        eprintln!("Run from the dashprove project root directory");
        std::process::exit(1);
    }

    info!("Loading tool knowledge from {}", tools_dir.display());
    let store = ToolKnowledgeStore::load_from_dir(&tools_dir).await?;
    info!("Loaded {} tools", store.len());

    // Setup output directory
    let output_dir = PathBuf::from("data/knowledge/docs");
    tokio::fs::create_dir_all(&output_dir).await?;

    let fetcher = DocFetcher::new(output_dir.clone());

    // Fetch based on mode
    let results = match mode {
        FetchMode::All => {
            info!("Fetching documentation for all tools...");
            fetcher.fetch_all(&store).await
        }
        FetchMode::Priority => {
            info!("Fetching documentation for priority tools...");
            let priority_tools = [
                "kani",
                "verus",
                "creusot",
                "prusti",
                "miri", // Rust
                "lean4",
                "coq",
                "isabelle",
                "dafny",
                "agda", // Theorem provers
                "z3",
                "cvc5",
                "yices", // SMT
                "tlaplus",
                "spin",
                "cbmc", // Model checkers
                "marabou",
                "alphabetacrown", // Neural network
                "tamarin",
                "proverif", // Security
            ];
            fetcher.fetch_tools(&store, &priority_tools).await
        }
        FetchMode::Single(tool_id) => {
            info!("Fetching documentation for {}...", tool_id);
            fetcher.fetch_tools(&store, &[tool_id.as_str()]).await
        }
    };

    // Print summary
    println!("\n=== Documentation Fetch Summary ===\n");

    let mut total_docs = 0;
    let mut total_errors = 0;
    let mut successful_tools = 0;

    for result in &results {
        if result.is_success() {
            successful_tools += 1;
        }
        total_docs += result.doc_count();
        total_errors += result.errors.len();

        if !result.fetched.is_empty() || !result.errors.is_empty() {
            println!(
                "{}: {} docs, {} errors",
                result.tool_id,
                result.fetched.len(),
                result.errors.len()
            );
            for meta in &result.fetched {
                println!("  ✓ {} ({} words)", meta.doc_type, meta.word_count);
            }
            for error in &result.errors {
                println!("  ✗ {}", error);
            }
        }
    }

    println!("\n=== Total ===");
    println!("Tools processed: {}", results.len());
    println!("Successful tools: {}", successful_tools);
    println!("Total documents: {}", total_docs);
    println!("Total errors: {}", total_errors);
    println!("Output directory: {}", output_dir.display());

    Ok(())
}

enum FetchMode {
    All,
    Priority,
    Single(String),
}
