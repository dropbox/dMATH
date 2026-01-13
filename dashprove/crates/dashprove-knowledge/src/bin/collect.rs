//! Knowledge collection CLI
//!
//! Downloads documentation, fetches ArXiv papers, and searches GitHub
//! for cutting-edge formal verification research.

use dashprove_knowledge::{
    get_all_backend_sources, ArxivConfig, ArxivFetcher, DocumentationCollector, GithubConfig,
    GithubSearcher, KnowledgeConfig,
};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let config = KnowledgeConfig::default();
    let base_dir = config.base_dir.clone();

    info!("Knowledge collection starting");
    info!("Base directory: {:?}", base_dir);

    // Phase 1: Collect backend documentation
    info!("=== Phase 1: Backend Documentation ===");
    let docs_dir = base_dir.join("docs");
    let collector = DocumentationCollector::new(docs_dir.clone());

    let sources = get_all_backend_sources();
    info!("Collecting documentation for {} backends", sources.len());

    for source in &sources {
        info!("Collecting: {} ({})", source.name, source.docs_url);
        match collector.collect_backend(source).await {
            Ok(docs) => {
                info!("  -> {} documents collected", docs.len());
                if let Err(e) = collector.save_documents(&docs).await {
                    tracing::warn!("  -> Failed to save: {}", e);
                }
            }
            Err(e) => {
                tracing::warn!("  -> Failed: {}", e);
            }
        }
        // Rate limit between backends
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }

    // Phase 2: Fetch ArXiv papers
    info!("=== Phase 2: ArXiv Research Papers ===");
    let arxiv_dir = base_dir.join("arxiv");
    let arxiv_config = ArxivConfig {
        categories: vec![
            "cs.LO".to_string(), // Logic in CS
            "cs.PL".to_string(), // Programming Languages
            "cs.SE".to_string(), // Software Engineering
            "cs.FL".to_string(), // Formal Languages
        ],
        start_date: "2024-01-01".to_string(),
        max_results_per_category: 50,
    };
    let arxiv = ArxivFetcher::new(arxiv_config, arxiv_dir);

    // Focused queries for formal verification
    let queries = vec![
        "ti:verification OR ti:proof assistant",
        "ti:lean4 OR ti:lean prover",
        "ti:coq proof OR ti:isabelle",
        "ti:SMT solver OR ti:SAT solver",
        "ti:neural network verification",
        "ti:rust verification OR ti:kani",
        "ti:formal methods AI",
        "ti:program synthesis verification",
    ];

    let mut all_papers = Vec::new();
    for query in queries {
        info!("ArXiv query: {}", query);
        match arxiv.fetch_papers(query, 30).await {
            Ok(papers) => {
                info!("  -> {} papers found", papers.len());
                all_papers.extend(papers);
            }
            Err(e) => {
                tracing::warn!("  -> Query failed: {}", e);
            }
        }
        // ArXiv rate limit: 3 seconds between requests
        tokio::time::sleep(std::time::Duration::from_secs(4)).await;
    }

    // Deduplicate papers
    all_papers.sort_by(|a, b| a.arxiv_id.cmp(&b.arxiv_id));
    all_papers.dedup_by(|a, b| a.arxiv_id == b.arxiv_id);
    info!("Total unique ArXiv papers: {}", all_papers.len());

    if let Err(e) = arxiv.save_papers(&all_papers).await {
        tracing::warn!("Failed to save papers: {}", e);
    }

    // Phase 3: Search GitHub
    info!("=== Phase 3: GitHub Repositories ===");
    let github_dir = base_dir.join("github");
    let github_config = GithubConfig {
        api_token: std::env::var("GITHUB_TOKEN").ok(),
        min_stars: 10,
        queries: vec![
            "formal verification rust".to_string(),
            "proof assistant".to_string(),
            "theorem prover".to_string(),
            "SMT solver".to_string(),
            "lean4".to_string(),
            "coq proof".to_string(),
            "kani verifier".to_string(),
            "neural network verification".to_string(),
            "program synthesis".to_string(),
        ],
    };
    let github = GithubSearcher::new(github_config, github_dir);

    match github.search_all().await {
        Ok(repos) => {
            info!("Found {} GitHub repositories", repos.len());
            if let Err(e) = github.save_repos(&repos).await {
                tracing::warn!("Failed to save repos: {}", e);
            }
        }
        Err(e) => {
            tracing::warn!("GitHub search failed: {}", e);
        }
    }

    info!("=== Collection Complete ===");
    info!("Documentation: {:?}", docs_dir);
    info!("ArXiv papers: {}", all_papers.len());
    info!("Results saved to: {:?}", base_dir);

    Ok(())
}
