//! CLI command for research paper fetching and management
//!
//! This module provides commands for:
//! - Fetching papers from ArXiv
//! - Downloading and extracting PDFs
//! - Searching the paper corpus
//! - Managing periodic fetching

use anyhow::Result;
use dashprove_knowledge::{ArxivConfig, ArxivFetcher, GithubConfig, GithubSearcher, PdfProcessor};
use std::path::PathBuf;

/// Configuration for ArXiv fetch command
pub struct ArxivFetchConfig<'a> {
    /// Output directory for fetched papers
    pub output_dir: Option<&'a str>,
    /// Categories to fetch (comma-separated)
    pub categories: Option<&'a str>,
    /// Start date for papers (YYYY-MM-DD)
    pub since: Option<&'a str>,
    /// Maximum papers per category
    pub max_per_category: usize,
    /// Whether to download PDFs
    pub download_pdfs: bool,
    /// Whether to extract text from PDFs
    pub extract_text: bool,
    /// Show verbose output
    pub verbose: bool,
}

/// Configuration for paper search command
pub struct PaperSearchConfig<'a> {
    /// Search query
    pub query: &'a str,
    /// Data directory containing paper index
    pub data_dir: Option<&'a str>,
    /// Maximum number of results
    pub limit: usize,
    /// Filter by category
    pub category: Option<&'a str>,
    /// Output format (text, json)
    pub format: &'a str,
}

/// Configuration for GitHub fetch command
pub struct GithubFetchConfig<'a> {
    /// Output directory for fetched repositories
    pub output_dir: Option<&'a str>,
    /// Search queries (comma-separated)
    pub queries: Option<&'a str>,
    /// Minimum stars filter
    pub min_stars: usize,
    /// Show verbose output
    pub verbose: bool,
}

/// Run the ArXiv fetch command
pub async fn run_arxiv_fetch(config: ArxivFetchConfig<'_>) -> Result<()> {
    let output_dir = config
        .output_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| default_data_dir().join("papers"));

    // Build ArXiv configuration
    let arxiv_config = ArxivConfig {
        categories: config
            .categories
            .map(|c| c.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_else(|| ArxivConfig::default().categories),
        start_date: config
            .since
            .map(String::from)
            .unwrap_or_else(|| ArxivConfig::default().start_date),
        max_results_per_category: config.max_per_category,
    };

    if config.verbose {
        println!("ArXiv Fetch Configuration:");
        println!("  Output directory: {}", output_dir.display());
        println!("  Categories: {:?}", arxiv_config.categories);
        println!("  Since: {}", arxiv_config.start_date);
        println!(
            "  Max per category: {}",
            arxiv_config.max_results_per_category
        );
        println!("  Download PDFs: {}", config.download_pdfs);
        println!("  Extract text: {}", config.extract_text);
        println!();
    }

    // Create fetcher and fetch papers
    let fetcher = ArxivFetcher::new(arxiv_config, output_dir.clone());

    println!("Fetching papers from ArXiv...");
    let mut papers = fetcher.fetch_all().await?;
    println!("Found {} papers", papers.len());

    // Save papers
    fetcher.save_papers(&papers).await?;
    println!("Saved paper metadata to {}", output_dir.display());

    // Optionally download PDFs and extract text
    if config.download_pdfs || config.extract_text {
        let pdf_dir = output_dir.join("pdfs");
        let pdf_processor = PdfProcessor::new(pdf_dir.clone());

        println!("\nProcessing PDFs...");
        let stats = pdf_processor.process_papers(&mut papers).await;

        println!("PDF Processing Complete:");
        println!("  Downloaded: {}", stats.downloaded);
        println!("  Extracted text: {}", stats.extracted);
        println!("  Failed: {}", stats.failed);

        // Re-save papers with updated paths and text
        fetcher.save_papers(&papers).await?;
    }

    // Summary
    println!("\nFetch Summary:");
    println!("  Total papers: {}", papers.len());

    // Group by category
    let mut by_category: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for paper in &papers {
        *by_category
            .entry(paper.primary_category.clone())
            .or_default() += 1;
    }
    println!("  By category:");
    let mut categories: Vec<_> = by_category.into_iter().collect();
    categories.sort_by(|a, b| b.1.cmp(&a.1));
    for (cat, count) in categories {
        println!("    {}: {}", cat, count);
    }

    Ok(())
}

/// Run the paper search command
pub async fn run_paper_search(config: PaperSearchConfig<'_>) -> Result<()> {
    let data_dir = config
        .data_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir);

    let papers_dir = data_dir.join("papers");

    // Load the paper index
    let index_path = papers_dir.join("index.json");
    if !index_path.exists() {
        eprintln!("No paper index found at {}", index_path.display());
        eprintln!("Run 'dashprove research arxiv fetch' first to populate the paper corpus.");
        return Ok(());
    }

    let index_content = std::fs::read_to_string(&index_path)?;
    let papers: Vec<dashprove_knowledge::ArxivPaper> = serde_json::from_str(&index_content)?;

    if papers.is_empty() {
        println!("No papers in corpus. Run 'dashprove research arxiv fetch' to populate.");
        return Ok(());
    }

    // Create search engine with embedded papers
    // For now, use simple keyword matching since we don't have embeddings
    let query_lower = config.query.to_lowercase();
    let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

    let mut results: Vec<(f32, &dashprove_knowledge::ArxivPaper)> = papers
        .iter()
        .filter_map(|paper| {
            // Optionally filter by category
            if let Some(cat) = config.category {
                if !paper.categories.iter().any(|c| c.contains(cat)) {
                    return None;
                }
            }

            // Score based on term matches
            let title_lower = paper.title.to_lowercase();
            let abstract_lower = paper.abstract_text.to_lowercase();

            let mut score = 0.0f32;
            for term in &query_terms {
                if title_lower.contains(term) {
                    score += 2.0; // Title matches are more important
                }
                if abstract_lower.contains(term) {
                    score += 1.0;
                }
            }

            if score > 0.0 {
                Some((score, paper))
            } else {
                None
            }
        })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(config.limit);

    match config.format {
        "json" => {
            let json_results: Vec<_> = results
                .iter()
                .map(|(score, paper)| {
                    serde_json::json!({
                        "score": score,
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "abstract": paper.abstract_text,
                        "categories": paper.categories,
                        "published": paper.published,
                        "pdf_url": paper.pdf_url,
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json_results)?);
        }
        _ => {
            if results.is_empty() {
                println!("No papers found matching '{}'", config.query);
            } else {
                println!(
                    "Found {} papers matching '{}':\n",
                    results.len(),
                    config.query
                );
                for (i, (score, paper)) in results.iter().enumerate() {
                    println!("{}. [score: {:.1}] {}", i + 1, score, paper.title);
                    println!("   ArXiv ID: {}", paper.arxiv_id);
                    println!(
                        "   Authors: {}",
                        paper
                            .authors
                            .iter()
                            .take(3)
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    if paper.authors.len() > 3 {
                        print!(" et al.");
                    }
                    println!();
                    println!("   Categories: {}", paper.categories.join(", "));
                    println!("   Published: {}", paper.published.format("%Y-%m-%d"));
                    // Show truncated abstract
                    let abstract_preview: String = paper.abstract_text.chars().take(200).collect();
                    println!("   Abstract: {}...", abstract_preview);
                    println!("   PDF: {}", paper.pdf_url);
                    println!();
                }
            }
        }
    }

    Ok(())
}

/// Run the GitHub fetch command
pub async fn run_github_fetch(config: GithubFetchConfig<'_>) -> Result<()> {
    let output_dir = config
        .output_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| default_data_dir().join("repos"));

    // Build GitHub configuration
    let github_config = GithubConfig {
        api_token: std::env::var("GITHUB_TOKEN").ok(),
        min_stars: config.min_stars,
        queries: config
            .queries
            .map(|q| q.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_else(|| GithubConfig::default().queries),
    };

    if config.verbose {
        println!("GitHub Fetch Configuration:");
        println!("  Output directory: {}", output_dir.display());
        println!("  Queries: {:?}", github_config.queries);
        println!("  Min stars: {}", github_config.min_stars);
        println!(
            "  API token: {}",
            if github_config.api_token.is_some() {
                "present"
            } else {
                "not set (rate limits apply)"
            }
        );
        println!();
    }

    // Create searcher and fetch repos
    let searcher = GithubSearcher::new(github_config, output_dir.clone());

    println!("Searching GitHub for repositories...");
    let repos = searcher.search_all().await?;
    println!("Found {} repositories", repos.len());

    // Save repos
    searcher.save_repos(&repos).await?;
    println!("Saved repository data to {}", output_dir.display());

    // Summary
    println!("\nFetch Summary:");
    println!("  Total repositories: {}", repos.len());

    // Top repos by stars
    println!("\nTop repositories by stars:");
    for repo in repos.iter().take(10) {
        println!(
            "  {} ({} ⭐) - {}",
            repo.full_name,
            repo.stars,
            repo.description.as_deref().unwrap_or("No description")
        );
    }

    Ok(())
}

/// Run the corpus stats command
pub fn run_research_stats(data_dir: Option<&str>) -> Result<()> {
    let data_dir = data_dir.map(PathBuf::from).unwrap_or_else(default_data_dir);

    let papers_dir = data_dir.join("papers");
    let repos_dir = data_dir.join("repos");

    println!("Research Corpus Statistics\n");

    // Papers stats
    let papers_index = papers_dir.join("index.json");
    if papers_index.exists() {
        let content = std::fs::read_to_string(&papers_index)?;
        let papers: Vec<dashprove_knowledge::ArxivPaper> = serde_json::from_str(&content)?;

        println!("ArXiv Papers:");
        println!("  Total papers: {}", papers.len());

        // Count by category
        let mut by_category: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut with_pdf = 0;
        let mut with_text = 0;

        for paper in &papers {
            *by_category
                .entry(paper.primary_category.clone())
                .or_default() += 1;
            if paper.local_path.is_some() {
                with_pdf += 1;
            }
            if paper.extracted_text.is_some() {
                with_text += 1;
            }
        }

        println!("  With downloaded PDF: {}", with_pdf);
        println!("  With extracted text: {}", with_text);
        println!("  By category:");

        let mut categories: Vec<_> = by_category.into_iter().collect();
        categories.sort_by(|a, b| b.1.cmp(&a.1));
        for (cat, count) in categories.iter().take(10) {
            println!("    {}: {}", cat, count);
        }

        // Date range
        if let (Some(oldest), Some(newest)) = (
            papers.iter().map(|p| &p.published).min(),
            papers.iter().map(|p| &p.published).max(),
        ) {
            println!(
                "  Date range: {} to {}",
                oldest.format("%Y-%m-%d"),
                newest.format("%Y-%m-%d")
            );
        }
    } else {
        println!("ArXiv Papers: No index found");
        println!("  Run 'dashprove research arxiv fetch' to populate");
    }

    println!();

    // Repos stats
    let repos_index = repos_dir.join("repos.json");
    if repos_index.exists() {
        let content = std::fs::read_to_string(&repos_index)?;
        let repos: Vec<dashprove_knowledge::GithubRepo> = serde_json::from_str(&content)?;

        println!("GitHub Repositories:");
        println!("  Total repositories: {}", repos.len());

        // Count by language
        let mut by_language: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut total_stars = 0;

        for repo in &repos {
            if let Some(ref lang) = repo.language {
                *by_language.entry(lang.clone()).or_default() += 1;
            }
            total_stars += repo.stars;
        }

        println!("  Total stars: {}", total_stars);
        println!("  By language:");

        let mut languages: Vec<_> = by_language.into_iter().collect();
        languages.sort_by(|a, b| b.1.cmp(&a.1));
        for (lang, count) in languages.iter().take(10) {
            println!("    {}: {}", lang, count);
        }
    } else {
        println!("GitHub Repositories: No index found");
        println!("  Run 'dashprove research github fetch' to populate");
    }

    Ok(())
}

/// Get the default data directory
fn default_data_dir() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".dashprove"))
        .unwrap_or_else(|| PathBuf::from("data/knowledge"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_data_dir() {
        let dir = default_data_dir();
        // Should return a valid path
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn test_arxiv_config_from_options() {
        let config = ArxivFetchConfig {
            output_dir: Some("/tmp/test"),
            categories: Some("cs.LO,cs.PL"),
            since: Some("2024-06-01"),
            max_per_category: 50,
            download_pdfs: false,
            extract_text: false,
            verbose: false,
        };

        assert_eq!(config.max_per_category, 50);
        assert!(config.categories.unwrap().contains("cs.LO"));
    }

    #[test]
    fn test_paper_search_config() {
        let config = PaperSearchConfig {
            query: "formal verification",
            data_dir: None,
            limit: 10,
            category: Some("cs.LO"),
            format: "text",
        };

        assert_eq!(config.query, "formal verification");
        assert_eq!(config.limit, 10);
    }

    #[test]
    fn test_github_fetch_config() {
        let config = GithubFetchConfig {
            output_dir: None,
            queries: Some("theorem prover,smt solver"),
            min_stars: 100,
            verbose: true,
        };

        assert_eq!(config.min_stars, 100);
        assert!(config.verbose);
    }
}
