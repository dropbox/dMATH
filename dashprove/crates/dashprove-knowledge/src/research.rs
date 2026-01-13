//! Research library - ArXiv papers and GitHub repositories

use crate::types::{ArxivPaper, GithubRepo};
use crate::{ArxivConfig, GithubConfig, KnowledgeError, Result};
use chrono::{DateTime, Utc};
use governor::{Quota, RateLimiter};
use reqwest::Client;
use serde::Deserialize;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, info, warn};

/// ArXiv paper fetcher
pub struct ArxivFetcher {
    client: Client,
    config: ArxivConfig,
    rate_limiter: Arc<
        RateLimiter<
            governor::state::NotKeyed,
            governor::state::InMemoryState,
            governor::clock::DefaultClock,
        >,
    >,
    output_dir: PathBuf,
}

impl ArxivFetcher {
    /// Create a new ArXiv fetcher
    pub fn new(config: ArxivConfig, output_dir: PathBuf) -> Self {
        // ArXiv rate limit: 1 request per 3 seconds
        let quota = Quota::per_second(NonZeroU32::new(1).unwrap());
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        Self {
            client: Client::builder()
                .user_agent("DashProve Research Collector/0.1 (https://github.com/dashprove)")
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("Failed to create HTTP client"),
            config,
            rate_limiter,
            output_dir,
        }
    }

    /// Fetch papers for a search query
    pub async fn fetch_papers(&self, query: &str, max_results: usize) -> Result<Vec<ArxivPaper>> {
        self.rate_limiter.until_ready().await;

        // ArXiv API query
        let url = format!(
            "http://export.arxiv.org/api/query?search_query={}&start=0&max_results={}&sortBy=submittedDate&sortOrder=descending",
            urlencoding::encode(query),
            max_results
        );

        info!("Fetching ArXiv papers for query: {}", query);
        debug!("ArXiv URL: {}", url);

        let response = self.client.get(&url).send().await?;
        let body = response.text().await?;

        // Parse Atom feed
        self.parse_arxiv_response(&body)
    }

    /// Fetch papers by category since a date
    pub async fn fetch_by_category(&self, category: &str) -> Result<Vec<ArxivPaper>> {
        let query = format!(
            "cat:{} AND submittedDate:[{} TO *]",
            category, self.config.start_date
        );
        self.fetch_papers(&query, self.config.max_results_per_category)
            .await
    }

    /// Fetch all papers for configured categories
    pub async fn fetch_all(&self) -> Result<Vec<ArxivPaper>> {
        let mut all_papers = Vec::new();

        for category in &self.config.categories {
            match self.fetch_by_category(category).await {
                Ok(papers) => {
                    info!("Found {} papers in category {}", papers.len(), category);
                    all_papers.extend(papers);
                }
                Err(e) => {
                    warn!("Failed to fetch papers for {}: {}", category, e);
                }
            }
            // Extra delay between categories
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }

        // Deduplicate by arxiv_id
        all_papers.sort_by(|a, b| a.arxiv_id.cmp(&b.arxiv_id));
        all_papers.dedup_by(|a, b| a.arxiv_id == b.arxiv_id);

        Ok(all_papers)
    }

    /// Parse ArXiv Atom response
    fn parse_arxiv_response(&self, xml: &str) -> Result<Vec<ArxivPaper>> {
        let mut papers = Vec::new();

        // Simple XML parsing (in production, use quick-xml or similar)
        // This is a simplified parser for the ArXiv Atom feed

        let entries: Vec<&str> = xml.split("<entry>").skip(1).collect();

        for entry in entries {
            if let Some(paper) = self.parse_entry(entry) {
                papers.push(paper);
            }
        }

        Ok(papers)
    }

    /// Parse a single ArXiv entry
    fn parse_entry(&self, entry: &str) -> Option<ArxivPaper> {
        let arxiv_id = extract_tag(entry, "id")?
            .split('/')
            .next_back()?
            .to_string();

        let title = extract_tag(entry, "title")?
            .replace('\n', " ")
            .trim()
            .to_string();

        let abstract_text = extract_tag(entry, "summary")?
            .replace('\n', " ")
            .trim()
            .to_string();

        let published = extract_tag(entry, "published")
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        let updated = extract_tag(entry, "updated")
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        // Extract authors
        let authors: Vec<String> = entry
            .split("<author>")
            .skip(1)
            .filter_map(|a| extract_tag(a, "name"))
            .collect();

        // Extract categories
        let primary_category = entry
            .split("arxiv:primary_category")
            .nth(1)
            .and_then(|s| s.split("term=\"").nth(1))
            .and_then(|s| s.split('"').next())
            .unwrap_or("cs.LO")
            .to_string();

        let categories: Vec<String> = entry
            .split("<category")
            .skip(1)
            .filter_map(|c| {
                c.split("term=\"")
                    .nth(1)
                    .and_then(|s| s.split('"').next())
                    .map(String::from)
            })
            .collect();

        let pdf_url = format!("https://arxiv.org/pdf/{}.pdf", arxiv_id);

        let comment = extract_tag(entry, "arxiv:comment");

        Some(ArxivPaper {
            arxiv_id,
            title,
            authors,
            abstract_text,
            primary_category,
            categories,
            published,
            updated,
            pdf_url,
            comment,
            local_path: None,
            extracted_text: None,
            tags: Vec::new(),
        })
    }

    /// Save papers to disk
    pub async fn save_papers(&self, papers: &[ArxivPaper]) -> Result<()> {
        fs::create_dir_all(&self.output_dir).await?;

        // Save index
        let index_path = self.output_dir.join("index.json");
        let index_json = serde_json::to_string_pretty(papers)?;
        fs::write(&index_path, index_json).await?;

        // Save individual papers
        for paper in papers {
            let year = paper.published.format("%Y").to_string();
            let category_dir = self.output_dir.join(&year).join(&paper.primary_category);
            fs::create_dir_all(&category_dir).await?;

            let paper_path = category_dir.join(format!("{}.json", paper.arxiv_id));
            let paper_json = serde_json::to_string_pretty(paper)?;
            fs::write(&paper_path, paper_json).await?;
        }

        info!("Saved {} papers to {:?}", papers.len(), self.output_dir);
        Ok(())
    }
}

/// GitHub repository searcher
pub struct GithubSearcher {
    client: Client,
    config: GithubConfig,
    rate_limiter: Arc<
        RateLimiter<
            governor::state::NotKeyed,
            governor::state::InMemoryState,
            governor::clock::DefaultClock,
        >,
    >,
    output_dir: PathBuf,
}

impl GithubSearcher {
    /// Create a new GitHub searcher
    pub fn new(config: GithubConfig, output_dir: PathBuf) -> Self {
        // GitHub rate limit: 10 requests per minute for unauthenticated
        let quota = Quota::per_minute(NonZeroU32::new(10).unwrap());
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        Self {
            client: Client::builder()
                .user_agent("DashProve Research Collector/0.1")
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            config,
            rate_limiter,
            output_dir,
        }
    }

    /// Search for repositories
    pub async fn search(&self, query: &str) -> Result<Vec<GithubRepo>> {
        self.rate_limiter.until_ready().await;

        let full_query = format!(
            "{} stars:>{} created:>2024-01-01",
            query, self.config.min_stars
        );

        let url = format!(
            "https://api.github.com/search/repositories?q={}&sort=stars&order=desc&per_page=30",
            urlencoding::encode(&full_query)
        );

        info!("Searching GitHub for: {}", query);

        let mut request = self.client.get(&url);
        if let Some(ref token) = self.config.api_token {
            request = request.header("Authorization", format!("token {}", token));
        }

        let response = request.send().await?;

        if response.status() == 403 {
            return Err(KnowledgeError::RateLimited(60));
        }

        let data: GithubSearchResponse = response.json().await?;
        Ok(data.items.into_iter().map(|item| item.into()).collect())
    }

    /// Search all configured queries
    pub async fn search_all(&self) -> Result<Vec<GithubRepo>> {
        let mut all_repos = Vec::new();

        for query in &self.config.queries.clone() {
            match self.search(query).await {
                Ok(repos) => {
                    info!("Found {} repos for query: {}", repos.len(), query);
                    all_repos.extend(repos);
                }
                Err(KnowledgeError::RateLimited(secs)) => {
                    warn!("Rate limited, waiting {}s", secs);
                    tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
                }
                Err(e) => {
                    warn!("Failed to search for {}: {}", query, e);
                }
            }
        }

        // Deduplicate
        all_repos.sort_by(|a, b| a.full_name.cmp(&b.full_name));
        all_repos.dedup_by(|a, b| a.full_name == b.full_name);

        // Sort by stars
        all_repos.sort_by(|a, b| b.stars.cmp(&a.stars));

        Ok(all_repos)
    }

    /// Save repositories to disk
    pub async fn save_repos(&self, repos: &[GithubRepo]) -> Result<()> {
        fs::create_dir_all(&self.output_dir).await?;

        let repos_path = self.output_dir.join("repos.json");
        let repos_json = serde_json::to_string_pretty(repos)?;
        fs::write(&repos_path, repos_json).await?;

        info!("Saved {} repos to {:?}", repos.len(), self.output_dir);
        Ok(())
    }
}

// GitHub API response types
#[derive(Debug, Deserialize)]
struct GithubSearchResponse {
    items: Vec<GithubRepoItem>,
}

#[derive(Debug, Deserialize)]
struct GithubRepoItem {
    full_name: String,
    description: Option<String>,
    stargazers_count: usize,
    forks_count: usize,
    language: Option<String>,
    topics: Option<Vec<String>>,
    html_url: String,
    updated_at: String,
    created_at: String,
}

impl From<GithubRepoItem> for GithubRepo {
    fn from(item: GithubRepoItem) -> Self {
        Self {
            full_name: item.full_name,
            description: item.description,
            stars: item.stargazers_count,
            forks: item.forks_count,
            language: item.language,
            topics: item.topics.unwrap_or_default(),
            url: item.html_url,
            updated_at: DateTime::parse_from_rfc3339(&item.updated_at)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            created_at: DateTime::parse_from_rfc3339(&item.created_at)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            readme: None,
            relevance_score: 0.0,
        }
    }
}

/// Helper to extract XML tag content
fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{}>", tag);
    let end_tag = format!("</{}>", tag);

    let start = xml.find(&start_tag)? + start_tag.len();
    let end = xml.find(&end_tag)?;

    if start < end {
        Some(xml[start..end].to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tag() {
        let xml = "<title>Test Title</title>";
        assert_eq!(extract_tag(xml, "title"), Some("Test Title".to_string()));
    }

    #[test]
    fn test_extract_tag_missing() {
        let xml = "<other>Content</other>";
        assert_eq!(extract_tag(xml, "title"), None);
    }
}
