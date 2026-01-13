//! Web documentation fetcher for tool knowledge
//!
//! This module fetches official documentation from URLs in ToolKnowledge entries
//! and stores them locally in `data/knowledge/docs/`.

use crate::tool_knowledge::{ToolKnowledge, ToolKnowledgeStore};
use crate::types::CodeBlock;
use crate::{KnowledgeError, Result};
use chrono::Utc;
use governor::{Quota, RateLimiter};
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, info, warn};

/// Metadata for fetched documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocMetadata {
    /// Tool ID
    pub tool_id: String,
    /// Tool name
    pub tool_name: String,
    /// Source URL
    pub source_url: String,
    /// Document type (official, tutorial, api_reference, examples)
    pub doc_type: String,
    /// When the document was fetched
    pub fetched_at: String,
    /// HTTP response status
    pub http_status: Option<u16>,
    /// Content type header
    pub content_type_header: Option<String>,
    /// Word count
    pub word_count: usize,
    /// Sections found
    pub sections: Vec<String>,
    /// Code blocks count
    pub code_block_count: usize,
}

/// Web documentation fetcher
pub struct DocFetcher {
    client: Client,
    rate_limiter: Arc<
        RateLimiter<
            governor::state::NotKeyed,
            governor::state::InMemoryState,
            governor::clock::DefaultClock,
        >,
    >,
    output_dir: PathBuf,
}

impl DocFetcher {
    /// Create a new documentation fetcher
    pub fn new(output_dir: PathBuf) -> Self {
        // Rate limit: 5 requests per second to be respectful
        let quota = Quota::per_second(NonZeroU32::new(5).unwrap());
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        Self {
            client: Client::builder()
                .user_agent("DashProve-Knowledge/0.1 (formal-verification-docs)")
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("Failed to create HTTP client"),
            rate_limiter,
            output_dir,
        }
    }

    /// Fetch documentation for a single tool
    pub async fn fetch_tool_docs(&self, tool: &ToolKnowledge) -> Result<FetchResult> {
        let mut result = FetchResult {
            tool_id: tool.id.clone(),
            fetched: Vec::new(),
            errors: Vec::new(),
        };

        let Some(ref docs) = tool.documentation else {
            info!("No documentation URLs for {}", tool.id);
            return Ok(result);
        };

        let tool_dir = self.output_dir.join(&tool.id);
        fs::create_dir_all(&tool_dir).await?;

        // Fetch each documentation type
        if let Some(ref url) = docs.official {
            match self.fetch_and_save(tool, url, "official", &tool_dir).await {
                Ok(meta) => result.fetched.push(meta),
                Err(e) => result.errors.push(format!("official: {}", e)),
            }
        }

        if let Some(ref url) = docs.tutorial {
            match self.fetch_and_save(tool, url, "tutorial", &tool_dir).await {
                Ok(meta) => result.fetched.push(meta),
                Err(e) => result.errors.push(format!("tutorial: {}", e)),
            }
        }

        if let Some(ref url) = docs.api_reference {
            match self
                .fetch_and_save(tool, url, "api_reference", &tool_dir)
                .await
            {
                Ok(meta) => result.fetched.push(meta),
                Err(e) => result.errors.push(format!("api_reference: {}", e)),
            }
        }

        if let Some(ref url) = docs.examples {
            match self.fetch_and_save(tool, url, "examples", &tool_dir).await {
                Ok(meta) => result.fetched.push(meta),
                Err(e) => result.errors.push(format!("examples: {}", e)),
            }
        }

        Ok(result)
    }

    /// Fetch and save a single documentation URL
    async fn fetch_and_save(
        &self,
        tool: &ToolKnowledge,
        url: &str,
        doc_type: &str,
        output_dir: &Path,
    ) -> Result<DocMetadata> {
        info!("Fetching {} docs for {}: {}", doc_type, tool.id, url);

        // Rate limit
        self.rate_limiter.until_ready().await;

        // Fetch the URL
        let response = match self.client.get(url).send().await {
            Ok(r) => r,
            Err(e) => {
                warn!("Failed to fetch {}: {}", url, e);
                return Err(KnowledgeError::HttpError(e));
            }
        };

        let status = response.status();
        let content_type_header = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        if !status.is_success() {
            return Err(KnowledgeError::ParseError(format!(
                "HTTP {} for {}",
                status, url
            )));
        }

        let body = response.text().await?;

        // Process based on content type
        let (markdown, sections, code_blocks) = if content_type_header
            .as_ref()
            .map(|ct| ct.contains("text/html"))
            .unwrap_or(false)
        {
            self.process_html(&body)?
        } else if url.ends_with(".md") || url.ends_with(".markdown") {
            // Already markdown
            (body.clone(), Vec::new(), Vec::new())
        } else {
            // Plain text or unknown - save as-is
            (body.clone(), Vec::new(), Vec::new())
        };

        // Save content as markdown
        let content_filename = format!("{}.md", doc_type);
        let content_path = output_dir.join(&content_filename);
        fs::write(&content_path, &markdown).await?;

        // Create metadata
        let metadata = DocMetadata {
            tool_id: tool.id.clone(),
            tool_name: tool.name.clone(),
            source_url: url.to_string(),
            doc_type: doc_type.to_string(),
            fetched_at: Utc::now().to_rfc3339(),
            http_status: Some(status.as_u16()),
            content_type_header,
            word_count: markdown.split_whitespace().count(),
            sections,
            code_block_count: code_blocks.len(),
        };

        // Save metadata
        let meta_path = output_dir.join(format!("{}.meta.json", doc_type));
        let meta_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(&meta_path, meta_json).await?;

        debug!(
            "Saved {} docs for {} ({} words)",
            doc_type, tool.id, metadata.word_count
        );

        Ok(metadata)
    }

    /// Process HTML to markdown
    fn process_html(&self, html: &str) -> Result<(String, Vec<String>, Vec<CodeBlock>)> {
        let document = Html::parse_document(html);
        let mut sections = Vec::new();
        let mut code_blocks = Vec::new();

        // Extract main content
        let content_selectors = [
            "article",
            "main",
            ".content",
            ".documentation",
            "#content",
            ".markdown-body",
            ".rst-content",
            "#main-content",
            "body",
        ];

        let mut content_html = None;
        for selector_str in content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    content_html = Some(element.html());
                    break;
                }
            }
        }

        let html_content = content_html.unwrap_or_else(|| html.to_string());

        // Convert HTML to text
        let markdown = html2text::from_read(html_content.as_bytes(), 100);

        // Extract code blocks
        if let Ok(selector) = Selector::parse("pre code, .highlight, .codehilite") {
            for element in document.select(&selector) {
                let code = element.text().collect::<String>();
                let lang = element
                    .value()
                    .attr("class")
                    .and_then(|c| {
                        c.split_whitespace()
                            .find(|s| s.starts_with("language-") || s.starts_with("highlight-"))
                    })
                    .map(|s| {
                        s.strip_prefix("language-")
                            .or_else(|| s.strip_prefix("highlight-"))
                            .unwrap_or(s)
                    })
                    .unwrap_or("text")
                    .to_string();

                code_blocks.push(CodeBlock {
                    language: lang,
                    code,
                    description: None,
                });
            }
        }

        // Extract sections
        if let Ok(selector) = Selector::parse("h1, h2, h3") {
            for element in document.select(&selector) {
                let section = element.text().collect::<String>().trim().to_string();
                if !section.is_empty() {
                    sections.push(section);
                }
            }
        }

        Ok((markdown, sections, code_blocks))
    }

    /// Fetch documentation for all tools in a store
    pub async fn fetch_all(&self, store: &ToolKnowledgeStore) -> Vec<FetchResult> {
        let mut results = Vec::new();

        for tool in store.all() {
            match self.fetch_tool_docs(tool).await {
                Ok(result) => {
                    info!(
                        "Fetched {} docs for {} ({} errors)",
                        result.fetched.len(),
                        result.tool_id,
                        result.errors.len()
                    );
                    results.push(result);
                }
                Err(e) => {
                    warn!("Failed to fetch docs for {}: {}", tool.id, e);
                    results.push(FetchResult {
                        tool_id: tool.id.clone(),
                        fetched: Vec::new(),
                        errors: vec![e.to_string()],
                    });
                }
            }
        }

        results
    }

    /// Fetch documentation for specific tool IDs
    pub async fn fetch_tools(
        &self,
        store: &ToolKnowledgeStore,
        tool_ids: &[&str],
    ) -> Vec<FetchResult> {
        let mut results = Vec::new();

        for tool_id in tool_ids {
            if let Some(tool) = store.get(tool_id) {
                match self.fetch_tool_docs(tool).await {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        warn!("Failed to fetch docs for {}: {}", tool_id, e);
                        results.push(FetchResult {
                            tool_id: tool_id.to_string(),
                            fetched: Vec::new(),
                            errors: vec![e.to_string()],
                        });
                    }
                }
            } else {
                warn!("Tool not found: {}", tool_id);
            }
        }

        results
    }
}

/// Result of fetching documentation for a tool
#[derive(Debug, Clone)]
pub struct FetchResult {
    /// Tool ID
    pub tool_id: String,
    /// Successfully fetched documents
    pub fetched: Vec<DocMetadata>,
    /// Errors encountered
    pub errors: Vec<String>,
}

impl FetchResult {
    /// Check if all fetches were successful
    pub fn is_success(&self) -> bool {
        self.errors.is_empty() && !self.fetched.is_empty()
    }

    /// Get total document count
    pub fn doc_count(&self) -> usize {
        self.fetched.len()
    }
}

/// Fetch documentation for priority tools
pub async fn fetch_priority_docs(
    output_dir: &Path,
    store: &ToolKnowledgeStore,
) -> Vec<FetchResult> {
    let fetcher = DocFetcher::new(output_dir.to_path_buf());

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

    fetcher.fetch_tools(store, &priority_tools).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_processing() {
        let fetcher = DocFetcher::new(PathBuf::from("/tmp"));
        let html = r#"
            <html>
            <head><title>Test Doc</title></head>
            <body>
                <main>
                    <h1>Main Title</h1>
                    <h2>Section One</h2>
                    <p>Some content here.</p>
                    <pre><code class="language-rust">fn main() {}</code></pre>
                    <h2>Section Two</h2>
                    <p>More content.</p>
                </main>
            </body>
            </html>
        "#;

        let (markdown, sections, code_blocks) = fetcher.process_html(html).unwrap();

        assert!(markdown.contains("Main Title"));
        assert!(markdown.contains("Section One"));
        assert!(!sections.is_empty());
        assert!(!code_blocks.is_empty());
        assert_eq!(code_blocks[0].language, "rust");
    }

    #[tokio::test]
    async fn test_fetch_result() {
        let result = FetchResult {
            tool_id: "test".to_string(),
            fetched: vec![],
            errors: vec![],
        };

        assert!(!result.is_success()); // No docs fetched
        assert_eq!(result.doc_count(), 0);

        let result_with_doc = FetchResult {
            tool_id: "test".to_string(),
            fetched: vec![DocMetadata {
                tool_id: "test".to_string(),
                tool_name: "Test".to_string(),
                source_url: "https://example.com".to_string(),
                doc_type: "official".to_string(),
                fetched_at: "2025-01-01T00:00:00Z".to_string(),
                http_status: Some(200),
                content_type_header: Some("text/html".to_string()),
                word_count: 100,
                sections: vec![],
                code_block_count: 0,
            }],
            errors: vec![],
        };

        assert!(result_with_doc.is_success());
        assert_eq!(result_with_doc.doc_count(), 1);
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// DocMetadata serialization roundtrips
        #[test]
        fn test_doc_metadata_roundtrip(
            tool_id in "[a-z_]{3,10}",
            tool_name in "[A-Za-z ]{3,20}",
            word_count in 0usize..10000
        ) {
            let metadata = DocMetadata {
                tool_id: tool_id.clone(),
                tool_name,
                source_url: "https://example.com".to_string(),
                doc_type: "official".to_string(),
                fetched_at: "2025-01-01T00:00:00Z".to_string(),
                http_status: Some(200),
                content_type_header: Some("text/html".to_string()),
                word_count,
                sections: vec!["Section 1".to_string()],
                code_block_count: 5,
            };

            let json = serde_json::to_string(&metadata).unwrap();
            let parsed: DocMetadata = serde_json::from_str(&json).unwrap();

            prop_assert_eq!(parsed.tool_id, tool_id);
            prop_assert_eq!(parsed.word_count, word_count);
        }

        /// FetchResult success condition
        #[test]
        fn test_fetch_result_success_condition(
            error_count in 0usize..5,
            fetched_count in 0usize..5
        ) {
            let fetched = (0..fetched_count)
                .map(|i| DocMetadata {
                    tool_id: format!("tool{}", i),
                    tool_name: "Test".to_string(),
                    source_url: "https://example.com".to_string(),
                    doc_type: "official".to_string(),
                    fetched_at: "2025-01-01T00:00:00Z".to_string(),
                    http_status: Some(200),
                    content_type_header: None,
                    word_count: 100,
                    sections: vec![],
                    code_block_count: 0,
                })
                .collect::<Vec<_>>();

            let errors = (0..error_count)
                .map(|i| format!("Error {}", i))
                .collect::<Vec<_>>();

            let result = FetchResult {
                tool_id: "test".to_string(),
                fetched,
                errors,
            };

            // Success requires no errors AND at least one fetch
            let expected_success = error_count == 0 && fetched_count > 0;
            prop_assert_eq!(result.is_success(), expected_success);
            prop_assert_eq!(result.doc_count(), fetched_count);
        }
    }
}
