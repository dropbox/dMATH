//! Documentation collector for verification backends

use crate::types::{CodeBlock, ContentType, Document, DocumentMetadata};
use crate::{BackendDocSource, KnowledgeError, Result};
use chrono::Utc;
use governor::{Quota, RateLimiter};
use reqwest::Client;
use scraper::{Html, Selector};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, info, warn};

/// Documentation collector for fetching and processing backend docs
pub struct DocumentationCollector {
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

impl DocumentationCollector {
    /// Create a new documentation collector
    pub fn new(output_dir: PathBuf) -> Self {
        // Rate limit: 10 requests per second
        let quota = Quota::per_second(NonZeroU32::new(10).unwrap());
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        Self {
            client: Client::builder()
                .user_agent("DashProve Knowledge Collector/0.1")
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            rate_limiter,
            output_dir,
        }
    }

    /// Collect documentation for a backend
    pub async fn collect_backend(&self, source: &BackendDocSource) -> Result<Vec<Document>> {
        info!("Collecting documentation for {}", source.name);
        let mut documents = Vec::new();

        // Create backend directory
        let backend_dir = self
            .output_dir
            .join(format!("{:?}", source.backend).to_lowercase());
        fs::create_dir_all(&backend_dir).await?;

        // Fetch main documentation
        if let Ok(doc) = self
            .fetch_url(source.docs_url, source, ContentType::Reference)
            .await
        {
            documents.push(doc);
        }

        // Fetch tutorial if available
        if let Some(tutorial_url) = source.tutorial_url {
            if let Ok(doc) = self
                .fetch_url(tutorial_url, source, ContentType::Tutorial)
                .await
            {
                documents.push(doc);
            }
        }

        // Fetch API docs if available
        if let Some(api_url) = source.api_url {
            if let Ok(doc) = self.fetch_url(api_url, source, ContentType::Api).await {
                documents.push(doc);
            }
        }

        info!(
            "Collected {} documents for {}",
            documents.len(),
            source.name
        );
        Ok(documents)
    }

    /// Fetch and process a URL
    async fn fetch_url(
        &self,
        url: &str,
        source: &BackendDocSource,
        content_type: ContentType,
    ) -> Result<Document> {
        // Rate limit
        self.rate_limiter.until_ready().await;

        debug!("Fetching {}", url);
        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(KnowledgeError::ParseError(format!(
                "HTTP {} for {}",
                response.status(),
                url
            )));
        }

        let content_type_header = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_default();

        let body = response.text().await?;

        // Process based on content type
        let (content, metadata) = if content_type_header.contains("text/html") {
            self.process_html(&body)?
        } else if url.ends_with(".pdf") {
            self.process_pdf(&body)?
        } else {
            // Assume markdown or plain text
            (body.clone(), DocumentMetadata::default())
        };

        let doc_id = format!(
            "{:?}-{}-{}",
            source.backend,
            content_type as u8,
            url.split('/').next_back().unwrap_or("doc")
        )
        .to_lowercase();

        Ok(Document {
            id: doc_id,
            source: url.to_string(),
            backend: Some(source.backend),
            title: self.extract_title(&body, source.name),
            content,
            content_type,
            fetched_at: Utc::now(),
            metadata,
        })
    }

    /// Process HTML content to markdown
    fn process_html(&self, html: &str) -> Result<(String, DocumentMetadata)> {
        let document = Html::parse_document(html);
        let mut metadata = DocumentMetadata::default();

        // Extract main content (try common selectors)
        let content_selectors = [
            "article",
            "main",
            ".content",
            ".documentation",
            "#content",
            ".markdown-body",
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

        // Convert HTML to text (simple conversion)
        let markdown = html2text::from_read(html_content.as_bytes(), 80);

        // Extract code blocks
        if let Ok(selector) = Selector::parse("pre code, .highlight") {
            for element in document.select(&selector) {
                let code = element.text().collect::<String>();
                let lang = element
                    .value()
                    .attr("class")
                    .and_then(|c| c.split_whitespace().find(|s| s.starts_with("language-")))
                    .map(|s| s.strip_prefix("language-").unwrap_or(s))
                    .unwrap_or("text")
                    .to_string();

                metadata.code_blocks.push(CodeBlock {
                    language: lang,
                    code,
                    description: None,
                });
            }
        }

        // Extract sections
        if let Ok(selector) = Selector::parse("h1, h2, h3") {
            for element in document.select(&selector) {
                let section = element.text().collect::<String>();
                metadata.sections.push(section);
            }
        }

        metadata.word_count = markdown.split_whitespace().count();
        metadata.original_format = Some("html".to_string());

        Ok((markdown, metadata))
    }

    /// Process PDF content
    fn process_pdf(&self, _content: &str) -> Result<(String, DocumentMetadata)> {
        // PDF processing would require the pdf-extract feature
        // For now, return placeholder
        warn!("PDF processing not implemented");
        Ok((
            "PDF content - extraction not available".to_string(),
            DocumentMetadata {
                original_format: Some("pdf".to_string()),
                ..Default::default()
            },
        ))
    }

    /// Extract title from HTML or use default
    fn extract_title(&self, html: &str, default: &str) -> String {
        let document = Html::parse_document(html);
        if let Ok(selector) = Selector::parse("title") {
            if let Some(element) = document.select(&selector).next() {
                let title = element.text().collect::<String>();
                if !title.is_empty() {
                    return title;
                }
            }
        }
        if let Ok(selector) = Selector::parse("h1") {
            if let Some(element) = document.select(&selector).next() {
                let title = element.text().collect::<String>();
                if !title.is_empty() {
                    return title;
                }
            }
        }
        default.to_string()
    }

    /// Save documents to disk
    pub async fn save_documents(&self, documents: &[Document]) -> Result<()> {
        for doc in documents {
            let backend_name = doc
                .backend
                .map(|b| format!("{:?}", b).to_lowercase())
                .unwrap_or_else(|| "general".to_string());

            let dir = self.output_dir.join(&backend_name);
            fs::create_dir_all(&dir).await?;

            // Save content as markdown
            let content_path = dir.join(format!("{}.md", doc.id));
            fs::write(&content_path, &doc.content).await?;

            // Save metadata as JSON
            let meta_path = dir.join(format!("{}.meta.json", doc.id));
            let meta_json = serde_json::to_string_pretty(&doc)?;
            fs::write(&meta_path, meta_json).await?;

            debug!("Saved document {} to {:?}", doc.id, content_path);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_processing() {
        let collector = DocumentationCollector::new(PathBuf::from("/tmp"));
        let html = r#"
            <html>
            <head><title>Test Doc</title></head>
            <body>
                <h1>Main Title</h1>
                <p>Some content here.</p>
                <pre><code class="language-rust">fn main() {}</code></pre>
            </body>
            </html>
        "#;

        let (content, metadata) = collector.process_html(html).unwrap();
        assert!(content.contains("Main Title"));
        assert!(!metadata.sections.is_empty());
    }
}
