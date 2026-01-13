//! PDF downloading and text extraction for ArXiv papers
//!
//! This module provides functionality to:
//! - Download PDFs from ArXiv URLs
//! - Extract text content using pdf-extract
//! - Integrate with ArxivPaper storage

use crate::types::ArxivPaper;
use crate::{KnowledgeError, Result};
use governor::{Quota, RateLimiter};
use reqwest::Client;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, info, warn};

/// PDF processor for downloading and extracting text from ArXiv papers
pub struct PdfProcessor {
    client: Client,
    rate_limiter: Arc<
        RateLimiter<
            governor::state::NotKeyed,
            governor::state::InMemoryState,
            governor::clock::DefaultClock,
        >,
    >,
    pdf_dir: PathBuf,
}

impl PdfProcessor {
    /// Create a new PDF processor
    ///
    /// # Arguments
    /// * `pdf_dir` - Directory to store downloaded PDFs
    pub fn new(pdf_dir: PathBuf) -> Self {
        // ArXiv rate limit: 1 request per 3 seconds (same as API)
        let quota = Quota::per_second(NonZeroU32::new(1).unwrap());
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        Self {
            client: Client::builder()
                .user_agent("DashProve Research Collector/0.1 (https://github.com/dashprove)")
                .timeout(std::time::Duration::from_secs(120)) // PDFs can be large
                .build()
                .expect("Failed to create HTTP client"),
            rate_limiter,
            pdf_dir,
        }
    }

    /// Download a PDF from ArXiv and extract its text
    ///
    /// Updates the paper's `local_path` and `extracted_text` fields
    pub async fn process_paper(&self, paper: &mut ArxivPaper) -> Result<()> {
        // Create storage directory structure: pdf_dir/year/category/
        let year = paper.published.format("%Y").to_string();
        let category_dir = self.pdf_dir.join(&year).join(&paper.primary_category);
        fs::create_dir_all(&category_dir).await?;

        let pdf_path = category_dir.join(format!("{}.pdf", paper.arxiv_id));

        // Download PDF if not already present
        if !pdf_path.exists() {
            self.download_pdf(&paper.pdf_url, &pdf_path).await?;
        } else {
            debug!("PDF already exists: {:?}", pdf_path);
        }

        paper.local_path = Some(pdf_path.clone());

        // Extract text from PDF
        match self.extract_text(&pdf_path).await {
            Ok(text) => {
                info!("Extracted {} chars from {}", text.len(), paper.arxiv_id);
                paper.extracted_text = Some(text);
            }
            Err(e) => {
                warn!("Failed to extract text from {}: {}", paper.arxiv_id, e);
                // Still keep the local_path even if extraction fails
            }
        }

        Ok(())
    }

    /// Download a PDF from a URL
    async fn download_pdf(&self, url: &str, dest: &Path) -> Result<()> {
        self.rate_limiter.until_ready().await;

        info!("Downloading PDF from {}", url);
        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(KnowledgeError::ParseError(format!(
                "HTTP {} downloading PDF from {}",
                response.status(),
                url
            )));
        }

        let bytes = response.bytes().await?;
        fs::write(dest, &bytes).await?;

        debug!("Saved {} bytes to {:?}", bytes.len(), dest);
        Ok(())
    }

    /// Extract text from a PDF file
    async fn extract_text(&self, path: &Path) -> Result<String> {
        let path = path.to_path_buf();

        // Run PDF extraction in blocking task since pdf-extract is synchronous
        tokio::task::spawn_blocking(move || extract_pdf_text_sync(&path))
            .await
            .map_err(|e| KnowledgeError::ParseError(format!("Task join error: {}", e)))?
    }

    /// Process multiple papers with progress tracking
    pub async fn process_papers(&self, papers: &mut [ArxivPaper]) -> ProcessingStats {
        let mut stats = ProcessingStats::default();
        let total = papers.len();

        for (i, paper) in papers.iter_mut().enumerate() {
            info!("Processing paper {}/{}: {}", i + 1, total, paper.arxiv_id);

            match self.process_paper(paper).await {
                Ok(()) => {
                    stats.downloaded += 1;
                    if paper.extracted_text.is_some() {
                        stats.extracted += 1;
                    }
                }
                Err(e) => {
                    warn!("Failed to process {}: {}", paper.arxiv_id, e);
                    stats.failed += 1;
                }
            }

            // Extra delay between downloads to be respectful of ArXiv
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }

        stats
    }
}

/// Statistics from PDF processing
#[derive(Debug, Default, Clone)]
pub struct ProcessingStats {
    /// Number of PDFs successfully downloaded
    pub downloaded: usize,
    /// Number of PDFs with text successfully extracted
    pub extracted: usize,
    /// Number of failures
    pub failed: usize,
}

/// Synchronous PDF text extraction
///
/// Uses pdf-extract when the feature is enabled, otherwise returns an error
#[cfg(feature = "pdf")]
fn extract_pdf_text_sync(path: &Path) -> Result<String> {
    use pdf_extract::extract_text;

    extract_text(path)
        .map_err(|e| KnowledgeError::ParseError(format!("PDF extraction failed: {}", e)))
}

#[cfg(not(feature = "pdf"))]
fn extract_pdf_text_sync(_path: &Path) -> Result<String> {
    Err(KnowledgeError::ParseError(
        "PDF extraction requires the 'pdf' feature to be enabled".to_string(),
    ))
}

/// Clean extracted PDF text by removing artifacts
///
/// ArXiv PDFs often contain artifacts from LaTeX compilation:
/// - Hyphenation at line breaks
/// - Running headers/footers
/// - Page numbers
/// - Reference markers
pub fn clean_pdf_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_line_hyphenated = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // Skip likely headers/footers (very short lines with just numbers)
        if trimmed.len() < 5
            && trimmed
                .chars()
                .all(|c| c.is_ascii_digit() || c.is_whitespace())
        {
            continue;
        }

        // Handle hyphenated word continuations
        if prev_line_hyphenated {
            // Remove hyphen and join with previous line (no space)
            if let Some(stripped) = result.strip_suffix('-') {
                result = stripped.to_string();
            }
            result.push_str(trimmed);
        } else if !result.is_empty() {
            result.push(' ');
            result.push_str(trimmed);
        } else {
            result.push_str(trimmed);
        }

        prev_line_hyphenated = trimmed.ends_with('-') && trimmed.len() > 1;
    }

    // Normalize whitespace
    let normalized: String = result.split_whitespace().collect::<Vec<_>>().join(" ");

    normalized
}

/// Extract sections from PDF text
///
/// Attempts to identify section headers and structure the text
pub fn extract_sections(text: &str) -> Vec<PaperSection> {
    let mut sections = Vec::new();
    let mut current_section = PaperSection {
        title: "Abstract".to_string(),
        content: String::new(),
        level: 0,
    };

    // Common section header patterns
    let section_patterns = [
        (r"^\d+\.\s+", 1),           // "1. Introduction"
        (r"^\d+\.\d+\s+", 2),        // "1.1 Background"
        (r"^[IVX]+\.\s+", 1),        // "I. Introduction"
        (r"^Abstract\s*$", 0),       // "Abstract"
        (r"^Introduction\s*$", 1),   // "Introduction"
        (r"^Conclusion[s]?\s*$", 1), // "Conclusion" or "Conclusions"
        (r"^References\s*$", 1),     // "References"
        (r"^Appendix\s*", 1),        // "Appendix"
    ];

    let compiled_patterns: Vec<_> = section_patterns
        .iter()
        .filter_map(|(pat, level)| regex::Regex::new(pat).ok().map(|r| (r, *level)))
        .collect();

    for line in text.lines() {
        let trimmed = line.trim();

        // Check if this line is a section header
        let mut is_header = false;
        let mut header_level = 0;

        for (pattern, level) in &compiled_patterns {
            if pattern.is_match(trimmed) {
                is_header = true;
                header_level = *level;
                break;
            }
        }

        // Also check for ALL CAPS lines as potential headers (but not too short)
        if !is_header
            && trimmed.len() >= 4
            && trimmed.len() <= 50
            && trimmed
                .chars()
                .all(|c| c.is_uppercase() || c.is_whitespace() || c.is_numeric())
        {
            is_header = true;
            header_level = 1;
        }

        if is_header && !current_section.content.is_empty() {
            // Save current section and start new one
            sections.push(current_section);
            current_section = PaperSection {
                title: trimmed.to_string(),
                content: String::new(),
                level: header_level,
            };
        } else if !is_header {
            if !current_section.content.is_empty() {
                current_section.content.push(' ');
            }
            current_section.content.push_str(trimmed);
        }
    }

    // Don't forget the last section
    if !current_section.content.is_empty() {
        sections.push(current_section);
    }

    sections
}

/// A section extracted from a paper
#[derive(Debug, Clone)]
pub struct PaperSection {
    /// Section title/header
    pub title: String,
    /// Section content
    pub content: String,
    /// Nesting level (0 = top level)
    pub level: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_pdf_text_removes_page_numbers() {
        let text = "This is some text\n\n42\n\nMore text here";
        let cleaned = clean_pdf_text(text);
        assert!(!cleaned.contains("42"));
        assert!(cleaned.contains("This is some text"));
        assert!(cleaned.contains("More text here"));
    }

    #[test]
    fn test_clean_pdf_text_joins_hyphenated_words() {
        let text = "This is a hyphen-\nated word in the text";
        let cleaned = clean_pdf_text(text);
        assert!(cleaned.contains("hyphenated"));
        assert!(!cleaned.contains("hyphen-"));
    }

    #[test]
    fn test_clean_pdf_text_normalizes_whitespace() {
        let text = "Too    many     spaces   here";
        let cleaned = clean_pdf_text(text);
        assert_eq!(cleaned, "Too many spaces here");
    }

    #[test]
    fn test_extract_sections_finds_numbered() {
        let text = "Abstract\nThis is the abstract.\n\n1. Introduction\nThis introduces the paper.\n\n2. Methods\nHere are methods.";
        let sections = extract_sections(text);
        assert!(sections.len() >= 2);
        assert!(sections.iter().any(|s| s.title.contains("Introduction")));
    }

    #[test]
    fn test_extract_sections_handles_roman_numerals() {
        let text = "I. Introduction\nSome intro text.\n\nII. Background\nBackground info.";
        let sections = extract_sections(text);
        assert!(sections.len() >= 2);
    }

    #[test]
    fn test_paper_section_default_values() {
        let section = PaperSection {
            title: "Test".to_string(),
            content: "Content".to_string(),
            level: 0,
        };
        assert_eq!(section.title, "Test");
        assert_eq!(section.level, 0);
    }

    #[test]
    fn test_processing_stats_default() {
        let stats = ProcessingStats::default();
        assert_eq!(stats.downloaded, 0);
        assert_eq!(stats.extracted, 0);
        assert_eq!(stats.failed, 0);
    }
}
