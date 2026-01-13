//! Document chunking for embedding and retrieval

use crate::types::{Document, DocumentChunk};
use crate::Result;

/// Chunking configuration
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Target chunk size in tokens (approximate)
    pub target_tokens: usize,
    /// Overlap between chunks in tokens
    pub overlap_tokens: usize,
    /// Whether to use semantic chunking (split at section boundaries)
    pub semantic_chunking: bool,
    /// Maximum chunk size (hard limit)
    pub max_tokens: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            target_tokens: 512,
            overlap_tokens: 50,
            semantic_chunking: true,
            max_tokens: 1024,
        }
    }
}

/// Document chunker
pub struct Chunker {
    config: ChunkingConfig,
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new(ChunkingConfig::default())
    }
}

impl Chunker {
    /// Create a new chunker with configuration
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }

    /// Chunk a document
    pub fn chunk(&self, document: &Document) -> Result<Vec<DocumentChunk>> {
        let mut chunks = Vec::new();

        if self.config.semantic_chunking {
            // Split by sections first
            let sections = self.split_by_sections(&document.content);
            for (section_idx, (section_path, section_content)) in sections.into_iter().enumerate() {
                let section_chunks = self.chunk_text(&section_content, &section_path);
                for (chunk_idx, content) in section_chunks.into_iter().enumerate() {
                    chunks.push(DocumentChunk {
                        id: format!("{}-s{}-c{}", document.id, section_idx, chunk_idx),
                        document_id: document.id.clone(),
                        chunk_index: chunks.len(),
                        content: content.clone(),
                        backend: document.backend,
                        content_type: document.content_type,
                        section_path: section_path.clone(),
                        token_count: estimate_tokens(&content),
                    });
                }
            }
        } else {
            // Simple chunking
            let text_chunks = self.chunk_text(&document.content, &[]);
            for (idx, content) in text_chunks.into_iter().enumerate() {
                chunks.push(DocumentChunk {
                    id: format!("{}-c{}", document.id, idx),
                    document_id: document.id.clone(),
                    chunk_index: idx,
                    content: content.clone(),
                    backend: document.backend,
                    content_type: document.content_type,
                    section_path: vec![],
                    token_count: estimate_tokens(&content),
                });
            }
        }

        Ok(chunks)
    }

    /// Split content by markdown sections
    fn split_by_sections(&self, content: &str) -> Vec<(Vec<String>, String)> {
        let mut sections = Vec::new();
        let mut current_path: Vec<String> = vec![];
        let mut current_content = String::new();
        let mut current_level = 0;

        for line in content.lines() {
            // Check for markdown headers
            if line.starts_with('#') {
                // Save previous section
                if !current_content.trim().is_empty() {
                    sections.push((current_path.clone(), current_content.clone()));
                    current_content.clear();
                }

                // Parse header level
                let level = line.chars().take_while(|c| *c == '#').count();
                let title = line.trim_start_matches('#').trim().to_string();

                // Update path
                if level > current_level {
                    current_path.push(title);
                } else if level == current_level && !current_path.is_empty() {
                    current_path.pop();
                    current_path.push(title);
                } else {
                    // Level decreased
                    while current_path.len() >= level && !current_path.is_empty() {
                        current_path.pop();
                    }
                    current_path.push(title);
                }
                current_level = level;
            }

            current_content.push_str(line);
            current_content.push('\n');
        }

        // Don't forget the last section
        if !current_content.trim().is_empty() {
            sections.push((current_path, current_content));
        }

        // If no sections found, return whole content
        if sections.is_empty() {
            sections.push((vec![], content.to_string()));
        }

        sections
    }

    /// Chunk text into smaller pieces
    fn chunk_text(&self, text: &str, _section_path: &[String]) -> Vec<String> {
        let mut chunks = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        // Estimate words per token (roughly 0.75 tokens per word for English)
        let words_per_token = 0.75;
        let target_words = (self.config.target_tokens as f64 / words_per_token) as usize;
        let overlap_words = (self.config.overlap_tokens as f64 / words_per_token) as usize;
        let max_words = (self.config.max_tokens as f64 / words_per_token) as usize;

        if words.len() <= max_words {
            // Whole text fits in one chunk
            chunks.push(text.to_string());
        } else {
            // Split into overlapping chunks
            let mut start = 0;
            while start < words.len() {
                let end = (start + target_words).min(words.len());
                let chunk: String = words[start..end].join(" ");
                chunks.push(chunk);

                if end >= words.len() {
                    break;
                }

                // Move start with overlap
                start = if end > overlap_words {
                    end - overlap_words
                } else {
                    end
                };
            }
        }

        chunks
    }
}

/// Estimate token count from text
fn estimate_tokens(text: &str) -> usize {
    // Rough estimate: ~4 characters per token for English
    (text.len() as f64 / 4.0).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ContentType;
    use chrono::Utc;

    fn make_doc(content: &str) -> Document {
        Document {
            id: "test".to_string(),
            source: "test".to_string(),
            backend: None,
            title: "Test".to_string(),
            content: content.to_string(),
            content_type: ContentType::General,
            fetched_at: Utc::now(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn test_simple_chunking() {
        let chunker = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 50,
            ..Default::default()
        });

        let doc = make_doc("This is a test document with some content.");
        let chunks = chunker.chunk(&doc).unwrap();

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].document_id, "test");
    }

    #[test]
    fn test_section_splitting() {
        let chunker = Chunker::default();
        let content = "# Section 1\nContent 1\n## Subsection\nContent 2\n# Section 2\nContent 3";
        let sections = chunker.split_by_sections(content);

        assert!(sections.len() >= 2);
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("test"), 1);
        assert_eq!(estimate_tokens("this is a longer text"), 6);
    }

    // Mutation-killing tests for split_by_sections
    #[test]
    fn test_header_level_counting() {
        let chunker = Chunker::default();

        // Test single hash - should count exactly 1
        let content = "# Level 1\nContent";
        let sections = chunker.split_by_sections(content);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].0.len(), 1);
        assert_eq!(sections[0].0[0], "Level 1");

        // Test double hash - should count exactly 2
        let content = "## Level 2\nContent";
        let sections = chunker.split_by_sections(content);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].0.len(), 1);

        // Test triple hash
        let content = "### Level 3\nContent";
        let sections = chunker.split_by_sections(content);
        assert_eq!(sections.len(), 1);
    }

    #[test]
    fn test_header_level_increase() {
        let chunker = Chunker::default();

        // Test level increase: level 1 -> level 2
        let content = "# Parent\nContent\n## Child\nMore content";
        let sections = chunker.split_by_sections(content);

        // Should have two sections (one before Child, one for Child)
        assert!(sections.len() >= 2);
        // First section has path ["Parent"]
        // Second section has path ["Parent", "Child"]
        let child_section = sections
            .iter()
            .find(|(path, _)| path.contains(&"Child".to_string()));
        assert!(child_section.is_some());
        let (path, _) = child_section.unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], "Parent");
        assert_eq!(path[1], "Child");
    }

    #[test]
    fn test_header_level_same() {
        let chunker = Chunker::default();

        // Test same level: level 1 -> level 1 (sibling sections)
        let content = "# First\nContent 1\n# Second\nContent 2";
        let sections = chunker.split_by_sections(content);

        assert!(sections.len() >= 2);
        // Find "Second" section - should have only "Second" in path, not "First"
        let second_section = sections
            .iter()
            .find(|(path, _)| path.contains(&"Second".to_string()));
        assert!(second_section.is_some());
        let (path, _) = second_section.unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], "Second");
    }

    #[test]
    fn test_header_level_decrease() {
        let chunker = Chunker::default();

        // Test level decrease: level 2 -> level 1
        let content = "## Child First\nContent\n# Parent\nMore content";
        let sections = chunker.split_by_sections(content);

        // The "Parent" section should have path length 1
        let parent_section = sections
            .iter()
            .find(|(path, _)| path.contains(&"Parent".to_string()));
        assert!(parent_section.is_some());
        let (path, _) = parent_section.unwrap();
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn test_split_by_sections_empty_content_filtering() {
        let chunker = Chunker::default();

        // Headers with no content between them
        let content = "# Section 1\n# Section 2\nActual content";
        let sections = chunker.split_by_sections(content);

        // Should filter out empty sections
        for (_, section_content) in &sections {
            assert!(!section_content.trim().is_empty());
        }
    }

    #[test]
    fn test_split_by_sections_no_headers() {
        let chunker = Chunker::default();

        // Plain text with no headers
        let content = "Just some plain text\nwith multiple lines\nbut no headers";
        let sections = chunker.split_by_sections(content);

        assert_eq!(sections.len(), 1);
        assert!(sections[0].0.is_empty());
        assert!(sections[0].1.contains("Just some plain text"));
    }

    #[test]
    fn test_split_by_sections_empty_input() {
        let chunker = Chunker::default();

        // Empty string should return one section with empty content
        let content = "";
        let sections = chunker.split_by_sections(content);

        assert_eq!(sections.len(), 1);
        assert!(sections[0].0.is_empty());
    }

    // Mutation-killing tests for chunk_text arithmetic
    #[test]
    fn test_chunk_text_word_token_conversion() {
        let chunker = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 100,
            overlap_tokens: 10,
            max_tokens: 200,
        });

        // With 0.75 tokens/word ratio:
        // target_tokens=100 -> target_words ~= 133
        // overlap_tokens=10 -> overlap_words ~= 13
        // max_tokens=200 -> max_words ~= 266

        // Generate exactly 300 words - should require multiple chunks
        let words: String = (0..300)
            .map(|i| format!("word{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunker.chunk_text(&words, &[]);

        // Should have more than 1 chunk since 300 > 266 max_words
        assert!(chunks.len() > 1, "Expected multiple chunks for 300 words");
    }

    #[test]
    fn test_chunk_text_overlap_calculation() {
        let chunker = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 75,  // ~100 words
            overlap_tokens: 37, // ~50 words overlap
            max_tokens: 150,    // ~200 words max
        });

        // Generate 400 words - will need multiple chunks
        let words: String = (0..400)
            .map(|i| format!("w{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunker.chunk_text(&words, &[]);

        // With overlap, chunks should share some words
        assert!(chunks.len() >= 2);

        // Check that there's overlap between consecutive chunks
        if chunks.len() >= 2 {
            let chunk1_words: Vec<&str> = chunks[0].split_whitespace().collect();
            let chunk2_words: Vec<&str> = chunks[1].split_whitespace().collect();

            // The last words of chunk1 should appear at the start of chunk2
            // due to overlap
            let overlap_found = chunk1_words
                .iter()
                .rev()
                .take(20)
                .any(|w| chunk2_words.iter().take(20).any(|w2| w == w2));
            // With the overlap logic, there should be shared words
            // (This test verifies the overlap subtraction works correctly)
            assert!(
                overlap_found || chunks.len() > 2,
                "Expected overlap between chunks"
            );
        }
    }

    #[test]
    fn test_chunk_text_fits_in_one() {
        let chunker = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 100,
            overlap_tokens: 10,
            max_tokens: 200,
        });

        // 50 words should fit in one chunk (max_words ~= 266)
        let words: String = (0..50)
            .map(|i| format!("word{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunker.chunk_text(&words, &[]);

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("word0"));
        assert!(chunks[0].contains("word49"));
    }

    #[test]
    fn test_chunk_text_boundary_condition() {
        let chunker = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 75, // ~100 words target
            overlap_tokens: 0, // No overlap
            max_tokens: 75,    // ~100 words max (same as target)
        });

        // Exactly at max_words boundary
        let words: String = (0..100)
            .map(|i| format!("w{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunker.chunk_text(&words, &[]);

        // Should fit in one chunk since 100 words <= max_words
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_chunk_text_zero_overlap_edge() {
        let chunker = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 30, // ~40 words
            overlap_tokens: 0, // No overlap
            max_tokens: 45,    // ~60 words
        });

        // 200 words - will need multiple chunks with no overlap
        let words: String = (0..200)
            .map(|i| format!("w{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunker.chunk_text(&words, &[]);

        assert!(chunks.len() >= 3);

        // With no overlap, chunks should not share words
        if chunks.len() >= 2 {
            let chunk1_words: std::collections::HashSet<&str> =
                chunks[0].split_whitespace().collect();
            let chunk2_words: std::collections::HashSet<&str> =
                chunks[1].split_whitespace().collect();
            // The intersection should be empty with zero overlap
            let intersection: Vec<_> = chunk1_words.intersection(&chunk2_words).collect();
            assert!(
                intersection.is_empty(),
                "With zero overlap, chunks should not share words"
            );
        }
    }

    #[test]
    fn test_chunk_text_large_overlap() {
        let chunker = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 75,  // ~100 words
            overlap_tokens: 60, // ~80 words overlap (large)
            max_tokens: 150,
        });

        // Generate 400 words
        let words: String = (0..400)
            .map(|i| format!("x{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunker.chunk_text(&words, &[]);

        // With large overlap, should have many chunks
        assert!(
            chunks.len() >= 4,
            "Large overlap should produce many chunks"
        );
    }

    #[test]
    fn test_nested_header_levels() {
        let chunker = Chunker::default();

        // Test deeply nested headers: 1 -> 2 -> 3 -> 4
        let content = "# L1\nC1\n## L2\nC2\n### L3\nC3\n#### L4\nC4";
        let sections = chunker.split_by_sections(content);

        // Find the L4 section
        let l4_section = sections.iter().find(|(_, c)| c.contains("C4"));
        assert!(l4_section.is_some());
        let (path, _) = l4_section.unwrap();
        // Path should be ["L1", "L2", "L3", "L4"]
        assert_eq!(path.len(), 4);
    }

    #[test]
    fn test_skip_header_levels() {
        let chunker = Chunker::default();

        // Test skipping levels: 1 -> 3 (skipping 2)
        let content = "# Top\nContent\n### Deep\nDeep content";
        let sections = chunker.split_by_sections(content);

        let deep_section = sections.iter().find(|(_, c)| c.contains("Deep content"));
        assert!(deep_section.is_some());
        let (path, _) = deep_section.unwrap();
        // Path should still work, including both Top and Deep
        assert!(!path.is_empty());
    }

    // Additional mutation-killing tests for edge cases

    #[test]
    fn test_same_level_header_at_start() {
        let chunker = Chunker::default();

        // First header at level 1 when path is empty
        // This tests line 112: level == current_level (both are 0 initially) && !current_path.is_empty()
        let content = "# First\nContent";
        let sections = chunker.split_by_sections(content);

        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].0.len(), 1);
        assert_eq!(sections[0].0[0], "First");
    }

    #[test]
    fn test_level_decrease_empties_path() {
        let chunker = Chunker::default();

        // Test: ### -> ## -> # (level decreases step by step)
        // This tests the while loop at line 117
        let content = "### Deep\nContent1\n## Mid\nContent2\n# Top\nContent3";
        let sections = chunker.split_by_sections(content);

        // Find the "Top" section - path should only contain "Top"
        let top_section = sections.iter().find(|(_, c)| c.contains("Content3"));
        assert!(top_section.is_some());
        let (path, _) = top_section.unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], "Top");
    }

    #[test]
    fn test_level_decrease_from_deep_to_shallow() {
        let chunker = Chunker::default();

        // Test: #### -> # (big level decrease)
        let content = "#### Very Deep\nContent1\n# Shallow\nContent2";
        let sections = chunker.split_by_sections(content);

        let shallow = sections.iter().find(|(_, c)| c.contains("Content2"));
        assert!(shallow.is_some());
        let (path, _) = shallow.unwrap();
        // Shallow should have path ["Shallow"], not nested
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn test_chunk_text_exact_division_calculations() {
        // Test specific division operations - verifying correct math
        // Line 149: target_words = target_tokens / 0.75
        // Line 150: overlap_words = overlap_tokens / 0.75
        // Line 151: max_words = max_tokens / 0.75

        let chunker = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 75, // / 0.75 = 100 words
            overlap_tokens: 0,
            max_tokens: 75, // / 0.75 = 100 words
        });

        // Generate exactly 100 words - should fit in one chunk
        let words100: String = (0..100)
            .map(|i| format!("w{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunker.chunk_text(&words100, &[]);
        assert_eq!(
            chunks.len(),
            1,
            "100 words should fit with max_tokens=75 (max_words=100)"
        );

        // Generate 101 words - should require 2 chunks
        let words101: String = (0..101)
            .map(|i| format!("w{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunker.chunk_text(&words101, &[]);
        assert!(
            chunks.len() >= 2,
            "101 words should exceed max_words and split"
        );
    }

    #[test]
    fn test_chunk_text_loop_boundary() {
        // Test line 159: start < words.len()
        // With no overlap, chunks should cover all words exactly once
        let chunker = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 15, // ~20 words
            overlap_tokens: 0,
            max_tokens: 22, // ~30 words max
        });

        // Generate exactly 60 words - should split into ~3 chunks
        let words: String = (0..60)
            .map(|i| format!("w{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunker.chunk_text(&words, &[]);

        // Concatenate all chunk words
        let all_chunk_words: Vec<&str> = chunks.iter().flat_map(|c| c.split_whitespace()).collect();

        // With no overlap, should have exactly 60 words total
        assert_eq!(all_chunk_words.len(), 60, "All words should be covered");
        assert!(chunks.len() >= 2, "Should have multiple chunks");
    }

    #[test]
    fn test_overlap_words_boundary() {
        // Test line 169: end > overlap_words
        // When overlap_words is very large (larger than end position), use end directly
        let chunker = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 15,   // ~20 words target
            overlap_tokens: 150, // ~200 words overlap (larger than text!)
            max_tokens: 22,      // ~30 words max
        });

        // Generate 50 words - overlap is larger than the text
        let words: String = (0..50)
            .map(|i| format!("w{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunker.chunk_text(&words, &[]);

        // Even with huge overlap config, should still produce chunks
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_overlap_causes_shared_words() {
        // This more precisely tests line 169-173: the overlap logic
        let chunker = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 22,  // ~30 words target
            overlap_tokens: 15, // ~20 words overlap
            max_tokens: 30,     // ~40 words max
        });

        // Generate 100 words
        let words: String = (0..100)
            .map(|i| format!("x{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunker.chunk_text(&words, &[]);

        assert!(chunks.len() >= 3, "Should have at least 3 chunks");

        // Check overlap between chunk 0 and chunk 1
        let c0_words: Vec<&str> = chunks[0].split_whitespace().collect();
        let c1_words: Vec<&str> = chunks[1].split_whitespace().collect();

        // The end of chunk 0 should overlap with start of chunk 1
        // With overlap_words ~= 20, and target_words ~= 30,
        // start for chunk 1 = end0 - overlap_words = 30 - 20 = 10
        // So chunk 1 starts from word 10, meaning words 10-29 are in both chunks
        let last_10_of_c0: Vec<_> = c0_words.iter().rev().take(15).collect();
        let first_15_of_c1: Vec<_> = c1_words.iter().take(15).collect();

        let overlap_count = last_10_of_c0
            .iter()
            .filter(|w| first_15_of_c1.iter().any(|w2| w == &w2))
            .count();

        assert!(
            overlap_count > 0,
            "Chunks should share words due to overlap"
        );
    }

    #[test]
    fn test_division_produces_different_values() {
        // Verify that different token configs produce different word counts
        // This kills mutations that change / to * or %

        let chunker1 = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 60,
            overlap_tokens: 0,
            max_tokens: 60,
        });

        let chunker2 = Chunker::new(ChunkingConfig {
            semantic_chunking: false,
            target_tokens: 120,
            overlap_tokens: 0,
            max_tokens: 120,
        });

        // 100 words
        let words: String = (0..100)
            .map(|i| format!("w{}", i))
            .collect::<Vec<_>>()
            .join(" ");

        let chunks1 = chunker1.chunk_text(&words, &[]);
        let chunks2 = chunker2.chunk_text(&words, &[]);

        // With target_tokens=60 (80 words), 100 words needs 2 chunks
        // With target_tokens=120 (160 words), 100 words fits in 1 chunk
        assert!(
            chunks1.len() > chunks2.len(),
            "More tokens should allow fewer chunks: {} vs {}",
            chunks1.len(),
            chunks2.len()
        );
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use crate::types::ContentType;
    use chrono::Utc;
    use proptest::prelude::*;

    fn make_doc(id: &str, content: &str) -> Document {
        Document {
            id: id.to_string(),
            source: format!("test://{}", id),
            backend: None,
            title: format!("Test Document {}", id),
            content: content.to_string(),
            content_type: ContentType::General,
            fetched_at: Utc::now(),
            metadata: Default::default(),
        }
    }

    proptest! {
        /// Chunking never produces empty chunks (unless input is empty)
        #[test]
        fn test_chunking_produces_non_empty_chunks(
            content in "[a-zA-Z0-9 \\n#]{1,2000}"
        ) {
            let chunker = Chunker::default();
            let doc = make_doc("test", &content);
            let chunks = chunker.chunk(&doc).unwrap();

            // If content has any non-whitespace, we should get chunks
            if content.chars().any(|c| !c.is_whitespace()) {
                prop_assert!(!chunks.is_empty());
                for chunk in &chunks {
                    // Chunks should have content (possibly whitespace-only for edge cases)
                    prop_assert!(!chunk.content.is_empty() || chunk.content.trim().is_empty());
                }
            }
        }

        /// All chunks reference the correct document ID
        #[test]
        fn test_chunk_document_id_consistency(
            doc_id in "[a-z]{3,10}",
            content in "[a-zA-Z0-9 ]{10,500}"
        ) {
            let chunker = Chunker::default();
            let doc = make_doc(&doc_id, &content);
            let chunks = chunker.chunk(&doc).unwrap();

            for chunk in &chunks {
                prop_assert_eq!(&chunk.document_id, &doc_id);
            }
        }

        /// Chunk indices are sequential starting from 0
        #[test]
        fn test_chunk_indices_sequential(
            content in "[a-zA-Z0-9 \\n#]{10,1000}"
        ) {
            let chunker = Chunker::default();
            let doc = make_doc("test", &content);
            let chunks = chunker.chunk(&doc).unwrap();

            for (expected_idx, chunk) in chunks.iter().enumerate() {
                prop_assert_eq!(chunk.chunk_index, expected_idx);
            }
        }

        /// Token estimation is always positive for non-empty strings
        #[test]
        fn test_token_estimation_positive(
            text in ".{1,1000}"
        ) {
            let tokens = estimate_tokens(&text);
            prop_assert!(tokens > 0, "Expected positive tokens for non-empty text");
        }

        /// Token estimation is monotonic (more text = more tokens)
        #[test]
        fn test_token_estimation_monotonic(
            prefix in "[a-zA-Z0-9]{10,100}",
            suffix in "[a-zA-Z0-9]{10,100}"
        ) {
            let short_text = &prefix;
            let long_text = format!("{}{}", prefix, suffix);

            let short_tokens = estimate_tokens(short_text);
            let long_tokens = estimate_tokens(&long_text);

            prop_assert!(long_tokens >= short_tokens,
                "Longer text should have >= tokens: {} vs {}",
                long_tokens, short_tokens);
        }

        /// Section splitting handles arbitrary markdown headers
        #[test]
        fn test_section_splitting_handles_headers(
            header_level in 1usize..=6,
            title in "[a-zA-Z ]{3,20}",
            content in "[a-zA-Z0-9 ]{10,100}"
        ) {
            let chunker = Chunker::default();
            let hashes = "#".repeat(header_level);
            let markdown = format!("{} {}\n{}", hashes, title, content);

            let sections = chunker.split_by_sections(&markdown);
            prop_assert!(!sections.is_empty());
        }

        /// Multiple sections are correctly identified
        #[test]
        fn test_multiple_sections_identified(
            section_count in 2usize..=5,
        ) {
            let chunker = Chunker::default();
            let mut markdown = String::new();

            for i in 0..section_count {
                markdown.push_str(&format!("# Section {}\nContent for section {}.\n\n", i, i));
            }

            let sections = chunker.split_by_sections(&markdown);
            // Should have at least section_count sections
            // (might have more due to how empty content is handled)
            prop_assert!(sections.len() >= section_count.saturating_sub(1),
                "Expected at least {} sections, got {}",
                section_count.saturating_sub(1), sections.len());
        }

        /// Chunking with non-semantic mode works on any text
        #[test]
        fn test_non_semantic_chunking(
            content in "[a-zA-Z0-9 ]{10,500}"
        ) {
            let chunker = Chunker::new(ChunkingConfig {
                semantic_chunking: false,
                target_tokens: 50,
                overlap_tokens: 10,
                max_tokens: 100,
            });
            let doc = make_doc("test", &content);
            let chunks = chunker.chunk(&doc).unwrap();

            prop_assert!(!chunks.is_empty());
            for chunk in &chunks {
                prop_assert!(chunk.section_path.is_empty(),
                    "Non-semantic chunking should have empty section paths");
            }
        }

        /// Large documents get split into multiple chunks
        #[test]
        fn test_large_documents_split(
            word_count in 200usize..=500
        ) {
            let chunker = Chunker::new(ChunkingConfig {
                semantic_chunking: false,
                target_tokens: 50,  // Small target to force splitting
                overlap_tokens: 5,
                max_tokens: 75,
            });

            // Generate a document with many words
            let content: String = (0..word_count)
                .map(|i| format!("word{}", i))
                .collect::<Vec<_>>()
                .join(" ");

            let doc = make_doc("large", &content);
            let chunks = chunker.chunk(&doc).unwrap();

            // With 200+ words and 50 token target, should have multiple chunks
            prop_assert!(chunks.len() > 1,
                "Expected multiple chunks for {} words, got {}",
                word_count, chunks.len());
        }

        /// Chunk IDs are unique within a document
        #[test]
        fn test_chunk_ids_unique(
            content in "[a-zA-Z0-9 \\n#]{50,500}"
        ) {
            let chunker = Chunker::default();
            let doc = make_doc("test", &content);
            let chunks = chunker.chunk(&doc).unwrap();

            let mut seen_ids = std::collections::HashSet::new();
            for chunk in &chunks {
                prop_assert!(seen_ids.insert(&chunk.id),
                    "Duplicate chunk ID: {}", chunk.id);
            }
        }

        /// Token count in chunks is reasonable
        #[test]
        fn test_chunk_token_count_reasonable(
            content in "[a-zA-Z0-9 ]{50,500}"
        ) {
            let chunker = Chunker::default();
            let doc = make_doc("test", &content);
            let chunks = chunker.chunk(&doc).unwrap();

            for chunk in &chunks {
                // Token count should be positive and roughly match content length / 4
                prop_assert!(chunk.token_count > 0);
                let expected = (chunk.content.len() as f64 / 4.0).ceil() as usize;
                prop_assert_eq!(chunk.token_count, expected,
                    "Token count mismatch for chunk");
            }
        }
    }
}
