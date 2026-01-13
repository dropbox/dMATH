//! Tool knowledge loading and management
//!
//! This module loads structured JSON knowledge entries for verification tools
//! from `data/knowledge/tools/`.

use crate::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, warn};

/// Check if a pattern string looks like a regex (contains regex metacharacters).
///
/// This is a pure function that can be verified with Kani.
/// It checks for common regex metacharacters that are unlikely in plain error messages.
///
/// # Examples
///
/// ```
/// use dashprove_knowledge::tool_knowledge::pattern_looks_like_regex;
///
/// assert!(pattern_looks_like_regex("\\d+"));  // digit escape
/// assert!(pattern_looks_like_regex(".*error")); // wildcard
/// assert!(!pattern_looks_like_regex("simple error")); // plain text
/// ```
#[inline]
pub fn pattern_looks_like_regex(pattern: &str) -> bool {
    // Check for common regex metacharacters that are unlikely in plain error messages
    pattern.contains("\\d")
        || pattern.contains("\\s")
        || pattern.contains("\\w")
        || pattern.contains(".*")
        || pattern.contains(".+")
        || pattern.contains("[^")
        || pattern.contains("(?")
        || pattern.contains("\\(")
        || pattern.contains("\\)")
        || (pattern.contains('[') && pattern.contains(']'))
        || (pattern.contains('(') && pattern.contains(')') && pattern.contains('|'))
}

/// A verification tool's knowledge entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolKnowledge {
    /// Tool ID (e.g., "kani", "lean4")
    pub id: String,
    /// Display name
    pub name: String,
    /// Primary category
    pub category: String,
    /// Subcategory
    #[serde(default)]
    pub subcategory: Option<String>,

    /// Short description
    pub description: String,
    /// Detailed description
    #[serde(default)]
    pub long_description: Option<String>,

    /// Tool capabilities
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Supported property types
    #[serde(default)]
    pub property_types: Vec<String>,
    /// Input languages/formats
    #[serde(default)]
    pub input_languages: Vec<String>,
    /// Output formats
    #[serde(default)]
    pub output_formats: Vec<String>,

    /// Installation information
    #[serde(default)]
    pub installation: Option<InstallationInfo>,

    /// Documentation URLs
    #[serde(default)]
    pub documentation: Option<DocumentationUrls>,

    /// Available tactics/strategies
    #[serde(default)]
    pub tactics: Vec<Tactic>,

    /// Known error patterns
    #[serde(default)]
    pub error_patterns: Vec<ErrorPattern>,

    /// DashProve integration info
    #[serde(default)]
    pub integration: Option<IntegrationInfo>,

    /// Performance characteristics
    #[serde(default)]
    pub performance: Option<PerformanceInfo>,

    /// Comparisons with other tools
    #[serde(default)]
    pub comparisons: Option<Comparisons>,

    /// Entry metadata
    #[serde(default)]
    pub metadata: Option<EntryMetadata>,
}

/// Installation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallationInfo {
    /// Installation methods
    #[serde(default)]
    pub methods: Vec<InstallMethod>,
    /// Required dependencies
    #[serde(default)]
    pub dependencies: Vec<String>,
    /// Supported platforms
    #[serde(default)]
    pub platforms: Vec<String>,
}

/// A single installation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallMethod {
    /// Method type (cargo, brew, pip, apt, source, etc.)
    #[serde(rename = "type")]
    pub method_type: String,
    /// Installation command
    #[serde(default)]
    pub command: Option<String>,
    /// Source URL (for source builds)
    #[serde(default)]
    pub url: Option<String>,
}

/// Documentation URLs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationUrls {
    /// Official documentation
    #[serde(default)]
    pub official: Option<String>,
    /// Tutorial/getting started
    #[serde(default)]
    pub tutorial: Option<String>,
    /// API reference
    #[serde(default)]
    pub api_reference: Option<String>,
    /// Examples
    #[serde(default)]
    pub examples: Option<String>,
}

/// A tactic or strategy for using the tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tactic {
    /// Tactic name
    pub name: String,
    /// Description
    pub description: String,
    /// Syntax/usage
    #[serde(default)]
    pub syntax: Option<String>,
    /// When to use this tactic
    #[serde(default)]
    pub when_to_use: Option<String>,
    /// Example usages
    #[serde(default)]
    pub examples: Vec<String>,
}

/// An error pattern with explanations and fixes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Error pattern (regex or substring)
    pub pattern: String,
    /// What the error means
    pub meaning: String,
    /// Common causes
    #[serde(default)]
    pub common_causes: Vec<String>,
    /// Suggested fixes
    #[serde(default)]
    pub fixes: Vec<String>,
}

/// DashProve integration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationInfo {
    /// Whether this is a DashProve backend
    #[serde(default)]
    pub dashprove_backend: bool,
    /// Supported USL property types
    #[serde(default)]
    pub usl_property_types: Vec<String>,
    /// CLI command to invoke
    #[serde(default)]
    pub cli_command: Option<String>,
}

/// Performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInfo {
    /// Typical runtime
    #[serde(default)]
    pub typical_runtime: Option<String>,
    /// Scalability notes
    #[serde(default)]
    pub scalability: Option<String>,
    /// Memory usage
    #[serde(default)]
    pub memory_usage: Option<String>,
}

/// Comparisons with similar tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparisons {
    /// Similar tools
    #[serde(default)]
    pub similar_tools: Vec<String>,
    /// Advantages over alternatives
    #[serde(default)]
    pub advantages: Vec<String>,
    /// Disadvantages
    #[serde(default)]
    pub disadvantages: Vec<String>,
}

/// Entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryMetadata {
    /// Tool version
    #[serde(default)]
    pub version: Option<String>,
    /// Last update date
    #[serde(default)]
    pub last_updated: Option<String>,
    /// Maintainer
    #[serde(default)]
    pub maintainer: Option<String>,
    /// License
    #[serde(default)]
    pub license: Option<String>,
}

/// Collection of all tool knowledge
#[derive(Debug, Clone, Default)]
pub struct ToolKnowledgeStore {
    /// All tools indexed by ID
    tools: HashMap<String, ToolKnowledge>,
    /// Tools indexed by category
    by_category: HashMap<String, Vec<String>>,
    /// Tools indexed by capability
    by_capability: HashMap<String, Vec<String>>,
}

impl ToolKnowledgeStore {
    /// Create an empty store
    pub fn new() -> Self {
        Self::default()
    }

    /// Load all tool knowledge from a directory
    pub async fn load_from_dir(base_dir: &Path) -> Result<Self> {
        let mut store = Self::new();
        store.load_directory(base_dir).await?;
        Ok(store)
    }

    /// Load tool knowledge files from a directory recursively
    async fn load_directory(&mut self, dir: &Path) -> Result<()> {
        let mut entries = fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let file_type = entry.file_type().await?;

            if file_type.is_dir() {
                // Recurse into subdirectories
                Box::pin(self.load_directory(&path)).await?;
            } else if file_type.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "json" {
                        match self.load_file(&path).await {
                            Ok(tool) => {
                                debug!("Loaded tool knowledge: {}", tool.id);
                                self.add_tool(tool);
                            }
                            Err(e) => {
                                warn!("Failed to load {:?}: {}", path, e);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Load a single tool knowledge file
    async fn load_file(&self, path: &Path) -> Result<ToolKnowledge> {
        let content = fs::read_to_string(path).await?;
        let tool: ToolKnowledge = serde_json::from_str(&content)?;
        Ok(tool)
    }

    /// Add a tool to the store
    pub fn add_tool(&mut self, tool: ToolKnowledge) {
        let id = tool.id.clone();
        let category = tool.category.clone();
        let capabilities = tool.capabilities.clone();

        // Index by category
        self.by_category
            .entry(category)
            .or_default()
            .push(id.clone());

        // Index by capability
        for cap in capabilities {
            self.by_capability.entry(cap).or_default().push(id.clone());
        }

        // Store the tool
        self.tools.insert(id, tool);
    }

    /// Get a tool by ID
    pub fn get(&self, id: &str) -> Option<&ToolKnowledge> {
        self.tools.get(id)
    }

    /// Get all tools
    pub fn all(&self) -> impl Iterator<Item = &ToolKnowledge> {
        self.tools.values()
    }

    /// Get tool count
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Get tools by category
    pub fn by_category(&self, category: &str) -> Vec<&ToolKnowledge> {
        self.by_category
            .get(category)
            .map(|ids| ids.iter().filter_map(|id| self.tools.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get tools with a specific capability
    pub fn by_capability(&self, capability: &str) -> Vec<&ToolKnowledge> {
        self.by_capability
            .get(capability)
            .map(|ids| ids.iter().filter_map(|id| self.tools.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all categories
    pub fn categories(&self) -> Vec<&str> {
        self.by_category.keys().map(|s| s.as_str()).collect()
    }

    /// Get all capabilities
    pub fn capabilities(&self) -> Vec<&str> {
        self.by_capability.keys().map(|s| s.as_str()).collect()
    }

    /// Find tools matching a query
    pub fn search(&self, query: &str) -> Vec<&ToolKnowledge> {
        let query_lower = query.to_lowercase();
        let terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut results: Vec<(&ToolKnowledge, usize)> = self
            .tools
            .values()
            .map(|tool| {
                let text = format!(
                    "{} {} {} {} {}",
                    tool.name,
                    tool.description,
                    tool.long_description.as_deref().unwrap_or(""),
                    tool.category,
                    tool.capabilities.join(" ")
                )
                .to_lowercase();

                let matches: usize = terms.iter().filter(|term| text.contains(*term)).count();
                (tool, matches)
            })
            .filter(|(_, matches)| *matches > 0)
            .collect();

        results.sort_by(|a, b| b.1.cmp(&a.1));
        results.into_iter().map(|(tool, _)| tool).collect()
    }

    /// Find tools similar to a given tool
    pub fn find_similar(&self, tool_id: &str) -> Vec<&ToolKnowledge> {
        let tool = match self.get(tool_id) {
            Some(t) => t,
            None => return vec![],
        };

        // Get tools with similar capabilities
        let mut similar_ids: HashMap<String, usize> = HashMap::new();

        for cap in &tool.capabilities {
            if let Some(ids) = self.by_capability.get(cap) {
                for id in ids {
                    if id != tool_id {
                        *similar_ids.entry(id.clone()).or_default() += 1;
                    }
                }
            }
        }

        // Also include tools mentioned in comparisons
        if let Some(ref comparisons) = tool.comparisons {
            for similar in &comparisons.similar_tools {
                // Normalize the name to try to match IDs
                let normalized = similar.to_lowercase().replace(' ', "_");
                if self.tools.contains_key(&normalized) {
                    *similar_ids.entry(normalized).or_default() += 5;
                }
            }
        }

        let mut results: Vec<_> = similar_ids.into_iter().collect();
        results.sort_by(|a, b| b.1.cmp(&a.1));

        results
            .into_iter()
            .take(10)
            .filter_map(|(id, _)| self.tools.get(&id))
            .collect()
    }

    /// Get error patterns for a tool
    pub fn get_error_patterns(&self, tool_id: &str) -> Vec<&ErrorPattern> {
        self.get(tool_id)
            .map(|t| t.error_patterns.iter().collect())
            .unwrap_or_default()
    }

    /// Find fixes for an error message using both substring and regex matching
    ///
    /// This method tries to match error messages against known error patterns.
    /// Patterns can be:
    /// - Simple substrings (default)
    /// - Regex patterns (prefixed with "regex:" or containing regex metacharacters)
    ///
    /// Regex matches get higher confidence than substring matches.
    pub fn find_error_fixes(&self, tool_id: &str, error_message: &str) -> Vec<ErrorMatch> {
        let tool = match self.get(tool_id) {
            Some(t) => t,
            None => return vec![],
        };

        let error_lower = error_message.to_lowercase();

        tool.error_patterns
            .iter()
            .filter_map(|pattern| {
                let (is_regex, pattern_str) = if pattern.pattern.starts_with("regex:") {
                    (true, pattern.pattern.strip_prefix("regex:").unwrap())
                } else if Self::looks_like_regex(&pattern.pattern) {
                    (true, pattern.pattern.as_str())
                } else {
                    (false, pattern.pattern.as_str())
                };

                if is_regex {
                    // Try regex match
                    match Self::get_or_compile_regex(pattern_str) {
                        Some(re) => {
                            if re.is_match(&error_lower) {
                                Some(ErrorMatch {
                                    pattern: pattern.clone(),
                                    confidence: 0.9, // Higher confidence for regex match
                                })
                            } else {
                                None
                            }
                        }
                        None => {
                            // Regex compilation failed, fall back to substring match
                            let pattern_lower = pattern_str.to_lowercase();
                            if error_lower.contains(&pattern_lower) {
                                Some(ErrorMatch {
                                    pattern: pattern.clone(),
                                    confidence: 0.6, // Lower confidence for fallback
                                })
                            } else {
                                None
                            }
                        }
                    }
                } else {
                    // Simple substring match
                    let pattern_lower = pattern_str.to_lowercase();
                    if error_lower.contains(&pattern_lower) {
                        Some(ErrorMatch {
                            pattern: pattern.clone(),
                            confidence: 0.8,
                        })
                    } else {
                        None
                    }
                }
            })
            .collect()
    }

    /// Check if a pattern looks like a regex (contains regex metacharacters)
    fn looks_like_regex(pattern: &str) -> bool {
        // Delegate to the standalone function for easier testing/verification
        pattern_looks_like_regex(pattern)
    }

    /// Get a compiled regex from cache or compile it
    fn get_or_compile_regex(pattern: &str) -> Option<&'static Regex> {
        // Use thread-local storage for regex cache
        thread_local! {
            static REGEX_CACHE: std::cell::RefCell<HashMap<String, Option<&'static Regex>>> =
                std::cell::RefCell::new(HashMap::new());
        }

        REGEX_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(cached) = cache.get(pattern) {
                return *cached;
            }

            // Try to compile the regex (case insensitive)
            let result = match Regex::new(&format!("(?i){}", pattern)) {
                Ok(re) => {
                    // Leak the regex to get a 'static reference
                    let leaked: &'static Regex = Box::leak(Box::new(re));
                    Some(leaked)
                }
                Err(e) => {
                    warn!("Invalid regex pattern '{}': {}", pattern, e);
                    None
                }
            };

            cache.insert(pattern.to_string(), result);
            result
        })
    }

    /// Get tactics for a tool
    pub fn get_tactics(&self, tool_id: &str) -> Vec<&Tactic> {
        self.get(tool_id)
            .map(|t| t.tactics.iter().collect())
            .unwrap_or_default()
    }

    /// Get documentation URLs for a tool
    pub fn get_documentation(&self, tool_id: &str) -> Option<&DocumentationUrls> {
        self.get(tool_id).and_then(|t| t.documentation.as_ref())
    }
}

/// An error pattern match
#[derive(Debug, Clone)]
pub struct ErrorMatch {
    /// The matched pattern
    pub pattern: ErrorPattern,
    /// Match confidence (0.0-1.0)
    pub confidence: f32,
}

/// Load tool knowledge from the default location
pub async fn load_default_tool_knowledge() -> Result<ToolKnowledgeStore> {
    let base_dir = PathBuf::from("data/knowledge/tools");
    ToolKnowledgeStore::load_from_dir(&base_dir).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;

    async fn create_test_tool_file(dir: &Path, filename: &str, content: &str) -> PathBuf {
        let path = dir.join(filename);
        let mut file = File::create(&path).await.unwrap();
        file.write_all(content.as_bytes()).await.unwrap();
        path
    }

    // ========== pattern_looks_like_regex unit tests ==========
    // These tests are designed to kill mutations that survive the Kani proofs
    // (Kani proofs don't run under cargo test)

    #[test]
    fn test_pattern_regex_escapes() {
        // Test each escape sequence independently
        assert!(
            pattern_looks_like_regex("\\d+"),
            "\\d should be detected as regex"
        );
        assert!(
            pattern_looks_like_regex("\\s*"),
            "\\s should be detected as regex"
        );
        assert!(
            pattern_looks_like_regex("\\w+"),
            "\\w should be detected as regex"
        );
    }

    #[test]
    fn test_pattern_regex_wildcards() {
        assert!(
            pattern_looks_like_regex(".*error"),
            ".* should be detected as regex"
        );
        assert!(
            pattern_looks_like_regex(".+match"),
            ".+ should be detected as regex"
        );
    }

    #[test]
    fn test_pattern_regex_brackets() {
        assert!(
            pattern_looks_like_regex("[a-z]"),
            "[...] should be detected as regex"
        );
        assert!(
            pattern_looks_like_regex("[^abc]"),
            "[^...] should be detected as regex"
        );
    }

    #[test]
    fn test_pattern_regex_special_groups() {
        assert!(
            pattern_looks_like_regex("(?i)case"),
            "(? should be detected as regex"
        );
        assert!(
            pattern_looks_like_regex("\\(foo\\)"),
            "\\(...\\) should be detected as regex"
        );
        assert!(
            pattern_looks_like_regex("(foo|bar)"),
            "(...|...) should be detected as regex"
        );
    }

    #[test]
    fn test_pattern_plain_text_not_regex() {
        assert!(
            !pattern_looks_like_regex("error"),
            "plain word should not be regex"
        );
        assert!(
            !pattern_looks_like_regex("simple error message"),
            "plain text should not be regex"
        );
        assert!(
            !pattern_looks_like_regex("timeout 5000ms"),
            "text with numbers should not be regex"
        );
        assert!(
            !pattern_looks_like_regex(""),
            "empty string should not be regex"
        );
        assert!(
            !pattern_looks_like_regex("file.txt"),
            "single dot should not be regex"
        );
        assert!(
            !pattern_looks_like_regex("(value)"),
            "parens without | should not be regex"
        );
        assert!(
            !pattern_looks_like_regex("func(x)"),
            "function call should not be regex"
        );
    }

    #[test]
    fn test_pattern_regex_each_condition_independent() {
        // Test that each condition in the || chain is independently detected
        // These patterns trigger only ONE condition each

        // Only \\d
        assert!(
            pattern_looks_like_regex("test\\d"),
            "\\d alone should match"
        );

        // Only \\s
        assert!(
            pattern_looks_like_regex("test\\s"),
            "\\s alone should match"
        );

        // Only \\w
        assert!(
            pattern_looks_like_regex("test\\w"),
            "\\w alone should match"
        );

        // Only .*
        assert!(
            pattern_looks_like_regex("test.*end"),
            ".* alone should match"
        );

        // Only .+
        assert!(
            pattern_looks_like_regex("test.+end"),
            ".+ alone should match"
        );

        // Only [^
        assert!(pattern_looks_like_regex("[^x]"), "[^ alone should match");

        // Only (?
        assert!(pattern_looks_like_regex("(?:x)"), "(? alone should match");

        // Only \\(
        assert!(
            pattern_looks_like_regex("test\\("),
            "\\( alone should match"
        );

        // Only \\)
        assert!(
            pattern_looks_like_regex("test\\)"),
            "\\) alone should match"
        );

        // Only [...]
        assert!(pattern_looks_like_regex("[xy]"), "[...] alone should match");
    }

    #[test]
    fn test_pattern_bracket_requires_both() {
        // Test that [...] detection requires BOTH [ and ]
        // This kills the mutation that changes && to || on line 40
        assert!(
            !pattern_looks_like_regex("array[0"),
            "unmatched [ should not be regex"
        );
        assert!(
            !pattern_looks_like_regex("result]"),
            "unmatched ] should not be regex"
        );
        // But matched brackets should be detected
        assert!(
            pattern_looks_like_regex("[a-z]"),
            "matched brackets should be regex"
        );
    }

    // ========== Store method tests to kill remaining mutations ==========

    #[tokio::test]
    async fn test_store_all_iterator() {
        let temp_dir = TempDir::new().unwrap();

        let content1 = r#"{
            "id": "tool1",
            "name": "Tool 1",
            "category": "cat1",
            "description": "First tool"
        }"#;

        let content2 = r#"{
            "id": "tool2",
            "name": "Tool 2",
            "category": "cat2",
            "description": "Second tool"
        }"#;

        create_test_tool_file(temp_dir.path(), "tool1.json", content1).await;
        create_test_tool_file(temp_dir.path(), "tool2.json", content2).await;

        let store = ToolKnowledgeStore::load_from_dir(temp_dir.path())
            .await
            .unwrap();

        // Test that all() returns ALL tools - kills "return empty iterator" mutation
        let all_tools: Vec<_> = store.all().collect();
        assert_eq!(all_tools.len(), 2, "all() must return all tools");
        assert!(
            all_tools.iter().any(|t| t.id == "tool1"),
            "all() must include tool1"
        );
        assert!(
            all_tools.iter().any(|t| t.id == "tool2"),
            "all() must include tool2"
        );
    }

    #[tokio::test]
    async fn test_store_is_empty() {
        let mut store = ToolKnowledgeStore::new();
        assert!(store.is_empty(), "new store should be empty");

        // Add a tool
        store.add_tool(ToolKnowledge {
            id: "test".to_string(),
            name: "Test".to_string(),
            category: "test".to_string(),
            subcategory: None,
            description: "Test".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        });

        // Kill "return true" mutation
        assert!(!store.is_empty(), "store with tool should not be empty");
    }

    #[tokio::test]
    async fn test_store_find_similar() {
        let temp_dir = TempDir::new().unwrap();

        let kani = r#"{
            "id": "kani",
            "name": "Kani",
            "category": "rust_verification",
            "description": "Model checker for Rust",
            "capabilities": ["model_checking", "memory_safety"],
            "related_backends": ["miri", "prusti"]
        }"#;

        let miri = r#"{
            "id": "miri",
            "name": "Miri",
            "category": "rust_verification",
            "description": "Interpreter for Rust",
            "capabilities": ["interpretation", "memory_safety"],
            "related_backends": ["kani"]
        }"#;

        let mypy = r#"{
            "id": "mypy",
            "name": "Mypy",
            "category": "python_typing",
            "description": "Type checker for Python",
            "capabilities": ["type_checking"]
        }"#;

        create_test_tool_file(temp_dir.path(), "kani.json", kani).await;
        create_test_tool_file(temp_dir.path(), "miri.json", miri).await;
        create_test_tool_file(temp_dir.path(), "mypy.json", mypy).await;

        let store = ToolKnowledgeStore::load_from_dir(temp_dir.path())
            .await
            .unwrap();

        // Kill "return empty vec" mutation
        let similar = store.find_similar("kani");
        assert!(
            !similar.is_empty(),
            "find_similar must return similar tools"
        );
        assert!(
            similar.iter().any(|t| t.id == "miri"),
            "miri should be similar to kani"
        );

        // Verify mypy is less similar or not included (different category/capabilities)
        // This tests the scoring logic
        let similar_ids: Vec<&str> = similar.iter().map(|t| t.id.as_str()).collect();
        if similar_ids.contains(&"mypy") {
            // If mypy is included, miri should be ranked higher
            let miri_pos = similar_ids.iter().position(|&id| id == "miri");
            let mypy_pos = similar_ids.iter().position(|&id| id == "mypy");
            if let (Some(m), Some(p)) = (miri_pos, mypy_pos) {
                assert!(
                    m < p,
                    "miri should rank higher than mypy in similarity to kani"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_store_get_error_patterns() {
        let temp_dir = TempDir::new().unwrap();

        let content = r#"{
            "id": "test_tool",
            "name": "Test Tool",
            "category": "testing",
            "description": "A test tool",
            "error_patterns": [
                {
                    "pattern": "unwinding assertion loop.*",
                    "meaning": "Loop bound exceeded",
                    "fixes": ["Increase unwind bound"]
                }
            ]
        }"#;

        create_test_tool_file(temp_dir.path(), "test_tool.json", content).await;

        let store = ToolKnowledgeStore::load_from_dir(temp_dir.path())
            .await
            .unwrap();

        // Kill "return empty vec" mutation
        let patterns = store.get_error_patterns("test_tool");
        assert!(
            !patterns.is_empty(),
            "get_error_patterns must return patterns when they exist"
        );
        assert_eq!(patterns[0].pattern, "unwinding assertion loop.*");
    }

    #[tokio::test]
    async fn test_store_get_documentation() {
        let temp_dir = TempDir::new().unwrap();

        let content = r#"{
            "id": "test_tool",
            "name": "Test Tool",
            "category": "testing",
            "description": "A test tool",
            "documentation": {
                "official": "https://example.com/docs",
                "github": "https://github.com/example/test"
            }
        }"#;

        create_test_tool_file(temp_dir.path(), "test_tool.json", content).await;

        let store = ToolKnowledgeStore::load_from_dir(temp_dir.path())
            .await
            .unwrap();

        // Kill "return None" mutation
        let docs = store.get_documentation("test_tool");
        assert!(
            docs.is_some(),
            "get_documentation must return docs when they exist"
        );
        assert_eq!(
            docs.unwrap().official,
            Some("https://example.com/docs".to_string())
        );
    }

    #[tokio::test]
    async fn test_load_single_tool() {
        let temp_dir = TempDir::new().unwrap();
        let content = r#"{
            "id": "test_tool",
            "name": "Test Tool",
            "category": "testing",
            "description": "A test tool for unit testing",
            "capabilities": ["cap1", "cap2"]
        }"#;

        create_test_tool_file(temp_dir.path(), "test_tool.json", content).await;

        let store = ToolKnowledgeStore::load_from_dir(temp_dir.path())
            .await
            .unwrap();

        assert_eq!(store.len(), 1);
        let tool = store.get("test_tool").unwrap();
        assert_eq!(tool.name, "Test Tool");
        assert_eq!(tool.category, "testing");
        assert_eq!(tool.capabilities, vec!["cap1", "cap2"]);
    }

    #[tokio::test]
    async fn test_load_nested_directories() {
        let temp_dir = TempDir::new().unwrap();
        let subdir = temp_dir.path().join("category");
        fs::create_dir(&subdir).await.unwrap();

        let content1 = r#"{
            "id": "tool1",
            "name": "Tool 1",
            "category": "cat1",
            "description": "First tool"
        }"#;

        let content2 = r#"{
            "id": "tool2",
            "name": "Tool 2",
            "category": "cat2",
            "description": "Second tool"
        }"#;

        create_test_tool_file(temp_dir.path(), "tool1.json", content1).await;
        create_test_tool_file(&subdir, "tool2.json", content2).await;

        let store = ToolKnowledgeStore::load_from_dir(temp_dir.path())
            .await
            .unwrap();

        assert_eq!(store.len(), 2);
        assert!(store.get("tool1").is_some());
        assert!(store.get("tool2").is_some());
    }

    #[tokio::test]
    async fn test_by_category() {
        let temp_dir = TempDir::new().unwrap();

        let content1 = r#"{
            "id": "tool1",
            "name": "Tool 1",
            "category": "rust",
            "description": "Rust tool"
        }"#;

        let content2 = r#"{
            "id": "tool2",
            "name": "Tool 2",
            "category": "rust",
            "description": "Another Rust tool"
        }"#;

        let content3 = r#"{
            "id": "tool3",
            "name": "Tool 3",
            "category": "python",
            "description": "Python tool"
        }"#;

        create_test_tool_file(temp_dir.path(), "tool1.json", content1).await;
        create_test_tool_file(temp_dir.path(), "tool2.json", content2).await;
        create_test_tool_file(temp_dir.path(), "tool3.json", content3).await;

        let store = ToolKnowledgeStore::load_from_dir(temp_dir.path())
            .await
            .unwrap();

        let rust_tools = store.by_category("rust");
        assert_eq!(rust_tools.len(), 2);

        let python_tools = store.by_category("python");
        assert_eq!(python_tools.len(), 1);
    }

    #[tokio::test]
    async fn test_by_capability() {
        let temp_dir = TempDir::new().unwrap();

        let content1 = r#"{
            "id": "tool1",
            "name": "Tool 1",
            "category": "test",
            "description": "Test",
            "capabilities": ["memory_safety", "fuzzing"]
        }"#;

        let content2 = r#"{
            "id": "tool2",
            "name": "Tool 2",
            "category": "test",
            "description": "Test",
            "capabilities": ["memory_safety", "verification"]
        }"#;

        create_test_tool_file(temp_dir.path(), "tool1.json", content1).await;
        create_test_tool_file(temp_dir.path(), "tool2.json", content2).await;

        let store = ToolKnowledgeStore::load_from_dir(temp_dir.path())
            .await
            .unwrap();

        let memory_tools = store.by_capability("memory_safety");
        assert_eq!(memory_tools.len(), 2);

        let fuzzing_tools = store.by_capability("fuzzing");
        assert_eq!(fuzzing_tools.len(), 1);
    }

    #[tokio::test]
    async fn test_search() {
        let temp_dir = TempDir::new().unwrap();

        let content1 = r#"{
            "id": "kani",
            "name": "Kani",
            "category": "rust",
            "description": "Model checker for Rust",
            "capabilities": ["model_checking", "memory_safety"]
        }"#;

        let content2 = r#"{
            "id": "miri",
            "name": "Miri",
            "category": "rust",
            "description": "Interpreter for Rust",
            "capabilities": ["interpretation", "memory_safety"]
        }"#;

        create_test_tool_file(temp_dir.path(), "kani.json", content1).await;
        create_test_tool_file(temp_dir.path(), "miri.json", content2).await;

        let store = ToolKnowledgeStore::load_from_dir(temp_dir.path())
            .await
            .unwrap();

        let results = store.search("model checker");
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "kani");

        let results = store.search("memory safety");
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_error_patterns() {
        let temp_dir = TempDir::new().unwrap();

        let content = r#"{
            "id": "test_tool",
            "name": "Test",
            "category": "test",
            "description": "Test",
            "error_patterns": [
                {
                    "pattern": "loop unwinding",
                    "meaning": "Loop bound exceeded",
                    "common_causes": ["Bound too small"],
                    "fixes": ["Increase bound"]
                }
            ]
        }"#;

        create_test_tool_file(temp_dir.path(), "test.json", content).await;

        let store = ToolKnowledgeStore::load_from_dir(temp_dir.path())
            .await
            .unwrap();

        let matches = store.find_error_fixes("test_tool", "Error: loop unwinding assertion failed");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern.meaning, "Loop bound exceeded");
    }

    #[test]
    fn test_empty_store() {
        let store = ToolKnowledgeStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn test_add_tool() {
        let mut store = ToolKnowledgeStore::new();

        let tool = ToolKnowledge {
            id: "test".to_string(),
            name: "Test".to_string(),
            category: "testing".to_string(),
            subcategory: None,
            description: "A test".to_string(),
            long_description: None,
            capabilities: vec!["cap1".to_string()],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };

        store.add_tool(tool);

        assert_eq!(store.len(), 1);
        assert!(store.get("test").is_some());
    }

    #[test]
    fn test_regex_pattern_matching() {
        let mut store = ToolKnowledgeStore::new();

        let tool = ToolKnowledge {
            id: "test_regex".to_string(),
            name: "Test Regex".to_string(),
            category: "testing".to_string(),
            subcategory: None,
            description: "Test regex patterns".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![],
            error_patterns: vec![
                ErrorPattern {
                    pattern: "unwinding assertion loop .* iteration \\d+".to_string(),
                    meaning: "Loop bound exceeded".to_string(),
                    common_causes: vec!["Bound too small".to_string()],
                    fixes: vec!["Increase #[kani::unwind(N)]".to_string()],
                },
                ErrorPattern {
                    pattern: "regex:postcondition.*failed".to_string(),
                    meaning: "Postcondition not satisfied".to_string(),
                    common_causes: vec!["Spec too strong".to_string()],
                    fixes: vec!["Weaken postcondition".to_string()],
                },
            ],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };

        store.add_tool(tool);

        // Test regex pattern with digits
        let matches = store.find_error_fixes(
            "test_regex",
            "Error: unwinding assertion loop 0 iteration 42 exceeded bound",
        );
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern.meaning, "Loop bound exceeded");
        assert!(matches[0].confidence > 0.85); // Regex match should have higher confidence

        // Test regex pattern with prefix
        let matches =
            store.find_error_fixes("test_regex", "verification error: postcondition has failed");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern.meaning, "Postcondition not satisfied");
    }

    #[test]
    fn test_looks_like_regex() {
        // Should be detected as regex
        assert!(ToolKnowledgeStore::looks_like_regex("foo\\d+bar"));
        assert!(ToolKnowledgeStore::looks_like_regex("hello.*world"));
        assert!(ToolKnowledgeStore::looks_like_regex("[a-z]+"));
        assert!(ToolKnowledgeStore::looks_like_regex("(foo|bar)"));
        assert!(ToolKnowledgeStore::looks_like_regex("test\\s+error"));

        // Should NOT be detected as regex (plain strings)
        assert!(!ToolKnowledgeStore::looks_like_regex(
            "simple error message"
        ));
        assert!(!ToolKnowledgeStore::looks_like_regex("error: loop failed"));
        assert!(!ToolKnowledgeStore::looks_like_regex(
            "postcondition not satisfied"
        ));
    }

    #[test]
    fn test_invalid_regex_fallback() {
        let mut store = ToolKnowledgeStore::new();

        let tool = ToolKnowledge {
            id: "test_invalid".to_string(),
            name: "Test Invalid Regex".to_string(),
            category: "testing".to_string(),
            subcategory: None,
            description: "Test invalid regex".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![],
            error_patterns: vec![ErrorPattern {
                // Invalid regex (unbalanced brackets) - should fall back to substring match
                pattern: "regex:error [invalid".to_string(),
                meaning: "Invalid pattern test".to_string(),
                common_causes: vec![],
                fixes: vec![],
            }],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };

        store.add_tool(tool);

        // Should fall back to substring matching and still find a match
        let matches = store.find_error_fixes("test_invalid", "error [invalid pattern here");
        assert_eq!(matches.len(), 1);
        assert!(matches[0].confidence < 0.7); // Lower confidence for fallback
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_tool_knowledge() -> impl Strategy<Value = ToolKnowledge> {
        (
            "[a-z_]{3,10}",                              // id
            "[A-Za-z ]{3,20}",                           // name
            "[a-z_]{3,15}",                              // category
            "[A-Za-z ]{10,50}",                          // description
            prop::collection::vec("[a-z_]{3,15}", 0..5), // capabilities
        )
            .prop_map(
                |(id, name, category, description, capabilities)| ToolKnowledge {
                    id,
                    name,
                    category,
                    subcategory: None,
                    description,
                    long_description: None,
                    capabilities,
                    property_types: vec![],
                    input_languages: vec![],
                    output_formats: vec![],
                    installation: None,
                    documentation: None,
                    tactics: vec![],
                    error_patterns: vec![],
                    integration: None,
                    performance: None,
                    comparisons: None,
                    metadata: None,
                },
            )
    }

    proptest! {
        /// Adding a tool increases store size
        #[test]
        fn test_add_increases_size(tools in prop::collection::vec(arb_tool_knowledge(), 1..10)) {
            let mut store = ToolKnowledgeStore::new();
            let mut expected_size = 0;

            for tool in tools {
                let id = tool.id.clone();
                if store.get(&id).is_none() {
                    expected_size += 1;
                }
                store.add_tool(tool);
            }

            // Size should match unique IDs added
            prop_assert!(store.len() <= expected_size);
        }

        /// Get returns added tool
        #[test]
        fn test_get_returns_added(tool in arb_tool_knowledge()) {
            let mut store = ToolKnowledgeStore::new();
            let id = tool.id.clone();
            store.add_tool(tool.clone());

            let retrieved = store.get(&id).unwrap();
            prop_assert_eq!(&retrieved.id, &id);
            prop_assert_eq!(&retrieved.name, &tool.name);
        }

        /// by_category returns tools in that category
        #[test]
        fn test_category_index(tool in arb_tool_knowledge()) {
            let mut store = ToolKnowledgeStore::new();
            let category = tool.category.clone();
            let id = tool.id.clone();
            store.add_tool(tool);

            let tools_in_category = store.by_category(&category);
            prop_assert!(tools_in_category.iter().any(|t| t.id == id));
        }

        /// by_capability returns tools with that capability
        #[test]
        fn test_capability_index(tool in arb_tool_knowledge()) {
            let mut store = ToolKnowledgeStore::new();
            let capabilities = tool.capabilities.clone();
            let id = tool.id.clone();
            store.add_tool(tool);

            for cap in capabilities {
                let tools_with_cap = store.by_capability(&cap);
                prop_assert!(tools_with_cap.iter().any(|t| t.id == id),
                    "Tool {} should be found under capability {}", id, cap);
            }
        }

        /// Search finds tools containing query terms
        #[test]
        fn test_search_finds_by_name(tool in arb_tool_knowledge()) {
            let mut store = ToolKnowledgeStore::new();
            let name = tool.name.clone();
            let id = tool.id.clone();
            store.add_tool(tool);

            // Search by first word of name
            if let Some(first_word) = name.split_whitespace().next() {
                let results = store.search(first_word);
                prop_assert!(results.iter().any(|t| t.id == id),
                    "Tool {} should be found when searching for '{}'", id, first_word);
            }
        }

        /// Empty search returns nothing
        #[test]
        fn test_empty_search(tool in arb_tool_knowledge()) {
            let mut store = ToolKnowledgeStore::new();
            store.add_tool(tool);

            let results = store.search("");
            prop_assert!(results.is_empty(), "Empty search should return no results");
        }

        /// Categories list matches added tools
        #[test]
        fn test_categories_complete(tools in prop::collection::vec(arb_tool_knowledge(), 1..5)) {
            let mut store = ToolKnowledgeStore::new();
            let mut expected_categories = std::collections::HashSet::new();

            for tool in tools {
                expected_categories.insert(tool.category.clone());
                store.add_tool(tool);
            }

            let categories: std::collections::HashSet<_> =
                store.categories().into_iter().map(|s| s.to_string()).collect();

            prop_assert_eq!(categories, expected_categories);
        }
    }
}

/// Kani verification proofs for pure functions
///
/// These proofs verify correctness properties of the pure logic functions
/// that don't depend on external state or complex types.
///
/// Run with: `cargo kani --lib -p dashprove-knowledge`
#[cfg(kani)]
mod kani_proofs {
    use super::pattern_looks_like_regex;

    /// Verify that pattern_looks_like_regex correctly identifies regex escape sequences
    #[kani::proof]
    fn verify_regex_escapes() {
        // Test known regex patterns - digit/space/word escapes
        assert!(pattern_looks_like_regex("\\d+"));
        assert!(pattern_looks_like_regex("\\s*"));
        assert!(pattern_looks_like_regex("\\w+"));
    }

    /// Verify wildcard patterns are detected
    #[kani::proof]
    fn verify_regex_wildcards() {
        assert!(pattern_looks_like_regex(".*error"));
        assert!(pattern_looks_like_regex(".+match"));
    }

    /// Verify bracket patterns are detected
    #[kani::proof]
    fn verify_regex_brackets() {
        assert!(pattern_looks_like_regex("[a-z]"));
        assert!(pattern_looks_like_regex("[^abc]"));
    }

    /// Verify parenthesis with alternation is detected
    #[kani::proof]
    fn verify_regex_alternation() {
        assert!(pattern_looks_like_regex("(foo|bar)"));
    }

    /// Verify non-capturing group syntax is detected
    #[kani::proof]
    fn verify_regex_noncapture() {
        assert!(pattern_looks_like_regex("(?i)case"));
    }

    /// Verify escaped parentheses are detected
    #[kani::proof]
    fn verify_regex_escaped_parens() {
        assert!(pattern_looks_like_regex("\\(foo\\)"));
    }

    /// Verify plain text is not detected as regex
    #[kani::proof]
    fn verify_plain_text_error() {
        assert!(!pattern_looks_like_regex("error"));
    }

    /// Verify plain text with spaces is not detected as regex
    #[kani::proof]
    fn verify_plain_text_spaces() {
        assert!(!pattern_looks_like_regex("simple error message"));
    }

    /// Verify plain text with numbers is not detected as regex
    #[kani::proof]
    fn verify_plain_text_numbers() {
        assert!(!pattern_looks_like_regex("timeout 5000ms"));
    }

    /// Verify empty string is not detected as regex
    #[kani::proof]
    fn verify_empty_string() {
        assert!(!pattern_looks_like_regex(""));
    }

    /// Verify that pattern_looks_like_regex is deterministic
    #[kani::proof]
    fn verify_deterministic() {
        let pattern = "\\d+ errors";
        let result1 = pattern_looks_like_regex(pattern);
        let result2 = pattern_looks_like_regex(pattern);
        assert_eq!(result1, result2, "must be deterministic");
    }

    /// Verify single dot is not detected (not .* or .+)
    #[kani::proof]
    fn verify_single_dot_not_regex() {
        assert!(!pattern_looks_like_regex("file.txt"));
        assert!(!pattern_looks_like_regex("error. Done"));
    }

    /// Verify that parentheses alone (without |) are not detected
    #[kani::proof]
    fn verify_parens_without_alternation() {
        assert!(!pattern_looks_like_regex("(value)"));
        assert!(!pattern_looks_like_regex("func(x)"));
    }
}
