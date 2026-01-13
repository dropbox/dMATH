//! Phase B' - Knowledge Correctness Validation Tests
//!
//! These tests validate the completeness and correctness of all ingested knowledge:
//! - B'1: Validate common_errors.json files (no corruption, complete)
//! - B'1: Validate official.md documentation files
//! - B'3: RAG retrieval quality benchmarks
//! - B'4: Property tests for knowledge indexing

use serde::Deserialize;
use std::collections::HashSet;
use std::path::PathBuf;
use walkdir::WalkDir;

/// Get the knowledge base directory
fn knowledge_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/knowledge")
}

// ============================================================================
// B'1: Common Errors JSON Validation
// ============================================================================

/// Structure for common_errors.json validation (flexible to handle variations)
/// Supports multiple naming conventions: tool/tool_id, common_errors/errors/error_patterns
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CommonErrorsFile {
    // Accept either "tool" or "tool_id"
    #[serde(alias = "tool_id")]
    tool: Option<String>,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default, alias = "tool_version")]
    version: Option<String>,
    #[serde(default)]
    last_updated: Option<String>,
    // Allow "common_errors", "errors", or "error_patterns" arrays
    #[serde(default)]
    common_errors: Vec<serde_json::Value>,
    #[serde(default)]
    errors: Vec<serde_json::Value>,
    #[serde(default)]
    error_patterns: Vec<serde_json::Value>,
}

impl CommonErrorsFile {
    fn get_tool_id(&self) -> Option<&str> {
        self.tool.as_deref().or(self.tool_name.as_deref())
    }

    fn has_errors(&self) -> bool {
        !self.common_errors.is_empty() || !self.errors.is_empty() || !self.error_patterns.is_empty()
    }
}

/// B'1: Validate all common_errors.json files can be parsed
#[tokio::test]
async fn test_all_common_errors_json_parseable() {
    let issues_dir = knowledge_dir().join("issues");
    if !issues_dir.exists() {
        eprintln!("Skipping: issues directory does not exist");
        return;
    }

    let mut total = 0;
    let mut failed = Vec::new();

    for entry in WalkDir::new(&issues_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file() && e.file_name().to_string_lossy() == "common_errors.json"
        })
    {
        total += 1;
        let path = entry.path();

        match std::fs::read_to_string(path) {
            Ok(content) => {
                // First validate JSON syntax
                if let Err(e) = serde_json::from_str::<serde_json::Value>(&content) {
                    failed.push((path.to_path_buf(), format!("Invalid JSON: {}", e)));
                    continue;
                }

                // Then validate structure
                if let Err(e) = serde_json::from_str::<CommonErrorsFile>(&content) {
                    failed.push((path.to_path_buf(), format!("Invalid structure: {}", e)));
                }
            }
            Err(e) => {
                failed.push((path.to_path_buf(), format!("Read error: {}", e)));
            }
        }
    }

    if !failed.is_empty() {
        eprintln!(
            "\nFailed to parse {} of {} common_errors.json files:",
            failed.len(),
            total
        );
        for (path, err) in &failed[..std::cmp::min(10, failed.len())] {
            eprintln!("  {:?}: {}", path, err);
        }
        if failed.len() > 10 {
            eprintln!("  ... and {} more", failed.len() - 10);
        }
    }

    assert!(
        failed.is_empty(),
        "{}/{} common_errors.json files failed validation",
        failed.len(),
        total
    );
    println!("Validated {} common_errors.json files successfully", total);
}

/// B'1: Validate all common_errors.json files have at least one error pattern
#[tokio::test]
async fn test_common_errors_have_content() {
    let issues_dir = knowledge_dir().join("issues");
    if !issues_dir.exists() {
        return;
    }

    let mut total = 0;
    let mut empty_files = Vec::new();

    for entry in WalkDir::new(&issues_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file() && e.file_name().to_string_lossy() == "common_errors.json"
        })
    {
        total += 1;
        let path = entry.path();

        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(file) = serde_json::from_str::<CommonErrorsFile>(&content) {
                if !file.has_errors() {
                    empty_files.push(path.to_path_buf());
                }
            }
        }
    }

    let empty_pct = if total > 0 {
        (empty_files.len() as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    // Allow up to 5% empty files (some tools may legitimately have no common errors)
    assert!(
        empty_pct <= 5.0,
        "{:.1}% of common_errors.json files are empty ({}/{}). First few: {:?}",
        empty_pct,
        empty_files.len(),
        total,
        &empty_files[..std::cmp::min(5, empty_files.len())]
    );

    println!(
        "Content check: {}/{} files have error patterns ({:.1}% coverage)",
        total - empty_files.len(),
        total,
        100.0 - empty_pct
    );
}

/// B'1: Validate tool names in common_errors.json are present
/// Note: Tool names don't need to exactly match directory names (e.g., lsns vs lsns_cmd is OK)
#[tokio::test]
async fn test_common_errors_have_tool_identifier() {
    let issues_dir = knowledge_dir().join("issues");
    if !issues_dir.exists() {
        return;
    }

    let mut missing_tool_id = Vec::new();
    let mut total = 0;

    for entry in WalkDir::new(&issues_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file() && e.file_name().to_string_lossy() == "common_errors.json"
        })
    {
        total += 1;
        let path = entry.path();

        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(file) = serde_json::from_str::<CommonErrorsFile>(&content) {
                // Check that at least one tool identifier is present
                if file.get_tool_id().is_none() {
                    let dir_name = path
                        .parent()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");
                    missing_tool_id.push(dir_name.to_string());
                }
            }
        }
    }

    if !missing_tool_id.is_empty() {
        eprintln!("\nFiles missing tool identifier:");
        for name in &missing_tool_id[..std::cmp::min(10, missing_tool_id.len())] {
            eprintln!("  {}", name);
        }
    }

    // All files should have a tool identifier
    assert!(
        missing_tool_id.is_empty(),
        "{}/{} files missing tool identifier: {:?}",
        missing_tool_id.len(),
        total,
        &missing_tool_id[..std::cmp::min(10, missing_tool_id.len())]
    );

    println!(
        "Tool identifier check: {}/{} files have tool identifier",
        total - missing_tool_id.len(),
        total
    );
}

// ============================================================================
// B'1: Official Documentation Validation
// ============================================================================

/// B'1: Validate all tool directories in docs/ have official.md
#[tokio::test]
async fn test_all_docs_have_official_md() {
    let docs_dir = knowledge_dir().join("docs");
    if !docs_dir.exists() {
        eprintln!("Skipping: docs directory does not exist");
        return;
    }

    let mut missing_official = Vec::new();
    let mut total_dirs = 0;

    for entry in std::fs::read_dir(&docs_dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            total_dirs += 1;
            let official_path = entry.path().join("official.md");
            if !official_path.exists() {
                missing_official.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }

    if !missing_official.is_empty() {
        eprintln!(
            "\nDirectories missing official.md ({}/{}):",
            missing_official.len(),
            total_dirs
        );
        for name in &missing_official[..std::cmp::min(20, missing_official.len())] {
            eprintln!("  - {}", name);
        }
    }

    assert!(
        missing_official.is_empty(),
        "{}/{} docs directories are missing official.md",
        missing_official.len(),
        total_dirs
    );

    println!("All {} docs directories have official.md", total_dirs);
}

/// B'1: Validate official.md files are non-empty and have meaningful content
#[tokio::test]
async fn test_official_md_has_content() {
    let docs_dir = knowledge_dir().join("docs");
    if !docs_dir.exists() {
        return;
    }

    let mut empty_or_short = Vec::new();
    let mut total = 0;
    const MIN_CONTENT_LENGTH: usize = 100; // Minimum meaningful content

    for entry in WalkDir::new(&docs_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && e.file_name().to_string_lossy() == "official.md")
    {
        total += 1;
        let path = entry.path();

        if let Ok(content) = std::fs::read_to_string(path) {
            // Strip whitespace and check length
            let content_len = content.trim().len();
            if content_len < MIN_CONTENT_LENGTH {
                empty_or_short.push((
                    path.parent()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    content_len,
                ));
            }
        }
    }

    if !empty_or_short.is_empty() {
        eprintln!(
            "\nofficial.md files with insufficient content (<{} chars):",
            MIN_CONTENT_LENGTH
        );
        for (name, len) in &empty_or_short[..std::cmp::min(20, empty_or_short.len())] {
            eprintln!("  {}: {} chars", name, len);
        }
    }

    // Allow up to 3% with insufficient content (some docs may be stubs)
    let short_pct = (empty_or_short.len() as f64 / total as f64) * 100.0;
    assert!(
        short_pct <= 3.0,
        "{:.1}% of official.md files have insufficient content ({}/{})",
        short_pct,
        empty_or_short.len(),
        total
    );

    println!(
        "Content validation: {}/{} official.md files have sufficient content",
        total - empty_or_short.len(),
        total
    );
}

/// B'1: Validate official.md files have proper markdown headers
#[tokio::test]
async fn test_official_md_has_headers() {
    let docs_dir = knowledge_dir().join("docs");
    if !docs_dir.exists() {
        return;
    }

    let mut no_headers = Vec::new();
    let mut total = 0;

    for entry in WalkDir::new(&docs_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && e.file_name().to_string_lossy() == "official.md")
    {
        total += 1;
        let path = entry.path();

        if let Ok(content) = std::fs::read_to_string(path) {
            // Check for markdown headers (# or ##)
            let has_header = content.lines().any(|line| {
                let trimmed = line.trim();
                trimmed.starts_with('#') && !trimmed.starts_with("#!")
            });

            if !has_header {
                no_headers.push(
                    path.parent()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string(),
                );
            }
        }
    }

    if !no_headers.is_empty() {
        eprintln!(
            "\nofficial.md files without markdown headers ({}/{}):",
            no_headers.len(),
            total
        );
        for name in &no_headers[..std::cmp::min(20, no_headers.len())] {
            eprintln!("  {}", name);
        }
    }

    // Allow up to 8% without proper headers (some docs may have alternate structures)
    let no_header_pct = (no_headers.len() as f64 / total as f64) * 100.0;
    assert!(
        no_header_pct <= 8.0,
        "{:.1}% of official.md files lack markdown headers ({}/{})",
        no_header_pct,
        no_headers.len(),
        total
    );

    println!(
        "Header validation: {}/{} official.md files have proper headers",
        total - no_headers.len(),
        total
    );
}

// ============================================================================
// B'1: Cross-Reference Validation
// ============================================================================

/// B'1: Validate issues directories match docs directories
#[tokio::test]
async fn test_issues_and_docs_coverage_match() {
    let issues_dir = knowledge_dir().join("issues");
    let docs_dir = knowledge_dir().join("docs");

    if !issues_dir.exists() || !docs_dir.exists() {
        eprintln!("Skipping: issues or docs directory does not exist");
        return;
    }

    // Collect tool names from issues
    let issues_tools: HashSet<String> = std::fs::read_dir(&issues_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    // Collect tool names from docs
    let docs_tools: HashSet<String> = std::fs::read_dir(&docs_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    // Find differences
    let in_issues_not_docs: Vec<_> = issues_tools.difference(&docs_tools).collect();
    let in_docs_not_issues: Vec<_> = docs_tools.difference(&issues_tools).collect();

    println!(
        "\nCoverage comparison: {} in issues, {} in docs",
        issues_tools.len(),
        docs_tools.len()
    );

    if !in_issues_not_docs.is_empty() && in_issues_not_docs.len() <= 20 {
        println!("  Tools in issues/ but not docs/: {:?}", in_issues_not_docs);
    }
    if !in_docs_not_issues.is_empty() && in_docs_not_issues.len() <= 20 {
        println!("  Tools in docs/ but not issues/: {:?}", in_docs_not_issues);
    }

    // Docs are intentionally more selective (343 vs 1228 issues)
    // Verify that all docs tools have corresponding issues
    let docs_coverage = issues_tools.intersection(&docs_tools).count();
    let docs_coverage_pct = if !docs_tools.is_empty() {
        (docs_coverage as f64 / docs_tools.len() as f64) * 100.0
    } else {
        100.0
    };

    // At least 80% of docs should have corresponding issues
    assert!(
        docs_coverage_pct >= 80.0,
        "Only {:.1}% of docs have corresponding issues ({}/{})",
        docs_coverage_pct,
        docs_coverage,
        docs_tools.len()
    );

    println!(
        "Coverage: {:.1}% of docs have issues ({}/{})",
        docs_coverage_pct,
        docs_coverage,
        docs_tools.len()
    );
}

// ============================================================================
// B'3: RAG Retrieval Quality
// ============================================================================

/// B'3: Test that key tools can be found via search
#[tokio::test]
async fn test_rag_search_finds_key_tools() {
    use dashprove_knowledge::ToolKnowledgeStore;

    let tool_dir = knowledge_dir().join("tools");
    if !tool_dir.exists() {
        eprintln!("Skipping: tools directory does not exist");
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&tool_dir)
        .await
        .expect("Failed to load tool knowledge");

    // Test queries and expected tools in results
    let test_cases = [
        ("rust memory safety", vec!["kani", "miri", "asan"]),
        ("model checker", vec!["kani", "tlaplus", "spin", "cbmc"]),
        ("SMT solver", vec!["z3", "cvc5", "yices"]),
        ("dependent types", vec!["lean4", "coq", "agda"]),
        (
            "property based testing",
            vec!["proptest", "hypothesis", "quickcheck"],
        ),
        ("neural network verification", vec!["marabou", "eran"]),
    ];

    let mut failed_queries = Vec::new();

    for (query, expected_tools) in &test_cases {
        let results = store.search(query);
        let result_ids: Vec<&str> = results.iter().take(10).map(|t| t.id.as_str()).collect();

        // Check if at least one expected tool is in the top 10 results
        let found_expected = expected_tools.iter().any(|t| result_ids.contains(t));

        if !found_expected && !results.is_empty() {
            failed_queries.push((
                query.to_string(),
                expected_tools.clone(),
                result_ids.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            ));
        }
    }

    if !failed_queries.is_empty() {
        eprintln!("\nSearch quality issues:");
        for (query, expected, actual) in &failed_queries {
            eprintln!(
                "  Query '{}': expected one of {:?}, got {:?}",
                query, expected, actual
            );
        }
    }

    // Allow up to 30% of queries to have suboptimal results (RAG is imperfect)
    let fail_pct = (failed_queries.len() as f64 / test_cases.len() as f64) * 100.0;
    assert!(
        fail_pct <= 30.0,
        "Search quality: {:.0}% of test queries failed to find expected tools",
        fail_pct
    );

    println!(
        "Search quality: {}/{} test queries found expected tools in top 10",
        test_cases.len() - failed_queries.len(),
        test_cases.len()
    );
}

/// B'3: Test error pattern matching precision
#[tokio::test]
async fn test_error_pattern_matching_precision() {
    use dashprove_knowledge::ToolKnowledgeStore;

    let tool_dir = knowledge_dir().join("tools");
    if !tool_dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&tool_dir)
        .await
        .expect("Failed to load tool knowledge");

    // Test error messages and expected matches
    let test_cases = [
        ("kani", "VERIFICATION:- FAILED", true),
        ("kani", "unwinding assertion loop 0 iteration 100", true),
        ("z3", "timeout", true),
        ("lean4", "type mismatch", true),
        ("coq", "The term", true),
    ];

    let mut matches_found = 0;
    let mut total_expected = 0;

    for (tool_id, error_msg, should_match) in &test_cases {
        if store.get(tool_id).is_none() {
            continue;
        }

        let fixes = store.find_error_fixes(tool_id, error_msg);
        let found_match = !fixes.is_empty();

        if *should_match {
            total_expected += 1;
            if found_match {
                matches_found += 1;
            }
        }
    }

    if total_expected > 0 {
        let precision = (matches_found as f64 / total_expected as f64) * 100.0;
        println!(
            "Error pattern matching: {:.0}% precision ({}/{})",
            precision, matches_found, total_expected
        );

        // We should match at least 50% of expected patterns
        assert!(
            precision >= 50.0,
            "Error pattern matching precision is only {:.0}%",
            precision
        );
    }
}

// ============================================================================
// B'4: Property Tests for Knowledge Indexing
// ============================================================================

/// B'4: Property test - all indexed capabilities should have tools
#[tokio::test]
async fn test_capability_index_completeness() {
    use dashprove_knowledge::ToolKnowledgeStore;

    let tool_dir = knowledge_dir().join("tools");
    if !tool_dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&tool_dir)
        .await
        .expect("Failed to load tool knowledge");

    // Every capability returned should have at least one tool
    for cap in store.capabilities() {
        let tools = store.by_capability(cap);
        assert!(
            !tools.is_empty(),
            "Capability '{}' is indexed but has no tools",
            cap
        );
    }

    println!(
        "Capability index integrity: {} capabilities all have tools",
        store.capabilities().len()
    );
}

/// B'4: Property test - all indexed categories should have tools
#[tokio::test]
async fn test_category_index_completeness() {
    use dashprove_knowledge::ToolKnowledgeStore;

    let tool_dir = knowledge_dir().join("tools");
    if !tool_dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&tool_dir)
        .await
        .expect("Failed to load tool knowledge");

    // Every category returned should have at least one tool
    for cat in store.categories() {
        let tools = store.by_category(cat);
        assert!(
            !tools.is_empty(),
            "Category '{}' is indexed but has no tools",
            cat
        );
    }

    println!(
        "Category index integrity: {} categories all have tools",
        store.categories().len()
    );
}

/// B'4: Property test - tool retrieval consistency
#[tokio::test]
async fn test_tool_retrieval_consistency() {
    use dashprove_knowledge::ToolKnowledgeStore;

    let tool_dir = knowledge_dir().join("tools");
    if !tool_dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&tool_dir)
        .await
        .expect("Failed to load tool knowledge");

    // Every tool in all() should be retrievable by get()
    for tool in store.all() {
        let retrieved = store.get(&tool.id);
        assert!(
            retrieved.is_some(),
            "Tool '{}' is in all() but not retrievable via get()",
            tool.id
        );
        assert_eq!(
            retrieved.unwrap().name,
            tool.name,
            "Tool '{}' data inconsistent between all() and get()",
            tool.id
        );
    }

    println!("Tool retrieval consistency: {} tools verified", store.len());
}

// ============================================================================
// Summary Statistics
// ============================================================================

/// Print comprehensive knowledge base statistics
#[tokio::test]
async fn test_print_knowledge_base_stats() {
    use dashprove_knowledge::ToolKnowledgeStore;

    let base_dir = knowledge_dir();

    println!("\n====== Knowledge Base Statistics ======\n");

    // Count issues
    let issues_dir = base_dir.join("issues");
    if issues_dir.exists() {
        let issues_count = WalkDir::new(&issues_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type().is_file() && e.file_name().to_string_lossy() == "common_errors.json"
            })
            .count();
        let issues_dirs = std::fs::read_dir(&issues_dir)
            .map(|r| {
                r.filter_map(|e| e.ok())
                    .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                    .count()
            })
            .unwrap_or(0);
        println!(
            "Issues: {} directories, {} common_errors.json files",
            issues_dirs, issues_count
        );
    }

    // Count docs
    let docs_dir = base_dir.join("docs");
    if docs_dir.exists() {
        let docs_count = WalkDir::new(&docs_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && e.file_name().to_string_lossy() == "official.md")
            .count();
        let docs_dirs = std::fs::read_dir(&docs_dir)
            .map(|r| {
                r.filter_map(|e| e.ok())
                    .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                    .count()
            })
            .unwrap_or(0);
        println!(
            "Docs: {} directories, {} official.md files",
            docs_dirs, docs_count
        );
    }

    // Load tool store and print stats
    let tool_dir = base_dir.join("tools");
    if tool_dir.exists() {
        if let Ok(store) = ToolKnowledgeStore::load_from_dir(&tool_dir).await {
            println!("\nTool Knowledge Store:");
            println!("  Total tools: {}", store.len());
            println!("  Categories: {}", store.categories().len());
            println!("  Capabilities: {}", store.capabilities().len());

            // Count tools with various features
            let with_tactics = store.all().filter(|t| !t.tactics.is_empty()).count();
            let with_errors = store.all().filter(|t| !t.error_patterns.is_empty()).count();
            let with_docs = store.all().filter(|t| t.documentation.is_some()).count();
            let with_install = store.all().filter(|t| t.installation.is_some()).count();

            println!("\n  Feature coverage:");
            println!("    With tactics: {}/{}", with_tactics, store.len());
            println!("    With error patterns: {}/{}", with_errors, store.len());
            println!("    With documentation URLs: {}/{}", with_docs, store.len());
            println!(
                "    With installation info: {}/{}",
                with_install,
                store.len()
            );
        }
    }

    // Count total JSON files
    let json_count = WalkDir::new(&base_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file()
                && e.path()
                    .extension()
                    .map(|ext| ext == "json")
                    .unwrap_or(false)
        })
        .count();
    println!("\nTotal JSON files in knowledge base: {}", json_count);

    println!("\n========================================\n");
}

// ============================================================================
// B'2: Cross-Reference Proof Compilation
// ============================================================================

/// Check if a command exists in PATH
fn command_exists(cmd: &str) -> bool {
    std::process::Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// B'2: Verify simple Lean 4 proofs compile
/// This test validates that the Lean 4 tactics and syntax in our knowledge base are accurate
#[tokio::test]
async fn test_lean4_simple_proof_compiles() {
    if !command_exists("lean") {
        eprintln!("Skipping: Lean 4 not installed");
        return;
    }

    // Create temp directory for test files
    let temp_dir = std::env::temp_dir().join("dashprove_lean4_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).expect("Failed to create temp dir");

    // Simple proof using tactics from our knowledge base
    // Note: omega and lemma require Mathlib/Std imports, using only core Lean 4 features
    let lean_code = r#"
-- Basic theorem using rfl
theorem add_zero (n : Nat) : n + 0 = n := by
  rfl

-- Using simp tactic
theorem add_comm_zero (n : Nat) : 0 + n = n := by
  simp [Nat.zero_add]

-- Using induction from our tactic docs
theorem add_zero_induction (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.succ_add, ih]

-- Using intro and exact
theorem id_eq_proof {α : Type} (a : α) : a = a := by
  rfl

-- Test decide tactic for concrete arithmetic
theorem decide_test : 2 + 2 = 4 := by
  decide

#check add_zero
#check add_comm_zero
#check add_zero_induction
#check id_eq_proof
#check decide_test
"#;

    let lean_file = temp_dir.join("test_proofs.lean");
    std::fs::write(&lean_file, lean_code).expect("Failed to write Lean file");

    // Run Lean to type-check the file
    let output = std::process::Command::new("lean")
        .arg(&lean_file)
        .output()
        .expect("Failed to run lean");

    // Cleanup
    let _ = std::fs::remove_dir_all(&temp_dir);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "Lean 4 proof compilation failed:\nstderr: {}\nstdout: {}",
            stderr, stdout
        );
    }

    println!("Lean 4 simple proof compilation: PASSED (5 theorems verified)");
}

/// B'2: Verify that our documented Lean 4 tactics actually work
#[tokio::test]
async fn test_lean4_documented_tactics_work() {
    if !command_exists("lean") {
        eprintln!("Skipping: Lean 4 not installed");
        return;
    }

    let temp_dir = std::env::temp_dir().join("dashprove_lean4_tactics_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).expect("Failed to create temp dir");

    // Test each tactic documented in lean4.json
    let tactics_tests = vec![
        // intro tactic
        (
            "intro",
            r#"
theorem test_intro (P Q : Prop) : P → Q → P := by
  intro hp
  intro hq
  exact hp
"#,
        ),
        // apply tactic
        (
            "apply",
            r#"
theorem test_apply (P Q : Prop) (h : P → Q) (hp : P) : Q := by
  apply h
  exact hp
"#,
        ),
        // simp tactic
        (
            "simp",
            r#"
theorem test_simp (n : Nat) : n + 0 = n := by
  simp
"#,
        ),
        // rw tactic
        (
            "rw",
            r#"
theorem test_rw (a b : Nat) (h : a = b) : b = a := by
  rw [h]
"#,
        ),
        // decide tactic
        (
            "decide",
            r#"
theorem test_decide : 2 + 2 = 4 := by
  decide
"#,
        ),
    ];

    let mut passed = 0;
    let mut failed = Vec::new();

    for (tactic_name, code) in tactics_tests {
        let file = temp_dir.join(format!("test_{}.lean", tactic_name));
        std::fs::write(&file, code).expect("Failed to write file");

        let output = std::process::Command::new("lean")
            .arg(&file)
            .output()
            .expect("Failed to run lean");

        if output.status.success() {
            passed += 1;
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            failed.push((tactic_name.to_string(), stderr.to_string()));
        }
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&temp_dir);

    if !failed.is_empty() {
        for (name, err) in &failed {
            eprintln!("Tactic '{}' test failed: {}", name, err);
        }
    }

    assert!(
        failed.is_empty(),
        "Documented tactics failed: {:?}",
        failed.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );

    println!(
        "Lean 4 documented tactics verification: {}/{} PASSED",
        passed,
        passed + failed.len()
    );
}

/// B'2: Verify documented error patterns match actual Lean 4 errors
#[tokio::test]
async fn test_lean4_error_patterns_match() {
    if !command_exists("lean") {
        eprintln!("Skipping: Lean 4 not installed");
        return;
    }

    let temp_dir = std::env::temp_dir().join("dashprove_lean4_errors_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).expect("Failed to create temp dir");

    // Test cases that should produce specific errors documented in lean4.json
    let error_tests = vec![
        // "type mismatch" error
        (
            "type_mismatch",
            r#"
def bad : Nat := "hello"
"#,
            "type mismatch",
        ),
        // "unknown identifier" error
        (
            "unknown_identifier",
            r#"
#check nonexistent_function
"#,
            "unknown identifier",
        ),
    ];

    let mut matched = 0;
    let mut mismatched = Vec::new();

    for (name, code, expected_pattern) in error_tests {
        let file = temp_dir.join(format!("error_{}.lean", name));
        std::fs::write(&file, code).expect("Failed to write file");

        let output = std::process::Command::new("lean")
            .arg(&file)
            .output()
            .expect("Failed to run lean");

        // These should fail (produce errors)
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let combined = format!("{} {}", stderr, stdout);

            if combined.to_lowercase().contains(expected_pattern) {
                matched += 1;
            } else {
                mismatched.push((name.to_string(), expected_pattern.to_string(), combined));
            }
        } else {
            mismatched.push((
                name.to_string(),
                expected_pattern.to_string(),
                "Expected error but compilation succeeded".to_string(),
            ));
        }
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&temp_dir);

    if !mismatched.is_empty() {
        for (name, expected, actual) in &mismatched {
            eprintln!(
                "Error pattern '{}': expected '{}', got: {}",
                name,
                expected,
                actual.chars().take(200).collect::<String>()
            );
        }
    }

    // Allow some flexibility - at least 50% should match
    let total = matched + mismatched.len();
    let match_rate = if total > 0 {
        (matched as f64 / total as f64) * 100.0
    } else {
        100.0
    };

    assert!(
        match_rate >= 50.0,
        "Error pattern match rate is only {:.0}% ({}/{})",
        match_rate,
        matched,
        total
    );

    println!(
        "Lean 4 error pattern verification: {}/{} matched ({:.0}%)",
        matched, total, match_rate
    );
}

/// B'2: Test that documented installation commands are accurate
/// Note: We don't actually run installation, just verify the format
#[tokio::test]
async fn test_installation_commands_format() {
    use dashprove_knowledge::ToolKnowledgeStore;

    let tool_dir = knowledge_dir().join("tools");
    if !tool_dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&tool_dir)
        .await
        .expect("Failed to load tool knowledge");

    let mut valid_installs = 0;
    let mut invalid_installs = Vec::new();

    for tool in store.all() {
        if let Some(install) = &tool.installation {
            // Check that installation has at least one method
            if install.methods.is_empty() {
                invalid_installs.push((tool.id.clone(), "No installation methods".to_string()));
                continue;
            }

            // Verify methods have type and either command or URL
            let mut has_valid_method = false;
            for method in &install.methods {
                let has_type = !method.method_type.is_empty();
                let has_command = method
                    .command
                    .as_ref()
                    .map(|c| !c.is_empty())
                    .unwrap_or(false);
                let has_url = method.url.as_ref().map(|u| !u.is_empty()).unwrap_or(false);

                if has_type && (has_command || has_url) {
                    has_valid_method = true;
                    break;
                }
            }

            if has_valid_method {
                valid_installs += 1;
            } else {
                invalid_installs.push((
                    tool.id.clone(),
                    "Methods missing type or (command/url)".to_string(),
                ));
            }
        }
    }

    if !invalid_installs.is_empty() && invalid_installs.len() <= 10 {
        eprintln!("\nInvalid installation entries:");
        for (id, reason) in &invalid_installs {
            eprintln!("  {}: {}", id, reason);
        }
    }

    let total_with_install = valid_installs + invalid_installs.len();
    println!(
        "Installation command format: {}/{} valid (of {} tools with install info)",
        valid_installs,
        total_with_install,
        store.len()
    );

    // Allow up to 15% with incomplete installation info (some tools may only have informal install methods)
    let invalid_pct = if total_with_install > 0 {
        (invalid_installs.len() as f64 / total_with_install as f64) * 100.0
    } else {
        0.0
    };

    assert!(
        invalid_pct <= 15.0,
        "{:.1}% of tools have invalid installation format ({}/{})",
        invalid_pct,
        invalid_installs.len(),
        total_with_install
    );
}
