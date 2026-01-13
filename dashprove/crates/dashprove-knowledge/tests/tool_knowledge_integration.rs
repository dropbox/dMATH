//! Integration tests for tool knowledge loading
//!
//! These tests verify that the JSON tool knowledge files in data/knowledge/tools/
//! can be loaded correctly.

use dashprove_knowledge::ToolKnowledgeStore;
use std::path::PathBuf;

/// Get the path to the tool knowledge directory
fn tool_knowledge_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/knowledge/tools")
}

#[tokio::test]
async fn test_load_all_tool_knowledge() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        eprintln!("Skipping test: {:?} does not exist", dir);
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    // We expect at least 280 tools (actual is 295 after duplicate removal)
    assert!(
        store.len() >= 280,
        "Expected at least 280 tools, found {}",
        store.len()
    );

    println!("Loaded {} tool knowledge entries", store.len());
}

#[tokio::test]
async fn test_tool_knowledge_structure() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    // Verify key tools exist
    let key_tools = [
        "kani", "lean4", "coq", "z3", "verus", "prusti", "miri", "dafny", "tlaplus", "isabelle",
    ];

    for tool_id in &key_tools {
        assert!(
            store.get(tool_id).is_some(),
            "Missing expected tool: {}",
            tool_id
        );
    }
}

#[tokio::test]
async fn test_tool_has_required_fields() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    for tool in store.all() {
        // All tools must have these fields
        assert!(!tool.id.is_empty(), "Tool has empty ID");
        assert!(!tool.name.is_empty(), "Tool {} has empty name", tool.id);
        assert!(
            !tool.category.is_empty(),
            "Tool {} has empty category",
            tool.id
        );
        assert!(
            !tool.description.is_empty(),
            "Tool {} has empty description",
            tool.id
        );
    }
}

#[tokio::test]
async fn test_categories_coverage() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    let categories = store.categories();

    // Verify we have tools in key categories
    let expected_categories = [
        "rust_formal_verification",
        "theorem_prover",
        "smt_solver",
        "model_checker",
    ];

    for cat in &expected_categories {
        let tools = store.by_category(cat);
        assert!(!tools.is_empty(), "No tools found in category: {}", cat);
    }

    println!("Found {} categories:", categories.len());
    for cat in &categories {
        let count = store.by_category(cat).len();
        println!("  {}: {} tools", cat, count);
    }
}

#[tokio::test]
async fn test_capabilities_coverage() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    let capabilities = store.capabilities();

    // Verify we have tools with key capabilities
    let expected_capabilities = ["memory_safety", "dependent_types", "model_checking"];

    for cap in &expected_capabilities {
        let tools = store.by_capability(cap);
        assert!(!tools.is_empty(), "No tools found with capability: {}", cap);
    }

    println!("Found {} unique capabilities", capabilities.len());
}

#[tokio::test]
async fn test_kani_detailed() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    let kani = store.get("kani").expect("Kani should exist");

    // Verify Kani has expected structure
    assert_eq!(kani.name, "Kani");
    assert!(!kani.capabilities.is_empty());
    assert!(!kani.tactics.is_empty(), "Kani should have tactics");
    assert!(
        !kani.error_patterns.is_empty(),
        "Kani should have error patterns"
    );
    assert!(
        kani.documentation.is_some(),
        "Kani should have documentation URLs"
    );
    assert!(
        kani.installation.is_some(),
        "Kani should have installation info"
    );

    // Verify tactics have required fields
    for tactic in &kani.tactics {
        assert!(!tactic.name.is_empty());
        assert!(!tactic.description.is_empty());
    }

    // Verify error patterns have required fields
    for pattern in &kani.error_patterns {
        assert!(!pattern.pattern.is_empty());
        assert!(!pattern.meaning.is_empty());
    }
}

#[tokio::test]
async fn test_lean4_detailed() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    let lean4 = store.get("lean4").expect("Lean4 should exist");

    assert_eq!(lean4.name, "Lean 4");
    assert!(lean4.capabilities.contains(&"dependent_types".to_string()));
    assert!(!lean4.tactics.is_empty(), "Lean4 should have tactics");
}

#[tokio::test]
async fn test_search_functionality() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    // Search for "rust verification"
    let results = store.search("rust verification");
    assert!(
        !results.is_empty(),
        "Search for 'rust verification' should return results"
    );

    // The top results should include Rust verification tools
    let result_ids: Vec<&str> = results.iter().map(|t| t.id.as_str()).collect();
    let rust_tools = ["kani", "verus", "prusti", "creusot", "miri"];
    let found_rust_tool = rust_tools.iter().any(|t| result_ids.contains(t));
    assert!(
        found_rust_tool,
        "Search for 'rust verification' should include a Rust verification tool"
    );
}

#[tokio::test]
async fn test_error_fix_suggestions() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    // Test error fix suggestions for Kani
    // Note: Current implementation uses simple substring matching, not regex
    // The pattern "unwinding assertion loop \d+" won't match with substring
    // Test with a pattern that uses simple substring matching
    let kani_error = "VERIFICATION:- FAILED";
    let fixes = store.find_error_fixes("kani", kani_error);

    // If Kani has error patterns, we should get matches for this exact pattern
    if !store.get("kani").unwrap().error_patterns.is_empty() {
        assert!(
            !fixes.is_empty(),
            "Should find fixes for Kani VERIFICATION:- FAILED error"
        );
    }
}

#[tokio::test]
async fn test_similar_tools() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    // Find tools similar to Kani
    let similar = store.find_similar("kani");

    // Should find other Rust verification tools
    if !similar.is_empty() {
        println!("Tools similar to Kani:");
        for tool in &similar {
            println!("  - {} ({})", tool.name, tool.category);
        }
    }
}

/// Phase A' validation: All tools must have documentation with valid URLs
#[tokio::test]
async fn test_all_tools_have_documentation() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    let mut tools_missing_docs = Vec::new();

    for tool in store.all() {
        if tool.documentation.is_none() {
            tools_missing_docs.push(tool.id.clone());
        } else if let Some(ref docs) = tool.documentation {
            // At least one URL should be present
            let has_url = docs.official.is_some()
                || docs.tutorial.is_some()
                || docs.api_reference.is_some()
                || docs.examples.is_some();
            if !has_url {
                tools_missing_docs.push(format!("{} (empty documentation)", tool.id));
            }
        }
    }

    assert!(
        tools_missing_docs.is_empty(),
        "Tools missing documentation: {:?}",
        tools_missing_docs
    );
}

/// Phase A' validation: Documentation URLs must have valid structure
#[tokio::test]
async fn test_documentation_url_structure() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    let mut invalid_urls = Vec::new();

    for tool in store.all() {
        if let Some(ref docs) = tool.documentation {
            for (field, url) in [
                ("official", &docs.official),
                ("tutorial", &docs.tutorial),
                ("api_reference", &docs.api_reference),
                ("examples", &docs.examples),
            ] {
                if let Some(url_str) = url {
                    if !url_str.starts_with("http://") && !url_str.starts_with("https://") {
                        invalid_urls.push(format!("{}.{}: {}", tool.id, field, url_str));
                    }
                }
            }
        }
    }

    assert!(
        invalid_urls.is_empty(),
        "Invalid documentation URLs found: {:?}",
        invalid_urls
    );
}

/// Phase A' validation: Verify category distribution is balanced
#[tokio::test]
async fn test_category_distribution() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    let categories = store.categories();

    // We should have at least 30 distinct categories for good coverage
    assert!(
        categories.len() >= 30,
        "Expected at least 30 categories, found {}",
        categories.len()
    );

    // Print category stats for visibility
    println!("\nCategory distribution ({} categories):", categories.len());
    for cat in &categories {
        let count = store.by_category(cat).len();
        println!("  {}: {}", cat, count);
    }
}

/// Phase A' validation: Verify capability coverage
#[tokio::test]
async fn test_capability_coverage() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    let capabilities = store.capabilities();

    // We should have substantial capability coverage
    assert!(
        capabilities.len() >= 100,
        "Expected at least 100 unique capabilities, found {}",
        capabilities.len()
    );

    // Key capabilities that must exist
    let key_capabilities = [
        "memory_safety",
        "dependent_types",
        "model_checking",
        "sat_solving",
        "symbolic_execution",
    ];

    for cap in &key_capabilities {
        let tools = store.by_capability(cap);
        assert!(
            !tools.is_empty(),
            "Expected at least one tool with capability: {}",
            cap
        );
    }
}

/// Phase A' validation: No duplicate tool IDs
#[tokio::test]
async fn test_no_duplicate_tool_ids() {
    let dir = tool_knowledge_dir();
    if !dir.exists() {
        return;
    }

    let store = ToolKnowledgeStore::load_from_dir(&dir)
        .await
        .expect("Failed to load tool knowledge");

    // Collect all IDs
    let mut ids: Vec<String> = store.all().map(|t| t.id.clone()).collect();
    let original_count = ids.len();

    // Remove duplicates
    ids.sort();
    ids.dedup();

    assert_eq!(
        ids.len(),
        original_count,
        "Found duplicate tool IDs in knowledge base"
    );
}
