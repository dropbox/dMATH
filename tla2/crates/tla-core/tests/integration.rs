//! Integration tests for parsing real TLA+ specs from the tlaplus/Examples repository

use std::path::{Path, PathBuf};
use tla_core::{lower, parse, parse_to_syntax_tree, resolve, FileId};

/// Test parsing a specific TLA+ file
fn test_parse_file(path: &Path) -> (bool, Vec<String>) {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return (false, vec![format!("Failed to read file: {}", e)]),
    };

    let result = parse(&source);
    let mut issues = Vec::new();

    if !result.errors.is_empty() {
        for err in &result.errors {
            issues.push(format!(
                "Parse error at {}-{}: {}",
                err.start, err.end, err.message
            ));
        }
    }

    (result.errors.is_empty(), issues)
}

/// Test full pipeline: parse + lower
fn test_full_pipeline(path: &Path) -> (bool, Vec<String>) {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return (false, vec![format!("Failed to read file: {}", e)]),
    };

    let result = parse(&source);
    let mut issues = Vec::new();

    if !result.errors.is_empty() {
        for err in &result.errors {
            issues.push(format!("Parse error: {}", err.message));
        }
        return (false, issues);
    }

    // Try to lower
    let tree = parse_to_syntax_tree(&source);
    let lower_result = lower(FileId(0), &tree);

    if !lower_result.errors.is_empty() {
        for err in &lower_result.errors {
            issues.push(format!("Lowering error: {:?}", err));
        }
        return (false, issues);
    }

    if lower_result.module.is_none() {
        issues.push("Lowering produced no module".to_string());
        return (false, issues);
    }

    (true, issues)
}

/// Test full pipeline: parse + lower + resolve
#[allow(dead_code)]
fn test_full_pipeline_with_resolve(path: &Path) -> (bool, usize, Vec<String>) {
    let (ok, count, issues, _) = test_full_pipeline_with_resolve_detailed(path);
    (ok, count, issues)
}

/// Test full pipeline with detailed undefined name tracking
fn test_full_pipeline_with_resolve_detailed(
    path: &Path,
) -> (bool, usize, Vec<String>, Vec<String>) {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            return (
                false,
                0,
                vec![format!("Failed to read file: {}", e)],
                vec![],
            )
        }
    };

    let result = parse(&source);
    let mut issues = Vec::new();

    if !result.errors.is_empty() {
        for err in &result.errors {
            issues.push(format!("Parse error: {}", err.message));
        }
        return (false, 0, issues, vec![]);
    }

    // Try to lower
    let tree = parse_to_syntax_tree(&source);
    let lower_result = lower(FileId(0), &tree);

    if !lower_result.errors.is_empty() {
        for err in &lower_result.errors {
            issues.push(format!("Lowering error: {:?}", err));
        }
        return (false, 0, issues, vec![]);
    }

    let module = match lower_result.module {
        Some(m) => m,
        None => {
            issues.push("Lowering produced no module".to_string());
            return (false, 0, issues, vec![]);
        }
    };

    // Try to resolve
    let resolve_result = resolve(&module);

    // Collect undefined names
    let undefined_names: Vec<String> = resolve_result
        .errors
        .iter()
        .filter_map(|e| {
            if let tla_core::resolve::ResolveErrorKind::Undefined { name } = &e.kind {
                Some(name.clone())
            } else {
                None
            }
        })
        .collect();

    let undefined_count = undefined_names.len();

    // We consider resolution "successful" even with undefined errors,
    // since many specs extend standard modules (Naturals, Sequences, etc.)
    // that we don't have access to yet
    (true, undefined_count, issues, undefined_names)
}

/// Recursively find all .tla files in a directory
fn find_tla_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_tla_files(&path));
            } else if path.extension().is_some_and(|ext| ext == "tla") {
                files.push(path);
            }
        }
    }
    files
}

#[test]
fn test_parse_majority_spec() {
    let path =
        Path::new(env!("HOME")).join("tlaplus-examples/specifications/Majority/Majority.tla");
    if !path.exists() {
        eprintln!("Skipping test - Examples repo not found");
        return;
    }

    let (ok, issues) = test_parse_file(&path);
    if !ok {
        for issue in &issues {
            eprintln!("  {}", issue);
        }
    }
    // Don't assert yet - we're gathering info
    eprintln!(
        "Majority.tla parse result: ok={}, issues={}",
        ok,
        issues.len()
    );
}

#[test]
fn test_parse_echo_spec() {
    let path = Path::new(env!("HOME")).join("tlaplus-examples/specifications/echo/Echo.tla");
    if !path.exists() {
        eprintln!("Skipping test - Examples repo not found");
        return;
    }

    let (ok, issues) = test_parse_file(&path);
    if !ok {
        for issue in &issues {
            eprintln!("  {}", issue);
        }
    }
    eprintln!("Echo.tla parse result: ok={}, issues={}", ok, issues.len());
}

#[test]
fn test_bulk_parse_examples() {
    let examples_dir = Path::new(env!("HOME")).join("tlaplus-examples/specifications");
    if !examples_dir.exists() {
        eprintln!(
            "Skipping test - Examples repo not found at {:?}",
            examples_dir
        );
        return;
    }

    let tla_files = find_tla_files(&examples_dir);
    let total = tla_files.len();
    let mut success = 0;
    let mut failures: Vec<(String, Vec<String>)> = Vec::new();

    for path in &tla_files {
        let (ok, issues) = test_parse_file(path);
        if ok {
            success += 1;
        } else {
            let rel_path = path
                .strip_prefix(&examples_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            failures.push((rel_path, issues));
        }
    }

    eprintln!("\n=== Parse Results ===");
    eprintln!(
        "Total: {}, Success: {}, Failed: {}",
        total,
        success,
        total - success
    );
    eprintln!(
        "Success rate: {:.1}%",
        (success as f64 / total as f64) * 100.0
    );

    if !failures.is_empty() {
        eprintln!("\n=== Failed Files (first 20) ===");
        for (path, issues) in failures.iter().take(20) {
            eprintln!("\n{}", path);
            for issue in issues.iter().take(3) {
                eprintln!("  {}", issue);
            }
        }
    }
}

#[test]
fn test_bulk_lower_examples() {
    let examples_dir = Path::new(env!("HOME")).join("tlaplus-examples/specifications");
    if !examples_dir.exists() {
        eprintln!(
            "Skipping test - Examples repo not found at {:?}",
            examples_dir
        );
        return;
    }

    let tla_files = find_tla_files(&examples_dir);
    let total = tla_files.len();
    let mut success = 0;
    let mut failures: Vec<(String, Vec<String>)> = Vec::new();

    for path in &tla_files {
        let (ok, issues) = test_full_pipeline(path);
        if ok {
            success += 1;
        } else {
            let rel_path = path
                .strip_prefix(&examples_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            failures.push((rel_path, issues));
        }
    }

    eprintln!("\n=== Full Pipeline (Parse + Lower) Results ===");
    eprintln!(
        "Total: {}, Success: {}, Failed: {}",
        total,
        success,
        total - success
    );
    eprintln!(
        "Success rate: {:.1}%",
        (success as f64 / total as f64) * 100.0
    );

    if !failures.is_empty() {
        eprintln!("\n=== Failed Files (first 20) ===");
        for (path, issues) in failures.iter().take(20) {
            eprintln!("\n{}", path);
            for issue in issues.iter().take(3) {
                eprintln!("  {}", issue);
            }
        }
    }
}

#[test]
fn test_bulk_resolve_examples() {
    use std::collections::HashMap;

    let examples_dir = Path::new(env!("HOME")).join("tlaplus-examples/specifications");
    if !examples_dir.exists() {
        eprintln!(
            "Skipping test - Examples repo not found at {:?}",
            examples_dir
        );
        return;
    }

    let tla_files = find_tla_files(&examples_dir);
    let total = tla_files.len();
    let mut success = 0;
    let mut total_undefined = 0;
    let mut failures: Vec<(String, Vec<String>)> = Vec::new();
    let mut undefined_counts: HashMap<String, usize> = HashMap::new();

    for path in &tla_files {
        let (ok, undefined_count, issues, undefined_names) =
            test_full_pipeline_with_resolve_detailed(path);
        if ok {
            success += 1;
            total_undefined += undefined_count;
            for name in undefined_names {
                *undefined_counts.entry(name).or_insert(0) += 1;
            }
        } else {
            let rel_path = path
                .strip_prefix(&examples_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            failures.push((rel_path, issues));
        }
    }

    eprintln!("\n=== Full Pipeline (Parse + Lower + Resolve) Results ===");
    eprintln!(
        "Total: {}, Success: {}, Failed: {}",
        total,
        success,
        total - success
    );
    eprintln!(
        "Success rate: {:.1}%",
        (success as f64 / total as f64) * 100.0
    );
    eprintln!(
        "Total undefined references: {} (expected - standard library not implemented)",
        total_undefined
    );

    // Show top undefined references
    let mut sorted: Vec<_> = undefined_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    eprintln!("\n=== Top 30 Undefined References ===");
    for (name, count) in sorted.iter().take(30) {
        eprintln!("  {:>4}  {}", count, name);
    }

    if !failures.is_empty() {
        eprintln!("\n=== Failed Files (first 20) ===");
        for (path, issues) in failures.iter().take(20) {
            eprintln!("\n{}", path);
            for issue in issues.iter().take(3) {
                eprintln!("  {}", issue);
            }
        }
    }
}
