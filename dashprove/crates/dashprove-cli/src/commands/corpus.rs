//! Corpus command implementations

use crate::commands::common::resolve_data_dir;
use dashprove::{
    backends::{BackendId, CounterexampleClusters, StructuredCounterexample},
    learning::{
        HistoryComparison, ProofLearningSystem, PropertyFeatures, TacticContext, TimePeriod,
    },
    usl::{parse, typecheck},
};

/// Run corpus stats command
pub fn run_corpus_stats(data_dir: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let dir = resolve_data_dir(data_dir);
    println!("Loading learning data from: {}", dir.display());

    let system = ProofLearningSystem::load_from_dir(&dir)
        .map_err(|e| format!("Failed to load learning data: {}", e))?;

    println!("\n=== Corpus Statistics ===");
    println!("Proofs in corpus: {}", system.corpus.len());

    // Count proofs by backend
    let lean_count = system.corpus.by_backend(BackendId::Lean4).len();
    let tla_count = system.corpus.by_backend(BackendId::TlaPlus).len();
    let kani_count = system.corpus.by_backend(BackendId::Kani).len();
    let alloy_count = system.corpus.by_backend(BackendId::Alloy).len();

    println!("\nBy backend:");
    println!("  LEAN 4:  {}", lean_count);
    println!("  TLA+:    {}", tla_count);
    println!("  Kani:    {}", kani_count);
    println!("  Alloy:   {}", alloy_count);

    println!("\n=== Counterexample Statistics ===");
    println!(
        "Counterexamples in corpus: {}",
        system.counterexample_count()
    );
    println!("Cluster patterns: {}", system.cluster_pattern_count());

    // Count counterexamples by backend
    let cx_lean_count = system.counterexamples.by_backend(BackendId::Lean4).len();
    let cx_tla_count = system.counterexamples.by_backend(BackendId::TlaPlus).len();
    let cx_kani_count = system.counterexamples.by_backend(BackendId::Kani).len();
    let cx_alloy_count = system.counterexamples.by_backend(BackendId::Alloy).len();

    if system.counterexample_count() > 0 {
        println!("\nBy backend:");
        println!("  LEAN 4:  {}", cx_lean_count);
        println!("  TLA+:    {}", cx_tla_count);
        println!("  Kani:    {}", cx_kani_count);
        println!("  Alloy:   {}", cx_alloy_count);
    }

    println!("\n=== Tactic Statistics ===");
    println!(
        "Total observations: {}",
        system.tactics.total_observations()
    );
    println!("Unique tactics: {}", system.tactics.unique_tactics());

    // Show top tactics if there are any
    if system.tactics.unique_tactics() > 0 {
        // Get global stats - use a generic context to query
        let dummy_features = PropertyFeatures {
            property_type: "invariant".to_string(),
            depth: 1,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            function_calls: 0,
            variable_count: 0,
            has_temporal: false,
            type_refs: vec![],
            keywords: vec![],
        };
        let dummy_context = TacticContext::from_features(&dummy_features);
        let top_tactics = system.tactics.best_for_context(&dummy_context, 10);

        if !top_tactics.is_empty() {
            println!("\nTop tactics (by Wilson score):");
            for (tactic, score) in top_tactics {
                println!("  {:<20} {:.3}", tactic, score);
            }
        }
    }

    Ok(())
}

/// Run corpus search command
pub fn run_corpus_search(
    query_path: &str,
    data_dir: Option<&str>,
    limit: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = resolve_data_dir(data_dir);

    // Read and parse the query specification
    let query_content = std::fs::read_to_string(query_path)
        .map_err(|e| format!("Failed to read query file '{}': {}", query_path, e))?;

    let spec = parse(&query_content).map_err(|e| format!("Parse error: {:?}", e))?;
    let typed_spec = typecheck(spec).map_err(|e| format!("Type error: {:?}", e))?;

    if typed_spec.spec.properties.is_empty() {
        return Err("No properties found in query file".into());
    }

    // Load the learning system
    let system = ProofLearningSystem::load_from_dir(&dir)
        .map_err(|e| format!("Failed to load learning data: {}", e))?;

    if system.corpus.is_empty() {
        println!("Corpus is empty. Run verification with --learn to populate it.");
        return Ok(());
    }

    println!(
        "Searching for similar proofs to {} properties...",
        typed_spec.spec.properties.len()
    );

    for property in &typed_spec.spec.properties {
        let prop_name = property.name();
        println!("\n=== Property: {} ===", prop_name);

        let similar = system.find_similar(property, limit);

        if similar.is_empty() {
            println!("  No similar proofs found");
        } else {
            println!("  Found {} similar proofs:", similar.len());
            for (i, proof) in similar.iter().enumerate() {
                let proof_name = proof.property.name();
                println!(
                    "  {}. {} (similarity: {:.2}%, backend: {:?})",
                    i + 1,
                    proof_name,
                    proof.similarity * 100.0,
                    proof.backend
                );
                if !proof.tactics.is_empty() {
                    println!("     Tactics: {}", proof.tactics.join(", "));
                }
            }

            // Suggest tactics based on similar proofs
            let suggestions = system.suggest_tactics(property, 5);
            if !suggestions.is_empty() {
                println!("\n  Suggested tactics:");
                for (tactic, score) in suggestions {
                    println!("    {} (score: {:.3})", tactic, score);
                }
            }
        }
    }

    Ok(())
}

/// Run text-based search command (top-level)
pub fn run_search(
    query: &str,
    data_dir: Option<&str>,
    limit: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = resolve_data_dir(data_dir);

    // Load the learning system
    let system = ProofLearningSystem::load_from_dir(&dir)
        .map_err(|e| format!("Failed to load learning data: {}", e))?;

    if system.corpus.is_empty() {
        println!("Corpus is empty. Run verification with --learn to populate it.");
        return Ok(());
    }

    println!("Searching for: \"{}\"", query);
    println!("Corpus size: {} proofs\n", system.corpus.len());

    // Use keyword-based search for text queries
    let results = system.search_by_keywords(query, limit);

    if results.is_empty() {
        println!("No matching proofs found.");
        println!("\nTip: Try different keywords or use 'corpus search <file.usl>' for structural similarity.");
    } else {
        println!("Found {} matching proofs:\n", results.len());
        for (i, proof) in results.iter().enumerate() {
            let proof_name = proof.property.name();
            println!(
                "{}. {} (backend: {:?}, score: {:.2}%)",
                i + 1,
                proof_name,
                proof.backend,
                proof.similarity * 100.0
            );
            if !proof.tactics.is_empty() {
                println!("   Tactics: {}", proof.tactics.join(", "));
            }
        }
    }

    Ok(())
}

/// Run counterexample similarity search
pub fn run_cx_search(
    path: &str,
    data_dir: Option<&str>,
    limit: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = resolve_data_dir(data_dir);

    // Load counterexample from file
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read counterexample file '{}': {}", path, e))?;

    let cx: StructuredCounterexample = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse counterexample JSON: {}", e))?;

    // Load the learning system
    let system = ProofLearningSystem::load_from_dir(&dir)
        .map_err(|e| format!("Failed to load learning data: {}", e))?;

    if system.counterexample_count() == 0 {
        println!("No counterexamples in corpus. Add counterexamples with 'corpus cx-add'.");
        return Ok(());
    }

    println!("Searching for similar counterexamples...");
    println!(
        "Corpus size: {} counterexamples\n",
        system.counterexample_count()
    );

    let similar = system.find_similar_counterexamples(&cx, limit);

    if similar.is_empty() {
        println!("No similar counterexamples found.");
    } else {
        println!("Found {} similar counterexamples:\n", similar.len());
        for (i, scx) in similar.iter().enumerate() {
            println!(
                "{}. {} (backend: {:?}, similarity: {:.2}%)",
                i + 1,
                scx.property_name,
                scx.backend,
                scx.similarity * 100.0
            );
            if let Some(ref label) = scx.cluster_label {
                println!("   Cluster: {}", label);
            }
        }
    }

    // Also try to classify against stored patterns
    if let Some((label, score)) = system.classify_counterexample(&cx) {
        println!(
            "\nClassification: {} (confidence: {:.2}%)",
            label,
            score * 100.0
        );
    }

    Ok(())
}

/// Add a counterexample to the corpus
pub fn run_cx_add(
    path: &str,
    property: &str,
    backend: &str,
    cluster: Option<&str>,
    data_dir: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = resolve_data_dir(data_dir);

    // Load counterexample from file
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read counterexample file '{}': {}", path, e))?;

    let cx: StructuredCounterexample = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse counterexample JSON: {}", e))?;

    // Parse backend
    let backend_id = match backend.to_lowercase().as_str() {
        "tlaplus" | "tla+" => BackendId::TlaPlus,
        "lean4" | "lean" => BackendId::Lean4,
        "kani" => BackendId::Kani,
        "alloy" => BackendId::Alloy,
        "coq" => BackendId::Coq,
        "dafny" => BackendId::Dafny,
        "platform_api" | "platform-api" | "platform" => BackendId::PlatformApi,
        _ => return Err(format!("Unknown backend: {}", backend).into()),
    };

    // Load or create learning system
    let mut system = ProofLearningSystem::load_from_dir(&dir)
        .map_err(|e| format!("Failed to load learning data: {}", e))?;

    let id = system.record_counterexample(property, backend_id, cx, cluster.map(|s| s.to_string()));

    // Save back
    system
        .save_to_dir(&dir)
        .map_err(|e| format!("Failed to save learning data: {}", e))?;

    println!("Added counterexample to corpus with ID: {}", id);
    println!(
        "Corpus now contains {} counterexamples",
        system.counterexample_count()
    );

    Ok(())
}

/// Classify a counterexample against stored patterns
pub fn run_cx_classify(
    path: &str,
    data_dir: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = resolve_data_dir(data_dir);

    // Load counterexample from file
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read counterexample file '{}': {}", path, e))?;

    let cx: StructuredCounterexample = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse counterexample JSON: {}", e))?;

    // Load the learning system
    let system = ProofLearningSystem::load_from_dir(&dir)
        .map_err(|e| format!("Failed to load learning data: {}", e))?;

    if system.cluster_pattern_count() == 0 {
        println!("No cluster patterns stored. Record patterns with 'corpus cx-record-clusters'.");
        return Ok(());
    }

    println!(
        "Classifying against {} cluster patterns...\n",
        system.cluster_pattern_count()
    );

    match system.classify_counterexample(&cx) {
        Some((label, score)) => {
            println!("Classification: {}", label);
            println!("Confidence: {:.2}%", score * 100.0);
        }
        None => {
            println!("No matching pattern found.");
            println!("This counterexample does not match any stored cluster patterns.");
        }
    }

    Ok(())
}

/// Record cluster patterns from clustering results
pub fn run_cx_record_clusters(
    paths: &[String],
    threshold: f64,
    data_dir: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = resolve_data_dir(data_dir);

    // Load all counterexamples
    let mut counterexamples = Vec::new();
    for path in paths {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read '{}': {}", path, e))?;
        let cx: StructuredCounterexample = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse '{}': {}", path, e))?;
        counterexamples.push(cx);
    }

    // Create clustering
    let clusters = CounterexampleClusters::from_counterexamples(counterexamples, threshold);

    println!(
        "Created {} clusters from {} counterexamples",
        clusters.clusters.len(),
        paths.len()
    );

    // Load or create learning system
    let mut system = ProofLearningSystem::load_from_dir(&dir)
        .map_err(|e| format!("Failed to load learning data: {}", e))?;

    // Record the cluster patterns
    system.record_cluster_patterns(&clusters);

    // Save back
    system
        .save_to_dir(&dir)
        .map_err(|e| format!("Failed to save learning data: {}", e))?;

    println!(
        "Recorded {} cluster patterns (total patterns: {})",
        clusters.clusters.len(),
        system.cluster_pattern_count()
    );

    for (i, cluster) in clusters.clusters.iter().enumerate() {
        println!(
            "  Cluster {}: {} ({} members)",
            i + 1,
            cluster.label,
            cluster.size()
        );
    }

    Ok(())
}

/// Parse a date string in YYYY-MM-DD format to `DateTime<Utc>`
fn parse_date(date_str: &str) -> Result<chrono::DateTime<chrono::Utc>, String> {
    use chrono::{NaiveDate, TimeZone, Utc};
    NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
        .map(|d| Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0).unwrap()))
        .map_err(|e| format!("Invalid date '{}': {}. Use YYYY-MM-DD format.", date_str, e))
}

/// Parse a date string to end-of-day `DateTime<Utc>` (for `--to` filtering)
fn parse_date_end_of_day(date_str: &str) -> Result<chrono::DateTime<chrono::Utc>, String> {
    use chrono::{NaiveDate, TimeZone, Utc};
    NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
        .map(|d| Utc.from_utc_datetime(&d.and_hms_opt(23, 59, 59).unwrap()))
        .map_err(|e| format!("Invalid date '{}': {}. Use YYYY-MM-DD format.", date_str, e))
}

/// Show corpus history over time
pub fn run_corpus_history(
    corpus: &str,
    period: &str,
    format: &str,
    output: Option<&str>,
    from: Option<&str>,
    to: Option<&str>,
    data_dir: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = resolve_data_dir(data_dir);

    // Parse period
    let period: TimePeriod = period
        .parse()
        .map_err(|e: String| -> Box<dyn std::error::Error> { e.into() })?;

    // Parse date filters
    let from_date = from.map(parse_date).transpose()?;
    let to_date = to.map(parse_date_end_of_day).transpose()?;

    // Load the learning system
    let system = ProofLearningSystem::load_from_dir(&dir)
        .map_err(|e| format!("Failed to load learning data: {}", e))?;

    let corpus_kind = corpus.to_lowercase();
    let format = format.to_lowercase();

    // Show date range info if filtering
    if from_date.is_some() || to_date.is_some() {
        let from_str = from.unwrap_or("beginning");
        let to_str = to.unwrap_or("present");
        println!("Filtering: {} to {}\n", from_str, to_str);
    }

    let (title, output_content) = match corpus_kind.as_str() {
        "proofs" | "proof" | "proof-corpus" => {
            if system.corpus.is_empty() {
                println!("Proof corpus is empty. Nothing to visualize.");
                return Ok(());
            }

            let history = system.proof_history_in_range(period, from_date, to_date);
            if history.total_count == 0 {
                println!("No proofs found in the specified date range.");
                return Ok(());
            }
            let title = "Proof Corpus History";
            let content = match format.as_str() {
                "text" | "txt" => history.summary(),
                "json" => serde_json::to_string_pretty(&history)?,
                "html" => history.to_html(title),
                _ => {
                    return Err(
                        format!("Unknown format: {}. Use text, json, or html.", format).into(),
                    )
                }
            };
            (title.to_string(), content)
        }
        "counterexamples" | "counterexample" | "cx" | "cex" => {
            if system.counterexample_count() == 0 {
                println!("No counterexamples in corpus. Nothing to visualize.");
                return Ok(());
            }

            let history = system
                .counterexamples
                .history_in_range(period, from_date, to_date);
            if history.total_count == 0 {
                println!("No counterexamples found in the specified date range.");
                return Ok(());
            }
            let title = "Counterexample Corpus History";
            let content = match format.as_str() {
                "text" | "txt" => history.summary(),
                "json" => serde_json::to_string_pretty(&history)?,
                "html" => history.to_html(title),
                _ => {
                    return Err(
                        format!("Unknown format: {}. Use text, json, or html.", format).into(),
                    )
                }
            };
            (title.to_string(), content)
        }
        _ => {
            return Err(format!(
                "Unknown corpus type: {}. Use proofs or counterexamples.",
                corpus
            )
            .into())
        }
    };

    if let Some(path) = output {
        std::fs::write(path, &output_content)?;
        println!("{} written to: {}", title, path);
    } else {
        println!("{}", output_content);
    }

    Ok(())
}

/// Compare two time periods in the corpus
#[allow(clippy::too_many_arguments)]
pub fn run_corpus_compare(
    corpus: &str,
    baseline_from: &str,
    baseline_to: &str,
    compare_from: &str,
    compare_to: &str,
    period: &str,
    format: &str,
    output: Option<&str>,
    data_dir: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use dashprove::learning::{format_suggestions, suggest_comparison_periods};

    let dir = resolve_data_dir(data_dir);

    // Parse period
    let period: TimePeriod = period
        .parse()
        .map_err(|e: String| -> Box<dyn std::error::Error> { e.into() })?;

    // Parse date ranges
    let baseline_start = parse_date(baseline_from)?;
    let baseline_end = parse_date_end_of_day(baseline_to)?;
    let compare_start = parse_date(compare_from)?;
    let compare_end = parse_date_end_of_day(compare_to)?;

    // Load the learning system
    let system = ProofLearningSystem::load_from_dir(&dir)
        .map_err(|e| format!("Failed to load learning data: {}", e))?;

    let corpus_kind = corpus.to_lowercase();
    let format = format.to_lowercase();

    let baseline_label = format!("{} to {}", baseline_from, baseline_to);
    let compare_label = format!("{} to {}", compare_from, compare_to);

    let comparison = match corpus_kind.as_str() {
        "proofs" | "proof" | "proof-corpus" => {
            if system.corpus.is_empty() {
                println!("Proof corpus is empty. Nothing to compare.");
                return Ok(());
            }

            let baseline =
                system.proof_history_in_range(period, Some(baseline_start), Some(baseline_end));
            let comparison =
                system.proof_history_in_range(period, Some(compare_start), Some(compare_end));

            HistoryComparison::from_proof_histories(
                &baseline,
                &comparison,
                &baseline_label,
                &compare_label,
            )
        }
        "counterexamples" | "counterexample" | "cx" | "cex" => {
            if system.counterexample_count() == 0 {
                println!("No counterexamples in corpus. Nothing to compare.");
                return Ok(());
            }

            let baseline = system.counterexamples.history_in_range(
                period,
                Some(baseline_start),
                Some(baseline_end),
            );
            let comparison = system.counterexamples.history_in_range(
                period,
                Some(compare_start),
                Some(compare_end),
            );

            HistoryComparison::from_corpus_histories(
                &baseline,
                &comparison,
                &baseline_label,
                &compare_label,
            )
        }
        _ => {
            return Err(format!(
                "Unknown corpus type: {}. Use proofs or counterexamples.",
                corpus
            )
            .into())
        }
    };

    // Get date range for suggestions
    let (first_recorded, last_recorded) = match corpus_kind.as_str() {
        "proofs" | "proof" | "proof-corpus" => {
            let history = system.proof_history(period);
            (history.first_recorded, history.last_recorded)
        }
        _ => {
            let history = system.counterexamples.history(period);
            (history.first_recorded, history.last_recorded)
        }
    };

    let suggestions = suggest_comparison_periods(first_recorded, last_recorded, None);
    let data_range = first_recorded.zip(last_recorded);
    let corpus_name = match corpus_kind.as_str() {
        "proofs" | "proof" | "proof-corpus" => "proofs",
        _ => "counterexamples",
    };

    let output_content = match format.as_str() {
        "text" | "txt" => {
            let mut content = comparison.summary();
            // Add suggestions footer for text output
            if !suggestions.is_empty() {
                content.push_str("\n\n");
                content.push_str(&"─".repeat(50));
                content.push_str("\n\n");
                content.push_str(&format_suggestions(&suggestions));
            }
            content
        }
        "json" => serde_json::to_string_pretty(&comparison)?,
        "html" => comparison.to_html_with_suggestions(&suggestions, corpus_name, data_range),
        _ => return Err(format!("Unknown format: {}. Use text, json, or html.", format).into()),
    };

    if let Some(path) = output {
        std::fs::write(path, &output_content)?;
        println!("Comparison written to: {}", path);
    } else {
        println!("{}", output_content);
    }

    Ok(())
}

/// Suggest comparison periods based on available data
pub fn run_corpus_suggest_compare(
    corpus: &str,
    format: &str,
    output: Option<&str>,
    data_dir: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use dashprove::learning::{format_suggestions, suggest_comparison_periods};

    let dir = resolve_data_dir(data_dir);

    // Load the learning system
    let system = ProofLearningSystem::load_from_dir(&dir)
        .map_err(|e| format!("Failed to load learning data: {}", e))?;

    let corpus_kind = corpus.to_lowercase();
    let format = format.to_lowercase();

    // Get date range from corpus
    let (first_recorded, last_recorded, corpus_name) = match corpus_kind.as_str() {
        "proofs" | "proof" | "proof-corpus" => {
            if system.corpus.is_empty() {
                println!("Proof corpus is empty. No suggestions available.");
                return Ok(());
            }
            let history = system.proof_history(TimePeriod::Day);
            (history.first_recorded, history.last_recorded, "proofs")
        }
        "counterexamples" | "counterexample" | "cx" | "cex" => {
            if system.counterexample_count() == 0 {
                println!("No counterexamples in corpus. No suggestions available.");
                return Ok(());
            }
            let history = system.counterexamples.history(TimePeriod::Day);
            (
                history.first_recorded,
                history.last_recorded,
                "counterexamples",
            )
        }
        _ => {
            return Err(format!(
                "Unknown corpus type: {}. Use proofs or counterexamples.",
                corpus
            )
            .into())
        }
    };

    // Generate suggestions
    let suggestions = suggest_comparison_periods(first_recorded, last_recorded, None);

    if suggestions.is_empty() {
        println!("No comparison suggestions available (insufficient data history).");
        return Ok(());
    }

    let output_content = match format.as_str() {
        "text" | "txt" => {
            let mut lines = vec![format!(
                "Period comparison suggestions for {} corpus:",
                corpus_name
            )];
            if let (Some(first), Some(last)) = (first_recorded, last_recorded) {
                lines.push(format!(
                    "Data range: {} to {}",
                    first.format("%Y-%m-%d"),
                    last.format("%Y-%m-%d")
                ));
            }
            lines.push(String::new());
            lines.push(format_suggestions(&suggestions));
            lines.join("\n")
        }
        "json" => serde_json::to_string_pretty(&suggestions)?,
        _ => return Err(format!("Unknown format: {}. Use text or json.", format).into()),
    };

    if let Some(path) = output {
        std::fs::write(path, &output_content)?;
        println!("Suggestions written to: {}", path);
    } else {
        println!("{}", output_content);
    }

    Ok(())
}
