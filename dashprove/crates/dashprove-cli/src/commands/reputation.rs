//! Reputation command implementations
//!
//! CLI commands for viewing and managing backend reputation data.

use crate::commands::common::resolve_data_dir;
use dashprove::learning::{DomainSummary, ReputationTracker};
use std::path::PathBuf;

/// Configuration for reputation commands
pub struct ReputationCmdConfig<'a> {
    /// Directory containing learning data
    pub data_dir: Option<&'a str>,
    /// Output format (text, json)
    pub format: &'a str,
    /// Whether to show domain-specific stats
    pub show_domains: bool,
    /// Filter to specific backend (by name)
    pub backend: Option<&'a str>,
}

/// Get the path to the reputation tracker file
fn reputation_path(data_dir: Option<&str>) -> PathBuf {
    resolve_data_dir(data_dir).join("reputation.json")
}

/// Run reputation stats command
pub fn run_reputation_stats(config: ReputationCmdConfig) -> Result<(), Box<dyn std::error::Error>> {
    let path = reputation_path(config.data_dir);

    if !path.exists() {
        println!("No reputation data found at: {}", path.display());
        println!("\nReputation tracking is enabled when you use:");
        println!("  dashprove verify <spec> --learn");
        println!("\nOr when using the learning dispatcher configuration.");
        return Ok(());
    }

    let tracker = ReputationTracker::load_from_file(&path)
        .map_err(|e| format!("Failed to load reputation data: {}", e))?;

    if config.format == "json" {
        output_json(&tracker, config.show_domains)?;
    } else {
        output_text(&tracker, config.show_domains, config.backend)?;
    }

    Ok(())
}

/// Output reputation stats as JSON
fn output_json(
    tracker: &ReputationTracker,
    show_domains: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    #[derive(serde::Serialize)]
    struct ReputationOutput {
        backends: Vec<BackendSummary>,
        total_observations: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        domains: Option<Vec<DomainSummary>>,
    }

    #[derive(serde::Serialize)]
    struct BackendSummary {
        backend: String,
        successes: usize,
        failures: usize,
        success_rate: f64,
        reputation: f64,
        avg_response_time_ms: Option<f64>,
    }

    let backends: Vec<BackendSummary> = tracker
        .backends()
        .map(|b| {
            let stats = tracker.get_stats(b).unwrap();
            let rep = tracker.compute_reputation(b);
            BackendSummary {
                backend: format!("{:?}", b),
                successes: stats.successes,
                failures: stats.failures,
                success_rate: stats.simple_success_rate(),
                reputation: rep,
                avg_response_time_ms: stats.avg_response_time_ms(),
            }
        })
        .collect();

    let domains = if show_domains {
        Some(tracker.domain_summary())
    } else {
        None
    };

    let output = ReputationOutput {
        backends,
        total_observations: tracker.total_observations(),
        domains,
    };

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

/// Output reputation stats as text
fn output_text(
    tracker: &ReputationTracker,
    show_domains: bool,
    backend_filter: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Backend Reputation Statistics ===\n");
    println!("Total observations: {}", tracker.total_observations());
    println!("Backends tracked: {}", tracker.backend_count());
    println!();

    // Collect and sort backends by reputation
    let mut backends: Vec<_> = tracker.backends().collect();
    backends.sort_by(|a, b| {
        let rep_a = tracker.compute_reputation(a);
        let rep_b = tracker.compute_reputation(b);
        rep_b
            .partial_cmp(&rep_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "{:<20} {:>8} {:>8} {:>10} {:>12} {:>12}",
        "Backend", "Success", "Fail", "Rate", "Reputation", "Avg Time"
    );
    println!("{}", "-".repeat(76));

    for backend in &backends {
        let backend_name = format!("{:?}", backend);

        // Apply filter if specified
        if let Some(filter) = backend_filter {
            if !backend_name.to_lowercase().contains(&filter.to_lowercase()) {
                continue;
            }
        }

        let stats = tracker.get_stats(backend).unwrap();
        let rep = tracker.compute_reputation(backend);
        let avg_time = stats
            .avg_response_time_ms()
            .map(|t| format!("{:.0}ms", t))
            .unwrap_or_else(|| "-".to_string());

        println!(
            "{:<20} {:>8} {:>8} {:>9.1}% {:>11.3} {:>12}",
            backend_name,
            stats.successes,
            stats.failures,
            stats.simple_success_rate() * 100.0,
            rep,
            avg_time
        );
    }

    if show_domains && tracker.domain_count() > 0 {
        println!("\n=== Domain-Specific Statistics ===\n");
        println!(
            "Total domain observations: {}",
            tracker.total_domain_observations()
        );
        println!("Unique domains: {}", tracker.domain_count());
        println!();

        let mut summaries = tracker.domain_summary();
        summaries.sort_by(|a, b| {
            b.reputation
                .partial_cmp(&a.reputation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        println!(
            "{:<20} {:<15} {:>8} {:>8} {:>10} {:>12}",
            "Backend", "Property Type", "Success", "Fail", "Rate", "Reputation"
        );
        println!("{}", "-".repeat(78));

        for summary in summaries {
            let backend_name = format!("{:?}", summary.backend);

            // Apply filter if specified
            if let Some(filter) = backend_filter {
                if !backend_name.to_lowercase().contains(&filter.to_lowercase()) {
                    continue;
                }
            }

            println!(
                "{:<20} {:<15} {:>8} {:>8} {:>9.1}% {:>11.3}",
                backend_name,
                format!("{:?}", summary.property_type),
                summary.successes,
                summary.failures,
                summary.success_rate * 100.0,
                summary.reputation
            );
        }
    }

    Ok(())
}

/// Run reputation reset command
pub fn run_reputation_reset(data_dir: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let path = reputation_path(data_dir);

    if !path.exists() {
        println!("No reputation data found at: {}", path.display());
        return Ok(());
    }

    std::fs::remove_file(&path)?;
    println!("Reputation data reset successfully.");
    println!("Deleted: {}", path.display());

    Ok(())
}

/// Run reputation export command
pub fn run_reputation_export(
    data_dir: Option<&str>,
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = reputation_path(data_dir);

    if !path.exists() {
        return Err(format!("No reputation data found at: {}", path.display()).into());
    }

    let tracker = ReputationTracker::load_from_file(&path)
        .map_err(|e| format!("Failed to load reputation data: {}", e))?;

    tracker
        .save_to_file(output)
        .map_err(|e| format!("Failed to save reputation data: {}", e))?;

    println!("Reputation data exported to: {}", output);
    Ok(())
}

/// Run reputation import command
pub fn run_reputation_import(
    data_dir: Option<&str>,
    input: &str,
    merge: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let dest_path = reputation_path(data_dir);

    let imported = ReputationTracker::load_from_file(input)
        .map_err(|e| format!("Failed to load reputation data from {}: {}", input, e))?;

    if merge && dest_path.exists() {
        let mut existing = ReputationTracker::load_from_file(&dest_path)
            .map_err(|e| format!("Failed to load existing reputation data: {}", e))?;

        let existing_obs = existing.total_observations();
        let imported_obs = imported.total_observations();

        existing.merge(&imported);

        existing
            .save_to_file(&dest_path)
            .map_err(|e| format!("Failed to save merged reputation data: {}", e))?;

        println!("Merged reputation data:");
        println!("  Existing observations: {}", existing_obs);
        println!("  Imported observations: {}", imported_obs);
        println!("  Total after merge: {}", existing.total_observations());
    } else {
        imported
            .save_to_file(&dest_path)
            .map_err(|e| format!("Failed to save reputation data: {}", e))?;

        println!(
            "Imported reputation data: {} observations",
            imported.total_observations()
        );
    }

    println!("Saved to: {}", dest_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reputation_path() {
        let path = reputation_path(None);
        assert!(path.ends_with("reputation.json"));
    }

    #[test]
    fn test_reputation_path_with_custom_dir() {
        let path = reputation_path(Some("/tmp/test"));
        assert_eq!(path, PathBuf::from("/tmp/test/reputation.json"));
    }
}
