//! Visualization and clustering command implementations

use dashprove::backends::{CounterexampleClusters, StructuredCounterexample};
use std::path::Path;

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify from_str returns Some(Mermaid) for "mermaid"
    #[kani::proof]
    fn verify_from_str_mermaid() {
        let result = VisualizationFormat::from_str("mermaid");
        kani::assert(result.is_some(), "mermaid should parse");
        kani::assert(
            result == Some(VisualizationFormat::Mermaid),
            "should map to Mermaid",
        );
    }

    /// Verify from_str returns Some(Mermaid) for "mmd"
    #[kani::proof]
    fn verify_from_str_mmd() {
        let result = VisualizationFormat::from_str("mmd");
        kani::assert(result.is_some(), "mmd should parse");
        kani::assert(
            result == Some(VisualizationFormat::Mermaid),
            "mmd should map to Mermaid",
        );
    }

    /// Verify from_str returns Some(Dot) for "dot"
    #[kani::proof]
    fn verify_from_str_dot() {
        let result = VisualizationFormat::from_str("dot");
        kani::assert(result.is_some(), "dot should parse");
        kani::assert(
            result == Some(VisualizationFormat::Dot),
            "dot should map to Dot",
        );
    }

    /// Verify from_str returns Some(Dot) for "graphviz"
    #[kani::proof]
    fn verify_from_str_graphviz() {
        let result = VisualizationFormat::from_str("graphviz");
        kani::assert(result.is_some(), "graphviz should parse");
        kani::assert(
            result == Some(VisualizationFormat::Dot),
            "graphviz should map to Dot",
        );
    }

    /// Verify from_str returns Some(Html) for "html"
    #[kani::proof]
    fn verify_from_str_html() {
        let result = VisualizationFormat::from_str("html");
        kani::assert(result.is_some(), "html should parse");
        kani::assert(
            result == Some(VisualizationFormat::Html),
            "html should map to Html",
        );
    }

    /// Verify from_str returns Some(Html) for "htm"
    #[kani::proof]
    fn verify_from_str_htm() {
        let result = VisualizationFormat::from_str("htm");
        kani::assert(result.is_some(), "htm should parse");
        kani::assert(
            result == Some(VisualizationFormat::Html),
            "htm should map to Html",
        );
    }

    /// Verify from_str returns None for unknown format
    #[kani::proof]
    fn verify_from_str_unknown() {
        let result = VisualizationFormat::from_str("unknown_format");
        kani::assert(result.is_none(), "unknown format should return None");
    }

    /// Verify from_str is case-insensitive for "MERMAID"
    #[kani::proof]
    fn verify_from_str_case_insensitive_mermaid() {
        let result = VisualizationFormat::from_str("MERMAID");
        kani::assert(result.is_some(), "MERMAID should parse");
        kani::assert(
            result == Some(VisualizationFormat::Mermaid),
            "MERMAID should map to Mermaid",
        );
    }

    /// Verify from_str is case-insensitive for "DOT"
    #[kani::proof]
    fn verify_from_str_case_insensitive_dot() {
        let result = VisualizationFormat::from_str("DOT");
        kani::assert(result.is_some(), "DOT should parse");
        kani::assert(
            result == Some(VisualizationFormat::Dot),
            "DOT should map to Dot",
        );
    }

    /// Verify from_str is case-insensitive for "HTML"
    #[kani::proof]
    fn verify_from_str_case_insensitive_html() {
        let result = VisualizationFormat::from_str("HTML");
        kani::assert(result.is_some(), "HTML should parse");
        kani::assert(
            result == Some(VisualizationFormat::Html),
            "HTML should map to Html",
        );
    }

    /// Verify from_str handles mixed case "MerMaid"
    #[kani::proof]
    fn verify_from_str_mixed_case() {
        let result = VisualizationFormat::from_str("MerMaid");
        kani::assert(result.is_some(), "MerMaid should parse");
        kani::assert(
            result == Some(VisualizationFormat::Mermaid),
            "MerMaid should map to Mermaid",
        );
    }
}

/// Visualization output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationFormat {
    Mermaid,
    Dot,
    Html,
}

impl VisualizationFormat {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "mermaid" | "mmd" => Some(Self::Mermaid),
            "dot" | "graphviz" => Some(Self::Dot),
            "html" | "htm" => Some(Self::Html),
            _ => None,
        }
    }
}

/// Run visualize command - export counterexample to various formats
pub fn run_visualize(
    path: &str,
    format: &str,
    output: Option<&str>,
    title: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ce_path = Path::new(path);
    if !ce_path.exists() {
        return Err(format!("Counterexample file not found: {}", path).into());
    }

    // Parse format
    let vis_format = VisualizationFormat::from_str(format)
        .ok_or_else(|| format!("Unknown format: {}. Use: mermaid, dot, html", format))?;

    // Read and parse the counterexample JSON
    let content = std::fs::read_to_string(ce_path)?;
    let counterexample: StructuredCounterexample = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse counterexample JSON: {}", e))?;

    // Generate output based on format
    let result = match vis_format {
        VisualizationFormat::Mermaid => counterexample.to_mermaid(),
        VisualizationFormat::Dot => counterexample.to_dot(),
        VisualizationFormat::Html => counterexample.to_html(title),
    };

    // Output
    if let Some(output_path) = output {
        std::fs::write(output_path, &result)?;
        let format_name = match vis_format {
            VisualizationFormat::Mermaid => "Mermaid",
            VisualizationFormat::Dot => "DOT",
            VisualizationFormat::Html => "HTML",
        };
        println!("{} visualization written to {}", format_name, output_path);
    } else {
        println!("{}", result);
    }

    Ok(())
}

/// Run cluster command - cluster multiple counterexamples
pub fn run_cluster(
    paths: &[String],
    threshold: f64,
    format: &str,
    output: Option<&str>,
    title: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Validate threshold
    if !(0.0..=1.0).contains(&threshold) {
        return Err(format!("Threshold must be between 0.0 and 1.0, got {}", threshold).into());
    }

    // Load all counterexamples
    let mut counterexamples = Vec::new();
    for path in paths {
        let ce_path = Path::new(path);
        if !ce_path.exists() {
            return Err(format!("Counterexample file not found: {}", path).into());
        }
        let content = std::fs::read_to_string(ce_path)?;
        let counterexample: StructuredCounterexample = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse counterexample JSON from {}: {}", path, e))?;
        counterexamples.push(counterexample);
    }

    // Create clusters
    let clusters = CounterexampleClusters::from_counterexamples(counterexamples, threshold);

    // Generate output based on format
    let result = match format.to_lowercase().as_str() {
        "mermaid" => clusters.to_mermaid(),
        "flowchart" => clusters.to_mermaid_flowchart(),
        "html" => clusters.to_html(title),
        "json" => {
            // JSON output with cluster summary
            let json_clusters: Vec<serde_json::Value> = clusters
                .clusters
                .iter()
                .enumerate()
                .map(|(i, cluster)| {
                    serde_json::json!({
                        "cluster_id": i + 1,
                        "label": cluster.label,
                        "size": cluster.size(),
                        "representative_trace_length": cluster.representative.trace.len(),
                        "failed_checks": cluster.representative.failed_checks.iter()
                            .map(|c| &c.description)
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            serde_json::to_string_pretty(&serde_json::json!({
                "total_counterexamples": clusters.total_counterexamples(),
                "num_clusters": clusters.num_clusters(),
                "similarity_threshold": clusters.similarity_threshold,
                "clusters": json_clusters
            }))?
        }
        _ => clusters.summary(),
    };

    // Output
    if let Some(output_path) = output {
        std::fs::write(output_path, &result)?;
        println!(
            "Clustering result ({} format) written to {}",
            format, output_path
        );
    } else {
        println!("{}", result);
    }

    Ok(())
}
