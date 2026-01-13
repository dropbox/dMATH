//! Wiring analysis reports and diagnostics.
//!
//! This module defines the types for reporting wiring issues and analysis results.

use crate::graph::{GraphStats, Location, NodeId, WiringGraph};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Severity level for wiring issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// Hints and suggestions for improvement.
    Hint,
    /// Potential issues that may indicate unwired code.
    Warning,
    /// Definite wiring issues that will cause the app to malfunction.
    Error,
    /// Critical issues that will cause the app to do nothing.
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Hint => write!(f, "hint"),
            Severity::Warning => write!(f, "warning"),
            Severity::Error => write!(f, "error"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A wiring issue detected during analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WiringIssue {
    /// Unique code for this type of issue (e.g., "CLAP001").
    pub code: String,
    /// Short message describing the issue.
    pub message: String,
    /// Detailed explanation of the issue.
    pub detail: Option<String>,
    /// Severity of the issue.
    pub severity: Severity,
    /// Source location where the issue was detected.
    pub location: Option<Location>,
    /// Suggested fix for the issue.
    pub suggestion: Option<String>,
}

impl WiringIssue {
    pub fn new(code: impl Into<String>, message: impl Into<String>, severity: Severity) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            detail: None,
            severity,
            location: None,
            suggestion: None,
        }
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    pub fn with_location(mut self, location: Location) -> Self {
        self.location = Some(location);
        self
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

/// Information about unreachable code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnreachableCode {
    /// Node ID of the unreachable code.
    pub node_id: NodeId,
    /// Name of the unreachable element.
    pub name: String,
    /// Kind of element (function, handler, etc.).
    pub kind: String,
    /// Location in source.
    pub location: Option<Location>,
    /// Number of lines of code.
    pub lines: Option<u32>,
}

/// Complete wiring analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WiringReport {
    /// All wiring issues detected.
    pub issues: Vec<WiringIssue>,
    /// Graph statistics.
    pub stats: GraphStats,
    /// Detected frameworks.
    pub frameworks: Vec<String>,
    /// Unreachable code.
    pub unreachable: Vec<UnreachableCode>,
    /// Wiring score (0-100).
    pub wiring_score: f64,
    /// Summary message.
    pub summary: String,
}

impl WiringReport {
    pub fn new(graph: &WiringGraph) -> Self {
        Self {
            issues: Vec::new(),
            stats: graph.stats(),
            frameworks: Vec::new(),
            unreachable: Vec::new(),
            wiring_score: 0.0,
            summary: String::new(),
        }
    }

    /// Count issues by severity.
    pub fn count_by_severity(&self, severity: Severity) -> usize {
        self.issues.iter().filter(|i| i.severity == severity).count()
    }

    /// Check if there are any critical errors.
    pub fn has_critical_errors(&self) -> bool {
        self.issues.iter().any(|i| i.severity == Severity::Critical)
    }

    /// Check if there are any errors (including critical).
    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|i| matches!(i.severity, Severity::Error | Severity::Critical))
    }

    /// Calculate the wiring score based on reachability.
    pub fn calculate_score(&mut self, graph: &WiringGraph) {
        let reachable = graph.reachable_from_entries();
        let total_functions = graph.functions().len();

        if total_functions == 0 {
            self.wiring_score = 100.0;
        } else {
            let reachable_functions = graph
                .functions()
                .iter()
                .filter(|id| reachable.contains(id))
                .count();
            self.wiring_score = (reachable_functions as f64 / total_functions as f64) * 100.0;
        }
    }

    /// Generate a summary message.
    pub fn generate_summary(&mut self) {
        let critical = self.count_by_severity(Severity::Critical);
        let errors = self.count_by_severity(Severity::Error);
        let warnings = self.count_by_severity(Severity::Warning);

        if critical > 0 {
            self.summary = format!(
                "CRITICAL: {} issue(s) will cause the application to do nothing. \
                 Wiring score: {:.0}%",
                critical, self.wiring_score
            );
        } else if errors > 0 {
            self.summary = format!(
                "ERROR: {} issue(s) will cause application malfunction. \
                 Wiring score: {:.0}%",
                errors, self.wiring_score
            );
        } else if warnings > 0 {
            self.summary = format!(
                "WARNING: {} potential issue(s) detected. Wiring score: {:.0}%",
                warnings, self.wiring_score
            );
        } else {
            self.summary = format!(
                "OK: No wiring issues detected. Wiring score: {:.0}%",
                self.wiring_score
            );
        }
    }

    /// Format the report for terminal output.
    pub fn format_terminal(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str("═══════════════════════════════════════════════════════════\n");
        output.push_str("                    WIRING ANALYSIS REPORT                  \n");
        output.push_str("═══════════════════════════════════════════════════════════\n\n");

        // Stats
        output.push_str(&format!(
            "Project Statistics:\n  Nodes: {} ({} functions, {} handlers, {} routes)\n  \
             Edges: {} ({} calls, {} data flows)\n  Entry points: {}\n  Effects: {}\n\n",
            self.stats.total_nodes,
            self.stats.functions,
            self.stats.handlers,
            self.stats.routes,
            self.stats.total_edges,
            self.stats.call_edges,
            self.stats.data_flow_edges,
            self.stats.entry_points,
            self.stats.effects,
        ));

        // Detected frameworks
        if !self.frameworks.is_empty() {
            output.push_str("Detected Frameworks:\n");
            for framework in &self.frameworks {
                output.push_str(&format!("  - {}\n", framework));
            }
            output.push('\n');
        }

        // Critical issues
        let critical_issues: Vec<_> = self
            .issues
            .iter()
            .filter(|i| i.severity == Severity::Critical)
            .collect();
        if !critical_issues.is_empty() {
            output.push_str("═══════════════════════════════════════════════════════════\n");
            output.push_str(" CRITICAL: Application will not function\n");
            output.push_str("═══════════════════════════════════════════════════════════\n\n");
            for issue in critical_issues {
                output.push_str(&format_issue(issue));
            }
        }

        // Errors
        let error_issues: Vec<_> = self
            .issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .collect();
        if !error_issues.is_empty() {
            output.push_str("═══════════════════════════════════════════════════════════\n");
            output.push_str(" ERRORS: Application will malfunction\n");
            output.push_str("═══════════════════════════════════════════════════════════\n\n");
            for issue in error_issues {
                output.push_str(&format_issue(issue));
            }
        }

        // Warnings
        let warning_issues: Vec<_> = self
            .issues
            .iter()
            .filter(|i| i.severity == Severity::Warning)
            .collect();
        if !warning_issues.is_empty() {
            output.push_str("═══════════════════════════════════════════════════════════\n");
            output.push_str(" WARNINGS: Potential issues detected\n");
            output.push_str("═══════════════════════════════════════════════════════════\n\n");
            for issue in warning_issues {
                output.push_str(&format_issue(issue));
            }
        }

        // Unreachable code
        if !self.unreachable.is_empty() {
            output.push_str("═══════════════════════════════════════════════════════════\n");
            output.push_str(" UNREACHABLE CODE\n");
            output.push_str("═══════════════════════════════════════════════════════════\n\n");
            for item in &self.unreachable {
                output.push_str(&format!(
                    "  {} '{}' ({}) - not reachable from any entry point\n",
                    item.kind, item.name, item.location.as_ref().map_or("unknown".to_string(), |l| format!("{}:{}", l.file.display(), l.line))
                ));
            }
            output.push('\n');
        }

        // Summary
        output.push_str("═══════════════════════════════════════════════════════════\n");
        output.push_str(" SUMMARY\n");
        output.push_str("═══════════════════════════════════════════════════════════\n\n");
        output.push_str(&format!(
            "  Critical: {}\n  Errors: {}\n  Warnings: {}\n  Hints: {}\n\n",
            self.count_by_severity(Severity::Critical),
            self.count_by_severity(Severity::Error),
            self.count_by_severity(Severity::Warning),
            self.count_by_severity(Severity::Hint),
        ));
        output.push_str(&format!("  Wiring Score: {:.0}%\n\n", self.wiring_score));
        output.push_str(&format!("  {}\n", self.summary));

        output
    }
}

fn format_issue(issue: &WiringIssue) -> String {
    let mut output = String::new();

    output.push_str(&format!("[{}] {}\n", issue.code, issue.message));

    if let Some(loc) = &issue.location {
        output.push_str(&format!("  at {}:{}\n", loc.file.display(), loc.line));
    }

    if let Some(detail) = &issue.detail {
        output.push_str(&format!("  {}\n", detail));
    }

    if let Some(suggestion) = &issue.suggestion {
        output.push_str(&format!("  Suggestion: {}\n", suggestion));
    }

    output.push('\n');
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Hint < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn test_issue_builder() {
        let issue = WiringIssue::new("TEST001", "Test issue", Severity::Warning)
            .with_detail("Detailed explanation")
            .with_suggestion("Fix it like this");

        assert_eq!(issue.code, "TEST001");
        assert_eq!(issue.severity, Severity::Warning);
        assert_eq!(issue.detail, Some("Detailed explanation".to_string()));
        assert_eq!(issue.suggestion, Some("Fix it like this".to_string()));
    }

    #[test]
    fn test_report_counts() {
        let graph = WiringGraph::new();
        let mut report = WiringReport::new(&graph);

        report.issues.push(WiringIssue::new("E1", "Error 1", Severity::Error));
        report.issues.push(WiringIssue::new("E2", "Error 2", Severity::Error));
        report.issues.push(WiringIssue::new("W1", "Warning 1", Severity::Warning));
        report.issues.push(WiringIssue::new("C1", "Critical 1", Severity::Critical));

        assert_eq!(report.count_by_severity(Severity::Error), 2);
        assert_eq!(report.count_by_severity(Severity::Warning), 1);
        assert_eq!(report.count_by_severity(Severity::Critical), 1);
        assert!(report.has_critical_errors());
        assert!(report.has_errors());
    }
}
