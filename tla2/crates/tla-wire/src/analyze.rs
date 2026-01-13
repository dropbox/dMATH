//! Wiring analysis engine.
//!
//! The analyzer checks a WiringGraph for common wiring issues:
//! - Entry points that don't reach effects
//! - Dead/unreachable code
//! - Framework-specific wiring problems

use crate::graph::{Node, WiringGraph};
use crate::patterns::PatternRegistry;
use crate::report::{Severity, UnreachableCode, WiringIssue, WiringReport};

/// Configuration for the wiring analyzer.
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// Check for unreachable code.
    pub check_unreachable: bool,
    /// Check that entry points reach effects.
    pub check_entry_to_effect: bool,
    /// Check framework-specific patterns.
    pub check_frameworks: bool,
    /// Minimum wiring score to pass (0-100).
    pub min_score: f64,
    /// Treat warnings as errors.
    pub warnings_as_errors: bool,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            check_unreachable: true,
            check_entry_to_effect: true,
            check_frameworks: true,
            min_score: 0.0,
            warnings_as_errors: false,
        }
    }
}

/// The wiring analyzer.
pub struct Analyzer {
    config: AnalyzerConfig,
    patterns: PatternRegistry,
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer {
    pub fn new() -> Self {
        Self {
            config: AnalyzerConfig::default(),
            patterns: PatternRegistry::new(),
        }
    }

    pub fn with_config(config: AnalyzerConfig) -> Self {
        Self {
            config,
            patterns: PatternRegistry::new(),
        }
    }

    /// Analyze a wiring graph and produce a report.
    pub fn analyze(&self, graph: &WiringGraph, sources: &[String]) -> WiringReport {
        let mut report = WiringReport::new(graph);

        // Universal checks
        if self.config.check_entry_to_effect {
            report.issues.extend(self.check_entry_to_effect(graph));
        }

        if self.config.check_unreachable {
            let (issues, unreachable) = self.check_unreachable(graph);
            report.issues.extend(issues);
            report.unreachable = unreachable;
        }

        // Framework-specific checks
        if self.config.check_frameworks {
            let detected = self.patterns.detect_patterns(graph, sources);
            report.frameworks = detected.iter().map(|p| p.name().to_string()).collect();
            report.issues.extend(self.patterns.check_all(graph, sources));
        }

        // Calculate score and summary
        report.calculate_score(graph);
        report.generate_summary();

        // Check minimum score
        if report.wiring_score < self.config.min_score {
            report.issues.push(WiringIssue::new(
                "SCORE001",
                format!(
                    "Wiring score {:.0}% is below minimum threshold {:.0}%",
                    report.wiring_score, self.config.min_score
                ),
                Severity::Error,
            ));
        }

        // Upgrade warnings if configured
        if self.config.warnings_as_errors {
            for issue in &mut report.issues {
                if issue.severity == Severity::Warning {
                    issue.severity = Severity::Error;
                }
            }
        }

        report
    }

    /// Check that entry points eventually produce observable effects.
    fn check_entry_to_effect(&self, graph: &WiringGraph) -> Vec<WiringIssue> {
        let mut issues = Vec::new();

        let entries = graph.entry_points();
        let effects = graph.effects();

        if entries.is_empty() {
            issues.push(
                WiringIssue::new(
                    "ENTRY001",
                    "No entry points found",
                    Severity::Critical,
                )
                .with_detail(
                    "The application has no entry points (main, tests, handlers). \
                     Nothing will execute.",
                )
                .with_suggestion("Add a main() function or other entry point"),
            );
            return issues;
        }

        for entry in &entries {
            let entry_reaches_effect = effects.iter().any(|e| graph.path_exists(*entry, *e));

            if !entry_reaches_effect {
                let entry_node = graph.get_node(*entry);
                let entry_name = entry_node.map(|n| n.name()).unwrap_or("unknown");

                issues.push(
                    WiringIssue::new(
                        "EFFECT001",
                        format!("Entry point '{}' never produces observable output", entry_name),
                        Severity::Critical,
                    )
                    .with_detail(
                        "This entry point doesn't reach any effects (stdout, network, file I/O). \
                         The application will appear to do nothing.",
                    )
                    .with_location(entry_node.and_then(|n| n.location()).cloned().unwrap_or_else(|| {
                        crate::graph::Location::new("unknown".into(), 0, 0)
                    }))
                    .with_suggestion("Ensure the entry point calls code that produces output"),
                );
            }
        }

        issues
    }

    /// Check for unreachable code.
    fn check_unreachable(&self, graph: &WiringGraph) -> (Vec<WiringIssue>, Vec<UnreachableCode>) {
        let mut issues = Vec::new();
        let mut unreachable = Vec::new();

        let reachable = graph.reachable_from_entries();
        let functions = graph.functions();

        let unreachable_count = functions.iter().filter(|id| !reachable.contains(id)).count();

        if unreachable_count > 0 {
            // Collect unreachable items
            for id in &functions {
                if !reachable.contains(id) {
                    if let Some(node) = graph.get_node(*id) {
                        let (kind, name) = match node {
                            Node::Function { name, .. } => ("function", name.clone()),
                            Node::Handler { name, .. } => ("handler", name.clone()),
                            _ => ("item", node.name().to_string()),
                        };

                        unreachable.push(UnreachableCode {
                            node_id: *id,
                            name: name.clone(),
                            kind: kind.to_string(),
                            location: node.location().cloned(),
                            lines: node.location().and_then(|l| {
                                l.end_line.map(|end| end.saturating_sub(l.line) + 1)
                            }),
                        });
                    }
                }
            }

            // Calculate total unreachable lines
            let unreachable_lines: u32 = unreachable.iter().filter_map(|u| u.lines).sum();

            issues.push(
                WiringIssue::new(
                    "DEAD001",
                    format!(
                        "{} functions ({} lines) are unreachable from any entry point",
                        unreachable_count, unreachable_lines
                    ),
                    Severity::Warning,
                )
                .with_detail(
                    "These functions are defined but never called from main() or other entry points. \
                     They may be dead code or missing connections.",
                ),
            );
        }

        (issues, unreachable)
    }
}

/// Quick analysis helper.
pub fn analyze(graph: &WiringGraph, sources: &[String]) -> WiringReport {
    Analyzer::new().analyze(graph, sources)
}

/// Analyze with custom configuration.
pub fn analyze_with_config(
    graph: &WiringGraph,
    sources: &[String],
    config: AnalyzerConfig,
) -> WiringReport {
    Analyzer::with_config(config).analyze(graph, sources)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EffectKind, EntryKind, Location, Edge};

    #[test]
    fn test_no_entry_points() {
        let graph = WiringGraph::new();
        let report = analyze(&graph, &[]);

        assert!(report.has_critical_errors());
        assert!(report.issues.iter().any(|i| i.code == "ENTRY001"));
    }

    #[test]
    fn test_entry_without_effect() {
        let mut graph = WiringGraph::new();

        graph.add_node(Node::EntryPoint {
            kind: EntryKind::Main,
            loc: Location::new("src/main.rs".into(), 1, 1),
        });

        let report = analyze(&graph, &[]);

        assert!(report.has_critical_errors());
        assert!(report.issues.iter().any(|i| i.code == "EFFECT001"));
    }

    #[test]
    fn test_proper_wiring() {
        let mut graph = WiringGraph::new();

        let main = graph.add_node(Node::EntryPoint {
            kind: EntryKind::Main,
            loc: Location::new("src/main.rs".into(), 1, 1),
        });

        let effect = graph.add_node(Node::Effect {
            kind: EffectKind::Stdout,
            loc: Location::new("src/main.rs".into(), 2, 5),
        });

        graph.add_edge(Edge::Calls {
            from: main,
            to: effect,
            loc: Location::new("src/main.rs".into(), 2, 5),
        });

        let report = analyze(&graph, &[]);

        assert!(!report.has_critical_errors());
        assert!(!report.issues.iter().any(|i| i.code == "EFFECT001"));
    }

    #[test]
    fn test_unreachable_code() {
        let mut graph = WiringGraph::new();

        let main = graph.add_node(Node::EntryPoint {
            kind: EntryKind::Main,
            loc: Location::new("src/main.rs".into(), 1, 1),
        });

        let effect = graph.add_node(Node::Effect {
            kind: EffectKind::Stdout,
            loc: Location::new("src/main.rs".into(), 2, 5),
        });

        // Unreachable function
        graph.add_node(Node::Function {
            name: "orphan".to_string(),
            loc: Location::new("src/main.rs".into(), 10, 1),
            is_async: false,
            is_public: false,
            params: vec![],
        });

        graph.add_edge(Edge::Calls {
            from: main,
            to: effect,
            loc: Location::new("src/main.rs".into(), 2, 5),
        });

        let report = analyze(&graph, &[]);

        assert!(report.issues.iter().any(|i| i.code == "DEAD001"));
        assert_eq!(report.unreachable.len(), 1);
        assert_eq!(report.unreachable[0].name, "orphan");
    }
}
