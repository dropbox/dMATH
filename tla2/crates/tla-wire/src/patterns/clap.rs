//! Clap CLI framework pattern detection.
//!
//! Detects common wiring issues with clap-based CLI applications:
//! - Commands defined but not registered
//! - Parser created but never called
//! - Args struct defined but not used

use super::{detection, FrameworkPattern};
use crate::graph::{Node, WiringGraph};
use crate::report::{Severity, WiringIssue};

/// Pattern checker for the clap CLI framework.
pub struct ClapPattern;

impl ClapPattern {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ClapPattern {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameworkPattern for ClapPattern {
    fn name(&self) -> &'static str {
        "clap"
    }

    fn detect(&self, _graph: &WiringGraph, sources: &[String]) -> bool {
        // Check for clap usage
        detection::has_import(sources, "clap")
            || detection::has_derive(sources, "Parser")
            || detection::has_derive(sources, "Args")
            || detection::has_derive(sources, "Subcommand")
            || detection::contains_pattern(sources, "Command::new")
    }

    fn check_wiring(&self, graph: &WiringGraph, sources: &[String]) -> Vec<WiringIssue> {
        let mut issues = Vec::new();

        // Check 1: Parser struct exists but parse() is never called
        if detection::has_derive(sources, "Parser") {
            let has_parse_call = detection::contains_any(
                sources,
                &[
                    ".parse()",
                    "::parse()",
                    ".try_parse()",
                    "::try_parse()",
                    "parse_from",
                ],
            );

            if !has_parse_call {
                issues.push(WiringIssue {
                    code: "CLAP001".to_string(),
                    message: "Parser struct defined but parse() never called".to_string(),
                    detail: Some(
                        "You have a #[derive(Parser)] struct but never call .parse() on it. \
                         Arguments will not be read from the command line."
                            .to_string(),
                    ),
                    severity: Severity::Error,
                    location: None,
                    suggestion: Some("Add `let args = Args::parse();` in main()".to_string()),
                });
            }
        }

        // Check 2: Subcommands defined but not matched
        if detection::has_derive(sources, "Subcommand") {
            // Look for match on the subcommand
            let has_subcommand_match = detection::contains_any(
                sources,
                &[
                    "match args.command",
                    "match &args.command",
                    "match cli.command",
                    "match &cli.command",
                    ".command {",
                ],
            );

            if !has_subcommand_match {
                issues.push(WiringIssue {
                    code: "CLAP002".to_string(),
                    message: "Subcommands defined but never matched".to_string(),
                    detail: Some(
                        "You have a #[derive(Subcommand)] enum but never match on the command. \
                         Subcommands will be parsed but never executed."
                            .to_string(),
                    ),
                    severity: Severity::Error,
                    location: None,
                    suggestion: Some(
                        "Add `match args.command { ... }` to dispatch to subcommand handlers"
                            .to_string(),
                    ),
                });
            }
        }

        // Check 3: Command handlers defined but not reachable
        for (id, node) in graph.nodes() {
            if let Node::Function { name, .. } = node {
                // Heuristic: functions named "run_*" or "*_command" are likely handlers
                if (name.starts_with("run_") || name.ends_with("_command"))
                    && !graph.reachable_from_entries().contains(&id)
                {
                    issues.push(WiringIssue {
                        code: "CLAP003".to_string(),
                        message: format!("Command handler '{}' is never called", name),
                        detail: Some(
                            "This function looks like a command handler but is not reachable \
                             from main(). It may not be registered with clap."
                                .to_string(),
                        ),
                        severity: Severity::Warning,
                        location: node.location().cloned(),
                        suggestion: Some(format!(
                            "Ensure '{}' is called from your command match block",
                            name
                        )),
                    });
                }
            }
        }

        issues
    }

    fn wiring_description(&self) -> &'static str {
        "Clap CLI apps need: (1) #[derive(Parser)] struct, (2) Args::parse() call in main, \
         (3) match on subcommands if using #[derive(Subcommand)]"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_clap() {
        let pattern = ClapPattern::new();
        let graph = WiringGraph::new();

        // Should detect clap
        let sources = vec!["use clap::Parser;".to_string()];
        assert!(pattern.detect(&graph, &sources));

        // Should detect derive(Parser)
        let sources = vec!["#[derive(Parser)]".to_string()];
        assert!(pattern.detect(&graph, &sources));

        // Should not detect without clap
        let sources = vec!["use std::io;".to_string()];
        assert!(!pattern.detect(&graph, &sources));
    }

    #[test]
    fn test_missing_parse_call() {
        let pattern = ClapPattern::new();
        let graph = WiringGraph::new();

        let sources = vec![
            r#"
use clap::Parser;

#[derive(Parser)]
struct Args {
    name: String,
}

fn main() {
    // Forgot to call Args::parse()!
    println!("Hello");
}
"#
            .to_string(),
        ];

        let issues = pattern.check_wiring(&graph, &sources);
        assert!(issues.iter().any(|i| i.code == "CLAP001"));
    }

    #[test]
    fn test_proper_clap_usage() {
        let pattern = ClapPattern::new();
        let graph = WiringGraph::new();

        let sources = vec![
            r#"
use clap::Parser;

#[derive(Parser)]
struct Args {
    name: String,
}

fn main() {
    let args = Args::parse();
    println!("Hello {}", args.name);
}
"#
            .to_string(),
        ];

        let issues = pattern.check_wiring(&graph, &sources);
        assert!(!issues.iter().any(|i| i.code == "CLAP001"));
    }
}
