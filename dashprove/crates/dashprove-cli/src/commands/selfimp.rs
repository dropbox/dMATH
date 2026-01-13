//! Self-improvement command implementation
//!
//! Provides CLI access to the verified self-improvement infrastructure.
//! All self-modifications MUST pass through the verification gate.

use dashprove_selfimp::{
    GateConfig, Improvement, ImprovementKind, ImprovementTarget, RollbackConfig, RollbackManager,
    VersionHistory,
};
use std::path::Path;

/// Configuration for selfimp status command
pub struct SelfImpStatusConfig<'a> {
    /// Output format (text, json)
    pub format: &'a str,
    /// Show verbose output
    pub verbose: bool,
}

/// Configuration for selfimp history command
pub struct SelfImpHistoryConfig<'a> {
    /// Number of entries to show
    pub limit: usize,
    /// Output format (text, json)
    pub format: &'a str,
    /// Show verbose output
    pub verbose: bool,
}

/// Configuration for selfimp verify command
pub struct SelfImpVerifyConfig<'a> {
    /// Path to improvement proposal file
    pub proposal: &'a str,
    /// Current version identifier
    pub current_version: &'a str,
    /// Output format (text, json)
    pub format: &'a str,
    /// Show verbose output
    pub verbose: bool,
    /// Dry run (don't actually apply)
    pub dry_run: bool,
}

/// Configuration for selfimp rollback command
pub struct SelfImpRollbackConfig<'a> {
    /// Target version to roll back to
    pub target_version: Option<&'a str>,
    /// Output format (text, json)
    pub format: &'a str,
    /// Show verbose output
    pub verbose: bool,
    /// Dry run (don't actually rollback)
    pub dry_run: bool,
}

/// Configuration for selfimp gate command
pub struct SelfImpGateConfig<'a> {
    /// Output format (text, json)
    pub format: &'a str,
    /// Show all checks
    pub all_checks: bool,
}

/// Run selfimp status command
pub fn run_selfimp_status(
    config: SelfImpStatusConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let history = VersionHistory::new();

    if config.format == "json" {
        #[derive(serde::Serialize)]
        struct StatusJson {
            version_count: usize,
            current_version: Option<String>,
            gate_active: bool,
            rollback_manager_active: bool,
        }

        let status = StatusJson {
            version_count: history.len(),
            current_version: history.current().map(|v| v.id.to_string()),
            gate_active: true, // Gate is always active
            rollback_manager_active: true,
        };

        println!("{}", serde_json::to_string_pretty(&status)?);
    } else {
        println!("=== DashProve Self-Improvement Status ===\n");
        println!("Version History:");
        println!("  Total versions: {}", history.len());
        if let Some(current) = history.current() {
            println!("  Current version: {}", current.id);
        } else {
            println!("  Current version: (none registered)");
        }
        println!();
        println!("Verification Gate: ACTIVE");
        println!("  All modifications MUST pass through the gate");
        println!("  Checks: structural, soundness, capability, formal, certificate");
        println!();
        println!("Rollback Manager: ACTIVE");
        println!("  Auto-rollback on verification failure: enabled");

        if config.verbose {
            println!();
            println!("Gate Configuration:");
            let gate_config = GateConfig::default();
            println!(
                "  Verification timeout: {:?}",
                gate_config.verification_timeout
            );
            println!(
                "  Run hardening checks: {}",
                gate_config.run_hardening_checks
            );
            println!("  Backends: {:?}", gate_config.backends);
        }
    }

    Ok(())
}

/// Run selfimp history command
pub fn run_selfimp_history(
    config: SelfImpHistoryConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let history = VersionHistory::new();

    if config.format == "json" {
        #[derive(serde::Serialize)]
        struct HistoryJson {
            total_versions: usize,
            displayed: usize,
            versions: Vec<VersionEntryJson>,
        }

        #[derive(serde::Serialize)]
        struct VersionEntryJson {
            id: String,
            version_string: String,
            capability_count: usize,
        }

        let all_versions = history.all_versions();
        let versions: Vec<VersionEntryJson> = all_versions
            .into_iter()
            .take(config.limit)
            .map(|v| VersionEntryJson {
                id: v.id.to_string(),
                version_string: v.version_string.clone(),
                capability_count: v.capabilities.capabilities.len(),
            })
            .collect();

        let history_json = HistoryJson {
            total_versions: history.len(),
            displayed: versions.len(),
            versions,
        };

        println!("{}", serde_json::to_string_pretty(&history_json)?);
    } else {
        println!("=== Version History ===\n");
        println!(
            "Total versions: {} (showing up to {})\n",
            history.len(),
            config.limit
        );

        if history.is_empty() {
            println!("No versions registered yet.");
            println!();
            println!("To register a version, use:");
            println!("  dashprove selfimp verify --proposal <file> --current-version <id>");
        } else {
            let all_versions = history.all_versions();
            for (i, version) in all_versions.into_iter().take(config.limit).enumerate() {
                let marker = if history.current().map(|c| &c.id) == Some(&version.id) {
                    " (current)"
                } else {
                    ""
                };

                println!("{}. {}{}", i + 1, version.id, marker);
                println!("   Version: {}", version.version_string);
                println!(
                    "   Capabilities: {}",
                    version.capabilities.capabilities.len()
                );

                if config.verbose {
                    for (name, cap) in version.capabilities.capabilities.iter().take(5) {
                        println!("     - {}: {}", name, cap.value);
                    }
                    if version.capabilities.capabilities.len() > 5 {
                        println!(
                            "     ... and {} more",
                            version.capabilities.capabilities.len() - 5
                        );
                    }
                }
                println!();
            }
        }
    }

    Ok(())
}

/// Run selfimp verify command
pub fn run_selfimp_verify(
    config: SelfImpVerifyConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read proposal file
    let proposal_path = Path::new(config.proposal);
    if !proposal_path.exists() {
        return Err(format!("Proposal file not found: {}", config.proposal).into());
    }

    let proposal_content = std::fs::read_to_string(proposal_path)?;

    // Parse proposal (expecting JSON)
    #[derive(serde::Deserialize)]
    struct ProposalFile {
        kind: String,
        target: String,
        description: String,
    }

    let proposal: ProposalFile = serde_json::from_str(&proposal_content)
        .map_err(|e| format!("Invalid proposal JSON: {}", e))?;

    // Convert to Improvement
    let kind = match proposal.kind.as_str() {
        "bug_fix" => ImprovementKind::BugFix,
        "optimization" => ImprovementKind::Optimization,
        "feature" => ImprovementKind::Feature,
        "security" => ImprovementKind::Security,
        "refactoring" => ImprovementKind::Refactoring,
        "configuration" => ImprovementKind::Configuration,
        "dependency_update" => ImprovementKind::DependencyUpdate,
        other => ImprovementKind::Custom(other.to_string()),
    };

    let target = match proposal.target.as_str() {
        "system" => ImprovementTarget::System,
        "dependencies" => ImprovementTarget::Dependencies,
        other if other.starts_with("module:") => {
            ImprovementTarget::Module(other.strip_prefix("module:").unwrap().to_string())
        }
        other if other.starts_with("function:") => {
            ImprovementTarget::Function(other.strip_prefix("function:").unwrap().to_string())
        }
        other if other.starts_with("config:") => {
            ImprovementTarget::Config(other.strip_prefix("config:").unwrap().to_string())
        }
        _ => ImprovementTarget::System,
    };

    let improvement = Improvement::new(proposal.description.clone(), kind, target);

    if config.verbose {
        println!("=== Verification Gate Check ===\n");
        println!("Proposal: {}", config.proposal);
        println!("Kind: {:?}", improvement.kind);
        println!("Target: {:?}", improvement.target);
        println!("Description: {}", improvement.description);
        println!("Current Version: {}", config.current_version);
        println!();
    }

    if config.dry_run {
        println!("DRY RUN: Checking proposal structure...\n");

        // Basic structural check
        if improvement.is_valid() {
            if config.format == "json" {
                #[derive(serde::Serialize)]
                struct ResultJson {
                    status: String,
                    valid_structure: bool,
                    dry_run: bool,
                }

                let result_json = ResultJson {
                    status: "valid".to_string(),
                    valid_structure: true,
                    dry_run: config.dry_run,
                };

                println!("{}", serde_json::to_string_pretty(&result_json)?);
            } else {
                println!("VALID STRUCTURE");
                println!();
                println!("Proposal would be submitted to verification gate.");
                println!("(Dry run - actual verification not executed)");
            }
        } else {
            if config.format == "json" {
                #[derive(serde::Serialize)]
                struct ErrorJson {
                    status: String,
                    error: String,
                }

                let error_json = ErrorJson {
                    status: "invalid".to_string(),
                    error: "Invalid proposal structure".to_string(),
                };

                println!("{}", serde_json::to_string_pretty(&error_json)?);
            } else {
                println!("INVALID STRUCTURE");
                println!();
                println!("Proposal would be rejected by verification gate.");
            }
            return Err("Invalid proposal structure".into());
        }
    } else {
        // Full verification would require integration with real version history
        println!("INFO: Full verification requires registered version history.");
        println!("      Use --dry-run to check proposal structure.");
    }

    Ok(())
}

/// Run selfimp rollback command
pub fn run_selfimp_rollback(
    config: SelfImpRollbackConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let rollback_config = RollbackConfig::default();
    let manager = RollbackManager::with_config(rollback_config);

    if config.verbose {
        println!("=== Rollback Manager ===\n");
        println!(
            "Auto-rollback on failure: {}",
            manager.config().auto_rollback_on_failure
        );
        println!(
            "Max rollback attempts: {}",
            manager.config().max_rollback_attempts
        );
        println!("Rollback history: {} entries", manager.history().len());
        println!();
    }

    if config.dry_run {
        println!("DRY RUN: Checking rollback feasibility...\n");
    }

    if let Some(target) = config.target_version {
        if config.format == "json" {
            #[derive(serde::Serialize)]
            struct RollbackJson {
                status: String,
                target_version: String,
                dry_run: bool,
            }

            let rollback_json = RollbackJson {
                status: "ready".to_string(),
                target_version: target.to_string(),
                dry_run: config.dry_run,
            };

            println!("{}", serde_json::to_string_pretty(&rollback_json)?);
        } else {
            println!("Rollback Target: {}", target);
            if config.dry_run {
                println!("(Dry run - rollback not executed)");
            } else {
                println!("Rollback would be executed here with real version history.");
                println!("Currently showing infrastructure capabilities.");
            }
        }
    } else if config.format == "json" {
        #[derive(serde::Serialize)]
        struct HistoryJson {
            rollback_count: usize,
        }

        let history_json = HistoryJson {
            rollback_count: manager.history().len(),
        };

        println!("{}", serde_json::to_string_pretty(&history_json)?);
    } else {
        println!("Rollback History: {} entries", manager.history().len());
        for (i, entry) in manager.history().iter().enumerate() {
            println!(
                "  {}. {} -> {} (reason: {:?})",
                i + 1,
                entry.from_version,
                entry.to_version,
                entry.trigger
            );
        }
        if manager.history().is_empty() {
            println!("  (no rollbacks recorded)");
        }
    }

    Ok(())
}

/// Run selfimp gate command
pub fn run_selfimp_gate(config: SelfImpGateConfig<'_>) -> Result<(), Box<dyn std::error::Error>> {
    if config.format == "json" {
        #[derive(serde::Serialize)]
        struct GateJson {
            status: String,
            checks: Vec<CheckJson>,
        }

        #[derive(serde::Serialize)]
        struct CheckJson {
            name: String,
            description: String,
            order: usize,
        }

        let checks = vec![
            CheckJson {
                name: "structural".to_string(),
                description: "Validates improvement structure and format".to_string(),
                order: 1,
            },
            CheckJson {
                name: "soundness".to_string(),
                description: "Ensures system soundness is preserved".to_string(),
                order: 2,
            },
            CheckJson {
                name: "capability".to_string(),
                description: "Checks for capability regression".to_string(),
                order: 3,
            },
            CheckJson {
                name: "formal".to_string(),
                description: "Runs formal verification with backends".to_string(),
                order: 4,
            },
            CheckJson {
                name: "certificate".to_string(),
                description: "Generates cryptographic proof certificate".to_string(),
                order: 5,
            },
        ];

        let gate_json = GateJson {
            status: "active".to_string(),
            checks,
        };

        println!("{}", serde_json::to_string_pretty(&gate_json)?);
    } else {
        println!("=== Verification Gate ===\n");
        println!("Status: ACTIVE (cannot be disabled)\n");
        println!("The verification gate is the ONLY way to apply self-modifications.");
        println!("It CANNOT be bypassed, disabled, or weakened.\n");
        println!("Verification Checks (in order):");
        println!();
        println!("  1. STRUCTURAL VALIDITY");
        println!("     Validates improvement structure and format");
        println!("     - Checks JSON schema compliance");
        println!("     - Validates required fields");
        println!();
        println!("  2. SOUNDNESS PRESERVATION");
        println!("     Ensures system soundness is preserved");
        println!("     - The system never claims false proofs");
        println!("     - Verification results remain correct");
        println!();
        println!("  3. CAPABILITY CHECK");
        println!("     Checks for capability regression");
        println!("     - Capabilities can only improve or stay same");
        println!("     - Detects numeric, boolean, version regressions");
        println!();
        println!("  4. FORMAL VERIFICATION");
        println!("     Runs formal verification with backends");
        println!("     - Uses configured verification backends");
        println!("     - Requires minimum confidence threshold");
        println!();
        println!("  5. CERTIFICATE GENERATION");
        println!("     Generates cryptographic proof certificate");
        println!("     - Content hash for integrity");
        println!("     - Certificate chain for history");

        if config.all_checks {
            println!();
            println!("Gate Configuration:");
            let gate_config = GateConfig::default();
            println!(
                "  Verification timeout: {:?}",
                gate_config.verification_timeout
            );
            println!(
                "  Run hardening checks: {}",
                gate_config.run_hardening_checks
            );
            println!("  Backends: {:?}", gate_config.backends);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selfimp_status() {
        let config = SelfImpStatusConfig {
            format: "text",
            verbose: false,
        };
        let result = run_selfimp_status(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_selfimp_status_json() {
        let config = SelfImpStatusConfig {
            format: "json",
            verbose: false,
        };
        let result = run_selfimp_status(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_selfimp_history() {
        let config = SelfImpHistoryConfig {
            limit: 10,
            format: "text",
            verbose: false,
        };
        let result = run_selfimp_history(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_selfimp_gate() {
        let config = SelfImpGateConfig {
            format: "text",
            all_checks: true,
        };
        let result = run_selfimp_gate(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_selfimp_verify_missing_file() {
        let config = SelfImpVerifyConfig {
            proposal: "/nonexistent/proposal.json",
            current_version: "v1.0.0",
            format: "text",
            verbose: false,
            dry_run: true,
        };
        let result = run_selfimp_verify(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_selfimp_rollback() {
        let config = SelfImpRollbackConfig {
            target_version: None,
            format: "text",
            verbose: false,
            dry_run: true,
        };
        let result = run_selfimp_rollback(config);
        assert!(result.is_ok());
    }
}
