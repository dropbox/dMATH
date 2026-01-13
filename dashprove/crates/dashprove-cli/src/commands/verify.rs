//! Verify command implementation

use crate::commands::common::{
    get_compiler_tactics, is_backend_available, parse_backend, resolve_data_dir,
};
use dashprove::{
    ai::{StrategyModel, StrategyPredictor},
    backends::{
        AlloyBackend, BackendId, CoqBackend, DafnyBackend, IsabelleBackend, KaniBackend,
        Lean4Backend, PlatformApiBackend, TlaPlusBackend, VerificationBackend, VerificationStatus,
    },
    dispatcher::{Dispatcher, DispatcherConfig, ProgressUpdate, SelectionStrategy},
    learning::{LearnableResult, ProofLearningSystem},
    usl::{parse, typecheck},
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

/// Configuration for the verify command
#[derive(Debug)]
pub struct VerifyConfig<'a> {
    pub path: &'a str,
    pub backends_filter: Option<&'a str>,
    pub timeout_secs: u64,
    pub skip_health_check: bool,
    pub learn: bool,
    pub data_dir: Option<&'a str>,
    pub suggest: bool,
    pub incremental: bool,
    pub since: Option<&'a str>,
    pub ml_enabled: bool,
    pub ml_model_path: Option<&'a str>,
    pub ml_min_confidence: f64,
    pub verbose: bool,
}

/// Get property names that have changed since a git ref
fn get_changed_properties(
    path: &str,
    since: Option<&str>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    use std::process::Command;

    let git_ref = since.unwrap_or("HEAD");

    // Get the diff of the specification file since the given ref
    let output = Command::new("git")
        .args(["diff", git_ref, "--", path])
        .output();

    let output = match output {
        Ok(o) => o,
        Err(_) => {
            // Git not available or not a git repo - verify all
            warn!("Git not available, verifying all properties");
            return Ok(vec!["*".to_string()]);
        }
    };

    if !output.status.success() {
        // Git ref doesn't exist or other error - verify all
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("unknown revision") {
            return Err(format!("Unknown git ref: {}", git_ref).into());
        }
        warn!("Git diff failed, verifying all properties");
        return Ok(vec!["*".to_string()]);
    }

    let diff = String::from_utf8_lossy(&output.stdout);

    // Parse diff to find changed property names
    let mut changed = Vec::new();
    let mut in_property = false;
    let mut current_property: Option<String> = None;

    for line in diff.lines() {
        // Lines starting with + or - indicate changes
        if line.starts_with('+') || line.starts_with('-') {
            // Skip diff headers
            if line.starts_with("+++") || line.starts_with("---") {
                continue;
            }

            let content = &line[1..]; // Remove the +/- prefix

            // Detect property declarations
            for keyword in &[
                "theorem",
                "temporal",
                "contract",
                "invariant",
                "refinement",
                "probabilistic",
                "security",
            ] {
                if content.trim().starts_with(keyword) {
                    // Extract property name
                    let rest = content.trim().strip_prefix(keyword).unwrap_or("");
                    if let Some(name) = rest.split_whitespace().next() {
                        current_property = Some(name.to_string());
                        in_property = true;
                    }
                    break;
                }
            }

            // If we're in a property block and see a change, mark it
            if in_property {
                if let Some(ref name) = current_property {
                    if !changed.contains(name) {
                        changed.push(name.clone());
                    }
                }
            }

            // Detect end of property block (closing brace at start of line)
            if content.trim() == "}" {
                in_property = false;
                current_property = None;
            }
        }
    }

    // If no specific changes detected but file changed, verify all
    if changed.is_empty() && !diff.is_empty() {
        changed.push("*".to_string());
    }

    Ok(changed)
}

/// Helper macro for verbose output
macro_rules! verbose_println {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!("[verbose] {}", format!($($arg)*));
        }
    };
}

/// Run verification command
pub async fn run_verify(config: VerifyConfig<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let VerifyConfig {
        path,
        backends_filter,
        timeout_secs,
        skip_health_check,
        learn,
        data_dir,
        suggest,
        incremental,
        since,
        ml_enabled,
        ml_model_path,
        ml_min_confidence,
        verbose,
    } = config;
    // Read specification file
    let spec_path = Path::new(path);
    if !spec_path.exists() {
        return Err(format!("Specification file not found: {}", path).into());
    }

    verbose_println!(verbose, "Reading specification from {}", path);
    let spec_content = std::fs::read_to_string(spec_path)?;
    info!("Read specification from {}", path);

    // Parse and type-check
    verbose_println!(verbose, "Parsing USL specification...");
    let spec = parse(&spec_content).map_err(|e| format!("Parse error: {:?}", e))?;
    verbose_println!(
        verbose,
        "Parsed {} properties successfully",
        spec.properties.len()
    );
    info!("Parsed {} properties", spec.properties.len());

    verbose_println!(verbose, "Type-checking specification...");
    let typed_spec = typecheck(spec).map_err(|e| format!("Type error: {:?}", e))?;
    verbose_println!(verbose, "Type checking passed");
    info!("Type checking passed");

    // Resolve data directory for learning/ML assets
    let data_dir_path = resolve_data_dir(data_dir);

    // Handle incremental verification
    let properties_to_verify = if incremental {
        let changed = get_changed_properties(path, since)?;
        if changed.is_empty() {
            println!(
                "No changes detected since {}. Skipping verification.",
                since.unwrap_or("HEAD")
            );
            return Ok(());
        }
        println!(
            "Incremental mode: {} of {} properties changed since {}",
            changed.len(),
            typed_spec.spec.properties.len(),
            since.unwrap_or("HEAD")
        );
        Some(changed)
    } else {
        None
    };

    // Show tactic suggestions if requested
    if suggest {
        println!("\n=== Tactic Suggestions ===");

        // Load learning system if available
        let learning_system = ProofLearningSystem::load_from_dir(&data_dir_path).ok();

        for property in &typed_spec.spec.properties {
            let prop_name = property.name();
            println!("\nProperty: {}", prop_name);

            // Compiler-based suggestions (from expression structure)
            let compiler_tactics = get_compiler_tactics(property);
            if !compiler_tactics.is_empty() {
                println!("  Compiler suggestions: {}", compiler_tactics.join(", "));
            }

            // Learning-based suggestions (from past proofs)
            if let Some(ref system) = learning_system {
                let learned = system.suggest_tactics(property, 5);
                if !learned.is_empty() {
                    println!("  Learned tactics:");
                    for (tactic, score) in learned {
                        println!("    {} (score: {:.3})", tactic, score);
                    }
                }

                // Similar proofs
                let similar = system.find_similar(property, 3);
                if !similar.is_empty() {
                    println!("  Similar proofs:");
                    for proof in similar {
                        let similar_name = proof.property.name();
                        println!(
                            "    {} ({:.0}% similar, {:?})",
                            similar_name,
                            proof.similarity * 100.0,
                            proof.backend
                        );
                    }
                }
            }
        }
        println!();
    }

    // Create dispatcher
    verbose_println!(
        verbose,
        "Configuring verification dispatcher (timeout: {}s)",
        timeout_secs
    );
    let mut config = DispatcherConfig {
        task_timeout: std::time::Duration::from_secs(timeout_secs),
        check_health: !skip_health_check,
        ..Default::default()
    };

    // Parse backend filter
    let requested_backends: Option<Vec<BackendId>> = backends_filter.map(|s| {
        s.split(',')
            .filter_map(|name| parse_backend(name.trim()))
            .collect()
    });

    if let Some(ref backends) = requested_backends {
        verbose_println!(verbose, "Requested backends: {:?}", backends);
        if backends.len() == 1 {
            config.selection_strategy = SelectionStrategy::Specific(backends[0]);
        }
    }

    if ml_enabled {
        let min_conf = ml_min_confidence.clamp(0.0, 1.0);
        verbose_println!(
            verbose,
            "ML-based backend selection enabled (min confidence: {:.2})",
            min_conf
        );
        config.selection_strategy = SelectionStrategy::MlBased {
            min_confidence: min_conf,
        };
    }

    let ml_predictor = if ml_enabled {
        Some(load_ml_predictor(ml_model_path, &data_dir_path, verbose))
    } else {
        None
    };

    let mut dispatcher = if let Some(predictor) = ml_predictor {
        Dispatcher::with_ml_predictor(config, predictor)
    } else {
        Dispatcher::new(config)
    };

    let progress_enabled = std::env::var("DASHPROVE_PROGRESS")
        .map(|v| !matches!(v.to_lowercase().as_str(), "0" | "false" | "off"))
        .unwrap_or(true);
    if progress_enabled {
        let start_time = Instant::now();
        dispatcher.set_progress_callback(move |update: ProgressUpdate| {
            let elapsed = start_time.elapsed().as_secs_f32();
            println!(
                "[progress] {}/{} tasks complete (property #{}, backend {:?}, {:.1}s elapsed)",
                update.completed,
                update.total,
                update.property_index + 1,
                update.backend,
                elapsed
            );
        });
    }

    // Register available backends
    verbose_println!(verbose, "Checking backend availability...");
    let lean4 = Arc::new(Lean4Backend::new());
    let tlaplus = Arc::new(TlaPlusBackend::new());
    let kani = Arc::new(KaniBackend::new());
    let alloy = Arc::new(AlloyBackend::new());
    let isabelle = Arc::new(IsabelleBackend::new());
    let coq = Arc::new(CoqBackend::new());
    let dafny = Arc::new(DafnyBackend::new());
    let platform_api = Arc::new(PlatformApiBackend::new());

    let all_backends: Vec<(BackendId, Arc<dyn VerificationBackend>)> = vec![
        (BackendId::Lean4, lean4.clone()),
        (BackendId::TlaPlus, tlaplus.clone()),
        (BackendId::Kani, kani.clone()),
        (BackendId::Alloy, alloy.clone()),
        (BackendId::Isabelle, isabelle.clone()),
        (BackendId::Coq, coq.clone()),
        (BackendId::Dafny, dafny.clone()),
        (BackendId::PlatformApi, platform_api.clone()),
    ];

    for (id, backend) in all_backends {
        // Skip if not in requested list
        if let Some(ref requested) = requested_backends {
            if !requested.contains(&id) {
                verbose_println!(verbose, "  {:?}: skipped (not requested)", id);
                continue;
            }
        }

        // Check availability (unless skipping health checks)
        if skip_health_check || is_backend_available(backend.as_ref()).await {
            dispatcher.register_backend(backend);
            verbose_println!(verbose, "  {:?}: registered (available)", id);
            info!("Registered backend: {:?}", id);
        } else {
            verbose_println!(verbose, "  {:?}: skipped (not available)", id);
            warn!("Backend {:?} not available, skipping", id);
        }
    }

    if dispatcher.registry().is_empty() {
        return Err("No backends available for verification".into());
    }
    verbose_println!(
        verbose,
        "Dispatcher configured with {} backends",
        dispatcher.registry().len()
    );

    // Filter properties for incremental verification
    let verify_count = if let Some(ref changed) = properties_to_verify {
        if changed.contains(&"*".to_string()) {
            typed_spec.spec.properties.len()
        } else {
            typed_spec
                .spec
                .properties
                .iter()
                .filter(|p| changed.contains(&p.name()))
                .count()
        }
    } else {
        typed_spec.spec.properties.len()
    };

    // Run verification
    println!("Verifying {} properties...", verify_count);
    verbose_println!(verbose, "Starting verification with dispatcher...");

    // For incremental, filter the typed_spec to only include changed properties
    let results = if let Some(ref changed) = properties_to_verify {
        if changed.contains(&"*".to_string()) {
            dispatcher.verify(&typed_spec).await?
        } else {
            // Create a filtered spec with only changed properties
            verbose_println!(verbose, "Filtering to changed properties: {:?}", changed);
            let mut filtered_spec = typed_spec.clone();
            filtered_spec
                .spec
                .properties
                .retain(|p| changed.contains(&p.name()));

            if filtered_spec.spec.properties.is_empty() {
                println!("No matching changed properties found.");
                return Ok(());
            }

            dispatcher.verify(&filtered_spec).await?
        }
    } else {
        dispatcher.verify(&typed_spec).await?
    };
    verbose_println!(verbose, "Verification complete, processing results...");

    // Print results
    println!("\n=== Verification Results ===");
    println!("Properties: {}", results.properties.len());
    println!("  Proven:    {}", results.summary.proven);
    println!("  Disproven: {}", results.summary.disproven);
    println!("  Unknown:   {}", results.summary.unknown);
    println!(
        "  Confidence: {:.1}%",
        results.summary.overall_confidence * 100.0
    );

    for (i, prop_result) in results.properties.iter().enumerate() {
        // Get property name from typed_spec if possible
        let prop_name = typed_spec
            .spec
            .properties
            .get(prop_result.property_index)
            .map(|p| match p {
                dashprove::usl::Property::Theorem(t) => t.name.clone(),
                dashprove::usl::Property::Temporal(t) => t.name.clone(),
                dashprove::usl::Property::Contract(c) => c.type_path.join("::"),
                dashprove::usl::Property::Invariant(inv) => inv.name.clone(),
                dashprove::usl::Property::Refinement(r) => r.name.clone(),
                dashprove::usl::Property::Probabilistic(p) => p.name.clone(),
                dashprove::usl::Property::Security(s) => s.name.clone(),
                dashprove::usl::Property::Semantic(s) => s.name.clone(),
                dashprove::usl::Property::PlatformApi(p) => p.name.clone(),
                dashprove::usl::Property::Bisimulation(b) => b.name.clone(),
                dashprove::usl::Property::Version(v) => v.name.clone(),
                dashprove::usl::Property::Capability(c) => c.name.clone(),
                dashprove::usl::Property::DistributedInvariant(d) => d.name.clone(),
                dashprove::usl::Property::DistributedTemporal(d) => d.name.clone(),
                dashprove::usl::Property::Composed(c) => c.name.clone(),
                dashprove::usl::Property::ImprovementProposal(p) => p.name.clone(),
                dashprove::usl::Property::VerificationGate(g) => g.name.clone(),
                dashprove::usl::Property::Rollback(r) => r.name.clone(),
            })
            .unwrap_or_else(|| format!("property_{}", i));

        println!("\nProperty: {}", prop_name);

        // Print merged result info
        let status_str = match &prop_result.status {
            VerificationStatus::Proven => "PROVEN".to_string(),
            VerificationStatus::Disproven => "DISPROVEN".to_string(),
            VerificationStatus::Unknown { reason } => format!("UNKNOWN ({})", reason),
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                format!("PARTIAL ({:.0}%)", verified_percentage * 100.0)
            }
        };
        println!(
            "  Status: {} (confidence: {:.1}%)",
            status_str,
            prop_result.confidence * 100.0
        );

        for backend_result in &prop_result.backend_results {
            let backend_status = match &backend_result.status {
                VerificationStatus::Proven => "PROVEN".to_string(),
                VerificationStatus::Disproven => "DISPROVEN".to_string(),
                VerificationStatus::Unknown { reason } => format!("UNKNOWN ({})", reason),
                VerificationStatus::Partial {
                    verified_percentage,
                } => {
                    format!("PARTIAL ({:.0}%)", verified_percentage * 100.0)
                }
            };
            println!(
                "    [{:?}] {} ({:?})",
                backend_result.backend, backend_status, backend_result.time_taken
            );

            if let Some(ref err) = backend_result.error {
                println!("      Error: {}", err);
            }
        }

        // Print counterexample and diagnostics from merged result
        if let Some(ref ce) = prop_result.counterexample {
            println!("  Counterexample: {}", ce);
        }

        for diag in &prop_result.diagnostics {
            println!("  {}", diag);
        }
    }

    // Record results to learning corpus if --learn flag is set
    if learn {
        println!(
            "\nRecording results to learning corpus: {}",
            data_dir_path.display()
        );

        // Load existing learning system or create new
        let mut system = ProofLearningSystem::load_from_dir(&data_dir_path)
            .unwrap_or_else(|_| ProofLearningSystem::new());

        let mut recorded_count = 0;
        for prop_result in &results.properties {
            // Get the property
            if let Some(property) = typed_spec.spec.properties.get(prop_result.property_index) {
                // Record each backend result
                for backend_result in &prop_result.backend_results {
                    // Extract tactics for backends that use them
                    // Lean4, Coq, and Isabelle use tactic-based proving
                    // Other backends use automated checking without explicit tactics
                    let tactics = match backend_result.backend {
                        BackendId::Lean4 | BackendId::Coq | BackendId::Isabelle => {
                            get_compiler_tactics(property)
                        }
                        _ => vec![],
                    };
                    let learnable = LearnableResult {
                        property: property.clone(),
                        backend: backend_result.backend,
                        status: backend_result.status.clone(),
                        tactics,
                        time_taken: backend_result.time_taken,
                        proof_output: None,
                    };
                    system.record(&learnable);
                    recorded_count += 1;
                }
            }
        }

        // Record counterexamples from failed verifications
        let mut cx_recorded_count = 0;
        for prop_result in &results.properties {
            // Only record if the property was disproven and has a counterexample
            if matches!(prop_result.status, VerificationStatus::Disproven) {
                if let Some(ref ce) = prop_result.counterexample {
                    // Get property name
                    let prop_name = typed_spec
                        .spec
                        .properties
                        .get(prop_result.property_index)
                        .map(|p| match p {
                            dashprove::usl::Property::Theorem(t) => t.name.clone(),
                            dashprove::usl::Property::Temporal(t) => t.name.clone(),
                            dashprove::usl::Property::Contract(c) => c.type_path.join("::"),
                            dashprove::usl::Property::Invariant(inv) => inv.name.clone(),
                            dashprove::usl::Property::Refinement(r) => r.name.clone(),
                            dashprove::usl::Property::Probabilistic(p) => p.name.clone(),
                            dashprove::usl::Property::Security(s) => s.name.clone(),
                            dashprove::usl::Property::Semantic(s) => s.name.clone(),
                            dashprove::usl::Property::PlatformApi(p) => p.name.clone(),
                            dashprove::usl::Property::Bisimulation(b) => b.name.clone(),
                            dashprove::usl::Property::Version(v) => v.name.clone(),
                            dashprove::usl::Property::Capability(c) => c.name.clone(),
                            dashprove::usl::Property::DistributedInvariant(d) => d.name.clone(),
                            dashprove::usl::Property::DistributedTemporal(d) => d.name.clone(),
                            dashprove::usl::Property::Composed(c) => c.name.clone(),
                            dashprove::usl::Property::ImprovementProposal(p) => p.name.clone(),
                            dashprove::usl::Property::VerificationGate(g) => g.name.clone(),
                            dashprove::usl::Property::Rollback(r) => r.name.clone(),
                        })
                        .unwrap_or_else(|| format!("property_{}", prop_result.property_index));

                    // Get the first backend that disproved it (for recording)
                    let backend = prop_result
                        .backend_results
                        .iter()
                        .find(|br| matches!(br.status, VerificationStatus::Disproven))
                        .map(|br| br.backend)
                        .unwrap_or(BackendId::TlaPlus);

                    system.record_counterexample(&prop_name, backend, ce.clone(), None);
                    cx_recorded_count += 1;
                }
            }
        }

        // Save the updated learning system
        if let Err(e) = system.save_to_dir(&data_dir_path) {
            warn!("Failed to save learning data: {}", e);
        } else {
            println!(
                "Recorded {} results (corpus now has {} proofs)",
                recorded_count,
                system.corpus.len()
            );
            if cx_recorded_count > 0 {
                println!(
                    "Recorded {} counterexamples (corpus now has {} counterexamples)",
                    cx_recorded_count,
                    system.counterexample_count()
                );
            }
        }
    }

    Ok(())
}

fn load_ml_predictor(
    explicit_path: Option<&str>,
    data_dir: &Path,
    verbose: bool,
) -> Arc<StrategyModel> {
    let default_path = data_dir.join("strategy_model.json");
    let candidate_path = explicit_path.map(PathBuf::from).or_else(|| {
        if default_path.exists() {
            Some(default_path.clone())
        } else {
            None
        }
    });

    if let Some(path) = candidate_path {
        verbose_println!(verbose, "Loading ML strategy model from {}", path.display());
        match StrategyModel::load(&path) {
            Ok(model) => {
                info!("Loaded ML strategy model from {}", path.display());
                return Arc::new(model);
            }
            Err(err) => {
                warn!(
                    "Failed to load ML model from {}: {}. Using fresh predictor.",
                    path.display(),
                    err
                );
            }
        }
    } else {
        verbose_println!(
            verbose,
            "No ML model file found at {} (or custom path). Using fresh predictor.",
            default_path.display()
        );
    }

    Arc::new(StrategyModel::from(StrategyPredictor::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::common::get_compiler_tactics;
    use tempfile::tempdir;

    #[test]
    fn load_ml_predictor_uses_explicit_path() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("custom_model.json");
        StrategyPredictor::new().save(&model_path).unwrap();

        let predictor = load_ml_predictor(model_path.to_str(), dir.path(), false);

        match predictor.as_ref() {
            StrategyModel::Single { model } => {
                let raw = StrategyPredictor::load(&model_path).unwrap();
                let loaded_json = serde_json::to_value(model.as_ref()).unwrap();
                let raw_json = serde_json::to_value(&raw).unwrap();
                assert_eq!(loaded_json, raw_json);
            }
            StrategyModel::Ensemble { .. } => panic!("expected single model"),
        }
    }

    #[test]
    fn load_ml_predictor_uses_default_model_in_data_dir() {
        let dir = tempdir().unwrap();
        let default_path = dir.path().join("strategy_model.json");
        StrategyPredictor::new().save(&default_path).unwrap();

        let predictor = load_ml_predictor(None, dir.path(), false);

        match predictor.as_ref() {
            StrategyModel::Single { model } => {
                let raw = StrategyPredictor::load(&default_path).unwrap();
                let loaded_json = serde_json::to_value(model.as_ref()).unwrap();
                let raw_json = serde_json::to_value(&raw).unwrap();
                assert_eq!(loaded_json, raw_json);
            }
            StrategyModel::Ensemble { .. } => panic!("expected single model"),
        }
    }

    #[test]
    fn load_ml_predictor_falls_back_to_fresh_model() {
        let dir = tempdir().unwrap();
        let default_path = dir.path().join("strategy_model.json");
        assert!(!default_path.exists());

        let predictor = load_ml_predictor(None, dir.path(), false);

        assert!(!default_path.exists());
        // Should still produce a predictor
        let property = dashprove::usl::Property::Theorem(dashprove::usl::ast::Theorem {
            name: "sample".into(),
            body: dashprove::usl::ast::Expr::Bool(true),
        });
        let prediction = predictor.predict_strategy(&property);
        assert!(prediction.backend.confidence.is_finite());
    }

    #[test]
    fn tactic_extraction_for_tactic_backends() {
        // Test that get_compiler_tactics returns tactics for theorem properties
        let property = dashprove::usl::Property::Theorem(dashprove::usl::ast::Theorem {
            name: "simple_bool".into(),
            body: dashprove::usl::ast::Expr::Bool(true),
        });

        let tactics = get_compiler_tactics(&property);

        // For a simple boolean, the compiler should suggest "trivial"
        assert!(
            !tactics.is_empty(),
            "tactics should not be empty for theorem"
        );
        assert!(
            tactics.iter().any(|t| t.contains("trivial")),
            "expected 'trivial' tactic for simple boolean, got: {:?}",
            tactics
        );
    }

    #[test]
    fn tactic_extraction_for_arithmetic() {
        // Test that arithmetic properties get appropriate tactics
        use dashprove::usl::ast::{BinaryOp, ComparisonOp, Expr};

        let property = dashprove::usl::Property::Theorem(dashprove::usl::ast::Theorem {
            name: "arithmetic".into(),
            body: Expr::Compare(
                Box::new(Expr::Binary(
                    Box::new(Expr::Var("x".into())),
                    BinaryOp::Add,
                    Box::new(Expr::Int(1)),
                )),
                ComparisonOp::Gt,
                Box::new(Expr::Var("x".into())),
            ),
        });

        let tactics = get_compiler_tactics(&property);

        // Arithmetic comparisons should get omega or similar tactic
        assert!(
            !tactics.is_empty(),
            "tactics should not be empty for arithmetic"
        );
    }

    #[test]
    fn tactic_extraction_empty_for_temporal() {
        // Temporal properties use model checking, not tactics
        use dashprove::usl::ast::{Expr, Temporal, TemporalExpr};

        let property = dashprove::usl::Property::Temporal(Temporal {
            name: "eventually_done".into(),
            body: TemporalExpr::Eventually(Box::new(TemporalExpr::Atom(Expr::Var("done".into())))),
            fairness: vec![],
        });

        let tactics = get_compiler_tactics(&property);

        // Temporal properties don't use LEAN tactics
        assert!(
            tactics.is_empty(),
            "temporal properties should have no tactics, got: {:?}",
            tactics
        );
    }
}
