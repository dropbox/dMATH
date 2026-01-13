//! Verification route handlers for DashProve server

use axum::{extract::State, http::StatusCode, Json};
use dashprove_ai::PropertyFeatureVector;
use dashprove_backends::BackendId;
use dashprove_usl::{
    ast::Property, compile::compile_to_platform_api, parse, typecheck, DependencyGraph, SpecDiff,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use super::types::{
    backend_metric_label, default_backends, BackendIdParam, ChangeKind, CompilationResult,
    ErrorResponse, IncrementalVerifyRequest, IncrementalVerifyResponse, MlPredictionInfo,
    VerifyRequest, VerifyResponse,
};
use super::AppState;
use crate::cache::ProofCache;

/// POST /verify - Verify a specification
pub async fn verify(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, (StatusCode, Json<ErrorResponse>)> {
    let overall_start = Instant::now();

    // Parse the specification
    let spec = match parse(&req.spec) {
        Ok(spec) => spec,
        Err(e) => {
            state
                .metrics
                .record_verification_with_duration(false, overall_start.elapsed().as_secs_f64())
                .await;
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Parse error".to_string(),
                    details: Some(e.to_string()),
                }),
            ));
        }
    };

    // Type-check
    let typed_spec = match typecheck(spec) {
        Ok(typed) => typed,
        Err(e) => {
            state
                .metrics
                .record_verification_with_duration(false, overall_start.elapsed().as_secs_f64())
                .await;
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Type error".to_string(),
                    details: Some(e.to_string()),
                }),
            ));
        }
    };

    // Determine backends to compile to
    let (backends, ml_prediction) = if req.use_ml {
        // Use ML-based backend selection
        ml_select_backends(&state, &typed_spec.spec.properties, req.ml_min_confidence)
    } else if let Some(backend) = req.backend {
        (vec![backend.into()], None)
    } else {
        (default_backends(), None)
    };

    let mut compilations = Vec::new();
    let errors = Vec::new();

    for backend in backends {
        let backend_label = backend_metric_label(backend);
        let backend_start = Instant::now();

        match backend {
            BackendId::Lean4 => {
                let output = dashprove_usl::compile_to_lean(&typed_spec);
                compilations.push(CompilationResult {
                    backend: BackendIdParam::Lean4,
                    code: output.code,
                });
            }
            BackendId::TlaPlus => {
                let output = dashprove_usl::compile_to_tlaplus(&typed_spec);
                compilations.push(CompilationResult {
                    backend: BackendIdParam::TlaPlus,
                    code: output.code,
                });
            }
            BackendId::Apalache => {
                // Apalache uses TLA+ specifications
                let output = dashprove_usl::compile_to_tlaplus(&typed_spec);
                compilations.push(CompilationResult {
                    backend: BackendIdParam::Apalache,
                    code: output.code,
                });
            }
            BackendId::Kani => {
                let output = dashprove_usl::compile_to_kani(&typed_spec);
                compilations.push(CompilationResult {
                    backend: BackendIdParam::Kani,
                    code: output.code,
                });
            }
            BackendId::Alloy => {
                let output = dashprove_usl::compile_to_alloy(&typed_spec);
                compilations.push(CompilationResult {
                    backend: BackendIdParam::Alloy,
                    code: output.code,
                });
            }
            BackendId::Isabelle => {
                let output = dashprove_usl::compile_to_isabelle(&typed_spec);
                compilations.push(CompilationResult {
                    backend: BackendIdParam::Isabelle,
                    code: output.code,
                });
            }
            BackendId::Coq => {
                let output = dashprove_usl::compile_to_coq(&typed_spec);
                compilations.push(CompilationResult {
                    backend: BackendIdParam::Coq,
                    code: output.code,
                });
            }
            BackendId::Dafny => {
                let output = dashprove_usl::compile_to_dafny(&typed_spec);
                compilations.push(CompilationResult {
                    backend: BackendIdParam::Dafny,
                    code: output.code,
                });
            }
            BackendId::PlatformApi => {
                let output = compile_to_platform_api(&typed_spec);
                let (backend, code) = if let Some(out) = output {
                    (BackendIdParam::PlatformApi, out.code)
                } else {
                    (
                        BackendIdParam::PlatformApi,
                        "// no platform_api specifications found in input".to_string(),
                    )
                };
                compilations.push(CompilationResult { backend, code });
            }
            // Neural network verification backends (not yet implemented)
            BackendId::Marabou | BackendId::AlphaBetaCrown | BackendId::Eran => {
                compilations.push(CompilationResult {
                    backend: backend.into(),
                    code: format!("// {} backend not yet implemented", backend_label),
                });
            }
            // Probabilistic verification backends (not yet implemented)
            BackendId::Storm | BackendId::Prism => {
                compilations.push(CompilationResult {
                    backend: backend.into(),
                    code: format!("// {} backend not yet implemented", backend_label),
                });
            }
            // Security protocol verification backends (not yet implemented)
            BackendId::Tamarin | BackendId::ProVerif | BackendId::Verifpal => {
                compilations.push(CompilationResult {
                    backend: backend.into(),
                    code: format!("// {} backend not yet implemented", backend_label),
                });
            }
            // Rust verification backends (not yet implemented)
            BackendId::Verus | BackendId::Creusot | BackendId::Prusti => {
                compilations.push(CompilationResult {
                    backend: backend.into(),
                    code: format!("// {} backend not yet implemented", backend_label),
                });
            }
            // SMT solver backends
            BackendId::Z3 | BackendId::Cvc5 => {
                let output = dashprove_usl::compile_to_smtlib2(&typed_spec);
                compilations.push(CompilationResult {
                    backend: backend.into(),
                    code: output.code,
                });
            }
            // All other backends (Phase 12 additions) - not yet implemented
            _ => {
                compilations.push(CompilationResult {
                    backend: backend.into(),
                    code: format!(
                        "// {} backend compilation not yet implemented",
                        backend_label
                    ),
                });
            }
        }

        state
            .metrics
            .record_backend_duration(backend_label, true, backend_start.elapsed().as_secs_f64())
            .await;
    }

    state
        .metrics
        .record_verification_with_duration(true, overall_start.elapsed().as_secs_f64())
        .await;

    Ok(Json(VerifyResponse {
        valid: true,
        property_count: typed_spec.spec.properties.len(),
        compilations,
        errors,
        ml_prediction,
    }))
}

/// Use ML predictor to select backends for a set of properties
fn ml_select_backends(
    state: &AppState,
    properties: &[Property],
    min_confidence: f64,
) -> (Vec<BackendId>, Option<MlPredictionInfo>) {
    let predictor = match &state.ml_predictor {
        Some(p) => p,
        None => {
            // No ML predictor configured, fall back to all backends
            return (default_backends(), None);
        }
    };

    if properties.is_empty() {
        return (default_backends(), None);
    }

    // For multiple properties, we aggregate predictions
    // Simple approach: use the first property for prediction (for now)
    // A more sophisticated approach would aggregate across all properties
    let first_property = &properties[0];
    let features = PropertyFeatureVector::from_property(first_property);
    let prediction = predictor.predict_backend(&features);

    let used = prediction.confidence >= min_confidence;
    let backends = if used {
        vec![prediction.backend]
    } else {
        // Confidence too low, fall back to default backends
        default_backends()
    };

    let ml_info = MlPredictionInfo {
        predicted_backend: prediction.backend.into(),
        confidence: prediction.confidence,
        used,
        alternatives: prediction
            .alternatives
            .into_iter()
            .map(|(b, c)| (b.into(), c))
            .collect(),
    };

    (backends, Some(ml_info))
}

/// POST /verify/incremental - Incremental verification after changes
///
/// Compares base and current specs, identifies affected properties,
/// and only re-verifies those that changed. Uses proof result caching
/// to return cached results for unchanged properties across requests.
pub async fn verify_incremental(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IncrementalVerifyRequest>,
) -> Result<Json<IncrementalVerifyResponse>, (StatusCode, Json<ErrorResponse>)> {
    let overall_start = Instant::now();

    // Parse both specifications
    let base_spec = match parse(&req.base_spec) {
        Ok(parsed) => parsed,
        Err(e) => {
            state
                .metrics
                .record_verification_with_duration(false, overall_start.elapsed().as_secs_f64())
                .await;
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Base spec parse error".to_string(),
                    details: Some(e.to_string()),
                }),
            ));
        }
    };

    let current_spec = match parse(&req.current_spec) {
        Ok(parsed) => parsed,
        Err(e) => {
            state
                .metrics
                .record_verification_with_duration(false, overall_start.elapsed().as_secs_f64())
                .await;
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Current spec parse error".to_string(),
                    details: Some(e.to_string()),
                }),
            ));
        }
    };

    // Type-check both
    let base_typed = match typecheck(base_spec) {
        Ok(typed) => typed,
        Err(e) => {
            state
                .metrics
                .record_verification_with_duration(false, overall_start.elapsed().as_secs_f64())
                .await;
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Base spec type error".to_string(),
                    details: Some(e.to_string()),
                }),
            ));
        }
    };

    let current_typed = match typecheck(current_spec) {
        Ok(typed) => typed,
        Err(e) => {
            state
                .metrics
                .record_verification_with_duration(false, overall_start.elapsed().as_secs_f64())
                .await;
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Current spec type error".to_string(),
                    details: Some(e.to_string()),
                }),
            ));
        }
    };

    // Use dependency graph and spec diff for accurate change detection
    let spec_diff = SpecDiff::diff(&base_typed.spec, &current_typed.spec);
    let dep_graph = DependencyGraph::from_spec(&current_typed.spec);

    // Collect changed types from both the diff and explicit change targets
    let mut changed_types: Vec<String> = spec_diff.added_types.clone();
    changed_types.extend(spec_diff.modified_types.clone());

    // Also include types from explicit change targets in the request
    for change in &req.changes {
        if matches!(
            change.kind,
            ChangeKind::TypeAdded | ChangeKind::TypeModified
        ) && !changed_types.contains(&change.target)
        {
            changed_types.push(change.target.clone());
        }
    }

    // Collect changed functions from explicit change targets
    let changed_functions: Vec<String> = req
        .changes
        .iter()
        .filter(|c| matches!(c.kind, ChangeKind::DependencyChanged))
        .map(|c| c.target.clone())
        .collect();

    // Properties that are directly modified
    let directly_changed: Vec<String> = spec_diff.modified_properties.clone();

    // Use dependency graph to compute all affected properties
    let affected_set =
        dep_graph.compute_affected(&changed_types, &changed_functions, &directly_changed);

    // Create property hash map for cache lookups
    let property_hashes: HashMap<String, u64> = current_typed
        .spec
        .properties
        .iter()
        .map(|p| (p.name().to_string(), ProofCache::hash_property(p)))
        .collect();

    // Invalidate cache entries for affected properties
    {
        let mut cache = state.proof_cache.write().await;
        let affected_names: Vec<String> = affected_set.iter().cloned().collect();
        cache.invalidate_affected(&affected_names);
    }

    // Build the affected and unchanged lists
    let current_names: std::collections::HashSet<_> = current_typed
        .spec
        .properties
        .iter()
        .map(Property::name)
        .collect();

    let mut affected_properties: Vec<String> = Vec::new();
    let mut unchanged_properties: Vec<String> = Vec::new();
    let mut cache_hits: Vec<String> = Vec::new();

    // New properties are always affected
    for name in &spec_diff.added_properties {
        affected_properties.push(name.clone());
    }

    // Check each current property against cache
    {
        let cache = state.proof_cache.read().await;
        for name in &current_names {
            if spec_diff.added_properties.contains(name) {
                continue; // Already added above
            }
            if affected_set.contains(name) {
                affected_properties.push(name.clone());
            } else {
                // Check if we have a valid cache entry for this property
                if let Some(&hash) = property_hashes.get(name) {
                    if cache.get(name, hash).is_some() {
                        cache_hits.push(name.clone());
                    }
                }
                unchanged_properties.push(name.clone());
            }
        }
    }

    // Track deleted properties
    for name in &spec_diff.removed_properties {
        affected_properties.push(format!("{name} (deleted)"));
    }

    // Compile only affected properties
    let backends: Vec<BackendId> = if let Some(backend) = req.backend {
        vec![backend.into()]
    } else {
        default_backends()
    };

    let mut compilations = Vec::new();
    let errors = Vec::new();

    // Compile current spec (for affected properties, we compile the whole spec
    // since property isolation isn't implemented yet)
    if !affected_properties.is_empty() {
        for backend in &backends {
            let backend_label = backend_metric_label(*backend);
            let backend_start = Instant::now();

            let (code, backend_param) = match backend {
                BackendId::Lean4 => {
                    let output = dashprove_usl::compile_to_lean(&current_typed);
                    (output.code, BackendIdParam::Lean4)
                }
                BackendId::TlaPlus => {
                    let output = dashprove_usl::compile_to_tlaplus(&current_typed);
                    (output.code, BackendIdParam::TlaPlus)
                }
                BackendId::Apalache => {
                    // Apalache uses TLA+ specifications
                    let output = dashprove_usl::compile_to_tlaplus(&current_typed);
                    (output.code, BackendIdParam::Apalache)
                }
                BackendId::Kani => {
                    let output = dashprove_usl::compile_to_kani(&current_typed);
                    (output.code, BackendIdParam::Kani)
                }
                BackendId::Alloy => {
                    let output = dashprove_usl::compile_to_alloy(&current_typed);
                    (output.code, BackendIdParam::Alloy)
                }
                BackendId::Isabelle => {
                    let output = dashprove_usl::compile_to_isabelle(&current_typed);
                    (output.code, BackendIdParam::Isabelle)
                }
                BackendId::Coq => {
                    let output = dashprove_usl::compile_to_coq(&current_typed);
                    (output.code, BackendIdParam::Coq)
                }
                BackendId::Dafny => {
                    let output = dashprove_usl::compile_to_dafny(&current_typed);
                    (output.code, BackendIdParam::Dafny)
                }
                BackendId::PlatformApi => {
                    let output = compile_to_platform_api(&current_typed);
                    if let Some(out) = output {
                        (out.code, BackendIdParam::PlatformApi)
                    } else {
                        (
                            "// no platform_api specifications found in input".to_string(),
                            BackendIdParam::PlatformApi,
                        )
                    }
                }
                // Neural network verification backends (not yet implemented)
                BackendId::Marabou | BackendId::AlphaBetaCrown | BackendId::Eran => (
                    format!("// {} backend not yet implemented", backend_label),
                    (*backend).into(),
                ),
                // Probabilistic verification backends (not yet implemented)
                BackendId::Storm | BackendId::Prism => (
                    format!("// {} backend not yet implemented", backend_label),
                    (*backend).into(),
                ),
                // Security protocol verification backends (not yet implemented)
                BackendId::Tamarin | BackendId::ProVerif | BackendId::Verifpal => (
                    format!("// {} backend not yet implemented", backend_label),
                    (*backend).into(),
                ),
                // Rust verification backends (not yet implemented)
                BackendId::Verus | BackendId::Creusot | BackendId::Prusti => (
                    format!("// {} backend not yet implemented", backend_label),
                    (*backend).into(),
                ),
                // SMT solver backends
                BackendId::Z3 | BackendId::Cvc5 => {
                    let output = dashprove_usl::compile_to_smtlib2(&current_typed);
                    (output.code, (*backend).into())
                }
                // All other backends (Phase 12 additions) - not yet implemented
                _ => (
                    format!(
                        "// {} backend compilation not yet implemented",
                        backend_label
                    ),
                    (*backend).into(),
                ),
            };

            state
                .metrics
                .record_backend_duration(backend_label, true, backend_start.elapsed().as_secs_f64())
                .await;

            compilations.push(CompilationResult {
                backend: backend_param,
                code: code.clone(),
            });

            // Cache the compilation results for affected properties
            {
                let mut cache = state.proof_cache.write().await;
                for name in affected_properties
                    .iter()
                    .filter(|p| !p.ends_with("(deleted)"))
                {
                    if let Some(&hash) = property_hashes.get(name) {
                        cache.put(name.clone(), hash, true, *backend, code.clone(), None);
                    }
                }
            }
        }
    }

    // Filter out deleted markers for counting
    let affected_count = affected_properties
        .iter()
        .filter(|p| !p.ends_with("(deleted)"))
        .count();

    state
        .metrics
        .record_verification_with_duration(true, overall_start.elapsed().as_secs_f64())
        .await;

    Ok(Json(IncrementalVerifyResponse {
        valid: true,
        cached_count: unchanged_properties.len() + cache_hits.len(),
        verified_count: affected_count,
        affected_properties,
        unchanged_properties,
        compilations,
        errors,
    }))
}
