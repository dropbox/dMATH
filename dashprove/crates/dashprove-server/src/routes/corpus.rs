//! Corpus route handlers for DashProve server
//!
//! Handles proof corpus and counterexample corpus endpoints:
//! - GET /corpus/search - Search the proof corpus
//! - GET /corpus/stats - Get corpus statistics
//! - GET /corpus/history - Get corpus history over time
//! - GET /corpus/compare - Compare two time periods in the corpus
//! - GET /corpus/suggest - Suggest comparison periods based on available data
//! - GET /corpus/counterexamples - List all counterexamples with pagination
//! - GET /corpus/counterexamples/:id - Get a single counterexample by ID
//! - POST /corpus/counterexamples/search - Search similar counterexamples (feature-based)
//! - GET /corpus/counterexamples/text-search - Search counterexamples by text keywords
//! - POST /corpus/counterexamples - Add counterexample to corpus
//! - POST /corpus/counterexamples/classify - Classify counterexample against cluster patterns
//! - POST /corpus/counterexamples/clusters - Record cluster patterns from clustering results

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    Json,
};
use chrono::TimeZone;
use dashprove_backends::{BackendId, CounterexampleClusters, StructuredCounterexample};
use dashprove_learning::{
    suggest_comparison_periods, CounterexampleEntry, HistoryComparison, SimilarProof, TimePeriod,
};
use dashprove_usl::{parse, typecheck};
use std::collections::HashMap;
use std::sync::Arc;

use super::types::{
    CorpusCompareQuery, CorpusCompareResponse, CorpusHistoryQuery, CorpusHistoryResponse,
    CorpusSearchQuery, CorpusSearchResponse, CorpusStatsResponse, CorpusSuggestQuery,
    CorpusSuggestResponse, CorpusType, CounterexampleAddRequest, CounterexampleAddResponse,
    CounterexampleClassifyRequest, CounterexampleClassifyResponse, CounterexampleClustersRequest,
    CounterexampleClustersResponse, CounterexampleEntryResponse, CounterexampleListQuery,
    CounterexampleListResponse, CounterexampleSearchRequest, CounterexampleSearchResponse,
    CounterexampleStatsResponse, CounterexampleTextSearchQuery, ErrorResponse, OutputFormat,
    ProofStatsResponse, SimilarProofResponse, TacticScoreResponse, TacticStatsResponse,
    SUPPORTED_BACKENDS,
};
use super::AppState;

/// GET /corpus/search - Search the proof corpus
pub async fn corpus_search(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CorpusSearchQuery>,
) -> Result<Json<CorpusSearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let learning = state.learning.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Learning system not configured".to_string(),
                details: Some("Start server with --data-dir to enable corpus search".to_string()),
            }),
        )
    })?;

    // Try to parse query as a property, otherwise search by keywords
    let learning_guard = learning.read().await;

    // Try parsing as USL property first, fallback to keyword search
    let results: Vec<SimilarProof> =
        if let Ok(spec) = parse(&format!("theorem query {{ {} }}", query.query)) {
            if let Ok(typed) = typecheck(spec) {
                if let Some(prop) = typed.spec.properties.first() {
                    learning_guard.find_similar(prop, query.k)
                } else {
                    // Valid USL but no properties - try keyword search
                    learning_guard.search_by_keywords(&query.query, query.k)
                }
            } else {
                // Type check failed - try keyword search
                learning_guard.search_by_keywords(&query.query, query.k)
            }
        } else {
            // Not valid USL - use keyword search for plain text queries
            learning_guard.search_by_keywords(&query.query, query.k)
        };

    let total_corpus_size = learning_guard.proof_count();

    let results = results
        .into_iter()
        .map(|r| {
            let property_name = r.property.name();
            SimilarProofResponse {
                proof_id: r.id.to_string(),
                similarity: r.similarity,
                property_name,
                backend: r.backend.into(),
                tactics: r.tactics,
            }
        })
        .collect();

    Ok(Json(CorpusSearchResponse {
        results,
        total_corpus_size,
    }))
}

/// GET /corpus/stats - Get corpus statistics
///
/// Returns statistics about the proof corpus, counterexample corpus, and tactics.
/// This includes counts by backend, cluster patterns, and top tactics by Wilson score.
pub async fn corpus_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<CorpusStatsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let learning = state.learning.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Learning system not configured".to_string(),
                details: Some("Start server with --data-dir to enable corpus stats".to_string()),
            }),
        )
    })?;

    let learning_guard = learning.read().await;

    // Proof statistics
    let proof_total = learning_guard.corpus.len();
    let mut proof_by_backend = HashMap::new();
    for backend in SUPPORTED_BACKENDS {
        let count = learning_guard.corpus.by_backend(backend).len();
        if count > 0 {
            proof_by_backend.insert(format!("{:?}", backend), count);
        }
    }

    // Counterexample statistics
    let cx_total = learning_guard.counterexample_count();
    let cluster_patterns = learning_guard.cluster_pattern_count();
    let mut cx_by_backend = HashMap::new();
    for backend in SUPPORTED_BACKENDS {
        let count = learning_guard.counterexamples.by_backend(backend).len();
        if count > 0 {
            cx_by_backend.insert(format!("{:?}", backend), count);
        }
    }

    // Tactic statistics
    let total_observations = learning_guard.tactics.total_observations();
    let unique_tactics = learning_guard.tactics.unique_tactics();

    // Get top tactics if there are any
    let top_tactics = if unique_tactics > 0 {
        use dashprove_learning::{PropertyFeatures, TacticContext};
        // Create a generic context to query top tactics globally
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
        learning_guard
            .tactics
            .best_for_context(&dummy_context, 10)
            .into_iter()
            .map(|(name, score)| TacticScoreResponse { name, score })
            .collect()
    } else {
        vec![]
    };

    Ok(Json(CorpusStatsResponse {
        proofs: ProofStatsResponse {
            total: proof_total,
            by_backend: proof_by_backend,
        },
        counterexamples: CounterexampleStatsResponse {
            total: cx_total,
            cluster_patterns,
            by_backend: cx_by_backend,
        },
        tactics: TacticStatsResponse {
            total_observations,
            unique_tactics,
            top_tactics,
        },
    }))
}

/// GET /corpus/history - Get corpus history over time
///
/// Returns corpus history data. Supports both JSON (default) and HTML output formats.
/// HTML output includes interactive Chart.js visualizations.
pub async fn corpus_history(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CorpusHistoryQuery>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let learning = state.learning.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Learning system not configured".to_string(),
                details: Some("Start server with --data-dir to enable corpus history".to_string()),
            }),
        )
    })?;

    let learning_guard = learning.read().await;
    let period: TimePeriod = query.period.into();

    // Parse date filters
    let from_date = query
        .from
        .as_ref()
        .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
        .map(|dt| chrono::Utc.from_utc_datetime(&dt));

    let to_date = query
        .to
        .as_ref()
        .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .map(|d| d.and_hms_opt(23, 59, 59).unwrap())
        .map(|dt| chrono::Utc.from_utc_datetime(&dt));

    let corpus_name = match query.corpus {
        CorpusType::Proofs => "Proofs",
        CorpusType::Counterexamples => "Counterexamples",
    };

    match query.corpus {
        CorpusType::Proofs => {
            let history = if from_date.is_some() || to_date.is_some() {
                learning_guard.proof_history_in_range(period, from_date, to_date)
            } else {
                learning_guard.proof_history(period)
            };

            match query.format {
                OutputFormat::Html => {
                    let title = format!("{} Corpus History", corpus_name);
                    let html = history.to_html(&title);
                    Ok(Html(html).into_response())
                }
                OutputFormat::Json => {
                    let response = CorpusHistoryResponse::from_proof_history(history);
                    Ok(Json(response).into_response())
                }
            }
        }
        CorpusType::Counterexamples => {
            let history = if from_date.is_some() || to_date.is_some() {
                learning_guard
                    .counterexamples
                    .history_in_range(period, from_date, to_date)
            } else {
                learning_guard.counterexamples.history(period)
            };

            match query.format {
                OutputFormat::Html => {
                    let title = format!("{} Corpus History", corpus_name);
                    let html = history.to_html(&title);
                    Ok(Html(html).into_response())
                }
                OutputFormat::Json => {
                    let response = CorpusHistoryResponse::from(history);
                    Ok(Json(response).into_response())
                }
            }
        }
    }
}

/// GET /corpus/compare - Compare two time periods in the corpus
///
/// Returns corpus comparison data. Supports both JSON (default) and HTML output formats.
/// HTML output includes interactive Chart.js visualizations with suggestions.
pub async fn corpus_compare(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CorpusCompareQuery>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let learning = state.learning.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Learning system not configured".to_string(),
                details: Some(
                    "Start server with --data-dir to enable corpus comparison".to_string(),
                ),
            }),
        )
    })?;

    // Parse dates
    let parse_date =
        |s: &str,
         name: &str|
         -> Result<chrono::DateTime<chrono::Utc>, (StatusCode, Json<ErrorResponse>)> {
            chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .map_err(|_| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("Invalid date format for {}", name),
                            details: Some(format!("Expected YYYY-MM-DD, got: {}", s)),
                        }),
                    )
                })
                .map(|d| chrono::Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0).unwrap()))
        };

    let baseline_from = parse_date(&query.baseline_from, "baseline_from")?;
    let baseline_to = parse_date(&query.baseline_to, "baseline_to")?;
    let compare_from = parse_date(&query.compare_from, "compare_from")?;
    let compare_to = parse_date(&query.compare_to, "compare_to")?;

    let learning_guard = learning.read().await;

    let baseline_label = format!("{} to {}", query.baseline_from, query.baseline_to);
    let compare_label = format!("{} to {}", query.compare_from, query.compare_to);

    let comparison = match query.corpus {
        CorpusType::Proofs => {
            let baseline = learning_guard.proof_history_in_range(
                TimePeriod::Day,
                Some(baseline_from),
                Some(baseline_to),
            );
            let comparison_hist = learning_guard.proof_history_in_range(
                TimePeriod::Day,
                Some(compare_from),
                Some(compare_to),
            );
            HistoryComparison::from_proof_histories(
                &baseline,
                &comparison_hist,
                &baseline_label,
                &compare_label,
            )
        }
        CorpusType::Counterexamples => {
            let baseline = learning_guard.counterexamples.history_in_range(
                TimePeriod::Day,
                Some(baseline_from),
                Some(baseline_to),
            );
            let comparison_hist = learning_guard.counterexamples.history_in_range(
                TimePeriod::Day,
                Some(compare_from),
                Some(compare_to),
            );
            HistoryComparison::from_corpus_histories(
                &baseline,
                &comparison_hist,
                &baseline_label,
                &compare_label,
            )
        }
    };

    match query.format {
        OutputFormat::Html => {
            // Get suggestions for additional comparisons
            let (first_recorded, last_recorded) = match query.corpus {
                CorpusType::Proofs => {
                    let history = learning_guard.proof_history(TimePeriod::Day);
                    (history.first_recorded, history.last_recorded)
                }
                CorpusType::Counterexamples => {
                    let history = learning_guard.counterexamples.history(TimePeriod::Day);
                    (history.first_recorded, history.last_recorded)
                }
            };
            let suggestions = suggest_comparison_periods(first_recorded, last_recorded, None);

            let corpus_name = match query.corpus {
                CorpusType::Proofs => "proofs",
                CorpusType::Counterexamples => "counterexamples",
            };
            let data_range = match (first_recorded, last_recorded) {
                (Some(first), Some(last)) => Some((first, last)),
                _ => None,
            };

            let html = comparison.to_html_with_suggestions(&suggestions, corpus_name, data_range);
            Ok(Html(html).into_response())
        }
        OutputFormat::Json => {
            let response: CorpusCompareResponse = comparison.into();
            Ok(Json(response).into_response())
        }
    }
}

/// GET /corpus/suggest - Suggest comparison periods based on available data
pub async fn corpus_suggest(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CorpusSuggestQuery>,
) -> Result<Json<CorpusSuggestResponse>, (StatusCode, Json<ErrorResponse>)> {
    let learning = state.learning.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Learning system not configured".to_string(),
                details: Some(
                    "Start server with --data-dir to enable period suggestions".to_string(),
                ),
            }),
        )
    })?;

    let learning_guard = learning.read().await;

    // Get the date range from the corpus
    let (first_recorded, last_recorded) = match query.corpus {
        CorpusType::Proofs => {
            let history = learning_guard.proof_history(TimePeriod::Day);
            (history.first_recorded, history.last_recorded)
        }
        CorpusType::Counterexamples => {
            let history = learning_guard.counterexamples.history(TimePeriod::Day);
            (history.first_recorded, history.last_recorded)
        }
    };

    let suggestions = suggest_comparison_periods(first_recorded, last_recorded, None);

    Ok(Json(CorpusSuggestResponse {
        suggestions: suggestions.into_iter().map(Into::into).collect(),
    }))
}

/// POST /corpus/counterexamples/search - Search for similar counterexamples
///
/// Takes a counterexample and finds similar ones in the corpus based on
/// feature similarity (witness variables, trace patterns, failed checks).
pub async fn counterexample_search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CounterexampleSearchRequest>,
) -> Result<Json<CounterexampleSearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let learning = state.learning.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Learning system not configured".to_string(),
                details: Some(
                    "Start server with --data-dir to enable counterexample search".to_string(),
                ),
            }),
        )
    })?;

    let learning_guard = learning.read().await;
    let cx: StructuredCounterexample = req.counterexample.into();
    let results = learning_guard.find_similar_counterexamples(&cx, req.k);
    let total_corpus_size = learning_guard.counterexample_count();

    Ok(Json(CounterexampleSearchResponse {
        results: results.into_iter().map(Into::into).collect(),
        total_corpus_size,
    }))
}

/// GET /corpus/counterexamples/text-search - Search counterexamples by text keywords
///
/// Searches the counterexample corpus for entries matching the query terms.
/// Matches against witness variable names, trace variable names, failed check
/// descriptions, and action names. Returns results sorted by relevance score.
pub async fn counterexample_text_search(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CounterexampleTextSearchQuery>,
) -> Result<Json<CounterexampleSearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let learning = state.learning.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Learning system not configured".to_string(),
                details: Some(
                    "Start server with --data-dir to enable counterexample text search".to_string(),
                ),
            }),
        )
    })?;

    let learning_guard = learning.read().await;
    let results = learning_guard.search_counterexamples_by_keywords(&query.query, query.k);
    let total_corpus_size = learning_guard.counterexample_count();

    Ok(Json(CounterexampleSearchResponse {
        results: results.into_iter().map(Into::into).collect(),
        total_corpus_size,
    }))
}

/// POST /corpus/counterexamples - Add a counterexample to the corpus
///
/// Records a counterexample for future similarity searches and classification.
/// Optionally assigns a cluster label for categorization.
pub async fn counterexample_add(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CounterexampleAddRequest>,
) -> Result<Json<CounterexampleAddResponse>, (StatusCode, Json<ErrorResponse>)> {
    let learning = state.learning.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Learning system not configured".to_string(),
                details: Some(
                    "Start server with --data-dir to enable counterexample storage".to_string(),
                ),
            }),
        )
    })?;

    let mut learning_guard = learning.write().await;
    let cx: StructuredCounterexample = req.counterexample.into();
    let backend: BackendId = req.backend.into();
    let id =
        learning_guard.record_counterexample(&req.property_name, backend, cx, req.cluster_label);
    let total_corpus_size = learning_guard.counterexample_count();

    Ok(Json(CounterexampleAddResponse {
        id: id.0,
        total_corpus_size,
    }))
}

/// POST /corpus/counterexamples/classify - Classify a counterexample against cluster patterns
///
/// Compares the counterexample against stored cluster patterns and returns
/// the best matching cluster label and similarity score.
pub async fn counterexample_classify(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CounterexampleClassifyRequest>,
) -> Result<Json<CounterexampleClassifyResponse>, (StatusCode, Json<ErrorResponse>)> {
    let learning = state.learning.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Learning system not configured".to_string(),
                details: Some(
                    "Start server with --data-dir to enable counterexample classification"
                        .to_string(),
                ),
            }),
        )
    })?;

    let learning_guard = learning.read().await;
    let cx: StructuredCounterexample = req.counterexample.into();
    let classification = learning_guard.classify_counterexample(&cx);
    let total_patterns = learning_guard.cluster_pattern_count();

    let (cluster_label, similarity) = match classification {
        Some((label, score)) => (Some(label), Some(score)),
        None => (None, None),
    };

    Ok(Json(CounterexampleClassifyResponse {
        cluster_label,
        similarity,
        total_patterns,
    }))
}

/// POST /corpus/counterexamples/clusters - Record cluster patterns from clustering results
///
/// Stores cluster patterns for future classification. Each pattern includes
/// a representative counterexample's features and a label.
pub async fn counterexample_clusters(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CounterexampleClustersRequest>,
) -> Result<Json<CounterexampleClustersResponse>, (StatusCode, Json<ErrorResponse>)> {
    let learning = state.learning.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Learning system not configured".to_string(),
                details: Some(
                    "Start server with --data-dir to enable cluster pattern storage".to_string(),
                ),
            }),
        )
    })?;

    // Convert request patterns to CounterexampleClusters
    use dashprove_backends::CounterexampleCluster;

    let clusters: Vec<CounterexampleCluster> = req
        .patterns
        .into_iter()
        .map(|p| {
            let representative: StructuredCounterexample = p.representative.into();
            CounterexampleCluster {
                label: p.label,
                members: vec![representative.clone()],
                representative,
                similarity_threshold: req.similarity_threshold,
            }
        })
        .collect();

    let cx_clusters = CounterexampleClusters {
        clusters,
        similarity_threshold: req.similarity_threshold,
    };

    let mut learning_guard = learning.write().await;
    let patterns_before = learning_guard.cluster_pattern_count();
    learning_guard.record_cluster_patterns(&cx_clusters);
    let total_patterns = learning_guard.cluster_pattern_count();
    let patterns_recorded = total_patterns - patterns_before;

    Ok(Json(CounterexampleClustersResponse {
        patterns_recorded,
        total_patterns,
    }))
}

/// GET /corpus/counterexamples - List all counterexamples with pagination
///
/// Returns a paginated list of counterexamples in the corpus. Supports filtering
/// by backend, property name, and optional date range (from/to, YYYY-MM-DD),
/// along with pagination via limit/offset parameters.
pub async fn counterexample_list(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CounterexampleListQuery>,
) -> Result<Json<CounterexampleListResponse>, (StatusCode, Json<ErrorResponse>)> {
    let learning = state.learning.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Learning system not configured".to_string(),
                details: Some(
                    "Start server with --data-dir to enable counterexample listing".to_string(),
                ),
            }),
        )
    })?;

    let learning_guard = learning.read().await;

    let backend_filter: Option<BackendId> = query.backend.map(|b| b.into());
    let property_filter = query.property_name.as_ref().map(|s| s.to_lowercase());

    // Parse optional date filters
    let parse_date = |value: &Option<String>,
                      name: &str,
                      end_of_day: bool|
     -> Result<
        Option<chrono::DateTime<chrono::Utc>>,
        (StatusCode, Json<ErrorResponse>),
    > {
        if let Some(date_str) = value {
            let naive_date =
                chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d").map_err(|_| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("Invalid date format for {}", name),
                            details: Some(format!("Expected YYYY-MM-DD, got: {}", date_str)),
                        }),
                    )
                })?;

            let datetime = if end_of_day {
                naive_date.and_hms_opt(23, 59, 59).unwrap()
            } else {
                naive_date.and_hms_opt(0, 0, 0).unwrap()
            };

            Ok(Some(chrono::Utc.from_utc_datetime(&datetime)))
        } else {
            Ok(None)
        }
    };

    let from_date = parse_date(&query.from, "from", false)?;
    let to_date = parse_date(&query.to, "to", true)?;

    // Apply filters
    let mut filtered: Vec<&CounterexampleEntry> =
        learning_guard.counterexamples.all_entries().collect();

    if let Some(backend) = backend_filter {
        filtered.retain(|e| e.backend == backend);
    }

    if let Some(property_name) = property_filter {
        filtered.retain(|e| e.property_name.to_lowercase().contains(&property_name));
    }

    if let Some(from) = from_date {
        filtered.retain(|e| e.recorded_at >= from);
    }

    if let Some(to) = to_date {
        filtered.retain(|e| e.recorded_at <= to);
    }

    let total = filtered.len();

    let counterexamples: Vec<CounterexampleEntryResponse> = filtered
        .into_iter()
        .skip(query.offset)
        .take(query.limit)
        .map(CounterexampleEntryResponse::from)
        .collect();

    Ok(Json(CounterexampleListResponse {
        counterexamples,
        total,
        offset: query.offset,
        limit: query.limit,
    }))
}

/// GET /corpus/counterexamples/:id - Get a single counterexample by ID
///
/// Returns the full details of a specific counterexample from the corpus.
pub async fn counterexample_get(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<CounterexampleEntryResponse>, (StatusCode, Json<ErrorResponse>)> {
    let learning = state.learning.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Learning system not configured".to_string(),
                details: Some(
                    "Start server with --data-dir to enable counterexample retrieval".to_string(),
                ),
            }),
        )
    })?;

    let learning_guard = learning.read().await;
    let cx_id = dashprove_learning::CounterexampleId(id.clone());
    let entry = learning_guard.get_counterexample(&cx_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Counterexample not found".to_string(),
                details: Some(format!("No counterexample with ID: {}", id)),
            }),
        )
    })?;

    Ok(Json(CounterexampleEntryResponse::from(entry)))
}
