//! DashProve MCP tools implementation
//!
//! Provides tools for AI agents to interact with DashProve verification.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use dashprove_backends::{
    AlloyBackend, BackendId, CoqBackend, DafnyBackend, HealthStatus, IsabelleBackend, KaniBackend,
    Lean4Backend, TlaPlusBackend, VerificationBackend, VerificationStatus,
};
use dashprove_dispatcher::{Dispatcher, DispatcherConfig, SelectionStrategy};

use crate::cache::{CacheKey, SharedVerificationCache, VerificationCache};
use crate::error::McpError;
use crate::protocol::ToolCallResult;

/// JSON Schema for tool input validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    /// Schema type (usually "object")
    #[serde(rename = "type")]
    pub schema_type: String,
    /// Object properties
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, PropertySchema>>,
    /// Required properties
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
}

/// Property schema within JSON Schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySchema {
    /// Property type
    #[serde(rename = "type")]
    pub property_type: String,
    /// Property description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Enum values
    #[serde(rename = "enum", default, skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// Default value
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    /// Array items schema
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<PropertySchema>>,
}

/// MCP tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinition {
    /// Tool name (unique identifier)
    pub name: String,
    /// Human-readable title
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Tool description
    pub description: String,
    /// Input schema (JSON Schema)
    pub input_schema: JsonSchema,
}

/// Tool call parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool name
    pub name: String,
    /// Tool arguments
    pub arguments: serde_json::Value,
}

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Success indicator
    pub success: bool,
    /// Result content
    pub content: serde_json::Value,
    /// Error message if failed
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Trait for implementing MCP tools
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get the tool definition
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with given arguments
    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError>;
}

// ============================================================================
// verify_usl tool
// ============================================================================

/// Verify a USL specification against backends
///
/// Optionally supports caching of verification results to avoid redundant computation.
pub struct VerifyUslTool {
    /// Optional shared cache for verification results
    cache: Option<SharedVerificationCache<VerifyUslResult>>,
}

impl VerifyUslTool {
    /// Create a new verify_usl tool without caching
    pub fn new() -> Self {
        Self { cache: None }
    }

    /// Create a new verify_usl tool with caching enabled
    pub fn with_cache(cache: SharedVerificationCache<VerifyUslResult>) -> Self {
        Self { cache: Some(cache) }
    }
}

impl Default for VerifyUslTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Arguments for verify_usl tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyUslArgs {
    /// USL specification to verify
    pub spec: String,
    /// Verification strategy: auto, single, redundant, all
    #[serde(default = "default_strategy")]
    pub strategy: String,
    /// Specific backends to use (lean4, tlaplus, kani, coq, alloy)
    #[serde(default)]
    pub backends: Vec<String>,
    /// Timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    /// Whether to skip health checks and run even if backends report unavailable
    #[serde(default)]
    pub skip_health_check: bool,
    /// Whether to run typecheck only (no backend execution)
    #[serde(default)]
    pub typecheck_only: bool,
}

fn default_strategy() -> String {
    "auto".to_string()
}

fn default_timeout() -> u64 {
    60
}

/// Result of verify_usl tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyUslResult {
    /// Overall success
    pub success: bool,
    /// Verification results per backend
    pub results: Vec<BackendResult>,
    /// Parse errors if any
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parse_errors: Option<Vec<String>>,
    /// Summary message
    pub summary: String,
    /// Whether this result was served from cache
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,
    /// Age of cached result in seconds (if cache hit)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_age_secs: Option<u64>,
}

/// Individual backend result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendResult {
    /// Backend name
    pub backend: String,
    /// Verification status
    pub status: String,
    /// Properties verified
    #[serde(default)]
    pub properties_verified: Vec<String>,
    /// Properties failed
    #[serde(default)]
    pub properties_failed: Vec<String>,
    /// Error message if any
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Duration in milliseconds
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

#[async_trait]
impl Tool for VerifyUslTool {
    fn definition(&self) -> ToolDefinition {
        let mut properties = HashMap::new();
        properties.insert(
            "spec".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some("USL specification to verify".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        properties.insert(
            "strategy".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some(
                    "Verification strategy: auto (default), single, redundant, all".to_string(),
                ),
                enum_values: Some(vec![
                    "auto".to_string(),
                    "single".to_string(),
                    "redundant".to_string(),
                    "all".to_string(),
                ]),
                default: Some(json!("auto")),
                items: None,
            },
        );
        properties.insert(
            "backends".to_string(),
            PropertySchema {
                property_type: "array".to_string(),
                description: Some(
                    "Specific backends to use: lean4, tlaplus, kani, coq, alloy, isabelle, dafny (optional)"
                        .to_string(),
                ),
                enum_values: None,
                default: Some(json!([])),
                items: Some(Box::new(PropertySchema {
                    property_type: "string".to_string(),
                    description: Some("Backend name".to_string()),
                    enum_values: Some(vec![
                        "lean4".to_string(),
                        "tlaplus".to_string(),
                        "kani".to_string(),
                        "coq".to_string(),
                        "alloy".to_string(),
                        "isabelle".to_string(),
                        "dafny".to_string(),
                    ]),
                    default: None,
                    items: None,
                })),
            },
        );
        properties.insert(
            "timeout".to_string(),
            PropertySchema {
                property_type: "integer".to_string(),
                description: Some("Timeout in seconds (default: 60)".to_string()),
                enum_values: None,
                default: Some(json!(60)),
                items: None,
            },
        );
        properties.insert(
            "skip_health_check".to_string(),
            PropertySchema {
                property_type: "boolean".to_string(),
                description: Some(
                    "Skip backend health checks and attempt verification even if backends \
                     report unavailable (default: false)"
                        .to_string(),
                ),
                enum_values: None,
                default: Some(json!(false)),
                items: None,
            },
        );
        properties.insert(
            "typecheck_only".to_string(),
            PropertySchema {
                property_type: "boolean".to_string(),
                description: Some(
                    "Only parse and typecheck the spec, skip backend verification \
                     (default: false)"
                        .to_string(),
                ),
                enum_values: None,
                default: Some(json!(false)),
                items: None,
            },
        );

        ToolDefinition {
            name: "dashprove.verify_usl".to_string(),
            title: Some("Verify USL Specification".to_string()),
            description: "Verify a USL (Unified Specification Language) specification against \
                         selected verification backends. Returns detailed results including \
                         which properties were verified or failed."
                .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(properties),
                required: vec!["spec".to_string()],
            },
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let args: VerifyUslArgs = serde_json::from_value(arguments)
            .map_err(|e| McpError::InvalidParams(format!("Invalid arguments: {}", e)))?;

        let start_time = Instant::now();

        // Create cache key for lookup
        let cache_key = CacheKey::new(
            &args.spec,
            &args.backends,
            &args.strategy,
            args.typecheck_only,
        );

        // Check cache first (only for non-typecheck-only requests with caching enabled)
        if let Some(ref cache) = self.cache {
            if let Some(mut cached_result) = cache.get(&cache_key).await {
                // Return cached result with cache metadata
                cached_result.cache_hit = Some(true);
                // Note: we can't easily get the exact age, but the entry was valid
                cached_result.cache_age_secs = Some(0); // Will be updated by cache
                tracing::debug!("Cache hit for verification request");
                return ToolCallResult::json(&cached_result)
                    .map_err(|e| McpError::InternalError(e.to_string()));
            }
        }

        // Parse the USL spec
        let spec = match dashprove_usl::parse(&args.spec) {
            Ok(spec) => spec,
            Err(e) => {
                let result = VerifyUslResult {
                    success: false,
                    results: vec![],
                    parse_errors: Some(vec![e.to_string()]),
                    summary: format!("Failed to parse USL specification: {}", e),
                    cache_hit: Some(false),
                    cache_age_secs: None,
                };
                return ToolCallResult::json(&result)
                    .map_err(|e| McpError::InternalError(e.to_string()));
            }
        };

        // Type check the spec
        let typed_spec = match dashprove_usl::typecheck(spec.clone()) {
            Ok(typed) => typed,
            Err(e) => {
                let result = VerifyUslResult {
                    success: false,
                    results: vec![],
                    parse_errors: Some(vec![format!("Type error: {}", e)]),
                    summary: format!("USL specification has type errors: {}", e),
                    cache_hit: Some(false),
                    cache_age_secs: None,
                };
                return ToolCallResult::json(&result)
                    .map_err(|e| McpError::InternalError(e.to_string()));
            }
        };

        // Get property names from spec
        let property_names: Vec<String> = spec.properties.iter().map(|p| p.name()).collect();

        // If typecheck_only, return early with typecheck success
        if args.typecheck_only {
            let result = VerifyUslResult {
                success: true,
                results: vec![BackendResult {
                    backend: "type_checker".to_string(),
                    status: "verified".to_string(),
                    properties_verified: property_names.clone(),
                    properties_failed: vec![],
                    error: None,
                    duration_ms: Some(start_time.elapsed().as_millis() as u64),
                }],
                parse_errors: None,
                summary: format!(
                    "USL specification parsed and type-checked successfully. {} properties defined.",
                    property_names.len()
                ),
                cache_hit: Some(false),
                cache_age_secs: None,
            };
            // Cache typecheck-only results too
            if let Some(ref cache) = self.cache {
                cache.insert(cache_key, result.clone()).await;
            }
            return ToolCallResult::json(&result)
                .map_err(|e| McpError::InternalError(e.to_string()));
        }

        // Configure dispatcher based on strategy
        let config = match args.strategy.as_str() {
            "single" => DispatcherConfig {
                selection_strategy: SelectionStrategy::Single,
                task_timeout: Duration::from_secs(args.timeout),
                check_health: !args.skip_health_check,
                ..Default::default()
            },
            "redundant" => DispatcherConfig {
                selection_strategy: SelectionStrategy::Redundant { min_backends: 2 },
                task_timeout: Duration::from_secs(args.timeout),
                check_health: !args.skip_health_check,
                ..Default::default()
            },
            "all" => DispatcherConfig {
                selection_strategy: SelectionStrategy::All,
                task_timeout: Duration::from_secs(args.timeout),
                check_health: !args.skip_health_check,
                ..Default::default()
            },
            _ => DispatcherConfig {
                // "auto" - use single backend selection
                selection_strategy: SelectionStrategy::Single,
                task_timeout: Duration::from_secs(args.timeout),
                check_health: !args.skip_health_check,
                ..Default::default()
            },
        };

        let mut dispatcher = Dispatcher::new(config);

        // Register backends based on request
        let requested_backends = if args.backends.is_empty() {
            // Default to all core backends if none specified
            vec![
                "lean4".to_string(),
                "tlaplus".to_string(),
                "kani".to_string(),
                "coq".to_string(),
                "alloy".to_string(),
                "isabelle".to_string(),
                "dafny".to_string(),
            ]
        } else {
            args.backends.clone()
        };

        let mut registered = Vec::new();
        for backend_name in &requested_backends {
            match backend_name.to_lowercase().as_str() {
                "lean4" | "lean" => {
                    let backend: Arc<dyn VerificationBackend> = Arc::new(Lean4Backend::new());
                    if args.skip_health_check || is_backend_available(&backend).await {
                        dispatcher.register_backend(backend);
                        registered.push(BackendId::Lean4);
                    }
                }
                "tlaplus" | "tla+" | "tla" => {
                    let backend: Arc<dyn VerificationBackend> = Arc::new(TlaPlusBackend::new());
                    if args.skip_health_check || is_backend_available(&backend).await {
                        dispatcher.register_backend(backend);
                        registered.push(BackendId::TlaPlus);
                    }
                }
                "kani" => {
                    let backend: Arc<dyn VerificationBackend> = Arc::new(KaniBackend::new());
                    if args.skip_health_check || is_backend_available(&backend).await {
                        dispatcher.register_backend(backend);
                        registered.push(BackendId::Kani);
                    }
                }
                "coq" => {
                    let backend: Arc<dyn VerificationBackend> = Arc::new(CoqBackend::new());
                    if args.skip_health_check || is_backend_available(&backend).await {
                        dispatcher.register_backend(backend);
                        registered.push(BackendId::Coq);
                    }
                }
                "alloy" => {
                    let backend: Arc<dyn VerificationBackend> = Arc::new(AlloyBackend::new());
                    if args.skip_health_check || is_backend_available(&backend).await {
                        dispatcher.register_backend(backend);
                        registered.push(BackendId::Alloy);
                    }
                }
                "isabelle" | "isabelle/hol" => {
                    let backend: Arc<dyn VerificationBackend> = Arc::new(IsabelleBackend::new());
                    if args.skip_health_check || is_backend_available(&backend).await {
                        dispatcher.register_backend(backend);
                        registered.push(BackendId::Isabelle);
                    }
                }
                "dafny" => {
                    let backend: Arc<dyn VerificationBackend> = Arc::new(DafnyBackend::new());
                    if args.skip_health_check || is_backend_available(&backend).await {
                        dispatcher.register_backend(backend);
                        registered.push(BackendId::Dafny);
                    }
                }
                _ => {
                    // Unknown backend - skip with warning
                    tracing::warn!("Unknown backend '{}', skipping", backend_name);
                }
            }
        }

        // If no backends registered, return typecheck-only result with warning
        if registered.is_empty() {
            let result = VerifyUslResult {
                success: true,
                results: vec![BackendResult {
                    backend: "type_checker".to_string(),
                    status: "verified".to_string(),
                    properties_verified: property_names.clone(),
                    properties_failed: vec![],
                    error: Some("No verification backends available. Type-check only.".to_string()),
                    duration_ms: Some(start_time.elapsed().as_millis() as u64),
                }],
                parse_errors: None,
                summary: format!(
                    "USL specification type-checked. {} properties defined. \
                     Warning: No backends available for verification.",
                    property_names.len()
                ),
                cache_hit: Some(false),
                cache_age_secs: None,
            };
            // Cache this result (no backends available is a valid cacheable state)
            if let Some(ref cache) = self.cache {
                cache.insert(cache_key, result.clone()).await;
            }
            return ToolCallResult::json(&result)
                .map_err(|e| McpError::InternalError(e.to_string()));
        }

        // Run verification via dispatcher
        let verify_result = dispatcher.verify(&typed_spec).await;

        match verify_result {
            Ok(merged_results) => {
                let mut backend_results = Vec::new();

                // Convert dispatcher results to MCP format
                for prop_result in &merged_results.properties {
                    for br in &prop_result.backend_results {
                        let (status_str, verified, failed) = match &br.status {
                            VerificationStatus::Proven => {
                                ("proven".to_string(), property_names.clone(), vec![])
                            }
                            VerificationStatus::Disproven => {
                                ("disproven".to_string(), vec![], property_names.clone())
                            }
                            VerificationStatus::Unknown { reason } => {
                                (format!("unknown: {}", reason), vec![], vec![])
                            }
                            VerificationStatus::Partial {
                                verified_percentage,
                            } => (
                                format!("partial: {:.1}%", verified_percentage),
                                vec![],
                                vec![],
                            ),
                        };

                        backend_results.push(BackendResult {
                            backend: format!("{:?}", br.backend),
                            status: status_str,
                            properties_verified: verified,
                            properties_failed: failed,
                            error: br.error.clone(),
                            duration_ms: Some(br.time_taken.as_millis() as u64),
                        });
                    }
                }

                let overall_success =
                    merged_results.summary.proven > 0 && merged_results.summary.disproven == 0;

                let result = VerifyUslResult {
                    success: overall_success,
                    results: backend_results,
                    parse_errors: None,
                    summary: format!(
                        "Verified {} properties: {} proven, {} disproven, {} unknown. \
                         Confidence: {:.1}%. Backends: {:?}",
                        merged_results.properties.len(),
                        merged_results.summary.proven,
                        merged_results.summary.disproven,
                        merged_results.summary.unknown,
                        merged_results.summary.overall_confidence * 100.0,
                        registered
                    ),
                    cache_hit: Some(false),
                    cache_age_secs: None,
                };

                // Cache successful verification results
                if let Some(ref cache) = self.cache {
                    cache.insert(cache_key, result.clone()).await;
                }

                ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
            }
            Err(e) => {
                let result = VerifyUslResult {
                    success: false,
                    results: vec![],
                    parse_errors: None,
                    summary: format!("Verification failed: {}", e),
                    cache_hit: Some(false),
                    cache_age_secs: None,
                };
                // Don't cache errors - they may be transient
                ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
            }
        }
    }
}

/// Check if a backend is available (health check)
async fn is_backend_available(backend: &Arc<dyn VerificationBackend>) -> bool {
    use dashprove_backends::HealthStatus;
    matches!(backend.health_check().await, HealthStatus::Healthy)
}

// ============================================================================
// select_backend tool
// ============================================================================

/// Get recommended backends for a property type
pub struct SelectBackendTool;

impl SelectBackendTool {
    /// Create a new select_backend tool
    pub fn new() -> Self {
        Self
    }
}

impl Default for SelectBackendTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Arguments for select_backend tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectBackendArgs {
    /// Property type to verify
    pub property_type: String,
    /// Domain hint
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    /// Maximum number of backends to return
    #[serde(default = "default_max_backends")]
    pub max_backends: usize,
}

fn default_max_backends() -> usize {
    5
}

/// Result of select_backend tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectBackendResult {
    /// Recommended backends
    pub recommendations: Vec<BackendRecommendation>,
    /// Reasoning
    pub reasoning: String,
}

/// Individual backend recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendRecommendation {
    /// Backend ID
    pub backend: String,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Why this backend was recommended
    pub reason: String,
    /// Known strengths
    pub strengths: Vec<String>,
    /// Known limitations
    pub limitations: Vec<String>,
}

#[async_trait]
impl Tool for SelectBackendTool {
    fn definition(&self) -> ToolDefinition {
        let mut properties = HashMap::new();
        properties.insert(
            "property_type".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some(
                    "Property type: safety, liveness, memory, termination, concurrency, \
                     functional, security, temporal, refinement"
                        .to_string(),
                ),
                enum_values: Some(vec![
                    "safety".to_string(),
                    "liveness".to_string(),
                    "memory".to_string(),
                    "termination".to_string(),
                    "concurrency".to_string(),
                    "functional".to_string(),
                    "security".to_string(),
                    "temporal".to_string(),
                    "refinement".to_string(),
                ]),
                default: None,
                items: None,
            },
        );
        properties.insert(
            "domain".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some(
                    "Domain hint: rust, distributed_systems, protocols, algorithms".to_string(),
                ),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        properties.insert(
            "max_backends".to_string(),
            PropertySchema {
                property_type: "integer".to_string(),
                description: Some("Maximum backends to recommend (default: 5)".to_string()),
                enum_values: None,
                default: Some(json!(5)),
                items: None,
            },
        );

        ToolDefinition {
            name: "dashprove.select_backend".to_string(),
            title: Some("Select Verification Backend".to_string()),
            description: "Get recommended verification backends for a given property type. \
                         Returns ranked backends with confidence scores and reasoning."
                .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(properties),
                required: vec!["property_type".to_string()],
            },
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let args: SelectBackendArgs = serde_json::from_value(arguments)
            .map_err(|e| McpError::InvalidParams(format!("Invalid arguments: {}", e)))?;

        // Backend recommendations based on property type
        let recommendations = match args.property_type.as_str() {
            "safety" => vec![
                BackendRecommendation {
                    backend: "tlaplus".to_string(),
                    confidence: 0.95,
                    reason: "TLA+ excels at safety property verification via model checking"
                        .to_string(),
                    strengths: vec![
                        "Exhaustive state exploration".to_string(),
                        "Temporal logic support".to_string(),
                    ],
                    limitations: vec!["State explosion for large systems".to_string()],
                },
                BackendRecommendation {
                    backend: "dafny".to_string(),
                    confidence: 0.92,
                    reason: "Dafny provides contract-based safety verification".to_string(),
                    strengths: vec![
                        "Pre/post conditions".to_string(),
                        "Invariant checking".to_string(),
                    ],
                    limitations: vec!["Dafny-specific syntax".to_string()],
                },
                BackendRecommendation {
                    backend: "kani".to_string(),
                    confidence: 0.90,
                    reason: "Kani provides bounded model checking for Rust safety properties"
                        .to_string(),
                    strengths: vec![
                        "Native Rust verification".to_string(),
                        "Memory safety proofs".to_string(),
                    ],
                    limitations: vec!["Bounded verification only".to_string()],
                },
            ],
            "liveness" => vec![
                BackendRecommendation {
                    backend: "tlaplus".to_string(),
                    confidence: 0.95,
                    reason: "TLA+ supports liveness with fairness constraints".to_string(),
                    strengths: vec![
                        "Fairness specification".to_string(),
                        "Temporal logic".to_string(),
                    ],
                    limitations: vec!["Requires explicit fairness modeling".to_string()],
                },
                BackendRecommendation {
                    backend: "spin".to_string(),
                    confidence: 0.85,
                    reason: "SPIN provides efficient LTL model checking".to_string(),
                    strengths: vec![
                        "On-the-fly verification".to_string(),
                        "LTL support".to_string(),
                    ],
                    limitations: vec!["Promela syntax learning curve".to_string()],
                },
            ],
            "memory" => vec![
                BackendRecommendation {
                    backend: "kani".to_string(),
                    confidence: 0.95,
                    reason: "Kani specializes in Rust memory safety verification".to_string(),
                    strengths: vec![
                        "Undefined behavior detection".to_string(),
                        "Pointer analysis".to_string(),
                    ],
                    limitations: vec!["Rust-specific".to_string()],
                },
                BackendRecommendation {
                    backend: "miri".to_string(),
                    confidence: 0.90,
                    reason: "Miri detects undefined behavior in Rust".to_string(),
                    strengths: vec![
                        "Precise UB detection".to_string(),
                        "Stacked borrows".to_string(),
                    ],
                    limitations: vec!["Interpretation overhead".to_string()],
                },
            ],
            "termination" => vec![
                BackendRecommendation {
                    backend: "lean4".to_string(),
                    confidence: 0.90,
                    reason: "Lean 4 proves termination via well-founded recursion".to_string(),
                    strengths: vec![
                        "Dependent types".to_string(),
                        "Built-in termination checker".to_string(),
                    ],
                    limitations: vec!["Requires proof annotations".to_string()],
                },
                BackendRecommendation {
                    backend: "isabelle".to_string(),
                    confidence: 0.88,
                    reason: "Isabelle/HOL provides strong termination proofs".to_string(),
                    strengths: vec![
                        "Structured proofs".to_string(),
                        "HOL logic foundation".to_string(),
                    ],
                    limitations: vec!["Interactive proof style".to_string()],
                },
                BackendRecommendation {
                    backend: "coq".to_string(),
                    confidence: 0.85,
                    reason: "Coq provides strong termination guarantees".to_string(),
                    strengths: vec![
                        "Structural recursion".to_string(),
                        "Well-founded induction".to_string(),
                    ],
                    limitations: vec!["Steep learning curve".to_string()],
                },
            ],
            "concurrency" => vec![
                BackendRecommendation {
                    backend: "tlaplus".to_string(),
                    confidence: 0.95,
                    reason: "TLA+ is designed for concurrent system verification".to_string(),
                    strengths: vec![
                        "Interleaving semantics".to_string(),
                        "Race detection".to_string(),
                    ],
                    limitations: vec!["Abstraction required".to_string()],
                },
                BackendRecommendation {
                    backend: "spin".to_string(),
                    confidence: 0.90,
                    reason: "SPIN efficiently verifies concurrent protocols".to_string(),
                    strengths: vec![
                        "Process algebra".to_string(),
                        "Deadlock detection".to_string(),
                    ],
                    limitations: vec!["Finite state required".to_string()],
                },
            ],
            "functional" => vec![
                BackendRecommendation {
                    backend: "lean4".to_string(),
                    confidence: 0.95,
                    reason: "Lean 4 excels at functional correctness proofs".to_string(),
                    strengths: vec![
                        "Dependent types".to_string(),
                        "Tactic automation".to_string(),
                    ],
                    limitations: vec!["Proof effort required".to_string()],
                },
                BackendRecommendation {
                    backend: "dafny".to_string(),
                    confidence: 0.92,
                    reason: "Dafny provides automated functional correctness verification"
                        .to_string(),
                    strengths: vec![
                        "Pre/post conditions".to_string(),
                        "Automated reasoning".to_string(),
                    ],
                    limitations: vec!["Dafny-specific syntax".to_string()],
                },
                BackendRecommendation {
                    backend: "isabelle".to_string(),
                    confidence: 0.88,
                    reason: "Isabelle/HOL provides rigorous functional proofs".to_string(),
                    strengths: vec![
                        "Large library (AFP)".to_string(),
                        "Structured proofs".to_string(),
                    ],
                    limitations: vec!["Interactive proof style".to_string()],
                },
                BackendRecommendation {
                    backend: "coq".to_string(),
                    confidence: 0.85,
                    reason: "Coq provides verified functional program extraction".to_string(),
                    strengths: vec!["Program extraction".to_string(), "Rich tactics".to_string()],
                    limitations: vec!["Verbose syntax".to_string()],
                },
            ],
            "security" => vec![
                BackendRecommendation {
                    backend: "tamarin".to_string(),
                    confidence: 0.95,
                    reason: "Tamarin specializes in cryptographic protocol verification"
                        .to_string(),
                    strengths: vec![
                        "Symbolic cryptography".to_string(),
                        "Attack finding".to_string(),
                    ],
                    limitations: vec!["Protocol-specific".to_string()],
                },
                BackendRecommendation {
                    backend: "proverif".to_string(),
                    confidence: 0.90,
                    reason: "ProVerif provides automated security protocol analysis".to_string(),
                    strengths: vec![
                        "Automatic verification".to_string(),
                        "Secrecy properties".to_string(),
                    ],
                    limitations: vec!["Approximation-based".to_string()],
                },
            ],
            "temporal" => vec![
                BackendRecommendation {
                    backend: "tlaplus".to_string(),
                    confidence: 0.95,
                    reason: "TLA+ is built on temporal logic".to_string(),
                    strengths: vec![
                        "TLA temporal operators".to_string(),
                        "State machine modeling".to_string(),
                    ],
                    limitations: vec!["Model size limits".to_string()],
                },
                BackendRecommendation {
                    backend: "nusmv".to_string(),
                    confidence: 0.85,
                    reason: "NuSMV provides CTL/LTL model checking".to_string(),
                    strengths: vec!["BDD-based".to_string(), "CTL and LTL".to_string()],
                    limitations: vec!["Symbolic model required".to_string()],
                },
            ],
            "refinement" => vec![
                BackendRecommendation {
                    backend: "tlaplus".to_string(),
                    confidence: 0.95,
                    reason: "TLA+ supports refinement mapping verification".to_string(),
                    strengths: vec![
                        "Refinement mapping proofs".to_string(),
                        "Behavioral substitutability".to_string(),
                        "Stuttering equivalence".to_string(),
                    ],
                    limitations: vec!["Manual refinement mapping required".to_string()],
                },
                BackendRecommendation {
                    backend: "alloy".to_string(),
                    confidence: 0.90,
                    reason: "Alloy provides simulation refinement checking".to_string(),
                    strengths: vec![
                        "Counterexample generation".to_string(),
                        "Relational modeling".to_string(),
                    ],
                    limitations: vec!["Bounded scope".to_string()],
                },
                BackendRecommendation {
                    backend: "isabelle".to_string(),
                    confidence: 0.88,
                    reason: "Isabelle/HOL supports data and process refinement proofs".to_string(),
                    strengths: vec![
                        "Data refinement".to_string(),
                        "IO automata refinement".to_string(),
                        "Archive of Formal Proofs".to_string(),
                    ],
                    limitations: vec!["Interactive proof style".to_string()],
                },
                BackendRecommendation {
                    backend: "lean4".to_string(),
                    confidence: 0.85,
                    reason: "Lean 4 can prove implementation refinement via simulation".to_string(),
                    strengths: vec![
                        "Simulation relations".to_string(),
                        "Bisimulation proofs".to_string(),
                    ],
                    limitations: vec!["Requires manual proof construction".to_string()],
                },
            ],
            _ => vec![BackendRecommendation {
                backend: "lean4".to_string(),
                confidence: 0.70,
                reason: "Lean 4 is a general-purpose theorem prover".to_string(),
                strengths: vec!["Flexible".to_string(), "Expressive".to_string()],
                limitations: vec!["Requires formalization".to_string()],
            }],
        };

        let max = args.max_backends.min(recommendations.len());
        let truncated: Vec<_> = recommendations.into_iter().take(max).collect();

        let result = SelectBackendResult {
            recommendations: truncated,
            reasoning: format!(
                "Selected backends optimized for {} verification{}",
                args.property_type,
                args.domain
                    .map(|d| format!(" in {} domain", d))
                    .unwrap_or_default()
            ),
        };

        ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
    }
}

// ============================================================================
// compile_to tool
// ============================================================================

/// Compile USL to a specific backend
pub struct CompileToTool;

impl CompileToTool {
    /// Create a new compile_to tool
    pub fn new() -> Self {
        Self
    }
}

impl Default for CompileToTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Arguments for compile_to tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileToArgs {
    /// USL specification to compile
    pub spec: String,
    /// Target backend
    pub backend: String,
}

/// Result of compile_to tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileToResult {
    /// Whether compilation succeeded
    pub success: bool,
    /// Compiled output
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Backend language name
    pub target_language: String,
    /// Compilation errors
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub errors: Option<Vec<String>>,
}

#[async_trait]
impl Tool for CompileToTool {
    fn definition(&self) -> ToolDefinition {
        let mut properties = HashMap::new();
        properties.insert(
            "spec".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some("USL specification to compile".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        properties.insert(
            "backend".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some(
                    "Target backend: lean4, tlaplus, kani, coq, alloy, isabelle, dafny".to_string(),
                ),
                enum_values: Some(vec![
                    "lean4".to_string(),
                    "tlaplus".to_string(),
                    "kani".to_string(),
                    "coq".to_string(),
                    "alloy".to_string(),
                    "isabelle".to_string(),
                    "dafny".to_string(),
                ]),
                default: None,
                items: None,
            },
        );

        ToolDefinition {
            name: "dashprove.compile_to".to_string(),
            title: Some("Compile USL to Backend".to_string()),
            description:
                "Compile a USL specification to a specific verification backend language. \
                         Returns the compiled source code ready for the target verifier."
                    .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(properties),
                required: vec!["spec".to_string(), "backend".to_string()],
            },
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let args: CompileToArgs = serde_json::from_value(arguments)
            .map_err(|e| McpError::InvalidParams(format!("Invalid arguments: {}", e)))?;

        // Parse the USL spec
        let spec = match dashprove_usl::parse(&args.spec) {
            Ok(spec) => spec,
            Err(e) => {
                let result = CompileToResult {
                    success: false,
                    output: None,
                    target_language: args.backend.clone(),
                    errors: Some(vec![format!("Parse error: {}", e)]),
                };
                return ToolCallResult::json(&result)
                    .map_err(|e| McpError::InternalError(e.to_string()));
            }
        };

        // Type check
        let typed_spec = match dashprove_usl::typecheck(spec) {
            Ok(typed) => typed,
            Err(e) => {
                let result = CompileToResult {
                    success: false,
                    output: None,
                    target_language: args.backend.clone(),
                    errors: Some(vec![format!("Type error: {}", e)]),
                };
                return ToolCallResult::json(&result)
                    .map_err(|e| McpError::InternalError(e.to_string()));
            }
        };

        // Compile to target backend
        let (output, target_language) = match args.backend.to_lowercase().as_str() {
            "lean4" | "lean" => {
                let compiled = dashprove_usl::compile_to_lean(&typed_spec);
                (compiled.code, "Lean 4".to_string())
            }
            "tlaplus" | "tla+" | "tla" => {
                let compiled = dashprove_usl::compile_to_tlaplus(&typed_spec);
                (compiled.code, "TLA+".to_string())
            }
            "coq" => {
                let compiled = dashprove_usl::compile_to_coq(&typed_spec);
                (compiled.code, "Coq".to_string())
            }
            "alloy" => {
                let compiled = dashprove_usl::compile_to_alloy(&typed_spec);
                (compiled.code, "Alloy".to_string())
            }
            "kani" => {
                // Kani uses Rust with special attributes
                let compiled = dashprove_usl::compile_to_kani(&typed_spec);
                (compiled.code, "Rust (Kani)".to_string())
            }
            "isabelle" | "isabelle/hol" => {
                let compiled = dashprove_usl::compile_to_isabelle(&typed_spec);
                (compiled.code, "Isabelle/HOL".to_string())
            }
            "dafny" => {
                let compiled = dashprove_usl::compile_to_dafny(&typed_spec);
                (compiled.code, "Dafny".to_string())
            }
            other => {
                let result = CompileToResult {
                    success: false,
                    output: None,
                    target_language: other.to_string(),
                    errors: Some(vec![format!(
                        "Unknown backend '{}'. Supported: lean4, tlaplus, kani, coq, alloy, isabelle, dafny",
                        other
                    )]),
                };
                return ToolCallResult::json(&result)
                    .map_err(|e| McpError::InternalError(e.to_string()));
            }
        };

        let result = CompileToResult {
            success: true,
            output: Some(output),
            target_language,
            errors: None,
        };

        ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
    }
}

// ============================================================================
// get_suggestions tool
// ============================================================================

/// Get proof suggestions for failed verification
pub struct GetSuggestionsTool;

impl GetSuggestionsTool {
    /// Create a new get_suggestions tool
    pub fn new() -> Self {
        Self
    }
}

impl Default for GetSuggestionsTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Arguments for get_suggestions tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetSuggestionsArgs {
    /// USL specification that failed verification
    pub spec: String,
    /// Backend that was used
    pub backend: String,
    /// Error message from verification
    pub error_message: String,
    /// Failed property name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub property_name: Option<String>,
}

/// Result of get_suggestions tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetSuggestionsResult {
    /// List of suggestions
    pub suggestions: Vec<Suggestion>,
    /// Analysis of the error
    pub analysis: String,
}

/// Individual suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    /// Suggestion type
    pub suggestion_type: String,
    /// Description
    pub description: String,
    /// Example code if applicable
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub example: Option<String>,
    /// Confidence score (0-1)
    pub confidence: f64,
}

#[async_trait]
impl Tool for GetSuggestionsTool {
    fn definition(&self) -> ToolDefinition {
        let mut properties = HashMap::new();
        properties.insert(
            "spec".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some("USL specification that failed verification".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        properties.insert(
            "backend".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some("Backend that was used for verification".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        properties.insert(
            "error_message".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some("Error message from verification attempt".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        properties.insert(
            "property_name".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some("Name of the property that failed (optional)".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );

        ToolDefinition {
            name: "dashprove.get_suggestions".to_string(),
            title: Some("Get Proof Suggestions".to_string()),
            description: "Get suggestions for fixing a failed verification attempt. Analyzes the \
                         error message and specification to provide actionable recommendations."
                .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(properties),
                required: vec![
                    "spec".to_string(),
                    "backend".to_string(),
                    "error_message".to_string(),
                ],
            },
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let args: GetSuggestionsArgs = serde_json::from_value(arguments)
            .map_err(|e| McpError::InvalidParams(format!("Invalid arguments: {}", e)))?;

        let error_lower = args.error_message.to_lowercase();
        let mut suggestions = Vec::new();

        // Analyze error patterns
        if error_lower.contains("timeout") || error_lower.contains("time limit") {
            suggestions.push(Suggestion {
                suggestion_type: "complexity".to_string(),
                description: "Verification timed out. Consider simplifying the specification or \
                             adding intermediate lemmas."
                    .to_string(),
                example: Some(
                    "// Add helper lemma\nlemma intermediate_step: P => Q\n\
                             // Then prove main theorem using lemma"
                        .to_string(),
                ),
                confidence: 0.85,
            });
            suggestions.push(Suggestion {
                suggestion_type: "bounds".to_string(),
                description:
                    "Try reducing bounds or using bounded verification for initial exploration."
                        .to_string(),
                example: None,
                confidence: 0.70,
            });
        }

        if error_lower.contains("counterexample") || error_lower.contains("counter-example") {
            suggestions.push(Suggestion {
                suggestion_type: "strengthen_precondition".to_string(),
                description: "A counterexample was found. Consider strengthening preconditions or \
                             adding invariants."
                    .to_string(),
                example: Some("// Add precondition\nrequires x > 0 && x < MAX_VALUE".to_string()),
                confidence: 0.90,
            });
            suggestions.push(Suggestion {
                suggestion_type: "invariant".to_string(),
                description: "Add loop or state invariants to guide the prover.".to_string(),
                example: Some("invariant I: forall x. property(x)".to_string()),
                confidence: 0.80,
            });
        }

        if error_lower.contains("type") || error_lower.contains("mismatch") {
            suggestions.push(Suggestion {
                suggestion_type: "type_annotation".to_string(),
                description:
                    "Type mismatch detected. Check type annotations and ensure consistent types."
                        .to_string(),
                example: None,
                confidence: 0.85,
            });
        }

        if error_lower.contains("unknown identifier") || error_lower.contains("undefined") {
            suggestions.push(Suggestion {
                suggestion_type: "definition".to_string(),
                description: "Undefined identifier. Ensure all referenced names are defined."
                    .to_string(),
                example: Some(
                    "// Define missing function\ndefine missing_fn(x: Int): Int = x + 1"
                        .to_string(),
                ),
                confidence: 0.95,
            });
        }

        if error_lower.contains("termination") || error_lower.contains("decreasing") {
            suggestions.push(Suggestion {
                suggestion_type: "termination".to_string(),
                description: "Termination check failed. Add explicit decreasing measures."
                    .to_string(),
                example: Some("decreases size(x) - size(result)".to_string()),
                confidence: 0.85,
            });
        }

        // Backend-specific suggestions
        match args.backend.to_lowercase().as_str() {
            "lean4" | "lean" => {
                suggestions.push(Suggestion {
                    suggestion_type: "tactic".to_string(),
                    description:
                        "Try using Lean 4 tactics like `simp`, `omega`, or `decide` for automation."
                            .to_string(),
                    example: Some("by simp [add_comm, add_assoc]".to_string()),
                    confidence: 0.70,
                });
            }
            "tlaplus" | "tla+" => {
                suggestions.push(Suggestion {
                    suggestion_type: "model_size".to_string(),
                    description: "Consider reducing model size or using symmetry reduction."
                        .to_string(),
                    example: None,
                    confidence: 0.70,
                });
            }
            "kani" => {
                suggestions.push(Suggestion {
                    suggestion_type: "unwinding".to_string(),
                    description:
                        "Try adjusting loop unwinding bounds with #[kani::unwind(N)] attribute."
                            .to_string(),
                    example: Some("#[kani::unwind(10)]".to_string()),
                    confidence: 0.75,
                });
            }
            _ => {}
        }

        // Default suggestion
        if suggestions.is_empty() {
            suggestions.push(Suggestion {
                suggestion_type: "general".to_string(),
                description: "Review the specification for logical errors or missing assumptions."
                    .to_string(),
                example: None,
                confidence: 0.50,
            });
        }

        let result = GetSuggestionsResult {
            suggestions,
            analysis: format!(
                "Analyzed error from {} backend{}: {}",
                args.backend,
                args.property_name
                    .map(|p| format!(" for property '{}'", p))
                    .unwrap_or_default(),
                if error_lower.len() > 100 {
                    format!("{}...", &args.error_message[..100])
                } else {
                    args.error_message.clone()
                }
            ),
        };

        ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
    }
}

// ============================================================================
// check_dependencies tool
// ============================================================================

/// Check backend dependencies and availability
pub struct CheckDependenciesTool;

impl CheckDependenciesTool {
    /// Create a new check_dependencies tool
    pub fn new() -> Self {
        Self
    }
}

impl Default for CheckDependenciesTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Arguments for check_dependencies tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckDependenciesArgs {
    /// Backends to check (default: core backends)
    #[serde(default)]
    pub backends: Vec<String>,
}

/// Dependency status for a backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendDependencyStatus {
    /// Backend name
    pub backend: String,
    /// Whether backend is available
    pub available: bool,
    /// Health status (healthy/degraded/unavailable/unknown_backend)
    pub status: String,
    /// Reason for degraded/unavailable status
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Installation hint
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install_hint: Option<String>,
    /// Expected dependencies or tools
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<String>,
}

/// Result payload for check_dependencies tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckDependenciesResult {
    /// Summary string
    pub summary: String,
    /// Backend statuses
    pub backends: Vec<BackendDependencyStatus>,
}

/// Default backends to check when none specified
fn default_dependency_backends() -> Vec<String> {
    vec![
        "lean4".to_string(),
        "tlaplus".to_string(),
        "kani".to_string(),
        "coq".to_string(),
        "alloy".to_string(),
        "isabelle".to_string(),
        "dafny".to_string(),
    ]
}

/// Map health status to tuple (available, status string, reason)
fn describe_health(status: &HealthStatus) -> (bool, String, Option<String>) {
    match status {
        HealthStatus::Healthy => (true, "healthy".to_string(), None),
        HealthStatus::Degraded { reason } => (true, "degraded".to_string(), Some(reason.clone())),
        HealthStatus::Unavailable { reason } => {
            (false, "unavailable".to_string(), Some(reason.clone()))
        }
    }
}

#[async_trait]
impl Tool for CheckDependenciesTool {
    fn definition(&self) -> ToolDefinition {
        let mut properties = HashMap::new();
        properties.insert(
            "backends".to_string(),
            PropertySchema {
                property_type: "array".to_string(),
                description: Some(
                    "Backends to check (default: lean4, tlaplus, kani, coq, alloy, isabelle, dafny)"
                        .to_string(),
                ),
                enum_values: None,
                default: Some(json!([])),
                items: Some(Box::new(PropertySchema {
                    property_type: "string".to_string(),
                    description: Some("Backend name".to_string()),
                    enum_values: Some(vec![
                        "lean4".to_string(),
                        "tlaplus".to_string(),
                        "kani".to_string(),
                        "coq".to_string(),
                        "alloy".to_string(),
                        "isabelle".to_string(),
                        "dafny".to_string(),
                    ]),
                    default: None,
                    items: None,
                })),
            },
        );

        ToolDefinition {
            name: "dashprove.check_dependencies".to_string(),
            title: Some("Check Backend Dependencies".to_string()),
            description: "Check availability of DashProve verification backends and provide \
                         installation hints for missing dependencies."
                .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(properties),
                required: vec![],
            },
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let args: CheckDependenciesArgs = serde_json::from_value(arguments)
            .map_err(|e| McpError::InvalidParams(format!("Invalid arguments: {}", e)))?;

        let target_backends = if args.backends.is_empty() {
            default_dependency_backends()
        } else {
            args.backends.clone()
        };

        let mut statuses = Vec::new();

        for backend_name in target_backends {
            let backend_key = backend_name.to_lowercase();
            let (backend, install_hint, dependencies): (
                Option<Arc<dyn VerificationBackend>>,
                Option<&'static str>,
                Vec<String>,
            ) = match backend_key.as_str() {
                "lean4" | "lean" => (
                    Some(Arc::new(Lean4Backend::new())),
                    Some("Install Lean 4 toolchain via elan and ensure `lake` is on PATH"),
                    vec!["elan".to_string(), "lake".to_string(), "mathlib".to_string()],
                ),
                "tlaplus" | "tla+" | "tla" => (
                    Some(Arc::new(TlaPlusBackend::new())),
                    Some("Install TLA+ tools (tlc) or tla2tools.jar with Java on PATH"),
                    vec!["tlc".to_string(), "java".to_string(), "tla2tools.jar".to_string()],
                ),
                "kani" => (
                    Some(Arc::new(KaniBackend::new())),
                    Some("Install Kani via `cargo install kani-verifier` and ensure cargo-kani is available"),
                    vec!["cargo-kani".to_string(), "rustc".to_string()],
                ),
                "coq" => (
                    Some(Arc::new(CoqBackend::new())),
                    Some("Install Coq (`coqc`) via opam or system package manager"),
                    vec!["coqc".to_string(), "coqtop".to_string()],
                ),
                "alloy" => (
                    Some(Arc::new(AlloyBackend::new())),
                    Some("Download Alloy JAR and ensure `alloy` launcher or java -jar path is available"),
                    vec!["java".to_string(), "alloy.jar".to_string()],
                ),
                "isabelle" | "isabelle/hol" => (
                    Some(Arc::new(IsabelleBackend::new())),
                    Some("Install Isabelle from isabelle.in.tum.de and ensure `isabelle` is on PATH"),
                    vec!["isabelle".to_string(), "isabelle_process".to_string()],
                ),
                "dafny" => (
                    Some(Arc::new(DafnyBackend::new())),
                    Some("Install Dafny from github.com/dafny-lang/dafny and ensure `dafny` is on PATH"),
                    vec!["dafny".to_string(), "dotnet".to_string()],
                ),
                _ => (None, None, vec![]),
            };

            if let Some(backend) = backend {
                let health = backend.health_check().await;
                let (available, status, reason) = describe_health(&health);

                statuses.push(BackendDependencyStatus {
                    backend: backend_key,
                    available,
                    status,
                    reason,
                    install_hint: install_hint.map(|s| s.to_string()),
                    dependencies,
                });
            } else {
                statuses.push(BackendDependencyStatus {
                    backend: backend_key,
                    available: false,
                    status: "unknown_backend".to_string(),
                    reason: Some("Unsupported backend".to_string()),
                    install_hint: None,
                    dependencies,
                });
            }
        }

        let healthy = statuses.iter().filter(|s| s.status == "healthy").count();
        let degraded = statuses.iter().filter(|s| s.status == "degraded").count();
        let unavailable = statuses.len().saturating_sub(healthy + degraded);
        let summary = format!(
            "{} healthy, {} degraded, {} unavailable ({} total)",
            healthy,
            degraded,
            unavailable,
            statuses.len()
        );

        let result = CheckDependenciesResult {
            summary,
            backends: statuses,
        };

        ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
    }
}

// ============================================================================
// verify_usl_streaming tool
// ============================================================================

use crate::streaming::{
    run_streaming_verification, CancelSessionResult, SessionManager, StreamingVerifyArgs,
    StreamingVerifyStartResult,
};

/// Start a streaming verification session
///
/// This tool initiates verification in the background and returns a session ID.
/// Clients should subscribe to the SSE endpoint `/events/{session_id}` to receive
/// real-time progress updates.
pub struct VerifyUslStreamingTool {
    session_manager: Arc<SessionManager>,
}

impl VerifyUslStreamingTool {
    /// Create a new streaming verify tool with a session manager
    pub fn new(session_manager: Arc<SessionManager>) -> Self {
        Self { session_manager }
    }
}

#[async_trait]
impl Tool for VerifyUslStreamingTool {
    fn definition(&self) -> ToolDefinition {
        let mut properties = HashMap::new();
        properties.insert(
            "spec".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some("USL specification to verify".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        properties.insert(
            "strategy".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some(
                    "Verification strategy: auto (default), single, redundant, all".to_string(),
                ),
                enum_values: Some(vec![
                    "auto".to_string(),
                    "single".to_string(),
                    "redundant".to_string(),
                    "all".to_string(),
                ]),
                default: Some(json!("auto")),
                items: None,
            },
        );
        properties.insert(
            "backends".to_string(),
            PropertySchema {
                property_type: "array".to_string(),
                description: Some(
                    "Specific backends to use: lean4, tlaplus, kani, coq, alloy, isabelle, dafny (optional)"
                        .to_string(),
                ),
                enum_values: None,
                default: Some(json!([])),
                items: Some(Box::new(PropertySchema {
                    property_type: "string".to_string(),
                    description: Some("Backend name".to_string()),
                    enum_values: Some(vec![
                        "lean4".to_string(),
                        "tlaplus".to_string(),
                        "kani".to_string(),
                        "coq".to_string(),
                        "alloy".to_string(),
                        "isabelle".to_string(),
                        "dafny".to_string(),
                    ]),
                    default: None,
                    items: None,
                })),
            },
        );
        properties.insert(
            "timeout".to_string(),
            PropertySchema {
                property_type: "integer".to_string(),
                description: Some("Timeout in seconds (default: 60)".to_string()),
                enum_values: None,
                default: Some(json!(60)),
                items: None,
            },
        );
        properties.insert(
            "skip_health_check".to_string(),
            PropertySchema {
                property_type: "boolean".to_string(),
                description: Some(
                    "Skip backend health checks and attempt verification even if backends \
                     report unavailable (default: false)"
                        .to_string(),
                ),
                enum_values: None,
                default: Some(json!(false)),
                items: None,
            },
        );

        ToolDefinition {
            name: "dashprove.verify_usl_streaming".to_string(),
            title: Some("Verify USL Specification (Streaming)".to_string()),
            description: "Start a streaming verification session for a USL specification. \
                         Returns a session ID immediately. Subscribe to /events/{session_id} \
                         (SSE) to receive real-time progress updates as verification proceeds."
                .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(properties),
                required: vec!["spec".to_string()],
            },
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let args: StreamingVerifyArgs = serde_json::from_value(arguments)
            .map_err(|e| McpError::InvalidParams(format!("Invalid arguments: {}", e)))?;

        // Create a new session
        let session = self.session_manager.create_session().await;
        let session_id = {
            let s = session.lock().await;
            s.id.clone()
        };

        // Start verification in background
        let session_for_task = session.clone();
        tokio::spawn(async move {
            run_streaming_verification(session_for_task, args).await;
        });

        // Return session ID immediately
        let result = StreamingVerifyStartResult {
            session_id: session_id.clone(),
            events_url: format!("/events/{}", session_id),
            message: "Verification started. Subscribe to the events URL for progress updates."
                .to_string(),
        };

        ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
    }
}

// ============================================================================
// get_session_status tool
// ============================================================================

/// Get the status of a streaming verification session
///
/// This tool queries the status of an ongoing or completed verification session.
/// Use this for polling-based clients or to check status without maintaining
/// an SSE connection.
pub struct GetSessionStatusTool {
    session_manager: Arc<SessionManager>,
}

impl GetSessionStatusTool {
    /// Create a new get_session_status tool with a session manager
    pub fn new(session_manager: Arc<SessionManager>) -> Self {
        Self { session_manager }
    }
}

/// Arguments for get_session_status tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetSessionStatusArgs {
    /// Session ID to query
    pub session_id: String,
}

#[async_trait]
impl Tool for GetSessionStatusTool {
    fn definition(&self) -> ToolDefinition {
        let mut properties = HashMap::new();
        properties.insert(
            "session_id".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some("Session ID returned from verify_usl_streaming tool".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );

        ToolDefinition {
            name: "get_session_status".to_string(),
            title: Some("Get Session Status".to_string()),
            description: "Query the status of a streaming verification session. Returns session state, completion status, and final results if available. Use this for polling-based status checks or to retrieve results after reconnection.".to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(properties),
                required: vec!["session_id".to_string()],
            },
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let args: GetSessionStatusArgs = serde_json::from_value(arguments)
            .map_err(|e| McpError::InvalidParams(format!("Invalid arguments: {}", e)))?;

        let status = self
            .session_manager
            .get_session_status(&args.session_id)
            .await;

        ToolCallResult::json(&status).map_err(|e| McpError::InternalError(e.to_string()))
    }
}

// ============================================================================
// cancel_session tool
// ============================================================================

/// Cancel a running streaming verification session
///
/// This tool cancels an ongoing verification session. Use this when you need to
/// stop a long-running verification that is no longer needed.
pub struct CancelSessionTool {
    session_manager: Arc<SessionManager>,
}

impl CancelSessionTool {
    /// Create a new cancel_session tool with a session manager
    pub fn new(session_manager: Arc<SessionManager>) -> Self {
        Self { session_manager }
    }
}

/// Arguments for cancel_session tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelSessionArgs {
    /// Session ID to cancel
    pub session_id: String,
}

#[async_trait]
impl Tool for CancelSessionTool {
    fn definition(&self) -> ToolDefinition {
        let mut properties = HashMap::new();
        properties.insert(
            "session_id".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some(
                    "Session ID returned from verify_usl_streaming tool to cancel".to_string(),
                ),
                enum_values: None,
                default: None,
                items: None,
            },
        );

        ToolDefinition {
            name: "cancel_session".to_string(),
            title: Some("Cancel Session".to_string()),
            description: "Cancel a running streaming verification session. Use this to stop \
                         long-running verifications that are no longer needed. Returns success \
                         if the session was cancelled, or an error if the session was already \
                         completed or not found."
                .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(properties),
                required: vec!["session_id".to_string()],
            },
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let args: CancelSessionArgs = serde_json::from_value(arguments)
            .map_err(|e| McpError::InvalidParams(format!("Invalid arguments: {}", e)))?;

        let result: CancelSessionResult =
            self.session_manager.cancel_session(&args.session_id).await;

        ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
    }
}

// ============================================================================
// Batch verification tool
// ============================================================================

/// Tool for verifying multiple USL specifications in batch
///
/// Optionally supports caching of verification results to avoid redundant computation.
pub struct BatchVerifyTool {
    /// Optional shared cache for verification results
    cache: Option<SharedVerificationCache<VerifyUslResult>>,
}

impl BatchVerifyTool {
    /// Create a new batch_verify tool without caching
    pub fn new() -> Self {
        Self { cache: None }
    }

    /// Create a new batch_verify tool with caching enabled
    pub fn with_cache(cache: SharedVerificationCache<VerifyUslResult>) -> Self {
        Self { cache: Some(cache) }
    }
}

impl Default for BatchVerifyTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Arguments for batch_verify tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerifyArgs {
    /// List of USL specifications to verify
    pub specs: Vec<BatchSpecItem>,
    /// Shared timeout in seconds per spec (default: 60)
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    /// Verification strategy: auto, single, redundant, all
    #[serde(default = "default_strategy")]
    pub strategy: String,
    /// Skip health checks
    #[serde(default)]
    pub skip_health_check: bool,
    /// Stop on first failure
    #[serde(default)]
    pub fail_fast: bool,
}

/// Individual specification item in batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSpecItem {
    /// Unique ID for this spec (optional)
    #[serde(default)]
    pub id: Option<String>,
    /// USL specification content
    pub spec: String,
    /// Backends to use for this spec (optional, uses defaults if empty)
    #[serde(default)]
    pub backends: Vec<String>,
}

/// Result of batch verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerifyResult {
    /// Overall success (all specs verified successfully)
    pub success: bool,
    /// Total specifications processed
    pub total: usize,
    /// Number of successful verifications
    pub successful: usize,
    /// Number of failed verifications
    pub failed: usize,
    /// Results for each specification
    pub results: Vec<BatchItemResult>,
    /// Summary message
    pub summary: String,
    /// Total duration in milliseconds
    pub duration_ms: u64,
    /// Number of cache hits in this batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_hits: Option<usize>,
}

/// Result for a single specification in batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchItemResult {
    /// ID of the spec (from input or auto-generated)
    pub id: String,
    /// Whether this spec verified successfully
    pub success: bool,
    /// Status message
    pub status: String,
    /// Backend results (if verification ran)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backend_results: Vec<BackendResult>,
    /// Error message if failed
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Whether this result was served from cache
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,
}

#[async_trait]
impl Tool for BatchVerifyTool {
    fn definition(&self) -> ToolDefinition {
        let mut properties = HashMap::new();

        // specs array property
        let mut spec_item_properties = HashMap::new();
        spec_item_properties.insert(
            "id".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some("Unique identifier for this spec (optional)".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        spec_item_properties.insert(
            "spec".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some("USL specification content".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        spec_item_properties.insert(
            "backends".to_string(),
            PropertySchema {
                property_type: "array".to_string(),
                description: Some(
                    "Backends to use for this spec (optional, uses defaults if empty)".to_string(),
                ),
                enum_values: None,
                default: Some(json!([])),
                items: Some(Box::new(PropertySchema {
                    property_type: "string".to_string(),
                    description: None,
                    enum_values: Some(vec![
                        "lean4".to_string(),
                        "tlaplus".to_string(),
                        "kani".to_string(),
                        "coq".to_string(),
                        "alloy".to_string(),
                        "isabelle".to_string(),
                        "dafny".to_string(),
                    ]),
                    default: None,
                    items: None,
                })),
            },
        );

        properties.insert(
            "specs".to_string(),
            PropertySchema {
                property_type: "array".to_string(),
                description: Some("List of USL specifications to verify".to_string()),
                enum_values: None,
                default: None,
                items: Some(Box::new(PropertySchema {
                    property_type: "object".to_string(),
                    description: Some("A specification item".to_string()),
                    enum_values: None,
                    default: None,
                    items: None,
                })),
            },
        );
        properties.insert(
            "timeout".to_string(),
            PropertySchema {
                property_type: "integer".to_string(),
                description: Some("Timeout in seconds per spec (default: 60)".to_string()),
                enum_values: None,
                default: Some(json!(60)),
                items: None,
            },
        );
        properties.insert(
            "strategy".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some(
                    "Verification strategy: auto (default), single, redundant, all".to_string(),
                ),
                enum_values: Some(vec![
                    "auto".to_string(),
                    "single".to_string(),
                    "redundant".to_string(),
                    "all".to_string(),
                ]),
                default: Some(json!("auto")),
                items: None,
            },
        );
        properties.insert(
            "skip_health_check".to_string(),
            PropertySchema {
                property_type: "boolean".to_string(),
                description: Some("Skip backend health checks (default: false)".to_string()),
                enum_values: None,
                default: Some(json!(false)),
                items: None,
            },
        );
        properties.insert(
            "fail_fast".to_string(),
            PropertySchema {
                property_type: "boolean".to_string(),
                description: Some(
                    "Stop on first failure instead of processing all specs (default: false)"
                        .to_string(),
                ),
                enum_values: None,
                default: Some(json!(false)),
                items: None,
            },
        );

        ToolDefinition {
            name: "dashprove.batch_verify".to_string(),
            title: Some("Batch Verify USL Specifications".to_string()),
            description: "Verify multiple USL specifications in a single batch operation. \
                         Returns results for each specification, with options for fail-fast \
                         behavior and per-spec backend selection."
                .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(properties),
                required: vec!["specs".to_string()],
            },
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let args: BatchVerifyArgs = serde_json::from_value(arguments)
            .map_err(|e| McpError::InvalidParams(format!("Invalid arguments: {}", e)))?;

        let overall_start = Instant::now();
        let mut results = Vec::new();
        let mut successful = 0;
        let mut failed = 0;
        let mut cache_hits = 0usize;

        for (idx, spec_item) in args.specs.iter().enumerate() {
            let spec_id = spec_item
                .id
                .clone()
                .unwrap_or_else(|| format!("spec_{}", idx + 1));
            let spec_start = Instant::now();

            // Create cache key for this spec
            let cache_key = CacheKey::new(
                &spec_item.spec,
                &spec_item.backends,
                &args.strategy,
                false, // batch verify doesn't have typecheck_only option per spec
            );

            // Check cache first
            if let Some(ref cache) = self.cache {
                if let Some(cached_result) = cache.get(&cache_key).await {
                    // Cache hit - convert VerifyUslResult to BatchItemResult
                    cache_hits += 1;
                    if cached_result.success {
                        successful += 1;
                    } else {
                        failed += 1;
                    }
                    results.push(BatchItemResult {
                        id: spec_id,
                        success: cached_result.success,
                        status: if cached_result.success {
                            "verified (cached)".to_string()
                        } else {
                            "failed (cached)".to_string()
                        },
                        backend_results: cached_result.results,
                        error: cached_result.parse_errors.map(|e| e.join("; ")),
                        duration_ms: 0, // Cached results are instant
                        cache_hit: Some(true),
                    });
                    continue;
                }
            }

            // Parse the spec
            let spec = match dashprove_usl::parse(&spec_item.spec) {
                Ok(spec) => spec,
                Err(e) => {
                    failed += 1;
                    results.push(BatchItemResult {
                        id: spec_id,
                        success: false,
                        status: "parse_error".to_string(),
                        backend_results: vec![],
                        error: Some(format!("Parse error: {}", e)),
                        duration_ms: spec_start.elapsed().as_millis() as u64,
                        cache_hit: Some(false),
                    });
                    if args.fail_fast {
                        break;
                    }
                    continue;
                }
            };

            // Type check
            let typed_spec = match dashprove_usl::typecheck(spec.clone()) {
                Ok(typed) => typed,
                Err(e) => {
                    failed += 1;
                    results.push(BatchItemResult {
                        id: spec_id,
                        success: false,
                        status: "type_error".to_string(),
                        backend_results: vec![],
                        error: Some(format!("Type error: {}", e)),
                        duration_ms: spec_start.elapsed().as_millis() as u64,
                        cache_hit: Some(false),
                    });
                    if args.fail_fast {
                        break;
                    }
                    continue;
                }
            };

            // Configure dispatcher
            let config = match args.strategy.as_str() {
                "single" => DispatcherConfig {
                    selection_strategy: SelectionStrategy::Single,
                    task_timeout: Duration::from_secs(args.timeout),
                    check_health: !args.skip_health_check,
                    ..Default::default()
                },
                "redundant" => DispatcherConfig {
                    selection_strategy: SelectionStrategy::Redundant { min_backends: 2 },
                    task_timeout: Duration::from_secs(args.timeout),
                    check_health: !args.skip_health_check,
                    ..Default::default()
                },
                "all" => DispatcherConfig {
                    selection_strategy: SelectionStrategy::All,
                    task_timeout: Duration::from_secs(args.timeout),
                    check_health: !args.skip_health_check,
                    ..Default::default()
                },
                _ => DispatcherConfig {
                    selection_strategy: SelectionStrategy::Single,
                    task_timeout: Duration::from_secs(args.timeout),
                    check_health: !args.skip_health_check,
                    ..Default::default()
                },
            };

            let mut dispatcher = Dispatcher::new(config);

            // Register backends
            let requested_backends = if spec_item.backends.is_empty() {
                vec![
                    "lean4".to_string(),
                    "tlaplus".to_string(),
                    "kani".to_string(),
                    "coq".to_string(),
                    "alloy".to_string(),
                ]
            } else {
                spec_item.backends.clone()
            };

            let mut registered = Vec::new();
            for backend_name in &requested_backends {
                match backend_name.to_lowercase().as_str() {
                    "lean4" | "lean" => {
                        let backend: Arc<dyn VerificationBackend> = Arc::new(Lean4Backend::new());
                        if args.skip_health_check || is_backend_available(&backend).await {
                            dispatcher.register_backend(backend);
                            registered.push("lean4".to_string());
                        }
                    }
                    "tlaplus" | "tla+" | "tla" => {
                        let backend: Arc<dyn VerificationBackend> = Arc::new(TlaPlusBackend::new());
                        if args.skip_health_check || is_backend_available(&backend).await {
                            dispatcher.register_backend(backend);
                            registered.push("tlaplus".to_string());
                        }
                    }
                    "kani" => {
                        let backend: Arc<dyn VerificationBackend> = Arc::new(KaniBackend::new());
                        if args.skip_health_check || is_backend_available(&backend).await {
                            dispatcher.register_backend(backend);
                            registered.push("kani".to_string());
                        }
                    }
                    "coq" => {
                        let backend: Arc<dyn VerificationBackend> = Arc::new(CoqBackend::new());
                        if args.skip_health_check || is_backend_available(&backend).await {
                            dispatcher.register_backend(backend);
                            registered.push("coq".to_string());
                        }
                    }
                    "alloy" => {
                        let backend: Arc<dyn VerificationBackend> = Arc::new(AlloyBackend::new());
                        if args.skip_health_check || is_backend_available(&backend).await {
                            dispatcher.register_backend(backend);
                            registered.push("alloy".to_string());
                        }
                    }
                    "isabelle" | "isabelle/hol" => {
                        let backend: Arc<dyn VerificationBackend> =
                            Arc::new(IsabelleBackend::new());
                        if args.skip_health_check || is_backend_available(&backend).await {
                            dispatcher.register_backend(backend);
                            registered.push("isabelle".to_string());
                        }
                    }
                    "dafny" => {
                        let backend: Arc<dyn VerificationBackend> = Arc::new(DafnyBackend::new());
                        if args.skip_health_check || is_backend_available(&backend).await {
                            dispatcher.register_backend(backend);
                            registered.push("dafny".to_string());
                        }
                    }
                    _ => {} // Skip unknown backends
                }
            }

            if registered.is_empty() {
                // Type-check only success
                successful += 1;

                // Create result for caching
                let verify_result = VerifyUslResult {
                    success: true,
                    results: vec![],
                    parse_errors: None,
                    summary: "Type-check only, no backends registered".to_string(),
                    cache_hit: Some(false),
                    cache_age_secs: None,
                };

                // Store in cache
                if let Some(ref cache) = self.cache {
                    cache.insert(cache_key, verify_result).await;
                }

                results.push(BatchItemResult {
                    id: spec_id,
                    success: true,
                    status: "typecheck_only".to_string(),
                    backend_results: vec![],
                    error: None,
                    duration_ms: spec_start.elapsed().as_millis() as u64,
                    cache_hit: Some(false),
                });
                continue;
            }

            // Run verification
            match dispatcher.verify(&typed_spec).await {
                Ok(merged_results) => {
                    let property_names: Vec<String> =
                        spec.properties.iter().map(|p| p.name()).collect();

                    let mut backend_results = Vec::new();
                    for prop_result in &merged_results.properties {
                        for br in &prop_result.backend_results {
                            let (status_str, verified, failed_props) = match &br.status {
                                VerificationStatus::Proven => {
                                    ("proven".to_string(), property_names.clone(), vec![])
                                }
                                VerificationStatus::Disproven => {
                                    ("disproven".to_string(), vec![], property_names.clone())
                                }
                                VerificationStatus::Unknown { reason } => {
                                    (format!("unknown: {}", reason), vec![], vec![])
                                }
                                VerificationStatus::Partial {
                                    verified_percentage,
                                } => (
                                    format!("partial: {:.1}%", verified_percentage),
                                    vec![],
                                    vec![],
                                ),
                            };

                            backend_results.push(BackendResult {
                                backend: format!("{:?}", br.backend),
                                status: status_str,
                                properties_verified: verified,
                                properties_failed: failed_props,
                                error: br.error.clone(),
                                duration_ms: Some(br.time_taken.as_millis() as u64),
                            });
                        }
                    }

                    let spec_success =
                        merged_results.summary.proven > 0 && merged_results.summary.disproven == 0;

                    if spec_success {
                        successful += 1;
                    } else {
                        failed += 1;
                    }

                    // Create result for caching
                    let verify_result = VerifyUslResult {
                        success: spec_success,
                        results: backend_results.clone(),
                        parse_errors: None,
                        summary: format!(
                            "Verified {} properties: {} proven, {} disproven, {} unknown",
                            merged_results.properties.len(),
                            merged_results.summary.proven,
                            merged_results.summary.disproven,
                            merged_results.summary.unknown
                        ),
                        cache_hit: Some(false),
                        cache_age_secs: None,
                    };

                    // Store in cache
                    if let Some(ref cache) = self.cache {
                        cache.insert(cache_key, verify_result).await;
                    }

                    results.push(BatchItemResult {
                        id: spec_id,
                        success: spec_success,
                        status: if spec_success {
                            "verified".to_string()
                        } else {
                            "verification_failed".to_string()
                        },
                        backend_results,
                        error: None,
                        duration_ms: spec_start.elapsed().as_millis() as u64,
                        cache_hit: Some(false),
                    });

                    if !spec_success && args.fail_fast {
                        break;
                    }
                }
                Err(e) => {
                    failed += 1;
                    results.push(BatchItemResult {
                        id: spec_id,
                        success: false,
                        status: "error".to_string(),
                        backend_results: vec![],
                        error: Some(format!("Verification error: {}", e)),
                        duration_ms: spec_start.elapsed().as_millis() as u64,
                        cache_hit: Some(false),
                    });
                    // Don't cache errors - they may be transient
                    if args.fail_fast {
                        break;
                    }
                }
            }
        }

        let total = args.specs.len();
        let overall_success = failed == 0 && successful > 0;
        let summary = if cache_hits > 0 {
            format!(
                "Batch verification: {}/{} successful, {}/{} failed ({} cache hits)",
                successful, total, failed, total, cache_hits
            )
        } else {
            format!(
                "Batch verification: {}/{} successful, {}/{} failed",
                successful, total, failed, total
            )
        };

        let result = BatchVerifyResult {
            success: overall_success,
            total,
            successful,
            failed,
            results,
            summary,
            duration_ms: overall_start.elapsed().as_millis() as u64,
            cache_hits: if cache_hits > 0 {
                Some(cache_hits)
            } else {
                None
            },
        };

        ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
    }
}

// ============================================================================
// get_cache_stats tool
// ============================================================================

/// Get cache statistics via MCP
pub struct GetCacheStatsTool {
    cache: SharedVerificationCache<VerifyUslResult>,
}

impl GetCacheStatsTool {
    /// Create a new get_cache_stats tool
    pub fn new(cache: SharedVerificationCache<VerifyUslResult>) -> Self {
        Self { cache }
    }
}

/// Result of get_cache_stats tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetCacheStatsResult {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Current number of entries
    pub entries: usize,
    /// Number of entries evicted (LRU)
    pub evictions: u64,
    /// Number of expired entries removed
    pub expirations: u64,
    /// Cache hit rate as percentage
    pub hit_rate: f64,
    /// Whether caching is enabled
    pub enabled: bool,
    /// Time-to-live in seconds
    pub ttl_secs: u64,
    /// Maximum cache entries
    pub max_entries: usize,
}

#[async_trait]
impl Tool for GetCacheStatsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_cache_stats".to_string(),
            title: Some("Get Cache Statistics".to_string()),
            description: "Get verification cache statistics including hits, misses, entries, \
                         hit rate, and configuration. Useful for monitoring cache performance."
                .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(HashMap::new()),
                required: vec![],
            },
        }
    }

    async fn execute(&self, _arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let stats = self.cache.stats().await;
        let config = self.cache.config().await;
        let enabled = self.cache.is_enabled().await;

        let result = GetCacheStatsResult {
            hits: stats.hits,
            misses: stats.misses,
            entries: stats.entries,
            evictions: stats.evictions,
            expirations: stats.expirations,
            hit_rate: stats.hit_rate(),
            enabled,
            ttl_secs: config.ttl.as_secs(),
            max_entries: config.max_entries,
        };

        ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
    }
}

// ============================================================================
// clear_cache tool
// ============================================================================

/// Clear the verification cache via MCP
pub struct ClearCacheTool {
    cache: SharedVerificationCache<VerifyUslResult>,
}

impl ClearCacheTool {
    /// Create a new clear_cache tool
    pub fn new(cache: SharedVerificationCache<VerifyUslResult>) -> Self {
        Self { cache }
    }
}

/// Result of clear_cache tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearCacheResult {
    /// Number of entries cleared
    pub cleared: usize,
    /// Success indicator
    pub success: bool,
    /// Message
    pub message: String,
}

#[async_trait]
impl Tool for ClearCacheTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "clear_cache".to_string(),
            title: Some("Clear Cache".to_string()),
            description: "Clear all cached verification results. Returns the number of entries \
                         that were cleared. Useful for forcing re-verification after code changes."
                .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(HashMap::new()),
                required: vec![],
            },
        }
    }

    async fn execute(&self, _arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let count_before = self.cache.len().await;
        self.cache.clear().await;

        let result = ClearCacheResult {
            cleared: count_before,
            success: true,
            message: format!("Cleared {} cached entries", count_before),
        };

        ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
    }
}

// ============================================================================
// configure_cache tool
// ============================================================================

/// Configure the verification cache at runtime via MCP
pub struct ConfigureCacheTool {
    cache: SharedVerificationCache<VerifyUslResult>,
}

impl ConfigureCacheTool {
    /// Create a new configure_cache tool
    pub fn new(cache: SharedVerificationCache<VerifyUslResult>) -> Self {
        Self { cache }
    }
}

/// Arguments for configure_cache tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigureCacheArgs {
    /// Time-to-live in seconds (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_secs: Option<u64>,
    /// Maximum number of entries (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_entries: Option<usize>,
    /// Enable or disable caching (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Whether to clear cache if config changes significantly (default: false)
    #[serde(default)]
    pub clear_on_change: bool,
}

/// Result of configure_cache tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigureCacheResult {
    /// Success indicator
    pub success: bool,
    /// Old configuration
    pub old_config: CacheConfigInfo,
    /// New configuration
    pub new_config: CacheConfigInfo,
    /// Message
    pub message: String,
}

/// Cache configuration info (for display)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfigInfo {
    /// Time-to-live in seconds
    pub ttl_secs: u64,
    /// Maximum entries
    pub max_entries: usize,
    /// Whether enabled
    pub enabled: bool,
}

#[async_trait]
impl Tool for ConfigureCacheTool {
    fn definition(&self) -> ToolDefinition {
        let mut properties = HashMap::new();
        properties.insert(
            "ttl_secs".to_string(),
            PropertySchema {
                property_type: "integer".to_string(),
                description: Some("Cache entry time-to-live in seconds".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        properties.insert(
            "max_entries".to_string(),
            PropertySchema {
                property_type: "integer".to_string(),
                description: Some("Maximum number of cache entries".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        properties.insert(
            "enabled".to_string(),
            PropertySchema {
                property_type: "boolean".to_string(),
                description: Some("Enable or disable caching".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        properties.insert(
            "clear_on_change".to_string(),
            PropertySchema {
                property_type: "boolean".to_string(),
                description: Some(
                    "Clear cache if TTL decreased, max_entries decreased, or cache disabled"
                        .to_string(),
                ),
                enum_values: None,
                default: Some(serde_json::json!(false)),
                items: None,
            },
        );

        ToolDefinition {
            name: "configure_cache".to_string(),
            title: Some("Configure Cache".to_string()),
            description: "Configure verification cache settings at runtime. Allows adjusting TTL, \
                         max entries, and enabling/disabling the cache. All parameters are optional; \
                         only specified values are updated."
                .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(properties),
                required: vec![],
            },
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let args: ConfigureCacheArgs = serde_json::from_value(arguments)
            .map_err(|e| McpError::InvalidParams(e.to_string()))?;

        // Get current config
        let old_config = self.cache.config().await;
        let old_info = CacheConfigInfo {
            ttl_secs: old_config.ttl.as_secs(),
            max_entries: old_config.max_entries,
            enabled: old_config.enabled,
        };

        // Build new config
        let new_config = crate::cache::CacheConfig {
            ttl: args
                .ttl_secs
                .map(Duration::from_secs)
                .unwrap_or(old_config.ttl),
            max_entries: args.max_entries.unwrap_or(old_config.max_entries),
            enabled: args.enabled.unwrap_or(old_config.enabled),
        };

        // Update config
        self.cache
            .update_config(new_config.clone(), args.clear_on_change)
            .await;

        let new_info = CacheConfigInfo {
            ttl_secs: new_config.ttl.as_secs(),
            max_entries: new_config.max_entries,
            enabled: new_config.enabled,
        };

        // Build change description
        let mut changes = Vec::new();
        if old_info.ttl_secs != new_info.ttl_secs {
            changes.push(format!(
                "TTL: {}s -> {}s",
                old_info.ttl_secs, new_info.ttl_secs
            ));
        }
        if old_info.max_entries != new_info.max_entries {
            changes.push(format!(
                "max_entries: {} -> {}",
                old_info.max_entries, new_info.max_entries
            ));
        }
        if old_info.enabled != new_info.enabled {
            changes.push(format!(
                "enabled: {} -> {}",
                old_info.enabled, new_info.enabled
            ));
        }

        let message = if changes.is_empty() {
            "No configuration changes applied".to_string()
        } else {
            format!("Updated cache configuration: {}", changes.join(", "))
        };

        let result = ConfigureCacheResult {
            success: true,
            old_config: old_info,
            new_config: new_info,
            message,
        };

        ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
    }
}

// ============================================================================
// save_cache tool
// ============================================================================

/// Save the verification cache to disk via MCP
pub struct SaveCacheTool {
    cache: SharedVerificationCache<VerifyUslResult>,
}

impl SaveCacheTool {
    /// Create a new save_cache tool
    pub fn new(cache: SharedVerificationCache<VerifyUslResult>) -> Self {
        Self { cache }
    }
}

/// Arguments for save_cache tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveCacheArgs {
    /// Path to save the cache file
    pub path: String,
}

/// Result of save_cache tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveCacheResult {
    /// Success indicator
    pub success: bool,
    /// Number of entries saved
    pub entries_saved: usize,
    /// Path where the cache was saved
    pub path: String,
    /// Size of the saved file in bytes
    pub size_bytes: u64,
    /// Message
    pub message: String,
}

#[async_trait]
impl Tool for SaveCacheTool {
    fn definition(&self) -> ToolDefinition {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some("File path to save the cache to (JSON format)".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );

        ToolDefinition {
            name: "save_cache".to_string(),
            title: Some("Save Cache".to_string()),
            description: "Save the verification cache to a file for persistence across server \
                         restarts. Creates a JSON snapshot including all non-expired entries, \
                         configuration, and statistics."
                .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(properties),
                required: vec!["path".to_string()],
            },
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let args: SaveCacheArgs = serde_json::from_value(arguments)
            .map_err(|e| McpError::InvalidParams(e.to_string()))?;

        match self.cache.save_to_file(&args.path).await {
            Ok(save_result) => {
                let result = SaveCacheResult {
                    success: true,
                    entries_saved: save_result.entries_saved,
                    path: save_result.path,
                    size_bytes: save_result.size_bytes,
                    message: format!(
                        "Saved {} cache entries to {} ({} bytes)",
                        save_result.entries_saved, args.path, save_result.size_bytes
                    ),
                };
                ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
            }
            Err(e) => {
                let result = SaveCacheResult {
                    success: false,
                    entries_saved: 0,
                    path: args.path.clone(),
                    size_bytes: 0,
                    message: format!("Failed to save cache: {}", e),
                };
                ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
            }
        }
    }
}

// ============================================================================
// load_cache tool
// ============================================================================

/// Load the verification cache from disk via MCP
pub struct LoadCacheTool {
    cache: SharedVerificationCache<VerifyUslResult>,
}

impl LoadCacheTool {
    /// Create a new load_cache tool
    pub fn new(cache: SharedVerificationCache<VerifyUslResult>) -> Self {
        Self { cache }
    }
}

/// Arguments for load_cache tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadCacheArgs {
    /// Path to load the cache file from
    pub path: String,
    /// Whether to merge with existing entries (default: false = replace)
    #[serde(default)]
    pub merge: bool,
}

/// Result of load_cache tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadCacheResult {
    /// Success indicator
    pub success: bool,
    /// Number of entries loaded
    pub entries_loaded: usize,
    /// Number of entries skipped (already expired)
    pub entries_expired: usize,
    /// Path from which the cache was loaded
    pub path: String,
    /// Age of the snapshot in seconds
    pub snapshot_age_secs: u64,
    /// Message
    pub message: String,
}

#[async_trait]
impl Tool for LoadCacheTool {
    fn definition(&self) -> ToolDefinition {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            PropertySchema {
                property_type: "string".to_string(),
                description: Some("File path to load the cache from".to_string()),
                enum_values: None,
                default: None,
                items: None,
            },
        );
        properties.insert(
            "merge".to_string(),
            PropertySchema {
                property_type: "boolean".to_string(),
                description: Some(
                    "If true, merge loaded entries with existing cache. If false (default), \
                     replace existing cache entirely."
                        .to_string(),
                ),
                enum_values: None,
                default: Some(serde_json::json!(false)),
                items: None,
            },
        );

        ToolDefinition {
            name: "load_cache".to_string(),
            title: Some("Load Cache".to_string()),
            description: "Load the verification cache from a previously saved file. Entries that \
                         have already expired (based on their age at save time plus time since \
                         save) are skipped. Configuration is updated from the file."
                .to_string(),
            input_schema: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some(properties),
                required: vec!["path".to_string()],
            },
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolCallResult, McpError> {
        let args: LoadCacheArgs = serde_json::from_value(arguments)
            .map_err(|e| McpError::InvalidParams(e.to_string()))?;

        match self.cache.load_from_file(&args.path, args.merge).await {
            Ok(load_result) => {
                let merge_mode = if args.merge { "merged" } else { "replaced" };
                let result = LoadCacheResult {
                    success: true,
                    entries_loaded: load_result.entries_loaded,
                    entries_expired: load_result.entries_expired,
                    path: load_result.path,
                    snapshot_age_secs: load_result.snapshot_age_secs,
                    message: format!(
                        "Loaded {} cache entries ({} expired, {}) from snapshot {}s old",
                        load_result.entries_loaded,
                        load_result.entries_expired,
                        merge_mode,
                        load_result.snapshot_age_secs
                    ),
                };
                ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
            }
            Err(e) => {
                let result = LoadCacheResult {
                    success: false,
                    entries_loaded: 0,
                    entries_expired: 0,
                    path: args.path.clone(),
                    snapshot_age_secs: 0,
                    message: format!("Failed to load cache: {}", e),
                };
                ToolCallResult::json(&result).map_err(|e| McpError::InternalError(e.to_string()))
            }
        }
    }
}

// ============================================================================
/// Registry of all available tools
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
    session_manager: Arc<SessionManager>,
    /// Shared verification cache for result caching
    verification_cache: SharedVerificationCache<VerifyUslResult>,
}

impl ToolRegistry {
    /// Create a new registry with default DashProve tools
    pub fn new() -> Self {
        Self::with_caching(true)
    }

    /// Create a new registry with optional caching
    pub fn with_caching(enable_cache: bool) -> Self {
        let verification_cache: SharedVerificationCache<VerifyUslResult> = if enable_cache {
            Arc::new(VerificationCache::new())
        } else {
            Arc::new(VerificationCache::disabled())
        };
        Self::with_shared_cache(verification_cache)
    }

    /// Create a new registry with a provided shared cache
    ///
    /// This allows external control over the cache, enabling persistence
    /// across server restarts (load on startup, save on shutdown).
    pub fn with_shared_cache(verification_cache: SharedVerificationCache<VerifyUslResult>) -> Self {
        let session_manager = Arc::new(SessionManager::new());

        Self {
            tools: vec![
                Box::new(VerifyUslTool::with_cache(verification_cache.clone())),
                Box::new(SelectBackendTool::new()),
                Box::new(CompileToTool::new()),
                Box::new(CheckDependenciesTool::new()),
                Box::new(GetSuggestionsTool::new()),
                Box::new(VerifyUslStreamingTool::new(session_manager.clone())),
                Box::new(GetSessionStatusTool::new(session_manager.clone())),
                Box::new(CancelSessionTool::new(session_manager.clone())),
                Box::new(BatchVerifyTool::with_cache(verification_cache.clone())),
                Box::new(GetCacheStatsTool::new(verification_cache.clone())),
                Box::new(ClearCacheTool::new(verification_cache.clone())),
                Box::new(ConfigureCacheTool::new(verification_cache.clone())),
                Box::new(SaveCacheTool::new(verification_cache.clone())),
                Box::new(LoadCacheTool::new(verification_cache.clone())),
            ],
            session_manager,
            verification_cache,
        }
    }

    /// Get the session manager for SSE endpoints
    pub fn session_manager(&self) -> Arc<SessionManager> {
        self.session_manager.clone()
    }

    /// Get the verification cache
    pub fn verification_cache(&self) -> SharedVerificationCache<VerifyUslResult> {
        self.verification_cache.clone()
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> crate::cache::CacheStats {
        self.verification_cache.stats().await
    }

    /// Get all tool definitions
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| t.definition()).collect()
    }

    /// Find a tool by name
    pub fn find(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.definition().name == name)
            .map(|t| t.as_ref())
    }

    /// Execute a tool by name
    pub async fn execute(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult, McpError> {
        let tool = self
            .find(name)
            .ok_or_else(|| McpError::ToolNotFound(name.to_string()))?;
        tool.execute(arguments).await
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
