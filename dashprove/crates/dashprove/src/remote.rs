//! Remote HTTP client for connecting to a DashProve server
//!
//! This module provides `RemoteClient` for DashFlow/Dasher integration when
//! running DashProve as a standalone service rather than embedded.
//!
//! # Example
//!
//! ```rust,no_run
//! use dashprove::remote::{RemoteClient, RemoteConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Connect to a running DashProve server
//! let client = RemoteClient::new(RemoteConfig {
//!     base_url: "http://localhost:3000".to_string(),
//!     api_key: Some("your-api-key".to_string()),
//!     timeout_secs: 30,
//! });
//!
//! // Verify a specification
//! let result = client.verify("theorem test { true }").await?;
//! println!("Valid: {}", result.valid);
//!
//! // Check server health
//! let health = client.health().await?;
//! println!("Server status: {}", health.status);
//! # Ok(())
//! # }
//! ```

use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

pub use crate::BackendIdParam;

/// Errors from the remote client
#[derive(Error, Debug)]
pub enum RemoteError {
    /// HTTP request failed
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Authentication configuration error
    #[error("Authentication error: {0}")]
    Auth(String),

    /// Server returned an error response
    #[error("Server error ({status}): {message}")]
    Server { status: u16, message: String },

    /// Failed to parse response
    #[error("Parse error: {0}")]
    Parse(String),

    /// Request timed out
    #[error("Request timed out after {0} seconds")]
    Timeout(u64),

    /// Server is draining/unavailable
    #[error("Server unavailable: {0}")]
    Unavailable(String),
}

/// Configuration for remote client
#[derive(Debug, Clone)]
pub struct RemoteConfig {
    /// Base URL of the DashProve server (e.g., "http://localhost:3000")
    pub base_url: String,
    /// Optional API key for authentication
    pub api_key: Option<String>,
    /// Request timeout in seconds (default: 30)
    pub timeout_secs: u64,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:3000".to_string(),
            api_key: None,
            timeout_secs: 30,
        }
    }
}

impl RemoteConfig {
    /// Create config for a local server
    pub fn local(port: u16) -> Self {
        Self {
            base_url: format!("http://localhost:{}", port),
            ..Default::default()
        }
    }

    /// Create config with API key
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

// ============ API Types ============
// These mirror the server's request/response types

/// Request to verify a specification
#[derive(Debug, Serialize)]
pub struct VerifyRequest {
    /// USL specification source code
    pub spec: String,
    /// Optional: specific backend to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<BackendIdParam>,
    /// Enable ML-based backend selection
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub use_ml: bool,
    /// Minimum confidence threshold for ML predictions (0.0-1.0)
    #[serde(skip_serializing_if = "is_default_confidence")]
    pub ml_min_confidence: f64,
}

fn is_default_confidence(v: &f64) -> bool {
    (*v - 0.5).abs() < f64::EPSILON
}

/// Response from verify endpoint
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VerifyResponse {
    /// Whether parsing and type-checking succeeded
    pub valid: bool,
    /// Number of properties found
    pub property_count: usize,
    /// Compilation outputs per backend
    pub compilations: Vec<CompilationResult>,
    /// Errors (if any)
    pub errors: Vec<String>,
    /// ML prediction info (when use_ml is true)
    #[serde(default)]
    pub ml_prediction: Option<MlPredictionInfo>,
}

/// ML prediction information in verification response
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MlPredictionInfo {
    /// Predicted best backend
    pub predicted_backend: BackendIdParam,
    /// Prediction confidence (0.0-1.0)
    pub confidence: f64,
    /// Whether the prediction was used (confidence >= threshold)
    pub used: bool,
    /// Alternative backends with their confidence scores
    pub alternatives: Vec<(BackendIdParam, f64)>,
}

/// Compilation result for a single backend
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompilationResult {
    /// Which backend this compilation is for
    pub backend: BackendIdParam,
    /// The compiled code for this backend
    pub code: String,
}

/// Request for incremental verification
#[derive(Debug, Serialize)]
pub struct IncrementalVerifyRequest {
    /// Base specification (previous state)
    pub base_spec: String,
    /// Current specification (after changes)
    pub current_spec: String,
    /// List of changes that occurred
    pub changes: Vec<Change>,
    /// Optional: specific backend to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<BackendIdParam>,
}

/// A change description for incremental verification
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Change {
    /// Type of change
    pub change_type: String,
    /// Name of changed element
    pub name: String,
}

/// Response from incremental verification
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IncrementalVerifyResponse {
    /// Overall validity
    pub valid: bool,
    /// Number of properties that were cached/reused
    pub cached_count: usize,
    /// Number of properties that were newly verified
    pub verified_count: usize,
    /// Properties that were affected by changes
    pub affected_properties: Vec<String>,
    /// Properties that were unchanged and cached
    pub unchanged_properties: Vec<String>,
    /// Compilation results for affected properties
    pub compilations: Vec<CompilationResult>,
    /// Errors (if any)
    pub errors: Vec<String>,
}

/// Health check response
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HealthResponse {
    /// Overall status: "healthy", "draining", or "unhealthy"
    pub status: String,
    /// Server is ready to accept requests
    pub ready: bool,
    /// Current shutdown state
    pub shutdown_state: String,
    /// Number of in-flight requests
    pub in_flight_requests: usize,
    /// Number of active WebSocket sessions
    pub active_sessions: usize,
}

/// Version response
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VersionResponse {
    /// Package name
    pub name: String,
    /// Package version
    pub version: String,
    /// API version
    pub api_version: String,
}

/// Corpus search response
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CorpusSearchResponse {
    /// List of similar proofs
    pub results: Vec<SimilarProofResponse>,
    /// Total corpus size
    pub total_corpus_size: usize,
}

/// Similar proof in search results
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SimilarProofResponse {
    /// Proof ID
    pub proof_id: String,
    /// Similarity score (0.0-1.0)
    pub similarity: f64,
    /// Property name
    pub property_name: String,
    /// Backend used
    pub backend: BackendIdParam,
    /// Tactics used
    pub tactics: Vec<String>,
}

/// Backend info response
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BackendsResponse {
    /// List of available backends
    pub backends: Vec<BackendInfo>,
}

/// Information about a single backend
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BackendInfo {
    /// Backend identifier
    pub id: BackendIdParam,
    /// Human-readable name
    pub name: String,
    /// Health status
    pub healthy: bool,
    /// Supported property types
    pub supports: Vec<String>,
}

/// Error response from server
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
    /// Optional details
    #[serde(default)]
    pub details: Option<String>,
}

// ============ Client Implementation ============

/// HTTP client for connecting to a remote DashProve server
///
/// This is the primary integration point for DashFlow and Dasher when
/// DashProve runs as a separate service.
pub struct RemoteClient {
    client: Client,
    config: RemoteConfig,
}

impl RemoteClient {
    /// Create a new remote client with the given configuration
    pub fn new(config: RemoteConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    /// Create a client for a local server on the given port
    pub fn local(port: u16) -> Self {
        Self::new(RemoteConfig::local(port))
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Build a request with common headers
    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.config.base_url, path);
        let mut req = self.client.request(method, &url);

        if let Some(key) = &self.config.api_key {
            req = req.header("X-API-Key", key);
        }

        req
    }

    /// Check server health
    pub async fn health(&self) -> Result<HealthResponse, RemoteError> {
        let resp = self.request(reqwest::Method::GET, "/health").send().await?;

        // Health endpoint returns 503 when draining but response is still valid JSON
        let status = resp.status().as_u16();
        let health: HealthResponse = resp.json().await?;

        if status == 503 {
            // Server is draining but we got valid response
            return Ok(health);
        }

        Ok(health)
    }

    /// Get server version
    pub async fn version(&self) -> Result<VersionResponse, RemoteError> {
        let resp = self
            .request(reqwest::Method::GET, "/version")
            .send()
            .await?;

        Self::handle_response(resp).await
    }

    /// Verify a USL specification
    pub async fn verify(&self, spec: &str) -> Result<VerifyResponse, RemoteError> {
        self.verify_with_options(spec, None, false, 0.5).await
    }

    /// Verify with a specific backend
    pub async fn verify_with_backend(
        &self,
        spec: &str,
        backend: Option<BackendIdParam>,
    ) -> Result<VerifyResponse, RemoteError> {
        self.verify_with_options(spec, backend, false, 0.5).await
    }

    /// Verify with ML-based backend selection
    pub async fn verify_with_ml(
        &self,
        spec: &str,
        min_confidence: f64,
    ) -> Result<VerifyResponse, RemoteError> {
        self.verify_with_options(spec, None, true, min_confidence)
            .await
    }

    /// Verify with full configuration options
    pub async fn verify_with_options(
        &self,
        spec: &str,
        backend: Option<BackendIdParam>,
        use_ml: bool,
        ml_min_confidence: f64,
    ) -> Result<VerifyResponse, RemoteError> {
        let req = VerifyRequest {
            spec: spec.to_string(),
            backend,
            use_ml,
            ml_min_confidence,
        };

        let resp = self
            .request(reqwest::Method::POST, "/verify")
            .json(&req)
            .send()
            .await?;

        Self::handle_response(resp).await
    }

    /// Incremental verification after changes
    pub async fn verify_incremental(
        &self,
        base_spec: &str,
        current_spec: &str,
        changes: Vec<Change>,
    ) -> Result<IncrementalVerifyResponse, RemoteError> {
        let req = IncrementalVerifyRequest {
            base_spec: base_spec.to_string(),
            current_spec: current_spec.to_string(),
            changes,
            backend: None,
        };

        let resp = self
            .request(reqwest::Method::POST, "/verify/incremental")
            .json(&req)
            .send()
            .await?;

        Self::handle_response(resp).await
    }

    /// Search the proof corpus for similar proofs
    pub async fn corpus_search(
        &self,
        query: &str,
        k: usize,
    ) -> Result<CorpusSearchResponse, RemoteError> {
        let resp = self
            .request(reqwest::Method::GET, "/corpus/search")
            .query(&[("query", query), ("k", &k.to_string())])
            .send()
            .await?;

        Self::handle_response(resp).await
    }

    /// Get list of available backends with health status
    pub async fn backends(&self) -> Result<BackendsResponse, RemoteError> {
        let resp = self
            .request(reqwest::Method::GET, "/backends")
            .send()
            .await?;

        Self::handle_response(resp).await
    }

    /// Handle response, extracting error details on failure
    async fn handle_response<T: serde::de::DeserializeOwned>(
        resp: reqwest::Response,
    ) -> Result<T, RemoteError> {
        let status = resp.status();

        if status.is_success() {
            Ok(resp.json().await?)
        } else {
            // Try to parse error response
            let status_code = status.as_u16();
            let error_text = resp.text().await.unwrap_or_default();

            // Try to parse as JSON error
            if let Ok(err) = serde_json::from_str::<ErrorResponse>(&error_text) {
                Err(RemoteError::Server {
                    status: status_code,
                    message: err.error,
                })
            } else {
                Err(RemoteError::Server {
                    status: status_code,
                    message: error_text,
                })
            }
        }
    }
}

// ============ DashFlow Integration Helper ============

/// DashFlow verification client
///
/// A higher-level wrapper around `RemoteClient` specifically for DashFlow's
/// verification needs. Provides methods matching DashFlow's verification workflow.
pub struct DashFlowClient {
    inner: RemoteClient,
}

impl DashFlowClient {
    /// Create a new DashFlow client
    pub fn new(config: RemoteConfig) -> Self {
        Self {
            inner: RemoteClient::new(config),
        }
    }

    /// Connect to a local DashProve server
    pub fn local(port: u16) -> Self {
        Self::new(RemoteConfig::local(port))
    }

    /// Check if the server is healthy and ready
    pub async fn is_ready(&self) -> bool {
        match self.inner.health().await {
            Ok(health) => health.ready,
            Err(_) => false,
        }
    }

    /// Verify a graph modification is safe
    ///
    /// Takes a USL specification describing the modification safety properties.
    pub async fn verify_modification(&self, spec: &str) -> Result<ModificationResult, RemoteError> {
        let response = self.inner.verify(spec).await?;

        Ok(ModificationResult {
            safe: response.valid && response.errors.is_empty(),
            property_count: response.property_count,
            errors: response.errors,
            ml_prediction: response.ml_prediction,
        })
    }

    /// Verify with ML-based backend selection
    ///
    /// Uses the ML predictor on the server to select the best backend for the spec.
    pub async fn verify_modification_with_ml(
        &self,
        spec: &str,
        min_confidence: f64,
    ) -> Result<ModificationResult, RemoteError> {
        let response = self.inner.verify_with_ml(spec, min_confidence).await?;

        Ok(ModificationResult {
            safe: response.valid && response.errors.is_empty(),
            property_count: response.property_count,
            errors: response.errors,
            ml_prediction: response.ml_prediction,
        })
    }

    /// Verify with incremental checking (for iterative development)
    pub async fn verify_incremental(
        &self,
        base_spec: &str,
        current_spec: &str,
        changes: Vec<Change>,
    ) -> Result<IncrementalResult, RemoteError> {
        let response = self
            .inner
            .verify_incremental(base_spec, current_spec, changes)
            .await?;

        Ok(IncrementalResult {
            valid: response.valid,
            cached_count: response.cached_count,
            verified_count: response.verified_count,
            affected_properties: response.affected_properties,
            errors: response.errors,
        })
    }

    /// Find similar proofs for learning
    pub async fn find_similar_proofs(
        &self,
        property: &str,
        k: usize,
    ) -> Result<Vec<SimilarProofResponse>, RemoteError> {
        let response = self.inner.corpus_search(property, k).await?;
        Ok(response.results)
    }

    /// Get inner client for advanced operations
    pub fn inner(&self) -> &RemoteClient {
        &self.inner
    }
}

/// Result of a modification verification
#[derive(Debug, Clone)]
pub struct ModificationResult {
    /// Whether the modification is safe
    pub safe: bool,
    /// Number of properties verified
    pub property_count: usize,
    /// Any errors encountered
    pub errors: Vec<String>,
    /// ML prediction info (if ML selection was used)
    pub ml_prediction: Option<MlPredictionInfo>,
}

/// Result of incremental verification
#[derive(Debug, Clone)]
pub struct IncrementalResult {
    /// Overall validity
    pub valid: bool,
    /// Number of cached results reused
    pub cached_count: usize,
    /// Number of newly verified properties
    pub verified_count: usize,
    /// Properties affected by changes
    pub affected_properties: Vec<String>,
    /// Any errors encountered
    pub errors: Vec<String>,
}

// ============ DashFlow ML Integration Types ============

/// Features extracted from properties for ML prediction
/// Matches DASHFLOW_INTEGRATION.md specification
///
/// This type mirrors `dashprove_learning::similarity::PropertyFeatures` but is
/// designed for network serialization in the DashFlow ML API. Use the `From`/`Into`
/// implementations to convert between them.
///
/// # Example
///
/// ```rust
/// use dashprove::remote::PropertyFeatures;
/// use dashprove::learning::similarity::PropertyFeatures as LearningFeatures;
///
/// // Convert from learning features to remote features for API calls
/// let learning: LearningFeatures = Default::default();
/// let remote: PropertyFeatures = learning.into();
///
/// // Convert back from remote to learning for local processing
/// let learning_again: LearningFeatures = remote.into();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyFeatures {
    /// Property type: "theorem", "invariant", "temporal", "contract", "refinement"
    pub property_type: String,

    /// Maximum nesting depth of expressions
    pub depth: usize,

    /// Number of quantifiers (forall, exists)
    pub quantifier_depth: usize,

    /// Number of implications
    pub implication_count: usize,

    /// Number of arithmetic operations
    pub arithmetic_ops: usize,

    /// Number of function calls
    pub function_calls: usize,

    /// Number of variables
    pub variable_count: usize,

    /// Uses temporal operators (always, eventually, until, leads_to)
    pub has_temporal: bool,

    /// Type names referenced
    pub type_refs: Vec<String>,

    /// Keywords for text-based search
    pub keywords: Vec<String>,
}

/// Convert from learning crate's PropertyFeatures to remote API PropertyFeatures
impl From<dashprove_learning::similarity::PropertyFeatures> for PropertyFeatures {
    fn from(f: dashprove_learning::similarity::PropertyFeatures) -> Self {
        Self {
            property_type: f.property_type,
            depth: f.depth,
            quantifier_depth: f.quantifier_depth,
            implication_count: f.implication_count,
            arithmetic_ops: f.arithmetic_ops,
            function_calls: f.function_calls,
            variable_count: f.variable_count,
            has_temporal: f.has_temporal,
            type_refs: f.type_refs,
            keywords: f.keywords,
        }
    }
}

/// Convert from remote API PropertyFeatures to learning crate's PropertyFeatures
impl From<PropertyFeatures> for dashprove_learning::similarity::PropertyFeatures {
    fn from(f: PropertyFeatures) -> Self {
        Self {
            property_type: f.property_type,
            depth: f.depth,
            quantifier_depth: f.quantifier_depth,
            implication_count: f.implication_count,
            arithmetic_ops: f.arithmetic_ops,
            function_calls: f.function_calls,
            variable_count: f.variable_count,
            has_temporal: f.has_temporal,
            type_refs: f.type_refs,
            keywords: f.keywords,
        }
    }
}

/// Additional context when verifying code contracts
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeContext {
    /// Target programming language
    pub language: String,

    /// Lines of code in target
    pub lines_of_code: usize,

    /// Cyclomatic complexity
    pub cyclomatic_complexity: usize,

    /// Has unsafe blocks (Rust)
    pub has_unsafe: bool,

    /// Has concurrency primitives
    pub has_concurrency: bool,

    /// Has heap allocation
    pub has_heap: bool,
}

/// Request to DashFlow /api/v1/predict endpoint
#[derive(Debug, Serialize)]
pub struct PredictRequest {
    /// Property features for the property to verify
    pub features: PropertyFeatures,
    /// Optional code context
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_context: Option<CodeContext>,
}

/// Response from DashFlow /api/v1/predict endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct PredictResponse {
    /// Predicted best backend
    pub backend: String,
    /// Confidence (0.0-1.0)
    pub confidence: f64,
    /// Alternative backends with their probabilities
    pub alternatives: Vec<(String, f64)>,
}

/// Verification status for feedback
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FeedbackVerificationStatus {
    Proven,
    Refuted,
    Unknown,
    Timeout,
    Error,
}

/// Feedback sent to DashFlow after verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationFeedback {
    /// Property features for the verified property
    pub features: PropertyFeatures,
    /// Optional code context
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_context: Option<CodeContext>,
    /// Backend used for verification
    pub backend: String,
    /// Verification outcome
    pub status: FeedbackVerificationStatus,
    /// Time taken for verification (seconds)
    pub time_seconds: f64,
    /// Proof size (if successful)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_size: Option<usize>,
    /// Error message (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Tactics used (if any)
    pub tactics: Vec<String>,
    /// Timestamp (ISO 8601)
    pub timestamp: String,
}

/// Response from DashFlow /api/v1/feedback endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct FeedbackResponse {
    /// Whether the feedback was recorded
    pub recorded: bool,
    /// Feedback ID for tracking
    pub feedback_id: String,
}

/// Batch feedback response from DashFlow
#[derive(Debug, Clone, Deserialize)]
pub struct BatchFeedbackResponse {
    /// Number of records received
    pub received: usize,
    /// Number of records accepted
    pub accepted: usize,
    /// Number of records rejected
    pub rejected: usize,
    /// Batch ID for tracking
    pub batch_id: String,
}

/// Model status response from DashFlow
#[derive(Debug, Clone, Deserialize)]
pub struct ModelStatusResponse {
    /// Model version
    pub version: String,
    /// Last training timestamp
    pub last_trained: String,
    /// Number of training samples
    pub training_samples: usize,
    /// Average prediction accuracy
    pub accuracy: f64,
    /// Supported backends
    pub supported_backends: Vec<String>,
}

// ============ DashFlow ML API Client ============

/// Client for DashFlow ML API endpoints
///
/// This client implements the ML-based backend selection protocol
/// as specified in DASHFLOW_INTEGRATION.md.
///
/// # Example
///
/// ```rust,no_run
/// use dashprove::remote::{DashFlowMlClient, DashFlowMlConfig, PropertyFeatures};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = DashFlowMlClient::new(
///     DashFlowMlConfig::default()
///         .with_base_url("https://dashflow.example.com")
///         .with_jwt_secret("shared-dashflow-secret")
///         .with_timeout(10),
/// );
///
/// // Get backend prediction
/// let features = PropertyFeatures {
///     property_type: "theorem".to_string(),
///     depth: 3,
///     quantifier_depth: 1,
///     implication_count: 2,
///     arithmetic_ops: 0,
///     function_calls: 1,
///     variable_count: 3,
///     has_temporal: false,
///     type_refs: vec!["Bool".to_string()],
///     keywords: vec!["excluded_middle".to_string()],
/// };
///
/// let prediction = client.predict(&features, None).await?;
/// println!("Predicted backend: {} (confidence: {:.2})", prediction.backend, prediction.confidence);
/// # Ok(())
/// # }
/// ```
pub struct DashFlowMlClient {
    client: Client,
    config: DashFlowMlConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DashFlowJwtClaims {
    iss: String,
    sub: String,
    aud: String,
    iat: i64,
    exp: i64,
}

/// Configuration for DashFlow ML client
#[derive(Debug, Clone)]
pub struct DashFlowMlConfig {
    /// Base URL of the DashFlow API (e.g., "https://dashflow.example.com")
    pub base_url: String,
    /// API key for authentication (legacy)
    pub api_key: Option<String>,
    /// Optional shared secret for signing JWTs (preferred)
    pub jwt_secret: Option<String>,
    /// Issuer claim for JWTs
    pub jwt_issuer: String,
    /// Audience claim for JWTs
    pub jwt_audience: String,
    /// Subject claim for JWTs
    pub jwt_subject: String,
    /// JWT time-to-live in seconds
    pub jwt_ttl_secs: u64,
    /// Request timeout in seconds (default: 10)
    pub timeout_secs: u64,
}

impl Default for DashFlowMlConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8000".to_string(),
            api_key: None,
            jwt_secret: None,
            jwt_issuer: "dashprove".to_string(),
            jwt_audience: "dashflow".to_string(),
            jwt_subject: "dashprove-client".to_string(),
            jwt_ttl_secs: 300,
            timeout_secs: 10,
        }
    }
}

impl DashFlowMlConfig {
    /// Create config for a local DashFlow server
    pub fn local(port: u16) -> Self {
        Self {
            base_url: format!("http://localhost:{}", port),
            ..Default::default()
        }
    }

    /// Set base URL
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set API key
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Configure JWT secret for bearer authentication
    ///
    /// When set, a short-lived JWT is minted per request using HS256.
    pub fn with_jwt_secret(mut self, secret: impl Into<String>) -> Self {
        self.jwt_secret = Some(secret.into());
        self
    }

    /// Override JWT issuer claim
    pub fn with_jwt_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.jwt_issuer = issuer.into();
        self
    }

    /// Override JWT audience claim
    pub fn with_jwt_audience(mut self, audience: impl Into<String>) -> Self {
        self.jwt_audience = audience.into();
        self
    }

    /// Override JWT subject claim
    pub fn with_jwt_subject(mut self, subject: impl Into<String>) -> Self {
        self.jwt_subject = subject.into();
        self
    }

    /// Override JWT TTL (seconds)
    pub fn with_jwt_ttl(mut self, ttl_secs: u64) -> Self {
        self.jwt_ttl_secs = ttl_secs;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

impl DashFlowMlClient {
    /// Create a new DashFlow ML client
    pub fn new(config: DashFlowMlConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    /// Create client for local server
    pub fn local(port: u16) -> Self {
        Self::new(DashFlowMlConfig::local(port))
    }

    /// Build a request with common headers
    fn request(
        &self,
        method: reqwest::Method,
        path: &str,
    ) -> Result<reqwest::RequestBuilder, RemoteError> {
        let url = format!("{}{}", self.config.base_url, path);
        let mut req = self.client.request(method, &url);

        if let Some(auth_header) = self.build_auth_header()? {
            req = req.header("Authorization", auth_header);
        }

        Ok(req.header("Content-Type", "application/json"))
    }

    /// Build Authorization header using JWT (preferred) or API key
    fn build_auth_header(&self) -> Result<Option<String>, RemoteError> {
        if let Some(secret) = &self.config.jwt_secret {
            let now = chrono::Utc::now();
            let exp = now + chrono::Duration::seconds(self.config.jwt_ttl_secs as i64);

            let claims = DashFlowJwtClaims {
                iss: self.config.jwt_issuer.clone(),
                sub: self.config.jwt_subject.clone(),
                aud: self.config.jwt_audience.clone(),
                iat: now.timestamp(),
                exp: exp.timestamp(),
            };

            let token = encode(
                &Header::new(Algorithm::HS256),
                &claims,
                &EncodingKey::from_secret(secret.as_bytes()),
            )
            .map_err(|e| RemoteError::Auth(format!("failed to sign JWT: {e}")))?;

            return Ok(Some(format!("Bearer {}", token)));
        }

        if let Some(key) = &self.config.api_key {
            return Ok(Some(format!("Bearer {}", key)));
        }

        Ok(None)
    }

    /// Get backend prediction for property features
    ///
    /// # Arguments
    /// * `features` - Property features extracted from the specification
    /// * `code_context` - Optional code context for contract verification
    ///
    /// # Returns
    /// Backend prediction with confidence and alternatives
    pub async fn predict(
        &self,
        features: &PropertyFeatures,
        code_context: Option<CodeContext>,
    ) -> Result<PredictResponse, RemoteError> {
        let request = PredictRequest {
            features: features.clone(),
            code_context,
        };

        let resp = self
            .request(reqwest::Method::POST, "/api/v1/predict")?
            .json(&request)
            .send()
            .await?;

        Self::handle_response(resp).await
    }

    /// Send verification feedback to DashFlow
    ///
    /// This should be called after each verification to train the ML model.
    ///
    /// # Arguments
    /// * `feedback` - Verification feedback with features, backend, and outcome
    ///
    /// # Returns
    /// Feedback response with recorded status and ID
    pub async fn send_feedback(
        &self,
        feedback: &VerificationFeedback,
    ) -> Result<FeedbackResponse, RemoteError> {
        let resp = self
            .request(reqwest::Method::POST, "/api/v1/feedback")?
            .json(feedback)
            .send()
            .await?;

        Self::handle_response(resp).await
    }

    /// Send batch verification feedback
    ///
    /// For bulk upload of historical verification results.
    pub async fn send_batch_feedback(
        &self,
        feedbacks: &[VerificationFeedback],
    ) -> Result<BatchFeedbackResponse, RemoteError> {
        let resp = self
            .request(reqwest::Method::POST, "/api/v1/feedback/batch")?
            .json(feedbacks)
            .send()
            .await?;

        Self::handle_response(resp).await
    }

    /// Get model status
    ///
    /// Returns information about the current ML model state.
    pub async fn model_status(&self) -> Result<ModelStatusResponse, RemoteError> {
        let resp = self
            .request(reqwest::Method::GET, "/api/v1/model/status")?
            .send()
            .await?;

        Self::handle_response(resp).await
    }

    /// Check if the DashFlow ML API is available
    pub async fn is_available(&self) -> bool {
        self.model_status().await.is_ok()
    }

    /// Handle response, extracting error details on failure
    async fn handle_response<T: serde::de::DeserializeOwned>(
        resp: reqwest::Response,
    ) -> Result<T, RemoteError> {
        let status = resp.status();

        if status.is_success() {
            Ok(resp.json().await?)
        } else {
            let status_code = status.as_u16();
            let error_text = resp.text().await.unwrap_or_default();

            if let Ok(err) = serde_json::from_str::<ErrorResponse>(&error_text) {
                Err(RemoteError::Server {
                    status: status_code,
                    message: err.error,
                })
            } else {
                Err(RemoteError::Server {
                    status: status_code,
                    message: error_text,
                })
            }
        }
    }
}

// ============ Feedback Queue with Persistence ============

/// A persistent feedback queue for offline feedback collection
///
/// When DashFlow is unavailable, feedback is queued locally and persisted to disk.
/// The queue automatically retries failed submissions when DashFlow becomes available.
///
/// # Example
///
/// ```rust,no_run
/// use dashprove::remote::{FeedbackQueue, VerificationFeedback, PropertyFeatures, FeedbackVerificationStatus};
/// use std::path::Path;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a queue with persistence
/// let mut queue = FeedbackQueue::new(Path::new("/tmp/feedback_queue.json"))?;
///
/// // Create mock feedback
/// let features = PropertyFeatures {
///     property_type: "theorem".to_string(),
///     depth: 2,
///     quantifier_depth: 1,
///     implication_count: 0,
///     arithmetic_ops: 0,
///     function_calls: 0,
///     variable_count: 1,
///     has_temporal: false,
///     type_refs: vec![],
///     keywords: vec![],
/// };
/// let feedback = VerificationFeedback {
///     features,
///     code_context: None,
///     backend: "Lean4".to_string(),
///     status: FeedbackVerificationStatus::Proven,
///     time_seconds: 1.5,
///     proof_size: Some(100),
///     error_message: None,
///     tactics: vec!["rfl".to_string()],
///     timestamp: "2025-12-27T00:00:00Z".to_string(),
/// };
///
/// // Push feedback to queue
/// queue.push(feedback);
///
/// // Try to flush queue to DashFlow
/// // queue.flush(&client).await?;
/// # Ok(())
/// # }
/// ```
pub struct FeedbackQueue {
    /// In-memory queue of pending feedbacks
    queue: std::collections::VecDeque<VerificationFeedback>,
    /// Path to persistence file
    persistence_path: std::path::PathBuf,
    /// Maximum queue size before oldest entries are dropped
    max_size: usize,
}

impl FeedbackQueue {
    /// Create a new feedback queue with persistence
    ///
    /// Loads any previously persisted feedbacks from disk.
    pub fn new(persistence_path: &std::path::Path) -> Result<Self, std::io::Error> {
        let mut queue = FeedbackQueue {
            queue: std::collections::VecDeque::new(),
            persistence_path: persistence_path.to_path_buf(),
            max_size: 10000, // Default max queue size
        };

        // Load persisted feedbacks if file exists
        if persistence_path.exists() {
            queue.load()?;
        }

        Ok(queue)
    }

    /// Create a new queue with custom max size
    pub fn with_max_size(
        persistence_path: &std::path::Path,
        max_size: usize,
    ) -> Result<Self, std::io::Error> {
        let mut queue = Self::new(persistence_path)?;
        queue.max_size = max_size;
        Ok(queue)
    }

    /// Push feedback to the queue
    ///
    /// If queue exceeds max_size, oldest entries are dropped.
    /// Changes are persisted to disk.
    pub fn push(&mut self, feedback: VerificationFeedback) {
        self.queue.push_back(feedback);

        // Drop oldest entries if over max size
        while self.queue.len() > self.max_size {
            self.queue.pop_front();
        }

        // Best-effort persist (ignore errors)
        let _ = self.save();
    }

    /// Push feedback to front of queue (for retry)
    pub fn push_front(&mut self, feedback: VerificationFeedback) {
        self.queue.push_front(feedback);

        while self.queue.len() > self.max_size {
            self.queue.pop_back();
        }

        let _ = self.save();
    }

    /// Pop feedback from the queue
    pub fn pop(&mut self) -> Option<VerificationFeedback> {
        let feedback = self.queue.pop_front();
        if feedback.is_some() {
            let _ = self.save();
        }
        feedback
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get queue length
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Flush all queued feedbacks to DashFlow
    ///
    /// Returns the number of successfully sent feedbacks.
    /// Failed feedbacks remain in the queue.
    pub async fn flush(&mut self, client: &DashFlowMlClient) -> Result<usize, RemoteError> {
        let mut success_count = 0;

        while let Some(feedback) = self.queue.pop_front() {
            match client.send_feedback(&feedback).await {
                Ok(_) => {
                    success_count += 1;
                }
                Err(e) => {
                    // Put the failed feedback back at the front and stop
                    self.queue.push_front(feedback);
                    let _ = self.save();
                    return Err(e);
                }
            }
        }

        let _ = self.save();
        Ok(success_count)
    }

    /// Flush up to N feedbacks (for rate limiting)
    pub async fn flush_batch(
        &mut self,
        client: &DashFlowMlClient,
        batch_size: usize,
    ) -> Result<usize, RemoteError> {
        let mut success_count = 0;

        for _ in 0..batch_size {
            if let Some(feedback) = self.queue.pop_front() {
                match client.send_feedback(&feedback).await {
                    Ok(_) => {
                        success_count += 1;
                    }
                    Err(e) => {
                        self.queue.push_front(feedback);
                        let _ = self.save();
                        return Err(e);
                    }
                }
            } else {
                break;
            }
        }

        let _ = self.save();
        Ok(success_count)
    }

    /// Try to send a single feedback, queueing on failure
    ///
    /// Returns true if sent successfully, false if queued.
    pub async fn send_or_queue(
        &mut self,
        client: &DashFlowMlClient,
        feedback: VerificationFeedback,
    ) -> bool {
        match client.send_feedback(&feedback).await {
            Ok(_) => true,
            Err(_) => {
                self.push(feedback);
                false
            }
        }
    }

    /// Save queue to disk
    fn save(&self) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(&Vec::from(self.queue.clone()))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Create parent directory if needed
        if let Some(parent) = self.persistence_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(&self.persistence_path, json)
    }

    /// Load queue from disk
    fn load(&mut self) -> Result<(), std::io::Error> {
        let json = std::fs::read_to_string(&self.persistence_path)?;
        let feedbacks: Vec<VerificationFeedback> = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        self.queue = feedbacks.into_iter().collect();
        Ok(())
    }

    /// Clear the queue and delete persistence file
    pub fn clear(&mut self) -> Result<(), std::io::Error> {
        self.queue.clear();
        if self.persistence_path.exists() {
            std::fs::remove_file(&self.persistence_path)?;
        }
        Ok(())
    }

    /// Get statistics about the queue
    pub fn stats(&self) -> FeedbackQueueStats {
        let mut status_counts = std::collections::HashMap::new();
        let mut backend_counts = std::collections::HashMap::new();

        for feedback in &self.queue {
            *status_counts.entry(feedback.status).or_insert(0) += 1;
            *backend_counts.entry(feedback.backend.clone()).or_insert(0) += 1;
        }

        FeedbackQueueStats {
            total: self.queue.len(),
            max_size: self.max_size,
            status_counts,
            backend_counts,
        }
    }
}

/// Statistics about the feedback queue
#[derive(Debug, Clone)]
pub struct FeedbackQueueStats {
    /// Total number of feedbacks in queue
    pub total: usize,
    /// Maximum queue size
    pub max_size: usize,
    /// Count by verification status
    pub status_counts: std::collections::HashMap<FeedbackVerificationStatus, usize>,
    /// Count by backend
    pub backend_counts: std::collections::HashMap<String, usize>,
}

// ============ Utility Functions ============

/// Extract PropertyFeatures from a USL Property
///
/// This is a convenience wrapper around `dashprove_learning::similarity::extract_features`
/// that returns the remote API type directly.
///
/// # Example
///
/// ```rust
/// use dashprove::remote::extract_property_features;
/// use dashprove::usl::ast::Property;
/// # fn example(property: &Property) {
/// let features = extract_property_features(property);
/// println!("Property type: {}", features.property_type);
/// # }
/// ```
pub fn extract_property_features(property: &dashprove_usl::ast::Property) -> PropertyFeatures {
    dashprove_learning::similarity::extract_features(property).into()
}

/// Create verification feedback from verification result
///
/// Helper function to construct feedback for sending to DashFlow.
pub fn create_feedback(
    features: PropertyFeatures,
    backend: &str,
    success: bool,
    duration_secs: f64,
    proof_size: Option<usize>,
    error_message: Option<String>,
    tactics: Vec<String>,
) -> VerificationFeedback {
    let status = if success {
        FeedbackVerificationStatus::Proven
    } else if error_message.is_some() {
        FeedbackVerificationStatus::Error
    } else {
        FeedbackVerificationStatus::Unknown
    };

    VerificationFeedback {
        features,
        code_context: None,
        backend: backend.to_string(),
        status,
        time_seconds: duration_secs,
        proof_size,
        error_message,
        tactics,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{decode, Algorithm, DecodingKey, Validation};

    #[test]
    fn test_config_default() {
        let config = RemoteConfig::default();
        assert_eq!(config.base_url, "http://localhost:3000");
        assert!(config.api_key.is_none());
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_config_local() {
        let config = RemoteConfig::local(8080);
        assert_eq!(config.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_config_with_api_key() {
        let config = RemoteConfig::default().with_api_key("secret");
        assert_eq!(config.api_key, Some("secret".to_string()));
    }

    #[test]
    fn test_config_with_timeout() {
        let config = RemoteConfig::default().with_timeout(60);
        assert_eq!(config.timeout_secs, 60);
    }

    #[test]
    fn test_client_creation() {
        let client = RemoteClient::new(RemoteConfig::default());
        assert_eq!(client.base_url(), "http://localhost:3000");
    }

    #[test]
    fn test_verify_request_serialization() {
        let req = VerifyRequest {
            spec: "theorem test { true }".to_string(),
            backend: Some(BackendIdParam::Lean4),
            use_ml: false,
            ml_min_confidence: 0.5,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("theorem test"));
        assert!(json.contains("lean4"));
        // Default ML options should be skipped
        assert!(!json.contains("use_ml"));
        assert!(!json.contains("ml_min_confidence"));
    }

    #[test]
    fn test_verify_request_with_ml() {
        let req = VerifyRequest {
            spec: "theorem test { true }".to_string(),
            backend: None,
            use_ml: true,
            ml_min_confidence: 0.7,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("use_ml"));
        assert!(json.contains("0.7"));
    }

    #[test]
    fn test_verify_response_deserialization() {
        let json = r#"{
            "valid": true,
            "property_count": 1,
            "compilations": [],
            "errors": []
        }"#;

        let resp: VerifyResponse = serde_json::from_str(json).unwrap();
        assert!(resp.valid);
        assert_eq!(resp.property_count, 1);
    }

    #[test]
    fn test_health_response_deserialization() {
        let json = r#"{
            "status": "healthy",
            "ready": true,
            "shutdown_state": "running",
            "in_flight_requests": 5,
            "active_sessions": 2
        }"#;

        let resp: HealthResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "healthy");
        assert!(resp.ready);
    }

    #[test]
    fn test_dashflow_client_creation() {
        let client = DashFlowClient::local(3000);
        assert_eq!(client.inner().base_url(), "http://localhost:3000");
    }

    // ============ DashFlow ML Client Tests ============

    #[test]
    fn test_dashflow_ml_config_default() {
        let config = DashFlowMlConfig::default();
        assert_eq!(config.base_url, "http://localhost:8000");
        assert!(config.api_key.is_none());
        assert!(config.jwt_secret.is_none());
        assert_eq!(config.jwt_issuer, "dashprove");
        assert_eq!(config.jwt_audience, "dashflow");
        assert_eq!(config.jwt_subject, "dashprove-client");
        assert_eq!(config.jwt_ttl_secs, 300);
        assert_eq!(config.timeout_secs, 10);
    }

    #[test]
    fn test_dashflow_ml_config_local() {
        let config = DashFlowMlConfig::local(9000);
        assert_eq!(config.base_url, "http://localhost:9000");
    }

    #[test]
    fn test_dashflow_ml_config_with_api_key() {
        let config = DashFlowMlConfig::default().with_api_key("test-key");
        assert_eq!(config.api_key, Some("test-key".to_string()));
    }

    #[test]
    fn test_dashflow_ml_config_with_jwt_settings() {
        let config = DashFlowMlConfig::default()
            .with_base_url("https://dashflow.example.com")
            .with_jwt_secret("jwt-secret")
            .with_jwt_issuer("issuer")
            .with_jwt_audience("audience")
            .with_jwt_subject("subject")
            .with_jwt_ttl(120);

        assert_eq!(config.base_url, "https://dashflow.example.com");
        assert_eq!(config.jwt_secret, Some("jwt-secret".to_string()));
        assert_eq!(config.jwt_issuer, "issuer");
        assert_eq!(config.jwt_audience, "audience");
        assert_eq!(config.jwt_subject, "subject");
        assert_eq!(config.jwt_ttl_secs, 120);
    }

    #[test]
    fn test_dashflow_ml_jwt_auth_header() {
        let config = DashFlowMlConfig::default()
            .with_base_url("https://dashflow.example.com")
            .with_api_key("legacy-key")
            .with_jwt_secret("super-secret")
            .with_jwt_issuer("issuer")
            .with_jwt_audience("audience")
            .with_jwt_subject("subject")
            .with_jwt_ttl(60);

        let client = DashFlowMlClient::new(config);
        let header = client
            .build_auth_header()
            .expect("auth header generation failed")
            .expect("no auth header produced");

        assert!(
            header.starts_with("Bearer "),
            "expected bearer prefix in header"
        );

        let token = header
            .strip_prefix("Bearer ")
            .expect("missing bearer prefix in auth header");

        let mut validation = Validation::new(Algorithm::HS256);
        validation.set_audience(&["audience"]);

        let data = decode::<DashFlowJwtClaims>(
            token,
            &DecodingKey::from_secret("super-secret".as_bytes()),
            &validation,
        )
        .expect("failed to decode JWT");

        assert_eq!(data.claims.iss, "issuer");
        assert_eq!(data.claims.aud, "audience");
        assert_eq!(data.claims.sub, "subject");
        let ttl = data.claims.exp - data.claims.iat;
        assert!(
            (55..=61).contains(&ttl),
            "unexpected TTL (seconds) in claims: {ttl}"
        );
    }

    #[test]
    fn test_dashflow_ml_api_key_header_when_no_jwt() {
        let client = DashFlowMlClient::new(DashFlowMlConfig::default().with_api_key("legacy"));
        let header = client.build_auth_header().unwrap();
        assert_eq!(header, Some("Bearer legacy".to_string()));
    }

    #[test]
    fn test_dashflow_ml_no_auth_header_when_unset() {
        let client = DashFlowMlClient::new(DashFlowMlConfig::default());
        let header = client.build_auth_header().unwrap();
        assert!(header.is_none());
    }

    #[test]
    fn test_property_features_serialization() {
        let features = PropertyFeatures {
            property_type: "theorem".to_string(),
            depth: 3,
            quantifier_depth: 1,
            implication_count: 2,
            arithmetic_ops: 0,
            function_calls: 1,
            variable_count: 3,
            has_temporal: false,
            type_refs: vec!["Bool".to_string()],
            keywords: vec!["excluded_middle".to_string()],
        };

        let json = serde_json::to_string(&features).unwrap();
        assert!(json.contains("theorem"));
        assert!(json.contains("excluded_middle"));
    }

    #[test]
    fn test_predict_request_serialization() {
        let features = PropertyFeatures {
            property_type: "invariant".to_string(),
            depth: 2,
            quantifier_depth: 1,
            implication_count: 0,
            arithmetic_ops: 1,
            function_calls: 0,
            variable_count: 2,
            has_temporal: false,
            type_refs: vec!["Int".to_string()],
            keywords: vec!["positive".to_string()],
        };

        let request = PredictRequest {
            features,
            code_context: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("invariant"));
        assert!(!json.contains("code_context")); // Should be skipped when None
    }

    #[test]
    fn test_predict_response_deserialization() {
        let json = r#"{
            "backend": "Lean4",
            "confidence": 0.87,
            "alternatives": [
                ["Coq", 0.09],
                ["Isabelle", 0.03]
            ]
        }"#;

        let resp: PredictResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.backend, "Lean4");
        assert!((resp.confidence - 0.87).abs() < f64::EPSILON);
        assert_eq!(resp.alternatives.len(), 2);
        assert_eq!(resp.alternatives[0].0, "Coq");
    }

    #[test]
    fn test_feedback_verification_status() {
        let proven: FeedbackVerificationStatus = serde_json::from_str("\"Proven\"").unwrap();
        assert_eq!(proven, FeedbackVerificationStatus::Proven);

        let json = serde_json::to_string(&FeedbackVerificationStatus::Timeout).unwrap();
        assert_eq!(json, "\"Timeout\"");
    }

    #[test]
    fn test_verification_feedback_serialization() {
        let features = PropertyFeatures {
            property_type: "theorem".to_string(),
            depth: 2,
            quantifier_depth: 1,
            implication_count: 1,
            arithmetic_ops: 0,
            function_calls: 0,
            variable_count: 2,
            has_temporal: false,
            type_refs: vec!["Bool".to_string()],
            keywords: vec![],
        };

        let feedback = VerificationFeedback {
            features,
            code_context: None,
            backend: "Lean4".to_string(),
            status: FeedbackVerificationStatus::Proven,
            time_seconds: 2.34,
            proof_size: Some(156),
            error_message: None,
            tactics: vec!["simp".to_string(), "rfl".to_string()],
            timestamp: "2025-12-26T12:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&feedback).unwrap();
        assert!(json.contains("Lean4"));
        assert!(json.contains("Proven"));
        assert!(json.contains("2.34"));
        assert!(json.contains("156"));
        assert!(!json.contains("code_context")); // Skipped when None
        assert!(!json.contains("error_message")); // Skipped when None
    }

    #[test]
    fn test_feedback_response_deserialization() {
        let json = r#"{
            "recorded": true,
            "feedback_id": "fb_abc123"
        }"#;

        let resp: FeedbackResponse = serde_json::from_str(json).unwrap();
        assert!(resp.recorded);
        assert_eq!(resp.feedback_id, "fb_abc123");
    }

    #[test]
    fn test_model_status_response_deserialization() {
        let json = r#"{
            "version": "1.2.0",
            "last_trained": "2025-12-26T10:00:00Z",
            "training_samples": 10000,
            "accuracy": 0.85,
            "supported_backends": ["Lean4", "Coq", "Kani", "TlaPlus"]
        }"#;

        let resp: ModelStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.version, "1.2.0");
        assert_eq!(resp.training_samples, 10000);
        assert!((resp.accuracy - 0.85).abs() < f64::EPSILON);
        assert_eq!(resp.supported_backends.len(), 4);
    }

    #[test]
    fn test_create_feedback_helper() {
        let features = PropertyFeatures {
            property_type: "theorem".to_string(),
            depth: 1,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            function_calls: 0,
            variable_count: 1,
            has_temporal: false,
            type_refs: vec![],
            keywords: vec![],
        };

        // Test successful verification
        let feedback = create_feedback(
            features.clone(),
            "Lean4",
            true,
            1.5,
            Some(100),
            None,
            vec!["rfl".to_string()],
        );

        assert_eq!(feedback.backend, "Lean4");
        assert_eq!(feedback.status, FeedbackVerificationStatus::Proven);
        assert!((feedback.time_seconds - 1.5).abs() < f64::EPSILON);
        assert_eq!(feedback.proof_size, Some(100));

        // Test failed verification with error
        let feedback_error = create_feedback(
            features,
            "Kani",
            false,
            10.0,
            None,
            Some("Verification timeout".to_string()),
            vec![],
        );

        assert_eq!(feedback_error.status, FeedbackVerificationStatus::Error);
        assert_eq!(
            feedback_error.error_message,
            Some("Verification timeout".to_string())
        );
    }

    #[test]
    fn test_dashflow_ml_client_creation() {
        let _client = DashFlowMlClient::local(8000);
        // Test passes if client is created without panic
    }

    #[test]
    fn test_code_context_default() {
        let context = CodeContext::default();
        assert_eq!(context.language, "");
        assert_eq!(context.lines_of_code, 0);
        assert!(!context.has_unsafe);
    }

    #[test]
    fn test_code_context_serialization() {
        let context = CodeContext {
            language: "rust".to_string(),
            lines_of_code: 500,
            cyclomatic_complexity: 15,
            has_unsafe: true,
            has_concurrency: true,
            has_heap: true,
        };

        let json = serde_json::to_string(&context).unwrap();
        assert!(json.contains("rust"));
        assert!(json.contains("500"));
        assert!(json.contains("has_unsafe"));
    }

    // ============ FeedbackQueue Tests ============

    fn create_test_feedback(backend: &str) -> VerificationFeedback {
        let features = PropertyFeatures {
            property_type: "theorem".to_string(),
            depth: 1,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            function_calls: 0,
            variable_count: 1,
            has_temporal: false,
            type_refs: vec![],
            keywords: vec![],
        };

        VerificationFeedback {
            features,
            code_context: None,
            backend: backend.to_string(),
            status: FeedbackVerificationStatus::Proven,
            time_seconds: 1.0,
            proof_size: Some(10),
            error_message: None,
            tactics: vec!["rfl".to_string()],
            timestamp: "2025-12-27T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn test_feedback_queue_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("queue.json");

        let queue = FeedbackQueue::new(&path).unwrap();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_feedback_queue_push_pop() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("queue.json");

        let mut queue = FeedbackQueue::new(&path).unwrap();

        let feedback1 = create_test_feedback("Lean4");
        let feedback2 = create_test_feedback("Coq");

        queue.push(feedback1);
        assert_eq!(queue.len(), 1);

        queue.push(feedback2);
        assert_eq!(queue.len(), 2);

        let popped = queue.pop().unwrap();
        assert_eq!(popped.backend, "Lean4");
        assert_eq!(queue.len(), 1);

        let popped = queue.pop().unwrap();
        assert_eq!(popped.backend, "Coq");
        assert!(queue.is_empty());

        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_feedback_queue_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("queue.json");

        // Create queue and add feedback
        {
            let mut queue = FeedbackQueue::new(&path).unwrap();
            queue.push(create_test_feedback("Lean4"));
            queue.push(create_test_feedback("Kani"));
        }

        // Create new queue from same path - should load persisted data
        {
            let queue = FeedbackQueue::new(&path).unwrap();
            assert_eq!(queue.len(), 2);
        }
    }

    #[test]
    fn test_feedback_queue_max_size() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("queue.json");

        let mut queue = FeedbackQueue::with_max_size(&path, 3).unwrap();

        queue.push(create_test_feedback("Lean4"));
        queue.push(create_test_feedback("Coq"));
        queue.push(create_test_feedback("Kani"));
        queue.push(create_test_feedback("Isabelle")); // Should cause oldest to be dropped

        assert_eq!(queue.len(), 3);

        // First one (Lean4) should have been dropped
        let popped = queue.pop().unwrap();
        assert_eq!(popped.backend, "Coq");
    }

    #[test]
    fn test_feedback_queue_push_front() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("queue.json");

        let mut queue = FeedbackQueue::new(&path).unwrap();

        queue.push(create_test_feedback("Lean4"));
        queue.push_front(create_test_feedback("Coq"));

        // Coq should be at front now
        let popped = queue.pop().unwrap();
        assert_eq!(popped.backend, "Coq");
    }

    #[test]
    fn test_feedback_queue_clear() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("queue.json");

        let mut queue = FeedbackQueue::new(&path).unwrap();
        queue.push(create_test_feedback("Lean4"));
        queue.push(create_test_feedback("Coq"));

        assert_eq!(queue.len(), 2);
        assert!(path.exists());

        queue.clear().unwrap();

        assert!(queue.is_empty());
        assert!(!path.exists());
    }

    #[test]
    fn test_feedback_queue_stats() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("queue.json");

        let mut queue = FeedbackQueue::new(&path).unwrap();
        queue.push(create_test_feedback("Lean4"));
        queue.push(create_test_feedback("Lean4"));
        queue.push(create_test_feedback("Coq"));

        let stats = queue.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(*stats.backend_counts.get("Lean4").unwrap(), 2);
        assert_eq!(*stats.backend_counts.get("Coq").unwrap(), 1);
        assert_eq!(
            *stats
                .status_counts
                .get(&FeedbackVerificationStatus::Proven)
                .unwrap(),
            3
        );
    }

    // ============ PropertyFeatures Conversion Tests ============

    #[test]
    fn test_property_features_from_learning() {
        use dashprove_learning::similarity::PropertyFeatures as LearningFeatures;

        let learning = LearningFeatures {
            property_type: "theorem".to_string(),
            depth: 3,
            quantifier_depth: 2,
            implication_count: 1,
            arithmetic_ops: 5,
            function_calls: 3,
            variable_count: 4,
            has_temporal: true,
            type_refs: vec!["Int".to_string(), "Bool".to_string()],
            keywords: vec!["test".to_string(), "proof".to_string()],
        };

        let remote: PropertyFeatures = learning.into();

        assert_eq!(remote.property_type, "theorem");
        assert_eq!(remote.depth, 3);
        assert_eq!(remote.quantifier_depth, 2);
        assert_eq!(remote.implication_count, 1);
        assert_eq!(remote.arithmetic_ops, 5);
        assert_eq!(remote.function_calls, 3);
        assert_eq!(remote.variable_count, 4);
        assert!(remote.has_temporal);
        assert_eq!(remote.type_refs, vec!["Int", "Bool"]);
        assert_eq!(remote.keywords, vec!["test", "proof"]);
    }

    #[test]
    fn test_property_features_to_learning() {
        use dashprove_learning::similarity::PropertyFeatures as LearningFeatures;

        let remote = PropertyFeatures {
            property_type: "invariant".to_string(),
            depth: 2,
            quantifier_depth: 1,
            implication_count: 0,
            arithmetic_ops: 2,
            function_calls: 1,
            variable_count: 3,
            has_temporal: false,
            type_refs: vec!["State".to_string()],
            keywords: vec!["safety".to_string()],
        };

        let learning: LearningFeatures = remote.into();

        assert_eq!(learning.property_type, "invariant");
        assert_eq!(learning.depth, 2);
        assert_eq!(learning.quantifier_depth, 1);
        assert_eq!(learning.implication_count, 0);
        assert_eq!(learning.arithmetic_ops, 2);
        assert_eq!(learning.function_calls, 1);
        assert_eq!(learning.variable_count, 3);
        assert!(!learning.has_temporal);
        assert_eq!(learning.type_refs, vec!["State"]);
        assert_eq!(learning.keywords, vec!["safety"]);
    }

    #[test]
    fn test_property_features_roundtrip() {
        use dashprove_learning::similarity::PropertyFeatures as LearningFeatures;

        let original = LearningFeatures {
            property_type: "contract".to_string(),
            depth: 5,
            quantifier_depth: 3,
            implication_count: 2,
            arithmetic_ops: 10,
            function_calls: 7,
            variable_count: 8,
            has_temporal: false,
            type_refs: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            keywords: vec!["pre".to_string(), "post".to_string()],
        };

        // Convert to remote and back
        let remote: PropertyFeatures = original.clone().into();
        let roundtrip: LearningFeatures = remote.into();

        assert_eq!(original.property_type, roundtrip.property_type);
        assert_eq!(original.depth, roundtrip.depth);
        assert_eq!(original.quantifier_depth, roundtrip.quantifier_depth);
        assert_eq!(original.implication_count, roundtrip.implication_count);
        assert_eq!(original.arithmetic_ops, roundtrip.arithmetic_ops);
        assert_eq!(original.function_calls, roundtrip.function_calls);
        assert_eq!(original.variable_count, roundtrip.variable_count);
        assert_eq!(original.has_temporal, roundtrip.has_temporal);
        assert_eq!(original.type_refs, roundtrip.type_refs);
        assert_eq!(original.keywords, roundtrip.keywords);
    }

    #[test]
    fn test_extract_property_features() {
        use dashprove_usl::ast::{Expr, Property, Theorem};

        // Create a simple theorem property
        let property = Property::Theorem(Theorem {
            name: "test_theorem".to_string(),
            body: Expr::Bool(true),
        });

        let features = extract_property_features(&property);

        assert_eq!(features.property_type, "theorem");
        assert!(features.keywords.contains(&"theorem".to_string()));
        assert!(
            features.keywords.contains(&"test".to_string())
                || features.keywords.contains(&"test_theorem".to_string())
        );
    }
}
