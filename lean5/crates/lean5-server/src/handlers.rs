//! Request handlers for Lean5 JSON-RPC server
//!
//! Each method handler takes parsed parameters and produces a result or error.

use crate::progress::ProgressSender;
use crate::rpc::{RequestId, Response, RpcError};
use lean5_auto::bridge::SmtBridge;
use lean5_c_sem::auto::ProofStatus;
use lean5_c_sem::parser::CParser;
use lean5_elab::{elaborate, elaborate_decl};
#[cfg(test)]
use lean5_kernel::Level;
use lean5_kernel::{
    archive_cert_with_algorithm_stats, batch_verify_with_stats_progress, compress_cert,
    compress_cert_with_stats, decompress_cert, unarchive_cert_envelope,
    zstd_archive_cert_with_dict, zstd_archive_cert_with_dict_level,
    zstd_archive_cert_with_dict_stats_level, zstd_unarchive_cert_with_dict, BatchVerifyInput,
    BatchVerifyResult, BatchVerifyStats, CertArchiveEnvelope, CertDictionary, CertVerifier,
    CompressedCert, CompressionAlgorithm, CompressionStats, DictCertArchive, Environment, Expr,
    ProofCert, TypeChecker,
};
use lean5_parser::{parse_decl, parse_expr, ParseError};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, instrument};

/// Server-wide runtime metrics for monitoring and optimization
///
/// All counters use relaxed ordering for maximum throughput - exact counts
/// aren't critical, but trends are useful for AI agents.
#[derive(Debug, Default)]
pub struct ServerMetrics {
    /// Server start time (Unix timestamp in seconds)
    pub start_time_secs: AtomicU64,
    /// Total requests processed
    pub total_requests: AtomicU64,
    /// Total successful requests
    pub successful_requests: AtomicU64,
    /// Total failed requests (errors)
    pub failed_requests: AtomicU64,
    /// Total check requests
    pub check_requests: AtomicU64,
    /// Total prove requests
    pub prove_requests: AtomicU64,
    /// Total getType requests
    pub get_type_requests: AtomicU64,
    /// Total batchCheck requests
    pub batch_check_requests: AtomicU64,
    /// Total verifyCert requests
    pub verify_cert_requests: AtomicU64,
    /// Total batchVerifyCert requests
    pub batch_verify_cert_requests: AtomicU64,
    /// Total verifyC requests
    pub verify_c_requests: AtomicU64,
    /// Total items processed in batch operations
    pub batch_items_processed: AtomicU64,
    /// Total certificates verified
    pub certificates_verified: AtomicU64,
    /// Cumulative time spent in handlers (microseconds)
    pub cumulative_time_us: AtomicU64,
    /// Cumulative time spent in type checking (microseconds)
    pub type_check_time_us: AtomicU64,
    /// Cumulative time spent in certificate verification (microseconds)
    pub cert_verify_time_us: AtomicU64,
}

impl ServerMetrics {
    /// Create new metrics with current timestamp
    #[must_use]
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let metrics = Self::default();
        metrics.start_time_secs.store(now, Ordering::Relaxed);
        metrics
    }

    /// Record a request
    pub fn record_request(&self, method: &str, success: bool, duration_us: u64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successful_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_requests.fetch_add(1, Ordering::Relaxed);
        }
        self.cumulative_time_us
            .fetch_add(duration_us, Ordering::Relaxed);

        // Track by method type
        match method {
            "check" => {
                self.check_requests.fetch_add(1, Ordering::Relaxed);
            }
            "prove" => {
                self.prove_requests.fetch_add(1, Ordering::Relaxed);
            }
            "getType" => {
                self.get_type_requests.fetch_add(1, Ordering::Relaxed);
            }
            "batchCheck" => {
                self.batch_check_requests.fetch_add(1, Ordering::Relaxed);
            }
            "verifyCert" => {
                self.verify_cert_requests.fetch_add(1, Ordering::Relaxed);
            }
            "batchVerifyCert" => {
                self.batch_verify_cert_requests
                    .fetch_add(1, Ordering::Relaxed);
            }
            "verifyC" => {
                self.verify_c_requests.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Record batch items processed
    pub fn record_batch_items(&self, count: u64) {
        self.batch_items_processed
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Record certificates verified
    pub fn record_certs_verified(&self, count: u64) {
        self.certificates_verified
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Record type checking time
    pub fn record_type_check_time(&self, duration_us: u64) {
        self.type_check_time_us
            .fetch_add(duration_us, Ordering::Relaxed);
    }

    /// Record certificate verification time
    pub fn record_cert_verify_time(&self, duration_us: u64) {
        self.cert_verify_time_us
            .fetch_add(duration_us, Ordering::Relaxed);
    }

    /// Get uptime in seconds
    pub fn uptime_secs(&self) -> u64 {
        let start = self.start_time_secs.load(Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(start)
    }

    /// Get average request latency in microseconds
    pub fn avg_latency_us(&self) -> u64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }
        self.cumulative_time_us.load(Ordering::Relaxed) / total
    }

    /// Get requests per second (based on uptime)
    pub fn requests_per_second(&self) -> f64 {
        let uptime = self.uptime_secs();
        if uptime == 0 {
            return 0.0;
        }
        self.total_requests.load(Ordering::Relaxed) as f64 / uptime as f64
    }

    /// Get success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        self.successful_requests.load(Ordering::Relaxed) as f64 / total as f64
    }
}

/// Shared server state
pub struct ServerState {
    /// The kernel environment (shared, read-mostly)
    pub env: Arc<RwLock<Environment>>,
    /// Default timeout for operations (ms)
    pub default_timeout_ms: u64,
    /// GPU acceleration enabled
    pub gpu_enabled: bool,
    /// Number of worker threads for batch operations (0 = auto/Rayon default)
    pub worker_threads: usize,
    /// Server-wide metrics
    pub metrics: ServerMetrics,
}

impl ServerState {
    /// Create a new server state
    #[must_use]
    pub fn new() -> Self {
        Self {
            env: Arc::new(RwLock::new(Environment::new())),
            default_timeout_ms: 5000,
            gpu_enabled: false,
            worker_threads: 0,
            metrics: ServerMetrics::new(),
        }
    }

    /// Create with GPU enabled
    #[must_use]
    pub fn with_gpu(mut self, enabled: bool) -> Self {
        self.gpu_enabled = enabled;
        self
    }

    /// Create with worker thread count
    #[must_use]
    pub fn with_worker_threads(mut self, threads: usize) -> Self {
        self.worker_threads = threads;
        self
    }
}

impl Default for ServerState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Check code request parameters
#[derive(Debug, Clone, Deserialize)]
pub struct CheckParams {
    /// Lean code to check
    pub code: String,
    /// Optional timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

/// Check code response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Whether the code is valid
    pub valid: bool,
    /// Inferred type (if expression)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inferred_type: Option<String>,
    /// Errors (if any)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<CheckError>,
    /// Time taken in milliseconds
    pub time_ms: u64,
}

/// Error from checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckError {
    /// Error message
    pub message: String,
    /// Line number (1-indexed, if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    /// Column number (1-indexed, if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<usize>,
}

/// Prove request parameters
#[derive(Debug, Clone, Deserialize)]
pub struct ProveParams {
    /// Goal to prove (Lean expression syntax)
    pub goal: String,
    /// Hypotheses to use (Lean expression syntax)
    #[serde(default)]
    pub hypotheses: Vec<String>,
    /// Optional timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

/// Prove response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProveResult {
    /// Whether a proof was found
    pub found: bool,
    /// Proof term (Lean syntax)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_term: Option<String>,
    /// Human-readable proof sketch
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_sketch: Option<String>,
    /// Method used (smt, superposition, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Time taken in milliseconds
    pub time_ms: u64,
}

/// Get type request parameters
#[derive(Debug, Clone, Deserialize)]
pub struct GetTypeParams {
    /// Expression to get type of
    pub expr: String,
}

/// Get type response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetTypeResult {
    /// The type (Lean syntax)
    #[serde(rename = "type")]
    pub type_: String,
    /// Time taken in milliseconds
    pub time_ms: u64,
}

/// Batch check request parameters
#[derive(Debug, Clone, Deserialize)]
pub struct BatchCheckParams {
    /// List of code snippets to check
    pub items: Vec<BatchCheckItem>,
    /// Whether to use GPU acceleration
    #[serde(default)]
    pub use_gpu: bool,
    /// Optional timeout for entire batch in milliseconds
    pub timeout_ms: Option<u64>,
}

/// Single item in batch check
#[derive(Debug, Clone, Deserialize)]
pub struct BatchCheckItem {
    /// Unique identifier for this item
    pub id: String,
    /// Code to check
    pub code: String,
}

/// Batch check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCheckResult {
    /// Results for each item (in same order)
    pub results: Vec<BatchCheckItemResult>,
    /// Total time in milliseconds
    pub time_ms: u64,
    /// Whether GPU was used
    pub gpu_used: bool,
}

/// Result for single batch item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCheckItemResult {
    /// Item ID (same as request)
    pub id: String,
    /// Whether valid
    pub valid: bool,
    /// Error message (if invalid)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Server info response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Server name
    pub name: String,
    /// Version
    pub version: String,
    /// Available methods
    pub methods: Vec<String>,
    /// GPU acceleration available
    pub gpu_available: bool,
}

/// Batch verify certificates request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerifyCertParams {
    /// List of certificates to verify
    pub items: Vec<BatchVerifyCertItem>,
    /// Number of threads to use (0 = auto, default)
    #[serde(default)]
    pub threads: usize,
    /// Optional timeout for entire batch in milliseconds
    pub timeout_ms: Option<u64>,
}

/// Single item in batch certificate verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerifyCertItem {
    /// Unique identifier for this item
    pub id: String,
    /// The proof certificate (JSON-encoded ProofCert)
    pub cert: ProofCert,
    /// The expression the certificate should verify
    pub expr: Expr,
}

/// Batch verify certificates response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerifyCertResult {
    /// Results for each item (in same order)
    pub results: Vec<BatchVerifyCertItemResult>,
    /// Aggregate statistics
    pub stats: BatchVerifyCertStats,
    /// Total time in milliseconds
    pub time_ms: u64,
}

/// Result for single batch certificate item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerifyCertItemResult {
    /// Item ID (same as request)
    pub id: String,
    /// Whether verification succeeded
    pub success: bool,
    /// Verified type as string (if successful)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verified_type: Option<String>,
    /// Error message (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Verification time in microseconds
    pub time_us: u64,
}

/// Aggregate statistics for batch certificate verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerifyCertStats {
    /// Total number of inputs
    pub total: usize,
    /// Number of successful verifications
    pub successful: usize,
    /// Number of failed verifications
    pub failed: usize,
    /// Total wall-clock time in microseconds
    pub wall_time_us: u64,
    /// Sum of individual verification times (useful for parallelism analysis)
    pub sum_verify_time_us: u64,
    /// Minimum verification time
    pub min_time_us: u64,
    /// Maximum verification time
    pub max_time_us: u64,
    /// Effective speedup (sum_verify_time / wall_time)
    pub speedup: f64,
}

impl From<BatchVerifyStats> for BatchVerifyCertStats {
    fn from(s: BatchVerifyStats) -> Self {
        Self {
            total: s.total,
            successful: s.successful,
            failed: s.failed,
            wall_time_us: s.wall_time_us,
            sum_verify_time_us: s.sum_verify_time_us,
            min_time_us: s.min_time_us,
            max_time_us: s.max_time_us,
            speedup: s.speedup,
        }
    }
}

// ============================================================================
// Single Certificate Verification Types
// ============================================================================

/// Verify single certificate request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyCertParams {
    /// The proof certificate to verify
    pub cert: ProofCert,
    /// The expression the certificate should verify
    pub expr: Expr,
    /// Optional timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

/// Verify single certificate response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyCertResult {
    /// Whether verification succeeded
    pub success: bool,
    /// Verified type as string (if successful)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verified_type: Option<String>,
    /// Error message (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Verification time in microseconds
    pub time_us: u64,
}

// ============================================================================
// Certificate Compression Types
// ============================================================================

/// Compress certificate request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressCertParams {
    /// The proof certificate to compress
    pub cert: ProofCert,
    /// Whether to include compression statistics
    #[serde(default)]
    pub include_stats: bool,
}

/// Compress certificate response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressCertResult {
    /// The compressed certificate
    pub compressed: CompressedCert,
    /// Compression statistics (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<CompressCertStats>,
    /// Time taken in microseconds
    pub time_us: u64,
}

/// Compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressCertStats {
    /// Number of unique expressions after sharing
    pub unique_exprs: usize,
    /// Number of unique levels after sharing
    pub unique_levels: usize,
    /// Number of unique certificates after sharing
    pub unique_certs: usize,
    /// Original size in bytes
    pub original_bytes: usize,
    /// Compressed size in bytes
    pub compressed_bytes: usize,
    /// Compression ratio
    pub ratio: f64,
}

impl From<CompressionStats> for CompressCertStats {
    fn from(s: CompressionStats) -> Self {
        Self {
            unique_exprs: s.unique_exprs,
            unique_levels: s.unique_levels,
            unique_certs: s.unique_certs,
            original_bytes: s.original_bytes,
            compressed_bytes: s.compressed_bytes,
            ratio: s.ratio,
        }
    }
}

/// Decompress certificate request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompressCertParams {
    /// The compressed certificate to decompress
    pub compressed: CompressedCert,
}

/// Decompress certificate response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompressCertResult {
    /// The decompressed proof certificate
    pub cert: ProofCert,
    /// Time taken in microseconds
    pub time_us: u64,
}

/// Archive certificate request parameters (byte-level compression)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveCertParams {
    /// The proof certificate to archive
    pub cert: ProofCert,
    /// Compression algorithm: "lz4" (default, fast) or "zstd" (better ratio)
    #[serde(default)]
    pub algorithm: Option<String>,
    /// Zstd compression level (1-22, default 3). Only used for "zstd" algorithm.
    #[serde(default)]
    pub level: Option<i32>,
    /// Whether to include compression statistics
    #[serde(default)]
    pub include_stats: bool,
}

/// Archive certificate response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveCertResult {
    /// The archived certificate as base64-encoded bytes
    pub archive: String,
    /// Compression algorithm used
    pub algorithm: String,
    /// Original size in bytes (before compression)
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Time taken in microseconds
    pub time_us: u64,
}

/// Unarchive certificate request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnarchiveCertParams {
    /// The archived certificate as base64-encoded bytes
    pub archive: String,
}

/// Unarchive certificate response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnarchiveCertResult {
    /// The restored proof certificate
    pub cert: ProofCert,
    /// Compression algorithm that was used
    pub algorithm: String,
    /// Time taken in microseconds
    pub time_us: u64,
}

// ============================================================================
// Dictionary Compression Types
// ============================================================================

/// Train dictionary request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainDictParams {
    /// Sample certificates to train the dictionary from (minimum 5)
    pub samples: Vec<ProofCert>,
    /// Maximum dictionary size in bytes (default 32KB)
    #[serde(default)]
    pub max_size: Option<usize>,
    /// Target compression level (default 3)
    #[serde(default)]
    pub level: Option<i32>,
}

/// Train dictionary response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainDictResult {
    /// The trained dictionary as base64-encoded bytes
    pub dictionary: String,
    /// Dictionary ID for validation
    pub dict_id: u32,
    /// Number of samples used for training
    pub sample_count: usize,
    /// Dictionary size in bytes
    pub size: usize,
    /// Target compression level
    pub target_level: i32,
    /// Time taken in microseconds
    pub time_us: u64,
}

/// Archive certificate with dictionary request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveCertWithDictParams {
    /// The proof certificate to archive
    pub cert: ProofCert,
    /// The dictionary as base64-encoded bytes (from trainDict)
    pub dictionary: String,
    /// Compression level (optional, uses dictionary's target level by default)
    #[serde(default)]
    pub level: Option<i32>,
    /// Whether to include compression statistics
    #[serde(default)]
    pub include_stats: bool,
}

/// Archive certificate with dictionary response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveCertWithDictResult {
    /// The archived certificate as base64-encoded bytes
    pub archive: String,
    /// Dictionary ID used for compression
    pub dict_id: u32,
    /// Original size in bytes (before any compression)
    pub original_size: usize,
    /// After structure sharing (intermediate)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structure_shared_size: Option<usize>,
    /// Final compressed size in bytes
    pub compressed_size: usize,
    /// Total compression ratio (original / compressed)
    pub compression_ratio: f64,
    /// Compression level used
    pub compression_level: i32,
    /// Time taken in microseconds
    pub time_us: u64,
}

/// Unarchive certificate with dictionary request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnarchiveCertWithDictParams {
    /// The archived certificate as base64-encoded bytes
    pub archive: String,
    /// The dictionary as base64-encoded bytes (must match the one used for compression)
    pub dictionary: String,
}

/// Unarchive certificate with dictionary response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnarchiveCertWithDictResult {
    /// The restored proof certificate
    pub cert: ProofCert,
    /// Dictionary ID that was used
    pub dict_id: u32,
    /// Time taken in microseconds
    pub time_us: u64,
}

// ============================================================================
// Metrics Types
// ============================================================================

/// Get server metrics response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetMetricsResult {
    /// Server uptime in seconds
    pub uptime_secs: u64,
    /// Total requests processed
    pub total_requests: u64,
    /// Total successful requests
    pub successful_requests: u64,
    /// Total failed requests
    pub failed_requests: u64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Average request latency in microseconds
    pub avg_latency_us: u64,
    /// Requests per second (based on uptime)
    pub requests_per_second: f64,
    /// Per-method request counts
    pub method_counts: MethodCounts,
    /// Aggregate batch statistics
    pub batch_stats: BatchStats,
    /// Timing breakdown (microseconds)
    pub timing: TimingStats,
}

/// Per-method request counts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodCounts {
    /// Check requests
    pub check: u64,
    /// Prove requests
    pub prove: u64,
    /// GetType requests
    pub get_type: u64,
    /// BatchCheck requests
    pub batch_check: u64,
    /// VerifyCert requests
    pub verify_cert: u64,
    /// BatchVerifyCert requests
    pub batch_verify_cert: u64,
    /// VerifyC requests
    pub verify_c: u64,
}

/// Aggregate batch operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    /// Total items processed in batch operations
    pub items_processed: u64,
    /// Total certificates verified
    pub certificates_verified: u64,
}

/// Timing breakdown statistics (microseconds)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    /// Cumulative time in handlers
    pub cumulative_handler_time_us: u64,
    /// Cumulative time in type checking
    pub type_check_time_us: u64,
    /// Cumulative time in certificate verification
    pub cert_verify_time_us: u64,
}

// ============================================================================
// Handler Implementations
// ============================================================================

/// Handle the "check" method
#[instrument(skip(state))]
pub async fn handle_check(state: &ServerState, id: RequestId, params: CheckParams) -> Response {
    let start = Instant::now();
    let timeout = Duration::from_millis(params.timeout_ms.unwrap_or(state.default_timeout_ms));

    // Try to complete within timeout
    let result = tokio::time::timeout(timeout, async {
        check_code_impl(state, &params.code).await
    })
    .await;

    let elapsed_us = start.elapsed().as_micros() as u64;
    let elapsed_ms = elapsed_us / 1000;

    match result {
        Ok(Ok(mut check_result)) => {
            check_result.time_ms = elapsed_ms;
            let success = check_result.valid;
            state.metrics.record_request("check", success, elapsed_us);
            Response::success_typed(id.clone(), &check_result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Ok(Err(e)) => {
            state.metrics.record_request("check", false, elapsed_us);
            Response::error(id, e)
        }
        Err(_) => {
            state.metrics.record_request("check", false, elapsed_us);
            Response::error(id, RpcError::timeout(timeout.as_millis() as u64))
        }
    }
}

async fn check_code_impl(state: &ServerState, code: &str) -> Result<CheckResult, RpcError> {
    // First try to parse as expression
    if let Ok(surface_expr) = parse_expr(code) {
        let env = state.env.read().await;

        match elaborate(&env, &surface_expr) {
            Ok(expr) => {
                let mut tc = TypeChecker::new(&env);
                match tc.infer_type(&expr) {
                    Ok(type_) => {
                        return Ok(CheckResult {
                            valid: true,
                            inferred_type: Some(format_expr(&type_)),
                            errors: vec![],
                            time_ms: 0,
                        });
                    }
                    Err(e) => {
                        return Ok(CheckResult {
                            valid: false,
                            inferred_type: None,
                            errors: vec![CheckError {
                                message: format!("Type error: {e}"),
                                line: None,
                                column: None,
                            }],
                            time_ms: 0,
                        });
                    }
                }
            }
            Err(e) => {
                return Ok(CheckResult {
                    valid: false,
                    inferred_type: None,
                    errors: vec![CheckError {
                        message: format!("Elaboration error: {e}"),
                        line: None,
                        column: None,
                    }],
                    time_ms: 0,
                });
            }
        }
    }

    // Try to parse as declaration
    match parse_decl(code) {
        Ok(surface_decl) => {
            let env = state.env.read().await;

            match elaborate_decl(&env, &surface_decl) {
                Ok(_decl) => Ok(CheckResult {
                    valid: true,
                    inferred_type: None,
                    errors: vec![],
                    time_ms: 0,
                }),
                Err(e) => Ok(CheckResult {
                    valid: false,
                    inferred_type: None,
                    errors: vec![CheckError {
                        message: format!("Elaboration error: {e}"),
                        line: None,
                        column: None,
                    }],
                    time_ms: 0,
                }),
            }
        }
        Err(e) => Ok(CheckResult {
            valid: false,
            inferred_type: None,
            errors: vec![CheckError {
                message: format_parse_error(&e),
                line: parse_error_line(&e),
                column: parse_error_col(&e),
            }],
            time_ms: 0,
        }),
    }
}

/// Handle the "prove" method
#[instrument(skip(state))]
pub async fn handle_prove(state: &ServerState, id: RequestId, params: ProveParams) -> Response {
    let start = Instant::now();
    let timeout = Duration::from_millis(params.timeout_ms.unwrap_or(state.default_timeout_ms));

    let result = tokio::time::timeout(timeout, async { prove_impl(state, &params).await }).await;

    let elapsed_us = start.elapsed().as_micros() as u64;
    let elapsed_ms = elapsed_us / 1000;

    match result {
        Ok(Ok(mut prove_result)) => {
            prove_result.time_ms = elapsed_ms;
            let success = prove_result.found;
            state.metrics.record_request("prove", success, elapsed_us);
            Response::success_typed(id.clone(), &prove_result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Ok(Err(e)) => {
            state.metrics.record_request("prove", false, elapsed_us);
            Response::error(id, e)
        }
        Err(_) => {
            state.metrics.record_request("prove", false, elapsed_us);
            Response::error(id, RpcError::timeout(timeout.as_millis() as u64))
        }
    }
}

async fn prove_impl(state: &ServerState, params: &ProveParams) -> Result<ProveResult, RpcError> {
    // Parse the goal
    let goal_surface = parse_expr(&params.goal)
        .map_err(|e| RpcError::lean_parse_error(format!("Failed to parse goal: {e}")))?;

    let env = state.env.read().await;

    // Elaborate the goal
    let goal_expr = elaborate(&env, &goal_surface)
        .map_err(|e| RpcError::elaboration_error(format!("Failed to elaborate goal: {e}")))?;

    // Create SMT bridge and add hypotheses
    let mut bridge = SmtBridge::new(&env);

    for (i, hyp_str) in params.hypotheses.iter().enumerate() {
        let hyp_surface = parse_expr(hyp_str).map_err(|e| {
            RpcError::lean_parse_error(format!("Failed to parse hypothesis {i}: {e}"))
        })?;

        let hyp_expr = elaborate(&env, &hyp_surface).map_err(|e| {
            RpcError::elaboration_error(format!("Failed to elaborate hypothesis {i}: {e}"))
        })?;

        bridge.add_hypothesis(&hyp_expr);
    }

    // Try to prove using SMT
    match bridge.prove(&goal_expr) {
        Some(proof_result) => Ok(ProveResult {
            found: true,
            proof_term: proof_result.proof_term.as_ref().map(format_expr),
            proof_sketch: Some(proof_result.proof_sketch),
            method: Some("smt".to_string()),
            time_ms: 0,
        }),
        None => Ok(ProveResult {
            found: false,
            proof_term: None,
            proof_sketch: None,
            method: None,
            time_ms: 0,
        }),
    }
}

/// Handle the "getType" method
#[instrument(skip(state))]
pub async fn handle_get_type(
    state: &ServerState,
    id: RequestId,
    params: GetTypeParams,
) -> Response {
    let start = Instant::now();

    let result = get_type_impl(state, &params.expr).await;
    let elapsed_us = start.elapsed().as_micros() as u64;
    let elapsed_ms = elapsed_us / 1000;

    match result {
        Ok(mut type_result) => {
            type_result.time_ms = elapsed_ms;
            state.metrics.record_request("getType", true, elapsed_us);
            Response::success_typed(id.clone(), &type_result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Err(e) => {
            state.metrics.record_request("getType", false, elapsed_us);
            Response::error(id, e)
        }
    }
}

async fn get_type_impl(state: &ServerState, expr_str: &str) -> Result<GetTypeResult, RpcError> {
    let surface_expr = parse_expr(expr_str)
        .map_err(|e| RpcError::lean_parse_error(format!("Parse error: {e}")))?;

    let env = state.env.read().await;

    let expr = elaborate(&env, &surface_expr)
        .map_err(|e| RpcError::elaboration_error(format!("Elaboration error: {e}")))?;

    let mut tc = TypeChecker::new(&env);
    let type_ = tc
        .infer_type(&expr)
        .map_err(|e| RpcError::type_error(format!("Type error: {e}")))?;

    Ok(GetTypeResult {
        type_: format_expr(&type_),
        time_ms: 0,
    })
}

/// Handle the "batchCheck" method
#[instrument(skip(state))]
pub async fn handle_batch_check(
    state: &ServerState,
    id: RequestId,
    params: BatchCheckParams,
    progress: Option<ProgressSender>,
) -> Response {
    let start = Instant::now();
    let item_count = params.items.len() as u64;
    let timeout = Duration::from_millis(params.timeout_ms.unwrap_or(state.default_timeout_ms * 10));

    let result = tokio::time::timeout(timeout, async {
        batch_check_impl(state, &params, progress.clone()).await
    })
    .await;

    let elapsed_us = start.elapsed().as_micros() as u64;
    let elapsed_ms = elapsed_us / 1000;

    match result {
        Ok(Ok(mut batch_result)) => {
            batch_result.time_ms = elapsed_ms;
            let all_valid = batch_result.results.iter().all(|r| r.valid);
            state
                .metrics
                .record_request("batchCheck", all_valid, elapsed_us);
            state.metrics.record_batch_items(item_count);
            Response::success_typed(id.clone(), &batch_result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Ok(Err(e)) => {
            state
                .metrics
                .record_request("batchCheck", false, elapsed_us);
            Response::error(id, e)
        }
        Err(_) => {
            state
                .metrics
                .record_request("batchCheck", false, elapsed_us);
            Response::error(id, RpcError::timeout(timeout.as_millis() as u64))
        }
    }
}

async fn batch_check_impl(
    state: &ServerState,
    params: &BatchCheckParams,
    progress: Option<ProgressSender>,
) -> Result<BatchCheckResult, RpcError> {
    let use_gpu = params.use_gpu && state.gpu_enabled;

    if use_gpu {
        debug!("GPU flag set but GPU acceleration is not used for batch check");
        // Note: GPU acceleration was benchmarked and found to be ~100x slower than CPU
        // for type checking due to dispatch overhead and CPU-friendly control flow.
        // CPU parallelism via Rayon (lean5-gpu::parallel) is preferred.
        // The use_gpu flag is retained for API compatibility but has no effect.
    }

    let mut results = Vec::with_capacity(params.items.len());

    if let Some(progress) = progress.as_ref() {
        progress
            .notify(
                format!("Batch check started ({} items)", params.items.len()),
                Some(0),
                None,
            )
            .await;
    }

    let total = params.items.len();

    for (idx, item) in params.items.iter().enumerate() {
        let check_result = check_code_impl(state, &item.code).await;
        let valid = check_result.as_ref().is_ok_and(|r| r.valid);
        let error_msg = check_result
            .as_ref()
            .ok()
            .and_then(|r| r.errors.first().map(|e| e.message.clone()))
            .or_else(|| check_result.err().map(|e| e.message));

        results.push(BatchCheckItemResult {
            id: item.id.clone(),
            valid,
            error: error_msg.clone(),
        });

        if let Some(progress) = progress.as_ref() {
            let percentage = if total == 0 {
                100
            } else {
                (((idx + 1) * 100) / total).min(100)
            } as u8;

            progress
                .notify(
                    format!("Checked {}/{} ({})", idx + 1, total, item.id),
                    Some(percentage),
                    Some(json!({
                        "id": item.id,
                        "valid": valid,
                        "error": error_msg,
                    })),
                )
                .await;
        }
    }

    Ok(BatchCheckResult {
        results,
        time_ms: 0,
        gpu_used: false,
    })
}

/// Handle the "batchVerifyCert" method
///
/// Verifies multiple proof certificates in parallel using rayon.
/// This is the high-throughput API for validating pre-computed proof certificates.
#[instrument(skip(state))]
pub async fn handle_batch_verify_cert(
    state: &ServerState,
    id: RequestId,
    params: BatchVerifyCertParams,
    progress: Option<ProgressSender>,
) -> Response {
    let start = Instant::now();
    let cert_count = params.items.len() as u64;
    let timeout = Duration::from_millis(params.timeout_ms.unwrap_or(state.default_timeout_ms * 10));

    let result = tokio::time::timeout(timeout, async {
        batch_verify_cert_impl(state, &params, progress.clone()).await
    })
    .await;

    let elapsed_us = start.elapsed().as_micros() as u64;
    let elapsed_ms = elapsed_us / 1000;

    match result {
        Ok(Ok(mut verify_result)) => {
            verify_result.time_ms = elapsed_ms;
            let all_success = verify_result.results.iter().all(|r| r.success);
            state
                .metrics
                .record_request("batchVerifyCert", all_success, elapsed_us);
            state.metrics.record_certs_verified(cert_count);
            state.metrics.record_cert_verify_time(elapsed_us);
            Response::success_typed(id.clone(), &verify_result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Ok(Err(e)) => {
            state
                .metrics
                .record_request("batchVerifyCert", false, elapsed_us);
            Response::error(id, e)
        }
        Err(_) => {
            state
                .metrics
                .record_request("batchVerifyCert", false, elapsed_us);
            Response::error(id, RpcError::timeout(timeout.as_millis() as u64))
        }
    }
}

async fn batch_verify_cert_impl(
    state: &ServerState,
    params: &BatchVerifyCertParams,
    progress: Option<ProgressSender>,
) -> Result<BatchVerifyCertResult, RpcError> {
    let total = params.items.len();

    if let Some(ref progress) = progress {
        progress
            .notify(
                format!("Batch verify started ({total} certs)"),
                Some(0),
                None,
            )
            .await;
    }

    // Convert to kernel BatchVerifyInput
    let inputs: Vec<BatchVerifyInput> = params
        .items
        .iter()
        .map(|item| BatchVerifyInput::new(item.id.clone(), item.cert.clone(), item.expr.clone()))
        .collect();

    let env = state.env.read().await;

    // Determine thread count: request param > server config > auto (0)
    let num_threads = if params.threads > 0 {
        params.threads
    } else {
        state.worker_threads
    };

    // Forward per-item completions over an unbounded channel when progress is requested
    let (progress_tx, progress_rx) = if progress.is_some() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<BatchVerifyResult>();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    // Stream progress notifications as results arrive
    let progress_forwarder = if let (Some(progress), Some(mut rx)) = (progress.clone(), progress_rx)
    {
        Some(tokio::spawn(async move {
            let mut completed = 0usize;
            while let Some(result) = rx.recv().await {
                completed += 1;
                let percentage = if total == 0 {
                    100
                } else {
                    (((completed * 100) / total).min(100)) as u8
                };

                let details = json!({
                    "id": result.id,
                    "success": result.success,
                    "verified_type": result.verified_type.as_ref().map(format_expr),
                    "error": result.error,
                    "time_us": result.time_us,
                });

                progress
                    .notify(
                        format!("Verified {}/{} ({})", completed, total, result.id),
                        Some(percentage),
                        Some(details),
                    )
                    .await;
            }
        }))
    } else {
        None
    };

    // Use tokio spawn_blocking to run the CPU-bound parallel verification
    let env_clone = env.clone();
    let progress_callback = progress_tx.map(|tx| {
        move |result: &BatchVerifyResult| {
            let _ = tx.send(result.clone());
        }
    });
    let threads = num_threads;
    let (results, stats) = tokio::task::spawn_blocking(move || match progress_callback {
        Some(cb) => batch_verify_with_stats_progress(&env_clone, inputs, threads, cb),
        None => batch_verify_with_stats_progress(&env_clone, inputs, threads, |_| {}),
    })
    .await
    .map_err(|e| RpcError::internal_error(format!("Task join error: {e}")))?;

    if let Some(task) = progress_forwarder {
        let _ = task.await;
    }

    // Convert results to API types
    let api_results: Vec<BatchVerifyCertItemResult> = results
        .into_iter()
        .map(|r| BatchVerifyCertItemResult {
            id: r.id,
            success: r.success,
            verified_type: r.verified_type.map(|t| format_expr(&t)),
            error: r.error,
            time_us: r.time_us,
        })
        .collect();

    if let Some(ref progress) = progress {
        progress
            .notify(
                format!(
                    "Batch verify complete: {}/{} succeeded",
                    stats.successful, stats.total
                ),
                Some(100),
                Some(json!({
                    "total": stats.total,
                    "successful": stats.successful,
                    "failed": stats.failed,
                    "speedup": stats.speedup,
                })),
            )
            .await;
    }

    Ok(BatchVerifyCertResult {
        results: api_results,
        stats: stats.into(),
        time_ms: 0,
    })
}

/// Handle the "verifyCert" method
///
/// Verifies a single proof certificate against an expression.
/// This is the lightweight API for verifying individual certificates,
/// as opposed to batchVerifyCert which is optimized for high-throughput parallel verification.
#[instrument(skip(state))]
pub async fn handle_verify_cert(
    state: &ServerState,
    id: RequestId,
    params: VerifyCertParams,
) -> Response {
    let start = Instant::now();
    let timeout = Duration::from_millis(params.timeout_ms.unwrap_or(state.default_timeout_ms));

    let result = tokio::time::timeout(timeout, async {
        let env = state.env.read().await;
        let mut verifier = CertVerifier::new(&env);
        verifier.verify(&params.cert, &params.expr)
    })
    .await;

    let time_us = start.elapsed().as_micros() as u64;

    match result {
        Ok(Ok(verified_type)) => {
            state.metrics.record_request("verifyCert", true, time_us);
            state.metrics.record_certs_verified(1);
            state.metrics.record_cert_verify_time(time_us);
            let result = VerifyCertResult {
                success: true,
                verified_type: Some(format_expr(&verified_type)),
                error: None,
                time_us,
            };
            Response::success_typed(id.clone(), &result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Ok(Err(e)) => {
            state.metrics.record_request("verifyCert", false, time_us);
            let result = VerifyCertResult {
                success: false,
                verified_type: None,
                error: Some(format!("{e:?}")),
                time_us,
            };
            Response::success_typed(id.clone(), &result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Err(_) => {
            state.metrics.record_request("verifyCert", false, time_us);
            Response::error(id, RpcError::timeout(timeout.as_millis() as u64))
        }
    }
}

/// Handle the "compressCert" method
///
/// Compresses a proof certificate using structure-sharing compression.
/// This is an in-memory compression that exploits shared subexpressions.
#[instrument(skip(_state))]
pub async fn handle_compress_cert(
    _state: &ServerState,
    id: RequestId,
    params: CompressCertParams,
) -> Response {
    let start = Instant::now();

    let result = if params.include_stats {
        let (compressed, stats) = compress_cert_with_stats(&params.cert);
        CompressCertResult {
            compressed,
            stats: Some(stats.into()),
            time_us: start.elapsed().as_micros() as u64,
        }
    } else {
        let compressed = compress_cert(&params.cert);
        CompressCertResult {
            compressed,
            stats: None,
            time_us: start.elapsed().as_micros() as u64,
        }
    };

    Response::success_typed(id.clone(), &result)
        .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
}

/// Handle the "decompressCert" method
///
/// Decompresses a structure-sharing compressed certificate back to ProofCert.
#[instrument(skip(_state))]
pub async fn handle_decompress_cert(
    _state: &ServerState,
    id: RequestId,
    params: DecompressCertParams,
) -> Response {
    let start = Instant::now();

    match decompress_cert(&params.compressed) {
        Ok(cert) => {
            let result = DecompressCertResult {
                cert,
                time_us: start.elapsed().as_micros() as u64,
            };
            Response::success_typed(id.clone(), &result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Err(e) => Response::error(
            id,
            RpcError::internal_error(format!("Decompression failed: {e:?}")),
        ),
    }
}

/// Handle the "archiveCert" method
///
/// Archives a proof certificate to a portable byte format using LZ4 or Zstd compression.
/// The result is a base64-encoded string suitable for storage or transmission.
#[instrument(skip(_state))]
pub async fn handle_archive_cert(
    _state: &ServerState,
    id: RequestId,
    params: ArchiveCertParams,
) -> Response {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use lean5_kernel::ArchiveVariantStats;

    let start = Instant::now();

    // Parse algorithm
    let algorithm = match params.algorithm.as_deref().unwrap_or("lz4") {
        "lz4" => CompressionAlgorithm::Lz4,
        "zstd" | "zstd_default" => CompressionAlgorithm::ZstdDefault,
        "zstd_high" => CompressionAlgorithm::ZstdHigh,
        "zstd_max" => CompressionAlgorithm::ZstdMax,
        other => {
            return Response::error(
                id,
                RpcError::invalid_params(format!(
                    "Unknown algorithm '{other}'. Use 'lz4', 'zstd', 'zstd_high', or 'zstd_max'."
                )),
            );
        }
    };

    // Archive with stats
    let result = archive_cert_with_algorithm_stats(&params.cert, algorithm);

    match result {
        Ok((envelope, stats)) => {
            // Serialize envelope to bytes
            let envelope_bytes =
                bincode::serialize(&envelope).expect("Envelope serialization should not fail");

            let archive_base64 = STANDARD.encode(&envelope_bytes);

            // Extract stats based on variant
            let (original_size, compressed_size, compression_ratio, algo_name) = match stats {
                ArchiveVariantStats::Lz4(s) => (
                    s.original_cert_bytes,
                    s.archive_bytes,
                    s.total_ratio,
                    "lz4".to_string(),
                ),
                ArchiveVariantStats::Zstd(s) => (
                    s.original_cert_bytes,
                    s.archive_bytes,
                    s.total_ratio,
                    "zstd".to_string(),
                ),
            };

            let api_result = ArchiveCertResult {
                archive: archive_base64,
                algorithm: algo_name,
                original_size,
                compressed_size,
                compression_ratio,
                time_us: start.elapsed().as_micros() as u64,
            };

            Response::success_typed(id.clone(), &api_result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Err(e) => Response::error(
            id,
            RpcError::internal_error(format!("Archive failed: {e:?}")),
        ),
    }
}

/// Handle the "unarchiveCert" method
///
/// Restores a proof certificate from a base64-encoded archive.
#[instrument(skip(_state))]
pub async fn handle_unarchive_cert(
    _state: &ServerState,
    id: RequestId,
    params: UnarchiveCertParams,
) -> Response {
    use base64::{engine::general_purpose::STANDARD, Engine};

    let start = Instant::now();

    // Decode base64
    let envelope_bytes = match STANDARD.decode(&params.archive) {
        Ok(bytes) => bytes,
        Err(e) => {
            return Response::error(id, RpcError::invalid_params(format!("Invalid base64: {e}")));
        }
    };

    // Deserialize envelope
    let envelope: CertArchiveEnvelope = match bincode::deserialize(&envelope_bytes) {
        Ok(env) => env,
        Err(e) => {
            return Response::error(
                id,
                RpcError::invalid_params(format!("Invalid archive format: {e}")),
            );
        }
    };

    // Determine algorithm from envelope
    let algorithm = match &envelope {
        CertArchiveEnvelope::Lz4(_) => "lz4".to_string(),
        CertArchiveEnvelope::Zstd(_) => "zstd".to_string(),
    };

    // Unarchive
    match unarchive_cert_envelope(&envelope) {
        Ok(cert) => {
            let result = UnarchiveCertResult {
                cert,
                algorithm,
                time_us: start.elapsed().as_micros() as u64,
            };
            Response::success_typed(id.clone(), &result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Err(e) => Response::error(
            id,
            RpcError::internal_error(format!("Unarchive failed: {e:?}")),
        ),
    }
}

// ============================================================================
// Dictionary Compression Handlers
// ============================================================================

/// Handle the "trainDict" method
///
/// Trains a compression dictionary from sample certificates.
/// The dictionary can then be used with archiveCertWithDict for improved compression.
#[instrument(skip(_state))]
pub async fn handle_train_dict(
    _state: &ServerState,
    id: RequestId,
    params: TrainDictParams,
) -> Response {
    use base64::{engine::general_purpose::STANDARD, Engine};

    let start = Instant::now();

    let max_size = params.max_size.unwrap_or(CertDictionary::DEFAULT_SIZE);
    let level = params.level.unwrap_or(3);

    // Train the dictionary
    match CertDictionary::train(&params.samples, max_size, level) {
        Ok(dict) => {
            // Serialize dictionary for transport
            let dict_bytes = match bincode::serialize(&dict) {
                Ok(bytes) => bytes,
                Err(e) => {
                    return Response::error(
                        id,
                        RpcError::internal_error(format!("Failed to serialize dictionary: {e}")),
                    );
                }
            };

            let result = TrainDictResult {
                dictionary: STANDARD.encode(&dict_bytes),
                dict_id: dict.dict_id,
                sample_count: dict.sample_count,
                size: dict.size(),
                target_level: dict.target_level,
                time_us: start.elapsed().as_micros() as u64,
            };
            Response::success_typed(id.clone(), &result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Err(e) => Response::error(
            id,
            RpcError::invalid_params(format!("Dictionary training failed: {e}")),
        ),
    }
}

/// Handle the "archiveCertWithDict" method
///
/// Archives a certificate using dictionary-based Zstd compression.
/// The dictionary must have been created with trainDict.
#[instrument(skip(_state))]
pub async fn handle_archive_cert_with_dict(
    _state: &ServerState,
    id: RequestId,
    params: ArchiveCertWithDictParams,
) -> Response {
    use base64::{engine::general_purpose::STANDARD, Engine};

    let start = Instant::now();

    // Decode and deserialize dictionary
    let dict_bytes = match STANDARD.decode(&params.dictionary) {
        Ok(bytes) => bytes,
        Err(e) => {
            return Response::error(
                id,
                RpcError::invalid_params(format!("Invalid dictionary base64: {e}")),
            );
        }
    };

    let dict: CertDictionary = match bincode::deserialize(&dict_bytes) {
        Ok(d) => d,
        Err(e) => {
            return Response::error(
                id,
                RpcError::invalid_params(format!("Invalid dictionary format: {e}")),
            );
        }
    };

    // Get original size for stats
    let original_bytes = match bincode::serialize(&params.cert) {
        Ok(b) => b,
        Err(e) => {
            return Response::error(
                id,
                RpcError::internal_error(format!("Failed to measure cert size: {e}")),
            );
        }
    };
    let original_size = original_bytes.len();

    // Archive with dictionary
    if params.include_stats {
        let level = params.level.unwrap_or(dict.target_level);
        match zstd_archive_cert_with_dict_stats_level(&params.cert, &dict, level) {
            Ok((archive, stats)) => {
                // Serialize archive for transport
                let archive_bytes = match bincode::serialize(&archive) {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        return Response::error(
                            id,
                            RpcError::internal_error(format!("Failed to serialize archive: {e}")),
                        );
                    }
                };

                let result = ArchiveCertWithDictResult {
                    archive: STANDARD.encode(&archive_bytes),
                    dict_id: archive.dict_id,
                    original_size: stats.original_cert_bytes,
                    structure_shared_size: Some(stats.structure_shared_bytes),
                    compressed_size: stats.archive_bytes,
                    compression_ratio: stats.total_ratio,
                    compression_level: stats.compression_level,
                    time_us: start.elapsed().as_micros() as u64,
                };
                Response::success_typed(id.clone(), &result).unwrap_or_else(|e| {
                    Response::error(id, RpcError::internal_error(e.to_string()))
                })
            }
            Err(e) => Response::error(
                id,
                RpcError::internal_error(format!("Dictionary archive failed: {e}")),
            ),
        }
    } else {
        let archive_result = if let Some(level) = params.level {
            zstd_archive_cert_with_dict_level(&params.cert, &dict, level)
        } else {
            zstd_archive_cert_with_dict(&params.cert, &dict)
        };

        match archive_result {
            Ok(archive) => {
                // Serialize archive for transport
                let archive_bytes = match bincode::serialize(&archive) {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        return Response::error(
                            id,
                            RpcError::internal_error(format!("Failed to serialize archive: {e}")),
                        );
                    }
                };

                let compressed_size = archive_bytes.len();
                let compression_ratio = if compressed_size > 0 {
                    original_size as f64 / compressed_size as f64
                } else {
                    0.0
                };

                let result = ArchiveCertWithDictResult {
                    archive: STANDARD.encode(&archive_bytes),
                    dict_id: archive.dict_id,
                    original_size,
                    structure_shared_size: None,
                    compressed_size,
                    compression_ratio,
                    compression_level: archive.compression_level,
                    time_us: start.elapsed().as_micros() as u64,
                };
                Response::success_typed(id.clone(), &result).unwrap_or_else(|e| {
                    Response::error(id, RpcError::internal_error(e.to_string()))
                })
            }
            Err(e) => Response::error(
                id,
                RpcError::internal_error(format!("Dictionary archive failed: {e}")),
            ),
        }
    }
}

/// Handle the "unarchiveCertWithDict" method
///
/// Restores a certificate from a dictionary-compressed archive.
/// The same dictionary used for compression must be provided.
#[instrument(skip(_state))]
pub async fn handle_unarchive_cert_with_dict(
    _state: &ServerState,
    id: RequestId,
    params: UnarchiveCertWithDictParams,
) -> Response {
    use base64::{engine::general_purpose::STANDARD, Engine};

    let start = Instant::now();

    // Decode and deserialize dictionary
    let dict_bytes = match STANDARD.decode(&params.dictionary) {
        Ok(bytes) => bytes,
        Err(e) => {
            return Response::error(
                id,
                RpcError::invalid_params(format!("Invalid dictionary base64: {e}")),
            );
        }
    };

    let dict: CertDictionary = match bincode::deserialize(&dict_bytes) {
        Ok(d) => d,
        Err(e) => {
            return Response::error(
                id,
                RpcError::invalid_params(format!("Invalid dictionary format: {e}")),
            );
        }
    };

    // Decode and deserialize archive
    let archive_bytes = match STANDARD.decode(&params.archive) {
        Ok(bytes) => bytes,
        Err(e) => {
            return Response::error(
                id,
                RpcError::invalid_params(format!("Invalid archive base64: {e}")),
            );
        }
    };

    let archive: DictCertArchive = match bincode::deserialize(&archive_bytes) {
        Ok(a) => a,
        Err(e) => {
            return Response::error(
                id,
                RpcError::invalid_params(format!("Invalid archive format: {e}")),
            );
        }
    };

    // Unarchive with dictionary
    match zstd_unarchive_cert_with_dict(&archive, &dict) {
        Ok(cert) => {
            let result = UnarchiveCertWithDictResult {
                cert,
                dict_id: dict.dict_id,
                time_us: start.elapsed().as_micros() as u64,
            };
            Response::success_typed(id.clone(), &result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Err(e) => Response::error(
            id,
            RpcError::internal_error(format!("Dictionary unarchive failed: {e}")),
        ),
    }
}

/// Handle the "serverInfo" method
pub async fn handle_server_info(state: &ServerState, id: RequestId) -> Response {
    let info = ServerInfo {
        name: "lean5-server".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        methods: vec![
            "check".to_string(),
            "prove".to_string(),
            "getType".to_string(),
            "batchCheck".to_string(),
            "verifyCert".to_string(),
            "batchVerifyCert".to_string(),
            "compressCert".to_string(),
            "decompressCert".to_string(),
            "archiveCert".to_string(),
            "unarchiveCert".to_string(),
            "trainDict".to_string(),
            "archiveCertWithDict".to_string(),
            "unarchiveCertWithDict".to_string(),
            "verifyC".to_string(),
            "serverInfo".to_string(),
            "saveEnvironment".to_string(),
            "loadEnvironment".to_string(),
            "getEnvironment".to_string(),
            "getConfig".to_string(),
            "getMetrics".to_string(),
        ],
        gpu_available: state.gpu_enabled,
    };

    Response::success_typed(id.clone(), &info)
        .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
}

/// Get configuration request parameters (empty, no params needed)
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GetConfigParams {}

/// Server configuration response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetConfigResult {
    /// GPU acceleration enabled
    pub gpu_enabled: bool,
    /// Default timeout for operations (milliseconds)
    pub default_timeout_ms: u64,
    /// Number of worker threads for batch operations (0 = auto)
    pub worker_threads: usize,
    /// Effective thread count (actual threads used when worker_threads=0)
    pub effective_threads: usize,
}

/// Handle the "getConfig" method
///
/// Returns current server configuration including thread settings.
#[instrument(skip(state))]
pub async fn handle_get_config(state: &ServerState, id: RequestId) -> Response {
    // Determine effective thread count
    let effective_threads = if state.worker_threads > 0 {
        state.worker_threads
    } else {
        rayon::current_num_threads()
    };

    let result = GetConfigResult {
        gpu_enabled: state.gpu_enabled,
        default_timeout_ms: state.default_timeout_ms,
        worker_threads: state.worker_threads,
        effective_threads,
    };

    Response::success_typed(id.clone(), &result)
        .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
}

/// Handle the "getMetrics" method
///
/// Returns server-wide runtime metrics for monitoring and optimization.
/// Useful for AI agents to track server health and performance.
#[instrument(skip(state))]
pub async fn handle_get_metrics(state: &ServerState, id: RequestId) -> Response {
    let metrics = &state.metrics;

    let result = GetMetricsResult {
        uptime_secs: metrics.uptime_secs(),
        total_requests: metrics.total_requests.load(Ordering::Relaxed),
        successful_requests: metrics.successful_requests.load(Ordering::Relaxed),
        failed_requests: metrics.failed_requests.load(Ordering::Relaxed),
        success_rate: metrics.success_rate(),
        avg_latency_us: metrics.avg_latency_us(),
        requests_per_second: metrics.requests_per_second(),
        method_counts: MethodCounts {
            check: metrics.check_requests.load(Ordering::Relaxed),
            prove: metrics.prove_requests.load(Ordering::Relaxed),
            get_type: metrics.get_type_requests.load(Ordering::Relaxed),
            batch_check: metrics.batch_check_requests.load(Ordering::Relaxed),
            verify_cert: metrics.verify_cert_requests.load(Ordering::Relaxed),
            batch_verify_cert: metrics.batch_verify_cert_requests.load(Ordering::Relaxed),
            verify_c: metrics.verify_c_requests.load(Ordering::Relaxed),
        },
        batch_stats: BatchStats {
            items_processed: metrics.batch_items_processed.load(Ordering::Relaxed),
            certificates_verified: metrics.certificates_verified.load(Ordering::Relaxed),
        },
        timing: TimingStats {
            cumulative_handler_time_us: metrics.cumulative_time_us.load(Ordering::Relaxed),
            type_check_time_us: metrics.type_check_time_us.load(Ordering::Relaxed),
            cert_verify_time_us: metrics.cert_verify_time_us.load(Ordering::Relaxed),
        },
    };

    Response::success_typed(id.clone(), &result)
        .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
}

/// Save environment request parameters
#[derive(Debug, Clone, Deserialize)]
pub struct SaveEnvironmentParams {
    /// File path to save to
    pub path: String,
    /// Format: "bincode" (default) or "json"
    #[serde(default)]
    pub format: Option<String>,
}

/// Save environment response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveEnvironmentResult {
    /// Whether the save succeeded
    pub success: bool,
    /// Number of constants saved
    pub num_constants: usize,
    /// Number of inductives saved
    pub num_inductives: usize,
    /// File size in bytes
    pub file_size: u64,
}

/// Handle the "saveEnvironment" method
#[instrument(skip(state))]
pub async fn handle_save_environment(
    state: &ServerState,
    id: RequestId,
    params: SaveEnvironmentParams,
) -> Response {
    let env = state.env.read().await;
    let path = std::path::Path::new(&params.path);
    let format = params.format.as_deref().unwrap_or("bincode");

    let result = match format {
        "json" => {
            let json = env.to_json_pretty().map_err(|e| e.to_string());
            match json {
                Ok(data) => std::fs::write(path, data.as_bytes()).map_err(|e| e.to_string()),
                Err(e) => Err(e),
            }
        }
        _ => env.save_to_file(path).map_err(|e| e.to_string()),
    };

    match result {
        Ok(_) => {
            let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            let save_result = SaveEnvironmentResult {
                success: true,
                num_constants: env.num_constants(),
                num_inductives: env.num_inductives(),
                file_size,
            };
            Response::success_typed(id.clone(), &save_result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Err(e) => Response::error(id, RpcError::internal_error(format!("Failed to save: {e}"))),
    }
}

/// Load environment request parameters
#[derive(Debug, Clone, Deserialize)]
pub struct LoadEnvironmentParams {
    /// File path to load from
    pub path: String,
    /// Format: "bincode" (default) or "json"
    #[serde(default)]
    pub format: Option<String>,
    /// Whether to replace or merge with current environment
    #[serde(default)]
    pub replace: bool,
}

/// Load environment response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadEnvironmentResult {
    /// Whether the load succeeded
    pub success: bool,
    /// Number of constants loaded
    pub num_constants: usize,
    /// Number of inductives loaded
    pub num_inductives: usize,
}

/// Handle the "loadEnvironment" method
#[instrument(skip(state))]
pub async fn handle_load_environment(
    state: &ServerState,
    id: RequestId,
    params: LoadEnvironmentParams,
) -> Response {
    let path = std::path::Path::new(&params.path);
    let format = params.format.as_deref().unwrap_or("bincode");

    let loaded_env = match format {
        "json" => {
            let data = std::fs::read_to_string(path).map_err(|e| e.to_string());
            match data {
                Ok(json) => Environment::from_json(&json).map_err(|e| e.to_string()),
                Err(e) => Err(e),
            }
        }
        _ => Environment::load_from_file(path).map_err(|e| e.to_string()),
    };

    match loaded_env {
        Ok(new_env) => {
            let num_constants = new_env.num_constants();
            let num_inductives = new_env.num_inductives();

            // Update the shared environment
            let mut env = state.env.write().await;
            if params.replace {
                *env = new_env;
            } else {
                // Merge: for now just replace (merge logic would be more complex)
                *env = new_env;
            }

            let load_result = LoadEnvironmentResult {
                success: true,
                num_constants,
                num_inductives,
            };
            Response::success_typed(id.clone(), &load_result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Err(e) => Response::error(id, RpcError::internal_error(format!("Failed to load: {e}"))),
    }
}

/// Get environment request parameters (optional filtering)
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GetEnvironmentParams {
    /// Return JSON representation (default: false returns summary)
    #[serde(default)]
    pub include_json: bool,
}

/// Get environment response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetEnvironmentResult {
    /// Number of constants
    pub num_constants: usize,
    /// Number of inductives
    pub num_inductives: usize,
    /// Constant names (first 100)
    pub constant_names: Vec<String>,
    /// JSON representation (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<String>,
}

/// Verify C code request parameters
#[derive(Debug, Clone, Deserialize)]
pub struct VerifyCParams {
    /// C source code with ACSL specifications
    pub code: String,
    /// Treat unknown obligations as failures
    #[serde(default)]
    pub fail_unknown: bool,
    /// Include per-VC details in response
    #[serde(default)]
    pub include_details: bool,
    /// Optional timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

/// Result for a single function verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyCFunctionResult {
    /// Function name
    pub name: String,
    /// Total verification conditions
    pub total_vcs: usize,
    /// Number of VCs proved
    pub proved: usize,
    /// Number of VCs failed
    pub failed: usize,
    /// Number of VCs with unknown status
    pub unknown: usize,
    /// Per-VC details (if requested)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub details: Vec<VerifyCVCDetail>,
}

/// Detail for a single verification condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyCVCDetail {
    /// Description of the VC
    pub description: String,
    /// Status: "proved", "failed", "unknown"
    pub status: String,
    /// Reason for failure (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Verify C code response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyCResult {
    /// Whether verification succeeded (all VCs proved)
    pub success: bool,
    /// Number of functions verified
    pub num_functions: usize,
    /// Total verification conditions
    pub total_vcs: usize,
    /// Total proved VCs
    pub proved: usize,
    /// Total failed VCs
    pub failed: usize,
    /// Total unknown VCs
    pub unknown: usize,
    /// Per-function results
    pub functions: Vec<VerifyCFunctionResult>,
    /// Parse errors (if any)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
    /// Time taken in milliseconds
    pub time_ms: u64,
}

/// Handle the "getEnvironment" method
#[instrument(skip(state))]
pub async fn handle_get_environment(
    state: &ServerState,
    id: RequestId,
    params: GetEnvironmentParams,
) -> Response {
    let env = state.env.read().await;

    let constant_names: Vec<String> = env
        .constants()
        .take(100)
        .map(|c| c.name.to_string())
        .collect();

    let json = if params.include_json {
        env.to_json_pretty().ok()
    } else {
        None
    };

    let result = GetEnvironmentResult {
        num_constants: env.num_constants(),
        num_inductives: env.num_inductives(),
        constant_names,
        json,
    };

    Response::success_typed(id.clone(), &result)
        .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
}

/// Handle the "verifyC" method
#[instrument(skip(state))]
pub async fn handle_verify_c(
    state: &ServerState,
    id: RequestId,
    params: VerifyCParams,
    progress: Option<ProgressSender>,
) -> Response {
    let start = Instant::now();
    let timeout = Duration::from_millis(params.timeout_ms.unwrap_or(state.default_timeout_ms * 2));

    let result = tokio::time::timeout(timeout, async {
        verify_c_impl(&params, progress.clone()).await
    })
    .await;

    let elapsed_us = start.elapsed().as_micros() as u64;
    let elapsed_ms = elapsed_us / 1000;

    match result {
        Ok(Ok(mut verify_result)) => {
            verify_result.time_ms = elapsed_ms;
            let success = verify_result.success;
            state.metrics.record_request("verifyC", success, elapsed_us);
            Response::success_typed(id.clone(), &verify_result)
                .unwrap_or_else(|e| Response::error(id, RpcError::internal_error(e.to_string())))
        }
        Ok(Err(e)) => {
            state.metrics.record_request("verifyC", false, elapsed_us);
            Response::error(id, e)
        }
        Err(_) => {
            state.metrics.record_request("verifyC", false, elapsed_us);
            Response::error(id, RpcError::timeout(timeout.as_millis() as u64))
        }
    }
}

async fn verify_c_impl(
    params: &VerifyCParams,
    progress: Option<ProgressSender>,
) -> Result<VerifyCResult, RpcError> {
    let mut parser = CParser::new();

    // Parse the C source with ACSL specs
    let functions = parser
        .parse_translation_unit_with_specs(&params.code)
        .map_err(|e| RpcError::lean_parse_error(format!("C parse error: {e}")))?;

    if functions.is_empty() {
        return Ok(VerifyCResult {
            success: true,
            num_functions: 0,
            total_vcs: 0,
            proved: 0,
            failed: 0,
            unknown: 0,
            functions: vec![],
            errors: vec!["No functions found in source".to_string()],
            time_ms: 0,
        });
    }

    let total_functions = functions.len();
    let mut func_results = Vec::with_capacity(total_functions);
    let mut total_vcs = 0;
    let mut total_proved = 0;
    let mut total_failed = 0;
    let mut total_unknown = 0;

    if let Some(ref progress) = progress {
        progress
            .notify(
                format!("Verifying {total_functions} function(s)"),
                Some(0),
                None,
            )
            .await;
    }

    for (idx, vf) in functions.into_iter().enumerate() {
        let summary = vf.verify();

        // Build per-VC details if requested
        let details = if params.include_details {
            summary
                .details
                .iter()
                .map(|(desc, status)| {
                    let (status_str, reason) = match status {
                        ProofStatus::Proved(_) => ("proved".to_string(), None),
                        ProofStatus::Failed(r) => ("failed".to_string(), Some(r.clone())),
                        ProofStatus::Unknown => ("unknown".to_string(), None),
                    };
                    VerifyCVCDetail {
                        description: desc.clone(),
                        status: status_str,
                        reason,
                    }
                })
                .collect()
        } else {
            vec![]
        };

        let func_result = VerifyCFunctionResult {
            name: vf.name.clone(),
            total_vcs: summary.total,
            proved: summary.proved,
            failed: summary.failed,
            unknown: summary.unknown,
            details,
        };

        total_vcs += summary.total;
        total_proved += summary.proved;
        total_failed += summary.failed;
        total_unknown += summary.unknown;

        func_results.push(func_result);

        if let Some(ref progress) = progress {
            let percentage = if total_functions == 0 {
                100
            } else {
                (((idx + 1) * 100) / total_functions).min(100)
            } as u8;

            progress
                .notify(
                    format!(
                        "Verified {}/{} ({}: {} proved)",
                        idx + 1,
                        total_functions,
                        vf.name,
                        summary.proved
                    ),
                    Some(percentage),
                    Some(json!({
                        "function": vf.name,
                        "proved": summary.proved,
                        "failed": summary.failed,
                        "unknown": summary.unknown,
                    })),
                )
                .await;
        }
    }

    // Determine success based on fail_unknown flag
    let success = total_failed == 0 && (!params.fail_unknown || total_unknown == 0);

    Ok(VerifyCResult {
        success,
        num_functions: func_results.len(),
        total_vcs,
        proved: total_proved,
        failed: total_failed,
        unknown: total_unknown,
        functions: func_results,
        errors: vec![],
        time_ms: 0,
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Format an expression for display
fn format_expr(expr: &Expr) -> String {
    // Simple formatting - could be enhanced with proper pretty printing
    format!("{expr:?}")
}

/// Format a parse error
fn format_parse_error(e: &ParseError) -> String {
    e.to_string()
}

/// Extract line number from parse error
fn parse_error_line(e: &ParseError) -> Option<usize> {
    match e {
        ParseError::UnexpectedToken { line, .. } => Some(*line),
        ParseError::UnexpectedEof | ParseError::NumericOverflow { .. } => None,
    }
}

/// Extract column number from parse error
fn parse_error_col(e: &ParseError) -> Option<usize> {
    match e {
        ParseError::UnexpectedToken { col, .. } => Some(*col),
        ParseError::UnexpectedEof | ParseError::NumericOverflow { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_check_simple_expr() {
        let state = ServerState::new();
        // Use fully-typed expression (no metavariables needed)
        let params = CheckParams {
            code: "fun (A : Type) (x : A) => x".to_string(),
            timeout_ms: None,
        };

        let response = handle_check(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );
        assert!(response.result.is_some());

        let result: CheckResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(
            result.valid,
            "Check result should be valid, got errors: {:?}",
            result.errors
        );
    }

    #[tokio::test]
    async fn test_check_invalid_syntax() {
        let state = ServerState::new();
        let params = CheckParams {
            // Use undefined identifier that will fail type checking
            code: "unknownIdent123".to_string(),
            timeout_ms: None,
        };

        let response = handle_check(&state, RequestId::Number(1), params).await;
        assert!(response.error.is_none());

        let result: CheckResult = serde_json::from_value(response.result.unwrap()).unwrap();
        // Should fail because unknownIdent123 is not defined
        assert!(
            !result.valid,
            "Expected invalid result for undefined identifier"
        );
        assert!(
            !result.errors.is_empty(),
            "Expected errors for undefined identifier"
        );
    }

    #[tokio::test]
    async fn test_server_info() {
        let state = ServerState::new();
        let response = handle_server_info(&state, RequestId::Number(1)).await;

        assert!(response.error.is_none());
        let info: ServerInfo = serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(info.name, "lean5-server");
        assert!(info.methods.contains(&"check".to_string()));
    }

    #[tokio::test]
    async fn test_batch_check() {
        let state = ServerState::new();
        let params = BatchCheckParams {
            items: vec![
                BatchCheckItem {
                    id: "1".to_string(),
                    // Use fully-typed expression
                    code: "fun (A : Type) (x : A) => x".to_string(),
                },
                BatchCheckItem {
                    id: "2".to_string(),
                    code: "Type".to_string(),
                },
            ],
            use_gpu: false,
            timeout_ms: None,
        };

        let response = handle_batch_check(&state, RequestId::Number(1), params, None).await;
        assert!(response.error.is_none());

        let result: BatchCheckResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.results.len(), 2);
        assert!(
            result.results[0].valid,
            "First item should be valid: {:?}",
            result.results[0]
        );
        assert!(
            result.results[1].valid,
            "Second item should be valid: {:?}",
            result.results[1]
        );
    }

    #[tokio::test]
    async fn test_get_type() {
        let state = ServerState::new();
        let params = GetTypeParams {
            expr: "fun (x : Type) => x".to_string(),
        };

        let response = handle_get_type(&state, RequestId::Number(1), params).await;
        assert!(response.error.is_none());

        let result: GetTypeResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(!result.type_.is_empty());
    }

    #[tokio::test]
    async fn test_verify_c_simple_function() {
        let state = ServerState::new();
        let params = VerifyCParams {
            code: r"
                //@ requires n >= 0;
                //@ ensures \result >= 0;
                int id(int n) { return n; }
            "
            .to_string(),
            fail_unknown: false,
            include_details: false,
            timeout_ms: None,
        };

        let response = handle_verify_c(&state, RequestId::Number(1), params, None).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );
        assert!(response.result.is_some());

        let result: VerifyCResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.num_functions, 1);
        assert!(result.total_vcs > 0, "Should generate VCs");
        assert!(
            result.functions[0].name == "id",
            "Function name should be 'id'"
        );
    }

    #[tokio::test]
    async fn test_verify_c_with_details() {
        let state = ServerState::new();
        let params = VerifyCParams {
            code: r"
                //@ requires x >= 0;
                //@ ensures \result >= 0;
                int identity(int x) { return x; }
            "
            .to_string(),
            fail_unknown: false,
            include_details: true,
            timeout_ms: None,
        };

        let response = handle_verify_c(&state, RequestId::Number(1), params, None).await;
        assert!(response.error.is_none());

        let result: VerifyCResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.num_functions, 1);
        // With include_details=true, details should be populated
        // (may be empty if all VCs are trivially proved without details)
    }

    #[tokio::test]
    async fn test_verify_c_no_functions() {
        let state = ServerState::new();
        let params = VerifyCParams {
            code: "int x;".to_string(), // just a declaration, no function
            fail_unknown: false,
            include_details: false,
            timeout_ms: None,
        };

        let response = handle_verify_c(&state, RequestId::Number(1), params, None).await;
        assert!(response.error.is_none());

        let result: VerifyCResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.num_functions, 0);
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_verify_c_malformed_code() {
        // tree-sitter is lenient and parses partial/invalid code as "error" nodes
        // so it won't raise a parse error, but may return no valid functions
        let state = ServerState::new();
        let params = VerifyCParams {
            code: "int func( { invalid".to_string(),
            fail_unknown: false,
            include_details: false,
            timeout_ms: None,
        };

        let response = handle_verify_c(&state, RequestId::Number(1), params, None).await;
        // tree-sitter tolerates invalid syntax - returns success with no functions
        assert!(response.error.is_none() || response.result.is_some());
        if let Some(result_json) = response.result {
            let result: VerifyCResult = serde_json::from_value(result_json).unwrap();
            // Either no functions found or some partial parse
            assert!(result.num_functions <= 1);
        }
    }

    #[tokio::test]
    async fn test_server_info_includes_verify_c() {
        let state = ServerState::new();
        let response = handle_server_info(&state, RequestId::Number(1)).await;

        assert!(response.error.is_none());
        let info: ServerInfo = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(
            info.methods.contains(&"verifyC".to_string()),
            "serverInfo should include verifyC method"
        );
    }

    #[tokio::test]
    async fn test_server_info_includes_batch_verify_cert() {
        let state = ServerState::new();
        let response = handle_server_info(&state, RequestId::Number(1)).await;

        assert!(response.error.is_none());
        let info: ServerInfo = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(
            info.methods.contains(&"batchVerifyCert".to_string()),
            "serverInfo should include batchVerifyCert method"
        );
    }

    // --- Single certificate verification tests (verifyCert) ---

    #[tokio::test]
    async fn test_verify_cert_valid_sort() {
        let state = ServerState::new();
        let level = Level::zero();
        let expr = Expr::Sort(level.clone());
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let params = VerifyCertParams {
            cert,
            expr,
            timeout_ms: None,
        };

        let response = handle_verify_cert(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );

        let result: VerifyCertResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.success);
        assert!(result.verified_type.is_some());
        assert!(result.error.is_none());
        // time_us is always non-negative (u64), just verify it exists
        let _ = result.time_us;
    }

    #[tokio::test]
    async fn test_verify_cert_invalid_certificate() {
        let state = ServerState::new();
        // Create mismatched cert and expression
        let level = Level::zero();
        let expr = Expr::Sort(level.clone());
        let cert = ProofCert::Sort {
            level: Level::succ(Level::zero()), // Mismatched level
        };

        let params = VerifyCertParams {
            cert,
            expr,
            timeout_ms: None,
        };

        let response = handle_verify_cert(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Should return result not RPC error"
        );

        let result: VerifyCertResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(!result.success);
        assert!(result.verified_type.is_none());
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn test_verify_cert_json_serialization() {
        let level = Level::zero();
        let cert = ProofCert::Sort {
            level: level.clone(),
        };
        let expr = Expr::Sort(level);

        let params = VerifyCertParams {
            cert,
            expr,
            timeout_ms: Some(1000),
        };

        // Test that params can be serialized/deserialized
        let json = serde_json::to_string(&params).expect("Should serialize");
        let _: VerifyCertParams = serde_json::from_str(&json).expect("Should deserialize");

        let result = VerifyCertResult {
            success: true,
            verified_type: Some("Sort(succ(zero))".to_string()),
            error: None,
            time_us: 42,
        };

        let json = serde_json::to_string(&result).expect("Should serialize");
        let deserialized: VerifyCertResult =
            serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.success, result.success);
        assert_eq!(deserialized.time_us, result.time_us);
    }

    #[tokio::test]
    async fn test_server_info_includes_verify_cert() {
        let state = ServerState::new();
        let response = handle_server_info(&state, RequestId::Number(1)).await;
        let info: ServerInfo = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(
            info.methods.contains(&"verifyCert".to_string()),
            "serverInfo should include verifyCert method"
        );
    }

    #[tokio::test]
    async fn test_verify_cert_valid_pi() {
        use lean5_kernel::{BinderInfo, Level};

        let state = ServerState::new();
        // Pi type: (x : Type) → Type
        // This is a valid Pi type in universe Type 1
        let type_0 = Expr::Sort(Level::zero());

        // The Pi expression: (x : Type 0) → Type 0
        let pi_expr = Expr::Pi(
            BinderInfo::Default,
            Arc::new(type_0.clone()),
            Arc::new(type_0.clone()), // Body doesn't use x, so it's just Type 0
        );

        // Use infer_type_with_cert to generate correct certificate
        use lean5_kernel::TypeChecker;
        let env = lean5_kernel::Environment::new();
        let mut tc = TypeChecker::new(&env);
        let (_, cert) = tc
            .infer_type_with_cert(&pi_expr)
            .expect("Pi should type-check");

        let params = VerifyCertParams {
            cert,
            expr: pi_expr,
            timeout_ms: None,
        };

        let response = handle_verify_cert(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );

        let result: VerifyCertResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(
            result.success,
            "Pi verification should succeed: {:?}",
            result.error
        );
        assert!(result.verified_type.is_some());
    }

    #[tokio::test]
    async fn test_verify_cert_valid_lambda() {
        use lean5_kernel::{BinderInfo, Level};

        let state = ServerState::new();
        // Lambda: λ (x : Type) => x
        // This is the identity function at Type level
        let type_0 = Expr::Sort(Level::zero());

        let lam_expr = Expr::Lam(
            BinderInfo::Default,
            Arc::new(type_0.clone()),
            Arc::new(Expr::BVar(0)), // Body is just x (BVar 0)
        );

        // Use infer_type_with_cert to generate correct certificate
        use lean5_kernel::TypeChecker;
        let env = lean5_kernel::Environment::new();
        let mut tc = TypeChecker::new(&env);
        let (_, cert) = tc
            .infer_type_with_cert(&lam_expr)
            .expect("Lambda should type-check");

        let params = VerifyCertParams {
            cert,
            expr: lam_expr,
            timeout_ms: None,
        };

        let response = handle_verify_cert(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );

        let result: VerifyCertResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(
            result.success,
            "Lambda verification should succeed: {:?}",
            result.error
        );
        assert!(result.verified_type.is_some());
    }

    #[tokio::test]
    async fn test_verify_cert_valid_app() {
        use lean5_kernel::{BinderInfo, Level};

        let state = ServerState::new();
        // Application: (λ (x : Type 1) => x) Type
        // The identity function at universe level 1 applied to Type 0
        // Type 0 : Type 1, so this is well-typed
        let type_0 = Expr::Sort(Level::zero());
        let type_1 = Expr::Sort(Level::succ(Level::zero()));

        // The identity lambda: λ (x : Type 1). x
        let id_lam = Expr::Lam(
            BinderInfo::Default,
            Arc::new(type_1.clone()), // Domain is Type 1
            Arc::new(Expr::BVar(0)),
        );

        // Application expression: (λ (x : Type 1). x) Type 0
        // Type 0 has type Type 1, so this is valid
        let app_expr = Expr::App(Arc::new(id_lam), Arc::new(type_0));

        // Use infer_type_with_cert to generate correct certificate
        use lean5_kernel::TypeChecker;
        let env = lean5_kernel::Environment::new();
        let mut tc = TypeChecker::new(&env);
        let (_, cert) = tc
            .infer_type_with_cert(&app_expr)
            .expect("App should type-check");

        let params = VerifyCertParams {
            cert,
            expr: app_expr,
            timeout_ms: None,
        };

        let response = handle_verify_cert(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );

        let result: VerifyCertResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(
            result.success,
            "App verification should succeed: {:?}",
            result.error
        );
        assert!(result.verified_type.is_some());
    }

    #[tokio::test]
    async fn test_verify_cert_invalid_pi_level_mismatch() {
        use lean5_kernel::{BinderInfo, Level};

        let state = ServerState::new();
        // Pi type with mismatched level in certificate
        let type_0 = Expr::Sort(Level::zero());

        let pi_expr = Expr::Pi(
            BinderInfo::Default,
            Arc::new(type_0.clone()),
            Arc::new(type_0.clone()),
        );

        // Certificate with wrong arg level
        let arg_type_cert = ProofCert::Sort {
            level: Level::zero(),
        };
        let body_type_cert = ProofCert::Sort {
            level: Level::zero(),
        };

        let cert = ProofCert::Pi {
            binder_info: BinderInfo::Default,
            arg_type_cert: Box::new(arg_type_cert),
            arg_level: Level::zero(), // WRONG: should be succ(zero)
            body_type_cert: Box::new(body_type_cert),
            body_level: Level::succ(Level::zero()),
        };

        let params = VerifyCertParams {
            cert,
            expr: pi_expr,
            timeout_ms: None,
        };

        let response = handle_verify_cert(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Should return result not RPC error"
        );

        let result: VerifyCertResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(
            !result.success,
            "Pi verification should fail due to level mismatch"
        );
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn test_verify_cert_nested_app_lambda() {
        use lean5_kernel::{BinderInfo, Level};

        let state = ServerState::new();
        // Test nested application: (λx:Type1. x) ((λy:Type1. y) Type0)
        // This is: id (id Type0) where id = λx:Type1. x
        // Type0 : Type1, so both applications are well-typed
        let type_0 = Expr::Sort(Level::zero());
        let type_1 = Expr::Sort(Level::succ(Level::zero()));

        // The identity lambda at Type 1: λx:Type1. x
        let id_lam = Expr::Lam(
            BinderInfo::Default,
            Arc::new(type_1.clone()),
            Arc::new(Expr::BVar(0)),
        );

        // Inner application: (λy:Type1. y) Type0
        let inner_app = Expr::App(Arc::new(id_lam.clone()), Arc::new(type_0.clone()));

        // Outer application: (λx:Type1. x) ((λy:Type1. y) Type0)
        // The inner app has type Type1 (from id's return type instantiated),
        // but we need to apply id to something of type Type1.
        // However, ((λy:Type1. y) Type0) has type Type1, not Type0!
        // So this should work.
        let nested_app = Expr::App(Arc::new(id_lam), Arc::new(inner_app));

        // Use infer_type_with_cert to generate correct certificate
        use lean5_kernel::TypeChecker;
        let env = lean5_kernel::Environment::new();
        let mut tc = TypeChecker::new(&env);
        let (_, cert) = tc
            .infer_type_with_cert(&nested_app)
            .expect("Nested app should type-check");

        let params = VerifyCertParams {
            cert,
            expr: nested_app,
            timeout_ms: None,
        };

        let response = handle_verify_cert(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );

        let result: VerifyCertResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(
            result.success,
            "Nested app/lambda verification should succeed: {:?}",
            result.error
        );
        assert!(result.verified_type.is_some());
    }

    // --- Batch certificate verification tests ---

    #[tokio::test]
    async fn test_batch_verify_cert_empty() {
        let state = ServerState::new();
        let params = BatchVerifyCertParams {
            items: vec![],
            threads: 0,
            timeout_ms: None,
        };

        let response = handle_batch_verify_cert(&state, RequestId::Number(1), params, None).await;
        assert!(response.error.is_none());

        let result: BatchVerifyCertResult =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.results.is_empty());
        assert_eq!(result.stats.total, 0);
    }

    #[tokio::test]
    async fn test_batch_verify_cert_single_valid() {
        use lean5_kernel::Level;

        let state = ServerState::new();
        let level = Level::zero();
        let expr = Expr::Sort(level.clone());
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let params = BatchVerifyCertParams {
            items: vec![BatchVerifyCertItem {
                id: "test1".to_string(),
                cert,
                expr,
            }],
            threads: 0,
            timeout_ms: None,
        };

        let response = handle_batch_verify_cert(&state, RequestId::Number(1), params, None).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );

        let result: BatchVerifyCertResult =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.results.len(), 1);
        assert!(result.results[0].success);
        assert_eq!(result.results[0].id, "test1");
        assert!(result.results[0].verified_type.is_some());
        assert_eq!(result.stats.total, 1);
        assert_eq!(result.stats.successful, 1);
        assert_eq!(result.stats.failed, 0);
    }

    #[tokio::test]
    async fn test_batch_verify_cert_multiple() {
        use lean5_kernel::Level;

        let state = ServerState::new();

        let items: Vec<BatchVerifyCertItem> = (0..10)
            .map(|i| {
                let level = Level::zero();
                let expr = Expr::Sort(level.clone());
                let cert = ProofCert::Sort {
                    level: level.clone(),
                };
                BatchVerifyCertItem {
                    id: format!("cert_{i}"),
                    cert,
                    expr,
                }
            })
            .collect();

        let params = BatchVerifyCertParams {
            items,
            threads: 0,
            timeout_ms: None,
        };

        let response = handle_batch_verify_cert(&state, RequestId::Number(1), params, None).await;
        assert!(response.error.is_none());

        let result: BatchVerifyCertResult =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.results.len(), 10);
        assert_eq!(result.stats.total, 10);
        assert_eq!(result.stats.successful, 10);

        for (i, item_result) in result.results.iter().enumerate() {
            assert!(item_result.success, "Failed at index {i}");
            assert_eq!(item_result.id, format!("cert_{i}"));
        }
    }

    #[tokio::test]
    async fn test_batch_verify_cert_with_failures() {
        use lean5_kernel::Level;

        let state = ServerState::new();

        let items: Vec<BatchVerifyCertItem> = (0..5)
            .map(|i| {
                let level = Level::zero();
                let expr = Expr::Sort(level.clone());
                if i % 2 == 0 {
                    // Valid certificate
                    let cert = ProofCert::Sort {
                        level: level.clone(),
                    };
                    BatchVerifyCertItem {
                        id: format!("valid_{i}"),
                        cert,
                        expr,
                    }
                } else {
                    // Invalid certificate (level mismatch)
                    let cert = ProofCert::Sort {
                        level: Level::succ(Level::zero()),
                    };
                    BatchVerifyCertItem {
                        id: format!("invalid_{i}"),
                        cert,
                        expr,
                    }
                }
            })
            .collect();

        let params = BatchVerifyCertParams {
            items,
            threads: 0,
            timeout_ms: None,
        };

        let response = handle_batch_verify_cert(&state, RequestId::Number(1), params, None).await;
        assert!(response.error.is_none());

        let result: BatchVerifyCertResult =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.results.len(), 5);
        assert!(result.results[0].success); // valid_0
        assert!(!result.results[1].success); // invalid_1
        assert!(result.results[2].success); // valid_2
        assert!(!result.results[3].success); // invalid_3
        assert!(result.results[4].success); // valid_4

        assert_eq!(result.stats.successful, 3);
        assert_eq!(result.stats.failed, 2);
    }

    #[tokio::test]
    async fn test_batch_verify_cert_stats() {
        use lean5_kernel::Level;

        let state = ServerState::new();

        let items: Vec<BatchVerifyCertItem> = (0..100)
            .map(|i| {
                let level = Level::zero();
                let expr = Expr::Sort(level.clone());
                let cert = ProofCert::Sort {
                    level: level.clone(),
                };
                BatchVerifyCertItem {
                    id: format!("{i}"),
                    cert,
                    expr,
                }
            })
            .collect();

        let params = BatchVerifyCertParams {
            items,
            threads: 0,
            timeout_ms: None,
        };

        let response = handle_batch_verify_cert(&state, RequestId::Number(1), params, None).await;
        assert!(response.error.is_none());

        let result: BatchVerifyCertResult =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.stats.total, 100);
        assert_eq!(result.stats.successful, 100);
        assert_eq!(result.stats.failed, 0);
        // Verify timing stats are populated
        assert!(result.stats.wall_time_us > 0 || result.stats.total == 0);
    }

    #[tokio::test]
    async fn test_batch_verify_cert_streams_progress() {
        use tokio::sync::mpsc;

        let state = ServerState::new();
        let level = Level::zero();
        let expr = Expr::Sort(level.clone());
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let params = BatchVerifyCertParams {
            items: vec![
                BatchVerifyCertItem {
                    id: "a".to_string(),
                    cert: cert.clone(),
                    expr: expr.clone(),
                },
                BatchVerifyCertItem {
                    id: "b".to_string(),
                    cert,
                    expr,
                },
            ],
            threads: 0,
            timeout_ms: None,
        };

        let (tx, mut rx) = mpsc::channel(16);
        let progress = ProgressSender::new(RequestId::Number(99), tx);

        let response =
            handle_batch_verify_cert(&state, RequestId::Number(1), params, Some(progress)).await;
        assert!(response.error.is_none());

        let mut updates = Vec::new();
        while let Ok(update) = tokio::time::timeout(Duration::from_secs(1), rx.recv()).await {
            match update {
                Some(u) => updates.push(u),
                None => break,
            }
        }

        assert!(
            updates.len() >= 3,
            "expected start + per-item + final progress updates"
        );
        assert!(
            updates.iter().any(|u| u.message.contains("Verified")),
            "should include per-item verification updates"
        );
        assert!(
            updates.iter().any(|u| u.percentage == Some(100)),
            "should mark completion percentage"
        );
    }

    #[tokio::test]
    async fn test_batch_verify_cert_json_serialization() {
        use lean5_kernel::Level;

        // Test that the types can be serialized/deserialized correctly
        let level = Level::zero();
        let expr = Expr::Sort(level.clone());
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let item = BatchVerifyCertItem {
            id: "test".to_string(),
            cert,
            expr,
        };

        // Should be able to serialize to JSON
        let json = serde_json::to_string(&item).expect("Should serialize");
        assert!(json.contains("test"));

        // Should be able to deserialize back
        let _: BatchVerifyCertItem = serde_json::from_str(&json).expect("Should deserialize");
    }

    // ========================================================================
    // Certificate Compression Tests
    // ========================================================================

    #[tokio::test]
    async fn test_compress_cert_simple() {
        use lean5_kernel::Level;

        let state = ServerState::new();
        let level = Level::zero();
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let params = CompressCertParams {
            cert,
            include_stats: false,
        };

        let response = handle_compress_cert(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );

        let result: CompressCertResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.stats.is_none());
        // time_us is always non-negative (u64), just verify it exists
        let _ = result.time_us;
    }

    #[tokio::test]
    async fn test_compress_cert_with_stats() {
        use lean5_kernel::Level;

        let state = ServerState::new();
        let level = Level::zero();
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let params = CompressCertParams {
            cert,
            include_stats: true,
        };

        let response = handle_compress_cert(&state, RequestId::Number(1), params).await;
        assert!(response.error.is_none());

        let result: CompressCertResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.stats.is_some());
        let stats = result.stats.unwrap();
        assert!(stats.unique_levels >= 1);
    }

    #[tokio::test]
    async fn test_compress_decompress_roundtrip() {
        use lean5_kernel::Level;

        let state = ServerState::new();
        let level = Level::zero();
        let original_cert = ProofCert::Sort {
            level: level.clone(),
        };

        // Compress
        let compress_params = CompressCertParams {
            cert: original_cert.clone(),
            include_stats: false,
        };
        let compress_response =
            handle_compress_cert(&state, RequestId::Number(1), compress_params).await;
        assert!(compress_response.error.is_none());

        let compress_result: CompressCertResult =
            serde_json::from_value(compress_response.result.unwrap()).unwrap();

        // Decompress
        let decompress_params = DecompressCertParams {
            compressed: compress_result.compressed,
        };
        let decompress_response =
            handle_decompress_cert(&state, RequestId::Number(2), decompress_params).await;
        assert!(decompress_response.error.is_none());

        let decompress_result: DecompressCertResult =
            serde_json::from_value(decompress_response.result.unwrap()).unwrap();

        // Verify roundtrip - certificates should be equivalent
        match (&original_cert, &decompress_result.cert) {
            (ProofCert::Sort { level: l1 }, ProofCert::Sort { level: l2 }) => {
                assert!(
                    Level::is_def_eq(l1, l2),
                    "Levels should match after roundtrip"
                );
            }
            _ => panic!("Certificate type mismatch after roundtrip"),
        }
    }

    #[tokio::test]
    async fn test_archive_cert_lz4() {
        use lean5_kernel::Level;

        let state = ServerState::new();
        let level = Level::zero();
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let params = ArchiveCertParams {
            cert,
            algorithm: Some("lz4".to_string()),
            level: None,
            include_stats: false,
        };

        let response = handle_archive_cert(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );

        let result: ArchiveCertResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.algorithm, "lz4");
        assert!(result.original_size > 0);
        assert!(result.compressed_size > 0);
        assert!(!result.archive.is_empty());
    }

    #[tokio::test]
    async fn test_archive_cert_zstd() {
        use lean5_kernel::Level;

        let state = ServerState::new();
        let level = Level::zero();
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let params = ArchiveCertParams {
            cert,
            algorithm: Some("zstd".to_string()),
            level: None,
            include_stats: false,
        };

        let response = handle_archive_cert(&state, RequestId::Number(1), params).await;
        assert!(response.error.is_none());

        let result: ArchiveCertResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.algorithm, "zstd");
        assert!(result.original_size > 0);
    }

    #[tokio::test]
    async fn test_archive_unarchive_roundtrip() {
        use lean5_kernel::Level;

        let state = ServerState::new();
        let level = Level::zero();
        let original_cert = ProofCert::Sort {
            level: level.clone(),
        };

        // Archive
        let archive_params = ArchiveCertParams {
            cert: original_cert.clone(),
            algorithm: Some("lz4".to_string()),
            level: None,
            include_stats: false,
        };
        let archive_response =
            handle_archive_cert(&state, RequestId::Number(1), archive_params).await;
        assert!(archive_response.error.is_none());

        let archive_result: ArchiveCertResult =
            serde_json::from_value(archive_response.result.unwrap()).unwrap();

        // Unarchive
        let unarchive_params = UnarchiveCertParams {
            archive: archive_result.archive,
        };
        let unarchive_response =
            handle_unarchive_cert(&state, RequestId::Number(2), unarchive_params).await;
        assert!(
            unarchive_response.error.is_none(),
            "Unarchive error: {:?}",
            unarchive_response.error
        );

        let unarchive_result: UnarchiveCertResult =
            serde_json::from_value(unarchive_response.result.unwrap()).unwrap();

        assert_eq!(unarchive_result.algorithm, "lz4");

        // Verify roundtrip
        match (&original_cert, &unarchive_result.cert) {
            (ProofCert::Sort { level: l1 }, ProofCert::Sort { level: l2 }) => {
                assert!(
                    Level::is_def_eq(l1, l2),
                    "Levels should match after roundtrip"
                );
            }
            _ => panic!("Certificate type mismatch after roundtrip"),
        }
    }

    #[tokio::test]
    async fn test_archive_invalid_algorithm() {
        use lean5_kernel::Level;

        let state = ServerState::new();
        let level = Level::zero();
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let params = ArchiveCertParams {
            cert,
            algorithm: Some("invalid_algo".to_string()),
            level: None,
            include_stats: false,
        };

        let response = handle_archive_cert(&state, RequestId::Number(1), params).await;
        assert!(response.error.is_some());
        assert!(response
            .error
            .unwrap()
            .message
            .contains("Unknown algorithm"));
    }

    #[tokio::test]
    async fn test_unarchive_invalid_base64() {
        let state = ServerState::new();

        let params = UnarchiveCertParams {
            archive: "not valid base64!!!".to_string(),
        };

        let response = handle_unarchive_cert(&state, RequestId::Number(1), params).await;
        assert!(response.error.is_some());
        assert!(response.error.unwrap().message.contains("Invalid base64"));
    }

    #[tokio::test]
    async fn test_server_info_includes_cert_methods() {
        let state = ServerState::new();
        let response = handle_server_info(&state, RequestId::Number(1)).await;

        assert!(response.error.is_none());
        let info: ServerInfo = serde_json::from_value(response.result.unwrap()).unwrap();

        // Verify all certificate methods are listed
        assert!(
            info.methods.contains(&"compressCert".to_string()),
            "serverInfo should include compressCert"
        );
        assert!(
            info.methods.contains(&"decompressCert".to_string()),
            "serverInfo should include decompressCert"
        );
        assert!(
            info.methods.contains(&"archiveCert".to_string()),
            "serverInfo should include archiveCert"
        );
        assert!(
            info.methods.contains(&"unarchiveCert".to_string()),
            "serverInfo should include unarchiveCert"
        );
    }

    // ========================================================================
    // Dictionary Compression Tests
    // ========================================================================

    /// Helper to create sample certificates for dictionary training
    fn create_sample_certs(count: usize) -> Vec<ProofCert> {
        use lean5_kernel::Level;

        (0..count)
            .map(|i| {
                // Create varied certificates with different universe levels
                let level = if i % 3 == 0 {
                    Level::zero()
                } else if i % 3 == 1 {
                    Level::succ(Level::zero())
                } else {
                    Level::succ(Level::succ(Level::zero()))
                };
                ProofCert::Sort { level }
            })
            .collect()
    }

    #[tokio::test]
    async fn test_train_dict_basic() {
        let state = ServerState::new();
        let samples = create_sample_certs(10);

        let params = TrainDictParams {
            samples,
            max_size: None,
            level: None,
        };

        let response = handle_train_dict(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );

        let result: TrainDictResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.sample_count, 10);
        assert!(result.size > 0, "Dictionary should have non-zero size");
        assert_eq!(result.target_level, 3); // Default level
        assert!(result.dict_id != 0, "Dictionary should have non-zero ID");
        assert!(
            !result.dictionary.is_empty(),
            "Dictionary base64 should not be empty"
        );
    }

    #[tokio::test]
    async fn test_train_dict_custom_params() {
        let state = ServerState::new();
        let samples = create_sample_certs(10);

        let params = TrainDictParams {
            samples,
            max_size: Some(16 * 1024), // 16KB
            level: Some(5),
        };

        let response = handle_train_dict(&state, RequestId::Number(1), params).await;
        assert!(response.error.is_none());

        let result: TrainDictResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(
            result.size <= 16 * 1024,
            "Dictionary should respect max_size"
        );
        assert_eq!(result.target_level, 5);
    }

    #[tokio::test]
    async fn test_train_dict_not_enough_samples() {
        let state = ServerState::new();
        let samples = create_sample_certs(3); // Less than minimum 5

        let params = TrainDictParams {
            samples,
            max_size: None,
            level: None,
        };

        let response = handle_train_dict(&state, RequestId::Number(1), params).await;
        assert!(response.error.is_some());
        let error = response.error.unwrap();
        assert!(
            error.message.contains("Not enough samples")
                || error.message.contains("training failed"),
            "Expected not enough samples error, got: {}",
            error.message
        );
    }

    #[tokio::test]
    async fn test_archive_cert_with_dict_basic() {
        use lean5_kernel::Level;

        let state = ServerState::new();
        let samples = create_sample_certs(10);

        // First train a dictionary
        let train_params = TrainDictParams {
            samples,
            max_size: None,
            level: None,
        };
        let train_response = handle_train_dict(&state, RequestId::Number(1), train_params).await;
        assert!(train_response.error.is_none());
        let train_result: TrainDictResult =
            serde_json::from_value(train_response.result.unwrap()).unwrap();

        // Now archive a certificate with the dictionary
        let level = Level::zero();
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let archive_params = ArchiveCertWithDictParams {
            cert,
            dictionary: train_result.dictionary,
            level: None,
            include_stats: false,
        };

        let archive_response =
            handle_archive_cert_with_dict(&state, RequestId::Number(2), archive_params).await;
        assert!(
            archive_response.error.is_none(),
            "Unexpected error: {:?}",
            archive_response.error
        );

        let archive_result: ArchiveCertWithDictResult =
            serde_json::from_value(archive_response.result.unwrap()).unwrap();
        assert_eq!(archive_result.dict_id, train_result.dict_id);
        assert!(archive_result.original_size > 0);
        assert!(archive_result.compressed_size > 0);
        assert!(archive_result.compression_ratio > 0.0);
        assert!(!archive_result.archive.is_empty());
    }

    #[tokio::test]
    async fn test_archive_cert_with_dict_with_stats() {
        use lean5_kernel::Level;

        let state = ServerState::new();
        let samples = create_sample_certs(10);

        // Train dictionary
        let train_params = TrainDictParams {
            samples,
            max_size: None,
            level: None,
        };
        let train_response = handle_train_dict(&state, RequestId::Number(1), train_params).await;
        let train_result: TrainDictResult =
            serde_json::from_value(train_response.result.unwrap()).unwrap();

        // Archive with stats
        let level = Level::zero();
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let archive_params = ArchiveCertWithDictParams {
            cert,
            dictionary: train_result.dictionary,
            level: Some(5),
            include_stats: true,
        };

        let archive_response =
            handle_archive_cert_with_dict(&state, RequestId::Number(2), archive_params).await;
        assert!(archive_response.error.is_none());

        let archive_result: ArchiveCertWithDictResult =
            serde_json::from_value(archive_response.result.unwrap()).unwrap();
        assert!(
            archive_result.structure_shared_size.is_some(),
            "Stats should include structure_shared_size when include_stats=true"
        );
        assert_eq!(archive_result.compression_level, 5);
    }

    #[tokio::test]
    async fn test_unarchive_cert_with_dict_roundtrip() {
        use lean5_kernel::Level;

        let state = ServerState::new();
        let samples = create_sample_certs(10);

        // Train dictionary
        let train_params = TrainDictParams {
            samples,
            max_size: None,
            level: None,
        };
        let train_response = handle_train_dict(&state, RequestId::Number(1), train_params).await;
        let train_result: TrainDictResult =
            serde_json::from_value(train_response.result.unwrap()).unwrap();

        // Create and archive certificate
        let level = Level::succ(Level::zero());
        let original_cert = ProofCert::Sort {
            level: level.clone(),
        };

        let archive_params = ArchiveCertWithDictParams {
            cert: original_cert.clone(),
            dictionary: train_result.dictionary.clone(),
            level: None,
            include_stats: false,
        };

        let archive_response =
            handle_archive_cert_with_dict(&state, RequestId::Number(2), archive_params).await;
        let archive_result: ArchiveCertWithDictResult =
            serde_json::from_value(archive_response.result.unwrap()).unwrap();

        // Unarchive
        let unarchive_params = UnarchiveCertWithDictParams {
            archive: archive_result.archive,
            dictionary: train_result.dictionary,
        };

        let unarchive_response =
            handle_unarchive_cert_with_dict(&state, RequestId::Number(3), unarchive_params).await;
        assert!(
            unarchive_response.error.is_none(),
            "Unexpected error: {:?}",
            unarchive_response.error
        );

        let unarchive_result: UnarchiveCertWithDictResult =
            serde_json::from_value(unarchive_response.result.unwrap()).unwrap();
        assert_eq!(unarchive_result.dict_id, train_result.dict_id);

        // Verify the certificate matches
        match (&original_cert, &unarchive_result.cert) {
            (ProofCert::Sort { level: l1 }, ProofCert::Sort { level: l2 }) => {
                assert!(
                    Level::is_def_eq(l1, l2),
                    "Levels should match after roundtrip"
                );
            }
            _ => panic!("Certificate type mismatch after roundtrip"),
        }
    }

    #[tokio::test]
    async fn test_unarchive_cert_with_dict_invalid_dict() {
        let state = ServerState::new();

        let params = UnarchiveCertWithDictParams {
            archive: "SGVsbG8gV29ybGQ=".to_string(), // "Hello World" in base64
            dictionary: "bm90IGEgcmVhbCBkaWN0".to_string(), // Invalid dictionary
        };

        let response = handle_unarchive_cert_with_dict(&state, RequestId::Number(1), params).await;
        assert!(response.error.is_some());
        let error = response.error.unwrap();
        assert!(
            error.message.contains("Invalid dictionary")
                || error.message.contains("Invalid archive"),
            "Expected format error, got: {}",
            error.message
        );
    }

    #[tokio::test]
    async fn test_unarchive_cert_with_dict_wrong_dict() {
        use lean5_kernel::Level;

        let state = ServerState::new();

        // Train two different dictionaries
        let samples1 = create_sample_certs(10);
        let samples2 = create_sample_certs(15); // Different sample count for variety

        let train_params1 = TrainDictParams {
            samples: samples1,
            max_size: Some(8 * 1024),
            level: Some(1),
        };
        let train_response1 = handle_train_dict(&state, RequestId::Number(1), train_params1).await;
        let train_result1: TrainDictResult =
            serde_json::from_value(train_response1.result.unwrap()).unwrap();

        let train_params2 = TrainDictParams {
            samples: samples2,
            max_size: Some(16 * 1024),
            level: Some(9),
        };
        let train_response2 = handle_train_dict(&state, RequestId::Number(2), train_params2).await;
        let train_result2: TrainDictResult =
            serde_json::from_value(train_response2.result.unwrap()).unwrap();

        // Archive with dict1
        let level = Level::zero();
        let cert = ProofCert::Sort { level };

        let archive_params = ArchiveCertWithDictParams {
            cert,
            dictionary: train_result1.dictionary,
            level: None,
            include_stats: false,
        };

        let archive_response =
            handle_archive_cert_with_dict(&state, RequestId::Number(3), archive_params).await;
        let archive_result: ArchiveCertWithDictResult =
            serde_json::from_value(archive_response.result.unwrap()).unwrap();

        // Try to unarchive with dict2 (wrong dictionary)
        let unarchive_params = UnarchiveCertWithDictParams {
            archive: archive_result.archive,
            dictionary: train_result2.dictionary,
        };

        let unarchive_response =
            handle_unarchive_cert_with_dict(&state, RequestId::Number(4), unarchive_params).await;
        assert!(
            unarchive_response.error.is_some(),
            "Should fail with wrong dictionary"
        );
    }

    #[tokio::test]
    async fn test_server_info_includes_dict_methods() {
        let state = ServerState::new();
        let response = handle_server_info(&state, RequestId::Number(1)).await;

        assert!(response.error.is_none());
        let info: ServerInfo = serde_json::from_value(response.result.unwrap()).unwrap();

        // Verify dictionary methods are listed
        assert!(
            info.methods.contains(&"trainDict".to_string()),
            "serverInfo should include trainDict"
        );
        assert!(
            info.methods.contains(&"archiveCertWithDict".to_string()),
            "serverInfo should include archiveCertWithDict"
        );
        assert!(
            info.methods.contains(&"unarchiveCertWithDict".to_string()),
            "serverInfo should include unarchiveCertWithDict"
        );
    }

    #[tokio::test]
    async fn test_dict_json_serialization() {
        use lean5_kernel::Level;

        let state = ServerState::new();
        let samples = create_sample_certs(10);

        // Train dictionary
        let train_params = TrainDictParams {
            samples,
            max_size: None,
            level: None,
        };

        // Verify TrainDictParams can be serialized (for API usage)
        let params_json = serde_json::to_value(&train_params).unwrap();
        assert!(params_json.get("samples").is_some());

        let train_response = handle_train_dict(&state, RequestId::Number(1), train_params).await;
        let train_result: TrainDictResult =
            serde_json::from_value(train_response.result.unwrap()).unwrap();

        // Verify result can be re-serialized
        let result_json = serde_json::to_value(&train_result).unwrap();
        assert!(result_json.get("dictionary").is_some());
        assert!(result_json.get("dict_id").is_some());

        // Archive params serialization
        let level = Level::zero();
        let cert = ProofCert::Sort { level };

        let archive_params = ArchiveCertWithDictParams {
            cert,
            dictionary: train_result.dictionary.clone(),
            level: Some(5),
            include_stats: true,
        };

        let archive_params_json = serde_json::to_value(&archive_params).unwrap();
        assert!(archive_params_json.get("cert").is_some());
        assert!(archive_params_json.get("dictionary").is_some());
    }

    // ========================================================================
    // getConfig Tests
    // ========================================================================

    #[tokio::test]
    async fn test_get_config_default() {
        let state = ServerState::new();
        let response = handle_get_config(&state, RequestId::Number(1)).await;

        let result: GetConfigResult = serde_json::from_value(response.result.unwrap()).unwrap();

        assert_eq!(result.worker_threads, 0); // Default is auto
        assert!(result.effective_threads > 0); // Should be at least 1
        assert!(!result.gpu_enabled);
        assert_eq!(result.default_timeout_ms, 5000);
    }

    #[tokio::test]
    async fn test_get_config_with_worker_threads() {
        let state = ServerState::new().with_worker_threads(4);
        let response = handle_get_config(&state, RequestId::Number(1)).await;

        let result: GetConfigResult = serde_json::from_value(response.result.unwrap()).unwrap();

        assert_eq!(result.worker_threads, 4);
        assert_eq!(result.effective_threads, 4);
    }

    #[tokio::test]
    async fn test_get_config_json_serialization() {
        let result = GetConfigResult {
            gpu_enabled: true,
            default_timeout_ms: 10000,
            worker_threads: 8,
            effective_threads: 8,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"worker_threads\":8"));
        assert!(json.contains("\"effective_threads\":8"));
        assert!(json.contains("\"gpu_enabled\":true"));
    }

    // ========================================================================
    // Thread Pool Configuration Tests
    // ========================================================================

    #[tokio::test]
    async fn test_batch_verify_cert_with_threads_param() {
        // Test that threads parameter in request is respected
        let state = ServerState::new();

        // Create a valid certificate
        let level = Level::zero();
        let expr = Expr::Sort(level.clone());
        let cert = ProofCert::Sort { level };

        let params = BatchVerifyCertParams {
            items: vec![BatchVerifyCertItem {
                id: "test1".to_string(),
                cert,
                expr,
            }],
            threads: 2, // Request 2 threads
            timeout_ms: None,
        };

        let response = handle_batch_verify_cert(&state, RequestId::Number(1), params, None).await;
        let result: BatchVerifyCertResult =
            serde_json::from_value(response.result.unwrap()).unwrap();

        assert_eq!(result.stats.total, 1);
        assert_eq!(result.stats.successful, 1);
    }

    #[tokio::test]
    async fn test_batch_verify_cert_uses_server_worker_threads() {
        // Test that server config worker_threads is used when request threads=0
        let state = ServerState::new().with_worker_threads(2);

        // Create a valid certificate
        let level = Level::zero();
        let expr = Expr::Sort(level.clone());
        let cert = ProofCert::Sort { level };

        let params = BatchVerifyCertParams {
            items: vec![BatchVerifyCertItem {
                id: "test1".to_string(),
                cert,
                expr,
            }],
            threads: 0, // Use server default
            timeout_ms: None,
        };

        let response = handle_batch_verify_cert(&state, RequestId::Number(1), params, None).await;
        let result: BatchVerifyCertResult =
            serde_json::from_value(response.result.unwrap()).unwrap();

        // Should succeed using server's 2 threads
        assert_eq!(result.stats.total, 1);
        assert_eq!(result.stats.successful, 1);
    }

    #[tokio::test]
    async fn test_server_info_includes_get_config() {
        let state = ServerState::new();
        let response = handle_server_info(&state, RequestId::Number(1)).await;

        let info: ServerInfo = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(info.methods.contains(&"getConfig".to_string()));
    }

    #[tokio::test]
    async fn test_server_info_includes_get_metrics() {
        let state = ServerState::new();
        let response = handle_server_info(&state, RequestId::Number(1)).await;

        let info: ServerInfo = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(info.methods.contains(&"getMetrics".to_string()));
    }

    #[tokio::test]
    async fn test_get_metrics_basic() {
        let state = ServerState::new();
        let response = handle_get_metrics(&state, RequestId::Number(1)).await;

        assert!(response.error.is_none());
        let result: GetMetricsResult = serde_json::from_value(response.result.unwrap()).unwrap();

        // Fresh server should have zero requests
        assert_eq!(result.total_requests, 0);
        assert_eq!(result.successful_requests, 0);
        assert_eq!(result.failed_requests, 0);
        assert_eq!(result.success_rate, 1.0); // 1.0 when no requests
        assert_eq!(result.avg_latency_us, 0);

        // Method counts should all be zero
        assert_eq!(result.method_counts.check, 0);
        assert_eq!(result.method_counts.prove, 0);
        assert_eq!(result.method_counts.get_type, 0);
        assert_eq!(result.method_counts.batch_check, 0);
        assert_eq!(result.method_counts.verify_cert, 0);
        assert_eq!(result.method_counts.batch_verify_cert, 0);
        assert_eq!(result.method_counts.verify_c, 0);

        // Batch stats should be zero
        assert_eq!(result.batch_stats.items_processed, 0);
        assert_eq!(result.batch_stats.certificates_verified, 0);
    }

    #[tokio::test]
    async fn test_metrics_record_request() {
        let metrics = ServerMetrics::new();

        // Record some requests
        metrics.record_request("check", true, 100);
        metrics.record_request("check", true, 200);
        metrics.record_request("prove", false, 500);

        assert_eq!(metrics.total_requests.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.successful_requests.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.failed_requests.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.check_requests.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.prove_requests.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.cumulative_time_us.load(Ordering::Relaxed), 800);
        assert_eq!(metrics.avg_latency_us(), 266); // 800 / 3
    }

    #[tokio::test]
    async fn test_metrics_batch_tracking() {
        let metrics = ServerMetrics::new();

        metrics.record_batch_items(50);
        metrics.record_batch_items(100);
        metrics.record_certs_verified(10);

        assert_eq!(metrics.batch_items_processed.load(Ordering::Relaxed), 150);
        assert_eq!(metrics.certificates_verified.load(Ordering::Relaxed), 10);
    }

    #[tokio::test]
    async fn test_metrics_timing_breakdown() {
        let metrics = ServerMetrics::new();

        metrics.record_type_check_time(1000);
        metrics.record_cert_verify_time(500);
        metrics.record_type_check_time(500);

        assert_eq!(metrics.type_check_time_us.load(Ordering::Relaxed), 1500);
        assert_eq!(metrics.cert_verify_time_us.load(Ordering::Relaxed), 500);
    }

    #[tokio::test]
    async fn test_get_metrics_json_serialization() {
        let state = ServerState::new();
        let response = handle_get_metrics(&state, RequestId::Number(1)).await;

        let result: GetMetricsResult = serde_json::from_value(response.result.unwrap()).unwrap();

        // Verify JSON round-trip works
        let json = serde_json::to_string(&result).unwrap();
        let _: GetMetricsResult = serde_json::from_str(&json).unwrap();
    }

    #[tokio::test]
    async fn test_metrics_success_rate() {
        let metrics = ServerMetrics::new();

        // 0 requests should return 1.0
        assert_eq!(metrics.success_rate(), 1.0);

        // Add some requests
        metrics.record_request("check", true, 100);
        metrics.record_request("check", true, 100);
        metrics.record_request("check", false, 100);
        metrics.record_request("check", false, 100);

        // 2 successful out of 4 = 0.5
        assert_eq!(metrics.success_rate(), 0.5);
    }

    #[tokio::test]
    async fn test_handler_metrics_integration_check() {
        let state = ServerState::new();

        // Initially no requests
        assert_eq!(state.metrics.check_requests.load(Ordering::Relaxed), 0);
        assert_eq!(state.metrics.total_requests.load(Ordering::Relaxed), 0);

        // Call check handler
        let params = CheckParams {
            code: "Type".to_string(),
            timeout_ms: None,
        };
        let _response = handle_check(&state, RequestId::Number(1), params).await;

        // Verify metrics were recorded
        assert_eq!(state.metrics.check_requests.load(Ordering::Relaxed), 1);
        assert_eq!(state.metrics.total_requests.load(Ordering::Relaxed), 1);
        assert_eq!(state.metrics.successful_requests.load(Ordering::Relaxed), 1);
        assert!(state.metrics.cumulative_time_us.load(Ordering::Relaxed) > 0);
    }

    #[tokio::test]
    async fn test_handler_metrics_integration_prove() {
        let state = ServerState::new();

        // Initially no requests
        assert_eq!(state.metrics.prove_requests.load(Ordering::Relaxed), 0);

        // Call prove handler with a simple goal
        let params = ProveParams {
            goal: "Type".to_string(),
            hypotheses: vec![],
            timeout_ms: Some(100),
        };
        let _response = handle_prove(&state, RequestId::Number(1), params).await;

        // Verify metrics were recorded
        assert_eq!(state.metrics.prove_requests.load(Ordering::Relaxed), 1);
        assert_eq!(state.metrics.total_requests.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_handler_metrics_integration_get_type() {
        let state = ServerState::new();

        // Initially no requests
        assert_eq!(state.metrics.get_type_requests.load(Ordering::Relaxed), 0);

        // Call get_type handler
        let params = GetTypeParams {
            expr: "Type".to_string(),
        };
        let _response = handle_get_type(&state, RequestId::Number(1), params).await;

        // Verify metrics were recorded
        assert_eq!(state.metrics.get_type_requests.load(Ordering::Relaxed), 1);
        assert_eq!(state.metrics.total_requests.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_handler_metrics_integration_batch_check() {
        let state = ServerState::new();

        // Initially no requests
        assert_eq!(
            state.metrics.batch_check_requests.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            state.metrics.batch_items_processed.load(Ordering::Relaxed),
            0
        );

        // Call batch_check handler with 3 items
        let params = BatchCheckParams {
            items: vec![
                BatchCheckItem {
                    id: "1".to_string(),
                    code: "Type".to_string(),
                },
                BatchCheckItem {
                    id: "2".to_string(),
                    code: "Prop".to_string(),
                },
                BatchCheckItem {
                    id: "3".to_string(),
                    code: "Type 1".to_string(),
                },
            ],
            use_gpu: false,
            timeout_ms: None,
        };
        let _response = handle_batch_check(&state, RequestId::Number(1), params, None).await;

        // Verify metrics were recorded
        assert_eq!(
            state.metrics.batch_check_requests.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            state.metrics.batch_items_processed.load(Ordering::Relaxed),
            3
        );
        assert_eq!(state.metrics.total_requests.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_handler_metrics_integration_verify_cert() {
        let state = ServerState::new();

        // Initially no requests
        assert_eq!(
            state.metrics.verify_cert_requests.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            state.metrics.certificates_verified.load(Ordering::Relaxed),
            0
        );

        // Call verify_cert handler with a simple Sort certificate
        let params = VerifyCertParams {
            cert: ProofCert::Sort { level: Level::Zero },
            expr: Expr::Sort(Level::Zero),
            timeout_ms: None,
        };
        let _response = handle_verify_cert(&state, RequestId::Number(1), params).await;

        // Verify metrics were recorded
        assert_eq!(
            state.metrics.verify_cert_requests.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            state.metrics.certificates_verified.load(Ordering::Relaxed),
            1
        );
        assert_eq!(state.metrics.total_requests.load(Ordering::Relaxed), 1);
        // Note: cert_verify_time_us not asserted > 0 because in optimized builds,
        // operations can complete in sub-microsecond time, rounding to 0
    }

    #[tokio::test]
    async fn test_handler_metrics_visible_in_get_metrics() {
        let state = ServerState::new();

        // Make some calls to exercise metrics
        let check_params = CheckParams {
            code: "Type".to_string(),
            timeout_ms: None,
        };
        let _ = handle_check(&state, RequestId::Number(1), check_params).await;

        let get_type_params = GetTypeParams {
            expr: "Prop".to_string(),
        };
        let _ = handle_get_type(&state, RequestId::Number(2), get_type_params).await;

        // Now call get_metrics and verify the results
        let response = handle_get_metrics(&state, RequestId::Number(3)).await;
        let result: GetMetricsResult = serde_json::from_value(response.result.unwrap()).unwrap();

        // Verify metrics are visible
        assert_eq!(result.total_requests, 2);
        assert_eq!(result.successful_requests, 2);
        assert_eq!(result.method_counts.check, 1);
        assert_eq!(result.method_counts.get_type, 1);
        // Note: avg_latency_us not asserted > 0 because in optimized builds,
        // operations can complete in sub-microsecond time, rounding to 0
    }

    // ========================================================================
    // Environment Handler Tests
    // ========================================================================

    #[tokio::test]
    async fn test_get_environment_empty() {
        let state = ServerState::new();

        let params = GetEnvironmentParams {
            include_json: false,
        };

        let response = handle_get_environment(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );

        let result: GetEnvironmentResult =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.num_constants, 0);
        assert_eq!(result.num_inductives, 0);
        assert!(result.constant_names.is_empty());
        assert!(result.json.is_none());
    }

    #[tokio::test]
    async fn test_get_environment_with_json() {
        let state = ServerState::new();

        let params = GetEnvironmentParams { include_json: true };

        let response = handle_get_environment(&state, RequestId::Number(1), params).await;
        assert!(response.error.is_none());

        let result: GetEnvironmentResult =
            serde_json::from_value(response.result.unwrap()).unwrap();
        // JSON should be present (even if empty environment)
        assert!(result.json.is_some());
        let json_str = result.json.unwrap();
        assert!(!json_str.is_empty());
    }

    #[tokio::test]
    async fn test_get_environment_json_serialization() {
        let result = GetEnvironmentResult {
            num_constants: 5,
            num_inductives: 3,
            constant_names: vec!["Nat".to_string(), "Bool".to_string()],
            json: Some("{\"test\": true}".to_string()),
        };

        let json = serde_json::to_string(&result).expect("Should serialize");
        let deserialized: GetEnvironmentResult =
            serde_json::from_str(&json).expect("Should deserialize");

        assert_eq!(deserialized.num_constants, 5);
        assert_eq!(deserialized.num_inductives, 3);
        assert_eq!(deserialized.constant_names.len(), 2);
        assert!(deserialized.json.is_some());
    }

    #[tokio::test]
    async fn test_save_environment_bincode() {
        use std::fs;

        let state = ServerState::new();
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("lean5_test_env_{}.bincode", std::process::id()));

        let params = SaveEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("bincode".to_string()),
        };

        let response = handle_save_environment(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );

        let result: SaveEnvironmentResult =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.success);
        assert_eq!(result.num_constants, 0);
        assert_eq!(result.num_inductives, 0);
        assert!(result.file_size > 0);

        // Cleanup
        let _ = fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_save_environment_json() {
        use std::fs;

        let state = ServerState::new();
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("lean5_test_env_{}.json", std::process::id()));

        let params = SaveEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("json".to_string()),
        };

        let response = handle_save_environment(&state, RequestId::Number(1), params).await;
        assert!(
            response.error.is_none(),
            "Unexpected error: {:?}",
            response.error
        );

        let result: SaveEnvironmentResult =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.success);

        // Verify the file is valid JSON
        let content = fs::read_to_string(&temp_file).expect("Should read file");
        let _: serde_json::Value = serde_json::from_str(&content).expect("Should be valid JSON");

        // Cleanup
        let _ = fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_save_environment_default_format() {
        use std::fs;

        let state = ServerState::new();
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("lean5_test_env_default_{}.bin", std::process::id()));

        let params = SaveEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: None, // Should default to bincode
        };

        let response = handle_save_environment(&state, RequestId::Number(1), params).await;
        assert!(response.error.is_none());

        let result: SaveEnvironmentResult =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.success);

        // Cleanup
        let _ = fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_save_environment_invalid_path() {
        let state = ServerState::new();

        let params = SaveEnvironmentParams {
            path: "/nonexistent/directory/that/does/not/exist/file.bin".to_string(),
            format: None,
        };

        let response = handle_save_environment(&state, RequestId::Number(1), params).await;
        // Should return an RPC error
        assert!(response.error.is_some());
        assert!(response.error.unwrap().message.contains("Failed to save"));
    }

    #[tokio::test]
    async fn test_load_environment_bincode() {
        use std::fs;

        let state = ServerState::new();
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("lean5_test_load_{}.bincode", std::process::id()));

        // First save an environment
        let save_params = SaveEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("bincode".to_string()),
        };
        let save_response =
            handle_save_environment(&state, RequestId::Number(1), save_params).await;
        assert!(save_response.error.is_none());

        // Then load it
        let load_params = LoadEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("bincode".to_string()),
            replace: true,
        };
        let load_response =
            handle_load_environment(&state, RequestId::Number(2), load_params).await;
        assert!(
            load_response.error.is_none(),
            "Unexpected error: {:?}",
            load_response.error
        );

        let result: LoadEnvironmentResult =
            serde_json::from_value(load_response.result.unwrap()).unwrap();
        assert!(result.success);
        assert_eq!(result.num_constants, 0);
        assert_eq!(result.num_inductives, 0);

        // Cleanup
        let _ = fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_load_environment_json() {
        use std::fs;

        let state = ServerState::new();
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("lean5_test_load_{}.json", std::process::id()));

        // First save as JSON
        let save_params = SaveEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("json".to_string()),
        };
        let save_response =
            handle_save_environment(&state, RequestId::Number(1), save_params).await;
        assert!(save_response.error.is_none());

        // Then load it
        let load_params = LoadEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("json".to_string()),
            replace: true,
        };
        let load_response =
            handle_load_environment(&state, RequestId::Number(2), load_params).await;
        assert!(load_response.error.is_none());

        let result: LoadEnvironmentResult =
            serde_json::from_value(load_response.result.unwrap()).unwrap();
        assert!(result.success);

        // Cleanup
        let _ = fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_load_environment_nonexistent_file() {
        let state = ServerState::new();

        let params = LoadEnvironmentParams {
            path: "/nonexistent/file/that/does/not/exist.bin".to_string(),
            format: None,
            replace: true,
        };

        let response = handle_load_environment(&state, RequestId::Number(1), params).await;
        assert!(response.error.is_some());
        assert!(response.error.unwrap().message.contains("Failed to load"));
    }

    #[tokio::test]
    async fn test_load_environment_replace_flag() {
        use std::fs;

        let state = ServerState::new();
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("lean5_test_replace_{}.bincode", std::process::id()));

        // Save environment
        let save_params = SaveEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("bincode".to_string()),
        };
        let _ = handle_save_environment(&state, RequestId::Number(1), save_params).await;

        // Load with replace=true
        let load_params = LoadEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("bincode".to_string()),
            replace: true,
        };
        let response = handle_load_environment(&state, RequestId::Number(2), load_params).await;
        assert!(response.error.is_none());

        // Load with replace=false (currently same behavior)
        let load_params2 = LoadEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("bincode".to_string()),
            replace: false,
        };
        let response2 = handle_load_environment(&state, RequestId::Number(3), load_params2).await;
        assert!(response2.error.is_none());

        // Cleanup
        let _ = fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_save_load_roundtrip_bincode() {
        use std::fs;

        let state = ServerState::new();
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!(
            "lean5_test_roundtrip_{}.bincode",
            std::process::id()
        ));

        // Get initial state
        let get_params1 = GetEnvironmentParams {
            include_json: false,
        };
        let get_response1 = handle_get_environment(&state, RequestId::Number(1), get_params1).await;
        let initial: GetEnvironmentResult =
            serde_json::from_value(get_response1.result.unwrap()).unwrap();

        // Save
        let save_params = SaveEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("bincode".to_string()),
        };
        let _ = handle_save_environment(&state, RequestId::Number(2), save_params).await;

        // Load into fresh state
        let state2 = ServerState::new();
        let load_params = LoadEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("bincode".to_string()),
            replace: true,
        };
        let _ = handle_load_environment(&state2, RequestId::Number(3), load_params).await;

        // Verify state matches
        let get_params2 = GetEnvironmentParams {
            include_json: false,
        };
        let get_response2 =
            handle_get_environment(&state2, RequestId::Number(4), get_params2).await;
        let final_state: GetEnvironmentResult =
            serde_json::from_value(get_response2.result.unwrap()).unwrap();

        assert_eq!(initial.num_constants, final_state.num_constants);
        assert_eq!(initial.num_inductives, final_state.num_inductives);

        // Cleanup
        let _ = fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_save_load_roundtrip_json() {
        use std::fs;

        let state = ServerState::new();
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("lean5_test_roundtrip_{}.json", std::process::id()));

        // Save as JSON
        let save_params = SaveEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("json".to_string()),
        };
        let save_response =
            handle_save_environment(&state, RequestId::Number(1), save_params).await;
        assert!(save_response.error.is_none());

        // Load from JSON
        let load_params = LoadEnvironmentParams {
            path: temp_file.to_string_lossy().to_string(),
            format: Some("json".to_string()),
            replace: true,
        };
        let load_response =
            handle_load_environment(&state, RequestId::Number(2), load_params).await;
        assert!(load_response.error.is_none());

        // Cleanup
        let _ = fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_save_environment_result_json_serialization() {
        let result = SaveEnvironmentResult {
            success: true,
            num_constants: 42,
            num_inductives: 7,
            file_size: 1024,
        };

        let json = serde_json::to_string(&result).expect("Should serialize");
        let deserialized: SaveEnvironmentResult =
            serde_json::from_str(&json).expect("Should deserialize");

        assert_eq!(deserialized.success, result.success);
        assert_eq!(deserialized.num_constants, result.num_constants);
        assert_eq!(deserialized.num_inductives, result.num_inductives);
        assert_eq!(deserialized.file_size, result.file_size);
    }

    #[tokio::test]
    async fn test_load_environment_result_json_serialization() {
        let result = LoadEnvironmentResult {
            success: true,
            num_constants: 100,
            num_inductives: 20,
        };

        let json = serde_json::to_string(&result).expect("Should serialize");
        let deserialized: LoadEnvironmentResult =
            serde_json::from_str(&json).expect("Should deserialize");

        assert_eq!(deserialized.success, result.success);
        assert_eq!(deserialized.num_constants, result.num_constants);
        assert_eq!(deserialized.num_inductives, result.num_inductives);
    }

    #[tokio::test]
    async fn test_server_info_includes_environment_methods() {
        let state = ServerState::new();
        let response = handle_server_info(&state, RequestId::Number(1)).await;

        let info: ServerInfo = serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(
            info.methods.contains(&"saveEnvironment".to_string()),
            "serverInfo should include saveEnvironment"
        );
        assert!(
            info.methods.contains(&"loadEnvironment".to_string()),
            "serverInfo should include loadEnvironment"
        );
        assert!(
            info.methods.contains(&"getEnvironment".to_string()),
            "serverInfo should include getEnvironment"
        );
    }
}
