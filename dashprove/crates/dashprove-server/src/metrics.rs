//! Prometheus-compatible metrics for DashProve server
//!
//! Provides lightweight in-memory metrics collection and a `/metrics` endpoint
//! that outputs Prometheus text format for scraping.
//!
//! # Metrics Exposed
//!
//! - `dashprove_http_requests_total` - Counter of HTTP requests by method and path
//! - `dashprove_http_request_duration_seconds` - Histogram of request durations
//! - `dashprove_active_http_requests` - Gauge of in-flight HTTP requests
//! - `dashprove_active_websocket_sessions` - Gauge of active WebSocket sessions
//! - `dashprove_cache_entries` - Gauge of cache entries (total, valid, expired)
//! - `dashprove_uptime_seconds` - Gauge of server uptime in seconds
//! - `dashprove_verification_duration_seconds` - Histogram of overall verification durations
//! - `dashprove_backend_verification_duration_seconds` - Histogram of verification durations by backend
//! - `dashprove_backend_verifications_success` - Counter of successful verifications by backend
//! - `dashprove_backend_verifications_failed` - Counter of failed verifications by backend

use axum::{body::Body, extract::State, http::Request, middleware::Next, response::Response};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Thread-safe metrics collector
pub struct Metrics {
    /// Server start time for uptime calculation
    start_time: Instant,

    /// Total HTTP requests by (method, path)
    request_counts: RwLock<HashMap<(String, String), u64>>,

    /// Request duration buckets by (method, path)
    /// Stores counts for each bucket: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    duration_buckets: RwLock<HashMap<(String, String), DurationHistogram>>,

    /// Total request duration sum by (method, path) in seconds
    duration_sums: RwLock<HashMap<(String, String), f64>>,

    /// Error counts by (method, path, status_code)
    error_counts: RwLock<HashMap<(String, String, u16), u64>>,

    /// Total verification requests
    verifications_total: AtomicU64,

    /// Successful verifications
    verifications_success: AtomicU64,

    /// Failed verifications
    verifications_failed: AtomicU64,

    /// Verification duration histogram (for tracking how long verifications take)
    verification_durations: RwLock<VerificationDurationHistogram>,

    /// Verification duration sum in seconds
    verification_duration_sum: RwLock<f64>,

    /// Per-backend verification duration histograms
    backend_durations: RwLock<HashMap<String, BackendDurationHistogram>>,
}

/// Histogram buckets for request duration
#[derive(Clone, Default)]
pub struct DurationHistogram {
    /// Bucket boundaries in seconds
    boundaries: Vec<f64>,
    /// Count of values <= each boundary
    bucket_counts: Vec<u64>,
    /// Total count of all observations
    count: u64,
}

impl DurationHistogram {
    fn new() -> Self {
        // Standard Prometheus histogram buckets for HTTP request latency
        let boundaries = vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ];
        let bucket_counts = vec![0; boundaries.len()];
        Self {
            boundaries,
            bucket_counts,
            count: 0,
        }
    }

    fn observe(&mut self, value_secs: f64) {
        self.count += 1;
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if value_secs <= boundary {
                self.bucket_counts[i] += 1;
            }
        }
    }
}

/// Histogram for verification durations (longer bucket ranges than HTTP requests)
#[derive(Clone, Default)]
pub struct VerificationDurationHistogram {
    /// Bucket boundaries in seconds (tuned for verification operations)
    boundaries: Vec<f64>,
    /// Count of values <= each boundary
    bucket_counts: Vec<u64>,
    /// Total count of all observations
    count: u64,
}

/// Per-backend verification metrics
#[derive(Clone, Default)]
pub struct BackendDurationHistogram {
    /// Bucket boundaries in seconds
    boundaries: Vec<f64>,
    /// Count of values <= each boundary
    bucket_counts: Vec<u64>,
    /// Total count of all observations
    count: u64,
    /// Sum of all durations for this backend
    sum: f64,
    /// Successful verifications for this backend
    success_count: u64,
    /// Failed verifications for this backend
    failure_count: u64,
}

impl BackendDurationHistogram {
    fn new() -> Self {
        // Same buckets as VerificationDurationHistogram
        let boundaries = vec![
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
        ];
        let bucket_counts = vec![0; boundaries.len()];
        Self {
            boundaries,
            bucket_counts,
            count: 0,
            sum: 0.0,
            success_count: 0,
            failure_count: 0,
        }
    }

    fn observe(&mut self, value_secs: f64, success: bool) {
        self.count += 1;
        self.sum += value_secs;
        if success {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if value_secs <= boundary {
                self.bucket_counts[i] += 1;
            }
        }
    }
}

impl VerificationDurationHistogram {
    fn new() -> Self {
        // Verification-specific buckets: from 10ms to 5 minutes
        // Typical compilation: 10ms-1s, with complex specs up to several minutes
        let boundaries = vec![
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
        ];
        let bucket_counts = vec![0; boundaries.len()];
        Self {
            boundaries,
            bucket_counts,
            count: 0,
        }
    }

    fn observe(&mut self, value_secs: f64) {
        self.count += 1;
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if value_secs <= boundary {
                self.bucket_counts[i] += 1;
            }
        }
    }
}

impl Metrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            request_counts: RwLock::new(HashMap::new()),
            duration_buckets: RwLock::new(HashMap::new()),
            duration_sums: RwLock::new(HashMap::new()),
            error_counts: RwLock::new(HashMap::new()),
            verifications_total: AtomicU64::new(0),
            verifications_success: AtomicU64::new(0),
            verifications_failed: AtomicU64::new(0),
            verification_durations: RwLock::new(VerificationDurationHistogram::new()),
            verification_duration_sum: RwLock::new(0.0),
            backend_durations: RwLock::new(HashMap::new()),
        }
    }

    /// Record a completed HTTP request
    pub async fn record_request(
        &self,
        method: &str,
        path: &str,
        status_code: u16,
        duration_secs: f64,
    ) {
        let key = (method.to_string(), normalize_path(path));

        // Increment request count
        {
            let mut counts = self.request_counts.write().await;
            *counts.entry(key.clone()).or_insert(0) += 1;
        }

        // Record duration in histogram
        {
            let mut buckets = self.duration_buckets.write().await;
            buckets
                .entry(key.clone())
                .or_insert_with(DurationHistogram::new)
                .observe(duration_secs);
        }

        // Add to duration sum
        {
            let mut sums = self.duration_sums.write().await;
            *sums.entry(key.clone()).or_insert(0.0) += duration_secs;
        }

        // Record errors (4xx and 5xx)
        if status_code >= 400 {
            let error_key = (method.to_string(), normalize_path(path), status_code);
            let mut errors = self.error_counts.write().await;
            *errors.entry(error_key).or_insert(0) += 1;
        }
    }

    /// Record a verification attempt (counters only, no duration)
    pub fn record_verification(&self, success: bool) {
        self.verifications_total.fetch_add(1, Ordering::Relaxed);
        if success {
            self.verifications_success.fetch_add(1, Ordering::Relaxed);
        } else {
            self.verifications_failed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a verification attempt with duration
    ///
    /// This method records both the success/failure counters and the duration
    /// in the verification duration histogram.
    pub async fn record_verification_with_duration(&self, success: bool, duration_secs: f64) {
        // Record counters
        self.record_verification(success);

        // Record duration in histogram
        {
            let mut hist = self.verification_durations.write().await;
            hist.observe(duration_secs);
        }

        // Add to duration sum
        {
            let mut sum = self.verification_duration_sum.write().await;
            *sum += duration_secs;
        }
    }

    /// Record a verification attempt with duration for a specific backend
    ///
    /// This method records both the overall verification metrics and per-backend
    /// duration metrics for more granular monitoring.
    pub async fn record_backend_verification(
        &self,
        backend: &str,
        success: bool,
        duration_secs: f64,
    ) {
        // Record overall metrics
        self.record_verification_with_duration(success, duration_secs)
            .await;

        // Record per-backend duration and success/failure
        self.record_backend_duration(backend, success, duration_secs)
            .await;
    }

    /// Record only the duration for a backend verification without updating
    /// global verification counters. Useful when overall verification metrics
    /// are recorded separately at the request level.
    ///
    /// Also records per-backend success/failure counts for backend-specific SLO tracking.
    pub async fn record_backend_duration(&self, backend: &str, success: bool, duration_secs: f64) {
        let mut backends = self.backend_durations.write().await;
        backends
            .entry(backend.to_string())
            .or_insert_with(BackendDurationHistogram::new)
            .observe(duration_secs, success);
    }

    /// Get server uptime in seconds
    pub fn uptime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Export metrics in Prometheus text format
    pub async fn export_prometheus(
        &self,
        active_requests: usize,
        active_websockets: usize,
        cache_total: usize,
        cache_valid: usize,
        cache_expired: usize,
    ) -> String {
        let mut output = String::with_capacity(4096);

        // Uptime gauge
        output.push_str("# HELP dashprove_uptime_seconds Server uptime in seconds\n");
        output.push_str("# TYPE dashprove_uptime_seconds gauge\n");
        output.push_str(&format!(
            "dashprove_uptime_seconds {:.3}\n",
            self.uptime_seconds()
        ));
        output.push('\n');

        // Active connections gauges
        output
            .push_str("# HELP dashprove_active_http_requests Number of in-flight HTTP requests\n");
        output.push_str("# TYPE dashprove_active_http_requests gauge\n");
        output.push_str(&format!(
            "dashprove_active_http_requests {}\n",
            active_requests
        ));
        output.push('\n');

        output.push_str(
            "# HELP dashprove_active_websocket_sessions Number of active WebSocket sessions\n",
        );
        output.push_str("# TYPE dashprove_active_websocket_sessions gauge\n");
        output.push_str(&format!(
            "dashprove_active_websocket_sessions {}\n",
            active_websockets
        ));
        output.push('\n');

        // Cache gauges
        output.push_str("# HELP dashprove_cache_entries Number of entries in proof cache\n");
        output.push_str("# TYPE dashprove_cache_entries gauge\n");
        output.push_str(&format!(
            "dashprove_cache_entries{{state=\"total\"}} {}\n",
            cache_total
        ));
        output.push_str(&format!(
            "dashprove_cache_entries{{state=\"valid\"}} {}\n",
            cache_valid
        ));
        output.push_str(&format!(
            "dashprove_cache_entries{{state=\"expired\"}} {}\n",
            cache_expired
        ));
        output.push('\n');

        // Verification counters
        output.push_str("# HELP dashprove_verifications_total Total verification requests\n");
        output.push_str("# TYPE dashprove_verifications_total counter\n");
        output.push_str(&format!(
            "dashprove_verifications_total {}\n",
            self.verifications_total.load(Ordering::Relaxed)
        ));
        output.push('\n');

        output.push_str("# HELP dashprove_verifications_success Successful verifications\n");
        output.push_str("# TYPE dashprove_verifications_success counter\n");
        output.push_str(&format!(
            "dashprove_verifications_success {}\n",
            self.verifications_success.load(Ordering::Relaxed)
        ));
        output.push('\n');

        output.push_str("# HELP dashprove_verifications_failed Failed verifications\n");
        output.push_str("# TYPE dashprove_verifications_failed counter\n");
        output.push_str(&format!(
            "dashprove_verifications_failed {}\n",
            self.verifications_failed.load(Ordering::Relaxed)
        ));
        output.push('\n');

        // HTTP request counters
        output.push_str(
            "# HELP dashprove_http_requests_total Total HTTP requests by method and path\n",
        );
        output.push_str("# TYPE dashprove_http_requests_total counter\n");
        {
            let counts = self.request_counts.read().await;
            let mut sorted_keys: Vec<_> = counts.keys().collect();
            sorted_keys.sort();
            for key in sorted_keys {
                let count = counts.get(key).unwrap_or(&0);
                output.push_str(&format!(
                    "dashprove_http_requests_total{{method=\"{}\",path=\"{}\"}} {}\n",
                    key.0, key.1, count
                ));
            }
        }
        output.push('\n');

        // HTTP error counters
        output.push_str(
            "# HELP dashprove_http_errors_total Total HTTP errors by method, path, and status\n",
        );
        output.push_str("# TYPE dashprove_http_errors_total counter\n");
        {
            let errors = self.error_counts.read().await;
            let mut sorted_keys: Vec<_> = errors.keys().collect();
            sorted_keys.sort();
            for key in sorted_keys {
                let count = errors.get(key).unwrap_or(&0);
                output.push_str(&format!(
                    "dashprove_http_errors_total{{method=\"{}\",path=\"{}\",status=\"{}\"}} {}\n",
                    key.0, key.1, key.2, count
                ));
            }
        }
        output.push('\n');

        // Request duration histogram
        output.push_str(
            "# HELP dashprove_http_request_duration_seconds HTTP request duration in seconds\n",
        );
        output.push_str("# TYPE dashprove_http_request_duration_seconds histogram\n");
        {
            let buckets = self.duration_buckets.read().await;
            let sums = self.duration_sums.read().await;
            let mut sorted_keys: Vec<_> = buckets.keys().collect();
            sorted_keys.sort();
            for key in sorted_keys {
                if let Some(histogram) = buckets.get(key) {
                    // Output bucket counts (cumulative)
                    let mut cumulative = 0u64;
                    for (i, &boundary) in histogram.boundaries.iter().enumerate() {
                        cumulative += histogram.bucket_counts[i];
                        output.push_str(&format!(
                            "dashprove_http_request_duration_seconds_bucket{{method=\"{}\",path=\"{}\",le=\"{}\"}} {}\n",
                            key.0, key.1, boundary, cumulative
                        ));
                    }
                    // +Inf bucket
                    output.push_str(&format!(
                        "dashprove_http_request_duration_seconds_bucket{{method=\"{}\",path=\"{}\",le=\"+Inf\"}} {}\n",
                        key.0, key.1, histogram.count
                    ));
                    // Sum and count
                    let sum = sums.get(key).unwrap_or(&0.0);
                    output.push_str(&format!(
                        "dashprove_http_request_duration_seconds_sum{{method=\"{}\",path=\"{}\"}} {:.6}\n",
                        key.0, key.1, sum
                    ));
                    output.push_str(&format!(
                        "dashprove_http_request_duration_seconds_count{{method=\"{}\",path=\"{}\"}} {}\n",
                        key.0, key.1, histogram.count
                    ));
                }
            }
        }
        output.push('\n');

        // Verification duration histogram
        output.push_str(
            "# HELP dashprove_verification_duration_seconds Verification request duration in seconds\n",
        );
        output.push_str("# TYPE dashprove_verification_duration_seconds histogram\n");
        {
            let histogram = self.verification_durations.read().await;
            let sum = *self.verification_duration_sum.read().await;

            // Output bucket counts (cumulative)
            let mut cumulative = 0u64;
            for (i, &boundary) in histogram.boundaries.iter().enumerate() {
                cumulative += histogram.bucket_counts[i];
                output.push_str(&format!(
                    "dashprove_verification_duration_seconds_bucket{{le=\"{}\"}} {}\n",
                    boundary, cumulative
                ));
            }
            // +Inf bucket
            output.push_str(&format!(
                "dashprove_verification_duration_seconds_bucket{{le=\"+Inf\"}} {}\n",
                histogram.count
            ));
            // Sum and count
            output.push_str(&format!(
                "dashprove_verification_duration_seconds_sum {:.6}\n",
                sum
            ));
            output.push_str(&format!(
                "dashprove_verification_duration_seconds_count {}\n",
                histogram.count
            ));
        }
        output.push('\n');

        // Per-backend verification duration histograms
        output.push_str(
            "# HELP dashprove_backend_verification_duration_seconds Verification duration by backend in seconds\n",
        );
        output.push_str("# TYPE dashprove_backend_verification_duration_seconds histogram\n");
        {
            let backends = self.backend_durations.read().await;
            let mut sorted_keys: Vec<_> = backends.keys().collect();
            sorted_keys.sort();
            for backend in sorted_keys {
                if let Some(histogram) = backends.get(backend) {
                    // Output bucket counts (cumulative)
                    let mut cumulative = 0u64;
                    for (i, &boundary) in histogram.boundaries.iter().enumerate() {
                        cumulative += histogram.bucket_counts[i];
                        output.push_str(&format!(
                            "dashprove_backend_verification_duration_seconds_bucket{{backend=\"{}\",le=\"{}\"}} {}\n",
                            backend, boundary, cumulative
                        ));
                    }
                    // +Inf bucket
                    output.push_str(&format!(
                        "dashprove_backend_verification_duration_seconds_bucket{{backend=\"{}\",le=\"+Inf\"}} {}\n",
                        backend, histogram.count
                    ));
                    // Sum and count
                    output.push_str(&format!(
                        "dashprove_backend_verification_duration_seconds_sum{{backend=\"{}\"}} {:.6}\n",
                        backend, histogram.sum
                    ));
                    output.push_str(&format!(
                        "dashprove_backend_verification_duration_seconds_count{{backend=\"{}\"}} {}\n",
                        backend, histogram.count
                    ));
                }
            }
        }
        output.push('\n');

        // Per-backend success/failure counters
        output.push_str(
            "# HELP dashprove_backend_verifications_success Successful verifications by backend\n",
        );
        output.push_str("# TYPE dashprove_backend_verifications_success counter\n");
        {
            let backends = self.backend_durations.read().await;
            let mut sorted_keys: Vec<_> = backends.keys().collect();
            sorted_keys.sort();
            for backend in sorted_keys {
                if let Some(histogram) = backends.get(backend) {
                    output.push_str(&format!(
                        "dashprove_backend_verifications_success{{backend=\"{}\"}} {}\n",
                        backend, histogram.success_count
                    ));
                }
            }
        }
        output.push('\n');

        output.push_str(
            "# HELP dashprove_backend_verifications_failed Failed verifications by backend\n",
        );
        output.push_str("# TYPE dashprove_backend_verifications_failed counter\n");
        {
            let backends = self.backend_durations.read().await;
            let mut sorted_keys: Vec<_> = backends.keys().collect();
            sorted_keys.sort();
            for backend in sorted_keys {
                if let Some(histogram) = backends.get(backend) {
                    output.push_str(&format!(
                        "dashprove_backend_verifications_failed{{backend=\"{}\"}} {}\n",
                        backend, histogram.failure_count
                    ));
                }
            }
        }

        output
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize paths for metrics aggregation
/// Replaces dynamic path segments with placeholders
fn normalize_path(path: &str) -> String {
    // Handle common dynamic path patterns
    let normalized = path
        // UUID-like segments
        .split('/')
        .map(|segment| {
            if looks_like_uuid(segment) || looks_like_id(segment) {
                ":id"
            } else {
                segment
            }
        })
        .collect::<Vec<_>>()
        .join("/");

    if normalized.is_empty() {
        "/".to_string()
    } else {
        normalized
    }
}

/// Check if a path segment looks like a UUID
fn looks_like_uuid(s: &str) -> bool {
    // Simple check: 36 chars with hyphens at positions 8,13,18,23
    if s.len() == 36 {
        let bytes = s.as_bytes();
        return bytes[8] == b'-'
            && bytes[13] == b'-'
            && bytes[18] == b'-'
            && bytes[23] == b'-'
            && s.chars().all(|c| c.is_ascii_hexdigit() || c == '-');
    }
    false
}

/// Check if a path segment looks like a numeric or short alphanumeric ID
fn looks_like_id(s: &str) -> bool {
    // Numeric IDs
    if s.chars().all(|c| c.is_ascii_digit()) && !s.is_empty() && s.len() <= 20 {
        return true;
    }
    // Short hex strings (like cache keys, session ids)
    if s.len() >= 8 && s.len() <= 64 && s.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
        return true;
    }
    false
}

/// Middleware that records request metrics
pub async fn metrics_middleware(
    State(metrics): State<Arc<Metrics>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    // Process the request
    let response = next.run(request).await;

    // Record metrics
    let duration = start.elapsed().as_secs_f64();
    let status = response.status().as_u16();

    metrics
        .record_request(&method, &path, status, duration)
        .await;

    response
}

/// Kani proofs for metrics invariants
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify request duration buckets remain monotonic after observations
    #[kani::proof]
    fn verify_request_duration_buckets_monotonic() {
        let mut hist = DurationHistogram::new();
        let values = [kani::any::<f64>(), kani::any::<f64>(), kani::any::<f64>()];

        for value in values {
            hist.observe(value);
        }

        for i in 0..hist.bucket_counts.len() - 1 {
            kani::assert(
                hist.bucket_counts[i] <= hist.bucket_counts[i + 1],
                "Bucket counts must be non-decreasing across boundaries",
            );
        }

        kani::assert(
            hist.count == values.len() as u64,
            "Count should track total observations",
        );
    }

    /// Verify backend success/failure counters stay aligned with total count
    #[kani::proof]
    fn verify_backend_success_failure_totals() {
        let mut hist = BackendDurationHistogram::new();
        let events = [
            (kani::any::<f64>(), kani::any::<bool>()),
            (kani::any::<f64>(), kani::any::<bool>()),
            (kani::any::<f64>(), kani::any::<bool>()),
        ];

        for (duration, success) in events {
            // NaN values can break floating point accumulations; limit to finite durations
            kani::assume(duration.is_finite());
            hist.observe(duration, success);
        }

        kani::assert(
            hist.success_count + hist.failure_count == hist.count,
            "Success and failure counts should equal total observations",
        );

        for i in 0..hist.bucket_counts.len() - 1 {
            kani::assert(
                hist.bucket_counts[i] <= hist.bucket_counts[i + 1],
                "Backend buckets must be cumulative",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path_basic() {
        assert_eq!(normalize_path("/health"), "/health");
        assert_eq!(normalize_path("/verify"), "/verify");
        assert_eq!(normalize_path("/cache/stats"), "/cache/stats");
    }

    #[test]
    fn test_normalize_path_uuid() {
        assert_eq!(
            normalize_path("/corpus/counterexamples/550e8400-e29b-41d4-a716-446655440000"),
            "/corpus/counterexamples/:id"
        );
    }

    #[test]
    fn test_normalize_path_numeric_id() {
        assert_eq!(normalize_path("/users/12345"), "/users/:id");
        assert_eq!(normalize_path("/items/9876543210"), "/items/:id");
    }

    #[test]
    fn test_normalize_path_hex_id() {
        // Valid hex string should be normalized
        assert_eq!(
            normalize_path("/cache/entries/abcd1234efab5678"),
            "/cache/entries/:id"
        );
    }

    #[test]
    fn test_normalize_path_empty() {
        assert_eq!(normalize_path(""), "/");
    }

    #[test]
    fn test_looks_like_uuid() {
        assert!(looks_like_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!looks_like_uuid("not-a-uuid"));
        assert!(!looks_like_uuid("health"));
    }

    #[test]
    fn test_looks_like_id() {
        assert!(looks_like_id("12345"));
        assert!(looks_like_id("abcd1234"));
        assert!(!looks_like_id("health"));
        assert!(!looks_like_id("a")); // Too short for hex
    }

    #[test]
    fn test_duration_histogram() {
        let mut hist = DurationHistogram::new();

        hist.observe(0.001); // <= 0.001 bucket
        hist.observe(0.002); // <= 0.005 bucket
        hist.observe(0.05); // <= 0.05 bucket
        hist.observe(1.5); // <= 2.5 bucket

        assert_eq!(hist.count, 4);
        // Prometheus histograms are cumulative:
        // bucket[0] (0.001): only 1 value (0.001 <= 0.001)
        assert_eq!(hist.bucket_counts[0], 1);
        // bucket[1] (0.005): 2 values (0.001 and 0.002 both <= 0.005)
        assert_eq!(hist.bucket_counts[1], 2);
        // bucket[4] (0.05): 3 values (0.001, 0.002, 0.05 all <= 0.05)
        assert_eq!(hist.bucket_counts[4], 3);
    }

    #[tokio::test]
    async fn test_metrics_record_request() {
        let metrics = Metrics::new();

        metrics.record_request("GET", "/health", 200, 0.001).await;
        metrics.record_request("GET", "/health", 200, 0.002).await;
        metrics.record_request("POST", "/verify", 200, 0.5).await;
        metrics.record_request("GET", "/invalid", 404, 0.001).await;

        let counts = metrics.request_counts.read().await;
        assert_eq!(
            counts.get(&("GET".to_string(), "/health".to_string())),
            Some(&2)
        );
        assert_eq!(
            counts.get(&("POST".to_string(), "/verify".to_string())),
            Some(&1)
        );

        let errors = metrics.error_counts.read().await;
        assert_eq!(
            errors.get(&("GET".to_string(), "/invalid".to_string(), 404)),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn test_metrics_verification_counters() {
        let metrics = Metrics::new();

        metrics.record_verification(true);
        metrics.record_verification(true);
        metrics.record_verification(false);

        assert_eq!(metrics.verifications_total.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.verifications_success.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.verifications_failed.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_export_prometheus_format() {
        let metrics = Metrics::new();

        metrics.record_request("GET", "/health", 200, 0.001).await;
        metrics.record_verification(true);

        let output = metrics.export_prometheus(5, 2, 100, 95, 5).await;

        // Check that output contains expected metrics
        assert!(output.contains("dashprove_uptime_seconds"));
        assert!(output.contains("dashprove_active_http_requests 5"));
        assert!(output.contains("dashprove_active_websocket_sessions 2"));
        assert!(output.contains("dashprove_cache_entries{state=\"total\"} 100"));
        assert!(output.contains("dashprove_cache_entries{state=\"valid\"} 95"));
        assert!(output.contains("dashprove_verifications_total 1"));
        assert!(output.contains("dashprove_http_requests_total{method=\"GET\",path=\"/health\"} 1"));
        assert!(output.contains("dashprove_http_request_duration_seconds_bucket"));
    }

    #[test]
    fn test_uptime_seconds() {
        let metrics = Metrics::new();
        // Sleep briefly to ensure non-zero uptime
        std::thread::sleep(std::time::Duration::from_millis(10));
        let uptime = metrics.uptime_seconds();
        assert!(uptime >= 0.01);
    }

    #[test]
    fn test_verification_duration_histogram() {
        let mut hist = VerificationDurationHistogram::new();

        hist.observe(0.005); // <= 0.01 bucket
        hist.observe(0.02); // <= 0.025 bucket
        hist.observe(0.5); // <= 0.5 bucket
        hist.observe(15.0); // <= 30.0 bucket

        assert_eq!(hist.count, 4);
        // Check bucket counts (non-cumulative, what's stored internally)
        // bucket[0] (0.01): 1 value (0.005 <= 0.01)
        assert_eq!(hist.bucket_counts[0], 1);
        // bucket[1] (0.025): 1 value (0.02 <= 0.025 but > 0.01)
        assert_eq!(hist.bucket_counts[1], 2); // 0.005 and 0.02
                                              // bucket[5] (0.5): 3 values (0.005, 0.02, 0.5)
        assert_eq!(hist.bucket_counts[5], 3);
    }

    #[tokio::test]
    async fn test_record_verification_with_duration() {
        let metrics = Metrics::new();

        metrics.record_verification_with_duration(true, 0.1).await;
        metrics.record_verification_with_duration(true, 0.5).await;
        metrics.record_verification_with_duration(false, 2.0).await;

        // Check counters
        assert_eq!(metrics.verifications_total.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.verifications_success.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.verifications_failed.load(Ordering::Relaxed), 1);

        // Check duration sum
        let sum = *metrics.verification_duration_sum.read().await;
        assert!((sum - 2.6).abs() < 0.001); // 0.1 + 0.5 + 2.0 = 2.6

        // Check histogram count
        let hist = metrics.verification_durations.read().await;
        assert_eq!(hist.count, 3);
    }

    #[tokio::test]
    async fn test_export_prometheus_includes_verification_duration() {
        let metrics = Metrics::new();

        metrics.record_verification_with_duration(true, 0.15).await;

        let output = metrics.export_prometheus(0, 0, 0, 0, 0).await;

        // Check that verification duration histogram is exported
        assert!(output.contains("dashprove_verification_duration_seconds"));
        assert!(output.contains("dashprove_verification_duration_seconds_bucket"));
        assert!(output.contains("dashprove_verification_duration_seconds_sum"));
        assert!(output.contains("dashprove_verification_duration_seconds_count 1"));
    }

    #[test]
    fn test_backend_duration_histogram() {
        let mut hist = BackendDurationHistogram::new();

        hist.observe(0.05, true); // <= 0.05 bucket, success
        hist.observe(1.5, true); // <= 2.5 bucket, success
        hist.observe(45.0, false); // <= 60.0 bucket, failure

        assert_eq!(hist.count, 3);
        assert!((hist.sum - 46.55).abs() < 0.001);
        // bucket[2] (0.05): 1 value
        assert_eq!(hist.bucket_counts[2], 1);
        // Check success/failure counts
        assert_eq!(hist.success_count, 2);
        assert_eq!(hist.failure_count, 1);
    }

    #[tokio::test]
    async fn test_record_backend_verification() {
        let metrics = Metrics::new();

        metrics
            .record_backend_verification("lean4", true, 0.1)
            .await;
        metrics
            .record_backend_verification("lean4", true, 0.2)
            .await;
        metrics
            .record_backend_verification("tlaplus", false, 1.5)
            .await;
        metrics
            .record_backend_verification("kani", true, 0.05)
            .await;

        // Check overall counters
        assert_eq!(metrics.verifications_total.load(Ordering::Relaxed), 4);
        assert_eq!(metrics.verifications_success.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.verifications_failed.load(Ordering::Relaxed), 1);

        // Check per-backend histograms
        let backends = metrics.backend_durations.read().await;
        assert!(backends.contains_key("lean4"));
        assert!(backends.contains_key("tlaplus"));
        assert!(backends.contains_key("kani"));

        let lean4 = backends.get("lean4").unwrap();
        assert_eq!(lean4.count, 2);
        assert!((lean4.sum - 0.3).abs() < 0.001);
        assert_eq!(lean4.success_count, 2);
        assert_eq!(lean4.failure_count, 0);

        let tlaplus = backends.get("tlaplus").unwrap();
        assert_eq!(tlaplus.count, 1);
        assert!((tlaplus.sum - 1.5).abs() < 0.001);
        assert_eq!(tlaplus.success_count, 0);
        assert_eq!(tlaplus.failure_count, 1);

        let kani = backends.get("kani").unwrap();
        assert_eq!(kani.success_count, 1);
        assert_eq!(kani.failure_count, 0);
    }

    #[tokio::test]
    async fn test_export_prometheus_includes_backend_duration() {
        let metrics = Metrics::new();

        metrics
            .record_backend_verification("lean4", true, 0.15)
            .await;
        metrics
            .record_backend_verification("tlaplus", true, 0.25)
            .await;

        let output = metrics.export_prometheus(0, 0, 0, 0, 0).await;

        // Check that per-backend duration histogram is exported
        assert!(output.contains("dashprove_backend_verification_duration_seconds"));
        assert!(output
            .contains("dashprove_backend_verification_duration_seconds_bucket{backend=\"lean4\""));
        assert!(output.contains(
            "dashprove_backend_verification_duration_seconds_bucket{backend=\"tlaplus\""
        ));
        assert!(output
            .contains("dashprove_backend_verification_duration_seconds_sum{backend=\"lean4\""));
        assert!(output
            .contains("dashprove_backend_verification_duration_seconds_count{backend=\"lean4\""));
    }

    #[tokio::test]
    async fn test_backend_metrics_sorted_output() {
        let metrics = Metrics::new();

        // Add backends in non-alphabetical order
        metrics
            .record_backend_verification("tlaplus", true, 0.1)
            .await;
        metrics
            .record_backend_verification("alloy", true, 0.2)
            .await;
        metrics
            .record_backend_verification("lean4", true, 0.3)
            .await;

        let output = metrics.export_prometheus(0, 0, 0, 0, 0).await;

        // Find positions of each backend in output - should be alphabetically sorted
        let alloy_pos = output.find("backend=\"alloy\"").unwrap();
        let lean4_pos = output.find("backend=\"lean4\"").unwrap();
        let tlaplus_pos = output.find("backend=\"tlaplus\"").unwrap();

        assert!(alloy_pos < lean4_pos);
        assert!(lean4_pos < tlaplus_pos);
    }

    #[tokio::test]
    async fn test_export_prometheus_includes_backend_success_failure() {
        let metrics = Metrics::new();

        metrics
            .record_backend_verification("lean4", true, 0.1)
            .await;
        metrics
            .record_backend_verification("lean4", true, 0.2)
            .await;
        metrics
            .record_backend_verification("tlaplus", false, 1.5)
            .await;
        metrics
            .record_backend_verification("kani", true, 0.05)
            .await;

        let output = metrics.export_prometheus(0, 0, 0, 0, 0).await;

        // Check that per-backend success/failure counters are exported
        assert!(output.contains("dashprove_backend_verifications_success"));
        assert!(output.contains("dashprove_backend_verifications_failed"));

        // Check specific counts
        assert!(output.contains("dashprove_backend_verifications_success{backend=\"lean4\"} 2"));
        assert!(output.contains("dashprove_backend_verifications_success{backend=\"kani\"} 1"));
        assert!(output.contains("dashprove_backend_verifications_success{backend=\"tlaplus\"} 0"));
        assert!(output.contains("dashprove_backend_verifications_failed{backend=\"lean4\"} 0"));
        assert!(output.contains("dashprove_backend_verifications_failed{backend=\"tlaplus\"} 1"));
        assert!(output.contains("dashprove_backend_verifications_failed{backend=\"kani\"} 0"));
    }

    #[tokio::test]
    async fn test_record_backend_duration_with_success() {
        let metrics = Metrics::new();

        // Use record_backend_duration directly (not record_backend_verification)
        metrics.record_backend_duration("alloy", true, 0.1).await;
        metrics.record_backend_duration("alloy", false, 0.2).await;
        metrics.record_backend_duration("alloy", true, 0.3).await;

        let backends = metrics.backend_durations.read().await;
        let alloy = backends.get("alloy").unwrap();

        assert_eq!(alloy.count, 3);
        assert_eq!(alloy.success_count, 2);
        assert_eq!(alloy.failure_count, 1);
        assert!((alloy.sum - 0.6).abs() < 0.001);
    }

    // ============================================
    // Mutation-killing tests for metrics.rs
    // ============================================

    #[test]
    fn test_normalize_path_root() {
        // Test that "/" is preserved (not normalized to something else)
        assert_eq!(normalize_path("/"), "/");
    }

    #[test]
    fn test_normalize_path_preserves_static_segments() {
        // Static path segments should NOT be replaced with :id
        assert_eq!(normalize_path("/api/v1/users"), "/api/v1/users");
        assert_eq!(
            normalize_path("/some/deeply/nested/path"),
            "/some/deeply/nested/path"
        );
    }

    #[test]
    fn test_looks_like_uuid_exact_format() {
        // Valid UUIDs
        assert!(looks_like_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(looks_like_uuid("00000000-0000-0000-0000-000000000000"));
        assert!(looks_like_uuid("ffffffff-ffff-ffff-ffff-ffffffffffff"));

        // Wrong length
        assert!(!looks_like_uuid("550e8400-e29b-41d4-a716-44665544000")); // 35 chars
        assert!(!looks_like_uuid("550e8400-e29b-41d4-a716-4466554400000")); // 37 chars

        // Wrong hyphen positions
        assert!(!looks_like_uuid("550e8400e-29b-41d4-a716-44665544000")); // hyphen at 9
        assert!(!looks_like_uuid("550e840-0e29b-41d4-a716-446655440000")); // hyphen at 7

        // Non-hex characters
        assert!(!looks_like_uuid("550e8400-e29b-41d4-a716-44665544000g")); // 'g' is not hex
    }

    #[test]
    fn test_looks_like_uuid_hyphen_positions() {
        // Test each hyphen position check individually
        // Position 8
        let mut chars: Vec<char> = "550e8400-e29b-41d4-a716-446655440000".chars().collect();
        chars[8] = 'x';
        let s: String = chars.iter().collect();
        assert!(!looks_like_uuid(&s));

        // Position 13
        let mut chars: Vec<char> = "550e8400-e29b-41d4-a716-446655440000".chars().collect();
        chars[13] = 'x';
        let s: String = chars.iter().collect();
        assert!(!looks_like_uuid(&s));

        // Position 18
        let mut chars: Vec<char> = "550e8400-e29b-41d4-a716-446655440000".chars().collect();
        chars[18] = 'x';
        let s: String = chars.iter().collect();
        assert!(!looks_like_uuid(&s));

        // Position 23
        let mut chars: Vec<char> = "550e8400-e29b-41d4-a716-446655440000".chars().collect();
        chars[23] = 'x';
        let s: String = chars.iter().collect();
        assert!(!looks_like_uuid(&s));
    }

    #[test]
    fn test_looks_like_id_numeric_boundaries() {
        // Empty string should NOT be an ID
        assert!(!looks_like_id(""));

        // Single digit IS an ID
        assert!(looks_like_id("0"));
        assert!(looks_like_id("9"));

        // Exactly 20 digits IS an ID (numeric path)
        assert!(looks_like_id("12345678901234567890"));

        // 21 digits IS also an ID (because it matches hex string check: 8-64 chars)
        assert!(looks_like_id("123456789012345678901"));

        // Non-numeric 3-letter string should not be an ID (too short for hex)
        assert!(!looks_like_id("abc"));

        // But 8 hex chars should be an ID
        assert!(looks_like_id("abc12345"));
    }

    #[test]
    fn test_looks_like_id_hex_boundaries() {
        // Exactly 8 chars hex is an ID
        assert!(looks_like_id("abcd1234"));
        assert!(looks_like_id("ABCD1234"));

        // 7 chars is NOT an ID (too short for hex)
        assert!(!looks_like_id("abcd123"));

        // Exactly 64 chars hex is an ID
        let hex64 = "a".repeat(64);
        assert!(looks_like_id(&hex64));

        // 65 chars hex is NOT an ID (too long)
        let hex65 = "a".repeat(65);
        assert!(!looks_like_id(&hex65));

        // Non-hex chars in long string
        assert!(!looks_like_id("abcdefgh12345678")); // 'g' and 'h' are not hex
    }

    #[test]
    fn test_looks_like_id_hyphenated_hex() {
        // Hex with hyphens should still match
        assert!(looks_like_id("abcd-1234-efab"));
    }

    #[test]
    fn test_duration_histogram_bucket_boundaries() {
        let mut hist = DurationHistogram::new();

        // Observe exactly at bucket boundaries
        hist.observe(0.001); // Exactly at first bucket
        hist.observe(0.005); // Exactly at second bucket
        hist.observe(10.0); // Exactly at last bucket

        // Count should be 3
        assert_eq!(hist.count, 3);

        // Check that values AT boundary are counted (<=)
        // 0.001 should be in bucket[0] (0.001)
        assert_eq!(hist.bucket_counts[0], 1);
        // 0.005 should be in bucket[1] (0.005)
        assert_eq!(hist.bucket_counts[1], 2); // both 0.001 and 0.005 <= 0.005
                                              // 10.0 should be in last bucket
        assert_eq!(hist.bucket_counts[11], 3); // all three <= 10.0
    }

    #[test]
    fn test_duration_histogram_value_above_all_buckets() {
        let mut hist = DurationHistogram::new();

        // Value above all buckets
        hist.observe(100.0);

        assert_eq!(hist.count, 1);
        // No bucket should contain this value
        for &count in &hist.bucket_counts {
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn test_verification_duration_histogram_boundaries() {
        let mut hist = VerificationDurationHistogram::new();

        // Test boundaries specific to verification histogram
        hist.observe(0.01); // First bucket
        hist.observe(300.0); // Last bucket

        assert_eq!(hist.count, 2);
        assert_eq!(hist.bucket_counts[0], 1); // 0.01 <= 0.01
        assert_eq!(hist.bucket_counts[13], 2); // both <= 300.0
    }

    #[test]
    fn test_backend_duration_histogram_boundaries() {
        let mut hist = BackendDurationHistogram::new();

        // Test that observe correctly updates all fields
        hist.observe(0.01, true);
        hist.observe(0.02, false);
        hist.observe(300.0, true);

        assert_eq!(hist.count, 3);
        assert!((hist.sum - 300.03).abs() < 0.001);
        assert_eq!(hist.success_count, 2);
        assert_eq!(hist.failure_count, 1);
    }

    #[test]
    fn test_metrics_default_impl() {
        let metrics = Metrics::default();
        assert!(metrics.uptime_seconds() >= 0.0);
        assert_eq!(metrics.verifications_total.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_record_request_error_threshold() {
        let metrics = Metrics::new();

        // 399 should NOT be counted as error
        metrics.record_request("GET", "/test", 399, 0.001).await;

        // 400 SHOULD be counted as error
        metrics.record_request("GET", "/test", 400, 0.001).await;

        // 500 SHOULD be counted as error
        metrics.record_request("GET", "/test", 500, 0.001).await;

        let errors = metrics.error_counts.read().await;
        // 399 not recorded as error
        assert!(errors
            .get(&("GET".to_string(), "/test".to_string(), 399))
            .is_none());
        // 400 recorded
        assert_eq!(
            errors.get(&("GET".to_string(), "/test".to_string(), 400)),
            Some(&1)
        );
        // 500 recorded
        assert_eq!(
            errors.get(&("GET".to_string(), "/test".to_string(), 500)),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn test_record_request_normalizes_path() {
        let metrics = Metrics::new();

        // Record request with a UUID path
        metrics
            .record_request(
                "GET",
                "/items/550e8400-e29b-41d4-a716-446655440000",
                200,
                0.001,
            )
            .await;

        let counts = metrics.request_counts.read().await;
        // Should be normalized to :id
        assert!(counts.contains_key(&("GET".to_string(), "/items/:id".to_string())));
    }

    #[tokio::test]
    async fn test_record_request_accumulates_duration_sum() {
        let metrics = Metrics::new();

        metrics.record_request("GET", "/health", 200, 0.100).await;
        metrics.record_request("GET", "/health", 200, 0.200).await;
        metrics.record_request("GET", "/health", 200, 0.300).await;

        let sums = metrics.duration_sums.read().await;
        let sum = sums
            .get(&("GET".to_string(), "/health".to_string()))
            .unwrap();
        assert!((sum - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_record_verification_increments_counters() {
        let metrics = Metrics::new();

        metrics.record_verification(true);
        metrics.record_verification(true);
        metrics.record_verification(false);

        // Check each counter is incremented correctly
        assert_eq!(metrics.verifications_total.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.verifications_success.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.verifications_failed.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_record_verification_with_duration_accumulates_sum() {
        let metrics = Metrics::new();

        metrics.record_verification_with_duration(true, 1.0).await;
        metrics.record_verification_with_duration(false, 2.0).await;
        metrics.record_verification_with_duration(true, 0.5).await;

        let sum = *metrics.verification_duration_sum.read().await;
        assert!((sum - 3.5).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_export_prometheus_format_contains_all_sections() {
        let metrics = Metrics::new();

        // Record various metrics to ensure all sections are populated
        metrics.record_request("GET", "/test", 200, 0.1).await;
        metrics.record_request("GET", "/test", 500, 0.2).await;
        metrics.record_verification_with_duration(true, 0.5).await;
        metrics.record_backend_duration("lean4", true, 0.3).await;

        let output = metrics.export_prometheus(10, 5, 100, 90, 10).await;

        // Check all HELP and TYPE lines are present
        assert!(output.contains("# HELP dashprove_uptime_seconds"));
        assert!(output.contains("# TYPE dashprove_uptime_seconds gauge"));
        assert!(output.contains("# HELP dashprove_active_http_requests"));
        assert!(output.contains("# HELP dashprove_active_websocket_sessions"));
        assert!(output.contains("# HELP dashprove_cache_entries"));
        assert!(output.contains("# HELP dashprove_verifications_total"));
        assert!(output.contains("# HELP dashprove_verifications_success"));
        assert!(output.contains("# HELP dashprove_verifications_failed"));
        assert!(output.contains("# HELP dashprove_http_requests_total"));
        assert!(output.contains("# HELP dashprove_http_errors_total"));
        assert!(output.contains("# HELP dashprove_http_request_duration_seconds"));
        assert!(output.contains("# HELP dashprove_verification_duration_seconds"));
        assert!(output.contains("# HELP dashprove_backend_verification_duration_seconds"));
        assert!(output.contains("# HELP dashprove_backend_verifications_success"));
        assert!(output.contains("# HELP dashprove_backend_verifications_failed"));
    }

    #[tokio::test]
    async fn test_export_prometheus_histogram_cumulative() {
        let metrics = Metrics::new();

        // Add 3 requests with different durations
        metrics.record_request("GET", "/test", 200, 0.001).await;
        metrics.record_request("GET", "/test", 200, 0.01).await;
        metrics.record_request("GET", "/test", 200, 0.1).await;

        let output = metrics.export_prometheus(0, 0, 0, 0, 0).await;

        // Cumulative counts should increase as buckets increase
        // Find the bucket lines and verify cumulative property
        assert!(output.contains(
            "dashprove_http_request_duration_seconds_count{method=\"GET\",path=\"/test\"} 3"
        ));
    }

    #[tokio::test]
    async fn test_export_prometheus_inf_bucket() {
        let metrics = Metrics::new();

        // Request with very long duration
        metrics.record_request("GET", "/slow", 200, 1000.0).await;

        let output = metrics.export_prometheus(0, 0, 0, 0, 0).await;

        // +Inf bucket should contain the value
        assert!(output.contains("dashprove_http_request_duration_seconds_bucket{method=\"GET\",path=\"/slow\",le=\"+Inf\"} 1"));
    }

    #[test]
    fn test_normalize_path_multiple_ids_in_path() {
        // Multiple IDs in same path
        let result = normalize_path("/users/12345/items/67890/details");
        assert_eq!(result, "/users/:id/items/:id/details");
    }

    #[test]
    fn test_normalize_path_mixed_ids() {
        // Mix of UUID and numeric ID
        let result = normalize_path("/items/550e8400-e29b-41d4-a716-446655440000/versions/42");
        assert_eq!(result, "/items/:id/versions/:id");
    }

    #[tokio::test]
    async fn test_backend_durations_creates_new_entry() {
        let metrics = Metrics::new();

        // Record for a new backend
        metrics
            .record_backend_duration("new_backend", true, 0.5)
            .await;

        let backends = metrics.backend_durations.read().await;
        assert!(backends.contains_key("new_backend"));
        let backend = backends.get("new_backend").unwrap();
        assert_eq!(backend.count, 1);
    }
}
