//! Prometheus-compatible metrics for DashProve MCP server
//!
//! Provides lightweight in-memory metrics collection and a `/metrics` endpoint
//! that outputs Prometheus text format for scraping.
//!
//! # Metrics Exposed
//!
//! - `mcp_uptime_seconds` - Gauge of server uptime in seconds
//! - `mcp_requests_total` - Counter of requests by type (jsonrpc, verify, batch, etc.)
//! - `mcp_request_duration_seconds` - Histogram of request durations
//! - `mcp_active_connections` - Gauge of active HTTP and WebSocket connections
//! - `mcp_verification_total` - Counter of verification operations
//! - `mcp_verification_success` - Counter of successful verifications
//! - `mcp_verification_failed` - Counter of failed verifications
//! - `mcp_verification_duration_seconds` - Histogram of verification durations
//! - `mcp_cache_entries` - Gauge of cache entries (total, valid, expired)
//! - `mcp_cache_hits` - Counter of cache hits
//! - `mcp_cache_misses` - Counter of cache misses
//! - `mcp_rate_limit_total` - Counter of rate limit events
//! - `mcp_rate_limit_rejected` - Counter of requests rejected due to rate limiting
//! - `mcp_websocket_connections` - Counter of WebSocket connections opened
//! - `mcp_websocket_messages` - Counter of WebSocket messages by type
//!
//! # Rolling Window Rate Metrics
//!
//! - `mcp_request_rate_1m` - Requests per second over the last 1 minute
//! - `mcp_request_rate_5m` - Requests per second over the last 5 minutes
//! - `mcp_request_rate_15m` - Requests per second over the last 15 minutes
//! - `mcp_error_rate_1m` - Errors per second over the last 1 minute
//! - `mcp_verification_rate_1m` - Verifications per second over the last 1 minute

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Rolling window configuration for rate metrics
#[derive(Debug, Clone, Copy)]
pub struct RollingWindowConfig {
    /// Window duration (e.g., 60 seconds for 1 minute)
    pub window_secs: u64,
    /// Bucket granularity in seconds (e.g., 1 second)
    pub bucket_secs: u64,
}

impl RollingWindowConfig {
    /// Create a 1-minute window with 1-second buckets
    pub fn one_minute() -> Self {
        Self {
            window_secs: 60,
            bucket_secs: 1,
        }
    }

    /// Create a 5-minute window with 5-second buckets
    pub fn five_minutes() -> Self {
        Self {
            window_secs: 300,
            bucket_secs: 5,
        }
    }

    /// Create a 15-minute window with 15-second buckets
    pub fn fifteen_minutes() -> Self {
        Self {
            window_secs: 900,
            bucket_secs: 15,
        }
    }
}

/// A single bucket in the rolling window
#[derive(Debug, Clone)]
struct WindowBucket {
    /// Timestamp when this bucket started
    start_time: Instant,
    /// Count of events in this bucket
    count: u64,
}

/// Rolling window for calculating rates over time periods
///
/// Uses a bucketed approach for memory efficiency. Events are grouped into
/// time buckets, and old buckets are discarded as time moves forward.
#[derive(Clone)]
pub struct RollingWindow {
    /// Configuration for this window
    config: RollingWindowConfig,
    /// Time buckets (newest at back)
    buckets: VecDeque<WindowBucket>,
    /// When the window was created
    start_time: Instant,
}

impl RollingWindow {
    /// Create a new rolling window with the given configuration
    pub fn new(config: RollingWindowConfig) -> Self {
        Self {
            config,
            buckets: VecDeque::new(),
            start_time: Instant::now(),
        }
    }

    /// Record an event at the current time
    pub fn record(&mut self) {
        self.record_n(1);
    }

    /// Record multiple events at the current time
    pub fn record_n(&mut self, count: u64) {
        let now = Instant::now();
        self.prune_old_buckets(now);

        // Find or create the current bucket
        let bucket_start = self.bucket_start_time(now);
        let bucket_secs = self.config.bucket_secs;

        // Check if we can add to the last bucket
        let should_add_to_last = self
            .buckets
            .back()
            .map(|last| Self::times_same_bucket(last.start_time, bucket_start, bucket_secs))
            .unwrap_or(false);

        if should_add_to_last {
            if let Some(last) = self.buckets.back_mut() {
                last.count += count;
            }
        } else {
            // Need a new bucket
            self.buckets.push_back(WindowBucket {
                start_time: bucket_start,
                count,
            });
        }
    }

    /// Get the current rate (events per second) over the window
    pub fn rate(&mut self) -> f64 {
        let now = Instant::now();
        self.prune_old_buckets(now);

        let total: u64 = self.buckets.iter().map(|b| b.count).sum();
        let window_secs = self.config.window_secs as f64;

        total as f64 / window_secs
    }

    /// Get the total count over the window
    pub fn count(&mut self) -> u64 {
        let now = Instant::now();
        self.prune_old_buckets(now);

        self.buckets.iter().map(|b| b.count).sum()
    }

    /// Remove buckets that are older than the window
    fn prune_old_buckets(&mut self, now: Instant) {
        let window_duration = Duration::from_secs(self.config.window_secs);

        while let Some(front) = self.buckets.front() {
            if now.duration_since(front.start_time) > window_duration {
                self.buckets.pop_front();
            } else {
                break;
            }
        }
    }

    /// Calculate the bucket start time for a given instant
    fn bucket_start_time(&self, now: Instant) -> Instant {
        let elapsed = now.duration_since(self.start_time);
        let bucket_secs = self.config.bucket_secs;
        let bucket_index = elapsed.as_secs() / bucket_secs;
        self.start_time + Duration::from_secs(bucket_index * bucket_secs)
    }

    /// Check if two times are in the same bucket (static version for borrow checker)
    fn times_same_bucket(a: Instant, b: Instant, bucket_secs: u64) -> bool {
        // They're in the same bucket if their bucket start times would be equal
        // We can check by comparing the time difference to bucket_secs
        let diff = if a > b {
            a.duration_since(b)
        } else {
            b.duration_since(a)
        };
        diff.as_secs() < bucket_secs
    }
}

/// Thread-safe rolling window
pub struct SharedRollingWindow {
    inner: RwLock<RollingWindow>,
}

impl SharedRollingWindow {
    /// Create a new shared rolling window
    pub fn new(config: RollingWindowConfig) -> Self {
        Self {
            inner: RwLock::new(RollingWindow::new(config)),
        }
    }

    /// Record an event
    pub async fn record(&self) {
        self.inner.write().await.record();
    }

    /// Record multiple events
    pub async fn record_n(&self, count: u64) {
        self.inner.write().await.record_n(count);
    }

    /// Get the current rate
    pub async fn rate(&self) -> f64 {
        self.inner.write().await.rate()
    }

    /// Get the total count
    pub async fn count(&self) -> u64 {
        self.inner.write().await.count()
    }
}

/// Collection of rolling windows for multiple time periods
pub struct RateWindows {
    /// 1-minute window
    pub one_minute: SharedRollingWindow,
    /// 5-minute window
    pub five_minutes: SharedRollingWindow,
    /// 15-minute window
    pub fifteen_minutes: SharedRollingWindow,
}

impl RateWindows {
    /// Create a new set of rate windows
    pub fn new() -> Self {
        Self {
            one_minute: SharedRollingWindow::new(RollingWindowConfig::one_minute()),
            five_minutes: SharedRollingWindow::new(RollingWindowConfig::five_minutes()),
            fifteen_minutes: SharedRollingWindow::new(RollingWindowConfig::fifteen_minutes()),
        }
    }

    /// Record an event in all windows
    pub async fn record(&self) {
        // Use join to record concurrently
        tokio::join!(
            self.one_minute.record(),
            self.five_minutes.record(),
            self.fifteen_minutes.record()
        );
    }

    /// Get rates for all windows
    pub async fn rates(&self) -> RateSnapshot {
        let (one_min, five_min, fifteen_min) = tokio::join!(
            self.one_minute.rate(),
            self.five_minutes.rate(),
            self.fifteen_minutes.rate()
        );
        RateSnapshot {
            rate_1m: one_min,
            rate_5m: five_min,
            rate_15m: fifteen_min,
        }
    }
}

impl Default for RateWindows {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of rates across all time windows
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct RateSnapshot {
    /// Rate over 1 minute (events per second)
    pub rate_1m: f64,
    /// Rate over 5 minutes (events per second)
    pub rate_5m: f64,
    /// Rate over 15 minutes (events per second)
    pub rate_15m: f64,
}

/// Configuration for the metrics collector
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Whether metrics collection is enabled
    pub enabled: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl MetricsConfig {
    /// Create a new enabled metrics configuration
    pub fn enabled() -> Self {
        Self { enabled: true }
    }

    /// Create a disabled metrics configuration
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

/// Histogram buckets for request duration (in seconds)
#[derive(Clone, Default)]
struct DurationHistogram {
    /// Bucket boundaries in seconds
    boundaries: Vec<f64>,
    /// Count of values <= each boundary
    bucket_counts: Vec<u64>,
    /// Total count of all observations
    count: u64,
    /// Sum of all observed values
    sum: f64,
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
            sum: 0.0,
        }
    }

    fn observe(&mut self, value_secs: f64) {
        self.count += 1;
        self.sum += value_secs;
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if value_secs <= boundary {
                self.bucket_counts[i] += 1;
            }
        }
    }
}

/// Histogram for verification durations (longer bucket ranges than HTTP requests)
#[derive(Clone, Default)]
struct VerificationDurationHistogram {
    /// Bucket boundaries in seconds (tuned for verification operations)
    boundaries: Vec<f64>,
    /// Count of values <= each boundary
    bucket_counts: Vec<u64>,
    /// Total count of all observations
    count: u64,
    /// Sum of all observed values
    sum: f64,
}

impl VerificationDurationHistogram {
    fn new() -> Self {
        // Buckets tuned for verification operations (can take longer)
        let boundaries = vec![
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
        ];
        let bucket_counts = vec![0; boundaries.len()];
        Self {
            boundaries,
            bucket_counts,
            count: 0,
            sum: 0.0,
        }
    }

    fn observe(&mut self, value_secs: f64) {
        self.count += 1;
        self.sum += value_secs;
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if value_secs <= boundary {
                self.bucket_counts[i] += 1;
            }
        }
    }
}

/// Thread-safe metrics collector for MCP server
pub struct McpMetrics {
    /// Configuration
    config: MetricsConfig,

    /// Server start time for uptime calculation
    start_time: Instant,

    /// Request counts by type (jsonrpc, verify, batch, websocket, etc.)
    request_counts: RwLock<HashMap<String, u64>>,

    /// Request duration histograms by type
    request_durations: RwLock<HashMap<String, DurationHistogram>>,

    /// HTTP error counts by status code
    error_counts: RwLock<HashMap<u16, u64>>,

    /// Total verification requests
    verifications_total: AtomicU64,

    /// Successful verifications
    verifications_success: AtomicU64,

    /// Failed verifications
    verifications_failed: AtomicU64,

    /// Verification duration histogram
    verification_durations: RwLock<VerificationDurationHistogram>,

    /// Per-backend verification counts
    backend_verifications: RwLock<HashMap<String, BackendMetrics>>,

    /// Cache hit counter
    cache_hits: AtomicU64,

    /// Cache miss counter
    cache_misses: AtomicU64,

    /// Rate limit total events (requests checked)
    rate_limit_total: AtomicU64,

    /// Rate limit rejected count
    rate_limit_rejected: AtomicU64,

    /// WebSocket connections opened
    websocket_connections: AtomicU64,

    /// WebSocket messages received by type
    websocket_messages: RwLock<HashMap<String, u64>>,

    /// Active HTTP connections gauge
    active_http: AtomicU64,

    /// Active WebSocket connections gauge
    active_websocket: AtomicU64,

    /// Rolling window request rate metrics
    request_rates: RateWindows,

    /// Rolling window error rate metrics (1-minute only for alerting)
    error_rate_1m: SharedRollingWindow,

    /// Rolling window verification rate metrics (1-minute only)
    verification_rate_1m: SharedRollingWindow,
}

/// Per-backend verification metrics
#[derive(Clone, Default)]
struct BackendMetrics {
    /// Total verifications for this backend
    total: u64,
    /// Successful verifications for this backend
    success: u64,
    /// Failed verifications for this backend
    failed: u64,
    /// Duration histogram for this backend
    durations: VerificationDurationHistogram,
}

impl BackendMetrics {
    fn new() -> Self {
        Self {
            total: 0,
            success: 0,
            failed: 0,
            durations: VerificationDurationHistogram::new(),
        }
    }

    fn record(&mut self, success: bool, duration_secs: f64) {
        self.total += 1;
        if success {
            self.success += 1;
        } else {
            self.failed += 1;
        }
        self.durations.observe(duration_secs);
    }
}

/// Thread-safe shared metrics
pub type SharedMcpMetrics = Arc<McpMetrics>;

impl McpMetrics {
    /// Create a new metrics collector with the given configuration
    pub fn new(config: MetricsConfig) -> SharedMcpMetrics {
        Arc::new(Self {
            config,
            start_time: Instant::now(),
            request_counts: RwLock::new(HashMap::new()),
            request_durations: RwLock::new(HashMap::new()),
            error_counts: RwLock::new(HashMap::new()),
            verifications_total: AtomicU64::new(0),
            verifications_success: AtomicU64::new(0),
            verifications_failed: AtomicU64::new(0),
            verification_durations: RwLock::new(VerificationDurationHistogram::new()),
            backend_verifications: RwLock::new(HashMap::new()),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            rate_limit_total: AtomicU64::new(0),
            rate_limit_rejected: AtomicU64::new(0),
            websocket_connections: AtomicU64::new(0),
            websocket_messages: RwLock::new(HashMap::new()),
            active_http: AtomicU64::new(0),
            active_websocket: AtomicU64::new(0),
            request_rates: RateWindows::new(),
            error_rate_1m: SharedRollingWindow::new(RollingWindowConfig::one_minute()),
            verification_rate_1m: SharedRollingWindow::new(RollingWindowConfig::one_minute()),
        })
    }

    /// Create an enabled metrics collector
    pub fn enabled() -> SharedMcpMetrics {
        Self::new(MetricsConfig::enabled())
    }

    /// Check if metrics collection is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Record a request by type with duration
    pub async fn record_request(&self, request_type: &str, duration_secs: f64) {
        if !self.config.enabled {
            return;
        }

        let key = request_type.to_string();

        // Increment request count
        {
            let mut counts = self.request_counts.write().await;
            *counts.entry(key.clone()).or_insert(0) += 1;
        }

        // Record duration in histogram
        {
            let mut durations = self.request_durations.write().await;
            durations
                .entry(key)
                .or_insert_with(DurationHistogram::new)
                .observe(duration_secs);
        }

        // Record in rolling windows for rate calculation
        self.request_rates.record().await;
    }

    /// Record an HTTP error by status code
    pub async fn record_error(&self, status_code: u16) {
        if !self.config.enabled {
            return;
        }

        let mut errors = self.error_counts.write().await;
        *errors.entry(status_code).or_insert(0) += 1;

        // Record in error rate window
        self.error_rate_1m.record().await;
    }

    /// Record a verification attempt
    pub fn record_verification(&self, success: bool) {
        if !self.config.enabled {
            return;
        }

        self.verifications_total.fetch_add(1, Ordering::Relaxed);
        if success {
            self.verifications_success.fetch_add(1, Ordering::Relaxed);
        } else {
            self.verifications_failed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a verification attempt with duration
    pub async fn record_verification_with_duration(&self, success: bool, duration_secs: f64) {
        if !self.config.enabled {
            return;
        }

        self.record_verification(success);

        let mut hist = self.verification_durations.write().await;
        hist.observe(duration_secs);

        // Record in verification rate window
        self.verification_rate_1m.record().await;
    }

    /// Record a verification attempt for a specific backend
    pub async fn record_backend_verification(
        &self,
        backend: &str,
        success: bool,
        duration_secs: f64,
    ) {
        if !self.config.enabled {
            return;
        }

        // Record overall metrics
        self.record_verification_with_duration(success, duration_secs)
            .await;

        // Record per-backend metrics
        let mut backends = self.backend_verifications.write().await;
        backends
            .entry(backend.to_string())
            .or_insert_with(BackendMetrics::new)
            .record(success, duration_secs);
    }

    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        if !self.config.enabled {
            return;
        }
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_cache_miss(&self) {
        if !self.config.enabled {
            return;
        }
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a rate limit check
    pub fn record_rate_limit_check(&self, rejected: bool) {
        if !self.config.enabled {
            return;
        }
        self.rate_limit_total.fetch_add(1, Ordering::Relaxed);
        if rejected {
            self.rate_limit_rejected.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a new WebSocket connection
    pub fn record_websocket_connection(&self) {
        if !self.config.enabled {
            return;
        }
        self.websocket_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a WebSocket message by type
    pub async fn record_websocket_message(&self, message_type: &str) {
        if !self.config.enabled {
            return;
        }

        let mut messages = self.websocket_messages.write().await;
        *messages.entry(message_type.to_string()).or_insert(0) += 1;
    }

    /// Increment active HTTP connections
    pub fn inc_active_http(&self) {
        if !self.config.enabled {
            return;
        }
        self.active_http.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active HTTP connections
    pub fn dec_active_http(&self) {
        if !self.config.enabled {
            return;
        }
        self.active_http.fetch_sub(1, Ordering::Relaxed);
    }

    /// Increment active WebSocket connections
    pub fn inc_active_websocket(&self) {
        if !self.config.enabled {
            return;
        }
        self.active_websocket.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active WebSocket connections
    pub fn dec_active_websocket(&self) {
        if !self.config.enabled {
            return;
        }
        self.active_websocket.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get server uptime in seconds
    pub fn uptime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get current rate metrics
    pub async fn rate_snapshot(&self) -> RatesSnapshot {
        let request_rates = self.request_rates.rates().await;
        let error_rate_1m = self.error_rate_1m.rate().await;
        let verification_rate_1m = self.verification_rate_1m.rate().await;

        RatesSnapshot {
            request_rate_1m: request_rates.rate_1m,
            request_rate_5m: request_rates.rate_5m,
            request_rate_15m: request_rates.rate_15m,
            error_rate_1m,
            verification_rate_1m,
        }
    }

    /// Get a snapshot of current metrics values
    pub async fn snapshot(&self) -> MetricsSnapshot {
        let rates = self.rate_snapshot().await;
        MetricsSnapshot {
            uptime_secs: self.uptime_seconds(),
            requests_total: self.request_counts.read().await.values().sum(),
            verifications_total: self.verifications_total.load(Ordering::Relaxed),
            verifications_success: self.verifications_success.load(Ordering::Relaxed),
            verifications_failed: self.verifications_failed.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            rate_limit_checks: self.rate_limit_total.load(Ordering::Relaxed),
            rate_limit_rejected: self.rate_limit_rejected.load(Ordering::Relaxed),
            websocket_connections: self.websocket_connections.load(Ordering::Relaxed),
            active_http: self.active_http.load(Ordering::Relaxed),
            active_websocket: self.active_websocket.load(Ordering::Relaxed),
            rates,
        }
    }

    /// Export metrics in Prometheus text format
    pub async fn export_prometheus(
        &self,
        cache_total: usize,
        cache_valid: usize,
        cache_expired: usize,
    ) -> String {
        let mut output = String::with_capacity(8192);

        // Uptime gauge
        output.push_str("# HELP mcp_uptime_seconds Server uptime in seconds\n");
        output.push_str("# TYPE mcp_uptime_seconds gauge\n");
        output.push_str(&format!(
            "mcp_uptime_seconds {:.3}\n",
            self.uptime_seconds()
        ));
        output.push('\n');

        // Active connections gauges
        output.push_str("# HELP mcp_active_connections Number of active connections by type\n");
        output.push_str("# TYPE mcp_active_connections gauge\n");
        output.push_str(&format!(
            "mcp_active_connections{{type=\"http\"}} {}\n",
            self.active_http.load(Ordering::Relaxed)
        ));
        output.push_str(&format!(
            "mcp_active_connections{{type=\"websocket\"}} {}\n",
            self.active_websocket.load(Ordering::Relaxed)
        ));
        output.push('\n');

        // Request counters by type
        output.push_str("# HELP mcp_requests_total Total requests by type\n");
        output.push_str("# TYPE mcp_requests_total counter\n");
        {
            let counts = self.request_counts.read().await;
            let mut sorted_keys: Vec<_> = counts.keys().collect();
            sorted_keys.sort();
            for key in sorted_keys {
                if let Some(count) = counts.get(key) {
                    output.push_str(&format!(
                        "mcp_requests_total{{type=\"{}\"}} {}\n",
                        key, count
                    ));
                }
            }
        }
        output.push('\n');

        // Request duration histogram
        output.push_str("# HELP mcp_request_duration_seconds Request duration histogram by type\n");
        output.push_str("# TYPE mcp_request_duration_seconds histogram\n");
        {
            let durations = self.request_durations.read().await;
            let mut sorted_keys: Vec<_> = durations.keys().collect();
            sorted_keys.sort();
            for key in sorted_keys {
                if let Some(hist) = durations.get(key) {
                    // Emit buckets
                    for (i, &boundary) in hist.boundaries.iter().enumerate() {
                        output.push_str(&format!(
                            "mcp_request_duration_seconds_bucket{{type=\"{}\",le=\"{}\"}} {}\n",
                            key, boundary, hist.bucket_counts[i]
                        ));
                    }
                    // +Inf bucket
                    output.push_str(&format!(
                        "mcp_request_duration_seconds_bucket{{type=\"{}\",le=\"+Inf\"}} {}\n",
                        key, hist.count
                    ));
                    // Sum and count
                    output.push_str(&format!(
                        "mcp_request_duration_seconds_sum{{type=\"{}\"}} {:.6}\n",
                        key, hist.sum
                    ));
                    output.push_str(&format!(
                        "mcp_request_duration_seconds_count{{type=\"{}\"}} {}\n",
                        key, hist.count
                    ));
                }
            }
        }
        output.push('\n');

        // HTTP errors
        {
            let errors = self.error_counts.read().await;
            if !errors.is_empty() {
                output.push_str("# HELP mcp_http_errors_total HTTP errors by status code\n");
                output.push_str("# TYPE mcp_http_errors_total counter\n");
                let mut sorted_codes: Vec<_> = errors.keys().collect();
                sorted_codes.sort();
                for code in sorted_codes {
                    if let Some(count) = errors.get(code) {
                        output.push_str(&format!(
                            "mcp_http_errors_total{{status=\"{}\"}} {}\n",
                            code, count
                        ));
                    }
                }
                output.push('\n');
            }
        }

        // Verification counters
        output.push_str("# HELP mcp_verifications_total Total verification requests\n");
        output.push_str("# TYPE mcp_verifications_total counter\n");
        output.push_str(&format!(
            "mcp_verifications_total {}\n",
            self.verifications_total.load(Ordering::Relaxed)
        ));
        output.push('\n');

        output.push_str("# HELP mcp_verifications_success Successful verifications\n");
        output.push_str("# TYPE mcp_verifications_success counter\n");
        output.push_str(&format!(
            "mcp_verifications_success {}\n",
            self.verifications_success.load(Ordering::Relaxed)
        ));
        output.push('\n');

        output.push_str("# HELP mcp_verifications_failed Failed verifications\n");
        output.push_str("# TYPE mcp_verifications_failed counter\n");
        output.push_str(&format!(
            "mcp_verifications_failed {}\n",
            self.verifications_failed.load(Ordering::Relaxed)
        ));
        output.push('\n');

        // Verification duration histogram
        output
            .push_str("# HELP mcp_verification_duration_seconds Verification duration histogram\n");
        output.push_str("# TYPE mcp_verification_duration_seconds histogram\n");
        {
            let hist = self.verification_durations.read().await;
            for (i, &boundary) in hist.boundaries.iter().enumerate() {
                output.push_str(&format!(
                    "mcp_verification_duration_seconds_bucket{{le=\"{}\"}} {}\n",
                    boundary, hist.bucket_counts[i]
                ));
            }
            output.push_str(&format!(
                "mcp_verification_duration_seconds_bucket{{le=\"+Inf\"}} {}\n",
                hist.count
            ));
            output.push_str(&format!(
                "mcp_verification_duration_seconds_sum {:.6}\n",
                hist.sum
            ));
            output.push_str(&format!(
                "mcp_verification_duration_seconds_count {}\n",
                hist.count
            ));
        }
        output.push('\n');

        // Per-backend verification metrics
        {
            let backends = self.backend_verifications.read().await;
            if !backends.is_empty() {
                output
                    .push_str("# HELP mcp_backend_verifications_total Verifications by backend\n");
                output.push_str("# TYPE mcp_backend_verifications_total counter\n");
                let mut sorted_keys: Vec<_> = backends.keys().collect();
                sorted_keys.sort();
                for key in &sorted_keys {
                    if let Some(metrics) = backends.get(*key) {
                        output.push_str(&format!(
                            "mcp_backend_verifications_total{{backend=\"{}\",result=\"success\"}} {}\n",
                            key, metrics.success
                        ));
                        output.push_str(&format!(
                            "mcp_backend_verifications_total{{backend=\"{}\",result=\"failed\"}} {}\n",
                            key, metrics.failed
                        ));
                    }
                }
                output.push('\n');

                // Per-backend duration histogram
                output.push_str("# HELP mcp_backend_verification_duration_seconds Verification duration by backend\n");
                output.push_str("# TYPE mcp_backend_verification_duration_seconds histogram\n");
                for key in &sorted_keys {
                    if let Some(metrics) = backends.get(*key) {
                        let hist = &metrics.durations;
                        for (i, &boundary) in hist.boundaries.iter().enumerate() {
                            output.push_str(&format!(
                                "mcp_backend_verification_duration_seconds_bucket{{backend=\"{}\",le=\"{}\"}} {}\n",
                                key, boundary, hist.bucket_counts[i]
                            ));
                        }
                        output.push_str(&format!(
                            "mcp_backend_verification_duration_seconds_bucket{{backend=\"{}\",le=\"+Inf\"}} {}\n",
                            key, hist.count
                        ));
                        output.push_str(&format!(
                            "mcp_backend_verification_duration_seconds_sum{{backend=\"{}\"}} {:.6}\n",
                            key, hist.sum
                        ));
                        output.push_str(&format!(
                            "mcp_backend_verification_duration_seconds_count{{backend=\"{}\"}} {}\n",
                            key, hist.count
                        ));
                    }
                }
                output.push('\n');
            }
        }

        // Cache gauges
        output.push_str("# HELP mcp_cache_entries Number of entries in verification cache\n");
        output.push_str("# TYPE mcp_cache_entries gauge\n");
        output.push_str(&format!(
            "mcp_cache_entries{{state=\"total\"}} {}\n",
            cache_total
        ));
        output.push_str(&format!(
            "mcp_cache_entries{{state=\"valid\"}} {}\n",
            cache_valid
        ));
        output.push_str(&format!(
            "mcp_cache_entries{{state=\"expired\"}} {}\n",
            cache_expired
        ));
        output.push('\n');

        // Cache hit/miss counters
        output.push_str("# HELP mcp_cache_hits Cache hit count\n");
        output.push_str("# TYPE mcp_cache_hits counter\n");
        output.push_str(&format!(
            "mcp_cache_hits {}\n",
            self.cache_hits.load(Ordering::Relaxed)
        ));
        output.push('\n');

        output.push_str("# HELP mcp_cache_misses Cache miss count\n");
        output.push_str("# TYPE mcp_cache_misses counter\n");
        output.push_str(&format!(
            "mcp_cache_misses {}\n",
            self.cache_misses.load(Ordering::Relaxed)
        ));
        output.push('\n');

        // Rate limiting counters
        output.push_str("# HELP mcp_rate_limit_checks Total rate limit checks\n");
        output.push_str("# TYPE mcp_rate_limit_checks counter\n");
        output.push_str(&format!(
            "mcp_rate_limit_checks {}\n",
            self.rate_limit_total.load(Ordering::Relaxed)
        ));
        output.push('\n');

        output.push_str("# HELP mcp_rate_limit_rejected Requests rejected by rate limiting\n");
        output.push_str("# TYPE mcp_rate_limit_rejected counter\n");
        output.push_str(&format!(
            "mcp_rate_limit_rejected {}\n",
            self.rate_limit_rejected.load(Ordering::Relaxed)
        ));
        output.push('\n');

        // WebSocket counters
        output.push_str(
            "# HELP mcp_websocket_connections_total Total WebSocket connections opened\n",
        );
        output.push_str("# TYPE mcp_websocket_connections_total counter\n");
        output.push_str(&format!(
            "mcp_websocket_connections_total {}\n",
            self.websocket_connections.load(Ordering::Relaxed)
        ));
        output.push('\n');

        // WebSocket messages by type
        {
            let messages = self.websocket_messages.read().await;
            if !messages.is_empty() {
                output.push_str("# HELP mcp_websocket_messages_total WebSocket messages by type\n");
                output.push_str("# TYPE mcp_websocket_messages_total counter\n");
                let mut sorted_keys: Vec<_> = messages.keys().collect();
                sorted_keys.sort();
                for key in sorted_keys {
                    if let Some(count) = messages.get(key) {
                        output.push_str(&format!(
                            "mcp_websocket_messages_total{{type=\"{}\"}} {}\n",
                            key, count
                        ));
                    }
                }
                output.push('\n');
            }
        }

        // Rolling window rate metrics
        let rates = self.rate_snapshot().await;

        output.push_str(
            "# HELP mcp_request_rate Request rate (requests per second) over time window\n",
        );
        output.push_str("# TYPE mcp_request_rate gauge\n");
        output.push_str(&format!(
            "mcp_request_rate{{window=\"1m\"}} {:.6}\n",
            rates.request_rate_1m
        ));
        output.push_str(&format!(
            "mcp_request_rate{{window=\"5m\"}} {:.6}\n",
            rates.request_rate_5m
        ));
        output.push_str(&format!(
            "mcp_request_rate{{window=\"15m\"}} {:.6}\n",
            rates.request_rate_15m
        ));
        output.push('\n');

        output.push_str("# HELP mcp_error_rate Error rate (errors per second) over 1 minute\n");
        output.push_str("# TYPE mcp_error_rate gauge\n");
        output.push_str(&format!(
            "mcp_error_rate{{window=\"1m\"}} {:.6}\n",
            rates.error_rate_1m
        ));
        output.push('\n');

        output.push_str(
            "# HELP mcp_verification_rate Verification rate (verifications per second) over 1 minute\n",
        );
        output.push_str("# TYPE mcp_verification_rate gauge\n");
        output.push_str(&format!(
            "mcp_verification_rate{{window=\"1m\"}} {:.6}\n",
            rates.verification_rate_1m
        ));
        output.push('\n');

        output
    }
}

/// Snapshot of rate metrics from rolling windows
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct RatesSnapshot {
    /// Request rate over 1 minute (requests per second)
    pub request_rate_1m: f64,
    /// Request rate over 5 minutes (requests per second)
    pub request_rate_5m: f64,
    /// Request rate over 15 minutes (requests per second)
    pub request_rate_15m: f64,
    /// Error rate over 1 minute (errors per second)
    pub error_rate_1m: f64,
    /// Verification rate over 1 minute (verifications per second)
    pub verification_rate_1m: f64,
}

impl Default for RatesSnapshot {
    fn default() -> Self {
        Self {
            request_rate_1m: 0.0,
            request_rate_5m: 0.0,
            request_rate_15m: 0.0,
            error_rate_1m: 0.0,
            verification_rate_1m: 0.0,
        }
    }
}

/// A snapshot of metrics values (for JSON export)
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    /// Server uptime in seconds
    pub uptime_secs: f64,
    /// Total requests processed
    pub requests_total: u64,
    /// Total verification attempts
    pub verifications_total: u64,
    /// Successful verifications
    pub verifications_success: u64,
    /// Failed verifications
    pub verifications_failed: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Rate limit checks
    pub rate_limit_checks: u64,
    /// Rate limit rejections
    pub rate_limit_rejected: u64,
    /// Total WebSocket connections opened
    pub websocket_connections: u64,
    /// Active HTTP connections
    pub active_http: u64,
    /// Active WebSocket connections
    pub active_websocket: u64,
    /// Rolling window rate metrics
    pub rates: RatesSnapshot,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_creation() {
        let metrics = McpMetrics::enabled();
        assert!(metrics.is_enabled());

        let disabled = McpMetrics::new(MetricsConfig::disabled());
        assert!(!disabled.is_enabled());
    }

    #[tokio::test]
    async fn test_request_recording() {
        let metrics = McpMetrics::enabled();

        metrics.record_request("jsonrpc", 0.05).await;
        metrics.record_request("jsonrpc", 0.10).await;
        metrics.record_request("verify", 1.5).await;

        let snapshot = metrics.snapshot().await;
        assert_eq!(snapshot.requests_total, 3);
    }

    #[tokio::test]
    async fn test_verification_recording() {
        let metrics = McpMetrics::enabled();

        metrics.record_verification(true);
        metrics.record_verification(true);
        metrics.record_verification(false);

        let snapshot = metrics.snapshot().await;
        assert_eq!(snapshot.verifications_total, 3);
        assert_eq!(snapshot.verifications_success, 2);
        assert_eq!(snapshot.verifications_failed, 1);
    }

    #[tokio::test]
    async fn test_verification_with_duration() {
        let metrics = McpMetrics::enabled();

        metrics.record_verification_with_duration(true, 0.5).await;
        metrics.record_verification_with_duration(false, 1.5).await;

        let snapshot = metrics.snapshot().await;
        assert_eq!(snapshot.verifications_total, 2);
        assert_eq!(snapshot.verifications_success, 1);
        assert_eq!(snapshot.verifications_failed, 1);
    }

    #[tokio::test]
    async fn test_backend_verification() {
        let metrics = McpMetrics::enabled();

        metrics
            .record_backend_verification("lean4", true, 0.5)
            .await;
        metrics
            .record_backend_verification("lean4", false, 1.0)
            .await;
        metrics
            .record_backend_verification("tlaplus", true, 2.0)
            .await;

        let snapshot = metrics.snapshot().await;
        assert_eq!(snapshot.verifications_total, 3);
        assert_eq!(snapshot.verifications_success, 2);
        assert_eq!(snapshot.verifications_failed, 1);
    }

    #[tokio::test]
    async fn test_cache_metrics() {
        let metrics = McpMetrics::enabled();

        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        let snapshot = metrics.snapshot().await;
        assert_eq!(snapshot.cache_hits, 2);
        assert_eq!(snapshot.cache_misses, 1);
    }

    #[tokio::test]
    async fn test_rate_limit_metrics() {
        let metrics = McpMetrics::enabled();

        metrics.record_rate_limit_check(false);
        metrics.record_rate_limit_check(false);
        metrics.record_rate_limit_check(true);

        let snapshot = metrics.snapshot().await;
        assert_eq!(snapshot.rate_limit_checks, 3);
        assert_eq!(snapshot.rate_limit_rejected, 1);
    }

    #[tokio::test]
    async fn test_websocket_metrics() {
        let metrics = McpMetrics::enabled();

        metrics.record_websocket_connection();
        metrics.record_websocket_connection();
        metrics.record_websocket_message("jsonrpc").await;
        metrics.record_websocket_message("subscribe").await;
        metrics.record_websocket_message("jsonrpc").await;

        let snapshot = metrics.snapshot().await;
        assert_eq!(snapshot.websocket_connections, 2);
    }

    #[tokio::test]
    async fn test_active_connection_gauges() {
        let metrics = McpMetrics::enabled();

        metrics.inc_active_http();
        metrics.inc_active_http();
        metrics.inc_active_websocket();

        let snapshot = metrics.snapshot().await;
        assert_eq!(snapshot.active_http, 2);
        assert_eq!(snapshot.active_websocket, 1);

        metrics.dec_active_http();
        metrics.dec_active_websocket();

        let snapshot = metrics.snapshot().await;
        assert_eq!(snapshot.active_http, 1);
        assert_eq!(snapshot.active_websocket, 0);
    }

    #[tokio::test]
    async fn test_uptime() {
        let metrics = McpMetrics::enabled();

        // Just verify uptime increases
        let uptime1 = metrics.uptime_seconds();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        let uptime2 = metrics.uptime_seconds();

        assert!(uptime2 > uptime1);
    }

    #[tokio::test]
    async fn test_prometheus_export() {
        let metrics = McpMetrics::enabled();

        metrics.record_request("jsonrpc", 0.05).await;
        metrics.record_verification(true);
        metrics.record_cache_hit();

        let output = metrics.export_prometheus(10, 8, 2).await;

        // Verify key metrics are present
        assert!(output.contains("mcp_uptime_seconds"));
        assert!(output.contains("mcp_requests_total"));
        assert!(output.contains("mcp_verifications_total"));
        assert!(output.contains("mcp_verifications_success"));
        assert!(output.contains("mcp_cache_entries"));
        assert!(output.contains("mcp_cache_hits"));

        // Verify format is correct (TYPE and HELP lines)
        assert!(output.contains("# HELP"));
        assert!(output.contains("# TYPE"));
        assert!(output.contains("counter"));
        assert!(output.contains("gauge"));
        assert!(output.contains("histogram"));
    }

    #[tokio::test]
    async fn test_disabled_metrics_noop() {
        let metrics = McpMetrics::new(MetricsConfig::disabled());

        // These should be no-ops when disabled
        metrics.record_request("jsonrpc", 0.05).await;
        metrics.record_verification(true);
        metrics.record_cache_hit();
        metrics.record_rate_limit_check(true);
        metrics.inc_active_http();

        let snapshot = metrics.snapshot().await;
        assert_eq!(snapshot.requests_total, 0);
        assert_eq!(snapshot.verifications_total, 0);
        assert_eq!(snapshot.cache_hits, 0);
        assert_eq!(snapshot.rate_limit_checks, 0);
        assert_eq!(snapshot.active_http, 0);
    }

    #[tokio::test]
    async fn test_http_errors() {
        let metrics = McpMetrics::enabled();

        metrics.record_error(400).await;
        metrics.record_error(400).await;
        metrics.record_error(500).await;

        let output = metrics.export_prometheus(0, 0, 0).await;
        assert!(output.contains("mcp_http_errors_total{status=\"400\"} 2"));
        assert!(output.contains("mcp_http_errors_total{status=\"500\"} 1"));
    }

    #[tokio::test]
    async fn test_duration_histogram_buckets() {
        let metrics = McpMetrics::enabled();

        // Record some requests with different durations
        metrics.record_request("test", 0.0005).await; // < 0.001
        metrics.record_request("test", 0.003).await; // < 0.005
        metrics.record_request("test", 0.05).await; // < 0.1
        metrics.record_request("test", 5.0).await; // < 10.0

        let output = metrics.export_prometheus(0, 0, 0).await;

        // Check histogram buckets are present
        assert!(output.contains("mcp_request_duration_seconds_bucket"));
        assert!(output.contains("le=\"0.001\""));
        assert!(output.contains("le=\"+Inf\""));
        assert!(output.contains("mcp_request_duration_seconds_sum"));
        assert!(output.contains("mcp_request_duration_seconds_count"));
    }

    // =========================================================================
    // Rolling window rate metrics tests
    // =========================================================================

    #[test]
    fn test_rolling_window_config() {
        let config = RollingWindowConfig::one_minute();
        assert_eq!(config.window_secs, 60);
        assert_eq!(config.bucket_secs, 1);

        let config = RollingWindowConfig::five_minutes();
        assert_eq!(config.window_secs, 300);
        assert_eq!(config.bucket_secs, 5);

        let config = RollingWindowConfig::fifteen_minutes();
        assert_eq!(config.window_secs, 900);
        assert_eq!(config.bucket_secs, 15);
    }

    #[test]
    fn test_rolling_window_record_and_count() {
        let mut window = RollingWindow::new(RollingWindowConfig::one_minute());

        // Record some events
        window.record();
        window.record();
        window.record_n(5);

        // Should count them all
        assert_eq!(window.count(), 7);
    }

    #[test]
    fn test_rolling_window_rate() {
        let mut window = RollingWindow::new(RollingWindowConfig {
            window_secs: 60,
            bucket_secs: 1,
        });

        // Record 120 events (should be 2/sec over 60s window)
        window.record_n(120);

        let rate = window.rate();
        assert!(
            (rate - 2.0).abs() < 0.001,
            "Expected rate ~2.0, got {}",
            rate
        );
    }

    #[test]
    fn test_rolling_window_empty() {
        let mut window = RollingWindow::new(RollingWindowConfig::one_minute());

        // Empty window should have 0 rate
        assert_eq!(window.count(), 0);
        assert_eq!(window.rate(), 0.0);
    }

    #[tokio::test]
    async fn test_shared_rolling_window() {
        let window = SharedRollingWindow::new(RollingWindowConfig::one_minute());

        window.record().await;
        window.record().await;
        window.record_n(3).await;

        assert_eq!(window.count().await, 5);
    }

    #[tokio::test]
    async fn test_rate_windows() {
        let windows = RateWindows::new();

        // Record some events
        windows.record().await;
        windows.record().await;
        windows.record().await;

        // Get rates
        let rates = windows.rates().await;

        // With 3 events in a 60-second window, rate should be 3/60 = 0.05/sec
        assert!(
            rates.rate_1m > 0.0,
            "Expected positive rate, got {}",
            rates.rate_1m
        );
    }

    #[tokio::test]
    async fn test_rate_snapshot_default() {
        let snapshot = RatesSnapshot::default();
        assert_eq!(snapshot.request_rate_1m, 0.0);
        assert_eq!(snapshot.request_rate_5m, 0.0);
        assert_eq!(snapshot.request_rate_15m, 0.0);
        assert_eq!(snapshot.error_rate_1m, 0.0);
        assert_eq!(snapshot.verification_rate_1m, 0.0);
    }

    #[tokio::test]
    async fn test_metrics_rate_snapshot() {
        let metrics = McpMetrics::enabled();

        // Record some requests
        metrics.record_request("test", 0.01).await;
        metrics.record_request("test", 0.01).await;
        metrics.record_request("test", 0.01).await;

        // Record some errors
        metrics.record_error(500).await;

        // Record some verifications
        metrics.record_verification_with_duration(true, 0.5).await;
        metrics.record_verification_with_duration(true, 0.5).await;

        // Get rate snapshot
        let rates = metrics.rate_snapshot().await;

        // Verify rates are positive (exact values depend on timing)
        assert!(
            rates.request_rate_1m > 0.0,
            "Expected positive request rate"
        );
        assert!(rates.error_rate_1m > 0.0, "Expected positive error rate");
        assert!(
            rates.verification_rate_1m > 0.0,
            "Expected positive verification rate"
        );
    }

    #[tokio::test]
    async fn test_prometheus_export_includes_rates() {
        let metrics = McpMetrics::enabled();

        // Record some data
        metrics.record_request("test", 0.01).await;
        metrics.record_error(500).await;
        metrics.record_verification_with_duration(true, 0.5).await;

        let output = metrics.export_prometheus(0, 0, 0).await;

        // Verify rate metrics are present in output
        assert!(output.contains("mcp_request_rate"));
        assert!(output.contains("window=\"1m\""));
        assert!(output.contains("window=\"5m\""));
        assert!(output.contains("window=\"15m\""));
        assert!(output.contains("mcp_error_rate"));
        assert!(output.contains("mcp_verification_rate"));
    }

    #[tokio::test]
    async fn test_metrics_snapshot_includes_rates() {
        let metrics = McpMetrics::enabled();

        metrics.record_request("test", 0.01).await;

        let snapshot = metrics.snapshot().await;

        // Verify rates field is present and has expected structure
        assert!(snapshot.rates.request_rate_1m >= 0.0);
        assert!(snapshot.rates.request_rate_5m >= 0.0);
        assert!(snapshot.rates.request_rate_15m >= 0.0);
        assert!(snapshot.rates.error_rate_1m >= 0.0);
        assert!(snapshot.rates.verification_rate_1m >= 0.0);
    }

    #[test]
    fn test_rate_snapshot_serialization() {
        let snapshot = RatesSnapshot {
            request_rate_1m: 1.5,
            request_rate_5m: 1.2,
            request_rate_15m: 1.0,
            error_rate_1m: 0.1,
            verification_rate_1m: 0.5,
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.contains("request_rate_1m"));
        assert!(json.contains("1.5"));

        let parsed: RatesSnapshot = serde_json::from_str(&json).unwrap();
        assert!((parsed.request_rate_1m - 1.5).abs() < 0.001);
    }
}
