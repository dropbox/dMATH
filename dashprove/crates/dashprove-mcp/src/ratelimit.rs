//! Rate limiting for MCP server transports
//!
//! Implements a token bucket rate limiter that can be applied to HTTP and WebSocket
//! transports to prevent abuse and ensure fair resource allocation.
//!
//! ## Algorithm
//!
//! Uses a token bucket algorithm where:
//! - Each client (identified by IP address) has a bucket
//! - Tokens are added to the bucket at a fixed rate (requests per second)
//! - Each request consumes one token
//! - If no tokens are available, the request is rejected with 429 Too Many Requests
//! - Buckets have a maximum capacity (burst limit) allowing short bursts of traffic
//!
//! ## Configuration
//!
//! - `requests_per_second`: Rate at which tokens are replenished
//! - `burst_size`: Maximum number of tokens in the bucket (allows bursts)
//! - `cleanup_interval`: How often to clean up old entries
//! - `enabled`: Whether rate limiting is active

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Configuration for rate limiting
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per second per client
    pub requests_per_second: f64,
    /// Maximum burst size (bucket capacity)
    pub burst_size: u32,
    /// How often to clean up stale entries (in seconds)
    pub cleanup_interval_secs: u64,
    /// Whether rate limiting is enabled
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 10.0,
            burst_size: 50,
            cleanup_interval_secs: 60,
            enabled: true,
        }
    }
}

impl RateLimitConfig {
    /// Create a new rate limit configuration
    pub fn new(requests_per_second: f64, burst_size: u32) -> Self {
        Self {
            requests_per_second,
            burst_size,
            ..Default::default()
        }
    }

    /// Create a disabled rate limiter
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

/// Token bucket state for a single client
#[derive(Debug)]
struct TokenBucket {
    /// Current number of tokens
    tokens: f64,
    /// Last time tokens were updated
    last_update: Instant,
}

impl TokenBucket {
    fn new(capacity: u32) -> Self {
        Self {
            tokens: capacity as f64,
            last_update: Instant::now(),
        }
    }

    /// Try to consume a token, returning true if successful
    fn try_consume(&mut self, config: &RateLimitConfig) -> bool {
        self.refill(config);

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on time elapsed
    fn refill(&mut self, config: &RateLimitConfig) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        let new_tokens = elapsed * config.requests_per_second;

        self.tokens = (self.tokens + new_tokens).min(config.burst_size as f64);
        self.last_update = now;
    }

    /// Check if this bucket is stale (no activity for a while)
    fn is_stale(&self, max_age: Duration) -> bool {
        self.last_update.elapsed() > max_age
    }
}

/// Rate limiter using token bucket algorithm
#[derive(Debug)]
pub struct RateLimiter {
    /// Configuration
    config: RwLock<RateLimitConfig>,
    /// Token buckets per client IP
    buckets: RwLock<HashMap<IpAddr, TokenBucket>>,
    /// Statistics
    stats: RwLock<RateLimitStats>,
}

/// Rate limiting statistics
#[derive(Debug, Default, Clone)]
pub struct RateLimitStats {
    /// Total requests allowed
    pub allowed: u64,
    /// Total requests rejected
    pub rejected: u64,
    /// Current number of tracked clients
    pub tracked_clients: usize,
}

impl RateLimitStats {
    /// Calculate rejection rate as percentage
    pub fn rejection_rate(&self) -> f64 {
        let total = self.allowed + self.rejected;
        if total == 0 {
            0.0
        } else {
            (self.rejected as f64 / total as f64) * 100.0
        }
    }
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration
    pub fn new(config: RateLimitConfig) -> Arc<Self> {
        Arc::new(Self {
            config: RwLock::new(config),
            buckets: RwLock::new(HashMap::new()),
            stats: RwLock::new(RateLimitStats::default()),
        })
    }

    /// Create a rate limiter with default configuration
    pub fn with_defaults() -> Arc<Self> {
        Self::new(RateLimitConfig::default())
    }

    /// Check if a request from the given IP should be allowed
    pub async fn check(&self, ip: IpAddr) -> bool {
        let config = self.config.read().await;

        if !config.enabled {
            return true;
        }

        let mut buckets = self.buckets.write().await;
        let bucket = buckets
            .entry(ip)
            .or_insert_with(|| TokenBucket::new(config.burst_size));

        let allowed = bucket.try_consume(&config);

        // Update stats
        let mut stats = self.stats.write().await;
        if allowed {
            stats.allowed += 1;
        } else {
            stats.rejected += 1;
            debug!("Rate limit exceeded for IP: {}", ip);
        }
        stats.tracked_clients = buckets.len();

        allowed
    }

    /// Get current statistics
    pub async fn stats(&self) -> RateLimitStats {
        self.stats.read().await.clone()
    }

    /// Get current configuration
    pub async fn config(&self) -> RateLimitConfig {
        self.config.read().await.clone()
    }

    /// Update configuration
    pub async fn update_config(&self, new_config: RateLimitConfig) {
        let mut config = self.config.write().await;
        *config = new_config;
    }

    /// Clean up stale entries
    pub async fn cleanup(&self) {
        let config = self.config.read().await;
        let max_age = Duration::from_secs(config.cleanup_interval_secs * 2);

        let mut buckets = self.buckets.write().await;
        let before = buckets.len();
        buckets.retain(|_, bucket| !bucket.is_stale(max_age));
        let after = buckets.len();

        if before != after {
            debug!(
                "Rate limiter cleanup: removed {} stale entries",
                before - after
            );
        }

        let mut stats = self.stats.write().await;
        stats.tracked_clients = after;
    }

    /// Reset all buckets and statistics
    pub async fn reset(&self) {
        let mut buckets = self.buckets.write().await;
        buckets.clear();

        let mut stats = self.stats.write().await;
        *stats = RateLimitStats::default();
    }

    /// Get remaining tokens for an IP (for informational headers)
    pub async fn remaining_tokens(&self, ip: IpAddr) -> Option<u32> {
        let buckets = self.buckets.read().await;
        buckets.get(&ip).map(|b| b.tokens as u32)
    }

    /// Start a background cleanup task
    pub fn start_cleanup_task(limiter: Arc<RateLimiter>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                let interval = {
                    let config = limiter.config.read().await;
                    Duration::from_secs(config.cleanup_interval_secs)
                };
                tokio::time::sleep(interval).await;
                limiter.cleanup().await;
            }
        })
    }
}

/// Shared rate limiter type for use in axum state
pub type SharedRateLimiter = Arc<RateLimiter>;

/// Rate limiting middleware for axum
///
/// Extracts client IP from connection info or X-Forwarded-For header
/// and applies rate limiting based on the configured limits.
pub async fn rate_limit_middleware(
    rate_limiter: Option<SharedRateLimiter>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // If no rate limiter is configured, allow all requests
    let Some(limiter) = rate_limiter else {
        return next.run(request).await;
    };

    // Check if rate limiting is enabled
    let config = limiter.config().await;
    if !config.enabled {
        return next.run(request).await;
    }

    // Extract client IP from request
    let ip = extract_client_ip(&request);

    // Check rate limit
    if limiter.check(ip).await {
        // Add rate limit headers
        let mut response = next.run(request).await;
        if let Some(remaining) = limiter.remaining_tokens(ip).await {
            response.headers_mut().insert(
                "X-RateLimit-Remaining",
                remaining.to_string().parse().unwrap(),
            );
        }
        response.headers_mut().insert(
            "X-RateLimit-Limit",
            format!("{}/s", config.requests_per_second).parse().unwrap(),
        );
        response
    } else {
        // Return 429 Too Many Requests
        warn!("Rate limit exceeded for client IP: {}", ip);

        let retry_after = (1.0 / config.requests_per_second).ceil() as u64;

        (
            StatusCode::TOO_MANY_REQUESTS,
            [
                ("Retry-After", retry_after.to_string()),
                (
                    "X-RateLimit-Limit",
                    format!("{}/s", config.requests_per_second),
                ),
                ("X-RateLimit-Remaining", "0".to_string()),
            ],
            Json(serde_json::json!({
                "error": "Too Many Requests",
                "message": format!(
                    "Rate limit exceeded. Maximum {} requests per second.",
                    config.requests_per_second
                ),
                "retry_after_seconds": retry_after
            })),
        )
            .into_response()
    }
}

/// Extract client IP from request
///
/// Checks in order:
/// 1. X-Forwarded-For header (first IP)
/// 2. X-Real-IP header
/// 3. Connected socket address
/// 4. Fallback to 0.0.0.0
fn extract_client_ip(request: &Request<Body>) -> IpAddr {
    // Try X-Forwarded-For header
    if let Some(xff) = request.headers().get("X-Forwarded-For") {
        if let Ok(xff_str) = xff.to_str() {
            if let Some(first_ip) = xff_str.split(',').next() {
                if let Ok(ip) = first_ip.trim().parse() {
                    return ip;
                }
            }
        }
    }

    // Try X-Real-IP header
    if let Some(real_ip) = request.headers().get("X-Real-IP") {
        if let Ok(ip_str) = real_ip.to_str() {
            if let Ok(ip) = ip_str.trim().parse() {
                return ip;
            }
        }
    }

    // Try to get from extensions (ConnectInfo)
    if let Some(connect_info) = request
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
    {
        return connect_info.0.ip();
    }

    // Fallback
    "0.0.0.0".parse().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[tokio::test]
    async fn test_rate_limiter_allows_within_limit() {
        let config = RateLimitConfig::new(10.0, 5);
        let limiter = RateLimiter::new(config);
        let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();

        // Should allow burst_size requests immediately
        for _ in 0..5 {
            assert!(limiter.check(ip).await);
        }

        let stats = limiter.stats().await;
        assert_eq!(stats.allowed, 5);
        assert_eq!(stats.rejected, 0);
    }

    #[tokio::test]
    async fn test_rate_limiter_rejects_over_limit() {
        let config = RateLimitConfig::new(10.0, 3);
        let limiter = RateLimiter::new(config);
        let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();

        // Exhaust the bucket
        for _ in 0..3 {
            assert!(limiter.check(ip).await);
        }

        // Next request should be rejected
        assert!(!limiter.check(ip).await);

        let stats = limiter.stats().await;
        assert_eq!(stats.allowed, 3);
        assert_eq!(stats.rejected, 1);
    }

    #[tokio::test]
    async fn test_rate_limiter_refills_over_time() {
        let config = RateLimitConfig::new(100.0, 3); // High rate for fast refill
        let limiter = RateLimiter::new(config);
        let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();

        // Exhaust the bucket
        for _ in 0..3 {
            assert!(limiter.check(ip).await);
        }

        // Should be rejected
        assert!(!limiter.check(ip).await);

        // Wait for refill
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Should now be allowed
        assert!(limiter.check(ip).await);
    }

    #[tokio::test]
    async fn test_rate_limiter_per_ip_isolation() {
        let config = RateLimitConfig::new(10.0, 2);
        let limiter = RateLimiter::new(config);
        let ip1: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();
        let ip2: IpAddr = Ipv4Addr::new(127, 0, 0, 2).into();

        // Exhaust ip1's bucket
        assert!(limiter.check(ip1).await);
        assert!(limiter.check(ip1).await);
        assert!(!limiter.check(ip1).await);

        // ip2 should still have tokens
        assert!(limiter.check(ip2).await);
        assert!(limiter.check(ip2).await);

        let stats = limiter.stats().await;
        assert_eq!(stats.tracked_clients, 2);
    }

    #[tokio::test]
    async fn test_rate_limiter_disabled() {
        let config = RateLimitConfig::disabled();
        let limiter = RateLimiter::new(config);
        let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();

        // Should always allow when disabled
        for _ in 0..100 {
            assert!(limiter.check(ip).await);
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_cleanup() {
        let mut config = RateLimitConfig::new(10.0, 5);
        config.cleanup_interval_secs = 1;
        let limiter = RateLimiter::new(config);
        let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();

        // Add an entry
        limiter.check(ip).await;
        assert_eq!(limiter.stats().await.tracked_clients, 1);

        // Cleanup shouldn't remove active entries
        limiter.cleanup().await;
        assert_eq!(limiter.stats().await.tracked_clients, 1);
    }

    #[tokio::test]
    async fn test_rate_limiter_reset() {
        let config = RateLimitConfig::new(10.0, 5);
        let limiter = RateLimiter::new(config);
        let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();

        // Add entries and stats
        limiter.check(ip).await;
        limiter.check(ip).await;

        let stats = limiter.stats().await;
        assert_eq!(stats.allowed, 2);
        assert_eq!(stats.tracked_clients, 1);

        // Reset
        limiter.reset().await;

        let stats = limiter.stats().await;
        assert_eq!(stats.allowed, 0);
        assert_eq!(stats.rejected, 0);
        assert_eq!(stats.tracked_clients, 0);
    }

    #[tokio::test]
    async fn test_rate_limiter_config_update() {
        let config = RateLimitConfig::new(10.0, 5);
        let limiter = RateLimiter::new(config);

        assert_eq!(limiter.config().await.requests_per_second, 10.0);

        limiter.update_config(RateLimitConfig::new(20.0, 10)).await;

        assert_eq!(limiter.config().await.requests_per_second, 20.0);
        assert_eq!(limiter.config().await.burst_size, 10);
    }

    #[tokio::test]
    async fn test_rejection_rate_calculation() {
        let stats = RateLimitStats {
            allowed: 80,
            rejected: 20,
            tracked_clients: 1,
        };

        assert!((stats.rejection_rate() - 20.0).abs() < 0.001);

        let empty_stats = RateLimitStats::default();
        assert_eq!(empty_stats.rejection_rate(), 0.0);
    }
}
