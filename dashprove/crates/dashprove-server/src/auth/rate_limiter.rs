//! Rate limiting implementation

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Rate limiter state
#[derive(Debug)]
pub struct RateLimiter {
    /// Request counts per key (or "anonymous" for unauthenticated)
    buckets: RwLock<HashMap<String, RateBucket>>,
    /// Window duration
    window: Duration,
}

#[derive(Debug, Clone)]
struct RateBucket {
    count: u32,
    window_start: Instant,
}

impl RateLimiter {
    /// Create a new rate limiter with 1-minute windows
    pub fn new() -> Self {
        Self {
            buckets: RwLock::new(HashMap::new()),
            window: Duration::from_secs(60),
        }
    }

    /// Check if a request should be rate limited
    /// Returns Ok(current_count) if allowed, Err(retry_after_secs) if rate limited
    pub async fn check(&self, key: &str, limit: u32) -> Result<u32, u64> {
        let now = Instant::now();
        let mut buckets = self.buckets.write().await;

        let bucket = buckets
            .entry(key.to_string())
            .or_insert_with(|| RateBucket {
                count: 0,
                window_start: now,
            });

        // Check if window has expired
        if now.duration_since(bucket.window_start) >= self.window {
            bucket.count = 0;
            bucket.window_start = now;
        }

        // Check if under limit
        if bucket.count >= limit {
            let retry_after =
                self.window.as_secs() - now.duration_since(bucket.window_start).as_secs();
            return Err(retry_after.max(1));
        }

        bucket.count += 1;
        Ok(bucket.count)
    }

    /// Get current count for a key (for monitoring)
    pub async fn current_count(&self, key: &str) -> u32 {
        let buckets = self.buckets.read().await;
        buckets.get(key).map(|b| b.count).unwrap_or(0)
    }

    /// Clean up expired buckets (call periodically)
    pub async fn cleanup(&self) {
        let now = Instant::now();
        let mut buckets = self.buckets.write().await;
        buckets.retain(|_, bucket| now.duration_since(bucket.window_start) < self.window * 2);
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}
