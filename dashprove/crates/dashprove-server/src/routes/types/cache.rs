use serde::{Deserialize, Serialize};

// ============ Cache Types ============

/// Response for cache statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct CacheStatsResponse {
    /// Total number of entries in the cache
    pub total_entries: usize,
    /// Number of valid (non-expired) entries
    pub valid_entries: usize,
    /// Number of expired entries pending eviction
    pub expired_entries: usize,
    /// Maximum entries allowed in the cache
    pub max_entries: usize,
    /// Default time-to-live in seconds for cache entries
    pub default_ttl_secs: u64,
}

/// Response for cache operations
#[derive(Debug, Serialize, Deserialize)]
pub struct CacheOperationResponse {
    /// Whether the operation succeeded
    pub success: bool,
    /// Number of entries affected
    pub entries: usize,
    /// Optional message
    pub message: Option<String>,
}
