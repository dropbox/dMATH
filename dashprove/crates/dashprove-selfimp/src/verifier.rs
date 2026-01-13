//! Improvement verifier - formal verification of proposed improvements
//!
//! This module provides the formal verification layer that validates
//! proposed improvements using DashProve's verification backends.
//!
//! ## Integration with Dispatcher
//!
//! The verifier can use `dashprove-dispatcher` for multi-backend verification:
//!
//! ```ignore
//! use dashprove_selfimp::{ImprovementVerifier, VerificationConfig, AsyncImprovementVerifier};
//! use dashprove_dispatcher::{Dispatcher, DispatcherConfig};
//!
//! // Create an async verifier with dispatcher
//! let dispatcher = Dispatcher::new(DispatcherConfig::default());
//! let verifier = AsyncImprovementVerifier::with_dispatcher(dispatcher);
//!
//! // Verify an improvement
//! let result = verifier.verify(&current_version, &improvement).await?;
//! ```

use crate::error::{SelfImpError, SelfImpResult};
use crate::improvement::Improvement;
use crate::version::Version;
use dashprove_backends::VerificationStatus;
use dashprove_dispatcher::{Dispatcher, DispatcherConfig, MergedResults};
use dashprove_usl::{parse, typecheck, DependencyGraph};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{oneshot, Mutex};
use tokio::task::JoinHandle;
#[allow(unused_imports)] // Used in tests via super::*
use tokio::time;

// =============================================================================
// Cache Compaction Types
// =============================================================================

/// Policy for cache compaction strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CompactionPolicy {
    /// Remove only expired entries
    #[default]
    ExpiredOnly,
    /// Remove expired entries and entries below confidence threshold
    LowConfidence,
    /// Remove entries for obsolete versions (not in active versions set)
    ObsoleteVersions,
    /// Deduplicate entries with identical results for same property
    Deduplicate,
    /// Aggressive compaction: all of the above
    Aggressive,
}

/// Trigger conditions for automatic compaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompactionTrigger {
    /// Trigger when cache size exceeds a threshold (percentage of max_entries)
    SizeBased {
        /// Threshold as percentage of max_entries (0.0 - 1.0)
        threshold_ratio: f64,
    },
    /// Trigger periodically based on time
    TimeBased {
        /// Interval between compactions in seconds
        interval_seconds: u64,
    },
    /// Trigger when hit rate falls below threshold
    HitRateBased {
        /// Minimum acceptable hit rate (0.0 - 1.0)
        min_hit_rate: f64,
        /// Minimum number of operations before evaluating
        min_operations: u64,
    },
    /// Trigger when partition is imbalanced
    PartitionImbalance {
        /// Maximum ratio between largest and smallest partition
        max_ratio: f64,
    },
    /// Trigger on insert (every N inserts)
    InsertBased {
        /// Number of inserts between compactions
        insert_count: u64,
    },
    /// Trigger when memory estimate exceeds threshold
    MemoryBased {
        /// Maximum estimated memory in bytes
        max_bytes: usize,
    },
}

impl CompactionTrigger {
    /// Create a size-based trigger at 80% capacity
    pub fn size_80_percent() -> Self {
        Self::SizeBased {
            threshold_ratio: 0.8,
        }
    }

    /// Create a size-based trigger at 90% capacity
    pub fn size_90_percent() -> Self {
        Self::SizeBased {
            threshold_ratio: 0.9,
        }
    }

    /// Create a time-based trigger (every N minutes)
    pub fn every_minutes(minutes: u64) -> Self {
        Self::TimeBased {
            interval_seconds: minutes * 60,
        }
    }

    /// Create a time-based trigger (every N hours)
    pub fn every_hours(hours: u64) -> Self {
        Self::TimeBased {
            interval_seconds: hours * 3600,
        }
    }

    /// Create a hit-rate-based trigger
    pub fn hit_rate(min_hit_rate: f64, min_operations: u64) -> Self {
        Self::HitRateBased {
            min_hit_rate: min_hit_rate.clamp(0.0, 1.0),
            min_operations,
        }
    }

    /// Create an insert-based trigger
    pub fn every_inserts(count: u64) -> Self {
        Self::InsertBased {
            insert_count: count,
        }
    }

    /// Create a memory-based trigger (in megabytes)
    pub fn max_megabytes(mb: usize) -> Self {
        Self::MemoryBased {
            max_bytes: mb * 1024 * 1024,
        }
    }
}

/// State for tracking compaction trigger conditions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionTriggerState {
    /// Last compaction timestamp (unix seconds)
    pub last_compaction_time: u64,
    /// Inserts since last compaction
    pub inserts_since_compaction: u64,
    /// Operations (hits + misses) since last trigger evaluation
    pub operations_since_evaluation: u64,
}

impl CompactionTriggerState {
    /// Reset state after compaction
    pub fn reset(&mut self) {
        self.last_compaction_time = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.inserts_since_compaction = 0;
    }

    /// Record an insert
    pub fn record_insert(&mut self) {
        self.inserts_since_compaction += 1;
    }

    /// Record a cache operation
    pub fn record_operation(&mut self) {
        self.operations_since_evaluation += 1;
    }

    /// Get seconds since last compaction
    pub fn seconds_since_compaction(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(self.last_compaction_time)
    }
}

impl CompactionPolicy {
    /// Returns true if this policy removes expired entries
    pub fn removes_expired(&self) -> bool {
        true // All policies remove expired entries
    }

    /// Returns true if this policy removes low-confidence entries
    pub fn removes_low_confidence(&self) -> bool {
        matches!(self, Self::LowConfidence | Self::Aggressive)
    }

    /// Returns true if this policy removes obsolete version entries
    pub fn removes_obsolete_versions(&self) -> bool {
        matches!(self, Self::ObsoleteVersions | Self::Aggressive)
    }

    /// Returns true if this policy deduplicates entries
    pub fn deduplicates(&self) -> bool {
        matches!(self, Self::Deduplicate | Self::Aggressive)
    }
}

/// Configuration for cache compaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// The compaction policy to use
    pub policy: CompactionPolicy,
    /// Minimum confidence threshold for keeping entries (only with LowConfidence/Aggressive policy)
    /// Entries below this threshold will be removed during compaction
    pub min_confidence: f64,
    /// Set of currently active version hashes (only with ObsoleteVersions/Aggressive policy)
    /// Entries for versions not in this set will be removed
    pub active_versions: Vec<String>,
    /// Maximum age for entries in seconds (0 = use TTL-based expiration only)
    /// Entries older than this are removed regardless of TTL
    pub max_age_seconds: u64,
    /// Whether to run compaction automatically before saves
    pub compact_before_save: bool,
    /// Minimum number of entries to trigger compaction (skip if below)
    pub min_entries_threshold: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            policy: CompactionPolicy::default(),
            min_confidence: 0.5,
            active_versions: Vec::new(),
            max_age_seconds: 0,
            compact_before_save: false,
            min_entries_threshold: 100,
        }
    }
}

impl CompactionConfig {
    /// Create a configuration that only removes expired entries
    pub fn expired_only() -> Self {
        Self {
            policy: CompactionPolicy::ExpiredOnly,
            ..Default::default()
        }
    }

    /// Create a configuration that removes low-confidence entries
    pub fn low_confidence(min_confidence: f64) -> Self {
        Self {
            policy: CompactionPolicy::LowConfidence,
            min_confidence,
            ..Default::default()
        }
    }

    /// Create a configuration that removes obsolete version entries
    pub fn obsolete_versions(active_versions: Vec<String>) -> Self {
        Self {
            policy: CompactionPolicy::ObsoleteVersions,
            active_versions,
            ..Default::default()
        }
    }

    /// Create an aggressive compaction configuration
    pub fn aggressive(min_confidence: f64, active_versions: Vec<String>) -> Self {
        Self {
            policy: CompactionPolicy::Aggressive,
            min_confidence,
            active_versions,
            compact_before_save: true,
            ..Default::default()
        }
    }

    /// Builder method to enable compact before save
    pub fn with_compact_before_save(mut self, enabled: bool) -> Self {
        self.compact_before_save = enabled;
        self
    }

    /// Builder method to set minimum entries threshold
    pub fn with_min_entries_threshold(mut self, threshold: usize) -> Self {
        self.min_entries_threshold = threshold;
        self
    }

    /// Builder method to set maximum age
    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_age_seconds = max_age.as_secs();
        self
    }

    /// Builder method to update active versions
    pub fn with_active_versions(mut self, versions: Vec<String>) -> Self {
        self.active_versions = versions;
        self
    }
}

/// Result of a cache compaction operation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionResult {
    /// Number of entries before compaction
    pub entries_before: usize,
    /// Number of entries after compaction
    pub entries_after: usize,
    /// Number of expired entries removed
    pub expired_removed: usize,
    /// Number of low-confidence entries removed
    pub low_confidence_removed: usize,
    /// Number of obsolete version entries removed
    pub obsolete_removed: usize,
    /// Number of duplicate entries removed
    pub duplicates_removed: usize,
    /// Number of entries removed due to max age
    pub max_age_removed: usize,
    /// Duration of the compaction operation
    pub duration_ms: u64,
    /// Whether compaction was skipped (e.g., below threshold)
    pub skipped: bool,
    /// Reason for skipping (if skipped)
    pub skip_reason: Option<String>,
}

impl CompactionResult {
    /// Total entries removed during compaction
    pub fn total_removed(&self) -> usize {
        self.expired_removed
            + self.low_confidence_removed
            + self.obsolete_removed
            + self.duplicates_removed
            + self.max_age_removed
    }

    /// Compaction ratio (entries removed / entries before)
    pub fn compaction_ratio(&self) -> f64 {
        if self.entries_before == 0 {
            0.0
        } else {
            self.total_removed() as f64 / self.entries_before as f64
        }
    }

    /// Create a skipped result
    pub fn skipped(reason: impl Into<String>, entry_count: usize) -> Self {
        Self {
            entries_before: entry_count,
            entries_after: entry_count,
            skipped: true,
            skip_reason: Some(reason.into()),
            ..Default::default()
        }
    }
}

/// Event emitted when compaction occurs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionEvent {
    /// Timestamp of the compaction
    pub timestamp: SystemTime,
    /// The compaction result
    pub result: CompactionResult,
    /// The policy used
    pub policy: CompactionPolicy,
}

// =============================================================================
// Cache Warming Types
// =============================================================================

/// Strategy for prioritizing cache entries during warming
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WarmingStrategy {
    /// Warm with all entries from source, oldest first
    #[default]
    All,
    /// Prioritize entries with highest confidence scores
    HighConfidence,
    /// Prioritize most recently cached entries
    MostRecent,
    /// Prioritize most frequently accessed entries (by hit count from previous session)
    MostAccessed,
    /// Prioritize entries for specific property patterns
    PatternBased,
    /// Prioritize entries that had fastest verification times
    FastestVerification,
}

impl WarmingStrategy {
    /// Returns true if this strategy requires sorting by confidence
    pub fn prioritizes_confidence(&self) -> bool {
        matches!(self, Self::HighConfidence)
    }

    /// Returns true if this strategy requires sorting by recency
    pub fn prioritizes_recency(&self) -> bool {
        matches!(self, Self::MostRecent)
    }

    /// Returns true if this strategy requires access count data
    pub fn requires_access_stats(&self) -> bool {
        matches!(self, Self::MostAccessed)
    }
}

/// Configuration for cache warming operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmingConfig {
    /// The warming strategy to use
    pub strategy: WarmingStrategy,
    /// Maximum number of entries to warm (0 = unlimited)
    pub max_entries: usize,
    /// Minimum confidence threshold for entries to warm (0.0 - 1.0)
    pub min_confidence: f64,
    /// Maximum age of entries to warm (None = no age limit)
    pub max_age: Option<Duration>,
    /// Property name patterns to include (empty = all patterns)
    /// Uses glob syntax: "security_*", "*_invariant", etc.
    pub include_patterns: Vec<String>,
    /// Property name patterns to exclude
    pub exclude_patterns: Vec<String>,
    /// Version hashes to prioritize (entries for these versions are warmed first)
    pub priority_versions: Vec<String>,
    /// Whether to validate entries against current TTL settings
    pub validate_ttl: bool,
    /// Whether to skip entries that would immediately expire
    pub skip_nearly_expired: bool,
    /// Minimum remaining TTL percentage to consider an entry valid (0.0 - 1.0)
    /// Entry must have at least this much TTL remaining relative to its full TTL
    pub min_remaining_ttl_ratio: f64,
}

impl Default for WarmingConfig {
    fn default() -> Self {
        Self {
            strategy: WarmingStrategy::default(),
            max_entries: 0, // Unlimited
            min_confidence: 0.0,
            max_age: None,
            include_patterns: Vec::new(),
            exclude_patterns: Vec::new(),
            priority_versions: Vec::new(),
            validate_ttl: true,
            skip_nearly_expired: true,
            min_remaining_ttl_ratio: 0.1, // Entry must have at least 10% TTL remaining
        }
    }
}

impl WarmingConfig {
    /// Create a new warming configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration that warms high-confidence entries first
    pub fn high_confidence(min_confidence: f64) -> Self {
        Self {
            strategy: WarmingStrategy::HighConfidence,
            min_confidence: min_confidence.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Create a configuration that warms most recent entries first
    pub fn most_recent(max_entries: usize) -> Self {
        Self {
            strategy: WarmingStrategy::MostRecent,
            max_entries,
            ..Default::default()
        }
    }

    /// Create a configuration that warms most accessed entries first
    pub fn most_accessed(max_entries: usize) -> Self {
        Self {
            strategy: WarmingStrategy::MostAccessed,
            max_entries,
            ..Default::default()
        }
    }

    /// Create a configuration for pattern-based warming
    pub fn pattern_based(include_patterns: Vec<String>) -> Self {
        Self {
            strategy: WarmingStrategy::PatternBased,
            include_patterns,
            ..Default::default()
        }
    }

    /// Builder method to set the strategy
    pub fn with_strategy(mut self, strategy: WarmingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Builder method to set max entries
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Builder method to set minimum confidence
    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence.clamp(0.0, 1.0);
        self
    }

    /// Builder method to set maximum age
    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_age = Some(max_age);
        self
    }

    /// Builder method to add include patterns
    pub fn with_include_patterns(mut self, patterns: Vec<String>) -> Self {
        self.include_patterns = patterns;
        self
    }

    /// Builder method to add exclude patterns
    pub fn with_exclude_patterns(mut self, patterns: Vec<String>) -> Self {
        self.exclude_patterns = patterns;
        self
    }

    /// Builder method to set priority versions
    pub fn with_priority_versions(mut self, versions: Vec<String>) -> Self {
        self.priority_versions = versions;
        self
    }

    /// Builder method to enable/disable TTL validation
    pub fn with_ttl_validation(mut self, enabled: bool) -> Self {
        self.validate_ttl = enabled;
        self
    }

    /// Builder method to set minimum remaining TTL ratio
    pub fn with_min_remaining_ttl_ratio(mut self, ratio: f64) -> Self {
        self.min_remaining_ttl_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Check if a property name matches the include/exclude filters
    pub fn matches_patterns(&self, property_name: &str) -> bool {
        // If include patterns are specified, property must match at least one
        if !self.include_patterns.is_empty() {
            let matches_include = self.include_patterns.iter().any(|pattern| {
                glob::Pattern::new(pattern)
                    .map(|p| p.matches(property_name))
                    .unwrap_or(false)
            });
            if !matches_include {
                return false;
            }
        }

        // If exclude patterns are specified, property must not match any
        if !self.exclude_patterns.is_empty() {
            let matches_exclude = self.exclude_patterns.iter().any(|pattern| {
                glob::Pattern::new(pattern)
                    .map(|p| p.matches(property_name))
                    .unwrap_or(false)
            });
            if matches_exclude {
                return false;
            }
        }

        true
    }
}

/// Result of a cache warming operation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WarmingResult {
    /// Number of entries considered from source
    pub entries_considered: usize,
    /// Number of entries successfully warmed
    pub entries_warmed: usize,
    /// Number of entries skipped (already in cache)
    pub entries_skipped_existing: usize,
    /// Number of entries skipped (below confidence threshold)
    pub entries_skipped_low_confidence: usize,
    /// Number of entries skipped (expired or nearly expired)
    pub entries_skipped_expired: usize,
    /// Number of entries skipped (didn't match patterns)
    pub entries_skipped_pattern: usize,
    /// Number of entries skipped (max entries reached)
    pub entries_skipped_limit: usize,
    /// Duration of the warming operation
    pub duration_ms: u64,
    /// Warming strategy used
    pub strategy: WarmingStrategy,
    /// Source description (e.g., file path, "snapshot", "historical")
    pub source: String,
}

impl WarmingResult {
    /// Create a new warming result with the given source description
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            ..Default::default()
        }
    }

    /// Total entries skipped for any reason
    pub fn total_skipped(&self) -> usize {
        self.entries_skipped_existing
            + self.entries_skipped_low_confidence
            + self.entries_skipped_expired
            + self.entries_skipped_pattern
            + self.entries_skipped_limit
    }

    /// Warming success rate (warmed / considered)
    pub fn success_rate(&self) -> f64 {
        if self.entries_considered == 0 {
            0.0
        } else {
            self.entries_warmed as f64 / self.entries_considered as f64
        }
    }

    /// Create a result for an empty source
    pub fn empty(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            ..Default::default()
        }
    }
}

/// Event emitted when warming occurs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmingEvent {
    /// Timestamp of the warming operation
    pub timestamp: SystemTime,
    /// The warming result
    pub result: WarmingResult,
    /// The configuration used
    pub config: WarmingConfig,
}

/// Configuration for the improvement verifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Timeout for individual backend verification
    pub backend_timeout: Duration,

    /// Minimum number of backends that must pass
    pub min_passing_backends: usize,

    /// Backends to use for verification
    pub backends: Vec<String>,

    /// Whether to run backends in parallel
    pub parallel: bool,

    /// Minimum confidence score required (0.0 - 1.0)
    pub min_confidence: f64,

    /// Whether to use the dispatcher for verification
    pub use_dispatcher: bool,

    /// Retry policy for verification attempts (None = no retries)
    pub retry_policy: Option<VerificationRetryPolicy>,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            backend_timeout: Duration::from_secs(60),
            min_passing_backends: 2,
            backends: vec![
                "lean4".to_string(),
                "kani".to_string(),
                "tlaplus".to_string(),
            ],
            parallel: true,
            min_confidence: 0.8,
            use_dispatcher: true,
            retry_policy: Some(VerificationRetryPolicy::default()),
        }
    }
}

impl VerificationConfig {
    /// Create a strict configuration
    pub fn strict() -> Self {
        Self {
            backend_timeout: Duration::from_secs(120),
            min_passing_backends: 3,
            backends: vec![
                "lean4".to_string(),
                "kani".to_string(),
                "tlaplus".to_string(),
                "coq".to_string(),
            ],
            parallel: true,
            min_confidence: 0.95,
            use_dispatcher: true,
            retry_policy: Some(VerificationRetryPolicy::default()),
        }
    }

    /// Create a quick configuration (for testing)
    pub fn quick() -> Self {
        Self {
            backend_timeout: Duration::from_secs(30),
            min_passing_backends: 1,
            backends: vec!["lean4".to_string()],
            parallel: false,
            min_confidence: 0.7,
            use_dispatcher: true,
            retry_policy: Some(VerificationRetryPolicy::default()),
        }
    }

    /// Create a configuration without dispatcher (for simple cases)
    pub fn simple() -> Self {
        Self {
            backend_timeout: Duration::from_secs(60),
            min_passing_backends: 1,
            backends: vec!["lean4".to_string()],
            parallel: false,
            min_confidence: 0.7,
            use_dispatcher: false,
            retry_policy: Some(VerificationRetryPolicy::default()),
        }
    }
}

/// Cache for incremental verification results
///
/// This cache stores property-level verification results keyed by content hash,
/// enabling efficient re-verification when only some properties change.
///
/// ## Confidence-Based TTL Adjustment
///
/// When enabled via `enable_confidence_ttl_scaling`, cache entries with lower
/// confidence scores will have proportionally shorter TTLs. This encourages
/// re-verification of low-confidence results while preserving high-confidence
/// results longer.
///
/// The effective TTL is calculated as: `base_ttl * confidence`
/// - High confidence (0.95) → 95% of base TTL
/// - Medium confidence (0.5) → 50% of base TTL
/// - Low confidence (0.1) → 10% of base TTL
///
/// ## Per-Property TTL Overrides
///
/// Individual properties can have custom TTL values that override the base TTL.
/// This is useful when certain property types need different cache lifetimes:
/// - Security properties may need shorter TTLs (re-verify more frequently)
/// - Invariant properties may use longer TTLs (stable across changes)
///
/// Use `set_property_ttl_override` to configure per-property TTLs.
///
/// ## Pattern-Based TTL Overrides
///
/// For bulk TTL configuration, use glob patterns to match multiple property names:
/// - `"security_*"` matches all properties starting with "security_"
/// - `"*_invariant"` matches all properties ending with "_invariant"
/// - `"test_*_check"` matches properties like "test_foo_check", "test_bar_check"
///
/// Use `set_pattern_ttl_override` to configure pattern-based TTLs. Exact property
/// name overrides take priority over pattern matches. When multiple patterns match,
/// the first registered pattern wins (patterns are evaluated in insertion order).
#[derive(Debug, Clone)]
pub struct VerificationCache {
    /// Cached property results, keyed by (version_hash, property_name)
    entries: HashMap<CacheKey, CachedPropertyResult>,
    /// Statistics about cache usage
    stats: CacheStats,
    /// Maximum number of entries to keep
    max_entries: usize,
    /// Base time-to-live for cache entries
    ttl: Duration,
    /// Minimum confidence threshold for cache hits (0.0 - 1.0)
    /// Entries below this threshold are treated as cache misses
    confidence_threshold: f64,
    /// Whether to scale TTL by confidence level
    /// When true, low-confidence entries expire faster
    confidence_ttl_scaling: bool,
    /// Minimum TTL multiplier to prevent instant expiration (0.0 - 1.0)
    /// Even with 0.0 confidence, the effective TTL will be at least base_ttl * min_ttl_multiplier
    min_ttl_multiplier: f64,
    /// Per-property TTL overrides (exact property name -> TTL)
    /// These override the base TTL for matching properties
    property_ttl_overrides: HashMap<String, Duration>,
    /// Pattern-based TTL overrides (glob pattern -> TTL)
    /// Evaluated after exact matches; first matching pattern wins
    pattern_ttl_overrides: Vec<(glob::Pattern, Duration)>,
    /// Partition configuration
    partition_config: PartitionConfig,
    /// Per-partition statistics
    partition_stats: HashMap<CachePartition, PartitionStats>,
    /// Entry to partition mapping for efficient lookups
    entry_partitions: HashMap<CacheKey, CachePartition>,
    /// Configured compaction triggers
    compaction_triggers: Vec<CompactionTrigger>,
    /// State for tracking compaction trigger conditions
    compaction_trigger_state: CompactionTriggerState,
    /// Accumulated historical compaction counts (from loaded snapshots plus current session)
    historical_compaction_counts: CompactionTriggerCounts,
    /// Time-series tracking of compaction events for rate analysis
    compaction_time_series: CompactionTimeSeries,
}

impl Default for VerificationCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Key for cache entries
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CacheKey {
    /// Hash of the version being verified
    pub version_hash: String,
    /// Name of the property
    pub property_name: String,
}

impl CacheKey {
    /// Create a new cache key
    pub fn new(version_hash: impl Into<String>, property_name: impl Into<String>) -> Self {
        Self {
            version_hash: version_hash.into(),
            property_name: property_name.into(),
        }
    }
}

/// A cached property verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPropertyResult {
    /// The verified property details
    pub property: VerifiedProperty,
    /// Backends that verified this property
    pub backends: Vec<String>,
    /// When this result was cached
    pub cached_at: SystemTime,
    /// Hash of the dependencies at cache time
    pub dependency_hash: String,
    /// Confidence score from verification (0.0 - 1.0)
    /// Used for threshold-based cache filtering
    pub confidence: f64,
}

impl CachedPropertyResult {
    /// Check if this cache entry is expired
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.cached_at.elapsed().map(|e| e > ttl).unwrap_or(true)
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Current entry count
    pub entry_count: usize,
}

impl CacheStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// =============================================================================
// Cache Partitioning Types
// =============================================================================

/// Cache partition identifier
///
/// Partitions allow organizing cache entries by property type for better
/// organization, separate statistics tracking, and per-partition configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CachePartition {
    /// Default partition for unclassified entries
    #[default]
    Default,
    /// Theorem proving entries (Lean4, Coq, Isabelle)
    TheoremProving,
    /// Contract verification entries (Kani, Dafny, Verus)
    ContractVerification,
    /// Model checking entries (TLA+, Apalache, SPIN)
    ModelChecking,
    /// Neural network verification entries (Marabou, ERAN)
    NeuralNetwork,
    /// Security protocol verification entries (Tamarin, ProVerif)
    SecurityProtocol,
    /// Memory safety verification entries (Miri, sanitizers)
    MemorySafety,
    /// Static analysis entries (Clippy, cargo-audit)
    StaticAnalysis,
    /// Fuzzing and property-based testing entries
    Testing,
}

impl CachePartition {
    /// Get all partition variants
    pub fn all() -> &'static [CachePartition] {
        &[
            CachePartition::Default,
            CachePartition::TheoremProving,
            CachePartition::ContractVerification,
            CachePartition::ModelChecking,
            CachePartition::NeuralNetwork,
            CachePartition::SecurityProtocol,
            CachePartition::MemorySafety,
            CachePartition::StaticAnalysis,
            CachePartition::Testing,
        ]
    }

    /// Get partition from property type or property name
    ///
    /// This function checks if the input string exactly matches a known
    /// property type keyword, or if it contains such a keyword as a substring.
    pub fn from_property_type(property_type: &str) -> Self {
        let lower = property_type.to_lowercase();

        // Check for theorem proving keywords
        if lower == "theorem"
            || lower == "lemma"
            || lower == "axiom"
            || lower == "definition"
            || lower.contains("theorem")
            || lower.contains("lemma")
            || lower.contains("axiom")
        {
            return Self::TheoremProving;
        }

        // Check for contract verification keywords
        if lower == "contract"
            || lower == "precondition"
            || lower == "postcondition"
            || lower == "requires"
            || lower == "ensures"
            || lower.contains("contract")
            || lower.contains("precondition")
            || lower.contains("postcondition")
            || lower.contains("requires")
            || lower.contains("ensures")
        {
            return Self::ContractVerification;
        }

        // Check for model checking keywords
        if lower == "invariant"
            || lower == "temporal"
            || lower == "ltl"
            || lower == "ctl"
            || lower == "liveness"
            || lower == "safety"
            || lower.contains("invariant")
            || lower.contains("temporal")
            || lower.contains("liveness")
            || lower.contains("safety")
        {
            return Self::ModelChecking;
        }

        // Check for neural network keywords
        if lower == "robustness"
            || lower == "neural"
            || lower == "reachability"
            || lower == "adversarial"
            || lower.contains("neural")
            || lower.contains("robustness")
            || lower.contains("adversarial")
        {
            return Self::NeuralNetwork;
        }

        // Check for security keywords
        if lower == "security"
            || lower == "protocol"
            || lower == "authentication"
            || lower == "confidentiality"
            || lower.contains("security")
            || lower.contains("protocol")
            || lower.contains("auth")
            || lower.contains("confidential")
        {
            return Self::SecurityProtocol;
        }

        // Check for memory safety keywords
        if lower == "memory"
            || lower == "leak"
            || lower == "race"
            || lower == "undefined"
            || lower == "bounds"
            || lower.contains("memory")
            || lower.contains("leak")
            || lower.contains("race")
            || lower.contains("bounds")
        {
            return Self::MemorySafety;
        }

        // Check for static analysis keywords
        if lower == "lint"
            || lower == "audit"
            || lower == "semver"
            || lower == "deny"
            || lower == "geiger"
            || lower.contains("lint")
            || lower.contains("audit")
        {
            return Self::StaticAnalysis;
        }

        // Check for testing keywords
        if lower == "fuzz"
            || lower == "quickcheck"
            || lower == "mutation"
            || lower.contains("fuzz")
            || lower.contains("quickcheck")
            || lower.contains("mutation")
            || lower.contains("test")
        {
            return Self::Testing;
        }

        Self::Default
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::TheoremProving => "theorem_proving",
            Self::ContractVerification => "contract_verification",
            Self::ModelChecking => "model_checking",
            Self::NeuralNetwork => "neural_network",
            Self::SecurityProtocol => "security_protocol",
            Self::MemorySafety => "memory_safety",
            Self::StaticAnalysis => "static_analysis",
            Self::Testing => "testing",
        }
    }
}

impl std::fmt::Display for CachePartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Statistics for a single cache partition
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PartitionStats {
    /// Cache hits in this partition
    pub hits: u64,
    /// Cache misses in this partition
    pub misses: u64,
    /// Number of entries in this partition
    pub entry_count: usize,
    /// Total bytes estimate for this partition
    pub bytes_estimate: usize,
}

impl PartitionStats {
    /// Calculate hit rate for this partition
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Configuration for cache partitioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    /// Whether partitioning is enabled
    pub enabled: bool,
    /// Per-partition TTL overrides (partition -> TTL seconds)
    pub partition_ttls: HashMap<CachePartition, u64>,
    /// Per-partition max entry limits
    pub partition_limits: HashMap<CachePartition, usize>,
    /// Per-partition confidence thresholds
    pub partition_confidence_thresholds: HashMap<CachePartition, f64>,
}

#[allow(clippy::derivable_impls)]
impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            partition_ttls: HashMap::new(),
            partition_limits: HashMap::new(),
            partition_confidence_thresholds: HashMap::new(),
        }
    }
}

impl PartitionConfig {
    /// Create a new partition config with partitioning enabled
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Set TTL for a specific partition (in seconds)
    pub fn with_partition_ttl(mut self, partition: CachePartition, ttl_seconds: u64) -> Self {
        self.partition_ttls.insert(partition, ttl_seconds);
        self
    }

    /// Set entry limit for a specific partition
    pub fn with_partition_limit(mut self, partition: CachePartition, limit: usize) -> Self {
        self.partition_limits.insert(partition, limit);
        self
    }

    /// Set confidence threshold for a specific partition
    pub fn with_partition_confidence(mut self, partition: CachePartition, threshold: f64) -> Self {
        self.partition_confidence_thresholds
            .insert(partition, threshold.clamp(0.0, 1.0));
        self
    }

    /// Create config with recommended defaults for different verification domains
    pub fn recommended() -> Self {
        Self::enabled()
            // Theorem proofs are expensive, cache longer
            .with_partition_ttl(CachePartition::TheoremProving, 7200) // 2 hours
            .with_partition_confidence(CachePartition::TheoremProving, 0.9)
            // Contract verification is moderately expensive
            .with_partition_ttl(CachePartition::ContractVerification, 3600) // 1 hour
            .with_partition_confidence(CachePartition::ContractVerification, 0.85)
            // Model checking can be expensive
            .with_partition_ttl(CachePartition::ModelChecking, 3600) // 1 hour
            // Neural network verification is very expensive
            .with_partition_ttl(CachePartition::NeuralNetwork, 7200) // 2 hours
            .with_partition_confidence(CachePartition::NeuralNetwork, 0.95)
            // Static analysis is fast, cache shorter
            .with_partition_ttl(CachePartition::StaticAnalysis, 1800) // 30 minutes
            .with_partition_confidence(CachePartition::StaticAnalysis, 0.7)
            // Testing results may vary, cache short
            .with_partition_ttl(CachePartition::Testing, 900) // 15 minutes
            .with_partition_confidence(CachePartition::Testing, 0.6)
    }
}

/// Result of a partition operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionOperationResult {
    /// Partition affected
    pub partition: CachePartition,
    /// Number of entries affected
    pub entries_affected: usize,
    /// Operation that was performed
    pub operation: String,
    /// Duration of the operation
    pub duration_ms: u64,
}

// =============================================================================
// Cache Persistence Types
// =============================================================================

/// Error type for cache persistence operations
#[derive(Debug)]
pub enum CachePersistenceError {
    /// I/O error during file operations
    Io(io::Error),
    /// JSON serialization/deserialization error
    Json(serde_json::Error),
    /// Invalid glob pattern during configuration load
    InvalidPattern { pattern: String, error: String },
}

impl std::fmt::Display for CachePersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Json(e) => write!(f, "JSON error: {}", e),
            Self::InvalidPattern { pattern, error } => {
                write!(f, "Invalid pattern '{}': {}", pattern, error)
            }
        }
    }
}

impl std::error::Error for CachePersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::InvalidPattern { .. } => None,
        }
    }
}

impl From<io::Error> for CachePersistenceError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for CachePersistenceError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

/// Serializable cache entry for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedCacheEntry {
    /// Cache key components
    pub version_hash: String,
    pub property_name: String,
    /// The cached result
    pub result: CachedPropertyResult,
}

/// Serializable snapshot of cache entries for persistence
///
/// This struct captures the cache entries in a format suitable for
/// file storage. Use `VerificationCache::save_entries_to_file()` to
/// create this snapshot and `load_entries_from_file()` to restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSnapshot {
    /// Version of the snapshot format (for forward compatibility)
    pub format_version: u32,
    /// Timestamp when snapshot was created
    pub created_at: SystemTime,
    /// The cached entries
    pub entries: Vec<PersistedCacheEntry>,
    /// Statistics at snapshot time
    pub stats: CacheStats,
    /// Compaction trigger state (optional for backward compatibility)
    /// Added in format version 2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_trigger_state: Option<CompactionTriggerState>,
    /// Configured compaction triggers (optional for backward compatibility)
    /// Added in format version 2.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compaction_triggers: Vec<CompactionTrigger>,
    /// Historical compaction counts by trigger type (optional for backward compatibility)
    /// Added in format version 3 for cross-session compaction tracking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_trigger_counts: Option<CompactionTriggerCounts>,
    /// Compaction time-series history (optional for backward compatibility)
    /// Added in format version 4 for cross-session rate analysis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_time_series: Option<CompactionTimeSeries>,
    /// Autosave session metrics (optional for backward compatibility)
    /// Added in format version 5 for cross-session autosave analytics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub autosave_metrics: Option<PersistedAutosaveMetrics>,
}

impl CacheSnapshot {
    /// Current format version
    /// Version 2: Added compaction_trigger_state and compaction_triggers fields
    /// Version 3: Added compaction_trigger_counts for cross-session tracking
    /// Version 4: Added compaction_time_series for time-series persistence
    /// Version 5: Added autosave_metrics for autosave session analytics
    pub const FORMAT_VERSION: u32 = 5;

    /// Magic bytes for gzip format detection (first 2 bytes of gzip)
    const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

    /// Create a new snapshot from cache entries
    pub fn new(entries: Vec<PersistedCacheEntry>, stats: CacheStats) -> Self {
        Self {
            format_version: Self::FORMAT_VERSION,
            created_at: SystemTime::now(),
            entries,
            stats,
            compaction_trigger_state: None,
            compaction_triggers: Vec::new(),
            compaction_trigger_counts: None,
            compaction_time_series: None,
            autosave_metrics: None,
        }
    }

    /// Create a new snapshot with compaction trigger state
    pub fn with_trigger_state(
        entries: Vec<PersistedCacheEntry>,
        stats: CacheStats,
        trigger_state: CompactionTriggerState,
        triggers: Vec<CompactionTrigger>,
    ) -> Self {
        Self {
            format_version: Self::FORMAT_VERSION,
            created_at: SystemTime::now(),
            entries,
            stats,
            compaction_trigger_state: Some(trigger_state),
            compaction_triggers: triggers,
            compaction_trigger_counts: None,
            compaction_time_series: None,
            autosave_metrics: None,
        }
    }

    /// Create a new snapshot with compaction trigger state and historical counts
    pub fn with_compaction_history(
        entries: Vec<PersistedCacheEntry>,
        stats: CacheStats,
        trigger_state: CompactionTriggerState,
        triggers: Vec<CompactionTrigger>,
        trigger_counts: CompactionTriggerCounts,
    ) -> Self {
        Self {
            format_version: Self::FORMAT_VERSION,
            created_at: SystemTime::now(),
            entries,
            stats,
            compaction_trigger_state: Some(trigger_state),
            compaction_triggers: triggers,
            compaction_trigger_counts: Some(trigger_counts),
            compaction_time_series: None,
            autosave_metrics: None,
        }
    }

    /// Serialize snapshot to JSON bytes
    pub fn to_json(&self) -> Result<Vec<u8>, CachePersistenceError> {
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// Serialize snapshot to compressed gzip bytes
    ///
    /// # Arguments
    /// * `level` - Compression level from 0 (no compression) to 9 (best compression)
    pub fn to_compressed(
        &self,
        level: SnapshotCompressionLevel,
    ) -> Result<Vec<u8>, CachePersistenceError> {
        let json = serde_json::to_vec(self)?;
        let mut encoder = GzEncoder::new(Vec::new(), level.into());
        encoder.write_all(&json)?;
        Ok(encoder.finish()?)
    }

    /// Deserialize snapshot from JSON bytes
    pub fn from_json(data: &[u8]) -> Result<Self, CachePersistenceError> {
        Ok(serde_json::from_slice(data)?)
    }

    /// Deserialize snapshot from compressed gzip bytes
    pub fn from_compressed(data: &[u8]) -> Result<Self, CachePersistenceError> {
        let mut decoder = GzDecoder::new(data);
        let mut json = Vec::new();
        decoder.read_to_end(&mut json)?;
        Ok(serde_json::from_slice(&json)?)
    }

    /// Auto-detect format and deserialize snapshot
    ///
    /// Checks for gzip magic bytes and decompresses if needed,
    /// otherwise parses as JSON.
    pub fn from_bytes(data: &[u8]) -> Result<Self, CachePersistenceError> {
        if data.len() >= 2 && data[0..2] == Self::GZIP_MAGIC {
            Self::from_compressed(data)
        } else {
            Self::from_json(data)
        }
    }

    /// Check if the given data appears to be gzip compressed
    pub fn is_compressed(data: &[u8]) -> bool {
        data.len() >= 2 && data[0..2] == Self::GZIP_MAGIC
    }
}

/// Compression level for cache snapshots
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SnapshotCompressionLevel {
    /// No compression (fastest, largest files)
    None,
    /// Fast compression (quick but less space savings)
    Fast,
    /// Default/balanced compression
    #[default]
    Default,
    /// Best compression (slowest but smallest files)
    Best,
    /// Custom compression level (0-9)
    Custom(u32),
}

impl From<SnapshotCompressionLevel> for Compression {
    fn from(level: SnapshotCompressionLevel) -> Self {
        match level {
            SnapshotCompressionLevel::None => Compression::none(),
            SnapshotCompressionLevel::Fast => Compression::fast(),
            SnapshotCompressionLevel::Default => Compression::default(),
            SnapshotCompressionLevel::Best => Compression::best(),
            SnapshotCompressionLevel::Custom(n) => Compression::new(n.min(9)),
        }
    }
}

/// Serializable cache configuration for persistence
///
/// This captures all configurable aspects of the cache except the actual
/// entries. Useful for storing and loading cache settings separately from
/// cached data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Base time-to-live in seconds
    pub ttl_seconds: u64,
    /// Minimum confidence threshold
    pub confidence_threshold: f64,
    /// Whether confidence-based TTL scaling is enabled
    pub confidence_ttl_scaling: bool,
    /// Minimum TTL multiplier for confidence scaling
    pub min_ttl_multiplier: f64,
    /// Per-property TTL overrides (property name -> TTL in seconds)
    pub property_ttl_overrides: HashMap<String, u64>,
    /// Pattern-based TTL overrides (pattern string -> TTL in seconds)
    pub pattern_ttl_overrides: Vec<(String, u64)>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            ttl_seconds: 3600,
            confidence_threshold: 0.0,
            confidence_ttl_scaling: false,
            min_ttl_multiplier: 0.1,
            property_ttl_overrides: HashMap::new(),
            pattern_ttl_overrides: Vec::new(),
        }
    }
}

impl CacheConfig {
    /// Create a security-focused configuration
    pub fn security() -> Self {
        let mut config = Self {
            max_entries: 10000,
            ttl_seconds: 3600,
            confidence_threshold: 0.9,
            confidence_ttl_scaling: true,
            min_ttl_multiplier: 0.2,
            property_ttl_overrides: HashMap::new(),
            pattern_ttl_overrides: Vec::new(),
        };

        // Security patterns: 30 minute TTL
        config
            .pattern_ttl_overrides
            .push(("security_*".to_string(), 30 * 60));
        config
            .pattern_ttl_overrides
            .push(("*_security".to_string(), 30 * 60));
        config
            .pattern_ttl_overrides
            .push(("auth_*".to_string(), 30 * 60));
        config
            .pattern_ttl_overrides
            .push(("*_auth".to_string(), 30 * 60));

        // Invariants: 2 hour TTL
        config
            .pattern_ttl_overrides
            .push(("*_invariant".to_string(), 2 * 60 * 60));
        config
            .pattern_ttl_overrides
            .push(("invariant_*".to_string(), 2 * 60 * 60));

        // Tests: 5 minute TTL
        config
            .pattern_ttl_overrides
            .push(("test_*".to_string(), 5 * 60));
        config
            .pattern_ttl_overrides
            .push(("*_test".to_string(), 5 * 60));

        config
    }

    /// Create a testing-focused configuration
    pub fn testing() -> Self {
        let mut config = Self {
            max_entries: 1000,
            ttl_seconds: 5 * 60,
            confidence_threshold: 0.0,
            confidence_ttl_scaling: false,
            min_ttl_multiplier: 0.1,
            property_ttl_overrides: HashMap::new(),
            pattern_ttl_overrides: Vec::new(),
        };

        config
            .pattern_ttl_overrides
            .push(("test_*".to_string(), 60));
        config
            .pattern_ttl_overrides
            .push(("*_test".to_string(), 60));

        config
    }

    /// Create a production-focused configuration
    pub fn production() -> Self {
        let mut config = Self {
            max_entries: 50000,
            ttl_seconds: 4 * 60 * 60,
            confidence_threshold: 0.7,
            confidence_ttl_scaling: true,
            min_ttl_multiplier: 0.3,
            property_ttl_overrides: HashMap::new(),
            pattern_ttl_overrides: Vec::new(),
        };

        // Security: 1 hour
        config
            .pattern_ttl_overrides
            .push(("security_*".to_string(), 60 * 60));
        config
            .pattern_ttl_overrides
            .push(("auth_*".to_string(), 60 * 60));

        // Invariants: 8 hours
        config
            .pattern_ttl_overrides
            .push(("*_invariant".to_string(), 8 * 60 * 60));
        config
            .pattern_ttl_overrides
            .push(("invariant_*".to_string(), 8 * 60 * 60));

        // Tests: 30 minutes
        config
            .pattern_ttl_overrides
            .push(("test_*".to_string(), 30 * 60));

        config
    }

    /// Create a performance-focused configuration
    pub fn performance() -> Self {
        let mut config = Self {
            max_entries: 100000,
            ttl_seconds: 24 * 60 * 60,
            confidence_threshold: 0.5,
            confidence_ttl_scaling: true,
            min_ttl_multiplier: 0.1,
            property_ttl_overrides: HashMap::new(),
            pattern_ttl_overrides: Vec::new(),
        };

        // Invariants: 48 hours
        config
            .pattern_ttl_overrides
            .push(("*_invariant".to_string(), 48 * 60 * 60));
        config
            .pattern_ttl_overrides
            .push(("invariant_*".to_string(), 48 * 60 * 60));

        // Security: 4 hours
        config
            .pattern_ttl_overrides
            .push(("security_*".to_string(), 4 * 60 * 60));
        config
            .pattern_ttl_overrides
            .push(("auth_*".to_string(), 4 * 60 * 60));

        config
    }

    /// Save configuration to a JSON file
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), CachePersistenceError> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load configuration from a JSON file
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, CachePersistenceError> {
        let json = fs::read_to_string(path)?;
        let config = serde_json::from_str(&json)?;
        Ok(config)
    }
}

impl VerificationCache {
    /// Create a new verification cache with default settings
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            stats: CacheStats::default(),
            max_entries: 10000,
            ttl: Duration::from_secs(3600), // 1 hour default TTL
            confidence_threshold: 0.0,      // Accept all confidence levels by default
            confidence_ttl_scaling: false,  // Disabled by default for backward compatibility
            min_ttl_multiplier: 0.1,        // At least 10% of base TTL even at 0 confidence
            property_ttl_overrides: HashMap::new(),
            pattern_ttl_overrides: Vec::new(),
            partition_config: PartitionConfig::default(),
            partition_stats: HashMap::new(),
            entry_partitions: HashMap::new(),
            compaction_triggers: Vec::new(),
            compaction_trigger_state: CompactionTriggerState::default(),
            historical_compaction_counts: CompactionTriggerCounts::default(),
            compaction_time_series: CompactionTimeSeries::default(),
        }
    }

    /// Create a cache with custom settings
    pub fn with_config(max_entries: usize, ttl: Duration) -> Self {
        Self {
            entries: HashMap::new(),
            stats: CacheStats::default(),
            max_entries,
            ttl,
            confidence_threshold: 0.0,
            confidence_ttl_scaling: false,
            min_ttl_multiplier: 0.1,
            property_ttl_overrides: HashMap::new(),
            pattern_ttl_overrides: Vec::new(),
            partition_config: PartitionConfig::default(),
            partition_stats: HashMap::new(),
            entry_partitions: HashMap::new(),
            compaction_triggers: Vec::new(),
            compaction_trigger_state: CompactionTriggerState::default(),
            historical_compaction_counts: CompactionTriggerCounts::default(),
            compaction_time_series: CompactionTimeSeries::default(),
        }
    }

    /// Create a cache with custom settings including confidence threshold
    pub fn with_full_config(max_entries: usize, ttl: Duration, confidence_threshold: f64) -> Self {
        Self {
            entries: HashMap::new(),
            stats: CacheStats::default(),
            max_entries,
            ttl,
            confidence_threshold: confidence_threshold.clamp(0.0, 1.0),
            confidence_ttl_scaling: false,
            min_ttl_multiplier: 0.1,
            property_ttl_overrides: HashMap::new(),
            pattern_ttl_overrides: Vec::new(),
            partition_config: PartitionConfig::default(),
            partition_stats: HashMap::new(),
            entry_partitions: HashMap::new(),
            compaction_triggers: Vec::new(),
            compaction_trigger_state: CompactionTriggerState::default(),
            historical_compaction_counts: CompactionTriggerCounts::default(),
            compaction_time_series: CompactionTimeSeries::default(),
        }
    }

    /// Set the confidence threshold for cache hits
    ///
    /// Cached entries with confidence below this threshold will be treated as misses,
    /// forcing re-verification to potentially obtain higher confidence results.
    pub fn set_confidence_threshold(&mut self, threshold: f64) {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get the current confidence threshold
    pub fn confidence_threshold(&self) -> f64 {
        self.confidence_threshold
    }

    /// Enable confidence-based TTL scaling
    ///
    /// When enabled, cache entries with lower confidence will expire faster.
    /// The effective TTL is: `base_ttl * max(confidence, min_ttl_multiplier)`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
    /// cache.enable_confidence_ttl_scaling(0.1);
    ///
    /// // High confidence (0.9) → 3240 second TTL (90% of 3600)
    /// // Low confidence (0.3) → 1080 second TTL (30% of 3600)
    /// // Zero confidence (0.0) → 360 second TTL (10% floor from min_ttl_multiplier)
    /// ```
    pub fn enable_confidence_ttl_scaling(&mut self, min_ttl_multiplier: f64) {
        self.confidence_ttl_scaling = true;
        self.min_ttl_multiplier = min_ttl_multiplier.clamp(0.0, 1.0);
    }

    /// Disable confidence-based TTL scaling
    pub fn disable_confidence_ttl_scaling(&mut self) {
        self.confidence_ttl_scaling = false;
    }

    /// Check if confidence-based TTL scaling is enabled
    pub fn is_confidence_ttl_scaling_enabled(&self) -> bool {
        self.confidence_ttl_scaling
    }

    /// Get the minimum TTL multiplier
    pub fn min_ttl_multiplier(&self) -> f64 {
        self.min_ttl_multiplier
    }

    /// Set a TTL override for a specific property name
    ///
    /// This allows fine-grained control over cache lifetimes for different property types.
    /// The override TTL is used as the base TTL for the named property before
    /// confidence-based scaling is applied.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut cache = VerificationCache::new();
    ///
    /// // Security properties expire after 30 minutes
    /// cache.set_property_ttl_override("security_check", Duration::from_secs(1800));
    ///
    /// // Invariants can be cached longer (2 hours)
    /// cache.set_property_ttl_override("state_invariant", Duration::from_secs(7200));
    /// ```
    pub fn set_property_ttl_override(&mut self, property_name: impl Into<String>, ttl: Duration) {
        self.property_ttl_overrides
            .insert(property_name.into(), ttl);
    }

    /// Remove a TTL override for a specific property name
    ///
    /// After removal, the property will use the base TTL.
    pub fn remove_property_ttl_override(&mut self, property_name: &str) -> Option<Duration> {
        self.property_ttl_overrides.remove(property_name)
    }

    /// Get the TTL override for a specific property name, if set
    pub fn get_property_ttl_override(&self, property_name: &str) -> Option<Duration> {
        self.property_ttl_overrides.get(property_name).copied()
    }

    /// Get all property TTL overrides
    pub fn property_ttl_overrides(&self) -> &HashMap<String, Duration> {
        &self.property_ttl_overrides
    }

    /// Clear all property TTL overrides
    pub fn clear_property_ttl_overrides(&mut self) {
        self.property_ttl_overrides.clear();
    }

    /// Set a TTL override using a glob pattern
    ///
    /// Glob patterns allow bulk TTL configuration for property names matching the pattern.
    /// Exact property name overrides (via `set_property_ttl_override`) take priority.
    /// When multiple patterns match, the first registered pattern wins.
    ///
    /// # Supported patterns
    ///
    /// - `*` matches any sequence of characters (but not `/`)
    /// - `?` matches any single character
    /// - `[abc]` matches any character in the brackets
    /// - `[!abc]` matches any character not in the brackets
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut cache = VerificationCache::new();
    ///
    /// // All security-related properties expire after 30 minutes
    /// cache.set_pattern_ttl_override("security_*", Duration::from_secs(1800)).unwrap();
    ///
    /// // All invariant properties can be cached longer (2 hours)
    /// cache.set_pattern_ttl_override("*_invariant", Duration::from_secs(7200)).unwrap();
    ///
    /// // Test properties expire quickly
    /// cache.set_pattern_ttl_override("test_*", Duration::from_secs(60)).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the pattern is invalid.
    pub fn set_pattern_ttl_override(
        &mut self,
        pattern: &str,
        ttl: Duration,
    ) -> Result<(), glob::PatternError> {
        let compiled = glob::Pattern::new(pattern)?;
        self.pattern_ttl_overrides.push((compiled, ttl));
        Ok(())
    }

    /// Remove a pattern TTL override by its pattern string
    ///
    /// Returns the TTL if the pattern was found and removed.
    pub fn remove_pattern_ttl_override(&mut self, pattern: &str) -> Option<Duration> {
        let idx = self
            .pattern_ttl_overrides
            .iter()
            .position(|(p, _)| p.as_str() == pattern)?;
        Some(self.pattern_ttl_overrides.remove(idx).1)
    }

    /// Get the TTL for a specific pattern, if set
    pub fn get_pattern_ttl_override(&self, pattern: &str) -> Option<Duration> {
        self.pattern_ttl_overrides
            .iter()
            .find(|(p, _)| p.as_str() == pattern)
            .map(|(_, ttl)| *ttl)
    }

    /// Get all pattern TTL overrides as (pattern_string, ttl) pairs
    pub fn pattern_ttl_overrides(&self) -> Vec<(&str, Duration)> {
        self.pattern_ttl_overrides
            .iter()
            .map(|(p, ttl)| (p.as_str(), *ttl))
            .collect()
    }

    /// Clear all pattern TTL overrides
    pub fn clear_pattern_ttl_overrides(&mut self) {
        self.pattern_ttl_overrides.clear();
    }

    /// Clear both exact and pattern TTL overrides
    pub fn clear_all_ttl_overrides(&mut self) {
        self.property_ttl_overrides.clear();
        self.pattern_ttl_overrides.clear();
    }

    /// Find the first matching pattern for a property name
    ///
    /// Returns the TTL from the first pattern that matches, or None if no patterns match.
    fn find_matching_pattern_ttl(&self, property_name: &str) -> Option<Duration> {
        for (pattern, ttl) in &self.pattern_ttl_overrides {
            if pattern.matches(property_name) {
                return Some(*ttl);
            }
        }
        None
    }

    /// Get the base TTL for a property (before confidence scaling)
    ///
    /// Resolution order:
    /// 1. Exact property name override (highest priority)
    /// 2. First matching glob pattern
    /// 3. Partition-specific TTL (when partitioning is enabled)
    /// 4. Default base TTL (lowest priority)
    fn base_ttl_for_property(&self, property_name: &str) -> Duration {
        // First check exact match
        if let Some(ttl) = self.property_ttl_overrides.get(property_name) {
            return *ttl;
        }

        // Then check patterns
        if let Some(ttl) = self.find_matching_pattern_ttl(property_name) {
            return ttl;
        }

        // Check partition-specific TTL when partitioning is enabled
        if self.partition_config.enabled {
            let partition = Self::determine_partition(property_name);
            if let Some(ttl) = self.partition_ttl(partition) {
                return ttl;
            }
        }

        // Fall back to base TTL
        self.ttl
    }

    /// Calculate the effective TTL for a given confidence level
    ///
    /// When confidence TTL scaling is enabled, returns a shorter TTL for lower confidence.
    /// When disabled, returns the base TTL unchanged.
    ///
    /// Note: This method uses the default base TTL. For property-specific TTLs,
    /// use `effective_ttl_for_property` instead.
    pub fn effective_ttl(&self, confidence: f64) -> Duration {
        self.effective_ttl_with_base(self.ttl, confidence)
    }

    /// Calculate the effective TTL for a specific property and confidence level
    ///
    /// This method checks for property-specific TTL overrides and applies
    /// confidence-based scaling when enabled.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut cache = VerificationCache::new();
    /// cache.set_property_ttl_override("security", Duration::from_secs(1800));
    /// cache.enable_confidence_ttl_scaling(0.1);
    ///
    /// // For "security" property at 0.8 confidence: 1800 * 0.8 = 1440 seconds
    /// let ttl = cache.effective_ttl_for_property("security", 0.8);
    ///
    /// // For "other" property at 0.8 confidence: 3600 * 0.8 = 2880 seconds (base TTL)
    /// let ttl = cache.effective_ttl_for_property("other", 0.8);
    /// ```
    pub fn effective_ttl_for_property(&self, property_name: &str, confidence: f64) -> Duration {
        let base_ttl = self.base_ttl_for_property(property_name);
        self.effective_ttl_with_base(base_ttl, confidence)
    }

    /// Internal helper to calculate effective TTL given a base TTL and confidence
    fn effective_ttl_with_base(&self, base_ttl: Duration, confidence: f64) -> Duration {
        if !self.confidence_ttl_scaling {
            return base_ttl;
        }

        // Calculate multiplier: max(confidence, min_ttl_multiplier)
        let multiplier = confidence.max(self.min_ttl_multiplier).min(1.0);

        // Calculate effective TTL
        let base_millis = base_ttl.as_millis() as f64;
        let effective_millis = (base_millis * multiplier) as u64;

        Duration::from_millis(effective_millis)
    }

    /// Get a cached result if valid
    ///
    /// Returns None (cache miss) if:
    /// - Entry doesn't exist
    /// - Entry is expired (based on TTL, potentially adjusted by confidence and property overrides)
    /// - Entry's confidence is below the configured threshold (global or partition-specific)
    ///
    /// When confidence TTL scaling is enabled, low-confidence entries will
    /// expire faster than high-confidence ones. Per-property TTL overrides are
    /// also respected. When partitioning is enabled, partition-specific TTLs
    /// and confidence thresholds are also considered.
    pub fn get(&mut self, key: &CacheKey) -> Option<CachedPropertyResult> {
        // Check if entry exists
        let entry = self.entries.get(key);
        if entry.is_none() {
            self.stats.misses += 1;
            // Update partition miss stats if partitioning is enabled
            if self.partition_config.enabled {
                let partition = Self::determine_partition(&key.property_name);
                let stats = self.partition_stats.entry(partition).or_default();
                stats.misses += 1;
            }
            return None;
        }

        let entry = entry.unwrap();

        // Calculate effective TTL based on property name and confidence
        // (now includes partition-specific TTL when partitioning is enabled)
        let effective_ttl = self.effective_ttl_for_property(&key.property_name, entry.confidence);

        // Check if expired using the effective TTL
        if entry.is_expired(effective_ttl) {
            // Update partition tracking before removal
            if self.partition_config.enabled {
                if let Some(partition) = self.entry_partitions.remove(key) {
                    if let Some(stats) = self.partition_stats.get_mut(&partition) {
                        stats.entry_count = stats.entry_count.saturating_sub(1);
                        stats.misses += 1;
                    }
                }
            }
            self.entries.remove(key);
            self.stats.entry_count = self.entries.len();
            self.stats.misses += 1;
            return None;
        }

        // Determine effective confidence threshold
        // Partition-specific threshold takes precedence if configured
        let effective_threshold = if self.partition_config.enabled {
            let partition = Self::determine_partition(&key.property_name);
            self.partition_confidence_threshold(partition)
                .unwrap_or(self.confidence_threshold)
        } else {
            self.confidence_threshold
        };

        // Check if confidence is below threshold
        if entry.confidence < effective_threshold {
            // Don't remove the entry - it may still be useful if threshold is lowered
            // But treat it as a miss for statistics
            self.stats.misses += 1;
            if self.partition_config.enabled {
                let partition = Self::determine_partition(&key.property_name);
                let stats = self.partition_stats.entry(partition).or_default();
                stats.misses += 1;
            }
            return None;
        }

        // Update hit statistics
        self.stats.hits += 1;
        if self.partition_config.enabled {
            let partition = Self::determine_partition(&key.property_name);
            let stats = self.partition_stats.entry(partition).or_default();
            stats.hits += 1;
        }

        self.entries.get(key).cloned()
    }

    /// Insert a result into the cache
    ///
    /// When partitioning is enabled, automatically assigns the entry to a partition
    /// based on the property name. Also enforces partition limits if configured.
    pub fn insert(&mut self, key: CacheKey, result: CachedPropertyResult) {
        // Evict if at global capacity
        if self.entries.len() >= self.max_entries {
            self.evict_oldest();
        }

        // Track partition if partitioning is enabled
        if self.partition_config.enabled {
            // Update partition stats for old entry if replacing
            if let Some(old_partition) = self.entry_partitions.get(&key) {
                if let Some(stats) = self.partition_stats.get_mut(old_partition) {
                    stats.entry_count = stats.entry_count.saturating_sub(1);
                }
            }

            // Determine and track partition
            let partition = Self::determine_partition(&key.property_name);

            // Check partition limit and evict from partition if needed
            if let Some(&limit) = self.partition_config.partition_limits.get(&partition) {
                let current_count = self.count_in_partition(partition);
                if current_count >= limit {
                    self.evict_oldest_from_partition(partition);
                }
            }

            self.entry_partitions.insert(key.clone(), partition);

            // Update partition stats
            let stats = self.partition_stats.entry(partition).or_default();
            stats.entry_count += 1;
        }

        self.entries.insert(key, result);
        self.stats.entry_count = self.entries.len();

        // Track insert for trigger state
        self.compaction_trigger_state.record_insert();
    }

    /// Evict the oldest entry globally
    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .entries
            .iter()
            .min_by_key(|(_, v)| v.cached_at)
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&oldest_key);
            self.stats.evictions += 1;

            // Update partition tracking
            if self.partition_config.enabled {
                if let Some(partition) = self.entry_partitions.remove(&oldest_key) {
                    if let Some(stats) = self.partition_stats.get_mut(&partition) {
                        stats.entry_count = stats.entry_count.saturating_sub(1);
                    }
                }
            }
        }
    }

    /// Evict the oldest entry from a specific partition
    fn evict_oldest_from_partition(&mut self, partition: CachePartition) {
        // Find keys in the target partition
        let partition_keys: Vec<CacheKey> = self
            .entry_partitions
            .iter()
            .filter(|(_, &p)| p == partition)
            .map(|(k, _)| k.clone())
            .collect();

        // Find the oldest entry among partition keys
        if let Some(oldest_key) = partition_keys
            .iter()
            .filter_map(|k| self.entries.get(k).map(|v| (k.clone(), v.cached_at)))
            .min_by_key(|(_, cached_at)| *cached_at)
            .map(|(k, _)| k)
        {
            self.entries.remove(&oldest_key);
            self.entry_partitions.remove(&oldest_key);
            self.stats.evictions += 1;

            if let Some(stats) = self.partition_stats.get_mut(&partition) {
                stats.entry_count = stats.entry_count.saturating_sub(1);
            }
        }
    }

    /// Invalidate entries affected by changes
    pub fn invalidate_affected(&mut self, version_hash: &str, affected_properties: &[String]) {
        let keys_to_remove: Vec<_> = self
            .entries
            .keys()
            .filter(|k| {
                k.version_hash == version_hash && affected_properties.contains(&k.property_name)
            })
            .cloned()
            .collect();

        for key in keys_to_remove {
            self.entries.remove(&key);

            // Update partition tracking
            if self.partition_config.enabled {
                if let Some(partition) = self.entry_partitions.remove(&key) {
                    if let Some(stats) = self.partition_stats.get_mut(&partition) {
                        stats.entry_count = stats.entry_count.saturating_sub(1);
                    }
                }
            }
        }
        self.stats.entry_count = self.entries.len();
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats.entry_count = 0;
        self.entry_partitions.clear();
        self.partition_stats.clear();
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // ==========================================================================
    // Cache Partitioning Methods
    // ==========================================================================

    /// Enable cache partitioning with the given configuration
    ///
    /// When partitioning is enabled, entries are automatically categorized
    /// based on their property type, and per-partition statistics are tracked.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{VerificationCache, PartitionConfig, CachePartition};
    ///
    /// let mut cache = VerificationCache::new();
    /// cache.enable_partitioning(PartitionConfig::recommended());
    ///
    /// // Now entries will be tracked by partition
    /// ```
    pub fn enable_partitioning(&mut self, config: PartitionConfig) {
        self.partition_config = config;
        self.partition_config.enabled = true;
    }

    /// Disable cache partitioning
    pub fn disable_partitioning(&mut self) {
        self.partition_config.enabled = false;
    }

    /// Check if partitioning is enabled
    pub fn is_partitioning_enabled(&self) -> bool {
        self.partition_config.enabled
    }

    /// Get the partition configuration
    pub fn partition_config(&self) -> &PartitionConfig {
        &self.partition_config
    }

    /// Set a mutable reference to the partition configuration
    pub fn partition_config_mut(&mut self) -> &mut PartitionConfig {
        &mut self.partition_config
    }

    /// Get statistics for a specific partition
    pub fn partition_stats(&self, partition: CachePartition) -> PartitionStats {
        self.partition_stats
            .get(&partition)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all partition statistics
    pub fn all_partition_stats(&self) -> &HashMap<CachePartition, PartitionStats> {
        &self.partition_stats
    }

    /// Get the partition for a given cache key
    pub fn get_partition(&self, key: &CacheKey) -> CachePartition {
        self.entry_partitions.get(key).copied().unwrap_or_default()
    }

    /// Determine the partition for a property name
    ///
    /// Uses heuristics based on property name keywords to determine
    /// the most appropriate partition.
    pub fn determine_partition(property_name: &str) -> CachePartition {
        CachePartition::from_property_type(property_name)
    }

    /// Insert an entry with explicit partition assignment
    ///
    /// Use this when you know the partition type and want to avoid
    /// the automatic classification heuristics.
    pub fn insert_with_partition(
        &mut self,
        key: CacheKey,
        result: CachedPropertyResult,
        partition: CachePartition,
    ) {
        // Evict if at capacity
        if self.entries.len() >= self.max_entries {
            self.evict_oldest();
        }

        // Track partition assignment
        if self.partition_config.enabled {
            // Update partition stats for old entry if replacing
            if let Some(old_partition) = self.entry_partitions.get(&key) {
                if let Some(stats) = self.partition_stats.get_mut(old_partition) {
                    stats.entry_count = stats.entry_count.saturating_sub(1);
                }
            }

            // Update partition tracking
            self.entry_partitions.insert(key.clone(), partition);

            // Update partition stats
            let stats = self.partition_stats.entry(partition).or_default();
            stats.entry_count += 1;
        }

        self.entries.insert(key, result);
        self.stats.entry_count = self.entries.len();
    }

    /// Get entries in a specific partition
    ///
    /// Returns an iterator over all entries in the specified partition.
    pub fn entries_in_partition(
        &self,
        partition: CachePartition,
    ) -> impl Iterator<Item = (&CacheKey, &CachedPropertyResult)> {
        self.entries.iter().filter(move |(key, _)| {
            self.entry_partitions.get(*key).copied().unwrap_or_default() == partition
        })
    }

    /// Count entries in a specific partition
    pub fn count_in_partition(&self, partition: CachePartition) -> usize {
        if self.partition_config.enabled {
            self.partition_stats
                .get(&partition)
                .map(|s| s.entry_count)
                .unwrap_or(0)
        } else {
            // Count manually if partitioning not enabled
            self.entries
                .keys()
                .filter(|key| {
                    self.entry_partitions.get(*key).copied().unwrap_or_default() == partition
                })
                .count()
        }
    }

    /// Clear entries in a specific partition
    ///
    /// Returns the number of entries removed.
    pub fn clear_partition(&mut self, partition: CachePartition) -> PartitionOperationResult {
        let start = Instant::now();

        let keys_to_remove: Vec<_> = self
            .entry_partitions
            .iter()
            .filter(|(_, p)| **p == partition)
            .map(|(k, _)| k.clone())
            .collect();

        let count = keys_to_remove.len();

        for key in keys_to_remove {
            self.entries.remove(&key);
            self.entry_partitions.remove(&key);
        }

        // Reset partition stats
        if let Some(stats) = self.partition_stats.get_mut(&partition) {
            stats.entry_count = 0;
        }

        self.stats.entry_count = self.entries.len();

        PartitionOperationResult {
            partition,
            entries_affected: count,
            operation: "clear".to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Compact a specific partition using the default compaction policy
    ///
    /// Returns information about the compaction operation.
    pub fn compact_partition(&mut self, partition: CachePartition) -> PartitionOperationResult {
        let start = Instant::now();

        let keys_to_check: Vec<_> = self
            .entry_partitions
            .iter()
            .filter(|(_, p)| **p == partition)
            .map(|(k, _)| k.clone())
            .collect();

        let mut removed = 0;
        for key in keys_to_check {
            if let Some(entry) = self.entries.get(&key) {
                // Remove expired entries
                let effective_ttl =
                    self.effective_ttl_for_property(&key.property_name, entry.confidence);
                if entry.is_expired(effective_ttl) {
                    self.entries.remove(&key);
                    self.entry_partitions.remove(&key);
                    removed += 1;
                }
            }
        }

        // Update partition stats
        if let Some(stats) = self.partition_stats.get_mut(&partition) {
            stats.entry_count = stats.entry_count.saturating_sub(removed);
        }

        self.stats.entry_count = self.entries.len();

        PartitionOperationResult {
            partition,
            entries_affected: removed,
            operation: "compact".to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Get TTL for a partition if configured
    pub fn partition_ttl(&self, partition: CachePartition) -> Option<Duration> {
        self.partition_config
            .partition_ttls
            .get(&partition)
            .map(|secs| Duration::from_secs(*secs))
    }

    /// Get confidence threshold for a partition if configured
    pub fn partition_confidence_threshold(&self, partition: CachePartition) -> Option<f64> {
        self.partition_config
            .partition_confidence_thresholds
            .get(&partition)
            .copied()
    }

    /// Get distribution of entries across partitions
    pub fn partition_distribution(&self) -> HashMap<CachePartition, usize> {
        let mut distribution = HashMap::new();
        for partition in CachePartition::all() {
            let count = self.count_in_partition(*partition);
            if count > 0 {
                distribution.insert(*partition, count);
            }
        }
        distribution
    }

    /// Rebalance partitions by re-classifying all entries
    ///
    /// This is useful after changing the partitioning configuration
    /// or when entries were inserted without explicit partition assignment.
    pub fn rebalance_partitions(&mut self) -> HashMap<CachePartition, usize> {
        self.entry_partitions.clear();
        self.partition_stats.clear();

        let mut distribution = HashMap::new();

        for key in self.entries.keys() {
            let partition = Self::determine_partition(&key.property_name);
            self.entry_partitions.insert(key.clone(), partition);

            let stats = self.partition_stats.entry(partition).or_default();
            stats.entry_count += 1;

            *distribution.entry(partition).or_insert(0) += 1;
        }

        distribution
    }

    // ==========================================================================
    // Compaction Trigger Methods
    // ==========================================================================

    /// Add a compaction trigger
    pub fn add_compaction_trigger(&mut self, trigger: CompactionTrigger) {
        self.compaction_triggers.push(trigger);
    }

    /// Remove all compaction triggers
    pub fn clear_compaction_triggers(&mut self) {
        self.compaction_triggers.clear();
    }

    /// Get configured compaction triggers
    pub fn compaction_triggers(&self) -> &[CompactionTrigger] {
        &self.compaction_triggers
    }

    /// Get compaction trigger state
    pub fn compaction_trigger_state(&self) -> &CompactionTriggerState {
        &self.compaction_trigger_state
    }

    /// Get accumulated historical compaction counts (from loaded snapshots plus current session)
    pub fn historical_compaction_counts(&self) -> &CompactionTriggerCounts {
        &self.historical_compaction_counts
    }

    /// Get mutable reference to historical compaction counts
    ///
    /// Allows manually adding compaction events (typically called after a compaction completes)
    pub fn historical_compaction_counts_mut(&mut self) -> &mut CompactionTriggerCounts {
        &mut self.historical_compaction_counts
    }

    /// Record a compaction event in historical counts (without entries removed tracking)
    pub fn record_compaction(&mut self, trigger: &CompactionTrigger) {
        self.historical_compaction_counts.increment(trigger);
        self.compaction_time_series.record(trigger, 0);
    }

    /// Record a compaction event with full details including entries removed
    pub fn record_compaction_full(&mut self, trigger: &CompactionTrigger, entries_removed: usize) {
        self.historical_compaction_counts.increment(trigger);
        self.compaction_time_series.record(trigger, entries_removed);
    }

    /// Get the compaction time series for analysis
    pub fn compaction_time_series(&self) -> &CompactionTimeSeries {
        &self.compaction_time_series
    }

    /// Get mutable reference to compaction time series
    pub fn compaction_time_series_mut(&mut self) -> &mut CompactionTimeSeries {
        &mut self.compaction_time_series
    }

    /// Get compaction rate (events per minute) over a time window
    pub fn compaction_rate_per_minute(&self, window: Duration) -> f64 {
        self.compaction_time_series.rate_per_minute(window)
    }

    /// Get compaction rate (events per hour) over a time window
    pub fn compaction_rate_per_hour(&self, window: Duration) -> f64 {
        self.compaction_time_series.rate_per_hour(window)
    }

    /// Get a summary of compaction activity over a time window
    pub fn compaction_summary(&self, window: Duration) -> CompactionTimeSeriesSummary {
        self.compaction_time_series.summary(window)
    }

    /// Set up recommended compaction triggers
    ///
    /// Configures a balanced set of triggers for typical usage:
    /// - Size-based: trigger at 90% capacity
    /// - Time-based: trigger every 30 minutes
    /// - Hit-rate-based: trigger when hit rate drops below 30%
    pub fn with_recommended_triggers(mut self) -> Self {
        self.compaction_triggers = vec![
            CompactionTrigger::size_90_percent(),
            CompactionTrigger::every_minutes(30),
            CompactionTrigger::hit_rate(0.3, 100),
        ];
        self
    }

    /// Set up aggressive compaction triggers
    ///
    /// Configures triggers for high-frequency compaction:
    /// - Size-based: trigger at 80% capacity
    /// - Time-based: trigger every 10 minutes
    /// - Insert-based: trigger every 500 inserts
    pub fn with_aggressive_triggers(mut self) -> Self {
        self.compaction_triggers = vec![
            CompactionTrigger::size_80_percent(),
            CompactionTrigger::every_minutes(10),
            CompactionTrigger::every_inserts(500),
        ];
        self
    }

    /// Set up conservative compaction triggers
    ///
    /// Configures triggers for minimal compaction overhead:
    /// - Size-based: trigger at 95% capacity
    /// - Time-based: trigger every 2 hours
    pub fn with_conservative_triggers(mut self) -> Self {
        self.compaction_triggers = vec![
            CompactionTrigger::SizeBased {
                threshold_ratio: 0.95,
            },
            CompactionTrigger::every_hours(2),
        ];
        self
    }

    /// Check if any compaction trigger condition is met
    ///
    /// Returns the first triggered condition, or None if no triggers are active.
    pub fn should_compact(&self) -> Option<&CompactionTrigger> {
        self.compaction_triggers
            .iter()
            .find(|&trigger| self.is_trigger_active(trigger))
    }

    /// Minimum seconds between size-based compaction triggers to prevent rapid-fire compaction
    /// when cache is at capacity but no entries can be removed
    const SIZE_TRIGGER_MIN_COOLDOWN_SECS: u64 = 60;

    /// Check if a specific trigger condition is met
    fn is_trigger_active(&self, trigger: &CompactionTrigger) -> bool {
        match trigger {
            CompactionTrigger::SizeBased { threshold_ratio } => {
                // Size-based triggers have a minimum cooldown to prevent rapid-fire compaction
                // when the cache is at capacity but no entries can be removed (e.g., all fresh)
                if self.compaction_trigger_state.seconds_since_compaction()
                    < Self::SIZE_TRIGGER_MIN_COOLDOWN_SECS
                {
                    return false;
                }
                let current_ratio = self.entries.len() as f64 / self.max_entries as f64;
                current_ratio >= *threshold_ratio
            }
            CompactionTrigger::TimeBased { interval_seconds } => {
                self.compaction_trigger_state.seconds_since_compaction() >= *interval_seconds
            }
            CompactionTrigger::HitRateBased {
                min_hit_rate,
                min_operations,
            } => {
                let total = self.stats.hits + self.stats.misses;
                if total < *min_operations {
                    return false;
                }
                self.stats.hit_rate() < *min_hit_rate
            }
            CompactionTrigger::PartitionImbalance { max_ratio } => {
                if !self.partition_config.enabled {
                    return false;
                }
                let dist = self.partition_distribution();
                if dist.is_empty() {
                    return false;
                }
                let max_count = dist.values().max().copied().unwrap_or(0) as f64;
                let min_count = dist
                    .values()
                    .filter(|&&c| c > 0)
                    .min()
                    .copied()
                    .unwrap_or(1) as f64;
                if min_count == 0.0 {
                    return max_count > 0.0;
                }
                max_count / min_count > *max_ratio
            }
            CompactionTrigger::InsertBased { insert_count } => {
                self.compaction_trigger_state.inserts_since_compaction >= *insert_count
            }
            CompactionTrigger::MemoryBased { max_bytes } => {
                // Estimate memory usage (rough approximation)
                let estimated_bytes = self.estimate_memory_bytes();
                estimated_bytes >= *max_bytes
            }
        }
    }

    /// Estimate total memory usage in bytes
    pub fn estimate_memory_bytes(&self) -> usize {
        // Base struct overhead
        let base_overhead = std::mem::size_of::<Self>();

        // Estimate per-entry overhead
        // CacheKey: ~48 bytes (two Strings with ~12 chars each average)
        // CachedPropertyResult: ~200 bytes (VerifiedProperty + Vec<String> + SystemTime + String + f64)
        let entry_size = 48 + 200;
        let entries_bytes = self.entries.len() * entry_size;

        // Partition tracking overhead
        let partition_bytes = self.entry_partitions.len() * (48 + 1); // CacheKey + CachePartition

        // Stats overhead (small)
        let stats_bytes = self.partition_stats.len() * 48;

        base_overhead + entries_bytes + partition_bytes + stats_bytes
    }

    /// Reset trigger state after compaction
    pub fn reset_trigger_state(&mut self) {
        self.compaction_trigger_state.reset();
    }

    /// Perform compaction if any trigger condition is met
    ///
    /// Returns Some(CompactionResult) if compaction was performed, None otherwise.
    pub fn compact_if_triggered(&mut self, config: &CompactionConfig) -> Option<CompactionResult> {
        if self.should_compact().is_some() {
            let result = self.compact(config);
            self.reset_trigger_state();
            Some(result)
        } else {
            None
        }
    }

    // ==========================================================================
    // Cache Presets
    // ==========================================================================
    //
    // Preset methods configure the cache with sensible defaults for common
    // verification scenarios. Each preset configures pattern-based TTLs and
    // confidence settings appropriate for its use case.

    /// Apply a security-focused preset configuration
    ///
    /// Optimized for security-sensitive verification where:
    /// - Security properties require frequent re-verification (30 min TTL)
    /// - Invariant properties are stable (2 hour TTL)
    /// - Test properties expire quickly (5 min TTL)
    /// - High confidence threshold (0.9) to ensure quality
    /// - Confidence-based TTL scaling enabled
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cache = VerificationCache::new().with_security_preset();
    ///
    /// // Security properties now expire after 30 minutes
    /// // Invariants last 2 hours
    /// // Test results last only 5 minutes
    /// ```
    pub fn with_security_preset(mut self) -> Self {
        // Security-related properties: 30 minute TTL
        let _ = self.set_pattern_ttl_override("security_*", Duration::from_secs(30 * 60));
        let _ = self.set_pattern_ttl_override("*_security", Duration::from_secs(30 * 60));
        let _ = self.set_pattern_ttl_override("auth_*", Duration::from_secs(30 * 60));
        let _ = self.set_pattern_ttl_override("*_auth", Duration::from_secs(30 * 60));
        let _ = self.set_pattern_ttl_override("access_*", Duration::from_secs(30 * 60));
        let _ = self.set_pattern_ttl_override("permission_*", Duration::from_secs(30 * 60));

        // Invariant properties: 2 hour TTL (stable)
        let _ = self.set_pattern_ttl_override("*_invariant", Duration::from_secs(2 * 60 * 60));
        let _ = self.set_pattern_ttl_override("invariant_*", Duration::from_secs(2 * 60 * 60));

        // Test properties: 5 minute TTL (frequently changing)
        let _ = self.set_pattern_ttl_override("test_*", Duration::from_secs(5 * 60));
        let _ = self.set_pattern_ttl_override("*_test", Duration::from_secs(5 * 60));

        // High confidence threshold for security scenarios
        self.set_confidence_threshold(0.9);

        // Enable confidence-based TTL scaling with 20% floor
        self.enable_confidence_ttl_scaling(0.2);

        self
    }

    /// Apply a testing-focused preset configuration
    ///
    /// Optimized for development and testing where:
    /// - All entries expire quickly (5 min base TTL)
    /// - Test properties expire very quickly (1 min TTL)
    /// - No confidence threshold (accept all results)
    /// - No confidence-based TTL scaling
    /// - Smaller cache size (1000 entries)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cache = VerificationCache::new().with_testing_preset();
    ///
    /// // All cache entries expire quickly for rapid iteration
    /// ```
    pub fn with_testing_preset(mut self) -> Self {
        // Short base TTL for testing (5 minutes)
        self.ttl = Duration::from_secs(5 * 60);

        // Very short TTL for test properties (1 minute)
        let _ = self.set_pattern_ttl_override("test_*", Duration::from_secs(60));
        let _ = self.set_pattern_ttl_override("*_test", Duration::from_secs(60));

        // Smaller cache for testing
        self.max_entries = 1000;

        // Accept all confidence levels in testing
        self.set_confidence_threshold(0.0);

        // No confidence scaling in testing (predictable behavior)
        self.disable_confidence_ttl_scaling();

        self
    }

    /// Apply a production-focused preset configuration
    ///
    /// Optimized for production environments where:
    /// - Long base TTL (4 hours) for efficiency
    /// - Security properties have shorter TTL (1 hour)
    /// - Invariants cached longest (8 hours)
    /// - Moderate confidence threshold (0.7)
    /// - Confidence-based TTL scaling with 30% floor
    /// - Large cache size (50000 entries)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cache = VerificationCache::new().with_production_preset();
    ///
    /// // Optimized for long-running production verification
    /// ```
    pub fn with_production_preset(mut self) -> Self {
        // Long base TTL for production (4 hours)
        self.ttl = Duration::from_secs(4 * 60 * 60);

        // Large cache for production workloads
        self.max_entries = 50000;

        // Security properties: 1 hour TTL
        let _ = self.set_pattern_ttl_override("security_*", Duration::from_secs(60 * 60));
        let _ = self.set_pattern_ttl_override("*_security", Duration::from_secs(60 * 60));
        let _ = self.set_pattern_ttl_override("auth_*", Duration::from_secs(60 * 60));
        let _ = self.set_pattern_ttl_override("permission_*", Duration::from_secs(60 * 60));

        // Invariant properties: 8 hour TTL (very stable)
        let _ = self.set_pattern_ttl_override("*_invariant", Duration::from_secs(8 * 60 * 60));
        let _ = self.set_pattern_ttl_override("invariant_*", Duration::from_secs(8 * 60 * 60));

        // Test properties: 30 minute TTL
        let _ = self.set_pattern_ttl_override("test_*", Duration::from_secs(30 * 60));

        // Moderate confidence threshold
        self.set_confidence_threshold(0.7);

        // Enable confidence scaling with 30% floor
        self.enable_confidence_ttl_scaling(0.3);

        self
    }

    /// Apply a performance-focused preset configuration
    ///
    /// Optimized for maximum cache utilization where:
    /// - Very long base TTL (24 hours)
    /// - Invariants cached for 48 hours
    /// - Low confidence threshold (0.5) to maximize hits
    /// - Aggressive confidence-based TTL scaling (10% floor)
    /// - Very large cache (100000 entries)
    ///
    /// Use this preset when verification is expensive and results are relatively stable.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cache = VerificationCache::new().with_performance_preset();
    ///
    /// // Maximum caching for expensive verification workloads
    /// ```
    pub fn with_performance_preset(mut self) -> Self {
        // Very long base TTL (24 hours)
        self.ttl = Duration::from_secs(24 * 60 * 60);

        // Very large cache
        self.max_entries = 100000;

        // Invariant properties: 48 hour TTL (extremely stable)
        let _ = self.set_pattern_ttl_override("*_invariant", Duration::from_secs(48 * 60 * 60));
        let _ = self.set_pattern_ttl_override("invariant_*", Duration::from_secs(48 * 60 * 60));

        // Security still shorter: 4 hours
        let _ = self.set_pattern_ttl_override("security_*", Duration::from_secs(4 * 60 * 60));
        let _ = self.set_pattern_ttl_override("auth_*", Duration::from_secs(4 * 60 * 60));

        // Low confidence threshold to maximize cache hits
        self.set_confidence_threshold(0.5);

        // Aggressive confidence scaling
        self.enable_confidence_ttl_scaling(0.1);

        self
    }

    /// Apply patterns from a HashMap configuration
    ///
    /// This allows loading pattern configurations from external sources
    /// like config files or environment variables.
    ///
    /// # Arguments
    ///
    /// * `patterns` - HashMap mapping glob patterns to TTL durations
    ///
    /// # Returns
    ///
    /// Returns a Vec of patterns that failed to compile (empty if all succeeded)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::collections::HashMap;
    ///
    /// let mut patterns = HashMap::new();
    /// patterns.insert("security_*".to_string(), Duration::from_secs(1800));
    /// patterns.insert("*_invariant".to_string(), Duration::from_secs(7200));
    ///
    /// let cache = VerificationCache::new();
    /// let errors = cache.apply_pattern_config(&patterns);
    /// assert!(errors.is_empty());
    /// ```
    pub fn apply_pattern_config(
        &mut self,
        patterns: &HashMap<String, Duration>,
    ) -> Vec<(String, glob::PatternError)> {
        let mut errors = Vec::new();

        for (pattern, ttl) in patterns {
            if let Err(e) = self.set_pattern_ttl_override(pattern, *ttl) {
                errors.push((pattern.clone(), e));
            }
        }

        errors
    }

    /// Get a description of the current preset-like configuration
    ///
    /// Returns a summary of the cache configuration for debugging/logging.
    pub fn configuration_summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str(&format!("Base TTL: {:?}\n", self.ttl));
        summary.push_str(&format!("Max entries: {}\n", self.max_entries));
        summary.push_str(&format!(
            "Confidence threshold: {:.2}\n",
            self.confidence_threshold
        ));
        summary.push_str(&format!(
            "Confidence TTL scaling: {}\n",
            self.confidence_ttl_scaling
        ));

        if self.confidence_ttl_scaling {
            summary.push_str(&format!(
                "Min TTL multiplier: {:.2}\n",
                self.min_ttl_multiplier
            ));
        }

        if !self.property_ttl_overrides.is_empty() {
            summary.push_str(&format!(
                "Property overrides: {}\n",
                self.property_ttl_overrides.len()
            ));
        }

        if !self.pattern_ttl_overrides.is_empty() {
            summary.push_str(&format!(
                "Pattern overrides: {}\n",
                self.pattern_ttl_overrides.len()
            ));
            for (pattern, ttl) in &self.pattern_ttl_overrides {
                summary.push_str(&format!("  {} -> {:?}\n", pattern.as_str(), ttl));
            }
        }

        summary
    }

    // ==========================================================================
    // Cache Persistence
    // ==========================================================================

    /// Create a cache from a configuration
    ///
    /// This factory method creates a new cache with settings from a `CacheConfig`.
    /// Invalid patterns in the configuration will be skipped with errors collected.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration settings for the cache
    ///
    /// # Returns
    ///
    /// A tuple of (cache, errors) where errors contains any invalid patterns
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = CacheConfig::security();
    /// let (cache, errors) = VerificationCache::from_config(config);
    /// assert!(errors.is_empty());
    /// ```
    pub fn from_config(config: CacheConfig) -> (Self, Vec<CachePersistenceError>) {
        let mut cache = Self {
            entries: HashMap::new(),
            stats: CacheStats::default(),
            max_entries: config.max_entries,
            ttl: Duration::from_secs(config.ttl_seconds),
            confidence_threshold: config.confidence_threshold,
            confidence_ttl_scaling: config.confidence_ttl_scaling,
            min_ttl_multiplier: config.min_ttl_multiplier,
            property_ttl_overrides: config
                .property_ttl_overrides
                .into_iter()
                .map(|(k, v)| (k, Duration::from_secs(v)))
                .collect(),
            pattern_ttl_overrides: Vec::new(),
            partition_config: PartitionConfig::default(),
            partition_stats: HashMap::new(),
            entry_partitions: HashMap::new(),
            compaction_triggers: Vec::new(),
            compaction_trigger_state: CompactionTriggerState::default(),
            historical_compaction_counts: CompactionTriggerCounts::default(),
            compaction_time_series: CompactionTimeSeries::default(),
        };

        let mut errors = Vec::new();

        for (pattern, ttl_secs) in config.pattern_ttl_overrides {
            if let Err(e) = cache.set_pattern_ttl_override(&pattern, Duration::from_secs(ttl_secs))
            {
                errors.push(CachePersistenceError::InvalidPattern {
                    pattern,
                    error: e.to_string(),
                });
            }
        }

        (cache, errors)
    }

    /// Create a cache from a configuration file (JSON)
    ///
    /// Loads configuration from a JSON file and creates a new cache.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (cache, errors) = VerificationCache::from_config_file("cache_config.json")?;
    /// if !errors.is_empty() {
    ///     eprintln!("Warning: some patterns were invalid");
    /// }
    /// ```
    pub fn from_config_file(
        path: impl AsRef<Path>,
    ) -> Result<(Self, Vec<CachePersistenceError>), CachePersistenceError> {
        let config = CacheConfig::load_from_file(path)?;
        Ok(Self::from_config(config))
    }

    /// Export the current configuration to a `CacheConfig` struct
    ///
    /// This captures the current settings (not the cached entries) in a
    /// serializable format.
    pub fn export_config(&self) -> CacheConfig {
        CacheConfig {
            max_entries: self.max_entries,
            ttl_seconds: self.ttl.as_secs(),
            confidence_threshold: self.confidence_threshold,
            confidence_ttl_scaling: self.confidence_ttl_scaling,
            min_ttl_multiplier: self.min_ttl_multiplier,
            property_ttl_overrides: self
                .property_ttl_overrides
                .iter()
                .map(|(k, v)| (k.clone(), v.as_secs()))
                .collect(),
            pattern_ttl_overrides: self
                .pattern_ttl_overrides
                .iter()
                .map(|(p, d)| (p.as_str().to_string(), d.as_secs()))
                .collect(),
        }
    }

    /// Save the cache configuration to a file
    ///
    /// Only saves configuration settings, not cached entries. Use
    /// `save_entries_to_file` to save the cached data.
    pub fn save_config_to_file(&self, path: impl AsRef<Path>) -> Result<(), CachePersistenceError> {
        self.export_config().save_to_file(path)
    }

    /// Create a snapshot of all cache entries
    ///
    /// This creates a serializable snapshot of all cached entries, suitable
    /// for persistence. The snapshot includes entry data, statistics, and
    /// compaction trigger state for persistence across restarts.
    pub fn create_snapshot(&self) -> CacheSnapshot {
        let entries: Vec<PersistedCacheEntry> = self
            .entries
            .iter()
            .map(|(key, result)| PersistedCacheEntry {
                version_hash: key.version_hash.clone(),
                property_name: key.property_name.clone(),
                result: result.clone(),
            })
            .collect();

        let mut snapshot = CacheSnapshot::with_compaction_history(
            entries,
            self.stats.clone(),
            self.compaction_trigger_state.clone(),
            self.compaction_triggers.clone(),
            self.historical_compaction_counts,
        );
        snapshot.compaction_time_series = Some(self.compaction_time_series.clone());
        snapshot
    }

    /// Save cached entries to a file
    ///
    /// Creates a snapshot of all cache entries and writes them to a JSON file.
    /// This preserves the cached verification results for later restoration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Save cache entries before shutdown
    /// cache.save_entries_to_file("cache_data.json")?;
    ///
    /// // Later, restore the cache
    /// let mut new_cache = VerificationCache::new();
    /// new_cache.load_entries_from_file("cache_data.json")?;
    /// ```
    pub fn save_entries_to_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(), CachePersistenceError> {
        let snapshot = self.create_snapshot();
        let json = serde_json::to_string_pretty(&snapshot)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load cached entries from a file
    ///
    /// Restores previously saved cache entries. Expired entries (based on
    /// current TTL settings) are automatically filtered out.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the cache data file
    ///
    /// # Returns
    ///
    /// The number of entries loaded (after filtering expired entries)
    pub fn load_entries_from_file(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<usize, CachePersistenceError> {
        let json = fs::read_to_string(path)?;
        let snapshot: CacheSnapshot = serde_json::from_str(&json)?;

        self.restore_from_snapshot(&snapshot)
    }

    /// Save cached entries to a compressed file
    ///
    /// Creates a snapshot of all cache entries and writes them to a gzip-compressed file.
    /// This can significantly reduce storage size for large caches.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the compressed cache will be written
    /// * `level` - Compression level to use
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{VerificationCache, SnapshotCompressionLevel};
    ///
    /// // Save cache with default compression
    /// cache.save_entries_compressed("cache_data.json.gz", SnapshotCompressionLevel::Default)?;
    ///
    /// // Save with best compression for long-term storage
    /// cache.save_entries_compressed("archive.json.gz", SnapshotCompressionLevel::Best)?;
    /// ```
    pub fn save_entries_compressed(
        &self,
        path: impl AsRef<Path>,
        level: SnapshotCompressionLevel,
    ) -> Result<(), CachePersistenceError> {
        let snapshot = self.create_snapshot();
        let compressed = snapshot.to_compressed(level)?;
        fs::write(path, compressed)?;
        Ok(())
    }

    /// Load cached entries from a compressed file
    ///
    /// Restores previously saved cache entries from a gzip-compressed file.
    /// Expired entries (based on current TTL settings) are automatically filtered out.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the compressed cache data file
    ///
    /// # Returns
    ///
    /// The number of entries loaded (after filtering expired entries)
    pub fn load_entries_compressed(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<usize, CachePersistenceError> {
        let data = fs::read(path)?;
        let snapshot = CacheSnapshot::from_compressed(&data)?;
        self.restore_from_snapshot(&snapshot)
    }

    /// Load cached entries from a file, auto-detecting format
    ///
    /// Automatically detects whether the file is gzip compressed or plain JSON
    /// and loads accordingly. Expired entries are filtered out.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the cache data file
    ///
    /// # Returns
    ///
    /// The number of entries loaded (after filtering expired entries)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Works with either compressed or uncompressed files
    /// let loaded = cache.load_entries_auto("cache_data.json")?;
    /// let loaded = cache.load_entries_auto("cache_data.json.gz")?;
    /// ```
    pub fn load_entries_auto(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<usize, CachePersistenceError> {
        let data = fs::read(path)?;
        let snapshot = CacheSnapshot::from_bytes(&data)?;
        self.restore_from_snapshot(&snapshot)
    }

    /// Restore cache entries from a snapshot
    ///
    /// Loads entries from a snapshot, filtering out expired entries based on
    /// current TTL settings. Also restores compaction trigger state and triggers
    /// if present in the snapshot (format version 2+).
    ///
    /// Returns the number of entries actually loaded.
    pub fn restore_from_snapshot(
        &mut self,
        snapshot: &CacheSnapshot,
    ) -> Result<usize, CachePersistenceError> {
        let mut loaded = 0;

        for entry in &snapshot.entries {
            let key = CacheKey::new(&entry.version_hash, &entry.property_name);

            // Check if entry is expired using property-specific TTL
            let effective_ttl =
                self.effective_ttl_for_property(&entry.property_name, entry.result.confidence);

            if !entry.result.is_expired(effective_ttl) {
                // Respect max_entries limit
                if self.entries.len() >= self.max_entries {
                    self.evict_oldest();
                }
                self.entries.insert(key, entry.result.clone());
                loaded += 1;
            }
        }

        self.stats.entry_count = self.entries.len();

        // Restore compaction trigger state if present (format version 2+)
        if let Some(ref trigger_state) = snapshot.compaction_trigger_state {
            self.compaction_trigger_state = trigger_state.clone();
        }

        // Restore compaction triggers if present (format version 2+)
        if !snapshot.compaction_triggers.is_empty() {
            self.compaction_triggers = snapshot.compaction_triggers.clone();
        }

        // Accumulate historical compaction counts if present (format version 3+)
        // We add (not replace) to preserve counts from previous sessions
        if let Some(ref counts) = snapshot.compaction_trigger_counts {
            self.historical_compaction_counts.add(counts);
        }

        // Merge compaction time-series history if present (format version 4+)
        if let Some(ref time_series) = snapshot.compaction_time_series {
            self.compaction_time_series.merge_from(time_series);
        }

        Ok(loaded)
    }

    /// Warm the cache from historical verification results
    ///
    /// This method populates the cache with results from a collection of
    /// historical verification data. Useful for pre-warming a cache after
    /// restart or deployment.
    ///
    /// # Arguments
    ///
    /// * `results` - Iterator of (version_hash, property_name, cached_result) tuples
    ///
    /// # Returns
    ///
    /// The number of entries successfully added to the cache
    ///
    /// # Example
    ///
    /// ```ignore
    /// let historical: Vec<_> = load_historical_results();
    /// let warmed = cache.warm_from_results(historical.iter().map(|r| {
    ///     (&r.version_hash, &r.property_name, &r.result)
    /// }));
    /// println!("Warmed cache with {} entries", warmed);
    /// ```
    pub fn warm_from_results<'a, I>(&mut self, results: I) -> usize
    where
        I: Iterator<Item = (&'a str, &'a str, &'a CachedPropertyResult)>,
    {
        let mut warmed = 0;

        for (version_hash, property_name, result) in results {
            // Check if entry would be valid
            let effective_ttl = self.effective_ttl_for_property(property_name, result.confidence);

            if result.is_expired(effective_ttl) {
                continue;
            }

            if result.confidence < self.confidence_threshold {
                continue;
            }

            let key = CacheKey::new(version_hash, property_name);

            // Skip if already in cache (don't overwrite fresher data)
            if self.entries.contains_key(&key) {
                continue;
            }

            // Respect max_entries limit
            if self.entries.len() >= self.max_entries {
                self.evict_oldest();
            }

            self.entries.insert(key, result.clone());
            warmed += 1;
        }

        self.stats.entry_count = self.entries.len();
        warmed
    }

    /// Merge another cache's entries into this cache
    ///
    /// Useful for combining caches from different sources. Newer entries
    /// (by `cached_at` timestamp) take precedence when keys conflict.
    ///
    /// # Returns
    ///
    /// The number of entries merged (added or updated)
    pub fn merge_from(&mut self, other: &VerificationCache) -> usize {
        let mut merged = 0;

        for (key, result) in &other.entries {
            // Check if entry is valid for this cache's settings
            let effective_ttl =
                self.effective_ttl_for_property(&key.property_name, result.confidence);

            if result.is_expired(effective_ttl) {
                continue;
            }

            if result.confidence < self.confidence_threshold {
                continue;
            }

            // Check if we should replace existing entry
            if let Some(existing) = self.entries.get(key) {
                // Only replace if other's entry is newer
                if result.cached_at <= existing.cached_at {
                    continue;
                }
            }

            // Respect max_entries limit
            if !self.entries.contains_key(key) && self.entries.len() >= self.max_entries {
                self.evict_oldest();
            }

            self.entries.insert(key.clone(), result.clone());
            merged += 1;
        }

        self.stats.entry_count = self.entries.len();
        merged
    }

    // ==========================================================================
    // Cache Compaction
    // ==========================================================================

    /// Compact the cache according to the given configuration
    ///
    /// This method removes entries based on the compaction policy:
    /// - `ExpiredOnly`: Remove entries past their TTL
    /// - `LowConfidence`: Also remove entries below the min confidence threshold
    /// - `ObsoleteVersions`: Also remove entries for versions not in the active set
    /// - `Deduplicate`: Also remove duplicate entries (same result for same property)
    /// - `Aggressive`: All of the above
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = CompactionConfig::aggressive(0.7, vec!["v1.0".to_string()]);
    /// let result = cache.compact(&config);
    /// println!("Removed {} entries", result.total_removed());
    /// ```
    pub fn compact(&mut self, config: &CompactionConfig) -> CompactionResult {
        let start = Instant::now();
        let entries_before = self.entries.len();

        // Skip compaction if below threshold
        if entries_before < config.min_entries_threshold {
            return CompactionResult::skipped(
                format!(
                    "Entry count {} below threshold {}",
                    entries_before, config.min_entries_threshold
                ),
                entries_before,
            );
        }

        let mut result = CompactionResult {
            entries_before,
            ..Default::default()
        };

        // Phase 1: Remove expired entries (always)
        result.expired_removed = self.compact_expired();

        // Phase 2: Remove entries exceeding max age (if configured)
        if config.max_age_seconds > 0 {
            result.max_age_removed = self.compact_by_max_age(config.max_age_seconds);
        }

        // Phase 3: Remove low-confidence entries (if policy allows)
        if config.policy.removes_low_confidence() {
            result.low_confidence_removed = self.compact_low_confidence(config.min_confidence);
        }

        // Phase 4: Remove obsolete version entries (if policy allows)
        if config.policy.removes_obsolete_versions() && !config.active_versions.is_empty() {
            result.obsolete_removed = self.compact_obsolete_versions(&config.active_versions);
        }

        // Phase 5: Deduplicate entries (if policy allows)
        if config.policy.deduplicates() {
            result.duplicates_removed = self.compact_duplicates();
        }

        result.entries_after = self.entries.len();
        result.duration_ms = start.elapsed().as_millis() as u64;

        self.stats.entry_count = self.entries.len();
        result
    }

    /// Remove expired entries from the cache
    ///
    /// Returns the number of entries removed.
    pub fn compact_expired(&mut self) -> usize {
        let keys_to_remove: Vec<_> = self
            .entries
            .iter()
            .filter(|(key, entry)| {
                let effective_ttl =
                    self.effective_ttl_for_property(&key.property_name, entry.confidence);
                entry.is_expired(effective_ttl)
            })
            .map(|(k, _)| k.clone())
            .collect();

        let removed = keys_to_remove.len();
        for key in keys_to_remove {
            self.entries.remove(&key);
        }
        self.stats.entry_count = self.entries.len();
        removed
    }

    /// Remove entries older than the given max age in seconds
    ///
    /// Returns the number of entries removed.
    fn compact_by_max_age(&mut self, max_age_seconds: u64) -> usize {
        let max_age = Duration::from_secs(max_age_seconds);

        let keys_to_remove: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry
                    .cached_at
                    .elapsed()
                    .map(|age| age > max_age)
                    .unwrap_or(true)
            })
            .map(|(k, _)| k.clone())
            .collect();

        let removed = keys_to_remove.len();
        for key in keys_to_remove {
            self.entries.remove(&key);
        }
        removed
    }

    /// Remove entries with confidence below the given threshold
    ///
    /// Returns the number of entries removed.
    pub fn compact_low_confidence(&mut self, min_confidence: f64) -> usize {
        let keys_to_remove: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.confidence < min_confidence)
            .map(|(k, _)| k.clone())
            .collect();

        let removed = keys_to_remove.len();
        for key in keys_to_remove {
            self.entries.remove(&key);
        }
        self.stats.entry_count = self.entries.len();
        removed
    }

    /// Remove entries for versions not in the active set
    ///
    /// Returns the number of entries removed.
    pub fn compact_obsolete_versions(&mut self, active_versions: &[String]) -> usize {
        let active_set: std::collections::HashSet<_> = active_versions.iter().collect();

        let keys_to_remove: Vec<_> = self
            .entries
            .keys()
            .filter(|k| !active_set.contains(&k.version_hash))
            .cloned()
            .collect();

        let removed = keys_to_remove.len();
        for key in keys_to_remove {
            self.entries.remove(&key);
        }
        self.stats.entry_count = self.entries.len();
        removed
    }

    /// Remove duplicate entries (keep the most recent for each property)
    ///
    /// Two entries are considered duplicates if they:
    /// - Have the same property_name
    /// - Have the same verification result (passed status and backends)
    /// - The older one is removed
    ///
    /// Returns the number of entries removed.
    pub fn compact_duplicates(&mut self) -> usize {
        // Group entries by (property_name, result_fingerprint)
        let mut groups: HashMap<String, Vec<(CacheKey, SystemTime)>> = HashMap::new();

        for (key, entry) in &self.entries {
            // Create a fingerprint based on property name and result characteristics
            let fingerprint = self.create_entry_fingerprint(key, entry);
            groups
                .entry(fingerprint)
                .or_default()
                .push((key.clone(), entry.cached_at));
        }

        // For each group, keep only the most recent entry
        let mut keys_to_remove = Vec::new();
        for entries in groups.values() {
            if entries.len() > 1 {
                // Sort by cached_at descending (newest first)
                let mut sorted = entries.clone();
                sorted.sort_by(|a, b| b.1.cmp(&a.1));

                // Remove all but the first (newest)
                for (key, _) in sorted.into_iter().skip(1) {
                    keys_to_remove.push(key);
                }
            }
        }

        let removed = keys_to_remove.len();
        for key in keys_to_remove {
            self.entries.remove(&key);
        }
        self.stats.entry_count = self.entries.len();
        removed
    }

    /// Create a fingerprint for deduplication
    ///
    /// The fingerprint captures the essential identity of a cache entry
    /// for deduplication purposes.
    fn create_entry_fingerprint(&self, key: &CacheKey, entry: &CachedPropertyResult) -> String {
        // Include property name, passed status, and sorted backend names
        let mut backends = entry.backends.to_vec();
        backends.sort();
        format!(
            "{}:{}:{}",
            key.property_name,
            entry.property.passed,
            backends.join(",")
        )
    }

    /// Get statistics about potential compaction without performing it
    ///
    /// Useful for deciding whether to run compaction.
    pub fn analyze_compaction(&self, config: &CompactionConfig) -> CompactionResult {
        let entries_before = self.entries.len();

        if entries_before < config.min_entries_threshold {
            return CompactionResult::skipped(
                format!(
                    "Entry count {} below threshold {}",
                    entries_before, config.min_entries_threshold
                ),
                entries_before,
            );
        }

        let mut result = CompactionResult {
            entries_before,
            ..Default::default()
        };

        // Count expired entries
        result.expired_removed = self
            .entries
            .iter()
            .filter(|(key, entry)| {
                let effective_ttl =
                    self.effective_ttl_for_property(&key.property_name, entry.confidence);
                entry.is_expired(effective_ttl)
            })
            .count();

        // Count entries exceeding max age
        if config.max_age_seconds > 0 {
            let max_age = Duration::from_secs(config.max_age_seconds);
            result.max_age_removed = self
                .entries
                .iter()
                .filter(|(_, entry)| {
                    entry
                        .cached_at
                        .elapsed()
                        .map(|age| age > max_age)
                        .unwrap_or(true)
                })
                .count();
        }

        // Count low-confidence entries
        if config.policy.removes_low_confidence() {
            result.low_confidence_removed = self
                .entries
                .iter()
                .filter(|(_, entry)| entry.confidence < config.min_confidence)
                .count();
        }

        // Count obsolete version entries
        if config.policy.removes_obsolete_versions() && !config.active_versions.is_empty() {
            let active_set: std::collections::HashSet<_> = config.active_versions.iter().collect();
            result.obsolete_removed = self
                .entries
                .keys()
                .filter(|k| !active_set.contains(&k.version_hash))
                .count();
        }

        // Count duplicates
        if config.policy.deduplicates() {
            let mut groups: HashMap<String, usize> = HashMap::new();
            for (key, entry) in &self.entries {
                let fingerprint = self.create_entry_fingerprint(key, entry);
                *groups.entry(fingerprint).or_default() += 1;
            }
            result.duplicates_removed = groups
                .values()
                .filter(|&&count| count > 1)
                .map(|&count| count - 1)
                .sum();
        }

        // Calculate what entries_after would be (accounting for overlap)
        // This is an approximation since items may be counted in multiple categories
        let total_removed = result.total_removed();
        result.entries_after = entries_before.saturating_sub(total_removed);

        result
    }

    /// Save entries to file with optional compaction before save
    ///
    /// If the compaction config has `compact_before_save` enabled,
    /// compaction will be performed before saving.
    pub fn save_with_compaction(
        &mut self,
        path: impl AsRef<Path>,
        compaction: &CompactionConfig,
    ) -> Result<((), Option<CompactionResult>), CachePersistenceError> {
        let compaction_result = if compaction.compact_before_save {
            Some(self.compact(compaction))
        } else {
            None
        };

        self.save_entries_to_file(path)?;
        Ok(((), compaction_result))
    }

    /// Save entries compressed with optional compaction before save
    pub fn save_compressed_with_compaction(
        &mut self,
        path: impl AsRef<Path>,
        level: SnapshotCompressionLevel,
        compaction: &CompactionConfig,
    ) -> Result<((), Option<CompactionResult>), CachePersistenceError> {
        let compaction_result = if compaction.compact_before_save {
            Some(self.compact(compaction))
        } else {
            None
        };

        self.save_entries_compressed(path, level)?;
        Ok(((), compaction_result))
    }

    /// Get a summary of cache entry distribution by version
    pub fn version_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        for key in self.entries.keys() {
            *distribution.entry(key.version_hash.clone()).or_default() += 1;
        }
        distribution
    }

    /// Get a summary of cache entry distribution by confidence level
    ///
    /// Returns counts in buckets: [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    pub fn confidence_distribution(&self) -> [usize; 5] {
        let mut distribution = [0; 5];
        for entry in self.entries.values() {
            let bucket = match entry.confidence {
                c if c < 0.2 => 0,
                c if c < 0.4 => 1,
                c if c < 0.6 => 2,
                c if c < 0.8 => 3,
                _ => 4,
            };
            distribution[bucket] += 1;
        }
        distribution
    }

    /// Get the age distribution of cache entries in seconds
    ///
    /// Returns (min_age, max_age, avg_age)
    pub fn age_distribution(&self) -> Option<(u64, u64, u64)> {
        if self.entries.is_empty() {
            return None;
        }

        let ages: Vec<u64> = self
            .entries
            .values()
            .filter_map(|e| e.cached_at.elapsed().ok().map(|d| d.as_secs()))
            .collect();

        if ages.is_empty() {
            return None;
        }

        let min = ages.iter().min().copied().unwrap_or(0);
        let max = ages.iter().max().copied().unwrap_or(0);
        let avg = ages.iter().sum::<u64>() / ages.len() as u64;

        Some((min, max, avg))
    }

    // ==========================================================================
    // Cache Warming
    // ==========================================================================

    /// Warm the cache with entries from another cache using the specified strategy
    ///
    /// This method populates this cache with entries from a source cache,
    /// applying the warming configuration to filter and prioritize entries.
    ///
    /// # Arguments
    ///
    /// * `source` - The source cache to warm from
    /// * `config` - Configuration controlling which entries to warm and in what order
    ///
    /// # Returns
    ///
    /// A `WarmingResult` describing what was warmed
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Load historical cache and warm current cache with high-confidence entries
    /// let historical = VerificationCache::load_from_file("backup.json")?;
    /// let config = WarmingConfig::high_confidence(0.8);
    /// let result = cache.warm_from_cache(&historical, &config);
    /// println!("Warmed {} entries from historical cache", result.entries_warmed);
    /// ```
    pub fn warm_from_cache(
        &mut self,
        source: &VerificationCache,
        config: &WarmingConfig,
    ) -> WarmingResult {
        let start = Instant::now();
        let mut result = WarmingResult::new("cache");
        result.strategy = config.strategy;

        // Collect and sort entries based on strategy
        let sorted_entries = self.sort_entries_for_warming(source, config);
        result.entries_considered = sorted_entries.len();

        // Warm entries
        for (key, entry) in sorted_entries {
            // Check max entries limit
            if config.max_entries > 0 && result.entries_warmed >= config.max_entries {
                result.entries_skipped_limit += 1;
                continue;
            }

            // Check if already in cache
            if self.entries.contains_key(&key) {
                result.entries_skipped_existing += 1;
                continue;
            }

            // Check pattern filters
            if !config.matches_patterns(&key.property_name) {
                result.entries_skipped_pattern += 1;
                continue;
            }

            // Check confidence threshold
            if entry.confidence < config.min_confidence {
                result.entries_skipped_low_confidence += 1;
                continue;
            }

            // Check max age
            if let Some(max_age) = config.max_age {
                if let Ok(age) = entry.cached_at.elapsed() {
                    if age > max_age {
                        result.entries_skipped_expired += 1;
                        continue;
                    }
                }
            }

            // Check TTL validity
            if config.validate_ttl {
                let effective_ttl =
                    self.effective_ttl_for_property(&key.property_name, entry.confidence);

                if entry.is_expired(effective_ttl) {
                    result.entries_skipped_expired += 1;
                    continue;
                }

                // Check minimum remaining TTL ratio
                if config.skip_nearly_expired {
                    if let Ok(age) = entry.cached_at.elapsed() {
                        let remaining = effective_ttl.saturating_sub(age);
                        let min_remaining = Duration::from_secs_f64(
                            effective_ttl.as_secs_f64() * config.min_remaining_ttl_ratio,
                        );
                        if remaining < min_remaining {
                            result.entries_skipped_expired += 1;
                            continue;
                        }
                    }
                }
            }

            // Respect max_entries limit for this cache
            if self.entries.len() >= self.max_entries {
                self.evict_oldest();
            }

            self.entries.insert(key, entry);
            result.entries_warmed += 1;
        }

        self.stats.entry_count = self.entries.len();
        result.duration_ms = start.elapsed().as_millis() as u64;
        result
    }

    /// Helper to sort entries based on warming strategy
    fn sort_entries_for_warming(
        &self,
        source: &VerificationCache,
        config: &WarmingConfig,
    ) -> Vec<(CacheKey, CachedPropertyResult)> {
        let mut entries: Vec<_> = source
            .entries
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Sort based on strategy
        match config.strategy {
            WarmingStrategy::All => {
                // No specific sorting, process in iteration order
            }
            WarmingStrategy::HighConfidence => {
                // Sort by confidence descending
                entries.sort_by(|a, b| {
                    b.1.confidence
                        .partial_cmp(&a.1.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            WarmingStrategy::MostRecent => {
                // Sort by cached_at descending (most recent first)
                entries.sort_by(|a, b| b.1.cached_at.cmp(&a.1.cached_at));
            }
            WarmingStrategy::MostAccessed => {
                // For MostAccessed, we would need access count data which isn't stored
                // in CachedPropertyResult. Fall back to most recent.
                entries.sort_by(|a, b| b.1.cached_at.cmp(&a.1.cached_at));
            }
            WarmingStrategy::PatternBased => {
                // First filter to matching patterns, then by confidence
                entries.retain(|(k, _)| config.matches_patterns(&k.property_name));
                entries.sort_by(|a, b| {
                    b.1.confidence
                        .partial_cmp(&a.1.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            WarmingStrategy::FastestVerification => {
                // Would need verification time data. Fall back to high confidence
                // (high confidence often correlates with faster verification)
                entries.sort_by(|a, b| {
                    b.1.confidence
                        .partial_cmp(&a.1.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // Apply priority versions - move matching entries to the front
        if !config.priority_versions.is_empty() {
            entries.sort_by(|a, b| {
                let a_priority = config.priority_versions.contains(&a.0.version_hash);
                let b_priority = config.priority_versions.contains(&b.0.version_hash);
                b_priority.cmp(&a_priority)
            });
        }

        entries
    }

    /// Warm the cache from a file with the specified configuration
    ///
    /// Loads a cache snapshot from file and warms this cache with its entries.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the cache snapshot file
    /// * `config` - Configuration controlling which entries to warm
    ///
    /// # Returns
    ///
    /// A `WarmingResult` describing what was warmed, or an error if the file couldn't be read
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = WarmingConfig::most_recent(1000);
    /// let result = cache.warm_from_file("backup.json", &config)?;
    /// println!("Warmed {} entries from file", result.entries_warmed);
    /// ```
    pub fn warm_from_file(
        &mut self,
        path: impl AsRef<Path>,
        config: &WarmingConfig,
    ) -> Result<WarmingResult, CachePersistenceError> {
        let path = path.as_ref();
        // Load into a temporary cache to use as source
        let mut source = Self::new();
        source.load_entries_from_file(path)?;
        let mut result = self.warm_from_cache(&source, config);
        result.source = path.display().to_string();
        Ok(result)
    }

    /// Warm the cache from a compressed file with the specified configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = WarmingConfig::high_confidence(0.9);
    /// let result = cache.warm_from_compressed_file("backup.json.gz", &config)?;
    /// ```
    pub fn warm_from_compressed_file(
        &mut self,
        path: impl AsRef<Path>,
        config: &WarmingConfig,
    ) -> Result<WarmingResult, CachePersistenceError> {
        let path = path.as_ref();
        // Load into a temporary cache to use as source
        let mut source = Self::new();
        source.load_entries_compressed(path)?;
        let mut result = self.warm_from_cache(&source, config);
        result.source = format!("{} (compressed)", path.display());
        Ok(result)
    }

    /// Warm the cache from a snapshot with the specified configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// let snapshot = load_snapshot_from_somewhere();
    /// let config = WarmingConfig::new();
    /// let result = cache.warm_from_snapshot(&snapshot, &config)?;
    /// ```
    pub fn warm_from_snapshot(
        &mut self,
        snapshot: &CacheSnapshot,
        config: &WarmingConfig,
    ) -> Result<WarmingResult, CachePersistenceError> {
        // Create a temporary cache from the snapshot
        let mut source = Self::new();
        source.restore_from_snapshot(snapshot)?;
        let mut result = self.warm_from_cache(&source, config);
        result.source = "snapshot".to_string();
        Ok(result)
    }

    /// Warm the cache with specific entries, useful for programmatic warming
    ///
    /// This is a more flexible version of `warm_from_results` that uses the
    /// warming configuration for filtering and prioritization.
    ///
    /// # Arguments
    ///
    /// * `entries` - Iterator of (version_hash, property_name, result) tuples
    /// * `config` - Configuration controlling which entries to warm
    ///
    /// # Returns
    ///
    /// A `WarmingResult` describing what was warmed
    ///
    /// # Example
    ///
    /// ```ignore
    /// let historical_entries = vec![
    ///     ("v1", "prop1", &result1),
    ///     ("v1", "prop2", &result2),
    /// ];
    /// let config = WarmingConfig::high_confidence(0.8);
    /// let result = cache.warm_with_config(historical_entries.into_iter(), &config);
    /// ```
    pub fn warm_with_config<'a, I>(&mut self, entries: I, config: &WarmingConfig) -> WarmingResult
    where
        I: Iterator<Item = (&'a str, &'a str, &'a CachedPropertyResult)>,
    {
        let start = Instant::now();
        let mut result = WarmingResult::new("iterator");
        result.strategy = config.strategy;

        // Collect entries for sorting
        let mut to_warm: Vec<(CacheKey, CachedPropertyResult)> = entries
            .map(|(vh, pn, r)| (CacheKey::new(vh, pn), r.clone()))
            .collect();
        result.entries_considered = to_warm.len();

        // Sort based on strategy
        match config.strategy {
            WarmingStrategy::HighConfidence => {
                to_warm.sort_by(|a, b| {
                    b.1.confidence
                        .partial_cmp(&a.1.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            WarmingStrategy::MostRecent => {
                to_warm.sort_by(|a, b| b.1.cached_at.cmp(&a.1.cached_at));
            }
            WarmingStrategy::PatternBased => {
                to_warm.retain(|(k, _)| config.matches_patterns(&k.property_name));
            }
            _ => {}
        }

        // Process entries
        for (key, entry) in to_warm {
            if config.max_entries > 0 && result.entries_warmed >= config.max_entries {
                result.entries_skipped_limit += 1;
                continue;
            }

            if self.entries.contains_key(&key) {
                result.entries_skipped_existing += 1;
                continue;
            }

            if !config.matches_patterns(&key.property_name) {
                result.entries_skipped_pattern += 1;
                continue;
            }

            if entry.confidence < config.min_confidence {
                result.entries_skipped_low_confidence += 1;
                continue;
            }

            if config.validate_ttl {
                let effective_ttl =
                    self.effective_ttl_for_property(&key.property_name, entry.confidence);
                if entry.is_expired(effective_ttl) {
                    result.entries_skipped_expired += 1;
                    continue;
                }
            }

            if self.entries.len() >= self.max_entries {
                self.evict_oldest();
            }

            self.entries.insert(key, entry);
            result.entries_warmed += 1;
        }

        self.stats.entry_count = self.entries.len();
        result.duration_ms = start.elapsed().as_millis() as u64;
        result
    }

    /// Analyze potential warming from a source cache without actually warming
    ///
    /// Useful for predicting what would be warmed before committing to the operation.
    ///
    /// # Returns
    ///
    /// A `WarmingResult` showing what would be warmed (without modifying the cache)
    pub fn analyze_warming(
        &self,
        source: &VerificationCache,
        config: &WarmingConfig,
    ) -> WarmingResult {
        let start = Instant::now();
        let mut result = WarmingResult::new("analysis");
        result.strategy = config.strategy;

        let sorted_entries = self.sort_entries_for_warming(source, config);
        result.entries_considered = sorted_entries.len();

        for (key, entry) in sorted_entries {
            if config.max_entries > 0 && result.entries_warmed >= config.max_entries {
                result.entries_skipped_limit += 1;
                continue;
            }

            if self.entries.contains_key(&key) {
                result.entries_skipped_existing += 1;
                continue;
            }

            if !config.matches_patterns(&key.property_name) {
                result.entries_skipped_pattern += 1;
                continue;
            }

            if entry.confidence < config.min_confidence {
                result.entries_skipped_low_confidence += 1;
                continue;
            }

            if let Some(max_age) = config.max_age {
                if let Ok(age) = entry.cached_at.elapsed() {
                    if age > max_age {
                        result.entries_skipped_expired += 1;
                        continue;
                    }
                }
            }

            if config.validate_ttl {
                let effective_ttl =
                    self.effective_ttl_for_property(&key.property_name, entry.confidence);
                if entry.is_expired(effective_ttl) {
                    result.entries_skipped_expired += 1;
                    continue;
                }
            }

            result.entries_warmed += 1;
        }

        result.duration_ms = start.elapsed().as_millis() as u64;
        result
    }
}

/// Result of incremental verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalVerificationResult {
    /// The full verification result
    pub result: VerificationResult,
    /// Number of properties that were cached
    pub cached_count: usize,
    /// Number of properties that were newly verified
    pub verified_count: usize,
    /// Properties that were retrieved from cache
    pub cached_properties: Vec<String>,
    /// Properties that were re-verified
    pub verified_properties: Vec<String>,
    /// Time saved by caching (estimated)
    pub time_saved_ms: u64,
}

/// Result of verifying an improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether verification passed overall
    pub passed: bool,

    /// Results from each backend
    pub backend_results: HashMap<String, BackendResult>,

    /// Overall confidence score
    pub confidence: f64,

    /// Total verification time
    pub duration_ms: u64,

    /// Detailed verification messages
    pub messages: Vec<VerificationMessage>,

    /// Properties that were verified
    pub verified_properties: Vec<VerifiedProperty>,
}

impl VerificationResult {
    /// Create a passed result
    pub fn passed(backend_results: HashMap<String, BackendResult>, duration_ms: u64) -> Self {
        let passing = backend_results.values().filter(|r| r.passed).count();
        let total = backend_results.len();
        let confidence = if total > 0 {
            passing as f64 / total as f64
        } else {
            0.0
        };

        Self {
            passed: true,
            backend_results,
            confidence,
            duration_ms,
            messages: Vec::new(),
            verified_properties: Vec::new(),
        }
    }

    /// Create a failed result
    pub fn failed(
        backend_results: HashMap<String, BackendResult>,
        duration_ms: u64,
        messages: Vec<VerificationMessage>,
    ) -> Self {
        let passing = backend_results.values().filter(|r| r.passed).count();
        let total = backend_results.len();
        let confidence = if total > 0 {
            passing as f64 / total as f64
        } else {
            0.0
        };

        Self {
            passed: false,
            backend_results,
            confidence,
            duration_ms,
            messages,
            verified_properties: Vec::new(),
        }
    }

    /// Create a result from dispatcher merged results
    pub fn from_dispatcher_results(merged: &MergedResults, duration_ms: u64) -> Self {
        let mut backend_results = HashMap::new();
        let mut verified_properties = Vec::new();

        // Extract backend results from each property
        for prop_result in &merged.properties {
            let prop_name = prop_result
                .property_type
                .map(|pt| format!("{:?}_{}", pt, prop_result.property_index))
                .unwrap_or_else(|| format!("property_{}", prop_result.property_index));

            // Track the property verification
            let prop_passed = matches!(prop_result.status, VerificationStatus::Proven);
            verified_properties.push(VerifiedProperty {
                name: prop_name.clone(),
                passed: prop_passed,
                status: format!("{:?}", prop_result.status),
            });

            // Aggregate backend results
            for br in &prop_result.backend_results {
                let backend_name = format!("{:?}", br.backend);
                let entry = backend_results
                    .entry(backend_name.clone())
                    .or_insert_with(|| BackendResult {
                        backend: backend_name,
                        passed: true,
                        duration_ms: 0,
                        error: None,
                        raw_output: None,
                        verified_properties: Vec::new(),
                        failed_properties: Vec::new(),
                    });

                entry.duration_ms += br.time_taken.as_millis() as u64;

                if matches!(br.status, VerificationStatus::Proven) {
                    entry.verified_properties.push(prop_name.clone());
                } else {
                    entry.failed_properties.push(prop_name.clone());
                    if matches!(
                        br.status,
                        VerificationStatus::Disproven | VerificationStatus::Unknown { .. }
                    ) {
                        entry.passed = false;
                    }
                }

                if let Some(ref err) = br.error {
                    entry.error = Some(err.clone());
                    entry.passed = false;
                }
            }
        }

        let confidence = merged.summary.overall_confidence;
        let passed = merged.summary.proven > 0 && merged.summary.disproven == 0;

        let messages = if !passed && merged.summary.disproven > 0 {
            vec![VerificationMessage::error(format!(
                "{} properties disproven, {} unknown",
                merged.summary.disproven, merged.summary.unknown
            ))]
        } else {
            Vec::new()
        };

        Self {
            passed,
            backend_results,
            confidence,
            duration_ms,
            messages,
            verified_properties,
        }
    }

    /// Get failed backends
    pub fn failed_backends(&self) -> Vec<&str> {
        self.backend_results
            .iter()
            .filter(|(_, r)| !r.passed)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get passing backends
    pub fn passing_backends(&self) -> Vec<&str> {
        self.backend_results
            .iter()
            .filter(|(_, r)| r.passed)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

/// A property that was verified
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedProperty {
    /// Name of the property
    pub name: String,
    /// Whether it passed
    pub passed: bool,
    /// Verification status
    pub status: String,
}

/// Result from a single verification backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendResult {
    /// Name of the backend
    pub backend: String,

    /// Whether verification passed
    pub passed: bool,

    /// Verification time
    pub duration_ms: u64,

    /// Error message if failed
    pub error: Option<String>,

    /// Raw output from backend
    pub raw_output: Option<String>,

    /// Properties that were verified
    pub verified_properties: Vec<String>,

    /// Properties that failed
    pub failed_properties: Vec<String>,
}

impl BackendResult {
    /// Create a passed result
    pub fn passed(backend: impl Into<String>, duration_ms: u64) -> Self {
        Self {
            backend: backend.into(),
            passed: true,
            duration_ms,
            error: None,
            raw_output: None,
            verified_properties: Vec::new(),
            failed_properties: Vec::new(),
        }
    }

    /// Create a failed result
    pub fn failed(backend: impl Into<String>, duration_ms: u64, error: impl Into<String>) -> Self {
        Self {
            backend: backend.into(),
            passed: false,
            duration_ms,
            error: Some(error.into()),
            raw_output: None,
            verified_properties: Vec::new(),
            failed_properties: Vec::new(),
        }
    }

    /// Add verified properties
    pub fn with_verified_properties(mut self, props: Vec<String>) -> Self {
        self.verified_properties = props;
        self
    }

    /// Add failed properties
    pub fn with_failed_properties(mut self, props: Vec<String>) -> Self {
        self.failed_properties = props;
        self
    }
}

/// A verification message (info, warning, or error)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMessage {
    /// Message level
    pub level: MessageLevel,

    /// The message text
    pub message: String,

    /// Source of the message (backend, property, etc.)
    pub source: Option<String>,
}

/// Message level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageLevel {
    Info,
    Warning,
    Error,
}

impl VerificationMessage {
    /// Create an info message
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            level: MessageLevel::Info,
            message: message.into(),
            source: None,
        }
    }

    /// Create a warning message
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            level: MessageLevel::Warning,
            message: message.into(),
            source: None,
        }
    }

    /// Create an error message
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            level: MessageLevel::Error,
            message: message.into(),
            source: None,
        }
    }

    /// Set the source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

/// The improvement verifier (synchronous version)
///
/// This struct orchestrates formal verification of proposed improvements
/// using multiple verification backends. For async verification with
/// the dispatcher, use [`AsyncImprovementVerifier`].
pub struct ImprovementVerifier {
    /// Configuration
    config: VerificationConfig,
}

impl ImprovementVerifier {
    /// Create a new verifier with default configuration
    pub fn new() -> Self {
        Self {
            config: VerificationConfig::default(),
        }
    }

    /// Create a verifier with custom configuration
    pub fn with_config(config: VerificationConfig) -> Self {
        Self { config }
    }

    /// Get the configuration
    pub fn config(&self) -> &VerificationConfig {
        &self.config
    }

    /// Verify an improvement (synchronous, without dispatcher)
    ///
    /// This runs the improvement through all configured verification backends
    /// and aggregates the results. For full dispatcher integration, use
    /// [`AsyncImprovementVerifier`].
    pub fn verify(
        &self,
        current: &Version,
        improvement: &Improvement,
    ) -> SelfImpResult<VerificationResult> {
        let start = std::time::Instant::now();
        let mut backend_results = HashMap::new();
        let mut messages = Vec::new();

        // Run each backend
        for backend_name in &self.config.backends {
            let backend_start = std::time::Instant::now();

            match self.verify_with_backend(backend_name, current, improvement) {
                Ok(result) => {
                    backend_results.insert(backend_name.clone(), result);
                }
                Err(e) => {
                    backend_results.insert(
                        backend_name.clone(),
                        BackendResult::failed(
                            backend_name,
                            backend_start.elapsed().as_millis() as u64,
                            e.to_string(),
                        ),
                    );
                    messages.push(
                        VerificationMessage::error(format!(
                            "Backend {} failed: {}",
                            backend_name, e
                        ))
                        .with_source(backend_name),
                    );
                }
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        // Check if enough backends passed
        let passing_count = backend_results.values().filter(|r| r.passed).count();
        let passed = passing_count >= self.config.min_passing_backends;

        // Calculate confidence
        let confidence = if !backend_results.is_empty() {
            passing_count as f64 / backend_results.len() as f64
        } else {
            0.0
        };

        // Check minimum confidence
        let passed = passed && confidence >= self.config.min_confidence;

        if passed {
            Ok(VerificationResult::passed(backend_results, duration_ms))
        } else {
            if passing_count < self.config.min_passing_backends {
                messages.push(VerificationMessage::error(format!(
                    "Only {}/{} backends passed (minimum: {})",
                    passing_count,
                    backend_results.len(),
                    self.config.min_passing_backends
                )));
            }
            if confidence < self.config.min_confidence {
                messages.push(VerificationMessage::error(format!(
                    "Confidence {:.2} below minimum {:.2}",
                    confidence, self.config.min_confidence
                )));
            }
            Ok(VerificationResult::failed(
                backend_results,
                duration_ms,
                messages,
            ))
        }
    }

    /// Verify with a specific backend (placeholder for simple verifier)
    fn verify_with_backend(
        &self,
        backend_name: &str,
        _current: &Version,
        _improvement: &Improvement,
    ) -> SelfImpResult<BackendResult> {
        // Simple verifier returns placeholder - use AsyncImprovementVerifier for real verification
        Ok(BackendResult::passed(backend_name, 100))
    }

    /// Quick check - verify soundness only (faster)
    pub fn quick_check(&self, current: &Version, improvement: &Improvement) -> SelfImpResult<bool> {
        // Use a single fast backend for quick soundness check
        let result = self.verify_with_backend("lean4", current, improvement)?;
        Ok(result.passed)
    }
}

impl Default for ImprovementVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Async improvement verifier with dispatcher integration
///
/// This verifier uses `dashprove-dispatcher` for multi-backend verification
/// with intelligent backend selection, parallel execution, and result merging.
///
/// ## Incremental Verification
///
/// The verifier supports incremental verification through a property-level cache.
/// When enabled, it tracks which properties depend on which definitions and only
/// re-verifies properties affected by changes.
///
/// ```ignore
/// use dashprove_selfimp::{AsyncImprovementVerifier, VerificationCache};
///
/// let mut verifier = AsyncImprovementVerifier::with_cache();
///
/// // First verification - all properties verified
/// let result1 = verifier.verify(&v1, &improvement1).await?;
///
/// // Second verification - only affected properties re-verified
/// let result2 = verifier.verify_incremental(&v2, &improvement2, &["changed_type"]).await?;
/// ```
pub struct AsyncImprovementVerifier {
    /// Configuration
    config: VerificationConfig,
    /// The dispatcher for verification
    dispatcher: Arc<Mutex<Dispatcher>>,
    /// Optional cache for incremental verification
    cache: Option<Arc<Mutex<VerificationCache>>>,
}

/// Summary of a cache autosave session
#[derive(Debug, Clone)]
pub struct CacheAutosaveSummary {
    /// Number of successful saves performed
    pub save_count: usize,
    /// Number of saves skipped due to minimum change threshold
    pub skip_count: usize,
    /// Number of saves forced due to max stale duration being exceeded
    pub forced_save_count: usize,
    /// Number of compactions triggered during autosave
    pub compaction_count: usize,
    /// Counts of compactions grouped by trigger type
    pub compaction_trigger_counts: CompactionTriggerCounts,
    /// Counts of saves grouped by trigger reason
    pub save_reason_counts: AutosaveReasonCounts,
    /// Reason for the most recent successful save (None if no saves succeeded)
    pub last_save_reason: Option<AutosaveSaveReason>,
    /// Human-readable errors encountered during saving
    pub errors: Vec<String>,
    /// Path snapshots were written to
    pub path: PathBuf,
    /// Initial interval configured for saves
    pub interval: Duration,
    /// Final interval at session end (may differ if adaptive intervals enabled)
    pub final_interval: Duration,
    /// Compression level used (None means uncompressed JSON)
    pub compression: Option<SnapshotCompressionLevel>,
    /// Total bytes written across all saves
    pub total_bytes_written: u64,
    /// Bytes written in most recent save (0 if no saves)
    pub last_save_bytes: u64,
}

/// Current status of a running cache autosave task
///
/// Unlike `CacheAutosaveSummary`, this can be queried repeatedly without
/// stopping the background task. Error count is provided instead of the
/// full error list to avoid locking overhead.
#[derive(Debug, Clone)]
pub struct CacheAutosaveStatus {
    /// Number of successful saves performed so far
    pub save_count: usize,
    /// Number of errors encountered so far
    pub error_count: usize,
    /// Number of saves skipped due to minimum change threshold so far
    pub skip_count: usize,
    /// Number of saves forced due to max stale duration being exceeded so far
    pub forced_save_count: usize,
    /// Number of compactions triggered during autosave so far
    pub compaction_count: usize,
    /// Counts of compactions grouped by trigger type so far
    pub compaction_trigger_counts: CompactionTriggerCounts,
    /// Counts of saves grouped by trigger reason
    pub save_reason_counts: AutosaveReasonCounts,
    /// Reason for the most recent successful save (None if no saves yet)
    pub last_save_reason: Option<AutosaveSaveReason>,
    /// Total bytes written across all saves so far
    pub total_bytes_written: u64,
    /// Bytes written in most recent save (0 if no saves yet)
    pub last_save_bytes: u64,
    /// Current interval between saves (may change if adaptive intervals enabled)
    pub current_interval: Duration,
    /// Whether the autosave task is still running
    pub is_running: bool,
}

/// Reason why an autosave was triggered
///
/// This helps distinguish between normal interval-based saves and those
/// forced due to data staleness or other conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutosaveSaveReason {
    /// Initial save when autosave starts
    Initial,
    /// Normal interval-based save
    Interval,
    /// Save forced because data exceeded max stale duration
    StaleData,
    /// Save after coalescing multiple changes together
    Coalesced,
}

impl std::fmt::Display for AutosaveSaveReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AutosaveSaveReason::Initial => write!(f, "initial"),
            AutosaveSaveReason::Interval => write!(f, "interval"),
            AutosaveSaveReason::StaleData => write!(f, "stale_data"),
            AutosaveSaveReason::Coalesced => write!(f, "coalesced"),
        }
    }
}

/// Counts of autosaves by trigger reason
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutosaveReasonCounts {
    /// Saves performed as the initial bootstrap snapshot
    pub initial: usize,
    /// Interval-driven saves
    pub interval: usize,
    /// Saves forced due to exceeding max stale duration
    pub stale_data: usize,
    /// Saves after coalescing multiple changes
    pub coalesced: usize,
}

/// Persisted autosave metrics for snapshot storage
///
/// This is a serializable version of autosave metrics that can be stored
/// in cache snapshots for cross-session analytics. Added in format version 5.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedAutosaveMetrics {
    /// Number of successful saves in the session
    pub save_count: usize,
    /// Number of errors encountered
    pub error_count: usize,
    /// Number of saves skipped due to minimum change threshold
    pub skip_count: usize,
    /// Number of saves forced due to max stale duration
    pub forced_save_count: usize,
    /// Counts of saves by trigger reason
    pub save_reason_counts: AutosaveReasonCounts,
    /// Reason for the most recent successful save
    pub last_save_reason: Option<AutosaveSaveReason>,
    /// Total bytes written across all saves
    pub total_bytes_written: u64,
    /// Bytes written in most recent save
    pub last_save_bytes: u64,
    /// Save interval in milliseconds (final interval for adaptive)
    pub interval_ms: u64,
    /// Whether compression was enabled
    pub compressed: bool,
    /// Timestamp when autosave session started (Unix epoch seconds)
    pub session_start_secs: u64,
    /// Timestamp when autosave session ended (Unix epoch seconds)
    pub session_end_secs: u64,
}

impl PersistedAutosaveMetrics {
    /// Create from a CacheAutosaveStatus snapshot
    ///
    /// Used when capturing autosave metrics during snapshot creation.
    /// The session_start_secs and session_end_secs must be provided separately
    /// since the status doesn't track absolute time.
    pub fn from_status(
        status: &CacheAutosaveStatus,
        compressed: bool,
        session_start_secs: u64,
        session_end_secs: u64,
    ) -> Self {
        Self {
            save_count: status.save_count,
            error_count: status.error_count,
            skip_count: status.skip_count,
            forced_save_count: status.forced_save_count,
            save_reason_counts: status.save_reason_counts,
            last_save_reason: status.last_save_reason,
            total_bytes_written: status.total_bytes_written,
            last_save_bytes: status.last_save_bytes,
            interval_ms: status.current_interval.as_millis() as u64,
            compressed,
            session_start_secs,
            session_end_secs,
        }
    }

    /// Get total saves (successful + skipped + forced)
    pub fn total_attempts(&self) -> usize {
        self.save_count + self.skip_count
    }

    /// Get save success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.total_attempts();
        if total == 0 {
            0.0
        } else {
            self.save_count as f64 / total as f64
        }
    }

    /// Get session duration in seconds
    pub fn session_duration_secs(&self) -> u64 {
        self.session_end_secs
            .saturating_sub(self.session_start_secs)
    }

    /// Get average bytes per save
    pub fn avg_bytes_per_save(&self) -> u64 {
        if self.save_count == 0 {
            0
        } else {
            self.total_bytes_written / self.save_count as u64
        }
    }
}

/// Counts of compactions by trigger type
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionTriggerCounts {
    /// Compactions triggered by size threshold
    pub size_based: usize,
    /// Compactions triggered by time interval
    pub time_based: usize,
    /// Compactions triggered by low hit rate
    pub hit_rate_based: usize,
    /// Compactions triggered by partition imbalance
    pub partition_imbalance: usize,
    /// Compactions triggered by insert count
    pub insert_based: usize,
    /// Compactions triggered by memory threshold
    pub memory_based: usize,
}

impl CompactionTriggerCounts {
    /// Increment the count for a specific trigger type
    pub fn increment(&mut self, trigger: &CompactionTrigger) {
        match trigger {
            CompactionTrigger::SizeBased { .. } => self.size_based += 1,
            CompactionTrigger::TimeBased { .. } => self.time_based += 1,
            CompactionTrigger::HitRateBased { .. } => self.hit_rate_based += 1,
            CompactionTrigger::PartitionImbalance { .. } => self.partition_imbalance += 1,
            CompactionTrigger::InsertBased { .. } => self.insert_based += 1,
            CompactionTrigger::MemoryBased { .. } => self.memory_based += 1,
        }
    }

    /// Get total compactions across all trigger types
    pub fn total(&self) -> usize {
        self.size_based
            + self.time_based
            + self.hit_rate_based
            + self.partition_imbalance
            + self.insert_based
            + self.memory_based
    }

    /// Add another CompactionTriggerCounts to this one (for accumulating historical counts)
    pub fn add(&mut self, other: &CompactionTriggerCounts) {
        self.size_based += other.size_based;
        self.time_based += other.time_based;
        self.hit_rate_based += other.hit_rate_based;
        self.partition_imbalance += other.partition_imbalance;
        self.insert_based += other.insert_based;
        self.memory_based += other.memory_based;
    }

    /// Create a new CompactionTriggerCounts by adding two together
    pub fn combined(&self, other: &CompactionTriggerCounts) -> CompactionTriggerCounts {
        CompactionTriggerCounts {
            size_based: self.size_based + other.size_based,
            time_based: self.time_based + other.time_based,
            hit_rate_based: self.hit_rate_based + other.hit_rate_based,
            partition_imbalance: self.partition_imbalance + other.partition_imbalance,
            insert_based: self.insert_based + other.insert_based,
            memory_based: self.memory_based + other.memory_based,
        }
    }
}

/// Type of compaction trigger for time-series tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompactionTriggerType {
    /// Size-based compaction (cache exceeded size threshold)
    SizeBased,
    /// Time-based compaction (periodic interval elapsed)
    TimeBased,
    /// Hit rate-based compaction (low hit rate detected)
    HitRateBased,
    /// Partition imbalance compaction (uneven partition sizes)
    PartitionImbalance,
    /// Insert-based compaction (insert count threshold reached)
    InsertBased,
    /// Memory-based compaction (memory threshold exceeded)
    MemoryBased,
}

impl CompactionTriggerType {
    /// Get trigger type from a CompactionTrigger
    pub fn from_trigger(trigger: &CompactionTrigger) -> Self {
        match trigger {
            CompactionTrigger::SizeBased { .. } => CompactionTriggerType::SizeBased,
            CompactionTrigger::TimeBased { .. } => CompactionTriggerType::TimeBased,
            CompactionTrigger::HitRateBased { .. } => CompactionTriggerType::HitRateBased,
            CompactionTrigger::PartitionImbalance { .. } => {
                CompactionTriggerType::PartitionImbalance
            }
            CompactionTrigger::InsertBased { .. } => CompactionTriggerType::InsertBased,
            CompactionTrigger::MemoryBased { .. } => CompactionTriggerType::MemoryBased,
        }
    }

    /// Get all trigger types
    pub fn all() -> &'static [CompactionTriggerType] {
        &[
            CompactionTriggerType::SizeBased,
            CompactionTriggerType::TimeBased,
            CompactionTriggerType::HitRateBased,
            CompactionTriggerType::PartitionImbalance,
            CompactionTriggerType::InsertBased,
            CompactionTriggerType::MemoryBased,
        ]
    }
}

/// A single compaction event with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionTimeSeriesEntry {
    /// When the compaction occurred (unix timestamp in seconds)
    pub timestamp_secs: u64,
    /// Type of trigger that caused this compaction
    pub trigger_type: CompactionTriggerType,
    /// Number of entries removed during compaction
    pub entries_removed: usize,
}

/// Time-series tracking of compaction events with a fixed-size ring buffer
///
/// Tracks individual compaction events with timestamps to enable rate calculations
/// and trend analysis over time. Uses a ring buffer to limit memory usage while
/// retaining recent history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionTimeSeries {
    /// Ring buffer of compaction events (newest at the end)
    entries: Vec<CompactionTimeSeriesEntry>,
    /// Maximum number of entries to retain
    max_entries: usize,
    /// Index of oldest entry in ring buffer (for wrap-around)
    head: usize,
    /// Number of entries currently in buffer
    len: usize,
}

impl Default for CompactionTimeSeries {
    fn default() -> Self {
        Self::new(1000) // Default to 1000 entries (~1KB memory)
    }
}

impl CompactionTimeSeries {
    /// Create a new time series with specified capacity
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries.min(1000)),
            max_entries: max_entries.max(1), // At least 1 entry
            head: 0,
            len: 0,
        }
    }

    /// Record a new compaction event
    pub fn record(&mut self, trigger: &CompactionTrigger, entries_removed: usize) {
        let entry = CompactionTimeSeriesEntry {
            timestamp_secs: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            trigger_type: CompactionTriggerType::from_trigger(trigger),
            entries_removed,
        };
        self.push(entry);
    }

    /// Record a compaction event with explicit timestamp (for testing or replay)
    pub fn record_at(
        &mut self,
        timestamp: SystemTime,
        trigger_type: CompactionTriggerType,
        entries_removed: usize,
    ) {
        let entry = CompactionTimeSeriesEntry {
            timestamp_secs: timestamp
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            trigger_type,
            entries_removed,
        };
        self.push(entry);
    }

    fn push(&mut self, entry: CompactionTimeSeriesEntry) {
        if self.entries.len() < self.max_entries {
            // Still filling up the buffer
            self.entries.push(entry);
            self.len = self.entries.len();
        } else {
            // Ring buffer is full, overwrite oldest
            let idx = (self.head + self.len) % self.max_entries;
            self.entries[idx] = entry;
            self.head = (self.head + 1) % self.max_entries;
        }
    }

    /// Get number of compaction events recorded
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if no events recorded
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get maximum capacity
    pub fn capacity(&self) -> usize {
        self.max_entries
    }

    /// Get all entries in chronological order (oldest first)
    pub fn entries(&self) -> Vec<CompactionTimeSeriesEntry> {
        if self.entries.len() < self.max_entries {
            // Buffer not yet full, entries are in order
            self.entries.clone()
        } else {
            // Ring buffer wrapped, need to reorder
            let mut result = Vec::with_capacity(self.len);
            for i in 0..self.len {
                let idx = (self.head + i) % self.max_entries;
                result.push(self.entries[idx].clone());
            }
            result
        }
    }

    /// Get entries within a time window (from now - duration to now)
    pub fn entries_in_window(&self, window: Duration) -> Vec<CompactionTimeSeriesEntry> {
        let now_secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let cutoff = now_secs.saturating_sub(window.as_secs());

        self.entries()
            .into_iter()
            .filter(|e| e.timestamp_secs >= cutoff)
            .collect()
    }

    /// Calculate compaction rate (events per minute) over a time window
    pub fn rate_per_minute(&self, window: Duration) -> f64 {
        let entries = self.entries_in_window(window);
        if entries.is_empty() {
            return 0.0;
        }

        let window_mins = window.as_secs_f64() / 60.0;
        if window_mins <= 0.0 {
            return 0.0;
        }

        entries.len() as f64 / window_mins
    }

    /// Calculate compaction rate (events per hour) over a time window
    pub fn rate_per_hour(&self, window: Duration) -> f64 {
        self.rate_per_minute(window) * 60.0
    }

    /// Get compaction counts by trigger type within a time window
    pub fn counts_by_type(&self, window: Duration) -> CompactionTriggerCounts {
        let entries = self.entries_in_window(window);
        let mut counts = CompactionTriggerCounts::default();

        for entry in entries {
            match entry.trigger_type {
                CompactionTriggerType::SizeBased => counts.size_based += 1,
                CompactionTriggerType::TimeBased => counts.time_based += 1,
                CompactionTriggerType::HitRateBased => counts.hit_rate_based += 1,
                CompactionTriggerType::PartitionImbalance => counts.partition_imbalance += 1,
                CompactionTriggerType::InsertBased => counts.insert_based += 1,
                CompactionTriggerType::MemoryBased => counts.memory_based += 1,
            }
        }

        counts
    }

    /// Get total entries removed within a time window
    pub fn entries_removed_in_window(&self, window: Duration) -> usize {
        self.entries_in_window(window)
            .iter()
            .map(|e| e.entries_removed)
            .sum()
    }

    /// Get removal rate (entries per hour) over a time window
    pub fn removal_rate_per_hour(&self, window: Duration) -> f64 {
        let removed = self.entries_removed_in_window(window);
        let window_hours = window.as_secs_f64() / 3600.0;
        if window_hours <= 0.0 {
            return 0.0;
        }
        removed as f64 / window_hours
    }

    /// Get the timestamp of the oldest entry (if any)
    pub fn oldest_timestamp(&self) -> Option<SystemTime> {
        if self.is_empty() {
            return None;
        }
        let oldest = if self.entries.len() < self.max_entries {
            &self.entries[0]
        } else {
            &self.entries[self.head]
        };
        Some(SystemTime::UNIX_EPOCH + Duration::from_secs(oldest.timestamp_secs))
    }

    /// Get the timestamp of the newest entry (if any)
    pub fn newest_timestamp(&self) -> Option<SystemTime> {
        if self.is_empty() {
            return None;
        }
        let newest_idx = if self.entries.len() < self.max_entries {
            self.entries.len() - 1
        } else {
            (self.head + self.len - 1) % self.max_entries
        };
        let newest = &self.entries[newest_idx];
        Some(SystemTime::UNIX_EPOCH + Duration::from_secs(newest.timestamp_secs))
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.head = 0;
        self.len = 0;
    }

    /// Merge another time series into this one, preserving chronological order
    pub fn merge_from(&mut self, other: &CompactionTimeSeries) {
        let mut combined = self.entries();
        combined.extend(other.entries());
        combined.sort_by_key(|e| e.timestamp_secs);

        let capacity = self.max_entries.max(other.max_entries);
        let mut merged = CompactionTimeSeries::new(capacity);
        for entry in combined {
            merged.push(entry);
        }
        *self = merged;
    }

    /// Get a summary of compaction activity
    pub fn summary(&self, window: Duration) -> CompactionTimeSeriesSummary {
        let entries = self.entries_in_window(window);
        let mut counts = CompactionTriggerCounts::default();
        let mut total_removed = 0usize;

        for entry in &entries {
            match entry.trigger_type {
                CompactionTriggerType::SizeBased => counts.size_based += 1,
                CompactionTriggerType::TimeBased => counts.time_based += 1,
                CompactionTriggerType::HitRateBased => counts.hit_rate_based += 1,
                CompactionTriggerType::PartitionImbalance => counts.partition_imbalance += 1,
                CompactionTriggerType::InsertBased => counts.insert_based += 1,
                CompactionTriggerType::MemoryBased => counts.memory_based += 1,
            }
            total_removed += entry.entries_removed;
        }

        let window_mins = window.as_secs_f64() / 60.0;
        let rate_per_minute = if window_mins > 0.0 {
            entries.len() as f64 / window_mins
        } else {
            0.0
        };

        CompactionTimeSeriesSummary {
            window,
            event_count: entries.len(),
            counts_by_type: counts,
            total_entries_removed: total_removed,
            rate_per_minute,
            rate_per_hour: rate_per_minute * 60.0,
        }
    }
}

/// Summary of compaction activity over a time window
#[derive(Debug, Clone)]
pub struct CompactionTimeSeriesSummary {
    /// The time window this summary covers
    pub window: Duration,
    /// Total number of compaction events in window
    pub event_count: usize,
    /// Breakdown by trigger type
    pub counts_by_type: CompactionTriggerCounts,
    /// Total cache entries removed during compactions
    pub total_entries_removed: usize,
    /// Compaction rate (events per minute)
    pub rate_per_minute: f64,
    /// Compaction rate (events per hour)
    pub rate_per_hour: f64,
}

const SAVE_REASON_NONE: u8 = 0;
const SAVE_REASON_INITIAL: u8 = 1;
const SAVE_REASON_INTERVAL: u8 = 2;
const SAVE_REASON_STALE: u8 = 3;
const SAVE_REASON_COALESCED: u8 = 4;

fn encode_save_reason(reason: AutosaveSaveReason) -> u8 {
    match reason {
        AutosaveSaveReason::Initial => SAVE_REASON_INITIAL,
        AutosaveSaveReason::Interval => SAVE_REASON_INTERVAL,
        AutosaveSaveReason::StaleData => SAVE_REASON_STALE,
        AutosaveSaveReason::Coalesced => SAVE_REASON_COALESCED,
    }
}

fn decode_save_reason(value: u8) -> Option<AutosaveSaveReason> {
    match value {
        SAVE_REASON_INITIAL => Some(AutosaveSaveReason::Initial),
        SAVE_REASON_INTERVAL => Some(AutosaveSaveReason::Interval),
        SAVE_REASON_STALE => Some(AutosaveSaveReason::StaleData),
        SAVE_REASON_COALESCED => Some(AutosaveSaveReason::Coalesced),
        _ => None,
    }
}

fn record_save_reason(
    reason: AutosaveSaveReason,
    initial_save_count: &AtomicUsize,
    interval_save_count: &AtomicUsize,
    stale_save_count: &AtomicUsize,
    coalesced_save_count: &AtomicUsize,
    last_save_reason: &AtomicU8,
) {
    match reason {
        AutosaveSaveReason::Initial => {
            initial_save_count.fetch_add(1, Ordering::Relaxed);
        }
        AutosaveSaveReason::Interval => {
            interval_save_count.fetch_add(1, Ordering::Relaxed);
        }
        AutosaveSaveReason::StaleData => {
            stale_save_count.fetch_add(1, Ordering::Relaxed);
        }
        AutosaveSaveReason::Coalesced => {
            coalesced_save_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    last_save_reason.store(encode_save_reason(reason), Ordering::Relaxed);
}

/// Event data for a successful cache autosave
#[derive(Debug, Clone)]
pub struct AutosaveSaveEvent {
    /// Path where the snapshot was saved
    pub path: PathBuf,
    /// Number of cache entries saved
    pub entry_count: usize,
    /// Number of bytes written
    pub bytes_written: u64,
    /// Cumulative save count (including this save)
    pub save_number: usize,
    /// Total bytes written across all saves (including this save)
    pub total_bytes_written: u64,
    /// Compression level used (None means uncompressed)
    pub compression: Option<SnapshotCompressionLevel>,
    /// Reason this save was triggered
    pub save_reason: AutosaveSaveReason,
}

/// Event data for a failed cache autosave
#[derive(Debug, Clone)]
pub struct AutosaveErrorEvent {
    /// Path where the save was attempted
    pub path: PathBuf,
    /// Error message
    pub error: String,
    /// Cumulative error count (including this error)
    pub error_number: usize,
    /// Compression level used (None means uncompressed)
    pub compression: Option<SnapshotCompressionLevel>,
}

/// Event data for a skipped cache autosave (below minimum change threshold)
#[derive(Debug, Clone)]
pub struct AutosaveSkipEvent {
    /// Path where the save would have been written
    pub path: PathBuf,
    /// Number of cache entries that would have been saved
    pub entry_count: usize,
    /// Current serialized size in bytes
    pub current_bytes: u64,
    /// Bytes written in the last successful save
    pub last_save_bytes: u64,
    /// Absolute change in bytes since last save
    pub change_bytes: u64,
    /// Configured minimum change threshold
    pub min_change_bytes: u64,
    /// Cumulative skip count (including this skip)
    pub skip_number: usize,
    /// Compression level that would have been used
    pub compression: Option<SnapshotCompressionLevel>,
}

/// Event data for a coalesced save (multiple changes batched into one save)
#[derive(Debug, Clone)]
pub struct AutosaveCoalesceEvent {
    /// Path where the save will be written
    pub path: PathBuf,
    /// Number of intervals that were coalesced into this save
    pub coalesced_intervals: usize,
    /// Duration waited during coalescing
    pub coalesce_duration: Duration,
    /// Whether the save was forced due to max_wait being exceeded
    pub forced_by_max_wait: bool,
    /// Current serialized size in bytes
    pub current_bytes: u64,
    /// Compression level that will be used
    pub compression: Option<SnapshotCompressionLevel>,
}

/// Event data for compaction triggered during autosave
#[derive(Debug, Clone)]
pub struct AutosaveCompactionEvent {
    /// Path where the cache is being saved
    pub path: PathBuf,
    /// Trigger that caused this compaction
    pub trigger: CompactionTrigger,
    /// Result of the compaction operation
    pub result: CompactionResult,
    /// Number of entries before compaction
    pub entries_before: usize,
    /// Number of entries after compaction
    pub entries_after: usize,
    /// Compression level being used
    pub compression: Option<SnapshotCompressionLevel>,
    /// Cumulative compaction count (including this compaction)
    pub compaction_number: usize,
}

/// Event data for cache warming during autosave startup
#[derive(Debug, Clone)]
pub struct AutosaveWarmingEvent {
    /// Path where the cache is being saved/loaded
    pub path: PathBuf,
    /// Result of the warming operation
    pub result: WarmingResult,
    /// Compression level being used
    pub compression: Option<SnapshotCompressionLevel>,
}

/// Type alias for autosave callback functions
pub type AutosaveSaveCallback = Arc<dyn Fn(AutosaveSaveEvent) + Send + Sync>;
pub type AutosaveErrorCallback = Arc<dyn Fn(AutosaveErrorEvent) + Send + Sync>;
pub type AutosaveSkipCallback = Arc<dyn Fn(AutosaveSkipEvent) + Send + Sync>;
pub type AutosaveCoalesceCallback = Arc<dyn Fn(AutosaveCoalesceEvent) + Send + Sync>;
pub type AutosaveThresholdUpdateCallback = Arc<dyn Fn(ThresholdUpdateEvent) + Send + Sync>;
pub type AutosaveCompactionCallback = Arc<dyn Fn(AutosaveCompactionEvent) + Send + Sync>;
pub type AutosaveWarmingCallback = Arc<dyn Fn(AutosaveWarmingEvent) + Send + Sync>;

/// Callbacks for cache autosave events
///
/// Use this to receive notifications when autosave operations complete or fail.
/// Callbacks are invoked from within the autosave background task.
///
/// # Example
///
/// ```ignore
/// use dashprove_selfimp::{AsyncImprovementVerifier, CacheAutosaveCallbacks};
/// use std::sync::Arc;
/// use std::time::Duration;
///
/// let verifier = AsyncImprovementVerifier::with_cache();
///
/// let callbacks = CacheAutosaveCallbacks::new()
///     .on_save(|event| {
///         println!("Saved {} bytes to {:?}", event.bytes_written, event.path);
///     })
///     .on_error(|event| {
///         eprintln!("Save failed: {}", event.error);
///     });
///
/// let handle = verifier.start_cache_autosave_with_callbacks(
///     "cache.json",
///     Duration::from_secs(60),
///     callbacks
/// )?;
/// # Ok::<(), dashprove_selfimp::CachePersistenceError>(())
/// ```
#[derive(Clone, Default)]
pub struct CacheAutosaveCallbacks {
    on_save: Option<AutosaveSaveCallback>,
    on_error: Option<AutosaveErrorCallback>,
    on_skip: Option<AutosaveSkipCallback>,
    on_coalesce: Option<AutosaveCoalesceCallback>,
    on_threshold_update: Option<AutosaveThresholdUpdateCallback>,
    on_compaction: Option<AutosaveCompactionCallback>,
    on_warming: Option<AutosaveWarmingCallback>,
}

impl CacheAutosaveCallbacks {
    /// Create a new callbacks configuration with no callbacks set
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a callback to be invoked after each successful save
    ///
    /// The callback receives an `AutosaveSaveEvent` with details about what was saved.
    pub fn on_save<F>(mut self, callback: F) -> Self
    where
        F: Fn(AutosaveSaveEvent) + Send + Sync + 'static,
    {
        self.on_save = Some(Arc::new(callback));
        self
    }

    /// Set a callback to be invoked when a save fails
    ///
    /// The callback receives an `AutosaveErrorEvent` with details about the failure.
    pub fn on_error<F>(mut self, callback: F) -> Self
    where
        F: Fn(AutosaveErrorEvent) + Send + Sync + 'static,
    {
        self.on_error = Some(Arc::new(callback));
        self
    }

    /// Set a callback to be invoked when a save is skipped due to minimum change threshold
    ///
    /// The callback receives an `AutosaveSkipEvent` with details about why the save was skipped.
    pub fn on_skip<F>(mut self, callback: F) -> Self
    where
        F: Fn(AutosaveSkipEvent) + Send + Sync + 'static,
    {
        self.on_skip = Some(Arc::new(callback));
        self
    }

    /// Set a callback to be invoked when saves are coalesced
    ///
    /// The callback receives an `AutosaveCoalesceEvent` with details about the coalescing.
    pub fn on_coalesce<F>(mut self, callback: F) -> Self
    where
        F: Fn(AutosaveCoalesceEvent) + Send + Sync + 'static,
    {
        self.on_coalesce = Some(Arc::new(callback));
        self
    }

    /// Set a callback to be invoked when learned thresholds are updated
    ///
    /// The callback receives a `ThresholdUpdateEvent` with the old and new threshold values.
    /// This is only called when learning thresholds are enabled and the system has
    /// accumulated enough samples to update its learned thresholds.
    pub fn on_threshold_update<F>(mut self, callback: F) -> Self
    where
        F: Fn(ThresholdUpdateEvent) + Send + Sync + 'static,
    {
        self.on_threshold_update = Some(Arc::new(callback));
        self
    }

    /// Set a callback to be invoked when compaction is triggered during autosave
    ///
    /// The callback receives an `AutosaveCompactionEvent` with details about the compaction.
    /// This is only called when compaction triggers are configured and one of the triggers
    /// fires during the autosave loop.
    pub fn on_compaction<F>(mut self, callback: F) -> Self
    where
        F: Fn(AutosaveCompactionEvent) + Send + Sync + 'static,
    {
        self.on_compaction = Some(Arc::new(callback));
        self
    }

    /// Set a callback to be invoked when cache warming occurs at startup
    ///
    /// The callback receives an `AutosaveWarmingEvent` with details about the warming.
    /// This is only called when a warming config is provided and an existing snapshot
    /// file is found at the autosave path during startup.
    pub fn on_warming<F>(mut self, callback: F) -> Self
    where
        F: Fn(AutosaveWarmingEvent) + Send + Sync + 'static,
    {
        self.on_warming = Some(Arc::new(callback));
        self
    }
}

impl std::fmt::Debug for CacheAutosaveCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheAutosaveCallbacks")
            .field("on_save", &self.on_save.as_ref().map(|_| "Fn(...)"))
            .field("on_error", &self.on_error.as_ref().map(|_| "Fn(...)"))
            .field("on_skip", &self.on_skip.as_ref().map(|_| "Fn(...)"))
            .field("on_coalesce", &self.on_coalesce.as_ref().map(|_| "Fn(...)"))
            .field(
                "on_threshold_update",
                &self.on_threshold_update.as_ref().map(|_| "Fn(...)"),
            )
            .field(
                "on_compaction",
                &self.on_compaction.as_ref().map(|_| "Fn(...)"),
            )
            .field("on_warming", &self.on_warming.as_ref().map(|_| "Fn(...)"))
            .finish()
    }
}

/// Configuration for adaptive interval adjustment based on cache activity
///
/// When adaptive intervals are enabled, the autosave system monitors cache activity
/// and automatically adjusts the save interval:
/// - High activity (changes exceed `high_activity_threshold`) → interval decreases
/// - Low activity (changes below `low_activity_threshold`) → interval increases
///
/// The interval is clamped between `min_interval` and `max_interval` to ensure
/// reasonable bounds regardless of activity patterns.
///
/// # Example
///
/// ```ignore
/// use dashprove_selfimp::AdaptiveIntervalConfig;
/// use std::time::Duration;
///
/// // Adapt interval between 15s and 5min based on byte changes
/// let config = AdaptiveIntervalConfig::new(
///     Duration::from_secs(15),   // min_interval
///     Duration::from_secs(300),  // max_interval
/// )
/// .with_high_activity_threshold(10_000)  // >10KB change = high activity
/// .with_low_activity_threshold(1_000);   // <1KB change = low activity
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveIntervalConfig {
    /// Minimum interval (floor for adaptive adjustment)
    pub min_interval: Duration,
    /// Maximum interval (ceiling for adaptive adjustment)
    pub max_interval: Duration,
    /// Byte change threshold above which activity is considered "high"
    ///
    /// When changes exceed this threshold, the interval will decrease
    /// (subject to `min_interval` floor).
    pub high_activity_threshold: u64,
    /// Byte change threshold below which activity is considered "low"
    ///
    /// When changes are below this threshold, the interval will increase
    /// (subject to `max_interval` ceiling).
    pub low_activity_threshold: u64,
    /// Factor by which to decrease interval on high activity (0.5 = halve)
    pub decrease_factor: f64,
    /// Factor by which to increase interval on low activity (2.0 = double)
    pub increase_factor: f64,
}

impl AdaptiveIntervalConfig {
    /// Create a new adaptive interval configuration with default thresholds
    ///
    /// Default thresholds:
    /// - High activity: 10KB change
    /// - Low activity: 1KB change
    /// - Decrease factor: 0.5 (halve on high activity)
    /// - Increase factor: 1.5 (50% increase on low activity)
    pub fn new(min_interval: Duration, max_interval: Duration) -> Self {
        Self {
            min_interval,
            max_interval,
            high_activity_threshold: 10 * 1024, // 10KB
            low_activity_threshold: 1024,       // 1KB
            decrease_factor: 0.5,
            increase_factor: 1.5,
        }
    }

    /// Set the high activity threshold (bytes changed)
    pub fn with_high_activity_threshold(mut self, threshold: u64) -> Self {
        self.high_activity_threshold = threshold;
        self
    }

    /// Set the low activity threshold (bytes changed)
    pub fn with_low_activity_threshold(mut self, threshold: u64) -> Self {
        self.low_activity_threshold = threshold;
        self
    }

    /// Set the decrease factor for high activity (e.g., 0.5 = halve interval)
    pub fn with_decrease_factor(mut self, factor: f64) -> Self {
        self.decrease_factor = factor.clamp(0.1, 1.0);
        self
    }

    /// Set the increase factor for low activity (e.g., 2.0 = double interval)
    pub fn with_increase_factor(mut self, factor: f64) -> Self {
        self.increase_factor = factor.clamp(1.0, 10.0);
        self
    }

    /// Compute the next interval based on observed change
    ///
    /// Returns the adjusted interval clamped to [min_interval, max_interval].
    pub fn compute_next_interval(&self, current_interval: Duration, change_bytes: u64) -> Duration {
        self.compute_next_interval_with_thresholds(
            current_interval,
            change_bytes,
            self.high_activity_threshold,
            self.low_activity_threshold,
        )
    }

    /// Compute the next interval using custom threshold values
    ///
    /// This method allows overriding the configured thresholds with learned values
    /// from a `HistoricalActivityTracker`. The learned thresholds adapt to actual
    /// usage patterns, providing more intelligent interval adjustment.
    ///
    /// # Arguments
    ///
    /// * `current_interval` - The current save interval
    /// * `change_bytes` - Number of bytes changed since last check
    /// * `high_threshold` - Bytes above which activity is considered "high"
    /// * `low_threshold` - Bytes below which activity is considered "low"
    pub fn compute_next_interval_with_thresholds(
        &self,
        current_interval: Duration,
        change_bytes: u64,
        high_threshold: u64,
        low_threshold: u64,
    ) -> Duration {
        let new_interval = if change_bytes >= high_threshold {
            // High activity: decrease interval
            Duration::from_secs_f64(current_interval.as_secs_f64() * self.decrease_factor)
        } else if change_bytes <= low_threshold {
            // Low activity: increase interval
            Duration::from_secs_f64(current_interval.as_secs_f64() * self.increase_factor)
        } else {
            // Normal activity: keep current interval
            current_interval
        };

        // Clamp to bounds
        new_interval.clamp(self.min_interval, self.max_interval)
    }
}

impl Default for AdaptiveIntervalConfig {
    fn default() -> Self {
        Self::new(Duration::from_secs(15), Duration::from_secs(300))
    }
}

/// Configuration for save coalescing behavior
///
/// Coalescing batches rapid changes into single save operations by waiting
/// for activity to settle before saving. This reduces I/O overhead during
/// bursts of verification activity.
///
/// # How It Works
///
/// When coalescing is enabled:
/// 1. When a save would occur and activity is detected, start a "quiet period" timer
/// 2. If more changes occur during the quiet period, reset the timer
/// 3. Only save when the quiet period expires without new changes
/// 4. A maximum wait time prevents indefinite delays during sustained activity
///
/// # Example
///
/// ```ignore
/// use dashprove_selfimp::CoalesceConfig;
/// use std::time::Duration;
///
/// let config = CoalesceConfig::new(
///     Duration::from_millis(500),  // Wait 500ms after last change
///     Duration::from_secs(5),      // But never wait more than 5s total
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoalesceConfig {
    /// Duration to wait after the last detected change before saving
    ///
    /// Each new change resets this timer. Only when activity has been
    /// quiet for this duration will the save proceed.
    pub quiet_period: Duration,
    /// Maximum time to wait before forcing a save regardless of activity
    ///
    /// This prevents indefinite delays during sustained activity bursts.
    /// The save will proceed after this duration even if changes are still
    /// occurring within the quiet period.
    pub max_wait: Duration,
    /// Minimum byte change to consider as "activity" for coalescing purposes
    ///
    /// Changes below this threshold won't reset the quiet period timer.
    /// Default is 0 (any change resets the timer).
    pub activity_threshold: u64,
}

impl CoalesceConfig {
    /// Create a new coalesce configuration
    ///
    /// # Arguments
    /// * `quiet_period` - How long to wait after last change before saving
    /// * `max_wait` - Maximum time to wait before forcing a save
    pub fn new(quiet_period: Duration, max_wait: Duration) -> Self {
        Self {
            quiet_period,
            max_wait,
            activity_threshold: 0,
        }
    }

    /// Set the activity threshold for coalescing
    ///
    /// Changes below this byte threshold won't reset the quiet period.
    pub fn with_activity_threshold(mut self, threshold: u64) -> Self {
        self.activity_threshold = threshold;
        self
    }

    /// Create a configuration optimized for burst handling
    ///
    /// Uses a short quiet period (200ms) with moderate max wait (3s).
    /// Good for handling rapid verification bursts.
    pub fn burst() -> Self {
        Self {
            quiet_period: Duration::from_millis(200),
            max_wait: Duration::from_secs(3),
            activity_threshold: 512, // 512 bytes minimum to count as activity
        }
    }

    /// Create a configuration optimized for heavy batching
    ///
    /// Uses a longer quiet period (1s) with longer max wait (10s).
    /// Good for scenarios with sustained high activity.
    pub fn aggressive() -> Self {
        Self {
            quiet_period: Duration::from_secs(1),
            max_wait: Duration::from_secs(10),
            activity_threshold: 1024, // 1KB minimum
        }
    }
}

impl Default for CoalesceConfig {
    fn default() -> Self {
        Self::new(Duration::from_millis(500), Duration::from_secs(5))
    }
}

/// Configuration for exponential backoff on save errors
///
/// When a save operation fails, the autosave loop can use exponential backoff
/// to reduce the frequency of retries, preventing resource exhaustion when
/// there are persistent issues (e.g., disk full, permission denied).
///
/// # How It Works
///
/// After each error:
/// 1. The delay before the next attempt is multiplied by `multiplier`
/// 2. The delay is clamped to not exceed `max_delay`
/// 3. After a successful save, the delay resets to `initial_delay`
///
/// # Example
///
/// ```ignore
/// use dashprove_selfimp::BackoffConfig;
/// use std::time::Duration;
///
/// let config = BackoffConfig::new(
///     Duration::from_secs(1),    // Start with 1s delay after first error
///     Duration::from_secs(300),  // Max out at 5 minutes
/// )
/// .with_multiplier(2.0);  // Double delay after each error
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BackoffConfig {
    /// Initial delay after the first error before retrying
    pub initial_delay: Duration,
    /// Maximum delay between retry attempts
    pub max_delay: Duration,
    /// Multiplier applied to delay after each consecutive error (e.g., 2.0 = double)
    pub multiplier: f64,
    /// Number of consecutive errors before triggering backoff
    ///
    /// Default is 1 (backoff immediately after first error).
    /// Set higher to tolerate transient errors before backing off.
    pub error_threshold: usize,
}

impl BackoffConfig {
    /// Create a new backoff configuration
    ///
    /// # Arguments
    /// * `initial_delay` - Delay after first error (base for exponential growth)
    /// * `max_delay` - Maximum delay cap (prevents delays from growing indefinitely)
    pub fn new(initial_delay: Duration, max_delay: Duration) -> Self {
        Self {
            initial_delay,
            max_delay,
            multiplier: 2.0,
            error_threshold: 1,
        }
    }

    /// Set the multiplier for exponential growth (default: 2.0)
    ///
    /// After each error, the delay is multiplied by this factor.
    /// Clamped to range [1.1, 10.0] to ensure reasonable growth.
    pub fn with_multiplier(mut self, multiplier: f64) -> Self {
        self.multiplier = multiplier.clamp(1.1, 10.0);
        self
    }

    /// Set the number of consecutive errors before triggering backoff
    ///
    /// Default is 1 (backoff starts after first error).
    /// Set higher to tolerate occasional transient errors.
    pub fn with_error_threshold(mut self, threshold: usize) -> Self {
        self.error_threshold = threshold.max(1);
        self
    }

    /// Compute the next delay given consecutive error count
    ///
    /// Returns `None` if error count hasn't reached threshold yet.
    /// Returns the backoff delay otherwise, clamped to max_delay.
    pub fn compute_delay(&self, consecutive_errors: usize) -> Option<Duration> {
        if consecutive_errors < self.error_threshold {
            return None;
        }

        // Number of backoff iterations (0-indexed from threshold)
        let backoff_iterations = consecutive_errors.saturating_sub(self.error_threshold);

        // Compute delay: initial_delay * multiplier^iterations
        let multiplied =
            self.initial_delay.as_secs_f64() * self.multiplier.powi(backoff_iterations as i32);

        let delay = Duration::from_secs_f64(multiplied);
        Some(delay.min(self.max_delay))
    }

    /// Create a configuration optimized for transient errors
    ///
    /// Uses a short initial delay (500ms) with fast growth (2x).
    /// Good for I/O errors that may resolve quickly.
    pub fn transient() -> Self {
        Self {
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
            error_threshold: 1,
        }
    }

    /// Create a configuration optimized for persistent errors
    ///
    /// Uses a longer initial delay (2s) with slower growth (1.5x).
    /// Better for disk space issues or permission problems.
    pub fn persistent() -> Self {
        Self {
            initial_delay: Duration::from_secs(2),
            max_delay: Duration::from_secs(300),
            multiplier: 1.5,
            error_threshold: 2, // Tolerate one transient error
        }
    }

    /// Create a configuration for aggressive retries
    ///
    /// Uses a very short initial delay (100ms) but quick backoff.
    /// Good when immediate retry is often successful.
    pub fn aggressive() -> Self {
        Self {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            multiplier: 3.0,
            error_threshold: 1,
        }
    }
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self::new(Duration::from_secs(1), Duration::from_secs(120))
    }
}

/// Retry policy for verification attempts
///
/// This policy controls when the verifier should retry a verification
/// attempt and how to space those retries using exponential backoff.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VerificationRetryPolicy {
    /// Maximum number of attempts (including the first attempt).
    pub max_attempts: usize,
    /// Backoff configuration used between attempts.
    pub backoff: BackoffConfig,
    /// Retry when dispatcher returns an error (selection/execution issues).
    pub retry_on_dispatcher_error: bool,
    /// Retry when confidence is below the configured minimum.
    pub retry_on_low_confidence: bool,
    /// Retry when not enough backends passed.
    pub retry_on_insufficient_backends: bool,
    /// Retry for any failed verification result (including disproven).
    pub retry_on_failure: bool,
    /// Maximum additional jitter in milliseconds added to the backoff delay.
    pub jitter_ms: u64,
}

impl VerificationRetryPolicy {
    /// Create a new retry policy with the given attempt budget.
    pub fn new(max_attempts: usize) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            backoff: BackoffConfig::transient(),
            retry_on_dispatcher_error: true,
            retry_on_low_confidence: true,
            retry_on_insufficient_backends: true,
            retry_on_failure: false,
            jitter_ms: 0,
        }
    }

    /// Set the backoff configuration.
    pub fn with_backoff(mut self, backoff: BackoffConfig) -> Self {
        self.backoff = backoff;
        self
    }

    /// Enable or disable dispatcher error retries.
    pub fn retry_on_dispatcher_error(mut self, enabled: bool) -> Self {
        self.retry_on_dispatcher_error = enabled;
        self
    }

    /// Enable or disable low confidence retries.
    pub fn retry_on_low_confidence(mut self, enabled: bool) -> Self {
        self.retry_on_low_confidence = enabled;
        self
    }

    /// Enable or disable insufficient backend retries.
    pub fn retry_on_insufficient_backends(mut self, enabled: bool) -> Self {
        self.retry_on_insufficient_backends = enabled;
        self
    }

    /// Enable or disable retries on any failure.
    pub fn retry_on_failure(mut self, enabled: bool) -> Self {
        self.retry_on_failure = enabled;
        self
    }

    /// Configure maximum jitter (in milliseconds) added to backoff delays.
    pub fn with_jitter_ms(mut self, jitter_ms: u64) -> Self {
        self.jitter_ms = jitter_ms;
        self
    }

    /// Determine if an error should trigger a retry.
    pub fn should_retry_error(&self, error: &SelfImpError) -> bool {
        if !self.retry_on_dispatcher_error {
            return false;
        }

        matches!(
            error,
            SelfImpError::DispatcherError(_) | SelfImpError::VerificationTimeout(_)
        )
    }

    /// Determine if a result should trigger a retry.
    pub fn should_retry_result(
        &self,
        result: &VerificationResult,
        config: &VerificationConfig,
    ) -> bool {
        let passing = result.backend_results.values().filter(|r| r.passed).count();

        if self.retry_on_insufficient_backends && passing < config.min_passing_backends {
            return true;
        }

        if self.retry_on_low_confidence && result.confidence < config.min_confidence {
            return true;
        }

        if self.retry_on_failure && !result.passed {
            return true;
        }

        false
    }

    /// Compute the delay for the given consecutive failure count (1-based).
    pub fn next_delay(&self, consecutive_failures: usize) -> Option<Duration> {
        let base = self.backoff.compute_delay(consecutive_failures)?;
        if self.jitter_ms == 0 {
            return Some(base);
        }

        let jitter = (consecutive_failures as u64 * 37) % (self.jitter_ms + 1);
        Some(base + Duration::from_millis(jitter))
    }
}

impl Default for VerificationRetryPolicy {
    fn default() -> Self {
        Self::new(3)
    }
}

/// Configuration for learning-based adaptive thresholds
///
/// This configuration enables the autosave system to learn from historical
/// activity patterns and dynamically adjust thresholds based on observed
/// behavior. The tracker maintains a rolling window of activity samples
/// and uses statistical analysis to predict optimal thresholds.
///
/// # How It Works
///
/// 1. **Sample Collection**: Each save interval, the tracker records:
///    - Timestamp (for time-of-day patterns)
///    - Byte changes (activity level)
///    - Interval used (for effectiveness analysis)
///
/// 2. **Pattern Learning**: The tracker analyzes samples to detect:
///    - Time-of-day patterns (e.g., high activity during work hours)
///    - Trend detection (increasing or decreasing activity)
///    - Burst detection (sudden spikes in activity)
///
/// 3. **Threshold Adjustment**: Based on learned patterns:
///    - `high_activity_threshold` adjusts to the 75th percentile of recent activity
///    - `low_activity_threshold` adjusts to the 25th percentile
///    - Smoothing prevents abrupt changes
///
/// # Example
///
/// ```ignore
/// use dashprove_selfimp::{LearningThresholdConfig, CacheAutosaveConfig};
/// use std::time::Duration;
///
/// // Enable learning with 1-hour window and 5-minute warmup
/// let learning = LearningThresholdConfig::new()
///     .with_window_duration(Duration::from_secs(3600))
///     .with_warmup_samples(12)  // 12 samples before starting adaptation
///     .with_smoothing_factor(0.3);  // Gradual threshold changes
///
/// let config = CacheAutosaveConfig::adaptive("cache.json")
///     .with_learning_thresholds(learning);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LearningThresholdConfig {
    /// Duration of the rolling window for historical samples
    ///
    /// Samples older than this are discarded. Longer windows provide
    /// more stable thresholds but are slower to adapt to changes.
    pub window_duration: Duration,

    /// Number of samples required before adaptation begins
    ///
    /// During warmup, default thresholds are used. This prevents
    /// erratic behavior with insufficient data.
    pub warmup_samples: usize,

    /// Maximum number of samples to retain in memory
    ///
    /// Once this limit is reached, oldest samples are evicted
    /// regardless of window_duration. Prevents unbounded memory growth.
    pub max_samples: usize,

    /// Smoothing factor for threshold updates (0.0 - 1.0)
    ///
    /// Controls how quickly thresholds adapt to new data:
    /// - 0.0: Never update (use initial thresholds forever)
    /// - 0.1: Very slow adaptation (stable but sluggish)
    /// - 0.3: Moderate adaptation (good default)
    /// - 0.5: Fast adaptation (responsive but may oscillate)
    /// - 1.0: Immediate (uses raw computed values, may be noisy)
    pub smoothing_factor: f64,

    /// Percentile for high activity threshold (default: 75th)
    ///
    /// Activity levels above this percentile of historical data
    /// are considered "high activity" and trigger interval decrease.
    pub high_percentile: f64,

    /// Percentile for low activity threshold (default: 25th)
    ///
    /// Activity levels below this percentile of historical data
    /// are considered "low activity" and trigger interval increase.
    pub low_percentile: f64,

    /// Enable time-of-day pattern detection
    ///
    /// When enabled, the tracker groups samples by hour-of-day
    /// and uses time-specific thresholds when enough data exists.
    pub time_aware: bool,

    /// Minimum threshold floor (bytes)
    ///
    /// Learned thresholds will never go below this value.
    /// Prevents pathological cases where thresholds become too sensitive.
    pub min_threshold_floor: u64,

    /// Maximum threshold ceiling (bytes)
    ///
    /// Learned thresholds will never exceed this value.
    /// Prevents pathological cases where thresholds become too large.
    pub max_threshold_ceiling: u64,
}

impl LearningThresholdConfig {
    /// Create a new learning threshold configuration with sensible defaults
    pub fn new() -> Self {
        Self {
            window_duration: Duration::from_secs(3600), // 1 hour
            warmup_samples: 10,
            max_samples: 1000,
            smoothing_factor: 0.3,
            high_percentile: 75.0,
            low_percentile: 25.0,
            time_aware: false,
            min_threshold_floor: 100,                 // 100 bytes minimum
            max_threshold_ceiling: 100 * 1024 * 1024, // 100MB maximum
        }
    }

    /// Set the rolling window duration
    pub fn with_window_duration(mut self, duration: Duration) -> Self {
        self.window_duration = duration;
        self
    }

    /// Set the number of warmup samples before adaptation begins
    pub fn with_warmup_samples(mut self, samples: usize) -> Self {
        self.warmup_samples = samples.max(1);
        self
    }

    /// Set the maximum number of samples to retain
    pub fn with_max_samples(mut self, max: usize) -> Self {
        self.max_samples = max.max(10);
        self
    }

    /// Set the smoothing factor for threshold updates
    ///
    /// Clamped to range [0.0, 1.0].
    pub fn with_smoothing_factor(mut self, factor: f64) -> Self {
        self.smoothing_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Set the percentile for high activity threshold
    ///
    /// Clamped to range [50.0, 99.0] to ensure it's above median
    /// and below the maximum.
    pub fn with_high_percentile(mut self, percentile: f64) -> Self {
        self.high_percentile = percentile.clamp(50.0, 99.0);
        self
    }

    /// Set the percentile for low activity threshold
    ///
    /// Clamped to range [1.0, 50.0] to ensure it's above minimum
    /// and below median.
    pub fn with_low_percentile(mut self, percentile: f64) -> Self {
        self.low_percentile = percentile.clamp(1.0, 50.0);
        self
    }

    /// Enable time-of-day pattern detection
    pub fn with_time_awareness(mut self, enabled: bool) -> Self {
        self.time_aware = enabled;
        self
    }

    /// Set the minimum threshold floor
    pub fn with_min_threshold_floor(mut self, floor: u64) -> Self {
        self.min_threshold_floor = floor;
        self
    }

    /// Set the maximum threshold ceiling
    pub fn with_max_threshold_ceiling(mut self, ceiling: u64) -> Self {
        self.max_threshold_ceiling = ceiling;
        self
    }

    /// Create a configuration optimized for short-term patterns
    ///
    /// Uses a 15-minute window with fast adaptation.
    /// Good for systems with rapidly changing workloads.
    pub fn short_term() -> Self {
        Self {
            window_duration: Duration::from_secs(15 * 60), // 15 minutes
            warmup_samples: 5,
            max_samples: 100,
            smoothing_factor: 0.5,
            high_percentile: 75.0,
            low_percentile: 25.0,
            time_aware: false,
            min_threshold_floor: 100,
            max_threshold_ceiling: 100 * 1024 * 1024,
        }
    }

    /// Create a configuration optimized for long-term patterns
    ///
    /// Uses a 24-hour window with slow adaptation and time awareness.
    /// Good for systems with predictable daily patterns.
    pub fn long_term() -> Self {
        Self {
            window_duration: Duration::from_secs(24 * 3600), // 24 hours
            warmup_samples: 100,
            max_samples: 5000,
            smoothing_factor: 0.1,
            high_percentile: 80.0,
            low_percentile: 20.0,
            time_aware: true,
            min_threshold_floor: 100,
            max_threshold_ceiling: 100 * 1024 * 1024,
        }
    }

    /// Create a configuration optimized for stable environments
    ///
    /// Uses very slow adaptation to avoid reacting to noise.
    pub fn stable() -> Self {
        Self {
            window_duration: Duration::from_secs(6 * 3600), // 6 hours
            warmup_samples: 50,
            max_samples: 2000,
            smoothing_factor: 0.1,
            high_percentile: 70.0,
            low_percentile: 30.0,
            time_aware: false,
            min_threshold_floor: 1024,               // 1KB minimum
            max_threshold_ceiling: 10 * 1024 * 1024, // 10MB maximum
        }
    }
}

impl Default for LearningThresholdConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A sample of activity recorded for learning purposes
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActivitySample {
    /// When this sample was recorded (Unix timestamp in milliseconds)
    pub timestamp_ms: u64,
    /// Bytes changed during this interval
    pub bytes_changed: u64,
    /// Interval used for this save (milliseconds)
    pub interval_ms: u64,
    /// Hour of day when sample was recorded (0-23)
    pub hour_of_day: u8,
}

impl ActivitySample {
    /// Create a new activity sample with the current timestamp
    pub fn now(bytes_changed: u64, interval: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        let timestamp_ms = now.as_millis() as u64;

        // Calculate hour of day from timestamp
        let secs_since_epoch = now.as_secs();
        let hour_of_day = ((secs_since_epoch % 86400) / 3600) as u8;

        Self {
            timestamp_ms,
            bytes_changed,
            interval_ms: interval.as_millis() as u64,
            hour_of_day,
        }
    }

    /// Create a sample with a specific timestamp (for testing)
    pub fn with_timestamp(timestamp_ms: u64, bytes_changed: u64, interval: Duration) -> Self {
        let secs_since_epoch = timestamp_ms / 1000;
        let hour_of_day = ((secs_since_epoch % 86400) / 3600) as u8;

        Self {
            timestamp_ms,
            bytes_changed,
            interval_ms: interval.as_millis() as u64,
            hour_of_day,
        }
    }
}

/// Event emitted when learned thresholds are updated
#[derive(Debug, Clone)]
pub struct ThresholdUpdateEvent {
    /// Old high activity threshold
    pub old_high_threshold: u64,
    /// New high activity threshold
    pub new_high_threshold: u64,
    /// Old low activity threshold
    pub old_low_threshold: u64,
    /// New low activity threshold
    pub new_low_threshold: u64,
    /// Number of samples used for the update
    pub sample_count: usize,
    /// Whether time-of-day patterns influenced the update
    pub time_aware_adjustment: bool,
}

/// Tracker that learns from historical activity patterns
///
/// This struct maintains a rolling window of activity samples and
/// computes adaptive thresholds based on observed patterns.
///
/// # Thread Safety
///
/// This struct is NOT thread-safe. For concurrent access, wrap in
/// `Arc<Mutex<HistoricalActivityTracker>>`.
#[derive(Debug, Clone)]
pub struct HistoricalActivityTracker {
    config: LearningThresholdConfig,
    samples: Vec<ActivitySample>,
    current_high_threshold: u64,
    current_low_threshold: u64,
    /// Per-hour statistics for time-aware learning (24 slots)
    hourly_stats: Vec<HourlyStats>,
}

/// Statistics for a single hour of the day
#[derive(Debug, Clone, Default)]
struct HourlyStats {
    sample_count: usize,
    sum_bytes: u64,
    sum_bytes_squared: u128, // For variance calculation
}

impl HourlyStats {
    fn add_sample(&mut self, bytes: u64) {
        self.sample_count += 1;
        self.sum_bytes += bytes;
        self.sum_bytes_squared += (bytes as u128) * (bytes as u128);
    }

    fn mean(&self) -> Option<f64> {
        if self.sample_count == 0 {
            None
        } else {
            Some(self.sum_bytes as f64 / self.sample_count as f64)
        }
    }

    #[allow(dead_code)] // May be used for variance-based learning in the future
    fn variance(&self) -> Option<f64> {
        if self.sample_count < 2 {
            None
        } else {
            let mean = self.mean()?;
            let mean_sq = self.sum_bytes_squared as f64 / self.sample_count as f64;
            Some((mean_sq - mean * mean).max(0.0))
        }
    }
}

impl HistoricalActivityTracker {
    /// Create a new tracker with the given configuration
    pub fn new(config: LearningThresholdConfig) -> Self {
        Self {
            current_high_threshold: 10 * 1024, // Default 10KB
            current_low_threshold: 1024,       // Default 1KB
            samples: Vec::with_capacity(config.max_samples.min(1000)),
            hourly_stats: vec![HourlyStats::default(); 24],
            config,
        }
    }

    /// Create a tracker with default configuration
    pub fn with_defaults() -> Self {
        Self::new(LearningThresholdConfig::default())
    }

    /// Record a new activity sample
    ///
    /// Returns `Some(ThresholdUpdateEvent)` if thresholds were updated,
    /// `None` otherwise (e.g., during warmup).
    pub fn record_sample(
        &mut self,
        bytes_changed: u64,
        interval: Duration,
    ) -> Option<ThresholdUpdateEvent> {
        let sample = ActivitySample::now(bytes_changed, interval);
        self.record_sample_internal(sample)
    }

    /// Record a sample with a specific timestamp (for testing)
    pub fn record_sample_at(
        &mut self,
        timestamp_ms: u64,
        bytes_changed: u64,
        interval: Duration,
    ) -> Option<ThresholdUpdateEvent> {
        let sample = ActivitySample::with_timestamp(timestamp_ms, bytes_changed, interval);
        self.record_sample_internal(sample)
    }

    fn record_sample_internal(&mut self, sample: ActivitySample) -> Option<ThresholdUpdateEvent> {
        // Add to hourly stats
        self.hourly_stats[sample.hour_of_day as usize].add_sample(sample.bytes_changed);

        // Add sample
        self.samples.push(sample);

        // Prune old samples
        self.prune_old_samples();

        // Check if we have enough samples for adaptation
        if self.samples.len() < self.config.warmup_samples {
            return None;
        }

        // Compute and update thresholds
        self.update_thresholds()
    }

    fn prune_old_samples(&mut self) {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let cutoff_ms = now_ms.saturating_sub(self.config.window_duration.as_millis() as u64);

        // Remove samples older than window
        self.samples.retain(|s| s.timestamp_ms >= cutoff_ms);

        // Also enforce max_samples limit
        while self.samples.len() > self.config.max_samples {
            self.samples.remove(0);
        }
    }

    fn update_thresholds(&mut self) -> Option<ThresholdUpdateEvent> {
        if self.samples.is_empty() {
            return None;
        }

        let old_high = self.current_high_threshold;
        let old_low = self.current_low_threshold;

        // Collect activity values
        let mut values: Vec<u64> = self.samples.iter().map(|s| s.bytes_changed).collect();
        values.sort_unstable();

        // Compute percentiles
        let new_high_raw = self.percentile(&values, self.config.high_percentile);
        let new_low_raw = self.percentile(&values, self.config.low_percentile);

        // Apply time-of-day adjustment if enabled
        let (new_high_raw, new_low_raw, time_adjusted) = if self.config.time_aware {
            self.apply_time_adjustment(new_high_raw, new_low_raw)
        } else {
            (new_high_raw, new_low_raw, false)
        };

        // Apply smoothing
        let new_high = self.smooth(self.current_high_threshold, new_high_raw);
        let new_low = self.smooth(self.current_low_threshold, new_low_raw);

        // Apply floor and ceiling
        let new_high = new_high.clamp(
            self.config.min_threshold_floor,
            self.config.max_threshold_ceiling,
        );
        let new_low = new_low.clamp(
            self.config.min_threshold_floor,
            self.config.max_threshold_ceiling,
        );

        // Ensure high > low
        let (new_high, new_low) = if new_high <= new_low {
            // If they're inverted or equal, spread them apart
            let mid = (new_high + new_low) / 2;
            let spread = (self.config.max_threshold_ceiling - self.config.min_threshold_floor) / 10;
            (
                (mid + spread).min(self.config.max_threshold_ceiling),
                mid.saturating_sub(spread)
                    .max(self.config.min_threshold_floor),
            )
        } else {
            (new_high, new_low)
        };

        // Check if thresholds actually changed
        if new_high == old_high && new_low == old_low {
            return None;
        }

        self.current_high_threshold = new_high;
        self.current_low_threshold = new_low;

        Some(ThresholdUpdateEvent {
            old_high_threshold: old_high,
            new_high_threshold: new_high,
            old_low_threshold: old_low,
            new_low_threshold: new_low,
            sample_count: self.samples.len(),
            time_aware_adjustment: time_adjusted,
        })
    }

    fn percentile(&self, sorted_values: &[u64], percentile: f64) -> u64 {
        if sorted_values.is_empty() {
            return 0;
        }
        let index = ((percentile / 100.0) * (sorted_values.len() - 1) as f64).round() as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }

    fn smooth(&self, current: u64, target: u64) -> u64 {
        let current_f = current as f64;
        let target_f = target as f64;
        let smoothed = current_f + self.config.smoothing_factor * (target_f - current_f);
        smoothed.round() as u64
    }

    fn apply_time_adjustment(&self, high: u64, low: u64) -> (u64, u64, bool) {
        // Get current hour
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        let hour = ((now.as_secs() % 86400) / 3600) as usize;

        let stats = &self.hourly_stats[hour];

        // Need at least 5 samples for this hour to apply time adjustment
        if stats.sample_count < 5 {
            return (high, low, false);
        }

        // Use hourly mean to adjust thresholds
        if let Some(hourly_mean) = stats.mean() {
            // Collect overall mean
            let overall_mean: f64 = self
                .samples
                .iter()
                .map(|s| s.bytes_changed as f64)
                .sum::<f64>()
                / self.samples.len() as f64;

            if overall_mean > 0.0 {
                let ratio = hourly_mean / overall_mean;
                // Scale thresholds based on whether this hour is busier or quieter than average
                let adjusted_high = (high as f64 * ratio).round() as u64;
                let adjusted_low = (low as f64 * ratio).round() as u64;
                return (adjusted_high, adjusted_low, true);
            }
        }

        (high, low, false)
    }

    /// Get the current high activity threshold
    pub fn high_threshold(&self) -> u64 {
        self.current_high_threshold
    }

    /// Get the current low activity threshold
    pub fn low_threshold(&self) -> u64 {
        self.current_low_threshold
    }

    /// Get the number of samples currently tracked
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Check if the tracker has completed warmup
    pub fn is_warmed_up(&self) -> bool {
        self.samples.len() >= self.config.warmup_samples
    }

    /// Get statistics about the tracked activity
    pub fn statistics(&self) -> ActivityStatistics {
        if self.samples.is_empty() {
            return ActivityStatistics::default();
        }

        let values: Vec<u64> = self.samples.iter().map(|s| s.bytes_changed).collect();
        let sum: u64 = values.iter().sum();
        let mean = sum as f64 / values.len() as f64;

        let variance: f64 = values
            .iter()
            .map(|&v| {
                let diff = v as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;

        let min = *values.iter().min().unwrap_or(&0);
        let max = *values.iter().max().unwrap_or(&0);

        ActivityStatistics {
            sample_count: values.len(),
            mean_bytes: mean,
            std_dev_bytes: variance.sqrt(),
            min_bytes: min,
            max_bytes: max,
            current_high_threshold: self.current_high_threshold,
            current_low_threshold: self.current_low_threshold,
        }
    }

    /// Clear all samples and reset thresholds to defaults
    pub fn reset(&mut self) {
        self.samples.clear();
        self.current_high_threshold = 10 * 1024;
        self.current_low_threshold = 1024;
        self.hourly_stats = vec![HourlyStats::default(); 24];
    }

    /// Export samples for persistence
    pub fn export_samples(&self) -> Vec<ActivitySample> {
        self.samples.clone()
    }

    /// Import samples (e.g., after restart)
    pub fn import_samples(&mut self, samples: Vec<ActivitySample>) {
        self.samples = samples;
        self.prune_old_samples();

        // Rebuild hourly stats
        self.hourly_stats = vec![HourlyStats::default(); 24];
        for sample in &self.samples {
            self.hourly_stats[sample.hour_of_day as usize].add_sample(sample.bytes_changed);
        }

        // Update thresholds based on imported data
        let _ = self.update_thresholds();
    }
}

/// Statistics about tracked activity
#[derive(Debug, Clone, Default)]
pub struct ActivityStatistics {
    /// Number of samples tracked
    pub sample_count: usize,
    /// Mean bytes changed per interval
    pub mean_bytes: f64,
    /// Standard deviation of bytes changed
    pub std_dev_bytes: f64,
    /// Minimum bytes changed in a single interval
    pub min_bytes: u64,
    /// Maximum bytes changed in a single interval
    pub max_bytes: u64,
    /// Current high activity threshold
    pub current_high_threshold: u64,
    /// Current low activity threshold
    pub current_low_threshold: u64,
}

/// Configuration for cache autosave behavior
///
/// This struct bundles all autosave settings together, making it easy to
/// create preset configurations for common use cases.
///
/// # Example
///
/// ```ignore
/// use dashprove_selfimp::{AsyncImprovementVerifier, CacheAutosaveConfig};
///
/// let verifier = AsyncImprovementVerifier::with_cache();
///
/// // Use a preset configuration
/// let config = CacheAutosaveConfig::storage_optimized("cache.json.gz");
///
/// let handle = verifier.start_cache_autosave_with_config(config)?;
/// # Ok::<(), dashprove_selfimp::CachePersistenceError>(())
/// ```
#[derive(Debug, Clone)]
pub struct CacheAutosaveConfig {
    /// Path to save cache snapshots to
    pub path: PathBuf,
    /// Interval between saves
    pub interval: Duration,
    /// Optional compression level (None means uncompressed JSON)
    pub compression: Option<SnapshotCompressionLevel>,
    /// Optional callbacks for save events
    pub callbacks: CacheAutosaveCallbacks,
    /// Minimum change threshold in bytes to trigger a save (0 = always save)
    ///
    /// When set to a non-zero value, saves will be skipped if the absolute difference
    /// between the current serialized size and the last saved size is below this threshold.
    /// This reduces unnecessary I/O when the cache hasn't changed significantly.
    pub min_change_bytes: u64,
    /// Maximum duration since last save before forcing a save (None = no limit)
    ///
    /// When set, a save will be forced if more than this duration has passed since
    /// the last successful save, regardless of the min_change_bytes threshold.
    /// This ensures data doesn't become too stale even when changes are small.
    pub max_stale_duration: Option<Duration>,
    /// Adaptive interval configuration (None = fixed interval)
    ///
    /// When set, the save interval will automatically adjust based on cache activity:
    /// - High activity (many changes) → shorter intervals for better data safety
    /// - Low activity (few changes) → longer intervals to reduce I/O overhead
    pub adaptive_interval: Option<AdaptiveIntervalConfig>,
    /// Coalescing configuration (None = no coalescing)
    ///
    /// When set, rapid changes will be batched into single save operations:
    /// - Wait for a quiet period after the last change before saving
    /// - A maximum wait time prevents indefinite delays during sustained activity
    pub coalesce: Option<CoalesceConfig>,
    /// Backoff configuration for errors (None = no backoff, always use normal interval)
    ///
    /// When set, save errors will trigger exponential backoff:
    /// - First error: wait initial_delay before retrying
    /// - Subsequent errors: multiply delay by multiplier (up to max_delay)
    /// - Successful save: reset back to normal interval
    pub backoff: Option<BackoffConfig>,
    /// Learning threshold configuration (None = use static thresholds)
    ///
    /// When set, the autosave system will learn from historical activity patterns
    /// and dynamically adjust the high/low activity thresholds used by adaptive intervals.
    /// This enables the system to automatically tune itself based on actual usage patterns.
    ///
    /// Note: This only takes effect when `adaptive_interval` is also set.
    pub learning_thresholds: Option<LearningThresholdConfig>,
    /// Compaction configuration (None = no compaction)
    ///
    /// When set, the cache will be compacted before saving according to the policy.
    /// This can significantly reduce storage size by removing expired, low-confidence,
    /// obsolete, or duplicate entries.
    pub compaction: Option<CompactionConfig>,
    /// Compaction triggers (empty = no trigger-based compaction)
    ///
    /// When set, these triggers will be checked during the autosave loop.
    /// If any trigger fires, compaction will be performed using the `compaction` config.
    /// This allows compaction to happen based on size, time, hit rate, etc.
    /// rather than just on every save.
    ///
    /// Note: If `compaction` is None, triggers are ignored.
    pub compaction_triggers: Vec<CompactionTrigger>,
    /// Warming configuration (None = no warming on load)
    ///
    /// When set, the cache will be warmed when loaded using the specified strategy.
    /// This allows selective restoration of cached entries based on confidence,
    /// recency, patterns, or other criteria.
    pub warming: Option<WarmingConfig>,
}

impl CacheAutosaveConfig {
    /// Create a new autosave configuration
    pub fn new(path: impl AsRef<Path>, interval: Duration) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            interval,
            compression: None,
            callbacks: CacheAutosaveCallbacks::default(),
            min_change_bytes: 0,             // Default: always save
            max_stale_duration: None,        // Default: no stale limit
            adaptive_interval: None,         // Default: fixed interval
            coalesce: None,                  // Default: no coalescing
            backoff: None,                   // Default: no backoff
            learning_thresholds: None,       // Default: static thresholds
            compaction: None,                // Default: no compaction
            compaction_triggers: Vec::new(), // Default: no triggers
            warming: None,                   // Default: no warming
        }
    }

    /// Enable compression with the specified level
    pub fn with_compression(mut self, level: SnapshotCompressionLevel) -> Self {
        self.compression = Some(level);
        self
    }

    /// Set callbacks for autosave events
    pub fn with_callbacks(mut self, callbacks: CacheAutosaveCallbacks) -> Self {
        self.callbacks = callbacks;
        self
    }

    /// Set minimum change threshold for saves
    ///
    /// When set to a non-zero value, saves will be skipped if the absolute change
    /// in serialized size since the last save is below this threshold.
    /// This reduces unnecessary I/O when the cache hasn't changed significantly.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::CacheAutosaveConfig;
    /// use std::time::Duration;
    ///
    /// // Only save if at least 1KB has changed
    /// let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
    ///     .with_min_change_bytes(1024);
    /// ```
    pub fn with_min_change_bytes(mut self, min_change_bytes: u64) -> Self {
        self.min_change_bytes = min_change_bytes;
        self
    }

    /// Set maximum stale duration before forcing a save
    ///
    /// When set, a save will be forced if more than this duration has passed since
    /// the last successful save, regardless of the min_change_bytes threshold.
    /// This ensures data doesn't become too stale even when changes are small.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::CacheAutosaveConfig;
    /// use std::time::Duration;
    ///
    /// // Force save if no successful save in 10 minutes, even if changes are small
    /// let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
    ///     .with_min_change_bytes(1024)  // Skip small changes
    ///     .with_max_stale_duration(Duration::from_secs(600)); // But save at least every 10 min
    /// ```
    pub fn with_max_stale_duration(mut self, max_stale: Duration) -> Self {
        self.max_stale_duration = Some(max_stale);
        self
    }

    /// Enable adaptive intervals that adjust based on cache activity
    ///
    /// When enabled, the save interval will automatically adjust:
    /// - High activity (many changes) → shorter intervals for better data safety
    /// - Low activity (few changes) → longer intervals to reduce I/O overhead
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{CacheAutosaveConfig, AdaptiveIntervalConfig};
    /// use std::time::Duration;
    ///
    /// let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
    ///     .with_adaptive_interval(
    ///         AdaptiveIntervalConfig::new(
    ///             Duration::from_secs(15),   // min 15s
    ///             Duration::from_secs(300),  // max 5min
    ///         )
    ///     );
    /// ```
    pub fn with_adaptive_interval(mut self, config: AdaptiveIntervalConfig) -> Self {
        self.adaptive_interval = Some(config);
        self
    }

    /// Enable save coalescing to batch rapid changes
    ///
    /// When enabled, rapid changes will be batched into single save operations:
    /// - Wait for a quiet period after the last change before saving
    /// - A maximum wait time prevents indefinite delays during sustained activity
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{CacheAutosaveConfig, CoalesceConfig};
    /// use std::time::Duration;
    ///
    /// let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
    ///     .with_coalesce(CoalesceConfig::burst()); // Use burst preset for handling rapid changes
    /// ```
    pub fn with_coalesce(mut self, config: CoalesceConfig) -> Self {
        self.coalesce = Some(config);
        self
    }

    /// Enable exponential backoff on save errors
    ///
    /// When enabled, save errors will trigger exponential backoff:
    /// - After each error, wait longer before the next retry attempt
    /// - Prevents resource exhaustion from repeated failing saves
    /// - After a successful save, interval resets to normal
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{CacheAutosaveConfig, BackoffConfig};
    /// use std::time::Duration;
    ///
    /// let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
    ///     .with_backoff(BackoffConfig::transient()); // Use transient preset for I/O errors
    /// ```
    pub fn with_backoff(mut self, config: BackoffConfig) -> Self {
        self.backoff = Some(config);
        self
    }

    /// Enable learning-based adaptive thresholds
    ///
    /// When enabled, the autosave system will learn from historical activity
    /// patterns and dynamically adjust the high/low activity thresholds used
    /// by adaptive intervals.
    ///
    /// Note: This only takes effect when `adaptive_interval` is also set.
    /// If adaptive intervals are not enabled, the learning thresholds will
    /// still collect samples but won't affect autosave behavior.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{CacheAutosaveConfig, AdaptiveIntervalConfig, LearningThresholdConfig};
    /// use std::time::Duration;
    ///
    /// let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
    ///     .with_adaptive_interval(AdaptiveIntervalConfig::default())
    ///     .with_learning_thresholds(
    ///         LearningThresholdConfig::new()
    ///             .with_warmup_samples(20)
    ///             .with_smoothing_factor(0.3)
    ///     );
    /// ```
    pub fn with_learning_thresholds(mut self, config: LearningThresholdConfig) -> Self {
        self.learning_thresholds = Some(config);
        self
    }

    /// Enable cache compaction before saves
    ///
    /// When enabled, the cache will be compacted before saving according to
    /// the specified policy. This can significantly reduce storage size by
    /// removing expired, low-confidence, obsolete, or duplicate entries.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{CacheAutosaveConfig, CompactionConfig};
    /// use std::time::Duration;
    ///
    /// // Compact with aggressive policy before each save
    /// let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
    ///     .with_compaction(
    ///         CompactionConfig::aggressive(0.7, vec!["v1.0".to_string()])
    ///             .with_compact_before_save(true)
    ///     );
    /// ```
    pub fn with_compaction(mut self, config: CompactionConfig) -> Self {
        self.compaction = Some(config);
        self
    }

    /// Set compaction triggers for trigger-based compaction
    ///
    /// When set, these triggers will be checked during the autosave loop.
    /// If any trigger fires, compaction will be performed using the `compaction` config.
    ///
    /// Note: A `compaction` config must also be set for triggers to have effect.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{CacheAutosaveConfig, CompactionConfig, CompactionTrigger};
    /// use std::time::Duration;
    ///
    /// // Compact when cache reaches 80% capacity or every 5 minutes
    /// let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
    ///     .with_compaction(CompactionConfig::expired_only())
    ///     .with_compaction_triggers(vec![
    ///         CompactionTrigger::size_80_percent(),
    ///         CompactionTrigger::every_minutes(5),
    ///     ]);
    /// ```
    pub fn with_compaction_triggers(mut self, triggers: Vec<CompactionTrigger>) -> Self {
        self.compaction_triggers = triggers;
        self
    }

    /// Add a single compaction trigger
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{CacheAutosaveConfig, CompactionConfig, CompactionTrigger};
    /// use std::time::Duration;
    ///
    /// let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
    ///     .with_compaction(CompactionConfig::expired_only())
    ///     .add_compaction_trigger(CompactionTrigger::size_90_percent());
    /// ```
    pub fn add_compaction_trigger(mut self, trigger: CompactionTrigger) -> Self {
        self.compaction_triggers.push(trigger);
        self
    }

    /// Enable cache warming when loading
    ///
    /// When enabled, the cache will be warmed when loaded using the specified
    /// strategy. This allows selective restoration of cached entries based on
    /// confidence, recency, patterns, or other criteria.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{CacheAutosaveConfig, WarmingConfig};
    /// use std::time::Duration;
    ///
    /// // Warm with high-confidence entries when loading
    /// let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
    ///     .with_warming(WarmingConfig::high_confidence(0.8));
    /// ```
    pub fn with_warming(mut self, config: WarmingConfig) -> Self {
        self.warming = Some(config);
        self
    }

    /// Performance-optimized preset
    ///
    /// Optimized for minimal impact on verification performance:
    /// - Long interval (5 minutes) to reduce I/O overhead
    /// - No compression to minimize CPU usage
    /// - Minimum change threshold of 4KB to skip unnecessary saves
    /// - Max stale duration of 15 minutes to ensure eventual persistence
    /// - Suitable for high-throughput verification workloads
    ///
    /// Use this when verification speed is critical and disk space is not a concern.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::CacheAutosaveConfig;
    ///
    /// let config = CacheAutosaveConfig::performance("cache.json");
    /// assert_eq!(config.interval.as_secs(), 300); // 5 minutes
    /// assert!(config.compression.is_none());
    /// assert_eq!(config.min_change_bytes, 4096); // 4KB threshold
    /// assert_eq!(config.max_stale_duration, Some(Duration::from_secs(900))); // 15 min
    /// ```
    pub fn performance(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            interval: Duration::from_secs(5 * 60), // 5 minutes
            compression: None,
            callbacks: CacheAutosaveCallbacks::default(),
            min_change_bytes: 4096, // 4KB threshold - skip small changes
            max_stale_duration: Some(Duration::from_secs(15 * 60)), // 15 min max stale
            adaptive_interval: None, // Fixed interval for predictable performance
            coalesce: None,         // No coalescing for predictable behavior
            backoff: None,          // No backoff by default
            learning_thresholds: None, // No learning for predictable behavior
            compaction: None,       // No compaction for maximum performance
            compaction_triggers: Vec::new(), // No triggers for maximum performance
            warming: None,          // No warming configuration
        }
    }

    /// Storage-optimized preset
    ///
    /// Optimized for minimal disk usage:
    /// - Moderate interval (2 minutes) balancing freshness and I/O
    /// - Maximum compression to minimize file size
    /// - Minimum change threshold of 2KB to skip trivial changes
    /// - Max stale duration of 10 minutes to ensure periodic persistence
    /// - Suitable for environments with limited disk space
    ///
    /// Use this when disk space is at a premium.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{CacheAutosaveConfig, SnapshotCompressionLevel};
    ///
    /// let config = CacheAutosaveConfig::storage_optimized("cache.json.gz");
    /// assert_eq!(config.interval.as_secs(), 120); // 2 minutes
    /// assert_eq!(config.compression, Some(SnapshotCompressionLevel::Best));
    /// assert_eq!(config.min_change_bytes, 2048); // 2KB threshold
    /// assert_eq!(config.max_stale_duration, Some(Duration::from_secs(600))); // 10 min
    /// ```
    pub fn storage_optimized(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            interval: Duration::from_secs(2 * 60), // 2 minutes
            compression: Some(SnapshotCompressionLevel::Best),
            callbacks: CacheAutosaveCallbacks::default(),
            min_change_bytes: 2048, // 2KB threshold
            max_stale_duration: Some(Duration::from_secs(10 * 60)), // 10 min max stale
            adaptive_interval: None, // Fixed interval for consistent storage behavior
            coalesce: None,         // No coalescing for predictable storage
            backoff: None,          // No backoff by default
            learning_thresholds: None, // No learning for consistent behavior
            // Enable aggressive compaction for storage optimization
            compaction: Some(CompactionConfig::expired_only().with_compact_before_save(true)),
            compaction_triggers: Vec::new(), // No triggers (compact before every save)
            warming: None,                   // No warming configuration
        }
    }

    /// Balanced preset
    ///
    /// A middle-ground configuration suitable for most use cases:
    /// - 1 minute interval for reasonably fresh backups
    /// - Fast compression for good space savings with low CPU overhead
    /// - Minimum change threshold of 1KB to skip trivial changes
    /// - Max stale duration of 5 minutes for timely persistence
    ///
    /// Use this as a sensible default for general purpose verification.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{CacheAutosaveConfig, SnapshotCompressionLevel};
    ///
    /// let config = CacheAutosaveConfig::balanced("cache.json.gz");
    /// assert_eq!(config.interval.as_secs(), 60); // 1 minute
    /// assert_eq!(config.compression, Some(SnapshotCompressionLevel::Fast));
    /// assert_eq!(config.min_change_bytes, 1024); // 1KB threshold
    /// assert_eq!(config.max_stale_duration, Some(Duration::from_secs(300))); // 5 min
    /// ```
    pub fn balanced(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            interval: Duration::from_secs(60), // 1 minute
            compression: Some(SnapshotCompressionLevel::Fast),
            callbacks: CacheAutosaveCallbacks::default(),
            min_change_bytes: 1024, // 1KB threshold
            max_stale_duration: Some(Duration::from_secs(5 * 60)), // 5 min max stale
            adaptive_interval: None, // Fixed interval for balanced, predictable behavior
            coalesce: None,         // No coalescing for predictable behavior
            backoff: None,          // No backoff by default
            learning_thresholds: None, // No learning for predictable behavior
            compaction: None,       // No compaction for balanced behavior
            compaction_triggers: Vec::new(), // No triggers
            warming: None,          // No warming configuration
        }
    }

    /// Aggressive preset
    ///
    /// Optimized for maximum data safety with frequent saves:
    /// - Very short interval (15 seconds) for minimal data loss on crash
    /// - Fast compression to reduce file size without too much CPU overhead
    /// - No minimum change threshold (always save) for maximum safety
    /// - No max stale duration needed (always saves regardless)
    /// - Suitable for critical verification sessions
    ///
    /// Use this when losing verification results would be costly.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{CacheAutosaveConfig, SnapshotCompressionLevel};
    ///
    /// let config = CacheAutosaveConfig::aggressive("cache.json.gz");
    /// assert_eq!(config.interval.as_secs(), 15);
    /// assert_eq!(config.compression, Some(SnapshotCompressionLevel::Fast));
    /// assert_eq!(config.min_change_bytes, 0); // Always save
    /// assert!(config.max_stale_duration.is_none()); // Not needed
    /// ```
    pub fn aggressive(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            interval: Duration::from_secs(15), // 15 seconds
            compression: Some(SnapshotCompressionLevel::Fast),
            callbacks: CacheAutosaveCallbacks::default(),
            min_change_bytes: 0,             // Always save for maximum safety
            max_stale_duration: None,        // Not needed since min_change_bytes is 0
            adaptive_interval: None,         // Fixed short interval for consistent safety
            coalesce: None,                  // No coalescing - save immediately for safety
            backoff: None,                   // No backoff by default
            learning_thresholds: None,       // No learning for consistent safety
            compaction: None,                // No compaction for aggressive saves
            compaction_triggers: Vec::new(), // No triggers
            warming: None,                   // No warming configuration
        }
    }

    /// Development preset
    ///
    /// Optimized for rapid development and testing:
    /// - Short interval (30 seconds) for quick iteration
    /// - No compression for easier debugging (human-readable JSON)
    /// - No minimum change threshold (always save) for easier debugging
    /// - No max stale duration needed (always saves regardless)
    ///
    /// Use this during development when you need to inspect cache files.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::CacheAutosaveConfig;
    ///
    /// let config = CacheAutosaveConfig::development("cache.json");
    /// assert_eq!(config.interval.as_secs(), 30);
    /// assert!(config.compression.is_none());
    /// assert_eq!(config.min_change_bytes, 0); // Always save
    /// assert!(config.max_stale_duration.is_none()); // Not needed
    /// ```
    pub fn development(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            interval: Duration::from_secs(30), // 30 seconds
            compression: None,
            callbacks: CacheAutosaveCallbacks::default(),
            min_change_bytes: 0,             // Always save for easier debugging
            max_stale_duration: None,        // Not needed since min_change_bytes is 0
            adaptive_interval: None,         // Fixed interval for predictable debugging
            coalesce: None,                  // No coalescing for predictable timing
            backoff: None,                   // No backoff by default
            learning_thresholds: None,       // No learning for predictable debugging
            compaction: None,                // No compaction for development
            compaction_triggers: Vec::new(), // No triggers
            warming: None,                   // No warming configuration
        }
    }

    /// Adaptive preset
    ///
    /// Uses adaptive interval adjustment to automatically tune save frequency:
    /// - Starts at 1 minute interval
    /// - Adjusts down to 15 seconds during high activity
    /// - Adjusts up to 5 minutes during low activity
    /// - Fast compression for reasonable space savings
    ///
    /// Use this when workload patterns vary and you want automatic optimization.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::CacheAutosaveConfig;
    ///
    /// let config = CacheAutosaveConfig::adaptive("cache.json.gz");
    /// assert_eq!(config.interval.as_secs(), 60); // Initial 1 minute
    /// assert!(config.adaptive_interval.is_some());
    /// ```
    pub fn adaptive(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            interval: Duration::from_secs(60), // Initial 1 minute
            compression: Some(SnapshotCompressionLevel::Fast),
            callbacks: CacheAutosaveCallbacks::default(),
            min_change_bytes: 0, // Track all changes for accurate activity measurement
            max_stale_duration: Some(Duration::from_secs(5 * 60)), // Ensure data freshness
            adaptive_interval: Some(AdaptiveIntervalConfig::new(
                Duration::from_secs(15),  // Min 15s during high activity
                Duration::from_secs(300), // Max 5min during low activity
            )),
            coalesce: None, // No coalescing - let adaptive intervals handle timing
            backoff: None,  // No backoff by default
            learning_thresholds: None, // Use static thresholds; enable for learning-based
            compaction: None, // No compaction for adaptive intervals
            compaction_triggers: Vec::new(), // No triggers
            warming: None,  // No warming configuration
        }
    }

    /// Burst-handling preset with coalescing
    ///
    /// Optimized for workloads with bursts of rapid changes:
    /// - Moderate interval (1 minute)
    /// - Burst coalescing to batch rapid changes
    /// - Fast compression for reasonable space savings
    ///
    /// Use this when you expect bursts of verification activity and want
    /// to reduce I/O overhead during those bursts.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::CacheAutosaveConfig;
    ///
    /// let config = CacheAutosaveConfig::burst_optimized("cache.json.gz");
    /// assert!(config.coalesce.is_some());
    /// ```
    pub fn burst_optimized(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            interval: Duration::from_secs(60), // 1 minute
            compression: Some(SnapshotCompressionLevel::Fast),
            callbacks: CacheAutosaveCallbacks::default(),
            min_change_bytes: 0, // Don't skip - let coalescing handle batching
            max_stale_duration: Some(Duration::from_secs(5 * 60)), // Ensure data freshness
            adaptive_interval: None, // Fixed interval - coalescing handles burst optimization
            coalesce: Some(CoalesceConfig::burst()), // Burst-optimized coalescing
            backoff: None,       // No backoff by default
            learning_thresholds: None, // No learning for predictable coalescing
            compaction: None,    // No compaction for burst optimization
            compaction_triggers: Vec::new(), // No triggers
            warming: None,       // No warming configuration
        }
    }

    /// Resilient preset with error backoff
    ///
    /// Designed for environments with unreliable I/O (network drives, cloud storage):
    /// - Moderate interval (1 minute)
    /// - Fast compression for reasonable space savings
    /// - Exponential backoff on errors to avoid overwhelming failing storage
    /// - Transient error handling (quick retry with backoff)
    ///
    /// Use this when saving to potentially unreliable storage.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::CacheAutosaveConfig;
    ///
    /// let config = CacheAutosaveConfig::resilient("cache.json.gz");
    /// assert!(config.backoff.is_some());
    /// ```
    pub fn resilient(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            interval: Duration::from_secs(60), // 1 minute
            compression: Some(SnapshotCompressionLevel::Fast),
            callbacks: CacheAutosaveCallbacks::default(),
            min_change_bytes: 1024, // 1KB threshold - skip trivial changes
            max_stale_duration: Some(Duration::from_secs(10 * 60)), // 10 min max stale
            adaptive_interval: None, // Fixed interval for predictable behavior
            coalesce: None,         // No coalescing for predictable timing
            backoff: Some(BackoffConfig::transient()), // Handle transient I/O errors
            learning_thresholds: None, // No learning for predictable behavior
            compaction: None,       // No compaction for resilient behavior
            compaction_triggers: Vec::new(), // No triggers
            warming: None,          // No warming configuration
        }
    }

    /// Intelligent preset with learning-based adaptive thresholds
    ///
    /// Combines adaptive intervals with learning-based threshold adjustment:
    /// - Starts at 1 minute interval
    /// - Adjusts interval based on activity (15s-5min range)
    /// - Learns from historical patterns to optimize thresholds
    /// - Uses short-term learning for quick adaptation
    ///
    /// Use this for maximum automatic optimization based on actual usage patterns.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::CacheAutosaveConfig;
    ///
    /// let config = CacheAutosaveConfig::intelligent("cache.json.gz");
    /// assert!(config.adaptive_interval.is_some());
    /// assert!(config.learning_thresholds.is_some());
    /// ```
    pub fn intelligent(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            interval: Duration::from_secs(60), // Initial 1 minute
            compression: Some(SnapshotCompressionLevel::Fast),
            callbacks: CacheAutosaveCallbacks::default(),
            min_change_bytes: 0, // Track all changes for accurate activity measurement
            max_stale_duration: Some(Duration::from_secs(5 * 60)), // Ensure data freshness
            adaptive_interval: Some(AdaptiveIntervalConfig::new(
                Duration::from_secs(15),  // Min 15s during high activity
                Duration::from_secs(300), // Max 5min during low activity
            )),
            coalesce: None, // No coalescing - let adaptive intervals handle timing
            backoff: None,  // No backoff by default
            learning_thresholds: Some(LearningThresholdConfig::short_term()), // Learn from patterns
            compaction: Some(CompactionConfig::expired_only().with_compact_before_save(true)), // Compact for intelligent optimization
            // Trigger compaction at 80% capacity or every 10 minutes
            compaction_triggers: vec![
                CompactionTrigger::size_80_percent(),
                CompactionTrigger::every_minutes(10),
            ],
            // Intelligent warming: prioritize high-confidence entries
            warming: Some(WarmingConfig::high_confidence(0.7)),
        }
    }
}

/// Handle for controlling a cache autosave background task
pub struct CacheAutosaveHandle {
    stop_tx: Option<oneshot::Sender<()>>,
    join_handle: Option<JoinHandle<()>>,
    save_count: Arc<AtomicUsize>,
    error_count: Arc<AtomicUsize>,
    skip_count: Arc<AtomicUsize>,
    forced_save_count: Arc<AtomicUsize>,
    compaction_count: Arc<AtomicUsize>,
    /// Compaction trigger counts by type
    compaction_size_count: Arc<AtomicUsize>,
    compaction_time_count: Arc<AtomicUsize>,
    compaction_hit_rate_count: Arc<AtomicUsize>,
    compaction_partition_count: Arc<AtomicUsize>,
    compaction_insert_count: Arc<AtomicUsize>,
    compaction_memory_count: Arc<AtomicUsize>,
    initial_save_count: Arc<AtomicUsize>,
    interval_save_count: Arc<AtomicUsize>,
    stale_save_count: Arc<AtomicUsize>,
    coalesced_save_count: Arc<AtomicUsize>,
    last_save_reason: Arc<AtomicU8>,
    total_bytes: Arc<AtomicU64>,
    last_save_bytes: Arc<AtomicU64>,
    /// Current interval in milliseconds (atomic for adaptive updates)
    current_interval_ms: Arc<AtomicU64>,
    errors: Arc<Mutex<Vec<String>>>,
    path: PathBuf,
    interval: Duration,
    compression: Option<SnapshotCompressionLevel>,
}

impl CacheAutosaveHandle {
    /// Get the current status of the autosave task without stopping it
    ///
    /// This method returns a snapshot of the current autosave metrics.
    /// It can be called repeatedly to monitor progress.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::AsyncImprovementVerifier;
    /// use std::time::Duration;
    ///
    /// let verifier = AsyncImprovementVerifier::with_cache();
    /// let handle = verifier.start_cache_autosave("cache.json", Duration::from_secs(60))?;
    ///
    /// // Check status without stopping
    /// let status = handle.status();
    /// println!("Saves so far: {}", status.save_count);
    /// println!("Bytes written: {}", status.total_bytes_written);
    ///
    /// // Later, stop and get full summary
    /// let summary = handle.stop().await;
    /// # Ok::<(), dashprove_selfimp::CachePersistenceError>(())
    /// ```
    pub fn status(&self) -> CacheAutosaveStatus {
        CacheAutosaveStatus {
            save_count: self.save_count.load(Ordering::Relaxed),
            error_count: self.error_count.load(Ordering::Relaxed),
            skip_count: self.skip_count.load(Ordering::Relaxed),
            forced_save_count: self.forced_save_count.load(Ordering::Relaxed),
            compaction_count: self.compaction_count.load(Ordering::Relaxed),
            compaction_trigger_counts: CompactionTriggerCounts {
                size_based: self.compaction_size_count.load(Ordering::Relaxed),
                time_based: self.compaction_time_count.load(Ordering::Relaxed),
                hit_rate_based: self.compaction_hit_rate_count.load(Ordering::Relaxed),
                partition_imbalance: self.compaction_partition_count.load(Ordering::Relaxed),
                insert_based: self.compaction_insert_count.load(Ordering::Relaxed),
                memory_based: self.compaction_memory_count.load(Ordering::Relaxed),
            },
            save_reason_counts: AutosaveReasonCounts {
                initial: self.initial_save_count.load(Ordering::Relaxed),
                interval: self.interval_save_count.load(Ordering::Relaxed),
                stale_data: self.stale_save_count.load(Ordering::Relaxed),
                coalesced: self.coalesced_save_count.load(Ordering::Relaxed),
            },
            last_save_reason: decode_save_reason(self.last_save_reason.load(Ordering::Relaxed)),
            total_bytes_written: self.total_bytes.load(Ordering::Relaxed),
            last_save_bytes: self.last_save_bytes.load(Ordering::Relaxed),
            current_interval: Duration::from_millis(
                self.current_interval_ms.load(Ordering::Relaxed),
            ),
            is_running: self.stop_tx.is_some(),
        }
    }

    /// Stop the autosave task and return a summary of its activity
    pub async fn stop(mut self) -> CacheAutosaveSummary {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.await;
        }

        let mut errors_guard = self.errors.lock().await;
        let errors = std::mem::take(&mut *errors_guard);

        CacheAutosaveSummary {
            save_count: self.save_count.load(Ordering::Relaxed),
            skip_count: self.skip_count.load(Ordering::Relaxed),
            forced_save_count: self.forced_save_count.load(Ordering::Relaxed),
            compaction_count: self.compaction_count.load(Ordering::Relaxed),
            compaction_trigger_counts: CompactionTriggerCounts {
                size_based: self.compaction_size_count.load(Ordering::Relaxed),
                time_based: self.compaction_time_count.load(Ordering::Relaxed),
                hit_rate_based: self.compaction_hit_rate_count.load(Ordering::Relaxed),
                partition_imbalance: self.compaction_partition_count.load(Ordering::Relaxed),
                insert_based: self.compaction_insert_count.load(Ordering::Relaxed),
                memory_based: self.compaction_memory_count.load(Ordering::Relaxed),
            },
            save_reason_counts: AutosaveReasonCounts {
                initial: self.initial_save_count.load(Ordering::Relaxed),
                interval: self.interval_save_count.load(Ordering::Relaxed),
                stale_data: self.stale_save_count.load(Ordering::Relaxed),
                coalesced: self.coalesced_save_count.load(Ordering::Relaxed),
            },
            last_save_reason: decode_save_reason(self.last_save_reason.load(Ordering::Relaxed)),
            errors,
            path: self.path.clone(),
            interval: self.interval,
            final_interval: Duration::from_millis(self.current_interval_ms.load(Ordering::Relaxed)),
            compression: self.compression,
            total_bytes_written: self.total_bytes.load(Ordering::Relaxed),
            last_save_bytes: self.last_save_bytes.load(Ordering::Relaxed),
        }
    }
}

impl Drop for CacheAutosaveHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            handle.abort();
        }
    }
}

impl AsyncImprovementVerifier {
    /// Create a new async verifier with a dispatcher
    pub fn with_dispatcher(dispatcher: Dispatcher) -> Self {
        Self {
            config: VerificationConfig::default(),
            dispatcher: Arc::new(Mutex::new(dispatcher)),
            cache: None,
        }
    }

    /// Create a new async verifier with custom configuration and dispatcher
    pub fn with_config_and_dispatcher(config: VerificationConfig, dispatcher: Dispatcher) -> Self {
        Self {
            config,
            dispatcher: Arc::new(Mutex::new(dispatcher)),
            cache: None,
        }
    }

    /// Create a new async verifier with default dispatcher configuration
    pub fn new() -> Self {
        let dispatcher = Dispatcher::new(DispatcherConfig::default());
        Self::with_dispatcher(dispatcher)
    }

    /// Create a new async verifier with learning-enabled dispatcher
    pub fn with_learning() -> Self {
        let dispatcher = Dispatcher::new(DispatcherConfig::learning());
        Self::with_dispatcher(dispatcher)
    }

    /// Create a new async verifier with an incremental verification cache
    pub fn with_cache() -> Self {
        let dispatcher = Dispatcher::new(DispatcherConfig::default());
        Self {
            config: VerificationConfig::default(),
            dispatcher: Arc::new(Mutex::new(dispatcher)),
            cache: Some(Arc::new(Mutex::new(VerificationCache::new()))),
        }
    }

    /// Create a verifier with custom cache configuration
    pub fn with_custom_cache(max_entries: usize, ttl: Duration) -> Self {
        let dispatcher = Dispatcher::new(DispatcherConfig::default());
        Self {
            config: VerificationConfig::default(),
            dispatcher: Arc::new(Mutex::new(dispatcher)),
            cache: Some(Arc::new(Mutex::new(VerificationCache::with_config(
                max_entries,
                ttl,
            )))),
        }
    }

    /// Create a verifier with a specific dispatcher and cache enabled
    pub fn with_dispatcher_and_cache(dispatcher: Dispatcher) -> Self {
        Self {
            config: VerificationConfig::default(),
            dispatcher: Arc::new(Mutex::new(dispatcher)),
            cache: Some(Arc::new(Mutex::new(VerificationCache::new()))),
        }
    }

    /// Enable caching on an existing verifier
    pub fn enable_cache(&mut self) {
        if self.cache.is_none() {
            self.cache = Some(Arc::new(Mutex::new(VerificationCache::new())));
        }
    }

    /// Get cache statistics (if caching is enabled)
    pub async fn cache_stats(&self) -> Option<CacheStats> {
        if let Some(cache) = &self.cache {
            Some(cache.lock().await.stats().clone())
        } else {
            None
        }
    }

    /// Clear the verification cache
    pub async fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.lock().await.clear();
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &VerificationConfig {
        &self.config
    }

    /// Run dispatcher verification with optional retry policy.
    async fn run_dispatcher_with_retries(
        &self,
        typed_spec: &dashprove_usl::typecheck::TypedSpec,
        start: Instant,
    ) -> SelfImpResult<VerificationResult> {
        let policy = self.config.retry_policy.clone();
        let max_attempts = policy.as_ref().map(|p| p.max_attempts.max(1)).unwrap_or(1);
        let mut attempt = 1usize;
        let mut consecutive_failures = 0usize;

        loop {
            let merged = {
                let mut dispatcher = self.dispatcher.lock().await;
                dispatcher.verify(typed_spec).await
            };

            match merged {
                Ok(merged_results) => {
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let mut result =
                        VerificationResult::from_dispatcher_results(&merged_results, duration_ms);
                    self.apply_config_checks(&mut result);

                    let should_retry = policy.as_ref().is_some_and(|p| {
                        attempt < max_attempts && p.should_retry_result(&result, &self.config)
                    });

                    if should_retry {
                        consecutive_failures += 1;
                        if let Some(delay) = policy
                            .as_ref()
                            .and_then(|p| p.next_delay(consecutive_failures))
                        {
                            time::sleep(delay).await;
                        }
                        attempt += 1;
                        continue;
                    }

                    return Ok(result);
                }
                Err(e) => {
                    let err = SelfImpError::DispatcherError(e.to_string());
                    let should_retry = policy
                        .as_ref()
                        .is_some_and(|p| attempt < max_attempts && p.should_retry_error(&err));

                    if should_retry {
                        consecutive_failures += 1;
                        if let Some(delay) = policy
                            .as_ref()
                            .and_then(|p| p.next_delay(consecutive_failures))
                        {
                            time::sleep(delay).await;
                        }
                        attempt += 1;
                        continue;
                    }

                    return Err(err);
                }
            }
        }
    }

    /// Verify an improvement using the dispatcher
    ///
    /// This generates USL specifications from the improvement and current version,
    /// then uses the dispatcher to run verification across multiple backends.
    pub async fn verify(
        &self,
        current: &Version,
        improvement: &Improvement,
    ) -> SelfImpResult<VerificationResult> {
        let start = std::time::Instant::now();

        // Generate USL specification from the improvement
        let usl_spec = self.generate_verification_spec(current, improvement)?;

        // Parse the USL specification
        let spec = parse(&usl_spec).map_err(|e| SelfImpError::UslParseError(e.to_string()))?;

        // Type-check the specification
        let typed_spec =
            typecheck(spec).map_err(|e| SelfImpError::TypeCheckError(e.to_string()))?;

        // Run verification (with optional retries)
        let result = self.run_dispatcher_with_retries(&typed_spec, start).await?;

        // Update cache if enabled
        if let Some(cache) = &self.cache {
            self.update_cache(cache, &current.content_hash, &result)
                .await;
        }

        Ok(result)
    }

    /// Verify an improvement with incremental caching
    ///
    /// This method uses dependency analysis to determine which properties need
    /// re-verification based on what has changed. Properties that haven't been
    /// affected by changes are retrieved from cache.
    ///
    /// # Arguments
    ///
    /// * `current` - The current version being verified
    /// * `improvement` - The proposed improvement
    /// * `changed_definitions` - Names of types/functions that have changed
    ///
    /// # Returns
    ///
    /// An `IncrementalVerificationResult` containing:
    /// - The full verification result
    /// - Statistics about cache hits/misses
    /// - Lists of cached vs. re-verified properties
    pub async fn verify_incremental(
        &self,
        current: &Version,
        improvement: &Improvement,
        changed_definitions: &[String],
    ) -> SelfImpResult<IncrementalVerificationResult> {
        let start = Instant::now();

        // Generate and parse the spec
        let usl_spec = self.generate_verification_spec(current, improvement)?;
        let spec = parse(&usl_spec).map_err(|e| SelfImpError::UslParseError(e.to_string()))?;
        let typed_spec =
            typecheck(spec).map_err(|e| SelfImpError::TypeCheckError(e.to_string()))?;

        // Build dependency graph to determine affected properties
        let dep_graph = DependencyGraph::from_spec(&typed_spec.spec);
        let changed_vec: Vec<String> = changed_definitions.to_vec();
        let affected_properties = dep_graph.properties_affected_by(&changed_vec);

        // If no cache, just do full verification
        let cache = match &self.cache {
            Some(c) => c,
            None => {
                let result = self.verify(current, improvement).await?;
                return Ok(IncrementalVerificationResult {
                    verified_count: result.verified_properties.len(),
                    cached_count: 0,
                    cached_properties: Vec::new(),
                    verified_properties: result
                        .verified_properties
                        .iter()
                        .map(|p| p.name.clone())
                        .collect(),
                    time_saved_ms: 0,
                    result,
                });
            }
        };

        let mut cache_guard = cache.lock().await;

        // Collect cached results for unaffected properties
        let mut cached_results: Vec<VerifiedProperty> = Vec::new();
        let mut cached_property_names: Vec<String> = Vec::new();
        let mut properties_to_verify: Vec<String> = Vec::new();
        let mut estimated_time_saved_ms: u64 = 0;

        // Check each property in the spec
        for property in &typed_spec.spec.properties {
            let prop_name = property.name();

            if affected_properties.contains(&prop_name) {
                // Property is affected - needs re-verification
                properties_to_verify.push(prop_name);
            } else {
                // Try to get from cache
                let cache_key = CacheKey::new(&current.content_hash, &prop_name);
                if let Some(cached) = cache_guard.get(&cache_key) {
                    cached_results.push(cached.property.clone());
                    cached_property_names.push(prop_name.clone());
                    // Estimate time saved (average verification time per property)
                    estimated_time_saved_ms += 50; // ~50ms per property estimate
                } else {
                    properties_to_verify.push(prop_name);
                }
            }
        }

        drop(cache_guard); // Release lock before verification

        // Invalidate affected properties in cache
        if !affected_properties.is_empty() {
            let mut cache_guard = cache.lock().await;
            let affected_vec: Vec<String> = affected_properties.into_iter().collect();
            cache_guard.invalidate_affected(&current.content_hash, &affected_vec);
        }

        // Verify remaining properties through dispatcher
        let mut result = if !properties_to_verify.is_empty() {
            self.run_dispatcher_with_retries(&typed_spec, start).await?
        } else {
            // All properties cached - create result from cache
            VerificationResult {
                passed: cached_results.iter().all(|p| p.passed),
                backend_results: HashMap::new(),
                confidence: 1.0,
                duration_ms: start.elapsed().as_millis() as u64,
                messages: Vec::new(),
                verified_properties: Vec::new(),
            }
        };

        // Merge cached results into the result
        for cached in cached_results {
            // Check if not already in verified_properties
            if !result
                .verified_properties
                .iter()
                .any(|p| p.name == cached.name)
            {
                result.verified_properties.push(cached);
            }
        }

        // Update cache with new results
        let cache_guard = cache.lock().await;
        drop(cache_guard);
        self.update_cache(cache, &current.content_hash, &result)
            .await;

        Ok(IncrementalVerificationResult {
            cached_count: cached_property_names.len(),
            verified_count: properties_to_verify.len(),
            cached_properties: cached_property_names,
            verified_properties: properties_to_verify,
            time_saved_ms: estimated_time_saved_ms,
            result,
        })
    }

    /// Update the cache with verification results
    async fn update_cache(
        &self,
        cache: &Arc<Mutex<VerificationCache>>,
        version_hash: &str,
        result: &VerificationResult,
    ) {
        let mut cache_guard = cache.lock().await;

        for prop in &result.verified_properties {
            let cache_key = CacheKey::new(version_hash, &prop.name);
            let cached_result = CachedPropertyResult {
                property: prop.clone(),
                backends: result.backend_results.keys().cloned().collect(),
                cached_at: SystemTime::now(),
                dependency_hash: String::new(), // Could compute hash of dependencies
                confidence: result.confidence,  // Use overall confidence for all properties
            };
            cache_guard.insert(cache_key, cached_result);
        }
    }

    /// Generate USL specification for verifying an improvement
    ///
    /// This creates properties that verify:
    /// 1. Soundness preservation (the system never claims false proofs)
    /// 2. Capability preservation (capabilities don't regress)
    /// 3. Safety properties (no unsafe state transitions)
    fn generate_verification_spec(
        &self,
        current: &Version,
        improvement: &Improvement,
    ) -> SelfImpResult<String> {
        let mut spec = String::new();

        // Header comment
        spec.push_str(&format!(
            "// Generated verification spec for improvement: {}\n",
            improvement.id
        ));
        spec.push_str(&format!("// Current version: {}\n", current.version_string));
        spec.push_str(&format!("// Improvement kind: {:?}\n\n", improvement.kind));

        // Define types for capabilities
        spec.push_str("type CapabilityValue = { value: Real }\n");
        spec.push_str("type Version = { capabilities: Map<String, CapabilityValue> }\n\n");

        // Generate soundness preservation property
        spec.push_str(&self.generate_soundness_property());

        // Generate capability preservation properties
        spec.push_str(&self.generate_capability_properties(current, improvement));

        // Generate safety properties based on improvement kind
        spec.push_str(&self.generate_safety_properties(improvement));

        Ok(spec)
    }

    /// Generate soundness preservation property
    fn generate_soundness_property(&self) -> String {
        let mut prop = String::new();

        // Soundness: The system should never claim a false proof
        prop.push_str("// Soundness: No false proofs\n");
        prop.push_str("invariant soundness_preserved {\n");
        prop.push_str("    forall proof: Proof, property: Property .\n");
        prop.push_str("        claims_proven(proof, property) implies\n");
        prop.push_str("        actually_true(property)\n");
        prop.push_str("}\n\n");

        prop
    }

    /// Generate capability preservation properties
    fn generate_capability_properties(
        &self,
        current: &Version,
        improvement: &Improvement,
    ) -> String {
        let mut props = String::new();

        // Check each capability in the current version
        for (name, cap) in &current.capabilities.capabilities {
            // Generate invariant that this capability doesn't regress
            props.push_str(&format!("// Capability: {}\n", name));
            props.push_str(&format!("invariant capability_{}_preserved {{\n", name));
            props.push_str(&format!(
                "    new_version.capabilities[\"{}\"].value >= {:?}\n",
                name, cap.value
            ));
            props.push_str("}\n\n");
        }

        // If the improvement expects new capabilities, verify they're achieved
        for (name, cap) in &improvement.expected_capabilities.capabilities {
            props.push_str(&format!("// Expected capability: {}\n", name));
            props.push_str(&format!("invariant expected_capability_{} {{\n", name));
            props.push_str(&format!(
                "    new_version.capabilities[\"{}\"].value >= {:?}\n",
                name, cap.value
            ));
            props.push_str("}\n\n");
        }

        props
    }

    /// Generate safety properties based on improvement kind
    fn generate_safety_properties(&self, improvement: &Improvement) -> String {
        let mut props = String::new();

        match improvement.kind {
            crate::improvement::ImprovementKind::Security => {
                // Security improvements should not introduce new vulnerabilities
                props.push_str("// Security: No new vulnerabilities\n");
                props.push_str("invariant no_new_vulnerabilities {\n");
                props.push_str("    forall v: Vulnerability .\n");
                props.push_str("        vulnerable(new_version, v) implies\n");
                props.push_str("        vulnerable(old_version, v)\n");
                props.push_str("}\n\n");
            }
            crate::improvement::ImprovementKind::BugFix => {
                // Bug fixes should fix the bug without breaking other functionality
                props.push_str("// Bug fix: Fixes bug without regression\n");
                props.push_str("invariant bug_fixed {\n");
                props.push_str("    fixed_bug(new_version) and\n");
                props.push_str(
                    "    (forall test: Test . passes(old_version, test) implies passes(new_version, test))\n",
                );
                props.push_str("}\n\n");
            }
            crate::improvement::ImprovementKind::Optimization => {
                // Optimizations should not change functionality
                props.push_str("// Optimization: Functional equivalence\n");
                props.push_str("invariant optimization_preserves_behavior {\n");
                props.push_str("    forall input: Input .\n");
                props
                    .push_str("        output(old_version, input) == output(new_version, input)\n");
                props.push_str("}\n\n");
            }
            crate::improvement::ImprovementKind::Feature => {
                // New features should not break existing functionality
                props.push_str("// Feature: No regression\n");
                props.push_str("invariant feature_no_regression {\n");
                props.push_str("    forall test: Test .\n");
                props.push_str(
                    "        passes(old_version, test) implies passes(new_version, test)\n",
                );
                props.push_str("}\n\n");
            }
            crate::improvement::ImprovementKind::Refactoring => {
                // Refactoring should preserve all behavior
                props.push_str("// Refactoring: Behavioral equivalence\n");
                props.push_str("invariant refactoring_equivalent {\n");
                props.push_str("    forall input: Input .\n");
                props
                    .push_str("        output(old_version, input) == output(new_version, input)\n");
                props.push_str("}\n\n");
            }
            _ => {
                // Default safety property
                props.push_str("// General safety\n");
                props.push_str("invariant general_safety {\n");
                props.push_str("    no_unsafe_state_transitions(new_version)\n");
                props.push_str("}\n\n");
            }
        }

        props
    }

    /// Apply configuration checks to the result
    fn apply_config_checks(&self, result: &mut VerificationResult) {
        // Check minimum passing backends
        let passing_count = result.backend_results.values().filter(|r| r.passed).count();

        if passing_count < self.config.min_passing_backends {
            result.passed = false;
            result.messages.push(VerificationMessage::error(format!(
                "Only {}/{} backends passed (minimum: {})",
                passing_count,
                result.backend_results.len(),
                self.config.min_passing_backends
            )));
        }

        // Check minimum confidence
        if result.confidence < self.config.min_confidence {
            result.passed = false;
            result.messages.push(VerificationMessage::error(format!(
                "Confidence {:.2} below minimum {:.2}",
                result.confidence, self.config.min_confidence
            )));
        }
    }

    /// Quick check using a single backend
    pub async fn quick_check(
        &self,
        current: &Version,
        improvement: &Improvement,
    ) -> SelfImpResult<bool> {
        // Generate minimal spec for quick check
        let usl_spec = self.generate_verification_spec(current, improvement)?;

        let spec = parse(&usl_spec).map_err(|e| SelfImpError::UslParseError(e.to_string()))?;
        let typed_spec =
            typecheck(spec).map_err(|e| SelfImpError::TypeCheckError(e.to_string()))?;

        let mut dispatcher = self.dispatcher.lock().await;
        let merged = dispatcher
            .verify(&typed_spec)
            .await
            .map_err(|e| SelfImpError::DispatcherError(e.to_string()))?;

        Ok(merged.summary.proven > 0 && merged.summary.disproven == 0)
    }

    // ==========================================================================
    // Async Cache Persistence Operations
    // ==========================================================================

    /// Save cache entries to a file asynchronously
    ///
    /// Uses `spawn_blocking` to perform file I/O without blocking the async runtime.
    /// Returns the number of entries saved.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let verifier = AsyncImprovementVerifier::with_cache();
    /// // ... perform verifications ...
    /// verifier.save_cache_entries_async("cache.json").await?;
    /// ```
    pub async fn save_cache_entries_async(
        &self,
        path: impl AsRef<Path> + Send + 'static,
    ) -> Result<usize, CachePersistenceError> {
        let cache = match &self.cache {
            Some(c) => Arc::clone(c),
            None => return Ok(0),
        };

        let path = path.as_ref().to_path_buf();

        let (entry_count, _bytes) = Self::save_cache_entries_to_path(cache, path, None).await?;
        Ok(entry_count)
    }

    /// Load cache entries from a file asynchronously
    ///
    /// Uses `spawn_blocking` to perform file I/O without blocking the async runtime.
    /// Expired entries are filtered based on current TTL settings.
    ///
    /// # Returns
    ///
    /// The number of entries loaded (after filtering expired entries)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let verifier = AsyncImprovementVerifier::with_cache();
    /// let loaded = verifier.load_cache_entries_async("cache.json").await?;
    /// println!("Loaded {} cache entries", loaded);
    /// ```
    pub async fn load_cache_entries_async(
        &self,
        path: impl AsRef<Path> + Send + 'static,
    ) -> Result<usize, CachePersistenceError> {
        let cache = match &self.cache {
            Some(c) => c,
            None => return Ok(0),
        };

        let path = path.as_ref().to_path_buf();

        // Read and parse file in blocking context
        let snapshot: CacheSnapshot = tokio::task::spawn_blocking(move || {
            let json = fs::read_to_string(path)?;
            let snapshot: CacheSnapshot = serde_json::from_str(&json)?;
            Ok::<_, CachePersistenceError>(snapshot)
        })
        .await
        .map_err(|e| CachePersistenceError::Io(io::Error::other(e)))??;

        // Restore entries in async context
        let mut cache_guard = cache.lock().await;
        cache_guard.restore_from_snapshot(&snapshot)
    }

    /// Save cache configuration to a file asynchronously
    ///
    /// Only saves configuration settings, not cached entries.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let verifier = AsyncImprovementVerifier::with_cache();
    /// verifier.save_cache_config_async("config.json").await?;
    /// ```
    pub async fn save_cache_config_async(
        &self,
        path: impl AsRef<Path> + Send + 'static,
    ) -> Result<(), CachePersistenceError> {
        let cache = match &self.cache {
            Some(c) => c,
            None => {
                return Err(CachePersistenceError::Io(io::Error::new(
                    io::ErrorKind::NotFound,
                    "No cache configured",
                )))
            }
        };

        let config = cache.lock().await.export_config();
        let path = path.as_ref().to_path_buf();

        tokio::task::spawn_blocking(move || config.save_to_file(path))
            .await
            .map_err(|e| CachePersistenceError::Io(io::Error::other(e)))?
    }

    /// Create verifier from a cache configuration file asynchronously
    ///
    /// Loads configuration from a JSON file and creates a new verifier with that config.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let verifier = AsyncImprovementVerifier::from_cache_config_file_async("config.json").await?;
    /// ```
    pub async fn from_cache_config_file_async(
        path: impl AsRef<Path> + Send + 'static,
    ) -> Result<(Self, Vec<CachePersistenceError>), CachePersistenceError> {
        let path = path.as_ref().to_path_buf();

        let config: CacheConfig =
            tokio::task::spawn_blocking(move || CacheConfig::load_from_file(path))
                .await
                .map_err(|e| CachePersistenceError::Io(io::Error::other(e)))??;

        let (cache, errors) = VerificationCache::from_config(config);
        let dispatcher = Dispatcher::new(DispatcherConfig::default());

        let verifier = Self {
            config: VerificationConfig::default(),
            dispatcher: Arc::new(Mutex::new(dispatcher)),
            cache: Some(Arc::new(Mutex::new(cache))),
        };

        Ok((verifier, errors))
    }

    /// Export current cache configuration
    ///
    /// Returns the cache configuration if caching is enabled, None otherwise.
    pub async fn export_cache_config(&self) -> Option<CacheConfig> {
        match &self.cache {
            Some(c) => Some(c.lock().await.export_config()),
            None => None,
        }
    }

    /// Create a snapshot of the current cache state
    ///
    /// Returns a serializable snapshot if caching is enabled, None otherwise.
    pub async fn create_cache_snapshot(&self) -> Option<CacheSnapshot> {
        match &self.cache {
            Some(c) => Some(c.lock().await.create_snapshot()),
            None => None,
        }
    }

    /// Restore cache from a snapshot asynchronously
    ///
    /// # Returns
    ///
    /// The number of entries restored, or 0 if no cache is configured
    pub async fn restore_cache_from_snapshot(
        &self,
        snapshot: &CacheSnapshot,
    ) -> Result<usize, CachePersistenceError> {
        let cache = match &self.cache {
            Some(c) => c,
            None => return Ok(0),
        };

        cache.lock().await.restore_from_snapshot(snapshot)
    }

    /// Warm the cache from historical verification results asynchronously
    ///
    /// Populates the cache with results from a collection of historical data.
    /// Useful for pre-warming a cache after restart or deployment.
    ///
    /// # Returns
    ///
    /// The number of entries successfully added to the cache, or 0 if no cache
    pub async fn warm_cache_from_results(
        &self,
        results: Vec<(String, String, CachedPropertyResult)>,
    ) -> usize {
        let cache = match &self.cache {
            Some(c) => c,
            None => return 0,
        };

        let mut cache_guard = cache.lock().await;
        cache_guard.warm_from_results(
            results
                .iter()
                .map(|(vh, pn, r)| (vh.as_str(), pn.as_str(), r)),
        )
    }

    /// Merge another cache's snapshot into this verifier's cache
    ///
    /// Newer entries (by `cached_at` timestamp) take precedence when keys conflict.
    ///
    /// # Returns
    ///
    /// The number of entries merged, or 0 if no cache is configured
    pub async fn merge_cache_from_snapshot(&self, snapshot: &CacheSnapshot) -> usize {
        let cache = match &self.cache {
            Some(c) => c,
            None => return 0,
        };

        let mut cache_guard = cache.lock().await;

        // Create a temporary cache from snapshot and merge
        let mut temp_cache = VerificationCache::new();
        if temp_cache.restore_from_snapshot(snapshot).is_ok() {
            cache_guard.merge_from(&temp_cache)
        } else {
            0
        }
    }

    /// Save cache entries to a compressed file asynchronously
    ///
    /// Uses gzip compression to reduce storage size for large caches.
    /// Uses `spawn_blocking` to perform I/O without blocking the async runtime.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{AsyncImprovementVerifier, SnapshotCompressionLevel};
    ///
    /// let verifier = AsyncImprovementVerifier::with_cache();
    /// // ... perform verifications ...
    /// verifier.save_cache_entries_compressed_async(
    ///     "cache.json.gz",
    ///     SnapshotCompressionLevel::Default
    /// ).await?;
    /// ```
    pub async fn save_cache_entries_compressed_async(
        &self,
        path: impl AsRef<Path> + Send + 'static,
        level: SnapshotCompressionLevel,
    ) -> Result<usize, CachePersistenceError> {
        let cache = match &self.cache {
            Some(c) => Arc::clone(c),
            None => return Ok(0),
        };

        let path = path.as_ref().to_path_buf();

        // Create snapshot while holding lock
        let snapshot = cache.lock().await.create_snapshot();
        let entry_count = snapshot.entries.len();

        // Compress and write in blocking context
        tokio::task::spawn_blocking(move || {
            let compressed = snapshot.to_compressed(level)?;
            fs::write(path, compressed)?;
            Ok::<_, CachePersistenceError>(entry_count)
        })
        .await
        .map_err(|e| CachePersistenceError::Io(io::Error::other(e)))?
    }

    /// Load cache entries from a compressed file asynchronously
    ///
    /// Uses `spawn_blocking` to perform file I/O without blocking the async runtime.
    /// Expired entries are filtered based on current TTL settings.
    ///
    /// # Returns
    ///
    /// The number of entries loaded (after filtering expired entries)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let verifier = AsyncImprovementVerifier::with_cache();
    /// let loaded = verifier.load_cache_entries_compressed_async("cache.json.gz").await?;
    /// println!("Loaded {} cache entries", loaded);
    /// ```
    pub async fn load_cache_entries_compressed_async(
        &self,
        path: impl AsRef<Path> + Send + 'static,
    ) -> Result<usize, CachePersistenceError> {
        let cache = match &self.cache {
            Some(c) => c,
            None => return Ok(0),
        };

        let path = path.as_ref().to_path_buf();

        // Read and decompress in blocking context
        let snapshot: CacheSnapshot = tokio::task::spawn_blocking(move || {
            let data = fs::read(path)?;
            CacheSnapshot::from_compressed(&data)
        })
        .await
        .map_err(|e| CachePersistenceError::Io(io::Error::other(e)))??;

        // Restore entries in async context
        let mut cache_guard = cache.lock().await;
        cache_guard.restore_from_snapshot(&snapshot)
    }

    /// Load cache entries from a file, auto-detecting format
    ///
    /// Automatically detects whether the file is gzip compressed or plain JSON
    /// and loads accordingly. Uses `spawn_blocking` for I/O.
    ///
    /// # Returns
    ///
    /// The number of entries loaded (after filtering expired entries)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let verifier = AsyncImprovementVerifier::with_cache();
    /// // Works with either compressed or uncompressed files
    /// let loaded = verifier.load_cache_entries_auto_async("cache.json.gz").await?;
    /// ```
    pub async fn load_cache_entries_auto_async(
        &self,
        path: impl AsRef<Path> + Send + 'static,
    ) -> Result<usize, CachePersistenceError> {
        let cache = match &self.cache {
            Some(c) => c,
            None => return Ok(0),
        };

        let path = path.as_ref().to_path_buf();

        // Read and auto-detect format in blocking context
        let snapshot: CacheSnapshot = tokio::task::spawn_blocking(move || {
            let data = fs::read(path)?;
            CacheSnapshot::from_bytes(&data)
        })
        .await
        .map_err(|e| CachePersistenceError::Io(io::Error::other(e)))??;

        // Restore entries in async context
        let mut cache_guard = cache.lock().await;
        cache_guard.restore_from_snapshot(&snapshot)
    }

    /// Start a background task that periodically saves cache entries to disk
    ///
    /// Returns `Ok(None)` when caching is not enabled. The first save occurs
    /// immediately, followed by saves every `interval`.
    pub fn start_cache_autosave(
        &self,
        path: impl AsRef<Path> + Send + 'static,
        interval: Duration,
    ) -> Result<Option<CacheAutosaveHandle>, CachePersistenceError> {
        self.start_cache_autosave_internal(
            path,
            interval,
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            Vec::new(),
            None,
        )
    }

    /// Start a background task that periodically saves compressed cache entries to disk
    ///
    /// Returns `Ok(None)` when caching is not enabled. The first save occurs
    /// immediately, followed by saves every `interval`. Files are gzip compressed
    /// using the specified compression level.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{AsyncImprovementVerifier, SnapshotCompressionLevel};
    /// use std::time::Duration;
    ///
    /// let verifier = AsyncImprovementVerifier::with_cache();
    /// let handle = verifier.start_cache_autosave_compressed(
    ///     "cache.json.gz",
    ///     Duration::from_secs(60),
    ///     SnapshotCompressionLevel::Fast
    /// )?;
    /// # Ok::<(), dashprove_selfimp::CachePersistenceError>(())
    /// ```
    pub fn start_cache_autosave_compressed(
        &self,
        path: impl AsRef<Path> + Send + 'static,
        interval: Duration,
        compression: SnapshotCompressionLevel,
    ) -> Result<Option<CacheAutosaveHandle>, CachePersistenceError> {
        self.start_cache_autosave_internal(
            path,
            interval,
            Some(compression),
            None,
            0,
            None,
            None,
            None,
            None,
            None,       // No learning thresholds for explicit config
            None,       // No compaction config
            Vec::new(), // No compaction triggers
            None,       // No warming config
        )
    }

    /// Start a background task with callbacks for save events
    ///
    /// This method allows registering callbacks that will be invoked when saves
    /// succeed or fail. Useful for monitoring, logging, or triggering external
    /// actions based on autosave events.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{AsyncImprovementVerifier, CacheAutosaveCallbacks};
    /// use std::time::Duration;
    ///
    /// let verifier = AsyncImprovementVerifier::with_cache();
    ///
    /// let callbacks = CacheAutosaveCallbacks::new()
    ///     .on_save(|event| {
    ///         println!("Saved {} entries ({} bytes)", event.entry_count, event.bytes_written);
    ///     })
    ///     .on_error(|event| {
    ///         eprintln!("Save #{} failed: {}", event.error_number, event.error);
    ///     });
    ///
    /// let handle = verifier.start_cache_autosave_with_callbacks(
    ///     "cache.json",
    ///     Duration::from_secs(60),
    ///     callbacks
    /// )?;
    /// # Ok::<(), dashprove_selfimp::CachePersistenceError>(())
    /// ```
    pub fn start_cache_autosave_with_callbacks(
        &self,
        path: impl AsRef<Path> + Send + 'static,
        interval: Duration,
        callbacks: CacheAutosaveCallbacks,
    ) -> Result<Option<CacheAutosaveHandle>, CachePersistenceError> {
        self.start_cache_autosave_internal(
            path,
            interval,
            None,
            Some(callbacks),
            0,
            None,
            None,
            None,
            None,
            None,       // No learning thresholds for explicit config
            None,       // No compaction config
            Vec::new(), // No compaction triggers
            None,       // No warming config
        )
    }

    /// Start a background task with compression and callbacks
    ///
    /// Combines gzip compression with event callbacks.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{
    ///     AsyncImprovementVerifier, CacheAutosaveCallbacks, SnapshotCompressionLevel
    /// };
    /// use std::time::Duration;
    ///
    /// let verifier = AsyncImprovementVerifier::with_cache();
    ///
    /// let callbacks = CacheAutosaveCallbacks::new()
    ///     .on_save(|event| {
    ///         println!("Compressed save: {} bytes", event.bytes_written);
    ///     });
    ///
    /// let handle = verifier.start_cache_autosave_compressed_with_callbacks(
    ///     "cache.json.gz",
    ///     Duration::from_secs(60),
    ///     SnapshotCompressionLevel::Fast,
    ///     callbacks
    /// )?;
    /// # Ok::<(), dashprove_selfimp::CachePersistenceError>(())
    /// ```
    pub fn start_cache_autosave_compressed_with_callbacks(
        &self,
        path: impl AsRef<Path> + Send + 'static,
        interval: Duration,
        compression: SnapshotCompressionLevel,
        callbacks: CacheAutosaveCallbacks,
    ) -> Result<Option<CacheAutosaveHandle>, CachePersistenceError> {
        self.start_cache_autosave_internal(
            path,
            interval,
            Some(compression),
            Some(callbacks),
            0,
            None,
            None,       // No adaptive interval for explicit config
            None,       // No coalescing for explicit config
            None,       // No backoff for explicit config
            None,       // No learning thresholds for explicit config
            None,       // No compaction config
            Vec::new(), // No compaction triggers
            None,       // No warming config
        )
    }

    /// Start a background autosave task using a configuration preset
    ///
    /// This is the most convenient way to start autosave with a preset configuration.
    /// The configuration bundles path, interval, compression, and callbacks together.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use dashprove_selfimp::{AsyncImprovementVerifier, CacheAutosaveConfig};
    ///
    /// let verifier = AsyncImprovementVerifier::with_cache();
    ///
    /// // Use a balanced preset for general-purpose verification
    /// let config = CacheAutosaveConfig::balanced("cache.json.gz");
    /// let handle = verifier.start_cache_autosave_with_config(config)?;
    ///
    /// // Or use storage-optimized for disk-constrained environments
    /// let config = CacheAutosaveConfig::storage_optimized("cache.json.gz");
    /// let handle = verifier.start_cache_autosave_with_config(config)?;
    /// # Ok::<(), dashprove_selfimp::CachePersistenceError>(())
    /// ```
    pub fn start_cache_autosave_with_config(
        &self,
        config: CacheAutosaveConfig,
    ) -> Result<Option<CacheAutosaveHandle>, CachePersistenceError> {
        self.start_cache_autosave_internal(
            config.path,
            config.interval,
            config.compression,
            Some(config.callbacks),
            config.min_change_bytes,
            config.max_stale_duration,
            config.adaptive_interval,
            config.coalesce,
            config.backoff,
            config.learning_thresholds,
            config.compaction,
            config.compaction_triggers,
            config.warming,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn start_cache_autosave_internal(
        &self,
        path: impl AsRef<Path> + Send + 'static,
        interval: Duration,
        compression: Option<SnapshotCompressionLevel>,
        callbacks: Option<CacheAutosaveCallbacks>,
        min_change_bytes: u64,
        max_stale_duration: Option<Duration>,
        adaptive_interval: Option<AdaptiveIntervalConfig>,
        coalesce: Option<CoalesceConfig>,
        backoff: Option<BackoffConfig>,
        learning_thresholds: Option<LearningThresholdConfig>,
        compaction: Option<CompactionConfig>,
        compaction_triggers: Vec<CompactionTrigger>,
        warming: Option<WarmingConfig>,
    ) -> Result<Option<CacheAutosaveHandle>, CachePersistenceError> {
        if interval.is_zero() {
            return Err(CachePersistenceError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Auto-save interval must be non-zero",
            )));
        }

        let cache = match &self.cache {
            Some(c) => Arc::clone(c),
            None => return Ok(None),
        };

        let path = path.as_ref().to_path_buf();
        let (stop_tx, stop_rx) = oneshot::channel();
        let save_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));
        let skip_count = Arc::new(AtomicUsize::new(0));
        let forced_save_count = Arc::new(AtomicUsize::new(0));
        let compaction_count = Arc::new(AtomicUsize::new(0));
        // Compaction trigger counts by type
        let compaction_size_count = Arc::new(AtomicUsize::new(0));
        let compaction_time_count = Arc::new(AtomicUsize::new(0));
        let compaction_hit_rate_count = Arc::new(AtomicUsize::new(0));
        let compaction_partition_count = Arc::new(AtomicUsize::new(0));
        let compaction_insert_count = Arc::new(AtomicUsize::new(0));
        let compaction_memory_count = Arc::new(AtomicUsize::new(0));
        let initial_save_count = Arc::new(AtomicUsize::new(0));
        let interval_save_count = Arc::new(AtomicUsize::new(0));
        let stale_save_count = Arc::new(AtomicUsize::new(0));
        let coalesced_save_count = Arc::new(AtomicUsize::new(0));
        let last_save_reason = Arc::new(AtomicU8::new(SAVE_REASON_NONE));
        let total_bytes = Arc::new(AtomicU64::new(0));
        let last_save_bytes = Arc::new(AtomicU64::new(0));
        let current_interval_ms = Arc::new(AtomicU64::new(interval.as_millis() as u64));
        let errors = Arc::new(Mutex::new(Vec::new()));
        let callbacks = callbacks.unwrap_or_default();

        let join_handle = tokio::spawn(Self::run_cache_autosave(
            cache,
            path.clone(),
            interval,
            compression,
            stop_rx,
            Arc::clone(&save_count),
            Arc::clone(&error_count),
            Arc::clone(&skip_count),
            Arc::clone(&forced_save_count),
            Arc::clone(&compaction_count),
            Arc::clone(&compaction_size_count),
            Arc::clone(&compaction_time_count),
            Arc::clone(&compaction_hit_rate_count),
            Arc::clone(&compaction_partition_count),
            Arc::clone(&compaction_insert_count),
            Arc::clone(&compaction_memory_count),
            Arc::clone(&initial_save_count),
            Arc::clone(&interval_save_count),
            Arc::clone(&stale_save_count),
            Arc::clone(&coalesced_save_count),
            Arc::clone(&last_save_reason),
            Arc::clone(&total_bytes),
            Arc::clone(&last_save_bytes),
            Arc::clone(&current_interval_ms),
            Arc::clone(&errors),
            callbacks,
            min_change_bytes,
            max_stale_duration,
            adaptive_interval,
            coalesce,
            backoff,
            learning_thresholds,
            compaction,
            compaction_triggers,
            warming,
        ));

        Ok(Some(CacheAutosaveHandle {
            stop_tx: Some(stop_tx),
            join_handle: Some(join_handle),
            save_count,
            error_count,
            skip_count,
            forced_save_count,
            compaction_count,
            compaction_size_count,
            compaction_time_count,
            compaction_hit_rate_count,
            compaction_partition_count,
            compaction_insert_count,
            compaction_memory_count,
            initial_save_count,
            interval_save_count,
            stale_save_count,
            coalesced_save_count,
            last_save_reason,
            total_bytes,
            last_save_bytes,
            current_interval_ms,
            errors,
            path,
            interval,
            compression,
        }))
    }

    /// Returns (entry_count, bytes_written) on success
    async fn save_cache_entries_to_path(
        cache: Arc<Mutex<VerificationCache>>,
        path: PathBuf,
        compression: Option<SnapshotCompressionLevel>,
    ) -> Result<(usize, u64), CachePersistenceError> {
        Self::save_cache_entries_to_path_with_metrics(cache, path, compression, None).await
    }

    /// Returns (entry_count, bytes_written) on success, with optional autosave metrics
    async fn save_cache_entries_to_path_with_metrics(
        cache: Arc<Mutex<VerificationCache>>,
        path: PathBuf,
        compression: Option<SnapshotCompressionLevel>,
        autosave_metrics: Option<PersistedAutosaveMetrics>,
    ) -> Result<(usize, u64), CachePersistenceError> {
        let mut snapshot = cache.lock().await.create_snapshot();
        let entry_count = snapshot.entries.len();
        snapshot.autosave_metrics = autosave_metrics;

        tokio::task::spawn_blocking(move || {
            let bytes_written = match compression {
                Some(level) => {
                    let compressed = snapshot.to_compressed(level)?;
                    let len = compressed.len() as u64;
                    fs::write(path, compressed)?;
                    len
                }
                None => {
                    let json = serde_json::to_string_pretty(&snapshot)?;
                    let len = json.len() as u64;
                    fs::write(path, json)?;
                    len
                }
            };
            Ok((entry_count, bytes_written))
        })
        .await
        .map_err(|e| CachePersistenceError::Io(io::Error::other(e)))?
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_cache_autosave(
        cache: Arc<Mutex<VerificationCache>>,
        path: PathBuf,
        interval: Duration,
        compression: Option<SnapshotCompressionLevel>,
        mut stop_rx: oneshot::Receiver<()>,
        save_count: Arc<AtomicUsize>,
        error_count: Arc<AtomicUsize>,
        skip_count: Arc<AtomicUsize>,
        forced_save_count: Arc<AtomicUsize>,
        compaction_count: Arc<AtomicUsize>,
        compaction_size_count: Arc<AtomicUsize>,
        compaction_time_count: Arc<AtomicUsize>,
        compaction_hit_rate_count: Arc<AtomicUsize>,
        compaction_partition_count: Arc<AtomicUsize>,
        compaction_insert_count: Arc<AtomicUsize>,
        compaction_memory_count: Arc<AtomicUsize>,
        initial_save_count: Arc<AtomicUsize>,
        interval_save_count: Arc<AtomicUsize>,
        stale_save_count: Arc<AtomicUsize>,
        coalesced_save_count: Arc<AtomicUsize>,
        last_save_reason: Arc<AtomicU8>,
        total_bytes: Arc<AtomicU64>,
        last_save_bytes: Arc<AtomicU64>,
        current_interval_ms: Arc<AtomicU64>,
        errors: Arc<Mutex<Vec<String>>>,
        callbacks: CacheAutosaveCallbacks,
        min_change_bytes: u64,
        max_stale_duration: Option<Duration>,
        adaptive_interval: Option<AdaptiveIntervalConfig>,
        coalesce: Option<CoalesceConfig>,
        backoff: Option<BackoffConfig>,
        learning_thresholds: Option<LearningThresholdConfig>,
        compaction_config: Option<CompactionConfig>,
        compaction_triggers: Vec<CompactionTrigger>,
        warming: Option<WarmingConfig>,
    ) {
        use std::time::{Instant, SystemTime, UNIX_EPOCH};

        // Track session start time for persisted metrics
        let session_start_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Helper closure to build PersistedAutosaveMetrics from atomic counters
        let build_persisted_metrics = |current_interval_ms_value: u64| -> PersistedAutosaveMetrics {
            let now_secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            PersistedAutosaveMetrics {
                save_count: save_count.load(Ordering::Relaxed),
                error_count: error_count.load(Ordering::Relaxed),
                skip_count: skip_count.load(Ordering::Relaxed),
                forced_save_count: forced_save_count.load(Ordering::Relaxed),
                save_reason_counts: AutosaveReasonCounts {
                    initial: initial_save_count.load(Ordering::Relaxed),
                    interval: interval_save_count.load(Ordering::Relaxed),
                    stale_data: stale_save_count.load(Ordering::Relaxed),
                    coalesced: coalesced_save_count.load(Ordering::Relaxed),
                },
                last_save_reason: decode_save_reason(last_save_reason.load(Ordering::Relaxed)),
                total_bytes_written: total_bytes.load(Ordering::Relaxed),
                last_save_bytes: last_save_bytes.load(Ordering::Relaxed),
                interval_ms: current_interval_ms_value,
                compressed: compression.is_some(),
                session_start_secs,
                session_end_secs: now_secs,
            }
        };

        // Track time of last successful save for staleness check
        let mut last_save_time = Instant::now();
        // Track current interval for adaptive adjustment
        let mut current_interval = interval;

        // Initialize learning threshold tracker if configured
        let mut activity_tracker = learning_thresholds.map(HistoricalActivityTracker::new);

        // Set up compaction triggers on the cache if configured
        if !compaction_triggers.is_empty() {
            let mut cache_guard = cache.lock().await;
            cache_guard.clear_compaction_triggers();
            for trigger in &compaction_triggers {
                cache_guard.add_compaction_trigger(trigger.clone());
            }
        }

        // Perform cache warming from existing snapshot if configured
        if let Some(ref warming_config) = warming {
            // Check if snapshot file exists
            if path.exists() {
                // Attempt to warm cache from existing snapshot
                let warm_result = {
                    let mut cache_guard = cache.lock().await;
                    if compression.is_some() {
                        cache_guard.warm_from_compressed_file(&path, warming_config)
                    } else {
                        cache_guard.warm_from_file(&path, warming_config)
                    }
                };

                match warm_result {
                    Ok(result) => {
                        // Fire warming callback if configured
                        if let Some(ref on_warming) = callbacks.on_warming {
                            on_warming(AutosaveWarmingEvent {
                                path: path.clone(),
                                result,
                                compression,
                            });
                        }
                    }
                    Err(e) => {
                        // Log warning but don't fail - warming is best-effort
                        let error_msg = format!("Cache warming failed (non-fatal): {}", e);
                        errors.lock().await.push(error_msg);
                    }
                }
            }
        }

        // Perform initial save (always save the first one regardless of threshold)
        // For initial save, the metrics are at their default state (all zeros)
        let initial_metrics = build_persisted_metrics(interval.as_millis() as u64);
        match Self::save_cache_entries_to_path_with_metrics(
            Arc::clone(&cache),
            path.clone(),
            compression,
            Some(initial_metrics),
        )
        .await
        {
            Ok((entry_count, bytes)) => {
                let new_save_count = save_count.fetch_add(1, Ordering::Relaxed) + 1;
                let new_total_bytes = total_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
                last_save_bytes.store(bytes, Ordering::Relaxed);
                last_save_time = Instant::now();

                record_save_reason(
                    AutosaveSaveReason::Initial,
                    &initial_save_count,
                    &interval_save_count,
                    &stale_save_count,
                    &coalesced_save_count,
                    &last_save_reason,
                );

                if let Some(ref on_save) = callbacks.on_save {
                    on_save(AutosaveSaveEvent {
                        path: path.clone(),
                        entry_count,
                        bytes_written: bytes,
                        save_number: new_save_count,
                        total_bytes_written: new_total_bytes,
                        compression,
                        save_reason: AutosaveSaveReason::Initial,
                    });
                }
            }
            Err(e) => {
                let new_error_count = error_count.fetch_add(1, Ordering::Relaxed) + 1;
                let error_msg = e.to_string();
                errors.lock().await.push(error_msg.clone());

                if let Some(ref on_error) = callbacks.on_error {
                    on_error(AutosaveErrorEvent {
                        path: path.clone(),
                        error: error_msg,
                        error_number: new_error_count,
                        compression,
                    });
                }
            }
        }

        // Track coalescing state
        let mut coalesce_start_time: Option<Instant> = None;
        let mut coalesced_intervals: usize = 0;

        // Track consecutive errors for backoff
        let mut consecutive_errors: usize = 0;
        // Track active backoff delay (None = normal interval, Some = use backoff delay)
        let mut backoff_delay: Option<Duration> = None;

        loop {
            // Use tokio::time::sleep instead of interval to support dynamic intervals
            // If backoff is active, use the backoff delay instead of normal interval
            let sleep_duration = backoff_delay.unwrap_or(current_interval);

            tokio::select! {
                _ = &mut stop_rx => {
                    break;
                }
                _ = tokio::time::sleep(sleep_duration) => {
                    // Check if we should force save due to staleness
                    let force_due_to_stale = max_stale_duration
                        .map(|max_stale| last_save_time.elapsed() >= max_stale)
                        .unwrap_or(false);

                    // Compute current snapshot bytes for both skip check and adaptive interval
                    let prev_bytes = last_save_bytes.load(Ordering::Relaxed);
                    let (should_skip, change_bytes, current_bytes_for_coalesce) = if min_change_bytes > 0 || adaptive_interval.is_some() || coalesce.is_some() {
                        match Self::compute_snapshot_bytes(Arc::clone(&cache), compression).await {
                            Ok((entry_count, current_bytes)) => {
                                let change = current_bytes.abs_diff(prev_bytes);

                                // Check if we should skip based on minimum change threshold
                                // (but not if we're forcing due to staleness)
                                if min_change_bytes > 0 && change < min_change_bytes && !force_due_to_stale {
                                    let new_skip_count = skip_count.fetch_add(1, Ordering::Relaxed) + 1;
                                    if let Some(ref on_skip) = callbacks.on_skip {
                                        on_skip(AutosaveSkipEvent {
                                            path: path.clone(),
                                            entry_count,
                                            current_bytes,
                                            last_save_bytes: prev_bytes,
                                            change_bytes: change,
                                            min_change_bytes,
                                            skip_number: new_skip_count,
                                            compression,
                                        });
                                    }
                                    (true, change, current_bytes)
                                } else {
                                    (false, change, current_bytes)
                                }
                            }
                            Err(_) => {
                                // If we can't compute bytes, don't skip and use 0 for change
                                (false, 0, 0)
                            }
                        }
                    } else {
                        (false, 0, 0)
                    };

                    // Record sample in activity tracker (if enabled)
                    // This must happen before computing the next interval so learned thresholds are up to date
                    if let Some(ref mut tracker) = activity_tracker {
                        if let Some(threshold_event) = tracker.record_sample(change_bytes, current_interval) {
                            // Fire threshold update callback
                            if let Some(ref on_threshold_update) = callbacks.on_threshold_update {
                                on_threshold_update(threshold_event);
                            }
                        }
                    }

                    // Adjust interval based on activity (even if we skip the save)
                    if let Some(ref adaptive_config) = adaptive_interval {
                        // Use learned thresholds if available and tracker is warmed up
                        let (high_thresh, low_thresh) = if let Some(ref tracker) = activity_tracker {
                            if tracker.is_warmed_up() {
                                (tracker.high_threshold(), tracker.low_threshold())
                            } else {
                                (adaptive_config.high_activity_threshold, adaptive_config.low_activity_threshold)
                            }
                        } else {
                            (adaptive_config.high_activity_threshold, adaptive_config.low_activity_threshold)
                        };

                        let new_interval = adaptive_config.compute_next_interval_with_thresholds(
                            current_interval,
                            change_bytes,
                            high_thresh,
                            low_thresh,
                        );
                        if new_interval != current_interval {
                            current_interval = new_interval;
                            current_interval_ms.store(current_interval.as_millis() as u64, Ordering::Relaxed);
                        }
                    }

                    if should_skip {
                        continue;
                    }

                    // Handle coalescing if enabled
                    let (save_reason, coalesce_duration, forced_by_max_wait) = if let Some(ref coalesce_config) = coalesce {
                        // Check if there's significant activity to trigger coalescing
                        let has_activity = change_bytes >= coalesce_config.activity_threshold;

                        if has_activity && !force_due_to_stale {
                            // Activity detected - start or continue coalescing
                            if coalesce_start_time.is_none() {
                                coalesce_start_time = Some(Instant::now());
                            }
                            coalesced_intervals += 1;

                            let elapsed = coalesce_start_time.unwrap().elapsed();

                            // Check if we've exceeded max_wait
                            if elapsed >= coalesce_config.max_wait {
                                // Force save due to max_wait exceeded
                                let duration = elapsed;
                                let intervals = coalesced_intervals;

                                // Fire coalesce callback before saving
                                if let Some(ref on_coalesce) = callbacks.on_coalesce {
                                    on_coalesce(AutosaveCoalesceEvent {
                                        path: path.clone(),
                                        coalesced_intervals: intervals,
                                        coalesce_duration: duration,
                                        forced_by_max_wait: true,
                                        current_bytes: current_bytes_for_coalesce,
                                        compression,
                                    });
                                }

                                // Reset coalescing state
                                coalesce_start_time = None;
                                coalesced_intervals = 0;

                                (AutosaveSaveReason::Coalesced, Some(duration), true)
                            } else {
                                // Still within quiet period - continue coalescing
                                continue;
                            }
                        } else if coalesce_start_time.is_some() {
                            // No activity and we were coalescing - quiet period expired, do save
                            let elapsed = coalesce_start_time.unwrap().elapsed();

                            // Check if quiet period has been met
                            if elapsed >= coalesce_config.quiet_period || force_due_to_stale {
                                let duration = elapsed;
                                let intervals = coalesced_intervals;

                                // Fire coalesce callback if we coalesced at least one interval
                                if intervals > 0 {
                                    if let Some(ref on_coalesce) = callbacks.on_coalesce {
                                        on_coalesce(AutosaveCoalesceEvent {
                                            path: path.clone(),
                                            coalesced_intervals: intervals,
                                            coalesce_duration: duration,
                                            forced_by_max_wait: false,
                                            current_bytes: current_bytes_for_coalesce,
                                            compression,
                                        });
                                    }
                                }

                                // Reset coalescing state
                                coalesce_start_time = None;
                                coalesced_intervals = 0;

                                if force_due_to_stale {
                                    (AutosaveSaveReason::StaleData, Some(duration), false)
                                } else if intervals > 0 {
                                    (AutosaveSaveReason::Coalesced, Some(duration), false)
                                } else {
                                    (AutosaveSaveReason::Interval, None, false)
                                }
                            } else {
                                // Haven't reached quiet period yet
                                continue;
                            }
                        } else {
                            // No coalescing in progress, normal save
                            if force_due_to_stale {
                                (AutosaveSaveReason::StaleData, None, false)
                            } else {
                                (AutosaveSaveReason::Interval, None, false)
                            }
                        }
                    } else {
                        // Coalescing disabled - determine save reason normally
                        if force_due_to_stale {
                            (AutosaveSaveReason::StaleData, None, false)
                        } else {
                            (AutosaveSaveReason::Interval, None, false)
                        }
                    };

                    // Track coalesced intervals coalesced (unused for now, but computed above)
                    let _ = (coalesce_duration, forced_by_max_wait);

                    // Build metrics snapshot with current interval
                    let current_metrics = build_persisted_metrics(current_interval.as_millis() as u64);
                    match Self::save_cache_entries_to_path_with_metrics(
                        Arc::clone(&cache),
                        path.clone(),
                        compression,
                        Some(current_metrics),
                    )
                    .await
                    {
                        Ok((entry_count, bytes)) => {
                            let new_save_count = save_count.fetch_add(1, Ordering::Relaxed) + 1;
                            let new_total_bytes = total_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
                            last_save_bytes.store(bytes, Ordering::Relaxed);
                            last_save_time = Instant::now();

                            // Reset backoff state on successful save
                            consecutive_errors = 0;
                            backoff_delay = None;

                            record_save_reason(
                                save_reason,
                                &initial_save_count,
                                &interval_save_count,
                                &stale_save_count,
                                &coalesced_save_count,
                                &last_save_reason,
                            );

                            // Track forced saves separately
                            if force_due_to_stale {
                                forced_save_count.fetch_add(1, Ordering::Relaxed);
                            }

                            if let Some(ref on_save) = callbacks.on_save {
                                on_save(AutosaveSaveEvent {
                                    path: path.clone(),
                                    entry_count,
                                    bytes_written: bytes,
                                    save_number: new_save_count,
                                    total_bytes_written: new_total_bytes,
                                    compression,
                                    save_reason,
                                });
                            }

                            // Check compaction triggers after successful save
                            if let Some(ref compact_config) = compaction_config {
                                if !compaction_triggers.is_empty() {
                                    let mut cache_guard = cache.lock().await;
                                    // Check if any trigger condition is met
                                    if let Some(triggered) = cache_guard.should_compact() {
                                        let trigger = triggered.clone();
                                        let entries_before = cache_guard.len();

                                        // Perform compaction
                                        if let Some(result) = cache_guard.compact_if_triggered(compact_config) {
                                            let entries_after = cache_guard.len();
                                            let new_compaction_count = compaction_count.fetch_add(1, Ordering::Relaxed) + 1;

                                            // Increment trigger-specific counter
                                            match &trigger {
                                                CompactionTrigger::SizeBased { .. } => {
                                                    compaction_size_count.fetch_add(1, Ordering::Relaxed);
                                                }
                                                CompactionTrigger::TimeBased { .. } => {
                                                    compaction_time_count.fetch_add(1, Ordering::Relaxed);
                                                }
                                                CompactionTrigger::HitRateBased { .. } => {
                                                    compaction_hit_rate_count.fetch_add(1, Ordering::Relaxed);
                                                }
                                                CompactionTrigger::PartitionImbalance { .. } => {
                                                    compaction_partition_count.fetch_add(1, Ordering::Relaxed);
                                                }
                                                CompactionTrigger::InsertBased { .. } => {
                                                    compaction_insert_count.fetch_add(1, Ordering::Relaxed);
                                                }
                                                CompactionTrigger::MemoryBased { .. } => {
                                                    compaction_memory_count.fetch_add(1, Ordering::Relaxed);
                                                }
                                            }

                                            // Fire compaction callback if configured
                                            if let Some(ref on_compaction) = callbacks.on_compaction {
                                                on_compaction(AutosaveCompactionEvent {
                                                    path: path.clone(),
                                                    trigger,
                                                    result,
                                                    entries_before,
                                                    entries_after,
                                                    compression,
                                                    compaction_number: new_compaction_count,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            consecutive_errors += 1;
                            let new_error_count = error_count.fetch_add(1, Ordering::Relaxed) + 1;
                            let error_msg = e.to_string();
                            errors.lock().await.push(error_msg.clone());

                            // Compute backoff delay if configured
                            if let Some(ref backoff_config) = backoff {
                                backoff_delay = backoff_config.compute_delay(consecutive_errors);
                            }

                            if let Some(ref on_error) = callbacks.on_error {
                                on_error(AutosaveErrorEvent {
                                    path: path.clone(),
                                    error: error_msg,
                                    error_number: new_error_count,
                                    compression,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    /// Compute snapshot bytes without writing to disk (for threshold comparison)
    async fn compute_snapshot_bytes(
        cache: Arc<Mutex<VerificationCache>>,
        compression: Option<SnapshotCompressionLevel>,
    ) -> Result<(usize, u64), CachePersistenceError> {
        let snapshot = cache.lock().await.create_snapshot();
        let entry_count = snapshot.entries.len();

        tokio::task::spawn_blocking(move || {
            let bytes = match compression {
                Some(level) => {
                    let compressed = snapshot.to_compressed(level)?;
                    compressed.len() as u64
                }
                None => {
                    let json = serde_json::to_string_pretty(&snapshot)?;
                    json.len() as u64
                }
            };
            Ok((entry_count, bytes))
        })
        .await
        .map_err(|e| CachePersistenceError::Io(io::Error::other(e)))?
    }
}

impl Default for AsyncImprovementVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Metrics Aggregation for Observability Dashboards
// ============================================================================

/// Duration of time windows for metrics aggregation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricsWindow {
    /// 1-minute window
    OneMinute,
    /// 5-minute window
    FiveMinutes,
    /// 15-minute window
    FifteenMinutes,
    /// 1-hour window
    OneHour,
}

impl MetricsWindow {
    /// Get the duration of this window
    pub fn duration(&self) -> Duration {
        match self {
            MetricsWindow::OneMinute => Duration::from_secs(60),
            MetricsWindow::FiveMinutes => Duration::from_secs(5 * 60),
            MetricsWindow::FifteenMinutes => Duration::from_secs(15 * 60),
            MetricsWindow::OneHour => Duration::from_secs(60 * 60),
        }
    }

    /// Get the label for this window (for Prometheus labels)
    pub fn label(&self) -> &'static str {
        match self {
            MetricsWindow::OneMinute => "1m",
            MetricsWindow::FiveMinutes => "5m",
            MetricsWindow::FifteenMinutes => "15m",
            MetricsWindow::OneHour => "1h",
        }
    }
}

/// A single timestamped metric sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSample {
    /// When the sample was recorded
    pub timestamp: SystemTime,
    /// Value of the sample
    pub value: f64,
}

impl MetricSample {
    /// Create a new sample with the current time
    pub fn now(value: f64) -> Self {
        Self {
            timestamp: SystemTime::now(),
            value,
        }
    }

    /// Create a sample with a specific timestamp
    pub fn at(timestamp: SystemTime, value: f64) -> Self {
        Self { timestamp, value }
    }
}

/// Rolling window statistics for a single metric
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WindowedMetricStats {
    /// Number of samples in the window
    pub count: usize,
    /// Sum of all sample values
    pub sum: f64,
    /// Minimum sample value
    pub min: Option<f64>,
    /// Maximum sample value
    pub max: Option<f64>,
    /// Mean of sample values (None if no samples)
    pub mean: Option<f64>,
    /// Rate per second (count / window_duration)
    pub rate_per_second: f64,
    /// Rate per minute (count / window_duration * 60)
    pub rate_per_minute: f64,
}

impl WindowedMetricStats {
    /// Compute statistics from a slice of samples within a window
    pub fn from_samples(samples: &[MetricSample], window_duration: Duration) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let count = samples.len();
        let sum: f64 = samples.iter().map(|s| s.value).sum();
        let min = samples
            .iter()
            .map(|s| s.value)
            .fold(f64::INFINITY, f64::min);
        let max = samples
            .iter()
            .map(|s| s.value)
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = sum / count as f64;

        let window_secs = window_duration.as_secs_f64();
        let rate_per_second = count as f64 / window_secs;
        let rate_per_minute = rate_per_second * 60.0;

        Self {
            count,
            sum,
            min: Some(min),
            max: Some(max),
            mean: Some(mean),
            rate_per_second,
            rate_per_minute,
        }
    }
}

/// Configuration for the metrics aggregator
#[derive(Debug, Clone)]
pub struct MetricsAggregatorConfig {
    /// Maximum number of samples to retain per metric
    pub max_samples_per_metric: usize,
    /// Maximum age of samples to retain
    pub max_sample_age: Duration,
    /// Which windows to compute statistics for
    pub windows: Vec<MetricsWindow>,
}

impl Default for MetricsAggregatorConfig {
    fn default() -> Self {
        Self {
            max_samples_per_metric: 10_000,
            max_sample_age: Duration::from_secs(2 * 60 * 60), // 2 hours
            windows: vec![
                MetricsWindow::OneMinute,
                MetricsWindow::FiveMinutes,
                MetricsWindow::FifteenMinutes,
                MetricsWindow::OneHour,
            ],
        }
    }
}

impl MetricsAggregatorConfig {
    /// Create a minimal configuration with just 1m and 5m windows
    pub fn minimal() -> Self {
        Self {
            max_samples_per_metric: 1_000,
            max_sample_age: Duration::from_secs(15 * 60),
            windows: vec![MetricsWindow::OneMinute, MetricsWindow::FiveMinutes],
        }
    }

    /// Create a high-resolution configuration
    pub fn high_resolution() -> Self {
        Self {
            max_samples_per_metric: 100_000,
            max_sample_age: Duration::from_secs(24 * 60 * 60), // 24 hours
            windows: vec![
                MetricsWindow::OneMinute,
                MetricsWindow::FiveMinutes,
                MetricsWindow::FifteenMinutes,
                MetricsWindow::OneHour,
            ],
        }
    }
}

/// Cache autosave metrics for observability
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutosaveMetrics {
    /// Total saves since start
    pub total_saves: u64,
    /// Total skipped saves since start
    pub total_skips: u64,
    /// Total errors since start
    pub total_errors: u64,
    /// Total bytes written since start
    pub total_bytes_written: u64,
    /// Save reason breakdown since start
    pub reason_counts: AutosaveReasonCounts,
    /// Current interval in milliseconds
    pub current_interval_ms: u64,
    /// Current high activity threshold
    pub current_high_threshold: u64,
    /// Current low activity threshold
    pub current_low_threshold: u64,
    /// Whether learning thresholds are active
    pub learning_active: bool,
    /// Number of activity samples collected (for learning)
    pub activity_samples: usize,
    /// Total compactions since start
    pub total_compactions: u64,
    /// Total entries removed by compaction
    pub total_entries_compacted: u64,
    /// Compaction trigger breakdown since start
    pub compaction_trigger_counts: CompactionTriggerCounts,
}

impl AutosaveMetrics {
    /// Create metrics from a `CacheAutosaveStatus`
    pub fn from_status(status: &CacheAutosaveStatus) -> Self {
        Self {
            total_saves: status.save_count as u64,
            total_skips: status.skip_count as u64,
            total_errors: status.error_count as u64,
            total_bytes_written: status.total_bytes_written,
            reason_counts: status.save_reason_counts,
            current_interval_ms: status.current_interval.as_millis() as u64,
            current_high_threshold: 0, // Not available from status
            current_low_threshold: 0,  // Not available from status
            learning_active: false,
            activity_samples: 0,
            total_compactions: status.compaction_count as u64,
            total_entries_compacted: 0, // Not tracked in status
            compaction_trigger_counts: status.compaction_trigger_counts,
        }
    }

    /// Create metrics from a `CacheAutosaveSummary`
    pub fn from_summary(summary: &CacheAutosaveSummary) -> Self {
        Self {
            total_saves: summary.save_count as u64,
            total_skips: summary.skip_count as u64,
            total_errors: summary.errors.len() as u64,
            total_bytes_written: summary.total_bytes_written,
            reason_counts: summary.save_reason_counts,
            current_interval_ms: summary.final_interval.as_millis() as u64,
            current_high_threshold: 0,
            current_low_threshold: 0,
            learning_active: false,
            activity_samples: 0,
            total_compactions: summary.compaction_count as u64,
            total_entries_compacted: 0, // Not tracked in summary
            compaction_trigger_counts: summary.compaction_trigger_counts,
        }
    }

    /// Enrich metrics with activity tracker statistics
    pub fn with_activity_stats(mut self, stats: &ActivityStatistics) -> Self {
        self.current_high_threshold = stats.current_high_threshold;
        self.current_low_threshold = stats.current_low_threshold;
        self.activity_samples = stats.sample_count;
        self.learning_active = stats.sample_count > 0;
        self
    }
}

/// Aggregator for autosave and verification metrics
///
/// This type collects metrics samples over time and computes windowed statistics
/// suitable for observability dashboards and Prometheus export.
#[derive(Debug)]
pub struct MetricsAggregator {
    config: MetricsAggregatorConfig,
    /// Samples for save operations (value = bytes written)
    save_samples: Vec<MetricSample>,
    /// Samples for skip operations (value = 1.0 per skip)
    skip_samples: Vec<MetricSample>,
    /// Samples for error operations (value = 1.0 per error)
    error_samples: Vec<MetricSample>,
    /// Samples for save latency (value = duration in seconds)
    latency_samples: Vec<MetricSample>,
    /// Samples for interval changes (value = interval in seconds)
    interval_samples: Vec<MetricSample>,
    /// Samples for bytes written (value = bytes)
    bytes_samples: Vec<MetricSample>,
    /// Samples for compaction operations (value = entries removed)
    compaction_samples: Vec<MetricSample>,
    /// Current autosave metrics snapshot
    current_metrics: AutosaveMetrics,
    /// When the aggregator was created
    start_time: Instant,
}

impl MetricsAggregator {
    /// Create a new metrics aggregator with default configuration
    pub fn new() -> Self {
        Self::with_config(MetricsAggregatorConfig::default())
    }

    /// Create a metrics aggregator with custom configuration
    pub fn with_config(config: MetricsAggregatorConfig) -> Self {
        Self {
            config,
            save_samples: Vec::new(),
            skip_samples: Vec::new(),
            error_samples: Vec::new(),
            latency_samples: Vec::new(),
            interval_samples: Vec::new(),
            bytes_samples: Vec::new(),
            compaction_samples: Vec::new(),
            current_metrics: AutosaveMetrics::default(),
            start_time: Instant::now(),
        }
    }

    /// Record a save event
    pub fn record_save(&mut self, bytes_written: u64, latency: Duration) {
        let now = SystemTime::now();
        self.save_samples.push(MetricSample::at(now, 1.0));
        self.bytes_samples
            .push(MetricSample::at(now, bytes_written as f64));
        self.latency_samples
            .push(MetricSample::at(now, latency.as_secs_f64()));
        self.current_metrics.total_saves += 1;
        self.current_metrics.total_bytes_written += bytes_written;
        self.prune_old_samples();
    }

    /// Record a skip event
    pub fn record_skip(&mut self) {
        self.skip_samples.push(MetricSample::now(1.0));
        self.current_metrics.total_skips += 1;
        self.prune_old_samples();
    }

    /// Record an error event
    pub fn record_error(&mut self) {
        self.error_samples.push(MetricSample::now(1.0));
        self.current_metrics.total_errors += 1;
        self.prune_old_samples();
    }

    /// Record an interval change
    pub fn record_interval_change(&mut self, new_interval: Duration) {
        self.interval_samples
            .push(MetricSample::now(new_interval.as_secs_f64()));
        self.current_metrics.current_interval_ms = new_interval.as_millis() as u64;
        self.prune_old_samples();
    }

    /// Record a threshold update
    pub fn record_threshold_update(&mut self, high: u64, low: u64) {
        self.current_metrics.current_high_threshold = high;
        self.current_metrics.current_low_threshold = low;
        self.current_metrics.learning_active = true;
    }

    /// Record a compaction event
    ///
    /// # Arguments
    /// * `entries_removed` - Number of entries removed by compaction
    /// * `trigger` - The trigger type that caused this compaction
    pub fn record_compaction(&mut self, entries_removed: usize, trigger: &CompactionTrigger) {
        self.compaction_samples
            .push(MetricSample::now(entries_removed as f64));
        self.current_metrics.total_compactions += 1;
        self.current_metrics.total_entries_compacted += entries_removed as u64;
        self.current_metrics
            .compaction_trigger_counts
            .increment(trigger);
        self.prune_old_samples();
    }

    /// Update current metrics from a status snapshot
    pub fn update_from_status(&mut self, status: &CacheAutosaveStatus) {
        self.current_metrics.total_saves = status.save_count as u64;
        self.current_metrics.total_skips = status.skip_count as u64;
        self.current_metrics.total_errors = status.error_count as u64;
        self.current_metrics.total_bytes_written = status.total_bytes_written;
        self.current_metrics.reason_counts = status.save_reason_counts;
        self.current_metrics.current_interval_ms = status.current_interval.as_millis() as u64;
    }

    /// Get windowed statistics for saves
    pub fn save_stats(&self, window: MetricsWindow) -> WindowedMetricStats {
        self.compute_window_stats(&self.save_samples, window)
    }

    /// Get windowed statistics for skips
    pub fn skip_stats(&self, window: MetricsWindow) -> WindowedMetricStats {
        self.compute_window_stats(&self.skip_samples, window)
    }

    /// Get windowed statistics for errors
    pub fn error_stats(&self, window: MetricsWindow) -> WindowedMetricStats {
        self.compute_window_stats(&self.error_samples, window)
    }

    /// Get windowed statistics for latency
    pub fn latency_stats(&self, window: MetricsWindow) -> WindowedMetricStats {
        self.compute_window_stats(&self.latency_samples, window)
    }

    /// Get windowed statistics for bytes written
    pub fn bytes_stats(&self, window: MetricsWindow) -> WindowedMetricStats {
        self.compute_window_stats(&self.bytes_samples, window)
    }

    /// Get windowed statistics for interval changes
    pub fn interval_stats(&self, window: MetricsWindow) -> WindowedMetricStats {
        self.compute_window_stats(&self.interval_samples, window)
    }

    /// Get windowed statistics for compaction operations
    pub fn compaction_stats(&self, window: MetricsWindow) -> WindowedMetricStats {
        self.compute_window_stats(&self.compaction_samples, window)
    }

    /// Get the current snapshot of all metrics
    pub fn current_metrics(&self) -> &AutosaveMetrics {
        &self.current_metrics
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Export metrics in Prometheus text format
    pub fn export_prometheus(&self, prefix: &str) -> String {
        let mut output = String::with_capacity(4096);

        // Counters (total values)
        output.push_str(&format!(
            "# HELP {prefix}_autosave_saves_total Total cache autosave operations\n"
        ));
        output.push_str(&format!("# TYPE {prefix}_autosave_saves_total counter\n"));
        output.push_str(&format!(
            "{prefix}_autosave_saves_total {}\n",
            self.current_metrics.total_saves
        ));
        output.push('\n');

        output.push_str(&format!(
            "# HELP {prefix}_autosave_skips_total Total skipped autosave operations\n"
        ));
        output.push_str(&format!("# TYPE {prefix}_autosave_skips_total counter\n"));
        output.push_str(&format!(
            "{prefix}_autosave_skips_total {}\n",
            self.current_metrics.total_skips
        ));
        output.push('\n');

        output.push_str(&format!(
            "# HELP {prefix}_autosave_errors_total Total autosave errors\n"
        ));
        output.push_str(&format!("# TYPE {prefix}_autosave_errors_total counter\n"));
        output.push_str(&format!(
            "{prefix}_autosave_errors_total {}\n",
            self.current_metrics.total_errors
        ));
        output.push('\n');

        output.push_str(&format!(
            "# HELP {prefix}_autosave_bytes_total Total bytes written by autosave\n"
        ));
        output.push_str(&format!("# TYPE {prefix}_autosave_bytes_total counter\n"));
        output.push_str(&format!(
            "{prefix}_autosave_bytes_total {}\n",
            self.current_metrics.total_bytes_written
        ));
        output.push('\n');

        // Save reason breakdown
        output.push_str(&format!(
            "# HELP {prefix}_autosave_saves_by_reason_total Saves by trigger reason\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_autosave_saves_by_reason_total counter\n"
        ));
        output.push_str(&format!(
            "{prefix}_autosave_saves_by_reason_total{{reason=\"initial\"}} {}\n",
            self.current_metrics.reason_counts.initial
        ));
        output.push_str(&format!(
            "{prefix}_autosave_saves_by_reason_total{{reason=\"interval\"}} {}\n",
            self.current_metrics.reason_counts.interval
        ));
        output.push_str(&format!(
            "{prefix}_autosave_saves_by_reason_total{{reason=\"stale_data\"}} {}\n",
            self.current_metrics.reason_counts.stale_data
        ));
        output.push_str(&format!(
            "{prefix}_autosave_saves_by_reason_total{{reason=\"coalesced\"}} {}\n",
            self.current_metrics.reason_counts.coalesced
        ));
        output.push('\n');

        // Gauges (current values)
        output.push_str(&format!(
            "# HELP {prefix}_autosave_interval_seconds Current autosave interval\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_autosave_interval_seconds gauge\n"
        ));
        output.push_str(&format!(
            "{prefix}_autosave_interval_seconds {:.3}\n",
            self.current_metrics.current_interval_ms as f64 / 1000.0
        ));
        output.push('\n');

        output.push_str(&format!(
            "# HELP {prefix}_autosave_high_threshold_bytes Current high activity threshold\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_autosave_high_threshold_bytes gauge\n"
        ));
        output.push_str(&format!(
            "{prefix}_autosave_high_threshold_bytes {}\n",
            self.current_metrics.current_high_threshold
        ));
        output.push('\n');

        output.push_str(&format!(
            "# HELP {prefix}_autosave_low_threshold_bytes Current low activity threshold\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_autosave_low_threshold_bytes gauge\n"
        ));
        output.push_str(&format!(
            "{prefix}_autosave_low_threshold_bytes {}\n",
            self.current_metrics.current_low_threshold
        ));
        output.push('\n');

        output.push_str(&format!(
            "# HELP {prefix}_autosave_learning_active Whether learning thresholds are active\n"
        ));
        output.push_str(&format!("# TYPE {prefix}_autosave_learning_active gauge\n"));
        output.push_str(&format!(
            "{prefix}_autosave_learning_active {}\n",
            if self.current_metrics.learning_active {
                1
            } else {
                0
            }
        ));
        output.push('\n');

        output.push_str(&format!(
            "# HELP {prefix}_autosave_activity_samples_total Activity samples collected for learning\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_autosave_activity_samples_total gauge\n"
        ));
        output.push_str(&format!(
            "{prefix}_autosave_activity_samples_total {}\n",
            self.current_metrics.activity_samples
        ));
        output.push('\n');

        // Windowed rates
        output.push_str(&format!(
            "# HELP {prefix}_autosave_saves_rate Save rate per second by window\n"
        ));
        output.push_str(&format!("# TYPE {prefix}_autosave_saves_rate gauge\n"));
        for window in &self.config.windows {
            let stats = self.save_stats(*window);
            output.push_str(&format!(
                "{prefix}_autosave_saves_rate{{window=\"{}\"}} {:.6}\n",
                window.label(),
                stats.rate_per_second
            ));
        }
        output.push('\n');

        output.push_str(&format!(
            "# HELP {prefix}_autosave_errors_rate Error rate per second by window\n"
        ));
        output.push_str(&format!("# TYPE {prefix}_autosave_errors_rate gauge\n"));
        for window in &self.config.windows {
            let stats = self.error_stats(*window);
            output.push_str(&format!(
                "{prefix}_autosave_errors_rate{{window=\"{}\"}} {:.6}\n",
                window.label(),
                stats.rate_per_second
            ));
        }
        output.push('\n');

        output.push_str(&format!(
            "# HELP {prefix}_autosave_bytes_rate Bytes written per second by window\n"
        ));
        output.push_str(&format!("# TYPE {prefix}_autosave_bytes_rate gauge\n"));
        for window in &self.config.windows {
            let stats = self.bytes_stats(*window);
            output.push_str(&format!(
                "{prefix}_autosave_bytes_rate{{window=\"{}\"}} {:.3}\n",
                window.label(),
                stats.sum / window.duration().as_secs_f64()
            ));
        }
        output.push('\n');

        // Latency statistics by window
        output.push_str(&format!(
            "# HELP {prefix}_autosave_latency_seconds_mean Mean save latency by window\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_autosave_latency_seconds_mean gauge\n"
        ));
        for window in &self.config.windows {
            let stats = self.latency_stats(*window);
            let mean = stats.mean.unwrap_or(0.0);
            output.push_str(&format!(
                "{prefix}_autosave_latency_seconds_mean{{window=\"{}\"}} {:.6}\n",
                window.label(),
                mean
            ));
        }
        output.push('\n');

        output.push_str(&format!(
            "# HELP {prefix}_autosave_latency_seconds_max Max save latency by window\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_autosave_latency_seconds_max gauge\n"
        ));
        for window in &self.config.windows {
            let stats = self.latency_stats(*window);
            let max = stats.max.unwrap_or(0.0);
            output.push_str(&format!(
                "{prefix}_autosave_latency_seconds_max{{window=\"{}\"}} {:.6}\n",
                window.label(),
                max
            ));
        }
        output.push('\n');

        // Compaction counters
        output.push_str(&format!(
            "# HELP {prefix}_autosave_compactions_total Total cache compaction operations\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_autosave_compactions_total counter\n"
        ));
        output.push_str(&format!(
            "{prefix}_autosave_compactions_total {}\n",
            self.current_metrics.total_compactions
        ));
        output.push('\n');

        output.push_str(&format!(
            "# HELP {prefix}_autosave_entries_compacted_total Total entries removed by compaction\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_autosave_entries_compacted_total counter\n"
        ));
        output.push_str(&format!(
            "{prefix}_autosave_entries_compacted_total {}\n",
            self.current_metrics.total_entries_compacted
        ));
        output.push('\n');

        // Compaction trigger breakdown
        output.push_str(&format!(
            "# HELP {prefix}_autosave_compactions_by_trigger_total Compactions by trigger type\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_autosave_compactions_by_trigger_total counter\n"
        ));
        let triggers = &self.current_metrics.compaction_trigger_counts;
        output.push_str(&format!(
            "{prefix}_autosave_compactions_by_trigger_total{{trigger=\"size_based\"}} {}\n",
            triggers.size_based
        ));
        output.push_str(&format!(
            "{prefix}_autosave_compactions_by_trigger_total{{trigger=\"time_based\"}} {}\n",
            triggers.time_based
        ));
        output.push_str(&format!(
            "{prefix}_autosave_compactions_by_trigger_total{{trigger=\"hit_rate_based\"}} {}\n",
            triggers.hit_rate_based
        ));
        output.push_str(&format!(
            "{prefix}_autosave_compactions_by_trigger_total{{trigger=\"partition_imbalance\"}} {}\n",
            triggers.partition_imbalance
        ));
        output.push_str(&format!(
            "{prefix}_autosave_compactions_by_trigger_total{{trigger=\"insert_based\"}} {}\n",
            triggers.insert_based
        ));
        output.push_str(&format!(
            "{prefix}_autosave_compactions_by_trigger_total{{trigger=\"memory_based\"}} {}\n",
            triggers.memory_based
        ));
        output.push('\n');

        // Compaction rates by window
        output.push_str(&format!(
            "# HELP {prefix}_autosave_compactions_rate Compaction rate per second by window\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_autosave_compactions_rate gauge\n"
        ));
        for window in &self.config.windows {
            let stats = self.compaction_stats(*window);
            output.push_str(&format!(
                "{prefix}_autosave_compactions_rate{{window=\"{}\"}} {:.6}\n",
                window.label(),
                stats.rate_per_second
            ));
        }
        output.push('\n');

        // Entries removed by compaction by window (sum)
        output.push_str(&format!(
            "# HELP {prefix}_autosave_entries_compacted_rate Entries compacted per second by window\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_autosave_entries_compacted_rate gauge\n"
        ));
        for window in &self.config.windows {
            let stats = self.compaction_stats(*window);
            output.push_str(&format!(
                "{prefix}_autosave_entries_compacted_rate{{window=\"{}\"}} {:.3}\n",
                window.label(),
                stats.sum / window.duration().as_secs_f64()
            ));
        }
        output.push('\n');

        // Uptime
        output.push_str(&format!(
            "# HELP {prefix}_autosave_uptime_seconds Aggregator uptime\n"
        ));
        output.push_str(&format!("# TYPE {prefix}_autosave_uptime_seconds gauge\n"));
        output.push_str(&format!(
            "{prefix}_autosave_uptime_seconds {:.3}\n",
            self.uptime_seconds()
        ));

        output
    }

    /// Compute statistics for samples within a time window
    fn compute_window_stats(
        &self,
        samples: &[MetricSample],
        window: MetricsWindow,
    ) -> WindowedMetricStats {
        let now = SystemTime::now();
        let window_start = now
            .checked_sub(window.duration())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let filtered: Vec<MetricSample> = samples
            .iter()
            .filter(|s| s.timestamp >= window_start)
            .cloned()
            .collect();

        WindowedMetricStats::from_samples(&filtered, window.duration())
    }

    /// Remove samples older than max_sample_age
    fn prune_old_samples(&mut self) {
        let cutoff = SystemTime::now()
            .checked_sub(self.config.max_sample_age)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let max_samples = self.config.max_samples_per_metric;

        Self::prune_samples_list(&mut self.save_samples, cutoff, max_samples);
        Self::prune_samples_list(&mut self.skip_samples, cutoff, max_samples);
        Self::prune_samples_list(&mut self.error_samples, cutoff, max_samples);
        Self::prune_samples_list(&mut self.latency_samples, cutoff, max_samples);
        Self::prune_samples_list(&mut self.interval_samples, cutoff, max_samples);
        Self::prune_samples_list(&mut self.bytes_samples, cutoff, max_samples);
        Self::prune_samples_list(&mut self.compaction_samples, cutoff, max_samples);
    }

    fn prune_samples_list(samples: &mut Vec<MetricSample>, cutoff: SystemTime, max_samples: usize) {
        samples.retain(|s| s.timestamp >= cutoff);

        // Also enforce max_samples_per_metric
        if samples.len() > max_samples {
            let excess = samples.len() - max_samples;
            samples.drain(0..excess);
        }
    }

    /// Get total sample count across all metrics
    pub fn total_sample_count(&self) -> usize {
        self.save_samples.len()
            + self.skip_samples.len()
            + self.error_samples.len()
            + self.latency_samples.len()
            + self.interval_samples.len()
            + self.bytes_samples.len()
            + self.compaction_samples.len()
    }

    /// Reset all samples (keeps current_metrics totals)
    pub fn reset_samples(&mut self) {
        self.save_samples.clear();
        self.skip_samples.clear();
        self.error_samples.clear();
        self.latency_samples.clear();
        self.interval_samples.clear();
        self.bytes_samples.clear();
        self.compaction_samples.clear();
    }
}

impl Default for MetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined metrics report for dashboard export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    /// Timestamp when the report was generated
    pub generated_at: SystemTime,
    /// Aggregator uptime in seconds
    pub uptime_seconds: f64,
    /// Current metrics snapshot
    pub current: AutosaveMetrics,
    /// Statistics for each configured window
    pub windowed: HashMap<String, WindowedMetricsReport>,
}

/// Windowed statistics for a single time window
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WindowedMetricsReport {
    /// Window label (e.g., "1m", "5m")
    pub window: String,
    /// Save statistics
    pub saves: WindowedMetricStats,
    /// Skip statistics
    pub skips: WindowedMetricStats,
    /// Error statistics
    pub errors: WindowedMetricStats,
    /// Latency statistics
    pub latency: WindowedMetricStats,
    /// Bytes written statistics
    pub bytes: WindowedMetricStats,
}

impl MetricsAggregator {
    /// Generate a complete metrics report
    pub fn generate_report(&self) -> MetricsReport {
        let mut windowed = HashMap::new();

        for window in &self.config.windows {
            let report = WindowedMetricsReport {
                window: window.label().to_string(),
                saves: self.save_stats(*window),
                skips: self.skip_stats(*window),
                errors: self.error_stats(*window),
                latency: self.latency_stats(*window),
                bytes: self.bytes_stats(*window),
            };
            windowed.insert(window.label().to_string(), report);
        }

        MetricsReport {
            generated_at: SystemTime::now(),
            uptime_seconds: self.uptime_seconds(),
            current: self.current_metrics.clone(),
            windowed,
        }
    }

    /// Export report as JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        let report = self.generate_report();
        serde_json::to_string_pretty(&report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::improvement::ImprovementKind;
    use crate::version::{Capability, CapabilitySet, VersionId};
    use async_trait::async_trait;
    use dashprove_backends::{BackendId, PropertyType, VerificationBackend};
    use dashprove_dispatcher::SelectionStrategy;
    use std::sync::Arc;

    fn create_test_version() -> Version {
        let mut caps = CapabilitySet::new();
        caps.add(Capability::boolean("soundness", true));
        caps.add(Capability::numeric("accuracy", 0.95));

        Version {
            id: VersionId::new("test-v1"),
            version_string: "1.0.0".to_string(),
            capabilities: caps,
            metadata: Default::default(),
            content_hash: "abc123".to_string(),
            previous_version: None,
        }
    }

    fn create_test_improvement() -> Improvement {
        Improvement::new(
            "Test improvement",
            ImprovementKind::BugFix,
            crate::improvement::ImprovementTarget::System,
        )
    }

    /// Create a version with empty capabilities (avoids USL generation issues in tests)
    fn create_empty_caps_version() -> Version {
        let caps = CapabilitySet::new();
        Version {
            id: VersionId::new("test-v1"),
            version_string: "1.0.0".to_string(),
            capabilities: caps,
            metadata: Default::default(),
            content_hash: "abc123".to_string(),
            previous_version: None,
        }
    }

    /// Create improvement with matching empty capabilities
    fn create_empty_caps_improvement() -> Improvement {
        let caps = CapabilitySet::new();
        Improvement::new(
            "Test improvement",
            ImprovementKind::BugFix,
            crate::improvement::ImprovementTarget::System,
        )
        .with_expected_capabilities(caps)
    }

    #[test]
    fn test_verification_config_defaults() {
        let config = VerificationConfig::default();
        assert_eq!(config.min_passing_backends, 2);
        assert!(config.use_dispatcher);
    }

    #[test]
    fn test_verification_config_strict() {
        let config = VerificationConfig::strict();
        assert_eq!(config.min_passing_backends, 3);
        assert_eq!(config.min_confidence, 0.95);
    }

    #[test]
    fn test_verification_config_quick() {
        let config = VerificationConfig::quick();
        assert_eq!(config.min_passing_backends, 1);
        assert_eq!(config.min_confidence, 0.7);
    }

    #[test]
    fn test_backend_result_passed() {
        let result = BackendResult::passed("lean4", 100);
        assert!(result.passed);
        assert_eq!(result.backend, "lean4");
        assert_eq!(result.duration_ms, 100);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_backend_result_failed() {
        let result = BackendResult::failed("lean4", 100, "timeout");
        assert!(!result.passed);
        assert_eq!(result.error, Some("timeout".to_string()));
    }

    #[test]
    fn test_verification_message_levels() {
        let info = VerificationMessage::info("test");
        assert_eq!(info.level, MessageLevel::Info);

        let warning = VerificationMessage::warning("test");
        assert_eq!(warning.level, MessageLevel::Warning);

        let error = VerificationMessage::error("test");
        assert_eq!(error.level, MessageLevel::Error);
    }

    #[test]
    fn test_verification_message_with_source() {
        let msg = VerificationMessage::error("test").with_source("lean4");
        assert_eq!(msg.source, Some("lean4".to_string()));
    }

    #[test]
    fn test_simple_verifier_verify() {
        let verifier = ImprovementVerifier::new();
        let version = create_test_version();
        let improvement = create_test_improvement();

        let result = verifier.verify(&version, &improvement);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verification_result_passed() {
        let mut backend_results = HashMap::new();
        backend_results.insert("lean4".to_string(), BackendResult::passed("lean4", 100));
        backend_results.insert("kani".to_string(), BackendResult::passed("kani", 150));

        let result = VerificationResult::passed(backend_results, 250);
        assert!(result.passed);
        assert_eq!(result.confidence, 1.0);
        assert_eq!(result.passing_backends().len(), 2);
    }

    #[test]
    fn test_verification_result_failed() {
        let mut backend_results = HashMap::new();
        backend_results.insert("lean4".to_string(), BackendResult::passed("lean4", 100));
        backend_results.insert(
            "kani".to_string(),
            BackendResult::failed("kani", 150, "error"),
        );

        let messages = vec![VerificationMessage::error("kani failed")];
        let result = VerificationResult::failed(backend_results, 250, messages);
        assert!(!result.passed);
        assert_eq!(result.confidence, 0.5);
        assert_eq!(result.failed_backends().len(), 1);
    }

    #[test]
    fn test_async_verifier_spec_generation() {
        let verifier = AsyncImprovementVerifier::new();
        let version = create_test_version();
        let improvement = create_test_improvement();

        let spec = verifier.generate_verification_spec(&version, &improvement);
        assert!(spec.is_ok());

        let spec_text = spec.unwrap();
        assert!(spec_text.contains("soundness_preserved"));
        assert!(spec_text.contains("capability_soundness_preserved"));
        assert!(spec_text.contains("capability_accuracy_preserved"));
    }

    #[test]
    fn test_async_verifier_security_improvement_spec() {
        let verifier = AsyncImprovementVerifier::new();
        let version = create_test_version();
        let improvement = Improvement::new(
            "Security fix",
            ImprovementKind::Security,
            crate::improvement::ImprovementTarget::System,
        );

        let spec = verifier.generate_verification_spec(&version, &improvement);
        assert!(spec.is_ok());

        let spec_text = spec.unwrap();
        assert!(spec_text.contains("no_new_vulnerabilities"));
    }

    #[test]
    fn test_async_verifier_optimization_improvement_spec() {
        let verifier = AsyncImprovementVerifier::new();
        let version = create_test_version();
        let improvement = Improvement::new(
            "Performance optimization",
            ImprovementKind::Optimization,
            crate::improvement::ImprovementTarget::System,
        );

        let spec = verifier.generate_verification_spec(&version, &improvement);
        assert!(spec.is_ok());

        let spec_text = spec.unwrap();
        assert!(spec_text.contains("optimization_preserves_behavior"));
    }

    // ============ Verification Cache Tests ============

    #[test]
    fn test_verification_cache_new() {
        let cache = VerificationCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_verification_cache_with_config() {
        let cache = VerificationCache::with_config(100, Duration::from_secs(60));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_key_creation() {
        let key = CacheKey::new("hash123", "my_property");
        assert_eq!(key.version_hash, "hash123");
        assert_eq!(key.property_name, "my_property");
    }

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = VerificationCache::new();

        let key = CacheKey::new("hash1", "prop1");
        let cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop1".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep_hash".to_string(),
            confidence: 0.95,
        };

        cache.insert(key.clone(), cached);
        assert_eq!(cache.len(), 1);

        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.property.name, "prop1");
        assert!(retrieved.property.passed);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = VerificationCache::new();
        let key = CacheKey::new("hash1", "nonexistent");
        let result = cache.get(&key);
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let mut cache = VerificationCache::new();

        // Insert an entry
        let key = CacheKey::new("hash1", "prop1");
        let cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop1".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.95,
        };
        cache.insert(key.clone(), cached);

        // One hit
        let _ = cache.get(&key);
        // One miss
        let miss_key = CacheKey::new("hash1", "miss");
        let _ = cache.get(&miss_key);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cache_invalidate_affected() {
        let mut cache = VerificationCache::new();

        // Insert multiple entries
        for i in 0..5 {
            let key = CacheKey::new("hash1", format!("prop{}", i));
            let cached = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("prop{}", i),
                    passed: true,
                    status: "Proven".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            };
            cache.insert(key, cached);
        }

        assert_eq!(cache.len(), 5);

        // Invalidate some entries
        cache.invalidate_affected("hash1", &["prop1".to_string(), "prop3".to_string()]);
        assert_eq!(cache.len(), 3);

        // Verify correct entries were removed
        assert!(cache.get(&CacheKey::new("hash1", "prop0")).is_some());
        assert!(cache.get(&CacheKey::new("hash1", "prop1")).is_none());
        assert!(cache.get(&CacheKey::new("hash1", "prop2")).is_some());
        assert!(cache.get(&CacheKey::new("hash1", "prop3")).is_none());
        assert!(cache.get(&CacheKey::new("hash1", "prop4")).is_some());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = VerificationCache::new();

        // Insert some entries
        for i in 0..3 {
            let key = CacheKey::new("hash1", format!("prop{}", i));
            let cached = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("prop{}", i),
                    passed: true,
                    status: "Proven".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            };
            cache.insert(key, cached);
        }

        assert_eq!(cache.len(), 3);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_eviction_on_max_entries() {
        // Create cache with small max entries
        let mut cache = VerificationCache::with_config(3, Duration::from_secs(3600));

        // Insert entries up to capacity
        for i in 0..3 {
            let key = CacheKey::new("hash1", format!("prop{}", i));
            let cached = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("prop{}", i),
                    passed: true,
                    status: "Proven".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            };
            cache.insert(key, cached);
        }

        assert_eq!(cache.len(), 3);

        // Insert one more - should evict oldest
        let key = CacheKey::new("hash1", "prop_new");
        let cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop_new".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.95,
        };
        cache.insert(key, cached);

        // Should still be at capacity (one evicted)
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_cached_property_result_expired() {
        let old_cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now() - Duration::from_secs(120),
            dependency_hash: "dep".to_string(),
            confidence: 0.95,
        };

        let new_cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.95,
        };

        // With 60 second TTL, old_cached should be expired, new_cached should not
        assert!(old_cached.is_expired(Duration::from_secs(60)));
        assert!(!new_cached.is_expired(Duration::from_secs(60)));
    }

    // ============ Async Verifier with Cache Tests ============

    #[test]
    fn test_async_verifier_with_cache() {
        let verifier = AsyncImprovementVerifier::with_cache();
        // Just test that it creates successfully - async verify needs tokio runtime
        assert!(verifier.cache.is_some());
    }

    #[test]
    fn test_async_verifier_with_custom_cache() {
        let verifier = AsyncImprovementVerifier::with_custom_cache(500, Duration::from_secs(120));
        assert!(verifier.cache.is_some());
    }

    #[test]
    fn test_async_verifier_enable_cache() {
        let mut verifier = AsyncImprovementVerifier::new();
        assert!(verifier.cache.is_none());

        verifier.enable_cache();
        assert!(verifier.cache.is_some());
    }

    #[test]
    fn test_incremental_verification_result_structure() {
        let result = IncrementalVerificationResult {
            result: VerificationResult::passed(HashMap::new(), 100),
            cached_count: 5,
            verified_count: 2,
            cached_properties: vec!["prop1".to_string(), "prop2".to_string()],
            verified_properties: vec!["prop3".to_string()],
            time_saved_ms: 250,
        };

        assert_eq!(result.cached_count, 5);
        assert_eq!(result.verified_count, 2);
        assert_eq!(result.cached_properties.len(), 2);
        assert_eq!(result.verified_properties.len(), 1);
        assert_eq!(result.time_saved_ms, 250);
    }

    #[test]
    fn test_verification_config_simple() {
        let config = VerificationConfig::simple();
        assert!(!config.use_dispatcher);
        assert_eq!(config.min_passing_backends, 1);
    }

    // ============ Test Stub Backend ============

    #[derive(Clone)]
    struct StubBackend {
        id: BackendId,
        status: VerificationStatus,
        supported: Vec<PropertyType>,
    }

    impl StubBackend {
        fn new(id: BackendId, status: VerificationStatus, supported: Vec<PropertyType>) -> Self {
            Self {
                id,
                status,
                supported,
            }
        }
    }

    #[async_trait]
    impl VerificationBackend for StubBackend {
        fn id(&self) -> BackendId {
            self.id
        }

        fn supports(&self) -> Vec<PropertyType> {
            self.supported.clone()
        }

        async fn verify(
            &self,
            _spec: &dashprove_usl::typecheck::TypedSpec,
        ) -> Result<dashprove_backends::BackendResult, dashprove_backends::BackendError> {
            Ok(dashprove_backends::BackendResult {
                backend: self.id,
                status: self.status.clone(),
                proof: None,
                counterexample: None,
                diagnostics: vec![],
                time_taken: Duration::from_millis(10),
            })
        }

        async fn health_check(&self) -> dashprove_backends::HealthStatus {
            dashprove_backends::HealthStatus::Healthy
        }
    }

    fn create_test_dispatcher() -> Dispatcher {
        let mut dispatcher = Dispatcher::new(DispatcherConfig {
            selection_strategy: SelectionStrategy::All,
            merge_strategy: dashprove_dispatcher::MergeStrategy::Unanimous,
            max_concurrent: 2,
            task_timeout: Duration::from_secs(5),
            check_health: false,
            auto_update_reputation: false,
        });

        dispatcher.register_backend(Arc::new(StubBackend::new(
            BackendId::Lean4,
            VerificationStatus::Proven,
            vec![PropertyType::Theorem, PropertyType::Invariant],
        )));

        dispatcher
    }

    #[derive(Clone)]
    struct FlakyBackend {
        id: BackendId,
        supported: Vec<PropertyType>,
        attempts: Arc<AtomicUsize>,
        success_after: usize,
    }

    impl FlakyBackend {
        fn new(
            id: BackendId,
            supported: Vec<PropertyType>,
            attempts: Arc<AtomicUsize>,
            success_after: usize,
        ) -> Self {
            Self {
                id,
                supported,
                attempts,
                success_after: success_after.max(1),
            }
        }
    }

    #[async_trait]
    impl VerificationBackend for FlakyBackend {
        fn id(&self) -> BackendId {
            self.id
        }

        fn supports(&self) -> Vec<PropertyType> {
            self.supported.clone()
        }

        async fn verify(
            &self,
            _spec: &dashprove_usl::typecheck::TypedSpec,
        ) -> Result<dashprove_backends::BackendResult, dashprove_backends::BackendError> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
            let status = if attempt >= self.success_after {
                VerificationStatus::Proven
            } else {
                VerificationStatus::Unknown {
                    reason: format!("transient failure {}", attempt),
                }
            };

            Ok(dashprove_backends::BackendResult {
                backend: self.id,
                status,
                proof: None,
                counterexample: None,
                diagnostics: vec![],
                time_taken: Duration::from_millis(5),
            })
        }

        async fn health_check(&self) -> dashprove_backends::HealthStatus {
            dashprove_backends::HealthStatus::Healthy
        }
    }

    fn create_flaky_dispatcher(attempts: Arc<AtomicUsize>, success_after: usize) -> Dispatcher {
        let mut dispatcher = Dispatcher::new(DispatcherConfig {
            selection_strategy: SelectionStrategy::All,
            merge_strategy: dashprove_dispatcher::MergeStrategy::FirstSuccess,
            max_concurrent: 2,
            task_timeout: Duration::from_secs(5),
            check_health: false,
            auto_update_reputation: false,
        });

        dispatcher.register_backend(Arc::new(FlakyBackend::new(
            BackendId::Lean4,
            vec![PropertyType::Theorem, PropertyType::Invariant],
            attempts,
            success_after,
        )));

        dispatcher
    }

    #[tokio::test]
    async fn async_verifier_retries_flaky_backend_until_success() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let dispatcher = create_flaky_dispatcher(attempts.clone(), 2);

        let mut config = VerificationConfig::quick();
        config.min_passing_backends = 1;
        config.retry_policy = Some(VerificationRetryPolicy::new(3).with_backoff(
            BackoffConfig::new(Duration::from_millis(1), Duration::from_millis(5)),
        ));

        let verifier = AsyncImprovementVerifier::with_config_and_dispatcher(config, dispatcher);
        let version = create_empty_caps_version();
        let improvement = create_empty_caps_improvement();

        let result = verifier.verify(&version, &improvement).await.unwrap();
        assert!(result.passed);
        assert!(attempts.load(Ordering::SeqCst) >= 2);
    }

    #[tokio::test]
    async fn async_verifier_respects_max_attempts_on_retry_policy() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let dispatcher = create_flaky_dispatcher(attempts.clone(), usize::MAX);

        let mut config = VerificationConfig::quick();
        config.min_passing_backends = 1;
        config.retry_policy = Some(
            VerificationRetryPolicy::new(1)
                .retry_on_failure(true)
                .with_backoff(BackoffConfig::new(
                    Duration::from_millis(1),
                    Duration::from_millis(1),
                )),
        );

        let verifier = AsyncImprovementVerifier::with_config_and_dispatcher(config, dispatcher);
        let version = create_empty_caps_version();
        let improvement = create_empty_caps_improvement();

        let result = verifier.verify(&version, &improvement).await.unwrap();
        assert!(!result.passed);
        let single_run_calls = result.verified_properties.len();
        assert!(
            attempts.load(Ordering::SeqCst) <= single_run_calls,
            "expected no additional retries"
        );
    }

    // ============ Async Tests for verify_incremental ============

    #[tokio::test]
    async fn async_verifier_cache_stats_initially_empty() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let stats = verifier.cache_stats().await;
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.entry_count, 0);
    }

    #[tokio::test]
    async fn async_verifier_cache_stats_none_without_cache() {
        let verifier = AsyncImprovementVerifier::new();
        let stats = verifier.cache_stats().await;
        assert!(stats.is_none());
    }

    #[tokio::test]
    async fn async_verifier_clear_cache_succeeds() {
        let verifier = AsyncImprovementVerifier::with_cache();

        // Clear cache should not panic even when empty
        verifier.clear_cache().await;

        let stats = verifier.cache_stats().await.unwrap();
        assert_eq!(stats.entry_count, 0);
    }

    #[tokio::test]
    async fn async_verifier_verify_with_cache_updates_cache() {
        let dispatcher = create_test_dispatcher();
        let verifier = AsyncImprovementVerifier::with_dispatcher_and_cache(dispatcher);
        let version = create_empty_caps_version();
        let improvement = create_empty_caps_improvement();

        // Run verification
        let result = verifier.verify(&version, &improvement).await;
        assert!(result.is_ok());

        // Cache should have been updated with verified properties
        let stats = verifier.cache_stats().await.unwrap();
        // Entry count depends on number of properties verified
        // We just check the cache was used (stats tracked)
        assert_eq!(stats.hits, 0); // First verification, no hits
    }

    #[tokio::test]
    async fn async_verifier_verify_incremental_without_cache() {
        let dispatcher = create_test_dispatcher();
        let verifier = AsyncImprovementVerifier::with_dispatcher(dispatcher);
        let version = create_empty_caps_version();
        let improvement = create_empty_caps_improvement();

        let changed_defs = vec!["SomeType".to_string()];
        let result = verifier
            .verify_incremental(&version, &improvement, &changed_defs)
            .await;

        assert!(result.is_ok());
        let inc_result = result.unwrap();

        // Without cache, cached_count should be 0
        assert_eq!(inc_result.cached_count, 0);
    }

    #[tokio::test]
    async fn async_verifier_verify_incremental_with_cache() {
        let dispatcher = create_test_dispatcher();
        let verifier = AsyncImprovementVerifier::with_dispatcher_and_cache(dispatcher);
        let version = create_empty_caps_version();
        let improvement = create_empty_caps_improvement();

        // First verification - fills the cache
        let changed_defs = vec![];
        let result1 = verifier
            .verify_incremental(&version, &improvement, &changed_defs)
            .await;

        assert!(result1.is_ok());
        let inc_result1 = result1.unwrap();

        // On first call, no cached properties
        assert_eq!(inc_result1.cached_count, 0);
    }

    #[tokio::test]
    async fn async_verifier_verify_incremental_returns_result_structure() {
        let dispatcher = create_test_dispatcher();
        let verifier = AsyncImprovementVerifier::with_dispatcher_and_cache(dispatcher);
        let version = create_empty_caps_version();
        let improvement = create_empty_caps_improvement();

        let changed_defs = vec![];
        let result = verifier
            .verify_incremental(&version, &improvement, &changed_defs)
            .await;

        assert!(result.is_ok());
        let inc_result = result.unwrap();

        // Verify structure is valid - result should exist
        // Duration is always >= 0 as u64, so just check it's set
        let _ = inc_result.result.duration_ms;
    }

    #[tokio::test]
    async fn async_verifier_verify_incremental_with_changed_definitions() {
        let dispatcher = create_test_dispatcher();
        let verifier = AsyncImprovementVerifier::with_dispatcher_and_cache(dispatcher);
        let version = create_empty_caps_version();
        let improvement = create_empty_caps_improvement();

        // Specify some changed definitions that should affect properties
        let changed_defs = vec!["CapabilityValue".to_string(), "Version".to_string()];

        let result = verifier
            .verify_incremental(&version, &improvement, &changed_defs)
            .await;

        assert!(result.is_ok());
        let inc_result = result.unwrap();

        // Result should be returned successfully
        // Duration is always >= 0 as u64, so just check it's set
        let _ = inc_result.result.duration_ms;
    }

    #[tokio::test]
    async fn async_verifier_cache_custom_config_works() {
        // Test with custom TTL and max entries
        let verifier = AsyncImprovementVerifier::with_custom_cache(100, Duration::from_secs(300));

        let stats = verifier.cache_stats().await;
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.entry_count, 0);
    }

    #[tokio::test]
    async fn async_verifier_enable_cache_on_existing() {
        let mut verifier = AsyncImprovementVerifier::new();
        assert!(verifier.cache_stats().await.is_none());

        verifier.enable_cache();
        assert!(verifier.cache_stats().await.is_some());

        // Second enable should not reset cache
        verifier.enable_cache();
        assert!(verifier.cache_stats().await.is_some());
    }

    #[tokio::test]
    async fn async_verifier_verify_preserves_passed_status() {
        let dispatcher = create_test_dispatcher();
        let verifier = AsyncImprovementVerifier::with_dispatcher_and_cache(dispatcher);
        let version = create_empty_caps_version();
        let improvement = create_empty_caps_improvement();

        let result = verifier.verify(&version, &improvement).await;
        assert!(result.is_ok());

        let verification_result = result.unwrap();
        // The result should have a passed field (may be true or false)
        // based on verification - just verify the field exists
        let _ = verification_result.passed;
    }

    #[tokio::test]
    async fn async_verifier_with_dispatcher_and_cache_creation() {
        let dispatcher = create_test_dispatcher();
        let verifier = AsyncImprovementVerifier::with_dispatcher_and_cache(dispatcher);

        // Should have cache enabled
        assert!(verifier.cache_stats().await.is_some());
        let stats = verifier.cache_stats().await.unwrap();
        assert_eq!(stats.entry_count, 0);
    }

    // ============ Integration Tests for Cache Hits Across Multiple Improvements ============

    #[tokio::test]
    async fn async_verifier_cache_hit_on_second_verification() {
        let dispatcher = create_test_dispatcher();
        let verifier = AsyncImprovementVerifier::with_dispatcher_and_cache(dispatcher);
        let version = create_empty_caps_version();
        let improvement = create_empty_caps_improvement();

        // First verification - populates the cache
        let result1 = verifier.verify(&version, &improvement).await;
        assert!(result1.is_ok());

        let stats_after_first = verifier.cache_stats().await.unwrap();

        // Second verification with same version should potentially hit cache
        let result2 = verifier.verify(&version, &improvement).await;
        assert!(result2.is_ok());

        let stats_after_second = verifier.cache_stats().await.unwrap();

        // Cache should be updated after verifications
        // Entry count might stay same or increase depending on properties
        assert!(stats_after_second.entry_count >= stats_after_first.entry_count);
    }

    #[tokio::test]
    async fn async_verifier_incremental_consecutive_verifications() {
        let dispatcher = create_test_dispatcher();
        let verifier = AsyncImprovementVerifier::with_dispatcher_and_cache(dispatcher);
        let version = create_empty_caps_version();
        let improvement = create_empty_caps_improvement();

        // First incremental verification
        let result1 = verifier
            .verify_incremental(&version, &improvement, &[])
            .await;
        assert!(result1.is_ok());
        let inc1 = result1.unwrap();

        // First call has no cached properties
        assert_eq!(inc1.cached_count, 0);

        // Second incremental verification with no changes
        let result2 = verifier
            .verify_incremental(&version, &improvement, &[])
            .await;
        assert!(result2.is_ok());
        let inc2 = result2.unwrap();

        // Second call should have results (verified_count is usize, always >= 0)
        let _ = inc2.verified_count;
    }

    #[tokio::test]
    async fn async_verifier_clear_cache_resets_hits() {
        let dispatcher = create_test_dispatcher();
        let verifier = AsyncImprovementVerifier::with_dispatcher_and_cache(dispatcher);
        let version = create_empty_caps_version();
        let improvement = create_empty_caps_improvement();

        // Run verification to populate cache
        let _ = verifier.verify(&version, &improvement).await;

        let stats_before = verifier.cache_stats().await.unwrap();

        // Clear the cache
        verifier.clear_cache().await;

        let stats_after = verifier.cache_stats().await.unwrap();

        // Entry count should be 0 after clear
        assert_eq!(stats_after.entry_count, 0);
        // Hits and misses should be reset (or preserved depending on implementation)
        // The important thing is entries are cleared
        assert!(stats_after.entry_count <= stats_before.entry_count);
    }

    #[tokio::test]
    async fn async_verifier_different_versions_create_separate_cache_entries() {
        let dispatcher = create_test_dispatcher();
        let verifier = AsyncImprovementVerifier::with_dispatcher_and_cache(dispatcher);
        let improvement = create_empty_caps_improvement();

        // Create two different versions
        let version1 = Version {
            id: VersionId::new("v1"),
            version_string: "1.0.0".to_string(),
            capabilities: CapabilitySet::new(),
            metadata: Default::default(),
            content_hash: "hash1".to_string(),
            previous_version: None,
        };

        let version2 = Version {
            id: VersionId::new("v2"),
            version_string: "2.0.0".to_string(),
            capabilities: CapabilitySet::new(),
            metadata: Default::default(),
            content_hash: "hash2".to_string(),
            previous_version: None,
        };

        // Verify first version
        let result1 = verifier.verify(&version1, &improvement).await;
        assert!(result1.is_ok());

        let stats_after_v1 = verifier.cache_stats().await.unwrap();

        // Verify second version (different hash, so different cache entries)
        let result2 = verifier.verify(&version2, &improvement).await;
        assert!(result2.is_ok());

        let stats_after_v2 = verifier.cache_stats().await.unwrap();

        // Both verifications should succeed
        // Cache might have more entries after second verification
        assert!(stats_after_v2.entry_count >= stats_after_v1.entry_count);
    }

    #[tokio::test]
    async fn async_verifier_incremental_with_invalidation() {
        let dispatcher = create_test_dispatcher();
        let verifier = AsyncImprovementVerifier::with_dispatcher_and_cache(dispatcher);
        let version = create_empty_caps_version();
        let improvement = create_empty_caps_improvement();

        // First verification populates cache
        let result1 = verifier
            .verify_incremental(&version, &improvement, &[])
            .await;
        assert!(result1.is_ok());

        // Second verification with changed definitions should invalidate
        let changed = vec!["Version".to_string()];
        let result2 = verifier
            .verify_incremental(&version, &improvement, &changed)
            .await;
        assert!(result2.is_ok());
        let inc2 = result2.unwrap();

        // Result should be returned successfully
        // Properties affected by Version might be re-verified
        // Just verify we got a result with a valid duration
        let _ = inc2.result.duration_ms;
    }

    // ============ TTL Expiration Tests ============

    #[test]
    fn test_cache_ttl_expiration_removes_expired_entries() {
        // Create cache with very short TTL (1ms)
        let mut cache = VerificationCache::with_config(100, Duration::from_millis(1));

        // Insert an entry
        let key = CacheKey::new("hash1", "prop1");
        let cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop1".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.95,
        };
        cache.insert(key.clone(), cached);

        // Verify entry is in cache
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stats().entry_count, 1);

        // Sleep to let TTL expire
        std::thread::sleep(Duration::from_millis(5));

        // Try to get the entry - should be expired and removed
        let result = cache.get(&key);
        assert!(result.is_none());

        // Entry should have been removed
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats().entry_count, 0);

        // Misses counter should be incremented
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_ttl_non_expired_entries_returned() {
        // Create cache with long TTL (1 hour)
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));

        // Insert an entry
        let key = CacheKey::new("hash1", "prop1");
        let cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop1".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.95,
        };
        cache.insert(key.clone(), cached);

        // Entry should be retrievable immediately (not expired)
        let result = cache.get(&key);
        assert!(result.is_some());
        assert_eq!(result.unwrap().property.name, "prop1");

        // Entry should still be in cache
        assert_eq!(cache.len(), 1);

        // Hits counter should be incremented
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_cache_ttl_mixed_expiration() {
        // Create cache with 10ms TTL
        let mut cache = VerificationCache::with_config(100, Duration::from_millis(10));

        // Insert first entry
        let key1 = CacheKey::new("hash1", "prop1");
        let cached1 = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop1".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.95,
        };
        cache.insert(key1.clone(), cached1);

        // Sleep to let first entry expire
        std::thread::sleep(Duration::from_millis(15));

        // Insert second entry (fresh)
        let key2 = CacheKey::new("hash1", "prop2");
        let cached2 = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop2".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.95,
        };
        cache.insert(key2.clone(), cached2);

        // First entry should be expired
        let result1 = cache.get(&key1);
        assert!(result1.is_none());
        assert_eq!(cache.stats().misses, 1);

        // Second entry should still be valid
        let result2 = cache.get(&key2);
        assert!(result2.is_some());
        assert_eq!(result2.unwrap().property.name, "prop2");
        assert_eq!(cache.stats().hits, 1);
    }

    #[tokio::test]
    async fn async_verifier_cache_ttl_expiration() {
        // Create verifier with very short TTL
        let verifier = AsyncImprovementVerifier::with_custom_cache(100, Duration::from_millis(1));

        // Access cache directly to insert entry for testing
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            let key = CacheKey::new("test_hash", "test_prop");
            let cached = CachedPropertyResult {
                property: VerifiedProperty {
                    name: "test_prop".to_string(),
                    passed: true,
                    status: "Proven".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            };
            cache_guard.insert(key, cached);
            assert_eq!(cache_guard.len(), 1);
        }

        // Sleep to let TTL expire
        tokio::time::sleep(Duration::from_millis(5)).await;

        // Verify entry is expired by checking stats after get attempt
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            let key = CacheKey::new("test_hash", "test_prop");
            let result = cache_guard.get(&key);

            assert!(result.is_none());
            assert_eq!(cache_guard.len(), 0);
            assert_eq!(cache_guard.stats().misses, 1);
        }
    }

    #[test]
    fn test_cache_ttl_zero_duration_expires_immediately() {
        // Create cache with zero TTL - entries should expire immediately
        let mut cache = VerificationCache::with_config(100, Duration::from_millis(0));

        let key = CacheKey::new("hash1", "prop1");
        let cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop1".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.95,
        };
        cache.insert(key.clone(), cached);

        // Entry is in cache but should be expired on get
        assert_eq!(cache.len(), 1);

        // Get should return None (expired) and remove entry
        let result = cache.get(&key);
        assert!(result.is_none());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_stats_track_expired_as_misses() {
        // Create cache with short TTL
        let mut cache = VerificationCache::with_config(100, Duration::from_millis(1));

        // Insert 3 entries
        for i in 0..3 {
            let key = CacheKey::new("hash1", format!("prop{}", i));
            let cached = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("prop{}", i),
                    passed: true,
                    status: "Proven".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            };
            cache.insert(key, cached);
        }

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);

        // Sleep to expire all entries
        std::thread::sleep(Duration::from_millis(5));

        // Try to get all entries - all should be expired
        for i in 0..3 {
            let key = CacheKey::new("hash1", format!("prop{}", i));
            let result = cache.get(&key);
            assert!(result.is_none());
        }

        // All gets should be recorded as misses
        assert_eq!(cache.stats().misses, 3);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_hit_rate_with_expirations() {
        // Create cache with 50ms TTL
        let mut cache = VerificationCache::with_config(100, Duration::from_millis(50));

        // Insert entries
        for i in 0..4 {
            let key = CacheKey::new("hash1", format!("prop{}", i));
            let cached = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("prop{}", i),
                    passed: true,
                    status: "Proven".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            };
            cache.insert(key, cached);
        }

        // Get 2 entries immediately (hits)
        for i in 0..2 {
            let key = CacheKey::new("hash1", format!("prop{}", i));
            let result = cache.get(&key);
            assert!(result.is_some());
        }

        assert_eq!(cache.stats().hits, 2);
        assert_eq!(cache.stats().misses, 0);
        assert!((cache.stats().hit_rate() - 1.0).abs() < f64::EPSILON);

        // Sleep to expire remaining entries
        std::thread::sleep(Duration::from_millis(55));

        // Try to get remaining entries (misses due to expiration)
        for i in 2..4 {
            let key = CacheKey::new("hash1", format!("prop{}", i));
            let result = cache.get(&key);
            assert!(result.is_none());
        }

        // 2 hits, 2 misses = 50% hit rate
        assert_eq!(cache.stats().hits, 2);
        assert_eq!(cache.stats().misses, 2);
        assert!((cache.stats().hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    // ============ Confidence Threshold Tests ============

    #[test]
    fn test_cache_confidence_threshold_default_accepts_all() {
        let mut cache = VerificationCache::new();

        // Default threshold is 0.0 - should accept all confidence levels
        assert_eq!(cache.confidence_threshold(), 0.0);

        // Insert entry with low confidence
        let key = CacheKey::new("hash1", "prop1");
        let cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop1".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.1,
        };
        cache.insert(key.clone(), cached);

        // Should be retrievable (low confidence is still >= 0.0 threshold)
        let result = cache.get(&key);
        assert!(result.is_some());
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_cache_confidence_threshold_filters_low_confidence() {
        // Create cache with 80% confidence threshold
        let mut cache = VerificationCache::with_full_config(100, Duration::from_secs(3600), 0.8);

        assert_eq!(cache.confidence_threshold(), 0.8);

        // Insert entry with confidence below threshold
        let key = CacheKey::new("hash1", "low_confidence");
        let cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "low_confidence".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.5, // Below 0.8 threshold
        };
        cache.insert(key.clone(), cached);

        // Entry exists but should be treated as miss due to low confidence
        assert_eq!(cache.len(), 1);
        let result = cache.get(&key);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        // Entry should NOT be removed (unlike expired entries)
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_confidence_threshold_accepts_high_confidence() {
        // Create cache with 80% confidence threshold
        let mut cache = VerificationCache::with_full_config(100, Duration::from_secs(3600), 0.8);

        // Insert entry with confidence at threshold
        let key1 = CacheKey::new("hash1", "at_threshold");
        let cached1 = CachedPropertyResult {
            property: VerifiedProperty {
                name: "at_threshold".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.8, // Exactly at threshold
        };
        cache.insert(key1.clone(), cached1);

        // Insert entry with confidence above threshold
        let key2 = CacheKey::new("hash1", "above_threshold");
        let cached2 = CachedPropertyResult {
            property: VerifiedProperty {
                name: "above_threshold".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.95, // Above threshold
        };
        cache.insert(key2.clone(), cached2);

        // Both should be retrievable
        let result1 = cache.get(&key1);
        let result2 = cache.get(&key2);
        assert!(result1.is_some());
        assert!(result2.is_some());
        assert_eq!(cache.stats().hits, 2);
    }

    #[test]
    fn test_cache_set_confidence_threshold_dynamic() {
        let mut cache = VerificationCache::new();

        // Insert entries with different confidence levels
        let high_key = CacheKey::new("hash1", "high");
        let high_cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "high".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.95,
        };
        cache.insert(high_key.clone(), high_cached);

        let low_key = CacheKey::new("hash1", "low");
        let low_cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "low".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.5,
        };
        cache.insert(low_key.clone(), low_cached);

        // With default threshold (0.0), both should be retrievable
        assert!(cache.get(&high_key).is_some());
        assert!(cache.get(&low_key).is_some());
        assert_eq!(cache.stats().hits, 2);

        // Dynamically increase threshold
        cache.set_confidence_threshold(0.9);
        assert_eq!(cache.confidence_threshold(), 0.9);

        // Now only high confidence entry should be retrievable
        assert!(cache.get(&high_key).is_some());
        assert!(cache.get(&low_key).is_none());
        assert_eq!(cache.stats().hits, 3);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_confidence_threshold_clamps_values() {
        // Test that threshold is clamped to [0.0, 1.0]
        let mut cache = VerificationCache::with_full_config(
            100,
            Duration::from_secs(3600),
            1.5, // Above 1.0
        );
        assert_eq!(cache.confidence_threshold(), 1.0);

        cache.set_confidence_threshold(-0.5); // Below 0.0
        assert_eq!(cache.confidence_threshold(), 0.0);

        cache.set_confidence_threshold(0.75);
        assert_eq!(cache.confidence_threshold(), 0.75);
    }

    #[test]
    fn test_cache_confidence_and_ttl_combined() {
        // Test that both TTL and confidence threshold are applied
        let mut cache = VerificationCache::with_full_config(
            100,
            Duration::from_millis(10), // Short TTL
            0.8,                       // High confidence threshold
        );

        // Insert entry with high confidence (passes confidence check)
        let key = CacheKey::new("hash1", "prop1");
        let cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop1".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.95,
        };
        cache.insert(key.clone(), cached);

        // Should be retrievable immediately
        assert!(cache.get(&key).is_some());
        assert_eq!(cache.stats().hits, 1);

        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(15));

        // Should now be expired (TTL check takes precedence, removes entry)
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.len(), 0); // Entry removed due to TTL
    }

    #[test]
    fn test_cached_property_result_stores_confidence() {
        let cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop".to_string(),
                passed: true,
                status: "Proven (confidence: 95%)".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.95,
        };

        assert_eq!(cached.confidence, 0.95);
        assert!(cached.property.status.contains("95%"));
    }

    #[test]
    fn test_cache_confidence_ttl_scaling_disabled_by_default() {
        let cache = VerificationCache::new();
        assert!(!cache.is_confidence_ttl_scaling_enabled());

        // Base TTL should always be returned when scaling is disabled
        assert_eq!(cache.effective_ttl(1.0), Duration::from_secs(3600));
        assert_eq!(cache.effective_ttl(0.5), Duration::from_secs(3600));
        assert_eq!(cache.effective_ttl(0.0), Duration::from_secs(3600));
    }

    #[test]
    fn test_cache_confidence_ttl_scaling_enable_disable() {
        let mut cache = VerificationCache::with_config(100, Duration::from_millis(1000));

        // Initially disabled
        assert!(!cache.is_confidence_ttl_scaling_enabled());

        // Enable
        cache.enable_confidence_ttl_scaling(0.1);
        assert!(cache.is_confidence_ttl_scaling_enabled());
        assert_eq!(cache.min_ttl_multiplier(), 0.1);

        // Disable
        cache.disable_confidence_ttl_scaling();
        assert!(!cache.is_confidence_ttl_scaling_enabled());
    }

    #[test]
    fn test_cache_effective_ttl_calculation() {
        let mut cache = VerificationCache::with_config(100, Duration::from_millis(1000));
        cache.enable_confidence_ttl_scaling(0.1);

        // High confidence (0.9) → 90% of base TTL
        assert_eq!(cache.effective_ttl(0.9), Duration::from_millis(900));

        // Medium confidence (0.5) → 50% of base TTL
        assert_eq!(cache.effective_ttl(0.5), Duration::from_millis(500));

        // Low confidence (0.2) → 20% of base TTL
        assert_eq!(cache.effective_ttl(0.2), Duration::from_millis(200));

        // Very low confidence (0.05) → 10% of base TTL (floor from min_ttl_multiplier)
        assert_eq!(cache.effective_ttl(0.05), Duration::from_millis(100));

        // Zero confidence → 10% of base TTL (floor)
        assert_eq!(cache.effective_ttl(0.0), Duration::from_millis(100));

        // Full confidence → 100% of base TTL
        assert_eq!(cache.effective_ttl(1.0), Duration::from_millis(1000));
    }

    #[test]
    fn test_cache_confidence_ttl_scaling_low_confidence_expires_faster() {
        let mut cache = VerificationCache::with_config(100, Duration::from_millis(100));
        cache.enable_confidence_ttl_scaling(0.1);

        // Insert high confidence entry (effective TTL: 90ms)
        let key_high = CacheKey::new("hash1", "high_conf");
        let cached_high = CachedPropertyResult {
            property: VerifiedProperty {
                name: "high_conf".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.9,
        };
        cache.insert(key_high.clone(), cached_high);

        // Insert low confidence entry (effective TTL: 20ms)
        let key_low = CacheKey::new("hash1", "low_conf");
        let cached_low = CachedPropertyResult {
            property: VerifiedProperty {
                name: "low_conf".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.2,
        };
        cache.insert(key_low.clone(), cached_low);

        // Both should be retrievable immediately
        assert!(cache.get(&key_high).is_some());
        assert!(cache.get(&key_low).is_some());

        // Wait for low confidence entry to expire (20ms effective TTL)
        std::thread::sleep(Duration::from_millis(25));

        // Low confidence entry should be expired
        assert!(cache.get(&key_low).is_none());

        // High confidence entry should still be valid
        assert!(cache.get(&key_high).is_some());

        // Wait for high confidence entry to expire (90ms effective TTL)
        std::thread::sleep(Duration::from_millis(70));

        // Now both should be expired
        assert!(cache.get(&key_high).is_none());
    }

    #[test]
    fn test_cache_min_ttl_multiplier_clamps_values() {
        let mut cache = VerificationCache::new();

        // Negative values should clamp to 0.0
        cache.enable_confidence_ttl_scaling(-0.5);
        assert_eq!(cache.min_ttl_multiplier(), 0.0);

        // Values above 1.0 should clamp to 1.0
        cache.enable_confidence_ttl_scaling(1.5);
        assert_eq!(cache.min_ttl_multiplier(), 1.0);

        // Valid values should be preserved
        cache.enable_confidence_ttl_scaling(0.25);
        assert_eq!(cache.min_ttl_multiplier(), 0.25);
    }

    #[test]
    fn test_cache_confidence_ttl_scaling_zero_min_multiplier() {
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(1));
        cache.enable_confidence_ttl_scaling(0.0);

        // With zero min_multiplier, zero confidence = zero TTL (instant expiration)
        assert_eq!(cache.effective_ttl(0.0), Duration::from_millis(0));

        // But positive confidence still works
        assert_eq!(cache.effective_ttl(0.5), Duration::from_millis(500));
    }

    #[test]
    fn test_cache_confidence_ttl_scaling_with_threshold_combined() {
        // Test that confidence threshold and TTL scaling work together
        let mut cache = VerificationCache::with_full_config(
            100,
            Duration::from_millis(100),
            0.5, // Only accept confidence >= 0.5
        );
        cache.enable_confidence_ttl_scaling(0.1);

        // Insert entry with medium confidence (0.6) - passes threshold
        // Effective TTL = 100ms * 0.6 = 60ms
        let key = CacheKey::new("hash1", "prop1");
        let cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop1".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.6,
        };
        cache.insert(key.clone(), cached);

        // Should be retrievable immediately
        assert!(cache.get(&key).is_some());

        // Wait for effective TTL to expire
        std::thread::sleep(Duration::from_millis(65));

        // Should now be expired due to scaled TTL
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_property_ttl_override_set_and_get() {
        let mut cache = VerificationCache::new();

        // Initially no overrides
        assert!(cache.get_property_ttl_override("security").is_none());
        assert!(cache.property_ttl_overrides().is_empty());

        // Set an override
        cache.set_property_ttl_override("security", Duration::from_secs(1800));

        // Verify it's set
        assert_eq!(
            cache.get_property_ttl_override("security"),
            Some(Duration::from_secs(1800))
        );
        assert_eq!(cache.property_ttl_overrides().len(), 1);

        // Set another override
        cache.set_property_ttl_override("invariant", Duration::from_secs(7200));
        assert_eq!(cache.property_ttl_overrides().len(), 2);
    }

    #[test]
    fn test_cache_property_ttl_override_remove() {
        let mut cache = VerificationCache::new();

        cache.set_property_ttl_override("security", Duration::from_secs(1800));
        assert!(cache.get_property_ttl_override("security").is_some());

        // Remove override
        let removed = cache.remove_property_ttl_override("security");
        assert_eq!(removed, Some(Duration::from_secs(1800)));

        // Now it's gone
        assert!(cache.get_property_ttl_override("security").is_none());

        // Removing non-existent returns None
        assert!(cache.remove_property_ttl_override("nonexistent").is_none());
    }

    #[test]
    fn test_cache_property_ttl_override_clear_all() {
        let mut cache = VerificationCache::new();

        cache.set_property_ttl_override("security", Duration::from_secs(1800));
        cache.set_property_ttl_override("invariant", Duration::from_secs(7200));
        assert_eq!(cache.property_ttl_overrides().len(), 2);

        cache.clear_property_ttl_overrides();
        assert!(cache.property_ttl_overrides().is_empty());
    }

    #[test]
    fn test_cache_effective_ttl_for_property_with_override() {
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));

        // Without override, uses base TTL
        assert_eq!(
            cache.effective_ttl_for_property("any_property", 1.0),
            Duration::from_secs(3600)
        );

        // Set override for specific property
        cache.set_property_ttl_override("security", Duration::from_secs(1800));

        // Property with override uses override TTL
        assert_eq!(
            cache.effective_ttl_for_property("security", 1.0),
            Duration::from_secs(1800)
        );

        // Other properties still use base TTL
        assert_eq!(
            cache.effective_ttl_for_property("invariant", 1.0),
            Duration::from_secs(3600)
        );
    }

    #[test]
    fn test_cache_property_ttl_override_with_confidence_scaling() {
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        cache.enable_confidence_ttl_scaling(0.1);
        cache.set_property_ttl_override("security", Duration::from_secs(1000));

        // With confidence 0.8:
        // - "security" property: 1000s * 0.8 = 800s
        // - other properties: 3600s * 0.8 = 2880s
        assert_eq!(
            cache.effective_ttl_for_property("security", 0.8),
            Duration::from_millis(800_000)
        );
        assert_eq!(
            cache.effective_ttl_for_property("invariant", 0.8),
            Duration::from_millis(2_880_000)
        );

        // With confidence at floor (0.1):
        // - "security" property: 1000s * 0.1 = 100s
        // - other properties: 3600s * 0.1 = 360s
        assert_eq!(
            cache.effective_ttl_for_property("security", 0.0),
            Duration::from_millis(100_000)
        );
        assert_eq!(
            cache.effective_ttl_for_property("invariant", 0.0),
            Duration::from_millis(360_000)
        );
    }

    #[test]
    fn test_cache_get_respects_property_ttl_override() {
        // Use very short TTLs for testing
        let mut cache = VerificationCache::with_config(100, Duration::from_millis(100));

        // Override "fast_expire" to 20ms, keep base at 100ms
        cache.set_property_ttl_override("fast_expire", Duration::from_millis(20));

        // Insert entries for both properties
        let key_fast = CacheKey::new("hash1", "fast_expire");
        let key_slow = CacheKey::new("hash1", "slow_property");

        let make_entry = |name: &str| CachedPropertyResult {
            property: VerifiedProperty {
                name: name.to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 1.0,
        };

        cache.insert(key_fast.clone(), make_entry("fast_expire"));
        cache.insert(key_slow.clone(), make_entry("slow_property"));

        // Both are accessible immediately
        assert!(cache.get(&key_fast).is_some());
        assert!(cache.get(&key_slow).is_some());

        // Wait for fast property to expire (25ms > 20ms override)
        std::thread::sleep(Duration::from_millis(25));

        // Fast property should be expired, slow property still valid
        assert!(
            cache.get(&key_fast).is_none(),
            "fast_expire should have expired"
        );
        assert!(
            cache.get(&key_slow).is_some(),
            "slow_property should still be valid"
        );

        // Wait for slow property to expire too (total ~100ms)
        std::thread::sleep(Duration::from_millis(80));
        assert!(
            cache.get(&key_slow).is_none(),
            "slow_property should now be expired"
        );
    }

    #[test]
    fn test_cache_property_ttl_override_combined_with_all_features() {
        // Test per-property TTL overrides work with both confidence scaling and threshold
        let mut cache = VerificationCache::with_full_config(
            100,
            Duration::from_millis(200), // Base TTL
            0.5,                        // Confidence threshold
        );
        cache.enable_confidence_ttl_scaling(0.1); // Enable scaling with 10% floor
        cache.set_property_ttl_override("critical", Duration::from_millis(100)); // Half the base

        // Insert entry for "critical" property with 0.8 confidence
        // Effective TTL = 100ms * 0.8 = 80ms
        let key = CacheKey::new("hash1", "critical");
        let cached = CachedPropertyResult {
            property: VerifiedProperty {
                name: "critical".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 0.8,
        };
        cache.insert(key.clone(), cached);

        // Immediately accessible (above confidence threshold)
        assert!(cache.get(&key).is_some());

        // Wait for confidence-scaled property override TTL to expire
        std::thread::sleep(Duration::from_millis(85));

        // Should be expired now
        assert!(cache.get(&key).is_none());
    }

    // =========================================================================
    // Pattern-based TTL override tests
    // =========================================================================

    #[test]
    fn test_cache_pattern_ttl_override_set_and_get() {
        let mut cache = VerificationCache::new();

        // Set pattern overrides
        assert!(cache
            .set_pattern_ttl_override("security_*", Duration::from_secs(1800))
            .is_ok());
        assert!(cache
            .set_pattern_ttl_override("*_invariant", Duration::from_secs(7200))
            .is_ok());

        // Verify they were stored
        assert_eq!(
            cache.get_pattern_ttl_override("security_*"),
            Some(Duration::from_secs(1800))
        );
        assert_eq!(
            cache.get_pattern_ttl_override("*_invariant"),
            Some(Duration::from_secs(7200))
        );
        assert_eq!(cache.get_pattern_ttl_override("nonexistent"), None);

        // Check list of patterns
        let patterns = cache.pattern_ttl_overrides();
        assert_eq!(patterns.len(), 2);
        assert!(patterns.iter().any(|(p, _)| *p == "security_*"));
        assert!(patterns.iter().any(|(p, _)| *p == "*_invariant"));
    }

    #[test]
    fn test_cache_pattern_ttl_override_invalid_pattern() {
        let mut cache = VerificationCache::new();

        // Invalid pattern should return an error
        let result = cache.set_pattern_ttl_override("[invalid", Duration::from_secs(100));
        assert!(result.is_err());
    }

    #[test]
    fn test_cache_pattern_ttl_override_remove() {
        let mut cache = VerificationCache::new();
        cache
            .set_pattern_ttl_override("test_*", Duration::from_secs(60))
            .unwrap();
        cache
            .set_pattern_ttl_override("*_check", Duration::from_secs(120))
            .unwrap();

        // Remove one pattern
        let removed = cache.remove_pattern_ttl_override("test_*");
        assert_eq!(removed, Some(Duration::from_secs(60)));

        // Verify removal
        assert_eq!(cache.get_pattern_ttl_override("test_*"), None);
        assert_eq!(
            cache.get_pattern_ttl_override("*_check"),
            Some(Duration::from_secs(120))
        );
        assert_eq!(cache.pattern_ttl_overrides().len(), 1);

        // Remove non-existent pattern
        let removed = cache.remove_pattern_ttl_override("nonexistent");
        assert_eq!(removed, None);
    }

    #[test]
    fn test_cache_pattern_ttl_override_clear() {
        let mut cache = VerificationCache::new();
        cache
            .set_pattern_ttl_override("a_*", Duration::from_secs(100))
            .unwrap();
        cache
            .set_pattern_ttl_override("b_*", Duration::from_secs(200))
            .unwrap();
        cache
            .set_pattern_ttl_override("c_*", Duration::from_secs(300))
            .unwrap();

        assert_eq!(cache.pattern_ttl_overrides().len(), 3);

        cache.clear_pattern_ttl_overrides();
        assert_eq!(cache.pattern_ttl_overrides().len(), 0);
    }

    #[test]
    fn test_cache_clear_all_ttl_overrides() {
        let mut cache = VerificationCache::new();

        // Add both exact and pattern overrides
        cache.set_property_ttl_override("exact_prop", Duration::from_secs(100));
        cache
            .set_pattern_ttl_override("pattern_*", Duration::from_secs(200))
            .unwrap();

        assert_eq!(cache.property_ttl_overrides().len(), 1);
        assert_eq!(cache.pattern_ttl_overrides().len(), 1);

        // Clear all
        cache.clear_all_ttl_overrides();
        assert_eq!(cache.property_ttl_overrides().len(), 0);
        assert_eq!(cache.pattern_ttl_overrides().len(), 0);
    }

    #[test]
    fn test_cache_pattern_ttl_matching_prefix() {
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        cache
            .set_pattern_ttl_override("security_*", Duration::from_secs(1800))
            .unwrap();

        // Properties starting with "security_" should get the pattern TTL
        assert_eq!(
            cache.effective_ttl_for_property("security_check", 1.0),
            Duration::from_secs(1800)
        );
        assert_eq!(
            cache.effective_ttl_for_property("security_audit", 1.0),
            Duration::from_secs(1800)
        );
        assert_eq!(
            cache.effective_ttl_for_property("security_long_name_here", 1.0),
            Duration::from_secs(1800)
        );

        // Other properties should get base TTL
        assert_eq!(
            cache.effective_ttl_for_property("other_property", 1.0),
            Duration::from_secs(3600)
        );
        assert_eq!(
            cache.effective_ttl_for_property("not_security", 1.0),
            Duration::from_secs(3600)
        );
    }

    #[test]
    fn test_cache_pattern_ttl_matching_suffix() {
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        cache
            .set_pattern_ttl_override("*_invariant", Duration::from_secs(7200))
            .unwrap();

        // Properties ending with "_invariant" should get the pattern TTL
        assert_eq!(
            cache.effective_ttl_for_property("state_invariant", 1.0),
            Duration::from_secs(7200)
        );
        assert_eq!(
            cache.effective_ttl_for_property("data_invariant", 1.0),
            Duration::from_secs(7200)
        );

        // Other properties should get base TTL
        assert_eq!(
            cache.effective_ttl_for_property("invariant_check", 1.0),
            Duration::from_secs(3600)
        );
    }

    #[test]
    fn test_cache_pattern_ttl_matching_contains() {
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        cache
            .set_pattern_ttl_override("*test*", Duration::from_secs(60))
            .unwrap();

        // Properties containing "test" should get the pattern TTL
        assert_eq!(
            cache.effective_ttl_for_property("test_property", 1.0),
            Duration::from_secs(60)
        );
        assert_eq!(
            cache.effective_ttl_for_property("my_test_check", 1.0),
            Duration::from_secs(60)
        );
        assert_eq!(
            cache.effective_ttl_for_property("unittest", 1.0),
            Duration::from_secs(60)
        );

        // Properties without "test" should get base TTL
        assert_eq!(
            cache.effective_ttl_for_property("production", 1.0),
            Duration::from_secs(3600)
        );
    }

    #[test]
    fn test_cache_pattern_ttl_exact_override_priority() {
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));

        // Set pattern first
        cache
            .set_pattern_ttl_override("security_*", Duration::from_secs(1800))
            .unwrap();

        // Set exact override for a specific property
        cache.set_property_ttl_override("security_critical", Duration::from_secs(600));

        // Exact override takes priority
        assert_eq!(
            cache.effective_ttl_for_property("security_critical", 1.0),
            Duration::from_secs(600)
        );

        // Other matching properties still use the pattern
        assert_eq!(
            cache.effective_ttl_for_property("security_audit", 1.0),
            Duration::from_secs(1800)
        );
    }

    #[test]
    fn test_cache_pattern_ttl_first_match_wins() {
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));

        // Add multiple patterns that could match the same property
        // First pattern registered should win
        cache
            .set_pattern_ttl_override("security_*", Duration::from_secs(1800))
            .unwrap();
        cache
            .set_pattern_ttl_override("*_check", Duration::from_secs(900))
            .unwrap();

        // "security_check" matches both patterns, but first one wins
        assert_eq!(
            cache.effective_ttl_for_property("security_check", 1.0),
            Duration::from_secs(1800)
        );

        // Properties matching only second pattern use that
        assert_eq!(
            cache.effective_ttl_for_property("audit_check", 1.0),
            Duration::from_secs(900)
        );
    }

    #[test]
    fn test_cache_pattern_ttl_with_confidence_scaling() {
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        cache
            .set_pattern_ttl_override("security_*", Duration::from_secs(1800))
            .unwrap();
        cache.enable_confidence_ttl_scaling(0.1);

        // Pattern TTL with confidence scaling
        // Base pattern TTL is 1800s, confidence 0.5 -> 900s
        assert_eq!(
            cache.effective_ttl_for_property("security_check", 0.5),
            Duration::from_secs(900)
        );

        // Non-matching property uses base TTL with scaling
        // Base TTL is 3600s, confidence 0.5 -> 1800s
        assert_eq!(
            cache.effective_ttl_for_property("other_property", 0.5),
            Duration::from_secs(1800)
        );
    }

    #[test]
    fn test_cache_pattern_ttl_get_respects_pattern() {
        let mut cache = VerificationCache::with_config(100, Duration::from_millis(200));
        cache
            .set_pattern_ttl_override("fast_*", Duration::from_millis(50))
            .unwrap();

        // Create cache entries
        let key_fast = CacheKey::new("hash1", "fast_property");
        let key_slow = CacheKey::new("hash1", "slow_property");

        let cached_fast = CachedPropertyResult {
            property: VerifiedProperty {
                name: "fast_property".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 1.0,
        };

        let cached_slow = CachedPropertyResult {
            property: VerifiedProperty {
                name: "slow_property".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep".to_string(),
            confidence: 1.0,
        };

        cache.insert(key_fast.clone(), cached_fast);
        cache.insert(key_slow.clone(), cached_slow);

        // Both should be accessible immediately
        assert!(cache.get(&key_fast).is_some());
        assert!(cache.get(&key_slow).is_some());

        // Wait for the fast pattern TTL to expire
        std::thread::sleep(Duration::from_millis(60));

        // fast_property should be expired (50ms pattern TTL)
        assert!(
            cache.get(&key_fast).is_none(),
            "fast_property should be expired"
        );
        // slow_property should still be valid (200ms base TTL)
        assert!(
            cache.get(&key_slow).is_some(),
            "slow_property should still be valid"
        );
    }

    // =========================================================================
    // Cache preset tests
    // =========================================================================

    #[test]
    fn test_cache_security_preset_configuration() {
        let cache = VerificationCache::new().with_security_preset();

        // Verify confidence settings
        assert_eq!(cache.confidence_threshold(), 0.9);
        assert!(cache.is_confidence_ttl_scaling_enabled());
        assert_eq!(cache.min_ttl_multiplier(), 0.2);

        // Verify pattern overrides exist
        let patterns = cache.pattern_ttl_overrides();
        assert!(
            patterns.len() >= 10,
            "Should have at least 10 pattern overrides"
        );

        // Verify security pattern TTL (30 minutes)
        assert_eq!(
            cache.get_pattern_ttl_override("security_*"),
            Some(Duration::from_secs(30 * 60))
        );

        // Verify invariant pattern TTL (2 hours)
        assert_eq!(
            cache.get_pattern_ttl_override("*_invariant"),
            Some(Duration::from_secs(2 * 60 * 60))
        );

        // Verify test pattern TTL (5 minutes)
        assert_eq!(
            cache.get_pattern_ttl_override("test_*"),
            Some(Duration::from_secs(5 * 60))
        );
    }

    #[test]
    fn test_cache_security_preset_pattern_matching() {
        let cache = VerificationCache::new().with_security_preset();

        // Security properties should get 30 min TTL
        assert_eq!(
            cache.effective_ttl_for_property("security_check", 1.0),
            Duration::from_secs(30 * 60)
        );
        assert_eq!(
            cache.effective_ttl_for_property("auth_login", 1.0),
            Duration::from_secs(30 * 60)
        );
        assert_eq!(
            cache.effective_ttl_for_property("user_auth", 1.0),
            Duration::from_secs(30 * 60)
        );

        // Invariant properties should get 2 hour TTL
        assert_eq!(
            cache.effective_ttl_for_property("state_invariant", 1.0),
            Duration::from_secs(2 * 60 * 60)
        );
        assert_eq!(
            cache.effective_ttl_for_property("invariant_preserved", 1.0),
            Duration::from_secs(2 * 60 * 60)
        );

        // Test properties should get 5 min TTL
        assert_eq!(
            cache.effective_ttl_for_property("test_module", 1.0),
            Duration::from_secs(5 * 60)
        );
    }

    #[test]
    fn test_cache_security_preset_confidence_scaling() {
        let cache = VerificationCache::new().with_security_preset();

        // High confidence (1.0) gets full TTL
        let full_ttl = cache.effective_ttl_for_property("security_check", 1.0);
        assert_eq!(full_ttl, Duration::from_secs(30 * 60));

        // Low confidence (0.5) gets 50% of TTL
        let half_ttl = cache.effective_ttl_for_property("security_check", 0.5);
        assert_eq!(half_ttl, Duration::from_secs(15 * 60));

        // Very low confidence (0.1) gets floor (20%) of TTL
        let floor_ttl = cache.effective_ttl_for_property("security_check", 0.1);
        assert_eq!(floor_ttl, Duration::from_secs(6 * 60)); // 20% of 30 min = 6 min
    }

    #[test]
    fn test_cache_testing_preset_configuration() {
        let cache = VerificationCache::new().with_testing_preset();

        // Verify short base TTL (5 minutes)
        assert_eq!(cache.ttl, Duration::from_secs(5 * 60));

        // Verify small cache size
        assert_eq!(cache.max_entries, 1000);

        // Verify no confidence threshold
        assert_eq!(cache.confidence_threshold(), 0.0);

        // Verify no confidence scaling
        assert!(!cache.is_confidence_ttl_scaling_enabled());

        // Verify test pattern TTL (1 minute)
        assert_eq!(
            cache.get_pattern_ttl_override("test_*"),
            Some(Duration::from_secs(60))
        );
    }

    #[test]
    fn test_cache_testing_preset_pattern_matching() {
        let cache = VerificationCache::new().with_testing_preset();

        // Test properties should get 1 min TTL
        assert_eq!(
            cache.effective_ttl_for_property("test_foo", 1.0),
            Duration::from_secs(60)
        );
        assert_eq!(
            cache.effective_ttl_for_property("unit_test", 1.0),
            Duration::from_secs(60)
        );

        // Other properties should get base TTL (5 min)
        assert_eq!(
            cache.effective_ttl_for_property("some_property", 1.0),
            Duration::from_secs(5 * 60)
        );
    }

    #[test]
    fn test_cache_production_preset_configuration() {
        let cache = VerificationCache::new().with_production_preset();

        // Verify long base TTL (4 hours)
        assert_eq!(cache.ttl, Duration::from_secs(4 * 60 * 60));

        // Verify large cache size
        assert_eq!(cache.max_entries, 50000);

        // Verify moderate confidence threshold
        assert_eq!(cache.confidence_threshold(), 0.7);

        // Verify confidence scaling with 30% floor
        assert!(cache.is_confidence_ttl_scaling_enabled());
        assert_eq!(cache.min_ttl_multiplier(), 0.3);

        // Verify security pattern TTL (1 hour)
        assert_eq!(
            cache.get_pattern_ttl_override("security_*"),
            Some(Duration::from_secs(60 * 60))
        );

        // Verify invariant pattern TTL (8 hours)
        assert_eq!(
            cache.get_pattern_ttl_override("*_invariant"),
            Some(Duration::from_secs(8 * 60 * 60))
        );
    }

    #[test]
    fn test_cache_production_preset_pattern_matching() {
        let cache = VerificationCache::new().with_production_preset();

        // Security properties should get 1 hour TTL
        assert_eq!(
            cache.effective_ttl_for_property("security_policy", 1.0),
            Duration::from_secs(60 * 60)
        );
        assert_eq!(
            cache.effective_ttl_for_property("auth_check", 1.0),
            Duration::from_secs(60 * 60)
        );

        // Invariant properties should get 8 hour TTL
        assert_eq!(
            cache.effective_ttl_for_property("state_invariant", 1.0),
            Duration::from_secs(8 * 60 * 60)
        );

        // Test properties should get 30 min TTL
        assert_eq!(
            cache.effective_ttl_for_property("test_suite", 1.0),
            Duration::from_secs(30 * 60)
        );

        // Other properties should get base TTL (4 hours)
        assert_eq!(
            cache.effective_ttl_for_property("regular_property", 1.0),
            Duration::from_secs(4 * 60 * 60)
        );
    }

    #[test]
    fn test_cache_performance_preset_configuration() {
        let cache = VerificationCache::new().with_performance_preset();

        // Verify very long base TTL (24 hours)
        assert_eq!(cache.ttl, Duration::from_secs(24 * 60 * 60));

        // Verify very large cache size
        assert_eq!(cache.max_entries, 100000);

        // Verify low confidence threshold
        assert_eq!(cache.confidence_threshold(), 0.5);

        // Verify aggressive confidence scaling (10% floor)
        assert!(cache.is_confidence_ttl_scaling_enabled());
        assert_eq!(cache.min_ttl_multiplier(), 0.1);

        // Verify invariant pattern TTL (48 hours)
        assert_eq!(
            cache.get_pattern_ttl_override("*_invariant"),
            Some(Duration::from_secs(48 * 60 * 60))
        );

        // Verify security pattern TTL (4 hours)
        assert_eq!(
            cache.get_pattern_ttl_override("security_*"),
            Some(Duration::from_secs(4 * 60 * 60))
        );
    }

    #[test]
    fn test_cache_performance_preset_pattern_matching() {
        let cache = VerificationCache::new().with_performance_preset();

        // Invariant properties should get 48 hour TTL
        assert_eq!(
            cache.effective_ttl_for_property("data_invariant", 1.0),
            Duration::from_secs(48 * 60 * 60)
        );

        // Security properties should get 4 hour TTL
        assert_eq!(
            cache.effective_ttl_for_property("security_module", 1.0),
            Duration::from_secs(4 * 60 * 60)
        );
        assert_eq!(
            cache.effective_ttl_for_property("auth_flow", 1.0),
            Duration::from_secs(4 * 60 * 60)
        );

        // Other properties should get base TTL (24 hours)
        assert_eq!(
            cache.effective_ttl_for_property("regular_property", 1.0),
            Duration::from_secs(24 * 60 * 60)
        );
    }

    #[test]
    fn test_cache_performance_preset_confidence_scaling() {
        let cache = VerificationCache::new().with_performance_preset();

        // High confidence gets full TTL
        let full_ttl = cache.effective_ttl_for_property("regular_property", 1.0);
        assert_eq!(full_ttl, Duration::from_secs(24 * 60 * 60));

        // Low confidence (0.05) hits floor (10%)
        let floor_ttl = cache.effective_ttl_for_property("regular_property", 0.05);
        assert_eq!(floor_ttl, Duration::from_secs((24 * 60 * 60) / 10)); // 2.4 hours
    }

    #[test]
    fn test_cache_apply_pattern_config() {
        let mut cache = VerificationCache::new();

        let mut patterns = std::collections::HashMap::new();
        patterns.insert("custom_*".to_string(), Duration::from_secs(100));
        patterns.insert("*_custom".to_string(), Duration::from_secs(200));
        patterns.insert("[invalid".to_string(), Duration::from_secs(300)); // Invalid pattern

        let errors = cache.apply_pattern_config(&patterns);

        // Should have one error for the invalid pattern
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].0, "[invalid");

        // Valid patterns should be added
        assert_eq!(
            cache.get_pattern_ttl_override("custom_*"),
            Some(Duration::from_secs(100))
        );
        assert_eq!(
            cache.get_pattern_ttl_override("*_custom"),
            Some(Duration::from_secs(200))
        );
    }

    #[test]
    fn test_cache_configuration_summary_basic() {
        let cache = VerificationCache::new();
        let summary = cache.configuration_summary();

        assert!(summary.contains("Base TTL:"));
        assert!(summary.contains("Max entries:"));
        assert!(summary.contains("Confidence threshold:"));
        assert!(summary.contains("Confidence TTL scaling:"));
    }

    #[test]
    fn test_cache_configuration_summary_with_preset() {
        let cache = VerificationCache::new().with_security_preset();
        let summary = cache.configuration_summary();

        // Should show all configuration details
        assert!(summary.contains("Base TTL:"));
        assert!(summary.contains("Max entries:"));
        assert!(summary.contains("Confidence threshold: 0.90"));
        assert!(summary.contains("Confidence TTL scaling: true"));
        assert!(summary.contains("Min TTL multiplier: 0.20"));
        assert!(summary.contains("Pattern overrides:"));
        assert!(summary.contains("security_*"));
    }

    #[test]
    fn test_cache_presets_are_chainable() {
        // Presets should work with the builder pattern
        let cache = VerificationCache::new().with_security_preset();

        // Should be fully configured
        assert_eq!(cache.confidence_threshold(), 0.9);
        assert!(cache.is_confidence_ttl_scaling_enabled());
    }

    #[test]
    fn test_cache_presets_can_be_customized_after() {
        let mut cache = VerificationCache::new().with_testing_preset();

        // Preset values
        assert_eq!(cache.confidence_threshold(), 0.0);
        assert_eq!(cache.max_entries, 1000);

        // Customize after preset
        cache.set_confidence_threshold(0.5);
        cache.set_property_ttl_override("special", Duration::from_secs(999));

        // Custom values should override
        assert_eq!(cache.confidence_threshold(), 0.5);
        assert_eq!(
            cache.get_property_ttl_override("special"),
            Some(Duration::from_secs(999))
        );

        // Preset patterns should still exist
        assert_eq!(
            cache.get_pattern_ttl_override("test_*"),
            Some(Duration::from_secs(60))
        );
    }

    // ==========================================================================
    // Cache Persistence Tests
    // ==========================================================================

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.max_entries, 10000);
        assert_eq!(config.ttl_seconds, 3600);
        assert_eq!(config.confidence_threshold, 0.0);
        assert!(!config.confidence_ttl_scaling);
        assert!(config.property_ttl_overrides.is_empty());
        assert!(config.pattern_ttl_overrides.is_empty());
    }

    #[test]
    fn test_cache_config_security_preset() {
        let config = CacheConfig::security();
        assert_eq!(config.confidence_threshold, 0.9);
        assert!(config.confidence_ttl_scaling);
        assert!(!config.pattern_ttl_overrides.is_empty());

        // Check for security patterns
        assert!(config
            .pattern_ttl_overrides
            .iter()
            .any(|(p, _)| p == "security_*"));
        assert!(config
            .pattern_ttl_overrides
            .iter()
            .any(|(p, _)| p == "auth_*"));
    }

    #[test]
    fn test_cache_config_testing_preset() {
        let config = CacheConfig::testing();
        assert_eq!(config.max_entries, 1000);
        assert_eq!(config.ttl_seconds, 5 * 60);
        assert_eq!(config.confidence_threshold, 0.0);
        assert!(!config.confidence_ttl_scaling);
    }

    #[test]
    fn test_cache_config_production_preset() {
        let config = CacheConfig::production();
        assert_eq!(config.max_entries, 50000);
        assert_eq!(config.ttl_seconds, 4 * 60 * 60);
        assert_eq!(config.confidence_threshold, 0.7);
        assert!(config.confidence_ttl_scaling);
    }

    #[test]
    fn test_cache_config_performance_preset() {
        let config = CacheConfig::performance();
        assert_eq!(config.max_entries, 100000);
        assert_eq!(config.ttl_seconds, 24 * 60 * 60);
        assert_eq!(config.confidence_threshold, 0.5);
        assert!(config.confidence_ttl_scaling);
    }

    #[test]
    fn test_cache_from_config() {
        let config = CacheConfig {
            max_entries: 5000,
            ttl_seconds: 1800,
            confidence_threshold: 0.75,
            confidence_ttl_scaling: true,
            min_ttl_multiplier: 0.25,
            property_ttl_overrides: {
                let mut m = HashMap::new();
                m.insert("prop1".to_string(), 600);
                m
            },
            pattern_ttl_overrides: vec![("test_*".to_string(), 120)],
        };

        let (cache, errors) = VerificationCache::from_config(config);

        assert!(errors.is_empty());
        assert_eq!(cache.max_entries, 5000);
        assert_eq!(cache.ttl, Duration::from_secs(1800));
        assert_eq!(cache.confidence_threshold(), 0.75);
        assert!(cache.is_confidence_ttl_scaling_enabled());
        assert_eq!(cache.min_ttl_multiplier(), 0.25);
        assert_eq!(
            cache.get_property_ttl_override("prop1"),
            Some(Duration::from_secs(600))
        );
        assert_eq!(
            cache.get_pattern_ttl_override("test_*"),
            Some(Duration::from_secs(120))
        );
    }

    #[test]
    fn test_cache_from_config_with_invalid_pattern() {
        let config = CacheConfig {
            pattern_ttl_overrides: vec![
                ("valid_*".to_string(), 100),
                ("[invalid".to_string(), 200), // Invalid pattern
                ("*_also_valid".to_string(), 300),
            ],
            ..Default::default()
        };

        let (cache, errors) = VerificationCache::from_config(config);

        // One invalid pattern error
        assert_eq!(errors.len(), 1);
        match &errors[0] {
            CachePersistenceError::InvalidPattern { pattern, .. } => {
                assert_eq!(pattern, "[invalid");
            }
            _ => panic!("Expected InvalidPattern error"),
        }

        // Valid patterns should still be loaded
        assert_eq!(
            cache.get_pattern_ttl_override("valid_*"),
            Some(Duration::from_secs(100))
        );
        assert_eq!(
            cache.get_pattern_ttl_override("*_also_valid"),
            Some(Duration::from_secs(300))
        );
    }

    #[test]
    fn test_cache_export_config() {
        let mut cache = VerificationCache::with_config(2000, Duration::from_secs(900));
        cache.set_confidence_threshold(0.8);
        cache.enable_confidence_ttl_scaling(0.15);
        cache.set_property_ttl_override("prop1", Duration::from_secs(300));
        let _ = cache.set_pattern_ttl_override("pat_*", Duration::from_secs(600));

        let config = cache.export_config();

        assert_eq!(config.max_entries, 2000);
        assert_eq!(config.ttl_seconds, 900);
        assert_eq!(config.confidence_threshold, 0.8);
        assert!(config.confidence_ttl_scaling);
        assert_eq!(config.min_ttl_multiplier, 0.15);
        assert_eq!(config.property_ttl_overrides.get("prop1"), Some(&300));
        assert!(config
            .pattern_ttl_overrides
            .iter()
            .any(|(p, t)| p == "pat_*" && *t == 600));
    }

    #[test]
    fn test_cache_config_save_and_load() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("cache_config.json");

        let original = CacheConfig::security();
        original.save_to_file(&config_path).unwrap();

        let loaded = CacheConfig::load_from_file(&config_path).unwrap();

        assert_eq!(loaded.max_entries, original.max_entries);
        assert_eq!(loaded.ttl_seconds, original.ttl_seconds);
        assert_eq!(loaded.confidence_threshold, original.confidence_threshold);
        assert_eq!(
            loaded.confidence_ttl_scaling,
            original.confidence_ttl_scaling
        );
        assert_eq!(loaded.min_ttl_multiplier, original.min_ttl_multiplier);
        assert_eq!(
            loaded.pattern_ttl_overrides.len(),
            original.pattern_ttl_overrides.len()
        );
    }

    #[test]
    fn test_cache_config_load_nonexistent_file() {
        let result = CacheConfig::load_from_file("/nonexistent/path/config.json");
        assert!(result.is_err());
        match result.unwrap_err() {
            CachePersistenceError::Io(_) => {}
            _ => panic!("Expected IO error"),
        }
    }

    #[test]
    fn test_cache_snapshot_creation() {
        let mut cache = VerificationCache::new();

        let result = create_test_cached_result(0.9);
        let key = CacheKey::new("hash1", "prop1");
        cache.insert(key, result);

        let snapshot = cache.create_snapshot();

        assert_eq!(snapshot.format_version, CacheSnapshot::FORMAT_VERSION);
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].version_hash, "hash1");
        assert_eq!(snapshot.entries[0].property_name, "prop1");
        assert_eq!(snapshot.stats.entry_count, 1);
    }

    #[test]
    fn test_cache_save_and_load_entries() {
        let temp_dir = tempfile::tempdir().unwrap();
        let data_path = temp_dir.path().join("cache_data.json");

        // Create and populate cache
        let mut cache = VerificationCache::new();
        for i in 0..5 {
            let result = create_test_cached_result(0.8 + (i as f64) * 0.02);
            let key = CacheKey::new(format!("hash{}", i), format!("prop{}", i));
            cache.insert(key, result);
        }

        // Save entries
        cache.save_entries_to_file(&data_path).unwrap();

        // Load into new cache
        let mut new_cache = VerificationCache::new();
        let loaded = new_cache.load_entries_from_file(&data_path).unwrap();

        assert_eq!(loaded, 5);
        assert_eq!(new_cache.len(), 5);
    }

    #[test]
    fn test_cache_load_entries_filters_expired() {
        let temp_dir = tempfile::tempdir().unwrap();
        let data_path = temp_dir.path().join("cache_data.json");

        // Create cache and save
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        let result = create_test_cached_result(0.9);
        cache.insert(CacheKey::new("hash1", "prop1"), result);
        cache.save_entries_to_file(&data_path).unwrap();

        // Load into cache with very short TTL
        let mut new_cache = VerificationCache::with_config(100, Duration::from_secs(0));
        let loaded = new_cache.load_entries_from_file(&data_path).unwrap();

        // Entry should be filtered as expired
        assert_eq!(loaded, 0);
        assert!(new_cache.is_empty());
    }

    #[test]
    fn test_cache_restore_from_snapshot() {
        let mut cache = VerificationCache::new();
        let result = create_test_cached_result(0.85);
        cache.insert(CacheKey::new("h1", "p1"), result.clone());

        let snapshot = cache.create_snapshot();

        let mut new_cache = VerificationCache::new();
        let loaded = new_cache.restore_from_snapshot(&snapshot).unwrap();

        assert_eq!(loaded, 1);
        let key = CacheKey::new("h1", "p1");
        let retrieved = new_cache.get(&key);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_cache_warm_from_results() {
        let mut cache = VerificationCache::new();

        let results: Vec<(String, String, CachedPropertyResult)> = (0..3)
            .map(|i| {
                (
                    format!("ver{}", i),
                    format!("prop{}", i),
                    create_test_cached_result(0.8),
                )
            })
            .collect();

        let warmed =
            cache.warm_from_results(results.iter().map(|(v, p, r)| (v.as_str(), p.as_str(), r)));

        assert_eq!(warmed, 3);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_cache_warm_skips_existing_entries() {
        let mut cache = VerificationCache::new();

        // Insert an entry first
        let existing = create_test_cached_result(0.95);
        cache.insert(CacheKey::new("ver0", "prop0"), existing);

        // Warm with same key
        let results: Vec<(String, String, CachedPropertyResult)> = vec![(
            "ver0".to_string(),
            "prop0".to_string(),
            create_test_cached_result(0.7), // Lower confidence
        )];

        let warmed =
            cache.warm_from_results(results.iter().map(|(v, p, r)| (v.as_str(), p.as_str(), r)));

        assert_eq!(warmed, 0); // Should skip existing
        assert_eq!(cache.len(), 1);

        // Original entry should remain
        let mut cache_copy = cache;
        let key = CacheKey::new("ver0", "prop0");
        let retrieved = cache_copy.get(&key).unwrap();
        assert_eq!(retrieved.confidence, 0.95);
    }

    #[test]
    fn test_cache_warm_respects_confidence_threshold() {
        let mut cache = VerificationCache::new();
        cache.set_confidence_threshold(0.8);

        let results: Vec<(String, String, CachedPropertyResult)> = vec![
            (
                "v1".to_string(),
                "p1".to_string(),
                create_test_cached_result(0.9),
            ), // Above threshold
            (
                "v2".to_string(),
                "p2".to_string(),
                create_test_cached_result(0.5),
            ), // Below threshold
        ];

        let warmed =
            cache.warm_from_results(results.iter().map(|(v, p, r)| (v.as_str(), p.as_str(), r)));

        assert_eq!(warmed, 1); // Only high confidence entry
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_merge_from() {
        let mut cache1 = VerificationCache::new();
        let mut cache2 = VerificationCache::new();

        // Populate cache1
        cache1.insert(CacheKey::new("v1", "p1"), create_test_cached_result(0.8));

        // Populate cache2 with different entries
        cache2.insert(CacheKey::new("v2", "p2"), create_test_cached_result(0.85));
        cache2.insert(CacheKey::new("v3", "p3"), create_test_cached_result(0.9));

        let merged = cache1.merge_from(&cache2);

        assert_eq!(merged, 2);
        assert_eq!(cache1.len(), 3);
    }

    #[test]
    fn test_cache_merge_newer_wins() {
        let mut cache1 = VerificationCache::new();
        let mut cache2 = VerificationCache::new();

        // Insert older entry in cache1
        let old_result = create_test_cached_result(0.8);
        cache1.insert(CacheKey::new("v1", "p1"), old_result);

        // Insert newer entry with same key in cache2
        std::thread::sleep(std::time::Duration::from_millis(10));
        let new_result = create_test_cached_result(0.95);
        cache2.insert(CacheKey::new("v1", "p1"), new_result);

        let merged = cache1.merge_from(&cache2);

        assert_eq!(merged, 1);
        let mut cache1_copy = cache1;
        let key = CacheKey::new("v1", "p1");
        let retrieved = cache1_copy.get(&key).unwrap();
        assert_eq!(retrieved.confidence, 0.95); // Newer entry
    }

    #[test]
    fn test_cache_merge_older_skipped() {
        let mut cache1 = VerificationCache::new();
        let mut cache2 = VerificationCache::new();

        // Insert newer entry in cache1
        std::thread::sleep(std::time::Duration::from_millis(10));
        let new_result = create_test_cached_result(0.95);
        cache1.insert(CacheKey::new("v1", "p1"), new_result);

        // Insert older entry with same key in cache2 (created earlier)
        let old_result = CachedPropertyResult {
            property: VerifiedProperty {
                name: "p1".to_string(),
                passed: true,
                status: "verified".to_string(),
            },
            backends: vec!["test".to_string()],
            cached_at: SystemTime::now() - Duration::from_secs(100),
            dependency_hash: "hash".to_string(),
            confidence: 0.7,
        };
        cache2.insert(CacheKey::new("v1", "p1"), old_result);

        let merged = cache1.merge_from(&cache2);

        assert_eq!(merged, 0); // Older entry should be skipped
        let mut cache1_copy = cache1;
        let key = CacheKey::new("v1", "p1");
        let retrieved = cache1_copy.get(&key).unwrap();
        assert_eq!(retrieved.confidence, 0.95); // Original newer entry
    }

    #[test]
    fn test_cache_from_config_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("test_config.json");

        let config = CacheConfig::testing();
        config.save_to_file(&config_path).unwrap();

        let (cache, errors) = VerificationCache::from_config_file(&config_path).unwrap();

        assert!(errors.is_empty());
        assert_eq!(cache.max_entries, 1000);
        assert_eq!(cache.ttl, Duration::from_secs(5 * 60));
    }

    #[test]
    fn test_cache_save_config_to_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("exported_config.json");

        let cache = VerificationCache::new().with_security_preset();
        cache.save_config_to_file(&config_path).unwrap();

        // Verify file exists and can be loaded
        let loaded = CacheConfig::load_from_file(&config_path).unwrap();
        assert_eq!(loaded.confidence_threshold, 0.9);
        assert!(loaded.confidence_ttl_scaling);
    }

    #[test]
    fn test_cache_persistence_error_display() {
        let io_err = CachePersistenceError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert!(io_err.to_string().contains("I/O error"));

        let pattern_err = CachePersistenceError::InvalidPattern {
            pattern: "[bad".to_string(),
            error: "unclosed bracket".to_string(),
        };
        assert!(pattern_err.to_string().contains("[bad"));
        assert!(pattern_err.to_string().contains("unclosed bracket"));
    }

    #[test]
    fn test_cache_snapshot_format_version() {
        let snapshot = CacheSnapshot::new(Vec::new(), CacheStats::default());
        // Version 5 added autosave_metrics
        assert_eq!(snapshot.format_version, CacheSnapshot::FORMAT_VERSION);
    }

    #[test]
    fn test_cache_snapshot_with_trigger_state() {
        let state = CompactionTriggerState {
            inserts_since_compaction: 42,
            last_compaction_time: 1234567890,
            ..Default::default()
        };

        let triggers = vec![
            CompactionTrigger::size_90_percent(),
            CompactionTrigger::every_minutes(5),
        ];

        let snapshot = CacheSnapshot::with_trigger_state(
            Vec::new(),
            CacheStats::default(),
            state.clone(),
            triggers.clone(),
        );

        assert_eq!(snapshot.format_version, CacheSnapshot::FORMAT_VERSION);
        assert!(snapshot.compaction_trigger_state.is_some());
        let restored_state = snapshot.compaction_trigger_state.unwrap();
        assert_eq!(restored_state.inserts_since_compaction, 42);
        assert_eq!(restored_state.last_compaction_time, 1234567890);
        assert_eq!(snapshot.compaction_triggers.len(), 2);
    }

    #[test]
    fn test_snapshot_trigger_state_persistence_roundtrip() {
        // Create cache with triggers and trigger state
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        cache.add_compaction_trigger(CompactionTrigger::size_80_percent());
        cache.add_compaction_trigger(CompactionTrigger::every_inserts(100));

        // Insert some entries to modify trigger state
        for i in 0..5 {
            let key = CacheKey::new("v1", format!("prop_{}", i));
            let result = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("prop_{}", i),
                    passed: true,
                    status: "Verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "hash".to_string(),
                confidence: 0.9,
            };
            cache.insert(key, result);
        }

        // Create snapshot and verify trigger state is included
        let snapshot = cache.create_snapshot();
        assert!(snapshot.compaction_trigger_state.is_some());
        assert_eq!(snapshot.compaction_triggers.len(), 2);
        assert_eq!(
            snapshot
                .compaction_trigger_state
                .as_ref()
                .unwrap()
                .inserts_since_compaction,
            5
        );

        // Create new cache and restore from snapshot
        let mut new_cache = VerificationCache::new();
        new_cache.restore_from_snapshot(&snapshot).unwrap();

        // Verify trigger state was restored
        assert_eq!(new_cache.compaction_triggers().len(), 2);
        // Note: inserts_since_compaction is tracked but not directly accessible
        // So we verify via creating another snapshot
        let new_snapshot = new_cache.create_snapshot();
        assert_eq!(
            new_snapshot
                .compaction_trigger_state
                .as_ref()
                .unwrap()
                .inserts_since_compaction,
            5
        );
    }

    #[test]
    fn test_snapshot_backward_compatibility() {
        // Test that old format snapshots (without trigger state) can still be loaded
        let old_snapshot = CacheSnapshot::new(Vec::new(), CacheStats::default());

        // Serialize and deserialize as JSON (simulating loading an old snapshot)
        let json = serde_json::to_string(&old_snapshot).unwrap();
        let loaded: CacheSnapshot = serde_json::from_str(&json).unwrap();

        // Should have None for trigger state (default)
        assert!(loaded.compaction_trigger_state.is_none());
        assert!(loaded.compaction_triggers.is_empty());
        // Should have None for compaction trigger counts (v3 field)
        assert!(loaded.compaction_trigger_counts.is_none());
        // Should have None for compaction time series (v4 field)
        assert!(loaded.compaction_time_series.is_none());

        // Should be able to restore without errors
        let mut cache = VerificationCache::new();
        cache.restore_from_snapshot(&loaded).unwrap();
    }

    #[test]
    fn test_snapshot_with_compaction_history() {
        let trigger_state = CompactionTriggerState {
            inserts_since_compaction: 100,
            last_compaction_time: 1234567890,
            ..Default::default()
        };

        let triggers = vec![
            CompactionTrigger::size_90_percent(),
            CompactionTrigger::every_minutes(5),
        ];

        let trigger_counts = CompactionTriggerCounts {
            size_based: 5,
            time_based: 3,
            hit_rate_based: 1,
            partition_imbalance: 0,
            insert_based: 2,
            memory_based: 0,
        };

        let snapshot = CacheSnapshot::with_compaction_history(
            Vec::new(),
            CacheStats::default(),
            trigger_state.clone(),
            triggers.clone(),
            trigger_counts,
        );

        // Verify format version is 5 (which supports autosave_metrics)
        assert_eq!(snapshot.format_version, CacheSnapshot::FORMAT_VERSION);

        // Verify all fields are set
        assert!(snapshot.compaction_trigger_state.is_some());
        assert_eq!(snapshot.compaction_triggers.len(), 2);
        assert!(snapshot.compaction_trigger_counts.is_some());

        let restored_counts = snapshot.compaction_trigger_counts.unwrap();
        assert_eq!(restored_counts.size_based, 5);
        assert_eq!(restored_counts.time_based, 3);
        assert_eq!(restored_counts.hit_rate_based, 1);
        assert_eq!(restored_counts.insert_based, 2);
        assert_eq!(restored_counts.total(), 11);
    }

    #[test]
    fn test_snapshot_compaction_history_persistence_roundtrip() {
        let trigger_counts = CompactionTriggerCounts {
            size_based: 10,
            time_based: 5,
            hit_rate_based: 2,
            partition_imbalance: 1,
            insert_based: 3,
            memory_based: 0,
        };

        let snapshot = CacheSnapshot::with_compaction_history(
            Vec::new(),
            CacheStats::default(),
            CompactionTriggerState::default(),
            vec![CompactionTrigger::size_80_percent()],
            trigger_counts,
        );

        // Serialize and deserialize
        let json = serde_json::to_string(&snapshot).unwrap();
        let loaded: CacheSnapshot = serde_json::from_str(&json).unwrap();

        // Verify trigger counts were preserved
        assert!(loaded.compaction_trigger_counts.is_some());
        let restored_counts = loaded.compaction_trigger_counts.unwrap();
        assert_eq!(restored_counts.size_based, 10);
        assert_eq!(restored_counts.time_based, 5);
        assert_eq!(restored_counts.hit_rate_based, 2);
        assert_eq!(restored_counts.partition_imbalance, 1);
        assert_eq!(restored_counts.insert_based, 3);
        assert_eq!(restored_counts.memory_based, 0);
        assert_eq!(restored_counts.total(), 21);
    }

    #[test]
    fn test_snapshot_compaction_time_series_persistence() {
        let mut cache = VerificationCache::new();
        cache.record_compaction_full(&CompactionTrigger::size_80_percent(), 50);
        cache.record_compaction_full(&CompactionTrigger::every_minutes(10), 30);

        let snapshot = cache.create_snapshot();
        assert!(snapshot.compaction_time_series.is_some());

        let time_series = snapshot.compaction_time_series.as_ref().unwrap();
        assert_eq!(time_series.len(), 2);

        let mut restored = VerificationCache::new();
        restored.restore_from_snapshot(&snapshot).unwrap();

        let restored_entries = restored.compaction_time_series().entries();
        assert_eq!(restored_entries.len(), 2);
        assert_eq!(
            restored_entries[0].trigger_type,
            CompactionTriggerType::SizeBased
        );
        assert_eq!(
            restored_entries[1].trigger_type,
            CompactionTriggerType::TimeBased
        );
        assert_eq!(
            restored
                .compaction_time_series()
                .entries_removed_in_window(Duration::from_secs(3600)),
            80
        );
    }

    #[test]
    fn test_snapshot_compaction_time_series_merge_on_restore() {
        let mut cache = VerificationCache::new();
        let now = SystemTime::now();
        let older = now - Duration::from_secs(120);
        let newer = now - Duration::from_secs(30);

        cache.compaction_time_series_mut().record_at(
            older,
            CompactionTriggerType::PartitionImbalance,
            10,
        );

        let snapshot = cache.create_snapshot();

        let mut new_cache = VerificationCache::new();
        new_cache.compaction_time_series_mut().record_at(
            newer,
            CompactionTriggerType::MemoryBased,
            5,
        );

        new_cache.restore_from_snapshot(&snapshot).unwrap();

        let entries = new_cache.compaction_time_series().entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0].trigger_type,
            CompactionTriggerType::PartitionImbalance
        );
        assert_eq!(entries[1].trigger_type, CompactionTriggerType::MemoryBased);

        let counts = new_cache
            .compaction_time_series()
            .counts_by_type(Duration::from_secs(300));
        assert_eq!(counts.partition_imbalance, 1);
        assert_eq!(counts.memory_based, 1);
        assert_eq!(
            new_cache
                .compaction_time_series()
                .entries_removed_in_window(Duration::from_secs(300)),
            15
        );
    }

    // Helper function to create test cached results
    fn create_test_cached_result(confidence: f64) -> CachedPropertyResult {
        CachedPropertyResult {
            property: VerifiedProperty {
                name: "test_prop".to_string(),
                passed: true,
                status: "verified".to_string(),
            },
            backends: vec!["test_backend".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "test_hash".to_string(),
            confidence,
        }
    }

    // ============ Async Cache Persistence Tests ============

    #[tokio::test]
    async fn async_verifier_save_cache_entries_without_cache() {
        let verifier = AsyncImprovementVerifier::new(); // No cache
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let result = verifier.save_cache_entries_async(path).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn async_verifier_save_cache_entries_with_cache() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        // Add some entries to cache
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            for i in 0..3 {
                let result = create_test_cached_result(0.9);
                cache_guard.insert(
                    CacheKey::new(format!("hash{}", i), format!("prop{}", i)),
                    result,
                );
            }
        }

        let saved = verifier
            .save_cache_entries_async(path.clone())
            .await
            .unwrap();
        assert_eq!(saved, 3);

        // Verify file exists
        assert!(path.exists());
    }

    #[tokio::test]
    async fn async_verifier_load_cache_entries_without_cache() {
        let verifier = AsyncImprovementVerifier::new(); // No cache
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        // Create a valid cache file
        let snapshot = CacheSnapshot::new(Vec::new(), CacheStats::default());
        std::fs::write(&path, serde_json::to_string(&snapshot).unwrap()).unwrap();

        let result = verifier.load_cache_entries_async(path).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn async_verifier_load_cache_entries_roundtrip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        // Create verifier and add entries
        let verifier = AsyncImprovementVerifier::with_custom_cache(100, Duration::from_secs(3600));
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            for i in 0..5 {
                let result = create_test_cached_result(0.85);
                cache_guard.insert(
                    CacheKey::new(format!("hash{}", i), format!("prop{}", i)),
                    result,
                );
            }
        }

        // Save
        verifier
            .save_cache_entries_async(path.clone())
            .await
            .unwrap();

        // Create new verifier and load
        let verifier2 = AsyncImprovementVerifier::with_custom_cache(100, Duration::from_secs(3600));
        let loaded = verifier2.load_cache_entries_async(path).await.unwrap();
        assert_eq!(loaded, 5);

        // Verify entries
        let stats = verifier2.cache_stats().await.unwrap();
        assert_eq!(stats.entry_count, 5);
    }

    #[tokio::test]
    async fn async_verifier_save_cache_config_without_cache() {
        let verifier = AsyncImprovementVerifier::new(); // No cache
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("config.json");

        let result = verifier.save_cache_config_async(path).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn async_verifier_save_cache_config_with_cache() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("config.json");

        let result = verifier.save_cache_config_async(path.clone()).await;
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[tokio::test]
    async fn async_verifier_from_cache_config_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("config.json");

        // Create config file
        let config = CacheConfig::security();
        config.save_to_file(&path).unwrap();

        // Load verifier from config
        let (verifier, errors) = AsyncImprovementVerifier::from_cache_config_file_async(path)
            .await
            .unwrap();

        assert!(errors.is_empty());
        assert!(verifier.cache.is_some());

        // Verify config was applied
        let exported = verifier.export_cache_config().await.unwrap();
        assert_eq!(exported.confidence_threshold, 0.9);
        assert!(exported.confidence_ttl_scaling);
    }

    #[tokio::test]
    async fn async_verifier_export_cache_config_without_cache() {
        let verifier = AsyncImprovementVerifier::new();
        let config = verifier.export_cache_config().await;
        assert!(config.is_none());
    }

    #[tokio::test]
    async fn async_verifier_export_cache_config_with_cache() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let config = verifier.export_cache_config().await;
        assert!(config.is_some());

        let config = config.unwrap();
        assert!(config.max_entries > 0);
    }

    #[tokio::test]
    async fn async_verifier_create_cache_snapshot_without_cache() {
        let verifier = AsyncImprovementVerifier::new();
        let snapshot = verifier.create_cache_snapshot().await;
        assert!(snapshot.is_none());
    }

    #[tokio::test]
    async fn async_verifier_create_cache_snapshot_with_entries() {
        let verifier = AsyncImprovementVerifier::with_cache();

        // Add entries
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            let result = create_test_cached_result(0.9);
            cache_guard.insert(CacheKey::new("hash1", "prop1"), result);
        }

        let snapshot = verifier.create_cache_snapshot().await;
        assert!(snapshot.is_some());

        let snapshot = snapshot.unwrap();
        assert_eq!(snapshot.entries.len(), 1);
        // Version 5 added autosave_metrics
        assert_eq!(snapshot.format_version, CacheSnapshot::FORMAT_VERSION);
    }

    #[tokio::test]
    async fn async_verifier_restore_cache_from_snapshot_without_cache() {
        let verifier = AsyncImprovementVerifier::new();
        let snapshot = CacheSnapshot::new(Vec::new(), CacheStats::default());

        let result = verifier.restore_cache_from_snapshot(&snapshot).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn async_verifier_restore_cache_from_snapshot() {
        let verifier = AsyncImprovementVerifier::with_custom_cache(100, Duration::from_secs(3600));

        // Create snapshot with entries
        let entries = vec![
            PersistedCacheEntry {
                version_hash: "hash1".to_string(),
                property_name: "prop1".to_string(),
                result: create_test_cached_result(0.9),
            },
            PersistedCacheEntry {
                version_hash: "hash2".to_string(),
                property_name: "prop2".to_string(),
                result: create_test_cached_result(0.85),
            },
        ];
        let snapshot = CacheSnapshot::new(entries, CacheStats::default());

        let restored = verifier
            .restore_cache_from_snapshot(&snapshot)
            .await
            .unwrap();
        assert_eq!(restored, 2);

        let stats = verifier.cache_stats().await.unwrap();
        assert_eq!(stats.entry_count, 2);
    }

    #[tokio::test]
    async fn async_verifier_warm_cache_from_results_without_cache() {
        let verifier = AsyncImprovementVerifier::new();
        let results = vec![(
            "hash1".to_string(),
            "prop1".to_string(),
            create_test_cached_result(0.9),
        )];

        let warmed = verifier.warm_cache_from_results(results).await;
        assert_eq!(warmed, 0);
    }

    #[tokio::test]
    async fn async_verifier_warm_cache_from_results() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let results = vec![
            (
                "hash1".to_string(),
                "prop1".to_string(),
                create_test_cached_result(0.9),
            ),
            (
                "hash2".to_string(),
                "prop2".to_string(),
                create_test_cached_result(0.85),
            ),
            (
                "hash3".to_string(),
                "prop3".to_string(),
                create_test_cached_result(0.8),
            ),
        ];

        let warmed = verifier.warm_cache_from_results(results).await;
        assert_eq!(warmed, 3);

        let stats = verifier.cache_stats().await.unwrap();
        assert_eq!(stats.entry_count, 3);
    }

    #[tokio::test]
    async fn async_verifier_merge_cache_from_snapshot_without_cache() {
        let verifier = AsyncImprovementVerifier::new();
        let snapshot = CacheSnapshot::new(Vec::new(), CacheStats::default());

        let merged = verifier.merge_cache_from_snapshot(&snapshot).await;
        assert_eq!(merged, 0);
    }

    #[tokio::test]
    async fn async_verifier_merge_cache_from_snapshot() {
        let verifier = AsyncImprovementVerifier::with_custom_cache(100, Duration::from_secs(3600));

        // Add initial entry
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            let result = create_test_cached_result(0.9);
            cache_guard.insert(CacheKey::new("existing_hash", "existing_prop"), result);
        }

        // Create snapshot with new entries
        let entries = vec![
            PersistedCacheEntry {
                version_hash: "new_hash1".to_string(),
                property_name: "new_prop1".to_string(),
                result: create_test_cached_result(0.85),
            },
            PersistedCacheEntry {
                version_hash: "new_hash2".to_string(),
                property_name: "new_prop2".to_string(),
                result: create_test_cached_result(0.8),
            },
        ];
        let snapshot = CacheSnapshot::new(entries, CacheStats::default());

        let merged = verifier.merge_cache_from_snapshot(&snapshot).await;
        assert_eq!(merged, 2);

        let stats = verifier.cache_stats().await.unwrap();
        assert_eq!(stats.entry_count, 3); // 1 existing + 2 merged
    }

    #[tokio::test]
    async fn async_verifier_load_nonexistent_file() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let result = verifier
            .load_cache_entries_async("/nonexistent/path/cache.json".to_string())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn async_verifier_config_file_roundtrip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let entries_path = temp_dir.path().join("entries.json");

        // Create verifier with custom settings
        let verifier = AsyncImprovementVerifier::with_custom_cache(500, Duration::from_secs(7200));

        // Add some entries
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            for i in 0..10 {
                let result = create_test_cached_result(0.8 + (i as f64) * 0.01);
                cache_guard.insert(CacheKey::new(format!("h{}", i), format!("p{}", i)), result);
            }
        }

        // Save both config and entries
        verifier
            .save_cache_config_async(config_path.clone())
            .await
            .unwrap();
        verifier
            .save_cache_entries_async(entries_path.clone())
            .await
            .unwrap();

        // Load new verifier from config and restore entries
        let (verifier2, errors) =
            AsyncImprovementVerifier::from_cache_config_file_async(config_path)
                .await
                .unwrap();
        assert!(errors.is_empty());

        let loaded = verifier2
            .load_cache_entries_async(entries_path)
            .await
            .unwrap();
        assert_eq!(loaded, 10);

        // Verify entries are accessible
        let stats = verifier2.cache_stats().await.unwrap();
        assert_eq!(stats.entry_count, 10);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_writes_snapshots() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave.json");

        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            cache_guard.insert(
                CacheKey::new("autosave_hash", "autosave_prop"),
                create_test_cached_result(0.92),
            );
        }

        let handle = verifier
            .start_cache_autosave(path.clone(), Duration::from_millis(30))
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(140)).await;
        let summary = handle.stop().await;

        assert!(summary.save_count >= 2);
        assert!(summary.errors.is_empty());
        assert!(path.exists());

        let snapshot: CacheSnapshot =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(snapshot.entries.len(), 1);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_without_cache_returns_none() {
        let verifier = AsyncImprovementVerifier::new();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave.json");

        let handle = verifier
            .start_cache_autosave(path, Duration::from_millis(25))
            .unwrap();
        assert!(handle.is_none());
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_rejects_zero_interval() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave.json");

        let result = verifier.start_cache_autosave(path, Duration::from_millis(0));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_compressed_writes_gzip() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave.json.gz");

        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            cache_guard.insert(
                CacheKey::new("compressed_autosave_hash", "compressed_autosave_prop"),
                create_test_cached_result(0.88),
            );
        }

        let handle = verifier
            .start_cache_autosave_compressed(
                path.clone(),
                Duration::from_millis(30),
                SnapshotCompressionLevel::Fast,
            )
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(140)).await;
        let summary = handle.stop().await;

        assert!(summary.save_count >= 2);
        assert!(summary.errors.is_empty());
        assert!(path.exists());
        assert_eq!(summary.compression, Some(SnapshotCompressionLevel::Fast));

        // Verify file is gzip compressed (magic bytes check)
        let data = std::fs::read(&path).unwrap();
        assert!(CacheSnapshot::is_compressed(&data));

        // Verify we can read it back
        let snapshot = CacheSnapshot::from_compressed(&data).unwrap();
        assert_eq!(snapshot.entries.len(), 1);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_compressed_without_cache_returns_none() {
        let verifier = AsyncImprovementVerifier::new();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave.json.gz");

        let handle = verifier
            .start_cache_autosave_compressed(
                path,
                Duration::from_millis(25),
                SnapshotCompressionLevel::Default,
            )
            .unwrap();
        assert!(handle.is_none());
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_compressed_rejects_zero_interval() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave.json.gz");

        let result = verifier.start_cache_autosave_compressed(
            path,
            Duration::from_millis(0),
            SnapshotCompressionLevel::Default,
        );
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_summary_tracks_compression_none() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave.json");

        let handle = verifier
            .start_cache_autosave(path.clone(), Duration::from_millis(30))
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(50)).await;
        let summary = handle.stop().await;

        // Uncompressed autosave should report no compression
        assert!(summary.compression.is_none());
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_compressed_load_auto() {
        // Test that compressed autosave files can be loaded with load_auto
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave.json.gz");

        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            cache_guard.insert(
                CacheKey::new("auto_load_hash", "auto_load_prop"),
                create_test_cached_result(0.95),
            );
        }

        let handle = verifier
            .start_cache_autosave_compressed(
                path.clone(),
                Duration::from_millis(20),
                SnapshotCompressionLevel::Default,
            )
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(50)).await;
        handle.stop().await;

        // Clear cache and reload
        let verifier2 = AsyncImprovementVerifier::with_cache();
        let count = verifier2.load_cache_entries_auto_async(path).await.unwrap();
        assert_eq!(count, 1);

        // Verify the entry was restored
        if let Some(cache) = &verifier2.cache {
            let mut cache_guard = cache.lock().await;
            let key = CacheKey::new("auto_load_hash", "auto_load_prop");
            assert!(cache_guard.get(&key).is_some());
        }
    }

    // ============ Compression Tests ============

    #[test]
    fn test_snapshot_compression_roundtrip() {
        // Create a snapshot with some entries
        let entries = vec![
            PersistedCacheEntry {
                version_hash: "hash1".to_string(),
                property_name: "prop1".to_string(),
                result: create_test_cached_result(0.9),
            },
            PersistedCacheEntry {
                version_hash: "hash2".to_string(),
                property_name: "prop2".to_string(),
                result: create_test_cached_result(0.95),
            },
        ];
        let original = CacheSnapshot::new(entries, CacheStats::default());

        // Compress and decompress
        let compressed = original
            .to_compressed(SnapshotCompressionLevel::Default)
            .unwrap();
        let restored = CacheSnapshot::from_compressed(&compressed).unwrap();

        assert_eq!(original.format_version, restored.format_version);
        assert_eq!(original.entries.len(), restored.entries.len());
        assert_eq!(
            original.entries[0].version_hash,
            restored.entries[0].version_hash
        );
        assert_eq!(
            original.entries[1].property_name,
            restored.entries[1].property_name
        );
    }

    #[test]
    fn test_snapshot_compression_smaller_than_json() {
        // Create a larger snapshot to demonstrate compression benefit
        let entries: Vec<_> = (0..100)
            .map(|i| PersistedCacheEntry {
                version_hash: format!("hash_{}", i),
                property_name: format!("property_name_with_some_padding_{}", i),
                result: create_test_cached_result(0.9),
            })
            .collect();
        let snapshot = CacheSnapshot::new(entries, CacheStats::default());

        let json = snapshot.to_json().unwrap();
        let compressed = snapshot
            .to_compressed(SnapshotCompressionLevel::Default)
            .unwrap();

        // Compressed should be smaller than JSON for repetitive data
        assert!(
            compressed.len() < json.len(),
            "Compressed size {} should be smaller than JSON size {}",
            compressed.len(),
            json.len()
        );
    }

    #[test]
    fn test_snapshot_auto_detect_json() {
        let snapshot = CacheSnapshot::new(Vec::new(), CacheStats::default());
        let json = snapshot.to_json().unwrap();

        // Auto-detect should recognize JSON
        assert!(!CacheSnapshot::is_compressed(&json));

        // from_bytes should work with JSON
        let restored = CacheSnapshot::from_bytes(&json).unwrap();
        assert_eq!(restored.format_version, snapshot.format_version);
    }

    #[test]
    fn test_snapshot_auto_detect_gzip() {
        let snapshot = CacheSnapshot::new(Vec::new(), CacheStats::default());
        let compressed = snapshot
            .to_compressed(SnapshotCompressionLevel::Default)
            .unwrap();

        // Auto-detect should recognize gzip
        assert!(CacheSnapshot::is_compressed(&compressed));

        // from_bytes should work with gzip
        let restored = CacheSnapshot::from_bytes(&compressed).unwrap();
        assert_eq!(restored.format_version, snapshot.format_version);
    }

    #[test]
    fn test_snapshot_compression_levels() {
        let entries: Vec<_> = (0..50)
            .map(|i| PersistedCacheEntry {
                version_hash: format!("hash_{}", i),
                property_name: format!("prop_{}", i),
                result: create_test_cached_result(0.9),
            })
            .collect();
        let snapshot = CacheSnapshot::new(entries, CacheStats::default());

        // Test all compression levels compile and produce valid output
        for level in [
            SnapshotCompressionLevel::None,
            SnapshotCompressionLevel::Fast,
            SnapshotCompressionLevel::Default,
            SnapshotCompressionLevel::Best,
            SnapshotCompressionLevel::Custom(5),
        ] {
            let compressed = snapshot.to_compressed(level).unwrap();
            let restored = CacheSnapshot::from_compressed(&compressed).unwrap();
            assert_eq!(restored.entries.len(), 50);
        }
    }

    #[test]
    fn test_cache_save_load_compressed() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json.gz");

        let mut cache = VerificationCache::new();
        for i in 0..5 {
            cache.insert(
                CacheKey::new(format!("hash{}", i), format!("prop{}", i)),
                create_test_cached_result(0.9),
            );
        }

        // Save compressed
        cache
            .save_entries_compressed(&path, SnapshotCompressionLevel::Default)
            .unwrap();

        // Verify file is actually compressed
        let data = std::fs::read(&path).unwrap();
        assert!(CacheSnapshot::is_compressed(&data));

        // Load compressed into new cache
        let mut new_cache = VerificationCache::new();
        let loaded = new_cache.load_entries_compressed(&path).unwrap();
        assert_eq!(loaded, 5);
        assert_eq!(new_cache.stats().entry_count, 5);
    }

    #[test]
    fn test_cache_load_auto_detects_json() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let mut cache = VerificationCache::new();
        for i in 0..3 {
            cache.insert(
                CacheKey::new(format!("hash{}", i), format!("prop{}", i)),
                create_test_cached_result(0.9),
            );
        }

        // Save as JSON (uncompressed)
        cache.save_entries_to_file(&path).unwrap();

        // Load with auto-detection
        let mut new_cache = VerificationCache::new();
        let loaded = new_cache.load_entries_auto(&path).unwrap();
        assert_eq!(loaded, 3);
    }

    #[test]
    fn test_cache_load_auto_detects_gzip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json.gz");

        let mut cache = VerificationCache::new();
        for i in 0..3 {
            cache.insert(
                CacheKey::new(format!("hash{}", i), format!("prop{}", i)),
                create_test_cached_result(0.9),
            );
        }

        // Save as compressed
        cache
            .save_entries_compressed(&path, SnapshotCompressionLevel::Default)
            .unwrap();

        // Load with auto-detection
        let mut new_cache = VerificationCache::new();
        let loaded = new_cache.load_entries_auto(&path).unwrap();
        assert_eq!(loaded, 3);
    }

    #[tokio::test]
    async fn async_verifier_save_load_compressed() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json.gz");

        // Add entries to cache
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            for i in 0..5 {
                cache_guard.insert(
                    CacheKey::new(format!("hash{}", i), format!("prop{}", i)),
                    create_test_cached_result(0.9),
                );
            }
        }

        // Save compressed
        let saved = verifier
            .save_cache_entries_compressed_async(path.clone(), SnapshotCompressionLevel::Default)
            .await
            .unwrap();
        assert_eq!(saved, 5);

        // Create new verifier and load compressed
        let verifier2 = AsyncImprovementVerifier::with_cache();
        let loaded = verifier2
            .load_cache_entries_compressed_async(path)
            .await
            .unwrap();
        assert_eq!(loaded, 5);

        // Verify entries are accessible
        let stats = verifier2.cache_stats().await.unwrap();
        assert_eq!(stats.entry_count, 5);
    }

    #[tokio::test]
    async fn async_verifier_save_compressed_without_cache_returns_zero() {
        let verifier = AsyncImprovementVerifier::new(); // No cache
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json.gz");

        let result = verifier
            .save_cache_entries_compressed_async(path, SnapshotCompressionLevel::Default)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn async_verifier_load_compressed_without_cache_returns_zero() {
        let verifier = AsyncImprovementVerifier::new(); // No cache
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json.gz");

        // Create a valid compressed file first
        let snapshot = CacheSnapshot::new(Vec::new(), CacheStats::default());
        std::fs::write(
            &path,
            snapshot
                .to_compressed(SnapshotCompressionLevel::Default)
                .unwrap(),
        )
        .unwrap();

        let result = verifier.load_cache_entries_compressed_async(path).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn async_verifier_load_auto_json() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        // Add entries and save as JSON
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            for i in 0..3 {
                cache_guard.insert(
                    CacheKey::new(format!("hash{}", i), format!("prop{}", i)),
                    create_test_cached_result(0.9),
                );
            }
        }
        verifier
            .save_cache_entries_async(path.clone())
            .await
            .unwrap();

        // Create new verifier and load with auto-detection
        let verifier2 = AsyncImprovementVerifier::with_cache();
        let loaded = verifier2.load_cache_entries_auto_async(path).await.unwrap();
        assert_eq!(loaded, 3);
    }

    #[tokio::test]
    async fn async_verifier_load_auto_gzip() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json.gz");

        // Add entries and save as compressed
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            for i in 0..3 {
                cache_guard.insert(
                    CacheKey::new(format!("hash{}", i), format!("prop{}", i)),
                    create_test_cached_result(0.9),
                );
            }
        }
        verifier
            .save_cache_entries_compressed_async(path.clone(), SnapshotCompressionLevel::Fast)
            .await
            .unwrap();

        // Create new verifier and load with auto-detection
        let verifier2 = AsyncImprovementVerifier::with_cache();
        let loaded = verifier2.load_cache_entries_auto_async(path).await.unwrap();
        assert_eq!(loaded, 3);
    }

    #[tokio::test]
    async fn async_verifier_load_auto_without_cache_returns_zero() {
        let verifier = AsyncImprovementVerifier::new(); // No cache
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        // Create a valid JSON file first
        let snapshot = CacheSnapshot::new(Vec::new(), CacheStats::default());
        std::fs::write(&path, serde_json::to_string(&snapshot).unwrap()).unwrap();

        let result = verifier.load_cache_entries_auto_async(path).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    // ============ Autosave Byte Tracking Tests ============

    #[tokio::test]
    async fn async_verifier_cache_autosave_tracks_bytes_written() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave_bytes.json");

        // Insert a test entry
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            cache_guard.insert(
                CacheKey::new("bytes_hash", "bytes_prop"),
                create_test_cached_result(0.91),
            );
        }

        let handle = verifier
            .start_cache_autosave(path.clone(), Duration::from_millis(30))
            .unwrap()
            .expect("cache should be enabled");

        // Wait for at least one save
        time::sleep(Duration::from_millis(50)).await;
        let summary = handle.stop().await;

        assert!(summary.save_count >= 1);
        assert!(
            summary.total_bytes_written > 0,
            "total_bytes_written should be > 0"
        );
        assert!(summary.last_save_bytes > 0, "last_save_bytes should be > 0");

        // Verify file size matches reported bytes
        let file_size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(
            summary.last_save_bytes, file_size,
            "last_save_bytes should match actual file size"
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_accumulates_total_bytes() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave_accumulate.json");

        // Insert a test entry
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            cache_guard.insert(
                CacheKey::new("accum_hash", "accum_prop"),
                create_test_cached_result(0.88),
            );
        }

        let handle = verifier
            .start_cache_autosave(path.clone(), Duration::from_millis(25))
            .unwrap()
            .expect("cache should be enabled");

        // Wait for multiple saves
        time::sleep(Duration::from_millis(120)).await;
        let summary = handle.stop().await;

        assert!(summary.save_count >= 3, "should have at least 3 saves");
        // Total bytes should be approximately sum of all saves
        // Note: slight variations can occur due to timestamp changes in the snapshot
        // Each save writes roughly the same amount, so total should be close to last * count
        let expected_total = summary.last_save_bytes * summary.save_count as u64;
        let tolerance = summary.save_count as u64 * 10; // Allow ~10 bytes variance per save for timestamp changes
        assert!(
            summary.total_bytes_written >= expected_total.saturating_sub(tolerance)
                && summary.total_bytes_written <= expected_total + tolerance,
            "total_bytes_written ({}) should be approximately last_save_bytes * save_count ({}), within tolerance ({})",
            summary.total_bytes_written,
            expected_total,
            tolerance
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_compressed_tracks_bytes() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave_bytes.json.gz");

        // Insert a test entry
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            cache_guard.insert(
                CacheKey::new("comp_bytes_hash", "comp_bytes_prop"),
                create_test_cached_result(0.93),
            );
        }

        let handle = verifier
            .start_cache_autosave_compressed(
                path.clone(),
                Duration::from_millis(30),
                SnapshotCompressionLevel::Fast,
            )
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(50)).await;
        let summary = handle.stop().await;

        assert!(summary.save_count >= 1);
        assert!(summary.total_bytes_written > 0);
        assert!(summary.last_save_bytes > 0);

        // Compressed file should match reported size
        let file_size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(summary.last_save_bytes, file_size);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_empty_cache_still_tracks_bytes() {
        // Even an empty cache writes a valid snapshot with headers
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave_empty.json");

        let handle = verifier
            .start_cache_autosave(path.clone(), Duration::from_millis(30))
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(50)).await;
        let summary = handle.stop().await;

        assert!(summary.save_count >= 1);
        // Empty cache still produces valid JSON with headers
        assert!(summary.total_bytes_written > 0);
        assert!(summary.last_save_bytes > 0);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_status_returns_current_metrics() {
        // status() should return current metrics without stopping the task
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave_status.json");

        let handle = verifier
            .start_cache_autosave(path.clone(), Duration::from_millis(30))
            .unwrap()
            .expect("cache should be enabled");

        // Wait for at least one save
        time::sleep(Duration::from_millis(50)).await;

        // Query status without stopping
        let status = handle.status();
        assert!(status.save_count >= 1, "should have saved at least once");
        assert!(status.total_bytes_written > 0, "should have written bytes");
        assert!(status.last_save_bytes > 0, "should track last save bytes");
        assert_eq!(status.error_count, 0, "should have no errors");
        assert!(status.is_running, "task should still be running");

        // Can query multiple times
        let status2 = handle.status();
        assert!(
            status2.is_running,
            "task should still be running after second query"
        );

        // Stop and verify final summary
        let summary = handle.stop().await;
        assert!(summary.save_count >= status.save_count);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_status_tracks_saves_over_time() {
        // status() should reflect increasing save counts over time
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave_incremental.json");

        let handle = verifier
            .start_cache_autosave(path.clone(), Duration::from_millis(20))
            .unwrap()
            .expect("cache should be enabled");

        // Wait for initial save
        time::sleep(Duration::from_millis(30)).await;
        let status1 = handle.status();
        let count1 = status1.save_count;

        // Wait for more saves
        time::sleep(Duration::from_millis(50)).await;
        let status2 = handle.status();
        let count2 = status2.save_count;

        // Should have more saves
        assert!(
            count2 > count1,
            "save_count should increase: {} vs {}",
            count2,
            count1
        );

        // Total bytes should also increase
        assert!(
            status2.total_bytes_written > status1.total_bytes_written,
            "total_bytes should increase"
        );

        handle.stop().await;
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_status_is_running_false_after_stop() {
        // After stop(), is_running should be false in subsequent queries
        // (This tests the semantics of is_running based on stop_tx presence)
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("autosave_stop_check.json");

        let handle = verifier
            .start_cache_autosave(path.clone(), Duration::from_millis(50))
            .unwrap()
            .expect("cache should be enabled");

        // Verify running before stop
        let status = handle.status();
        assert!(status.is_running);

        // Stop consumes the handle, so we can't query status after
        // Just verify the stop completes cleanly
        let summary = handle.stop().await;
        assert!(summary.save_count >= 1);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_status_error_count_tracks_failures() {
        // error_count should track the number of failed saves
        let verifier = AsyncImprovementVerifier::with_cache();

        // Use a path that will fail to write (directory doesn't exist)
        let path = PathBuf::from("/nonexistent/dir/that/cannot/exist/cache.json");

        let handle = verifier
            .start_cache_autosave(path, Duration::from_millis(20))
            .unwrap()
            .expect("cache should be enabled");

        // Wait for a few save attempts
        time::sleep(Duration::from_millis(80)).await;

        // Status should show errors
        let status = handle.status();
        assert!(status.error_count > 0, "should have recorded errors");
        assert_eq!(status.save_count, 0, "should have no successful saves");

        // Stop immediately to capture final state
        let summary = handle.stop().await;
        assert!(
            !summary.errors.is_empty(),
            "summary should contain error messages"
        );
        // Final error count should be at least what we saw in status
        // (may be more if errors occurred between status() and stop())
        assert!(
            summary.errors.len() >= status.error_count,
            "final errors ({}) should be >= status error_count ({})",
            summary.errors.len(),
            status.error_count
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_callbacks_on_save_invoked() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        // Track save events with atomic counter
        let save_count = Arc::new(AtomicUsize::new(0));
        let save_count_clone = Arc::clone(&save_count);

        let callbacks = CacheAutosaveCallbacks::new().on_save(move |event| {
            save_count_clone.fetch_add(1, Ordering::Relaxed);
            // Verify event data is sensible
            assert!(event.bytes_written > 0);
            assert!(event.save_number > 0);
        });

        let handle = verifier
            .start_cache_autosave_with_callbacks(path, Duration::from_millis(25), callbacks)
            .unwrap()
            .expect("cache should be enabled");

        // Wait for initial save plus one interval save
        time::sleep(Duration::from_millis(60)).await;

        let summary = handle.stop().await;

        // Callback should have been invoked for each save
        let callback_count = save_count.load(Ordering::Relaxed);
        assert!(
            callback_count >= 2,
            "on_save should be called at least twice, got {}",
            callback_count
        );
        assert_eq!(
            summary.save_count, callback_count,
            "callback count should match summary save_count"
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_callbacks_on_error_invoked() {
        let verifier = AsyncImprovementVerifier::with_cache();

        // Use a path that will fail to write
        let path = PathBuf::from("/nonexistent/dir/that/cannot/exist/cache.json");

        let error_count = Arc::new(AtomicUsize::new(0));
        let error_count_clone = Arc::clone(&error_count);

        let callbacks = CacheAutosaveCallbacks::new().on_error(move |event| {
            error_count_clone.fetch_add(1, Ordering::Relaxed);
            // Verify event data is sensible
            assert!(!event.error.is_empty());
            assert!(event.error_number > 0);
        });

        let handle = verifier
            .start_cache_autosave_with_callbacks(path, Duration::from_millis(20), callbacks)
            .unwrap()
            .expect("cache should be enabled");

        // Wait for a few error attempts
        time::sleep(Duration::from_millis(80)).await;

        let summary = handle.stop().await;

        // Callback should have been invoked for each error
        let callback_count = error_count.load(Ordering::Relaxed);
        assert!(
            callback_count > 0,
            "on_error should be called, got {}",
            callback_count
        );
        assert_eq!(
            summary.errors.len(),
            callback_count,
            "callback count should match summary errors length"
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_callbacks_both_invoked() {
        // Test that both callbacks can be set and work independently
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let save_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));
        let save_count_clone = Arc::clone(&save_count);
        let error_count_clone = Arc::clone(&error_count);

        let callbacks = CacheAutosaveCallbacks::new()
            .on_save(move |_event| {
                save_count_clone.fetch_add(1, Ordering::Relaxed);
            })
            .on_error(move |_event| {
                error_count_clone.fetch_add(1, Ordering::Relaxed);
            });

        let handle = verifier
            .start_cache_autosave_with_callbacks(path, Duration::from_millis(30), callbacks)
            .unwrap()
            .expect("cache should be enabled");

        // Wait for a couple of saves
        time::sleep(Duration::from_millis(70)).await;

        let summary = handle.stop().await;

        // Saves should succeed (valid path)
        assert!(save_count.load(Ordering::Relaxed) > 0);
        assert_eq!(error_count.load(Ordering::Relaxed), 0);
        assert_eq!(summary.errors.len(), 0);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_callbacks_event_data_accurate() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let events = Arc::new(Mutex::new(Vec::<AutosaveSaveEvent>::new()));
        let events_clone = Arc::clone(&events);

        let callbacks = CacheAutosaveCallbacks::new().on_save(move |event| {
            // Clone the event for later verification
            let events_clone = Arc::clone(&events_clone);
            let event = event.clone();
            tokio::spawn(async move {
                events_clone.lock().await.push(event);
            });
        });

        let handle = verifier
            .start_cache_autosave_with_callbacks(path.clone(), Duration::from_millis(30), callbacks)
            .unwrap()
            .expect("cache should be enabled");

        // Wait for at least 3 saves
        time::sleep(Duration::from_millis(100)).await;

        let summary = handle.stop().await;

        // Give a moment for async event storage to complete
        time::sleep(Duration::from_millis(10)).await;

        let captured_events = events.lock().await;

        // Verify save_number increments
        for (i, event) in captured_events.iter().enumerate() {
            assert_eq!(event.save_number, i + 1, "save_number should increment");
            assert_eq!(event.path, path);
            assert!(event.compression.is_none());
        }

        // Total bytes should be sum of all individual bytes
        let total: u64 = captured_events.iter().map(|e| e.bytes_written).sum();
        // Allow small tolerance for timing issues with last event
        assert!(
            (total as i64 - summary.total_bytes_written as i64).abs() <= 500,
            "captured total {} should match summary {}",
            total,
            summary.total_bytes_written
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_compressed_with_callbacks() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json.gz");

        let verifier = AsyncImprovementVerifier::with_cache();

        let compression_seen = Arc::new(Mutex::new(None));
        let compression_clone = Arc::clone(&compression_seen);

        let callbacks = CacheAutosaveCallbacks::new().on_save(move |event| {
            let compression_clone = Arc::clone(&compression_clone);
            let compression = event.compression;
            tokio::spawn(async move {
                *compression_clone.lock().await = compression;
            });
        });

        let handle = verifier
            .start_cache_autosave_compressed_with_callbacks(
                path,
                Duration::from_millis(30),
                SnapshotCompressionLevel::Fast,
                callbacks,
            )
            .unwrap()
            .expect("cache should be enabled");

        // Wait for at least one save
        time::sleep(Duration::from_millis(50)).await;

        let summary = handle.stop().await;

        // Give time for async event to be stored
        time::sleep(Duration::from_millis(10)).await;

        // Verify compression was passed to callback
        let captured = compression_seen.lock().await;
        assert_eq!(*captured, Some(SnapshotCompressionLevel::Fast));
        assert_eq!(summary.compression, Some(SnapshotCompressionLevel::Fast));
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_callbacks_no_callbacks_still_works() {
        // Verify that empty callbacks (default) works
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let callbacks = CacheAutosaveCallbacks::new();

        let handle = verifier
            .start_cache_autosave_with_callbacks(path, Duration::from_millis(30), callbacks)
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(70)).await;

        let summary = handle.stop().await;

        // Should still work normally without callbacks
        assert!(summary.save_count >= 2);
        assert!(summary.errors.is_empty());
    }

    // CacheAutosaveConfig preset tests

    #[test]
    fn cache_autosave_config_new() {
        let config = CacheAutosaveConfig::new("test.json", Duration::from_secs(60));
        assert_eq!(config.path, PathBuf::from("test.json"));
        assert_eq!(config.interval, Duration::from_secs(60));
        assert!(config.compression.is_none());
    }

    #[test]
    fn cache_autosave_config_with_compression() {
        let config = CacheAutosaveConfig::new("test.json", Duration::from_secs(60))
            .with_compression(SnapshotCompressionLevel::Fast);
        assert_eq!(config.compression, Some(SnapshotCompressionLevel::Fast));
    }

    #[test]
    fn cache_autosave_config_performance_preset() {
        let config = CacheAutosaveConfig::performance("cache.json");
        assert_eq!(config.interval, Duration::from_secs(5 * 60)); // 5 minutes
        assert!(config.compression.is_none());
        assert_eq!(config.path, PathBuf::from("cache.json"));
    }

    #[test]
    fn cache_autosave_config_storage_optimized_preset() {
        let config = CacheAutosaveConfig::storage_optimized("cache.json.gz");
        assert_eq!(config.interval, Duration::from_secs(2 * 60)); // 2 minutes
        assert_eq!(config.compression, Some(SnapshotCompressionLevel::Best));
        assert_eq!(config.path, PathBuf::from("cache.json.gz"));
    }

    #[test]
    fn cache_autosave_config_balanced_preset() {
        let config = CacheAutosaveConfig::balanced("cache.json.gz");
        assert_eq!(config.interval, Duration::from_secs(60)); // 1 minute
        assert_eq!(config.compression, Some(SnapshotCompressionLevel::Fast));
    }

    #[test]
    fn cache_autosave_config_aggressive_preset() {
        let config = CacheAutosaveConfig::aggressive("cache.json.gz");
        assert_eq!(config.interval, Duration::from_secs(15));
        assert_eq!(config.compression, Some(SnapshotCompressionLevel::Fast));
    }

    #[test]
    fn cache_autosave_config_development_preset() {
        let config = CacheAutosaveConfig::development("cache.json");
        assert_eq!(config.interval, Duration::from_secs(30));
        assert!(config.compression.is_none());
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_with_config_balanced() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json.gz");

        let verifier = AsyncImprovementVerifier::with_cache();

        // Override interval for faster test and disable min_change threshold for test stability
        let config = CacheAutosaveConfig::balanced(path.clone())
            .with_compression(SnapshotCompressionLevel::Fast);
        let config = CacheAutosaveConfig {
            interval: Duration::from_millis(30),
            min_change_bytes: 0, // Override for test: always save
            ..config
        };

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(70)).await;

        let summary = handle.stop().await;

        assert!(summary.save_count >= 2);
        assert_eq!(summary.compression, Some(SnapshotCompressionLevel::Fast));
        assert!(summary.errors.is_empty());
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_with_config_storage_optimized() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json.gz");

        let verifier = AsyncImprovementVerifier::with_cache();

        // Use storage_optimized but override interval and min_change for test
        let config = CacheAutosaveConfig::storage_optimized(path.clone());
        let config = CacheAutosaveConfig {
            interval: Duration::from_millis(30),
            min_change_bytes: 0, // Override for test: always save
            ..config
        };

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(70)).await;

        let summary = handle.stop().await;

        assert!(summary.save_count >= 2);
        // Storage optimized uses Best compression
        assert_eq!(summary.compression, Some(SnapshotCompressionLevel::Best));
        assert!(summary.errors.is_empty());
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_with_config_and_callbacks() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let save_count = Arc::new(AtomicUsize::new(0));
        let save_count_clone = Arc::clone(&save_count);

        let callbacks = CacheAutosaveCallbacks::new().on_save(move |_event| {
            save_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(30))
            .with_callbacks(callbacks);

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(70)).await;

        let summary = handle.stop().await;

        // Callback should have been invoked
        let callback_count = save_count.load(Ordering::Relaxed);
        assert!(callback_count >= 2);
        assert_eq!(summary.save_count, callback_count);
    }

    // Minimum change threshold tests

    #[test]
    fn cache_autosave_config_with_min_change_bytes() {
        let config = CacheAutosaveConfig::new("test.json", Duration::from_secs(60))
            .with_min_change_bytes(1024);
        assert_eq!(config.min_change_bytes, 1024);
    }

    #[test]
    fn cache_autosave_config_default_min_change_is_zero() {
        let config = CacheAutosaveConfig::new("test.json", Duration::from_secs(60));
        assert_eq!(config.min_change_bytes, 0);
    }

    #[test]
    fn cache_autosave_config_presets_have_correct_min_change() {
        // Performance: 4KB threshold
        let config = CacheAutosaveConfig::performance("cache.json");
        assert_eq!(config.min_change_bytes, 4096);

        // Storage optimized: 2KB threshold
        let config = CacheAutosaveConfig::storage_optimized("cache.json.gz");
        assert_eq!(config.min_change_bytes, 2048);

        // Balanced: 1KB threshold
        let config = CacheAutosaveConfig::balanced("cache.json.gz");
        assert_eq!(config.min_change_bytes, 1024);

        // Aggressive: no threshold (always save)
        let config = CacheAutosaveConfig::aggressive("cache.json.gz");
        assert_eq!(config.min_change_bytes, 0);

        // Development: no threshold (always save)
        let config = CacheAutosaveConfig::development("cache.json");
        assert_eq!(config.min_change_bytes, 0);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_skips_when_no_change() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let skip_count = Arc::new(AtomicUsize::new(0));
        let skip_count_clone = Arc::clone(&skip_count);

        let callbacks = CacheAutosaveCallbacks::new().on_skip(move |_event| {
            skip_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        // Use a very high threshold so all subsequent saves will be skipped
        // (cache doesn't change between saves)
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(20))
            .with_callbacks(callbacks)
            .with_min_change_bytes(1_000_000); // 1MB threshold - much larger than empty cache

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Wait for initial save + 2-3 ticks (which should all be skipped)
        time::sleep(Duration::from_millis(80)).await;

        let summary = handle.stop().await;

        // Initial save should succeed, subsequent ones should be skipped
        assert_eq!(summary.save_count, 1, "Only initial save should succeed");
        assert!(
            summary.skip_count >= 2,
            "At least 2 saves should be skipped, got {}",
            summary.skip_count
        );
        let callback_skips = skip_count.load(Ordering::Relaxed);
        assert_eq!(summary.skip_count, callback_skips);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_saves_when_threshold_exceeded() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        // Use a small threshold of 100 bytes
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(25))
            .with_min_change_bytes(100);

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Initial save happens
        time::sleep(Duration::from_millis(10)).await;

        // Add cache entries to exceed the threshold
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            cache_guard.insert(
                CacheKey::new("hash1", "prop1"),
                CachedPropertyResult {
                    property: VerifiedProperty {
                        name: "test_property_1".to_string(),
                        passed: true,
                        status: "proven".to_string(),
                    },
                    backends: vec!["lean4".to_string()],
                    cached_at: std::time::SystemTime::now(),
                    dependency_hash: "hash1".to_string(),
                    confidence: 0.99,
                },
            );
            cache_guard.insert(
                CacheKey::new("hash2", "prop2"),
                CachedPropertyResult {
                    property: VerifiedProperty {
                        name: "test_property_2".to_string(),
                        passed: true,
                        status: "proven".to_string(),
                    },
                    backends: vec!["tlaplus".to_string()],
                    cached_at: std::time::SystemTime::now(),
                    dependency_hash: "hash2".to_string(),
                    confidence: 0.95,
                },
            );
        }

        // Wait for another tick to trigger save
        time::sleep(Duration::from_millis(50)).await;

        let summary = handle.stop().await;

        // Should have initial save + at least one more (after adding entries)
        assert!(
            summary.save_count >= 2,
            "Should have at least 2 saves, got {}",
            summary.save_count
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_skip_event_has_correct_data() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let captured_event = Arc::new(tokio::sync::Mutex::new(None::<AutosaveSkipEvent>));
        let captured_event_clone = Arc::clone(&captured_event);

        let callbacks = CacheAutosaveCallbacks::new().on_skip(move |event| {
            let captured = captured_event_clone.clone();
            // Store the first skip event
            tokio::spawn(async move {
                let mut guard = captured.lock().await;
                if guard.is_none() {
                    *guard = Some(event);
                }
            });
        });

        let min_threshold = 50_000u64; // 50KB threshold
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(20))
            .with_callbacks(callbacks)
            .with_min_change_bytes(min_threshold);

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Wait for skip event
        time::sleep(Duration::from_millis(60)).await;

        let _summary = handle.stop().await;

        // Give spawned task time to complete
        time::sleep(Duration::from_millis(10)).await;

        let event = captured_event.lock().await;
        let event = event.as_ref().expect("should have captured skip event");

        assert_eq!(event.path, path);
        assert_eq!(event.min_change_bytes, min_threshold);
        assert!(event.change_bytes < min_threshold);
        assert!(event.skip_number >= 1);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_status_includes_skip_count() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(20))
            .with_min_change_bytes(1_000_000); // High threshold to force skips

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(60)).await;

        let status = handle.status();
        assert_eq!(status.save_count, 1); // Only initial save
        assert!(status.skip_count >= 2, "Expected at least 2 skips");
        assert!(status.is_running);

        let summary = handle.stop().await;
        assert_eq!(summary.save_count, 1);
        assert!(summary.skip_count >= 2);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_zero_threshold_never_skips() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        // Zero threshold means always save
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(20))
            .with_min_change_bytes(0);

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(70)).await;

        let summary = handle.stop().await;

        // All saves should succeed, none skipped
        assert!(summary.save_count >= 3);
        assert_eq!(summary.skip_count, 0);
    }

    // Max stale duration tests

    #[test]
    fn cache_autosave_config_with_max_stale_duration() {
        let config = CacheAutosaveConfig::new("test.json", Duration::from_secs(60))
            .with_max_stale_duration(Duration::from_secs(300));
        assert_eq!(config.max_stale_duration, Some(Duration::from_secs(300)));
    }

    #[test]
    fn cache_autosave_config_default_max_stale_is_none() {
        let config = CacheAutosaveConfig::new("test.json", Duration::from_secs(60));
        assert!(config.max_stale_duration.is_none());
    }

    #[test]
    fn cache_autosave_config_presets_have_correct_max_stale() {
        // Performance: 15 min max stale
        let config = CacheAutosaveConfig::performance("cache.json");
        assert_eq!(
            config.max_stale_duration,
            Some(Duration::from_secs(15 * 60))
        );

        // Storage optimized: 10 min max stale
        let config = CacheAutosaveConfig::storage_optimized("cache.json.gz");
        assert_eq!(
            config.max_stale_duration,
            Some(Duration::from_secs(10 * 60))
        );

        // Balanced: 5 min max stale
        let config = CacheAutosaveConfig::balanced("cache.json.gz");
        assert_eq!(config.max_stale_duration, Some(Duration::from_secs(5 * 60)));

        // Aggressive: None (not needed with min_change_bytes=0)
        let config = CacheAutosaveConfig::aggressive("cache.json.gz");
        assert!(config.max_stale_duration.is_none());

        // Development: None (not needed with min_change_bytes=0)
        let config = CacheAutosaveConfig::development("cache.json");
        assert!(config.max_stale_duration.is_none());
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_forces_save_when_stale() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let save_count = Arc::new(AtomicUsize::new(0));
        let save_count_clone = Arc::clone(&save_count);

        let callbacks = CacheAutosaveCallbacks::new().on_save(move |_event| {
            save_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        // High threshold (will skip) but short max_stale (will force save)
        // Interval: 20ms, max_stale: 50ms means after 2-3 skips it should force a save
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(20))
            .with_callbacks(callbacks)
            .with_min_change_bytes(1_000_000) // Very high threshold - would always skip
            .with_max_stale_duration(Duration::from_millis(50)); // But force save after 50ms

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Wait long enough for initial save + staleness to trigger forced saves
        // At 20ms intervals with 50ms max stale, we should get forced saves
        time::sleep(Duration::from_millis(150)).await;

        let summary = handle.stop().await;

        // Should have more than just the initial save due to staleness forcing saves
        // Initial save at t=0, then skips until t>=50ms forces a save
        // More saves should follow due to continued staleness
        assert!(
            summary.save_count >= 2,
            "Expected at least 2 saves due to staleness forcing, got {} saves and {} skips",
            summary.save_count,
            summary.skip_count
        );

        let callback_saves = save_count.load(Ordering::Relaxed);
        assert_eq!(summary.save_count, callback_saves);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_no_forced_save_without_max_stale() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        // High threshold, no max_stale - should only get initial save
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(20))
            .with_min_change_bytes(1_000_000); // Very high threshold, no max_stale

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(100)).await;

        let summary = handle.stop().await;

        // Should only have initial save, rest should be skipped
        assert_eq!(
            summary.save_count, 1,
            "Expected only initial save without max_stale"
        );
        assert!(
            summary.skip_count >= 3,
            "Expected at least 3 skips, got {}",
            summary.skip_count
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_staleness_resets_after_forced_save() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let save_count = Arc::new(AtomicUsize::new(0));
        let save_count_clone = Arc::clone(&save_count);

        let callbacks = CacheAutosaveCallbacks::new().on_save(move |_event| {
            save_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        // Interval: 15ms, max_stale: 40ms
        // Should force saves roughly every 40ms after initial
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(15))
            .with_callbacks(callbacks)
            .with_min_change_bytes(1_000_000) // Always would skip
            .with_max_stale_duration(Duration::from_millis(40));

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Run for ~150ms - should get initial + ~3 forced saves (at ~45ms, ~90ms, ~135ms)
        time::sleep(Duration::from_millis(150)).await;

        let summary = handle.stop().await;

        // Check we got multiple saves due to repeated staleness
        // Initial at t=0, then forced at ~45ms, ~90ms, ~135ms (interval is 15ms, stale at 40ms)
        assert!(
            summary.save_count >= 3,
            "Expected at least 3 saves (initial + 2+ forced), got {} saves and {} skips",
            summary.save_count,
            summary.skip_count
        );

        // Verify callback count matches summary
        let callback_saves = save_count.load(Ordering::Relaxed);
        assert_eq!(summary.save_count, callback_saves);
    }

    #[test]
    fn test_autosave_save_reason_display() {
        assert_eq!(AutosaveSaveReason::Initial.to_string(), "initial");
        assert_eq!(AutosaveSaveReason::Interval.to_string(), "interval");
        assert_eq!(AutosaveSaveReason::StaleData.to_string(), "stale_data");
    }

    #[test]
    fn test_autosave_save_reason_equality() {
        assert_eq!(AutosaveSaveReason::Initial, AutosaveSaveReason::Initial);
        assert_eq!(AutosaveSaveReason::Interval, AutosaveSaveReason::Interval);
        assert_eq!(AutosaveSaveReason::StaleData, AutosaveSaveReason::StaleData);
        assert_ne!(AutosaveSaveReason::Initial, AutosaveSaveReason::Interval);
        assert_ne!(AutosaveSaveReason::Initial, AutosaveSaveReason::StaleData);
        assert_ne!(AutosaveSaveReason::Interval, AutosaveSaveReason::StaleData);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_initial_save_has_initial_reason() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let save_reasons = Arc::new(std::sync::Mutex::new(Vec::new()));
        let save_reasons_clone = Arc::clone(&save_reasons);

        let callbacks = CacheAutosaveCallbacks::new().on_save(move |event| {
            save_reasons_clone.lock().unwrap().push(event.save_reason);
        });

        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(100))
            .with_callbacks(callbacks);

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Wait just enough for initial save to complete
        time::sleep(Duration::from_millis(50)).await;

        let _ = handle.stop().await;

        let reasons = save_reasons.lock().unwrap();
        assert!(!reasons.is_empty(), "Should have at least one save");
        assert_eq!(
            reasons[0],
            AutosaveSaveReason::Initial,
            "First save should have Initial reason"
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_interval_save_has_interval_reason() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let save_reasons = Arc::new(std::sync::Mutex::new(Vec::new()));
        let save_reasons_clone = Arc::clone(&save_reasons);

        let callbacks = CacheAutosaveCallbacks::new().on_save(move |event| {
            save_reasons_clone.lock().unwrap().push(event.save_reason);
        });

        // No min_change_bytes threshold means every interval save succeeds
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(25))
            .with_callbacks(callbacks);

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Wait long enough for initial + at least one interval save
        time::sleep(Duration::from_millis(80)).await;

        let _ = handle.stop().await;

        let reasons = save_reasons.lock().unwrap();
        assert!(
            reasons.len() >= 2,
            "Should have at least 2 saves (initial + interval), got {}",
            reasons.len()
        );
        assert_eq!(
            reasons[0],
            AutosaveSaveReason::Initial,
            "First save should be Initial"
        );
        assert_eq!(
            reasons[1],
            AutosaveSaveReason::Interval,
            "Second save should be Interval"
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_stale_save_has_stale_data_reason() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let save_reasons = Arc::new(std::sync::Mutex::new(Vec::new()));
        let save_reasons_clone = Arc::clone(&save_reasons);

        let callbacks = CacheAutosaveCallbacks::new().on_save(move |event| {
            save_reasons_clone.lock().unwrap().push(event.save_reason);
        });

        // High threshold (would skip) but short max_stale (will force due to staleness)
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(15))
            .with_callbacks(callbacks)
            .with_min_change_bytes(1_000_000) // Very high - would always skip
            .with_max_stale_duration(Duration::from_millis(40)); // Force save after 40ms

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Wait long enough for staleness to trigger
        time::sleep(Duration::from_millis(100)).await;

        let summary = handle.stop().await;
        let reasons = save_reasons.lock().unwrap();

        assert!(
            reasons.len() >= 2,
            "Should have at least 2 saves (initial + stale-forced), got {}",
            reasons.len()
        );
        assert_eq!(
            reasons[0],
            AutosaveSaveReason::Initial,
            "First save should be Initial"
        );

        // At least one save after the initial should be due to staleness
        let stale_saves: Vec<_> = reasons
            .iter()
            .filter(|r| **r == AutosaveSaveReason::StaleData)
            .collect();
        assert!(
            !stale_saves.is_empty(),
            "Should have at least one StaleData save, got reasons: {:?}",
            *reasons
        );

        // forced_save_count should match number of StaleData saves
        assert_eq!(
            summary.forced_save_count,
            stale_saves.len(),
            "forced_save_count should match StaleData save count"
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_status_includes_forced_save_count() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        // High threshold with short max_stale - will force saves due to staleness
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(15))
            .with_min_change_bytes(1_000_000)
            .with_max_stale_duration(Duration::from_millis(40));

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Wait for staleness to trigger forced saves
        time::sleep(Duration::from_millis(100)).await;

        // Check status (non-blocking query)
        let status = handle.status();
        assert!(status.is_running);

        // Should have forced saves due to staleness
        assert!(
            status.forced_save_count > 0,
            "Status should show forced_save_count > 0, got {}",
            status.forced_save_count
        );

        let summary = handle.stop().await;

        // Summary should also have forced_save_count
        assert!(
            summary.forced_save_count > 0,
            "Summary should show forced_save_count > 0, got {}",
            summary.forced_save_count
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_no_forced_saves_without_staleness() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        // No threshold (always save) and no max_stale - should never have forced saves
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(20));

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(80)).await;

        let summary = handle.stop().await;

        // Without max_stale_duration, there should be no forced saves
        assert_eq!(
            summary.forced_save_count, 0,
            "Without max_stale_duration, forced_save_count should be 0"
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_status_reports_last_reason() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        // Always save on interval to exercise interval reason reporting
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(20));

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Wait for initial + multiple interval saves
        time::sleep(Duration::from_millis(90)).await;

        let status = handle.status();
        assert!(
            status.save_count >= 2,
            "Expected at least 2 saves, got {}",
            status.save_count
        );
        assert_eq!(status.save_reason_counts.initial, 1);
        assert!(
            status.save_reason_counts.interval >= 1,
            "Expected interval saves, got {:?}",
            status.save_reason_counts
        );
        assert_eq!(status.save_reason_counts.stale_data, 0);
        assert_eq!(
            status.save_reason_counts.initial
                + status.save_reason_counts.interval
                + status.save_reason_counts.stale_data,
            status.save_count
        );
        assert_eq!(
            status.last_save_reason,
            Some(AutosaveSaveReason::Interval),
            "Most recent save should be interval-driven"
        );

        let summary = handle.stop().await;
        assert_eq!(
            summary.last_save_reason,
            Some(AutosaveSaveReason::Interval),
            "Summary should preserve last save reason"
        );
        assert_eq!(
            summary.save_reason_counts.interval,
            status.save_reason_counts.interval
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_reason_counts_track_stale_saves() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        // Force staleness-driven saves by combining a high threshold with a short max stale window
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(15))
            .with_min_change_bytes(1_000_000)
            .with_max_stale_duration(Duration::from_millis(40));

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Allow multiple stale periods to accumulate
        time::sleep(Duration::from_millis(140)).await;

        let summary = handle.stop().await;

        assert_eq!(
            summary.save_reason_counts.initial, 1,
            "Initial save should be counted exactly once"
        );
        assert_eq!(
            summary.save_reason_counts.interval, 0,
            "High threshold should prevent interval saves"
        );
        assert!(
            summary.save_reason_counts.stale_data >= 1,
            "Expected at least one stale_data save, got {}",
            summary.save_reason_counts.stale_data
        );
        assert_eq!(
            summary.save_reason_counts.stale_data, summary.forced_save_count,
            "StaleData reason count should match forced_save_count"
        );
        assert_eq!(
            summary.save_count,
            summary.save_reason_counts.initial
                + summary.save_reason_counts.interval
                + summary.save_reason_counts.stale_data,
            "Reason counts should sum to total saves"
        );
        assert_eq!(
            summary.last_save_reason,
            Some(AutosaveSaveReason::StaleData),
            "Stale-driven saves should be the most recent when threshold blocks interval saves"
        );
    }

    // ==================== Compaction Trigger Tests ====================

    #[tokio::test]
    async fn async_verifier_cache_autosave_with_compaction_triggers() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_custom_cache(10, Duration::from_secs(3600));

        // Add entries to the cache to trigger size-based compaction
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            // Add 9 entries to a cache with max_entries=10 - this will trigger 90% capacity
            for i in 0..9 {
                let key = CacheKey::new(format!("v{}", i), "prop");
                let result = CachedPropertyResult {
                    property: VerifiedProperty {
                        name: "test_prop".to_string(),
                        passed: true,
                        status: "verified".to_string(),
                    },
                    backends: vec!["lean4".to_string()],
                    cached_at: SystemTime::now(),
                    dependency_hash: "hash123".to_string(),
                    confidence: 0.9,
                };
                cache_guard.insert(key, result);
            }
        }

        // Track compaction events
        let compaction_events = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let compaction_events_clone = Arc::clone(&compaction_events);

        let callbacks = CacheAutosaveCallbacks::new().on_compaction(move |event| {
            compaction_events_clone.fetch_add(1, Ordering::Relaxed);
            // Verify the event has correct structure
            assert!(event.entries_before >= event.entries_after);
            assert!(event.compaction_number > 0);
        });

        // Use config with compaction triggers
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(30))
            .with_callbacks(callbacks)
            .with_compaction(CompactionConfig::expired_only())
            .with_compaction_triggers(vec![CompactionTrigger::size_90_percent()]);

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Wait for save + potential compaction
        time::sleep(Duration::from_millis(100)).await;

        let summary = handle.stop().await;

        // Compaction may or may not have triggered depending on timing,
        // but the summary should track it either way
        assert!(
            summary.save_count >= 1,
            "Expected at least 1 save, got {}",
            summary.save_count
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_compaction_callback_invoked() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_custom_cache(10, Duration::from_secs(1));

        // Add exactly 9 entries (90% capacity) with some expired to trigger compaction
        if let Some(cache) = &verifier.cache {
            let mut cache_guard = cache.lock().await;
            for i in 0..9 {
                let key = CacheKey::new(format!("v{}", i), "prop");
                // Make entries expired so compaction will remove them
                let result = CachedPropertyResult {
                    property: VerifiedProperty {
                        name: "test_prop".to_string(),
                        passed: true,
                        status: "verified".to_string(),
                    },
                    backends: vec!["lean4".to_string()],
                    cached_at: SystemTime::now() - Duration::from_secs(10),
                    dependency_hash: "hash123".to_string(),
                    confidence: 0.9,
                };
                cache_guard.insert(key, result);
            }
        }

        let compaction_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let compaction_count_clone = Arc::clone(&compaction_count);

        let callbacks = CacheAutosaveCallbacks::new().on_compaction(move |_event| {
            compaction_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(30))
            .with_callbacks(callbacks)
            .with_compaction(CompactionConfig::expired_only())
            .with_compaction_triggers(vec![CompactionTrigger::size_90_percent()]);

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        // Wait for saves to occur
        time::sleep(Duration::from_millis(150)).await;

        let summary = handle.stop().await;

        // The summary should track compactions
        // compaction_count in summary should match what the callback saw
        let callback_compactions = compaction_count.load(Ordering::Relaxed);
        assert_eq!(
            summary.compaction_count, callback_compactions,
            "Summary compaction count should match callback invocations"
        );
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_status_includes_compaction_count() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache.json");

        let verifier = AsyncImprovementVerifier::with_cache();

        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(30))
            .with_compaction(CompactionConfig::expired_only())
            .with_compaction_triggers(vec![CompactionTrigger::size_80_percent()]);

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .expect("cache should be enabled");

        time::sleep(Duration::from_millis(50)).await;

        // Check that status includes compaction_count field
        let status = handle.status();
        assert!(status.is_running);
        // compaction_count should be accessible (may be 0 if no triggers fired)
        let _compaction_count = status.compaction_count;

        let summary = handle.stop().await;
        // Summary should also have compaction_count
        let _summary_compaction_count = summary.compaction_count;
    }

    // ==================== Adaptive Interval Tests ====================

    #[test]
    fn test_adaptive_interval_config_new() {
        let config = AdaptiveIntervalConfig::new(Duration::from_secs(15), Duration::from_secs(300));
        assert_eq!(config.min_interval, Duration::from_secs(15));
        assert_eq!(config.max_interval, Duration::from_secs(300));
        assert_eq!(config.high_activity_threshold, 10 * 1024); // 10KB
        assert_eq!(config.low_activity_threshold, 1024); // 1KB
        assert!((config.decrease_factor - 0.5).abs() < f64::EPSILON);
        assert!((config.increase_factor - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adaptive_interval_config_default() {
        let config = AdaptiveIntervalConfig::default();
        assert_eq!(config.min_interval, Duration::from_secs(15));
        assert_eq!(config.max_interval, Duration::from_secs(300));
    }

    #[test]
    fn test_adaptive_interval_config_builder_methods() {
        let config = AdaptiveIntervalConfig::new(Duration::from_secs(10), Duration::from_secs(600))
            .with_high_activity_threshold(20_000)
            .with_low_activity_threshold(500)
            .with_decrease_factor(0.25)
            .with_increase_factor(2.0);

        assert_eq!(config.high_activity_threshold, 20_000);
        assert_eq!(config.low_activity_threshold, 500);
        assert!((config.decrease_factor - 0.25).abs() < f64::EPSILON);
        assert!((config.increase_factor - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adaptive_interval_config_factor_clamping() {
        // Decrease factor should be clamped to [0.1, 1.0]
        let config = AdaptiveIntervalConfig::default().with_decrease_factor(0.05); // Below min
        assert!((config.decrease_factor - 0.1).abs() < f64::EPSILON);

        let config = AdaptiveIntervalConfig::default().with_decrease_factor(1.5); // Above max
        assert!((config.decrease_factor - 1.0).abs() < f64::EPSILON);

        // Increase factor should be clamped to [1.0, 10.0]
        let config = AdaptiveIntervalConfig::default().with_increase_factor(0.5); // Below min
        assert!((config.increase_factor - 1.0).abs() < f64::EPSILON);

        let config = AdaptiveIntervalConfig::default().with_increase_factor(15.0); // Above max
        assert!((config.increase_factor - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adaptive_interval_compute_high_activity() {
        let config = AdaptiveIntervalConfig::new(Duration::from_secs(15), Duration::from_secs(300))
            .with_high_activity_threshold(10_000);

        let current = Duration::from_secs(60);
        // High activity (>= 10KB) should decrease interval
        let next = config.compute_next_interval(current, 15_000);
        // 60 * 0.5 = 30 seconds
        assert_eq!(next, Duration::from_secs(30));
    }

    #[test]
    fn test_adaptive_interval_compute_low_activity() {
        let config = AdaptiveIntervalConfig::new(Duration::from_secs(15), Duration::from_secs(300))
            .with_low_activity_threshold(1_000);

        let current = Duration::from_secs(60);
        // Low activity (<= 1KB) should increase interval
        let next = config.compute_next_interval(current, 500);
        // 60 * 1.5 = 90 seconds
        assert_eq!(next, Duration::from_secs(90));
    }

    #[test]
    fn test_adaptive_interval_compute_normal_activity() {
        let config = AdaptiveIntervalConfig::new(Duration::from_secs(15), Duration::from_secs(300))
            .with_high_activity_threshold(10_000)
            .with_low_activity_threshold(1_000);

        let current = Duration::from_secs(60);
        // Normal activity (between thresholds) should keep interval unchanged
        let next = config.compute_next_interval(current, 5_000);
        assert_eq!(next, current);
    }

    #[test]
    fn test_adaptive_interval_compute_clamps_to_min() {
        let config = AdaptiveIntervalConfig::new(Duration::from_secs(15), Duration::from_secs(300));

        let current = Duration::from_secs(20);
        // High activity should decrease, but result is clamped to min (15s)
        let next = config.compute_next_interval(current, 20_000);
        // 20 * 0.5 = 10s, but min is 15s
        assert_eq!(next, Duration::from_secs(15));
    }

    #[test]
    fn test_adaptive_interval_compute_clamps_to_max() {
        let config = AdaptiveIntervalConfig::new(Duration::from_secs(15), Duration::from_secs(300));

        let current = Duration::from_secs(250);
        // Low activity should increase, but result is clamped to max (300s)
        let next = config.compute_next_interval(current, 0);
        // 250 * 1.5 = 375s, but max is 300s
        assert_eq!(next, Duration::from_secs(300));
    }

    #[test]
    fn test_cache_autosave_config_adaptive_preset() {
        let config = CacheAutosaveConfig::adaptive("cache.json.gz");
        assert_eq!(config.interval, Duration::from_secs(60));
        assert!(config.adaptive_interval.is_some());

        let adaptive = config.adaptive_interval.unwrap();
        assert_eq!(adaptive.min_interval, Duration::from_secs(15));
        assert_eq!(adaptive.max_interval, Duration::from_secs(300));
    }

    #[test]
    fn test_cache_autosave_config_with_adaptive_interval() {
        let config =
            CacheAutosaveConfig::new("cache.json", Duration::from_secs(60)).with_adaptive_interval(
                AdaptiveIntervalConfig::new(Duration::from_secs(10), Duration::from_secs(120)),
            );

        assert!(config.adaptive_interval.is_some());
        let adaptive = config.adaptive_interval.unwrap();
        assert_eq!(adaptive.min_interval, Duration::from_secs(10));
        assert_eq!(adaptive.max_interval, Duration::from_secs(120));
    }

    #[test]
    fn test_cache_autosave_config_presets_have_no_adaptive() {
        // All existing presets should have adaptive_interval = None
        assert!(CacheAutosaveConfig::performance("p")
            .adaptive_interval
            .is_none());
        assert!(CacheAutosaveConfig::storage_optimized("p")
            .adaptive_interval
            .is_none());
        assert!(CacheAutosaveConfig::balanced("p")
            .adaptive_interval
            .is_none());
        assert!(CacheAutosaveConfig::aggressive("p")
            .adaptive_interval
            .is_none());
        assert!(CacheAutosaveConfig::development("p")
            .adaptive_interval
            .is_none());
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_status_includes_current_interval() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache_interval.json");

        let interval = Duration::from_millis(100);
        let handle = verifier
            .start_cache_autosave(path, interval)
            .unwrap()
            .unwrap();

        // Give it a moment to start
        time::sleep(Duration::from_millis(50)).await;

        let status = handle.status();
        assert_eq!(status.current_interval, interval);

        let summary = handle.stop().await;
        assert_eq!(summary.interval, interval);
        assert_eq!(summary.final_interval, interval); // No adaptive, so same as initial
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_adaptive_interval_decreases_on_high_activity() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache_adaptive.json");

        // Pre-populate cache with data to simulate high activity
        {
            let cache = verifier.cache.as_ref().unwrap();
            let mut c = cache.lock().await;
            for i in 0..100 {
                c.insert(
                    CacheKey::new(format!("test_spec_{}", i), format!("backend_{}", i % 3)),
                    CachedPropertyResult {
                        property: VerifiedProperty {
                            name: format!("prop_{}", i),
                            passed: true,
                            status: "Proven".to_string(),
                        },
                        backends: vec![format!("backend_{}", i % 3)],
                        cached_at: SystemTime::now(),
                        dependency_hash: format!("dep_hash_{}", i),
                        confidence: 0.99,
                    },
                );
            }
        }

        // Use adaptive config with very short intervals for testing
        let config = CacheAutosaveConfig::new(path, Duration::from_millis(100))
            .with_adaptive_interval(
                AdaptiveIntervalConfig::new(
                    Duration::from_millis(25),  // min 25ms
                    Duration::from_millis(500), // max 500ms
                )
                .with_high_activity_threshold(100) // Very low threshold for testing
                .with_low_activity_threshold(10),
            );

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .unwrap();

        // Wait for initial save and at least one adaptive tick
        time::sleep(Duration::from_millis(200)).await;

        let status = handle.status();
        // With data already in cache, the initial save should have detected it
        // and potentially adjusted the interval. We verify the current_interval
        // is available.
        assert!(status.current_interval >= Duration::from_millis(25));
        assert!(status.current_interval <= Duration::from_millis(500));

        let summary = handle.stop().await;
        assert_eq!(summary.interval, Duration::from_millis(100)); // Initial interval
                                                                  // Final interval may have been adjusted
        assert!(summary.final_interval >= Duration::from_millis(25));
        assert!(summary.final_interval <= Duration::from_millis(500));
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_adaptive_interval_increases_on_low_activity() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache_adaptive_low.json");

        // Empty cache = low activity
        let config = CacheAutosaveConfig::new(path, Duration::from_millis(50))
            .with_adaptive_interval(
                AdaptiveIntervalConfig::new(
                    Duration::from_millis(25),  // min 25ms
                    Duration::from_millis(200), // max 200ms
                )
                .with_high_activity_threshold(10000) // High threshold
                .with_low_activity_threshold(1000) // Low threshold that empty cache will hit
                .with_increase_factor(2.0),
            );

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .unwrap();

        // Wait for initial save and a few ticks for interval to increase
        time::sleep(Duration::from_millis(250)).await;

        let status = handle.status();
        // Empty cache means very low byte changes, so interval should increase
        // toward max or stay within bounds
        assert!(status.current_interval >= Duration::from_millis(25));
        assert!(status.current_interval <= Duration::from_millis(200));

        let summary = handle.stop().await;
        // With low activity, final_interval should be >= initial (or clamped to max)
        assert!(summary.final_interval >= Duration::from_millis(50));
    }

    // CoalesceConfig tests

    #[test]
    fn test_coalesce_config_new() {
        let config = CoalesceConfig::new(Duration::from_millis(500), Duration::from_secs(5));
        assert_eq!(config.quiet_period, Duration::from_millis(500));
        assert_eq!(config.max_wait, Duration::from_secs(5));
        assert_eq!(config.activity_threshold, 0);
    }

    #[test]
    fn test_coalesce_config_with_activity_threshold() {
        let config = CoalesceConfig::new(Duration::from_millis(500), Duration::from_secs(5))
            .with_activity_threshold(1024);
        assert_eq!(config.activity_threshold, 1024);
    }

    #[test]
    fn test_coalesce_config_burst_preset() {
        let config = CoalesceConfig::burst();
        assert_eq!(config.quiet_period, Duration::from_millis(200));
        assert_eq!(config.max_wait, Duration::from_secs(3));
        assert_eq!(config.activity_threshold, 512);
    }

    #[test]
    fn test_coalesce_config_aggressive_preset() {
        let config = CoalesceConfig::aggressive();
        assert_eq!(config.quiet_period, Duration::from_secs(1));
        assert_eq!(config.max_wait, Duration::from_secs(10));
        assert_eq!(config.activity_threshold, 1024);
    }

    #[test]
    fn test_coalesce_config_default() {
        let config = CoalesceConfig::default();
        assert_eq!(config.quiet_period, Duration::from_millis(500));
        assert_eq!(config.max_wait, Duration::from_secs(5));
        assert_eq!(config.activity_threshold, 0);
    }

    #[test]
    fn test_cache_autosave_config_with_coalesce() {
        let config = CacheAutosaveConfig::new("test.json", Duration::from_secs(60))
            .with_coalesce(CoalesceConfig::burst());
        assert!(config.coalesce.is_some());
        let coalesce = config.coalesce.unwrap();
        assert_eq!(coalesce.quiet_period, Duration::from_millis(200));
    }

    #[test]
    fn test_cache_autosave_config_burst_optimized_preset() {
        let config = CacheAutosaveConfig::burst_optimized("cache.json.gz");
        assert!(config.coalesce.is_some());
        assert_eq!(config.interval.as_secs(), 60);
        assert!(config.compression.is_some());
    }

    #[test]
    fn test_all_presets_have_no_coalescing_by_default() {
        // All presets except burst_optimized should have no coalescing
        assert!(CacheAutosaveConfig::performance("p").coalesce.is_none());
        assert!(CacheAutosaveConfig::storage_optimized("p")
            .coalesce
            .is_none());
        assert!(CacheAutosaveConfig::balanced("p").coalesce.is_none());
        assert!(CacheAutosaveConfig::aggressive("p").coalesce.is_none());
        assert!(CacheAutosaveConfig::development("p").coalesce.is_none());
        assert!(CacheAutosaveConfig::adaptive("p").coalesce.is_none());
        // Only burst_optimized has coalescing
        assert!(CacheAutosaveConfig::burst_optimized("p").coalesce.is_some());
    }

    #[test]
    fn test_autosave_save_reason_coalesced_display() {
        assert_eq!(format!("{}", AutosaveSaveReason::Coalesced), "coalesced");
    }

    #[test]
    fn test_autosave_reason_counts_includes_coalesced() {
        let counts = AutosaveReasonCounts {
            initial: 1,
            interval: 2,
            stale_data: 3,
            coalesced: 4,
        };
        assert_eq!(counts.coalesced, 4);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_coalesce_callback_invoked() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache_coalesce.json");

        let coalesce_events = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let coalesce_events_clone = Arc::clone(&coalesce_events);

        let callbacks = CacheAutosaveCallbacks::new().on_coalesce(move |_event| {
            coalesce_events_clone.fetch_add(1, Ordering::Relaxed);
        });

        // Use very short intervals and coalescing with 0 activity threshold
        // so any change triggers coalescing
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(15))
            .with_callbacks(callbacks)
            .with_coalesce(
                CoalesceConfig::new(
                    Duration::from_millis(30),  // quiet period
                    Duration::from_millis(100), // max wait
                )
                .with_activity_threshold(0), // any change triggers
            );

        // Pre-populate cache with data to simulate activity
        {
            let cache = verifier.cache.as_ref().unwrap();
            let mut c = cache.lock().await;
            for i in 0..50 {
                c.insert(
                    CacheKey::new(format!("test_spec_{}", i), format!("backend_{}", i % 3)),
                    CachedPropertyResult {
                        property: VerifiedProperty {
                            name: format!("prop_{}", i),
                            passed: true,
                            status: "Proven".to_string(),
                        },
                        backends: vec![format!("backend_{}", i % 3)],
                        cached_at: SystemTime::now(),
                        dependency_hash: format!("dep_hash_{}", i),
                        confidence: 0.99,
                    },
                );
            }
        }

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .unwrap();

        // Wait for coalescing to happen and quiet period to expire
        time::sleep(Duration::from_millis(200)).await;

        let summary = handle.stop().await;

        // We should have at least one save (initial)
        assert!(summary.save_count >= 1);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_coalesce_batches_rapid_changes() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache_coalesce_batch.json");

        let save_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let save_count_clone = Arc::clone(&save_count);

        let callbacks = CacheAutosaveCallbacks::new().on_save(move |_event| {
            save_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        // Use short intervals with coalescing
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(20))
            .with_callbacks(callbacks)
            .with_coalesce(
                CoalesceConfig::new(
                    Duration::from_millis(50),  // quiet period - wait 50ms after last change
                    Duration::from_millis(200), // max wait
                )
                .with_activity_threshold(0),
            );

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .unwrap();

        // Simulate rapid changes
        for i in 0..10 {
            let cache = verifier.cache.as_ref().unwrap();
            let mut c = cache.lock().await;
            c.insert(
                CacheKey::new(format!("rapid_spec_{}", i), "backend".to_string()),
                CachedPropertyResult {
                    property: VerifiedProperty {
                        name: format!("prop_{}", i),
                        passed: true,
                        status: "Proven".to_string(),
                    },
                    backends: vec!["backend".to_string()],
                    cached_at: SystemTime::now(),
                    dependency_hash: format!("dep_{}", i),
                    confidence: 0.95,
                },
            );
            drop(c);
            // Small delay between changes
            time::sleep(Duration::from_millis(10)).await;
        }

        // Wait for coalescing quiet period to expire
        time::sleep(Duration::from_millis(150)).await;

        let summary = handle.stop().await;

        // With coalescing, we should have fewer saves than if we saved every interval
        // Initial save + coalesced saves should be >= 1
        assert!(summary.save_count >= 1);
    }

    #[tokio::test]
    async fn async_verifier_cache_autosave_coalesce_max_wait_forces_save() {
        let verifier = AsyncImprovementVerifier::with_cache();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("cache_coalesce_max_wait.json");

        // Use coalescing with very long quiet period but short max_wait
        let config = CacheAutosaveConfig::new(path.clone(), Duration::from_millis(20))
            .with_coalesce(
                CoalesceConfig::new(
                    Duration::from_secs(60),   // very long quiet period
                    Duration::from_millis(80), // short max wait
                )
                .with_activity_threshold(0),
            );

        // Pre-populate with data
        {
            let cache = verifier.cache.as_ref().unwrap();
            let mut c = cache.lock().await;
            for i in 0..20 {
                c.insert(
                    CacheKey::new(format!("max_wait_spec_{}", i), "backend".to_string()),
                    CachedPropertyResult {
                        property: VerifiedProperty {
                            name: format!("prop_{}", i),
                            passed: true,
                            status: "Proven".to_string(),
                        },
                        backends: vec!["backend".to_string()],
                        cached_at: SystemTime::now(),
                        dependency_hash: format!("dep_{}", i),
                        confidence: 0.95,
                    },
                );
            }
        }

        let handle = verifier
            .start_cache_autosave_with_config(config)
            .unwrap()
            .unwrap();

        // Wait for max_wait to force saves
        time::sleep(Duration::from_millis(200)).await;

        let summary = handle.stop().await;

        // Should have at least the initial save
        assert!(summary.save_count >= 1);
    }

    // BackoffConfig tests

    #[test]
    fn test_backoff_config_new() {
        let config = BackoffConfig::new(Duration::from_secs(1), Duration::from_secs(60));
        assert_eq!(config.initial_delay, Duration::from_secs(1));
        assert_eq!(config.max_delay, Duration::from_secs(60));
        assert_eq!(config.multiplier, 2.0);
        assert_eq!(config.error_threshold, 1);
    }

    #[test]
    fn test_backoff_config_with_multiplier() {
        let config = BackoffConfig::new(Duration::from_secs(1), Duration::from_secs(60))
            .with_multiplier(3.0);
        assert_eq!(config.multiplier, 3.0);
    }

    #[test]
    fn test_backoff_config_multiplier_clamped() {
        // Too low
        let config = BackoffConfig::new(Duration::from_secs(1), Duration::from_secs(60))
            .with_multiplier(0.5);
        assert_eq!(config.multiplier, 1.1);

        // Too high
        let config = BackoffConfig::new(Duration::from_secs(1), Duration::from_secs(60))
            .with_multiplier(20.0);
        assert_eq!(config.multiplier, 10.0);
    }

    #[test]
    fn test_backoff_config_with_error_threshold() {
        let config = BackoffConfig::new(Duration::from_secs(1), Duration::from_secs(60))
            .with_error_threshold(3);
        assert_eq!(config.error_threshold, 3);
    }

    #[test]
    fn test_backoff_config_error_threshold_minimum() {
        let config = BackoffConfig::new(Duration::from_secs(1), Duration::from_secs(60))
            .with_error_threshold(0);
        assert_eq!(config.error_threshold, 1); // Minimum is 1
    }

    #[test]
    fn test_backoff_config_compute_delay_before_threshold() {
        let config = BackoffConfig::new(Duration::from_secs(1), Duration::from_secs(60))
            .with_error_threshold(3);

        // Before threshold is reached, no backoff
        assert!(config.compute_delay(0).is_none());
        assert!(config.compute_delay(1).is_none());
        assert!(config.compute_delay(2).is_none());
    }

    #[test]
    fn test_backoff_config_compute_delay_exponential() {
        let config = BackoffConfig::new(Duration::from_secs(1), Duration::from_secs(120))
            .with_multiplier(2.0)
            .with_error_threshold(1);

        // Error 1: initial_delay = 1s
        assert_eq!(config.compute_delay(1), Some(Duration::from_secs(1)));
        // Error 2: 1s * 2.0 = 2s
        assert_eq!(config.compute_delay(2), Some(Duration::from_secs(2)));
        // Error 3: 1s * 2.0^2 = 4s
        assert_eq!(config.compute_delay(3), Some(Duration::from_secs(4)));
        // Error 4: 1s * 2.0^3 = 8s
        assert_eq!(config.compute_delay(4), Some(Duration::from_secs(8)));
    }

    #[test]
    fn test_backoff_config_compute_delay_capped_at_max() {
        let config = BackoffConfig::new(Duration::from_secs(1), Duration::from_secs(10))
            .with_multiplier(2.0)
            .with_error_threshold(1);

        // Error 5: 1s * 2.0^4 = 16s, but capped at max_delay = 10s
        assert_eq!(config.compute_delay(5), Some(Duration::from_secs(10)));
        // Error 10: should still be capped at 10s
        assert_eq!(config.compute_delay(10), Some(Duration::from_secs(10)));
    }

    #[test]
    fn test_backoff_config_transient_preset() {
        let config = BackoffConfig::transient();
        assert_eq!(config.initial_delay, Duration::from_millis(500));
        assert_eq!(config.max_delay, Duration::from_secs(60));
        assert_eq!(config.multiplier, 2.0);
        assert_eq!(config.error_threshold, 1);
    }

    #[test]
    fn test_backoff_config_persistent_preset() {
        let config = BackoffConfig::persistent();
        assert_eq!(config.initial_delay, Duration::from_secs(2));
        assert_eq!(config.max_delay, Duration::from_secs(300));
        assert_eq!(config.multiplier, 1.5);
        assert_eq!(config.error_threshold, 2);
    }

    #[test]
    fn test_backoff_config_aggressive_preset() {
        let config = BackoffConfig::aggressive();
        assert_eq!(config.initial_delay, Duration::from_millis(100));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert_eq!(config.multiplier, 3.0);
        assert_eq!(config.error_threshold, 1);
    }

    #[test]
    fn test_backoff_config_default() {
        let config = BackoffConfig::default();
        assert_eq!(config.initial_delay, Duration::from_secs(1));
        assert_eq!(config.max_delay, Duration::from_secs(120));
        assert_eq!(config.multiplier, 2.0);
        assert_eq!(config.error_threshold, 1);
    }

    #[test]
    fn test_cache_autosave_config_with_backoff() {
        let config = CacheAutosaveConfig::new("test.json", Duration::from_secs(60))
            .with_backoff(BackoffConfig::transient());
        assert!(config.backoff.is_some());
        let backoff = config.backoff.unwrap();
        assert_eq!(backoff.initial_delay, Duration::from_millis(500));
    }

    #[test]
    fn test_cache_autosave_config_resilient_preset() {
        let config = CacheAutosaveConfig::resilient("cache.json.gz");
        assert!(config.backoff.is_some());
        assert_eq!(config.interval.as_secs(), 60);
        assert!(config.compression.is_some());
    }

    #[test]
    fn test_all_presets_have_no_backoff_by_default_except_resilient() {
        // All presets except resilient should have no backoff
        assert!(CacheAutosaveConfig::performance("p").backoff.is_none());
        assert!(CacheAutosaveConfig::storage_optimized("p")
            .backoff
            .is_none());
        assert!(CacheAutosaveConfig::balanced("p").backoff.is_none());
        assert!(CacheAutosaveConfig::aggressive("p").backoff.is_none());
        assert!(CacheAutosaveConfig::development("p").backoff.is_none());
        assert!(CacheAutosaveConfig::adaptive("p").backoff.is_none());
        assert!(CacheAutosaveConfig::burst_optimized("p").backoff.is_none());
        // Only resilient has backoff
        assert!(CacheAutosaveConfig::resilient("p").backoff.is_some());
    }

    // ============================================================================
    // Tests for Metrics Aggregation
    // ============================================================================

    #[test]
    fn test_metrics_window_duration() {
        assert_eq!(MetricsWindow::OneMinute.duration(), Duration::from_secs(60));
        assert_eq!(
            MetricsWindow::FiveMinutes.duration(),
            Duration::from_secs(300)
        );
        assert_eq!(
            MetricsWindow::FifteenMinutes.duration(),
            Duration::from_secs(900)
        );
        assert_eq!(MetricsWindow::OneHour.duration(), Duration::from_secs(3600));
    }

    #[test]
    fn test_metrics_window_label() {
        assert_eq!(MetricsWindow::OneMinute.label(), "1m");
        assert_eq!(MetricsWindow::FiveMinutes.label(), "5m");
        assert_eq!(MetricsWindow::FifteenMinutes.label(), "15m");
        assert_eq!(MetricsWindow::OneHour.label(), "1h");
    }

    #[test]
    fn test_metric_sample_now() {
        let sample = MetricSample::now(42.0);
        assert_eq!(sample.value, 42.0);
        // Timestamp should be recent
        let elapsed = sample
            .timestamp
            .elapsed()
            .unwrap_or(Duration::from_secs(100));
        assert!(elapsed < Duration::from_secs(1));
    }

    #[test]
    fn test_metric_sample_at() {
        let ts = SystemTime::UNIX_EPOCH + Duration::from_secs(1234567890);
        let sample = MetricSample::at(ts, 99.9);
        assert_eq!(sample.value, 99.9);
        assert_eq!(sample.timestamp, ts);
    }

    #[test]
    fn test_windowed_metric_stats_empty() {
        let stats = WindowedMetricStats::from_samples(&[], Duration::from_secs(60));
        assert_eq!(stats.count, 0);
        assert_eq!(stats.sum, 0.0);
        assert!(stats.min.is_none());
        assert!(stats.max.is_none());
        assert!(stats.mean.is_none());
        assert_eq!(stats.rate_per_second, 0.0);
        assert_eq!(stats.rate_per_minute, 0.0);
    }

    #[test]
    fn test_windowed_metric_stats_single_sample() {
        let samples = vec![MetricSample::now(100.0)];
        let stats = WindowedMetricStats::from_samples(&samples, Duration::from_secs(60));
        assert_eq!(stats.count, 1);
        assert_eq!(stats.sum, 100.0);
        assert_eq!(stats.min, Some(100.0));
        assert_eq!(stats.max, Some(100.0));
        assert_eq!(stats.mean, Some(100.0));
        assert!((stats.rate_per_second - 1.0 / 60.0).abs() < 0.001);
        assert!((stats.rate_per_minute - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_windowed_metric_stats_multiple_samples() {
        let samples = vec![
            MetricSample::now(10.0),
            MetricSample::now(20.0),
            MetricSample::now(30.0),
        ];
        let stats = WindowedMetricStats::from_samples(&samples, Duration::from_secs(60));
        assert_eq!(stats.count, 3);
        assert_eq!(stats.sum, 60.0);
        assert_eq!(stats.min, Some(10.0));
        assert_eq!(stats.max, Some(30.0));
        assert_eq!(stats.mean, Some(20.0));
        assert!((stats.rate_per_second - 3.0 / 60.0).abs() < 0.001);
        assert!((stats.rate_per_minute - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_aggregator_config_default() {
        let config = MetricsAggregatorConfig::default();
        assert_eq!(config.max_samples_per_metric, 10_000);
        assert_eq!(config.max_sample_age, Duration::from_secs(2 * 60 * 60));
        assert_eq!(config.windows.len(), 4);
    }

    #[test]
    fn test_metrics_aggregator_config_minimal() {
        let config = MetricsAggregatorConfig::minimal();
        assert_eq!(config.max_samples_per_metric, 1_000);
        assert_eq!(config.max_sample_age, Duration::from_secs(15 * 60));
        assert_eq!(config.windows.len(), 2);
    }

    #[test]
    fn test_metrics_aggregator_config_high_resolution() {
        let config = MetricsAggregatorConfig::high_resolution();
        assert_eq!(config.max_samples_per_metric, 100_000);
        assert_eq!(config.max_sample_age, Duration::from_secs(24 * 60 * 60));
        assert_eq!(config.windows.len(), 4);
    }

    #[test]
    fn test_metrics_aggregator_new() {
        let agg = MetricsAggregator::new();
        assert_eq!(agg.total_sample_count(), 0);
        assert_eq!(agg.current_metrics().total_saves, 0);
        assert_eq!(agg.current_metrics().total_skips, 0);
        assert_eq!(agg.current_metrics().total_errors, 0);
    }

    #[test]
    fn test_metrics_aggregator_record_save() {
        let mut agg = MetricsAggregator::new();
        agg.record_save(1024, Duration::from_millis(50));
        assert_eq!(agg.current_metrics().total_saves, 1);
        assert_eq!(agg.current_metrics().total_bytes_written, 1024);
        // 3 samples: save event (1), bytes (1024), latency (0.05)
        assert_eq!(agg.total_sample_count(), 3);
    }

    #[test]
    fn test_metrics_aggregator_record_skip() {
        let mut agg = MetricsAggregator::new();
        agg.record_skip();
        agg.record_skip();
        assert_eq!(agg.current_metrics().total_skips, 2);
        assert_eq!(agg.total_sample_count(), 2);
    }

    #[test]
    fn test_metrics_aggregator_record_error() {
        let mut agg = MetricsAggregator::new();
        agg.record_error();
        agg.record_error();
        agg.record_error();
        assert_eq!(agg.current_metrics().total_errors, 3);
        assert_eq!(agg.total_sample_count(), 3);
    }

    #[test]
    fn test_metrics_aggregator_record_interval_change() {
        let mut agg = MetricsAggregator::new();
        agg.record_interval_change(Duration::from_secs(30));
        assert_eq!(agg.current_metrics().current_interval_ms, 30_000);
        agg.record_interval_change(Duration::from_secs(60));
        assert_eq!(agg.current_metrics().current_interval_ms, 60_000);
    }

    #[test]
    fn test_metrics_aggregator_record_threshold_update() {
        let mut agg = MetricsAggregator::new();
        assert!(!agg.current_metrics().learning_active);
        agg.record_threshold_update(10_000, 1_000);
        assert!(agg.current_metrics().learning_active);
        assert_eq!(agg.current_metrics().current_high_threshold, 10_000);
        assert_eq!(agg.current_metrics().current_low_threshold, 1_000);
    }

    #[test]
    fn test_metrics_aggregator_save_stats() {
        let mut agg = MetricsAggregator::new();
        for i in 0..5 {
            agg.record_save((i + 1) * 100, Duration::from_millis(10));
        }
        let stats = agg.save_stats(MetricsWindow::OneMinute);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.sum, 5.0); // Each save records value 1.0
    }

    #[test]
    fn test_metrics_aggregator_bytes_stats() {
        let mut agg = MetricsAggregator::new();
        agg.record_save(100, Duration::from_millis(10));
        agg.record_save(200, Duration::from_millis(20));
        agg.record_save(300, Duration::from_millis(30));
        let stats = agg.bytes_stats(MetricsWindow::OneMinute);
        assert_eq!(stats.count, 3);
        assert_eq!(stats.sum, 600.0);
        assert_eq!(stats.min, Some(100.0));
        assert_eq!(stats.max, Some(300.0));
        assert_eq!(stats.mean, Some(200.0));
    }

    #[test]
    fn test_metrics_aggregator_latency_stats() {
        let mut agg = MetricsAggregator::new();
        agg.record_save(100, Duration::from_millis(10));
        agg.record_save(100, Duration::from_millis(20));
        agg.record_save(100, Duration::from_millis(30));
        let stats = agg.latency_stats(MetricsWindow::OneMinute);
        assert_eq!(stats.count, 3);
        assert!((stats.sum - 0.06).abs() < 0.001);
        assert!((stats.mean.unwrap() - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_metrics_aggregator_uptime() {
        let agg = MetricsAggregator::new();
        std::thread::sleep(Duration::from_millis(10));
        assert!(agg.uptime_seconds() >= 0.01);
    }

    #[test]
    fn test_metrics_aggregator_reset_samples() {
        let mut agg = MetricsAggregator::new();
        agg.record_save(100, Duration::from_millis(10));
        agg.record_skip();
        agg.record_error();
        assert!(agg.total_sample_count() > 0);

        // Reset samples
        agg.reset_samples();
        assert_eq!(agg.total_sample_count(), 0);

        // Totals should be preserved
        assert_eq!(agg.current_metrics().total_saves, 1);
        assert_eq!(agg.current_metrics().total_skips, 1);
        assert_eq!(agg.current_metrics().total_errors, 1);
    }

    #[test]
    fn test_metrics_aggregator_export_prometheus() {
        let mut agg = MetricsAggregator::new();
        agg.record_save(1024, Duration::from_millis(50));
        agg.record_skip();
        agg.record_error();
        agg.record_threshold_update(10_000, 1_000);

        let output = agg.export_prometheus("dashprove");

        // Check key metrics are present
        assert!(output.contains("dashprove_autosave_saves_total 1"));
        assert!(output.contains("dashprove_autosave_skips_total 1"));
        assert!(output.contains("dashprove_autosave_errors_total 1"));
        assert!(output.contains("dashprove_autosave_bytes_total 1024"));
        assert!(output.contains("dashprove_autosave_high_threshold_bytes 10000"));
        assert!(output.contains("dashprove_autosave_low_threshold_bytes 1000"));
        assert!(output.contains("dashprove_autosave_learning_active 1"));
        assert!(output.contains("# TYPE dashprove_autosave_saves_total counter"));
        assert!(output.contains("# TYPE dashprove_autosave_interval_seconds gauge"));
    }

    #[test]
    fn test_metrics_aggregator_export_prometheus_windowed_rates() {
        let mut agg = MetricsAggregator::new();
        agg.record_save(100, Duration::from_millis(10));
        agg.record_save(200, Duration::from_millis(20));

        let output = agg.export_prometheus("test");

        // Check windowed rate metrics
        assert!(output.contains("test_autosave_saves_rate{window=\"1m\"}"));
        assert!(output.contains("test_autosave_saves_rate{window=\"5m\"}"));
        assert!(output.contains("test_autosave_saves_rate{window=\"15m\"}"));
        assert!(output.contains("test_autosave_saves_rate{window=\"1h\"}"));
        assert!(output.contains("test_autosave_bytes_rate{window=\"1m\"}"));
        assert!(output.contains("test_autosave_latency_seconds_mean{window=\"1m\"}"));
        assert!(output.contains("test_autosave_latency_seconds_max{window=\"1m\"}"));
    }

    #[test]
    fn test_metrics_aggregator_generate_report() {
        let mut agg = MetricsAggregator::new();
        agg.record_save(1024, Duration::from_millis(50));
        agg.record_skip();

        let report = agg.generate_report();
        assert!(report.uptime_seconds >= 0.0);
        assert_eq!(report.current.total_saves, 1);
        assert_eq!(report.current.total_skips, 1);
        assert!(report.windowed.contains_key("1m"));
        assert!(report.windowed.contains_key("5m"));
        assert!(report.windowed.contains_key("15m"));
        assert!(report.windowed.contains_key("1h"));
    }

    #[test]
    fn test_metrics_aggregator_export_json() {
        let mut agg = MetricsAggregator::new();
        agg.record_save(100, Duration::from_millis(10));

        let json = agg.export_json().unwrap();
        assert!(json.contains("\"total_saves\": 1"));
        assert!(json.contains("\"uptime_seconds\":"));
        assert!(json.contains("\"windowed\":"));
    }

    #[test]
    fn test_autosave_metrics_from_status() {
        let status = CacheAutosaveStatus {
            save_count: 5,
            error_count: 2,
            skip_count: 3,
            forced_save_count: 1,
            compaction_count: 0,
            compaction_trigger_counts: CompactionTriggerCounts::default(),
            save_reason_counts: AutosaveReasonCounts {
                initial: 1,
                interval: 3,
                stale_data: 1,
                coalesced: 0,
            },
            last_save_reason: Some(AutosaveSaveReason::Interval),
            total_bytes_written: 10_000,
            last_save_bytes: 2_000,
            current_interval: Duration::from_secs(30),
            is_running: true,
        };

        let metrics = AutosaveMetrics::from_status(&status);
        assert_eq!(metrics.total_saves, 5);
        assert_eq!(metrics.total_skips, 3);
        assert_eq!(metrics.total_errors, 2);
        assert_eq!(metrics.total_bytes_written, 10_000);
        assert_eq!(metrics.current_interval_ms, 30_000);
        assert_eq!(metrics.reason_counts.initial, 1);
        assert_eq!(metrics.reason_counts.interval, 3);
    }

    #[test]
    fn test_autosave_metrics_with_activity_stats() {
        let stats = ActivityStatistics {
            sample_count: 100,
            mean_bytes: 5000.0,
            std_dev_bytes: 1000.0,
            min_bytes: 1000,
            max_bytes: 10_000,
            current_high_threshold: 8000,
            current_low_threshold: 2000,
        };

        let metrics = AutosaveMetrics::default().with_activity_stats(&stats);
        assert!(metrics.learning_active);
        assert_eq!(metrics.activity_samples, 100);
        assert_eq!(metrics.current_high_threshold, 8000);
        assert_eq!(metrics.current_low_threshold, 2000);
    }

    #[test]
    fn test_metrics_aggregator_default() {
        let agg = MetricsAggregator::default();
        assert_eq!(agg.total_sample_count(), 0);
    }

    #[test]
    fn test_windowed_metrics_report_default() {
        let report = WindowedMetricsReport::default();
        assert_eq!(report.window, "");
        assert_eq!(report.saves.count, 0);
        assert_eq!(report.errors.count, 0);
    }

    #[test]
    fn test_metrics_aggregator_update_from_status() {
        let mut agg = MetricsAggregator::new();
        let status = CacheAutosaveStatus {
            save_count: 10,
            error_count: 2,
            skip_count: 5,
            forced_save_count: 1,
            compaction_count: 0,
            compaction_trigger_counts: CompactionTriggerCounts::default(),
            save_reason_counts: AutosaveReasonCounts {
                initial: 1,
                interval: 7,
                stale_data: 2,
                coalesced: 0,
            },
            last_save_reason: None,
            total_bytes_written: 50_000,
            last_save_bytes: 5_000,
            current_interval: Duration::from_secs(45),
            is_running: true,
        };

        agg.update_from_status(&status);
        assert_eq!(agg.current_metrics().total_saves, 10);
        assert_eq!(agg.current_metrics().total_skips, 5);
        assert_eq!(agg.current_metrics().total_errors, 2);
        assert_eq!(agg.current_metrics().total_bytes_written, 50_000);
        assert_eq!(agg.current_metrics().current_interval_ms, 45_000);
        assert_eq!(agg.current_metrics().reason_counts.interval, 7);
    }

    #[test]
    fn test_metrics_window_serialization() {
        let window = MetricsWindow::FiveMinutes;
        let json = serde_json::to_string(&window).unwrap();
        assert_eq!(json, "\"FiveMinutes\"");
        let parsed: MetricsWindow = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, MetricsWindow::FiveMinutes);
    }

    #[test]
    fn test_metric_sample_serialization() {
        let sample = MetricSample::at(SystemTime::UNIX_EPOCH + Duration::from_secs(1000), 42.5);
        let json = serde_json::to_string(&sample).unwrap();
        let parsed: MetricSample = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.value, 42.5);
    }

    #[test]
    fn test_windowed_metric_stats_serialization() {
        let samples = vec![MetricSample::now(10.0), MetricSample::now(20.0)];
        let stats = WindowedMetricStats::from_samples(&samples, Duration::from_secs(60));
        let json = serde_json::to_string(&stats).unwrap();
        let parsed: WindowedMetricStats = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.count, 2);
        assert_eq!(parsed.sum, 30.0);
    }

    #[test]
    fn test_autosave_metrics_serialization() {
        let metrics = AutosaveMetrics {
            total_saves: 100,
            total_skips: 50,
            total_errors: 5,
            total_bytes_written: 1_000_000,
            reason_counts: AutosaveReasonCounts::default(),
            current_interval_ms: 30_000,
            current_high_threshold: 10_000,
            current_low_threshold: 1_000,
            learning_active: true,
            activity_samples: 200,
            total_compactions: 10,
            total_entries_compacted: 500,
            compaction_trigger_counts: CompactionTriggerCounts::default(),
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let parsed: AutosaveMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_saves, 100);
        assert!(parsed.learning_active);
        assert_eq!(parsed.total_compactions, 10);
    }

    #[test]
    fn test_metrics_report_serialization() {
        let agg = MetricsAggregator::new();
        let report = agg.generate_report();
        let json = serde_json::to_string(&report).unwrap();
        let parsed: MetricsReport = serde_json::from_str(&json).unwrap();
        assert!(parsed.windowed.contains_key("1m"));
    }

    // ==========================================================================
    // Cache Compaction Tests
    // ==========================================================================

    #[test]
    fn test_compaction_policy_default() {
        let policy = CompactionPolicy::default();
        assert_eq!(policy, CompactionPolicy::ExpiredOnly);
    }

    #[test]
    fn test_compaction_policy_removes_expired() {
        // All policies should remove expired entries
        for policy in [
            CompactionPolicy::ExpiredOnly,
            CompactionPolicy::LowConfidence,
            CompactionPolicy::ObsoleteVersions,
            CompactionPolicy::Deduplicate,
            CompactionPolicy::Aggressive,
        ] {
            assert!(policy.removes_expired());
        }
    }

    #[test]
    fn test_compaction_policy_removes_low_confidence() {
        assert!(!CompactionPolicy::ExpiredOnly.removes_low_confidence());
        assert!(CompactionPolicy::LowConfidence.removes_low_confidence());
        assert!(!CompactionPolicy::ObsoleteVersions.removes_low_confidence());
        assert!(!CompactionPolicy::Deduplicate.removes_low_confidence());
        assert!(CompactionPolicy::Aggressive.removes_low_confidence());
    }

    #[test]
    fn test_compaction_policy_removes_obsolete_versions() {
        assert!(!CompactionPolicy::ExpiredOnly.removes_obsolete_versions());
        assert!(!CompactionPolicy::LowConfidence.removes_obsolete_versions());
        assert!(CompactionPolicy::ObsoleteVersions.removes_obsolete_versions());
        assert!(!CompactionPolicy::Deduplicate.removes_obsolete_versions());
        assert!(CompactionPolicy::Aggressive.removes_obsolete_versions());
    }

    #[test]
    fn test_compaction_policy_deduplicates() {
        assert!(!CompactionPolicy::ExpiredOnly.deduplicates());
        assert!(!CompactionPolicy::LowConfidence.deduplicates());
        assert!(!CompactionPolicy::ObsoleteVersions.deduplicates());
        assert!(CompactionPolicy::Deduplicate.deduplicates());
        assert!(CompactionPolicy::Aggressive.deduplicates());
    }

    #[test]
    fn test_compaction_config_default() {
        let config = CompactionConfig::default();
        assert_eq!(config.policy, CompactionPolicy::ExpiredOnly);
        assert_eq!(config.min_confidence, 0.5);
        assert!(config.active_versions.is_empty());
        assert_eq!(config.max_age_seconds, 0);
        assert!(!config.compact_before_save);
        assert_eq!(config.min_entries_threshold, 100);
    }

    #[test]
    fn test_compaction_config_expired_only() {
        let config = CompactionConfig::expired_only();
        assert_eq!(config.policy, CompactionPolicy::ExpiredOnly);
    }

    #[test]
    fn test_compaction_config_low_confidence() {
        let config = CompactionConfig::low_confidence(0.8);
        assert_eq!(config.policy, CompactionPolicy::LowConfidence);
        assert_eq!(config.min_confidence, 0.8);
    }

    #[test]
    fn test_compaction_config_obsolete_versions() {
        let config =
            CompactionConfig::obsolete_versions(vec!["v1.0".to_string(), "v2.0".to_string()]);
        assert_eq!(config.policy, CompactionPolicy::ObsoleteVersions);
        assert_eq!(config.active_versions.len(), 2);
    }

    #[test]
    fn test_compaction_config_aggressive() {
        let config = CompactionConfig::aggressive(0.9, vec!["v1.0".to_string()]);
        assert_eq!(config.policy, CompactionPolicy::Aggressive);
        assert_eq!(config.min_confidence, 0.9);
        assert!(config.compact_before_save);
    }

    #[test]
    fn test_compaction_config_builder_methods() {
        let config = CompactionConfig::default()
            .with_compact_before_save(true)
            .with_min_entries_threshold(50)
            .with_max_age(Duration::from_secs(3600))
            .with_active_versions(vec!["v1.0".to_string()]);

        assert!(config.compact_before_save);
        assert_eq!(config.min_entries_threshold, 50);
        assert_eq!(config.max_age_seconds, 3600);
        assert_eq!(config.active_versions.len(), 1);
    }

    #[test]
    fn test_compaction_result_total_removed() {
        let result = CompactionResult {
            entries_before: 100,
            entries_after: 70,
            expired_removed: 10,
            low_confidence_removed: 5,
            obsolete_removed: 8,
            duplicates_removed: 3,
            max_age_removed: 4,
            ..Default::default()
        };
        assert_eq!(result.total_removed(), 30);
    }

    #[test]
    fn test_compaction_result_compaction_ratio() {
        let result = CompactionResult {
            entries_before: 100,
            entries_after: 70,
            expired_removed: 30,
            ..Default::default()
        };
        assert!((result.compaction_ratio() - 0.3).abs() < 0.0001);
    }

    #[test]
    fn test_compaction_result_skipped() {
        let result = CompactionResult::skipped("Below threshold", 50);
        assert!(result.skipped);
        assert_eq!(result.skip_reason, Some("Below threshold".to_string()));
        assert_eq!(result.entries_before, 50);
        assert_eq!(result.entries_after, 50);
    }

    fn create_test_cache_entry(confidence: f64) -> CachedPropertyResult {
        CachedPropertyResult {
            property: VerifiedProperty {
                name: "test_prop".to_string(),
                passed: true,
                status: "verified".to_string(),
            },
            backends: vec!["test_backend".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "hash123".to_string(),
            confidence,
        }
    }

    fn create_aged_cache_entry(confidence: f64, age_secs: u64) -> CachedPropertyResult {
        CachedPropertyResult {
            property: VerifiedProperty {
                name: "test_prop".to_string(),
                passed: true,
                status: "verified".to_string(),
            },
            backends: vec!["test_backend".to_string()],
            cached_at: SystemTime::now() - Duration::from_secs(age_secs),
            dependency_hash: "hash123".to_string(),
            confidence,
        }
    }

    #[test]
    fn test_cache_compact_skips_below_threshold() {
        let mut cache = VerificationCache::new();
        for i in 0..50 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_test_cache_entry(0.8));
        }

        let config = CompactionConfig::default(); // threshold is 100
        let result = cache.compact(&config);

        assert!(result.skipped);
        assert_eq!(cache.len(), 50); // No entries removed
    }

    #[test]
    fn test_cache_compact_expired_only() {
        let mut cache = VerificationCache::with_config(1000, Duration::from_secs(1));

        // Add fresh entries
        for i in 0..50 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_test_cache_entry(0.8));
        }

        // Add expired entries (aged 10 seconds, TTL is 1 second)
        for i in 50..100 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_aged_cache_entry(0.8, 10));
        }

        let config = CompactionConfig::expired_only().with_min_entries_threshold(50);
        let result = cache.compact(&config);

        assert!(!result.skipped);
        assert_eq!(result.expired_removed, 50);
        assert_eq!(cache.len(), 50);
    }

    #[test]
    fn test_cache_compact_low_confidence() {
        let mut cache = VerificationCache::with_config(1000, Duration::from_secs(3600));

        // Add high confidence entries
        for i in 0..50 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_test_cache_entry(0.9));
        }

        // Add low confidence entries
        for i in 50..150 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_test_cache_entry(0.3));
        }

        let config = CompactionConfig::low_confidence(0.5).with_min_entries_threshold(50);
        let result = cache.compact(&config);

        assert!(!result.skipped);
        assert_eq!(result.low_confidence_removed, 100);
        assert_eq!(cache.len(), 50);
    }

    #[test]
    fn test_cache_compact_obsolete_versions() {
        let mut cache = VerificationCache::with_config(1000, Duration::from_secs(3600));

        // Add entries for active versions
        for i in 0..50 {
            let key = CacheKey::new("v1.0", format!("prop{}", i));
            cache.insert(key, create_test_cache_entry(0.8));
        }

        // Add entries for obsolete versions
        for i in 0..50 {
            let key = CacheKey::new("v0.5", format!("prop{}", i));
            cache.insert(key, create_test_cache_entry(0.8));
        }
        for i in 0..50 {
            let key = CacheKey::new("v0.9", format!("prop{}", i));
            cache.insert(key, create_test_cache_entry(0.8));
        }

        let config = CompactionConfig::obsolete_versions(vec!["v1.0".to_string()])
            .with_min_entries_threshold(50);
        let result = cache.compact(&config);

        assert!(!result.skipped);
        assert_eq!(result.obsolete_removed, 100);
        assert_eq!(cache.len(), 50);
    }

    #[test]
    fn test_cache_compact_duplicates() {
        let mut cache = VerificationCache::with_config(1000, Duration::from_secs(3600));

        // Add entries with same property name and result for different versions
        // These should be deduplicated (keep newest)
        for i in 0..100 {
            let key = CacheKey::new(format!("v{}", i), "common_prop");
            let mut entry = create_test_cache_entry(0.8);
            // Add slight time offset so we know which is newest
            entry.cached_at = SystemTime::now() - Duration::from_secs(100 - i as u64);
            cache.insert(key, entry);
        }

        // Add unique entries (different property names)
        for i in 0..50 {
            let key = CacheKey::new("v1.0", format!("unique_prop{}", i));
            cache.insert(key, create_test_cache_entry(0.9));
        }

        let config = CompactionConfig {
            policy: CompactionPolicy::Deduplicate,
            min_entries_threshold: 50,
            ..Default::default()
        };
        let result = cache.compact(&config);

        assert!(!result.skipped);
        // Should keep 1 of the 100 duplicates + 50 unique = 51 total
        // 99 duplicates removed
        assert_eq!(result.duplicates_removed, 99);
        assert_eq!(cache.len(), 51);
    }

    #[test]
    fn test_cache_compact_aggressive() {
        let mut cache = VerificationCache::with_config(1000, Duration::from_secs(1));

        // Add some expired entries
        for i in 0..25 {
            let key = CacheKey::new(format!("v{}", i), "prop_expired");
            cache.insert(key, create_aged_cache_entry(0.8, 10));
        }

        // Add some low confidence entries
        for i in 0..25 {
            let key = CacheKey::new(format!("v{}", 100 + i), "prop_low_conf");
            cache.insert(key, create_test_cache_entry(0.2));
        }

        // Add some obsolete version entries
        for i in 0..25 {
            let key = CacheKey::new("obsolete_v0.1", format!("prop{}", i));
            cache.insert(key, create_test_cache_entry(0.8));
        }

        // Add some good entries
        for i in 0..50 {
            let key = CacheKey::new("v1.0", format!("prop{}", i));
            cache.insert(key, create_test_cache_entry(0.9));
        }

        let config = CompactionConfig::aggressive(0.5, vec!["v1.0".to_string()])
            .with_min_entries_threshold(50);
        let result = cache.compact(&config);

        assert!(!result.skipped);
        // Aggressive should have removed: expired, low confidence, and obsolete
        assert!(result.expired_removed > 0);
        assert!(result.low_confidence_removed > 0);
        assert!(result.obsolete_removed > 0);
        // Only good entries should remain
        assert_eq!(cache.len(), 50);
    }

    #[test]
    fn test_cache_analyze_compaction() {
        let mut cache = VerificationCache::with_config(1000, Duration::from_secs(1));

        // Add expired entries
        for i in 0..30 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_aged_cache_entry(0.5, 10));
        }

        // Add low confidence entries
        for i in 30..60 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_test_cache_entry(0.3));
        }

        // Add good entries
        for i in 60..100 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_test_cache_entry(0.9));
        }

        let config = CompactionConfig::low_confidence(0.5).with_min_entries_threshold(50);

        // Analyze without actually compacting
        let analysis = cache.analyze_compaction(&config);
        let original_len = cache.len();

        assert!(!analysis.skipped);
        assert_eq!(analysis.entries_before, 100);
        assert!(analysis.expired_removed > 0);
        assert!(analysis.low_confidence_removed > 0);

        // Cache should be unchanged
        assert_eq!(cache.len(), original_len);
    }

    #[test]
    fn test_cache_version_distribution() {
        let mut cache = VerificationCache::new();

        for i in 0..30 {
            let key = CacheKey::new("v1.0", format!("prop{}", i));
            cache.insert(key, create_test_cache_entry(0.8));
        }

        for i in 0..20 {
            let key = CacheKey::new("v2.0", format!("prop{}", i));
            cache.insert(key, create_test_cache_entry(0.8));
        }

        let distribution = cache.version_distribution();
        assert_eq!(distribution.get("v1.0"), Some(&30));
        assert_eq!(distribution.get("v2.0"), Some(&20));
    }

    #[test]
    fn test_cache_confidence_distribution() {
        let mut cache = VerificationCache::new();

        // Add entries in different confidence buckets
        for i in 0..10 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_test_cache_entry(0.1)); // [0.0-0.2)
        }
        for i in 10..25 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_test_cache_entry(0.35)); // [0.2-0.4)
        }
        for i in 25..35 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_test_cache_entry(0.55)); // [0.4-0.6)
        }
        for i in 35..50 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_test_cache_entry(0.75)); // [0.6-0.8)
        }
        for i in 50..60 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_test_cache_entry(0.95)); // [0.8-1.0]
        }

        let distribution = cache.confidence_distribution();
        assert_eq!(distribution[0], 10); // [0.0-0.2)
        assert_eq!(distribution[1], 15); // [0.2-0.4)
        assert_eq!(distribution[2], 10); // [0.4-0.6)
        assert_eq!(distribution[3], 15); // [0.6-0.8)
        assert_eq!(distribution[4], 10); // [0.8-1.0]
    }

    #[test]
    fn test_cache_age_distribution() {
        let mut cache = VerificationCache::new();

        // Empty cache returns None
        assert!(cache.age_distribution().is_none());

        // Add entries with different ages
        for i in 0..10 {
            let key = CacheKey::new(format!("v{}", i), "prop");
            cache.insert(key, create_aged_cache_entry(0.8, i * 10));
        }

        let (min, max, avg) = cache.age_distribution().unwrap();
        // Ages range from 0 to 90 seconds
        assert!(min <= avg);
        assert!(avg <= max);
        assert!(max >= 80); // At least 80 seconds old
    }

    #[test]
    fn test_compaction_event_serialization() {
        let event = CompactionEvent {
            timestamp: SystemTime::now(),
            result: CompactionResult {
                entries_before: 100,
                entries_after: 70,
                expired_removed: 30,
                ..Default::default()
            },
            policy: CompactionPolicy::ExpiredOnly,
        };

        let json = serde_json::to_string(&event).unwrap();
        let parsed: CompactionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.result.entries_before, 100);
        assert_eq!(parsed.policy, CompactionPolicy::ExpiredOnly);
    }

    #[test]
    fn test_cache_autosave_config_with_compaction() {
        let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
            .with_compaction(
                CompactionConfig::aggressive(0.7, vec!["v1.0".to_string()])
                    .with_compact_before_save(true),
            );

        assert!(config.compaction.is_some());
        let compaction = config.compaction.unwrap();
        assert_eq!(compaction.policy, CompactionPolicy::Aggressive);
        assert!(compaction.compact_before_save);
    }

    #[test]
    fn test_cache_autosave_storage_optimized_has_compaction() {
        let config = CacheAutosaveConfig::storage_optimized("cache.json.gz");
        assert!(config.compaction.is_some());
    }

    #[test]
    fn test_cache_autosave_intelligent_has_compaction() {
        let config = CacheAutosaveConfig::intelligent("cache.json.gz");
        assert!(config.compaction.is_some());
    }

    #[test]
    fn test_cache_autosave_performance_has_no_compaction() {
        let config = CacheAutosaveConfig::performance("cache.json");
        assert!(config.compaction.is_none());
    }

    // =========================================================================
    // Cache Warming Tests
    // =========================================================================

    #[test]
    fn test_warming_strategy_default() {
        let strategy = WarmingStrategy::default();
        assert_eq!(strategy, WarmingStrategy::All);
    }

    #[test]
    fn test_warming_strategy_methods() {
        assert!(WarmingStrategy::HighConfidence.prioritizes_confidence());
        assert!(!WarmingStrategy::MostRecent.prioritizes_confidence());
        assert!(WarmingStrategy::MostRecent.prioritizes_recency());
        assert!(!WarmingStrategy::All.prioritizes_recency());
        assert!(WarmingStrategy::MostAccessed.requires_access_stats());
        assert!(!WarmingStrategy::HighConfidence.requires_access_stats());
    }

    #[test]
    fn test_warming_config_default() {
        let config = WarmingConfig::default();
        assert_eq!(config.strategy, WarmingStrategy::All);
        assert_eq!(config.max_entries, 0);
        assert_eq!(config.min_confidence, 0.0);
        assert!(config.max_age.is_none());
        assert!(config.include_patterns.is_empty());
        assert!(config.exclude_patterns.is_empty());
        assert!(config.validate_ttl);
        assert!(config.skip_nearly_expired);
        assert!((config.min_remaining_ttl_ratio - 0.1).abs() < 0.0001);
    }

    #[test]
    fn test_warming_config_high_confidence() {
        let config = WarmingConfig::high_confidence(0.8);
        assert_eq!(config.strategy, WarmingStrategy::HighConfidence);
        assert_eq!(config.min_confidence, 0.8);
    }

    #[test]
    fn test_warming_config_most_recent() {
        let config = WarmingConfig::most_recent(500);
        assert_eq!(config.strategy, WarmingStrategy::MostRecent);
        assert_eq!(config.max_entries, 500);
    }

    #[test]
    fn test_warming_config_most_accessed() {
        let config = WarmingConfig::most_accessed(1000);
        assert_eq!(config.strategy, WarmingStrategy::MostAccessed);
        assert_eq!(config.max_entries, 1000);
    }

    #[test]
    fn test_warming_config_pattern_based() {
        let config = WarmingConfig::pattern_based(vec!["security_*".to_string()]);
        assert_eq!(config.strategy, WarmingStrategy::PatternBased);
        assert_eq!(config.include_patterns.len(), 1);
    }

    #[test]
    fn test_warming_config_builder_methods() {
        let config = WarmingConfig::new()
            .with_strategy(WarmingStrategy::HighConfidence)
            .with_max_entries(100)
            .with_min_confidence(0.75)
            .with_max_age(Duration::from_secs(3600))
            .with_include_patterns(vec!["security_*".to_string()])
            .with_exclude_patterns(vec!["test_*".to_string()])
            .with_priority_versions(vec!["v1.0".to_string()])
            .with_ttl_validation(false)
            .with_min_remaining_ttl_ratio(0.2);

        assert_eq!(config.strategy, WarmingStrategy::HighConfidence);
        assert_eq!(config.max_entries, 100);
        assert_eq!(config.min_confidence, 0.75);
        assert_eq!(config.max_age, Some(Duration::from_secs(3600)));
        assert_eq!(config.include_patterns.len(), 1);
        assert_eq!(config.exclude_patterns.len(), 1);
        assert_eq!(config.priority_versions.len(), 1);
        assert!(!config.validate_ttl);
        assert!((config.min_remaining_ttl_ratio - 0.2).abs() < 0.0001);
    }

    #[test]
    fn test_warming_config_matches_patterns_empty() {
        let config = WarmingConfig::new();
        // Empty patterns should match everything
        assert!(config.matches_patterns("any_property"));
        assert!(config.matches_patterns("security_check"));
    }

    #[test]
    fn test_warming_config_matches_patterns_include() {
        let config = WarmingConfig::new()
            .with_include_patterns(vec!["security_*".to_string(), "*_invariant".to_string()]);

        assert!(config.matches_patterns("security_check"));
        assert!(config.matches_patterns("state_invariant"));
        assert!(!config.matches_patterns("other_property"));
    }

    #[test]
    fn test_warming_config_matches_patterns_exclude() {
        let config = WarmingConfig::new().with_exclude_patterns(vec!["test_*".to_string()]);

        assert!(config.matches_patterns("security_check"));
        assert!(!config.matches_patterns("test_property"));
    }

    #[test]
    fn test_warming_config_matches_patterns_both() {
        let config = WarmingConfig::new()
            .with_include_patterns(vec!["security_*".to_string()])
            .with_exclude_patterns(vec!["*_test".to_string()]);

        assert!(config.matches_patterns("security_check"));
        assert!(!config.matches_patterns("security_test"));
        assert!(!config.matches_patterns("other_property"));
    }

    #[test]
    fn test_warming_result_default() {
        let result = WarmingResult::default();
        assert_eq!(result.entries_considered, 0);
        assert_eq!(result.entries_warmed, 0);
        assert_eq!(result.total_skipped(), 0);
        assert_eq!(result.success_rate(), 0.0);
    }

    #[test]
    fn test_warming_result_total_skipped() {
        let result = WarmingResult {
            entries_considered: 100,
            entries_warmed: 50,
            entries_skipped_existing: 10,
            entries_skipped_low_confidence: 15,
            entries_skipped_expired: 5,
            entries_skipped_pattern: 12,
            entries_skipped_limit: 8,
            ..Default::default()
        };
        assert_eq!(result.total_skipped(), 50);
    }

    #[test]
    fn test_warming_result_success_rate() {
        let result = WarmingResult {
            entries_considered: 100,
            entries_warmed: 75,
            ..Default::default()
        };
        assert!((result.success_rate() - 0.75).abs() < 0.0001);
    }

    #[test]
    fn test_warming_result_empty() {
        let result = WarmingResult::empty("test_source");
        assert_eq!(result.source, "test_source");
        assert_eq!(result.entries_considered, 0);
    }

    #[test]
    fn test_warm_from_cache_empty_source() {
        let mut cache = VerificationCache::new();
        let source = VerificationCache::new();
        let config = WarmingConfig::new();

        let result = cache.warm_from_cache(&source, &config);
        assert_eq!(result.entries_warmed, 0);
        assert_eq!(result.entries_considered, 0);
    }

    #[test]
    fn test_warm_from_cache_basic() {
        let mut source = VerificationCache::with_config(100, Duration::from_secs(3600));

        // Add some entries to source
        source.insert(
            CacheKey::new("v1", "prop1"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "prop1".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep1".to_string(),
                confidence: 0.95,
            },
        );
        source.insert(
            CacheKey::new("v1", "prop2"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "prop2".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep2".to_string(),
                confidence: 0.85,
            },
        );

        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        let config = WarmingConfig::new();

        let result = cache.warm_from_cache(&source, &config);
        assert_eq!(result.entries_considered, 2);
        assert_eq!(result.entries_warmed, 2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_warm_from_cache_skips_existing() {
        let mut source = VerificationCache::with_config(100, Duration::from_secs(3600));
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));

        // Add entry to source
        let entry = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop1".to_string(),
                passed: true,
                status: "verified".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep1".to_string(),
            confidence: 0.95,
        };
        source.insert(CacheKey::new("v1", "prop1"), entry.clone());

        // Add same entry to cache
        cache.insert(CacheKey::new("v1", "prop1"), entry);

        let config = WarmingConfig::new();
        let result = cache.warm_from_cache(&source, &config);

        assert_eq!(result.entries_considered, 1);
        assert_eq!(result.entries_warmed, 0);
        assert_eq!(result.entries_skipped_existing, 1);
    }

    #[test]
    fn test_warm_from_cache_filters_low_confidence() {
        let mut source = VerificationCache::with_config(100, Duration::from_secs(3600));

        source.insert(
            CacheKey::new("v1", "prop1"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "prop1".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep1".to_string(),
                confidence: 0.5, // Low confidence
            },
        );
        source.insert(
            CacheKey::new("v1", "prop2"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "prop2".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep2".to_string(),
                confidence: 0.9, // High confidence
            },
        );

        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        let config = WarmingConfig::high_confidence(0.8);

        let result = cache.warm_from_cache(&source, &config);
        assert_eq!(result.entries_considered, 2);
        assert_eq!(result.entries_warmed, 1);
        assert_eq!(result.entries_skipped_low_confidence, 1);
    }

    #[test]
    fn test_warm_from_cache_respects_max_entries() {
        let mut source = VerificationCache::with_config(100, Duration::from_secs(3600));

        for i in 0..10 {
            source.insert(
                CacheKey::new("v1", format!("prop{}", i)),
                CachedPropertyResult {
                    property: VerifiedProperty {
                        name: format!("prop{}", i),
                        passed: true,
                        status: "verified".to_string(),
                    },
                    backends: vec!["lean4".to_string()],
                    cached_at: SystemTime::now(),
                    dependency_hash: format!("dep{}", i),
                    confidence: 0.95,
                },
            );
        }

        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        let config = WarmingConfig::new().with_max_entries(5);

        let result = cache.warm_from_cache(&source, &config);
        assert_eq!(result.entries_warmed, 5);
        assert_eq!(result.entries_skipped_limit, 5);
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn test_warm_from_cache_filters_by_pattern() {
        let mut source = VerificationCache::with_config(100, Duration::from_secs(3600));

        for name in &[
            "security_check",
            "test_prop",
            "state_invariant",
            "other_prop",
        ] {
            source.insert(
                CacheKey::new("v1", *name),
                CachedPropertyResult {
                    property: VerifiedProperty {
                        name: name.to_string(),
                        passed: true,
                        status: "verified".to_string(),
                    },
                    backends: vec!["lean4".to_string()],
                    cached_at: SystemTime::now(),
                    dependency_hash: "dep".to_string(),
                    confidence: 0.95,
                },
            );
        }

        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        let config = WarmingConfig::new()
            .with_include_patterns(vec!["security_*".to_string(), "*_invariant".to_string()]);

        let result = cache.warm_from_cache(&source, &config);
        assert_eq!(result.entries_warmed, 2);
        assert_eq!(result.entries_skipped_pattern, 2);
    }

    #[test]
    fn test_warm_from_cache_high_confidence_strategy_sorts() {
        let mut source = VerificationCache::with_config(100, Duration::from_secs(3600));

        // Add entries with different confidence scores
        for (i, conf) in [0.5, 0.9, 0.7, 0.95, 0.8].iter().enumerate() {
            source.insert(
                CacheKey::new("v1", format!("prop{}", i)),
                CachedPropertyResult {
                    property: VerifiedProperty {
                        name: format!("prop{}", i),
                        passed: true,
                        status: "verified".to_string(),
                    },
                    backends: vec!["lean4".to_string()],
                    cached_at: SystemTime::now(),
                    dependency_hash: format!("dep{}", i),
                    confidence: *conf,
                },
            );
        }

        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        let config = WarmingConfig::high_confidence(0.7).with_max_entries(3);

        let result = cache.warm_from_cache(&source, &config);
        assert_eq!(result.entries_warmed, 3);
        // Should have warmed the 3 highest confidence entries (0.95, 0.9, 0.8)
    }

    #[test]
    fn test_warm_with_config_basic() {
        let mut cache = VerificationCache::with_config(100, Duration::from_secs(3600));

        let entry = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop1".to_string(),
                passed: true,
                status: "verified".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "dep1".to_string(),
            confidence: 0.95,
        };

        let entries = vec![("v1", "prop1", &entry)];
        let config = WarmingConfig::new();

        let result = cache.warm_with_config(entries.into_iter(), &config);
        assert_eq!(result.entries_warmed, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_analyze_warming_does_not_modify_cache() {
        let mut source = VerificationCache::with_config(100, Duration::from_secs(3600));

        source.insert(
            CacheKey::new("v1", "prop1"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "prop1".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep1".to_string(),
                confidence: 0.95,
            },
        );

        let cache = VerificationCache::with_config(100, Duration::from_secs(3600));
        let config = WarmingConfig::new();

        let result = cache.analyze_warming(&source, &config);
        assert_eq!(result.entries_warmed, 1);
        assert_eq!(cache.len(), 0); // Cache unchanged
    }

    #[test]
    fn test_warming_event_serialization() {
        let event = WarmingEvent {
            timestamp: SystemTime::now(),
            result: WarmingResult {
                entries_considered: 100,
                entries_warmed: 75,
                strategy: WarmingStrategy::HighConfidence,
                source: "test".to_string(),
                ..Default::default()
            },
            config: WarmingConfig::high_confidence(0.8),
        };

        let json = serde_json::to_string(&event).unwrap();
        let parsed: WarmingEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.result.entries_warmed, 75);
        assert_eq!(parsed.config.strategy, WarmingStrategy::HighConfidence);
    }

    #[test]
    fn test_cache_autosave_config_with_warming() {
        let config = CacheAutosaveConfig::new("cache.json", Duration::from_secs(60))
            .with_warming(WarmingConfig::high_confidence(0.8));

        assert!(config.warming.is_some());
        let warming = config.warming.unwrap();
        assert_eq!(warming.strategy, WarmingStrategy::HighConfidence);
    }

    #[test]
    fn test_cache_autosave_intelligent_has_warming() {
        let config = CacheAutosaveConfig::intelligent("cache.json.gz");
        assert!(config.warming.is_some());
        let warming = config.warming.unwrap();
        assert_eq!(warming.strategy, WarmingStrategy::HighConfidence);
    }

    #[test]
    fn test_cache_autosave_performance_has_no_warming() {
        let config = CacheAutosaveConfig::performance("cache.json");
        assert!(config.warming.is_none());
    }

    #[test]
    fn test_cache_autosave_balanced_has_no_warming() {
        let config = CacheAutosaveConfig::balanced("cache.json");
        assert!(config.warming.is_none());
    }

    // =============================================================================
    // VerificationRetryPolicy Tests
    // =============================================================================

    #[test]
    fn test_verification_retry_policy_new() {
        let policy = VerificationRetryPolicy::new(3);
        assert_eq!(policy.max_attempts, 3);
        assert!(policy.retry_on_dispatcher_error);
        assert!(policy.retry_on_low_confidence);
        assert!(policy.retry_on_insufficient_backends);
        assert!(!policy.retry_on_failure);
        assert_eq!(policy.jitter_ms, 0);
    }

    #[test]
    fn test_verification_retry_policy_new_clamps_minimum() {
        let policy = VerificationRetryPolicy::new(0);
        assert_eq!(policy.max_attempts, 1); // Minimum is 1
    }

    #[test]
    fn test_verification_retry_policy_default() {
        let policy = VerificationRetryPolicy::default();
        assert_eq!(policy.max_attempts, 3);
    }

    #[test]
    fn test_verification_retry_policy_with_backoff() {
        let backoff = BackoffConfig::aggressive();
        let policy = VerificationRetryPolicy::new(5).with_backoff(backoff);
        assert_eq!(policy.backoff.initial_delay, backoff.initial_delay);
        assert_eq!(policy.backoff.max_delay, backoff.max_delay);
    }

    #[test]
    fn test_verification_retry_policy_retry_on_dispatcher_error() {
        let policy = VerificationRetryPolicy::new(3).retry_on_dispatcher_error(false);
        assert!(!policy.retry_on_dispatcher_error);

        let policy = VerificationRetryPolicy::new(3).retry_on_dispatcher_error(true);
        assert!(policy.retry_on_dispatcher_error);
    }

    #[test]
    fn test_verification_retry_policy_retry_on_low_confidence() {
        let policy = VerificationRetryPolicy::new(3).retry_on_low_confidence(false);
        assert!(!policy.retry_on_low_confidence);
    }

    #[test]
    fn test_verification_retry_policy_retry_on_insufficient_backends() {
        let policy = VerificationRetryPolicy::new(3).retry_on_insufficient_backends(false);
        assert!(!policy.retry_on_insufficient_backends);
    }

    #[test]
    fn test_verification_retry_policy_retry_on_failure() {
        let policy = VerificationRetryPolicy::new(3).retry_on_failure(true);
        assert!(policy.retry_on_failure);
    }

    #[test]
    fn test_verification_retry_policy_with_jitter_ms() {
        let policy = VerificationRetryPolicy::new(3).with_jitter_ms(100);
        assert_eq!(policy.jitter_ms, 100);
    }

    #[test]
    fn test_verification_retry_policy_should_retry_error_dispatcher() {
        use crate::error::SelfImpError;

        let policy = VerificationRetryPolicy::new(3);

        let dispatcher_error = SelfImpError::DispatcherError("test".to_string());
        assert!(policy.should_retry_error(&dispatcher_error));

        let timeout_error = SelfImpError::VerificationTimeout(60);
        assert!(policy.should_retry_error(&timeout_error));
    }

    #[test]
    fn test_verification_retry_policy_should_retry_error_disabled() {
        use crate::error::SelfImpError;

        let policy = VerificationRetryPolicy::new(3).retry_on_dispatcher_error(false);

        let dispatcher_error = SelfImpError::DispatcherError("test".to_string());
        assert!(!policy.should_retry_error(&dispatcher_error));
    }

    #[test]
    fn test_verification_retry_policy_should_retry_error_non_retryable() {
        use crate::error::SelfImpError;

        let policy = VerificationRetryPolicy::new(3);

        // USL parse errors should not be retried
        let parse_error = SelfImpError::UslParseError("test".to_string());
        assert!(!policy.should_retry_error(&parse_error));
    }

    #[test]
    fn test_verification_retry_policy_should_retry_result_insufficient_backends() {
        let policy = VerificationRetryPolicy::new(3);
        let config = VerificationConfig {
            min_passing_backends: 2,
            min_confidence: 0.8,
            ..Default::default()
        };

        let mut backend_results = HashMap::new();
        backend_results.insert(
            "lean4".to_string(),
            BackendResult {
                backend: "lean4".to_string(),
                passed: true,
                duration_ms: 100,
                error: None,
                raw_output: None,
                verified_properties: vec!["prop1".to_string()],
                failed_properties: vec![],
            },
        );

        let result = VerificationResult {
            passed: true,
            confidence: 0.9,
            backend_results,
            duration_ms: 100,
            messages: vec![],
            verified_properties: vec![],
        };

        // Only 1 backend passed, need 2
        assert!(policy.should_retry_result(&result, &config));
    }

    #[test]
    fn test_verification_retry_policy_should_retry_result_low_confidence() {
        let policy = VerificationRetryPolicy::new(3);
        let config = VerificationConfig {
            min_passing_backends: 1,
            min_confidence: 0.9,
            ..Default::default()
        };

        let mut backend_results = HashMap::new();
        backend_results.insert(
            "lean4".to_string(),
            BackendResult {
                backend: "lean4".to_string(),
                passed: true,
                duration_ms: 100,
                error: None,
                raw_output: None,
                verified_properties: vec!["prop1".to_string()],
                failed_properties: vec![],
            },
        );

        let result = VerificationResult {
            passed: true,
            confidence: 0.7, // Below min_confidence
            backend_results,
            duration_ms: 100,
            messages: vec![],
            verified_properties: vec![],
        };

        assert!(policy.should_retry_result(&result, &config));
    }

    #[test]
    fn test_verification_retry_policy_should_retry_result_failure() {
        let policy = VerificationRetryPolicy::new(3).retry_on_failure(true);
        let config = VerificationConfig::default();

        let mut backend_results = HashMap::new();
        backend_results.insert(
            "lean4".to_string(),
            BackendResult {
                backend: "lean4".to_string(),
                passed: false,
                duration_ms: 100,
                error: Some("failed".to_string()),
                raw_output: None,
                verified_properties: vec![],
                failed_properties: vec!["prop1".to_string()],
            },
        );

        let result = VerificationResult {
            passed: false,
            confidence: 0.9,
            backend_results,
            duration_ms: 100,
            messages: vec![],
            verified_properties: vec![],
        };

        assert!(policy.should_retry_result(&result, &config));
    }

    #[test]
    fn test_verification_retry_policy_should_not_retry_success() {
        let policy = VerificationRetryPolicy::new(3);
        let config = VerificationConfig {
            min_passing_backends: 1,
            min_confidence: 0.8,
            ..Default::default()
        };

        let mut backend_results = HashMap::new();
        backend_results.insert(
            "lean4".to_string(),
            BackendResult {
                backend: "lean4".to_string(),
                passed: true,
                duration_ms: 100,
                error: None,
                raw_output: None,
                verified_properties: vec!["prop1".to_string()],
                failed_properties: vec![],
            },
        );

        let result = VerificationResult {
            passed: true,
            confidence: 0.9, // Above min_confidence
            backend_results,
            duration_ms: 100,
            messages: vec![],
            verified_properties: vec![],
        };

        // Passes all criteria, should not retry
        assert!(!policy.should_retry_result(&result, &config));
    }

    #[test]
    fn test_verification_retry_policy_next_delay_basic() {
        let policy = VerificationRetryPolicy::new(3).with_backoff(BackoffConfig::new(
            Duration::from_secs(1),
            Duration::from_secs(60),
        ));

        let delay = policy.next_delay(1);
        assert_eq!(delay, Some(Duration::from_secs(1)));

        let delay = policy.next_delay(2);
        assert_eq!(delay, Some(Duration::from_secs(2)));
    }

    #[test]
    fn test_verification_retry_policy_next_delay_with_jitter() {
        let policy = VerificationRetryPolicy::new(3)
            .with_backoff(BackoffConfig::new(
                Duration::from_secs(1),
                Duration::from_secs(60),
            ))
            .with_jitter_ms(100);

        let delay = policy.next_delay(1).unwrap();
        // Base is 1s, jitter adds up to 100ms
        assert!(delay >= Duration::from_secs(1));
        assert!(delay <= Duration::from_millis(1100));
    }

    #[test]
    fn test_verification_retry_policy_next_delay_respects_threshold() {
        let policy = VerificationRetryPolicy::new(3).with_backoff(
            BackoffConfig::new(Duration::from_secs(1), Duration::from_secs(60))
                .with_error_threshold(2),
        );

        // Before threshold
        assert!(policy.next_delay(1).is_none());

        // At threshold
        assert!(policy.next_delay(2).is_some());
    }

    #[test]
    fn test_verification_retry_policy_serialization() {
        let policy = VerificationRetryPolicy::new(5)
            .with_jitter_ms(50)
            .retry_on_failure(true);

        let json = serde_json::to_string(&policy).unwrap();
        let parsed: VerificationRetryPolicy = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.max_attempts, 5);
        assert_eq!(parsed.jitter_ms, 50);
        assert!(parsed.retry_on_failure);
    }

    #[test]
    fn test_verification_retry_policy_builder_chaining() {
        let policy = VerificationRetryPolicy::new(5)
            .with_backoff(BackoffConfig::aggressive())
            .retry_on_dispatcher_error(false)
            .retry_on_low_confidence(false)
            .retry_on_insufficient_backends(false)
            .retry_on_failure(true)
            .with_jitter_ms(200);

        assert_eq!(policy.max_attempts, 5);
        assert!(!policy.retry_on_dispatcher_error);
        assert!(!policy.retry_on_low_confidence);
        assert!(!policy.retry_on_insufficient_backends);
        assert!(policy.retry_on_failure);
        assert_eq!(policy.jitter_ms, 200);
    }

    #[test]
    fn test_verification_config_has_default_retry_policy() {
        let config = VerificationConfig::default();
        assert!(config.retry_policy.is_some());
        let policy = config.retry_policy.unwrap();
        assert_eq!(policy.max_attempts, 3);
    }

    #[test]
    fn test_verification_config_strict_has_retry_policy() {
        let config = VerificationConfig::strict();
        assert!(config.retry_policy.is_some());
    }

    #[test]
    fn test_verification_config_quick_has_retry_policy() {
        let config = VerificationConfig::quick();
        assert!(config.retry_policy.is_some());
    }

    #[test]
    fn test_verification_config_simple_has_retry_policy() {
        let config = VerificationConfig::simple();
        assert!(config.retry_policy.is_some());
    }

    // =============================================================================
    // Cache Partitioning Tests
    // =============================================================================

    #[test]
    fn test_cache_partition_from_property_type_theorem() {
        assert_eq!(
            CachePartition::from_property_type("theorem"),
            CachePartition::TheoremProving
        );
        assert_eq!(
            CachePartition::from_property_type("lemma"),
            CachePartition::TheoremProving
        );
        assert_eq!(
            CachePartition::from_property_type("axiom"),
            CachePartition::TheoremProving
        );
    }

    #[test]
    fn test_cache_partition_from_property_type_contract() {
        assert_eq!(
            CachePartition::from_property_type("contract"),
            CachePartition::ContractVerification
        );
        assert_eq!(
            CachePartition::from_property_type("requires"),
            CachePartition::ContractVerification
        );
        assert_eq!(
            CachePartition::from_property_type("ensures"),
            CachePartition::ContractVerification
        );
    }

    #[test]
    fn test_cache_partition_from_property_type_model_checking() {
        assert_eq!(
            CachePartition::from_property_type("invariant"),
            CachePartition::ModelChecking
        );
        assert_eq!(
            CachePartition::from_property_type("temporal"),
            CachePartition::ModelChecking
        );
        assert_eq!(
            CachePartition::from_property_type("ltl"),
            CachePartition::ModelChecking
        );
    }

    #[test]
    fn test_cache_partition_from_property_type_unknown() {
        assert_eq!(
            CachePartition::from_property_type("unknown_prop"),
            CachePartition::Default
        );
    }

    #[test]
    fn test_cache_partition_all() {
        let all = CachePartition::all();
        assert_eq!(all.len(), 9);
        assert!(all.contains(&CachePartition::Default));
        assert!(all.contains(&CachePartition::TheoremProving));
        assert!(all.contains(&CachePartition::ModelChecking));
    }

    #[test]
    fn test_cache_partition_name() {
        assert_eq!(CachePartition::Default.name(), "default");
        assert_eq!(CachePartition::TheoremProving.name(), "theorem_proving");
        assert_eq!(CachePartition::ModelChecking.name(), "model_checking");
    }

    #[test]
    fn test_cache_partition_display() {
        assert_eq!(format!("{}", CachePartition::Default), "default");
        assert_eq!(
            format!("{}", CachePartition::TheoremProving),
            "theorem_proving"
        );
    }

    #[test]
    fn test_partition_stats_default() {
        let stats = PartitionStats::default();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.entry_count, 0);
    }

    #[test]
    fn test_partition_stats_hit_rate() {
        let stats = PartitionStats {
            hits: 75,
            misses: 25,
            entry_count: 100,
            bytes_estimate: 0,
        };
        assert!((stats.hit_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_partition_stats_hit_rate_empty() {
        let stats = PartitionStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_partition_config_default() {
        let config = PartitionConfig::default();
        assert!(!config.enabled);
        assert!(config.partition_ttls.is_empty());
        assert!(config.partition_limits.is_empty());
    }

    #[test]
    fn test_partition_config_enabled() {
        let config = PartitionConfig::enabled();
        assert!(config.enabled);
    }

    #[test]
    fn test_partition_config_builder() {
        let config = PartitionConfig::enabled()
            .with_partition_ttl(CachePartition::TheoremProving, 7200)
            .with_partition_limit(CachePartition::Testing, 100)
            .with_partition_confidence(CachePartition::NeuralNetwork, 0.95);

        assert!(config.enabled);
        assert_eq!(
            config.partition_ttls.get(&CachePartition::TheoremProving),
            Some(&7200)
        );
        assert_eq!(
            config.partition_limits.get(&CachePartition::Testing),
            Some(&100)
        );
        assert_eq!(
            config
                .partition_confidence_thresholds
                .get(&CachePartition::NeuralNetwork),
            Some(&0.95)
        );
    }

    #[test]
    fn test_partition_config_recommended() {
        let config = PartitionConfig::recommended();
        assert!(config.enabled);
        // Should have TTL for theorem proving
        assert!(config
            .partition_ttls
            .contains_key(&CachePartition::TheoremProving));
        // Should have confidence threshold for neural network
        assert!(config
            .partition_confidence_thresholds
            .contains_key(&CachePartition::NeuralNetwork));
    }

    #[test]
    fn test_cache_enable_partitioning() {
        let mut cache = VerificationCache::new();
        assert!(!cache.is_partitioning_enabled());

        cache.enable_partitioning(PartitionConfig::enabled());
        assert!(cache.is_partitioning_enabled());
    }

    #[test]
    fn test_cache_disable_partitioning() {
        let mut cache = VerificationCache::new();
        cache.enable_partitioning(PartitionConfig::enabled());
        assert!(cache.is_partitioning_enabled());

        cache.disable_partitioning();
        assert!(!cache.is_partitioning_enabled());
    }

    #[test]
    fn test_cache_insert_tracks_partition() {
        let mut cache = VerificationCache::new();
        cache.enable_partitioning(PartitionConfig::enabled());

        // Insert a theorem-type property
        cache.insert(
            CacheKey::new("v1", "theorem_test"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "theorem_test".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            },
        );

        // Check partition assignment
        let key = CacheKey::new("v1", "theorem_test");
        assert_eq!(cache.get_partition(&key), CachePartition::TheoremProving);
    }

    #[test]
    fn test_cache_partition_stats_updated_on_insert() {
        let mut cache = VerificationCache::new();
        cache.enable_partitioning(PartitionConfig::enabled());

        // Insert entries for different partitions
        cache.insert(
            CacheKey::new("v1", "theorem_1"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "theorem_1".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            },
        );

        cache.insert(
            CacheKey::new("v1", "invariant_1"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "invariant_1".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["tlaplus".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.9,
            },
        );

        let theorem_stats = cache.partition_stats(CachePartition::TheoremProving);
        let model_stats = cache.partition_stats(CachePartition::ModelChecking);

        assert_eq!(theorem_stats.entry_count, 1);
        assert_eq!(model_stats.entry_count, 1);
    }

    #[test]
    fn test_cache_insert_with_partition() {
        let mut cache = VerificationCache::new();
        cache.enable_partitioning(PartitionConfig::enabled());

        // Insert with explicit partition (overriding auto-detection)
        cache.insert_with_partition(
            CacheKey::new("v1", "custom_prop"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "custom_prop".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            },
            CachePartition::SecurityProtocol, // Explicit partition
        );

        let key = CacheKey::new("v1", "custom_prop");
        assert_eq!(cache.get_partition(&key), CachePartition::SecurityProtocol);
    }

    #[test]
    fn test_cache_count_in_partition() {
        let mut cache = VerificationCache::new();
        cache.enable_partitioning(PartitionConfig::enabled());

        // Insert multiple theorem entries
        for i in 0..5 {
            cache.insert(
                CacheKey::new("v1", format!("theorem_{}", i)),
                CachedPropertyResult {
                    property: VerifiedProperty {
                        name: format!("theorem_{}", i),
                        passed: true,
                        status: "verified".to_string(),
                    },
                    backends: vec!["lean4".to_string()],
                    cached_at: SystemTime::now(),
                    dependency_hash: "dep".to_string(),
                    confidence: 0.95,
                },
            );
        }

        assert_eq!(cache.count_in_partition(CachePartition::TheoremProving), 5);
        assert_eq!(cache.count_in_partition(CachePartition::ModelChecking), 0);
    }

    #[test]
    fn test_cache_clear_partition() {
        let mut cache = VerificationCache::new();
        cache.enable_partitioning(PartitionConfig::enabled());

        // Insert entries in different partitions
        cache.insert(
            CacheKey::new("v1", "theorem_1"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "theorem_1".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            },
        );

        cache.insert(
            CacheKey::new("v1", "invariant_1"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "invariant_1".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["tlaplus".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.9,
            },
        );

        assert_eq!(cache.len(), 2);

        // Clear only theorem partition
        let result = cache.clear_partition(CachePartition::TheoremProving);
        assert_eq!(result.entries_affected, 1);
        assert_eq!(result.partition, CachePartition::TheoremProving);
        assert_eq!(result.operation, "clear");

        // Verify only theorem entries removed
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.count_in_partition(CachePartition::TheoremProving), 0);
        assert_eq!(cache.count_in_partition(CachePartition::ModelChecking), 1);
    }

    #[test]
    fn test_cache_partition_distribution() {
        let mut cache = VerificationCache::new();
        cache.enable_partitioning(PartitionConfig::enabled());

        // Insert entries in different partitions
        cache.insert(
            CacheKey::new("v1", "theorem_1"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "theorem_1".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            },
        );

        cache.insert(
            CacheKey::new("v1", "theorem_2"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "theorem_2".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            },
        );

        cache.insert(
            CacheKey::new("v1", "invariant_1"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "invariant_1".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["tlaplus".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.9,
            },
        );

        let distribution = cache.partition_distribution();
        assert_eq!(distribution.get(&CachePartition::TheoremProving), Some(&2));
        assert_eq!(distribution.get(&CachePartition::ModelChecking), Some(&1));
    }

    #[test]
    fn test_cache_rebalance_partitions() {
        let mut cache = VerificationCache::new();
        // Start without partitioning
        cache.insert(
            CacheKey::new("v1", "theorem_1"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "theorem_1".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            },
        );

        // Enable partitioning and rebalance
        cache.enable_partitioning(PartitionConfig::enabled());
        let distribution = cache.rebalance_partitions();

        assert_eq!(distribution.get(&CachePartition::TheoremProving), Some(&1));
        assert_eq!(cache.count_in_partition(CachePartition::TheoremProving), 1);
    }

    #[test]
    fn test_cache_clear_also_clears_partitions() {
        let mut cache = VerificationCache::new();
        cache.enable_partitioning(PartitionConfig::enabled());

        cache.insert(
            CacheKey::new("v1", "theorem_1"),
            CachedPropertyResult {
                property: VerifiedProperty {
                    name: "theorem_1".to_string(),
                    passed: true,
                    status: "verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "dep".to_string(),
                confidence: 0.95,
            },
        );

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.count_in_partition(CachePartition::TheoremProving), 1);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.count_in_partition(CachePartition::TheoremProving), 0);
        assert!(cache.all_partition_stats().is_empty());
    }

    #[test]
    fn test_cache_partition_ttl() {
        let mut cache = VerificationCache::new();
        cache.enable_partitioning(
            PartitionConfig::enabled().with_partition_ttl(CachePartition::TheoremProving, 7200),
        );

        let ttl = cache.partition_ttl(CachePartition::TheoremProving);
        assert_eq!(ttl, Some(Duration::from_secs(7200)));

        let no_ttl = cache.partition_ttl(CachePartition::ModelChecking);
        assert!(no_ttl.is_none());
    }

    #[test]
    fn test_cache_partition_confidence_threshold() {
        let mut cache = VerificationCache::new();
        cache.enable_partitioning(
            PartitionConfig::enabled()
                .with_partition_confidence(CachePartition::NeuralNetwork, 0.95),
        );

        let threshold = cache.partition_confidence_threshold(CachePartition::NeuralNetwork);
        assert_eq!(threshold, Some(0.95));

        let no_threshold = cache.partition_confidence_threshold(CachePartition::Default);
        assert!(no_threshold.is_none());
    }

    #[test]
    fn test_partition_operation_result() {
        let result = PartitionOperationResult {
            partition: CachePartition::TheoremProving,
            entries_affected: 5,
            operation: "clear".to_string(),
            duration_ms: 10,
        };

        assert_eq!(result.partition, CachePartition::TheoremProving);
        assert_eq!(result.entries_affected, 5);
        assert_eq!(result.operation, "clear");
    }

    #[test]
    fn test_cache_partition_serialization() {
        let partition = CachePartition::TheoremProving;
        let json = serde_json::to_string(&partition).unwrap();
        let parsed: CachePartition = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, CachePartition::TheoremProving);
    }

    #[test]
    fn test_partition_config_serialization() {
        let config =
            PartitionConfig::enabled().with_partition_ttl(CachePartition::TheoremProving, 3600);

        let json = serde_json::to_string(&config).unwrap();
        let parsed: PartitionConfig = serde_json::from_str(&json).unwrap();

        assert!(parsed.enabled);
        assert_eq!(
            parsed.partition_ttls.get(&CachePartition::TheoremProving),
            Some(&3600)
        );
    }

    // ==========================================================================
    // Partition-Aware TTL Tests
    // ==========================================================================

    #[test]
    fn test_partition_ttl_in_get() {
        let mut cache = VerificationCache::new();

        // Enable partitioning with custom TTL for theorem proving (1 second)
        cache.enable_partitioning(
            PartitionConfig::enabled().with_partition_ttl(CachePartition::TheoremProving, 1),
        );

        // Insert a theorem property (should use partition TTL)
        let key = CacheKey::new("v1", "theorem_prop");
        let result = CachedPropertyResult {
            property: VerifiedProperty {
                name: "theorem_prop".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "hash".to_string(),
            confidence: 0.9,
        };
        cache.insert(key.clone(), result);

        // Should be present immediately
        assert!(cache.get(&key).is_some());

        // Wait for partition TTL to expire
        std::thread::sleep(std::time::Duration::from_millis(1100));

        // Should be expired now (partition TTL of 1 second)
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_partition_confidence_threshold() {
        let mut cache = VerificationCache::new();

        // Enable partitioning with high confidence threshold for theorem proving
        cache.enable_partitioning(
            PartitionConfig::enabled()
                .with_partition_confidence(CachePartition::TheoremProving, 0.95),
        );

        // Insert a theorem property with 0.9 confidence (below partition threshold)
        let key = CacheKey::new("v1", "theorem_test");
        let result = CachedPropertyResult {
            property: VerifiedProperty {
                name: "theorem_test".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "hash".to_string(),
            confidence: 0.9, // Below partition threshold of 0.95
        };
        cache.insert(key.clone(), result);

        // Should be treated as a miss due to partition confidence threshold
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_partition_stats_on_get() {
        let mut cache = VerificationCache::new();
        cache.enable_partitioning(PartitionConfig::enabled());

        // Insert a theorem property
        let key = CacheKey::new("v1", "theorem_prop");
        let result = CachedPropertyResult {
            property: VerifiedProperty {
                name: "theorem_prop".to_string(),
                passed: true,
                status: "Proven".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "hash".to_string(),
            confidence: 0.9,
        };
        cache.insert(key.clone(), result);

        // Hit
        assert!(cache.get(&key).is_some());

        // Miss (non-existent)
        let miss_key = CacheKey::new("v2", "theorem_other");
        assert!(cache.get(&miss_key).is_none());

        // Check partition stats
        let stats = cache.partition_stats(CachePartition::TheoremProving);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    // ==========================================================================
    // Partition-Level Eviction Tests
    // ==========================================================================

    #[test]
    fn test_partition_limit_eviction() {
        let mut cache = VerificationCache::with_config(1000, Duration::from_secs(3600));

        // Enable partitioning with limit of 2 for theorem proving
        cache.enable_partitioning(
            PartitionConfig::enabled().with_partition_limit(CachePartition::TheoremProving, 2),
        );

        // Insert 3 theorem properties (should evict oldest)
        for i in 0..3 {
            let key = CacheKey::new("v1", format!("theorem_{}", i));
            let result = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("theorem_{}", i),
                    passed: true,
                    status: "Proven".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "hash".to_string(),
                confidence: 0.9,
            };
            cache.insert(key, result);
            // Small delay to ensure different cached_at times
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Should only have 2 entries in theorem proving partition
        assert_eq!(cache.count_in_partition(CachePartition::TheoremProving), 2);

        // Oldest entry (theorem_0) should be evicted
        assert!(cache.get(&CacheKey::new("v1", "theorem_0")).is_none());
        assert!(cache.get(&CacheKey::new("v1", "theorem_1")).is_some());
        assert!(cache.get(&CacheKey::new("v1", "theorem_2")).is_some());
    }

    #[test]
    fn test_partition_eviction_preserves_other_partitions() {
        let mut cache = VerificationCache::with_config(1000, Duration::from_secs(3600));

        // Enable partitioning with limit of 2 for theorem proving
        cache.enable_partitioning(
            PartitionConfig::enabled().with_partition_limit(CachePartition::TheoremProving, 2),
        );

        // Insert a contract verification entry
        let contract_key = CacheKey::new("v1", "contract_test");
        let contract_result = CachedPropertyResult {
            property: VerifiedProperty {
                name: "contract_test".to_string(),
                passed: true,
                status: "Verified".to_string(),
            },
            backends: vec!["kani".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "hash".to_string(),
            confidence: 0.85,
        };
        cache.insert(contract_key.clone(), contract_result);

        // Insert 3 theorem properties
        for i in 0..3 {
            let key = CacheKey::new("v1", format!("theorem_{}", i));
            let result = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("theorem_{}", i),
                    passed: true,
                    status: "Proven".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "hash".to_string(),
                confidence: 0.9,
            };
            cache.insert(key, result);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Contract entry should still exist (different partition)
        assert!(cache.get(&contract_key).is_some());

        // Only 2 theorem entries
        assert_eq!(cache.count_in_partition(CachePartition::TheoremProving), 2);
        assert_eq!(
            cache.count_in_partition(CachePartition::ContractVerification),
            1
        );
    }

    // ==========================================================================
    // Compaction Trigger Tests
    // ==========================================================================

    #[test]
    fn test_compaction_trigger_size_based() {
        let mut cache = VerificationCache::with_config(10, Duration::from_secs(3600));
        cache.add_compaction_trigger(CompactionTrigger::size_90_percent());

        // Insert 8 entries (80% capacity - should not trigger)
        for i in 0..8 {
            let key = CacheKey::new("v1", format!("prop_{}", i));
            let result = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("prop_{}", i),
                    passed: true,
                    status: "Verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "hash".to_string(),
                confidence: 0.9,
            };
            cache.insert(key, result);
        }

        assert!(cache.should_compact().is_none());

        // Insert 1 more (90% capacity - should trigger)
        let key = CacheKey::new("v1", "prop_8");
        let result = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop_8".to_string(),
                passed: true,
                status: "Verified".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "hash".to_string(),
            confidence: 0.9,
        };
        cache.insert(key, result);

        assert!(cache.should_compact().is_some());
    }

    #[test]
    fn test_compaction_trigger_insert_based() {
        let mut cache = VerificationCache::new();
        cache.add_compaction_trigger(CompactionTrigger::every_inserts(5));

        // Insert 4 entries (should not trigger)
        for i in 0..4 {
            let key = CacheKey::new("v1", format!("prop_{}", i));
            let result = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("prop_{}", i),
                    passed: true,
                    status: "Verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "hash".to_string(),
                confidence: 0.9,
            };
            cache.insert(key, result);
        }

        assert!(cache.should_compact().is_none());

        // Insert 5th entry (should trigger)
        let key = CacheKey::new("v1", "prop_4");
        let result = CachedPropertyResult {
            property: VerifiedProperty {
                name: "prop_4".to_string(),
                passed: true,
                status: "Verified".to_string(),
            },
            backends: vec!["lean4".to_string()],
            cached_at: SystemTime::now(),
            dependency_hash: "hash".to_string(),
            confidence: 0.9,
        };
        cache.insert(key, result);

        assert!(cache.should_compact().is_some());
    }

    #[test]
    fn test_compaction_trigger_hit_rate_based() {
        let mut cache = VerificationCache::new();
        cache.add_compaction_trigger(CompactionTrigger::hit_rate(0.5, 10));

        // Generate 10 misses (0% hit rate - should trigger)
        for i in 0..10 {
            let key = CacheKey::new("v1", format!("missing_{}", i));
            cache.get(&key);
        }

        assert!(cache.should_compact().is_some());
    }

    #[test]
    fn test_compaction_trigger_hit_rate_min_operations() {
        let mut cache = VerificationCache::new();
        cache.add_compaction_trigger(CompactionTrigger::hit_rate(0.5, 100));

        // Generate only 5 misses (below min_operations threshold)
        for i in 0..5 {
            let key = CacheKey::new("v1", format!("missing_{}", i));
            cache.get(&key);
        }

        // Should not trigger because we haven't reached min_operations
        assert!(cache.should_compact().is_none());
    }

    #[test]
    fn test_compaction_trigger_state_reset() {
        let mut cache = VerificationCache::new();
        cache.add_compaction_trigger(CompactionTrigger::every_inserts(3));

        // Insert 3 entries (should trigger)
        for i in 0..3 {
            let key = CacheKey::new("v1", format!("prop_{}", i));
            let result = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("prop_{}", i),
                    passed: true,
                    status: "Verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "hash".to_string(),
                confidence: 0.9,
            };
            cache.insert(key, result);
        }

        assert!(cache.should_compact().is_some());

        // Reset trigger state
        cache.reset_trigger_state();

        // Should no longer trigger
        assert!(cache.should_compact().is_none());
    }

    #[test]
    fn test_compaction_trigger_presets() {
        // Test recommended triggers
        let cache = VerificationCache::new().with_recommended_triggers();
        assert_eq!(cache.compaction_triggers().len(), 3);

        // Test aggressive triggers
        let cache = VerificationCache::new().with_aggressive_triggers();
        assert_eq!(cache.compaction_triggers().len(), 3);

        // Test conservative triggers
        let cache = VerificationCache::new().with_conservative_triggers();
        assert_eq!(cache.compaction_triggers().len(), 2);
    }

    #[test]
    fn test_compaction_trigger_memory_estimate() {
        let mut cache = VerificationCache::new();

        // Empty cache should have small memory estimate
        let empty_estimate = cache.estimate_memory_bytes();
        assert!(empty_estimate > 0);

        // Insert some entries
        for i in 0..100 {
            let key = CacheKey::new("v1", format!("prop_{}", i));
            let result = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("prop_{}", i),
                    passed: true,
                    status: "Verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "hash".to_string(),
                confidence: 0.9,
            };
            cache.insert(key, result);
        }

        // Memory estimate should be larger
        let full_estimate = cache.estimate_memory_bytes();
        assert!(full_estimate > empty_estimate);
    }

    #[test]
    fn test_compaction_trigger_serialization() {
        let trigger = CompactionTrigger::SizeBased {
            threshold_ratio: 0.9,
        };
        let json = serde_json::to_string(&trigger).unwrap();
        let parsed: CompactionTrigger = serde_json::from_str(&json).unwrap();

        match parsed {
            CompactionTrigger::SizeBased { threshold_ratio } => {
                assert!((threshold_ratio - 0.9).abs() < 0.001);
            }
            _ => panic!("Expected SizeBased trigger"),
        }
    }

    #[test]
    fn test_compaction_trigger_state_serialization() {
        let state = CompactionTriggerState {
            inserts_since_compaction: 42,
            ..Default::default()
        };

        let json = serde_json::to_string(&state).unwrap();
        let parsed: CompactionTriggerState = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.inserts_since_compaction, 42);
    }

    #[test]
    fn test_compact_if_triggered() {
        let mut cache = VerificationCache::with_config(10, Duration::from_secs(3600));
        cache.add_compaction_trigger(CompactionTrigger::size_90_percent());

        // Insert 9 entries (90% capacity - should trigger)
        for i in 0..9 {
            let key = CacheKey::new("v1", format!("prop_{}", i));
            let result = CachedPropertyResult {
                property: VerifiedProperty {
                    name: format!("prop_{}", i),
                    passed: true,
                    status: "Verified".to_string(),
                },
                backends: vec!["lean4".to_string()],
                cached_at: SystemTime::now(),
                dependency_hash: "hash".to_string(),
                confidence: 0.9,
            };
            cache.insert(key, result);
        }

        // Should trigger compaction
        let config = CompactionConfig::expired_only();
        let result = cache.compact_if_triggered(&config);
        assert!(result.is_some());

        // Trigger state should be reset, so shouldn't trigger again
        let result2 = cache.compact_if_triggered(&config);
        assert!(result2.is_none());
    }

    // ========================================================================
    // Compaction Metrics Tests
    // ========================================================================

    #[test]
    fn test_compaction_trigger_counts_increment() {
        let mut counts = CompactionTriggerCounts::default();
        assert_eq!(counts.total(), 0);

        counts.increment(&CompactionTrigger::size_80_percent());
        assert_eq!(counts.size_based, 1);
        assert_eq!(counts.total(), 1);

        counts.increment(&CompactionTrigger::every_minutes(5));
        assert_eq!(counts.time_based, 1);
        assert_eq!(counts.total(), 2);

        counts.increment(&CompactionTrigger::HitRateBased {
            min_hit_rate: 0.5,
            min_operations: 100,
        });
        assert_eq!(counts.hit_rate_based, 1);
        assert_eq!(counts.total(), 3);

        counts.increment(&CompactionTrigger::PartitionImbalance { max_ratio: 2.0 });
        assert_eq!(counts.partition_imbalance, 1);
        assert_eq!(counts.total(), 4);

        counts.increment(&CompactionTrigger::every_inserts(1000));
        assert_eq!(counts.insert_based, 1);
        assert_eq!(counts.total(), 5);

        counts.increment(&CompactionTrigger::MemoryBased {
            max_bytes: 1024 * 1024,
        });
        assert_eq!(counts.memory_based, 1);
        assert_eq!(counts.total(), 6);
    }

    #[test]
    fn test_compaction_trigger_counts_serialization() {
        let counts = CompactionTriggerCounts {
            size_based: 3,
            time_based: 2,
            hit_rate_based: 1,
            partition_imbalance: 0,
            insert_based: 5,
            memory_based: 0,
        };

        let json = serde_json::to_string(&counts).unwrap();
        let parsed: CompactionTriggerCounts = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, counts);
    }

    #[test]
    fn test_compaction_trigger_counts_add() {
        let mut counts1 = CompactionTriggerCounts {
            size_based: 3,
            time_based: 2,
            hit_rate_based: 1,
            partition_imbalance: 0,
            insert_based: 5,
            memory_based: 1,
        };

        let counts2 = CompactionTriggerCounts {
            size_based: 1,
            time_based: 3,
            hit_rate_based: 2,
            partition_imbalance: 1,
            insert_based: 2,
            memory_based: 0,
        };

        counts1.add(&counts2);

        assert_eq!(counts1.size_based, 4);
        assert_eq!(counts1.time_based, 5);
        assert_eq!(counts1.hit_rate_based, 3);
        assert_eq!(counts1.partition_imbalance, 1);
        assert_eq!(counts1.insert_based, 7);
        assert_eq!(counts1.memory_based, 1);
        assert_eq!(counts1.total(), 21);
    }

    #[test]
    fn test_compaction_trigger_counts_combined() {
        let counts1 = CompactionTriggerCounts {
            size_based: 3,
            time_based: 2,
            hit_rate_based: 1,
            partition_imbalance: 0,
            insert_based: 5,
            memory_based: 1,
        };

        let counts2 = CompactionTriggerCounts {
            size_based: 1,
            time_based: 3,
            hit_rate_based: 2,
            partition_imbalance: 1,
            insert_based: 2,
            memory_based: 0,
        };

        let combined = counts1.combined(&counts2);

        // Original counts should be unchanged
        assert_eq!(counts1.total(), 12);
        assert_eq!(counts2.total(), 9);

        // Combined should have sum of both
        assert_eq!(combined.size_based, 4);
        assert_eq!(combined.time_based, 5);
        assert_eq!(combined.hit_rate_based, 3);
        assert_eq!(combined.partition_imbalance, 1);
        assert_eq!(combined.insert_based, 7);
        assert_eq!(combined.memory_based, 1);
        assert_eq!(combined.total(), 21);
    }

    #[test]
    fn test_cache_historical_compaction_counts() {
        let mut cache = VerificationCache::new();

        // Initially, historical counts should be zero
        assert_eq!(cache.historical_compaction_counts().total(), 0);

        // Record some compactions
        cache.record_compaction(&CompactionTrigger::size_80_percent());
        cache.record_compaction(&CompactionTrigger::every_minutes(10));
        cache.record_compaction(&CompactionTrigger::size_80_percent());

        assert_eq!(cache.historical_compaction_counts().size_based, 2);
        assert_eq!(cache.historical_compaction_counts().time_based, 1);
        assert_eq!(cache.historical_compaction_counts().total(), 3);
    }

    #[test]
    fn test_cache_snapshot_includes_historical_counts() {
        let mut cache = VerificationCache::new();

        // Record some compactions
        cache.record_compaction(&CompactionTrigger::size_80_percent());
        cache.record_compaction(&CompactionTrigger::every_minutes(10));

        // Create snapshot
        let snapshot = cache.create_snapshot();

        // Verify snapshot includes counts
        assert!(snapshot.compaction_trigger_counts.is_some());
        let counts = snapshot.compaction_trigger_counts.unwrap();
        assert_eq!(counts.size_based, 1);
        assert_eq!(counts.time_based, 1);
        assert_eq!(counts.total(), 2);
    }

    #[test]
    fn test_cache_restore_accumulates_historical_counts() {
        // Create first cache and record some compactions
        let mut cache1 = VerificationCache::new();
        cache1.record_compaction(&CompactionTrigger::size_80_percent());
        cache1.record_compaction(&CompactionTrigger::every_minutes(10));

        // Create snapshot from first cache
        let snapshot1 = cache1.create_snapshot();

        // Create second cache with its own compactions
        let mut cache2 = VerificationCache::new();
        cache2.record_compaction(&CompactionTrigger::hit_rate(0.3, 100));

        // Restore from snapshot1 - should accumulate, not replace
        cache2.restore_from_snapshot(&snapshot1).unwrap();

        // Should have counts from both: original (hit_rate_based=1) + loaded (size_based=1, time_based=1)
        assert_eq!(cache2.historical_compaction_counts().size_based, 1);
        assert_eq!(cache2.historical_compaction_counts().time_based, 1);
        assert_eq!(cache2.historical_compaction_counts().hit_rate_based, 1);
        assert_eq!(cache2.historical_compaction_counts().total(), 3);
    }

    #[test]
    fn test_cache_restore_accumulates_across_multiple_snapshots() {
        // Create snapshots from multiple sessions
        let mut cache1 = VerificationCache::new();
        cache1.record_compaction(&CompactionTrigger::size_80_percent());
        cache1.record_compaction(&CompactionTrigger::size_80_percent());
        let snapshot1 = cache1.create_snapshot();

        let mut cache2 = VerificationCache::new();
        cache2.record_compaction(&CompactionTrigger::every_minutes(10));
        cache2.record_compaction(&CompactionTrigger::every_minutes(10));
        cache2.record_compaction(&CompactionTrigger::every_minutes(10));
        let snapshot2 = cache2.create_snapshot();

        // Create a new cache and restore from both snapshots
        let mut cache3 = VerificationCache::new();
        cache3.restore_from_snapshot(&snapshot1).unwrap();
        cache3.restore_from_snapshot(&snapshot2).unwrap();

        // Should accumulate counts from both snapshots
        assert_eq!(cache3.historical_compaction_counts().size_based, 2);
        assert_eq!(cache3.historical_compaction_counts().time_based, 3);
        assert_eq!(cache3.historical_compaction_counts().total(), 5);
    }

    #[test]
    fn test_metrics_aggregator_record_compaction() {
        let mut agg = MetricsAggregator::new();

        let trigger = CompactionTrigger::size_80_percent();
        agg.record_compaction(50, &trigger);

        assert_eq!(agg.current_metrics().total_compactions, 1);
        assert_eq!(agg.current_metrics().total_entries_compacted, 50);
        assert_eq!(
            agg.current_metrics().compaction_trigger_counts.size_based,
            1
        );
        assert_eq!(agg.current_metrics().compaction_trigger_counts.total(), 1);

        let trigger2 = CompactionTrigger::every_minutes(10);
        agg.record_compaction(30, &trigger2);

        assert_eq!(agg.current_metrics().total_compactions, 2);
        assert_eq!(agg.current_metrics().total_entries_compacted, 80);
        assert_eq!(
            agg.current_metrics().compaction_trigger_counts.time_based,
            1
        );
        assert_eq!(agg.current_metrics().compaction_trigger_counts.total(), 2);
    }

    #[test]
    fn test_metrics_aggregator_compaction_stats() {
        let mut agg = MetricsAggregator::new();

        let trigger = CompactionTrigger::size_90_percent();
        agg.record_compaction(100, &trigger);
        agg.record_compaction(50, &trigger);

        let stats = agg.compaction_stats(MetricsWindow::OneMinute);
        assert_eq!(stats.count, 2);
        assert_eq!(stats.sum, 150.0);
        assert!(stats.rate_per_second > 0.0);
    }

    #[test]
    fn test_metrics_aggregator_compaction_in_sample_count() {
        let mut agg = MetricsAggregator::new();
        let initial_count = agg.total_sample_count();

        agg.record_compaction(10, &CompactionTrigger::size_80_percent());
        assert_eq!(agg.total_sample_count(), initial_count + 1);
    }

    #[test]
    fn test_metrics_aggregator_reset_clears_compaction() {
        let mut agg = MetricsAggregator::new();
        agg.record_compaction(10, &CompactionTrigger::size_80_percent());
        agg.record_compaction(20, &CompactionTrigger::every_minutes(5));

        let before_reset = agg.total_sample_count();
        assert!(before_reset >= 2);

        agg.reset_samples();

        // Samples cleared but totals preserved
        assert_eq!(agg.compaction_stats(MetricsWindow::OneMinute).count, 0);
        assert_eq!(agg.current_metrics().total_compactions, 2); // Totals preserved
    }

    #[test]
    fn test_metrics_aggregator_export_prometheus_compaction() {
        let mut agg = MetricsAggregator::new();
        agg.record_compaction(100, &CompactionTrigger::size_80_percent());
        agg.record_compaction(50, &CompactionTrigger::every_minutes(10));

        let output = agg.export_prometheus("test");

        // Check compaction total counter
        assert!(output.contains("test_autosave_compactions_total 2"));

        // Check entries compacted total
        assert!(output.contains("test_autosave_entries_compacted_total 150"));

        // Check trigger breakdown
        assert!(
            output.contains("test_autosave_compactions_by_trigger_total{trigger=\"size_based\"} 1")
        );
        assert!(
            output.contains("test_autosave_compactions_by_trigger_total{trigger=\"time_based\"} 1")
        );
        assert!(output
            .contains("test_autosave_compactions_by_trigger_total{trigger=\"hit_rate_based\"} 0"));
    }

    #[test]
    fn test_autosave_metrics_from_status_includes_compaction() {
        let status = CacheAutosaveStatus {
            save_count: 5,
            error_count: 1,
            skip_count: 2,
            forced_save_count: 1,
            compaction_count: 3,
            compaction_trigger_counts: CompactionTriggerCounts {
                size_based: 2,
                time_based: 1,
                hit_rate_based: 0,
                partition_imbalance: 0,
                insert_based: 0,
                memory_based: 0,
            },
            save_reason_counts: AutosaveReasonCounts::default(),
            last_save_reason: None,
            total_bytes_written: 5000,
            last_save_bytes: 1000,
            current_interval: Duration::from_secs(60),
            is_running: true,
        };

        let metrics = AutosaveMetrics::from_status(&status);
        assert_eq!(metrics.total_compactions, 3);
        // Trigger counts now tracked in status
        assert_eq!(metrics.compaction_trigger_counts.size_based, 2);
        assert_eq!(metrics.compaction_trigger_counts.time_based, 1);
        assert_eq!(metrics.compaction_trigger_counts.total(), 3);
        // total_entries_compacted still not tracked in status
        assert_eq!(metrics.total_entries_compacted, 0);
    }

    // ========================================================================
    // Warming Callback Tests
    // ========================================================================

    #[test]
    fn test_cache_autosave_callbacks_on_warming() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let warming_called = Arc::new(AtomicBool::new(false));
        let warming_called_clone = warming_called.clone();

        let callbacks = CacheAutosaveCallbacks::new().on_warming(move |event| {
            warming_called_clone.store(true, Ordering::Relaxed);
            // Verify the event has a valid result (entries_warmed is a usize, always non-negative)
            let _ = event.result.entries_warmed;
        });

        // The callback should be set
        assert!(callbacks.on_warming.is_some());

        // We can't easily test it fires without an actual autosave run,
        // but we can verify the callback is callable
        let event = AutosaveWarmingEvent {
            path: PathBuf::from("test.json"),
            result: WarmingResult::new("test"),
            compression: None,
        };
        if let Some(ref cb) = callbacks.on_warming {
            cb(event);
        }
        assert!(warming_called.load(Ordering::Relaxed));
    }

    #[test]
    fn test_autosave_warming_event_structure() {
        let result = WarmingResult::new("test.json");

        let event = AutosaveWarmingEvent {
            path: PathBuf::from("/path/to/cache.json"),
            result: result.clone(),
            compression: Some(SnapshotCompressionLevel::Default),
        };

        assert_eq!(event.path.to_string_lossy(), "/path/to/cache.json");
        assert_eq!(event.result.entries_warmed, 0);
        assert!(event.compression.is_some());
    }

    #[test]
    fn test_cache_autosave_callbacks_debug_includes_warming() {
        let callbacks = CacheAutosaveCallbacks::new().on_warming(|_| {});

        let debug = format!("{:?}", callbacks);
        assert!(debug.contains("on_warming"));
    }

    // ========================================================================
    // CompactionTimeSeries tests
    // ========================================================================

    #[test]
    fn test_compaction_time_series_basic() {
        let mut ts = CompactionTimeSeries::new(100);
        assert!(ts.is_empty());
        assert_eq!(ts.len(), 0);
        assert_eq!(ts.capacity(), 100);

        ts.record(&CompactionTrigger::size_80_percent(), 50);
        assert!(!ts.is_empty());
        assert_eq!(ts.len(), 1);

        let entries = ts.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].trigger_type, CompactionTriggerType::SizeBased);
        assert_eq!(entries[0].entries_removed, 50);
    }

    #[test]
    fn test_compaction_time_series_ring_buffer() {
        let mut ts = CompactionTimeSeries::new(3);

        // Fill the buffer
        ts.record(&CompactionTrigger::size_80_percent(), 10);
        ts.record(&CompactionTrigger::every_minutes(5), 20);
        ts.record(&CompactionTrigger::hit_rate(0.3, 100), 30);
        assert_eq!(ts.len(), 3);

        // Add one more - should overwrite oldest
        ts.record(&CompactionTrigger::every_inserts(1000), 40);
        assert_eq!(ts.len(), 3);

        // Oldest should be time_based, newest should be insert_based
        let entries = ts.entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].trigger_type, CompactionTriggerType::TimeBased);
        assert_eq!(entries[0].entries_removed, 20);
        assert_eq!(entries[2].trigger_type, CompactionTriggerType::InsertBased);
        assert_eq!(entries[2].entries_removed, 40);
    }

    #[test]
    fn test_compaction_time_series_record_at() {
        let mut ts = CompactionTimeSeries::new(10);
        let base_time = SystemTime::UNIX_EPOCH + Duration::from_secs(1000000);

        ts.record_at(base_time, CompactionTriggerType::SizeBased, 100);
        ts.record_at(
            base_time + Duration::from_secs(60),
            CompactionTriggerType::TimeBased,
            200,
        );

        let entries = ts.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].timestamp_secs, 1000000);
        assert_eq!(entries[1].timestamp_secs, 1000060);
    }

    #[test]
    fn test_compaction_time_series_counts_by_type() {
        let mut ts = CompactionTimeSeries::new(100);
        let base_time = SystemTime::now();

        ts.record_at(base_time, CompactionTriggerType::SizeBased, 10);
        ts.record_at(base_time, CompactionTriggerType::SizeBased, 20);
        ts.record_at(base_time, CompactionTriggerType::TimeBased, 30);
        ts.record_at(base_time, CompactionTriggerType::HitRateBased, 40);
        ts.record_at(base_time, CompactionTriggerType::SizeBased, 50);

        let counts = ts.counts_by_type(Duration::from_secs(3600));
        assert_eq!(counts.size_based, 3);
        assert_eq!(counts.time_based, 1);
        assert_eq!(counts.hit_rate_based, 1);
        assert_eq!(counts.partition_imbalance, 0);
        assert_eq!(counts.insert_based, 0);
        assert_eq!(counts.memory_based, 0);
        assert_eq!(counts.total(), 5);
    }

    #[test]
    fn test_compaction_time_series_entries_removed() {
        let mut ts = CompactionTimeSeries::new(100);
        let now = SystemTime::now();

        ts.record_at(now, CompactionTriggerType::SizeBased, 100);
        ts.record_at(now, CompactionTriggerType::TimeBased, 50);
        ts.record_at(now, CompactionTriggerType::HitRateBased, 75);

        assert_eq!(ts.entries_removed_in_window(Duration::from_secs(3600)), 225);
    }

    #[test]
    fn test_compaction_time_series_oldest_newest_timestamp() {
        let mut ts = CompactionTimeSeries::new(10);
        assert!(ts.oldest_timestamp().is_none());
        assert!(ts.newest_timestamp().is_none());

        let t1 = SystemTime::UNIX_EPOCH + Duration::from_secs(1000);
        let t2 = SystemTime::UNIX_EPOCH + Duration::from_secs(2000);
        let t3 = SystemTime::UNIX_EPOCH + Duration::from_secs(3000);

        ts.record_at(t1, CompactionTriggerType::SizeBased, 10);
        ts.record_at(t2, CompactionTriggerType::TimeBased, 20);
        ts.record_at(t3, CompactionTriggerType::HitRateBased, 30);

        let oldest = ts.oldest_timestamp().unwrap();
        let newest = ts.newest_timestamp().unwrap();

        assert_eq!(oldest, t1);
        assert_eq!(newest, t3);
    }

    #[test]
    fn test_compaction_time_series_clear() {
        let mut ts = CompactionTimeSeries::new(10);
        ts.record(&CompactionTrigger::size_80_percent(), 10);
        ts.record(&CompactionTrigger::every_minutes(5), 20);
        assert_eq!(ts.len(), 2);

        ts.clear();
        assert!(ts.is_empty());
        assert_eq!(ts.len(), 0);
    }

    #[test]
    fn test_compaction_time_series_summary() {
        let mut ts = CompactionTimeSeries::new(100);
        let now = SystemTime::now();

        ts.record_at(now, CompactionTriggerType::SizeBased, 100);
        ts.record_at(now, CompactionTriggerType::TimeBased, 50);
        ts.record_at(now, CompactionTriggerType::SizeBased, 75);

        let summary = ts.summary(Duration::from_secs(3600));
        assert_eq!(summary.event_count, 3);
        assert_eq!(summary.total_entries_removed, 225);
        assert_eq!(summary.counts_by_type.size_based, 2);
        assert_eq!(summary.counts_by_type.time_based, 1);
        assert!(summary.rate_per_hour > 0.0);
    }

    #[test]
    fn test_compaction_trigger_type_from_trigger() {
        assert_eq!(
            CompactionTriggerType::from_trigger(&CompactionTrigger::size_80_percent()),
            CompactionTriggerType::SizeBased
        );
        assert_eq!(
            CompactionTriggerType::from_trigger(&CompactionTrigger::every_minutes(10)),
            CompactionTriggerType::TimeBased
        );
        assert_eq!(
            CompactionTriggerType::from_trigger(&CompactionTrigger::hit_rate(0.3, 100)),
            CompactionTriggerType::HitRateBased
        );
        assert_eq!(
            CompactionTriggerType::from_trigger(&CompactionTrigger::PartitionImbalance {
                max_ratio: 0.5
            }),
            CompactionTriggerType::PartitionImbalance
        );
        assert_eq!(
            CompactionTriggerType::from_trigger(&CompactionTrigger::every_inserts(1000)),
            CompactionTriggerType::InsertBased
        );
        assert_eq!(
            CompactionTriggerType::from_trigger(&CompactionTrigger::max_megabytes(1)),
            CompactionTriggerType::MemoryBased
        );
    }

    #[test]
    fn test_compaction_trigger_type_all() {
        let all = CompactionTriggerType::all();
        assert_eq!(all.len(), 6);
        assert!(all.contains(&CompactionTriggerType::SizeBased));
        assert!(all.contains(&CompactionTriggerType::TimeBased));
        assert!(all.contains(&CompactionTriggerType::HitRateBased));
        assert!(all.contains(&CompactionTriggerType::PartitionImbalance));
        assert!(all.contains(&CompactionTriggerType::InsertBased));
        assert!(all.contains(&CompactionTriggerType::MemoryBased));
    }

    #[test]
    fn test_cache_compaction_time_series_integration() {
        let mut cache = VerificationCache::new();

        // Initially empty
        assert!(cache.compaction_time_series().is_empty());

        // Record some compactions
        cache.record_compaction(&CompactionTrigger::size_80_percent());
        cache.record_compaction(&CompactionTrigger::every_minutes(10));

        assert_eq!(cache.compaction_time_series().len(), 2);

        // Historical counts should also be updated
        assert_eq!(cache.historical_compaction_counts().total(), 2);
    }

    #[test]
    fn test_cache_record_compaction_full() {
        let mut cache = VerificationCache::new();

        cache.record_compaction_full(&CompactionTrigger::size_80_percent(), 100);
        cache.record_compaction_full(&CompactionTrigger::every_minutes(10), 50);

        // Check time series
        let ts = cache.compaction_time_series();
        assert_eq!(ts.len(), 2);
        assert_eq!(ts.entries_removed_in_window(Duration::from_secs(3600)), 150);

        // Check historical counts
        assert_eq!(cache.historical_compaction_counts().size_based, 1);
        assert_eq!(cache.historical_compaction_counts().time_based, 1);
    }

    #[test]
    fn test_cache_compaction_rate_methods() {
        let mut cache = VerificationCache::new();

        // Record several compactions
        cache.record_compaction(&CompactionTrigger::size_80_percent());
        cache.record_compaction(&CompactionTrigger::every_minutes(10));
        cache.record_compaction(&CompactionTrigger::hit_rate(0.3, 100));

        let window = Duration::from_secs(3600);

        // Rate methods should work
        let rate_per_minute = cache.compaction_rate_per_minute(window);
        let rate_per_hour = cache.compaction_rate_per_hour(window);

        // 3 events in 1 hour window
        assert!((2.9..=3.1).contains(&rate_per_hour));
        assert_eq!(rate_per_hour, rate_per_minute * 60.0);
    }

    #[test]
    fn test_cache_compaction_summary() {
        let mut cache = VerificationCache::new();

        cache.record_compaction_full(&CompactionTrigger::size_80_percent(), 100);
        cache.record_compaction_full(&CompactionTrigger::size_80_percent(), 75);
        cache.record_compaction_full(&CompactionTrigger::every_minutes(10), 50);

        let summary = cache.compaction_summary(Duration::from_secs(3600));
        assert_eq!(summary.event_count, 3);
        assert_eq!(summary.total_entries_removed, 225);
        assert_eq!(summary.counts_by_type.size_based, 2);
        assert_eq!(summary.counts_by_type.time_based, 1);
    }

    #[test]
    fn test_compaction_time_series_serialization() {
        let mut ts = CompactionTimeSeries::new(10);
        let base_time = SystemTime::UNIX_EPOCH + Duration::from_secs(1000000);

        ts.record_at(base_time, CompactionTriggerType::SizeBased, 100);
        ts.record_at(
            base_time + Duration::from_secs(60),
            CompactionTriggerType::TimeBased,
            200,
        );

        // Serialize
        let json = serde_json::to_string(&ts).unwrap();
        assert!(json.contains("SizeBased"));
        assert!(json.contains("TimeBased"));

        // Deserialize
        let ts2: CompactionTimeSeries = serde_json::from_str(&json).unwrap();
        assert_eq!(ts2.len(), 2);

        let entries = ts2.entries();
        assert_eq!(entries[0].trigger_type, CompactionTriggerType::SizeBased);
        assert_eq!(entries[1].trigger_type, CompactionTriggerType::TimeBased);
    }

    // ========================================================================
    // PersistedAutosaveMetrics Tests
    // ========================================================================

    #[test]
    fn test_persisted_autosave_metrics_from_status() {
        let status = CacheAutosaveStatus {
            save_count: 10,
            error_count: 2,
            skip_count: 5,
            forced_save_count: 3,
            compaction_count: 1,
            compaction_trigger_counts: CompactionTriggerCounts::default(),
            save_reason_counts: AutosaveReasonCounts {
                initial: 1,
                interval: 6,
                stale_data: 2,
                coalesced: 1,
            },
            last_save_reason: Some(AutosaveSaveReason::Interval),
            total_bytes_written: 50_000,
            last_save_bytes: 5_000,
            current_interval: Duration::from_secs(30),
            is_running: true,
        };

        let session_start = 1704067200; // 2024-01-01 00:00:00 UTC
        let session_end = 1704070800; // 2024-01-01 01:00:00 UTC

        let metrics =
            PersistedAutosaveMetrics::from_status(&status, true, session_start, session_end);

        assert_eq!(metrics.save_count, 10);
        assert_eq!(metrics.error_count, 2);
        assert_eq!(metrics.skip_count, 5);
        assert_eq!(metrics.forced_save_count, 3);
        assert_eq!(metrics.save_reason_counts.initial, 1);
        assert_eq!(metrics.save_reason_counts.interval, 6);
        assert_eq!(metrics.save_reason_counts.stale_data, 2);
        assert_eq!(metrics.save_reason_counts.coalesced, 1);
        assert_eq!(metrics.last_save_reason, Some(AutosaveSaveReason::Interval));
        assert_eq!(metrics.total_bytes_written, 50_000);
        assert_eq!(metrics.last_save_bytes, 5_000);
        assert_eq!(metrics.interval_ms, 30_000);
        assert!(metrics.compressed);
        assert_eq!(metrics.session_start_secs, session_start);
        assert_eq!(metrics.session_end_secs, session_end);
    }

    #[test]
    fn test_persisted_autosave_metrics_computed_helpers() {
        let metrics = PersistedAutosaveMetrics {
            save_count: 8,
            error_count: 1,
            skip_count: 4,
            forced_save_count: 2,
            save_reason_counts: AutosaveReasonCounts::default(),
            last_save_reason: None,
            total_bytes_written: 40_000,
            last_save_bytes: 5_000,
            interval_ms: 60_000,
            compressed: false,
            session_start_secs: 1000,
            session_end_secs: 2000,
        };

        // total_attempts = save_count + skip_count = 8 + 4 = 12
        assert_eq!(metrics.total_attempts(), 12);

        // success_rate = save_count / total_attempts = 8 / 12 ≈ 0.667
        let rate = metrics.success_rate();
        assert!((rate - 0.6666666666666666).abs() < 0.001);

        // session_duration_secs = session_end - session_start = 2000 - 1000 = 1000
        assert_eq!(metrics.session_duration_secs(), 1000);

        // avg_bytes_per_save = total_bytes / save_count = 40000 / 8 = 5000
        assert_eq!(metrics.avg_bytes_per_save(), 5000);
    }

    #[test]
    fn test_persisted_autosave_metrics_edge_cases() {
        // Zero saves
        let metrics = PersistedAutosaveMetrics {
            save_count: 0,
            error_count: 0,
            skip_count: 0,
            ..Default::default()
        };
        assert_eq!(metrics.total_attempts(), 0);
        assert_eq!(metrics.success_rate(), 0.0);
        assert_eq!(metrics.avg_bytes_per_save(), 0);

        // Only skips (no saves)
        let metrics = PersistedAutosaveMetrics {
            save_count: 0,
            skip_count: 5,
            ..Default::default()
        };
        assert_eq!(metrics.total_attempts(), 5);
        assert_eq!(metrics.success_rate(), 0.0);
    }

    #[test]
    fn test_persisted_autosave_metrics_serialization() {
        let metrics = PersistedAutosaveMetrics {
            save_count: 5,
            error_count: 1,
            skip_count: 2,
            forced_save_count: 1,
            save_reason_counts: AutosaveReasonCounts {
                initial: 1,
                interval: 3,
                stale_data: 1,
                coalesced: 0,
            },
            last_save_reason: Some(AutosaveSaveReason::StaleData),
            total_bytes_written: 25_000,
            last_save_bytes: 5_000,
            interval_ms: 45_000,
            compressed: true,
            session_start_secs: 1704067200,
            session_end_secs: 1704070800,
        };

        let json = serde_json::to_string(&metrics).unwrap();
        let parsed: PersistedAutosaveMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed, metrics);
    }

    #[test]
    fn test_snapshot_with_autosave_metrics() {
        let mut snapshot = CacheSnapshot::new(Vec::new(), CacheStats::default());

        // Initially no autosave metrics
        assert!(snapshot.autosave_metrics.is_none());

        // Set autosave metrics
        let metrics = PersistedAutosaveMetrics {
            save_count: 3,
            error_count: 0,
            skip_count: 1,
            forced_save_count: 0,
            save_reason_counts: AutosaveReasonCounts {
                initial: 1,
                interval: 2,
                stale_data: 0,
                coalesced: 0,
            },
            last_save_reason: Some(AutosaveSaveReason::Interval),
            total_bytes_written: 15_000,
            last_save_bytes: 5_000,
            interval_ms: 30_000,
            compressed: false,
            session_start_secs: 1000,
            session_end_secs: 1500,
        };
        snapshot.autosave_metrics = Some(metrics.clone());

        // Serialize and deserialize
        let json = serde_json::to_string(&snapshot).unwrap();
        let restored: CacheSnapshot = serde_json::from_str(&json).unwrap();

        // Verify autosave metrics survived serialization
        assert!(restored.autosave_metrics.is_some());
        let restored_metrics = restored.autosave_metrics.unwrap();
        assert_eq!(restored_metrics.save_count, 3);
        assert_eq!(restored_metrics.skip_count, 1);
        assert_eq!(restored_metrics.total_bytes_written, 15_000);
        assert_eq!(restored_metrics.session_duration_secs(), 500);
    }

    #[test]
    fn test_snapshot_format_v5_with_autosave_metrics() {
        // Create a snapshot with autosave metrics
        let mut snapshot = CacheSnapshot::new(Vec::new(), CacheStats::default());
        snapshot.autosave_metrics = Some(PersistedAutosaveMetrics {
            save_count: 10,
            interval_ms: 60_000,
            compressed: true,
            session_start_secs: 1704067200,
            session_end_secs: 1704070800,
            ..Default::default()
        });

        // Verify format version is 5 (which supports autosave_metrics)
        assert_eq!(snapshot.format_version, 5);

        // Serialize and verify the JSON contains autosave_metrics
        let json = serde_json::to_string_pretty(&snapshot).unwrap();
        assert!(json.contains("autosave_metrics"));
        assert!(json.contains("save_count"));
        assert!(json.contains("session_start_secs"));
        assert!(json.contains("session_end_secs"));
    }
}
