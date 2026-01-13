//! Result types for incremental BMC verification

use std::time::Duration;

/// Result of an incremental BMC verification
#[derive(Debug, Clone)]
pub struct IncrementalResult {
    /// Whether the property was proven
    pub proven: bool,
    /// Total verification time
    pub duration: Duration,
    /// Cache statistics
    pub cache_stats: CacheStats,
    /// Number of clauses used from cache
    pub clauses_from_cache: u64,
    /// Number of new clauses learned
    pub clauses_learned: u64,
    /// Number of functions verified
    pub functions_verified: u64,
    /// Number of functions that used cached results
    pub functions_cached: u64,
    /// Whether the result was fully from cache
    pub from_cache: bool,
    /// Any counterexample found
    pub counterexample: Option<String>,
}

impl IncrementalResult {
    /// Create a proven result
    pub fn proven(duration: Duration, cache_stats: CacheStats) -> Self {
        Self {
            proven: true,
            duration,
            cache_stats,
            clauses_from_cache: 0,
            clauses_learned: 0,
            functions_verified: 0,
            functions_cached: 0,
            from_cache: false,
            counterexample: None,
        }
    }

    /// Create a disproven result with counterexample
    pub fn disproven(duration: Duration, counterexample: String, cache_stats: CacheStats) -> Self {
        Self {
            proven: false,
            duration,
            cache_stats,
            clauses_from_cache: 0,
            clauses_learned: 0,
            functions_verified: 0,
            functions_cached: 0,
            from_cache: false,
            counterexample: Some(counterexample),
        }
    }

    /// Create a result from cache
    pub fn from_cache(proven: bool, duration: Duration, cache_stats: CacheStats) -> Self {
        Self {
            proven,
            duration,
            cache_stats,
            clauses_from_cache: 0,
            clauses_learned: 0,
            functions_verified: 0,
            functions_cached: 0,
            from_cache: true,
            counterexample: None,
        }
    }

    /// Set clause statistics
    pub fn with_clause_stats(mut self, from_cache: u64, learned: u64) -> Self {
        self.clauses_from_cache = from_cache;
        self.clauses_learned = learned;
        self
    }

    /// Set function statistics
    pub fn with_function_stats(mut self, verified: u64, cached: u64) -> Self {
        self.functions_verified = verified;
        self.functions_cached = cached;
        self
    }

    /// Calculate speedup compared to full verification
    pub fn speedup(&self, full_verification_time: Duration) -> f64 {
        if self.duration.is_zero() {
            return f64::INFINITY;
        }
        full_verification_time.as_secs_f64() / self.duration.as_secs_f64()
    }

    /// Calculate cache hit rate for clauses
    pub fn clause_cache_hit_rate(&self) -> f64 {
        let total = self.clauses_from_cache + self.clauses_learned;
        if total == 0 {
            return 0.0;
        }
        self.clauses_from_cache as f64 / total as f64
    }

    /// Calculate cache hit rate for functions
    pub fn function_cache_hit_rate(&self) -> f64 {
        let total = self.functions_verified + self.functions_cached;
        if total == 0 {
            return 0.0;
        }
        self.functions_cached as f64 / total as f64
    }
}

impl std::fmt::Display for IncrementalResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.proven {
            write!(f, "PROVEN")?;
        } else {
            write!(f, "DISPROVEN")?;
        }

        write!(f, " in {:?}", self.duration)?;

        if self.from_cache {
            write!(f, " (cached)")?;
        } else {
            write!(
                f,
                " ({} clauses from cache, {} learned)",
                self.clauses_from_cache, self.clauses_learned
            )?;
        }

        Ok(())
    }
}

/// Statistics about cache usage
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total clauses in database
    pub total_clauses: u64,
    /// Clauses loaded for this verification
    pub clauses_loaded: u64,
    /// Clauses that were still valid
    pub clauses_valid: u64,
    /// Clauses that were invalidated
    pub clauses_invalidated: u64,
    /// Time spent loading clauses
    pub load_time: Duration,
    /// Time spent storing clauses
    pub store_time: Duration,
    /// Database size in bytes
    pub database_size: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
}

impl CacheStats {
    /// Create new cache stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / total as f64
    }

    /// Calculate clause validity rate
    pub fn validity_rate(&self) -> f64 {
        if self.clauses_loaded == 0 {
            return 0.0;
        }
        self.clauses_valid as f64 / self.clauses_loaded as f64
    }

    /// Accumulate stats from another instance
    pub fn accumulate(&mut self, other: &CacheStats) {
        self.total_clauses = self.total_clauses.max(other.total_clauses);
        self.clauses_loaded += other.clauses_loaded;
        self.clauses_valid += other.clauses_valid;
        self.clauses_invalidated += other.clauses_invalidated;
        self.load_time += other.load_time;
        self.store_time += other.store_time;
        self.database_size = self.database_size.max(other.database_size);
        self.cache_hits += other.cache_hits;
        self.cache_misses += other.cache_misses;
    }
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cache: {} total, {} loaded ({:.1}% valid), {:.1}% hit rate, {} bytes",
            self.total_clauses,
            self.clauses_loaded,
            self.validity_rate() * 100.0,
            self.hit_rate() * 100.0,
            self.database_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== IncrementalResult Constructor Tests ====================

    #[test]
    fn test_proven_result() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new());

        assert!(result.proven);
        assert!(!result.from_cache);
        assert!(result.counterexample.is_none());
    }

    #[test]
    fn test_proven_result_defaults() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new());

        assert_eq!(result.clauses_from_cache, 0);
        assert_eq!(result.clauses_learned, 0);
        assert_eq!(result.functions_verified, 0);
        assert_eq!(result.functions_cached, 0);
    }

    #[test]
    fn test_disproven_result() {
        let result = IncrementalResult::disproven(
            Duration::from_millis(50),
            "x = 5".to_string(),
            CacheStats::new(),
        );

        assert!(!result.proven);
        assert!(result.counterexample.is_some());
        assert_eq!(result.counterexample.unwrap(), "x = 5");
    }

    #[test]
    fn test_disproven_result_not_from_cache() {
        let result = IncrementalResult::disproven(
            Duration::from_millis(50),
            "ce".to_string(),
            CacheStats::new(),
        );

        assert!(!result.from_cache);
    }

    #[test]
    fn test_from_cache_result() {
        let result =
            IncrementalResult::from_cache(true, Duration::from_millis(1), CacheStats::new());

        assert!(result.proven);
        assert!(result.from_cache);
    }

    #[test]
    fn test_from_cache_disproven() {
        let result =
            IncrementalResult::from_cache(false, Duration::from_millis(1), CacheStats::new());

        assert!(!result.proven);
        assert!(result.from_cache);
        assert!(result.counterexample.is_none());
    }

    // ==================== Builder Methods Tests ====================

    #[test]
    fn test_with_clause_stats() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_clause_stats(80, 20);

        assert_eq!(result.clauses_from_cache, 80);
        assert_eq!(result.clauses_learned, 20);
    }

    #[test]
    fn test_with_clause_stats_zero() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_clause_stats(0, 0);

        assert_eq!(result.clauses_from_cache, 0);
        assert_eq!(result.clauses_learned, 0);
    }

    #[test]
    fn test_with_function_stats() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_function_stats(5, 15);

        assert_eq!(result.functions_verified, 5);
        assert_eq!(result.functions_cached, 15);
    }

    #[test]
    fn test_chained_builder() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_clause_stats(100, 50)
            .with_function_stats(10, 20);

        assert_eq!(result.clauses_from_cache, 100);
        assert_eq!(result.clauses_learned, 50);
        assert_eq!(result.functions_verified, 10);
        assert_eq!(result.functions_cached, 20);
    }

    // ==================== Speedup Tests ====================

    #[test]
    fn test_speedup() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new());

        let speedup = result.speedup(Duration::from_secs(1));
        assert!((speedup - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_speedup_same_duration() {
        let result = IncrementalResult::proven(Duration::from_secs(1), CacheStats::new());

        let speedup = result.speedup(Duration::from_secs(1));
        assert!((speedup - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_speedup_faster() {
        let result = IncrementalResult::proven(Duration::from_millis(10), CacheStats::new());

        let speedup = result.speedup(Duration::from_secs(1));
        assert!((speedup - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_speedup_zero_duration_infinity() {
        let result = IncrementalResult::proven(Duration::ZERO, CacheStats::new());

        let speedup = result.speedup(Duration::from_secs(1));
        assert!(speedup.is_infinite());
    }

    // ==================== Cache Hit Rate Tests ====================

    #[test]
    fn test_cache_hit_rate() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_clause_stats(80, 20);

        let hit_rate = result.clause_cache_hit_rate();
        assert!((hit_rate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_clause_cache_hit_rate_all_cached() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_clause_stats(100, 0);

        let hit_rate = result.clause_cache_hit_rate();
        assert!((hit_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_clause_cache_hit_rate_none_cached() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_clause_stats(0, 100);

        let hit_rate = result.clause_cache_hit_rate();
        assert!((hit_rate - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_clause_cache_hit_rate_zero_total() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_clause_stats(0, 0);

        let hit_rate = result.clause_cache_hit_rate();
        assert!((hit_rate - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_function_cache_hit_rate() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_function_stats(5, 15);

        let hit_rate = result.function_cache_hit_rate();
        assert!((hit_rate - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_function_cache_hit_rate_all_cached() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_function_stats(0, 100);

        let hit_rate = result.function_cache_hit_rate();
        assert!((hit_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_function_cache_hit_rate_none_cached() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_function_stats(100, 0);

        let hit_rate = result.function_cache_hit_rate();
        assert!((hit_rate - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_function_cache_hit_rate_zero_total() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new());

        let hit_rate = result.function_cache_hit_rate();
        assert!((hit_rate - 0.0).abs() < 0.001);
    }

    // ==================== Display Tests ====================

    #[test]
    fn test_display_proven() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_clause_stats(80, 20);

        let display = format!("{}", result);
        assert!(display.contains("PROVEN"));
        assert!(display.contains("80 clauses from cache"));
        assert!(display.contains("20 learned"));
    }

    #[test]
    fn test_display_disproven() {
        let result = IncrementalResult::disproven(
            Duration::from_millis(100),
            "x=5".to_string(),
            CacheStats::new(),
        );

        let display = format!("{}", result);
        assert!(display.contains("DISPROVEN"));
    }

    #[test]
    fn test_display_from_cache() {
        let result =
            IncrementalResult::from_cache(true, Duration::from_millis(1), CacheStats::new());

        let display = format!("{}", result);
        assert!(display.contains("(cached)"));
    }

    #[test]
    fn test_display_includes_duration() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new());

        let display = format!("{}", result);
        assert!(display.contains("100"));
    }

    // ==================== Debug and Clone Tests ====================

    #[test]
    fn test_incremental_result_debug() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new());
        let debug = format!("{:?}", result);
        assert!(debug.contains("IncrementalResult"));
        assert!(debug.contains("proven"));
    }

    #[test]
    fn test_incremental_result_clone() {
        let result = IncrementalResult::proven(Duration::from_millis(100), CacheStats::new())
            .with_clause_stats(50, 50);
        let cloned = result.clone();

        assert_eq!(cloned.proven, result.proven);
        assert_eq!(cloned.duration, result.duration);
        assert_eq!(cloned.clauses_from_cache, result.clauses_from_cache);
    }

    // ==================== CacheStats Tests ====================

    #[test]
    fn test_cache_stats_new() {
        let stats = CacheStats::new();
        assert_eq!(stats.total_clauses, 0);
        assert_eq!(stats.clauses_loaded, 0);
        assert_eq!(stats.cache_hits, 0);
    }

    #[test]
    fn test_cache_stats_default() {
        let stats = CacheStats::default();
        assert_eq!(stats.total_clauses, 0);
        assert_eq!(stats.clauses_loaded, 0);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            cache_hits: 80,
            cache_misses: 20,
            ..Default::default()
        };

        let hit_rate = stats.hit_rate();
        assert!((hit_rate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_hit_rate_all_hits() {
        let stats = CacheStats {
            cache_hits: 100,
            cache_misses: 0,
            ..Default::default()
        };

        let hit_rate = stats.hit_rate();
        assert!((hit_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_hit_rate_all_misses() {
        let stats = CacheStats {
            cache_hits: 0,
            cache_misses: 100,
            ..Default::default()
        };

        let hit_rate = stats.hit_rate();
        assert!((hit_rate - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_hit_rate_zero() {
        let stats = CacheStats::new();
        let hit_rate = stats.hit_rate();
        assert!((hit_rate - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_validity_rate() {
        let stats = CacheStats {
            clauses_loaded: 100,
            clauses_valid: 90,
            ..Default::default()
        };

        let validity_rate = stats.validity_rate();
        assert!((validity_rate - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_validity_rate_all_valid() {
        let stats = CacheStats {
            clauses_loaded: 100,
            clauses_valid: 100,
            ..Default::default()
        };

        let validity_rate = stats.validity_rate();
        assert!((validity_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_validity_rate_none_valid() {
        let stats = CacheStats {
            clauses_loaded: 100,
            clauses_valid: 0,
            ..Default::default()
        };

        let validity_rate = stats.validity_rate();
        assert!((validity_rate - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_validity_rate_zero_loaded() {
        let stats = CacheStats::new();
        let validity_rate = stats.validity_rate();
        assert!((validity_rate - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_accumulate() {
        let mut stats1 = CacheStats {
            cache_hits: 10,
            cache_misses: 5,
            clauses_loaded: 100,
            ..Default::default()
        };

        let stats2 = CacheStats {
            cache_hits: 20,
            cache_misses: 10,
            clauses_loaded: 50,
            ..Default::default()
        };

        stats1.accumulate(&stats2);

        assert_eq!(stats1.cache_hits, 30);
        assert_eq!(stats1.cache_misses, 15);
        assert_eq!(stats1.clauses_loaded, 150);
    }

    #[test]
    fn test_cache_stats_accumulate_total_clauses_max() {
        let mut stats1 = CacheStats {
            total_clauses: 1000,
            ..Default::default()
        };

        let stats2 = CacheStats {
            total_clauses: 500,
            ..Default::default()
        };

        stats1.accumulate(&stats2);
        // Should take max
        assert_eq!(stats1.total_clauses, 1000);
    }

    #[test]
    fn test_cache_stats_accumulate_database_size_max() {
        let mut stats1 = CacheStats {
            database_size: 1024,
            ..Default::default()
        };

        let stats2 = CacheStats {
            database_size: 2048,
            ..Default::default()
        };

        stats1.accumulate(&stats2);
        // Should take max
        assert_eq!(stats1.database_size, 2048);
    }

    #[test]
    fn test_cache_stats_accumulate_durations() {
        let mut stats1 = CacheStats {
            load_time: Duration::from_millis(100),
            store_time: Duration::from_millis(50),
            ..Default::default()
        };

        let stats2 = CacheStats {
            load_time: Duration::from_millis(200),
            store_time: Duration::from_millis(75),
            ..Default::default()
        };

        stats1.accumulate(&stats2);

        assert_eq!(stats1.load_time, Duration::from_millis(300));
        assert_eq!(stats1.store_time, Duration::from_millis(125));
    }

    #[test]
    fn test_cache_stats_accumulate_clauses_invalidated() {
        let mut stats1 = CacheStats {
            clauses_valid: 90,
            clauses_invalidated: 10,
            ..Default::default()
        };

        let stats2 = CacheStats {
            clauses_valid: 80,
            clauses_invalidated: 20,
            ..Default::default()
        };

        stats1.accumulate(&stats2);

        assert_eq!(stats1.clauses_valid, 170);
        assert_eq!(stats1.clauses_invalidated, 30);
    }

    #[test]
    fn test_cache_stats_display() {
        let stats = CacheStats {
            total_clauses: 1000,
            clauses_loaded: 100,
            clauses_valid: 90,
            cache_hits: 80,
            cache_misses: 20,
            database_size: 1024,
            ..Default::default()
        };

        let display = format!("{}", stats);
        assert!(display.contains("1000 total"));
        assert!(display.contains("100 loaded"));
        assert!(display.contains("1024 bytes"));
    }

    #[test]
    fn test_cache_stats_display_percentages() {
        let stats = CacheStats {
            total_clauses: 1000,
            clauses_loaded: 100,
            clauses_valid: 50, // 50% valid
            cache_hits: 75,
            cache_misses: 25, // 75% hit rate
            database_size: 1024,
            ..Default::default()
        };

        let display = format!("{}", stats);
        assert!(display.contains("50.0%")); // validity rate
        assert!(display.contains("75.0%")); // hit rate
    }

    #[test]
    fn test_cache_stats_debug() {
        let stats = CacheStats::new();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("CacheStats"));
    }

    #[test]
    fn test_cache_stats_clone() {
        let stats = CacheStats {
            total_clauses: 100,
            cache_hits: 50,
            ..Default::default()
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_clauses, stats.total_clauses);
        assert_eq!(cloned.cache_hits, stats.cache_hits);
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_full_result_workflow() {
        // Simulate a full verification workflow
        let mut cache_stats = CacheStats {
            total_clauses: 1000,
            clauses_loaded: 500,
            clauses_valid: 450,
            clauses_invalidated: 50,
            cache_hits: 400,
            cache_misses: 100,
            load_time: Duration::from_millis(50),
            store_time: Duration::from_millis(25),
            database_size: 10240,
        };

        let result = IncrementalResult::proven(Duration::from_millis(500), cache_stats.clone())
            .with_clause_stats(300, 100)
            .with_function_stats(10, 5);

        // Verify all stats are preserved
        assert!(result.proven);
        assert_eq!(result.clauses_from_cache, 300);
        assert_eq!(result.clauses_learned, 100);
        assert_eq!(result.functions_verified, 10);
        assert_eq!(result.functions_cached, 5);

        // Verify rates
        assert!((result.clause_cache_hit_rate() - 0.75).abs() < 0.01);
        assert!((result.function_cache_hit_rate() - 0.333).abs() < 0.01);

        // Accumulate more stats
        let more_stats = CacheStats {
            cache_hits: 100,
            cache_misses: 50,
            clauses_loaded: 200,
            clauses_valid: 180,
            ..Default::default()
        };
        cache_stats.accumulate(&more_stats);

        assert_eq!(cache_stats.cache_hits, 500);
        assert_eq!(cache_stats.clauses_loaded, 700);
    }

    #[test]
    fn test_large_values() {
        let result = IncrementalResult::proven(Duration::from_secs(3600), CacheStats::new())
            .with_clause_stats(1_000_000, 500_000)
            .with_function_stats(10_000, 5_000);

        assert_eq!(result.clauses_from_cache, 1_000_000);
        assert_eq!(result.functions_verified, 10_000);
        assert!((result.clause_cache_hit_rate() - 0.666).abs() < 0.01);
    }
}
