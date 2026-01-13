//! Configuration for incremental BMC

use std::path::PathBuf;
use std::time::Duration;

/// Configuration for incremental BMC verification
#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    /// Path to the clause database
    pub database_path: PathBuf,

    /// Maximum age of cached clauses before invalidation
    pub max_clause_age: Duration,

    /// Maximum number of clauses to store per function
    pub max_clauses_per_function: usize,

    /// Maximum total database size in bytes (0 = unlimited)
    pub max_database_size: u64,

    /// Whether to use content-addressable storage
    pub content_addressable: bool,

    /// Whether to track clause provenance for debugging
    pub track_provenance: bool,

    /// Whether to enable watch mode
    pub watch_mode: bool,

    /// Debounce duration for watch mode file changes
    pub watch_debounce: Duration,

    /// Minimum clause size to store (smaller clauses are recomputed)
    pub min_clause_size: usize,

    /// Maximum clause size to store (very large clauses are discarded)
    pub max_clause_size: usize,

    /// Timeout for solver invocations
    pub solver_timeout: Duration,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            database_path: PathBuf::from(".kani_fast_cache.db"),
            max_clause_age: Duration::from_secs(7 * 24 * 60 * 60), // 1 week
            max_clauses_per_function: 100_000,
            max_database_size: 100 * 1024 * 1024, // 100 MB
            content_addressable: true,
            track_provenance: false,
            watch_mode: false,
            watch_debounce: Duration::from_millis(500),
            min_clause_size: 3,   // Don't store binary or unit clauses
            max_clause_size: 100, // Don't store very large clauses
            solver_timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Builder for IncrementalConfig
#[derive(Debug, Default)]
pub struct IncrementalConfigBuilder {
    config: IncrementalConfig,
}

impl IncrementalConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the database path
    pub fn database_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.database_path = path.into();
        self
    }

    /// Set the maximum clause age
    pub fn max_clause_age(mut self, age: Duration) -> Self {
        self.config.max_clause_age = age;
        self
    }

    /// Set the maximum clauses per function
    pub fn max_clauses_per_function(mut self, count: usize) -> Self {
        self.config.max_clauses_per_function = count;
        self
    }

    /// Set the maximum database size
    pub fn max_database_size(mut self, size: u64) -> Self {
        self.config.max_database_size = size;
        self
    }

    /// Enable or disable content-addressable storage
    pub fn content_addressable(mut self, enabled: bool) -> Self {
        self.config.content_addressable = enabled;
        self
    }

    /// Enable or disable provenance tracking
    pub fn track_provenance(mut self, enabled: bool) -> Self {
        self.config.track_provenance = enabled;
        self
    }

    /// Enable or disable watch mode
    pub fn watch_mode(mut self, enabled: bool) -> Self {
        self.config.watch_mode = enabled;
        self
    }

    /// Set the watch debounce duration
    pub fn watch_debounce(mut self, duration: Duration) -> Self {
        self.config.watch_debounce = duration;
        self
    }

    /// Set the minimum clause size to store
    pub fn min_clause_size(mut self, size: usize) -> Self {
        self.config.min_clause_size = size;
        self
    }

    /// Set the maximum clause size to store
    pub fn max_clause_size(mut self, size: usize) -> Self {
        self.config.max_clause_size = size;
        self
    }

    /// Set the solver timeout
    pub fn solver_timeout(mut self, timeout: Duration) -> Self {
        self.config.solver_timeout = timeout;
        self
    }

    /// Build the configuration
    pub fn build(self) -> IncrementalConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = IncrementalConfig::default();
        assert!(!config.database_path.as_os_str().is_empty());
        assert!(config.max_clause_age > Duration::ZERO);
        assert!(config.max_clauses_per_function > 0);
    }

    #[test]
    fn test_default_config_all_fields() {
        let config = IncrementalConfig::default();

        // Check all default values
        assert_eq!(config.database_path, PathBuf::from(".kani_fast_cache.db"));
        assert_eq!(config.max_clause_age, Duration::from_secs(7 * 24 * 60 * 60)); // 1 week
        assert_eq!(config.max_clauses_per_function, 100_000);
        assert_eq!(config.max_database_size, 100 * 1024 * 1024); // 100 MB
        assert!(config.content_addressable);
        assert!(!config.track_provenance);
        assert!(!config.watch_mode);
        assert_eq!(config.watch_debounce, Duration::from_millis(500));
        assert_eq!(config.min_clause_size, 3);
        assert_eq!(config.max_clause_size, 100);
        assert_eq!(config.solver_timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_builder() {
        let config = IncrementalConfigBuilder::new()
            .database_path("/tmp/test.db")
            .max_clause_age(Duration::from_secs(3600))
            .max_clauses_per_function(1000)
            .track_provenance(true)
            .build();

        assert_eq!(config.database_path, PathBuf::from("/tmp/test.db"));
        assert_eq!(config.max_clause_age, Duration::from_secs(3600));
        assert_eq!(config.max_clauses_per_function, 1000);
        assert!(config.track_provenance);
    }

    #[test]
    fn test_builder_new_returns_default() {
        let builder = IncrementalConfigBuilder::new();
        let config = builder.build();

        // Should match defaults
        assert_eq!(
            config.database_path,
            IncrementalConfig::default().database_path
        );
        assert_eq!(
            config.max_clause_age,
            IncrementalConfig::default().max_clause_age
        );
    }

    #[test]
    fn test_builder_default_trait() {
        let builder = IncrementalConfigBuilder::default();
        let config = builder.build();

        // Should also match defaults
        assert_eq!(config.max_clauses_per_function, 100_000);
    }

    #[test]
    fn test_builder_database_path() {
        let config = IncrementalConfigBuilder::new()
            .database_path("custom/path/cache.db")
            .build();

        assert_eq!(config.database_path, PathBuf::from("custom/path/cache.db"));
    }

    #[test]
    fn test_builder_database_path_from_pathbuf() {
        let path = PathBuf::from("/absolute/path/db.sqlite");
        let config = IncrementalConfigBuilder::new()
            .database_path(path.clone())
            .build();

        assert_eq!(config.database_path, path);
    }

    #[test]
    fn test_builder_max_clause_age() {
        let config = IncrementalConfigBuilder::new()
            .max_clause_age(Duration::from_secs(86400)) // 1 day
            .build();

        assert_eq!(config.max_clause_age, Duration::from_secs(86400));
    }

    #[test]
    fn test_builder_max_clause_age_zero() {
        let config = IncrementalConfigBuilder::new()
            .max_clause_age(Duration::ZERO)
            .build();

        assert_eq!(config.max_clause_age, Duration::ZERO);
    }

    #[test]
    fn test_builder_max_clauses_per_function() {
        let config = IncrementalConfigBuilder::new()
            .max_clauses_per_function(500)
            .build();

        assert_eq!(config.max_clauses_per_function, 500);
    }

    #[test]
    fn test_builder_max_database_size() {
        let config = IncrementalConfigBuilder::new()
            .max_database_size(1024 * 1024 * 1024) // 1 GB
            .build();

        assert_eq!(config.max_database_size, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_builder_max_database_size_zero() {
        let config = IncrementalConfigBuilder::new()
            .max_database_size(0) // unlimited
            .build();

        assert_eq!(config.max_database_size, 0);
    }

    #[test]
    fn test_builder_content_addressable_true() {
        let config = IncrementalConfigBuilder::new()
            .content_addressable(true)
            .build();

        assert!(config.content_addressable);
    }

    #[test]
    fn test_builder_content_addressable_false() {
        let config = IncrementalConfigBuilder::new()
            .content_addressable(false)
            .build();

        assert!(!config.content_addressable);
    }

    #[test]
    fn test_builder_track_provenance_true() {
        let config = IncrementalConfigBuilder::new()
            .track_provenance(true)
            .build();

        assert!(config.track_provenance);
    }

    #[test]
    fn test_builder_track_provenance_false() {
        let config = IncrementalConfigBuilder::new()
            .track_provenance(false)
            .build();

        assert!(!config.track_provenance);
    }

    #[test]
    fn test_builder_watch_mode_true() {
        let config = IncrementalConfigBuilder::new().watch_mode(true).build();

        assert!(config.watch_mode);
    }

    #[test]
    fn test_builder_watch_mode_false() {
        let config = IncrementalConfigBuilder::new().watch_mode(false).build();

        assert!(!config.watch_mode);
    }

    #[test]
    fn test_builder_watch_debounce() {
        let config = IncrementalConfigBuilder::new()
            .watch_debounce(Duration::from_millis(1000))
            .build();

        assert_eq!(config.watch_debounce, Duration::from_millis(1000));
    }

    #[test]
    fn test_builder_watch_debounce_short() {
        let config = IncrementalConfigBuilder::new()
            .watch_debounce(Duration::from_millis(10))
            .build();

        assert_eq!(config.watch_debounce, Duration::from_millis(10));
    }

    #[test]
    fn test_builder_min_clause_size() {
        let config = IncrementalConfigBuilder::new().min_clause_size(1).build();

        assert_eq!(config.min_clause_size, 1);
    }

    #[test]
    fn test_builder_min_clause_size_large() {
        let config = IncrementalConfigBuilder::new().min_clause_size(10).build();

        assert_eq!(config.min_clause_size, 10);
    }

    #[test]
    fn test_builder_max_clause_size() {
        let config = IncrementalConfigBuilder::new().max_clause_size(50).build();

        assert_eq!(config.max_clause_size, 50);
    }

    #[test]
    fn test_builder_max_clause_size_large() {
        let config = IncrementalConfigBuilder::new()
            .max_clause_size(1000)
            .build();

        assert_eq!(config.max_clause_size, 1000);
    }

    #[test]
    fn test_builder_solver_timeout() {
        let config = IncrementalConfigBuilder::new()
            .solver_timeout(Duration::from_secs(60))
            .build();

        assert_eq!(config.solver_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_builder_solver_timeout_long() {
        let config = IncrementalConfigBuilder::new()
            .solver_timeout(Duration::from_secs(3600)) // 1 hour
            .build();

        assert_eq!(config.solver_timeout, Duration::from_secs(3600));
    }

    #[test]
    fn test_builder_chained_all_methods() {
        let config = IncrementalConfigBuilder::new()
            .database_path("/tmp/full_test.db")
            .max_clause_age(Duration::from_secs(3600))
            .max_clauses_per_function(5000)
            .max_database_size(50 * 1024 * 1024)
            .content_addressable(false)
            .track_provenance(true)
            .watch_mode(true)
            .watch_debounce(Duration::from_millis(250))
            .min_clause_size(2)
            .max_clause_size(200)
            .solver_timeout(Duration::from_secs(120))
            .build();

        assert_eq!(config.database_path, PathBuf::from("/tmp/full_test.db"));
        assert_eq!(config.max_clause_age, Duration::from_secs(3600));
        assert_eq!(config.max_clauses_per_function, 5000);
        assert_eq!(config.max_database_size, 50 * 1024 * 1024);
        assert!(!config.content_addressable);
        assert!(config.track_provenance);
        assert!(config.watch_mode);
        assert_eq!(config.watch_debounce, Duration::from_millis(250));
        assert_eq!(config.min_clause_size, 2);
        assert_eq!(config.max_clause_size, 200);
        assert_eq!(config.solver_timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_builder_override_same_field() {
        // Later calls should override earlier ones
        let config = IncrementalConfigBuilder::new()
            .max_clauses_per_function(100)
            .max_clauses_per_function(200)
            .max_clauses_per_function(300)
            .build();

        assert_eq!(config.max_clauses_per_function, 300);
    }

    #[test]
    fn test_config_clone() {
        let config = IncrementalConfigBuilder::new()
            .database_path("/tmp/clone_test.db")
            .max_clause_age(Duration::from_secs(1800))
            .build();

        let cloned = config.clone();

        assert_eq!(cloned.database_path, config.database_path);
        assert_eq!(cloned.max_clause_age, config.max_clause_age);
        assert_eq!(
            cloned.max_clauses_per_function,
            config.max_clauses_per_function
        );
    }

    #[test]
    fn test_config_debug() {
        let config = IncrementalConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("IncrementalConfig"));
        assert!(debug_str.contains("database_path"));
        assert!(debug_str.contains("max_clause_age"));
    }

    #[test]
    fn test_builder_debug() {
        let builder = IncrementalConfigBuilder::new();
        let debug_str = format!("{:?}", builder);

        assert!(debug_str.contains("IncrementalConfigBuilder"));
    }
}
