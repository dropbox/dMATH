//! Configuration for k-induction engine

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for k-induction verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KInductionConfig {
    /// Maximum k value to try before giving up
    pub max_k: u32,

    /// Timeout for each BMC/induction step
    pub timeout_per_step: Duration,

    /// Total timeout for entire k-induction process
    pub total_timeout: Duration,

    /// Enable simple path constraint (strengthens induction)
    pub use_simple_path: bool,

    /// Enable auxiliary invariant synthesis
    pub enable_invariant_synthesis: bool,

    /// Maximum number of invariant synthesis attempts per k
    pub max_invariant_attempts: u32,

    /// Start with this k value (for incremental verification)
    pub initial_k: u32,

    /// Use portfolio solving for each step
    pub use_portfolio: bool,

    /// Number of parallel solvers in portfolio
    pub portfolio_size: usize,
}

impl Default for KInductionConfig {
    fn default() -> Self {
        Self {
            max_k: 50,
            timeout_per_step: Duration::from_secs(30),
            total_timeout: Duration::from_secs(300),
            use_simple_path: true,
            enable_invariant_synthesis: true,
            max_invariant_attempts: 10,
            initial_k: 1,
            use_portfolio: true,
            portfolio_size: 3,
        }
    }
}

/// Builder for KInductionConfig
#[derive(Debug, Default)]
pub struct KInductionConfigBuilder {
    max_k: Option<u32>,
    timeout_per_step: Option<Duration>,
    total_timeout: Option<Duration>,
    use_simple_path: Option<bool>,
    enable_invariant_synthesis: Option<bool>,
    max_invariant_attempts: Option<u32>,
    initial_k: Option<u32>,
    use_portfolio: Option<bool>,
    portfolio_size: Option<usize>,
}

impl KInductionConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_k(mut self, value: u32) -> Self {
        self.max_k = Some(value);
        self
    }

    pub fn timeout_per_step(mut self, value: Duration) -> Self {
        self.timeout_per_step = Some(value);
        self
    }

    pub fn timeout_per_step_ms(mut self, millis: u64) -> Self {
        self.timeout_per_step = Some(Duration::from_millis(millis));
        self
    }

    pub fn total_timeout(mut self, value: Duration) -> Self {
        self.total_timeout = Some(value);
        self
    }

    pub fn total_timeout_secs(mut self, secs: u64) -> Self {
        self.total_timeout = Some(Duration::from_secs(secs));
        self
    }

    pub fn use_simple_path(mut self, value: bool) -> Self {
        self.use_simple_path = Some(value);
        self
    }

    pub fn enable_invariant_synthesis(mut self, value: bool) -> Self {
        self.enable_invariant_synthesis = Some(value);
        self
    }

    pub fn max_invariant_attempts(mut self, value: u32) -> Self {
        self.max_invariant_attempts = Some(value);
        self
    }

    pub fn initial_k(mut self, value: u32) -> Self {
        self.initial_k = Some(value);
        self
    }

    pub fn use_portfolio(mut self, value: bool) -> Self {
        self.use_portfolio = Some(value);
        self
    }

    pub fn portfolio_size(mut self, value: usize) -> Self {
        self.portfolio_size = Some(value);
        self
    }

    pub fn build(self) -> KInductionConfig {
        let defaults = KInductionConfig::default();
        KInductionConfig {
            max_k: self.max_k.unwrap_or(defaults.max_k),
            timeout_per_step: self.timeout_per_step.unwrap_or(defaults.timeout_per_step),
            total_timeout: self.total_timeout.unwrap_or(defaults.total_timeout),
            use_simple_path: self.use_simple_path.unwrap_or(defaults.use_simple_path),
            enable_invariant_synthesis: self
                .enable_invariant_synthesis
                .unwrap_or(defaults.enable_invariant_synthesis),
            max_invariant_attempts: self
                .max_invariant_attempts
                .unwrap_or(defaults.max_invariant_attempts),
            initial_k: self.initial_k.unwrap_or(defaults.initial_k),
            use_portfolio: self.use_portfolio.unwrap_or(defaults.use_portfolio),
            portfolio_size: self.portfolio_size.unwrap_or(defaults.portfolio_size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ======== Default config tests ========

    #[test]
    fn test_default_config() {
        let config = KInductionConfig::default();
        assert_eq!(config.max_k, 50);
        assert_eq!(config.timeout_per_step, Duration::from_secs(30));
        assert!(config.use_simple_path);
        assert!(config.enable_invariant_synthesis);
    }

    #[test]
    fn test_default_config_all_fields() {
        let config = KInductionConfig::default();
        assert_eq!(config.max_k, 50);
        assert_eq!(config.timeout_per_step, Duration::from_secs(30));
        assert_eq!(config.total_timeout, Duration::from_secs(300));
        assert!(config.use_simple_path);
        assert!(config.enable_invariant_synthesis);
        assert_eq!(config.max_invariant_attempts, 10);
        assert_eq!(config.initial_k, 1);
        assert!(config.use_portfolio);
        assert_eq!(config.portfolio_size, 3);
    }

    // ======== Builder tests ========

    #[test]
    fn test_builder() {
        let config = KInductionConfigBuilder::new()
            .max_k(100)
            .timeout_per_step_ms(10000)
            .use_simple_path(false)
            .enable_invariant_synthesis(false)
            .build();

        assert_eq!(config.max_k, 100);
        assert_eq!(config.timeout_per_step, Duration::from_millis(10000));
        assert!(!config.use_simple_path);
        assert!(!config.enable_invariant_synthesis);
    }

    #[test]
    fn test_builder_default() {
        let builder = KInductionConfigBuilder::default();
        let config = builder.build();
        // Should have default values
        assert_eq!(config.max_k, 50);
    }

    #[test]
    fn test_builder_max_k() {
        let config = KInductionConfigBuilder::new().max_k(200).build();
        assert_eq!(config.max_k, 200);
    }

    #[test]
    fn test_builder_timeout_per_step() {
        let config = KInductionConfigBuilder::new()
            .timeout_per_step(Duration::from_secs(60))
            .build();
        assert_eq!(config.timeout_per_step, Duration::from_secs(60));
    }

    #[test]
    fn test_builder_timeout_per_step_ms() {
        let config = KInductionConfigBuilder::new()
            .timeout_per_step_ms(5000)
            .build();
        assert_eq!(config.timeout_per_step, Duration::from_millis(5000));
    }

    #[test]
    fn test_builder_total_timeout() {
        let config = KInductionConfigBuilder::new()
            .total_timeout(Duration::from_secs(600))
            .build();
        assert_eq!(config.total_timeout, Duration::from_secs(600));
    }

    #[test]
    fn test_builder_total_timeout_secs() {
        let config = KInductionConfigBuilder::new()
            .total_timeout_secs(120)
            .build();
        assert_eq!(config.total_timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_builder_use_simple_path() {
        let config_true = KInductionConfigBuilder::new().use_simple_path(true).build();
        assert!(config_true.use_simple_path);

        let config_false = KInductionConfigBuilder::new()
            .use_simple_path(false)
            .build();
        assert!(!config_false.use_simple_path);
    }

    #[test]
    fn test_builder_enable_invariant_synthesis() {
        let config_true = KInductionConfigBuilder::new()
            .enable_invariant_synthesis(true)
            .build();
        assert!(config_true.enable_invariant_synthesis);

        let config_false = KInductionConfigBuilder::new()
            .enable_invariant_synthesis(false)
            .build();
        assert!(!config_false.enable_invariant_synthesis);
    }

    #[test]
    fn test_builder_max_invariant_attempts() {
        let config = KInductionConfigBuilder::new()
            .max_invariant_attempts(20)
            .build();
        assert_eq!(config.max_invariant_attempts, 20);
    }

    #[test]
    fn test_builder_initial_k() {
        let config = KInductionConfigBuilder::new().initial_k(5).build();
        assert_eq!(config.initial_k, 5);
    }

    #[test]
    fn test_builder_use_portfolio() {
        let config_true = KInductionConfigBuilder::new().use_portfolio(true).build();
        assert!(config_true.use_portfolio);

        let config_false = KInductionConfigBuilder::new().use_portfolio(false).build();
        assert!(!config_false.use_portfolio);
    }

    #[test]
    fn test_builder_portfolio_size() {
        let config = KInductionConfigBuilder::new().portfolio_size(5).build();
        assert_eq!(config.portfolio_size, 5);
    }

    // ======== Builder chaining tests ========

    #[test]
    fn test_builder_chain_all_options() {
        let config = KInductionConfigBuilder::new()
            .max_k(75)
            .timeout_per_step(Duration::from_secs(45))
            .total_timeout(Duration::from_secs(450))
            .use_simple_path(false)
            .enable_invariant_synthesis(true)
            .max_invariant_attempts(15)
            .initial_k(2)
            .use_portfolio(true)
            .portfolio_size(4)
            .build();

        assert_eq!(config.max_k, 75);
        assert_eq!(config.timeout_per_step, Duration::from_secs(45));
        assert_eq!(config.total_timeout, Duration::from_secs(450));
        assert!(!config.use_simple_path);
        assert!(config.enable_invariant_synthesis);
        assert_eq!(config.max_invariant_attempts, 15);
        assert_eq!(config.initial_k, 2);
        assert!(config.use_portfolio);
        assert_eq!(config.portfolio_size, 4);
    }

    #[test]
    fn test_builder_overwrite_values() {
        // Later calls should overwrite earlier ones
        let config = KInductionConfigBuilder::new()
            .max_k(10)
            .max_k(20)
            .max_k(30)
            .build();

        assert_eq!(config.max_k, 30);
    }

    // ======== Serialization tests ========

    #[test]
    fn test_config_serialization() {
        let config = KInductionConfig::default();
        let json = serde_json::to_string(&config);
        assert!(json.is_ok());

        let json_str = json.unwrap();
        assert!(json_str.contains("max_k"));
        assert!(json_str.contains("50"));
    }

    #[test]
    fn test_config_deserialization() {
        let json = r#"{
            "max_k": 100,
            "timeout_per_step": {"secs": 60, "nanos": 0},
            "total_timeout": {"secs": 600, "nanos": 0},
            "use_simple_path": false,
            "enable_invariant_synthesis": true,
            "max_invariant_attempts": 5,
            "initial_k": 3,
            "use_portfolio": false,
            "portfolio_size": 2
        }"#;

        let config: Result<KInductionConfig, _> = serde_json::from_str(json);
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.max_k, 100);
        assert_eq!(config.timeout_per_step, Duration::from_secs(60));
        assert!(!config.use_simple_path);
        assert_eq!(config.initial_k, 3);
    }

    #[test]
    fn test_config_round_trip() {
        let original = KInductionConfigBuilder::new()
            .max_k(42)
            .use_simple_path(false)
            .build();

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: KInductionConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.max_k, 42);
        assert!(!deserialized.use_simple_path);
    }

    // ======== Clone and Debug tests ========

    #[test]
    fn test_config_clone() {
        let config = KInductionConfigBuilder::new()
            .max_k(77)
            .use_simple_path(false)
            .build();

        let cloned = config.clone();
        assert_eq!(cloned.max_k, 77);
        assert!(!cloned.use_simple_path);
    }

    #[test]
    fn test_config_debug() {
        let config = KInductionConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("KInductionConfig"));
        assert!(debug_str.contains("max_k"));
    }

    #[test]
    fn test_builder_debug() {
        let builder = KInductionConfigBuilder::new().max_k(10);
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("KInductionConfigBuilder"));
    }

    // ======== Edge cases ========

    #[test]
    fn test_config_zero_max_k() {
        let config = KInductionConfigBuilder::new().max_k(0).build();
        assert_eq!(config.max_k, 0);
    }

    #[test]
    fn test_config_large_max_k() {
        let config = KInductionConfigBuilder::new().max_k(u32::MAX).build();
        assert_eq!(config.max_k, u32::MAX);
    }

    #[test]
    fn test_config_zero_timeout() {
        let config = KInductionConfigBuilder::new()
            .timeout_per_step(Duration::ZERO)
            .build();
        assert_eq!(config.timeout_per_step, Duration::ZERO);
    }

    #[test]
    fn test_config_portfolio_size_one() {
        let config = KInductionConfigBuilder::new().portfolio_size(1).build();
        assert_eq!(config.portfolio_size, 1);
    }

    #[test]
    fn test_config_initial_k_zero() {
        let config = KInductionConfigBuilder::new().initial_k(0).build();
        assert_eq!(config.initial_k, 0);
    }
}
