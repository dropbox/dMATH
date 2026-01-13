//! WhyLogs backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Profile type for WhyLogs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProfileType {
    /// Standard profile with all metrics
    #[default]
    Standard,
    /// Lightweight profile for streaming
    Lightweight,
    /// Detailed profile with distributions
    Detailed,
}

impl ProfileType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProfileType::Standard => "standard",
            ProfileType::Lightweight => "lightweight",
            ProfileType::Detailed => "detailed",
        }
    }
}

/// Constraint type for validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConstraintType {
    /// Schema constraints
    #[default]
    Schema,
    /// Value constraints (ranges, sets)
    Value,
    /// Distribution constraints
    Distribution,
    /// All constraints
    All,
}

impl ConstraintType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ConstraintType::Schema => "schema",
            ConstraintType::Value => "value",
            ConstraintType::Distribution => "distribution",
            ConstraintType::All => "all",
        }
    }
}

/// Output format for profiles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WhyLogsOutputFormat {
    /// JSON format
    #[default]
    JSON,
    /// Protobuf binary
    Protobuf,
    /// Flat format (metrics only)
    Flat,
}

impl WhyLogsOutputFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            WhyLogsOutputFormat::JSON => "json",
            WhyLogsOutputFormat::Protobuf => "protobuf",
            WhyLogsOutputFormat::Flat => "flat",
        }
    }
}

/// WhyLogs backend configuration
#[derive(Debug, Clone)]
pub struct WhyLogsConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Profile type
    pub profile_type: ProfileType,
    /// Constraint type for validation
    pub constraint_type: ConstraintType,
    /// Output format
    pub output_format: WhyLogsOutputFormat,
    /// Verification timeout
    pub timeout: Duration,
    /// Number of samples for profiling
    pub n_samples: usize,
    /// Track distribution histograms
    pub track_histograms: bool,
    /// Track frequent items
    pub track_frequent_items: bool,
    /// Enable cardinality tracking
    pub track_cardinality: bool,
}

impl Default for WhyLogsConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            profile_type: ProfileType::Standard,
            constraint_type: ConstraintType::All,
            output_format: WhyLogsOutputFormat::JSON,
            timeout: Duration::from_secs(300),
            n_samples: 1000,
            track_histograms: true,
            track_frequent_items: true,
            track_cardinality: true,
        }
    }
}

impl WhyLogsConfig {
    /// Create config for lightweight profiling
    pub fn lightweight() -> Self {
        Self {
            profile_type: ProfileType::Lightweight,
            track_histograms: false,
            track_frequent_items: false,
            ..Default::default()
        }
    }

    /// Create config for detailed profiling
    pub fn detailed() -> Self {
        Self {
            profile_type: ProfileType::Detailed,
            track_histograms: true,
            track_frequent_items: true,
            track_cardinality: true,
            ..Default::default()
        }
    }

    /// Create config for schema validation only
    pub fn schema_only() -> Self {
        Self {
            constraint_type: ConstraintType::Schema,
            ..Default::default()
        }
    }

    /// Create config for value validation
    pub fn value_constraints() -> Self {
        Self {
            constraint_type: ConstraintType::Value,
            ..Default::default()
        }
    }
}
