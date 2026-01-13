//! Great Expectations backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Validation level for data quality checks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationLevel {
    /// Basic schema validation
    Basic,
    /// Standard validation suite
    #[default]
    Standard,
    /// Comprehensive validation with statistical tests
    Comprehensive,
}

impl ValidationLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            ValidationLevel::Basic => "basic",
            ValidationLevel::Standard => "standard",
            ValidationLevel::Comprehensive => "comprehensive",
        }
    }
}

/// Data source type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataSourceType {
    /// Pandas DataFrame
    #[default]
    Pandas,
    /// Spark DataFrame
    Spark,
    /// SQL Database
    SQL,
    /// File-based (CSV, Parquet, etc.)
    File,
}

impl DataSourceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataSourceType::Pandas => "pandas",
            DataSourceType::Spark => "spark",
            DataSourceType::SQL => "sql",
            DataSourceType::File => "file",
        }
    }
}

/// Result format for validation output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResultFormat {
    /// JSON format
    #[default]
    JSON,
    /// HTML report
    HTML,
    /// Markdown report
    Markdown,
}

impl ResultFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResultFormat::JSON => "json",
            ResultFormat::HTML => "html",
            ResultFormat::Markdown => "markdown",
        }
    }
}

/// Great Expectations backend configuration
#[derive(Debug, Clone)]
pub struct GreatExpectationsConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Validation level
    pub validation_level: ValidationLevel,
    /// Data source type
    pub data_source_type: DataSourceType,
    /// Result format
    pub result_format: ResultFormat,
    /// Path to expectation suite
    pub expectation_suite_path: Option<PathBuf>,
    /// Data path for validation
    pub data_path: Option<PathBuf>,
    /// Verification timeout
    pub timeout: Duration,
    /// Enable evaluation parameters
    pub evaluation_parameters: bool,
    /// Catch exceptions during validation
    pub catch_exceptions: bool,
    /// Include unexpected rows in results
    pub include_unexpected_rows: bool,
    /// Maximum unexpected values to return
    pub max_unexpected_values: usize,
}

impl Default for GreatExpectationsConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            validation_level: ValidationLevel::Standard,
            data_source_type: DataSourceType::Pandas,
            result_format: ResultFormat::JSON,
            expectation_suite_path: None,
            data_path: None,
            timeout: Duration::from_secs(300),
            evaluation_parameters: true,
            catch_exceptions: true,
            include_unexpected_rows: false,
            max_unexpected_values: 20,
        }
    }
}

impl GreatExpectationsConfig {
    /// Create config for basic schema validation
    pub fn basic() -> Self {
        Self {
            validation_level: ValidationLevel::Basic,
            ..Default::default()
        }
    }

    /// Create config for comprehensive validation
    pub fn comprehensive() -> Self {
        Self {
            validation_level: ValidationLevel::Comprehensive,
            include_unexpected_rows: true,
            ..Default::default()
        }
    }

    /// Create config for Spark data sources
    pub fn spark() -> Self {
        Self {
            data_source_type: DataSourceType::Spark,
            ..Default::default()
        }
    }

    /// Create config for SQL data sources
    pub fn sql() -> Self {
        Self {
            data_source_type: DataSourceType::SQL,
            ..Default::default()
        }
    }
}
