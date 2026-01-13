//! Training pipeline for DashFlow integration
//!
//! This module provides a training pipeline that:
//! 1. Collects verification feedback from local runs
//! 2. Queues feedback for batch upload to DashFlow
//! 3. Exports training data in DashFlow-compatible format
//!
//! # Architecture
//!
//! The pipeline operates in two modes:
//!
//! - **Online mode**: Feedback is queued and periodically uploaded to DashFlow
//! - **Offline mode**: Feedback is collected locally and exported for batch import
//!
//! # Example
//!
//! ```rust,no_run
//! use dashprove_learning::pipeline::{TrainingPipeline, PipelineConfig, FeedbackRecord};
//! use dashprove_backends::BackendId;
//! use std::time::Duration;
//!
//! let pipeline = TrainingPipeline::new(PipelineConfig {
//!     queue_capacity: 1000,
//!     batch_size: 50,
//!     flush_interval: Duration::from_secs(60),
//!     export_dir: Some("/tmp/dashprove_feedback".into()),
//!     deduplicate: true,
//! });
//!
//! // Record feedback
//! pipeline.record(FeedbackRecord {
//!     property_name: "test_theorem".to_string(),
//!     property_type: "theorem".to_string(),
//!     backend: BackendId::Lean4,
//!     success: true,
//!     time_taken: Duration::from_millis(150),
//!     tactics: vec!["simp".to_string()],
//!     ..Default::default()
//! });
//!
//! // Export for batch upload
//! let records = pipeline.export_pending();
//! ```

use chrono::{DateTime, Utc};
use dashprove_backends::BackendId;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Configuration for the training pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum number of feedback records to queue
    pub queue_capacity: usize,
    /// Number of records per batch upload
    pub batch_size: usize,
    /// How often to flush the queue (online mode)
    pub flush_interval: Duration,
    /// Directory to export training data (offline mode)
    pub export_dir: Option<PathBuf>,
    /// Whether to deduplicate similar feedback records
    pub deduplicate: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 10000,
            batch_size: 100,
            flush_interval: Duration::from_secs(300), // 5 minutes
            export_dir: None,
            deduplicate: true,
        }
    }
}

impl PipelineConfig {
    /// Create config optimized for online mode (frequent uploads)
    pub fn online() -> Self {
        Self {
            queue_capacity: 1000,
            batch_size: 50,
            flush_interval: Duration::from_secs(60),
            export_dir: None,
            deduplicate: true,
        }
    }

    /// Create config optimized for offline mode (batch exports)
    pub fn offline(export_dir: PathBuf) -> Self {
        Self {
            queue_capacity: 100_000,
            batch_size: 1000,
            flush_interval: Duration::from_secs(3600), // 1 hour
            export_dir: Some(export_dir),
            deduplicate: false,
        }
    }
}

/// A single feedback record for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackRecord {
    /// Unique identifier for this record
    pub id: u64,
    /// Property name
    pub property_name: String,
    /// Property type (theorem, invariant, temporal, etc.)
    pub property_type: String,
    /// Backend used for verification
    pub backend: BackendId,
    /// Whether verification succeeded
    pub success: bool,
    /// Time taken for verification
    #[serde(with = "duration_serde")]
    pub time_taken: Duration,
    /// Tactics used (if any)
    pub tactics: Vec<String>,
    /// Expression depth
    pub depth: usize,
    /// Number of quantifiers
    pub quantifier_depth: usize,
    /// Number of implications
    pub implication_count: usize,
    /// Number of arithmetic operations
    pub arithmetic_ops: usize,
    /// Number of function calls
    pub function_calls: usize,
    /// Number of variables
    pub variable_count: usize,
    /// Uses temporal operators
    pub has_temporal: bool,
    /// When the verification was recorded
    pub timestamp: DateTime<Utc>,
    /// Optional error message
    pub error_message: Option<String>,
    /// Optional proof size in bytes
    pub proof_size: Option<usize>,
}

impl Default for FeedbackRecord {
    fn default() -> Self {
        Self {
            id: 0,
            property_name: String::new(),
            property_type: "unknown".to_string(),
            backend: BackendId::Lean4,
            success: false,
            time_taken: Duration::ZERO,
            tactics: Vec::new(),
            depth: 0,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            function_calls: 0,
            variable_count: 0,
            has_temporal: false,
            timestamp: Utc::now(),
            error_message: None,
            proof_size: None,
        }
    }
}

/// Duration serialization helper
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs_f64().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

/// Statistics about the training pipeline
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total records received
    pub total_records: u64,
    /// Records currently queued
    pub queued_records: usize,
    /// Records successfully uploaded/exported
    pub exported_records: u64,
    /// Records dropped due to queue overflow
    pub dropped_records: u64,
    /// Successful verifications recorded
    pub successful_verifications: u64,
    /// Failed verifications recorded
    pub failed_verifications: u64,
    /// Records per backend
    pub by_backend: std::collections::HashMap<BackendId, u64>,
}

/// Training pipeline for collecting and exporting verification feedback
pub struct TrainingPipeline {
    config: PipelineConfig,
    queue: Arc<Mutex<VecDeque<FeedbackRecord>>>,
    next_id: AtomicU64,
    stats: Arc<Mutex<PipelineStats>>,
}

impl TrainingPipeline {
    /// Create a new training pipeline with the given configuration
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            next_id: AtomicU64::new(1),
            stats: Arc::new(Mutex::new(PipelineStats::default())),
        }
    }

    /// Create a pipeline with default configuration
    pub fn default_pipeline() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Get the pipeline configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Record a verification feedback event
    ///
    /// Returns the record ID if queued successfully, None if dropped.
    pub fn record(&self, mut record: FeedbackRecord) -> Option<u64> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        record.id = id;

        let mut queue = self.queue.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        stats.total_records += 1;
        if record.success {
            stats.successful_verifications += 1;
        } else {
            stats.failed_verifications += 1;
        }
        *stats.by_backend.entry(record.backend).or_insert(0) += 1;

        // Check capacity
        if queue.len() >= self.config.queue_capacity {
            // Drop oldest record
            queue.pop_front();
            stats.dropped_records += 1;
            tracing::warn!(id, "Training pipeline queue full, dropping oldest record");
        }

        queue.push_back(record);
        stats.queued_records = queue.len();

        Some(id)
    }

    /// Export pending records and clear the queue
    ///
    /// Returns the exported records for upload to DashFlow.
    pub fn export_pending(&self) -> Vec<FeedbackRecord> {
        let mut queue = self.queue.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let records: Vec<_> = queue.drain(..).collect();
        stats.exported_records += records.len() as u64;
        stats.queued_records = 0;

        records
    }

    /// Export a batch of records (up to batch_size)
    ///
    /// Returns the exported records, leaving remaining records in queue.
    pub fn export_batch(&self) -> Vec<FeedbackRecord> {
        let mut queue = self.queue.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let count = std::cmp::min(self.config.batch_size, queue.len());
        let records: Vec<_> = queue.drain(..count).collect();

        stats.exported_records += records.len() as u64;
        stats.queued_records = queue.len();

        records
    }

    /// Get current pipeline statistics
    pub fn stats(&self) -> PipelineStats {
        let stats = self.stats.lock().unwrap();
        let queue = self.queue.lock().unwrap();

        PipelineStats {
            queued_records: queue.len(),
            ..stats.clone()
        }
    }

    /// Get the number of pending records
    pub fn pending_count(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }

    /// Clear all pending records
    pub fn clear(&self) {
        let mut queue = self.queue.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        queue.clear();
        stats.queued_records = 0;
    }

    /// Save pending records to a JSON file
    ///
    /// If export_dir is configured, uses that directory.
    /// Otherwise uses the provided path.
    pub fn save_to_file(&self, path: Option<&std::path::Path>) -> std::io::Result<usize> {
        let export_path = path
            .map(|p| p.to_path_buf())
            .or_else(|| {
                self.config.export_dir.as_ref().map(|dir| {
                    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
                    dir.join(format!("feedback_{}.json", timestamp))
                })
            })
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "No export path specified")
            })?;

        // Ensure parent directory exists
        if let Some(parent) = export_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let records = self.export_pending();
        let count = records.len();

        let json = serde_json::to_string_pretty(&records)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        std::fs::write(&export_path, json)?;

        tracing::info!(
            path = %export_path.display(),
            count,
            "Exported training feedback to file"
        );

        Ok(count)
    }

    /// Load records from a JSON file and add them to the queue
    pub fn load_from_file(&self, path: &std::path::Path) -> std::io::Result<usize> {
        let json = std::fs::read_to_string(path)?;
        let records: Vec<FeedbackRecord> = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let count = records.len();
        for record in records {
            self.record(record);
        }

        tracing::info!(
            path = %path.display(),
            count,
            "Loaded training feedback from file"
        );

        Ok(count)
    }
}

impl Default for TrainingPipeline {
    fn default() -> Self {
        Self::default_pipeline()
    }
}

/// Batch of feedback records for upload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackBatch {
    /// Batch ID for tracking
    pub batch_id: String,
    /// Records in this batch
    pub records: Vec<FeedbackRecord>,
    /// When this batch was created
    pub created_at: DateTime<Utc>,
}

impl FeedbackBatch {
    /// Create a new batch from records
    pub fn new(records: Vec<FeedbackRecord>) -> Self {
        Self {
            batch_id: format!("batch_{}", Utc::now().timestamp_millis()),
            records,
            created_at: Utc::now(),
        }
    }

    /// Number of records in the batch
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.queue_capacity, 10000);
        assert_eq!(config.batch_size, 100);
        assert!(config.deduplicate);
    }

    #[test]
    fn test_pipeline_config_online() {
        let config = PipelineConfig::online();
        assert_eq!(config.queue_capacity, 1000);
        assert_eq!(config.batch_size, 50);
    }

    #[test]
    fn test_pipeline_config_offline() {
        let config = PipelineConfig::offline("/tmp/test".into());
        assert_eq!(config.queue_capacity, 100_000);
        assert!(config.export_dir.is_some());
    }

    #[test]
    fn test_pipeline_record() {
        let pipeline = TrainingPipeline::default();

        let id = pipeline.record(FeedbackRecord {
            property_name: "test".to_string(),
            success: true,
            ..Default::default()
        });

        assert!(id.is_some());
        assert_eq!(pipeline.pending_count(), 1);
    }

    #[test]
    fn test_pipeline_export_pending() {
        let pipeline = TrainingPipeline::default();

        pipeline.record(FeedbackRecord {
            property_name: "test1".to_string(),
            ..Default::default()
        });
        pipeline.record(FeedbackRecord {
            property_name: "test2".to_string(),
            ..Default::default()
        });

        assert_eq!(pipeline.pending_count(), 2);

        let records = pipeline.export_pending();
        assert_eq!(records.len(), 2);
        assert_eq!(pipeline.pending_count(), 0);
    }

    #[test]
    fn test_pipeline_export_batch() {
        let pipeline = TrainingPipeline::new(PipelineConfig {
            batch_size: 2,
            ..Default::default()
        });

        for i in 0..5 {
            pipeline.record(FeedbackRecord {
                property_name: format!("test{}", i),
                ..Default::default()
            });
        }

        assert_eq!(pipeline.pending_count(), 5);

        let batch1 = pipeline.export_batch();
        assert_eq!(batch1.len(), 2);
        assert_eq!(pipeline.pending_count(), 3);

        let batch2 = pipeline.export_batch();
        assert_eq!(batch2.len(), 2);
        assert_eq!(pipeline.pending_count(), 1);

        let batch3 = pipeline.export_batch();
        assert_eq!(batch3.len(), 1);
        assert_eq!(pipeline.pending_count(), 0);
    }

    #[test]
    fn test_pipeline_stats() {
        let pipeline = TrainingPipeline::default();

        pipeline.record(FeedbackRecord {
            property_name: "success".to_string(),
            backend: BackendId::Lean4,
            success: true,
            ..Default::default()
        });
        pipeline.record(FeedbackRecord {
            property_name: "failure".to_string(),
            backend: BackendId::Alloy,
            success: false,
            ..Default::default()
        });

        let stats = pipeline.stats();
        assert_eq!(stats.total_records, 2);
        assert_eq!(stats.queued_records, 2);
        assert_eq!(stats.successful_verifications, 1);
        assert_eq!(stats.failed_verifications, 1);
        assert_eq!(*stats.by_backend.get(&BackendId::Lean4).unwrap(), 1);
        assert_eq!(*stats.by_backend.get(&BackendId::Alloy).unwrap(), 1);
    }

    #[test]
    fn test_pipeline_capacity_overflow() {
        let pipeline = TrainingPipeline::new(PipelineConfig {
            queue_capacity: 2,
            ..Default::default()
        });

        pipeline.record(FeedbackRecord {
            property_name: "first".to_string(),
            ..Default::default()
        });
        pipeline.record(FeedbackRecord {
            property_name: "second".to_string(),
            ..Default::default()
        });
        pipeline.record(FeedbackRecord {
            property_name: "third".to_string(),
            ..Default::default()
        });

        // Queue should have 2 records (first dropped)
        assert_eq!(pipeline.pending_count(), 2);

        let stats = pipeline.stats();
        assert_eq!(stats.dropped_records, 1);
    }

    #[test]
    fn test_feedback_batch() {
        let records = vec![
            FeedbackRecord {
                property_name: "test1".to_string(),
                ..Default::default()
            },
            FeedbackRecord {
                property_name: "test2".to_string(),
                ..Default::default()
            },
        ];

        let batch = FeedbackBatch::new(records);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert!(batch.batch_id.starts_with("batch_"));
    }

    #[test]
    fn test_feedback_record_default() {
        let record = FeedbackRecord::default();
        assert_eq!(record.property_type, "unknown");
        assert!(!record.success);
        assert!(record.tactics.is_empty());
    }

    #[test]
    fn test_pipeline_clear() {
        let pipeline = TrainingPipeline::default();

        pipeline.record(FeedbackRecord::default());
        pipeline.record(FeedbackRecord::default());
        assert_eq!(pipeline.pending_count(), 2);

        pipeline.clear();
        assert!(pipeline.is_empty());
    }
}
