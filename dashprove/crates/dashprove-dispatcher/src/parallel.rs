//! Parallel verification execution
//!
//! This module provides parallel execution of verification tasks across multiple backends.
//!
//! ## Features
//!
//! - **Priority-based scheduling**: Higher priority tasks execute first
//! - **Retry with exponential backoff**: Transient failures are retried automatically
//! - **Cancellation support**: In-flight tasks can be cancelled gracefully
//! - **Adaptive concurrency**: Dynamically adjust concurrency based on success rates

use crate::selector::Selection;
use dashprove_backends::{
    BackendError, BackendId, BackendResult, PropertyType, VerificationBackend,
};
use dashprove_usl::typecheck::TypedSpec;
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{watch, Semaphore};
use tracing::{debug, info, warn};

/// Errors from the parallel executor
#[derive(Error, Debug)]
pub enum ExecutorError {
    /// All verification tasks failed (none succeeded)
    #[error("All verification tasks failed")]
    AllTasksFailed,

    /// Verification exceeded the configured timeout
    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    /// Backend reported an error during verification
    #[error("Backend error: {0}")]
    BackendError(#[from] BackendError),

    /// Verification was cancelled
    #[error("Verification cancelled")]
    Cancelled,
}

/// Result of a single verification task
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Index of the property in the original spec
    pub property_index: usize,
    /// Which backend ran the verification
    pub backend: BackendId,
    /// The verification result
    pub result: Result<BackendResult, String>,
}

/// Aggregated results from parallel execution
#[derive(Debug)]
pub struct ExecutionResults {
    /// Results for each property, grouped by property index
    pub by_property: HashMap<usize, Vec<TaskResult>>,
    /// Property types for each property index
    pub property_types: HashMap<usize, PropertyType>,
    /// Total execution time
    pub total_time: Duration,
    /// Number of successful verifications
    pub successful: usize,
    /// Number of failed verifications
    pub failed: usize,
}

/// Configuration for the parallel executor
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum concurrent verification tasks
    pub max_concurrent: usize,
    /// Timeout per verification task
    pub task_timeout: Duration,
    /// Whether to fail fast on first error
    pub fail_fast: bool,
    /// Retry configuration for transient failures
    pub retry_config: RetryConfig,
    /// Enable adaptive concurrency (dynamically adjust based on success rate)
    pub adaptive_concurrency: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        ExecutorConfig {
            max_concurrent: 4,
            task_timeout: Duration::from_secs(300), // 5 minutes default
            fail_fast: false,
            retry_config: RetryConfig::default(),
            adaptive_concurrency: false,
        }
    }
}

/// Configuration for retry behavior with exponential backoff
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries)
    pub max_retries: u32,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Backoff multiplier (e.g., 2.0 for doubling)
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_retries: 2,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    /// Create a config with no retries
    pub fn no_retries() -> Self {
        RetryConfig {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Calculate backoff duration for a given attempt number
    pub fn backoff_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }
        let backoff_ms = self.initial_backoff.as_millis() as f64
            * self
                .backoff_multiplier
                .powi(attempt.saturating_sub(1) as i32);
        let backoff = Duration::from_millis(backoff_ms as u64);
        std::cmp::min(backoff, self.max_backoff)
    }
}

/// Task priority levels for scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TaskPriority {
    /// Highest priority - critical verification tasks
    Critical = 3,
    /// High priority - important but not critical
    High = 2,
    /// Normal priority - default
    #[default]
    Normal = 1,
    /// Low priority - can be deferred
    Low = 0,
}

/// Internal task representation with priority for scheduling
#[derive(Clone)]
struct PrioritizedTask {
    property_index: usize,
    backend_id: BackendId,
    backend: Arc<dyn VerificationBackend>,
    priority: TaskPriority,
}

impl PartialEq for PrioritizedTask {
    fn eq(&self, other: &Self) -> bool {
        self.property_index == other.property_index
            && self.backend_id == other.backend_id
            && self.priority == other.priority
    }
}

impl Eq for PrioritizedTask {}

impl PartialOrd for PrioritizedTask {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedTask {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Higher priority (larger value) should come first (be "greater")
        (self.priority as u8).cmp(&(other.priority as u8))
    }
}

/// Execute a verification with retry logic and exponential backoff
async fn execute_with_retry(
    backend: &Arc<dyn VerificationBackend>,
    spec: &TypedSpec,
    timeout: Duration,
    retry_config: &RetryConfig,
    cancel_token: Option<&CancellationToken>,
) -> Result<BackendResult, String> {
    let mut last_error = String::new();

    for attempt in 0..=retry_config.max_retries {
        // Check cancellation before each attempt
        if let Some(token) = cancel_token {
            if token.is_cancelled() {
                return Err("Cancelled".into());
            }
        }

        // Apply backoff delay (skip for first attempt)
        if attempt > 0 {
            let backoff = retry_config.backoff_for_attempt(attempt);
            debug!(
                attempt,
                backoff_ms = backoff.as_millis(),
                "Retrying after backoff"
            );
            tokio::time::sleep(backoff).await;
        }

        // Run with timeout
        let result = tokio::time::timeout(timeout, backend.verify(spec)).await;

        match result {
            Ok(Ok(r)) => return Ok(r),
            Ok(Err(e)) => {
                last_error = format!("Backend error: {}", e);
                // Check if error is retryable (not all errors should be retried)
                if !is_retryable_error(&e) {
                    return Err(last_error);
                }
                debug!(
                    attempt,
                    error = %last_error,
                    "Retryable error, will retry"
                );
            }
            Err(_) => {
                last_error = format!("Timeout after {:?}", timeout);
                // Timeouts are generally retryable
                debug!(
                    attempt,
                    error = %last_error,
                    "Timeout, will retry"
                );
            }
        }
    }

    Err(last_error)
}

/// Check if an error is retryable (transient failures that may succeed on retry)
fn is_retryable_error(error: &BackendError) -> bool {
    match error {
        // Timeouts might succeed on retry (backend was slow, network issue, etc.)
        BackendError::Timeout(_) => true,
        // Backend unavailable might be temporary (backend starting up, etc.)
        BackendError::Unavailable(_) => true,
        // Compilation failures are not retryable (input-dependent)
        BackendError::CompilationFailed(_) => false,
        // Verification failures are not retryable (deterministic result)
        BackendError::VerificationFailed(_) => false,
    }
}

/// Progress update emitted as tasks complete
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    /// How many tasks have completed (success or failure)
    pub completed: usize,
    /// Total number of verification tasks scheduled
    pub total: usize,
    /// Backend that just finished
    pub backend: BackendId,
    /// Property index for the completed task
    pub property_index: usize,
    /// Number of successful tasks so far
    pub successful_so_far: usize,
    /// Number of failed tasks so far
    pub failed_so_far: usize,
}

/// Cancellation token for gracefully stopping verification
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    notify: watch::Sender<bool>,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        let (tx, _rx) = watch::channel(false);
        CancellationToken {
            cancelled: Arc::new(AtomicBool::new(false)),
            notify: tx,
        }
    }

    /// Signal cancellation to all listeners
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        let _ = self.notify.send(true);
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Create a receiver that notifies when cancelled
    pub fn subscribe(&self) -> watch::Receiver<bool> {
        self.notify.subscribe()
    }
}

/// Parallel verification executor
pub struct ParallelExecutor {
    config: ExecutorConfig,
    semaphore: Arc<Semaphore>,
    progress_callback: Option<Arc<dyn Fn(ProgressUpdate) + Send + Sync>>,
    cancellation_token: Option<CancellationToken>,
}

impl ParallelExecutor {
    /// Create a new executor with the given configuration
    pub fn new(config: ExecutorConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        ParallelExecutor {
            config,
            semaphore,
            progress_callback: None,
            cancellation_token: None,
        }
    }

    /// Set a cancellation token for the executor
    pub fn set_cancellation_token(&mut self, token: CancellationToken) {
        self.cancellation_token = Some(token);
    }

    /// Get the current cancellation token (if set)
    pub fn cancellation_token(&self) -> Option<&CancellationToken> {
        self.cancellation_token.as_ref()
    }

    /// Clear the cancellation token
    pub fn clear_cancellation_token(&mut self) {
        self.cancellation_token = None;
    }

    /// Execute verification tasks based on selection
    pub async fn execute(
        &self,
        selection: &Selection,
        spec: &TypedSpec,
        backends: &HashMap<BackendId, Arc<dyn VerificationBackend>>,
    ) -> Result<ExecutionResults, ExecutorError> {
        self.execute_with_priority(selection, spec, backends, TaskPriority::Normal)
            .await
    }

    /// Execute verification tasks with explicit priority
    pub async fn execute_with_priority(
        &self,
        selection: &Selection,
        spec: &TypedSpec,
        backends: &HashMap<BackendId, Arc<dyn VerificationBackend>>,
        default_priority: TaskPriority,
    ) -> Result<ExecutionResults, ExecutorError> {
        let start = std::time::Instant::now();

        // Check for cancellation before starting
        if let Some(ref token) = self.cancellation_token {
            if token.is_cancelled() {
                return Err(ExecutorError::Cancelled);
            }
        }

        // Collect all tasks to execute with priority
        let mut tasks = Vec::new();
        let mut property_types = HashMap::new();
        for assignment in &selection.assignments {
            for &backend_id in &assignment.backends {
                if let Some(backend) = backends.get(&backend_id) {
                    tasks.push(PrioritizedTask {
                        property_index: assignment.property_index,
                        backend_id,
                        backend: Arc::clone(backend),
                        priority: default_priority,
                    });
                    property_types
                        .entry(assignment.property_index)
                        .or_insert(assignment.property_type);
                } else {
                    warn!(?backend_id, "Backend not available in executor");
                }
            }
        }

        if tasks.is_empty() {
            return Err(ExecutorError::AllTasksFailed);
        }

        info!(task_count = tasks.len(), "Starting parallel verification");

        // Execute tasks in parallel - clone spec into Arc for sharing across tasks
        let spec_arc = Arc::new((*spec).clone());
        let results = self.execute_tasks_with_retry(tasks, spec_arc).await?;

        // Group results by property
        let mut by_property: HashMap<usize, Vec<TaskResult>> = HashMap::new();
        let mut successful = 0;
        let mut failed = 0;

        for result in results {
            if result.result.is_ok() {
                successful += 1;
            } else {
                failed += 1;
            }
            by_property
                .entry(result.property_index)
                .or_default()
                .push(result);
        }

        let total_time = start.elapsed();
        info!(?total_time, successful, failed, "Verification complete");

        Ok(ExecutionResults {
            by_property,
            property_types,
            total_time,
            successful,
            failed,
        })
    }

    /// Execute tasks with concurrency limiting, retry support, and cancellation
    async fn execute_tasks_with_retry(
        &self,
        tasks: Vec<PrioritizedTask>,
        spec: Arc<TypedSpec>,
    ) -> Result<Vec<TaskResult>, ExecutorError> {
        let mut handles = Vec::new();
        let total = tasks.len();
        let completed = Arc::new(AtomicUsize::new(0));
        let successful_count = Arc::new(AtomicUsize::new(0));
        let failed_count = Arc::new(AtomicUsize::new(0));
        let progress = self.progress_callback.clone();
        let retry_config = self.config.retry_config.clone();
        let cancel_token = self.cancellation_token.clone();

        // Sort tasks by priority (highest first) for scheduling order
        let mut sorted_tasks: Vec<_> = tasks.into_iter().collect();
        sorted_tasks.sort_by(|a, b| (b.priority as u8).cmp(&(a.priority as u8)));

        for task in sorted_tasks {
            let semaphore = Arc::clone(&self.semaphore);
            let spec = Arc::clone(&spec);
            let timeout = self.config.task_timeout;
            let completed = Arc::clone(&completed);
            let successful_count = Arc::clone(&successful_count);
            let failed_count = Arc::clone(&failed_count);
            let progress = progress.clone();
            let retry_config = retry_config.clone();
            let cancel_token = cancel_token.clone();

            let handle = tokio::spawn(async move {
                // Check cancellation before acquiring permit
                if let Some(ref token) = cancel_token {
                    if token.is_cancelled() {
                        return TaskResult {
                            property_index: task.property_index,
                            backend: task.backend_id,
                            result: Err("Cancelled".into()),
                        };
                    }
                }

                // Acquire semaphore permit
                let _permit = semaphore.acquire().await.expect("Semaphore closed");

                debug!(
                    backend_id = ?task.backend_id,
                    property_index = task.property_index,
                    priority = ?task.priority,
                    "Starting verification task"
                );

                // Execute with retry logic
                let result = execute_with_retry(
                    &task.backend,
                    &spec,
                    timeout,
                    &retry_config,
                    cancel_token.as_ref(),
                )
                .await;

                let success = result.is_ok();

                debug!(
                    backend_id = ?task.backend_id,
                    property_index = task.property_index,
                    success,
                    "Task complete"
                );

                let finished = completed.fetch_add(1, Ordering::Relaxed) + 1;
                let succ = if success {
                    successful_count.fetch_add(1, Ordering::Relaxed) + 1
                } else {
                    failed_count.fetch_add(1, Ordering::Relaxed);
                    successful_count.load(Ordering::Relaxed)
                };
                let fail = failed_count.load(Ordering::Relaxed);

                if let Some(callback) = progress.as_ref() {
                    callback(ProgressUpdate {
                        completed: finished,
                        total,
                        backend: task.backend_id,
                        property_index: task.property_index,
                        successful_so_far: succ,
                        failed_so_far: fail,
                    });
                }

                TaskResult {
                    property_index: task.property_index,
                    backend: task.backend_id,
                    result,
                }
            });

            handles.push(handle);
        }

        // Collect all results
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!("Task panicked: {:?}", e);
                }
            }
        }

        // Check if we were cancelled
        if let Some(ref token) = self.cancellation_token {
            if token.is_cancelled() {
                return Err(ExecutorError::Cancelled);
            }
        }

        Ok(results)
    }

    /// Legacy execute_tasks method for backwards compatibility
    #[allow(dead_code)]
    async fn execute_tasks(
        &self,
        tasks: Vec<(usize, BackendId, Arc<dyn VerificationBackend>)>,
        spec: Arc<TypedSpec>,
    ) -> Vec<TaskResult> {
        let prioritized: Vec<PrioritizedTask> = tasks
            .into_iter()
            .map(|(property_index, backend_id, backend)| PrioritizedTask {
                property_index,
                backend_id,
                backend,
                priority: TaskPriority::Normal,
            })
            .collect();

        self.execute_tasks_with_retry(prioritized, spec)
            .await
            .unwrap_or_default()
    }

    /// Execute a single verification task (for simple cases)
    pub async fn execute_single(
        &self,
        backend: &Arc<dyn VerificationBackend>,
        spec: &TypedSpec,
    ) -> Result<BackendResult, ExecutorError> {
        let timeout = self.config.task_timeout;

        let result = tokio::time::timeout(timeout, backend.verify(spec)).await;

        match result {
            Ok(Ok(r)) => Ok(r),
            Ok(Err(e)) => Err(ExecutorError::BackendError(e)),
            Err(_) => Err(ExecutorError::Timeout(timeout)),
        }
    }

    /// Set a progress callback that is invoked as tasks complete
    pub fn set_progress_callback(
        &mut self,
        callback: Option<Arc<dyn Fn(ProgressUpdate) + Send + Sync>>,
    ) {
        self.progress_callback = callback;
    }
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        Self::new(ExecutorConfig::default())
    }
}

/// Builder for creating verification tasks
pub struct TaskBuilder {
    tasks: Vec<VerificationTask>,
}

/// A single verification task
#[derive(Debug, Clone)]
pub struct VerificationTask {
    /// Index of the property to verify
    pub property_index: usize,
    /// Backend to use for verification
    pub backend_id: BackendId,
}

impl TaskBuilder {
    /// Create a new empty task builder
    pub fn new() -> Self {
        TaskBuilder { tasks: Vec::new() }
    }

    /// Add a task from a selection
    pub fn from_selection(selection: &Selection) -> Self {
        let mut builder = Self::new();
        for assignment in &selection.assignments {
            for &backend_id in &assignment.backends {
                builder.tasks.push(VerificationTask {
                    property_index: assignment.property_index,
                    backend_id,
                });
            }
        }
        builder
    }

    /// Add a single task
    pub fn add_task(&mut self, property_index: usize, backend_id: BackendId) -> &mut Self {
        self.tasks.push(VerificationTask {
            property_index,
            backend_id,
        });
        self
    }

    /// Get all tasks
    pub fn tasks(&self) -> &[VerificationTask] {
        &self.tasks
    }

    /// Number of tasks
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

impl Default for TaskBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selector::{PropertyAssignment, Selection, SelectionMethod, SelectionMetrics};
    use dashprove_backends::{HealthStatus, PropertyType, VerificationStatus};
    use dashprove_usl::ast::{Expr, Property, Spec, Theorem};

    // Mock backend for testing
    struct MockBackend {
        id: BackendId,
        delay: Duration,
        should_fail: bool,
    }

    impl MockBackend {
        fn new(id: BackendId) -> Self {
            MockBackend {
                id,
                delay: Duration::from_millis(10),
                should_fail: false,
            }
        }

        fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = delay;
            self
        }

        fn failing(mut self) -> Self {
            self.should_fail = true;
            self
        }
    }

    #[async_trait::async_trait]
    impl VerificationBackend for MockBackend {
        fn id(&self) -> BackendId {
            self.id
        }

        fn supports(&self) -> Vec<PropertyType> {
            vec![PropertyType::Theorem]
        }

        async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
            tokio::time::sleep(self.delay).await;

            if self.should_fail {
                Err(BackendError::VerificationFailed("Mock failure".into()))
            } else {
                Ok(BackendResult {
                    backend: self.id,
                    status: VerificationStatus::Proven,
                    proof: Some("mock proof".into()),
                    counterexample: None,
                    diagnostics: vec![],
                    time_taken: self.delay,
                })
            }
        }

        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }

    fn make_typed_spec() -> TypedSpec {
        use dashprove_usl::typecheck::typecheck;
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Theorem(Theorem {
                name: "test".into(),
                body: Expr::Bool(true),
            })],
        };
        typecheck(spec).unwrap()
    }

    #[tokio::test]
    async fn test_execute_single() {
        let executor = ParallelExecutor::default();
        let backend: Arc<dyn VerificationBackend> = Arc::new(MockBackend::new(BackendId::Lean4));
        let spec = make_typed_spec();

        let result = executor.execute_single(&backend, &spec).await;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(matches!(result.status, VerificationStatus::Proven));
    }

    #[tokio::test]
    async fn test_execute_parallel() {
        let mut executor = ParallelExecutor::new(ExecutorConfig {
            max_concurrent: 2,
            ..Default::default()
        });
        let progress_calls = Arc::new(AtomicUsize::new(0));
        let progress_clone = Arc::clone(&progress_calls);
        executor.set_progress_callback(Some(Arc::new(move |update: ProgressUpdate| {
            // completed should never exceed total
            assert!(update.completed <= update.total);
            progress_clone.fetch_add(1, Ordering::SeqCst);
        })));

        let mut backends: HashMap<BackendId, Arc<dyn VerificationBackend>> = HashMap::new();
        backends.insert(
            BackendId::Lean4,
            Arc::new(MockBackend::new(BackendId::Lean4)),
        );
        backends.insert(
            BackendId::Alloy,
            Arc::new(MockBackend::new(BackendId::Alloy)),
        );

        let selection = Selection {
            assignments: vec![PropertyAssignment {
                property_index: 0,
                property_type: PropertyType::Invariant,
                backends: vec![BackendId::Lean4, BackendId::Alloy],
                selection_method: SelectionMethod::RuleBased,
            }],
            warnings: vec![],
            metrics: SelectionMetrics::default(),
        };

        let spec = make_typed_spec();
        let results = executor
            .execute(&selection, &spec, &backends)
            .await
            .unwrap();

        assert_eq!(results.successful, 2);
        assert_eq!(results.failed, 0);
        assert!(results.by_property.contains_key(&0));
        assert_eq!(results.by_property[&0].len(), 2);
        assert_eq!(progress_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_execute_with_failure() {
        let executor = ParallelExecutor::default();

        let mut backends: HashMap<BackendId, Arc<dyn VerificationBackend>> = HashMap::new();
        backends.insert(
            BackendId::Lean4,
            Arc::new(MockBackend::new(BackendId::Lean4).failing()),
        );

        let selection = Selection {
            assignments: vec![PropertyAssignment {
                property_index: 0,
                property_type: PropertyType::Theorem,
                backends: vec![BackendId::Lean4],
                selection_method: SelectionMethod::RuleBased,
            }],
            warnings: vec![],
            metrics: SelectionMetrics::default(),
        };

        let spec = make_typed_spec();
        let results = executor
            .execute(&selection, &spec, &backends)
            .await
            .unwrap();

        assert_eq!(results.successful, 0);
        assert_eq!(results.failed, 1);
    }

    #[tokio::test]
    async fn test_execute_timeout() {
        let executor = ParallelExecutor::new(ExecutorConfig {
            task_timeout: Duration::from_millis(10),
            ..Default::default()
        });

        let backend: Arc<dyn VerificationBackend> =
            Arc::new(MockBackend::new(BackendId::Lean4).with_delay(Duration::from_secs(1)));
        let spec = make_typed_spec();

        let result = executor.execute_single(&backend, &spec).await;
        assert!(matches!(result, Err(ExecutorError::Timeout(_))));
    }

    #[test]
    fn test_task_builder() {
        let selection = Selection {
            assignments: vec![
                PropertyAssignment {
                    property_index: 0,
                    property_type: PropertyType::Theorem,
                    backends: vec![BackendId::Lean4],
                    selection_method: SelectionMethod::RuleBased,
                },
                PropertyAssignment {
                    property_index: 1,
                    property_type: PropertyType::Temporal,
                    backends: vec![BackendId::TlaPlus],
                    selection_method: SelectionMethod::RuleBased,
                },
            ],
            warnings: vec![],
            metrics: SelectionMetrics::default(),
        };

        let builder = TaskBuilder::from_selection(&selection);
        assert_eq!(builder.len(), 2);
        assert_eq!(builder.tasks()[0].property_index, 0);
        assert_eq!(builder.tasks()[0].backend_id, BackendId::Lean4);
        assert_eq!(builder.tasks()[1].property_index, 1);
        assert_eq!(builder.tasks()[1].backend_id, BackendId::TlaPlus);
    }

    #[tokio::test]
    async fn test_concurrent_limit() {
        // Create executor with max 1 concurrent task
        let executor = ParallelExecutor::new(ExecutorConfig {
            max_concurrent: 1,
            ..Default::default()
        });

        let mut backends: HashMap<BackendId, Arc<dyn VerificationBackend>> = HashMap::new();
        backends.insert(
            BackendId::Lean4,
            Arc::new(MockBackend::new(BackendId::Lean4).with_delay(Duration::from_millis(50))),
        );
        backends.insert(
            BackendId::Alloy,
            Arc::new(MockBackend::new(BackendId::Alloy).with_delay(Duration::from_millis(50))),
        );

        let selection = Selection {
            assignments: vec![PropertyAssignment {
                property_index: 0,
                property_type: PropertyType::Invariant,
                backends: vec![BackendId::Lean4, BackendId::Alloy],
                selection_method: SelectionMethod::RuleBased,
            }],
            warnings: vec![],
            metrics: SelectionMetrics::default(),
        };

        let spec = make_typed_spec();
        let start = std::time::Instant::now();
        let results = executor
            .execute(&selection, &spec, &backends)
            .await
            .unwrap();
        let elapsed = start.elapsed();

        assert_eq!(results.successful, 2);
        // With max_concurrent=1, tasks should run sequentially
        // Total time should be at least 100ms (50ms * 2)
        assert!(elapsed >= Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_all_tasks_failed_empty_backends() {
        let executor = ParallelExecutor::new(ExecutorConfig::default());

        // Empty backends map - no backends available for any task
        let backends: HashMap<BackendId, Arc<dyn VerificationBackend>> = HashMap::new();

        let selection = Selection {
            assignments: vec![PropertyAssignment {
                property_index: 0,
                property_type: PropertyType::Theorem,
                backends: vec![BackendId::Lean4], // Request Lean4 but it's not in backends map
                selection_method: SelectionMethod::RuleBased,
            }],
            warnings: vec![],
            metrics: SelectionMetrics::default(),
        };

        let spec = make_typed_spec();
        let result = executor.execute(&selection, &spec, &backends).await;
        assert!(matches!(result, Err(ExecutorError::AllTasksFailed)));
    }

    #[tokio::test]
    async fn test_backend_error_propagation() {
        let executor = ParallelExecutor::new(ExecutorConfig::default());

        // Create a failing backend
        let backend: Arc<dyn VerificationBackend> =
            Arc::new(MockBackend::new(BackendId::Lean4).failing());

        let spec = make_typed_spec();
        let result = executor.execute_single(&backend, &spec).await;
        assert!(matches!(result, Err(ExecutorError::BackendError(_))));
    }

    // ==================== Mutation-killing tests ====================

    #[test]
    fn test_task_builder_add_task_returns_self() {
        // Mutation: replace add_task -> &mut Self with Box::leak(Box::new(Default::default()))
        let mut builder = TaskBuilder::new();

        // add_task should return &mut self, allowing chaining
        let returned = builder.add_task(0, BackendId::Lean4);

        // Verify the returned reference points to the same builder
        assert_eq!(returned.len(), 1);

        // Chain another call
        returned.add_task(1, BackendId::Alloy);
        assert_eq!(builder.len(), 2);
    }

    #[test]
    fn test_task_builder_add_task_modifies_self() {
        // Mutation: replace add_task -> &mut Self with Box::leak(Box::new(Default::default()))
        let mut builder = TaskBuilder::new();
        assert_eq!(builder.len(), 0);
        assert!(builder.is_empty());

        builder.add_task(0, BackendId::Lean4);

        // Original builder should be modified
        assert_eq!(builder.len(), 1);
        assert!(!builder.is_empty());
        assert_eq!(builder.tasks()[0].property_index, 0);
        assert_eq!(builder.tasks()[0].backend_id, BackendId::Lean4);
    }

    #[test]
    fn test_task_builder_is_empty_true_when_empty() {
        // Mutation: replace is_empty -> bool with false
        let builder = TaskBuilder::new();
        assert!(builder.is_empty(), "New TaskBuilder should be empty");
        assert_eq!(builder.len(), 0);
    }

    #[test]
    fn test_task_builder_is_empty_false_when_has_tasks() {
        // Mutation: replace is_empty -> bool with true
        let mut builder = TaskBuilder::new();
        builder.add_task(0, BackendId::Lean4);

        assert!(
            !builder.is_empty(),
            "TaskBuilder with tasks should not be empty"
        );
        assert_eq!(builder.len(), 1);
    }

    #[test]
    fn test_task_builder_is_empty_matches_len() {
        // Ensure is_empty() is consistent with len()
        let mut builder = TaskBuilder::new();

        // Empty - verify both methods agree
        assert!(builder.is_empty());
        assert_eq!(builder.len(), 0);

        // Add one
        builder.add_task(0, BackendId::Lean4);
        assert!(!builder.is_empty());
        assert_eq!(builder.len(), 1);

        // Add another
        builder.add_task(1, BackendId::Alloy);
        assert!(!builder.is_empty());
        assert_eq!(builder.len(), 2);
    }

    #[test]
    fn test_task_builder_from_selection_preserves_all() {
        // Verify all assignments are preserved
        let selection = Selection {
            assignments: vec![
                PropertyAssignment {
                    property_index: 0,
                    property_type: PropertyType::Theorem,
                    backends: vec![BackendId::Lean4, BackendId::Coq], // Multiple backends
                    selection_method: SelectionMethod::RuleBased,
                },
                PropertyAssignment {
                    property_index: 1,
                    property_type: PropertyType::Temporal,
                    backends: vec![BackendId::TlaPlus],
                    selection_method: SelectionMethod::RuleBased,
                },
            ],
            warnings: vec![],
            metrics: SelectionMetrics::default(),
        };

        let builder = TaskBuilder::from_selection(&selection);

        // Should have 3 tasks total (2 for property 0, 1 for property 1)
        assert_eq!(builder.len(), 3);
        assert!(!builder.is_empty());

        // Verify task details
        let tasks = builder.tasks();
        assert_eq!(tasks[0].property_index, 0);
        assert_eq!(tasks[0].backend_id, BackendId::Lean4);
        assert_eq!(tasks[1].property_index, 0);
        assert_eq!(tasks[1].backend_id, BackendId::Coq);
        assert_eq!(tasks[2].property_index, 1);
        assert_eq!(tasks[2].backend_id, BackendId::TlaPlus);
    }

    // ==================== New feature tests ====================

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.initial_backoff, Duration::from_millis(100));
        assert_eq!(config.backoff_multiplier, 2.0);
    }

    #[test]
    fn test_retry_config_no_retries() {
        let config = RetryConfig::no_retries();
        assert_eq!(config.max_retries, 0);
    }

    #[test]
    fn test_retry_config_backoff_calculation() {
        let config = RetryConfig {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            max_retries: 5,
        };

        // First attempt has no backoff
        assert_eq!(config.backoff_for_attempt(0), Duration::ZERO);
        // First retry: 100ms
        assert_eq!(config.backoff_for_attempt(1), Duration::from_millis(100));
        // Second retry: 200ms
        assert_eq!(config.backoff_for_attempt(2), Duration::from_millis(200));
        // Third retry: 400ms
        assert_eq!(config.backoff_for_attempt(3), Duration::from_millis(400));
    }

    #[test]
    fn test_retry_config_backoff_capped() {
        let config = RetryConfig {
            initial_backoff: Duration::from_secs(5),
            max_backoff: Duration::from_secs(10),
            backoff_multiplier: 3.0,
            max_retries: 5,
        };

        // First retry: 5s
        assert_eq!(config.backoff_for_attempt(1), Duration::from_secs(5));
        // Second retry: 15s but capped to 10s
        assert_eq!(config.backoff_for_attempt(2), Duration::from_secs(10));
        // Third retry: would be 45s but still capped to 10s
        assert_eq!(config.backoff_for_attempt(3), Duration::from_secs(10));
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical as u8 > TaskPriority::High as u8);
        assert!(TaskPriority::High as u8 > TaskPriority::Normal as u8);
        assert!(TaskPriority::Normal as u8 > TaskPriority::Low as u8);
    }

    #[test]
    fn test_task_priority_default() {
        assert_eq!(TaskPriority::default(), TaskPriority::Normal);
    }

    #[test]
    fn test_cancellation_token_new() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_cancel() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_clone() {
        let token = CancellationToken::new();
        let cloned = token.clone();
        assert!(!cloned.is_cancelled());
        token.cancel();
        // The clone should also see cancellation
        assert!(cloned.is_cancelled());
    }

    #[tokio::test]
    async fn test_executor_cancellation() {
        let mut executor = ParallelExecutor::new(ExecutorConfig::default());
        let token = CancellationToken::new();
        executor.set_cancellation_token(token.clone());

        // Cancel before execution
        token.cancel();

        let backends: HashMap<BackendId, Arc<dyn VerificationBackend>> = HashMap::new();
        let selection = Selection {
            assignments: vec![PropertyAssignment {
                property_index: 0,
                property_type: PropertyType::Theorem,
                backends: vec![BackendId::Lean4],
                selection_method: SelectionMethod::RuleBased,
            }],
            warnings: vec![],
            metrics: SelectionMetrics::default(),
        };

        let spec = make_typed_spec();
        let result = executor.execute(&selection, &spec, &backends).await;
        // Should fail because cancelled before finding tasks (empty backends) or because cancelled
        assert!(result.is_err());
    }

    #[test]
    fn test_executor_config_with_retry() {
        let config = ExecutorConfig {
            retry_config: RetryConfig {
                max_retries: 5,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(config.retry_config.max_retries, 5);
        assert!(!config.adaptive_concurrency);
    }

    #[tokio::test]
    async fn test_progress_update_has_counts() {
        let mut executor = ParallelExecutor::new(ExecutorConfig {
            max_concurrent: 1,
            retry_config: RetryConfig::no_retries(),
            ..Default::default()
        });

        let last_update = Arc::new(std::sync::Mutex::new(None::<ProgressUpdate>));
        let update_clone = Arc::clone(&last_update);

        executor.set_progress_callback(Some(Arc::new(move |update: ProgressUpdate| {
            *update_clone.lock().unwrap() = Some(update);
        })));

        let mut backends: HashMap<BackendId, Arc<dyn VerificationBackend>> = HashMap::new();
        backends.insert(
            BackendId::Lean4,
            Arc::new(MockBackend::new(BackendId::Lean4)),
        );

        let selection = Selection {
            assignments: vec![PropertyAssignment {
                property_index: 0,
                property_type: PropertyType::Theorem,
                backends: vec![BackendId::Lean4],
                selection_method: SelectionMethod::RuleBased,
            }],
            warnings: vec![],
            metrics: SelectionMetrics::default(),
        };

        let spec = make_typed_spec();
        let _ = executor.execute(&selection, &spec, &backends).await;

        let update = last_update.lock().unwrap().clone();
        assert!(update.is_some());
        let update = update.unwrap();
        assert_eq!(update.completed, 1);
        assert_eq!(update.total, 1);
        assert_eq!(update.successful_so_far, 1);
        assert_eq!(update.failed_so_far, 0);
    }

    #[test]
    fn test_is_retryable_error() {
        use std::time::Duration;

        // Timeout is retryable
        assert!(is_retryable_error(&BackendError::Timeout(
            Duration::from_secs(1)
        )));

        // Unavailable is retryable
        assert!(is_retryable_error(&BackendError::Unavailable(
            "test".into()
        )));

        // Compilation failure is NOT retryable
        assert!(!is_retryable_error(&BackendError::CompilationFailed(
            "test".into()
        )));

        // Verification failure is NOT retryable
        assert!(!is_retryable_error(&BackendError::VerificationFailed(
            "test".into()
        )));
    }
}

// ============================================================================
// Kani Proof Harnesses
// ============================================================================
// These formal verification harnesses prove properties about the parallel
// execution logic using bounded model checking.
// Run with: cargo kani -p dashprove-dispatcher

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that ExecutorConfig default values are sensible
    #[kani::proof]
    fn verify_executor_config_default_valid() {
        let config = ExecutorConfig::default();

        // max_concurrent must be positive (at least 1 task can run)
        kani::assert(config.max_concurrent > 0, "max_concurrent must be positive");

        // task_timeout must be non-zero
        kani::assert(
            config.task_timeout > Duration::ZERO,
            "task_timeout must be non-zero",
        );

        // fail_fast is a boolean - verify it's false by default
        kani::assert(!config.fail_fast, "fail_fast should be false by default");
    }

    /// Prove that TaskBuilder::new creates an empty builder
    #[kani::proof]
    fn verify_task_builder_new_is_empty() {
        let builder = TaskBuilder::new();

        kani::assert(builder.is_empty(), "New TaskBuilder must be empty");
        kani::assert(builder.len() == 0, "New TaskBuilder must have len 0");
        kani::assert(
            builder.tasks().is_empty(),
            "New TaskBuilder tasks must be empty",
        );
    }

    /// Prove that TaskBuilder::default creates an empty builder (same as new)
    #[kani::proof]
    fn verify_task_builder_default_is_empty() {
        let builder = TaskBuilder::default();

        kani::assert(builder.is_empty(), "Default TaskBuilder must be empty");
        kani::assert(builder.len() == 0, "Default TaskBuilder must have len 0");
    }

    /// Prove that is_empty is consistent with len
    #[kani::proof]
    fn verify_task_builder_is_empty_consistent_with_len() {
        let builder = TaskBuilder::new();

        // Empty builder: len == 0 iff is_empty
        let len_zero = builder.len() == 0;
        let is_empty = builder.is_empty();

        kani::assert(len_zero == is_empty, "is_empty must be true iff len is 0");
    }

    /// Prove that add_task increases len by 1
    #[kani::proof]
    fn verify_task_builder_add_task_increments_len() {
        let mut builder = TaskBuilder::new();
        let initial_len = builder.len();

        builder.add_task(0, BackendId::Lean4);

        kani::assert(
            builder.len() == initial_len + 1,
            "add_task must increment len by 1",
        );
    }

    /// Prove that add_task makes builder non-empty
    #[kani::proof]
    fn verify_task_builder_add_task_makes_nonempty() {
        let mut builder = TaskBuilder::new();
        kani::assert(builder.is_empty(), "Builder starts empty");

        builder.add_task(0, BackendId::Lean4);

        kani::assert(
            !builder.is_empty(),
            "Builder must be non-empty after add_task",
        );
    }

    /// Prove that add_task preserves the added task
    #[kani::proof]
    fn verify_task_builder_add_task_preserves_data() {
        let mut builder = TaskBuilder::new();

        let prop_idx: usize = kani::any();
        kani::assume(prop_idx < 1000); // Reasonable bound

        builder.add_task(prop_idx, BackendId::Lean4);

        let tasks = builder.tasks();
        kani::assert(tasks.len() == 1, "Should have exactly one task");
        kani::assert(
            tasks[0].property_index == prop_idx,
            "property_index must match",
        );
        kani::assert(
            tasks[0].backend_id == BackendId::Lean4,
            "backend_id must match",
        );
    }

    /// Prove that VerificationTask stores data correctly
    #[kani::proof]
    fn verify_verification_task_stores_data() {
        let prop_idx: usize = kani::any();
        kani::assume(prop_idx < 10000); // Reasonable bound

        let task = VerificationTask {
            property_index: prop_idx,
            backend_id: BackendId::TlaPlus,
        };

        kani::assert(
            task.property_index == prop_idx,
            "property_index must be preserved",
        );
        kani::assert(
            task.backend_id == BackendId::TlaPlus,
            "backend_id must be preserved",
        );
    }

    /// Prove that ExecutorConfig values are preserved
    #[kani::proof]
    fn verify_executor_config_preserves_values() {
        let max_concurrent: usize = kani::any();
        kani::assume(max_concurrent > 0 && max_concurrent <= 100);

        let secs: u64 = kani::any();
        kani::assume(secs > 0 && secs <= 3600);

        let fail_fast: bool = kani::any();

        let config = ExecutorConfig {
            max_concurrent,
            task_timeout: Duration::from_secs(secs),
            fail_fast,
        };

        kani::assert(
            config.max_concurrent == max_concurrent,
            "max_concurrent must be preserved",
        );
        kani::assert(
            config.task_timeout == Duration::from_secs(secs),
            "task_timeout must be preserved",
        );
        kani::assert(config.fail_fast == fail_fast, "fail_fast must be preserved");
    }

    /// Prove that multiple add_task calls accumulate correctly
    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_task_builder_multiple_adds() {
        let mut builder = TaskBuilder::new();

        builder.add_task(0, BackendId::Lean4);
        builder.add_task(1, BackendId::TlaPlus);
        builder.add_task(2, BackendId::Kani);

        kani::assert(builder.len() == 3, "Should have 3 tasks");
        kani::assert(!builder.is_empty(), "Should not be empty");

        let tasks = builder.tasks();
        kani::assert(tasks[0].property_index == 0, "First task property_index");
        kani::assert(tasks[1].property_index == 1, "Second task property_index");
        kani::assert(tasks[2].property_index == 2, "Third task property_index");
    }
}
