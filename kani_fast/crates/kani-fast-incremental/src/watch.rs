//! Watch mode for continuous verification
//!
//! This module provides file watching capabilities that trigger re-verification
//! when source files change. Changes are debounced to avoid excessive verification
//! during rapid editing.

use crate::config::IncrementalConfig;
use crate::engine::{DimacsResult, IncrementalBmc, IncrementalError};
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::watch;
use tracing::{debug, warn};

/// Errors that can occur during watch mode
#[derive(Debug, Error)]
pub enum WatchError {
    #[error("Failed to create file watcher: {0}")]
    WatcherCreation(#[from] notify::Error),

    #[error("Incremental engine error: {0}")]
    Engine(#[from] IncrementalError),

    #[error("Watch mode was stopped")]
    Stopped,
}

/// Events emitted by the watch mode
#[derive(Debug, Clone)]
pub enum WatchEvent {
    /// Files have changed and verification is starting
    VerificationStarted { changed_files: Vec<PathBuf> },

    /// Verification completed
    VerificationCompleted {
        result: WatchResult,
        changed_files: Vec<PathBuf>,
    },

    /// An error occurred during verification
    VerificationError { error: String },

    /// Watch mode is shutting down
    Shutdown,
}

/// Result of a watch-mode verification
#[derive(Debug, Clone)]
pub struct WatchResult {
    /// Whether the verification proved the property
    pub proven: bool,
    /// Whether a counterexample was found
    pub counterexample: bool,
    /// Number of files that changed
    pub files_changed: usize,
    /// Verification duration
    pub duration: Duration,
    /// Whether the result came from cache
    pub from_cache: bool,
}

impl From<DimacsResult> for WatchResult {
    fn from(r: DimacsResult) -> Self {
        WatchResult {
            proven: r.proven,
            counterexample: r.satisfiable,
            files_changed: 1,
            duration: r.duration,
            from_cache: r.from_cache,
        }
    }
}

/// Watch mode controller
pub struct WatchMode {
    /// Project root path
    project_root: PathBuf,
    /// Configuration
    config: IncrementalConfig,
    /// Shutdown signal sender
    shutdown_tx: Option<watch::Sender<bool>>,
}

impl WatchMode {
    /// Create a new watch mode controller
    pub fn new(project_root: impl Into<PathBuf>, config: IncrementalConfig) -> Self {
        Self {
            project_root: project_root.into(),
            config,
            shutdown_tx: None,
        }
    }

    /// Start watching and return a receiver for events
    ///
    /// This function spawns a background task that watches for file changes
    /// and triggers verification. Use the returned receiver to get events.
    pub fn start(&mut self) -> Result<(Receiver<WatchEvent>, watch::Receiver<bool>), WatchError> {
        let (event_tx, event_rx) = mpsc::channel();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        self.shutdown_tx = Some(shutdown_tx);

        let project_root = self.project_root.clone();
        let config = self.config.clone();
        let debounce_duration = config.watch_debounce;

        // Create file watcher
        let (file_tx, file_rx): (Sender<Result<Event, notify::Error>>, _) = mpsc::channel();
        let mut watcher = RecommendedWatcher::new(
            move |res| {
                let _ = file_tx.send(res);
            },
            notify::Config::default(),
        )?;

        watcher.watch(&project_root, RecursiveMode::Recursive)?;

        // Spawn watch task
        let shutdown = shutdown_rx.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime");

            rt.block_on(async {
                // Keep watcher alive
                let _watcher = watcher;

                // Debounce state
                let mut pending_changes: HashSet<PathBuf> = HashSet::new();
                let mut last_change: Option<Instant> = None;

                loop {
                    // Check for shutdown
                    if *shutdown.borrow() {
                        let _ = event_tx.send(WatchEvent::Shutdown);
                        break;
                    }

                    // Wait for file changes with timeout
                    match file_rx.recv_timeout(Duration::from_millis(50)) {
                        Ok(Ok(event)) => {
                            // Filter and collect changed files
                            for path in event.paths {
                                if should_watch(&path) {
                                    pending_changes.insert(path);
                                }
                            }
                            last_change = Some(Instant::now());
                        }
                        Ok(Err(e)) => {
                            warn!("File watch error: {:?}", e);
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            // Check if we should trigger verification
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            let _ = event_tx.send(WatchEvent::Shutdown);
                            break;
                        }
                    }

                    // Check debounce timer
                    if let Some(last) = last_change {
                        if last.elapsed() >= debounce_duration && !pending_changes.is_empty() {
                            let changed: Vec<PathBuf> = pending_changes.drain().collect();
                            last_change = None;

                            debug!("Files changed: {:?}", changed);
                            let _ = event_tx.send(WatchEvent::VerificationStarted {
                                changed_files: changed.clone(),
                            });

                            // Run verification
                            match run_verification(&project_root, &config).await {
                                Ok(result) => {
                                    let _ = event_tx.send(WatchEvent::VerificationCompleted {
                                        result,
                                        changed_files: changed,
                                    });
                                }
                                Err(e) => {
                                    let _ = event_tx.send(WatchEvent::VerificationError {
                                        error: e.to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
            });
        });

        Ok((event_rx, shutdown_rx))
    }

    /// Stop watch mode
    pub fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(true);
        }
    }
}

impl Drop for WatchMode {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Check if a path should be watched
fn should_watch(path: &Path) -> bool {
    // Only care about Rust files and CNF files
    path.extension()
        .is_some_and(|ext| matches!(ext.to_str(), Some("rs" | "cnf")))
}

/// Run verification on changed files
async fn run_verification(
    project_root: &Path,
    config: &IncrementalConfig,
) -> Result<WatchResult, IncrementalError> {
    let mut engine = IncrementalBmc::new(project_root, config.clone())?;

    // For now, run full project verification
    // In the future, we could be smarter about which files to verify
    let result = engine.verify().await?;

    Ok(WatchResult {
        proven: result.proven,
        counterexample: !result.proven,
        files_changed: 1,
        duration: result.duration,
        from_cache: result.from_cache,
    })
}

/// Builder for WatchMode with fluent API
pub struct WatchModeBuilder {
    project_root: PathBuf,
    config: IncrementalConfig,
}

impl WatchModeBuilder {
    /// Create a new builder
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            config: IncrementalConfig::default(),
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: IncrementalConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the debounce duration
    pub fn debounce(mut self, duration: Duration) -> Self {
        self.config.watch_debounce = duration;
        self
    }

    /// Build the watch mode controller
    pub fn build(self) -> WatchMode {
        WatchMode::new(self.project_root, self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::TempDir;

    #[test]
    fn test_watch_result_from_dimacs() {
        let dimacs = DimacsResult {
            satisfiable: false,
            proven: true,
            learned_clauses: 10,
            cached_clauses_used: 5,
            duration: Duration::from_millis(100),
            from_cache: false,
        };

        let watch_result: WatchResult = dimacs.into();
        assert!(watch_result.proven);
        assert!(!watch_result.counterexample);
        assert!(!watch_result.from_cache);
    }

    #[test]
    fn test_watch_result_from_dimacs_sat() {
        let dimacs = DimacsResult {
            satisfiable: true,
            proven: false,
            learned_clauses: 0,
            cached_clauses_used: 0,
            duration: Duration::from_millis(50),
            from_cache: false,
        };

        let watch_result: WatchResult = dimacs.into();
        assert!(!watch_result.proven);
        assert!(watch_result.counterexample);
        assert_eq!(watch_result.files_changed, 1);
        assert_eq!(watch_result.duration, Duration::from_millis(50));
    }

    #[test]
    fn test_watch_result_from_dimacs_cached() {
        let dimacs = DimacsResult {
            satisfiable: false,
            proven: true,
            learned_clauses: 0,
            cached_clauses_used: 100,
            duration: Duration::from_millis(1),
            from_cache: true,
        };

        let watch_result: WatchResult = dimacs.into();
        assert!(watch_result.from_cache);
        assert!(watch_result.proven);
    }

    #[test]
    fn test_watch_result_fields() {
        let result = WatchResult {
            proven: true,
            counterexample: false,
            files_changed: 5,
            duration: Duration::from_secs(2),
            from_cache: false,
        };

        assert!(result.proven);
        assert!(!result.counterexample);
        assert_eq!(result.files_changed, 5);
        assert_eq!(result.duration, Duration::from_secs(2));
        assert!(!result.from_cache);
    }

    #[test]
    fn test_watch_result_clone() {
        let result = WatchResult {
            proven: true,
            counterexample: false,
            files_changed: 3,
            duration: Duration::from_millis(500),
            from_cache: true,
        };

        let cloned = result.clone();
        assert_eq!(cloned.proven, result.proven);
        assert_eq!(cloned.counterexample, result.counterexample);
        assert_eq!(cloned.files_changed, result.files_changed);
        assert_eq!(cloned.duration, result.duration);
        assert_eq!(cloned.from_cache, result.from_cache);
    }

    #[test]
    fn test_watch_result_debug() {
        let result = WatchResult {
            proven: true,
            counterexample: false,
            files_changed: 1,
            duration: Duration::from_millis(100),
            from_cache: false,
        };

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("WatchResult"));
        assert!(debug_str.contains("proven"));
        assert!(debug_str.contains("true"));
    }

    #[test]
    fn test_watch_mode_builder() {
        let temp_dir = TempDir::new().unwrap();
        let watch_mode = WatchModeBuilder::new(temp_dir.path())
            .debounce(Duration::from_millis(200))
            .build();

        assert_eq!(watch_mode.project_root, temp_dir.path());
    }

    #[test]
    fn test_watch_mode_builder_with_config() {
        let temp_dir = TempDir::new().unwrap();
        let config = IncrementalConfig {
            watch_debounce: Duration::from_millis(1000),
            ..Default::default()
        };

        let watch_mode = WatchModeBuilder::new(temp_dir.path())
            .config(config.clone())
            .build();

        assert_eq!(
            watch_mode.config.watch_debounce,
            Duration::from_millis(1000)
        );
    }

    #[test]
    fn test_watch_mode_builder_debounce_override() {
        let temp_dir = TempDir::new().unwrap();

        let watch_mode = WatchModeBuilder::new(temp_dir.path())
            .debounce(Duration::from_millis(100))
            .debounce(Duration::from_millis(300))
            .build();

        assert_eq!(watch_mode.config.watch_debounce, Duration::from_millis(300));
    }

    #[test]
    fn test_should_watch() {
        assert!(should_watch(Path::new("src/lib.rs")));
        assert!(should_watch(Path::new("test.cnf")));
        assert!(!should_watch(Path::new("readme.txt")));
        assert!(!should_watch(Path::new("Cargo.toml")));
    }

    #[test]
    fn test_should_watch_rust_files() {
        assert!(should_watch(Path::new("main.rs")));
        assert!(should_watch(Path::new("src/module.rs")));
        assert!(should_watch(Path::new("tests/integration.rs")));
        assert!(should_watch(Path::new("deep/nested/path/file.rs")));
    }

    #[test]
    fn test_should_watch_cnf_files() {
        assert!(should_watch(Path::new("formula.cnf")));
        assert!(should_watch(Path::new("benchmarks/hard.cnf")));
        assert!(should_watch(Path::new("test.cnf")));
    }

    #[test]
    fn test_should_watch_excluded_files() {
        assert!(!should_watch(Path::new("Cargo.lock")));
        assert!(!should_watch(Path::new("README.md")));
        assert!(!should_watch(Path::new(".gitignore")));
        assert!(!should_watch(Path::new("script.py")));
        assert!(!should_watch(Path::new("data.json")));
        assert!(!should_watch(Path::new("image.png")));
    }

    #[test]
    fn test_should_watch_no_extension() {
        assert!(!should_watch(Path::new("Makefile")));
        assert!(!should_watch(Path::new("LICENSE")));
        assert!(!should_watch(Path::new("Dockerfile")));
    }

    #[test]
    fn test_watch_mode_create() {
        let temp_dir = TempDir::new().unwrap();
        let watch_mode = WatchMode::new(temp_dir.path(), IncrementalConfig::default());
        assert!(watch_mode.shutdown_tx.is_none());
    }

    #[test]
    fn test_watch_mode_create_with_custom_config() {
        let temp_dir = TempDir::new().unwrap();
        let config = IncrementalConfig {
            watch_mode: true,
            watch_debounce: Duration::from_millis(100),
            ..Default::default()
        };

        let watch_mode = WatchMode::new(temp_dir.path(), config);
        assert_eq!(watch_mode.project_root, temp_dir.path());
        assert!(watch_mode.config.watch_mode);
        assert_eq!(watch_mode.config.watch_debounce, Duration::from_millis(100));
    }

    #[test]
    fn test_watch_mode_stop_without_start() {
        let temp_dir = TempDir::new().unwrap();
        let mut watch_mode = WatchMode::new(temp_dir.path(), IncrementalConfig::default());

        // Should not panic when stopping without starting
        watch_mode.stop();
        assert!(watch_mode.shutdown_tx.is_none());
    }

    #[test]
    fn test_watch_event_verification_started() {
        let event = WatchEvent::VerificationStarted {
            changed_files: vec![PathBuf::from("src/lib.rs")],
        };

        if let WatchEvent::VerificationStarted { changed_files } = event {
            assert_eq!(changed_files.len(), 1);
            assert_eq!(changed_files[0], PathBuf::from("src/lib.rs"));
        } else {
            panic!("Expected VerificationStarted event");
        }
    }

    #[test]
    fn test_watch_event_verification_started_multiple_files() {
        let event = WatchEvent::VerificationStarted {
            changed_files: vec![
                PathBuf::from("src/lib.rs"),
                PathBuf::from("src/main.rs"),
                PathBuf::from("tests/test.rs"),
            ],
        };

        if let WatchEvent::VerificationStarted { changed_files } = event {
            assert_eq!(changed_files.len(), 3);
        } else {
            panic!("Expected VerificationStarted event");
        }
    }

    #[test]
    fn test_watch_event_verification_completed() {
        let result = WatchResult {
            proven: true,
            counterexample: false,
            files_changed: 1,
            duration: Duration::from_millis(500),
            from_cache: false,
        };

        let event = WatchEvent::VerificationCompleted {
            result: result.clone(),
            changed_files: vec![PathBuf::from("src/lib.rs")],
        };

        if let WatchEvent::VerificationCompleted {
            result: r,
            changed_files,
        } = event
        {
            assert!(r.proven);
            assert_eq!(changed_files.len(), 1);
        } else {
            panic!("Expected VerificationCompleted event");
        }
    }

    #[test]
    fn test_watch_event_verification_error() {
        let event = WatchEvent::VerificationError {
            error: "Solver timeout".to_string(),
        };

        if let WatchEvent::VerificationError { error } = event {
            assert_eq!(error, "Solver timeout");
        } else {
            panic!("Expected VerificationError event");
        }
    }

    #[test]
    fn test_watch_event_shutdown() {
        let event = WatchEvent::Shutdown;

        if let WatchEvent::Shutdown = event {
            // Success
        } else {
            panic!("Expected Shutdown event");
        }
    }

    #[test]
    fn test_watch_event_debug() {
        let event = WatchEvent::VerificationStarted {
            changed_files: vec![PathBuf::from("test.rs")],
        };
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("VerificationStarted"));
        assert!(debug_str.contains("test.rs"));
    }

    #[test]
    fn test_watch_event_clone() {
        let event = WatchEvent::VerificationStarted {
            changed_files: vec![PathBuf::from("src/lib.rs")],
        };

        let cloned = event.clone();
        if let WatchEvent::VerificationStarted { changed_files } = cloned {
            assert_eq!(changed_files.len(), 1);
        } else {
            panic!("Clone failed");
        }
    }

    #[test]
    fn test_watch_error_watcher_creation() {
        // Create a notify error and wrap it
        let notify_err = notify::Error::generic("test error");
        let watch_err = WatchError::WatcherCreation(notify_err);

        let err_str = format!("{}", watch_err);
        assert!(err_str.contains("Failed to create file watcher"));
    }

    #[test]
    fn test_watch_error_stopped() {
        let err = WatchError::Stopped;
        let err_str = format!("{}", err);
        assert!(err_str.contains("Watch mode was stopped"));
    }

    #[test]
    fn test_watch_error_debug() {
        let err = WatchError::Stopped;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Stopped"));
    }
}
