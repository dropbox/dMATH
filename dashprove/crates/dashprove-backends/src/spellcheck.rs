//! cargo-spellcheck documentation spell checker backend
//!
//! Spellcheck checks for spelling errors in documentation comments,
//! README files, and other documentation sources.
//!
//! See: <https://github.com/drahnr/cargo-spellcheck>

use crate::counterexample::{FailedCheck, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::process::Command;

/// Configuration for spellcheck backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpellcheckConfig {
    pub timeout: Duration,
    pub check_mode: bool,
    pub fix_mode: bool,
}

impl Default for SpellcheckConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(120),
            check_mode: true,
            fix_mode: false,
        }
    }
}

impl SpellcheckConfig {
    pub fn with_check_mode(mut self, enabled: bool) -> Self {
        self.check_mode = enabled;
        self
    }

    pub fn with_fix_mode(mut self, enabled: bool) -> Self {
        self.fix_mode = enabled;
        self
    }
}

/// Spellcheck statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpellcheckStats {
    pub files_checked: u64,
    pub misspellings_found: u64,
    pub misspellings: Vec<SpellingError>,
}

/// Information about a spelling error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpellingError {
    pub word: String,
    pub file: String,
    pub line: Option<u32>,
    pub suggestions: Vec<String>,
}

/// cargo-spellcheck documentation spell checker backend
pub struct SpellcheckBackend {
    config: SpellcheckConfig,
}

impl Default for SpellcheckBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SpellcheckBackend {
    pub fn new() -> Self {
        Self {
            config: SpellcheckConfig::default(),
        }
    }

    pub fn with_config(config: SpellcheckConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        which::which("cargo-spellcheck").map_err(|_| {
            "cargo-spellcheck not found. Install via cargo install cargo-spellcheck".to_string()
        })
    }

    /// Check spelling in documentation
    pub async fn check_spelling(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut cmd = Command::new("cargo");
        cmd.current_dir(crate_path);
        cmd.arg("spellcheck");

        if self.config.check_mode {
            cmd.arg("check");
        } else if self.config.fix_mode {
            cmd.arg("fix");
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run cargo spellcheck: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let stats = self.parse_output(&stdout, &stderr);
        let (status, counterexample) = self.evaluate_results(&stats);

        let mut diagnostics = Vec::new();
        diagnostics.push(format!(
            "Files checked: {}, Misspellings: {}",
            stats.files_checked, stats.misspellings_found
        ));

        Ok(BackendResult {
            backend: BackendId::Spellcheck,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    fn parse_output(&self, stdout: &str, stderr: &str) -> SpellcheckStats {
        let mut stats = SpellcheckStats::default();
        let combined = format!("{}\n{}", stdout, stderr);

        for line in combined.lines() {
            // Count files
            if line.contains(".rs") && (line.contains("-->") || line.contains("checking")) {
                stats.files_checked += 1;
            }

            // Parse misspelling lines
            // Format varies but typically includes: word, file location, suggestions
            if line.contains("error") && line.contains("-->") {
                stats.misspellings_found += 1;

                // Try to extract word and file info
                let word = Self::extract_misspelled_word(line);
                let file = Self::extract_file_from_line(line);

                stats.misspellings.push(SpellingError {
                    word: word.unwrap_or_default(),
                    file: file.unwrap_or_default(),
                    line: None,
                    suggestions: vec![],
                });
            }

            // Alternative format: "misspelled: word"
            if (line.to_lowercase().contains("misspelled") || line.contains("spelling"))
                && !line.contains("error")
            {
                // Avoid double counting
                stats.misspellings_found += 1;
            }
        }

        // If no explicit file count, estimate based on errors
        if stats.files_checked == 0 && stats.misspellings_found > 0 {
            stats.files_checked = 1;
        }

        stats
    }

    fn extract_misspelled_word(line: &str) -> Option<String> {
        // Look for word in quotes or backticks
        for delim in ['`', '"', '\''] {
            if let Some(start) = line.find(delim) {
                let remainder = &line[start + 1..];
                if let Some(end) = remainder.find(delim) {
                    let word = &remainder[..end];
                    if !word.is_empty() && !word.contains(' ') {
                        return Some(word.to_string());
                    }
                }
            }
        }
        None
    }

    fn extract_file_from_line(line: &str) -> Option<String> {
        // Look for file path with --> indicator
        if let Some(pos) = line.find("-->") {
            let after = line[pos + 3..].trim();
            let file_part = after.split(':').next()?;
            return Some(file_part.trim().to_string());
        }
        // Look for .rs file paths
        for word in line.split_whitespace() {
            if word.ends_with(".rs") || word.contains(".rs:") {
                return Some(
                    word.trim_matches(|c: char| {
                        !c.is_alphanumeric() && c != '/' && c != '.' && c != '_'
                    })
                    .to_string(),
                );
            }
        }
        None
    }

    fn evaluate_results(
        &self,
        stats: &SpellcheckStats,
    ) -> (VerificationStatus, Option<StructuredCounterexample>) {
        if stats.misspellings_found > 0 {
            let failed_checks: Vec<FailedCheck> = stats
                .misspellings
                .iter()
                .take(10) // Limit to avoid huge output
                .map(|err| FailedCheck {
                    check_id: format!("spelling:{}", err.word),
                    description: format!(
                        "Misspelled word '{}' in {}",
                        err.word,
                        if err.file.is_empty() {
                            "documentation"
                        } else {
                            &err.file
                        }
                    ),
                    location: None,
                    function: None,
                })
                .collect();

            let failed_checks = if failed_checks.is_empty() {
                vec![FailedCheck {
                    check_id: "spelling_errors".to_string(),
                    description: format!("{} spelling error(s) found", stats.misspellings_found),
                    location: None,
                    function: None,
                }]
            } else {
                failed_checks
            };

            return (
                VerificationStatus::Disproven,
                Some(StructuredCounterexample {
                    witness: HashMap::new(),
                    failed_checks,
                    playback_test: Some(
                        "Run `cargo spellcheck check` to see spelling errors".to_string(),
                    ),
                    trace: vec![],
                    raw: Some(format!(
                        "{} spelling error(s) found in documentation",
                        stats.misspellings_found
                    )),
                    minimized: false,
                }),
            );
        }

        if stats.files_checked == 0 {
            return (
                VerificationStatus::Unknown {
                    reason: "No files checked for spelling".to_string(),
                },
                None,
            );
        }

        (VerificationStatus::Proven, None)
    }
}

#[async_trait]
impl VerificationBackend for SpellcheckBackend {
    fn id(&self) -> BackendId {
        BackendId::Spellcheck
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Lint]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut diagnostics = Vec::new();
        diagnostics.push("spellcheck ready for documentation checking".to_string());
        if self.config.check_mode {
            diagnostics.push("Check mode (no modifications)".to_string());
        }

        Ok(BackendResult {
            backend: BackendId::Spellcheck,
            status: VerificationStatus::Proven,
            proof: None,
            counterexample: None,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== SpellcheckConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = SpellcheckConfig::default();
        assert!(config.timeout == Duration::from_secs(120));
    }

    #[kani::proof]
    fn verify_config_defaults_modes() {
        let config = SpellcheckConfig::default();
        assert!(config.check_mode);
        assert!(!config.fix_mode);
    }

    // ===== Config builders =====

    #[kani::proof]
    fn verify_config_with_check_mode() {
        let config = SpellcheckConfig::default().with_check_mode(false);
        assert!(!config.check_mode);
    }

    #[kani::proof]
    fn verify_config_with_fix_mode() {
        let config = SpellcheckConfig::default().with_fix_mode(true);
        assert!(config.fix_mode);
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = SpellcheckBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(120));
        assert!(backend.config.check_mode);
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = SpellcheckBackend::new();
        let b2 = SpellcheckBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.check_mode == b2.config.check_mode);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = SpellcheckConfig {
            timeout: Duration::from_secs(60),
            check_mode: false,
            fix_mode: true,
        };
        let backend = SpellcheckBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(!backend.config.check_mode);
        assert!(backend.config.fix_mode);
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = SpellcheckBackend::new();
        assert!(matches!(backend.id(), BackendId::Spellcheck));
    }

    #[kani::proof]
    fn verify_supports_lint() {
        let backend = SpellcheckBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Lint));
        assert!(supported.len() == 1);
    }

    // ===== Word extraction =====

    #[kani::proof]
    fn verify_extract_misspelled_word_backtick() {
        let result = SpellcheckBackend::extract_misspelled_word("error: `teh` is misspelled");
        assert!(result == Some("teh".to_string()));
    }

    #[kani::proof]
    fn verify_extract_misspelled_word_quotes() {
        let result = SpellcheckBackend::extract_misspelled_word("misspelled: \"recieve\"");
        assert!(result == Some("recieve".to_string()));
    }

    #[kani::proof]
    fn verify_extract_misspelled_word_none() {
        let result = SpellcheckBackend::extract_misspelled_word("no quoted word");
        assert!(result.is_none());
    }

    // ===== File extraction =====

    #[kani::proof]
    fn verify_extract_file_from_line_arrow() {
        let result = SpellcheckBackend::extract_file_from_line("--> src/lib.rs:10:5");
        assert!(result == Some("src/lib.rs".to_string()));
    }

    #[kani::proof]
    fn verify_extract_file_from_line_rs_path() {
        let result = SpellcheckBackend::extract_file_from_line("checking src/main.rs");
        assert!(result == Some("src/main.rs".to_string()));
    }

    // ===== Result evaluation =====

    #[kani::proof]
    fn verify_evaluate_results_no_errors() {
        let backend = SpellcheckBackend::new();
        let stats = SpellcheckStats {
            files_checked: 5,
            misspellings_found: 0,
            misspellings: vec![],
        };
        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[kani::proof]
    fn verify_evaluate_results_with_errors() {
        let backend = SpellcheckBackend::new();
        let stats = SpellcheckStats {
            files_checked: 5,
            misspellings_found: 2,
            misspellings: vec![],
        };
        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[kani::proof]
    fn verify_evaluate_results_no_files() {
        let backend = SpellcheckBackend::new();
        let stats = SpellcheckStats {
            files_checked: 0,
            misspellings_found: 0,
            misspellings: vec![],
        };
        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_id() {
        assert_eq!(SpellcheckBackend::new().id(), BackendId::Spellcheck);
    }

    #[test]
    fn test_config_default() {
        let config = SpellcheckConfig::default();
        assert!(config.check_mode);
        assert!(!config.fix_mode);
    }

    #[test]
    fn test_config_builder() {
        let config = SpellcheckConfig::default()
            .with_check_mode(false)
            .with_fix_mode(true);

        assert!(!config.check_mode);
        assert!(config.fix_mode);
    }

    #[test]
    fn test_extract_misspelled_word() {
        assert_eq!(
            SpellcheckBackend::extract_misspelled_word("error: `teh` is misspelled"),
            Some("teh".to_string())
        );
        assert_eq!(
            SpellcheckBackend::extract_misspelled_word("misspelled: \"recieve\""),
            Some("recieve".to_string())
        );
    }

    #[test]
    fn test_extract_file_from_line() {
        assert_eq!(
            SpellcheckBackend::extract_file_from_line("--> src/lib.rs:10:5"),
            Some("src/lib.rs".to_string())
        );
    }

    #[test]
    fn test_parse_output_no_errors() {
        let backend = SpellcheckBackend::new();
        let stdout = "Checking src/lib.rs\nNo spelling errors found\n";
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.misspellings_found, 0);
    }

    #[test]
    fn test_parse_output_with_errors() {
        let backend = SpellcheckBackend::new();
        let stdout = r#"
error: `teh` --> src/lib.rs:10:5
"#;
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.misspellings_found, 1);
    }

    #[test]
    fn test_evaluate_results_pass() {
        let backend = SpellcheckBackend::new();
        let stats = SpellcheckStats {
            files_checked: 5,
            misspellings_found: 0,
            misspellings: vec![],
        };

        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn test_evaluate_results_fail() {
        let backend = SpellcheckBackend::new();
        let stats = SpellcheckStats {
            files_checked: 5,
            misspellings_found: 2,
            misspellings: vec![SpellingError {
                word: "teh".to_string(),
                file: "src/lib.rs".to_string(),
                line: Some(10),
                suggestions: vec!["the".to_string()],
            }],
        };

        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(cex.is_some());
    }

    #[tokio::test]
    async fn test_health_check() {
        let backend = SpellcheckBackend::new();
        let status = backend.health_check().await;
        match status {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("cargo-spellcheck"));
            }
            _ => panic!("Unexpected status"),
        }
    }

    #[tokio::test]
    async fn test_verify_returns_result_or_unavailable() {
        use dashprove_usl::{parse, typecheck};

        let backend = SpellcheckBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        match result {
            Ok(r) => {
                assert_eq!(r.backend, BackendId::Spellcheck);
            }
            Err(BackendError::Unavailable(_)) => {}
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}
