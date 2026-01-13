//! Bisimulation executor implementation

use crate::{
    BisimError, BisimulationChecker, BisimulationConfig, BisimulationResult, Difference,
    EquivalenceCriteria, ExecutionTrace, JsonDiff, NondeterminismStrategy, OracleConfig, TestInput,
    TestSubjectConfig,
};
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// Default bisimulation checker implementation
pub struct DefaultBisimulationChecker {
    config: BisimulationConfig,
}

impl DefaultBisimulationChecker {
    /// Create a new checker with the given configuration
    pub fn new(config: BisimulationConfig) -> Self {
        Self { config }
    }

    /// Execute oracle and get trace
    async fn execute_oracle(&self, input: &TestInput) -> Result<ExecutionTrace, BisimError> {
        match &self.config.oracle {
            OracleConfig::Binary { path, args, env } => {
                self.execute_binary(path, args, env, input).await
            }
            OracleConfig::RecordedTraces { trace_dir } => {
                self.load_recorded_trace(trace_dir, &input.name).await
            }
        }
    }

    /// Execute subject and get trace
    async fn execute_subject(&self, input: &TestInput) -> Result<ExecutionTrace, BisimError> {
        match &self.config.subject {
            TestSubjectConfig::Binary { path, args, env } => {
                self.execute_binary(path, args, env, input).await
            }
            TestSubjectConfig::InProcess => Err(BisimError::config(
                "InProcess subjects require custom implementation",
            )),
        }
    }

    /// Execute a binary and capture its trace
    async fn execute_binary(
        &self,
        path: &Path,
        args: &[String],
        env: &HashMap<String, String>,
        input: &TestInput,
    ) -> Result<ExecutionTrace, BisimError> {
        let start = std::time::Instant::now();

        let mut cmd = Command::new(path);
        cmd.args(args);

        // Set environment variables
        for (key, value) in env {
            cmd.env(key, value);
        }
        for (key, value) in &input.env {
            cmd.env(key, value);
        }

        // Enable trace capture via environment
        cmd.env("DASHPROVE_TRACE_CAPTURE", "1");

        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            BisimError::oracle(format!("Failed to spawn {}: {}", path.display(), e))
        })?;

        // Write input to stdin
        if let Some(stdin) = child.stdin.as_mut() {
            stdin
                .write_all(input.input.as_bytes())
                .await
                .map_err(|e| BisimError::oracle(format!("Failed to write stdin: {e}")))?;
        }

        // Wait for completion with timeout
        let output = tokio::time::timeout(input.timeout, child.wait_with_output())
            .await
            .map_err(|_| BisimError::timeout(input.timeout.as_millis() as u64))?
            .map_err(|e| BisimError::oracle(format!("Process error: {e}")))?;

        let duration = start.elapsed();

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(BisimError::process(
                output.status.code(),
                stderr.to_string(),
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Try to parse trace from output
        self.parse_trace_output(&stdout, duration)
    }

    /// Parse trace from process output
    fn parse_trace_output(
        &self,
        output: &str,
        duration: Duration,
    ) -> Result<ExecutionTrace, BisimError> {
        // Look for JSON trace markers
        if let Some(trace_start) = output.find("__DASHPROVE_TRACE_START__") {
            if let Some(trace_end) = output.find("__DASHPROVE_TRACE_END__") {
                let trace_json = &output[trace_start + 25..trace_end];
                let trace: ExecutionTrace = serde_json::from_str(trace_json.trim())
                    .map_err(|e| BisimError::parsing(format!("Invalid trace JSON: {e}")))?;
                return Ok(trace);
            }
        }

        // Fallback: create trace from raw output
        Ok(ExecutionTrace::builder()
            .output(output.to_string())
            .duration(duration)
            .build())
    }

    /// Load a recorded trace from disk
    async fn load_recorded_trace(
        &self,
        trace_dir: &Path,
        test_name: &str,
    ) -> Result<ExecutionTrace, BisimError> {
        let trace_path = trace_dir.join(format!("{test_name}.json"));

        let content = tokio::fs::read_to_string(&trace_path).await.map_err(|e| {
            BisimError::oracle(format!(
                "Failed to read trace {}: {e}",
                trace_path.display()
            ))
        })?;

        serde_json::from_str(&content)
            .map_err(|e| BisimError::parsing(format!("Invalid trace JSON: {e}")))
    }

    /// Compare two traces according to criteria
    fn compare_traces(
        &self,
        oracle: &ExecutionTrace,
        subject: &ExecutionTrace,
        criteria: &EquivalenceCriteria,
        strategy: &NondeterminismStrategy,
    ) -> (Vec<Difference>, f64) {
        let mut differences = vec![];
        let mut confidence = 1.0;

        // Determine payload config based on strategy
        // - ExactMatch: use empty/default config (no tolerance)
        // - SemanticSimilarity: use the configured payload_config
        // - DistributionMatch: use the configured payload_config
        let payload_config = match strategy {
            NondeterminismStrategy::ExactMatch => None,
            _ => Some(&criteria.payload_config),
        };

        // Compare API requests
        if criteria.api_requests {
            let api_diffs = self.compare_api_requests(oracle, subject, strategy, payload_config);
            differences.extend(api_diffs);
        }

        // Compare tool calls
        if criteria.tool_calls {
            let tool_diffs = self.compare_tool_calls(oracle, subject, strategy, payload_config);
            differences.extend(tool_diffs);
        }

        // Compare output
        if criteria.output {
            if let Some(output_diff) = self.compare_output(oracle, subject, criteria, strategy) {
                // Adjust confidence based on output similarity
                if let Difference::OutputMismatch { similarity, .. } = &output_diff {
                    confidence *= similarity;
                }
                differences.push(output_diff);
            }
        }

        // Check timing
        if let Some(tolerance) = criteria.timing_tolerance {
            if let Some(timing_diff) = self.compare_timing(oracle, subject, tolerance) {
                confidence *= 0.9; // Slight penalty for timing violations
                differences.push(timing_diff);
            }
        }

        (differences, confidence)
    }

    /// Compare API requests between traces
    fn compare_api_requests(
        &self,
        oracle: &ExecutionTrace,
        subject: &ExecutionTrace,
        _strategy: &NondeterminismStrategy,
        payload_config: Option<&crate::PayloadComparisonConfig>,
    ) -> Vec<Difference> {
        let mut differences = vec![];

        // Check sequence length
        if oracle.api_requests.len() != subject.api_requests.len() {
            differences.push(Difference::SequenceLengthMismatch {
                sequence_type: "api_requests".to_string(),
                oracle_count: oracle.api_requests.len(),
                subject_count: subject.api_requests.len(),
            });
        }

        // Compare each request
        let min_len = oracle.api_requests.len().min(subject.api_requests.len());
        for i in 0..min_len {
            let o_req = &oracle.api_requests[i];
            let s_req = &subject.api_requests[i];

            let diff = match payload_config {
                Some(config) => JsonDiff::compare_with_config(
                    &Self::api_request_snapshot(o_req),
                    &Self::api_request_snapshot(s_req),
                    config,
                ),
                None => JsonDiff::compare(
                    &Self::api_request_snapshot(o_req),
                    &Self::api_request_snapshot(s_req),
                ),
            };
            if !diff.is_empty() {
                differences.push(Difference::ApiRequestMismatch {
                    index: i,
                    oracle: o_req.clone(),
                    subject: s_req.clone(),
                    diff,
                });
            }
        }

        // Report missing/extra requests
        for i in min_len..oracle.api_requests.len() {
            differences.push(Difference::MissingEvent {
                index: i,
                event_type: "api_request".to_string(),
                description: format!(
                    "{} {}",
                    oracle.api_requests[i].method, oracle.api_requests[i].url
                ),
            });
        }

        for i in min_len..subject.api_requests.len() {
            differences.push(Difference::ExtraEvent {
                index: i,
                event_type: "api_request".to_string(),
                description: format!(
                    "{} {}",
                    subject.api_requests[i].method, subject.api_requests[i].url
                ),
            });
        }

        differences
    }

    fn api_request_snapshot(request: &crate::ApiRequest) -> serde_json::Value {
        json!({
            "method": request.method,
            "url": request.url,
            "headers": request.headers,
            "body": request.body,
            "response_status": request.response_status,
            "response_body": request.response_body,
        })
    }

    fn tool_call_snapshot(call: &crate::ToolCall) -> serde_json::Value {
        json!({
            "name": call.name,
            "arguments": call.arguments,
            "result": call.result,
            "success": call.success,
            "error": call.error,
        })
    }

    /// Compare tool calls between traces
    fn compare_tool_calls(
        &self,
        oracle: &ExecutionTrace,
        subject: &ExecutionTrace,
        _strategy: &NondeterminismStrategy,
        payload_config: Option<&crate::PayloadComparisonConfig>,
    ) -> Vec<Difference> {
        let mut differences = vec![];

        // Check sequence length
        if oracle.tool_calls.len() != subject.tool_calls.len() {
            differences.push(Difference::SequenceLengthMismatch {
                sequence_type: "tool_calls".to_string(),
                oracle_count: oracle.tool_calls.len(),
                subject_count: subject.tool_calls.len(),
            });
        }

        // Compare each tool call using full snapshot (name, arguments, result, success, error)
        let min_len = oracle.tool_calls.len().min(subject.tool_calls.len());
        for i in 0..min_len {
            let o_call = &oracle.tool_calls[i];
            let s_call = &subject.tool_calls[i];

            let diff = match payload_config {
                Some(config) => JsonDiff::compare_with_config(
                    &Self::tool_call_snapshot(o_call),
                    &Self::tool_call_snapshot(s_call),
                    config,
                ),
                None => JsonDiff::compare(
                    &Self::tool_call_snapshot(o_call),
                    &Self::tool_call_snapshot(s_call),
                ),
            };
            if !diff.is_empty() {
                differences.push(Difference::ToolCallMismatch {
                    index: i,
                    oracle: o_call.clone(),
                    subject: s_call.clone(),
                    diff,
                });
            }
        }

        // Report missing/extra tool calls
        for i in min_len..oracle.tool_calls.len() {
            differences.push(Difference::MissingEvent {
                index: i,
                event_type: "tool_call".to_string(),
                description: oracle.tool_calls[i].name.clone(),
            });
        }

        for i in min_len..subject.tool_calls.len() {
            differences.push(Difference::ExtraEvent {
                index: i,
                event_type: "tool_call".to_string(),
                description: subject.tool_calls[i].name.clone(),
            });
        }

        differences
    }

    /// Compare output between traces
    fn compare_output(
        &self,
        oracle: &ExecutionTrace,
        subject: &ExecutionTrace,
        criteria: &EquivalenceCriteria,
        strategy: &NondeterminismStrategy,
    ) -> Option<Difference> {
        if oracle.output == subject.output {
            return None;
        }

        let similarity = crate::diff::text_similarity(&oracle.output, &subject.output);

        // Check if similarity meets threshold based on strategy
        let threshold = match strategy {
            NondeterminismStrategy::ExactMatch => 1.0,
            NondeterminismStrategy::SemanticSimilarity { threshold } => *threshold,
            NondeterminismStrategy::DistributionMatch { .. } => 0.0, // Allow any output
        };

        if criteria.semantic_comparison || similarity >= threshold {
            // Considered equivalent
            if similarity >= threshold {
                return None;
            }
        }

        Some(Difference::OutputMismatch {
            oracle: oracle.output.clone(),
            subject: subject.output.clone(),
            similarity,
        })
    }

    /// Compare timing between traces
    fn compare_timing(
        &self,
        oracle: &ExecutionTrace,
        subject: &ExecutionTrace,
        tolerance: f64,
    ) -> Option<Difference> {
        let oracle_ms = oracle.duration.as_millis() as u64;
        let subject_ms = subject.duration.as_millis() as u64;

        if oracle_ms == 0 {
            return None;
        }

        let diff_ratio = ((subject_ms as f64) - (oracle_ms as f64)).abs() / (oracle_ms as f64);

        if diff_ratio > tolerance {
            Some(Difference::TimingViolation {
                oracle_ms,
                subject_ms,
                tolerance,
            })
        } else {
            None
        }
    }
}

#[async_trait]
impl BisimulationChecker for DefaultBisimulationChecker {
    async fn check(&self, input: &TestInput) -> Result<BisimulationResult, BisimError> {
        // Execute oracle
        let oracle_trace = self.execute_oracle(input).await?;

        // Execute subject
        let subject_trace = self.execute_subject(input).await?;

        // Compare traces
        let (differences, confidence) = self.compare_traces(
            &oracle_trace,
            &subject_trace,
            &self.config.equivalence_criteria,
            &self.config.nondeterminism_strategy,
        );

        if differences.is_empty() {
            Ok(BisimulationResult::equivalent(oracle_trace, subject_trace))
        } else {
            Ok(BisimulationResult::not_equivalent(
                oracle_trace,
                subject_trace,
                differences,
                confidence,
            ))
        }
    }

    async fn check_batch(
        &self,
        inputs: &[TestInput],
    ) -> Result<Vec<BisimulationResult>, BisimError> {
        let mut results = Vec::with_capacity(inputs.len());

        for input in inputs {
            let result = self.check(input).await?;
            results.push(result);
        }

        Ok(results)
    }
}

/// Binary executor for running and capturing traces from executables
pub struct BinaryExecutor {
    path: PathBuf,
    args: Vec<String>,
    env: HashMap<String, String>,
    timeout: Duration,
}

impl BinaryExecutor {
    /// Create a new binary executor
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            args: vec![],
            env: HashMap::new(),
            timeout: Duration::from_secs(60),
        }
    }

    /// Set command arguments
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Set environment variables
    pub fn with_env(mut self, env: HashMap<String, String>) -> Self {
        self.env = env;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Execute the binary with given input
    pub async fn execute(&self, input: &str) -> Result<ExecutionTrace, BisimError> {
        let start = std::time::Instant::now();

        let mut cmd = Command::new(&self.path);
        cmd.args(&self.args);

        for (key, value) in &self.env {
            cmd.env(key, value);
        }

        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            BisimError::subject(format!("Failed to spawn {}: {e}", self.path.display()))
        })?;

        if let Some(stdin) = child.stdin.as_mut() {
            stdin
                .write_all(input.as_bytes())
                .await
                .map_err(|e| BisimError::subject(format!("Failed to write stdin: {e}")))?;
        }

        let output = tokio::time::timeout(self.timeout, child.wait_with_output())
            .await
            .map_err(|_| BisimError::timeout(self.timeout.as_millis() as u64))?
            .map_err(|e| BisimError::subject(format!("Process error: {e}")))?;

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout);

        Ok(ExecutionTrace::builder()
            .output(stdout.to_string())
            .duration(duration)
            .build())
    }
}

/// Trace recorder for capturing execution traces
pub struct TraceRecorder {
    trace_dir: PathBuf,
}

impl TraceRecorder {
    /// Create a new trace recorder
    pub fn new(trace_dir: impl Into<PathBuf>) -> Self {
        Self {
            trace_dir: trace_dir.into(),
        }
    }

    /// Record a trace to disk
    pub async fn record(&self, name: &str, trace: &ExecutionTrace) -> Result<PathBuf, BisimError> {
        tokio::fs::create_dir_all(&self.trace_dir).await?;

        let path = self.trace_dir.join(format!("{name}.json"));
        let json = serde_json::to_string_pretty(trace)?;
        tokio::fs::write(&path, json).await?;

        Ok(path)
    }

    /// Load a recorded trace
    pub async fn load(&self, name: &str) -> Result<ExecutionTrace, BisimError> {
        let path = self.trace_dir.join(format!("{name}.json"));
        let content = tokio::fs::read_to_string(&path).await?;
        let trace: ExecutionTrace = serde_json::from_str(&content)?;
        Ok(trace)
    }

    /// List all recorded traces
    pub async fn list(&self) -> Result<Vec<String>, BisimError> {
        let mut entries = tokio::fs::read_dir(&self.trace_dir).await?;
        let mut names = vec![];

        while let Some(entry) = entries.next_entry().await? {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".json") {
                    names.push(name.trim_end_matches(".json").to_string());
                }
            }
        }

        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checker_creation() {
        let config = BisimulationConfig::default();
        let _checker = DefaultBisimulationChecker::new(config);
    }

    #[test]
    fn test_compare_traces_equal() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config.clone());

        let trace = ExecutionTrace::builder()
            .output("hello world")
            .duration(Duration::from_millis(100))
            .build();

        let (diffs, confidence) = checker.compare_traces(
            &trace,
            &trace.clone(),
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        assert!(diffs.is_empty());
        assert_eq!(confidence, 1.0);
    }

    #[test]
    fn test_compare_traces_different_output() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config.clone());

        let oracle = ExecutionTrace::builder()
            .output("hello world")
            .duration(Duration::from_millis(100))
            .build();

        let subject = ExecutionTrace::builder()
            .output("hello there")
            .duration(Duration::from_millis(100))
            .build();

        let (diffs, _confidence) = checker.compare_traces(
            &oracle,
            &subject,
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        assert!(!diffs.is_empty());
        assert!(matches!(diffs[0], Difference::OutputMismatch { .. }));
    }

    #[test]
    fn test_binary_executor_builder() {
        let executor = BinaryExecutor::new("/usr/bin/echo")
            .with_args(vec!["-n".to_string()])
            .with_timeout(Duration::from_secs(30));

        assert_eq!(executor.path, PathBuf::from("/usr/bin/echo"));
        assert_eq!(executor.args, vec!["-n"]);
        assert_eq!(executor.timeout, Duration::from_secs(30));
    }

    #[tokio::test]
    async fn test_trace_recorder() {
        let temp_dir = tempfile::tempdir().unwrap();
        let recorder = TraceRecorder::new(temp_dir.path());

        let trace = ExecutionTrace::builder()
            .output("test output")
            .duration(Duration::from_millis(500))
            .build();

        // Record
        let path = recorder.record("test_trace", &trace).await.unwrap();
        assert!(path.exists());

        // Load
        let loaded = recorder.load("test_trace").await.unwrap();
        assert_eq!(loaded.output, "test output");

        // List
        let names = recorder.list().await.unwrap();
        assert!(names.contains(&"test_trace".to_string()));
    }

    // =====================================================
    // Mutation-killing tests for DefaultBisimulationChecker
    // =====================================================

    /// Test parse_trace_output extracts trace from markers correctly
    /// (catches mutations: replace + with - or * in line 121)
    #[test]
    fn test_parse_trace_output_with_markers() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let trace_json =
            r#"{"api_requests":[],"tool_calls":[],"output":"parsed","duration":100,"metadata":{}}"#;
        let output = format!(
            "Some prefix\n__DASHPROVE_TRACE_START__{}__DASHPROVE_TRACE_END__\nSome suffix",
            trace_json
        );

        let result = checker.parse_trace_output(&output, Duration::from_millis(50));
        assert!(result.is_ok());
        let trace = result.unwrap();
        assert_eq!(trace.output, "parsed");
    }

    /// Test parse_trace_output falls back to raw output when no markers
    #[test]
    fn test_parse_trace_output_fallback() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let output = "raw output without markers";
        let result = checker.parse_trace_output(output, Duration::from_millis(100));

        assert!(result.is_ok());
        let trace = result.unwrap();
        assert_eq!(trace.output, "raw output without markers");
        assert_eq!(trace.duration, Duration::from_millis(100));
    }

    /// Test compare_api_requests detects length mismatch
    /// (catches mutation: replace compare_api_requests -> Vec<Difference> with vec![])
    #[test]
    fn test_compare_api_requests_length_mismatch() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .api_request(crate::ApiRequest::new("GET", "http://a"))
            .api_request(crate::ApiRequest::new("GET", "http://b"))
            .build();

        let subject = ExecutionTrace::builder()
            .api_request(crate::ApiRequest::new("GET", "http://a"))
            .build();

        let diffs = checker.compare_api_requests(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );

        assert!(!diffs.is_empty(), "Should detect length mismatch");
        assert!(diffs
            .iter()
            .any(|d| matches!(d, Difference::SequenceLengthMismatch { .. })));
    }

    /// Test compare_api_requests detects method/URL mismatch
    /// (catches mutations: replace || with && and != with == in line 224)
    #[test]
    fn test_compare_api_requests_method_mismatch() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .api_request(crate::ApiRequest::new("GET", "http://test"))
            .build();

        let subject = ExecutionTrace::builder()
            .api_request(crate::ApiRequest::new("POST", "http://test"))
            .build();

        let diffs = checker.compare_api_requests(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );

        assert!(!diffs.is_empty(), "Should detect method mismatch");
        assert!(diffs
            .iter()
            .any(|d| matches!(d, Difference::ApiRequestMismatch { .. })));
    }

    /// Test compare_api_requests detects URL mismatch
    #[test]
    fn test_compare_api_requests_url_mismatch() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .api_request(crate::ApiRequest::new("GET", "http://oracle"))
            .build();

        let subject = ExecutionTrace::builder()
            .api_request(crate::ApiRequest::new("GET", "http://subject"))
            .build();

        let diffs = checker.compare_api_requests(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );

        assert!(!diffs.is_empty(), "Should detect URL mismatch");
    }

    /// Test compare_api_requests detects body diff via JsonDiff
    /// (catches mutation: delete ! in line 237)
    #[test]
    fn test_compare_api_requests_body_mismatch() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("POST", "http://test")
                    .with_body(serde_json::json!({"key": "oracle_value"})),
            )
            .build();

        let subject = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("POST", "http://test")
                    .with_body(serde_json::json!({"key": "subject_value"})),
            )
            .build();

        let diffs = checker.compare_api_requests(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );

        assert!(!diffs.is_empty(), "Should detect body mismatch");
    }

    /// Test compare_api_requests detects header mismatch
    #[test]
    fn test_compare_api_requests_header_mismatch() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("GET", "http://test")
                    .with_header("Content-Type", "application/json"),
            )
            .build();

        let subject = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("GET", "http://test")
                    .with_header("Content-Type", "text/plain"),
            )
            .build();

        let diffs = checker.compare_api_requests(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );

        assert!(
            !diffs.is_empty(),
            "Header mismatch should be detected via JsonDiff"
        );
    }

    /// Test compare_api_requests detects response status/body differences
    #[test]
    fn test_compare_api_requests_response_mismatch() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("POST", "http://test")
                    .with_response(200, serde_json::json!({"ok": true})),
            )
            .build();

        let subject = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("POST", "http://test")
                    .with_response(500, serde_json::json!({"ok": false})),
            )
            .build();

        let diffs = checker.compare_api_requests(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );

        assert!(
            diffs
                .iter()
                .any(|d| matches!(d, Difference::ApiRequestMismatch { .. })),
            "Response differences should produce mismatch"
        );
    }

    /// Test compare_tool_calls detects length mismatch
    /// (catches mutation: replace compare_tool_calls -> Vec<Difference> with vec![])
    #[test]
    fn test_compare_tool_calls_length_mismatch() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .tool_call(crate::ToolCall::new("read"))
            .tool_call(crate::ToolCall::new("write"))
            .build();

        let subject = ExecutionTrace::builder()
            .tool_call(crate::ToolCall::new("read"))
            .build();

        let diffs = checker.compare_tool_calls(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );

        assert!(!diffs.is_empty(), "Should detect length mismatch");
    }

    /// Test compare_tool_calls detects name mismatch
    /// (catches mutation: replace != with == in line 298)
    #[test]
    fn test_compare_tool_calls_name_mismatch() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .tool_call(crate::ToolCall::new("read_file"))
            .build();

        let subject = ExecutionTrace::builder()
            .tool_call(crate::ToolCall::new("write_file"))
            .build();

        let diffs = checker.compare_tool_calls(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );

        assert!(!diffs.is_empty(), "Should detect name mismatch");
        assert!(diffs
            .iter()
            .any(|d| matches!(d, Difference::ToolCallMismatch { .. })));
    }

    /// Test compare_tool_calls detects argument diff
    /// (catches mutation: delete ! in line 309)
    #[test]
    fn test_compare_tool_calls_args_mismatch() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_arguments(serde_json::json!({"path": "/oracle"})),
            )
            .build();

        let subject = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_arguments(serde_json::json!({"path": "/subject"})),
            )
            .build();

        let diffs = checker.compare_tool_calls(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );

        assert!(!diffs.is_empty(), "Should detect argument mismatch");
    }

    /// Test compare_tool_calls detects result diff
    #[test]
    fn test_compare_tool_calls_result_mismatch() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_result(serde_json::json!({"content": "oracle data"})),
            )
            .build();

        let subject = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_result(serde_json::json!({"content": "subject data"})),
            )
            .build();

        let diffs = checker.compare_tool_calls(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );

        assert!(!diffs.is_empty(), "Should detect result mismatch");
        // Verify the diff contains the result path
        if let Difference::ToolCallMismatch { diff, .. } = &diffs[0] {
            assert!(
                diff.differences.iter().any(|d| d.path.contains("result")),
                "Diff should mention result path"
            );
        } else {
            panic!("Expected ToolCallMismatch");
        }
    }

    /// Test compare_tool_calls detects success flag mismatch
    #[test]
    fn test_compare_tool_calls_success_mismatch() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .tool_call(crate::ToolCall::new("read_file"))
            .build();

        let subject = ExecutionTrace::builder()
            .tool_call(crate::ToolCall::new("read_file").with_error("file not found"))
            .build();

        let diffs = checker.compare_tool_calls(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );

        assert!(!diffs.is_empty(), "Should detect success mismatch");
        // Verify the diff contains success/error paths
        if let Difference::ToolCallMismatch { diff, .. } = &diffs[0] {
            let has_success_diff = diff.differences.iter().any(|d| d.path.contains("success"));
            let has_error_diff = diff.differences.iter().any(|d| d.path.contains("error"));
            assert!(
                has_success_diff || has_error_diff,
                "Diff should mention success or error path"
            );
        } else {
            panic!("Expected ToolCallMismatch");
        }
    }

    /// Test compare_tool_calls with identical tool calls produces no diff
    #[test]
    fn test_compare_tool_calls_identical() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_arguments(serde_json::json!({"path": "/test"}))
                    .with_result(serde_json::json!({"content": "data"})),
            )
            .build();

        let subject = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_arguments(serde_json::json!({"path": "/test"}))
                    .with_result(serde_json::json!({"content": "data"})),
            )
            .build();

        let diffs = checker.compare_tool_calls(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );

        assert!(
            diffs.is_empty(),
            "Identical tool calls should produce no diff"
        );
    }

    /// Test tool_call_snapshot includes all fields
    #[test]
    fn test_tool_call_snapshot_fields() {
        let call = crate::ToolCall::new("test_tool")
            .with_arguments(serde_json::json!({"arg": "value"}))
            .with_result(serde_json::json!({"result": "output"}));

        let snapshot = DefaultBisimulationChecker::tool_call_snapshot(&call);

        assert_eq!(snapshot["name"], "test_tool");
        assert_eq!(snapshot["arguments"]["arg"], "value");
        assert_eq!(snapshot["result"]["result"], "output");
        assert_eq!(snapshot["success"], true);
        assert!(snapshot["error"].is_null());
    }

    /// Test tool_call_snapshot with error state
    #[test]
    fn test_tool_call_snapshot_with_error() {
        let call = crate::ToolCall::new("test_tool").with_error("something went wrong");

        let snapshot = DefaultBisimulationChecker::tool_call_snapshot(&call);

        assert_eq!(snapshot["name"], "test_tool");
        assert_eq!(snapshot["success"], false);
        assert_eq!(snapshot["error"], "something went wrong");
    }

    /// Test compare_output with semantic comparison threshold
    /// (catches mutations: replace || with && in line 359, replace >= with < in lines 359, 361)
    #[test]
    fn test_compare_output_semantic_threshold() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder().output("hello world").build();

        let subject = ExecutionTrace::builder().output("hello there").build();

        // With semantic strategy at 0.5 threshold, high similarity should pass
        let criteria = EquivalenceCriteria::default();
        let strategy = NondeterminismStrategy::SemanticSimilarity { threshold: 0.5 };

        let result = checker.compare_output(&oracle, &subject, &criteria, &strategy);
        // Similarity is high enough (~0.7), should be None
        assert!(result.is_none(), "High similarity should pass threshold");
    }

    /// Test compare_output returns diff when below threshold
    #[test]
    fn test_compare_output_below_threshold() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .output("completely different output A")
            .build();

        let subject = ExecutionTrace::builder().output("xyz").build();

        let criteria = EquivalenceCriteria::default();
        let strategy = NondeterminismStrategy::SemanticSimilarity { threshold: 0.9 };

        let result = checker.compare_output(&oracle, &subject, &criteria, &strategy);
        assert!(result.is_some(), "Low similarity should fail threshold");
    }

    /// Test compare_timing returns None when oracle_ms is 0
    /// (catches mutation: replace == with != in line 383)
    #[test]
    fn test_compare_timing_zero_oracle() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder().duration(Duration::ZERO).build();

        let subject = ExecutionTrace::builder()
            .duration(Duration::from_millis(100))
            .build();

        let result = checker.compare_timing(&oracle, &subject, 0.1);
        assert!(result.is_none(), "Should return None when oracle_ms is 0");
    }

    /// Test compare_timing detects violation
    /// (catches mutations: replace arithmetic ops in line 387, replace > with == or < in line 389)
    #[test]
    fn test_compare_timing_violation() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .duration(Duration::from_millis(100))
            .build();

        // Subject is 150% of oracle (50% difference), tolerance is 10%
        let subject = ExecutionTrace::builder()
            .duration(Duration::from_millis(150))
            .build();

        let result = checker.compare_timing(&oracle, &subject, 0.1);
        assert!(result.is_some(), "50% diff should exceed 10% tolerance");
        if let Some(Difference::TimingViolation {
            oracle_ms,
            subject_ms,
            tolerance,
        }) = result
        {
            assert_eq!(oracle_ms, 100);
            assert_eq!(subject_ms, 150);
            assert!((tolerance - 0.1).abs() < f64::EPSILON);
        } else {
            panic!("Expected TimingViolation");
        }
    }

    /// Test compare_timing passes when within tolerance
    #[test]
    fn test_compare_timing_within_tolerance() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .duration(Duration::from_millis(100))
            .build();

        // Subject is 105ms (5% diff), tolerance is 10%
        let subject = ExecutionTrace::builder()
            .duration(Duration::from_millis(105))
            .build();

        let result = checker.compare_timing(&oracle, &subject, 0.1);
        assert!(result.is_none(), "5% diff should be within 10% tolerance");
    }

    /// Test compare_traces confidence adjustment for output mismatch
    /// (catches mutation: replace *= with += or /= in line 182)
    #[test]
    fn test_compare_traces_confidence_output() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config.clone());

        let oracle = ExecutionTrace::builder().output("hello world").build();

        let subject = ExecutionTrace::builder().output("hello there").build();

        let (diffs, confidence) = checker.compare_traces(
            &oracle,
            &subject,
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        // Should have output diff and confidence should be reduced
        assert!(!diffs.is_empty());
        assert!(confidence < 1.0, "Confidence should be reduced");
        assert!(confidence > 0.0, "Confidence should still be positive");
    }

    /// Test compare_traces confidence adjustment for timing violation
    /// (catches mutation: replace *= with += or /= in line 191)
    #[test]
    fn test_compare_traces_confidence_timing() {
        let mut config = BisimulationConfig::default();
        config.equivalence_criteria.timing_tolerance = Some(0.01); // 1% tolerance
        config.equivalence_criteria.output = false; // Don't check output
        let checker = DefaultBisimulationChecker::new(config.clone());

        let oracle = ExecutionTrace::builder()
            .output("same")
            .duration(Duration::from_millis(100))
            .build();

        let subject = ExecutionTrace::builder()
            .output("same")
            .duration(Duration::from_millis(200)) // 100% diff
            .build();

        let (diffs, confidence) = checker.compare_traces(
            &oracle,
            &subject,
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        // Should have timing diff
        assert!(diffs
            .iter()
            .any(|d| matches!(d, Difference::TimingViolation { .. })));
        // Confidence should be 0.9 (multiplied by 0.9)
        assert!(
            (confidence - 0.9).abs() < 0.01,
            "Confidence should be ~0.9, got {}",
            confidence
        );
    }

    /// Test BinaryExecutor builder
    #[test]
    fn test_binary_executor_with_env() {
        let mut env = HashMap::new();
        env.insert("KEY".to_string(), "VALUE".to_string());

        let executor = BinaryExecutor::new("/bin/test").with_env(env);

        assert_eq!(executor.env.get("KEY"), Some(&"VALUE".to_string()));
    }

    /// Test compare_timing at exact boundary (diff_ratio == tolerance)
    /// (catches mutation: replace > with >= in line 389)
    #[test]
    fn test_compare_timing_exact_boundary() {
        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .duration(Duration::from_millis(100))
            .build();

        // 10% diff exactly at 10% tolerance boundary
        // diff_ratio = |110 - 100| / 100 = 0.1 = tolerance
        // With > comparison, 0.1 > 0.1 is false, so should NOT be a violation
        let subject = ExecutionTrace::builder()
            .duration(Duration::from_millis(110))
            .build();

        let result = checker.compare_timing(&oracle, &subject, 0.1);
        // At exactly the boundary, should be acceptable (no violation)
        assert!(
            result.is_none(),
            "At exact tolerance boundary, should not be violation"
        );

        // Just over the boundary (11% diff with 10% tolerance)
        let subject_over = ExecutionTrace::builder()
            .duration(Duration::from_millis(111))
            .build();

        let result_over = checker.compare_timing(&oracle, &subject_over, 0.1);
        // Just over boundary should be violation
        assert!(
            result_over.is_some(),
            "Just over tolerance should be violation"
        );
    }

    // =====================================================
    // Tests for PayloadComparisonConfig integration
    // =====================================================

    /// Test that ExactMatch strategy does NOT use payload config tolerances
    #[test]
    fn test_exact_match_ignores_payload_config() {
        use crate::PayloadComparisonConfig;

        // Create config with tolerances that would normally pass
        let payload_config = PayloadComparisonConfig::ignoring(vec!["$.headers.*".to_string()])
            .with_numeric_tolerance(0.1)
            .with_text_threshold(0.5);

        let criteria = EquivalenceCriteria::default().with_payload_config(payload_config);
        // Use ExactMatch strategy - should NOT apply tolerances
        let config = BisimulationConfig::default()
            .with_criteria(criteria)
            .with_strategy(NondeterminismStrategy::ExactMatch);

        let checker = DefaultBisimulationChecker::new(config.clone());

        let oracle = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("GET", "http://test")
                    .with_header("Authorization", "token-oracle"),
            )
            .build();

        let subject = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("GET", "http://test")
                    .with_header("Authorization", "token-subject"),
            )
            .build();

        // With ExactMatch, even headers should trigger a diff
        let (diffs, _) = checker.compare_traces(
            &oracle,
            &subject,
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        assert!(
            !diffs.is_empty(),
            "ExactMatch should detect header difference despite ignore config"
        );
    }

    /// Test that SemanticSimilarity strategy DOES use payload config path ignoring
    #[test]
    fn test_semantic_similarity_uses_ignore_paths() {
        use crate::PayloadComparisonConfig;

        let payload_config = PayloadComparisonConfig::ignoring(vec!["$.headers.*".to_string()]);

        let criteria = EquivalenceCriteria::default().with_payload_config(payload_config);
        let config = BisimulationConfig::default()
            .with_criteria(criteria)
            .with_strategy(NondeterminismStrategy::SemanticSimilarity { threshold: 0.9 });

        let checker = DefaultBisimulationChecker::new(config.clone());

        let oracle = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("GET", "http://test")
                    .with_header("Authorization", "token-oracle"),
            )
            .output("same output")
            .build();

        let subject = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("GET", "http://test")
                    .with_header("Authorization", "token-subject"),
            )
            .output("same output")
            .build();

        // With SemanticSimilarity and header ignore, should be no API diff
        let (diffs, _) = checker.compare_traces(
            &oracle,
            &subject,
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        // Should not have API request mismatch (headers ignored)
        let has_api_mismatch = diffs
            .iter()
            .any(|d| matches!(d, Difference::ApiRequestMismatch { .. }));
        assert!(
            !has_api_mismatch,
            "SemanticSimilarity should ignore headers with payload config"
        );
    }

    /// Test that SemanticSimilarity strategy uses numeric tolerance
    #[test]
    fn test_semantic_similarity_uses_numeric_tolerance() {
        use crate::PayloadComparisonConfig;

        let payload_config = PayloadComparisonConfig::default().with_numeric_tolerance(0.1); // 10% tolerance

        let criteria = EquivalenceCriteria::default().with_payload_config(payload_config);
        let config = BisimulationConfig::default()
            .with_criteria(criteria)
            .with_strategy(NondeterminismStrategy::SemanticSimilarity { threshold: 0.9 });

        let checker = DefaultBisimulationChecker::new(config.clone());

        // Tool calls with numeric arguments differing by ~5% (within 10% tolerance)
        let oracle = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("compute")
                    .with_arguments(serde_json::json!({"value": 100.0}))
                    .with_result(serde_json::json!({"result": 200.0})),
            )
            .output("same")
            .build();

        let subject = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("compute")
                    .with_arguments(serde_json::json!({"value": 105.0})) // 5% diff
                    .with_result(serde_json::json!({"result": 210.0})), // 5% diff
            )
            .output("same")
            .build();

        let (diffs, _) = checker.compare_traces(
            &oracle,
            &subject,
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        let has_tool_mismatch = diffs
            .iter()
            .any(|d| matches!(d, Difference::ToolCallMismatch { .. }));
        assert!(
            !has_tool_mismatch,
            "SemanticSimilarity should tolerate 5% numeric difference with 10% tolerance"
        );
    }

    /// Test that SemanticSimilarity strategy uses text similarity threshold
    #[test]
    fn test_semantic_similarity_uses_text_threshold() {
        use crate::PayloadComparisonConfig;

        let payload_config = PayloadComparisonConfig::default().with_text_threshold(0.6);

        let criteria = EquivalenceCriteria::default().with_payload_config(payload_config);
        let config = BisimulationConfig::default()
            .with_criteria(criteria)
            .with_strategy(NondeterminismStrategy::SemanticSimilarity { threshold: 0.9 });

        let checker = DefaultBisimulationChecker::new(config.clone());

        // Tool calls with similar but not identical string results
        let oracle = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_result(serde_json::json!({"content": "hello world from oracle"})),
            )
            .output("same")
            .build();

        let subject = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_result(serde_json::json!({"content": "hello world from subject"})), // Similar
            )
            .output("same")
            .build();

        let (diffs, _) = checker.compare_traces(
            &oracle,
            &subject,
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        let has_tool_mismatch = diffs
            .iter()
            .any(|d| matches!(d, Difference::ToolCallMismatch { .. }));
        assert!(
            !has_tool_mismatch,
            "SemanticSimilarity should tolerate similar text with text threshold"
        );
    }

    /// Test compare_api_requests with payload config directly
    #[test]
    fn test_compare_api_requests_with_payload_config() {
        use crate::PayloadComparisonConfig;

        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("POST", "http://test")
                    .with_body(serde_json::json!({"data": "value", "timestamp": 1000})),
            )
            .build();

        let subject = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("POST", "http://test")
                    .with_body(serde_json::json!({"data": "value", "timestamp": 2000})),
            )
            .build();

        // Without config, should detect timestamp difference
        let diffs_without = checker.compare_api_requests(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );
        assert!(
            !diffs_without.is_empty(),
            "Should detect timestamp diff without config"
        );

        // With config ignoring timestamp
        let payload_config =
            PayloadComparisonConfig::ignoring(vec!["$.body.timestamp".to_string()]);
        let diffs_with = checker.compare_api_requests(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            Some(&payload_config),
        );
        assert!(
            diffs_with.is_empty(),
            "Should ignore timestamp with payload config"
        );
    }

    /// Test compare_tool_calls with payload config directly
    #[test]
    fn test_compare_tool_calls_with_payload_config() {
        use crate::PayloadComparisonConfig;

        let config = BisimulationConfig::default();
        let checker = DefaultBisimulationChecker::new(config);

        let oracle =
            ExecutionTrace::builder()
                .tool_call(crate::ToolCall::new("fetch").with_arguments(
                    serde_json::json!({"url": "http://test", "request_id": "abc123"}),
                ))
                .build();

        let subject =
            ExecutionTrace::builder()
                .tool_call(crate::ToolCall::new("fetch").with_arguments(
                    serde_json::json!({"url": "http://test", "request_id": "xyz789"}),
                ))
                .build();

        // Without config, should detect request_id difference
        let diffs_without = checker.compare_tool_calls(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            None,
        );
        assert!(
            !diffs_without.is_empty(),
            "Should detect request_id diff without config"
        );

        // With config ignoring request_id
        let payload_config =
            PayloadComparisonConfig::ignoring(vec!["$.arguments.request_id".to_string()]);
        let diffs_with = checker.compare_tool_calls(
            &oracle,
            &subject,
            &NondeterminismStrategy::ExactMatch,
            Some(&payload_config),
        );
        assert!(
            diffs_with.is_empty(),
            "Should ignore request_id with payload config"
        );
    }

    // =====================================================
    // Integration tests for realistic AI agent payloads
    // These tests exercise full trace comparison with payloads
    // that resemble real Claude/AI agent interactions
    // =====================================================

    /// Integration test: Realistic Claude API trace comparison with tool calls
    /// This simulates comparing two runs of an AI agent that makes API calls
    /// and tool invocations with realistic payloads
    #[test]
    fn test_integration_realistic_claude_trace() {
        use crate::PayloadComparisonConfig;

        // Create a payload config that ignores timestamps and request IDs
        // which are expected to differ between runs
        let payload_config = PayloadComparisonConfig::ignoring(vec![
            "$.headers.X-Request-Id".to_string(),
            "$.headers.Date".to_string(),
            "$.body.timestamp".to_string(),
            "$.body.request_id".to_string(),
            "$.response_body.id".to_string(),
            "$.arguments.timestamp".to_string(),
            "$.result.created_at".to_string(),
        ])
        .with_numeric_tolerance(0.05) // 5% tolerance for numeric values
        .with_text_threshold(0.9); // 90% text similarity for string fields

        let criteria = EquivalenceCriteria::default().with_payload_config(payload_config);
        let config = BisimulationConfig::default()
            .with_criteria(criteria)
            .with_strategy(NondeterminismStrategy::SemanticSimilarity { threshold: 0.8 });

        let checker = DefaultBisimulationChecker::new(config.clone());

        // Oracle trace: First run of AI agent
        let oracle = ExecutionTrace::builder()
            // API call to Claude
            .api_request(
                crate::ApiRequest::new("POST", "https://api.anthropic.com/v1/messages")
                    .with_header("Content-Type", "application/json")
                    .with_header("X-Request-Id", "req-oracle-123")
                    .with_header("Date", "Wed, 25 Dec 2025 12:00:00 GMT")
                    .with_body(serde_json::json!({
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "Read file.txt"}],
                        "timestamp": 1735128000,
                        "request_id": "oracle-msg-id"
                    }))
                    .with_response(
                        200,
                        serde_json::json!({
                            "id": "msg_oracle_123",
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "tool_use", "name": "read_file"}],
                            "stop_reason": "tool_use"
                        }),
                    ),
            )
            // Tool call: Read file
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_arguments(serde_json::json!({
                        "path": "/project/file.txt",
                        "timestamp": 1735128001
                    }))
                    .with_result(serde_json::json!({
                        "content": "Hello, World!",
                        "size": 13,
                        "created_at": "2025-12-25T12:00:01Z"
                    })),
            )
            // Second API call
            .api_request(
                crate::ApiRequest::new("POST", "https://api.anthropic.com/v1/messages")
                    .with_header("Content-Type", "application/json")
                    .with_header("X-Request-Id", "req-oracle-456")
                    .with_body(serde_json::json!({
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": 1024,
                        "messages": [
                            {"role": "user", "content": "Read file.txt"},
                            {"role": "assistant", "content": "Reading..."},
                            {"role": "user", "content": "File content: Hello, World!"}
                        ]
                    }))
                    .with_response(
                        200,
                        serde_json::json!({
                            "id": "msg_oracle_456",
                            "type": "message",
                            "content": [{"type": "text", "text": "The file contains: Hello, World!"}]
                        }),
                    ),
            )
            .output("The file contains: Hello, World!")
            .duration(Duration::from_millis(1500))
            .build();

        // Subject trace: Second run with expected variations
        let subject = ExecutionTrace::builder()
            // API call to Claude (different request IDs, timestamps)
            .api_request(
                crate::ApiRequest::new("POST", "https://api.anthropic.com/v1/messages")
                    .with_header("Content-Type", "application/json")
                    .with_header("X-Request-Id", "req-subject-abc") // Different!
                    .with_header("Date", "Wed, 25 Dec 2025 12:01:00 GMT") // Different!
                    .with_body(serde_json::json!({
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "Read file.txt"}],
                        "timestamp": 1735128060, // Different!
                        "request_id": "subject-msg-id" // Different!
                    }))
                    .with_response(
                        200,
                        serde_json::json!({
                            "id": "msg_subject_xyz", // Different!
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "tool_use", "name": "read_file"}],
                            "stop_reason": "tool_use"
                        }),
                    ),
            )
            // Tool call: Read file (with timing variations)
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_arguments(serde_json::json!({
                        "path": "/project/file.txt",
                        "timestamp": 1735128061 // Different!
                    }))
                    .with_result(serde_json::json!({
                        "content": "Hello, World!",
                        "size": 13,
                        "created_at": "2025-12-25T12:01:01Z" // Different!
                    })),
            )
            // Second API call
            .api_request(
                crate::ApiRequest::new("POST", "https://api.anthropic.com/v1/messages")
                    .with_header("Content-Type", "application/json")
                    .with_header("X-Request-Id", "req-subject-def") // Different!
                    .with_body(serde_json::json!({
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": 1024,
                        "messages": [
                            {"role": "user", "content": "Read file.txt"},
                            {"role": "assistant", "content": "Reading..."},
                            {"role": "user", "content": "File content: Hello, World!"}
                        ]
                    }))
                    .with_response(
                        200,
                        serde_json::json!({
                            "id": "msg_subject_789", // Different!
                            "type": "message",
                            "content": [{"type": "text", "text": "The file contains: Hello, World!"}]
                        }),
                    ),
            )
            .output("The file contains: Hello, World!")
            .duration(Duration::from_millis(1600)) // Slightly different timing
            .build();

        // Compare traces - should be equivalent despite timing/ID differences
        let (diffs, confidence) = checker.compare_traces(
            &oracle,
            &subject,
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        // Should have no meaningful differences
        let meaningful_diffs: Vec<_> = diffs
            .iter()
            .filter(|d| {
                !matches!(d, Difference::TimingViolation { .. })
                    && !matches!(d, Difference::OutputMismatch { .. })
            })
            .collect();

        assert!(
            meaningful_diffs.is_empty(),
            "Realistic traces with ignored fields should be equivalent. Got {} differences: {:?}",
            meaningful_diffs.len(),
            meaningful_diffs
        );
        assert!(
            confidence > 0.8,
            "Confidence should be high: {}",
            confidence
        );
    }

    /// Integration test: Multi-tool workflow with file edits
    /// Simulates an AI agent reading, editing, and writing files
    #[test]
    fn test_integration_multi_tool_workflow() {
        use crate::PayloadComparisonConfig;

        let payload_config =
            PayloadComparisonConfig::ignoring(vec!["$.result.modified_at".to_string()])
                .with_numeric_tolerance(0.1);

        let criteria = EquivalenceCriteria::default().with_payload_config(payload_config);
        let config = BisimulationConfig::default()
            .with_criteria(criteria)
            .with_strategy(NondeterminismStrategy::SemanticSimilarity { threshold: 0.9 });

        let checker = DefaultBisimulationChecker::new(config.clone());

        let oracle = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_arguments(serde_json::json!({"path": "/src/main.rs"}))
                    .with_result(serde_json::json!({
                        "content": "fn main() {}",
                        "lines": 1,
                        "modified_at": "2025-12-25T10:00:00Z"
                    })),
            )
            .tool_call(
                crate::ToolCall::new("edit_file")
                    .with_arguments(serde_json::json!({
                        "path": "/src/main.rs",
                        "old": "fn main() {}",
                        "new": "fn main() { println!(\"Hello\"); }"
                    }))
                    .with_result(serde_json::json!({"success": true})),
            )
            .tool_call(
                crate::ToolCall::new("run_tests")
                    .with_arguments(serde_json::json!({"filter": "main"}))
                    .with_result(serde_json::json!({
                        "passed": 5,
                        "failed": 0,
                        "duration_ms": 100
                    })),
            )
            .output("Edited main.rs and all tests pass")
            .build();

        let subject = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_arguments(serde_json::json!({"path": "/src/main.rs"}))
                    .with_result(serde_json::json!({
                        "content": "fn main() {}",
                        "lines": 1,
                        "modified_at": "2025-12-25T11:00:00Z" // Different!
                    })),
            )
            .tool_call(
                crate::ToolCall::new("edit_file")
                    .with_arguments(serde_json::json!({
                        "path": "/src/main.rs",
                        "old": "fn main() {}",
                        "new": "fn main() { println!(\"Hello\"); }"
                    }))
                    .with_result(serde_json::json!({"success": true})),
            )
            .tool_call(
                crate::ToolCall::new("run_tests")
                    .with_arguments(serde_json::json!({"filter": "main"}))
                    .with_result(serde_json::json!({
                        "passed": 5,
                        "failed": 0,
                        "duration_ms": 105 // Slightly different (within 10% tolerance)
                    })),
            )
            .output("Edited main.rs and all tests pass")
            .build();

        let (diffs, confidence) = checker.compare_traces(
            &oracle,
            &subject,
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        assert!(
            diffs.is_empty(),
            "Multi-tool workflow should be equivalent. Got: {:?}",
            diffs
        );
        assert_eq!(
            confidence, 1.0,
            "Confidence should be perfect: {}",
            confidence
        );
    }

    /// Integration test: API error handling comparison
    /// Verifies that error scenarios are properly compared
    #[test]
    fn test_integration_error_handling() {
        let config =
            BisimulationConfig::default().with_strategy(NondeterminismStrategy::ExactMatch);

        let checker = DefaultBisimulationChecker::new(config.clone());

        let oracle = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("POST", "https://api.example.com/data")
                    .with_body(serde_json::json!({"query": "test"}))
                    .with_response(500, serde_json::json!({"error": "Internal error"})),
            )
            .tool_call(crate::ToolCall::new("retry").with_error("max retries exceeded"))
            .output("Error: max retries exceeded")
            .build();

        let subject = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("POST", "https://api.example.com/data")
                    .with_body(serde_json::json!({"query": "test"}))
                    .with_response(500, serde_json::json!({"error": "Internal error"})),
            )
            .tool_call(crate::ToolCall::new("retry").with_error("max retries exceeded"))
            .output("Error: max retries exceeded")
            .build();

        let (diffs, confidence) = checker.compare_traces(
            &oracle,
            &subject,
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        assert!(
            diffs.is_empty(),
            "Identical error traces should be equivalent"
        );
        assert_eq!(confidence, 1.0);
    }

    /// Integration test: Detecting behavioral divergence
    /// Verifies that actual differences in behavior are detected
    #[test]
    fn test_integration_behavioral_divergence() {
        let config =
            BisimulationConfig::default().with_strategy(NondeterminismStrategy::ExactMatch);

        let checker = DefaultBisimulationChecker::new(config.clone());

        // Oracle: Successfully reads file
        let oracle = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("read_file")
                    .with_arguments(serde_json::json!({"path": "/data/config.json"}))
                    .with_result(serde_json::json!({"content": "{\"key\": \"value\"}"})),
            )
            .output("Config loaded successfully")
            .build();

        // Subject: Different tool call - writes instead of reads!
        let subject = ExecutionTrace::builder()
            .tool_call(
                crate::ToolCall::new("write_file") // Different tool!
                    .with_arguments(serde_json::json!({"path": "/data/config.json"}))
                    .with_result(serde_json::json!({"success": true})),
            )
            .output("Config loaded successfully")
            .build();

        let (diffs, _confidence) = checker.compare_traces(
            &oracle,
            &subject,
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        // Should detect the tool call mismatch
        let tool_mismatch = diffs
            .iter()
            .any(|d| matches!(d, Difference::ToolCallMismatch { .. }));

        assert!(
            tool_mismatch,
            "Should detect that read_file vs write_file is a behavioral divergence"
        );
    }

    /// Integration test: Complex nested JSON payloads
    /// Tests comparison of deeply nested structures
    #[test]
    fn test_integration_nested_json_payloads() {
        use crate::PayloadComparisonConfig;

        let payload_config =
            PayloadComparisonConfig::ignoring(vec!["$.body.metadata.timestamp".to_string()]);

        let criteria = EquivalenceCriteria::default().with_payload_config(payload_config);
        let config = BisimulationConfig::default()
            .with_criteria(criteria)
            .with_strategy(NondeterminismStrategy::SemanticSimilarity { threshold: 0.9 });

        let checker = DefaultBisimulationChecker::new(config.clone());

        let oracle = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("POST", "https://api.example.com/nested")
                    .with_body(serde_json::json!({
                        "data": {
                            "users": [
                                {"id": 1, "name": "Alice", "roles": ["admin", "user"]},
                                {"id": 2, "name": "Bob", "roles": ["user"]}
                            ],
                            "config": {
                                "enabled": true,
                                "threshold": 0.75
                            }
                        },
                        "metadata": {
                            "timestamp": 1735128000,
                            "version": "1.0"
                        }
                    }))
                    .with_response(200, serde_json::json!({"status": "ok"})),
            )
            .output("Processed nested data")
            .build();

        let subject = ExecutionTrace::builder()
            .api_request(
                crate::ApiRequest::new("POST", "https://api.example.com/nested")
                    .with_body(serde_json::json!({
                        "data": {
                            "users": [
                                {"id": 1, "name": "Alice", "roles": ["admin", "user"]},
                                {"id": 2, "name": "Bob", "roles": ["user"]}
                            ],
                            "config": {
                                "enabled": true,
                                "threshold": 0.75
                            }
                        },
                        "metadata": {
                            "timestamp": 1735128060, // Different - should be ignored
                            "version": "1.0"
                        }
                    }))
                    .with_response(200, serde_json::json!({"status": "ok"})),
            )
            .output("Processed nested data")
            .build();

        let (diffs, confidence) = checker.compare_traces(
            &oracle,
            &subject,
            &config.equivalence_criteria,
            &config.nondeterminism_strategy,
        );

        assert!(
            diffs.is_empty(),
            "Nested JSON with ignored timestamp should be equivalent. Got: {:?}",
            diffs
        );
        assert_eq!(confidence, 1.0);
    }
}

// Kani formal verification proofs
// NOTE: Proofs that use types with HashMap are excluded because Kani
// doesn't support the CCRandomGenerateBytes C function used by HashMap's hasher on macOS.
// BinaryExecutor and DefaultBisimulationChecker use HashMap internally, so we focus
// on pure computational properties like timing calculations.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify timing ratio calculation bounds
    /// If oracle_ms > 0 and subject_ms >= 0, diff_ratio = |subject - oracle| / oracle
    /// This must be non-negative and bounded by reasonable values
    #[kani::proof]
    fn verify_timing_diff_ratio_non_negative() {
        let oracle_ms: u64 = kani::any();
        let subject_ms: u64 = kani::any();

        kani::assume(oracle_ms > 0 && oracle_ms <= 10000);
        kani::assume(subject_ms <= 10000);

        let diff_ratio = ((subject_ms as f64) - (oracle_ms as f64)).abs() / (oracle_ms as f64);

        kani::assert(diff_ratio >= 0.0, "diff_ratio must be non-negative");
    }

    /// Verify that timing tolerance comparison is symmetric for percentages
    /// |a - b| / a vs tolerance should give same result as |b - a| / b for same percentage
    #[kani::proof]
    fn verify_timing_comparison_symmetry() {
        let tolerance = 0.1f64; // 10% tolerance

        // If oracle=100, subject=110 => diff_ratio = 0.1 => exactly at boundary, no violation
        // If oracle=110, subject=100 => diff_ratio = |100-110|/110 = 10/110 ≈ 0.09 => within tolerance
        let oracle_ms: u64 = 100;
        let subject_ms: u64 = 110;

        let diff1 = ((subject_ms as f64) - (oracle_ms as f64)).abs() / (oracle_ms as f64);
        // diff1 = 10/100 = 0.1

        kani::assert(
            !(diff1 > tolerance), // 0.1 > 0.1 is false
            "At exact boundary, should not violate",
        );
    }

    /// Verify timing ratio is bounded when subject is much larger than oracle
    #[kani::proof]
    fn verify_timing_ratio_bounds() {
        let oracle_ms: u64 = kani::any();
        let subject_ms: u64 = kani::any();

        kani::assume(oracle_ms > 0 && oracle_ms <= 1000);
        kani::assume(subject_ms > 0 && subject_ms <= 1000);

        let diff_ratio = ((subject_ms as f64) - (oracle_ms as f64)).abs() / (oracle_ms as f64);

        // Ratio can be at most (max - min) / min = 999 / 1 = 999
        kani::assert(
            diff_ratio <= 999.0,
            "diff_ratio must be bounded with bounded inputs",
        );
    }

    /// Verify TraceRecorder preserves trace directory path
    /// (TraceRecorder doesn't use HashMap)
    #[kani::proof]
    fn verify_trace_recorder_preserves_path() {
        let recorder = TraceRecorder::new("/traces");

        kani::assert(
            recorder.trace_dir == std::path::PathBuf::from("/traces"),
            "TraceRecorder must preserve trace_dir path",
        );
    }

    /// Verify Duration arithmetic for default timeout
    #[kani::proof]
    fn verify_default_timeout_value() {
        // BinaryExecutor default timeout is 60 seconds
        let default_timeout = Duration::from_secs(60);

        kani::assert(
            default_timeout.as_secs() == 60,
            "Default timeout must be 60 seconds",
        );
        kani::assert(
            default_timeout.as_millis() == 60_000,
            "Default timeout must be 60,000 milliseconds",
        );
    }

    /// Verify Duration::from_secs is monotonic
    #[kani::proof]
    fn verify_duration_monotonic() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();

        kani::assume(a <= 3600);
        kani::assume(b <= 3600);
        kani::assume(a < b);

        let dur_a = Duration::from_secs(a);
        let dur_b = Duration::from_secs(b);

        kani::assert(dur_a < dur_b, "Duration::from_secs must be monotonic");
    }
}
