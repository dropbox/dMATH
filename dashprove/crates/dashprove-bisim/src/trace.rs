//! Execution trace types for bisimulation checking

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Execution trace capturing all observable behavior
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// API requests made during execution
    pub api_requests: Vec<ApiRequest>,
    /// Tool calls made during execution
    pub tool_calls: Vec<ToolCall>,
    /// Final output produced
    pub output: String,
    /// Total execution duration
    #[serde(with = "duration_serde")]
    pub duration: Duration,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ExecutionTrace {
    /// Create an empty trace
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create a new trace builder
    pub fn builder() -> ExecutionTraceBuilder {
        ExecutionTraceBuilder::default()
    }

    /// Add an API request to the trace
    pub fn add_api_request(&mut self, request: ApiRequest) {
        self.api_requests.push(request);
    }

    /// Add a tool call to the trace
    pub fn add_tool_call(&mut self, call: ToolCall) {
        self.tool_calls.push(call);
    }

    /// Get count of all events (requests + tool calls)
    pub fn event_count(&self) -> usize {
        self.api_requests.len() + self.tool_calls.len()
    }
}

/// Builder for ExecutionTrace
#[derive(Debug, Default)]
pub struct ExecutionTraceBuilder {
    api_requests: Vec<ApiRequest>,
    tool_calls: Vec<ToolCall>,
    output: String,
    duration: Duration,
    metadata: HashMap<String, serde_json::Value>,
}

impl ExecutionTraceBuilder {
    /// Add an API request
    pub fn api_request(mut self, request: ApiRequest) -> Self {
        self.api_requests.push(request);
        self
    }

    /// Add a tool call
    pub fn tool_call(mut self, call: ToolCall) -> Self {
        self.tool_calls.push(call);
        self
    }

    /// Set the output
    pub fn output(mut self, output: impl Into<String>) -> Self {
        self.output = output.into();
        self
    }

    /// Set the duration
    pub fn duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Build the trace
    pub fn build(self) -> ExecutionTrace {
        ExecutionTrace {
            api_requests: self.api_requests,
            tool_calls: self.tool_calls,
            output: self.output,
            duration: self.duration,
            metadata: self.metadata,
        }
    }
}

/// An API request made during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRequest {
    /// HTTP method (GET, POST, etc.)
    pub method: String,
    /// Request URL
    pub url: String,
    /// Request headers
    pub headers: HashMap<String, String>,
    /// Request body as JSON
    pub body: serde_json::Value,
    /// Response status code
    #[serde(default)]
    pub response_status: Option<u16>,
    /// Response body as JSON
    #[serde(default)]
    pub response_body: Option<serde_json::Value>,
}

impl ApiRequest {
    /// Create a new API request
    pub fn new(method: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            url: url.into(),
            headers: HashMap::new(),
            body: serde_json::Value::Null,
            response_status: None,
            response_body: None,
        }
    }

    /// Add a header
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Set the body
    pub fn with_body(mut self, body: serde_json::Value) -> Self {
        self.body = body;
        self
    }

    /// Set the response
    pub fn with_response(mut self, status: u16, body: serde_json::Value) -> Self {
        self.response_status = Some(status);
        self.response_body = Some(body);
        self
    }
}

/// A tool call made during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool name
    pub name: String,
    /// Tool arguments as JSON
    pub arguments: serde_json::Value,
    /// Tool result as JSON
    pub result: serde_json::Value,
    /// Whether the call succeeded
    #[serde(default = "default_success")]
    pub success: bool,
    /// Error message if failed
    #[serde(default)]
    pub error: Option<String>,
}

fn default_success() -> bool {
    true
}

impl ToolCall {
    /// Create a new tool call
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: serde_json::Value::Null,
            result: serde_json::Value::Null,
            success: true,
            error: None,
        }
    }

    /// Set the arguments
    pub fn with_arguments(mut self, args: serde_json::Value) -> Self {
        self.arguments = args;
        self
    }

    /// Set the result
    pub fn with_result(mut self, result: serde_json::Value) -> Self {
        self.result = result;
        self
    }

    /// Mark as failed with an error
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.success = false;
        self.error = Some(error.into());
        self
    }
}

/// Custom serialization for Duration
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_millis().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Property tests
    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// ExecutionTraceBuilder should preserve output field
            #[test]
            fn trace_builder_preserves_output(output in ".*") {
                let trace = ExecutionTrace::builder()
                    .output(output.clone())
                    .build();
                prop_assert_eq!(trace.output, output);
            }

            /// ExecutionTraceBuilder should preserve duration
            #[test]
            fn trace_builder_preserves_duration(millis in 0u64..1_000_000) {
                let duration = Duration::from_millis(millis);
                let trace = ExecutionTrace::builder()
                    .duration(duration)
                    .build();
                prop_assert_eq!(trace.duration, duration);
            }

            /// ExecutionTrace event_count should equal sum of api_requests and tool_calls
            #[test]
            fn trace_event_count_correct(
                api_count in 0usize..10,
                tool_count in 0usize..10
            ) {
                let mut builder = ExecutionTrace::builder();
                for i in 0..api_count {
                    builder = builder.api_request(
                        ApiRequest::new("GET", format!("https://api.test/{}", i))
                    );
                }
                for i in 0..tool_count {
                    builder = builder.tool_call(
                        ToolCall::new(format!("tool_{}", i))
                    );
                }
                let trace = builder.build();
                prop_assert_eq!(trace.event_count(), api_count + tool_count);
            }

            /// ApiRequest builder should preserve method and url
            #[test]
            fn api_request_preserves_fields(
                method in "(GET|POST|PUT|DELETE)",
                url in "https://[a-z]+\\.[a-z]+/[a-z]+"
            ) {
                let req = ApiRequest::new(method.clone(), url.clone());
                prop_assert_eq!(req.method, method);
                prop_assert_eq!(req.url, url);
            }

            /// ApiRequest with_header should add header
            #[test]
            fn api_request_with_header(
                key in "[A-Za-z-]+",
                value in "[a-z0-9]+"
            ) {
                let req = ApiRequest::new("GET", "https://test.com")
                    .with_header(key.clone(), value.clone());
                prop_assert_eq!(req.headers.get(&key), Some(&value));
            }

            /// ApiRequest with_response should set response fields
            #[test]
            fn api_request_with_response(status in 100u16..600) {
                let body = serde_json::json!({"ok": true});
                let req = ApiRequest::new("GET", "https://test.com")
                    .with_response(status, body.clone());
                prop_assert_eq!(req.response_status, Some(status));
                prop_assert_eq!(req.response_body, Some(body));
            }

            /// ToolCall builder should preserve name
            #[test]
            fn tool_call_preserves_name(name in "[a-z_]+") {
                let call = ToolCall::new(name.clone());
                prop_assert_eq!(call.name, name);
                prop_assert!(call.success);
                prop_assert!(call.error.is_none());
            }

            /// ToolCall with_error should set success to false
            #[test]
            fn tool_call_with_error_sets_failed(error in ".+") {
                let call = ToolCall::new("test")
                    .with_error(error.clone());
                prop_assert!(!call.success);
                prop_assert_eq!(call.error, Some(error));
            }

            /// ExecutionTrace serialization roundtrip should preserve data
            #[test]
            fn trace_serialization_roundtrip(
                output in ".*",
                millis in 0u64..1_000_000
            ) {
                let original = ExecutionTrace::builder()
                    .output(output)
                    .duration(Duration::from_millis(millis))
                    .build();

                let json = serde_json::to_string(&original).unwrap();
                let parsed: ExecutionTrace = serde_json::from_str(&json).unwrap();

                prop_assert_eq!(parsed.output, original.output);
                prop_assert_eq!(parsed.duration, original.duration);
            }

            /// Adding metadata should preserve it
            #[test]
            fn trace_builder_preserves_metadata(
                key in "[a-z]+",
                value in any::<i64>()
            ) {
                let trace = ExecutionTrace::builder()
                    .metadata(key.clone(), serde_json::json!(value))
                    .build();
                prop_assert_eq!(
                    trace.metadata.get(&key),
                    Some(&serde_json::json!(value))
                );
            }
        }
    }

    #[test]
    fn test_execution_trace_builder() {
        let trace = ExecutionTrace::builder()
            .api_request(ApiRequest::new("POST", "https://api.example.com/v1/chat"))
            .tool_call(
                ToolCall::new("read_file").with_arguments(serde_json::json!({"path": "/test.txt"})),
            )
            .output("Hello, world!")
            .duration(Duration::from_millis(1500))
            .build();

        assert_eq!(trace.api_requests.len(), 1);
        assert_eq!(trace.tool_calls.len(), 1);
        assert_eq!(trace.output, "Hello, world!");
        assert_eq!(trace.duration.as_millis(), 1500);
    }

    #[test]
    fn test_api_request_builder() {
        let req = ApiRequest::new("POST", "https://example.com")
            .with_header("Content-Type", "application/json")
            .with_body(serde_json::json!({"message": "hello"}))
            .with_response(200, serde_json::json!({"status": "ok"}));

        assert_eq!(req.method, "POST");
        assert_eq!(
            req.headers.get("Content-Type"),
            Some(&"application/json".to_string())
        );
        assert_eq!(req.response_status, Some(200));
    }

    #[test]
    fn test_tool_call_builder() {
        let call = ToolCall::new("bash")
            .with_arguments(serde_json::json!({"command": "ls"}))
            .with_result(serde_json::json!({"output": "file1.txt\nfile2.txt"}));

        assert_eq!(call.name, "bash");
        assert!(call.success);
        assert!(call.error.is_none());

        let failed_call = ToolCall::new("bash").with_error("command not found");
        assert!(!failed_call.success);
        assert!(failed_call.error.is_some());
    }

    #[test]
    fn test_trace_serialization() {
        let trace = ExecutionTrace::builder()
            .output("test")
            .duration(Duration::from_secs(1))
            .build();

        let json = serde_json::to_string(&trace).unwrap();
        let parsed: ExecutionTrace = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.output, "test");
        assert_eq!(parsed.duration.as_secs(), 1);
    }

    // =====================================================
    // Mutation-killing tests for ExecutionTrace
    // =====================================================

    /// Test ExecutionTrace::empty returns actually empty trace
    /// (catches mutation: replace empty -> Self with Default::default())
    #[test]
    fn test_execution_trace_empty_is_truly_empty() {
        let empty = ExecutionTrace::empty();
        assert!(empty.api_requests.is_empty(), "Should have no API requests");
        assert!(empty.tool_calls.is_empty(), "Should have no tool calls");
        assert!(empty.output.is_empty(), "Should have empty output");
        assert!(empty.metadata.is_empty(), "Should have no metadata");
        assert_eq!(empty.duration, Duration::ZERO, "Should have zero duration");
    }

    /// Test ExecutionTrace::builder returns working builder
    /// (catches mutation: replace builder -> ExecutionTraceBuilder with Default::default())
    #[test]
    fn test_execution_trace_builder_works() {
        let builder = ExecutionTrace::builder();
        let trace = builder
            .output("test output")
            .duration(Duration::from_millis(500))
            .build();
        assert_eq!(trace.output, "test output");
        assert_eq!(trace.duration, Duration::from_millis(500));
    }

    /// Test ExecutionTrace::add_api_request actually adds request
    /// (catches mutation: replace add_api_request with ())
    #[test]
    fn test_add_api_request_modifies_trace() {
        let mut trace = ExecutionTrace::empty();
        assert_eq!(trace.api_requests.len(), 0);

        trace.add_api_request(ApiRequest::new("GET", "https://example.com"));
        assert_eq!(trace.api_requests.len(), 1);
        assert_eq!(trace.api_requests[0].method, "GET");
        assert_eq!(trace.api_requests[0].url, "https://example.com");

        // Add another
        trace.add_api_request(ApiRequest::new("POST", "https://example.com/data"));
        assert_eq!(trace.api_requests.len(), 2);
    }

    /// Test ExecutionTrace::add_tool_call actually adds tool call
    /// (catches mutation: replace add_tool_call with ())
    #[test]
    fn test_add_tool_call_modifies_trace() {
        let mut trace = ExecutionTrace::empty();
        assert_eq!(trace.tool_calls.len(), 0);

        trace.add_tool_call(ToolCall::new("read_file"));
        assert_eq!(trace.tool_calls.len(), 1);
        assert_eq!(trace.tool_calls[0].name, "read_file");

        // Add another
        trace.add_tool_call(ToolCall::new("write_file"));
        assert_eq!(trace.tool_calls.len(), 2);
    }

    /// Test default_success returns true
    /// (catches mutation: replace default_success -> bool with false)
    #[test]
    fn test_default_success_is_true() {
        // Deserialize a ToolCall without explicit success field
        let json = r#"{"name": "test", "arguments": null, "result": null}"#;
        let call: ToolCall = serde_json::from_str(json).unwrap();
        assert!(call.success, "Default success should be true");
    }

    /// Test that new ToolCall has success=true by default
    #[test]
    fn test_new_tool_call_success_default() {
        let call = ToolCall::new("test");
        assert!(call.success, "New ToolCall should have success=true");
        assert!(call.error.is_none(), "New ToolCall should have no error");
    }
}

// Kani formal verification proofs
// NOTE: Proofs that would trigger HashMap random generation are excluded.
// Kani doesn't support CCRandomGenerateBytes used by HashMap's hasher on macOS.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify ToolCall::new sets success=true and error=None
    #[kani::proof]
    fn verify_tool_call_new_defaults() {
        let call = ToolCall::new("test");

        kani::assert(call.success, "New ToolCall must have success=true");
        kani::assert(call.error.is_none(), "New ToolCall must have error=None");
    }

    /// Verify ToolCall::with_error sets success=false
    #[kani::proof]
    fn verify_tool_call_with_error_sets_failed() {
        let call = ToolCall::new("test").with_error("error message");

        kani::assert(!call.success, "with_error must set success=false");
        kani::assert(call.error.is_some(), "with_error must set error to Some");
    }

    /// Verify ToolCall name is preserved
    #[kani::proof]
    fn verify_tool_call_name_preserved() {
        let call = ToolCall::new("my_tool");
        kani::assert(call.name == "my_tool", "ToolCall name must be preserved");
    }

    /// Verify default_success returns true
    #[kani::proof]
    fn verify_default_success_returns_true() {
        let result = default_success();
        kani::assert(result, "default_success must return true");
    }

    /// Verify Duration serialization roundtrip for milliseconds
    #[kani::proof]
    fn verify_duration_millis_roundtrip() {
        let millis: u64 = kani::any();
        kani::assume(millis <= 1_000_000_000); // ~11.5 days max

        let d = Duration::from_millis(millis);
        kani::assert(
            d.as_millis() as u64 == millis,
            "Duration::from_millis must preserve milliseconds",
        );
    }
}
