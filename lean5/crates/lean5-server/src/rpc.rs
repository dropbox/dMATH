//! JSON-RPC 2.0 protocol handling
//!
//! Implements the JSON-RPC 2.0 specification for the Lean5 server.
//! See: <https://www.jsonrpc.org/specification>

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC version string
pub const JSONRPC_VERSION: &str = "2.0";

/// JSON-RPC request ID (can be string, number, or null)
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    #[default]
    Null,
    Number(i64),
    String(String),
}

/// JSON-RPC 2.0 request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// Protocol version (must be "2.0")
    pub jsonrpc: String,
    /// Method name
    pub method: String,
    /// Parameters (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    /// Request ID (None for notifications)
    #[serde(default, skip_serializing_if = "is_null_id")]
    pub id: Option<RequestId>,
}

fn is_null_id(id: &Option<RequestId>) -> bool {
    matches!(id, None | Some(RequestId::Null))
}

impl Request {
    /// Create a new request
    pub fn new(method: impl Into<String>, params: Option<Value>, id: RequestId) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.into(),
            params,
            id: Some(id),
        }
    }

    /// Create a notification (no id, no response expected)
    pub fn notification(method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.into(),
            params,
            id: None,
        }
    }

    /// Check if this is a notification (no response expected)
    #[must_use]
    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }
}

/// JSON-RPC 2.0 response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// Protocol version (must be "2.0")
    pub jsonrpc: String,
    /// Result (on success)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error (on failure)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
    /// Request ID (same as request, null for errors without id)
    pub id: RequestId,
}

impl Response {
    /// Create a success response
    #[must_use]
    pub fn success(id: RequestId, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: Some(result),
            error: None,
            id,
        }
    }

    /// Create an error response
    #[must_use]
    pub fn error(id: RequestId, error: RpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: None,
            error: Some(error),
            id,
        }
    }

    /// Create a success response from a serializable value
    pub fn success_typed<T: Serialize>(
        id: RequestId,
        result: &T,
    ) -> Result<Self, serde_json::Error> {
        Ok(Self::success(id, serde_json::to_value(result)?))
    }
}

/// JSON-RPC 2.0 error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Additional data (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Standard JSON-RPC error codes
pub mod error_codes {
    /// Parse error: Invalid JSON was received
    pub const PARSE_ERROR: i32 = -32700;
    /// Invalid Request: The JSON sent is not a valid Request object
    pub const INVALID_REQUEST: i32 = -32600;
    /// Method not found: The method does not exist
    pub const METHOD_NOT_FOUND: i32 = -32601;
    /// Invalid params: Invalid method parameter(s)
    pub const INVALID_PARAMS: i32 = -32602;
    /// Internal error: Internal JSON-RPC error
    pub const INTERNAL_ERROR: i32 = -32603;

    // Server error: Reserved for implementation-defined server-errors (-32000 to -32099)

    /// Type checking error
    pub const TYPE_ERROR: i32 = -32000;
    /// Parse error (Lean syntax)
    pub const LEAN_PARSE_ERROR: i32 = -32001;
    /// Elaboration error
    pub const ELABORATION_ERROR: i32 = -32002;
    /// Proof search failed
    pub const PROOF_NOT_FOUND: i32 = -32003;
    /// Timeout
    pub const TIMEOUT: i32 = -32004;
}

impl RpcError {
    /// Create a new error
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// Create an error with additional data
    pub fn with_data(code: i32, message: impl Into<String>, data: Value) -> Self {
        Self {
            code,
            message: message.into(),
            data: Some(data),
        }
    }

    /// Parse error
    pub fn parse_error(details: impl Into<String>) -> Self {
        Self::new(error_codes::PARSE_ERROR, details)
    }

    /// Invalid request
    pub fn invalid_request(details: impl Into<String>) -> Self {
        Self::new(error_codes::INVALID_REQUEST, details)
    }

    /// Method not found
    #[must_use]
    pub fn method_not_found(method: &str) -> Self {
        Self::new(
            error_codes::METHOD_NOT_FOUND,
            format!("Method not found: {method}"),
        )
    }

    /// Invalid params
    pub fn invalid_params(details: impl Into<String>) -> Self {
        Self::new(error_codes::INVALID_PARAMS, details)
    }

    /// Internal error
    pub fn internal_error(details: impl Into<String>) -> Self {
        Self::new(error_codes::INTERNAL_ERROR, details)
    }

    /// Type error
    pub fn type_error(details: impl Into<String>) -> Self {
        Self::new(error_codes::TYPE_ERROR, details)
    }

    /// Parse error (Lean)
    pub fn lean_parse_error(details: impl Into<String>) -> Self {
        Self::new(error_codes::LEAN_PARSE_ERROR, details)
    }

    /// Elaboration error
    pub fn elaboration_error(details: impl Into<String>) -> Self {
        Self::new(error_codes::ELABORATION_ERROR, details)
    }

    /// Proof not found
    #[must_use]
    pub fn proof_not_found() -> Self {
        Self::new(error_codes::PROOF_NOT_FOUND, "Proof search failed")
    }

    /// Timeout
    #[must_use]
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::new(
            error_codes::TIMEOUT,
            format!("Operation timed out after {timeout_ms}ms"),
        )
    }
}

/// Batch request (array of requests)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BatchRequest(pub Vec<Request>);

/// Batch response (array of responses)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BatchResponse(pub Vec<Response>);

/// Parse a JSON-RPC message (single or batch)
pub fn parse_message(json: &str) -> Result<ParsedMessage, RpcError> {
    // First, try to parse as valid JSON
    let value: Value = serde_json::from_str(json)
        .map_err(|e| RpcError::parse_error(format!("Invalid JSON: {e}")))?;

    // Check if it's a batch (array) or single request (object)
    match value {
        Value::Array(arr) => {
            if arr.is_empty() {
                return Err(RpcError::invalid_request("Empty batch request"));
            }
            let mut requests = Vec::with_capacity(arr.len());
            for item in arr {
                let req: Request = serde_json::from_value(item).map_err(|e| {
                    RpcError::invalid_request(format!("Invalid request in batch: {e}"))
                })?;
                requests.push(req);
            }
            Ok(ParsedMessage::Batch(BatchRequest(requests)))
        }
        Value::Object(_) => {
            let req: Request = serde_json::from_value(value)
                .map_err(|e| RpcError::invalid_request(format!("Invalid request: {e}")))?;
            Ok(ParsedMessage::Single(req))
        }
        _ => Err(RpcError::invalid_request(
            "Request must be an object or array",
        )),
    }
}

/// Parsed message (single or batch)
#[derive(Debug, Clone)]
pub enum ParsedMessage {
    Single(Request),
    Batch(BatchRequest),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_request() {
        let json =
            r#"{"jsonrpc": "2.0", "method": "check", "params": {"code": "def x := 1"}, "id": 1}"#;
        let msg = parse_message(json).unwrap();
        match msg {
            ParsedMessage::Single(req) => {
                assert_eq!(req.jsonrpc, "2.0");
                assert_eq!(req.method, "check");
                assert_eq!(req.id, Some(RequestId::Number(1)));
            }
            _ => panic!("Expected single request"),
        }
    }

    #[test]
    fn test_parse_notification() {
        let json = r#"{"jsonrpc": "2.0", "method": "cancel"}"#;
        let msg = parse_message(json).unwrap();
        match msg {
            ParsedMessage::Single(req) => {
                assert!(req.is_notification());
                assert_eq!(req.method, "cancel");
            }
            _ => panic!("Expected single request"),
        }
    }

    #[test]
    fn test_parse_batch_request() {
        let json = r#"[
            {"jsonrpc": "2.0", "method": "check", "params": {"code": "1"}, "id": 1},
            {"jsonrpc": "2.0", "method": "check", "params": {"code": "2"}, "id": 2}
        ]"#;
        let msg = parse_message(json).unwrap();
        match msg {
            ParsedMessage::Batch(batch) => {
                assert_eq!(batch.0.len(), 2);
            }
            _ => panic!("Expected batch request"),
        }
    }

    #[test]
    fn test_parse_empty_batch_error() {
        let json = "[]";
        let err = parse_message(json).unwrap_err();
        assert_eq!(err.code, error_codes::INVALID_REQUEST);
    }

    #[test]
    fn test_parse_invalid_json() {
        let json = "{invalid}";
        let err = parse_message(json).unwrap_err();
        assert_eq!(err.code, error_codes::PARSE_ERROR);
    }

    #[test]
    fn test_serialize_response() {
        let resp = Response::success(RequestId::Number(1), serde_json::json!({"valid": true}));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"result\""));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn test_serialize_error_response() {
        let resp = Response::error(
            RequestId::String("abc".into()),
            RpcError::method_not_found("unknown"),
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"error\""));
        assert!(json.contains("-32601"));
    }

    #[test]
    fn test_request_id_types() {
        // Number ID
        let json = r#"{"jsonrpc": "2.0", "method": "test", "id": 42}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        assert_eq!(req.id, Some(RequestId::Number(42)));

        // String ID
        let json = r#"{"jsonrpc": "2.0", "method": "test", "id": "abc-123"}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        assert_eq!(req.id, Some(RequestId::String("abc-123".into())));

        // Null ID - in JSON-RPC, null id is typically treated as missing
        // (notification) so it deserializes to None
        let json = r#"{"jsonrpc": "2.0", "method": "test", "id": null}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        // Note: JSON null deserializes to None for Option<T>
        assert!(req.id.is_none() || req.id == Some(RequestId::Null));
    }

    #[test]
    fn test_request_new_constructor() {
        let req = Request::new(
            "testMethod",
            Some(serde_json::json!({"key": "value"})),
            RequestId::Number(42),
        );
        assert_eq!(req.jsonrpc, JSONRPC_VERSION);
        assert_eq!(req.method, "testMethod");
        assert_eq!(req.id, Some(RequestId::Number(42)));
        assert!(req.params.is_some());
    }

    #[test]
    fn test_request_notification_constructor() {
        let req = Request::notification("cancel", None);
        assert_eq!(req.jsonrpc, JSONRPC_VERSION);
        assert_eq!(req.method, "cancel");
        assert!(req.is_notification());
        assert!(req.id.is_none());
    }

    #[test]
    fn test_request_is_notification_false() {
        let req = Request::new("check", None, RequestId::Number(1));
        assert!(!req.is_notification());
    }

    #[test]
    fn test_response_success_typed() {
        #[derive(serde::Serialize)]
        struct TestResult {
            valid: bool,
            count: i32,
        }
        let result = TestResult {
            valid: true,
            count: 42,
        };
        let resp = Response::success_typed(RequestId::Number(1), &result).unwrap();

        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
        let result_value = resp.result.unwrap();
        assert_eq!(result_value["valid"], true);
        assert_eq!(result_value["count"], 42);
    }

    #[test]
    fn test_rpc_error_new() {
        let err = RpcError::new(-32000, "Custom error");
        assert_eq!(err.code, -32000);
        assert_eq!(err.message, "Custom error");
        assert!(err.data.is_none());
    }

    #[test]
    fn test_rpc_error_with_data() {
        let data = serde_json::json!({"line": 10, "column": 5});
        let err = RpcError::with_data(-32001, "Parse error", data.clone());
        assert_eq!(err.code, -32001);
        assert_eq!(err.message, "Parse error");
        assert_eq!(err.data, Some(data));
    }

    #[test]
    fn test_rpc_error_constructors() {
        // Test all error constructor methods
        let err = RpcError::parse_error("invalid syntax");
        assert_eq!(err.code, error_codes::PARSE_ERROR);

        let err = RpcError::invalid_request("missing field");
        assert_eq!(err.code, error_codes::INVALID_REQUEST);

        let err = RpcError::method_not_found("unknown");
        assert_eq!(err.code, error_codes::METHOD_NOT_FOUND);
        assert!(err.message.contains("unknown"));

        let err = RpcError::invalid_params("wrong type");
        assert_eq!(err.code, error_codes::INVALID_PARAMS);

        let err = RpcError::internal_error("server crash");
        assert_eq!(err.code, error_codes::INTERNAL_ERROR);

        let err = RpcError::type_error("type mismatch");
        assert_eq!(err.code, error_codes::TYPE_ERROR);

        let err = RpcError::lean_parse_error("syntax error");
        assert_eq!(err.code, error_codes::LEAN_PARSE_ERROR);

        let err = RpcError::elaboration_error("unification failed");
        assert_eq!(err.code, error_codes::ELABORATION_ERROR);

        let err = RpcError::proof_not_found();
        assert_eq!(err.code, error_codes::PROOF_NOT_FOUND);

        let err = RpcError::timeout(5000);
        assert_eq!(err.code, error_codes::TIMEOUT);
        assert!(err.message.contains("5000"));
    }

    #[test]
    fn test_batch_request_serde() {
        let batch = BatchRequest(vec![
            Request::new("check", None, RequestId::Number(1)),
            Request::new("check", None, RequestId::Number(2)),
        ]);

        let json = serde_json::to_string(&batch).unwrap();
        assert!(json.starts_with('['));

        let parsed: BatchRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.0.len(), 2);
    }

    #[test]
    fn test_batch_response_serde() {
        let batch = BatchResponse(vec![
            Response::success(RequestId::Number(1), serde_json::json!({"ok": true})),
            Response::error(RequestId::Number(2), RpcError::internal_error("test")),
        ]);

        let json = serde_json::to_string(&batch).unwrap();
        assert!(json.starts_with('['));

        let parsed: BatchResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.0.len(), 2);
    }

    #[test]
    fn test_parse_message_primitive_types() {
        // Primitive values should be rejected
        let err = parse_message("42").unwrap_err();
        assert_eq!(err.code, error_codes::INVALID_REQUEST);

        let err = parse_message("\"string\"").unwrap_err();
        assert_eq!(err.code, error_codes::INVALID_REQUEST);

        let err = parse_message("true").unwrap_err();
        assert_eq!(err.code, error_codes::INVALID_REQUEST);

        let err = parse_message("null").unwrap_err();
        assert_eq!(err.code, error_codes::INVALID_REQUEST);
    }

    #[test]
    fn test_parse_message_invalid_request_in_batch() {
        // Batch with an invalid request object
        let json = r#"[{"jsonrpc": "2.0", "method": "test", "id": 1}, {"not_valid": true}]"#;
        let err = parse_message(json).unwrap_err();
        assert_eq!(err.code, error_codes::INVALID_REQUEST);
    }

    #[test]
    fn test_request_id_default() {
        let id = RequestId::default();
        assert_eq!(id, RequestId::Null);
    }

    #[test]
    fn test_request_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(RequestId::Number(1));
        set.insert(RequestId::String("abc".into()));
        set.insert(RequestId::Null);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_response_clone() {
        let resp = Response::success(RequestId::Number(1), serde_json::json!({"test": true}));
        let cloned = resp.clone();
        assert_eq!(cloned.id, resp.id);
        assert_eq!(cloned.result, resp.result);
    }

    #[test]
    fn test_rpc_error_clone() {
        let err = RpcError::with_data(-32000, "test", serde_json::json!({"extra": "data"}));
        let cloned = err.clone();
        assert_eq!(cloned.code, err.code);
        assert_eq!(cloned.message, err.message);
        assert_eq!(cloned.data, err.data);
    }

    #[test]
    fn test_request_debug() {
        let req = Request::new("test", None, RequestId::Number(1));
        let debug_str = format!("{req:?}");
        assert!(debug_str.contains("Request"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_response_debug() {
        let resp = Response::success(RequestId::Number(1), serde_json::json!(null));
        let debug_str = format!("{resp:?}");
        assert!(debug_str.contains("Response"));
    }

    #[test]
    fn test_rpc_error_debug() {
        let err = RpcError::internal_error("debug test");
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("RpcError"));
    }

    #[test]
    fn test_parsed_message_debug() {
        let json = r#"{"jsonrpc": "2.0", "method": "test", "id": 1}"#;
        let msg = parse_message(json).unwrap();
        let debug_str = format!("{msg:?}");
        assert!(debug_str.contains("Single"));
    }

    #[test]
    fn test_is_null_id_helper() {
        // Test the internal helper function through serialization behavior
        let req = Request {
            jsonrpc: "2.0".to_string(),
            method: "test".to_string(),
            params: None,
            id: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        // id should be skipped when None
        assert!(!json.contains("\"id\""));

        let req = Request {
            jsonrpc: "2.0".to_string(),
            method: "test".to_string(),
            params: None,
            id: Some(RequestId::Null),
        };
        let json = serde_json::to_string(&req).unwrap();
        // id should be skipped when Null
        assert!(!json.contains("\"id\""));

        let req = Request {
            jsonrpc: "2.0".to_string(),
            method: "test".to_string(),
            params: None,
            id: Some(RequestId::Number(1)),
        };
        let json = serde_json::to_string(&req).unwrap();
        // id should be present when it has a value
        assert!(json.contains("\"id\""));
    }

    #[test]
    fn test_negative_request_id() {
        let json = r#"{"jsonrpc": "2.0", "method": "test", "id": -42}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        assert_eq!(req.id, Some(RequestId::Number(-42)));
    }

    #[test]
    fn test_empty_string_request_id() {
        let json = r#"{"jsonrpc": "2.0", "method": "test", "id": ""}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        assert_eq!(req.id, Some(RequestId::String("".into())));
    }
}
