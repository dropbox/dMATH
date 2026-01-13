//! MCP error types

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// MCP protocol error codes as defined in JSON-RPC 2.0
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    /// Parse error - Invalid JSON was received
    ParseError = -32700,
    /// Invalid Request - The JSON sent is not a valid Request object
    InvalidRequest = -32600,
    /// Method not found - The method does not exist / is not available
    MethodNotFound = -32601,
    /// Invalid params - Invalid method parameter(s)
    InvalidParams = -32602,
    /// Internal error - Internal JSON-RPC error
    InternalError = -32603,
    /// Tool not found
    ToolNotFound = -32000,
    /// Tool execution failed
    ToolExecutionError = -32001,
    /// Verification failed
    VerificationError = -32002,
    /// Backend not available
    BackendNotAvailable = -32003,
}

impl ErrorCode {
    /// Get the numeric code value
    pub fn code(self) -> i32 {
        self as i32
    }
}

/// MCP error type
#[derive(Debug, Error)]
pub enum McpError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Method not found: {0}")]
    MethodNotFound(String),

    #[error("Invalid params: {0}")]
    InvalidParams(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Tool execution error: {0}")]
    ToolExecutionError(String),

    #[error("Verification error: {0}")]
    VerificationError(String),

    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl McpError {
    /// Get the error code for this error type
    pub fn code(&self) -> ErrorCode {
        match self {
            McpError::ParseError(_) => ErrorCode::ParseError,
            McpError::InvalidRequest(_) => ErrorCode::InvalidRequest,
            McpError::MethodNotFound(_) => ErrorCode::MethodNotFound,
            McpError::InvalidParams(_) => ErrorCode::InvalidParams,
            McpError::InternalError(_) => ErrorCode::InternalError,
            McpError::ToolNotFound(_) => ErrorCode::ToolNotFound,
            McpError::ToolExecutionError(_) => ErrorCode::ToolExecutionError,
            McpError::VerificationError(_) => ErrorCode::VerificationError,
            McpError::BackendNotAvailable(_) => ErrorCode::BackendNotAvailable,
            McpError::Io(_) => ErrorCode::InternalError,
            McpError::Json(_) => ErrorCode::ParseError,
        }
    }

    /// Convert to JSON-RPC error object
    pub fn to_json_error(&self) -> JsonRpcError {
        JsonRpcError {
            code: self.code().code(),
            message: self.to_string(),
            data: None,
        }
    }
}

/// JSON-RPC 2.0 error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcError {
    /// Create a new error
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code: code.code(),
            message: message.into(),
            data: None,
        }
    }

    /// Add error data
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }
}
