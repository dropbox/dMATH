//! MCP protocol types implementing JSON-RPC 2.0

use serde::{Deserialize, Serialize};

use crate::error::JsonRpcError;

/// JSON-RPC 2.0 version string
pub const JSONRPC_VERSION: &str = "2.0";

/// MCP protocol version
pub const MCP_VERSION: &str = "2024-11-05";

/// JSON-RPC request ID - can be string or number
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    /// Numeric ID
    Number(i64),
    /// String ID
    String(String),
}

impl From<i64> for RequestId {
    fn from(n: i64) -> Self {
        RequestId::Number(n)
    }
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        RequestId::String(s)
    }
}

impl From<&str> for RequestId {
    fn from(s: &str) -> Self {
        RequestId::String(s.to_string())
    }
}

/// JSON-RPC 2.0 request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// Protocol version - must be "2.0"
    pub jsonrpc: String,
    /// Request ID
    pub id: RequestId,
    /// Method name
    pub method: String,
    /// Method parameters
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    /// Create a new request
    pub fn new(id: impl Into<RequestId>, method: impl Into<String>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: id.into(),
            method: method.into(),
            params: None,
        }
    }

    /// Add parameters to the request
    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = Some(params);
        self
    }
}

/// JSON-RPC 2.0 response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// Protocol version - must be "2.0"
    pub jsonrpc: String,
    /// Request ID (must match request)
    pub id: RequestId,
    /// Result (mutually exclusive with error)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Error (mutually exclusive with result)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    /// Create a success response
    pub fn success(id: RequestId, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response
    pub fn error(id: RequestId, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// JSON-RPC 2.0 notification (request without ID)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    /// Protocol version - must be "2.0"
    pub jsonrpc: String,
    /// Method name
    pub method: String,
    /// Method parameters
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcNotification {
    /// Create a new notification
    pub fn new(method: impl Into<String>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.into(),
            params: None,
        }
    }

    /// Add parameters to the notification
    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = Some(params);
        self
    }
}

/// An MCP message - either request, response, or notification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpMessage {
    /// Request
    Request(JsonRpcRequest),
    /// Response
    Response(JsonRpcResponse),
    /// Notification
    Notification(JsonRpcNotification),
}

/// MCP server capabilities announced during initialization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Tool-related capabilities
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolCapabilities>,
    /// Resource-related capabilities
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceCapabilities>,
    /// Prompt-related capabilities
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptCapabilities>,
}

/// Tool capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCapabilities {
    /// Supports tool list changes
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Resource capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceCapabilities {
    /// Supports subscriptions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subscribe: Option<bool>,
    /// Supports resource list changes
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Prompt capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptCapabilities {
    /// Supports prompt list changes
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Server information returned during initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
}

impl Default for ServerInfo {
    fn default() -> Self {
        Self {
            name: "dashprove-mcp".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Client information received during initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Client name
    pub name: String,
    /// Client version
    pub version: String,
}

/// Initialize request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    /// Protocol version
    pub protocol_version: String,
    /// Client capabilities
    pub capabilities: ClientCapabilities,
    /// Client information
    pub client_info: ClientInfo,
}

/// Client capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Supports roots
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roots: Option<RootsCapability>,
    /// Supports sampling
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<serde_json::Value>,
}

/// Roots capability
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RootsCapability {
    /// Supports list changes
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Initialize response result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    /// Protocol version
    pub protocol_version: String,
    /// Server capabilities
    pub capabilities: ServerCapabilities,
    /// Server information
    pub server_info: ServerInfo,
}

impl Default for InitializeResult {
    fn default() -> Self {
        Self {
            protocol_version: MCP_VERSION.to_string(),
            capabilities: ServerCapabilities {
                tools: Some(ToolCapabilities {
                    list_changed: Some(false),
                }),
                resources: None,
                prompts: None,
            },
            server_info: ServerInfo::default(),
        }
    }
}

/// Tools list request parameters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolsListParams {
    /// Pagination cursor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
}

/// Tools list response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsListResult {
    /// Available tools
    pub tools: Vec<crate::tools::ToolDefinition>,
    /// Next page cursor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Tool call request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallParams {
    /// Tool name
    pub name: String,
    /// Tool arguments
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Content type in tool results
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Content {
    /// Text content
    #[serde(rename = "text")]
    Text {
        /// Text value
        text: String,
    },
    /// Image content
    #[serde(rename = "image")]
    Image {
        /// Base64-encoded image data
        data: String,
        /// MIME type
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// Resource content
    #[serde(rename = "resource")]
    Resource {
        /// Resource reference
        resource: ResourceReference,
    },
}

impl Content {
    /// Create text content
    pub fn text(text: impl Into<String>) -> Self {
        Content::Text { text: text.into() }
    }
}

/// Resource reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReference {
    /// Resource URI
    pub uri: String,
    /// MIME type
    #[serde(rename = "mimeType", default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Resource text content
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Tool call result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallResult {
    /// Result content
    pub content: Vec<Content>,
    /// Whether this is an error result
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl ToolCallResult {
    /// Create a success result with text content
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![Content::text(text)],
            is_error: None,
        }
    }

    /// Create an error result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![Content::text(message)],
            is_error: Some(true),
        }
    }

    /// Create a result with JSON content
    pub fn json<T: Serialize>(value: &T) -> Result<Self, serde_json::Error> {
        let text = serde_json::to_string_pretty(value)?;
        Ok(Self {
            content: vec![Content::text(text)],
            is_error: None,
        })
    }
}
