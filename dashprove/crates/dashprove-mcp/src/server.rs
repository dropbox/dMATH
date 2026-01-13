//! MCP server implementation

use std::sync::Arc;

use tracing::{debug, error, info};

use crate::cache::SharedVerificationCache;
use crate::error::{ErrorCode, JsonRpcError, McpError};
use crate::protocol::{
    InitializeParams, InitializeResult, JsonRpcRequest, JsonRpcResponse, RequestId, ToolCallParams,
    ToolCallResult, ToolsListParams, ToolsListResult,
};
use crate::streaming::SessionManager;
use crate::tools::{ToolRegistry, VerifyUslResult};

/// MCP server that handles JSON-RPC requests
pub struct McpServer {
    /// Tool registry
    tools: ToolRegistry,
    /// Whether the server has been initialized
    initialized: bool,
}

impl McpServer {
    /// Create a new MCP server
    pub fn new() -> Self {
        Self {
            tools: ToolRegistry::new(),
            initialized: false,
        }
    }

    /// Create a new MCP server with a custom cache
    ///
    /// This allows sharing a cache instance that can be persisted
    /// to disk on server shutdown and loaded on startup.
    pub fn with_cache(cache: SharedVerificationCache<VerifyUslResult>) -> Self {
        Self {
            tools: ToolRegistry::with_shared_cache(cache),
            initialized: false,
        }
    }

    /// Get the verification cache
    ///
    /// Used for cache persistence (save/load) operations.
    pub fn verification_cache(&self) -> SharedVerificationCache<VerifyUslResult> {
        self.tools.verification_cache()
    }

    /// Handle a JSON-RPC request and return a response
    pub async fn handle_request(&mut self, request: JsonRpcRequest) -> JsonRpcResponse {
        debug!(
            "Handling request: {} (id: {:?})",
            request.method, request.id
        );

        let result = match request.method.as_str() {
            "initialize" => {
                self.handle_initialize(request.id.clone(), request.params)
                    .await
            }
            "initialized" => {
                // Notification acknowledgment - return empty result
                Ok(JsonRpcResponse::success(
                    request.id.clone(),
                    serde_json::json!({}),
                ))
            }
            "tools/list" => {
                self.handle_tools_list(request.id.clone(), request.params)
                    .await
            }
            "tools/call" => {
                self.handle_tools_call(request.id.clone(), request.params)
                    .await
            }
            "ping" => Ok(JsonRpcResponse::success(
                request.id.clone(),
                serde_json::json!({}),
            )),
            method => {
                error!("Unknown method: {}", method);
                Err(McpError::MethodNotFound(method.to_string()))
            }
        };

        match result {
            Ok(response) => response,
            Err(e) => JsonRpcResponse::error(request.id, e.to_json_error()),
        }
    }

    /// Handle initialize request
    async fn handle_initialize(
        &mut self,
        id: RequestId,
        params: Option<serde_json::Value>,
    ) -> Result<JsonRpcResponse, McpError> {
        let _params: InitializeParams = params
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| McpError::InvalidParams(e.to_string()))?
            .unwrap_or_else(|| InitializeParams {
                protocol_version: crate::protocol::MCP_VERSION.to_string(),
                capabilities: Default::default(),
                client_info: crate::protocol::ClientInfo {
                    name: "unknown".to_string(),
                    version: "0.0.0".to_string(),
                },
            });

        info!(
            "Initializing MCP server for client: {} v{}",
            _params.client_info.name, _params.client_info.version
        );

        self.initialized = true;

        let result = InitializeResult::default();
        let value =
            serde_json::to_value(result).map_err(|e| McpError::InternalError(e.to_string()))?;

        Ok(JsonRpcResponse::success(id, value))
    }

    /// Handle tools/list request
    async fn handle_tools_list(
        &self,
        id: RequestId,
        params: Option<serde_json::Value>,
    ) -> Result<JsonRpcResponse, McpError> {
        if !self.initialized {
            return Err(McpError::InvalidRequest(
                "Server not initialized".to_string(),
            ));
        }

        let _params: ToolsListParams = params
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| McpError::InvalidParams(e.to_string()))?
            .unwrap_or_default();

        let result = ToolsListResult {
            tools: self.tools.definitions(),
            next_cursor: None,
        };

        let value =
            serde_json::to_value(result).map_err(|e| McpError::InternalError(e.to_string()))?;

        Ok(JsonRpcResponse::success(id, value))
    }

    /// Handle tools/call request
    async fn handle_tools_call(
        &self,
        id: RequestId,
        params: Option<serde_json::Value>,
    ) -> Result<JsonRpcResponse, McpError> {
        if !self.initialized {
            return Err(McpError::InvalidRequest(
                "Server not initialized".to_string(),
            ));
        }

        let params: ToolCallParams = serde_json::from_value(
            params.ok_or_else(|| McpError::InvalidParams("Missing params".to_string()))?,
        )
        .map_err(|e| McpError::InvalidParams(e.to_string()))?;

        debug!("Calling tool: {}", params.name);

        let result: ToolCallResult = self.tools.execute(&params.name, params.arguments).await?;

        let value =
            serde_json::to_value(result).map_err(|e| McpError::InternalError(e.to_string()))?;

        Ok(JsonRpcResponse::success(id, value))
    }

    /// Process a single JSON line and return a response
    pub async fn process_line(&mut self, line: &str) -> Result<String, McpError> {
        let request: JsonRpcRequest =
            serde_json::from_str(line).map_err(|e| McpError::ParseError(e.to_string()))?;

        let response = self.handle_request(request).await;

        serde_json::to_string(&response).map_err(|e| McpError::InternalError(e.to_string()))
    }

    /// Parse a request from JSON
    pub fn parse_request(json: &str) -> Result<JsonRpcRequest, McpError> {
        serde_json::from_str(json).map_err(|e| McpError::ParseError(e.to_string()))
    }

    /// Create an error response for parse failures
    pub fn parse_error_response(id: Option<RequestId>, message: &str) -> JsonRpcResponse {
        let id = id.unwrap_or(RequestId::Number(0));
        JsonRpcResponse::error(id, JsonRpcError::new(ErrorCode::ParseError, message))
    }

    /// Get the session manager from the tool registry
    ///
    /// Used by HTTP transport for SSE streaming endpoints
    pub fn tools_session_manager(&self) -> Arc<SessionManager> {
        self.tools.session_manager()
    }

    /// Get access to the tool registry
    ///
    /// Used by HTTP transport for direct tool execution (e.g., batch endpoint)
    pub fn tools_registry(&self) -> &ToolRegistry {
        &self.tools
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}
