//! Tests for dashprove-mcp

use serde_json::json;

use crate::error::{ErrorCode, McpError};
use crate::protocol::{
    Content, InitializeResult, JsonRpcRequest, JsonRpcResponse, RequestId, ToolCallResult,
    ToolsListResult, MCP_VERSION,
};
use crate::server::McpServer;
use crate::tools::{
    CheckDependenciesTool, CompileToTool, GetSuggestionsTool, SelectBackendTool, Tool,
    ToolRegistry, VerifyUslTool,
};

// ============================================================================
// Protocol tests
// ============================================================================

#[test]
fn test_request_id_number() {
    let id = RequestId::from(42i64);
    assert_eq!(id, RequestId::Number(42));
}

#[test]
fn test_request_id_string() {
    let id = RequestId::from("test-id");
    assert_eq!(id, RequestId::String("test-id".to_string()));
}

#[test]
fn test_json_rpc_request_creation() {
    let req = JsonRpcRequest::new(1i64, "test/method");
    assert_eq!(req.jsonrpc, "2.0");
    assert_eq!(req.method, "test/method");
    assert!(req.params.is_none());
}

#[test]
fn test_json_rpc_request_with_params() {
    let req = JsonRpcRequest::new(1i64, "test/method").with_params(json!({"key": "value"}));
    assert!(req.params.is_some());
    assert_eq!(req.params.unwrap()["key"], "value");
}

#[test]
fn test_json_rpc_response_success() {
    let resp = JsonRpcResponse::success(RequestId::from(1i64), json!({"result": "ok"}));
    assert_eq!(resp.jsonrpc, "2.0");
    assert!(resp.result.is_some());
    assert!(resp.error.is_none());
}

#[test]
fn test_json_rpc_response_error() {
    let error = crate::error::JsonRpcError::new(ErrorCode::InvalidParams, "test error");
    let resp = JsonRpcResponse::error(RequestId::from(1i64), error);
    assert!(resp.result.is_none());
    assert!(resp.error.is_some());
    assert_eq!(resp.error.as_ref().unwrap().code, -32602);
}

#[test]
fn test_tool_call_result_text() {
    let result = ToolCallResult::text("Hello, world!");
    assert_eq!(result.content.len(), 1);
    assert!(result.is_error.is_none());
    match &result.content[0] {
        Content::Text { text } => assert_eq!(text, "Hello, world!"),
        _ => panic!("Expected text content"),
    }
}

#[test]
fn test_tool_call_result_error() {
    let result = ToolCallResult::error("Something went wrong");
    assert_eq!(result.is_error, Some(true));
}

#[test]
fn test_tool_call_result_json() {
    #[derive(serde::Serialize)]
    struct TestData {
        name: String,
        value: i32,
    }
    let data = TestData {
        name: "test".to_string(),
        value: 42,
    };
    let result = ToolCallResult::json(&data).unwrap();
    assert!(result.is_error.is_none());
}

#[test]
fn test_initialize_result_defaults() {
    let result = InitializeResult::default();
    assert_eq!(result.protocol_version, MCP_VERSION);
    assert!(result.capabilities.tools.is_some());
    assert_eq!(result.server_info.name, "dashprove-mcp");
}

// ============================================================================
// Error tests
// ============================================================================

#[test]
fn test_error_codes() {
    assert_eq!(ErrorCode::ParseError.code(), -32700);
    assert_eq!(ErrorCode::InvalidRequest.code(), -32600);
    assert_eq!(ErrorCode::MethodNotFound.code(), -32601);
    assert_eq!(ErrorCode::InvalidParams.code(), -32602);
    assert_eq!(ErrorCode::InternalError.code(), -32603);
    assert_eq!(ErrorCode::ToolNotFound.code(), -32000);
}

#[test]
fn test_mcp_error_to_json_error() {
    let error = McpError::InvalidParams("test".to_string());
    let json_error = error.to_json_error();
    assert_eq!(json_error.code, -32602);
    assert!(json_error.message.contains("test"));
}

// ============================================================================
// Tool definition tests
// ============================================================================

#[test]
fn test_verify_usl_tool_definition() {
    let tool = VerifyUslTool::new();
    let def = tool.definition();
    assert_eq!(def.name, "dashprove.verify_usl");
    assert!(def.description.contains("USL"));
    assert!(def.input_schema.properties.is_some());
    assert!(def.input_schema.required.contains(&"spec".to_string()));
}

#[test]
fn test_select_backend_tool_definition() {
    let tool = SelectBackendTool::new();
    let def = tool.definition();
    assert_eq!(def.name, "dashprove.select_backend");
    assert!(def.description.contains("backend"));
    assert!(def
        .input_schema
        .required
        .contains(&"property_type".to_string()));
}

#[test]
fn test_compile_to_tool_definition() {
    let tool = CompileToTool::new();
    let def = tool.definition();
    assert_eq!(def.name, "dashprove.compile_to");
    assert!(def.description.contains("Compile"));
    assert!(def.input_schema.required.contains(&"spec".to_string()));
    assert!(def.input_schema.required.contains(&"backend".to_string()));
}

#[test]
fn test_get_suggestions_tool_definition() {
    let tool = GetSuggestionsTool::new();
    let def = tool.definition();
    assert_eq!(def.name, "dashprove.get_suggestions");
    assert!(def.description.contains("suggestion"));
    assert!(def.input_schema.required.contains(&"spec".to_string()));
    assert!(def.input_schema.required.contains(&"backend".to_string()));
    assert!(def
        .input_schema
        .required
        .contains(&"error_message".to_string()));
}

#[test]
fn test_check_dependencies_tool_definition() {
    let tool = CheckDependenciesTool::new();
    let def = tool.definition();
    assert_eq!(def.name, "dashprove.check_dependencies");
    assert!(def.description.contains("availability"));
    let props = def.input_schema.properties.as_ref().unwrap();
    assert!(props.contains_key("backends"));
}

#[test]
fn test_tool_registry_has_all_tools() {
    let registry = ToolRegistry::new();
    let defs = registry.definitions();
    assert_eq!(defs.len(), 14);

    let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"dashprove.verify_usl"));
    assert!(names.contains(&"dashprove.select_backend"));
    assert!(names.contains(&"dashprove.compile_to"));
    assert!(names.contains(&"dashprove.get_suggestions"));
    assert!(names.contains(&"dashprove.check_dependencies"));
    assert!(names.contains(&"dashprove.verify_usl_streaming"));
    assert!(names.contains(&"get_session_status"));
    assert!(names.contains(&"cancel_session"));
    assert!(names.contains(&"dashprove.batch_verify"));
}

#[test]
fn test_tool_registry_find() {
    let registry = ToolRegistry::new();
    assert!(registry.find("dashprove.verify_usl").is_some());
    assert!(registry.find("dashprove.nonexistent").is_none());
}

// ============================================================================
// Tool execution tests
// ============================================================================

#[tokio::test]
async fn test_verify_usl_valid_spec() {
    let tool = VerifyUslTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }"
    });

    let result = tool.execute(args).await.unwrap();
    assert!(result.is_error.is_none() || result.is_error == Some(false));

    // Parse the result content
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["success"].as_bool().unwrap());
}

#[tokio::test]
async fn test_verify_usl_invalid_spec() {
    let tool = VerifyUslTool::new();
    let args = json!({
        "spec": "this is not valid USL syntax {"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(!parsed["success"].as_bool().unwrap());
    assert!(parsed["parse_errors"].is_array());
}

#[tokio::test]
async fn test_select_backend_safety() {
    let tool = SelectBackendTool::new();
    let args = json!({
        "property_type": "safety"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["recommendations"].is_array());
    let recs = parsed["recommendations"].as_array().unwrap();
    assert!(!recs.is_empty());
    // TLA+ should be recommended for safety
    let backends: Vec<_> = recs
        .iter()
        .map(|r| r["backend"].as_str().unwrap())
        .collect();
    assert!(backends.contains(&"tlaplus"));
}

#[tokio::test]
async fn test_select_backend_memory() {
    let tool = SelectBackendTool::new();
    let args = json!({
        "property_type": "memory"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let recs = parsed["recommendations"].as_array().unwrap();
    let backends: Vec<_> = recs
        .iter()
        .map(|r| r["backend"].as_str().unwrap())
        .collect();
    // Kani should be recommended for memory
    assert!(backends.contains(&"kani"));
}

#[tokio::test]
async fn test_select_backend_with_max() {
    let tool = SelectBackendTool::new();
    let args = json!({
        "property_type": "safety",
        "max_backends": 1
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let recs = parsed["recommendations"].as_array().unwrap();
    assert_eq!(recs.len(), 1);
}

#[tokio::test]
async fn test_compile_to_lean4() {
    let tool = CompileToTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }",
        "backend": "lean4"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["success"].as_bool().unwrap());
    assert_eq!(parsed["target_language"].as_str().unwrap(), "Lean 4");
    assert!(parsed["output"].is_string());
}

#[tokio::test]
async fn test_compile_to_tlaplus() {
    let tool = CompileToTool::new();
    let args = json!({
        "spec": "invariant inv { forall x: Int . x >= 0 }",
        "backend": "tlaplus"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["success"].as_bool().unwrap());
    assert_eq!(parsed["target_language"].as_str().unwrap(), "TLA+");
}

#[tokio::test]
async fn test_compile_to_unknown_backend() {
    let tool = CompileToTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }",
        "backend": "unknown_backend"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(!parsed["success"].as_bool().unwrap());
    assert!(parsed["errors"].is_array());
}

#[tokio::test]
async fn test_get_suggestions_timeout() {
    let tool = GetSuggestionsTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }",
        "backend": "lean4",
        "error_message": "Verification timeout after 60 seconds"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let suggestions = parsed["suggestions"].as_array().unwrap();
    assert!(!suggestions.is_empty());
    // Should suggest complexity reduction
    let types: Vec<_> = suggestions
        .iter()
        .map(|s| s["suggestion_type"].as_str().unwrap())
        .collect();
    assert!(types.contains(&"complexity"));
}

#[tokio::test]
async fn test_get_suggestions_counterexample() {
    let tool = GetSuggestionsTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Int . x > 0 }",
        "backend": "kani",
        "error_message": "Counterexample found: x = -5"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let suggestions = parsed["suggestions"].as_array().unwrap();
    let types: Vec<_> = suggestions
        .iter()
        .map(|s| s["suggestion_type"].as_str().unwrap())
        .collect();
    assert!(types.contains(&"strengthen_precondition"));
}

#[tokio::test]
async fn test_tool_registry_execute() {
    let registry = ToolRegistry::new();
    let args = json!({
        "property_type": "termination"
    });

    let result = registry
        .execute("dashprove.select_backend", args)
        .await
        .unwrap();
    assert!(result.is_error.is_none() || result.is_error == Some(false));
}

#[tokio::test]
async fn test_tool_registry_execute_unknown() {
    let registry = ToolRegistry::new();
    let result = registry.execute("nonexistent.tool", json!({})).await;
    assert!(matches!(result, Err(McpError::ToolNotFound(_))));
}

#[tokio::test]
async fn test_check_dependencies_executes() {
    let tool = CheckDependenciesTool::new();
    let args = json!({});

    let result = tool.execute(args).await.unwrap();
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["backends"].is_array());
    assert!(parsed["summary"].as_str().is_some());
}

// ============================================================================
// Server tests
// ============================================================================

#[tokio::test]
async fn test_server_initialize() {
    let mut server = McpServer::new();
    let request = JsonRpcRequest::new(1i64, "initialize").with_params(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {
            "name": "test-client",
            "version": "1.0.0"
        }
    }));

    let response = server.handle_request(request).await;
    assert!(response.error.is_none());
    assert!(response.result.is_some());

    let result = response.result.unwrap();
    assert!(result["protocolVersion"].is_string());
    assert!(result["capabilities"].is_object());
    assert!(result["serverInfo"].is_object());
}

#[tokio::test]
async fn test_server_tools_list_before_init() {
    let mut server = McpServer::new();
    let request = JsonRpcRequest::new(1i64, "tools/list");

    let response = server.handle_request(request).await;
    // Should fail because not initialized
    assert!(response.error.is_some());
}

#[tokio::test]
async fn test_server_tools_list_after_init() {
    let mut server = McpServer::new();

    // Initialize first
    let init_request = JsonRpcRequest::new(1i64, "initialize").with_params(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"}
    }));
    server.handle_request(init_request).await;

    // Now list tools
    let request = JsonRpcRequest::new(2i64, "tools/list");
    let response = server.handle_request(request).await;

    assert!(response.error.is_none());
    let result: ToolsListResult = serde_json::from_value(response.result.unwrap()).unwrap();
    assert_eq!(result.tools.len(), 14);

    let tool_names: Vec<_> = result.tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tool_names.contains(&"dashprove.check_dependencies"));
    assert!(tool_names.contains(&"dashprove.verify_usl_streaming"));
    assert!(tool_names.contains(&"get_session_status"));
    assert!(tool_names.contains(&"cancel_session"));
    assert!(tool_names.contains(&"dashprove.batch_verify"));
    assert!(tool_names.contains(&"save_cache"));
    assert!(tool_names.contains(&"load_cache"));
}

#[tokio::test]
async fn test_server_tools_call() {
    let mut server = McpServer::new();

    // Initialize
    let init_request = JsonRpcRequest::new(1i64, "initialize").with_params(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"}
    }));
    server.handle_request(init_request).await;

    // Call a tool
    let request = JsonRpcRequest::new(2i64, "tools/call").with_params(json!({
        "name": "dashprove.select_backend",
        "arguments": {
            "property_type": "safety"
        }
    }));
    let response = server.handle_request(request).await;

    assert!(response.error.is_none());
    let result: ToolCallResult = serde_json::from_value(response.result.unwrap()).unwrap();
    assert!(result.is_error.is_none() || result.is_error == Some(false));
}

#[tokio::test]
async fn test_server_unknown_method() {
    let mut server = McpServer::new();
    let request = JsonRpcRequest::new(1i64, "unknown/method");

    let response = server.handle_request(request).await;
    assert!(response.error.is_some());
    assert_eq!(response.error.as_ref().unwrap().code, -32601); // Method not found
}

#[tokio::test]
async fn test_server_ping() {
    let mut server = McpServer::new();
    let request = JsonRpcRequest::new(1i64, "ping");

    let response = server.handle_request(request).await;
    assert!(response.error.is_none());
}

#[tokio::test]
async fn test_server_process_line() {
    let mut server = McpServer::new();
    let line = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;

    let response = server.process_line(line).await.unwrap();
    let parsed: JsonRpcResponse = serde_json::from_str(&response).unwrap();
    assert!(parsed.error.is_none());
}

#[tokio::test]
async fn test_server_process_line_invalid_json() {
    let mut server = McpServer::new();
    let line = "not valid json";

    let result = server.process_line(line).await;
    assert!(result.is_err());
}

// ============================================================================
// Dispatcher-backed verify_usl tests
// ============================================================================

#[tokio::test]
async fn test_verify_usl_typecheck_only() {
    let tool = VerifyUslTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }",
        "typecheck_only": true
    });

    let result = tool.execute(args).await.unwrap();
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(parsed["success"].as_bool().unwrap());
    // Should only have type_checker result, no backend verification
    let results = parsed["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["backend"].as_str().unwrap(), "type_checker");
    assert!(parsed["summary"].as_str().unwrap().contains("type-checked"));
}

#[tokio::test]
async fn test_verify_usl_with_strategy_single() {
    let tool = VerifyUslTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Bool . x or not x }",
        "strategy": "single",
        "typecheck_only": true // Skip actual backend execution for test
    });

    let result = tool.execute(args).await.unwrap();
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(parsed["success"].as_bool().unwrap());
}

#[tokio::test]
async fn test_verify_usl_with_specific_backends() {
    let tool = VerifyUslTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }",
        "backends": ["lean4"],
        "typecheck_only": true
    });

    let result = tool.execute(args).await.unwrap();
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(parsed["success"].as_bool().unwrap());
}

#[tokio::test]
async fn test_verify_usl_with_timeout() {
    let tool = VerifyUslTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }",
        "timeout": 30,
        "typecheck_only": true
    });

    let result = tool.execute(args).await.unwrap();
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(parsed["success"].as_bool().unwrap());
}

#[tokio::test]
async fn test_verify_usl_skip_health_check() {
    let tool = VerifyUslTool::new();
    // When skip_health_check is true and typecheck_only is false,
    // we'll attempt to register backends even if they're unavailable
    let args = json!({
        "spec": "theorem test { forall x: Bool . x or not x }",
        "skip_health_check": true,
        "backends": ["lean4"],
        "typecheck_only": true // Still use typecheck_only for test stability
    });

    let result = tool.execute(args).await.unwrap();
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    // Should still succeed with typecheck
    assert!(parsed["success"].as_bool().unwrap());
}

#[tokio::test]
async fn test_verify_usl_no_backends_available_fallback() {
    let tool = VerifyUslTool::new();
    // Request only backends that are likely not installed
    // With no skip_health_check, this will fall back to typecheck-only
    let args = json!({
        "spec": "theorem test { forall x: Bool . x or not x }",
        "backends": ["lean4", "coq", "alloy", "kani", "tlaplus"]
        // Not setting skip_health_check - backends must pass health check
        // Not setting typecheck_only - will try backends but fallback to typecheck
    });

    let result = tool.execute(args).await.unwrap();
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    // Should still succeed (fallback to typecheck or backends if available)
    assert!(parsed["success"].as_bool().unwrap());
}

#[tokio::test]
async fn test_verify_usl_strategy_all() {
    let tool = VerifyUslTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Bool . x or not x }",
        "strategy": "all",
        "typecheck_only": true
    });

    let result = tool.execute(args).await.unwrap();
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(parsed["success"].as_bool().unwrap());
}

#[tokio::test]
async fn test_verify_usl_strategy_redundant() {
    let tool = VerifyUslTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Bool . x or not x }",
        "strategy": "redundant",
        "typecheck_only": true
    });

    let result = tool.execute(args).await.unwrap();
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(parsed["success"].as_bool().unwrap());
}

#[tokio::test]
async fn test_verify_usl_definition_has_new_params() {
    let tool = VerifyUslTool::new();
    let def = tool.definition();

    // Check that all parameters are defined in schema
    let props = def.input_schema.properties.as_ref().unwrap();
    assert!(props.contains_key("spec"));
    assert!(props.contains_key("strategy"));
    assert!(props.contains_key("backends"));
    assert!(props.contains_key("timeout"));
    assert!(props.contains_key("skip_health_check"));
    assert!(props.contains_key("typecheck_only"));

    // Check skip_health_check schema
    let skip_hc = &props["skip_health_check"];
    assert_eq!(skip_hc.property_type, "boolean");

    // Check typecheck_only schema
    let tc_only = &props["typecheck_only"];
    assert_eq!(tc_only.property_type, "boolean");
}

#[tokio::test]
async fn test_verify_usl_backends_schema_has_enum() {
    let tool = VerifyUslTool::new();
    let def = tool.definition();

    let props = def.input_schema.properties.as_ref().unwrap();
    let backends_schema = &props["backends"];

    // backends is an array with items that have enum values
    assert_eq!(backends_schema.property_type, "array");
    let items = backends_schema.items.as_ref().unwrap();
    assert!(items.enum_values.is_some());
    let enum_vals = items.enum_values.as_ref().unwrap();
    assert!(enum_vals.contains(&"lean4".to_string()));
    assert!(enum_vals.contains(&"tlaplus".to_string()));
    assert!(enum_vals.contains(&"kani".to_string()));
    assert!(enum_vals.contains(&"coq".to_string()));
    assert!(enum_vals.contains(&"alloy".to_string()));
}

// ============================================================================
// HTTP transport tests
// ============================================================================

#[test]
fn test_http_transport_default_address() {
    use crate::transport::HttpTransport;

    let transport = HttpTransport::default();
    assert_eq!(transport.bind_addr, "127.0.0.1:3001");
}

#[test]
fn test_http_transport_custom_address() {
    use crate::transport::HttpTransport;

    let transport = HttpTransport::new("0.0.0.0:8080");
    assert_eq!(transport.bind_addr, "0.0.0.0:8080");
}

#[test]
fn test_http_error_from_mcp_error() {
    use crate::transport::HttpError;
    use axum::http::StatusCode;

    let mcp_error = McpError::InvalidParams("test error".to_string());
    let http_error: HttpError = mcp_error.into();

    assert_eq!(http_error.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(http_error.message.contains("test error"));
}

#[tokio::test]
async fn test_http_transport_invalid_bind_address() {
    use crate::transport::HttpTransport;

    let transport = HttpTransport::new("not-a-valid-address");
    let server = McpServer::new();
    let result = transport.run_async(server).await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Invalid bind address"));
}

/// Integration test using axum's test utilities
#[tokio::test]
async fn test_http_jsonrpc_endpoint() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    // Build the router manually (same as HttpTransport::run_async)
    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/jsonrpc",
            axum::routing::post(crate::transport::handle_jsonrpc),
        )
        .route(
            "/health",
            axum::routing::get(crate::transport::handle_health),
        )
        .route("/", axum::routing::get(crate::transport::handle_info))
        .with_state(state)
        .layer(cors);

    // Test health endpoint
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");
    assert_eq!(json["service"], "dashprove-mcp");
}

#[tokio::test]
async fn test_http_info_endpoint() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/jsonrpc",
            axum::routing::post(crate::transport::handle_jsonrpc),
        )
        .route(
            "/health",
            axum::routing::get(crate::transport::handle_health),
        )
        .route("/", axum::routing::get(crate::transport::handle_info))
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["name"], "dashprove-mcp");
    assert_eq!(json["protocol"], "MCP");
    assert_eq!(json["transport"], "HTTP");
}

#[tokio::test]
async fn test_http_jsonrpc_ping() {
    use axum::{
        body::Body,
        http::{header, Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/jsonrpc",
            axum::routing::post(crate::transport::handle_jsonrpc),
        )
        .route(
            "/health",
            axum::routing::get(crate::transport::handle_health),
        )
        .route("/", axum::routing::get(crate::transport::handle_info))
        .with_state(state)
        .layer(cors);

    let request_body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/jsonrpc")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: JsonRpcResponse = serde_json::from_slice(&body).unwrap();
    assert!(json.error.is_none());
    assert_eq!(json.id, RequestId::Number(1));
}

#[tokio::test]
async fn test_http_jsonrpc_initialize() {
    use axum::{
        body::Body,
        http::{header, Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/jsonrpc",
            axum::routing::post(crate::transport::handle_jsonrpc),
        )
        .with_state(state)
        .layer(cors);

    let request_body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "http-test",
                "version": "1.0.0"
            }
        }
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/jsonrpc")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: JsonRpcResponse = serde_json::from_slice(&body).unwrap();
    assert!(json.error.is_none());

    let result = json.result.unwrap();
    assert!(result["protocolVersion"].is_string());
    assert!(result["serverInfo"]["name"]
        .as_str()
        .unwrap()
        .contains("dashprove"));
}

#[tokio::test]
async fn test_http_jsonrpc_tools_list() {
    use axum::{
        body::Body,
        http::{header, Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let mut server = McpServer::new();

    // Initialize the server first
    let init_req = JsonRpcRequest::new(1i64, "initialize").with_params(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"}
    }));
    server.handle_request(init_req).await;

    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/jsonrpc",
            axum::routing::post(crate::transport::handle_jsonrpc),
        )
        .with_state(state)
        .layer(cors);

    let request_body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/jsonrpc")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: JsonRpcResponse = serde_json::from_slice(&body).unwrap();
    assert!(json.error.is_none());

    let result: ToolsListResult = serde_json::from_value(json.result.unwrap()).unwrap();
    assert_eq!(result.tools.len(), 14);

    let tool_names: Vec<_> = result.tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tool_names.contains(&"dashprove.verify_usl"));
    assert!(tool_names.contains(&"dashprove.select_backend"));
    assert!(tool_names.contains(&"save_cache"));
    assert!(tool_names.contains(&"load_cache"));
    assert!(tool_names.contains(&"dashprove.check_dependencies"));
    assert!(tool_names.contains(&"dashprove.verify_usl_streaming"));
    assert!(tool_names.contains(&"get_session_status"));
    assert!(tool_names.contains(&"cancel_session"));
    assert!(tool_names.contains(&"dashprove.batch_verify"));
}

#[tokio::test]
async fn test_http_jsonrpc_tools_call() {
    use axum::{
        body::Body,
        http::{header, Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let mut server = McpServer::new();

    // Initialize the server first
    let init_req = JsonRpcRequest::new(1i64, "initialize").with_params(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"}
    }));
    server.handle_request(init_req).await;

    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/jsonrpc",
            axum::routing::post(crate::transport::handle_jsonrpc),
        )
        .with_state(state)
        .layer(cors);

    let request_body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "dashprove.select_backend",
            "arguments": {
                "property_type": "safety"
            }
        }
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/jsonrpc")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: JsonRpcResponse = serde_json::from_slice(&body).unwrap();
    assert!(json.error.is_none());

    let result: ToolCallResult = serde_json::from_value(json.result.unwrap()).unwrap();
    assert!(result.is_error.is_none() || result.is_error == Some(false));
}

#[test]
fn test_stdio_transport_new() {
    use crate::transport::StdioTransport;

    let transport = StdioTransport::new();
    // Just verify it can be constructed
    let _ = transport;
}

// ============================================================================
// Streaming verification tests
// ============================================================================

#[test]
fn test_verify_usl_streaming_tool_definition() {
    use crate::streaming::SessionManager;
    use crate::tools::VerifyUslStreamingTool;
    use std::sync::Arc;

    let session_manager = Arc::new(SessionManager::new());
    let tool = VerifyUslStreamingTool::new(session_manager);
    let def = tool.definition();

    assert_eq!(def.name, "dashprove.verify_usl_streaming");
    assert!(def.title.is_some());
    assert!(def.description.contains("streaming"));

    let props = def.input_schema.properties.as_ref().unwrap();
    assert!(props.contains_key("spec"));
    assert!(props.contains_key("strategy"));
    assert!(props.contains_key("backends"));
    assert!(props.contains_key("timeout"));
    assert!(props.contains_key("skip_health_check"));
}

#[tokio::test]
async fn test_session_manager_create_session() {
    use crate::streaming::SessionManager;

    let manager = SessionManager::new();
    let session = manager.create_session().await;

    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    // Session should be retrievable
    assert!(manager.get_session(&session_id).await.is_some());

    // Non-existent session should return None
    assert!(manager.get_session("nonexistent").await.is_none());
}

#[tokio::test]
async fn test_streaming_verify_tool_returns_session_id() {
    use crate::streaming::SessionManager;
    use crate::tools::{Tool, VerifyUslStreamingTool};
    use std::sync::Arc;

    let session_manager = Arc::new(SessionManager::new());
    let tool = VerifyUslStreamingTool::new(session_manager.clone());

    let result = tool
        .execute(json!({
            "spec": "theorem test: 1 + 1 = 2"
        }))
        .await;

    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.is_error.is_none() || result.is_error == Some(false));

    // Check the result contains a session_id
    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
            assert!(parsed.get("session_id").is_some());
            assert!(parsed.get("events_url").is_some());
            assert!(parsed.get("message").is_some());
        }
        _ => panic!("Expected text content"),
    }
}

#[tokio::test]
async fn test_verification_event_serialization() {
    use crate::streaming::VerificationEvent;

    let started = VerificationEvent::Started {
        session_id: "sess_test".to_string(),
        spec_summary: "1 property defined".to_string(),
        backends: vec!["lean4".to_string()],
        total_properties: 1,
    };

    let json = serde_json::to_string(&started).unwrap();
    assert!(json.contains("started"));
    assert!(json.contains("sess_test"));

    let progress = VerificationEvent::Progress {
        session_id: "sess_test".to_string(),
        completed: 1,
        total: 2,
        backend: "Lean4".to_string(),
        property_index: 0,
        successful_so_far: 1,
        failed_so_far: 0,
    };

    let json = serde_json::to_string(&progress).unwrap();
    assert!(json.contains("progress"));
    assert!(json.contains("completed"));

    let completed = VerificationEvent::Completed {
        session_id: "sess_test".to_string(),
        success: true,
        summary: "All properties verified".to_string(),
        total_duration_ms: 100,
    };

    let json = serde_json::to_string(&completed).unwrap();
    assert!(json.contains("completed"));
    assert!(json.contains("success"));
}

#[tokio::test]
async fn test_session_cleanup() {
    use crate::streaming::SessionManager;

    let manager = SessionManager::new();

    // Create a session
    let session = manager.create_session().await;
    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    // Session should exist
    assert!(manager.get_session(&session_id).await.is_some());

    // Cleanup shouldn't remove new sessions
    manager.cleanup_old_sessions().await;
    assert!(manager.get_session(&session_id).await.is_some());
}

#[test]
fn test_session_id_generation() {
    use crate::streaming::generate_session_id;

    let id1 = generate_session_id();
    // Small delay to ensure different nanosecond timestamp
    std::thread::sleep(std::time::Duration::from_micros(1));
    let id2 = generate_session_id();

    // IDs should be unique (with delay to avoid timing collision)
    assert_ne!(id1, id2);

    // IDs should start with "sess_"
    assert!(id1.starts_with("sess_"));
    assert!(id2.starts_with("sess_"));
}

#[tokio::test]
async fn test_session_broadcast() {
    use crate::streaming::{SessionManager, VerificationEvent};
    use tokio::time::{timeout, Duration};

    let manager = SessionManager::new();
    let session = manager.create_session().await;

    // Subscribe to events
    let mut receiver = {
        let s = session.lock().await;
        s.subscribe()
    };

    // Send an event
    {
        let s = session.lock().await;
        s.send(VerificationEvent::Started {
            session_id: "test".to_string(),
            spec_summary: "test".to_string(),
            backends: vec![],
            total_properties: 0,
        });
    }

    // Receive the event
    let result = timeout(Duration::from_millis(100), receiver.recv()).await;
    assert!(result.is_ok());
    let event = result.unwrap().unwrap();
    match event {
        VerificationEvent::Started { session_id, .. } => {
            assert_eq!(session_id, "test");
        }
        _ => panic!("Expected Started event"),
    }
}

// ============================================================================
// GetSessionStatus tool tests
// ============================================================================

#[test]
fn test_get_session_status_tool_definition() {
    use crate::streaming::SessionManager;
    use crate::tools::GetSessionStatusTool;
    use std::sync::Arc;

    let session_manager = Arc::new(SessionManager::new());
    let tool = GetSessionStatusTool::new(session_manager);
    let def = tool.definition();

    assert_eq!(def.name, "get_session_status");
    assert!(def.title.is_some());
    assert!(def.description.contains("session"));

    let props = def.input_schema.properties.as_ref().unwrap();
    assert!(props.contains_key("session_id"));
    assert!(def
        .input_schema
        .required
        .contains(&"session_id".to_string()));
}

#[tokio::test]
async fn test_get_session_status_nonexistent() {
    use crate::streaming::SessionManager;
    use crate::tools::{GetSessionStatusTool, Tool};
    use std::sync::Arc;

    let session_manager = Arc::new(SessionManager::new());
    let tool = GetSessionStatusTool::new(session_manager);

    let result = tool
        .execute(json!({
            "session_id": "nonexistent_session"
        }))
        .await
        .expect("Should not fail");

    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let status: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(status["exists"], false);
            assert_eq!(status["session_id"], "nonexistent_session");
        }
        _ => panic!("Expected text content"),
    }
}

#[tokio::test]
async fn test_get_session_status_existing() {
    use crate::streaming::SessionManager;
    use crate::tools::{GetSessionStatusTool, Tool};
    use std::sync::Arc;

    let session_manager = Arc::new(SessionManager::new());

    // Create a session
    let session = session_manager.create_session().await;
    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    let tool = GetSessionStatusTool::new(session_manager.clone());

    let result = tool
        .execute(json!({
            "session_id": session_id
        }))
        .await
        .expect("Should not fail");

    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let status: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(status["exists"], true);
            assert_eq!(status["completed"], false);
            assert!(status["final_result"].is_null());
        }
        _ => panic!("Expected text content"),
    }
}

#[tokio::test]
async fn test_get_session_status_completed() {
    use crate::streaming::{SessionManager, StreamingVerifyResult};
    use crate::tools::{GetSessionStatusTool, Tool};
    use std::sync::Arc;

    let session_manager = Arc::new(SessionManager::new());

    // Create a session and mark it as completed
    let session = session_manager.create_session().await;
    let session_id = {
        let mut s = session.lock().await;
        let id = s.id.clone();
        s.completed = true;
        s.final_result = Some(StreamingVerifyResult {
            success: true,
            session_id: id.clone(),
            summary: "Test completed successfully".to_string(),
            duration_ms: 100,
        });
        id
    };

    let tool = GetSessionStatusTool::new(session_manager.clone());

    let result = tool
        .execute(json!({
            "session_id": session_id
        }))
        .await
        .expect("Should not fail");

    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let status: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(status["exists"], true);
            assert_eq!(status["completed"], true);
            assert!(status["final_result"].is_object());
            assert_eq!(status["final_result"]["success"], true);
            assert_eq!(
                status["final_result"]["summary"],
                "Test completed successfully"
            );
        }
        _ => panic!("Expected text content"),
    }
}

#[tokio::test]
async fn test_session_manager_get_status() {
    use crate::streaming::SessionManager;

    let manager = SessionManager::new();

    // Check non-existent session
    let status = manager.get_session_status("nonexistent").await;
    assert!(!status.exists);
    assert_eq!(status.session_id, "nonexistent");

    // Create a session
    let session = manager.create_session().await;
    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    // Check existing session
    let status = manager.get_session_status(&session_id).await;
    assert!(status.exists);
    assert!(!status.completed);
    assert!(status.final_result.is_none());
}

#[tokio::test]
async fn test_session_manager_session_count() {
    use crate::streaming::SessionManager;

    let manager = SessionManager::new();

    // Initially no sessions
    assert_eq!(manager.session_count().await, 0);

    // Create sessions
    let _s1 = manager.create_session().await;
    assert_eq!(manager.session_count().await, 1);

    let _s2 = manager.create_session().await;
    assert_eq!(manager.session_count().await, 2);
}

// ============================================================================
// Isabelle and Dafny backend tests
// ============================================================================

#[tokio::test]
async fn test_compile_to_isabelle() {
    let tool = CompileToTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }",
        "backend": "isabelle"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["success"].as_bool().unwrap());
    assert_eq!(parsed["target_language"].as_str().unwrap(), "Isabelle/HOL");
    assert!(parsed["output"].is_string());
}

#[tokio::test]
async fn test_compile_to_dafny() {
    let tool = CompileToTool::new();
    let args = json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }",
        "backend": "dafny"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["success"].as_bool().unwrap());
    assert_eq!(parsed["target_language"].as_str().unwrap(), "Dafny");
    assert!(parsed["output"].is_string());
}

#[test]
fn test_verify_usl_backends_schema_includes_isabelle_dafny() {
    let tool = VerifyUslTool::new();
    let def = tool.definition();
    let props = def.input_schema.properties.as_ref().unwrap();
    let backends_prop = props.get("backends").unwrap();
    let items = backends_prop.items.as_ref().unwrap();
    let enum_values = items.enum_values.as_ref().unwrap();

    // Verify Isabelle and Dafny are in the enum
    assert!(enum_values.contains(&"isabelle".to_string()));
    assert!(enum_values.contains(&"dafny".to_string()));
}

#[test]
fn test_compile_to_schema_includes_isabelle_dafny() {
    let tool = CompileToTool::new();
    let def = tool.definition();
    let props = def.input_schema.properties.as_ref().unwrap();
    let backend_prop = props.get("backend").unwrap();
    let enum_values = backend_prop.enum_values.as_ref().unwrap();

    // Verify Isabelle and Dafny are in the enum
    assert!(enum_values.contains(&"isabelle".to_string()));
    assert!(enum_values.contains(&"dafny".to_string()));
}

#[test]
fn test_check_dependencies_schema_includes_isabelle_dafny() {
    let tool = CheckDependenciesTool::new();
    let def = tool.definition();
    let props = def.input_schema.properties.as_ref().unwrap();
    let backends_prop = props.get("backends").unwrap();
    let items = backends_prop.items.as_ref().unwrap();
    let enum_values = items.enum_values.as_ref().unwrap();

    // Verify Isabelle and Dafny are in the enum
    assert!(enum_values.contains(&"isabelle".to_string()));
    assert!(enum_values.contains(&"dafny".to_string()));
}

#[tokio::test]
async fn test_select_backend_functional_includes_isabelle_dafny() {
    let tool = SelectBackendTool::new();
    let args = json!({
        "property_type": "functional"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let recs = parsed["recommendations"].as_array().unwrap();
    let backends: Vec<_> = recs
        .iter()
        .map(|r| r["backend"].as_str().unwrap())
        .collect();

    // Functional correctness should recommend both Isabelle and Dafny
    assert!(backends.contains(&"isabelle"));
    assert!(backends.contains(&"dafny"));
}

#[tokio::test]
async fn test_select_backend_safety_includes_dafny() {
    let tool = SelectBackendTool::new();
    let args = json!({
        "property_type": "safety"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let recs = parsed["recommendations"].as_array().unwrap();
    let backends: Vec<_> = recs
        .iter()
        .map(|r| r["backend"].as_str().unwrap())
        .collect();

    // Safety should include Dafny for contract-based verification
    assert!(backends.contains(&"dafny"));
}

#[tokio::test]
async fn test_select_backend_termination_includes_isabelle() {
    let tool = SelectBackendTool::new();
    let args = json!({
        "property_type": "termination"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let recs = parsed["recommendations"].as_array().unwrap();
    let backends: Vec<_> = recs
        .iter()
        .map(|r| r["backend"].as_str().unwrap())
        .collect();

    // Termination should include Isabelle for HOL proofs
    assert!(backends.contains(&"isabelle"));
}

#[tokio::test]
async fn test_select_backend_refinement() {
    let tool = SelectBackendTool::new();
    let args = json!({
        "property_type": "refinement"
    });

    let result = tool.execute(args).await.unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let recs = parsed["recommendations"].as_array().unwrap();
    let backends: Vec<_> = recs
        .iter()
        .map(|r| r["backend"].as_str().unwrap())
        .collect();

    // Refinement should include TLA+, Alloy, Isabelle, Lean4
    assert!(
        backends.contains(&"tlaplus"),
        "Expected tlaplus for refinement"
    );
    assert!(backends.contains(&"alloy"), "Expected alloy for refinement");
    assert!(
        backends.contains(&"isabelle"),
        "Expected isabelle for refinement"
    );
    assert!(backends.contains(&"lean4"), "Expected lean4 for refinement");

    // TLA+ should be first (highest confidence)
    assert_eq!(backends[0], "tlaplus");
}

// ============================================================================
// HTTP Session Status Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_http_session_status_not_found() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/sessions/:session_id",
            axum::routing::get(crate::transport::handle_session_status),
        )
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/sessions/nonexistent_session_123")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "Session not found");
    assert_eq!(json["session_id"], "nonexistent_session_123");
}

#[tokio::test]
async fn test_http_session_status_existing() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();

    // Create a session
    let session = session_manager.create_session().await;
    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/sessions/:session_id",
            axum::routing::get(crate::transport::handle_session_status),
        )
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/sessions/{}", session_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["session_id"], session_id);
    assert_eq!(json["exists"], true);
    assert_eq!(json["completed"], false);
}

#[tokio::test]
async fn test_http_session_status_completed() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();

    // Create a session and mark it completed
    let session = session_manager.create_session().await;
    let session_id = {
        let mut s = session.lock().await;
        s.completed = true;
        s.final_result = Some(crate::streaming::StreamingVerifyResult {
            success: true,
            session_id: s.id.clone(),
            summary: "All verified".to_string(),
            duration_ms: 100,
        });
        s.id.clone()
    };

    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/sessions/:session_id",
            axum::routing::get(crate::transport::handle_session_status),
        )
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/sessions/{}", session_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["session_id"], session_id);
    assert_eq!(json["exists"], true);
    assert_eq!(json["completed"], true);
    assert!(json["final_result"].is_object());
    assert_eq!(json["final_result"]["success"], true);
    assert_eq!(json["final_result"]["summary"], "All verified");
}

// ============================================================================
// Cancel Session Tests
// ============================================================================

#[test]
fn test_cancel_session_tool_definition() {
    use crate::streaming::SessionManager;
    use crate::tools::CancelSessionTool;
    use std::sync::Arc;

    let session_manager = Arc::new(SessionManager::new());
    let tool = CancelSessionTool::new(session_manager);
    let def = tool.definition();

    assert_eq!(def.name, "cancel_session");
    assert!(def.title.is_some());
    assert!(def.description.contains("Cancel"));

    let props = def.input_schema.properties.as_ref().unwrap();
    assert!(props.contains_key("session_id"));
    assert!(def
        .input_schema
        .required
        .contains(&"session_id".to_string()));
}

#[tokio::test]
async fn test_cancel_session_nonexistent() {
    use crate::streaming::SessionManager;
    use crate::tools::{CancelSessionTool, Tool};
    use std::sync::Arc;

    let session_manager = Arc::new(SessionManager::new());
    let tool = CancelSessionTool::new(session_manager);

    let result = tool
        .execute(json!({
            "session_id": "nonexistent_session"
        }))
        .await
        .expect("Should not fail");

    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let cancel_result: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(cancel_result["success"], false);
            assert!(cancel_result["message"]
                .as_str()
                .unwrap()
                .contains("not found"));
            assert_eq!(cancel_result["session_id"], "nonexistent_session");
        }
        _ => panic!("Expected text content"),
    }
}

#[tokio::test]
async fn test_cancel_session_running() {
    use crate::streaming::SessionManager;
    use crate::tools::{CancelSessionTool, Tool};
    use std::sync::Arc;

    let session_manager = Arc::new(SessionManager::new());

    // Create a session (simulating a running verification)
    let session = session_manager.create_session().await;
    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    let tool = CancelSessionTool::new(session_manager.clone());

    let result = tool
        .execute(json!({
            "session_id": session_id
        }))
        .await
        .expect("Should not fail");

    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let cancel_result: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(cancel_result["success"], true);
            assert!(cancel_result["message"]
                .as_str()
                .unwrap()
                .contains("cancelled"));
            assert_eq!(cancel_result["was_completed"], false);
        }
        _ => panic!("Expected text content"),
    }

    // Verify the session is now marked as cancelled and completed
    let status = session_manager.get_session_status(&session_id).await;
    assert!(status.completed);
    assert!(status.cancelled);
    assert!(status.final_result.is_some());
    assert!(!status.final_result.as_ref().unwrap().success);
}

#[tokio::test]
async fn test_cancel_session_already_completed() {
    use crate::streaming::{SessionManager, StreamingVerifyResult};
    use crate::tools::{CancelSessionTool, Tool};
    use std::sync::Arc;

    let session_manager = Arc::new(SessionManager::new());

    // Create a session and mark it as completed
    let session = session_manager.create_session().await;
    let session_id = {
        let mut s = session.lock().await;
        let id = s.id.clone();
        s.completed = true;
        s.final_result = Some(StreamingVerifyResult {
            success: true,
            session_id: id.clone(),
            summary: "Verification succeeded".to_string(),
            duration_ms: 100,
        });
        id
    };

    let tool = CancelSessionTool::new(session_manager.clone());

    let result = tool
        .execute(json!({
            "session_id": session_id
        }))
        .await
        .expect("Should not fail");

    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let cancel_result: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(cancel_result["success"], false);
            assert!(cancel_result["message"]
                .as_str()
                .unwrap()
                .contains("already completed"));
            assert_eq!(cancel_result["was_completed"], true);
        }
        _ => panic!("Expected text content"),
    }
}

#[tokio::test]
async fn test_cancel_session_already_cancelled() {
    use crate::streaming::SessionManager;
    use crate::tools::{CancelSessionTool, Tool};
    use std::sync::Arc;

    let session_manager = Arc::new(SessionManager::new());

    // Create a session
    let session = session_manager.create_session().await;
    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    let tool = CancelSessionTool::new(session_manager.clone());

    // First cancel should succeed
    let result = tool
        .execute(json!({
            "session_id": session_id.clone()
        }))
        .await
        .expect("Should not fail");

    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let cancel_result: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(cancel_result["success"], true);
        }
        _ => panic!("Expected text content"),
    }

    // Second cancel should fail (already completed due to cancellation)
    let result2 = tool
        .execute(json!({
            "session_id": session_id
        }))
        .await
        .expect("Should not fail");

    let content2 = &result2.content[0];
    match content2 {
        Content::Text { text } => {
            let cancel_result: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(cancel_result["success"], false);
            // It will be "already completed" because cancellation marks it complete
            assert!(cancel_result["message"]
                .as_str()
                .unwrap()
                .contains("completed"));
        }
        _ => panic!("Expected text content"),
    }
}

#[tokio::test]
async fn test_session_manager_cancel_session() {
    use crate::streaming::SessionManager;

    let manager = SessionManager::new();

    // Try to cancel non-existent session
    let result = manager.cancel_session("nonexistent").await;
    assert!(!result.success);
    assert!(result.message.contains("not found"));

    // Create a session and cancel it
    let session = manager.create_session().await;
    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    let result = manager.cancel_session(&session_id).await;
    assert!(result.success);
    assert!(!result.was_completed);

    // Check session state after cancellation
    let status = manager.get_session_status(&session_id).await;
    assert!(status.exists);
    assert!(status.completed);
    assert!(status.cancelled);
}

#[tokio::test]
async fn test_verification_event_cancelled_serialization() {
    use crate::streaming::VerificationEvent;

    let event = VerificationEvent::Cancelled {
        session_id: "test_session".to_string(),
        message: "Cancelled by user".to_string(),
        elapsed_ms: 5000,
    };

    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "cancelled");
    assert_eq!(json["session_id"], "test_session");
    assert_eq!(json["message"], "Cancelled by user");
    assert_eq!(json["elapsed_ms"], 5000);
}

// ============================================================================
// HTTP Cancel Session Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_http_cancel_session_not_found() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/sessions/:session_id",
            axum::routing::delete(crate::transport::handle_cancel_session),
        )
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::DELETE)
                .uri("/sessions/nonexistent_session_123")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["success"], false);
    assert!(json["message"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn test_http_cancel_session_success() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();

    // Create a session
    let session = session_manager.create_session().await;
    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/sessions/:session_id",
            axum::routing::delete(crate::transport::handle_cancel_session),
        )
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::DELETE)
                .uri(format!("/sessions/{}", session_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["success"], true);
    assert!(json["message"]
        .as_str()
        .unwrap()
        .contains("cancelled successfully"));
}

#[tokio::test]
async fn test_http_cancel_session_post_endpoint() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();

    // Create a session
    let session = session_manager.create_session().await;
    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/sessions/:session_id/cancel",
            axum::routing::post(crate::transport::handle_cancel_session),
        )
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(format!("/sessions/{}/cancel", session_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["success"], true);
}

#[tokio::test]
async fn test_http_cancel_session_already_completed() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();

    // Create a session and mark it completed
    let session = session_manager.create_session().await;
    let session_id = {
        let mut s = session.lock().await;
        s.completed = true;
        s.final_result = Some(crate::streaming::StreamingVerifyResult {
            success: true,
            session_id: s.id.clone(),
            summary: "Done".to_string(),
            duration_ms: 100,
        });
        s.id.clone()
    };

    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/sessions/:session_id",
            axum::routing::delete(crate::transport::handle_cancel_session),
        )
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::DELETE)
                .uri(format!("/sessions/{}", session_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return CONFLICT for already completed sessions
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["success"], false);
    assert!(json["message"]
        .as_str()
        .unwrap()
        .contains("already completed"));
}

// ============================================================================
// Cooperative cancellation tests
// ============================================================================

#[tokio::test]
async fn test_cooperative_cancellation_before_verification_starts() {
    use crate::streaming::{
        run_streaming_verification, SessionManager, StreamingVerifyArgs, VerificationEvent,
    };

    let manager = SessionManager::new();
    let session = manager.create_session().await;
    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    // Subscribe before cancelling
    let mut receiver = {
        let s = session.lock().await;
        s.subscribe()
    };

    // Cancel immediately before starting verification
    let cancel_result = manager.cancel_session(&session_id).await;
    assert!(cancel_result.success);

    // Start verification (should exit early due to cancellation)
    let args = StreamingVerifyArgs {
        spec: "theorem test { forall x: Int . x + 0 == x }".to_string(),
        strategy: "single".to_string(),
        backends: vec!["lean4".to_string()],
        timeout: 60,
        skip_health_check: true,
    };

    run_streaming_verification(session.clone(), args).await;

    // Verify session is marked completed
    let s = session.lock().await;
    assert!(s.completed);
    assert!(s.cancelled);

    // Verify we got a Cancelled event
    let mut found_cancelled = false;
    while let Ok(event) = receiver.try_recv() {
        if matches!(event, VerificationEvent::Cancelled { .. }) {
            found_cancelled = true;
            break;
        }
    }
    assert!(found_cancelled, "Should have received Cancelled event");
}

#[tokio::test]
async fn test_cooperative_cancellation_returns_cancelled_result() {
    use crate::streaming::{run_streaming_verification, SessionManager, StreamingVerifyArgs};

    let manager = SessionManager::new();
    let session = manager.create_session().await;
    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    // Cancel before starting
    manager.cancel_session(&session_id).await;

    let args = StreamingVerifyArgs {
        spec: "theorem test { forall x: Int . x + 0 == x }".to_string(),
        strategy: "single".to_string(),
        backends: vec![],
        timeout: 60,
        skip_health_check: true,
    };

    run_streaming_verification(session.clone(), args).await;

    // Check final result indicates cancellation
    let s = session.lock().await;
    assert!(s.final_result.is_some());
    let result = s.final_result.as_ref().unwrap();
    assert!(!result.success);
    assert!(
        result.summary.contains("cancelled"),
        "Summary should mention cancellation: {}",
        result.summary
    );
}

// ============================================================================
// Batch verification tests
// ============================================================================

#[test]
fn test_batch_verify_tool_definition() {
    use crate::tools::BatchVerifyTool;

    let tool = BatchVerifyTool::new();
    let def = tool.definition();

    assert_eq!(def.name, "dashprove.batch_verify");
    assert!(def.description.contains("batch"));

    // Check required fields
    assert!(def.input_schema.required.contains(&"specs".to_string()));

    // Check properties exist
    let props = def.input_schema.properties.as_ref().unwrap();
    assert!(props.contains_key("specs"));
    assert!(props.contains_key("timeout"));
    assert!(props.contains_key("strategy"));
    assert!(props.contains_key("skip_health_check"));
    assert!(props.contains_key("fail_fast"));
}

#[tokio::test]
async fn test_batch_verify_empty_specs() {
    use crate::tools::BatchVerifyTool;

    let tool = BatchVerifyTool::new();
    let result = tool
        .execute(serde_json::json!({
            "specs": []
        }))
        .await
        .unwrap();

    // Empty specs should succeed with 0 successful, 0 failed
    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(parsed["total"], 0);
            assert_eq!(parsed["successful"], 0);
            assert_eq!(parsed["failed"], 0);
            // Empty batch is not considered successful (no specs processed)
            assert_eq!(parsed["success"], false);
        }
        _ => panic!("Expected text content"),
    }
}

#[tokio::test]
async fn test_batch_verify_single_valid_spec() {
    use crate::tools::BatchVerifyTool;

    let tool = BatchVerifyTool::new();
    // Use empty backends to force typecheck-only mode
    let result = tool
        .execute(serde_json::json!({
            "specs": [
                {"spec": "theorem test { forall x: Int . x + 0 == x }", "backends": []}
            ]
        }))
        .await
        .unwrap();

    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(parsed["total"], 1);
            // With empty backends specified, but backend availability varies by system
            // So we accept either typecheck_only (no backends available) or verified (backends ran)
            assert_eq!(parsed["successful"], 1, "Full response: {}", text);
            assert_eq!(parsed["failed"], 0);
            let results = parsed["results"].as_array().unwrap();
            let status = results[0]["status"].as_str().unwrap();
            assert!(
                status == "typecheck_only" || status == "verified",
                "Expected typecheck_only or verified, got: {}",
                status
            );
        }
        _ => panic!("Expected text content"),
    }
}

#[tokio::test]
async fn test_batch_verify_with_parse_error() {
    use crate::tools::BatchVerifyTool;

    let tool = BatchVerifyTool::new();
    // Use empty backends to force typecheck-only mode for the valid spec
    let result = tool
        .execute(serde_json::json!({
            "specs": [
                {"id": "valid_spec", "spec": "theorem test { forall x: Int . x + 0 == x }", "backends": []},
                {"id": "invalid_spec", "spec": "this is not valid USL syntax!!!"}
            ]
        }))
        .await
        .unwrap();

    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(parsed["total"], 2);
            assert_eq!(parsed["successful"], 1);
            assert_eq!(parsed["failed"], 1);
            // Overall success should be false due to parse error
            assert_eq!(parsed["success"], false);

            // Check individual results
            let results = parsed["results"].as_array().unwrap();
            assert_eq!(results.len(), 2);

            // First should be successful (typecheck_only or verified)
            assert_eq!(results[0]["id"], "valid_spec");
            assert_eq!(results[0]["success"], true);
            let status = results[0]["status"].as_str().unwrap();
            assert!(
                status == "typecheck_only" || status == "verified",
                "Expected typecheck_only or verified, got: {}",
                status
            );

            // Second should have parse error
            assert_eq!(results[1]["id"], "invalid_spec");
            assert_eq!(results[1]["success"], false);
            assert_eq!(results[1]["status"], "parse_error");
            assert!(results[1]["error"]
                .as_str()
                .unwrap()
                .contains("Parse error"));
        }
        _ => panic!("Expected text content"),
    }
}

#[tokio::test]
async fn test_batch_verify_fail_fast() {
    use crate::tools::BatchVerifyTool;

    let tool = BatchVerifyTool::new();
    let result = tool
        .execute(serde_json::json!({
            "specs": [
                {"id": "first_invalid", "spec": "invalid spec!!!"},
                {"id": "second_valid", "spec": "theorem test { forall x: Int . x + 0 == x }", "backends": []},
                {"id": "third_valid", "spec": "theorem test2 { forall y: Int . y > 0 }", "backends": []}
            ],
            "fail_fast": true
        }))
        .await
        .unwrap();

    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
            // With fail_fast, we should only see 1 result (the first failure)
            let results = parsed["results"].as_array().unwrap();
            assert_eq!(
                results.len(),
                1,
                "fail_fast should stop after first failure"
            );
            assert_eq!(results[0]["id"], "first_invalid");
            assert_eq!(results[0]["success"], false);
        }
        _ => panic!("Expected text content"),
    }
}

#[tokio::test]
async fn test_batch_verify_custom_ids() {
    use crate::tools::BatchVerifyTool;

    let tool = BatchVerifyTool::new();
    let result = tool
        .execute(serde_json::json!({
            "specs": [
                {"id": "my_custom_id", "spec": "theorem test { forall x: Int . x + 0 == x }", "backends": []},
                {"spec": "theorem test2 { forall y: Int . y > 0 }", "backends": []}  // No ID
            ]
        }))
        .await
        .unwrap();

    let content = &result.content[0];
    match content {
        Content::Text { text } => {
            let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
            let results = parsed["results"].as_array().unwrap();
            assert_eq!(results.len(), 2);

            // First should have custom ID
            assert_eq!(results[0]["id"], "my_custom_id");

            // Second should have auto-generated ID
            assert_eq!(results[1]["id"], "spec_2");
        }
        _ => panic!("Expected text content"),
    }
}

// ============================================================================
// HTTP batch endpoint tests
// ============================================================================

#[tokio::test]
async fn test_http_batch_empty_specs() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/batch",
            axum::routing::post(crate::transport::handle_batch_verify),
        )
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/batch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"specs": []}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Empty batch returns success=false (nothing to verify counts as not successful)
    // This matches the tool's logic: success = failed == 0 && successful > 0
    assert_eq!(json["success"], false);
    assert_eq!(json["total"], 0);
    assert_eq!(json["successful"], 0);
    assert_eq!(json["failed"], 0);
}

#[tokio::test]
async fn test_http_batch_single_spec() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/batch",
            axum::routing::post(crate::transport::handle_batch_verify),
        )
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/batch")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"specs": [{"spec": "theorem test { forall x: Int . x + 0 == x }"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Single spec batch
    assert_eq!(json["total"], 1);
    assert!(json["results"].as_array().is_some());
    assert_eq!(json["results"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn test_http_batch_multiple_specs() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/batch",
            axum::routing::post(crate::transport::handle_batch_verify),
        )
        .with_state(state)
        .layer(cors);

    let body_json = serde_json::json!({
        "specs": [
            {"id": "spec_a", "spec": "theorem test1 { forall x: Int . x + 0 == x }"},
            {"id": "spec_b", "spec": "theorem test2 { forall y: Int . y > 0 }"}
        ],
        "strategy": "auto"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["total"], 2);
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["id"], "spec_a");
    assert_eq!(results[1]["id"], "spec_b");
}

#[tokio::test]
async fn test_http_batch_with_parse_error() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/batch",
            axum::routing::post(crate::transport::handle_batch_verify),
        )
        .with_state(state)
        .layer(cors);

    let body_json = serde_json::json!({
        "specs": [
            {"id": "valid", "spec": "theorem test { forall x: Int . x + 0 == x }"},
            {"id": "invalid", "spec": "this is not valid USL syntax @#$%"}
        ]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["total"], 2);
    assert!(json["failed"].as_u64().unwrap() >= 1); // At least the invalid one failed
}

#[tokio::test]
async fn test_http_batch_invalid_request_body() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/batch",
            axum::routing::post(crate::transport::handle_batch_verify),
        )
        .with_state(state)
        .layer(cors);

    // Send invalid JSON
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/batch")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return 422 Unprocessable Entity (axum's default for JSON parse errors)
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_http_batch_with_options() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/batch",
            axum::routing::post(crate::transport::handle_batch_verify),
        )
        .with_state(state)
        .layer(cors);

    let body_json = serde_json::json!({
        "specs": [
            {"spec": "theorem test { forall x: Int . x + 0 == x }"}
        ],
        "timeout": 30,
        "strategy": "single",
        "skip_health_check": true,
        "fail_fast": false
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should succeed even with custom options
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["total"], 1);
    assert!(json["duration_ms"].is_number());
}

// ============================================================================
// HTTP /verify Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_http_verify_valid_spec() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/verify",
            axum::routing::post(crate::transport::handle_verify),
        )
        .with_state(state)
        .layer(cors);

    let body_json = serde_json::json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/verify")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Should have standard VerifyUslResult fields
    assert!(json["success"].is_boolean());
    assert!(json["summary"].is_string());
    assert!(json["results"].is_array());
}

#[tokio::test]
async fn test_http_verify_with_backends() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/verify",
            axum::routing::post(crate::transport::handle_verify),
        )
        .with_state(state)
        .layer(cors);

    let body_json = serde_json::json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }",
        "backends": ["lean4"],
        "strategy": "single"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/verify")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json["success"].is_boolean());
    assert!(json["results"].is_array());
}

#[tokio::test]
async fn test_http_verify_with_parse_error() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/verify",
            axum::routing::post(crate::transport::handle_verify),
        )
        .with_state(state)
        .layer(cors);

    let body_json = serde_json::json!({
        "spec": "this is not valid USL syntax @#$%"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/verify")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should still return 200 OK with error info in body
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Should report failure with parse errors
    assert_eq!(json["success"], false);
    assert!(json["parse_errors"].is_array());
    assert!(!json["parse_errors"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_http_verify_invalid_request_body() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/verify",
            axum::routing::post(crate::transport::handle_verify),
        )
        .with_state(state)
        .layer(cors);

    // Send invalid JSON
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/verify")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return 400 or 422 for invalid JSON
    let status = response.status();
    assert!(status == StatusCode::BAD_REQUEST || status == StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_http_verify_with_options() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/verify",
            axum::routing::post(crate::transport::handle_verify),
        )
        .with_state(state)
        .layer(cors);

    let body_json = serde_json::json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }",
        "timeout": 30,
        "strategy": "single",
        "skip_health_check": true,
        "typecheck_only": false
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/verify")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should succeed with custom options
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json["success"].is_boolean());
    assert!(json["results"].is_array());
}

#[tokio::test]
async fn test_http_verify_typecheck_only() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/verify",
            axum::routing::post(crate::transport::handle_verify),
        )
        .with_state(state)
        .layer(cors);

    let body_json = serde_json::json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }",
        "typecheck_only": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/verify")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // typecheck_only should succeed for valid spec
    assert_eq!(json["success"], true);
    // typecheck_only returns a type_checker pseudo-backend result
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["backend"], "type_checker");
    assert_eq!(results[0]["status"], "verified");
}

// ============================================================================
// HTTP /verify/stream Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_http_verify_stream_start() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager: session_manager.clone(),
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/verify/stream",
            axum::routing::post(crate::transport::handle_verify_streaming),
        )
        .with_state(state)
        .layer(cors);

    let body_json = serde_json::json!({
        "spec": "theorem test { forall x: Int . x + 0 == x }",
        "strategy": "single",
        "skip_health_check": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/verify/stream")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body_json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    let session_id = json["session_id"].as_str().unwrap();
    assert!(!session_id.is_empty());
    assert!(json["events_url"].as_str().unwrap().contains(session_id));

    // Session should be registered with the manager
    let session = session_manager.get_session(session_id).await;
    assert!(session.is_some());
}

#[tokio::test]
async fn test_http_verify_stream_invalid_request_body() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/verify/stream",
            axum::routing::post(crate::transport::handle_verify_streaming),
        )
        .with_state(state)
        .layer(cors);

    // Send invalid JSON
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/verify/stream")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    assert!(status == StatusCode::BAD_REQUEST || status == StatusCode::UNPROCESSABLE_ENTITY);
}

// ============================================================================
// Cache integration tests
// ============================================================================

#[tokio::test]
async fn test_verify_usl_cache_hit() {
    use crate::cache::VerificationCache;
    use crate::tools::{VerifyUslResult, VerifyUslTool};
    use std::sync::Arc;

    // Create a tool with cache enabled
    let cache = Arc::new(VerificationCache::new());
    let tool = VerifyUslTool::with_cache(cache.clone());

    // First request - should be a miss
    // Use typecheck_only to ensure result gets cached (errors from actual
    // verification are not cached since backends aren't installed in test env)
    // Use theorem syntax (matching existing tests)
    let args = json!({
        "spec": "theorem test { forall x: Int . x > 0 implies x >= 0 }",
        "strategy": "auto",
        "backends": [],
        "typecheck_only": true
    });

    let result1 = tool.execute(args.clone()).await.unwrap();
    let text1 = match &result1.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed1: VerifyUslResult = serde_json::from_str(text1).unwrap();
    assert_eq!(parsed1.cache_hit, Some(false));
    assert!(parsed1.success, "Typecheck-only should succeed");

    // Second request with same args - should be a cache hit
    let result2 = tool.execute(args).await.unwrap();
    let text2 = match &result2.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed2: VerifyUslResult = serde_json::from_str(text2).unwrap();
    assert_eq!(parsed2.cache_hit, Some(true));

    // Verify cache stats
    let stats = cache.stats().await;
    assert_eq!(stats.hits, 1);
    assert!(stats.misses >= 1);
}

#[tokio::test]
async fn test_verify_usl_cache_different_params() {
    use crate::cache::VerificationCache;
    use crate::tools::{VerifyUslResult, VerifyUslTool};
    use std::sync::Arc;

    // Test that different cache keys (via different strategy) produce cache misses
    // Use typecheck_only to ensure results get cached (backends aren't installed in test)
    let cache = Arc::new(VerificationCache::new());
    let tool = VerifyUslTool::with_cache(cache.clone());

    // Use theorem syntax (matching existing tests)
    let spec = "theorem test { forall x: Int . x > 0 }";

    // First request with typecheck_only=true
    let args1 = json!({
        "spec": spec,
        "backends": ["lean4"],
        "typecheck_only": true
    });

    let result1 = tool.execute(args1).await.unwrap();
    let text1 = match &result1.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed1: VerifyUslResult = serde_json::from_str(text1).unwrap();
    assert_eq!(parsed1.cache_hit, Some(false));

    // Second request with DIFFERENT backend list - should be a miss (different cache key)
    let args2 = json!({
        "spec": spec,
        "backends": ["kani"],
        "typecheck_only": true
    });

    let result2 = tool.execute(args2).await.unwrap();
    let text2 = match &result2.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed2: VerifyUslResult = serde_json::from_str(text2).unwrap();
    assert_eq!(parsed2.cache_hit, Some(false)); // Different backends = miss

    // Third request with same params as first - should be a hit
    let args3 = json!({
        "spec": spec,
        "backends": ["lean4"],
        "typecheck_only": true
    });

    let result3 = tool.execute(args3).await.unwrap();
    let text3 = match &result3.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed3: VerifyUslResult = serde_json::from_str(text3).unwrap();
    assert_eq!(parsed3.cache_hit, Some(true)); // Same as first = hit
}

#[tokio::test]
async fn test_verify_usl_no_cache() {
    use crate::tools::{VerifyUslResult, VerifyUslTool};

    // Create a tool without cache
    let tool = VerifyUslTool::new();

    // Use theorem syntax (matching existing tests)
    let args = json!({
        "spec": "theorem test { forall x: Int . x > 0 }",
        "typecheck_only": true  // Need typecheck_only for consistent test behavior
    });

    // First request
    let result1 = tool.execute(args.clone()).await.unwrap();
    let text1 = match &result1.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed1: VerifyUslResult = serde_json::from_str(text1).unwrap();
    // Without cache, cache_hit should still be false
    assert_eq!(parsed1.cache_hit, Some(false));

    // Second request - still no cache
    let result2 = tool.execute(args).await.unwrap();
    let text2 = match &result2.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed2: VerifyUslResult = serde_json::from_str(text2).unwrap();
    assert_eq!(parsed2.cache_hit, Some(false));
}

#[tokio::test]
async fn test_batch_verify_with_cache() {
    use crate::cache::VerificationCache;
    use crate::tools::{BatchVerifyResult, BatchVerifyTool, VerifyUslResult};
    use std::sync::Arc;

    // Test that BatchVerifyTool can be created with a cache
    let cache = Arc::new(VerificationCache::<VerifyUslResult>::new());
    let tool = BatchVerifyTool::with_cache(cache.clone());

    // Use a simple provable spec with typecheck syntax
    let spec = "theorem test { forall x: Int . x + 0 == x }";

    // Execute batch verification
    let args = json!({
        "specs": [
            {"id": "spec1", "spec": spec, "backends": []}
        ]
    });

    let result = tool.execute(args).await.unwrap();
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    let parsed: BatchVerifyResult = serde_json::from_str(text).unwrap();

    // Result should indicate cache_hit status (false on first run)
    assert_eq!(parsed.total, 1);
    assert!(parsed.results[0].cache_hit.is_some());
    assert_eq!(parsed.results[0].cache_hit, Some(false));
}

#[tokio::test]
async fn test_tool_registry_has_shared_cache() {
    // Verify that the ToolRegistry creates and shares a cache
    let registry = ToolRegistry::new();

    // Get cache stats
    let stats = registry.cache_stats().await;
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}

#[tokio::test]
async fn test_tool_registry_cache_disabled() {
    // Create registry with caching disabled
    let registry = ToolRegistry::with_caching(false);

    // Execute a verification twice
    let args = json!({
        "spec": "property P: forall x: Int . x > 0",
        "skip_health_check": true
    });

    let _ = registry.execute("dashprove.verify_usl", args.clone()).await;
    let _ = registry.execute("dashprove.verify_usl", args).await;

    // With cache disabled, stats should show no entries
    let stats = registry.cache_stats().await;
    assert_eq!(stats.entries, 0);
}

// ============================================================================
// HTTP Cache Endpoints Tests
// ============================================================================

#[tokio::test]
async fn test_http_cache_stats_endpoint() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/cache/stats",
            axum::routing::get(crate::transport::handle_cache_stats),
        )
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/cache/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Verify response structure
    assert!(json.get("hits").is_some());
    assert!(json.get("misses").is_some());
    assert!(json.get("entries").is_some());
    assert!(json.get("evictions").is_some());
    assert!(json.get("expirations").is_some());
    assert!(json.get("hit_rate").is_some());
    assert!(json.get("enabled").is_some());

    // Initial stats should be zeros
    assert_eq!(json["hits"], 0);
    assert_eq!(json["misses"], 0);
    assert_eq!(json["entries"], 0);
    assert_eq!(json["enabled"], true);
}

#[tokio::test]
async fn test_http_cache_clear_endpoint() {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/cache/clear",
            axum::routing::post(crate::transport::handle_cache_clear),
        )
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/cache/clear")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["success"], true);
    assert!(json.get("entries_cleared").is_some());
    assert_eq!(json["entries_cleared"], 0); // No entries to clear initially
}

#[tokio::test]
async fn test_http_cache_stats_after_verification() {
    use axum::{
        body::Body,
        http::{header, Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/verify",
            axum::routing::post(crate::transport::handle_verify),
        )
        .route(
            "/cache/stats",
            axum::routing::get(crate::transport::handle_cache_stats),
        )
        .with_state(state.clone())
        .layer(cors);

    // First, do a verification
    let verify_body = json!({
        "spec": "property P: forall x: Int . x > 0",
        "skip_health_check": true
    });

    let _ = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/verify")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&verify_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Now check cache stats - should show 1 miss (from initial verification)
    let stats_response = app
        .oneshot(
            Request::builder()
                .uri("/cache/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(stats_response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(stats_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Should have at least 1 miss (the initial verification) and possibly 1 entry cached
    assert!(json["misses"].as_u64().unwrap() >= 1);
}

#[tokio::test]
async fn test_http_cache_clear_with_entries() {
    use axum::{
        body::Body,
        http::{header, Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route(
            "/verify",
            axum::routing::post(crate::transport::handle_verify),
        )
        .route(
            "/cache/stats",
            axum::routing::get(crate::transport::handle_cache_stats),
        )
        .route(
            "/cache/clear",
            axum::routing::post(crate::transport::handle_cache_clear),
        )
        .with_state(state.clone())
        .layer(cors);

    // Do a verification to populate the cache
    let verify_body = json!({
        "spec": "property P: forall x: Int . x > 0",
        "skip_health_check": true
    });

    let _ = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/verify")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&verify_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Clear the cache
    let clear_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/cache/clear")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(clear_response.status(), StatusCode::OK);

    // Verify stats show 0 entries after clear
    let stats_response = app
        .oneshot(
            Request::builder()
                .uri("/cache/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(stats_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["entries"], 0);
}

#[tokio::test]
async fn test_http_cache_config_get_endpoint() {
    use axum::{body::Body, http::Request};
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let app = axum::Router::new()
        .route(
            "/cache/config",
            axum::routing::get(crate::transport::handle_cache_config_get),
        )
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/cache/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Check default values
    assert_eq!(json["ttl_secs"], 300); // 5 minutes default
    assert_eq!(json["max_entries"], 1000);
    assert_eq!(json["enabled"], true);
}

#[tokio::test]
async fn test_http_cache_config_update_endpoint() {
    use axum::{
        body::Body,
        http::{header, Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let app = axum::Router::new()
        .route(
            "/cache/config",
            axum::routing::get(crate::transport::handle_cache_config_get),
        )
        .route(
            "/cache/config",
            axum::routing::post(crate::transport::handle_cache_config_update),
        )
        .with_state(state);

    // Update TTL to 10 minutes (600 seconds)
    let update_body = json!({
        "ttl_secs": 600,
        "max_entries": 2000
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/cache/config")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&update_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["success"], true);
    assert_eq!(json["old_config"]["ttl_secs"], 300);
    assert_eq!(json["old_config"]["max_entries"], 1000);
    assert_eq!(json["new_config"]["ttl_secs"], 600);
    assert_eq!(json["new_config"]["max_entries"], 2000);

    // Verify new config persists via GET
    let get_response = app
        .oneshot(
            Request::builder()
                .uri("/cache/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(get_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["ttl_secs"], 600);
    assert_eq!(json["max_entries"], 2000);
    assert_eq!(json["enabled"], true);
}

#[tokio::test]
async fn test_http_cache_config_disable_enable() {
    use axum::{
        body::Body,
        http::{header, Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let app = axum::Router::new()
        .route(
            "/cache/config",
            axum::routing::get(crate::transport::handle_cache_config_get),
        )
        .route(
            "/cache/config",
            axum::routing::post(crate::transport::handle_cache_config_update),
        )
        .with_state(state);

    // Disable caching
    let disable_body = json!({
        "enabled": false
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/cache/config")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&disable_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify disabled
    let get_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/cache/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(get_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["enabled"], false);

    // Re-enable caching
    let enable_body = json!({
        "enabled": true
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/cache/config")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&enable_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify enabled
    let get_response = app
        .oneshot(
            Request::builder()
                .uri("/cache/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(get_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["enabled"], true);
}

#[tokio::test]
async fn test_http_cache_config_stats_includes_config() {
    use axum::{body::Body, http::Request};
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let app = axum::Router::new()
        .route(
            "/cache/stats",
            axum::routing::get(crate::transport::handle_cache_stats),
        )
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/cache/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Stats endpoint now includes config info
    assert!(json.get("ttl_secs").is_some());
    assert!(json.get("max_entries").is_some());
    assert!(json.get("enabled").is_some());
    assert_eq!(json["ttl_secs"], 300);
    assert_eq!(json["max_entries"], 1000);
    assert_eq!(json["enabled"], true);
}

// ============================================================================
// MCP Cache Tools Tests
// ============================================================================

/// Helper to extract text from Content enum
fn extract_text(content: &Content) -> &str {
    match content {
        Content::Text { text } => text,
        _ => panic!("Expected text content"),
    }
}

#[tokio::test]
async fn test_mcp_get_cache_stats_tool() {
    // Test the get_cache_stats MCP tool
    let registry = ToolRegistry::new();

    // Execute get_cache_stats tool
    let result = registry.execute("get_cache_stats", json!({})).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    let text = extract_text(&result.content[0]);
    let stats: serde_json::Value = serde_json::from_str(text).unwrap();

    // Verify response structure
    assert!(stats.get("hits").is_some());
    assert!(stats.get("misses").is_some());
    assert!(stats.get("entries").is_some());
    assert!(stats.get("evictions").is_some());
    assert!(stats.get("expirations").is_some());
    assert!(stats.get("hit_rate").is_some());
    assert!(stats.get("enabled").is_some());
    assert!(stats.get("ttl_secs").is_some());
    assert!(stats.get("max_entries").is_some());

    // Verify default values
    assert_eq!(stats["enabled"], true);
    assert_eq!(stats["ttl_secs"], 300);
    assert_eq!(stats["max_entries"], 1000);
}

#[tokio::test]
async fn test_mcp_get_cache_stats_after_verification() {
    // Test that get_cache_stats reflects verification activity
    let registry = ToolRegistry::new();

    // Run a verification to populate the cache
    // Use typecheck_only to ensure result gets cached (backends aren't installed in test env)
    let verify_args = json!({
        "spec": "theorem test { forall x: Int . x > 0 implies x >= 0 }",
        "strategy": "auto",
        "backends": [],
        "typecheck_only": true
    });
    let _ = registry
        .execute("dashprove.verify_usl", verify_args.clone())
        .await;

    // Get stats
    let result = registry
        .execute("get_cache_stats", json!({}))
        .await
        .unwrap();
    let text = extract_text(&result.content[0]);
    let stats: serde_json::Value = serde_json::from_str(text).unwrap();

    // Should have entries in the cache now
    assert_eq!(stats["entries"], 1);
    assert!(stats["misses"].as_u64().unwrap() >= 1); // First run is a miss

    // Run same verification again
    let _ = registry.execute("dashprove.verify_usl", verify_args).await;

    // Get stats again
    let result = registry
        .execute("get_cache_stats", json!({}))
        .await
        .unwrap();
    let text = extract_text(&result.content[0]);
    let stats: serde_json::Value = serde_json::from_str(text).unwrap();

    // Should have a hit now
    assert_eq!(stats["hits"], 1);
    assert_eq!(stats["entries"], 1);
}

#[tokio::test]
async fn test_mcp_clear_cache_tool() {
    // Test the clear_cache MCP tool
    let registry = ToolRegistry::new();

    // Run a verification to populate the cache
    // Use typecheck_only to ensure result gets cached (backends aren't installed in test env)
    let verify_args = json!({
        "spec": "theorem test { forall x: Int . x > 0 implies x >= 0 }",
        "strategy": "auto",
        "backends": [],
        "typecheck_only": true
    });
    let _ = registry.execute("dashprove.verify_usl", verify_args).await;

    // Verify cache has entries
    let stats = registry.cache_stats().await;
    assert_eq!(stats.entries, 1);

    // Clear the cache
    let result = registry.execute("clear_cache", json!({})).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    let text = extract_text(&result.content[0]);
    let clear_result: serde_json::Value = serde_json::from_str(text).unwrap();

    // Verify response
    assert_eq!(clear_result["success"], true);
    assert_eq!(clear_result["cleared"], 1);
    assert!(clear_result["message"]
        .as_str()
        .unwrap()
        .contains("Cleared 1 cached entries"));

    // Verify cache is empty
    let stats = registry.cache_stats().await;
    assert_eq!(stats.entries, 0);
}

#[tokio::test]
async fn test_mcp_clear_cache_empty() {
    // Test clear_cache when cache is already empty
    let registry = ToolRegistry::new();

    let result = registry.execute("clear_cache", json!({})).await.unwrap();
    let text = extract_text(&result.content[0]);
    let clear_result: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(clear_result["success"], true);
    assert_eq!(clear_result["cleared"], 0);
    assert!(clear_result["message"]
        .as_str()
        .unwrap()
        .contains("Cleared 0 cached entries"));
}

#[tokio::test]
async fn test_mcp_configure_cache_tool_update_ttl() {
    // Test updating TTL via configure_cache
    let registry = ToolRegistry::new();

    let result = registry
        .execute(
            "configure_cache",
            json!({
                "ttl_secs": 600
            }),
        )
        .await;
    assert!(result.is_ok());

    let result = result.unwrap();
    let text = extract_text(&result.content[0]);
    let config_result: serde_json::Value = serde_json::from_str(text).unwrap();

    // Verify response
    assert_eq!(config_result["success"], true);
    assert_eq!(config_result["old_config"]["ttl_secs"], 300);
    assert_eq!(config_result["new_config"]["ttl_secs"], 600);
    assert!(config_result["message"]
        .as_str()
        .unwrap()
        .contains("TTL: 300s -> 600s"));

    // Verify change persisted via get_cache_stats
    let stats_result = registry
        .execute("get_cache_stats", json!({}))
        .await
        .unwrap();
    let text = extract_text(&stats_result.content[0]);
    let stats: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(stats["ttl_secs"], 600);
}

#[tokio::test]
async fn test_mcp_configure_cache_tool_update_max_entries() {
    // Test updating max_entries via configure_cache
    let registry = ToolRegistry::new();

    let result = registry
        .execute(
            "configure_cache",
            json!({
                "max_entries": 500
            }),
        )
        .await
        .unwrap();

    let text = extract_text(&result.content[0]);
    let config_result: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(config_result["success"], true);
    assert_eq!(config_result["old_config"]["max_entries"], 1000);
    assert_eq!(config_result["new_config"]["max_entries"], 500);
    assert!(config_result["message"]
        .as_str()
        .unwrap()
        .contains("max_entries: 1000 -> 500"));
}

#[tokio::test]
async fn test_mcp_configure_cache_tool_disable_enable() {
    // Test disabling and re-enabling the cache
    let registry = ToolRegistry::new();

    // Disable cache
    let result = registry
        .execute(
            "configure_cache",
            json!({
                "enabled": false
            }),
        )
        .await
        .unwrap();

    let text = extract_text(&result.content[0]);
    let config_result: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(config_result["old_config"]["enabled"], true);
    assert_eq!(config_result["new_config"]["enabled"], false);

    // Verify via get_cache_stats
    let stats_result = registry
        .execute("get_cache_stats", json!({}))
        .await
        .unwrap();
    let text = extract_text(&stats_result.content[0]);
    let stats: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(stats["enabled"], false);

    // Re-enable cache
    let result = registry
        .execute(
            "configure_cache",
            json!({
                "enabled": true
            }),
        )
        .await
        .unwrap();

    let text = extract_text(&result.content[0]);
    let config_result: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(config_result["old_config"]["enabled"], false);
    assert_eq!(config_result["new_config"]["enabled"], true);
}

#[tokio::test]
async fn test_mcp_configure_cache_no_changes() {
    // Test configure_cache with no actual changes
    let registry = ToolRegistry::new();

    let result = registry
        .execute("configure_cache", json!({}))
        .await
        .unwrap();

    let text = extract_text(&result.content[0]);
    let config_result: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(config_result["success"], true);
    assert_eq!(config_result["message"], "No configuration changes applied");
    assert_eq!(
        config_result["old_config"]["ttl_secs"],
        config_result["new_config"]["ttl_secs"]
    );
}

#[tokio::test]
async fn test_mcp_configure_cache_clear_on_change() {
    // Test that clear_on_change works when reducing TTL
    let registry = ToolRegistry::new();

    // Populate cache
    // Use typecheck_only to ensure result gets cached (backends aren't installed in test env)
    let verify_args = json!({
        "spec": "theorem test { forall x: Int . x > 0 implies x >= 0 }",
        "strategy": "auto",
        "backends": [],
        "typecheck_only": true
    });
    let _ = registry.execute("dashprove.verify_usl", verify_args).await;

    // Verify cache has entries
    let stats = registry.cache_stats().await;
    assert_eq!(stats.entries, 1);

    // Reduce TTL with clear_on_change = true
    let _ = registry
        .execute(
            "configure_cache",
            json!({
                "ttl_secs": 60,
                "clear_on_change": true
            }),
        )
        .await
        .unwrap();

    // Cache should be cleared
    let stats = registry.cache_stats().await;
    assert_eq!(stats.entries, 0);
}

#[tokio::test]
async fn test_mcp_configure_cache_multiple_changes() {
    // Test updating multiple values at once
    let registry = ToolRegistry::new();

    let result = registry
        .execute(
            "configure_cache",
            json!({
                "ttl_secs": 120,
                "max_entries": 200,
                "enabled": true
            }),
        )
        .await
        .unwrap();

    let text = extract_text(&result.content[0]);
    let config_result: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(config_result["new_config"]["ttl_secs"], 120);
    assert_eq!(config_result["new_config"]["max_entries"], 200);
    assert_eq!(config_result["new_config"]["enabled"], true);

    // Message should contain both changes
    let message = config_result["message"].as_str().unwrap();
    assert!(message.contains("TTL: 300s -> 120s"));
    assert!(message.contains("max_entries: 1000 -> 200"));
}

#[tokio::test]
async fn test_mcp_cache_tools_in_tool_definitions() {
    // Verify the new tools are in the registry's definitions
    let registry = ToolRegistry::new();
    let definitions = registry.definitions();

    let tool_names: Vec<&str> = definitions.iter().map(|d| d.name.as_str()).collect();

    assert!(tool_names.contains(&"get_cache_stats"));
    assert!(tool_names.contains(&"clear_cache"));
    assert!(tool_names.contains(&"configure_cache"));
    assert!(tool_names.contains(&"save_cache"));
    assert!(tool_names.contains(&"load_cache"));
}

// ============================================================================
// Cache persistence tools tests (save_cache, load_cache)
// ============================================================================

#[tokio::test]
async fn test_mcp_save_cache_tool() {
    // Test saving the cache to a file
    let registry = ToolRegistry::new();
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("test_mcp_save_cache.json");

    // First add some entries to the cache using typecheck_only
    // (backends aren't installed in test environment)
    let _ = registry
        .execute(
            "dashprove.verify_usl",
            json!({
                "spec": "theorem test { forall x: Int . x > 0 implies x >= 0 }",
                "backends": [],
                "typecheck_only": true
            }),
        )
        .await
        .unwrap();

    // Save the cache
    let result = registry
        .execute(
            "save_cache",
            json!({
                "path": path.to_str().unwrap()
            }),
        )
        .await;
    assert!(result.is_ok());

    let result = result.unwrap();
    let text = extract_text(&result.content[0]);
    let save_result: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(save_result["success"], true);
    assert!(save_result["entries_saved"].as_u64().unwrap() >= 1);
    assert!(save_result["size_bytes"].as_u64().unwrap() > 0);
    assert!(save_result["message"].as_str().unwrap().contains("Saved"));

    // Verify file exists
    assert!(std::path::Path::new(path.to_str().unwrap()).exists());

    // Cleanup
    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn test_mcp_save_cache_tool_empty() {
    // Test saving an empty cache
    let registry = ToolRegistry::new();
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("test_mcp_save_cache_empty.json");

    // Save without any entries
    let result = registry
        .execute(
            "save_cache",
            json!({
                "path": path.to_str().unwrap()
            }),
        )
        .await
        .unwrap();

    let text = extract_text(&result.content[0]);
    let save_result: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(save_result["success"], true);
    assert_eq!(save_result["entries_saved"], 0);

    // Cleanup
    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn test_mcp_load_cache_tool() {
    // Test loading the cache from a file
    let registry = ToolRegistry::new();
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("test_mcp_load_cache.json");

    // Add an entry and save using typecheck_only (backends not installed in test)
    let _ = registry
        .execute(
            "dashprove.verify_usl",
            json!({
                "spec": "theorem load_test { forall x: Int . x >= 0 }",
                "backends": [],
                "typecheck_only": true
            }),
        )
        .await;

    let _ = registry
        .execute(
            "save_cache",
            json!({
                "path": path.to_str().unwrap()
            }),
        )
        .await;

    // Clear the cache
    let _ = registry.execute("clear_cache", json!({})).await;

    // Verify cache is empty
    let stats_result = registry
        .execute("get_cache_stats", json!({}))
        .await
        .unwrap();
    let text = extract_text(&stats_result.content[0]);
    let stats: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(stats["entries"], 0);

    // Load the cache
    let result = registry
        .execute(
            "load_cache",
            json!({
                "path": path.to_str().unwrap()
            }),
        )
        .await
        .unwrap();

    let text = extract_text(&result.content[0]);
    let load_result: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(load_result["success"], true);
    assert!(load_result["entries_loaded"].as_u64().unwrap() >= 1);
    assert_eq!(load_result["entries_expired"], 0);
    assert!(load_result["message"].as_str().unwrap().contains("Loaded"));

    // Verify entries restored
    let stats_result = registry
        .execute("get_cache_stats", json!({}))
        .await
        .unwrap();
    let text = extract_text(&stats_result.content[0]);
    let stats: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(stats["entries"].as_u64().unwrap() >= 1);

    // Cleanup
    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn test_mcp_load_cache_tool_merge() {
    // Test loading the cache with merge mode
    let registry = ToolRegistry::new();
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("test_mcp_load_cache_merge.json");

    // Add entry and save using typecheck_only
    let _ = registry
        .execute(
            "dashprove.verify_usl",
            json!({
                "spec": "theorem merge_spec1 { forall x: Int . x > 0 }",
                "backends": [],
                "typecheck_only": true
            }),
        )
        .await;

    let _ = registry
        .execute(
            "save_cache",
            json!({
                "path": path.to_str().unwrap()
            }),
        )
        .await;

    // Clear and add a different entry
    let _ = registry.execute("clear_cache", json!({})).await;

    let _ = registry
        .execute(
            "dashprove.verify_usl",
            json!({
                "spec": "theorem merge_spec2 { forall y: Int . y >= 0 }",
                "backends": [],
                "typecheck_only": true
            }),
        )
        .await;

    // Verify we have 1 entry
    let stats_result = registry
        .execute("get_cache_stats", json!({}))
        .await
        .unwrap();
    let text = extract_text(&stats_result.content[0]);
    let stats: serde_json::Value = serde_json::from_str(text).unwrap();
    let entries_before = stats["entries"].as_u64().unwrap();
    assert_eq!(entries_before, 1);

    // Load with merge=true
    let result = registry
        .execute(
            "load_cache",
            json!({
                "path": path.to_str().unwrap(),
                "merge": true
            }),
        )
        .await
        .unwrap();

    let text = extract_text(&result.content[0]);
    let load_result: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(load_result["success"], true);
    assert!(load_result["message"].as_str().unwrap().contains("merged"));

    // Verify we have 2 entries (original + loaded)
    let stats_result = registry
        .execute("get_cache_stats", json!({}))
        .await
        .unwrap();
    let text = extract_text(&stats_result.content[0]);
    let stats: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(stats["entries"], 2);

    // Cleanup
    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn test_mcp_load_cache_tool_replace() {
    // Test loading the cache with replace mode (default)
    let registry = ToolRegistry::new();
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("test_mcp_load_cache_replace.json");

    // Add entry and save using typecheck_only
    let _ = registry
        .execute(
            "dashprove.verify_usl",
            json!({
                "spec": "theorem replace_spec1 { forall x: Int . x > 0 }",
                "backends": [],
                "typecheck_only": true
            }),
        )
        .await;

    let _ = registry
        .execute(
            "save_cache",
            json!({
                "path": path.to_str().unwrap()
            }),
        )
        .await;

    // Clear and add a different entry
    let _ = registry.execute("clear_cache", json!({})).await;

    let _ = registry
        .execute(
            "dashprove.verify_usl",
            json!({
                "spec": "theorem replace_spec2 { forall y: Int . y >= 0 }",
                "backends": [],
                "typecheck_only": true
            }),
        )
        .await;

    // Load with merge=false (replace)
    let result = registry
        .execute(
            "load_cache",
            json!({
                "path": path.to_str().unwrap(),
                "merge": false
            }),
        )
        .await
        .unwrap();

    let text = extract_text(&result.content[0]);
    let load_result: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(load_result["success"], true);
    assert!(load_result["message"]
        .as_str()
        .unwrap()
        .contains("replaced"));

    // Verify we have 1 entry (only the loaded one)
    let stats_result = registry
        .execute("get_cache_stats", json!({}))
        .await
        .unwrap();
    let text = extract_text(&stats_result.content[0]);
    let stats: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(stats["entries"], 1);

    // Cleanup
    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn test_mcp_save_cache_tool_invalid_path() {
    // Test saving to an invalid path
    let registry = ToolRegistry::new();

    let result = registry
        .execute(
            "save_cache",
            json!({
                "path": "/nonexistent/directory/cache.json"
            }),
        )
        .await
        .unwrap();

    let text = extract_text(&result.content[0]);
    let save_result: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(save_result["success"], false);
    assert!(save_result["message"].as_str().unwrap().contains("Failed"));
}

#[tokio::test]
async fn test_mcp_load_cache_tool_file_not_found() {
    // Test loading from a non-existent file
    let registry = ToolRegistry::new();

    let result = registry
        .execute(
            "load_cache",
            json!({
                "path": "/nonexistent/cache.json"
            }),
        )
        .await
        .unwrap();

    let text = extract_text(&result.content[0]);
    let load_result: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(load_result["success"], false);
    assert!(load_result["message"].as_str().unwrap().contains("Failed"));
}

#[tokio::test]
async fn test_mcp_cache_persistence_tools_in_definitions() {
    // Verify save_cache and load_cache are in the registry definitions
    let registry = ToolRegistry::new();
    let definitions = registry.definitions();

    // Should now have 14 tools (12 original + 2 new persistence tools)
    assert_eq!(definitions.len(), 14);

    // Verify save_cache definition
    let save_def = definitions.iter().find(|d| d.name == "save_cache").unwrap();
    assert_eq!(save_def.title, Some("Save Cache".to_string()));
    assert!(save_def.description.contains("persistence"));
    let props = save_def.input_schema.properties.as_ref().unwrap();
    assert!(props.contains_key("path"));

    // Verify load_cache definition
    let load_def = definitions.iter().find(|d| d.name == "load_cache").unwrap();
    assert_eq!(load_def.title, Some("Load Cache".to_string()));
    assert!(load_def.description.contains("previously saved"));
    let props = load_def.input_schema.properties.as_ref().unwrap();
    assert!(props.contains_key("path"));
    assert!(props.contains_key("merge"));
}

// ============================================================================
// WebSocket Transport tests
// ============================================================================

#[test]
fn test_websocket_transport_default_address() {
    let transport = crate::transport::WebSocketTransport::default();
    assert_eq!(transport.bind_addr, "127.0.0.1:3002");
}

#[test]
fn test_websocket_transport_custom_address() {
    let transport = crate::transport::WebSocketTransport::new("0.0.0.0:8080");
    assert_eq!(transport.bind_addr, "0.0.0.0:8080");
}

#[test]
fn test_ws_client_message_json_rpc_serialization() {
    use crate::protocol::JsonRpcRequest;
    use crate::transport::WsClientMessage;

    let request = JsonRpcRequest::new(1i64, "test/method");
    let msg = WsClientMessage::JsonRpc(request);

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"json_rpc\""));
    assert!(json.contains("\"method\":\"test/method\""));

    // Deserialize back
    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::JsonRpc(req) => {
            assert_eq!(req.method, "test/method");
        }
        _ => panic!("Expected JsonRpc variant"),
    }
}

#[test]
fn test_ws_client_message_subscribe_serialization() {
    use crate::transport::WsClientMessage;

    let msg = WsClientMessage::Subscribe {
        session_id: "test-session-123".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"subscribe\""));
    assert!(json.contains("\"session_id\":\"test-session-123\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::Subscribe { session_id } => {
            assert_eq!(session_id, "test-session-123");
        }
        _ => panic!("Expected Subscribe variant"),
    }
}

#[test]
fn test_ws_client_message_unsubscribe_serialization() {
    use crate::transport::WsClientMessage;

    let msg = WsClientMessage::Unsubscribe {
        session_id: "test-session-456".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"unsubscribe\""));
    assert!(json.contains("\"session_id\":\"test-session-456\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::Unsubscribe { session_id } => {
            assert_eq!(session_id, "test-session-456");
        }
        _ => panic!("Expected Unsubscribe variant"),
    }
}

#[test]
fn test_ws_client_message_ping_serialization() {
    use crate::transport::WsClientMessage;

    let msg = WsClientMessage::Ping;

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"ping\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    assert!(matches!(parsed, WsClientMessage::Ping));
}

#[test]
fn test_ws_server_message_json_rpc_serialization() {
    use crate::protocol::JsonRpcResponse;
    use crate::transport::WsServerMessage;

    let response = JsonRpcResponse::success(
        crate::protocol::RequestId::Number(1),
        json!({"result": "success"}),
    );
    let msg = WsServerMessage::JsonRpc(response);

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"json_rpc\""));
    assert!(json.contains("\"result\":"));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::JsonRpc(resp) => {
            assert!(resp.result.is_some());
        }
        _ => panic!("Expected JsonRpc variant"),
    }
}

#[test]
fn test_ws_server_message_subscribed_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::Subscribed {
        session_id: "session-abc".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"subscribed\""));
    assert!(json.contains("\"session_id\":\"session-abc\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::Subscribed { session_id } => {
            assert_eq!(session_id, "session-abc");
        }
        _ => panic!("Expected Subscribed variant"),
    }
}

#[test]
fn test_ws_server_message_unsubscribed_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::Unsubscribed {
        session_id: "session-xyz".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"unsubscribed\""));
    assert!(json.contains("\"session_id\":\"session-xyz\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::Unsubscribed { session_id } => {
            assert_eq!(session_id, "session-xyz");
        }
        _ => panic!("Expected Unsubscribed variant"),
    }
}

#[test]
fn test_ws_server_message_subscription_error_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SubscriptionError {
        session_id: "bad-session".to_string(),
        message: "Session not found".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"subscription_error\""));
    assert!(json.contains("\"session_id\":\"bad-session\""));
    assert!(json.contains("\"message\":\"Session not found\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SubscriptionError {
            session_id,
            message,
        } => {
            assert_eq!(session_id, "bad-session");
            assert_eq!(message, "Session not found");
        }
        _ => panic!("Expected SubscriptionError variant"),
    }
}

#[test]
fn test_ws_server_message_pong_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::Pong;

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"pong\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    assert!(matches!(parsed, WsServerMessage::Pong));
}

#[test]
fn test_ws_server_message_error_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::Error {
        message: "Something went wrong".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"error\""));
    assert!(json.contains("\"message\":\"Something went wrong\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::Error { message } => {
            assert_eq!(message, "Something went wrong");
        }
        _ => panic!("Expected Error variant"),
    }
}

#[test]
fn test_ws_server_message_event_serialization() {
    use crate::streaming::VerificationEvent;
    use crate::transport::WsServerMessage;

    let event = VerificationEvent::Started {
        session_id: "event-session".to_string(),
        spec_summary: "property P1: always(x > 0)".to_string(),
        total_properties: 1,
        backends: vec!["lean4".to_string()],
    };

    let msg = WsServerMessage::Event {
        session_id: "event-session".to_string(),
        event,
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"event\""));
    assert!(json.contains("\"session_id\":\"event-session\""));
    assert!(json.contains("\"started\"") || json.contains("Started"));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::Event { session_id, event } => {
            assert_eq!(session_id, "event-session");
            match event {
                VerificationEvent::Started { session_id, .. } => {
                    assert_eq!(session_id, "event-session");
                }
                _ => panic!("Expected Started event"),
            }
        }
        _ => panic!("Expected Event variant"),
    }
}

#[tokio::test]
async fn test_websocket_info_endpoint() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use tower_http::cors::{Any, CorsLayer};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let state = crate::transport::HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = axum::Router::new()
        .route("/", axum::routing::get(crate::transport::handle_ws_info))
        .with_state(state)
        .layer(cors);

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["name"], "dashprove-mcp");
    assert_eq!(json["transport"], "WebSocket");
    assert!(json["websocket_endpoint"].as_str().unwrap().contains("/ws"));
    assert!(json["message_types"].is_object());
    assert!(json["message_types"]["client"].is_object());
    assert!(json["message_types"]["server"].is_object());
}

#[tokio::test]
async fn test_websocket_transport_invalid_bind_address() {
    use crate::transport::WebSocketTransport;

    let transport = WebSocketTransport::new("invalid-address");
    let server = McpServer::new();

    let result = transport.run_async(server).await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Invalid bind address"));
}

// ============================================================================
// Server with shared cache tests
// ============================================================================

#[tokio::test]
async fn test_server_with_cache() {
    use crate::cache::{SharedVerificationCache, VerificationCache};
    use crate::tools::VerifyUslResult;
    use std::sync::Arc;

    // Create a shared cache externally
    let cache: SharedVerificationCache<VerifyUslResult> = Arc::new(VerificationCache::new());

    // Create server with the shared cache
    let server = McpServer::with_cache(cache.clone());

    // Verify the server returns the same cache
    let server_cache = server.verification_cache();
    assert!(Arc::ptr_eq(&cache, &server_cache));
}

#[tokio::test]
async fn test_server_verification_cache_method() {
    let server = McpServer::new();
    let cache = server.verification_cache();

    // Cache should be enabled by default
    assert!(cache.is_enabled().await);

    // Stats should be empty
    let stats = cache.stats().await;
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}

#[tokio::test]
async fn test_tool_registry_with_shared_cache() {
    use crate::cache::{SharedVerificationCache, VerificationCache};
    use crate::tools::VerifyUslResult;
    use std::sync::Arc;

    // Create a shared cache externally
    let cache: SharedVerificationCache<VerifyUslResult> = Arc::new(VerificationCache::new());

    // Create registry with the shared cache
    let registry = ToolRegistry::with_shared_cache(cache.clone());

    // Verify the registry returns the same cache
    let registry_cache = registry.verification_cache();
    assert!(Arc::ptr_eq(&cache, &registry_cache));
}

#[tokio::test]
async fn test_tool_registry_with_shared_cache_disabled() {
    use crate::cache::{SharedVerificationCache, VerificationCache};
    use crate::tools::VerifyUslResult;
    use std::sync::Arc;

    // Create a disabled cache externally
    let cache: SharedVerificationCache<VerifyUslResult> = Arc::new(VerificationCache::disabled());

    // Create registry with the disabled cache
    let registry = ToolRegistry::with_shared_cache(cache.clone());

    // Verify the cache is disabled
    let registry_cache = registry.verification_cache();
    assert!(!registry_cache.is_enabled().await);
}

#[tokio::test]
async fn test_server_with_cache_shares_across_tools() {
    use crate::cache::{CacheKey, SharedVerificationCache, VerificationCache};
    use crate::tools::VerifyUslResult;
    use std::sync::Arc;

    // Create a shared cache externally
    let cache: SharedVerificationCache<VerifyUslResult> = Arc::new(VerificationCache::new());

    // Create server with the shared cache
    let server = McpServer::with_cache(cache.clone());

    // Insert a value directly into the cache
    let key = CacheKey::new("test_spec", &["lean4".to_string()], "auto", false);
    let result = VerifyUslResult {
        success: true,
        results: vec![],
        parse_errors: None,
        summary: "Test result".to_string(),
        cache_hit: None,
        cache_age_secs: None,
    };
    cache.insert(key.clone(), result).await;

    // Verify the server's cache also has the entry
    let server_cache = server.verification_cache();
    let stats = server_cache.stats().await;
    assert_eq!(stats.entries, 1);

    // The cache should be the same object
    let fetched = server_cache.get(&key).await;
    assert!(fetched.is_some());
    assert_eq!(fetched.unwrap().summary, "Test result");
}

// ============================================================================
// Authentication tests
// ============================================================================

#[test]
fn test_http_transport_with_auth_default() {
    use crate::transport::HttpTransport;

    let transport = HttpTransport::new("127.0.0.1:3001");
    assert_eq!(transport.bind_addr, "127.0.0.1:3001");
    assert!(transport.api_token.is_none());
}

#[test]
fn test_http_transport_with_auth_token() {
    use crate::transport::HttpTransport;

    let transport = HttpTransport::with_auth("127.0.0.1:3001", Some("test-token".to_string()));
    assert_eq!(transport.bind_addr, "127.0.0.1:3001");
    assert_eq!(transport.api_token, Some("test-token".to_string()));
}

#[test]
fn test_http_transport_with_auth_no_token() {
    use crate::transport::HttpTransport;

    let transport = HttpTransport::with_auth("127.0.0.1:3001", None);
    assert_eq!(transport.bind_addr, "127.0.0.1:3001");
    assert!(transport.api_token.is_none());
}

#[test]
fn test_websocket_transport_with_auth_default() {
    use crate::transport::WebSocketTransport;

    let transport = WebSocketTransport::new("127.0.0.1:3002");
    assert_eq!(transport.bind_addr, "127.0.0.1:3002");
    assert!(transport.api_token.is_none());
}

#[test]
fn test_websocket_transport_with_auth_token() {
    use crate::transport::WebSocketTransport;

    let transport = WebSocketTransport::with_auth("127.0.0.1:3002", Some("ws-token".to_string()));
    assert_eq!(transport.bind_addr, "127.0.0.1:3002");
    assert_eq!(transport.api_token, Some("ws-token".to_string()));
}

#[test]
fn test_websocket_transport_with_auth_no_token() {
    use crate::transport::WebSocketTransport;

    let transport = WebSocketTransport::with_auth("127.0.0.1:3002", None);
    assert_eq!(transport.bind_addr, "127.0.0.1:3002");
    assert!(transport.api_token.is_none());
}

#[tokio::test]
async fn test_http_server_state_with_auth() {
    use crate::streaming::SessionManager;
    use crate::transport::HttpServerState;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let server = McpServer::new();
    let session_manager = Arc::new(SessionManager::new());

    let state_with_token = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager: session_manager.clone(),
        api_token: Some("secret-token".to_string()),
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    assert_eq!(state_with_token.api_token, Some("secret-token".to_string()));
}

#[tokio::test]
async fn test_http_server_state_no_auth() {
    use crate::streaming::SessionManager;
    use crate::transport::HttpServerState;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let server = McpServer::new();
    let session_manager = Arc::new(SessionManager::new());

    let state_no_token = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    assert!(state_no_token.api_token.is_none());
}

#[tokio::test]
async fn test_auth_middleware_no_token_configured() {
    use crate::streaming::SessionManager;
    use crate::transport::{auth_middleware, handle_health, HttpServerState};
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = Arc::new(SessionManager::new());

    // No token configured - should allow all requests
    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let app = Router::new()
        .route("/test", get(handle_health))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    let response = app
        .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
        .await
        .unwrap();

    // Should succeed without auth when no token is configured
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
async fn test_auth_middleware_missing_token() {
    use crate::streaming::SessionManager;
    use crate::transport::{auth_middleware, handle_health, HttpServerState};
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = Arc::new(SessionManager::new());

    // Token configured but request has no auth
    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: Some("secret-token".to_string()),
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let app = Router::new()
        .route("/test", get(handle_health))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    let response = app
        .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
        .await
        .unwrap();

    // Should fail - no auth provided
    assert_eq!(response.status(), axum::http::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_middleware_valid_bearer_token() {
    use crate::streaming::SessionManager;
    use crate::transport::{auth_middleware, handle_health, HttpServerState};
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = Arc::new(SessionManager::new());

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: Some("secret-token".to_string()),
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let app = Router::new()
        .route("/test", get(handle_health))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/test")
                .header("Authorization", "Bearer secret-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should succeed - valid Bearer token
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
async fn test_auth_middleware_invalid_bearer_token() {
    use crate::streaming::SessionManager;
    use crate::transport::{auth_middleware, handle_health, HttpServerState};
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = Arc::new(SessionManager::new());

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: Some("secret-token".to_string()),
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let app = Router::new()
        .route("/test", get(handle_health))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/test")
                .header("Authorization", "Bearer wrong-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should fail - wrong token
    assert_eq!(response.status(), axum::http::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_middleware_valid_query_param() {
    use crate::streaming::SessionManager;
    use crate::transport::{auth_middleware, handle_health, HttpServerState};
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = Arc::new(SessionManager::new());

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: Some("secret-token".to_string()),
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let app = Router::new()
        .route("/test", get(handle_health))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/test?token=secret-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should succeed - valid query param token
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
async fn test_auth_middleware_invalid_query_param() {
    use crate::streaming::SessionManager;
    use crate::transport::{auth_middleware, handle_health, HttpServerState};
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = Arc::new(SessionManager::new());

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: Some("secret-token".to_string()),
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let app = Router::new()
        .route("/test", get(handle_health))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/test?token=wrong-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should fail - wrong token
    assert_eq!(response.status(), axum::http::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_middleware_raw_token_header() {
    use crate::streaming::SessionManager;
    use crate::transport::{auth_middleware, handle_health, HttpServerState};
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = Arc::new(SessionManager::new());

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: Some("secret-token".to_string()),
        rate_limiter: None,
        request_logger: None,
        metrics: None,
    };

    let app = Router::new()
        .route("/test", get(handle_health))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    // Also supports raw token without "Bearer " prefix
    let response = app
        .oneshot(
            Request::builder()
                .uri("/test")
                .header("Authorization", "secret-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should succeed - raw token in header
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

// ============================================================================
// Request Logging Tests
// ============================================================================

#[tokio::test]
async fn test_request_logging_middleware_captures_bodies() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::middleware;
    use axum::routing::post;
    use axum::{Json, Router};
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    use crate::logging::{LogConfig, RequestLogger};
    use crate::transport::{request_logging_middleware, HttpServerState};

    async fn echo(Json(payload): Json<serde_json::Value>) -> Json<serde_json::Value> {
        Json(payload)
    }

    let log_config = LogConfig {
        enabled: true,
        log_request_body: true,
        log_response_body: true,
        max_body_size: 128,
        ..Default::default()
    };
    let request_logger = RequestLogger::new(log_config);

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: Some(request_logger.clone()),
        metrics: None,
    };

    let app = Router::new()
        .route("/echo", post(echo))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            request_logging_middleware,
        ))
        .with_state(state);

    let body = r#"{"hello":"world"}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/echo")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let entries = request_logger.recent(1).await;
    assert_eq!(entries.len(), 1);

    let entry = &entries[0];
    assert_eq!(entry.request_body.as_deref(), Some(body));

    let response_body = entry.response_body.as_ref().expect("response body logged");
    let parsed: serde_json::Value = serde_json::from_str(response_body).unwrap();
    assert_eq!(parsed["hello"], "world");
}

#[tokio::test]
async fn test_request_logging_middleware_truncates_large_bodies() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::middleware;
    use axum::routing::post;
    use axum::{Json, Router};
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    use crate::logging::{LogConfig, RequestLogger};
    use crate::transport::{request_logging_middleware, HttpServerState};

    async fn echo(Json(payload): Json<serde_json::Value>) -> Json<serde_json::Value> {
        Json(payload)
    }

    let log_config = LogConfig {
        enabled: true,
        log_request_body: true,
        log_response_body: true,
        max_body_size: 16,
        ..Default::default()
    };
    let request_logger = RequestLogger::new(log_config.clone());

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: Some(request_logger.clone()),
        metrics: None,
    };

    let app = Router::new()
        .route("/echo", post(echo))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            request_logging_middleware,
        ))
        .with_state(state);

    let body = r#"{"text":"abcdefghijklmnopqrstuvwxyz"}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/echo")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let entries = request_logger.recent(1).await;
    assert_eq!(entries.len(), 1);

    let entry = &entries[0];
    let logged_request = entry.request_body.as_ref().expect("request body logged");
    assert!(logged_request.ends_with("...[truncated]"));
    assert!(logged_request.len() <= log_config.max_body_size);

    let logged_response = entry.response_body.as_ref().expect("response body logged");
    assert!(logged_response.ends_with("...[truncated]"));
    assert!(logged_response.len() <= log_config.max_body_size);
}

// ============================================================================
// Rate Limiting Tests
// ============================================================================

#[tokio::test]
async fn test_rate_limiter_allows_requests_within_limit() {
    use crate::ratelimit::{RateLimitConfig, RateLimiter};
    use std::net::{IpAddr, Ipv4Addr};

    let config = RateLimitConfig::new(100.0, 10); // 100 req/s, burst of 10
    let limiter = RateLimiter::new(config);
    let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();

    // Should allow up to burst_size requests
    for _ in 0..10 {
        assert!(limiter.check(ip).await, "Request should be allowed");
    }

    let stats = limiter.stats().await;
    assert_eq!(stats.allowed, 10);
    assert_eq!(stats.rejected, 0);
}

#[tokio::test]
async fn test_rate_limiter_rejects_requests_over_limit() {
    use crate::ratelimit::{RateLimitConfig, RateLimiter};
    use std::net::{IpAddr, Ipv4Addr};

    let config = RateLimitConfig::new(100.0, 3); // 100 req/s, burst of 3
    let limiter = RateLimiter::new(config);
    let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();

    // Exhaust the burst
    for _ in 0..3 {
        assert!(limiter.check(ip).await);
    }

    // Next request should be rejected
    assert!(!limiter.check(ip).await, "Request should be rejected");

    let stats = limiter.stats().await;
    assert_eq!(stats.allowed, 3);
    assert_eq!(stats.rejected, 1);
    assert!(stats.rejection_rate() > 0.0);
}

#[tokio::test]
async fn test_rate_limiter_disabled_allows_all() {
    use crate::ratelimit::{RateLimitConfig, RateLimiter};
    use std::net::{IpAddr, Ipv4Addr};

    let config = RateLimitConfig::disabled();
    let limiter = RateLimiter::new(config);
    let ip: IpAddr = Ipv4Addr::new(127, 0, 0, 1).into();

    // Should allow unlimited requests when disabled
    for _ in 0..100 {
        assert!(
            limiter.check(ip).await,
            "All requests should be allowed when disabled"
        );
    }
}

#[tokio::test]
async fn test_rate_limiter_per_ip_isolation() {
    use crate::ratelimit::{RateLimitConfig, RateLimiter};
    use std::net::{IpAddr, Ipv4Addr};

    let config = RateLimitConfig::new(100.0, 2); // 100 req/s, burst of 2
    let limiter = RateLimiter::new(config);
    let ip1: IpAddr = Ipv4Addr::new(192, 168, 1, 1).into();
    let ip2: IpAddr = Ipv4Addr::new(192, 168, 1, 2).into();

    // Exhaust ip1's burst
    assert!(limiter.check(ip1).await);
    assert!(limiter.check(ip1).await);
    assert!(!limiter.check(ip1).await, "ip1 should be rate limited");

    // ip2 should still have its full burst available
    assert!(
        limiter.check(ip2).await,
        "ip2 should not be affected by ip1"
    );
    assert!(limiter.check(ip2).await, "ip2 should still have tokens");

    let stats = limiter.stats().await;
    assert_eq!(stats.tracked_clients, 2);
}

#[tokio::test]
async fn test_rate_limiter_config_update() {
    use crate::ratelimit::{RateLimitConfig, RateLimiter};

    let config = RateLimitConfig::new(10.0, 5);
    let limiter = RateLimiter::new(config);

    // Initial config
    let cfg = limiter.config().await;
    assert_eq!(cfg.requests_per_second, 10.0);
    assert_eq!(cfg.burst_size, 5);

    // Update config
    let new_config = RateLimitConfig::new(50.0, 20);
    limiter.update_config(new_config).await;

    let cfg = limiter.config().await;
    assert_eq!(cfg.requests_per_second, 50.0);
    assert_eq!(cfg.burst_size, 20);
}

#[tokio::test]
async fn test_rate_limit_middleware_allows_requests() {
    use crate::ratelimit::{RateLimitConfig, RateLimiter};
    use crate::transport::{handle_health, rate_limit_middleware_fn, HttpServerState};
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let rate_limiter = RateLimiter::new(RateLimitConfig::new(100.0, 10));

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: Some(rate_limiter),
        request_logger: None,
        metrics: None,
    };

    let app = Router::new()
        .route("/health", get(handle_health))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            rate_limit_middleware_fn,
        ))
        .with_state(state);

    // Request should be allowed
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);

    // Check rate limit headers are present
    assert!(response.headers().contains_key("x-ratelimit-limit"));
    assert!(response.headers().contains_key("x-ratelimit-remaining"));
}

#[tokio::test]
async fn test_rate_limit_middleware_rejects_over_limit() {
    use crate::ratelimit::{RateLimitConfig, RateLimiter};
    use crate::server::McpServer;
    use crate::transport::{handle_health, rate_limit_middleware_fn, HttpServerState};
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let rate_limiter = RateLimiter::new(RateLimitConfig::new(100.0, 2)); // Only 2 burst

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: Some(rate_limiter),
        request_logger: None,
        metrics: None,
    };

    let app = Router::new()
        .route("/health", get(handle_health))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            rate_limit_middleware_fn,
        ))
        .with_state(state);

    // First 2 requests should succeed
    for _ in 0..2 {
        let app_clone = app.clone();
        let response = app_clone
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
    }

    // Third request should be rate limited
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
    assert!(response.headers().contains_key("retry-after"));
}

#[tokio::test]
async fn test_rate_limit_middleware_no_limiter_allows_all() {
    use crate::server::McpServer;
    use crate::transport::{handle_health, rate_limit_middleware_fn, HttpServerState};
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None, // No rate limiter
        request_logger: None,
        metrics: None,
    };

    let app = Router::new()
        .route("/health", get(handle_health))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            rate_limit_middleware_fn,
        ))
        .with_state(state);

    // All requests should be allowed when no rate limiter is configured
    for _ in 0..10 {
        let app_clone = app.clone();
        let response = app_clone
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
    }
}

#[tokio::test]
async fn test_http_transport_with_rate_limit() {
    use crate::ratelimit::RateLimitConfig;
    use crate::transport::HttpTransport;

    // Test that HttpTransport correctly stores rate limit config
    let transport =
        HttpTransport::with_rate_limit("127.0.0.1:3099", RateLimitConfig::new(50.0, 100));

    assert!(transport.rate_limit_config.is_some());
    let config = transport.rate_limit_config.unwrap();
    assert_eq!(config.requests_per_second, 50.0);
    assert_eq!(config.burst_size, 100);
}

#[tokio::test]
async fn test_http_transport_with_auth_and_rate_limit() {
    use crate::ratelimit::RateLimitConfig;
    use crate::transport::HttpTransport;

    let transport = HttpTransport::with_auth_and_rate_limit(
        "127.0.0.1:3099",
        Some("token123".to_string()),
        Some(RateLimitConfig::new(25.0, 50)),
    );

    assert_eq!(transport.api_token, Some("token123".to_string()));
    assert!(transport.rate_limit_config.is_some());
    let config = transport.rate_limit_config.unwrap();
    assert_eq!(config.requests_per_second, 25.0);
    assert_eq!(config.burst_size, 50);
}

#[tokio::test]
async fn test_websocket_transport_with_rate_limit() {
    use crate::ratelimit::RateLimitConfig;
    use crate::transport::WebSocketTransport;

    let transport =
        WebSocketTransport::with_rate_limit("127.0.0.1:3099", RateLimitConfig::new(30.0, 60));

    assert!(transport.rate_limit_config.is_some());
    let config = transport.rate_limit_config.unwrap();
    assert_eq!(config.requests_per_second, 30.0);
    assert_eq!(config.burst_size, 60);
}

#[tokio::test]
async fn test_websocket_transport_with_auth_and_rate_limit() {
    use crate::ratelimit::RateLimitConfig;
    use crate::transport::WebSocketTransport;

    let transport = WebSocketTransport::with_auth_and_rate_limit(
        "127.0.0.1:3099",
        Some("ws-token".to_string()),
        Some(RateLimitConfig::new(15.0, 30)),
    );

    assert_eq!(transport.api_token, Some("ws-token".to_string()));
    assert!(transport.rate_limit_config.is_some());
    let config = transport.rate_limit_config.unwrap();
    assert_eq!(config.requests_per_second, 15.0);
    assert_eq!(config.burst_size, 30);
}

// ==================== Metrics Endpoint Tests ====================

#[tokio::test]
async fn test_metrics_endpoint_disabled_returns_503() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        routing::get,
        Router,
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    use crate::server::McpServer;
    use crate::transport::{handle_metrics, HttpServerState};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();

    // State without metrics enabled
    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None, // Metrics disabled
    };

    let app = Router::new()
        .route("/metrics", get(handle_metrics))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn test_metrics_endpoint_enabled_returns_prometheus_format() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        routing::get,
        Router,
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    use crate::metrics::McpMetrics;
    use crate::server::McpServer;
    use crate::transport::{handle_metrics, HttpServerState};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let metrics = McpMetrics::enabled();

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: Some(metrics),
    };

    let app = Router::new()
        .route("/metrics", get(handle_metrics))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Check content type
    let content_type = response.headers().get("content-type").unwrap();
    assert!(content_type.to_str().unwrap().contains("text/plain"));

    // Check body contains Prometheus-format metrics
    let body = axum::body::to_bytes(response.into_body(), 100_000)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();

    assert!(body_str.contains("# HELP mcp_uptime_seconds"));
    assert!(body_str.contains("# TYPE mcp_uptime_seconds gauge"));
    assert!(body_str.contains("mcp_verifications_total"));
    assert!(body_str.contains("mcp_cache_entries"));
}

#[tokio::test]
async fn test_metrics_json_endpoint_disabled_returns_503() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        routing::get,
        Router,
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    use crate::server::McpServer;
    use crate::transport::{handle_metrics_json, HttpServerState};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: None, // Metrics disabled
    };

    let app = Router::new()
        .route("/metrics/json", get(handle_metrics_json))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn test_metrics_json_endpoint_returns_json() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        routing::get,
        Router,
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    use crate::metrics::McpMetrics;
    use crate::server::McpServer;
    use crate::transport::{handle_metrics_json, HttpServerState};

    let server = McpServer::new();
    let session_manager = server.tools_session_manager();
    let metrics = McpMetrics::enabled();

    // Record some metrics
    metrics.record_verification(true);
    metrics.record_cache_hit();

    let state = HttpServerState {
        server: Arc::new(Mutex::new(server)),
        session_manager,
        api_token: None,
        rate_limiter: None,
        request_logger: None,
        metrics: Some(metrics),
    };

    let app = Router::new()
        .route("/metrics/json", get(handle_metrics_json))
        .with_state(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), 10_000)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json.get("uptime_secs").is_some());
    assert_eq!(json["verifications_total"], 1);
    assert_eq!(json["verifications_success"], 1);
    assert_eq!(json["cache_hits"], 1);
}

#[tokio::test]
async fn test_http_transport_with_metrics() {
    use crate::metrics::MetricsConfig;
    use crate::transport::HttpTransport;

    let transport = HttpTransport::with_all_options(
        "127.0.0.1:3099",
        None,
        None,
        None,
        Some(MetricsConfig::enabled()),
    );

    assert!(transport.metrics_config.is_some());
    assert!(transport.metrics_config.unwrap().enabled);
}

#[tokio::test]
async fn test_websocket_transport_with_metrics() {
    use crate::metrics::MetricsConfig;
    use crate::transport::WebSocketTransport;

    let transport = WebSocketTransport::with_all_options(
        "127.0.0.1:3099",
        None,
        None,
        None,
        Some(MetricsConfig::enabled()),
    );

    assert!(transport.metrics_config.is_some());
    assert!(transport.metrics_config.unwrap().enabled);
}

// ============================================================================
// WebSocket Multiplexing Tests
// ============================================================================

#[test]
fn test_ws_multiplex_manager_create_session() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();
    assert_eq!(manager.session_count(), 0);

    // Create session with auto-generated ID
    let id1 = manager.create_session(None, None);
    assert!(!id1.is_empty());
    assert!(id1.starts_with("mux-"));
    assert_eq!(manager.session_count(), 1);
    assert!(manager.has_session(&id1));

    // Create session with custom ID
    let id2 = manager.create_session(Some("my-custom-session".to_string()), None);
    assert_eq!(id2, "my-custom-session");
    assert_eq!(manager.session_count(), 2);
    assert!(manager.has_session("my-custom-session"));
}

#[test]
fn test_ws_multiplex_manager_destroy_session() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();
    let id = manager.create_session(Some("test-session".to_string()), None);

    assert!(manager.has_session(&id));
    assert!(manager.destroy_session(&id));
    assert!(!manager.has_session(&id));
    assert_eq!(manager.session_count(), 0);

    // Destroying non-existent session returns false
    assert!(!manager.destroy_session("nonexistent"));
}

#[test]
fn test_ws_multiplex_manager_default_session() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();

    // First session becomes default
    let id1 = manager.create_session(Some("first".to_string()), None);
    assert_eq!(manager.default_session, Some("first".to_string()));

    // Second session doesn't change default
    let _id2 = manager.create_session(Some("second".to_string()), None);
    assert_eq!(manager.default_session, Some("first".to_string()));

    // Destroying default session switches to another
    manager.destroy_session(&id1);
    assert!(manager.default_session.is_some());
    // Default should now be "second" (the remaining session)
    assert_eq!(manager.default_session, Some("second".to_string()));
}

#[test]
fn test_ws_multiplex_manager_session_info() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();
    let metadata = Some(serde_json::json!({"user": "test", "project": "example"}));
    let id = manager.create_session(Some("info-test".to_string()), metadata.clone());

    let session = manager.get_session(&id).unwrap();
    let info = session.info();

    assert_eq!(info.session_id, "info-test");
    assert_eq!(info.metadata, metadata);
    assert_eq!(info.active_subscriptions, 0);
    assert_eq!(info.request_count, 0);
}

#[test]
fn test_ws_multiplex_manager_list_sessions() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();
    manager.create_session(Some("session-1".to_string()), None);
    manager.create_session(Some("session-2".to_string()), None);
    manager.create_session(Some("session-3".to_string()), None);

    let sessions = manager.list_sessions();
    assert_eq!(sessions.len(), 3);

    let ids: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
    assert!(ids.contains(&"session-1"));
    assert!(ids.contains(&"session-2"));
    assert!(ids.contains(&"session-3"));
}

#[test]
fn test_ws_multiplex_manager_increment_request_count() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();
    let id = manager.create_session(Some("counter-test".to_string()), None);

    {
        let session = manager.get_session_mut(&id).unwrap();
        session.increment_request_count();
        session.increment_request_count();
        session.increment_request_count();
    }

    let session = manager.get_session(&id).unwrap();
    assert_eq!(session.request_count, 3);
}

#[test]
fn test_ws_client_message_create_session_serialization() {
    use crate::transport::WsClientMessage;

    // Without metadata
    let msg = WsClientMessage::CreateSession {
        session_id: Some("my-session".to_string()),
        metadata: None,
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"create_session\""));
    assert!(json.contains("\"session_id\":\"my-session\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::CreateSession {
            session_id,
            metadata,
        } => {
            assert_eq!(session_id, Some("my-session".to_string()));
            assert!(metadata.is_none());
        }
        _ => panic!("Expected CreateSession variant"),
    }

    // With metadata
    let msg_with_meta = WsClientMessage::CreateSession {
        session_id: None,
        metadata: Some(serde_json::json!({"key": "value"})),
    };

    let json_with_meta = serde_json::to_string(&msg_with_meta).unwrap();
    assert!(json_with_meta.contains("\"metadata\""));
    assert!(json_with_meta.contains("\"key\":\"value\""));
}

#[test]
fn test_ws_client_message_destroy_session_serialization() {
    use crate::transport::WsClientMessage;

    let msg = WsClientMessage::DestroySession {
        session_id: "session-to-destroy".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"destroy_session\""));
    assert!(json.contains("\"session_id\":\"session-to-destroy\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::DestroySession { session_id } => {
            assert_eq!(session_id, "session-to-destroy");
        }
        _ => panic!("Expected DestroySession variant"),
    }
}

#[test]
fn test_ws_client_message_list_sessions_serialization() {
    use crate::transport::WsClientMessage;

    let msg = WsClientMessage::ListSessions;

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"list_sessions\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    assert!(matches!(parsed, WsClientMessage::ListSessions));
}

#[test]
fn test_ws_client_message_get_session_info_serialization() {
    use crate::transport::WsClientMessage;

    let msg = WsClientMessage::GetSessionInfo {
        session_id: "query-session".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"get_session_info\""));
    assert!(json.contains("\"session_id\":\"query-session\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::GetSessionInfo { session_id } => {
            assert_eq!(session_id, "query-session");
        }
        _ => panic!("Expected GetSessionInfo variant"),
    }
}

#[test]
fn test_ws_server_message_session_created_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionCreated {
        session_id: "new-session".to_string(),
        metadata: Some(serde_json::json!({"created_by": "test"})),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_created\""));
    assert!(json.contains("\"session_id\":\"new-session\""));
    assert!(json.contains("\"metadata\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionCreated {
            session_id,
            metadata,
        } => {
            assert_eq!(session_id, "new-session");
            assert!(metadata.is_some());
        }
        _ => panic!("Expected SessionCreated variant"),
    }
}

#[test]
fn test_ws_server_message_session_destroyed_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionDestroyed {
        session_id: "destroyed-session".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_destroyed\""));
    assert!(json.contains("\"session_id\":\"destroyed-session\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionDestroyed { session_id } => {
            assert_eq!(session_id, "destroyed-session");
        }
        _ => panic!("Expected SessionDestroyed variant"),
    }
}

#[test]
fn test_ws_server_message_session_destroy_error_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionDestroyError {
        session_id: "bad-session".to_string(),
        message: "Session not found".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_destroy_error\""));
    assert!(json.contains("\"session_id\":\"bad-session\""));
    assert!(json.contains("\"message\":\"Session not found\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionDestroyError {
            session_id,
            message,
        } => {
            assert_eq!(session_id, "bad-session");
            assert_eq!(message, "Session not found");
        }
        _ => panic!("Expected SessionDestroyError variant"),
    }
}

#[test]
fn test_ws_server_message_session_list_serialization() {
    use crate::transport::{MultiplexedSessionInfo, WsServerMessage};

    let sessions = vec![
        MultiplexedSessionInfo {
            session_id: "session-1".to_string(),
            metadata: None,
            active_subscriptions: 0,
            created_at: 1234567890000,
            request_count: 5,
            last_activity: 1234567890000,
            is_expired: false,
        },
        MultiplexedSessionInfo {
            session_id: "session-2".to_string(),
            metadata: Some(serde_json::json!({"tag": "test"})),
            active_subscriptions: 2,
            created_at: 1234567891000,
            request_count: 10,
            last_activity: 1234567891000,
            is_expired: false,
        },
    ];

    let msg = WsServerMessage::SessionList { sessions };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_list\""));
    assert!(json.contains("\"sessions\""));
    assert!(json.contains("\"session-1\""));
    assert!(json.contains("\"session-2\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionList { sessions } => {
            assert_eq!(sessions.len(), 2);
            assert_eq!(sessions[0].session_id, "session-1");
            assert_eq!(sessions[0].request_count, 5);
            assert_eq!(sessions[1].session_id, "session-2");
            assert_eq!(sessions[1].active_subscriptions, 2);
        }
        _ => panic!("Expected SessionList variant"),
    }
}

#[test]
fn test_ws_server_message_session_info_serialization() {
    use crate::transport::{MultiplexedSessionInfo, WsServerMessage};

    let info = MultiplexedSessionInfo {
        session_id: "info-session".to_string(),
        metadata: Some(serde_json::json!({"user": "alice"})),
        active_subscriptions: 3,
        created_at: 1735200000000,
        request_count: 42,
        last_activity: 1735200000000,
        is_expired: false,
    };

    let msg = WsServerMessage::SessionInfo(info);

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_info\""));
    assert!(json.contains("\"session_id\":\"info-session\""));
    assert!(json.contains("\"active_subscriptions\":3"));
    assert!(json.contains("\"request_count\":42"));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionInfo(info) => {
            assert_eq!(info.session_id, "info-session");
            assert_eq!(info.active_subscriptions, 3);
            assert_eq!(info.request_count, 42);
        }
        _ => panic!("Expected SessionInfo variant"),
    }
}

#[test]
fn test_ws_server_message_session_info_error_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionInfoError {
        session_id: "unknown-session".to_string(),
        message: "Session not found".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_info_error\""));
    assert!(json.contains("\"session_id\":\"unknown-session\""));
    assert!(json.contains("\"message\":\"Session not found\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionInfoError {
            session_id,
            message,
        } => {
            assert_eq!(session_id, "unknown-session");
            assert_eq!(message, "Session not found");
        }
        _ => panic!("Expected SessionInfoError variant"),
    }
}

#[test]
fn test_multiplexed_session_info_struct() {
    use crate::transport::MultiplexedSessionInfo;

    let info = MultiplexedSessionInfo {
        session_id: "test-struct".to_string(),
        metadata: None,
        active_subscriptions: 0,
        created_at: 1735200000000,
        request_count: 0,
        last_activity: 1735200000000,
        is_expired: false,
    };

    let json = serde_json::to_string(&info).unwrap();
    // metadata should be skipped when None
    assert!(!json.contains("\"metadata\""));
    assert!(json.contains("\"session_id\":\"test-struct\""));
    assert!(json.contains("\"last_activity\""));

    let parsed: MultiplexedSessionInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.session_id, "test-struct");
    assert!(parsed.metadata.is_none());
    assert!(!parsed.is_expired);
}

// ============================================================================
// Session Timeout and Metadata Tests
// ============================================================================

#[test]
fn test_session_timeout_config_default() {
    use crate::transport::SessionTimeoutConfig;
    use std::time::Duration;

    let config = SessionTimeoutConfig::default();
    assert_eq!(config.idle_timeout, Duration::from_secs(30 * 60)); // 30 minutes
    assert!(config.enabled);
    assert_eq!(config.cleanup_interval, Duration::from_secs(60));
}

#[test]
fn test_session_timeout_config_custom() {
    use crate::transport::SessionTimeoutConfig;
    use std::time::Duration;

    let config = SessionTimeoutConfig::with_idle_timeout_secs(300); // 5 minutes
    assert_eq!(config.idle_timeout, Duration::from_secs(300));
    assert!(config.enabled);

    let disabled = SessionTimeoutConfig::disabled();
    assert!(!disabled.enabled);
}

#[test]
fn test_multiplexed_session_last_activity() {
    use crate::transport::MultiplexedSession;
    use std::time::Duration;

    let session = MultiplexedSession::new("activity-test".to_string(), None);

    // Session starts fresh
    assert!(session.idle_duration() < Duration::from_secs(1));
    assert!(!session.is_expired(Duration::from_secs(60)));
}

#[test]
fn test_multiplexed_session_touch() {
    use crate::transport::MultiplexedSession;

    let mut session = MultiplexedSession::new("touch-test".to_string(), None);

    // Simulate some time passing (we can't actually wait, so we check that touch works)
    let before = session.last_activity;
    session.touch();
    // Touch should update to current time (or later)
    assert!(session.last_activity >= before);
}

#[test]
fn test_multiplexed_session_update_metadata() {
    use crate::transport::MultiplexedSession;

    let mut session = MultiplexedSession::new("metadata-test".to_string(), None);
    assert!(session.metadata.is_none());

    session.update_metadata(Some(serde_json::json!({"user": "test"})));
    assert!(session.metadata.is_some());
    assert_eq!(session.metadata.as_ref().unwrap()["user"], "test");

    session.update_metadata(None);
    assert!(session.metadata.is_none());
}

#[test]
fn test_multiplexed_session_info_with_timeout() {
    use crate::transport::MultiplexedSession;
    use std::time::Duration;

    let session = MultiplexedSession::new("timeout-info-test".to_string(), None);

    // Fresh session should not be expired
    let info = session.info_with_timeout(Duration::from_secs(60));
    assert!(!info.is_expired);
    assert!(info.last_activity > 0);
}

#[test]
fn test_ws_multiplex_manager_with_timeout_config() {
    use crate::transport::{SessionTimeoutConfig, WsMultiplexManager};
    use std::time::Duration;

    let config = SessionTimeoutConfig::with_idle_timeout_secs(120);
    let manager = WsMultiplexManager::with_timeout_config(config);

    assert_eq!(
        manager.timeout_config().idle_timeout,
        Duration::from_secs(120)
    );
}

#[test]
fn test_ws_multiplex_manager_update_metadata() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();
    let id = manager.create_session(Some("meta-update-test".to_string()), None);

    // Update metadata
    assert!(manager.update_session_metadata(&id, Some(serde_json::json!({"key": "value"}))));

    let session = manager.get_session(&id).unwrap();
    assert_eq!(session.metadata.as_ref().unwrap()["key"], "value");

    // Update non-existent session should return false
    assert!(!manager.update_session_metadata("nonexistent", Some(serde_json::json!({}))));
}

#[test]
fn test_ws_multiplex_manager_touch_session() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();
    let id = manager.create_session(Some("touch-manager-test".to_string()), None);

    // Touch should succeed
    assert!(manager.touch_session(&id));

    // Touch non-existent session should fail
    assert!(!manager.touch_session("nonexistent"));
}

#[test]
fn test_ws_multiplex_manager_list_sessions_with_timeout() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();
    manager.create_session(Some("session-1".to_string()), None);
    manager.create_session(Some("session-2".to_string()), None);

    let sessions = manager.list_sessions_with_timeout();
    assert_eq!(sessions.len(), 2);

    // Fresh sessions should not be expired
    for session in &sessions {
        assert!(!session.is_expired);
        assert!(session.last_activity > 0);
    }
}

#[test]
fn test_ws_multiplex_manager_cleanup_expired_sessions_disabled() {
    use crate::transport::{SessionTimeoutConfig, WsMultiplexManager};

    let config = SessionTimeoutConfig::disabled();
    let mut manager = WsMultiplexManager::with_timeout_config(config);
    manager.create_session(Some("session-1".to_string()), None);

    // Cleanup should return empty when disabled
    let expired = manager.cleanup_expired_sessions();
    assert!(expired.is_empty());
    assert_eq!(manager.session_count(), 1);
}

#[test]
fn test_ws_multiplex_manager_get_expired_sessions() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();
    manager.create_session(Some("session-1".to_string()), None);
    manager.create_session(Some("session-2".to_string()), None);

    // Fresh sessions should not be expired (default 30 minute timeout)
    let expired = manager.get_expired_sessions();
    assert!(expired.is_empty());
}

#[test]
fn test_ws_client_message_update_metadata_serialization() {
    use crate::transport::WsClientMessage;

    let msg = WsClientMessage::UpdateSessionMetadata {
        session_id: "update-meta-session".to_string(),
        metadata: Some(serde_json::json!({"project": "dashprove", "version": 2})),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"update_session_metadata\""));
    assert!(json.contains("\"session_id\":\"update-meta-session\""));
    assert!(json.contains("\"project\":\"dashprove\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::UpdateSessionMetadata {
            session_id,
            metadata,
        } => {
            assert_eq!(session_id, "update-meta-session");
            assert!(metadata.is_some());
            assert_eq!(metadata.unwrap()["project"], "dashprove");
        }
        _ => panic!("Expected UpdateSessionMetadata variant"),
    }
}

#[test]
fn test_ws_client_message_touch_session_serialization() {
    use crate::transport::WsClientMessage;

    let msg = WsClientMessage::TouchSession {
        session_id: "keep-alive-session".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"touch_session\""));
    assert!(json.contains("\"session_id\":\"keep-alive-session\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::TouchSession { session_id } => {
            assert_eq!(session_id, "keep-alive-session");
        }
        _ => panic!("Expected TouchSession variant"),
    }
}

#[test]
fn test_ws_server_message_metadata_updated_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionMetadataUpdated {
        session_id: "updated-session".to_string(),
        metadata: Some(serde_json::json!({"status": "active"})),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_metadata_updated\""));
    assert!(json.contains("\"session_id\":\"updated-session\""));
    assert!(json.contains("\"status\":\"active\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionMetadataUpdated {
            session_id,
            metadata,
        } => {
            assert_eq!(session_id, "updated-session");
            assert!(metadata.is_some());
        }
        _ => panic!("Expected SessionMetadataUpdated variant"),
    }
}

#[test]
fn test_ws_server_message_session_touched_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionTouched {
        session_id: "touched-session".to_string(),
        last_activity: 1735200000000,
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_touched\""));
    assert!(json.contains("\"session_id\":\"touched-session\""));
    assert!(json.contains("\"last_activity\":1735200000000"));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionTouched {
            session_id,
            last_activity,
        } => {
            assert_eq!(session_id, "touched-session");
            assert_eq!(last_activity, 1735200000000);
        }
        _ => panic!("Expected SessionTouched variant"),
    }
}

#[test]
fn test_ws_server_message_session_expired_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionExpired {
        session_id: "expired-session".to_string(),
        idle_duration_secs: 1800, // 30 minutes
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_expired\""));
    assert!(json.contains("\"session_id\":\"expired-session\""));
    assert!(json.contains("\"idle_duration_secs\":1800"));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionExpired {
            session_id,
            idle_duration_secs,
        } => {
            assert_eq!(session_id, "expired-session");
            assert_eq!(idle_duration_secs, 1800);
        }
        _ => panic!("Expected SessionExpired variant"),
    }
}

#[test]
fn test_ws_server_message_metadata_update_error_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionMetadataUpdateError {
        session_id: "missing-session".to_string(),
        message: "Session not found".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_metadata_update_error\""));
    assert!(json.contains("\"message\":\"Session not found\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionMetadataUpdateError {
            session_id,
            message,
        } => {
            assert_eq!(session_id, "missing-session");
            assert_eq!(message, "Session not found");
        }
        _ => panic!("Expected SessionMetadataUpdateError variant"),
    }
}

#[test]
fn test_ws_server_message_touch_error_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionTouchError {
        session_id: "nonexistent".to_string(),
        message: "Session not found".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_touch_error\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionTouchError {
            session_id,
            message,
        } => {
            assert_eq!(session_id, "nonexistent");
            assert_eq!(message, "Session not found");
        }
        _ => panic!("Expected SessionTouchError variant"),
    }
}

#[test]
fn test_ws_client_message_session_scoped_json_rpc_serialization() {
    use crate::protocol::JsonRpcRequest;
    use crate::transport::WsClientMessage;

    let request = JsonRpcRequest::new(42i64, "tools/list");
    let msg = WsClientMessage::SessionScopedJsonRpc {
        session_id: "mux-session-1".to_string(),
        request,
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_scoped_json_rpc\""));
    assert!(json.contains("\"session_id\":\"mux-session-1\""));
    assert!(json.contains("\"method\":\"tools/list\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::SessionScopedJsonRpc {
            session_id,
            request,
        } => {
            assert_eq!(session_id, "mux-session-1");
            assert_eq!(request.method, "tools/list");
        }
        _ => panic!("Expected SessionScopedJsonRpc variant"),
    }
}

#[test]
fn test_ws_client_message_session_scoped_json_rpc_with_params() {
    use crate::protocol::JsonRpcRequest;
    use crate::transport::WsClientMessage;

    let request = JsonRpcRequest::new(1i64, "tools/call")
        .with_params(serde_json::json!({"name": "verify", "arguments": {"spec": "test"}}));
    let msg = WsClientMessage::SessionScopedJsonRpc {
        session_id: "my-session".to_string(),
        request,
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_scoped_json_rpc\""));
    assert!(json.contains("\"params\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::SessionScopedJsonRpc {
            session_id,
            request,
        } => {
            assert_eq!(session_id, "my-session");
            assert!(request.params.is_some());
        }
        _ => panic!("Expected SessionScopedJsonRpc variant"),
    }
}

#[test]
fn test_ws_server_message_session_scoped_json_rpc_serialization() {
    use crate::protocol::{JsonRpcResponse, RequestId};
    use crate::transport::WsServerMessage;

    let response =
        JsonRpcResponse::success(RequestId::Number(42), serde_json::json!({"status": "ok"}));
    let msg = WsServerMessage::SessionScopedJsonRpc {
        session_id: "mux-session-1".to_string(),
        response,
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_scoped_json_rpc\""));
    assert!(json.contains("\"session_id\":\"mux-session-1\""));
    assert!(json.contains("\"result\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionScopedJsonRpc {
            session_id,
            response,
        } => {
            assert_eq!(session_id, "mux-session-1");
            assert!(response.result.is_some());
            assert!(response.error.is_none());
        }
        _ => panic!("Expected SessionScopedJsonRpc variant"),
    }
}

#[test]
fn test_ws_server_message_session_scoped_json_rpc_error_response() {
    use crate::error::{ErrorCode, JsonRpcError};
    use crate::protocol::{JsonRpcResponse, RequestId};
    use crate::transport::WsServerMessage;

    let error = JsonRpcError::new(ErrorCode::MethodNotFound, "Unknown method");
    let response = JsonRpcResponse::error(RequestId::Number(99), error);
    let msg = WsServerMessage::SessionScopedJsonRpc {
        session_id: "error-session".to_string(),
        response,
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_scoped_json_rpc\""));
    assert!(json.contains("\"error\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionScopedJsonRpc {
            session_id,
            response,
        } => {
            assert_eq!(session_id, "error-session");
            assert!(response.result.is_none());
            assert!(response.error.is_some());
        }
        _ => panic!("Expected SessionScopedJsonRpc variant"),
    }
}

#[test]
fn test_ws_server_message_session_scoped_json_rpc_error_serialization() {
    use crate::protocol::RequestId;
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionScopedJsonRpcError {
        session_id: "nonexistent-session".to_string(),
        request_id: RequestId::Number(123),
        message: "Session not found".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_scoped_json_rpc_error\""));
    assert!(json.contains("\"session_id\":\"nonexistent-session\""));
    assert!(json.contains("\"request_id\":123"));
    assert!(json.contains("\"message\":\"Session not found\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionScopedJsonRpcError {
            session_id,
            request_id,
            message,
        } => {
            assert_eq!(session_id, "nonexistent-session");
            assert_eq!(request_id, RequestId::Number(123));
            assert_eq!(message, "Session not found");
        }
        _ => panic!("Expected SessionScopedJsonRpcError variant"),
    }
}

#[test]
fn test_ws_server_message_session_scoped_json_rpc_error_string_id() {
    use crate::protocol::RequestId;
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionScopedJsonRpcError {
        session_id: "missing".to_string(),
        request_id: RequestId::String("req-abc-123".to_string()),
        message: "Session does not exist".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"request_id\":\"req-abc-123\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionScopedJsonRpcError { request_id, .. } => {
            assert_eq!(request_id, RequestId::String("req-abc-123".to_string()));
        }
        _ => panic!("Expected SessionScopedJsonRpcError variant"),
    }
}

// Tests for automatic session cleanup functionality

#[test]
fn test_get_expired_sessions_with_idle_empty_when_disabled() {
    use crate::transport::{SessionTimeoutConfig, WsMultiplexManager};

    let mut manager = WsMultiplexManager::with_timeout_config(SessionTimeoutConfig::disabled());
    manager.create_session(Some("test-session".to_string()), None);

    // Even with sessions, should return empty when disabled
    let expired = manager.get_expired_sessions_with_idle();
    assert!(expired.is_empty());
}

#[test]
fn test_get_expired_sessions_with_idle_returns_idle_duration() {
    use crate::transport::{SessionTimeoutConfig, WsMultiplexManager};
    use std::time::Duration;

    // Create manager with very short timeout (1ms)
    let config = SessionTimeoutConfig {
        idle_timeout: Duration::from_millis(1),
        enabled: true,
        cleanup_interval: Duration::from_secs(60),
    };
    let mut manager = WsMultiplexManager::with_timeout_config(config);
    manager.create_session(Some("expired-session".to_string()), None);

    // Wait for session to expire
    std::thread::sleep(Duration::from_millis(10));

    let expired = manager.get_expired_sessions_with_idle();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].0, "expired-session");
    // Verify idle duration is present (u64 is always >= 0)
    let _ = expired[0].1; // Just verify access works
}

#[test]
fn test_get_expired_sessions_with_idle_returns_multiple() {
    use crate::transport::{SessionTimeoutConfig, WsMultiplexManager};
    use std::time::Duration;

    // Create manager with very short timeout
    let config = SessionTimeoutConfig {
        idle_timeout: Duration::from_millis(1),
        enabled: true,
        cleanup_interval: Duration::from_secs(60),
    };
    let mut manager = WsMultiplexManager::with_timeout_config(config);
    manager.create_session(Some("session-1".to_string()), None);
    manager.create_session(Some("session-2".to_string()), None);
    manager.create_session(Some("session-3".to_string()), None);

    // Wait for sessions to expire
    std::thread::sleep(Duration::from_millis(10));

    let expired = manager.get_expired_sessions_with_idle();
    assert_eq!(expired.len(), 3);

    let session_ids: Vec<&str> = expired.iter().map(|(id, _)| id.as_str()).collect();
    assert!(session_ids.contains(&"session-1"));
    assert!(session_ids.contains(&"session-2"));
    assert!(session_ids.contains(&"session-3"));
}

#[test]
fn test_get_expired_sessions_with_idle_excludes_active() {
    use crate::transport::{SessionTimeoutConfig, WsMultiplexManager};
    use std::time::Duration;

    // Create manager with timeout of 100ms
    let config = SessionTimeoutConfig {
        idle_timeout: Duration::from_millis(100),
        enabled: true,
        cleanup_interval: Duration::from_secs(60),
    };
    let mut manager = WsMultiplexManager::with_timeout_config(config);
    manager.create_session(Some("session-1".to_string()), None);
    manager.create_session(Some("session-2".to_string()), None);

    // Wait 60ms (less than 100ms timeout)
    std::thread::sleep(Duration::from_millis(60));

    // Touch session-1 to keep it active (session-2 has been idle 60ms)
    manager.touch_session("session-1");

    // Wait another 50ms - session-2 is now idle 110ms (expired), session-1 is idle 50ms (not expired)
    std::thread::sleep(Duration::from_millis(50));

    let expired = manager.get_expired_sessions_with_idle();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].0, "session-2");
}

#[test]
fn test_cleanup_expired_sessions_returns_removed_ids() {
    use crate::transport::{SessionTimeoutConfig, WsMultiplexManager};
    use std::time::Duration;

    let config = SessionTimeoutConfig {
        idle_timeout: Duration::from_millis(1),
        enabled: true,
        cleanup_interval: Duration::from_secs(60),
    };
    let mut manager = WsMultiplexManager::with_timeout_config(config);
    manager.create_session(Some("cleanup-session".to_string()), None);

    std::thread::sleep(Duration::from_millis(10));

    let removed = manager.cleanup_expired_sessions();
    assert_eq!(removed.len(), 1);
    assert_eq!(removed[0], "cleanup-session");

    // Session should be gone
    assert!(!manager.has_session("cleanup-session"));
}

#[test]
fn test_session_expired_message_roundtrip() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionExpired {
        session_id: "expired-session-123".to_string(),
        idle_duration_secs: 1800, // 30 minutes
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"session_expired\""));
    assert!(json.contains("\"session_id\":\"expired-session-123\""));
    assert!(json.contains("\"idle_duration_secs\":1800"));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::SessionExpired {
            session_id,
            idle_duration_secs,
        } => {
            assert_eq!(session_id, "expired-session-123");
            assert_eq!(idle_duration_secs, 1800);
        }
        _ => panic!("Expected SessionExpired variant"),
    }
}

#[test]
fn test_session_expired_message_from_json() {
    use crate::transport::WsServerMessage;

    let json =
        r#"{"type":"session_expired","session_id":"test-session","idle_duration_secs":3600}"#;

    let msg: WsServerMessage = serde_json::from_str(json).unwrap();
    match msg {
        WsServerMessage::SessionExpired {
            session_id,
            idle_duration_secs,
        } => {
            assert_eq!(session_id, "test-session");
            assert_eq!(idle_duration_secs, 3600);
        }
        _ => panic!("Expected SessionExpired variant"),
    }
}

#[test]
fn test_session_timeout_config_cleanup_interval() {
    use crate::transport::SessionTimeoutConfig;
    use std::time::Duration;

    let default_config = SessionTimeoutConfig::default();
    assert_eq!(default_config.cleanup_interval, Duration::from_secs(60));

    let custom_config = SessionTimeoutConfig {
        idle_timeout: Duration::from_secs(600),
        enabled: true,
        cleanup_interval: Duration::from_secs(30),
    };
    assert_eq!(custom_config.cleanup_interval, Duration::from_secs(30));
}

// ============================================================================
// Batch Session-Scoped JSON-RPC Tests
// ============================================================================

#[test]
fn test_ws_client_message_batch_session_scoped_json_rpc_serialization() {
    use crate::protocol::JsonRpcRequest;
    use crate::transport::WsClientMessage;

    let requests = vec![
        JsonRpcRequest::new(1i64, "tools/list"),
        JsonRpcRequest::new(2i64, "resources/list"),
        JsonRpcRequest::new(3i64, "prompts/list"),
    ];
    let msg = WsClientMessage::BatchSessionScopedJsonRpc {
        session_id: "batch-session-1".to_string(),
        requests,
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"batch_session_scoped_json_rpc\""));
    assert!(json.contains("\"session_id\":\"batch-session-1\""));
    assert!(json.contains("\"requests\""));
    assert!(json.contains("\"tools/list\""));
    assert!(json.contains("\"resources/list\""));
    assert!(json.contains("\"prompts/list\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::BatchSessionScopedJsonRpc {
            session_id,
            requests,
        } => {
            assert_eq!(session_id, "batch-session-1");
            assert_eq!(requests.len(), 3);
            assert_eq!(requests[0].method, "tools/list");
            assert_eq!(requests[1].method, "resources/list");
            assert_eq!(requests[2].method, "prompts/list");
        }
        _ => panic!("Expected BatchSessionScopedJsonRpc variant"),
    }
}

#[test]
fn test_ws_client_message_batch_session_scoped_json_rpc_empty() {
    use crate::transport::WsClientMessage;

    let msg = WsClientMessage::BatchSessionScopedJsonRpc {
        session_id: "empty-batch".to_string(),
        requests: vec![],
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"batch_session_scoped_json_rpc\""));
    assert!(json.contains("\"requests\":[]"));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::BatchSessionScopedJsonRpc {
            session_id,
            requests,
        } => {
            assert_eq!(session_id, "empty-batch");
            assert!(requests.is_empty());
        }
        _ => panic!("Expected BatchSessionScopedJsonRpc variant"),
    }
}

#[test]
fn test_ws_client_message_batch_session_scoped_json_rpc_with_params() {
    use crate::protocol::JsonRpcRequest;
    use crate::transport::WsClientMessage;

    let requests = vec![
        JsonRpcRequest::new(1i64, "tools/call")
            .with_params(serde_json::json!({"name": "verify", "arguments": {"spec": "test1"}})),
        JsonRpcRequest::new(2i64, "tools/call")
            .with_params(serde_json::json!({"name": "compile", "arguments": {"target": "lean4"}})),
    ];
    let msg = WsClientMessage::BatchSessionScopedJsonRpc {
        session_id: "params-batch".to_string(),
        requests,
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"batch_session_scoped_json_rpc\""));
    assert!(json.contains("\"params\""));

    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsClientMessage::BatchSessionScopedJsonRpc {
            session_id,
            requests,
        } => {
            assert_eq!(session_id, "params-batch");
            assert_eq!(requests.len(), 2);
            assert!(requests[0].params.is_some());
            assert!(requests[1].params.is_some());
        }
        _ => panic!("Expected BatchSessionScopedJsonRpc variant"),
    }
}

#[test]
fn test_ws_client_message_batch_session_scoped_json_rpc_from_json() {
    use crate::transport::WsClientMessage;

    let json = r#"{
        "type": "batch_session_scoped_json_rpc",
        "session_id": "my-session",
        "requests": [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 2, "method": "resources/list"}
        ]
    }"#;

    let msg: WsClientMessage = serde_json::from_str(json).unwrap();
    match msg {
        WsClientMessage::BatchSessionScopedJsonRpc {
            session_id,
            requests,
        } => {
            assert_eq!(session_id, "my-session");
            assert_eq!(requests.len(), 2);
            assert_eq!(requests[0].method, "tools/list");
            assert_eq!(requests[1].method, "resources/list");
        }
        _ => panic!("Expected BatchSessionScopedJsonRpc variant"),
    }
}

#[test]
fn test_ws_server_message_batch_session_scoped_json_rpc_serialization() {
    use crate::protocol::{JsonRpcResponse, RequestId};
    use crate::transport::WsServerMessage;

    let responses = vec![
        JsonRpcResponse::success(RequestId::Number(1), serde_json::json!({"tools": []})),
        JsonRpcResponse::success(RequestId::Number(2), serde_json::json!({"resources": []})),
        JsonRpcResponse::success(RequestId::Number(3), serde_json::json!({"prompts": []})),
    ];
    let msg = WsServerMessage::BatchSessionScopedJsonRpc {
        session_id: "batch-session-1".to_string(),
        responses,
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"batch_session_scoped_json_rpc\""));
    assert!(json.contains("\"session_id\":\"batch-session-1\""));
    assert!(json.contains("\"responses\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::BatchSessionScopedJsonRpc {
            session_id,
            responses,
        } => {
            assert_eq!(session_id, "batch-session-1");
            assert_eq!(responses.len(), 3);
            assert!(responses[0].result.is_some());
            assert!(responses[1].result.is_some());
            assert!(responses[2].result.is_some());
        }
        _ => panic!("Expected BatchSessionScopedJsonRpc variant"),
    }
}

#[test]
fn test_ws_server_message_batch_session_scoped_json_rpc_empty() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::BatchSessionScopedJsonRpc {
        session_id: "empty-batch".to_string(),
        responses: vec![],
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"batch_session_scoped_json_rpc\""));
    assert!(json.contains("\"responses\":[]"));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::BatchSessionScopedJsonRpc {
            session_id,
            responses,
        } => {
            assert_eq!(session_id, "empty-batch");
            assert!(responses.is_empty());
        }
        _ => panic!("Expected BatchSessionScopedJsonRpc variant"),
    }
}

#[test]
fn test_ws_server_message_batch_session_scoped_json_rpc_mixed_success_error() {
    use crate::error::{ErrorCode, JsonRpcError};
    use crate::protocol::{JsonRpcResponse, RequestId};
    use crate::transport::WsServerMessage;

    let responses = vec![
        JsonRpcResponse::success(RequestId::Number(1), serde_json::json!({"tools": []})),
        JsonRpcResponse::error(
            RequestId::Number(2),
            JsonRpcError::new(ErrorCode::MethodNotFound, "Unknown method"),
        ),
        JsonRpcResponse::success(RequestId::Number(3), serde_json::json!({"prompts": []})),
    ];
    let msg = WsServerMessage::BatchSessionScopedJsonRpc {
        session_id: "mixed-batch".to_string(),
        responses,
    };

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::BatchSessionScopedJsonRpc {
            session_id,
            responses,
        } => {
            assert_eq!(session_id, "mixed-batch");
            assert_eq!(responses.len(), 3);
            assert!(responses[0].result.is_some());
            assert!(responses[0].error.is_none());
            assert!(responses[1].result.is_none());
            assert!(responses[1].error.is_some());
            assert!(responses[2].result.is_some());
            assert!(responses[2].error.is_none());
        }
        _ => panic!("Expected BatchSessionScopedJsonRpc variant"),
    }
}

#[test]
fn test_ws_server_message_batch_session_scoped_json_rpc_error_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::BatchSessionScopedJsonRpcError {
        session_id: "nonexistent-session".to_string(),
        message: "Session not found".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"batch_session_scoped_json_rpc_error\""));
    assert!(json.contains("\"session_id\":\"nonexistent-session\""));
    assert!(json.contains("\"message\":\"Session not found\""));

    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        WsServerMessage::BatchSessionScopedJsonRpcError {
            session_id,
            message,
        } => {
            assert_eq!(session_id, "nonexistent-session");
            assert_eq!(message, "Session not found");
        }
        _ => panic!("Expected BatchSessionScopedJsonRpcError variant"),
    }
}

#[test]
fn test_ws_server_message_batch_session_scoped_json_rpc_error_from_json() {
    use crate::transport::WsServerMessage;

    let json = r#"{"type":"batch_session_scoped_json_rpc_error","session_id":"test-session","message":"Session expired"}"#;

    let msg: WsServerMessage = serde_json::from_str(json).unwrap();
    match msg {
        WsServerMessage::BatchSessionScopedJsonRpcError {
            session_id,
            message,
        } => {
            assert_eq!(session_id, "test-session");
            assert_eq!(message, "Session expired");
        }
        _ => panic!("Expected BatchSessionScopedJsonRpcError variant"),
    }
}

#[test]
fn test_ws_client_message_batch_preserves_request_order() {
    use crate::protocol::JsonRpcRequest;
    use crate::transport::WsClientMessage;

    // Create requests with specific IDs to verify order preservation
    let requests: Vec<JsonRpcRequest> = (1..=10)
        .map(|i| JsonRpcRequest::new(i as i64, format!("method_{}", i)))
        .collect();

    let msg = WsClientMessage::BatchSessionScopedJsonRpc {
        session_id: "order-test".to_string(),
        requests,
    };

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();

    match parsed {
        WsClientMessage::BatchSessionScopedJsonRpc { requests, .. } => {
            for (i, request) in requests.iter().enumerate() {
                let expected_id = (i + 1) as i64;
                match &request.id {
                    crate::protocol::RequestId::Number(n) => assert_eq!(*n, expected_id),
                    _ => panic!("Expected Number request ID"),
                }
                assert_eq!(request.method, format!("method_{}", i + 1));
            }
        }
        _ => panic!("Expected BatchSessionScopedJsonRpc variant"),
    }
}

#[test]
fn test_ws_server_message_batch_preserves_response_order() {
    use crate::protocol::{JsonRpcResponse, RequestId};
    use crate::transport::WsServerMessage;

    // Create responses with specific IDs to verify order preservation
    let responses: Vec<JsonRpcResponse> = (1..=10)
        .map(|i| JsonRpcResponse::success(RequestId::Number(i), serde_json::json!({"index": i})))
        .collect();

    let msg = WsServerMessage::BatchSessionScopedJsonRpc {
        session_id: "order-test".to_string(),
        responses,
    };

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();

    match parsed {
        WsServerMessage::BatchSessionScopedJsonRpc { responses, .. } => {
            for (i, response) in responses.iter().enumerate() {
                let expected_id = (i + 1) as i64;
                match &response.id {
                    crate::protocol::RequestId::Number(n) => assert_eq!(*n, expected_id),
                    _ => panic!("Expected Number request ID"),
                }
                // Verify the result contains the expected index
                let result = response.result.as_ref().unwrap();
                assert_eq!(result["index"], expected_id);
            }
        }
        _ => panic!("Expected BatchSessionScopedJsonRpc variant"),
    }
}

// ============================================================================
// Connection Metrics Tests
// ============================================================================

#[test]
fn test_connection_metrics_default() {
    use crate::transport::ConnectionMetrics;

    let metrics = ConnectionMetrics::default();
    assert_eq!(metrics.sessions_created, 0);
    assert_eq!(metrics.sessions_destroyed, 0);
    assert_eq!(metrics.sessions_expired, 0);
    assert_eq!(metrics.active_sessions, 0);
    assert_eq!(metrics.total_requests, 0);
    assert_eq!(metrics.uptime_ms, 0);
    assert!(metrics.avg_session_lifetime_ms.is_none());
    assert_eq!(metrics.subscriptions_created, 0);
    assert_eq!(metrics.subscriptions_removed, 0);
    assert_eq!(metrics.active_subscriptions, 0);
}

#[test]
fn test_connection_metrics_serialization() {
    use crate::transport::ConnectionMetrics;

    let metrics = ConnectionMetrics {
        sessions_created: 5,
        sessions_destroyed: 2,
        sessions_expired: 1,
        active_sessions: 2,
        total_requests: 100,
        uptime_ms: 60000,
        avg_session_lifetime_ms: Some(30000),
        subscriptions_created: 10,
        subscriptions_removed: 5,
        active_subscriptions: 5,
    };

    let json = serde_json::to_string(&metrics).unwrap();
    let parsed: ConnectionMetrics = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.sessions_created, 5);
    assert_eq!(parsed.sessions_destroyed, 2);
    assert_eq!(parsed.sessions_expired, 1);
    assert_eq!(parsed.active_sessions, 2);
    assert_eq!(parsed.total_requests, 100);
    assert_eq!(parsed.uptime_ms, 60000);
    assert_eq!(parsed.avg_session_lifetime_ms, Some(30000));
    assert_eq!(parsed.subscriptions_created, 10);
    assert_eq!(parsed.subscriptions_removed, 5);
    assert_eq!(parsed.active_subscriptions, 5);
}

#[test]
fn test_ws_multiplex_manager_connection_metrics() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();

    // Initial metrics
    let metrics = manager.connection_metrics();
    assert_eq!(metrics.sessions_created, 0);
    assert_eq!(metrics.active_sessions, 0);
    // uptime_ms can be 0 if test runs fast, just check it's a valid number (not panicking)
    let _uptime = metrics.uptime_ms;

    // Create some sessions
    let session1 = manager.create_session(Some("s1".to_string()), None);
    let _session2 = manager.create_session(Some("s2".to_string()), None);

    let metrics = manager.connection_metrics();
    assert_eq!(metrics.sessions_created, 2);
    assert_eq!(metrics.active_sessions, 2);

    // Destroy a session
    manager.destroy_session(&session1);

    let metrics = manager.connection_metrics();
    assert_eq!(metrics.sessions_created, 2);
    assert_eq!(metrics.sessions_destroyed, 1);
    assert_eq!(metrics.active_sessions, 1);
    assert!(metrics.avg_session_lifetime_ms.is_some());

    // Create another and destroy it
    let session3 = manager.create_session(Some("s3".to_string()), None);
    manager.destroy_session(&session3);

    let metrics = manager.connection_metrics();
    assert_eq!(metrics.sessions_created, 3);
    assert_eq!(metrics.sessions_destroyed, 2);
    assert_eq!(metrics.active_sessions, 1);
}

#[test]
fn test_ws_multiplex_manager_connection_id() {
    use crate::transport::WsMultiplexManager;

    let manager1 = WsMultiplexManager::new();
    let manager2 = WsMultiplexManager::new();

    // Each manager should have a unique connection ID
    assert!(!manager1.connection_id().is_empty());
    assert!(manager1.connection_id().starts_with("conn-"));
    assert_ne!(manager1.connection_id(), manager2.connection_id());
}

#[test]
fn test_ws_multiplex_manager_tracks_requests() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();
    let session_id = manager.create_session(Some("test".to_string()), None);

    // Simulate requests
    if let Some(session) = manager.get_session_mut(&session_id) {
        session.increment_request_count();
        session.increment_request_count();
        session.increment_request_count();
    }

    let metrics = manager.connection_metrics();
    assert_eq!(metrics.total_requests, 3);
}

#[test]
fn test_ws_multiplex_manager_track_subscriptions() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();

    manager.track_subscription_created();
    manager.track_subscription_created();
    manager.track_subscription_removed();

    let metrics = manager.connection_metrics();
    assert_eq!(metrics.subscriptions_created, 2);
    assert_eq!(metrics.subscriptions_removed, 1);
}

// ============================================================================
// Session Recovery Tests
// ============================================================================

#[test]
fn test_session_recovery_config_default() {
    use crate::transport::SessionRecoveryConfig;

    let config = SessionRecoveryConfig::default();
    assert!(config.enabled);
    assert_eq!(config.token_ttl_secs, 300);
}

#[test]
fn test_session_recovery_config_disabled() {
    use crate::transport::SessionRecoveryConfig;

    let config = SessionRecoveryConfig::disabled();
    assert!(!config.enabled);
}

#[test]
fn test_session_recovery_config_custom_ttl() {
    use crate::transport::SessionRecoveryConfig;

    let config = SessionRecoveryConfig::with_ttl_secs(600);
    assert!(config.enabled);
    assert_eq!(config.token_ttl_secs, 600);
}

#[test]
fn test_recovery_token_serialization() {
    use crate::transport::{RecoverableSession, SessionRecoveryToken};

    let token = SessionRecoveryToken {
        connection_id: "conn-123".to_string(),
        sessions: vec![
            RecoverableSession {
                session_id: "s1".to_string(),
                metadata: Some(serde_json::json!({"key": "value"})),
                request_count: 10,
            },
            RecoverableSession {
                session_id: "s2".to_string(),
                metadata: None,
                request_count: 5,
            },
        ],
        created_at: 1000000,
        expires_at: 1300000,
    };

    let json = serde_json::to_string(&token).unwrap();
    let parsed: SessionRecoveryToken = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.connection_id, "conn-123");
    assert_eq!(parsed.sessions.len(), 2);
    assert_eq!(parsed.sessions[0].session_id, "s1");
    assert_eq!(parsed.sessions[0].request_count, 10);
    assert_eq!(parsed.sessions[1].session_id, "s2");
    assert!(parsed.sessions[1].metadata.is_none());
    assert_eq!(parsed.created_at, 1000000);
    assert_eq!(parsed.expires_at, 1300000);
}

#[test]
fn test_generate_recovery_token_with_sessions() {
    use crate::transport::WsMultiplexManager;

    let mut manager = WsMultiplexManager::new();

    // No sessions - no token
    assert!(manager.generate_recovery_token().is_none());

    // Create sessions
    manager.create_session(
        Some("s1".to_string()),
        Some(serde_json::json!({"name": "session1"})),
    );
    manager.create_session(Some("s2".to_string()), None);

    // Simulate some requests
    if let Some(session) = manager.get_session_mut("s1") {
        session.increment_request_count();
        session.increment_request_count();
    }

    let token = manager.generate_recovery_token().unwrap();
    assert_eq!(token.connection_id, manager.connection_id());
    assert_eq!(token.sessions.len(), 2);

    // Find s1 in token
    let s1 = token
        .sessions
        .iter()
        .find(|s| s.session_id == "s1")
        .unwrap();
    assert_eq!(s1.request_count, 2);
    assert!(s1.metadata.is_some());

    let s2 = token
        .sessions
        .iter()
        .find(|s| s.session_id == "s2")
        .unwrap();
    assert_eq!(s2.request_count, 0);
    assert!(s2.metadata.is_none());
}

#[test]
fn test_generate_recovery_token_disabled() {
    use crate::transport::{SessionRecoveryConfig, WsMultiplexManager};

    let mut manager = WsMultiplexManager::new();
    manager.set_recovery_config(SessionRecoveryConfig::disabled());

    manager.create_session(Some("s1".to_string()), None);

    // Should return None when disabled
    assert!(manager.generate_recovery_token().is_none());
}

#[test]
fn test_recover_sessions_success() {
    use crate::transport::{RecoverableSession, SessionRecoveryToken, WsMultiplexManager};
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let token = SessionRecoveryToken {
        connection_id: "old-conn".to_string(),
        sessions: vec![
            RecoverableSession {
                session_id: "recovered-1".to_string(),
                metadata: Some(serde_json::json!({"recovered": true})),
                request_count: 42,
            },
            RecoverableSession {
                session_id: "recovered-2".to_string(),
                metadata: None,
                request_count: 10,
            },
        ],
        created_at: now,
        expires_at: now + 300000, // 5 minutes from now
    };

    let mut manager = WsMultiplexManager::new();
    let recovered = manager.recover_sessions(&token).unwrap();

    assert_eq!(recovered.len(), 2);
    assert!(recovered.contains(&"recovered-1".to_string()));
    assert!(recovered.contains(&"recovered-2".to_string()));

    // Verify sessions exist
    assert!(manager.has_session("recovered-1"));
    assert!(manager.has_session("recovered-2"));

    // Verify request counts were restored
    let s1 = manager.get_session("recovered-1").unwrap();
    assert_eq!(s1.request_count, 42);

    let s2 = manager.get_session("recovered-2").unwrap();
    assert_eq!(s2.request_count, 10);

    // Metrics should show sessions created
    let metrics = manager.connection_metrics();
    assert_eq!(metrics.sessions_created, 2);
}

#[test]
fn test_recover_sessions_expired_token() {
    use crate::transport::{RecoverableSession, SessionRecoveryToken, WsMultiplexManager};
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let token = SessionRecoveryToken {
        connection_id: "old-conn".to_string(),
        sessions: vec![RecoverableSession {
            session_id: "expired".to_string(),
            metadata: None,
            request_count: 0,
        }],
        created_at: now - 600000, // 10 minutes ago
        expires_at: now - 300000, // Expired 5 minutes ago
    };

    let mut manager = WsMultiplexManager::new();
    let result = manager.recover_sessions(&token);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("expired"));
}

#[test]
fn test_recover_sessions_disabled() {
    use crate::transport::{
        RecoverableSession, SessionRecoveryConfig, SessionRecoveryToken, WsMultiplexManager,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let token = SessionRecoveryToken {
        connection_id: "old-conn".to_string(),
        sessions: vec![RecoverableSession {
            session_id: "test".to_string(),
            metadata: None,
            request_count: 0,
        }],
        created_at: now,
        expires_at: now + 300000,
    };

    let mut manager = WsMultiplexManager::new();
    manager.set_recovery_config(SessionRecoveryConfig::disabled());

    let result = manager.recover_sessions(&token);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("disabled"));
}

#[test]
fn test_recover_sessions_skips_existing() {
    use crate::transport::{RecoverableSession, SessionRecoveryToken, WsMultiplexManager};
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let mut manager = WsMultiplexManager::new();

    // Create an existing session
    manager.create_session(Some("existing".to_string()), None);

    let token = SessionRecoveryToken {
        connection_id: "old-conn".to_string(),
        sessions: vec![
            RecoverableSession {
                session_id: "existing".to_string(), // Already exists
                metadata: None,
                request_count: 100,
            },
            RecoverableSession {
                session_id: "new".to_string(),
                metadata: None,
                request_count: 50,
            },
        ],
        created_at: now,
        expires_at: now + 300000,
    };

    let recovered = manager.recover_sessions(&token).unwrap();

    // Only "new" should be recovered
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0], "new");

    // Existing session should not have its request count changed
    let existing = manager.get_session("existing").unwrap();
    assert_eq!(existing.request_count, 0);

    // New session should have restored request count
    let new = manager.get_session("new").unwrap();
    assert_eq!(new.request_count, 50);
}

// ============================================================================
// WebSocket Message Tests for New Types
// ============================================================================

#[test]
fn test_ws_client_message_get_connection_metrics_serialization() {
    use crate::transport::WsClientMessage;

    let msg = WsClientMessage::GetConnectionMetrics;
    let json = serde_json::to_string(&msg).unwrap();
    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();

    assert!(matches!(parsed, WsClientMessage::GetConnectionMetrics));
}

#[test]
fn test_ws_client_message_generate_recovery_token_serialization() {
    use crate::transport::WsClientMessage;

    let msg = WsClientMessage::GenerateRecoveryToken;
    let json = serde_json::to_string(&msg).unwrap();
    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();

    assert!(matches!(parsed, WsClientMessage::GenerateRecoveryToken));
}

#[test]
fn test_ws_client_message_recover_sessions_serialization() {
    use crate::transport::{RecoverableSession, SessionRecoveryToken, WsClientMessage};

    let token = SessionRecoveryToken {
        connection_id: "conn-xyz".to_string(),
        sessions: vec![RecoverableSession {
            session_id: "s1".to_string(),
            metadata: Some(serde_json::json!({"test": true})),
            request_count: 5,
        }],
        created_at: 1000,
        expires_at: 2000,
    };

    let msg = WsClientMessage::RecoverSessions { token };
    let json = serde_json::to_string(&msg).unwrap();
    let parsed: WsClientMessage = serde_json::from_str(&json).unwrap();

    match parsed {
        WsClientMessage::RecoverSessions { token } => {
            assert_eq!(token.connection_id, "conn-xyz");
            assert_eq!(token.sessions.len(), 1);
            assert_eq!(token.sessions[0].session_id, "s1");
        }
        _ => panic!("Expected RecoverSessions variant"),
    }
}

#[test]
fn test_ws_server_message_connection_metrics_response_serialization() {
    use crate::transport::{ConnectionMetrics, WsServerMessage};

    let metrics = ConnectionMetrics {
        sessions_created: 3,
        sessions_destroyed: 1,
        sessions_expired: 0,
        active_sessions: 2,
        total_requests: 50,
        uptime_ms: 120000,
        avg_session_lifetime_ms: Some(60000),
        subscriptions_created: 5,
        subscriptions_removed: 2,
        active_subscriptions: 3,
    };

    let msg = WsServerMessage::ConnectionMetricsResponse {
        connection_id: "conn-abc".to_string(),
        metrics,
    };

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();

    match parsed {
        WsServerMessage::ConnectionMetricsResponse {
            connection_id,
            metrics,
        } => {
            assert_eq!(connection_id, "conn-abc");
            assert_eq!(metrics.sessions_created, 3);
            assert_eq!(metrics.total_requests, 50);
            assert_eq!(metrics.avg_session_lifetime_ms, Some(60000));
        }
        _ => panic!("Expected ConnectionMetricsResponse variant"),
    }
}

#[test]
fn test_ws_server_message_recovery_token_generated_serialization() {
    use crate::transport::{RecoverableSession, SessionRecoveryToken, WsServerMessage};

    let token = SessionRecoveryToken {
        connection_id: "conn-123".to_string(),
        sessions: vec![RecoverableSession {
            session_id: "s1".to_string(),
            metadata: None,
            request_count: 10,
        }],
        created_at: 5000,
        expires_at: 6000,
    };

    let msg = WsServerMessage::RecoveryTokenGenerated { token };
    let json = serde_json::to_string(&msg).unwrap();
    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();

    match parsed {
        WsServerMessage::RecoveryTokenGenerated { token } => {
            assert_eq!(token.connection_id, "conn-123");
            assert_eq!(token.sessions.len(), 1);
        }
        _ => panic!("Expected RecoveryTokenGenerated variant"),
    }
}

#[test]
fn test_ws_server_message_recovery_token_unavailable_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::RecoveryTokenUnavailable {
        message: "No sessions to recover".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();

    match parsed {
        WsServerMessage::RecoveryTokenUnavailable { message } => {
            assert_eq!(message, "No sessions to recover");
        }
        _ => panic!("Expected RecoveryTokenUnavailable variant"),
    }
}

#[test]
fn test_ws_server_message_sessions_recovered_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionsRecovered {
        recovered_sessions: vec!["s1".to_string(), "s2".to_string()],
        original_connection_id: "old-conn".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();

    match parsed {
        WsServerMessage::SessionsRecovered {
            recovered_sessions,
            original_connection_id,
        } => {
            assert_eq!(recovered_sessions.len(), 2);
            assert!(recovered_sessions.contains(&"s1".to_string()));
            assert_eq!(original_connection_id, "old-conn");
        }
        _ => panic!("Expected SessionsRecovered variant"),
    }
}

#[test]
fn test_ws_server_message_session_recovery_error_serialization() {
    use crate::transport::WsServerMessage;

    let msg = WsServerMessage::SessionRecoveryError {
        message: "Token expired".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: WsServerMessage = serde_json::from_str(&json).unwrap();

    match parsed {
        WsServerMessage::SessionRecoveryError { message } => {
            assert_eq!(message, "Token expired");
        }
        _ => panic!("Expected SessionRecoveryError variant"),
    }
}

#[test]
fn test_ws_multiplex_manager_with_configs() {
    use crate::transport::{SessionRecoveryConfig, SessionTimeoutConfig, WsMultiplexManager};
    use std::time::Duration;

    let timeout_config = SessionTimeoutConfig {
        idle_timeout: Duration::from_secs(600),
        enabled: true,
        cleanup_interval: Duration::from_secs(30),
    };

    let recovery_config = SessionRecoveryConfig::with_ttl_secs(120);

    let manager = WsMultiplexManager::with_configs(timeout_config, recovery_config);

    assert_eq!(
        manager.timeout_config().idle_timeout,
        Duration::from_secs(600)
    );
    assert!(manager.recovery_config().enabled);
    assert_eq!(manager.recovery_config().token_ttl_secs, 120);
}

#[test]
fn test_connection_metrics_expired_sessions() {
    use crate::transport::{SessionTimeoutConfig, WsMultiplexManager};
    use std::time::Duration;

    let config = SessionTimeoutConfig {
        idle_timeout: Duration::from_millis(1), // Very short timeout
        enabled: true,
        cleanup_interval: Duration::from_secs(60),
    };

    let mut manager = WsMultiplexManager::with_timeout_config(config);

    // Create sessions
    manager.create_session(Some("s1".to_string()), None);
    manager.create_session(Some("s2".to_string()), None);

    // Wait for sessions to expire
    std::thread::sleep(Duration::from_millis(10));

    // Cleanup expired sessions
    let expired = manager.cleanup_expired_sessions();
    assert_eq!(expired.len(), 2);

    let metrics = manager.connection_metrics();
    assert_eq!(metrics.sessions_created, 2);
    assert_eq!(metrics.sessions_expired, 2);
    assert_eq!(metrics.sessions_destroyed, 0);
    assert_eq!(metrics.active_sessions, 0);
    assert!(metrics.avg_session_lifetime_ms.is_some());
}
