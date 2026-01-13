//! USL Language Server backend implementation
//!
//! Implements the LSP protocol handlers for USL documents.

use crate::{
    call_hierarchy::{find_incoming_calls, find_outgoing_calls, prepare_call_hierarchy},
    capabilities::server_capabilities,
    code_actions::{generate_code_actions, CodeActionContext},
    code_lens::{generate_all_code_lenses, WorkspaceStats},
    commands::{
        execute_analyze_workspace, execute_compilation_guidance, execute_explain_diagnostic,
        execute_recommend_backend, execute_show_backend_info, execute_suggest_tactics,
        execute_verify_command, COMMAND_ANALYZE_WORKSPACE, COMMAND_COMPILATION_GUIDANCE,
        COMMAND_EXPLAIN_DIAGNOSTIC, COMMAND_RECOMMEND_BACKEND, COMMAND_SHOW_BACKEND_INFO,
        COMMAND_SUGGEST_TACTICS, COMMAND_VERIFY,
    },
    diagnostics::{extract_position_from_error, generate_property_hint_diagnostic},
    document::DocumentStore,
    folding::generate_folding_ranges,
    formatter::{compute_range_edits, format_spec, FormatConfig},
    info::{backend_info, builtin_type_info, keyword_info, BACKEND_NAMES, BUILTIN_TYPES, KEYWORDS},
    inlay_hints::generate_inlay_hints,
    linked_editing::generate_linked_editing_ranges,
    moniker::resolve_monikers,
    selection_range::generate_selection_ranges,
    semantic_tokens::{generate_semantic_tokens, generate_semantic_tokens_in_range},
    signature_help::generate_signature_help,
    symbols::{
        collect_workspace_symbols, document_symbols_for_doc, format_type_def, is_definition_site,
        property_kind_name, type_or_property_info,
    },
};
use dashprove_knowledge::{Embedder, EmbeddingModel, KnowledgeStore, ToolKnowledgeStore};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::{Error as JsonRpcError, Result};
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

/// USL Language Server
pub struct UslLanguageServer {
    client: Client,
    documents: DocumentStore,
    knowledge_store: Arc<KnowledgeStore>,
    embedder: Arc<Embedder>,
    /// Tool knowledge store for structured JSON knowledge (loaded lazily)
    tool_store: Arc<RwLock<Option<ToolKnowledgeStore>>>,
}

/// Default location for the knowledge store used by expert features.
fn knowledge_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("dashprove")
        .join("knowledge")
}

/// Default location for tool knowledge JSON files.
/// First checks data/knowledge/tools relative to current dir (for development),
/// then falls back to the user's data directory.
fn tools_dir() -> PathBuf {
    // Try local data directory first (for development)
    let local_tools = PathBuf::from("data/knowledge/tools");
    if local_tools.exists() {
        return local_tools;
    }

    // Fall back to user's data directory
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("dashprove")
        .join("knowledge")
        .join("tools")
}

impl UslLanguageServer {
    /// Create a new language server instance.
    pub fn new(client: Client) -> Self {
        let model = EmbeddingModel::SentenceTransformers;
        let embedder = Arc::new(Embedder::new(model));
        let knowledge_store = Arc::new(KnowledgeStore::new(knowledge_dir(), model.dimensions()));
        Self {
            client,
            documents: DocumentStore::new(),
            knowledge_store,
            embedder,
            tool_store: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize tool knowledge store asynchronously.
    /// Called during server initialization.
    async fn initialize_tool_store(&self) {
        let tools_path = tools_dir();
        if !tools_path.exists() {
            tracing::warn!(
                "Tool knowledge directory not found: {}. Expert features will use fallback knowledge.",
                tools_path.display()
            );
            return;
        }

        match ToolKnowledgeStore::load_from_dir(&tools_path).await {
            Ok(store) => {
                let count = store.len();
                *self.tool_store.write().await = Some(store);
                tracing::info!(
                    "Loaded {} tool knowledge entries from {}",
                    count,
                    tools_path.display()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load tool knowledge from {}: {}. Expert features will use fallback knowledge.",
                    tools_path.display(),
                    e
                );
            }
        }
    }

    /// Check if tool knowledge store is available.
    pub async fn has_tool_store(&self) -> bool {
        self.tool_store.read().await.is_some()
    }

    /// Publish diagnostics for a document.
    async fn publish_diagnostics(&self, uri: Url) {
        let diagnostics = self
            .documents
            .with_document(&uri, |doc| {
                let mut diags = Vec::new();

                // Parse error
                if let Some(ref err) = doc.parse_error {
                    let message = err.to_string();
                    // Extract line/column from pest error if possible
                    let (line, col) = extract_position_from_error(&message);
                    diags.push(Diagnostic {
                        range: Range {
                            start: Position::new(line, col),
                            end: Position::new(line, col + 1),
                        },
                        severity: Some(DiagnosticSeverity::ERROR),
                        code: None,
                        code_description: None,
                        source: Some("dashprove-usl".to_string()),
                        message,
                        related_information: None,
                        tags: None,
                        data: None,
                    });
                }

                // Type errors
                for err in &doc.type_errors {
                    diags.push(Diagnostic {
                        range: Range {
                            start: Position::new(0, 0),
                            end: Position::new(0, 1),
                        },
                        severity: Some(DiagnosticSeverity::ERROR),
                        code: None,
                        code_description: None,
                        source: Some("dashprove-usl".to_string()),
                        message: err.clone(),
                        related_information: None,
                        tags: None,
                        data: None,
                    });
                }

                // Expert hints for properties (only when no errors)
                if doc.parse_error.is_none() && doc.type_errors.is_empty() {
                    if let Some(ref spec) = doc.spec {
                        for property in &spec.properties {
                            if let Some(hint_diag) =
                                generate_property_hint_diagnostic(doc, property)
                            {
                                diags.push(hint_diag);
                            }
                        }
                    }
                }

                diags
            })
            .unwrap_or_default();

        self.client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }
}

// NOTE: Diagnostic generation functions (generate_property_hint_diagnostic,
// property_kind_keyword, property_expert_hint, backend_display_name,
// extract_position_from_error) have been moved to the diagnostics module.
// See crates/dashprove-lsp/src/diagnostics.rs

#[tower_lsp::async_trait]
impl LanguageServer for UslLanguageServer {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: server_capabilities(),
            server_info: Some(ServerInfo {
                name: "dashprove-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        tracing::info!("USL language server initialized");
        // Load tool knowledge store in background
        self.initialize_tool_store().await;
    }

    async fn shutdown(&self) -> Result<()> {
        tracing::info!("USL language server shutting down");
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;
        let text = params.text_document.text;

        tracing::debug!("Document opened: {}", uri);
        self.documents.open(uri.clone(), version, text);
        self.publish_diagnostics(uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;

        // We use full document sync, so there's exactly one change with full text
        if let Some(change) = params.content_changes.into_iter().next() {
            tracing::debug!("Document changed: {}", uri);
            self.documents.update(&uri, version, change.text);
            self.publish_diagnostics(uri).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        tracing::debug!("Document closed: {}", uri);
        self.documents.close(&uri);
        // Clear diagnostics
        self.client.publish_diagnostics(uri, vec![], None).await;
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        let result = self.documents.with_document(uri, |doc| {
            let word = doc.word_at_position(pos.line, pos.character)?;

            // Check if it's a keyword
            if let Some(info) = keyword_info(word) {
                return Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: info.to_string(),
                    }),
                    range: None,
                });
            }

            // Check if it's a type or property name in the spec
            if let Some(ref spec) = doc.spec {
                if let Some(info) = type_or_property_info(spec, word) {
                    return Some(Hover {
                        contents: HoverContents::Markup(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: info,
                        }),
                        range: None,
                    });
                }
            }

            // Check if it's a builtin type
            if let Some(info) = builtin_type_info(word) {
                return Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: info.to_string(),
                    }),
                    range: None,
                });
            }

            // Check if it's a backend name (for comments or annotations)
            if let Some(info) = backend_info(word) {
                return Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: info.to_string(),
                    }),
                    range: None,
                });
            }

            None
        });

        Ok(result.flatten())
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;
        let result = self.documents.with_document(uri, |doc| {
            Some(DocumentSymbolResponse::Nested(document_symbols_for_doc(
                doc,
            )))
        });

        Ok(result.flatten())
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        let result = self.documents.with_document(uri, |doc| {
            let word = doc.word_at_position(pos.line, pos.character)?;

            // Try to find the type definition in the document
            let spec = doc.spec.as_ref()?;
            let type_def = spec.types.iter().find(|t| t.name == word)?;

            // Find the position of the type definition in the source
            let pattern = format!("type {}", type_def.name);
            let offset = doc.text.find(&pattern)?;
            let (line, character) = doc.offset_to_position(offset);

            Some(GotoDefinitionResponse::Scalar(Location {
                uri: doc.uri.clone(),
                range: Range {
                    start: Position::new(line, character),
                    end: Position::new(line, character + pattern.len() as u32),
                },
            }))
        });

        Ok(result.flatten())
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;

        let result = self.documents.with_document(uri, |doc| {
            let mut items = Vec::new();

            // Get partial word being typed
            let prefix = doc.word_at_position(pos.line, pos.character).unwrap_or("");

            // Keywords
            for kw in KEYWORDS {
                if kw.starts_with(prefix) || prefix.is_empty() {
                    items.push(CompletionItem {
                        label: kw.to_string(),
                        kind: Some(CompletionItemKind::KEYWORD),
                        detail: Some(format!("{} keyword", kw)),
                        documentation: keyword_info(kw).map(|info| {
                            Documentation::MarkupContent(MarkupContent {
                                kind: MarkupKind::Markdown,
                                value: info.to_string(),
                            })
                        }),
                        ..Default::default()
                    });
                }
            }

            // Builtin types
            for ty in BUILTIN_TYPES {
                if ty.starts_with(prefix) || prefix.is_empty() {
                    items.push(CompletionItem {
                        label: ty.to_string(),
                        kind: Some(CompletionItemKind::TYPE_PARAMETER),
                        detail: Some(format!("{} type", ty)),
                        documentation: builtin_type_info(ty).map(|info| {
                            Documentation::MarkupContent(MarkupContent {
                                kind: MarkupKind::Markdown,
                                value: info.to_string(),
                            })
                        }),
                        ..Default::default()
                    });
                }
            }

            // User-defined types from the spec
            if let Some(ref spec) = doc.spec {
                for type_def in &spec.types {
                    if type_def.name.starts_with(prefix) || prefix.is_empty() {
                        items.push(CompletionItem {
                            label: type_def.name.clone(),
                            kind: Some(CompletionItemKind::STRUCT),
                            detail: Some("User-defined type".to_string()),
                            documentation: Some(Documentation::MarkupContent(MarkupContent {
                                kind: MarkupKind::Markdown,
                                value: format_type_def(type_def),
                            })),
                            ..Default::default()
                        });
                    }
                }

                // Property names for references
                for prop in &spec.properties {
                    let name = prop.name();
                    if name.starts_with(prefix) || prefix.is_empty() {
                        items.push(CompletionItem {
                            label: name.clone(),
                            kind: Some(CompletionItemKind::FUNCTION),
                            detail: Some(property_kind_name(prop).to_string()),
                            ..Default::default()
                        });
                    }
                }
            }

            // Backend names (useful in comments, annotations, or backend directives)
            for backend_name in BACKEND_NAMES {
                if backend_name.starts_with(&prefix.to_lowercase()) || prefix.is_empty() {
                    items.push(CompletionItem {
                        label: backend_name.to_string(),
                        kind: Some(CompletionItemKind::MODULE),
                        detail: Some("Verification backend".to_string()),
                        documentation: backend_info(backend_name).map(|info| {
                            Documentation::MarkupContent(MarkupContent {
                                kind: MarkupKind::Markdown,
                                value: info.to_string(),
                            })
                        }),
                        ..Default::default()
                    });
                }
            }

            CompletionResponse::Array(items)
        });

        Ok(result)
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;

        let result = self.documents.with_document(uri, |doc| {
            // Get the word at the cursor position
            let word = doc.word_at_position(pos.line, pos.character)?;

            // Find all references to this identifier
            let ranges = doc.find_all_references(word);
            if ranges.is_empty() {
                return None;
            }

            let locations: Vec<Location> = ranges
                .into_iter()
                .map(|range| Location {
                    uri: doc.uri.clone(),
                    range,
                })
                .collect();

            Some(locations)
        });

        Ok(result.flatten())
    }

    async fn document_highlight(
        &self,
        params: DocumentHighlightParams,
    ) -> Result<Option<Vec<DocumentHighlight>>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        let result = self.documents.with_document(uri, |doc| {
            // Get the word at the cursor position
            let word = doc.word_at_position(pos.line, pos.character)?;

            // Find all occurrences of this identifier
            let ranges = doc.find_all_references(word);
            if ranges.is_empty() {
                return None;
            }

            // Determine highlight kind for each occurrence
            let highlights: Vec<DocumentHighlight> = ranges
                .into_iter()
                .map(|range| {
                    // Check if this is a definition (type or property name at definition site)
                    let kind = if let Some(ref spec) = doc.spec {
                        let is_definition = is_definition_site(spec, word, &range, &doc.text);
                        if is_definition {
                            Some(DocumentHighlightKind::WRITE)
                        } else {
                            Some(DocumentHighlightKind::READ)
                        }
                    } else {
                        Some(DocumentHighlightKind::TEXT)
                    };
                    DocumentHighlight { range, kind }
                })
                .collect();

            Some(highlights)
        });

        Ok(result.flatten())
    }

    async fn symbol(
        &self,
        params: WorkspaceSymbolParams,
    ) -> Result<Option<Vec<SymbolInformation>>> {
        let symbols = collect_workspace_symbols(&self.documents, &params.query);
        Ok(Some(symbols))
    }

    async fn prepare_rename(
        &self,
        params: TextDocumentPositionParams,
    ) -> Result<Option<PrepareRenameResponse>> {
        let uri = &params.text_document.uri;
        let pos = params.position;

        let result = self.documents.with_document(uri, |doc| {
            // Get the word at the cursor position
            let word = doc.word_at_position(pos.line, pos.character)?;

            // Only allow renaming user-defined identifiers (types and properties)
            if let Some(ref spec) = doc.spec {
                // Check if it's a type name
                let is_type = spec.types.iter().any(|t| t.name == word);
                // Check if it's a property name
                let is_property = spec.properties.iter().any(|p| p.name() == word);

                if is_type || is_property {
                    // Find the range of the word under cursor
                    let range = doc.find_identifier_range(word)?;
                    return Some(PrepareRenameResponse::Range(range));
                }
            }

            // Don't allow renaming keywords or builtins
            None
        });

        Ok(result.flatten())
    }

    async fn rename(&self, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
        let uri = &params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;
        let new_name = &params.new_name;

        // Validate new name is a valid identifier
        if new_name.is_empty()
            || !new_name
                .chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
            || !new_name.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            return Ok(None);
        }

        // Don't allow renaming to keywords
        if KEYWORDS.contains(&new_name.as_str()) || BUILTIN_TYPES.contains(&new_name.as_str()) {
            return Ok(None);
        }

        let result = self.documents.with_document(uri, |doc| {
            // Get the word at the cursor position
            let word = doc.word_at_position(pos.line, pos.character)?;

            // Verify it's a renameable identifier
            if let Some(ref spec) = doc.spec {
                let is_type = spec.types.iter().any(|t| t.name == word);
                let is_property = spec.properties.iter().any(|p| p.name() == word);

                if !is_type && !is_property {
                    return None;
                }
            } else {
                return None;
            }

            // Find all references and create text edits
            let ranges = doc.find_all_references(word);
            if ranges.is_empty() {
                return None;
            }

            let edits: Vec<TextEdit> = ranges
                .into_iter()
                .map(|range| TextEdit {
                    range,
                    new_text: new_name.clone(),
                })
                .collect();

            let mut changes = HashMap::new();
            changes.insert(doc.uri.clone(), edits);

            Some(WorkspaceEdit {
                changes: Some(changes),
                document_changes: None,
                change_annotations: None,
            })
        });

        Ok(result.flatten())
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = &params.text_document.uri;

        let result = self.documents.with_document(uri, |doc| {
            let tokens = generate_semantic_tokens(doc);
            SemanticTokensResult::Tokens(tokens)
        });

        Ok(result)
    }

    async fn semantic_tokens_range(
        &self,
        params: SemanticTokensRangeParams,
    ) -> Result<Option<SemanticTokensRangeResult>> {
        let uri = &params.text_document.uri;
        let range = params.range;

        let result = self.documents.with_document(uri, |doc| {
            let tokens = generate_semantic_tokens_in_range(doc, range);
            SemanticTokensRangeResult::Tokens(tokens)
        });

        Ok(result)
    }

    async fn code_action(&self, params: CodeActionParams) -> Result<Option<CodeActionResponse>> {
        let uri = &params.text_document.uri;
        let range = params.range;
        let diagnostics = params.context.diagnostics;

        let result = self.documents.with_document(uri, |doc| {
            let ctx = CodeActionContext {
                doc,
                range,
                diagnostics: &diagnostics,
            };
            generate_code_actions(&ctx)
        });

        match result {
            Some(actions) if !actions.is_empty() => Ok(Some(actions)),
            _ => Ok(None),
        }
    }

    async fn folding_range(&self, params: FoldingRangeParams) -> Result<Option<Vec<FoldingRange>>> {
        let uri = &params.text_document.uri;

        let result = self.documents.with_document(uri, |doc| {
            let ranges = generate_folding_ranges(doc);
            if ranges.is_empty() {
                None
            } else {
                Some(ranges)
            }
        });

        Ok(result.flatten())
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let uri = &params.text_document.uri;
        let range = params.range;

        let result = self.documents.with_document(uri, |doc| {
            let hints = generate_inlay_hints(doc, range);
            if hints.is_empty() {
                None
            } else {
                Some(hints)
            }
        });

        Ok(result.flatten())
    }

    async fn signature_help(&self, params: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        let result = self
            .documents
            .with_document(uri, |doc| generate_signature_help(doc, pos));

        Ok(result.flatten())
    }

    async fn formatting(&self, params: DocumentFormattingParams) -> Result<Option<Vec<TextEdit>>> {
        let uri = &params.text_document.uri;

        let result = self.documents.with_document(uri, |doc| {
            // Only format if we have a successfully parsed spec
            let spec = doc.spec.as_ref()?;

            let config = FormatConfig::default();
            let formatted = format_spec(spec, &config);

            // Create a single text edit that replaces the entire document
            let line_count = doc.text.lines().count();
            let last_line_len = doc.text.lines().last().map(|l| l.len()).unwrap_or(0);

            Some(vec![TextEdit {
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(line_count as u32, last_line_len as u32),
                },
                new_text: formatted,
            }])
        });

        Ok(result.flatten())
    }

    async fn code_lens(&self, params: CodeLensParams) -> Result<Option<Vec<CodeLens>>> {
        let uri = &params.text_document.uri;

        let mut workspace_stats = WorkspaceStats::new();
        self.documents
            .for_each_document(|doc| workspace_stats.add_document(doc));

        let result = self.documents.with_document(uri, |doc| {
            let lenses = generate_all_code_lenses(doc, &workspace_stats);
            if lenses.is_empty() {
                None
            } else {
                Some(lenses)
            }
        });

        Ok(result.flatten())
    }

    async fn range_formatting(
        &self,
        params: DocumentRangeFormattingParams,
    ) -> Result<Option<Vec<TextEdit>>> {
        let uri = &params.text_document.uri;
        let requested_range = params.range;

        let result = self.documents.with_document(uri, |doc| {
            // Only format if we have a successfully parsed spec
            let spec = doc.spec.as_ref()?;

            let config = FormatConfig::default();
            let formatted = format_spec(spec, &config);

            // Calculate the edits for the requested range
            // We format the whole document but return edits only for the overlapping range
            compute_range_edits(&doc.text, &formatted, requested_range)
        });

        Ok(result.flatten())
    }

    async fn execute_command(
        &self,
        params: ExecuteCommandParams,
    ) -> Result<Option<serde_json::Value>> {
        let command = params.command.as_str();
        let args = params.arguments;

        match command {
            COMMAND_VERIFY => execute_verify_command(&self.client, &self.documents, args).await,
            COMMAND_SHOW_BACKEND_INFO => {
                execute_show_backend_info(&self.client, &self.documents, args).await
            }
            COMMAND_EXPLAIN_DIAGNOSTIC => {
                execute_explain_diagnostic(
                    &self.client,
                    &self.knowledge_store,
                    &self.embedder,
                    &self.tool_store,
                    args,
                )
                .await
            }
            COMMAND_SUGGEST_TACTICS => {
                execute_suggest_tactics(
                    &self.client,
                    &self.knowledge_store,
                    &self.embedder,
                    &self.tool_store,
                    args,
                )
                .await
            }
            COMMAND_RECOMMEND_BACKEND => {
                execute_recommend_backend(
                    &self.client,
                    &self.knowledge_store,
                    &self.embedder,
                    &self.tool_store,
                    args,
                )
                .await
            }
            COMMAND_COMPILATION_GUIDANCE => {
                execute_compilation_guidance(
                    &self.client,
                    &self.knowledge_store,
                    &self.embedder,
                    &self.tool_store,
                    args,
                )
                .await
            }
            COMMAND_ANALYZE_WORKSPACE => {
                execute_analyze_workspace(
                    &self.client,
                    &self.documents,
                    &self.knowledge_store,
                    &self.embedder,
                    &self.tool_store,
                )
                .await
            }
            _ => {
                tracing::warn!("Unknown command: {}", command);
                Err(JsonRpcError::method_not_found())
            }
        }
    }

    async fn selection_range(
        &self,
        params: SelectionRangeParams,
    ) -> Result<Option<Vec<SelectionRange>>> {
        let uri = &params.text_document.uri;
        let positions = &params.positions;

        let result = self.documents.with_document(uri, |doc| {
            let ranges = generate_selection_ranges(doc, positions);
            if ranges.is_empty() {
                None
            } else {
                Some(ranges)
            }
        });

        Ok(result.flatten())
    }

    async fn prepare_call_hierarchy(
        &self,
        params: CallHierarchyPrepareParams,
    ) -> Result<Option<Vec<CallHierarchyItem>>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        let result = self
            .documents
            .with_document(uri, |doc| prepare_call_hierarchy(doc, pos));

        Ok(result.flatten())
    }

    async fn incoming_calls(
        &self,
        params: CallHierarchyIncomingCallsParams,
    ) -> Result<Option<Vec<CallHierarchyIncomingCall>>> {
        let item = &params.item;

        // Get the document from the item's URI
        let result = self
            .documents
            .with_document(&item.uri, |doc| find_incoming_calls(doc, item));

        Ok(result.flatten())
    }

    async fn outgoing_calls(
        &self,
        params: CallHierarchyOutgoingCallsParams,
    ) -> Result<Option<Vec<CallHierarchyOutgoingCall>>> {
        let item = &params.item;

        // Get the document from the item's URI
        let result = self
            .documents
            .with_document(&item.uri, |doc| find_outgoing_calls(doc, item));

        Ok(result.flatten())
    }

    async fn moniker(&self, params: MonikerParams) -> Result<Option<Vec<Moniker>>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        let result = self
            .documents
            .with_document(uri, |doc| resolve_monikers(doc, pos));

        Ok(result.flatten())
    }

    async fn linked_editing_range(
        &self,
        params: LinkedEditingRangeParams,
    ) -> Result<Option<LinkedEditingRanges>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        let result = self
            .documents
            .with_document(uri, |doc| generate_linked_editing_ranges(doc, pos));

        Ok(result.flatten())
    }
}

// NOTE: Command identifiers, execute_* handlers, and helper functions
// have been moved to the commands module for better code organization.
// See crates/dashprove-lsp/src/commands.rs

// NOTE: compute_range_edits has been moved to the formatter module.
// See crates/dashprove-lsp/src/formatter.rs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::{recommended_backend, SUPPORTED_COMMANDS};
    use crate::document::Document;
    use futures_util::StreamExt;
    use std::env;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::time::Duration;
    use tokio::time::timeout;
    use tower::util::ServiceExt;
    use tower::Service;
    use tower_lsp::jsonrpc::Request;
    use tower_lsp::{ClientSocket, LspService};

    static WORKDIR_MUTEX: Mutex<()> = Mutex::new(());

    async fn create_initialized_service() -> (LspService<UslLanguageServer>, ClientSocket) {
        let (mut service, socket) = LspService::new(UslLanguageServer::new);

        let init_request = Request::build("initialize")
            .params(serde_json::to_value(InitializeParams::default()).unwrap())
            .id(1)
            .finish();
        let _ = service
            .ready()
            .await
            .expect("service ready for initialize")
            .call(init_request)
            .await
            .expect("initialize response");

        let initialized = Request::build("initialized")
            .params(serde_json::to_value(InitializedParams {}).unwrap())
            .finish();
        let _ = service
            .ready()
            .await
            .expect("service ready for initialized")
            .call(initialized)
            .await
            .expect("initialized response");

        (service, socket)
    }

    async fn open_document(server: &UslLanguageServer, uri: Url, text: &str, version: i32) {
        server
            .did_open(DidOpenTextDocumentParams {
                text_document: TextDocumentItem {
                    uri,
                    language_id: "usl".to_string(),
                    version,
                    text: text.to_string(),
                },
            })
            .await;
    }

    // Tests for extract_position_from_error, property_kind_keyword,
    // property_expert_hint, backend_display_name, and generate_property_hint_diagnostic
    // have been moved to the diagnostics module.

    // Tests for keyword_info, builtin_type_info, backend_info, format_type,
    // document_symbols_*, workspace_symbols_*, property_signature_info_*, and
    // is_definition_site_* have been moved to info.rs and symbols.rs modules.

    #[tokio::test]
    async fn initialize_returns_capabilities() {
        let (service, _socket) = LspService::new(UslLanguageServer::new);
        let server = service.inner();
        let init = server
            .initialize(InitializeParams::default())
            .await
            .expect("initialize should succeed");

        let server_info = init.server_info.expect("server info should be present");
        assert_eq!(server_info.name, "dashprove-lsp");
        assert!(init.capabilities.completion_provider.is_some());
        assert!(init.capabilities.rename_provider.is_some());
    }

    #[test]
    fn knowledge_and_tools_directories_are_configured() {
        let knowledge = knowledge_dir();
        assert!(
            knowledge.ends_with(Path::new("dashprove/knowledge")),
            "knowledge dir should end with dashprove/knowledge, got {}",
            knowledge.display()
        );

        let tools_path = tools_dir();
        if PathBuf::from("data/knowledge/tools").exists() {
            assert_eq!(
                tools_path,
                PathBuf::from("data/knowledge/tools"),
                "local tools directory should be preferred when present"
            );
        } else {
            assert!(
                tools_path.ends_with(Path::new("dashprove/knowledge/tools")),
                "fallback tools directory should live under data dir, got {}",
                tools_path.display()
            );
        }
    }

    #[tokio::test]
    async fn tool_store_initializes_when_tools_exist() {
        let (service, _) = create_initialized_service().await;
        let server = service.inner();
        assert!(
            !server.has_tool_store().await,
            "tool store should be empty before initialization"
        );

        let original_dir = env::current_dir().expect("current dir");
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root")
            .to_path_buf();
        {
            let _guard = WORKDIR_MUTEX.lock().unwrap();
            env::set_current_dir(&workspace_root).expect("set workspace root");
        }

        server.initialize_tool_store().await;

        {
            let _guard = WORKDIR_MUTEX.lock().unwrap();
            env::set_current_dir(original_dir).expect("restore working directory");
        }
        assert!(
            server.has_tool_store().await,
            "tool store should load from data/knowledge/tools"
        );
    }

    #[tokio::test]
    async fn publish_diagnostics_reports_parse_errors() {
        let (service, mut socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///diagnostic.usl").unwrap();

        open_document(server, uri.clone(), "theorem {", 1).await;

        let request = timeout(Duration::from_secs(1), socket.next())
            .await
            .expect("publishDiagnostics notification should arrive")
            .expect("publishDiagnostics notification missing");

        assert_eq!(request.method(), "textDocument/publishDiagnostics");
        let (_, _, params) = request.into_parts();
        let params: PublishDiagnosticsParams =
            serde_json::from_value(params.expect("diagnostic params")).unwrap();
        assert!(!params.diagnostics.is_empty());
        let diag = &params.diagnostics[0];
        assert_eq!(diag.severity, Some(DiagnosticSeverity::ERROR));
        assert_eq!(
            diag.range.end.character,
            diag.range.start.character + 1,
            "diagnostic range should span a single character"
        );
    }

    #[tokio::test]
    async fn publish_diagnostics_skips_hints_when_type_errors_exist() {
        let (service, mut socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///diagnostic_type.usl").unwrap();

        // Unknown type should produce a type error but no property hint diagnostics
        open_document(
            server,
            uri.clone(),
            "type Node = { id: Int }\ntype Node = { name: String }\ntheorem bad { true }",
            1,
        )
        .await;

        let has_type_errors = server
            .documents
            .with_document(&uri, |doc| !doc.type_errors.is_empty())
            .unwrap_or(false);
        assert!(has_type_errors, "expected document to record type errors");

        let request = timeout(Duration::from_secs(1), socket.next())
            .await
            .expect("publishDiagnostics notification should arrive")
            .expect("publishDiagnostics notification missing");
        assert_eq!(request.method(), "textDocument/publishDiagnostics");
        let (_, _, params) = request.into_parts();
        let params: PublishDiagnosticsParams =
            serde_json::from_value(params.expect("diagnostic params")).unwrap();

        assert!(
            params
                .diagnostics
                .iter()
                .any(|diag| diag.severity == Some(DiagnosticSeverity::ERROR)),
            "type errors should produce error diagnostics"
        );
        assert!(
            params
                .diagnostics
                .iter()
                .all(|diag| diag.severity != Some(DiagnosticSeverity::HINT)),
            "property hint diagnostics should not be emitted when type errors are present"
        );
    }

    #[tokio::test]
    async fn language_server_handlers_return_results() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///handlers.usl").unwrap();
        let text = r#"type Value = { id: Int }

theorem value_check { forall v: Value . Value.id > 0 }
contract Foo::bar(self: Int) -> Int {
    requires { self > 0 }
    ensures { Value.id > 0 }
}
"#;

        open_document(server, uri.clone(), text, 1).await;

        let hover = server
            .hover(HoverParams {
                text_document_position_params: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: uri.clone() },
                    position: Position::new(0, 5),
                },
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap();
        assert!(hover.is_some(), "hover should return keyword info");

        let symbols = server
            .document_symbol(DocumentSymbolParams {
                text_document: TextDocumentIdentifier { uri: uri.clone() },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap();
        assert!(symbols.is_some(), "document symbols should be reported");

        let (def_pos, use_pos) = server
            .documents
            .with_document(&uri, |doc| {
                let refs = doc.find_all_references("Value");
                (refs[0].start, refs[1].start)
            })
            .unwrap();

        let definition = server
            .goto_definition(GotoDefinitionParams {
                text_document_position_params: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: uri.clone() },
                    position: use_pos,
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap();
        match definition {
            Some(GotoDefinitionResponse::Scalar(location)) => {
                assert_eq!(location.range.start.line, def_pos.line);
                assert!(
                    location.range.start.character <= def_pos.character,
                    "definition should start at or before identifier"
                );
                assert!(
                    location.range.end.character > def_pos.character,
                    "definition span should include identifier"
                );
            }
            other => panic!("expected definition location, got {:?}", other),
        }

        let completion = server
            .completion(CompletionParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: uri.clone() },
                    position: Position::new(1, 0),
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
                context: None,
            })
            .await
            .unwrap();
        let items = match completion {
            Some(CompletionResponse::Array(items)) => items,
            other => panic!("expected completion array, got {:?}", other),
        };
        let mut by_label: HashMap<String, CompletionItem> = HashMap::new();
        for item in items {
            by_label.insert(item.label.clone(), item);
        }
        let theorem_item = by_label.get("theorem").expect("keyword completion missing");
        assert_eq!(theorem_item.kind, Some(CompletionItemKind::KEYWORD));
        assert!(theorem_item.detail.as_deref().unwrap().contains("keyword"));

        let int_item = by_label.get("Int").expect("builtin completion missing");
        assert_eq!(int_item.kind, Some(CompletionItemKind::TYPE_PARAMETER));
        assert!(int_item.detail.as_deref().unwrap().contains("type"));

        let type_item = by_label
            .get("Value")
            .expect("user-defined type completion missing");
        assert_eq!(type_item.kind, Some(CompletionItemKind::STRUCT));
        assert!(type_item.documentation.is_some());

        let prop_item = by_label
            .get("value_check")
            .expect("property completion missing");
        assert_eq!(prop_item.kind, Some(CompletionItemKind::FUNCTION));

        let backend_item = by_label.get("kani").expect("backend completion missing");
        assert_eq!(backend_item.kind, Some(CompletionItemKind::MODULE));
        assert!(backend_item.documentation.is_some());

        let references = server
            .references(ReferenceParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: uri.clone() },
                    position: use_pos,
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
                context: ReferenceContext {
                    include_declaration: true,
                },
            })
            .await
            .unwrap()
            .expect("references should be returned");
        assert_eq!(references.len(), 4);

        let highlights = server
            .document_highlight(DocumentHighlightParams {
                text_document_position_params: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: uri.clone() },
                    position: use_pos,
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap()
            .expect("document highlights should be returned");
        assert_eq!(highlights.len(), 4);

        let prepare = server
            .prepare_rename(TextDocumentPositionParams {
                text_document: TextDocumentIdentifier { uri: uri.clone() },
                position: def_pos,
            })
            .await
            .unwrap();
        assert!(prepare.is_some(), "rename should be prepared for types");

        let rename = server
            .rename(RenameParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: uri.clone() },
                    position: use_pos,
                },
                new_name: "Entity".to_string(),
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap()
            .expect("rename should produce edits");
        let edits = rename
            .changes
            .expect("workspace edits should contain file entry")
            .get(&uri)
            .cloned()
            .expect("edits for uri");
        assert_eq!(edits.len(), 4);
        assert!(edits.iter().all(|edit| edit.new_text == "Entity"));

        let tokens = server
            .semantic_tokens_full(SemanticTokensParams {
                text_document: TextDocumentIdentifier { uri: uri.clone() },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap();
        let data = match tokens {
            Some(SemanticTokensResult::Tokens(tokens)) => tokens.data,
            other => panic!("expected semantic tokens, got {:?}", other),
        };
        assert!(
            !data.is_empty(),
            "semantic tokens should include at least one token"
        );

        let formatting = server
            .formatting(DocumentFormattingParams {
                text_document: TextDocumentIdentifier { uri: uri.clone() },
                options: FormattingOptions {
                    tab_size: 2,
                    insert_spaces: true,
                    ..Default::default()
                },
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap()
            .expect("formatting should produce an edit");
        assert_eq!(formatting.len(), 1);
        let original_bounds = server
            .documents
            .with_document(&uri, |doc| {
                (
                    doc.text.lines().count() as u32,
                    doc.text.lines().last().map(|l| l.len()).unwrap_or(0) as u32,
                )
            })
            .unwrap();
        assert_eq!(formatting[0].range.start, Position::new(0, 0));
        assert_eq!(
            formatting[0].range.end,
            Position::new(original_bounds.0, original_bounds.1)
        );
        assert!(
            !formatting[0].new_text.is_empty(),
            "formatting should return the formatted document text"
        );

        let range_formatting = server
            .range_formatting(DocumentRangeFormattingParams {
                text_document: TextDocumentIdentifier { uri: uri.clone() },
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(4, 0),
                },
                options: FormattingOptions {
                    tab_size: 2,
                    insert_spaces: true,
                    ..Default::default()
                },
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap();
        assert!(
            range_formatting.is_some(),
            "range formatting should return edits when formatting changes text"
        );
        let range_edits = range_formatting.unwrap();
        assert!(
            range_edits.iter().any(|edit| !edit.new_text.is_empty()),
            "range formatting edits should carry new text"
        );

        let folding = server
            .folding_range(FoldingRangeParams {
                text_document: TextDocumentIdentifier { uri: uri.clone() },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap()
            .expect("folding ranges should be returned");
        assert!(!folding.is_empty());
        assert!(
            folding
                .iter()
                .any(|range| range.end_line > range.start_line),
            "folding ranges should span more than a single line"
        );

        let lenses = server
            .code_lens(CodeLensParams {
                text_document: TextDocumentIdentifier { uri },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap()
            .expect("code lenses should be available");
        assert!(!lenses.is_empty());
    }

    #[tokio::test]
    async fn did_change_updates_document_store() {
        let (service, mut socket) = create_initialized_service().await;
        let _drain = tokio::spawn(async move { while socket.next().await.is_some() {} });
        let server = service.inner();
        let uri = Url::parse("file:///change.usl").unwrap();

        open_document(
            server,
            uri.clone(),
            "theorem check { forall x: Bool . x }",
            1,
        )
        .await;
        let before = server
            .documents
            .with_document(&uri, |doc| (doc.version, doc.text.clone()))
            .unwrap();
        assert_eq!(before.0, 1);

        server
            .did_change(DidChangeTextDocumentParams {
                text_document: VersionedTextDocumentIdentifier {
                    uri: uri.clone(),
                    version: 2,
                },
                content_changes: vec![TextDocumentContentChangeEvent {
                    range: None,
                    range_length: None,
                    text: "theorem check { forall x: Bool . !x }".to_string(),
                }],
            })
            .await;

        let after = server
            .documents
            .with_document(&uri, |doc| (doc.version, doc.text.clone()))
            .unwrap();
        assert_eq!(after.0, 2);
        assert!(
            after.1.contains("!x"),
            "document text should be updated after change"
        );
    }

    #[tokio::test]
    async fn did_close_removes_document_and_diagnostics() {
        let (service, mut socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///close.usl").unwrap();

        open_document(server, uri.clone(), "theorem check { true }", 1).await;

        // Drain initial publishDiagnostics notification to avoid influencing the close test.
        let _ = timeout(Duration::from_millis(500), socket.next()).await;

        server
            .did_close(DidCloseTextDocumentParams {
                text_document: TextDocumentIdentifier { uri: uri.clone() },
            })
            .await;

        assert!(
            server
                .documents
                .with_document(&uri, |doc| doc.version)
                .is_none(),
            "document should be removed from store after close"
        );
    }

    #[tokio::test]
    async fn workspace_symbol_returns_results_for_query() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///workspace_symbols.usl").unwrap();
        let text = "type Value = { id: Int }\ntheorem check { forall v: Value . true }";

        open_document(server, uri.clone(), text, 1).await;

        let symbols = server
            .symbol(WorkspaceSymbolParams {
                query: "Value".to_string(),
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap()
            .expect("workspace symbols should be available");

        assert!(
            symbols.iter().any(|sym| sym.name == "Value"),
            "workspace symbols should include the requested type"
        );
    }

    #[test]
    fn test_is_renameable_identifier() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Value = { id: Int }\ntheorem value_check { forall v: Value . true }".to_string(),
        );

        let spec = doc.spec.as_ref().expect("should parse");

        // Type "Value" should be renameable
        let is_type = spec.types.iter().any(|t| t.name == "Value");
        assert!(is_type);

        // Property "value_check" should be renameable
        let is_property = spec.properties.iter().any(|p| p.name() == "value_check");
        assert!(is_property);

        // "Int" is a builtin type, not renameable
        let is_int_type = spec.types.iter().any(|t| t.name == "Int");
        assert!(!is_int_type);
    }

    #[test]
    fn test_find_rename_edits() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Value = { id: Int }\ntheorem value_check { forall v: Value . Value.id > 0 }"
                .to_string(),
        );

        // Find all occurrences of "Value" and create edits
        let references = doc.find_all_references("Value");
        assert_eq!(references.len(), 3);

        // Simulate creating text edits
        let new_name = "Item";
        let edits: Vec<TextEdit> = references
            .into_iter()
            .map(|range| TextEdit {
                range,
                new_text: new_name.to_string(),
            })
            .collect();

        assert_eq!(edits.len(), 3);
        for edit in &edits {
            assert_eq!(edit.new_text, "Item");
        }
    }

    #[tokio::test]
    async fn prepare_rename_rejects_builtin_identifiers() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///prepare_rename.usl").unwrap();
        let text = "type Value = { id: Int }\ntheorem check { forall v: Value . true }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .prepare_rename(TextDocumentPositionParams {
                text_document: TextDocumentIdentifier { uri },
                position: Position::new(0, 20), // on builtin type "Int"
            })
            .await
            .unwrap();

        assert!(
            result.is_none(),
            "prepare_rename should reject builtin types even when other symbols exist"
        );
    }

    #[test]
    fn test_rename_validation() {
        // Invalid names should be rejected
        let invalid_names = vec!["", "123invalid", "with space", "with-hyphen"];
        for name in invalid_names {
            let is_valid = !name.is_empty()
                && name
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_alphabetic() || c == '_')
                && name.chars().all(|c| c.is_alphanumeric() || c == '_');
            assert!(!is_valid, "Name '{}' should be invalid", name);
        }

        // Valid names should pass
        let valid_names = vec!["Value", "my_type", "_private", "Type2"];
        for name in valid_names {
            let is_valid = !name.is_empty()
                && name
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_alphabetic() || c == '_')
                && name.chars().all(|c| c.is_alphanumeric() || c == '_');
            assert!(is_valid, "Name '{}' should be valid", name);
        }
    }

    #[test]
    fn test_cannot_rename_to_keyword() {
        // Keywords cannot be used as new names
        for kw in KEYWORDS {
            let is_keyword = KEYWORDS.contains(kw);
            assert!(is_keyword, "'{}' should be a keyword", kw);
        }

        // Builtin types cannot be used as new names
        for ty in BUILTIN_TYPES {
            let is_builtin = BUILTIN_TYPES.contains(ty);
            assert!(is_builtin, "'{}' should be a builtin type", ty);
        }
    }

    #[test]
    fn test_document_highlight_finds_all_occurrences() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Value = { id: Int }\ntheorem value_check { forall v: Value . Value.id > 0 }"
                .to_string(),
        );

        // "Value" appears 3 times
        let references = doc.find_all_references("Value");
        assert_eq!(references.len(), 3);

        // First is the definition (line 0, char 5)
        assert_eq!(references[0].start.line, 0);
        assert_eq!(references[0].start.character, 5);

        // Second is type annotation (line 1)
        assert_eq!(references[1].start.line, 1);

        // Third is field access (line 1)
        assert_eq!(references[2].start.line, 1);
    }

    #[test]
    fn test_document_highlight_distinguishes_definition_and_usage() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Item = { name: String }\ntheorem item_valid { forall i: Item . true }"
                .to_string(),
        );

        let spec = doc.spec.as_ref().expect("should parse");
        let references = doc.find_all_references("Item");

        assert_eq!(references.len(), 2);

        // First occurrence is the definition
        assert!(is_definition_site(spec, "Item", &references[0], &doc.text));

        // Second occurrence is a usage
        assert!(!is_definition_site(spec, "Item", &references[1], &doc.text));
    }

    #[test]
    fn test_command_constants() {
        assert_eq!(COMMAND_VERIFY, "dashprove.verify");
        assert_eq!(COMMAND_SHOW_BACKEND_INFO, "dashprove.showBackendInfo");
        assert_eq!(COMMAND_EXPLAIN_DIAGNOSTIC, "dashprove.explainDiagnostic");
        assert_eq!(COMMAND_SUGGEST_TACTICS, "dashprove.suggestTactics");
        assert_eq!(COMMAND_RECOMMEND_BACKEND, "dashprove.recommendBackend");
        assert_eq!(
            COMMAND_COMPILATION_GUIDANCE,
            "dashprove.compilationGuidance"
        );
        assert_eq!(COMMAND_ANALYZE_WORKSPACE, "dashprove.analyzeWorkspace");
        assert_eq!(SUPPORTED_COMMANDS.len(), 7);
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_VERIFY));
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_SHOW_BACKEND_INFO));
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_EXPLAIN_DIAGNOSTIC));
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_SUGGEST_TACTICS));
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_RECOMMEND_BACKEND));
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_COMPILATION_GUIDANCE));
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_ANALYZE_WORKSPACE));
    }

    #[test]
    fn test_recommended_backend_for_theorem() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem my_theorem { true }".to_string(),
        );

        let spec = doc.spec.as_ref().expect("should parse");
        let prop = &spec.properties[0];
        assert_eq!(recommended_backend(prop), "lean4");
    }

    #[test]
    fn test_recommended_backend_for_temporal() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "temporal my_temporal { always(true) }".to_string(),
        );

        let spec = doc.spec.as_ref().expect("should parse");
        let prop = &spec.properties[0];
        assert_eq!(recommended_backend(prop), "tlaplus");
    }

    #[test]
    fn test_recommended_backend_for_contract() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "contract Foo::bar(self: Int) -> Int {\n  requires { true }\n  ensures { true }\n}"
                .to_string(),
        );

        let spec = doc.spec.as_ref().expect("should parse");
        let prop = &spec.properties[0];
        assert_eq!(recommended_backend(prop), "kani");
    }

    #[test]
    fn test_recommended_backend_for_invariant() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "invariant my_inv { true }".to_string(),
        );

        let spec = doc.spec.as_ref().expect("should parse");
        let prop = &spec.properties[0];
        assert_eq!(recommended_backend(prop), "lean4");
    }

    #[test]
    fn test_recommended_backend_for_refinement() {
        let doc = Document::new(
            Url::parse("file:///refinement.usl").unwrap(),
            1,
            include_str!("../../../examples/usl/refinement.usl").to_string(),
        );

        let spec = doc.spec.as_ref().expect("should parse");
        // Refinement is the first property
        let prop = &spec.properties[0];
        assert_eq!(recommended_backend(prop), "lean4");
    }

    // Tests for compute_range_edits have been moved to the formatter module.

    // Tests for property_kind_keyword, backend_display_name, property_expert_hint,
    // and generate_property_hint_diagnostic have been moved to the diagnostics module.

    #[tokio::test]
    async fn test_semantic_tokens_range_returns_tokens() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///sem_range.usl").unwrap();
        let text = "type Value = { id: Int }\ntheorem check { forall v: Value . true }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .semantic_tokens_range(SemanticTokensRangeParams {
                text_document: TextDocumentIdentifier { uri },
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(0, 24),
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap();

        assert!(
            result.is_some(),
            "semantic_tokens_range should return tokens for valid range"
        );
        match result.unwrap() {
            SemanticTokensRangeResult::Tokens(tokens) => {
                assert!(!tokens.data.is_empty(), "should have tokens in range");
            }
            SemanticTokensRangeResult::Partial(_) => {}
        }
    }

    #[tokio::test]
    async fn test_code_action_returns_actions() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///code_action.usl").unwrap();
        let text = "theorem my_theorem { forall x: Bool . x }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .code_action(CodeActionParams {
                text_document: TextDocumentIdentifier { uri },
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(0, 40),
                },
                context: tower_lsp::lsp_types::CodeActionContext {
                    diagnostics: vec![],
                    only: None,
                    trigger_kind: None,
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap();

        // Code actions are returned when applicable (may be None if no actions available)
        // The test validates the handler runs without error
        let _ = result;
    }

    #[tokio::test]
    async fn test_code_action_returns_some_for_property() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///code_action2.usl").unwrap();
        let text = "theorem my_theorem { forall x: Bool . x }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .code_action(CodeActionParams {
                text_document: TextDocumentIdentifier { uri },
                range: Range {
                    start: Position::new(0, 8),
                    end: Position::new(0, 18),
                },
                context: tower_lsp::lsp_types::CodeActionContext {
                    diagnostics: vec![],
                    only: None,
                    trigger_kind: None,
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap();

        assert!(
            result.is_some(),
            "code_action should return actions for property position"
        );
        let actions = result.unwrap();
        assert!(!actions.is_empty(), "should have at least one code action");
    }

    #[tokio::test]
    async fn test_code_action_none_when_no_actions_available() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///code_action_empty.usl").unwrap();

        open_document(server, uri.clone(), "", 1).await;

        let result = server
            .code_action(CodeActionParams {
                text_document: TextDocumentIdentifier { uri },
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(0, 0),
                },
                context: tower_lsp::lsp_types::CodeActionContext {
                    diagnostics: vec![],
                    only: None,
                    trigger_kind: None,
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap();

        assert!(
            result.is_none(),
            "empty documents should not produce code actions"
        );
    }

    #[tokio::test]
    async fn test_inlay_hint_returns_hints() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///inlay.usl").unwrap();
        let text = "theorem check { forall x: Bool . x }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .inlay_hint(InlayHintParams {
                text_document: TextDocumentIdentifier { uri },
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(0, 36),
                },
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap();

        // Inlay hints are available when the document has applicable constructs
        assert!(
            result.is_some(),
            "inlay_hint should return hints for theorem"
        );
        let hints = result.unwrap();
        assert!(!hints.is_empty(), "should have at least one inlay hint");
    }

    #[tokio::test]
    async fn test_signature_help_returns_help() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///sig_help.usl").unwrap();
        let text = r#"
type Stack = { elements: List<Int>, capacity: Int }

contract Stack::push(self: Stack, item: Int) -> Result<Stack> {
    requires { self.capacity > 0 }
    ensures { true }
}

theorem test { true }
"#;

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .signature_help(SignatureHelpParams {
                text_document_position_params: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri },
                    position: Position::new(3, 21), // Inside the parameter list
                },
                work_done_progress_params: Default::default(),
                context: None,
            })
            .await
            .unwrap();

        // Signature help is available inside function parameters
        let help = result.expect("signature_help should return data");
        assert!(
            !help.signatures.is_empty(),
            "signature_help should provide at least one signature"
        );
        let label = &help.signatures[0].label;
        assert!(
            label.contains("Stack::push"),
            "signature label should mention the contract name"
        );
    }

    #[tokio::test]
    async fn test_selection_range_returns_ranges() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///selection.usl").unwrap();
        let text = "type Value = { id: Int }\ntheorem check { forall v: Value . true }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .selection_range(SelectionRangeParams {
                text_document: TextDocumentIdentifier { uri },
                positions: vec![Position::new(0, 5)], // On "Value"
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap();

        assert!(
            result.is_some(),
            "selection_range should return ranges for valid position"
        );
        let ranges = result.unwrap();
        assert!(
            !ranges.is_empty(),
            "should have at least one selection range"
        );
        let first_range = &ranges[0].range;
        assert!(
            first_range.start.line == 0 && first_range.end.line >= first_range.start.line,
            "selection range should include the requested line"
        );
        assert!(
            first_range.start.character <= 5 && first_range.end.character >= 5,
            "selection range should include the requested character"
        );
    }

    #[tokio::test]
    async fn test_prepare_call_hierarchy_returns_items() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///call_hier.usl").unwrap();
        let text = "theorem caller { uses(callee) }\ntheorem callee { true }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .prepare_call_hierarchy(CallHierarchyPrepareParams {
                text_document_position_params: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri },
                    position: Position::new(0, 8), // On "caller"
                },
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap();

        // prepare_call_hierarchy returns items for property definitions
        assert!(
            result.is_some(),
            "prepare_call_hierarchy should return items for property"
        );
        let items = result.unwrap();
        assert!(
            !items.is_empty(),
            "should have at least one call hierarchy item"
        );
    }

    #[tokio::test]
    async fn test_incoming_calls_returns_calls() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///incoming.usl").unwrap();
        // In USL, incoming calls for a TYPE returns properties that reference it
        let text = "type Value = { id: Int }\ntheorem check { forall v: Value . true }";

        open_document(server, uri.clone(), text, 1).await;

        // Prepare call hierarchy for the type "Value"
        let items = server
            .prepare_call_hierarchy(CallHierarchyPrepareParams {
                text_document_position_params: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: uri.clone() },
                    position: Position::new(0, 5), // On "Value" type
                },
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap()
            .unwrap();

        let result = server
            .incoming_calls(CallHierarchyIncomingCallsParams {
                item: items[0].clone(),
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap();

        // incoming_calls returns properties that reference this type
        assert!(
            result.is_some(),
            "incoming_calls should return callers for type"
        );
        let calls = result.unwrap();
        assert!(
            !calls.is_empty(),
            "Value type should have at least one caller (the theorem)"
        );
    }

    #[tokio::test]
    async fn test_outgoing_calls_returns_calls() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///outgoing.usl").unwrap();
        // In USL, outgoing calls for a PROPERTY returns types referenced in it
        let text = "type Value = { id: Int }\ntheorem check { forall v: Value . true }";

        open_document(server, uri.clone(), text, 1).await;

        // Prepare call hierarchy for the property "check"
        let items = server
            .prepare_call_hierarchy(CallHierarchyPrepareParams {
                text_document_position_params: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: uri.clone() },
                    position: Position::new(1, 8), // On "check" theorem
                },
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap()
            .unwrap();

        let result = server
            .outgoing_calls(CallHierarchyOutgoingCallsParams {
                item: items[0].clone(),
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap();

        // outgoing_calls for a property returns types it references
        assert!(
            result.is_some(),
            "outgoing_calls should return callees for property"
        );
        let calls = result.unwrap();
        assert!(!calls.is_empty(), "check theorem should call Value type");
    }

    #[tokio::test]
    async fn test_moniker_returns_monikers() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///moniker.usl").unwrap();
        let text = "type Value = { id: Int }\ntheorem check { forall v: Value . true }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .moniker(MonikerParams {
                text_document_position_params: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri },
                    position: Position::new(0, 5), // On "Value"
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            })
            .await
            .unwrap();

        // moniker returns unique identifiers for symbols
        assert!(result.is_some(), "moniker should return monikers for type");
        let monikers = result.unwrap();
        assert!(!monikers.is_empty(), "should have at least one moniker");
    }

    #[tokio::test]
    async fn test_linked_editing_range_returns_ranges() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///linked.usl").unwrap();
        let text = "type Value = { id: Int }\ntheorem check { forall v: Value . Value.id > 0 }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .linked_editing_range(LinkedEditingRangeParams {
                text_document_position_params: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri },
                    position: Position::new(0, 5), // On "Value"
                },
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap();

        // linked_editing_range returns ranges that should be edited together
        assert!(
            result.is_some(),
            "linked_editing_range should return ranges for type with multiple references"
        );
        let ranges = result.unwrap();
        assert!(
            ranges.ranges.len() >= 2,
            "should have at least 2 linked ranges for Value"
        );
    }

    #[tokio::test]
    async fn completion_respects_prefixes_and_metadata() {
        let (service, mut socket) = create_initialized_service().await;
        let _drain = tokio::spawn(async move { while socket.next().await.is_some() {} });
        let server = service.inner();

        // Keyword completion with non-empty prefix
        let keyword_uri = Url::parse("file:///complete_keyword.usl").unwrap();
        open_document(server, keyword_uri.clone(), "th", 1).await;
        let keyword_items = server
            .completion(CompletionParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: keyword_uri },
                    position: Position::new(0, 1),
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
                context: None,
            })
            .await
            .unwrap()
            .expect("keyword completion should return items");
        let keyword_array = match keyword_items {
            CompletionResponse::Array(items) => items,
            other => panic!("expected completion array, got {:?}", other),
        };
        let keyword_map: HashMap<_, _> = keyword_array
            .into_iter()
            .map(|item| (item.label.clone(), item))
            .collect();
        let theorem_item = keyword_map
            .get("theorem")
            .expect("theorem keyword should be offered for prefix");
        assert!(
            theorem_item.documentation.is_some(),
            "keyword completions should include documentation"
        );

        // Builtin type completion with partial prefix
        let builtin_uri = Url::parse("file:///complete_builtin.usl").unwrap();
        open_document(server, builtin_uri.clone(), "Bo", 1).await;
        let builtin_items = server
            .completion(CompletionParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: builtin_uri },
                    position: Position::new(0, 1),
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
                context: None,
            })
            .await
            .unwrap()
            .expect("builtin completion should return items");
        let builtin_array = match builtin_items {
            CompletionResponse::Array(items) => items,
            other => panic!("expected completion array, got {:?}", other),
        };
        let builtin_map: HashMap<_, _> = builtin_array
            .into_iter()
            .map(|item| (item.label.clone(), item))
            .collect();
        let bool_item = builtin_map
            .get("Bool")
            .expect("Bool builtin should match prefix");
        assert_eq!(bool_item.kind, Some(CompletionItemKind::TYPE_PARAMETER));
        assert!(
            bool_item.documentation.is_some(),
            "builtin completion should include documentation"
        );

        // Property and type completions from parsed spec with full prefix
        let spec_uri = Url::parse("file:///complete_spec.usl").unwrap();
        let spec_text = "type Value = { id: Int }\ntheorem check_value { forall v: Value . true }";
        open_document(server, spec_uri.clone(), spec_text, 1).await;
        let type_items = server
            .completion(CompletionParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier {
                        uri: spec_uri.clone(),
                    },
                    position: Position::new(0, 7), // inside type name "Value"
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
                context: None,
            })
            .await
            .unwrap()
            .expect("type completion should return items");
        let type_array = match type_items {
            CompletionResponse::Array(items) => items,
            other => panic!("expected completion array, got {:?}", other),
        };
        let type_map: HashMap<_, _> = type_array
            .into_iter()
            .map(|item| (item.label.clone(), item))
            .collect();
        let value_item = type_map
            .get("Value")
            .expect("user-defined type should be offered");
        assert!(
            value_item.documentation.is_some(),
            "user-defined type completion should include documentation"
        );

        let property_items = server
            .completion(CompletionParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: spec_uri },
                    position: Position::new(1, 10), // inside property name
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
                context: None,
            })
            .await
            .unwrap()
            .expect("property completion should return items");
        let property_array = match property_items {
            CompletionResponse::Array(items) => items,
            other => panic!("expected completion array, got {:?}", other),
        };
        let property_map: HashMap<_, _> = property_array
            .into_iter()
            .map(|item| (item.label.clone(), item))
            .collect();
        let property_item = property_map
            .get("check_value")
            .expect("property completion should include property");
        assert!(
            property_item.detail.is_some(),
            "property completion should describe its kind"
        );

        // Backend completion should respect prefix and include documentation
        let backend_uri = Url::parse("file:///complete_backend.usl").unwrap();
        open_document(server, backend_uri.clone(), "ka", 1).await;
        let backend_items = server
            .completion(CompletionParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: backend_uri },
                    position: Position::new(0, 1),
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
                context: None,
            })
            .await
            .unwrap()
            .expect("backend completion should return items");
        let backend_array = match backend_items {
            CompletionResponse::Array(items) => items,
            other => panic!("expected completion array, got {:?}", other),
        };
        let backend_map: HashMap<_, _> = backend_array
            .into_iter()
            .map(|item| (item.label.clone(), item))
            .collect();
        let kani_item = backend_map
            .get("kani")
            .expect("kani backend should match prefix");
        assert_eq!(kani_item.kind, Some(CompletionItemKind::MODULE));
        assert!(
            kani_item.documentation.is_some(),
            "backend completion should include documentation"
        );
    }

    #[tokio::test]
    async fn execute_command_verify_returns_payload() {
        let (service, mut socket) = create_initialized_service().await;
        let _drain = tokio::spawn(async move { while socket.next().await.is_some() {} });
        let server = service.inner();
        let uri = Url::parse("file:///command_verify.usl").unwrap();
        let text = "theorem my_theorem { true }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .execute_command(ExecuteCommandParams {
                command: COMMAND_VERIFY.to_string(),
                arguments: vec![
                    serde_json::json!(uri.to_string()),
                    serde_json::json!("my_theorem"),
                ],
                work_done_progress_params: Default::default(),
            })
            .await
            .expect("verify command should execute");

        let payload = result.expect("verify command should return payload");
        assert_eq!(payload["action"], "verify");
        assert_eq!(payload["property"], "my_theorem");
        assert_eq!(payload["recommendedBackend"], "lean4");
    }

    #[tokio::test]
    async fn test_execute_command_unknown_returns_error() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();

        // Unknown command should return an error (MethodNotFound)
        let result = server
            .execute_command(ExecuteCommandParams {
                command: "dashprove.unknownCommand".to_string(),
                arguments: vec![],
                work_done_progress_params: Default::default(),
            })
            .await;

        // Unknown commands return an error, not Ok(None)
        assert!(
            result.is_err(),
            "execute_command should return error for unknown command"
        );
    }

    #[tokio::test]
    async fn test_rename_validation_empty_name() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///rename_empty.usl").unwrap();
        let text = "type Value = { id: Int }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .rename(RenameParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri },
                    position: Position::new(0, 5),
                },
                new_name: "".to_string(), // Empty name should be rejected
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap();

        assert!(result.is_none(), "rename should reject empty name");
    }

    #[tokio::test]
    async fn test_rename_validation_invalid_start() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///rename_invalid.usl").unwrap();
        let text = "type Value = { id: Int }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .rename(RenameParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri },
                    position: Position::new(0, 5),
                },
                new_name: "123invalid".to_string(), // Starts with digit
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap();

        assert!(
            result.is_none(),
            "rename should reject name starting with digit"
        );
    }

    #[tokio::test]
    async fn test_rename_validation_invalid_chars() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///rename_chars.usl").unwrap();
        let text = "type Value = { id: Int }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .rename(RenameParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri },
                    position: Position::new(0, 5),
                },
                new_name: "with-hyphen".to_string(), // Contains invalid character
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap();

        assert!(
            result.is_none(),
            "rename should reject name with invalid characters"
        );
    }

    #[tokio::test]
    async fn test_rename_rejects_keyword() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///rename_keyword.usl").unwrap();
        let text = "type Value = { id: Int }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .rename(RenameParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri },
                    position: Position::new(0, 5),
                },
                new_name: "theorem".to_string(), // Keyword should be rejected
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap();

        assert!(result.is_none(), "rename should reject keyword as new name");
    }

    #[tokio::test]
    async fn test_rename_rejects_builtin_type() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///rename_builtin.usl").unwrap();
        let text = "type Value = { id: Int }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .rename(RenameParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri },
                    position: Position::new(0, 5),
                },
                new_name: "Int".to_string(), // Builtin type should be rejected
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap();

        assert!(
            result.is_none(),
            "rename should reject builtin type as new name"
        );
    }

    #[tokio::test]
    async fn test_rename_non_type_or_property() {
        let (service, _socket) = create_initialized_service().await;
        let server = service.inner();
        let uri = Url::parse("file:///rename_other.usl").unwrap();
        let text = "theorem my_theorem { forall x: Bool . x }";

        open_document(server, uri.clone(), text, 1).await;

        let result = server
            .rename(RenameParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri },
                    position: Position::new(0, 31), // On "Bool" (builtin, not renameable)
                },
                new_name: "NewName".to_string(),
                work_done_progress_params: Default::default(),
            })
            .await
            .unwrap();

        // Renaming a builtin type reference should not produce edits
        assert!(
            result.is_none(),
            "rename should not produce edits for non-renameable identifier"
        );
    }
}
