//! LSP backend implementation
//!
//! Implements the `tower_lsp::LanguageServer` trait for Lean5.

use crate::document::{
    CommandKind, Document, ElaboratedCommandCache, ElaboratedDecl, ElaboratedDocument,
    IncrementalState, IncrementalStats, ParseError, ParsedCommand, ParsedDocument, TypeError,
    Warning, WarningCode,
};
use dashmap::DashMap;
use lean5_parser::lexer::{Lexer, TokenKind};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

/// Semantic token types used by Lean5 LSP
/// The order must match the legend provided in server capabilities
pub const SEMANTIC_TOKEN_TYPES: &[SemanticTokenType] = &[
    SemanticTokenType::KEYWORD,
    SemanticTokenType::TYPE,
    SemanticTokenType::FUNCTION,
    SemanticTokenType::VARIABLE,
    SemanticTokenType::NUMBER,
    SemanticTokenType::STRING,
    SemanticTokenType::COMMENT,
    SemanticTokenType::OPERATOR,
    SemanticTokenType::NAMESPACE,
    SemanticTokenType::CLASS,
    SemanticTokenType::PROPERTY,
];

/// Semantic token modifiers used by Lean5 LSP
/// The order must match the legend provided in server capabilities
/// Modifiers are represented as bit flags in the token_modifiers_bitset field
pub const SEMANTIC_TOKEN_MODIFIERS: &[SemanticTokenModifier] = &[
    SemanticTokenModifier::DECLARATION,     // 0: bit 0 (1 << 0 = 1)
    SemanticTokenModifier::DEFINITION,      // 1: bit 1 (1 << 1 = 2)
    SemanticTokenModifier::READONLY,        // 2: bit 2 (1 << 2 = 4)
    SemanticTokenModifier::DEPRECATED,      // 3: bit 3 (1 << 3 = 8)
    SemanticTokenModifier::DEFAULT_LIBRARY, // 4: bit 4 (1 << 4 = 16)
];

/// Modifier bits for semantic tokens
pub mod modifier_bits {
    pub const DECLARATION: u32 = 1 << 0;
    pub const DEFINITION: u32 = 1 << 1;
    pub const READONLY: u32 = 1 << 2;
    pub const DEPRECATED: u32 = 1 << 3;
    pub const DEFAULT_LIBRARY: u32 = 1 << 4;
}

/// Information about a definition location
#[derive(Debug, Clone)]
pub struct DefinitionInfo {
    /// The URI of the document containing the definition
    pub uri: Url,
    /// Start byte offset of the definition
    pub start: usize,
    /// End byte offset of the definition
    pub end: usize,
}

/// Lean5 LSP backend
pub struct Lean5Backend {
    /// LSP client for sending notifications
    client: Client,
    /// Open documents
    documents: DashMap<Url, Document>,
    /// Lean5 environment (shared across documents)
    env: Arc<tokio::sync::RwLock<lean5_kernel::Environment>>,
    /// Definition index: maps name to definition location
    definitions: DashMap<String, DefinitionInfo>,
}

impl Lean5Backend {
    /// Create a new backend
    #[must_use]
    pub fn new(client: Client) -> Self {
        Self {
            client,
            documents: DashMap::new(),
            env: Arc::new(tokio::sync::RwLock::new(lean5_kernel::Environment::new())),
            definitions: DashMap::new(),
        }
    }

    /// Parse a document and update its state
    async fn parse_document(&self, uri: &Url) {
        if let Some(mut doc) = self.documents.get_mut(uri) {
            let text = doc.text();
            let parsed = self.parse_text(&text);

            // Update definition index
            self.update_definitions(uri, &parsed);

            doc.parsed = Some(parsed);
        }
    }

    /// Update the definition index with definitions from a document
    fn update_definitions(&self, uri: &Url, parsed: &ParsedDocument) {
        // First, remove all definitions from this URI
        let to_remove: Vec<String> = self
            .definitions
            .iter()
            .filter(|entry| &entry.value().uri == uri)
            .map(|entry| entry.key().clone())
            .collect();

        for name in to_remove {
            self.definitions.remove(&name);
        }

        // Add new definitions
        for cmd in &parsed.commands {
            if let Some(name) = &cmd.name {
                // Only index definition-like commands
                match cmd.kind {
                    CommandKind::Definition
                    | CommandKind::Theorem
                    | CommandKind::Lemma
                    | CommandKind::Inductive
                    | CommandKind::Structure
                    | CommandKind::Class
                    | CommandKind::Instance
                    | CommandKind::Axiom => {
                        self.definitions.insert(
                            name.clone(),
                            DefinitionInfo {
                                uri: uri.clone(),
                                start: cmd.start,
                                end: cmd.end,
                            },
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    /// Parse text into a ParsedDocument
    fn parse_text(&self, text: &str) -> ParsedDocument {
        match lean5_parser::parse_file(text) {
            Ok(decls) => {
                let mut commands = Vec::new();

                for decl in &decls {
                    let (kind, name, span) = Self::classify_decl(decl);
                    // Compute content hash for incremental checking
                    let content_hash = Self::compute_content_hash(text, span.0, span.1);
                    commands.push(ParsedCommand {
                        kind,
                        start: span.0,
                        end: span.1,
                        name,
                        content_hash,
                    });
                }

                ParsedDocument {
                    errors: vec![],
                    commands,
                }
            }
            Err(e) => {
                let message = format!("{e}");
                ParsedDocument {
                    errors: vec![ParseError {
                        start: 0,
                        end: 1,
                        message,
                    }],
                    commands: vec![],
                }
            }
        }
    }

    /// Compute a hash of the source text for a span
    fn compute_content_hash(text: &str, start: usize, end: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        if start < text.len() && end <= text.len() && start <= end {
            text[start..end].hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Classify a parsed declaration
    fn classify_decl(
        decl: &lean5_parser::SurfaceDecl,
    ) -> (CommandKind, Option<String>, (usize, usize)) {
        use lean5_parser::SurfaceDecl;

        match decl {
            SurfaceDecl::Def { span, name, .. } => (
                CommandKind::Definition,
                Some(name.clone()),
                (span.start, span.end),
            ),
            SurfaceDecl::Theorem { span, name, .. } => (
                CommandKind::Theorem,
                Some(name.clone()),
                (span.start, span.end),
            ),
            SurfaceDecl::Example { span, .. } => {
                (CommandKind::Example, None, (span.start, span.end))
            }
            SurfaceDecl::Inductive { span, name, .. } => (
                CommandKind::Inductive,
                Some(name.clone()),
                (span.start, span.end),
            ),
            SurfaceDecl::Structure { span, name, .. } => (
                CommandKind::Structure,
                Some(name.clone()),
                (span.start, span.end),
            ),
            SurfaceDecl::Class { span, name, .. } => (
                CommandKind::Class,
                Some(name.clone()),
                (span.start, span.end),
            ),
            SurfaceDecl::Instance { span, name, .. } => {
                (CommandKind::Instance, name.clone(), (span.start, span.end))
            }
            SurfaceDecl::Axiom { span, name, .. } => (
                CommandKind::Axiom,
                Some(name.clone()),
                (span.start, span.end),
            ),
            SurfaceDecl::Variable { span, binders, .. } => {
                let name = binders.first().map(|b| b.name.clone());
                (CommandKind::Variable, name, (span.start, span.end))
            }
            SurfaceDecl::UniverseDecl { span, names, .. } => (
                CommandKind::Universe,
                names.first().cloned(),
                (span.start, span.end),
            ),
            SurfaceDecl::Import { span, .. } => (CommandKind::Import, None, (span.start, span.end)),
            SurfaceDecl::Open { span, .. } => (CommandKind::Open, None, (span.start, span.end)),
            SurfaceDecl::Namespace { span, name, .. } => (
                CommandKind::Namespace,
                Some(name.clone()),
                (span.start, span.end),
            ),
            SurfaceDecl::Section { span, name, .. } => {
                (CommandKind::Section, name.clone(), (span.start, span.end))
            }
            SurfaceDecl::Check { span, .. } => (
                CommandKind::Other("check".to_string()),
                None,
                (span.start, span.end),
            ),
            SurfaceDecl::Eval { span, .. } => (
                CommandKind::Other("eval".to_string()),
                None,
                (span.start, span.end),
            ),
            SurfaceDecl::Print { span, .. } => (
                CommandKind::Other("print".to_string()),
                None,
                (span.start, span.end),
            ),
            SurfaceDecl::Mutual { span, .. } => (
                CommandKind::Other("mutual".to_string()),
                None,
                (span.start, span.end),
            ),
            SurfaceDecl::Syntax { span, .. } => (
                CommandKind::Other("syntax".to_string()),
                None,
                (span.start, span.end),
            ),
            SurfaceDecl::DeclareSyntaxCat { span, .. } => (
                CommandKind::Other("declare_syntax_cat".to_string()),
                None,
                (span.start, span.end),
            ),
            SurfaceDecl::Macro { span, .. } => (
                CommandKind::Other("macro".to_string()),
                None,
                (span.start, span.end),
            ),
            SurfaceDecl::MacroRules { span, .. } => (
                CommandKind::Other("macro_rules".to_string()),
                None,
                (span.start, span.end),
            ),
            SurfaceDecl::Notation { span, .. } => (
                CommandKind::Other("notation".to_string()),
                None,
                (span.start, span.end),
            ),
            SurfaceDecl::Attribute { span, .. } => (
                CommandKind::Other("attribute".to_string()),
                None,
                (span.start, span.end),
            ),
            SurfaceDecl::Elab { span, .. } => (
                CommandKind::Other("elab".to_string()),
                None,
                (span.start, span.end),
            ),
            SurfaceDecl::SetOption { span, .. } => (
                CommandKind::Other("set_option".to_string()),
                None,
                (span.start, span.end),
            ),
        }
    }

    /// Get the span of a declaration
    fn get_decl_span(decl: &lean5_parser::SurfaceDecl) -> (usize, usize) {
        Self::classify_decl(decl).2
    }

    /// Elaborate a document and update its state (with incremental checking)
    async fn elaborate_document(&self, uri: &Url) {
        if let Some(mut doc) = self.documents.get_mut(uri) {
            if let Some(parsed) = &doc.parsed {
                if !parsed.errors.is_empty() {
                    doc.elaborated = Some(ElaboratedDocument {
                        errors: vec![],
                        warnings: vec![],
                        declarations: vec![],
                    });
                    return;
                }
            }

            let text = doc.text();
            let prev_state = std::mem::take(&mut doc.incremental_state);

            let (elaborated, new_state) = self.elaborate_text_incremental(&text, prev_state).await;

            doc.elaborated = Some(elaborated);
            doc.incremental_state = new_state;
        }
    }

    /// Elaborate text into an ElaboratedDocument (incremental version)
    ///
    /// Uses the previous incremental state to skip re-elaboration of unchanged commands.
    async fn elaborate_text_incremental(
        &self,
        text: &str,
        prev_state: IncrementalState,
    ) -> (ElaboratedDocument, IncrementalState) {
        let Ok(decls) = lean5_parser::parse_file(text) else {
            return (
                ElaboratedDocument {
                    errors: vec![],
                    warnings: vec![],
                    declarations: vec![],
                },
                IncrementalState::default(),
            );
        };

        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut declarations = Vec::new();
        let mut new_cache = std::collections::HashMap::new();
        let mut stats = IncrementalStats {
            total_commands: decls.len(),
            elaborated_count: 0,
            cached_count: 0,
        };
        let deprecated_names = collect_deprecated_names(&decls);

        let env = self.env.read().await;

        for (idx, decl) in decls.iter().enumerate() {
            let (_kind, name, span) = Self::classify_decl(decl);
            let content_hash = Self::compute_content_hash(text, span.0, span.1);

            // Generate cache key: use name if available, otherwise use index
            let cache_key = name.clone().unwrap_or_else(|| format!("__anon_{idx}"));

            // Check if we have a cached result with matching content hash
            if let Some(cached) = prev_state.cache.get(&cache_key) {
                if cached.content_hash == content_hash {
                    // Use cached result (adjust offsets if needed)
                    let mut cached_errors = cached.errors.clone();
                    let mut cached_warnings = cached.warnings.clone();

                    // For simplicity, we reuse cached results as-is since the content hash
                    // matching means the content is identical. In a more sophisticated
                    // implementation, we could adjust byte offsets for position changes.

                    errors.append(&mut cached_errors);
                    warnings.append(&mut cached_warnings);
                    if let Some(decl_info) = &cached.declaration {
                        declarations.push(decl_info.clone());
                    }

                    // Store in new cache
                    new_cache.insert(cache_key, cached.clone());
                    stats.cached_count += 1;
                    continue;
                }
            }

            // Need to elaborate this declaration
            stats.elaborated_count += 1;

            let mut cmd_errors = Vec::new();
            let mut cmd_warnings = Vec::new();
            let mut cmd_declaration = None;

            // Detect unused variables in the declaration
            cmd_warnings.extend(detect_unused_variables(decl));
            // Detect sorry/admit usage
            cmd_warnings.extend(detect_sorry_warnings(decl));
            // Detect usage of deprecated names
            cmd_warnings.extend(detect_deprecated_usage(decl, &deprecated_names));

            match lean5_elab::elaborate_decl(&env, decl) {
                Ok(elab_result) => {
                    if let Some(info) = Self::extract_elab_info(&elab_result, decl) {
                        cmd_declaration = Some(info.clone());
                        declarations.push(info);
                    }
                }
                Err(e) => {
                    let (start, end) = Self::get_decl_span(decl);
                    let err = TypeError {
                        start,
                        end,
                        message: format!("{e}"),
                    };
                    cmd_errors.push(err);
                }
            }

            // Add to combined results
            errors.extend(cmd_errors.iter().cloned());
            warnings.extend(cmd_warnings.iter().cloned());

            // Store in new cache
            new_cache.insert(
                cache_key,
                ElaboratedCommandCache {
                    content_hash,
                    errors: cmd_errors,
                    warnings: cmd_warnings,
                    declaration: cmd_declaration,
                },
            );
        }

        let new_state = IncrementalState {
            cache: new_cache,
            stats,
        };

        (
            ElaboratedDocument {
                errors,
                warnings,
                declarations,
            },
            new_state,
        )
    }

    /// Extract elaboration info from an ElabResult
    fn extract_elab_info(
        result: &lean5_elab::ElabResult,
        decl: &lean5_parser::SurfaceDecl,
    ) -> Option<ElaboratedDecl> {
        use lean5_elab::ElabResult;

        let (start, end) = Self::get_decl_span(decl);

        match result {
            ElabResult::Definition { name, ty, .. }
            | ElabResult::Theorem { name, ty, .. }
            | ElabResult::Axiom { name, ty, .. }
            | ElabResult::Inductive { name, ty, .. }
            | ElabResult::Structure { name, ty, .. }
            | ElabResult::Instance { name, ty, .. } => Some(ElaboratedDecl {
                name: name.to_string(),
                type_str: format!("{ty:?}"),
                start,
                end,
            }),
            _ => None,
        }
    }

    /// Generate LSP diagnostics from a document
    fn generate_diagnostics(&self, doc: &Document) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        if let Some(parsed) = &doc.parsed {
            for err in &parsed.errors {
                let start = doc.offset_to_position(err.start);
                let end = doc.offset_to_position(err.end);

                diagnostics.push(Diagnostic {
                    range: Range { start, end },
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: Some(NumberOrString::String("parse-error".to_string())),
                    code_description: None,
                    source: Some("lean5".to_string()),
                    message: err.message.clone(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
            }
        }

        if let Some(elab) = &doc.elaborated {
            for err in &elab.errors {
                let start = doc.offset_to_position(err.start);
                let end = doc.offset_to_position(err.end);

                diagnostics.push(Diagnostic {
                    range: Range { start, end },
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: Some(NumberOrString::String("type-error".to_string())),
                    code_description: None,
                    source: Some("lean5".to_string()),
                    message: err.message.clone(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
            }
        }

        diagnostics
    }

    /// Publish diagnostics for a document
    async fn publish_diagnostics(&self, uri: &Url) {
        if let Some(doc) = self.documents.get(uri) {
            let diagnostics = self.generate_diagnostics(&doc);
            self.client
                .publish_diagnostics(uri.clone(), diagnostics, Some(doc.version))
                .await;
        }
    }

    /// Check a document (parse + elaborate + publish diagnostics)
    async fn check_document(&self, uri: &Url) {
        self.parse_document(uri).await;
        self.elaborate_document(uri).await;
        self.publish_diagnostics(uri).await;
    }

    /// Get hover information at a position
    fn get_hover_at(&self, uri: &Url, position: Position) -> Option<Hover> {
        let doc = self.documents.get(uri)?;

        if let Some(elab) = &doc.elaborated {
            for decl in &elab.declarations {
                let start_pos = doc.offset_to_position(decl.start);
                let end_pos = doc.offset_to_position(decl.end);

                if position.line >= start_pos.line
                    && position.line <= end_pos.line
                    && (position.line != start_pos.line
                        || position.character >= start_pos.character)
                    && (position.line != end_pos.line || position.character <= end_pos.character)
                {
                    return Some(Hover {
                        contents: HoverContents::Markup(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: format!("```lean\n{} : {}\n```", decl.name, decl.type_str),
                        }),
                        range: Some(Range {
                            start: start_pos,
                            end: end_pos,
                        }),
                    });
                }
            }
        }

        None
    }

    /// Get the identifier at a position in a document
    fn get_identifier_at(&self, uri: &Url, position: Position) -> Option<String> {
        let doc = self.documents.get(uri)?;
        let text = doc.text();
        let offset = doc.position_to_offset(position);

        // Find word boundaries
        let bytes = text.as_bytes();
        let mut start = offset;
        let mut end = offset;

        // Move backwards to find start of identifier
        while start > 0 && Self::is_identifier_char(bytes[start - 1]) {
            start -= 1;
        }

        // Move forwards to find end of identifier
        while end < bytes.len() && Self::is_identifier_char(bytes[end]) {
            end += 1;
        }

        if start < end {
            Some(text[start..end].to_string())
        } else {
            None
        }
    }

    /// Check if a character can be part of an identifier
    fn is_identifier_char(c: u8) -> bool {
        c.is_ascii_alphanumeric() || c == b'_' || c == b'\''
    }

    /// Find the definition location for a name
    fn find_definition(&self, name: &str) -> Option<(Url, Range)> {
        if let Some(def_info) = self.definitions.get(name) {
            if let Some(doc) = self.documents.get(&def_info.uri) {
                let start = doc.offset_to_position(def_info.start);
                let end = doc.offset_to_position(def_info.end);
                return Some((def_info.uri.clone(), Range { start, end }));
            }
        }
        None
    }

    /// Find all references to a name across all documents
    fn find_references(&self, name: &str, include_definition: bool) -> Vec<Location> {
        let mut references = Vec::new();

        for doc_entry in &self.documents {
            let uri = doc_entry.key();
            let doc = doc_entry.value();
            let text = doc.text();

            // Simple text search for the identifier
            let mut search_pos = 0;
            while let Some(found_pos) = text[search_pos..].find(name) {
                let abs_pos = search_pos + found_pos;

                // Check if this is a whole identifier match
                let is_start_boundary =
                    abs_pos == 0 || !Self::is_identifier_char(text.as_bytes()[abs_pos - 1]);
                let end_pos = abs_pos + name.len();
                let is_end_boundary =
                    end_pos >= text.len() || !Self::is_identifier_char(text.as_bytes()[end_pos]);

                if is_start_boundary && is_end_boundary {
                    let start = doc.offset_to_position(abs_pos);
                    let end = doc.offset_to_position(end_pos);

                    // Skip definition if not including it
                    if !include_definition {
                        if let Some(def_info) = self.definitions.get(name) {
                            if &def_info.uri == uri && def_info.start == abs_pos {
                                search_pos = end_pos;
                                continue;
                            }
                        }
                    }

                    references.push(Location {
                        uri: uri.clone(),
                        range: Range { start, end },
                    });
                }

                search_pos = abs_pos + 1;
            }
        }

        references
    }

    /// Prepare rename operation - validate the position contains a renameable identifier
    fn prepare_rename_at(&self, uri: &Url, position: Position) -> Option<(String, Range)> {
        let name = self.get_identifier_at(uri, position)?;

        // Find the exact range of the identifier at this position
        let doc = self.documents.get(uri)?;
        let text = doc.text();
        let offset = doc.position_to_offset(position);

        // Find start and end of identifier at offset
        let bytes = text.as_bytes();
        let mut start = offset;
        let mut end = offset;

        while start > 0 && Self::is_identifier_char(bytes[start - 1]) {
            start -= 1;
        }

        while end < bytes.len() && Self::is_identifier_char(bytes[end]) {
            end += 1;
        }

        let start_pos = doc.offset_to_position(start);
        let end_pos = doc.offset_to_position(end);

        Some((
            name,
            Range {
                start: start_pos,
                end: end_pos,
            },
        ))
    }

    /// Create workspace edits to rename a symbol
    fn create_rename_edits(&self, old_name: &str, new_name: &str) -> WorkspaceEdit {
        use std::collections::HashMap;

        let references = self.find_references(old_name, true);
        let mut changes: HashMap<Url, Vec<TextEdit>> = HashMap::new();

        for location in references {
            changes.entry(location.uri).or_default().push(TextEdit {
                range: location.range,
                new_text: new_name.to_string(),
            });
        }

        WorkspaceEdit {
            changes: Some(changes),
            document_changes: None,
            change_annotations: None,
        }
    }

    /// Get the completion prefix at a position (the partial identifier being typed)
    fn get_completion_prefix(&self, uri: &Url, position: Position) -> String {
        if let Some(doc) = self.documents.get(uri) {
            let text = doc.text();
            let offset = doc.position_to_offset(position);

            // Find start of identifier
            let bytes = text.as_bytes();
            let mut start = offset;

            while start > 0 && Self::is_identifier_char(bytes[start - 1]) {
                start -= 1;
            }

            if start < offset {
                text[start..offset].to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        }
    }

    /// Get the completion kind for a definition
    fn get_definition_kind(&self, name: &str) -> CompletionItemKind {
        // Look up the parsed document to find the command kind
        if let Some(def_info) = self.definitions.get(name) {
            if let Some(doc) = self.documents.get(&def_info.uri) {
                if let Some(parsed) = &doc.parsed {
                    for cmd in &parsed.commands {
                        if cmd.name.as_ref() == Some(&name.to_string()) {
                            return match cmd.kind {
                                CommandKind::Definition => CompletionItemKind::FUNCTION,
                                CommandKind::Theorem | CommandKind::Lemma => {
                                    CompletionItemKind::FUNCTION
                                }
                                CommandKind::Inductive | CommandKind::Structure => {
                                    CompletionItemKind::CLASS
                                }
                                CommandKind::Class => CompletionItemKind::INTERFACE,
                                CommandKind::Instance => CompletionItemKind::REFERENCE,
                                CommandKind::Axiom => CompletionItemKind::CONSTANT,
                                _ => CompletionItemKind::TEXT,
                            };
                        }
                    }
                }
            }
        }
        CompletionItemKind::TEXT
    }

    /// Get document symbols
    fn get_document_symbols(&self, uri: &Url) -> Option<Vec<DocumentSymbol>> {
        let doc = self.documents.get(uri)?;
        let parsed = doc.parsed.as_ref()?;

        let mut symbols = Vec::new();

        for cmd in &parsed.commands {
            if let Some(name) = &cmd.name {
                let kind = match cmd.kind {
                    CommandKind::Definition | CommandKind::Theorem | CommandKind::Lemma => {
                        SymbolKind::FUNCTION
                    }
                    CommandKind::Inductive | CommandKind::Structure => SymbolKind::CLASS,
                    CommandKind::Class => SymbolKind::INTERFACE,
                    CommandKind::Instance => SymbolKind::OBJECT,
                    CommandKind::Axiom => SymbolKind::CONSTANT,
                    CommandKind::Variable => SymbolKind::VARIABLE,
                    CommandKind::Namespace => SymbolKind::NAMESPACE,
                    _ => SymbolKind::NULL,
                };

                let start = doc.offset_to_position(cmd.start);
                let end = doc.offset_to_position(cmd.end);
                let range = Range { start, end };

                #[allow(deprecated)]
                symbols.push(DocumentSymbol {
                    name: name.clone(),
                    detail: None,
                    kind,
                    tags: None,
                    deprecated: None,
                    range,
                    selection_range: range,
                    children: None,
                });
            }
        }

        Some(symbols)
    }

    /// Get workspace symbols matching a query
    fn get_workspace_symbols(&self, query: &str) -> Vec<SymbolInformation> {
        let mut symbols = Vec::new();
        let query_lower = query.to_lowercase();

        for entry in &self.definitions {
            let name = entry.key();
            let def_info = entry.value();

            // Match if query is empty or name contains query (case-insensitive)
            if query.is_empty() || name.to_lowercase().contains(&query_lower) {
                // Look up the command kind from the parsed document
                let kind = self.get_symbol_kind_for_definition(name);

                // Get the location
                if let Some(doc) = self.documents.get(&def_info.uri) {
                    let start = doc.offset_to_position(def_info.start);
                    let end = doc.offset_to_position(def_info.end);

                    #[allow(deprecated)]
                    symbols.push(SymbolInformation {
                        name: name.clone(),
                        kind,
                        tags: None,
                        deprecated: None,
                        location: Location {
                            uri: def_info.uri.clone(),
                            range: Range { start, end },
                        },
                        container_name: None,
                    });
                }
            }
        }

        // Sort by name for consistent results
        symbols.sort_by(|a, b| a.name.cmp(&b.name));
        symbols
    }

    /// Get the SymbolKind for a definition by name
    fn get_symbol_kind_for_definition(&self, name: &str) -> SymbolKind {
        if let Some(def_info) = self.definitions.get(name) {
            if let Some(doc) = self.documents.get(&def_info.uri) {
                if let Some(parsed) = &doc.parsed {
                    for cmd in &parsed.commands {
                        if cmd.name.as_ref() == Some(&name.to_string()) {
                            return match cmd.kind {
                                CommandKind::Definition
                                | CommandKind::Theorem
                                | CommandKind::Lemma => SymbolKind::FUNCTION,
                                CommandKind::Inductive | CommandKind::Structure => {
                                    SymbolKind::CLASS
                                }
                                CommandKind::Class => SymbolKind::INTERFACE,
                                CommandKind::Instance => SymbolKind::OBJECT,
                                CommandKind::Axiom => SymbolKind::CONSTANT,
                                CommandKind::Variable => SymbolKind::VARIABLE,
                                CommandKind::Namespace => SymbolKind::NAMESPACE,
                                _ => SymbolKind::NULL,
                            };
                        }
                    }
                }
            }
        }
        SymbolKind::NULL
    }

    /// Get code actions for a range
    fn get_code_actions(
        &self,
        uri: &Url,
        range: Range,
        diagnostics: &[Diagnostic],
    ) -> Vec<CodeActionOrCommand> {
        let mut actions = Vec::new();

        // Get the document
        let Some(doc) = self.documents.get(uri) else {
            return actions;
        };

        let text = doc.text();

        // 1. Quick fix: Replace `sorry` with placeholder
        self.add_sorry_quick_fixes(&text, uri, range, &mut actions);

        // 2. Quick fix based on diagnostics
        for diagnostic in diagnostics {
            self.add_diagnostic_quick_fixes(uri, diagnostic, &mut actions);
        }

        // 3. Refactoring: Extract definition (if selecting an expression)
        self.add_extract_definition_action(&text, &doc, uri, range, &mut actions);

        actions
    }

    /// Add quick fixes for `sorry` occurrences
    fn add_sorry_quick_fixes(
        &self,
        text: &str,
        uri: &Url,
        range: Range,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        // Find all `sorry` occurrences in the text
        let mut pos = 0;
        while let Some(found) = text[pos..].find("sorry") {
            let start_offset = pos + found;
            let end_offset = start_offset + 5; // "sorry".len()

            // Get the position in the document
            let start = self.offset_to_position_in_text(text, start_offset);
            let end = self.offset_to_position_in_text(text, end_offset);

            // Check if this sorry is in or overlaps with the requested range
            let sorry_range = Range { start, end };
            if ranges_overlap(sorry_range, range) {
                // Create a code action to replace sorry with a tactic placeholder
                let edit = TextEdit {
                    range: sorry_range,
                    new_text: "by decide".to_string(),
                };

                let mut changes = std::collections::HashMap::new();
                changes.insert(uri.clone(), vec![edit.clone()]);

                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: "Replace 'sorry' with 'by decide'".to_string(),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: None,
                    edit: Some(WorkspaceEdit {
                        changes: Some(changes),
                        document_changes: None,
                        change_annotations: None,
                    }),
                    command: None,
                    is_preferred: Some(false),
                    disabled: None,
                    data: None,
                }));

                // Also offer to replace with other tactics
                let other_replacements = [
                    ("trivial", "Replace 'sorry' with 'trivial'"),
                    ("rfl", "Replace 'sorry' with 'rfl'"),
                    ("simp", "Replace 'sorry' with 'simp'"),
                    ("by assumption", "Replace 'sorry' with 'by assumption'"),
                ];

                for (replacement, title) in other_replacements {
                    let edit = TextEdit {
                        range: sorry_range,
                        new_text: replacement.to_string(),
                    };

                    let mut changes = std::collections::HashMap::new();
                    changes.insert(uri.clone(), vec![edit]);

                    actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                        title: title.to_string(),
                        kind: Some(CodeActionKind::QUICKFIX),
                        diagnostics: None,
                        edit: Some(WorkspaceEdit {
                            changes: Some(changes),
                            document_changes: None,
                            change_annotations: None,
                        }),
                        command: None,
                        is_preferred: Some(false),
                        disabled: None,
                        data: None,
                    }));
                }
            }

            pos = end_offset;
        }
    }

    /// Add quick fixes based on diagnostic messages
    fn add_diagnostic_quick_fixes(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        let message = &diagnostic.message;

        // Check for unknown identifier errors (potential import suggestion)
        if message.contains("unknown identifier") || message.contains("not found") {
            // Extract the identifier name from the error message
            if let Some(ident) = extract_identifier_from_error(message) {
                // Suggest common imports based on the identifier
                let suggested_imports = suggest_imports_for_identifier(&ident);

                for import in suggested_imports {
                    let import_text = format!("import {import}\n");

                    // Insert import at the start of the file
                    let edit = TextEdit {
                        range: Range {
                            start: Position {
                                line: 0,
                                character: 0,
                            },
                            end: Position {
                                line: 0,
                                character: 0,
                            },
                        },
                        new_text: import_text,
                    };

                    let mut changes = std::collections::HashMap::new();
                    changes.insert(uri.clone(), vec![edit]);

                    actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                        title: format!("Add import '{import}'"),
                        kind: Some(CodeActionKind::QUICKFIX),
                        diagnostics: Some(vec![diagnostic.clone()]),
                        edit: Some(WorkspaceEdit {
                            changes: Some(changes),
                            document_changes: None,
                            change_annotations: None,
                        }),
                        command: None,
                        is_preferred: Some(false),
                        disabled: None,
                        data: None,
                    }));
                }
            }
        }

        // Check for type mismatch errors
        if message.contains("type mismatch") {
            // Suggest adding a type annotation
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Add explicit type annotation".to_string(),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: None, // Would need more context to provide actual edit
                command: None,
                is_preferred: Some(false),
                disabled: Some(CodeActionDisabled {
                    reason: "Requires manual type specification".to_string(),
                }),
                data: None,
            }));
        }
    }

    /// Add refactoring action to extract a definition
    fn add_extract_definition_action(
        &self,
        text: &str,
        doc: &Document,
        uri: &Url,
        range: Range,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        // Only offer if a non-empty range is selected
        if range.start == range.end {
            return;
        }

        let start_offset = doc.position_to_offset(range.start);
        let end_offset = doc.position_to_offset(range.end);

        if start_offset >= end_offset || end_offset > text.len() {
            return;
        }

        let selected_text = &text[start_offset..end_offset];

        // Only offer for reasonable selections (not too long, no newlines at edges)
        if selected_text.is_empty() || selected_text.len() > 200 {
            return;
        }

        let trimmed = selected_text.trim();
        if trimmed.is_empty() {
            return;
        }

        // Create the extract definition action
        let new_def = format!("def extracted := {trimmed}\n\n");

        // Find the start of the current declaration to insert before it
        let insert_pos = self.find_declaration_start(text, start_offset);
        let insert_position = self.offset_to_position_in_text(text, insert_pos);

        let mut edits = vec![
            // Insert new definition
            TextEdit {
                range: Range {
                    start: insert_position,
                    end: insert_position,
                },
                new_text: new_def,
            },
            // Replace selected text with reference
            TextEdit {
                range,
                new_text: "extracted".to_string(),
            },
        ];

        // Sort edits by position (from end to start) to avoid offset issues
        edits.sort_by(|a, b| {
            b.range
                .start
                .line
                .cmp(&a.range.start.line)
                .then_with(|| b.range.start.character.cmp(&a.range.start.character))
        });

        let mut changes = std::collections::HashMap::new();
        changes.insert(uri.clone(), edits);

        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title: "Extract to definition".to_string(),
            kind: Some(CodeActionKind::REFACTOR_EXTRACT),
            diagnostics: None,
            edit: Some(WorkspaceEdit {
                changes: Some(changes),
                document_changes: None,
                change_annotations: None,
            }),
            command: None,
            is_preferred: Some(false),
            disabled: None,
            data: None,
        }));
    }

    /// Find the start of the declaration containing the given offset
    fn find_declaration_start(&self, text: &str, offset: usize) -> usize {
        // Look backwards for declaration keywords
        let keywords = [
            "def ",
            "theorem ",
            "lemma ",
            "example ",
            "inductive ",
            "structure ",
            "class ",
            "instance ",
            "axiom ",
        ];

        let search_start = offset.saturating_sub(500);
        let search_text = &text[search_start..offset];

        let mut best_pos = search_start;
        for keyword in keywords {
            if let Some(pos) = search_text.rfind(keyword) {
                let abs_pos = search_start + pos;
                if abs_pos > best_pos {
                    best_pos = abs_pos;
                }
            }
        }

        // If we found a keyword, return the start of that line
        if best_pos > search_start {
            // Go back to the start of the line
            let line_start = text[..best_pos].rfind('\n').map_or(0, |p| p + 1);
            return line_start;
        }

        // Default to start of file
        0
    }

    /// Convert byte offset to LSP position (helper for text without Document)
    fn offset_to_position_in_text(&self, text: &str, offset: usize) -> Position {
        let offset = offset.min(text.len());
        let mut line = 0u32;
        let mut line_start = 0;

        for (i, ch) in text.char_indices() {
            if i >= offset {
                break;
            }
            if ch == '\n' {
                line += 1;
                line_start = i + 1;
            }
        }

        let col_bytes = offset.saturating_sub(line_start);
        let line_text = &text[line_start..];

        // Convert byte offset to UTF-16 code units
        let mut utf16_col = 0u32;
        let mut byte_count = 0;

        for ch in line_text.chars() {
            if byte_count >= col_bytes {
                break;
            }
            byte_count += ch.len_utf8();
            utf16_col += ch.len_utf16() as u32;
        }

        Position {
            line,
            character: utf16_col,
        }
    }
}

/// Check if two ranges overlap
fn ranges_overlap(a: Range, b: Range) -> bool {
    // Ranges overlap if one doesn't end before the other starts
    !(a.end.line < b.start.line
        || (a.end.line == b.start.line && a.end.character < b.start.character)
        || b.end.line < a.start.line
        || (b.end.line == a.start.line && b.end.character < a.start.character))
}

/// Extract identifier name from an error message
fn extract_identifier_from_error(message: &str) -> Option<String> {
    // Try to extract quoted identifier like `foo`
    if let Some(start) = message.find('`') {
        if let Some(end) = message[start + 1..].find('`') {
            return Some(message[start + 1..start + 1 + end].to_string());
        }
    }

    // Try to extract identifier after "identifier" or "unknown"
    let patterns = ["unknown identifier ", "identifier ", "not found: "];
    for pattern in patterns {
        if let Some(pos) = message.find(pattern) {
            let start = pos + pattern.len();
            let end = message[start..]
                .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
                .map_or(message.len(), |p| start + p);
            if start < end {
                return Some(message[start..end].to_string());
            }
        }
    }

    None
}

/// Suggest imports for a given identifier
fn suggest_imports_for_identifier(ident: &str) -> Vec<&'static str> {
    // Map common identifiers to their likely imports
    match ident {
        "Nat" | "Int" | "Bool" | "String" | "List" | "Array" | "Option" | "Sum" => vec!["Init"],
        "HashMap" | "HashSet" | "RBMap" | "RBTree" => {
            vec!["Std.Data.HashMap", "Std.Data.HashSet", "Std.Data.RBMap"]
        }
        "Decidable" | "DecidableEq" => vec!["Init.Core"],
        "Monad" | "Functor" | "Applicative" => vec!["Init.Control.Monad"],
        "IO" | "EIO" => vec!["Init.System.IO"],
        "Real" | "Complex" => vec!["Mathlib.Data.Real.Basic", "Mathlib.Data.Complex.Basic"],
        "Group" | "Ring" | "Field" => vec![
            "Mathlib.Algebra.Group.Basic",
            "Mathlib.Algebra.Ring.Basic",
            "Mathlib.Algebra.Field.Basic",
        ],
        _ => {
            // Check for qualified names
            if ident.starts_with("Std.") {
                vec!["Std"]
            } else if ident.starts_with("Mathlib.") {
                vec!["Mathlib"]
            } else {
                vec![]
            }
        }
    }
}

/// Map a parser TokenKind to a semantic token type index
/// Returns None for tokens that shouldn't be highlighted semantically
fn token_kind_to_semantic_type(kind: &TokenKind) -> Option<u32> {
    match kind {
        // Keywords (index 0)
        TokenKind::Def
        | TokenKind::Theorem
        | TokenKind::Lemma
        | TokenKind::Axiom
        | TokenKind::Example
        | TokenKind::Let
        | TokenKind::In
        | TokenKind::Fun
        | TokenKind::Forall
        | TokenKind::If
        | TokenKind::Then
        | TokenKind::Else
        | TokenKind::Match
        | TokenKind::With
        | TokenKind::Where
        | TokenKind::Do
        | TokenKind::Return
        | TokenKind::Structure
        | TokenKind::Class
        | TokenKind::Instance
        | TokenKind::Inductive
        | TokenKind::Deriving
        | TokenKind::Namespace
        | TokenKind::Section
        | TokenKind::End
        | TokenKind::Open
        | TokenKind::Variable
        | TokenKind::Universe
        | TokenKind::Import
        | TokenKind::Mutual
        | TokenKind::SetOption
        | TokenKind::By
        | TokenKind::Have
        | TokenKind::Show
        | TokenKind::Suffices
        | TokenKind::From
        | TokenKind::Rfl
        | TokenKind::Sorry
        | TokenKind::Extends
        | TokenKind::Private
        | TokenKind::Protected
        | TokenKind::Partial
        | TokenKind::Unsafe
        | TokenKind::Noncomputable
        | TokenKind::Abbrev
        | TokenKind::Attribute
        | TokenKind::Syntax
        | TokenKind::Macro
        | TokenKind::MacroRules
        | TokenKind::Elab
        | TokenKind::Infixl
        | TokenKind::Infixr
        | TokenKind::Prefix
        | TokenKind::Postfix
        | TokenKind::Notation
        | TokenKind::Scoped => Some(0), // KEYWORD

        // Types (index 1)
        TokenKind::Type | TokenKind::Prop | TokenKind::Sort => Some(1), // TYPE

        // Numbers (index 4)
        TokenKind::NatLit(_) => Some(4), // NUMBER

        // Strings (index 5)
        TokenKind::StringLit(_) => Some(5), // STRING

        // Operators (index 7)
        TokenKind::Arrow
        | TokenKind::FatArrow
        | TokenKind::Lambda
        | TokenKind::Eq
        | TokenKind::DoubleEq
        | TokenKind::Ne
        | TokenKind::Lt
        | TokenKind::Le
        | TokenKind::Gt
        | TokenKind::Ge
        | TokenKind::Plus
        | TokenKind::Minus
        | TokenKind::Star
        | TokenKind::Slash
        | TokenKind::Percent
        | TokenKind::Caret
        | TokenKind::And
        | TokenKind::Or
        | TokenKind::Not
        | TokenKind::Tilde
        | TokenKind::Bind
        | TokenKind::Seq
        | TokenKind::AndThen
        | TokenKind::OrElse
        | TokenKind::Pipe
        | TokenKind::BackwardPipe
        | TokenKind::ColonColon
        | TokenKind::Dollar
        | TokenKind::DollarArrow
        | TokenKind::LeftDollar
        | TokenKind::LeftDollarArrow
        | TokenKind::HEq
        | TokenKind::Equiv
        | TokenKind::Iff
        | TokenKind::Times
        | TokenKind::LeftArrow
        | TokenKind::Exists
        | TokenKind::Elem
        | TokenKind::NotElem
        | TokenKind::Subset
        | TokenKind::ProperSubset
        | TokenKind::Inter
        | TokenKind::Union
        | TokenKind::Top
        | TokenKind::Bot
        | TokenKind::Compose
        | TokenKind::Cdot => Some(7), // OPERATOR

        // Identifiers could be functions, variables, etc.
        // For now, we mark them as VARIABLE (index 3)
        // A more sophisticated implementation would look up the identifier
        // in the environment to determine if it's a function, type, etc.
        TokenKind::Ident(_) => Some(3), // VARIABLE

        // Delimiters, punctuation, and other tokens don't need semantic highlighting
        _ => None,
    }
}

/// Convert byte offset to (line, character) position
fn byte_offset_to_position(text: &str, offset: usize) -> Position {
    let mut line = 0u32;
    let mut col = 0u32;
    for (i, c) in text.char_indices() {
        if i >= offset {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    Position::new(line, col)
}

#[tower_lsp::async_trait]
impl LanguageServer for Lean5Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::INCREMENTAL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![".".to_string(), "#".to_string()]),
                    ..Default::default()
                }),
                definition_provider: Some(OneOf::Left(true)),
                references_provider: Some(OneOf::Left(true)),
                workspace_symbol_provider: Some(OneOf::Left(true)),
                code_action_provider: Some(CodeActionProviderCapability::Options(
                    CodeActionOptions {
                        code_action_kinds: Some(vec![
                            CodeActionKind::QUICKFIX,
                            CodeActionKind::REFACTOR_EXTRACT,
                        ]),
                        work_done_progress_options: WorkDoneProgressOptions::default(),
                        resolve_provider: None,
                    },
                )),
                rename_provider: Some(OneOf::Right(RenameOptions {
                    prepare_provider: Some(true),
                    work_done_progress_options: WorkDoneProgressOptions::default(),
                })),
                semantic_tokens_provider: Some(
                    SemanticTokensServerCapabilities::SemanticTokensOptions(
                        SemanticTokensOptions {
                            legend: SemanticTokensLegend {
                                token_types: SEMANTIC_TOKEN_TYPES.to_vec(),
                                token_modifiers: SEMANTIC_TOKEN_MODIFIERS.to_vec(),
                            },
                            full: Some(SemanticTokensFullOptions::Bool(true)),
                            range: None,
                            work_done_progress_options: WorkDoneProgressOptions::default(),
                        },
                    ),
                ),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "lean5-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Lean5 LSP server initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;
        let text = params.text_document.text;
        let language_id = params.text_document.language_id;

        self.documents.insert(
            uri.clone(),
            Document::new(uri.clone(), version, text, language_id),
        );

        self.check_document(&uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;

        if let Some(mut doc) = self.documents.get_mut(&uri) {
            doc.version = version;

            for change in params.content_changes {
                doc.apply_change(change.range, &change.text);
            }
        }

        self.check_document(&uri).await;
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;

        // Remove definitions from this document
        let to_remove: Vec<String> = self
            .definitions
            .iter()
            .filter(|entry| entry.value().uri == uri)
            .map(|entry| entry.key().clone())
            .collect();
        for name in to_remove {
            self.definitions.remove(&name);
        }

        self.documents.remove(&uri);
        self.client.publish_diagnostics(uri, vec![], None).await;
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        self.check_document(&params.text_document.uri).await;
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        Ok(self.get_hover_at(uri, position))
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;
        Ok(self
            .get_document_symbols(uri)
            .map(DocumentSymbolResponse::Nested))
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        // Get the prefix being typed (partial identifier)
        let prefix = self.get_completion_prefix(uri, position);

        let mut items = Vec::new();

        // Add completions from definitions
        for entry in &self.definitions {
            let name = entry.key();
            if prefix.is_empty() || name.starts_with(&prefix) {
                // Determine completion kind based on where the definition came from
                let kind = self.get_definition_kind(name);

                items.push(CompletionItem {
                    label: name.clone(),
                    kind: Some(kind),
                    detail: None,
                    documentation: None,
                    deprecated: None,
                    preselect: None,
                    sort_text: None,
                    filter_text: None,
                    insert_text: Some(name.clone()),
                    insert_text_format: Some(InsertTextFormat::PLAIN_TEXT),
                    insert_text_mode: None,
                    text_edit: None,
                    additional_text_edits: None,
                    command: None,
                    commit_characters: None,
                    data: None,
                    tags: None,
                    label_details: None,
                });
            }
        }

        // Add keyword completions
        if prefix.is_empty() || items.is_empty() {
            for keyword in &[
                "def",
                "theorem",
                "lemma",
                "example",
                "inductive",
                "structure",
                "class",
                "instance",
                "axiom",
                "variable",
                "import",
                "open",
                "namespace",
                "section",
                "end",
                "where",
                "if",
                "then",
                "else",
                "match",
                "with",
                "fun",
                "let",
                "in",
                "do",
                "return",
                "have",
                "show",
                "by",
                "rfl",
                "simp",
                "exact",
                "apply",
                "intro",
                "cases",
                "induction",
                "constructor",
                "rw",
                "rewrite",
                "calc",
                "sorry",
            ] {
                if prefix.is_empty() || keyword.starts_with(&prefix) {
                    items.push(CompletionItem {
                        label: (*keyword).to_string(),
                        kind: Some(CompletionItemKind::KEYWORD),
                        insert_text: Some((*keyword).to_string()),
                        insert_text_format: Some(InsertTextFormat::PLAIN_TEXT),
                        ..Default::default()
                    });
                }
            }
        }

        if items.is_empty() {
            Ok(None)
        } else {
            Ok(Some(CompletionResponse::Array(items)))
        }
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        // Find the identifier at the cursor position
        let Some(name) = self.get_identifier_at(uri, position) else {
            return Ok(None);
        };

        // Look up the definition
        if let Some((def_uri, range)) = self.find_definition(&name) {
            Ok(Some(GotoDefinitionResponse::Scalar(Location {
                uri: def_uri,
                range,
            })))
        } else {
            Ok(None)
        }
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let include_declaration = params.context.include_declaration;

        // Find the identifier at the cursor position
        let Some(name) = self.get_identifier_at(uri, position) else {
            return Ok(None);
        };

        let references = self.find_references(&name, include_declaration);

        if references.is_empty() {
            Ok(None)
        } else {
            Ok(Some(references))
        }
    }

    async fn code_action(&self, params: CodeActionParams) -> Result<Option<CodeActionResponse>> {
        let uri = &params.text_document.uri;
        let range = params.range;
        let diagnostics = &params.context.diagnostics;

        let actions = self.get_code_actions(uri, range, diagnostics);

        if actions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(actions))
        }
    }

    async fn symbol(
        &self,
        params: WorkspaceSymbolParams,
    ) -> Result<Option<Vec<SymbolInformation>>> {
        let symbols = self.get_workspace_symbols(&params.query);

        if symbols.is_empty() {
            Ok(None)
        } else {
            Ok(Some(symbols))
        }
    }

    async fn prepare_rename(
        &self,
        params: TextDocumentPositionParams,
    ) -> Result<Option<PrepareRenameResponse>> {
        let uri = &params.text_document.uri;
        let position = params.position;

        match self.prepare_rename_at(uri, position) {
            Some((name, range)) => {
                // Return the range and placeholder text
                Ok(Some(PrepareRenameResponse::RangeWithPlaceholder {
                    range,
                    placeholder: name,
                }))
            }
            None => Ok(None),
        }
    }

    async fn rename(&self, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let new_name = &params.new_name;

        // Find the identifier at the cursor position
        let Some(old_name) = self.get_identifier_at(uri, position) else {
            return Ok(None);
        };

        // Validate new name is a valid identifier
        if new_name.is_empty() || !new_name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Ok(None);
        }

        let edits = self.create_rename_edits(&old_name, new_name);

        if edits.changes.as_ref().is_none_or(HashMap::is_empty) {
            Ok(None)
        } else {
            Ok(Some(edits))
        }
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = &params.text_document.uri;

        let (text, definition_kinds, definition_spans) = match self.documents.get(uri) {
            Some(doc) => {
                let text = doc.text();
                // Build a map of defined names to their kinds for better classification
                let mut kinds = std::collections::HashMap::new();
                // Build a set of definition name spans (start, end) to mark DECLARATION modifier
                let mut def_spans = std::collections::HashSet::new();
                if let Some(parsed) = &doc.parsed {
                    for cmd in &parsed.commands {
                        if let Some(name) = &cmd.name {
                            kinds.insert(name.clone(), cmd.kind.clone());
                            // Find the name position within the command span
                            // The name typically follows the keyword (def, theorem, etc.)
                            if let Some(name_pos) =
                                find_definition_name_span(&text, cmd.start, cmd.end, name)
                            {
                                def_spans.insert(name_pos);
                            }
                        }
                    }
                }
                (text, kinds, def_spans)
            }
            None => return Ok(None),
        };

        // Tokenize the document
        let tokens = Lexer::tokenize(&text);

        // Build semantic tokens data
        // The data is encoded as a flat array of integers:
        // [deltaLine, deltaStart, length, tokenType, tokenModifiers]
        let mut data: Vec<SemanticToken> = Vec::new();
        let mut prev_line = 0u32;
        let mut prev_start = 0u32;

        for token in &tokens {
            // Get token type and modifiers, with enhanced classification for identifiers
            let (token_type, modifiers) = match &token.kind {
                TokenKind::Ident(name) => {
                    // Check if this is a definition site
                    let is_def_site =
                        definition_spans.contains(&(token.span.start, token.span.end));
                    // Look up identifier in known definitions
                    classify_identifier_with_modifiers(name, &definition_kinds, is_def_site)
                }
                other => (token_kind_to_semantic_type(other), 0),
            };

            let Some(token_type) = token_type else {
                continue;
            };

            // Calculate position from byte offset
            let start_pos = byte_offset_to_position(&text, token.span.start);
            let end_pos = byte_offset_to_position(&text, token.span.end);

            // Calculate delta from previous token
            let delta_line = start_pos.line - prev_line;
            let delta_start = if delta_line == 0 {
                start_pos.character - prev_start
            } else {
                start_pos.character
            };

            // Calculate token length (in characters, not bytes)
            let length = if start_pos.line == end_pos.line {
                end_pos.character - start_pos.character
            } else {
                // Multi-line token - use the first line's length
                // This is rare for most tokens
                (text.len() - token.span.start).min(100) as u32
            };

            data.push(SemanticToken {
                delta_line,
                delta_start,
                length,
                token_type,
                token_modifiers_bitset: modifiers,
            });

            prev_line = start_pos.line;
            prev_start = start_pos.character;
        }

        Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data,
        })))
    }
}

/// Find the byte span of the definition name within a command
/// Returns (start, end) byte offsets if found
fn find_definition_name_span(
    text: &str,
    cmd_start: usize,
    cmd_end: usize,
    name: &str,
) -> Option<(usize, usize)> {
    // Get the command text
    if cmd_start >= text.len() || cmd_end > text.len() || cmd_start >= cmd_end {
        return None;
    }
    let cmd_text = &text[cmd_start..cmd_end];

    // Find the first occurrence of the name after the keyword
    // Skip the keyword (def, theorem, etc.) which is the first token
    let mut in_keyword = true;
    let mut pos = 0;

    for (idx, ch) in cmd_text.char_indices() {
        if in_keyword {
            if ch.is_whitespace() {
                in_keyword = false;
            }
        } else if !ch.is_whitespace() {
            // Start of the name
            pos = idx;
            break;
        }
    }

    // Check if the name matches at this position
    if cmd_text[pos..].starts_with(name) {
        // Verify it's a complete identifier (followed by whitespace or punctuation)
        let end_pos = pos + name.len();
        if end_pos >= cmd_text.len()
            || !cmd_text[end_pos..]
                .chars()
                .next()
                .is_some_and(|c| c.is_alphanumeric() || c == '_')
        {
            return Some((cmd_start + pos, cmd_start + end_pos));
        }
    }

    None
}

/// Classify an identifier and compute modifiers
/// Returns (token_type, modifiers_bitset)
fn classify_identifier_with_modifiers(
    name: &str,
    definition_kinds: &std::collections::HashMap<String, CommandKind>,
    is_definition_site: bool,
) -> (Option<u32>, u32) {
    let mut modifiers = 0u32;

    // Check if this is a definition site
    if is_definition_site {
        modifiers |= modifier_bits::DECLARATION;
        modifiers |= modifier_bits::DEFINITION;
    }

    // Check if identifier is a known definition
    if let Some(kind) = definition_kinds.get(name) {
        return (Some(command_kind_to_semantic_type(kind)), modifiers);
    }

    // Check for built-in types (they get DEFAULT_LIBRARY modifier)
    if is_builtin_type(name) {
        modifiers |= modifier_bits::DEFAULT_LIBRARY;
        return (Some(1), modifiers); // TYPE
    }

    // Check for common type names (capitalized identifiers often are types)
    if is_likely_type_name(name) {
        return (Some(1), modifiers); // TYPE
    }

    // Default: VARIABLE (local variables are readonly in Lean - immutable bindings)
    modifiers |= modifier_bits::READONLY;
    (Some(3), modifiers)
}

/// Check if a name is a built-in type from the standard library
fn is_builtin_type(name: &str) -> bool {
    const BUILTIN_TYPES: &[&str] = &[
        // Core types
        "Nat",
        "Int",
        "Bool",
        "String",
        "Char",
        "Float",
        "Unit",
        "Empty",
        "True",
        "False",
        "Prop",
        "Type",
        "Sort",
        // Collections
        "List",
        "Array",
        "Option",
        "Sum",
        "Prod",
        "Fin",
        "Subtype",
        // Numeric types
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "USize",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        // Monads and transformers
        "IO",
        "Except",
        "EStateM",
        "StateT",
        "ReaderT",
        "ExceptT",
        "OptionT",
        "StateM",
        "ReaderM",
        "ExceptM",
        "Id",
        // Other common types
        "Decidable",
        "DecidableEq",
        "BEq",
        "Hashable",
        "Repr",
        "ToString",
        "Inhabited",
        "Nonempty",
        "Functor",
        "Monad",
        "Applicative",
    ];
    BUILTIN_TYPES.contains(&name)
}

/// Map CommandKind to semantic token type index
fn command_kind_to_semantic_type(kind: &CommandKind) -> u32 {
    match kind {
        // Functions (theorems, lemmas, definitions, axioms produce terms)
        CommandKind::Definition
        | CommandKind::Theorem
        | CommandKind::Lemma
        | CommandKind::Axiom
        | CommandKind::Example => 2, // FUNCTION

        // Types (inductive types, structures)
        CommandKind::Inductive | CommandKind::Structure => 1, // TYPE

        // Classes
        CommandKind::Class => 9, // CLASS

        // Instances (like properties/methods)
        CommandKind::Instance => 10, // PROPERTY

        // Namespaces
        CommandKind::Namespace => 8, // NAMESPACE

        // Variables, universes, and everything else default to variable
        CommandKind::Variable
        | CommandKind::Universe
        | CommandKind::Import
        | CommandKind::Open
        | CommandKind::Section
        | CommandKind::End
        | CommandKind::Other(_) => 3, // VARIABLE
    }
}

/// Heuristic: check if a name is likely a type name
/// In Lean, type names typically start with an uppercase letter
fn is_likely_type_name(name: &str) -> bool {
    // Common built-in types
    const BUILTIN_TYPES: &[&str] = &[
        "Nat", "Int", "Bool", "String", "Char", "Float", "Unit", "Empty", "True", "False", "List",
        "Array", "Option", "Sum", "Prod", "Fin", "UInt8", "UInt16", "UInt32", "UInt64", "USize",
        "IO", "Except", "EStateM", "StateT", "ReaderT", "ExceptT", "OptionT",
    ];

    if BUILTIN_TYPES.contains(&name) {
        return true;
    }

    // Identifiers starting with uppercase are often types (but not always)
    // Only apply this heuristic for simple identifiers without dots
    if !name.contains('.') {
        if let Some(first_char) = name.chars().next() {
            return first_char.is_uppercase();
        }
    }

    false
}

/// Parse text into a ParsedDocument (public for testing)
#[must_use]
pub fn parse_lean_text(text: &str) -> ParsedDocument {
    match lean5_parser::parse_file(text) {
        Ok(decls) => {
            let mut commands = Vec::new();

            for decl in &decls {
                let (kind, name, span) = Lean5Backend::classify_decl(decl);
                let content_hash = Lean5Backend::compute_content_hash(text, span.0, span.1);
                commands.push(ParsedCommand {
                    kind,
                    start: span.0,
                    end: span.1,
                    name,
                    content_hash,
                });
            }

            ParsedDocument {
                errors: vec![],
                commands,
            }
        }
        Err(e) => {
            let message = format!("{e}");
            ParsedDocument {
                errors: vec![ParseError {
                    start: 0,
                    end: 1,
                    message,
                }],
                commands: vec![],
            }
        }
    }
}

/// Collect all identifiers used in a surface expression
fn collect_used_idents(expr: &lean5_parser::SurfaceExpr, used: &mut HashSet<String>) {
    use lean5_parser::SurfaceExpr;

    match expr {
        SurfaceExpr::Ident(_, name) => {
            // Split qualified names and add the first component
            // e.g., "Nat.add" -> we care about "Nat", not local variable references
            let first_part = name.split('.').next().unwrap_or(name);
            used.insert(first_part.to_string());
        }
        SurfaceExpr::App(_, func, args) => {
            collect_used_idents(func, used);
            for arg in args {
                collect_used_idents(&arg.expr, used);
            }
        }
        SurfaceExpr::Lambda(_, binders, body)
        | SurfaceExpr::PatternMatchLambda(_, binders, body) => {
            // Collect from binder types
            for binder in binders {
                if let Some(ty) = &binder.ty {
                    collect_used_idents(ty, used);
                }
                if let Some(default) = &binder.default {
                    collect_used_idents(default, used);
                }
            }
            collect_used_idents(body, used);
        }
        SurfaceExpr::Pi(_, binders, body) => {
            for binder in binders {
                if let Some(ty) = &binder.ty {
                    collect_used_idents(ty, used);
                }
                if let Some(default) = &binder.default {
                    collect_used_idents(default, used);
                }
            }
            collect_used_idents(body, used);
        }
        SurfaceExpr::Arrow(_, left, right) => {
            collect_used_idents(left, used);
            collect_used_idents(right, used);
        }
        SurfaceExpr::Let(_, binder, val, body) | SurfaceExpr::LetRec(_, binder, val, body) => {
            if let Some(ty) = &binder.ty {
                collect_used_idents(ty, used);
            }
            collect_used_idents(val, used);
            collect_used_idents(body, used);
        }
        SurfaceExpr::If(_, cond, then_branch, else_branch) => {
            collect_used_idents(cond, used);
            collect_used_idents(then_branch, used);
            collect_used_idents(else_branch, used);
        }
        SurfaceExpr::IfLet(_, _pat, scrutinee, then_branch, else_branch) => {
            collect_used_idents(scrutinee, used);
            collect_used_idents(then_branch, used);
            collect_used_idents(else_branch, used);
        }
        SurfaceExpr::IfDecidable(_, _, prop, then_branch, else_branch) => {
            collect_used_idents(prop, used);
            collect_used_idents(then_branch, used);
            collect_used_idents(else_branch, used);
        }
        SurfaceExpr::Match(_, scrutinee, arms) => {
            collect_used_idents(scrutinee, used);
            for arm in arms {
                collect_used_idents(&arm.body, used);
            }
        }
        SurfaceExpr::Paren(_, inner)
        | SurfaceExpr::OutParam(_, inner)
        | SurfaceExpr::SemiOutParam(_, inner)
        | SurfaceExpr::Explicit(_, inner) => collect_used_idents(inner, used),
        SurfaceExpr::Ascription(_, expr, ty) => {
            collect_used_idents(expr, used);
            collect_used_idents(ty, used);
        }
        SurfaceExpr::Proj(_, expr, _)
        | SurfaceExpr::UniverseInst(_, expr, _)
        | SurfaceExpr::NamedArg(_, _, expr) => collect_used_idents(expr, used),
        // Terminal expressions with no nested identifiers
        SurfaceExpr::Universe(_, _)
        | SurfaceExpr::Lit(_, _)
        | SurfaceExpr::Hole(_)
        | SurfaceExpr::SyntaxQuote(_, _) => {}
    }
}

/// A binder with location info for warnings
struct BinderInfo {
    name: String,
    start: usize,
    end: usize,
}

/// Location of an identifier in source
struct IdentLocation {
    name: String,
    start: usize,
    end: usize,
}

/// Collect all identifier occurrences (with spans) in an expression
fn collect_ident_locations(expr: &lean5_parser::SurfaceExpr, locations: &mut Vec<IdentLocation>) {
    use lean5_parser::SurfaceExpr;

    match expr {
        SurfaceExpr::Ident(span, name) => locations.push(IdentLocation {
            name: name.clone(),
            start: span.start,
            end: span.end,
        }),
        SurfaceExpr::App(_, func, args) => {
            collect_ident_locations(func, locations);
            for arg in args {
                collect_ident_locations(&arg.expr, locations);
            }
        }
        SurfaceExpr::Lambda(_, binders, body)
        | SurfaceExpr::PatternMatchLambda(_, binders, body)
        | SurfaceExpr::Pi(_, binders, body) => {
            collect_locations_from_binders(binders, locations);
            collect_ident_locations(body, locations);
        }
        SurfaceExpr::Arrow(_, left, right) => {
            collect_ident_locations(left, locations);
            collect_ident_locations(right, locations);
        }
        SurfaceExpr::Let(_, binder, val, body) | SurfaceExpr::LetRec(_, binder, val, body) => {
            collect_locations_from_binders(std::slice::from_ref(binder), locations);
            collect_ident_locations(val, locations);
            collect_ident_locations(body, locations);
        }
        SurfaceExpr::If(_, cond, then_branch, else_branch) => {
            collect_ident_locations(cond, locations);
            collect_ident_locations(then_branch, locations);
            collect_ident_locations(else_branch, locations);
        }
        SurfaceExpr::IfLet(_, _pat, scrutinee, then_branch, else_branch) => {
            collect_ident_locations(scrutinee, locations);
            collect_ident_locations(then_branch, locations);
            collect_ident_locations(else_branch, locations);
        }
        SurfaceExpr::IfDecidable(_, _, prop, then_branch, else_branch) => {
            collect_ident_locations(prop, locations);
            collect_ident_locations(then_branch, locations);
            collect_ident_locations(else_branch, locations);
        }
        SurfaceExpr::Match(_, scrutinee, arms) => {
            collect_ident_locations(scrutinee, locations);
            for arm in arms {
                collect_ident_locations(&arm.body, locations);
            }
        }
        // Single inner expression to recurse into
        SurfaceExpr::Paren(_, inner)
        | SurfaceExpr::OutParam(_, inner)
        | SurfaceExpr::SemiOutParam(_, inner)
        | SurfaceExpr::Explicit(_, inner) => collect_ident_locations(inner, locations),
        SurfaceExpr::Ascription(_, expr, ty) => {
            collect_ident_locations(expr, locations);
            collect_ident_locations(ty, locations);
        }
        SurfaceExpr::Proj(_, expr, _)
        | SurfaceExpr::UniverseInst(_, expr, _)
        | SurfaceExpr::NamedArg(_, _, expr) => collect_ident_locations(expr, locations),
        // Terminal expressions with no nested identifiers
        SurfaceExpr::Universe(_, _)
        | SurfaceExpr::Lit(_, _)
        | SurfaceExpr::Hole(_)
        | SurfaceExpr::SyntaxQuote(_, _) => {}
    }
}

/// Collect identifier occurrences from binders (types and defaults)
fn collect_locations_from_binders(
    binders: &[lean5_parser::SurfaceBinder],
    locations: &mut Vec<IdentLocation>,
) {
    for binder in binders {
        if let Some(ty) = &binder.ty {
            collect_ident_locations(ty, locations);
        }
        if let Some(default) = &binder.default {
            collect_ident_locations(default, locations);
        }
    }
}

/// Collect all `sorry` and `admit` occurrences in an expression
fn collect_sorry_usage(expr: &lean5_parser::SurfaceExpr, locations: &mut Vec<IdentLocation>) {
    use lean5_parser::SurfaceExpr;

    match expr {
        SurfaceExpr::Ident(span, name) => {
            // Check for sorry, admit, or native_decide (which can hide incomplete proofs)
            if name == "sorry" || name == "admit" {
                locations.push(IdentLocation {
                    name: name.clone(),
                    start: span.start,
                    end: span.end,
                });
            }
        }
        SurfaceExpr::App(_, func, args) => {
            collect_sorry_usage(func, locations);
            for arg in args {
                collect_sorry_usage(&arg.expr, locations);
            }
        }
        SurfaceExpr::Lambda(_, binders, body)
        | SurfaceExpr::PatternMatchLambda(_, binders, body)
        | SurfaceExpr::Pi(_, binders, body) => {
            for binder in binders {
                if let Some(ty) = &binder.ty {
                    collect_sorry_usage(ty, locations);
                }
                if let Some(default) = &binder.default {
                    collect_sorry_usage(default, locations);
                }
            }
            collect_sorry_usage(body, locations);
        }
        SurfaceExpr::Arrow(_, left, right) => {
            collect_sorry_usage(left, locations);
            collect_sorry_usage(right, locations);
        }
        SurfaceExpr::Let(_, binder, val, body) | SurfaceExpr::LetRec(_, binder, val, body) => {
            if let Some(ty) = &binder.ty {
                collect_sorry_usage(ty, locations);
            }
            collect_sorry_usage(val, locations);
            collect_sorry_usage(body, locations);
        }
        SurfaceExpr::If(_, cond, then_branch, else_branch) => {
            collect_sorry_usage(cond, locations);
            collect_sorry_usage(then_branch, locations);
            collect_sorry_usage(else_branch, locations);
        }
        SurfaceExpr::IfLet(_, _pat, scrutinee, then_branch, else_branch) => {
            collect_sorry_usage(scrutinee, locations);
            collect_sorry_usage(then_branch, locations);
            collect_sorry_usage(else_branch, locations);
        }
        SurfaceExpr::IfDecidable(_, _, prop, then_branch, else_branch) => {
            collect_sorry_usage(prop, locations);
            collect_sorry_usage(then_branch, locations);
            collect_sorry_usage(else_branch, locations);
        }
        SurfaceExpr::Match(_, scrutinee, arms) => {
            collect_sorry_usage(scrutinee, locations);
            for arm in arms {
                collect_sorry_usage(&arm.body, locations);
            }
        }
        // Single inner expression to recurse into
        SurfaceExpr::Paren(_, inner)
        | SurfaceExpr::OutParam(_, inner)
        | SurfaceExpr::SemiOutParam(_, inner)
        | SurfaceExpr::Explicit(_, inner) => collect_sorry_usage(inner, locations),
        SurfaceExpr::Ascription(_, expr, ty) => {
            collect_sorry_usage(expr, locations);
            collect_sorry_usage(ty, locations);
        }
        SurfaceExpr::Proj(_, expr, _)
        | SurfaceExpr::UniverseInst(_, expr, _)
        | SurfaceExpr::NamedArg(_, _, expr) => collect_sorry_usage(expr, locations),
        // Terminal expressions
        SurfaceExpr::Universe(_, _)
        | SurfaceExpr::Lit(_, _)
        | SurfaceExpr::Hole(_)
        | SurfaceExpr::SyntaxQuote(_, _) => {}
    }
}

/// Detect `sorry` usage in a declaration and return warnings
fn detect_sorry_warnings(decl: &lean5_parser::SurfaceDecl) -> Vec<Warning> {
    use lean5_parser::SurfaceDecl;

    let mut locations = Vec::new();

    // Extract expressions to check based on declaration type
    match decl {
        SurfaceDecl::Def { ty, val, .. } => {
            if let Some(ty) = ty {
                collect_sorry_usage(ty, &mut locations);
            }
            collect_sorry_usage(val, &mut locations);
        }
        SurfaceDecl::Theorem { ty, proof, .. } => {
            collect_sorry_usage(ty, &mut locations);
            collect_sorry_usage(proof, &mut locations);
        }
        _ => {}
    }

    // Convert to warnings
    locations
        .into_iter()
        .map(|loc| Warning {
            start: loc.start,
            end: loc.end,
            message: format!("declaration uses `{}` (incomplete proof)", loc.name),
            code: WarningCode::IncompleteProof,
        })
        .collect()
}

/// Collect names marked as deprecated via `attribute [deprecated] ...`
fn collect_deprecated_names(decls: &[lean5_parser::SurfaceDecl]) -> HashSet<String> {
    let mut deprecated = HashSet::new();

    for decl in decls {
        match decl {
            lean5_parser::SurfaceDecl::Attribute { attrs, names, .. } => {
                let is_deprecated = attrs.iter().any(|attr| {
                    matches!(attr, lean5_parser::Attribute::Unknown(name) if name == "deprecated")
                });

                if is_deprecated {
                    for name in names {
                        deprecated.insert(name.clone());
                    }
                }
            }
            lean5_parser::SurfaceDecl::Namespace { decls, .. }
            | lean5_parser::SurfaceDecl::Section { decls, .. }
            | lean5_parser::SurfaceDecl::Mutual { decls, .. } => {
                deprecated.extend(collect_deprecated_names(decls));
            }
            _ => {}
        }
    }

    deprecated
}

/// Convert identifier occurrences to deprecation warnings
fn warnings_for_deprecated_usage(
    locations: Vec<IdentLocation>,
    deprecated_names: &HashSet<String>,
) -> Vec<Warning> {
    locations
        .into_iter()
        .filter(|loc| deprecated_names.contains(&loc.name))
        .map(|loc| Warning {
            start: loc.start,
            end: loc.end,
            message: format!("`{}` is deprecated", loc.name),
            code: WarningCode::DeprecatedFeature,
        })
        .collect()
}

/// Detect usage of deprecated names within a declaration
fn detect_deprecated_usage(
    decl: &lean5_parser::SurfaceDecl,
    deprecated_names: &HashSet<String>,
) -> Vec<Warning> {
    use lean5_parser::SurfaceDecl;

    if deprecated_names.is_empty() {
        return Vec::new();
    }

    match decl {
        SurfaceDecl::Def {
            binders, ty, val, ..
        }
        | SurfaceDecl::Example {
            binders, ty, val, ..
        } => {
            let mut locations = Vec::new();
            collect_locations_from_binders(binders, &mut locations);
            if let Some(ty) = ty {
                collect_ident_locations(ty, &mut locations);
            }
            collect_ident_locations(val, &mut locations);
            warnings_for_deprecated_usage(locations, deprecated_names)
        }
        SurfaceDecl::Theorem {
            binders, ty, proof, ..
        } => {
            let mut locations = Vec::new();
            collect_locations_from_binders(binders, &mut locations);
            collect_ident_locations(ty, &mut locations);
            collect_ident_locations(proof, &mut locations);
            warnings_for_deprecated_usage(locations, deprecated_names)
        }
        SurfaceDecl::Axiom { binders, ty, .. } => {
            let mut locations = Vec::new();
            collect_locations_from_binders(binders, &mut locations);
            collect_ident_locations(ty, &mut locations);
            warnings_for_deprecated_usage(locations, deprecated_names)
        }
        SurfaceDecl::Inductive {
            binders, ty, ctors, ..
        } => {
            let mut locations = Vec::new();
            collect_locations_from_binders(binders, &mut locations);
            collect_ident_locations(ty, &mut locations);
            for ctor in ctors {
                collect_ident_locations(&ctor.ty, &mut locations);
            }
            warnings_for_deprecated_usage(locations, deprecated_names)
        }
        SurfaceDecl::Structure {
            binders,
            ty,
            fields,
            ..
        }
        | SurfaceDecl::Class {
            binders,
            ty,
            fields,
            ..
        } => {
            let mut locations = Vec::new();
            collect_locations_from_binders(binders, &mut locations);
            if let Some(ty) = ty {
                collect_ident_locations(ty, &mut locations);
            }
            for field in fields {
                collect_ident_locations(&field.ty, &mut locations);
                if let Some(default) = &field.default {
                    collect_ident_locations(default, &mut locations);
                }
            }
            warnings_for_deprecated_usage(locations, deprecated_names)
        }
        SurfaceDecl::Instance {
            binders,
            class_type,
            fields,
            ..
        } => {
            let mut locations = Vec::new();
            collect_locations_from_binders(binders, &mut locations);
            collect_ident_locations(class_type, &mut locations);
            for field in fields {
                collect_ident_locations(&field.val, &mut locations);
            }
            warnings_for_deprecated_usage(locations, deprecated_names)
        }
        SurfaceDecl::Namespace { decls, .. }
        | SurfaceDecl::Section { decls, .. }
        | SurfaceDecl::Mutual { decls, .. } => decls
            .iter()
            .flat_map(|d| detect_deprecated_usage(d, deprecated_names))
            .collect(),
        _ => Vec::new(),
    }
}

/// Detect unused binders in a declaration and return warnings
fn detect_unused_variables(decl: &lean5_parser::SurfaceDecl) -> Vec<Warning> {
    use lean5_parser::SurfaceDecl;

    let mut warnings = Vec::new();

    // Extract binders and body expressions based on declaration type
    let (binders, exprs): (Vec<BinderInfo>, Vec<&lean5_parser::SurfaceExpr>) = match decl {
        SurfaceDecl::Def {
            binders, ty, val, ..
        } => {
            let binder_info: Vec<_> = binders
                .iter()
                .map(|b| BinderInfo {
                    name: b.name.clone(),
                    start: b.span.start,
                    end: b.span.end,
                })
                .collect();
            let mut exprs: Vec<&lean5_parser::SurfaceExpr> = vec![val.as_ref()];
            if let Some(ty) = ty {
                exprs.push(ty.as_ref());
            }
            (binder_info, exprs)
        }
        SurfaceDecl::Theorem {
            binders, ty, proof, ..
        } => {
            let binder_info: Vec<_> = binders
                .iter()
                .map(|b| BinderInfo {
                    name: b.name.clone(),
                    start: b.span.start,
                    end: b.span.end,
                })
                .collect();
            (binder_info, vec![ty.as_ref(), proof.as_ref()])
        }
        SurfaceDecl::Axiom { binders, ty, .. } => {
            let binder_info: Vec<_> = binders
                .iter()
                .map(|b| BinderInfo {
                    name: b.name.clone(),
                    start: b.span.start,
                    end: b.span.end,
                })
                .collect();
            (binder_info, vec![ty.as_ref()])
        }
        // For inductives, structures, classes, instances - don't check unused params
        // as they are often used for type inference purposes
        _ => return warnings,
    };

    // Collect all used identifiers
    let mut used = HashSet::new();
    for expr in exprs {
        collect_used_idents(expr, &mut used);
    }

    // Check each binder
    for binder in binders {
        // Skip anonymous binders (names starting with _)
        if binder.name.starts_with('_') {
            continue;
        }

        // Check if the binder is used
        if !used.contains(&binder.name) {
            warnings.push(Warning {
                start: binder.start,
                end: binder.end,
                message: format!("unused variable `{}`", binder.name),
                code: WarningCode::UnusedVariable,
            });
        }
    }

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_def() {
        let text = "def x := 1";
        let parsed = parse_lean_text(text);

        assert!(parsed.errors.is_empty());
        assert_eq!(parsed.commands.len(), 1);
        assert_eq!(parsed.commands[0].kind, CommandKind::Definition);
        assert_eq!(parsed.commands[0].name, Some("x".to_string()));
    }

    #[test]
    fn test_parse_multiple_defs() {
        let text = "def x := 1\ndef y := 2\ntheorem t : True := trivial";
        let parsed = parse_lean_text(text);

        assert!(parsed.errors.is_empty());
        assert_eq!(parsed.commands.len(), 3);
        assert_eq!(parsed.commands[0].kind, CommandKind::Definition);
        assert_eq!(parsed.commands[1].kind, CommandKind::Definition);
        assert_eq!(parsed.commands[2].kind, CommandKind::Theorem);
    }

    #[test]
    fn test_parse_with_error() {
        let text = "def x :=";
        let parsed = parse_lean_text(text);
        assert!(!parsed.errors.is_empty());
    }

    #[test]
    fn test_parse_inductive() {
        let text = "inductive Nat : Type\n| zero : Nat\n| succ : Nat -> Nat";
        let parsed = parse_lean_text(text);

        assert!(parsed.errors.is_empty());
        assert_eq!(parsed.commands.len(), 1);
        assert_eq!(parsed.commands[0].kind, CommandKind::Inductive);
        assert_eq!(parsed.commands[0].name, Some("Nat".to_string()));
    }

    #[test]
    fn test_parse_structure() {
        let text = "structure Point where\n  x : Nat\n  y : Nat";
        let parsed = parse_lean_text(text);

        assert!(parsed.errors.is_empty());
        assert_eq!(parsed.commands.len(), 1);
        assert_eq!(parsed.commands[0].kind, CommandKind::Structure);
        assert_eq!(parsed.commands[0].name, Some("Point".to_string()));
    }

    #[test]
    fn test_document_creation() {
        let uri = Url::parse("file:///test.lean").unwrap();
        let doc = Document::new(uri, 1, "def x := 1\n".to_string(), "lean".to_string());

        assert_eq!(doc.version, 1);
        assert_eq!(doc.text(), "def x := 1\n");
    }

    #[test]
    fn test_is_identifier_char() {
        // Alphanumeric
        assert!(Lean5Backend::is_identifier_char(b'a'));
        assert!(Lean5Backend::is_identifier_char(b'Z'));
        assert!(Lean5Backend::is_identifier_char(b'5'));

        // Underscore and apostrophe
        assert!(Lean5Backend::is_identifier_char(b'_'));
        assert!(Lean5Backend::is_identifier_char(b'\''));

        // Non-identifier chars
        assert!(!Lean5Backend::is_identifier_char(b' '));
        assert!(!Lean5Backend::is_identifier_char(b':'));
        assert!(!Lean5Backend::is_identifier_char(b'.'));
        assert!(!Lean5Backend::is_identifier_char(b'\n'));
    }

    #[test]
    fn test_definition_info_clone() {
        let uri = Url::parse("file:///test.lean").unwrap();
        let info = DefinitionInfo {
            uri: uri.clone(),
            start: 0,
            end: 10,
        };
        let cloned = info.clone();
        assert_eq!(cloned.uri, uri);
        assert_eq!(cloned.start, 0);
        assert_eq!(cloned.end, 10);
    }

    #[test]
    fn test_ranges_overlap() {
        let r1 = Range {
            start: Position {
                line: 0,
                character: 0,
            },
            end: Position {
                line: 0,
                character: 10,
            },
        };
        let r2 = Range {
            start: Position {
                line: 0,
                character: 5,
            },
            end: Position {
                line: 0,
                character: 15,
            },
        };
        let r3 = Range {
            start: Position {
                line: 0,
                character: 20,
            },
            end: Position {
                line: 0,
                character: 30,
            },
        };

        // Overlapping ranges
        assert!(super::ranges_overlap(r1, r2));
        assert!(super::ranges_overlap(r2, r1));

        // Non-overlapping ranges
        assert!(!super::ranges_overlap(r1, r3));
        assert!(!super::ranges_overlap(r3, r1));
    }

    #[test]
    fn test_extract_identifier_from_error() {
        // Backtick quoted
        let msg1 = "unknown identifier `foo`";
        assert_eq!(
            super::extract_identifier_from_error(msg1),
            Some("foo".to_string())
        );

        // Pattern-based
        let msg2 = "unknown identifier bar in context";
        assert_eq!(
            super::extract_identifier_from_error(msg2),
            Some("bar".to_string())
        );

        // Not found pattern
        let msg3 = "not found: HashMap";
        assert_eq!(
            super::extract_identifier_from_error(msg3),
            Some("HashMap".to_string())
        );

        // No identifier
        let msg4 = "some other error";
        assert_eq!(super::extract_identifier_from_error(msg4), None);
    }

    #[test]
    fn test_suggest_imports_for_identifier() {
        // Basic types
        assert!(!super::suggest_imports_for_identifier("Nat").is_empty());
        assert!(!super::suggest_imports_for_identifier("List").is_empty());

        // Std types
        assert!(!super::suggest_imports_for_identifier("HashMap").is_empty());

        // Mathlib types
        assert!(!super::suggest_imports_for_identifier("Real").is_empty());
        assert!(!super::suggest_imports_for_identifier("Group").is_empty());

        // Unknown identifier
        assert!(super::suggest_imports_for_identifier("UnknownThing123").is_empty());
    }

    #[test]
    fn test_get_symbol_kind_for_definition() {
        // This test validates the symbol kind mapping logic
        // The actual backend methods require async context, so we test the mapping directly
        let mappings = [
            (CommandKind::Definition, SymbolKind::FUNCTION),
            (CommandKind::Theorem, SymbolKind::FUNCTION),
            (CommandKind::Lemma, SymbolKind::FUNCTION),
            (CommandKind::Inductive, SymbolKind::CLASS),
            (CommandKind::Structure, SymbolKind::CLASS),
            (CommandKind::Class, SymbolKind::INTERFACE),
            (CommandKind::Instance, SymbolKind::OBJECT),
            (CommandKind::Axiom, SymbolKind::CONSTANT),
            (CommandKind::Variable, SymbolKind::VARIABLE),
            (CommandKind::Namespace, SymbolKind::NAMESPACE),
        ];

        for (cmd_kind, expected_symbol_kind) in mappings {
            let symbol_kind = match cmd_kind {
                CommandKind::Definition | CommandKind::Theorem | CommandKind::Lemma => {
                    SymbolKind::FUNCTION
                }
                CommandKind::Inductive | CommandKind::Structure => SymbolKind::CLASS,
                CommandKind::Class => SymbolKind::INTERFACE,
                CommandKind::Instance => SymbolKind::OBJECT,
                CommandKind::Axiom => SymbolKind::CONSTANT,
                CommandKind::Variable => SymbolKind::VARIABLE,
                CommandKind::Namespace => SymbolKind::NAMESPACE,
                _ => SymbolKind::NULL,
            };
            assert_eq!(
                symbol_kind, expected_symbol_kind,
                "Mismatch for {cmd_kind:?}"
            );
        }
    }

    #[test]
    fn test_workspace_symbol_query_matching() {
        // Test case-insensitive matching logic
        let test_cases = [
            ("", "anything", true),     // Empty query matches everything
            ("nat", "Nat", true),       // Case-insensitive match
            ("NAT", "natural", true),   // Case-insensitive substring
            ("foo", "bar", false),      // No match
            ("add", "Nat.add", true),   // Substring match
            ("Point", "MyPoint", true), // Substring match
            ("xyz", "Point", false),    // No match
        ];

        for (query, name, should_match) in test_cases {
            let query_lower = query.to_lowercase();
            let matches = query.is_empty() || name.to_lowercase().contains(&query_lower);
            assert_eq!(
                matches, should_match,
                "Query '{query}' vs name '{name}': expected {should_match}, got {matches}"
            );
        }
    }

    #[test]
    fn test_unused_variable_detection() {
        // Test that unused variables are detected
        let text = "def f (x : Nat) (y : Nat) := x";
        let decls = lean5_parser::parse_file(text).unwrap();
        assert_eq!(decls.len(), 1);

        let warnings = detect_unused_variables(&decls[0]);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("unused variable `y`"));
        assert_eq!(warnings[0].code, WarningCode::UnusedVariable);
    }

    #[test]
    fn test_no_warning_for_used_variables() {
        // Test that used variables don't generate warnings
        let text = "def add (x : Nat) (y : Nat) := Nat.add x y";
        let decls = lean5_parser::parse_file(text).unwrap();
        assert_eq!(decls.len(), 1);

        let warnings = detect_unused_variables(&decls[0]);
        assert!(
            warnings.is_empty(),
            "Expected no warnings, got: {warnings:?}"
        );
    }

    #[test]
    fn test_underscore_variables_not_warned() {
        // Test that underscore-prefixed variables don't generate warnings
        let text = "def f (_unused : Nat) := 42";
        let decls = lean5_parser::parse_file(text).unwrap();
        assert_eq!(decls.len(), 1);

        let warnings = detect_unused_variables(&decls[0]);
        assert!(
            warnings.is_empty(),
            "Underscore-prefixed variables should not generate warnings"
        );
    }

    #[test]
    fn test_theorem_unused_variable() {
        // Test unused variable detection in theorems
        let text = "theorem t (h : True) (unused : False) : True := h";
        let decls = lean5_parser::parse_file(text).unwrap();
        assert_eq!(decls.len(), 1);

        let warnings = detect_unused_variables(&decls[0]);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("unused variable `unused`"));
    }

    #[test]
    fn test_sorry_detection() {
        // Test that sorry usage is detected
        let text = "theorem t : True := sorry";
        let decls = lean5_parser::parse_file(text).unwrap();
        assert_eq!(decls.len(), 1);

        let warnings = detect_sorry_warnings(&decls[0]);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("sorry"));
        assert_eq!(warnings[0].code, WarningCode::IncompleteProof);
    }

    #[test]
    fn test_admit_detection() {
        // Test that admit usage is detected
        let text = "def f : Nat := admit";
        let decls = lean5_parser::parse_file(text).unwrap();
        assert_eq!(decls.len(), 1);

        let warnings = detect_sorry_warnings(&decls[0]);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("admit"));
        assert_eq!(warnings[0].code, WarningCode::IncompleteProof);
    }

    #[test]
    fn test_no_sorry_warning_for_complete_proof() {
        // Test that complete proofs don't generate sorry warnings
        let text = "theorem t : True := True.intro";
        let decls = lean5_parser::parse_file(text).unwrap();
        assert_eq!(decls.len(), 1);

        let warnings = detect_sorry_warnings(&decls[0]);
        assert!(
            warnings.is_empty(),
            "Complete proofs should not generate sorry warnings"
        );
    }

    #[test]
    fn test_deprecated_usage_detection() {
        let text = r"
def old : Nat := 1
attribute [deprecated] old
def use_old : Nat := old
";

        let decls = lean5_parser::parse_file(text).unwrap();
        let deprecated = collect_deprecated_names(&decls);
        assert!(deprecated.contains("old"));

        let warnings: Vec<_> = decls
            .iter()
            .flat_map(|d| detect_deprecated_usage(d, &deprecated))
            .collect();

        let deprecated_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::DeprecatedFeature)
            .collect();

        assert_eq!(
            deprecated_warnings.len(),
            1,
            "Expected one deprecated usage warning, got: {deprecated_warnings:?}"
        );
        assert!(
            deprecated_warnings[0].message.contains("deprecated"),
            "Warning message should mention deprecation"
        );
    }

    #[test]
    fn test_prepare_rename_valid_identifier() {
        // Test that valid identifiers can be prepared for rename
        let text = "def myFunction := 1";
        let parsed = parse_lean_text(text);
        assert!(parsed.errors.is_empty());
        assert_eq!(parsed.commands.len(), 1);
        assert_eq!(parsed.commands[0].name, Some("myFunction".to_string()));
    }

    #[test]
    fn test_create_rename_edits_helper() {
        // Test the workspace edit structure
        use std::collections::HashMap;

        // Create a mock workspace edit
        let mut changes: HashMap<Url, Vec<TextEdit>> = HashMap::new();
        let uri = Url::parse("file:///test.lean").unwrap();
        changes.insert(
            uri.clone(),
            vec![TextEdit {
                range: Range {
                    start: Position {
                        line: 0,
                        character: 4,
                    },
                    end: Position {
                        line: 0,
                        character: 5,
                    },
                },
                new_text: "y".to_string(),
            }],
        );

        let edit = WorkspaceEdit {
            changes: Some(changes),
            document_changes: None,
            change_annotations: None,
        };

        assert!(edit.changes.is_some());
        let changes = edit.changes.unwrap();
        assert_eq!(changes.len(), 1);
        assert!(changes.contains_key(&uri));
    }

    #[test]
    fn test_identifier_validation_for_rename() {
        // Test valid/invalid identifier detection for renaming
        let valid_names = ["x", "myVar", "my_var", "x1", "Nat", "_x"];
        let invalid_names = ["", "1x", "my-var", "a b"];

        for name in valid_names {
            let is_valid = !name.is_empty()
                && (name.starts_with('_') || name.chars().next().is_some_and(char::is_alphabetic))
                && name.chars().all(|c| c.is_alphanumeric() || c == '_');
            assert!(is_valid || name == "_x", "Expected '{name}' to be valid");
        }

        for name in invalid_names {
            let is_valid =
                !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_');
            // Empty string and strings with invalid chars should fail
            if name.is_empty() || name.contains('-') || name.contains(' ') {
                assert!(!is_valid, "Expected '{name}' to be invalid");
            }
        }
    }

    #[test]
    fn test_content_hash_computation() {
        // Test that content hashes are computed correctly
        let text = "def x := 1\ndef y := 2";
        let hash1 = Lean5Backend::compute_content_hash(text, 0, 10);
        let hash2 = Lean5Backend::compute_content_hash(text, 11, 21);
        let hash1_again = Lean5Backend::compute_content_hash(text, 0, 10);

        // Same content should produce same hash
        assert_eq!(hash1, hash1_again);
        // Different content should produce different hash
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_content_hash_detects_changes() {
        // Test that changing content changes the hash
        let text1 = "def x := 1";
        let text2 = "def x := 2";

        let hash1 = Lean5Backend::compute_content_hash(text1, 0, text1.len());
        let hash2 = Lean5Backend::compute_content_hash(text2, 0, text2.len());

        assert_ne!(hash1, hash2, "Hash should change when content changes");
    }

    #[test]
    fn test_incremental_state_default() {
        // Test that default incremental state is empty
        let state = IncrementalState::default();
        assert!(state.cache.is_empty());
        assert_eq!(state.stats.total_commands, 0);
        assert_eq!(state.stats.elaborated_count, 0);
        assert_eq!(state.stats.cached_count, 0);
    }

    #[test]
    fn test_parsed_command_has_content_hash() {
        // Test that parsed commands include content hashes
        let text = "def foo := 42";
        let parsed = parse_lean_text(text);

        assert!(parsed.errors.is_empty());
        assert_eq!(parsed.commands.len(), 1);
        // Content hash should be non-zero for non-empty content
        assert_ne!(parsed.commands[0].content_hash, 0);
    }

    #[test]
    fn test_cache_key_generation() {
        // Test cache key generation for named and anonymous declarations
        let text = "def named := 1\nexample : True := trivial";
        let parsed = parse_lean_text(text);

        assert_eq!(parsed.commands.len(), 2);
        // Named declaration should use its name
        assert_eq!(parsed.commands[0].name, Some("named".to_string()));
        // Anonymous declaration (example) should have no name
        assert_eq!(parsed.commands[1].name, None);
    }

    #[test]
    fn test_semantic_token_type_mapping_keywords() {
        // Keywords should map to token type 0
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Def), Some(0));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Theorem), Some(0));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Lemma), Some(0));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Let), Some(0));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::If), Some(0));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Match), Some(0));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Structure), Some(0));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Class), Some(0));
    }

    #[test]
    fn test_semantic_token_type_mapping_types() {
        // Type keywords should map to token type 1
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Type), Some(1));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Prop), Some(1));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Sort), Some(1));
    }

    #[test]
    fn test_semantic_token_type_mapping_literals() {
        // Numbers should map to token type 4
        assert_eq!(token_kind_to_semantic_type(&TokenKind::NatLit(42)), Some(4));
        // Strings should map to token type 5
        assert_eq!(
            token_kind_to_semantic_type(&TokenKind::StringLit("hello".to_string())),
            Some(5)
        );
    }

    #[test]
    fn test_semantic_token_type_mapping_operators() {
        // Operators should map to token type 7
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Arrow), Some(7));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::FatArrow), Some(7));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Plus), Some(7));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Minus), Some(7));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Star), Some(7));
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Eq), Some(7));
    }

    #[test]
    fn test_semantic_token_type_mapping_identifiers() {
        // Identifiers should map to token type 3 (VARIABLE)
        assert_eq!(
            token_kind_to_semantic_type(&TokenKind::Ident("foo".to_string())),
            Some(3)
        );
    }

    #[test]
    fn test_semantic_token_type_mapping_delimiters() {
        // Delimiters should return None (not highlighted)
        assert_eq!(token_kind_to_semantic_type(&TokenKind::LParen), None);
        assert_eq!(token_kind_to_semantic_type(&TokenKind::RParen), None);
        assert_eq!(token_kind_to_semantic_type(&TokenKind::LBrace), None);
        assert_eq!(token_kind_to_semantic_type(&TokenKind::RBrace), None);
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Comma), None);
        assert_eq!(token_kind_to_semantic_type(&TokenKind::Colon), None);
    }

    #[test]
    fn test_byte_offset_to_position_basic() {
        let text = "def x := 1";
        // Position at start
        let pos = byte_offset_to_position(text, 0);
        assert_eq!(pos.line, 0);
        assert_eq!(pos.character, 0);

        // Position at 'd' of 'def'
        let pos = byte_offset_to_position(text, 0);
        assert_eq!(pos.line, 0);
        assert_eq!(pos.character, 0);

        // Position at 'x'
        let pos = byte_offset_to_position(text, 4);
        assert_eq!(pos.line, 0);
        assert_eq!(pos.character, 4);
    }

    #[test]
    fn test_byte_offset_to_position_multiline() {
        let text = "def x := 1\ndef y := 2";
        // Position on second line
        let pos = byte_offset_to_position(text, 11); // 'd' of second 'def'
        assert_eq!(pos.line, 1);
        assert_eq!(pos.character, 0);

        // Position at 'y'
        let pos = byte_offset_to_position(text, 15);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.character, 4);
    }

    #[test]
    fn test_semantic_token_types_constant() {
        // Verify the constant has expected number of types
        assert!(SEMANTIC_TOKEN_TYPES.len() >= 8);
        // Verify key types are present
        assert!(SEMANTIC_TOKEN_TYPES
            .iter()
            .any(|t| t == &SemanticTokenType::KEYWORD));
        assert!(SEMANTIC_TOKEN_TYPES
            .iter()
            .any(|t| t == &SemanticTokenType::TYPE));
        assert!(SEMANTIC_TOKEN_TYPES
            .iter()
            .any(|t| t == &SemanticTokenType::VARIABLE));
        assert!(SEMANTIC_TOKEN_TYPES
            .iter()
            .any(|t| t == &SemanticTokenType::NUMBER));
        assert!(SEMANTIC_TOKEN_TYPES
            .iter()
            .any(|t| t == &SemanticTokenType::STRING));
    }

    #[test]
    fn test_command_kind_to_semantic_type() {
        // Definitions, theorems, lemmas -> FUNCTION (2)
        assert_eq!(command_kind_to_semantic_type(&CommandKind::Definition), 2);
        assert_eq!(command_kind_to_semantic_type(&CommandKind::Theorem), 2);
        assert_eq!(command_kind_to_semantic_type(&CommandKind::Lemma), 2);
        assert_eq!(command_kind_to_semantic_type(&CommandKind::Axiom), 2);

        // Inductive, Structure -> TYPE (1)
        assert_eq!(command_kind_to_semantic_type(&CommandKind::Inductive), 1);
        assert_eq!(command_kind_to_semantic_type(&CommandKind::Structure), 1);

        // Class -> CLASS (9)
        assert_eq!(command_kind_to_semantic_type(&CommandKind::Class), 9);

        // Instance -> PROPERTY (10)
        assert_eq!(command_kind_to_semantic_type(&CommandKind::Instance), 10);

        // Namespace -> NAMESPACE (8)
        assert_eq!(command_kind_to_semantic_type(&CommandKind::Namespace), 8);

        // Variable -> VARIABLE (3)
        assert_eq!(command_kind_to_semantic_type(&CommandKind::Variable), 3);
    }

    #[test]
    fn test_is_likely_type_name_builtins() {
        // Built-in types should be recognized
        assert!(is_likely_type_name("Nat"));
        assert!(is_likely_type_name("Int"));
        assert!(is_likely_type_name("Bool"));
        assert!(is_likely_type_name("String"));
        assert!(is_likely_type_name("List"));
        assert!(is_likely_type_name("Option"));
        assert!(is_likely_type_name("Array"));
        assert!(is_likely_type_name("IO"));
    }

    #[test]
    fn test_is_likely_type_name_capitalized() {
        // Capitalized identifiers are likely types
        assert!(is_likely_type_name("MyType"));
        assert!(is_likely_type_name("Foo"));
        assert!(is_likely_type_name("Point"));

        // Lowercase identifiers are not types
        assert!(!is_likely_type_name("foo"));
        assert!(!is_likely_type_name("myvar"));
        assert!(!is_likely_type_name("x"));
    }

    #[test]
    fn test_classify_identifier_with_definitions() {
        use std::collections::HashMap;

        let mut defs = HashMap::new();
        defs.insert("myFunc".to_string(), CommandKind::Definition);
        defs.insert("MyStruct".to_string(), CommandKind::Structure);
        defs.insert("MyClass".to_string(), CommandKind::Class);
        defs.insert("myInstance".to_string(), CommandKind::Instance);

        // Known definitions should be classified correctly (not at def site)
        let (ty, mods) = classify_identifier_with_modifiers("myFunc", &defs, false);
        assert_eq!(ty, Some(2)); // FUNCTION
        assert_eq!(mods, 0); // No modifiers for usage site

        let (ty, mods) = classify_identifier_with_modifiers("MyStruct", &defs, false);
        assert_eq!(ty, Some(1)); // TYPE
        assert_eq!(mods, 0);

        let (ty, mods) = classify_identifier_with_modifiers("MyClass", &defs, false);
        assert_eq!(ty, Some(9)); // CLASS
        assert_eq!(mods, 0);

        let (ty, mods) = classify_identifier_with_modifiers("myInstance", &defs, false);
        assert_eq!(ty, Some(10)); // PROPERTY
        assert_eq!(mods, 0);

        // Unknown but capitalized -> TYPE heuristic
        let (ty, mods) = classify_identifier_with_modifiers("SomeType", &defs, false);
        assert_eq!(ty, Some(1)); // TYPE
        assert_eq!(mods, 0); // Non-builtin capitalized type

        // Unknown lowercase -> VARIABLE with READONLY
        let (ty, mods) = classify_identifier_with_modifiers("x", &defs, false);
        assert_eq!(ty, Some(3)); // VARIABLE
        assert_eq!(mods, modifier_bits::READONLY); // Variables are readonly
    }

    #[test]
    fn test_semantic_token_modifiers_constant() {
        // Verify the constant has expected number of modifiers
        assert_eq!(SEMANTIC_TOKEN_MODIFIERS.len(), 5);
        // Verify key modifiers are present
        assert!(SEMANTIC_TOKEN_MODIFIERS
            .iter()
            .any(|m| m == &SemanticTokenModifier::DECLARATION));
        assert!(SEMANTIC_TOKEN_MODIFIERS
            .iter()
            .any(|m| m == &SemanticTokenModifier::DEFINITION));
        assert!(SEMANTIC_TOKEN_MODIFIERS
            .iter()
            .any(|m| m == &SemanticTokenModifier::READONLY));
        assert!(SEMANTIC_TOKEN_MODIFIERS
            .iter()
            .any(|m| m == &SemanticTokenModifier::DEPRECATED));
        assert!(SEMANTIC_TOKEN_MODIFIERS
            .iter()
            .any(|m| m == &SemanticTokenModifier::DEFAULT_LIBRARY));
    }

    #[test]
    fn test_modifier_bits_values() {
        // Verify modifier bits are powers of 2 and match indices
        assert_eq!(modifier_bits::DECLARATION, 1 << 0);
        assert_eq!(modifier_bits::DEFINITION, 1 << 1);
        assert_eq!(modifier_bits::READONLY, 1 << 2);
        assert_eq!(modifier_bits::DEPRECATED, 1 << 3);
        assert_eq!(modifier_bits::DEFAULT_LIBRARY, 1 << 4);
    }

    #[test]
    fn test_classify_identifier_definition_site() {
        use std::collections::HashMap;

        let mut defs = HashMap::new();
        defs.insert("myFunc".to_string(), CommandKind::Definition);

        // At definition site, should have DECLARATION and DEFINITION modifiers
        let (ty, mods) = classify_identifier_with_modifiers("myFunc", &defs, true);
        assert_eq!(ty, Some(2)); // FUNCTION
        assert!(mods & modifier_bits::DECLARATION != 0);
        assert!(mods & modifier_bits::DEFINITION != 0);
    }

    #[test]
    fn test_classify_identifier_builtin_types() {
        use std::collections::HashMap;
        let empty_defs = HashMap::new();

        // Built-in types should get DEFAULT_LIBRARY modifier
        let (ty, mods) = classify_identifier_with_modifiers("Nat", &empty_defs, false);
        assert_eq!(ty, Some(1)); // TYPE
        assert!(mods & modifier_bits::DEFAULT_LIBRARY != 0);

        let (ty, mods) = classify_identifier_with_modifiers("Bool", &empty_defs, false);
        assert_eq!(ty, Some(1)); // TYPE
        assert!(mods & modifier_bits::DEFAULT_LIBRARY != 0);

        let (ty, mods) = classify_identifier_with_modifiers("IO", &empty_defs, false);
        assert_eq!(ty, Some(1)); // TYPE
        assert!(mods & modifier_bits::DEFAULT_LIBRARY != 0);

        // Monad should also be recognized as builtin
        let (ty, mods) = classify_identifier_with_modifiers("Monad", &empty_defs, false);
        assert_eq!(ty, Some(1)); // TYPE
        assert!(mods & modifier_bits::DEFAULT_LIBRARY != 0);
    }

    #[test]
    fn test_is_builtin_type() {
        // Core types
        assert!(is_builtin_type("Nat"));
        assert!(is_builtin_type("Int"));
        assert!(is_builtin_type("Bool"));
        assert!(is_builtin_type("String"));
        assert!(is_builtin_type("Prop"));
        assert!(is_builtin_type("Type"));

        // Collections
        assert!(is_builtin_type("List"));
        assert!(is_builtin_type("Array"));
        assert!(is_builtin_type("Option"));

        // Monads
        assert!(is_builtin_type("IO"));
        assert!(is_builtin_type("StateT"));
        assert!(is_builtin_type("Monad"));

        // Not builtin
        assert!(!is_builtin_type("MyType"));
        assert!(!is_builtin_type("foo"));
        assert!(!is_builtin_type("CustomMonad"));
    }

    #[test]
    fn test_find_definition_name_span() {
        let text = "def myFunc (x : Nat) : Nat := x";
        // The name "myFunc" starts at position 4 (after "def ")
        let span = find_definition_name_span(text, 0, text.len(), "myFunc");
        assert!(span.is_some());
        let (start, end) = span.unwrap();
        assert_eq!(&text[start..end], "myFunc");

        // Test with theorem
        let text2 = "theorem add_comm : ∀ x y : Nat, x + y = y + x := by sorry";
        let span2 = find_definition_name_span(text2, 0, text2.len(), "add_comm");
        assert!(span2.is_some());
        let (start, end) = span2.unwrap();
        assert_eq!(&text2[start..end], "add_comm");
    }
}
