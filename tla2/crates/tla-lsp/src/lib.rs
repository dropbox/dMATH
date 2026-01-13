//! TLA2 Language Server Protocol implementation
//!
//! Provides IDE features for TLA+ including:
//! - Diagnostics (parse and resolution errors)
//! - Document symbols (outline)
//! - Go to definition
//! - Find references
//! - Hover information
//! - Completion (keywords, stdlib modules/operators, local symbols)
//! - Workspace symbol search

use dashmap::DashMap;
use tla_core::{
    ast::{Module, Unit},
    lower, parse, resolve,
    stdlib::{get_module_operators, STDLIB_MODULES},
    FileId, ResolveResult, Span, Symbol, SymbolKind as TlaSymbolKind, SyntaxNode,
};
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

/// Document state stored for each open file
#[derive(Debug)]
struct DocumentState {
    /// Source text
    source: String,
    /// Lowered AST (if parse succeeded)
    module: Option<Module>,
    /// Resolution result (if lowering succeeded)
    resolve: Option<ResolveResult>,
    /// Parse errors
    parse_errors: Vec<tla_core::ParseError>,
    /// Lower errors
    lower_errors: Vec<tla_core::LowerError>,
}

impl DocumentState {
    fn new(source: String) -> Self {
        Self {
            source,
            module: None,
            resolve: None,
            parse_errors: Vec::new(),
            lower_errors: Vec::new(),
        }
    }

    /// Parse and analyze the document
    fn analyze(&mut self) {
        // Parse
        let parse_result = parse(&self.source);
        self.parse_errors = parse_result.errors.clone();

        if !parse_result.errors.is_empty() {
            self.module = None;
            self.resolve = None;
            return;
        }

        // Lower
        let tree = SyntaxNode::new_root(parse_result.green_node);
        let lower_result = lower(FileId(0), &tree);
        self.lower_errors = lower_result.errors.clone();

        if let Some(module) = lower_result.module {
            // Resolve
            let resolve_result = resolve(&module);
            self.resolve = Some(resolve_result);
            self.module = Some(module);
        } else {
            self.module = None;
            self.resolve = None;
        }
    }
}

/// TLA+ Language Server backend
pub struct TlaBackend {
    /// LSP client for sending notifications
    client: Client,
    /// Open documents
    documents: DashMap<Url, DocumentState>,
}

impl TlaBackend {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            documents: DashMap::new(),
        }
    }

    /// Convert a byte offset to LSP Position
    fn offset_to_position(source: &str, offset: u32) -> Position {
        let offset = offset as usize;
        let mut line = 0u32;
        let mut col = 0u32;
        for (i, c) in source.char_indices() {
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

    /// Convert LSP Position to byte offset
    fn position_to_offset(source: &str, pos: Position) -> u32 {
        let mut current_line = 0u32;
        let mut offset = 0usize;
        for c in source.chars() {
            if current_line == pos.line {
                break;
            }
            if c == '\n' {
                current_line += 1;
            }
            offset += c.len_utf8();
        }
        // Add column offset
        for (col, c) in source[offset..].chars().enumerate() {
            if col as u32 >= pos.character {
                break;
            }
            offset += c.len_utf8();
        }
        offset as u32
    }

    /// Convert Span to LSP Range
    fn span_to_range(source: &str, span: Span) -> Range {
        Range::new(
            Self::offset_to_position(source, span.start),
            Self::offset_to_position(source, span.end),
        )
    }

    /// Publish diagnostics for a document
    async fn publish_diagnostics(&self, uri: &Url) {
        let diagnostics = if let Some(doc) = self.documents.get(uri) {
            let mut diags = Vec::new();

            // Parse errors
            for err in &doc.parse_errors {
                diags.push(Diagnostic {
                    range: Range::new(
                        Self::offset_to_position(&doc.source, err.start),
                        Self::offset_to_position(&doc.source, err.end),
                    ),
                    severity: Some(DiagnosticSeverity::ERROR),
                    source: Some("tla2".to_string()),
                    message: err.message.clone(),
                    ..Default::default()
                });
            }

            // Lower errors
            for err in &doc.lower_errors {
                diags.push(Diagnostic {
                    range: Self::span_to_range(&doc.source, err.span),
                    severity: Some(DiagnosticSeverity::ERROR),
                    source: Some("tla2".to_string()),
                    message: err.message.clone(),
                    ..Default::default()
                });
            }

            // Resolve errors (as warnings - undefined refs might be from unloaded modules)
            if let Some(resolve) = &doc.resolve {
                for err in &resolve.errors {
                    diags.push(Diagnostic {
                        range: Self::span_to_range(&doc.source, err.span),
                        severity: Some(DiagnosticSeverity::WARNING),
                        source: Some("tla2".to_string()),
                        message: err.to_string(),
                        ..Default::default()
                    });
                }
            }

            diags
        } else {
            Vec::new()
        };

        self.client
            .publish_diagnostics(uri.clone(), diagnostics, None)
            .await;
    }

    /// Get document symbols from a module
    fn get_document_symbols(module: &Module, source: &str) -> Vec<DocumentSymbol> {
        let mut symbols = Vec::new();

        for unit in &module.units {
            match &unit.node {
                Unit::Variable(vars) => {
                    for var in vars {
                        #[allow(deprecated)]
                        symbols.push(DocumentSymbol {
                            name: var.node.clone(),
                            detail: Some("VARIABLE".to_string()),
                            kind: SymbolKind::VARIABLE,
                            tags: None,
                            deprecated: None,
                            range: Self::span_to_range(source, var.span),
                            selection_range: Self::span_to_range(source, var.span),
                            children: None,
                        });
                    }
                }
                Unit::Constant(consts) => {
                    for c in consts {
                        let detail = if let Some(arity) = c.arity {
                            format!("CONSTANT (arity {})", arity)
                        } else {
                            "CONSTANT".to_string()
                        };
                        #[allow(deprecated)]
                        symbols.push(DocumentSymbol {
                            name: c.name.node.clone(),
                            detail: Some(detail),
                            kind: SymbolKind::CONSTANT,
                            tags: None,
                            deprecated: None,
                            range: Self::span_to_range(source, c.name.span),
                            selection_range: Self::span_to_range(source, c.name.span),
                            children: None,
                        });
                    }
                }
                Unit::Recursive(decls) => {
                    for r in decls {
                        let detail = format!("RECURSIVE (arity {})", r.arity);
                        #[allow(deprecated)]
                        symbols.push(DocumentSymbol {
                            name: r.name.node.clone(),
                            detail: Some(detail),
                            kind: SymbolKind::FUNCTION,
                            tags: None,
                            deprecated: None,
                            range: Self::span_to_range(source, r.name.span),
                            selection_range: Self::span_to_range(source, r.name.span),
                            children: None,
                        });
                    }
                }
                Unit::Operator(op) => {
                    let kind = if op.params.is_empty() {
                        SymbolKind::CONSTANT
                    } else {
                        SymbolKind::FUNCTION
                    };
                    let detail = if op.params.is_empty() {
                        None
                    } else {
                        Some(format!(
                            "({})",
                            op.params
                                .iter()
                                .map(|p| p.name.node.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ))
                    };
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: op.name.node.clone(),
                        detail,
                        kind,
                        tags: None,
                        deprecated: None,
                        range: Self::span_to_range(source, unit.span),
                        selection_range: Self::span_to_range(source, op.name.span),
                        children: None,
                    });
                }
                Unit::Theorem(thm) => {
                    if let Some(name) = &thm.name {
                        #[allow(deprecated)]
                        symbols.push(DocumentSymbol {
                            name: name.node.clone(),
                            detail: Some("THEOREM".to_string()),
                            kind: SymbolKind::CLASS,
                            tags: None,
                            deprecated: None,
                            range: Self::span_to_range(source, unit.span),
                            selection_range: Self::span_to_range(source, name.span),
                            children: None,
                        });
                    }
                }
                Unit::Instance(inst) => {
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: format!("INSTANCE {}", inst.module.node),
                        detail: None,
                        kind: SymbolKind::MODULE,
                        tags: None,
                        deprecated: None,
                        range: Self::span_to_range(source, unit.span),
                        selection_range: Self::span_to_range(source, inst.module.span),
                        children: None,
                    });
                }
                Unit::Assume(_) => {
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: "ASSUME".to_string(),
                        detail: None,
                        kind: SymbolKind::BOOLEAN,
                        tags: None,
                        deprecated: None,
                        range: Self::span_to_range(source, unit.span),
                        selection_range: Self::span_to_range(source, unit.span),
                        children: None,
                    });
                }
                Unit::Separator => {}
            }
        }

        symbols
    }

    /// Find the symbol at a position
    fn find_symbol_at_position(resolve: &ResolveResult, offset: u32) -> Option<Span> {
        // Check references first (more specific)
        for (use_span, def_span) in &resolve.references {
            if offset >= use_span.start && offset <= use_span.end {
                return Some(*def_span);
            }
        }
        None
    }

    /// Find all references to a definition
    fn find_references_to_def(resolve: &ResolveResult, def_span: Span) -> Vec<Span> {
        resolve
            .references
            .iter()
            .filter(|(_, d)| *d == def_span)
            .map(|(u, _)| *u)
            .collect()
    }

    /// Find definition span at position (either definition itself or reference)
    fn find_definition_span_at_position(resolve: &ResolveResult, offset: u32) -> Option<Span> {
        // Check if cursor is on a reference
        for (use_span, def_span) in &resolve.references {
            if offset >= use_span.start && offset <= use_span.end {
                return Some(*def_span);
            }
        }
        // Check if cursor is on a definition
        for sym in &resolve.symbols {
            if offset >= sym.def_span.start && offset <= sym.def_span.end {
                return Some(sym.def_span);
            }
        }
        None
    }

    /// Get hover info for a symbol
    fn get_hover_info(resolve: &ResolveResult, offset: u32) -> Option<String> {
        // Check if on a definition
        for sym in &resolve.symbols {
            if offset >= sym.def_span.start && offset <= sym.def_span.end {
                return Some(Self::format_symbol_info(sym));
            }
        }
        // Check if on a reference
        for (use_span, def_span) in &resolve.references {
            if offset >= use_span.start && offset <= use_span.end {
                // Find the symbol
                for sym in &resolve.symbols {
                    if sym.def_span == *def_span {
                        return Some(Self::format_symbol_info(sym));
                    }
                }
            }
        }
        None
    }

    fn format_symbol_info(sym: &Symbol) -> String {
        let kind_str = match sym.kind {
            TlaSymbolKind::Variable => "VARIABLE",
            TlaSymbolKind::Constant => "CONSTANT",
            TlaSymbolKind::Operator => "OPERATOR",
            TlaSymbolKind::BoundVar => "bound variable",
            TlaSymbolKind::OpParam => "parameter",
            TlaSymbolKind::Module => "MODULE",
        };
        let local_str = if sym.local { "LOCAL " } else { "" };
        if sym.arity > 0 {
            format!(
                "{}{} {} (arity {})",
                local_str, kind_str, sym.name, sym.arity
            )
        } else {
            format!("{}{} {}", local_str, kind_str, sym.name)
        }
    }

    /// Get completion items for TLA+ keywords
    fn get_keyword_completions() -> Vec<CompletionItem> {
        const KEYWORDS: &[(&str, &str)] = &[
            ("MODULE", "Module declaration"),
            ("EXTENDS", "Import module definitions"),
            ("VARIABLE", "Declare state variable(s)"),
            ("VARIABLES", "Declare state variable(s)"),
            ("CONSTANT", "Declare constant(s)"),
            ("CONSTANTS", "Declare constant(s)"),
            ("ASSUME", "Add assumption"),
            ("THEOREM", "Declare theorem"),
            ("LEMMA", "Declare lemma"),
            ("PROPOSITION", "Declare proposition"),
            ("COROLLARY", "Declare corollary"),
            ("AXIOM", "Declare axiom"),
            ("LOCAL", "Make definition module-local"),
            ("INSTANCE", "Instantiate module"),
            ("LET", "Local definitions"),
            ("IN", "Body of LET expression"),
            ("IF", "Conditional expression"),
            ("THEN", "Then branch"),
            ("ELSE", "Else branch"),
            ("CASE", "Case expression"),
            ("OTHER", "Default case branch"),
            ("CHOOSE", "Choice operator"),
            ("EXCEPT", "Function update"),
            ("ENABLED", "Enabled predicate"),
            ("UNCHANGED", "Unchanged predicate"),
            ("SUBSET", "Powerset operator"),
            ("UNION", "Generalized union"),
            ("DOMAIN", "Function domain"),
            ("BOOLEAN", "Set {TRUE, FALSE}"),
            ("STRING", "Set of strings"),
            ("TRUE", "Boolean true"),
            ("FALSE", "Boolean false"),
            ("LAMBDA", "Lambda expression"),
            ("WF_", "Weak fairness"),
            ("SF_", "Strong fairness"),
            // Proof keywords
            ("PROOF", "Begin proof"),
            ("BY", "Proof justification"),
            ("OBVIOUS", "Obvious proof step"),
            ("OMITTED", "Omitted proof"),
            ("QED", "End of proof"),
            ("PROVE", "Proof goal"),
            ("SUFFICES", "Suffices to prove"),
            ("HAVE", "Assert in proof"),
            ("TAKE", "Universal instantiation"),
            ("WITNESS", "Existential instantiation"),
            ("PICK", "Choose witness"),
            ("DEFINE", "Local definition in proof"),
            ("HIDE", "Hide fact in proof"),
            ("USE", "Use fact in proof"),
            ("DEFS", "Use definitions"),
            ("DEF", "Use definition"),
            ("ONLY", "Use only specified facts"),
            ("NEW", "Introduce new constant"),
        ];

        KEYWORDS
            .iter()
            .map(|(name, doc)| CompletionItem {
                label: name.to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some(doc.to_string()),
                ..Default::default()
            })
            .collect()
    }

    /// Get completion items for standard library modules
    fn get_stdlib_module_completions() -> Vec<CompletionItem> {
        STDLIB_MODULES
            .iter()
            .map(|name| CompletionItem {
                label: name.to_string(),
                kind: Some(CompletionItemKind::MODULE),
                detail: Some("Standard library module".to_string()),
                ..Default::default()
            })
            .collect()
    }

    /// Get completion items for operators from extended modules
    fn get_stdlib_operator_completions(module: &Module) -> Vec<CompletionItem> {
        let mut completions = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Process extended modules and their transitive dependencies
        fn process_module(
            module_name: &str,
            completions: &mut Vec<CompletionItem>,
            seen: &mut std::collections::HashSet<String>,
        ) {
            if seen.contains(module_name) {
                return;
            }
            seen.insert(module_name.to_string());

            // Handle transitive extends
            match module_name {
                "Integers" => process_module("Naturals", completions, seen),
                "Reals" => process_module("Integers", completions, seen),
                _ => {}
            }

            // Add operators from this module
            for (name, arity) in get_module_operators(module_name) {
                let detail = if *arity == 0 {
                    format!("{} (constant)", module_name)
                } else if *arity < 0 {
                    format!("{} (variadic)", module_name)
                } else {
                    format!("{} (arity {})", module_name, arity)
                };

                completions.push(CompletionItem {
                    label: name.to_string(),
                    kind: Some(CompletionItemKind::FUNCTION),
                    detail: Some(detail),
                    ..Default::default()
                });
            }

            // Add built-in constants for certain modules
            match module_name {
                "Naturals" => {
                    completions.push(CompletionItem {
                        label: "Nat".to_string(),
                        kind: Some(CompletionItemKind::CONSTANT),
                        detail: Some("Set of natural numbers".to_string()),
                        ..Default::default()
                    });
                }
                "Integers" => {
                    completions.push(CompletionItem {
                        label: "Int".to_string(),
                        kind: Some(CompletionItemKind::CONSTANT),
                        detail: Some("Set of integers".to_string()),
                        ..Default::default()
                    });
                }
                "Reals" => {
                    completions.push(CompletionItem {
                        label: "Real".to_string(),
                        kind: Some(CompletionItemKind::CONSTANT),
                        detail: Some("Set of real numbers".to_string()),
                        ..Default::default()
                    });
                    completions.push(CompletionItem {
                        label: "Infinity".to_string(),
                        kind: Some(CompletionItemKind::CONSTANT),
                        detail: Some("Infinity constant".to_string()),
                        ..Default::default()
                    });
                }
                _ => {}
            }
        }

        for ext in &module.extends {
            process_module(&ext.node, &mut completions, &mut seen);
        }

        completions
    }

    /// Get completion items for local symbols
    fn get_local_symbol_completions(resolve: &ResolveResult) -> Vec<CompletionItem> {
        resolve
            .symbols
            .iter()
            .filter(|sym| {
                // Only include module-level definitions, not bound variables
                matches!(
                    sym.kind,
                    TlaSymbolKind::Variable
                        | TlaSymbolKind::Constant
                        | TlaSymbolKind::Operator
                        | TlaSymbolKind::Module
                )
            })
            .map(|sym| {
                let kind = match sym.kind {
                    TlaSymbolKind::Variable => CompletionItemKind::VARIABLE,
                    TlaSymbolKind::Constant => CompletionItemKind::CONSTANT,
                    TlaSymbolKind::Operator => {
                        if sym.arity > 0 {
                            CompletionItemKind::FUNCTION
                        } else {
                            CompletionItemKind::CONSTANT
                        }
                    }
                    TlaSymbolKind::Module => CompletionItemKind::MODULE,
                    _ => CompletionItemKind::TEXT,
                };

                let detail = if sym.arity > 0 {
                    Some(format!("arity {}", sym.arity))
                } else {
                    None
                };

                CompletionItem {
                    label: sym.name.clone(),
                    kind: Some(kind),
                    detail,
                    ..Default::default()
                }
            })
            .collect()
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for TlaBackend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                document_symbol_provider: Some(OneOf::Left(true)),
                definition_provider: Some(OneOf::Left(true)),
                references_provider: Some(OneOf::Left(true)),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![
                        "\\".to_string(), // For \A, \E, etc.
                        ".".to_string(),  // For module.operator
                        "_".to_string(),  // For WF_, SF_
                    ]),
                    resolve_provider: Some(false),
                    ..Default::default()
                }),
                workspace_symbol_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "tla2-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        tracing::info!("TLA2 Language Server initialized");
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let mut doc = DocumentState::new(params.text_document.text);
        doc.analyze();
        self.documents.insert(uri.clone(), doc);
        self.publish_diagnostics(&uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        if let Some(change) = params.content_changes.into_iter().last() {
            let mut doc = DocumentState::new(change.text);
            doc.analyze();
            self.documents.insert(uri.clone(), doc);
            self.publish_diagnostics(&uri).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        self.documents.remove(&params.text_document.uri);
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;
        if let Some(doc) = self.documents.get(uri) {
            if let Some(module) = &doc.module {
                let symbols = Self::get_document_symbols(module, &doc.source);
                return Ok(Some(DocumentSymbolResponse::Nested(symbols)));
            }
        }
        Ok(None)
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        if let Some(doc) = self.documents.get(uri) {
            if let Some(resolve) = &doc.resolve {
                let offset = Self::position_to_offset(&doc.source, pos);
                if let Some(def_span) = Self::find_symbol_at_position(resolve, offset) {
                    let range = Self::span_to_range(&doc.source, def_span);
                    return Ok(Some(GotoDefinitionResponse::Scalar(Location::new(
                        uri.clone(),
                        range,
                    ))));
                }
            }
        }
        Ok(None)
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;

        if let Some(doc) = self.documents.get(uri) {
            if let Some(resolve) = &doc.resolve {
                let offset = Self::position_to_offset(&doc.source, pos);
                if let Some(def_span) = Self::find_definition_span_at_position(resolve, offset) {
                    let refs = Self::find_references_to_def(resolve, def_span);
                    let mut locations: Vec<Location> = refs
                        .into_iter()
                        .map(|span| {
                            Location::new(uri.clone(), Self::span_to_range(&doc.source, span))
                        })
                        .collect();

                    // Include definition itself if requested
                    if params.context.include_declaration {
                        locations.push(Location::new(
                            uri.clone(),
                            Self::span_to_range(&doc.source, def_span),
                        ));
                    }

                    return Ok(Some(locations));
                }
            }
        }
        Ok(None)
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;

        if let Some(doc) = self.documents.get(uri) {
            if let Some(resolve) = &doc.resolve {
                let offset = Self::position_to_offset(&doc.source, pos);
                if let Some(info) = Self::get_hover_info(resolve, offset) {
                    return Ok(Some(Hover {
                        contents: HoverContents::Markup(MarkupContent {
                            kind: MarkupKind::PlainText,
                            value: info,
                        }),
                        range: None,
                    }));
                }
            }
        }
        Ok(None)
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;

        let mut items = Vec::new();

        // Always provide keywords
        items.extend(Self::get_keyword_completions());

        // Always provide standard library modules (for EXTENDS)
        items.extend(Self::get_stdlib_module_completions());

        // If we have a parsed document, provide context-specific completions
        if let Some(doc) = self.documents.get(uri) {
            // Add local symbols
            if let Some(resolve) = &doc.resolve {
                items.extend(Self::get_local_symbol_completions(resolve));
            }

            // Add operators from extended modules
            if let Some(module) = &doc.module {
                items.extend(Self::get_stdlib_operator_completions(module));
            }
        }

        // Deduplicate by label
        let mut seen = std::collections::HashSet::new();
        items.retain(|item| seen.insert(item.label.clone()));

        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn symbol(
        &self,
        params: WorkspaceSymbolParams,
    ) -> Result<Option<Vec<SymbolInformation>>> {
        let query = params.query.to_lowercase();
        let mut symbols = Vec::new();

        // Search through all open documents
        for entry in self.documents.iter() {
            let uri = entry.key().clone();
            let doc = entry.value();

            if let Some(module) = &doc.module {
                // Get symbols from module
                for unit in &module.units {
                    let (name, kind) = match &unit.node {
                        Unit::Variable(vars) => {
                            for var in vars {
                                if var.node.to_lowercase().contains(&query) {
                                    #[allow(deprecated)]
                                    symbols.push(SymbolInformation {
                                        name: var.node.clone(),
                                        kind: SymbolKind::VARIABLE,
                                        tags: None,
                                        deprecated: None,
                                        location: Location::new(
                                            uri.clone(),
                                            Self::span_to_range(&doc.source, var.span),
                                        ),
                                        container_name: Some(module.name.node.clone()),
                                    });
                                }
                            }
                            continue;
                        }
                        Unit::Constant(consts) => {
                            for c in consts {
                                if c.name.node.to_lowercase().contains(&query) {
                                    #[allow(deprecated)]
                                    symbols.push(SymbolInformation {
                                        name: c.name.node.clone(),
                                        kind: SymbolKind::CONSTANT,
                                        tags: None,
                                        deprecated: None,
                                        location: Location::new(
                                            uri.clone(),
                                            Self::span_to_range(&doc.source, c.name.span),
                                        ),
                                        container_name: Some(module.name.node.clone()),
                                    });
                                }
                            }
                            continue;
                        }
                        Unit::Operator(op) => (op.name.node.clone(), SymbolKind::FUNCTION),
                        Unit::Theorem(thm) => {
                            if let Some(name) = &thm.name {
                                (name.node.clone(), SymbolKind::CLASS)
                            } else {
                                continue;
                            }
                        }
                        _ => continue,
                    };

                    if name.to_lowercase().contains(&query) {
                        #[allow(deprecated)]
                        symbols.push(SymbolInformation {
                            name,
                            kind,
                            tags: None,
                            deprecated: None,
                            location: Location::new(
                                uri.clone(),
                                Self::span_to_range(&doc.source, unit.span),
                            ),
                            container_name: Some(module.name.node.clone()),
                        });
                    }
                }
            }
        }

        Ok(Some(symbols))
    }
}

/// Run the LSP server on stdin/stdout
pub async fn run_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = tower_lsp::LspService::new(TlaBackend::new);
    tower_lsp::Server::new(stdin, stdout, socket)
        .serve(service)
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MODULE: &str = r#"
---- MODULE Test ----
EXTENDS Naturals, Sequences

VARIABLE x, y
CONSTANT N

Init == x = 0 /\ y = 0

Inc(n) == x' = x + n /\ y' = y

Next == Inc(1) \/ Inc(2)

Spec == Init /\ [][Next]_<<x, y>>
====
"#;

    #[test]
    fn test_offset_to_position() {
        let source = "line1\nline2\nline3";
        // Position at start of line2
        let pos = TlaBackend::offset_to_position(source, 6);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.character, 0);

        // Position in middle of line2
        let pos = TlaBackend::offset_to_position(source, 8);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.character, 2);

        // Position at start
        let pos = TlaBackend::offset_to_position(source, 0);
        assert_eq!(pos.line, 0);
        assert_eq!(pos.character, 0);
    }

    #[test]
    fn test_position_to_offset() {
        let source = "line1\nline2\nline3";
        // Start of line2
        let offset = TlaBackend::position_to_offset(source, Position::new(1, 0));
        assert_eq!(offset, 6);

        // Character 2 of line2
        let offset = TlaBackend::position_to_offset(source, Position::new(1, 2));
        assert_eq!(offset, 8);

        // Start of file
        let offset = TlaBackend::position_to_offset(source, Position::new(0, 0));
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_keyword_completions() {
        let completions = TlaBackend::get_keyword_completions();
        let labels: Vec<_> = completions.iter().map(|c| c.label.as_str()).collect();

        // Check for essential keywords
        assert!(labels.contains(&"MODULE"));
        assert!(labels.contains(&"EXTENDS"));
        assert!(labels.contains(&"VARIABLE"));
        assert!(labels.contains(&"CONSTANT"));
        assert!(labels.contains(&"THEOREM"));
        assert!(labels.contains(&"LET"));
        assert!(labels.contains(&"IF"));
        assert!(labels.contains(&"CHOOSE"));

        // Check all items have KEYWORD kind
        for item in &completions {
            assert_eq!(item.kind, Some(CompletionItemKind::KEYWORD));
        }
    }

    #[test]
    fn test_stdlib_module_completions() {
        let completions = TlaBackend::get_stdlib_module_completions();
        let labels: Vec<_> = completions.iter().map(|c| c.label.as_str()).collect();

        // Check for common stdlib modules
        assert!(labels.contains(&"Naturals"));
        assert!(labels.contains(&"Integers"));
        assert!(labels.contains(&"Sequences"));
        assert!(labels.contains(&"FiniteSets"));
        assert!(labels.contains(&"TLC"));

        // Check all items have MODULE kind
        for item in &completions {
            assert_eq!(item.kind, Some(CompletionItemKind::MODULE));
        }
    }

    #[test]
    fn test_document_state_analyze() {
        let mut doc = DocumentState::new(TEST_MODULE.to_string());
        doc.analyze();

        // Should parse successfully
        assert!(doc.parse_errors.is_empty());
        assert!(doc.lower_errors.is_empty());
        assert!(doc.module.is_some());
        assert!(doc.resolve.is_some());

        let module = doc.module.as_ref().unwrap();
        assert_eq!(module.name.node, "Test");

        // Check EXTENDS
        let extends: Vec<_> = module.extends.iter().map(|e| e.node.as_str()).collect();
        assert!(extends.contains(&"Naturals"));
        assert!(extends.contains(&"Sequences"));
    }

    #[test]
    fn test_document_symbols() {
        let mut doc = DocumentState::new(TEST_MODULE.to_string());
        doc.analyze();

        let module = doc.module.as_ref().unwrap();
        let symbols = TlaBackend::get_document_symbols(module, &doc.source);

        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();

        // Check variables
        assert!(names.contains(&"x"));
        assert!(names.contains(&"y"));

        // Check constant
        assert!(names.contains(&"N"));

        // Check operators
        assert!(names.contains(&"Init"));
        assert!(names.contains(&"Inc"));
        assert!(names.contains(&"Next"));
        assert!(names.contains(&"Spec"));
    }

    #[test]
    fn test_local_symbol_completions() {
        let mut doc = DocumentState::new(TEST_MODULE.to_string());
        doc.analyze();

        let resolve = doc.resolve.as_ref().unwrap();
        let completions = TlaBackend::get_local_symbol_completions(resolve);
        let labels: Vec<_> = completions.iter().map(|c| c.label.as_str()).collect();

        // Variables and constants
        assert!(labels.contains(&"x"));
        assert!(labels.contains(&"y"));
        assert!(labels.contains(&"N"));

        // Operators
        assert!(labels.contains(&"Init"));
        assert!(labels.contains(&"Inc"));
        assert!(labels.contains(&"Next"));
        assert!(labels.contains(&"Spec"));
    }

    #[test]
    fn test_stdlib_operator_completions() {
        let mut doc = DocumentState::new(TEST_MODULE.to_string());
        doc.analyze();

        let module = doc.module.as_ref().unwrap();
        let completions = TlaBackend::get_stdlib_operator_completions(module);
        let labels: Vec<_> = completions.iter().map(|c| c.label.as_str()).collect();

        // Operators from Naturals (extended)
        assert!(labels.contains(&"Nat"));

        // Operators from Sequences (extended)
        assert!(labels.contains(&"Seq"));
        assert!(labels.contains(&"Len"));
        assert!(labels.contains(&"Head"));
        assert!(labels.contains(&"Tail"));
        assert!(labels.contains(&"Append"));
    }

    #[test]
    fn test_hover_info() {
        let mut doc = DocumentState::new(TEST_MODULE.to_string());
        doc.analyze();

        let resolve = doc.resolve.as_ref().unwrap();

        // Find the Init operator definition
        let init_sym = resolve
            .symbols
            .iter()
            .find(|s| s.name == "Init")
            .expect("Init should be defined");

        let info = TlaBackend::get_hover_info(resolve, init_sym.def_span.start + 1);
        assert!(info.is_some());
        let info = info.unwrap();
        assert!(info.contains("OPERATOR"));
        assert!(info.contains("Init"));
    }

    #[test]
    fn test_span_to_range() {
        use tla_core::FileId;
        let source = "line1\nline2\nline3";
        let span = Span::new(FileId(0), 6, 11); // "line2"
        let range = TlaBackend::span_to_range(source, span);

        assert_eq!(range.start.line, 1);
        assert_eq!(range.start.character, 0);
        assert_eq!(range.end.line, 1);
        assert_eq!(range.end.character, 5);
    }

    #[test]
    fn test_parse_error_diagnostics() {
        let mut doc = DocumentState::new("---- MODULE Broken ----\nVARIABLE".to_string());
        doc.analyze();

        // Should have parse errors
        assert!(!doc.parse_errors.is_empty());
        assert!(doc.module.is_none());
    }

    #[test]
    fn test_goto_definition() {
        let mut doc = DocumentState::new(TEST_MODULE.to_string());
        doc.analyze();

        let resolve = doc.resolve.as_ref().unwrap();

        // Find where 'x' variable is defined
        let x_var = resolve
            .symbols
            .iter()
            .find(|s| s.name == "x")
            .expect("x should be defined");
        let x_def_span = x_var.def_span;

        // Find 'Init' operator
        let init_sym = resolve
            .symbols
            .iter()
            .find(|s| s.name == "Init")
            .expect("Init should be defined");
        let init_def_span = init_sym.def_span;

        // Check references exist
        assert!(
            !resolve.references.is_empty(),
            "Should have some references"
        );

        // Find a reference to x - should resolve back to x's definition
        for (use_span, def_span) in &resolve.references {
            if *def_span == x_def_span {
                // Found a reference to x - test that we can navigate to definition
                let found_def = TlaBackend::find_symbol_at_position(resolve, use_span.start + 1);
                assert_eq!(found_def, Some(x_def_span));
                break;
            }
        }

        // Find a reference to Init - should resolve to Init's definition
        for (use_span, def_span) in &resolve.references {
            if *def_span == init_def_span {
                let found_def = TlaBackend::find_symbol_at_position(resolve, use_span.start + 1);
                assert_eq!(found_def, Some(init_def_span));
                break;
            }
        }
    }

    #[test]
    fn test_find_references() {
        let mut doc = DocumentState::new(TEST_MODULE.to_string());
        doc.analyze();

        let resolve = doc.resolve.as_ref().unwrap();

        // Find 'x' variable definition
        let x_var = resolve
            .symbols
            .iter()
            .find(|s| s.name == "x")
            .expect("x should be defined");
        let x_def_span = x_var.def_span;

        // Find all references to x
        let refs = TlaBackend::find_references_to_def(resolve, x_def_span);

        // x is used in Init (x = 0), Inc (x' = x + n), and Spec (<<x, y>>)
        // At least 2 references expected
        assert!(
            refs.len() >= 2,
            "Expected at least 2 references to x, found {}",
            refs.len()
        );

        // Test that clicking on definition also finds references
        let def_span = TlaBackend::find_definition_span_at_position(resolve, x_def_span.start + 1);
        assert_eq!(def_span, Some(x_def_span), "Should find definition span");
    }

    #[test]
    fn test_find_references_from_reference() {
        let mut doc = DocumentState::new(TEST_MODULE.to_string());
        doc.analyze();

        let resolve = doc.resolve.as_ref().unwrap();

        // Find 'Inc' operator
        let inc_sym = resolve
            .symbols
            .iter()
            .find(|s| s.name == "Inc")
            .expect("Inc should be defined");
        let inc_def_span = inc_sym.def_span;

        // Inc is called twice in Next: "Inc(1) \/ Inc(2)"
        let refs = TlaBackend::find_references_to_def(resolve, inc_def_span);
        assert!(
            refs.len() >= 2,
            "Expected at least 2 references to Inc, found {}",
            refs.len()
        );

        // Test finding definition from a reference position
        if let Some((use_span, _)) = resolve.references.iter().find(|(_, d)| *d == inc_def_span) {
            let found_def =
                TlaBackend::find_definition_span_at_position(resolve, use_span.start + 1);
            assert_eq!(found_def, Some(inc_def_span));
        }
    }
}
