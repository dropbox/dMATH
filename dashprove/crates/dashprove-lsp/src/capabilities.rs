//! LSP server capabilities
//!
//! Defines what features the USL language server supports.

use crate::commands::SUPPORTED_COMMANDS;
use crate::semantic_tokens::semantic_tokens_legend;
use tower_lsp::lsp_types::{
    CallHierarchyServerCapability, CodeActionKind, CodeActionOptions, CodeActionProviderCapability,
    CodeLensOptions, CompletionOptions, ExecuteCommandOptions, FoldingRangeProviderCapability,
    HoverProviderCapability, InlayHintOptions, InlayHintServerCapabilities,
    LinkedEditingRangeServerCapabilities, MonikerOptions, MonikerServerCapabilities, OneOf,
    RenameOptions, SelectionRangeProviderCapability, SemanticTokensFullOptions,
    SemanticTokensOptions, SemanticTokensServerCapabilities, ServerCapabilities,
    SignatureHelpOptions, TextDocumentSyncCapability, TextDocumentSyncKind,
    TextDocumentSyncOptions,
};

/// Build the server capabilities to advertise to clients.
#[must_use]
pub fn server_capabilities() -> ServerCapabilities {
    ServerCapabilities {
        // Full document sync (receive entire document on change)
        text_document_sync: Some(TextDocumentSyncCapability::Options(
            TextDocumentSyncOptions {
                open_close: Some(true),
                change: Some(TextDocumentSyncKind::FULL),
                will_save: None,
                will_save_wait_until: None,
                save: None,
            },
        )),

        // Hover support (show type information on hover)
        hover_provider: Some(HoverProviderCapability::Simple(true)),

        // Completion support (suggest keywords, types, properties)
        completion_provider: Some(CompletionOptions {
            resolve_provider: Some(false),
            trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
            all_commit_characters: None,
            work_done_progress_options: Default::default(),
            completion_item: None,
        }),

        // Definition provider (go to type definition)
        definition_provider: Some(OneOf::Left(true)),

        // Document symbols for types and properties
        document_symbol_provider: Some(OneOf::Left(true)),

        // References provider (find all usages)
        references_provider: Some(OneOf::Left(true)),

        // Signature help for contract parameter hints
        signature_help_provider: Some(SignatureHelpOptions {
            trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
            retrigger_characters: Some(vec![",".to_string()]),
            work_done_progress_options: Default::default(),
        }),

        // Rename support (with prepare for validation)
        rename_provider: Some(OneOf::Right(RenameOptions {
            prepare_provider: Some(true),
            work_done_progress_options: Default::default(),
        })),

        // We don't support these yet
        declaration_provider: None,
        type_definition_provider: None,
        implementation_provider: None,
        // Document highlight (highlight all occurrences of symbol under cursor)
        document_highlight_provider: Some(OneOf::Left(true)),
        workspace_symbol_provider: Some(OneOf::Left(true)),
        // Code actions for quick fixes and refactoring
        code_action_provider: Some(CodeActionProviderCapability::Options(CodeActionOptions {
            code_action_kinds: Some(vec![
                CodeActionKind::QUICKFIX,
                CodeActionKind::REFACTOR_EXTRACT,
            ]),
            work_done_progress_options: Default::default(),
            resolve_provider: Some(false),
        })),
        // Code lenses for verification actions
        code_lens_provider: Some(CodeLensOptions {
            resolve_provider: Some(false),
        }),
        // Document formatting support
        document_formatting_provider: Some(OneOf::Left(true)),
        document_range_formatting_provider: Some(OneOf::Left(true)),
        document_on_type_formatting_provider: None,
        document_link_provider: None,
        color_provider: None,
        // Folding ranges for collapsible blocks
        folding_range_provider: Some(FoldingRangeProviderCapability::Simple(true)),
        // Execute command for verification actions triggered from code lenses
        execute_command_provider: Some(ExecuteCommandOptions {
            commands: SUPPORTED_COMMANDS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            work_done_progress_options: Default::default(),
        }),
        workspace: None,
        // Call hierarchy for navigating type and property references
        call_hierarchy_provider: Some(CallHierarchyServerCapability::Simple(true)),
        // Semantic tokens for syntax highlighting
        semantic_tokens_provider: Some(SemanticTokensServerCapabilities::SemanticTokensOptions(
            SemanticTokensOptions {
                legend: semantic_tokens_legend(),
                range: Some(true),
                full: Some(SemanticTokensFullOptions::Bool(true)),
                work_done_progress_options: Default::default(),
            },
        )),
        moniker_provider: Some(OneOf::Right(MonikerServerCapabilities::Options(
            MonikerOptions {
                work_done_progress_options: Default::default(),
            },
        ))),
        // Linked editing ranges for simultaneous identifier editing
        linked_editing_range_provider: Some(LinkedEditingRangeServerCapabilities::Simple(true)),
        inline_value_provider: None,
        // Inlay hints for type annotations
        inlay_hint_provider: Some(OneOf::Right(InlayHintServerCapabilities::Options(
            InlayHintOptions {
                work_done_progress_options: Default::default(),
                resolve_provider: Some(false),
            },
        ))),
        diagnostic_provider: None,
        // Selection range support (expand selection based on syntax)
        selection_range_provider: Some(SelectionRangeProviderCapability::Simple(true)),
        position_encoding: None,
        experimental: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_include_document_symbols() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.document_symbol_provider,
            Some(OneOf::Left(true))
        ));
    }

    #[test]
    fn capabilities_include_workspace_symbols() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.workspace_symbol_provider,
            Some(OneOf::Left(true))
        ));
    }

    #[test]
    fn capabilities_include_references() {
        let caps = server_capabilities();
        assert!(matches!(caps.references_provider, Some(OneOf::Left(true))));
    }

    #[test]
    fn capabilities_include_rename_with_prepare() {
        let caps = server_capabilities();
        match &caps.rename_provider {
            Some(OneOf::Right(opts)) => {
                assert_eq!(opts.prepare_provider, Some(true));
            }
            _ => panic!("Expected rename provider with options"),
        }
    }

    #[test]
    fn capabilities_include_semantic_tokens() {
        let caps = server_capabilities();
        assert!(caps.semantic_tokens_provider.is_some());
        match &caps.semantic_tokens_provider {
            Some(SemanticTokensServerCapabilities::SemanticTokensOptions(opts)) => {
                assert!(!opts.legend.token_types.is_empty());
                assert!(matches!(
                    opts.full,
                    Some(SemanticTokensFullOptions::Bool(true))
                ));
                assert_eq!(opts.range, Some(true));
            }
            _ => panic!("Expected semantic tokens options"),
        }
    }

    #[test]
    fn capabilities_include_code_actions() {
        let caps = server_capabilities();
        match &caps.code_action_provider {
            Some(CodeActionProviderCapability::Options(opts)) => {
                let kinds = opts.code_action_kinds.as_ref().expect("should have kinds");
                assert!(kinds.contains(&CodeActionKind::QUICKFIX));
                assert!(kinds.contains(&CodeActionKind::REFACTOR_EXTRACT));
            }
            _ => panic!("Expected code action provider with options"),
        }
    }

    #[test]
    fn capabilities_include_folding_ranges() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.folding_range_provider,
            Some(FoldingRangeProviderCapability::Simple(true))
        ));
    }

    #[test]
    fn capabilities_include_inlay_hints() {
        let caps = server_capabilities();
        assert!(caps.inlay_hint_provider.is_some());
        match &caps.inlay_hint_provider {
            Some(OneOf::Right(InlayHintServerCapabilities::Options(opts))) => {
                // We don't support resolve
                assert_eq!(opts.resolve_provider, Some(false));
            }
            _ => panic!("Expected inlay hint options"),
        }
    }

    #[test]
    fn capabilities_include_signature_help() {
        let caps = server_capabilities();
        assert!(caps.signature_help_provider.is_some());
        let opts = caps
            .signature_help_provider
            .as_ref()
            .expect("should have signature help");
        assert!(opts
            .trigger_characters
            .as_ref()
            .is_some_and(|t| t.contains(&"(".to_string())));
        assert!(opts
            .trigger_characters
            .as_ref()
            .is_some_and(|t| t.contains(&",".to_string())));
    }

    #[test]
    fn capabilities_include_document_formatting() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.document_formatting_provider,
            Some(OneOf::Left(true))
        ));
    }

    #[test]
    fn capabilities_include_document_highlight() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.document_highlight_provider,
            Some(OneOf::Left(true))
        ));
    }

    #[test]
    fn capabilities_include_code_lens() {
        let caps = server_capabilities();
        assert!(caps.code_lens_provider.is_some());
        let opts = caps.code_lens_provider.as_ref().unwrap();
        // We don't support resolve
        assert_eq!(opts.resolve_provider, Some(false));
    }

    #[test]
    fn capabilities_include_execute_command() {
        let caps = server_capabilities();
        assert!(caps.execute_command_provider.is_some());
        let opts = caps.execute_command_provider.as_ref().unwrap();
        assert_eq!(opts.commands.len(), 7);
        assert!(opts.commands.contains(&"dashprove.verify".to_string()));
        assert!(opts
            .commands
            .contains(&"dashprove.showBackendInfo".to_string()));
        assert!(opts
            .commands
            .contains(&"dashprove.explainDiagnostic".to_string()));
        assert!(opts
            .commands
            .contains(&"dashprove.suggestTactics".to_string()));
        assert!(opts
            .commands
            .contains(&"dashprove.recommendBackend".to_string()));
        assert!(opts
            .commands
            .contains(&"dashprove.compilationGuidance".to_string()));
        assert!(opts
            .commands
            .contains(&"dashprove.analyzeWorkspace".to_string()));
    }

    #[test]
    fn capabilities_include_document_range_formatting() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.document_range_formatting_provider,
            Some(OneOf::Left(true))
        ));
    }

    #[test]
    fn capabilities_include_selection_range() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.selection_range_provider,
            Some(SelectionRangeProviderCapability::Simple(true))
        ));
    }

    #[test]
    fn capabilities_include_call_hierarchy() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.call_hierarchy_provider,
            Some(CallHierarchyServerCapability::Simple(true))
        ));
    }

    #[test]
    fn capabilities_include_linked_editing_range() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.linked_editing_range_provider,
            Some(LinkedEditingRangeServerCapabilities::Simple(true))
        ));
    }

    #[test]
    fn capabilities_include_moniker_provider() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.moniker_provider,
            Some(OneOf::Right(MonikerServerCapabilities::Options(_)))
        ));
    }
}
