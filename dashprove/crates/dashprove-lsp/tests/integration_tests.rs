//! Integration tests for dashprove-lsp crate
//!
//! Tests the Language Server Protocol features for USL documents including:
//! - Document management and analysis
//! - Server capabilities advertisement
//! - Semantic tokens for syntax highlighting
//! - Code actions for quick fixes and refactoring
//! - Folding ranges for collapsible blocks
//! - Code lenses for verification actions
//! - Selection ranges for syntax-aware selection

use dashprove_lsp::{server_capabilities, Document, DocumentStore, UslLanguageServer};
use dashprove_lsp::{
    COMMAND_ANALYZE_WORKSPACE, COMMAND_COMPILATION_GUIDANCE, COMMAND_EXPLAIN_DIAGNOSTIC,
    COMMAND_RECOMMEND_BACKEND, COMMAND_SHOW_BACKEND_INFO, COMMAND_SUGGEST_TACTICS, COMMAND_VERIFY,
    SUPPORTED_COMMANDS,
};
use tower_lsp::lsp_types::*;

// =============================================================================
// Document and DocumentStore Tests
// =============================================================================

mod document_tests {
    use super::*;

    #[test]
    fn test_document_creation_with_valid_usl() {
        let uri = Url::parse("file:///test.usl").unwrap();
        let doc = Document::new(
            uri.clone(),
            1,
            "type Value = { id: Int }\ntheorem value_valid { forall v: Value . true }".to_string(),
        );

        assert_eq!(doc.uri, uri);
        assert_eq!(doc.version, 1);
        assert!(doc.spec.is_some());
        assert!(doc.typed_spec.is_some());
        assert!(doc.parse_error.is_none());
        assert!(doc.type_errors.is_empty());
    }

    #[test]
    fn test_document_creation_with_parse_error() {
        let uri = Url::parse("file:///test.usl").unwrap();
        let doc = Document::new(uri, 1, "theorem { invalid syntax }".to_string());

        assert!(doc.spec.is_none());
        assert!(doc.parse_error.is_some());
    }

    #[test]
    fn test_document_position_conversion_roundtrip() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "line one\nline two\nline three".to_string(),
        );

        // Test position to offset and back
        let offset = doc.position_to_offset(1, 5);
        let (line, character) = doc.offset_to_position(offset);
        assert_eq!(line, 1);
        assert_eq!(character, 5);
    }

    #[test]
    fn test_document_word_at_position() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem my_theorem { forall x: Bool . x }".to_string(),
        );

        assert_eq!(doc.word_at_position(0, 0), Some("theorem"));
        assert_eq!(doc.word_at_position(0, 8), Some("my_theorem"));
        assert_eq!(doc.word_at_position(0, 21), Some("forall"));
    }

    #[test]
    fn test_document_find_identifier_range() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Value = { id: Int }\ntheorem check { forall v: Value . true }".to_string(),
        );

        let range = doc.find_identifier_range("Value");
        assert!(range.is_some());

        let range = range.unwrap();
        assert_eq!(range.start.line, 0);
        assert_eq!(range.start.character, 5);
    }

    #[test]
    fn test_document_find_all_references() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Value = { id: Int }\ntheorem check { forall v: Value . Value.id > 0 }"
                .to_string(),
        );

        let refs = doc.find_all_references("Value");
        assert_eq!(refs.len(), 3); // Definition + 2 usages
    }

    #[test]
    fn test_document_line_text() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "line 0\nline 1\nline 2".to_string(),
        );

        assert_eq!(doc.line_text(0), Some("line 0"));
        assert_eq!(doc.line_text(1), Some("line 1"));
        assert_eq!(doc.line_text(2), Some("line 2"));
        assert_eq!(doc.line_text(3), None);
    }

    #[test]
    fn test_document_store_lifecycle() {
        let store = DocumentStore::new();
        let uri = Url::parse("file:///test.usl").unwrap();

        // Open document
        store.open(uri.clone(), 1, "theorem foo { true }".to_string());

        let version = store.with_document(&uri, |doc| doc.version);
        assert_eq!(version, Some(1));

        // Update document
        store.update(&uri, 2, "theorem bar { false }".to_string());

        let version = store.with_document(&uri, |doc| doc.version);
        assert_eq!(version, Some(2));

        // Close document
        store.close(&uri);

        let result = store.with_document(&uri, |doc| doc.version);
        assert!(result.is_none());
    }

    #[test]
    fn test_document_store_multiple_documents() {
        let store = DocumentStore::new();
        let uri1 = Url::parse("file:///one.usl").unwrap();
        let uri2 = Url::parse("file:///two.usl").unwrap();
        let uri3 = Url::parse("file:///three.usl").unwrap();

        store.open(uri1.clone(), 1, "theorem one { true }".to_string());
        store.open(uri2.clone(), 1, "theorem two { true }".to_string());
        store.open(uri3.clone(), 1, "theorem three { true }".to_string());

        let uris = store.all_uris();
        assert_eq!(uris.len(), 3);
        assert!(uris.contains(&uri1));
        assert!(uris.contains(&uri2));
        assert!(uris.contains(&uri3));
    }

    #[test]
    fn test_document_with_complex_types() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type Graph = { nodes: Set<Int>, edges: Relation<Int, Int> }
type Cache = { data: Map<String, Int> }
theorem graph_valid { forall g: Graph . true }"#
                .to_string(),
        );

        assert!(doc.spec.is_some());
        let spec = doc.spec.unwrap();
        assert_eq!(spec.types.len(), 2);
        assert_eq!(spec.properties.len(), 1);
    }

    #[test]
    fn test_document_with_all_property_types() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"theorem thm { true }
invariant inv { true }
temporal temp { always(true) }
contract Foo::bar(self: Int) -> Int {
    requires { true }
    ensures { true }
}"#
            .to_string(),
        );

        assert!(doc.spec.is_some());
        let spec = doc.spec.unwrap();
        assert_eq!(spec.properties.len(), 4);
    }
}

// =============================================================================
// Server Capabilities Tests
// =============================================================================

mod capabilities_tests {
    use super::*;

    #[test]
    fn test_server_capabilities_text_sync() {
        let caps = server_capabilities();

        match &caps.text_document_sync {
            Some(TextDocumentSyncCapability::Options(opts)) => {
                assert_eq!(opts.open_close, Some(true));
                assert_eq!(opts.change, Some(TextDocumentSyncKind::FULL));
            }
            _ => panic!("Expected text document sync options"),
        }
    }

    #[test]
    fn test_server_capabilities_hover() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.hover_provider,
            Some(HoverProviderCapability::Simple(true))
        ));
    }

    #[test]
    fn test_server_capabilities_completion() {
        let caps = server_capabilities();
        assert!(caps.completion_provider.is_some());

        let completion = caps.completion_provider.unwrap();
        let triggers = completion.trigger_characters.unwrap();
        assert!(triggers.contains(&".".to_string()));
        assert!(triggers.contains(&":".to_string()));
    }

    #[test]
    fn test_server_capabilities_definition() {
        let caps = server_capabilities();
        assert!(matches!(caps.definition_provider, Some(OneOf::Left(true))));
    }

    #[test]
    fn test_server_capabilities_document_symbol() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.document_symbol_provider,
            Some(OneOf::Left(true))
        ));
    }

    #[test]
    fn test_server_capabilities_references() {
        let caps = server_capabilities();
        assert!(matches!(caps.references_provider, Some(OneOf::Left(true))));
    }

    #[test]
    fn test_server_capabilities_rename() {
        let caps = server_capabilities();

        match &caps.rename_provider {
            Some(OneOf::Right(opts)) => {
                assert_eq!(opts.prepare_provider, Some(true));
            }
            _ => panic!("Expected rename options with prepare_provider"),
        }
    }

    #[test]
    fn test_server_capabilities_document_highlight() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.document_highlight_provider,
            Some(OneOf::Left(true))
        ));
    }

    #[test]
    fn test_server_capabilities_workspace_symbol() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.workspace_symbol_provider,
            Some(OneOf::Left(true))
        ));
    }

    #[test]
    fn test_server_capabilities_code_action() {
        let caps = server_capabilities();

        match &caps.code_action_provider {
            Some(CodeActionProviderCapability::Options(opts)) => {
                let kinds = opts.code_action_kinds.as_ref().unwrap();
                assert!(kinds.contains(&CodeActionKind::QUICKFIX));
                assert!(kinds.contains(&CodeActionKind::REFACTOR_EXTRACT));
            }
            _ => panic!("Expected code action options"),
        }
    }

    #[test]
    fn test_server_capabilities_code_lens() {
        let caps = server_capabilities();
        assert!(caps.code_lens_provider.is_some());
    }

    #[test]
    fn test_server_capabilities_formatting() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.document_formatting_provider,
            Some(OneOf::Left(true))
        ));
        assert!(matches!(
            caps.document_range_formatting_provider,
            Some(OneOf::Left(true))
        ));
    }

    #[test]
    fn test_server_capabilities_folding_range() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.folding_range_provider,
            Some(FoldingRangeProviderCapability::Simple(true))
        ));
    }

    #[test]
    fn test_server_capabilities_execute_command() {
        let caps = server_capabilities();
        assert!(caps.execute_command_provider.is_some());

        let execute = caps.execute_command_provider.unwrap();
        assert!(execute.commands.contains(&COMMAND_VERIFY.to_string()));
        assert!(execute
            .commands
            .contains(&COMMAND_SHOW_BACKEND_INFO.to_string()));
    }

    #[test]
    fn test_server_capabilities_semantic_tokens() {
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
    fn test_server_capabilities_inlay_hints() {
        let caps = server_capabilities();
        assert!(caps.inlay_hint_provider.is_some());
    }

    #[test]
    fn test_server_capabilities_signature_help() {
        let caps = server_capabilities();
        assert!(caps.signature_help_provider.is_some());

        let sig_help = caps.signature_help_provider.unwrap();
        let triggers = sig_help.trigger_characters.unwrap();
        assert!(triggers.contains(&"(".to_string()));
        assert!(triggers.contains(&",".to_string()));
    }

    #[test]
    fn test_server_capabilities_selection_range() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.selection_range_provider,
            Some(SelectionRangeProviderCapability::Simple(true))
        ));
    }

    #[test]
    fn test_server_capabilities_call_hierarchy() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.call_hierarchy_provider,
            Some(CallHierarchyServerCapability::Simple(true))
        ));
    }

    #[test]
    fn test_server_capabilities_linked_editing_range() {
        let caps = server_capabilities();
        assert!(matches!(
            caps.linked_editing_range_provider,
            Some(LinkedEditingRangeServerCapabilities::Simple(true))
        ));
    }

    #[test]
    fn test_server_capabilities_moniker() {
        let caps = server_capabilities();
        assert!(caps.moniker_provider.is_some());
    }
}

// =============================================================================
// Command Constants Tests
// =============================================================================

mod command_tests {
    use super::*;

    #[test]
    fn test_command_verify_constant() {
        assert_eq!(COMMAND_VERIFY, "dashprove.verify");
    }

    #[test]
    fn test_command_show_backend_info_constant() {
        assert_eq!(COMMAND_SHOW_BACKEND_INFO, "dashprove.showBackendInfo");
    }

    #[test]
    fn test_command_explain_diagnostic_constant() {
        assert_eq!(COMMAND_EXPLAIN_DIAGNOSTIC, "dashprove.explainDiagnostic");
    }

    #[test]
    fn test_supported_commands_count() {
        assert_eq!(SUPPORTED_COMMANDS.len(), 7);
    }

    #[test]
    fn test_supported_commands_contains_verify() {
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_VERIFY));
    }

    #[test]
    fn test_supported_commands_contains_backend_info() {
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_SHOW_BACKEND_INFO));
    }

    #[test]
    fn test_supported_commands_contains_explain_diagnostic() {
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_EXPLAIN_DIAGNOSTIC));
    }

    #[test]
    fn test_supported_commands_contains_suggest_tactics() {
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_SUGGEST_TACTICS));
    }

    #[test]
    fn test_command_recommend_backend_constant() {
        assert_eq!(COMMAND_RECOMMEND_BACKEND, "dashprove.recommendBackend");
    }

    #[test]
    fn test_supported_commands_contains_recommend_backend() {
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_RECOMMEND_BACKEND));
    }

    #[test]
    fn test_command_compilation_guidance_constant() {
        assert_eq!(
            COMMAND_COMPILATION_GUIDANCE,
            "dashprove.compilationGuidance"
        );
    }

    #[test]
    fn test_supported_commands_contains_compilation_guidance() {
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_COMPILATION_GUIDANCE));
    }

    #[test]
    fn test_command_analyze_workspace_constant() {
        assert_eq!(COMMAND_ANALYZE_WORKSPACE, "dashprove.analyzeWorkspace");
    }

    #[test]
    fn test_supported_commands_contains_analyze_workspace() {
        assert!(SUPPORTED_COMMANDS.contains(&COMMAND_ANALYZE_WORKSPACE));
    }
}

// =============================================================================
// Document Analysis Integration Tests
// =============================================================================

mod analysis_tests {
    use super::*;

    #[test]
    fn test_parse_theorem_property() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem excluded_middle { forall x: Bool . x or not x }".to_string(),
        );

        assert!(doc.spec.is_some());
        let spec = doc.spec.unwrap();
        assert_eq!(spec.properties.len(), 1);
    }

    #[test]
    fn test_parse_temporal_property() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "temporal liveness { eventually(true) }".to_string(),
        );

        assert!(doc.spec.is_some());
        let spec = doc.spec.unwrap();
        assert_eq!(spec.properties.len(), 1);
    }

    #[test]
    fn test_parse_invariant_property() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type State = { value: Int }\ninvariant state_positive { forall s: State . s.value >= 0 }"
                .to_string(),
        );

        assert!(doc.spec.is_some());
        let spec = doc.spec.unwrap();
        assert_eq!(spec.types.len(), 1);
        assert_eq!(spec.properties.len(), 1);
    }

    #[test]
    fn test_parse_contract_property() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type Counter = { value: Int }
contract Counter::increment(self: Counter) -> Counter {
    requires { self.value >= 0 }
    ensures { result.value == self.value + 1 }
}"#
            .to_string(),
        );

        assert!(doc.spec.is_some());
        let spec = doc.spec.unwrap();
        assert_eq!(spec.types.len(), 1);
        assert_eq!(spec.properties.len(), 1);
    }

    #[test]
    fn test_parse_refinement_property() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type AbstractSet = { elements: Set<Int> }
type ConcreteList = { items: List<Int> }
refinement list_refines_set refines AbstractSet {
    abstraction { forall l: ConcreteList . true }
    simulation { forall a: AbstractSet, c: ConcreteList . true }
}"#
            .to_string(),
        );

        assert!(doc.spec.is_some());
        let spec = doc.spec.unwrap();
        assert_eq!(spec.types.len(), 2);
        assert_eq!(spec.properties.len(), 1);
    }

    #[test]
    fn test_parse_probabilistic_property() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "probabilistic high_prob { probability(true) >= 0.9 }".to_string(),
        );

        assert!(doc.spec.is_some());
    }

    #[test]
    fn test_parse_security_property() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "security no_leak { forall x: Int . true }".to_string(),
        );

        assert!(doc.spec.is_some());
    }

    #[test]
    fn test_multiple_types_and_properties() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type User = { id: Int, name: String }
type Account = { owner: User, balance: Int }
type Transaction = { from: Account, to: Account, amount: Int }

theorem positive_balance { forall a: Account . a.balance >= 0 }
invariant valid_transfer { forall t: Transaction . t.amount > 0 }
temporal eventual_consistency { eventually(true) }"#
                .to_string(),
        );

        assert!(doc.spec.is_some());
        let spec = doc.spec.unwrap();
        assert_eq!(spec.types.len(), 3);
        assert_eq!(spec.properties.len(), 3);
    }
}

// =============================================================================
// Example Files Integration Tests
// =============================================================================

mod example_files_tests {
    use super::*;

    fn load_example(name: &str) -> Document {
        let content = match name {
            "refinement" => include_str!("../../../examples/usl/refinement.usl"),
            "contracts" => include_str!("../../../examples/usl/contracts.usl"),
            "temporal" => include_str!("../../../examples/usl/temporal.usl"),
            "security" => include_str!("../../../examples/usl/security.usl"),
            _ => panic!("Unknown example: {}", name),
        };

        Document::new(
            Url::parse(&format!("file:///{}.usl", name)).unwrap(),
            1,
            content.to_string(),
        )
    }

    #[test]
    fn test_parse_refinement_example() {
        let doc = load_example("refinement");

        assert!(doc.spec.is_some(), "refinement.usl should parse");
        assert!(
            doc.parse_error.is_none(),
            "refinement.usl should have no parse errors"
        );

        let spec = doc.spec.unwrap();
        assert!(!spec.types.is_empty(), "Should have types defined");
        assert!(
            !spec.properties.is_empty(),
            "Should have properties defined"
        );
    }

    #[test]
    fn test_parse_contracts_example() {
        let doc = load_example("contracts");

        assert!(doc.spec.is_some(), "contracts.usl should parse");
        assert!(
            doc.parse_error.is_none(),
            "contracts.usl should have no parse errors"
        );

        let spec = doc.spec.unwrap();
        assert!(
            !spec.properties.is_empty(),
            "Should have contract properties"
        );
    }

    #[test]
    fn test_parse_temporal_example() {
        let doc = load_example("temporal");

        assert!(doc.spec.is_some(), "temporal.usl should parse");
        assert!(
            doc.parse_error.is_none(),
            "temporal.usl should have no parse errors"
        );

        let spec = doc.spec.unwrap();
        assert!(
            !spec.properties.is_empty(),
            "Should have temporal properties"
        );
    }

    #[test]
    fn test_parse_security_example() {
        let doc = load_example("security");

        assert!(doc.spec.is_some(), "security.usl should parse");
        assert!(
            doc.parse_error.is_none(),
            "security.usl should have no parse errors"
        );
    }

    #[test]
    fn test_refinement_example_types() {
        let doc = load_example("refinement");
        let spec = doc.spec.unwrap();

        let type_names: Vec<_> = spec.types.iter().map(|t| t.name.as_str()).collect();
        assert!(
            type_names.contains(&"AbstractSet"),
            "Should have AbstractSet type"
        );
        assert!(
            type_names.contains(&"SortedListSet"),
            "Should have SortedListSet type"
        );
    }

    #[test]
    fn test_contracts_example_has_contracts() {
        let doc = load_example("contracts");
        let spec = doc.spec.unwrap();

        let has_contract = spec
            .properties
            .iter()
            .any(|p| matches!(p, dashprove_usl::Property::Contract(_)));
        assert!(has_contract, "Should have at least one contract property");
    }

    #[test]
    fn test_temporal_example_has_temporal() {
        let doc = load_example("temporal");
        let spec = doc.spec.unwrap();

        let has_temporal = spec
            .properties
            .iter()
            .any(|p| matches!(p, dashprove_usl::Property::Temporal(_)));
        assert!(has_temporal, "Should have at least one temporal property");
    }
}

// =============================================================================
// Position and Range Utilities Tests
// =============================================================================

mod position_tests {
    use super::*;

    #[test]
    fn test_utf8_position_handling() {
        // Test that UTF-8 characters are handled correctly
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "// Comment with unicode: \u{1F600}\ntheorem foo { true }".to_string(),
        );

        // The emoji takes multiple bytes but should be handled in UTF-16 for LSP
        let word = doc.word_at_position(1, 8);
        assert_eq!(word, Some("foo"));
    }

    #[test]
    fn test_multiline_position_tracking() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type A = {\n    x: Int,\n    y: Int\n}".to_string(),
        );

        // Position on each line
        assert_eq!(doc.word_at_position(0, 5), Some("A"));
        assert_eq!(doc.word_at_position(1, 4), Some("x"));
        assert_eq!(doc.word_at_position(2, 4), Some("y"));
    }

    #[test]
    fn test_position_at_document_end() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { true }".to_string(),
        );

        // Position beyond document should return document length
        let offset = doc.position_to_offset(100, 100);
        assert_eq!(offset, doc.text.len());
    }

    #[test]
    fn test_position_at_line_end() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "short\nlonger line".to_string(),
        );

        // Position beyond line end
        let offset = doc.position_to_offset(0, 100);
        // Should be at end of first line content (before newline)
        assert!(offset <= doc.text.len());
    }
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

mod concurrent_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_document_store_thread_safety() {
        let store = Arc::new(DocumentStore::new());
        let mut handles = vec![];

        // Spawn multiple threads to read/write
        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = thread::spawn(move || {
                let uri = Url::parse(&format!("file:///test{}.usl", i)).unwrap();
                store_clone.open(uri.clone(), 1, format!("theorem t{} {{ true }}", i));

                let version = store_clone.with_document(&uri, |doc| doc.version);
                assert_eq!(version, Some(1));

                store_clone.update(&uri, 2, format!("theorem t{}_updated {{ true }}", i));

                let version = store_clone.with_document(&uri, |doc| doc.version);
                assert_eq!(version, Some(2));
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    #[test]
    fn test_document_store_concurrent_reads() {
        let store = Arc::new(DocumentStore::new());
        let uri = Url::parse("file:///shared.usl").unwrap();
        store.open(
            uri.clone(),
            1,
            "theorem shared { forall x: Bool . x }".to_string(),
        );

        let mut handles = vec![];

        for _ in 0..10 {
            let store_clone = Arc::clone(&store);
            let uri_clone = uri.clone();
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let has_spec = store_clone.with_document(&uri_clone, |doc| doc.spec.is_some());
                    assert_eq!(has_spec, Some(true));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_document() {
        let doc = Document::new(Url::parse("file:///test.usl").unwrap(), 1, "".to_string());

        // USL grammar allows empty spec (0 items)
        assert!(doc.spec.is_some());
        let spec = doc.spec.as_ref().unwrap();
        assert!(spec.types.is_empty());
        assert!(spec.properties.is_empty());

        // But there are no words or identifiers
        assert!(doc.word_at_position(0, 0).is_none());
        assert!(doc.find_identifier_range("anything").is_none());
        assert!(doc.find_all_references("anything").is_empty());
    }

    #[test]
    fn test_whitespace_only_document() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "   \n\n\t\t\n   ".to_string(),
        );

        // USL grammar allows empty spec (whitespace is ignored)
        assert!(doc.spec.is_some());
        let spec = doc.spec.unwrap();
        assert!(spec.types.is_empty());
        assert!(spec.properties.is_empty());
    }

    #[test]
    fn test_comment_only_document() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "// This is a comment\n// Another comment".to_string(),
        );

        // USL grammar allows empty spec (comments are ignored)
        assert!(doc.spec.is_some());
        let spec = doc.spec.unwrap();
        assert!(spec.types.is_empty());
        assert!(spec.properties.is_empty());
    }

    #[test]
    fn test_deeply_nested_types() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Deep = { data: Map<String, List<Set<Int>>> }".to_string(),
        );

        assert!(doc.spec.is_some());
        let spec = doc.spec.unwrap();
        assert_eq!(spec.types.len(), 1);
    }

    #[test]
    fn test_very_long_identifier() {
        let long_name = "a".repeat(100);
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            format!("theorem {} {{ true }}", long_name),
        );

        assert!(doc.spec.is_some());
        let range = doc.find_identifier_range(&long_name);
        assert!(range.is_some());
    }

    #[test]
    fn test_special_characters_in_strings() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"theorem test { "hello\nworld\t\"escaped\"" == "test" }"#.to_string(),
        );

        // This tests the parser's handling of escape sequences
        // The spec may or may not parse depending on USL grammar
        // This test documents the behavior
        let _ = doc.spec;
    }

    #[test]
    fn test_unicode_identifiers() {
        // Test with ASCII-compatible identifiers (USL likely doesn't support Unicode identifiers)
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem test_123 { true }".to_string(),
        );

        assert!(doc.spec.is_some());
        let range = doc.find_identifier_range("test_123");
        assert!(range.is_some());
    }

    #[test]
    fn test_document_update_reanalyzes() {
        let store = DocumentStore::new();
        let uri = Url::parse("file:///test.usl").unwrap();

        // Open with invalid content
        store.open(uri.clone(), 1, "invalid { syntax".to_string());

        let has_error = store.with_document(&uri, |doc| doc.parse_error.is_some());
        assert_eq!(has_error, Some(true));

        // Update with valid content
        store.update(&uri, 2, "theorem valid { true }".to_string());

        let has_spec = store.with_document(&uri, |doc| doc.spec.is_some());
        assert_eq!(has_spec, Some(true));
    }
}

// =============================================================================
// UslLanguageServer Export Tests
// =============================================================================

mod server_export_tests {
    use super::*;

    #[test]
    fn test_usl_language_server_type_exported() {
        // This test verifies that UslLanguageServer is properly exported
        // We can't instantiate it without a Client, but we can verify the type exists
        fn _takes_server(_: &UslLanguageServer) {}
    }

    #[test]
    fn test_all_public_exports_available() {
        // Verify all public exports are available
        let _ = server_capabilities();
        let _ = DocumentStore::new();
        let _ = Document::new(Url::parse("file:///test.usl").unwrap(), 1, "".to_string());
        let _ = COMMAND_VERIFY;
        let _ = COMMAND_SHOW_BACKEND_INFO;
        let _ = SUPPORTED_COMMANDS;
    }
}
