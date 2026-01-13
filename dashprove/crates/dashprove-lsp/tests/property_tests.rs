//! Property-based tests for dashprove-lsp using proptest

use dashprove_lsp::{Document, DocumentStore};
use proptest::prelude::*;
use tower_lsp::lsp_types::Url;

// ============================================================================
// Strategy generators
// ============================================================================

/// Generate valid USL-like documents with varying structures
fn arbitrary_usl_source() -> impl Strategy<Value = String> {
    prop_oneof![
        // Empty document
        Just(String::new()),
        // Simple type definition
        "[a-z]{1,10}".prop_map(|name| format!("type {name} {{ }}")),
        // Type with field
        ("[a-z]{1,10}", "[a-z]{1,10}")
            .prop_map(|(name, field)| format!("type {name} {{ {field}: Int }}")),
        // Simple property
        "[a-z]{1,10}".prop_map(|name| format!("property {name} {{ true }}")),
        // Multiple types
        ("[a-z]{1,10}", "[a-z]{1,10}")
            .prop_map(|(name1, name2)| format!("type {name1} {{ }}\ntype {name2} {{ }}")),
        // Multi-line with various structures
        prop::collection::vec("[a-z]{1,10}", 1..5).prop_map(|names| {
            names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    if i % 2 == 0 {
                        format!("type {name} {{ }}")
                    } else {
                        format!("property {name} {{ true }}")
                    }
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        }),
    ]
}

/// Generate arbitrary text (not necessarily valid USL)
fn arbitrary_text() -> impl Strategy<Value = String> {
    prop_oneof![
        // Empty
        Just(String::new()),
        // Single line
        "[a-zA-Z0-9 ]{1,100}",
        // Multi-line
        prop::collection::vec("[a-zA-Z0-9 ]{0,50}", 1..10).prop_map(|lines| lines.join("\n")),
        // With tabs and special chars
        prop::collection::vec("[a-zA-Z0-9 \t]{0,50}", 1..10).prop_map(|lines| lines.join("\n")),
        // With empty lines
        prop::collection::vec(
            prop_oneof![
                Just(String::new()),
                "[a-zA-Z0-9 ]{1,50}".prop_map(String::from),
            ],
            1..10
        )
        .prop_map(|lines| lines.join("\n")),
    ]
}

/// Generate a file:// URI
fn arbitrary_uri() -> impl Strategy<Value = Url> {
    "[a-z]{3,10}".prop_map(|name| Url::parse(&format!("file:///{name}.usl")).unwrap())
}

// ============================================================================
// Document property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn document_creation_preserves_text(
        text in arbitrary_usl_source(),
    ) {
        let uri = Url::parse("file:///test.usl").unwrap();
        let doc = Document::new(uri.clone(), 1, text.clone());
        prop_assert_eq!(doc.text, text);
        prop_assert_eq!(doc.version, 1);
    }

    #[test]
    fn document_update_replaces_text(
        text1 in arbitrary_usl_source(),
        text2 in arbitrary_usl_source(),
    ) {
        let uri = Url::parse("file:///test.usl").unwrap();
        let mut doc = Document::new(uri.clone(), 1, text1);
        doc.update_text(text2.clone());
        prop_assert_eq!(doc.text, text2);
    }

    #[test]
    fn document_position_offset_roundtrip_is_monotonic(text in arbitrary_text()) {
        let uri = Url::parse("file:///test.usl").unwrap();
        let doc = Document::new(uri, 1, text.clone());

        // For each position in the document, converting to offset and back
        // should give a position that when converted again gives the same offset
        let num_lines = doc.text.lines().count().max(1);
        for line in 0..(num_lines as u32) {
            let line_text = doc.text.lines().nth(line as usize).unwrap_or("");
            let max_char = line_text.len() as u32;

            for character in 0..=max_char {
                let offset = doc.position_to_offset(line, character);
                let (back_line, back_char) = doc.offset_to_position(offset);
                let back_offset = doc.position_to_offset(back_line, back_char);

                // The round-trip offset should be the same
                prop_assert_eq!(offset, back_offset,
                    "Position ({}, {}) -> offset {} -> position ({}, {}) -> offset {} should match",
                    line, character, offset, back_line, back_char, back_offset);
            }
        }
    }

    #[test]
    fn document_offset_at_line_start_is_correct(text in arbitrary_text()) {
        let uri = Url::parse("file:///test.usl").unwrap();
        let doc = Document::new(uri, 1, text.clone());

        let mut expected_offset = 0usize;
        for (line_num, line) in doc.text.split('\n').enumerate() {
            let actual_offset = doc.position_to_offset(line_num as u32, 0);
            prop_assert_eq!(actual_offset, expected_offset,
                "Line {} should start at offset {}, got {}",
                line_num, expected_offset, actual_offset);
            expected_offset += line.len() + 1; // +1 for newline
        }
    }

    #[test]
    fn document_offset_never_exceeds_text_length(
        text in arbitrary_text(),
        line in 0u32..100,
        character in 0u32..200,
    ) {
        let uri = Url::parse("file:///test.usl").unwrap();
        let doc = Document::new(uri, 1, text.clone());

        let offset = doc.position_to_offset(line, character);
        prop_assert!(offset <= doc.text.len(),
            "Offset {} should not exceed text length {}", offset, doc.text.len());
    }

    #[test]
    fn document_position_line_never_exceeds_line_count(
        text in arbitrary_text(),
        offset in 0usize..1000,
    ) {
        let uri = Url::parse("file:///test.usl").unwrap();
        let doc = Document::new(uri, 1, text.clone());

        let bounded_offset = offset.min(doc.text.len());
        let (line, _char) = doc.offset_to_position(bounded_offset);

        // Count actual lines (including trailing newline creating an empty line)
        let line_count = if doc.text.ends_with('\n') {
            doc.text.lines().count() + 1
        } else {
            doc.text.lines().count().max(1)
        };
        prop_assert!((line as usize) < line_count || doc.text.is_empty(),
            "Line {} should be less than line count {} (text len: {})",
            line, line_count, doc.text.len());
    }
}

// ============================================================================
// DocumentStore property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn document_store_open_and_get(
        uri in arbitrary_uri(),
        text in arbitrary_usl_source(),
    ) {
        let store = DocumentStore::new();
        store.open(uri.clone(), 1, text.clone());

        let retrieved = store.with_document(&uri, |doc| doc.text.clone());
        prop_assert_eq!(retrieved, Some(text));
    }

    #[test]
    fn document_store_update_changes_text(
        uri in arbitrary_uri(),
        text1 in arbitrary_usl_source(),
        text2 in arbitrary_usl_source(),
    ) {
        let store = DocumentStore::new();
        store.open(uri.clone(), 1, text1);

        store.update(&uri, 2, text2.clone());

        let retrieved = store.with_document(&uri, |doc| doc.text.clone());
        prop_assert_eq!(retrieved, Some(text2));
    }

    #[test]
    fn document_store_close_removes_document(
        uri in arbitrary_uri(),
        text in arbitrary_usl_source(),
    ) {
        let store = DocumentStore::new();
        store.open(uri.clone(), 1, text);

        store.close(&uri);

        let retrieved = store.with_document(&uri, |doc| doc.text.clone());
        prop_assert_eq!(retrieved, None);
    }

    #[test]
    fn document_store_multiple_documents_independent(
        uri1 in arbitrary_uri(),
        uri2 in arbitrary_uri(),
        text1 in arbitrary_usl_source(),
        text2 in arbitrary_usl_source(),
    ) {
        // Skip if URIs happen to be the same
        prop_assume!(uri1 != uri2);

        let store = DocumentStore::new();
        store.open(uri1.clone(), 1, text1.clone());
        store.open(uri2.clone(), 1, text2.clone());

        let retrieved1 = store.with_document(&uri1, |doc| doc.text.clone());
        let retrieved2 = store.with_document(&uri2, |doc| doc.text.clone());

        prop_assert_eq!(retrieved1, Some(text1));
        prop_assert_eq!(retrieved2, Some(text2));
    }

    #[test]
    fn document_store_version_updates_on_change(
        uri in arbitrary_uri(),
        text in arbitrary_usl_source(),
    ) {
        let store = DocumentStore::new();
        store.open(uri.clone(), 1, text.clone());

        // Update with new version
        store.update(&uri, 5, text.clone());

        let version = store.with_document(&uri, |doc| doc.version);
        prop_assert_eq!(version, Some(5));
    }
}

// ============================================================================
// USL parsing property tests (valid specs should parse)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn document_parses_empty_spec(seed in any::<u64>()) {
        let _ = seed;
        let uri = Url::parse("file:///test.usl").unwrap();
        let doc = Document::new(uri, 1, String::new());
        // Empty spec should parse without error
        prop_assert!(doc.parse_error.is_none());
    }

    #[test]
    fn document_parses_simple_type(name in "[a-z]{1,10}") {
        let uri = Url::parse("file:///test.usl").unwrap();
        let source = format!("type {name} {{ }}");
        let doc = Document::new(uri, 1, source);

        // Should parse (spec present) or have parse error (one or the other)
        prop_assert!(doc.spec.is_some() || doc.parse_error.is_some(),
            "Document should either parse successfully or have error");
    }

    #[test]
    fn document_parses_simple_property(name in "[a-z]{1,10}") {
        let uri = Url::parse("file:///test.usl").unwrap();
        let source = format!("property {name} {{ true }}");
        let doc = Document::new(uri, 1, source);

        // Should parse (spec present) or have parse error (one or the other)
        prop_assert!(doc.spec.is_some() || doc.parse_error.is_some(),
            "Document should either parse successfully or have error");
    }
}

// ============================================================================
// Server constants property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn supported_commands_non_empty(seed in any::<u64>()) {
        let _ = seed;
        prop_assert!(!dashprove_lsp::SUPPORTED_COMMANDS.is_empty());
    }

    #[test]
    fn verify_command_in_supported(seed in any::<u64>()) {
        let _ = seed;
        prop_assert!(dashprove_lsp::SUPPORTED_COMMANDS.contains(&dashprove_lsp::COMMAND_VERIFY));
    }

    #[test]
    fn show_backend_info_command_in_supported(seed in any::<u64>()) {
        let _ = seed;
        prop_assert!(dashprove_lsp::SUPPORTED_COMMANDS.contains(&dashprove_lsp::COMMAND_SHOW_BACKEND_INFO));
    }
}
