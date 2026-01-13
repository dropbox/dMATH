//! Tests for code lens generation.

use super::*;
use crate::document::Document;
use tower_lsp::lsp_types::Url;

#[test]
fn test_generate_code_lenses_for_theorem() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type Value = { x: Int }\ntheorem value_positive { forall v: Value . v.x > 0 }".to_string(),
    );

    let lenses = generate_code_lenses(&doc);

    // Should have 4 lenses: 1 type backend rec + 3 theorem (verify + backend info + tactic suggestions)
    assert_eq!(lenses.len(), 4);

    // First lens is for the type definition
    let type_lens = &lenses[0];
    assert!(type_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("Recommend"));
    assert_eq!(
        type_lens.command.as_ref().unwrap().command,
        "dashprove.recommendBackend"
    );

    let verify_lens = &lenses[1];
    assert!(verify_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("Verify"));
    assert_eq!(
        verify_lens.command.as_ref().unwrap().command,
        "dashprove.verify"
    );

    let info_lens = &lenses[2];
    assert!(info_lens.command.as_ref().unwrap().title.contains("LEAN"));

    let tactic_lens = &lenses[3];
    assert!(tactic_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("Suggest tactics"));
    assert_eq!(
        tactic_lens.command.as_ref().unwrap().command,
        "dashprove.suggestTactics"
    );
}

#[test]
fn test_generate_code_lenses_for_contract() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "contract Stack::push(self: Stack, item: Int) -> Result<Unit> {\n  requires { true }\n  ensures { true }\n}"
            .to_string(),
    );

    let lenses = generate_code_lenses(&doc);

    // Now 3 lenses: verify + backend info + verification strategies
    assert_eq!(lenses.len(), 3);

    let verify_lens = &lenses[0];
    assert!(verify_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("contract"));

    let info_lens = &lenses[1];
    assert!(info_lens.command.as_ref().unwrap().title.contains("Kani"));

    let strategy_lens = &lenses[2];
    assert!(strategy_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("verification strategies"));
    assert_eq!(
        strategy_lens.command.as_ref().unwrap().command,
        "dashprove.suggestTactics"
    );
}

#[test]
fn test_generate_code_lenses_for_temporal() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "temporal eventual_response {\n  always(request implies eventually(response))\n}"
            .to_string(),
    );

    let lenses = generate_code_lenses(&doc);

    assert_eq!(lenses.len(), 2);

    let info_lens = &lenses[1];
    assert!(info_lens.command.as_ref().unwrap().title.contains("TLA"));
}

#[test]
fn test_generate_code_lenses_for_invariant() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type Counter = { value: Int }\ninvariant counter_non_negative { forall c: Counter . c.value >= 0 }"
            .to_string(),
    );

    let lenses = generate_code_lenses(&doc);

    // 4 lenses: 1 type backend rec + 3 invariant (verify + backend info + tactic suggestions)
    assert_eq!(lenses.len(), 4);

    // First lens is for the type definition
    let type_lens = &lenses[0];
    assert!(type_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("Recommend"));

    let verify_lens = &lenses[1];
    assert!(verify_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("invariant"));

    let tactic_lens = &lenses[3];
    assert!(tactic_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("Suggest tactics"));
    assert_eq!(
        tactic_lens.command.as_ref().unwrap().command,
        "dashprove.suggestTactics"
    );
}

#[test]
fn test_generate_code_lenses_for_refinement() {
    let doc = Document::new(
        Url::parse("file:///refinement.usl").unwrap(),
        1,
        include_str!("../../../../examples/usl/refinement.usl").to_string(),
    );

    let lenses = generate_code_lenses(&doc);

    // Refinement file has 4 types + 2 refinements (each with 2 lenses)
    // 4 type lenses + 2 * 2 refinement lenses = 8 lenses
    assert_eq!(lenses.len(), 8);

    // Check that refinement lenses exist
    let verify_titles: Vec<_> = lenses
        .iter()
        .filter_map(|l| l.command.as_ref())
        .map(|c| c.title.as_str())
        .collect();

    assert!(verify_titles.iter().any(|t| t.contains("refinement")));
    assert!(verify_titles.iter().any(|t| t.contains("Recommend")));
}

#[test]
fn test_no_lenses_for_invalid_document() {
    let doc = Document::new(
        Url::parse("file:///invalid.usl").unwrap(),
        1,
        "this is not valid usl syntax { }".to_string(),
    );

    let lenses = generate_code_lenses(&doc);
    assert!(lenses.is_empty());
}

#[test]
fn test_code_lens_positions() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "// Comment\ntype T = { x: Int }\n\ntheorem my_theorem {\n    true\n}".to_string(),
    );

    let lenses = generate_code_lenses(&doc);

    // 1 type lens + 3 theorem lenses (verify + backend + tactic suggestions)
    assert_eq!(lenses.len(), 4);
    // Type lens on line 1 (0-indexed)
    assert_eq!(lenses[0].range.start.line, 1);
    // Theorem lenses on line 3 (0-indexed)
    assert_eq!(lenses[1].range.start.line, 3);
    assert_eq!(lenses[2].range.start.line, 3);
    assert_eq!(lenses[3].range.start.line, 3);
}

#[test]
fn test_multiple_properties() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t1 { true }\ntheorem t2 { true }\ninvariant i1 { true }".to_string(),
    );

    let lenses = generate_code_lenses(&doc);

    // 2 theorems * 3 lenses + 1 invariant * 3 lenses = 9 lenses
    assert_eq!(lenses.len(), 9);

    // Verify different lines
    let lines: Vec<_> = lenses.iter().map(|l| l.range.start.line).collect();
    assert!(lines.contains(&0)); // t1
    assert!(lines.contains(&1)); // t2
    assert!(lines.contains(&2)); // i1
}

#[test]
fn test_code_lens_command_arguments() {
    let doc = Document::new(
        Url::parse("file:///myfile.usl").unwrap(),
        1,
        "theorem check_it { true }".to_string(),
    );

    let lenses = generate_code_lenses(&doc);
    let verify_lens = &lenses[0];
    let cmd = verify_lens.command.as_ref().unwrap();

    let args = cmd.arguments.as_ref().unwrap();
    assert_eq!(args.len(), 2);
    assert_eq!(
        args[0],
        serde_json::Value::String("file:///myfile.usl".to_string())
    );
    assert_eq!(args[1], serde_json::Value::String("check_it".to_string()));
}

#[test]
fn test_probabilistic_code_lens() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "probabilistic error_bound {\n  probability(failure) <= 0.001\n}".to_string(),
    );

    let lenses = generate_code_lenses(&doc);
    assert_eq!(lenses.len(), 2);

    let info_lens = &lenses[1];
    assert!(info_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("Probabilistic"));
}

#[test]
fn test_security_code_lens() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "security no_leak {\n  forall x: Int . true\n}".to_string(),
    );

    let lenses = generate_code_lenses(&doc);
    assert_eq!(lenses.len(), 2);

    let info_lens = &lenses[1];
    assert!(info_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("Security"));
}

#[test]
fn test_type_backend_recommendation_lens() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type Person = { name: String, age: Int }".to_string(),
    );

    let lenses = generate_code_lenses(&doc);

    // Should have 1 lens for the type (backend recommendation)
    assert_eq!(lenses.len(), 1);

    let type_lens = &lenses[0];
    let cmd = type_lens.command.as_ref().unwrap();

    assert!(cmd.title.contains("Recommend"));
    assert_eq!(cmd.command, "dashprove.recommendBackend");

    // Check arguments
    let args = cmd.arguments.as_ref().unwrap();
    assert_eq!(args.len(), 2);
    // First arg is the type description
    let type_desc = args[0].as_str().unwrap();
    assert!(type_desc.contains("Person"));
    assert!(type_desc.contains("name"));
    assert!(type_desc.contains("age"));
    // Second arg is the document URI
    assert_eq!(args[1].as_str().unwrap(), "file:///test.usl");
}

#[test]
fn test_format_type_def_inline() {
    use super::lens_generation::generate_code_lenses;

    // Test indirectly through the lens generation
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type TestType = { x: Int, data: List<String> }".to_string(),
    );

    let lenses = generate_code_lenses(&doc);
    let type_lens = &lenses[0];
    let args = type_lens
        .command
        .as_ref()
        .unwrap()
        .arguments
        .as_ref()
        .unwrap();
    let formatted = args[0].as_str().unwrap();
    assert_eq!(formatted, "type TestType = { x: Int, data: List<String> }");
}

// ==================== Workspace Stats Tests ====================

#[test]
fn test_workspace_stats_empty() {
    let stats = WorkspaceStats::new();
    assert_eq!(stats.file_count, 0);
    assert_eq!(stats.type_count, 0);
    assert_eq!(stats.property_count(), 0);
    assert_eq!(stats.summary(), "No USL definitions");
    assert_eq!(stats.detailed_breakdown(), "No properties");
    assert!(stats.error_summary().is_none());
}

#[test]
fn test_workspace_stats_single_document() {
    let mut stats = WorkspaceStats::new();
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type Value = { x: Int }\ntheorem test { true }".to_string(),
    );

    stats.add_document(&doc);

    assert_eq!(stats.file_count, 1);
    assert_eq!(stats.type_count, 1);
    assert_eq!(stats.property_count(), 1);
    assert_eq!(stats.theorem_count(), 1);
    assert_eq!(stats.summary(), "1 file, 1 type, 1 property");
    assert_eq!(stats.detailed_breakdown(), "1 thm");
}

#[test]
fn test_workspace_stats_multiple_documents() {
    let mut stats = WorkspaceStats::new();

    let doc1 = Document::new(
        Url::parse("file:///test1.usl").unwrap(),
        1,
        "type A = { x: Int }\ntheorem t1 { true }\ntheorem t2 { true }".to_string(),
    );

    let doc2 = Document::new(
        Url::parse("file:///test2.usl").unwrap(),
        1,
        "type B = { y: Int }\ncontract B::foo() -> Result<Unit> { requires { true } ensures { true } }".to_string(),
    );

    stats.add_document(&doc1);
    stats.add_document(&doc2);

    assert_eq!(stats.file_count, 2);
    assert_eq!(stats.type_count, 2);
    assert_eq!(stats.property_count(), 3);
    assert_eq!(stats.theorem_count(), 2);
    assert_eq!(stats.contract_count(), 1);
    assert_eq!(stats.summary(), "2 files, 2 types, 3 properties");
    assert_eq!(stats.detailed_breakdown(), "2 thm | 1 ctr");
}

#[test]
fn test_workspace_stats_all_property_types() {
    let mut stats = WorkspaceStats::new();

    // Document with theorems
    let doc1 = Document::new(
        Url::parse("file:///theorems.usl").unwrap(),
        1,
        "theorem t1 { true }\ntheorem t2 { true }".to_string(),
    );

    // Document with contract and invariant
    let doc2 = Document::new(
        Url::parse("file:///contracts.usl").unwrap(),
        1,
        "type X = { v: Int }\ncontract X::m() -> Result<Unit> { requires { true } }\ninvariant inv { true }".to_string(),
    );

    // Document with temporal and security
    let doc3 = Document::new(
        Url::parse("file:///temporal.usl").unwrap(),
        1,
        "temporal tmp { always(true) }\nsecurity sec { forall x: Int . true }".to_string(),
    );

    stats.add_document(&doc1);
    stats.add_document(&doc2);
    stats.add_document(&doc3);

    assert_eq!(stats.file_count, 3);
    assert_eq!(stats.theorem_count(), 2);
    assert_eq!(stats.contract_count(), 1);
    assert_eq!(stats.invariant_count(), 1);
    assert_eq!(stats.temporal_count(), 1);
    assert_eq!(stats.security_count(), 1);
    assert_eq!(stats.property_count(), 6);
    assert_eq!(
        stats.detailed_breakdown(),
        "2 thm | 1 ctr | 1 tmp | 1 inv | 1 sec"
    );
}

#[test]
fn test_workspace_stats_with_parse_error() {
    let mut stats = WorkspaceStats::new();

    let good_doc = Document::new(
        Url::parse("file:///good.usl").unwrap(),
        1,
        "theorem test { true }".to_string(),
    );

    let bad_doc = Document::new(
        Url::parse("file:///bad.usl").unwrap(),
        1,
        "this is not valid { }".to_string(),
    );

    stats.add_document(&good_doc);
    stats.add_document(&bad_doc);

    assert_eq!(stats.file_count, 2);
    assert_eq!(stats.files_with_errors, 1);
    assert_eq!(stats.property_count(), 1); // Only from good doc
    assert_eq!(stats.error_summary(), Some("1 parse error".to_string()));
}

#[test]
fn test_workspace_stats_pluralization() {
    // Test singular
    let mut stats1 = WorkspaceStats::new();
    let doc1 = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type T = { x: Int }".to_string(),
    );
    stats1.add_document(&doc1);
    assert_eq!(stats1.summary(), "1 file, 1 type");

    // Test plural
    let mut stats2 = WorkspaceStats::new();
    let doc2a = Document::new(
        Url::parse("file:///test1.usl").unwrap(),
        1,
        "type A = { x: Int }\ntype B = { y: Int }".to_string(),
    );
    let doc2b = Document::new(
        Url::parse("file:///test2.usl").unwrap(),
        1,
        "theorem t1 { true }\ntheorem t2 { true }".to_string(),
    );
    stats2.add_document(&doc2a);
    stats2.add_document(&doc2b);
    assert_eq!(stats2.summary(), "2 files, 2 types, 2 properties");
}

#[test]
fn test_generate_workspace_stats_lenses_empty() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "// Just a comment".to_string(),
    );
    let stats = WorkspaceStats::new();

    let lenses = generate_workspace_stats_lenses(&doc, &stats);
    // No lenses for empty workspace with single file
    assert!(lenses.is_empty());
}

#[test]
fn test_generate_workspace_stats_lenses_with_content() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem test { true }".to_string(),
    );

    let mut stats = WorkspaceStats::new();
    stats.add_document(&doc);
    // Simulate second file to trigger workspace lenses
    stats.file_count = 2;

    let lenses = generate_workspace_stats_lenses(&doc, &stats);

    // Should have: summary, breakdown, verify all
    assert_eq!(lenses.len(), 3);

    // All lenses at line 0
    assert!(lenses.iter().all(|l| l.range.start.line == 0));

    // Check summary lens
    let summary_lens = &lenses[0];
    assert!(summary_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("Workspace"));
    assert_eq!(
        summary_lens.command.as_ref().unwrap().command,
        "dashprove.showWorkspaceStats"
    );

    // Check breakdown lens
    let breakdown_lens = &lenses[1];
    assert!(breakdown_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("thm"));
    assert_eq!(
        breakdown_lens.command.as_ref().unwrap().command,
        "dashprove.showPropertyBreakdown"
    );

    // Check verify all lens
    let verify_lens = &lenses[2];
    assert!(verify_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("Verify all"));
    assert_eq!(
        verify_lens.command.as_ref().unwrap().command,
        "dashprove.verifyAll"
    );
}

#[test]
fn test_generate_workspace_stats_lenses_with_errors() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem test { true }".to_string(),
    );

    let mut stats = WorkspaceStats::new();
    stats.file_count = 3;
    stats.files_with_errors = 2;
    stats.properties.total = 1;

    let lenses = generate_workspace_stats_lenses(&doc, &stats);

    // Should have: summary + error lens (no breakdown/verify since no properties counted)
    // Actually stats has property_count=1, so: summary, breakdown, verify, error = 4
    assert_eq!(lenses.len(), 4);

    // Check error lens
    let error_lens = &lenses[3];
    assert!(error_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("parse error"));
    assert_eq!(
        error_lens.command.as_ref().unwrap().command,
        "dashprove.showErrors"
    );
}

#[test]
fn test_workspace_stats_lens_arguments() {
    let doc = Document::new(
        Url::parse("file:///myfile.usl").unwrap(),
        1,
        "type T = { x: Int }".to_string(),
    );

    let mut stats = WorkspaceStats::new();
    stats.add_document(&doc);
    stats.file_count = 2; // Trigger workspace lenses

    let lenses = generate_workspace_stats_lenses(&doc, &stats);

    // Check that arguments include the document URI
    let lens = &lenses[0];
    let args = lens.command.as_ref().unwrap().arguments.as_ref().unwrap();
    assert_eq!(args.len(), 1);
    assert_eq!(args[0].as_str().unwrap(), "file:///myfile.usl");
}

#[test]
fn test_workspace_stats_verify_all_count() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t1 { true }\ntheorem t2 { true }\ntheorem t3 { true }".to_string(),
    );

    let mut stats = WorkspaceStats::new();
    stats.add_document(&doc);
    stats.file_count = 2; // Trigger workspace lenses

    let lenses = generate_workspace_stats_lenses(&doc, &stats);

    // Find verify all lens
    let verify_lens = lenses
        .iter()
        .find(|l| {
            l.command
                .as_ref()
                .map(|c| c.command == "dashprove.verifyAll")
                .unwrap_or(false)
        })
        .unwrap();

    // Should show correct count
    assert!(verify_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("3 properties"));
}

#[test]
fn test_workspace_stats_refinement_and_probabilistic() {
    let mut stats = WorkspaceStats::new();

    // Document with refinement
    let doc1 = Document::new(
        Url::parse("file:///refinement.usl").unwrap(),
        1,
        include_str!("../../../../examples/usl/refinement.usl").to_string(),
    );

    // Document with probabilistic
    let doc2 = Document::new(
        Url::parse("file:///prob.usl").unwrap(),
        1,
        "probabilistic error_bound {\n  probability(failure) <= 0.001\n}".to_string(),
    );

    stats.add_document(&doc1);
    stats.add_document(&doc2);

    assert!(stats.refinement_count() > 0);
    assert_eq!(stats.probabilistic_count(), 1);

    let breakdown = stats.detailed_breakdown();
    assert!(breakdown.contains("ref"));
    assert!(breakdown.contains("prob"));
}

#[test]
fn test_workspace_stats_no_lenses_single_empty_file() {
    let doc = Document::new(
        Url::parse("file:///empty.usl").unwrap(),
        1,
        "// Empty file".to_string(),
    );

    let mut stats = WorkspaceStats::new();
    stats.file_count = 1;
    // No types, no properties

    let lenses = generate_workspace_stats_lenses(&doc, &stats);
    // Single file with no content should not show workspace lenses
    assert!(lenses.is_empty());
}

#[test]
fn test_workspace_stats_shows_types_only_workspace() {
    let doc = Document::new(
        Url::parse("file:///types.usl").unwrap(),
        1,
        "type A = { x: Int }\ntype B = { y: Int }".to_string(),
    );

    let mut stats = WorkspaceStats::new();
    stats.add_document(&doc);
    stats.file_count = 2; // Trigger workspace lenses

    let lenses = generate_workspace_stats_lenses(&doc, &stats);

    // Should show summary even with just types (no properties)
    assert_eq!(lenses.len(), 1); // Only summary, no breakdown/verify for 0 properties

    let summary = &lenses[0].command.as_ref().unwrap().title;
    assert!(summary.contains("2 types"));
}

// ==================== Document Stats Tests ====================

#[test]
fn test_document_stats_empty() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "// Just a comment".to_string(),
    );

    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.type_count, 0);
    assert_eq!(stats.property_count(), 0);
    assert!(!stats.has_content());
    assert_eq!(stats.summary(), "Empty file");
    assert!(stats.detailed_breakdown().is_none());
}

#[test]
fn test_document_stats_with_types_only() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type Person = { name: String, age: Int }\ntype Address = { city: String }".to_string(),
    );

    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.type_count, 2);
    assert_eq!(stats.property_count(), 0);
    assert!(stats.has_content());
    assert_eq!(stats.summary(), "2 types");
    assert!(stats.detailed_breakdown().is_none());
}

#[test]
fn test_document_stats_single_theorem() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem test { true }".to_string(),
    );

    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.property_count(), 1);
    assert_eq!(stats.theorem_count(), 1);
    assert_eq!(stats.summary(), "1 property");
    assert_eq!(stats.detailed_breakdown(), Some("1 thm".to_string()));
}

#[test]
fn test_document_stats_mixed_properties() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type T = { x: Int }\ntheorem t1 { true }\ntheorem t2 { true }\ncontract T::m() -> Result<Unit> { requires { true } }\ninvariant i { true }".to_string(),
    );

    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.type_count, 1);
    assert_eq!(stats.property_count(), 4);
    assert_eq!(stats.theorem_count(), 2);
    assert_eq!(stats.contract_count(), 1);
    assert_eq!(stats.invariant_count(), 1);
    assert_eq!(stats.summary(), "1 type, 4 properties");
    assert_eq!(
        stats.detailed_breakdown(),
        Some("2 thm | 1 ctr | 1 inv".to_string())
    );
}

#[test]
fn test_document_stats_all_property_types() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type X = { v: Int }\ntheorem t { true }\ncontract X::m() -> Result<Unit> { requires { true } }\ntemporal tmp { always(true) }\ninvariant inv { true }\nprobabilistic prob { probability(ok) >= 0.99 }\nsecurity sec { forall x: Int . true }".to_string(),
    );

    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.theorem_count(), 1);
    assert_eq!(stats.contract_count(), 1);
    assert_eq!(stats.temporal_count(), 1);
    assert_eq!(stats.invariant_count(), 1);
    assert_eq!(stats.probabilistic_count(), 1);
    assert_eq!(stats.security_count(), 1);
    assert_eq!(stats.property_count(), 6);
    assert_eq!(
        stats.detailed_breakdown(),
        Some("1 thm | 1 ctr | 1 tmp | 1 inv | 1 prob | 1 sec".to_string())
    );
}

#[test]
fn test_document_stats_with_parse_error() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "this is not valid { }".to_string(),
    );

    let stats = DocumentStats::from_document(&doc);
    assert!(stats.has_parse_error);
    assert_eq!(stats.type_count, 0);
    assert_eq!(stats.property_count(), 0);
    assert!(!stats.has_content());
    assert_eq!(stats.summary(), "Parse error");
    assert_eq!(stats.error_summary(), Some("parse error".to_string()));
}

#[test]
fn test_document_stats_pluralization() {
    // Singular
    let doc1 = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type T = { x: Int }\ntheorem t { true }".to_string(),
    );
    let stats1 = DocumentStats::from_document(&doc1);
    assert_eq!(stats1.summary(), "1 type, 1 property");

    // Plural
    let doc2 = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type A = { x: Int }\ntype B = { y: Int }\ntheorem t1 { true }\ntheorem t2 { true }"
            .to_string(),
    );
    let stats2 = DocumentStats::from_document(&doc2);
    assert_eq!(stats2.summary(), "2 types, 2 properties");
}

#[test]
fn test_generate_document_stats_lenses_empty() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "// Just a comment".to_string(),
    );

    let lenses = generate_document_stats_lenses(&doc, 0);
    assert!(lenses.is_empty());
}

#[test]
fn test_generate_document_stats_lenses_types_only() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type T = { x: Int }".to_string(),
    );

    let lenses = generate_document_stats_lenses(&doc, 0);

    // Should have just the summary lens (no breakdown for types-only)
    assert_eq!(lenses.len(), 1);

    let summary_lens = &lenses[0];
    assert!(summary_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("This file"));
    assert!(summary_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("1 type"));
    assert_eq!(
        summary_lens.command.as_ref().unwrap().command,
        "dashprove.showDocumentStats"
    );
}

#[test]
fn test_generate_document_stats_lenses_with_properties() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type T = { x: Int }\ntheorem t1 { true }\ntheorem t2 { true }\ncontract T::m() -> Result<Unit> { requires { true } }".to_string(),
    );

    let lenses = generate_document_stats_lenses(&doc, 0);

    // Should have: summary, breakdown (multiple types), verify
    assert_eq!(lenses.len(), 3);

    // Check summary lens
    let summary_lens = &lenses[0];
    assert!(summary_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("This file"));
    assert!(summary_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("3 properties"));

    // Check breakdown lens
    let breakdown_lens = &lenses[1];
    assert!(breakdown_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("thm"));
    assert!(breakdown_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("ctr"));
    assert_eq!(
        breakdown_lens.command.as_ref().unwrap().command,
        "dashprove.showDocumentBreakdown"
    );

    // Check verify lens
    let verify_lens = &lenses[2];
    assert!(verify_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("Verify 3 in this file"));
    assert_eq!(
        verify_lens.command.as_ref().unwrap().command,
        "dashprove.verifyDocument"
    );
}

#[test]
fn test_generate_document_stats_lenses_with_parse_error() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "this is not valid { }".to_string(),
    );

    let lenses = generate_document_stats_lenses(&doc, 0);

    // Should have: summary, error lens
    assert_eq!(lenses.len(), 2);

    let summary_lens = &lenses[0];
    assert!(summary_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("Parse error"));

    let error_lens = &lenses[1];
    assert!(error_lens
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("parse error"));
    assert_eq!(
        error_lens.command.as_ref().unwrap().command,
        "dashprove.showDocumentErrors"
    );
}

#[test]
fn test_generate_document_stats_lenses_line_parameter() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type T = { x: Int }".to_string(),
    );

    // Test with different line numbers
    let lenses_line_0 = generate_document_stats_lenses(&doc, 0);
    let lenses_line_5 = generate_document_stats_lenses(&doc, 5);

    assert_eq!(lenses_line_0[0].range.start.line, 0);
    assert_eq!(lenses_line_5[0].range.start.line, 5);
}

#[test]
fn test_generate_document_stats_lenses_arguments() {
    let doc = Document::new(
        Url::parse("file:///myfile.usl").unwrap(),
        1,
        "theorem test { true }".to_string(),
    );

    let lenses = generate_document_stats_lenses(&doc, 0);

    // Check that arguments include the document URI
    let lens = &lenses[0];
    let args = lens.command.as_ref().unwrap().arguments.as_ref().unwrap();
    assert_eq!(args.len(), 1);
    assert_eq!(args[0].as_str().unwrap(), "file:///myfile.usl");
}

#[test]
fn test_generate_document_stats_no_breakdown_for_single_type() {
    // Single property type should not show breakdown
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t1 { true }\ntheorem t2 { true }".to_string(),
    );

    let lenses = generate_document_stats_lenses(&doc, 0);

    // Should have: summary, verify (no breakdown for 2 of same type)
    assert_eq!(lenses.len(), 2);
    assert!(lenses[0]
        .command
        .as_ref()
        .unwrap()
        .title
        .contains("This file"));
    assert!(lenses[1].command.as_ref().unwrap().title.contains("Verify"));
}

#[test]
fn test_generate_document_stats_breakdown_for_many_properties() {
    // More than 2 properties of same type should show breakdown
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t1 { true }\ntheorem t2 { true }\ntheorem t3 { true }".to_string(),
    );

    let lenses = generate_document_stats_lenses(&doc, 0);

    // Should have: summary, breakdown (> 2 properties), verify
    assert_eq!(lenses.len(), 3);
    assert!(lenses[1].command.as_ref().unwrap().title.contains("3 thm"));
}

#[test]
fn test_document_stats_with_refinement() {
    let doc = Document::new(
        Url::parse("file:///refinement.usl").unwrap(),
        1,
        include_str!("../../../../examples/usl/refinement.usl").to_string(),
    );

    let stats = DocumentStats::from_document(&doc);
    assert!(stats.refinement_count() > 0);
    assert!(stats.has_content());

    let breakdown = stats.detailed_breakdown().unwrap();
    assert!(breakdown.contains("ref"));
}

#[test]
fn test_document_stats_error_summary_type_errors() {
    // Test the error_summary method directly with synthetic data
    let stats = DocumentStats {
        type_error_count: 1,
        ..Default::default()
    };

    let error = stats.error_summary().unwrap();
    assert!(error.contains("type error"));
    assert!(!error.contains("type errors")); // Singular
}

#[test]
fn test_document_stats_error_summary_multiple_type_errors() {
    // Test the error_summary method directly with synthetic data
    let stats = DocumentStats {
        type_error_count: 2,
        ..Default::default()
    };

    let error = stats.error_summary().unwrap();
    assert!(error.contains("2 type errors")); // Plural
}

#[test]
fn test_document_stats_error_summary_combined() {
    // Test parse error + type errors combined
    let stats = DocumentStats {
        has_parse_error: true,
        type_error_count: 3,
        ..Default::default()
    };

    let error = stats.error_summary().unwrap();
    assert!(error.contains("parse error"));
    assert!(error.contains("3 type errors"));
}

#[test]
fn test_document_stats_no_error_summary() {
    let stats = DocumentStats::default();
    assert!(stats.error_summary().is_none());
}

#[test]
fn test_generate_all_code_lenses_with_stats() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type T = { x: Int }\ntheorem t { true }".to_string(),
    );

    let mut workspace_stats = WorkspaceStats::new();
    workspace_stats.file_count = 2;
    workspace_stats.type_count = 1;
    workspace_stats.properties.total = 1;
    workspace_stats.properties.theorems = 1;

    let lenses = generate_all_code_lenses(&doc, &workspace_stats);
    assert!(!lenses.is_empty());

    let titles: Vec<_> = lenses
        .iter()
        .filter_map(|l| l.command.as_ref().map(|c| c.title.clone()))
        .collect();

    assert!(titles.iter().any(|t| t.contains("Workspace")));

    let document_lens = lenses
        .iter()
        .find(|l| {
            l.command
                .as_ref()
                .map(|c| c.title.contains("This file"))
                .unwrap_or(false)
        })
        .expect("document stats lens missing");
    assert_eq!(document_lens.range.start.line, 1);

    assert!(lenses.iter().any(|l| {
        l.command
            .as_ref()
            .map(|c| c.command == "dashprove.verify")
            .unwrap_or(false)
    }));
}

#[test]
fn test_generate_all_code_lenses_without_workspace_lenses() {
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t { true }".to_string(),
    );

    let workspace_stats = WorkspaceStats::new();
    let lenses = generate_all_code_lenses(&doc, &workspace_stats);

    assert!(!lenses.iter().any(|l| l
        .command
        .as_ref()
        .is_some_and(|c| c.title.contains("Workspace"))));

    let document_lens = lenses
        .iter()
        .find(|l| {
            l.command
                .as_ref()
                .map(|c| c.title.contains("This file"))
                .unwrap_or(false)
        })
        .expect("document stats lens missing");
    assert_eq!(document_lens.range.start.line, 0);
}

// Mutation-killing tests for count methods with zero values
#[test]
fn test_document_stats_contract_count_zero() {
    // Test contract_count returns 0 when no contracts exist
    // Kills mutant: replace contract_count -> usize with 1
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t { true }".to_string(), // theorem, not contract
    );
    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.contract_count(), 0);
}

#[test]
fn test_document_stats_temporal_count_zero() {
    // Test temporal_count returns 0 when no temporal properties exist
    // Kills mutant: replace temporal_count -> usize with 1
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t { true }".to_string(), // theorem, not temporal
    );
    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.temporal_count(), 0);
}

#[test]
fn test_document_stats_invariant_count_zero() {
    // Test invariant_count returns 0 when no invariants exist
    // Kills mutant: replace invariant_count -> usize with 1
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t { true }".to_string(), // theorem, not invariant
    );
    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.invariant_count(), 0);
}

#[test]
fn test_document_stats_refinement_count_zero() {
    // Test refinement_count returns 0 when no refinements exist
    // Kills mutant: replace refinement_count -> usize with 1
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t { true }".to_string(), // theorem, not refinement
    );
    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.refinement_count(), 0);
}

#[test]
fn test_document_stats_probabilistic_count_zero() {
    // Test probabilistic_count returns 0 when no probabilistic properties exist
    // Kills mutant: replace probabilistic_count -> usize with 1
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t { true }".to_string(), // theorem, not probabilistic
    );
    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.probabilistic_count(), 0);
}

#[test]
fn test_document_stats_security_count_zero() {
    // Test security_count returns 0 when no security properties exist
    // Kills mutant: replace security_count -> usize with 1
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t { true }".to_string(), // theorem, not security
    );
    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.security_count(), 0);
}

#[test]
fn test_document_stats_type_variety_zero() {
    // Test type_variety returns 0 for empty document
    // Kills mutant: replace type_variety -> usize with 1
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "type T = { x: Int }".to_string(), // Only types, no properties
    );
    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.type_variety(), 0);
}

#[test]
fn test_document_stats_type_variety_two() {
    // Test type_variety returns >1 for multiple property types
    // Kills mutant: replace type_variety -> usize with 0 or 1
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t1 { true }\ninvariant i1 { true }".to_string(),
    );
    let stats = DocumentStats::from_document(&doc);
    assert_eq!(stats.type_variety(), 2); // theorem + invariant
}

#[test]
fn test_generate_document_stats_lenses_breakdown_condition() {
    // Test that breakdown lens appears when type_variety > 1
    // Kills mutant: replace > with < in line 178
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t1 { true }\ninvariant i1 { true }".to_string(),
    );
    let lenses = generate_document_stats_lenses(&doc, 0);

    // Should have breakdown lens since type_variety == 2 > 1
    let breakdown_lens = lenses.iter().find(|l| {
        l.command
            .as_ref()
            .map(|c| c.title.contains("📋"))
            .unwrap_or(false)
    });
    assert!(
        breakdown_lens.is_some(),
        "breakdown lens should appear when type_variety > 1"
    );
}

#[test]
fn test_generate_document_stats_lenses_no_breakdown_single_type() {
    // Test that breakdown lens does NOT appear when type_variety == 1 and property_count <= 2
    // Kills mutant: replace > with < in line 178
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t1 { true }\ntheorem t2 { true }".to_string(), // 2 theorems, same type
    );
    let lenses = generate_document_stats_lenses(&doc, 0);

    // Should NOT have breakdown lens since type_variety == 1 and property_count == 2
    let breakdown_lens = lenses.iter().find(|l| {
        l.command
            .as_ref()
            .map(|c| c.title.contains("📋"))
            .unwrap_or(false)
    });
    assert!(
        breakdown_lens.is_none(),
        "breakdown lens should not appear with single type and 2 properties"
    );
}

#[test]
fn test_contract_verification_strategy_goal_description() {
    // Test that format_contract_goal produces correct output
    // Kills mutant: replace format_contract_goal -> String with String::new() or "xyzzy"
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "contract Stack::push(self: Stack, item: Int) -> Result<Unit> {\n  requires { true }\n  ensures { true }\n}"
            .to_string(),
    );

    let lenses = generate_code_lenses(&doc);

    // Find the verification strategies lens
    let strategy_lens = lenses
        .iter()
        .find(|l| {
            l.command
                .as_ref()
                .map(|c| c.title.contains("verification strategies"))
                .unwrap_or(false)
        })
        .expect("verification strategies lens should exist");

    // Check that the first argument (goal_description) contains the contract path
    let args = strategy_lens
        .command
        .as_ref()
        .unwrap()
        .arguments
        .as_ref()
        .unwrap();
    let goal_description = args[0].as_str().unwrap();
    assert!(
        goal_description.contains("Stack::push"),
        "goal_description should contain contract path, got: {}",
        goal_description
    );
    assert!(
        goal_description.contains("requires") || goal_description.contains("contract"),
        "goal_description should contain requires/ensures or contract, got: {}",
        goal_description
    );
}

#[test]
fn test_contract_goal_description_empty_conditions() {
    // Test that format_contract_goal handles contracts with no requires/ensures
    // Kills mutant: replace format_contract_goal -> String with String::new() or "xyzzy"
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "contract Stack::empty(self: Stack) -> Result<bool> {\n}\n".to_string(),
    );

    let lenses = generate_code_lenses(&doc);

    // Find the verification strategies lens
    let strategy_lens = lenses
        .iter()
        .find(|l| {
            l.command
                .as_ref()
                .map(|c| c.title.contains("verification strategies"))
                .unwrap_or(false)
        })
        .expect("verification strategies lens should exist");

    // Check that the first argument contains "contract Stack::empty"
    let args = strategy_lens
        .command
        .as_ref()
        .unwrap()
        .arguments
        .as_ref()
        .unwrap();
    let goal_description = args[0].as_str().unwrap();
    assert!(
        goal_description.contains("contract"),
        "goal_description should contain 'contract', got: {}",
        goal_description
    );
    assert!(
        goal_description.contains("Stack::empty"),
        "goal_description should contain 'Stack::empty', got: {}",
        goal_description
    );
}

// Tests for PropertyCounts::merge
#[test]
fn test_property_counts_merge_total() {
    // Kills mutant: replace merge with () and += with -= or *= in merge
    let mut counts1 = PropertyCounts {
        total: 2,
        theorems: 1,
        contracts: 1,
        ..Default::default()
    };

    let counts2 = PropertyCounts {
        total: 3,
        theorems: 2,
        contracts: 0,
        temporal: 1,
        ..Default::default()
    };

    counts1.merge(&counts2);

    assert_eq!(counts1.total, 5, "total should be merged (2 + 3 = 5)");
    assert_eq!(counts1.theorems, 3, "theorems should be merged (1 + 2 = 3)");
    assert_eq!(counts1.contracts, 1, "contracts should remain (1 + 0 = 1)");
    assert_eq!(counts1.temporal, 1, "temporal should be merged (0 + 1 = 1)");
}

#[test]
fn test_property_counts_merge_all_fields() {
    // Kills mutant: replace += with -= or *= in merge for each field
    let mut counts1 = PropertyCounts {
        total: 7,
        theorems: 1,
        contracts: 1,
        temporal: 1,
        invariants: 1,
        refinements: 1,
        probabilistic: 1,
        security: 1,
    };

    let counts2 = PropertyCounts {
        total: 7,
        theorems: 2,
        contracts: 2,
        temporal: 2,
        invariants: 2,
        refinements: 2,
        probabilistic: 2,
        security: 2,
    };

    counts1.merge(&counts2);

    assert_eq!(counts1.total, 14, "total should be merged");
    assert_eq!(counts1.theorems, 3, "theorems should be merged");
    assert_eq!(counts1.contracts, 3, "contracts should be merged");
    assert_eq!(counts1.temporal, 3, "temporal should be merged");
    assert_eq!(counts1.invariants, 3, "invariants should be merged");
    assert_eq!(counts1.refinements, 3, "refinements should be merged");
    assert_eq!(counts1.probabilistic, 3, "probabilistic should be merged");
    assert_eq!(counts1.security, 3, "security should be merged");
}

// Tests for PropertyCounts::type_variety comparisons
#[test]
fn test_property_counts_type_variety_comparison_contracts() {
    // Kills mutant: replace > with < in type_variety for contracts
    let counts = PropertyCounts {
        contracts: 1,
        ..Default::default()
    };
    assert_eq!(
        counts.type_variety(),
        1,
        "contracts > 0 should count as 1 type"
    );

    let counts_zero = PropertyCounts {
        contracts: 0,
        theorems: 1,
        ..Default::default()
    };
    assert_eq!(
        counts_zero.type_variety(),
        1,
        "contracts == 0 should not contribute"
    );
}

#[test]
fn test_property_counts_type_variety_comparison_temporal() {
    // Kills mutant: replace > with < in type_variety for temporal
    let counts = PropertyCounts {
        temporal: 1,
        ..Default::default()
    };
    assert_eq!(
        counts.type_variety(),
        1,
        "temporal > 0 should count as 1 type"
    );
}

#[test]
fn test_property_counts_type_variety_comparison_refinements() {
    // Kills mutant: replace > with < in type_variety for refinements
    let counts = PropertyCounts {
        refinements: 1,
        ..Default::default()
    };
    assert_eq!(
        counts.type_variety(),
        1,
        "refinements > 0 should count as 1 type"
    );
}

#[test]
fn test_property_counts_type_variety_comparison_security() {
    // Kills mutant: replace > with < in type_variety for security
    let counts = PropertyCounts {
        security: 1,
        ..Default::default()
    };
    assert_eq!(
        counts.type_variety(),
        1,
        "security > 0 should count as 1 type"
    );
}

#[test]
fn test_property_counts_type_variety_comparison_probabilistic() {
    // Kills mutant: replace > with < in type_variety for probabilistic (line 105)
    let counts = PropertyCounts {
        probabilistic: 1,
        ..Default::default()
    };
    assert_eq!(
        counts.type_variety(),
        1,
        "probabilistic > 0 should count as 1 type"
    );

    // Also verify that 0 doesn't count
    let counts_zero = PropertyCounts {
        probabilistic: 0,
        theorems: 1,
        ..Default::default()
    };
    assert_eq!(
        counts_zero.type_variety(),
        1,
        "probabilistic == 0 should not contribute"
    );
}

// Tests for WorkspaceStats
#[test]
fn test_workspace_stats_add_document_type_errors() {
    // Kills mutant: replace += with -= or *= for files_with_type_errors
    // Create documents with duplicate type definitions to trigger type errors
    let mut stats = WorkspaceStats::new();

    // First document with type error (duplicate type definition)
    let doc1 = Document::new(
        Url::parse("file:///test1.usl").unwrap(),
        1,
        "type Node = { id: Int }\ntype Node = { name: String }".to_string(),
    );
    assert!(!doc1.type_errors.is_empty(), "doc1 should have type errors");
    stats.add_document(&doc1);
    assert_eq!(
        stats.files_with_type_errors, 1,
        "files_with_type_errors should be 1"
    );

    // Second document with type error
    let doc2 = Document::new(
        Url::parse("file:///test2.usl").unwrap(),
        1,
        "type Foo = { x: Int }\ntype Foo = { y: Int }".to_string(),
    );
    assert!(!doc2.type_errors.is_empty(), "doc2 should have type errors");
    stats.add_document(&doc2);
    assert_eq!(
        stats.files_with_type_errors, 2,
        "files_with_type_errors should be 2"
    );
}

#[test]
fn test_workspace_stats_contract_count_zero() {
    // Kills mutant: replace contract_count -> usize with 1
    let stats = WorkspaceStats::new(); // Empty stats
    assert_eq!(
        stats.contract_count(),
        0,
        "contract_count should be 0 for empty stats"
    );
}

#[test]
fn test_workspace_stats_temporal_count_zero() {
    // Kills mutant: replace temporal_count -> usize with 1
    let stats = WorkspaceStats::new();
    assert_eq!(
        stats.temporal_count(),
        0,
        "temporal_count should be 0 for empty stats"
    );
}

#[test]
fn test_workspace_stats_invariant_count_zero() {
    // Kills mutant: replace invariant_count -> usize with 1
    let stats = WorkspaceStats::new();
    assert_eq!(
        stats.invariant_count(),
        0,
        "invariant_count should be 0 for empty stats"
    );
}

#[test]
fn test_workspace_stats_refinement_count_zero() {
    // Kills mutant: replace refinement_count -> usize with 1
    let stats = WorkspaceStats::new();
    assert_eq!(
        stats.refinement_count(),
        0,
        "refinement_count should be 0 for empty stats"
    );
}

#[test]
fn test_workspace_stats_probabilistic_count_zero() {
    // Kills mutant: replace probabilistic_count -> usize with 1
    let stats = WorkspaceStats::new();
    assert_eq!(
        stats.probabilistic_count(),
        0,
        "probabilistic_count should be 0 for empty stats"
    );
}

#[test]
fn test_workspace_stats_security_count_zero() {
    // Kills mutant: replace security_count -> usize with 1
    let stats = WorkspaceStats::new();
    assert_eq!(
        stats.security_count(),
        0,
        "security_count should be 0 for empty stats"
    );
}

#[test]
fn test_workspace_stats_error_summary_files_with_errors_boundary() {
    // Kills mutant: replace > with >= in error_summary (line 131)
    // Test that files_with_errors == 0 returns None
    let stats = WorkspaceStats::new();
    assert!(
        stats.error_summary().is_none(),
        "error_summary should be None when no errors"
    );

    // Test that files_with_errors == 1 returns Some
    let mut stats_with_error = WorkspaceStats::new();
    stats_with_error.files_with_errors = 1;
    assert!(
        stats_with_error.error_summary().is_some(),
        "error_summary should be Some when files_with_errors > 0"
    );
    let summary = stats_with_error.error_summary().unwrap();
    assert!(
        summary.contains("parse error"),
        "summary should mention parse error"
    );

    // Critical: Test that files_with_errors == 0 does NOT add "parse error" to summary
    // even when files_with_type_errors > 0 (this kills the > to >= mutation)
    let mut stats_type_errors_only = WorkspaceStats::new();
    stats_type_errors_only.files_with_errors = 0;
    stats_type_errors_only.files_with_type_errors = 1;
    let summary = stats_type_errors_only.error_summary().unwrap();
    assert!(
        !summary.contains("parse error"),
        "summary should NOT mention parse error when files_with_errors == 0, got: {}",
        summary
    );
    assert!(
        summary.contains("type error"),
        "summary should mention type error when files_with_type_errors > 0"
    );
}

#[test]
fn test_workspace_stats_error_summary_type_errors_boundary() {
    // Kills mutant: replace > with < in error_summary (line 139)
    let mut stats = WorkspaceStats::new();
    stats.files_with_type_errors = 1;
    let summary = stats
        .error_summary()
        .expect("should have summary with type errors");
    assert!(
        summary.contains("type error"),
        "summary should mention type error"
    );

    // Also test with 0 type errors
    let mut stats_no_type_err = WorkspaceStats::new();
    stats_no_type_err.files_with_errors = 1; // Has parse errors but no type errors
    let summary = stats_no_type_err.error_summary().unwrap();
    assert!(
        !summary.contains("type error"),
        "summary should not mention type error when 0"
    );
}

#[test]
fn test_generate_workspace_stats_lenses_file_count_boundary() {
    // Kills mutant: replace < with <= in generate_workspace_stats_lenses (line 159)
    // and replace && with || (line 159)
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "".to_string(), // Empty file
    );

    // With file_count == 1 and no content, should return empty
    let mut stats_one_file = WorkspaceStats::new();
    stats_one_file.file_count = 1;
    let lenses = generate_workspace_stats_lenses(&doc, &stats_one_file);
    assert!(
        lenses.is_empty(),
        "should not show workspace lenses with 1 empty file"
    );

    // With file_count == 2, should show workspace lenses
    let mut stats_two_files = WorkspaceStats::new();
    stats_two_files.file_count = 2;
    let lenses = generate_workspace_stats_lenses(&doc, &stats_two_files);
    assert!(
        !lenses.is_empty(),
        "should show workspace lenses with 2 files"
    );
}

#[test]
fn test_generate_workspace_stats_lenses_with_content_single_file() {
    // Tests that single file WITH content shows workspace lenses
    // This tests the && vs || mutation
    let doc = Document::new(
        Url::parse("file:///test.usl").unwrap(),
        1,
        "theorem t { true }".to_string(),
    );

    let mut stats = WorkspaceStats::new();
    stats.file_count = 1;
    stats.properties.total = 1;
    stats.properties.theorems = 1;

    let lenses = generate_workspace_stats_lenses(&doc, &stats);
    // With content (property_count > 0), should show workspace lenses even with 1 file
    assert!(
        !lenses.is_empty(),
        "should show workspace lenses when single file has content"
    );
}
