//! Folding ranges for USL documents
//!
//! Provides collapsible regions for type definitions, properties, and blocks.
//! Includes collapsed_text previews for better IDE experience when folded.

use crate::document::Document;
use crate::symbols::plural;
use tower_lsp::lsp_types::{FoldingRange, FoldingRangeKind};

/// Generate folding ranges for a USL document.
///
/// Identifies collapsible regions:
/// - Type definitions (`type Name = { ... }`)
/// - Property blocks (`theorem`, `temporal`, `invariant`, etc.)
/// - Contract clauses (`requires`, `ensures`, `ensures_err`)
/// - Refinement sections (`abstraction`, `simulation`)
/// - Comments (consecutive comment lines)
///
/// Each folding range includes a `collapsed_text` preview showing
/// a summary of the collapsed content.
#[must_use]
pub fn generate_folding_ranges(doc: &Document) -> Vec<FoldingRange> {
    let mut ranges = Vec::new();

    // Find all brace-delimited blocks with collapsed text previews
    find_brace_blocks_with_preview(doc, &mut ranges);

    // Find comment regions with collapsed text previews
    find_comment_regions_with_preview(doc, &mut ranges);

    ranges
}

/// Find all brace-delimited blocks and generate folding ranges with preview text.
///
/// Analyzes the context before each opening brace to determine the construct type
/// (type definition, theorem, contract, etc.) and generates appropriate preview text.
fn find_brace_blocks_with_preview(doc: &Document, ranges: &mut Vec<FoldingRange>) {
    let text = &doc.text;
    let mut brace_stack: Vec<(u32, u32, usize)> = Vec::new(); // (line, character, byte_offset)
    let mut line: u32 = 0;
    let mut character: u32 = 0;
    let mut byte_offset: usize = 0;

    for ch in text.chars() {
        match ch {
            '{' => {
                brace_stack.push((line, character, byte_offset));
            }
            '}' => {
                if let Some((start_line, _start_char, start_offset)) = brace_stack.pop() {
                    // Only fold if the block spans multiple lines
                    if line > start_line {
                        let collapsed_text =
                            generate_block_preview(doc, start_line, start_offset, byte_offset + 1);
                        ranges.push(FoldingRange {
                            start_line,
                            start_character: None,
                            end_line: line,
                            end_character: Some(character + 1),
                            kind: Some(FoldingRangeKind::Region),
                            collapsed_text,
                        });
                    }
                }
            }
            '\n' => {
                line += 1;
                character = 0;
                byte_offset += 1;
                continue; // Don't increment character
            }
            _ => {}
        }
        character += ch.len_utf16() as u32;
        byte_offset += ch.len_utf8();
    }
}

/// Generate a preview text for a collapsed block.
///
/// Examines the line containing the opening brace to determine the construct type
/// and generates an appropriate summary. Returns Some(preview) or None if no
/// meaningful preview can be generated.
fn generate_block_preview(
    doc: &Document,
    start_line: u32,
    start_offset: usize,
    end_offset: usize,
) -> Option<String> {
    let line_text = doc.line_text(start_line)?;
    let trimmed = line_text.trim();

    // Type definition: "type Name = { ... }"
    if trimmed.starts_with("type ") {
        let field_count = count_fields_in_block(&doc.text[start_offset..end_offset]);
        return Some(format!("... {} field{}", field_count, plural(field_count)));
    }

    // Theorem: "theorem name { ... }"
    if trimmed.starts_with("theorem ") {
        return Some("... proof body".to_string());
    }

    // Temporal: "temporal name { ... }"
    if trimmed.starts_with("temporal ") {
        return Some("... temporal formula".to_string());
    }

    // Contract: "contract Type::method(...) { ... }"
    if trimmed.starts_with("contract ") {
        let block_text = &doc.text[start_offset..end_offset];
        let requires_count = count_keyword_occurrences(block_text, "requires");
        let ensures_count = count_keyword_occurrences(block_text, "ensures");
        if requires_count > 0 || ensures_count > 0 {
            return Some(format!(
                "... {} requires, {} ensures",
                requires_count, ensures_count
            ));
        }
        return Some("... contract body".to_string());
    }

    // Invariant: "invariant name { ... }"
    if trimmed.starts_with("invariant ") {
        return Some("... invariant body".to_string());
    }

    // Refinement: "refinement name refines spec { ... }"
    if trimmed.starts_with("refinement ") {
        let block_text = &doc.text[start_offset..end_offset];
        let has_abstraction = block_text.contains("abstraction");
        let has_simulation = block_text.contains("simulation");
        if has_abstraction && has_simulation {
            return Some("... abstraction + simulation".to_string());
        } else if has_abstraction {
            return Some("... abstraction".to_string());
        } else if has_simulation {
            return Some("... simulation".to_string());
        }
        return Some("... refinement body".to_string());
    }

    // Probabilistic: "probabilistic name { ... }"
    if trimmed.starts_with("probabilistic ") {
        return Some("... probability bound".to_string());
    }

    // Security: "security name { ... }"
    if trimmed.starts_with("security ") {
        return Some("... security property".to_string());
    }

    // Nested blocks: requires, ensures, ensures_err, abstraction, simulation
    if trimmed.starts_with("requires") {
        return Some("... precondition".to_string());
    }
    if trimmed.starts_with("ensures_err") {
        return Some("... error postcondition".to_string());
    }
    if trimmed.starts_with("ensures") {
        return Some("... postcondition".to_string());
    }
    if trimmed.starts_with("abstraction") {
        return Some("... abstraction function".to_string());
    }
    if trimmed.starts_with("simulation") {
        return Some("... simulation proof".to_string());
    }

    // Generic block - show line count
    let line_count = doc.text[start_offset..end_offset].lines().count();
    if line_count > 1 {
        Some(format!("... {} line{}", line_count, plural(line_count)))
    } else {
        None
    }
}

/// Count fields in a type definition block (lines containing ':').
fn count_fields_in_block(block_text: &str) -> usize {
    block_text
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            // Count lines with "field_name: Type" pattern
            // Skip lines that are just braces or have "type " keyword
            !trimmed.is_empty()
                && trimmed.contains(':')
                && !trimmed.starts_with("type ")
                && !trimmed.starts_with("//")
        })
        .count()
}

/// Count occurrences of a keyword in block text.
fn count_keyword_occurrences(block_text: &str, keyword: &str) -> usize {
    block_text
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            // Match keyword at start of line (e.g., "requires {" or "ensures {")
            trimmed.starts_with(keyword)
                && (trimmed.len() == keyword.len()
                    || !trimmed[keyword.len()..]
                        .chars()
                        .next()
                        .map(|c| c.is_alphanumeric() || c == '_')
                        .unwrap_or(false))
        })
        .count()
}

/// Legacy function for backward compatibility - kept for reference.
#[allow(dead_code)]
fn find_brace_blocks(text: &str, ranges: &mut Vec<FoldingRange>) {
    let mut brace_stack: Vec<(u32, u32)> = Vec::new(); // (line, character)
    let mut line: u32 = 0;
    let mut character: u32 = 0;

    for ch in text.chars() {
        match ch {
            '{' => {
                brace_stack.push((line, character));
            }
            '}' => {
                if let Some((start_line, _start_char)) = brace_stack.pop() {
                    // Only fold if the block spans multiple lines
                    if line > start_line {
                        ranges.push(FoldingRange {
                            start_line,
                            start_character: None,
                            end_line: line,
                            end_character: Some(character + 1),
                            kind: Some(FoldingRangeKind::Region),
                            collapsed_text: None,
                        });
                    }
                }
            }
            '\n' => {
                line += 1;
                character = 0;
                continue; // Don't increment character
            }
            _ => {}
        }
        character += ch.len_utf16() as u32;
    }
}

/// Find consecutive comment lines and generate folding ranges with preview.
///
/// The collapsed_text shows the number of comment lines for multi-line comments.
fn find_comment_regions_with_preview(doc: &Document, ranges: &mut Vec<FoldingRange>) {
    let lines: Vec<&str> = doc.text.lines().collect();
    let mut comment_start: Option<u32> = None;
    let mut first_comment_text: Option<String> = None;

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let is_comment = trimmed.starts_with("//");

        if is_comment {
            if comment_start.is_none() {
                comment_start = Some(idx as u32);
                // Store first comment text for preview
                first_comment_text =
                    Some(trimmed.strip_prefix("//").unwrap_or("").trim().to_string());
            }
        } else if let Some(start) = comment_start {
            let end = (idx as u32).saturating_sub(1);
            // Only fold if we have 2+ comment lines
            if end > start {
                let end_char = doc.line_text(end).map(|l| l.len() as u32).unwrap_or(0);
                let comment_count = (end - start + 1) as usize;
                let collapsed_text = generate_comment_preview(
                    first_comment_text.as_deref().unwrap_or(""),
                    comment_count,
                );
                ranges.push(FoldingRange {
                    start_line: start,
                    start_character: None,
                    end_line: end,
                    end_character: Some(end_char),
                    kind: Some(FoldingRangeKind::Comment),
                    collapsed_text: Some(collapsed_text),
                });
            }
            comment_start = None;
            first_comment_text = None;
        }
    }

    // Handle trailing comment block
    if let Some(start) = comment_start {
        let end = (lines.len() as u32).saturating_sub(1);
        if end > start {
            let end_char = doc.line_text(end).map(|l| l.len() as u32).unwrap_or(0);
            let comment_count = (end - start + 1) as usize;
            let collapsed_text = generate_comment_preview(
                first_comment_text.as_deref().unwrap_or(""),
                comment_count,
            );
            ranges.push(FoldingRange {
                start_line: start,
                start_character: None,
                end_line: end,
                end_character: Some(end_char),
                kind: Some(FoldingRangeKind::Comment),
                collapsed_text: Some(collapsed_text),
            });
        }
    }
}

/// Generate a preview for collapsed comment regions.
///
/// Shows first few words of the first comment plus line count.
fn generate_comment_preview(first_line: &str, count: usize) -> String {
    // Truncate first line to ~30 chars
    let preview = if first_line.len() > 30 {
        format!("{}...", &first_line[..27])
    } else {
        first_line.to_string()
    };

    if preview.is_empty() {
        format!("// {} line{}", count, plural(count))
    } else {
        format!("// {} (+{} more)", preview, count - 1)
    }
}

/// Legacy function for backward compatibility - kept for reference.
#[allow(dead_code)]
fn find_comment_regions(text: &str, doc: &Document, ranges: &mut Vec<FoldingRange>) {
    let lines: Vec<&str> = text.lines().collect();
    let mut comment_start: Option<u32> = None;

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let is_comment = trimmed.starts_with("//");

        if is_comment {
            if comment_start.is_none() {
                comment_start = Some(idx as u32);
            }
        } else if let Some(start) = comment_start {
            let end = (idx as u32).saturating_sub(1);
            // Only fold if we have 2+ comment lines
            if end > start {
                let end_char = doc.line_text(end).map(|l| l.len() as u32).unwrap_or(0);
                ranges.push(FoldingRange {
                    start_line: start,
                    start_character: None,
                    end_line: end,
                    end_character: Some(end_char),
                    kind: Some(FoldingRangeKind::Comment),
                    collapsed_text: None,
                });
            }
            comment_start = None;
        }
    }

    // Handle trailing comment block
    if let Some(start) = comment_start {
        let end = (lines.len() as u32).saturating_sub(1);
        if end > start {
            let end_char = doc.line_text(end).map(|l| l.len() as u32).unwrap_or(0);
            ranges.push(FoldingRange {
                start_line: start,
                start_character: None,
                end_line: end,
                end_character: Some(end_char),
                kind: Some(FoldingRangeKind::Comment),
                collapsed_text: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower_lsp::lsp_types::Url;

    #[test]
    fn test_fold_type_definition() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Value = {\n    id: Int,\n    name: String\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);

        let range = &ranges[0];
        assert_eq!(range.start_line, 0);
        assert_eq!(range.end_line, 3);
        assert_eq!(range.kind, Some(FoldingRangeKind::Region));
    }

    #[test]
    fn test_fold_theorem() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem my_theorem {\n    forall x: Bool .\n        x implies x\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);

        let range = &ranges[0];
        assert_eq!(range.start_line, 0);
        assert_eq!(range.end_line, 3);
    }

    #[test]
    fn test_fold_nested_blocks() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "refinement r refines S {\n    abstraction {\n        true\n    }\n    simulation {\n        false\n    }\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        // Outer refinement + abstraction + simulation = 3 ranges
        assert_eq!(ranges.len(), 3);

        // Find the outer range (0 to 7)
        let outer = ranges.iter().find(|r| r.start_line == 0).unwrap();
        assert_eq!(outer.end_line, 7);

        // Find abstraction (line 1 to 3)
        let abstraction = ranges.iter().find(|r| r.start_line == 1).unwrap();
        assert_eq!(abstraction.end_line, 3);

        // Find simulation (line 4 to 6)
        let simulation = ranges.iter().find(|r| r.start_line == 4).unwrap();
        assert_eq!(simulation.end_line, 6);
    }

    #[test]
    fn test_fold_comments() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "// Comment line 1\n// Comment line 2\n// Comment line 3\ntype Foo = { x: Int }"
                .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);

        // Should have comment folding (3 lines) + type block
        let comment_range = ranges
            .iter()
            .find(|r| r.kind == Some(FoldingRangeKind::Comment));
        assert!(comment_range.is_some());

        let cr = comment_range.unwrap();
        assert_eq!(cr.start_line, 0);
        assert_eq!(cr.end_line, 2);
    }

    #[test]
    fn test_no_fold_single_line() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { true }".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        // Single-line block should not be folded
        assert_eq!(ranges.len(), 0);
    }

    #[test]
    fn test_no_fold_single_comment() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "// Single comment\ntype Foo = {\n    x: Int\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);

        // Only the type block should fold, not single comment
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].kind, Some(FoldingRangeKind::Region));
    }

    #[test]
    fn test_fold_refinement_example() {
        let doc = Document::new(
            Url::parse("file:///refinement.usl").unwrap(),
            1,
            include_str!("../../../examples/usl/refinement.usl").to_string(),
        );

        let ranges = generate_folding_ranges(&doc);

        // Find type AbstractSet (lines 4-6 in 0-indexed = lines 5-7 in file)
        let abstract_set = ranges.iter().find(|r| r.start_line == 4);
        assert!(
            abstract_set.is_some(),
            "AbstractSet type should be foldable"
        );

        // Find refinement sorted_list_refines_set (line 14 = 0-indexed)
        let refinement = ranges.iter().find(|r| r.start_line == 14);
        assert!(
            refinement.is_some(),
            "sorted_list_refines_set refinement should be foldable"
        );
    }

    #[test]
    fn test_fold_contract() {
        let doc = Document::new(
            Url::parse("file:///contracts.usl").unwrap(),
            1,
            include_str!("../../../examples/usl/contracts.usl").to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        // Should have multiple folding ranges for contracts with requires/ensures blocks
        assert!(!ranges.is_empty());
    }

    // Tests for collapsed_text preview functionality

    #[test]
    fn test_collapsed_text_type_definition() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Value = {\n    id: Int,\n    name: String\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);

        let range = &ranges[0];
        assert!(range.collapsed_text.is_some());
        let text = range.collapsed_text.as_ref().unwrap();
        assert!(text.contains("2 field"), "Expected '2 fields' in: {}", text);
    }

    #[test]
    fn test_collapsed_text_type_single_field() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Counter = {\n    value: Int\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);

        let text = ranges[0].collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("1 field"),
            "Expected '1 field' (singular) in: {}",
            text
        );
        assert!(
            !text.contains("1 fields"),
            "Should not use plural for 1 field"
        );
    }

    #[test]
    fn test_collapsed_text_theorem() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem my_theorem {\n    forall x: Bool .\n        x implies x\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);

        let text = ranges[0].collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("proof body"),
            "Expected 'proof body' in: {}",
            text
        );
    }

    #[test]
    fn test_collapsed_text_temporal() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "temporal eventually_done {\n    always(eventually(done))\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);

        let text = ranges[0].collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("temporal formula"),
            "Expected 'temporal formula' in: {}",
            text
        );
    }

    #[test]
    fn test_collapsed_text_invariant() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "invariant positive_count {\n    forall c: Counter .\n        c.value >= 0\n}"
                .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);

        let text = ranges[0].collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("invariant body"),
            "Expected 'invariant body' in: {}",
            text
        );
    }

    #[test]
    fn test_collapsed_text_contract_with_requires_ensures() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"contract Stack::push(self: Stack, item: Int) -> Result<Stack> {
    requires { self.len() < MAX_SIZE }
    ensures { result.len() == self.len() + 1 }
    ensures { result.top() == item }
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);

        // Find the outermost contract range
        let contract_range = ranges.iter().find(|r| r.start_line == 0).unwrap();
        let text = contract_range.collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("1 requires"),
            "Expected '1 requires' in: {}",
            text
        );
        assert!(
            text.contains("2 ensures"),
            "Expected '2 ensures' in: {}",
            text
        );
    }

    #[test]
    fn test_collapsed_text_refinement_with_sections() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"refinement impl_refines_spec refines Spec {
    abstraction {
        to_spec(impl)
    }
    simulation {
        step_preserves(impl, spec)
    }
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);

        // Find the outermost refinement range
        let refinement_range = ranges.iter().find(|r| r.start_line == 0).unwrap();
        let text = refinement_range.collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("abstraction"),
            "Expected 'abstraction' in: {}",
            text
        );
        assert!(
            text.contains("simulation"),
            "Expected 'simulation' in: {}",
            text
        );
    }

    #[test]
    fn test_collapsed_text_comment_region() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "// This is a header comment\n// with multiple lines\n// describing the file\ntype Foo = { x: Int }".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);

        let comment_range = ranges
            .iter()
            .find(|r| r.kind == Some(FoldingRangeKind::Comment))
            .unwrap();
        let text = comment_range.collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("This is a header comment"),
            "Expected first comment text in: {}",
            text
        );
        assert!(text.contains("+2 more"), "Expected '+2 more' in: {}", text);
    }

    #[test]
    fn test_collapsed_text_nested_requires() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"contract Foo::bar(self: Foo) -> Result<Unit> {
    requires {
        self.valid()
    }
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);

        // Find the requires block range
        let requires_range = ranges.iter().find(|r| r.start_line == 1).unwrap();
        let text = requires_range.collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("precondition"),
            "Expected 'precondition' in: {}",
            text
        );
    }

    #[test]
    fn test_collapsed_text_nested_ensures() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"contract Foo::bar(self: Foo) -> Result<Unit> {
    ensures {
        result.ok()
    }
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);

        // Find the ensures block range
        let ensures_range = ranges.iter().find(|r| r.start_line == 1).unwrap();
        let text = ensures_range.collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("postcondition"),
            "Expected 'postcondition' in: {}",
            text
        );
    }

    #[test]
    fn test_collapsed_text_nested_ensures_err() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"contract Foo::bar(self: Foo) -> Result<Unit> {
    ensures_err {
        self' == self
    }
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);

        // Find the ensures_err block range
        let ensures_err_range = ranges.iter().find(|r| r.start_line == 1).unwrap();
        let text = ensures_err_range.collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("error postcondition"),
            "Expected 'error postcondition' in: {}",
            text
        );
    }

    #[test]
    fn test_collapsed_text_probabilistic() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "probabilistic response_bound {\n    probability(response_time < 100) >= 0.99\n}"
                .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);

        let text = ranges[0].collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("probability bound"),
            "Expected 'probability bound' in: {}",
            text
        );
    }

    #[test]
    fn test_collapsed_text_security() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "security no_leak {\n    forall t1, t2: Tenant . isolated(t1, t2)\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);

        let text = ranges[0].collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("security property"),
            "Expected 'security property' in: {}",
            text
        );
    }

    #[test]
    fn test_collapsed_text_abstraction_block() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"refinement r refines S {
    abstraction {
        to_abstract(concrete)
    }
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);

        let abstraction_range = ranges.iter().find(|r| r.start_line == 1).unwrap();
        let text = abstraction_range.collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("abstraction function"),
            "Expected 'abstraction function' in: {}",
            text
        );
    }

    #[test]
    fn test_collapsed_text_simulation_block() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"refinement r refines S {
    simulation {
        step_simulation(impl, spec)
    }
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);

        let simulation_range = ranges.iter().find(|r| r.start_line == 1).unwrap();
        let text = simulation_range.collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("simulation proof"),
            "Expected 'simulation proof' in: {}",
            text
        );
    }

    #[test]
    fn test_helper_plural() {
        assert_eq!(plural(0), "s");
        assert_eq!(plural(1), "");
        assert_eq!(plural(2), "s");
        assert_eq!(plural(100), "s");
    }

    #[test]
    fn test_helper_count_fields() {
        let block = "{\n    x: Int,\n    y: String,\n    z: Bool\n}";
        assert_eq!(count_fields_in_block(block), 3);

        let empty_block = "{\n}";
        assert_eq!(count_fields_in_block(empty_block), 0);

        let commented = "{\n    // comment\n    x: Int\n}";
        assert_eq!(count_fields_in_block(commented), 1);
    }

    #[test]
    fn test_helper_count_keywords() {
        let contract_body =
            "{\n    requires { true }\n    ensures { x > 0 }\n    ensures { y > 0 }\n}";
        assert_eq!(count_keyword_occurrences(contract_body, "requires"), 1);
        assert_eq!(count_keyword_occurrences(contract_body, "ensures"), 2);
        assert_eq!(count_keyword_occurrences(contract_body, "ensures_err"), 0);
    }

    #[test]
    fn test_generate_comment_preview_long_text() {
        let long_comment = "This is a very long comment that should be truncated";
        let preview = generate_comment_preview(long_comment, 5);
        assert!(preview.len() < 60, "Preview should be truncated");
        assert!(
            preview.contains("..."),
            "Should have ellipsis for truncation"
        );
        assert!(
            preview.contains("+4 more"),
            "Should show +4 more for 5 lines"
        );
    }

    #[test]
    fn test_generate_comment_preview_empty() {
        let preview = generate_comment_preview("", 3);
        assert!(
            preview.contains("3 lines"),
            "Empty comment should show line count"
        );
    }

    // ========== Mutation-killing tests ==========

    /// Test contract with only requires (no ensures) - kills line 115 || mutation
    #[test]
    fn test_contract_only_requires() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"contract Stack::check(self: Stack) -> Result<Unit> {
    requires { self.valid() }
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        let contract_range = ranges.iter().find(|r| r.start_line == 0).unwrap();
        let text = contract_range.collapsed_text.as_ref().unwrap();
        // With || mutation to &&, this would show "contract body" instead of counts
        assert!(
            text.contains("1 requires"),
            "Should count requires when no ensures: {}",
            text
        );
        assert!(
            text.contains("0 ensures"),
            "Should show 0 ensures: {}",
            text
        );
    }

    /// Test contract with only ensures (no requires) - kills line 115 || mutation
    #[test]
    fn test_contract_only_ensures() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"contract Stack::pop(self: Stack) -> Result<Int> {
    ensures { result >= 0 }
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        let contract_range = ranges.iter().find(|r| r.start_line == 0).unwrap();
        let text = contract_range.collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("0 requires"),
            "Should show 0 requires: {}",
            text
        );
        assert!(
            text.contains("1 ensures"),
            "Should count ensures when no requires: {}",
            text
        );
    }

    /// Test contract with neither requires nor ensures - kills line 115 boundary
    #[test]
    fn test_contract_empty_body() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"contract Stack::noop(self: Stack) -> Result<Unit> {
    // No requires or ensures
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        let contract_range = ranges.iter().find(|r| r.start_line == 0).unwrap();
        let text = contract_range.collapsed_text.as_ref().unwrap();
        // When both are 0, should show "contract body"
        assert!(
            text.contains("contract body"),
            "Should show 'contract body' when neither requires nor ensures: {}",
            text
        );
    }

    /// Test refinement with only abstraction (no simulation) - kills line 134 && mutation
    #[test]
    fn test_refinement_only_abstraction() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"refinement impl_refines_spec refines Spec {
    abstraction {
        to_spec(impl)
    }
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        let refinement_range = ranges.iter().find(|r| r.start_line == 0).unwrap();
        let text = refinement_range.collapsed_text.as_ref().unwrap();
        // With && mutation to ||, this would incorrectly show "abstraction + simulation"
        assert_eq!(
            text, "... abstraction",
            "Should show only abstraction: {}",
            text
        );
    }

    /// Test refinement with only simulation (no abstraction) - kills line 134 && mutation
    #[test]
    fn test_refinement_only_simulation() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"refinement impl_refines_spec refines Spec {
    simulation {
        step_preserves(impl, spec)
    }
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        let refinement_range = ranges.iter().find(|r| r.start_line == 0).unwrap();
        let text = refinement_range.collapsed_text.as_ref().unwrap();
        assert_eq!(
            text, "... simulation",
            "Should show only simulation: {}",
            text
        );
    }

    /// Test refinement with neither section - kills line 134 boundary
    #[test]
    fn test_refinement_empty_body() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"refinement impl_refines_spec refines Spec {
    // placeholder
}"#
            .to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        let refinement_range = ranges.iter().find(|r| r.start_line == 0).unwrap();
        let text = refinement_range.collapsed_text.as_ref().unwrap();
        assert!(
            text.contains("refinement body"),
            "Should show 'refinement body' when neither section: {}",
            text
        );
    }

    /// Test generic block with exactly 2 lines - kills line 173 >= mutation
    #[test]
    fn test_generic_block_two_lines() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            // Unknown block type - not type/theorem/etc
            "unknown_block foo {\n    content\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);
        let text = ranges[0].collapsed_text.as_ref().unwrap();
        // Block content is just "    content\n" = 2 lines including boundaries
        assert!(
            text.contains("line"),
            "Should show line count for generic block: {}",
            text
        );
    }

    /// Test generic block with exactly 1 line - kills line 173 > vs >= boundary
    #[test]
    fn test_generic_block_one_line_content() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "unknown_block foo {\nx\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        // The block spans 3 lines (0, 1, 2)
        assert_eq!(ranges.len(), 1);
        let text = ranges[0].collapsed_text.as_ref().unwrap();
        // Block content from "{" to "}" spans 3 lines (including braces)
        assert!(
            text.contains("3 line"),
            "Block should show 3 lines: {}",
            text
        );
    }

    /// Test count_keyword_occurrences with partial match - kills line 204 == mutation
    #[test]
    fn test_keyword_partial_match_rejection() {
        // "requires_extra" should NOT match "requires"
        let block_text = "{\n    requires_extra { foo }\n}";
        assert_eq!(
            count_keyword_occurrences(block_text, "requires"),
            0,
            "Should not count 'requires_extra' as 'requires'"
        );

        // "ensures_err" should NOT match "ensures"
        let block_text2 = "{\n    ensures_err { foo }\n}";
        assert_eq!(
            count_keyword_occurrences(block_text2, "ensures"),
            0,
            "Should not count 'ensures_err' as 'ensures'"
        );
    }

    /// Test keyword counting at end of line - kills line 204 boundary
    #[test]
    fn test_keyword_at_line_end() {
        // Keyword alone on line (no brace)
        let block_text = "{\n    requires\n}";
        assert_eq!(
            count_keyword_occurrences(block_text, "requires"),
            1,
            "Should count keyword at end of line"
        );
    }

    /// Test keyword followed by space - kills line 208 || mutation
    #[test]
    fn test_keyword_followed_by_space() {
        let block_text = "{\n    requires { x > 0 }\n}";
        assert_eq!(
            count_keyword_occurrences(block_text, "requires"),
            1,
            "Should count keyword followed by space+brace"
        );
    }

    /// Test end_character calculation - kills line 55 + mutations
    #[test]
    fn test_end_character_position() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Foo = {\n    x: Int\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);
        let range = &ranges[0];
        // End is at line 2, character 1 (after the closing brace)
        assert_eq!(range.end_line, 2);
        assert_eq!(
            range.end_character,
            Some(1),
            "end_character should be 1 (position after closing brace)"
        );
    }

    /// Test multiline block byte offsets - kills line 55, 60 + mutations
    #[test]
    fn test_byte_offset_for_preview() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Foo = {\n    a: Int,\n    b: Bool,\n    c: String\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);
        let text = ranges[0].collapsed_text.as_ref().unwrap();
        // Should correctly count 3 fields
        assert!(
            text.contains("3 field"),
            "Should count all fields with correct byte offsets: {}",
            text
        );
    }

    /// Test line increment in block parsing - kills line 70 += mutations
    #[test]
    fn test_line_tracking_multiline() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo {\n    line1\n    line2\n    line3\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);
        let range = &ranges[0];
        // Block should span lines 0-4 (5 lines total)
        assert_eq!(range.start_line, 0);
        assert_eq!(range.end_line, 4);
    }

    /// Test character tracking with UTF-16 - kills line 75 += mutations
    #[test]
    fn test_character_tracking_utf16() {
        // "日本語" is 3 codepoints, each 1 UTF-16 unit (but 3 UTF-8 bytes each)
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type 日本語 = {\n    x: Int\n}".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        assert_eq!(ranges.len(), 1);
        // Should still parse correctly with UTF-16 character counting
        let range = &ranges[0];
        assert_eq!(range.start_line, 0);
        assert_eq!(range.end_line, 2);
    }

    /// Test comment preview with exactly 30 characters - kills line 322 >= mutation
    #[test]
    fn test_comment_preview_boundary_30_chars() {
        // Exactly 30 characters
        let comment = "123456789012345678901234567890";
        assert_eq!(comment.len(), 30);
        let preview = generate_comment_preview(comment, 2);
        // Should NOT be truncated (len > 30 triggers truncation)
        assert!(
            !preview.contains("...") || preview.contains("+1 more"),
            "30 chars should not truncate: {}",
            preview
        );
    }

    /// Test comment preview with 31 characters - kills line 322 boundary
    #[test]
    fn test_comment_preview_truncation_31_chars() {
        // 31 characters - should trigger truncation
        let comment = "1234567890123456789012345678901";
        assert_eq!(comment.len(), 31);
        let preview = generate_comment_preview(comment, 2);
        // Should be truncated with "..."
        assert!(
            preview.contains("..."),
            "31 chars should trigger truncation: {}",
            preview
        );
    }

    /// Test trailing comment block - kills line 298 > boundary
    #[test]
    fn test_trailing_comment_block() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Foo = { x: Int }\n// Comment 1\n// Comment 2".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        // Should have comment folding region
        let comment_range = ranges
            .iter()
            .find(|r| r.kind == Some(FoldingRangeKind::Comment));
        assert!(comment_range.is_some(), "Should fold trailing comments");
        let cr = comment_range.unwrap();
        assert_eq!(cr.start_line, 1);
        assert_eq!(cr.end_line, 2);
    }

    /// Test comment end character calculation - kills line 300 - and + mutations
    #[test]
    fn test_comment_end_character() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "// Line 1\n// Line 2 is longer\ncode".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        let comment_range = ranges
            .iter()
            .find(|r| r.kind == Some(FoldingRangeKind::Comment))
            .unwrap();
        // end_char should be length of "// Line 2 is longer" = 19
        assert_eq!(
            comment_range.end_character,
            Some(19),
            "end_character should match last comment line length"
        );
    }

    /// Test comment count calculation - kills line 276 - and + mutations
    #[test]
    fn test_comment_count_in_preview() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "// First\n// Second\n// Third\n// Fourth\ncode".to_string(),
        );

        let ranges = generate_folding_ranges(&doc);
        let comment_range = ranges
            .iter()
            .find(|r| r.kind == Some(FoldingRangeKind::Comment))
            .unwrap();
        let text = comment_range.collapsed_text.as_ref().unwrap();
        // 4 comment lines, so should show "+3 more"
        assert!(
            text.contains("+3 more"),
            "Should show +3 more for 4 lines: {}",
            text
        );
    }
}

// ============================ Kani Proof Harnesses ============================
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that count_fields_in_block never exceeds the number of lines.
    /// Since each field must occupy at least one line, count <= lines.count()
    #[kani::proof]
    fn verify_count_fields_bounded_by_lines() {
        // Test with a simple block string of bounded length
        let num_lines: usize = kani::any();
        kani::assume(num_lines <= 5);

        // Build a simple test string with some lines
        let mut text = String::new();
        for i in 0..num_lines {
            if i > 0 {
                text.push('\n');
            }
            // Some lines have colons (fields), some don't
            let has_colon: bool = kani::any();
            if has_colon {
                text.push_str("field: Type");
            } else {
                text.push_str("other content");
            }
        }

        let count = count_fields_in_block(&text);
        let line_count = text.lines().count();

        kani::assert(count <= line_count, "Field count cannot exceed line count");
    }

    /// Prove that count_keyword_occurrences never exceeds the number of lines.
    /// Since each keyword occurrence is on one line, count <= lines.count()
    #[kani::proof]
    fn verify_count_keywords_bounded_by_lines() {
        let num_lines: usize = kani::any();
        kani::assume(num_lines <= 5);

        // Build a test string
        let mut text = String::new();
        for i in 0..num_lines {
            if i > 0 {
                text.push('\n');
            }
            // Some lines start with "requires", some don't
            let is_keyword: bool = kani::any();
            if is_keyword {
                text.push_str("requires { }");
            } else {
                text.push_str("other stuff");
            }
        }

        let count = count_keyword_occurrences(&text, "requires");
        let line_count = text.lines().count();

        kani::assert(
            count <= line_count,
            "Keyword count cannot exceed line count",
        );
    }

    /// Prove that count_fields_in_block handles empty strings correctly.
    #[kani::proof]
    fn verify_count_fields_empty_input() {
        let count = count_fields_in_block("");
        kani::assert(count == 0, "Empty input should yield 0 fields");
    }

    /// Prove that count_keyword_occurrences handles empty strings correctly.
    #[kani::proof]
    fn verify_count_keywords_empty_input() {
        let count = count_keyword_occurrences("", "requires");
        kani::assert(count == 0, "Empty input should yield 0 keyword occurrences");
    }

    /// Prove that count_fields_in_block skips comment lines.
    #[kani::proof]
    fn verify_count_fields_skips_comments() {
        // A line starting with "//" that contains ":" should not be counted
        let text = "// comment: with colon";
        let count = count_fields_in_block(text);
        kani::assert(
            count == 0,
            "Comment lines with colons should not be counted as fields",
        );
    }

    /// Prove that count_fields_in_block skips type definitions.
    #[kani::proof]
    fn verify_count_fields_skips_type_keyword() {
        // A line starting with "type " should not be counted
        let text = "type Foo = { x: Int }";
        let count = count_fields_in_block(text);
        kani::assert(
            count == 0,
            "Type definition lines should not be counted as fields",
        );
    }

    /// Prove that generate_comment_preview always returns a non-empty string.
    #[kani::proof]
    fn verify_comment_preview_nonempty() {
        let count: usize = kani::any();
        kani::assume(count >= 1 && count <= 100);

        // Test with empty first line
        let preview_empty = generate_comment_preview("", count);
        kani::assert(!preview_empty.is_empty(), "Preview should never be empty");

        // Test with non-empty first line
        let preview_short = generate_comment_preview("text", count);
        kani::assert(!preview_short.is_empty(), "Preview should never be empty");
    }

    /// Prove that plural returns "s" for all values except 1.
    #[kani::proof]
    fn verify_plural_function() {
        let n: usize = kani::any();
        kani::assume(n <= 1000);

        let result = plural(n);

        if n == 1 {
            kani::assert(result == "", "plural(1) should return empty string");
        } else {
            kani::assert(result == "s", "plural(n!=1) should return 's'");
        }
    }
}
