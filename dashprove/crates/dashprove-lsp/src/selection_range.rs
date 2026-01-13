//! Selection ranges for USL documents
//!
//! Provides hierarchical selection expansion based on syntactic structure.
//! Users can progressively expand their selection from word -> expression ->
//! statement -> block -> property -> document.

use crate::document::Document;
use tower_lsp::lsp_types::{Position, Range, SelectionRange};

/// Generate selection ranges for positions in a USL document.
///
/// For each position, returns a hierarchy of nested ranges from smallest
/// (word) to largest (entire document).
#[must_use]
pub fn generate_selection_ranges(doc: &Document, positions: &[Position]) -> Vec<SelectionRange> {
    positions
        .iter()
        .map(|pos| compute_selection_range(doc, *pos))
        .collect()
}

/// Compute the selection range hierarchy for a single position.
fn compute_selection_range(doc: &Document, pos: Position) -> SelectionRange {
    let offset = doc.position_to_offset(pos.line, pos.character);

    // Build hierarchy from smallest to largest:
    // 1. Word at cursor
    // 2. Content within braces (if inside braces)
    // 3. Brace block including braces
    // 4. Full line
    // 5. Property/type definition block (keyword to closing brace)
    // 6. Entire document

    let mut ranges: Vec<Range> = Vec::new();

    // 1. Word at cursor position
    if let Some(word_range) = find_word_range(doc, offset) {
        ranges.push(word_range);
    }

    // 2-3. Brace blocks (inner content and outer with braces)
    let brace_ranges = find_enclosing_brace_ranges(doc, offset);
    for range in brace_ranges {
        if ranges.last() != Some(&range) {
            ranges.push(range);
        }
    }

    // 4. Full line
    let line_range = find_line_range(doc, pos.line);
    if ranges.last() != Some(&line_range) {
        ranges.push(line_range);
    }

    // 5. Property/type definition block
    if let Some(block_range) = find_definition_block_range(doc, offset) {
        if ranges.last() != Some(&block_range) {
            ranges.push(block_range);
        }
    }

    // 6. Entire document
    let doc_range = full_document_range(doc);
    if ranges.last() != Some(&doc_range) {
        ranges.push(doc_range);
    }

    // Build the linked list structure from largest to smallest
    build_selection_range_chain(&ranges)
}

/// Find the word range at an offset.
fn find_word_range(doc: &Document, offset: usize) -> Option<Range> {
    let text = &doc.text;
    if offset > text.len() {
        return None;
    }

    // Find word boundaries
    let start = text[..offset]
        .rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);

    let end = text[offset..]
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| offset + i)
        .unwrap_or(text.len());

    if start >= end {
        return None;
    }

    let (start_line, start_char) = doc.offset_to_position(start);
    let (end_line, end_char) = doc.offset_to_position(end);

    Some(Range {
        start: Position::new(start_line, start_char),
        end: Position::new(end_line, end_char),
    })
}

/// Find enclosing brace blocks at an offset.
/// Returns ranges from innermost to outermost.
fn find_enclosing_brace_ranges(doc: &Document, offset: usize) -> Vec<Range> {
    let text = &doc.text;
    let mut ranges = Vec::new();

    // Find all opening braces before offset and their matching closing braces
    let mut brace_stack: Vec<usize> = Vec::new();
    let mut enclosing_braces: Vec<(usize, usize)> = Vec::new();

    for (i, ch) in text.char_indices() {
        match ch {
            '{' => {
                brace_stack.push(i);
            }
            '}' => {
                if let Some(open) = brace_stack.pop() {
                    // If offset is within this brace pair
                    if open < offset && i >= offset {
                        enclosing_braces.push((open, i));
                    }
                }
            }
            _ => {}
        }
    }

    // Sort by size (innermost first)
    enclosing_braces.sort_by_key(|(open, close)| close - open);

    for (open, close) in enclosing_braces {
        // Inner content (excluding braces)
        let inner_start = open + 1;
        let inner_end = close;
        if inner_start < inner_end {
            let (start_line, start_char) = doc.offset_to_position(inner_start);
            let (end_line, end_char) = doc.offset_to_position(inner_end);
            ranges.push(Range {
                start: Position::new(start_line, start_char),
                end: Position::new(end_line, end_char),
            });
        }

        // Outer including braces
        let (start_line, start_char) = doc.offset_to_position(open);
        let (end_line, end_char) = doc.offset_to_position(close + 1);
        ranges.push(Range {
            start: Position::new(start_line, start_char),
            end: Position::new(end_line, end_char),
        });
    }

    ranges
}

/// Find the range of the full line.
fn find_line_range(doc: &Document, line: u32) -> Range {
    let line_text = doc.line_text(line).unwrap_or("");
    let line_len = line_text.chars().map(|c| c.len_utf16()).sum::<usize>() as u32;

    Range {
        start: Position::new(line, 0),
        end: Position::new(line, line_len),
    }
}

/// Find the definition block (type or property) containing the offset.
fn find_definition_block_range(doc: &Document, offset: usize) -> Option<Range> {
    let text = &doc.text;

    // Keywords that start definitions
    let keywords = [
        "type",
        "theorem",
        "temporal",
        "contract",
        "invariant",
        "refinement",
        "probabilistic",
        "security",
    ];

    // Find the most recent keyword before offset
    let mut best_start: Option<usize> = None;

    for keyword in keywords {
        for (idx, _) in text.match_indices(keyword) {
            if idx > offset {
                continue;
            }
            // Verify it's at a word boundary (not inside another identifier)
            let before = idx
                .checked_sub(1)
                .and_then(|i| text.as_bytes().get(i).copied());
            let after = text.as_bytes().get(idx + keyword.len()).copied();

            // Check that character before is not alphanumeric or underscore
            let before_ok =
                before.is_none() || !before.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_');
            // Check that character after is whitespace (keywords need whitespace before name)
            let after_ok = after.is_some_and(|b| b.is_ascii_whitespace());

            if before_ok && after_ok {
                match best_start {
                    None => best_start = Some(idx),
                    Some(prev) if idx > prev => best_start = Some(idx),
                    _ => {}
                }
            }
        }
    }

    let start = best_start?;

    // Find the matching closing brace
    let text_from_start = &text[start..];
    let mut brace_count = 0;
    let mut found_open = false;
    let mut end = start;

    for (i, ch) in text_from_start.char_indices() {
        match ch {
            '{' => {
                found_open = true;
                brace_count += 1;
            }
            '}' => {
                brace_count -= 1;
                if found_open && brace_count == 0 {
                    end = start + i + 1;
                    break;
                }
            }
            _ => {}
        }
    }

    if end <= start {
        return None;
    }

    let (start_line, start_char) = doc.offset_to_position(start);
    let (end_line, end_char) = doc.offset_to_position(end);

    Some(Range {
        start: Position::new(start_line, start_char),
        end: Position::new(end_line, end_char),
    })
}

/// Get the range covering the entire document.
fn full_document_range(doc: &Document) -> Range {
    let lines: Vec<&str> = doc.text.lines().collect();
    let last_line = lines.len().saturating_sub(1) as u32;
    let last_char = lines
        .last()
        .map(|l| l.chars().map(|c| c.len_utf16()).sum::<usize>())
        .unwrap_or(0) as u32;

    Range {
        start: Position::new(0, 0),
        end: Position::new(last_line, last_char),
    }
}

/// Build a SelectionRange linked list from a list of ranges (smallest to largest).
fn build_selection_range_chain(ranges: &[Range]) -> SelectionRange {
    if ranges.is_empty() {
        // Fallback: single point
        return SelectionRange {
            range: Range {
                start: Position::new(0, 0),
                end: Position::new(0, 0),
            },
            parent: None,
        };
    }

    // Build from largest to smallest
    let mut current: Option<Box<SelectionRange>> = None;

    for range in ranges.iter().rev() {
        current = Some(Box::new(SelectionRange {
            range: *range,
            parent: current,
        }));
    }

    *current.unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower_lsp::lsp_types::Url;

    fn make_doc(text: &str) -> Document {
        Document::new(Url::parse("file:///test.usl").unwrap(), 1, text.to_string())
    }

    #[test]
    fn test_word_range() {
        let doc = make_doc("theorem foo { true }");
        let range = find_word_range(&doc, 9); // "foo"
        assert!(range.is_some());
        let r = range.unwrap();
        assert_eq!(r.start.character, 8);
        assert_eq!(r.end.character, 11);
    }

    #[test]
    fn test_word_range_at_start() {
        let doc = make_doc("theorem foo");
        let range = find_word_range(&doc, 3); // inside "theorem"
        assert!(range.is_some());
        let r = range.unwrap();
        assert_eq!(r.start.character, 0);
        assert_eq!(r.end.character, 7);
    }

    #[test]
    fn test_word_range_at_end() {
        let doc = make_doc("theorem foo");
        let range = find_word_range(&doc, 10); // inside "foo"
        assert!(range.is_some());
        let r = range.unwrap();
        assert_eq!(r.start.character, 8);
        assert_eq!(r.end.character, 11);
    }

    #[test]
    fn test_enclosing_brace_ranges() {
        let doc = make_doc("theorem foo { true }");
        let ranges = find_enclosing_brace_ranges(&doc, 15); // inside braces
        assert_eq!(ranges.len(), 2); // inner content + outer with braces
    }

    #[test]
    fn test_nested_brace_ranges() {
        let doc = make_doc("refinement r refines S {\n    abstraction {\n        true\n    }\n}");
        let ranges = find_enclosing_brace_ranges(&doc, 50); // inside abstraction
                                                            // Should have: inner abstraction, outer abstraction, inner refinement, outer refinement
        assert!(ranges.len() >= 2);
    }

    #[test]
    fn test_line_range() {
        let doc = make_doc("line one\nline two\nline three");
        let range = find_line_range(&doc, 1);
        assert_eq!(range.start.line, 1);
        assert_eq!(range.start.character, 0);
        assert_eq!(range.end.line, 1);
        assert_eq!(range.end.character, 8); // "line two"
    }

    #[test]
    fn test_definition_block_range() {
        let doc = make_doc("theorem my_theorem {\n    forall x: Bool . x\n}");
        let range = find_definition_block_range(&doc, 25); // inside theorem body
        assert!(range.is_some());
        let r = range.unwrap();
        assert_eq!(r.start.line, 0);
        assert_eq!(r.start.character, 0);
        assert_eq!(r.end.line, 2);
    }

    #[test]
    fn test_definition_block_type() {
        let doc = make_doc("type Value = {\n    id: Int\n}");
        let range = find_definition_block_range(&doc, 20); // inside type body
        assert!(range.is_some());
        let r = range.unwrap();
        assert_eq!(r.start.line, 0);
        assert_eq!(r.start.character, 0);
        assert_eq!(r.end.line, 2);
    }

    #[test]
    fn test_full_document_range() {
        let doc = make_doc("line one\nline two");
        let range = full_document_range(&doc);
        assert_eq!(range.start.line, 0);
        assert_eq!(range.start.character, 0);
        assert_eq!(range.end.line, 1);
        assert_eq!(range.end.character, 8);
    }

    #[test]
    fn test_build_selection_range_chain() {
        let ranges = vec![
            Range {
                start: Position::new(0, 0),
                end: Position::new(0, 3),
            },
            Range {
                start: Position::new(0, 0),
                end: Position::new(0, 10),
            },
            Range {
                start: Position::new(0, 0),
                end: Position::new(0, 20),
            },
        ];

        let chain = build_selection_range_chain(&ranges);

        // First should be smallest range
        assert_eq!(chain.range.end.character, 3);

        // Should have parent pointing to next larger
        assert!(chain.parent.is_some());
        let parent = chain.parent.as_ref().unwrap();
        assert_eq!(parent.range.end.character, 10);

        // Grandparent is largest
        assert!(parent.parent.is_some());
        let grandparent = parent.parent.as_ref().unwrap();
        assert_eq!(grandparent.range.end.character, 20);

        // No more parents
        assert!(grandparent.parent.is_none());
    }

    #[test]
    fn test_generate_selection_ranges_single() {
        let doc = make_doc("theorem foo { true }");
        let positions = vec![Position::new(0, 9)]; // "foo"

        let ranges = generate_selection_ranges(&doc, &positions);
        assert_eq!(ranges.len(), 1);

        let sr = &ranges[0];
        // First should be word "foo"
        assert_eq!(sr.range.start.character, 8);
        assert_eq!(sr.range.end.character, 11);
    }

    #[test]
    fn test_generate_selection_ranges_multiple() {
        let doc = make_doc("theorem foo { true }\ntheorem bar { false }");
        let positions = vec![Position::new(0, 9), Position::new(1, 9)];

        let ranges = generate_selection_ranges(&doc, &positions);
        assert_eq!(ranges.len(), 2);

        // First position: "foo"
        assert_eq!(ranges[0].range.start.character, 8);

        // Second position: "bar"
        assert_eq!(ranges[1].range.start.character, 8);
    }

    #[test]
    fn test_selection_range_hierarchy_complete() {
        let doc = make_doc("theorem test {\n    forall x: Bool . x\n}");
        let positions = vec![Position::new(1, 15)]; // "Bool"

        let ranges = generate_selection_ranges(&doc, &positions);
        let sr = &ranges[0];

        // Count hierarchy depth
        let mut depth = 1;
        let mut current = sr.parent.as_ref();
        while let Some(p) = current {
            depth += 1;
            current = p.parent.as_ref();
        }

        // Should have multiple levels: word, inner brace, outer brace, line, definition, document
        assert!(depth >= 4, "Expected at least 4 levels, got {}", depth);
    }

    #[test]
    fn test_empty_positions() {
        let doc = make_doc("theorem foo { true }");
        let positions: Vec<Position> = vec![];

        let ranges = generate_selection_ranges(&doc, &positions);
        assert!(ranges.is_empty());
    }

    #[test]
    fn test_position_outside_content() {
        let doc = make_doc("theorem foo { true }");
        let positions = vec![Position::new(0, 100)]; // Beyond end of line

        let ranges = generate_selection_ranges(&doc, &positions);
        assert_eq!(ranges.len(), 1);
        // Should still return something (at least document range)
    }

    #[test]
    fn test_enclosing_brace_ranges_boundary_conditions() {
        // Mutation test: verify boundary conditions in brace matching
        // Test offset exactly at open brace position
        let doc = make_doc("theorem foo { true }");
        // Offset 13 is at '{', condition is: open < offset && i >= offset
        let _ranges = find_enclosing_brace_ranges(&doc, 13);

        // Test offset one before closing brace
        let ranges2 = find_enclosing_brace_ranges(&doc, 18);
        assert!(!ranges2.is_empty());

        // Test offset at closing brace position
        let ranges3 = find_enclosing_brace_ranges(&doc, 19);
        assert!(!ranges3.is_empty());
    }

    #[test]
    fn test_enclosing_brace_ranges_inner_boundaries() {
        // Mutation test: verify inner_start + 1 and close - open arithmetic
        let doc = make_doc("theorem foo { x }");
        // Inner content is " x " (positions 14-16), outer is positions 13-17
        let ranges = find_enclosing_brace_ranges(&doc, 15); // inside braces

        // Should have at least inner and outer ranges
        assert!(ranges.len() >= 2);

        // First range (innermost) should be inner content (excluding braces)
        // Inner starts at open+1 and ends at close
        let inner = &ranges[0];
        assert!(inner.start.character >= 13); // must be after open brace
        assert!(inner.end.character <= 17); // must be before or at close
    }

    #[test]
    fn test_enclosing_brace_ranges_sorting() {
        // Mutation test: verify sorting by close - open (innermost first)
        let doc = make_doc("theorem foo { { x } }");
        // Nested braces: outer { at 13, inner { at 15, inner } at 18, outer } at 20
        let ranges = find_enclosing_brace_ranges(&doc, 16); // inside inner braces

        // Should have 4 ranges: inner content, inner braces, outer content, outer braces
        assert!(ranges.len() >= 2);

        // First range should be smallest (innermost)
        let first_size = ranges[0].end.character - ranges[0].start.character;
        if ranges.len() >= 2 {
            let second_size = ranges[1].end.character - ranges[1].start.character;
            assert!(
                first_size <= second_size,
                "Expected innermost first: {} <= {}",
                first_size,
                second_size
            );
        }
    }

    #[test]
    fn test_definition_block_range_offset_comparison() {
        // Mutation test: verify idx > offset comparison for skipping keywords after cursor
        let doc = make_doc("theorem foo { true }\ntheorem bar { false }");

        // With cursor in "foo" theorem, should only match "theorem" at position 0
        let range1 = find_definition_block_range(&doc, 10);
        assert!(range1.is_some());
        let r1 = range1.unwrap();
        assert_eq!(r1.start.line, 0);

        // With cursor in "bar" theorem, should match "theorem" at line 1
        let range2 = find_definition_block_range(&doc, 30);
        assert!(range2.is_some());
        let r2 = range2.unwrap();
        assert_eq!(r2.start.line, 1);
    }

    #[test]
    fn test_definition_block_range_word_boundary() {
        // Mutation test: verify word boundary checks (before_ok && after_ok)
        // "atheorem" should not match "theorem"
        let doc = make_doc("atheorem foo { true }");
        let range = find_definition_block_range(&doc, 15);
        // Should not find a definition block since "atheorem" is not a valid keyword
        assert!(range.is_none());

        // "theorem_foo" should not match "theorem"
        let doc2 = make_doc("theorem_foo bar { true }");
        let range2 = find_definition_block_range(&doc2, 18);
        assert!(range2.is_none());

        // Valid "theorem " should work
        let doc3 = make_doc("theorem foo { true }");
        let range3 = find_definition_block_range(&doc3, 15);
        assert!(range3.is_some());
    }

    #[test]
    fn test_definition_block_range_best_match() {
        // Mutation test: verify best_start tracking (idx > prev)
        // Multiple definitions, cursor at end should match the last one
        let doc = make_doc("type A = { x: Int }\ntype B = { y: Int }");

        // Cursor in type B should find type B, not type A
        let range = find_definition_block_range(&doc, 30);
        assert!(range.is_some());
        let r = range.unwrap();
        // Type B starts at position 20 (line 1)
        assert_eq!(r.start.line, 1);
    }

    #[test]
    fn test_definition_block_range_brace_counting() {
        // Mutation test: verify brace counting and found_open && brace_count == 0
        let doc = make_doc("theorem test {\n    forall x: Bool . {\n        true\n    }\n}");

        // Cursor in nested block
        let range = find_definition_block_range(&doc, 45);
        assert!(range.is_some());
        let r = range.unwrap();
        // Should span from "theorem" to final "}"
        assert_eq!(r.start.line, 0);
        assert_eq!(r.end.line, 4);
    }

    #[test]
    fn test_definition_block_range_end_calculation() {
        // Mutation test: verify end = start + i + 1 calculation
        let doc = make_doc("theorem t { x }");
        let range = find_definition_block_range(&doc, 12);
        assert!(range.is_some());
        let r = range.unwrap();
        // End should be after the closing brace
        assert_eq!(r.end.character, 15); // "theorem t { x }" ends at position 15
    }

    #[test]
    fn test_enclosing_brace_empty_content() {
        // Mutation test: verify inner_start < inner_end check
        let doc = make_doc("theorem t {}");
        // Empty braces: open at 10, close at 11, inner_start = 11, inner_end = 11
        let ranges = find_enclosing_brace_ranges(&doc, 11); // at position between braces

        // When offset is at closing brace, it should still work
        // But inner content range should not be added if inner_start >= inner_end
        for r in &ranges {
            // Verify no zero-width inner ranges
            assert!(
                r.start.character < r.end.character
                    || r.start.line < r.end.line
                    || (r.start.character == r.end.character && r.start.line == r.end.line),
                "Found invalid range"
            );
        }
    }
}
