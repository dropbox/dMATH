//! Semantic token support for USL syntax highlighting
//!
//! This module provides semantic token generation for USL documents,
//! enabling rich syntax highlighting in IDEs that support LSP.

use crate::document::Document;
use tower_lsp::lsp_types::{
    Range, SemanticToken, SemanticTokenModifier, SemanticTokenType, SemanticTokens,
    SemanticTokensLegend,
};

/// Standard semantic token types for USL.
pub const TOKEN_TYPES: &[SemanticTokenType] = &[
    SemanticTokenType::KEYWORD,   // 0: type, theorem, invariant, forall, etc.
    SemanticTokenType::TYPE,      // 1: Int, Bool, Set, List, user-defined types
    SemanticTokenType::FUNCTION,  // 2: property names (theorems, contracts, etc.)
    SemanticTokenType::VARIABLE,  // 3: quantified variables
    SemanticTokenType::PROPERTY,  // 4: field access
    SemanticTokenType::OPERATOR,  // 5: ==, ., ->, etc.
    SemanticTokenType::COMMENT,   // 6: // comments
    SemanticTokenType::NUMBER,    // 7: numeric literals
    SemanticTokenType::STRING,    // 8: string literals
    SemanticTokenType::NAMESPACE, // 9: type paths like Type::method
    SemanticTokenType::PARAMETER, // 10: function parameters
];

/// Standard semantic token modifiers for USL.
pub const TOKEN_MODIFIERS: &[SemanticTokenModifier] = &[
    SemanticTokenModifier::DECLARATION, // 0: where identifier is declared
    SemanticTokenModifier::DEFINITION,  // 1: where identifier is defined
    SemanticTokenModifier::READONLY,    // 2: immutable/constant
    SemanticTokenModifier::DEFAULT_LIBRARY, // 3: builtin types
];

/// Get the semantic tokens legend for capability advertisement.
#[must_use]
pub fn semantic_tokens_legend() -> SemanticTokensLegend {
    SemanticTokensLegend {
        token_types: TOKEN_TYPES.to_vec(),
        token_modifiers: TOKEN_MODIFIERS.to_vec(),
    }
}

/// USL keywords that should be highlighted.
const KEYWORDS: &[&str] = &[
    "type",
    "theorem",
    "temporal",
    "contract",
    "invariant",
    "refinement",
    "probabilistic",
    "security",
    "forall",
    "exists",
    "implies",
    "and",
    "or",
    "not",
    "always",
    "eventually",
    "requires",
    "ensures",
    "ensures_err",
    "abstraction",
    "simulation",
    "probability",
    "true",
    "false",
    "refines",
    "in",
];

/// USL builtin types.
const BUILTIN_TYPES: &[&str] = &[
    "Bool", "Int", "Float", "String", "Unit", "Set", "List", "Map", "Relation", "Result",
];

/// A raw token before delta encoding.
#[derive(Debug, Clone)]
struct RawToken {
    line: u32,
    start_char: u32,
    length: u32,
    token_type: u32,
    token_modifiers: u32,
}

/// Generate semantic tokens for a USL document.
#[must_use]
pub fn generate_semantic_tokens(doc: &Document) -> SemanticTokens {
    let mut raw_tokens = Vec::new();

    // Tokenize the document
    tokenize_document(doc, &mut raw_tokens, None);

    // Sort tokens by position (line, then character)
    raw_tokens.sort_by(|a, b| {
        a.line
            .cmp(&b.line)
            .then_with(|| a.start_char.cmp(&b.start_char))
    });

    // Convert to delta-encoded LSP format
    let tokens = delta_encode(&raw_tokens);

    SemanticTokens {
        result_id: None,
        data: tokens,
    }
}

/// Generate semantic tokens restricted to a specific range.
#[must_use]
pub fn generate_semantic_tokens_in_range(doc: &Document, range: Range) -> SemanticTokens {
    let mut raw_tokens = Vec::new();

    tokenize_document(doc, &mut raw_tokens, Some(&range));

    raw_tokens.sort_by(|a, b| {
        a.line
            .cmp(&b.line)
            .then_with(|| a.start_char.cmp(&b.start_char))
    });

    let tokens = delta_encode(&raw_tokens);

    SemanticTokens {
        result_id: None,
        data: tokens,
    }
}

/// Tokenize the entire document.
fn tokenize_document(doc: &Document, tokens: &mut Vec<RawToken>, range: Option<&Range>) {
    let text = &doc.text;
    let mut chars = text.char_indices().peekable();
    let mut current_line = 0u32;
    let mut line_start = 0usize;

    // Get user-defined types for recognition
    let user_types: Vec<String> = doc
        .spec
        .as_ref()
        .map(|s| s.types.iter().map(|t| t.name.clone()).collect())
        .unwrap_or_default();
    let user_type_refs: Vec<&str> = user_types.iter().map(String::as_str).collect();

    // Get property names for recognition
    let property_names: Vec<String> = doc
        .spec
        .as_ref()
        .map(|s| s.properties.iter().map(|p| p.name()).collect())
        .unwrap_or_default();
    let property_name_refs: Vec<&str> = property_names.iter().map(String::as_str).collect();

    while let Some((i, ch)) = chars.next() {
        // Track line positions
        if ch == '\n' {
            current_line += 1;
            line_start = i + 1;
            continue;
        }

        // Skip whitespace
        if ch.is_whitespace() {
            continue;
        }

        // Comments
        if ch == '/' && chars.peek().is_some_and(|(_, c)| *c == '/') {
            let comment_start = i;
            // Consume until end of line
            for (_, c) in chars.by_ref() {
                if c == '\n' {
                    break;
                }
            }
            let comment_len = text[comment_start..]
                .find('\n')
                .unwrap_or(text.len() - comment_start);
            let token = RawToken {
                line: current_line,
                start_char: (comment_start - line_start) as u32,
                length: comment_len as u32,
                token_type: 6, // COMMENT
                token_modifiers: 0,
            };
            if range.is_none_or(|r| token_overlaps_range(&token, r)) {
                tokens.push(token);
            }
            current_line += 1;
            line_start = comment_start + comment_len + 1;
            continue;
        }

        // Identifiers and keywords
        if ch.is_alphabetic() || ch == '_' {
            let word_start = i;
            let mut word_end = i + ch.len_utf8();
            while let Some(&(next_i, next_ch)) = chars.peek() {
                if next_ch.is_alphanumeric() || next_ch == '_' {
                    chars.next();
                    word_end = next_i + next_ch.len_utf8();
                } else {
                    break;
                }
            }

            let word = &text[word_start..word_end];
            let start_char = (word_start - line_start) as u32;
            let length = word.len() as u32;

            // Determine token type
            let (token_type, token_modifiers) =
                classify_identifier(word, &user_type_refs, &property_name_refs);

            let token = RawToken {
                line: current_line,
                start_char,
                length,
                token_type,
                token_modifiers,
            };
            if range.is_none_or(|r| token_overlaps_range(&token, r)) {
                tokens.push(token);
            }
            continue;
        }

        // Numbers
        if ch.is_ascii_digit() {
            let num_start = i;
            let mut num_end = i + 1;
            while let Some(&(next_i, next_ch)) = chars.peek() {
                if next_ch.is_ascii_digit() || next_ch == '.' || next_ch == '_' {
                    chars.next();
                    num_end = next_i + 1;
                } else {
                    break;
                }
            }
            let token = RawToken {
                line: current_line,
                start_char: (num_start - line_start) as u32,
                length: (num_end - num_start) as u32,
                token_type: 7, // NUMBER
                token_modifiers: 0,
            };
            if range.is_none_or(|r| token_overlaps_range(&token, r)) {
                tokens.push(token);
            }
            continue;
        }

        // String literals
        if ch == '"' {
            let str_start = i;
            let mut str_end = i + 1;
            let mut escaped = false;
            for (next_i, next_ch) in chars.by_ref() {
                str_end = next_i + next_ch.len_utf8();
                if escaped {
                    escaped = false;
                } else if next_ch == '\\' {
                    escaped = true;
                } else if next_ch == '"' {
                    break;
                } else if next_ch == '\n' {
                    // Unterminated string
                    break;
                }
            }
            let token = RawToken {
                line: current_line,
                start_char: (str_start - line_start) as u32,
                length: (str_end - str_start) as u32,
                token_type: 8, // STRING
                token_modifiers: 0,
            };
            if range.is_none_or(|r| token_overlaps_range(&token, r)) {
                tokens.push(token);
            }
            continue;
        }

        // Operators (multi-char)
        if let Some(op_len) = classify_operator(text, i) {
            let token = RawToken {
                line: current_line,
                start_char: (i - line_start) as u32,
                length: op_len as u32,
                token_type: 5, // OPERATOR
                token_modifiers: 0,
            };
            if range.is_none_or(|r| token_overlaps_range(&token, r)) {
                tokens.push(token);
            }
            // Skip the remaining characters of the operator
            for _ in 1..op_len {
                chars.next();
            }
            continue;
        }
    }
}

/// Classify an identifier as keyword, type, function, or variable.
fn classify_identifier(word: &str, user_types: &[&str], property_names: &[&str]) -> (u32, u32) {
    // Keywords
    if KEYWORDS.contains(&word) {
        return (0, 0); // KEYWORD
    }

    // Builtin types
    if BUILTIN_TYPES.contains(&word) {
        return (1, 8); // TYPE with DEFAULT_LIBRARY modifier
    }

    // User-defined types
    if user_types.contains(&word) {
        return (1, 1); // TYPE with DEFINITION modifier
    }

    // Property names (theorems, contracts, etc.)
    if property_names.contains(&word) {
        return (2, 1); // FUNCTION with DEFINITION modifier
    }

    // Default to variable (for quantified variables, etc.)
    (3, 0) // VARIABLE
}

/// Check if position starts a multi-character operator.
fn classify_operator(text: &str, pos: usize) -> Option<usize> {
    let remaining = &text[pos..];

    // Two-character operators
    if remaining.starts_with("==")
        || remaining.starts_with("!=")
        || remaining.starts_with("<=")
        || remaining.starts_with(">=")
        || remaining.starts_with("->")
        || remaining.starts_with("::")
    {
        return Some(2);
    }

    // Single-character operators
    let ch = remaining.chars().next()?;
    if matches!(
        ch,
        '=' | '<' | '>' | '+' | '-' | '*' | '/' | '%' | '.' | ':' | '|' | '&' | '!'
    ) {
        return Some(1);
    }

    None
}

/// Check whether a raw token overlaps the provided range.
fn token_overlaps_range(token: &RawToken, range: &Range) -> bool {
    if token.line < range.start.line || token.line > range.end.line {
        return false;
    }

    let token_start = token.start_char;
    let token_end = token.start_char + token.length;

    if token.line == range.start.line && token_end <= range.start.character {
        return false;
    }
    if token.line == range.end.line && token_start >= range.end.character {
        return false;
    }

    true
}

/// Delta-encode raw tokens into LSP format.
fn delta_encode(raw_tokens: &[RawToken]) -> Vec<SemanticToken> {
    let mut result = Vec::with_capacity(raw_tokens.len());
    let mut prev_line = 0u32;
    let mut prev_char = 0u32;

    for token in raw_tokens {
        let delta_line = token.line - prev_line;
        let delta_start = if delta_line == 0 {
            token.start_char - prev_char
        } else {
            token.start_char
        };

        result.push(SemanticToken {
            delta_line,
            delta_start,
            length: token.length,
            token_type: token.token_type,
            token_modifiers_bitset: token.token_modifiers,
        });

        prev_line = token.line;
        prev_char = token.start_char;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower_lsp::lsp_types::{Position, Range, Url};

    #[test]
    fn test_semantic_tokens_legend() {
        let legend = semantic_tokens_legend();
        assert!(!legend.token_types.is_empty());
        assert!(legend.token_types.contains(&SemanticTokenType::KEYWORD));
        assert!(legend.token_types.contains(&SemanticTokenType::TYPE));
        assert!(legend.token_types.contains(&SemanticTokenType::COMMENT));
    }

    #[test]
    fn test_generate_tokens_simple() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { true }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);
        // Should have tokens for: theorem, foo, {, true, }
        assert!(!tokens.data.is_empty());
    }

    #[test]
    fn test_generate_tokens_with_types() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Value = { id: Int }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);
        assert!(!tokens.data.is_empty());

        // Find the Int token (should be builtin type)
        let int_token = tokens.data.iter().find(|t| t.length == 3);
        assert!(int_token.is_some());
    }

    #[test]
    fn test_generate_tokens_with_comments() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "// This is a comment\ntheorem foo { true }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // First token should be the comment
        let first = &tokens.data[0];
        assert_eq!(first.token_type, 6); // COMMENT
        assert_eq!(first.delta_line, 0);
        assert_eq!(first.delta_start, 0);
    }

    #[test]
    fn test_generate_tokens_multiline() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Value = {\n    id: Int\n}".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);
        assert!(!tokens.data.is_empty());

        // Verify delta encoding spans lines correctly
        let has_line_delta = tokens.data.iter().any(|t| t.delta_line > 0);
        assert!(has_line_delta);
    }

    #[test]
    fn test_generate_tokens_in_range() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Alpha = { a: Int }\ntype Beta = { b: Int }".to_string(),
        );

        let full = generate_semantic_tokens(&doc);
        let range = Range::new(Position::new(1, 0), Position::new(1, 100));
        let ranged = generate_semantic_tokens_in_range(&doc, range);

        assert!(!ranged.data.is_empty());
        assert!(ranged.data.len() < full.data.len());

        let mut line = 0u32;
        for token in &ranged.data {
            line += token.delta_line;
            assert_eq!(line, 1);
        }
    }

    #[test]
    fn test_classify_identifier_keywords() {
        let user_types: Vec<&str> = vec![];
        let props: Vec<&str> = vec![];

        let (token_type, _) = classify_identifier("theorem", &user_types, &props);
        assert_eq!(token_type, 0); // KEYWORD

        let (token_type, _) = classify_identifier("forall", &user_types, &props);
        assert_eq!(token_type, 0); // KEYWORD

        let (token_type, _) = classify_identifier("implies", &user_types, &props);
        assert_eq!(token_type, 0); // KEYWORD
    }

    #[test]
    fn test_classify_identifier_builtin_types() {
        let user_types: Vec<&str> = vec![];
        let props: Vec<&str> = vec![];

        let (token_type, modifiers) = classify_identifier("Int", &user_types, &props);
        assert_eq!(token_type, 1); // TYPE
        assert_ne!(modifiers, 0); // Should have DEFAULT_LIBRARY modifier

        let (token_type, _) = classify_identifier("Bool", &user_types, &props);
        assert_eq!(token_type, 1); // TYPE
    }

    #[test]
    fn test_classify_identifier_user_types() {
        let user_types: Vec<&str> = vec!["Value", "MyType"];
        let props: Vec<&str> = vec![];

        let (token_type, _) = classify_identifier("Value", &user_types, &props);
        assert_eq!(token_type, 1); // TYPE

        let (token_type, _) = classify_identifier("MyType", &user_types, &props);
        assert_eq!(token_type, 1); // TYPE
    }

    #[test]
    fn test_classify_identifier_properties() {
        let user_types: Vec<&str> = vec![];
        let props: Vec<&str> = vec!["my_theorem", "my_contract"];

        let (token_type, _) = classify_identifier("my_theorem", &user_types, &props);
        assert_eq!(token_type, 2); // FUNCTION

        let (token_type, _) = classify_identifier("my_contract", &user_types, &props);
        assert_eq!(token_type, 2); // FUNCTION
    }

    #[test]
    fn test_classify_identifier_variables() {
        let user_types: Vec<&str> = vec![];
        let props: Vec<&str> = vec![];

        let (token_type, _) = classify_identifier("x", &user_types, &props);
        assert_eq!(token_type, 3); // VARIABLE

        let (token_type, _) = classify_identifier("my_var", &user_types, &props);
        assert_eq!(token_type, 3); // VARIABLE
    }

    #[test]
    fn test_operators() {
        assert_eq!(classify_operator("==", 0), Some(2));
        assert_eq!(classify_operator("->", 0), Some(2));
        assert_eq!(classify_operator("::", 0), Some(2));
        assert_eq!(classify_operator("=", 0), Some(1));
        assert_eq!(classify_operator(".", 0), Some(1));
        assert_eq!(classify_operator("abc", 0), None);
    }

    #[test]
    fn test_delta_encoding() {
        let raw = vec![
            RawToken {
                line: 0,
                start_char: 0,
                length: 7,
                token_type: 0,
                token_modifiers: 0,
            },
            RawToken {
                line: 0,
                start_char: 8,
                length: 3,
                token_type: 2,
                token_modifiers: 0,
            },
            RawToken {
                line: 1,
                start_char: 4,
                length: 5,
                token_type: 1,
                token_modifiers: 0,
            },
        ];

        let encoded = delta_encode(&raw);
        assert_eq!(encoded.len(), 3);

        // First token: absolute position
        assert_eq!(encoded[0].delta_line, 0);
        assert_eq!(encoded[0].delta_start, 0);

        // Second token: same line, delta from previous
        assert_eq!(encoded[1].delta_line, 0);
        assert_eq!(encoded[1].delta_start, 8); // 8 - 0 = 8

        // Third token: new line, absolute column
        assert_eq!(encoded[2].delta_line, 1);
        assert_eq!(encoded[2].delta_start, 4); // absolute on new line
    }

    #[test]
    fn test_string_literal_tokens() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"theorem foo { "hello" == "world" }"#.to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // Should have string tokens
        let string_tokens: Vec<_> = tokens.data.iter().filter(|t| t.token_type == 8).collect();
        assert_eq!(string_tokens.len(), 2);
    }

    #[test]
    fn test_number_literal_tokens() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { 42 > 0 }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // Should have number tokens
        let number_tokens: Vec<_> = tokens.data.iter().filter(|t| t.token_type == 7).collect();
        assert_eq!(number_tokens.len(), 2); // 42 and 0
    }

    #[test]
    fn test_refinement_tokens() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "refinement foo refines Bar { abstraction { true } simulation { true } }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);
        assert!(!tokens.data.is_empty());

        // Check keywords are recognized
        let keyword_tokens: Vec<_> = tokens.data.iter().filter(|t| t.token_type == 0).collect();
        assert!(keyword_tokens.len() >= 4); // refinement, refines, abstraction, simulation
    }

    // ============== Mutation-killing tests for tokenize_document arithmetic ==============

    #[test]
    fn test_comment_line_start_calculation() {
        // Tests line 161: line_start = i + 1 (after newline)
        // And line 193: line_start = comment_start + comment_len + 1
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "// comment\ntheorem foo { true }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // The "theorem" token should be on line 1, column 0
        let theorem_token = tokens.data.iter().find(|t| t.length == 7).unwrap();
        assert_eq!(theorem_token.delta_line, 1, "theorem should be on line 1");
        assert_eq!(
            theorem_token.delta_start, 0,
            "theorem should start at column 0"
        );
    }

    #[test]
    fn test_word_end_calculation() {
        // Tests line 200: word_end = i + ch.len_utf8() and line 204: word_end = next_i + next_ch.len_utf8()
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem my_theorem { true }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // my_theorem should have length 10
        let my_theorem_token = tokens.data.iter().find(|t| t.length == 10).unwrap();
        assert_eq!(my_theorem_token.length, 10);
    }

    #[test]
    fn test_word_start_char_calculation() {
        // Tests line 211: start_char = (word_start - line_start) as u32
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { true }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // "foo" should be at column 8 (after "theorem ")
        let foo_token = tokens.data.iter().find(|t| t.length == 3).unwrap();
        // Since it's on the same line as theorem, delta_start is relative
        // theorem (0) -> foo (8): delta = 8
        assert_eq!(foo_token.delta_start, 8, "foo should start at column 8");
    }

    #[test]
    fn test_number_end_calculation() {
        // Tests line 234: num_end = i + 1 and line 238: num_end = next_i + 1
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { 12345 > 0 }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // 12345 should have length 5
        let num_token = tokens.data.iter().find(|t| t.length == 5).unwrap();
        assert_eq!(num_token.token_type, 7, "should be NUMBER token");
        assert_eq!(num_token.length, 5);
    }

    #[test]
    fn test_number_start_char_calculation() {
        // Tests line 245: start_char = (num_start - line_start) as u32
        // and line 246: length = (num_end - num_start) as u32
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { 99 }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // Find the number 99 (length 2, type 7)
        let num_token = tokens
            .data
            .iter()
            .find(|t| t.token_type == 7 && t.length == 2)
            .unwrap();
        assert_eq!(num_token.length, 2, "99 should have length 2");
    }

    #[test]
    fn test_string_length_calculation() {
        // Tests line 259: str_end = i + 1 (initial)
        // and line 262: str_end = next_i + next_ch.len_utf8() (during iteration)
        // and line 277: length = (str_end - str_start) as u32
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"theorem foo { "hello" }"#.to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // "hello" including quotes should have length 7
        let str_token = tokens.data.iter().find(|t| t.token_type == 8).unwrap();
        assert_eq!(str_token.length, 7, "\"hello\" should have length 7");
    }

    #[test]
    fn test_string_start_char_calculation() {
        // Tests line 276: start_char = (str_start - line_start) as u32
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"theorem foo { "x" }"#.to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // The string "x" should be at a specific column
        let str_token = tokens.data.iter().find(|t| t.token_type == 8).unwrap();
        assert_eq!(str_token.length, 3, "\"x\" should have length 3");
    }

    #[test]
    fn test_operator_start_char_calculation() {
        // Tests line 291: start_char = (i - line_start) as u32
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { x == y }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // Find the == operator (type 5, length 2)
        let op_token = tokens
            .data
            .iter()
            .find(|t| t.token_type == 5 && t.length == 2)
            .unwrap();
        assert_eq!(op_token.length, 2, "== should have length 2");
    }

    #[test]
    fn test_multiline_line_start_tracking() {
        // Tests line 181: start_char = (comment_start - line_start) as u32
        // and line 184: length = comment_len as u32
        // and line 192: current_line += 1
        // and line 193: line_start = comment_start + comment_len + 1
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "// line 0\n// line 1\ntheorem foo { true }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // Should have 2 comment tokens and then theorem
        let comment_tokens: Vec<_> = tokens.data.iter().filter(|t| t.token_type == 6).collect();
        assert_eq!(comment_tokens.len(), 2);

        // Check line deltas
        assert_eq!(comment_tokens[0].delta_line, 0, "first comment on line 0");
        assert_eq!(comment_tokens[1].delta_line, 1, "second comment on line 1");
    }

    // ============== Mutation-killing tests for classify_operator ==============

    #[test]
    fn test_classify_operator_two_char_operators() {
        // Tests the || chain at lines 339-345 (mutations flip || to &&)
        // All two-char operators should return Some(2)
        assert_eq!(
            classify_operator("==xxx", 0),
            Some(2),
            "== should be recognized"
        );
        assert_eq!(
            classify_operator("!=xxx", 0),
            Some(2),
            "!= should be recognized"
        );
        assert_eq!(
            classify_operator("<=xxx", 0),
            Some(2),
            "<= should be recognized"
        );
        assert_eq!(
            classify_operator(">=xxx", 0),
            Some(2),
            ">= should be recognized"
        );
        assert_eq!(
            classify_operator("->xxx", 0),
            Some(2),
            "-> should be recognized"
        );
        assert_eq!(
            classify_operator("::xxx", 0),
            Some(2),
            ":: should be recognized"
        );
    }

    #[test]
    fn test_classify_operator_single_char() {
        // Test single character operators
        assert_eq!(
            classify_operator("=xxx", 0),
            Some(1),
            "= should be recognized"
        );
        assert_eq!(
            classify_operator("<xxx", 0),
            Some(1),
            "< should be recognized"
        );
        assert_eq!(
            classify_operator(">xxx", 0),
            Some(1),
            "> should be recognized"
        );
        assert_eq!(
            classify_operator("+xxx", 0),
            Some(1),
            "+ should be recognized"
        );
        assert_eq!(
            classify_operator("-xxx", 0),
            Some(1),
            "- should be recognized"
        );
        assert_eq!(
            classify_operator("*xxx", 0),
            Some(1),
            "* should be recognized"
        );
        assert_eq!(
            classify_operator("/xxx", 0),
            Some(1),
            "/ should be recognized"
        );
        assert_eq!(
            classify_operator("%xxx", 0),
            Some(1),
            "% should be recognized"
        );
        assert_eq!(
            classify_operator(".xxx", 0),
            Some(1),
            ". should be recognized"
        );
        assert_eq!(
            classify_operator(":xxx", 0),
            Some(1),
            ": should be recognized"
        );
        assert_eq!(
            classify_operator("|xxx", 0),
            Some(1),
            "| should be recognized"
        );
        assert_eq!(
            classify_operator("&xxx", 0),
            Some(1),
            "& should be recognized"
        );
        assert_eq!(
            classify_operator("!xxx", 0),
            Some(1),
            "! should be recognized"
        );
    }

    #[test]
    fn test_classify_operator_not_operator() {
        // Non-operators should return None
        assert_eq!(classify_operator("abc", 0), None, "abc not an operator");
        assert_eq!(classify_operator("xyz", 0), None, "xyz not an operator");
        assert_eq!(classify_operator("123", 0), None, "123 not an operator");
    }

    #[test]
    fn test_classify_operator_at_offset() {
        // Test at different positions
        assert_eq!(classify_operator("xx==yy", 2), Some(2), "== at offset 2");
        assert_eq!(classify_operator("xx+yy", 2), Some(1), "+ at offset 2");
    }

    // ============== Mutation-killing tests for token_overlaps_range ==============

    #[test]
    fn test_token_overlaps_range_line_boundaries() {
        // Tests line 363: if token.line < range.start.line || token.line > range.end.line
        let token = RawToken {
            line: 5,
            start_char: 0,
            length: 10,
            token_type: 0,
            token_modifiers: 0,
        };

        // Range on lines 3-4: token on line 5 should NOT overlap
        let range_before = Range::new(Position::new(3, 0), Position::new(4, 100));
        assert!(
            !token_overlaps_range(&token, &range_before),
            "token on line 5 should not overlap range 3-4"
        );

        // Range on lines 6-7: token on line 5 should NOT overlap
        let range_after = Range::new(Position::new(6, 0), Position::new(7, 100));
        assert!(
            !token_overlaps_range(&token, &range_after),
            "token on line 5 should not overlap range 6-7"
        );

        // Range on lines 4-6: token on line 5 SHOULD overlap
        let range_includes = Range::new(Position::new(4, 0), Position::new(6, 100));
        assert!(
            token_overlaps_range(&token, &range_includes),
            "token on line 5 should overlap range 4-6"
        );
    }

    #[test]
    fn test_token_overlaps_range_character_boundaries_start_line() {
        // Tests line 368: token_end = token.start_char + token.length
        // and line 370: if token.line == range.start.line && token_end <= range.start.character
        let token = RawToken {
            line: 5,
            start_char: 10,
            length: 5, // token spans chars 10-15
            token_type: 0,
            token_modifiers: 0,
        };

        // Range starts at char 20: token ends at 15, so NO overlap
        let range_after = Range::new(Position::new(5, 20), Position::new(5, 30));
        assert!(
            !token_overlaps_range(&token, &range_after),
            "token at 10-15 should not overlap range starting at 20"
        );

        // Range starts at char 15: token ends at 15, token_end <= start.character, NO overlap
        let range_at_end = Range::new(Position::new(5, 15), Position::new(5, 30));
        assert!(
            !token_overlaps_range(&token, &range_at_end),
            "token at 10-15 should not overlap range starting at 15"
        );

        // Range starts at char 14: token ends at 15 > 14, so DOES overlap
        let range_just_inside = Range::new(Position::new(5, 14), Position::new(5, 30));
        assert!(
            token_overlaps_range(&token, &range_just_inside),
            "token at 10-15 should overlap range starting at 14"
        );
    }

    #[test]
    fn test_token_overlaps_range_character_boundaries_end_line() {
        // Tests line 373: if token.line == range.end.line && token_start >= range.end.character
        let token = RawToken {
            line: 5,
            start_char: 20,
            length: 5, // token spans chars 20-25
            token_type: 0,
            token_modifiers: 0,
        };

        // Range ends at char 15: token starts at 20 >= 15, NO overlap
        let range_before = Range::new(Position::new(5, 0), Position::new(5, 15));
        assert!(
            !token_overlaps_range(&token, &range_before),
            "token at 20-25 should not overlap range ending at 15"
        );

        // Range ends at char 20: token starts at 20 >= 20, NO overlap
        let range_at_start = Range::new(Position::new(5, 0), Position::new(5, 20));
        assert!(
            !token_overlaps_range(&token, &range_at_start),
            "token at 20-25 should not overlap range ending at 20"
        );

        // Range ends at char 21: token starts at 20 < 21, DOES overlap
        let range_just_inside = Range::new(Position::new(5, 0), Position::new(5, 21));
        assert!(
            token_overlaps_range(&token, &range_just_inside),
            "token at 20-25 should overlap range ending at 21"
        );
    }

    // ============== Mutation-killing tests for delta_encode ==============

    #[test]
    fn test_delta_encode_line_subtraction() {
        // Tests line 387: delta_line = token.line - prev_line
        // If we use + instead of -, delta would be wrong
        let raw = vec![
            RawToken {
                line: 0,
                start_char: 0,
                length: 5,
                token_type: 0,
                token_modifiers: 0,
            },
            RawToken {
                line: 3,
                start_char: 0,
                length: 5,
                token_type: 0,
                token_modifiers: 0,
            },
        ];

        let encoded = delta_encode(&raw);
        assert_eq!(encoded[0].delta_line, 0, "first token delta_line is 0");
        assert_eq!(
            encoded[1].delta_line, 3,
            "second token delta_line is 3 (3 - 0)"
        );
    }

    #[test]
    fn test_delta_encode_char_subtraction_same_line() {
        // Tests line 389: token.start_char - prev_char (when on same line)
        // If we use + instead of -, delta would be wrong
        let raw = vec![
            RawToken {
                line: 0,
                start_char: 5,
                length: 3,
                token_type: 0,
                token_modifiers: 0,
            },
            RawToken {
                line: 0,
                start_char: 15,
                length: 4,
                token_type: 0,
                token_modifiers: 0,
            },
        ];

        let encoded = delta_encode(&raw);
        assert_eq!(encoded[0].delta_start, 5, "first token delta_start is 5");
        assert_eq!(
            encoded[1].delta_start, 10,
            "second token delta_start is 10 (15 - 5)"
        );
    }

    #[test]
    fn test_delta_encode_absolute_on_new_line() {
        // Tests line 391: token.start_char (absolute when on new line)
        let raw = vec![
            RawToken {
                line: 0,
                start_char: 10,
                length: 3,
                token_type: 0,
                token_modifiers: 0,
            },
            RawToken {
                line: 1,
                start_char: 4,
                length: 5,
                token_type: 0,
                token_modifiers: 0,
            },
        ];

        let encoded = delta_encode(&raw);
        assert_eq!(encoded[1].delta_line, 1, "moved to new line");
        assert_eq!(
            encoded[1].delta_start, 4,
            "on new line, delta_start is absolute (4)"
        );
    }

    // ============== Additional mutation-killing tests for remaining mutations ==============

    #[test]
    fn test_line_start_after_newline() {
        // Tests line 161: line_start = i + 1 (mutation: + to - or *)
        // Need to verify token positions on line 1+ are correct
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "x\nabc".to_string(), // 'x' at 0, '\n' at 1, 'abc' at 2-4
        );
        let tokens = generate_semantic_tokens(&doc);

        // First token: 'x' on line 0, col 0
        let first = &tokens.data[0];
        assert_eq!(first.delta_line, 0);
        assert_eq!(first.delta_start, 0);
        assert_eq!(first.length, 1);

        // Second token: 'abc' on line 1, col 0 (line_start should be 2)
        let second = &tokens.data[1];
        assert_eq!(second.delta_line, 1, "abc is on line 1");
        assert_eq!(second.delta_start, 0, "abc starts at col 0"); // If line_start=i-1=0, this would be 2
        assert_eq!(second.length, 3);
    }

    #[test]
    fn test_comment_start_char_offset() {
        // Tests line 184: start_char = (comment_start - line_start) as u32
        // Mutation: - to +
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "   // comment".to_string(), // 3 spaces then comment starting at col 3
        );
        let tokens = generate_semantic_tokens(&doc);

        assert_eq!(tokens.data.len(), 1);
        let comment = &tokens.data[0];
        assert_eq!(comment.delta_start, 3, "comment starts at col 3"); // If + instead of -, would be much larger
    }

    #[test]
    fn test_comment_peek_and_condition() {
        // Tests line 171: ch == '/' && chars.peek().is_some_and(|(_, c)| *c == '/')
        // Mutation: && to ||
        // Single '/' should NOT be a comment
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { x / y }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        // Should have operator tokens for '/'
        let op_tokens: Vec<_> = tokens.data.iter().filter(|t| t.token_type == 5).collect();
        assert!(
            !op_tokens.is_empty(),
            "single / should be an operator, not a comment"
        );

        // No comment tokens
        let comment_tokens: Vec<_> = tokens.data.iter().filter(|t| t.token_type == 6).collect();
        assert!(
            comment_tokens.is_empty(),
            "single / should not create a comment"
        );
    }

    #[test]
    fn test_word_end_initial_value() {
        // Tests line 200: word_end = i + ch.len_utf8()
        // Mutation: + to *
        // Single char identifier should have length 1
        let doc = Document::new(Url::parse("file:///test.usl").unwrap(), 1, "x".to_string());
        let tokens = generate_semantic_tokens(&doc);

        assert_eq!(tokens.data.len(), 1);
        assert_eq!(
            tokens.data[0].length, 1,
            "single char identifier has length 1"
        ); // If * instead of +, length would be 0
    }

    #[test]
    fn test_number_initial_end() {
        // Tests line 234: num_end = i + 1
        // Mutation: + to *
        // Single digit should have length 1
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { 5 }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        let num_token = tokens.data.iter().find(|t| t.token_type == 7).unwrap();
        assert_eq!(num_token.length, 1, "single digit has length 1"); // If * instead of +, would be 0
    }

    #[test]
    fn test_number_with_underscore_condition() {
        // Tests line 236: next_ch.is_ascii_digit() || next_ch == '.' || next_ch == '_'
        // Mutation: || to &&
        // A number with underscore should parse correctly
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { 1_000 }".to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        let num_token = tokens.data.iter().find(|t| t.token_type == 7).unwrap();
        assert_eq!(num_token.length, 5, "1_000 should have length 5");
    }

    #[test]
    fn test_number_start_char_with_offset() {
        // Tests line 245: start_char = (num_start - line_start) as u32
        // Mutation: - to +
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "     42".to_string(), // 5 spaces then 42 at col 5
        );
        let tokens = generate_semantic_tokens(&doc);

        assert_eq!(tokens.data.len(), 1);
        let num = &tokens.data[0];
        assert_eq!(num.delta_start, 5, "42 starts at col 5"); // If +, would be much larger
    }

    #[test]
    fn test_string_initial_end() {
        // Tests line 259: str_end = i + 1
        // Mutation: + to - or *
        // Empty string "" should have length 2
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"theorem foo { "" }"#.to_string(),
        );
        let tokens = generate_semantic_tokens(&doc);

        let str_token = tokens.data.iter().find(|t| t.token_type == 8).unwrap();
        assert_eq!(str_token.length, 2, "empty string \"\" has length 2"); // If - or *, would be wrong
    }

    #[test]
    fn test_string_start_char_with_offset() {
        // Tests line 276: start_char = (str_start - line_start) as u32
        // Mutation: - to +
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"    "x""#.to_string(), // 4 spaces then "x" at col 4
        );
        let tokens = generate_semantic_tokens(&doc);

        assert_eq!(tokens.data.len(), 1);
        let str = &tokens.data[0];
        assert_eq!(str.delta_start, 4, "string starts at col 4"); // If +, would be much larger
    }

    #[test]
    fn test_operator_start_char_with_offset() {
        // Tests line 291: start_char = (i - line_start) as u32
        // Mutation: - to +
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "   +".to_string(), // 3 spaces then + at col 3
        );
        let tokens = generate_semantic_tokens(&doc);

        assert_eq!(tokens.data.len(), 1);
        let op = &tokens.data[0];
        assert_eq!(op.delta_start, 3, "+ starts at col 3"); // If +, would be much larger
    }

    #[test]
    fn test_multiline_token_positions() {
        // Tests multiple lines with varying positions to catch arithmetic errors
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "  a\n    bb\n      ccc".to_string(),
            // Line 0: "  a"      -> 'a' at col 2
            // Line 1: "    bb"   -> 'bb' at col 4
            // Line 2: "      ccc" -> 'ccc' at col 6
        );
        let tokens = generate_semantic_tokens(&doc);

        assert_eq!(tokens.data.len(), 3);

        // Token 1: 'a' at line 0, col 2
        assert_eq!(tokens.data[0].delta_line, 0);
        assert_eq!(tokens.data[0].delta_start, 2);
        assert_eq!(tokens.data[0].length, 1);

        // Token 2: 'bb' at line 1, col 4
        assert_eq!(tokens.data[1].delta_line, 1);
        assert_eq!(tokens.data[1].delta_start, 4); // Absolute on new line
        assert_eq!(tokens.data[1].length, 2);

        // Token 3: 'ccc' at line 2, col 6
        assert_eq!(tokens.data[2].delta_line, 1);
        assert_eq!(tokens.data[2].delta_start, 6); // Absolute on new line
        assert_eq!(tokens.data[2].length, 3);
    }

    // ============== Additional tests for remaining mutations ==============

    #[test]
    fn test_comment_at_end_of_file_no_newline() {
        // Tests line 181: .unwrap_or(text.len() - comment_start)
        // When comment is at EOF without newline, length should be correct
        // Use "x // end" so comment_start is NOT 0, making the mutation detectable
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "x // end".to_string(), // Comment at EOF starting at position 2, no newline
        );
        let tokens = generate_semantic_tokens(&doc);

        // Should have 'x' identifier and '// end' comment
        let comment = tokens.data.iter().find(|t| t.token_type == 6).unwrap();
        // text.len() = 8, comment_start = 2, so length should be 8 - 2 = 6
        // If + instead of -, length would be 8 + 2 = 10 (wrong)
        assert_eq!(comment.length, 6, "// end has length 6");
    }

    #[test]
    fn test_comment_position_nonzero_line_start() {
        // Tests line 184: start_char = (comment_start - line_start) as u32
        // When line_start is nonzero (after newlines), comment position should be correct
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "a\n  // comment".to_string(), // Line 1: "  // comment" -> comment at col 2
        );
        let tokens = generate_semantic_tokens(&doc);

        // Find the comment token (on line 1)
        let comment = tokens.data.iter().find(|t| t.token_type == 6).unwrap();
        assert_eq!(comment.delta_start, 2, "comment starts at col 2 on line 1");
        // If + instead of -, would be much larger
    }

    #[test]
    fn test_number_position_second_line() {
        // Tests line 245: start_char = (num_start - line_start) as u32
        // Number on second line with offset
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "a\n  123".to_string(), // Line 1: "  123" -> number at col 2
        );
        let tokens = generate_semantic_tokens(&doc);

        // Find the number token
        let num = tokens.data.iter().find(|t| t.token_type == 7).unwrap();
        assert_eq!(num.delta_start, 2, "number starts at col 2 on second line"); // If + instead of -, would be much larger
        assert_eq!(num.length, 3);
    }

    #[test]
    fn test_string_initial_end_empty_string() {
        // Tests line 259: str_end = i + 1
        // Empty string "" - verify the initial str_end is correct
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "\"\"".to_string(), // Just empty string
        );
        let tokens = generate_semantic_tokens(&doc);

        assert_eq!(tokens.data.len(), 1);
        let str_token = &tokens.data[0];
        assert_eq!(str_token.token_type, 8, "should be string");
        assert_eq!(str_token.length, 2, "empty string has length 2"); // If - or *, would be 0 or wrong
    }

    #[test]
    fn test_string_unterminated_at_eof() {
        // Tests line 259: str_end = i + 1 (mutation: + to *)
        // When string is unterminated and at EOF, the initial str_end matters
        // For string starting at position p and no closing quote, str_end stays at i + 1
        // This only happens if the loop doesn't run (empty content after opening quote at EOF)
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "x \"".to_string(), // 'x' then space then unterminated string " at position 2
        );
        let tokens = generate_semantic_tokens(&doc);

        // Find the string token (should be at position 2 with length 1)
        let str_token = tokens.data.iter().find(|t| t.token_type == 8).unwrap();
        // str_start = 2, str_end should be 3 (2 + 1), length = 1
        // If mutation * 1: str_end = 2 * 1 = 2, length = 0 (wrong)
        assert_eq!(
            str_token.length, 1,
            "unterminated string at EOF has length 1"
        );
    }

    #[test]
    fn test_string_position_second_line() {
        // Tests line 276: start_char = (str_start - line_start) as u32
        // String on second line with offset
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "a\n  \"x\"".to_string(), // Line 1: "  \"x\"" -> string at col 2
        );
        let tokens = generate_semantic_tokens(&doc);

        // Find the string token
        let str_token = tokens.data.iter().find(|t| t.token_type == 8).unwrap();
        assert_eq!(
            str_token.delta_start, 2,
            "string starts at col 2 on second line"
        ); // If + instead of -, would be much larger
        assert_eq!(str_token.length, 3);
    }

    #[test]
    fn test_operator_position_second_line() {
        // Tests line 291: start_char = (i - line_start) as u32
        // Operator on second line with offset
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "a\n  +".to_string(), // Line 1: "  +" -> operator at col 2
        );
        let tokens = generate_semantic_tokens(&doc);

        // Find the operator token (type 5)
        let op = tokens.data.iter().find(|t| t.token_type == 5).unwrap();
        assert_eq!(op.delta_start, 2, "operator starts at col 2 on second line"); // If + instead of -, would be much larger
        assert_eq!(op.length, 1);
    }
}

// ============================ Kani Proof Harnesses ============================
#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use tower_lsp::lsp_types::{Position, Range};

    /// Prove that delta_encode preserves the number of tokens.
    /// Output length always equals input length.
    #[kani::proof]
    fn verify_delta_encode_length_preservation() {
        // Create a small array of raw tokens (bounded to make Kani tractable)
        let len: usize = kani::any();
        kani::assume(len <= 4);

        let mut tokens: Vec<RawToken> = Vec::with_capacity(len);
        for _ in 0..len {
            tokens.push(RawToken {
                line: kani::any(),
                start_char: kani::any(),
                length: kani::any(),
                token_type: kani::any(),
                token_modifiers: kani::any(),
            });
        }

        let result = delta_encode(&tokens);

        kani::assert(
            result.len() == tokens.len(),
            "delta_encode must preserve token count",
        );
    }

    /// Prove that for sorted tokens on the same line, delta_start correctly
    /// computes the difference (no wraparound/underflow).
    #[kani::proof]
    fn verify_delta_encode_same_line_no_underflow() {
        // Two tokens on the same line, both starting from line 0
        let start1: u32 = kani::any();
        let start2: u32 = kani::any();

        // Assume sorted order (typical for semantic tokens)
        kani::assume(start1 <= start2);

        let tokens = vec![
            RawToken {
                line: 0, // First token on line 0 means delta_line = 0
                start_char: start1,
                length: 1,
                token_type: 0,
                token_modifiers: 0,
            },
            RawToken {
                line: 0, // Same line
                start_char: start2,
                length: 1,
                token_type: 0,
                token_modifiers: 0,
            },
        ];

        let result = delta_encode(&tokens);

        // First token: delta_start = start1 (absolute), delta_line = 0 - 0 = 0
        kani::assert(
            result[0].delta_start == start1,
            "First token start is absolute",
        );
        kani::assert(
            result[0].delta_line == 0,
            "First token on line 0 has delta_line 0",
        );

        // Second token (same line): delta_start = start2 - start1
        kani::assert(
            result[1].delta_start == start2 - start1,
            "Second token delta_start is difference",
        );
        kani::assert(result[1].delta_line == 0, "Same line means delta_line is 0");
    }

    /// Prove that for tokens on different lines, delta_start becomes absolute
    /// (reset to the start_char position on the new line).
    #[kani::proof]
    fn verify_delta_encode_new_line_absolute_start() {
        let line1: u32 = kani::any();
        let line2: u32 = kani::any();
        let start1: u32 = kani::any();
        let start2: u32 = kani::any();

        // Second token on a later line
        kani::assume(line2 > line1);

        let tokens = vec![
            RawToken {
                line: line1,
                start_char: start1,
                length: 1,
                token_type: 0,
                token_modifiers: 0,
            },
            RawToken {
                line: line2,
                start_char: start2,
                length: 1,
                token_type: 0,
                token_modifiers: 0,
            },
        ];

        let result = delta_encode(&tokens);

        // Second token: on new line, delta_start is absolute (start2)
        kani::assert(
            result[1].delta_start == start2,
            "On new line, delta_start is absolute",
        );
        kani::assert(
            result[1].delta_line == line2 - line1,
            "delta_line is difference between lines",
        );
    }

    /// Prove that token properties (length, type, modifiers) are preserved.
    #[kani::proof]
    fn verify_delta_encode_preserves_token_properties() {
        let length: u32 = kani::any();
        let token_type: u32 = kani::any();
        let token_modifiers: u32 = kani::any();

        let tokens = vec![RawToken {
            line: kani::any(),
            start_char: kani::any(),
            length,
            token_type,
            token_modifiers,
        }];

        let result = delta_encode(&tokens);

        kani::assert(result[0].length == length, "length must be preserved");
        kani::assert(
            result[0].token_type == token_type,
            "token_type must be preserved",
        );
        kani::assert(
            result[0].token_modifiers_bitset == token_modifiers,
            "token_modifiers must be preserved",
        );
    }

    /// Prove that classify_operator returns correct lengths for known operators.
    #[kani::proof]
    fn verify_classify_operator_known_operators() {
        // Test that two-char operators return Some(2)
        let result_eq = classify_operator("==", 0);
        kani::assert(result_eq == Some(2), "== should return Some(2)");

        let result_ne = classify_operator("!=", 0);
        kani::assert(result_ne == Some(2), "!= should return Some(2)");

        let result_arrow = classify_operator("->", 0);
        kani::assert(result_arrow == Some(2), "-> should return Some(2)");

        // Test that single-char operators return Some(1)
        let result_plus = classify_operator("+x", 0);
        kani::assert(result_plus == Some(1), "+ should return Some(1)");

        // Test that non-operators return None
        let result_alpha = classify_operator("abc", 0);
        kani::assert(result_alpha == None, "abc should return None");
    }

    /// Prove that token_overlaps_range correctly identifies non-overlapping tokens.
    #[kani::proof]
    fn verify_token_overlaps_range_line_boundaries() {
        let token_line: u32 = kani::any();
        let range_start_line: u32 = kani::any();
        let range_end_line: u32 = kani::any();

        // Assume valid range
        kani::assume(range_start_line <= range_end_line);

        let token = RawToken {
            line: token_line,
            start_char: 0,
            length: 10,
            token_type: 0,
            token_modifiers: 0,
        };

        let range = Range {
            start: Position::new(range_start_line, 0),
            end: Position::new(range_end_line, 100),
        };

        let overlaps = token_overlaps_range(&token, &range);

        // If token is strictly before or after range lines, no overlap
        if token_line < range_start_line || token_line > range_end_line {
            kani::assert(!overlaps, "Token outside line range should not overlap");
        }
    }
}
