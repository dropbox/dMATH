//! Signature help for USL contracts
//!
//! Provides parameter hints when typing contract invocations or references.

use crate::document::Document;
use crate::symbols::format_type;
use dashprove_usl::{Contract, Property};
use tower_lsp::lsp_types::{
    ParameterInformation, ParameterLabel, Position, SignatureHelp, SignatureInformation,
};

/// Generate signature help at a given position.
///
/// Returns signature information for contracts when the cursor is:
/// - After a contract name followed by `(`
/// - Inside a contract's parameter list
#[must_use]
pub fn generate_signature_help(doc: &Document, pos: Position) -> Option<SignatureHelp> {
    let spec = doc.spec.as_ref()?;

    // Get text before cursor position
    let offset = doc.position_to_offset(pos.line, pos.character);
    let text_before = &doc.text[..offset];

    // Find the context: look for pattern like "ContractName(" or "Type::method("
    let (contract_name, active_param) = find_contract_context(text_before)?;

    // Find matching contract in spec
    let contract = find_contract_by_name(spec.properties.iter(), &contract_name)?;

    // Build signature help
    let signature = build_signature_info(contract);

    Some(SignatureHelp {
        signatures: vec![signature],
        active_signature: Some(0),
        active_parameter: Some(active_param),
    })
}

/// Find contract context from text before cursor.
///
/// Returns (contract_name, active_parameter_index) if cursor is in a contract call context.
fn find_contract_context(text: &str) -> Option<(String, u32)> {
    // Find the last unmatched opening paren
    let mut paren_depth = 0;
    let mut last_open_paren = None;

    for (i, ch) in text.char_indices().rev() {
        match ch {
            ')' => paren_depth += 1,
            '(' => {
                if paren_depth == 0 {
                    last_open_paren = Some(i);
                    break;
                }
                paren_depth -= 1;
            }
            _ => {}
        }
    }

    let paren_pos = last_open_paren?;

    // Extract identifier before the paren
    let before_paren = &text[..paren_pos];
    let name = extract_contract_name(before_paren)?;

    // Count commas to determine active parameter
    let after_paren = &text[paren_pos + 1..];
    let active_param = count_commas_at_top_level(after_paren);

    Some((name, active_param))
}

/// Extract contract name from text ending just before `(`.
///
/// Handles both simple names and qualified names like `Type::method`.
fn extract_contract_name(text: &str) -> Option<String> {
    let trimmed = text.trim_end();
    if trimmed.is_empty() {
        return None;
    }

    // Find the start of the identifier (may include ::)
    let mut start = trimmed.len();
    let chars: Vec<char> = trimmed.chars().collect();

    while start > 0 {
        let ch = chars[start - 1];
        if ch.is_alphanumeric() || ch == '_' || ch == ':' {
            start -= 1;
        } else {
            break;
        }
    }

    let name = &trimmed[start..];
    if name.is_empty() || name.starts_with(':') {
        return None;
    }

    Some(name.to_string())
}

/// Count commas at the top level (not inside nested parens/braces).
fn count_commas_at_top_level(text: &str) -> u32 {
    let mut count: u32 = 0;
    let mut paren_depth: i32 = 0;
    let mut brace_depth: i32 = 0;
    let mut bracket_depth: i32 = 0;

    for ch in text.chars() {
        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '{' => brace_depth += 1,
            '}' => brace_depth = brace_depth.saturating_sub(1),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            ',' if paren_depth == 0 && brace_depth == 0 && bracket_depth == 0 => {
                count += 1;
            }
            _ => {}
        }
    }

    count
}

/// Find a contract by name in the properties.
fn find_contract_by_name<'a>(
    properties: impl Iterator<Item = &'a Property>,
    name: &str,
) -> Option<&'a Contract> {
    for prop in properties {
        if let Property::Contract(contract) = prop {
            let contract_name = contract.type_path.join("::");
            if contract_name == name || contract.type_path.last().is_some_and(|last| last == name) {
                return Some(contract);
            }
        }
    }
    None
}

/// Build signature information for a contract.
fn build_signature_info(contract: &Contract) -> SignatureInformation {
    let name = contract.type_path.join("::");
    let params_str = contract
        .params
        .iter()
        .map(|p| format!("{}: {}", p.name, format_type(&p.ty)))
        .collect::<Vec<_>>()
        .join(", ");

    let return_str = contract
        .return_type
        .as_ref()
        .map(|t| format!(" -> {}", format_type(t)))
        .unwrap_or_default();

    let label = format!("contract {}({}){}", name, params_str, return_str);

    // Build parameter information with label offsets
    let parameters = build_parameter_info(contract, &label);

    SignatureInformation {
        label,
        documentation: Some(tower_lsp::lsp_types::Documentation::String(format!(
            "Contract for `{}`\n\n{} precondition(s), {} postcondition(s)",
            name,
            contract.requires.len(),
            contract.ensures.len()
        ))),
        parameters: Some(parameters),
        active_parameter: None,
    }
}

/// Build parameter information with correct label offsets.
fn build_parameter_info(contract: &Contract, label: &str) -> Vec<ParameterInformation> {
    let mut params = Vec::new();

    // Find the opening paren position
    let paren_pos = label.find('(').unwrap_or(0) + 1;
    let params_section = &label[paren_pos..];

    for param in &contract.params {
        let param_str = format!("{}: {}", param.name, format_type(&param.ty));

        // Find this parameter in the params section
        if let Some(offset) = params_section.find(&param_str) {
            let start = paren_pos as u32 + offset as u32;
            let end = start + param_str.len() as u32;

            params.push(ParameterInformation {
                label: ParameterLabel::LabelOffsets([start, end]),
                documentation: Some(tower_lsp::lsp_types::Documentation::String(format!(
                    "Parameter `{}` of type `{}`",
                    param.name,
                    format_type(&param.ty)
                ))),
            });
        }
    }

    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::{Param, Type};
    use tower_lsp::lsp_types::Url;

    fn make_doc(text: &str) -> Document {
        Document::new(Url::parse("file:///test.usl").unwrap(), 1, text.to_string())
    }

    #[test]
    fn test_extract_contract_name_simple() {
        assert_eq!(extract_contract_name("foo"), Some("foo".to_string()));
        assert_eq!(
            extract_contract_name("  myContract"),
            Some("myContract".to_string())
        );
        assert_eq!(extract_contract_name("x = bar"), Some("bar".to_string()));
    }

    #[test]
    fn test_extract_contract_name_qualified() {
        assert_eq!(
            extract_contract_name("Type::method"),
            Some("Type::method".to_string())
        );
        assert_eq!(
            extract_contract_name("call Foo::bar"),
            Some("Foo::bar".to_string())
        );
    }

    #[test]
    fn test_extract_contract_name_invalid() {
        assert_eq!(extract_contract_name(""), None);
        assert_eq!(extract_contract_name("   "), None);
        assert_eq!(extract_contract_name("::invalid"), None);
    }

    #[test]
    fn test_count_commas_simple() {
        assert_eq!(count_commas_at_top_level(""), 0);
        assert_eq!(count_commas_at_top_level("a"), 0);
        assert_eq!(count_commas_at_top_level("a, b"), 1);
        assert_eq!(count_commas_at_top_level("a, b, c"), 2);
    }

    #[test]
    fn test_count_commas_nested() {
        // Commas inside parens shouldn't count
        assert_eq!(count_commas_at_top_level("f(a, b)"), 0);
        assert_eq!(count_commas_at_top_level("f(a, b), c"), 1);
        assert_eq!(count_commas_at_top_level("f(a, b), g(c, d)"), 1);

        // Commas inside braces shouldn't count
        assert_eq!(count_commas_at_top_level("{a, b}"), 0);
        assert_eq!(count_commas_at_top_level("{a, b}, c"), 1);
    }

    #[test]
    fn test_find_contract_context() {
        assert_eq!(find_contract_context("foo("), Some(("foo".to_string(), 0)));
        assert_eq!(find_contract_context("foo(a"), Some(("foo".to_string(), 0)));
        assert_eq!(
            find_contract_context("foo(a, "),
            Some(("foo".to_string(), 1))
        );
        assert_eq!(
            find_contract_context("foo(a, b"),
            Some(("foo".to_string(), 1))
        );
        assert_eq!(
            find_contract_context("foo(a, b, "),
            Some(("foo".to_string(), 2))
        );
    }

    #[test]
    fn test_find_contract_context_qualified() {
        assert_eq!(
            find_contract_context("Type::method("),
            Some(("Type::method".to_string(), 0))
        );
        assert_eq!(
            find_contract_context("Type::method(self, "),
            Some(("Type::method".to_string(), 1))
        );
    }

    #[test]
    fn test_find_contract_context_no_context() {
        assert_eq!(find_contract_context("foo"), None);
        assert_eq!(find_contract_context("foo)"), None);
        assert_eq!(find_contract_context("foo()"), None); // Closed paren
    }

    #[test]
    fn test_signature_help_for_contract() {
        // Test signature help using a contract definition with cursor inside the parens.
        // The document parses successfully.
        let doc = make_doc(
            r#"
type Stack = { elements: List<Int>, capacity: Int }

contract Stack::push(self: Stack, item: Int) -> Result<Stack> {
    requires { self.capacity > 0 }
    ensures { true }
}

theorem test { true }
"#,
        );

        // Verify spec parsed
        assert!(doc.spec.is_some(), "Spec should parse");

        // Position inside the contract's parameter list (after "push(")
        // Line 3: "contract Stack::push(self: Stack, item: Int) -> Result<Stack> {"
        // Position at character 21 is after "push("
        let pos = Position::new(3, 21);
        let help = generate_signature_help(&doc, pos);

        assert!(help.is_some(), "Should provide signature help");
        let help = help.unwrap();
        assert_eq!(help.signatures.len(), 1);

        let sig = &help.signatures[0];
        assert!(sig.label.contains("Stack::push"));
        assert!(sig.label.contains("self: Stack"));
        assert!(sig.label.contains("item: Int"));
        assert!(sig.label.contains("-> Result<Stack>"));

        assert_eq!(help.active_parameter, Some(0));
    }

    #[test]
    fn test_signature_help_second_param() {
        let doc = make_doc(
            r#"
type Stack = { elements: List<Int>, capacity: Int }

contract Stack::push(self: Stack, item: Int) -> Result<Stack> {
    requires { self.capacity > 0 }
    ensures { true }
}

theorem test { true }
"#,
        );

        // Position inside contract parameter list, after "self: Stack, "
        // Line 3: "contract Stack::push(self: Stack, item: Int) -> Result<Stack> {"
        // Character 34 is after "self: Stack, "
        let pos = Position::new(3, 34);
        let help = generate_signature_help(&doc, pos);

        assert!(help.is_some());
        let help = help.unwrap();
        assert_eq!(help.active_parameter, Some(1));
    }

    #[test]
    fn test_signature_help_no_contract() {
        let doc = make_doc(
            r#"
theorem simple { forall x: Bool . x }
"#,
        );

        // No contract in this document
        let pos = Position::new(1, 10);
        let help = generate_signature_help(&doc, pos);
        assert!(help.is_none());
    }

    #[test]
    fn test_signature_help_outside_parens() {
        let doc = make_doc(
            r#"
contract Foo::bar(x: Int) -> Bool {
    ensures { result }
}

theorem test { true }
"#,
        );

        // Position outside any paren context
        let pos = Position::new(5, 15);
        let help = generate_signature_help(&doc, pos);
        assert!(help.is_none());
    }

    #[test]
    fn test_parameter_info_offsets() {
        let contract = Contract {
            type_path: vec!["Foo".to_string(), "bar".to_string()],
            params: vec![
                Param {
                    name: "x".to_string(),
                    ty: Type::Named("Int".to_string()),
                },
                Param {
                    name: "y".to_string(),
                    ty: Type::Named("Bool".to_string()),
                },
            ],
            return_type: Some(Type::Named("Unit".to_string())),
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        };

        let sig = build_signature_info(&contract);

        assert_eq!(sig.label, "contract Foo::bar(x: Int, y: Bool) -> Unit");

        let params = sig.parameters.as_ref().expect("should have parameters");
        assert_eq!(params.len(), 2);

        // Check first parameter offset
        if let ParameterLabel::LabelOffsets([start, end]) = params[0].label {
            let param_text = &sig.label[start as usize..end as usize];
            assert_eq!(param_text, "x: Int");
        } else {
            panic!("Expected label offsets");
        }

        // Check second parameter offset
        if let ParameterLabel::LabelOffsets([start, end]) = params[1].label {
            let param_text = &sig.label[start as usize..end as usize];
            assert_eq!(param_text, "y: Bool");
        } else {
            panic!("Expected label offsets");
        }
    }

    // ========== MUTATION-KILLING TESTS ==========

    // Test paren_depth tracking for nested parens
    #[test]
    fn test_find_contract_context_nested_parens() {
        // Nested parens: innermost unclosed paren is the context
        // "outer(inner(a, b)" - inner's paren is unclosed
        assert_eq!(
            find_contract_context("outer(inner(a, b"),
            Some(("inner".to_string(), 1))
        );
        // Three levels deep - innermost
        assert_eq!(find_contract_context("a(b(c(d"), Some(("c".to_string(), 0)));
        // Multiple closed nested parens - foo's paren is the unclosed one
        assert_eq!(
            find_contract_context("foo(bar(), baz()"),
            Some(("foo".to_string(), 1))
        );
    }

    #[test]
    fn test_find_contract_context_paren_depth_decrement() {
        // Ensure paren_depth is correctly decremented with '('
        // If -= was replaced with /= or +=, this would fail
        assert_eq!(
            find_contract_context("foo(nested()) then bar("),
            Some(("bar".to_string(), 0))
        );
        // Multiple nested parens followed by new call
        assert_eq!(
            find_contract_context("x(a(), b()) and y("),
            Some(("y".to_string(), 0))
        );
    }

    #[test]
    fn test_find_contract_context_paren_depth_increment() {
        // ')' should increment paren_depth
        // If += was replaced with -=, matched parens wouldn't be found
        assert_eq!(
            find_contract_context("foo(bar())"),
            None // Closed paren, no open context
        );
        assert_eq!(
            find_contract_context("prefix foo("),
            Some(("foo".to_string(), 0))
        );
    }

    // Test bracket depth tracking
    #[test]
    fn test_count_commas_brackets_depth() {
        // Commas inside brackets shouldn't count
        assert_eq!(count_commas_at_top_level("[a, b]"), 0);
        assert_eq!(count_commas_at_top_level("x, [a, b], y"), 2);
        assert_eq!(count_commas_at_top_level("[a, b], [c, d]"), 1);
        // Nested brackets
        assert_eq!(count_commas_at_top_level("[[a, b], [c, d]]"), 0);
        assert_eq!(count_commas_at_top_level("a, [[b, c]]"), 1);
    }

    #[test]
    fn test_count_commas_bracket_increment() {
        // '[' should increment bracket_depth
        // If += was replaced with -=, commas inside brackets would count incorrectly
        assert_eq!(count_commas_at_top_level("[a, b, c]"), 0);
        assert_eq!(count_commas_at_top_level("x, [a, b, c], y"), 2);
    }

    #[test]
    fn test_count_commas_bracket_decrement() {
        // ']' should decrement bracket_depth back to 0 (via saturating_sub)
        // If the ']' match arm was deleted, closing brackets wouldn't decrement
        assert_eq!(count_commas_at_top_level("a, [b], c"), 2);
        assert_eq!(count_commas_at_top_level("[a], [b], [c]"), 2);
    }

    // Test find_contract_by_name with partial/full names
    #[test]
    fn test_find_contract_by_name_full_path() {
        let doc = make_doc(
            r#"
type Foo = { x: Int }

contract Foo::method1(self: Foo) -> Bool {
    requires { true }
    ensures { result }
}

contract Foo::method2(self: Foo) -> Bool {
    requires { true }
    ensures { result }
}
"#,
        );

        let spec = doc.spec.as_ref().unwrap();

        // Full path should match
        let found1 = find_contract_by_name(spec.properties.iter(), "Foo::method1");
        assert!(found1.is_some(), "Should find Foo::method1 by full path");
        assert_eq!(found1.unwrap().type_path.join("::"), "Foo::method1");

        // Just method name (last part) should also match
        let found2 = find_contract_by_name(spec.properties.iter(), "method2");
        assert!(found2.is_some(), "Should find by last path component");
        assert_eq!(found2.unwrap().type_path.join("::"), "Foo::method2");
    }

    #[test]
    fn test_find_contract_by_name_last_part_match() {
        let doc = make_doc(
            r#"
type Bar = { y: Int }

contract Bar::process(self: Bar) -> Int {
    requires { true }
    ensures { result > 0 }
}
"#,
        );

        let spec = doc.spec.as_ref().unwrap();

        // Test that last() == name check works
        let found = find_contract_by_name(spec.properties.iter(), "process");
        assert!(found.is_some(), "Should find contract by last path part");

        // Should NOT find non-existent contract
        let not_found = find_contract_by_name(spec.properties.iter(), "nonexistent");
        assert!(not_found.is_none(), "Should not find nonexistent contract");
    }

    // Test build_parameter_info offset calculations
    #[test]
    fn test_build_parameter_info_offsets_exact() {
        let contract = Contract {
            type_path: vec!["Test".to_string()],
            params: vec![Param {
                name: "a".to_string(),
                ty: Type::Named("Int".to_string()),
            }],
            return_type: None,
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        };

        let sig = build_signature_info(&contract);
        // Label: "contract Test(a: Int)"
        // paren_pos = 13 (after "contract Test"), so offset starts at 14

        let params = sig.parameters.as_ref().expect("should have parameters");
        assert_eq!(params.len(), 1);

        if let ParameterLabel::LabelOffsets([start, end]) = params[0].label {
            // The param "a: Int" should be exactly extracted
            let extracted = &sig.label[start as usize..end as usize];
            assert_eq!(extracted, "a: Int");
            // start should be paren_pos + 1 + offset within params section
            // The + operations in build_parameter_info are critical
            assert!(start > 0);
            assert!(end > start);
        } else {
            panic!("Expected label offsets");
        }
    }

    #[test]
    fn test_build_parameter_info_multiple_params() {
        let contract = Contract {
            type_path: vec!["Multi".to_string()],
            params: vec![
                Param {
                    name: "first".to_string(),
                    ty: Type::Named("String".to_string()),
                },
                Param {
                    name: "second".to_string(),
                    ty: Type::Named("Bool".to_string()),
                },
                Param {
                    name: "third".to_string(),
                    ty: Type::Named("Int".to_string()),
                },
            ],
            return_type: Some(Type::Named("Unit".to_string())),
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        };

        let sig = build_signature_info(&contract);
        let params = sig.parameters.as_ref().expect("should have parameters");
        assert_eq!(params.len(), 3);

        // Each parameter should be correctly located
        for (i, param) in params.iter().enumerate() {
            if let ParameterLabel::LabelOffsets([start, end]) = param.label {
                let extracted = &sig.label[start as usize..end as usize];
                match i {
                    0 => assert_eq!(extracted, "first: String"),
                    1 => assert_eq!(extracted, "second: Bool"),
                    2 => assert_eq!(extracted, "third: Int"),
                    _ => panic!("Unexpected parameter index"),
                }
            } else {
                panic!("Expected label offsets for param {}", i);
            }
        }
    }

    #[test]
    fn test_build_parameter_info_paren_pos_offset() {
        // Test that paren_pos + 1 calculation is correct
        // If + was replaced with -, offset would be wrong
        let contract = Contract {
            type_path: vec!["LongTypeName".to_string(), "method".to_string()],
            params: vec![Param {
                name: "x".to_string(),
                ty: Type::Named("Int".to_string()),
            }],
            return_type: None,
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        };

        let sig = build_signature_info(&contract);
        // Label: "contract LongTypeName::method(x: Int)"
        let params = sig.parameters.as_ref().expect("should have parameters");

        if let ParameterLabel::LabelOffsets([start, end]) = params[0].label {
            let extracted = &sig.label[start as usize..end as usize];
            assert_eq!(
                extracted, "x: Int",
                "Parameter should be correctly extracted"
            );
            // The character at start should be 'x', not '(' or anything else
            assert_eq!(
                sig.label.chars().nth(start as usize),
                Some('x'),
                "Start offset should point to parameter name"
            );
        } else {
            panic!("Expected label offsets");
        }
    }

    // Test extract_contract_name edge cases
    #[test]
    fn test_extract_contract_name_with_colons() {
        // Ensure :: is correctly included in name
        assert_eq!(
            extract_contract_name("Module::Type::method"),
            Some("Module::Type::method".to_string())
        );
        // Colons at end should be excluded
        assert_eq!(extract_contract_name("::invalid"), None);
    }

    #[test]
    fn test_extract_contract_name_decrement() {
        // Test the start -= 1 loop
        // If -= was replaced with /=, the loop would behave incorrectly
        assert_eq!(
            extract_contract_name("prefix_foo"),
            Some("prefix_foo".to_string())
        );
        assert_eq!(
            extract_contract_name("  spaced_name"),
            Some("spaced_name".to_string())
        );
        // Single char
        assert_eq!(extract_contract_name("x"), Some("x".to_string()));
        // Underscore in name
        assert_eq!(
            extract_contract_name("my_function_name"),
            Some("my_function_name".to_string())
        );
    }
}
