//! Code actions for USL
//!
//! Provides quick fixes and refactoring suggestions for USL documents.
//! Code actions are available when there are diagnostics (errors/warnings)
//! that can be automatically fixed.

use crate::commands::{
    COMMAND_COMPILATION_GUIDANCE, COMMAND_EXPLAIN_DIAGNOSTIC, COMMAND_SUGGEST_TACTICS,
};
use crate::document::Document;
use crate::symbols::format_expr;
use serde_json::json;
use std::collections::HashMap;
use tower_lsp::lsp_types::{
    CodeAction, CodeActionKind, CodeActionOrCommand, Command, Diagnostic, Position, Range,
    TextEdit, WorkspaceEdit,
};

/// Code action context containing diagnostics and document info
pub struct CodeActionContext<'a> {
    /// The document to generate actions for
    pub doc: &'a Document,
    /// The range requested by the client
    pub range: Range,
    /// Diagnostics in the requested range
    pub diagnostics: &'a [Diagnostic],
}

/// Generate code actions for a document based on diagnostics
#[must_use]
pub fn generate_code_actions(ctx: &CodeActionContext) -> Vec<CodeActionOrCommand> {
    let mut actions = Vec::new();

    // Generate actions for each diagnostic in the range
    for diagnostic in ctx.diagnostics {
        // Check if this diagnostic intersects with the requested range
        if !ranges_intersect(&diagnostic.range, &ctx.range) {
            continue;
        }

        let message = &diagnostic.message;

        if let Some(action) = expert_explain_action(diagnostic) {
            actions.push(CodeActionOrCommand::CodeAction(action));
        }

        // Try each action generator
        if let Some(action) = create_type_action(ctx.doc, diagnostic, message) {
            actions.push(CodeActionOrCommand::CodeAction(action));
        }

        if let Some(action) = suggest_field_action(ctx.doc, diagnostic, message) {
            actions.push(CodeActionOrCommand::CodeAction(action));
        }
    }

    // Add refactoring actions for the selected range
    actions.extend(refactoring_actions(ctx));

    // Add tactic suggestion action when in theorem/proof context
    if let Some(action) = tactic_suggestion_action(ctx) {
        actions.push(CodeActionOrCommand::CodeAction(action));
    }

    // Add compilation guidance action when inside a property context
    if let Some(action) = compilation_guidance_action(ctx) {
        actions.push(CodeActionOrCommand::CodeAction(action));
    }

    actions
}

/// Check if two ranges intersect
fn ranges_intersect(a: &Range, b: &Range) -> bool {
    // Ranges intersect if neither ends before the other starts
    !(a.end.line < b.start.line
        || (a.end.line == b.start.line && a.end.character < b.start.character)
        || b.end.line < a.start.line
        || (b.end.line == a.start.line && b.end.character < a.start.character))
}

/// Create a code action that routes the diagnostic to the RAG expert system.
fn expert_explain_action(diagnostic: &Diagnostic) -> Option<CodeAction> {
    let command = Command {
        title: "Explain with DashProve Expert".to_string(),
        command: COMMAND_EXPLAIN_DIAGNOSTIC.to_string(),
        arguments: Some(vec![json!(diagnostic.message.clone())]),
    };

    Some(CodeAction {
        title: "Explain with DashProve Expert".to_string(),
        kind: Some(CodeActionKind::QUICKFIX),
        diagnostics: Some(vec![diagnostic.clone()]),
        edit: None,
        command: Some(command),
        is_preferred: None,
        disabled: None,
        data: None,
    })
}

/// Create a code action for tactic suggestions when inside a theorem/proof context.
fn tactic_suggestion_action(ctx: &CodeActionContext) -> Option<CodeAction> {
    let spec = ctx.doc.spec.as_ref()?;

    // Find if we're inside a theorem property
    let theorem_info = find_theorem_at_position(ctx.doc, ctx.range.start, spec)?;

    // Create command with goal description and backend
    let command = Command {
        title: "Suggest tactics for this proof".to_string(),
        command: COMMAND_SUGGEST_TACTICS.to_string(),
        arguments: Some(vec![
            json!(theorem_info.goal),
            json!(theorem_info.backend),
            json!(theorem_info.context),
        ]),
    };

    Some(CodeAction {
        title: format!("Suggest tactics for '{}'", theorem_info.name),
        kind: Some(CodeActionKind::SOURCE),
        diagnostics: None,
        edit: None,
        command: Some(command),
        is_preferred: None,
        disabled: None,
        data: None,
    })
}

/// Information about a theorem at a position
struct TheoremInfo {
    name: String,
    goal: String,
    backend: String,
    context: String,
}

/// Information about any property for compilation guidance
struct PropertyInfo {
    name: String,
    kind: String,
    target_backend: String,
    source_code: String,
}

/// Create a code action for compilation guidance when inside any property context.
fn compilation_guidance_action(ctx: &CodeActionContext) -> Option<CodeAction> {
    let spec = ctx.doc.spec.as_ref()?;

    // Find if we're inside any property
    let property_info = find_property_at_position(ctx.doc, ctx.range.start, spec)?;

    // Create command with property info for compilation guidance
    let command = Command {
        title: "Get compilation guidance".to_string(),
        command: COMMAND_COMPILATION_GUIDANCE.to_string(),
        arguments: Some(vec![
            json!(property_info.kind),
            json!(property_info.target_backend),
            json!(property_info.source_code),
        ]),
    };

    Some(CodeAction {
        title: format!(
            "Get compilation guidance for '{}' (→ {})",
            property_info.name, property_info.target_backend
        ),
        kind: Some(CodeActionKind::SOURCE),
        diagnostics: None,
        edit: None,
        command: Some(command),
        is_preferred: None,
        disabled: None,
        data: None,
    })
}

/// Find any property containing the given position and extract its info
fn find_property_at_position(
    doc: &Document,
    pos: Position,
    spec: &dashprove_usl::Spec,
) -> Option<PropertyInfo> {
    use dashprove_usl::Property;

    for prop in &spec.properties {
        let (kind, name, target_backend) = match prop {
            Property::Theorem(t) => ("theorem", t.name.clone(), "lean4"),
            Property::Temporal(t) => ("temporal", t.name.clone(), "tlaplus"),
            Property::Contract(c) => ("contract", c.type_path.join("::"), "kani"),
            Property::Invariant(i) => ("invariant", i.name.clone(), "lean4"),
            Property::Refinement(r) => ("refinement", r.name.clone(), "lean4"),
            Property::Probabilistic(p) => ("probabilistic", p.name.clone(), "probabilistic"),
            Property::Security(s) => ("security", s.name.clone(), "security"),
            Property::Semantic(s) => ("semantic", s.name.clone(), "semantic"),
            Property::PlatformApi(p) => ("platform_api", p.name.clone(), "platform_api"),
            Property::Bisimulation(b) => ("bisimulation", b.name.clone(), "bisim"),
            Property::Version(v) => ("version", v.name.clone(), "lean4"),
            Property::Capability(c) => ("capability", c.name.clone(), "lean4"),
            Property::DistributedInvariant(d) => {
                ("distributed_invariant", d.name.clone(), "tlaplus")
            }
            Property::DistributedTemporal(d) => ("distributed_temporal", d.name.clone(), "tlaplus"),
            Property::Composed(c) => ("composed", c.name.clone(), "lean4"),
            Property::ImprovementProposal(p) => ("improvement_proposal", p.name.clone(), "lean4"),
            Property::VerificationGate(g) => ("verification_gate", g.name.clone(), "lean4"),
            Property::Rollback(r) => ("rollback", r.name.clone(), "lean4"),
        };

        // Find the pattern for this property
        let pattern = if kind == "contract" {
            format!("contract {}", name)
        } else {
            format!("{} {}", kind, name)
        };

        if let Some(start_offset) = doc.text.find(&pattern) {
            let (start_line, _) = doc.offset_to_position(start_offset);

            // Find the end of this property (closing brace)
            let rest = &doc.text[start_offset..];
            if let Some(end_rel) = find_closing_brace(rest) {
                let (end_line, _) = doc.offset_to_position(start_offset + end_rel);

                if pos.line >= start_line && pos.line <= end_line {
                    // Extract the source code of this property
                    let source_code = rest[..=end_rel].to_string();

                    return Some(PropertyInfo {
                        name,
                        kind: kind.to_string(),
                        target_backend: target_backend.to_string(),
                        source_code,
                    });
                }
            }
        }
    }

    None
}

/// Find the theorem containing the given position and extract its info
fn find_theorem_at_position(
    doc: &Document,
    pos: Position,
    spec: &dashprove_usl::Spec,
) -> Option<TheoremInfo> {
    use dashprove_usl::Property;

    // Get line text for context
    let lines: Vec<&str> = doc.text.lines().collect();
    let current_line = lines.get(pos.line as usize).unwrap_or(&"");

    // Look through properties to find theorems
    for prop in &spec.properties {
        if let Property::Theorem(theorem) = prop {
            // Check if position is within this theorem's range
            // We search for the theorem in the document text
            let pattern = format!("theorem {}", theorem.name);
            if let Some(start_offset) = doc.text.find(&pattern) {
                let (start_line, _) = doc.offset_to_position(start_offset);

                // Find the end of this theorem (closing brace)
                let rest = &doc.text[start_offset..];
                if let Some(end_rel) = find_closing_brace(rest) {
                    let (end_line, _) = doc.offset_to_position(start_offset + end_rel);

                    if pos.line >= start_line && pos.line <= end_line {
                        // We're inside this theorem
                        return Some(TheoremInfo {
                            name: theorem.name.clone(),
                            goal: format_theorem_goal(theorem),
                            backend: "lean4".to_string(), // Default to Lean4 for theorems
                            context: current_line.to_string(),
                        });
                    }
                }
            }
        }
    }

    None
}

/// Find the position of the closing brace for a construct
fn find_closing_brace(text: &str) -> Option<usize> {
    let mut depth = 0;
    let mut found_open = false;

    for (i, c) in text.char_indices() {
        if c == '{' {
            found_open = true;
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if found_open && depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Format the theorem goal for the tactic suggestion query
fn format_theorem_goal(theorem: &dashprove_usl::Theorem) -> String {
    // Format the body expression as the goal
    format_expr(&theorem.body)
}

/// Create a code action to define a new type when "Unknown type" error occurs
fn create_type_action(
    doc: &Document,
    diagnostic: &Diagnostic,
    message: &str,
) -> Option<CodeAction> {
    // Match "Unknown type: TypeName" pattern
    if !message.starts_with("Unknown type:") {
        return None;
    }

    let type_name = message.strip_prefix("Unknown type:")?.trim();
    if type_name.is_empty() {
        return None;
    }

    // Don't suggest creating built-in types
    let builtins = [
        "Bool", "Int", "Float", "String", "Unit", "Set", "List", "Map", "Relation", "Result",
    ];
    if builtins.contains(&type_name) {
        return None;
    }

    // Find the best insertion point (after existing types, or at the start)
    let insert_position = find_type_insert_position(doc);

    // Generate the type definition template
    let type_def = format!("type {} = {{ /* fields */ }}\n\n", type_name);

    let edit = TextEdit {
        range: Range {
            start: insert_position,
            end: insert_position,
        },
        new_text: type_def,
    };

    let mut changes = HashMap::new();
    changes.insert(doc.uri.clone(), vec![edit]);

    Some(CodeAction {
        title: format!("Create type '{}'", type_name),
        kind: Some(CodeActionKind::QUICKFIX),
        diagnostics: Some(vec![diagnostic.clone()]),
        edit: Some(WorkspaceEdit {
            changes: Some(changes),
            document_changes: None,
            change_annotations: None,
        }),
        command: None,
        is_preferred: Some(true),
        disabled: None,
        data: None,
    })
}

/// Find the position to insert a new type definition
fn find_type_insert_position(doc: &Document) -> Position {
    // If we have a spec, find the last type definition and insert after it
    if let Some(ref spec) = doc.spec {
        if let Some(last_type) = spec.types.last() {
            // Find the end of this type definition in the source
            if let Some(pos) = find_type_end_position(doc, &last_type.name) {
                return pos;
            }
        }
    }

    // Default: insert at the beginning of the document
    Position::new(0, 0)
}

/// Find the end position of a type definition
fn find_type_end_position(doc: &Document, type_name: &str) -> Option<Position> {
    let pattern = format!("type {}", type_name);
    let start = doc.text.find(&pattern)?;

    // Find the closing brace
    let rest = &doc.text[start..];
    let mut brace_depth = 0;
    let mut found_open = false;

    for (i, ch) in rest.char_indices() {
        if ch == '{' {
            found_open = true;
            brace_depth += 1;
        } else if ch == '}' {
            brace_depth -= 1;
            if found_open && brace_depth == 0 {
                // Found the closing brace, position after it (and any newline)
                let end_offset = start + i + 1;
                let after = &doc.text[end_offset..];
                let skip = after
                    .chars()
                    .take_while(|c| *c == '\n' || *c == '\r')
                    .count();
                let (line, character) = doc.offset_to_position(end_offset + skip);
                return Some(Position::new(line, character));
            }
        }
    }

    None
}

/// Suggest valid field names when an invalid field access error occurs
fn suggest_field_action(
    doc: &Document,
    diagnostic: &Diagnostic,
    message: &str,
) -> Option<CodeAction> {
    // Match "Invalid field access: type X has no field Y" pattern
    if !message.starts_with("Invalid field access:") {
        return None;
    }

    // Parse the error message
    let rest = message.strip_prefix("Invalid field access:")?.trim();
    let parts: Vec<&str> = rest.split(" has no field ").collect();
    if parts.len() != 2 {
        return None;
    }

    let type_part = parts[0].strip_prefix("type ")?.trim();
    let invalid_field = parts[1].trim();

    // Get the type definition to find valid fields
    let spec = doc.spec.as_ref()?;
    let type_def = spec.types.iter().find(|t| t.name == type_part)?;

    if type_def.fields.is_empty() {
        return None;
    }

    // Find the most similar field name (simple edit distance check)
    let best_match = type_def
        .fields
        .iter()
        .min_by_key(|f| levenshtein_distance(&f.name, invalid_field))?;

    // Only suggest if reasonably similar
    let distance = levenshtein_distance(&best_match.name, invalid_field);
    if distance > 3 {
        return None;
    }

    // Create the fix
    let edit = TextEdit {
        range: diagnostic.range,
        new_text: best_match.name.clone(),
    };

    let mut changes = HashMap::new();
    changes.insert(doc.uri.clone(), vec![edit]);

    Some(CodeAction {
        title: format!("Replace with '{}'", best_match.name),
        kind: Some(CodeActionKind::QUICKFIX),
        diagnostics: Some(vec![diagnostic.clone()]),
        edit: Some(WorkspaceEdit {
            changes: Some(changes),
            document_changes: None,
            change_annotations: None,
        }),
        command: None,
        is_preferred: Some(true),
        disabled: None,
        data: None,
    })
}

/// Simple Levenshtein distance for fuzzy matching
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut dp = vec![vec![0; n + 1]; m + 1];

    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n]
}

/// Generate refactoring actions for the selected range
fn refactoring_actions(ctx: &CodeActionContext) -> Vec<CodeActionOrCommand> {
    let mut actions = Vec::new();

    // Check if a type name is selected - offer extract type
    if let Some(word) = ctx
        .doc
        .word_at_position(ctx.range.start.line, ctx.range.start.character)
    {
        // Only offer extract if it looks like a type usage (capitalized)
        if word.chars().next().is_some_and(|c| c.is_uppercase()) {
            if let Some(ref spec) = ctx.doc.spec {
                // Check if this is NOT already a defined type
                let is_defined = spec.types.iter().any(|t| t.name == word);
                if !is_defined {
                    // Check if it's not a builtin
                    let builtins = [
                        "Bool", "Int", "Float", "String", "Unit", "Set", "List", "Map", "Relation",
                        "Result",
                    ];
                    if !builtins.contains(&word) {
                        let insert_pos = find_type_insert_position(ctx.doc);
                        let type_def = format!("type {} = {{ /* fields */ }}\n\n", word);

                        let edit = TextEdit {
                            range: Range {
                                start: insert_pos,
                                end: insert_pos,
                            },
                            new_text: type_def,
                        };

                        let mut changes = HashMap::new();
                        changes.insert(ctx.doc.uri.clone(), vec![edit]);

                        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                            title: format!("Extract type '{}'", word),
                            kind: Some(CodeActionKind::REFACTOR_EXTRACT),
                            diagnostics: None,
                            edit: Some(WorkspaceEdit {
                                changes: Some(changes),
                                document_changes: None,
                                change_annotations: None,
                            }),
                            command: None,
                            is_preferred: None,
                            disabled: None,
                            data: None,
                        }));
                    }
                }
            }
        }
    }

    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower_lsp::lsp_types::{DiagnosticSeverity, Url};

    fn make_doc(text: &str) -> Document {
        Document::new(Url::parse("file:///test.usl").unwrap(), 1, text.to_string())
    }

    fn make_diagnostic(message: &str, start_line: u32, start_char: u32) -> Diagnostic {
        Diagnostic {
            range: Range {
                start: Position::new(start_line, start_char),
                end: Position::new(start_line, start_char + 5),
            },
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("dashprove-usl".to_string()),
            message: message.to_string(),
            related_information: None,
            tags: None,
            data: None,
        }
    }

    #[test]
    fn test_create_type_action_for_unknown_type() {
        let doc = make_doc("theorem foo { forall x: MyType . true }");
        let diagnostic = make_diagnostic("Unknown type: MyType", 0, 23);

        let action = create_type_action(&doc, &diagnostic, &diagnostic.message);
        assert!(action.is_some());

        let action = action.unwrap();
        assert_eq!(action.title, "Create type 'MyType'");
        assert!(action.edit.is_some());
    }

    #[test]
    fn test_no_action_for_builtin_type() {
        let doc = make_doc("theorem foo { forall x: Int . true }");
        let diagnostic = make_diagnostic("Unknown type: Int", 0, 23);

        let action = create_type_action(&doc, &diagnostic, &diagnostic.message);
        assert!(action.is_none());
    }

    #[test]
    fn test_suggest_field_action() {
        let doc = make_doc("type Person = { name: String, age: Int }\ntheorem foo { forall p: Person . p.nme == \"\" }");
        let diagnostic =
            make_diagnostic("Invalid field access: type Person has no field nme", 1, 35);

        let action = suggest_field_action(&doc, &diagnostic, &diagnostic.message);
        assert!(action.is_some());

        let action = action.unwrap();
        assert_eq!(action.title, "Replace with 'name'");
    }

    #[test]
    fn test_expert_explain_action_added() {
        let doc = make_doc("theorem foo { true }");
        let diagnostic = make_diagnostic("Type mismatch", 0, 10);
        let diagnostics = vec![diagnostic.clone()];
        let ctx = CodeActionContext {
            doc: &doc,
            range: diagnostic.range,
            diagnostics: &diagnostics,
        };

        let actions = generate_code_actions(&ctx);
        let has_expert = actions.iter().any(|action| match action {
            CodeActionOrCommand::CodeAction(code_action) => code_action
                .command
                .as_ref()
                .is_some_and(|cmd| cmd.command == COMMAND_EXPLAIN_DIAGNOSTIC),
            CodeActionOrCommand::Command(cmd) => cmd.command == COMMAND_EXPLAIN_DIAGNOSTIC,
        });

        assert!(
            has_expert,
            "should include expert diagnostic explanation action"
        );
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("abc", "ab"), 1);
        assert_eq!(levenshtein_distance("abc", "abd"), 1);
        assert_eq!(levenshtein_distance("name", "nme"), 1);
        assert_eq!(levenshtein_distance("name", "nmae"), 2);
    }

    #[test]
    fn test_ranges_intersect() {
        let r1 = Range {
            start: Position::new(0, 0),
            end: Position::new(0, 10),
        };
        let r2 = Range {
            start: Position::new(0, 5),
            end: Position::new(0, 15),
        };
        let r3 = Range {
            start: Position::new(1, 0),
            end: Position::new(1, 10),
        };

        assert!(ranges_intersect(&r1, &r2));
        assert!(ranges_intersect(&r2, &r1));
        assert!(!ranges_intersect(&r1, &r3));
    }

    #[test]
    fn test_find_type_insert_position_empty_doc() {
        let doc = make_doc("");
        let pos = find_type_insert_position(&doc);
        assert_eq!(pos.line, 0);
        assert_eq!(pos.character, 0);
    }

    #[test]
    fn test_find_type_insert_position_after_existing_type() {
        let doc = make_doc("type Foo = { x: Int }\n\ntheorem bar { true }");
        let pos = find_type_insert_position(&doc);
        // Should be after the type definition
        assert!(pos.line >= 1 || pos.character > 0);
    }

    #[test]
    fn test_generate_code_actions_with_diagnostics() {
        let doc = make_doc("theorem foo { forall x: CustomType . true }");
        let diagnostics = vec![make_diagnostic("Unknown type: CustomType", 0, 23)];

        let ctx = CodeActionContext {
            doc: &doc,
            range: Range {
                start: Position::new(0, 0),
                end: Position::new(0, 50),
            },
            diagnostics: &diagnostics,
        };

        let actions = generate_code_actions(&ctx);
        assert!(!actions.is_empty());
    }

    #[test]
    fn test_refactoring_action_extract_type() {
        // "NewType" starts at position 23 in "theorem foo { forall x: NewType . true }"
        let doc = make_doc("theorem foo { forall x: NewType . true }");
        let diagnostics = vec![];

        // Position must be within the word "NewType"
        let ctx = CodeActionContext {
            doc: &doc,
            range: Range {
                start: Position::new(0, 24), // Inside "NewType"
                end: Position::new(0, 24),
            },
            diagnostics: &diagnostics,
        };

        let actions = refactoring_actions(&ctx);
        // Should offer to extract the undefined type
        let extract_actions: Vec<_> = actions
            .iter()
            .filter(|a| match a {
                CodeActionOrCommand::CodeAction(ca) => {
                    ca.kind == Some(CodeActionKind::REFACTOR_EXTRACT)
                }
                _ => false,
            })
            .collect();
        assert!(
            !extract_actions.is_empty(),
            "Should have extract type action for NewType"
        );
    }

    #[test]
    fn test_compilation_guidance_action_for_theorem() {
        let doc = make_doc("theorem test_theorem { true }");
        let diagnostics = vec![];

        let ctx = CodeActionContext {
            doc: &doc,
            range: Range {
                start: Position::new(0, 10), // Inside the theorem
                end: Position::new(0, 10),
            },
            diagnostics: &diagnostics,
        };

        let actions = generate_code_actions(&ctx);
        let has_guidance = actions.iter().any(|action| match action {
            CodeActionOrCommand::CodeAction(code_action) => code_action
                .command
                .as_ref()
                .is_some_and(|cmd| cmd.command == COMMAND_COMPILATION_GUIDANCE),
            CodeActionOrCommand::Command(cmd) => cmd.command == COMMAND_COMPILATION_GUIDANCE,
        });

        assert!(
            has_guidance,
            "should include compilation guidance action for theorem"
        );
    }

    #[test]
    fn test_compilation_guidance_action_for_contract() {
        let doc = make_doc(
            "contract Stack::push(self: Stack, item: Int) -> Result<Unit> {\n  requires { true }\n  ensures { true }\n}",
        );
        let diagnostics = vec![];

        let ctx = CodeActionContext {
            doc: &doc,
            range: Range {
                start: Position::new(1, 5), // Inside the contract
                end: Position::new(1, 5),
            },
            diagnostics: &diagnostics,
        };

        let actions = generate_code_actions(&ctx);
        let guidance_action = actions.iter().find(|action| match action {
            CodeActionOrCommand::CodeAction(code_action) => code_action
                .command
                .as_ref()
                .is_some_and(|cmd| cmd.command == COMMAND_COMPILATION_GUIDANCE),
            CodeActionOrCommand::Command(cmd) => cmd.command == COMMAND_COMPILATION_GUIDANCE,
        });

        assert!(
            guidance_action.is_some(),
            "should include compilation guidance action for contract"
        );

        // Verify the action title mentions kani
        if let Some(CodeActionOrCommand::CodeAction(action)) = guidance_action {
            assert!(
                action.title.contains("kani"),
                "contract guidance should target kani backend"
            );
        }
    }

    #[test]
    fn test_find_property_at_position_returns_none_outside() {
        let doc = make_doc("// comment\ntheorem test { true }");
        let spec = doc.spec.as_ref().unwrap();

        // Position at the comment line (outside theorem)
        let result = find_property_at_position(&doc, Position::new(0, 5), spec);
        assert!(
            result.is_none(),
            "should return None when outside any property"
        );
    }

    // Additional mutation-killing tests

    #[test]
    fn test_find_closing_brace_simple() {
        assert_eq!(find_closing_brace("{ }"), Some(2));
        assert_eq!(find_closing_brace("{}"), Some(1));
    }

    #[test]
    fn test_find_closing_brace_nested() {
        assert_eq!(find_closing_brace("{ { } }"), Some(6));
        assert_eq!(find_closing_brace("{ a { b } c }"), Some(12));
    }

    #[test]
    fn test_find_closing_brace_no_open() {
        assert_eq!(find_closing_brace("}"), None);
        assert_eq!(find_closing_brace("no braces"), None);
    }

    #[test]
    fn test_find_closing_brace_unmatched() {
        assert_eq!(find_closing_brace("{ { }"), None); // Missing closing
    }

    #[test]
    fn test_ranges_intersect_edge_cases() {
        // Same range
        let r = Range {
            start: Position::new(0, 0),
            end: Position::new(0, 10),
        };
        assert!(ranges_intersect(&r, &r));

        // Adjacent but not overlapping (end of r1 == start of r2)
        let r1 = Range {
            start: Position::new(0, 0),
            end: Position::new(0, 5),
        };
        let r2 = Range {
            start: Position::new(0, 5),
            end: Position::new(0, 10),
        };
        assert!(ranges_intersect(&r1, &r2)); // They share a point

        // Same line, completely separate
        let r3 = Range {
            start: Position::new(0, 0),
            end: Position::new(0, 3),
        };
        let r4 = Range {
            start: Position::new(0, 5),
            end: Position::new(0, 10),
        };
        assert!(!ranges_intersect(&r3, &r4));

        // Multi-line ranges that overlap
        let r5 = Range {
            start: Position::new(0, 0),
            end: Position::new(2, 10),
        };
        let r6 = Range {
            start: Position::new(1, 5),
            end: Position::new(3, 0),
        };
        assert!(ranges_intersect(&r5, &r6));
    }

    #[test]
    fn test_levenshtein_distance_edge_cases() {
        // One empty string
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);

        // Single character difference
        assert_eq!(levenshtein_distance("a", "b"), 1);

        // Insertions
        assert_eq!(levenshtein_distance("ac", "abc"), 1);

        // Deletions
        assert_eq!(levenshtein_distance("abc", "ac"), 1);

        // Transposition (counted as 2 operations in basic levenshtein)
        assert_eq!(levenshtein_distance("ab", "ba"), 2);
    }

    #[test]
    fn test_create_type_action_empty_type_name() {
        let doc = make_doc("theorem foo { true }");
        let diagnostic = make_diagnostic("Unknown type:   ", 0, 10);

        let action = create_type_action(&doc, &diagnostic, &diagnostic.message);
        assert!(action.is_none(), "empty type name should not create action");
    }

    #[test]
    fn test_create_type_action_all_builtins() {
        let builtins = [
            "Bool", "Int", "Float", "String", "Unit", "Set", "List", "Map", "Relation", "Result",
        ];
        let doc = make_doc("theorem foo { true }");

        for builtin in builtins {
            let message = format!("Unknown type: {}", builtin);
            let diagnostic = make_diagnostic(&message, 0, 10);
            let action = create_type_action(&doc, &diagnostic, &diagnostic.message);
            assert!(
                action.is_none(),
                "builtin type {} should not create action",
                builtin
            );
        }
    }

    #[test]
    fn test_suggest_field_action_no_similar_field() {
        let doc = make_doc(
            "type Person = { name: String, age: Int }\ntheorem foo { forall p: Person . p.abcdefg == \"\" }",
        );
        let diagnostic = make_diagnostic(
            "Invalid field access: type Person has no field abcdefg",
            1,
            35,
        );

        let action = suggest_field_action(&doc, &diagnostic, &diagnostic.message);
        // "abcdefg" is too different from "name" or "age" (distance > 3)
        assert!(
            action.is_none(),
            "should not suggest for very different field names"
        );
    }

    #[test]
    fn test_suggest_field_action_empty_fields() {
        let doc =
            make_doc("type Empty = { }\ntheorem foo { forall e: Empty . e.something == \"\" }");
        let diagnostic = make_diagnostic(
            "Invalid field access: type Empty has no field something",
            1,
            34,
        );

        let action = suggest_field_action(&doc, &diagnostic, &diagnostic.message);
        assert!(
            action.is_none(),
            "should not suggest when type has no fields"
        );
    }

    #[test]
    fn test_suggest_field_action_invalid_message_format() {
        let doc = make_doc("type Person = { name: String }");
        let diagnostic = make_diagnostic("Some other error", 0, 10);

        let action = suggest_field_action(&doc, &diagnostic, &diagnostic.message);
        assert!(
            action.is_none(),
            "should return None for non-field-access errors"
        );
    }

    #[test]
    fn test_find_type_end_position_not_found() {
        let doc = make_doc("theorem foo { true }");
        let result = find_type_end_position(&doc, "NonExistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_find_property_at_position_inside_theorem() {
        let doc = make_doc("theorem test { true }");
        let spec = doc.spec.as_ref().unwrap();

        // Position inside the theorem
        let result = find_property_at_position(&doc, Position::new(0, 10), spec);
        assert!(result.is_some(), "should find property when inside");

        let info = result.unwrap();
        assert_eq!(info.name, "test");
        assert_eq!(info.kind, "theorem");
        assert_eq!(info.target_backend, "lean4");
    }

    #[test]
    fn test_find_theorem_at_position() {
        let doc = make_doc("theorem my_theorem { forall x: Int . x == x }");
        let spec = doc.spec.as_ref().unwrap();

        let result = find_theorem_at_position(&doc, Position::new(0, 10), spec);
        assert!(result.is_some());

        let info = result.unwrap();
        assert_eq!(info.name, "my_theorem");
        assert_eq!(info.backend, "lean4");
    }

    #[test]
    fn test_find_theorem_at_position_outside() {
        let doc = make_doc("// comment\ntheorem my_theorem { true }");
        let spec = doc.spec.as_ref().unwrap();

        // Position at the comment line
        let result = find_theorem_at_position(&doc, Position::new(0, 3), spec);
        assert!(result.is_none());
    }

    #[test]
    fn test_tactic_suggestion_action_not_in_theorem() {
        let doc = make_doc("invariant inv { true }");
        let diagnostics = vec![];

        let ctx = CodeActionContext {
            doc: &doc,
            range: Range {
                start: Position::new(0, 10),
                end: Position::new(0, 10),
            },
            diagnostics: &diagnostics,
        };

        let action = tactic_suggestion_action(&ctx);
        // Invariant is not a theorem, so no tactic action
        assert!(
            action.is_none(),
            "should not offer tactics for non-theorem properties"
        );
    }

    #[test]
    fn test_tactic_suggestion_action_in_theorem() {
        let doc = make_doc("theorem test { forall x: Int . x == x }");
        let diagnostics = vec![];

        let ctx = CodeActionContext {
            doc: &doc,
            range: Range {
                start: Position::new(0, 10),
                end: Position::new(0, 10),
            },
            diagnostics: &diagnostics,
        };

        let action = tactic_suggestion_action(&ctx);
        assert!(action.is_some(), "should offer tactics for theorem");

        let action = action.unwrap();
        assert!(action.title.contains("test"));
        assert_eq!(action.kind, Some(CodeActionKind::SOURCE));
    }

    #[test]
    fn test_refactoring_actions_lowercase_identifier() {
        // lowercase identifiers should not trigger type extraction
        let doc = make_doc("theorem foo { forall x: MyType . lowercase == 5 }");
        let diagnostics = vec![];

        let ctx = CodeActionContext {
            doc: &doc,
            range: Range {
                start: Position::new(0, 35), // Inside "lowercase"
                end: Position::new(0, 35),
            },
            diagnostics: &diagnostics,
        };

        let actions = refactoring_actions(&ctx);
        let extract_actions: Vec<_> = actions
            .iter()
            .filter(|a| match a {
                CodeActionOrCommand::CodeAction(ca) => {
                    ca.kind == Some(CodeActionKind::REFACTOR_EXTRACT)
                }
                _ => false,
            })
            .collect();
        assert!(
            extract_actions.is_empty(),
            "lowercase identifiers should not trigger extract type"
        );
    }

    #[test]
    fn test_refactoring_actions_already_defined_type() {
        let doc =
            make_doc("type Person = { name: String }\ntheorem foo { forall p: Person . true }");
        let diagnostics = vec![];

        let ctx = CodeActionContext {
            doc: &doc,
            range: Range {
                start: Position::new(1, 24), // Inside "Person"
                end: Position::new(1, 24),
            },
            diagnostics: &diagnostics,
        };

        let actions = refactoring_actions(&ctx);
        let extract_actions: Vec<_> = actions
            .iter()
            .filter(|a| match a {
                CodeActionOrCommand::CodeAction(ca) => {
                    ca.kind == Some(CodeActionKind::REFACTOR_EXTRACT)
                }
                _ => false,
            })
            .collect();
        assert!(
            extract_actions.is_empty(),
            "already defined types should not trigger extract"
        );
    }

    #[test]
    fn test_expert_explain_action_created() {
        let diagnostic = make_diagnostic("Type mismatch: expected Int, got String", 0, 10);
        let action = expert_explain_action(&diagnostic);

        assert!(action.is_some());
        let action = action.unwrap();
        assert!(action.title.contains("Expert"));
        assert!(action.command.is_some());
        assert_eq!(
            action.command.as_ref().unwrap().command,
            COMMAND_EXPLAIN_DIAGNOSTIC
        );
    }

    #[test]
    fn test_compilation_guidance_for_all_property_types() {
        // Test temporal property
        let doc = make_doc("temporal t { always(true) }");
        let diagnostics = vec![];
        let ctx = CodeActionContext {
            doc: &doc,
            range: Range {
                start: Position::new(0, 10),
                end: Position::new(0, 10),
            },
            diagnostics: &diagnostics,
        };
        let action = compilation_guidance_action(&ctx);
        assert!(action.is_some());
        if let Some(a) = action {
            assert!(a.title.contains("tlaplus"));
        }

        // Test invariant property
        let doc = make_doc("invariant inv { true }");
        let ctx = CodeActionContext {
            doc: &doc,
            range: Range {
                start: Position::new(0, 10),
                end: Position::new(0, 10),
            },
            diagnostics: &diagnostics,
        };
        let action = compilation_guidance_action(&ctx);
        assert!(action.is_some());
        if let Some(a) = action {
            assert!(a.title.contains("lean4"));
        }
    }

    // Additional mutation-killing tests for remaining mutants

    #[test]
    fn test_ranges_intersect_character_boundary_same_line() {
        // Kills mutant line 79: replace == with != in ranges_intersect
        // When b ends at same line/character as a starts, they should intersect
        let r1 = Range {
            start: Position::new(5, 10),
            end: Position::new(5, 20),
        };
        let r2 = Range {
            start: Position::new(5, 5),
            end: Position::new(5, 10), // ends exactly where r1 starts
        };
        assert!(
            ranges_intersect(&r1, &r2),
            "ranges sharing a point should intersect"
        );

        // And when a ends at same line as b starts
        let r3 = Range {
            start: Position::new(5, 0),
            end: Position::new(5, 5),
        };
        let r4 = Range {
            start: Position::new(5, 5),
            end: Position::new(5, 10),
        };
        assert!(
            ranges_intersect(&r3, &r4),
            "ranges sharing a point should intersect"
        );
    }

    #[test]
    fn test_ranges_intersect_character_less_than_boundary() {
        // Kills mutant line 79: replace < with ==/<= in ranges_intersect
        // Test that r1 ending at character 5 does NOT intersect r2 starting at character 6
        let r1 = Range {
            start: Position::new(0, 0),
            end: Position::new(0, 5),
        };
        let r2 = Range {
            start: Position::new(0, 6),
            end: Position::new(0, 10),
        };
        assert!(
            !ranges_intersect(&r1, &r2),
            "ranges with gap should not intersect"
        );

        // Also verify the symmetric case
        assert!(
            !ranges_intersect(&r2, &r1),
            "ranges with gap should not intersect (reversed)"
        );
    }

    #[test]
    fn test_find_theorem_position_closing_brace() {
        // Kills mutant line 260: replace + with * in find_theorem_at_position
        // The mutant changes (start_offset + end_rel) to (start_offset * end_rel)
        // Need a theorem where position calc matters
        let doc =
            make_doc("// leading comment\ntheorem multiline_test {\n  forall x: Int . x == x\n}");
        let spec = doc.spec.as_ref().unwrap();

        // Position on the last line (inside the closing brace line)
        let result = find_theorem_at_position(&doc, Position::new(3, 0), spec);
        assert!(
            result.is_some(),
            "should find theorem when on closing brace line"
        );
        assert_eq!(result.unwrap().name, "multiline_test");

        // Verify we're NOT inside when clearly after the theorem
        let doc2 = make_doc("theorem short { true }\n// after");
        let spec2 = doc2.spec.as_ref().unwrap();
        let result2 = find_theorem_at_position(&doc2, Position::new(1, 0), spec2);
        assert!(
            result2.is_none(),
            "should NOT find theorem when after closing brace"
        );
    }

    #[test]
    fn test_format_theorem_goal_returns_content() {
        // Kills mutants line 301: replace format_theorem_goal -> String with String::new() or "xyzzy"
        // Need to verify that format_theorem_goal returns actual meaningful content
        let doc = make_doc("theorem goal_test { forall x: Int . x + 1 > x }");
        let spec = doc.spec.as_ref().unwrap();

        // Find the theorem
        let result = find_theorem_at_position(&doc, Position::new(0, 20), spec);
        assert!(result.is_some());

        let info = result.unwrap();
        // The goal should contain the expression, not be empty or "xyzzy"
        assert!(!info.goal.is_empty(), "goal should not be empty");
        assert_ne!(info.goal, "xyzzy", "goal should not be xyzzy");
        // Should contain something related to the theorem body
        assert!(
            info.goal.contains("x") || info.goal.contains("forall"),
            "goal should contain expression content"
        );
    }

    #[test]
    fn test_find_type_end_position_with_newlines() {
        // Kills mutants at lines 393-401 in find_type_end_position
        // Test that position correctly accounts for newlines after closing brace
        let doc = make_doc("type Person = {\n  name: String\n}\n\ntheorem test { true }");

        let pos = find_type_end_position(&doc, "Person");
        assert!(pos.is_some());
        let p = pos.unwrap();
        // After the type def and its trailing newlines, we should be at line 3 or 4
        assert!(
            p.line >= 3,
            "position should be after the type definition (line {})",
            p.line
        );
    }

    #[test]
    fn test_find_type_end_position_newline_counting() {
        // Kills mutants: line 399 || with &&, line 399 == with !=
        // Test the newline/carriage return skipping logic specifically
        let doc = make_doc("type Test = { x: Int }\r\n\nrest");
        let pos = find_type_end_position(&doc, "Test");
        assert!(pos.is_some());
        let p = pos.unwrap();
        // Should skip past \r\n\n (3 chars of whitespace)
        assert!(p.line >= 2, "should skip past newlines (line {})", p.line);
    }

    #[test]
    fn test_find_type_end_position_arithmetic() {
        // Kills mutants at line 395: replace + with -/* in find_type_end_position
        // and line 401: replace + with -/*
        // We need to verify the exact character position calculation
        let doc = make_doc("type A = { }\ntype B = { }");

        // Get position after type A
        let pos_a = find_type_end_position(&doc, "A");
        assert!(pos_a.is_some());

        // Get position after type B
        let pos_b = find_type_end_position(&doc, "B");
        assert!(pos_b.is_some());

        // B should be on a later line
        assert!(
            pos_b.unwrap().line > pos_a.unwrap().line
                || (pos_b.unwrap().line == pos_a.unwrap().line
                    && pos_b.unwrap().character > pos_a.unwrap().character),
            "type B should be after type A"
        );
    }

    #[test]
    fn test_suggest_field_action_distance_boundary() {
        // Kills mutant line 447: replace > with >= in suggest_field_action
        // Distance of exactly 3 should still suggest (> 3 rejects, so = 3 accepts)
        let doc = make_doc(
            "type Person = { name: String, age: Int }\ntheorem foo { forall p: Person . p.nxme == \"\" }",
        );
        // "nxme" vs "name" has distance 1 (replace x with a) - should suggest
        let diagnostic =
            make_diagnostic("Invalid field access: type Person has no field nxme", 1, 35);
        let action = suggest_field_action(&doc, &diagnostic, &diagnostic.message);
        assert!(action.is_some(), "distance 1 should suggest");

        // Now test with distance exactly 3 (should still suggest)
        // "abcd" vs "name" = distance 4 (too far)
        // "nabc" vs "name" = distance 2 (replace b->m, c->e)
        // "nXYZ" vs "name" = distance 3 (X->a, Y->m, Z->e)
        let doc2 = make_doc(
            "type Person = { name: String, age: Int }\ntheorem foo { forall p: Person . p.nXYZ == \"\" }",
        );
        let diagnostic2 =
            make_diagnostic("Invalid field access: type Person has no field nXYZ", 1, 35);
        let action2 = suggest_field_action(&doc2, &diagnostic2, &diagnostic2.message);
        // Distance "nXYZ" to "name" = 3 (substitute X->a, Y->m, Z->e)
        // 3 is NOT > 3, so should suggest
        assert!(action2.is_some(), "distance exactly 3 should still suggest");

        // Distance 4 should NOT suggest
        let doc3 = make_doc(
            "type Person = { name: String, age: Int }\ntheorem foo { forall p: Person . p.WXYZ == \"\" }",
        );
        let diagnostic3 =
            make_diagnostic("Invalid field access: type Person has no field WXYZ", 1, 35);
        let action3 = suggest_field_action(&doc3, &diagnostic3, &diagnostic3.message);
        // Distance "WXYZ" to "name" = 4 (W->n, X->a, Y->m, Z->e)
        // 4 > 3, so should NOT suggest
        assert!(
            action3.is_none(),
            "distance 4 should NOT suggest (distance > 3)"
        );
    }

    #[test]
    fn test_find_theorem_at_position_nonzero_offset() {
        // Kills mutant line 260: replace + with * in find_theorem_at_position
        // The mutant changes (start_offset + end_rel) to (start_offset * end_rel)
        // offset_to_position clamps to doc length, so we need a case where the
        // clamped result produces a DIFFERENT line than the correct calculation.
        //
        // Test setup: position 0 on line 1, theorem starts mid-line 1, ends on line 1
        // We want: if mutant calculates wrong end_line, position check should fail
        //
        // With large padding before theorem:
        // "// long padding text here\ntheorem test { true }\n// after"
        // start_offset = 27 (after first newline + "theorem" at position 27)
        // Actually let's trace: "// long padding text here\n" = 27 chars
        // "theorem test { true }" starts at offset 27
        // find_closing_brace returns ~20 for the }, so end_rel = 20
        // correct: 27 + 20 = 47 (on line 1)
        // mutant:  27 * 20 = 540 (clamped to doc length ~55, on line 2)
        //
        // If we query position (1, 0), start_line=1, end_line differs:
        // correct: end_line=1, check passes
        // mutant: end_line=2, check passes (1 <= 2)
        //
        // Need to query a position that FAILS with mutant's wrong end_line.
        // Try: position on line 1 where correct says "inside" but mutant says wrong range

        // Actually the issue is: clamped offset still maps to last line of doc.
        // We need the theorem to NOT be at the end of the document, so that
        // the correct end_line differs from the clamped end_line.

        // Better approach: test that we DON'T find a theorem when position is
        // outside the correct range but inside the mutant's wrong range.
        // "// comment\ntheorem test { true }\n// after"
        // start_offset ~= 11 ("// comment\n" = 11 chars), end_rel ~= 20
        // correct end_offset = 31, end_line = 1
        // mutant end_offset = 220 (clamped to doc len ~44), end_line = 2
        //
        // Query position (2, 5) - on "// after" line:
        // correct: pos.line(2) >= start_line(1) && pos.line(2) <= end_line(1) = false
        // mutant:  pos.line(2) >= start_line(1) && pos.line(2) <= end_line(2) = true (WRONG!)
        let doc = make_doc("// comment\ntheorem test { true }\n// after this");
        let spec = doc.spec.as_ref().unwrap();

        // Position on line 2 (the "after" line) should NOT be inside the theorem
        let result = find_theorem_at_position(&doc, Position::new(2, 5), spec);
        assert!(
            result.is_none(),
            "position after theorem should NOT find theorem"
        );

        // Verify line 1 (inside theorem) still works
        let result_inside = find_theorem_at_position(&doc, Position::new(1, 10), spec);
        assert!(
            result_inside.is_some(),
            "position inside theorem should work"
        );
    }

    #[test]
    fn test_find_type_end_position_early_close_brace() {
        // This test verifies proper brace matching behavior
        // The && vs || mutant at line 393 is equivalent/unviable because:
        // - brace_depth == 0 only occurs after balanced {}, at which point found_open is always true
        // - When found_open is false, brace_depth is either 0 (no braces seen) or negative (unbalanced)
        // - We only check the condition on '}', so if first brace is '}', depth becomes -1, not 0
        let doc = make_doc("// comment with } brace\ntype Foo = { x: Int }");

        let pos = find_type_end_position(&doc, "Foo");
        assert!(pos.is_some());
        let p = pos.unwrap();
        // The position should be after the type's closing brace on line 1
        assert!(
            p.line >= 1,
            "position should be after the type definition (line {})",
            p.line
        );
    }
}
