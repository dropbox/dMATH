//! Code lens generation for individual properties and types.
//!
//! Generates clickable action buttons above verifiable properties
//! to trigger verification, show backend info, and suggest tactics.

use super::document_stats::generate_document_stats_lenses;
use super::workspace_stats::{generate_workspace_stats_lenses, WorkspaceStats};
use crate::document::Document;
use crate::symbols::{format_expr, format_type, property_kind_name};
use dashprove_usl::{Contract, Property, TypeDef};
use tower_lsp::lsp_types::{CodeLens, Command, Position, Range};

/// Generate all code lenses for a document, including workspace and document stats.
pub fn generate_all_code_lenses(doc: &Document, workspace_stats: &WorkspaceStats) -> Vec<CodeLens> {
    let mut lenses = Vec::new();

    let workspace_lenses = generate_workspace_stats_lenses(doc, workspace_stats);
    let document_line = if workspace_lenses.is_empty() { 0 } else { 1 };
    let document_lenses = generate_document_stats_lenses(doc, document_line);

    lenses.extend(workspace_lenses);
    lenses.extend(document_lenses);
    lenses.extend(generate_code_lenses(doc));

    lenses
}

/// Generate code lenses for a USL document.
///
/// Creates "Verify" lenses above each verifiable property (theorem, temporal,
/// contract, invariant, refinement, probabilistic, security).
/// Also creates "Recommend backend" lenses for type definitions.
pub fn generate_code_lenses(doc: &Document) -> Vec<CodeLens> {
    let mut lenses = Vec::new();

    let spec = match &doc.spec {
        Some(spec) => spec,
        None => return lenses,
    };

    // Add code lens for each type definition (backend recommendation)
    for type_def in &spec.types {
        if let Some(range) = find_type_definition_range(doc, &type_def.name) {
            // Create a description of the type for the recommendation query
            let type_description = format_type_def_inline(type_def);

            lenses.push(CodeLens {
                range,
                command: Some(Command {
                    title: "🎯 Recommend verification backend".to_string(),
                    command: "dashprove.recommendBackend".to_string(),
                    arguments: Some(vec![
                        serde_json::Value::String(type_description),
                        serde_json::Value::String(doc.uri.to_string()),
                    ]),
                }),
                data: None,
            });
        }
    }

    // Add code lens for each property
    for prop in &spec.properties {
        let name = prop.name();
        let kind = property_kind_name(prop);

        // Find the position of the property definition
        if let Some(range) = find_property_definition_range(doc, &name, kind) {
            // Primary lens: Verify this property
            lenses.push(CodeLens {
                range,
                command: Some(Command {
                    title: format!("▶ Verify {}", kind),
                    command: "dashprove.verify".to_string(),
                    arguments: Some(vec![
                        serde_json::Value::String(doc.uri.to_string()),
                        serde_json::Value::String(name.clone()),
                    ]),
                }),
                data: None,
            });

            // Secondary lens: Show proof strategy info
            lenses.push(CodeLens {
                range,
                command: Some(Command {
                    title: backend_hint(prop),
                    command: "dashprove.showBackendInfo".to_string(),
                    arguments: Some(vec![
                        serde_json::Value::String(doc.uri.to_string()),
                        serde_json::Value::String(name.clone()),
                    ]),
                }),
                data: None,
            });

            // Tertiary lens for theorems: Suggest tactics
            if let Property::Theorem(theorem) = prop {
                let goal_description = format_expr(&theorem.body);
                let backend = "lean4"; // Default backend for theorems

                lenses.push(CodeLens {
                    range,
                    command: Some(Command {
                        title: "💡 Suggest tactics".to_string(),
                        command: "dashprove.suggestTactics".to_string(),
                        arguments: Some(vec![
                            serde_json::Value::String(goal_description),
                            serde_json::Value::String(backend.to_string()),
                            serde_json::Value::String(name.clone()),
                        ]),
                    }),
                    data: None,
                });
            }

            // Tactic suggestion for invariants (similar to theorems)
            if let Property::Invariant(invariant) = prop {
                let goal_description = format_expr(&invariant.body);
                let backend = "lean4"; // Invariants typically use Lean4 or Alloy

                lenses.push(CodeLens {
                    range,
                    command: Some(Command {
                        title: "💡 Suggest tactics".to_string(),
                        command: "dashprove.suggestTactics".to_string(),
                        arguments: Some(vec![
                            serde_json::Value::String(goal_description),
                            serde_json::Value::String(backend.to_string()),
                            serde_json::Value::String(name.clone()),
                        ]),
                    }),
                    data: None,
                });
            }

            // Tactic suggestion for contracts (Kani/verification strategies)
            if let Property::Contract(contract) = prop {
                let goal_description = format_contract_goal(contract);
                let backend = "kani"; // Contracts use Kani

                lenses.push(CodeLens {
                    range,
                    command: Some(Command {
                        title: "💡 Suggest verification strategies".to_string(),
                        command: "dashprove.suggestTactics".to_string(),
                        arguments: Some(vec![
                            serde_json::Value::String(goal_description),
                            serde_json::Value::String(backend.to_string()),
                            serde_json::Value::String(name.clone()),
                        ]),
                    }),
                    data: None,
                });
            }
        }
    }

    lenses
}

/// Find the range for a property definition (the line where keyword appears).
fn find_property_definition_range(doc: &Document, name: &str, kind: &str) -> Option<Range> {
    // Search for the pattern "keyword name"
    let pattern = format!("{} {}", kind, name);

    // For contracts, the pattern is different: "contract Type::method"
    let search_pattern = if kind == "contract" {
        format!("contract {}", name)
    } else {
        pattern
    };

    let offset = doc.text.find(&search_pattern)?;
    let (line, _) = doc.offset_to_position(offset);

    // Code lens appears at the start of the line containing the property
    Some(Range {
        start: Position::new(line, 0),
        end: Position::new(line, 0),
    })
}

/// Find the range for a type definition (the line where "type TypeName" appears).
fn find_type_definition_range(doc: &Document, name: &str) -> Option<Range> {
    let pattern = format!("type {}", name);
    let offset = doc.text.find(&pattern)?;
    let (line, _) = doc.offset_to_position(offset);

    Some(Range {
        start: Position::new(line, 0),
        end: Position::new(line, 0),
    })
}

/// Format a type definition for display in backend recommendation queries.
fn format_type_def_inline(type_def: &TypeDef) -> String {
    let fields: Vec<String> = type_def
        .fields
        .iter()
        .map(|f| format!("{}: {}", f.name, format_type(&f.ty)))
        .collect();

    format!("type {} = {{ {} }}", type_def.name, fields.join(", "))
}

/// Get a hint about which backend will be used for verification.
fn backend_hint(prop: &Property) -> String {
    match prop {
        Property::Theorem(_) => "→ LEAN 4 / Coq".to_string(),
        Property::Temporal(_) => "→ TLA+".to_string(),
        Property::Contract(_) => "→ Kani".to_string(),
        Property::Invariant(_) => "→ LEAN 4 / Alloy".to_string(),
        Property::Refinement(_) => "→ LEAN 4".to_string(),
        Property::Probabilistic(_) => "→ Probabilistic checker".to_string(),
        Property::Security(_) => "→ Security analyzer".to_string(),
        Property::Semantic(_) => "→ Semantic verifier".to_string(),
        Property::PlatformApi(_) => "→ Platform constraint".to_string(),
        Property::Bisimulation(_) => "→ Bisimulation checker".to_string(),
        Property::Version(_) => "→ LEAN 4 (version improvement)".to_string(),
        Property::Capability(_) => "→ LEAN 4 (capability spec)".to_string(),
        Property::DistributedInvariant(_) => "→ TLA+ (distributed invariant)".to_string(),
        Property::DistributedTemporal(_) => "→ TLA+ (distributed temporal)".to_string(),
        Property::Composed(_) => "→ LEAN 4 (composed theorem)".to_string(),
        Property::ImprovementProposal(_) => "→ LEAN 4 (improvement proposal)".to_string(),
        Property::VerificationGate(_) => "→ LEAN 4 (verification gate)".to_string(),
        Property::Rollback(_) => "→ LEAN 4 (rollback)".to_string(),
    }
}

/// Format a contract's goal description for tactic/verification strategy suggestions
fn format_contract_goal(contract: &Contract) -> String {
    let mut parts = Vec::new();

    // Add preconditions
    for req in &contract.requires {
        parts.push(format!("requires: {}", format_expr(req)));
    }

    // Add postconditions
    for ens in &contract.ensures {
        parts.push(format!("ensures: {}", format_expr(ens)));
    }

    // Add error postconditions
    for ens_err in &contract.ensures_err {
        parts.push(format!("ensures_err: {}", format_expr(ens_err)));
    }

    if parts.is_empty() {
        format!("contract {}", contract.type_path.join("::"))
    } else {
        format!(
            "contract {} with {}",
            contract.type_path.join("::"),
            parts.join(", ")
        )
    }
}
