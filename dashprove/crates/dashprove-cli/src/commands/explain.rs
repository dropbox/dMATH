//! Explain command implementation

use crate::commands::common::parse_backend;
use dashprove::{
    ai::explain_counterexample,
    usl::{ast::Invariant, parse, typecheck, Property},
};
use std::path::Path;

/// Counterexample JSON file format
#[derive(Debug, serde::Deserialize)]
struct CounterexampleFile {
    /// The property that was violated (as USL source)
    property: Option<String>,
    /// The backend that produced the counterexample
    backend: Option<String>,
    /// The raw counterexample output
    counterexample: String,
}

/// Run explain command
pub fn run_explain(path: &str, backend: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let ce_path = Path::new(path);
    if !ce_path.exists() {
        return Err(format!("Counterexample file not found: {}", path).into());
    }

    // Try to parse as JSON first
    let content = std::fs::read_to_string(ce_path)?;

    let (property, backend_id, counterexample) = if let Ok(ce_file) =
        serde_json::from_str::<CounterexampleFile>(&content)
    {
        // JSON format
        let backend_str = backend
            .or(ce_file.backend.as_deref())
            .ok_or("Backend not specified. Use --backend flag.")?;
        let backend_id =
            parse_backend(backend_str).ok_or(format!("Unknown backend: {}", backend_str))?;

        // Parse property if provided
        let property = if let Some(ref prop_src) = ce_file.property {
            let spec = parse(&format!("theorem query {{ {} }}", prop_src))
                .map_err(|e| format!("Failed to parse property: {:?}", e))?;
            let typed = typecheck(spec).map_err(|e| format!("Type error: {:?}", e))?;
            typed.spec.properties.into_iter().next()
        } else {
            None
        };

        (property, backend_id, ce_file.counterexample)
    } else {
        // Plain text format - use backend flag or default
        let backend_str = backend
            .ok_or("For plain text counterexamples, use --backend flag to specify the backend")?;
        let backend_id =
            parse_backend(backend_str).ok_or(format!("Unknown backend: {}", backend_str))?;
        (None, backend_id, content)
    };

    // Create a dummy property if none provided
    let dummy_property = Property::Invariant(Invariant {
        name: "unknown".to_string(),
        body: dashprove::usl::ast::Expr::Bool(true),
    });
    let prop = property.as_ref().unwrap_or(&dummy_property);

    // Generate explanation
    let explanation = explain_counterexample(prop, &counterexample, &backend_id);

    // Print explanation
    println!("=== Counterexample Explanation ===\n");
    println!("Type: {:?}", explanation.kind);
    println!("\nSummary: {}", explanation.summary);

    if !explanation.bindings.is_empty() {
        println!("\nVariable Bindings:");
        for binding in &explanation.bindings {
            if let Some(ref ty) = binding.ty {
                println!("  {} : {} = {}", binding.name, ty, binding.value);
            } else {
                println!("  {} = {}", binding.name, binding.value);
            }
        }
    }

    if !explanation.trace.is_empty() {
        println!("\nExecution Trace:");
        for step in &explanation.trace {
            println!("\n  Step {}:", step.step_number);
            if let Some(ref action) = step.action {
                println!("    Action: {}", action);
            }
            for binding in &step.state {
                println!("    {} = {}", binding.name, binding.value);
            }
        }
    }

    if !explanation.suggestions.is_empty() {
        println!("\nSuggestions:");
        for (i, suggestion) in explanation.suggestions.iter().enumerate() {
            println!("  {}. {}", i + 1, suggestion);
        }
    }

    println!("\n--- Raw Details ---");
    println!("{}", explanation.details);

    Ok(())
}
