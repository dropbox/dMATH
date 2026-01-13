//! Export command implementation

use dashprove::usl::{
    compile_to_alloy, compile_to_coq, compile_to_dafny, compile_to_isabelle, compile_to_kani,
    compile_to_lean, compile_to_smtlib2, compile_to_smtlib2_with_logic, compile_to_tlaplus, parse,
    typecheck,
};
use std::path::Path;

/// Run export command
pub async fn run_export(
    path: &str,
    target: &str,
    output: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read specification file
    let spec_path = Path::new(path);
    if !spec_path.exists() {
        return Err(format!("Specification file not found: {}", path).into());
    }

    let spec_content = std::fs::read_to_string(spec_path)?;

    // Parse and type-check
    let spec = parse(&spec_content).map_err(|e| format!("Parse error: {:?}", e))?;
    let typed_spec = typecheck(spec).map_err(|e| format!("Type error: {:?}", e))?;

    // Compile to target
    let compiled = match target.to_lowercase().as_str() {
        "lean" | "lean4" => compile_to_lean(&typed_spec),
        "tla+" | "tlaplus" | "tla" => compile_to_tlaplus(&typed_spec),
        "kani" => compile_to_kani(&typed_spec),
        "alloy" => compile_to_alloy(&typed_spec),
        "coq" => compile_to_coq(&typed_spec),
        "isabelle" | "isabelle/hol" => compile_to_isabelle(&typed_spec),
        "dafny" => compile_to_dafny(&typed_spec),
        "smtlib" | "smtlib2" | "smt" => compile_to_smtlib2(&typed_spec),
        // Allow specifying SMT-LIB logic explicitly via smtlib:LOGIC syntax
        target if target.starts_with("smtlib:") || target.starts_with("smt:") => {
            let logic = target.split(':').nth(1).unwrap_or("ALL");
            compile_to_smtlib2_with_logic(&typed_spec, logic)
        }
        _ => {
            return Err(format!(
                "Unknown target: {}. Supported targets:\n  \
                 lean, tla+, kani, alloy, coq, isabelle, dafny, smtlib\n  \
                 For SMT-LIB with specific logic: smtlib:QF_LIA, smtlib:QF_BV, etc.",
                target
            )
            .into())
        }
    };

    // Output
    if let Some(output_path) = output {
        std::fs::write(output_path, &compiled.code)?;
        println!("Exported to {}", output_path);
    } else {
        println!("{}", compiled.code);
    }

    Ok(())
}
