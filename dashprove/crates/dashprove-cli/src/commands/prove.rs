//! Interactive proof mode command implementation

use crate::commands::common::{default_data_dir, get_compiler_tactics};
use dashprove::{
    learning::ProofLearningSystem,
    usl::{
        compile_to_alloy, compile_to_kani, compile_to_lean, compile_to_tlaplus, parse, typecheck,
        Property,
    },
};
use std::io::{self, BufRead, Write};
use std::path::Path;

fn property_kind(prop: &Property) -> &'static str {
    match prop {
        Property::Theorem(_) => "Theorem",
        Property::Temporal(_) => "Temporal",
        Property::Contract(_) => "Contract",
        Property::Invariant(_) => "Invariant",
        Property::Refinement(_) => "Refinement",
        Property::Probabilistic(_) => "Probabilistic",
        Property::Security(_) => "Security",
        Property::Semantic(_) => "Semantic",
        Property::PlatformApi(_) => "Platform API",
        Property::Bisimulation(_) => "Bisimulation",
        Property::Version(_) => "Version",
        Property::Capability(_) => "Capability",
        Property::DistributedInvariant(_) => "Distributed Invariant",
        Property::DistributedTemporal(_) => "Distributed Temporal",
        Property::Composed(_) => "Composed",
        Property::ImprovementProposal(_) => "Improvement Proposal",
        Property::VerificationGate(_) => "Verification Gate",
        Property::Rollback(_) => "Rollback",
    }
}

fn describe_property(prop: &Property) -> (String, String) {
    let kind = property_kind(prop).to_string();
    let detail = match prop {
        Property::Theorem(t) => format!("Body: {:?}", t.body),
        Property::Temporal(t) => format!("Temporal formula: {:?}", t.body),
        Property::Contract(c) => format!(
            "Params: {}, requires: {}, ensures: {}, error ensures: {}",
            c.params.len(),
            c.requires.len(),
            c.ensures.len(),
            c.ensures_err.len()
        ),
        Property::Invariant(i) => format!("Invariant body: {:?}", i.body),
        Property::Refinement(r) => format!(
            "Refines {} with {} mappings, {} invariants, {} actions",
            r.refines,
            r.mappings.len(),
            r.invariants.len(),
            r.actions.len()
        ),
        Property::DistributedInvariant(d) => format!("Distributed invariant body: {:?}", d.body),
        Property::DistributedTemporal(d) => {
            format!("Distributed temporal formula: {:?}", d.body)
        }
        other => format!("{:?}", other),
    };

    (kind, detail)
}

/// Run interactive proof mode
pub fn run_prove(path: &str, hints: bool) -> Result<(), Box<dyn std::error::Error>> {
    // Read and parse specification
    let spec_path = Path::new(path);
    if !spec_path.exists() {
        return Err(format!("Specification file not found: {}", path).into());
    }

    let spec_content = std::fs::read_to_string(spec_path)?;
    let spec = parse(&spec_content).map_err(|e| format!("Parse error: {:?}", e))?;
    let typed_spec = typecheck(spec).map_err(|e| format!("Type error: {:?}", e))?;

    if typed_spec.spec.properties.is_empty() {
        println!("No properties found in specification.");
        return Ok(());
    }

    println!("=== DashProve Interactive Proof Mode ===");
    println!("File: {}", path);
    println!("Properties: {}\n", typed_spec.spec.properties.len());
    println!("Commands:");
    println!("  list              - List all properties");
    println!("  select <n>        - Select property to prove");
    println!("  show [n]          - Show details for current or numbered property");
    println!("  search <keyword>  - Find properties by name");
    println!("  export <backend>  - Export current property to backend format");
    println!("  tactics           - Show suggested tactics");
    println!("  similar           - Find similar proofs");
    println!("  quit              - Exit interactive mode\n");

    let mut current_property: Option<usize> = None;
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    // Load learning system for hints
    let learning_system = if hints {
        let dir = default_data_dir();
        ProofLearningSystem::load_from_dir(&dir).ok()
    } else {
        None
    };

    loop {
        // Show prompt
        let prompt = if let Some(idx) = current_property {
            let name = typed_spec
                .spec
                .properties
                .get(idx)
                .map(|p| p.name())
                .unwrap_or_default();
            format!("[{}]> ", name)
        } else {
            "> ".to_string()
        };
        print!("{}", prompt);
        stdout.flush()?;

        // Read command
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF
        }
        let line = line.trim();

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let cmd = parts[0].to_lowercase();

        match cmd.as_str() {
            "quit" | "q" | "exit" => {
                println!("Exiting interactive mode.");
                break;
            }
            "list" | "ls" => {
                println!("\nProperties:");
                for (i, prop) in typed_spec.spec.properties.iter().enumerate() {
                    let marker = if current_property == Some(i) {
                        "*"
                    } else {
                        " "
                    };
                    println!("{} {}. {}", marker, i + 1, prop.name());
                }
                println!();
            }
            "select" | "sel" => {
                if parts.len() < 2 {
                    println!("Usage: select <number>");
                    continue;
                }
                match parts[1].parse::<usize>() {
                    Ok(n) if n > 0 && n <= typed_spec.spec.properties.len() => {
                        current_property = Some(n - 1);
                        let prop = &typed_spec.spec.properties[n - 1];
                        println!("Selected: {}", prop.name());

                        if hints {
                            // Show compiler suggestions
                            let compiler_tactics = get_compiler_tactics(prop);
                            if !compiler_tactics.is_empty() {
                                println!("  Suggested tactics: {}", compiler_tactics.join(", "));
                            }
                        }
                    }
                    _ => {
                        println!("Invalid property number. Use 'list' to see available properties.")
                    }
                }
            }
            "show" => {
                let target_idx = if parts.len() > 1 {
                    match parts[1].parse::<usize>() {
                        Ok(n) if n > 0 && n <= typed_spec.spec.properties.len() => Some(n - 1),
                        _ => {
                            println!(
                                "Invalid property number. Use 'list' to see available properties."
                            );
                            continue;
                        }
                    }
                } else {
                    current_property
                };

                let Some(idx) = target_idx else {
                    println!("No property selected. Use 'select <n>' first.");
                    continue;
                };

                let prop = &typed_spec.spec.properties[idx];
                let (kind, detail) = describe_property(prop);
                println!("\n{} #{}: {}", kind, idx + 1, prop.name());
                println!("  {}", detail);
                println!("  Supported exports: lean, tla+, kani, alloy");
                if hints {
                    let compiler_tactics = get_compiler_tactics(prop);
                    if !compiler_tactics.is_empty() {
                        println!("  Suggested tactics: {}", compiler_tactics.join(", "));
                    }
                }
                println!();
            }
            "search" => {
                if parts.len() < 2 {
                    println!("Usage: search <keyword>");
                    continue;
                }
                let needle = parts[1..].join(" ").to_lowercase();
                let mut found = 0usize;
                println!("\nSearch results for '{}':", needle);
                for (i, prop) in typed_spec.spec.properties.iter().enumerate() {
                    if prop.name().to_lowercase().contains(&needle) {
                        let marker = if current_property == Some(i) {
                            "*"
                        } else {
                            " "
                        };
                        println!(
                            "{} {}. {} ({})",
                            marker,
                            i + 1,
                            prop.name(),
                            property_kind(prop)
                        );
                        found += 1;
                    }
                }
                if found == 0 {
                    println!("  No properties matched '{}'", needle);
                }
                println!();
            }
            "export" => {
                if current_property.is_none() {
                    println!("No property selected. Use 'select <n>' first.");
                    continue;
                }
                let target = parts.get(1).unwrap_or(&"lean");

                // Create a spec with just the current property
                let prop = typed_spec.spec.properties[current_property.unwrap()].clone();
                let single_spec = dashprove::usl::Spec {
                    types: typed_spec.spec.types.clone(),
                    properties: vec![prop],
                };
                let single_typed = typecheck(single_spec)?;

                let compiled = match *target {
                    "lean" | "lean4" => compile_to_lean(&single_typed),
                    "tla+" | "tlaplus" | "tla" => compile_to_tlaplus(&single_typed),
                    "kani" => compile_to_kani(&single_typed),
                    "alloy" => compile_to_alloy(&single_typed),
                    _ => {
                        println!("Unknown target: {}. Use: lean, tla+, kani, alloy", target);
                        continue;
                    }
                };
                println!("\n--- {} output ---\n{}\n", target, compiled.code);
            }
            "tactics" | "tac" => {
                if current_property.is_none() {
                    println!("No property selected. Use 'select <n>' first.");
                    continue;
                }
                let prop = &typed_spec.spec.properties[current_property.unwrap()];

                // Compiler-based suggestions
                let compiler_tactics = get_compiler_tactics(prop);
                println!(
                    "\nCompiler suggestions: {}",
                    if compiler_tactics.is_empty() {
                        "none".to_string()
                    } else {
                        compiler_tactics.join(", ")
                    }
                );

                // Learning-based suggestions
                if let Some(ref system) = learning_system {
                    let learned = system.suggest_tactics(prop, 5);
                    if !learned.is_empty() {
                        println!("Learned tactics:");
                        for (tactic, score) in learned {
                            println!("  {} (score: {:.3})", tactic, score);
                        }
                    }
                }
                println!();
            }
            "similar" | "sim" => {
                if current_property.is_none() {
                    println!("No property selected. Use 'select <n>' first.");
                    continue;
                }
                if let Some(ref system) = learning_system {
                    let prop = &typed_spec.spec.properties[current_property.unwrap()];
                    let similar = system.find_similar(prop, 5);
                    if similar.is_empty() {
                        println!("No similar proofs found in corpus.");
                    } else {
                        println!("\nSimilar proofs:");
                        for (i, proof) in similar.iter().enumerate() {
                            println!(
                                "  {}. {} ({:.0}% similar, {:?})",
                                i + 1,
                                proof.property.name(),
                                proof.similarity * 100.0,
                                proof.backend
                            );
                        }
                        println!();
                    }
                } else {
                    println!(
                        "Learning data not loaded. Use --hints flag or run with --learn first."
                    );
                }
            }
            "help" | "?" => {
                println!("\nCommands:");
                println!("  list              - List all properties");
                println!("  select <n>        - Select property to prove");
                println!("  show [n]          - Show details for current or numbered property");
                println!("  search <keyword>  - Find properties by name");
                println!("  export <backend>  - Export current property to backend format");
                println!("  tactics           - Show suggested tactics");
                println!("  similar           - Find similar proofs");
                println!("  quit              - Exit interactive mode\n");
            }
            _ => {
                println!(
                    "Unknown command: '{}'. Type 'help' for available commands.",
                    cmd
                );
            }
        }
    }

    Ok(())
}
