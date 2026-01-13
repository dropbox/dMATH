use kani_fast_chc::mir::MirStatement;
use kani_fast_chc::mir_parser::{generate_mir_from_file, MirParser};

fn main() {
    let mir_text = generate_mir_from_file("/tmp/trait_static.rs", Some("2021")).unwrap();
    println!("=== Full MIR ===\n{}\n", mir_text);

    let parser = MirParser::new();
    let functions = parser.parse(&mir_text).unwrap();

    println!("=== Parsed Functions ===");
    for func in &functions {
        println!("Function: {}", func.name);
        for (i, block) in func.basic_blocks.iter().enumerate() {
            println!("  Block {}: {:?}", i, block.terminator);
        }
    }

    // Find trait_static_proof function
    if let Some(func) = functions.iter().find(|f| f.name == "trait_static_proof") {
        println!("\n=== trait_static_proof MirProgram ===");
        let program = func.to_mir_program_with_all_inlines(&functions);

        println!("Locals:");
        for local in &program.locals {
            println!("  {} : {:?}", local.name, local.ty);
        }

        println!("\nBlocks:");
        for block in &program.basic_blocks {
            println!("Block {}:", block.id);
            for stmt in &block.statements {
                match stmt {
                    MirStatement::Assign { lhs, rhs } => println!("  {} = {}", lhs, rhs),
                    _ => println!("  {:?}", stmt),
                }
            }
            println!("  Term: {:?}", block.terminator);
        }

        println!("\nTrait Impls:");
        for (key, info) in &program.trait_impls {
            println!("  {}: body = {:?}", key, info.body_expr);
        }
    }

    // Find impl function
    for func in &functions {
        if func.name.contains("add_value") {
            println!("\n=== {} (impl function) ===", func.name);
            let program = func.to_mir_program_with_all_inlines(&functions);
            println!("Locals:");
            for local in &program.locals {
                println!("  {} : {:?}", local.name, local.ty);
            }
            println!("\nBlocks:");
            for block in &program.basic_blocks {
                println!("Block {}:", block.id);
                for stmt in &block.statements {
                    match stmt {
                        MirStatement::Assign { lhs, rhs } => println!("  {} = {}", lhs, rhs),
                        _ => println!("  {:?}", stmt),
                    }
                }
                println!("  Term: {:?}", block.terminator);
            }
        }
    }
}
