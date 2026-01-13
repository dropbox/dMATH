//! Example: Load a model from native format (PyTorch/SafeTensors) without ONNX export.
//!
//! Usage: cargo run --example load_native --features pytorch -- <model_path>

use gamma_onnx::native::NativeModel;
use std::env;

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_path>", args[0]);
        eprintln!("Supported formats: .pt, .pth, .bin, .safetensors");
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model from: {}", model_path);

    match NativeModel::load(model_path) {
        Ok(model) => {
            println!("\n=== Model Loaded Successfully ===");
            println!("Architecture: {:?}", model.config.architecture);
            println!("Hidden dim: {}", model.config.hidden_dim);
            if let Some(heads) = model.config.num_heads {
                println!("Attention heads: {}", heads);
            }
            if let Some(layers) = model.config.num_layers {
                println!("Layers: {}", layers);
            }

            println!("\n=== Network Structure ===");
            println!("Name: {}", model.network.name);
            println!("Inputs: {:?}", model.network.inputs);
            println!("Outputs: {:?}", model.network.outputs);
            println!("Layer count: {}", model.network.layers.len());
            println!("Parameter count: {}", model.network.param_count);

            println!("\n=== Weights ===");
            println!("Total tensors: {}", model.weights.len());

            // Show first 10 weight names
            for (idx, name) in model.weights.keys().take(10).enumerate() {
                if let Some(weight) = model.weights.get(name) {
                    println!(
                        "  [{}] {} - shape {:?}, {} params",
                        idx,
                        name,
                        weight.shape(),
                        weight.len()
                    );
                }
            }
            if model.weights.len() > 10 {
                println!("  ... and {} more", model.weights.len() - 10);
            }
        }
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            std::process::exit(1);
        }
    }
}
