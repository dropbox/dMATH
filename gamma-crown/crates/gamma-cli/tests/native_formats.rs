//! Integration tests for native model format support.
//!
//! Tests that γ-CROWN can load and verify models in various native formats:
//! - SafeTensors (.safetensors)
//! - PyTorch (.pt, .pth, .bin)
//! - GGUF (.gguf)

use std::process::Command;

/// Get the path to the gamma binary (debug or release)
fn gamma_binary() -> String {
    // Try to get the path relative to workspace root
    let workspace_root = std::env::var("CARGO_MANIFEST_DIR")
        .map(|d| {
            std::path::PathBuf::from(d)
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf()
        })
        .unwrap_or_else(|_| std::path::PathBuf::from("."));

    let release_bin = workspace_root.join("target/release/gamma");
    let debug_bin = workspace_root.join("target/debug/gamma");

    if release_bin.exists() {
        release_bin.to_string_lossy().to_string()
    } else if debug_bin.exists() {
        debug_bin.to_string_lossy().to_string()
    } else {
        // Fallback to simple path
        "gamma".to_string()
    }
}

/// Test that inspect command works with SafeTensors format
#[test]
fn test_inspect_safetensors() {
    // Skip if test model doesn't exist
    let model_path = "models/whisper-tiny/model.safetensors";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: {} not found", model_path);
        return;
    }

    let output = Command::new(gamma_binary())
        .args(["inspect", model_path, "--json"])
        .output()
        .expect("Failed to run gamma inspect");

    assert!(
        output.status.success(),
        "gamma inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("WhisperEncoder"),
        "Expected WhisperEncoder architecture"
    );
}

/// Test that inspect command works with PyTorch format
#[test]
fn test_inspect_pytorch() {
    // Skip if test model doesn't exist
    let model_path = "models/kokoro/kokoro-v1_0.pth";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: {} not found", model_path);
        return;
    }

    let output = Command::new(gamma_binary())
        .args(["inspect", model_path, "--json"])
        .output()
        .expect("Failed to run gamma inspect");

    assert!(
        output.status.success(),
        "gamma inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Kokoro"), "Expected Kokoro architecture");
}

/// Test that inspect command works with GGUF format
#[test]
fn test_inspect_gguf() {
    // Skip if test model doesn't exist
    let model_path = "models/gemma-2b-q4.gguf";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: {} not found", model_path);
        return;
    }

    let output = Command::new(gamma_binary())
        .args(["inspect", model_path, "--json"])
        .output()
        .expect("Failed to run gamma inspect");

    assert!(
        output.status.success(),
        "gamma inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // GGUF models load with detected architecture
    assert!(
        stdout.contains("parameters") || stdout.contains("Weight tensors"),
        "Expected model info in output"
    );
}

/// Test simple ONNX verification still works
#[test]
fn test_verify_simple_mlp() {
    let model_path = "tests/models/simple_mlp.onnx";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: {} not found", model_path);
        return;
    }

    let output = Command::new(gamma_binary())
        .args([
            "verify",
            model_path,
            "--epsilon",
            "0.01",
            "--method",
            "ibp",
            "--json",
        ])
        .output()
        .expect("Failed to run gamma verify");

    assert!(
        output.status.success(),
        "gamma verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("\"status\":\"verified\""),
        "Expected verified status, got: {}",
        stdout
    );
}

/// Test CROWN method verification
#[test]
fn test_verify_crown_method() {
    let model_path = "tests/models/simple_mlp.onnx";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: {} not found", model_path);
        return;
    }

    let output = Command::new(gamma_binary())
        .args([
            "verify",
            model_path,
            "--epsilon",
            "0.01",
            "--method",
            "crown",
            "--json",
        ])
        .output()
        .expect("Failed to run gamma verify");

    assert!(
        output.status.success(),
        "gamma verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("\"status\":\"verified\""),
        "Expected verified status, got: {}",
        stdout
    );
    assert!(
        stdout.contains("\"method\":\"crown\""),
        "Expected crown method, got: {}",
        stdout
    );
}

/// Test layer benchmarks run correctly
#[test]
fn test_bench_layer() {
    let output = Command::new(gamma_binary())
        .args(["bench", "--benchmark", "layer", "--json"])
        .output()
        .expect("Failed to run gamma bench");

    assert!(
        output.status.success(),
        "gamma bench failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Linear IBP"),
        "Expected Linear IBP benchmark"
    );
    assert!(stdout.contains("GELU IBP"), "Expected GELU IBP benchmark");
    assert!(
        stdout.contains("LayerNorm IBP"),
        "Expected LayerNorm IBP benchmark"
    );
}

/// Test output bounds are sound (lower <= upper)
#[test]
fn test_bounds_soundness() {
    let model_path = "tests/models/simple_mlp.onnx";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: {} not found", model_path);
        return;
    }

    let output = Command::new(gamma_binary())
        .args([
            "verify",
            model_path,
            "--epsilon",
            "0.1",
            "--method",
            "ibp",
            "--json",
        ])
        .output()
        .expect("Failed to run gamma verify");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse JSON and check bounds
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("Failed to parse JSON output");

    if let Some(bounds) = v.get("output_bounds").and_then(|b| b.as_array()) {
        for bound in bounds {
            let lower = bound.get("lower").and_then(|l| l.as_f64()).unwrap_or(0.0);
            let upper = bound.get("upper").and_then(|u| u.as_f64()).unwrap_or(0.0);
            assert!(
                lower <= upper,
                "Bound soundness violated: lower {} > upper {}",
                lower,
                upper
            );
        }
    }
}
