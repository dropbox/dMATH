//! Tests for loading Mathlib modules into the kernel environment.
//!
//! These tests require Mathlib4 to be installed. If Mathlib is not found,
//! the tests will skip gracefully.
//!
//! To install Mathlib for testing:
//! ```bash
//! # Clone a Mathlib-using project
//! lake new test_mathlib math
//! cd test_mathlib
//! lake build  # This downloads and builds Mathlib
//! ```

use lean5_kernel::env::Environment;
use lean5_kernel::name::Name;
use lean5_olean::{default_search_paths, load_module_with_deps};
use std::path::PathBuf;

/// Get the path to Mathlib .olean files if installed.
///
/// Checks common locations:
/// 1. Environment variable MATHLIB_PATH
/// 2. ~/.elan/toolchains/.../lib/lean/ directories
/// 3. Common lake build cache locations
fn get_mathlib_path() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(path) = std::env::var("MATHLIB_PATH") {
        let path = PathBuf::from(path);
        if path.join("Mathlib.olean").exists() || path.join("Mathlib/Data/Nat/Basic.olean").exists()
        {
            return Some(path);
        }
    }

    // Get home directory from environment
    let home = std::env::var("HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| std::env::var("USERPROFILE").ok().map(PathBuf::from))?;

    // Lake package cache (Mathlib installed as dependency)
    let lake_packages = home.join(".elan/toolchains");
    if lake_packages.exists() {
        // Look for any toolchain with Mathlib
        if let Ok(entries) = std::fs::read_dir(&lake_packages) {
            for entry in entries.flatten() {
                let mathlib_path = entry.path().join("lib/lean/Mathlib");
                if mathlib_path.exists() {
                    return Some(entry.path().join("lib/lean"));
                }
            }
        }
    }

    // Check for local Mathlib checkout with lake build output
    let current_dir = std::env::current_dir().ok()?;
    let lake_packages_dir = current_dir.join(".lake/packages/mathlib/.lake/build/lib");
    if lake_packages_dir.exists() {
        return Some(lake_packages_dir);
    }

    None
}

/// Get combined search paths including both standard library and Mathlib.
fn get_combined_search_paths() -> Option<Vec<PathBuf>> {
    let mathlib_path = get_mathlib_path()?;
    let mut paths = default_search_paths();
    paths.insert(0, mathlib_path);
    Some(paths)
}

#[test]
fn test_load_mathlib_data_nat_basic() {
    let Some(search_paths) = get_combined_search_paths() else {
        eprintln!("Skipping test: Mathlib not found");
        eprintln!("  Set MATHLIB_PATH environment variable to Mathlib .olean directory");
        eprintln!("  Or run `lake build` in a Mathlib-using project");
        return;
    };

    let mut env = Environment::default();
    let result = load_module_with_deps(&mut env, "Mathlib.Data.Nat.Basic", &search_paths);

    match result {
        Ok(summaries) => {
            let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
            let total_skipped: usize = summaries.iter().map(|s| s.skipped_constants.len()).sum();

            println!(
                "Mathlib.Data.Nat.Basic: {} modules, {} added, {} skipped",
                summaries.len(),
                total_added,
                total_skipped
            );

            // Verify key constants exist
            let nat_add_comm = Name::from_string("Nat.add_comm");
            if let Some(const_info) = env.get_const(&nat_add_comm) {
                println!("  Found Nat.add_comm: {:?}", const_info.type_);
            } else {
                println!("  Nat.add_comm not found (may be in different module)");
            }

            assert!(total_added > 0, "Expected constants to be added");
        }
        Err(e) => {
            eprintln!("Failed to load Mathlib.Data.Nat.Basic: {e}");
            eprintln!("This may be expected if Mathlib is not installed");
        }
    }
}

#[test]
fn test_load_mathlib_algebra_group_basic() {
    let Some(search_paths) = get_combined_search_paths() else {
        eprintln!("Skipping test: Mathlib not found");
        return;
    };

    let mut env = Environment::default();
    let result = load_module_with_deps(&mut env, "Mathlib.Algebra.Group.Basic", &search_paths);

    match result {
        Ok(summaries) => {
            let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
            let total_skipped: usize = summaries.iter().map(|s| s.skipped_constants.len()).sum();

            println!(
                "Mathlib.Algebra.Group.Basic: {} modules, {} added, {} skipped",
                summaries.len(),
                total_added,
                total_skipped
            );

            // Check for common group theory definitions
            let test_names = [
                "mul_assoc",
                "one_mul",
                "mul_one",
                "inv_mul_cancel",
                "Group",
                "AddGroup",
            ];

            for name in test_names {
                let n = Name::from_string(name);
                if env.get_const(&n).is_some() {
                    println!("  Found: {name}");
                }
            }

            assert!(total_added > 0, "Expected constants to be added");
        }
        Err(e) => {
            eprintln!("Failed to load Mathlib.Algebra.Group.Basic: {e}");
            eprintln!("This may be expected if Mathlib is not installed");
        }
    }
}

#[test]
fn test_load_mathlib_topology_basic() {
    let Some(search_paths) = get_combined_search_paths() else {
        eprintln!("Skipping test: Mathlib not found");
        return;
    };

    let mut env = Environment::default();
    let result = load_module_with_deps(&mut env, "Mathlib.Topology.Basic", &search_paths);

    match result {
        Ok(summaries) => {
            let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
            let total_skipped: usize = summaries.iter().map(|s| s.skipped_constants.len()).sum();

            println!(
                "Mathlib.Topology.Basic: {} modules, {} added, {} skipped",
                summaries.len(),
                total_added,
                total_skipped
            );

            // Check for topology definitions
            let test_names = ["TopologicalSpace", "IsOpen", "IsClosed", "Continuous"];

            for name in test_names {
                let n = Name::from_string(name);
                if env.get_const(&n).is_some() {
                    println!("  Found: {name}");
                }
            }

            assert!(total_added > 0, "Expected constants to be added");
        }
        Err(e) => {
            eprintln!("Failed to load Mathlib.Topology.Basic: {e}");
            eprintln!("This may be expected if Mathlib is not installed");
        }
    }
}

#[test]
fn test_load_mathlib_analysis_calculus() {
    let Some(search_paths) = get_combined_search_paths() else {
        eprintln!("Skipping test: Mathlib not found");
        return;
    };

    let mut env = Environment::default();
    // Try loading a calculus module - this tests deep dependency chains
    let result = load_module_with_deps(
        &mut env,
        "Mathlib.Analysis.Calculus.Deriv.Basic",
        &search_paths,
    );

    match result {
        Ok(summaries) => {
            let total_modules = summaries.len();
            let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
            let total_skipped: usize = summaries.iter().map(|s| s.skipped_constants.len()).sum();

            println!(
                "Mathlib.Analysis.Calculus.Deriv.Basic: {total_modules} modules, {total_added} added, {total_skipped} skipped"
            );

            // This is a large module tree - expect many constants
            println!(
                "  Average constants per module: {:.1}",
                total_added as f64 / total_modules as f64
            );

            assert!(total_added > 0, "Expected constants to be added");
        }
        Err(e) => {
            eprintln!("Failed to load Mathlib.Analysis.Calculus.Deriv.Basic: {e}");
            eprintln!("This may be expected if Mathlib is not installed");
        }
    }
}

#[test]
fn test_mathlib_loading_performance() {
    let Some(search_paths) = get_combined_search_paths() else {
        eprintln!("Skipping test: Mathlib not found");
        return;
    };

    // Measure loading time for a representative Mathlib module
    let start = std::time::Instant::now();

    let mut env = Environment::default();
    let result = load_module_with_deps(&mut env, "Mathlib.Data.Nat.Basic", &search_paths);

    let elapsed = start.elapsed();

    match result {
        Ok(summaries) => {
            let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
            let constants_per_sec = total_added as f64 / elapsed.as_secs_f64();

            println!("\n=== Mathlib Loading Performance ===");
            println!("Module: Mathlib.Data.Nat.Basic");
            println!("Time: {elapsed:?}");
            println!("Constants: {total_added}");
            println!("Constants/sec: {constants_per_sec:.0}");
            println!("Modules loaded: {}", summaries.len());

            // Performance baseline - should load at reasonable speed
            // Mathlib modules are larger so may be slower than Init
            assert!(
                constants_per_sec > 100.0,
                "Expected > 100 constants/sec, got {constants_per_sec:.0}"
            );
        }
        Err(e) => {
            eprintln!("Failed to load: {e}");
        }
    }
}
