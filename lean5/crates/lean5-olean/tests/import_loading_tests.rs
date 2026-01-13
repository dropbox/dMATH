//! Integration tests for .olean module import functionality.
//!
//! These tests validate importing Lean 4 standard library modules into the
//! Lean5 kernel environment. They require a Lean 4 installation via elan.

// Use mimalloc globally when the feature is enabled
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use lean5_kernel::env::Environment;
use lean5_kernel::name::Name;
use lean5_olean::{
    default_search_paths, load_module_with_deps, load_module_with_deps_cached,
    load_module_with_deps_parallel, load_olean_file, ModuleCache,
};

fn get_lean_lib_path() -> Option<std::path::PathBuf> {
    default_search_paths()
        .into_iter()
        .find(|p| p.join("Init/Prelude.olean").exists())
}

#[test]
fn test_load_prelude_into_environment() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let prelude_path = lib_path.join("Init/Prelude.olean");
    if !prelude_path.exists() {
        eprintln!("Skipping test: Init/Prelude.olean not found at {prelude_path:?}");
        return;
    }

    let mut env = Environment::default();
    let summary = load_olean_file(&mut env, &prelude_path).expect("Failed to load Prelude.olean");

    println!(
        "Prelude summary: added={}, skipped={}, duplicates={}",
        summary.added_constants,
        summary.skipped_constants.len(),
        summary.duplicate_constants
    );
    if !summary.skipped_constants.is_empty() {
        println!(
            "Sample skipped constants: {:?}",
            summary.skipped_constants.iter().take(5).collect::<Vec<_>>()
        );
    }

    assert_eq!(summary.module_name.as_deref(), Some("Init.Prelude"));
    assert!(
        summary.added_constants > 0,
        "Expected constants to be added"
    );

    let nat = Name::from_string("Nat");
    assert!(
        env.get_const(&nat).is_some(),
        "Nat constant should be available after import"
    );
}

#[test]
fn test_load_core_with_dependencies() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.Core", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Core with dependencies");

    println!(
        "Loaded modules (name, added, skipped, dupes): {:?}",
        summaries
            .iter()
            .map(|s| (
                s.module_name.clone(),
                s.added_constants,
                s.skipped_constants.len(),
                s.duplicate_constants
            ))
            .collect::<Vec<_>>()
    );

    assert!(
        !summaries.is_empty(),
        "Expected at least one module to be loaded"
    );
    assert!(
        summaries
            .iter()
            .any(|s| s.module_name.as_deref() == Some("Init.Prelude")),
        "Prelude should be loaded as a dependency"
    );

    // Nat comes from Prelude, so loading Core recursively should import it.
    assert!(
        env.get_const(&Name::from_string("Nat")).is_some(),
        "Nat constant should exist after recursive import"
    );
}

#[test]
fn test_load_init_data_list_basic() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.List.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.List.Basic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    let total_skipped: usize = summaries.iter().map(|s| s.skipped_constants.len()).sum();

    println!(
        "Init.Data.List.Basic: {} modules, {} added, {} skipped",
        summaries.len(),
        total_added,
        total_skipped
    );

    for summary in &summaries {
        if !summary.skipped_constants.is_empty() {
            println!(
                "  Skipped in {:?}: {}",
                summary.module_name,
                summary.skipped_constants.len()
            );
        }
    }

    assert!(total_added > 0, "Expected constants to be added");
}

#[test]
fn test_load_multiple_init_modules() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Test loading a variety of Init modules with their dependencies
    let modules = [
        "Init.Data.Nat.Basic",
        "Init.Data.List.Basic",
        "Init.Data.Array.Basic",
        "Init.Data.String.Basic",
        "Init.Control.Basic",
        "Init.Data.Option.Basic",
    ];

    let mut total_added = 0;
    let mut total_skipped = 0;
    let mut total_modules = 0;
    let mut failures: Vec<(String, String)> = Vec::new();

    for module in modules {
        let mut env = Environment::default();
        match load_module_with_deps(&mut env, module, std::slice::from_ref(&lib_path)) {
            Ok(summaries) => {
                let added: usize = summaries.iter().map(|s| s.added_constants).sum();
                let skipped: usize = summaries.iter().map(|s| s.skipped_constants.len()).sum();
                total_added += added;
                total_skipped += skipped;
                total_modules += summaries.len();
                println!(
                    "{}: {} modules loaded, {} added, {} skipped",
                    module,
                    summaries.len(),
                    added,
                    skipped
                );

                // Report any skipped constants
                for summary in &summaries {
                    for skip in &summary.skipped_constants {
                        println!(
                            "  Skipped in {:?}: {} - {}",
                            summary.module_name, skip.name, skip.reason
                        );
                    }
                }
            }
            Err(e) => {
                failures.push((module.to_string(), e.to_string()));
                println!("FAILED {module}: {e}");
            }
        }
    }

    println!("\n=== Summary ===");
    println!(
        "Total: {total_modules} modules, {total_added} constants added, {total_skipped} skipped"
    );
    println!("Failures: {}", failures.len());

    assert!(
        failures.is_empty(),
        "Failed to load some modules: {failures:?}"
    );

    // Ensure we loaded a reasonable number of constants
    assert!(
        total_added > 1000,
        "Expected at least 1000 constants, got {total_added}"
    );

    // Skipped constants should be minimal
    let skip_ratio = total_skipped as f64 / (total_added + total_skipped) as f64;
    assert!(
        skip_ratio < 0.01,
        "Skip ratio too high: {:.2}% ({} skipped / {} total)",
        skip_ratio * 100.0,
        total_skipped,
        total_added + total_skipped
    );
}

#[test]
fn test_load_std_modules() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Test loading some Std modules
    let modules = ["Std.Data.HashMap", "Std.Data.HashSet"];

    let mut total_added = 0;
    let mut total_skipped = 0;
    let mut total_modules = 0;
    let mut failures: Vec<(String, String)> = Vec::new();

    for module in modules {
        let mut env = Environment::default();
        match load_module_with_deps(&mut env, module, std::slice::from_ref(&lib_path)) {
            Ok(summaries) => {
                let added: usize = summaries.iter().map(|s| s.added_constants).sum();
                let skipped: usize = summaries.iter().map(|s| s.skipped_constants.len()).sum();
                total_added += added;
                total_skipped += skipped;
                total_modules += summaries.len();
                println!(
                    "{}: {} modules loaded, {} added, {} skipped",
                    module,
                    summaries.len(),
                    added,
                    skipped
                );
                // Print details about skipped constants
                for summary in &summaries {
                    for skip in &summary.skipped_constants {
                        println!(
                            "  SKIPPED in {:?}: {} - {}",
                            summary.module_name, skip.name, skip.reason
                        );
                    }
                }
            }
            Err(e) => {
                failures.push((module.to_string(), e.to_string()));
                println!("FAILED {module}: {e}");
            }
        }
    }

    println!("\n=== Std Summary ===");
    println!(
        "Total: {} modules, {} added, {} skipped, {} failures",
        total_modules,
        total_added,
        total_skipped,
        failures.len()
    );

    assert!(
        failures.is_empty(),
        "Failed to load some Std modules: {failures:?}"
    );

    assert!(
        total_added > 5000,
        "Expected at least 5000 constants from Std, got {total_added}"
    );
}

#[test]
fn test_load_lean_compiler_modules() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Test loading Lean compiler modules - these are large and complex
    let modules = [
        "Lean.Data.Name",
        "Lean.Expr",
        "Lean.Declaration",
        "Lean.Environment",
    ];

    let mut total_added = 0;
    let mut total_skipped = 0;
    let mut total_modules = 0;
    let mut failures: Vec<(String, String)> = Vec::new();

    for module in modules {
        let mut env = Environment::default();
        match load_module_with_deps(&mut env, module, std::slice::from_ref(&lib_path)) {
            Ok(summaries) => {
                let added: usize = summaries.iter().map(|s| s.added_constants).sum();
                let skipped: usize = summaries.iter().map(|s| s.skipped_constants.len()).sum();
                total_added += added;
                total_skipped += skipped;
                total_modules += summaries.len();
                println!(
                    "{}: {} modules loaded, {} added, {} skipped",
                    module,
                    summaries.len(),
                    added,
                    skipped
                );
                // Print details about skipped constants
                for summary in &summaries {
                    for skip in &summary.skipped_constants {
                        println!(
                            "  SKIPPED in {:?}: {} - {}",
                            summary.module_name, skip.name, skip.reason
                        );
                    }
                }
            }
            Err(e) => {
                failures.push((module.to_string(), e.to_string()));
                println!("FAILED {module}: {e}");
            }
        }
    }

    println!("\n=== Lean Compiler Summary ===");
    println!(
        "Total: {} modules, {} added, {} skipped, {} failures",
        total_modules,
        total_added,
        total_skipped,
        failures.len()
    );

    assert!(
        failures.is_empty(),
        "Failed to load some Lean compiler modules: {failures:?}"
    );

    assert!(
        total_added > 10000,
        "Expected at least 10000 constants from Lean compiler, got {total_added}"
    );
}

#[test]
fn test_typecheck_imported_constants() {
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.Nat.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.Nat.Basic");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!("Loaded {total_added} constants");

    // Type-check some well-known constants
    let test_constants = [
        "Nat",
        "Nat.succ",
        "Nat.zero",
        "Nat.add",
        "Bool",
        "Bool.true",
        "Bool.false",
        "List",
        "List.nil",
        "List.cons",
    ];

    let mut successes = 0;
    let mut failures = 0;

    for const_name in test_constants {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            // Try to type-check the type of the constant
            let type_ = &const_info.type_;
            let mut tc = TypeChecker::new(&env);
            match tc.infer_type(type_) {
                Ok(sort) => {
                    println!("  {const_name} : {type_:?} (inferred: {sort:?})");
                    successes += 1;
                }
                Err(e) => {
                    println!("  {const_name} : {type_:?} (TYPE ERROR: {e:?})");
                    failures += 1;
                }
            }
        } else {
            println!("  {const_name} NOT FOUND");
        }
    }

    println!("\n=== Type-checking Summary ===");
    println!("Successes: {successes}, Failures: {failures}");

    // Most should succeed (some may have unknown dependencies)
    assert!(
        successes >= 5,
        "Expected at least 5 constants to type-check, got {successes}"
    );
}

#[test]
fn test_typecheck_definition_values() {
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.Nat.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.Nat.Basic");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!("Loaded {total_added} constants");

    // Type-check definition values: infer type of value and check it matches declared type
    // Focus on definitions that have values (not just axioms/inductives)
    let test_definitions = [
        "Nat.add", "Nat.mul", "Nat.sub", "Nat.beq", "Nat.ble", "Bool.and", "Bool.or", "Bool.not",
        "decide", "ite",
    ];

    let mut value_successes = 0;
    let mut type_successes = 0;
    let mut value_failures = 0;
    let mut not_found = 0;
    let mut no_value = 0;

    for const_name in test_definitions {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            // Type-check the type first
            let mut tc = TypeChecker::new(&env);
            match tc.infer_type(&const_info.type_) {
                Ok(_) => type_successes += 1,
                Err(e) => {
                    println!("  {const_name} type error: {e:?}");
                }
            }

            // Type-check the value if present
            if let Some(ref value) = const_info.value {
                let mut tc = TypeChecker::new(&env);
                match tc.infer_type(value) {
                    Ok(inferred_type) => {
                        // Check that inferred type is definitionally equal to declared type
                        let tc = TypeChecker::new(&env);
                        let is_def_eq = tc.is_def_eq(&inferred_type, &const_info.type_);
                        if is_def_eq {
                            println!("  {const_name} value ✓ (type match)");
                            value_successes += 1;
                        } else {
                            println!(
                                "  {} value: inferred {:?}, declared {:?} (MISMATCH)",
                                const_name, inferred_type, const_info.type_
                            );
                            value_failures += 1;
                        }
                    }
                    Err(e) => {
                        println!("  {const_name} value error: {e:?}");
                        value_failures += 1;
                    }
                }
            } else {
                println!("  {const_name} has no value (axiom/inductive)");
                no_value += 1;
            }
        } else {
            println!("  {const_name} NOT FOUND");
            not_found += 1;
        }
    }

    println!("\n=== Definition Type-checking Summary ===");
    println!(
        "Value successes: {value_successes}, Value failures: {value_failures}, No value: {no_value}, Not found: {not_found}"
    );
    println!("Type successes: {type_successes}");

    // We expect most types to check and some values to check
    // Definitions may not all be present or may have primitives
    assert!(
        type_successes >= 5,
        "Expected at least 5 types to check, got {type_successes}"
    );
}

#[test]
fn test_load_init_meta() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Init.Meta is a large module that tests reflection/metaprogramming
    let mut env = Environment::default();
    match load_module_with_deps(&mut env, "Init.Meta", std::slice::from_ref(&lib_path)) {
        Ok(summaries) => {
            let total_modules = summaries.len();
            let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
            let total_skipped: usize = summaries.iter().map(|s| s.skipped_constants.len()).sum();

            println!(
                "Init.Meta: {total_modules} modules, {total_added} constants added, {total_skipped} skipped"
            );

            // Print sample skipped constants
            for summary in &summaries {
                for skip in summary.skipped_constants.iter().take(3) {
                    println!(
                        "  SKIPPED in {:?}: {} - {}",
                        summary.module_name, skip.name, skip.reason
                    );
                }
            }

            // Init.Meta should have a substantial number of constants
            assert!(
                total_added > 15000,
                "Expected > 15000 constants from Init.Meta, got {total_added}"
            );
        }
        Err(e) => {
            panic!("Failed to load Init.Meta: {e}");
        }
    }
}

#[test]
fn test_olean_loading_performance() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Measure loading performance for Init.Core (medium-sized module)
    let start = std::time::Instant::now();

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.Core", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Core");

    let elapsed = start.elapsed();
    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();

    let constants_per_sec = total_added as f64 / elapsed.as_secs_f64();
    let us_per_constant = elapsed.as_micros() as f64 / total_added as f64;

    println!("\n=== Performance Summary (Init.Core) ===");
    println!("Time: {elapsed:?}");
    println!("Constants: {total_added}");
    println!("Constants/sec: {constants_per_sec:.0}");
    println!("µs/constant: {us_per_constant:.2}");

    // Performance target: should load at least 500 constants/sec in release mode
    // Debug mode is ~20x slower, so skip assertion there
    #[cfg(not(debug_assertions))]
    assert!(
        constants_per_sec > 500.0,
        "Expected > 500 constants/sec, got {:.0}",
        constants_per_sec
    );
    #[cfg(debug_assertions)]
    if constants_per_sec < 500.0 {
        eprintln!(
            "Note: {constants_per_sec:.0} const/sec is below threshold, but this is debug mode"
        );
    }
}

#[test]
fn test_profile_loading_breakdown() {
    use lean5_olean::parse_module;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Profile individual phases for Init.Prelude (a representative module)
    let prelude_path = lib_path.join("Init/Prelude.olean");
    if !prelude_path.exists() {
        eprintln!("Skipping test: Init/Prelude.olean not found");
        return;
    }

    // Measure file read time
    let start = std::time::Instant::now();
    let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
    let read_time = start.elapsed();

    // Measure parsing time
    let start = std::time::Instant::now();
    let module = parse_module(&bytes).expect("Failed to parse module");
    let parse_time = start.elapsed();

    // Measure environment registration time
    let start = std::time::Instant::now();
    let mut env = Environment::default();
    let summary =
        lean5_olean::load_parsed_module(&mut env, module.clone(), Some("Init.Prelude".to_string()))
            .expect("Failed to load module");
    let load_time = start.elapsed();

    let total_time = read_time + parse_time + load_time;
    let file_size_kb = bytes.len() as f64 / 1024.0;

    println!("\n=== Init/Prelude.olean Profile ===");
    println!("File size: {file_size_kb:.1} KB");
    println!("Constants parsed: {}", module.constants.len());
    println!("Constants added: {}", summary.added_constants);
    println!();
    println!("Phase breakdown:");
    println!(
        "  File read:    {:>8.2}ms ({:>5.1}%)",
        read_time.as_secs_f64() * 1000.0,
        read_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "  Parse module: {:>8.2}ms ({:>5.1}%)",
        parse_time.as_secs_f64() * 1000.0,
        parse_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "  Load to env:  {:>8.2}ms ({:>5.1}%)",
        load_time.as_secs_f64() * 1000.0,
        load_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!("  -----------");
    println!(
        "  Total:        {:>8.2}ms",
        total_time.as_secs_f64() * 1000.0
    );
    println!();
    println!("Performance:");
    println!(
        "  Read throughput: {:.1} MB/s",
        file_size_kb / 1024.0 / read_time.as_secs_f64()
    );
    println!(
        "  Parse rate: {:.0} constants/sec",
        module.constants.len() as f64 / parse_time.as_secs_f64()
    );
    println!(
        "  Load rate: {:.0} constants/sec",
        summary.added_constants as f64 / load_time.as_secs_f64()
    );

    // Profile full dependency loading
    println!("\n=== Init.Core Full Load Profile ===");
    let mut env = Environment::default();

    let start = std::time::Instant::now();
    let summaries = load_module_with_deps(&mut env, "Init.Core", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Core");
    let total_load_time = start.elapsed();

    let total_constants: usize = summaries.iter().map(|s| s.added_constants).sum();
    let total_modules = summaries.len();

    println!("Modules: {total_modules}");
    println!("Total constants: {total_constants}");
    println!("Total time: {total_load_time:?}");
    println!(
        "Avg per module: {:.2}ms",
        total_load_time.as_secs_f64() * 1000.0 / total_modules as f64
    );
    println!(
        "Constants/sec: {:.0}",
        total_constants as f64 / total_load_time.as_secs_f64()
    );
}

#[test]
fn test_cached_loading_performance() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let cache = ModuleCache::new();

    // First load - cold cache
    let start = std::time::Instant::now();
    let mut env1 = Environment::default();
    let summaries1 = load_module_with_deps_cached(
        &mut env1,
        "Init.Core",
        std::slice::from_ref(&lib_path),
        &cache,
    )
    .expect("First load failed");
    let cold_time = start.elapsed();
    let total1: usize = summaries1.iter().map(|s| s.added_constants).sum();

    println!("\n=== Cached Loading Performance ===");
    println!("Cache size after first load: {} modules", cache.len());
    println!("Cold cache load: {cold_time:?} ({total1} constants)");

    // Second load - warm cache (same module)
    let start = std::time::Instant::now();
    let mut env2 = Environment::default();
    let summaries2 = load_module_with_deps_cached(
        &mut env2,
        "Init.Core",
        std::slice::from_ref(&lib_path),
        &cache,
    )
    .expect("Second load failed");
    let warm_time = start.elapsed();
    let total2: usize = summaries2.iter().map(|s| s.added_constants).sum();

    println!("Warm cache load: {warm_time:?} ({total2} constants)");
    println!(
        "Speedup: {:.2}x",
        cold_time.as_secs_f64() / warm_time.as_secs_f64()
    );

    // Load a different module that shares dependencies
    let start = std::time::Instant::now();
    let mut env3 = Environment::default();
    let summaries3 = load_module_with_deps_cached(
        &mut env3,
        "Init.Data.Nat.Basic",
        std::slice::from_ref(&lib_path),
        &cache,
    )
    .expect("Third load failed");
    let partial_warm_time = start.elapsed();
    let total3: usize = summaries3.iter().map(|s| s.added_constants).sum();

    println!(
        "Partial cache load (Init.Data.Nat.Basic): {partial_warm_time:?} ({total3} constants)"
    );
    println!("Cache size after all loads: {} modules", cache.len());

    // Verify correctness
    assert_eq!(total1, total2, "Same module should load same constants");
    // Note: warm_time may not be faster than cold_time because:
    // 1. The cache only saves parsing time, not conversion/registration time
    // 2. Conversion + registration dominates the total time
    // 3. Clone overhead for ParsedModule may exceed parsing savings for small modules
    // The real benefit of caching is for incremental workflows loading different modules
    // that share dependencies (e.g., loading Init.Core, then Init.Data.Nat.Basic reuses
    // the already-parsed Init.Prelude and Init.Core modules).
    let speedup = cold_time.as_secs_f64() / warm_time.as_secs_f64();
    if speedup < 1.0 {
        println!(
            "Warning: warm cache slower than cold ({speedup:.2}x) - expected when conversion dominates"
        );
    }

    // Compare with uncached loading
    let start = std::time::Instant::now();
    let mut env4 = Environment::default();
    let summaries4 = load_module_with_deps(&mut env4, "Init.Core", std::slice::from_ref(&lib_path))
        .expect("Uncached load failed");
    let uncached_time = start.elapsed();
    let total4: usize = summaries4.iter().map(|s| s.added_constants).sum();

    println!("\nComparison:");
    println!("Uncached: {uncached_time:?} ({total4} constants)");
    println!("Warm cache: {warm_time:?}");
    println!(
        "Cache benefit: {:.1}% faster",
        (1.0 - warm_time.as_secs_f64() / uncached_time.as_secs_f64()) * 100.0
    );
}

#[test]
fn test_cached_loading_correctness() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let cache = ModuleCache::new();

    // Load with cache
    let mut env_cached = Environment::default();
    let summaries_cached = load_module_with_deps_cached(
        &mut env_cached,
        "Init.Core",
        std::slice::from_ref(&lib_path),
        &cache,
    )
    .expect("Cached load failed");

    // Load without cache
    let mut env_uncached = Environment::default();
    let summaries_uncached = load_module_with_deps(
        &mut env_uncached,
        "Init.Core",
        std::slice::from_ref(&lib_path),
    )
    .expect("Uncached load failed");

    let cached_total: usize = summaries_cached.iter().map(|s| s.added_constants).sum();
    let uncached_total: usize = summaries_uncached.iter().map(|s| s.added_constants).sum();

    // Both should produce identical results
    assert_eq!(
        cached_total, uncached_total,
        "Cached and uncached should load same constants"
    );
    assert_eq!(
        summaries_cached.len(),
        summaries_uncached.len(),
        "Should load same number of modules"
    );

    // Verify specific constants exist in both
    let test_names = ["Nat", "Bool", "List", "String"];
    for name in test_names {
        let n = Name::from_string(name);
        assert!(
            env_cached.get_const(&n).is_some(),
            "{name} should exist in cached env"
        );
        assert!(
            env_uncached.get_const(&n).is_some(),
            "{name} should exist in uncached env"
        );
    }
}

#[test]
fn test_profile_registration_phases() {
    use lean5_olean::parse_module;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let prelude_path = lib_path.join("Init/Prelude.olean");
    if !prelude_path.exists() {
        eprintln!("Skipping test: Init/Prelude.olean not found");
        return;
    }

    // Read and parse module
    let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
    let module = parse_module(&bytes).expect("Failed to parse module");

    // Count constants by type
    let mut inductives = 0;
    let mut constructors = 0;
    let mut recursors = 0;
    let mut others = 0;
    for c in &module.constants {
        match c.kind {
            lean5_olean::module::ConstantKind::Inductive => inductives += 1,
            lean5_olean::module::ConstantKind::Constructor => constructors += 1,
            lean5_olean::module::ConstantKind::Recursor => recursors += 1,
            _ => others += 1,
        }
    }

    println!("\n=== Constant Type Breakdown ===");
    println!("Inductives:   {inductives:>5}");
    println!("Constructors: {constructors:>5}");
    println!("Recursors:    {recursors:>5}");
    println!("Others:       {others:>5}");
    println!("Total:        {:>5}", module.constants.len());

    // Profile multiple iterations for more accurate timing
    const ITERATIONS: u32 = 5;
    let mut total_time = std::time::Duration::ZERO;

    for _ in 0..ITERATIONS {
        let module_clone = module.clone();
        let mut env = Environment::default();
        let start = std::time::Instant::now();
        let _summary = lean5_olean::load_parsed_module(
            &mut env,
            module_clone,
            Some("Init.Prelude".to_string()),
        )
        .expect("Failed to load module");
        total_time += start.elapsed();
    }

    let avg_time = total_time / ITERATIONS;
    let constants_per_sec = module.constants.len() as f64 / avg_time.as_secs_f64();
    let us_per_constant = avg_time.as_micros() as f64 / module.constants.len() as f64;

    println!("\n=== Registration Performance ({ITERATIONS} iterations) ===");
    println!("Average time: {avg_time:?}");
    println!("Constants/sec: {constants_per_sec:.0}");
    println!("µs/constant: {us_per_constant:.2}");
}

#[test]
fn test_batch_registration_with_capacity() {
    use lean5_olean::parse_module;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let prelude_path = lib_path.join("Init/Prelude.olean");
    if !prelude_path.exists() {
        eprintln!("Skipping test: Init/Prelude.olean not found");
        return;
    }

    let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
    let module = parse_module(&bytes).expect("Failed to parse module");
    let num_constants = module.constants.len();

    // Test with default environment (no pre-allocation)
    const ITERATIONS: u32 = 5;
    let mut default_time = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let module_clone = module.clone();
        let mut env = Environment::default();
        let start = std::time::Instant::now();
        let _summary = lean5_olean::load_parsed_module(
            &mut env,
            module_clone,
            Some("Init.Prelude".to_string()),
        )
        .expect("Failed to load module");
        default_time += start.elapsed();
    }
    let avg_default = default_time / ITERATIONS;

    // Test with pre-allocated environment
    let mut preallocated_time = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let module_clone = module.clone();
        let mut env = Environment::with_capacity(num_constants);
        let start = std::time::Instant::now();
        let _summary = lean5_olean::load_parsed_module(
            &mut env,
            module_clone,
            Some("Init.Prelude".to_string()),
        )
        .expect("Failed to load module");
        preallocated_time += start.elapsed();
    }
    let avg_preallocated = preallocated_time / ITERATIONS;

    let speedup = avg_default.as_secs_f64() / avg_preallocated.as_secs_f64();

    println!("\n=== Capacity Pre-allocation Comparison ===");
    println!("Default env:      {avg_default:?}");
    println!("Pre-allocated:    {avg_preallocated:?}");
    println!("Speedup:          {speedup:.2}x");
    println!("Improvement:      {:.1}%", (speedup - 1.0) * 100.0);
}

#[test]
fn test_profile_with_release_build() {
    use lean5_olean::parse_module;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let prelude_path = lib_path.join("Init/Prelude.olean");
    if !prelude_path.exists() {
        eprintln!("Skipping test: Init/Prelude.olean not found");
        return;
    }

    let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
    let module = parse_module(&bytes).expect("Failed to parse module");

    // Run multiple iterations to get stable timing
    const WARMUP: u32 = 2;
    const ITERATIONS: u32 = 10;

    // Warmup
    for _ in 0..WARMUP {
        let module_clone = module.clone();
        let mut env = Environment::default();
        let _ = lean5_olean::load_parsed_module(&mut env, module_clone, None);
    }

    // Measure
    let mut times = Vec::with_capacity(ITERATIONS as usize);
    for _ in 0..ITERATIONS {
        let module_clone = module.clone();
        let mut env = Environment::default();
        let start = std::time::Instant::now();
        let _summary = lean5_olean::load_parsed_module(
            &mut env,
            module_clone,
            Some("Init.Prelude".to_string()),
        )
        .expect("Failed to load module");
        times.push(start.elapsed());
    }

    times.sort();
    let median = times[ITERATIONS as usize / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    let avg: std::time::Duration = times.iter().sum::<std::time::Duration>() / ITERATIONS;

    let constants_per_sec = module.constants.len() as f64 / median.as_secs_f64();

    println!("\n=== Loading Performance Stats ({ITERATIONS} iterations) ===");
    println!("Min:     {min:?}");
    println!("Median:  {median:?}");
    println!("Mean:    {avg:?}");
    println!("Max:     {max:?}");
    println!("Constants/sec (median): {constants_per_sec:.0}");
}

#[test]
fn test_name_interning_effectiveness() {
    use lean5_kernel::name::NameInterner;
    use lean5_olean::parse_module;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let prelude_path = lib_path.join("Init/Prelude.olean");
    if !prelude_path.exists() {
        eprintln!("Skipping test: Init/Prelude.olean not found");
        return;
    }

    // Clear the interner to get fresh stats
    NameInterner::global().clear();
    let initial_count = NameInterner::global().len();
    assert_eq!(initial_count, 0, "Interner should be empty after clear");

    // Load Prelude
    let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
    let module = parse_module(&bytes).expect("Failed to parse module");
    let num_constants = module.constants.len();

    let mut env = Environment::default();
    let _ = lean5_olean::load_parsed_module(&mut env, module, Some("Init.Prelude".to_string()))
        .expect("Failed to load module");

    // Check interning stats
    let interned_count = NameInterner::global().len();

    println!("\n=== Name Interning Stats ===");
    println!("Constants loaded: {num_constants}");
    println!("Unique names interned: {interned_count}");
    println!(
        "Reuse ratio: {:.1}x (lower means more reuse)",
        interned_count as f64 / num_constants as f64
    );

    // Each constant uses multiple names (constant name, type names, expression names),
    // so total interned names will exceed constant count. We expect some reuse though -
    // common names like "Nat", "Bool", "Prop" appear across many constants.
    // A ratio < 10 indicates meaningful interning is occurring.
    let ratio = interned_count as f64 / num_constants as f64;
    assert!(
        ratio < 10.0,
        "Expected meaningful name reuse (ratio < 10); got {interned_count} interned for {num_constants} constants (ratio {ratio:.1})"
    );
}

#[test]
fn test_init_core_with_interning() {
    use lean5_kernel::name::NameInterner;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Clear interner for fresh measurement
    NameInterner::global().clear();

    let start = std::time::Instant::now();
    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.Core", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Core");
    let elapsed = start.elapsed();

    let total_constants: usize = summaries.iter().map(|s| s.added_constants).sum();
    let interned_count = NameInterner::global().len();

    println!("\n=== Init.Core Loading with Interning ===");
    println!("Total time: {elapsed:?}");
    println!("Constants loaded: {total_constants}");
    println!("Unique names interned: {interned_count}");
    println!(
        "Constants per second: {:.0}",
        total_constants as f64 / elapsed.as_secs_f64()
    );
}

#[test]
fn test_profile_parallel_phases() {
    use lean5_olean::{parse_imports_only, parse_module};
    use rayon::prelude::*;
    use std::collections::HashSet;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Test with Init.Core for a representative workload
    let module_name = "Init.Core";
    let paths = std::slice::from_ref(&lib_path);

    // Phase 1: Discover dependency graph using fast imports-only parsing
    // (This matches the actual implementation in discover_and_parse_parallel)
    let start = std::time::Instant::now();
    let mut discovered: HashSet<String> = HashSet::new();
    let mut order: Vec<String> = Vec::new();
    let mut module_bytes: std::collections::HashMap<String, Vec<u8>> =
        std::collections::HashMap::new();
    let mut current = vec![module_name.to_string()];
    discovered.insert(module_name.to_string());

    while !current.is_empty() {
        let mut next = Vec::new();
        for mod_name in &current {
            let rel_path = mod_name.replace('.', "/") + ".olean";
            for base in paths {
                let path = base.join(&rel_path);
                if path.exists() {
                    let bytes = std::fs::read(&path).expect("read");
                    // Use fast imports-only parsing (33000x faster than full parse!)
                    let imports = parse_imports_only(&bytes).expect("parse imports");
                    for imp in &imports {
                        if !imp.module_name.is_empty() && !discovered.contains(&imp.module_name) {
                            discovered.insert(imp.module_name.clone());
                            next.push(imp.module_name.clone());
                        }
                    }
                    // Cache bytes for full parsing later
                    module_bytes.insert(mod_name.clone(), bytes);
                    break;
                }
            }
        }
        order.extend(current);
        current = next;
    }
    order.reverse();
    let discovery_time = start.elapsed();

    // Phase 2: Parse all modules in parallel using cached bytes (no disk I/O)
    let start = std::time::Instant::now();
    let parsed_modules: Vec<(String, lean5_olean::module::ParsedModule)> = order
        .par_iter()
        .filter_map(|mod_name| {
            let bytes = module_bytes.get(mod_name)?;
            let parsed = parse_module(bytes).ok()?;
            Some((mod_name.clone(), parsed))
        })
        .collect();
    let parse_time = start.elapsed();

    // Phase 3: Convert constants in parallel (simulation)
    let start = std::time::Instant::now();
    let total_constants: usize = parsed_modules.iter().map(|(_, m)| m.constants.len()).sum();
    let conversion_time = start.elapsed();

    // Phase 4: Register in environment (sequential)
    let start = std::time::Instant::now();
    let mut env = Environment::default();
    for (mod_name, parsed) in parsed_modules {
        let _ = lean5_olean::load_parsed_module(&mut env, parsed, Some(mod_name));
    }
    let registration_time = start.elapsed();

    let total_time = discovery_time + parse_time + conversion_time + registration_time;

    println!("\n=== Parallel Loading Phase Breakdown ({module_name}) ===");
    println!("Modules: {}", order.len());
    println!("Constants: {total_constants}");
    println!();
    println!(
        "Phase 1 - Discover deps:  {:>8.2}ms ({:>5.1}%)",
        discovery_time.as_secs_f64() * 1000.0,
        discovery_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "Phase 2 - Parse parallel: {:>8.2}ms ({:>5.1}%)",
        parse_time.as_secs_f64() * 1000.0,
        parse_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "Phase 3 - Count (proxy):  {:>8.2}ms ({:>5.1}%)",
        conversion_time.as_secs_f64() * 1000.0,
        conversion_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "Phase 4 - Registration:   {:>8.2}ms ({:>5.1}%)",
        registration_time.as_secs_f64() * 1000.0,
        registration_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!("  -----------");
    println!(
        "  Total:                  {:>8.2}ms",
        total_time.as_secs_f64() * 1000.0
    );
}

#[test]
fn test_profile_registration_detail() {
    use lean5_olean::parse_module;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Load Init.Core modules
    let modules = ["Init.Prelude", "Init.Core"];
    let mut all_parsed = Vec::new();

    for mod_name in &modules {
        let rel_path = mod_name.replace('.', "/") + ".olean";
        let path = lib_path.join(&rel_path);
        if path.exists() {
            let bytes = std::fs::read(&path).expect("read");
            let parsed = parse_module(&bytes).expect("parse");
            all_parsed.push((mod_name.to_string(), parsed));
        }
    }

    // Profile registration with different capacities
    let total_constants: usize = all_parsed.iter().map(|(_, m)| m.constants.len()).sum();

    println!("\n=== Registration Performance with Capacity ===");
    println!("Total constants to register: {total_constants}");

    // Without pre-allocation
    let start = std::time::Instant::now();
    let mut env_default = Environment::default();
    for (name, parsed) in &all_parsed {
        let cloned = parsed.clone();
        let _ = lean5_olean::load_parsed_module(&mut env_default, cloned, Some(name.clone()));
    }
    let default_time = start.elapsed();

    // With pre-allocation
    let start = std::time::Instant::now();
    let mut env_preallocated = Environment::with_capacity(total_constants);
    for (name, parsed) in &all_parsed {
        let cloned = parsed.clone();
        let _ = lean5_olean::load_parsed_module(&mut env_preallocated, cloned, Some(name.clone()));
    }
    let preallocated_time = start.elapsed();

    println!(
        "Default env:     {:?} ({:.0} constants/sec)",
        default_time,
        total_constants as f64 / default_time.as_secs_f64()
    );
    println!(
        "Pre-allocated:   {:?} ({:.0} constants/sec)",
        preallocated_time,
        total_constants as f64 / preallocated_time.as_secs_f64()
    );
    println!(
        "Speedup: {:.2}x",
        default_time.as_secs_f64() / preallocated_time.as_secs_f64()
    );
}

#[test]
fn test_profile_lean_environment_loading() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Lean.Environment is a large module (163 modules, 47k constants as noted in #492)
    // This tests loading a substantial portion of the Lean compiler infrastructure
    let module = "Lean.Environment";

    println!("\n=== Lean.Environment Loading Profile ===");
    println!("(This is a large workload - 163 modules, ~47k constants)");

    let start = std::time::Instant::now();
    let mut env = Environment::default();
    let summaries = match load_module_with_deps(&mut env, module, std::slice::from_ref(&lib_path)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Load failed: {e}");
            return;
        }
    };
    let elapsed = start.elapsed();
    let total_constants: usize = summaries.iter().map(|s| s.added_constants).sum();
    let num_modules = summaries.len();

    println!("\nLoading results:");
    println!("  Time: {elapsed:?}");
    println!("  Modules: {num_modules}");
    println!("  Constants: {total_constants}");
    println!(
        "  Constants/sec: {:.0}",
        total_constants as f64 / elapsed.as_secs_f64()
    );
    println!(
        "  ms/module: {:.2}",
        elapsed.as_secs_f64() * 1000.0 / num_modules as f64
    );

    // Performance target: should load at least 500 constants/sec in release mode
    // Debug mode is ~20x slower, so skip assertion there
    let constants_per_sec = total_constants as f64 / elapsed.as_secs_f64();
    #[cfg(not(debug_assertions))]
    assert!(
        constants_per_sec > 500.0,
        "Expected > 500 constants/sec, got {:.0}",
        constants_per_sec
    );
    #[cfg(debug_assertions)]
    if constants_per_sec < 500.0 {
        eprintln!(
            "Note: {constants_per_sec:.0} const/sec is below threshold, but this is debug mode"
        );
    }
}

#[test]
fn test_profile_conversion_phases() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Test with Init.Core for a quick workload
    let module_name = "Init.Core";
    let paths = std::slice::from_ref(&lib_path);

    println!("\n=== Conversion Phase Analysis ({module_name}) ===");

    let start = std::time::Instant::now();
    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, module_name, paths).expect("Failed to load module");
    let total_time = start.elapsed();

    let total_constants: usize = summaries.iter().map(|s| s.added_constants).sum();
    let total_modules = summaries.len();

    println!("Load: {total_time:?} ({total_modules} modules, {total_constants} constants)");
    println!(
        "Constants/sec: {:.0}",
        total_constants as f64 / total_time.as_secs_f64()
    );

    // Also test with Lean.Environment for larger workload
    println!("\n=== Lean.Environment Load ===");
    let start = std::time::Instant::now();
    let mut env2 = Environment::default();
    let summaries2 = load_module_with_deps(&mut env2, "Lean.Environment", paths)
        .expect("Failed to load Lean.Environment");
    let total_time2 = start.elapsed();

    let total_constants2: usize = summaries2.iter().map(|s| s.added_constants).sum();
    let total_modules2 = summaries2.len();

    println!("Load: {total_time2:?} ({total_modules2} modules, {total_constants2} constants)");
    println!(
        "Constants/sec: {:.0}",
        total_constants2 as f64 / total_time2.as_secs_f64()
    );
}

#[test]
fn test_profile_file_io_vs_parsing() {
    use lean5_olean::{parse_imports_only, parse_module};

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Profile file I/O vs parsing for different modules
    let modules = [
        ("Init.Prelude", "Init/Prelude.olean"),
        ("Init.Meta", "Init/Meta.olean"),
    ];

    println!("\n=== File I/O vs Parsing Analysis ===");

    for (name, rel_path) in modules {
        let path = lib_path.join(rel_path);
        if !path.exists() {
            println!("Skipping {name} - not found");
            continue;
        }

        // Profile file read
        let start = std::time::Instant::now();
        let bytes = std::fs::read(&path).expect("Failed to read file");
        let read_time = start.elapsed();
        let file_size = bytes.len();

        // Profile imports-only parse
        let start = std::time::Instant::now();
        let imports = parse_imports_only(&bytes).expect("Failed to parse imports");
        let imports_time = start.elapsed();

        // Profile full parse
        let start = std::time::Instant::now();
        let module = parse_module(&bytes).expect("Failed to parse module");
        let parse_time = start.elapsed();

        let read_throughput = file_size as f64 / 1024.0 / 1024.0 / read_time.as_secs_f64();
        let full_parse_speedup = parse_time.as_secs_f64() / imports_time.as_secs_f64();

        println!("\n{name}:");
        println!("  File size:     {:.1} KB", file_size as f64 / 1024.0);
        println!("  Constants:     {}", module.constants.len());
        println!("  Imports:       {}", imports.len());
        println!(
            "  File read:     {:>8.2}ms ({:.0} MB/s)",
            read_time.as_secs_f64() * 1000.0,
            read_throughput
        );
        println!(
            "  Imports-only:  {:>8.2}ms",
            imports_time.as_secs_f64() * 1000.0
        );
        println!(
            "  Full parse:    {:>8.2}ms",
            parse_time.as_secs_f64() * 1000.0
        );
        println!("  Parse speedup: {full_parse_speedup:>8.1}x (imports vs full)");
    }

    // Verify file caching benefit: compare double-read vs cached approach
    println!("\n=== File Caching Benefit ===");
    let meta_path = lib_path.join("Init/Meta.olean");
    if meta_path.exists() {
        // Simulate old approach: read twice
        let start = std::time::Instant::now();
        let bytes1 = std::fs::read(&meta_path).expect("Failed to read file");
        let _ = parse_imports_only(&bytes1).unwrap();
        let bytes2 = std::fs::read(&meta_path).expect("Failed to read file");
        let _ = parse_module(&bytes2).unwrap();
        let old_time = start.elapsed();

        // New approach: read once, use cached bytes
        let start = std::time::Instant::now();
        let bytes = std::fs::read(&meta_path).expect("Failed to read file");
        let _ = parse_imports_only(&bytes).unwrap();
        let _ = parse_module(&bytes).unwrap();
        let new_time = start.elapsed();

        // Use signed arithmetic to handle cases where OS caching makes new approach appear slower
        let savings_ms = (old_time.as_secs_f64() - new_time.as_secs_f64()) * 1000.0;
        let savings_pct = (1.0 - new_time.as_secs_f64() / old_time.as_secs_f64()) * 100.0;

        println!("Init.Meta double-read vs cached:");
        println!("  Old (double read): {old_time:?}");
        println!("  New (cached):      {new_time:?}");
        if savings_ms >= 0.0 {
            println!("  Savings:           {savings_ms:.2}ms ({savings_pct:.1}% faster)");
        } else {
            println!(
                "  Note: OS file cache may have warmed during measurement (new: {:.2}ms slower)",
                -savings_ms
            );
        }
    }
}

#[test]
fn test_profile_std_hashmap_loading() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Check if Std.Data.HashMap exists
    let test_path = lib_path.join("Std/Data/HashMap.olean");
    if !test_path.exists() {
        eprintln!("Skipping test: Std.Data.HashMap not found at {test_path:?}");
        return;
    }

    let module = "Std.Data.HashMap";
    let paths = std::slice::from_ref(&lib_path);

    println!("\n=== Std.Data.HashMap Loading Profile ===");

    let start = std::time::Instant::now();
    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, module, paths).expect("Load failed");
    let elapsed = start.elapsed();
    let total_constants: usize = summaries.iter().map(|s| s.added_constants).sum();
    let num_modules = summaries.len();

    println!("Results:");
    println!("  Time: {elapsed:?}");
    println!("  Modules: {num_modules}");
    println!("  Constants: {total_constants}");
    println!(
        "  Constants/sec: {:.0}",
        total_constants as f64 / elapsed.as_secs_f64()
    );
    println!(
        "  ms/module: {:.2}",
        elapsed.as_secs_f64() * 1000.0 / num_modules as f64
    );

    // Performance target for cold cache loading of large dependency tree
    // Std.Data.HashMap loads ~36k constants from ~130 modules
    // Cold cache performance varies with system load; warm cache achieves 1000+/s
    // Debug mode is ~20x slower, so skip assertion there
    let constants_per_sec = total_constants as f64 / elapsed.as_secs_f64();
    #[cfg(not(debug_assertions))]
    assert!(
        constants_per_sec > 250.0,
        "Expected > 250 constants/sec (cold cache), got {:.0}",
        constants_per_sec
    );
    #[cfg(debug_assertions)]
    if constants_per_sec < 250.0 {
        eprintln!(
            "Note: {constants_per_sec:.0} const/sec is below threshold, but this is debug mode"
        );
    }
}

#[test]
fn test_memory_fragmentation_effect() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Check if Std.Data.HashMap exists for a larger workload
    let test_path = lib_path.join("Std/Data/HashMap.olean");
    if !test_path.exists() {
        eprintln!("Skipping test: Std.Data.HashMap not found at {test_path:?}");
        return;
    }

    let module = "Std.Data.HashMap";
    let paths = std::slice::from_ref(&lib_path);

    println!("\n=== Memory Fragmentation Analysis ===");
    println!("Testing with {module} (130 modules, 36k constants)\n");

    // Test 1: Load into fresh environment (baseline)
    let start = std::time::Instant::now();
    let mut env1 = Environment::default();
    let summaries1 = load_module_with_deps(&mut env1, module, paths).expect("First load failed");
    let time1 = start.elapsed();
    let constants1: usize = summaries1.iter().map(|s| s.added_constants).sum();

    println!(
        "Load 1 (fresh env):    {:>8.2}s ({} constants, {:.0}/s)",
        time1.as_secs_f64(),
        constants1,
        constants1 as f64 / time1.as_secs_f64()
    );

    // Test 2: Load into fresh environment while env1 is still alive
    // This tests memory fragmentation effects
    let start = std::time::Instant::now();
    let mut env2 = Environment::default();
    let summaries2 = load_module_with_deps(&mut env2, module, paths).expect("Second load failed");
    let time2 = start.elapsed();
    let constants2: usize = summaries2.iter().map(|s| s.added_constants).sum();

    println!(
        "Load 2 (env1 alive):   {:>8.2}s ({} constants, {:.0}/s)",
        time2.as_secs_f64(),
        constants2,
        constants2 as f64 / time2.as_secs_f64()
    );

    // Test 3: Drop env1, load into fresh environment (should be faster)
    drop(env1);
    let start = std::time::Instant::now();
    let mut env3 = Environment::default();
    let summaries3 = load_module_with_deps(&mut env3, module, paths).expect("Third load failed");
    let time3 = start.elapsed();
    let constants3: usize = summaries3.iter().map(|s| s.added_constants).sum();

    println!(
        "Load 3 (env1 dropped): {:>8.2}s ({} constants, {:.0}/s)",
        time3.as_secs_f64(),
        constants3,
        constants3 as f64 / time3.as_secs_f64()
    );

    // Test 4: Load while env2 and env3 are alive
    let start = std::time::Instant::now();
    let mut env4 = Environment::default();
    let summaries4 = load_module_with_deps(&mut env4, module, paths).expect("Fourth load failed");
    let time4 = start.elapsed();
    let constants4: usize = summaries4.iter().map(|s| s.added_constants).sum();

    println!(
        "Load 4 (env2,3 alive): {:>8.2}s ({} constants, {:.0}/s)",
        time4.as_secs_f64(),
        constants4,
        constants4 as f64 / time4.as_secs_f64()
    );

    // Analysis
    let slowdown_2vs1 = time2.as_secs_f64() / time1.as_secs_f64();
    let slowdown_3vs1 = time3.as_secs_f64() / time1.as_secs_f64();
    let slowdown_4vs1 = time4.as_secs_f64() / time1.as_secs_f64();

    println!("\n=== Analysis ===");
    println!("Load 2 vs Load 1: {slowdown_2vs1:.2}x");
    println!("Load 3 vs Load 1: {slowdown_3vs1:.2}x (after dropping env1)");
    println!("Load 4 vs Load 1: {slowdown_4vs1:.2}x (with 2 live envs)");

    // Memory pressure test interpretation
    if slowdown_2vs1 > 1.5 {
        println!("\n⚠️  Significant memory fragmentation detected!");
        println!("   Possible causes:");
        println!("   - HashMap allocator contention");
        println!("   - Memory fragmentation from 36k+ constants");
        println!("   - Cache pollution from large existing environment");
    }

    // Keep environments alive for accurate measurements
    let _ = (
        env2.num_constants(),
        env3.num_constants(),
        env4.num_constants(),
    );
}

#[test]
fn test_profile_phase_timing_detailed() {
    use lean5_olean::parse_module;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    // Use Init.Prelude for detailed per-phase profiling
    let prelude_path = lib_path.join("Init/Prelude.olean");
    if !prelude_path.exists() {
        eprintln!("Skipping test: Init/Prelude.olean not found");
        return;
    }

    println!("\n=== Detailed Phase Timing (Init.Prelude) ===");

    // Measure file read
    let start = std::time::Instant::now();
    let bytes = std::fs::read(&prelude_path).expect("Failed to read");
    let read_time = start.elapsed();

    // Measure parsing
    let start = std::time::Instant::now();
    let module = parse_module(&bytes).expect("Failed to parse");
    let parse_time = start.elapsed();

    let num_constants = module.constants.len();

    // Count constant types
    let mut inductives = 0;
    let mut constructors = 0;
    let mut recursors = 0;
    let mut others = 0;
    for c in &module.constants {
        match c.kind {
            lean5_olean::module::ConstantKind::Inductive => inductives += 1,
            lean5_olean::module::ConstantKind::Constructor => constructors += 1,
            lean5_olean::module::ConstantKind::Recursor => recursors += 1,
            _ => others += 1,
        }
    }

    // Measure environment registration (includes conversion + insertion)
    let start = std::time::Instant::now();
    let mut env = Environment::default();
    let summary =
        lean5_olean::load_parsed_module(&mut env, module, Some("Init.Prelude".to_string()))
            .expect("Failed to load");
    let load_time = start.elapsed();

    let total_time = read_time + parse_time + load_time;

    println!("File size: {:.1} KB", bytes.len() as f64 / 1024.0);
    println!("Constants: {num_constants} total");
    println!(
        "  Inductives:   {:>5} ({:.1}%)",
        inductives,
        100.0 * inductives as f64 / num_constants as f64
    );
    println!(
        "  Constructors: {:>5} ({:.1}%)",
        constructors,
        100.0 * constructors as f64 / num_constants as f64
    );
    println!(
        "  Recursors:    {:>5} ({:.1}%)",
        recursors,
        100.0 * recursors as f64 / num_constants as f64
    );
    println!(
        "  Others:       {:>5} ({:.1}%)",
        others,
        100.0 * others as f64 / num_constants as f64
    );

    println!("\nPhase timing:");
    println!(
        "  File read:  {:>8.2}ms ({:>5.1}%)",
        read_time.as_secs_f64() * 1000.0,
        100.0 * read_time.as_secs_f64() / total_time.as_secs_f64()
    );
    println!(
        "  Parse:      {:>8.2}ms ({:>5.1}%)",
        parse_time.as_secs_f64() * 1000.0,
        100.0 * parse_time.as_secs_f64() / total_time.as_secs_f64()
    );
    println!(
        "  Load:       {:>8.2}ms ({:>5.1}%)",
        load_time.as_secs_f64() * 1000.0,
        100.0 * load_time.as_secs_f64() / total_time.as_secs_f64()
    );
    println!("  Total:      {:>8.2}ms", total_time.as_secs_f64() * 1000.0);

    println!("\nPerformance:");
    println!(
        "  Parse rate: {:.0} constants/sec",
        num_constants as f64 / parse_time.as_secs_f64()
    );
    println!(
        "  Load rate:  {:.0} constants/sec",
        summary.added_constants as f64 / load_time.as_secs_f64()
    );
}

#[test]
fn test_parallel_vs_sequential_loading() {
    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let module = "Lean.Environment";
    let paths = std::slice::from_ref(&lib_path);

    println!("\n=== Parallel vs Sequential Loading Comparison ===");
    println!("Module: {module} (163 modules, ~47k constants)");

    // Sequential (cached)
    let cache1 = ModuleCache::new();
    let start = std::time::Instant::now();
    let mut env1 = Environment::default();
    let summaries1 = load_module_with_deps_cached(&mut env1, module, paths, &cache1)
        .expect("Sequential load failed");
    let sequential_time = start.elapsed();
    let sequential_constants: usize = summaries1.iter().map(|s| s.added_constants).sum();

    // Parallel
    let cache2 = ModuleCache::new();
    let start = std::time::Instant::now();
    let mut env2 = Environment::default();
    let summaries2 = load_module_with_deps_parallel(&mut env2, module, paths, &cache2)
        .expect("Parallel load failed");
    let parallel_time = start.elapsed();
    let parallel_constants: usize = summaries2.iter().map(|s| s.added_constants).sum();

    println!("\nResults:");
    println!(
        "  Sequential: {:>8.2}s ({} constants, {:.0} const/sec)",
        sequential_time.as_secs_f64(),
        sequential_constants,
        sequential_constants as f64 / sequential_time.as_secs_f64()
    );
    println!(
        "  Parallel:   {:>8.2}s ({} constants, {:.0} const/sec)",
        parallel_time.as_secs_f64(),
        parallel_constants,
        parallel_constants as f64 / parallel_time.as_secs_f64()
    );

    let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
    println!("\n  Speedup: {speedup:.2}x");

    // Verify correctness - same number of constants loaded
    assert_eq!(
        sequential_constants, parallel_constants,
        "Parallel and sequential should load same number of constants"
    );
}
