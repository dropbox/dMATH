//! Advanced tests for .olean module import functionality.
//!
//! These tests validate advanced type system features (data structures,
//! axioms, recursion, automation modules) from Lean 4 standard library.
//! Requires a Lean 4 installation via elan.

use lean5_kernel::env::Environment;
use lean5_kernel::name::Name;
use lean5_olean::{default_search_paths, load_module_with_deps};

fn get_lean_lib_path() -> Option<std::path::PathBuf> {
    default_search_paths()
        .into_iter()
        .find(|p| p.join("Init/Prelude.olean").exists())
}

#[test]
fn test_array_types() {
    // Validate array and related data structure types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.Array.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.Array.Basic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.Array.Basic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test array types
    for const_name in ["Array", "Array.push", "Array.get", "Array.size"] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 array types to validate, got {type_successes}"
    );
}

#[test]
fn test_persistent_array_types() {
    // Validate PersistentArray data structure types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Data.PersistentArray",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Data.PersistentArray with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Data.PersistentArray: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test PersistentArray types
    for const_name in [
        "Lean.PersistentArray",
        "Lean.PersistentArray.push",
        "Lean.PersistentArray.get!",
        "Lean.PersistentArray.size",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 PersistentArray types to validate, got {type_successes}"
    );
}

#[test]
fn test_persistent_hashmap_types() {
    // Validate PersistentHashMap data structure types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Data.PersistentHashMap",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Data.PersistentHashMap with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Data.PersistentHashMap: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test PersistentHashMap types
    for const_name in [
        "Lean.PersistentHashMap",
        "Lean.PersistentHashMap.insert",
        "Lean.PersistentHashMap.find?",
        "Lean.PersistentHashMap.empty",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 PersistentHashMap types to validate, got {type_successes}"
    );
}

#[test]
fn test_trie_types() {
    // Validate Trie data structure types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Lean.Data.Trie", std::slice::from_ref(&lib_path))
            .expect("Failed to load Lean.Data.Trie with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Data.Trie: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Trie types
    for const_name in [
        "Lean.Data.Trie",
        "Lean.Data.Trie.empty",
        "Lean.Data.Trie.insert",
        "Lean.Data.Trie.find?",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 2,
        "Expected at least 2 Trie types to validate, got {type_successes}"
    );
}

#[test]
fn test_bytearray_types() {
    // Validate ByteArray data structure types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.ByteArray",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.ByteArray with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.ByteArray: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test ByteArray types
    for const_name in [
        "ByteArray",
        "ByteArray.push",
        "ByteArray.size",
        "ByteArray.get",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 ByteArray types to validate, got {type_successes}"
    );
}

#[test]
fn test_fin_types() {
    // Validate Fin (bounded natural) types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.Fin.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.Fin.Basic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.Fin.Basic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Fin types
    for const_name in ["Fin", "Fin.val", "Fin.mk", "Fin.last"] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 Fin types to validate, got {type_successes}"
    );
}

#[test]
fn test_int_types() {
    // Validate Int (signed integer) types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.Int.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.Int.Basic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.Int.Basic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Int types
    for const_name in ["Int", "Int.add", "Int.sub", "Int.neg"] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 Int types to validate, got {type_successes}"
    );
}

#[test]
fn test_float_types() {
    // Validate Float (floating point) types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Init.Data.Float", std::slice::from_ref(&lib_path))
            .expect("Failed to load Init.Data.Float with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.Float: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Float types
    for const_name in ["Float", "Float.add", "Float.sub", "Float.mul"] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 Float types to validate, got {type_successes}"
    );
}

#[test]
fn test_list_types() {
    // Validate List data structure types
    use lean5_kernel::tc::TypeChecker;

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
    println!(
        "Init.Data.List.Basic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test List types
    for const_name in [
        "List",
        "List.cons",
        "List.nil",
        "List.append",
        "List.length",
        "List.reverse",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 4,
        "Expected at least 4 List types to validate, got {type_successes}"
    );
}

#[test]
fn test_queue_types() {
    // Validate Queue data structure types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Init.Data.Queue", std::slice::from_ref(&lib_path))
            .expect("Failed to load Init.Data.Queue with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.Queue: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Queue types
    for const_name in [
        "Std.Queue",
        "Std.Queue.empty",
        "Std.Queue.enqueue",
        "Std.Queue.dequeue?",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 Queue types to validate, got {type_successes}"
    );
}

#[test]
fn test_channel_types() {
    // Validate Channel (concurrent communication) types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.Channel",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.Channel with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.Channel: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Channel types
    for const_name in [
        "IO.Channel",
        "IO.Channel.new",
        "IO.Channel.send",
        "IO.Channel.recv",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 Channel types to validate, got {type_successes}"
    );
}

#[test]
fn test_bitvec_types() {
    // Validate BitVec (fixed-width bitvector) types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.BitVec.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.BitVec.Basic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.BitVec.Basic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test BitVec types
    for const_name in [
        "BitVec",
        "BitVec.ofNat",
        "BitVec.toNat",
        "BitVec.and",
        "BitVec.or",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 BitVec types to validate, got {type_successes}"
    );
}

#[test]
fn test_floatarray_types() {
    // Validate FloatArray (packed float array) types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.FloatArray.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.FloatArray.Basic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.FloatArray.Basic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test FloatArray types
    for const_name in [
        "FloatArray",
        "FloatArray.push",
        "FloatArray.get",
        "FloatArray.size",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 FloatArray types to validate, got {type_successes}"
    );
}

#[test]
fn test_format_types() {
    // Validate Format (pretty printing) types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.Format.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.Format.Basic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.Format.Basic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Format types
    for const_name in [
        "Std.Format",
        "Std.Format.text",
        "Std.Format.nil",
        "Std.Format.append",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 Format types to validate, got {type_successes}"
    );
}

#[test]
fn test_random_types() {
    // Validate Random (random number generation) types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.Random",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.Random with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.Random: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Random types
    for const_name in ["RandomGen", "StdGen", "IO.stdGenRef"] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 2,
        "Expected at least 2 Random types to validate, got {type_successes}"
    );
}

#[test]
fn test_getelem_types() {
    // Validate GetElem (indexing operations for collections) types
    // GetElem allows using arr[i] syntax for arrays, lists, etc.
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Init.GetElem", std::slice::from_ref(&lib_path))
            .expect("Failed to load Init.GetElem with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.GetElem: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test GetElem types - the core indexing typeclass
    for const_name in ["GetElem", "GetElem.getElem", "GetElem?", "getElem!"] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 2,
        "Expected at least 2 GetElem types to validate, got {type_successes}"
    );
}

#[test]
fn test_sizeof_types() {
    // Validate SizeOf (size computation for termination proofs) types
    // SizeOf is used to prove recursive functions terminate
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.SizeOf", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.SizeOf with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.SizeOf: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test SizeOf types - used in termination proofs
    for const_name in ["SizeOf", "SizeOf.sizeOf", "sizeOf", "default_has_sizeof"] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 2,
        "Expected at least 2 SizeOf types to validate, got {type_successes}"
    );
}

#[test]
fn test_classical_logic_types() {
    // Validate Classical (classical logic principles) types
    // Classical provides law of excluded middle, choice, etc.
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Init.Classical", std::slice::from_ref(&lib_path))
            .expect("Failed to load Init.Classical with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Classical: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Classical types - axioms of classical logic
    for const_name in [
        "Classical.choice",
        "Classical.em",
        "Classical.propDecidable",
        "Classical.byContradiction",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 2,
        "Expected at least 2 Classical types to validate, got {type_successes}"
    );
}

#[test]
fn test_dynamic_types() {
    // Validate Dynamic (dynamic typing support) types
    // Dynamic allows type-erased values at runtime
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Init.Dynamic", std::slice::from_ref(&lib_path))
            .expect("Failed to load Init.Dynamic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Dynamic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Dynamic types
    for const_name in ["Dynamic", "Dynamic.mk", "Dynamic.get?"] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 2,
        "Expected at least 2 Dynamic types to validate, got {type_successes}"
    );
}

#[test]
fn test_char_types() {
    // Validate Char (Unicode character) types
    // Char is used for single Unicode scalar values
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.Char.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.Char.Basic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.Char.Basic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Char types
    for const_name in [
        "Char",
        "Char.toNat",
        "Char.ofNat",
        "Char.isAlpha",
        "Char.isDigit",
        "Char.isWhitespace",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 4,
        "Expected at least 4 Char types to validate, got {type_successes}"
    );
}

#[test]
fn test_string_types() {
    // Validate String types
    // String is the primary text type in Lean
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.String.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.String.Basic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.String.Basic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test String types
    for const_name in [
        "String",
        "String.length",
        "String.append",
        "String.push",
        "String.isEmpty",
        "String.toList",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 4,
        "Expected at least 4 String types to validate, got {type_successes}"
    );
}

#[test]
fn test_option_types() {
    // Validate Option types
    // Option is the fundamental type for nullable values
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.Option.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.Option.Basic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.Option.Basic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Option types
    for const_name in [
        "Option",
        "Option.some",
        "Option.none",
        "Option.getD",
        "Option.map",
        "Option.bind",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 4,
        "Expected at least 4 Option types to validate, got {type_successes}"
    );
}

#[test]
fn test_inhabited_types() {
    // Validate Inhabited types
    // Inhabited is used for types that have at least one value
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Init.Prelude", std::slice::from_ref(&lib_path))
            .expect("Failed to load Init.Prelude with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Prelude: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Inhabited types
    for const_name in [
        "Inhabited",
        "Inhabited.default",
        "Nonempty",
        "instInhabitedNat",
        "instInhabitedBool",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 Inhabited types to validate, got {type_successes}"
    );
}

#[test]
fn test_subtype_types() {
    // Validate Subtype types
    // Subtype {p : α → Prop} is a type of elements satisfying predicate p
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.Core", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Core with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Core: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Subtype types
    for const_name in ["Subtype", "Subtype.mk", "Subtype.val", "Subtype.property"] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 Subtype types to validate, got {type_successes}"
    );
}

#[test]
fn test_sum_types() {
    // Validate Sum types (Either in Haskell)
    // Sum α β represents a value that is either Left α or Right β
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.Core", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Core with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Core: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Sum types
    for const_name in ["Sum", "Sum.inl", "Sum.inr", "Sum.rec"] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 Sum types to validate, got {type_successes}"
    );
}

#[test]
fn test_prod_types() {
    // Validate Prod types (product/pair types)
    // Prod α β represents a pair of values (α × β)
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.Core", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Core with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Core: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Prod types
    for const_name in ["Prod", "Prod.mk", "Prod.fst", "Prod.snd", "Prod.rec"] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 4,
        "Expected at least 4 Prod types to validate, got {type_successes}"
    );
}

#[test]
fn test_sigma_types() {
    // Validate Sigma types (dependent pair types)
    // Sigma α β represents a dependent pair (a : α) × β a
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.Core", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Core with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Core: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test Sigma types
    for const_name in [
        "Sigma",
        "Sigma.mk",
        "Sigma.fst",
        "Sigma.snd",
        "PSigma",
        "PSigma.mk",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else if env.get_inductive(&name).is_some() {
            println!("  {const_name} is inductive");
            type_successes += 1;
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 4,
        "Expected at least 4 Sigma types to validate, got {type_successes}"
    );
}

#[test]
fn test_funext_propext_axioms() {
    // Validate function extensionality and propositional extensionality axioms
    // These are fundamental axioms of Lean 4's type theory
    // funext: (∀ x, f x = g x) → f = g
    // propext: (a ↔ b) → a = b
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.Core", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Core with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Core: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut axiom_successes = 0;

    // Test funext (function extensionality)
    // axiom funext {α : Sort u} {β : α → Sort v} {f g : (x : α) → β x}
    //   (h : ∀ x, f x = g x) : f = g
    let funext_name = Name::from_string("funext");
    if let Some(funext_const) = env.get_const(&funext_name) {
        println!("\n=== funext axiom ===");
        println!("  type: {:?}", funext_const.type_);
        match tc.infer_type(&funext_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                axiom_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    } else {
        println!("  funext not found");
    }

    // Test propext (propositional extensionality)
    // axiom propext {a b : Prop} (h : a ↔ b) : a = b
    let propext_name = Name::from_string("propext");
    if let Some(propext_const) = env.get_const(&propext_name) {
        println!("\n=== propext axiom ===");
        println!("  type: {:?}", propext_const.type_);
        match tc.infer_type(&propext_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                axiom_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    } else {
        println!("  propext not found");
    }

    // Test proof_irrel (proof irrelevance - implicit via Prop)
    // This is built into the type theory via definitional equality in Prop
    let proof_irrel_name = Name::from_string("proof_irrel");
    if let Some(proof_irrel_const) = env.get_const(&proof_irrel_name) {
        println!("\n=== proof_irrel axiom ===");
        println!("  type: {:?}", proof_irrel_const.type_);
        match tc.infer_type(&proof_irrel_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                axiom_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    }

    // Test congrArg (congruence for function application)
    // theorem congrArg {α : Sort u} {β : Sort v} {a₁ a₂ : α} (f : α → β) (h : a₁ = a₂) : f a₁ = f a₂
    let congr_arg_name = Name::from_string("congrArg");
    if let Some(congr_const) = env.get_const(&congr_arg_name) {
        println!("\n=== congrArg theorem ===");
        println!("  type: {:?}", congr_const.type_);
        match tc.infer_type(&congr_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                axiom_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    }

    // Test congrFun (congruence for equal functions)
    // theorem congrFun {α : Sort u} {β : α → Sort v} {f g : (x : α) → β x} (h : f = g) (a : α) : f a = g a
    let congr_fun_name = Name::from_string("congrFun");
    if let Some(congr_fun_const) = env.get_const(&congr_fun_name) {
        println!("\n=== congrFun theorem ===");
        println!("  type: {:?}", congr_fun_const.type_);
        match tc.infer_type(&congr_fun_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                axiom_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    }

    assert!(
        axiom_successes >= 3,
        "Expected at least 3 extensionality/congruence axioms to validate, got {axiom_successes}"
    );
}

#[test]
fn test_no_confusion_principles() {
    // Validate NoConfusion principles for inductive types
    // NoConfusion proves that different constructors produce different values
    // and that constructors are injective
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.Core", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Core with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Core: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut nc_successes = 0;

    // Test Nat.noConfusion
    // Nat.noConfusion : {P : Sort u} {v1 v2 : Nat} → v1 = v2 → Nat.noConfusionType P v1 v2
    let nat_nc_name = Name::from_string("Nat.noConfusion");
    if let Some(nat_nc_const) = env.get_const(&nat_nc_name) {
        println!("\n=== Nat.noConfusion ===");
        println!("  type: {:?}", nat_nc_const.type_);
        match tc.infer_type(&nat_nc_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                nc_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    } else {
        println!("  Nat.noConfusion not found");
    }

    // Test Nat.noConfusionType
    let nat_nct_name = Name::from_string("Nat.noConfusionType");
    if let Some(nat_nct_const) = env.get_const(&nat_nct_name) {
        println!("\n=== Nat.noConfusionType ===");
        println!("  type: {:?}", nat_nct_const.type_);
        match tc.infer_type(&nat_nct_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                nc_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    }

    // Test Bool.noConfusion
    let bool_nc_name = Name::from_string("Bool.noConfusion");
    if let Some(bool_nc_const) = env.get_const(&bool_nc_name) {
        println!("\n=== Bool.noConfusion ===");
        println!("  type: {:?}", bool_nc_const.type_);
        match tc.infer_type(&bool_nc_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                nc_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    }

    // Test List.noConfusion
    let list_nc_name = Name::from_string("List.noConfusion");
    if let Some(list_nc_const) = env.get_const(&list_nc_name) {
        println!("\n=== List.noConfusion ===");
        println!("  type: {:?}", list_nc_const.type_);
        match tc.infer_type(&list_nc_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                nc_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    }

    // Test Option.noConfusion
    let option_nc_name = Name::from_string("Option.noConfusion");
    if let Some(option_nc_const) = env.get_const(&option_nc_name) {
        println!("\n=== Option.noConfusion ===");
        println!("  type: {:?}", option_nc_const.type_);
        match tc.infer_type(&option_nc_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                nc_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    }

    assert!(
        nc_successes >= 3,
        "Expected at least 3 NoConfusion principles to validate, got {nc_successes}"
    );
}

#[test]
fn test_beq_hashable_typeclasses() {
    // Validate BEq (boolean equality) and Hashable typeclasses
    // These are fundamental for data structures and comparisons
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Data.Hashable",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Data.Hashable with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.Hashable: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut tc_successes = 0;

    // Test BEq typeclass
    // class BEq (α : Type u) where
    //   beq : α → α → Bool
    let beq_name = Name::from_string("BEq");
    if let Some(beq_ind) = env.get_inductive(&beq_name) {
        println!("\n=== BEq Typeclass (Inductive) ===");
        println!("  num_params: {}", beq_ind.num_params);
        println!("  constructors: {:?}", beq_ind.constructor_names);
        tc_successes += 1;
    } else if let Some(beq_const) = env.get_const(&beq_name) {
        println!("\n=== BEq Typeclass (Constant) ===");
        println!("  type: {:?}", beq_const.type_);
        match tc.infer_type(&beq_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                tc_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    }

    // Test BEq.beq method
    let beq_beq_name = Name::from_string("BEq.beq");
    if let Some(beq_beq_const) = env.get_const(&beq_beq_name) {
        println!("\n=== BEq.beq method ===");
        println!("  type: {:?}", beq_beq_const.type_);
        match tc.infer_type(&beq_beq_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                tc_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    }

    // Test instBEqNat instance
    let inst_beq_nat_name = Name::from_string("instBEqNat");
    if let Some(inst_const) = env.get_const(&inst_beq_nat_name) {
        println!("\n=== instBEqNat instance ===");
        println!("  type: {:?}", inst_const.type_);
        match tc.infer_type(&inst_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                tc_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    }

    // Test Hashable typeclass
    // class Hashable (α : Type u) where
    //   hash : α → UInt64
    let hashable_name = Name::from_string("Hashable");
    if let Some(hashable_ind) = env.get_inductive(&hashable_name) {
        println!("\n=== Hashable Typeclass (Inductive) ===");
        println!("  num_params: {}", hashable_ind.num_params);
        println!("  constructors: {:?}", hashable_ind.constructor_names);
        tc_successes += 1;
    } else if let Some(hashable_const) = env.get_const(&hashable_name) {
        println!("\n=== Hashable Typeclass (Constant) ===");
        println!("  type: {:?}", hashable_const.type_);
        match tc.infer_type(&hashable_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                tc_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    }

    // Test Hashable.hash method
    let hashable_hash_name = Name::from_string("Hashable.hash");
    if let Some(hash_const) = env.get_const(&hashable_hash_name) {
        println!("\n=== Hashable.hash method ===");
        println!("  type: {:?}", hash_const.type_);
        match tc.infer_type(&hash_const.type_) {
            Ok(sort) => {
                println!("  sort: {sort:?}");
                tc_successes += 1;
            }
            Err(e) => println!("  type error: {e:?}"),
        }
    }

    assert!(
        tc_successes >= 3,
        "Expected at least 3 BEq/Hashable types to validate, got {tc_successes}"
    );
}

#[test]
fn test_inductive_recursor_types() {
    // Validate recursor types for core inductives
    // Recursors are the primitive elimination principles
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.Core", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Core with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Core: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut rec_successes = 0;

    // Test Nat.rec
    // Nat.rec : {motive : Nat → Sort u} → motive 0 → ((n : Nat) → motive n → motive (n + 1)) → (t : Nat) → motive t
    for rec_name_str in [
        "Nat.rec",
        "Nat.recOn",
        "Nat.casesOn",
        "Bool.rec",
        "Bool.casesOn",
        "List.rec",
        "List.recOn",
        "Option.rec",
        "Option.casesOn",
    ] {
        let rec_name = Name::from_string(rec_name_str);
        if let Some(rec_const) = env.get_const(&rec_name) {
            println!("\n=== {rec_name_str} ===");
            println!("  type: {:?}", rec_const.type_);
            match tc.infer_type(&rec_const.type_) {
                Ok(sort) => {
                    println!("  sort: {sort:?}");
                    rec_successes += 1;
                }
                Err(e) => println!("  type error: {e:?}"),
            }
        } else {
            println!("  {rec_name_str} not found");
        }
    }

    assert!(
        rec_successes >= 6,
        "Expected at least 6 recursor types to validate, got {rec_successes}"
    );
}

#[test]
fn test_conditional_dite_and_ite_control_flow() {
    // Validate conditional combinators: ite, dite, byCases, iteInduction
    // Ensures dependent and non-dependent conditionals import with well-typed values
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let mut total_added = 0usize;

    let summaries_core =
        load_module_with_deps(&mut env, "Init.Core", std::slice::from_ref(&lib_path))
            .expect("Failed to load Init.Core with dependencies");
    total_added += summaries_core
        .iter()
        .map(|s| s.added_constants)
        .sum::<usize>();

    let summaries_prelude =
        load_module_with_deps(&mut env, "Init.Prelude", std::slice::from_ref(&lib_path))
            .expect("Failed to load Init.Prelude with dependencies");
    total_added += summaries_prelude
        .iter()
        .map(|s| s.added_constants)
        .sum::<usize>();

    let total_modules = summaries_core.len() + summaries_prelude.len();
    println!("Init.Core/Prelude: {total_modules} modules, {total_added} constants added");

    let mut type_successes = 0;
    let mut value_successes = 0;

    for name_str in [
        "ite",
        "dite",
        "Decidable.byCases",
        "iteInduction",
        "if_pos",
        "if_neg",
        "dif_pos",
        "dif_neg",
    ] {
        let name = Name::from_string(name_str);
        if let Some(const_info) = env.get_const(&name) {
            println!("\n=== {name_str} ===");
            println!("  type: {:?}", const_info.type_);
            let type_check = TypeChecker::new(&env).infer_type(&const_info.type_);
            if type_check.is_ok() {
                type_successes += 1;
            } else {
                println!("  type error: {type_check:?}");
            }

            if let Some(value) = &const_info.value {
                match TypeChecker::new(&env).infer_type(value) {
                    Ok(inferred) => {
                        let def_eq = TypeChecker::new(&env).is_def_eq(&inferred, &const_info.type_);
                        if def_eq {
                            value_successes += 1;
                        } else {
                            println!(
                                "  value type mismatch: inferred {:?}, declared {:?}",
                                inferred, const_info.type_
                            );
                        }
                    }
                    Err(e) => println!("  value type error: {e:?}"),
                }
            } else {
                println!("  no value for {name_str}");
            }
        } else {
            println!("  {name_str} not found");
        }
    }

    assert!(
        type_successes >= 4,
        "Expected at least 4 conditional combinators to type-check, got {type_successes}"
    );
    assert!(
        value_successes >= 3,
        "Expected at least 3 conditional definitions to have well-typed values, got {value_successes}"
    );
}

#[test]
fn test_lawful_functor_and_monad_laws() {
    // Validate LawfulFunctor/LawfulMonad infrastructure and exported law lemmas
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Control.Lawful",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Control.Lawful with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Control.Lawful: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut typeclass_successes = 0;

    for class_name in ["LawfulFunctor", "LawfulApplicative", "LawfulMonad"] {
        let name = Name::from_string(class_name);
        if let Some(class_ind) = env.get_inductive(&name) {
            println!("\n=== {class_name} ===");
            println!("  num_params: {}", class_ind.num_params);
            println!("  constructors: {:?}", class_ind.constructor_names);
            typeclass_successes += 1;
        } else if let Some(class_const) = env.get_const(&name) {
            println!("\n=== {class_name} definition ===");
            println!("  type: {:?}", class_const.type_);
            if tc.infer_type(&class_const.type_).is_ok() {
                typeclass_successes += 1;
            }
        } else {
            println!("  {class_name} not found");
        }
    }

    let mut lemma_successes = 0;
    for lemma in [
        "monadLift_self",
        "map_eq_pure_bind",
        "seq_eq_bind_map",
        "seqRight_eq_bind",
        "seqLeft_eq_bind",
        "map_bind",
        "bind_map_left",
        "bind_pure",
    ] {
        let name = Name::from_string(lemma);
        if let Some(const_info) = env.get_const(&name) {
            println!("\n{} : {:?}", lemma, const_info.type_);
            if tc.infer_type(&const_info.type_).is_ok() {
                lemma_successes += 1;
            }
        } else {
            println!("  {lemma} not found");
        }
    }

    let mut instance_successes = 0;
    for inst_name in ["instLawfulMonadOption", "instLawfulMonadExcept"] {
        let name = Name::from_string(inst_name);
        if let Some(inst_const) = env.get_const(&name) {
            println!("\n{} type: {:?}", inst_name, inst_const.type_);
            if tc.infer_type(&inst_const.type_).is_ok() {
                instance_successes += 1;
            }
        } else {
            println!("  {inst_name} not found");
        }
    }

    assert!(
        typeclass_successes >= 2,
        "Expected at least 2 lawful typeclasses to be present, got {typeclass_successes}"
    );
    assert!(
        lemma_successes >= 5,
        "Expected at least 5 lawful functor/monad lemmas to type-check, got {lemma_successes}"
    );
    assert!(
        instance_successes >= 1,
        "Expected at least 1 lawful monad instance to type-check, got {instance_successes}"
    );
}

#[test]
fn test_omega_linear_arithmetic_types() {
    // Validate Omega decision procedure types (linear integer arithmetic)
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.Omega", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Omega with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Omega: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Core Omega types - these exist as private internal types, we check the module loads
    // and type-checks Int/Nat lemmas that omega uses
    for type_name in [
        "Int.Linear.Poly",
        "Int.Linear.Expr",
        "Int.Linear.ExprCnstr",
        "Nat.Linear.Poly",
        "Nat.Linear.Expr",
        "Nat.Linear.ExprCnstr",
    ] {
        let name = Name::from_string(type_name);
        if let Some(ind) = env.get_inductive(&name) {
            println!("\n=== {type_name} (inductive) ===");
            println!("  num_params: {}", ind.num_params);
            type_successes += 1;
        } else if let Some(const_info) = env.get_const(&name) {
            println!("\n=== {type_name} (def) ===");
            if tc.infer_type(&const_info.type_).is_ok() {
                type_successes += 1;
            }
        } else {
            println!("  {type_name} not found");
        }
    }

    // Omega validation functions and lemmas
    let mut func_successes = 0;
    for func_name in [
        "Int.Linear.Poly.denote",
        "Int.Linear.Expr.denote",
        "Nat.Linear.Poly.denote",
        "Nat.Linear.Expr.denote",
        "Int.ofNat_sub_ofNat",
        "Int.ofNat_add_negSucc",
    ] {
        let name = Name::from_string(func_name);
        if let Some(const_info) = env.get_const(&name) {
            println!("\n{func_name} : ...");
            if tc.infer_type(&const_info.type_).is_ok() {
                func_successes += 1;
            }
        } else {
            println!("  {func_name} not found");
        }
    }

    // Omega-relevant arithmetic lemmas
    let mut lemma_successes = 0;
    for lemma_name in [
        "Int.sub_add_cancel",
        "Int.add_sub_cancel",
        "Int.neg_neg",
        "Nat.sub_add_cancel",
        "Nat.add_sub_cancel",
    ] {
        let name = Name::from_string(lemma_name);
        if let Some(const_info) = env.get_const(&name) {
            println!("\n{lemma_name} : ...");
            if tc.infer_type(&const_info.type_).is_ok() {
                lemma_successes += 1;
            }
        } else {
            println!("  {lemma_name} not found");
        }
    }

    assert!(
        type_successes >= 2,
        "Expected at least 2 Omega/Linear types to load, got {type_successes}"
    );
    assert!(
        func_successes + lemma_successes >= 4,
        "Expected at least 4 Omega-related functions/lemmas to type-check, got {}",
        func_successes + lemma_successes
    );
}

#[test]
fn test_grind_automation_tactic_types() {
    // Validate Grind automation tactic infrastructure - this tests that
    // Init.Grind module loads successfully and provides normalization lemmas
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Init.Grind", std::slice::from_ref(&lib_path))
        .expect("Failed to load Init.Grind with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Grind: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Core types used by Grind - congruence closure and normalization
    // These may be internal but related datatypes should be available
    for type_name in ["Decidable", "DecidableEq", "BEq", "Hashable"] {
        let name = Name::from_string(type_name);
        if let Some(ind) = env.get_inductive(&name) {
            println!("\n=== {type_name} (inductive) ===");
            println!("  num_params: {}", ind.num_params);
            type_successes += 1;
        } else if let Some(const_info) = env.get_const(&name) {
            println!("\n=== {type_name} (def) ===");
            if tc.infer_type(&const_info.type_).is_ok() {
                type_successes += 1;
            }
        } else {
            println!("  {type_name} not found");
        }
    }

    // Grind-relevant normalization lemmas (these come from Init.Core and PropLemmas)
    let mut lemma_successes = 0;
    for lemma_name in [
        "ne_eq",
        "not_not",
        "and_self",
        "or_self",
        "Bool.not_not",
        "Bool.and_self",
        "Bool.or_self",
        "beq_self_eq_true",
    ] {
        let name = Name::from_string(lemma_name);
        if let Some(const_info) = env.get_const(&name) {
            println!("\n{lemma_name} : ...");
            if tc.infer_type(&const_info.type_).is_ok() {
                lemma_successes += 1;
            }
        } else {
            println!("  {lemma_name} not found");
        }
    }

    // Grind uses equality reasoning lemmas
    let mut eq_successes = 0;
    for eq_name in ["Eq.subst", "Eq.symm", "Eq.trans", "congrArg", "congrFun"] {
        let name = Name::from_string(eq_name);
        if let Some(const_info) = env.get_const(&name) {
            println!("\n{eq_name} : ...");
            if tc.infer_type(&const_info.type_).is_ok() {
                eq_successes += 1;
            }
        } else {
            println!("  {eq_name} not found");
        }
    }

    // Verify that the module loaded and we have the core infrastructure
    assert!(
        type_successes >= 3,
        "Expected at least 3 Grind-related types to load, got {type_successes}"
    );
    assert!(
        lemma_successes >= 4,
        "Expected at least 4 normalization lemmas to type-check, got {lemma_successes}"
    );
    assert!(
        eq_successes >= 4,
        "Expected at least 4 equality lemmas to type-check, got {eq_successes}"
    );
}

#[test]
fn test_std_sat_solver_types() {
    // Validate Std.Sat SAT solver types (CNF, AIG)
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();

    // Load Std.Sat
    let summaries = load_module_with_deps(&mut env, "Std.Sat", std::slice::from_ref(&lib_path))
        .expect("Failed to load Std.Sat with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Std.Sat: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // SAT core types - CNF
    for type_name in [
        "Std.Sat.CNF",
        "Std.Sat.CNF.Clause",
        "Std.Sat.CNF.Literal",
        "Std.Sat.CNF.DefaultClause",
    ] {
        let name = Name::from_string(type_name);
        if let Some(ind) = env.get_inductive(&name) {
            println!("\n=== {type_name} (inductive) ===");
            println!("  num_params: {}", ind.num_params);
            println!("  constructors: {:?}", ind.constructor_names);
            type_successes += 1;
        } else if let Some(const_info) = env.get_const(&name) {
            println!("\n=== {type_name} (def) ===");
            println!("  type: {:?}", const_info.type_);
            if tc.infer_type(&const_info.type_).is_ok() {
                type_successes += 1;
            }
        } else {
            println!("  {type_name} not found");
        }
    }

    // SAT core types - AIG (And-Inverter Graphs)
    for type_name in ["Std.Sat.AIG", "Std.Sat.AIG.Decl", "Std.Sat.AIG.RefVec"] {
        let name = Name::from_string(type_name);
        if let Some(ind) = env.get_inductive(&name) {
            println!("\n=== {type_name} (inductive) ===");
            println!("  num_params: {}", ind.num_params);
            println!("  constructors: {:?}", ind.constructor_names);
            type_successes += 1;
        } else if let Some(const_info) = env.get_const(&name) {
            println!("\n=== {type_name} (def) ===");
            println!("  type: {:?}", const_info.type_);
            if tc.infer_type(&const_info.type_).is_ok() {
                type_successes += 1;
            }
        } else {
            println!("  {type_name} not found");
        }
    }

    // SAT operations
    let mut func_successes = 0;
    for func_name in [
        "Std.Sat.CNF.eval",
        "Std.Sat.CNF.sat",
        "Std.Sat.CNF.unsat",
        "Std.Sat.AIG.mkAtom",
        "Std.Sat.AIG.mkGate",
        "Std.Sat.AIG.denote",
    ] {
        let name = Name::from_string(func_name);
        if let Some(const_info) = env.get_const(&name) {
            println!("\n{func_name} : ...");
            if tc.infer_type(&const_info.type_).is_ok() {
                func_successes += 1;
            }
        } else {
            println!("  {func_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 SAT types to load, got {type_successes}"
    );
    assert!(
        func_successes >= 2,
        "Expected at least 2 SAT functions to type-check, got {func_successes}"
    );
}

#[test]
fn test_proplemmas_propositional_reasoning() {
    // Validate Init.PropLemmas for propositional reasoning support
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Init.PropLemmas", std::slice::from_ref(&lib_path))
            .expect("Failed to load Init.PropLemmas with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.PropLemmas: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut lemma_successes = 0;

    // Core propositional lemmas
    for lemma_name in [
        "true_and",
        "and_true",
        "false_and",
        "and_false",
        "true_or",
        "or_true",
        "false_or",
        "or_false",
        "and_self",
        "or_self",
        "not_not_intro",
        "And.comm",
        "Or.comm",
        "And.assoc",
        "Or.assoc",
        "and_or_left",
        "or_and_left",
    ] {
        let name = Name::from_string(lemma_name);
        if let Some(const_info) = env.get_const(&name) {
            println!("\n{lemma_name} : ...");
            if tc.infer_type(&const_info.type_).is_ok() {
                lemma_successes += 1;
            } else {
                println!("  type error for {lemma_name}");
            }
        } else {
            println!("  {lemma_name} not found");
        }
    }

    // Implication and negation lemmas
    let mut impl_successes = 0;
    for lemma_name in [
        "imp_iff_not_or",
        "not_and",
        "not_or",
        "Decidable.not_not",
        "Decidable.em",
    ] {
        let name = Name::from_string(lemma_name);
        if let Some(const_info) = env.get_const(&name) {
            println!("\n{lemma_name} : ...");
            if tc.infer_type(&const_info.type_).is_ok() {
                impl_successes += 1;
            }
        } else {
            println!("  {lemma_name} not found");
        }
    }

    assert!(
        lemma_successes >= 10,
        "Expected at least 10 propositional lemmas to type-check, got {lemma_successes}"
    );
    assert!(
        impl_successes >= 2,
        "Expected at least 2 implication/negation lemmas to type-check, got {impl_successes}"
    );
}

#[test]
fn test_simplemmas_simplification_lemmas() {
    // Validate Init.SimpLemmas for simp tactic support
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Init.SimpLemmas", std::slice::from_ref(&lib_path))
            .expect("Failed to load Init.SimpLemmas with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.SimpLemmas: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut lemma_successes = 0;

    // Core simp lemmas for standard types
    for lemma_name in [
        "Bool.not_not",
        "Bool.true_eq",
        "Bool.false_eq",
        "Bool.and_true",
        "Bool.and_false",
        "Bool.or_true",
        "Bool.or_false",
        "Option.some_ne_none",
        "List.nil_append",
        "List.append_nil",
        "Nat.add_zero",
        "Nat.zero_add",
    ] {
        let name = Name::from_string(lemma_name);
        if let Some(const_info) = env.get_const(&name) {
            println!("\n{lemma_name} : ...");
            if tc.infer_type(&const_info.type_).is_ok() {
                lemma_successes += 1;
            } else {
                println!("  type error for {lemma_name}");
            }
        } else {
            println!("  {lemma_name} not found");
        }
    }

    // Simp infrastructure - eq lemmas and simp-specific attributes
    let mut eq_successes = 0;
    for lemma_name in [
        "eq_self",
        "ne_eq",
        "beq_eq_true",
        "beq_eq_false",
        "decide_eq_true_eq",
        "decide_eq_false_eq",
    ] {
        let name = Name::from_string(lemma_name);
        if let Some(const_info) = env.get_const(&name) {
            println!("\n{lemma_name} : ...");
            if tc.infer_type(&const_info.type_).is_ok() {
                eq_successes += 1;
            }
        } else {
            println!("  {lemma_name} not found");
        }
    }

    assert!(
        lemma_successes >= 6,
        "Expected at least 6 simp lemmas to type-check, got {lemma_successes}"
    );
    assert!(
        eq_successes >= 3,
        "Expected at least 3 eq/decide simp lemmas to type-check, got {eq_successes}"
    );
}
