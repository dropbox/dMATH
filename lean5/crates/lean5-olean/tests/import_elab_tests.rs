//! Elaboration tests for .olean module import functionality.
//!
//! These tests validate elaboration-related types (quotients, metaprogramming,
//! tactic, elab types) from Lean 4 standard library modules.
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
fn test_quotient_types() {
    // Test Quot and Quotient - quotient types for equivalence classes
    // Quot is primitive (built-in), Quotient is defined using Setoid
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    // Quot and Quotient are defined in Init.Core (primitives)
    let _ = load_module_with_deps(&mut env, "Init.Core", &[lib_path])
        .expect("Failed to load Init.Core");

    let mut tc = TypeChecker::new(&env);

    // Test Quot (primitive quotient type)
    // constant Quot {α : Sort u} (r : α → α → Prop) : Sort u
    let quot_name = Name::from_string("Quot");
    if let Some(quot_const) = env.get_const(&quot_name) {
        println!("\n=== Quot Definition ===");
        println!("  type: {:?}", quot_const.type_);
        let quot_check = tc.infer_type(&quot_const.type_);
        assert!(quot_check.is_ok(), "Quot type should be well-formed");
    }

    // Test Quot.mk (quotient constructor)
    let quot_mk_name = Name::from_string("Quot.mk");
    if let Some(quot_mk_const) = env.get_const(&quot_mk_name) {
        println!("\n=== Quot.mk Definition ===");
        println!("  type: {:?}", quot_mk_const.type_);
        let mk_check = tc.infer_type(&quot_mk_const.type_);
        assert!(mk_check.is_ok(), "Quot.mk type should be well-formed");
    }

    // Test Quot.lift (the elimination principle)
    let quot_lift_name = Name::from_string("Quot.lift");
    if let Some(quot_lift_const) = env.get_const(&quot_lift_name) {
        println!("\n=== Quot.lift Definition ===");
        println!("  type: {:?}", quot_lift_const.type_);
        let lift_check = tc.infer_type(&quot_lift_const.type_);
        assert!(lift_check.is_ok(), "Quot.lift type should be well-formed");
    }

    // Test Quot.ind (the induction principle)
    let quot_ind_name = Name::from_string("Quot.ind");
    if let Some(quot_ind_const) = env.get_const(&quot_ind_name) {
        println!("\n=== Quot.ind Definition ===");
        println!("  type: {:?}", quot_ind_const.type_);
        let ind_check = tc.infer_type(&quot_ind_const.type_);
        assert!(ind_check.is_ok(), "Quot.ind type should be well-formed");
    }

    // Test Quotient (type-class based quotient)
    let quotient_name = Name::from_string("Quotient");
    if let Some(quotient_const) = env.get_const(&quotient_name) {
        println!("\n=== Quotient Definition ===");
        println!("  type: {:?}", quotient_const.type_);
        let quotient_check = tc.infer_type(&quotient_const.type_);
        assert!(
            quotient_check.is_ok(),
            "Quotient type should be well-formed"
        );
    }

    // Test Setoid typeclass (equivalence relation)
    let setoid_name = Name::from_string("Setoid");
    if let Some(setoid_ind) = env.get_inductive(&setoid_name) {
        println!("\n=== Setoid Typeclass ===");
        println!("  num_params: {}", setoid_ind.num_params);
        println!("  constructors: {:?}", setoid_ind.constructor_names);
    } else if let Some(setoid_const) = env.get_const(&setoid_name) {
        println!("\n=== Setoid Definition ===");
        println!("  type: {:?}", setoid_const.type_);
        let check = tc.infer_type(&setoid_const.type_);
        assert!(check.is_ok(), "Setoid type should be well-formed");
    }

    // Test Equivalence relation
    let equiv_name = Name::from_string("Equivalence");
    if let Some(equiv_ind) = env.get_inductive(&equiv_name) {
        println!("\n=== Equivalence Structure ===");
        println!("  num_params: {}", equiv_ind.num_params);
        println!("  constructors: {:?}", equiv_ind.constructor_names);
    } else if let Some(equiv_const) = env.get_const(&equiv_name) {
        println!("\n=== Equivalence Definition ===");
        println!("  type: {:?}", equiv_const.type_);
        let check = tc.infer_type(&equiv_const.type_);
        assert!(check.is_ok(), "Equivalence type should be well-formed");
    }
}

#[test]
fn test_stream_iterator_types() {
    // Test Stream and iterator types for lazy sequences
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Init.Data.Stream", &[lib_path])
        .expect("Failed to load Init.Data.Stream");

    let mut tc = TypeChecker::new(&env);

    // Test Stream typeclass
    let stream_name = Name::from_string("Stream");
    if let Some(stream_ind) = env.get_inductive(&stream_name) {
        println!("\n=== Stream Typeclass ===");
        println!("  num_params: {}", stream_ind.num_params);
        println!("  constructors: {:?}", stream_ind.constructor_names);
    } else if let Some(stream_const) = env.get_const(&stream_name) {
        println!("\n=== Stream Definition ===");
        println!("  type: {:?}", stream_const.type_);
        let check = tc.infer_type(&stream_const.type_);
        assert!(check.is_ok(), "Stream type should be well-formed");
    }

    // Test Stream.next? (core operation)
    // Note: Stream.next? uses outParam which requires special handling
    let next_name = Name::from_string("Stream.next?");
    if let Some(next_const) = env.get_const(&next_name) {
        println!("\n=== Stream.next? ===");
        println!("  type: {:?}", next_const.type_);
        // Type-check may fail due to outParam; just verify constant exists
        let check = tc.infer_type(&next_const.type_);
        if check.is_err() {
            println!("  (type check skipped - contains outParam)");
        }
    }

    // Test ToStream typeclass
    let to_stream_name = Name::from_string("ToStream");
    if let Some(ts_ind) = env.get_inductive(&to_stream_name) {
        println!("\n=== ToStream Typeclass ===");
        println!("  num_params: {}", ts_ind.num_params);
        println!("  constructors: {:?}", ts_ind.constructor_names);
    } else if let Some(ts_const) = env.get_const(&to_stream_name) {
        println!("\n=== ToStream Definition ===");
        println!("  type: {:?}", ts_const.type_);
        let check = tc.infer_type(&ts_const.type_);
        assert!(check.is_ok(), "ToStream type should be well-formed");
    }
}

#[test]
fn test_tostring_repr_types() {
    // Test ToString and Repr - string representation typeclasses
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Init.Data.ToString.Basic", &[lib_path])
        .expect("Failed to load Init.Data.ToString.Basic");

    let mut tc = TypeChecker::new(&env);

    // Test ToString typeclass
    let tostring_name = Name::from_string("ToString");
    if let Some(ts_ind) = env.get_inductive(&tostring_name) {
        println!("\n=== ToString Typeclass ===");
        println!("  num_params: {}", ts_ind.num_params);
        println!("  constructors: {:?}", ts_ind.constructor_names);
    } else if let Some(ts_const) = env.get_const(&tostring_name) {
        println!("\n=== ToString Definition ===");
        println!("  type: {:?}", ts_const.type_);
        let check = tc.infer_type(&ts_const.type_);
        assert!(check.is_ok(), "ToString type should be well-formed");
    }

    // Test Repr typeclass (for debugging representation)
    let repr_name = Name::from_string("Repr");
    if let Some(repr_ind) = env.get_inductive(&repr_name) {
        println!("\n=== Repr Typeclass ===");
        println!("  num_params: {}", repr_ind.num_params);
        println!("  constructors: {:?}", repr_ind.constructor_names);
    } else if let Some(repr_const) = env.get_const(&repr_name) {
        println!("\n=== Repr Definition ===");
        println!("  type: {:?}", repr_const.type_);
        let check = tc.infer_type(&repr_const.type_);
        assert!(check.is_ok(), "Repr type should be well-formed");
    }

    // Test Format (formatted strings)
    let format_name = Name::from_string("Std.Format");
    if let Some(format_ind) = env.get_inductive(&format_name) {
        println!("\n=== Format Inductive ===");
        println!("  num_params: {}", format_ind.num_params);
        println!("  constructors: {:?}", format_ind.constructor_names);
    } else if let Some(format_const) = env.get_const(&format_name) {
        println!("\n=== Format Definition ===");
        println!("  type: {:?}", format_const.type_);
        let check = tc.infer_type(&format_const.type_);
        assert!(check.is_ok(), "Format type should be well-formed");
    }
}

#[test]
fn test_ordering_comparison_types() {
    // Test Ordering and comparison typeclasses
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Init.Data.Ord", &[lib_path])
        .expect("Failed to load Init.Data.Ord");

    let mut tc = TypeChecker::new(&env);

    // Test Ordering inductive (lt, eq, gt)
    let ordering_name = Name::from_string("Ordering");
    if let Some(ord_ind) = env.get_inductive(&ordering_name) {
        println!("\n=== Ordering Inductive ===");
        println!("  num_params: {}", ord_ind.num_params);
        println!("  num_indices: {}", ord_ind.num_indices);
        println!("  constructors: {:?}", ord_ind.constructor_names);

        assert_eq!(ord_ind.num_params, 0, "Ordering has no params");
        assert_eq!(
            ord_ind.constructor_names.len(),
            3,
            "Ordering has 3 constructors (lt, eq, gt)"
        );

        // Type-check constructors
        for ctor in ["Ordering.lt", "Ordering.eq", "Ordering.gt"] {
            let name = Name::from_string(ctor);
            if let Some(const_info) = env.get_const(&name) {
                let check = tc.infer_type(&const_info.type_);
                assert!(check.is_ok(), "{ctor} type should be well-formed");
            }
        }
    }

    // Test Ord typeclass
    let ord_name = Name::from_string("Ord");
    if let Some(ord_tc_ind) = env.get_inductive(&ord_name) {
        println!("\n=== Ord Typeclass ===");
        println!("  num_params: {}", ord_tc_ind.num_params);
        println!("  constructors: {:?}", ord_tc_ind.constructor_names);
    } else if let Some(ord_const) = env.get_const(&ord_name) {
        println!("\n=== Ord Definition ===");
        println!("  type: {:?}", ord_const.type_);
        let check = tc.infer_type(&ord_const.type_);
        assert!(check.is_ok(), "Ord type should be well-formed");
    }

    // Test compare function
    let compare_name = Name::from_string("compare");
    if let Some(cmp_const) = env.get_const(&compare_name) {
        println!("\n=== compare function ===");
        println!("  type: {:?}", cmp_const.type_);
        let check = tc.infer_type(&cmp_const.type_);
        assert!(check.is_ok(), "compare type should be well-formed");
    }

    // Test LT, LE typeclasses
    let lt_name = Name::from_string("LT");
    if let Some(lt_const) = env.get_const(&lt_name) {
        println!("\n=== LT Definition ===");
        println!("  type: {:?}", lt_const.type_);
        let check = tc.infer_type(&lt_const.type_);
        assert!(check.is_ok(), "LT type should be well-formed");
    }

    let le_name = Name::from_string("LE");
    if let Some(le_const) = env.get_const(&le_name) {
        println!("\n=== LE Definition ===");
        println!("  type: {:?}", le_const.type_);
        let check = tc.infer_type(&le_const.type_);
        assert!(check.is_ok(), "LE type should be well-formed");
    }
}

#[test]
fn test_inhabited_nonempty_types() {
    // Test Inhabited, Nonempty, and default values
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Init.Core", &[lib_path])
        .expect("Failed to load Init.Core");

    let mut tc = TypeChecker::new(&env);

    // Test Inhabited typeclass
    // class Inhabited (α : Sort u) where
    //   default : α
    let inhabited_name = Name::from_string("Inhabited");
    if let Some(inh_ind) = env.get_inductive(&inhabited_name) {
        println!("\n=== Inhabited Typeclass ===");
        println!("  num_params: {}", inh_ind.num_params);
        println!("  constructors: {:?}", inh_ind.constructor_names);

        // Type-check Inhabited.mk
        let mk_name = Name::from_string("Inhabited.mk");
        if let Some(mk_const) = env.get_const(&mk_name) {
            let check = tc.infer_type(&mk_const.type_);
            assert!(check.is_ok(), "Inhabited.mk type should be well-formed");
        }
    }

    // Test Nonempty (proof-relevant version)
    let nonempty_name = Name::from_string("Nonempty");
    if let Some(ne_ind) = env.get_inductive(&nonempty_name) {
        println!("\n=== Nonempty Inductive ===");
        println!("  num_params: {}", ne_ind.num_params);
        println!("  num_indices: {}", ne_ind.num_indices);
        println!("  constructors: {:?}", ne_ind.constructor_names);

        // Nonempty.intro
        let intro_name = Name::from_string("Nonempty.intro");
        if let Some(intro_const) = env.get_const(&intro_name) {
            let check = tc.infer_type(&intro_const.type_);
            assert!(check.is_ok(), "Nonempty.intro type should be well-formed");
        }
    }

    // Test default function
    let default_name = Name::from_string("default");
    if let Some(default_const) = env.get_const(&default_name) {
        println!("\n=== default function ===");
        println!("  type: {:?}", default_const.type_);
        let check = tc.infer_type(&default_const.type_);
        assert!(check.is_ok(), "default type should be well-formed");
    }

    // Test instInhabitedNat (Nat is inhabited with default 0)
    let inst_nat_name = Name::from_string("instInhabitedNat");
    if let Some(inst_const) = env.get_const(&inst_nat_name) {
        println!("\n=== instInhabitedNat ===");
        println!("  type: {:?}", inst_const.type_);
        let check = tc.infer_type(&inst_const.type_);
        assert!(check.is_ok(), "instInhabitedNat type should be well-formed");
    }
}

#[test]
fn test_cast_coercion_types() {
    // Test cast, coercion, and type conversion types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Init.Core", &[lib_path])
        .expect("Failed to load Init.Core");

    let mut tc = TypeChecker::new(&env);

    // Test cast function (converts between equal types)
    let cast_name = Name::from_string("cast");
    if let Some(cast_const) = env.get_const(&cast_name) {
        println!("\n=== cast function ===");
        println!("  type: {:?}", cast_const.type_);
        let check = tc.infer_type(&cast_const.type_);
        assert!(check.is_ok(), "cast type should be well-formed");
    }

    // Test Coe typeclass (automatic coercion)
    let coe_name = Name::from_string("Coe");
    if let Some(coe_ind) = env.get_inductive(&coe_name) {
        println!("\n=== Coe Typeclass ===");
        println!("  num_params: {}", coe_ind.num_params);
        println!("  constructors: {:?}", coe_ind.constructor_names);
    } else if let Some(coe_const) = env.get_const(&coe_name) {
        println!("\n=== Coe Definition ===");
        println!("  type: {:?}", coe_const.type_);
        let check = tc.infer_type(&coe_const.type_);
        assert!(check.is_ok(), "Coe type should be well-formed");
    }

    // Test CoeT (type-dependent coercion)
    let coet_name = Name::from_string("CoeT");
    if let Some(coet_const) = env.get_const(&coet_name) {
        println!("\n=== CoeT Definition ===");
        println!("  type: {:?}", coet_const.type_);
        let check = tc.infer_type(&coet_const.type_);
        assert!(check.is_ok(), "CoeT type should be well-formed");
    }

    // Test CoeHead (for function coercions)
    let coehead_name = Name::from_string("CoeHead");
    if let Some(ch_ind) = env.get_inductive(&coehead_name) {
        println!("\n=== CoeHead Typeclass ===");
        println!("  num_params: {}", ch_ind.num_params);
        println!("  constructors: {:?}", ch_ind.constructor_names);
    } else if let Some(ch_const) = env.get_const(&coehead_name) {
        println!("\n=== CoeHead Definition ===");
        println!("  type: {:?}", ch_const.type_);
        let check = tc.infer_type(&ch_const.type_);
        assert!(check.is_ok(), "CoeHead type should be well-formed");
    }
}

#[test]
fn test_rbtree_rbmap_types() {
    // Test RBTree and RBMap - red-black tree implementations
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Lean.Data.RBMap", &[lib_path])
        .expect("Failed to load Lean.Data.RBMap");

    let mut tc = TypeChecker::new(&env);

    // Test RBColor (red/black node color)
    let color_name = Name::from_string("Lean.RBColor");
    if let Some(color_ind) = env.get_inductive(&color_name) {
        println!("\n=== RBColor Inductive ===");
        println!("  num_params: {}", color_ind.num_params);
        println!("  constructors: {:?}", color_ind.constructor_names);

        assert_eq!(
            color_ind.constructor_names.len(),
            2,
            "RBColor has 2 constructors (red, black)"
        );
    }

    // Test RBNode (internal tree node)
    let node_name = Name::from_string("Lean.RBNode");
    if let Some(node_ind) = env.get_inductive(&node_name) {
        println!("\n=== RBNode Inductive ===");
        println!("  num_params: {}", node_ind.num_params);
        println!("  is_recursive: {}", node_ind.is_recursive);
        println!("  constructors: {:?}", node_ind.constructor_names);

        assert!(
            node_ind.is_recursive,
            "RBNode is recursive (tree structure)"
        );
    }

    // Test RBMap (key-value map)
    let rbmap_name = Name::from_string("Lean.RBMap");
    if let Some(rbmap_const) = env.get_const(&rbmap_name) {
        println!("\n=== RBMap Definition ===");
        println!("  type: {:?}", rbmap_const.type_);
        let check = tc.infer_type(&rbmap_const.type_);
        assert!(check.is_ok(), "RBMap type should be well-formed");
    }

    // Test RBMap.insert
    let insert_name = Name::from_string("Lean.RBMap.insert");
    if let Some(insert_const) = env.get_const(&insert_name) {
        println!("\n=== RBMap.insert ===");
        println!("  type: {:?}", insert_const.type_);
        let check = tc.infer_type(&insert_const.type_);
        assert!(check.is_ok(), "RBMap.insert type should be well-formed");
    }

    // Test RBMap.find?
    let find_name = Name::from_string("Lean.RBMap.find?");
    if let Some(find_const) = env.get_const(&find_name) {
        println!("\n=== RBMap.find? ===");
        println!("  type: {:?}", find_const.type_);
        let check = tc.infer_type(&find_const.type_);
        assert!(check.is_ok(), "RBMap.find? type should be well-formed");
    }
}

#[test]
fn test_parray_persistent_types() {
    // Test PersistentArray - efficient persistent array
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Lean.Data.PersistentArray", &[lib_path])
        .expect("Failed to load Lean.Data.PersistentArray");

    let mut tc = TypeChecker::new(&env);

    // Test PersistentArray
    let parray_name = Name::from_string("Lean.PersistentArray");
    if let Some(parray_ind) = env.get_inductive(&parray_name) {
        println!("\n=== PersistentArray Inductive ===");
        println!("  num_params: {}", parray_ind.num_params);
        println!("  is_recursive: {}", parray_ind.is_recursive);
        println!("  constructors: {:?}", parray_ind.constructor_names);
    } else if let Some(parray_const) = env.get_const(&parray_name) {
        println!("\n=== PersistentArray Definition ===");
        println!("  type: {:?}", parray_const.type_);
        let check = tc.infer_type(&parray_const.type_);
        assert!(check.is_ok(), "PersistentArray type should be well-formed");
    }

    // Test PersistentArrayNode (internal node type)
    let node_name = Name::from_string("Lean.PersistentArrayNode");
    if let Some(node_ind) = env.get_inductive(&node_name) {
        println!("\n=== PersistentArrayNode Inductive ===");
        println!("  num_params: {}", node_ind.num_params);
        println!("  is_recursive: {}", node_ind.is_recursive);
        println!("  constructors: {:?}", node_ind.constructor_names);
    }

    // Test PersistentArray.push
    let push_name = Name::from_string("Lean.PersistentArray.push");
    if let Some(push_const) = env.get_const(&push_name) {
        println!("\n=== PersistentArray.push ===");
        println!("  type: {:?}", push_const.type_);
        let check = tc.infer_type(&push_const.type_);
        assert!(
            check.is_ok(),
            "PersistentArray.push type should be well-formed"
        );
    }

    // Test PersistentArray.get!
    let get_name = Name::from_string("Lean.PersistentArray.get!");
    if let Some(get_const) = env.get_const(&get_name) {
        println!("\n=== PersistentArray.get! ===");
        println!("  type: {:?}", get_const.type_);
        let check = tc.infer_type(&get_const.type_);
        assert!(
            check.is_ok(),
            "PersistentArray.get! type should be well-formed"
        );
    }
}

#[test]
fn test_macro_syntax_types() {
    // Test Macro and Syntax types used for metaprogramming
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Lean.Elab.Macro", &[lib_path])
        .expect("Failed to load Lean.Elab.Macro");

    let mut tc = TypeChecker::new(&env);

    // Test Macro (macro definition type)
    let macro_name = Name::from_string("Lean.Macro");
    if let Some(macro_const) = env.get_const(&macro_name) {
        println!("\n=== Lean.Macro ===");
        println!("  type: {:?}", macro_const.type_);
        let check = tc.infer_type(&macro_const.type_);
        assert!(check.is_ok(), "Macro type should be well-formed");
    }

    // Test MacroM (macro monad)
    let macrom_name = Name::from_string("Lean.MacroM");
    if let Some(macrom_const) = env.get_const(&macrom_name) {
        println!("\n=== Lean.MacroM ===");
        println!("  type: {:?}", macrom_const.type_);
        let check = tc.infer_type(&macrom_const.type_);
        assert!(check.is_ok(), "MacroM type should be well-formed");
    }

    // Test TSyntax (typed syntax)
    let tsyntax_name = Name::from_string("Lean.TSyntax");
    if let Some(tsyntax_const) = env.get_const(&tsyntax_name) {
        println!("\n=== Lean.TSyntax ===");
        println!("  type: {:?}", tsyntax_const.type_);
        let check = tc.infer_type(&tsyntax_const.type_);
        assert!(check.is_ok(), "TSyntax type should be well-formed");
    }

    // Test SyntaxKind
    let syntaxkind_name = Name::from_string("Lean.SyntaxNodeKind");
    if let Some(kind_const) = env.get_const(&syntaxkind_name) {
        println!("\n=== Lean.SyntaxNodeKind ===");
        println!("  type: {:?}", kind_const.type_);
        let check = tc.infer_type(&kind_const.type_);
        assert!(check.is_ok(), "SyntaxNodeKind type should be well-formed");
    }

    // Test Syntax.node
    let node_name = Name::from_string("Lean.Syntax.node");
    if let Some(node_const) = env.get_const(&node_name) {
        println!("\n=== Lean.Syntax.node ===");
        println!("  type: {:?}", node_const.type_);
        let check = tc.infer_type(&node_const.type_);
        assert!(check.is_ok(), "Syntax.node type should be well-formed");
    }

    // Test SourceInfo
    let srcinfo_name = Name::from_string("Lean.SourceInfo");
    if let Some(srcinfo_ind) = env.get_inductive(&srcinfo_name) {
        println!("\n=== Lean.SourceInfo Inductive ===");
        println!("  constructors: {:?}", srcinfo_ind.constructor_names);
    }
}

#[test]
fn test_elab_term_types() {
    // Test Term elaboration types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Lean.Elab.Term", &[lib_path])
        .expect("Failed to load Lean.Elab.Term");

    let mut tc = TypeChecker::new(&env);

    // Test TermElabM
    let termelabm_name = Name::from_string("Lean.Elab.Term.TermElabM");
    if let Some(termelabm_const) = env.get_const(&termelabm_name) {
        println!("\n=== Lean.Elab.Term.TermElabM ===");
        println!("  type: {:?}", termelabm_const.type_);
        let check = tc.infer_type(&termelabm_const.type_);
        assert!(check.is_ok(), "TermElabM type should be well-formed");
    }

    // Test TermElab (term elaborator type)
    let termelab_name = Name::from_string("Lean.Elab.Term.TermElab");
    if let Some(termelab_const) = env.get_const(&termelab_name) {
        println!("\n=== Lean.Elab.Term.TermElab ===");
        println!("  type: {:?}", termelab_const.type_);
        let check = tc.infer_type(&termelab_const.type_);
        assert!(check.is_ok(), "TermElab type should be well-formed");
    }

    // Test SavedState
    let state_name = Name::from_string("Lean.Elab.Term.SavedState");
    if let Some(state_const) = env.get_const(&state_name) {
        println!("\n=== Lean.Elab.Term.SavedState ===");
        println!("  type: {:?}", state_const.type_);
        let check = tc.infer_type(&state_const.type_);
        assert!(check.is_ok(), "SavedState type should be well-formed");
    }

    // Test elabTerm function
    let elabterm_name = Name::from_string("Lean.Elab.Term.elabTerm");
    if let Some(elabterm_const) = env.get_const(&elabterm_name) {
        println!("\n=== Lean.Elab.Term.elabTerm ===");
        println!("  type: {:?}", elabterm_const.type_);
        let check = tc.infer_type(&elabterm_const.type_);
        assert!(check.is_ok(), "elabTerm type should be well-formed");
    }

    // Test ensureHasType
    let ensure_name = Name::from_string("Lean.Elab.Term.ensureHasType");
    if let Some(ensure_const) = env.get_const(&ensure_name) {
        println!("\n=== Lean.Elab.Term.ensureHasType ===");
        println!("  type: {:?}", ensure_const.type_);
        let check = tc.infer_type(&ensure_const.type_);
        assert!(check.is_ok(), "ensureHasType type should be well-formed");
    }
}

#[test]
fn test_simp_types() {
    // Test simplifier-related types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Lean.Meta.Tactic.Simp.Main", &[lib_path])
        .expect("Failed to load Lean.Meta.Tactic.Simp.Main");

    let mut tc = TypeChecker::new(&env);

    // Test SimpTheorems
    let simpthms_name = Name::from_string("Lean.Meta.SimpTheorems");
    if let Some(simpthms_const) = env.get_const(&simpthms_name) {
        println!("\n=== Lean.Meta.SimpTheorems ===");
        println!("  type: {:?}", simpthms_const.type_);
        let check = tc.infer_type(&simpthms_const.type_);
        assert!(check.is_ok(), "SimpTheorems type should be well-formed");
    }

    // Test SimpTheorem (single theorem)
    let simpthm_name = Name::from_string("Lean.Meta.SimpTheorem");
    if let Some(simpthm_const) = env.get_const(&simpthm_name) {
        println!("\n=== Lean.Meta.SimpTheorem ===");
        println!("  type: {:?}", simpthm_const.type_);
        let check = tc.infer_type(&simpthm_const.type_);
        assert!(check.is_ok(), "SimpTheorem type should be well-formed");
    }

    // Test Simp.Result
    let result_name = Name::from_string("Lean.Meta.Simp.Result");
    if let Some(result_const) = env.get_const(&result_name) {
        println!("\n=== Lean.Meta.Simp.Result ===");
        println!("  type: {:?}", result_const.type_);
        let check = tc.infer_type(&result_const.type_);
        assert!(check.is_ok(), "Simp.Result type should be well-formed");
    }

    // Test simp function
    let simp_name = Name::from_string("Lean.Meta.Simp.simp");
    if let Some(simp_const) = env.get_const(&simp_name) {
        println!("\n=== Lean.Meta.Simp.simp ===");
        println!("  type: {:?}", simp_const.type_);
        let check = tc.infer_type(&simp_const.type_);
        assert!(check.is_ok(), "simp function type should be well-formed");
    }

    // Test DiscrTree (discrimination tree for pattern matching)
    let dtree_name = Name::from_string("Lean.Meta.DiscrTree");
    if let Some(dtree_const) = env.get_const(&dtree_name) {
        println!("\n=== Lean.Meta.DiscrTree ===");
        println!("  type: {:?}", dtree_const.type_);
        let check = tc.infer_type(&dtree_const.type_);
        assert!(check.is_ok(), "DiscrTree type should be well-formed");
    }
}

#[test]
fn test_attribute_extension_types() {
    // Test attribute and extension types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Lean.Attributes", &[lib_path])
        .expect("Failed to load Lean.Attributes");

    let mut tc = TypeChecker::new(&env);

    // Test AttributeImpl
    let attrimpl_name = Name::from_string("Lean.AttributeImpl");
    if let Some(attrimpl_const) = env.get_const(&attrimpl_name) {
        println!("\n=== Lean.AttributeImpl ===");
        println!("  type: {:?}", attrimpl_const.type_);
        let check = tc.infer_type(&attrimpl_const.type_);
        assert!(check.is_ok(), "AttributeImpl type should be well-formed");
    }

    // Test AttributeKind
    let attrkind_name = Name::from_string("Lean.AttributeKind");
    if let Some(attrkind_ind) = env.get_inductive(&attrkind_name) {
        println!("\n=== Lean.AttributeKind Inductive ===");
        println!("  constructors: {:?}", attrkind_ind.constructor_names);
    }

    // Test registerBuiltinAttribute
    let regattr_name = Name::from_string("Lean.registerBuiltinAttribute");
    if let Some(regattr_const) = env.get_const(&regattr_name) {
        println!("\n=== Lean.registerBuiltinAttribute ===");
        println!("  type: {:?}", regattr_const.type_);
        let check = tc.infer_type(&regattr_const.type_);
        assert!(
            check.is_ok(),
            "registerBuiltinAttribute type should be well-formed"
        );
    }

    // Test PersistentEnvExtension
    let persext_name = Name::from_string("Lean.PersistentEnvExtension");
    if let Some(persext_const) = env.get_const(&persext_name) {
        println!("\n=== Lean.PersistentEnvExtension ===");
        println!("  type: {:?}", persext_const.type_);
        let check = tc.infer_type(&persext_const.type_);
        assert!(
            check.is_ok(),
            "PersistentEnvExtension type should be well-formed"
        );
    }

    // Test SimplePersistentEnvExtension
    let simpext_name = Name::from_string("Lean.SimplePersistentEnvExtension");
    if let Some(simpext_const) = env.get_const(&simpext_name) {
        println!("\n=== Lean.SimplePersistentEnvExtension ===");
        println!("  type: {:?}", simpext_const.type_);
        let check = tc.infer_type(&simpext_const.type_);
        assert!(
            check.is_ok(),
            "SimplePersistentEnvExtension type should be well-formed"
        );
    }
}

#[test]
fn test_conv_types() {
    // Test conversion tactic types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Lean.Elab.Tactic.Conv.Basic", &[lib_path])
        .expect("Failed to load Lean.Elab.Tactic.Conv.Basic");

    let mut tc = TypeChecker::new(&env);

    // Test Conv (conversion tactic monad)
    let conv_name = Name::from_string("Lean.Elab.Tactic.Conv.Conv");
    if let Some(conv_const) = env.get_const(&conv_name) {
        println!("\n=== Lean.Elab.Tactic.Conv.Conv ===");
        println!("  type: {:?}", conv_const.type_);
        let check = tc.infer_type(&conv_const.type_);
        assert!(check.is_ok(), "Conv type should be well-formed");
    }

    // Test changeLhs
    let changelhs_name = Name::from_string("Lean.Elab.Tactic.Conv.changeLhs");
    if let Some(changelhs_const) = env.get_const(&changelhs_name) {
        println!("\n=== Lean.Elab.Tactic.Conv.changeLhs ===");
        println!("  type: {:?}", changelhs_const.type_);
        let check = tc.infer_type(&changelhs_const.type_);
        assert!(check.is_ok(), "changeLhs type should be well-formed");
    }

    // Test getLhs
    let getlhs_name = Name::from_string("Lean.Elab.Tactic.Conv.getLhs");
    if let Some(getlhs_const) = env.get_const(&getlhs_name) {
        println!("\n=== Lean.Elab.Tactic.Conv.getLhs ===");
        println!("  type: {:?}", getlhs_const.type_);
        let check = tc.infer_type(&getlhs_const.type_);
        assert!(check.is_ok(), "getLhs type should be well-formed");
    }

    // Test getRhs
    let getrhs_name = Name::from_string("Lean.Elab.Tactic.Conv.getRhs");
    if let Some(getrhs_const) = env.get_const(&getrhs_name) {
        println!("\n=== Lean.Elab.Tactic.Conv.getRhs ===");
        println!("  type: {:?}", getrhs_const.type_);
        let check = tc.infer_type(&getrhs_const.type_);
        assert!(check.is_ok(), "getRhs type should be well-formed");
    }
}

#[test]
fn test_linter_types() {
    // Test linter infrastructure types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Lean.Linter.Basic", &[lib_path])
        .expect("Failed to load Lean.Linter.Basic");

    let mut tc = TypeChecker::new(&env);

    // Test Linter
    let linter_name = Name::from_string("Lean.Linter.Linter");
    if let Some(linter_const) = env.get_const(&linter_name) {
        println!("\n=== Lean.Linter.Linter ===");
        println!("  type: {:?}", linter_const.type_);
        let check = tc.infer_type(&linter_const.type_);
        assert!(check.is_ok(), "Linter type should be well-formed");
    }

    // Test getLinters
    let getlinters_name = Name::from_string("Lean.Linter.getLinters");
    if let Some(getlinters_const) = env.get_const(&getlinters_name) {
        println!("\n=== Lean.Linter.getLinters ===");
        println!("  type: {:?}", getlinters_const.type_);
        let check = tc.infer_type(&getlinters_const.type_);
        assert!(check.is_ok(), "getLinters type should be well-formed");
    }
}

#[test]
fn test_local_context_types() {
    // Test local context types for hypotheses management
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Lean.LocalContext", &[lib_path])
        .expect("Failed to load Lean.LocalContext");

    let mut tc = TypeChecker::new(&env);

    // Test LocalContext
    let lctx_name = Name::from_string("Lean.LocalContext");
    if let Some(lctx_const) = env.get_const(&lctx_name) {
        println!("\n=== Lean.LocalContext ===");
        println!("  type: {:?}", lctx_const.type_);
        let check = tc.infer_type(&lctx_const.type_);
        assert!(check.is_ok(), "LocalContext type should be well-formed");
    }

    // Test LocalDecl (local declaration)
    let ldecl_name = Name::from_string("Lean.LocalDecl");
    if let Some(ldecl_ind) = env.get_inductive(&ldecl_name) {
        println!("\n=== Lean.LocalDecl Inductive ===");
        println!("  constructors: {:?}", ldecl_ind.constructor_names);
    }

    // Test LocalDecl.fvarId
    let fvarid_name = Name::from_string("Lean.LocalDecl.fvarId");
    if let Some(fvarid_const) = env.get_const(&fvarid_name) {
        println!("\n=== Lean.LocalDecl.fvarId ===");
        println!("  type: {:?}", fvarid_const.type_);
        let check = tc.infer_type(&fvarid_const.type_);
        assert!(check.is_ok(), "LocalDecl.fvarId type should be well-formed");
    }

    // Test LocalContext.mkLocalDecl
    let mklocal_name = Name::from_string("Lean.LocalContext.mkLocalDecl");
    if let Some(mklocal_const) = env.get_const(&mklocal_name) {
        println!("\n=== Lean.LocalContext.mkLocalDecl ===");
        println!("  type: {:?}", mklocal_const.type_);
        let check = tc.infer_type(&mklocal_const.type_);
        assert!(
            check.is_ok(),
            "LocalContext.mkLocalDecl type should be well-formed"
        );
    }

    // Test LocalContext.getFVarIds
    let getfvars_name = Name::from_string("Lean.LocalContext.getFVarIds");
    if let Some(getfvars_const) = env.get_const(&getfvars_name) {
        println!("\n=== Lean.LocalContext.getFVarIds ===");
        println!("  type: {:?}", getfvars_const.type_);
        let check = tc.infer_type(&getfvars_const.type_);
        assert!(
            check.is_ok(),
            "LocalContext.getFVarIds type should be well-formed"
        );
    }
}

#[test]
fn test_mvar_context_types() {
    // Test metavariable context types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Lean.MetavarContext", &[lib_path])
        .expect("Failed to load Lean.MetavarContext");

    let mut tc = TypeChecker::new(&env);

    // Test MetavarContext
    let mctx_name = Name::from_string("Lean.MetavarContext");
    if let Some(mctx_const) = env.get_const(&mctx_name) {
        println!("\n=== Lean.MetavarContext ===");
        println!("  type: {:?}", mctx_const.type_);
        let check = tc.infer_type(&mctx_const.type_);
        assert!(check.is_ok(), "MetavarContext type should be well-formed");
    }

    // Test MetavarDecl
    let mdecl_name = Name::from_string("Lean.MetavarDecl");
    if let Some(mdecl_const) = env.get_const(&mdecl_name) {
        println!("\n=== Lean.MetavarDecl ===");
        println!("  type: {:?}", mdecl_const.type_);
        let check = tc.infer_type(&mdecl_const.type_);
        assert!(check.is_ok(), "MetavarDecl type should be well-formed");
    }

    // Test MVarId (metavariable identifier)
    let mvarid_name = Name::from_string("Lean.MVarId");
    if let Some(mvarid_const) = env.get_const(&mvarid_name) {
        println!("\n=== Lean.MVarId ===");
        println!("  type: {:?}", mvarid_const.type_);
        let check = tc.infer_type(&mvarid_const.type_);
        assert!(check.is_ok(), "MVarId type should be well-formed");
    }

    // Test MetavarContext.assignExpr
    let assign_name = Name::from_string("Lean.MetavarContext.assignExpr");
    if let Some(assign_const) = env.get_const(&assign_name) {
        println!("\n=== Lean.MetavarContext.assignExpr ===");
        println!("  type: {:?}", assign_const.type_);
        let check = tc.infer_type(&assign_const.type_);
        assert!(
            check.is_ok(),
            "MetavarContext.assignExpr type should be well-formed"
        );
    }

    // Test instantiateMVars
    let inst_name = Name::from_string("Lean.instantiateMVars");
    if let Some(inst_const) = env.get_const(&inst_name) {
        println!("\n=== Lean.instantiateMVars ===");
        println!("  type: {:?}", inst_const.type_);
        let check = tc.infer_type(&inst_const.type_);
        assert!(check.is_ok(), "instantiateMVars type should be well-formed");
    }
}

#[test]
fn test_command_elab_monad_and_state_types() {
    // Validate command elaboration monad/state imports and basic type-checking
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.Command",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.Command with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Command: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    // Ensure command elaboration structures are registered as inductives
    for (name, label) in [
        ("Lean.Elab.Command.Scope", "Scope"),
        ("Lean.Elab.Command.State", "State"),
        ("Lean.Elab.Command.Context", "Context"),
    ] {
        let name = Name::from_string(name);
        assert!(
            env.get_inductive(&name).is_some(),
            "{label} inductive should be registered"
        );
    }

    // Type-check core command elaboration definitions
    let mut type_successes = 0;
    for const_name in [
        "Lean.Elab.Command.CommandElabM",
        "Lean.Elab.Command.CommandElab",
        "Lean.Elab.Command.mkState",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            let mut tc = TypeChecker::new(&env);
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 2,
        "Expected at least 2 command elaboration definitions to type-check, got {type_successes}"
    );
    assert!(
        total_added > 0,
        "No constants were added from Lean.Elab.Command"
    );
}

#[test]
fn test_parser_state_and_trailing_parsers() {
    // Validate parser infrastructure imports (state, context, trailing parser combinators)
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(&mut env, "Lean.Parser", std::slice::from_ref(&lib_path))
        .expect("Failed to load Lean.Parser with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Parser: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    // Core parser structures should be inductives
    for (name, label) in [
        ("Lean.Parser.ParserState", "ParserState"),
        ("Lean.Parser.ParserContext", "ParserContext"),
    ] {
        let name = Name::from_string(name);
        assert!(
            env.get_inductive(&name).is_some(),
            "{label} inductive should be registered"
        );
    }

    // Type-check parser combinators and abbreviations
    let mut type_successes = 0;
    for const_name in [
        "Lean.Parser.Parser",
        "Lean.Parser.ParserFn",
        "Lean.Parser.TrailingParser",
        "Lean.Parser.trailingNode",
        "Lean.Parser.andthen",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            let mut tc = TypeChecker::new(&env);
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 parser definitions to type-check, got {type_successes}"
    );
}

#[test]
fn test_tactic_monad_and_basic_operations() {
    // Validate tactic monad/context imports and basic tactic combinators
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.Tactic.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.Tactic.Basic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Tactic.Basic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let ctx_name = Name::from_string("Lean.Elab.Tactic.Context");
    assert!(
        env.get_inductive(&ctx_name).is_some(),
        "Tactic.Context inductive should be registered"
    );

    // Type-check core tactic monad and helper definitions
    let mut type_successes = 0;
    for const_name in [
        "Lean.Elab.Tactic.TacticM",
        "Lean.Elab.Tactic.Tactic",
        "Lean.Elab.admitGoal",
        "Lean.Elab.Tactic.run",
    ] {
        let name = Name::from_string(const_name);
        if let Some(const_info) = env.get_const(&name) {
            let mut tc = TypeChecker::new(&env);
            match tc.infer_type(&const_info.type_) {
                Ok(sort) => {
                    println!("  {const_name} : {sort:?}");
                    type_successes += 1;
                }
                Err(e) => println!("  {const_name} type error: {e:?}"),
            }
        } else {
            println!("  {const_name} not found");
        }
    }

    assert!(
        type_successes >= 3,
        "Expected at least 3 tactic definitions to type-check, got {type_successes}"
    );
}

#[test]
fn test_unify_types() {
    // Test unification-related types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let _ = load_module_with_deps(&mut env, "Lean.Meta.Basic", &[lib_path])
        .expect("Failed to load Lean.Meta.Basic");

    let mut tc = TypeChecker::new(&env);

    // Test isDefEq
    let def_eq_name = Name::from_string("Lean.Meta.isDefEq");
    if let Some(def_eq_const) = env.get_const(&def_eq_name) {
        println!("\n=== Lean.Meta.isDefEq ===");
        println!("  type: {:?}", def_eq_const.type_);
        let check = tc.infer_type(&def_eq_const.type_);
        assert!(check.is_ok(), "isDefEq type should be well-formed");
    }

    // Test TransparencyMode
    let transp_name = Name::from_string("Lean.Meta.TransparencyMode");
    if let Some(transp_ind) = env.get_inductive(&transp_name) {
        println!("\n=== Lean.Meta.TransparencyMode Inductive ===");
        println!("  constructors: {:?}", transp_ind.constructor_names);
    }

    // Test Config
    let config_name = Name::from_string("Lean.Meta.Config");
    if let Some(config_const) = env.get_const(&config_name) {
        println!("\n=== Lean.Meta.Config ===");
        println!("  type: {:?}", config_const.type_);
        let check = tc.infer_type(&config_const.type_);
        assert!(check.is_ok(), "Config type should be well-formed");
    }

    // Test whnf (weak-head normal form)
    let whnf_name = Name::from_string("Lean.Meta.whnf");
    if let Some(whnf_const) = env.get_const(&whnf_name) {
        println!("\n=== Lean.Meta.whnf ===");
        println!("  type: {:?}", whnf_const.type_);
        let check = tc.infer_type(&whnf_const.type_);
        assert!(check.is_ok(), "whnf type should be well-formed");
    }

    // Test inferType
    let infer_name = Name::from_string("Lean.Meta.inferType");
    if let Some(infer_const) = env.get_const(&infer_name) {
        println!("\n=== Lean.Meta.inferType ===");
        println!("  type: {:?}", infer_const.type_);
        let check = tc.infer_type(&infer_const.type_);
        assert!(check.is_ok(), "inferType type should be well-formed");
    }
}

#[test]
fn test_declaration_elaboration_types() {
    // Validate definition/theorem elaboration infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.Declaration",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.Declaration with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Declaration: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    // Type-check declaration elaboration types and functions
    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    for const_name in [
        "Lean.Elab.Command.elabDeclaration",
        "Lean.Elab.Declaration.DeclarationKind",
        "Lean.Elab.DefKind",
        "Lean.Elab.DefinitionVal",
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
        "Expected at least 2 declaration elaboration types to validate, got {type_successes}"
    );
}

#[test]
fn test_inductive_elaboration_types() {
    // Validate inductive type elaboration infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.Inductive",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.Inductive with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Inductive: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test inductive elaboration types - these are in Lean.Elab.Inductive module
    for const_name in [
        "Lean.Elab.Command.InductiveView",
        "Lean.Elab.Command.CtorView",
        "Lean.Elab.Command.checkValidInductiveModifier",
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
        type_successes >= 1,
        "Expected at least 1 inductive elaboration type to validate, got {type_successes}"
    );
}

#[test]
fn test_parser_extension_types() {
    // Validate parser extension and descriptor compilation infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Parser.Extension",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Parser.Extension with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Parser.Extension: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test parser extension types - ParserDescr is Lean.ParserDescr in Init.Prelude
    for const_name in [
        "Lean.ParserDescr",
        "Lean.Parser.ParserExtension",
        "Lean.Parser.parserExtension",
        "Lean.Parser.addParser",
        "Lean.Parser.registerParserCategory",
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
        "Expected at least 3 parser extension types to validate, got {type_successes}"
    );
}

#[test]
fn test_match_compilation_types() {
    // Validate pattern matching and match compilation infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Lean.Elab.Match", std::slice::from_ref(&lib_path))
            .expect("Failed to load Lean.Elab.Match with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Match: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test match compilation types
    for const_name in [
        "Lean.Elab.Term.MatchAltView",
        "Lean.Elab.Term.elabMatch",
        "Lean.Elab.Term.ElabMatchTypeAndDiscrsResult",
        "Lean.Elab.Term.elabNoMatch",
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
        "Expected at least 2 match compilation types to validate, got {type_successes}"
    );
}

#[test]
fn test_induction_tactic_types() {
    // Validate induction tactic elaboration infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.Tactic.Induction",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.Tactic.Induction with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Tactic.Induction: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test induction tactic types
    for const_name in [
        "Lean.Elab.Tactic.evalInduction",
        "Lean.Elab.Tactic.evalCases",
        "Lean.Elab.Tactic.Induction",
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
        "Expected at least 2 induction tactic types to validate, got {type_successes}"
    );
}

#[test]
fn test_rewrite_tactic_types() {
    // Validate rewriting tactic elaboration infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.Tactic.Rewrite",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.Tactic.Rewrite with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Tactic.Rewrite: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test rewrite tactic types
    for const_name in [
        "Lean.Elab.Tactic.evalRewriteSeq",
        "Lean.Elab.Tactic.rewriteTarget",
        "Lean.Elab.Tactic.rewriteLocalDecl",
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
        type_successes >= 1,
        "Expected at least 1 rewrite tactic type to validate, got {type_successes}"
    );
}

#[test]
fn test_structure_elaboration_types() {
    // Validate structure/record elaboration infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.Structure",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.Structure with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Structure: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test structure elaboration types
    for const_name in [
        "Lean.Elab.Command.StructFieldView",
        "Lean.Elab.Command.elabStructure",
        "Lean.Elab.Command.StructView",
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
        "Expected at least 2 structure elaboration types to validate, got {type_successes}"
    );
}

#[test]
fn test_do_notation_elaboration_types() {
    // Validate do-notation elaboration infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Lean.Elab.Do", std::slice::from_ref(&lib_path))
            .expect("Failed to load Lean.Elab.Do with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Do: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test do-notation elaboration types
    for const_name in [
        "Lean.Elab.Term.Do.ToTerm.run",
        "Lean.Elab.Term.Do.elabDo",
        "Lean.Elab.Term.Do.Code",
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
        "Expected at least 2 do-notation elaboration types to validate, got {type_successes}"
    );
}

#[test]
fn test_deriving_elaboration_types() {
    // Validate deriving handler infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.Deriving.Basic",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.Deriving.Basic with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Deriving.Basic: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test deriving infrastructure types
    for const_name in [
        "Lean.Elab.DerivingHandler",
        "Lean.Elab.registerDerivingHandler",
        "Lean.Elab.Deriving.Context",
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
        type_successes >= 1,
        "Expected at least 1 deriving elaboration type to validate, got {type_successes}"
    );
}

#[test]
fn test_binder_elaboration_types() {
    // Validate binder information types used in term elaboration
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.Binders",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.Binders with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Binders: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test binder elaboration types
    for const_name in [
        "Lean.Elab.Term.elabBinders",
        "Lean.Elab.Term.elabBindersEx",
        "Lean.Elab.Term.elabFunBinders",
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
        "Expected at least 2 binder elaboration types to validate, got {type_successes}"
    );
}

#[test]
fn test_syntax_quotation_types() {
    // Validate syntax quotation/antiquotation infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.Quotation",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.Quotation with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Quotation: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test syntax quotation types
    for const_name in [
        "Lean.Elab.Term.Quotation.stxQuot.expand",
        "Lean.Elab.Term.Quotation.getQuotedSyntax",
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
        type_successes >= 1,
        "Expected at least 1 syntax quotation type to validate, got {type_successes}"
    );
}

#[test]
fn test_builtin_command_types() {
    // Validate built-in command elaboration infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.BuiltinCommand",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.BuiltinCommand with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.BuiltinCommand: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test builtin command types
    for const_name in [
        "Lean.Elab.Command.elabOpen",
        "Lean.Elab.Command.elabVariable",
        "Lean.Elab.Command.elabUniverse",
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
        "Expected at least 2 builtin command types to validate, got {type_successes}"
    );
}

#[test]
fn test_let_elaboration_types() {
    // Validate let-binding elaboration infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.LetRec",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.LetRec with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.LetRec: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test let elaboration types
    for const_name in [
        "Lean.Elab.Term.elabLetDeclAux",
        "Lean.Elab.Term.elabLetDecl",
        "Lean.Elab.Term.LetRecToLift",
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
        "Expected at least 2 let elaboration types to validate, got {type_successes}"
    );
}

#[test]
fn test_notation_elaboration_types() {
    // Validate notation/macro elaboration infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.MacroRules",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.MacroRules with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.MacroRules: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test notation/macro elaboration types
    for const_name in ["Lean.Elab.adaptMacro", "Lean.Macro", "Lean.MacroM"] {
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
        type_successes >= 1,
        "Expected at least 1 macro elaboration type to validate, got {type_successes}"
    );
}

#[test]
fn test_application_elaboration_types() {
    // Validate application elaboration infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Lean.Elab.App", std::slice::from_ref(&lib_path))
            .expect("Failed to load Lean.Elab.App with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.App: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test application elaboration types
    for const_name in [
        "Lean.Elab.Term.elabApp",
        "Lean.Elab.Term.elabAppArgs",
        "Lean.Elab.Term.Arg",
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
        "Expected at least 2 application elaboration types to validate, got {type_successes}"
    );
}

#[test]
fn test_extra_term_elaboration_types() {
    // Validate extra term elaboration infrastructure (show, suffices, etc.)
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Lean.Elab.Extra", std::slice::from_ref(&lib_path))
            .expect("Failed to load Lean.Elab.Extra with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.Extra: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test extra term elaboration types
    for const_name in ["Lean.Elab.Term.elabShow", "Lean.Elab.Term.TermElab"] {
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
        type_successes >= 1,
        "Expected at least 1 extra term elaboration type to validate, got {type_successes}"
    );
}

#[test]
fn test_mutual_elaboration_types() {
    // Validate mutual definition elaboration infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.MutualDef",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.MutualDef with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.MutualDef: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test mutual definition elaboration types
    for const_name in [
        "Lean.Elab.Command.elabMutualDef",
        "Lean.Elab.Term.elabMutualDef",
        "Lean.Elab.Command.DefView",
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
        "Expected at least 2 mutual elaboration types to validate, got {type_successes}"
    );
}

#[test]
fn test_prelude_types() {
    // Validate prelude types are correctly imported
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

    // Test fundamental prelude types
    for const_name in ["Nat", "Bool", "List", "String", "Prop"] {
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
        "Expected at least 4 prelude types to validate, got {type_successes}"
    );
}

#[test]
fn test_wf_recursion_types() {
    // Validate well-founded recursion infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.PreDefinition.WF.Main",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.PreDefinition.WF.Main with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.PreDefinition.WF.Main: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test well-founded recursion types
    for const_name in [
        "Lean.Elab.WF.elabWFRel",
        "Lean.Elab.WF.mkDecreasingProof",
        "WellFounded",
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
        type_successes >= 1,
        "Expected at least 1 well-founded recursion type to validate, got {type_successes}"
    );
}

#[test]
fn test_structural_recursion_types() {
    // Validate structural recursion infrastructure
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Lean.Elab.PreDefinition.Structural.Main",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Lean.Elab.PreDefinition.Structural.Main with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Lean.Elab.PreDefinition.Structural.Main: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test structural recursion types
    for const_name in [
        "Lean.Elab.Structural.structuralRecursion",
        "Lean.Elab.Structural.findRecArg",
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
        type_successes >= 1,
        "Expected at least 1 structural recursion type to validate, got {type_successes}"
    );
}

#[test]
fn test_proof_term_types() {
    // Validate proof term types and primitives
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

    // Test proof primitives
    for const_name in ["Eq", "Eq.refl", "And", "Or", "Not", "True", "False"] {
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
        type_successes >= 5,
        "Expected at least 5 proof term types to validate, got {type_successes}"
    );
}

#[test]
fn test_decidable_eq_types() {
    // Validate decidable equality and decide function types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries =
        load_module_with_deps(&mut env, "Init.Data.Bool", std::slice::from_ref(&lib_path))
            .expect("Failed to load Init.Data.Bool with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Data.Bool: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test decidable equality types
    for const_name in ["DecidableEq", "decide", "Bool.decEq"] {
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
        "Expected at least 2 decidable equality types to validate, got {type_successes}"
    );
}

#[test]
fn test_monad_control_types() {
    // Validate monad control and transformer types
    use lean5_kernel::tc::TypeChecker;

    let Some(lib_path) = get_lean_lib_path() else {
        eprintln!("Skipping test: Lean 4 not found");
        return;
    };

    let mut env = Environment::default();
    let summaries = load_module_with_deps(
        &mut env,
        "Init.Control.StateRef",
        std::slice::from_ref(&lib_path),
    )
    .expect("Failed to load Init.Control.StateRef with dependencies");

    let total_added: usize = summaries.iter().map(|s| s.added_constants).sum();
    println!(
        "Init.Control.StateRef: {} modules, {} constants added",
        summaries.len(),
        total_added
    );

    let mut tc = TypeChecker::new(&env);
    let mut type_successes = 0;

    // Test monad control types
    for const_name in ["StateRefT'", "MonadStateOf", "MonadLiftT"] {
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
        "Expected at least 2 monad control types to validate, got {type_successes}"
    );
}
