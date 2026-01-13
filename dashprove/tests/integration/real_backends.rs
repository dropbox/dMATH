//! Real backend integration tests
//!
//! These tests run against actual verification tools installed on the system.
//! Tests are marked with #[ignore] by default and can be run with:
//! `cargo test --test integration real_backends -- --ignored`
//!
//! Tool availability:
//! - TLA+: `brew install tla-toolbox` or download from https://lamport.azurewebsites.net/tla/tools.html
//! - Lean 4: `brew install lean4` or https://leanprover.github.io/
//! - Kani: `cargo install --locked kani-verifier && kani setup`
//! - Alloy: Download from https://alloytools.org/
//! - Miri: `rustup +nightly component add miri`
//! - Clippy: `rustup component add clippy`
//! - Z3: `brew install z3` or https://github.com/Z3Prover/z3/releases
//! - CVC5: `brew install cvc5` or https://cvc5.github.io/
//! - Coq: `brew install coq` or https://coq.inria.fr/
//! - Isabelle: Download from https://isabelle.in.tum.de/
//! - Dafny: `brew install dafny` or https://github.com/dafny-lang/dafny/releases
//! - Verus: Install from https://github.com/verus-lang/verus
//! - Prusti: Install from https://github.com/viperproject/prusti-dev
//! - ACL2: `brew install acl2` or https://www.cs.utexas.edu/users/moore/acl2/
//! - HOL4: Build from https://github.com/HOL-Theorem-Prover/HOL
//! - Agda: `cabal install Agda` or https://agda.readthedocs.io/
//! - Idris: `pack install idris2` or https://www.idris-lang.org/
//! - F*: `opam install fstar` or https://www.fstar-lang.org/

use dashprove::dispatcher::{Dispatcher, DispatcherConfig};
use dashprove::usl::{parse, typecheck};
use dashprove_backends::{
    acl2::Acl2Backend, agda::AgdaBackend, alloy::AlloyBackend, coq::CoqBackend, cvc5::Cvc5Backend,
    dafny::DafnyBackend, fstar::FStarBackend, hol4::Hol4Backend, idris::IdrisBackend,
    isabelle::IsabelleBackend, kani::KaniBackend, lean4::Lean4Backend, prusti::PrustiBackend,
    tlaplus::TlaPlusBackend, verus::VerusBackend, z3::Z3Backend, BackendId, VerificationBackend,
    VerificationStatus,
};
use std::sync::Arc;

/// Check if Kani is available
fn kani_available() -> bool {
    std::process::Command::new("cargo-kani")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if Alloy is available
fn alloy_available() -> bool {
    std::process::Command::new("alloy")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
        || std::path::Path::new("/opt/homebrew/bin/alloy").exists()
}

/// Check if TLA+ TLC is available
fn tlaplus_available() -> bool {
    // Check for tlc2 command (TLA+ command-line tools)
    std::process::Command::new("tlc2")
        .arg("-h")
        .output()
        .map(|o| o.status.success() || String::from_utf8_lossy(&o.stderr).contains("TLC"))
        .unwrap_or(false)
        // Also check for java + tla2tools.jar
        || std::path::Path::new("/opt/homebrew/share/tla-toolbox/tla2tools.jar").exists()
        || std::env::var("TLA_TOOLS_PATH").is_ok()
}

/// Check if Lean 4 is available
fn lean4_available() -> bool {
    std::process::Command::new("lean")
        .arg("--version")
        .output()
        .map(|o| {
            o.status.success() && String::from_utf8_lossy(&o.stdout).contains("Lean (version 4")
        })
        .unwrap_or(false)
}

/// Check if Miri is available
fn miri_available() -> bool {
    std::process::Command::new("cargo")
        .args(["+nightly", "miri", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if Clippy is available
fn clippy_available() -> bool {
    std::process::Command::new("cargo")
        .args(["clippy", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if Z3 is available
fn z3_available() -> bool {
    std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if CVC5 is available
fn cvc5_available() -> bool {
    std::process::Command::new("cvc5")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if Coq is available
fn coq_available() -> bool {
    std::process::Command::new("coqc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if Isabelle is available
fn isabelle_available() -> bool {
    // Check for isabelle in PATH
    if let Ok(output) = std::process::Command::new("isabelle")
        .arg("version")
        .output()
    {
        if output.status.success() {
            return true;
        }
    }
    // Check common installation locations
    std::path::Path::new("/Applications/Isabelle2024.app/bin/isabelle").exists()
        || std::path::Path::new("/Applications/Isabelle2023.app/bin/isabelle").exists()
        || std::path::Path::new("/opt/Isabelle2024/bin/isabelle").exists()
}

/// Check if Dafny is available
fn dafny_available() -> bool {
    std::process::Command::new("dafny")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if Verus is available
fn verus_available() -> bool {
    std::process::Command::new("verus")
        .arg("--version")
        .output()
        .map(|o| {
            // Verus may exit non-zero but still be available
            o.status.success() || !String::from_utf8_lossy(&o.stderr).is_empty()
        })
        .unwrap_or(false)
}

/// Check if Prusti is available
fn prusti_available() -> bool {
    // Check for cargo-prusti first
    if let Ok(output) = std::process::Command::new("cargo")
        .args(["prusti", "--version"])
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stdout.contains("Prusti") || stderr.contains("Prusti") {
            return true;
        }
    }
    // Check for prusti-rustc directly
    std::process::Command::new("prusti-rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if ACL2 is available
fn acl2_available() -> bool {
    // Try acl2 first, then acl2s (ACL2 Sedan)
    if let Ok(output) = std::process::Command::new("acl2").arg("--help").output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stdout.contains("ACL2") || stderr.contains("ACL2") || output.status.success() {
            return true;
        }
    }
    std::process::Command::new("acl2s")
        .arg("--help")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if HOL4 is available
fn hol4_available() -> bool {
    // Check for Holmake (preferred for batch mode) or hol
    if let Ok(output) = std::process::Command::new("Holmake").arg("--help").output() {
        if output.status.success() {
            return true;
        }
    }
    std::process::Command::new("hol")
        .arg("--help")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if Agda is available
fn agda_available() -> bool {
    std::process::Command::new("agda")
        .arg("--version")
        .output()
        .map(|o| {
            o.status.success()
                && String::from_utf8_lossy(&o.stdout)
                    .to_lowercase()
                    .contains("agda")
        })
        .unwrap_or(false)
}

/// Check if Idris 2 is available
fn idris_available() -> bool {
    std::process::Command::new("idris2")
        .arg("--version")
        .output()
        .map(|o| {
            o.status.success()
                && String::from_utf8_lossy(&o.stdout)
                    .to_lowercase()
                    .contains("idris")
        })
        .unwrap_or(false)
}

/// Check if F* is available
fn fstar_available() -> bool {
    // F* binary is often named fstar.exe even on Unix
    if let Ok(output) = std::process::Command::new("fstar.exe")
        .arg("--version")
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if output.status.success() && stdout.to_lowercase().contains("f*") {
            return true;
        }
    }
    // Also check for 'fstar' without .exe
    std::process::Command::new("fstar")
        .arg("--version")
        .output()
        .map(|o| {
            o.status.success()
                && String::from_utf8_lossy(&o.stdout)
                    .to_lowercase()
                    .contains("f*")
        })
        .unwrap_or(false)
}

// =============================================================================
// Kani Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Kani installation"]
async fn test_kani_contract_verification_pass() {
    if !kani_available() {
        eprintln!("Skipping: Kani not available");
        return;
    }

    let spec_src = r#"
        contract safe_add(x: Int, y: Int) -> Int {
            requires { x >= 0 }
            requires { y >= 0 }
            ensures { result >= x }
            ensures { result >= y }
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = KaniBackend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: Kani health check failed");
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "Kani verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("Kani result: {:?}", backend_result.status);
}

#[tokio::test]
#[ignore = "requires Kani installation"]
async fn test_kani_contract_verification_fail() {
    if !kani_available() {
        eprintln!("Skipping: Kani not available");
        return;
    }

    // This contract has an impossible postcondition
    let spec_src = r#"
        contract impossible(x: Int) -> Int {
            requires { x > 0 }
            ensures { result < 0 }
            ensures { result > 0 }
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = KaniBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        // Should either be disproven or unknown (depending on Kani behavior)
        assert!(
            matches!(
                backend_result.status,
                VerificationStatus::Disproven | VerificationStatus::Unknown { .. }
            ),
            "Expected disproven or unknown, got: {:?}",
            backend_result.status
        );
    }
}

#[tokio::test]
#[ignore = "requires Kani installation"]
async fn test_kani_via_dispatcher() {
    if !kani_available() {
        eprintln!("Skipping: Kani not available");
        return;
    }

    let spec_src = r#"
        contract increment(n: Int) -> Int {
            requires { n >= 0 }
            requires { n < 1000000 }
            ensures { result == n + 1 }
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Kani));
    dispatcher.register_backend(Arc::new(KaniBackend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "Dispatcher results: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// Alloy Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Alloy installation"]
async fn test_alloy_invariant_verification() {
    if !alloy_available() {
        eprintln!("Skipping: Alloy not available");
        return;
    }

    let spec_src = r#"
        type Counter = { value: Int }

        invariant non_negative {
            forall c: Counter . c.value >= 0
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = AlloyBackend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: Alloy health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "Alloy verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("Alloy result: {:?}", backend_result.status);
}

#[tokio::test]
#[ignore = "requires Alloy installation"]
async fn test_alloy_theorem_verification() {
    if !alloy_available() {
        eprintln!("Skipping: Alloy not available");
        return;
    }

    let spec_src = r#"
        theorem reflexive {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = AlloyBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Alloy theorem result: {:?}", backend_result.status);
        if let Some(proof) = &backend_result.proof {
            println!("Proof: {}", proof);
        }
    }
}

#[tokio::test]
#[ignore = "requires Alloy installation"]
async fn test_alloy_counterexample_generation() {
    if !alloy_available() {
        eprintln!("Skipping: Alloy not available");
        return;
    }

    // This theorem is false - should generate a counterexample
    let spec_src = r#"
        theorem always_zero {
            forall x: Int . x == 0
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = AlloyBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Alloy counterexample result: {:?}", backend_result.status);
        if let Some(ce) = &backend_result.counterexample {
            println!("Counterexample found: {:?}", ce);
        }
        // Should be disproven with a counterexample
        if matches!(backend_result.status, VerificationStatus::Disproven) {
            assert!(
                backend_result.counterexample.is_some(),
                "Should have counterexample"
            );
        }
    }
}

#[tokio::test]
#[ignore = "requires Alloy installation"]
async fn test_alloy_via_dispatcher() {
    if !alloy_available() {
        eprintln!("Skipping: Alloy not available");
        return;
    }

    let spec_src = r#"
        type Node = { id: Int }

        theorem unique_ids {
            forall n1: Node, n2: Node . n1.id == n2.id implies n1 == n2
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Alloy));
    dispatcher.register_backend(Arc::new(AlloyBackend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "Alloy via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// TLA+ Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires TLA+ installation"]
async fn test_tlaplus_temporal_property() {
    if !tlaplus_available() {
        eprintln!("Skipping: TLA+ not available");
        return;
    }

    let spec_src = r#"
        type Counter = { value: Int }

        temporal always_positive {
            always(forall c: Counter . c.value >= 0)
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = TlaPlusBackend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: TLA+ health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "TLA+ verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("TLA+ result: {:?}", backend_result.status);
}

#[tokio::test]
#[ignore = "requires TLA+ installation"]
async fn test_tlaplus_invariant_check() {
    if !tlaplus_available() {
        eprintln!("Skipping: TLA+ not available");
        return;
    }

    let spec_src = r#"
        type State = { x: Int, y: Int }

        invariant positive_sum {
            forall s: State . s.x + s.y >= 0
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = TlaPlusBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("TLA+ invariant result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires TLA+ installation"]
async fn test_tlaplus_via_dispatcher() {
    if !tlaplus_available() {
        eprintln!("Skipping: TLA+ not available");
        return;
    }

    let spec_src = r#"
        type Lock = { held: Bool }

        invariant mutual_exclusion {
            forall l1: Lock, l2: Lock . not(l1.held and l2.held)
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::TlaPlus));
    dispatcher.register_backend(Arc::new(TlaPlusBackend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "TLA+ via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// Lean 4 Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Lean 4 installation"]
async fn test_lean4_theorem_proving() {
    if !lean4_available() {
        eprintln!("Skipping: Lean 4 not available");
        return;
    }

    let spec_src = r#"
        theorem addition_commutative {
            forall x: Int, y: Int . x + y == y + x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = Lean4Backend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: Lean 4 health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "Lean 4 verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("Lean 4 result: {:?}", backend_result.status);
    if let Some(proof) = &backend_result.proof {
        println!("Proof: {}", proof);
    }
}

#[tokio::test]
#[ignore = "requires Lean 4 installation"]
async fn test_lean4_type_definition() {
    if !lean4_available() {
        eprintln!("Skipping: Lean 4 not available");
        return;
    }

    let spec_src = r#"
        type Nat = { value: Int }

        theorem nat_non_negative {
            forall n: Nat . n.value >= 0
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = Lean4Backend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Lean 4 type theorem result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires Lean 4 installation"]
async fn test_lean4_via_dispatcher() {
    if !lean4_available() {
        eprintln!("Skipping: Lean 4 not available");
        return;
    }

    let spec_src = r#"
        theorem identity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Lean4));
    dispatcher.register_backend(Arc::new(Lean4Backend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "Lean 4 via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// Clippy Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Clippy installation"]
async fn test_clippy_lint_check() {
    if !clippy_available() {
        eprintln!("Skipping: Clippy not available");
        return;
    }

    // Run clippy on the project itself
    let output = std::process::Command::new("cargo")
        .args(["clippy", "--workspace", "--", "-D", "warnings"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to run clippy");

    if output.status.success() {
        println!("Clippy: No warnings");
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("Clippy output:\n{}", stderr);
        // Note: We don't fail the test - just report results
    }
}

// =============================================================================
// Miri Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Miri installation"]
async fn test_miri_ub_detection() {
    if !miri_available() {
        eprintln!("Skipping: Miri not available");
        return;
    }

    // Test that Miri can run on a simple test
    let output = std::process::Command::new("cargo")
        .args([
            "+nightly",
            "miri",
            "test",
            "-p",
            "dashprove-usl",
            "--",
            "--test-threads=1",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("MIRIFLAGS", "-Zmiri-disable-isolation")
        .output()
        .expect("Failed to run miri");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("Miri stdout:\n{}", stdout);
    if !stderr.is_empty() {
        println!("Miri stderr:\n{}", stderr);
    }

    // Report success/failure
    if output.status.success() {
        println!("Miri: No undefined behavior detected");
    } else {
        println!("Miri: Test run completed with issues (may be expected)");
    }
}

// =============================================================================
// Z3 SMT Solver Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Z3 installation"]
async fn test_z3_theorem_verification() {
    if !z3_available() {
        eprintln!("Skipping: Z3 not available");
        return;
    }

    let spec_src = r#"
        theorem simple_arith {
            forall x: Int, y: Int . x + y == y + x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = Z3Backend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: Z3 health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "Z3 verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("Z3 result: {:?}", backend_result.status);
    if let Some(proof) = &backend_result.proof {
        println!("Proof: {}", proof);
    }
}

#[tokio::test]
#[ignore = "requires Z3 installation"]
async fn test_z3_invariant_check() {
    if !z3_available() {
        eprintln!("Skipping: Z3 not available");
        return;
    }

    let spec_src = r#"
        type Counter = { value: Int }

        invariant non_negative {
            forall c: Counter . c.value >= 0
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = Z3Backend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Z3 invariant result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires Z3 installation"]
async fn test_z3_counterexample_generation() {
    if !z3_available() {
        eprintln!("Skipping: Z3 not available");
        return;
    }

    // This theorem is false - should generate a counterexample
    let spec_src = r#"
        theorem always_zero {
            forall x: Int . x == 0
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = Z3Backend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Z3 counterexample result: {:?}", backend_result.status);
        if let Some(ce) = &backend_result.counterexample {
            println!("Counterexample found: {:?}", ce);
        }
        // Should be disproven with a counterexample
        if matches!(backend_result.status, VerificationStatus::Disproven) {
            assert!(
                backend_result.counterexample.is_some(),
                "Should have counterexample"
            );
        }
    }
}

#[tokio::test]
#[ignore = "requires Z3 installation"]
async fn test_z3_via_dispatcher() {
    if !z3_available() {
        eprintln!("Skipping: Z3 not available");
        return;
    }

    let spec_src = r#"
        theorem identity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Z3));
    dispatcher.register_backend(Arc::new(Z3Backend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "Z3 via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// CVC5 SMT Solver Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires CVC5 installation"]
async fn test_cvc5_theorem_verification() {
    if !cvc5_available() {
        eprintln!("Skipping: CVC5 not available");
        return;
    }

    let spec_src = r#"
        theorem simple_arith {
            forall x: Int, y: Int . x + y == y + x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = Cvc5Backend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: CVC5 health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "CVC5 verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("CVC5 result: {:?}", backend_result.status);
    if let Some(proof) = &backend_result.proof {
        println!("Proof: {}", proof);
    }
}

#[tokio::test]
#[ignore = "requires CVC5 installation"]
async fn test_cvc5_invariant_check() {
    if !cvc5_available() {
        eprintln!("Skipping: CVC5 not available");
        return;
    }

    let spec_src = r#"
        type Counter = { value: Int }

        invariant non_negative {
            forall c: Counter . c.value >= 0
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = Cvc5Backend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("CVC5 invariant result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires CVC5 installation"]
async fn test_cvc5_counterexample_generation() {
    if !cvc5_available() {
        eprintln!("Skipping: CVC5 not available");
        return;
    }

    // This theorem is false - should generate a counterexample
    let spec_src = r#"
        theorem always_zero {
            forall x: Int . x == 0
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = Cvc5Backend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("CVC5 counterexample result: {:?}", backend_result.status);
        if let Some(ce) = &backend_result.counterexample {
            println!("Counterexample found: {:?}", ce);
        }
        // Should be disproven with a counterexample
        if matches!(backend_result.status, VerificationStatus::Disproven) {
            assert!(
                backend_result.counterexample.is_some(),
                "Should have counterexample"
            );
        }
    }
}

#[tokio::test]
#[ignore = "requires CVC5 installation"]
async fn test_cvc5_via_dispatcher() {
    if !cvc5_available() {
        eprintln!("Skipping: CVC5 not available");
        return;
    }

    let spec_src = r#"
        theorem identity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Cvc5));
    dispatcher.register_backend(Arc::new(Cvc5Backend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "CVC5 via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// Coq Proof Assistant Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Coq installation"]
async fn test_coq_theorem_proving() {
    if !coq_available() {
        eprintln!("Skipping: Coq not available");
        return;
    }

    let spec_src = r#"
        theorem reflexivity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = CoqBackend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: Coq health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "Coq verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("Coq result: {:?}", backend_result.status);
    if let Some(proof) = &backend_result.proof {
        println!("Proof: {}", proof);
    }
}

#[tokio::test]
#[ignore = "requires Coq installation"]
async fn test_coq_type_theorem() {
    if !coq_available() {
        eprintln!("Skipping: Coq not available");
        return;
    }

    let spec_src = r#"
        type Nat = { value: Int }

        theorem nat_non_negative {
            forall n: Nat . n.value >= 0
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = CoqBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Coq type theorem result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires Coq installation"]
async fn test_coq_via_dispatcher() {
    if !coq_available() {
        eprintln!("Skipping: Coq not available");
        return;
    }

    let spec_src = r#"
        theorem identity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Coq));
    dispatcher.register_backend(Arc::new(CoqBackend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "Coq via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// Isabelle/HOL Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Isabelle installation"]
async fn test_isabelle_theorem_proving() {
    if !isabelle_available() {
        eprintln!("Skipping: Isabelle not available");
        return;
    }

    let spec_src = r#"
        theorem reflexivity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = IsabelleBackend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: Isabelle health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "Isabelle verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("Isabelle result: {:?}", backend_result.status);
    if let Some(proof) = &backend_result.proof {
        println!("Proof: {}", proof);
    }
}

#[tokio::test]
#[ignore = "requires Isabelle installation"]
async fn test_isabelle_type_theorem() {
    if !isabelle_available() {
        eprintln!("Skipping: Isabelle not available");
        return;
    }

    let spec_src = r#"
        type Natural = { value: Int }

        theorem natural_reflexive {
            forall n: Natural . n.value == n.value
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = IsabelleBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Isabelle type theorem result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires Isabelle installation"]
async fn test_isabelle_via_dispatcher() {
    if !isabelle_available() {
        eprintln!("Skipping: Isabelle not available");
        return;
    }

    let spec_src = r#"
        theorem identity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Isabelle));
    dispatcher.register_backend(Arc::new(IsabelleBackend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "Isabelle via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// Dafny Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Dafny installation"]
async fn test_dafny_theorem_verification() {
    if !dafny_available() {
        eprintln!("Skipping: Dafny not available");
        return;
    }

    let spec_src = r#"
        theorem reflexivity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = DafnyBackend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: Dafny health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "Dafny verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("Dafny result: {:?}", backend_result.status);
    if let Some(proof) = &backend_result.proof {
        println!("Proof: {}", proof);
    }
}

#[tokio::test]
#[ignore = "requires Dafny installation"]
async fn test_dafny_contract_verification() {
    if !dafny_available() {
        eprintln!("Skipping: Dafny not available");
        return;
    }

    let spec_src = r#"
        contract safe_increment(x: Int) -> Int {
            requires { x >= 0 }
            requires { x < 1000000 }
            ensures { result == x + 1 }
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = DafnyBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Dafny contract result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires Dafny installation"]
async fn test_dafny_invariant_check() {
    if !dafny_available() {
        eprintln!("Skipping: Dafny not available");
        return;
    }

    let spec_src = r#"
        type Counter = { value: Int }

        invariant non_negative {
            forall c: Counter . c.value >= 0
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = DafnyBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Dafny invariant result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires Dafny installation"]
async fn test_dafny_via_dispatcher() {
    if !dafny_available() {
        eprintln!("Skipping: Dafny not available");
        return;
    }

    let spec_src = r#"
        theorem identity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Dafny));
    dispatcher.register_backend(Arc::new(DafnyBackend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "Dafny via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// Verus Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Verus installation"]
async fn test_verus_contract_verification() {
    if !verus_available() {
        eprintln!("Skipping: Verus not available");
        return;
    }

    let spec_src = r#"
        contract safe_add(x: Int, y: Int) -> Int {
            requires { x >= 0 }
            requires { y >= 0 }
            ensures { result >= x }
            ensures { result >= y }
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = VerusBackend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: Verus health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "Verus verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("Verus result: {:?}", backend_result.status);
}

#[tokio::test]
#[ignore = "requires Verus installation"]
async fn test_verus_invariant_check() {
    if !verus_available() {
        eprintln!("Skipping: Verus not available");
        return;
    }

    let spec_src = r#"
        type Buffer = { size: Int }

        invariant size_positive {
            forall b: Buffer . b.size >= 0
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = VerusBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Verus invariant result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires Verus installation"]
async fn test_verus_via_dispatcher() {
    if !verus_available() {
        eprintln!("Skipping: Verus not available");
        return;
    }

    let spec_src = r#"
        contract identity(x: Int) -> Int {
            ensures { result == x }
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Verus));
    dispatcher.register_backend(Arc::new(VerusBackend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "Verus via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// Prusti Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Prusti installation"]
async fn test_prusti_contract_verification() {
    if !prusti_available() {
        eprintln!("Skipping: Prusti not available");
        return;
    }

    let spec_src = r#"
        contract safe_add(x: Int, y: Int) -> Int {
            requires { x >= 0 }
            requires { y >= 0 }
            ensures { result >= 0 }
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = PrustiBackend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: Prusti health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "Prusti verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("Prusti result: {:?}", backend_result.status);
}

#[tokio::test]
#[ignore = "requires Prusti installation"]
async fn test_prusti_invariant_check() {
    if !prusti_available() {
        eprintln!("Skipping: Prusti not available");
        return;
    }

    let spec_src = r#"
        type Vector = { length: Int }

        invariant length_non_negative {
            forall v: Vector . v.length >= 0
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = PrustiBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Prusti invariant result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires Prusti installation"]
async fn test_prusti_overflow_check() {
    if !prusti_available() {
        eprintln!("Skipping: Prusti not available");
        return;
    }

    // This spec tests overflow detection
    let spec_src = r#"
        contract multiply(x: Int, y: Int) -> Int {
            requires { x >= 0 }
            requires { x <= 1000 }
            requires { y >= 0 }
            requires { y <= 1000 }
            ensures { result >= 0 }
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = PrustiBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Prusti overflow check result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires Prusti installation"]
async fn test_prusti_via_dispatcher() {
    if !prusti_available() {
        eprintln!("Skipping: Prusti not available");
        return;
    }

    let spec_src = r#"
        contract identity(x: Int) -> Int {
            requires { x >= 0 }
            ensures { result == x }
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Prusti));
    dispatcher.register_backend(Arc::new(PrustiBackend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "Prusti via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// ACL2 Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires ACL2 installation"]
async fn test_acl2_theorem_proving() {
    if !acl2_available() {
        eprintln!("Skipping: ACL2 not available");
        return;
    }

    let spec_src = r#"
        theorem reflexivity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = Acl2Backend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: ACL2 health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "ACL2 verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("ACL2 result: {:?}", backend_result.status);
    if let Some(proof) = &backend_result.proof {
        println!("Proof: {}", proof);
    }
}

#[tokio::test]
#[ignore = "requires ACL2 installation"]
async fn test_acl2_arithmetic_theorem() {
    if !acl2_available() {
        eprintln!("Skipping: ACL2 not available");
        return;
    }

    let spec_src = r#"
        theorem add_commutative {
            forall x: Int, y: Int . x + y == y + x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = Acl2Backend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!(
            "ACL2 arithmetic theorem result: {:?}",
            backend_result.status
        );
    }
}

#[tokio::test]
#[ignore = "requires ACL2 installation"]
async fn test_acl2_via_dispatcher() {
    if !acl2_available() {
        eprintln!("Skipping: ACL2 not available");
        return;
    }

    let spec_src = r#"
        theorem identity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::ACL2));
    dispatcher.register_backend(Arc::new(Acl2Backend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "ACL2 via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// HOL4 Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires HOL4 installation"]
async fn test_hol4_theorem_proving() {
    if !hol4_available() {
        eprintln!("Skipping: HOL4 not available");
        return;
    }

    let spec_src = r#"
        theorem reflexivity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = Hol4Backend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: HOL4 health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "HOL4 verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("HOL4 result: {:?}", backend_result.status);
    if let Some(proof) = &backend_result.proof {
        println!("Proof: {}", proof);
    }
}

#[tokio::test]
#[ignore = "requires HOL4 installation"]
async fn test_hol4_type_theorem() {
    if !hol4_available() {
        eprintln!("Skipping: HOL4 not available");
        return;
    }

    let spec_src = r#"
        type Natural = { value: Int }

        theorem natural_reflexive {
            forall n: Natural . n.value == n.value
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = Hol4Backend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("HOL4 type theorem result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires HOL4 installation"]
async fn test_hol4_via_dispatcher() {
    if !hol4_available() {
        eprintln!("Skipping: HOL4 not available");
        return;
    }

    let spec_src = r#"
        theorem identity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::HOL4));
    dispatcher.register_backend(Arc::new(Hol4Backend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "HOL4 via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// Agda Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Agda installation"]
async fn test_agda_theorem_proving() {
    if !agda_available() {
        eprintln!("Skipping: Agda not available");
        return;
    }

    let spec_src = r#"
        theorem reflexivity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = AgdaBackend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: Agda health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "Agda verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("Agda result: {:?}", backend_result.status);
    if let Some(proof) = &backend_result.proof {
        println!("Proof: {}", proof);
    }
}

#[tokio::test]
#[ignore = "requires Agda installation"]
async fn test_agda_dependent_type() {
    if !agda_available() {
        eprintln!("Skipping: Agda not available");
        return;
    }

    let spec_src = r#"
        type Vector = { length: Int }

        theorem vector_length_eq {
            forall v: Vector . v.length == v.length
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = AgdaBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Agda dependent type result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires Agda installation"]
async fn test_agda_via_dispatcher() {
    if !agda_available() {
        eprintln!("Skipping: Agda not available");
        return;
    }

    let spec_src = r#"
        theorem identity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Agda));
    dispatcher.register_backend(Arc::new(AgdaBackend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "Agda via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// Idris 2 Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires Idris 2 installation"]
async fn test_idris_theorem_proving() {
    if !idris_available() {
        eprintln!("Skipping: Idris 2 not available");
        return;
    }

    let spec_src = r#"
        theorem reflexivity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = IdrisBackend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: Idris 2 health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "Idris 2 verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("Idris 2 result: {:?}", backend_result.status);
    if let Some(proof) = &backend_result.proof {
        println!("Proof: {}", proof);
    }
}

#[tokio::test]
#[ignore = "requires Idris 2 installation"]
async fn test_idris_dependent_type() {
    if !idris_available() {
        eprintln!("Skipping: Idris 2 not available");
        return;
    }

    let spec_src = r#"
        type Nat = { value: Int }

        theorem nat_equality {
            forall n: Nat . n.value == n.value
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = IdrisBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Idris dependent type result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires Idris 2 installation"]
async fn test_idris_totality_check() {
    if !idris_available() {
        eprintln!("Skipping: Idris 2 not available");
        return;
    }

    let spec_src = r#"
        contract total_function(x: Int) -> Int {
            requires { x >= 0 }
            ensures { result >= 0 }
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = IdrisBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("Idris totality check result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires Idris 2 installation"]
async fn test_idris_via_dispatcher() {
    if !idris_available() {
        eprintln!("Skipping: Idris 2 not available");
        return;
    }

    let spec_src = r#"
        theorem identity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::Idris));
    dispatcher.register_backend(Arc::new(IdrisBackend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "Idris via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// F* Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires F* installation"]
async fn test_fstar_theorem_proving() {
    if !fstar_available() {
        eprintln!("Skipping: F* not available");
        return;
    }

    let spec_src = r#"
        theorem reflexivity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = FStarBackend::new();
    let health = backend.health_check().await;

    if !matches!(health, dashprove_backends::HealthStatus::Healthy) {
        eprintln!("Skipping: F* health check failed: {:?}", health);
        return;
    }

    let result = backend.verify(&typed_spec).await;
    assert!(
        result.is_ok(),
        "F* verification should complete: {:?}",
        result.err()
    );

    let backend_result = result.unwrap();
    println!("F* result: {:?}", backend_result.status);
    if let Some(proof) = &backend_result.proof {
        println!("Proof: {}", proof);
    }
}

#[tokio::test]
#[ignore = "requires F* installation"]
async fn test_fstar_refinement_type() {
    if !fstar_available() {
        eprintln!("Skipping: F* not available");
        return;
    }

    let spec_src = r#"
        type NonNegative = { value: Int }

        theorem non_negative_reflexive {
            forall n: NonNegative . n.value == n.value
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = FStarBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!("F* refinement type result: {:?}", backend_result.status);
    }
}

#[tokio::test]
#[ignore = "requires F* installation"]
async fn test_fstar_contract_verification() {
    if !fstar_available() {
        eprintln!("Skipping: F* not available");
        return;
    }

    let spec_src = r#"
        contract safe_add(x: Int, y: Int) -> Int {
            requires { x >= 0 }
            requires { y >= 0 }
            ensures { result >= x }
            ensures { result >= y }
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let backend = FStarBackend::new();
    let result = backend.verify(&typed_spec).await;

    if let Ok(backend_result) = result {
        println!(
            "F* contract verification result: {:?}",
            backend_result.status
        );
    }
}

#[tokio::test]
#[ignore = "requires F* installation"]
async fn test_fstar_via_dispatcher() {
    if !fstar_available() {
        eprintln!("Skipping: F* not available");
        return;
    }

    let spec_src = r#"
        theorem identity {
            forall x: Int . x == x
        }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::specific(BackendId::FStar));
    dispatcher.register_backend(Arc::new(FStarBackend::new()));

    let results = dispatcher.verify(&typed_spec).await;
    assert!(results.is_ok(), "Dispatcher verification should complete");

    let verification = results.unwrap();
    println!(
        "F* via dispatcher: proven={}, disproven={}, unknown={}",
        verification.summary.proven, verification.summary.disproven, verification.summary.unknown
    );
}

// =============================================================================
// Multi-Backend Integration Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires multiple backends"]
async fn test_redundant_verification_with_real_backends() {
    let kani_ok = kani_available();
    let alloy_ok = alloy_available();
    let tla_ok = tlaplus_available();
    let lean_ok = lean4_available();
    let z3_ok = z3_available();
    let cvc5_ok = cvc5_available();
    let coq_ok = coq_available();
    let isabelle_ok = isabelle_available();
    let dafny_ok = dafny_available();
    let verus_ok = verus_available();
    let prusti_ok = prusti_available();
    let acl2_ok = acl2_available();
    let hol4_ok = hol4_available();
    let agda_ok = agda_available();
    let idris_ok = idris_available();
    let fstar_ok = fstar_available();

    let available_count = [
        kani_ok,
        alloy_ok,
        tla_ok,
        lean_ok,
        z3_ok,
        cvc5_ok,
        coq_ok,
        isabelle_ok,
        dafny_ok,
        verus_ok,
        prusti_ok,
        acl2_ok,
        hol4_ok,
        agda_ok,
        idris_ok,
        fstar_ok,
    ]
    .iter()
    .filter(|&&x| x)
    .count();

    if available_count < 2 {
        eprintln!(
            "Skipping: Need at least 2 backends (have {})",
            available_count
        );
        return;
    }

    println!(
        "Available backends: Kani={}, Alloy={}, TLA+={}, Lean4={}, Z3={}, CVC5={}, Coq={}, Isabelle={}, Dafny={}, Verus={}, Prusti={}, ACL2={}, HOL4={}, Agda={}, Idris={}, F*={}",
        kani_ok, alloy_ok, tla_ok, lean_ok, z3_ok, cvc5_ok, coq_ok, isabelle_ok, dafny_ok, verus_ok, prusti_ok, acl2_ok, hol4_ok, agda_ok, idris_ok, fstar_ok
    );

    let spec_src = r#"
        theorem tautology { true }
    "#;

    let spec = parse(spec_src).expect("parse should succeed");
    let typed_spec = typecheck(spec).expect("typecheck should succeed");

    let mut dispatcher = Dispatcher::new(DispatcherConfig::redundant(available_count));

    if kani_ok {
        dispatcher.register_backend(Arc::new(KaniBackend::new()));
    }
    if alloy_ok {
        dispatcher.register_backend(Arc::new(AlloyBackend::new()));
    }
    if tla_ok {
        dispatcher.register_backend(Arc::new(TlaPlusBackend::new()));
    }
    if lean_ok {
        dispatcher.register_backend(Arc::new(Lean4Backend::new()));
    }
    if z3_ok {
        dispatcher.register_backend(Arc::new(Z3Backend::new()));
    }
    if cvc5_ok {
        dispatcher.register_backend(Arc::new(Cvc5Backend::new()));
    }
    if coq_ok {
        dispatcher.register_backend(Arc::new(CoqBackend::new()));
    }
    if isabelle_ok {
        dispatcher.register_backend(Arc::new(IsabelleBackend::new()));
    }
    if dafny_ok {
        dispatcher.register_backend(Arc::new(DafnyBackend::new()));
    }
    if verus_ok {
        dispatcher.register_backend(Arc::new(VerusBackend::new()));
    }
    if prusti_ok {
        dispatcher.register_backend(Arc::new(PrustiBackend::new()));
    }
    if acl2_ok {
        dispatcher.register_backend(Arc::new(Acl2Backend::new()));
    }
    if hol4_ok {
        dispatcher.register_backend(Arc::new(Hol4Backend::new()));
    }
    if agda_ok {
        dispatcher.register_backend(Arc::new(AgdaBackend::new()));
    }
    if idris_ok {
        dispatcher.register_backend(Arc::new(IdrisBackend::new()));
    }
    if fstar_ok {
        dispatcher.register_backend(Arc::new(FStarBackend::new()));
    }

    let results = dispatcher.verify(&typed_spec).await;

    if let Ok(verification) = results {
        println!("Multi-backend verification:");
        println!("  Proven: {}", verification.summary.proven);
        println!("  Disproven: {}", verification.summary.disproven);
        println!("  Confidence: {}", verification.summary.overall_confidence);

        for (i, prop_result) in verification.properties.iter().enumerate() {
            println!(
                "  Property {}: {} backends responded",
                i,
                prop_result.backend_results.len()
            );
        }
    }
}

// =============================================================================
// Backend Health Check Tests
// =============================================================================

#[tokio::test]
async fn test_all_backend_health_checks() {
    println!("Backend Health Status:");
    println!("======================");

    // Kani
    let kani = KaniBackend::new();
    let kani_health = kani.health_check().await;
    println!("Kani: {:?}", kani_health);

    // Alloy
    let alloy = AlloyBackend::new();
    let alloy_health = alloy.health_check().await;
    println!("Alloy: {:?}", alloy_health);

    // Lean 4
    let lean = Lean4Backend::new();
    let lean_health = lean.health_check().await;
    println!("Lean 4: {:?}", lean_health);

    // TLA+
    let tla = TlaPlusBackend::new();
    let tla_health = tla.health_check().await;
    println!("TLA+: {:?}", tla_health);

    // Isabelle
    let isabelle = IsabelleBackend::new();
    let isabelle_health = isabelle.health_check().await;
    println!("Isabelle: {:?}", isabelle_health);

    // Coq
    let coq = CoqBackend::new();
    let coq_health = coq.health_check().await;
    println!("Coq: {:?}", coq_health);

    // Dafny
    let dafny = DafnyBackend::new();
    let dafny_health = dafny.health_check().await;
    println!("Dafny: {:?}", dafny_health);

    // Z3
    let z3 = Z3Backend::new();
    let z3_health = z3.health_check().await;
    println!("Z3: {:?}", z3_health);

    // CVC5
    let cvc5 = Cvc5Backend::new();
    let cvc5_health = cvc5.health_check().await;
    println!("CVC5: {:?}", cvc5_health);

    // Verus
    let verus = VerusBackend::new();
    let verus_health = verus.health_check().await;
    println!("Verus: {:?}", verus_health);

    // Prusti
    let prusti = PrustiBackend::new();
    let prusti_health = prusti.health_check().await;
    println!("Prusti: {:?}", prusti_health);

    // ACL2
    let acl2 = Acl2Backend::new();
    let acl2_health = acl2.health_check().await;
    println!("ACL2: {:?}", acl2_health);

    // HOL4
    let hol4 = Hol4Backend::new();
    let hol4_health = hol4.health_check().await;
    println!("HOL4: {:?}", hol4_health);

    // Agda
    let agda = AgdaBackend::new();
    let agda_health = agda.health_check().await;
    println!("Agda: {:?}", agda_health);

    // Idris 2
    let idris = IdrisBackend::new();
    let idris_health = idris.health_check().await;
    println!("Idris 2: {:?}", idris_health);

    // F*
    let fstar = FStarBackend::new();
    let fstar_health = fstar.health_check().await;
    println!("F*: {:?}", fstar_health);
}
