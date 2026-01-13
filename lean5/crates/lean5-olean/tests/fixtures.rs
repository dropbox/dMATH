//! Offline tests using checked-in .olean fixtures
//!
//! These tests do NOT require a system Lean installation.
//! They use pre-compiled .olean files from tests/fixtures/olean/v4.13.0/

use lean5_kernel::Environment;
use lean5_olean::{parse_header, parse_module, ConstantKind};
use std::fs;
use std::path::{Path, PathBuf};

fn fixtures_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/olean/v4.13.0")
}

fn read_fixture(relative_path: &str) -> Vec<u8> {
    let path = fixtures_path().join(relative_path);
    fs::read(&path).unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", path.display(), e))
}

// =============================================================================
// Header Parsing Tests
// =============================================================================

#[test]
fn test_parse_minimal_header() {
    let bytes = read_fixture("custom/Minimal.olean");
    let header = parse_header(&bytes).expect("Failed to parse Minimal.olean header");

    assert_eq!(header.version, 1, "Expected .olean version 1");
    assert!(header.base_addr > 0, "Expected non-zero base address");
}

#[test]
fn test_parse_inductive_header() {
    let bytes = read_fixture("custom/Inductive.olean");
    let header = parse_header(&bytes).expect("Failed to parse Inductive.olean header");

    assert_eq!(header.version, 1);
}

#[test]
fn test_parse_structure_header() {
    let bytes = read_fixture("custom/Structure.olean");
    let header = parse_header(&bytes).expect("Failed to parse Structure.olean header");

    assert_eq!(header.version, 1);
}

#[test]
fn test_parse_init_header() {
    let bytes = read_fixture("stdlib/Init.olean");
    let header = parse_header(&bytes).expect("Failed to parse Init.olean header");

    assert_eq!(header.version, 1);
}

// =============================================================================
// Module Parsing Tests
// =============================================================================

#[test]
fn test_parse_minimal_module() {
    let bytes = read_fixture("custom/Minimal.olean");
    let module = parse_module(&bytes).expect("Failed to parse Minimal.olean module");

    // Check module has expected constants
    let const_names: Vec<&str> = module.constants.iter().map(|c| c.name.as_str()).collect();

    assert!(
        const_names.contains(&"identity"),
        "Expected 'identity' constant, got: {const_names:?}"
    );
    assert!(
        const_names.contains(&"id_id"),
        "Expected 'id_id' theorem, got: {const_names:?}"
    );
}

#[test]
fn test_parse_inductive_module() {
    let bytes = read_fixture("custom/Inductive.olean");
    let module = parse_module(&bytes).expect("Failed to parse Inductive.olean module");

    let const_names: Vec<&str> = module.constants.iter().map(|c| c.name.as_str()).collect();

    // Check for inductive type and constructors
    assert!(
        const_names.contains(&"MyBool"),
        "Expected 'MyBool' inductive type"
    );
    assert!(
        const_names.contains(&"MyBool.myTrue"),
        "Expected 'MyBool.myTrue' constructor"
    );
    assert!(
        const_names.contains(&"MyBool.myFalse"),
        "Expected 'MyBool.myFalse' constructor"
    );
    assert!(const_names.contains(&"myNot"), "Expected 'myNot' function");
}

#[test]
fn test_parse_structure_module() {
    let bytes = read_fixture("custom/Structure.olean");
    let module = parse_module(&bytes).expect("Failed to parse Structure.olean module");

    let const_names: Vec<&str> = module.constants.iter().map(|c| c.name.as_str()).collect();

    // Check for structure and projections
    assert!(
        const_names.contains(&"MyPair"),
        "Expected 'MyPair' structure"
    );
    assert!(const_names.contains(&"swap"), "Expected 'swap' function");
}

#[test]
fn test_parse_init_module() {
    let bytes = read_fixture("stdlib/Init.olean");
    let module = parse_module(&bytes).expect("Failed to parse Init.olean module");

    // Init.olean should have imports to Init submodules
    assert!(!module.imports.is_empty(), "Init.olean should have imports");
}

// =============================================================================
// Import Count Tests
// =============================================================================

#[test]
fn test_minimal_has_init_import() {
    let bytes = read_fixture("custom/Minimal.olean");
    let module = parse_module(&bytes).expect("Failed to parse module");

    // Custom modules import Init
    let import_names: Vec<&str> = module
        .imports
        .iter()
        .map(|i| i.module_name.as_str())
        .collect();
    assert!(
        import_names.iter().any(|n: &&str| n.contains("Init")),
        "Custom module should import Init, got: {import_names:?}"
    );
}

// =============================================================================
// Constant Type Tests
// =============================================================================

#[test]
fn test_identity_type() {
    let bytes = read_fixture("custom/Minimal.olean");
    let module = parse_module(&bytes).expect("Failed to parse module");

    let identity = module
        .constants
        .iter()
        .find(|c| c.name == "identity")
        .expect("identity constant not found");

    // identity should be a definition
    assert!(
        matches!(identity.kind, ConstantKind::Definition),
        "identity should be a definition"
    );
}

#[test]
fn test_id_id_is_theorem() {
    let bytes = read_fixture("custom/Minimal.olean");
    let module = parse_module(&bytes).expect("Failed to parse module");

    let id_id = module
        .constants
        .iter()
        .find(|c| c.name == "id_id")
        .expect("id_id constant not found");

    // id_id should be a theorem
    assert!(
        matches!(id_id.kind, ConstantKind::Theorem),
        "id_id should be a theorem"
    );
}

#[test]
fn test_mybool_is_inductive() {
    let bytes = read_fixture("custom/Inductive.olean");
    let module = parse_module(&bytes).expect("Failed to parse module");

    let mybool = module
        .constants
        .iter()
        .find(|c| c.name == "MyBool")
        .expect("MyBool constant not found");

    // MyBool should be an inductive
    assert!(
        matches!(mybool.kind, ConstantKind::Inductive),
        "MyBool should be an inductive type"
    );
}

#[test]
fn test_mybool_true_is_constructor() {
    let bytes = read_fixture("custom/Inductive.olean");
    let module = parse_module(&bytes).expect("Failed to parse module");

    let mytrue = module
        .constants
        .iter()
        .find(|c| c.name == "MyBool.myTrue")
        .expect("MyBool.myTrue constant not found");

    // myTrue should be a constructor
    assert!(
        matches!(mytrue.kind, ConstantKind::Constructor),
        "MyBool.myTrue should be a constructor"
    );
}

// =============================================================================
// Load Into Environment Tests
// =============================================================================

#[test]
fn test_load_minimal_into_env() {
    let bytes = read_fixture("custom/Minimal.olean");
    let module = parse_module(&bytes).expect("Failed to parse module");

    let _env = Environment::new();

    // This tests that parsed constants can be loaded
    // We just check it doesn't panic - full loading requires dependency resolution
    assert!(
        !module.constants.is_empty(),
        "Module should have constants to load"
    );
}

// =============================================================================
// Header Git Hash Tests
// =============================================================================

#[test]
fn test_headers_have_same_lean_version() {
    let minimal_bytes = read_fixture("custom/Minimal.olean");
    let inductive_bytes = read_fixture("custom/Inductive.olean");
    let structure_bytes = read_fixture("custom/Structure.olean");

    let h1 = parse_header(&minimal_bytes).unwrap();
    let h2 = parse_header(&inductive_bytes).unwrap();
    let h3 = parse_header(&structure_bytes).unwrap();

    // All custom fixtures compiled with same Lean version
    assert_eq!(h1.git_hash, h2.git_hash, "Git hashes should match");
    assert_eq!(h2.git_hash, h3.git_hash, "Git hashes should match");
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_invalid_header_rejected() {
    let bad_bytes = b"not an olean file";
    let result = parse_header(bad_bytes);
    assert!(result.is_err(), "Invalid header should be rejected");
}

#[test]
fn test_empty_bytes_rejected() {
    let empty: &[u8] = &[];
    let result = parse_header(empty);
    assert!(result.is_err(), "Empty input should be rejected");
}

#[test]
fn test_truncated_header_rejected() {
    let bytes = read_fixture("custom/Minimal.olean");
    let truncated = &bytes[..20]; // Header is 56 bytes
    let result = parse_header(truncated);
    assert!(result.is_err(), "Truncated header should be rejected");
}
