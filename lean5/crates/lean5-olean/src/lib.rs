//! Lean5 .olean file parser
//!
//! This crate parses Lean 4 compiled `.olean` files and loads them
//! into the Lean5 kernel environment.
//!
//! # .olean File Format (Lean 4 v4.x)
//!
//! The file consists of a fixed header followed by a compacted region:
//!
//! | Offset | Size | Field      | Description                           |
//! |--------|------|------------|---------------------------------------|
//! | 0      | 5    | magic      | "olean" ASCII bytes                   |
//! | 5      | 1    | version    | Format version (currently 1)          |
//! | 6      | 42   | git_hash   | Build git hash, null-padded           |
//! | 48     | 8    | base_addr  | Memory address for mmap (little-endian) |
//! | 56     | n    | data       | Compacted region (serialized objects) |
//!
//! The compacted region is a memory dump of Lean 4 runtime objects with
//! pointer fixups for relocation. It uses a sharing-optimized format.

pub mod error;
pub mod export;
pub mod expr;
pub mod header;
pub mod import;
pub mod level;
pub mod module;
pub mod payload;
pub mod region;

pub use error::{OleanError, OleanResult};
pub use expr::{expr_tags, ParsedBinderInfo, ParsedExpr, ParsedLiteral};
pub use header::{OleanHeader, HEADER_SIZE, MAGIC, VERSION};
pub use import::{
    default_search_paths, load_module_with_deps, load_module_with_deps_cached,
    load_module_with_deps_parallel, load_olean_file, load_parsed_module, parse_module,
    parse_module_file, ImportError, LoadSummary, ModuleCache, SkippedConstant,
};
pub use level::{level_tags, ParsedLevel};
pub use module::{
    ConstantKind, ConstructorValData, InductiveValData, ParsedConstant, ParsedImport, ParsedModule,
    RecursorRuleData, RecursorValData, RootAnalysis,
};
pub use payload::{
    decode_lean5_payload, encode_lean5_payload, Lean5Payload, LEAN5_PAYLOAD_MAGIC,
    LEAN5_PAYLOAD_VERSION,
};

/// Parse an .olean file header from bytes
///
/// # Example
///
/// ```ignore
/// use lean5_olean::parse_header;
///
/// let bytes = std::fs::read("Init.Prelude.olean")?;
/// let header = parse_header(&bytes)?;
/// println!("Git hash: {}", header.git_hash);
/// ```
pub fn parse_header(bytes: &[u8]) -> OleanResult<OleanHeader> {
    OleanHeader::parse(bytes)
}

/// Parse only the imports from an .olean file, skipping constant parsing.
///
/// This is much faster than `parse_module` when you only need the dependency list
/// (e.g., for building a module dependency graph).
///
/// # Example
///
/// ```ignore
/// use lean5_olean::parse_imports_only;
///
/// let bytes = std::fs::read("Init.Core.olean")?;
/// let imports = parse_imports_only(&bytes)?;
/// for import in imports {
///     println!("Depends on: {}", import.module_name);
/// }
/// ```
pub fn parse_imports_only(bytes: &[u8]) -> OleanResult<Vec<ParsedImport>> {
    let header = parse_header(bytes)?;
    let region = CompactedRegion::new(bytes, header.base_addr);
    region.read_imports_only()
}

pub use export::OleanExporter;
pub use region::{is_ptr, is_scalar, unbox_scalar, CompactedRegion};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_lean_lib_path() -> Option<PathBuf> {
        // Look for lean installation
        let home = std::env::var("HOME").ok()?;
        let elan_path = PathBuf::from(home).join(".elan/toolchains");

        if elan_path.exists() {
            // Find first lean4 toolchain
            for entry in std::fs::read_dir(&elan_path).ok()? {
                let entry = entry.ok()?;
                let name = entry.file_name();
                if name.to_string_lossy().contains("lean4") {
                    return Some(entry.path().join("lib/lean"));
                }
            }
        }
        None
    }

    #[test]
    fn test_parse_init_prelude_header() {
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        let prelude_path = lib_path.join("Init/Prelude.olean");
        if !prelude_path.exists() {
            eprintln!("Skipping test: Init/Prelude.olean not found at {prelude_path:?}");
            return;
        }

        let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
        let header = parse_header(&bytes).expect("Failed to parse header");

        // Verify magic
        assert_eq!(header.magic, *b"olean");

        // Version should be 1
        assert_eq!(header.version, 1);

        // Git hash should be 40 hex characters
        let hash_str = header.git_hash_str();
        assert!(
            hash_str.chars().all(|c| c.is_ascii_hexdigit()),
            "Git hash should be hex: {hash_str}"
        );

        // Base address should be non-zero
        assert!(header.base_addr != 0, "Base address should be non-zero");

        println!("Parsed header successfully:");
        println!("  Magic: {:?}", std::str::from_utf8(&header.magic));
        println!("  Version: {}", header.version);
        println!("  Git hash: {hash_str}");
        println!("  Base addr: 0x{:x}", header.base_addr);
        println!("  Data size: {} bytes", bytes.len() - HEADER_SIZE);
    }

    #[test]
    fn test_parse_multiple_oleans() {
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        let mut count = 0;
        let mut total_size = 0usize;

        // Test a few different .olean files
        let test_files = ["Init/Prelude.olean", "Init/Core.olean", "Init/Coe.olean"];

        for file in test_files {
            let path = lib_path.join(file);
            if !path.exists() {
                continue;
            }

            let bytes = std::fs::read(&path).expect("Failed to read file");
            let header = parse_header(&bytes).expect("Failed to parse header");

            assert_eq!(header.magic, *b"olean");
            assert_eq!(header.version, 1);

            count += 1;
            total_size += bytes.len();
        }

        println!("Parsed {count} .olean files, total {total_size} bytes");
    }

    #[test]
    fn test_find_names_in_prelude() {
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        let prelude_path = lib_path.join("Init/Prelude.olean");
        if !prelude_path.exists() {
            eprintln!("Skipping test: Init/Prelude.olean not found at {prelude_path:?}");
            return;
        }

        let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
        let header = parse_header(&bytes).expect("Failed to parse header");

        // Create compacted region from full file
        let region = CompactedRegion::new(&bytes, header.base_addr);

        // Find all names
        let names = region.find_all_names();
        println!("Found {} Name objects in Init/Prelude.olean", names.len());

        // Should find many names
        assert!(
            names.len() > 100,
            "Expected > 100 names, got {}",
            names.len()
        );

        // Should find some well-known names
        let name_set: std::collections::HashSet<_> =
            names.iter().map(|(_, n)| n.as_str()).collect();

        // Print first 30 names for debugging
        println!("First 30 names:");
        for (off, name) in names.iter().take(30) {
            println!("  {off}: {name}");
        }

        // Check for expected names (these should exist in Prelude)
        let expected = ["Nat", "Bool", "List", "String", "Prop"];
        for exp in expected {
            if name_set.contains(exp) {
                println!("Found expected name: {exp}");
            }
        }
    }

    #[test]
    fn test_read_specific_name() {
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        let prelude_path = lib_path.join("Init/Prelude.olean");
        if !prelude_path.exists() {
            return;
        }

        let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
        let header = parse_header(&bytes).expect("Failed to parse header");
        let region = CompactedRegion::new(&bytes, header.base_addr);

        // Find where "Nat" string object is by searching for the pattern
        // The string "Nat" should be in a String object at some offset
        let names = region.find_all_names();
        let nat_names: Vec<_> = names.iter().filter(|(_, n)| n == "Nat").collect();

        if nat_names.is_empty() {
            println!("'Nat' name not found - checking available names...");
        } else {
            println!("Found 'Nat' name at offsets: {nat_names:?}");
        }
    }

    #[test]
    fn test_parse_imports_only() {
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        // Init.Core imports Init.Prelude
        let core_path = lib_path.join("Init/Core.olean");
        if !core_path.exists() {
            eprintln!("Skipping test: Init/Core.olean not found");
            return;
        }

        let bytes = std::fs::read(&core_path).expect("Failed to read file");

        // Test fast parse_imports_only
        let imports_fast = parse_imports_only(&bytes).expect("Failed to parse imports");

        // Also parse full module and compare
        let full_module = parse_module(&bytes).expect("Failed to parse module");

        // Both should have same imports
        assert_eq!(
            imports_fast.len(),
            full_module.imports.len(),
            "Import counts should match"
        );

        // Check that Init.Core imports Init.Prelude
        let import_names: Vec<_> = imports_fast
            .iter()
            .map(|i| i.module_name.as_str())
            .collect();
        println!("Init.Core imports: {import_names:?}");

        assert!(
            import_names.contains(&"Init.Prelude"),
            "Init.Core should import Init.Prelude"
        );

        // Verify imports match between fast and full parse
        for (fast_imp, full_imp) in imports_fast.iter().zip(full_module.imports.iter()) {
            assert_eq!(
                fast_imp.module_name, full_imp.module_name,
                "Import names should match"
            );
        }

        println!(
            "parse_imports_only works correctly - {} imports",
            imports_fast.len()
        );
    }

    #[test]
    fn test_parse_imports_only_performance() {
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        // Test with Init.Meta which has many imports
        let meta_path = lib_path.join("Init/Meta.olean");
        if !meta_path.exists() {
            eprintln!("Skipping test: Init/Meta.olean not found");
            return;
        }

        let bytes = std::fs::read(&meta_path).expect("Failed to read file");

        // Time fast path
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = parse_imports_only(&bytes).unwrap();
        }
        let fast_time = start.elapsed() / 10;

        // Time full parse
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = parse_module(&bytes).unwrap();
        }
        let full_time = start.elapsed() / 10;

        let speedup = full_time.as_secs_f64() / fast_time.as_secs_f64();

        println!("\n=== parse_imports_only Performance ===");
        println!("Fast path (imports only): {fast_time:?}");
        println!("Full parse:               {full_time:?}");
        println!("Speedup:                  {speedup:.1}x");

        // Fast path should be significantly faster
        assert!(
            speedup > 2.0,
            "parse_imports_only should be at least 2x faster than full parse, got {speedup:.1}x"
        );
    }
}
