//! .olean file export
//!
//! This module provides functionality to serialize Lean5 environments to .olean files.
//! The format matches Lean 4's compacted region format for compatibility.
//!
//! # Usage
//!
//! ```ignore
//! use lean5_olean::export::OleanExporter;
//! use lean5_kernel::Environment;
//!
//! let env = /* elaborated environment */;
//! let bytes = OleanExporter::export(&env, &["MyModule.Def1"], "abcd1234...")?;
//! std::fs::write("MyModule.olean", bytes)?;
//! ```

use crate::error::OleanResult;
use crate::header::{OleanHeader, HEADER_SIZE};
use crate::payload::{encode_lean5_payload, Lean5Payload};
use crate::region::tags;

/// Default base address for exported .olean files
/// This matches the typical base address used by Lean 4
const DEFAULT_BASE_ADDR: u64 = 0x10000;

/// .olean file exporter
///
/// Builds a compacted region containing serialized Lean objects.
pub struct OleanExporter {
    /// Output buffer
    data: Vec<u8>,
    /// Base address for pointer calculation
    base_addr: u64,
    /// String interning table (string -> offset)
    strings: std::collections::HashMap<String, usize>,
    /// Name interning table (name -> offset)
    names: std::collections::HashMap<String, usize>,
}

impl OleanExporter {
    /// Create a new exporter with default settings
    pub fn new() -> Self {
        Self::with_base_addr(DEFAULT_BASE_ADDR)
    }

    /// Create a new exporter with a specific base address
    pub fn with_base_addr(base_addr: u64) -> Self {
        let mut exporter = Self {
            data: Vec::new(),
            base_addr,
            strings: std::collections::HashMap::new(),
            names: std::collections::HashMap::new(),
        };

        // Reserve space for header (56 bytes) + root pointer (8 bytes)
        exporter.data.resize(HEADER_SIZE + 8, 0);
        exporter
    }

    /// Get current write offset
    fn current_offset(&self) -> usize {
        self.data.len()
    }

    /// Convert an offset to a pointer value
    fn offset_to_ptr(&self, offset: usize) -> u64 {
        self.base_addr + offset as u64
    }

    /// Align to 8-byte boundary
    fn align8(&mut self) {
        while !self.data.len().is_multiple_of(8) {
            self.data.push(0);
        }
    }

    /// Write an object header
    fn write_header(&mut self, tag: u8, other: u8, cs_sz: u16) {
        // rc = 0 for compacted objects
        self.data.extend_from_slice(&0i32.to_le_bytes());
        self.data.extend_from_slice(&cs_sz.to_le_bytes());
        self.data.push(other);
        self.data.push(tag);
    }

    /// Write a u64 value
    fn write_u64(&mut self, value: u64) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a tagged scalar (small nat)
    fn scalar_ptr(value: u64) -> u64 {
        (value << 1) | 1
    }

    /// Write a string object and return its offset
    pub fn write_string(&mut self, s: &str) -> usize {
        // Check if already interned
        if let Some(&offset) = self.strings.get(s) {
            return offset;
        }

        self.align8();
        let offset = self.current_offset();

        // String header
        self.write_header(tags::STRING, 0, 0);

        // m_size: byte length including null terminator
        let size = s.len() + 1;
        self.write_u64(size as u64);

        // m_capacity: same as size for compacted strings
        self.write_u64(size as u64);

        // m_length: UTF-8 character count
        let char_count = s.chars().count();
        self.write_u64(char_count as u64);

        // String data with null terminator
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0);

        // Pad to alignment
        self.align8();

        self.strings.insert(s.to_string(), offset);
        offset
    }

    /// Write a Name.anonymous object (tag 0, 0 fields)
    fn write_name_anonymous(&mut self) -> usize {
        self.align8();
        let offset = self.current_offset();
        self.write_header(0, 0, 0);
        offset
    }

    /// Write a Name.str object (tag 1, 2 fields: parent, string)
    fn write_name_str(&mut self, parent_ptr: u64, string_ptr: u64) -> usize {
        self.align8();
        let offset = self.current_offset();
        self.write_header(1, 2, 0);
        self.write_u64(parent_ptr);
        self.write_u64(string_ptr);
        offset
    }

    /// Write a Name.num object (tag 2, 2 fields: parent, number)
    fn write_name_num(&mut self, parent_ptr: u64, num: u64) -> usize {
        self.align8();
        let offset = self.current_offset();
        self.write_header(2, 2, 0);
        self.write_u64(parent_ptr);
        self.write_u64(num);
        offset
    }

    /// Write a hierarchical name (e.g., "Nat.add") and return its offset
    pub fn write_name(&mut self, name: &str) -> usize {
        // Check if already interned
        if let Some(&offset) = self.names.get(name) {
            return offset;
        }

        if name.is_empty() {
            // Name.anonymous
            let offset = self.write_name_anonymous();
            self.names.insert(name.to_string(), offset);
            return offset;
        }

        // Split name into components
        let components: Vec<&str> = name.split('.').collect();

        // Build name from root to leaf
        let mut parent_ptr: u64 = Self::scalar_ptr(0); // Name.anonymous as scalar

        for component in components {
            // Check if component is a number
            if let Ok(num) = component.parse::<u64>() {
                let offset = self.write_name_num(parent_ptr, num);
                parent_ptr = self.offset_to_ptr(offset);
            } else {
                let str_offset = self.write_string(component);
                let str_ptr = self.offset_to_ptr(str_offset);
                let offset = self.write_name_str(parent_ptr, str_ptr);
                parent_ptr = self.offset_to_ptr(offset);
            }
        }

        // The last offset is the full name
        let final_offset = (parent_ptr - self.base_addr) as usize;
        self.names.insert(name.to_string(), final_offset);
        final_offset
    }

    /// Write an Array object (tag 246)
    ///
    /// Array layout:
    /// - header (8 bytes)
    /// - size (8 bytes)
    /// - capacity (8 bytes)
    /// - elements\[size\] (each 8 bytes)
    pub fn write_array(&mut self, elements: &[u64]) -> usize {
        self.align8();
        let offset = self.current_offset();

        // Array header (tag 246, other = 0)
        self.write_header(tags::ARRAY, 0, 0);

        // Size
        self.write_u64(elements.len() as u64);

        // Capacity (same as size for compacted)
        self.write_u64(elements.len() as u64);

        // Elements
        for &elem in elements {
            self.write_u64(elem);
        }

        offset
    }

    /// Write an Import object
    ///
    /// Import is a constructor with:
    /// - tag = 0
    /// - 1 pointer field (module_name: Name)
    /// - 1 scalar byte (runtime_only: Bool)
    ///
    /// The Bool is stored as a scalar byte after the pointer fields, not as a tagged pointer.
    fn write_import(&mut self, module_name: &str, runtime_only: bool) -> usize {
        let name_offset = self.write_name(module_name);
        let name_ptr = self.offset_to_ptr(name_offset);

        self.align8();
        let offset = self.current_offset();

        // Import constructor (tag 0, 1 pointer field)
        self.write_header(0, 1, 0);
        self.write_u64(name_ptr);
        // Bool as scalar byte after the pointer field
        self.data.push(u8::from(runtime_only));
        // Pad to alignment
        self.align8();

        offset
    }

    /// Write a minimal ModuleData structure
    ///
    /// ModuleData fields:
    /// 0: imports (Array Import)
    /// 1: constNames (Array Name)
    /// 2: constants (Array ConstantInfo) - empty for now
    /// 3: extraConstNames (Array Name) - empty
    /// 4: entries (Array EnvExtensionEntry) - empty
    pub fn write_module_data(&mut self, imports: &[(&str, bool)], const_names: &[&str]) -> usize {
        // Write imports array
        let import_ptrs: Vec<u64> = imports
            .iter()
            .map(|(name, rt_only)| {
                let off = self.write_import(name, *rt_only);
                self.offset_to_ptr(off)
            })
            .collect();
        let imports_array_offset = self.write_array(&import_ptrs);
        let imports_ptr = self.offset_to_ptr(imports_array_offset);

        // Write constant names array
        let name_ptrs: Vec<u64> = const_names
            .iter()
            .map(|name| {
                let off = self.write_name(name);
                self.offset_to_ptr(off)
            })
            .collect();
        let names_array_offset = self.write_array(&name_ptrs);
        let names_ptr = self.offset_to_ptr(names_array_offset);

        // Empty arrays for constants, extraConstNames, entries
        let empty_array_offset = self.write_array(&[]);
        let empty_ptr = self.offset_to_ptr(empty_array_offset);

        // Write ModuleData constructor (5 fields)
        self.align8();
        let offset = self.current_offset();
        self.write_header(0, 5, 0);
        self.write_u64(imports_ptr); // imports
        self.write_u64(names_ptr); // constNames
        self.write_u64(empty_ptr); // constants (empty for now)
        self.write_u64(empty_ptr); // extraConstNames
        self.write_u64(empty_ptr); // entries

        offset
    }

    /// Finalize and return the .olean file bytes
    pub fn finalize(mut self, git_hash: &str) -> OleanResult<Vec<u8>> {
        // Create header
        let header = OleanHeader::new(git_hash, self.base_addr)?;
        let header_bytes = header.serialize();

        // Copy header to start of data
        self.data[0..HEADER_SIZE].copy_from_slice(&header_bytes);

        Ok(self.data)
    }

    /// Set the root pointer (module data offset)
    pub fn set_root(&mut self, offset: usize) {
        let ptr = self.offset_to_ptr(offset);
        self.data[HEADER_SIZE..HEADER_SIZE + 8].copy_from_slice(&ptr.to_le_bytes());
    }

    /// Export a minimal .olean file with just imports and constant names
    ///
    /// This is a simplified export that doesn't include full constant definitions,
    /// but is sufficient for dependency tracking.
    pub fn export_minimal(
        imports: &[(&str, bool)],
        const_names: &[&str],
        git_hash: &str,
    ) -> OleanResult<Vec<u8>> {
        let mut exporter = Self::new();

        // Write module data
        let module_offset = exporter.write_module_data(imports, const_names);
        exporter.set_root(module_offset);

        exporter.finalize(git_hash)
    }

    /// Export an .olean file with a Lean5 payload appended after the compacted region.
    ///
    /// The payload carries serialized kernel objects so dependent modules can
    /// load definitions without needing full Lean 4 ConstantInfo serialization.
    pub fn export_with_payload(
        imports: &[(&str, bool)],
        const_names: &[&str],
        git_hash: &str,
        payload: &Lean5Payload,
    ) -> OleanResult<Vec<u8>> {
        let mut exporter = Self::new();

        let module_offset = exporter.write_module_data(imports, const_names);
        exporter.set_root(module_offset);

        let mut bytes = exporter.finalize(git_hash)?;
        let payload_bytes = encode_lean5_payload(payload)?;
        bytes.extend_from_slice(&payload_bytes);
        Ok(bytes)
    }
}

impl Default for OleanExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::CompactedRegion;
    use crate::{parse_header, parse_imports_only};

    #[test]
    fn test_write_string() {
        let mut exp = OleanExporter::new();
        let offset = exp.write_string("hello");

        // Verify we can read it back
        let region = CompactedRegion::new(&exp.data, exp.base_addr);
        let s = region.read_lean_string_at(offset).unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn test_string_interning() {
        let mut exp = OleanExporter::new();
        let off1 = exp.write_string("test");
        let off2 = exp.write_string("test");
        assert_eq!(off1, off2, "same string should return same offset");
    }

    #[test]
    fn test_write_name_simple() {
        let mut exp = OleanExporter::new();
        let offset = exp.write_name("Nat");

        let region = CompactedRegion::new(&exp.data, exp.base_addr);
        let name = region.read_name_at(offset).unwrap();
        assert_eq!(name, "Nat");
    }

    #[test]
    fn test_write_name_hierarchical() {
        let mut exp = OleanExporter::new();
        let offset = exp.write_name("Nat.add");

        let region = CompactedRegion::new(&exp.data, exp.base_addr);
        let name = region.read_name_at(offset).unwrap();
        assert_eq!(name, "Nat.add");
    }

    #[test]
    fn test_write_name_deep() {
        let mut exp = OleanExporter::new();
        let offset = exp.write_name("Lean.Meta.Tactic.simp");

        let region = CompactedRegion::new(&exp.data, exp.base_addr);
        let name = region.read_name_at(offset).unwrap();
        assert_eq!(name, "Lean.Meta.Tactic.simp");
    }

    #[test]
    fn test_name_interning() {
        let mut exp = OleanExporter::new();
        let off1 = exp.write_name("Nat.add");
        let off2 = exp.write_name("Nat.add");
        assert_eq!(off1, off2, "same name should return same offset");
    }

    #[test]
    fn test_export_minimal_roundtrip() {
        let git_hash = "0123456789abcdef0123456789abcdef01234567";

        let bytes = OleanExporter::export_minimal(
            &[("Init.Prelude", false), ("Init.Core", false)],
            &["MyDef", "MyTheorem"],
            git_hash,
        )
        .unwrap();

        // Parse header
        let header = parse_header(&bytes).unwrap();
        assert_eq!(header.git_hash_str(), git_hash);

        // Parse imports
        let imports = parse_imports_only(&bytes).unwrap();
        assert_eq!(imports.len(), 2);
        assert_eq!(imports[0].module_name, "Init.Prelude");
        assert_eq!(imports[1].module_name, "Init.Core");
    }

    #[test]
    fn test_export_empty_module() {
        let git_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

        let bytes = OleanExporter::export_minimal(&[], &[], git_hash).unwrap();

        let header = parse_header(&bytes).unwrap();
        assert_eq!(header.git_hash_str(), git_hash);

        let imports = parse_imports_only(&bytes).unwrap();
        assert!(imports.is_empty());
    }

    #[test]
    fn test_export_full_module_roundtrip() {
        use crate::parse_module;

        let git_hash = "1ea5000000000000000000000000000000000000"; // 40 hex chars

        // Export a module with imports and constants
        let bytes = OleanExporter::export_minimal(
            &[
                ("Init.Prelude", false),
                ("Init.Core", false),
                ("Mathlib.Algebra.Group", false),
            ],
            &[
                "MyModule.myDef",
                "MyModule.myTheorem",
                "MyModule.Helper.util",
            ],
            git_hash,
        )
        .unwrap();

        // Parse the full module
        let module = parse_module(&bytes).unwrap();

        // Verify imports
        assert_eq!(module.imports.len(), 3);
        assert_eq!(module.imports[0].module_name, "Init.Prelude");
        assert_eq!(module.imports[1].module_name, "Init.Core");
        assert_eq!(module.imports[2].module_name, "Mathlib.Algebra.Group");

        // Verify constant names
        assert_eq!(module.const_names.len(), 3);
        assert_eq!(module.const_names[0], "MyModule.myDef");
        assert_eq!(module.const_names[1], "MyModule.myTheorem");
        assert_eq!(module.const_names[2], "MyModule.Helper.util");
    }

    #[test]
    fn test_export_runtime_only_import() {
        let git_hash = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

        let bytes = OleanExporter::export_minimal(
            &[("Init.Data.String", true)], // runtime_only = true
            &[],
            git_hash,
        )
        .unwrap();

        let imports = parse_imports_only(&bytes).unwrap();
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module_name, "Init.Data.String");
        assert!(imports[0].runtime_only);
    }

    #[test]
    fn test_write_name_with_numbers() {
        let mut exp = OleanExporter::new();

        // Test names with numeric components (e.g., _hyg.123)
        let offset = exp.write_name("_hyg.123");

        let region = CompactedRegion::new(&exp.data, exp.base_addr);
        let name = region.read_name_at(offset).unwrap();
        assert_eq!(name, "_hyg.123");
    }
}
