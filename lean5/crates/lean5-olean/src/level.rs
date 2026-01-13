//! Level (universe level) parsing from .olean files
//!
//! Lean 4 universe levels are represented as an inductive type:
//!
//! ```text
//! inductive Level where
//!   | zero                                -- tag 0
//!   | succ   (pred : Level)               -- tag 1, 1 field
//!   | max    (l r : Level)                -- tag 2, 2 fields
//!   | imax   (l r : Level)                -- tag 3, 2 fields
//!   | param  (name : Name)                -- tag 4, 1 field
//!   | mvar   (mvarId : LMVarId)           -- tag 5, 1 field
//! ```
//!
//! In the compacted region, these are constructor objects with the appropriate
//! number of pointer fields.

use crate::error::{OleanError, OleanResult};
use crate::region::{is_ptr, is_scalar, unbox_scalar, CompactedRegion};
use std::fmt;

/// Level tags (constructor indices)
pub mod level_tags {
    pub const ZERO: u8 = 0;
    pub const SUCC: u8 = 1;
    pub const MAX: u8 = 2;
    pub const IMAX: u8 = 3;
    pub const PARAM: u8 = 4;
    pub const MVAR: u8 = 5;
}

/// A parsed universe level
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedLevel {
    /// Level 0 (Prop)
    Zero,
    /// Successor level
    Succ(Box<ParsedLevel>),
    /// Maximum of two levels
    Max(Box<ParsedLevel>, Box<ParsedLevel>),
    /// Impredicative maximum
    IMax(Box<ParsedLevel>, Box<ParsedLevel>),
    /// Universe parameter
    Param(String),
    /// Metavariable (shouldn't appear in .olean)
    MVar(String),
}

impl fmt::Display for ParsedLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParsedLevel::Zero => write!(f, "0"),
            ParsedLevel::Succ(l) => {
                // Try to collapse successive Succs into a number
                let mut level = l.as_ref();
                let mut count = 1u64;
                while let ParsedLevel::Succ(inner) = level {
                    count += 1;
                    level = inner;
                }
                if let ParsedLevel::Zero = level {
                    write!(f, "{count}")
                } else {
                    write!(f, "(succ {l})")
                }
            }
            ParsedLevel::Max(l, r) => write!(f, "(max {l} {r})"),
            ParsedLevel::IMax(l, r) => write!(f, "(imax {l} {r})"),
            ParsedLevel::Param(n) => write!(f, "{n}"),
            ParsedLevel::MVar(n) => write!(f, "?{n}"),
        }
    }
}

impl ParsedLevel {
    /// Count the depth of the level (for detecting infinite loops)
    pub fn depth(&self) -> usize {
        match self {
            ParsedLevel::Zero | ParsedLevel::Param(_) | ParsedLevel::MVar(_) => 0,
            ParsedLevel::Succ(l) => 1 + l.depth(),
            ParsedLevel::Max(l, r) | ParsedLevel::IMax(l, r) => 1 + l.depth().max(r.depth()),
        }
    }
}

impl<'a> CompactedRegion<'a> {
    /// Read a Level object at a file offset
    pub fn read_level_at(&self, offset: usize) -> OleanResult<ParsedLevel> {
        self.read_level_at_depth(offset, 0)
    }

    pub(crate) fn read_level_at_depth(
        &self,
        offset: usize,
        depth: usize,
    ) -> OleanResult<ParsedLevel> {
        if depth > 1000 {
            return Err(OleanError::Region("Level depth limit exceeded".into()));
        }

        let header = self.read_header_at(offset)?;

        let field_base = offset + 8;
        let _scalar_base = field_base + header.other as usize * 8;

        match header.tag {
            level_tags::ZERO => {
                // Level.zero: constructor 0, 0 fields
                // But may have a Data scalar field
                Ok(ParsedLevel::Zero)
            }

            level_tags::SUCC => {
                // Level.succ: constructor 1, 1 field (pred)
                // Layout: header(8) + data(8) + pred(8)
                // The Data field comes before the pred pointer
                let pred_ptr = self.read_u64_at(field_base)?; // Skip header
                let pred = self.resolve_level_ptr(pred_ptr, depth + 1)?;
                Ok(ParsedLevel::Succ(Box::new(pred)))
            }

            level_tags::MAX => {
                // Level.max: constructor 2, 2 fields (l, r)
                // Layout: header(8) + data(8) + l(8) + r(8)
                let l_ptr = self.read_u64_at(field_base)?;
                let r_ptr = self.read_u64_at(field_base + 8)?;
                let l = self.resolve_level_ptr(l_ptr, depth + 1)?;
                let r = self.resolve_level_ptr(r_ptr, depth + 1)?;
                Ok(ParsedLevel::Max(Box::new(l), Box::new(r)))
            }

            level_tags::IMAX => {
                // Level.imax: constructor 3, 2 fields (l, r)
                // Layout: header(8) + data(8) + l(8) + r(8)
                let l_ptr = self.read_u64_at(field_base)?;
                let r_ptr = self.read_u64_at(field_base + 8)?;
                let l = self.resolve_level_ptr(l_ptr, depth + 1)?;
                let r = self.resolve_level_ptr(r_ptr, depth + 1)?;
                Ok(ParsedLevel::IMax(Box::new(l), Box::new(r)))
            }

            level_tags::PARAM => {
                // Level.param: constructor 4, 1 field (name)
                // Layout: header(8) + data(8) + name(8)
                let name_ptr = self.read_u64_at(field_base)?;
                let name = if is_scalar(name_ptr) {
                    // Name.anonymous encoded as scalar 0
                    String::new()
                } else if is_ptr(name_ptr) {
                    let name_off = self.ptr_to_offset(name_ptr)?;
                    self.read_name_at(name_off)?
                } else {
                    String::new()
                };
                Ok(ParsedLevel::Param(name))
            }

            level_tags::MVAR => {
                // Level.mvar: constructor 5, 1 field (mvarId)
                // Layout: header(8) + data(8) + mvarId(8)
                let id_ptr = self.read_u64_at(field_base)?;
                let name = if is_scalar(id_ptr) {
                    format!("mvar_{}", unbox_scalar(id_ptr))
                } else if is_ptr(id_ptr) {
                    let name_off = self.ptr_to_offset(id_ptr)?;
                    // LMVarId contains a Name
                    self.read_name_at(name_off)?
                } else {
                    "?".to_string()
                };
                Ok(ParsedLevel::MVar(name))
            }

            _ => Err(OleanError::InvalidObjectTag {
                tag: header.tag,
                offset,
            }),
        }
    }

    /// Resolve a level pointer (handling scalars for Level.zero)
    pub(crate) fn resolve_level_ptr(&self, ptr: u64, depth: usize) -> OleanResult<ParsedLevel> {
        if is_scalar(ptr) {
            // Level.zero is often encoded as scalar 0 (pointer value 1)
            // Similar to Name.anonymous
            let val = unbox_scalar(ptr);
            if val == 0 {
                return Ok(ParsedLevel::Zero);
            }
            // Could be a numeric level directly
            return Err(OleanError::Region(format!(
                "Unexpected scalar level value: {val}"
            )));
        }

        if !is_ptr(ptr) {
            return Ok(ParsedLevel::Zero);
        }

        let offset = self.ptr_to_offset(ptr)?;
        self.read_level_at_depth(offset, depth)
    }

    /// Find all Level objects in the file
    pub fn find_all_levels(&self) -> Vec<(usize, ParsedLevel)> {
        let mut levels = Vec::new();

        // Start at offset 64 (after header + root pointer)
        let mut offset = 64;
        while offset + 24 < self.data.len() {
            if let Ok(header) = self.read_header_at(offset) {
                // Check for Level constructors (tags 0-5 with appropriate field counts)
                let is_level = matches!(
                    (header.tag, header.other),
                    (level_tags::ZERO, 0)
                        | (level_tags::SUCC | level_tags::PARAM | level_tags::MVAR, 1)
                        | (level_tags::MAX | level_tags::IMAX, 2)
                );

                if is_level {
                    if let Ok(level) = self.read_level_at(offset) {
                        // Only include non-trivial levels
                        if !matches!(level, ParsedLevel::Zero) || header.tag == level_tags::ZERO {
                            levels.push((offset, level));
                        }
                    }
                }
            }
            offset += 8;
        }

        levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_lean_lib_path() -> Option<std::path::PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let elan_path = std::path::PathBuf::from(home).join(".elan/toolchains");

        if elan_path.exists() {
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
    fn test_parsed_level_to_string() {
        assert_eq!(ParsedLevel::Zero.to_string(), "0");
        assert_eq!(
            ParsedLevel::Succ(Box::new(ParsedLevel::Zero)).to_string(),
            "1"
        );
        assert_eq!(
            ParsedLevel::Succ(Box::new(ParsedLevel::Succ(Box::new(ParsedLevel::Zero)))).to_string(),
            "2"
        );
        assert_eq!(ParsedLevel::Param("u".to_string()).to_string(), "u");
        assert_eq!(
            ParsedLevel::Max(
                Box::new(ParsedLevel::Param("u".to_string())),
                Box::new(ParsedLevel::Param("v".to_string()))
            )
            .to_string(),
            "(max u v)"
        );
    }

    #[test]
    fn test_find_levels_in_prelude() {
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
        let header = crate::parse_header(&bytes).expect("Failed to parse header");
        let _region = CompactedRegion::new(&bytes, header.base_addr);

        // This is exploratory - we won't find levels easily with the simple scan
        // because tags 0-5 conflict with Name constructors and other types
        // We'll primarily read levels when traversing from known expression objects

        // Try to read a few levels from known structures
        println!("Searching for level-like objects in Init/Prelude.olean...");
    }
}
