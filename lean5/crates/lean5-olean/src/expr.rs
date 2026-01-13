//! Expression parsing from .olean files
//!
//! Lean 4 expressions are represented as an inductive type with these constructors:
//!
//! ```text
//! inductive Expr where
//!   | bvar   (deBruijnIndex : Nat)                        -- tag 0, scalar
//!   | fvar   (fvarId : FVarId)                            -- tag 1, 1 field
//!   | mvar   (mvarId : MVarId)                            -- tag 2, 1 field
//!   | sort   (u : Level)                                  -- tag 3, 1 field
//!   | const  (declName : Name) (us : List Level)          -- tag 4, 2 fields
//!   | app    (fn : Expr) (arg : Expr)                     -- tag 5, 2 fields
//!   | lam    (binderName : Name) (binderType : Expr)
//!            (body : Expr) (binderInfo : BinderInfo)      -- tag 6, 3 fields + scalar
//!   | forallE(binderName : Name) (binderType : Expr)
//!            (body : Expr) (binderInfo : BinderInfo)      -- tag 7, 3 fields + scalar
//!   | letE   (declName : Name) (type : Expr) (value : Expr)
//!            (body : Expr) (nondep : Bool)                -- tag 8, 4 fields + scalar
//!   | lit    (value : Literal)                            -- tag 9, 1 field
//!   | mdata  (data : MData) (expr : Expr)                 -- tag 10, 2 fields
//!   | proj   (typeName : Name) (idx : Nat) (struct : Expr)-- tag 11, 2 fields + scalar
//! ```

use crate::error::{OleanError, OleanResult};
use crate::level::ParsedLevel;
use crate::region::{is_ptr, is_scalar, tags, unbox_scalar, CompactedRegion};

/// Expression constructor tags
pub mod expr_tags {
    pub const BVAR: u8 = 0;
    pub const FVAR: u8 = 1;
    pub const MVAR: u8 = 2;
    pub const SORT: u8 = 3;
    pub const CONST: u8 = 4;
    pub const APP: u8 = 5;
    pub const LAM: u8 = 6;
    pub const FORALL_E: u8 = 7;
    pub const LET_E: u8 = 8;
    pub const LIT: u8 = 9;
    pub const MDATA: u8 = 10;
    pub const PROJ: u8 = 11;
}

/// Binder information (matches kernel BinderInfo)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParsedBinderInfo {
    Default,
    Implicit,
    StrictImplicit,
    InstImplicit,
}

impl ParsedBinderInfo {
    /// Decode from u8 value (as stored in .olean)
    pub fn from_u8(val: u8) -> Self {
        match val {
            1 => ParsedBinderInfo::Implicit,
            2 => ParsedBinderInfo::StrictImplicit,
            3 => ParsedBinderInfo::InstImplicit,
            // 0 or invalid values -> Default
            _ => ParsedBinderInfo::Default,
        }
    }
}

/// A literal value
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedLiteral {
    Nat(u64),
    String(String),
}

/// A parsed expression
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedExpr {
    /// Bound variable (de Bruijn index)
    BVar(u64),
    /// Free variable
    FVar(String),
    /// Metavariable
    MVar(String),
    /// Sort (Type u)
    Sort(ParsedLevel),
    /// Constant with universe levels
    Const(String, Vec<ParsedLevel>),
    /// Application
    App(Box<ParsedExpr>, Box<ParsedExpr>),
    /// Lambda
    Lam(String, Box<ParsedExpr>, Box<ParsedExpr>, ParsedBinderInfo),
    /// Forall/Pi type
    ForallE(String, Box<ParsedExpr>, Box<ParsedExpr>, ParsedBinderInfo),
    /// Let binding
    LetE(
        String,
        Box<ParsedExpr>,
        Box<ParsedExpr>,
        Box<ParsedExpr>,
        bool,
    ),
    /// Literal
    Lit(ParsedLiteral),
    /// Metadata
    MData(Box<ParsedExpr>),
    /// Projection
    Proj(String, u64, Box<ParsedExpr>),
}

impl ParsedExpr {
    /// Get a short description of the expression kind
    pub fn kind(&self) -> &'static str {
        match self {
            ParsedExpr::BVar(_) => "bvar",
            ParsedExpr::FVar(_) => "fvar",
            ParsedExpr::MVar(_) => "mvar",
            ParsedExpr::Sort(_) => "sort",
            ParsedExpr::Const(_, _) => "const",
            ParsedExpr::App(_, _) => "app",
            ParsedExpr::Lam(_, _, _, _) => "lam",
            ParsedExpr::ForallE(_, _, _, _) => "forallE",
            ParsedExpr::LetE(_, _, _, _, _) => "letE",
            ParsedExpr::Lit(_) => "lit",
            ParsedExpr::MData(_) => "mdata",
            ParsedExpr::Proj(_, _, _) => "proj",
        }
    }

    /// Count the depth of the expression (for limiting recursion)
    pub fn depth(&self) -> usize {
        match self {
            ParsedExpr::BVar(_)
            | ParsedExpr::FVar(_)
            | ParsedExpr::MVar(_)
            | ParsedExpr::Sort(_)
            | ParsedExpr::Lit(_)
            | ParsedExpr::Const(_, _) => 0,
            ParsedExpr::App(f, a) => 1 + f.depth().max(a.depth()),
            ParsedExpr::Lam(_, t, b, _) | ParsedExpr::ForallE(_, t, b, _) => {
                1 + t.depth().max(b.depth())
            }
            ParsedExpr::LetE(_, t, v, b, _) => 1 + t.depth().max(v.depth()).max(b.depth()),
            ParsedExpr::MData(e) | ParsedExpr::Proj(_, _, e) => 1 + e.depth(),
        }
    }
}

/// Work item for iterative expression parsing
enum ExprWork {
    /// Parse expression at this pointer
    Parse(u64),
    /// Build App from top 2 results
    BuildApp,
    /// Build Lam from top 2 results
    BuildLam(String, ParsedBinderInfo),
    /// Build ForallE from top 2 results
    BuildForallE(String, ParsedBinderInfo),
    /// Build LetE from top 3 results
    BuildLetE(String, bool),
    /// Build MData from top result
    BuildMData,
    /// Build Proj from top result
    BuildProj(String, u64),
}

impl<'a> CompactedRegion<'a> {
    /// Read an Expr object at a file offset (iterative to avoid stack overflow)
    pub fn read_expr_at(&self, offset: usize) -> OleanResult<ParsedExpr> {
        // Convert offset to pointer for the unified parsing loop
        let ptr = self.offset_to_ptr(offset);
        self.read_expr_iterative(ptr)
    }

    /// Iterative expression parser to avoid stack overflow on deeply nested expressions
    fn read_expr_iterative(&self, initial_ptr: u64) -> OleanResult<ParsedExpr> {
        let mut work: Vec<ExprWork> = vec![ExprWork::Parse(initial_ptr)];
        let mut results: Vec<ParsedExpr> = Vec::new();

        // Depth limit to prevent infinite loops
        let mut iterations = 0usize;
        const MAX_ITERATIONS: usize = 100_000_000;

        while let Some(item) = work.pop() {
            iterations += 1;
            if iterations > MAX_ITERATIONS {
                return Err(OleanError::Region("Expression too complex".into()));
            }

            match item {
                ExprWork::Parse(ptr) => {
                    // Handle scalar/null pointers
                    if is_scalar(ptr) {
                        results.push(ParsedExpr::BVar(unbox_scalar(ptr)));
                        continue;
                    }
                    if !is_ptr(ptr) {
                        return Err(OleanError::Region("Null expression pointer".into()));
                    }

                    let offset = self.ptr_to_offset(ptr)?;
                    let header = self.read_header_at(offset)?;
                    let field_base = offset + 8;
                    let scalar_base = field_base + header.other as usize * 8;

                    match header.tag {
                        expr_tags::BVAR => {
                            let idx_ptr = self.read_u64_at(field_base)?;
                            let idx = if is_scalar(idx_ptr) {
                                unbox_scalar(idx_ptr)
                            } else if is_ptr(idx_ptr) {
                                self.read_nat_value(idx_ptr)?
                            } else {
                                0
                            };
                            results.push(ParsedExpr::BVar(idx));
                        }

                        expr_tags::FVAR => {
                            let id_ptr = self.read_u64_at(field_base)?;
                            let name = self.resolve_name_ptr(id_ptr)?;
                            results.push(ParsedExpr::FVar(name));
                        }

                        expr_tags::MVAR => {
                            let id_ptr = self.read_u64_at(field_base)?;
                            let name = self.resolve_name_ptr(id_ptr)?;
                            results.push(ParsedExpr::MVar(name));
                        }

                        expr_tags::SORT => {
                            let level_ptr = self.read_u64_at(field_base)?;
                            let level = self.resolve_level_ptr(level_ptr, 0)?;
                            results.push(ParsedExpr::Sort(level));
                        }

                        expr_tags::CONST => {
                            let name_ptr = self.read_u64_at(field_base)?;
                            let levels_ptr = self.read_u64_at(field_base + 8)?;
                            let name = self.resolve_name_ptr(name_ptr)?;
                            let levels = self.read_level_list(levels_ptr)?;
                            results.push(ParsedExpr::Const(name, levels));
                        }

                        expr_tags::LIT => {
                            let lit_ptr = self.read_u64_at(field_base)?;
                            let lit = self.read_literal(lit_ptr)?;
                            results.push(ParsedExpr::Lit(lit));
                        }

                        expr_tags::APP => {
                            let fn_ptr = self.read_u64_at(field_base)?;
                            let arg_ptr = self.read_u64_at(field_base + 8)?;
                            // Push build instruction, then children (arg on top when popped)
                            work.push(ExprWork::BuildApp);
                            work.push(ExprWork::Parse(arg_ptr));
                            work.push(ExprWork::Parse(fn_ptr));
                        }

                        expr_tags::LAM => {
                            let name_ptr = self.read_u64_at(field_base)?;
                            let type_ptr = self.read_u64_at(field_base + 8)?;
                            let body_ptr = self.read_u64_at(field_base + 16)?;
                            let binder_name = self.resolve_name_ptr(name_ptr)?;
                            let binder_info_byte = self.bytes_at(scalar_base, 1)?[0];
                            let binder_info = ParsedBinderInfo::from_u8(binder_info_byte);
                            // Push build, then body, then type (type first on results stack)
                            work.push(ExprWork::BuildLam(binder_name, binder_info));
                            work.push(ExprWork::Parse(body_ptr));
                            work.push(ExprWork::Parse(type_ptr));
                        }

                        expr_tags::FORALL_E => {
                            let name_ptr = self.read_u64_at(field_base)?;
                            let type_ptr = self.read_u64_at(field_base + 8)?;
                            let body_ptr = self.read_u64_at(field_base + 16)?;
                            let binder_name = self.resolve_name_ptr(name_ptr)?;
                            let binder_info_byte = self.bytes_at(scalar_base, 1)?[0];
                            let binder_info = ParsedBinderInfo::from_u8(binder_info_byte);
                            work.push(ExprWork::BuildForallE(binder_name, binder_info));
                            work.push(ExprWork::Parse(body_ptr));
                            work.push(ExprWork::Parse(type_ptr));
                        }

                        expr_tags::LET_E => {
                            let name_ptr = self.read_u64_at(field_base)?;
                            let type_ptr = self.read_u64_at(field_base + 8)?;
                            let value_ptr = self.read_u64_at(field_base + 16)?;
                            let body_ptr = self.read_u64_at(field_base + 24)?;
                            let decl_name = self.resolve_name_ptr(name_ptr)?;
                            let nondep = self.bytes_at(scalar_base, 1)?[0] != 0;
                            // Order: type, value, body -> results stack has body on top
                            work.push(ExprWork::BuildLetE(decl_name, nondep));
                            work.push(ExprWork::Parse(body_ptr));
                            work.push(ExprWork::Parse(value_ptr));
                            work.push(ExprWork::Parse(type_ptr));
                        }

                        expr_tags::MDATA => {
                            let expr_ptr = self.read_u64_at(field_base + 8)?;
                            work.push(ExprWork::BuildMData);
                            work.push(ExprWork::Parse(expr_ptr));
                        }

                        expr_tags::PROJ => {
                            let type_name_ptr = self.read_u64_at(field_base)?;
                            let idx_ptr = self.read_u64_at(field_base + 8)?;
                            let struct_ptr = self.read_u64_at(field_base + 16)?;
                            let type_name = self.resolve_name_ptr(type_name_ptr)?;
                            let idx = if is_scalar(idx_ptr) {
                                unbox_scalar(idx_ptr)
                            } else if is_ptr(idx_ptr) {
                                self.read_nat_value(idx_ptr).unwrap_or(0)
                            } else {
                                0
                            };
                            work.push(ExprWork::BuildProj(type_name, idx));
                            work.push(ExprWork::Parse(struct_ptr));
                        }

                        _ => {
                            return Err(OleanError::InvalidObjectTag {
                                tag: header.tag,
                                offset,
                            })
                        }
                    }
                }

                ExprWork::BuildApp => {
                    let arg = results.pop().expect("stack balance invariant");
                    let func = results.pop().expect("stack balance invariant");
                    results.push(ParsedExpr::App(Box::new(func), Box::new(arg)));
                }

                ExprWork::BuildLam(name, info) => {
                    let body = results.pop().expect("stack balance invariant");
                    let ty = results.pop().expect("stack balance invariant");
                    results.push(ParsedExpr::Lam(name, Box::new(ty), Box::new(body), info));
                }

                ExprWork::BuildForallE(name, info) => {
                    let body = results.pop().expect("stack balance invariant");
                    let ty = results.pop().expect("stack balance invariant");
                    results.push(ParsedExpr::ForallE(
                        name,
                        Box::new(ty),
                        Box::new(body),
                        info,
                    ));
                }

                ExprWork::BuildLetE(name, nondep) => {
                    let body = results.pop().expect("stack balance invariant");
                    let val = results.pop().expect("stack balance invariant");
                    let ty = results.pop().expect("stack balance invariant");
                    results.push(ParsedExpr::LetE(
                        name,
                        Box::new(ty),
                        Box::new(val),
                        Box::new(body),
                        nondep,
                    ));
                }

                ExprWork::BuildMData => {
                    let inner = results.pop().expect("stack balance invariant");
                    results.push(ParsedExpr::MData(Box::new(inner)));
                }

                ExprWork::BuildProj(name, idx) => {
                    let inner = results.pop().expect("stack balance invariant");
                    results.push(ParsedExpr::Proj(name, idx, Box::new(inner)));
                }
            }
        }

        debug_assert_eq!(results.len(), 1);
        Ok(results.pop().expect("stack balance invariant"))
    }

    /// Resolve a name pointer (helper)
    fn resolve_name_ptr(&self, ptr: u64) -> OleanResult<String> {
        if is_scalar(ptr) {
            // Name.anonymous encoded as scalar 0
            return Ok(String::new());
        }

        if !is_ptr(ptr) {
            return Ok(String::new());
        }

        let offset = self.ptr_to_offset(ptr)?;
        self.read_name_at(offset)
    }

    /// Read a list of levels
    fn read_level_list(&self, ptr: u64) -> OleanResult<Vec<ParsedLevel>> {
        let mut levels = Vec::new();
        let mut current_ptr = ptr;

        // Max iterations to prevent infinite loops
        for _ in 0..100 {
            if is_scalar(current_ptr) {
                // Empty list is often scalar 0 (pointer value 1)
                break;
            }

            if !is_ptr(current_ptr) {
                break;
            }

            let offset = self.ptr_to_offset(current_ptr)?;
            let header = self.read_header_at(offset)?;

            // List has two constructors:
            // - nil (tag 0, 0 fields)
            // - cons (tag 1, 2 fields: head, tail)
            match (header.tag, header.other) {
                (0, 0) => {
                    // nil
                    break;
                }
                (1, 2) => {
                    // cons
                    let head_ptr = self.read_u64_at(offset + 8)?;
                    let tail_ptr = self.read_u64_at(offset + 16)?;

                    let level = self.resolve_level_ptr(head_ptr, 0)?;
                    levels.push(level);

                    current_ptr = tail_ptr;
                }
                _ => {
                    // Unknown list structure
                    break;
                }
            }
        }

        Ok(levels)
    }

    /// Read a Literal (Nat or String)
    fn read_literal(&self, ptr: u64) -> OleanResult<ParsedLiteral> {
        if is_scalar(ptr) {
            // Small Nat encoded as scalar
            return Ok(ParsedLiteral::Nat(unbox_scalar(ptr)));
        }

        if !is_ptr(ptr) {
            return Ok(ParsedLiteral::Nat(0));
        }

        let offset = self.ptr_to_offset(ptr)?;
        let header = self.read_header_at(offset)?;

        // Literal has two constructors:
        // - natVal (tag 0, 1 field: Nat)
        // - strVal (tag 1, 1 field: String)
        match header.tag {
            0 => {
                // natVal
                let nat_ptr = self.read_u64_at(offset + 8)?;
                let val = if is_scalar(nat_ptr) {
                    unbox_scalar(nat_ptr)
                } else {
                    // Big nat - not handled yet
                    0
                };
                Ok(ParsedLiteral::Nat(val))
            }
            1 => {
                // strVal
                let str_ptr = self.read_u64_at(offset + 8)?;
                if is_ptr(str_ptr) {
                    let str_off = self.ptr_to_offset(str_ptr)?;
                    let s = self.read_lean_string_at(str_off)?;
                    Ok(ParsedLiteral::String(s.to_string()))
                } else {
                    Ok(ParsedLiteral::String(String::new()))
                }
            }
            tags::STRING => {
                // Direct string (not wrapped in Literal)
                let s = self.read_lean_string_at(offset)?;
                Ok(ParsedLiteral::String(s.to_string()))
            }
            _ => Err(OleanError::InvalidObjectTag {
                tag: header.tag,
                offset,
            }),
        }
    }

    /// Find expression-like objects in the file (exploratory)
    pub fn find_expr_objects(&self) -> Vec<(usize, u8, u8)> {
        let mut results = Vec::new();

        let mut offset = 64;
        while offset + 8 < self.data.len() {
            if let Ok(header) = self.read_header_at(offset) {
                // Check for Expr tags with expected field counts
                let is_expr = matches!(
                    (header.tag, header.other),
                    (expr_tags::BVAR, 0 | 1)
                        | (
                            expr_tags::FVAR | expr_tags::MVAR | expr_tags::SORT | expr_tags::LIT,
                            1
                        )
                        | (expr_tags::CONST | expr_tags::APP | expr_tags::MDATA, 2)
                        | (expr_tags::LAM | expr_tags::FORALL_E, 3)
                        | (expr_tags::LET_E, 4)
                        | (expr_tags::PROJ, 2 | 3)
                );

                if is_expr {
                    results.push((offset, header.tag, header.other));
                }
            }
            offset += 8;
        }

        results
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
    fn test_parsed_expr_kind() {
        let expr = ParsedExpr::BVar(0);
        assert_eq!(expr.kind(), "bvar");

        let expr = ParsedExpr::Const("Nat".to_string(), vec![]);
        assert_eq!(expr.kind(), "const");
    }

    #[test]
    fn test_find_expr_objects_in_prelude() {
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
        let region = CompactedRegion::new(&bytes, header.base_addr);

        let exprs = region.find_expr_objects();
        println!("Found {} potential Expr objects", exprs.len());

        // Group by tag
        let mut by_tag: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
        for (_, tag, _) in &exprs {
            *by_tag.entry(*tag).or_insert(0) += 1;
        }

        println!("Expression types found:");
        for (tag, count) in &by_tag {
            let name = match *tag {
                expr_tags::BVAR => "bvar",
                expr_tags::FVAR => "fvar",
                expr_tags::MVAR => "mvar",
                expr_tags::SORT => "sort",
                expr_tags::CONST => "const",
                expr_tags::APP => "app",
                expr_tags::LAM => "lam",
                expr_tags::FORALL_E => "forallE",
                expr_tags::LET_E => "letE",
                expr_tags::LIT => "lit",
                expr_tags::MDATA => "mdata",
                expr_tags::PROJ => "proj",
                _ => "unknown",
            };
            println!("  {name}: {count}");
        }

        let mut bvar_shapes: std::collections::HashMap<(u8, u16), usize> =
            std::collections::HashMap::new();
        for (off, tag, _) in &exprs {
            if *tag == expr_tags::BVAR {
                if let Ok(h) = region.read_header_at(*off) {
                    *bvar_shapes.entry((h.other, h.cs_sz)).or_insert(0) += 1;
                }
            }
        }
        let mut shapes: Vec<_> = bvar_shapes.into_iter().collect();
        shapes.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        println!("Top bvar shapes (other, cs_sz):");
        for ((other, cs_sz), count) in shapes.iter().take(5) {
            println!("  ({other}, {cs_sz}) -> {count}");
        }

        if let Some((offset, tag, other)) = exprs.iter().find(|(off, tag, _)| {
            *tag == expr_tags::BVAR
                && region
                    .read_header_at(*off)
                    .map(|h| h.cs_sz > 0)
                    .unwrap_or(false)
        }) {
            if let Ok(header) = region.read_header_at(*offset) {
                if let Ok(bytes) = region.bytes_at(*offset, header.cs_sz as usize) {
                    println!(
                        "Sample bvar offset={}, tag={}, other={}, cs_sz={}, bytes={:x?}",
                        offset, tag, other, header.cs_sz, bytes
                    );
                }
            }
        }

        // We expect to find many expression objects
        assert!(
            exprs.len() > 100,
            "Expected > 100 expr objects, got {}",
            exprs.len()
        );
    }

    #[test]
    fn test_read_sample_exprs() {
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        let prelude_path = lib_path.join("Init/Prelude.olean");
        if !prelude_path.exists() {
            return;
        }

        let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
        let header = crate::parse_header(&bytes).expect("Failed to parse header");
        let region = CompactedRegion::new(&bytes, header.base_addr);

        let exprs = region.find_expr_objects();

        // Try to read the first few expressions of each type
        let mut successes = 0;
        let mut failures = 0;

        for (offset, tag, _) in exprs.iter().take(100) {
            match region.read_expr_at(*offset) {
                Ok(expr) => {
                    successes += 1;
                    if successes <= 10 {
                        println!("offset {}: tag {} -> {:?}", offset, tag, expr.kind());
                    }
                }
                Err(_e) => {
                    failures += 1;
                }
            }
        }

        println!("Read {successes} expressions successfully, {failures} failures");

        // We should be able to read at least some expressions
        assert!(
            successes > 0,
            "Should read at least some expressions, got {successes} successes and {failures} failures"
        );
    }
}
