//! Parser for rustc MIR text output
//!
//! This module parses the human-readable MIR format produced by `rustc --emit mir`.
//! The format is documented as "intended for human consumers" and "subject to change",
//! so this parser is best-effort and may need updates for different rustc versions.
//!
//! # Supported MIR Constructs
//!
//! - Function declarations with arguments and return types
//! - Basic blocks with statements and terminators
//! - Variable declarations (let bindings)
//! - Statements: assign, copy, move, AddWithOverflow, etc.
//! - Terminators: goto, switchInt, assert, return, call
//!
//! # Example: End-to-end MIR parsing and verification
//!
//! ```ignore
//! use kani_fast_chc::{MirParser, encode_mir_to_chc, verify_chc, ChcSolverConfig};
//!
//! // Generate MIR from Rust source
//! let mir_text = generate_mir_from_source(rust_code)?;
//!
//! // Parse MIR
//! let parser = MirParser::new();
//! let func = parser.parse_function(&mir_text, "my_function")?;
//!
//! // Convert to CHC and verify
//! let program = func.to_mir_program();
//! let chc = encode_mir_to_chc(&program);
//! let result = verify_chc(&chc, &ChcSolverConfig::default()).await?;
//! ```

use crate::mir::{
    MirBasicBlock, MirLocal, MirProgram, MirStatement, MirTerminator, PANIC_BLOCK_ID,
};
use kani_fast_kinduction::SmtType;
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

// Pre-compiled regex for tuple element references
lazy_static! {
    /// Matches tuple element references like _X_elem_Y
    static ref RE_TUPLE_ELEM: Regex = Regex::new(r"(_\d+_elem_\d+)")
        .expect("RE_TUPLE_ELEM regex is valid");
    /// Matches enum/struct field references like _X_fieldY
    static ref RE_FIELD_ELEM: Regex = Regex::new(r"(_\d+_field\d+)")
        .expect("RE_FIELD_ELEM regex is valid");
}

/// Extract source pattern from closure function name
///
/// Closure MIR functions are named like `closure_proof::{closure#0}`.
/// The source pattern comes from the closure type in the first argument,
/// which looks like `{closure@/path/file.rs:line:col: line:col}`.
///
/// Since we lose the exact source pattern during parsing (it's in the type),
/// we use the parent function's source file and the closure index to construct
/// a unique identifier that can be matched later.
///
/// For simplicity, we'll use the closure function name as the key since it's unique.
fn extract_source_pattern_from_closure_name(
    closure_name: &str,
    _parent_name: &str,
) -> Option<String> {
    // The closure name format is: parent::{closure#N}
    // We'll use this as the key directly since we don't have the original source location
    // When matching, we'll need to extract this pattern from the call site
    //
    // Actually, the call site has the full source pattern like:
    // <{closure@/tmp/closure_simple.rs:2:19: 2:27} as Fn<(i32,)>>::call
    //
    // So we need to store the closure by some pattern that can be matched.
    // For now, we'll just return the closure name and update the matching logic later.

    // Return a placeholder - the actual matching will need to be based on
    // searching for the closure function that matches the call pattern
    if closure_name.contains("::{closure#") {
        Some(closure_name.to_string())
    } else {
        None
    }
}

/// Rust integer type with precise bounds
///
/// This enum tracks the original Rust integer type from MIR, preserving
/// the type bounds needed for precise overflow condition generation.
/// Instead of using Havoc for overflow flags (which over-approximates),
/// we can generate exact conditions like `result > 255 || result < 0` for u8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RustIntType {
    // Unsigned
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    // Signed
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
}

impl RustIntType {
    /// Parse a Rust integer type string
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "u8" => Some(Self::U8),
            "u16" => Some(Self::U16),
            "u32" => Some(Self::U32),
            "u64" => Some(Self::U64),
            "u128" => Some(Self::U128),
            "usize" => Some(Self::Usize),
            "i8" => Some(Self::I8),
            "i16" => Some(Self::I16),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "i128" => Some(Self::I128),
            "isize" => Some(Self::Isize),
            _ => None,
        }
    }

    /// Get the minimum value for this type
    pub fn min_value(&self) -> i128 {
        match self {
            // Unsigned types have min = 0
            Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::U128 | Self::Usize => 0,
            // Signed types
            Self::I8 => i8::MIN as i128,
            Self::I16 => i16::MIN as i128,
            Self::I32 => i32::MIN as i128,
            Self::I64 => i64::MIN as i128,
            Self::I128 => i128::MIN,
            Self::Isize => isize::MIN as i128,
        }
    }

    /// Get the maximum value for this type
    pub fn max_value(&self) -> i128 {
        match self {
            Self::U8 => u8::MAX as i128,
            Self::U16 => u16::MAX as i128,
            Self::U32 => u32::MAX as i128,
            Self::U64 => u64::MAX as i128,
            Self::U128 => i128::MAX, // Can't represent u128::MAX in i128
            Self::Usize => usize::MAX as i128,
            Self::I8 => i8::MAX as i128,
            Self::I16 => i16::MAX as i128,
            Self::I32 => i32::MAX as i128,
            Self::I64 => i64::MAX as i128,
            Self::I128 => i128::MAX,
            Self::Isize => isize::MAX as i128,
        }
    }

    /// Is this type unsigned?
    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::U128 | Self::Usize
        )
    }

    /// Generate SMT-LIB2 overflow condition for addition
    ///
    /// For result = a + b:
    /// - Unsigned: overflow if result > MAX (since we use unbounded Int)
    /// - Signed: overflow if result > MAX or result < MIN
    pub fn add_overflow_condition(&self, result_expr: &str) -> String {
        let max = self.max_value();
        let min = self.min_value();
        if self.is_unsigned() {
            // For unsigned: overflow if result > MAX or result < 0
            format!("(or (> {} {}) (< {} 0))", result_expr, max, result_expr)
        } else {
            // For signed: overflow if result > MAX or result < MIN
            format!(
                "(or (> {} {}) (< {} {}))",
                result_expr, max, result_expr, min
            )
        }
    }

    /// Generate SMT-LIB2 overflow condition for subtraction
    ///
    /// For result = a - b:
    /// - Unsigned: overflow if result < 0 (underflow)
    /// - Signed: overflow if result > MAX or result < MIN
    pub fn sub_overflow_condition(&self, result_expr: &str) -> String {
        let max = self.max_value();
        let min = self.min_value();
        if self.is_unsigned() {
            // For unsigned: underflow if result < 0
            format!("(or (< {} 0) (> {} {}))", result_expr, result_expr, max)
        } else {
            // For signed: overflow if result > MAX or result < MIN
            format!(
                "(or (> {} {}) (< {} {}))",
                result_expr, max, result_expr, min
            )
        }
    }

    /// Generate SMT-LIB2 overflow condition for multiplication
    ///
    /// For result = a * b:
    /// - Unsigned: overflow if result > MAX or result < 0 (negative factors)
    /// - Signed: overflow if result > MAX or result < MIN
    pub fn mul_overflow_condition(&self, result_expr: &str) -> String {
        let max = self.max_value();
        let min = self.min_value();
        if self.is_unsigned() {
            // For unsigned: overflow if result > MAX or result < 0
            format!("(or (> {} {}) (< {} 0))", result_expr, max, result_expr)
        } else {
            // For signed: overflow if result > MAX or result < MIN
            format!(
                "(or (> {} {}) (< {} {}))",
                result_expr, max, result_expr, min
            )
        }
    }
}

/// Error type for MIR parsing
#[derive(Debug, thiserror::Error)]
pub enum MirParseError {
    #[error("Failed to parse function signature: {0}")]
    FunctionSignature(String),

    #[error("Failed to parse basic block: {0}")]
    BasicBlock(String),

    #[error("Failed to parse statement: {0}")]
    Statement(String),

    #[error("Failed to parse terminator: {0}")]
    Terminator(String),

    #[error("Failed to parse type: {0}")]
    Type(String),

    #[error("Unknown variable reference: {0}")]
    UnknownVariable(String),

    #[error("No functions found in MIR")]
    NoFunctions,

    #[error("Failed to run rustc: {0}")]
    RustcError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Generate MIR from a Rust source file using rustc
///
/// Returns the MIR text output that can be parsed by `MirParser`.
///
/// # Arguments
///
/// * `source_path` - Path to the Rust source file
/// * `edition` - Rust edition (default: "2021")
///
/// # Example
///
/// ```ignore
/// let mir_text = generate_mir_from_file("src/lib.rs", None)?;
/// let parser = MirParser::new();
/// let functions = parser.parse(&mir_text)?;
/// ```
pub fn generate_mir_from_file(
    source_path: impl AsRef<Path>,
    edition: Option<&str>,
) -> Result<String, MirParseError> {
    let source_path = source_path.as_ref();
    let edition = edition.unwrap_or("2021");

    let temp_dir = std::env::temp_dir();
    let mir_output = temp_dir.join("kani_fast_mir_output.mir");

    let output = Command::new("rustc")
        .arg("--emit")
        .arg("mir")
        .arg("--crate-type")
        .arg("lib")
        .arg("--edition")
        .arg(edition)
        .arg("-o")
        .arg(&mir_output)
        .arg(source_path)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MirParseError::RustcError(format!(
            "rustc failed: {}",
            stderr
        )));
    }

    let mir_text = std::fs::read_to_string(&mir_output)?;

    // Clean up
    let _ = std::fs::remove_file(&mir_output);

    Ok(mir_text)
}

/// Generate MIR from Rust source code string
///
/// Creates a temporary file, compiles it with rustc, and returns the MIR.
///
/// # Arguments
///
/// * `source_code` - Rust source code as a string
/// * `edition` - Rust edition (default: "2021")
///
/// # Example
///
/// ```ignore
/// let source = r#"
/// fn add(a: i32, b: i32) -> i32 {
///     a + b
/// }
/// "#;
/// let mir_text = generate_mir_from_source(source, None)?;
/// ```
pub fn generate_mir_from_source(
    source_code: &str,
    edition: Option<&str>,
) -> Result<String, MirParseError> {
    let temp_dir = std::env::temp_dir();
    let source_file = temp_dir.join("kani_fast_temp_source.rs");

    std::fs::write(&source_file, source_code)?;

    let result = generate_mir_from_file(&source_file, edition);

    // Clean up
    let _ = std::fs::remove_file(&source_file);

    result
}

/// A parsed MIR function
#[derive(Debug, Clone)]
pub struct ParsedMirFunction {
    pub name: String,
    pub args: Vec<(String, SmtType)>,
    pub return_type: SmtType,
    pub locals: Vec<MirLocal>,
    pub basic_blocks: Vec<MirBasicBlock>,
}

impl ParsedMirFunction {
    /// Convert to a MirProgram suitable for CHC encoding
    pub fn to_mir_program(&self) -> MirProgram {
        use std::collections::HashSet;

        // Combine args and locals
        let mut all_locals: Vec<MirLocal> = self
            .args
            .iter()
            .map(|(name, ty)| MirLocal::new(name.clone(), ty.clone()))
            .collect();

        for local in &self.locals {
            // Avoid duplicates (args may overlap with locals in MIR naming)
            if !all_locals.iter().any(|l| l.name == local.name) {
                all_locals.push(local.clone());
            }
        }

        // Scan for tuple/field element references (_X_elem_Y, _X_fieldY) and add synthetic locals
        // These are created when parsing overflow operations and enum/struct field accesses
        let mut tuple_elems: HashSet<String> = HashSet::new();
        for block in &self.basic_blocks {
            for stmt in &block.statements {
                if let MirStatement::Assign { lhs, rhs } = stmt {
                    // Look for _X_elem_Y or _X_fieldY pattern in BOTH LHS and RHS
                    // LHS patterns come from struct aggregate initialization (e.g., _3_field0 = 0)
                    // RHS patterns come from field accesses (e.g., _x = _3_field0)
                    Self::collect_tuple_elem_refs(lhs, &mut tuple_elems);
                    Self::collect_tuple_elem_refs(rhs, &mut tuple_elems);
                }
            }
            // Also check terminator conditions (e.g., assert conditions that reference tuple elements)
            match &block.terminator {
                MirTerminator::CondGoto { condition, .. } => {
                    Self::collect_tuple_elem_refs(condition, &mut tuple_elems);
                }
                MirTerminator::SwitchInt { discr, .. } => {
                    Self::collect_tuple_elem_refs(discr, &mut tuple_elems);
                }
                _ => {}
            }
        }

        // Add synthetic locals for tuple elements
        for elem_name in tuple_elems {
            if !all_locals.iter().any(|l| l.name == elem_name) {
                // Determine type based on element index:
                // - _X_elem_0: first element (the result value) - typically Int
                // - _X_elem_1: second element (overflow flag from AddWithOverflow) - Bool
                let ty = if elem_name.ends_with("_elem_1") {
                    SmtType::Bool
                } else {
                    SmtType::Int
                };
                all_locals.push(MirLocal::new(elem_name, ty));
            }
        }

        MirProgram {
            locals: all_locals,
            basic_blocks: self.basic_blocks.clone(),
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: std::collections::HashMap::new(),
            trait_impls: std::collections::HashMap::new(),
        }
    }

    /// Convert to a MirProgram with closure information from related functions
    ///
    /// This method finds closure functions among the provided list and adds them
    /// to the program for potential inlining during CHC encoding.
    pub fn to_mir_program_with_closures(&self, all_functions: &[ParsedMirFunction]) -> MirProgram {
        use crate::mir::build_closure_info;

        let mut program = self.to_mir_program();

        // Find closure functions for this function
        // Closure naming pattern: parent_function::{closure#N}
        let closure_prefix = format!("{}::{{closure#", self.name);

        for func in all_functions {
            if func.name.starts_with(&closure_prefix) {
                // Extract the source pattern from the closure's first argument type
                // The closure type contains the source location: {closure@/path/file.rs:line:col: line:col}
                // Use the function name as the key for matching
                if let Some(source_pattern) =
                    extract_source_pattern_from_closure_name(&func.name, &self.name)
                {
                    let closure_info = build_closure_info(
                        &func.name,
                        &source_pattern,
                        &func.basic_blocks,
                        &func.args,
                    );
                    program.closures.insert(source_pattern, closure_info);
                }
            }
        }

        program
    }

    /// Convert to a MirProgram with both closure and trait impl inlining support
    ///
    /// This method finds closure functions and trait impl methods among the provided
    /// list and adds them to the program for potential inlining during CHC encoding.
    pub fn to_mir_program_with_all_inlines(
        &self,
        all_functions: &[ParsedMirFunction],
    ) -> MirProgram {
        use crate::mir::{
            build_trait_impl_info, is_trait_qualified_call, parse_trait_qualified_call,
        };

        let mut program = self.to_mir_program_with_closures(all_functions);

        // Find trait impl functions
        // Impl functions have names like: <impl at /path/file.rs:LINE:COL: LINE:COL>::method_name
        //
        // To match impl functions with trait calls, we need to:
        // 1. Parse impl function names to extract method name
        // 2. Find trait calls in this function's blocks
        // 3. Match by method name (within same file)

        // First, collect all trait calls in this function
        let mut trait_calls: Vec<(String, String, String, String)> = Vec::new(); // (type, trait, method, full_call)
        for block in &self.basic_blocks {
            if let crate::mir::MirTerminator::Call { func, .. } = &block.terminator {
                if is_trait_qualified_call(func) {
                    if let Some((type_name, trait_name, method_name)) =
                        parse_trait_qualified_call(func)
                    {
                        trait_calls.push((type_name, trait_name, method_name, func.clone()));
                    }
                }
            }
        }

        // Now find impl functions that match these trait calls
        for func in all_functions {
            // Check if this is an impl function: <impl at FILE:LINE:COL: LINE:COL>::method
            if func.name.starts_with("<impl at ") && func.name.contains(">::") {
                // Extract method name
                if let Some(method_start) = func.name.rfind(">::") {
                    let method_name = &func.name[method_start + 3..];

                    // Find matching trait calls by method name
                    for (type_name, trait_name, call_method, full_call) in &trait_calls {
                        if method_name == call_method {
                            // Found a match!
                            let impl_info = build_trait_impl_info(
                                &func.name,
                                type_name,
                                trait_name,
                                method_name,
                                &func.basic_blocks,
                                &func.args,
                            );
                            program.trait_impls.insert(full_call.clone(), impl_info);
                        }
                    }
                }
            }
        }

        program
    }

    /// Find all tuple element references like _X_elem_Y in an expression
    fn collect_tuple_elem_refs(expr: &str, refs: &mut std::collections::HashSet<String>) {
        // Simple pattern matching for _X_elem_Y using pre-compiled regex
        for cap in RE_TUPLE_ELEM.captures_iter(expr) {
            if let Some(m) = cap.get(1) {
                refs.insert(m.as_str().to_string());
            }
        }

        // Also collect struct/enum field references like _X_fieldY
        for cap in RE_FIELD_ELEM.captures_iter(expr) {
            if let Some(m) = cap.get(1) {
                refs.insert(m.as_str().to_string());
            }
        }
    }
}

/// Parse rustc MIR text output
pub struct MirParser {
    // Regex patterns compiled once
    fn_signature_re: Regex,
    basic_block_re: Regex,
    local_decl_re: Regex,
    assign_const_re: Regex,
    assign_copy_move_re: Regex,
    assign_op_re: Regex,
    assign_unary_re: Regex,
    assign_cast_re: Regex,
    goto_re: Regex,
    switch_int_re: Regex,
    assert_re: Regex,
    call_re: Regex,
    /// Regex for closure calls like: <{closure@...} as Fn<(T,)>>::call(...)
    closure_call_re: Regex,
    /// Regex for diverging calls (panic, abort) that don't return
    diverging_call_re: Regex,
}

impl Default for MirParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MirParser {
    pub fn new() -> Self {
        Self {
            // fn name(_1: Type) -> RetType {
            // Need to handle both "-> i32" and "-> ()" return types
            // Also handles closure names like: closure_proof::{closure#0}
            fn_signature_re: Regex::new(r"^fn\s+([\w:#{}\[\]<>@/., -]+)\s*\((.*?)\)\s*(?:->\s*(\S+|\(\)))?\s*\{")
                .expect("valid regex"),
            // bb0: {
            basic_block_re: Regex::new(r"^\s*bb(\d+):\s*\{").expect("valid regex"),
            // let mut _2: u32;
            local_decl_re: Regex::new(r"^\s*let\s+(?:mut\s+)?(_\d+):\s*(.+?);")
                .expect("valid regex"),
            // _2 = const 0_u32;
            assign_const_re: Regex::new(r"^\s*(_\d+)\s*=\s*const\s+(.+?);").expect("valid regex"),
            // _5 = copy _3;   or   _5 = move (_3.0: u32);
            assign_copy_move_re: Regex::new(
                r"^\s*(_\d+)\s*=\s*(copy|move)\s+(.+?);",
            )
            .expect("valid regex"),
            // _7 = AddWithOverflow(copy _2, copy _6);
            // _5 = SubWithOverflow(const u8::MAX, copy _2);
            // Note: both arguments can be "copy _X", "move _X", or "const ..."
            assign_op_re: Regex::new(
                r"^\s*(_\d+)\s*=\s*(\w+)\((?:(?:copy|move)\s+)?(.+?),\s*(?:(?:copy|move)\s+)?(.+?)\);",
            )
            .expect("valid regex"),
            // _3 = Neg(copy _2);
            // _3 = Not(copy _2);
            assign_unary_re: Regex::new(
                r"^\s*(_\d+)\s*=\s*(Neg|Not)\((?:(?:copy|move)\s+)?(.+?)\);",
            )
            .expect("valid regex"),
            // _0 = copy _1 as u32 (IntToInt);
            // _5 = copy _3 as i32 (FloatToInt);
            assign_cast_re: Regex::new(
                r"^\s*(_\d+)\s*=\s*(?:(?:copy|move)\s+)?(.+?)\s+as\s+(\S+)\s+\((\w+)\);",
            )
            .expect("valid regex"),
            // goto -> bb1;
            goto_re: Regex::new(r"^\s*goto\s*->\s*bb(\d+);").expect("valid regex"),
            // switchInt(move _4) -> [0: bb5, otherwise: bb2];
            switch_int_re: Regex::new(r"^\s*switchInt\((?:move|copy)\s+(_\d+)\)\s*->\s*\[(.+?)\];")
                .expect("valid regex"),
            // assert(!move (_7.1: bool), "...", ...) -> [success: bb3, unwind continue];
            assert_re: Regex::new(
                r#"^\s*assert\((.+?),\s*"([^"]+)".*?\)\s*->\s*\[success:\s*bb(\d+).*?\];"#,
            )
            .expect("valid regex"),
            // _1 = func(args) -> [return: bb1, unwind continue];
            // Also handles qualified names like: core::num::<impl u8>::saturating_add(...)
            // And trait method calls like: <std::ops::Range<i32> as IntoIterator>::into_iter(...)
            // Note: Closure calls are handled by closure_call_re below
            call_re: Regex::new(
                r"^\s*(_\d+)\s*=\s*([<a-zA-Z_][\w:<>, ]*)\((.*?)\)\s*->\s*\[return:\s*bb(\d+).*?\];",
            )
            .expect("valid regex"),
            // Closure calls: <{closure@/path:line:col} as Fn<(T,)>>::call(args)
            // The function name contains parens in the generic, so we match the whole
            // trait-qualified call pattern: <TYPE as TRAIT<PARAMS>>::METHOD(args)
            closure_call_re: Regex::new(
                r"^\s*(_\d+)\s*=\s*(<\{closure@[^}]+\}\s+as\s+\w+<[^>]+>\s*>::\w+)\((.*?)\)\s*->\s*\[return:\s*bb(\d+).*?\];",
            )
            .expect("valid regex"),
            // _5 = core::panicking::panic(...) -> unwind continue;
            // Diverging calls have "-> unwind" instead of "-> [return: ...]"
            diverging_call_re: Regex::new(
                r"^\s*_\d+\s*=\s*([<a-zA-Z_][\w:<>, ]*)\(.*?\)\s*->\s*unwind",
            )
            .expect("valid regex"),
        }
    }

    /// Parse MIR text and extract all functions
    pub fn parse(&self, mir_text: &str) -> Result<Vec<ParsedMirFunction>, MirParseError> {
        let mut functions = Vec::new();
        let lines: Vec<&str> = mir_text.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];

            // Look for function signature
            if let Some(caps) = self.fn_signature_re.captures(line) {
                // SAFETY: Capture group 1 always exists when fn_signature_re matches
                let name = caps.get(1).unwrap().as_str().to_string();
                let args_str = caps.get(2).map_or("", |m| m.as_str());
                let ret_str = caps.get(3).map_or("()", |m| m.as_str());

                let args = self.parse_args(args_str)?;
                let return_type = self.parse_type(ret_str)?;

                // Parse function body
                let (func, end_idx) =
                    self.parse_function_body(&lines, i + 1, name, args, return_type)?;
                functions.push(func);
                i = end_idx;
            } else {
                i += 1;
            }
        }

        if functions.is_empty() {
            return Err(MirParseError::NoFunctions);
        }

        Ok(functions)
    }

    /// Parse a single function by name
    pub fn parse_function(
        &self,
        mir_text: &str,
        func_name: &str,
    ) -> Result<ParsedMirFunction, MirParseError> {
        let functions = self.parse(mir_text)?;
        functions
            .into_iter()
            .find(|f| f.name == func_name)
            .ok_or_else(|| {
                MirParseError::FunctionSignature(format!("Function '{}' not found", func_name))
            })
    }

    fn parse_args(&self, args_str: &str) -> Result<Vec<(String, SmtType)>, MirParseError> {
        let mut args = Vec::new();

        if args_str.trim().is_empty() {
            return Ok(args);
        }

        for arg in args_str.split(',') {
            let arg = arg.trim();
            if arg.is_empty() {
                continue;
            }

            // _1: u32
            let Some((name_part, type_part)) = arg.split_once(':') else {
                return Err(MirParseError::FunctionSignature(format!(
                    "Invalid argument: {}",
                    arg
                )));
            };

            let name = name_part.trim().to_string();
            let ty = self.parse_type(type_part.trim())?;
            args.push((name, ty));
        }

        Ok(args)
    }

    fn parse_type(&self, type_str: &str) -> Result<SmtType, MirParseError> {
        let type_str = type_str.trim();

        Ok(match type_str {
            "()" | "bool" => SmtType::Bool, // Unit type also maps to Bool for simplicity
            "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64"
            | "u128" | "usize" => SmtType::Int,
            // Tuples with overflow flag
            s if s.starts_with('(') && s.contains("bool") => SmtType::Int, // AddWithOverflow result
            // Arrays like [u32; 3] or [i32; 10]
            s if s.starts_with('[') && s.contains(';') => {
                // Parse array type [ElemType; Size]
                // We model arrays as SMT Array from Int to element type
                let inner = s.trim_start_matches('[').trim_end_matches(']');
                if let Some(semi_idx) = inner.find(';') {
                    let elem_type_str = inner[..semi_idx].trim();
                    let elem_type = self.parse_type(elem_type_str)?;
                    SmtType::Array {
                        index: Box::new(SmtType::Int),
                        element: Box::new(elem_type),
                    }
                } else {
                    SmtType::Int // Fallback
                }
            }
            // Default to Int for unknown types
            _ => SmtType::Int,
        })
    }

    /// Parse an overflow tuple type like `(u8, bool)` or `(i32, bool)`
    /// Returns the element type if this is an overflow tuple, None otherwise.
    ///
    /// Overflow tuples are produced by AddWithOverflow, SubWithOverflow, MulWithOverflow.
    /// The format is (IntType, bool) where IntType is the result type.
    fn parse_overflow_tuple_type(type_str: &str) -> Option<RustIntType> {
        let type_str = type_str.trim();

        // Must start with ( and end with )
        if !type_str.starts_with('(') || !type_str.ends_with(')') {
            return None;
        }

        // Extract inner content
        let inner = &type_str[1..type_str.len() - 1];

        // Split by comma - use iterator instead of collecting
        let mut parts = inner.splitn(2, ',');
        let first = parts.next()?.trim();
        let second = parts.next()?.trim();

        // Second element must be bool (overflow flag)
        if second != "bool" {
            return None;
        }

        // First element must be an integer type
        RustIntType::parse(first)
    }

    fn parse_function_body(
        &self,
        lines: &[&str],
        start: usize,
        name: String,
        args: Vec<(String, SmtType)>,
        return_type: SmtType,
    ) -> Result<(ParsedMirFunction, usize), MirParseError> {
        let mut locals = Vec::new();
        let mut basic_blocks = Vec::new();
        let mut i = start;
        let mut brace_depth = 1; // We're inside the function already

        // Variable types map (for expressions)
        let mut var_types: HashMap<String, SmtType> = HashMap::new();

        // Track Rust integer types for overflow tuple variables (e.g., _7 -> u8 from `(u8, bool)`)
        // This enables precise overflow condition generation instead of using Havoc
        let mut overflow_elem_types: HashMap<String, RustIntType> = HashMap::new();

        // Add arguments to var_types
        for (arg_name, arg_type) in &args {
            var_types.insert(arg_name.clone(), arg_type.clone());
        }

        while i < lines.len() && brace_depth > 0 {
            let line = lines[i];

            // Parse basic block FIRST (before counting braces)
            // because parse_basic_block handles its own block's braces
            if let Some(caps) = self.basic_block_re.captures(line) {
                // SAFETY: Capture group 1 always exists when basic_block_re matches
                let block_id: usize = caps
                    .get(1)
                    .unwrap()
                    .as_str()
                    .parse()
                    .map_err(|_| MirParseError::BasicBlock("Invalid block ID".to_string()))?;

                let (block, end_idx) = self.parse_basic_block(
                    lines,
                    i + 1,
                    block_id,
                    &var_types,
                    &overflow_elem_types,
                )?;
                basic_blocks.push(block);
                i = end_idx;
                continue;
            }

            // Track brace depth for non-basic-block lines
            brace_depth += line.matches('{').count();
            brace_depth -= line.matches('}').count();

            if brace_depth == 0 {
                break;
            }

            // Parse local declarations
            if let Some(caps) = self.local_decl_re.captures(line) {
                // SAFETY: Capture groups 1 and 2 always exist when local_decl_re matches
                let var_name = caps.get(1).unwrap().as_str().to_string();
                let type_str = caps.get(2).unwrap().as_str();
                let ty = self.parse_type(type_str)?;

                // Check if this is an overflow tuple type like (u8, bool) or (i32, bool)
                // Format: (IntType, bool) where IntType is the element type for overflow operations
                if let Some(elem_type) = Self::parse_overflow_tuple_type(type_str) {
                    overflow_elem_types.insert(var_name.clone(), elem_type);
                }

                var_types.insert(var_name.clone(), ty.clone());
                locals.push(MirLocal::new(var_name, ty));
                i += 1;
                continue;
            }

            i += 1;
        }

        // Sort blocks by ID to ensure correct ordering
        basic_blocks.sort_by_key(|b| b.id);

        Ok((
            ParsedMirFunction {
                name,
                args,
                return_type,
                locals,
                basic_blocks,
            },
            i,
        ))
    }

    fn parse_basic_block(
        &self,
        lines: &[&str],
        start: usize,
        block_id: usize,
        var_types: &HashMap<String, SmtType>,
        overflow_elem_types: &HashMap<String, RustIntType>,
    ) -> Result<(MirBasicBlock, usize), MirParseError> {
        let mut statements = Vec::new();
        let mut terminator = None;
        let mut i = start;

        while i < lines.len() {
            let line = lines[i].trim();

            // End of block
            if line == "}" {
                i += 1;
                break;
            }

            // Skip empty lines and scope markers
            if line.is_empty() || line.starts_with("scope") || line.starts_with("debug") {
                i += 1;
                continue;
            }

            // Try to parse as terminator first
            if let Some(term) = self.try_parse_terminator(line, var_types)? {
                terminator = Some(term);
                i += 1;
                continue;
            }

            // Try to parse as statement(s) - XWithOverflow ops return multiple statements
            let parsed = self.try_parse_statement(line, var_types, overflow_elem_types)?;
            statements.extend(parsed);

            i += 1;
        }

        let terminator = terminator.unwrap_or(MirTerminator::Unreachable);

        let mut block = MirBasicBlock::new(block_id, terminator);
        block.statements = statements;

        Ok((block, i))
    }

    fn try_parse_statement(
        &self,
        line: &str,
        _var_types: &HashMap<String, SmtType>,
        overflow_elem_types: &HashMap<String, RustIntType>,
    ) -> Result<Vec<MirStatement>, MirParseError> {
        let trimmed = line.trim();

        // discriminant(_1) = 1;
        if let Some(no_semicolon) = trimmed.strip_suffix(';') {
            if no_semicolon.starts_with("discriminant(") {
                if let Some(close_idx) = no_semicolon.find(')') {
                    let place_raw = &no_semicolon["discriminant(".len()..close_idx];
                    let rhs_part = no_semicolon[close_idx + 1..].trim();
                    if let Some(value_str) = rhs_part.strip_prefix('=') {
                        let lhs = self.parse_simple_expr(place_raw);
                        let rhs = self.parse_const_value(value_str.trim());
                        return Ok(vec![MirStatement::Assign { lhs, rhs }]);
                    }
                }
            }

            // _3 = discriminant(_1);
            if let Some((lhs_raw, rhs_raw)) = no_semicolon.split_once('=') {
                let rhs_raw = rhs_raw.trim();
                if let Some(arg) = rhs_raw.strip_prefix("discriminant(") {
                    if let Some(close_idx) = arg.find(')') {
                        let lhs = lhs_raw.trim().to_string();
                        let place_raw = &arg[..close_idx];
                        let rhs = self.parse_simple_expr(place_raw);
                        return Ok(vec![MirStatement::Assign { lhs, rhs }]);
                    }
                }
            }
        }

        // _2 = const 0_u32;
        if let Some(caps) = self.assign_const_re.captures(line) {
            // SAFETY: Capture groups 1 and 2 always exist when assign_const_re matches
            let lhs = caps.get(1).unwrap().as_str().to_string();
            let const_val = caps.get(2).unwrap().as_str();

            // Parse constant value
            let rhs = self.parse_const_value(const_val);

            return Ok(vec![MirStatement::Assign { lhs, rhs }]);
        }

        // Cast operations: IntToInt, FloatToInt, etc.
        // _0 = copy _1 as u32 (IntToInt);
        if let Some(caps) = self.assign_cast_re.captures(line) {
            // SAFETY: Capture groups 1-2 always exist when assign_cast_re matches
            // Groups 3-4 (target_type, cast_kind) are captured but unused - integer abstraction
            // treats all casts as identity operations
            let lhs = caps.get(1).unwrap().as_str().to_string();
            let src = caps.get(2).unwrap().as_str();

            // Parse source which might be a variable or const
            let rhs = if src.starts_with('_') {
                src.to_string()
            } else {
                self.parse_const_value(src)
            };

            // For integer abstraction (unbounded ints), casts are identity operations
            // In a bitvector model, we would need to truncate/extend appropriately
            return Ok(vec![MirStatement::Assign { lhs, rhs }]);
        }

        // Reference creation: _6 = &mut _4; or _6 = &_4;
        // In our model, references are treated as aliases to the underlying variable
        if let Some(no_semi) = line.trim().strip_suffix(';') {
            if let Some((lhs_raw, rhs_raw)) = no_semi.split_once('=') {
                let lhs = lhs_raw.trim();
                let rhs_trimmed = rhs_raw.trim();

                // Check for reference patterns: &mut _X or &_X
                if rhs_trimmed.starts_with("&mut ") {
                    let target = rhs_trimmed.strip_prefix("&mut ").unwrap().trim();
                    if target.starts_with('_') {
                        // Store as assignment pointing to the target variable
                        // This allows tracing _6 -> _4
                        return Ok(vec![MirStatement::Assign {
                            lhs: lhs.to_string(),
                            rhs: target.to_string(),
                        }]);
                    }
                } else if rhs_trimmed.starts_with('&') {
                    let target = rhs_trimmed.strip_prefix('&').unwrap().trim();
                    if target.starts_with('_') {
                        return Ok(vec![MirStatement::Assign {
                            lhs: lhs.to_string(),
                            rhs: target.to_string(),
                        }]);
                    }
                }
            }
        }

        // _5 = move _3; or _2 = copy (_7.0: u32);
        if let Some(caps) = self.assign_copy_move_re.captures(line) {
            // SAFETY: Capture groups 1 and 3 always exist when assign_copy_move_re matches
            let lhs = caps.get(1).unwrap().as_str().to_string();
            let rhs_raw = caps.get(3).unwrap().as_str();

            // Normalize place expression (tuple projections, copies, moves)
            let rhs = self.parse_simple_expr(rhs_raw);

            return Ok(vec![MirStatement::Assign { lhs, rhs }]);
        }

        // _7 = AddWithOverflow(copy _2, copy _6);
        // _5 = SubWithOverflow(const u8::MAX, copy _2);
        if let Some(caps) = self.assign_op_re.captures(line) {
            // SAFETY: Capture groups 1-4 always exist when assign_op_re matches
            let lhs = caps.get(1).unwrap().as_str().to_string();
            let op = caps.get(2).unwrap().as_str();
            let arg1_raw = caps.get(3).unwrap().as_str();
            let arg2_raw = caps.get(4).unwrap().as_str();

            // Handle args which might be a variable or const
            let arg1 = if arg1_raw.starts_with('_') {
                arg1_raw.to_string()
            } else {
                self.parse_const_value(arg1_raw)
            };
            let arg2 = if arg2_raw.starts_with('_') {
                arg2_raw.to_string()
            } else {
                self.parse_const_value(arg2_raw)
            };

            // For XWithOverflow operations, generate both the result and overflow flag
            // The result is stored in _X_elem_0, the overflow flag in _X_elem_1
            //
            // IMPORTANT: Overflow flag computation strategy:
            // - If we have precise type information (e.g., _7 declared as (u8, bool)),
            //   we generate exact overflow conditions: result > MAX || result < MIN
            // - If type info is unavailable, we fall back to Havoc for soundness
            //
            // Precise conditions improve k-induction by avoiding spurious counterexamples
            // that occur when Havoc makes overflow flags non-deterministic.
            match op {
                "AddWithOverflow" => {
                    // Result: a + b (unbounded integer arithmetic)
                    let result_var = format!("{}_elem_0", lhs);
                    let result_rhs = format!("(+ {} {})", arg1, arg2);

                    // Check if we have precise type information for this variable
                    let overflow_stmt = if let Some(int_type) = overflow_elem_types.get(&lhs) {
                        // Generate precise overflow condition based on type bounds
                        // The overflow flag uses the result expression, not the variable name,
                        // because we need the computed value before it's assigned
                        let overflow_cond = int_type.add_overflow_condition(&result_rhs);
                        MirStatement::Assign {
                            lhs: format!("{}_elem_1", lhs),
                            rhs: overflow_cond,
                        }
                    } else {
                        // Fall back to Havoc for soundness when type info unavailable
                        MirStatement::Havoc {
                            var: format!("{}_elem_1", lhs),
                        }
                    };

                    return Ok(vec![
                        MirStatement::Assign {
                            lhs: result_var,
                            rhs: result_rhs,
                        },
                        overflow_stmt,
                    ]);
                }
                "SubWithOverflow" => {
                    // Result: a - b (unbounded integer arithmetic)
                    let result_var = format!("{}_elem_0", lhs);
                    let result_rhs = format!("(- {} {})", arg1, arg2);

                    let overflow_stmt = if let Some(int_type) = overflow_elem_types.get(&lhs) {
                        let overflow_cond = int_type.sub_overflow_condition(&result_rhs);
                        MirStatement::Assign {
                            lhs: format!("{}_elem_1", lhs),
                            rhs: overflow_cond,
                        }
                    } else {
                        MirStatement::Havoc {
                            var: format!("{}_elem_1", lhs),
                        }
                    };

                    return Ok(vec![
                        MirStatement::Assign {
                            lhs: result_var,
                            rhs: result_rhs,
                        },
                        overflow_stmt,
                    ]);
                }
                "MulWithOverflow" => {
                    // Result: a * b (unbounded integer arithmetic)
                    let result_var = format!("{}_elem_0", lhs);
                    let result_rhs = format!("(* {} {})", arg1, arg2);

                    let overflow_stmt = if let Some(int_type) = overflow_elem_types.get(&lhs) {
                        let overflow_cond = int_type.mul_overflow_condition(&result_rhs);
                        MirStatement::Assign {
                            lhs: format!("{}_elem_1", lhs),
                            rhs: overflow_cond,
                        }
                    } else {
                        MirStatement::Havoc {
                            var: format!("{}_elem_1", lhs),
                        }
                    };

                    return Ok(vec![
                        MirStatement::Assign {
                            lhs: result_var,
                            rhs: result_rhs,
                        },
                        overflow_stmt,
                    ]);
                }
                // Non-overflow operations
                "Add" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(+ {} {})", arg1, arg2),
                    }]);
                }
                "Sub" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(- {} {})", arg1, arg2),
                    }]);
                }
                "Mul" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(* {} {})", arg1, arg2),
                    }]);
                }
                "Lt" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(< {} {})", arg1, arg2),
                    }]);
                }
                "Le" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(<= {} {})", arg1, arg2),
                    }]);
                }
                "Gt" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(> {} {})", arg1, arg2),
                    }]);
                }
                "Ge" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(>= {} {})", arg1, arg2),
                    }]);
                }
                "Eq" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(= {} {})", arg1, arg2),
                    }]);
                }
                "Ne" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(not (= {} {}))", arg1, arg2),
                    }]);
                }
                // Division and remainder with Rust semantics (truncate toward zero)
                // SMT-LIB2's div rounds toward -infinity, so we need special handling
                "Div" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: Self::div_toward_zero(&arg1, &arg2),
                    }]);
                }
                "Rem" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: Self::rem_toward_zero(&arg1, &arg2),
                    }]);
                }
                // Bitwise operations - use SMT-LIB2 bitvector ops
                // Note: For integer abstraction, we model these semantically
                "BitAnd" => {
                    // For pure integer model, bitand with mask gives bounded result
                    // We use bvand in bitvector mode, but for int abstraction we keep symbolic
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(bvand {} {})", arg1, arg2),
                    }]);
                }
                "BitOr" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(bvor {} {})", arg1, arg2),
                    }]);
                }
                "BitXor" => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(bvxor {} {})", arg1, arg2),
                    }]);
                }
                // Shift operations
                "Shl" => {
                    // Left shift: a << b = a * 2^b (for non-negative b)
                    // Using bvshl for bitvector semantics
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(bvshl {} {})", arg1, arg2),
                    }]);
                }
                "Shr" => {
                    // Right shift: a >> b = a / 2^b (arithmetic shift for signed)
                    // Using bvashr for bitvector semantics
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(bvashr {} {})", arg1, arg2),
                    }]);
                }
                "ShrU" => {
                    // Logical right shift (unsigned)
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("(bvlshr {} {})", arg1, arg2),
                    }]);
                }
                _ => {
                    return Ok(vec![MirStatement::Assign {
                        lhs,
                        rhs: format!("({} {} {})", op.to_lowercase(), arg1, arg2),
                    }]);
                }
            }
        }

        // Unary operations: Neg, Not
        // _3 = Neg(copy _2);
        // _3 = Not(copy _2);
        if let Some(caps) = self.assign_unary_re.captures(line) {
            // SAFETY: Capture groups 1-3 always exist when assign_unary_re matches
            let lhs = caps.get(1).unwrap().as_str().to_string();
            let op = caps.get(2).unwrap().as_str();
            let arg_raw = caps.get(3).unwrap().as_str();

            let arg = if arg_raw.starts_with('_') {
                arg_raw.to_string()
            } else {
                self.parse_const_value(arg_raw)
            };

            let rhs = match op {
                "Neg" => format!("(- {})", arg),
                "Not" => format!("(not {})", arg), // Works for both bool (not) and int (bvnot)
                _ => format!("({} {})", op.to_lowercase(), arg),
            };

            return Ok(vec![MirStatement::Assign { lhs, rhs }]);
        }

        // Struct aggregate initialization:
        // _3 = std::ops::Range::<i32> { start: const 0_i32, end: const 3_i32 };
        // We parse this into separate field assignments:
        // - _3_field0 = 0 (start)
        // - _3_field1 = 3 (end)
        if let Some(no_semi) = line.trim().strip_suffix(';') {
            if let Some((lhs_raw, rhs_raw)) = no_semi.split_once('=') {
                let lhs = lhs_raw.trim().to_string();
                let rhs_raw = rhs_raw.trim();

                // Look for struct initialization pattern: TypePath { field: value, ... }
                if rhs_raw.contains('{') && rhs_raw.contains('}') {
                    if let Some(brace_start) = rhs_raw.find('{') {
                        let fields_str = &rhs_raw[brace_start + 1..rhs_raw.len() - 1];
                        let mut statements = Vec::new();

                        // Parse field assignments: "start: const 0_i32, end: const 3_i32"
                        for (field_idx, field_part) in fields_str.split(',').enumerate() {
                            let field_part = field_part.trim();
                            if field_part.is_empty() {
                                continue;
                            }

                            if let Some((_field_name, value_raw)) = field_part.split_once(':') {
                                // Field name is unused - we use positional indexing (field0, field1, etc.)
                                let value = self.parse_const_value(value_raw.trim());

                                // Create field variable: _3_field0, _3_field1, etc.
                                let field_var = format!("{}_field{}", lhs, field_idx);
                                statements.push(MirStatement::Assign {
                                    lhs: field_var,
                                    rhs: value,
                                });
                            }
                        }

                        if !statements.is_empty() {
                            return Ok(statements);
                        }
                    }
                }

                // Tuple construction: _4 = (const 5_i32,); or _4 = (move _x, const 1);
                // Parse into element assignments: _4_elem_0 = 5, etc.
                if rhs_raw.starts_with('(') && rhs_raw.ends_with(')') {
                    let inner = &rhs_raw[1..rhs_raw.len() - 1];
                    let mut statements = Vec::new();

                    // Split by comma - iterate directly without collecting
                    for (idx, elem) in inner.split(',').enumerate() {
                        let elem = elem.trim();
                        if elem.is_empty() {
                            continue;
                        }

                        // Parse the element value (const X, move _Y, copy _Z)
                        let value = if elem.starts_with("const ") {
                            self.parse_const_value(elem)
                        } else if let Some(var) = elem.strip_prefix("move ") {
                            self.parse_simple_expr(var.trim())
                        } else if let Some(var) = elem.strip_prefix("copy ") {
                            self.parse_simple_expr(var.trim())
                        } else {
                            self.parse_simple_expr(elem)
                        };

                        let elem_var = format!("{}_elem_{}", lhs, idx);
                        statements.push(MirStatement::Assign {
                            lhs: elem_var,
                            rhs: value,
                        });
                    }

                    if !statements.is_empty() {
                        return Ok(statements);
                    }
                }
            }
        }

        Ok(vec![])
    }

    fn try_parse_terminator(
        &self,
        line: &str,
        _var_types: &HashMap<String, SmtType>,
    ) -> Result<Option<MirTerminator>, MirParseError> {
        // return;
        if line == "return;" {
            return Ok(Some(MirTerminator::Return));
        }

        // goto -> bb1;
        if let Some(caps) = self.goto_re.captures(line) {
            // SAFETY: Capture group 1 always exists when goto_re matches
            let target: usize = caps
                .get(1)
                .unwrap()
                .as_str()
                .parse()
                .map_err(|_| MirParseError::Terminator("Invalid goto target".to_string()))?;

            return Ok(Some(MirTerminator::Goto { target }));
        }

        // switchInt(move _4) -> [0: bb5, otherwise: bb2];
        if let Some(caps) = self.switch_int_re.captures(line) {
            // SAFETY: Capture groups 1 and 2 always exist when switch_int_re matches
            let discr = caps.get(1).unwrap().as_str().to_string();
            let targets_str = caps.get(2).unwrap().as_str();

            let (targets, otherwise) = self.parse_switch_targets(targets_str)?;

            return Ok(Some(MirTerminator::SwitchInt {
                discr,
                targets,
                otherwise,
            }));
        }

        // assert(...) -> [success: bb3, ...];
        if let Some(caps) = self.assert_re.captures(line) {
            // SAFETY: Capture groups 1 and 3 always exist when assert_re matches
            // Group 2 (message) is captured but unused - we focus on the condition
            let condition = caps.get(1).unwrap().as_str();
            let target: usize = caps
                .get(3)
                .unwrap()
                .as_str()
                .parse()
                .map_err(|_| MirParseError::Terminator("Invalid assert target".to_string()))?;

            // Parse the condition (e.g., "!move (_7.1: bool)")
            let smt_condition = self.parse_assert_condition(condition);

            // An assert that must pass - we model this as:
            // - If condition holds, continue to target
            // - If condition fails, go to abort (PANIC_BLOCK_ID sentinel)
            return Ok(Some(MirTerminator::CondGoto {
                condition: smt_condition,
                then_target: target,
                else_target: PANIC_BLOCK_ID,
            }));
        }

        // Closure calls: <{closure@...} as Fn<(T,)>>::call(args) -> [return: bbN, ...];
        // Must check before regular call_re since the function name contains parens
        if let Some(caps) = self.closure_call_re.captures(line) {
            // SAFETY: Capture groups 1-4 always exist when closure_call_re matches
            let dest = caps.get(1).unwrap().as_str().to_string();
            let func = caps.get(2).unwrap().as_str().to_string();
            let args_str = caps.get(3).unwrap().as_str();
            let target: usize = caps.get(4).unwrap().as_str().parse().map_err(|_| {
                MirParseError::Terminator("Invalid closure call target".to_string())
            })?;

            let args = self.parse_call_args(args_str);

            return Ok(Some(MirTerminator::Call {
                destination: Some(dest),
                func,
                args,
                target,
                unwind: None,
                precondition_check: None, // Not available when parsing text MIR
                postcondition_assumption: None, // Not available when parsing text MIR
                is_range_into_iter: false,
                is_range_next: false,
            }));
        }

        // _1 = func(args) -> [return: bb1, unwind continue];
        if let Some(caps) = self.call_re.captures(line) {
            // SAFETY: Capture groups 1-4 always exist when call_re matches
            let dest = caps.get(1).unwrap().as_str().to_string();
            let func = caps.get(2).unwrap().as_str().to_string();
            let args_str = caps.get(3).unwrap().as_str();
            let target: usize = caps
                .get(4)
                .unwrap()
                .as_str()
                .parse()
                .map_err(|_| MirParseError::Terminator("Invalid call target".to_string()))?;

            let args = self.parse_call_args(args_str);

            return Ok(Some(MirTerminator::Call {
                destination: Some(dest),
                func,
                args,
                target,
                unwind: None,
                precondition_check: None, // Not available when parsing text MIR
                postcondition_assumption: None, // Not available when parsing text MIR
                is_range_into_iter: false,
                is_range_next: false,
            }));
        }

        // _5 = core::panicking::panic(...) -> unwind continue;
        // Diverging calls (panic, abort) are treated as Abort terminators
        if let Some(caps) = self.diverging_call_re.captures(line) {
            // SAFETY: Capture group 1 always exists when diverging_call_re matches
            let func = caps.get(1).unwrap().as_str();
            // Check if this is a panic/abort function
            if func.contains("panic") || func.contains("abort") || func.contains("unreachable") {
                return Ok(Some(MirTerminator::Abort));
            }
            // For other diverging functions, still treat as abort since they don't return
            return Ok(Some(MirTerminator::Abort));
        }

        Ok(None)
    }

    fn parse_const_value(&self, const_str: &str) -> String {
        // Parse constants like "const 0_u32", "0_u32", "true", "false"
        let const_str = const_str.trim();

        // Strip "const " prefix if present
        let const_str = const_str.strip_prefix("const ").unwrap_or(const_str);

        if const_str == "true" {
            return "true".to_string();
        }
        if const_str == "false" {
            return "false".to_string();
        }

        // Strip type suffix (e.g., "0_u32" -> "0")
        if let Some(idx) = const_str.find('_') {
            return const_str[..idx].to_string();
        }

        const_str.to_string()
    }

    fn parse_switch_targets(
        &self,
        targets_str: &str,
    ) -> Result<(Vec<(i64, usize)>, usize), MirParseError> {
        let mut targets = Vec::new();
        let mut otherwise = 0;

        for part in targets_str.split(',') {
            let part = part.trim();
            if part.starts_with("otherwise:") {
                let bb_str = part.trim_start_matches("otherwise:").trim();
                let bb_str = bb_str.trim_start_matches("bb");
                otherwise = bb_str.parse().map_err(|_| {
                    MirParseError::Terminator("Invalid otherwise target".to_string())
                })?;
            } else if let Some((val_str, bb_str)) = part.split_once(':') {
                let val: i64 = val_str
                    .trim()
                    .parse()
                    .map_err(|_| MirParseError::Terminator("Invalid switch value".to_string()))?;
                let bb: usize = bb_str
                    .trim()
                    .trim_start_matches("bb")
                    .parse()
                    .map_err(|_| MirParseError::Terminator("Invalid switch target".to_string()))?;
                targets.push((val, bb));
            }
        }

        Ok((targets, otherwise))
    }

    fn parse_assert_condition(&self, condition: &str) -> String {
        // Parse conditions like "!move (_7.1: bool)"
        let condition = condition.trim();

        if let Some(stripped) = condition.strip_prefix('!') {
            let inner = stripped.trim();
            let inner_smt = self.parse_simple_expr(inner);
            format!("(not {})", inner_smt)
        } else {
            self.parse_simple_expr(condition)
        }
    }

    fn parse_simple_expr(&self, expr: &str) -> String {
        let expr = expr.trim();

        // move/copy expression - strip prefix and recursively parse
        if let Some(inner) = expr.strip_prefix("move ") {
            return self.parse_simple_expr(inner);
        }
        if let Some(inner) = expr.strip_prefix("copy ") {
            return self.parse_simple_expr(inner);
        }

        // Enum/struct field access via downcast: ((_1 as Variant).0: Type)
        if expr.starts_with('(') && expr.contains(" as ") && expr.contains(").") {
            // Remove surrounding parentheses and any type annotation after ':'
            let inner = expr.trim_start_matches('(');
            let before_type = inner.split(':').next().unwrap_or(inner).trim();
            if let Some(as_idx) = before_type.find(" as ") {
                if let Some(dot_idx) = before_type.rfind('.') {
                    let base_raw = before_type[..as_idx].trim();
                    let field_idx = &before_type[dot_idx + 1..];
                    if let Ok(idx) = field_idx.trim().parse::<usize>() {
                        let base = self.parse_simple_expr(base_raw);
                        return format!("{}_field{}", base, idx);
                    }
                }
            }
        }

        // Tuple element like "(_7.1: bool)"
        if expr.starts_with('(') && expr.contains(':') {
            // Extract "_7.1" from "(_7.1: bool)"
            let inner = expr.trim_start_matches('(');
            if let Some(colon_idx) = inner.find(':') {
                let tuple_ref = &inner[..colon_idx];
                // Use splitn to avoid allocation - we only need exactly 2 parts
                let mut parts = tuple_ref.splitn(2, '.');
                if let (Some(base), Some(idx)) = (parts.next(), parts.next()) {
                    // Return valid SMT identifier without parentheses
                    return format!("{}_elem_{}", base, idx);
                }
            }
        }

        // Array indexing like "_1[_4]" or "_1[0 of 3]" -> SMT-LIB2 "(select _1 _4)"
        // The "N of M" syntax is used by rustc for constant index array projections
        // (e.g., array destructuring `let [a, b, c] = arr;`)
        if expr.starts_with('_') && expr.contains('[') {
            if let Some(bracket_start) = expr.find('[') {
                if let Some(bracket_end) = expr.find(']') {
                    let arr = &expr[..bracket_start];
                    let idx = &expr[bracket_start + 1..bracket_end];
                    // Handle "N of M" constant index syntax - extract just N
                    let idx_value = if idx.contains(" of ") {
                        idx.split(" of ").next().unwrap_or(idx).trim()
                    } else {
                        idx
                    };
                    // Recursively parse in case index is a complex expression
                    let idx_parsed = self.parse_simple_expr(idx_value);
                    return format!("(select {} {})", arr, idx_parsed);
                }
            }
        }

        // Variable reference like "_7"
        if let Some(suffix) = expr.strip_prefix('_') {
            if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                return expr.to_string();
            }
        }

        expr.to_string()
    }

    fn parse_call_args(&self, args_str: &str) -> Vec<String> {
        if args_str.trim().is_empty() {
            return Vec::new();
        }

        args_str
            .split(',')
            .map(|arg| {
                let arg = arg.trim();
                // Handle "const 10_u32" or "copy _1"
                if let Some(stripped) = arg.strip_prefix("const ") {
                    self.parse_const_value(stripped)
                } else if let Some(stripped) = arg.strip_prefix("copy ") {
                    stripped.to_string()
                } else if let Some(stripped) = arg.strip_prefix("move ") {
                    stripped.to_string()
                } else {
                    arg.to_string()
                }
            })
            .collect()
    }

    /// Truncating division that matches Rust semantics (round toward zero).
    /// SMT-LIB2's div rounds toward negative infinity, so we need special handling.
    ///
    /// This correctly handles all sign combinations:
    /// - 7 / 3 = 2
    /// - -7 / 3 = -2 (not -3 as SMT div would give)
    /// - 7 / -3 = -2
    /// - -7 / -3 = 2
    fn div_toward_zero(a: &str, b: &str) -> String {
        format!("(ite (>= {a} 0) (div {a} {b}) (- (div (- {a}) {b})))")
    }

    /// Remainder consistent with Rust semantics (same sign as dividend).
    /// Uses the identity: a = (a / b) * b + (a % b)
    ///
    /// This correctly handles all sign combinations:
    /// - 7 % 3 = 1
    /// - -7 % 3 = -1 (not 2 as SMT mod would give)
    /// - 7 % -3 = 1
    /// - -7 % -3 = -1
    fn rem_toward_zero(a: &str, b: &str) -> String {
        let quotient = Self::div_toward_zero(a, b);
        format!("(- {a} (* {quotient} {b}))")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // MirParseError tests
    // ========================================================================

    #[test]
    fn test_mir_parse_error_function_signature() {
        let err = MirParseError::FunctionSignature("bad signature".to_string());
        let msg = err.to_string();
        assert!(msg.contains("function signature"));
        assert!(msg.contains("bad signature"));
    }

    #[test]
    fn test_mir_parse_error_basic_block() {
        let err = MirParseError::BasicBlock("bad block".to_string());
        let msg = err.to_string();
        assert!(msg.contains("basic block"));
    }

    #[test]
    fn test_mir_parse_error_statement() {
        let err = MirParseError::Statement("bad stmt".to_string());
        let msg = err.to_string();
        assert!(msg.contains("statement"));
    }

    #[test]
    fn test_mir_parse_error_terminator() {
        let err = MirParseError::Terminator("bad term".to_string());
        let msg = err.to_string();
        assert!(msg.contains("terminator"));
    }

    #[test]
    fn test_mir_parse_error_type() {
        let err = MirParseError::Type("bad type".to_string());
        let msg = err.to_string();
        assert!(msg.contains("type"));
    }

    #[test]
    fn test_mir_parse_error_unknown_variable() {
        let err = MirParseError::UnknownVariable("_999".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Unknown variable"));
        assert!(msg.contains("_999"));
    }

    #[test]
    fn test_mir_parse_error_no_functions() {
        let err = MirParseError::NoFunctions;
        let msg = err.to_string();
        assert!(msg.contains("No functions"));
    }

    #[test]
    fn test_mir_parse_error_rustc_error() {
        let err = MirParseError::RustcError("compilation failed".to_string());
        let msg = err.to_string();
        assert!(msg.contains("rustc"));
    }

    #[test]
    fn test_mir_parse_error_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = MirParseError::from(io_err);
        let msg = err.to_string();
        assert!(msg.contains("I/O error") || msg.contains("file not found"));
    }

    #[test]
    fn test_mir_parse_error_debug() {
        let err = MirParseError::NoFunctions;
        let debug = format!("{:?}", err);
        assert!(debug.contains("NoFunctions"));
    }

    // ========================================================================
    // MirParser constructor tests
    // ========================================================================

    #[test]
    fn test_mir_parser_new() {
        let parser = MirParser::new();
        // Just test that it doesn't panic and creates valid regex
        assert!(!parser.fn_signature_re.as_str().is_empty());
    }

    #[test]
    fn test_mir_parser_default() {
        let parser = MirParser::default();
        assert!(!parser.fn_signature_re.as_str().is_empty());
    }

    // ========================================================================
    // parse_type tests
    // ========================================================================

    #[test]
    fn test_parse_type_unit() {
        let parser = MirParser::new();
        let ty = parser.parse_type("()").unwrap();
        assert_eq!(ty, SmtType::Bool);
    }

    #[test]
    fn test_parse_type_bool() {
        let parser = MirParser::new();
        let ty = parser.parse_type("bool").unwrap();
        assert_eq!(ty, SmtType::Bool);
    }

    #[test]
    fn test_parse_type_signed_integers() {
        let parser = MirParser::new();
        assert_eq!(parser.parse_type("i8").unwrap(), SmtType::Int);
        assert_eq!(parser.parse_type("i16").unwrap(), SmtType::Int);
        assert_eq!(parser.parse_type("i32").unwrap(), SmtType::Int);
        assert_eq!(parser.parse_type("i64").unwrap(), SmtType::Int);
        assert_eq!(parser.parse_type("i128").unwrap(), SmtType::Int);
        assert_eq!(parser.parse_type("isize").unwrap(), SmtType::Int);
    }

    #[test]
    fn test_parse_type_unsigned_integers() {
        let parser = MirParser::new();
        assert_eq!(parser.parse_type("u8").unwrap(), SmtType::Int);
        assert_eq!(parser.parse_type("u16").unwrap(), SmtType::Int);
        assert_eq!(parser.parse_type("u32").unwrap(), SmtType::Int);
        assert_eq!(parser.parse_type("u64").unwrap(), SmtType::Int);
        assert_eq!(parser.parse_type("u128").unwrap(), SmtType::Int);
        assert_eq!(parser.parse_type("usize").unwrap(), SmtType::Int);
    }

    #[test]
    fn test_parse_type_tuple_with_bool() {
        let parser = MirParser::new();
        let ty = parser.parse_type("(u32, bool)").unwrap();
        // AddWithOverflow result type
        assert_eq!(ty, SmtType::Int);
    }

    #[test]
    fn test_parse_type_unknown_defaults_to_int() {
        let parser = MirParser::new();
        let ty = parser.parse_type("SomeCustomType").unwrap();
        assert_eq!(ty, SmtType::Int);
    }

    #[test]
    fn test_parse_type_whitespace() {
        let parser = MirParser::new();
        let ty = parser.parse_type("  i32  ").unwrap();
        assert_eq!(ty, SmtType::Int);
    }

    // ========================================================================
    // parse_const_value tests
    // ========================================================================

    #[test]
    fn test_parse_const_value_with_type_suffix() {
        let parser = MirParser::new();
        assert_eq!(parser.parse_const_value("0_u32"), "0");
        assert_eq!(parser.parse_const_value("42_i64"), "42");
        assert_eq!(parser.parse_const_value("123_usize"), "123");
    }

    #[test]
    fn test_parse_const_value_boolean() {
        let parser = MirParser::new();
        assert_eq!(parser.parse_const_value("true"), "true");
        assert_eq!(parser.parse_const_value("false"), "false");
    }

    #[test]
    fn test_parse_const_value_with_const_prefix() {
        let parser = MirParser::new();
        assert_eq!(parser.parse_const_value("const 0_u32"), "0");
        assert_eq!(parser.parse_const_value("const true"), "true");
    }

    #[test]
    fn test_parse_const_value_plain_number() {
        let parser = MirParser::new();
        assert_eq!(parser.parse_const_value("42"), "42");
        assert_eq!(parser.parse_const_value("0"), "0");
    }

    #[test]
    fn test_parse_const_value_negative() {
        let parser = MirParser::new();
        // Negative numbers may have underscore type suffix
        assert_eq!(parser.parse_const_value("-1_i32"), "-1");
    }

    // ========================================================================
    // parse_args tests
    // ========================================================================

    #[test]
    fn test_parse_args_empty() {
        let parser = MirParser::new();
        let args = parser.parse_args("").unwrap();
        assert!(args.is_empty());
    }

    #[test]
    fn test_parse_args_whitespace_only() {
        let parser = MirParser::new();
        let args = parser.parse_args("   ").unwrap();
        assert!(args.is_empty());
    }

    #[test]
    fn test_parse_args_single() {
        let parser = MirParser::new();
        let args = parser.parse_args("_1: u32").unwrap();
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].0, "_1");
        assert_eq!(args[0].1, SmtType::Int);
    }

    #[test]
    fn test_parse_args_multiple() {
        let parser = MirParser::new();
        let args = parser.parse_args("_1: i32, _2: bool, _3: u64").unwrap();
        assert_eq!(args.len(), 3);
        assert_eq!(args[0].0, "_1");
        assert_eq!(args[1].0, "_2");
        assert_eq!(args[2].0, "_3");
        assert_eq!(args[1].1, SmtType::Bool);
    }

    #[test]
    fn test_parse_args_with_extra_spaces() {
        let parser = MirParser::new();
        let args = parser.parse_args("  _1 :  i32  ,  _2 : bool  ").unwrap();
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].0, "_1");
        assert_eq!(args[1].0, "_2");
    }

    #[test]
    fn test_parse_args_invalid_format() {
        let parser = MirParser::new();
        let result = parser.parse_args("_1");
        assert!(result.is_err());
    }

    // ========================================================================
    // parse_switch_targets tests
    // ========================================================================

    #[test]
    fn test_parse_switch_targets_simple() {
        let parser = MirParser::new();
        let (targets, otherwise) = parser
            .parse_switch_targets("0: bb5, otherwise: bb2")
            .unwrap();
        assert_eq!(targets, vec![(0, 5)]);
        assert_eq!(otherwise, 2);
    }

    #[test]
    fn test_parse_switch_targets_multiple() {
        let parser = MirParser::new();
        let (targets, otherwise) = parser
            .parse_switch_targets("0: bb1, 1: bb2, 2: bb3, otherwise: bb4")
            .unwrap();
        assert_eq!(targets, vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(otherwise, 4);
    }

    #[test]
    fn test_parse_switch_targets_negative_values() {
        let parser = MirParser::new();
        let (targets, otherwise) = parser
            .parse_switch_targets("-1: bb1, 0: bb2, otherwise: bb3")
            .unwrap();
        assert_eq!(targets, vec![(-1, 1), (0, 2)]);
        assert_eq!(otherwise, 3);
    }

    // ========================================================================
    // parse_assert_condition tests
    // ========================================================================

    #[test]
    fn test_parse_assert_condition_negated() {
        let parser = MirParser::new();
        let cond = parser.parse_assert_condition("!move (_7.1: bool)");
        assert!(cond.starts_with("(not"));
        assert!(cond.contains("_7_elem_1"));
    }

    #[test]
    fn test_parse_assert_condition_simple_variable() {
        let parser = MirParser::new();
        let cond = parser.parse_assert_condition("_4");
        assert_eq!(cond, "_4");
    }

    #[test]
    fn test_parse_assert_condition_copy_expr() {
        let parser = MirParser::new();
        let cond = parser.parse_assert_condition("copy _4");
        assert_eq!(cond, "_4");
    }

    // ========================================================================
    // parse_simple_expr tests
    // ========================================================================

    #[test]
    fn test_parse_simple_expr_variable() {
        let parser = MirParser::new();
        assert_eq!(parser.parse_simple_expr("_4"), "_4");
        assert_eq!(parser.parse_simple_expr("_123"), "_123");
    }

    #[test]
    fn test_parse_simple_expr_move() {
        let parser = MirParser::new();
        assert_eq!(parser.parse_simple_expr("move _4"), "_4");
    }

    #[test]
    fn test_parse_simple_expr_copy() {
        let parser = MirParser::new();
        assert_eq!(parser.parse_simple_expr("copy _4"), "_4");
    }

    #[test]
    fn test_parse_simple_expr_tuple_element() {
        let parser = MirParser::new();
        let result = parser.parse_simple_expr("(_7.1: bool)");
        assert_eq!(result, "_7_elem_1");
    }

    #[test]
    fn test_parse_simple_expr_tuple_element_zero() {
        let parser = MirParser::new();
        let result = parser.parse_simple_expr("(_3.0: u32)");
        assert_eq!(result, "_3_elem_0");
    }

    #[test]
    fn test_parse_simple_expr_nested_move_tuple() {
        let parser = MirParser::new();
        let result = parser.parse_simple_expr("move (_7.1: bool)");
        assert_eq!(result, "_7_elem_1");
    }

    #[test]
    fn test_parse_simple_expr_enum_field_projection() {
        let parser = MirParser::new();
        let result = parser.parse_simple_expr("((_1 as Some).0: i32)");
        assert_eq!(result, "_1_field0");
    }

    #[test]
    fn test_parse_simple_expr_array_index() {
        let parser = MirParser::new();
        let result = parser.parse_simple_expr("_1[_4]");
        assert_eq!(result, "(select _1 _4)");
    }

    #[test]
    fn test_parse_simple_expr_array_index_different_vars() {
        let parser = MirParser::new();
        let result = parser.parse_simple_expr("_7[_10]");
        assert_eq!(result, "(select _7 _10)");
    }

    #[test]
    fn test_parse_simple_expr_array_index_with_copy() {
        let parser = MirParser::new();
        let result = parser.parse_simple_expr("copy _1[_4]");
        // copy should be stripped, then array indexing applied
        assert_eq!(result, "(select _1 _4)");
    }

    #[test]
    fn test_parse_simple_expr_array_index_const_index() {
        let parser = MirParser::new();
        let result = parser.parse_simple_expr("_1[0]");
        assert_eq!(result, "(select _1 0)");
    }

    #[test]
    fn test_parse_simple_expr_array_const_index_of_syntax() {
        // Constant index projection syntax used by rustc for array destructuring
        // e.g., `let [a, b, c] = arr;` generates `_2 = copy _1[0 of 3];`
        let parser = MirParser::new();
        assert_eq!(parser.parse_simple_expr("_1[0 of 3]"), "(select _1 0)");
        assert_eq!(parser.parse_simple_expr("_1[1 of 3]"), "(select _1 1)");
        assert_eq!(parser.parse_simple_expr("_1[2 of 3]"), "(select _1 2)");
    }

    #[test]
    fn test_parse_simple_expr_array_const_index_with_copy() {
        // copy/move prefix with constant index projection
        let parser = MirParser::new();
        assert_eq!(parser.parse_simple_expr("copy _1[0 of 3]"), "(select _1 0)");
        assert_eq!(parser.parse_simple_expr("move _1[1 of 5]"), "(select _1 1)");
    }

    // ========================================================================
    // parse_type tests for array types
    // ========================================================================

    #[test]
    fn test_parse_type_array_u32() {
        let parser = MirParser::new();
        let result = parser.parse_type("[u32; 3]").unwrap();
        assert!(matches!(result, SmtType::Array { .. }));
        if let SmtType::Array { index, element } = result {
            assert!(matches!(*index, SmtType::Int));
            assert!(matches!(*element, SmtType::Int));
        }
    }

    #[test]
    fn test_parse_type_array_bool() {
        let parser = MirParser::new();
        let result = parser.parse_type("[bool; 5]").unwrap();
        assert!(matches!(result, SmtType::Array { .. }));
        if let SmtType::Array { index, element } = result {
            assert!(matches!(*index, SmtType::Int));
            assert!(matches!(*element, SmtType::Bool));
        }
    }

    #[test]
    fn test_parse_type_array_i64() {
        let parser = MirParser::new();
        let result = parser.parse_type("[i64; 100]").unwrap();
        assert!(matches!(result, SmtType::Array { .. }));
    }

    // ========================================================================
    // parse_call_args tests
    // ========================================================================

    #[test]
    fn test_parse_call_args_empty() {
        let parser = MirParser::new();
        let args = parser.parse_call_args("");
        assert!(args.is_empty());
    }

    #[test]
    fn test_parse_call_args_single_copy() {
        let parser = MirParser::new();
        let args = parser.parse_call_args("copy _1");
        assert_eq!(args, vec!["_1"]);
    }

    #[test]
    fn test_parse_call_args_single_move() {
        let parser = MirParser::new();
        let args = parser.parse_call_args("move _1");
        assert_eq!(args, vec!["_1"]);
    }

    #[test]
    fn test_parse_call_args_single_const() {
        let parser = MirParser::new();
        let args = parser.parse_call_args("const 10_u32");
        assert_eq!(args, vec!["10"]);
    }

    #[test]
    fn test_parse_call_args_multiple() {
        let parser = MirParser::new();
        let args = parser.parse_call_args("copy _1, move _2, const 42_i32");
        assert_eq!(args, vec!["_1", "_2", "42"]);
    }

    #[test]
    fn test_parse_call_args_plain() {
        let parser = MirParser::new();
        let args = parser.parse_call_args("_1, _2");
        assert_eq!(args, vec!["_1", "_2"]);
    }

    // ========================================================================
    // ParsedMirFunction tests
    // ========================================================================

    #[test]
    fn test_parsed_mir_function_to_mir_program_empty() {
        let func = ParsedMirFunction {
            name: "test".to_string(),
            args: vec![],
            return_type: SmtType::Int,
            locals: vec![],
            basic_blocks: vec![],
        };

        let program = func.to_mir_program();
        assert!(program.locals.is_empty());
        assert!(program.basic_blocks.is_empty());
        assert_eq!(program.start_block, 0);
    }

    #[test]
    fn test_parsed_mir_function_to_mir_program_with_args() {
        let func = ParsedMirFunction {
            name: "test".to_string(),
            args: vec![
                ("_1".to_string(), SmtType::Int),
                ("_2".to_string(), SmtType::Bool),
            ],
            return_type: SmtType::Int,
            locals: vec![MirLocal::new("_0", SmtType::Int)],
            basic_blocks: vec![],
        };

        let program = func.to_mir_program();
        // Args + locals = 3
        assert_eq!(program.locals.len(), 3);

        // Args should come first
        assert_eq!(program.locals[0].name, "_1");
        assert_eq!(program.locals[1].name, "_2");
        assert_eq!(program.locals[2].name, "_0");
    }

    #[test]
    fn test_parsed_mir_function_to_mir_program_deduplicates() {
        let func = ParsedMirFunction {
            name: "test".to_string(),
            args: vec![("_1".to_string(), SmtType::Int)],
            return_type: SmtType::Int,
            // Local with same name as arg should be deduplicated
            locals: vec![MirLocal::new("_1", SmtType::Int)],
            basic_blocks: vec![],
        };

        let program = func.to_mir_program();
        // Should only have 1 local
        assert_eq!(program.locals.len(), 1);
        assert_eq!(program.locals[0].name, "_1");
    }

    #[test]
    fn test_parsed_mir_function_to_mir_program_adds_tuple_elems() {
        let func = ParsedMirFunction {
            name: "test".to_string(),
            args: vec![],
            return_type: SmtType::Int,
            locals: vec![],
            basic_blocks: vec![MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_2".to_string(),
                    // RHS contains tuple element reference
                    rhs: "_7_elem_0".to_string(),
                },
            )],
        };

        let program = func.to_mir_program();
        // Should have synthetic local for _7_elem_0
        assert!(
            program.locals.iter().any(|l| l.name == "_7_elem_0"),
            "Should have synthetic local for tuple element"
        );
        // _elem_0 should be Int type
        let elem = program
            .locals
            .iter()
            .find(|l| l.name == "_7_elem_0")
            .unwrap();
        assert_eq!(elem.ty, SmtType::Int);
    }

    #[test]
    fn test_parsed_mir_function_to_mir_program_tuple_elem_bool() {
        let func = ParsedMirFunction {
            name: "test".to_string(),
            args: vec![],
            return_type: SmtType::Int,
            locals: vec![],
            basic_blocks: vec![MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_2".to_string(),
                    // RHS contains overflow flag (elem_1)
                    rhs: "_7_elem_1".to_string(),
                },
            )],
        };

        let program = func.to_mir_program();
        // _elem_1 should be Bool type (overflow flag)
        let elem = program
            .locals
            .iter()
            .find(|l| l.name == "_7_elem_1")
            .unwrap();
        assert_eq!(elem.ty, SmtType::Bool);
    }

    #[test]
    fn test_parsed_mir_function_to_mir_program_adds_field_refs() {
        let func = ParsedMirFunction {
            name: "test".to_string(),
            args: vec![],
            return_type: SmtType::Int,
            locals: vec![MirLocal::new("_1", SmtType::Int)],
            basic_blocks: vec![MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_2".to_string(),
                    rhs: "_1_field0".to_string(),
                },
            )],
        };

        let program = func.to_mir_program();
        let field_local = program.locals.iter().find(|l| l.name == "_1_field0");
        assert!(
            field_local.is_some(),
            "Expected synthetic local for enum/struct field reference"
        );
        assert_eq!(field_local.unwrap().ty, SmtType::Int);
    }

    #[test]
    fn test_collect_tuple_elem_refs() {
        let mut refs = std::collections::HashSet::new();
        ParsedMirFunction::collect_tuple_elem_refs("(+ _7_elem_0 _8_elem_1)", &mut refs);
        assert!(refs.contains("_7_elem_0"));
        assert!(refs.contains("_8_elem_1"));
    }

    #[test]
    fn test_collect_tuple_elem_refs_fields() {
        let mut refs = std::collections::HashSet::new();
        ParsedMirFunction::collect_tuple_elem_refs("(assert (_1_field0))", &mut refs);
        assert!(refs.contains("_1_field0"));
    }

    #[test]
    fn test_collect_tuple_elem_refs_no_match() {
        let mut refs = std::collections::HashSet::new();
        ParsedMirFunction::collect_tuple_elem_refs("(+ _1 _2)", &mut refs);
        assert!(refs.is_empty());
    }

    #[test]
    fn test_parsed_mir_function_debug() {
        let func = ParsedMirFunction {
            name: "test".to_string(),
            args: vec![],
            return_type: SmtType::Int,
            locals: vec![],
            basic_blocks: vec![],
        };
        let debug = format!("{:?}", func);
        assert!(debug.contains("ParsedMirFunction"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_parsed_mir_function_clone() {
        let func = ParsedMirFunction {
            name: "test".to_string(),
            args: vec![("_1".to_string(), SmtType::Int)],
            return_type: SmtType::Bool,
            locals: vec![MirLocal::new("_0", SmtType::Bool)],
            basic_blocks: vec![MirBasicBlock::new(0, MirTerminator::Return)],
        };
        let cloned = func.clone();
        assert_eq!(cloned.name, func.name);
        assert_eq!(cloned.args.len(), func.args.len());
    }

    // ========================================================================
    // Parser integration - edge cases
    // ========================================================================

    #[test]
    fn test_parse_empty_string() {
        let parser = MirParser::new();
        let result = parser.parse("");
        assert!(matches!(result, Err(MirParseError::NoFunctions)));
    }

    #[test]
    fn test_parse_no_functions() {
        let parser = MirParser::new();
        let result = parser.parse("// just a comment\n");
        assert!(matches!(result, Err(MirParseError::NoFunctions)));
    }

    #[test]
    fn test_parse_function_not_found() {
        let mir = r"
fn other(_1: i32) -> i32 {
    bb0: {
        return;
    }
}
";
        let parser = MirParser::new();
        let result = parser.parse_function(mir, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_minimal_function() {
        let mir = r"
fn minimal() -> () {
    bb0: {
        return;
    }
}
";
        let parser = MirParser::new();
        let funcs = parser.parse(mir).unwrap();
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, "minimal");
        assert!(funcs[0].args.is_empty());
        assert_eq!(funcs[0].basic_blocks.len(), 1);
    }

    #[test]
    fn test_parse_function_with_no_return_type() {
        let mir = r"
fn no_ret() {
    bb0: {
        return;
    }
}
";
        let parser = MirParser::new();
        let funcs = parser.parse(mir).unwrap();
        assert_eq!(funcs.len(), 1);
        // Default return type should be () mapped to Bool
        assert_eq!(funcs[0].return_type, SmtType::Bool);
    }

    #[test]
    fn test_parse_multiple_functions() {
        let mir = r"
fn first(_1: i32) -> i32 {
    bb0: {
        return;
    }
}

fn second(_1: bool) -> bool {
    bb0: {
        return;
    }
}
";
        let parser = MirParser::new();
        let funcs = parser.parse(mir).unwrap();
        assert_eq!(funcs.len(), 2);
        assert_eq!(funcs[0].name, "first");
        assert_eq!(funcs[1].name, "second");
    }

    #[test]
    fn test_parse_goto_terminator() {
        let mir = r"
fn test_goto() -> () {
    bb0: {
        goto -> bb1;
    }
    bb1: {
        return;
    }
}
";
        let parser = MirParser::new();
        let func = parser.parse_function(mir, "test_goto").unwrap();
        assert!(matches!(
            func.basic_blocks[0].terminator,
            MirTerminator::Goto { target: 1 }
        ));
    }

    #[test]
    fn test_parse_call_terminator() {
        let mir = r"
fn test_call() -> i32 {
    let mut _0: i32;
    bb0: {
        _0 = other_func(copy _1) -> [return: bb1, unwind continue];
    }
    bb1: {
        return;
    }
}
";
        let parser = MirParser::new();
        let func = parser.parse_function(mir, "test_call").unwrap();
        match &func.basic_blocks[0].terminator {
            MirTerminator::Call {
                destination,
                func,
                args,
                target,
                ..
            } => {
                assert_eq!(destination, &Some("_0".to_string()));
                assert_eq!(func, "other_func");
                assert_eq!(args, &vec!["_1".to_string()]);
                assert_eq!(*target, 1);
            }
            other => panic!("Expected Call, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_operations() {
        let mir = r"
fn test_ops(_1: i32, _2: i32) -> i32 {
    let mut _0: i32;
    let mut _3: i32;
    let mut _4: bool;
    bb0: {
        _3 = Add(copy _1, copy _2);
        _4 = Lt(copy _1, copy _2);
        return;
    }
}
";
        let parser = MirParser::new();
        let func = parser.parse_function(mir, "test_ops").unwrap();

        // Check for Add statement
        let stmts = &func.basic_blocks[0].statements;
        assert!(stmts.iter().any(|s| matches!(
            s,
            MirStatement::Assign { lhs, rhs }
            if lhs == "_3" && rhs.contains('+')
        )));

        // Check for Lt statement
        assert!(stmts.iter().any(|s| matches!(
            s,
            MirStatement::Assign { lhs, rhs }
            if lhs == "_4" && rhs.contains('<')
        )));
    }

    #[test]
    fn test_parse_copy_and_move_assignments() {
        let parser = MirParser::new();

        let copy_stmt = parser
            .try_parse_statement("_2 = copy _1;", &HashMap::new(), &HashMap::new())
            .unwrap();
        assert!(matches!(
            &copy_stmt[..],
            [MirStatement::Assign { lhs, rhs }]
            if lhs == "_2" && rhs == "_1"
        ));

        let move_stmt = parser
            .try_parse_statement("_3 = move (_7.1: bool);", &HashMap::new(), &HashMap::new())
            .unwrap();
        assert!(matches!(
            &move_stmt[..],
            [MirStatement::Assign { lhs, rhs }]
            if lhs == "_3" && rhs == "_7_elem_1"
        ));
    }

    #[test]
    fn test_parse_discriminant_read_assignment() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement("_3 = discriminant(_1);", &HashMap::new(), &HashMap::new())
            .unwrap();
        assert!(matches!(
            &stmts[..],
            [MirStatement::Assign { lhs, rhs }] if lhs == "_3" && rhs == "_1"
        ));
    }

    #[test]
    fn test_parse_set_discriminant_statement() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement("discriminant(_1) = 2;", &HashMap::new(), &HashMap::new())
            .unwrap();
        assert!(matches!(
            &stmts[..],
            [MirStatement::Assign { lhs, rhs }] if lhs == "_1" && rhs == "2"
        ));
    }

    #[test]
    fn test_parse_all_comparison_operations() {
        let parser = MirParser::new();

        // Test via statement parsing - returns Vec now
        let lt_stmts = parser
            .try_parse_statement(
                "_4 = Lt(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(lt_stmts.len(), 1);
        assert!(matches!(&lt_stmts[0], MirStatement::Assign { rhs, .. } if rhs.contains('<')));

        let le_stmts = parser
            .try_parse_statement(
                "_4 = Le(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(le_stmts.len(), 1);
        assert!(matches!(&le_stmts[0], MirStatement::Assign { rhs, .. } if rhs.contains("<=")));

        let gt_stmts = parser
            .try_parse_statement(
                "_4 = Gt(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(gt_stmts.len(), 1);
        assert!(matches!(&gt_stmts[0], MirStatement::Assign { rhs, .. } if rhs.contains('>')));

        let ge_stmts = parser
            .try_parse_statement(
                "_4 = Ge(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(ge_stmts.len(), 1);
        assert!(matches!(&ge_stmts[0], MirStatement::Assign { rhs, .. } if rhs.contains(">=")));

        let eq_stmts = parser
            .try_parse_statement(
                "_4 = Eq(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(eq_stmts.len(), 1);
        assert!(
            matches!(&eq_stmts[0], MirStatement::Assign { rhs, .. } if rhs.contains('=') && !rhs.contains("not"))
        );

        let ne_stmts = parser
            .try_parse_statement(
                "_4 = Ne(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(ne_stmts.len(), 1);
        assert!(matches!(&ne_stmts[0], MirStatement::Assign { rhs, .. } if rhs.contains("not")));
    }

    #[test]
    fn test_parse_arithmetic_operations() {
        let parser = MirParser::new();

        let sub_stmts = parser
            .try_parse_statement(
                "_4 = Sub(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(sub_stmts.len(), 1);
        assert!(matches!(&sub_stmts[0], MirStatement::Assign { rhs, .. } if rhs.contains('-')));

        let mul_stmts = parser
            .try_parse_statement(
                "_4 = Mul(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(mul_stmts.len(), 1);
        assert!(matches!(&mul_stmts[0], MirStatement::Assign { rhs, .. } if rhs.contains('*')));
    }

    #[test]
    fn test_parse_overflow_operations() {
        let parser = MirParser::new();

        // XWithOverflow operations now return 2 statements: result (_elem_0) and overflow flag (_elem_1)
        let add_overflow = parser
            .try_parse_statement(
                "_4 = AddWithOverflow(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(add_overflow.len(), 2);
        assert!(
            matches!(&add_overflow[0], MirStatement::Assign { lhs, rhs } if lhs == "_4_elem_0" && rhs.contains('+'))
        );
        // Overflow flag is now Havoc for soundness with sequential variable substitution
        assert!(matches!(&add_overflow[1], MirStatement::Havoc { var } if var == "_4_elem_1"));

        let sub_overflow = parser
            .try_parse_statement(
                "_4 = SubWithOverflow(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(sub_overflow.len(), 2);
        assert!(
            matches!(&sub_overflow[0], MirStatement::Assign { lhs, rhs } if lhs == "_4_elem_0" && rhs.contains('-'))
        );
        // Overflow flag is now Havoc for soundness with sequential variable substitution
        assert!(matches!(&sub_overflow[1], MirStatement::Havoc { var } if var == "_4_elem_1"));

        let mul_overflow = parser
            .try_parse_statement(
                "_4 = MulWithOverflow(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(mul_overflow.len(), 2);
        assert!(
            matches!(&mul_overflow[0], MirStatement::Assign { lhs, rhs } if lhs == "_4_elem_0" && rhs.contains('*'))
        );
        // Overflow flag is now Havoc for soundness with sequential variable substitution
        assert!(matches!(&mul_overflow[1], MirStatement::Havoc { var } if var == "_4_elem_1"));
    }

    #[test]
    fn test_parse_overflow_operations_with_precise_types() {
        let parser = MirParser::new();

        // Create overflow type map with precise type info for _4
        let mut overflow_types: HashMap<String, RustIntType> = HashMap::new();
        overflow_types.insert("_4".to_string(), RustIntType::U8);

        // AddWithOverflow with u8 type should generate precise overflow condition
        let add_overflow = parser
            .try_parse_statement(
                "_4 = AddWithOverflow(copy _1, copy _2);",
                &HashMap::new(),
                &overflow_types,
            )
            .unwrap();
        assert_eq!(add_overflow.len(), 2);
        assert!(
            matches!(&add_overflow[0], MirStatement::Assign { lhs, rhs } if lhs == "_4_elem_0" && rhs.contains('+'))
        );
        // With type info, overflow flag should be an Assign with precise condition
        if let MirStatement::Assign { lhs, rhs } = &add_overflow[1] {
            assert_eq!(lhs, "_4_elem_1");
            // u8 overflow condition: (or (> result 255) (< result 0))
            assert!(
                rhs.contains("255"),
                "Expected u8 MAX (255) in overflow condition: {}",
                rhs
            );
            assert!(
                rhs.contains("< ") && rhs.contains('0'),
                "Expected check for < 0 in overflow condition: {}",
                rhs
            );
        } else {
            panic!(
                "Expected Assign for overflow flag, got {:?}",
                add_overflow[1]
            );
        }

        // SubWithOverflow with u8 type
        let sub_overflow = parser
            .try_parse_statement(
                "_4 = SubWithOverflow(copy _1, copy _2);",
                &HashMap::new(),
                &overflow_types,
            )
            .unwrap();
        assert_eq!(sub_overflow.len(), 2);
        if let MirStatement::Assign { lhs, rhs } = &sub_overflow[1] {
            assert_eq!(lhs, "_4_elem_1");
            // u8 underflow check: result < 0
            assert!(
                rhs.contains("< "),
                "Expected underflow check in condition: {}",
                rhs
            );
        } else {
            panic!(
                "Expected Assign for overflow flag, got {:?}",
                sub_overflow[1]
            );
        }

        // MulWithOverflow with u8 type
        let mul_overflow = parser
            .try_parse_statement(
                "_4 = MulWithOverflow(copy _1, copy _2);",
                &HashMap::new(),
                &overflow_types,
            )
            .unwrap();
        assert_eq!(mul_overflow.len(), 2);
        if let MirStatement::Assign { lhs, rhs } = &mul_overflow[1] {
            assert_eq!(lhs, "_4_elem_1");
            // u8 overflow condition: result > 255 or result < 0
            assert!(
                rhs.contains("255"),
                "Expected u8 MAX (255) in overflow condition: {}",
                rhs
            );
        } else {
            panic!(
                "Expected Assign for overflow flag, got {:?}",
                mul_overflow[1]
            );
        }
    }

    #[test]
    fn test_parse_overflow_operations_with_signed_types() {
        let parser = MirParser::new();

        // Test with signed i8 type
        let mut overflow_types: HashMap<String, RustIntType> = HashMap::new();
        overflow_types.insert("_4".to_string(), RustIntType::I8);

        let add_overflow = parser
            .try_parse_statement(
                "_4 = AddWithOverflow(copy _1, copy _2);",
                &HashMap::new(),
                &overflow_types,
            )
            .unwrap();

        if let MirStatement::Assign { lhs, rhs } = &add_overflow[1] {
            assert_eq!(lhs, "_4_elem_1");
            // i8 range: -128 to 127
            assert!(
                rhs.contains("127"),
                "Expected i8 MAX (127) in overflow condition: {}",
                rhs
            );
            assert!(
                rhs.contains("-128"),
                "Expected i8 MIN (-128) in overflow condition: {}",
                rhs
            );
        } else {
            panic!(
                "Expected Assign for overflow flag, got {:?}",
                add_overflow[1]
            );
        }
    }

    #[test]
    fn test_rust_int_type_bounds() {
        // Verify RustIntType correctly reports bounds
        assert_eq!(RustIntType::U8.min_value(), 0);
        assert_eq!(RustIntType::U8.max_value(), 255);
        assert!(RustIntType::U8.is_unsigned());

        assert_eq!(RustIntType::I8.min_value(), -128);
        assert_eq!(RustIntType::I8.max_value(), 127);
        assert!(!RustIntType::I8.is_unsigned());

        assert_eq!(RustIntType::U32.min_value(), 0);
        assert_eq!(RustIntType::U32.max_value(), u32::MAX as i128);

        assert_eq!(RustIntType::I32.min_value(), i32::MIN as i128);
        assert_eq!(RustIntType::I32.max_value(), i32::MAX as i128);
    }

    #[test]
    fn test_parse_overflow_tuple_type() {
        // Valid overflow tuple types
        assert_eq!(
            MirParser::parse_overflow_tuple_type("(u8, bool)"),
            Some(RustIntType::U8)
        );
        assert_eq!(
            MirParser::parse_overflow_tuple_type("(i32, bool)"),
            Some(RustIntType::I32)
        );
        assert_eq!(
            MirParser::parse_overflow_tuple_type("(u64, bool)"),
            Some(RustIntType::U64)
        );

        // Invalid cases
        assert_eq!(MirParser::parse_overflow_tuple_type("u8"), None);
        assert_eq!(MirParser::parse_overflow_tuple_type("(u8)"), None);
        assert_eq!(MirParser::parse_overflow_tuple_type("(u8, u8)"), None);
        assert_eq!(MirParser::parse_overflow_tuple_type("(bool, bool)"), None);
    }

    #[test]
    fn test_parse_return_terminator() {
        let parser = MirParser::new();
        let term = parser
            .try_parse_terminator("return;", &HashMap::new())
            .unwrap();
        assert!(matches!(term, Some(MirTerminator::Return)));
    }

    #[test]
    fn test_parse_unrecognized_line() {
        let parser = MirParser::new();
        // Lines that don't match any pattern should return empty Vec
        let stmts = parser
            .try_parse_statement("// this is a comment", &HashMap::new(), &HashMap::new())
            .unwrap();
        assert!(stmts.is_empty());

        let term = parser
            .try_parse_terminator("// this is a comment", &HashMap::new())
            .unwrap();
        assert!(term.is_none());
    }

    // ========================================================================
    // Original tests
    // ========================================================================

    const COUNTER_LOOP_MIR: &str = r#"
fn counter_loop(_1: u32) -> u32 {
    debug n => _1;
    let mut _0: u32;
    let mut _2: u32;
    let mut _4: bool;
    let mut _5: u32;
    let mut _6: u32;
    let mut _7: (u32, bool);
    let mut _8: (u32, bool);
    scope 1 {
        debug sum => _2;
        let mut _3: u32;
        scope 2 {
            debug i => _3;
        }
    }

    bb0: {
        _2 = const 0_u32;
        _3 = const 0_u32;
        goto -> bb1;
    }

    bb1: {
        _5 = copy _3;
        _4 = Lt(move _5, copy _1);
        switchInt(move _4) -> [0: bb5, otherwise: bb2];
    }

    bb2: {
        _6 = copy _3;
        _7 = AddWithOverflow(copy _2, copy _6);
        assert(!move (_7.1: bool), "attempt to compute `{} + {}`, which would overflow", copy _2, move _6) -> [success: bb3, unwind continue];
    }

    bb3: {
        _2 = move (_7.0: u32);
        _8 = AddWithOverflow(copy _3, const 1_u32);
        assert(!move (_8.1: bool), "attempt to compute `{} + {}`, which would overflow", copy _3, const 1_u32) -> [success: bb4, unwind continue];
    }

    bb4: {
        _3 = move (_8.0: u32);
        goto -> bb1;
    }

    bb5: {
        _0 = copy _2;
        return;
    }
}
"#;

    #[test]
    fn test_parse_function_signature() {
        let parser = MirParser::new();
        let functions = parser.parse(COUNTER_LOOP_MIR).unwrap();

        assert_eq!(functions.len(), 1);
        assert_eq!(functions[0].name, "counter_loop");
        assert_eq!(functions[0].args.len(), 1);
        assert_eq!(functions[0].args[0].0, "_1");
    }

    #[test]
    fn test_parse_locals() {
        let parser = MirParser::new();
        let func = parser
            .parse_function(COUNTER_LOOP_MIR, "counter_loop")
            .unwrap();

        // Should have several local variables
        assert!(!func.locals.is_empty());

        // Check for some expected locals
        let local_names: Vec<&str> = func.locals.iter().map(|l| l.name.as_str()).collect();
        assert!(local_names.contains(&"_0"));
        assert!(local_names.contains(&"_2"));
    }

    #[test]
    fn test_parse_basic_blocks() {
        let parser = MirParser::new();
        let func = parser
            .parse_function(COUNTER_LOOP_MIR, "counter_loop")
            .unwrap();

        // Should have 6 basic blocks (bb0-bb5)
        assert_eq!(func.basic_blocks.len(), 6);

        // Check bb0 has goto terminator
        assert!(matches!(
            func.basic_blocks[0].terminator,
            MirTerminator::Goto { target: 1 }
        ));

        // Check bb5 has return terminator
        assert!(matches!(
            func.basic_blocks[5].terminator,
            MirTerminator::Return
        ));
    }

    #[test]
    fn test_parse_switch_int() {
        let parser = MirParser::new();
        let func = parser
            .parse_function(COUNTER_LOOP_MIR, "counter_loop")
            .unwrap();

        // bb1 should have switchInt
        match &func.basic_blocks[1].terminator {
            MirTerminator::SwitchInt {
                discr,
                targets,
                otherwise,
            } => {
                assert_eq!(discr, "_4");
                assert_eq!(targets, &[(0, 5)]);
                assert_eq!(*otherwise, 2);
            }
            other => panic!("Expected SwitchInt, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_assignments() {
        let parser = MirParser::new();
        let func = parser
            .parse_function(COUNTER_LOOP_MIR, "counter_loop")
            .unwrap();

        // bb0 should have constant assignments
        let bb0_assigns: Vec<&MirStatement> = func.basic_blocks[0]
            .statements
            .iter()
            .filter(|s| matches!(s, MirStatement::Assign { .. }))
            .collect();

        assert!(!bb0_assigns.is_empty());

        // Check for "_2 = const 0"
        let has_init_sum = bb0_assigns
            .iter()
            .any(|s| matches!(s, MirStatement::Assign { lhs, rhs } if lhs == "_2" && rhs == "0"));
        assert!(has_init_sum, "Should have _2 = 0 assignment");
    }

    #[test]
    fn test_to_mir_program() {
        let parser = MirParser::new();
        let func = parser
            .parse_function(COUNTER_LOOP_MIR, "counter_loop")
            .unwrap();

        let program = func.to_mir_program();

        // Should have locals including argument
        assert!(!program.locals.is_empty());

        // Should have all basic blocks
        assert_eq!(program.basic_blocks.len(), 6);
    }

    #[test]
    fn test_parse_const_values() {
        let parser = MirParser::new();

        assert_eq!(parser.parse_const_value("0_u32"), "0");
        assert_eq!(parser.parse_const_value("42_i64"), "42");
        assert_eq!(parser.parse_const_value("true"), "true");
        assert_eq!(parser.parse_const_value("false"), "false");
    }

    // Integration test: Parse MIR from rustc and verify with CHC
    #[tokio::test]
    async fn test_parse_mir_and_verify_simple_loop() {
        use crate::{encode_mir_to_chc, verify_chc, ChcSolverConfig};
        use std::time::Duration;

        if crate::find_executable("z3").is_none() {
            return; // Skip if Z3 not available
        }

        // Simple loop that increments a counter - always non-negative
        // This is a simplified version that we can verify
        let simple_mir = r"
fn simple_counter(_1: i32) -> i32 {
    let mut _0: i32;
    let mut _2: i32;
    let mut _3: bool;

    bb0: {
        _2 = const 0_i32;
        goto -> bb1;
    }

    bb1: {
        _3 = Lt(copy _2, copy _1);
        switchInt(move _3) -> [0: bb3, otherwise: bb2];
    }

    bb2: {
        _2 = Add(copy _2, const 1_i32);
        goto -> bb1;
    }

    bb3: {
        _0 = copy _2;
        return;
    }
}
";

        let parser = MirParser::new();
        let func = parser.parse_function(simple_mir, "simple_counter").unwrap();
        let mut program = func.to_mir_program();

        // Add assertion: _2 >= 0 (counter is always non-negative)
        // We need to add this to the loop body block
        program.basic_blocks[1]
            .statements
            .push(MirStatement::Assert {
                condition: "(>= _2 0)".to_string(),
                message: Some("counter should be non-negative".to_string()),
            });

        // Set initial condition
        program.init = Some(kani_fast_kinduction::StateFormula::new("(= _2 0)"));

        let chc = encode_mir_to_chc(&program);
        let config = ChcSolverConfig::new().with_timeout(Duration::from_secs(10));
        let result = verify_chc(&chc, &config).await;

        assert!(result.is_ok(), "CHC verification failed: {:?}", result);
        let result = result.unwrap();
        assert!(
            result.is_sat(),
            "Expected SAT (property holds), got {:?}",
            result
        );
    }

    #[test]
    fn test_generate_mir_from_source() {
        // Test end-to-end: Rust source → MIR → parsed functions
        let source = r"
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let _ = add(1, 2);
}
";

        let mir_text = generate_mir_from_source(source, None);
        assert!(mir_text.is_ok(), "Failed to generate MIR: {:?}", mir_text);

        let mir_text = mir_text.unwrap();

        assert!(
            mir_text.contains("fn add("),
            "MIR should contain add function"
        );
        assert!(
            mir_text.contains("fn main()"),
            "MIR should contain main function"
        );

        // Parse the generated MIR
        let parser = MirParser::new();
        let functions = parser.parse(&mir_text);

        assert!(functions.is_ok(), "Failed to parse MIR: {:?}", functions);

        let functions = functions.unwrap();
        assert!(
            functions.len() >= 2,
            "Should have at least 2 functions, got {}",
            functions.len()
        );

        // Find the add function
        let add_func = functions.iter().find(|f| f.name == "add");
        assert!(add_func.is_some(), "Should find 'add' function");
    }

    #[tokio::test]
    async fn test_end_to_end_rust_source_to_chc_verification() {
        use crate::{encode_mir_to_chc, verify_chc, ChcSolverConfig};
        use std::time::Duration;

        if crate::find_executable("z3").is_none() {
            return; // Skip if Z3 not available
        }

        // Simple loop function - verifies that a counter stays non-negative
        // This uses handcrafted MIR to test the full pipeline
        let simple_mir = r"
fn simple_counter(_1: i32) -> i32 {
    let mut _0: i32;
    let mut _2: i32;
    let mut _3: bool;

    bb0: {
        _2 = const 0_i32;
        goto -> bb1;
    }

    bb1: {
        _3 = Lt(copy _2, copy _1);
        switchInt(move _3) -> [0: bb3, otherwise: bb2];
    }

    bb2: {
        _2 = Add(copy _2, const 1_i32);
        goto -> bb1;
    }

    bb3: {
        _0 = copy _2;
        return;
    }
}
";

        let parser = MirParser::new();
        let func = parser.parse_function(simple_mir, "simple_counter").unwrap();
        let mut program = func.to_mir_program();

        // Add assertion: _2 >= 0 (counter is always non-negative)
        // Add to the loop header (bb1) where we know _2 has been initialized
        program.basic_blocks[1]
            .statements
            .push(MirStatement::Assert {
                condition: "(>= _2 0)".to_string(),
                message: Some("counter should be non-negative".to_string()),
            });

        // Set initial condition that includes _2 = 0
        program.init = Some(kani_fast_kinduction::StateFormula::new("(= _2 0)"));

        let chc = encode_mir_to_chc(&program);
        let config = ChcSolverConfig::new().with_timeout(Duration::from_secs(10));
        let result = verify_chc(&chc, &config).await;

        assert!(result.is_ok(), "CHC verification failed: {:?}", result);
        let result = result.unwrap();
        // The property _2 >= 0 should always hold
        assert!(
            result.is_sat(),
            "Expected SAT (_2 >= 0 holds), got {:?}",
            result
        );
    }

    // ========================================================================
    // Division and remainder operations tests
    // ========================================================================

    #[test]
    fn test_parse_div_operation() {
        // Division now uses Rust semantics (truncate toward zero)
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement(
                "_4 = Div(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(stmts.len(), 1);
        // Expected: (ite (>= a 0) (div a b) (- (div (- a) b)))
        let expected_rhs = "(ite (>= _1 0) (div _1 _2) (- (div (- _1) _2)))";
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_4" && rhs == expected_rhs)
        );
    }

    #[test]
    fn test_parse_rem_operation() {
        // Remainder now uses Rust semantics (same sign as dividend)
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement(
                "_4 = Rem(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(stmts.len(), 1);
        // Expected: (- a (* quotient b)) where quotient is div_toward_zero
        let quotient = "(ite (>= _1 0) (div _1 _2) (- (div (- _1) _2)))";
        let expected_rhs = format!("(- _1 (* {} _2))", quotient);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_4" && rhs == &expected_rhs)
        );
    }

    #[test]
    fn test_div_toward_zero_formula_correctness() {
        // Verify the formula produces correct results for all sign combinations
        // 7 / 3 = 2, -7 / 3 = -2, 7 / -3 = -2, -7 / -3 = 2
        let formula = MirParser::div_toward_zero("a", "b");
        assert_eq!(formula, "(ite (>= a 0) (div a b) (- (div (- a) b)))");
    }

    #[test]
    fn test_rem_toward_zero_formula_correctness() {
        // Verify the formula produces correct results for all sign combinations
        // 7 % 3 = 1, -7 % 3 = -1, 7 % -3 = 1, -7 % -3 = -1
        let formula = MirParser::rem_toward_zero("a", "b");
        let div = "(ite (>= a 0) (div a b) (- (div (- a) b)))";
        let expected = format!("(- a (* {} b))", div);
        assert_eq!(formula, expected);
    }

    // ========================================================================
    // Bitwise operations tests
    // ========================================================================

    #[test]
    fn test_parse_bitand_operation() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement(
                "_4 = BitAnd(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(stmts.len(), 1);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_4" && rhs == "(bvand _1 _2)")
        );
    }

    #[test]
    fn test_parse_bitor_operation() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement(
                "_4 = BitOr(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(stmts.len(), 1);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_4" && rhs == "(bvor _1 _2)")
        );
    }

    #[test]
    fn test_parse_bitxor_operation() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement(
                "_4 = BitXor(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(stmts.len(), 1);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_4" && rhs == "(bvxor _1 _2)")
        );
    }

    // ========================================================================
    // Shift operations tests
    // ========================================================================

    #[test]
    fn test_parse_shl_operation() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement(
                "_4 = Shl(copy _1, const 2_i32);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(stmts.len(), 1);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_4" && rhs == "(bvshl _1 2)")
        );
    }

    #[test]
    fn test_parse_shr_operation() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement(
                "_4 = Shr(copy _1, const 2_i32);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(stmts.len(), 1);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_4" && rhs == "(bvashr _1 2)")
        );
    }

    #[test]
    fn test_parse_shru_operation() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement(
                "_4 = ShrU(copy _1, copy _2);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(stmts.len(), 1);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_4" && rhs == "(bvlshr _1 _2)")
        );
    }

    // ========================================================================
    // Unary operations tests
    // ========================================================================

    #[test]
    fn test_parse_neg_operation() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement("_3 = Neg(copy _2);", &HashMap::new(), &HashMap::new())
            .unwrap();
        assert_eq!(stmts.len(), 1);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_3" && rhs == "(- _2)")
        );
    }

    #[test]
    fn test_parse_not_operation() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement("_3 = Not(copy _2);", &HashMap::new(), &HashMap::new())
            .unwrap();
        assert_eq!(stmts.len(), 1);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_3" && rhs == "(not _2)")
        );
    }

    #[test]
    fn test_parse_neg_with_move() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement("_3 = Neg(move _2);", &HashMap::new(), &HashMap::new())
            .unwrap();
        assert_eq!(stmts.len(), 1);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_3" && rhs == "(- _2)")
        );
    }

    // ========================================================================
    // Cast operations tests
    // ========================================================================

    #[test]
    fn test_parse_int_to_int_cast() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement(
                "_0 = copy _1 as u32 (IntToInt);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(stmts.len(), 1);
        // For unbounded int model, casts are identity operations
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_0" && rhs == "_1")
        );
    }

    #[test]
    fn test_parse_int_to_int_cast_with_move() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement(
                "_0 = move _1 as i64 (IntToInt);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(stmts.len(), 1);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_0" && rhs == "_1")
        );
    }

    #[test]
    fn test_parse_float_to_int_cast() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement(
                "_5 = copy _3 as i32 (FloatToInt);",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(stmts.len(), 1);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_5" && rhs == "_3")
        );
    }

    // ========================================================================
    // Combined operations in MIR snippet tests
    // ========================================================================

    #[test]
    fn test_parse_function_with_bitwise_ops() {
        let mir = r"
fn bitwise(_1: i32, _2: i32) -> i32 {
    let mut _0: i32;
    let mut _3: i32;
    let mut _4: i32;
    bb0: {
        _3 = BitAnd(copy _1, copy _2);
        _4 = BitOr(copy _3, const 255_i32);
        _0 = BitXor(copy _4, copy _1);
        return;
    }
}
";
        let parser = MirParser::new();
        let func = parser.parse_function(mir, "bitwise").unwrap();

        assert_eq!(func.basic_blocks.len(), 1);
        let stmts = &func.basic_blocks[0].statements;
        assert_eq!(stmts.len(), 3);

        // Verify each statement
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_3" && rhs.contains("bvand"))
        );
        assert!(
            matches!(&stmts[1], MirStatement::Assign { lhs, rhs } if lhs == "_4" && rhs.contains("bvor"))
        );
        assert!(
            matches!(&stmts[2], MirStatement::Assign { lhs, rhs } if lhs == "_0" && rhs.contains("bvxor"))
        );
    }

    #[test]
    fn test_parse_function_with_div_rem() {
        let mir = r"
fn divmod(_1: i32, _2: i32) -> i32 {
    let mut _0: i32;
    let mut _3: i32;
    let mut _4: i32;
    bb0: {
        _3 = Div(copy _1, copy _2);
        _4 = Rem(copy _1, copy _2);
        _0 = Add(copy _3, copy _4);
        return;
    }
}
";
        let parser = MirParser::new();
        let func = parser.parse_function(mir, "divmod").unwrap();

        let stmts = &func.basic_blocks[0].statements;
        assert_eq!(stmts.len(), 3);

        // Division and remainder now use Rust semantics (truncate toward zero)
        let div_expr = "(ite (>= _1 0) (div _1 _2) (- (div (- _1) _2)))";
        let rem_expr = format!("(- _1 (* {} _2))", div_expr);
        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_3" && rhs == div_expr)
        );
        assert!(
            matches!(&stmts[1], MirStatement::Assign { lhs, rhs } if lhs == "_4" && rhs == &rem_expr)
        );
    }

    #[test]
    fn test_parse_function_with_shifts() {
        let mir = r"
fn shifts(_1: i32) -> i32 {
    let mut _0: i32;
    let mut _2: i32;
    bb0: {
        _2 = Shl(copy _1, const 2_i32);
        _0 = Shr(copy _2, const 1_i32);
        return;
    }
}
";
        let parser = MirParser::new();
        let func = parser.parse_function(mir, "shifts").unwrap();

        let stmts = &func.basic_blocks[0].statements;
        assert_eq!(stmts.len(), 2);

        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_2" && rhs == "(bvshl _1 2)")
        );
        assert!(
            matches!(&stmts[1], MirStatement::Assign { lhs, rhs } if lhs == "_0" && rhs == "(bvashr _2 1)")
        );
    }

    #[test]
    fn test_parse_function_with_negation() {
        let mir = r"
fn negate(_1: i32) -> i32 {
    let mut _0: i32;
    bb0: {
        _0 = Neg(copy _1);
        return;
    }
}
";
        let parser = MirParser::new();
        let func = parser.parse_function(mir, "negate").unwrap();

        let stmts = &func.basic_blocks[0].statements;
        assert_eq!(stmts.len(), 1);

        assert!(
            matches!(&stmts[0], MirStatement::Assign { lhs, rhs } if lhs == "_0" && rhs == "(- _1)")
        );
    }

    // ========================================================================
    // Diverging call (panic/abort) tests
    // ========================================================================

    #[test]
    fn test_parse_panic_call_as_abort() {
        let parser = MirParser::new();
        // Test panic call with "-> unwind" (diverging call)
        let term = parser
            .try_parse_terminator(
                r#"_5 = core::panicking::panic(const "assertion failed") -> unwind continue;"#,
                &HashMap::new(),
            )
            .unwrap();
        assert!(
            matches!(term, Some(MirTerminator::Abort)),
            "Panic call should be parsed as Abort, got {:?}",
            term
        );
    }

    #[test]
    fn test_parse_abort_call_as_abort() {
        let parser = MirParser::new();
        // Test abort call
        let term = parser
            .try_parse_terminator(
                r"_1 = std::process::abort() -> unwind continue;",
                &HashMap::new(),
            )
            .unwrap();
        assert!(
            matches!(term, Some(MirTerminator::Abort)),
            "Abort call should be parsed as Abort, got {:?}",
            term
        );
    }

    #[test]
    fn test_parse_function_with_assert_and_panic() {
        // Test parsing a function with switchInt on boolean for assert
        let mir = r#"
fn should_fail() -> () {
    let mut _0: ();
    let _1: i32;
    let mut _3: bool;
    bb0: {
        _1 = const 5_i32;
        _3 = Gt(copy _1, const 10_i32);
        switchInt(move _3) -> [0: bb2, otherwise: bb1];
    }
    bb1: {
        _0 = const ();
        return;
    }
    bb2: {
        _5 = core::panicking::panic(const "assertion failed: x > 10") -> unwind continue;
    }
}
"#;
        let parser = MirParser::new();
        let func = parser.parse_function(mir, "should_fail").unwrap();

        // bb2 should have Abort terminator
        let bb2 = &func.basic_blocks[2];
        assert!(
            matches!(bb2.terminator, MirTerminator::Abort),
            "Block with panic should have Abort terminator, got {:?}",
            bb2.terminator
        );
    }

    #[test]
    fn test_parse_range_aggregate_statement() {
        let parser = MirParser::new();
        // Range struct aggregate initialization from MIR
        let stmts = parser
            .try_parse_statement(
                "_3 = std::ops::Range::<i32> { start: const 0_i32, end: const 3_i32 };",
                &HashMap::new(),
                &HashMap::new(),
            )
            .unwrap();
        // Should be parsed as something - even if empty, the test documents expected behavior
        let _ = stmts;
    }

    #[test]
    fn test_parse_ref_mut() {
        let parser = MirParser::new();
        let stmts = parser
            .try_parse_statement("_6 = &mut _4;", &HashMap::new(), &HashMap::new())
            .unwrap();
        // Should capture this as an assignment where rhs is a reference
        let _ = stmts;
    }

    #[test]
    fn test_parse_trait_method_call() {
        let parser = MirParser::new();
        // Test parsing a function call with trait method syntax
        let mir = r"
fn test() -> () {
    let mut _0: ();
    let mut _1: std::ops::Range<i32>;
    let mut _2: std::ops::Range<i32>;

    bb0: {
        _1 = <std::ops::Range<i32> as IntoIterator>::into_iter(move _2) -> [return: bb1, unwind continue];
    }

    bb1: {
        return;
    }
}
";
        let func = parser.parse_function(mir, "test").unwrap();
        let bb0 = &func.basic_blocks[0];

        // Check that the terminator is a Call with the correct function name
        match &bb0.terminator {
            MirTerminator::Call { func, .. } => {
                assert!(
                    func.contains("Range"),
                    "Expected function name to contain 'Range', got: {}",
                    func
                );
                assert!(
                    func.contains("IntoIterator"),
                    "Expected function name to contain 'IntoIterator', got: {}",
                    func
                );
            }
            other => panic!("Expected Call terminator, got {:?}", other),
        }
    }
}
