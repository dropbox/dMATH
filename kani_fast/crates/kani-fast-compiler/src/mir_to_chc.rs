//! Convert rustc MIR to kani-fast-chc MirProgram
//!
//! This module bridges the gap between rustc's MIR representation and our
//! internal MirProgram type used for CHC encoding.
//!
//! # Architecture
//!
//! This module is the backend converter in the Kani Fast pipeline:
//!
//! ```text
//! Rust Source → rustc → MIR → kani_middle (analysis) → mir_to_chc → CHC → Z4
//!                                    ↓
//!                         [reachability, transforms, safety checks]
//! ```
//!
//! The `kani_middle/` module (copied from upstream Kani) provides battle-tested
//! MIR analysis including:
//! - Attribute parsing (`#[kani::proof]`, contracts)
//! - Reachability analysis
//! - MIR transforms (safety checks, stubbing)
//! - Full Rust type support
//!
//! This module provides the final conversion to our simplified MirProgram:
//!
//! - rustc::mir::Body → MirProgram
//! - rustc::mir::BasicBlockData → MirBasicBlock
//! - rustc::mir::Statement → MirStatement
//! - rustc::mir::Terminator → MirTerminator
//!
//! Our MirProgram uses SMT-LIB2 strings for expressions, making it easy to
//! generate CHC constraints for Z4 PDR verification.
//!
//! # Testing Strategy
//!
//! This module uses a two-tier testing approach:
//!
//! 1. **Unit tests** (in `#[cfg(test)] mod tests`): Test static helper functions
//!    that don't require rustc types:
//!    - `is_panic_call()` - panic function detection
//!    - `binop_to_string()` - binary operation SMT encoding
//!    - `unop_to_string()` - unary operation SMT encoding
//!    - `div_toward_zero()` / `rem_toward_zero()` - Rust division semantics
//!
//! 2. **Integration tests** (`test_driver.sh`): Test the full conversion pipeline
//!    through the actual Rust compiler. These 678 end-to-end tests exercise all
//!    MirConverter methods by compiling real Rust programs.
//!
//! **Mutation testing note**: `cargo mutants` reports ~12% coverage because it
//! only sees unit tests. The 62 "missed" mutants are in functions requiring
//! `TyCtxt`, `Body`, etc., which are tested by integration tests. This is the
//! correct architecture for compiler integration code.

use kani_fast_chc::mir::{
    MirBasicBlock, MirLocal, MirProgram, MirStatement, MirTerminator, PANIC_BLOCK_ID,
};
use kani_fast_kinduction::SmtType;
use rustc_middle::mir::{
    AggregateKind, BasicBlockData, BinOp, Body, Const, ConstValue, Local, Operand, Place, Rvalue,
    Statement, StatementKind, Terminator, TerminatorKind, UnOp,
};
use rustc_middle::ty::{Ty, TyCtxt, TyKind};
use std::collections::HashMap;

/// Check if a function call is a panic/assertion failure
/// These functions never return and should be treated as error transitions
fn is_panic_call(func_name: &str) -> bool {
    // Common panic function patterns in Rust MIR
    let panic_patterns = [
        "panic",
        "assert_failed",
        "panic_cold",
        "panic_cold_explicit",
        "panic_explicit",
        "panic_bounds_check",
        "panic_misaligned_pointer_dereference",
        "panic_nounwind",
        "panic_cannot_unwind",
        "begin_panic",
        "panic_fmt",
        "panic_const",
    ];

    let func_lower = func_name.to_lowercase();
    panic_patterns.iter().any(|p| func_lower.contains(p))
}

/// Configuration for MIR to CHC conversion
#[derive(Debug, Clone, Default)]
pub struct MirConvertConfig {
    /// Enable explicit overflow checking for checked arithmetic operations.
    /// When enabled, operations like `i32::checked_add()` compute actual overflow
    /// flags instead of returning `false`. This catches overflow bugs but
    /// significantly increases CHC solving time.
    pub enable_overflow_checks: bool,
}

impl MirConvertConfig {
    /// Create a new default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable overflow checking
    pub fn with_overflow_checks(mut self, enable: bool) -> Self {
        self.enable_overflow_checks = enable;
        self
    }
}

/// Convert a rustc MIR Body to our MirProgram representation
pub fn convert_body_to_mir_program<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> MirProgram {
    convert_body_to_mir_program_with_config(tcx, body, &MirConvertConfig::default())
}

/// Convert a rustc MIR Body to our MirProgram representation with configuration
pub fn convert_body_to_mir_program_with_config<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    config: &MirConvertConfig,
) -> MirProgram {
    let converter = MirConverter::new(tcx, body, config);
    converter.convert()
}

/// Converter from rustc MIR to our MirProgram
struct MirConverter<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    /// Map from rustc Local to our variable name
    local_names: HashMap<Local, String>,
    /// Configuration for the conversion
    config: MirConvertConfig,
    /// Track reference targets: when we see `_x = &place`, record `_x → place_string`.
    /// This allows us to resolve `(deref _x)` to `place_string` instead of leaving
    /// it as an unconstrained uninterpreted function.
    ///
    /// This fixes soundness bugs where match guards and similar patterns use
    /// references that are immediately dereferenced:
    /// ```text
    /// _4 = &_1;           // Creates reference
    /// _6 = copy (*_4);    // Without tracking: (deref _4) is unconstrained!
    ///                     // With tracking: resolves to _1
    /// ```
    ref_targets: HashMap<Local, String>,
}

#[derive(Debug)]
struct CallInfo {
    func: String,
    is_range_into_iter: bool,
    is_range_next: bool,
}

impl<'a, 'tcx> MirConverter<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, config: &MirConvertConfig) -> Self {
        Self {
            tcx,
            body,
            local_names: HashMap::new(),
            config: config.clone(),
            ref_targets: HashMap::new(),
        }
    }

    /// Convert the entire body to a MirProgram
    fn convert(mut self) -> MirProgram {
        // First, collect all locals and their types
        let locals = self.collect_locals();

        // Pre-pass: collect reference targets for deref resolution.
        // This scans all assignments of the form `_x = &place` and records
        // the mapping `_x → place` so that later `*_x` can be resolved.
        self.collect_ref_targets();

        // Build the program
        let mut builder = MirProgram::builder(0);

        // Add all locals
        for local in &locals {
            builder = builder.local(&local.name, local.ty.clone());
        }

        // Convert each basic block
        for (bb_idx, bb_data) in self.body.basic_blocks.iter_enumerated() {
            let block = self.convert_basic_block(bb_idx.as_usize(), bb_data);
            builder = builder.block(block);
        }

        builder.finish()
    }

    /// Trace back through assignments to find if a local ultimately refers to
    /// a fixed-size array, and if so, return its length.
    ///
    /// This handles cases where `.len()` is called on a reference and the
    /// array type has been erased to a slice. We trace the chain:
    ///   _3 = _4, _4 = &_1, _1: [T; N] => returns N
    fn trace_array_length(&self, start_local: Local) -> Option<u64> {
        let mut visited = std::collections::HashSet::new();
        let mut current = start_local;

        // Follow the chain with cycle detection
        while visited.insert(current) {
            // Check if current local's declared type is an array
            let local_ty = self.body.local_decls[current].ty;
            let inner_ty = local_ty.peel_refs();
            if let TyKind::Array(_, len) = inner_ty.kind()
                && let Some(len_val) = len.try_to_target_usize(self.tcx)
            {
                return Some(len_val);
            }

            // Try to find what this local is assigned from
            let mut next_local = None;
            for bb_data in self.body.basic_blocks.iter() {
                for stmt in &bb_data.statements {
                    if let StatementKind::Assign(assign) = &stmt.kind {
                        let (place, rvalue) = assign.as_ref();
                        // Check if this assigns to our current local
                        if place.local == current && place.projection.is_empty() {
                            // Check what it's assigned from
                            match rvalue {
                                Rvalue::Use(operand) | Rvalue::Cast(_, operand, _) => {
                                    if let Operand::Copy(src) | Operand::Move(src) = operand
                                        && src.projection.is_empty()
                                    {
                                        next_local = Some(src.local);
                                    }
                                }
                                Rvalue::Ref(_, _, ref_place) => {
                                    // &_1 - check if the referenced place is an array
                                    if ref_place.projection.is_empty() {
                                        next_local = Some(ref_place.local);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            if let Some(next) = next_local {
                current = next;
            } else {
                break;
            }
        }

        None
    }

    /// Pre-pass to collect reference targets for deref resolution.
    ///
    /// When we see `_x = &place`, we record `_x → place_string` so that
    /// dereferencing `_x` later can be resolved to `place` instead of
    /// being an unconstrained uninterpreted function.
    ///
    /// This fixes soundness bugs where match guards and similar patterns
    /// create references that are immediately dereferenced.
    fn collect_ref_targets(&mut self) {
        for bb_data in self.body.basic_blocks.iter() {
            for stmt in &bb_data.statements {
                if let StatementKind::Assign(assign) = &stmt.kind {
                    let (place, rvalue) = assign.as_ref();
                    // Only track simple local assignments (no projections on LHS)
                    if !place.projection.is_empty() {
                        continue;
                    }
                    // Check if this is a Ref rvalue
                    if let Rvalue::Ref(_, _, ref_place) = rvalue {
                        // Record the mapping: lhs_local → ref_place_string
                        let target = self.place_to_string(ref_place);
                        self.ref_targets.insert(place.local, target);
                    }
                }
            }
        }
    }

    /// Collect all local variables from the body
    ///
    /// For composite types (tuples, structs) accessed via field projections, we expand
    /// them into separate field variables (e.g., `_6_field0`, `_6_field1`) since MIR
    /// accesses fields via projections that become these variable names.
    ///
    /// This is critical for soundness: if we don't declare field variables, they become
    /// unconstrained in the SMT formula, causing spurious satisfiability.
    fn collect_locals(&mut self) -> Vec<MirLocal> {
        let mut locals = Vec::new();

        for (local, local_decl) in self.body.local_decls.iter_enumerated() {
            let name = format!("_{}", local.as_usize());
            self.local_names.insert(local, name.clone());

            let ty = local_decl.ty;

            // Check if this is a tuple type - if so, expand into field variables
            if let TyKind::Tuple(fields) = ty.kind()
                && !fields.is_empty()
            {
                // Add the base local (for direct assignments to the tuple)
                locals.push(MirLocal::new(name.clone(), SmtType::Int));

                // Add field locals for each tuple element
                for (i, field_ty) in fields.iter().enumerate() {
                    let field_name = format!("{}_field{}", name, i);
                    let field_smt_ty = self.convert_type(field_ty);
                    locals.push(MirLocal::new(field_name.clone(), field_smt_ty));

                    // If this field is itself a tuple, also expand its nested fields
                    // This supports patterns like outer.0.1 where outer is ((T, T), (T, T))
                    if let TyKind::Tuple(nested_fields) = field_ty.kind() {
                        for (j, nested_field_ty) in nested_fields.iter().enumerate() {
                            let nested_field_name = format!("{}_field{}", field_name, j);
                            let nested_field_smt_ty = self.convert_type(nested_field_ty);
                            locals.push(MirLocal::new(nested_field_name, nested_field_smt_ty));
                        }
                    }
                }
                continue;
            }

            // Check if this is a struct/enum type - if so, expand into field variables
            if let TyKind::Adt(adt_def, substs) = ty.kind() {
                // Add the base local (for discriminant / direct assignments)
                locals.push(MirLocal::new(name.clone(), SmtType::Int));

                if adt_def.is_struct() {
                    // Add field locals for each struct field
                    let variant = adt_def.non_enum_variant();
                    for (i, field) in variant.fields.iter().enumerate() {
                        let field_name = format!("{}_field{}", name, i);
                        let field_ty = field.ty(self.tcx, substs);
                        let field_smt_ty = self.convert_type(field_ty);
                        locals.push(MirLocal::new(field_name.clone(), field_smt_ty));

                        // If this field is itself a struct, also expand its nested fields
                        // This supports patterns like outer.inner.value where inner is a struct
                        if let TyKind::Adt(nested_adt, nested_substs) = field_ty.kind()
                            && nested_adt.is_struct()
                        {
                            let nested_variant = nested_adt.non_enum_variant();
                            for (j, nested_field) in nested_variant.fields.iter().enumerate() {
                                let nested_field_name = format!("{}_field{}", field_name, j);
                                let nested_field_ty = nested_field.ty(self.tcx, nested_substs);
                                let nested_field_smt_ty = self.convert_type(nested_field_ty);
                                locals.push(MirLocal::new(nested_field_name, nested_field_smt_ty));
                            }
                        }
                    }
                } else if adt_def.is_enum() {
                    // For enums, we need to expand fields for ALL variants since
                    // we don't know statically which variant will be active.
                    // We use the maximum field count across all variants.
                    //
                    // Note: Different variants can have different field counts,
                    // so we use a union approach where _X_fieldN exists if any
                    // variant has at least N+1 fields.
                    let max_fields = adt_def
                        .variants()
                        .iter()
                        .map(|v| v.fields.len())
                        .max()
                        .unwrap_or(0);

                    for i in 0..max_fields {
                        let field_name = format!("{}_field{}", name, i);
                        // Use Int as default type - in a more sophisticated approach,
                        // we could track per-variant field types
                        locals.push(MirLocal::new(field_name, SmtType::Int));
                    }
                }
                continue;
            }

            // Non-composite types: just add the single local
            let smt_ty = self.convert_type(ty);
            locals.push(MirLocal::new(name, smt_ty));
        }

        locals
    }

    /// Convert a rustc type to our SmtType
    fn convert_type(&self, ty: Ty<'tcx>) -> SmtType {
        match ty.kind() {
            TyKind::Bool => SmtType::Bool,
            TyKind::Int(_) | TyKind::Uint(_) => SmtType::Int,
            TyKind::Float(_) => SmtType::Real,
            TyKind::Tuple(fields) if fields.is_empty() => SmtType::Bool, // unit type
            TyKind::Array(elem_ty, _) => {
                let elem_smt = self.convert_type(*elem_ty);
                SmtType::Array {
                    index: Box::new(SmtType::Int),
                    element: Box::new(elem_smt),
                }
            }
            _ => SmtType::Int, // Default to Int for unsupported types
        }
    }

    /// Convert a SwitchInt value to its correct signed representation
    ///
    /// MIR stores switch values as u128, but for signed integer types we need to
    /// interpret the bit pattern correctly. For example, -5 as i32 is stored as
    /// 4294967291 (0xFFFFFFFB) in the u128, but we need to convert it to -5.
    fn convert_switch_value(&self, val: u128, ty: Ty<'tcx>) -> i64 {
        use rustc_middle::ty::IntTy;

        match ty.kind() {
            TyKind::Int(int_ty) => {
                // For signed types, reinterpret the bit pattern
                match int_ty {
                    IntTy::I8 => val as i8 as i64,
                    IntTy::I16 => val as i16 as i64,
                    IntTy::I32 => val as i32 as i64,
                    IntTy::I64 => val as i64,
                    IntTy::I128 => val as i128 as i64, // May truncate
                    IntTy::Isize => val as isize as i64,
                }
            }
            _ => {
                // For unsigned types and others, just cast to i64
                val as i64
            }
        }
    }

    /// Extract bit width and signedness from an integer type.
    /// Returns (bits, signed) tuple. Defaults to (32, false) for non-integer types.
    ///
    /// NOTE: This helper is prepared for future `--overflow-checks` flag implementation.
    /// See commit #296 for context.
    fn get_int_type_info(ty: Ty<'_>) -> (u32, bool) {
        use rustc_middle::ty::{IntTy, UintTy};

        match ty.kind() {
            TyKind::Int(int_ty) => {
                let bits = match int_ty {
                    IntTy::I8 => 8,
                    IntTy::I16 => 16,
                    IntTy::I32 => 32,
                    IntTy::I64 => 64,
                    IntTy::I128 => 128,
                    IntTy::Isize => 64, // Assume 64-bit platform
                };
                (bits, true)
            }
            TyKind::Uint(uint_ty) => {
                let bits = match uint_ty {
                    UintTy::U8 => 8,
                    UintTy::U16 => 16,
                    UintTy::U32 => 32,
                    UintTy::U64 => 64,
                    UintTy::U128 => 128,
                    UintTy::Usize => 64, // Assume 64-bit platform
                };
                (bits, false)
            }
            _ => (32, false), // Default for non-integer types
        }
    }

    /// Generate an SMT expression for the overflow condition of a checked arithmetic operation.
    ///
    /// This produces an expression that can be used as a boolean flag indicating
    /// whether overflow occurred. The expression uses SMT-LIB2 syntax with:
    /// - Integer comparisons against type bounds for overflow detection
    /// - Symbolic variables from the MIR program
    ///
    /// For unsigned types:
    /// - Add overflow: result < lhs (wraparound detection) OR result > MAX
    /// - Sub overflow: lhs < rhs (would go negative)
    /// - Mul overflow: result > MAX
    ///
    /// For signed types:
    /// - Add overflow: result > MAX OR result < MIN
    /// - Sub overflow: result > MAX OR result < MIN
    /// - Mul overflow: result > MAX OR result < MIN
    ///
    /// This is used when `KANI_FAST_OVERFLOW_CHECKS` is enabled to catch
    /// integer overflow bugs at the cost of CHC solving performance.
    fn generate_overflow_condition(
        op: BinOp,
        lhs: &str,
        rhs: &str,
        result: &str,
        bits: u32,
        signed: bool,
    ) -> String {
        // Calculate bounds based on bit width and signedness
        let (min_bound, max_bound) = if signed {
            if bits >= 128 {
                (i128::MIN.to_string(), i128::MAX.to_string())
            } else {
                let shift = bits.saturating_sub(1);
                let max = (1i128 << shift) - 1;
                let min = -(1i128 << shift);
                (min.to_string(), max.to_string())
            }
        } else {
            let max = if bits >= 128 {
                u128::MAX
            } else {
                (1u128 << bits) - 1
            };
            ("0".to_string(), max.to_string())
        };

        match op {
            BinOp::AddWithOverflow => {
                if signed {
                    // Signed add overflow: result out of [MIN, MAX]
                    // Overflow if: (lhs > 0 AND rhs > 0 AND result < 0) OR
                    //              (lhs < 0 AND rhs < 0 AND result > 0)
                    // Simpler bound check: result > MAX OR result < MIN
                    format!(
                        "(or (> {} {}) (< {} {}))",
                        result, max_bound, result, min_bound
                    )
                } else {
                    // Unsigned add overflow: result > MAX
                    // With unbounded Int, this is: result > MAX
                    // Alternative: result < lhs (wraparound detection for bounded types)
                    format!("(> {} {})", result, max_bound)
                }
            }
            BinOp::SubWithOverflow => {
                if signed {
                    // Signed sub overflow: result out of [MIN, MAX]
                    format!(
                        "(or (> {} {}) (< {} {}))",
                        result, max_bound, result, min_bound
                    )
                } else {
                    // Unsigned sub overflow: lhs < rhs (would go negative)
                    format!("(< {} {})", lhs, rhs)
                }
            }
            BinOp::MulWithOverflow => {
                // Multiplication overflow: result out of bounds
                if signed {
                    format!(
                        "(or (> {} {}) (< {} {}))",
                        result, max_bound, result, min_bound
                    )
                } else {
                    format!("(> {} {})", result, max_bound)
                }
            }
            _ => "false".to_string(),
        }
    }

    /// Convert a basic block
    fn convert_basic_block(&self, idx: usize, bb: &BasicBlockData<'tcx>) -> MirBasicBlock {
        let mut block = MirBasicBlock::new(idx, MirTerminator::Return);

        // Convert statements
        for stmt in &bb.statements {
            for mir_stmt in self.convert_statement(stmt) {
                block = block.with_statement(mir_stmt);
            }
        }

        // Convert terminator
        if let Some(term) = &bb.terminator {
            block.terminator = self.convert_terminator(term);
        }

        block
    }

    /// Convert a statement
    ///
    /// Returns a vector of MirStatements since some operations (like overflow-checked
    /// arithmetic) expand into multiple assignments.
    fn convert_statement(&self, stmt: &Statement<'tcx>) -> Vec<MirStatement> {
        match &stmt.kind {
            StatementKind::Assign(assign) => {
                let (place, rvalue) = assign.as_ref();

                // Handle array index assignment: arr[idx] = val
                // We need to convert this to: arr = (store arr idx val)
                // instead of the incorrect: (select arr idx) = val
                if let Some(array_write) = self.check_array_index_write(place, rvalue) {
                    return array_write;
                }

                // Handle struct/tuple assignment to a field: propagate nested fields
                // When _1_field0 = _2 (where _2 is a tuple/struct), we need:
                //   _1_field0 = _2
                //   _1_field0_field0 = _2_field0
                //   _1_field0_field1 = _2_field1
                //   etc.
                if let Some(nested_stmts) = self.check_nested_struct_assignment(place, rvalue) {
                    return nested_stmts;
                }

                let lhs = self.place_to_string(place);

                // Check if this is an overflow-checked operation that returns (T, bool)
                if let Rvalue::BinaryOp(op, operands) = rvalue
                    && matches!(
                        op,
                        BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow
                    )
                {
                    let (lhs_op, rhs_op) = operands.as_ref();
                    let lhs_str = self.operand_to_string(lhs_op);
                    let rhs_str = self.operand_to_string(rhs_op);

                    // Get operand type to determine overflow semantics
                    let operand_ty = lhs_op.ty(self.body, self.tcx);
                    let (bits, signed) = Self::get_int_type_info(operand_ty);

                    // These operations return (result, overflow_flag)
                    // We need to set up _X_field0 = result and _X_field1 = overflow_flag
                    let result_expr = match op {
                        BinOp::AddWithOverflow => format!("(+ {} {})", lhs_str, rhs_str),
                        BinOp::SubWithOverflow => format!("(- {} {})", lhs_str, rhs_str),
                        BinOp::MulWithOverflow => format!("(* {} {})", lhs_str, rhs_str),
                        _ => unreachable!(),
                    };

                    // Compute overflow flag based on configuration
                    let overflow_flag = if self.config.enable_overflow_checks {
                        // Use a fresh variable for the result in the overflow condition
                        // since we need to reference the computed result value
                        let result_var = format!("{}_field0", lhs);
                        Self::generate_overflow_condition(
                            *op,
                            &lhs_str,
                            &rhs_str,
                            &result_var,
                            bits,
                            signed,
                        )
                    } else {
                        // For unbounded integers (SMT Int sort), overflow never happens.
                        // This is sound because SMT Int represents mathematical integers.
                        //
                        // NOTE: Full overflow checking is available via the generate_overflow_condition
                        // method but significantly degrades CHC solving performance. For practical
                        // verification, bitvector mode (--bitvec) should be used for overflow-sensitive
                        // code. With bitvec mode, arithmetic naturally wraps and overflow is implicit.
                        "false".to_string()
                    };

                    return vec![
                        // Assign the result to _X_field0
                        MirStatement::Assign {
                            lhs: format!("{}_field0", lhs),
                            rhs: result_expr.clone(),
                        },
                        // Assign overflow flag to _X_field1
                        MirStatement::Assign {
                            lhs: format!("{}_field1", lhs),
                            rhs: overflow_flag,
                        },
                        // Also assign to the base variable for compatibility
                        MirStatement::Assign {
                            lhs,
                            rhs: result_expr,
                        },
                    ];
                }

                // Check if this is an aggregate construction (struct/tuple/array/enum)
                if let Rvalue::Aggregate(kind, fields) = rvalue {
                    let mut stmts = Vec::new();

                    // Check aggregate kind
                    match kind.as_ref() {
                        AggregateKind::Array(_) => {
                            // For arrays, build the SMT array using nested store operations
                            // Start with a constant array of zeros and store each element
                            // Final form: (store (store (store const_arr 0 v0) 1 v1) 2 v2)
                            // Using ((as const (Array Int Int)) 0) creates a constant array
                            let mut arr_expr = "((as const (Array Int Int)) 0)".to_string();
                            for (i, field_op) in fields.iter().enumerate() {
                                let field_value = self.operand_to_string(field_op);
                                arr_expr = format!("(store {} {} {})", arr_expr, i, field_value);
                            }
                            stmts.push(MirStatement::Assign { lhs, rhs: arr_expr });
                            return stmts;
                        }
                        AggregateKind::Adt(adt_def_id, variant_idx, substs, _, _) => {
                            // For ADT (enum/struct), assign the discriminant to the base variable
                            // This is what Discriminant reads from
                            stmts.push(MirStatement::Assign {
                                lhs: lhs.clone(),
                                rhs: variant_idx.as_u32().to_string(),
                            });

                            // Get the ADT definition to check field types
                            let adt_def = self.tcx.adt_def(*adt_def_id);
                            let variant = &adt_def.variants()[*variant_idx];

                            // Then assign each field to its corresponding _X_fieldN variable
                            for (i, field_op) in fields.iter().enumerate() {
                                let field_value = self.operand_to_string(field_op);
                                let field_var = format!("{}_field{}", lhs, i);
                                stmts.push(MirStatement::Assign {
                                    lhs: field_var.clone(),
                                    rhs: field_value.clone(),
                                });

                                // Check if this field is itself a struct type
                                // If so, propagate its nested field values
                                if i < variant.fields.len() {
                                    use rustc_abi::FieldIdx;
                                    let field_ty = variant.fields[FieldIdx::from_usize(i)]
                                        .ty(self.tcx, substs);
                                    if let TyKind::Adt(nested_adt, _nested_substs) = field_ty.kind()
                                        && nested_adt.is_struct()
                                    {
                                        // The field value is a variable like "_1" that has fields "_1_field0", etc.
                                        // Copy those to "_2_field0_field0", etc.
                                        let nested_variant = nested_adt.non_enum_variant();
                                        for (j, _nested_field) in
                                            nested_variant.fields.iter().enumerate()
                                        {
                                            stmts.push(MirStatement::Assign {
                                                lhs: format!("{}_field{}", field_var, j),
                                                rhs: format!("{}_field{}", field_value, j),
                                            });
                                        }
                                    }
                                }
                            }
                            return stmts;
                        }
                        AggregateKind::Tuple => {
                            // For tuples, assign fields and propagate nested tuple fields
                            if !fields.is_empty() {
                                for (i, field_op) in fields.iter().enumerate() {
                                    let field_value = self.operand_to_string(field_op);
                                    let field_var = format!("{}_field{}", lhs, i);
                                    stmts.push(MirStatement::Assign {
                                        lhs: field_var.clone(),
                                        rhs: field_value.clone(),
                                    });

                                    // Check if this field is itself a tuple - propagate nested fields
                                    let field_ty = field_op.ty(self.body, self.tcx);
                                    if let TyKind::Tuple(nested_fields) = field_ty.kind() {
                                        for j in 0..nested_fields.len() {
                                            stmts.push(MirStatement::Assign {
                                                lhs: format!("{}_field{}", field_var, j),
                                                rhs: format!("{}_field{}", field_value, j),
                                            });
                                        }
                                    }
                                }

                                // Also assign to the base variable (for MIR compatibility)
                                let rhs = self.rvalue_to_string(rvalue);
                                stmts.push(MirStatement::Assign { lhs, rhs });
                                return stmts;
                            }
                        }
                        _ => {
                            // For other aggregates (closure env, etc.), assign fields
                            if !fields.is_empty() {
                                for (i, field_op) in fields.iter().enumerate() {
                                    let field_value = self.operand_to_string(field_op);
                                    stmts.push(MirStatement::Assign {
                                        lhs: format!("{}_field{}", lhs, i),
                                        rhs: field_value,
                                    });
                                }

                                // Also assign to the base variable (for MIR compatibility)
                                let rhs = self.rvalue_to_string(rvalue);
                                stmts.push(MirStatement::Assign { lhs, rhs });
                                return stmts;
                            }
                        }
                    }
                }

                let rhs = self.rvalue_to_string(rvalue);
                vec![MirStatement::Assign { lhs, rhs }]
            }
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {
                // Ignore storage markers
                vec![]
            }
            StatementKind::Nop => vec![],
            StatementKind::SetDiscriminant {
                place,
                variant_index,
            } => {
                // SetDiscriminant sets the variant tag of an enum
                // We encode this as an assignment to the base variable (which holds discriminant)
                let place_name = self.place_to_string(place);
                let discr_value = variant_index.as_u32();

                // Set the base variable to the variant index
                // This is consistent with how Discriminant reads the place directly
                vec![MirStatement::Assign {
                    lhs: place_name,
                    rhs: discr_value.to_string(),
                }]
            }
            _ => {
                // Log unsupported statement types
                tracing::warn!("Unsupported statement: {:?}", stmt.kind);
                vec![]
            }
        }
    }

    /// Convert a terminator
    fn convert_terminator(&self, term: &Terminator<'tcx>) -> MirTerminator {
        match &term.kind {
            TerminatorKind::Return => MirTerminator::Return,
            TerminatorKind::Unreachable => MirTerminator::Unreachable,
            TerminatorKind::Goto { target } => MirTerminator::Goto {
                target: target.as_usize(),
            },
            TerminatorKind::SwitchInt { discr, targets } => {
                let discr_str = self.operand_to_string(discr);

                // Check if this is a boolean switch (common case)
                let switch_ty = discr.ty(self.body, self.tcx);
                if switch_ty.is_bool() {
                    // Boolean switch: 0 = false branch, otherwise = true branch
                    let false_target = targets.target_for_value(0).as_usize();
                    let true_target = targets.otherwise().as_usize();
                    MirTerminator::CondGoto {
                        condition: discr_str,
                        then_target: true_target,
                        else_target: false_target,
                    }
                } else {
                    // Integer switch - need to handle signed conversion properly
                    // MIR values come as u128, but we need to interpret them as signed
                    // for signed integer types (i8, i16, i32, i64, isize)
                    let mut cases = Vec::new();
                    for (val, target) in targets.iter() {
                        // Convert the value based on the discriminant type
                        let signed_val = self.convert_switch_value(val, switch_ty);
                        cases.push((signed_val, target.as_usize()));
                    }
                    MirTerminator::SwitchInt {
                        discr: discr_str,
                        targets: cases,
                        otherwise: targets.otherwise().as_usize(),
                    }
                }
            }
            TerminatorKind::Assert {
                cond,
                expected,
                msg: _,
                target,
                ..
            } => {
                let cond_str = self.operand_to_string(cond);
                let condition = if *expected {
                    cond_str
                } else {
                    format!("(not {})", cond_str)
                };

                // Create assert followed by goto
                // We encode this as CondGoto where failure goes to an error block
                MirTerminator::CondGoto {
                    condition,
                    then_target: target.as_usize(),
                    else_target: PANIC_BLOCK_ID, // Error/unreachable
                }
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                ..
            } => {
                // Get the function name from the operand
                // For function references, we want the actual function path, not the operand representation
                let call_info = self.get_call_func_info(func);

                // Check if this is a panic/assertion failure call
                // These should be treated as direct transitions to error state
                if is_panic_call(&call_info.func) {
                    // Panic calls are no-return, go directly to error state
                    MirTerminator::Goto {
                        target: PANIC_BLOCK_ID,
                    }
                } else {
                    let args_str: Vec<_> = args
                        .iter()
                        .map(|arg| self.operand_to_string(&arg.node))
                        .collect();
                    let dest_str = self.place_to_string(destination);

                    MirTerminator::Call {
                        destination: Some(dest_str),
                        func: call_info.func,
                        args: args_str,
                        target: target.map_or(PANIC_BLOCK_ID, |t| t.as_usize()),
                        unwind: None,
                        // Contract attributes (#[requires], #[ensures]) are not yet extracted from MIR.
                        // See CLAUDE.md "Known Soundness Limitations" for current workarounds.
                        precondition_check: None,
                        postcondition_assumption: None,
                        is_range_into_iter: call_info.is_range_into_iter,
                        is_range_next: call_info.is_range_next,
                    }
                }
            }
            TerminatorKind::Drop { target, .. } => MirTerminator::Goto {
                target: target.as_usize(),
            },
            TerminatorKind::UnwindResume | TerminatorKind::UnwindTerminate { .. } => {
                MirTerminator::Abort
            }
            _ => {
                tracing::warn!("Unsupported terminator: {:?}", term.kind);
                MirTerminator::Unreachable
            }
        }
    }

    /// Check if this is an array index write (arr[idx] = val) and handle it specially.
    ///
    /// MIR represents array index writes as `Place(arr, [Index(idx)]) = val`, but our
    /// `place_to_string` converts this to `(select arr idx) = val`, which is invalid SMT
    /// (select returns a value, not a location).
    ///
    /// Instead, we need to generate: `arr = (store arr idx val)`
    ///
    /// Returns Some(statements) if this is an array index write, None otherwise.
    fn check_array_index_write(
        &self,
        place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
    ) -> Option<Vec<MirStatement>> {
        // Check if the place has an Index projection
        for (i, proj) in place.projection.iter().enumerate() {
            if let rustc_middle::mir::ProjectionElem::Index(idx_local) = proj {
                // This is an array index write!
                // Get the array base (everything before the Index projection)
                let array_base = if i == 0 {
                    // Direct index on the local: arr[idx]
                    self.local_names
                        .get(&place.local)
                        .cloned()
                        .unwrap_or_else(|| format!("_{}", place.local.as_usize()))
                } else {
                    // There are projections before Index (e.g., field access then index)
                    // Build the base up to (but not including) the Index projection
                    let mut base = self
                        .local_names
                        .get(&place.local)
                        .cloned()
                        .unwrap_or_else(|| format!("_{}", place.local.as_usize()));
                    for proj_before in place.projection.iter().take(i) {
                        match proj_before {
                            rustc_middle::mir::ProjectionElem::Field(f, _) => {
                                base = format!("{}_field{}", base, f.as_usize());
                            }
                            rustc_middle::mir::ProjectionElem::Deref => {
                                base = format!("(deref {})", base);
                            }
                            _ => {
                                // Other projections before index - fallback to default handling
                                return None;
                            }
                        }
                    }
                    base
                };

                // Get the index variable name
                let idx_name = self
                    .local_names
                    .get(&idx_local)
                    .cloned()
                    .unwrap_or_else(|| format!("_{}", idx_local.as_usize()));

                // Get the value being assigned
                let value = self.rvalue_to_string(rvalue);

                // Generate: arr = (store arr idx val)
                let store_expr = format!("(store {} {} {})", array_base, idx_name, value);

                return Some(vec![MirStatement::Assign {
                    lhs: array_base,
                    rhs: store_expr,
                }]);
            }

            // Also handle ConstantIndex writes (less common but possible)
            if let rustc_middle::mir::ProjectionElem::ConstantIndex {
                offset,
                min_length: _,
                from_end,
            } = proj
            {
                if from_end {
                    // from_end indexing is complex, skip for now
                    continue;
                }

                // This is a constant index write: arr[constant] = val
                let array_base = if i == 0 {
                    self.local_names
                        .get(&place.local)
                        .cloned()
                        .unwrap_or_else(|| format!("_{}", place.local.as_usize()))
                } else {
                    // Build base for projections before the ConstantIndex
                    let mut base = self
                        .local_names
                        .get(&place.local)
                        .cloned()
                        .unwrap_or_else(|| format!("_{}", place.local.as_usize()));
                    for proj_before in place.projection.iter().take(i) {
                        match proj_before {
                            rustc_middle::mir::ProjectionElem::Field(f, _) => {
                                base = format!("{}_field{}", base, f.as_usize());
                            }
                            rustc_middle::mir::ProjectionElem::Deref => {
                                base = format!("(deref {})", base);
                            }
                            _ => {
                                return None;
                            }
                        }
                    }
                    base
                };

                let value = self.rvalue_to_string(rvalue);
                let store_expr = format!("(store {} {} {})", array_base, offset, value);

                return Some(vec![MirStatement::Assign {
                    lhs: array_base,
                    rhs: store_expr,
                }]);
            }
        }

        None
    }

    /// Check if this is an assignment of a struct/tuple to a field, and propagate nested fields.
    ///
    /// When we have `_1_field0 = _2` where `_2` is a tuple/struct, we need to also emit:
    ///   `_1_field0_field0 = _2_field0`
    ///   `_1_field0_field1 = _2_field1`
    /// etc.
    ///
    /// This is necessary because accessing `_1_field0_field0` later won't find a value
    /// if we only assigned `_1_field0 = 0` (the discriminant).
    fn check_nested_struct_assignment(
        &self,
        place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
    ) -> Option<Vec<MirStatement>> {
        // Only handle Copy/Move of a place (not complex rvalues)
        let source_place = match rvalue {
            Rvalue::Use(Operand::Copy(p) | Operand::Move(p)) => p,
            _ => return None,
        };

        // Get the type of what we're assigning
        let source_ty = source_place.ty(self.body, self.tcx).ty;

        // Check if it's a tuple or struct with fields
        let field_count = match source_ty.kind() {
            TyKind::Tuple(fields) => fields.len(),
            TyKind::Adt(adt_def, _) if adt_def.is_struct() => {
                adt_def.non_enum_variant().fields.len()
            }
            _ => return None,
        };

        // Only propagate if there are fields
        if field_count == 0 {
            return None;
        }

        // Get the LHS and RHS variable names
        let lhs = self.place_to_string(place);
        let rhs = self.place_to_string(source_place);

        // Build the statements: base assignment + nested field assignments
        let mut stmts = Vec::new();

        // The base assignment (assigns the discriminant/tag)
        stmts.push(MirStatement::Assign {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        });

        // Propagate each nested field
        for i in 0..field_count {
            stmts.push(MirStatement::Assign {
                lhs: format!("{}_field{}", lhs, i),
                rhs: format!("{}_field{}", rhs, i),
            });
        }

        Some(stmts)
    }

    /// Convert a Place to a variable name string
    fn place_to_string(&self, place: &Place<'tcx>) -> String {
        let base = self
            .local_names
            .get(&place.local)
            .cloned()
            .unwrap_or_else(|| format!("_{}", place.local.as_usize()));

        // Handle projections (field access, indexing, etc.)
        if place.projection.is_empty() {
            base
        } else {
            // Simplified: just append projection indices
            let mut result = base;
            for proj in place.projection {
                match proj {
                    rustc_middle::mir::ProjectionElem::Field(f, _) => {
                        result = format!("{}_field{}", result, f.as_usize());
                    }
                    rustc_middle::mir::ProjectionElem::Index(idx) => {
                        let idx_name = self
                            .local_names
                            .get(&idx)
                            .cloned()
                            .unwrap_or_else(|| format!("_{}", idx.as_usize()));
                        result = format!("(select {} {})", result, idx_name);
                    }
                    rustc_middle::mir::ProjectionElem::ConstantIndex {
                        offset,
                        min_length: _,
                        from_end,
                    } => {
                        // ConstantIndex is used for array/slice destructuring patterns
                        // like `let [a, b, c] = arr;` which compiles to:
                        //   _2 = copy _1[0 of 3];  (ConstantIndex { offset: 0, from_end: false })
                        //   _3 = copy _1[1 of 3];  (ConstantIndex { offset: 1, from_end: false })
                        // When from_end is true, the offset is from the end of the array
                        if from_end {
                            // For from_end, we would need to know the array length to compute
                            // the actual index: actual_index = length - 1 - offset
                            // For now, use a placeholder that will be constrained
                            result = format!("{}_from_end_{}", result, offset);
                        } else {
                            // Simple case: offset from start
                            result = format!("(select {} {})", result, offset);
                        }
                    }
                    rustc_middle::mir::ProjectionElem::Deref => {
                        // Check if the base is a tracked reference that we can resolve.
                        // This handles patterns like:
                        //   _4 = &_1;           // Tracked: _4 → _1
                        //   _6 = copy (*_4);    // Resolves *_4 → _1 instead of (deref _4)
                        //
                        // We can only resolve simple derefs where `result` is just the
                        // base local name (no prior projections applied).
                        let base_name = self
                            .local_names
                            .get(&place.local)
                            .cloned()
                            .unwrap_or_else(|| format!("_{}", place.local.as_usize()));
                        if result == base_name {
                            // This is a simple deref of the base local, check ref_targets
                            if let Some(target) = self.ref_targets.get(&place.local) {
                                result = target.clone();
                            } else {
                                result = format!("(deref {})", result);
                            }
                        } else {
                            // Complex projection before deref (e.g., field access then deref)
                            result = format!("(deref {})", result);
                        }
                    }
                    rustc_middle::mir::ProjectionElem::Downcast(_, _) => {
                        // Downcast is a type-level cast to a specific enum variant.
                        // It doesn't change the memory location, so we don't modify the
                        // variable name. The subsequent Field projection will access
                        // the correct field.
                    }
                    _ => {
                        result = format!("{}_proj", result);
                    }
                }
            }
            result
        }
    }

    /// Convert an Operand to an SMT expression string
    fn operand_to_string(&self, op: &Operand<'tcx>) -> String {
        match op {
            Operand::Copy(place) | Operand::Move(place) => self.place_to_string(place),
            Operand::Constant(constant) => self.const_to_string(&constant.const_),
        }
    }

    /// Get call info from a Call operand, including the resolved name and Range hints.
    ///
    /// This resolves function references to their DefPath string representation.
    /// For trait method calls, returns the fully qualified name including the Self type:
    /// `<std::ops::Range<i32> as Iterator>::next` instead of just `Iterator::next`.
    /// Range iterator detection is surfaced via boolean flags to avoid string matching later.
    fn get_call_func_info(&self, func: &Operand<'tcx>) -> CallInfo {
        let mut info = CallInfo {
            func: self.operand_to_string(func),
            is_range_into_iter: false,
            is_range_next: false,
        };

        if let Operand::Constant(constant) = func {
            // For function calls, the constant should be a ZeroSized type representing the function
            let ty = constant.const_.ty();
            if let TyKind::FnDef(def_id, substs) = ty.kind() {
                // Get the full path of the function
                let func_name = self.tcx.def_path_str(*def_id);
                info.func = func_name.clone();

                // Check if this is a Fn/FnMut/FnOnce trait call on a closure
                // The trait method call's first type parameter will be the closure type
                let is_fn_call = func_name.contains("Fn") && func_name.contains("call");

                if is_fn_call {
                    // Debug: print the substs
                    tracing::debug!("FnDef call: {}, substs: {:?}", func_name, substs);

                    // Try to extract the closure type from the generic arguments
                    for arg in substs.iter() {
                        if let Some(arg_ty) = arg.as_type() {
                            tracing::debug!("  subst type: {:?}", arg_ty);
                            if let TyKind::Closure(closure_def_id, _) = arg_ty.kind() {
                                // Return the closure's path which matches our cache
                                let closure_name = self.tcx.def_path_str(*closure_def_id);
                                tracing::debug!("  Found closure: {}", closure_name);
                                info.func = closure_name;
                                return info;
                            }
                        }
                    }
                }

                // For Iterator/IntoIterator trait methods, include the Self type
                // This allows us to detect Range<T>::next() vs other iterator types
                if (func_name.contains("Iterator") || func_name.contains("IntoIterator"))
                    && !substs.is_empty()
                    && let Some(first_arg) = substs.first()
                    && let Some(self_ty) = first_arg.as_type()
                {
                    // Check if this is a Range type
                    if let TyKind::Adt(adt_def, _) = self_ty.kind() {
                        let type_name = self.tcx.def_path_str(adt_def.did());
                        if type_name.contains("Range") {
                            info.is_range_into_iter = func_name.contains("IntoIterator")
                                && func_name.contains("into_iter");
                            info.is_range_next =
                                func_name.contains("Iterator") && func_name.contains("::next");

                            // Return the fully qualified name with Self type
                            // e.g., "<std::ops::Range<i32> as Iterator>::next"
                            let method_name = func_name.rsplit("::").next().unwrap_or(&func_name);
                            let trait_name = if func_name.contains("IntoIterator") {
                                "IntoIterator"
                            } else {
                                "Iterator"
                            };
                            info.func =
                                format!("<{:?} as {}>::{}", self_ty, trait_name, method_name);
                            return info;
                        }
                    }
                }

                return info;
            }
        }

        info
    }

    /// Convert a constant to an SMT expression string
    fn const_to_string(&self, constant: &Const<'tcx>) -> String {
        match constant {
            Const::Val(val, ty) => {
                if let ConstValue::Scalar(scalar) = val {
                    if ty.is_bool() {
                        // Use discard_err() to convert InterpResult to Option
                        if scalar.to_bool().discard_err().unwrap_or(false) {
                            "true".to_string()
                        } else {
                            "false".to_string()
                        }
                    } else if ty.is_integral() || ty.is_char() {
                        // Handle integer and char constants - char is u32 internally
                        // Need to properly sign-extend for signed types
                        if let Some(bits) = scalar.to_bits(scalar.size()).discard_err() {
                            self.convert_int_bits_to_string(bits, *ty)
                        } else {
                            "0".to_string()
                        }
                    } else if matches!(ty.kind(), TyKind::Float(_)) {
                        if let Some(bits) = scalar.to_bits(scalar.size()).discard_err() {
                            self.convert_float_bits_to_real_string(bits, *ty)
                        } else {
                            "0.0".to_string()
                        }
                    } else {
                        // Default for other types
                        "0".to_string()
                    }
                } else {
                    "0".to_string()
                }
            }
            Const::Ty(ty, ty_const) => {
                // Type-level constant (like array lengths from const generics, or
                // integer constants like 1_i32, 10_i32 used in range patterns)
                //
                // Try to evaluate the constant to a value
                use rustc_middle::ty::TypingEnv;
                if let Some(value) = ty_const.try_to_value() {
                    let typing_env = TypingEnv::fully_monomorphized();
                    // Valtree scalars can be converted to integer values
                    if ty.is_bool() {
                        if let Some(b) = value.try_to_bool() {
                            return if b { "true" } else { "false" }.to_string();
                        }
                    } else if (ty.is_integral() || ty.is_char())
                        && let Some(bits) = value.try_to_bits(self.tcx, typing_env)
                    {
                        return self.convert_int_bits_to_string(bits, *ty);
                    } else if matches!(ty.kind(), TyKind::Float(_))
                        && let Some(bits) = value.try_to_bits(self.tcx, typing_env)
                    {
                        return self.convert_float_bits_to_real_string(bits, *ty);
                    }
                }

                // Fallback: try to evaluate as usize (works for array lengths)
                if let Some(val) = ty_const.try_to_target_usize(self.tcx) {
                    return val.to_string();
                }
                // Check if this is an array type with embedded length
                if let TyKind::Array(_, len) = ty.kind()
                    && let Some(len_val) = len.try_to_target_usize(self.tcx)
                {
                    return len_val.to_string();
                }
                // Fallback for unevaluated type-level constants
                tracing::warn!("Unevaluated type constant: {:?}", constant);
                "0".to_string()
            }
            Const::Unevaluated(uneval, ty) => {
                // Try to evaluate the constant
                use rustc_middle::mir::interpret::ErrorHandled;
                use rustc_middle::ty::TypingEnv;
                match self.tcx.const_eval_resolve(
                    TypingEnv::fully_monomorphized(),
                    *uneval,
                    rustc_span::DUMMY_SP,
                ) {
                    Ok(val) => {
                        if let ConstValue::Scalar(scalar) = val {
                            if ty.is_bool() {
                                if scalar.to_bool().discard_err().unwrap_or(false) {
                                    return "true".to_string();
                                } else {
                                    return "false".to_string();
                                }
                            } else if (ty.is_integral() || ty.is_char())
                                && let Some(bits) = scalar.to_bits(scalar.size()).discard_err()
                            {
                                return self.convert_int_bits_to_string(bits, *ty);
                            } else if matches!(ty.kind(), TyKind::Float(_))
                                && let Some(bits) = scalar.to_bits(scalar.size()).discard_err()
                            {
                                return self.convert_float_bits_to_real_string(bits, *ty);
                            }
                        }
                        "0".to_string()
                    }
                    Err(ErrorHandled::Reported(_, _)) => {
                        tracing::warn!("Const eval error (reported): {:?}", constant);
                        "0".to_string()
                    }
                    Err(ErrorHandled::TooGeneric(_)) => {
                        tracing::warn!("Const eval too generic: {:?}", constant);
                        "0".to_string()
                    }
                }
            }
        }
    }

    /// Convert integer bits to a string, properly handling signed types
    ///
    /// MIR stores integer constants as raw bits (u128). For signed types, we need to
    /// sign-extend properly. For example, -10i32 is stored as 4294967286, but we need
    /// to output "-10" for SMT.
    fn convert_int_bits_to_string(&self, bits: u128, ty: Ty<'tcx>) -> String {
        use rustc_middle::ty::IntTy;

        match ty.kind() {
            TyKind::Int(int_ty) => {
                // For signed types, reinterpret the bit pattern
                let signed: i128 = match int_ty {
                    IntTy::I8 => bits as i8 as i128,
                    IntTy::I16 => bits as i16 as i128,
                    IntTy::I32 => bits as i32 as i128,
                    IntTy::I64 => bits as i64 as i128,
                    IntTy::I128 => bits as i128,
                    IntTy::Isize => bits as isize as i128,
                };
                signed.to_string()
            }
            _ => {
                // For unsigned types, just use the bits directly
                bits.to_string()
            }
        }
    }

    /// Convert IEEE-754 float bits to an exact Real value in SMT-LIB2.
    ///
    /// Note: This preserves the exact value of the float constant as a rational with
    /// a power-of-two denominator, which avoids scientific notation (not SMT-LIB2).
    fn convert_float_bits_to_real_string(&self, bits: u128, ty: Ty<'tcx>) -> String {
        use rustc_middle::ty::FloatTy;

        #[derive(Clone, Debug)]
        struct BigUint10 {
            // Base 1e9 digits, little-endian.
            digits: Vec<u32>,
        }

        impl BigUint10 {
            const BASE: u64 = 1_000_000_000;

            fn one() -> Self {
                Self { digits: vec![1] }
            }

            fn from_u64(mut value: u64) -> Self {
                if value == 0 {
                    return Self { digits: vec![0] };
                }
                let mut digits = Vec::new();
                while value > 0 {
                    digits.push((value % Self::BASE) as u32);
                    value /= Self::BASE;
                }
                Self { digits }
            }

            fn mul_small(&mut self, multiplier: u32) {
                let mut carry: u64 = 0;
                for digit in &mut self.digits {
                    let product = (*digit as u64) * (multiplier as u64) + carry;
                    *digit = (product % Self::BASE) as u32;
                    carry = product / Self::BASE;
                }
                while carry > 0 {
                    self.digits.push((carry % Self::BASE) as u32);
                    carry /= Self::BASE;
                }
            }

            fn shl_pow2(&mut self, pow: u32) {
                for _ in 0..pow {
                    self.mul_small(2);
                }
            }

            fn to_decimal_string(&self) -> String {
                let mut iter = self.digits.iter().rev();
                let Some(first) = iter.next() else {
                    return "0".to_string();
                };
                let mut out = first.to_string();
                for digit in iter {
                    out.push_str(&format!("{:09}", digit));
                }
                out
            }
        }

        let (float_ty, raw_bits) = match ty.kind() {
            TyKind::Float(float_ty) => {
                let raw_bits = match float_ty {
                    FloatTy::F32 => bits as u32 as u64,
                    FloatTy::F64 => bits as u64,
                    FloatTy::F16 | FloatTy::F128 => {
                        tracing::warn!(
                            "Float constant type {:?} not supported in Real encoding; encoding as 0.0",
                            float_ty
                        );
                        return "0.0".to_string();
                    }
                };
                (*float_ty, raw_bits)
            }
            _ => return "0.0".to_string(),
        };

        let (sign, exp_raw, mantissa, exp_bits, mantissa_bits, bias): (
            bool,
            u32,
            u64,
            u32,
            u32,
            i32,
        ) = match float_ty {
            FloatTy::F32 => {
                let sign = (raw_bits >> 31) & 1 == 1;
                let exp_raw = ((raw_bits >> 23) & 0xFF) as u32;
                let mantissa = raw_bits & 0x7F_FFFF;
                (sign, exp_raw, mantissa, 8, 23, 127)
            }
            FloatTy::F64 => {
                let sign = (raw_bits >> 63) & 1 == 1;
                let exp_raw = ((raw_bits >> 52) & 0x7FF) as u32;
                let mantissa = raw_bits & 0xF_FFFF_FFFF_FFFF;
                (sign, exp_raw, mantissa, 11, 52, 1023)
            }
            FloatTy::F16 | FloatTy::F128 => unreachable!("handled above"),
        };

        let exp_all_ones = (1u32 << exp_bits) - 1;
        if exp_raw == exp_all_ones {
            // NaN / Infinity are not representable as Real.
            tracing::warn!("Float constant is NaN/Inf; encoding as 0.0: bits={:#x}", raw_bits);
            return "0.0".to_string();
        }

        if exp_raw == 0 && mantissa == 0 {
            // +0.0 and -0.0 are equivalent in Real.
            return "0.0".to_string();
        }

        let (significand, exp2): (u64, i32) = if exp_raw == 0 {
            // Subnormal: 0.mantissa * 2^(1-bias)
            let exp2 = 1 - bias - mantissa_bits as i32;
            (mantissa, exp2)
        } else {
            // Normal: 1.mantissa * 2^(exp-bias)
            let exp2 = exp_raw as i32 - bias - mantissa_bits as i32;
            ((1u64 << mantissa_bits) | mantissa, exp2)
        };

        let mut numerator = BigUint10::from_u64(significand);
        let mut denominator = BigUint10::one();

        if exp2 >= 0 {
            numerator.shl_pow2(exp2 as u32);
        } else {
            denominator.shl_pow2((-exp2) as u32);
        }

        let term = format!(
            "(/ {} {})",
            numerator.to_decimal_string(),
            denominator.to_decimal_string()
        );
        if sign {
            format!("(- {term})")
        } else {
            term
        }
    }

    /// Convert an Rvalue to an SMT expression string
    fn rvalue_to_string(&self, rvalue: &Rvalue<'tcx>) -> String {
        match rvalue {
            Rvalue::Use(op) => self.operand_to_string(op),
            Rvalue::BinaryOp(op, operands) => {
                let (lhs, rhs) = operands.as_ref();
                let lhs_str = self.operand_to_string(lhs);
                let rhs_str = self.operand_to_string(rhs);
                let result_ty = rvalue.ty(self.body, self.tcx);
                let is_bool = result_ty.is_bool();
                Self::binop_to_string_typed(*op, &lhs_str, &rhs_str, is_bool)
            }
            Rvalue::UnaryOp(op, operand) => {
                let operand_str = self.operand_to_string(operand);
                let operand_ty = operand.ty(self.body, self.tcx);
                let is_bool = operand_ty.is_bool();

                // Special handling for PtrMetadata on fixed-size arrays
                // `arr.len()` on `&[T; N]` should return N, not an unconstrained value
                // Note: By the time MIR reaches us, `.len()` has often been desugared to slice
                // operations where the array type is erased to `&[T]`. In those cases, we can't
                // recover the length from the operand type.
                //
                // HOWEVER, we can look at the Place the operand refers to and check if its
                // declared type in local_decls is an array. This recovers the length for simple
                // cases like `let arr = [1,2,3]; arr.len()`.
                if *op == UnOp::PtrMetadata {
                    // First, try the operand type directly (works for some cases)
                    let inner_ty = operand_ty.peel_refs();
                    if let TyKind::Array(_, len) = inner_ty.kind()
                        && let Some(len_val) = len.try_to_target_usize(self.tcx)
                    {
                        return len_val.to_string();
                    }

                    // Second, if operand is a Place, try to trace back through
                    // assignments to find an original array type.
                    // This handles cases like:
                    //   _1: [i32; 3] = [1, 2, 3]
                    //   _4: &[i32] = &_1  // type erased!
                    //   _3 = _4
                    //   ptr_metadata(_3)  // we need to trace _3 -> _4 -> _1 to find [i32; 3]
                    if let Operand::Copy(place) | Operand::Move(place) = operand {
                        // Try to find array length by tracing the assignment chain
                        if let Some(len) = self.trace_array_length(place.local) {
                            return len.to_string();
                        }
                    }
                }

                Self::unop_to_string_typed(*op, &operand_str, is_bool)
            }
            Rvalue::Ref(_, _, place) => {
                // Simplification: treat references as the place itself
                self.place_to_string(place)
            }
            Rvalue::Cast(_, op, _) => {
                // Simplification: ignore cast for now
                self.operand_to_string(op)
            }
            Rvalue::Aggregate(kind, fields) => {
                // Handle enum variants specially - return the discriminant value
                if let AggregateKind::Adt(_, variant_idx, _, _, _) = kind.as_ref() {
                    // For enum variants, return the variant index as the value
                    // This is the discriminant that determines which variant is active
                    return variant_idx.as_usize().to_string();
                }

                // Check if this is an array
                if let AggregateKind::Array(_) = kind.as_ref() {
                    // Arrays are constructed using nested store operations
                    // This case is handled in convert_statement, but we provide a fallback
                    if fields.is_empty() {
                        // Empty array - constant array of zeros
                        return "((as const (Array Int Int)) 0)".to_string();
                    }
                    let mut arr_expr = "((as const (Array Int Int)) 0)".to_string();
                    for (i, field_op) in fields.iter().enumerate() {
                        let field_value = self.operand_to_string(field_op);
                        arr_expr = format!("(store {} {} {})", arr_expr, i, field_value);
                    }
                    return arr_expr;
                }

                // Tuple/struct construction
                // Note: Field assignments are handled in convert_statement, so the base
                // variable gets a dummy value. The actual field values are in _X_fieldN.
                if fields.is_empty() {
                    "true".to_string() // Unit type
                } else {
                    // Use 0 as a placeholder for the base aggregate variable
                    // The real values are in the field variables (_X_field0, _X_field1, etc.)
                    "0".to_string()
                }
            }
            Rvalue::Discriminant(place) => {
                // Get the discriminant of an enum - read the place value directly
                // The place holds the variant index as an integer (set by Aggregate or SetDiscriminant)
                self.place_to_string(place)
            }
            Rvalue::NullaryOp(op) => {
                // Size/align of type - return constant
                format!("(nullary {:?})", op)
            }
            Rvalue::Repeat(op, count) => {
                // Array repeat syntax: [value; count] creates an array filled with `value`
                // We use SMT's constant array syntax: ((as const (Array Int Int)) value)
                // For [5; 3], this creates an array where ALL indices return 5.
                // This is sound: any valid index will return the correct value.
                let value = self.operand_to_string(op);

                // Try to get the count as a concrete integer
                // count is a Const which may be a ty::ConstKind::Value
                if let Some(val) = count.try_to_target_usize(self.tcx) {
                    if val <= 8 {
                        // For small arrays, build explicitly with store for better solver performance
                        // (store (store ... 0 v) 1 v) etc
                        let mut arr_expr = format!("((as const (Array Int Int)) {})", value);
                        for i in 0..val {
                            arr_expr = format!("(store {} {} {})", arr_expr, i, value);
                        }
                        arr_expr
                    } else {
                        // For larger arrays, use constant array (all indices return same value)
                        format!("((as const (Array Int Int)) {})", value)
                    }
                } else {
                    // Unknown count - use constant array (sound but imprecise for length checks)
                    format!("((as const (Array Int Int)) {})", value)
                }
            }
            _ => {
                tracing::warn!("Unsupported rvalue: {:?}", rvalue);
                "0".to_string()
            }
        }
    }

    /// Convert a binary operation to SMT
    ///
    /// Note on bitwise operations: Since we use SMT Int sort (not bitvectors),
    /// we model bitwise operations using:
    /// - Shifts: multiplication/division by powers of 2
    /// - AND/OR/XOR: uninterpreted functions (bitand/bitor/bitxor)
    ///
    /// This is sound but imprecise for bitwise ops - the solver will treat them
    /// as abstract operations rather than computing concrete bit values.
    /// For precise bitwise reasoning, we would need to use BitVec sort.
    #[cfg(test)]
    fn binop_to_string(op: BinOp, lhs: &str, rhs: &str) -> String {
        Self::binop_to_string_typed(op, lhs, rhs, false)
    }

    /// Convert a binary operation to SMT with optional boolean result handling
    ///
    /// When the result type is boolean (e.g., bool & bool), encode bitwise
    /// operators using logical connectives to avoid bitvector operators on Bool.
    fn binop_to_string_typed(op: BinOp, lhs: &str, rhs: &str, result_is_bool: bool) -> String {
        if result_is_bool {
            match op {
                BinOp::BitAnd => return format!("(and {} {})", lhs, rhs),
                BinOp::BitOr => return format!("(or {} {})", lhs, rhs),
                BinOp::BitXor => return format!("(xor {} {})", lhs, rhs),
                _ => {}
            }
        }

        match op {
            BinOp::Add | BinOp::AddUnchecked | BinOp::AddWithOverflow => {
                format!("(+ {} {})", lhs, rhs)
            }
            BinOp::Sub | BinOp::SubUnchecked | BinOp::SubWithOverflow => {
                format!("(- {} {})", lhs, rhs)
            }
            BinOp::Mul | BinOp::MulUnchecked | BinOp::MulWithOverflow => {
                format!("(* {} {})", lhs, rhs)
            }
            // Division with Rust semantics (truncation toward zero).
            // SMT-LIB2's div rounds toward -infinity, so we need special handling for negative dividend.
            BinOp::Div => Self::div_toward_zero(lhs, rhs),
            // Remainder with Rust semantics (same sign as dividend).
            // SMT-LIB2's mod has different semantics for negative dividend.
            BinOp::Rem => Self::rem_toward_zero(lhs, rhs),
            // Bitwise operations: use uninterpreted functions for soundness with Int sort
            // These are declared in the CHC encoder's SMT-LIB2 preamble
            BinOp::BitXor => format!("(bitxor {} {})", lhs, rhs),
            BinOp::BitAnd => format!("(bitand {} {})", lhs, rhs),
            BinOp::BitOr => format!("(bitor {} {})", lhs, rhs),
            // Shifts: use multiplication/division by powers of 2
            // shl(x, n) = x * 2^n (for non-negative n)
            // shr(x, n) = x / 2^n (integer division, for non-negative x and n)
            BinOp::Shl | BinOp::ShlUnchecked => {
                // x << n = x * (2 ^ n), implemented as x * (pow2 n)
                format!("(* {} (pow2 {}))", lhs, rhs)
            }
            BinOp::Shr | BinOp::ShrUnchecked => {
                // x >> n = x / (2 ^ n), implemented as (div x (pow2 n))
                format!("(div {} (pow2 {}))", lhs, rhs)
            }
            BinOp::Eq => format!("(= {} {})", lhs, rhs),
            BinOp::Lt => format!("(< {} {})", lhs, rhs),
            BinOp::Le => format!("(<= {} {})", lhs, rhs),
            BinOp::Ne => format!("(not (= {} {}))", lhs, rhs),
            BinOp::Ge => format!("(>= {} {})", lhs, rhs),
            BinOp::Gt => format!("(> {} {})", lhs, rhs),
            BinOp::Cmp => format!("(cmp {} {})", lhs, rhs),
            BinOp::Offset => format!("(+ {} {})", lhs, rhs),
        }
    }

    /// Convert a unary operation to SMT with type awareness
    ///
    /// For Not:
    /// - Boolean: `(not x)` - logical NOT
    /// - Integer: `(- (- x) 1)` = `-x - 1` - two's complement identity: !x = -x - 1
    ///
    /// For Neg:
    /// - Always `(- x)` - arithmetic negation
    fn unop_to_string_typed(op: UnOp, operand: &str, is_bool: bool) -> String {
        match op {
            UnOp::Not => {
                if is_bool {
                    format!("(not {})", operand)
                } else {
                    // Integer bitwise NOT: !x = -x - 1 (two's complement)
                    format!("(- (- {}) 1)", operand)
                }
            }
            UnOp::Neg => format!("(- {})", operand),
            UnOp::PtrMetadata => format!("(ptr_metadata {})", operand),
        }
    }

    /// Truncating division that matches Rust semantics (round toward zero).
    /// SMT-LIB2's div rounds toward negative infinity, so we need special handling.
    ///
    /// For a >= 0: div(a, b) is correct
    /// For a < 0:  we negate a, divide, then negate the result
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
    /// So: a % b = a - (a / b) * b
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

    // =========================================================================
    // Tests for is_panic_call
    // =========================================================================

    #[test]
    fn test_is_panic_call_basic_panic() {
        assert!(is_panic_call("panic"));
        assert!(is_panic_call("std::panic"));
        assert!(is_panic_call("core::panic"));
    }

    #[test]
    fn test_is_panic_call_panic_variants() {
        assert!(is_panic_call("panic_cold"));
        assert!(is_panic_call("panic_cold_explicit"));
        assert!(is_panic_call("panic_explicit"));
        assert!(is_panic_call("panic_bounds_check"));
        assert!(is_panic_call("panic_misaligned_pointer_dereference"));
        assert!(is_panic_call("panic_nounwind"));
        assert!(is_panic_call("panic_cannot_unwind"));
        assert!(is_panic_call("panic_fmt"));
        assert!(is_panic_call("panic_const"));
    }

    #[test]
    fn test_is_panic_call_assert_failed() {
        assert!(is_panic_call("assert_failed"));
        assert!(is_panic_call("core::panicking::assert_failed"));
        assert!(is_panic_call("std::panicking::assert_failed"));
    }

    #[test]
    fn test_is_panic_call_begin_panic() {
        assert!(is_panic_call("begin_panic"));
        assert!(is_panic_call("std::rt::begin_panic"));
    }

    #[test]
    fn test_is_panic_call_case_insensitive() {
        assert!(is_panic_call("PANIC"));
        assert!(is_panic_call("Panic"));
        assert!(is_panic_call("ASSERT_FAILED"));
        assert!(is_panic_call("Assert_Failed"));
    }

    #[test]
    fn test_is_panic_call_not_panic() {
        assert!(!is_panic_call("add"));
        assert!(!is_panic_call("println"));
        assert!(!is_panic_call("format"));
        // Note: "panicky_function" contains "panic" so it matches
        // Test things that truly don't match any panic pattern
        assert!(!is_panic_call("pin"));
        assert!(!is_panic_call("pair"));
        assert!(!is_panic_call("compare"));
        assert!(!is_panic_call("my_function"));
        assert!(!is_panic_call("handler")); // contains "andle" but not "panic"
    }

    #[test]
    fn test_is_panic_call_with_full_path() {
        assert!(is_panic_call("core::panicking::panic_bounds_check"));
        assert!(is_panic_call("std::panic::panic_any"));
        assert!(is_panic_call("<T as core::ops::Index>::panic_on_bounds"));
    }

    #[test]
    fn test_is_panic_call_empty_string() {
        assert!(!is_panic_call(""));
    }

    // =========================================================================
    // Tests for PANIC_BLOCK_ID constant (imported from kani_fast_chc::mir)
    // =========================================================================

    #[test]
    fn test_panic_block_id_value() {
        // The CHC encoder expects 999_999 as the panic sentinel
        assert_eq!(PANIC_BLOCK_ID, 999_999);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_panic_block_id_larger_than_typical_blocks() {
        // Typical programs have far fewer blocks than this
        // (intentionally a constant assertion to verify the sentinel value)
        assert!(PANIC_BLOCK_ID > 100_000);
    }

    // =========================================================================
    // Tests for div_toward_zero
    // =========================================================================

    #[test]
    fn test_div_toward_zero_format() {
        let result = MirConverter::div_toward_zero("a", "b");
        assert!(result.contains("ite"));
        assert!(result.contains(">= a 0"));
        assert!(result.contains("div a b"));
        assert!(result.contains("- (div (- a) b)"));
    }

    #[test]
    fn test_div_toward_zero_with_numbers() {
        let result = MirConverter::div_toward_zero("7", "3");
        assert!(result.contains("7"));
        assert!(result.contains("3"));
    }

    #[test]
    fn test_div_toward_zero_with_variables() {
        let result = MirConverter::div_toward_zero("_1", "_2");
        assert!(result.contains("_1"));
        assert!(result.contains("_2"));
    }

    #[test]
    fn test_div_toward_zero_with_expressions() {
        let result = MirConverter::div_toward_zero("(+ x y)", "(- a b)");
        assert!(result.contains("(+ x y)"));
        assert!(result.contains("(- a b)"));
    }

    #[test]
    fn test_div_toward_zero_structure() {
        // Should generate: (ite (>= a 0) (div a b) (- (div (- a) b)))
        let result = MirConverter::div_toward_zero("x", "y");
        // Check it's a well-formed ite expression
        assert!(result.starts_with("(ite"));
        assert!(result.ends_with(")"));
        // Count parentheses - should be balanced
        let open = result.matches('(').count();
        let close = result.matches(')').count();
        assert_eq!(open, close, "Unbalanced parentheses in: {}", result);
    }

    // =========================================================================
    // Tests for rem_toward_zero
    // =========================================================================

    #[test]
    fn test_rem_toward_zero_format() {
        let result = MirConverter::rem_toward_zero("a", "b");
        // Should be: (- a (* <div_toward_zero(a,b)> b))
        assert!(result.starts_with("(- a"));
        assert!(result.contains("ite")); // from div_toward_zero
    }

    #[test]
    fn test_rem_toward_zero_with_numbers() {
        let result = MirConverter::rem_toward_zero("7", "3");
        assert!(result.contains("7"));
        assert!(result.contains("3"));
    }

    #[test]
    fn test_rem_toward_zero_structure() {
        let result = MirConverter::rem_toward_zero("x", "y");
        // Check balanced parentheses
        let open = result.matches('(').count();
        let close = result.matches(')').count();
        assert_eq!(open, close, "Unbalanced parentheses in: {}", result);
    }

    #[test]
    fn test_rem_uses_div_toward_zero() {
        // rem = a - (div_toward_zero(a, b) * b)
        let div_result = MirConverter::div_toward_zero("a", "b");
        let rem_result = MirConverter::rem_toward_zero("a", "b");
        // The rem result should contain the div result as a subexpression
        assert!(
            rem_result.contains(&div_result),
            "rem should contain div expression"
        );
    }

    // =========================================================================
    // Tests for generate_overflow_condition (overflow helpers)
    // =========================================================================

    #[test]
    fn test_generate_overflow_condition_unsigned_add_u8_bounds() {
        let cond = MirConverter::generate_overflow_condition(
            BinOp::AddWithOverflow,
            "lhs",
            "rhs",
            "res",
            8,
            false,
        );
        assert_eq!(cond, "(> res 255)");
    }

    #[test]
    fn test_generate_overflow_condition_signed_add_i32_bounds() {
        let cond = MirConverter::generate_overflow_condition(
            BinOp::AddWithOverflow,
            "_1",
            "_2",
            "_r",
            32,
            true,
        );
        assert_eq!(cond, "(or (> _r 2147483647) (< _r -2147483648))");
    }

    #[test]
    fn test_generate_overflow_condition_unsigned_sub_wrapcheck() {
        let cond = MirConverter::generate_overflow_condition(
            BinOp::SubWithOverflow,
            "lhs",
            "rhs",
            "res",
            64,
            false,
        );
        assert_eq!(cond, "(< lhs rhs)");
    }

    #[test]
    fn test_generate_overflow_condition_signed_mul_i128_bounds() {
        let cond = MirConverter::generate_overflow_condition(
            BinOp::MulWithOverflow,
            "lhs",
            "rhs",
            "res128",
            128,
            true,
        );
        let expected = format!(
            "(or (> {} {}) (< {} {}))",
            "res128",
            i128::MAX,
            "res128",
            i128::MIN
        );
        assert_eq!(cond, expected);
    }

    #[test]
    fn test_generate_overflow_condition_unsigned_add_u128_bounds() {
        let cond = MirConverter::generate_overflow_condition(
            BinOp::AddWithOverflow,
            "lhs",
            "rhs",
            "res_u128",
            128,
            false,
        );
        let expected = format!("(> {} {})", "res_u128", u128::MAX);
        assert_eq!(cond, expected);
    }

    // =========================================================================
    // SMT output format validation tests
    // These test the expected SMT-LIB2 output format for various constructs
    // =========================================================================

    #[test]
    fn test_smt_output_format_ite() {
        // Validate that our division generates valid SMT-LIB2 ite syntax
        let result = MirConverter::div_toward_zero("x", "y");
        // SMT-LIB2 ite has form: (ite condition then-branch else-branch)
        let tokens: Vec<&str> = result.split_whitespace().collect();
        assert_eq!(tokens[0], "(ite", "Should start with (ite");
    }

    #[test]
    fn test_smt_output_nesting() {
        // Verify deeply nested expressions produce valid output
        let inner = MirConverter::div_toward_zero("a", "b");
        let outer = MirConverter::rem_toward_zero(&inner, "c");
        // Should still have balanced parens
        let open = outer.matches('(').count();
        let close = outer.matches(')').count();
        assert_eq!(open, close);
    }

    // =========================================================================
    // Tests for binop_to_string and unop_to_string
    // =========================================================================

    #[test]
    fn test_binop_to_string_arithmetic_and_division() {
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Add, "a", "b"),
            "(+ a b)"
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Sub, "x", "y"),
            "(- x y)"
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Mul, "m", "n"),
            "(* m n)"
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Div, "d1", "d2"),
            MirConverter::div_toward_zero("d1", "d2")
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Rem, "r1", "r2"),
            MirConverter::rem_toward_zero("r1", "r2")
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::AddWithOverflow, "o1", "o2"),
            "(+ o1 o2)"
        );
    }

    #[test]
    fn test_binop_to_string_bitwise_and_shifts() {
        assert_eq!(
            MirConverter::binop_to_string(BinOp::BitAnd, "lhs", "rhs"),
            "(bitand lhs rhs)"
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::BitOr, "a", "b"),
            "(bitor a b)"
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::BitXor, "x", "y"),
            "(bitxor x y)"
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Shl, "s1", "s2"),
            "(* s1 (pow2 s2))"
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Shr, "r1", "r2"),
            "(div r1 (pow2 r2))"
        );
    }

    #[test]
    fn test_binop_to_string_bitwise_bool_result() {
        assert_eq!(
            MirConverter::binop_to_string_typed(BinOp::BitAnd, "a", "b", true),
            "(and a b)"
        );
        assert_eq!(
            MirConverter::binop_to_string_typed(BinOp::BitOr, "c1", "c2", true),
            "(or c1 c2)"
        );
        assert_eq!(
            MirConverter::binop_to_string_typed(BinOp::BitXor, "x1", "x2", true),
            "(xor x1 x2)"
        );
    }

    #[test]
    fn test_binop_to_string_comparisons_and_offset() {
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Eq, "l", "r"),
            "(= l r)"
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Ne, "l", "r"),
            "(not (= l r))"
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Le, "a", "b"),
            "(<= a b)"
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Gt, "x", "y"),
            "(> x y)"
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Cmp, "c1", "c2"),
            "(cmp c1 c2)"
        );
        assert_eq!(
            MirConverter::binop_to_string(BinOp::Offset, "p", "off"),
            "(+ p off)"
        );
    }

    #[test]
    fn test_unop_to_string_typed_variants() {
        // Boolean NOT: logical not
        assert_eq!(
            MirConverter::unop_to_string_typed(UnOp::Not, "flag", true),
            "(not flag)"
        );
        // Integer NOT: bitwise not = -x - 1 (two's complement)
        assert_eq!(
            MirConverter::unop_to_string_typed(UnOp::Not, "x", false),
            "(- (- x) 1)"
        );
        // Negation (both bool and int use arithmetic negation)
        assert_eq!(
            MirConverter::unop_to_string_typed(UnOp::Neg, "value", false),
            "(- value)"
        );
        assert_eq!(
            MirConverter::unop_to_string_typed(UnOp::PtrMetadata, "ptr", false),
            "(ptr_metadata ptr)"
        );
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_is_panic_call_unicode() {
        // Should handle unicode without crashing
        assert!(!is_panic_call("пánic")); // Cyrillic п
        assert!(!is_panic_call("pánic")); // Accented a
    }

    #[test]
    fn test_is_panic_call_special_chars() {
        // "panic!" contains "panic" so it matches (case-insensitive substring match)
        assert!(is_panic_call("panic!"));
        assert!(is_panic_call("panic_handler")); // contains "panic"
    }

    #[test]
    fn test_div_rem_with_special_smt_chars() {
        // Variable names can contain underscores
        let result = MirConverter::div_toward_zero("_var_1", "_var_2");
        assert!(result.contains("_var_1"));
        assert!(result.contains("_var_2"));
    }

    #[test]
    fn test_div_rem_with_complex_expressions() {
        // Test with nested SMT expressions
        let complex_a = "(+ (- x 1) (* y 2))";
        let complex_b = "(ite (> z 0) z (- z))";
        let result = MirConverter::div_toward_zero(complex_a, complex_b);
        // Should embed expressions correctly
        assert!(result.contains(complex_a));
        assert!(result.contains(complex_b));
    }
}
