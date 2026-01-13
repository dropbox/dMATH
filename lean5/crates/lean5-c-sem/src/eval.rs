//! C Operational Semantics Interpreter
//!
//! This module implements a big-step operational semantics interpreter
//! for C programs. It can be used to:
//!
//! 1. Execute C programs for testing
//! 2. Generate execution traces for verification
//! 3. Detect undefined behavior at runtime
//!
//! ## Execution Model
//!
//! The interpreter uses a big-step semantics:
//! - Expressions evaluate to values (or UB)
//! - Statements execute and modify state (or UB)
//! - Function calls use a call stack
//!
//! ## State
//!
//! The execution state consists of:
//! - Memory: heap and stack allocations
//! - Environment: variable name → location mapping
//! - Call stack: for function calls
//! - Control flow state: for break/continue/return

use crate::expr::{BinOp, CExpr, ExprResult, Ident, Initializer, SizeOfArg, UnaryOp};
use crate::memory::{Memory, Pointer};
use crate::stmt::{CStmt, FuncDef, StorageClass, TranslationUnit, VarDecl};
use crate::types::{CType, IntKind, Signedness};
use crate::ub::{UBKind, UBResult};
use crate::values::{to_pointer_offset, CValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Variable binding in the environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarBinding {
    /// Pointer to the variable's storage
    pub ptr: Pointer,
    /// Type of the variable
    pub ty: CType,
    /// Is this a const variable?
    pub is_const: bool,
}

/// Local environment (variable scope)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LocalEnv {
    /// Variable bindings: name → binding
    vars: HashMap<Ident, VarBinding>,
}

impl LocalEnv {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    pub fn bind(&mut self, name: Ident, binding: VarBinding) {
        self.vars.insert(name, binding);
    }

    pub fn lookup(&self, name: &str) -> Option<&VarBinding> {
        self.vars.get(name)
    }
}

/// Control flow outcome from statement execution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ControlFlow {
    /// Normal continuation
    Continue,
    /// Break from loop/switch
    Break,
    /// Continue to next iteration
    LoopContinue,
    /// Return from function
    Return(Option<CValue>),
    /// Goto a label
    Goto(Ident),
}

/// Call frame for function calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallFrame {
    /// Function name (for debugging)
    pub func_name: Ident,
    /// Return type
    pub return_type: CType,
    /// Local environment
    pub locals: LocalEnv,
    /// Return address (not used in interpreter, for debugging)
    pub call_depth: usize,
}

/// The execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// Memory
    pub memory: Memory,
    /// Global environment
    pub globals: LocalEnv,
    /// Call stack
    pub call_stack: Vec<CallFrame>,
    /// Current local environment (top of stack)
    current_locals: LocalEnv,
    /// Function definitions
    pub functions: HashMap<Ident, FuncDef>,
    /// String literals (allocated once, reused)
    string_literals: HashMap<String, Pointer>,
    /// Execution depth limit (for recursion)
    pub max_depth: usize,
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl State {
    /// Create a new empty state
    pub fn new() -> Self {
        Self {
            memory: Memory::new(),
            globals: LocalEnv::new(),
            call_stack: Vec::new(),
            current_locals: LocalEnv::new(),
            functions: HashMap::new(),
            string_literals: HashMap::new(),
            max_depth: 1000,
        }
    }

    /// Initialize state from a translation unit
    pub fn from_translation_unit(tu: &TranslationUnit) -> UBResult<Self> {
        let mut state = Self::new();

        // First pass: register function definitions
        for decl in &tu.decls {
            if let crate::stmt::TopLevel::FuncDef(func) = decl {
                state.functions.insert(func.name.clone(), func.clone());
            }
        }

        // Second pass: allocate global variables
        for decl in &tu.decls {
            if let crate::stmt::TopLevel::VarDecl(var) = decl {
                state.alloc_global(var)?;
            }
        }

        Ok(state)
    }

    /// Allocate a global variable
    fn alloc_global(&mut self, decl: &VarDecl) -> UBResult<()> {
        let size = decl.ty.size();
        let align = decl.ty.align();
        let ptr = self.memory.alloc(size, align)?;

        // Initialize to zero by default
        let zero = CValue::zero(&decl.ty);
        self.store_value(ptr, &zero, &decl.ty)?;

        // Apply initializer if present
        if let Some(ref init) = decl.init {
            let val = self.eval_initializer(init, &decl.ty)?;
            self.store_value(ptr, &val, &decl.ty)?;
        }

        self.globals.bind(
            decl.name.clone(),
            VarBinding {
                ptr,
                ty: decl.ty.clone(),
                is_const: false,
            },
        );

        Ok(())
    }

    /// Lookup a variable (local then global)
    pub fn lookup_var(&self, name: &str) -> Option<&VarBinding> {
        // Try local first
        if let Some(binding) = self.current_locals.lookup(name) {
            return Some(binding);
        }
        // Then global
        self.globals.lookup(name)
    }

    /// Store a value at a memory location
    pub fn store_value(&mut self, ptr: Pointer, val: &CValue, ty: &CType) -> UBResult<()> {
        match val {
            CValue::Bool(b) => self.memory.store_u8(ptr, u8::from(*b)),
            // SAFETY: Intentional truncation - C semantics require storing low bits matching type size
            #[allow(clippy::cast_possible_truncation)]
            CValue::Int(i) => match ty.size() {
                1 => self.memory.store_u8(ptr, *i as u8),
                2 => self.memory.store_u16(ptr, *i as u16),
                4 => self.memory.store_i32(ptr, *i as i32),
                8 => self.memory.store_i64(ptr, *i as i64),
                _ => Err(UBKind::Other("unsupported integer size".to_string())),
            },
            // SAFETY: Intentional truncation - C semantics require storing low bits matching type size
            #[allow(clippy::cast_possible_truncation)]
            CValue::UInt(u) => match ty.size() {
                1 => self.memory.store_u8(ptr, *u as u8),
                2 => self.memory.store_u16(ptr, *u as u16),
                4 => self.memory.store_u32(ptr, *u as u32),
                8 => self.memory.store_u64(ptr, *u as u64),
                _ => Err(UBKind::Other("unsupported integer size".to_string())),
            },
            CValue::Float(f) => self.memory.store_f32(ptr, *f),
            CValue::Double(d) => self.memory.store_f64(ptr, *d),
            CValue::Pointer(p) => self.memory.store_ptr(ptr, *p),
            CValue::Struct(fields) => {
                if let CType::Struct {
                    fields: field_types,
                    ..
                } = ty.unqualified()
                {
                    let mut offset = 0usize;
                    for (val, field_ty) in fields.iter().zip(field_types.iter()) {
                        let field_align = field_ty.ty.align();
                        let padding = (field_align - (offset % field_align)) % field_align;
                        offset += padding;
                        let field_ptr = ptr.offset(offset as i64).ok_or(UBKind::PointerOverflow)?;
                        self.store_value(field_ptr, val, &field_ty.ty)?;
                        offset += field_ty.ty.size();
                    }
                    Ok(())
                } else {
                    Err(UBKind::Other("type mismatch in struct store".to_string()))
                }
            }
            CValue::Array(elems) => {
                if let CType::Array(elem_ty, _) = ty.unqualified() {
                    let elem_size = elem_ty.size();
                    for (i, val) in elems.iter().enumerate() {
                        let byte_offset = i
                            .checked_mul(elem_size)
                            .and_then(|o| i64::try_from(o).ok())
                            .ok_or(UBKind::PointerOverflow)?;
                        let elem_ptr = ptr.offset(byte_offset).ok_or(UBKind::PointerOverflow)?;
                        self.store_value(elem_ptr, val, elem_ty)?;
                    }
                    Ok(())
                } else {
                    Err(UBKind::Other("type mismatch in array store".to_string()))
                }
            }
            CValue::Union { value, .. } => {
                // Store the active field value at offset 0
                self.store_value(ptr, value, ty)
            }
            CValue::Undef => {
                // Don't store anything for undef
                Ok(())
            }
        }
    }

    /// Load a value from a memory location
    pub fn load_value(&self, ptr: Pointer, ty: &CType) -> UBResult<CValue> {
        match ty.unqualified() {
            CType::Int(IntKind::Bool, _) => {
                let b = self.memory.load_u8(ptr)?;
                Ok(CValue::Bool(b != 0))
            }
            // SAFETY: For unsigned loads, we reinterpret the signed bits as unsigned.
            // `d as u32` on i32 and `q as u64` on i64 are bit-preserving reinterpretations,
            // not truncation, since the types have the same width.
            CType::Int(kind, sign) => {
                let val = match kind.size() {
                    1 => {
                        let b = self.memory.load_u8(ptr)?;
                        match sign {
                            Signedness::Signed => CValue::Int(b as i8 as i128),
                            Signedness::Unsigned => CValue::UInt(b as u128),
                        }
                    }
                    2 => {
                        let w = self.memory.load_u16(ptr)?;
                        match sign {
                            Signedness::Signed => CValue::Int(w as i16 as i128),
                            Signedness::Unsigned => CValue::UInt(w as u128),
                        }
                    }
                    4 => {
                        let d = self.memory.load_i32(ptr)?;
                        match sign {
                            Signedness::Signed => CValue::Int(d as i128),
                            Signedness::Unsigned => CValue::UInt(d as u32 as u128),
                        }
                    }
                    8 => {
                        let q = self.memory.load_i64(ptr)?;
                        match sign {
                            Signedness::Signed => CValue::Int(q as i128),
                            Signedness::Unsigned => CValue::UInt(q as u64 as u128),
                        }
                    }
                    _ => return Err(UBKind::Other("unsupported integer size".to_string())),
                };
                Ok(val)
            }
            CType::Float(crate::types::FloatKind::Float) => {
                Ok(CValue::Float(self.memory.load_f32(ptr)?))
            }
            CType::Float(_) => Ok(CValue::Double(self.memory.load_f64(ptr)?)),
            CType::Pointer(_) => Ok(CValue::Pointer(self.memory.load_ptr(ptr)?)),
            CType::Enum { .. } => {
                // Enums are ints
                Ok(CValue::Int(self.memory.load_i32(ptr)? as i128))
            }
            CType::Struct { fields, .. } => {
                let mut values = Vec::new();
                let mut offset = 0usize;
                for field in fields {
                    let field_align = field.ty.align();
                    let padding = (field_align - (offset % field_align)) % field_align;
                    offset += padding;
                    let field_ptr = ptr.offset(offset as i64).ok_or(UBKind::PointerOverflow)?;
                    values.push(self.load_value(field_ptr, &field.ty)?);
                    offset += field.ty.size();
                }
                Ok(CValue::Struct(values))
            }
            CType::Array(elem_ty, count) => {
                let elem_size = elem_ty.size();
                let mut values = Vec::new();
                for i in 0..*count {
                    let byte_offset = i
                        .checked_mul(elem_size)
                        .and_then(|o| i64::try_from(o).ok())
                        .ok_or(UBKind::PointerOverflow)?;
                    let elem_ptr = ptr.offset(byte_offset).ok_or(UBKind::PointerOverflow)?;
                    values.push(self.load_value(elem_ptr, elem_ty)?);
                }
                Ok(CValue::Array(values))
            }
            _ => Err(UBKind::Other(format!("cannot load type {ty:?}"))),
        }
    }

    /// Get or create a string literal
    fn get_string_literal(&mut self, s: &str) -> UBResult<Pointer> {
        if let Some(ptr) = self.string_literals.get(s) {
            return Ok(*ptr);
        }

        // Allocate string with null terminator
        let bytes = s.as_bytes();
        let ptr = self.memory.alloc(bytes.len() + 1, 1)?;

        // Store bytes
        for (i, &b) in bytes.iter().enumerate() {
            let char_ptr = ptr.offset(i as i64).ok_or(UBKind::PointerOverflow)?;
            self.memory.store_u8(char_ptr, b)?;
        }
        // Null terminator
        let null_ptr = ptr
            .offset(bytes.len() as i64)
            .ok_or(UBKind::PointerOverflow)?;
        self.memory.store_u8(null_ptr, 0)?;

        self.string_literals.insert(s.to_string(), ptr);
        Ok(ptr)
    }

    /// Evaluate an initializer
    fn eval_initializer(&mut self, init: &Initializer, ty: &CType) -> UBResult<CValue> {
        match init {
            Initializer::Expr(expr) => self.eval_expr_to_value(expr),
            Initializer::List(inits) => match ty.unqualified() {
                CType::Array(elem_ty, count) => {
                    let mut values = vec![CValue::zero(elem_ty); *count];
                    for (i, init) in inits.iter().enumerate() {
                        if i < *count {
                            values[i] = self.eval_initializer(init, elem_ty)?;
                        }
                    }
                    Ok(CValue::Array(values))
                }
                CType::Struct { fields, .. } => {
                    let mut values = Vec::new();
                    for (i, field) in fields.iter().enumerate() {
                        if i < inits.len() {
                            values.push(self.eval_initializer(&inits[i], &field.ty)?);
                        } else {
                            values.push(CValue::zero(&field.ty));
                        }
                    }
                    Ok(CValue::Struct(values))
                }
                _ => {
                    if let Some(first) = inits.first() {
                        self.eval_initializer(first, ty)
                    } else {
                        Ok(CValue::zero(ty))
                    }
                }
            },
            Initializer::Designated { .. } => {
                // Designated initializers need type context
                // For now, just return zero
                Ok(CValue::zero(ty))
            }
        }
    }
}

/// The interpreter
pub struct Interpreter<'a> {
    state: &'a mut State,
}

impl<'a> Interpreter<'a> {
    pub fn new(state: &'a mut State) -> Self {
        Self { state }
    }

    /// Evaluate an expression to a value (rvalue)
    pub fn eval_expr_to_value(&mut self, expr: &CExpr) -> UBResult<CValue> {
        let result = self.eval_expr(expr)?;
        match result {
            ExprResult::RValue(val) => Ok(val),
            ExprResult::LValue(lv) => self.state.load_value(lv.ptr, &lv.ty),
        }
    }

    /// Evaluate an expression
    pub fn eval_expr(&mut self, expr: &CExpr) -> UBResult<ExprResult> {
        match expr {
            CExpr::IntLit(i) => Ok(ExprResult::rvalue(CValue::Int(*i as i128))),

            CExpr::UIntLit(u) => Ok(ExprResult::rvalue(CValue::UInt(*u as u128))),

            CExpr::FloatLit(f) => Ok(ExprResult::rvalue(CValue::Double(*f))),

            CExpr::CharLit(c) => Ok(ExprResult::rvalue(CValue::Int(*c as i128))),

            CExpr::StringLit(s) => {
                let ptr = self.state.get_string_literal(s)?;
                Ok(ExprResult::rvalue(CValue::Pointer(ptr)))
            }

            CExpr::Var(name) => {
                let binding = self
                    .state
                    .lookup_var(name)
                    .ok_or_else(|| UBKind::Other(format!("undefined variable: {name}")))?;
                Ok(ExprResult::lvalue(binding.ptr, binding.ty.clone()))
            }

            CExpr::BinOp { op, left, right } => self.eval_binop(*op, left, right),

            CExpr::UnaryOp { op, operand } => self.eval_unary(*op, operand),

            CExpr::Conditional {
                cond,
                then_expr,
                else_expr,
            } => {
                let cond_val = self.eval_expr_to_value(cond)?;
                if cond_val.to_bool()? {
                    self.eval_expr(then_expr)
                } else {
                    self.eval_expr(else_expr)
                }
            }

            CExpr::Cast { ty, expr } => {
                let val = self.eval_expr_to_value(expr)?;
                // Get source type (approximate)
                let from_ty = self.infer_type(expr)?;
                let casted = val.cast(&from_ty, ty)?;
                Ok(ExprResult::rvalue(casted))
            }

            CExpr::SizeOf(arg) => {
                let size = match arg {
                    SizeOfArg::Type(ty) => ty.size(),
                    SizeOfArg::Expr(e) => {
                        // Don't evaluate, just get type
                        let ty = self.infer_type(e)?;
                        ty.size()
                    }
                };
                Ok(ExprResult::rvalue(CValue::UInt(size as u128)))
            }

            CExpr::AlignOf(ty) => Ok(ExprResult::rvalue(CValue::UInt(ty.align() as u128))),

            CExpr::Index { array, index } => {
                let arr_val = self.eval_expr(array)?;
                let idx_val = self.eval_expr_to_value(index)?;
                let idx = to_pointer_offset(idx_val.to_int()?)?;

                match arr_val {
                    ExprResult::LValue(lv) => {
                        // Array subscript
                        if let Some(elem_ty) = lv.ty.element() {
                            let offset = idx
                                .checked_mul(elem_ty.size() as i64)
                                .ok_or(UBKind::PointerOverflow)?;
                            let elem_ptr = lv.ptr.offset(offset).ok_or(UBKind::PointerOverflow)?;
                            Ok(ExprResult::lvalue(elem_ptr, elem_ty.clone()))
                        } else if let Some(pointee) = lv.ty.pointee() {
                            // Pointer decay
                            let ptr = self.state.load_value(lv.ptr, &lv.ty)?;
                            let base = ptr.to_ptr()?;
                            let offset = idx
                                .checked_mul(pointee.size() as i64)
                                .ok_or(UBKind::PointerOverflow)?;
                            let elem_ptr = base.offset(offset).ok_or(UBKind::PointerOverflow)?;
                            Ok(ExprResult::lvalue(elem_ptr, pointee.clone()))
                        } else {
                            Err(UBKind::Other("subscript on non-array/pointer".to_string()))
                        }
                    }
                    ExprResult::RValue(CValue::Pointer(ptr)) => {
                        // Pointer indexing
                        let arr_ty = self.infer_type(array)?;
                        if let Some(pointee) = arr_ty.pointee() {
                            let offset = idx
                                .checked_mul(pointee.size() as i64)
                                .ok_or(UBKind::PointerOverflow)?;
                            let elem_ptr = ptr.offset(offset).ok_or(UBKind::PointerOverflow)?;
                            Ok(ExprResult::lvalue(elem_ptr, pointee.clone()))
                        } else {
                            Err(UBKind::Other("subscript on non-pointer".to_string()))
                        }
                    }
                    _ => Err(UBKind::Other("subscript on non-array".to_string())),
                }
            }

            CExpr::Member { object, field } => {
                let obj_result = self.eval_expr(object)?;
                match obj_result {
                    ExprResult::LValue(lv) => {
                        let offset = lv
                            .ty
                            .field_offset(field)
                            .ok_or_else(|| UBKind::Other(format!("no field: {field}")))?;
                        let (_, field_info) = lv
                            .ty
                            .get_field(field)
                            .ok_or_else(|| UBKind::Other(format!("no field: {field}")))?;
                        let field_ptr = lv
                            .ptr
                            .offset(offset as i64)
                            .ok_or(UBKind::PointerOverflow)?;
                        Ok(ExprResult::lvalue(field_ptr, field_info.ty.clone()))
                    }
                    _ => Err(UBKind::Other("member access on non-lvalue".to_string())),
                }
            }

            CExpr::Arrow { pointer, field } => {
                let ptr_val = self.eval_expr_to_value(pointer)?;
                let ptr = ptr_val.to_ptr()?;

                let ptr_ty = self.infer_type(pointer)?;
                let struct_ty = ptr_ty
                    .pointee()
                    .ok_or_else(|| UBKind::Other("arrow on non-pointer".to_string()))?;

                let offset = struct_ty
                    .field_offset(field)
                    .ok_or_else(|| UBKind::Other(format!("no field: {field}")))?;
                let (_, field_info) = struct_ty
                    .get_field(field)
                    .ok_or_else(|| UBKind::Other(format!("no field: {field}")))?;

                let field_ptr = ptr.offset(offset as i64).ok_or(UBKind::PointerOverflow)?;
                Ok(ExprResult::lvalue(field_ptr, field_info.ty.clone()))
            }

            CExpr::Call { func, args } => self.eval_call(func, args),

            CExpr::CompoundLiteral { ty, init } => {
                let size = ty.size();
                let align = ty.align();
                let ptr = self.state.memory.alloc(size, align)?;

                // Initialize
                let full_init = Initializer::List(init.clone());
                let val = self.state.eval_initializer(&full_init, ty)?;
                self.state.store_value(ptr, &val, ty)?;

                Ok(ExprResult::lvalue(ptr, ty.clone()))
            }

            CExpr::Generic { .. } => {
                Err(UBKind::Other("_Generic not fully implemented".to_string()))
            }

            CExpr::StmtExpr(stmts) => {
                // GCC extension: ({ stmts; expr })
                let mut last_val = CValue::Undef;
                for stmt in stmts {
                    if let ControlFlow::Return(Some(v)) = self.exec_stmt(stmt)? {
                        return Ok(ExprResult::rvalue(v));
                    }
                    // If last statement is expression, capture its value
                    if let CStmt::Expr(e) = stmt {
                        last_val = self.eval_expr_to_value(e)?;
                    }
                }
                Ok(ExprResult::rvalue(last_val))
            }
        }
    }

    /// Evaluate a binary operation
    fn eval_binop(&mut self, op: BinOp, left: &CExpr, right: &CExpr) -> UBResult<ExprResult> {
        // Handle short-circuit operators
        if op == BinOp::LogAnd {
            let l = self.eval_expr_to_value(left)?;
            if !l.to_bool()? {
                return Ok(ExprResult::rvalue(CValue::Int(0)));
            }
            let r = self.eval_expr_to_value(right)?;
            return Ok(ExprResult::rvalue(CValue::Int(i128::from(r.to_bool()?))));
        }

        if op == BinOp::LogOr {
            let l = self.eval_expr_to_value(left)?;
            if l.to_bool()? {
                return Ok(ExprResult::rvalue(CValue::Int(1)));
            }
            let r = self.eval_expr_to_value(right)?;
            return Ok(ExprResult::rvalue(CValue::Int(i128::from(r.to_bool()?))));
        }

        // Handle comma operator
        if op == BinOp::Comma {
            let _ = self.eval_expr_to_value(left)?;
            return self.eval_expr(right);
        }

        // Handle assignment operators
        if op.is_assignment() {
            return self.eval_assignment(op, left, right);
        }

        // Regular binary operators
        let l = self.eval_expr_to_value(left)?;
        let r = self.eval_expr_to_value(right)?;

        let result_ty = self.infer_type(left)?;
        let promoted_ty = result_ty.usual_arithmetic_conversion(&self.infer_type(right)?);

        let result = match op {
            BinOp::Add => l.add(&r, &promoted_ty)?,
            BinOp::Sub => l.sub(&r, &promoted_ty)?,
            BinOp::Mul => l.mul(&r, &promoted_ty)?,
            BinOp::Div => l.div(&r, &promoted_ty)?,
            BinOp::Mod => l.rem(&r, &promoted_ty)?,
            BinOp::BitAnd => l.bit_and(&r, &promoted_ty)?,
            BinOp::BitOr => l.bit_or(&r, &promoted_ty)?,
            BinOp::BitXor => l.bit_xor(&r, &promoted_ty)?,
            BinOp::Shl => l.shl(&r, &promoted_ty)?,
            BinOp::Shr => l.shr(&r, &promoted_ty)?,
            BinOp::Eq => l.eq(&r)?,
            BinOp::Ne => l.ne(&r)?,
            BinOp::Lt => l.lt(&r)?,
            BinOp::Le => l.le(&r)?,
            BinOp::Gt => l.gt(&r)?,
            BinOp::Ge => l.ge(&r)?,
            _ => unreachable!("handled above"),
        };

        Ok(ExprResult::rvalue(result))
    }

    /// Evaluate an assignment operation
    fn eval_assignment(&mut self, op: BinOp, left: &CExpr, right: &CExpr) -> UBResult<ExprResult> {
        let ExprResult::LValue(lv) = self.eval_expr(left)? else {
            return Err(UBKind::Other("assignment to non-lvalue".to_string()));
        };

        let rhs = self.eval_expr_to_value(right)?;

        let new_val = if op == BinOp::Assign {
            // Simple assignment
            rhs.cast(&self.infer_type(right)?, &lv.ty)?
        } else {
            // Compound assignment: load, operate, store
            let old_val = self.state.load_value(lv.ptr, &lv.ty)?;
            match op {
                BinOp::AddAssign => old_val.add(&rhs, &lv.ty)?,
                BinOp::SubAssign => old_val.sub(&rhs, &lv.ty)?,
                BinOp::MulAssign => old_val.mul(&rhs, &lv.ty)?,
                BinOp::DivAssign => old_val.div(&rhs, &lv.ty)?,
                BinOp::ModAssign => old_val.rem(&rhs, &lv.ty)?,
                BinOp::BitAndAssign => old_val.bit_and(&rhs, &lv.ty)?,
                BinOp::BitOrAssign => old_val.bit_or(&rhs, &lv.ty)?,
                BinOp::BitXorAssign => old_val.bit_xor(&rhs, &lv.ty)?,
                BinOp::ShlAssign => old_val.shl(&rhs, &lv.ty)?,
                BinOp::ShrAssign => old_val.shr(&rhs, &lv.ty)?,
                _ => unreachable!(),
            }
        };

        self.state.store_value(lv.ptr, &new_val, &lv.ty)?;
        Ok(ExprResult::rvalue(new_val))
    }

    /// Evaluate a unary operation
    fn eval_unary(&mut self, op: UnaryOp, operand: &CExpr) -> UBResult<ExprResult> {
        match op {
            UnaryOp::Neg => {
                let val = self.eval_expr_to_value(operand)?;
                let ty = self.infer_type(operand)?;
                Ok(ExprResult::rvalue(val.neg(&ty)?))
            }

            UnaryOp::Pos => {
                // No-op for arithmetic types
                self.eval_expr(operand)
            }

            UnaryOp::BitNot => {
                let val = self.eval_expr_to_value(operand)?;
                let ty = self.infer_type(operand)?;
                Ok(ExprResult::rvalue(val.bit_not(&ty)?))
            }

            UnaryOp::LogNot => {
                let val = self.eval_expr_to_value(operand)?;
                Ok(ExprResult::rvalue(val.log_not()?))
            }

            UnaryOp::Deref => {
                let ptr_val = self.eval_expr_to_value(operand)?;
                let ptr = ptr_val.to_ptr()?;
                let ptr_ty = self.infer_type(operand)?;
                let pointee_ty = ptr_ty
                    .pointee()
                    .ok_or_else(|| UBKind::Other("deref non-pointer".to_string()))?;
                Ok(ExprResult::lvalue(ptr, pointee_ty.clone()))
            }

            UnaryOp::AddrOf => {
                let result = self.eval_expr(operand)?;
                match result {
                    ExprResult::LValue(lv) => Ok(ExprResult::rvalue(CValue::Pointer(lv.ptr))),
                    _ => Err(UBKind::Other("address-of non-lvalue".to_string())),
                }
            }

            UnaryOp::PreInc | UnaryOp::PreDec => {
                let ExprResult::LValue(lv) = self.eval_expr(operand)? else {
                    return Err(UBKind::Other("increment/decrement non-lvalue".to_string()));
                };

                let old_val = self.state.load_value(lv.ptr, &lv.ty)?;
                let one = CValue::Int(1);
                let new_val = if op == UnaryOp::PreInc {
                    old_val.add(&one, &lv.ty)?
                } else {
                    old_val.sub(&one, &lv.ty)?
                };

                self.state.store_value(lv.ptr, &new_val, &lv.ty)?;
                Ok(ExprResult::rvalue(new_val))
            }

            UnaryOp::PostInc | UnaryOp::PostDec => {
                let ExprResult::LValue(lv) = self.eval_expr(operand)? else {
                    return Err(UBKind::Other("increment/decrement non-lvalue".to_string()));
                };

                let old_val = self.state.load_value(lv.ptr, &lv.ty)?;
                let one = CValue::Int(1);
                let new_val = if op == UnaryOp::PostInc {
                    old_val.add(&one, &lv.ty)?
                } else {
                    old_val.sub(&one, &lv.ty)?
                };

                self.state.store_value(lv.ptr, &new_val, &lv.ty)?;
                Ok(ExprResult::rvalue(old_val)) // Return old value
            }
        }
    }

    /// Evaluate a function call
    fn eval_call(&mut self, func: &CExpr, args: &[CExpr]) -> UBResult<ExprResult> {
        // Get function name
        let func_name = match func {
            CExpr::Var(name) => name.clone(),
            _ => return Err(UBKind::Other("indirect call not supported".to_string())),
        };

        // Check recursion depth
        if self.state.call_stack.len() >= self.state.max_depth {
            return Err(UBKind::StackOverflow);
        }

        // Find function definition
        let func_def = self
            .state
            .functions
            .get(&func_name)
            .ok_or_else(|| UBKind::Other(format!("undefined function: {func_name}")))?
            .clone();

        // Check argument count
        if !func_def.variadic && args.len() != func_def.params.len() {
            return Err(UBKind::ArgumentCountMismatch);
        }
        if func_def.variadic && args.len() < func_def.params.len() {
            return Err(UBKind::ArgumentCountMismatch);
        }

        // Evaluate arguments
        let mut arg_vals = Vec::new();
        for arg in args {
            arg_vals.push(self.eval_expr_to_value(arg)?);
        }

        // Push stack frame
        self.state.memory.push_frame();
        let old_locals = std::mem::take(&mut self.state.current_locals);
        self.state.call_stack.push(CallFrame {
            func_name: func_name.clone(),
            return_type: func_def.return_type.clone(),
            locals: LocalEnv::new(),
            call_depth: self.state.call_stack.len(),
        });

        // Bind parameters
        for (i, param) in func_def.params.iter().enumerate() {
            let size = param.ty.size();
            let align = param.ty.align();
            let ptr = self
                .state
                .memory
                .alloc_stack(size, align, Some(param.name.clone()))?;
            self.state.store_value(ptr, &arg_vals[i], &param.ty)?;
            self.state.current_locals.bind(
                param.name.clone(),
                VarBinding {
                    ptr,
                    ty: param.ty.clone(),
                    is_const: false,
                },
            );
        }

        // Execute body
        let result = self.exec_stmt(&func_def.body);

        // Pop stack frame
        self.state.memory.pop_frame();
        self.state.call_stack.pop();
        self.state.current_locals = old_locals;

        // Handle return value
        match result? {
            ControlFlow::Return(Some(val)) => Ok(ExprResult::rvalue(val)),
            ControlFlow::Return(None) | ControlFlow::Continue => {
                if func_def.return_type == CType::Void {
                    Ok(ExprResult::rvalue(CValue::Undef))
                } else {
                    Err(UBKind::MissingReturn)
                }
            }
            _ => Err(UBKind::Other(
                "unexpected control flow from function".to_string(),
            )),
        }
    }

    /// Execute a statement
    pub fn exec_stmt(&mut self, stmt: &CStmt) -> UBResult<ControlFlow> {
        match stmt {
            CStmt::Empty => Ok(ControlFlow::Continue),

            CStmt::Expr(e) => {
                let _ = self.eval_expr_to_value(e)?;
                Ok(ControlFlow::Continue)
            }

            CStmt::Decl(decl) => {
                self.exec_decl(decl)?;
                Ok(ControlFlow::Continue)
            }

            CStmt::DeclList(decls) => {
                for decl in decls {
                    self.exec_decl(decl)?;
                }
                Ok(ControlFlow::Continue)
            }

            CStmt::Block(stmts) => {
                for stmt in stmts {
                    match self.exec_stmt(stmt)? {
                        ControlFlow::Continue => {}
                        other => return Ok(other),
                    }
                }
                Ok(ControlFlow::Continue)
            }

            CStmt::If {
                cond,
                then_stmt,
                else_stmt,
            } => {
                let cond_val = self.eval_expr_to_value(cond)?;
                if cond_val.to_bool()? {
                    self.exec_stmt(then_stmt)
                } else if let Some(else_s) = else_stmt {
                    self.exec_stmt(else_s)
                } else {
                    Ok(ControlFlow::Continue)
                }
            }

            CStmt::While { cond, body } => {
                loop {
                    let cond_val = self.eval_expr_to_value(cond)?;
                    if !cond_val.to_bool()? {
                        break;
                    }
                    match self.exec_stmt(body)? {
                        ControlFlow::Continue | ControlFlow::LoopContinue => {}
                        ControlFlow::Break => break,
                        other => return Ok(other),
                    }
                }
                Ok(ControlFlow::Continue)
            }

            CStmt::DoWhile { body, cond } => {
                loop {
                    match self.exec_stmt(body)? {
                        ControlFlow::Continue | ControlFlow::LoopContinue => {}
                        ControlFlow::Break => break,
                        other => return Ok(other),
                    }
                    let cond_val = self.eval_expr_to_value(cond)?;
                    if !cond_val.to_bool()? {
                        break;
                    }
                }
                Ok(ControlFlow::Continue)
            }

            CStmt::For {
                init,
                cond,
                update,
                body,
            } => {
                // Execute init
                if let Some(init_stmt) = init {
                    match self.exec_stmt(init_stmt)? {
                        ControlFlow::Continue => {}
                        other => return Ok(other),
                    }
                }

                loop {
                    // Check condition
                    if let Some(cond_expr) = cond {
                        let cond_val = self.eval_expr_to_value(cond_expr)?;
                        if !cond_val.to_bool()? {
                            break;
                        }
                    }

                    // Execute body
                    match self.exec_stmt(body)? {
                        ControlFlow::Continue | ControlFlow::LoopContinue => {}
                        ControlFlow::Break => break,
                        other => return Ok(other),
                    }

                    // Execute update
                    if let Some(update_expr) = update {
                        let _ = self.eval_expr_to_value(update_expr)?;
                    }
                }
                Ok(ControlFlow::Continue)
            }

            CStmt::Break => Ok(ControlFlow::Break),

            CStmt::Continue => Ok(ControlFlow::LoopContinue),

            CStmt::Return(expr) => {
                let val = if let Some(e) = expr {
                    Some(self.eval_expr_to_value(e)?)
                } else {
                    None
                };
                Ok(ControlFlow::Return(val))
            }

            CStmt::Goto(label) => Ok(ControlFlow::Goto(label.clone())),

            CStmt::Label { stmt, .. } => {
                // Just execute the statement (labels handled by goto)
                self.exec_stmt(stmt)
            }

            CStmt::Switch { cond, body } => {
                let _val = self.eval_expr_to_value(cond)?;
                // Switch execution requires special handling
                // For now, just execute body
                self.exec_stmt(body)
            }

            CStmt::Case { stmt, .. } => self.exec_stmt(stmt),

            CStmt::FuncDef(_) => {
                // Function definitions at statement level are already handled
                Ok(ControlFlow::Continue)
            }

            CStmt::Asm(_) => {
                // Inline assembly not supported
                Err(UBKind::Other("inline assembly not supported".to_string()))
            }

            CStmt::Assert(_spec) => {
                // Assertions are handled by verification, not execution
                // At runtime, we just skip them (or could check and panic)
                Ok(ControlFlow::Continue)
            }

            CStmt::Assume(_spec) => {
                // Assumptions are handled by verification, not execution
                // At runtime, we assume the spec holds
                Ok(ControlFlow::Continue)
            }
        }
    }

    /// Execute a variable declaration
    fn exec_decl(&mut self, decl: &VarDecl) -> UBResult<()> {
        let size = decl.ty.size();
        let align = decl.ty.align();

        let ptr = match decl.storage {
            StorageClass::Static => {
                // Static locals go in global memory
                self.state.memory.alloc(size, align)?
            }
            _ => {
                // Stack allocation
                self.state
                    .memory
                    .alloc_stack(size, align, Some(decl.name.clone()))?
            }
        };

        // Initialize
        if let Some(ref init) = decl.init {
            let val = self.state.eval_initializer(init, &decl.ty)?;
            self.state.store_value(ptr, &val, &decl.ty)?;
        }

        // Bind to local scope
        self.state.current_locals.bind(
            decl.name.clone(),
            VarBinding {
                ptr,
                ty: decl.ty.clone(),
                is_const: false,
            },
        );

        Ok(())
    }

    /// Infer the type of an expression (approximate)
    fn infer_type(&self, expr: &CExpr) -> UBResult<CType> {
        match expr {
            CExpr::UIntLit(_) => Ok(CType::uint()),
            CExpr::FloatLit(_) => Ok(CType::Float(crate::types::FloatKind::Double)),
            CExpr::CharLit(_) => Ok(CType::char()),
            CExpr::StringLit(_) => Ok(CType::ptr(CType::char())),

            CExpr::Var(name) => {
                let binding = self
                    .state
                    .lookup_var(name)
                    .ok_or_else(|| UBKind::Other(format!("undefined variable: {name}")))?;
                Ok(binding.ty.clone())
            }

            CExpr::BinOp { op, left, right } => {
                if op.is_comparison() || op.is_logical() {
                    Ok(CType::int())
                } else if op.is_assignment() {
                    self.infer_type(left)
                } else {
                    let lt = self.infer_type(left)?;
                    let rt = self.infer_type(right)?;
                    Ok(lt.usual_arithmetic_conversion(&rt))
                }
            }

            CExpr::UnaryOp { op, operand } => match op {
                UnaryOp::Deref => {
                    let ptr_ty = self.infer_type(operand)?;
                    ptr_ty
                        .pointee()
                        .cloned()
                        .ok_or_else(|| UBKind::Other("deref non-pointer".to_string()))
                }
                UnaryOp::AddrOf => {
                    let inner_ty = self.infer_type(operand)?;
                    Ok(CType::ptr(inner_ty))
                }
                UnaryOp::LogNot => Ok(CType::int()),
                _ => self.infer_type(operand),
            },

            CExpr::Conditional { then_expr, .. } => self.infer_type(then_expr),

            CExpr::Cast { ty, .. } | CExpr::CompoundLiteral { ty, .. } => Ok(ty.clone()),

            CExpr::SizeOf(_) | CExpr::AlignOf(_) => {
                Ok(CType::Int(IntKind::Long, Signedness::Unsigned)) // size_t
            }

            CExpr::Index { array, .. } => {
                let arr_ty = self.infer_type(array)?;
                arr_ty
                    .element()
                    .or_else(|| arr_ty.pointee())
                    .cloned()
                    .ok_or_else(|| UBKind::Other("subscript on non-array".to_string()))
            }

            CExpr::Member { object, field }
            | CExpr::Arrow {
                pointer: object,
                field,
            } => {
                let obj_ty = if matches!(expr, CExpr::Arrow { .. }) {
                    let ptr_ty = self.infer_type(object)?;
                    ptr_ty
                        .pointee()
                        .ok_or_else(|| UBKind::Other("arrow on non-pointer".to_string()))?
                        .clone()
                } else {
                    self.infer_type(object)?
                };

                obj_ty
                    .get_field(field)
                    .map(|(_, f)| f.ty.clone())
                    .ok_or_else(|| UBKind::Other(format!("no field: {field}")))
            }

            CExpr::Call { func, .. } => {
                if let CExpr::Var(name) = func.as_ref() {
                    if let Some(func_def) = self.state.functions.get(name) {
                        return Ok(func_def.return_type.clone());
                    }
                }
                Ok(CType::int()) // Default
            }

            // Default to int for unknown/unhandled expression types (including IntLit)
            _ => Ok(CType::int()),
        }
    }
}

// Add this method to State to allow direct expression evaluation
impl State {
    /// Evaluate an expression (convenience method)
    pub fn eval_expr_to_value(&mut self, expr: &CExpr) -> UBResult<CValue> {
        Interpreter::new(self).eval_expr_to_value(expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::BinOp;
    use crate::stmt::{FuncDef, FuncParam};

    #[test]
    fn test_eval_int_literal() {
        let mut state = State::new();
        let val = state.eval_expr_to_value(&CExpr::int(42)).unwrap();
        assert_eq!(val, CValue::Int(42));
    }

    #[test]
    fn test_eval_arithmetic() {
        let mut state = State::new();
        let expr = CExpr::add(CExpr::int(10), CExpr::int(5));
        let val = state.eval_expr_to_value(&expr).unwrap();
        assert_eq!(val, CValue::Int(15));
    }

    #[test]
    fn test_eval_comparison() {
        let mut state = State::new();
        let expr = CExpr::binop(BinOp::Lt, CExpr::int(3), CExpr::int(5));
        let val = state.eval_expr_to_value(&expr).unwrap();
        assert_eq!(val, CValue::Int(1)); // true
    }

    #[test]
    fn test_eval_short_circuit_and() {
        let mut state = State::new();
        // 0 && undefined -> 0 (short circuit)
        let expr = CExpr::binop(BinOp::LogAnd, CExpr::int(0), CExpr::var("undefined"));
        let val = state.eval_expr_to_value(&expr).unwrap();
        assert_eq!(val, CValue::Int(0));
    }

    #[test]
    fn test_eval_short_circuit_or() {
        let mut state = State::new();
        // 1 || undefined -> 1 (short circuit)
        let expr = CExpr::binop(BinOp::LogOr, CExpr::int(1), CExpr::var("undefined"));
        let val = state.eval_expr_to_value(&expr).unwrap();
        assert_eq!(val, CValue::Int(1));
    }

    #[test]
    fn test_exec_if() {
        let mut state = State::new();

        // int x = 0; if (1) x = 10;
        let stmt = CStmt::block(vec![
            CStmt::decl_init("x", CType::int(), CExpr::int(0)),
            CStmt::if_stmt(
                CExpr::int(1),
                CStmt::expr(CExpr::assign(CExpr::var("x"), CExpr::int(10))),
            ),
        ]);

        let mut interp = Interpreter::new(&mut state);
        interp.exec_stmt(&stmt).unwrap();

        let binding = state.lookup_var("x").unwrap();
        let val = state.load_value(binding.ptr, &binding.ty).unwrap();
        assert_eq!(val, CValue::Int(10));
    }

    #[test]
    fn test_exec_while_loop() {
        let mut state = State::new();

        // int i = 0; int sum = 0; while (i < 5) { sum += i; i++; }
        let stmt = CStmt::block(vec![
            CStmt::decl_init("i", CType::int(), CExpr::int(0)),
            CStmt::decl_init("sum", CType::int(), CExpr::int(0)),
            CStmt::while_loop(
                CExpr::binop(BinOp::Lt, CExpr::var("i"), CExpr::int(5)),
                CStmt::block(vec![
                    CStmt::expr(CExpr::binop(
                        BinOp::AddAssign,
                        CExpr::var("sum"),
                        CExpr::var("i"),
                    )),
                    CStmt::expr(CExpr::unary(UnaryOp::PostInc, CExpr::var("i"))),
                ]),
            ),
        ]);

        let mut interp = Interpreter::new(&mut state);
        interp.exec_stmt(&stmt).unwrap();

        let binding = state.lookup_var("sum").unwrap();
        let val = state.load_value(binding.ptr, &binding.ty).unwrap();
        assert_eq!(val, CValue::Int(10)); // 0+1+2+3+4 = 10
    }

    #[test]
    fn test_function_call() {
        let mut state = State::new();

        // Define: int add(int a, int b) { return a + b; }
        let add_func = FuncDef::new(
            "add",
            CType::int(),
            vec![
                FuncParam::new("a", CType::int()),
                FuncParam::new("b", CType::int()),
            ],
            CStmt::return_stmt(Some(CExpr::add(CExpr::var("a"), CExpr::var("b")))),
        );

        state.functions.insert("add".to_string(), add_func);

        // Call: add(3, 4)
        let call = CExpr::call(CExpr::var("add"), vec![CExpr::int(3), CExpr::int(4)]);
        let result = state.eval_expr_to_value(&call).unwrap();
        assert_eq!(result, CValue::Int(7));
    }

    #[test]
    fn test_factorial() {
        let mut state = State::new();

        // int fact(int n) { if (n <= 1) return 1; return n * fact(n-1); }
        let fact_func = FuncDef::new(
            "fact",
            CType::int(),
            vec![FuncParam::new("n", CType::int())],
            CStmt::block(vec![
                CStmt::if_stmt(
                    CExpr::binop(BinOp::Le, CExpr::var("n"), CExpr::int(1)),
                    CStmt::return_stmt(Some(CExpr::int(1))),
                ),
                CStmt::return_stmt(Some(CExpr::mul(
                    CExpr::var("n"),
                    CExpr::call(
                        CExpr::var("fact"),
                        vec![CExpr::sub(CExpr::var("n"), CExpr::int(1))],
                    ),
                ))),
            ]),
        );

        state.functions.insert("fact".to_string(), fact_func);

        // fact(5) = 120
        let call = CExpr::call(CExpr::var("fact"), vec![CExpr::int(5)]);
        let result = state.eval_expr_to_value(&call).unwrap();
        assert_eq!(result, CValue::Int(120));
    }

    #[test]
    fn test_sizeof() {
        let mut state = State::new();

        let expr = CExpr::SizeOf(SizeOfArg::Type(CType::int()));
        let val = state.eval_expr_to_value(&expr).unwrap();
        assert_eq!(val, CValue::UInt(4));
    }

    #[test]
    fn test_pointer_arithmetic_in_expr() {
        let mut state = State::new();

        // Allocate array (10 ints)
        let array_ptr = state.memory.alloc(40, 4).unwrap();

        // Store values 0..9 in the array
        for i in 0..10 {
            let elem_ptr = array_ptr.offset((i * 4) as i64).unwrap();
            state.memory.store_i32(elem_ptr, i).unwrap();
        }

        // Allocate storage for the pointer variable 'arr'
        // Pointer storage is 8 bytes (4 for block_id + 4 for offset)
        let var_ptr = state.memory.alloc(8, 8).unwrap();
        // Store the array pointer value into 'arr'
        state.memory.store_ptr(var_ptr, array_ptr).unwrap();

        // Bind 'arr' as a pointer variable
        state.globals.bind(
            "arr".to_string(),
            VarBinding {
                ptr: var_ptr, // This is where 'arr' is stored
                ty: CType::ptr(CType::int()),
                is_const: false,
            },
        );

        // arr[5] should be 5
        let expr = CExpr::index(CExpr::var("arr"), CExpr::int(5));
        let val = state.eval_expr_to_value(&expr).unwrap();
        assert_eq!(val, CValue::Int(5));
    }

    #[test]
    fn test_division_by_zero_detection() {
        let mut state = State::new();

        let expr = CExpr::div(CExpr::int(10), CExpr::int(0));
        let result = state.eval_expr_to_value(&expr);
        assert!(matches!(result, Err(UBKind::DivisionByZero)));
    }

    #[test]
    fn test_break_statement() {
        let mut state = State::new();

        // int i = 0; while (1) { if (i == 5) break; i++; }
        let stmt = CStmt::block(vec![
            CStmt::decl_init("i", CType::int(), CExpr::int(0)),
            CStmt::while_loop(
                CExpr::int(1),
                CStmt::block(vec![
                    CStmt::if_stmt(
                        CExpr::binop(BinOp::Eq, CExpr::var("i"), CExpr::int(5)),
                        CStmt::break_stmt(),
                    ),
                    CStmt::expr(CExpr::unary(UnaryOp::PostInc, CExpr::var("i"))),
                ]),
            ),
        ]);

        let mut interp = Interpreter::new(&mut state);
        interp.exec_stmt(&stmt).unwrap();

        let binding = state.lookup_var("i").unwrap();
        let val = state.load_value(binding.ptr, &binding.ty).unwrap();
        assert_eq!(val, CValue::Int(5));
    }
}
