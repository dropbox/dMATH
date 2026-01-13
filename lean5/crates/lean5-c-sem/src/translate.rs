//! C → Lean5 Kernel Translation
//!
//! This module provides translation from C programs to Lean5 kernel terms.
//! This enables verification of C programs using Lean5's theorem prover.
//!
//! ## Translation Strategy
//!
//! C programs are translated to Lean5 terms following a deep embedding:
//!
//! 1. **Types**: C types → Lean5 inductive types
//! 2. **Values**: C values → Lean5 terms
//! 3. **Expressions**: C expressions → Lean5 functions (state → result)
//! 4. **Statements**: C statements → Lean5 monadic operations
//! 5. **Specifications**: ACSL specs → Lean5 propositions
//!
//! ## Memory Model
//!
//! The C memory model is represented in Lean5 as:
//!
//! ```text
//! inductive CType where
//!   | int : IntKind → Signedness → CType
//!   | ptr : CType → CType
//!   | struct : List (String × CType) → CType
//!   | ...
//!
//! structure Memory where
//!   blocks : BlockId → Option Block
//!   next_id : Nat
//!
//! def load : Memory → Pointer → CType → Result CValue
//! def store : Memory → Pointer → CValue → Result Memory
//! ```

use crate::expr::{BinOp, CExpr, UnaryOp};
use crate::spec::Spec;
use crate::stmt::CStmt;
use crate::types::{CType, FloatKind, IntKind, Signedness};
use lean5_kernel::{Expr, Name};
use std::str::FromStr;
use std::sync::Arc;

/// Create a Name from a string
fn name(s: &str) -> Name {
    Name::from_str(s).unwrap()
}

/// Translation context
pub struct TranslationContext {
    /// Variable name to de Bruijn level mapping
    var_levels: std::collections::HashMap<String, u32>,
    /// Current de Bruijn level
    current_level: u32,
    /// Generated definitions (for future use: emitting definitions to environment)
    #[allow(dead_code)]
    definitions: Vec<(Name, Expr)>,
}

impl Default for TranslationContext {
    fn default() -> Self {
        Self::new()
    }
}

impl TranslationContext {
    pub fn new() -> Self {
        Self {
            var_levels: std::collections::HashMap::new(),
            current_level: 0,
            definitions: Vec::new(),
        }
    }

    /// Translate a C type to a Lean5 expression
    ///
    /// C types are represented as terms of an inductive `CType` type:
    /// ```text
    /// inductive CType where
    ///   | void : CType
    ///   | int : IntKind → Signedness → CType
    ///   | float : FloatKind → CType
    ///   | ptr : CType → CType
    ///   | array : CType → Nat → CType
    ///   | struct : String → List (String × CType) → CType
    /// ```
    pub fn translate_type(&self, ty: &CType) -> Expr {
        match ty {
            CType::Void => {
                // CType.void
                Expr::const_(name("CType.void"), vec![])
            }

            CType::Int(kind, sign) => {
                // CType.int kind sign
                let kind_expr = self.translate_int_kind(*kind);
                let sign_expr = self.translate_signedness(*sign);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name("CType.int"), vec![])),
                        Arc::new(kind_expr),
                    )),
                    Arc::new(sign_expr),
                )
            }

            CType::Float(kind) => {
                // CType.float kind
                let kind_expr = self.translate_float_kind(*kind);
                Expr::App(
                    Arc::new(Expr::const_(name("CType.float"), vec![])),
                    Arc::new(kind_expr),
                )
            }

            CType::Pointer(inner) => {
                // CType.ptr inner
                let inner_expr = self.translate_type(inner);
                Expr::App(
                    Arc::new(Expr::const_(name("CType.ptr"), vec![])),
                    Arc::new(inner_expr),
                )
            }

            CType::Array(elem, size) => {
                // CType.array elem size
                let elem_expr = self.translate_type(elem);
                let size_expr = self.translate_nat(*size);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name("CType.array"), vec![])),
                        Arc::new(elem_expr),
                    )),
                    Arc::new(size_expr),
                )
            }

            CType::Struct {
                name: struct_name,
                fields,
            } => {
                // CType.struct name fields
                let name_expr = self.translate_string(struct_name.as_deref().unwrap_or(""));
                let fields_expr = self.translate_field_list(fields);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name("CType.struct"), vec![])),
                        Arc::new(name_expr),
                    )),
                    Arc::new(fields_expr),
                )
            }

            CType::Union {
                name: union_name,
                fields,
            } => {
                let name_expr = self.translate_string(union_name.as_deref().unwrap_or(""));
                let fields_expr = self.translate_field_list(fields);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name("CType.union"), vec![])),
                        Arc::new(name_expr),
                    )),
                    Arc::new(fields_expr),
                )
            }

            CType::Enum {
                name: enum_name,
                variants: _,
            } => {
                // Enums are represented as ints in C
                let name_expr = self.translate_string(enum_name.as_deref().unwrap_or(""));
                Expr::App(
                    Arc::new(Expr::const_(name("CType.enum"), vec![])),
                    Arc::new(name_expr),
                )
            }

            CType::Function {
                return_type,
                params,
                variadic,
            } => {
                let ret_expr = self.translate_type(return_type);
                let params_expr = self.translate_param_types(params);
                let variadic_expr = self.translate_bool(*variadic);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::App(
                            Arc::new(Expr::const_(name("CType.func"), vec![])),
                            Arc::new(ret_expr),
                        )),
                        Arc::new(params_expr),
                    )),
                    Arc::new(variadic_expr),
                )
            }

            CType::TypeDef(typedef_name) => {
                // Reference to typedef (should be resolved)
                Expr::const_(name(&format!("CType.typedef.{typedef_name}")), vec![])
            }

            CType::Qualified { ty, .. } => {
                // Ignore qualifiers for now
                self.translate_type(ty)
            }
        }
    }

    fn translate_int_kind(&self, kind: IntKind) -> Expr {
        let kind_name = match kind {
            IntKind::Bool => "IntKind.bool",
            IntKind::Char => "IntKind.char",
            IntKind::Short => "IntKind.short",
            IntKind::Int => "IntKind.int",
            IntKind::Long => "IntKind.long",
            IntKind::LongLong => "IntKind.longLong",
        };
        Expr::const_(name(kind_name), vec![])
    }

    fn translate_signedness(&self, sign: Signedness) -> Expr {
        let sign_name = match sign {
            Signedness::Signed => "Signedness.signed",
            Signedness::Unsigned => "Signedness.unsigned",
        };
        Expr::const_(name(sign_name), vec![])
    }

    fn translate_float_kind(&self, kind: FloatKind) -> Expr {
        let float_name = match kind {
            FloatKind::Float => "FloatKind.float",
            FloatKind::Double => "FloatKind.double",
            FloatKind::LongDouble => "FloatKind.longDouble",
        };
        Expr::const_(name(float_name), vec![])
    }

    fn translate_nat(&self, n: usize) -> Expr {
        if n == 0 {
            Expr::const_(name("Nat.zero"), vec![])
        } else {
            Expr::App(
                Arc::new(Expr::const_(name("Nat.succ"), vec![])),
                Arc::new(self.translate_nat(n - 1)),
            )
        }
    }

    fn translate_int(&self, n: i64) -> Expr {
        // Represent as Int constructor
        if n >= 0 {
            Expr::App(
                Arc::new(Expr::const_(name("Int.ofNat"), vec![])),
                Arc::new(self.translate_nat(n as usize)),
            )
        } else {
            // Handle i64::MIN correctly by computing magnitude as u64.
            // For negative i64 values, the magnitude is 0u64.wrapping_sub(n as u64).
            // This works for all negative values including i64::MIN:
            // - i64::MIN as u64 = 9223372036854775808
            // - 0u64.wrapping_sub(9223372036854775808) = 9223372036854775808
            // For other values like -5: -5i64 as u64 = 18446744073709551611
            // - 0u64.wrapping_sub(18446744073709551611) = 5
            let magnitude = 0u64.wrapping_sub(n as u64) as usize;
            Expr::App(
                Arc::new(Expr::const_(name("Int.negOfNat"), vec![])),
                Arc::new(self.translate_nat(magnitude)),
            )
        }
    }

    fn translate_bool(&self, b: bool) -> Expr {
        if b {
            Expr::const_(name("Bool.true"), vec![])
        } else {
            Expr::const_(name("Bool.false"), vec![])
        }
    }

    fn translate_string(&self, s: &str) -> Expr {
        // Strings as lists of chars
        Expr::Lit(lean5_kernel::Literal::String(s.into()))
    }

    fn translate_field_list(&self, fields: &[crate::types::StructField]) -> Expr {
        // Build List (String × CType)
        let mut result = Expr::const_(name("List.nil"), vec![]);
        for field in fields.iter().rev() {
            let name_expr = self.translate_string(&field.name);
            let ty_expr = self.translate_type(&field.ty);
            let pair = Expr::App(
                Arc::new(Expr::App(
                    Arc::new(Expr::const_(name("Prod.mk"), vec![])),
                    Arc::new(name_expr),
                )),
                Arc::new(ty_expr),
            );
            result = Expr::App(
                Arc::new(Expr::App(
                    Arc::new(Expr::const_(name("List.cons"), vec![])),
                    Arc::new(pair),
                )),
                Arc::new(result),
            );
        }
        result
    }

    fn translate_param_types(&self, params: &[crate::types::FuncParam]) -> Expr {
        let mut result = Expr::const_(name("List.nil"), vec![]);
        for param in params.iter().rev() {
            let ty_expr = self.translate_type(&param.ty);
            result = Expr::App(
                Arc::new(Expr::App(
                    Arc::new(Expr::const_(name("List.cons"), vec![])),
                    Arc::new(ty_expr),
                )),
                Arc::new(result),
            );
        }
        result
    }

    /// Translate a C expression to a Lean5 term
    ///
    /// C expressions are translated to functions: State → Result CValue
    pub fn translate_expr(&mut self, expr: &CExpr) -> Expr {
        match expr {
            CExpr::IntLit(n) => {
                // CValue.int n
                Expr::App(
                    Arc::new(Expr::const_(name("CValue.int"), vec![])),
                    Arc::new(self.translate_int(*n)),
                )
            }

            CExpr::UIntLit(n) => Expr::App(
                Arc::new(Expr::const_(name("CValue.uint"), vec![])),
                Arc::new(self.translate_nat(*n as usize)),
            ),

            CExpr::FloatLit(f) => {
                // CValue.float f (as string for now)
                Expr::App(
                    Arc::new(Expr::const_(name("CValue.float"), vec![])),
                    Arc::new(self.translate_string(&f.to_string())),
                )
            }

            CExpr::CharLit(c) => Expr::App(
                Arc::new(Expr::const_(name("CValue.int"), vec![])),
                Arc::new(self.translate_int(*c as i64)),
            ),

            CExpr::StringLit(s) => Expr::App(
                Arc::new(Expr::const_(name("CValue.string"), vec![])),
                Arc::new(self.translate_string(s)),
            ),

            CExpr::Var(var_name) => {
                if let Some(&level) = self.var_levels.get(var_name) {
                    // Use bound variable
                    let index = self.current_level - level - 1;
                    Expr::BVar(index)
                } else {
                    // Free variable / global
                    Expr::const_(name(&format!("var.{var_name}")), vec![])
                }
            }

            CExpr::BinOp { op, left, right } => {
                let left_expr = self.translate_expr(left);
                let right_expr = self.translate_expr(right);
                let op_name = self.translate_binop(*op);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name(op_name), vec![])),
                        Arc::new(left_expr),
                    )),
                    Arc::new(right_expr),
                )
            }

            CExpr::UnaryOp { op, operand } => {
                let operand_expr = self.translate_expr(operand);
                let op_name = self.translate_unaryop(*op);
                Expr::App(
                    Arc::new(Expr::const_(name(op_name), vec![])),
                    Arc::new(operand_expr),
                )
            }

            CExpr::Conditional {
                cond,
                then_expr,
                else_expr,
            } => {
                let cond_expr = self.translate_expr(cond);
                let then_e = self.translate_expr(then_expr);
                let else_e = self.translate_expr(else_expr);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::App(
                            Arc::new(Expr::const_(name("CExpr.cond"), vec![])),
                            Arc::new(cond_expr),
                        )),
                        Arc::new(then_e),
                    )),
                    Arc::new(else_e),
                )
            }

            CExpr::Cast { ty, expr: e } => {
                let ty_expr = self.translate_type(ty);
                let e_expr = self.translate_expr(e);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name("CExpr.cast"), vec![])),
                        Arc::new(ty_expr),
                    )),
                    Arc::new(e_expr),
                )
            }

            CExpr::SizeOf(arg) => {
                let size = match arg {
                    crate::expr::SizeOfArg::Type(ty) => ty.size(),
                    crate::expr::SizeOfArg::Expr(_) => 0, // Would need type inference
                };
                Expr::App(
                    Arc::new(Expr::const_(name("CValue.uint"), vec![])),
                    Arc::new(self.translate_nat(size)),
                )
            }

            CExpr::AlignOf(ty) => Expr::App(
                Arc::new(Expr::const_(name("CValue.uint"), vec![])),
                Arc::new(self.translate_nat(ty.align())),
            ),

            CExpr::Call { func, args } => {
                let func_expr = self.translate_expr(func);
                let args_expr = self.translate_expr_list(args);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name("CExpr.call"), vec![])),
                        Arc::new(func_expr),
                    )),
                    Arc::new(args_expr),
                )
            }

            CExpr::Index { array, index } => {
                let arr_expr = self.translate_expr(array);
                let idx_expr = self.translate_expr(index);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name("CExpr.index"), vec![])),
                        Arc::new(arr_expr),
                    )),
                    Arc::new(idx_expr),
                )
            }

            CExpr::Member { object, field } => {
                let obj_expr = self.translate_expr(object);
                let field_expr = self.translate_string(field);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name("CExpr.member"), vec![])),
                        Arc::new(obj_expr),
                    )),
                    Arc::new(field_expr),
                )
            }

            CExpr::Arrow { pointer, field } => {
                // p->field = (*p).field
                let deref = CExpr::UnaryOp {
                    op: UnaryOp::Deref,
                    operand: pointer.clone(),
                };
                let member = CExpr::Member {
                    object: Box::new(deref),
                    field: field.clone(),
                };
                self.translate_expr(&member)
            }

            _ => {
                // Default: unsupported expression
                Expr::const_(name("CExpr.unsupported"), vec![])
            }
        }
    }

    fn translate_binop(&self, op: BinOp) -> &'static str {
        match op {
            BinOp::Add => "CExpr.add",
            BinOp::Sub => "CExpr.sub",
            BinOp::Mul => "CExpr.mul",
            BinOp::Div => "CExpr.div",
            BinOp::Mod => "CExpr.mod",
            BinOp::BitAnd => "CExpr.bitAnd",
            BinOp::BitOr => "CExpr.bitOr",
            BinOp::BitXor => "CExpr.bitXor",
            BinOp::Shl => "CExpr.shl",
            BinOp::Shr => "CExpr.shr",
            BinOp::Eq => "CExpr.eq",
            BinOp::Ne => "CExpr.ne",
            BinOp::Lt => "CExpr.lt",
            BinOp::Le => "CExpr.le",
            BinOp::Gt => "CExpr.gt",
            BinOp::Ge => "CExpr.ge",
            BinOp::LogAnd => "CExpr.logAnd",
            BinOp::LogOr => "CExpr.logOr",
            BinOp::Assign => "CExpr.assign",
            BinOp::AddAssign => "CExpr.addAssign",
            BinOp::SubAssign => "CExpr.subAssign",
            BinOp::MulAssign => "CExpr.mulAssign",
            BinOp::DivAssign => "CExpr.divAssign",
            BinOp::ModAssign => "CExpr.modAssign",
            BinOp::BitAndAssign => "CExpr.bitAndAssign",
            BinOp::BitOrAssign => "CExpr.bitOrAssign",
            BinOp::BitXorAssign => "CExpr.bitXorAssign",
            BinOp::ShlAssign => "CExpr.shlAssign",
            BinOp::ShrAssign => "CExpr.shrAssign",
            BinOp::Comma => "CExpr.comma",
        }
    }

    fn translate_unaryop(&self, op: UnaryOp) -> &'static str {
        match op {
            UnaryOp::Neg => "CExpr.neg",
            UnaryOp::Pos => "CExpr.pos",
            UnaryOp::BitNot => "CExpr.bitNot",
            UnaryOp::LogNot => "CExpr.logNot",
            UnaryOp::Deref => "CExpr.deref",
            UnaryOp::AddrOf => "CExpr.addrOf",
            UnaryOp::PreInc => "CExpr.preInc",
            UnaryOp::PreDec => "CExpr.preDec",
            UnaryOp::PostInc => "CExpr.postInc",
            UnaryOp::PostDec => "CExpr.postDec",
        }
    }

    fn translate_expr_list(&mut self, exprs: &[CExpr]) -> Expr {
        let mut result = Expr::const_(name("List.nil"), vec![]);
        for expr in exprs.iter().rev() {
            let e = self.translate_expr(expr);
            result = Expr::App(
                Arc::new(Expr::App(
                    Arc::new(Expr::const_(name("List.cons"), vec![])),
                    Arc::new(e),
                )),
                Arc::new(result),
            );
        }
        result
    }

    /// Translate a C statement to a Lean5 term
    ///
    /// Statements are translated to monadic operations: State → State × Result Unit
    pub fn translate_stmt(&mut self, stmt: &CStmt) -> Expr {
        match stmt {
            CStmt::Empty => Expr::const_(name("CStmt.skip"), vec![]),

            CStmt::Expr(e) => {
                let e_expr = self.translate_expr(e);
                Expr::App(
                    Arc::new(Expr::const_(name("CStmt.expr"), vec![])),
                    Arc::new(e_expr),
                )
            }

            CStmt::Decl(decl) => {
                let name_expr = self.translate_string(&decl.name);
                let ty_expr = self.translate_type(&decl.ty);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name("CStmt.decl"), vec![])),
                        Arc::new(name_expr),
                    )),
                    Arc::new(ty_expr),
                )
            }

            CStmt::Block(stmts) => {
                let stmts_expr = self.translate_stmt_list(stmts);
                Expr::App(
                    Arc::new(Expr::const_(name("CStmt.block"), vec![])),
                    Arc::new(stmts_expr),
                )
            }

            CStmt::If {
                cond,
                then_stmt,
                else_stmt,
            } => {
                let cond_expr = self.translate_expr(cond);
                let then_expr = self.translate_stmt(then_stmt);
                let else_expr = else_stmt.as_ref().map_or_else(
                    || Expr::const_(name("CStmt.skip"), vec![]),
                    |s| self.translate_stmt(s),
                );
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::App(
                            Arc::new(Expr::const_(name("CStmt.if"), vec![])),
                            Arc::new(cond_expr),
                        )),
                        Arc::new(then_expr),
                    )),
                    Arc::new(else_expr),
                )
            }

            CStmt::While { cond, body } => {
                let cond_expr = self.translate_expr(cond);
                let body_expr = self.translate_stmt(body);
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name("CStmt.while"), vec![])),
                        Arc::new(cond_expr),
                    )),
                    Arc::new(body_expr),
                )
            }

            CStmt::Return(expr) => {
                let val_expr = expr.as_ref().map_or_else(
                    || Expr::const_(name("CValue.unit"), vec![]),
                    |e| self.translate_expr(e),
                );
                Expr::App(
                    Arc::new(Expr::const_(name("CStmt.return"), vec![])),
                    Arc::new(val_expr),
                )
            }

            CStmt::Break => Expr::const_(name("CStmt.break"), vec![]),

            CStmt::Continue => Expr::const_(name("CStmt.continue"), vec![]),

            _ => {
                // Default: unsupported statement
                Expr::const_(name("CStmt.unsupported"), vec![])
            }
        }
    }

    fn translate_stmt_list(&mut self, stmts: &[CStmt]) -> Expr {
        let mut result = Expr::const_(name("List.nil"), vec![]);
        for stmt in stmts.iter().rev() {
            let s = self.translate_stmt(stmt);
            result = Expr::App(
                Arc::new(Expr::App(
                    Arc::new(Expr::const_(name("List.cons"), vec![])),
                    Arc::new(s),
                )),
                Arc::new(result),
            );
        }
        result
    }

    /// Translate a specification to a Lean5 proposition
    pub fn translate_spec(&mut self, spec: &Spec) -> Expr {
        match spec {
            Spec::True => Expr::const_(name("True"), vec![]),

            Spec::False => Expr::const_(name("False"), vec![]),

            Spec::Result => Expr::const_(name("Spec.result"), vec![]),

            Spec::Var(var_name) => {
                if let Some(&level) = self.var_levels.get(var_name) {
                    let index = self.current_level - level - 1;
                    Expr::BVar(index)
                } else {
                    Expr::const_(name(&format!("spec.{var_name}")), vec![])
                }
            }

            Spec::Int(n) => self.translate_int(*n),

            Spec::Expr(e) => self.translate_expr(e),

            Spec::Old(e) => {
                let e_expr = self.translate_spec(e);
                Expr::App(
                    Arc::new(Expr::const_(name("Spec.old"), vec![])),
                    Arc::new(e_expr),
                )
            }

            Spec::And(specs) => {
                if specs.is_empty() {
                    return Expr::const_(name("True"), vec![]);
                }
                let mut result = self.translate_spec(&specs[0]);
                for spec in &specs[1..] {
                    let s = self.translate_spec(spec);
                    result = Expr::App(
                        Arc::new(Expr::App(
                            Arc::new(Expr::const_(name("And"), vec![])),
                            Arc::new(result),
                        )),
                        Arc::new(s),
                    );
                }
                result
            }

            Spec::Or(specs) => {
                if specs.is_empty() {
                    return Expr::const_(name("False"), vec![]);
                }
                let mut result = self.translate_spec(&specs[0]);
                for spec in &specs[1..] {
                    let s = self.translate_spec(spec);
                    result = Expr::App(
                        Arc::new(Expr::App(
                            Arc::new(Expr::const_(name("Or"), vec![])),
                            Arc::new(result),
                        )),
                        Arc::new(s),
                    );
                }
                result
            }

            Spec::Not(s) => {
                let s_expr = self.translate_spec(s);
                Expr::App(
                    Arc::new(Expr::const_(name("Not"), vec![])),
                    Arc::new(s_expr),
                )
            }

            Spec::Implies(p, q) => {
                let p_expr = self.translate_spec(p);
                let q_expr = self.translate_spec(q);
                // Implication as Pi type: P → Q
                Expr::Pi(
                    lean5_kernel::BinderInfo::Default,
                    Arc::new(p_expr),
                    Arc::new(q_expr),
                )
            }

            Spec::Forall { var, ty, body } => {
                let ty_expr = self.translate_type(ty);
                // Bind variable
                self.var_levels.insert(var.clone(), self.current_level);
                self.current_level += 1;
                let body_expr = self.translate_spec(body);
                self.current_level -= 1;
                self.var_levels.remove(var);

                // ∀ x : ty, body
                Expr::Pi(
                    lean5_kernel::BinderInfo::Default,
                    Arc::new(ty_expr),
                    Arc::new(body_expr),
                )
            }

            Spec::Exists { var, ty, body } => {
                let ty_expr = self.translate_type(ty);
                self.var_levels.insert(var.clone(), self.current_level);
                self.current_level += 1;
                let body_expr = self.translate_spec(body);
                self.current_level -= 1;
                self.var_levels.remove(var);

                // Exists as Sigma type
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name("Exists"), vec![])),
                        Arc::new(ty_expr.clone()),
                    )),
                    Arc::new(Expr::Lam(
                        lean5_kernel::BinderInfo::Default,
                        Arc::new(ty_expr),
                        Arc::new(body_expr),
                    )),
                )
            }

            Spec::BinOp { op, left, right } => {
                let l = self.translate_spec(left);
                let r = self.translate_spec(right);
                let op_name = match op {
                    BinOp::Eq => "Eq",
                    BinOp::Ne => "Ne",
                    BinOp::Lt => "LT.lt",
                    BinOp::Le => "LE.le",
                    BinOp::Gt => "GT.gt",
                    BinOp::Ge => "GE.ge",
                    BinOp::Add => "HAdd.hAdd",
                    BinOp::Sub => "HSub.hSub",
                    BinOp::Mul => "HMul.hMul",
                    BinOp::Div => "HDiv.hDiv",
                    _ => "Spec.binop",
                };
                Expr::App(
                    Arc::new(Expr::App(
                        Arc::new(Expr::const_(name(op_name), vec![])),
                        Arc::new(l),
                    )),
                    Arc::new(r),
                )
            }

            Spec::Valid(ptr) => {
                let ptr_expr = self.translate_spec(ptr);
                Expr::App(
                    Arc::new(Expr::const_(name("Spec.valid"), vec![])),
                    Arc::new(ptr_expr),
                )
            }

            _ => {
                // Default: unsupported spec
                Expr::const_(name("Spec.unsupported"), vec![])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_int_type() {
        let ctx = TranslationContext::new();
        let ty = CType::int();
        let expr = ctx.translate_type(&ty);

        // Should produce CType.int IntKind.int Signedness.signed
        assert!(matches!(expr, Expr::App(_, _)));
    }

    #[test]
    fn test_translate_pointer_type() {
        let ctx = TranslationContext::new();
        let ty = CType::ptr(CType::int());
        let expr = ctx.translate_type(&ty);

        assert!(matches!(expr, Expr::App(_, _)));
    }

    #[test]
    fn test_translate_int_lit() {
        let mut ctx = TranslationContext::new();
        let e = CExpr::int(42);
        let expr = ctx.translate_expr(&e);

        assert!(matches!(expr, Expr::App(_, _)));
    }

    #[test]
    fn test_translate_binop() {
        let mut ctx = TranslationContext::new();
        let e = CExpr::add(CExpr::int(1), CExpr::int(2));
        let expr = ctx.translate_expr(&e);

        // Should produce CExpr.add (CValue.int 1) (CValue.int 2)
        assert!(matches!(expr, Expr::App(_, _)));
    }

    #[test]
    fn test_translate_if_stmt() {
        let mut ctx = TranslationContext::new();
        let stmt = CStmt::if_stmt(CExpr::int(1), CStmt::return_stmt(Some(CExpr::int(0))));
        let expr = ctx.translate_stmt(&stmt);

        assert!(matches!(expr, Expr::App(_, _)));
    }

    #[test]
    fn test_translate_forall_spec() {
        let mut ctx = TranslationContext::new();
        let spec = Spec::forall("i", CType::int(), Spec::ge(Spec::var("i"), Spec::int(0)));
        let expr = ctx.translate_spec(&spec);

        // Should produce Pi type
        assert!(matches!(expr, Expr::Pi(_, _, _)));
    }

    #[test]
    fn test_translate_and_spec() {
        let mut ctx = TranslationContext::new();
        let spec = Spec::and(vec![
            Spec::ge(Spec::var("x"), Spec::int(0)),
            Spec::le(Spec::var("x"), Spec::int(10)),
        ]);
        let expr = ctx.translate_spec(&spec);

        // Should produce nested And
        assert!(matches!(expr, Expr::App(_, _)));
    }

    #[test]
    fn test_translate_int_negative_one() {
        // Test a simple negative value
        let ctx = TranslationContext::new();
        let expr = ctx.translate_int(-1);

        // Should produce Int.negOfNat applied to 1
        assert!(matches!(expr, Expr::App(_, _)));
    }

    #[test]
    fn test_translate_int_negative_large() {
        // Test a larger negative value to ensure magnitude calculation is correct
        let ctx = TranslationContext::new();
        let expr = ctx.translate_int(-100);

        // Should produce Int.negOfNat applied to 100
        assert!(matches!(expr, Expr::App(_, _)));
    }

    #[test]
    fn test_translate_int_i64_min_magnitude_calculation() {
        // Verify that i64::MIN magnitude is calculated correctly without overflow.
        // We can't actually call translate_int(i64::MIN) because translate_nat
        // uses recursion and would stack overflow on such large values.
        // Instead, test the magnitude calculation logic directly.

        // For negative n, magnitude = 0u64.wrapping_sub(n as u64)
        let n = i64::MIN;
        let magnitude = 0u64.wrapping_sub(n as u64);

        // i64::MIN = -9223372036854775808
        // i64::MIN as u64 = 9223372036854775808 (two's complement)
        // 0u64.wrapping_sub(9223372036854775808) = 9223372036854775808
        assert_eq!(magnitude, 9_223_372_036_854_775_808_u64);

        // Test -1 similarly
        let n = -1i64;
        let magnitude = 0u64.wrapping_sub(n as u64);
        assert_eq!(magnitude, 1u64);

        // Test -5
        let n = -5i64;
        let magnitude = 0u64.wrapping_sub(n as u64);
        assert_eq!(magnitude, 5u64);
    }
}
