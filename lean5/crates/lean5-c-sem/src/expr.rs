//! C Expression Semantics
//!
//! This module defines the abstract syntax and evaluation of C expressions.
//!
//! ## Expression Categories
//!
//! C expressions are categorized by their value category:
//!
//! - **lvalue**: Designates an object (can be assigned to)
//! - **rvalue**: A value that doesn't designate an object
//!
//! This distinction is crucial for correct semantics of:
//! - Assignment (LHS must be lvalue)
//! - Address-of operator (operand must be lvalue)
//! - Increment/decrement (operand must be lvalue)
//!
//! ## Side Effects
//!
//! C expressions can have side effects:
//! - Assignment operators
//! - Increment/decrement
//! - Function calls
//!
//! The order of side effects is partially determined by sequence points.

use crate::memory::Pointer;
use crate::types::CType;
use crate::values::CValue;
use serde::{Deserialize, Serialize};

/// Identifier (variable/function name)
pub type Ident = String;

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    // Arithmetic
    Add, // +
    Sub, // -
    Mul, // *
    Div, // /
    Mod, // %

    // Bitwise
    BitAnd, // &
    BitOr,  // |
    BitXor, // ^
    Shl,    // <<
    Shr,    // >>

    // Comparison
    Eq, // ==
    Ne, // !=
    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=

    // Logical
    LogAnd, // &&
    LogOr,  // ||

    // Assignment (returns new value)
    Assign,       // =
    AddAssign,    // +=
    SubAssign,    // -=
    MulAssign,    // *=
    DivAssign,    // /=
    ModAssign,    // %=
    BitAndAssign, // &=
    BitOrAssign,  // |=
    BitXorAssign, // ^=
    ShlAssign,    // <<=
    ShrAssign,    // >>=

    // Comma (evaluates left, discards, returns right)
    Comma, // ,
}

impl BinOp {
    /// Check if this is an assignment operator
    pub fn is_assignment(&self) -> bool {
        matches!(
            self,
            BinOp::Assign
                | BinOp::AddAssign
                | BinOp::SubAssign
                | BinOp::MulAssign
                | BinOp::DivAssign
                | BinOp::ModAssign
                | BinOp::BitAndAssign
                | BinOp::BitOrAssign
                | BinOp::BitXorAssign
                | BinOp::ShlAssign
                | BinOp::ShrAssign
        )
    }

    /// Check if this is a comparison operator
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }

    /// Check if this is a logical operator (short-circuit)
    pub fn is_logical(&self) -> bool {
        matches!(self, BinOp::LogAnd | BinOp::LogOr)
    }
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    // Arithmetic
    Neg, // -
    Pos, // + (no-op for arithmetic)

    // Bitwise
    BitNot, // ~

    // Logical
    LogNot, // !

    // Pointer
    Deref,  // *
    AddrOf, // &

    // Increment/Decrement
    PreInc,  // ++x
    PreDec,  // --x
    PostInc, // x++
    PostDec, // x--

             // Sizeof (handled specially)
             // Cast (handled as separate expression kind)
}

impl UnaryOp {
    /// Check if this operator requires an lvalue operand
    pub fn requires_lvalue(&self) -> bool {
        matches!(
            self,
            UnaryOp::AddrOf
                | UnaryOp::PreInc
                | UnaryOp::PreDec
                | UnaryOp::PostInc
                | UnaryOp::PostDec
        )
    }

    /// Check if this operator is an increment/decrement
    pub fn is_inc_dec(&self) -> bool {
        matches!(
            self,
            UnaryOp::PreInc | UnaryOp::PreDec | UnaryOp::PostInc | UnaryOp::PostDec
        )
    }
}

/// C Expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CExpr {
    /// Integer literal
    IntLit(i64),

    /// Unsigned integer literal
    UIntLit(u64),

    /// Float literal
    FloatLit(f64),

    /// Character literal
    CharLit(u8),

    /// String literal (returns pointer to static string)
    StringLit(String),

    /// Variable reference
    Var(Ident),

    /// Binary operation
    BinOp {
        op: BinOp,
        left: Box<CExpr>,
        right: Box<CExpr>,
    },

    /// Unary operation
    UnaryOp { op: UnaryOp, operand: Box<CExpr> },

    /// Conditional/ternary expression: cond ? then_expr : else_expr
    Conditional {
        cond: Box<CExpr>,
        then_expr: Box<CExpr>,
        else_expr: Box<CExpr>,
    },

    /// Type cast: (type)expr
    Cast { ty: CType, expr: Box<CExpr> },

    /// Sizeof expression: sizeof(expr) or sizeof(type)
    SizeOf(SizeOfArg),

    /// Alignof expression (C11): _Alignof(type)
    AlignOf(CType),

    /// Function call: func(args...)
    Call { func: Box<CExpr>, args: Vec<CExpr> },

    /// Array subscript: `arr[index]`
    Index {
        array: Box<CExpr>,
        index: Box<CExpr>,
    },

    /// Struct/union member access: s.field
    Member { object: Box<CExpr>, field: Ident },

    /// Pointer member access: p->field (equivalent to (*p).field)
    Arrow { pointer: Box<CExpr>, field: Ident },

    /// Compound literal (C99): (type){initializers}
    CompoundLiteral { ty: CType, init: Vec<Initializer> },

    /// Generic selection (C11): _Generic(expr, type1: expr1, ...)
    Generic {
        control: Box<CExpr>,
        associations: Vec<(Option<CType>, CExpr)>, // None = default
    },

    /// Statement expression (GCC extension): ({ stmts; expr })
    StmtExpr(Vec<crate::stmt::CStmt>),
}

/// Argument to sizeof operator
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SizeOfArg {
    /// sizeof(type)
    Type(CType),
    /// sizeof(expr) - without evaluating expr
    Expr(Box<CExpr>),
}

/// Initializer for compound literals and variable init
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Initializer {
    /// Simple expression initializer
    Expr(CExpr),

    /// Designated initializer: `.field = value` or `[index] = value`
    Designated {
        designator: Designator,
        init: Box<Initializer>,
    },

    /// Brace-enclosed initializer list
    List(Vec<Initializer>),
}

/// Designator for designated initializers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Designator {
    /// .field
    Field(Ident),
    /// `[index]`
    Index(Box<CExpr>),
    /// Chained: `.field[index].subfield`
    Chain(Vec<Designator>),
}

impl CExpr {
    // Constructors for convenience

    /// Create an integer literal
    pub fn int(val: i64) -> Self {
        CExpr::IntLit(val)
    }

    /// Create a null pointer (casts 0 to void*)
    pub fn null() -> Self {
        CExpr::Cast {
            ty: CType::Pointer(Box::new(CType::Void)),
            expr: Box::new(CExpr::IntLit(0)),
        }
    }

    /// Create an unsigned integer literal
    pub fn uint(val: u64) -> Self {
        CExpr::UIntLit(val)
    }

    /// Create a float literal
    pub fn float(val: f64) -> Self {
        CExpr::FloatLit(val)
    }

    /// Create a variable reference
    pub fn var(name: impl Into<String>) -> Self {
        CExpr::Var(name.into())
    }

    /// Create a binary operation
    pub fn binop(op: BinOp, left: CExpr, right: CExpr) -> Self {
        CExpr::BinOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a unary operation
    pub fn unary(op: UnaryOp, operand: CExpr) -> Self {
        CExpr::UnaryOp {
            op,
            operand: Box::new(operand),
        }
    }

    /// Create an addition
    #[allow(clippy::should_implement_trait)]
    pub fn add(left: CExpr, right: CExpr) -> Self {
        Self::binop(BinOp::Add, left, right)
    }

    /// Create a subtraction
    #[allow(clippy::should_implement_trait)]
    pub fn sub(left: CExpr, right: CExpr) -> Self {
        Self::binop(BinOp::Sub, left, right)
    }

    /// Create a multiplication
    #[allow(clippy::should_implement_trait)]
    pub fn mul(left: CExpr, right: CExpr) -> Self {
        Self::binop(BinOp::Mul, left, right)
    }

    /// Create a division
    #[allow(clippy::should_implement_trait)]
    pub fn div(left: CExpr, right: CExpr) -> Self {
        Self::binop(BinOp::Div, left, right)
    }

    /// Create an assignment
    pub fn assign(lhs: CExpr, rhs: CExpr) -> Self {
        Self::binop(BinOp::Assign, lhs, rhs)
    }

    /// Create a function call
    pub fn call(func: CExpr, args: Vec<CExpr>) -> Self {
        CExpr::Call {
            func: Box::new(func),
            args,
        }
    }

    /// Create an array index
    pub fn index(array: CExpr, idx: CExpr) -> Self {
        CExpr::Index {
            array: Box::new(array),
            index: Box::new(idx),
        }
    }

    /// Create a member access
    pub fn member(object: CExpr, field: impl Into<String>) -> Self {
        CExpr::Member {
            object: Box::new(object),
            field: field.into(),
        }
    }

    /// Create an arrow (pointer member) access
    pub fn arrow(pointer: CExpr, field: impl Into<String>) -> Self {
        CExpr::Arrow {
            pointer: Box::new(pointer),
            field: field.into(),
        }
    }

    /// Create a dereference
    pub fn deref(ptr: CExpr) -> Self {
        Self::unary(UnaryOp::Deref, ptr)
    }

    /// Create an address-of
    pub fn addr_of(operand: CExpr) -> Self {
        Self::unary(UnaryOp::AddrOf, operand)
    }

    /// Create a cast
    pub fn cast(ty: CType, expr: CExpr) -> Self {
        CExpr::Cast {
            ty,
            expr: Box::new(expr),
        }
    }

    /// Create a conditional expression
    pub fn conditional(cond: CExpr, then_expr: CExpr, else_expr: CExpr) -> Self {
        CExpr::Conditional {
            cond: Box::new(cond),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        }
    }

    /// Check if this expression is an lvalue
    ///
    /// lvalues designate objects and can appear on the LHS of assignment
    pub fn is_lvalue(&self) -> bool {
        match self {
            // Variables, dereference, subscript, arrow, compound literals (C99), and string literals are lvalues
            CExpr::Var(_)
            | CExpr::UnaryOp {
                op: UnaryOp::Deref, ..
            }
            | CExpr::Index { .. }
            | CExpr::Arrow { .. }
            | CExpr::CompoundLiteral { .. }
            | CExpr::StringLit(_) => true,

            // Member access is lvalue if object is lvalue
            CExpr::Member { object, .. } => object.is_lvalue(),

            // Everything else is an rvalue
            _ => false,
        }
    }

    /// Check if this expression has side effects
    pub fn has_side_effects(&self) -> bool {
        match self {
            // Literals, variables, sizeof/alignof don't have side effects
            CExpr::IntLit(_)
            | CExpr::UIntLit(_)
            | CExpr::FloatLit(_)
            | CExpr::CharLit(_)
            | CExpr::StringLit(_)
            | CExpr::Var(_)
            | CExpr::SizeOf(_)
            | CExpr::AlignOf(_) => false,

            CExpr::BinOp { op, left, right } => {
                op.is_assignment() || left.has_side_effects() || right.has_side_effects()
            }

            CExpr::UnaryOp { op, operand } => op.is_inc_dec() || operand.has_side_effects(),

            // Function calls and statement expressions always have potential side effects
            CExpr::Call { .. } | CExpr::StmtExpr(_) => true,

            CExpr::Conditional {
                cond,
                then_expr,
                else_expr,
            } => {
                cond.has_side_effects()
                    || then_expr.has_side_effects()
                    || else_expr.has_side_effects()
            }

            CExpr::Cast { expr, .. } => expr.has_side_effects(),

            CExpr::Index { array, index } => array.has_side_effects() || index.has_side_effects(),

            CExpr::Member { object, .. } => object.has_side_effects(),

            CExpr::Arrow { pointer, .. } => pointer.has_side_effects(),

            CExpr::CompoundLiteral { init, .. } => init.iter().any(initializer_has_side_effects),

            CExpr::Generic {
                control,
                associations,
            } => {
                control.has_side_effects() || associations.iter().any(|(_, e)| e.has_side_effects())
            }
        }
    }
}

fn initializer_has_side_effects(init: &Initializer) -> bool {
    match init {
        Initializer::Expr(e) => e.has_side_effects(),
        Initializer::Designated { init, .. } => initializer_has_side_effects(init),
        Initializer::List(inits) => inits.iter().any(initializer_has_side_effects),
    }
}

/// LValue - a typed location that can be read from or written to
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LValue {
    /// Pointer to the location
    pub ptr: Pointer,
    /// Type of the value at this location
    pub ty: CType,
}

impl LValue {
    /// Create a new LValue
    pub fn new(ptr: Pointer, ty: CType) -> Self {
        Self { ptr, ty }
    }

    /// Get the address of this lvalue
    pub fn address(&self) -> CValue {
        CValue::Pointer(self.ptr)
    }
}

/// Result of evaluating an expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExprResult {
    /// An rvalue (just the value)
    RValue(CValue),
    /// An lvalue (location + type)
    LValue(LValue),
}

impl ExprResult {
    /// Create an rvalue result
    pub fn rvalue(val: CValue) -> Self {
        ExprResult::RValue(val)
    }

    /// Create an lvalue result
    pub fn lvalue(ptr: Pointer, ty: CType) -> Self {
        ExprResult::LValue(LValue::new(ptr, ty))
    }

    /// Check if this is an lvalue
    pub fn is_lvalue(&self) -> bool {
        matches!(self, ExprResult::LValue(_))
    }

    /// Get the type of this result
    pub fn get_type(&self, default: &CType) -> CType {
        match self {
            ExprResult::RValue(_) => default.clone(),
            ExprResult::LValue(lv) => lv.ty.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binop_properties() {
        assert!(BinOp::Assign.is_assignment());
        assert!(BinOp::AddAssign.is_assignment());
        assert!(!BinOp::Add.is_assignment());

        assert!(BinOp::Eq.is_comparison());
        assert!(BinOp::Lt.is_comparison());
        assert!(!BinOp::Add.is_comparison());

        assert!(BinOp::LogAnd.is_logical());
        assert!(BinOp::LogOr.is_logical());
        assert!(!BinOp::BitAnd.is_logical());
    }

    #[test]
    fn test_unary_properties() {
        assert!(UnaryOp::AddrOf.requires_lvalue());
        assert!(UnaryOp::PreInc.requires_lvalue());
        assert!(!UnaryOp::Neg.requires_lvalue());

        assert!(UnaryOp::PreInc.is_inc_dec());
        assert!(UnaryOp::PostDec.is_inc_dec());
        assert!(!UnaryOp::Neg.is_inc_dec());
    }

    #[test]
    fn test_expr_is_lvalue() {
        assert!(CExpr::var("x").is_lvalue());
        assert!(CExpr::deref(CExpr::var("p")).is_lvalue());
        assert!(CExpr::index(CExpr::var("arr"), CExpr::int(0)).is_lvalue());
        assert!(CExpr::member(CExpr::var("s"), "field").is_lvalue());
        assert!(CExpr::arrow(CExpr::var("p"), "field").is_lvalue());

        assert!(!CExpr::int(42).is_lvalue());
        assert!(!CExpr::add(CExpr::var("x"), CExpr::var("y")).is_lvalue());
        assert!(!CExpr::call(CExpr::var("f"), vec![]).is_lvalue());
    }

    #[test]
    fn test_expr_has_side_effects() {
        // No side effects
        assert!(!CExpr::int(42).has_side_effects());
        assert!(!CExpr::var("x").has_side_effects());
        assert!(!CExpr::add(CExpr::var("x"), CExpr::var("y")).has_side_effects());

        // Has side effects
        assert!(CExpr::assign(CExpr::var("x"), CExpr::int(1)).has_side_effects());
        assert!(CExpr::unary(UnaryOp::PreInc, CExpr::var("x")).has_side_effects());
        assert!(CExpr::call(CExpr::var("f"), vec![]).has_side_effects());
    }

    #[test]
    fn test_expr_construction() {
        let expr = CExpr::add(CExpr::mul(CExpr::var("x"), CExpr::int(2)), CExpr::var("y"));

        match &expr {
            CExpr::BinOp {
                op: BinOp::Add,
                left,
                right,
            } => {
                assert!(matches!(&**left, CExpr::BinOp { op: BinOp::Mul, .. }));
                assert!(matches!(&**right, CExpr::Var(name) if name == "y"));
            }
            _ => panic!("Wrong expression structure"),
        }
    }

    #[test]
    fn test_conditional() {
        let cond = CExpr::conditional(
            CExpr::binop(BinOp::Lt, CExpr::var("x"), CExpr::int(0)),
            CExpr::unary(UnaryOp::Neg, CExpr::var("x")),
            CExpr::var("x"),
        );

        match cond {
            CExpr::Conditional { .. } => (),
            _ => panic!("Expected conditional"),
        }
    }

    #[test]
    fn test_sizeof() {
        let sizeof_type = CExpr::SizeOf(SizeOfArg::Type(CType::int()));
        let sizeof_expr = CExpr::SizeOf(SizeOfArg::Expr(Box::new(CExpr::var("x"))));

        assert!(!sizeof_type.has_side_effects());
        assert!(!sizeof_expr.has_side_effects());
    }
}
