//! Rust Expression Semantics
//!
//! This module defines the semantics of Rust expressions,
//! including evaluation rules and type checking.
//!
//! ## Expression Categories
//!
//! - **Literals**: Numbers, booleans, strings, characters
//! - **Place Expressions**: Variables, field access, indexing
//! - **Operators**: Arithmetic, logical, comparison
//! - **Control Flow**: if/else, match, loops
//! - **Calls**: Function and method calls
//! - **Blocks**: Sequences of statements with final expression

use crate::types::{Mutability, RustType};
use crate::values::{BinOp, UnOp, Value};
use serde::{Deserialize, Serialize};

/// A Rust expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    /// Literal value
    Literal(Value),

    /// Variable reference (place expression)
    Var { name: String, local_idx: u32 },

    /// Field access: expr.field
    Field { base: Box<Expr>, field: String },

    /// Array/slice indexing: `expr[index]`
    Index { base: Box<Expr>, index: Box<Expr> },

    /// Dereference: *expr
    Deref(Box<Expr>),

    /// Address-of: &expr or &mut expr
    AddrOf {
        mutability: Mutability,
        expr: Box<Expr>,
    },

    /// Binary operation
    BinOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Unary operation
    UnOp { op: UnOp, expr: Box<Expr> },

    /// Type cast: expr as Type
    Cast { expr: Box<Expr>, target: RustType },

    /// Function call
    Call { func: Box<Expr>, args: Vec<Expr> },

    /// Method call: expr.method(args)
    MethodCall {
        receiver: Box<Expr>,
        method: String,
        args: Vec<Expr>,
    },

    /// If expression
    If {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
    },

    /// Match expression
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },

    /// Block expression
    Block {
        stmts: Vec<Stmt>,
        expr: Option<Box<Expr>>,
    },

    /// Tuple construction
    Tuple(Vec<Expr>),

    /// Array construction
    Array(Vec<Expr>),

    /// Array repeat: [expr; count]
    ArrayRepeat { value: Box<Expr>, count: usize },

    /// Struct construction
    Struct {
        name: String,
        fields: Vec<(String, Expr)>,
    },

    /// Enum variant construction
    EnumVariant {
        enum_name: String,
        variant: String,
        payload: EnumVariantPayload,
    },

    /// Closure
    Closure {
        params: Vec<(String, RustType)>,
        body: Box<Expr>,
        captures: Vec<(String, Mutability)>,
    },

    /// Range: start..end or start..=end
    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        inclusive: bool,
    },

    /// Return expression
    Return(Option<Box<Expr>>),

    /// Break expression (with optional value and label)
    Break {
        label: Option<String>,
        value: Option<Box<Expr>>,
    },

    /// Continue expression
    Continue { label: Option<String> },

    /// Loop (infinite)
    Loop {
        label: Option<String>,
        body: Box<Expr>,
    },

    /// While loop
    While {
        label: Option<String>,
        condition: Box<Expr>,
        body: Box<Expr>,
    },

    /// For loop
    For {
        label: Option<String>,
        pattern: Pattern,
        iter: Box<Expr>,
        body: Box<Expr>,
    },
}

/// Enum variant payload for construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnumVariantPayload {
    Unit,
    Tuple(Vec<Expr>),
    Struct(Vec<(String, Expr)>),
}

/// Match arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

/// Pattern for matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Pattern {
    /// Wildcard: _
    Wildcard,

    /// Binding: x or mut x
    Binding {
        name: String,
        mutable: bool,
        /// Optional sub-pattern: x @ pattern
        subpattern: Option<Box<Pattern>>,
    },

    /// Literal pattern
    Literal(Value),

    /// Reference pattern: &pattern or &mut pattern
    Ref {
        mutability: Mutability,
        pattern: Box<Pattern>,
    },

    /// Tuple pattern: (p1, p2, ...)
    Tuple(Vec<Pattern>),

    /// Struct pattern: Struct { field: pattern, .. }
    Struct {
        name: String,
        fields: Vec<(String, Pattern)>,
        rest: bool,
    },

    /// Enum variant pattern
    EnumVariant {
        enum_name: String,
        variant: String,
        payload: EnumPatternPayload,
    },

    /// Or pattern: p1 | p2
    Or(Vec<Pattern>),

    /// Range pattern: start..=end
    Range {
        start: Value,
        end: Value,
        inclusive: bool,
    },
}

/// Enum variant pattern payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnumPatternPayload {
    Unit,
    Tuple(Vec<Pattern>),
    Struct(Vec<(String, Pattern)>),
}

/// Statement (used in blocks)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stmt {
    /// Let binding: let pattern = expr
    Let {
        pattern: Pattern,
        ty: Option<RustType>,
        init: Option<Expr>,
    },

    /// Expression statement (value discarded)
    Expr(Expr),

    /// Item declaration (function, struct, etc.)
    Item(Item),
}

/// Item declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Item {
    /// Function definition
    Fn {
        name: String,
        params: Vec<(String, RustType)>,
        ret: RustType,
        body: Expr,
    },

    /// Struct definition
    Struct {
        name: String,
        fields: Vec<(String, RustType)>,
    },

    /// Enum definition
    Enum {
        name: String,
        variants: Vec<(String, EnumVariantPayload)>,
    },

    /// Impl block
    Impl { self_ty: RustType, items: Vec<Item> },

    /// Const item
    Const {
        name: String,
        ty: RustType,
        value: Expr,
    },

    /// Static item
    Static {
        name: String,
        ty: RustType,
        mutable: bool,
        value: Expr,
    },
}

/// Expression evaluation result
#[derive(Debug, Clone)]
pub enum EvalResult {
    /// Normal value
    Value(Value),
    /// Return from function
    Return(Value),
    /// Break from loop (with optional value)
    Break(Option<Value>),
    /// Continue loop
    Continue,
    /// Error during evaluation
    Error(String),
}

impl EvalResult {
    /// Check if result is a normal value
    pub fn is_value(&self) -> bool {
        matches!(self, EvalResult::Value(_))
    }

    /// Get value if normal
    pub fn value(self) -> Option<Value> {
        match self {
            EvalResult::Value(v) => Some(v),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_expr() {
        let expr = Expr::Literal(Value::u32(42));
        assert!(matches!(expr, Expr::Literal(_)));
    }

    #[test]
    fn test_binop_expr() {
        let expr = Expr::BinOp {
            op: BinOp::Add,
            left: Box::new(Expr::Literal(Value::u32(1))),
            right: Box::new(Expr::Literal(Value::u32(2))),
        };
        assert!(matches!(expr, Expr::BinOp { .. }));
    }

    #[test]
    fn test_if_expr() {
        let expr = Expr::If {
            condition: Box::new(Expr::Literal(Value::Bool(true))),
            then_branch: Box::new(Expr::Literal(Value::u32(1))),
            else_branch: Some(Box::new(Expr::Literal(Value::u32(2)))),
        };
        assert!(matches!(expr, Expr::If { .. }));
    }

    #[test]
    fn test_pattern_matching() {
        let pattern = Pattern::Tuple(vec![
            Pattern::Binding {
                name: "x".to_string(),
                mutable: false,
                subpattern: None,
            },
            Pattern::Wildcard,
        ]);
        assert!(matches!(pattern, Pattern::Tuple(_)));
    }

    #[test]
    fn test_closure_expr() {
        let closure = Expr::Closure {
            params: vec![("x".to_string(), RustType::Uint(crate::types::UintType::U32))],
            body: Box::new(Expr::Var {
                name: "x".to_string(),
                local_idx: 0,
            }),
            captures: vec![],
        };
        assert!(matches!(closure, Expr::Closure { .. }));
    }

    #[test]
    fn test_match_expr() {
        let match_expr = Expr::Match {
            scrutinee: Box::new(Expr::Var {
                name: "opt".to_string(),
                local_idx: 0,
            }),
            arms: vec![
                MatchArm {
                    pattern: Pattern::EnumVariant {
                        enum_name: "Option".to_string(),
                        variant: "Some".to_string(),
                        payload: EnumPatternPayload::Tuple(vec![Pattern::Binding {
                            name: "x".to_string(),
                            mutable: false,
                            subpattern: None,
                        }]),
                    },
                    guard: None,
                    body: Expr::Var {
                        name: "x".to_string(),
                        local_idx: 1,
                    },
                },
                MatchArm {
                    pattern: Pattern::EnumVariant {
                        enum_name: "Option".to_string(),
                        variant: "None".to_string(),
                        payload: EnumPatternPayload::Unit,
                    },
                    guard: None,
                    body: Expr::Literal(Value::u32(0)),
                },
            ],
        };
        assert!(matches!(match_expr, Expr::Match { .. }));
    }
}
