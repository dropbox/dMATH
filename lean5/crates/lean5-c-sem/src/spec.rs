//! ACSL-Style Specification Language
//!
//! This module provides specification constructs inspired by ACSL (ANSI/ISO C
//! Specification Language) from Frama-C. These are used to express:
//!
//! - Pre/postconditions (requires/ensures)
//! - Loop invariants
//! - Memory footprints (assigns/reads)
//! - Ghost variables and assertions
//!
//! ## Example
//!
//! ```ignore
//! // /*@ requires n >= 0;
//! //     ensures \result >= 0;
//! //     ensures \result * \result <= n;
//! // */
//! // int isqrt(int n) { ... }
//!
//! let spec = FuncSpec {
//!     requires: vec![Spec::binop(BinOp::Ge, Spec::var("n"), Spec::int(0))],
//!     ensures: vec![
//!         Spec::binop(BinOp::Ge, Spec::result(), Spec::int(0)),
//!     ],
//!     ..Default::default()
//! };
//! ```

use crate::expr::{BinOp, CExpr, Ident, UnaryOp};
use crate::types::CType;
use serde::{Deserialize, Serialize};

/// A specification expression (extends C expressions with logic constructs)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Spec {
    /// C expression
    Expr(CExpr),

    /// Logical true
    True,

    /// Logical false
    False,

    /// Result of function (\result in ACSL)
    Result,

    /// Old value (\old(e) - value at function entry)
    Old(Box<Spec>),

    /// At label (\at(e, label))
    At { expr: Box<Spec>, label: String },

    /// Forall quantifier: \forall ty x; P(x)
    Forall {
        var: Ident,
        ty: CType,
        body: Box<Spec>,
    },

    /// Exists quantifier: \exists ty x; P(x)
    Exists {
        var: Ident,
        ty: CType,
        body: Box<Spec>,
    },

    /// Implication: P ==> Q
    Implies(Box<Spec>, Box<Spec>),

    /// Bi-implication: P <==> Q
    Iff(Box<Spec>, Box<Spec>),

    /// Logical conjunction (can have > 2 args)
    And(Vec<Spec>),

    /// Logical disjunction
    Or(Vec<Spec>),

    /// Logical negation
    Not(Box<Spec>),

    /// Pointer validity: \valid(p)
    Valid(Box<Spec>),

    /// Pointer validity for read: \valid_read(p)
    ValidRead(Box<Spec>),

    /// Pointer range validity: \valid(p + (lo..hi))
    ValidRange {
        ptr: Box<Spec>,
        lo: Box<Spec>,
        hi: Box<Spec>,
    },

    /// Separation: \separated(p1, p2, ...)
    Separated(Vec<Spec>),

    /// Freshness: \fresh(p)
    Fresh(Box<Spec>),

    /// Freeable: \freeable(p)
    Freeable(Box<Spec>),

    /// Null pointer: \null
    Null,

    /// Block length: \block_length(p)
    BlockLength(Box<Spec>),

    /// Offset: \offset(p)
    Offset(Box<Spec>),

    /// Base address: \base_addr(p)
    BaseAddr(Box<Spec>),

    /// Let binding: \let x = e; body
    Let {
        var: Ident,
        value: Box<Spec>,
        body: Box<Spec>,
    },

    /// Conditional: P ? Q : R
    If {
        cond: Box<Spec>,
        then_spec: Box<Spec>,
        else_spec: Box<Spec>,
    },

    /// Comparison
    BinOp {
        op: BinOp,
        left: Box<Spec>,
        right: Box<Spec>,
    },

    /// Unary operation
    UnaryOp { op: UnaryOp, operand: Box<Spec> },

    /// Integer literal
    Int(i64),

    /// Variable reference
    Var(Ident),

    /// Function call (in logic)
    Call { func: Ident, args: Vec<Spec> },

    /// Array/pointer subscript
    Index { base: Box<Spec>, index: Box<Spec> },

    /// Member access
    Member { object: Box<Spec>, field: Ident },

    /// Sum: \sum(lo, hi, lambda)
    Sum {
        lo: Box<Spec>,
        hi: Box<Spec>,
        var: Ident,
        body: Box<Spec>,
    },

    /// Product: \product(lo, hi, lambda)
    Product {
        lo: Box<Spec>,
        hi: Box<Spec>,
        var: Ident,
        body: Box<Spec>,
    },

    /// Min over range: \min(lo, hi, lambda)
    Min {
        lo: Box<Spec>,
        hi: Box<Spec>,
        var: Ident,
        body: Box<Spec>,
    },

    /// Max over range: \max(lo, hi, lambda)
    Max {
        lo: Box<Spec>,
        hi: Box<Spec>,
        var: Ident,
        body: Box<Spec>,
    },

    /// Count satisfying predicate: \numof(lo, hi, lambda)
    NumOf {
        lo: Box<Spec>,
        hi: Box<Spec>,
        var: Ident,
        body: Box<Spec>,
    },
}

impl Spec {
    // Constructors

    pub fn true_() -> Self {
        Spec::True
    }

    pub fn false_() -> Self {
        Spec::False
    }

    pub fn result() -> Self {
        Spec::Result
    }

    pub fn old(e: Spec) -> Self {
        Spec::Old(Box::new(e))
    }

    pub fn var(name: impl Into<String>) -> Self {
        Spec::Var(name.into())
    }

    pub fn int(val: i64) -> Self {
        Spec::Int(val)
    }

    pub fn expr(e: CExpr) -> Self {
        Spec::Expr(e)
    }

    pub fn and(specs: Vec<Spec>) -> Self {
        if specs.is_empty() {
            Spec::True
        } else if specs.len() == 1 {
            specs.into_iter().next().unwrap()
        } else {
            Spec::And(specs)
        }
    }

    pub fn or(specs: Vec<Spec>) -> Self {
        if specs.is_empty() {
            Spec::False
        } else if specs.len() == 1 {
            specs.into_iter().next().unwrap()
        } else {
            Spec::Or(specs)
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(s: Spec) -> Self {
        Spec::Not(Box::new(s))
    }

    pub fn implies(p: Spec, q: Spec) -> Self {
        Spec::Implies(Box::new(p), Box::new(q))
    }

    pub fn iff(p: Spec, q: Spec) -> Self {
        Spec::Iff(Box::new(p), Box::new(q))
    }

    pub fn forall(var: impl Into<String>, ty: CType, body: Spec) -> Self {
        Spec::Forall {
            var: var.into(),
            ty,
            body: Box::new(body),
        }
    }

    pub fn exists(var: impl Into<String>, ty: CType, body: Spec) -> Self {
        Spec::Exists {
            var: var.into(),
            ty,
            body: Box::new(body),
        }
    }

    pub fn valid(ptr: Spec) -> Self {
        Spec::Valid(Box::new(ptr))
    }

    pub fn valid_read(ptr: Spec) -> Self {
        Spec::ValidRead(Box::new(ptr))
    }

    pub fn binop(op: BinOp, left: Spec, right: Spec) -> Self {
        Spec::BinOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a comparison
    pub fn eq(left: Spec, right: Spec) -> Self {
        Self::binop(BinOp::Eq, left, right)
    }

    pub fn ne(left: Spec, right: Spec) -> Self {
        Self::binop(BinOp::Ne, left, right)
    }

    pub fn lt(left: Spec, right: Spec) -> Self {
        Self::binop(BinOp::Lt, left, right)
    }

    pub fn le(left: Spec, right: Spec) -> Self {
        Self::binop(BinOp::Le, left, right)
    }

    pub fn gt(left: Spec, right: Spec) -> Self {
        Self::binop(BinOp::Gt, left, right)
    }

    pub fn ge(left: Spec, right: Spec) -> Self {
        Self::binop(BinOp::Ge, left, right)
    }
}

/// Memory footprint specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Location {
    /// A single location: *p
    Deref(Spec),
    /// A range: p[lo..hi]
    Range { base: Spec, lo: Spec, hi: Spec },
    /// All memory reachable from p
    Reachable(Spec),
    /// Nothing
    Nothing,
    /// Everything (default)
    Everything,
}

/// Function behavior (for named behaviors in ACSL)
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Behavior {
    /// Behavior name
    pub name: Option<String>,
    /// Assumes clause (when this behavior applies)
    pub assumes: Vec<Spec>,
    /// Requires clause (preconditions)
    pub requires: Vec<Spec>,
    /// Ensures clause (postconditions)
    pub ensures: Vec<Spec>,
    /// Assigns clause (memory locations that may be modified)
    pub assigns: Vec<Location>,
}

/// Function specification
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FuncSpec {
    /// Formal parameter names (for substitution in interprocedural analysis)
    pub params: Vec<String>,
    /// Preconditions (all must hold)
    pub requires: Vec<Spec>,
    /// Postconditions (all must hold on normal return)
    pub ensures: Vec<Spec>,
    /// Memory locations that may be modified
    pub assigns: Vec<Location>,
    /// Memory locations that may be read
    pub reads: Vec<Location>,
    /// Termination: does the function always terminate?
    pub terminates: Option<Spec>,
    /// Named behaviors
    pub behaviors: Vec<Behavior>,
    /// Complete behaviors: together they cover all cases
    pub complete: Vec<Vec<String>>,
    /// Disjoint behaviors: no overlap between them
    pub disjoint: Vec<Vec<String>>,
}

/// Loop specification
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LoopSpec {
    /// Loop invariants
    pub invariant: Vec<Spec>,
    /// Loop variant (for termination)
    pub variant: Option<Spec>,
    /// Assigns clause
    pub assigns: Vec<Location>,
}

/// Statement annotation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Annotation {
    /// Assert: //@ assert P;
    Assert(Spec),
    /// Assume: //@ assume P;
    Assume(Spec),
    /// Loop specification
    Loop(LoopSpec),
    /// Function specification
    Func(FuncSpec),
    /// Ghost variable declaration
    Ghost {
        name: Ident,
        ty: CType,
        init: Option<Spec>,
    },
    /// Ghost assignment
    GhostAssign { target: Spec, value: Spec },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_spec() {
        // requires n >= 0
        let req = Spec::ge(Spec::var("n"), Spec::int(0));
        assert!(matches!(req, Spec::BinOp { op: BinOp::Ge, .. }));
    }

    #[test]
    fn test_forall() {
        // \forall int i; 0 <= i < n ==> a[i] >= 0
        let spec = Spec::forall(
            "i",
            CType::int(),
            Spec::implies(
                Spec::and(vec![
                    Spec::ge(Spec::var("i"), Spec::int(0)),
                    Spec::lt(Spec::var("i"), Spec::var("n")),
                ]),
                Spec::ge(
                    Spec::Index {
                        base: Box::new(Spec::var("a")),
                        index: Box::new(Spec::var("i")),
                    },
                    Spec::int(0),
                ),
            ),
        );

        assert!(matches!(spec, Spec::Forall { .. }));
    }

    #[test]
    fn test_func_spec() {
        // int abs(int x);
        // requires true;
        // ensures \result >= 0;
        // ensures \result == x || \result == -x;
        let spec = FuncSpec {
            requires: vec![Spec::true_()],
            ensures: vec![
                Spec::ge(Spec::result(), Spec::int(0)),
                Spec::or(vec![
                    Spec::eq(Spec::result(), Spec::var("x")),
                    Spec::eq(
                        Spec::result(),
                        Spec::UnaryOp {
                            op: UnaryOp::Neg,
                            operand: Box::new(Spec::var("x")),
                        },
                    ),
                ]),
            ],
            ..Default::default()
        };

        assert_eq!(spec.requires.len(), 1);
        assert_eq!(spec.ensures.len(), 2);
    }

    #[test]
    fn test_loop_spec() {
        // loop invariant 0 <= i <= n;
        // loop invariant sum == \sum(0, i, \lambda j; a[j]);
        // loop variant n - i;
        let spec = LoopSpec {
            invariant: vec![Spec::and(vec![
                Spec::ge(Spec::var("i"), Spec::int(0)),
                Spec::le(Spec::var("i"), Spec::var("n")),
            ])],
            variant: Some(Spec::binop(BinOp::Sub, Spec::var("n"), Spec::var("i"))),
            ..Default::default()
        };

        assert_eq!(spec.invariant.len(), 1);
        assert!(spec.variant.is_some());
    }

    #[test]
    fn test_memory_spec() {
        // assigns a[0..n-1];
        let location = Location::Range {
            base: Spec::var("a"),
            lo: Spec::int(0),
            hi: Spec::binop(BinOp::Sub, Spec::var("n"), Spec::int(1)),
        };

        assert!(matches!(location, Location::Range { .. }));
    }

    #[test]
    fn test_old() {
        // ensures \result == \old(x) + 1;
        let spec = Spec::eq(
            Spec::result(),
            Spec::binop(BinOp::Add, Spec::old(Spec::var("x")), Spec::int(1)),
        );

        match &spec {
            Spec::BinOp {
                op: BinOp::Eq,
                right,
                ..
            } => match right.as_ref() {
                Spec::BinOp { left, .. } => {
                    assert!(matches!(left.as_ref(), Spec::Old(_)));
                }
                _ => panic!("Expected binop"),
            },
            _ => panic!("Expected eq"),
        }
    }
}
