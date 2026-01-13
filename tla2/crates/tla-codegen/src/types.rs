//! Type inference for TLA+ expressions
//!
//! TLA+ is untyped, but for code generation we need to infer Rust types.
//! This module provides a simple type inference system that:
//! - Infers types from expression structure (literals, operators)
//! - Propagates type constraints through the expression tree
//! - Reports errors when types cannot be determined or are inconsistent

use std::collections::HashMap;
use tla_core::ast::{BoundVar, Expr, Module, OperatorDef, Unit};
use tla_core::span::Spanned;

/// Inferred TLA+ types that map to Rust types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TlaType {
    /// Boolean type -> bool
    Bool,
    /// Integer type -> i64 (or BigInt for unbounded)
    Int,
    /// String type -> String
    String,
    /// Set type -> `TlaSet<T>`
    Set(Box<TlaType>),
    /// Sequence type -> `Vec<T>`
    Seq(Box<TlaType>),
    /// Tuple type -> (T1, T2, ...)
    Tuple(Vec<TlaType>),
    /// Record type -> struct
    Record(Vec<(String, TlaType)>),
    /// Function type -> TlaFunc<K, V>
    Func(Box<TlaType>, Box<TlaType>),
    /// Type variable (unresolved)
    Var(usize),
    /// Unknown type (inference failed)
    Unknown,
}

impl TlaType {
    /// Convert TLA+ type to Rust type string
    pub fn to_rust_type(&self) -> String {
        match self {
            TlaType::Bool => "bool".to_string(),
            TlaType::Int => "i64".to_string(),
            TlaType::String => "String".to_string(),
            TlaType::Set(elem) => format!("TlaSet<{}>", elem.to_rust_type()),
            TlaType::Seq(elem) => format!("Vec<{}>", elem.to_rust_type()),
            TlaType::Tuple(elems) => {
                let types: Vec<_> = elems.iter().map(|t| t.to_rust_type()).collect();
                format!("({})", types.join(", "))
            }
            TlaType::Record(fields) => {
                // Use TlaRecord<T> if all fields have the same type, else use tuple
                if fields.is_empty() {
                    "TlaRecord<()>".to_string()
                } else {
                    let first_type = &fields[0].1;
                    if fields.iter().all(|(_, t)| t == first_type) {
                        format!("TlaRecord<{}>", first_type.to_rust_type())
                    } else {
                        // Heterogeneous record - use tuple (may need runtime support later)
                        let types: Vec<_> = fields.iter().map(|(_, t)| t.to_rust_type()).collect();
                        format!("({})", types.join(", "))
                    }
                }
            }
            TlaType::Func(key, val) => {
                format!("TlaFunc<{}, {}>", key.to_rust_type(), val.to_rust_type())
            }
            TlaType::Var(id) => format!("T{}", id),
            TlaType::Unknown => "/* unknown */".to_string(),
        }
    }

    /// Check if type is fully resolved (no type variables)
    pub fn is_resolved(&self) -> bool {
        match self {
            TlaType::Var(_) | TlaType::Unknown => false,
            TlaType::Set(t) | TlaType::Seq(t) => t.is_resolved(),
            TlaType::Tuple(ts) => ts.iter().all(|t| t.is_resolved()),
            TlaType::Record(fs) => fs.iter().all(|(_, t)| t.is_resolved()),
            TlaType::Func(k, v) => k.is_resolved() && v.is_resolved(),
            _ => true,
        }
    }
}

/// Type inference errors
#[derive(Debug, Clone)]
pub enum TypeInferError {
    /// Type mismatch between expected and actual
    TypeMismatch {
        expected: TlaType,
        actual: TlaType,
        context: String,
    },
    /// Unknown identifier
    UnknownIdent(String),
    /// Cannot infer type for expression
    CannotInfer(String),
    /// Unsupported construct for code generation
    Unsupported(String),
}

impl std::fmt::Display for TypeInferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeInferError::TypeMismatch {
                expected,
                actual,
                context,
            } => {
                write!(
                    f,
                    "type mismatch in {}: expected {:?}, got {:?}",
                    context, expected, actual
                )
            }
            TypeInferError::UnknownIdent(name) => write!(f, "unknown identifier: {}", name),
            TypeInferError::CannotInfer(msg) => write!(f, "cannot infer type: {}", msg),
            TypeInferError::Unsupported(msg) => write!(f, "unsupported for codegen: {}", msg),
        }
    }
}

impl std::error::Error for TypeInferError {}

/// Type inference context
pub struct TypeContext {
    /// Variable name to type mapping
    vars: HashMap<String, TlaType>,
    /// Operator definitions
    ops: HashMap<String, TlaType>,
    /// Next type variable id
    next_var: usize,
    /// Collected errors
    errors: Vec<TypeInferError>,
}

impl TypeContext {
    /// Create a new type context
    pub fn new() -> Self {
        TypeContext {
            vars: HashMap::new(),
            ops: HashMap::new(),
            next_var: 0,
            errors: Vec::new(),
        }
    }

    /// Create a fresh type variable
    pub fn fresh_var(&mut self) -> TlaType {
        let id = self.next_var;
        self.next_var += 1;
        TlaType::Var(id)
    }

    /// Add a variable binding
    pub fn bind_var(&mut self, name: &str, ty: TlaType) {
        self.vars.insert(name.to_string(), ty);
    }

    /// Look up a variable type
    pub fn lookup_var(&self, name: &str) -> Option<&TlaType> {
        self.vars.get(name)
    }

    /// Record an error
    pub fn error(&mut self, err: TypeInferError) {
        self.errors.push(err);
    }

    /// Take collected errors
    pub fn take_errors(&mut self) -> Vec<TypeInferError> {
        std::mem::take(&mut self.errors)
    }

    /// Try to unify a variable with a concrete type.
    /// If the expression is an identifier with unresolved type, update it.
    fn try_unify_with_var(&mut self, expr: &Expr, ty: &TlaType) {
        // Only unify if the type is concrete (resolved)
        if !ty.is_resolved() {
            return;
        }

        if let Expr::Ident(name) = expr {
            if let Some(existing) = self.vars.get(name) {
                // Only update if existing type is unresolved (a type variable)
                if !existing.is_resolved() {
                    self.vars.insert(name.clone(), ty.clone());
                }
            }
        }
    }

    /// Infer types for a module
    pub fn infer_module(&mut self, module: &Module) -> HashMap<String, TlaType> {
        // First pass: collect variable declarations
        for unit in &module.units {
            if let Unit::Variable(vars) = &unit.node {
                for var in vars {
                    // Initially unknown, will be inferred from Init
                    let fresh = self.fresh_var();
                    self.bind_var(&var.node, fresh);
                }
            }
        }

        // Second pass: pre-register stdlib operators based on EXTENDS
        self.register_stdlib_operators(&module.extends);

        // Third pass: pre-register operator names (supports forward references)
        for unit in &module.units {
            if let Unit::Operator(op) = &unit.node {
                if !self.ops.contains_key(&op.name.node) {
                    let fresh = self.fresh_var();
                    self.ops.insert(op.name.node.clone(), fresh);
                }
            }
        }

        // Fourth pass: infer from operator definitions
        for unit in &module.units {
            if let Unit::Operator(op) = &unit.node {
                self.infer_operator(op);
            }
        }

        // Return variable types
        self.vars.clone()
    }

    /// Register stdlib operators based on EXTENDS clause
    fn register_stdlib_operators(&mut self, extends: &[Spanned<String>]) {
        for ext in extends {
            let ops_to_add: Vec<(&str, TlaType)> = match ext.node.as_str() {
                "Sequences" => vec![
                    ("Len", TlaType::Int),
                    ("Head", TlaType::Unknown),
                    ("Tail", TlaType::Unknown),
                    ("Append", TlaType::Unknown),
                    ("SubSeq", TlaType::Unknown),
                    ("Seq", TlaType::Unknown),
                    ("SelectSeq", TlaType::Unknown),
                ],
                "FiniteSets" => vec![
                    ("Cardinality", TlaType::Int),
                    ("IsFiniteSet", TlaType::Bool),
                ],
                "TLC" => vec![
                    ("Print", TlaType::Unknown),
                    ("PrintT", TlaType::Bool),
                    ("Assert", TlaType::Bool),
                    ("ToString", TlaType::String),
                    ("JavaTime", TlaType::Int),
                    ("TLCGet", TlaType::Unknown),
                    ("TLCSet", TlaType::Bool),
                    ("Permutations", TlaType::Unknown),
                    ("SortSeq", TlaType::Unknown),
                    ("RandomElement", TlaType::Unknown),
                    ("TLCEval", TlaType::Unknown),
                    ("Any", TlaType::Unknown),
                ],
                "SequencesExt" => vec![
                    ("SetToSeq", TlaType::Unknown),
                    ("SetToSortSeq", TlaType::Unknown),
                    ("Reverse", TlaType::Unknown),
                    ("Remove", TlaType::Unknown),
                    ("ReplaceAt", TlaType::Unknown),
                    ("InsertAt", TlaType::Unknown),
                    ("RemoveAt", TlaType::Unknown),
                    ("Front", TlaType::Unknown),
                    ("Last", TlaType::Unknown),
                    ("IsPrefix", TlaType::Bool),
                    ("IsSuffix", TlaType::Bool),
                    ("Contains", TlaType::Bool),
                    ("Cons", TlaType::Unknown),
                    ("FlattenSeq", TlaType::Unknown),
                    ("Zip", TlaType::Unknown),
                    ("FoldLeft", TlaType::Unknown),
                    ("FoldRight", TlaType::Unknown),
                    ("ToSet", TlaType::Unknown),
                    ("Range", TlaType::Unknown),
                    ("Indices", TlaType::Unknown),
                ],
                "FiniteSetsExt" => vec![
                    ("FoldSet", TlaType::Unknown),
                    ("ReduceSet", TlaType::Unknown),
                    ("Quantify", TlaType::Int),
                    ("Ksubsets", TlaType::Unknown),
                    ("Symmetry", TlaType::Unknown),
                    ("Sum", TlaType::Int),
                    ("Product", TlaType::Int),
                    ("Max", TlaType::Int),
                    ("Min", TlaType::Int),
                    ("Mean", TlaType::Int),
                    ("SymDiff", TlaType::Unknown),
                    ("Flatten", TlaType::Unknown),
                    ("Choose", TlaType::Unknown),
                ],
                "Functions" => vec![
                    ("Range", TlaType::Unknown),
                    ("Inverse", TlaType::Unknown),
                    ("Restrict", TlaType::Unknown),
                    ("IsInjective", TlaType::Bool),
                    ("IsSurjective", TlaType::Bool),
                    ("IsBijection", TlaType::Bool),
                    ("AntiFunction", TlaType::Unknown),
                    ("FoldFunction", TlaType::Unknown),
                    ("FoldFunctionOnSet", TlaType::Unknown),
                ],
                "Bags" => vec![
                    ("IsABag", TlaType::Bool),
                    ("BagToSet", TlaType::Unknown),
                    ("SetToBag", TlaType::Unknown),
                    ("BagIn", TlaType::Bool),
                    ("EmptyBag", TlaType::Unknown),
                    ("CopiesIn", TlaType::Int),
                    ("BagCup", TlaType::Unknown),
                    ("BagDiff", TlaType::Unknown),
                    ("BagUnion", TlaType::Unknown),
                    ("SqSubseteq", TlaType::Bool),
                    ("SubBag", TlaType::Unknown),
                    ("BagOfAll", TlaType::Unknown),
                    ("BagCardinality", TlaType::Int),
                ],
                "TLCExt" => vec![
                    ("AssertError", TlaType::Unknown),
                    ("AssertEq", TlaType::Bool),
                    ("Trace", TlaType::Unknown),
                    ("TLCDefer", TlaType::Unknown),
                    ("PickSuccessor", TlaType::Unknown),
                ],
                // Naturals and Integers don't add callable operators (just built-in syntax)
                "Naturals" | "Integers" | "Reals" => vec![],
                _ => vec![],
            };

            for (name, ty) in ops_to_add {
                self.ops.insert(name.to_string(), ty);
            }
        }
    }

    /// Infer type for an operator definition
    fn infer_operator(&mut self, op: &OperatorDef) {
        // Add parameters to scope
        for param in &op.params {
            let fresh = self.fresh_var();
            self.bind_var(&param.name.node, fresh);
        }

        // Infer body type
        let body_type = self.infer_expr(&op.body);

        // Store operator type
        self.ops.insert(op.name.node.clone(), body_type);
    }

    /// Infer type for an expression
    pub fn infer_expr(&mut self, expr: &Spanned<Expr>) -> TlaType {
        match &expr.node {
            // Literals
            Expr::Bool(_) => TlaType::Bool,
            Expr::Int(_) => TlaType::Int,
            Expr::String(_) => TlaType::String,

            // Variables
            Expr::Ident(name) => {
                if let Some(ty) = self.lookup_var(name) {
                    return ty.clone();
                }

                // Check for built-in constants
                match name.as_str() {
                    "TRUE" | "FALSE" => TlaType::Bool,
                    "BOOLEAN" => TlaType::Set(Box::new(TlaType::Bool)),
                    "Nat" | "Int" => TlaType::Set(Box::new(TlaType::Int)),
                    "STRING" => TlaType::Set(Box::new(TlaType::String)),
                    _ => self.ops.get(name).cloned().unwrap_or_else(|| {
                        self.error(TypeInferError::UnknownIdent(name.clone()));
                        TlaType::Unknown
                    }),
                }
            }

            // Logic
            Expr::And(a, b) | Expr::Or(a, b) => {
                self.infer_expr(a);
                self.infer_expr(b);
                TlaType::Bool
            }
            Expr::Not(a) => {
                self.infer_expr(a);
                TlaType::Bool
            }
            Expr::Implies(a, b) | Expr::Equiv(a, b) => {
                self.infer_expr(a);
                self.infer_expr(b);
                TlaType::Bool
            }

            // Comparison
            Expr::Eq(a, b) | Expr::Neq(a, b) => {
                let ty_a = self.infer_expr(a);
                let ty_b = self.infer_expr(b);
                // Propagate types: if one side is a variable with unresolved type,
                // update it to match the other side's concrete type
                self.try_unify_with_var(&a.node, &ty_b);
                self.try_unify_with_var(&b.node, &ty_a);
                TlaType::Bool
            }
            Expr::Lt(a, b) | Expr::Leq(a, b) | Expr::Gt(a, b) | Expr::Geq(a, b) => {
                self.infer_expr(a);
                self.infer_expr(b);
                TlaType::Bool
            }

            // Arithmetic
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                self.infer_expr(a);
                self.infer_expr(b);
                TlaType::Int
            }
            Expr::IntDiv(a, b) | Expr::Mod(a, b) | Expr::Pow(a, b) => {
                self.infer_expr(a);
                self.infer_expr(b);
                TlaType::Int
            }
            Expr::Neg(a) => {
                self.infer_expr(a);
                TlaType::Int
            }
            Expr::Range(a, b) => {
                self.infer_expr(a);
                self.infer_expr(b);
                TlaType::Set(Box::new(TlaType::Int))
            }

            // Sets
            Expr::SetEnum(elems) => {
                if elems.is_empty() {
                    TlaType::Set(Box::new(self.fresh_var()))
                } else {
                    let elem_type = self.infer_expr(&elems[0]);
                    for elem in elems.iter().skip(1) {
                        self.infer_expr(elem);
                    }
                    TlaType::Set(Box::new(elem_type))
                }
            }
            Expr::In(elem, set) | Expr::NotIn(elem, set) => {
                self.infer_expr(elem);
                let set_ty = self.infer_expr(set);
                // If set has type Set(T), then elem should have type T
                if let TlaType::Set(elem_ty) = set_ty {
                    self.try_unify_with_var(&elem.node, &elem_ty);
                }
                TlaType::Bool
            }
            Expr::Subseteq(a, b) => {
                self.infer_expr(a);
                self.infer_expr(b);
                TlaType::Bool
            }
            Expr::Union(a, b) | Expr::Intersect(a, b) | Expr::SetMinus(a, b) => {
                let ta = self.infer_expr(a);
                self.infer_expr(b);
                ta
            }
            Expr::Powerset(s) => {
                let inner = self.infer_expr(s);
                TlaType::Set(Box::new(inner))
            }
            Expr::BigUnion(s) => {
                // UNION {{1,2}, {3}} has type Set(Int) if s has type Set(Set(Int))
                let s_type = self.infer_expr(s);
                if let TlaType::Set(inner) = s_type {
                    if let TlaType::Set(elem) = *inner {
                        TlaType::Set(elem)
                    } else {
                        TlaType::Unknown
                    }
                } else {
                    TlaType::Unknown
                }
            }
            Expr::SetBuilder(body, bounds) => {
                for bound in bounds {
                    self.bind_bound_var(bound);
                }
                let elem_type = self.infer_expr(body);
                TlaType::Set(Box::new(elem_type))
            }
            Expr::SetFilter(bound, pred) => {
                let elem_type = self.bind_bound_var(bound);
                self.infer_expr(pred);
                TlaType::Set(Box::new(elem_type))
            }

            // Tuples and sequences
            Expr::Tuple(elems) => {
                let types: Vec<_> = elems.iter().map(|e| self.infer_expr(e)).collect();
                TlaType::Tuple(types)
            }
            Expr::Times(sets) => {
                let types: Vec<_> = sets
                    .iter()
                    .map(|s| {
                        let st = self.infer_expr(s);
                        if let TlaType::Set(inner) = st {
                            *inner
                        } else {
                            TlaType::Unknown
                        }
                    })
                    .collect();
                TlaType::Set(Box::new(TlaType::Tuple(types)))
            }

            // Records
            Expr::Record(fields) => {
                let field_types: Vec<_> = fields
                    .iter()
                    .map(|(name, val)| (name.node.clone(), self.infer_expr(val)))
                    .collect();
                TlaType::Record(field_types)
            }
            Expr::RecordAccess(rec, field) => {
                let rec_type = self.infer_expr(rec);
                if let TlaType::Record(fields) = rec_type {
                    // Find field type
                    for (name, ty) in &fields {
                        if name == &field.node {
                            return ty.clone();
                        }
                    }
                }
                TlaType::Unknown
            }
            Expr::RecordSet(fields) => {
                let field_types: Vec<_> = fields
                    .iter()
                    .map(|(name, set)| {
                        let set_type = self.infer_expr(set);
                        let elem_type = if let TlaType::Set(inner) = set_type {
                            *inner
                        } else {
                            TlaType::Unknown
                        };
                        (name.node.clone(), elem_type)
                    })
                    .collect();
                TlaType::Set(Box::new(TlaType::Record(field_types)))
            }

            // Functions
            Expr::FuncDef(bounds, body) => {
                let mut domain_types = Vec::new();
                for bound in bounds {
                    domain_types.push(self.bind_bound_var(bound));
                }
                let range_type = self.infer_expr(body);

                let domain_type = if domain_types.len() == 1 {
                    domain_types.pop().unwrap()
                } else {
                    TlaType::Tuple(domain_types)
                };

                TlaType::Func(Box::new(domain_type), Box::new(range_type))
            }
            Expr::FuncApply(func, arg) => {
                let func_type = self.infer_expr(func);
                self.infer_expr(arg);
                if let TlaType::Func(_, range) = func_type {
                    *range
                } else {
                    TlaType::Unknown
                }
            }
            Expr::Domain(func) => {
                let func_type = self.infer_expr(func);
                if let TlaType::Func(domain, _) = func_type {
                    TlaType::Set(domain)
                } else {
                    TlaType::Unknown
                }
            }
            Expr::FuncSet(domain, range) => {
                let domain_type = self.infer_expr(domain);
                let range_type = self.infer_expr(range);
                let d = if let TlaType::Set(inner) = domain_type {
                    *inner
                } else {
                    TlaType::Unknown
                };
                let r = if let TlaType::Set(inner) = range_type {
                    *inner
                } else {
                    TlaType::Unknown
                };
                TlaType::Set(Box::new(TlaType::Func(Box::new(d), Box::new(r))))
            }
            Expr::Except(func, _specs) => self.infer_expr(func),

            // Quantifiers
            Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
                for bound in bounds {
                    self.bind_bound_var(bound);
                }
                self.infer_expr(body);
                TlaType::Bool
            }
            Expr::Choose(bound, body) => {
                let elem_type = self.bind_bound_var(bound);
                self.infer_expr(body);
                elem_type
            }

            // Control
            Expr::If(cond, then_e, else_e) => {
                self.infer_expr(cond);
                let t1 = self.infer_expr(then_e);
                self.infer_expr(else_e);
                t1
            }
            Expr::Case(arms, other) => {
                let ty = if let Some(arm) = arms.first() {
                    self.infer_expr(&arm.guard);
                    self.infer_expr(&arm.body)
                } else {
                    TlaType::Unknown
                };
                for arm in arms.iter().skip(1) {
                    self.infer_expr(&arm.guard);
                    self.infer_expr(&arm.body);
                }
                if let Some(other_expr) = other {
                    self.infer_expr(other_expr);
                }
                ty
            }
            Expr::Let(defs, body) => {
                for def in defs {
                    self.infer_operator(def);
                }
                self.infer_expr(body)
            }
            Expr::Lambda(params, body) => {
                for param in params {
                    let fresh = self.fresh_var();
                    self.bind_var(&param.node, fresh);
                }
                self.infer_expr(body)
            }
            Expr::Apply(op, args) => {
                let op_ty = self.infer_expr(op);
                for arg in args {
                    self.infer_expr(arg);
                }
                op_ty
            }

            // Prime is used in Next actions - the type of x' is the type of x
            Expr::Prime(inner) => self.infer_expr(inner),
            Expr::Always(_) | Expr::Eventually(_) | Expr::LeadsTo(_, _) => {
                self.error(TypeInferError::Unsupported(
                    "temporal operators".to_string(),
                ));
                TlaType::Unknown
            }
            Expr::WeakFair(_, _) | Expr::StrongFair(_, _) => {
                self.error(TypeInferError::Unsupported("fairness".to_string()));
                TlaType::Unknown
            }
            Expr::Enabled(_) => {
                self.error(TypeInferError::Unsupported("ENABLED".to_string()));
                TlaType::Unknown
            }
            Expr::Unchanged(_) => TlaType::Bool,
            Expr::ModuleRef(_, _, _) => {
                self.error(TypeInferError::Unsupported(
                    "module reference (M!Op)".to_string(),
                ));
                TlaType::Unknown
            }
            Expr::InstanceExpr(_, _) => {
                self.error(TypeInferError::Unsupported(
                    "INSTANCE expression".to_string(),
                ));
                TlaType::Unknown
            }
            Expr::OpRef(op) => {
                self.error(TypeInferError::Unsupported(format!(
                    "operator reference ({})",
                    op
                )));
                TlaType::Unknown
            }
        }
    }

    /// Bind a bound variable and return its type
    fn bind_bound_var(&mut self, bound: &BoundVar) -> TlaType {
        let elem_type = if let Some(domain) = &bound.domain {
            let domain_type = self.infer_expr(domain);
            if let TlaType::Set(inner) = domain_type {
                *inner
            } else {
                self.fresh_var()
            }
        } else {
            self.fresh_var()
        };
        self.bind_var(&bound.name.node, elem_type.clone());
        elem_type
    }
}

impl Default for TypeContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_to_rust() {
        assert_eq!(TlaType::Bool.to_rust_type(), "bool");
        assert_eq!(TlaType::Int.to_rust_type(), "i64");
        assert_eq!(
            TlaType::Set(Box::new(TlaType::Int)).to_rust_type(),
            "TlaSet<i64>"
        );
        assert_eq!(
            TlaType::Func(Box::new(TlaType::Int), Box::new(TlaType::Bool)).to_rust_type(),
            "TlaFunc<i64, bool>"
        );
    }

    #[test]
    fn test_infer_literals() {
        let mut ctx = TypeContext::new();

        use tla_core::span::{FileId, Span, Spanned};
        let span = Span::new(FileId(0), 0, 0);

        let bool_expr = Spanned::new(Expr::Bool(true), span);
        assert_eq!(ctx.infer_expr(&bool_expr), TlaType::Bool);

        let int_expr = Spanned::new(Expr::Int(42.into()), span);
        assert_eq!(ctx.infer_expr(&int_expr), TlaType::Int);
    }

    #[test]
    fn test_infer_set() {
        let mut ctx = TypeContext::new();

        use tla_core::span::{FileId, Span, Spanned};
        let span = Span::new(FileId(0), 0, 0);

        // {1, 2, 3}
        let set_expr = Spanned::new(
            Expr::SetEnum(vec![
                Spanned::new(Expr::Int(1.into()), span),
                Spanned::new(Expr::Int(2.into()), span),
            ]),
            span,
        );
        assert_eq!(
            ctx.infer_expr(&set_expr),
            TlaType::Set(Box::new(TlaType::Int))
        );
    }
}
