//! C Statement Semantics
//!
//! This module defines the abstract syntax for C statements and
//! control flow structures.
//!
//! ## Statement Categories
//!
//! - **Expression statements**: expr;
//! - **Declaration statements**: type var = init;
//! - **Control flow**: if, switch, while, for, do-while
//! - **Jump statements**: break, continue, return, goto
//! - **Compound statements**: { ... }
//!
//! ## Control Flow
//!
//! Control flow is modeled using a small-step or big-step semantics.
//! The evaluator in `eval.rs` implements the actual execution.

use crate::expr::{CExpr, Ident, Initializer};
use crate::types::CType;
use serde::{Deserialize, Serialize};

/// Storage class specifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageClass {
    /// No explicit storage class
    Auto,
    /// Static storage duration
    Static,
    /// External linkage
    Extern,
    /// Register hint
    Register,
    /// Thread-local (C11)
    ThreadLocal,
}

/// Variable declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VarDecl {
    /// Variable name
    pub name: Ident,
    /// Variable type
    pub ty: CType,
    /// Storage class
    pub storage: StorageClass,
    /// Initial value (if any)
    pub init: Option<Initializer>,
}

impl VarDecl {
    /// Create a simple variable declaration
    pub fn new(name: impl Into<String>, ty: CType) -> Self {
        Self {
            name: name.into(),
            ty,
            storage: StorageClass::Auto,
            init: None,
        }
    }

    /// With initializer
    #[must_use]
    pub fn with_init(mut self, init: Initializer) -> Self {
        self.init = Some(init);
        self
    }

    /// With storage class
    #[must_use]
    pub fn with_storage(mut self, storage: StorageClass) -> Self {
        self.storage = storage;
        self
    }
}

/// Function parameter
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FuncParam {
    /// Parameter name
    pub name: Ident,
    /// Parameter type
    pub ty: CType,
}

impl FuncParam {
    pub fn new(name: impl Into<String>, ty: CType) -> Self {
        Self {
            name: name.into(),
            ty,
        }
    }
}

/// Function definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuncDef {
    /// Function name
    pub name: Ident,
    /// Return type
    pub return_type: CType,
    /// Parameters
    pub params: Vec<FuncParam>,
    /// Is variadic (...)
    pub variadic: bool,
    /// Storage class (static, extern)
    pub storage: StorageClass,
    /// Function body
    pub body: Box<CStmt>,
}

impl FuncDef {
    /// Create a simple function definition
    pub fn new(
        name: impl Into<String>,
        return_type: CType,
        params: Vec<FuncParam>,
        body: CStmt,
    ) -> Self {
        Self {
            name: name.into(),
            return_type,
            params,
            variadic: false,
            storage: StorageClass::Auto,
            body: Box::new(body),
        }
    }

    /// Get the function type
    pub fn func_type(&self) -> CType {
        CType::Function {
            return_type: Box::new(self.return_type.clone()),
            params: self
                .params
                .iter()
                .map(|p| crate::types::FuncParam {
                    name: Some(p.name.clone()),
                    ty: p.ty.clone(),
                })
                .collect(),
            variadic: self.variadic,
        }
    }
}

/// Switch case label
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CaseLabel {
    /// case expr:
    Case(CExpr),
    /// default:
    Default,
}

/// C Statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CStmt {
    /// Empty statement: ;
    Empty,

    /// Expression statement: expr;
    Expr(CExpr),

    /// Variable declaration: type var = init;
    Decl(VarDecl),

    /// Multiple declarations: type v1, v2, v3;
    DeclList(Vec<VarDecl>),

    /// Compound statement (block): { stmts }
    Block(Vec<CStmt>),

    /// If statement: if (cond) then_stmt else else_stmt
    If {
        cond: CExpr,
        then_stmt: Box<CStmt>,
        else_stmt: Option<Box<CStmt>>,
    },

    /// Switch statement
    Switch { cond: CExpr, body: Box<CStmt> },

    /// Case label: case expr: or default:
    Case { label: CaseLabel, stmt: Box<CStmt> },

    /// While loop: while (cond) body
    While { cond: CExpr, body: Box<CStmt> },

    /// Do-while loop: do body while (cond);
    DoWhile { body: Box<CStmt>, cond: CExpr },

    /// For loop: for (init; cond; update) body
    For {
        init: Option<Box<CStmt>>, // Can be Expr, Decl, or DeclList
        cond: Option<CExpr>,
        update: Option<CExpr>,
        body: Box<CStmt>,
    },

    /// Break statement: break;
    Break,

    /// Continue statement: continue;
    Continue,

    /// Return statement: return expr;
    Return(Option<CExpr>),

    /// Goto statement: goto label;
    Goto(Ident),

    /// Label: label:
    Label { name: Ident, stmt: Box<CStmt> },

    /// Function definition (at top level)
    FuncDef(FuncDef),

    /// Inline assembly (not fully supported)
    Asm(String),

    /// Assert annotation: //@ assert spec;
    Assert(crate::spec::Spec),

    /// Assume annotation: //@ assume spec;
    Assume(crate::spec::Spec),
}

impl CStmt {
    // Constructors for convenience

    /// Create an empty statement
    pub fn empty() -> Self {
        CStmt::Empty
    }

    /// Create an expression statement
    pub fn expr(e: CExpr) -> Self {
        CStmt::Expr(e)
    }

    /// Create a variable declaration
    pub fn decl(name: impl Into<String>, ty: CType) -> Self {
        CStmt::Decl(VarDecl::new(name, ty))
    }

    /// Create a declaration with initializer
    pub fn decl_init(name: impl Into<String>, ty: CType, init: CExpr) -> Self {
        CStmt::Decl(VarDecl::new(name, ty).with_init(Initializer::Expr(init)))
    }

    /// Create a block statement
    pub fn block(stmts: Vec<CStmt>) -> Self {
        CStmt::Block(stmts)
    }

    /// Create an if statement
    pub fn if_stmt(cond: CExpr, then_stmt: CStmt) -> Self {
        CStmt::If {
            cond,
            then_stmt: Box::new(then_stmt),
            else_stmt: None,
        }
    }

    /// Create an if-else statement
    pub fn if_else(cond: CExpr, then_stmt: CStmt, else_stmt: CStmt) -> Self {
        CStmt::If {
            cond,
            then_stmt: Box::new(then_stmt),
            else_stmt: Some(Box::new(else_stmt)),
        }
    }

    /// Create a while loop
    pub fn while_loop(cond: CExpr, body: CStmt) -> Self {
        CStmt::While {
            cond,
            body: Box::new(body),
        }
    }

    /// Create a do-while loop
    pub fn do_while(body: CStmt, cond: CExpr) -> Self {
        CStmt::DoWhile {
            body: Box::new(body),
            cond,
        }
    }

    /// Create a for loop
    pub fn for_loop(
        init: Option<CStmt>,
        cond: Option<CExpr>,
        update: Option<CExpr>,
        body: CStmt,
    ) -> Self {
        CStmt::For {
            init: init.map(Box::new),
            cond,
            update,
            body: Box::new(body),
        }
    }

    /// Create a return statement
    pub fn return_stmt(expr: Option<CExpr>) -> Self {
        CStmt::Return(expr)
    }

    /// Create a break statement
    pub fn break_stmt() -> Self {
        CStmt::Break
    }

    /// Create a continue statement
    pub fn continue_stmt() -> Self {
        CStmt::Continue
    }

    /// Check if this statement is a declaration
    pub fn is_decl(&self) -> bool {
        matches!(
            self,
            CStmt::Decl(_) | CStmt::DeclList(_) | CStmt::FuncDef(_)
        )
    }

    /// Check if this statement contains any jump statements
    pub fn contains_jump(&self) -> bool {
        match self {
            CStmt::Break | CStmt::Continue | CStmt::Return(_) | CStmt::Goto(_) => true,
            CStmt::Block(stmts) => stmts.iter().any(CStmt::contains_jump),
            CStmt::If {
                then_stmt,
                else_stmt,
                ..
            } => then_stmt.contains_jump() || else_stmt.as_ref().is_some_and(|s| s.contains_jump()),
            CStmt::While { body, .. }
            | CStmt::DoWhile { body, .. }
            | CStmt::For { body, .. }
            | CStmt::Switch { body, .. } => body.contains_jump(),
            CStmt::Case { stmt, .. } | CStmt::Label { stmt, .. } => stmt.contains_jump(),
            _ => false,
        }
    }

    /// Get all labels defined in this statement
    pub fn get_labels(&self) -> Vec<&str> {
        let mut labels = Vec::new();
        self.collect_labels(&mut labels);
        labels
    }

    fn collect_labels<'a>(&'a self, labels: &mut Vec<&'a str>) {
        match self {
            CStmt::Label { name, stmt } => {
                labels.push(name);
                stmt.collect_labels(labels);
            }
            CStmt::Block(stmts) => {
                for s in stmts {
                    s.collect_labels(labels);
                }
            }
            CStmt::If {
                then_stmt,
                else_stmt,
                ..
            } => {
                then_stmt.collect_labels(labels);
                if let Some(e) = else_stmt {
                    e.collect_labels(labels);
                }
            }
            CStmt::While { body, .. }
            | CStmt::DoWhile { body, .. }
            | CStmt::For { body, .. }
            | CStmt::Switch { body, .. } => {
                body.collect_labels(labels);
            }
            CStmt::Case { stmt, .. } => {
                stmt.collect_labels(labels);
            }
            _ => {}
        }
    }
}

/// A translation unit (a single C source file)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationUnit {
    /// Top-level declarations and definitions
    pub decls: Vec<TopLevel>,
}

/// Top-level declaration or definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TopLevel {
    /// Variable declaration
    VarDecl(VarDecl),
    /// Function definition
    FuncDef(FuncDef),
    /// Function declaration (prototype)
    FuncDecl {
        name: Ident,
        return_type: CType,
        params: Vec<FuncParam>,
        variadic: bool,
    },
    /// Type definition: typedef
    TypeDef { name: Ident, ty: CType },
    /// Struct/union/enum declaration
    TypeDecl(CType),
}

impl TranslationUnit {
    /// Create a new empty translation unit
    pub fn new() -> Self {
        Self { decls: Vec::new() }
    }

    /// Add a declaration
    pub fn add(&mut self, decl: TopLevel) {
        self.decls.push(decl);
    }

    /// Find a function definition by name
    pub fn find_func(&self, name: &str) -> Option<&FuncDef> {
        for decl in &self.decls {
            if let TopLevel::FuncDef(func) = decl {
                if func.name == name {
                    return Some(func);
                }
            }
        }
        None
    }

    /// Find a global variable by name
    pub fn find_global(&self, name: &str) -> Option<&VarDecl> {
        for decl in &self.decls {
            if let TopLevel::VarDecl(var) = decl {
                if var.name == name {
                    return Some(var);
                }
            }
        }
        None
    }

    /// Get all function names
    pub fn function_names(&self) -> Vec<&str> {
        self.decls
            .iter()
            .filter_map(|d| match d {
                TopLevel::FuncDef(f) => Some(f.name.as_str()),
                _ => None,
            })
            .collect()
    }
}

impl Default for TranslationUnit {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::BinOp;

    #[test]
    fn test_var_decl() {
        let decl = VarDecl::new("x", CType::int()).with_init(Initializer::Expr(CExpr::int(42)));

        assert_eq!(decl.name, "x");
        assert!(decl.init.is_some());
    }

    #[test]
    fn test_func_def() {
        let func = FuncDef::new(
            "add",
            CType::int(),
            vec![
                FuncParam::new("a", CType::int()),
                FuncParam::new("b", CType::int()),
            ],
            CStmt::return_stmt(Some(CExpr::add(CExpr::var("a"), CExpr::var("b")))),
        );

        assert_eq!(func.name, "add");
        assert_eq!(func.params.len(), 2);
        assert!(!func.variadic);
    }

    #[test]
    fn test_if_stmt() {
        let stmt = CStmt::if_else(
            CExpr::binop(BinOp::Lt, CExpr::var("x"), CExpr::int(0)),
            CStmt::return_stmt(Some(CExpr::unary(
                crate::expr::UnaryOp::Neg,
                CExpr::var("x"),
            ))),
            CStmt::return_stmt(Some(CExpr::var("x"))),
        );

        match stmt {
            CStmt::If { else_stmt, .. } => assert!(else_stmt.is_some()),
            _ => panic!("Expected If statement"),
        }
    }

    #[test]
    fn test_for_loop() {
        // for (int i = 0; i < n; i++) { sum += i; }
        let stmt = CStmt::for_loop(
            Some(CStmt::decl_init("i", CType::int(), CExpr::int(0))),
            Some(CExpr::binop(BinOp::Lt, CExpr::var("i"), CExpr::var("n"))),
            Some(CExpr::unary(crate::expr::UnaryOp::PostInc, CExpr::var("i"))),
            CStmt::expr(CExpr::binop(
                BinOp::AddAssign,
                CExpr::var("sum"),
                CExpr::var("i"),
            )),
        );

        match stmt {
            CStmt::For {
                init, cond, update, ..
            } => {
                assert!(init.is_some());
                assert!(cond.is_some());
                assert!(update.is_some());
            }
            _ => panic!("Expected For loop"),
        }
    }

    #[test]
    fn test_contains_jump() {
        let no_jump = CStmt::expr(CExpr::int(42));
        assert!(!no_jump.contains_jump());

        let with_return = CStmt::block(vec![
            CStmt::expr(CExpr::int(1)),
            CStmt::return_stmt(Some(CExpr::int(0))),
        ]);
        assert!(with_return.contains_jump());

        let with_break = CStmt::while_loop(CExpr::int(1), CStmt::break_stmt());
        assert!(with_break.contains_jump());
    }

    #[test]
    fn test_get_labels() {
        let stmt = CStmt::block(vec![
            CStmt::Label {
                name: "start".to_string(),
                stmt: Box::new(CStmt::expr(CExpr::int(1))),
            },
            CStmt::Label {
                name: "end".to_string(),
                stmt: Box::new(CStmt::return_stmt(None)),
            },
        ]);

        let labels = stmt.get_labels();
        assert_eq!(labels.len(), 2);
        assert!(labels.contains(&"start"));
        assert!(labels.contains(&"end"));
    }

    #[test]
    fn test_translation_unit() {
        let mut tu = TranslationUnit::new();

        tu.add(TopLevel::VarDecl(VarDecl::new("global_x", CType::int())));
        tu.add(TopLevel::FuncDef(FuncDef::new(
            "main",
            CType::int(),
            vec![],
            CStmt::return_stmt(Some(CExpr::int(0))),
        )));

        assert!(tu.find_func("main").is_some());
        assert!(tu.find_global("global_x").is_some());
        assert_eq!(tu.function_names(), vec!["main"]);
    }
}
