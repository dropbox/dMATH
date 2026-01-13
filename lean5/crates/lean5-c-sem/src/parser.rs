//! C Parser using tree-sitter-c
//!
//! This module provides parsing of C source code into lean5-c-sem AST structures.
//! It uses tree-sitter-c for incremental, error-tolerant parsing.
//!
//! ## Features
//!
//! - Parse C source files into `FuncDef`, `CStmt`, `CExpr`, `CType`
//! - Handle standard C11 constructs
//! - Extract ACSL-style comments for specifications (/* @requires ... */)
//! - Error reporting with source locations
//!
//! ## Usage
//!
//! ```ignore
//! use lean5_c_sem::parser::CParser;
//!
//! let parser = CParser::new();
//! let code = r#"
//!     int abs(int x) {
//!         if (x < 0) return -x;
//!         return x;
//!     }
//! "#;
//!
//! let result = parser.parse_function(code)?;
//! println!("Parsed function: {}", result.name);
//! ```

use crate::expr::{BinOp, CExpr, Initializer, SizeOfArg, UnaryOp};
use crate::spec::{FuncSpec, Spec};
use crate::stmt::{CStmt, FuncDef, FuncParam, StorageClass, VarDecl};
use crate::types::{CType, FloatKind, IntKind, Signedness, StructField};
use crate::verified::VerifiedFunction;
use thiserror::Error;
use tree_sitter::{Node, Parser, Tree};

/// Parse errors
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Tree-sitter parser initialization failed")]
    ParserInit,

    #[error("Parse failed: no tree produced")]
    NoTree,

    #[error("Syntax error at {line}:{column}: {message}")]
    SyntaxError {
        line: usize,
        column: usize,
        message: String,
    },

    #[error("Unsupported construct at {line}:{column}: {kind}")]
    Unsupported {
        line: usize,
        column: usize,
        kind: String,
    },

    #[error("Missing required field: {field} in {node_kind}")]
    MissingField { field: String, node_kind: String },

    #[error("Type error: {message}")]
    TypeError { message: String },

    #[error("Invalid integer literal: {value}")]
    InvalidInt { value: String },

    #[error("Invalid float literal: {value}")]
    InvalidFloat { value: String },
}

/// Result type for parsing operations
pub type ParseResult<T> = Result<T, ParseError>;

trait NodeExt<'tree> {
    fn child_at(&self, index: usize) -> Option<Node<'tree>>;
}

impl<'tree> NodeExt<'tree> for Node<'tree> {
    fn child_at(&self, index: usize) -> Option<Node<'tree>> {
        self.child(index as u32)
    }
}

/// C Parser
///
/// Wraps tree-sitter-c for parsing C source code.
pub struct CParser {
    parser: Parser,
}

impl Default for CParser {
    fn default() -> Self {
        Self::new()
    }
}

impl CParser {
    /// Create a new C parser
    pub fn new() -> Self {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_c::LANGUAGE.into())
            .expect("Error loading C grammar");
        Self { parser }
    }

    /// Parse a C source string
    pub fn parse(&mut self, source: &str) -> ParseResult<Tree> {
        self.parser.parse(source, None).ok_or(ParseError::NoTree)
    }

    /// Parse a complete translation unit (multiple functions)
    pub fn parse_translation_unit(&mut self, source: &str) -> ParseResult<Vec<FuncDef>> {
        let tree = self.parse(source)?;
        let root = tree.root_node();
        let mut functions = Vec::new();

        for i in 0..root.child_count() {
            if let Some(child) = root.child_at(i) {
                if child.kind() == "function_definition" {
                    functions.push(self.parse_function_node(child, source)?);
                }
            }
        }

        Ok(functions)
    }

    /// Parse a complete translation unit and attach ACSL specs when present
    pub fn parse_translation_unit_with_specs(
        &mut self,
        source: &str,
    ) -> ParseResult<Vec<VerifiedFunction>> {
        let tree = self.parse(source)?;
        let root = tree.root_node();
        let mut functions = Vec::new();

        for i in 0..root.child_count() {
            if let Some(child) = root.child_at(i) {
                if child.kind() == "function_definition" {
                    let func = self.parse_function_node(child, source)?;
                    let spec = self.extract_func_spec(child, source).unwrap_or_default();
                    functions.push(VerifiedFunction {
                        name: func.name.clone(),
                        description: format!("Parsed function {}", func.name),
                        func,
                        spec,
                        sep_spec: None,
                    });
                }
            }
        }

        Ok(functions)
    }

    /// Parse a single function along with an optional ACSL spec
    pub fn parse_function_with_spec(&mut self, source: &str) -> ParseResult<VerifiedFunction> {
        let tree = self.parse(source)?;
        let root = tree.root_node();

        for i in 0..root.child_count() {
            if let Some(child) = root.child_at(i) {
                if child.kind() == "function_definition" {
                    let func = self.parse_function_node(child, source)?;
                    let spec = self.extract_func_spec(child, source).unwrap_or_default();
                    return Ok(VerifiedFunction {
                        name: func.name.clone(),
                        description: format!("Parsed function {}", func.name),
                        func,
                        spec,
                        sep_spec: None,
                    });
                }
            }
        }

        Err(ParseError::SyntaxError {
            line: 0,
            column: 0,
            message: "No function definition found".to_string(),
        })
    }

    /// Parse a single function definition from source
    pub fn parse_function(&mut self, source: &str) -> ParseResult<FuncDef> {
        let tree = self.parse(source)?;
        let root = tree.root_node();

        // Find function_definition node
        for i in 0..root.child_count() {
            if let Some(child) = root.child_at(i) {
                if child.kind() == "function_definition" {
                    return self.parse_function_node(child, source);
                }
            }
        }

        Err(ParseError::SyntaxError {
            line: 0,
            column: 0,
            message: "No function definition found".to_string(),
        })
    }

    /// Parse a function definition node
    fn parse_function_node(&self, node: Node, source: &str) -> ParseResult<FuncDef> {
        let mut return_type = CType::Void;
        let mut name = String::new();
        let mut params = Vec::new();
        let mut body = CStmt::Empty;
        let mut storage = StorageClass::Auto;
        let mut variadic = false;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    // Storage class specifiers
                    "storage_class_specifier" => {
                        let text = self.node_text(child, source);
                        storage = match text.as_str() {
                            "static" => StorageClass::Static,
                            "extern" => StorageClass::Extern,
                            "register" => StorageClass::Register,
                            "_Thread_local" | "thread_local" => StorageClass::ThreadLocal,
                            _ => StorageClass::Auto,
                        };
                    }
                    // Type specifiers
                    "primitive_type"
                    | "type_identifier"
                    | "sized_type_specifier"
                    | "struct_specifier"
                    | "union_specifier"
                    | "enum_specifier" => {
                        return_type = self.parse_type_node(child, source)?;
                    }
                    // Declarator (function name and parameters)
                    "function_declarator" => {
                        (name, params, variadic) = self.parse_func_declarator(child, source)?;
                    }
                    "pointer_declarator" => {
                        // Function returning pointer
                        if let Some(inner) = child.child_by_field_name("declarator") {
                            if inner.kind() == "function_declarator" {
                                (name, params, variadic) =
                                    self.parse_func_declarator(inner, source)?;
                                return_type = CType::Pointer(Box::new(return_type));
                            }
                        }
                    }
                    // Function body
                    "compound_statement" => {
                        body = self.parse_compound_stmt(child, source)?;
                    }
                    _ => {}
                }
            }
        }

        Ok(FuncDef {
            name,
            return_type,
            params,
            variadic,
            storage,
            body: Box::new(body),
        })
    }

    /// Parse function declarator (name and parameters)
    fn parse_func_declarator(
        &self,
        node: Node,
        source: &str,
    ) -> ParseResult<(String, Vec<FuncParam>, bool)> {
        let mut name = String::new();
        let mut params = Vec::new();
        let mut variadic = false;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "identifier" | "field_identifier" => {
                        name = self.node_text(child, source);
                    }
                    "parenthesized_declarator" => {
                        // Handle (*func)(...) patterns
                        if let Some(inner) = self.find_identifier(child, source) {
                            name = inner;
                        }
                    }
                    "parameter_list" => {
                        (params, variadic) = self.parse_param_list(child, source)?;
                    }
                    _ => {}
                }
            }
        }

        Ok((name, params, variadic))
    }

    /// Parse parameter list
    fn parse_param_list(&self, node: Node, source: &str) -> ParseResult<(Vec<FuncParam>, bool)> {
        let mut params = Vec::new();
        let mut variadic = false;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "parameter_declaration" => {
                        params.push(self.parse_param_decl(child, source)?);
                    }
                    "variadic_parameter" | "..." => {
                        variadic = true;
                    }
                    _ => {}
                }
            }
        }

        Ok((params, variadic))
    }

    /// Parse a parameter declaration
    fn parse_param_decl(&self, node: Node, source: &str) -> ParseResult<FuncParam> {
        let mut ty = CType::Void;
        let mut name = String::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "primitive_type"
                    | "type_identifier"
                    | "sized_type_specifier"
                    | "struct_specifier"
                    | "union_specifier"
                    | "enum_specifier" => {
                        ty = self.parse_type_node(child, source)?;
                    }
                    "identifier" => {
                        name = self.node_text(child, source);
                    }
                    "pointer_declarator" => {
                        ty = CType::Pointer(Box::new(ty));
                        if let Some(inner_name) = self.find_identifier(child, source) {
                            name = inner_name;
                        }
                    }
                    "abstract_pointer_declarator" => {
                        ty = CType::Pointer(Box::new(ty));
                    }
                    "array_declarator" => {
                        // Handle array parameters (decay to pointers in function params)
                        ty = CType::Pointer(Box::new(ty));
                        if let Some(inner_name) = self.find_identifier(child, source) {
                            name = inner_name;
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(FuncParam { name, ty })
    }

    /// Parse a type node
    fn parse_type_node(&self, node: Node, source: &str) -> ParseResult<CType> {
        match node.kind() {
            "primitive_type" => {
                let text = self.node_text(node, source);
                self.parse_primitive_type(&text)
            }
            "type_identifier" => {
                let name = self.node_text(node, source);
                // Common typedefs
                match name.as_str() {
                    "int8_t" => Ok(CType::Int(IntKind::Char, Signedness::Signed)),
                    "uint8_t" => Ok(CType::Int(IntKind::Char, Signedness::Unsigned)),
                    "int16_t" => Ok(CType::Int(IntKind::Short, Signedness::Signed)),
                    "uint16_t" => Ok(CType::Int(IntKind::Short, Signedness::Unsigned)),
                    "int32_t" => Ok(CType::Int(IntKind::Int, Signedness::Signed)),
                    "uint32_t" => Ok(CType::Int(IntKind::Int, Signedness::Unsigned)),
                    "int64_t" => Ok(CType::Int(IntKind::LongLong, Signedness::Signed)),
                    "uint64_t" => Ok(CType::Int(IntKind::LongLong, Signedness::Unsigned)),
                    "size_t" | "uintptr_t" => Ok(CType::Int(IntKind::Long, Signedness::Unsigned)),
                    "ssize_t" | "ptrdiff_t" | "intptr_t" => {
                        Ok(CType::Int(IntKind::Long, Signedness::Signed))
                    }
                    "bool" | "_Bool" => Ok(CType::Int(IntKind::Bool, Signedness::Unsigned)),
                    _ => Ok(CType::TypeDef(name)),
                }
            }
            "sized_type_specifier" => {
                let text = self.node_text(node, source);
                self.parse_sized_type(&text)
            }
            "struct_specifier" => self.parse_struct_node(node, source),
            "union_specifier" => self.parse_union_node(node, source),
            "enum_specifier" => self.parse_enum_node(node, source),
            _ => Ok(CType::TypeDef(self.node_text(node, source))),
        }
    }

    /// Parse primitive type
    fn parse_primitive_type(&self, text: &str) -> ParseResult<CType> {
        match text {
            "void" => Ok(CType::Void),
            "char" => Ok(CType::Int(IntKind::Char, Signedness::Signed)),
            "short" => Ok(CType::Int(IntKind::Short, Signedness::Signed)),
            "int" => Ok(CType::Int(IntKind::Int, Signedness::Signed)),
            "long" => Ok(CType::Int(IntKind::Long, Signedness::Signed)),
            "float" => Ok(CType::Float(FloatKind::Float)),
            "double" => Ok(CType::Float(FloatKind::Double)),
            "_Bool" | "bool" => Ok(CType::Int(IntKind::Bool, Signedness::Unsigned)),
            _ => Ok(CType::TypeDef(text.to_string())),
        }
    }

    /// Parse sized type specifier (unsigned int, long long, etc.)
    fn parse_sized_type(&self, text: &str) -> ParseResult<CType> {
        let text = text.trim();
        let parts: Vec<&str> = text.split_whitespace().collect();

        let mut signed = true;
        let mut kind = IntKind::Int;

        for part in parts {
            match part {
                "unsigned" => signed = false,
                "signed" => signed = true,
                "char" => kind = IntKind::Char,
                "short" => kind = IntKind::Short,
                "long" => {
                    kind = if kind == IntKind::Long {
                        IntKind::LongLong
                    } else {
                        IntKind::Long
                    };
                }
                _ => {} // default and unknown
            }
        }

        let signedness = if signed {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        };
        Ok(CType::Int(kind, signedness))
    }

    /// Parse struct specifier
    fn parse_struct_node(&self, node: Node, source: &str) -> ParseResult<CType> {
        let mut name = None;
        let mut fields = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "type_identifier" => {
                        name = Some(self.node_text(child, source));
                    }
                    "field_declaration_list" => {
                        fields = self.parse_field_list(child, source)?;
                    }
                    _ => {}
                }
            }
        }

        Ok(CType::Struct { name, fields })
    }

    /// Parse union specifier
    fn parse_union_node(&self, node: Node, source: &str) -> ParseResult<CType> {
        let mut name = None;
        let mut fields = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "type_identifier" => {
                        name = Some(self.node_text(child, source));
                    }
                    "field_declaration_list" => {
                        fields = self.parse_field_list(child, source)?;
                    }
                    _ => {}
                }
            }
        }

        Ok(CType::Union { name, fields })
    }

    /// Parse enum specifier
    fn parse_enum_node(&self, node: Node, source: &str) -> ParseResult<CType> {
        let mut name = None;
        let mut variants = Vec::new();
        let mut current_value: i64 = 0;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "type_identifier" => {
                        name = Some(self.node_text(child, source));
                    }
                    "enumerator_list" => {
                        for j in 0..child.child_count() {
                            if let Some(enum_child) = child.child_at(j) {
                                if enum_child.kind() == "enumerator" {
                                    let (variant_name, value) =
                                        self.parse_enumerator(enum_child, source, current_value)?;
                                    current_value = value + 1;
                                    variants.push((variant_name, value));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(CType::Enum { name, variants })
    }

    /// Parse an enumerator
    fn parse_enumerator(
        &self,
        node: Node,
        source: &str,
        default_value: i64,
    ) -> ParseResult<(String, i64)> {
        let mut name = String::new();
        let mut value = default_value;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "identifier" => {
                        name = self.node_text(child, source);
                    }
                    "number_literal" => {
                        let text = self.node_text(child, source);
                        value = self.parse_int_literal(&text)?;
                    }
                    _ => {}
                }
            }
        }

        Ok((name, value))
    }

    /// Parse field declaration list
    fn parse_field_list(&self, node: Node, source: &str) -> ParseResult<Vec<StructField>> {
        let mut fields = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                if child.kind() == "field_declaration" {
                    if let Some(field) = self.parse_field_decl(child, source)? {
                        fields.push(field);
                    }
                }
            }
        }

        Ok(fields)
    }

    /// Parse a field declaration
    fn parse_field_decl(&self, node: Node, source: &str) -> ParseResult<Option<StructField>> {
        let mut ty = CType::Void;
        let mut name = String::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "primitive_type"
                    | "type_identifier"
                    | "sized_type_specifier"
                    | "struct_specifier"
                    | "union_specifier"
                    | "enum_specifier" => {
                        ty = self.parse_type_node(child, source)?;
                    }
                    "field_identifier" => {
                        name = self.node_text(child, source);
                    }
                    "pointer_declarator" => {
                        ty = CType::Pointer(Box::new(ty));
                        if let Some(inner_name) = self.find_field_identifier(child, source) {
                            name = inner_name;
                        }
                    }
                    "array_declarator" => {
                        if let Some((array_name, size)) =
                            self.parse_array_declarator(child, source)?
                        {
                            name = array_name;
                            ty = CType::Array(Box::new(ty), size);
                        }
                    }
                    _ => {}
                }
            }
        }

        if name.is_empty() {
            Ok(None)
        } else {
            Ok(Some(StructField { name, ty }))
        }
    }

    /// Parse array declarator
    fn parse_array_declarator(
        &self,
        node: Node,
        source: &str,
    ) -> ParseResult<Option<(String, usize)>> {
        let mut name = String::new();
        let mut size = 0usize;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "identifier" | "field_identifier" => {
                        name = self.node_text(child, source);
                    }
                    "number_literal" => {
                        let text = self.node_text(child, source);
                        size = self.parse_int_literal(&text)? as usize;
                    }
                    _ => {}
                }
            }
        }

        if name.is_empty() {
            Ok(None)
        } else {
            Ok(Some((name, size)))
        }
    }

    /// Parse compound statement (block)
    fn parse_compound_stmt(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        let mut stmts = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                // Skip braces
                if child.kind() == "{" || child.kind() == "}" {
                    continue;
                }

                let stmt = self.parse_stmt(child, source)?;
                if !matches!(stmt, CStmt::Empty) {
                    stmts.push(stmt);
                }
            }
        }

        Ok(CStmt::Block(stmts))
    }

    /// Parse a statement
    fn parse_stmt(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        match node.kind() {
            "compound_statement" => self.parse_compound_stmt(node, source),
            "expression_statement" => {
                if let Some(child) = node.child_at(0) {
                    if child.kind() == ";" {
                        return Ok(CStmt::Empty);
                    }
                    let expr = self.parse_expr(child, source)?;
                    Ok(CStmt::Expr(expr))
                } else {
                    Ok(CStmt::Empty)
                }
            }
            "declaration" => self.parse_declaration(node, source),
            "if_statement" => self.parse_if_stmt(node, source),
            "while_statement" => self.parse_while_stmt(node, source),
            "for_statement" => self.parse_for_stmt(node, source),
            "do_statement" => self.parse_do_stmt(node, source),
            "return_statement" => self.parse_return_stmt(node, source),
            "break_statement" => Ok(CStmt::Break),
            "continue_statement" => Ok(CStmt::Continue),
            "goto_statement" => self.parse_goto_stmt(node, source),
            "labeled_statement" => self.parse_labeled_stmt(node, source),
            "switch_statement" => self.parse_switch_stmt(node, source),
            ";" => Ok(CStmt::Empty),
            _ => {
                // Try parsing as expression
                if let Ok(expr) = self.parse_expr(node, source) {
                    Ok(CStmt::Expr(expr))
                } else {
                    Err(ParseError::Unsupported {
                        line: node.start_position().row + 1,
                        column: node.start_position().column + 1,
                        kind: node.kind().to_string(),
                    })
                }
            }
        }
    }

    /// Parse declaration statement
    fn parse_declaration(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        let mut ty = CType::Void;
        let mut storage = StorageClass::Auto;
        let mut decls = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "storage_class_specifier" => {
                        let text = self.node_text(child, source);
                        storage = match text.as_str() {
                            "static" => StorageClass::Static,
                            "extern" => StorageClass::Extern,
                            "register" => StorageClass::Register,
                            "_Thread_local" | "thread_local" => StorageClass::ThreadLocal,
                            _ => StorageClass::Auto,
                        };
                    }
                    "primitive_type"
                    | "type_identifier"
                    | "sized_type_specifier"
                    | "struct_specifier"
                    | "union_specifier"
                    | "enum_specifier" => {
                        ty = self.parse_type_node(child, source)?;
                    }
                    "init_declarator" => {
                        let decl =
                            self.parse_init_declarator(child, source, ty.clone(), storage)?;
                        decls.push(decl);
                    }
                    "identifier" => {
                        let name = self.node_text(child, source);
                        decls.push(VarDecl::new(name, ty.clone()).with_storage(storage));
                    }
                    "pointer_declarator" => {
                        let ptr_ty = CType::Pointer(Box::new(ty.clone()));
                        if let Some(name) = self.find_identifier(child, source) {
                            decls.push(VarDecl::new(name, ptr_ty).with_storage(storage));
                        }
                    }
                    "array_declarator" => {
                        if let Some((name, size)) = self.parse_array_declarator(child, source)? {
                            let arr_ty = CType::Array(Box::new(ty.clone()), size);
                            decls.push(VarDecl::new(name, arr_ty).with_storage(storage));
                        }
                    }
                    _ => {}
                }
            }
        }

        // Convert to statements
        if decls.is_empty() {
            Ok(CStmt::Empty)
        } else if decls.len() == 1 {
            Ok(CStmt::Decl(decls.into_iter().next().unwrap()))
        } else {
            Ok(CStmt::Block(decls.into_iter().map(CStmt::Decl).collect()))
        }
    }

    /// Parse init_declarator
    fn parse_init_declarator(
        &self,
        node: Node,
        source: &str,
        base_ty: CType,
        storage: StorageClass,
    ) -> ParseResult<VarDecl> {
        let mut name = String::new();
        let mut ty = base_ty;
        let mut init = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "identifier" => {
                        name = self.node_text(child, source);
                    }
                    "pointer_declarator" => {
                        ty = CType::Pointer(Box::new(ty));
                        if let Some(inner_name) = self.find_identifier(child, source) {
                            name = inner_name;
                        }
                    }
                    "array_declarator" => {
                        if let Some((arr_name, size)) =
                            self.parse_array_declarator(child, source)?
                        {
                            name = arr_name;
                            ty = CType::Array(Box::new(ty), size);
                        }
                    }
                    "=" => {} // Skip assignment operator
                    _ => {
                        // Try to parse as initializer
                        if let Ok(expr) = self.parse_expr(child, source) {
                            init = Some(Initializer::Expr(expr));
                        } else if child.kind() == "initializer_list" {
                            init = Some(self.parse_initializer_list(child, source)?);
                        }
                    }
                }
            }
        }

        let mut decl = VarDecl::new(name, ty).with_storage(storage);
        if let Some(initializer) = init {
            decl = decl.with_init(initializer);
        }
        Ok(decl)
    }

    /// Parse initializer list
    fn parse_initializer_list(&self, node: Node, source: &str) -> ParseResult<Initializer> {
        let mut items = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "{" | "}" | "," => continue,
                    "initializer_list" => {
                        items.push(self.parse_initializer_list(child, source)?);
                    }
                    _ => {
                        if let Ok(expr) = self.parse_expr(child, source) {
                            items.push(Initializer::Expr(expr));
                        }
                    }
                }
            }
        }

        Ok(Initializer::List(items))
    }

    /// Parse if statement
    fn parse_if_stmt(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        let mut cond = None;
        let mut then_branch = None;
        let mut else_branch = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "parenthesized_expression" => {
                        // Get the inner expression
                        if let Some(inner) = child.child_at(1) {
                            cond = Some(self.parse_expr(inner, source)?);
                        }
                    }
                    "if" | "else" | "(" | ")" => {} // Skip keywords
                    _ => {
                        if cond.is_some() && then_branch.is_none() {
                            then_branch = Some(Box::new(self.parse_stmt(child, source)?));
                        } else if then_branch.is_some() {
                            else_branch = Some(Box::new(self.parse_stmt(child, source)?));
                        }
                    }
                }
            }
        }

        Ok(CStmt::If {
            cond: cond.unwrap_or(CExpr::IntLit(1)),
            then_stmt: then_branch.unwrap_or_else(|| Box::new(CStmt::Empty)),
            else_stmt: else_branch,
        })
    }

    /// Parse while statement
    fn parse_while_stmt(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        let mut cond = None;
        let mut body = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "parenthesized_expression" => {
                        if let Some(inner) = child.child_at(1) {
                            cond = Some(self.parse_expr(inner, source)?);
                        }
                    }
                    "while" | "(" | ")" => {}
                    _ => {
                        body = Some(Box::new(self.parse_stmt(child, source)?));
                    }
                }
            }
        }

        Ok(CStmt::While {
            cond: cond.unwrap_or(CExpr::IntLit(1)),
            body: body.unwrap_or_else(|| Box::new(CStmt::Empty)),
        })
    }

    /// Parse for statement
    fn parse_for_stmt(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        let mut init = None;
        let mut cond = None;
        let mut update = None;
        let mut body = None;
        let mut phase = 0; // 0=init, 1=cond, 2=update, 3=body

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "for" | "(" | ")" => {}
                    ";" => {
                        phase += 1;
                    }
                    "declaration" => {
                        if phase == 0 {
                            init = Some(Box::new(self.parse_declaration(child, source)?));
                        }
                    }
                    _ => {
                        if phase == 0 {
                            if let Ok(expr) = self.parse_expr(child, source) {
                                init = Some(Box::new(CStmt::Expr(expr)));
                            }
                        } else if phase == 1 {
                            if let Ok(expr) = self.parse_expr(child, source) {
                                cond = Some(expr);
                            }
                        } else if phase == 2 {
                            if let Ok(expr) = self.parse_expr(child, source) {
                                update = Some(expr);
                            }
                        } else {
                            body = Some(Box::new(self.parse_stmt(child, source)?));
                        }
                    }
                }
            }
        }

        Ok(CStmt::For {
            init,
            cond,
            update,
            body: body.unwrap_or_else(|| Box::new(CStmt::Empty)),
        })
    }

    /// Parse do-while statement
    fn parse_do_stmt(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        let mut body = None;
        let mut cond = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "do" | "while" | "(" | ")" | ";" => {}
                    "parenthesized_expression" => {
                        if let Some(inner) = child.child_at(1) {
                            cond = Some(self.parse_expr(inner, source)?);
                        }
                    }
                    _ => {
                        body = Some(Box::new(self.parse_stmt(child, source)?));
                    }
                }
            }
        }

        Ok(CStmt::DoWhile {
            body: body.unwrap_or_else(|| Box::new(CStmt::Empty)),
            cond: cond.unwrap_or(CExpr::IntLit(1)),
        })
    }

    /// Parse return statement
    fn parse_return_stmt(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                if child.kind() != "return" && child.kind() != ";" {
                    let expr = self.parse_expr(child, source)?;
                    return Ok(CStmt::Return(Some(expr)));
                }
            }
        }
        Ok(CStmt::Return(None))
    }

    /// Parse goto statement
    fn parse_goto_stmt(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                // tree-sitter-c uses "statement_identifier" for goto labels
                if child.kind() == "identifier" || child.kind() == "statement_identifier" {
                    return Ok(CStmt::Goto(self.node_text(child, source)));
                }
            }
        }
        Err(ParseError::MissingField {
            field: "label".to_string(),
            node_kind: "goto_statement".to_string(),
        })
    }

    /// Parse labeled statement
    fn parse_labeled_stmt(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        let mut label = String::new();
        let mut stmt = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    // tree-sitter-c uses "statement_identifier" for labels
                    "statement_identifier" | "identifier" => {
                        label = self.node_text(child, source);
                    }
                    ":" => {}
                    _ => {
                        stmt = Some(Box::new(self.parse_stmt(child, source)?));
                    }
                }
            }
        }

        Ok(CStmt::Label {
            name: label,
            stmt: stmt.unwrap_or_else(|| Box::new(CStmt::Empty)),
        })
    }

    /// Parse switch statement
    fn parse_switch_stmt(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        let mut cond = None;
        let mut body = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "parenthesized_expression" => {
                        if let Some(inner) = child.child_at(1) {
                            cond = Some(self.parse_expr(inner, source)?);
                        }
                    }
                    "compound_statement" => {
                        body = Some(Box::new(self.parse_switch_body(child, source)?));
                    }
                    _ => {}
                }
            }
        }

        Ok(CStmt::Switch {
            cond: cond.unwrap_or(CExpr::IntLit(0)),
            body: body.unwrap_or_else(|| Box::new(CStmt::Empty)),
        })
    }

    /// Parse switch body (convert case labels to Case statements)
    fn parse_switch_body(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        let mut stmts = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "{" | "}" => {}
                    "case_statement" => {
                        let stmt = self.parse_case_stmt(child, source)?;
                        stmts.push(stmt);
                    }
                    _ => {
                        let stmt = self.parse_stmt(child, source)?;
                        if !matches!(stmt, CStmt::Empty) {
                            stmts.push(stmt);
                        }
                    }
                }
            }
        }

        Ok(CStmt::Block(stmts))
    }

    /// Parse case statement
    fn parse_case_stmt(&self, node: Node, source: &str) -> ParseResult<CStmt> {
        let mut label = crate::stmt::CaseLabel::Default;
        let mut body_stmts = Vec::new();
        let mut after_colon = false;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "case" => {}
                    "default" => {
                        label = crate::stmt::CaseLabel::Default;
                    }
                    ":" => {
                        after_colon = true;
                    }
                    _ => {
                        if after_colon {
                            // This is a statement in the case
                            let stmt = self.parse_stmt(child, source)?;
                            if !matches!(stmt, CStmt::Empty) {
                                body_stmts.push(stmt);
                            }
                        } else {
                            // This is the case expression
                            if let Ok(expr) = self.parse_expr(child, source) {
                                label = crate::stmt::CaseLabel::Case(expr);
                            }
                        }
                    }
                }
            }
        }

        // Create a Case statement containing all body statements
        let body = if body_stmts.is_empty() {
            CStmt::Empty
        } else if body_stmts.len() == 1 {
            body_stmts.into_iter().next().unwrap()
        } else {
            CStmt::Block(body_stmts)
        };

        Ok(CStmt::Case {
            label,
            stmt: Box::new(body),
        })
    }

    /// Parse an expression
    fn parse_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        match node.kind() {
            "number_literal" => {
                let text = self.node_text(node, source);
                self.parse_number_literal(&text)
            }
            "char_literal" => {
                let text = self.node_text(node, source);
                self.parse_char_literal(&text)
            }
            "string_literal" => {
                let text = self.node_text(node, source);
                // Remove quotes
                let inner = text.trim_start_matches('"').trim_end_matches('"');
                Ok(CExpr::StringLit(inner.to_string()))
            }
            "identifier" => Ok(CExpr::Var(self.node_text(node, source))),
            "true" => Ok(CExpr::IntLit(1)),
            "false" => Ok(CExpr::IntLit(0)),
            "null" | "NULL" => Ok(CExpr::null()),
            "parenthesized_expression" => {
                // Get inner expression
                if let Some(inner) = node.child_at(1) {
                    self.parse_expr(inner, source)
                } else {
                    Err(ParseError::MissingField {
                        field: "expression".to_string(),
                        node_kind: "parenthesized_expression".to_string(),
                    })
                }
            }
            "binary_expression" => self.parse_binary_expr(node, source),
            "unary_expression" => self.parse_unary_expr(node, source),
            "update_expression" => self.parse_update_expr(node, source),
            "assignment_expression" => self.parse_assignment_expr(node, source),
            "conditional_expression" => self.parse_conditional_expr(node, source),
            "call_expression" => self.parse_call_expr(node, source),
            "cast_expression" => self.parse_cast_expr(node, source),
            "subscript_expression" => self.parse_subscript_expr(node, source),
            "field_expression" => self.parse_field_expr(node, source),
            "pointer_expression" => self.parse_pointer_expr(node, source),
            "sizeof_expression" => self.parse_sizeof_expr(node, source),
            "comma_expression" => self.parse_comma_expr(node, source),
            _ => Err(ParseError::Unsupported {
                line: node.start_position().row + 1,
                column: node.start_position().column + 1,
                kind: format!("expression: {}", node.kind()),
            }),
        }
    }

    /// Parse number literal
    fn parse_number_literal(&self, text: &str) -> ParseResult<CExpr> {
        let text = text.trim().trim_end_matches(['u', 'U', 'l', 'L']);

        // Check for hex
        if text.starts_with("0x") || text.starts_with("0X") {
            let val = i64::from_str_radix(&text[2..], 16).map_err(|_| ParseError::InvalidInt {
                value: text.to_string(),
            })?;
            return Ok(CExpr::IntLit(val));
        }

        // Check for octal
        if text.starts_with('0') && text.len() > 1 && !text.contains('.') {
            if let Ok(val) = i64::from_str_radix(&text[1..], 8) {
                return Ok(CExpr::IntLit(val));
            }
        }

        // Check for float
        if text.contains('.') || text.contains('e') || text.contains('E') {
            let val: f64 = text.parse().map_err(|_| ParseError::InvalidFloat {
                value: text.to_string(),
            })?;
            return Ok(CExpr::FloatLit(val));
        }

        // Decimal integer
        let val: i64 = text.parse().map_err(|_| ParseError::InvalidInt {
            value: text.to_string(),
        })?;
        Ok(CExpr::IntLit(val))
    }

    /// Parse char literal
    fn parse_char_literal(&self, text: &str) -> ParseResult<CExpr> {
        let inner = text.trim_start_matches('\'').trim_end_matches('\'');

        let ch = if inner.starts_with('\\') {
            match inner.chars().nth(1) {
                Some('n') => b'\n',
                Some('r') => b'\r',
                Some('t') => b'\t',
                Some('0') | None => b'\0',
                Some('\\') => b'\\',
                Some('\'') => b'\'',
                Some('"') => b'"',
                Some(c) => c as u8,
            }
        } else {
            inner.chars().next().unwrap_or('\0') as u8
        };

        Ok(CExpr::CharLit(ch))
    }

    /// Parse int literal (for enums, etc)
    fn parse_int_literal(&self, text: &str) -> ParseResult<i64> {
        let text = text.trim().trim_end_matches(['u', 'U', 'l', 'L']);

        if text.starts_with("0x") || text.starts_with("0X") {
            return i64::from_str_radix(&text[2..], 16).map_err(|_| ParseError::InvalidInt {
                value: text.to_string(),
            });
        }

        if text.starts_with('0') && text.len() > 1 {
            return i64::from_str_radix(&text[1..], 8).map_err(|_| ParseError::InvalidInt {
                value: text.to_string(),
            });
        }

        text.parse().map_err(|_| ParseError::InvalidInt {
            value: text.to_string(),
        })
    }

    /// Parse binary expression
    fn parse_binary_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        let mut left = None;
        let mut op = None;
        let mut right = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                let kind = child.kind();
                if let Some(bin_op) = self.parse_binary_op(kind) {
                    op = Some(bin_op);
                } else if left.is_none() {
                    left = Some(self.parse_expr(child, source)?);
                } else {
                    right = Some(self.parse_expr(child, source)?);
                }
            }
        }

        Ok(CExpr::BinOp {
            op: op.ok_or_else(|| ParseError::MissingField {
                field: "operator".to_string(),
                node_kind: "binary_expression".to_string(),
            })?,
            left: Box::new(left.ok_or_else(|| ParseError::MissingField {
                field: "left".to_string(),
                node_kind: "binary_expression".to_string(),
            })?),
            right: Box::new(right.ok_or_else(|| ParseError::MissingField {
                field: "right".to_string(),
                node_kind: "binary_expression".to_string(),
            })?),
        })
    }

    /// Parse binary operator
    fn parse_binary_op(&self, kind: &str) -> Option<BinOp> {
        match kind {
            "+" => Some(BinOp::Add),
            "-" => Some(BinOp::Sub),
            "*" => Some(BinOp::Mul),
            "/" => Some(BinOp::Div),
            "%" => Some(BinOp::Mod),
            "&" => Some(BinOp::BitAnd),
            "|" => Some(BinOp::BitOr),
            "^" => Some(BinOp::BitXor),
            "<<" => Some(BinOp::Shl),
            ">>" => Some(BinOp::Shr),
            "==" => Some(BinOp::Eq),
            "!=" => Some(BinOp::Ne),
            "<" => Some(BinOp::Lt),
            "<=" => Some(BinOp::Le),
            ">" => Some(BinOp::Gt),
            ">=" => Some(BinOp::Ge),
            "&&" => Some(BinOp::LogAnd),
            "||" => Some(BinOp::LogOr),
            "," => Some(BinOp::Comma),
            _ => None,
        }
    }

    /// Parse unary expression
    fn parse_unary_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        let mut op = None;
        let mut operand = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                let kind = child.kind();
                if let Some(unary_op) = self.parse_unary_op(kind) {
                    op = Some(unary_op);
                } else {
                    operand = Some(self.parse_expr(child, source)?);
                }
            }
        }

        Ok(CExpr::UnaryOp {
            op: op.ok_or_else(|| ParseError::MissingField {
                field: "operator".to_string(),
                node_kind: "unary_expression".to_string(),
            })?,
            operand: Box::new(operand.ok_or_else(|| ParseError::MissingField {
                field: "operand".to_string(),
                node_kind: "unary_expression".to_string(),
            })?),
        })
    }

    /// Parse unary operator
    fn parse_unary_op(&self, kind: &str) -> Option<UnaryOp> {
        match kind {
            "-" => Some(UnaryOp::Neg),
            "+" => Some(UnaryOp::Pos),
            "~" => Some(UnaryOp::BitNot),
            "!" => Some(UnaryOp::LogNot),
            "&" => Some(UnaryOp::AddrOf),
            "*" => Some(UnaryOp::Deref),
            _ => None,
        }
    }

    /// Parse update expression (++x, --x, x++, x--)
    fn parse_update_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        let mut op = None;
        let mut operand = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                let kind = child.kind();
                match kind {
                    "++" => {
                        let is_prefix = operand.is_none();
                        op = Some(if is_prefix {
                            UnaryOp::PreInc
                        } else {
                            UnaryOp::PostInc
                        });
                    }
                    "--" => {
                        let is_prefix = operand.is_none();
                        op = Some(if is_prefix {
                            UnaryOp::PreDec
                        } else {
                            UnaryOp::PostDec
                        });
                    }
                    _ => {
                        operand = Some(self.parse_expr(child, source)?);
                    }
                }
            }
        }

        Ok(CExpr::UnaryOp {
            op: op.ok_or_else(|| ParseError::MissingField {
                field: "operator".to_string(),
                node_kind: "update_expression".to_string(),
            })?,
            operand: Box::new(operand.ok_or_else(|| ParseError::MissingField {
                field: "operand".to_string(),
                node_kind: "update_expression".to_string(),
            })?),
        })
    }

    /// Parse assignment expression
    fn parse_assignment_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        let mut left = None;
        let mut op = None;
        let mut right = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                let kind = child.kind();
                if let Some(assign_op) = self.parse_assign_op(kind) {
                    op = Some(assign_op);
                } else if left.is_none() {
                    left = Some(self.parse_expr(child, source)?);
                } else {
                    right = Some(self.parse_expr(child, source)?);
                }
            }
        }

        Ok(CExpr::BinOp {
            op: op.ok_or_else(|| ParseError::MissingField {
                field: "operator".to_string(),
                node_kind: "assignment_expression".to_string(),
            })?,
            left: Box::new(left.ok_or_else(|| ParseError::MissingField {
                field: "left".to_string(),
                node_kind: "assignment_expression".to_string(),
            })?),
            right: Box::new(right.ok_or_else(|| ParseError::MissingField {
                field: "right".to_string(),
                node_kind: "assignment_expression".to_string(),
            })?),
        })
    }

    /// Parse assignment operator
    fn parse_assign_op(&self, kind: &str) -> Option<BinOp> {
        match kind {
            "=" => Some(BinOp::Assign),
            "+=" => Some(BinOp::AddAssign),
            "-=" => Some(BinOp::SubAssign),
            "*=" => Some(BinOp::MulAssign),
            "/=" => Some(BinOp::DivAssign),
            "%=" => Some(BinOp::ModAssign),
            "&=" => Some(BinOp::BitAndAssign),
            "|=" => Some(BinOp::BitOrAssign),
            "^=" => Some(BinOp::BitXorAssign),
            "<<=" => Some(BinOp::ShlAssign),
            ">>=" => Some(BinOp::ShrAssign),
            _ => None,
        }
    }

    /// Parse conditional expression (ternary)
    fn parse_conditional_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        let mut cond = None;
        let mut then_expr = None;
        let mut else_expr = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                let kind = child.kind();
                if kind == "?" || kind == ":" {
                    continue;
                }
                if cond.is_none() {
                    cond = Some(self.parse_expr(child, source)?);
                } else if then_expr.is_none() {
                    then_expr = Some(self.parse_expr(child, source)?);
                } else {
                    else_expr = Some(self.parse_expr(child, source)?);
                }
            }
        }

        Ok(CExpr::Conditional {
            cond: Box::new(cond.ok_or_else(|| ParseError::MissingField {
                field: "condition".to_string(),
                node_kind: "conditional_expression".to_string(),
            })?),
            then_expr: Box::new(then_expr.ok_or_else(|| ParseError::MissingField {
                field: "then".to_string(),
                node_kind: "conditional_expression".to_string(),
            })?),
            else_expr: Box::new(else_expr.ok_or_else(|| ParseError::MissingField {
                field: "else".to_string(),
                node_kind: "conditional_expression".to_string(),
            })?),
        })
    }

    /// Parse function call expression
    fn parse_call_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        let mut func = None;
        let mut args = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "identifier" | "field_expression" => {
                        func = Some(self.parse_expr(child, source)?);
                    }
                    "argument_list" => {
                        args = self.parse_arg_list(child, source)?;
                    }
                    _ => {}
                }
            }
        }

        Ok(CExpr::Call {
            func: Box::new(func.ok_or_else(|| ParseError::MissingField {
                field: "function".to_string(),
                node_kind: "call_expression".to_string(),
            })?),
            args,
        })
    }

    /// Parse argument list
    fn parse_arg_list(&self, node: Node, source: &str) -> ParseResult<Vec<CExpr>> {
        let mut args = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                if child.kind() != "(" && child.kind() != ")" && child.kind() != "," {
                    args.push(self.parse_expr(child, source)?);
                }
            }
        }

        Ok(args)
    }

    /// Parse cast expression
    fn parse_cast_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        let mut ty = CType::Void;
        let mut expr = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "(" | ")" => {}
                    "type_descriptor" => {
                        ty = self.parse_type_descriptor(child, source)?;
                    }
                    _ => {
                        expr = Some(self.parse_expr(child, source)?);
                    }
                }
            }
        }

        Ok(CExpr::Cast {
            ty,
            expr: Box::new(expr.ok_or_else(|| ParseError::MissingField {
                field: "expression".to_string(),
                node_kind: "cast_expression".to_string(),
            })?),
        })
    }

    /// Parse type descriptor
    fn parse_type_descriptor(&self, node: Node, source: &str) -> ParseResult<CType> {
        let mut ty = CType::Void;
        let mut pointer_count = 0;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "primitive_type"
                    | "type_identifier"
                    | "sized_type_specifier"
                    | "struct_specifier"
                    | "union_specifier"
                    | "enum_specifier" => {
                        ty = self.parse_type_node(child, source)?;
                    }
                    "abstract_pointer_declarator" => {
                        pointer_count += self.count_pointers(child);
                    }
                    "*" => {
                        pointer_count += 1;
                    }
                    _ => {}
                }
            }
        }

        for _ in 0..pointer_count {
            ty = CType::Pointer(Box::new(ty));
        }

        Ok(ty)
    }

    /// Count pointer levels
    fn count_pointers(&self, node: Node) -> usize {
        let mut count = 0;
        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                if child.kind() == "*" {
                    count += 1;
                } else if child.kind() == "abstract_pointer_declarator" {
                    count += self.count_pointers(child);
                }
            }
        }
        if count == 0 {
            1
        } else {
            count
        }
    }

    /// Parse subscript expression (array access)
    fn parse_subscript_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        let mut array = None;
        let mut index = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                if child.kind() == "[" || child.kind() == "]" {
                    continue;
                }
                if array.is_none() {
                    array = Some(self.parse_expr(child, source)?);
                } else {
                    index = Some(self.parse_expr(child, source)?);
                }
            }
        }

        Ok(CExpr::Index {
            array: Box::new(array.ok_or_else(|| ParseError::MissingField {
                field: "array".to_string(),
                node_kind: "subscript_expression".to_string(),
            })?),
            index: Box::new(index.ok_or_else(|| ParseError::MissingField {
                field: "index".to_string(),
                node_kind: "subscript_expression".to_string(),
            })?),
        })
    }

    /// Parse field expression (struct.field or struct->field)
    fn parse_field_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        let mut base = None;
        let mut field = String::new();
        let mut is_ptr = false;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "." => is_ptr = false,
                    "->" => is_ptr = true,
                    "field_identifier" => {
                        field = self.node_text(child, source);
                    }
                    _ => {
                        base = Some(self.parse_expr(child, source)?);
                    }
                }
            }
        }

        let base_expr = base.ok_or_else(|| ParseError::MissingField {
            field: "base".to_string(),
            node_kind: "field_expression".to_string(),
        })?;

        if is_ptr {
            // a->b
            Ok(CExpr::Arrow {
                pointer: Box::new(base_expr),
                field,
            })
        } else {
            // a.b
            Ok(CExpr::Member {
                object: Box::new(base_expr),
                field,
            })
        }
    }

    /// Parse pointer expression (& or *)
    fn parse_pointer_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        let mut op = None;
        let mut operand = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "&" => op = Some(UnaryOp::AddrOf),
                    "*" => op = Some(UnaryOp::Deref),
                    _ => {
                        operand = Some(self.parse_expr(child, source)?);
                    }
                }
            }
        }

        Ok(CExpr::UnaryOp {
            op: op.ok_or_else(|| ParseError::MissingField {
                field: "operator".to_string(),
                node_kind: "pointer_expression".to_string(),
            })?,
            operand: Box::new(operand.ok_or_else(|| ParseError::MissingField {
                field: "operand".to_string(),
                node_kind: "pointer_expression".to_string(),
            })?),
        })
    }

    /// Parse sizeof expression
    fn parse_sizeof_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                match child.kind() {
                    "sizeof" | "(" | ")" => {}
                    "type_descriptor" => {
                        let ty = self.parse_type_descriptor(child, source)?;
                        return Ok(CExpr::SizeOf(SizeOfArg::Type(ty)));
                    }
                    "parenthesized_expression" => {
                        // sizeof(expr)
                        if let Some(inner) = child.child_at(1) {
                            let expr = self.parse_expr(inner, source)?;
                            return Ok(CExpr::SizeOf(SizeOfArg::Expr(Box::new(expr))));
                        }
                    }
                    _ => {
                        let expr = self.parse_expr(child, source)?;
                        return Ok(CExpr::SizeOf(SizeOfArg::Expr(Box::new(expr))));
                    }
                }
            }
        }

        Err(ParseError::MissingField {
            field: "operand".to_string(),
            node_kind: "sizeof_expression".to_string(),
        })
    }

    /// Parse comma expression
    fn parse_comma_expr(&self, node: Node, source: &str) -> ParseResult<CExpr> {
        let mut left = None;
        let mut right = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                if child.kind() == "," {
                    continue;
                }
                if left.is_none() {
                    left = Some(self.parse_expr(child, source)?);
                } else {
                    right = Some(self.parse_expr(child, source)?);
                }
            }
        }

        Ok(CExpr::BinOp {
            op: BinOp::Comma,
            left: Box::new(left.unwrap_or(CExpr::IntLit(0))),
            right: Box::new(right.unwrap_or(CExpr::IntLit(0))),
        })
    }

    /// Helper: get text of a node
    fn node_text(&self, node: Node, source: &str) -> String {
        source[node.start_byte()..node.end_byte()].to_string()
    }

    /// Helper: find identifier in nested declarators
    fn find_identifier(&self, node: Node, source: &str) -> Option<String> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                if child.kind() == "identifier" {
                    return Some(self.node_text(child, source));
                }
                if let Some(found) = self.find_identifier(child, source) {
                    return Some(found);
                }
            }
        }
        None
    }

    /// Helper: find field_identifier in nested declarators
    fn find_field_identifier(&self, node: Node, source: &str) -> Option<String> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child_at(i) {
                if child.kind() == "field_identifier" {
                    return Some(self.node_text(child, source));
                }
                if let Some(found) = self.find_field_identifier(child, source) {
                    return Some(found);
                }
            }
        }
        None
    }

    /// Extract ACSL spec attached to a function (block or line comment)
    fn extract_func_spec(&self, node: Node, source: &str) -> Option<FuncSpec> {
        let start_byte = node.start_byte();
        let nearest_comment = find_nearest_acsl_comment(source, start_byte)?;
        parse_acsl_spec(&nearest_comment)
    }
}

/// Parse ACSL-style specification comments
pub fn parse_acsl_spec(comment: &str) -> Option<FuncSpec> {
    use crate::spec::Location;

    // Look for /*@ ... */ or //@ ...
    let content = if let Some(inner) = comment
        .strip_prefix("/*@")
        .and_then(|s| s.strip_suffix("*/"))
    {
        inner
    } else if let Some(inner) = comment.strip_prefix("//@") {
        inner
    } else {
        return None;
    };

    let mut requires = Vec::new();
    let mut ensures = Vec::new();
    let mut assigns: Vec<Location> = Vec::new();

    for line in content.lines() {
        let mut line = line.trim();
        if let Some(rest) = line.strip_prefix("//@") {
            line = rest.trim();
        }
        let line = line.trim_start_matches('@').trim();
        if let Some(rest) = line.strip_prefix("requires") {
            requires.push(parse_spec_expr(rest.trim().trim_end_matches(';')));
        } else if let Some(rest) = line.strip_prefix("ensures") {
            ensures.push(parse_spec_expr(rest.trim().trim_end_matches(';')));
        } else if let Some(rest) = line.strip_prefix("assigns") {
            let text = rest.trim().trim_end_matches(';');
            if text == "\\nothing" {
                assigns.push(Location::Nothing);
            } else {
                // Parse each assigned location
                let parts: Vec<&str> = text.split(',').collect();
                for part in parts {
                    let part = part.trim();
                    if part == "\\nothing" {
                        assigns.push(Location::Nothing);
                    } else {
                        // Treat as a dereference of the variable
                        assigns.push(Location::Deref(parse_spec_expr(part)));
                    }
                }
            }
        }
    }

    Some(FuncSpec {
        requires,
        ensures,
        assigns,
        ..Default::default()
    })
}

/// Find the ACSL comment nearest to the given byte offset
fn find_nearest_acsl_comment(source: &str, start_byte: usize) -> Option<String> {
    let block = find_block_acsl_comment_before(source, start_byte);
    let line = find_line_acsl_comment_before(source, start_byte);

    match (block, line) {
        (Some((text, end)), Some((line_text, line_end))) => {
            if line_end > end {
                Some(line_text)
            } else {
                Some(text)
            }
        }
        (Some((text, _)), None) | (None, Some((text, _))) => Some(text),
        (None, None) => None,
    }
}

/// Find the last block ACSL comment (/*@ ... */) before the given byte offset
fn find_block_acsl_comment_before(source: &str, start_byte: usize) -> Option<(String, usize)> {
    if start_byte > source.len() {
        return None;
    }

    let prefix = &source[..start_byte];
    let start_idx = prefix.rfind("/*@")?;
    let remainder = &source[start_idx..start_byte];
    let end_rel = remainder.find("*/")?;
    let end_idx = start_idx + end_rel + 2;

    // Ensure only whitespace between comment end and the target offset
    if source[end_idx..start_byte].trim().is_empty() {
        Some((source[start_idx..end_idx].to_string(), end_idx))
    } else {
        None
    }
}

/// Find the trailing block of line ACSL comments (//@ ...) before the offset
fn find_line_acsl_comment_before(source: &str, start_byte: usize) -> Option<(String, usize)> {
    let prefix = &source[..start_byte];
    let trimmed = prefix.trim_end_matches(|c: char| c.is_whitespace());
    let trimmed_len = trimmed.len();

    if trimmed_len == 0 {
        return None;
    }

    let mut collected = Vec::new();
    for line in trimmed.rsplit('\n') {
        let trimmed_line = line.trim_start();
        if trimmed_line.starts_with("//@") {
            collected.push(trimmed_line.to_string());
        } else if trimmed_line.is_empty() {
            if collected.is_empty() {
                continue;
            }
            break;
        } else {
            break;
        }
    }

    if collected.is_empty() {
        return None;
    }

    collected.reverse();
    Some((collected.join("\n"), trimmed_len))
}

/// Parse a simple specification expression
fn parse_spec_expr(s: &str) -> Spec {
    let s = s.trim();

    // Handle \result
    if s == "\\result" {
        return Spec::Result;
    }

    // Handle \old(x)
    if let Some(inner) = s.strip_prefix("\\old(").and_then(|s| s.strip_suffix(')')) {
        return Spec::Old(Box::new(parse_spec_expr(inner)));
    }

    // Handle \valid(p)
    if let Some(inner) = s.strip_prefix("\\valid(").and_then(|s| s.strip_suffix(')')) {
        return Spec::Valid(Box::new(parse_spec_expr(inner)));
    }

    // Handle comparisons using BinOp variant
    if let Some((left, right)) = s.split_once("==") {
        return Spec::eq(parse_spec_expr(left), parse_spec_expr(right));
    }
    if let Some((left, right)) = s.split_once("!=") {
        return Spec::ne(parse_spec_expr(left), parse_spec_expr(right));
    }
    if let Some((left, right)) = s.split_once("<=") {
        return Spec::le(parse_spec_expr(left), parse_spec_expr(right));
    }
    if let Some((left, right)) = s.split_once(">=") {
        return Spec::ge(parse_spec_expr(left), parse_spec_expr(right));
    }
    if let Some((left, right)) = s.split_once('<') {
        return Spec::lt(parse_spec_expr(left), parse_spec_expr(right));
    }
    if let Some((left, right)) = s.split_once('>') {
        return Spec::gt(parse_spec_expr(left), parse_spec_expr(right));
    }

    // Handle integers
    if let Ok(n) = s.parse::<i64>() {
        return Spec::Int(n);
    }

    // Default: treat as variable
    Spec::Var(s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_function() {
        let mut parser = CParser::new();
        let code = r"
            int add(int a, int b) {
                return a + b;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "add");
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.params[0].name, "a");
        assert_eq!(func.params[1].name, "b");
    }

    #[test]
    fn test_parse_void_function() {
        let mut parser = CParser::new();
        let code = r"
            void noop(void) {
                return;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "noop");
        assert!(matches!(func.return_type, CType::Void));
    }

    #[test]
    fn test_parse_pointer_params() {
        let mut parser = CParser::new();
        let code = r"
            void swap(int *a, int *b) {
                int tmp = *a;
                *a = *b;
                *b = tmp;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "swap");
        assert_eq!(func.params.len(), 2);
        assert!(matches!(func.params[0].ty, CType::Pointer(_)));
    }

    #[test]
    fn test_parse_if_statement() {
        let mut parser = CParser::new();
        let code = r"
            int abs(int x) {
                if (x < 0) {
                    return -x;
                }
                return x;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "abs");

        // Check body contains if statement
        match func.body.as_ref() {
            CStmt::Block(stmts) => {
                assert!(stmts.len() >= 2);
                assert!(matches!(stmts[0], CStmt::If { .. }));
            }
            _ => panic!("Expected block"),
        }
    }

    #[test]
    fn test_parse_for_loop() {
        let mut parser = CParser::new();
        let code = r"
            int sum(int n) {
                int total = 0;
                for (int i = 0; i < n; i++) {
                    total += i;
                }
                return total;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "sum");
    }

    #[test]
    fn test_parse_while_loop() {
        let mut parser = CParser::new();
        let code = r"
            int count_down(int n) {
                while (n > 0) {
                    n--;
                }
                return n;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "count_down");
    }

    #[test]
    fn test_parse_struct() {
        let mut parser = CParser::new();
        let code = r"
            struct Point {
                int x;
                int y;
            };

            int get_x(struct Point p) {
                return p.x;
            }
        ";

        let funcs = parser.parse_translation_unit(code).unwrap();
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, "get_x");
    }

    #[test]
    fn test_parse_array_param() {
        let mut parser = CParser::new();
        let code = r"
            int sum_array(int arr[], int n) {
                int sum = 0;
                for (int i = 0; i < n; i++) {
                    sum += arr[i];
                }
                return sum;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "sum_array");
        assert!(matches!(func.params[0].ty, CType::Pointer(_)));
    }

    #[test]
    fn test_parse_cast() {
        let mut parser = CParser::new();
        let code = r"
            void* to_void(int* p) {
                return (void*)p;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "to_void");
    }

    #[test]
    fn test_parse_ternary() {
        let mut parser = CParser::new();
        let code = r"
            int max(int a, int b) {
                return a > b ? a : b;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "max");
    }

    #[test]
    fn test_parse_sizeof() {
        let mut parser = CParser::new();
        let code = r"
            int get_int_size(void) {
                return sizeof(int);
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "get_int_size");
    }

    #[test]
    fn test_parse_multiple_functions() {
        let mut parser = CParser::new();
        let code = r"
            int add(int a, int b) { return a + b; }
            int sub(int a, int b) { return a - b; }
            int mul(int a, int b) { return a * b; }
        ";

        let funcs = parser.parse_translation_unit(code).unwrap();
        assert_eq!(funcs.len(), 3);
        assert_eq!(funcs[0].name, "add");
        assert_eq!(funcs[1].name, "sub");
        assert_eq!(funcs[2].name, "mul");
    }

    #[test]
    fn test_parse_variadic_function() {
        let mut parser = CParser::new();
        let code = r"
            int printf_wrapper(const char* fmt, ...) {
                return 0;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "printf_wrapper");
        assert!(func.variadic);
    }

    #[test]
    fn test_parse_static_function() {
        let mut parser = CParser::new();
        let code = r"
            static int helper(int x) {
                return x * 2;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "helper");
        assert_eq!(func.storage, StorageClass::Static);
    }

    #[test]
    fn test_parse_number_literals() {
        let mut parser = CParser::new();
        let code = r"
            int literals(void) {
                int a = 42;
                int b = 0xFF;
                int c = 077;
                return a + b + c;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "literals");
    }

    #[test]
    fn test_parse_switch() {
        let mut parser = CParser::new();
        let code = r"
            int switch_test(int x) {
                switch (x) {
                    case 0: return 1;
                    case 1: return 2;
                    default: return 0;
                }
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "switch_test");
    }

    #[test]
    fn test_acsl_spec_parsing() {
        let comment = r"/*@
            requires x >= 0;
            ensures \result >= 0;
            assigns \nothing;
        */";

        let spec = parse_acsl_spec(comment).unwrap();
        assert_eq!(spec.requires.len(), 1);
        assert_eq!(spec.ensures.len(), 1);
    }

    #[test]
    fn test_parse_do_while() {
        let mut parser = CParser::new();
        let code = r"
            int do_loop(int n) {
                int i = 0;
                do {
                    i++;
                } while (i < n);
                return i;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "do_loop");
    }

    #[test]
    fn test_parse_goto_label() {
        let mut parser = CParser::new();
        let code = r"
            void with_goto(void) {
                goto end;
                return;
            end:
                return;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "with_goto");
    }

    #[test]
    fn test_parse_pointer_return() {
        let mut parser = CParser::new();
        let code = r"
            int* get_ptr(int* p) {
                return p;
            }
        ";

        let func = parser.parse_function(code).unwrap();
        assert_eq!(func.name, "get_ptr");
        assert!(matches!(func.return_type, CType::Pointer(_)));
    }

    #[test]
    fn test_parse_function_with_acsl_block_spec_attached() {
        let mut parser = CParser::new();
        let code = r"
            /*@
                requires x >= 0;
                ensures \result >= 0;
            */
            int clamp(int x) { return x; }
        ";

        let vf = parser.parse_function_with_spec(code).unwrap();
        assert_eq!(vf.name, "clamp");
        assert_eq!(vf.spec.requires.len(), 1);
        assert_eq!(vf.spec.ensures.len(), 1);
        assert!(!vf.generate_vcs().is_empty());
    }

    #[test]
    fn test_parse_translation_unit_with_line_acsl_spec() {
        let mut parser = CParser::new();
        let code = r"
            //@ requires n >= 0;
            //@ ensures \result >= 0;
            int id(int n) { return n; }

            int plain(int x) { return x; }
        ";

        let funcs = parser.parse_translation_unit_with_specs(code).unwrap();
        assert_eq!(funcs.len(), 2);
        assert_eq!(funcs[0].name, "id");
        assert_eq!(funcs[0].spec.requires.len(), 1);
        assert_eq!(funcs[0].spec.ensures.len(), 1);
        assert!(funcs[1].spec.requires.is_empty());
        assert!(funcs[1].spec.ensures.is_empty());
    }
}
