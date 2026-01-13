//! Rust Statement Semantics
//!
//! This module provides statement execution semantics, building
//! on the expression evaluation from expr.rs.
//!
//! ## Statement Types
//!
//! - **Let bindings**: Variable declarations
//! - **Expression statements**: Expressions evaluated for side effects
//! - **Item declarations**: Functions, structs, etc.

// Re-export from expr.rs where Stmt is defined
pub use crate::expr::{Item, Stmt};

use crate::expr::{Expr, Pattern};
use crate::memory::{Memory, Stack};
use crate::ownership::OwnershipState;
use crate::types::RustType;
use crate::values::Value;
use std::collections::HashMap;

/// Execution context for statement evaluation
#[derive(Debug)]
pub struct ExecContext {
    /// Memory model
    pub memory: Memory,
    /// Call stack
    pub stack: Stack,
    /// Ownership state
    pub ownership: OwnershipState,
    /// Named functions
    pub functions: HashMap<String, FunctionDef>,
    /// Named types (structs, enums)
    pub types: HashMap<String, TypeDef>,
}

/// Function definition
#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<(String, RustType)>,
    pub ret_ty: RustType,
    pub body: Expr,
}

/// Type definition (struct or enum)
#[derive(Debug, Clone)]
pub enum TypeDef {
    Struct {
        name: String,
        fields: Vec<(String, RustType)>,
    },
    Enum {
        name: String,
        variants: Vec<EnumVariantDef>,
    },
}

/// Enum variant definition
#[derive(Debug, Clone)]
pub struct EnumVariantDef {
    pub name: String,
    pub payload: EnumVariantType,
}

/// Enum variant payload type
#[derive(Debug, Clone)]
pub enum EnumVariantType {
    Unit,
    Tuple(Vec<RustType>),
    Struct(Vec<(String, RustType)>),
}

impl ExecContext {
    /// Create a new execution context
    pub fn new() -> Self {
        Self {
            memory: Memory::new(),
            stack: Stack::new(),
            ownership: OwnershipState::new(),
            functions: HashMap::new(),
            types: HashMap::new(),
        }
    }

    /// Register a function
    pub fn register_function(&mut self, def: FunctionDef) {
        self.functions.insert(def.name.clone(), def);
    }

    /// Register a type
    pub fn register_type(&mut self, def: TypeDef) {
        let name = match &def {
            TypeDef::Struct { name, .. } | TypeDef::Enum { name, .. } => name.clone(),
        };
        self.types.insert(name, def);
    }

    /// Look up a function
    pub fn get_function(&self, name: &str) -> Option<&FunctionDef> {
        self.functions.get(name)
    }

    /// Look up a type
    pub fn get_type(&self, name: &str) -> Option<&TypeDef> {
        self.types.get(name)
    }
}

impl Default for ExecContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Statement execution result
#[derive(Debug, Clone)]
pub enum StmtResult {
    /// Statement completed normally
    Ok,
    /// Return from function
    Return(Value),
    /// Break from loop
    Break(Option<Value>),
    /// Continue loop
    Continue,
    /// Error during execution
    Error(String),
}

impl StmtResult {
    pub fn is_ok(&self) -> bool {
        matches!(self, StmtResult::Ok)
    }

    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            StmtResult::Return(_) | StmtResult::Break(_) | StmtResult::Continue
        )
    }
}

/// Pattern matching result
#[derive(Debug, Clone)]
pub struct PatternBindings {
    /// Bindings created by the pattern
    pub bindings: Vec<(String, Value, bool)>, // (name, value, mutable)
}

impl PatternBindings {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    pub fn add(&mut self, name: String, value: Value, mutable: bool) {
        self.bindings.push((name, value, mutable));
    }

    pub fn merge(&mut self, other: PatternBindings) {
        self.bindings.extend(other.bindings);
    }
}

impl Default for PatternBindings {
    fn default() -> Self {
        Self::new()
    }
}

/// Match a pattern against a value
pub fn match_pattern(pattern: &Pattern, value: &Value) -> Option<PatternBindings> {
    let mut bindings = PatternBindings::new();

    match pattern {
        Pattern::Wildcard => Some(bindings),

        Pattern::Binding {
            name,
            mutable,
            subpattern,
        } => {
            if let Some(sub) = subpattern {
                let sub_bindings = match_pattern(sub, value)?;
                bindings.merge(sub_bindings);
            }
            bindings.add(name.clone(), value.clone(), *mutable);
            Some(bindings)
        }

        Pattern::Literal(lit) => {
            if value == lit {
                Some(bindings)
            } else {
                None
            }
        }

        Pattern::Tuple(patterns) => {
            if let Value::Tuple(values) = value {
                if patterns.len() != values.len() {
                    return None;
                }
                for (p, v) in patterns.iter().zip(values.iter()) {
                    let sub_bindings = match_pattern(p, v)?;
                    bindings.merge(sub_bindings);
                }
                Some(bindings)
            } else {
                None
            }
        }

        Pattern::Struct {
            name,
            fields,
            rest: _,
        } => {
            if let Value::Struct {
                name: struct_name,
                fields: struct_fields,
            } = value
            {
                if name != struct_name {
                    return None;
                }
                for (field_name, field_pattern) in fields {
                    let field_value = struct_fields.get(field_name)?;
                    let sub_bindings = match_pattern(field_pattern, field_value)?;
                    bindings.merge(sub_bindings);
                }
                Some(bindings)
            } else {
                None
            }
        }

        Pattern::EnumVariant {
            enum_name,
            variant,
            payload,
        } => {
            if let Value::Enum {
                name,
                variant: var,
                payload: val_payload,
            } = value
            {
                if enum_name != name || variant != var {
                    return None;
                }
                match (payload, val_payload.as_ref()) {
                    (crate::expr::EnumPatternPayload::Unit, crate::values::EnumPayload::Unit) => {
                        Some(bindings)
                    }
                    (
                        crate::expr::EnumPatternPayload::Tuple(patterns),
                        crate::values::EnumPayload::Tuple(values),
                    ) => {
                        if patterns.len() != values.len() {
                            return None;
                        }
                        for (p, v) in patterns.iter().zip(values.iter()) {
                            let sub_bindings = match_pattern(p, v)?;
                            bindings.merge(sub_bindings);
                        }
                        Some(bindings)
                    }
                    (
                        crate::expr::EnumPatternPayload::Struct(patterns),
                        crate::values::EnumPayload::Struct(fields),
                    ) => {
                        for (field_name, field_pattern) in patterns {
                            let field_value = fields.get(field_name)?;
                            let sub_bindings = match_pattern(field_pattern, field_value)?;
                            bindings.merge(sub_bindings);
                        }
                        Some(bindings)
                    }
                    _ => None,
                }
            } else {
                None
            }
        }

        Pattern::Or(patterns) => {
            for p in patterns {
                if let Some(b) = match_pattern(p, value) {
                    return Some(b);
                }
            }
            None
        }

        Pattern::Range {
            start,
            end,
            inclusive,
        } => {
            // Check if value is in range
            match (start, end, value) {
                (
                    Value::Uint { value: s, ty: ty_s },
                    Value::Uint { value: e, ty: ty_e },
                    Value::Uint { value: v, ty: ty_v },
                ) if ty_s == ty_e && ty_s == ty_v => {
                    let in_range = if *inclusive {
                        v >= s && v <= e
                    } else {
                        v >= s && v < e
                    };
                    if in_range {
                        Some(bindings)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }

        Pattern::Ref {
            mutability: _,
            pattern,
        } => {
            // For references, we'd need to dereference
            // This is a simplified version
            match_pattern(pattern, value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::UintType;
    use crate::values::EnumPayload;

    #[test]
    fn test_exec_context() {
        let mut ctx = ExecContext::new();

        let func = FunctionDef {
            name: "add".to_string(),
            params: vec![
                ("a".to_string(), RustType::Uint(UintType::U32)),
                ("b".to_string(), RustType::Uint(UintType::U32)),
            ],
            ret_ty: RustType::Uint(UintType::U32),
            body: Expr::Literal(Value::u32(0)), // Placeholder
        };

        ctx.register_function(func);
        assert!(ctx.get_function("add").is_some());
    }

    #[test]
    fn test_match_wildcard() {
        let pattern = Pattern::Wildcard;
        let value = Value::u32(42);

        let result = match_pattern(&pattern, &value);
        assert!(result.is_some());
        assert!(result.unwrap().bindings.is_empty());
    }

    #[test]
    fn test_match_binding() {
        let pattern = Pattern::Binding {
            name: "x".to_string(),
            mutable: false,
            subpattern: None,
        };
        let value = Value::u32(42);

        let result = match_pattern(&pattern, &value);
        assert!(result.is_some());

        let bindings = result.unwrap();
        assert_eq!(bindings.bindings.len(), 1);
        assert_eq!(bindings.bindings[0].0, "x");
        assert_eq!(bindings.bindings[0].1, Value::u32(42));
    }

    #[test]
    fn test_match_literal_success() {
        let pattern = Pattern::Literal(Value::u32(42));
        let value = Value::u32(42);

        assert!(match_pattern(&pattern, &value).is_some());
    }

    #[test]
    fn test_match_literal_failure() {
        let pattern = Pattern::Literal(Value::u32(42));
        let value = Value::u32(43);

        assert!(match_pattern(&pattern, &value).is_none());
    }

    #[test]
    fn test_match_tuple() {
        let pattern = Pattern::Tuple(vec![
            Pattern::Binding {
                name: "a".to_string(),
                mutable: false,
                subpattern: None,
            },
            Pattern::Binding {
                name: "b".to_string(),
                mutable: false,
                subpattern: None,
            },
        ]);
        let value = Value::Tuple(vec![Value::u32(1), Value::u32(2)]);

        let result = match_pattern(&pattern, &value);
        assert!(result.is_some());

        let bindings = result.unwrap();
        assert_eq!(bindings.bindings.len(), 2);
    }

    #[test]
    fn test_match_enum_unit() {
        let pattern = Pattern::EnumVariant {
            enum_name: "Option".to_string(),
            variant: "None".to_string(),
            payload: crate::expr::EnumPatternPayload::Unit,
        };
        let value = Value::Enum {
            name: "Option".to_string(),
            variant: "None".to_string(),
            payload: Box::new(EnumPayload::Unit),
        };

        assert!(match_pattern(&pattern, &value).is_some());
    }

    #[test]
    fn test_match_enum_tuple() {
        let pattern = Pattern::EnumVariant {
            enum_name: "Option".to_string(),
            variant: "Some".to_string(),
            payload: crate::expr::EnumPatternPayload::Tuple(vec![Pattern::Binding {
                name: "x".to_string(),
                mutable: false,
                subpattern: None,
            }]),
        };
        let value = Value::Enum {
            name: "Option".to_string(),
            variant: "Some".to_string(),
            payload: Box::new(EnumPayload::Tuple(vec![Value::u32(42)])),
        };

        let result = match_pattern(&pattern, &value);
        assert!(result.is_some());
        assert_eq!(result.unwrap().bindings.len(), 1);
    }

    #[test]
    fn test_match_or_pattern() {
        let pattern = Pattern::Or(vec![
            Pattern::Literal(Value::u32(1)),
            Pattern::Literal(Value::u32(2)),
            Pattern::Literal(Value::u32(3)),
        ]);

        assert!(match_pattern(&pattern, &Value::u32(1)).is_some());
        assert!(match_pattern(&pattern, &Value::u32(2)).is_some());
        assert!(match_pattern(&pattern, &Value::u32(3)).is_some());
        assert!(match_pattern(&pattern, &Value::u32(4)).is_none());
    }
}
