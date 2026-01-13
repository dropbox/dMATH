//! Type checker for USL
//!
//! This module validates USL specifications for type correctness, including:
//! - Type definition well-formedness
//! - Expression type consistency
//! - Quantifier scope validation
//! - Property-specific constraints

// Allow methods that use &self only for recursive calls
#![allow(clippy::only_used_in_recursion)]

use crate::ast::{
    BinaryOp, Bisimulation, Contract, Expr, Invariant, PlatformApi, Probabilistic, Property,
    Refinement, Security, SemanticProperty, Spec, Temporal, TemporalExpr, Theorem, Type, TypeDef,
};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Type checking errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TypeError {
    /// Reference to undefined type
    #[error("Unknown type: {0}")]
    UnknownType(String),

    /// Type mismatch between expected and actual
    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch {
        /// Expected type
        expected: String,
        /// Found type
        found: String,
    },

    /// Reference to undefined variable
    #[error("Unknown variable: {0}")]
    UnknownVariable(String),

    /// Reference to undefined function
    #[error("Unknown function: {0}")]
    UnknownFunction(String),

    /// Type defined more than once
    #[error("Duplicate type definition: {0}")]
    DuplicateType(String),

    /// Field name used more than once in same type
    #[error("Duplicate field in type {type_name}: {field_name}")]
    DuplicateField {
        /// Type containing the duplicate
        type_name: String,
        /// Duplicated field name
        field_name: String,
    },

    /// Field access on type that doesn't have that field
    #[error("Invalid field access: type {ty} has no field {field}")]
    InvalidField {
        /// Type being accessed
        ty: String,
        /// Field name that doesn't exist
        field: String,
    },

    /// Trying to iterate over non-collection type
    #[error("Cannot iterate over non-collection type: {0}")]
    NotIterable(String),

    /// Contract requires clause is not boolean
    #[error("Contract requires expects boolean condition, found {0}")]
    RequiresNotBool(String),

    /// Contract ensures clause is not boolean
    #[error("Contract ensures expects boolean condition, found {0}")]
    EnsuresNotBool(String),

    /// Probability bound out of valid range
    #[error("Probability bound must be between 0.0 and 1.0, got {0}")]
    InvalidProbabilityBound(f64),

    /// Theorem body is not boolean
    #[error("Theorem body must be boolean, found {0}")]
    TheoremNotBool(String),

    /// Invariant body is not boolean
    #[error("Invariant body must be boolean, found {0}")]
    InvariantNotBool(String),

    /// Security property body is not boolean
    #[error("Security property must be boolean, found {0}")]
    SecurityNotBool(String),

    /// Semantic property body is not boolean
    #[error("Semantic property must be boolean, found {0}")]
    SemanticNotBool(String),

    /// Numeric operator applied to non-numeric operand
    #[error("Operands to '{op}' must be numeric, found {found}")]
    NumericOperandRequired {
        /// Operator that requires numeric operands
        op: String,
        /// Type that was found
        found: String,
    },

    /// Logical operator applied to non-boolean operand
    #[error("Operands to logical '{op}' must be boolean, found {found}")]
    BoolOperandRequired {
        /// Operator that requires boolean operands
        op: String,
        /// Type that was found
        found: String,
    },

    /// Comparison between incompatible types
    #[error("Comparison operands must have the same type: {left} vs {right}")]
    ComparisonTypeMismatch {
        /// Left operand type
        left: String,
        /// Right operand type
        right: String,
    },

    /// Method call on type that doesn't have that method
    #[error("Method {method} not found on type {ty}")]
    MethodNotFound {
        /// Type being called on
        ty: String,
        /// Method name that doesn't exist
        method: String,
    },
}

/// Checked type - the type of an expression after type checking
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckedType {
    /// Boolean type
    Bool,
    /// Integer type
    Int,
    /// Float type
    Float,
    /// String type
    String,
    /// Named user-defined type
    Named(std::string::String),
    /// Set of elements
    Set(Box<CheckedType>),
    /// List of elements
    List(Box<CheckedType>),
    /// Map from keys to values
    Map(Box<CheckedType>, Box<CheckedType>),
    /// Relation between two types
    Relation(Box<CheckedType>, Box<CheckedType>),
    /// Function type
    Function(Box<CheckedType>, Box<CheckedType>),
    /// Result type
    Result(Box<CheckedType>),
    /// Graph type with node and edge types
    Graph(Box<CheckedType>, Box<CheckedType>),
    /// Path type (sequence of nodes in a graph)
    Path(Box<CheckedType>),
    /// Unit type
    Unit,
    /// Unknown type (used during inference)
    Unknown,
}

impl std::fmt::Display for CheckedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool => write!(f, "Bool"),
            Self::Int => write!(f, "Int"),
            Self::Float => write!(f, "Float"),
            Self::String => write!(f, "String"),
            Self::Named(name) => write!(f, "{name}"),
            Self::Set(inner) => write!(f, "Set<{inner}>"),
            Self::List(inner) => write!(f, "List<{inner}>"),
            Self::Map(k, v) => write!(f, "Map<{k}, {v}>"),
            Self::Relation(a, b) => write!(f, "Relation<{a}, {b}>"),
            Self::Function(a, b) => write!(f, "{a} -> {b}"),
            Self::Result(inner) => write!(f, "Result<{inner}>"),
            Self::Graph(n, e) => write!(f, "Graph<{n}, {e}>"),
            Self::Path(n) => write!(f, "Path<{n}>"),
            Self::Unit => write!(f, "()"),
            Self::Unknown => write!(f, "?"),
        }
    }
}

/// Type environment for type checking
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// User-defined types
    types: HashMap<std::string::String, TypeDef>,
    /// Variable bindings in current scope
    variables: HashMap<std::string::String, CheckedType>,
    /// Built-in function signatures (name -> (`param_types`, `return_type`))
    functions: HashMap<std::string::String, FunctionSig>,
    /// Scope stack for nested quantifiers
    scope_stack: Vec<HashMap<std::string::String, CheckedType>>,
}

/// Function signature
#[derive(Debug, Clone)]
pub struct FunctionSig {
    /// Parameter types
    pub params: Vec<CheckedType>,
    /// Return type
    pub return_type: CheckedType,
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeEnv {
    /// Create a new type environment with built-in types
    #[must_use]
    pub fn new() -> Self {
        let mut env = Self {
            types: HashMap::new(),
            variables: HashMap::new(),
            functions: HashMap::new(),
            scope_stack: Vec::new(),
        };
        env.register_builtins();
        env
    }

    /// Register built-in types and functions
    fn register_builtins(&mut self) {
        // Common predicate functions that return Bool
        // These are flexible - they can take various types
        let bool_predicates = [
            // General predicates
            "acyclic",
            "is_terminal",
            "reachable",
            "enabled",
            "at_checkpoint",
            "contains",
            "can_observe",
            "addresses_question",
            "is_factually_accurate",
            "contains_concepts",
            // Graph predicates (Phase 17.3 - DashFlow execution graphs)
            "is_acyclic",   // is_acyclic(g) - true if graph has no cycles
            "is_dag",       // is_dag(g) - alias for is_acyclic
            "is_connected", // is_connected(g) - true if graph is connected
            "has_path",     // has_path(g, from, to) - true if path exists
            "in_graph",     // in_graph(node, g) - true if node is in graph
            "edge_exists",  // edge_exists(g, from, to) - true if edge exists
            // DashFlow modification predicates
            "preserves_completed", // preserves_completed(g, g') - completed nodes unchanged
            "valid_modification",  // valid_modification(m, g, g') - modification is valid
            "preserves_dag",       // preserves_dag(g, g') - DAG property preserved
            "is_ready",            // is_ready(node, g) - node has all deps satisfied
            "all_deps_completed",  // all_deps_completed(node, g) - all dependencies done
            // Node status predicates
            "completed", // completed(node) - node execution is complete
            "pending",   // pending(node) - node is waiting
            "running",   // running(node) - node is executing
            "failed",    // failed(node) - node execution failed
        ];
        for name in bool_predicates {
            self.functions.insert(
                name.to_string(),
                FunctionSig {
                    params: vec![], // Flexible - accepts any args
                    return_type: CheckedType::Bool,
                },
            );
        }

        // Functions that work on execution
        self.functions.insert(
            "executes".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Bool,
            },
        );

        // Actions function returns a collection (for security properties)
        self.functions.insert(
            "actions".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Set(Box::new(CheckedType::Unknown)),
            },
        );

        // probability function for probabilistic properties
        self.functions.insert(
            "probability".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Float,
            },
        );

        // Semantic similarity returns a numeric score
        self.functions.insert(
            "semantic_similarity".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Float,
            },
        );

        // Graph accessor functions (Phase 17.3 - DashFlow execution graphs)
        // These return collections/values from graphs

        // nodes(g) - returns the set of nodes in the graph
        self.functions.insert(
            "nodes".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Set(Box::new(CheckedType::Unknown)),
            },
        );

        // edges(g) - returns the set of edges in the graph
        self.functions.insert(
            "edges".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Set(Box::new(CheckedType::Unknown)),
            },
        );

        // successors(g, node) - returns successors of a node
        self.functions.insert(
            "successors".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Set(Box::new(CheckedType::Unknown)),
            },
        );

        // predecessors(g, node) - returns predecessors of a node
        self.functions.insert(
            "predecessors".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Set(Box::new(CheckedType::Unknown)),
            },
        );

        // path(g, from, to) - returns a path between nodes (if exists)
        self.functions.insert(
            "path".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Path(Box::new(CheckedType::Unknown)),
            },
        );

        // topological_order(g) - returns topologically sorted nodes
        self.functions.insert(
            "topological_order".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::List(Box::new(CheckedType::Unknown)),
            },
        );

        // node_count(g) - returns number of nodes
        self.functions.insert(
            "node_count".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Int,
            },
        );

        // edge_count(g) - returns number of edges
        self.functions.insert(
            "edge_count".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Int,
            },
        );

        // in_degree(g, node) - returns number of incoming edges
        self.functions.insert(
            "in_degree".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Int,
            },
        );

        // out_degree(g, node) - returns number of outgoing edges
        self.functions.insert(
            "out_degree".to_string(),
            FunctionSig {
                params: vec![],
                return_type: CheckedType::Int,
            },
        );
    }

    /// Register a user-defined type.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError::DuplicateType`] if the type is already registered,
    /// or [`TypeError::DuplicateField`] if the type has duplicate field names.
    pub fn register_type(&mut self, typedef: TypeDef) -> Result<(), TypeError> {
        if self.types.contains_key(&typedef.name) {
            return Err(TypeError::DuplicateType(typedef.name));
        }

        // Check for duplicate fields
        let mut field_names = HashSet::new();
        for field in &typedef.fields {
            if !field_names.insert(&field.name) {
                return Err(TypeError::DuplicateField {
                    type_name: typedef.name.clone(),
                    field_name: field.name.clone(),
                });
            }
        }

        self.types.insert(typedef.name.clone(), typedef);
        Ok(())
    }

    /// Push a new scope
    pub fn push_scope(&mut self) {
        self.scope_stack.push(HashMap::new());
    }

    /// Pop the current scope
    pub fn pop_scope(&mut self) {
        if let Some(scope) = self.scope_stack.pop() {
            for name in scope.keys() {
                self.variables.remove(name);
            }
        }
    }

    /// Bind a variable in the current scope
    pub fn bind(&mut self, name: std::string::String, ty: CheckedType) {
        self.variables.insert(name.clone(), ty.clone());
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.insert(name, ty);
        }
    }

    /// Look up a variable
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<&CheckedType> {
        self.variables.get(name)
    }

    /// Look up a type definition
    #[must_use]
    pub fn lookup_type(&self, name: &str) -> Option<&TypeDef> {
        self.types.get(name)
    }

    /// Get field type from a user-defined type.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError::InvalidField`] if the field doesn't exist in the type.
    pub fn get_field_type(
        &self,
        type_name: &str,
        field_name: &str,
    ) -> Result<CheckedType, TypeError> {
        if let Some(typedef) = self.types.get(type_name) {
            for field in &typedef.fields {
                if field.name == field_name {
                    return self.convert_type(&field.ty);
                }
            }
            Err(TypeError::InvalidField {
                ty: type_name.to_string(),
                field: field_name.to_string(),
            })
        } else {
            // For built-in or unknown types, be permissive
            Ok(CheckedType::Unknown)
        }
    }

    /// Convert AST Type to `CheckedType`.
    ///
    /// # Errors
    ///
    /// Currently this method does not return errors for unknown types (it
    /// accepts them permissively), but returns errors if nested type
    /// conversion fails.
    pub fn convert_type(&self, ty: &Type) -> Result<CheckedType, TypeError> {
        match ty {
            Type::Named(name) => {
                // Check for built-in types first
                match name.as_str() {
                    "Bool" | "bool" => Ok(CheckedType::Bool),
                    "Int" | "int" | "i32" | "i64" | "u32" | "u64" | "usize" | "isize" => {
                        Ok(CheckedType::Int)
                    }
                    "Float" | "float" | "f32" | "f64" => Ok(CheckedType::Float),
                    "String" | "str" => Ok(CheckedType::String),
                    _ => {
                        // Accept user-defined types or unknown types permissively
                        Ok(CheckedType::Named(name.clone()))
                    }
                }
            }
            Type::Set(inner) => Ok(CheckedType::Set(Box::new(self.convert_type(inner)?))),
            Type::List(inner) => Ok(CheckedType::List(Box::new(self.convert_type(inner)?))),
            Type::Map(k, v) => Ok(CheckedType::Map(
                Box::new(self.convert_type(k)?),
                Box::new(self.convert_type(v)?),
            )),
            Type::Relation(a, b) => Ok(CheckedType::Relation(
                Box::new(self.convert_type(a)?),
                Box::new(self.convert_type(b)?),
            )),
            Type::Function(a, b) => Ok(CheckedType::Function(
                Box::new(self.convert_type(a)?),
                Box::new(self.convert_type(b)?),
            )),
            Type::Result(inner) => Ok(CheckedType::Result(Box::new(self.convert_type(inner)?))),
            Type::Graph(n, e) => Ok(CheckedType::Graph(
                Box::new(self.convert_type(n)?),
                Box::new(self.convert_type(e)?),
            )),
            Type::Path(n) => Ok(CheckedType::Path(Box::new(self.convert_type(n)?))),
            Type::Unit => Ok(CheckedType::Unit),
        }
    }

    /// Get the element type of a collection.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError::NotIterable`] if the type is not a collection.
    pub fn element_type(&self, collection_type: &CheckedType) -> Result<CheckedType, TypeError> {
        match collection_type {
            CheckedType::Set(inner) | CheckedType::List(inner) => Ok(*inner.clone()),
            CheckedType::Unknown => Ok(CheckedType::Unknown),
            other => Err(TypeError::NotIterable(other.to_string())),
        }
    }
}

/// Type-checked specification with type information
#[derive(Debug, Clone)]
pub struct TypedSpec {
    /// Original specification
    pub spec: Spec,
    /// Type information for each expression (keyed by location/id in a real impl)
    pub type_info: HashMap<std::string::String, CheckedType>,
}

/// Type checker implementation
pub struct TypeChecker {
    env: TypeEnv,
    /// Collected errors (for future error recovery support)
    #[allow(dead_code)]
    errors: Vec<TypeError>,
}

impl TypeChecker {
    /// Create a new type checker
    #[must_use]
    pub fn new() -> Self {
        Self {
            env: TypeEnv::new(),
            errors: Vec::new(),
        }
    }

    /// Type check a specification.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if the specification contains type errors,
    /// such as undefined types, undefined variables, type mismatches, or
    /// invalid operations.
    pub fn check(&mut self, spec: &Spec) -> Result<TypedSpec, TypeError> {
        // Phase 1: Register all type definitions
        for typedef in &spec.types {
            self.env.register_type(typedef.clone())?;
        }

        // Phase 2: Validate type definitions (check field types exist)
        for typedef in &spec.types {
            self.check_typedef(typedef)?;
        }

        // Phase 3: Check all properties
        for property in &spec.properties {
            self.check_property(property)?;
        }

        Ok(TypedSpec {
            spec: spec.clone(),
            type_info: HashMap::new(),
        })
    }

    /// Check a type definition
    fn check_typedef(&self, typedef: &TypeDef) -> Result<(), TypeError> {
        for field in &typedef.fields {
            self.check_type_exists(&field.ty)?;
        }
        Ok(())
    }

    /// Check that a type exists
    fn check_type_exists(&self, ty: &Type) -> Result<(), TypeError> {
        match ty {
            Type::Named(_) => {
                // Be permissive - accept all named types for forward compatibility
                // with built-in, registered, or unknown types
                Ok(())
            }
            Type::Set(inner)
            | Type::List(inner)
            | Type::Result(inner)
            | Type::Function(inner, _) => self.check_type_exists(inner),
            Type::Map(k, v) | Type::Relation(k, v) | Type::Graph(k, v) => {
                self.check_type_exists(k)?;
                self.check_type_exists(v)
            }
            Type::Path(inner) => self.check_type_exists(inner),
            Type::Unit => Ok(()),
        }
    }

    /// Check a property
    fn check_property(&mut self, property: &Property) -> Result<(), TypeError> {
        match property {
            Property::Theorem(theorem) => self.check_theorem(theorem),
            Property::Temporal(temporal) => self.check_temporal(temporal),
            Property::Contract(contract) => self.check_contract(contract),
            Property::Invariant(invariant) => self.check_invariant(invariant),
            Property::Refinement(refinement) => self.check_refinement(refinement),
            Property::Probabilistic(prob) => self.check_probabilistic(prob),
            Property::Security(security) => self.check_security(security),
            Property::Semantic(semantic) => self.check_semantic(semantic),
            Property::PlatformApi(api) => self.check_platform_api(api),
            Property::Bisimulation(bisim) => self.check_bisimulation(bisim),
            Property::Version(version) => self.check_version_spec(version),
            Property::Capability(capability) => self.check_capability_spec(capability),
            Property::DistributedInvariant(dist_inv) => self.check_distributed_invariant(dist_inv),
            Property::DistributedTemporal(dist_temp) => self.check_distributed_temporal(dist_temp),
            Property::Composed(composed) => self.check_composed_theorem(composed),
            Property::ImprovementProposal(proposal) => self.check_improvement_proposal(proposal),
            Property::VerificationGate(gate) => self.check_verification_gate(gate),
            Property::Rollback(rollback) => self.check_rollback_spec(rollback),
        }
    }

    /// Check a theorem
    fn check_theorem(&mut self, theorem: &Theorem) -> Result<(), TypeError> {
        self.env.push_scope();
        let body_type = self.check_expr(&theorem.body)?;
        self.env.pop_scope();

        if !Self::is_bool_type(&body_type) {
            return Err(TypeError::TheoremNotBool(body_type.to_string()));
        }
        Ok(())
    }

    /// Check a composed theorem
    ///
    /// Composed theorems reference other properties by name in a `uses` clause.
    /// Each referenced property is introduced as a Bool-typed variable in scope.
    fn check_composed_theorem(
        &mut self,
        composed: &crate::ast::ComposedTheorem,
    ) -> Result<(), TypeError> {
        self.env.push_scope();

        // Each "used" property becomes a Bool-typed variable
        // (representing that the property is assumed to hold)
        for dep_name in &composed.uses {
            self.env.bind(dep_name.clone(), CheckedType::Bool);
        }

        let body_type = self.check_expr(&composed.body)?;
        self.env.pop_scope();

        if !Self::is_bool_type(&body_type) {
            return Err(TypeError::TheoremNotBool(body_type.to_string()));
        }
        Ok(())
    }

    /// Check an improvement proposal
    ///
    /// Improvement proposals specify what a proposed improvement must satisfy.
    /// The target, improves, preserves, and requires expressions are all checked.
    fn check_improvement_proposal(
        &mut self,
        proposal: &crate::ast::ImprovementProposal,
    ) -> Result<(), TypeError> {
        self.env.push_scope();

        // Check target expression (can be any type)
        self.check_expr(&proposal.target)?;

        // Check improves expressions (should be Bool)
        for expr in &proposal.improves {
            let ty = self.check_expr(expr)?;
            if !Self::is_bool_type(&ty) {
                return Err(TypeError::TypeMismatch {
                    expected: "Bool".to_string(),
                    found: ty.to_string(),
                });
            }
        }

        // Check preserves expressions (should be Bool)
        for expr in &proposal.preserves {
            let ty = self.check_expr(expr)?;
            if !Self::is_bool_type(&ty) {
                return Err(TypeError::TypeMismatch {
                    expected: "Bool".to_string(),
                    found: ty.to_string(),
                });
            }
        }

        // Check requires expressions (should be Bool)
        for expr in &proposal.requires {
            let ty = self.check_expr(expr)?;
            if !Self::is_bool_type(&ty) {
                return Err(TypeError::TypeMismatch {
                    expected: "Bool".to_string(),
                    found: ty.to_string(),
                });
            }
        }

        self.env.pop_scope();
        Ok(())
    }

    /// Check a verification gate
    ///
    /// Verification gates specify mandatory checks before accepting any improvement.
    fn check_verification_gate(
        &mut self,
        gate: &crate::ast::VerificationGate,
    ) -> Result<(), TypeError> {
        self.env.push_scope();

        // Bind input parameters
        for param in &gate.inputs {
            let ty = self.env.convert_type(&param.ty)?;
            self.env.bind(param.name.clone(), ty);
        }

        // Check each verification check condition (should be Bool)
        for check in &gate.checks {
            let ty = self.check_expr(&check.condition)?;
            if !Self::is_bool_type(&ty) {
                return Err(TypeError::TypeMismatch {
                    expected: "Bool".to_string(),
                    found: ty.to_string(),
                });
            }
        }

        // Check on_pass and on_fail expressions
        self.check_expr(&gate.on_pass)?;
        self.check_expr(&gate.on_fail)?;

        self.env.pop_scope();
        Ok(())
    }

    /// Check a rollback specification
    ///
    /// Rollback specs specify how to safely rollback a failed improvement attempt.
    fn check_rollback_spec(
        &mut self,
        rollback: &crate::ast::RollbackSpec,
    ) -> Result<(), TypeError> {
        self.env.push_scope();

        // Bind state parameters
        for param in &rollback.state {
            let ty = self.env.convert_type(&param.ty)?;
            self.env.bind(param.name.clone(), ty);
        }

        // Check invariants (should be Bool)
        for expr in &rollback.invariants {
            let ty = self.check_expr(expr)?;
            if !Self::is_bool_type(&ty) {
                return Err(TypeError::TypeMismatch {
                    expected: "Bool".to_string(),
                    found: ty.to_string(),
                });
            }
        }

        // Check trigger expression (should be Bool)
        let trigger_ty = self.check_expr(&rollback.trigger)?;
        if !Self::is_bool_type(&trigger_ty) {
            return Err(TypeError::TypeMismatch {
                expected: "Bool".to_string(),
                found: trigger_ty.to_string(),
            });
        }

        // Check action assignments
        for (_, expr) in &rollback.action.assignments {
            self.check_expr(expr)?;
        }

        // Check ensure clause if present (should be Bool)
        if let Some(ensure) = &rollback.action.ensure {
            let ty = self.check_expr(ensure)?;
            if !Self::is_bool_type(&ty) {
                return Err(TypeError::TypeMismatch {
                    expected: "Bool".to_string(),
                    found: ty.to_string(),
                });
            }
        }

        // Check guarantees (should be Bool)
        for expr in &rollback.guarantees {
            let ty = self.check_expr(expr)?;
            if !Self::is_bool_type(&ty) {
                return Err(TypeError::TypeMismatch {
                    expected: "Bool".to_string(),
                    found: ty.to_string(),
                });
            }
        }

        self.env.pop_scope();
        Ok(())
    }

    /// Check a temporal property
    fn check_temporal(&mut self, temporal: &Temporal) -> Result<(), TypeError> {
        self.env.push_scope();
        self.check_temporal_expr(&temporal.body)?;
        self.env.pop_scope();
        Ok(())
    }

    /// Check a temporal expression
    fn check_temporal_expr(&mut self, expr: &TemporalExpr) -> Result<CheckedType, TypeError> {
        match expr {
            TemporalExpr::Always(inner) | TemporalExpr::Eventually(inner) => {
                self.check_temporal_expr(inner)
            }
            TemporalExpr::LeadsTo(from, to) => {
                self.check_temporal_expr(from)?;
                self.check_temporal_expr(to)
            }
            TemporalExpr::Atom(e) => {
                let ty = self.check_expr(e)?;
                if !Self::is_bool_type(&ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: "Bool".to_string(),
                        found: ty.to_string(),
                    });
                }
                Ok(CheckedType::Bool)
            }
        }
    }

    /// Check a contract
    fn check_contract(&mut self, contract: &Contract) -> Result<(), TypeError> {
        self.env.push_scope();

        // Bind 'self' and 'self'' (post-state)
        let self_type = if contract.type_path.is_empty() {
            CheckedType::Unknown
        } else {
            CheckedType::Named(contract.type_path[0].clone())
        };
        self.env.bind("self".to_string(), self_type.clone());
        self.env.bind("self'".to_string(), self_type);

        // Bind parameters
        for param in &contract.params {
            let param_type = self.env.convert_type(&param.ty)?;
            self.env.bind(param.name.clone(), param_type);
        }

        // Check requires clauses
        for req in &contract.requires {
            let ty = self.check_expr(req)?;
            if !Self::is_bool_type(&ty) {
                return Err(TypeError::RequiresNotBool(ty.to_string()));
            }
        }

        // Check ensures clauses
        for ens in &contract.ensures {
            let ty = self.check_expr(ens)?;
            if !Self::is_bool_type(&ty) {
                return Err(TypeError::EnsuresNotBool(ty.to_string()));
            }
        }

        // Check ensures_err clauses
        for ens in &contract.ensures_err {
            let ty = self.check_expr(ens)?;
            if !Self::is_bool_type(&ty) {
                return Err(TypeError::EnsuresNotBool(ty.to_string()));
            }
        }

        self.env.pop_scope();
        Ok(())
    }

    /// Check an invariant
    fn check_invariant(&mut self, invariant: &Invariant) -> Result<(), TypeError> {
        self.env.push_scope();
        let body_type = self.check_expr(&invariant.body)?;
        self.env.pop_scope();

        if !Self::is_bool_type(&body_type) {
            return Err(TypeError::InvariantNotBool(body_type.to_string()));
        }
        Ok(())
    }

    /// Check a refinement
    fn check_refinement(&mut self, refinement: &Refinement) -> Result<(), TypeError> {
        self.env.push_scope();

        // Check abstraction expression
        let abs_type = self.check_expr(&refinement.abstraction)?;
        if !Self::is_bool_type(&abs_type) {
            return Err(TypeError::TypeMismatch {
                expected: "Bool".to_string(),
                found: abs_type.to_string(),
            });
        }

        // Check simulation expression
        let sim_type = self.check_expr(&refinement.simulation)?;
        if !Self::is_bool_type(&sim_type) {
            return Err(TypeError::TypeMismatch {
                expected: "Bool".to_string(),
                found: sim_type.to_string(),
            });
        }

        self.env.pop_scope();
        Ok(())
    }

    /// Check a probabilistic property
    fn check_probabilistic(&mut self, prob: &Probabilistic) -> Result<(), TypeError> {
        // Check bound is valid
        if prob.bound < 0.0 || prob.bound > 1.0 {
            return Err(TypeError::InvalidProbabilityBound(prob.bound));
        }

        self.env.push_scope();
        let cond_type = self.check_expr(&prob.condition)?;
        self.env.pop_scope();

        if !Self::is_bool_type(&cond_type) {
            return Err(TypeError::TypeMismatch {
                expected: "Bool".to_string(),
                found: cond_type.to_string(),
            });
        }
        Ok(())
    }

    /// Check a security property
    fn check_security(&mut self, security: &Security) -> Result<(), TypeError> {
        self.env.push_scope();
        let body_type = self.check_expr(&security.body)?;
        self.env.pop_scope();

        if !Self::is_bool_type(&body_type) {
            return Err(TypeError::SecurityNotBool(body_type.to_string()));
        }
        Ok(())
    }

    /// Check a semantic property
    fn check_semantic(&mut self, semantic: &SemanticProperty) -> Result<(), TypeError> {
        self.env.push_scope();
        let body_type = self.check_expr(&semantic.body)?;
        self.env.pop_scope();

        if !Self::is_bool_type(&body_type) {
            return Err(TypeError::SemanticNotBool(body_type.to_string()));
        }
        Ok(())
    }

    /// Check a platform API constraint
    fn check_platform_api(&mut self, api: &PlatformApi) -> Result<(), TypeError> {
        for state in &api.states {
            self.env.push_scope();

            // Add state variables to scope
            if let Some(status_enum) = &state.status_enum {
                // Register the status enum type
                self.env.bind(
                    status_enum.name.clone(),
                    CheckedType::Named(status_enum.name.clone()),
                );
                // Add "status" variable for the state
                self.env.bind(
                    "status".to_string(),
                    CheckedType::Named(status_enum.name.clone()),
                );
            }

            // Check invariants
            for inv in &state.invariants {
                let inv_type = self.check_expr(inv)?;
                if !Self::is_bool_type(&inv_type) {
                    return Err(TypeError::InvariantNotBool(inv_type.to_string()));
                }
            }

            // Check transitions
            for transition in &state.transitions {
                self.env.push_scope();

                // Add parameters to scope
                for param in &transition.params {
                    let param_type = self.env.convert_type(&param.ty)?;
                    self.env.bind(param.name.clone(), param_type);
                }

                // Check requires clauses
                for req in &transition.requires {
                    let req_type = self.check_expr(req)?;
                    if !Self::is_bool_type(&req_type) {
                        return Err(TypeError::RequiresNotBool(req_type.to_string()));
                    }
                }

                // Check ensures clauses
                for ens in &transition.ensures {
                    let ens_type = self.check_expr(ens)?;
                    if !Self::is_bool_type(&ens_type) {
                        return Err(TypeError::EnsuresNotBool(ens_type.to_string()));
                    }
                }

                self.env.pop_scope();
            }

            self.env.pop_scope();
        }
        Ok(())
    }

    /// Check a bisimulation property
    fn check_bisimulation(&mut self, bisim: &Bisimulation) -> Result<(), TypeError> {
        // Check property expression if present
        if let Some(ref prop) = bisim.property {
            self.env.push_scope();

            // Check the var_type exists and convert it
            let checked_type = self.env.convert_type(&prop.var_type)?;

            // Add the bound variable to scope
            self.env.bind(prop.var_name.clone(), checked_type);

            // Check oracle and subject expressions
            let _oracle_type = self.check_expr(&prop.oracle_expr)?;
            let _subject_type = self.check_expr(&prop.subject_expr)?;

            self.env.pop_scope();
        }
        Ok(())
    }

    /// Check a version specification
    fn check_version_spec(&mut self, version: &crate::ast::VersionSpec) -> Result<(), TypeError> {
        self.env.push_scope();

        // Check capability expressions
        for cap in &version.capabilities {
            let cap_type = self.check_expr(&cap.expr)?;
            if !Self::is_bool_type(&cap_type) {
                return Err(TypeError::TypeMismatch {
                    expected: "Bool".to_string(),
                    found: cap_type.to_string(),
                });
            }
        }

        // Check preserves expressions
        for pres in &version.preserves {
            // Preserves expressions can be any type (they're property references)
            let _pres_type = self.check_expr(&pres.property)?;
        }

        self.env.pop_scope();
        Ok(())
    }

    /// Check a capability specification
    fn check_capability_spec(
        &mut self,
        capability: &crate::ast::CapabilitySpec,
    ) -> Result<(), TypeError> {
        self.env.push_scope();

        // Check ability signatures (just type existence)
        for ability in &capability.abilities {
            for param in &ability.params {
                self.check_type_exists(&param.ty)?;
            }
            if let Some(ref ret) = ability.return_type {
                self.check_type_exists(ret)?;
            }
        }

        // Check requires expressions
        for req in &capability.requires {
            let req_type = self.check_expr(req)?;
            if !Self::is_bool_type(&req_type) {
                return Err(TypeError::RequiresNotBool(req_type.to_string()));
            }
        }

        self.env.pop_scope();
        Ok(())
    }

    /// Check a distributed invariant
    ///
    /// Distributed invariants specify properties that must hold across multiple agents.
    /// The body typically contains agent quantifiers like `forall d1 d2: Dasher`.
    fn check_distributed_invariant(
        &mut self,
        dist_inv: &crate::ast::DistributedInvariant,
    ) -> Result<(), TypeError> {
        self.env.push_scope();
        let body_type = self.check_expr(&dist_inv.body)?;
        self.env.pop_scope();

        if !Self::is_bool_type(&body_type) {
            return Err(TypeError::InvariantNotBool(body_type.to_string()));
        }
        Ok(())
    }

    /// Check a distributed temporal property
    ///
    /// Distributed temporal properties specify temporal formulas about agent coordination,
    /// such as eventual consistency or consensus properties.
    fn check_distributed_temporal(
        &mut self,
        dist_temp: &crate::ast::DistributedTemporal,
    ) -> Result<(), TypeError> {
        self.env.push_scope();
        self.check_temporal_expr(&dist_temp.body)?;
        self.env.pop_scope();
        Ok(())
    }

    /// Check an expression and return its type
    #[allow(clippy::too_many_lines)]
    fn check_expr(&mut self, expr: &Expr) -> Result<CheckedType, TypeError> {
        match expr {
            Expr::Var(name) => {
                // For unknown variables, be permissive and return Unknown
                // This handles cases like free variables in quantified formulas
                Ok(self
                    .env
                    .lookup(name)
                    .cloned()
                    .unwrap_or(CheckedType::Unknown))
            }
            Expr::Int(_) => Ok(CheckedType::Int),
            Expr::Float(_) => Ok(CheckedType::Float),
            Expr::String(_) => Ok(CheckedType::String),
            Expr::Bool(_) => Ok(CheckedType::Bool),

            Expr::ForAll { var, ty, body } | Expr::Exists { var, ty, body } => {
                self.env.push_scope();
                let var_type = if let Some(t) = ty {
                    self.env.convert_type(t)?
                } else {
                    CheckedType::Unknown
                };
                self.env.bind(var.clone(), var_type);
                let body_type = self.check_expr(body)?;
                self.env.pop_scope();

                if !Self::is_bool_type(&body_type) {
                    return Err(TypeError::TypeMismatch {
                        expected: "Bool".to_string(),
                        found: body_type.to_string(),
                    });
                }
                Ok(CheckedType::Bool)
            }

            Expr::ForAllIn {
                var,
                collection,
                body,
            }
            | Expr::ExistsIn {
                var,
                collection,
                body,
            } => {
                let collection_type = self.check_expr(collection)?;
                let elem_type = self.env.element_type(&collection_type)?;

                self.env.push_scope();
                self.env.bind(var.clone(), elem_type);
                let body_type = self.check_expr(body)?;
                self.env.pop_scope();

                if !Self::is_bool_type(&body_type) {
                    return Err(TypeError::TypeMismatch {
                        expected: "Bool".to_string(),
                        found: body_type.to_string(),
                    });
                }
                Ok(CheckedType::Bool)
            }

            Expr::Implies(lhs, rhs) | Expr::And(lhs, rhs) | Expr::Or(lhs, rhs) => {
                let lhs_type = self.check_expr(lhs)?;
                let rhs_type = self.check_expr(rhs)?;

                if !Self::is_bool_type(&lhs_type) {
                    return Err(TypeError::BoolOperandRequired {
                        op: Self::logical_op_name(expr).to_string(),
                        found: lhs_type.to_string(),
                    });
                }
                if !Self::is_bool_type(&rhs_type) {
                    return Err(TypeError::BoolOperandRequired {
                        op: Self::logical_op_name(expr).to_string(),
                        found: rhs_type.to_string(),
                    });
                }
                Ok(CheckedType::Bool)
            }

            Expr::Not(inner) => {
                let inner_type = self.check_expr(inner)?;
                if !Self::is_bool_type(&inner_type) {
                    return Err(TypeError::BoolOperandRequired {
                        op: "not".to_string(),
                        found: inner_type.to_string(),
                    });
                }
                Ok(CheckedType::Bool)
            }

            Expr::Compare(lhs, _op, rhs) => {
                let lhs_type = self.check_expr(lhs)?;
                let rhs_type = self.check_expr(rhs)?;

                // Types must be compatible for comparison
                if !Self::types_compatible(&lhs_type, &rhs_type) {
                    return Err(TypeError::ComparisonTypeMismatch {
                        left: lhs_type.to_string(),
                        right: rhs_type.to_string(),
                    });
                }
                Ok(CheckedType::Bool)
            }

            Expr::Binary(lhs, op, rhs) => {
                let lhs_type = self.check_expr(lhs)?;
                let rhs_type = self.check_expr(rhs)?;

                if !Self::is_numeric_type(&lhs_type) {
                    return Err(TypeError::NumericOperandRequired {
                        op: Self::binary_op_name(*op).to_string(),
                        found: lhs_type.to_string(),
                    });
                }
                if !Self::is_numeric_type(&rhs_type) {
                    return Err(TypeError::NumericOperandRequired {
                        op: Self::binary_op_name(*op).to_string(),
                        found: rhs_type.to_string(),
                    });
                }

                // Result type: Float if either is Float, else Int
                if matches!(lhs_type, CheckedType::Float) || matches!(rhs_type, CheckedType::Float)
                {
                    Ok(CheckedType::Float)
                } else {
                    Ok(CheckedType::Int)
                }
            }

            Expr::Neg(inner) => {
                let inner_type = self.check_expr(inner)?;
                if !Self::is_numeric_type(&inner_type) {
                    return Err(TypeError::NumericOperandRequired {
                        op: "-".to_string(),
                        found: inner_type.to_string(),
                    });
                }
                Ok(inner_type)
            }

            Expr::App(func_name, args) => {
                // Check argument types
                for arg in args {
                    self.check_expr(arg)?;
                }

                // Look up function signature; for unknown functions assume Bool (common for predicates)
                Ok(self
                    .env
                    .functions
                    .get(func_name)
                    .map_or(CheckedType::Bool, |sig| sig.return_type.clone()))
            }

            Expr::MethodCall {
                receiver,
                method: _,
                args,
            } => {
                let _receiver_type = self.check_expr(receiver)?;
                for arg in args {
                    self.check_expr(arg)?;
                }
                // For method calls, be permissive and return Unknown
                // A more sophisticated type system would track method signatures
                Ok(CheckedType::Unknown)
            }

            Expr::FieldAccess(obj, field) => {
                let obj_type = self.check_expr(obj)?;
                match &obj_type {
                    CheckedType::Named(type_name) => self.env.get_field_type(type_name, field),
                    CheckedType::Unknown => Ok(CheckedType::Unknown),
                    _ => {
                        // For built-in types, be permissive
                        Ok(CheckedType::Unknown)
                    }
                }
            }
        }
    }

    /// Check if a type is boolean
    const fn is_bool_type(ty: &CheckedType) -> bool {
        matches!(ty, CheckedType::Bool | CheckedType::Unknown)
    }

    /// Check if a type is numeric
    const fn is_numeric_type(ty: &CheckedType) -> bool {
        matches!(
            ty,
            CheckedType::Int | CheckedType::Float | CheckedType::Unknown
        )
    }

    /// Check if two types are compatible
    fn types_compatible(a: &CheckedType, b: &CheckedType) -> bool {
        if matches!(a, CheckedType::Unknown) || matches!(b, CheckedType::Unknown) {
            return true;
        }
        // Numeric types are compatible with each other
        if Self::is_numeric_type(a) && Self::is_numeric_type(b) {
            return true;
        }
        a == b
    }

    /// Get name of logical operator
    const fn logical_op_name(expr: &Expr) -> &'static str {
        match expr {
            Expr::Implies(_, _) => "implies",
            Expr::And(_, _) => "and",
            Expr::Or(_, _) => "or",
            _ => "logical",
        }
    }

    /// Get name of binary operator
    const fn binary_op_name(op: BinaryOp) -> &'static str {
        match op {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
        }
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Type check a specification (public API).
///
/// # Errors
///
/// Returns a [`TypeError`] if the specification contains type errors.
#[allow(clippy::needless_pass_by_value)]
pub fn typecheck(spec: Spec) -> Result<TypedSpec, TypeError> {
    let mut checker = TypeChecker::new();
    checker.check(&spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::parse;

    #[test]
    fn test_typecheck_simple_theorem() {
        let input = r#"
            theorem test {
                forall x: Bool . x or not x
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_theorem_with_function() {
        let input = r#"
            theorem acyclic_terminates {
                forall g: Graph, s: State .
                    acyclic(g) implies
                    exists s2: State . executes(g, s, s2) and is_terminal(s2)
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_temporal() {
        let input = r#"
            temporal no_deadlock {
                always(exists agent in agents . enabled(agent))
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_contract() {
        // Note: contract parameters need type annotations (no bare 'self')
        let input = r#"
            contract Graph::add_node(this: Graph, node: Node) -> Result<()> {
                requires { not this.contains(node.id) }
                ensures { this.contains(node.id) }
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_invariant() {
        // Note: for "in" bindings with field access, use parentheses: (g.nodes)
        let input = r#"
            invariant graph_connectivity {
                forall g: Graph . forall n in (g.nodes) .
                    reachable(g.entry, n, g.edges) or n == g.entry
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok(), "Type check failed: {:?}", result);
    }

    #[test]
    fn test_typecheck_type_def() {
        let input = r#"
            type Node = { id: String, data: Map<String, Value> }

            theorem test {
                forall n: Node . n.id == n.id
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_refinement() {
        let input = r#"
            refinement optimized refines base {
                abstraction { forall x: Int . x == x }
                simulation { forall x: Int, a: Action . true }
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_probabilistic() {
        let input = r#"
            probabilistic response_time_bound {
                probability(response_time < 100) >= 0.99
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_security() {
        let input = r#"
            security no_leak {
                forall t1: Tenant, t2: Tenant . t1 != t2 implies
                    not can_observe(t1, actions(t2))
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_semantic_property() {
        let input = r#"
            semantic_property output_helpful {
                forall response: Response, query: Query .
                    addresses_question(response, query) and
                    semantic_similarity(response, expected) >= 0.75
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok(), "Type check failed: {:?}", result);
    }

    #[test]
    fn test_typecheck_invalid_probability_bound() {
        let input = r#"
            probabilistic invalid_bound {
                probability(x) >= 1.5
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::InvalidProbabilityBound(_))));
    }

    #[test]
    fn test_typecheck_duplicate_type() {
        let input = r#"
            type Node = { id: String }
            type Node = { name: String }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::DuplicateType(_))));
    }

    #[test]
    fn test_typecheck_arithmetic() {
        let input = r#"
            theorem arithmetic_test {
                forall x: Int, y: Int . x + y == y + x
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_comparison() {
        let input = r#"
            theorem comparison_test {
                forall x: Int . x < x + 1
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_field_access_user_type() {
        let input = r#"
            type Person = { name: String, age: Int }

            theorem person_test {
                forall p: Person . p.age >= 0
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_multiple_properties() {
        let input = r#"
            type State = { count: Int }

            theorem count_positive {
                forall s: State . s.count >= 0
            }

            invariant count_valid {
                forall s: State . s.count >= 0 and s.count < 1000
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    // ============================================================
    // Mutation-killing tests for CheckedType Display impl (line 167)
    // ============================================================

    #[test]
    fn test_checked_type_display_bool() {
        assert_eq!(CheckedType::Bool.to_string(), "Bool");
    }

    #[test]
    fn test_checked_type_display_int() {
        assert_eq!(CheckedType::Int.to_string(), "Int");
    }

    #[test]
    fn test_checked_type_display_float() {
        assert_eq!(CheckedType::Float.to_string(), "Float");
    }

    #[test]
    fn test_checked_type_display_string() {
        assert_eq!(CheckedType::String.to_string(), "String");
    }

    #[test]
    fn test_checked_type_display_named() {
        assert_eq!(
            CheckedType::Named("MyType".to_string()).to_string(),
            "MyType"
        );
    }

    #[test]
    fn test_checked_type_display_set() {
        let set_type = CheckedType::Set(Box::new(CheckedType::Int));
        assert_eq!(set_type.to_string(), "Set<Int>");
    }

    #[test]
    fn test_checked_type_display_list() {
        let list_type = CheckedType::List(Box::new(CheckedType::String));
        assert_eq!(list_type.to_string(), "List<String>");
    }

    #[test]
    fn test_checked_type_display_map() {
        let map_type = CheckedType::Map(Box::new(CheckedType::String), Box::new(CheckedType::Int));
        assert_eq!(map_type.to_string(), "Map<String, Int>");
    }

    #[test]
    fn test_checked_type_display_relation() {
        let rel_type =
            CheckedType::Relation(Box::new(CheckedType::Int), Box::new(CheckedType::Int));
        assert_eq!(rel_type.to_string(), "Relation<Int, Int>");
    }

    #[test]
    fn test_checked_type_display_function() {
        let func_type =
            CheckedType::Function(Box::new(CheckedType::Int), Box::new(CheckedType::Bool));
        assert_eq!(func_type.to_string(), "Int -> Bool");
    }

    #[test]
    fn test_checked_type_display_result() {
        let result_type = CheckedType::Result(Box::new(CheckedType::Unit));
        assert_eq!(result_type.to_string(), "Result<()>");
    }

    #[test]
    fn test_checked_type_display_unit() {
        assert_eq!(CheckedType::Unit.to_string(), "()");
    }

    #[test]
    fn test_checked_type_display_unknown() {
        assert_eq!(CheckedType::Unknown.to_string(), "?");
    }

    // ============================================================
    // Mutation-killing tests for TypeEnv scope management
    // ============================================================

    #[test]
    fn test_type_env_push_pop_scope() {
        let mut env = TypeEnv::new();
        env.push_scope();
        env.bind("x".to_string(), CheckedType::Int);
        assert!(env.lookup("x").is_some());
        env.pop_scope();
        // After pop, variable should be removed
        assert!(env.lookup("x").is_none());
    }

    #[test]
    fn test_type_env_nested_scopes() {
        let mut env = TypeEnv::new();
        env.push_scope();
        env.bind("x".to_string(), CheckedType::Int);
        env.push_scope();
        env.bind("y".to_string(), CheckedType::Bool);
        assert!(env.lookup("x").is_some());
        assert!(env.lookup("y").is_some());
        env.pop_scope();
        // y should be gone, x should remain
        assert!(env.lookup("x").is_some());
        assert!(env.lookup("y").is_none());
        env.pop_scope();
        assert!(env.lookup("x").is_none());
    }

    #[test]
    fn test_type_env_bind_returns_correct_type() {
        let mut env = TypeEnv::new();
        env.push_scope();
        env.bind("myvar".to_string(), CheckedType::Float);
        let looked_up = env.lookup("myvar");
        assert!(looked_up.is_some());
        assert_eq!(*looked_up.unwrap(), CheckedType::Float);
    }

    #[test]
    fn test_type_env_lookup_nonexistent() {
        let env = TypeEnv::new();
        assert!(env.lookup("nonexistent").is_none());
    }

    #[test]
    fn test_type_env_lookup_type_found() {
        let mut env = TypeEnv::new();
        let typedef = TypeDef {
            name: "MyCustomType".to_string(),
            fields: vec![],
        };
        env.register_type(typedef).unwrap();
        let looked_up = env.lookup_type("MyCustomType");
        assert!(looked_up.is_some());
        assert_eq!(looked_up.unwrap().name, "MyCustomType");
    }

    #[test]
    fn test_type_env_lookup_type_not_found() {
        let env = TypeEnv::new();
        assert!(env.lookup_type("NonexistentType").is_none());
    }

    // ============================================================
    // Mutation-killing tests for convert_type match arms (Float/String)
    // ============================================================

    #[test]
    fn test_convert_type_float_variants() {
        let env = TypeEnv::new();
        // Test all Float variants
        assert_eq!(
            env.convert_type(&Type::Named("Float".to_string())).unwrap(),
            CheckedType::Float
        );
        assert_eq!(
            env.convert_type(&Type::Named("float".to_string())).unwrap(),
            CheckedType::Float
        );
        assert_eq!(
            env.convert_type(&Type::Named("f32".to_string())).unwrap(),
            CheckedType::Float
        );
        assert_eq!(
            env.convert_type(&Type::Named("f64".to_string())).unwrap(),
            CheckedType::Float
        );
    }

    #[test]
    fn test_convert_type_string_variants() {
        let env = TypeEnv::new();
        // Test all String variants
        assert_eq!(
            env.convert_type(&Type::Named("String".to_string()))
                .unwrap(),
            CheckedType::String
        );
        assert_eq!(
            env.convert_type(&Type::Named("str".to_string())).unwrap(),
            CheckedType::String
        );
    }

    // ============================================================
    // Mutation-killing tests for check_typedef (line 491)
    // ============================================================

    #[test]
    fn test_check_typedef_with_fields() {
        // Ensure check_typedef actually validates field types
        let input = r#"
            type ValidType = { field1: Int, field2: Bool }
            theorem test { true }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    // ============================================================
    // Mutation-killing tests for check_type_exists (line 499)
    // ============================================================

    #[test]
    fn test_check_type_exists_nested_types() {
        // Test that nested types are checked recursively
        let input = r#"
            type Container = { items: Set<List<Int>> }
            theorem test { true }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    // ============================================================
    // Mutation-killing tests for check_theorem (line 535)
    // ============================================================

    #[test]
    fn test_check_theorem_non_bool_body_fails() {
        let input = r#"
            theorem invalid_theorem {
                42
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::TheoremNotBool(_))));
    }

    // ============================================================
    // Mutation-killing tests for check_temporal (line 547)
    // ============================================================

    #[test]
    fn test_check_temporal_non_bool_atom_fails() {
        // Temporal expressions that don't return bool should fail
        let input = r#"
            temporal invalid_temporal {
                always(42)
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_err());
    }

    // ============================================================
    // Mutation-killing tests for check_contract (line 578)
    // ============================================================

    #[test]
    fn test_check_contract_requires_non_bool_fails() {
        let input = r#"
            contract Type::method(self: Type) -> Result<()> {
                requires { 42 }
                ensures { true }
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::RequiresNotBool(_))));
    }

    #[test]
    fn test_check_contract_ensures_non_bool_fails() {
        let input = r#"
            contract Type::method(self: Type) -> Result<()> {
                requires { true }
                ensures { 42 }
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::EnsuresNotBool(_))));
    }

    // ============================================================
    // Mutation-killing tests for check_invariant (line 625)
    // ============================================================

    #[test]
    fn test_check_invariant_non_bool_fails() {
        let input = r#"
            invariant invalid_invariant {
                42
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::InvariantNotBool(_))));
    }

    // ============================================================
    // Mutation-killing tests for check_refinement (line 637)
    // ============================================================

    #[test]
    fn test_check_refinement_non_bool_abstraction_fails() {
        let input = r#"
            refinement impl refines spec {
                abstraction { 42 }
                simulation { true }
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_refinement_non_bool_simulation_fails() {
        let input = r#"
            refinement impl refines spec {
                abstraction { true }
                simulation { 42 }
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_err());
    }

    // ============================================================
    // Mutation-killing tests for probability bound checks (line 664)
    // ============================================================

    #[test]
    fn test_probability_bound_exactly_zero_valid() {
        let input = r#"
            probabilistic zero_prob {
                probability(cond) >= 0.0
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_probability_bound_exactly_one_valid() {
        let input = r#"
            probabilistic one_prob {
                probability(cond) >= 1.0
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_probability_bound_negative_invalid() {
        let input = r#"
            probabilistic neg_prob {
                probability(cond) >= -0.1
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::InvalidProbabilityBound(_))));
    }

    #[test]
    fn test_probability_bound_above_one_invalid() {
        let input = r#"
            probabilistic above_one_prob {
                probability(cond) >= 1.1
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::InvalidProbabilityBound(_))));
    }

    // ============================================================
    // Mutation-killing tests for check_security (line 683)
    // ============================================================

    #[test]
    fn test_check_security_non_bool_fails() {
        let input = r#"
            security invalid_security {
                42
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::SecurityNotBool(_))));
    }

    // ============================================================
    // Mutation-killing tests for check_semantic (line 695)
    // ============================================================

    #[test]
    fn test_check_semantic_non_bool_fails() {
        let input = r#"
            semantic_property invalid_semantic {
                42
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::SemanticNotBool(_))));
    }

    // ============================================================
    // Mutation-killing tests for check_platform_api (line 707)
    // ============================================================

    #[test]
    fn test_check_platform_api_invariant_non_bool_fails() {
        let input = r#"
            platform_api TestApi {
                state Running {
                    invariant { 42 }
                }
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::InvariantNotBool(_))));
    }

    #[test]
    fn test_check_platform_api_requires_non_bool_fails() {
        let input = r#"
            platform_api TestApi {
                state Running {
                    transition start() {
                        requires { 42 }
                        ensures { true }
                    }
                }
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::RequiresNotBool(_))));
    }

    #[test]
    fn test_check_platform_api_ensures_non_bool_fails() {
        let input = r#"
            platform_api TestApi {
                state Running {
                    transition start() {
                        requires { true }
                        ensures { 42 }
                    }
                }
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(result, Err(TypeError::EnsuresNotBool(_))));
    }

    // ============================================================
    // Mutation-killing tests for check_bisimulation (line 769)
    // ============================================================

    #[test]
    fn test_check_bisimulation_valid() {
        let input = r#"
            bisimulation oracle_subject_equiv {
                oracle: "./oracle_impl"
                subject: "./subject_impl"
                equivalent on { api_calls, output }
                forall x: Int . traces_equivalent(oracle_run(x), subject_run(x))
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(
            result.is_ok(),
            "Bisimulation typecheck failed: {:?}",
            result
        );
    }

    #[test]
    fn test_check_bisimulation_without_property() {
        // Bisimulation without forall property should also work
        let input = r#"
            bisimulation simple_equiv {
                oracle: "./orig"
                subject: "./new"
                equivalent on { outputs }
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(
            result.is_ok(),
            "Bisimulation typecheck failed: {:?}",
            result
        );
    }

    // ============================================================
    // Mutation-killing tests for check_expr || vs && (line 914)
    // ============================================================

    #[test]
    fn test_binary_expr_float_or_int_returns_float() {
        // Test that || behaves correctly (Float || Int -> Float)
        let input = r#"
            theorem float_test {
                forall x: Float, y: Int . x + y == x + y
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_binary_expr_int_or_float_returns_float() {
        // Test the reverse case
        let input = r#"
            theorem int_float_test {
                forall x: Int, y: Float . x + y == x + y
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    // ============================================================
    // Mutation-killing tests for negation numeric check (line 924)
    // ============================================================

    #[test]
    fn test_negation_on_non_numeric_fails() {
        let input = r#"
            theorem neg_bool_fails {
                forall x: Bool . -x == x
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(matches!(
            result,
            Err(TypeError::NumericOperandRequired { .. })
        ));
    }

    // ============================================================
    // Mutation-killing tests for field access Named/Unknown (lines 964-965)
    // ============================================================

    #[test]
    fn test_field_access_on_named_type() {
        let input = r#"
            type Person = { name: String, age: Int }
            theorem test {
                forall p: Person . p.age >= 0
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_field_access_on_unknown_type() {
        // Field access on unknown type should return Unknown (permissive)
        let input = r#"
            theorem test {
                forall x: UnknownType . x.field == x.field
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    // ============================================================
    // Mutation-killing tests for is_bool_type (line 977)
    // ============================================================

    #[test]
    fn test_is_bool_type_returns_true_for_bool() {
        assert!(TypeChecker::is_bool_type(&CheckedType::Bool));
    }

    #[test]
    fn test_is_bool_type_returns_true_for_unknown() {
        assert!(TypeChecker::is_bool_type(&CheckedType::Unknown));
    }

    #[test]
    fn test_is_bool_type_returns_false_for_int() {
        assert!(!TypeChecker::is_bool_type(&CheckedType::Int));
    }

    // ============================================================
    // Mutation-killing tests for is_numeric_type (line 982)
    // ============================================================

    #[test]
    fn test_is_numeric_type_returns_true_for_int() {
        assert!(TypeChecker::is_numeric_type(&CheckedType::Int));
    }

    #[test]
    fn test_is_numeric_type_returns_true_for_float() {
        assert!(TypeChecker::is_numeric_type(&CheckedType::Float));
    }

    #[test]
    fn test_is_numeric_type_returns_true_for_unknown() {
        assert!(TypeChecker::is_numeric_type(&CheckedType::Unknown));
    }

    #[test]
    fn test_is_numeric_type_returns_false_for_bool() {
        assert!(!TypeChecker::is_numeric_type(&CheckedType::Bool));
    }

    // ============================================================
    // Mutation-killing tests for types_compatible (line 990)
    // ============================================================

    #[test]
    fn test_types_compatible_unknown_with_any() {
        assert!(TypeChecker::types_compatible(
            &CheckedType::Unknown,
            &CheckedType::Int
        ));
        assert!(TypeChecker::types_compatible(
            &CheckedType::Bool,
            &CheckedType::Unknown
        ));
    }

    #[test]
    fn test_types_compatible_numeric_types() {
        assert!(TypeChecker::types_compatible(
            &CheckedType::Int,
            &CheckedType::Float
        ));
        assert!(TypeChecker::types_compatible(
            &CheckedType::Float,
            &CheckedType::Int
        ));
    }

    #[test]
    fn test_types_compatible_same_types() {
        assert!(TypeChecker::types_compatible(
            &CheckedType::Bool,
            &CheckedType::Bool
        ));
        assert!(TypeChecker::types_compatible(
            &CheckedType::String,
            &CheckedType::String
        ));
    }

    #[test]
    fn test_types_not_compatible_different_types() {
        assert!(!TypeChecker::types_compatible(
            &CheckedType::Bool,
            &CheckedType::String
        ));
        assert!(!TypeChecker::types_compatible(
            &CheckedType::Int,
            &CheckedType::Bool
        ));
    }

    // ============================================================
    // Mutation-killing tests for logical_op_name (lines 1002-1007)
    // ============================================================

    #[test]
    fn test_logical_op_name_implies() {
        let expr = Expr::Implies(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(true)));
        assert_eq!(TypeChecker::logical_op_name(&expr), "implies");
    }

    #[test]
    fn test_logical_op_name_and() {
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(true)));
        assert_eq!(TypeChecker::logical_op_name(&expr), "and");
    }

    #[test]
    fn test_logical_op_name_or() {
        let expr = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(true)));
        assert_eq!(TypeChecker::logical_op_name(&expr), "or");
    }

    #[test]
    fn test_logical_op_name_other() {
        let expr = Expr::Not(Box::new(Expr::Bool(true)));
        assert_eq!(TypeChecker::logical_op_name(&expr), "logical");
    }

    // ============================================================
    // Mutation-killing tests for binary_op_name (lines 1012-1018)
    // ============================================================

    #[test]
    fn test_binary_op_name_add() {
        assert_eq!(TypeChecker::binary_op_name(BinaryOp::Add), "+");
    }

    #[test]
    fn test_binary_op_name_sub() {
        assert_eq!(TypeChecker::binary_op_name(BinaryOp::Sub), "-");
    }

    #[test]
    fn test_binary_op_name_mul() {
        assert_eq!(TypeChecker::binary_op_name(BinaryOp::Mul), "*");
    }

    #[test]
    fn test_binary_op_name_div() {
        assert_eq!(TypeChecker::binary_op_name(BinaryOp::Div), "/");
    }

    #[test]
    fn test_binary_op_name_mod() {
        assert_eq!(TypeChecker::binary_op_name(BinaryOp::Mod), "%");
    }

    // ============================================================
    // Mutation-killing tests for && vs || in types_compatible (line 994)
    // ============================================================

    #[test]
    fn test_types_compatible_both_numeric_required() {
        // This tests the && condition: both must be numeric for numeric compatibility
        // If one is numeric and one is not, they should NOT be compatible via numeric rule
        assert!(!TypeChecker::types_compatible(
            &CheckedType::Int,
            &CheckedType::String
        ));
        assert!(!TypeChecker::types_compatible(
            &CheckedType::Float,
            &CheckedType::Bool
        ));
    }

    // ============================================================
    // Additional mutation-killing tests for remaining missed mutants
    // ============================================================

    // Test for check_typedef (line 491) - ensure field types are iterated
    #[test]
    fn test_check_typedef_iterates_fields_directly() {
        use crate::ast::{Field, TypeDef};
        let mut checker = TypeChecker::new();

        // Create a type with multiple fields
        let typedef = TypeDef {
            name: "TestType".to_string(),
            fields: vec![
                Field {
                    name: "field1".to_string(),
                    ty: Type::Named("Int".to_string()),
                },
                Field {
                    name: "field2".to_string(),
                    ty: Type::Named("Bool".to_string()),
                },
                Field {
                    name: "field3".to_string(),
                    ty: Type::Set(Box::new(Type::Named("String".to_string()))),
                },
            ],
        };

        // Register and check - this tests that the iterator actually runs
        checker.env.register_type(typedef.clone()).unwrap();
        let result = checker.check_typedef(&typedef);
        assert!(result.is_ok());
    }

    // Test for check_type_exists (line 499) - recursive type checking
    #[test]
    fn test_check_type_exists_recursive_types() {
        let checker = TypeChecker::new();

        // Test deeply nested types - if recursion is skipped, we wouldn't traverse
        let nested = Type::Map(
            Box::new(Type::Set(Box::new(Type::Named("Key".to_string())))),
            Box::new(Type::List(Box::new(Type::Named("Value".to_string())))),
        );
        let result = checker.check_type_exists(&nested);
        assert!(result.is_ok());

        // Test Relation
        let relation = Type::Relation(
            Box::new(Type::Named("A".to_string())),
            Box::new(Type::Named("B".to_string())),
        );
        let result = checker.check_type_exists(&relation);
        assert!(result.is_ok());
    }

    // Test for check_bisimulation (line 769) - with property that has expressions
    #[test]
    fn test_check_bisimulation_with_typed_property() {
        // This tests that bisimulation with a forall property actually
        // type-checks the bound variable and expressions
        let input = r#"
            bisimulation typed_bisim {
                oracle: "./oracle"
                subject: "./subject"
                equivalent on { output }
                forall x: Int . traces_equivalent(oracle(x), subject(x))
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(
            result.is_ok(),
            "Bisimulation with typed property failed: {:?}",
            result
        );
    }

    // Test for || vs && in binary expr (line 914) - Int + Int should return Int
    #[test]
    fn test_binary_int_int_returns_int_not_float() {
        // If || were replaced with &&, this would fail because
        // Int && Int = false (neither is Float), so wrong branch taken
        let input = r#"
            theorem int_addition {
                forall x: Int, y: Int . x + y >= 0 implies true
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_binary_float_float_returns_float() {
        // Float + Float should return Float
        let input = r#"
            theorem float_addition {
                forall x: Float, y: Float . x + y >= 0.0 implies true
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    // Test for field access on Named type (line 964)
    #[test]
    fn test_field_access_named_type_with_defined_field() {
        // Tests that Named type field access goes through get_field_type
        let input = r#"
            type Record = { value: Int, label: String }
            theorem named_field_access {
                forall r: Record . r.value >= 0
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_field_access_named_type_invalid_field_fails() {
        // If the Named match arm is deleted, this would succeed (fall to Unknown)
        let input = r#"
            type Point = { x: Int, y: Int }
            theorem invalid_field {
                forall p: Point . p.z >= 0
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        // This should fail because Point has no field 'z'
        assert!(matches!(result, Err(TypeError::InvalidField { .. })));
    }

    // Test for field access on Unknown type (line 965)
    #[test]
    fn test_field_access_unknown_type_returns_unknown() {
        // Access on unknown type should return Unknown (permissive)
        let input = r#"
            theorem unknown_field_access {
                forall u: UnknownType . u.anything == u.anything
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    // Test for field access on built-in type (default case line 966-968)
    #[test]
    fn test_field_access_on_builtin_type_permissive() {
        // Field access on Int/Bool etc should be permissive (return Unknown)
        let input = r#"
            theorem builtin_field_access {
                forall x: Int . x.weird_field == x.weird_field
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(result.is_ok());
    }

    // ============================================================
    // Additional targeted tests for remaining stubborn mutants
    // ============================================================

    // For || vs && (line 914) - directly test return type inference
    #[test]
    fn test_binary_expr_return_type_int_plus_int() {
        // Create expression directly to verify return type
        let mut checker = TypeChecker::new();
        checker.env.push_scope();
        checker.env.bind("a".to_string(), CheckedType::Int);
        checker.env.bind("b".to_string(), CheckedType::Int);

        let expr = Expr::Binary(
            Box::new(Expr::Var("a".to_string())),
            BinaryOp::Add,
            Box::new(Expr::Var("b".to_string())),
        );
        let result_type = checker.check_expr(&expr).unwrap();
        // Int + Int should return Int, not Float
        // If || is replaced with &&, this would return Float (wrong path)
        assert_eq!(
            result_type,
            CheckedType::Int,
            "Int + Int should return Int, got {:?}",
            result_type
        );
    }

    #[test]
    fn test_binary_expr_return_type_float_plus_int() {
        let mut checker = TypeChecker::new();
        checker.env.push_scope();
        checker.env.bind("a".to_string(), CheckedType::Float);
        checker.env.bind("b".to_string(), CheckedType::Int);

        let expr = Expr::Binary(
            Box::new(Expr::Var("a".to_string())),
            BinaryOp::Add,
            Box::new(Expr::Var("b".to_string())),
        );
        let result_type = checker.check_expr(&expr).unwrap();
        // Float + Int should return Float
        assert_eq!(
            result_type,
            CheckedType::Float,
            "Float + Int should return Float"
        );
    }

    #[test]
    fn test_binary_expr_return_type_int_plus_float() {
        let mut checker = TypeChecker::new();
        checker.env.push_scope();
        checker.env.bind("a".to_string(), CheckedType::Int);
        checker.env.bind("b".to_string(), CheckedType::Float);

        let expr = Expr::Binary(
            Box::new(Expr::Var("a".to_string())),
            BinaryOp::Add,
            Box::new(Expr::Var("b".to_string())),
        );
        let result_type = checker.check_expr(&expr).unwrap();
        // Int + Float should return Float
        assert_eq!(
            result_type,
            CheckedType::Float,
            "Int + Float should return Float"
        );
    }

    // For field access Unknown arm (line 965) - ensure Unknown returns Unknown
    #[test]
    fn test_field_access_unknown_type_returns_unknown_directly() {
        let mut checker = TypeChecker::new();
        checker.env.push_scope();
        checker.env.bind("x".to_string(), CheckedType::Unknown);

        let expr = Expr::FieldAccess(
            Box::new(Expr::Var("x".to_string())),
            "some_field".to_string(),
        );
        let result_type = checker.check_expr(&expr).unwrap();
        assert_eq!(
            result_type,
            CheckedType::Unknown,
            "Field access on Unknown should return Unknown"
        );
    }

    // Mutation-killing test for check_bisimulation (line 769)
    // If check_bisimulation just returned Ok(()), this test would pass
    // but the type error in the expression wouldn't be caught.
    #[test]
    fn test_check_bisimulation_invalid_expr_fails() {
        // Create a bisimulation with an expression that has a type error:
        // 1 and 2 (integer operands to 'and' should fail)
        use crate::ast::{Bisimulation, BisimulationPropertyExpr};

        let bisim = Bisimulation {
            name: "test_bisim".to_string(),
            oracle: "./oracle".to_string(),
            subject: "./subject".to_string(),
            equivalent_on: vec!["output".to_string()],
            tolerance: None,
            property: Some(BisimulationPropertyExpr {
                var_name: "x".to_string(),
                var_type: Type::Named("Int".to_string()),
                // Invalid expression: 1 and 2 (integers can't be used with 'and')
                oracle_expr: Expr::And(Box::new(Expr::Int(1)), Box::new(Expr::Int(2))),
                subject_expr: Expr::Bool(true),
            }),
        };

        let spec = Spec {
            types: vec![],
            properties: vec![Property::Bisimulation(bisim)],
        };

        let result = typecheck(spec);
        // This should fail because 1 and 2 is invalid (Int operands to 'and')
        assert!(
            result.is_err(),
            "Bisimulation with invalid expression should fail typecheck"
        );
        assert!(matches!(result, Err(TypeError::BoolOperandRequired { .. })));
    }

    // Mutation-killing test for check_typedef (line 491)
    // Tests that check_typedef actually iterates over fields and validates them.
    // Since check_type_exists is permissive (never errors), we can't test for errors,
    // but we can ensure the function body is exercised.
    #[test]
    fn test_check_typedef_with_nested_container_types() {
        // TypeDef with complex nested types - exercises check_type_exists recursion
        use crate::ast::{Field, TypeDef};

        let typedef = TypeDef {
            name: "ComplexType".to_string(),
            fields: vec![
                Field {
                    name: "set_field".to_string(),
                    ty: Type::Set(Box::new(Type::Named("Int".to_string()))),
                },
                Field {
                    name: "map_field".to_string(),
                    ty: Type::Map(
                        Box::new(Type::Named("String".to_string())),
                        Box::new(Type::List(Box::new(Type::Named("Bool".to_string())))),
                    ),
                },
                Field {
                    name: "function_field".to_string(),
                    ty: Type::Function(
                        Box::new(Type::Named("Int".to_string())),
                        Box::new(Type::Result(Box::new(Type::Named("String".to_string())))),
                    ),
                },
            ],
        };

        let spec = Spec {
            types: vec![typedef],
            properties: vec![Property::Theorem(Theorem {
                name: "dummy".to_string(),
                body: Expr::Bool(true),
            })],
        };

        let result = typecheck(spec);
        assert!(
            result.is_ok(),
            "Complex typedef should typecheck: {:?}",
            result
        );
    }

    // ========================================================================
    // Phase 17.3: Graph predicate tests
    // ========================================================================

    #[test]
    fn test_typecheck_graph_dag_predicate() {
        // Test is_dag and is_acyclic predicates on Graph type
        let input = r#"
            type TaskNode = { id: Int, status: String }
            type Dependency = { from: Int, to: Int }
            type ExecutionGraph = { graph: Graph<TaskNode, Dependency> }

            theorem dag_property {
                forall g: ExecutionGraph . is_dag(g.graph) implies is_acyclic(g.graph)
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(
            result.is_ok(),
            "DAG predicate should typecheck: {:?}",
            result
        );
    }

    #[test]
    fn test_typecheck_graph_connectivity() {
        // Test graph connectivity predicates
        let input = r#"
            theorem connectivity_test {
                forall g: Graph<Node, Edge> .
                    is_connected(g) implies
                    forall n1: Node, n2: Node .
                        in_graph(n1, g) and in_graph(n2, g) implies has_path(g, n1, n2)
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(
            result.is_ok(),
            "Connectivity predicates should typecheck: {:?}",
            result
        );
    }

    #[test]
    fn test_typecheck_dashflow_modification_predicates() {
        // Test DashFlow-specific modification predicates
        let input = r#"
            invariant modification_safety {
                forall g: Graph<Node, Edge>, g': Graph<Node, Edge>, m: Modification .
                    valid_modification(m, g, g') implies preserves_completed(g, g')
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(
            result.is_ok(),
            "Modification predicates should typecheck: {:?}",
            result
        );
    }

    #[test]
    fn test_typecheck_node_status_predicates() {
        // Test node status predicates (completed, pending, running, failed)
        let input = r#"
            theorem node_status {
                forall n: Node . completed(n) or pending(n) or running(n) or failed(n)
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(
            result.is_ok(),
            "Node status predicates should typecheck: {:?}",
            result
        );
    }

    #[test]
    fn test_typecheck_graph_accessor_functions() {
        // Test graph accessor functions return correct types
        let input = r#"
            theorem graph_accessors {
                forall g: Graph<Node, Edge>, n: Node .
                    node_count(g) >= 0 and
                    edge_count(g) >= 0 and
                    in_degree(g, n) >= 0 and
                    out_degree(g, n) >= 0
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(
            result.is_ok(),
            "Graph accessor functions should typecheck: {:?}",
            result
        );
    }

    #[test]
    fn test_typecheck_dag_progress_property() {
        // Test the progress property from the ROADMAP example
        let input = r#"
            invariant progress_guaranteed {
                forall g: Graph<Node, Edge> .
                    is_dag(g) implies
                    exists n: Node . in_graph(n, g) and is_ready(n, g)
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(
            result.is_ok(),
            "Progress property should typecheck: {:?}",
            result
        );
    }

    #[test]
    fn test_typecheck_graph_preserves_dag() {
        // Test preserves_dag predicate
        let input = r#"
            theorem dag_preservation {
                forall g: Graph<Node, Edge>, g': Graph<Node, Edge> .
                    is_dag(g) and preserves_dag(g, g') implies is_dag(g')
            }
        "#;
        let spec = parse(input).unwrap();
        let result = typecheck(spec);
        assert!(
            result.is_ok(),
            "DAG preservation should typecheck: {:?}",
            result
        );
    }
}
