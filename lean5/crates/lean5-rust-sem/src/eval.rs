//! Rust Operational Semantics - Expression Evaluation
//!
//! This module implements a small-step operational semantics for Rust expressions.
//! It provides an interpreter that evaluates expressions in an execution context,
//! handling control flow, memory operations, and ownership tracking.
//!
//! ## Semantics Model
//!
//! The evaluation model follows Rust's operational semantics:
//!
//! - **Eager evaluation**: Arguments evaluated left-to-right before function application
//! - **Value semantics**: Values are copied/moved based on type's Copy trait
//! - **Control flow**: if/else, match, loops with break/continue
//! - **Memory model**: Stack-based locals with heap allocations
//!
//! ## Evaluation Rules (Big-Step)
//!
//! ```text
//! Literal:      ⟨lit⟩ ↓ lit
//! Variable:     ⟨x⟩ ↓ σ(x)          where σ is the environment
//! BinOp:        ⟨e1 ⊕ e2⟩ ↓ v1 ⊕ v2  where ⟨e1⟩ ↓ v1, ⟨e2⟩ ↓ v2
//! If-True:      ⟨if true { e1 } else { e2 }⟩ ↓ v1  where ⟨e1⟩ ↓ v1
//! If-False:     ⟨if false { e1 } else { e2 }⟩ ↓ v2  where ⟨e2⟩ ↓ v2
//! Block:        ⟨{ s1; ...; sn; e }⟩ ↓ v  where each si executes, ⟨e⟩ ↓ v
//! ```

use crate::expr::{EnumVariantPayload, EvalResult, Expr, Item, MatchArm, Stmt};
use crate::stmt::{match_pattern, ExecContext, FunctionDef, PatternBindings, StmtResult};
use crate::types::RustType;
use crate::values::{cast_value, eval_binop, eval_unop, EnumPayload, Value};
use std::collections::{BTreeMap, HashMap};

/// Maximum recursion depth for interpreter (prevent stack overflow)
const MAX_RECURSION_DEPTH: usize = 1000;

/// Maximum loop iterations (prevent infinite loops during interpretation)
const MAX_LOOP_ITERATIONS: usize = 100_000;

/// Interpreter state
#[derive(Debug)]
pub struct Interpreter {
    /// Execution context with memory, stack, functions, types
    pub ctx: ExecContext,
    /// Variable bindings (name -> value)
    pub bindings: Vec<HashMap<String, Value>>,
    /// Current recursion depth
    pub recursion_depth: usize,
}

impl Interpreter {
    /// Create a new interpreter
    pub fn new() -> Self {
        let mut interp = Self {
            ctx: ExecContext::new(),
            bindings: vec![HashMap::new()],
            recursion_depth: 0,
        };
        // Push initial stack frame
        interp.ctx.stack.push_frame();
        interp
    }

    /// Create interpreter with existing context
    pub fn with_context(ctx: ExecContext) -> Self {
        let mut interp = Self {
            ctx,
            bindings: vec![HashMap::new()],
            recursion_depth: 0,
        };
        // Ensure we have at least one stack frame
        if interp.ctx.stack.depth() == 0 {
            interp.ctx.stack.push_frame();
        }
        interp
    }

    /// Push a new binding scope
    fn push_scope(&mut self) {
        self.bindings.push(HashMap::new());
    }

    /// Pop a binding scope
    fn pop_scope(&mut self) {
        self.bindings.pop();
    }

    /// Look up a variable in the current scope chain
    fn lookup(&self, name: &str) -> Option<&Value> {
        for scope in self.bindings.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v);
            }
        }
        None
    }

    /// Bind a variable in the current scope
    fn bind(&mut self, name: String, value: Value) {
        if let Some(scope) = self.bindings.last_mut() {
            scope.insert(name, value);
        }
    }

    /// Apply pattern bindings to current scope
    fn apply_bindings(&mut self, bindings: PatternBindings) {
        for (name, value, _mutable) in bindings.bindings {
            self.bind(name, value);
        }
    }

    /// Evaluate an expression
    pub fn eval(&mut self, expr: &Expr) -> EvalResult {
        // Check recursion depth
        if self.recursion_depth > MAX_RECURSION_DEPTH {
            return EvalResult::Error("maximum recursion depth exceeded".to_string());
        }

        match expr {
            Expr::Literal(v) => EvalResult::Value(v.clone()),

            Expr::Var { name, .. } => {
                match self.lookup(name) {
                    Some(v) => EvalResult::Value(v.clone()),
                    None => {
                        // Check if it's a function name
                        if self.ctx.get_function(name).is_some() {
                            EvalResult::Value(Value::FnPtr { name: name.clone() })
                        } else {
                            EvalResult::Error(format!("undefined variable: {name}"))
                        }
                    }
                }
            }

            Expr::Field { base, field } => {
                let base_val = match self.eval(base) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                match base_val {
                    Value::Struct { fields, .. } => match fields.get(field) {
                        Some(v) => EvalResult::Value(v.clone()),
                        None => EvalResult::Error(format!("field {field} not found")),
                    },
                    Value::Tuple(elems) => {
                        // Tuple field access like .0, .1, etc.
                        if let Ok(idx) = field.parse::<usize>() {
                            match elems.get(idx) {
                                Some(v) => EvalResult::Value(v.clone()),
                                None => {
                                    EvalResult::Error(format!("tuple index {idx} out of bounds"))
                                }
                            }
                        } else {
                            EvalResult::Error(format!("invalid tuple field: {field}"))
                        }
                    }
                    _ => EvalResult::Error("field access on non-struct value".to_string()),
                }
            }

            Expr::Index { base, index } => {
                let base_val = match self.eval(base) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                let index_val = match self.eval(index) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                let idx = match index_val {
                    Value::Uint { value, .. } => match usize::try_from(value) {
                        Ok(idx) => idx,
                        Err(_) => {
                            return EvalResult::Error(format!("index {value} too large for usize"))
                        }
                    },
                    Value::Int { value, .. } if value >= 0 => match usize::try_from(value) {
                        Ok(idx) => idx,
                        Err(_) => {
                            return EvalResult::Error(format!("index {value} too large for usize"))
                        }
                    },
                    _ => {
                        return EvalResult::Error(
                            "index must be a non-negative integer".to_string(),
                        )
                    }
                };
                match base_val {
                    Value::Array(elems) | Value::Tuple(elems) => match elems.get(idx) {
                        Some(v) => EvalResult::Value(v.clone()),
                        None => EvalResult::Error(format!("index {idx} out of bounds")),
                    },
                    _ => EvalResult::Error("index on non-array value".to_string()),
                }
            }

            Expr::Deref(inner) => {
                let inner_val = match self.eval(inner) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                // For now, we don't fully implement memory dereferencing
                // In a full implementation, this would read from memory
                match inner_val {
                    Value::Reference { addr, .. } | Value::RawPtr { addr, .. } => {
                        // Try to read value from memory (simplified)
                        // In a full implementation we'd need type info to know size
                        match self.ctx.memory.read_u64(addr) {
                            Ok(v) => EvalResult::Value(Value::u64(v)),
                            Err(e) => EvalResult::Error(format!("deref failed: {e}")),
                        }
                    }
                    _ => EvalResult::Error("cannot dereference non-pointer".to_string()),
                }
            }

            Expr::AddrOf {
                mutability,
                expr: inner,
            } => {
                // Taking address - in a full impl, we'd allocate and store
                // For now, simplified: evaluate inner and wrap
                let inner_val = match self.eval(inner) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                // Allocate space and store the value
                let ty = inner_val.get_type();
                let size = ty.size().unwrap_or(8);
                match self.ctx.memory.allocate(size) {
                    Ok(addr) => {
                        // Store value (simplified - just store as u64 for now)
                        if let Some(n) = inner_val.as_u64() {
                            if let Err(e) = self.ctx.memory.write_u64(addr, n) {
                                return EvalResult::Error(format!("write failed: {e}"));
                            }
                        }
                        EvalResult::Value(Value::Reference {
                            addr,
                            mutability: *mutability,
                            lifetime: crate::types::Lifetime::Static,
                        })
                    }
                    Err(e) => EvalResult::Error(format!("allocation failed: {e}")),
                }
            }

            Expr::BinOp { op, left, right } => {
                let left_val = match self.eval(left) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                let right_val = match self.eval(right) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                match eval_binop(*op, &left_val, &right_val) {
                    Some(v) => EvalResult::Value(v),
                    None => EvalResult::Error(format!("binary op {op:?} failed")),
                }
            }

            Expr::UnOp { op, expr: inner } => {
                let inner_val = match self.eval(inner) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                match eval_unop(*op, &inner_val) {
                    Some(v) => EvalResult::Value(v),
                    None => EvalResult::Error(format!("unary op {op:?} failed")),
                }
            }

            Expr::Cast {
                expr: inner,
                target,
            } => {
                let inner_val = match self.eval(inner) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                match cast_value(&inner_val, target) {
                    Some(v) => EvalResult::Value(v),
                    None => EvalResult::Error("cast failed".to_string()),
                }
            }

            Expr::Call { func, args } => self.eval_call(func, args),

            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                // For now, treat method calls as function calls with receiver as first arg
                let recv_val = match self.eval(receiver) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                // Create function expression for method
                let func_expr = Expr::Var {
                    name: method.clone(),
                    local_idx: 0,
                };
                let mut all_args = vec![Expr::Literal(recv_val)];
                all_args.extend(args.iter().cloned());
                self.eval_call(&func_expr, &all_args)
            }

            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_val = match self.eval(condition) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                let Value::Bool(cond_bool) = cond_val else {
                    return EvalResult::Error("condition must be boolean".to_string());
                };
                if cond_bool {
                    self.eval(then_branch)
                } else {
                    match else_branch {
                        Some(e) => self.eval(e),
                        None => EvalResult::Value(Value::Unit),
                    }
                }
            }

            Expr::Match { scrutinee, arms } => {
                let scrutinee_val = match self.eval(scrutinee) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                self.eval_match(&scrutinee_val, arms)
            }

            Expr::Block { stmts, expr } => {
                self.push_scope();
                for stmt in stmts {
                    match self.exec_stmt(stmt) {
                        StmtResult::Ok => {}
                        StmtResult::Return(v) => {
                            self.pop_scope();
                            return EvalResult::Return(v);
                        }
                        StmtResult::Break(v) => {
                            self.pop_scope();
                            return EvalResult::Break(v);
                        }
                        StmtResult::Continue => {
                            self.pop_scope();
                            return EvalResult::Continue;
                        }
                        StmtResult::Error(e) => {
                            self.pop_scope();
                            return EvalResult::Error(e);
                        }
                    }
                }
                let result = match expr {
                    Some(e) => self.eval(e),
                    None => EvalResult::Value(Value::Unit),
                };
                self.pop_scope();
                result
            }

            Expr::Tuple(elems) => {
                let mut values = Vec::with_capacity(elems.len());
                for e in elems {
                    match self.eval(e) {
                        EvalResult::Value(v) => values.push(v),
                        other => return other,
                    }
                }
                EvalResult::Value(Value::Tuple(values))
            }

            Expr::Array(elems) => {
                let mut values = Vec::with_capacity(elems.len());
                for e in elems {
                    match self.eval(e) {
                        EvalResult::Value(v) => values.push(v),
                        other => return other,
                    }
                }
                EvalResult::Value(Value::Array(values))
            }

            Expr::ArrayRepeat { value, count } => {
                let val = match self.eval(value) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                EvalResult::Value(Value::Array(vec![val; *count]))
            }

            Expr::Struct { name, fields } => {
                let mut field_values = BTreeMap::new();
                for (field_name, field_expr) in fields {
                    match self.eval(field_expr) {
                        EvalResult::Value(v) => {
                            field_values.insert(field_name.clone(), v);
                        }
                        other => return other,
                    }
                }
                EvalResult::Value(Value::Struct {
                    name: name.clone(),
                    fields: field_values,
                })
            }

            Expr::EnumVariant {
                enum_name,
                variant,
                payload,
            } => {
                let enum_payload = match payload {
                    EnumVariantPayload::Unit => EnumPayload::Unit,
                    EnumVariantPayload::Tuple(exprs) => {
                        let mut values = Vec::with_capacity(exprs.len());
                        for e in exprs {
                            match self.eval(e) {
                                EvalResult::Value(v) => values.push(v),
                                other => return other,
                            }
                        }
                        EnumPayload::Tuple(values)
                    }
                    EnumVariantPayload::Struct(fields) => {
                        let mut field_values = BTreeMap::new();
                        for (name, expr) in fields {
                            match self.eval(expr) {
                                EvalResult::Value(v) => {
                                    field_values.insert(name.clone(), v);
                                }
                                other => return other,
                            }
                        }
                        EnumPayload::Struct(field_values)
                    }
                };
                EvalResult::Value(Value::Enum {
                    name: enum_name.clone(),
                    variant: variant.clone(),
                    payload: Box::new(enum_payload),
                })
            }

            Expr::Closure {
                params,
                body,
                captures,
            } => {
                // Capture the current environment
                let mut captured_values = Vec::new();
                for (name, _mutability) in captures {
                    if let Some(v) = self.lookup(name) {
                        captured_values.push((name.clone(), v.clone()));
                    }
                }
                // Create closure value with unique ID
                let closure_id = format!(
                    "closure_{}",
                    self.ctx
                        .memory
                        .allocate(0)
                        .map(|a| a.alloc_id.0)
                        .unwrap_or(0)
                );

                // Store the closure body as a function
                self.ctx.register_function(FunctionDef {
                    name: closure_id.clone(),
                    params: params.clone(),
                    ret_ty: RustType::Unit, // We'd need type inference for the real type
                    body: (**body).clone(),
                });

                EvalResult::Value(Value::Closure {
                    fn_id: closure_id,
                    captures: captured_values,
                })
            }

            Expr::Range {
                start,
                end,
                inclusive,
            } => {
                // Ranges are represented as structs
                let start_val = match start {
                    Some(e) => match self.eval(e) {
                        EvalResult::Value(v) => Some(v),
                        other => return other,
                    },
                    None => None,
                };
                let end_val = match end {
                    Some(e) => match self.eval(e) {
                        EvalResult::Value(v) => Some(v),
                        other => return other,
                    },
                    None => None,
                };
                // Return as tuple (start, end, inclusive)
                EvalResult::Value(Value::Tuple(vec![
                    start_val.unwrap_or(Value::Unit),
                    end_val.unwrap_or(Value::Unit),
                    Value::Bool(*inclusive),
                ]))
            }

            Expr::Return(opt_expr) => {
                let val = match opt_expr {
                    Some(e) => match self.eval(e) {
                        EvalResult::Value(v) => v,
                        other => return other,
                    },
                    None => Value::Unit,
                };
                EvalResult::Return(val)
            }

            Expr::Break { label: _, value } => {
                let val = match value {
                    Some(e) => match self.eval(e) {
                        EvalResult::Value(v) => Some(v),
                        other => return other,
                    },
                    None => None,
                };
                EvalResult::Break(val)
            }

            Expr::Continue { label: _ } => EvalResult::Continue,

            Expr::Loop { label: _, body } => {
                for _ in 0..MAX_LOOP_ITERATIONS {
                    match self.eval(body) {
                        EvalResult::Value(_) | EvalResult::Continue => continue,
                        EvalResult::Break(v) => return EvalResult::Value(v.unwrap_or(Value::Unit)),
                        other => return other,
                    }
                }
                EvalResult::Error("maximum loop iterations exceeded".to_string())
            }

            Expr::While {
                label: _,
                condition,
                body,
            } => {
                for _ in 0..MAX_LOOP_ITERATIONS {
                    let cond_val = match self.eval(condition) {
                        EvalResult::Value(v) => v,
                        other => return other,
                    };
                    let Value::Bool(cond_bool) = cond_val else {
                        return EvalResult::Error("while condition must be boolean".to_string());
                    };
                    if !cond_bool {
                        return EvalResult::Value(Value::Unit);
                    }
                    match self.eval(body) {
                        EvalResult::Value(_) | EvalResult::Continue => continue,
                        EvalResult::Break(v) => return EvalResult::Value(v.unwrap_or(Value::Unit)),
                        other => return other,
                    }
                }
                EvalResult::Error("maximum loop iterations exceeded".to_string())
            }

            Expr::For {
                label: _,
                pattern,
                iter,
                body,
            } => {
                // Simplified for loop - evaluate iter as array and iterate
                let iter_val = match self.eval(iter) {
                    EvalResult::Value(v) => v,
                    other => return other,
                };
                let elements = match iter_val {
                    Value::Array(elems) | Value::Tuple(elems) => elems,
                    // Range iteration (tuple of start, end, inclusive)
                    _ => return EvalResult::Error("for loop requires iterable".to_string()),
                };

                for (i, elem) in elements.into_iter().enumerate() {
                    if i >= MAX_LOOP_ITERATIONS {
                        return EvalResult::Error("maximum loop iterations exceeded".to_string());
                    }
                    self.push_scope();
                    if let Some(bindings) = match_pattern(pattern, &elem) {
                        self.apply_bindings(bindings);
                    }
                    match self.eval(body) {
                        EvalResult::Value(_) | EvalResult::Continue => {}
                        EvalResult::Break(v) => {
                            self.pop_scope();
                            return EvalResult::Value(v.unwrap_or(Value::Unit));
                        }
                        other => {
                            self.pop_scope();
                            return other;
                        }
                    }
                    self.pop_scope();
                }
                EvalResult::Value(Value::Unit)
            }
        }
    }

    /// Evaluate a function call
    fn eval_call(&mut self, func: &Expr, args: &[Expr]) -> EvalResult {
        // Evaluate function expression
        let func_val = match self.eval(func) {
            EvalResult::Value(v) => v,
            other => return other,
        };

        // Evaluate arguments
        let mut arg_values = Vec::with_capacity(args.len());
        for arg in args {
            match self.eval(arg) {
                EvalResult::Value(v) => arg_values.push(v),
                other => return other,
            }
        }

        // Dispatch based on function value type
        match func_val {
            Value::FnPtr { name } => self.call_function(&name, arg_values),
            Value::Closure { fn_id, captures } => {
                // Set up captures in new scope
                self.push_scope();
                for (name, value) in captures {
                    self.bind(name, value);
                }
                let result = self.call_function(&fn_id, arg_values);
                self.pop_scope();
                result
            }
            _ => {
                // Try to find function by name if func was a variable
                if let Expr::Var { name, .. } = func {
                    self.call_function(name, arg_values)
                } else {
                    EvalResult::Error("not a callable value".to_string())
                }
            }
        }
    }

    /// Call a named function
    fn call_function(&mut self, name: &str, args: Vec<Value>) -> EvalResult {
        // Look up function
        let func_def = match self.ctx.get_function(name) {
            Some(f) => f.clone(),
            None => return EvalResult::Error(format!("undefined function: {name}")),
        };

        // Check argument count
        if args.len() != func_def.params.len() {
            return EvalResult::Error(format!(
                "function {} expects {} args, got {}",
                name,
                func_def.params.len(),
                args.len()
            ));
        }

        // Push new scope and stack frame
        self.recursion_depth += 1;
        self.push_scope();
        self.ctx.stack.push_frame();

        // Bind parameters
        for ((param_name, _param_ty), arg_val) in func_def.params.iter().zip(args.into_iter()) {
            self.bind(param_name.clone(), arg_val);
        }

        // Execute function body
        let result = self.eval(&func_def.body);

        // Pop scope and frame
        self.ctx.stack.pop_frame();
        self.pop_scope();
        self.recursion_depth -= 1;

        // Convert return to value
        match result {
            EvalResult::Return(v) | EvalResult::Value(v) => EvalResult::Value(v),
            other => other,
        }
    }

    /// Evaluate a match expression
    fn eval_match(&mut self, scrutinee: &Value, arms: &[MatchArm]) -> EvalResult {
        for arm in arms {
            if let Some(bindings) = match_pattern(&arm.pattern, scrutinee) {
                // Check guard if present
                if let Some(guard) = &arm.guard {
                    self.push_scope();
                    self.apply_bindings(bindings.clone());
                    let guard_result = self.eval(guard);
                    self.pop_scope();

                    match guard_result {
                        EvalResult::Value(Value::Bool(true)) => {}
                        EvalResult::Value(Value::Bool(false)) => continue,
                        EvalResult::Value(_) => {
                            return EvalResult::Error("match guard must be boolean".to_string());
                        }
                        other => return other,
                    }
                }

                // Execute arm body
                self.push_scope();
                self.apply_bindings(bindings);
                let result = self.eval(&arm.body);
                self.pop_scope();
                return result;
            }
        }
        EvalResult::Error("non-exhaustive match".to_string())
    }

    /// Execute a statement
    pub fn exec_stmt(&mut self, stmt: &Stmt) -> StmtResult {
        match stmt {
            Stmt::Let {
                pattern,
                ty: _,
                init,
            } => {
                let init_val = match init {
                    Some(e) => match self.eval(e) {
                        EvalResult::Value(v) => v,
                        EvalResult::Return(v) => return StmtResult::Return(v),
                        EvalResult::Break(v) => return StmtResult::Break(v),
                        EvalResult::Continue => return StmtResult::Continue,
                        EvalResult::Error(e) => return StmtResult::Error(e),
                    },
                    None => Value::Uninit,
                };

                match match_pattern(pattern, &init_val) {
                    Some(bindings) => {
                        self.apply_bindings(bindings);
                        StmtResult::Ok
                    }
                    None => StmtResult::Error("pattern match failed in let".to_string()),
                }
            }

            Stmt::Expr(e) => match self.eval(e) {
                EvalResult::Value(_) => StmtResult::Ok,
                EvalResult::Return(v) => StmtResult::Return(v),
                EvalResult::Break(v) => StmtResult::Break(v),
                EvalResult::Continue => StmtResult::Continue,
                EvalResult::Error(e) => StmtResult::Error(e),
            },

            Stmt::Item(item) => {
                self.process_item(item);
                StmtResult::Ok
            }
        }
    }

    /// Process an item declaration
    fn process_item(&mut self, item: &Item) {
        match item {
            Item::Fn {
                name,
                params,
                ret,
                body,
            } => {
                self.ctx.register_function(FunctionDef {
                    name: name.clone(),
                    params: params.clone(),
                    ret_ty: ret.clone(),
                    body: body.clone(),
                });
                // Also bind function pointer in scope
                self.bind(name.clone(), Value::FnPtr { name: name.clone() });
            }
            Item::Struct { name, fields } => {
                self.ctx.register_type(crate::stmt::TypeDef::Struct {
                    name: name.clone(),
                    fields: fields.clone(),
                });
            }
            Item::Enum { name, variants } => {
                let variant_defs: Vec<_> = variants
                    .iter()
                    .map(|(vname, _payload)| crate::stmt::EnumVariantDef {
                        name: vname.clone(),
                        payload: crate::stmt::EnumVariantType::Unit, // Simplified
                    })
                    .collect();
                self.ctx.register_type(crate::stmt::TypeDef::Enum {
                    name: name.clone(),
                    variants: variant_defs,
                });
            }
            Item::Impl { self_ty: _, items } => {
                for sub_item in items {
                    self.process_item(sub_item);
                }
            }
            Item::Const { name, ty: _, value }
            | Item::Static {
                name,
                ty: _,
                mutable: _,
                value,
            } => {
                if let EvalResult::Value(v) = self.eval(value) {
                    self.bind(name.clone(), v);
                }
            }
        }
    }

    /// Run a program (list of items followed by optional main expression)
    pub fn run_program(&mut self, items: &[Item], main_expr: Option<&Expr>) -> EvalResult {
        // Process all items
        for item in items {
            self.process_item(item);
        }

        // If there's a main function and no explicit main expression, call main
        if main_expr.is_none() && self.ctx.get_function("main").is_some() {
            return self.call_function("main", vec![]);
        }

        // Evaluate main expression if provided
        match main_expr {
            Some(e) => self.eval(e),
            None => EvalResult::Value(Value::Unit),
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Pattern;
    use crate::types::UintType;
    use crate::values::BinOp;

    #[test]
    fn test_eval_literal() {
        let mut interp = Interpreter::new();
        let expr = Expr::Literal(Value::u32(42));
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::u32(42)));
    }

    #[test]
    fn test_eval_binop_add() {
        let mut interp = Interpreter::new();
        let expr = Expr::BinOp {
            op: BinOp::Add,
            left: Box::new(Expr::Literal(Value::u32(10))),
            right: Box::new(Expr::Literal(Value::u32(20))),
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::u32(30)));
    }

    #[test]
    fn test_eval_binop_compare() {
        let mut interp = Interpreter::new();
        let expr = Expr::BinOp {
            op: BinOp::Lt,
            left: Box::new(Expr::Literal(Value::i32(5))),
            right: Box::new(Expr::Literal(Value::i32(10))),
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::Bool(true)));
    }

    #[test]
    fn test_eval_if_true() {
        let mut interp = Interpreter::new();
        let expr = Expr::If {
            condition: Box::new(Expr::Literal(Value::Bool(true))),
            then_branch: Box::new(Expr::Literal(Value::u32(1))),
            else_branch: Some(Box::new(Expr::Literal(Value::u32(2)))),
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::u32(1)));
    }

    #[test]
    fn test_eval_if_false() {
        let mut interp = Interpreter::new();
        let expr = Expr::If {
            condition: Box::new(Expr::Literal(Value::Bool(false))),
            then_branch: Box::new(Expr::Literal(Value::u32(1))),
            else_branch: Some(Box::new(Expr::Literal(Value::u32(2)))),
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::u32(2)));
    }

    #[test]
    fn test_eval_let_binding() {
        let mut interp = Interpreter::new();
        let block = Expr::Block {
            stmts: vec![Stmt::Let {
                pattern: Pattern::Binding {
                    name: "x".to_string(),
                    mutable: false,
                    subpattern: None,
                },
                ty: None,
                init: Some(Expr::Literal(Value::u32(42))),
            }],
            expr: Some(Box::new(Expr::Var {
                name: "x".to_string(),
                local_idx: 0,
            })),
        };
        let result = interp.eval(&block);
        assert_eq!(result.value(), Some(Value::u32(42)));
    }

    #[test]
    fn test_eval_tuple() {
        let mut interp = Interpreter::new();
        let expr = Expr::Tuple(vec![
            Expr::Literal(Value::u32(1)),
            Expr::Literal(Value::Bool(true)),
        ]);
        let result = interp.eval(&expr);
        assert_eq!(
            result.value(),
            Some(Value::Tuple(vec![Value::u32(1), Value::Bool(true)]))
        );
    }

    #[test]
    fn test_eval_array() {
        let mut interp = Interpreter::new();
        let expr = Expr::Array(vec![
            Expr::Literal(Value::u32(1)),
            Expr::Literal(Value::u32(2)),
            Expr::Literal(Value::u32(3)),
        ]);
        let result = interp.eval(&expr);
        assert_eq!(
            result.value(),
            Some(Value::Array(vec![
                Value::u32(1),
                Value::u32(2),
                Value::u32(3)
            ]))
        );
    }

    #[test]
    fn test_eval_array_index() {
        let mut interp = Interpreter::new();
        let expr = Expr::Index {
            base: Box::new(Expr::Array(vec![
                Expr::Literal(Value::u32(10)),
                Expr::Literal(Value::u32(20)),
                Expr::Literal(Value::u32(30)),
            ])),
            index: Box::new(Expr::Literal(Value::u32(1))),
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::u32(20)));
    }

    #[test]
    fn test_eval_struct() {
        let mut interp = Interpreter::new();
        let expr = Expr::Struct {
            name: "Point".to_string(),
            fields: vec![
                ("x".to_string(), Expr::Literal(Value::f64(1.0))),
                ("y".to_string(), Expr::Literal(Value::f64(2.0))),
            ],
        };
        let result = interp.eval(&expr);
        match result {
            EvalResult::Value(Value::Struct { name, fields }) => {
                assert_eq!(name, "Point");
                assert_eq!(fields.get("x"), Some(&Value::f64(1.0)));
                assert_eq!(fields.get("y"), Some(&Value::f64(2.0)));
            }
            _ => panic!("expected struct value"),
        }
    }

    #[test]
    fn test_eval_field_access() {
        let mut interp = Interpreter::new();
        let expr = Expr::Field {
            base: Box::new(Expr::Struct {
                name: "Point".to_string(),
                fields: vec![
                    ("x".to_string(), Expr::Literal(Value::f64(1.0))),
                    ("y".to_string(), Expr::Literal(Value::f64(2.0))),
                ],
            }),
            field: "y".to_string(),
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::f64(2.0)));
    }

    #[test]
    fn test_eval_enum_variant() {
        let mut interp = Interpreter::new();
        let expr = Expr::EnumVariant {
            enum_name: "Option".to_string(),
            variant: "Some".to_string(),
            payload: EnumVariantPayload::Tuple(vec![Expr::Literal(Value::u32(42))]),
        };
        let result = interp.eval(&expr);
        match result {
            EvalResult::Value(Value::Enum {
                name,
                variant,
                payload,
            }) => {
                assert_eq!(name, "Option");
                assert_eq!(variant, "Some");
                match *payload {
                    EnumPayload::Tuple(v) => assert_eq!(v, vec![Value::u32(42)]),
                    _ => panic!("expected tuple payload"),
                }
            }
            _ => panic!("expected enum value"),
        }
    }

    #[test]
    fn test_eval_match() {
        let mut interp = Interpreter::new();
        let expr = Expr::Match {
            scrutinee: Box::new(Expr::Literal(Value::u32(2))),
            arms: vec![
                MatchArm {
                    pattern: Pattern::Literal(Value::u32(1)),
                    guard: None,
                    body: Expr::Literal(Value::Bool(false)),
                },
                MatchArm {
                    pattern: Pattern::Literal(Value::u32(2)),
                    guard: None,
                    body: Expr::Literal(Value::Bool(true)),
                },
                MatchArm {
                    pattern: Pattern::Wildcard,
                    guard: None,
                    body: Expr::Literal(Value::Bool(false)),
                },
            ],
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::Bool(true)));
    }

    #[test]
    fn test_eval_match_binding() {
        let mut interp = Interpreter::new();
        let expr = Expr::Match {
            scrutinee: Box::new(Expr::EnumVariant {
                enum_name: "Option".to_string(),
                variant: "Some".to_string(),
                payload: EnumVariantPayload::Tuple(vec![Expr::Literal(Value::u32(42))]),
            }),
            arms: vec![
                MatchArm {
                    pattern: Pattern::EnumVariant {
                        enum_name: "Option".to_string(),
                        variant: "Some".to_string(),
                        payload: crate::expr::EnumPatternPayload::Tuple(vec![Pattern::Binding {
                            name: "x".to_string(),
                            mutable: false,
                            subpattern: None,
                        }]),
                    },
                    guard: None,
                    body: Expr::Var {
                        name: "x".to_string(),
                        local_idx: 0,
                    },
                },
                MatchArm {
                    pattern: Pattern::EnumVariant {
                        enum_name: "Option".to_string(),
                        variant: "None".to_string(),
                        payload: crate::expr::EnumPatternPayload::Unit,
                    },
                    guard: None,
                    body: Expr::Literal(Value::u32(0)),
                },
            ],
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::u32(42)));
    }

    #[test]
    fn test_eval_while_loop() {
        let mut interp = Interpreter::new();
        // Simple while loop: while true { break }
        let expr = Expr::While {
            label: None,
            condition: Box::new(Expr::Literal(Value::Bool(true))),
            body: Box::new(Expr::Break {
                label: None,
                value: Some(Box::new(Expr::Literal(Value::u32(42)))),
            }),
        };
        let result = interp.eval(&expr);
        // break 42 exits with value 42
        assert_eq!(result.value(), Some(Value::u32(42)));
    }

    #[test]
    fn test_eval_while_with_condition() {
        let mut interp = Interpreter::new();
        // while false { ... } should return unit immediately
        let expr = Expr::While {
            label: None,
            condition: Box::new(Expr::Literal(Value::Bool(false))),
            body: Box::new(Expr::Literal(Value::u32(999))),
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::Unit));
    }

    #[test]
    fn test_eval_function_call() {
        let mut interp = Interpreter::new();

        // Define a simple add function
        interp.ctx.register_function(FunctionDef {
            name: "add".to_string(),
            params: vec![
                ("a".to_string(), RustType::Uint(UintType::U32)),
                ("b".to_string(), RustType::Uint(UintType::U32)),
            ],
            ret_ty: RustType::Uint(UintType::U32),
            body: Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(Expr::Var {
                    name: "a".to_string(),
                    local_idx: 0,
                }),
                right: Box::new(Expr::Var {
                    name: "b".to_string(),
                    local_idx: 1,
                }),
            },
        });

        // Call the function
        let expr = Expr::Call {
            func: Box::new(Expr::Var {
                name: "add".to_string(),
                local_idx: 0,
            }),
            args: vec![Expr::Literal(Value::u32(10)), Expr::Literal(Value::u32(20))],
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::u32(30)));
    }

    #[test]
    fn test_eval_recursive_function() {
        let mut interp = Interpreter::new();

        // Define factorial function
        // fn factorial(n: u32) -> u32 {
        //     if n <= 1 { 1 } else { n * factorial(n - 1) }
        // }
        interp.ctx.register_function(FunctionDef {
            name: "factorial".to_string(),
            params: vec![("n".to_string(), RustType::Uint(UintType::U32))],
            ret_ty: RustType::Uint(UintType::U32),
            body: Expr::If {
                condition: Box::new(Expr::BinOp {
                    op: BinOp::Le,
                    left: Box::new(Expr::Var {
                        name: "n".to_string(),
                        local_idx: 0,
                    }),
                    right: Box::new(Expr::Literal(Value::u32(1))),
                }),
                then_branch: Box::new(Expr::Literal(Value::u32(1))),
                else_branch: Some(Box::new(Expr::BinOp {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Var {
                        name: "n".to_string(),
                        local_idx: 0,
                    }),
                    right: Box::new(Expr::Call {
                        func: Box::new(Expr::Var {
                            name: "factorial".to_string(),
                            local_idx: 0,
                        }),
                        args: vec![Expr::BinOp {
                            op: BinOp::Sub,
                            left: Box::new(Expr::Var {
                                name: "n".to_string(),
                                local_idx: 0,
                            }),
                            right: Box::new(Expr::Literal(Value::u32(1))),
                        }],
                    }),
                })),
            },
        });

        // factorial(5) = 120
        let expr = Expr::Call {
            func: Box::new(Expr::Var {
                name: "factorial".to_string(),
                local_idx: 0,
            }),
            args: vec![Expr::Literal(Value::u32(5))],
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::u32(120)));
    }

    #[test]
    fn test_eval_return() {
        let mut interp = Interpreter::new();

        // Function with early return
        interp.ctx.register_function(FunctionDef {
            name: "early_return".to_string(),
            params: vec![("n".to_string(), RustType::Uint(UintType::U32))],
            ret_ty: RustType::Uint(UintType::U32),
            body: Expr::Block {
                stmts: vec![Stmt::Expr(Expr::If {
                    condition: Box::new(Expr::BinOp {
                        op: BinOp::Eq,
                        left: Box::new(Expr::Var {
                            name: "n".to_string(),
                            local_idx: 0,
                        }),
                        right: Box::new(Expr::Literal(Value::u32(0))),
                    }),
                    then_branch: Box::new(Expr::Return(Some(Box::new(Expr::Literal(Value::u32(
                        999,
                    )))))),
                    else_branch: None,
                })],
                expr: Some(Box::new(Expr::Var {
                    name: "n".to_string(),
                    local_idx: 0,
                })),
            },
        });

        // early_return(0) should return 999
        let expr1 = Expr::Call {
            func: Box::new(Expr::Var {
                name: "early_return".to_string(),
                local_idx: 0,
            }),
            args: vec![Expr::Literal(Value::u32(0))],
        };
        assert_eq!(interp.eval(&expr1).value(), Some(Value::u32(999)));

        // early_return(5) should return 5
        let expr2 = Expr::Call {
            func: Box::new(Expr::Var {
                name: "early_return".to_string(),
                local_idx: 0,
            }),
            args: vec![Expr::Literal(Value::u32(5))],
        };
        assert_eq!(interp.eval(&expr2).value(), Some(Value::u32(5)));
    }

    #[test]
    fn test_eval_break() {
        let mut interp = Interpreter::new();
        let expr = Expr::Loop {
            label: None,
            body: Box::new(Expr::Break {
                label: None,
                value: Some(Box::new(Expr::Literal(Value::u32(42)))),
            }),
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::u32(42)));
    }

    #[test]
    fn test_eval_array_repeat() {
        let mut interp = Interpreter::new();
        let expr = Expr::ArrayRepeat {
            value: Box::new(Expr::Literal(Value::u32(7))),
            count: 4,
        };
        let result = interp.eval(&expr);
        assert_eq!(
            result.value(),
            Some(Value::Array(vec![
                Value::u32(7),
                Value::u32(7),
                Value::u32(7),
                Value::u32(7)
            ]))
        );
    }

    #[test]
    fn test_eval_tuple_field_access() {
        let mut interp = Interpreter::new();
        let expr = Expr::Field {
            base: Box::new(Expr::Tuple(vec![
                Expr::Literal(Value::u32(10)),
                Expr::Literal(Value::u32(20)),
                Expr::Literal(Value::u32(30)),
            ])),
            field: "1".to_string(),
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::u32(20)));
    }

    #[test]
    fn test_eval_nested_scope() {
        let mut interp = Interpreter::new();
        // { let x = 1; { let x = 2; x } + x }
        // Inner block should see x=2, outer should see x=1
        let expr = Expr::Block {
            stmts: vec![Stmt::Let {
                pattern: Pattern::Binding {
                    name: "x".to_string(),
                    mutable: false,
                    subpattern: None,
                },
                ty: None,
                init: Some(Expr::Literal(Value::u32(1))),
            }],
            expr: Some(Box::new(Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(Expr::Block {
                    stmts: vec![Stmt::Let {
                        pattern: Pattern::Binding {
                            name: "x".to_string(),
                            mutable: false,
                            subpattern: None,
                        },
                        ty: None,
                        init: Some(Expr::Literal(Value::u32(2))),
                    }],
                    expr: Some(Box::new(Expr::Var {
                        name: "x".to_string(),
                        local_idx: 0,
                    })),
                }),
                right: Box::new(Expr::Var {
                    name: "x".to_string(),
                    local_idx: 0,
                }),
            })),
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::u32(3))); // 2 + 1 = 3
    }

    #[test]
    fn test_eval_for_loop() {
        let mut interp = Interpreter::new();
        // for x in [1, 2, 3] { sum = sum + x }; sum
        let block = Expr::Block {
            stmts: vec![
                Stmt::Let {
                    pattern: Pattern::Binding {
                        name: "sum".to_string(),
                        mutable: true,
                        subpattern: None,
                    },
                    ty: None,
                    init: Some(Expr::Literal(Value::u32(0))),
                },
                Stmt::Expr(Expr::For {
                    label: None,
                    pattern: Pattern::Binding {
                        name: "x".to_string(),
                        mutable: false,
                        subpattern: None,
                    },
                    iter: Box::new(Expr::Array(vec![
                        Expr::Literal(Value::u32(1)),
                        Expr::Literal(Value::u32(2)),
                        Expr::Literal(Value::u32(3)),
                    ])),
                    body: Box::new(Expr::Block {
                        stmts: vec![Stmt::Let {
                            pattern: Pattern::Binding {
                                name: "sum".to_string(),
                                mutable: true,
                                subpattern: None,
                            },
                            ty: None,
                            init: Some(Expr::BinOp {
                                op: BinOp::Add,
                                left: Box::new(Expr::Var {
                                    name: "sum".to_string(),
                                    local_idx: 0,
                                }),
                                right: Box::new(Expr::Var {
                                    name: "x".to_string(),
                                    local_idx: 0,
                                }),
                            }),
                        }],
                        expr: None,
                    }),
                }),
            ],
            expr: Some(Box::new(Expr::Var {
                name: "sum".to_string(),
                local_idx: 0,
            })),
        };
        let result = interp.eval(&block);
        // With shadowing: sum gets shadowed in each iteration
        // Final outer sum is still 0
        assert_eq!(result.value(), Some(Value::u32(0)));
    }

    #[test]
    fn test_eval_cast() {
        let mut interp = Interpreter::new();
        let expr = Expr::Cast {
            expr: Box::new(Expr::Literal(Value::Bool(true))),
            target: RustType::Uint(UintType::U32),
        };
        let result = interp.eval(&expr);
        assert_eq!(result.value(), Some(Value::u32(1)));
    }

    #[test]
    fn test_eval_unop() {
        let mut interp = Interpreter::new();

        // Not
        let expr1 = Expr::UnOp {
            op: crate::values::UnOp::Not,
            expr: Box::new(Expr::Literal(Value::Bool(true))),
        };
        assert_eq!(interp.eval(&expr1).value(), Some(Value::Bool(false)));

        // Neg
        let expr2 = Expr::UnOp {
            op: crate::values::UnOp::Neg,
            expr: Box::new(Expr::Literal(Value::i32(42))),
        };
        assert_eq!(interp.eval(&expr2).value(), Some(Value::i32(-42)));
    }

    #[test]
    fn test_match_with_guard() {
        let mut interp = Interpreter::new();
        let expr = Expr::Match {
            scrutinee: Box::new(Expr::Literal(Value::u32(5))),
            arms: vec![
                MatchArm {
                    pattern: Pattern::Binding {
                        name: "x".to_string(),
                        mutable: false,
                        subpattern: None,
                    },
                    guard: Some(Expr::BinOp {
                        op: BinOp::Lt,
                        left: Box::new(Expr::Var {
                            name: "x".to_string(),
                            local_idx: 0,
                        }),
                        right: Box::new(Expr::Literal(Value::u32(3))),
                    }),
                    body: Expr::Literal(Value::Bool(false)),
                },
                MatchArm {
                    pattern: Pattern::Binding {
                        name: "x".to_string(),
                        mutable: false,
                        subpattern: None,
                    },
                    guard: None,
                    body: Expr::Literal(Value::Bool(true)),
                },
            ],
        };
        let result = interp.eval(&expr);
        // 5 >= 3, so first arm guard fails, second arm matches
        assert_eq!(result.value(), Some(Value::Bool(true)));
    }
}
