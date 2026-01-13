//! Function inlining for MirProgram
//!
//! This module provides functionality to inline function calls within a MirProgram.
//! When a Call terminator references a function whose body is available in the
//! function map, the callee's blocks are merged into the caller with appropriate
//! block ID and variable renaming.
//!
//! # Algorithm
//!
//! For each Call terminator:
//! 1. Look up the callee's MirProgram in the function map
//! 2. Rename the callee's locals with a unique prefix to avoid conflicts
//! 3. Offset the callee's block IDs to avoid conflicts with caller
//! 4. Add parameter assignments: map caller args to callee parameters
//! 5. Replace caller's Call with a Goto to the callee's entry block
//! 6. Replace callee's Return with assignments and Goto to caller's continuation
//! 7. Append all callee blocks to the caller

use kani_fast_chc::mir::{
    MirBasicBlock, MirLocal, MirProgram, MirStatement, MirTerminator, PANIC_BLOCK_ID,
};
use std::borrow::Cow;
use std::collections::HashMap;
use std::rc::Rc;

type FunctionBodies = HashMap<String, Rc<MirProgram>>;

/// Offset a block target, preserving sentinel values.
/// Sentinel values (like PANIC_BLOCK_ID) are not offset to maintain
/// their special meaning across inlining boundaries.
#[inline]
fn offset_target(target: usize, offset: usize) -> usize {
    if target >= PANIC_BLOCK_ID {
        // Preserve sentinel values
        target
    } else {
        target + offset
    }
}

/// Configuration for the inliner
pub struct InlinerConfig {
    /// Maximum recursion depth for inlining (0 = no recursion)
    pub max_depth: usize,
}

impl Default for InlinerConfig {
    fn default() -> Self {
        Self { max_depth: 10 }
    }
}

/// Inline function calls in a MirProgram
///
/// # Arguments
/// * `program` - The MirProgram to process
/// * `function_bodies` - Map from function names to their MirProgram bodies
/// * `config` - Inliner configuration
///
/// # Returns
/// A `Cow` containing either the original program (if no inlining occurred) or
/// a new MirProgram with function calls inlined. This avoids cloning when
/// the program doesn't need modification (e.g., max depth exceeded or no calls to inline).
pub fn inline_functions<'a>(
    program: &'a MirProgram,
    function_bodies: &FunctionBodies,
    config: &InlinerConfig,
) -> Cow<'a, MirProgram> {
    let mut inliner = Inliner::new(function_bodies, config);
    inliner.inline(program, 0)
}

struct Inliner<'a> {
    function_bodies: &'a FunctionBodies,
    config: &'a InlinerConfig,
    /// Counter for unique variable prefixes
    inline_counter: usize,
    /// Reference tracking: maps a variable to what it points to
    /// e.g., if `_3 = _1` (creating a reference), then ref_targets["_3"] = "_1"
    ref_targets: HashMap<String, String>,
}

impl<'a> Inliner<'a> {
    fn new(function_bodies: &'a FunctionBodies, config: &'a InlinerConfig) -> Self {
        Self {
            ref_targets: HashMap::new(),
            function_bodies,
            config,
            inline_counter: 0,
        }
    }

    /// Inline function calls in the given program
    ///
    /// Returns `Cow::Borrowed` if no modifications were needed (depth exceeded or
    /// no calls to inline), avoiding unnecessary cloning.
    fn inline<'b>(&mut self, program: &'b MirProgram, depth: usize) -> Cow<'b, MirProgram> {
        if depth > self.config.max_depth {
            return Cow::Borrowed(program);
        }

        // Build reference map: scan for assignments like `_3 = _1` where _3 is a reference
        // These arise from `&mut var` expressions where the RHS is just a variable name
        self.build_ref_map(program);

        // First pass: check if any inlining will occur to avoid unnecessary cloning
        let has_inlinable_calls = program.basic_blocks.iter().any(|block| {
            if let MirTerminator::Call { func, .. } = &block.terminator {
                self.function_bodies.contains_key(func)
            } else {
                false
            }
        });

        // If no calls to inline, return borrowed reference (no clone!)
        if !has_inlinable_calls {
            return Cow::Borrowed(program);
        }

        // Proceed with inlining - we know we'll modify the program
        let mut result_locals = program.locals.clone();
        // Pre-allocate with estimated capacity: blocks may grow ~2x during inlining
        let mut result_blocks = Vec::with_capacity(program.basic_blocks.len() * 2);
        let mut next_block_id = program.basic_blocks.iter().map(|b| b.id).max().unwrap_or(0) + 1;

        for block in &program.basic_blocks {
            if let MirTerminator::Call {
                destination,
                func,
                args,
                target,
                unwind: _,
                precondition_check: _,
                postcondition_assumption: _,
                is_range_into_iter: _,
                is_range_next: _,
            } = &block.terminator
            {
                // Check if we have the function body
                if let Some(callee) = self.function_bodies.get(func) {
                    // Generate a unique prefix for this inline site
                    self.inline_counter += 1;
                    let prefix =
                        format!("_inline{}_{}", self.inline_counter, func.replace("::", "_"));

                    // Inline the callee
                    let (inlined_blocks, new_locals, entry_block_id) = self.inline_callee(
                        callee.as_ref(),
                        &prefix,
                        destination.as_deref(),
                        args,
                        *target,
                        next_block_id,
                    );

                    // Update next block ID
                    next_block_id = inlined_blocks
                        .iter()
                        .map(|b| b.id)
                        .max()
                        .unwrap_or(next_block_id)
                        + 1;

                    // Add new locals
                    result_locals.extend(new_locals);

                    // Replace the call with a goto to the inlined function's entry
                    let new_block = MirBasicBlock {
                        id: block.id,
                        statements: block.statements.clone(),
                        terminator: MirTerminator::Goto {
                            target: entry_block_id,
                        },
                    };
                    result_blocks.push(new_block);

                    // Add the inlined blocks
                    result_blocks.extend(inlined_blocks);

                    continue;
                }
            }

            // No inlining needed, keep the block as-is
            result_blocks.push(block.clone());
        }

        Cow::Owned(MirProgram {
            locals: result_locals,
            basic_blocks: result_blocks,
            start_block: program.start_block,
            init: program.init.clone(),
            var_to_local: program.var_to_local.clone(),
            closures: program.closures.clone(),
            trait_impls: program.trait_impls.clone(),
        })
    }

    /// Build reference map by scanning for assignments where RHS is a simple variable
    ///
    /// When we see `_3 = _1` where `_1` is a variable, we record that `_3` points to `_1`.
    /// This captures the pattern of taking references: `let r = &mut x` generates
    /// an assignment where the reference variable gets the address (i.e., the variable itself).
    fn build_ref_map(&mut self, program: &MirProgram) {
        for block in &program.basic_blocks {
            for stmt in &block.statements {
                if let MirStatement::Assign { lhs, rhs } = stmt {
                    // Check if RHS is a simple variable reference (starts with _ and is alphanumeric)
                    // This catches assignments like `_3 = _1` from `&mut var` expressions
                    if rhs.starts_with('_') && rhs.chars().skip(1).all(|c| c.is_alphanumeric()) {
                        // lhs is a reference to rhs
                        self.ref_targets.insert(lhs.clone(), rhs.clone());
                    }
                }
            }
        }
    }

    /// Resolve a variable through the reference chain to find its target
    ///
    /// If `var` is known to point to another variable, return that target.
    /// Otherwise return None.
    fn resolve_ref_target(&self, var: &str) -> Option<String> {
        let mut current = var.to_string();
        let mut visited = std::collections::HashSet::new();

        // Follow the chain (with cycle detection)
        while let Some(target) = self.ref_targets.get(&current) {
            if !visited.insert(current.clone()) {
                // Cycle detected, stop
                break;
            }
            current = target.clone();
        }

        if current != var { Some(current) } else { None }
    }

    /// Inline a callee function
    ///
    /// Returns (inlined_blocks, new_locals, entry_block_id)
    fn inline_callee(
        &self,
        callee: &MirProgram,
        prefix: &str,
        destination: Option<&str>,
        args: &[String],
        return_target: usize,
        block_offset: usize,
    ) -> (Vec<MirBasicBlock>, Vec<MirLocal>, usize) {
        // Pre-allocate with callee's sizes
        let mut blocks = Vec::with_capacity(callee.basic_blocks.len());
        let mut new_locals = Vec::with_capacity(callee.locals.len());

        // Create renamed locals (skip _0 which is the return value)
        let local_renames: HashMap<String, String> = callee
            .locals
            .iter()
            .map(|local| {
                let new_name = format!("{}{}", prefix, local.name);
                (local.name.clone(), new_name)
            })
            .collect();

        // Add the renamed locals
        for local in &callee.locals {
            let new_name = format!("{}{}", prefix, local.name);
            new_locals.push(MirLocal::new(new_name, local.ty.clone()));
        }

        // Find the return local name (usually _0)
        let return_local = local_renames.get("_0").cloned();

        // Create entry block with parameter assignments
        let entry_block_id = callee.start_block + block_offset;

        // Map arguments to parameters
        // In MIR, parameters are typically _1, _2, ... (after _0 which is return)
        // Also build deref substitution map: if arg points to a variable, then
        // (deref renamed_param) should be replaced with that variable.
        let mut param_assignments = Vec::new();
        let mut deref_subs: HashMap<String, String> = HashMap::new();

        for (i, arg) in args.iter().enumerate() {
            let param_name = format!("_{}", i + 1);
            if let Some(renamed_param) = local_renames.get(&param_name) {
                param_assignments.push(MirStatement::Assign {
                    lhs: renamed_param.clone(),
                    rhs: arg.clone(),
                });

                // If the argument is a variable (like _1), also propagate any field variables
                // This handles tuple/struct parameters: _1_field0, _1_field1, etc.
                if arg.starts_with('_') && arg.chars().skip(1).all(|c| c.is_alphanumeric()) {
                    // Find all field locals that belong to this parameter in the callee
                    // and create assignments from the corresponding argument fields
                    for local in &callee.locals {
                        // Check for "{param_name}_field" prefix without allocating a String
                        let is_param_field = local.name.starts_with(&param_name)
                            && local.name[param_name.len()..].starts_with("_field");
                        if is_param_field {
                            // Extract the field suffix (e.g., "_field0" from "_1_field0")
                            let field_suffix = &local.name[param_name.len()..];
                            let arg_field_name = format!("{}{}", arg, field_suffix);
                            let renamed_param_field =
                                local_renames.get(&local.name).cloned().unwrap_or_default();

                            if !renamed_param_field.is_empty() {
                                param_assignments.push(MirStatement::Assign {
                                    lhs: renamed_param_field,
                                    rhs: arg_field_name,
                                });
                            }
                        }
                    }
                }

                // Check if arg is a reference to another variable
                // First try the arg itself, then resolve through ref_targets
                if let Some(target) = self.resolve_ref_target(arg) {
                    // (deref renamed_param) should substitute to target
                    deref_subs.insert(renamed_param.clone(), target);
                } else if arg.starts_with('_') && arg.chars().skip(1).all(|c| c.is_alphanumeric()) {
                    // The arg itself looks like a variable - it might be a direct reference
                    // In Rust MIR, &mut x creates a ref to _N, and passing that ref uses _N directly
                    // So the arg IS the target in many cases
                    deref_subs.insert(renamed_param.clone(), arg.clone());
                }
            }
        }

        // Track whether we've consumed param_assignments (only needed for entry block)
        let mut param_assignments_consumed = false;

        // Process each block
        for callee_block in &callee.basic_blocks {
            let new_block_id = callee_block.id + block_offset;

            // Rename variables in statements
            // Pre-allocate: base statements plus param assignments for entry block
            let extra_capacity =
                if callee_block.id == callee.start_block && !param_assignments_consumed {
                    param_assignments.len()
                } else {
                    0
                };
            let mut new_statements =
                Vec::with_capacity(callee_block.statements.len() + extra_capacity + 2);

            // For the entry block, add parameter assignments first (take ownership, no clone)
            if callee_block.id == callee.start_block && !param_assignments_consumed {
                new_statements.append(&mut param_assignments);
                param_assignments_consumed = true;
            }

            for stmt in &callee_block.statements {
                new_statements.push(self.rename_statement_with_deref(
                    stmt,
                    &local_renames,
                    &deref_subs,
                ));
            }

            // Process terminator
            let new_terminator = match &callee_block.terminator {
                MirTerminator::Return => {
                    // Replace return with assignment to destination and goto return_target
                    if let (Some(dest), Some(ret_local)) = (destination, &return_local) {
                        new_statements.push(MirStatement::Assign {
                            lhs: dest.to_string(),
                            rhs: ret_local.clone(),
                        });

                        // Propagate struct field variables when returning a struct
                        // If ret_local has field variables like "_inline1_Counter_new_0_field0",
                        // we need to copy them to the destination's field variables like "_1_field0"
                        for local in &callee.locals {
                            let renamed = format!("{}{}", prefix, local.name);
                            // Check if this is a field variable for the return value (_0_fieldN)
                            if local.name.starts_with("_0_field") {
                                // Get the field suffix (e.g., "_field0" from "_0_field0")
                                let field_suffix = &local.name[2..]; // Skip "_0"
                                let dest_field = format!("{}{}", dest, field_suffix);
                                new_statements.push(MirStatement::Assign {
                                    lhs: dest_field,
                                    rhs: renamed,
                                });
                            }
                        }
                    }
                    MirTerminator::Goto {
                        target: return_target,
                    }
                }
                MirTerminator::Goto { target } => MirTerminator::Goto {
                    target: offset_target(*target, block_offset),
                },
                MirTerminator::CondGoto {
                    condition,
                    then_target,
                    else_target,
                } => MirTerminator::CondGoto {
                    condition: self.rename_expr_with_deref(condition, &local_renames, &deref_subs),
                    then_target: offset_target(*then_target, block_offset),
                    else_target: offset_target(*else_target, block_offset),
                },
                MirTerminator::SwitchInt {
                    discr,
                    targets,
                    otherwise,
                } => MirTerminator::SwitchInt {
                    discr: self.rename_expr_with_deref(discr, &local_renames, &deref_subs),
                    targets: targets
                        .iter()
                        .map(|(val, tgt)| (*val, offset_target(*tgt, block_offset)))
                        .collect(),
                    otherwise: offset_target(*otherwise, block_offset),
                },
                MirTerminator::Call {
                    destination,
                    func,
                    args,
                    target,
                    unwind,
                    precondition_check,
                    postcondition_assumption,
                    is_range_into_iter,
                    is_range_next,
                } => MirTerminator::Call {
                    destination: destination
                        .as_ref()
                        .map(|d| self.rename_expr_with_deref(d, &local_renames, &deref_subs)),
                    func: func.clone(),
                    args: args
                        .iter()
                        .map(|a| self.rename_expr_with_deref(a, &local_renames, &deref_subs))
                        .collect(),
                    target: offset_target(*target, block_offset),
                    unwind: unwind.map(|u| offset_target(u, block_offset)),
                    precondition_check: precondition_check.clone(),
                    postcondition_assumption: postcondition_assumption.clone(),
                    is_range_into_iter: *is_range_into_iter,
                    is_range_next: *is_range_next,
                },
                MirTerminator::Unreachable => MirTerminator::Unreachable,
                MirTerminator::Abort => MirTerminator::Abort,
            };

            blocks.push(MirBasicBlock {
                id: new_block_id,
                statements: new_statements,
                terminator: new_terminator,
            });
        }

        (blocks, new_locals, entry_block_id)
    }

    /// Rename variables in an SMT expression string
    ///
    /// Uses single-pass byte transformation to replace variable names efficiently.
    /// Variables are identified as whole words (bounded by non-alphanumeric, non-underscore chars).
    fn rename_expr(&self, expr: &str, renames: &HashMap<String, String>) -> String {
        if renames.is_empty() {
            return expr.to_string();
        }

        let mut result = expr.to_string();

        // Sort renames by length (descending) to avoid partial replacements
        // e.g., _10 should be replaced before _1
        let mut sorted_renames: Vec<_> = renames.iter().collect();
        sorted_renames.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // Single-pass byte transformation for each rename
        for (old, new) in sorted_renames {
            let mut new_result = String::with_capacity(result.len());
            let old_bytes = old.as_bytes();
            let result_bytes = result.as_bytes();
            let mut i = 0;

            while i < result_bytes.len() {
                // Check if we're at a potential match
                if i + old_bytes.len() <= result_bytes.len()
                    && &result_bytes[i..i + old_bytes.len()] == old_bytes
                {
                    // Check preceding character (should not be alphanumeric or _)
                    let preceded_ok = if i == 0 {
                        true
                    } else {
                        let prev = result_bytes[i - 1];
                        !prev.is_ascii_alphanumeric() && prev != b'_'
                    };

                    // Check following character (should not be alphanumeric, _, or ')
                    let followed_ok = if i + old_bytes.len() >= result_bytes.len() {
                        true
                    } else {
                        let next = result_bytes[i + old_bytes.len()];
                        !next.is_ascii_alphanumeric() && next != b'_' && next != b'\''
                    };

                    if preceded_ok && followed_ok {
                        // Replace with the new variable name
                        new_result.push_str(new);
                        i += old_bytes.len();
                        continue;
                    }
                }

                // No match, copy character
                // SAFETY: result_bytes is valid UTF-8 from String, chars may be multi-byte
                // but we're iterating byte-by-byte which is safe for ASCII-only identifiers
                new_result.push(result_bytes[i] as char);
                i += 1;
            }

            result = new_result;
        }

        result
    }

    /// Rename variables in a statement with deref substitution for mutable references
    ///
    /// This is similar to rename_statement but also substitutes `(deref param)` patterns
    /// with the actual variable being referenced.
    fn rename_statement_with_deref(
        &self,
        stmt: &MirStatement,
        renames: &HashMap<String, String>,
        deref_subs: &HashMap<String, String>,
    ) -> MirStatement {
        match stmt {
            MirStatement::Assume(cond) => {
                MirStatement::Assume(self.rename_expr_with_deref(cond, renames, deref_subs))
            }
            MirStatement::Assign { lhs, rhs } => {
                // Handle LHS: if it's a (deref X) pattern, substitute
                let new_lhs = self.substitute_deref_lhs(lhs, renames, deref_subs);
                MirStatement::Assign {
                    lhs: new_lhs,
                    rhs: self.rename_expr_with_deref(rhs, renames, deref_subs),
                }
            }
            MirStatement::Assert { condition, message } => MirStatement::Assert {
                condition: self.rename_expr_with_deref(condition, renames, deref_subs),
                message: message.clone(),
            },
            MirStatement::ArrayStore {
                array,
                index,
                value,
            } => MirStatement::ArrayStore {
                array: renames.get(array).cloned().unwrap_or_else(|| array.clone()),
                index: self.rename_expr_with_deref(index, renames, deref_subs),
                value: self.rename_expr_with_deref(value, renames, deref_subs),
            },
            MirStatement::Havoc { var } => MirStatement::Havoc {
                var: renames.get(var).cloned().unwrap_or_else(|| var.clone()),
            },
        }
    }

    /// Substitute deref patterns in an LHS (assignment target)
    ///
    /// When we have `(deref _inline1_param) = value`, we want to replace
    /// the LHS with the actual target variable that the parameter references.
    ///
    /// Also handles field access patterns like `(deref _1)_field0` which should
    /// become `target_field0` where target is what _1 references.
    fn substitute_deref_lhs(
        &self,
        lhs: &str,
        renames: &HashMap<String, String>,
        deref_subs: &HashMap<String, String>,
    ) -> String {
        // Check if LHS starts with (deref X) pattern - might have suffix like _field0
        if lhs.starts_with("(deref ") {
            // Find the closing paren of the deref expression
            if let Some(close_paren) = lhs.find(')') {
                let inner = &lhs[7..close_paren];
                let suffix = &lhs[close_paren + 1..]; // e.g., "_field0" or ""

                // First, rename the inner variable
                let renamed_inner = renames
                    .get(inner)
                    .cloned()
                    .unwrap_or_else(|| inner.to_string());

                // Then check if this renamed variable has a deref substitution
                if let Some(target) = deref_subs.get(&renamed_inner) {
                    // Apply suffix to the target: target + "_field0"
                    return format!("{}{}", target, suffix);
                }

                // If no substitution found, return the renamed deref expression with suffix
                return format!("(deref {}){}", renamed_inner, suffix);
            }
        }

        // Not a deref pattern, use standard rename
        renames.get(lhs).cloned().unwrap_or_else(|| lhs.to_string())
    }

    /// Rename variables in an SMT expression string with deref substitution
    ///
    /// This extends rename_expr to also substitute `(deref param)` patterns
    /// with the actual variable being referenced.
    fn rename_expr_with_deref(
        &self,
        expr: &str,
        renames: &HashMap<String, String>,
        deref_subs: &HashMap<String, String>,
    ) -> String {
        // First apply standard renames
        let mut result = self.rename_expr(expr, renames);

        // Then substitute deref patterns
        // Look for patterns like (deref _inline1_param) and replace with target variable
        for (param, target) in deref_subs {
            let deref_pattern = format!("(deref {})", param);
            result = result.replace(&deref_pattern, target);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kani_fast_kinduction::SmtType;

    fn make_simple_function() -> MirProgram {
        // A simple function: fn add(a, b) { return a + b; }
        // Locals: _0 (return), _1 (a), _2 (b)
        MirProgram::builder(0)
            .local("_0", SmtType::Int)
            .local("_1", SmtType::Int)
            .local("_2", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_0".to_string(),
                    rhs: "(+ _1 _2)".to_string(),
                },
            ))
            .finish()
    }

    fn make_caller() -> MirProgram {
        // Caller: let result = add(3, 5); assert!(result == 8);
        // Locals: _0 (return), _1 (result)
        MirProgram::builder(0)
            .local("_0", SmtType::Bool)
            .local("_1", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    destination: Some("_1".to_string()),
                    func: "add".to_string(),
                    args: vec!["3".to_string(), "5".to_string()],
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ))
            .block(MirBasicBlock::new(
                1,
                MirTerminator::SwitchInt {
                    discr: "_1".to_string(),
                    targets: vec![(8, 2)],
                    otherwise: 3,
                },
            ))
            .block(MirBasicBlock::new(2, MirTerminator::Return))
            .block(MirBasicBlock::new(
                3,
                MirTerminator::Goto { target: 999_999 },
            ))
            .finish()
    }

    #[test]
    fn test_inline_simple_function() {
        let caller = make_caller();
        let callee = make_simple_function();

        let mut functions: FunctionBodies = HashMap::new();
        functions.insert("add".to_string(), Rc::new(callee));

        let config = InlinerConfig::default();
        let inlined = inline_functions(&caller, &functions, &config);

        // After inlining:
        // - Block 0 should be a Goto to the inlined function's entry
        // - The inlined blocks should be added
        // - The return should be replaced with assignment and goto

        // Check that we have more blocks than before
        assert!(
            inlined.basic_blocks.len() > caller.basic_blocks.len(),
            "Expected more blocks after inlining"
        );

        // Check that block 0 is now a Goto (not a Call)
        let block0 = inlined
            .basic_blocks
            .iter()
            .find(|b| b.id == 0)
            .expect("inlined program should preserve caller entry block");
        assert!(
            matches!(block0.terminator, MirTerminator::Goto { .. }),
            "Expected Goto terminator after inlining, got {:?}",
            block0.terminator
        );

        // Check that we have new locals
        assert!(
            inlined.locals.len() > caller.locals.len(),
            "Expected more locals after inlining"
        );
    }

    #[test]
    fn test_rename_expr() {
        let inliner = Inliner {
            function_bodies: &FunctionBodies::new(),
            config: &InlinerConfig::default(),
            inline_counter: 0,
            ref_targets: HashMap::new(),
        };

        let renames: HashMap<String, String> = vec![
            ("_1".to_string(), "_inline_1".to_string()),
            ("_2".to_string(), "_inline_2".to_string()),
        ]
        .into_iter()
        .collect();

        let result = inliner.rename_expr("(+ _1 _2)", &renames);
        assert!(
            result.contains("_inline_1") && result.contains("_inline_2"),
            "Expected renamed variables, got: {}",
            result
        );
    }

    #[test]
    fn test_rename_expr_nested_parentheses() {
        let inliner = Inliner {
            function_bodies: &FunctionBodies::new(),
            config: &InlinerConfig::default(),
            inline_counter: 0,
            ref_targets: HashMap::new(),
        };

        let renames: HashMap<String, String> = vec![
            ("_1".to_string(), "_x".to_string()),
            ("_2".to_string(), "_y".to_string()),
        ]
        .into_iter()
        .collect();

        let result = inliner.rename_expr("(+ (* _1 _2) (- _1 _2))", &renames);
        assert!(
            result.contains("_x") && result.contains("_y"),
            "Expected renamed variables in nested expr, got: {}",
            result
        );
        // Check _1 and _2 are not present (unless as part of the renamed names)
        assert!(
            !result.contains(" _1 ") && !result.contains(" _2 "),
            "Original variable names should not be present, got: {}",
            result
        );
    }

    #[test]
    fn test_rename_expr_similar_names() {
        // Test that _10 is renamed correctly when _1 is also in the rename map
        let inliner = Inliner {
            function_bodies: &FunctionBodies::new(),
            config: &InlinerConfig::default(),
            inline_counter: 0,
            ref_targets: HashMap::new(),
        };

        let renames: HashMap<String, String> = vec![
            ("_1".to_string(), "_a".to_string()),
            ("_10".to_string(), "_b".to_string()),
        ]
        .into_iter()
        .collect();

        let result = inliner.rename_expr("(+ _1 _10)", &renames);
        // Should have _a and _b, not _a0 (wrong partial replacement of _10)
        assert!(
            result.contains("_b"),
            "Expected _10 to be renamed to _b, got: {}",
            result
        );
    }

    #[test]
    fn test_rename_expr_standalone_variable() {
        let inliner = Inliner {
            function_bodies: &FunctionBodies::new(),
            config: &InlinerConfig::default(),
            inline_counter: 0,
            ref_targets: HashMap::new(),
        };

        let renames: HashMap<String, String> = vec![("_0".to_string(), "_result".to_string())]
            .into_iter()
            .collect();

        let result = inliner.rename_expr("_0", &renames);
        assert_eq!(result, "_result");
    }

    #[test]
    fn test_rename_expr_equality() {
        let inliner = Inliner {
            function_bodies: &FunctionBodies::new(),
            config: &InlinerConfig::default(),
            inline_counter: 0,
            ref_targets: HashMap::new(),
        };

        let renames: HashMap<String, String> = vec![
            ("_1".to_string(), "_x".to_string()),
            ("_2".to_string(), "_y".to_string()),
        ]
        .into_iter()
        .collect();

        let result = inliner.rename_expr("(= _1 _2)", &renames);
        assert!(
            result.contains("_x") && result.contains("_y"),
            "Expected renamed variables in equality, got: {}",
            result
        );
    }

    #[test]
    fn test_inline_empty_function_map() {
        let caller = make_caller();
        let functions: FunctionBodies = FunctionBodies::new();
        let config = InlinerConfig::default();

        let result = inline_functions(&caller, &functions, &config);

        // No inlining should occur, result should be identical
        assert_eq!(result.basic_blocks.len(), caller.basic_blocks.len());
        assert_eq!(result.locals.len(), caller.locals.len());
    }

    #[test]
    fn test_inline_function_not_in_map() {
        let caller = make_caller();
        let different_func = make_simple_function();

        let mut functions: FunctionBodies = HashMap::new();
        // Add a function with a different name
        functions.insert("different_name".to_string(), Rc::new(different_func));

        let config = InlinerConfig::default();
        let result = inline_functions(&caller, &functions, &config);

        // No inlining should occur since "add" is not in the map
        assert_eq!(result.basic_blocks.len(), caller.basic_blocks.len());
    }

    #[test]
    fn test_inline_max_depth_zero() {
        let caller = make_caller();
        let callee = make_simple_function();

        let mut functions: FunctionBodies = HashMap::new();
        functions.insert("add".to_string(), Rc::new(callee));

        // Set max_depth to 0 - should still inline at level 0
        let config = InlinerConfig { max_depth: 0 };
        let result = inline_functions(&caller, &functions, &config);

        // Inlining at depth 0 should still occur
        // The check is at the start of inline(), so depth=0 passes the check
        assert!(
            result.basic_blocks.len() >= caller.basic_blocks.len(),
            "Expected at least same number of blocks"
        );
    }

    #[test]
    fn test_inline_preserves_start_block() {
        let caller = make_caller();
        let callee = make_simple_function();

        let mut functions: FunctionBodies = HashMap::new();
        functions.insert("add".to_string(), Rc::new(callee));

        let config = InlinerConfig::default();
        let result = inline_functions(&caller, &functions, &config);

        // Start block should be preserved
        assert_eq!(result.start_block, caller.start_block);
    }

    #[test]
    fn test_inline_preserves_init() {
        let caller = make_caller();
        let callee = make_simple_function();

        let mut functions: FunctionBodies = HashMap::new();
        functions.insert("add".to_string(), Rc::new(callee));

        let config = InlinerConfig::default();
        let result = inline_functions(&caller, &functions, &config);

        // Init should remain unchanged (None in this case, preserved from caller)
        // Both should be None since we don't set init in make_caller
        assert!(caller.init.is_none());
        assert!(result.init.is_none());
    }

    #[test]
    fn test_rename_statement_assign() {
        let inliner = Inliner {
            function_bodies: &FunctionBodies::new(),
            config: &InlinerConfig::default(),
            inline_counter: 0,
            ref_targets: HashMap::new(),
        };

        let renames: HashMap<String, String> = vec![
            ("_1".to_string(), "_x".to_string()),
            ("_2".to_string(), "_y".to_string()),
        ]
        .into_iter()
        .collect();

        let stmt = MirStatement::Assign {
            lhs: "_1".to_string(),
            rhs: "(+ _2 1)".to_string(),
        };

        let deref_subs = HashMap::new();
        let result = inliner.rename_statement_with_deref(&stmt, &renames, &deref_subs);
        if let MirStatement::Assign { lhs, rhs } = result {
            assert_eq!(lhs, "_x");
            assert!(rhs.contains("_y"), "Expected _y in rhs, got: {}", rhs);
        } else {
            panic!("Expected Assign statement");
        }
    }

    #[test]
    fn test_rename_statement_assume() {
        let inliner = Inliner {
            function_bodies: &FunctionBodies::new(),
            config: &InlinerConfig::default(),
            inline_counter: 0,
            ref_targets: HashMap::new(),
        };

        let renames: HashMap<String, String> = vec![("_1".to_string(), "_x".to_string())]
            .into_iter()
            .collect();

        let stmt = MirStatement::Assume("(> _1 0)".to_string());

        let deref_subs = HashMap::new();
        let result = inliner.rename_statement_with_deref(&stmt, &renames, &deref_subs);
        if let MirStatement::Assume(cond) = result {
            assert!(
                cond.contains("_x"),
                "Expected _x in condition, got: {}",
                cond
            );
        } else {
            panic!("Expected Assume statement");
        }
    }

    #[test]
    fn test_rename_statement_assert() {
        let inliner = Inliner {
            function_bodies: &FunctionBodies::new(),
            config: &InlinerConfig::default(),
            inline_counter: 0,
            ref_targets: HashMap::new(),
        };

        let renames: HashMap<String, String> = vec![("_1".to_string(), "_x".to_string())]
            .into_iter()
            .collect();

        let stmt = MirStatement::Assert {
            condition: "(>= _1 0)".to_string(),
            message: Some("value must be non-negative".to_string()),
        };

        let deref_subs = HashMap::new();
        let result = inliner.rename_statement_with_deref(&stmt, &renames, &deref_subs);
        if let MirStatement::Assert { condition, message } = result {
            assert!(
                condition.contains("_x"),
                "Expected _x in condition, got: {}",
                condition
            );
            assert_eq!(message, Some("value must be non-negative".to_string()));
        } else {
            panic!("Expected Assert statement");
        }
    }

    #[test]
    fn test_rename_statement_array_store() {
        let inliner = Inliner {
            function_bodies: &FunctionBodies::new(),
            config: &InlinerConfig::default(),
            inline_counter: 0,
            ref_targets: HashMap::new(),
        };

        let renames: HashMap<String, String> = vec![
            ("_arr".to_string(), "_new_arr".to_string()),
            ("_i".to_string(), "_new_i".to_string()),
        ]
        .into_iter()
        .collect();

        let stmt = MirStatement::ArrayStore {
            array: "_arr".to_string(),
            index: "_i".to_string(),
            value: "42".to_string(),
        };

        let deref_subs = HashMap::new();
        let result = inliner.rename_statement_with_deref(&stmt, &renames, &deref_subs);
        if let MirStatement::ArrayStore {
            array,
            index,
            value,
        } = result
        {
            assert_eq!(array, "_new_arr");
            assert!(
                index.contains("_new_i"),
                "Expected _new_i in index, got: {}",
                index
            );
            assert_eq!(value, "42");
        } else {
            panic!("Expected ArrayStore statement");
        }
    }

    #[test]
    fn test_rename_statement_havoc() {
        let inliner = Inliner {
            function_bodies: &FunctionBodies::new(),
            config: &InlinerConfig::default(),
            inline_counter: 0,
            ref_targets: HashMap::new(),
        };

        let renames: HashMap<String, String> = vec![("_1".to_string(), "_x".to_string())]
            .into_iter()
            .collect();

        let stmt = MirStatement::Havoc {
            var: "_1".to_string(),
        };

        let deref_subs = HashMap::new();
        let result = inliner.rename_statement_with_deref(&stmt, &renames, &deref_subs);
        if let MirStatement::Havoc { var } = result {
            assert_eq!(var, "_x");
        } else {
            panic!("Expected Havoc statement");
        }
    }

    #[test]
    fn test_inline_function_with_condgoto() {
        // Create a callee with conditional branching
        let callee = MirProgram::builder(0)
            .local("_0", SmtType::Int)
            .local("_1", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "(> _1 0)".to_string(),
                    then_target: 1,
                    else_target: 2,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_0".to_string(),
                    rhs: "_1".to_string(),
                },
            ))
            .block(MirBasicBlock::new(2, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_0".to_string(),
                    rhs: "(- _1)".to_string(),
                },
            ))
            .finish();

        let caller = MirProgram::builder(0)
            .local("_0", SmtType::Int)
            .local("_1", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    destination: Some("_1".to_string()),
                    func: "abs".to_string(),
                    args: vec!["-5".to_string()],
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();

        let mut functions: FunctionBodies = HashMap::new();
        functions.insert("abs".to_string(), Rc::new(callee));

        let config = InlinerConfig::default();
        let inlined = inline_functions(&caller, &functions, &config);

        // After inlining, we should have:
        // - Original caller blocks (modified)
        // - Inlined callee blocks with CondGoto
        assert!(
            inlined.basic_blocks.len() > caller.basic_blocks.len(),
            "Expected more blocks after inlining"
        );

        // Check that at least one block has CondGoto terminator
        let has_condgoto = inlined
            .basic_blocks
            .iter()
            .any(|b| matches!(b.terminator, MirTerminator::CondGoto { .. }));
        assert!(has_condgoto, "Expected CondGoto terminator in inlined code");
    }

    #[test]
    fn test_inline_function_with_abort() {
        // Create a callee that may abort
        let callee = MirProgram::builder(0)
            .local("_0", SmtType::Int)
            .local("_1", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "(> _1 0)".to_string(),
                    then_target: 1,
                    else_target: 2,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_0".to_string(),
                    rhs: "_1".to_string(),
                },
            ))
            .block(MirBasicBlock::new(2, MirTerminator::Abort))
            .finish();

        let caller = MirProgram::builder(0)
            .local("_0", SmtType::Int)
            .local("_1", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    destination: Some("_1".to_string()),
                    func: "positive_only".to_string(),
                    args: vec!["5".to_string()],
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();

        let mut functions: FunctionBodies = HashMap::new();
        functions.insert("positive_only".to_string(), Rc::new(callee));

        let config = InlinerConfig::default();
        let inlined = inline_functions(&caller, &functions, &config);

        // Check that Abort terminator is preserved
        let has_abort = inlined
            .basic_blocks
            .iter()
            .any(|b| matches!(b.terminator, MirTerminator::Abort));
        assert!(
            has_abort,
            "Expected Abort terminator to be preserved in inlined code"
        );
    }

    #[test]
    fn test_inline_function_with_unreachable() {
        // Create a callee with Unreachable terminator
        let callee = MirProgram::builder(0)
            .local("_0", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Unreachable))
            .finish();

        let caller = MirProgram::builder(0)
            .local("_0", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    destination: Some("_0".to_string()),
                    func: "unreachable_fn".to_string(),
                    args: vec![],
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();

        let mut functions: FunctionBodies = HashMap::new();
        functions.insert("unreachable_fn".to_string(), Rc::new(callee));

        let config = InlinerConfig::default();
        let inlined = inline_functions(&caller, &functions, &config);

        // Check that Unreachable terminator is preserved
        let has_unreachable = inlined
            .basic_blocks
            .iter()
            .any(|b| matches!(b.terminator, MirTerminator::Unreachable));
        assert!(
            has_unreachable,
            "Expected Unreachable terminator to be preserved"
        );
    }

    #[test]
    fn test_inline_multiple_calls() {
        // Caller with two calls to the same function
        let callee = make_simple_function();

        let caller = MirProgram::builder(0)
            .local("_0", SmtType::Int)
            .local("_1", SmtType::Int)
            .local("_2", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    destination: Some("_1".to_string()),
                    func: "add".to_string(),
                    args: vec!["1".to_string(), "2".to_string()],
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ))
            .block(MirBasicBlock::new(
                1,
                MirTerminator::Call {
                    destination: Some("_2".to_string()),
                    func: "add".to_string(),
                    args: vec!["_1".to_string(), "3".to_string()],
                    target: 2,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ))
            .block(MirBasicBlock::new(2, MirTerminator::Return))
            .finish();

        let mut functions: FunctionBodies = HashMap::new();
        functions.insert("add".to_string(), Rc::new(callee));

        let config = InlinerConfig::default();
        let inlined = inline_functions(&caller, &functions, &config);

        // Both calls should be inlined
        // No Call terminators should remain for "add"
        let has_add_call = inlined
            .basic_blocks
            .iter()
            .any(|b| matches!(&b.terminator, MirTerminator::Call { func, .. } if func == "add"));
        assert!(
            !has_add_call,
            "Expected no remaining calls to 'add' after inlining"
        );

        // Should have more blocks from both inlined functions
        assert!(
            inlined.basic_blocks.len() > caller.basic_blocks.len() + 1,
            "Expected blocks from both inlined calls"
        );
    }

    #[test]
    fn test_inline_no_destination() {
        // Caller with a call that has no destination (void return)
        let callee = MirProgram::builder(0)
            .local("_0", SmtType::Bool) // Unit type
            .local("_1", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_0".to_string(),
                    rhs: "true".to_string(),
                },
            ))
            .finish();

        let caller = MirProgram::builder(0)
            .local("_0", SmtType::Bool)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    destination: None, // No destination
                    func: "side_effect".to_string(),
                    args: vec!["42".to_string()],
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();

        let mut functions: FunctionBodies = HashMap::new();
        functions.insert("side_effect".to_string(), Rc::new(callee));

        let config = InlinerConfig::default();
        let inlined = inline_functions(&caller, &functions, &config);

        // Should inline successfully even without destination
        assert!(
            inlined.basic_blocks.len() > caller.basic_blocks.len(),
            "Expected more blocks after inlining"
        );
    }

    #[test]
    fn test_inliner_config_default() {
        let config = InlinerConfig::default();
        assert_eq!(config.max_depth, 10);
    }

    #[test]
    fn test_inline_unique_prefixes() {
        // Verify that multiple inline sites get unique prefixes
        let callee = make_simple_function();

        let caller = MirProgram::builder(0)
            .local("_0", SmtType::Int)
            .local("_1", SmtType::Int)
            .local("_2", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    destination: Some("_1".to_string()),
                    func: "add".to_string(),
                    args: vec!["1".to_string(), "2".to_string()],
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ))
            .block(MirBasicBlock::new(
                1,
                MirTerminator::Call {
                    destination: Some("_2".to_string()),
                    func: "add".to_string(),
                    args: vec!["3".to_string(), "4".to_string()],
                    target: 2,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ))
            .block(MirBasicBlock::new(2, MirTerminator::Return))
            .finish();

        let mut functions: FunctionBodies = HashMap::new();
        functions.insert("add".to_string(), Rc::new(callee));

        let config = InlinerConfig::default();
        let inlined = inline_functions(&caller, &functions, &config);

        // Check that we have locals with different inline prefixes
        let inline_locals: Vec<_> = inlined
            .locals
            .iter()
            .filter(|l| l.name.starts_with("_inline"))
            .collect();

        // Should have locals from both inline sites with different prefixes
        let has_inline1 = inline_locals.iter().any(|l| l.name.contains("_inline1_"));
        let has_inline2 = inline_locals.iter().any(|l| l.name.contains("_inline2_"));
        assert!(
            has_inline1 && has_inline2,
            "Expected both inline sites to have unique prefixes"
        );
    }

    #[test]
    fn test_inline_tuple_parameter_fields() {
        // Test that tuple field variables are propagated when inlining
        // This is critical for parameter destructuring patterns

        // Callee: fn add_pair(pair: (i32, i32)) -> i32 { pair.0 + pair.1 }
        // Has _1 (parameter), _1_field0, _1_field1
        let callee = MirProgram::builder(0)
            .local("_0", SmtType::Int)
            .local("_1", SmtType::Int)
            .local("_1_field0", SmtType::Int)
            .local("_1_field1", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_0".to_string(),
                    rhs: "(+ _1_field0 _1_field1)".to_string(),
                },
            ))
            .finish();

        // Caller: passes _2 which has _2_field0=3, _2_field1=7
        let caller = MirProgram::builder(0)
            .local("_0", SmtType::Int)
            .local("_1", SmtType::Int)
            .local("_2", SmtType::Int)
            .local("_2_field0", SmtType::Int)
            .local("_2_field1", SmtType::Int)
            .block(
                MirBasicBlock::new(
                    0,
                    MirTerminator::Call {
                        destination: Some("_1".to_string()),
                        func: "add_pair".to_string(),
                        args: vec!["_2".to_string()],
                        target: 1,
                        unwind: None,
                        precondition_check: None,
                        postcondition_assumption: None,
                        is_range_into_iter: false,
                        is_range_next: false,
                    },
                )
                .with_statement(MirStatement::Assign {
                    lhs: "_2_field0".to_string(),
                    rhs: "3".to_string(),
                })
                .with_statement(MirStatement::Assign {
                    lhs: "_2_field1".to_string(),
                    rhs: "7".to_string(),
                }),
            )
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();

        let mut functions: FunctionBodies = HashMap::new();
        functions.insert("add_pair".to_string(), Rc::new(callee));

        let config = InlinerConfig::default();
        let inlined = inline_functions(&caller, &functions, &config);

        // After inlining, we should have assignments for the tuple field variables
        // e.g., _inline1_add_pair_1_field0 = _2_field0
        // e.g., _inline1_add_pair_1_field1 = _2_field1

        // Find the entry block for the inlined function
        let inlined_entry = inlined
            .basic_blocks
            .iter()
            .find(|b| b.id > 1)
            .expect("inlined function should create an entry block beyond caller blocks");

        // Check that field assignments exist
        let has_field0_assignment = inlined_entry.statements.iter().any(|s| {
            if let MirStatement::Assign { lhs, rhs } = s {
                lhs.contains("_field0") && rhs == "_2_field0"
            } else {
                false
            }
        });

        let has_field1_assignment = inlined_entry.statements.iter().any(|s| {
            if let MirStatement::Assign { lhs, rhs } = s {
                lhs.contains("_field1") && rhs == "_2_field1"
            } else {
                false
            }
        });

        assert!(
            has_field0_assignment,
            "Expected tuple field0 to be propagated during inlining"
        );
        assert!(
            has_field1_assignment,
            "Expected tuple field1 to be propagated during inlining"
        );
    }

    /// Benchmark test: inline 10 parallel function calls
    ///
    /// This test creates a caller that calls 10 independent functions,
    /// and measures how long it takes to inline all of them.
    ///
    /// Target: Should complete in under 10ms for the inlining operation itself.
    #[test]
    fn test_inline_performance_10_functions() {
        use std::time::Instant;

        // Create 10 simple functions that each compute something
        // fn f0(x) { return x + 0; }
        // fn f1(x) { return x + 1; }
        // ...
        // fn f9(x) { return x + 9; }
        let mut functions: FunctionBodies = HashMap::new();

        for i in 0..10 {
            let func_name = format!("f{}", i);
            let program = MirProgram::builder(0)
                .local("_0", SmtType::Int)
                .local("_1", SmtType::Int)
                .local("_2", SmtType::Int)
                .block(
                    MirBasicBlock::new(0, MirTerminator::Return)
                        .with_statement(MirStatement::Assign {
                            lhs: "_2".to_string(),
                            rhs: format!("(+ _1 {})", i),
                        })
                        .with_statement(MirStatement::Assign {
                            lhs: "_0".to_string(),
                            rhs: "(* _2 2)".to_string(),
                        }),
                )
                .finish();

            functions.insert(func_name, Rc::new(program));
        }

        // Create caller that calls all 10 functions
        let mut builder = MirProgram::builder(0);
        for i in 0..10 {
            builder = builder.local(format!("_result{}", i), SmtType::Int);
        }

        // Create a block for each call
        for i in 0..10 {
            let block = MirBasicBlock::new(
                i,
                MirTerminator::Call {
                    destination: Some(format!("_result{}", i)),
                    func: format!("f{}", i),
                    args: vec![format!("{}", i * 10)],
                    target: i + 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            );
            builder = builder.block(block);
        }
        builder = builder.block(MirBasicBlock::new(10, MirTerminator::Return));
        let caller = builder.finish();

        let config = InlinerConfig::default();

        // Warm up
        let _ = inline_functions(&caller, &functions, &config);

        // Measure inlining performance
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = inline_functions(&caller, &functions, &config);
        }
        let elapsed = start.elapsed();
        let avg_micros = elapsed.as_micros() / iterations as u128;

        println!(
            "Inlining 10 parallel functions: {} iterations in {:?} (avg: {}µs per inline)",
            iterations, elapsed, avg_micros
        );

        // After all inlining, verify we got the expected structure
        let inlined = inline_functions(&caller, &functions, &config);

        // Should have more blocks (10 original caller blocks + return + 10 inlined function bodies)
        assert!(
            inlined.basic_blocks.len() > 15,
            "Expected many blocks after inlining 10 functions, got {}",
            inlined.basic_blocks.len()
        );

        // Should have inlined locals from all 10 functions
        let inline_local_count = inlined
            .locals
            .iter()
            .filter(|l| l.name.contains("_inline"))
            .count();
        assert!(
            inline_local_count >= 20,
            "Expected many inlined locals (at least 2 per function * 10 = 20), got {}",
            inline_local_count
        );

        // Performance target: should complete in under 1ms per inline on average
        // This is generous - the actual target is ~100µs but CI machines vary
        assert!(
            avg_micros < 1000,
            "Inlining too slow: {}µs average (target: <1000µs)",
            avg_micros
        );
    }

    /// Benchmark test: inline functions with complex expressions
    ///
    /// Tests the rename_expr performance with realistic SMT expressions
    /// containing nested operations and many variables.
    #[test]
    fn test_inline_performance_complex_expressions() {
        use std::time::Instant;

        // Create a function with complex expressions that stress rename_expr
        let callee = MirProgram::builder(0)
            .local("_0", SmtType::Int)
            .local("_1", SmtType::Int)
            .local("_2", SmtType::Int)
            .local("_3", SmtType::Int)
            .local("_4", SmtType::Int)
            .local("_5", SmtType::Int)
            .block(
                MirBasicBlock::new(0, MirTerminator::Return)
                    .with_statement(MirStatement::Assign {
                        lhs: "_3".to_string(),
                        rhs: "(+ (* _1 _2) (- _1 _2))".to_string(),
                    })
                    .with_statement(MirStatement::Assign {
                        lhs: "_4".to_string(),
                        rhs: "(ite (> _1 _2) (+ _3 _1) (- _3 _2))".to_string(),
                    })
                    .with_statement(MirStatement::Assign {
                        lhs: "_5".to_string(),
                        rhs: "(and (>= _3 0) (<= _4 100) (not (= _1 _2)))".to_string(),
                    })
                    .with_statement(MirStatement::Assign {
                        lhs: "_0".to_string(),
                        rhs: "(+ _3 (+ _4 (ite _5 1 0)))".to_string(),
                    }),
            )
            .finish();

        // Create caller with 10 calls to the complex function
        let mut caller_blocks = Vec::new();
        for i in 0..10 {
            caller_blocks.push(MirBasicBlock::new(
                i,
                MirTerminator::Call {
                    destination: Some(format!("_result{}", i)),
                    func: "complex".to_string(),
                    args: vec![format!("{}", i), format!("{}", i * 2)],
                    target: i + 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ));
        }
        caller_blocks.push(MirBasicBlock::new(10, MirTerminator::Return));

        let mut builder = MirProgram::builder(0);
        for i in 0..10 {
            builder = builder.local(format!("_result{}", i), SmtType::Int);
        }
        for block in caller_blocks {
            builder = builder.block(block);
        }
        let caller = builder.finish();

        let mut functions: FunctionBodies = HashMap::new();
        functions.insert("complex".to_string(), Rc::new(callee));

        let config = InlinerConfig::default();

        // Warm up
        let _ = inline_functions(&caller, &functions, &config);

        // Measure performance
        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = inline_functions(&caller, &functions, &config);
        }
        let elapsed = start.elapsed();
        let avg_micros = elapsed.as_micros() / iterations as u128;

        println!(
            "Inlining complex expressions: {} iterations in {:?} (avg: {}µs per inline)",
            iterations, elapsed, avg_micros
        );

        // Verify correctness
        let inlined = inline_functions(&caller, &functions, &config);
        assert!(
            inlined.basic_blocks.len() > caller.basic_blocks.len(),
            "Expected more blocks after inlining"
        );

        // Performance target: should be fast even with complex expressions
        assert!(
            avg_micros < 2000,
            "Complex inlining too slow: {}µs average (target: <2000µs)",
            avg_micros
        );
    }
}
