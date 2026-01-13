//! Proof Relevance Analysis
//!
//! This module performs backward dataflow analysis to determine which variables
//! and operations affect the verification property being checked. This enables
//! selective application of rewrites and informs delegation decisions.
//!
//! # Overview
//!
//! Not all code in a program affects the verification property. By analyzing
//! which variables "flow" to assertions, we can:
//!
//! 1. Apply algebraic rewrites only when necessary
//! 2. Skip bitwise handling for proof-irrelevant operations
//! 3. Make informed delegation decisions
//!
//! # Algorithm
//!
//! The analysis performs backward dataflow from assertions:
//!
//! 1. Start with variables appearing in `Assert` statements
//! 2. For each proof-relevant variable `x`, add variables from its definition
//! 3. Repeat until fixpoint
//!
//! # Example
//!
//! ```ignore
//! // y is NOT proof-relevant (never flows to assertion)
//! // x IS proof-relevant (used in assertion)
//! fn example() {
//!     let y = a & b;  // Bitwise, but not proof-relevant
//!     let x = y + 1;  // Wait, y flows to x...
//!     assert!(x > 0); // x is proof-relevant
//! }
//! // Actually y IS proof-relevant because x depends on y
//! ```

use crate::algebraic_rewrite::BitwiseOp;
use crate::mir::{MirProgram, MirStatement, MirTerminator, PANIC_BLOCK_ID};
use std::collections::{HashMap, HashSet};

/// Result of proof relevance analysis
#[derive(Debug, Clone)]
pub struct ProofRelevanceAnalysis {
    /// Variables that flow to assertions (proof-relevant)
    proof_relevant_vars: HashSet<String>,
    /// Bitwise operations that are proof-relevant
    relevant_bitwise_ops: Vec<BitwiseLocation>,
    /// All bitwise operations in the program
    all_bitwise_ops: Vec<BitwiseLocation>,
    /// Variable definitions: var -> expression
    definitions: HashMap<String, String>,
}

/// Location of a bitwise operation in the program
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitwiseLocation {
    /// Block index containing the operation
    pub block_idx: usize,
    /// Statement index within the block
    pub stmt_idx: usize,
    /// The bitwise operation type
    pub op: BitwiseOp,
    /// Variable being assigned (if any)
    pub target_var: Option<String>,
    /// Expression containing the bitwise operation
    pub expression: String,
}

impl ProofRelevanceAnalysis {
    /// Analyze a MIR program for proof relevance
    pub fn analyze(program: &MirProgram) -> Self {
        let mut analysis = ProofRelevanceAnalysis {
            proof_relevant_vars: HashSet::new(),
            relevant_bitwise_ops: Vec::new(),
            all_bitwise_ops: Vec::new(),
            definitions: HashMap::new(),
        };

        // Phase 1: Collect all definitions and bitwise operations
        analysis.collect_definitions_and_bitwise(program);

        // Phase 2: Seed with assertion variables
        analysis.seed_from_assertions(program);

        // Phase 3: Propagate backward to fixpoint
        analysis.propagate_relevance();

        // Phase 4: Classify bitwise operations
        analysis.classify_bitwise_ops();

        analysis
    }

    /// Collect variable definitions and identify bitwise operations
    fn collect_definitions_and_bitwise(&mut self, program: &MirProgram) {
        for (block_idx, block) in program.basic_blocks.iter().enumerate() {
            for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                if let MirStatement::Assign { lhs, rhs } = stmt {
                    // Record the definition
                    self.definitions.insert(lhs.clone(), rhs.clone());

                    // Check for bitwise operations
                    let ops = collect_bitwise_from_expr(rhs);
                    for op in ops {
                        self.all_bitwise_ops.push(BitwiseLocation {
                            block_idx,
                            stmt_idx,
                            op,
                            target_var: Some(lhs.clone()),
                            expression: rhs.clone(),
                        });
                    }
                }
            }

            // Check terminator for bitwise operations
            if let MirTerminator::CondGoto { condition, .. } = &block.terminator {
                let ops = collect_bitwise_from_expr(condition);
                for op in ops {
                    self.all_bitwise_ops.push(BitwiseLocation {
                        block_idx,
                        stmt_idx: block.statements.len(), // After last statement
                        op,
                        target_var: None,
                        expression: condition.clone(),
                    });
                }
            }
        }
    }

    /// Seed proof-relevant variables from assertions
    ///
    /// Assertions can appear as:
    /// 1. MirStatement::Assert { condition, .. } - explicit assert statements
    /// 2. CondGoto that branches to error block PANIC_BLOCK_ID - compiled Rust assert!() macro
    fn seed_from_assertions(&mut self, program: &MirProgram) {
        for block in &program.basic_blocks {
            // Check explicit Assert statements
            for stmt in &block.statements {
                if let MirStatement::Assert { condition, .. } = stmt {
                    // Extract variables from the assertion condition
                    let vars = extract_variables(condition);
                    self.proof_relevant_vars.extend(vars);
                }
            }

            // Check CondGoto terminators that branch to error block (assertion failures)
            // The Rust assert!() macro compiles to: if !condition { panic!() }
            // In MIR this may be:
            // 1. Direct: CondGoto { condition: _n, else_target: PANIC_BLOCK_ID }
            // 2. Indirect: CondGoto { condition: _n, else_target: M } where block M is Goto PANIC_BLOCK_ID
            if let MirTerminator::CondGoto {
                condition,
                else_target,
                ..
            } = &block.terminator
            {
                // Direct: else branch goes to error block PANIC_BLOCK_ID
                if *else_target == PANIC_BLOCK_ID {
                    let vars = extract_variables(condition);
                    self.proof_relevant_vars.extend(vars);
                }
            }
        }

        // Second pass: check for indirect paths to PANIC_BLOCK_ID
        // If block M has only Goto { target: PANIC_BLOCK_ID }, and some CondGoto else_targets M,
        // then that CondGoto is also an assertion
        let error_predecessor_blocks: std::collections::HashSet<usize> = program
            .basic_blocks
            .iter()
            .filter_map(|b| {
                if let MirTerminator::Goto { target } = &b.terminator {
                    if *target == PANIC_BLOCK_ID && b.statements.is_empty() {
                        return Some(b.id);
                    }
                }
                None
            })
            .collect();

        for block in &program.basic_blocks {
            if let MirTerminator::CondGoto {
                condition,
                else_target,
                ..
            } = &block.terminator
            {
                // Indirect: else branch goes to a block that immediately goes to error
                if error_predecessor_blocks.contains(else_target) {
                    let vars = extract_variables(condition);
                    self.proof_relevant_vars.extend(vars);
                }
            }
        }
    }

    /// Propagate relevance backward through definitions
    fn propagate_relevance(&mut self) {
        loop {
            let before = self.proof_relevant_vars.len();

            // For each proof-relevant variable, add variables from its definition
            let current_relevant: Vec<String> = self.proof_relevant_vars.iter().cloned().collect();

            for var in current_relevant {
                if let Some(def_expr) = self.definitions.get(&var) {
                    let used_vars = extract_variables(def_expr);
                    self.proof_relevant_vars.extend(used_vars);
                }
            }

            // Fixpoint check
            if self.proof_relevant_vars.len() == before {
                break;
            }
        }
    }

    /// Classify which bitwise operations are proof-relevant
    fn classify_bitwise_ops(&mut self) {
        for loc in &self.all_bitwise_ops {
            // A bitwise operation is proof-relevant if:
            // 1. Its target variable is proof-relevant, OR
            // 2. It appears in a proof-relevant expression

            let is_relevant = if let Some(target) = &loc.target_var {
                self.proof_relevant_vars.contains(target)
            } else {
                // No target var - check if expression variables are relevant
                let vars = extract_variables(&loc.expression);
                vars.iter().any(|v| self.proof_relevant_vars.contains(v))
            };

            if is_relevant {
                self.relevant_bitwise_ops.push(loc.clone());
            }
        }
    }

    /// Check if any proof-relevant code uses bitwise operations
    pub fn has_proof_relevant_bitwise(&self) -> bool {
        !self.relevant_bitwise_ops.is_empty()
    }

    /// Get the list of proof-relevant bitwise operations
    pub fn get_relevant_bitwise_ops(&self) -> &[BitwiseLocation] {
        &self.relevant_bitwise_ops
    }

    /// Get all bitwise operations in the program
    pub fn get_all_bitwise_ops(&self) -> &[BitwiseLocation] {
        &self.all_bitwise_ops
    }

    /// Check if a specific variable is proof-relevant
    pub fn is_proof_relevant(&self, var: &str) -> bool {
        self.proof_relevant_vars.contains(var)
    }

    /// Get the set of proof-relevant variables
    pub fn get_proof_relevant_vars(&self) -> &HashSet<String> {
        &self.proof_relevant_vars
    }

    /// Get the number of proof-relevant variables
    pub fn num_proof_relevant_vars(&self) -> usize {
        self.proof_relevant_vars.len()
    }

    /// Get the number of proof-relevant bitwise operations
    pub fn num_relevant_bitwise_ops(&self) -> usize {
        self.relevant_bitwise_ops.len()
    }
}

/// Extract variable names from an SMT expression
fn extract_variables(expr: &str) -> HashSet<String> {
    let mut vars = HashSet::new();

    // Simple tokenizer for SMT expressions
    let mut chars = expr.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c == '(' || c == ')' || c.is_whitespace() {
            chars.next();
            continue;
        }

        // Start of a token
        let mut token = String::new();
        while let Some(&c) = chars.peek() {
            if c == '(' || c == ')' || c.is_whitespace() {
                break;
            }
            token.push(c);
            chars.next();
        }

        // Check if it's a variable (starts with letter or underscore, not a keyword)
        if !token.is_empty() && is_variable_name(&token) {
            vars.insert(token);
        }
    }

    vars
}

/// Check if a token is a variable name (not a keyword or number)
fn is_variable_name(token: &str) -> bool {
    // Must start with letter or underscore
    let Some(first) = token.chars().next() else {
        return false;
    };

    if !first.is_alphabetic() && first != '_' {
        return false;
    }

    // Check it's not a number
    if token.parse::<i128>().is_ok() {
        return false;
    }

    // Skip SMT keywords
    let keywords = [
        "and", "or", "not", "ite", "true", "false", "mod", "div", "abs", "Int", "Bool", "forall",
        "exists", "let", "assert", "declare", "define", "check", "get", "set", "push", "pop",
        "bitand", "bitor", "bitxor", "bitshl", "bitshr", // Our bitwise functions
    ];

    !keywords.contains(&token)
}

/// Collect bitwise operations from an expression
fn collect_bitwise_from_expr(expr: &str) -> Vec<BitwiseOp> {
    let mut ops = Vec::new();

    if expr.contains("bitand") {
        ops.push(BitwiseOp::And);
    }
    if expr.contains("bitor") {
        ops.push(BitwiseOp::Or);
    }
    if expr.contains("bitxor") {
        ops.push(BitwiseOp::Xor);
    }
    if expr.contains("bitshl") {
        ops.push(BitwiseOp::Shl);
    }
    if expr.contains("bitshr") {
        ops.push(BitwiseOp::ShrLogical);
    }

    ops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{MirBasicBlock, MirProgramBuilder, MirTerminator};
    use kani_fast_kinduction::SmtType;

    // ============================================================
    // Variable extraction tests
    // ============================================================

    #[test]
    fn test_extract_variables_simple() {
        let vars = extract_variables("x");
        assert!(vars.contains("x"));
        assert_eq!(vars.len(), 1);
    }

    #[test]
    fn test_extract_variables_expression() {
        let vars = extract_variables("(+ x y)");
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(!vars.contains("+"));
    }

    #[test]
    fn test_extract_variables_nested() {
        let vars = extract_variables("(and (> x 0) (< y 10))");
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(!vars.contains("and"));
        assert!(!vars.contains(">"));
    }

    #[test]
    fn test_extract_variables_underscored() {
        let vars = extract_variables("(= _0 (+ _1 1))");
        assert!(vars.contains("_0"));
        assert!(vars.contains("_1"));
    }

    #[test]
    fn test_extract_variables_skips_numbers() {
        let vars = extract_variables("(+ x 42)");
        assert!(vars.contains("x"));
        assert!(!vars.contains("42"));
    }

    #[test]
    fn test_extract_variables_skips_keywords() {
        let vars = extract_variables("(and (or x y) true)");
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(!vars.contains("and"));
        assert!(!vars.contains("or"));
        assert!(!vars.contains("true"));
    }

    #[test]
    fn test_extract_variables_bitwise() {
        let vars = extract_variables("(bitand x y)");
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(!vars.contains("bitand"));
    }

    // ============================================================
    // Bitwise collection tests
    // ============================================================

    #[test]
    fn test_collect_bitwise_empty() {
        let ops = collect_bitwise_from_expr("(+ x 1)");
        assert!(ops.is_empty());
    }

    #[test]
    fn test_collect_bitwise_and() {
        let ops = collect_bitwise_from_expr("(bitand x 255)");
        assert_eq!(ops.len(), 1);
        assert!(ops.contains(&BitwiseOp::And));
    }

    #[test]
    fn test_collect_bitwise_multiple() {
        let ops = collect_bitwise_from_expr("(bitor (bitand x y) z)");
        assert_eq!(ops.len(), 2);
        assert!(ops.contains(&BitwiseOp::And));
        assert!(ops.contains(&BitwiseOp::Or));
    }

    #[test]
    fn test_collect_bitwise_shift() {
        let ops = collect_bitwise_from_expr("(bitshl x 4)");
        assert_eq!(ops.len(), 1);
        assert!(ops.contains(&BitwiseOp::Shl));
    }

    // ============================================================
    // Analysis tests with simple programs
    // ============================================================

    #[test]
    fn test_analysis_no_bitwise() {
        let program = MirProgramBuilder::new(0)
            .local("x", SmtType::Int)
            .init("(= x 0)")
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assert {
                    condition: "(>= x 0)".to_string(),
                    message: None,
                },
            ))
            .finish();

        let analysis = ProofRelevanceAnalysis::analyze(&program);

        assert!(!analysis.has_proof_relevant_bitwise());
        assert!(analysis.is_proof_relevant("x"));
        assert_eq!(analysis.num_relevant_bitwise_ops(), 0);
    }

    #[test]
    fn test_analysis_irrelevant_bitwise() {
        // y has bitwise but is never used in assertion
        let program = MirProgramBuilder::new(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .init("(and (= x 5) (= y 0))")
            .block(
                MirBasicBlock::new(0, MirTerminator::Return)
                    .with_statement(MirStatement::Assign {
                        lhs: "y".to_string(),
                        rhs: "(bitand y 255)".to_string(),
                    })
                    .with_statement(MirStatement::Assert {
                        condition: "(> x 0)".to_string(),
                        message: None,
                    }),
            )
            .finish();

        let analysis = ProofRelevanceAnalysis::analyze(&program);

        // There is a bitwise op in the program
        assert!(!analysis.get_all_bitwise_ops().is_empty());
        // But it's NOT proof-relevant because y is not used in assertion
        assert!(!analysis.has_proof_relevant_bitwise());
        assert!(analysis.is_proof_relevant("x"));
        assert!(!analysis.is_proof_relevant("y"));
    }

    #[test]
    fn test_analysis_relevant_bitwise() {
        // y has bitwise and flows to assertion
        let program = MirProgramBuilder::new(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .init("(and (= x 100) (= y 0))")
            .block(
                MirBasicBlock::new(0, MirTerminator::Return)
                    .with_statement(MirStatement::Assign {
                        lhs: "y".to_string(),
                        rhs: "(bitand x 255)".to_string(),
                    })
                    .with_statement(MirStatement::Assert {
                        condition: "(< y 256)".to_string(),
                        message: None,
                    }),
            )
            .finish();

        let analysis = ProofRelevanceAnalysis::analyze(&program);

        // y is used in assertion, so the bitwise op is proof-relevant
        assert!(analysis.has_proof_relevant_bitwise());
        assert!(analysis.is_proof_relevant("y"));
        // x flows to y which is proof-relevant
        assert!(analysis.is_proof_relevant("x"));
    }

    #[test]
    fn test_analysis_transitive_relevance() {
        // a -> b -> c -> assertion
        let program = MirProgramBuilder::new(0)
            .local("a", SmtType::Int)
            .local("b", SmtType::Int)
            .local("c", SmtType::Int)
            .init("(and (= a 0) (and (= b 0) (= c 0)))")
            .block(
                MirBasicBlock::new(0, MirTerminator::Return)
                    .with_statement(MirStatement::Assign {
                        lhs: "b".to_string(),
                        rhs: "(bitand a 255)".to_string(),
                    })
                    .with_statement(MirStatement::Assign {
                        lhs: "c".to_string(),
                        rhs: "(+ b 1)".to_string(),
                    })
                    .with_statement(MirStatement::Assert {
                        condition: "(> c 0)".to_string(),
                        message: None,
                    }),
            )
            .finish();

        let analysis = ProofRelevanceAnalysis::analyze(&program);

        // c is in assertion, b flows to c, a flows to b (with bitwise)
        assert!(analysis.has_proof_relevant_bitwise());
        assert!(analysis.is_proof_relevant("a"));
        assert!(analysis.is_proof_relevant("b"));
        assert!(analysis.is_proof_relevant("c"));
    }

    #[test]
    fn test_analysis_multiple_bitwise_ops() {
        let program = MirProgramBuilder::new(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .local("z", SmtType::Int)
            .init("(and (= x 0) (and (= y 0) (= z 0)))")
            .block(
                MirBasicBlock::new(0, MirTerminator::Return)
                    .with_statement(MirStatement::Assign {
                        lhs: "y".to_string(),
                        rhs: "(bitand x 255)".to_string(),
                    })
                    .with_statement(MirStatement::Assign {
                        lhs: "z".to_string(),
                        rhs: "(bitor y 1)".to_string(),
                    })
                    .with_statement(MirStatement::Assert {
                        condition: "(> z 0)".to_string(),
                        message: None,
                    }),
            )
            .finish();

        let analysis = ProofRelevanceAnalysis::analyze(&program);

        // Both bitwise ops are proof-relevant
        assert!(analysis.has_proof_relevant_bitwise());
        assert_eq!(analysis.num_relevant_bitwise_ops(), 2);

        let ops: Vec<_> = analysis
            .get_relevant_bitwise_ops()
            .iter()
            .map(|l| l.op)
            .collect();
        assert!(ops.contains(&BitwiseOp::And));
        assert!(ops.contains(&BitwiseOp::Or));
    }

    // ============================================================
    // Edge case tests
    // ============================================================

    #[test]
    fn test_analysis_empty_program() {
        let program = MirProgramBuilder::new(0)
            .local("x", SmtType::Int)
            .init("(= x 0)")
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();

        let analysis = ProofRelevanceAnalysis::analyze(&program);

        assert!(!analysis.has_proof_relevant_bitwise());
        assert_eq!(analysis.num_proof_relevant_vars(), 0);
    }

    #[test]
    fn test_analysis_bitwise_in_condition() {
        // Bitwise in conditional, not assignment
        let program = MirProgramBuilder::new(0)
            .local("x", SmtType::Int)
            .init("(= x 100)")
            .block(
                MirBasicBlock::new(
                    0,
                    MirTerminator::CondGoto {
                        condition: "(= (bitand x 1) 0)".to_string(),
                        then_target: 1,
                        else_target: 1,
                    },
                )
                .with_statement(MirStatement::Assert {
                    condition: "(>= x 0)".to_string(),
                    message: None,
                }),
            )
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();

        let analysis = ProofRelevanceAnalysis::analyze(&program);

        // The bitwise in condition is found
        assert!(!analysis.get_all_bitwise_ops().is_empty());
    }

    #[test]
    fn test_analysis_condgoto_assertion_direct() {
        // Tests that CondGoto branches DIRECTLY to error block (PANIC_BLOCK_ID) are recognized
        let program = MirProgramBuilder::new(0)
            .local("_1", SmtType::Int)
            .local("_2", SmtType::Int)
            .local("_3", SmtType::Bool)
            .init("(= _2 12345)")
            .block(
                MirBasicBlock::new(
                    0,
                    MirTerminator::CondGoto {
                        condition: "_3".to_string(),
                        then_target: 1,
                        else_target: PANIC_BLOCK_ID, // Direct error block = assertion failure
                    },
                )
                .with_statement(MirStatement::Assign {
                    lhs: "_1".to_string(),
                    rhs: "(bitand _2 255)".to_string(),
                })
                .with_statement(MirStatement::Assign {
                    lhs: "_3".to_string(),
                    rhs: "(>= _1 0)".to_string(),
                }),
            )
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();

        let analysis = ProofRelevanceAnalysis::analyze(&program);

        // _3 is the assertion condition (CondGoto else_target=PANIC_BLOCK_ID)
        assert!(analysis.is_proof_relevant("_3"));
        // _1 is used in the definition of _3
        assert!(analysis.is_proof_relevant("_1"));
        // _2 is used in the definition of _1 which has bitwise
        assert!(analysis.is_proof_relevant("_2"));

        // The bitwise operation in (bitand _2 255) should be detected as proof-relevant
        assert!(analysis.has_proof_relevant_bitwise());
        assert_eq!(analysis.num_relevant_bitwise_ops(), 1);
    }

    #[test]
    fn test_analysis_condgoto_assertion_indirect() {
        // Tests that CondGoto branches INDIRECTLY to error block (via intermediate block)
        // This is how rustc actually compiles assert!():
        // Block 0: CondGoto { condition: "_3", then_target: 1, else_target: 2 }
        // Block 1: Return (success)
        // Block 2: Goto { target: PANIC_BLOCK_ID } (failure, empty block)
        let program = MirProgramBuilder::new(0)
            .local("_1", SmtType::Int)
            .local("_2", SmtType::Int)
            .local("_3", SmtType::Bool)
            .init("(= _2 12345)")
            .block(
                MirBasicBlock::new(
                    0,
                    MirTerminator::CondGoto {
                        condition: "_3".to_string(),
                        then_target: 1,
                        else_target: 2, // Indirect: goes to block 2
                    },
                )
                .with_statement(MirStatement::Assign {
                    lhs: "_1".to_string(),
                    rhs: "(bitand _2 255)".to_string(),
                })
                .with_statement(MirStatement::Assign {
                    lhs: "_3".to_string(),
                    rhs: "(>= _1 0)".to_string(),
                }),
            )
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .block(MirBasicBlock::new(
                2,
                MirTerminator::Goto {
                    target: PANIC_BLOCK_ID,
                }, // Block 2 goes to error
            ))
            .finish();

        let analysis = ProofRelevanceAnalysis::analyze(&program);

        // _3 is the assertion condition (CondGoto else_target=2, which goes to PANIC_BLOCK_ID)
        assert!(analysis.is_proof_relevant("_3"));
        // _1 is used in the definition of _3
        assert!(analysis.is_proof_relevant("_1"));
        // _2 is used in the definition of _1 which has bitwise
        assert!(analysis.is_proof_relevant("_2"));

        // The bitwise operation in (bitand _2 255) should be detected as proof-relevant
        assert!(analysis.has_proof_relevant_bitwise());
        assert_eq!(analysis.num_relevant_bitwise_ops(), 1);
    }

    #[test]
    fn test_is_variable_name() {
        assert!(is_variable_name("x"));
        assert!(is_variable_name("_0"));
        assert!(is_variable_name("foo_bar"));
        assert!(is_variable_name("myVar123"));

        assert!(!is_variable_name("and"));
        assert!(!is_variable_name("or"));
        assert!(!is_variable_name("true"));
        assert!(!is_variable_name("false"));
        assert!(!is_variable_name("bitand"));
        assert!(!is_variable_name("42"));
        assert!(!is_variable_name("-1"));
    }

    #[test]
    fn test_bitwise_location_fields() {
        let loc = BitwiseLocation {
            block_idx: 0,
            stmt_idx: 1,
            op: BitwiseOp::And,
            target_var: Some("y".to_string()),
            expression: "(bitand x 255)".to_string(),
        };

        assert_eq!(loc.block_idx, 0);
        assert_eq!(loc.stmt_idx, 1);
        assert_eq!(loc.op, BitwiseOp::And);
        assert_eq!(loc.target_var, Some("y".to_string()));
    }
}
