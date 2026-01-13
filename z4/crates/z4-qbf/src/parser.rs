//! QDIMACS parser
//!
//! Parses QBF formulas in QDIMACS format (standard format for QBF benchmarks).
//!
//! ## Format
//! ```text
//! c comment line
//! p cnf <num_vars> <num_clauses>
//! e <var1> <var2> ... 0    // existential block
//! a <var1> <var2> ... 0    // universal block
//! <lit1> <lit2> ... 0      // clause
//! ...
//! ```
//!
//! Variables are 1-indexed positive integers.
//! Literals are signed integers (positive = true, negative = false).
//! Each line ends with 0.

use crate::formula::{QbfFormula, Quantifier, QuantifierBlock};
use z4_sat::{Literal, Variable};

/// Error type for QDIMACS parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QdimacsError {
    /// Missing problem line (p cnf ...)
    MissingProblemLine,
    /// Invalid problem line format
    InvalidProblemLine(String),
    /// Invalid quantifier line
    InvalidQuantifierLine(String),
    /// Invalid clause format
    InvalidClause(String),
    /// Variable out of range
    VariableOutOfRange(u32, usize),
    /// Unexpected end of file
    UnexpectedEof,
}

impl std::fmt::Display for QdimacsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QdimacsError::MissingProblemLine => write!(f, "missing problem line"),
            QdimacsError::InvalidProblemLine(s) => write!(f, "invalid problem line: {}", s),
            QdimacsError::InvalidQuantifierLine(s) => write!(f, "invalid quantifier line: {}", s),
            QdimacsError::InvalidClause(s) => write!(f, "invalid clause: {}", s),
            QdimacsError::VariableOutOfRange(v, n) => {
                write!(f, "variable {} out of range (max {})", v, n)
            }
            QdimacsError::UnexpectedEof => write!(f, "unexpected end of file"),
        }
    }
}

impl std::error::Error for QdimacsError {}

/// Parse a QDIMACS string into a QBF formula
pub fn parse_qdimacs(input: &str) -> Result<QbfFormula, QdimacsError> {
    let mut lines = input.lines().peekable();
    let mut num_vars = 0;
    let mut num_clauses = 0;
    let mut problem_found = false;

    // Skip comments and find problem line
    while let Some(line) = lines.peek() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') {
            lines.next();
            continue;
        }
        if line.starts_with('p') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 || parts[1] != "cnf" {
                return Err(QdimacsError::InvalidProblemLine(line.to_string()));
            }
            num_vars = parts[2]
                .parse()
                .map_err(|_| QdimacsError::InvalidProblemLine(line.to_string()))?;
            num_clauses = parts[3]
                .parse()
                .map_err(|_| QdimacsError::InvalidProblemLine(line.to_string()))?;
            problem_found = true;
            lines.next();
            break;
        }
        // Unknown line before problem line
        return Err(QdimacsError::InvalidProblemLine(line.to_string()));
    }

    if !problem_found {
        return Err(QdimacsError::MissingProblemLine);
    }

    let mut prefix = Vec::new();
    let mut clauses = Vec::with_capacity(num_clauses);

    // Parse quantifier prefix and clauses
    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "e" | "a" => {
                // Quantifier block
                let quantifier = if parts[0] == "e" {
                    Quantifier::Exists
                } else {
                    Quantifier::Forall
                };

                let mut variables = Vec::new();
                for part in &parts[1..] {
                    let var: i32 = part
                        .parse()
                        .map_err(|_| QdimacsError::InvalidQuantifierLine(line.to_string()))?;
                    if var == 0 {
                        break; // End of block
                    }
                    if var < 0 {
                        return Err(QdimacsError::InvalidQuantifierLine(line.to_string()));
                    }
                    let var = var as u32;
                    if var > num_vars as u32 {
                        return Err(QdimacsError::VariableOutOfRange(var, num_vars));
                    }
                    variables.push(var);
                }

                if !variables.is_empty() {
                    prefix.push(QuantifierBlock::new(quantifier, variables));
                }
            }
            _ => {
                // Clause
                let mut clause = Vec::new();
                for part in &parts {
                    let lit: i32 = part
                        .parse()
                        .map_err(|_| QdimacsError::InvalidClause(line.to_string()))?;
                    if lit == 0 {
                        break; // End of clause
                    }
                    let var = lit.unsigned_abs();
                    if var as usize > num_vars {
                        return Err(QdimacsError::VariableOutOfRange(var, num_vars));
                    }
                    clause.push(if lit > 0 {
                        Literal::positive(Variable(var))
                    } else {
                        Literal::negative(Variable(var))
                    });
                }
                if !clause.is_empty() {
                    clauses.push(clause);
                }
            }
        }
    }

    Ok(QbfFormula::new(num_vars, prefix, clauses))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_qdimacs() {
        let input = r#"
c Simple QBF example
p cnf 3 2
e 1 3 0
a 2 0
1 2 0
-1 -2 3 0
"#;

        let formula = parse_qdimacs(input).unwrap();
        assert_eq!(formula.num_vars, 3);
        assert_eq!(formula.prefix.len(), 2);
        assert_eq!(formula.clauses.len(), 2);

        // Check prefix
        assert_eq!(formula.prefix[0].quantifier, Quantifier::Exists);
        assert_eq!(formula.prefix[0].variables, vec![1, 3]);
        assert_eq!(formula.prefix[1].quantifier, Quantifier::Forall);
        assert_eq!(formula.prefix[1].variables, vec![2]);

        // Check quantifier info
        assert!(formula.is_existential(1));
        assert!(formula.is_universal(2));
        assert!(formula.is_existential(3));
    }

    #[test]
    fn test_parse_minimal_qdimacs() {
        let input = "p cnf 2 1\ne 1 2 0\n1 2 0\n";
        let formula = parse_qdimacs(input).unwrap();
        assert_eq!(formula.num_vars, 2);
        assert_eq!(formula.prefix.len(), 1);
        assert_eq!(formula.clauses.len(), 1);
    }

    #[test]
    fn test_parse_alternating_quantifiers() {
        let input = r#"
p cnf 4 1
e 1 0
a 2 0
e 3 0
a 4 0
1 -2 3 -4 0
"#;

        let formula = parse_qdimacs(input).unwrap();
        assert_eq!(formula.prefix.len(), 4);

        // Check alternation
        assert_eq!(formula.prefix[0].quantifier, Quantifier::Exists);
        assert_eq!(formula.prefix[1].quantifier, Quantifier::Forall);
        assert_eq!(formula.prefix[2].quantifier, Quantifier::Exists);
        assert_eq!(formula.prefix[3].quantifier, Quantifier::Forall);

        // Check levels
        assert_eq!(formula.var_level(1), 0);
        assert_eq!(formula.var_level(2), 1);
        assert_eq!(formula.var_level(3), 2);
        assert_eq!(formula.var_level(4), 3);
    }

    #[test]
    fn test_parse_error_missing_problem() {
        let input = "e 1 2 0\n1 2 0\n";
        assert!(matches!(
            parse_qdimacs(input),
            Err(QdimacsError::InvalidProblemLine(_))
        ));
    }

    #[test]
    fn test_parse_error_var_out_of_range() {
        let input = "p cnf 2 1\ne 1 3 0\n1 2 0\n";
        assert!(matches!(
            parse_qdimacs(input),
            Err(QdimacsError::VariableOutOfRange(3, 2))
        ));
    }
}
