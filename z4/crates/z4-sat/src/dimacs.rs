//! DIMACS CNF parser
//!
//! Parses the standard DIMACS CNF format used in SAT competitions.

use crate::literal::{Literal, Variable};
use crate::solver::Solver;
use std::io::{BufRead, BufReader, Read};

/// Error type for DIMACS parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimacsError {
    /// Missing problem line (p cnf ...)
    MissingProblemLine,
    /// Invalid problem line format
    InvalidProblemLine(String),
    /// Invalid literal in clause
    InvalidLiteral(String),
    /// I/O error description
    IoError(String),
    /// More clauses than declared
    TooManyClauses {
        /// Expected number of clauses
        expected: usize,
        /// Actual number of clauses
        got: usize,
    },
    /// Literal variable exceeds declared variable count
    VariableOutOfRange {
        /// The variable that was out of range
        var: u32,
        /// Maximum allowed variable
        max: u32,
    },
}

impl std::fmt::Display for DimacsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DimacsError::MissingProblemLine => write!(f, "Missing problem line (p cnf ...)"),
            DimacsError::InvalidProblemLine(s) => write!(f, "Invalid problem line: {}", s),
            DimacsError::InvalidLiteral(s) => write!(f, "Invalid literal: {}", s),
            DimacsError::IoError(s) => write!(f, "I/O error: {}", s),
            DimacsError::TooManyClauses { expected, got } => {
                write!(f, "Too many clauses: expected {}, got {}", expected, got)
            }
            DimacsError::VariableOutOfRange { var, max } => {
                write!(f, "Variable {} out of range (max {})", var, max)
            }
        }
    }
}

impl std::error::Error for DimacsError {}

/// Result of parsing a DIMACS file
#[derive(Debug)]
pub struct DimacsFormula {
    /// Number of variables
    pub num_vars: usize,
    /// Number of clauses (declared)
    pub num_clauses: usize,
    /// The clauses
    pub clauses: Vec<Vec<Literal>>,
}

impl DimacsFormula {
    /// Create a solver from this formula
    pub fn into_solver(self) -> Solver {
        let mut solver = Solver::new(self.num_vars);
        for clause in self.clauses {
            solver.add_clause(clause);
        }
        solver
    }
}

/// Parse a DIMACS CNF formula from a reader
pub fn parse<R: Read>(reader: R) -> Result<DimacsFormula, DimacsError> {
    let reader = BufReader::new(reader);
    let mut num_vars: Option<usize> = None;
    let mut num_clauses: Option<usize> = None;
    let mut clauses: Vec<Vec<Literal>> = Vec::new();
    let mut current_clause: Vec<Literal> = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| DimacsError::IoError(e.to_string()))?;
        let line = line.trim();

        // Skip empty lines, comments, and terminator lines
        // The '%' character is used as a terminator in some DIMACS files
        if line.is_empty() || line.starts_with('c') || line.starts_with('%') {
            continue;
        }

        // Problem line
        if line.starts_with('p') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 || parts[1] != "cnf" {
                return Err(DimacsError::InvalidProblemLine(line.to_string()));
            }
            num_vars = Some(
                parts[2]
                    .parse()
                    .map_err(|_| DimacsError::InvalidProblemLine(line.to_string()))?,
            );
            num_clauses = Some(
                parts[3]
                    .parse()
                    .map_err(|_| DimacsError::InvalidProblemLine(line.to_string()))?,
            );
            continue;
        }

        // Clause line
        let max_var = num_vars.ok_or(DimacsError::MissingProblemLine)? as u32;

        for token in line.split_whitespace() {
            let lit_val: i32 = token
                .parse()
                .map_err(|_| DimacsError::InvalidLiteral(token.to_string()))?;

            if lit_val == 0 {
                // End of clause
                if !current_clause.is_empty() {
                    clauses.push(std::mem::take(&mut current_clause));
                }
            } else {
                // Convert to our literal representation
                let var = lit_val.unsigned_abs();
                if var > max_var {
                    return Err(DimacsError::VariableOutOfRange { var, max: max_var });
                }
                // DIMACS variables are 1-indexed, we use 0-indexed
                let variable = Variable(var - 1);
                let literal = if lit_val > 0 {
                    Literal::positive(variable)
                } else {
                    Literal::negative(variable)
                };
                current_clause.push(literal);
            }
        }
    }

    // Handle final clause if not terminated with 0
    if !current_clause.is_empty() {
        clauses.push(current_clause);
    }

    let num_vars = num_vars.ok_or(DimacsError::MissingProblemLine)?;
    let num_clauses = num_clauses.ok_or(DimacsError::MissingProblemLine)?;

    Ok(DimacsFormula {
        num_vars,
        num_clauses,
        clauses,
    })
}

/// Parse a DIMACS CNF formula from a string
pub fn parse_str(input: &str) -> Result<DimacsFormula, DimacsError> {
    parse(input.as_bytes())
}

/// Write a CNF formula in DIMACS format
pub fn write_dimacs<W: std::io::Write>(
    writer: &mut W,
    num_vars: usize,
    clauses: &[Vec<Literal>],
) -> std::io::Result<()> {
    writeln!(writer, "p cnf {} {}", num_vars, clauses.len())?;
    for clause in clauses {
        for lit in clause {
            // Convert back to 1-indexed DIMACS format
            let var = lit.variable().0 as i32 + 1;
            let dimacs_lit = if lit.is_positive() { var } else { -var };
            write!(writer, "{} ", dimacs_lit)?;
        }
        writeln!(writer, "0")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let input = r"
c A simple CNF
p cnf 3 2
1 -2 3 0
-1 2 0
";
        let formula = parse_str(input).unwrap();
        assert_eq!(formula.num_vars, 3);
        assert_eq!(formula.num_clauses, 2);
        assert_eq!(formula.clauses.len(), 2);

        // First clause: x1 OR NOT x2 OR x3
        assert_eq!(formula.clauses[0].len(), 3);
        assert_eq!(formula.clauses[0][0], Literal::positive(Variable(0)));
        assert_eq!(formula.clauses[0][1], Literal::negative(Variable(1)));
        assert_eq!(formula.clauses[0][2], Literal::positive(Variable(2)));

        // Second clause: NOT x1 OR x2
        assert_eq!(formula.clauses[1].len(), 2);
        assert_eq!(formula.clauses[1][0], Literal::negative(Variable(0)));
        assert_eq!(formula.clauses[1][1], Literal::positive(Variable(1)));
    }

    #[test]
    fn test_parse_multiline_clause() {
        let input = r"
p cnf 5 1
1 2 3
4 5 0
";
        let formula = parse_str(input).unwrap();
        assert_eq!(formula.clauses.len(), 1);
        assert_eq!(formula.clauses[0].len(), 5);
    }

    #[test]
    fn test_parse_empty_clause() {
        let input = r"
p cnf 3 2
1 2 0
0
-1 0
";
        let formula = parse_str(input).unwrap();
        // Empty clauses (just "0") are skipped
        assert_eq!(formula.clauses.len(), 2);
    }

    #[test]
    fn test_missing_problem_line() {
        let input = "1 2 0";
        let result = parse_str(input);
        assert!(matches!(result, Err(DimacsError::MissingProblemLine)));
    }

    #[test]
    fn test_variable_out_of_range() {
        let input = r"
p cnf 3 1
1 2 4 0
";
        let result = parse_str(input);
        assert!(matches!(
            result,
            Err(DimacsError::VariableOutOfRange { var: 4, max: 3 })
        ));
    }

    #[test]
    fn test_roundtrip() {
        let input = r"
p cnf 3 2
1 -2 3 0
-1 2 0
";
        let formula = parse_str(input).unwrap();

        let mut output = Vec::new();
        write_dimacs(&mut output, formula.num_vars, &formula.clauses).unwrap();

        let reparsed = parse(&output[..]).unwrap();
        assert_eq!(reparsed.num_vars, formula.num_vars);
        assert_eq!(reparsed.clauses.len(), formula.clauses.len());
    }

    #[test]
    fn test_into_solver() {
        let input = r"
p cnf 3 2
1 -2 0
-1 2 0
";
        let formula = parse_str(input).unwrap();
        let solver = formula.into_solver();
        // Just verify it doesn't panic
        assert_eq!(solver.value(Variable(0)), None);
    }

    #[test]
    fn test_percent_terminator() {
        // Some DIMACS files use '%' as end-of-file marker
        // This format is common in SAT competition benchmarks
        let input = r"
p cnf 3 2
1 -2 0
-1 2 0
%
0
";
        let formula = parse_str(input).unwrap();
        assert_eq!(formula.num_vars, 3);
        assert_eq!(formula.clauses.len(), 2);
    }

    #[test]
    fn test_percent_in_middle_ignored() {
        // '%' lines should be skipped wherever they appear
        let input = r"
p cnf 3 2
1 -2 0
% comment line
-1 2 0
";
        let formula = parse_str(input).unwrap();
        assert_eq!(formula.clauses.len(), 2);
    }
}
