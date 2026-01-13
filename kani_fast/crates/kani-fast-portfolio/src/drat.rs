//! DRAT proof parsing for learned clause extraction
//!
//! DRAT (Deletion Resolution Asymmetric Tautology) proofs contain all clauses
//! learned during SAT solving. This module parses DRAT proofs to extract
//! learned clauses for incremental solving.
//!
//! # DRAT Format
//!
//! The text format is:
//! - `literals 0` - Add a clause
//! - `d literals 0` - Delete a clause
//!
//! The binary format uses variable-length encoding.

use std::io::BufRead;
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during DRAT parsing
#[derive(Debug, Error)]
pub enum DratError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    #[error("Invalid binary format at byte {offset}")]
    BinaryFormat { offset: usize },
}

/// A clause from a DRAT proof
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DratClause {
    /// Literals in the clause (positive = true, negative = false)
    pub literals: Vec<i32>,
    /// Whether this is an addition (true) or deletion (false)
    pub is_addition: bool,
}

impl DratClause {
    /// Create a new addition clause
    pub fn add(literals: Vec<i32>) -> Self {
        Self {
            literals,
            is_addition: true,
        }
    }

    /// Create a new deletion clause
    pub fn delete(literals: Vec<i32>) -> Self {
        Self {
            literals,
            is_addition: false,
        }
    }

    /// Get the clause size
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Check if clause is empty
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }
}

/// Parser for DRAT proofs
pub struct DratParser;

impl DratParser {
    /// Parse a DRAT proof from a file
    pub fn parse_file(path: &Path) -> Result<Vec<DratClause>, DratError> {
        // Read entire file once
        let bytes = std::fs::read(path)?;
        if bytes.is_empty() {
            return Ok(Vec::new());
        }

        // Binary DRAT uses values >= 128 for encoding
        // Text DRAT starts with digits, 'd', '-', or whitespace
        let is_binary = bytes[0] >= 128 || bytes.iter().take(100).any(|&b| b == b'a');

        if is_binary {
            Self::parse_binary(&bytes)
        } else {
            // Parse as text
            let reader = std::io::Cursor::new(bytes);
            Self::parse_text(reader)
        }
    }

    /// Parse text-format DRAT proof
    pub fn parse_text<R: BufRead>(reader: R) -> Result<Vec<DratClause>, DratError> {
        let mut clauses = Vec::new();

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('c') {
                continue;
            }

            // Check for deletion
            let (is_addition, literals_str) = if let Some(rest) = line.strip_prefix('d') {
                (false, rest.trim())
            } else {
                (true, line)
            };

            // Parse literals
            let mut literals = Vec::new();
            for token in literals_str.split_whitespace() {
                match token.parse::<i32>() {
                    Ok(0) => break, // End of clause
                    Ok(lit) => literals.push(lit),
                    Err(_) => {
                        return Err(DratError::Parse {
                            line: line_num + 1,
                            message: format!("Invalid literal: {token}"),
                        });
                    }
                }
            }

            // The empty clause (just "0") signals end of proof
            if literals.is_empty() && is_addition {
                // Empty clause = proof of UNSAT, we're done
                break;
            }

            clauses.push(DratClause {
                literals,
                is_addition,
            });
        }

        Ok(clauses)
    }

    /// Parse binary DRAT proof
    pub fn parse_binary(bytes: &[u8]) -> Result<Vec<DratClause>, DratError> {
        let mut clauses = Vec::new();
        let mut offset = 0;

        while offset < bytes.len() {
            // Check for deletion marker 'd' (ASCII 100)
            let is_addition = if bytes[offset] == b'd' {
                offset += 1;
                false
            } else if bytes[offset] == b'a' {
                // 'a' is the binary add marker
                offset += 1;
                true
            } else {
                true
            };

            // Read literals until we get 0
            let mut literals = Vec::new();
            loop {
                if offset >= bytes.len() {
                    break;
                }

                let (lit, consumed) = Self::read_binary_lit(&bytes[offset..])?;
                offset += consumed;

                if lit == 0 {
                    break;
                }
                literals.push(lit);
            }

            if !literals.is_empty() {
                clauses.push(DratClause {
                    literals,
                    is_addition,
                });
            }
        }

        Ok(clauses)
    }

    /// Read a literal in binary DRAT format (variable-length encoding)
    fn read_binary_lit(bytes: &[u8]) -> Result<(i32, usize), DratError> {
        if bytes.is_empty() {
            return Ok((0, 0));
        }

        // Binary format: variable-length encoding
        // Each byte has 7 data bits and 1 continuation bit (MSB)
        let mut value: u32 = 0;
        let mut shift = 0;
        let mut consumed = 0;

        for &byte in bytes {
            consumed += 1;
            value |= ((byte & 0x7F) as u32) << shift;

            if byte & 0x80 == 0 {
                // Last byte
                break;
            }
            shift += 7;

            if shift > 28 {
                return Err(DratError::BinaryFormat { offset: consumed });
            }
        }

        // Convert from unsigned to signed
        // Least significant bit indicates sign
        let lit = if value & 1 == 0 {
            (value >> 1) as i32
        } else {
            -((value >> 1) as i32)
        };

        Ok((lit, consumed))
    }
}

/// Extract learned clauses (additions only) from a DRAT proof
pub fn extract_learned_clauses(clauses: &[DratClause]) -> Vec<Vec<i32>> {
    clauses
        .iter()
        .filter(|c| c.is_addition && !c.is_empty())
        .map(|c| c.literals.clone())
        .collect()
}

/// Filter learned clauses by size and activity heuristics
pub fn filter_learned_clauses(
    clauses: Vec<Vec<i32>>,
    min_size: usize,
    max_size: usize,
    max_count: usize,
) -> Vec<Vec<i32>> {
    let mut filtered: Vec<Vec<i32>> = clauses
        .into_iter()
        .filter(|c| c.len() >= min_size && c.len() <= max_size)
        .collect();

    // Sort by size (prefer shorter clauses)
    filtered.sort_by_key(|c| c.len());

    // Limit count
    filtered.truncate(max_count);

    filtered
}

/// Statistics about a DRAT proof
#[derive(Debug, Clone, Default)]
pub struct DratStats {
    /// Total number of clause additions
    pub additions: usize,
    /// Total number of clause deletions
    pub deletions: usize,
    /// Size distribution of added clauses
    pub size_histogram: Vec<usize>,
    /// Maximum clause size
    pub max_size: usize,
    /// Minimum clause size
    pub min_size: usize,
}

impl DratStats {
    /// Compute statistics from parsed clauses
    pub fn from_clauses(clauses: &[DratClause]) -> Self {
        let mut stats = Self {
            min_size: usize::MAX,
            ..Default::default()
        };

        for clause in clauses {
            if clause.is_addition {
                stats.additions += 1;
                let size = clause.len();
                stats.max_size = stats.max_size.max(size);
                stats.min_size = stats.min_size.min(size);

                // Update histogram
                if size >= stats.size_histogram.len() {
                    stats.size_histogram.resize(size + 1, 0);
                }
                stats.size_histogram[size] += 1;
            } else {
                stats.deletions += 1;
            }
        }

        if stats.min_size == usize::MAX {
            stats.min_size = 0;
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ==================== DratClause Tests ====================

    #[test]
    fn test_drat_clause_add() {
        let clause = DratClause::add(vec![1, 2, 3]);
        assert_eq!(clause.literals, vec![1, 2, 3]);
        assert!(clause.is_addition);
    }

    #[test]
    fn test_drat_clause_delete() {
        let clause = DratClause::delete(vec![1, 2]);
        assert_eq!(clause.literals, vec![1, 2]);
        assert!(!clause.is_addition);
    }

    #[test]
    fn test_drat_clause_len() {
        let clause = DratClause::add(vec![1, 2, 3]);
        assert_eq!(clause.len(), 3);
    }

    #[test]
    fn test_drat_clause_is_empty() {
        let empty = DratClause::add(vec![]);
        assert!(empty.is_empty());

        let non_empty = DratClause::add(vec![1]);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_drat_clause_equality() {
        let c1 = DratClause::add(vec![1, 2, 3]);
        let c2 = DratClause::add(vec![1, 2, 3]);
        let c3 = DratClause::add(vec![1, 2]);
        let c4 = DratClause::delete(vec![1, 2, 3]);

        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
        assert_ne!(c1, c4);
    }

    #[test]
    fn test_drat_clause_clone() {
        let clause = DratClause::add(vec![1, 2, 3]);
        let cloned = clause.clone();
        assert_eq!(clause, cloned);
    }

    #[test]
    fn test_drat_clause_debug() {
        let clause = DratClause::add(vec![1, 2]);
        let debug = format!("{:?}", clause);
        assert!(debug.contains("DratClause"));
        assert!(debug.contains("literals"));
    }

    #[test]
    fn test_clause_methods() {
        let clause = DratClause::add(vec![1, 2, 3]);
        assert_eq!(clause.len(), 3);
        assert!(!clause.is_empty());

        let empty = DratClause::add(vec![]);
        assert!(empty.is_empty());
    }

    // ==================== DratParser::parse_text Tests ====================

    #[test]
    fn test_parse_text_drat() {
        let drat = "1 2 3 0\n-1 -2 0\nd 1 2 0\n0\n";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();

        assert_eq!(clauses.len(), 3);
        assert_eq!(clauses[0], DratClause::add(vec![1, 2, 3]));
        assert_eq!(clauses[1], DratClause::add(vec![-1, -2]));
        assert_eq!(clauses[2], DratClause::delete(vec![1, 2]));
    }

    #[test]
    fn test_parse_empty_drat() {
        let drat = "";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();
        assert!(clauses.is_empty());
    }

    #[test]
    fn test_parse_comments() {
        let drat = "c comment\n1 2 0\nc another\n";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();
        assert_eq!(clauses.len(), 1);
    }

    #[test]
    fn test_parse_single_clause() {
        let drat = "1 2 3 0\n";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0], DratClause::add(vec![1, 2, 3]));
    }

    #[test]
    fn test_parse_unit_clause() {
        let drat = "42 0\n";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0], DratClause::add(vec![42]));
    }

    #[test]
    fn test_parse_negative_literals() {
        let drat = "-1 -2 -3 0\n";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0], DratClause::add(vec![-1, -2, -3]));
    }

    #[test]
    fn test_parse_mixed_literals() {
        let drat = "1 -2 3 -4 0\n";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0], DratClause::add(vec![1, -2, 3, -4]));
    }

    #[test]
    fn test_parse_multiple_deletions() {
        let drat = "d 1 2 0\nd 3 4 0\n";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();
        assert_eq!(clauses.len(), 2);
        assert!(!clauses[0].is_addition);
        assert!(!clauses[1].is_addition);
    }

    #[test]
    fn test_parse_whitespace_handling() {
        let drat = "  1   2   3   0  \n";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0], DratClause::add(vec![1, 2, 3]));
    }

    #[test]
    fn test_parse_empty_lines() {
        let drat = "1 2 0\n\n\n3 4 0\n";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();
        assert_eq!(clauses.len(), 2);
    }

    #[test]
    fn test_parse_stops_at_empty_clause() {
        // Empty addition clause (0 alone) signals end
        let drat = "1 2 0\n0\n3 4 0\n";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();
        // Should stop at the empty clause (just "0")
        assert_eq!(clauses.len(), 1);
    }

    #[test]
    fn test_parse_large_literals() {
        let drat = "1000000 2000000 0\n";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0], DratClause::add(vec![1000000, 2000000]));
    }

    #[test]
    fn test_parse_error_invalid_literal() {
        let drat = "1 abc 0\n";
        let reader = Cursor::new(drat);
        let result = DratParser::parse_text(reader);
        assert!(result.is_err());
        if let Err(DratError::Parse { line, message }) = result {
            assert_eq!(line, 1);
            assert!(message.contains("Invalid literal"));
        }
    }

    #[test]
    fn test_parse_deletion_with_space() {
        let drat = "d 1 2 3 0\n";
        let reader = Cursor::new(drat);
        let clauses = DratParser::parse_text(reader).unwrap();
        assert_eq!(clauses.len(), 1);
        assert!(!clauses[0].is_addition);
        assert_eq!(clauses[0].literals, vec![1, 2, 3]);
    }

    // ==================== DratParser::parse_binary Tests ====================

    #[test]
    fn test_parse_binary_simple() {
        // Binary format: variable-length encoding
        // Literal 1 = 2 (1 << 1), encoded as single byte 0x02
        // Literal -1 = 3 ((1 << 1) | 1), encoded as single byte 0x03
        // 0 = terminator, encoded as 0x00
        let bytes = vec![0x02, 0x00]; // 1 0 (clause with literal 1)
        let clauses = DratParser::parse_binary(&bytes).unwrap();
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].literals, vec![1]);
    }

    #[test]
    fn test_parse_binary_negative_literal() {
        // Literal -2 = (2 << 1) | 1 = 5
        let bytes = vec![0x05, 0x00];
        let clauses = DratParser::parse_binary(&bytes).unwrap();
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].literals, vec![-2]);
    }

    #[test]
    fn test_parse_binary_addition_marker() {
        // 'a' is binary add marker (ASCII 97 = 0x61)
        let bytes = vec![b'a', 0x02, 0x00];
        let clauses = DratParser::parse_binary(&bytes).unwrap();
        assert_eq!(clauses.len(), 1);
        assert!(clauses[0].is_addition);
    }

    #[test]
    fn test_parse_binary_deletion_marker() {
        // 'd' is deletion marker (ASCII 100 = 0x64)
        let bytes = vec![b'd', 0x02, 0x00];
        let clauses = DratParser::parse_binary(&bytes).unwrap();
        assert_eq!(clauses.len(), 1);
        assert!(!clauses[0].is_addition);
    }

    #[test]
    fn test_parse_binary_empty() {
        let bytes: Vec<u8> = vec![];
        let clauses = DratParser::parse_binary(&bytes).unwrap();
        assert!(clauses.is_empty());
    }

    #[test]
    fn test_parse_binary_multiple_clauses() {
        // Two clauses: [1] and [2]
        let bytes = vec![0x02, 0x00, 0x04, 0x00];
        let clauses = DratParser::parse_binary(&bytes).unwrap();
        assert_eq!(clauses.len(), 2);
        assert_eq!(clauses[0].literals, vec![1]);
        assert_eq!(clauses[1].literals, vec![2]);
    }

    #[test]
    fn test_read_binary_lit_simple() {
        // Literal 1 = 0x02
        let bytes = vec![0x02];
        let (lit, consumed) = DratParser::read_binary_lit(&bytes).unwrap();
        assert_eq!(lit, 1);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_read_binary_lit_zero() {
        let bytes = vec![0x00];
        let (lit, consumed) = DratParser::read_binary_lit(&bytes).unwrap();
        assert_eq!(lit, 0);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_read_binary_lit_negative() {
        // -1 = 0x03
        let bytes = vec![0x03];
        let (lit, consumed) = DratParser::read_binary_lit(&bytes).unwrap();
        assert_eq!(lit, -1);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_read_binary_lit_empty() {
        let bytes: Vec<u8> = vec![];
        let (lit, consumed) = DratParser::read_binary_lit(&bytes).unwrap();
        assert_eq!(lit, 0);
        assert_eq!(consumed, 0);
    }

    #[test]
    fn test_read_binary_lit_multi_byte() {
        // Multi-byte encoding: value >= 64 requires multiple bytes
        // Literal 64 = 128 (64 << 1)
        // 128 in VLQ: 0x80 0x01 (continuation bit set on first byte)
        let bytes = vec![0x80, 0x01];
        let (lit, consumed) = DratParser::read_binary_lit(&bytes).unwrap();
        assert_eq!(lit, 64);
        assert_eq!(consumed, 2);
    }

    // ==================== extract_learned_clauses Tests ====================

    #[test]
    fn test_extract_learned() {
        let clauses = vec![
            DratClause::add(vec![1, 2]),
            DratClause::delete(vec![1, 2]),
            DratClause::add(vec![3, 4]),
        ];

        let learned = extract_learned_clauses(&clauses);
        assert_eq!(learned.len(), 2);
        assert_eq!(learned[0], vec![1, 2]);
        assert_eq!(learned[1], vec![3, 4]);
    }

    #[test]
    fn test_extract_learned_empty() {
        let clauses: Vec<DratClause> = vec![];
        let learned = extract_learned_clauses(&clauses);
        assert!(learned.is_empty());
    }

    #[test]
    fn test_extract_learned_only_deletions() {
        let clauses = vec![
            DratClause::delete(vec![1, 2]),
            DratClause::delete(vec![3, 4]),
        ];
        let learned = extract_learned_clauses(&clauses);
        assert!(learned.is_empty());
    }

    #[test]
    fn test_extract_learned_skip_empty() {
        let clauses = vec![DratClause::add(vec![]), DratClause::add(vec![1, 2])];
        let learned = extract_learned_clauses(&clauses);
        // Empty clauses are filtered out
        assert_eq!(learned.len(), 1);
        assert_eq!(learned[0], vec![1, 2]);
    }

    #[test]
    fn test_extract_learned_preserves_order() {
        let clauses = vec![
            DratClause::add(vec![1]),
            DratClause::add(vec![2]),
            DratClause::add(vec![3]),
        ];
        let learned = extract_learned_clauses(&clauses);
        assert_eq!(learned, vec![vec![1], vec![2], vec![3]]);
    }

    // ==================== filter_learned_clauses Tests ====================

    #[test]
    fn test_filter_learned() {
        let clauses = vec![
            vec![1],             // size 1
            vec![1, 2],          // size 2
            vec![1, 2, 3],       // size 3
            vec![1, 2, 3, 4],    // size 4
            vec![1, 2, 3, 4, 5], // size 5
        ];

        let filtered = filter_learned_clauses(clauses, 2, 4, 2);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0], vec![1, 2]); // size 2
        assert_eq!(filtered[1], vec![1, 2, 3]); // size 3
    }

    #[test]
    fn test_filter_learned_empty_input() {
        let clauses: Vec<Vec<i32>> = vec![];
        let filtered = filter_learned_clauses(clauses, 1, 10, 100);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_learned_all_too_small() {
        let clauses = vec![vec![1], vec![2]];
        let filtered = filter_learned_clauses(clauses, 5, 10, 100);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_learned_all_too_large() {
        let clauses = vec![vec![1, 2, 3, 4, 5], vec![1, 2, 3, 4, 5, 6]];
        let filtered = filter_learned_clauses(clauses, 1, 3, 100);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_learned_max_count() {
        let clauses = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8]];
        let filtered = filter_learned_clauses(clauses, 1, 10, 2);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_learned_sorts_by_size() {
        let clauses = vec![
            vec![1, 2, 3, 4], // size 4
            vec![5, 6],       // size 2
            vec![7, 8, 9],    // size 3
        ];
        let filtered = filter_learned_clauses(clauses, 1, 10, 10);
        // Should be sorted by size ascending
        assert_eq!(filtered[0].len(), 2);
        assert_eq!(filtered[1].len(), 3);
        assert_eq!(filtered[2].len(), 4);
    }

    #[test]
    fn test_filter_learned_exact_bounds() {
        let clauses = vec![
            vec![1],          // size 1 - excluded
            vec![1, 2],       // size 2 - included
            vec![1, 2, 3],    // size 3 - included
            vec![1, 2, 3, 4], // size 4 - excluded
        ];
        let filtered = filter_learned_clauses(clauses, 2, 3, 100);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].len(), 2);
        assert_eq!(filtered[1].len(), 3);
    }

    // ==================== DratStats Tests ====================

    #[test]
    fn test_drat_stats() {
        let clauses = vec![
            DratClause::add(vec![1, 2]),
            DratClause::add(vec![1, 2, 3]),
            DratClause::delete(vec![1]),
            DratClause::add(vec![1]),
        ];

        let stats = DratStats::from_clauses(&clauses);
        assert_eq!(stats.additions, 3);
        assert_eq!(stats.deletions, 1);
        assert_eq!(stats.min_size, 1);
        assert_eq!(stats.max_size, 3);
    }

    #[test]
    fn test_drat_stats_empty() {
        let clauses: Vec<DratClause> = vec![];
        let stats = DratStats::from_clauses(&clauses);
        assert_eq!(stats.additions, 0);
        assert_eq!(stats.deletions, 0);
        assert_eq!(stats.min_size, 0);
        assert_eq!(stats.max_size, 0);
    }

    #[test]
    fn test_drat_stats_only_deletions() {
        let clauses = vec![
            DratClause::delete(vec![1, 2]),
            DratClause::delete(vec![3, 4, 5]),
        ];
        let stats = DratStats::from_clauses(&clauses);
        assert_eq!(stats.additions, 0);
        assert_eq!(stats.deletions, 2);
        assert_eq!(stats.min_size, 0);
        assert_eq!(stats.max_size, 0);
    }

    #[test]
    fn test_drat_stats_histogram() {
        let clauses = vec![
            DratClause::add(vec![1]),
            DratClause::add(vec![1, 2]),
            DratClause::add(vec![1, 2]),
            DratClause::add(vec![1, 2, 3]),
        ];
        let stats = DratStats::from_clauses(&clauses);
        // Histogram: [0]=0, [1]=1, [2]=2, [3]=1
        assert_eq!(stats.size_histogram.len(), 4);
        assert_eq!(stats.size_histogram[1], 1);
        assert_eq!(stats.size_histogram[2], 2);
        assert_eq!(stats.size_histogram[3], 1);
    }

    #[test]
    fn test_drat_stats_default() {
        let stats = DratStats::default();
        assert_eq!(stats.additions, 0);
        assert_eq!(stats.deletions, 0);
        assert!(stats.size_histogram.is_empty());
    }

    #[test]
    fn test_drat_stats_clone() {
        let clauses = vec![DratClause::add(vec![1, 2])];
        let stats = DratStats::from_clauses(&clauses);
        let cloned = stats.clone();
        assert_eq!(stats.additions, cloned.additions);
        assert_eq!(stats.deletions, cloned.deletions);
    }

    #[test]
    fn test_drat_stats_debug() {
        let stats = DratStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("DratStats"));
    }

    // ==================== DratError Tests ====================

    #[test]
    fn test_drat_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let drat_err: DratError = io_err.into();
        let msg = format!("{}", drat_err);
        assert!(msg.contains("I/O error"));
    }

    #[test]
    fn test_drat_error_parse() {
        let err = DratError::Parse {
            line: 5,
            message: "invalid literal".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("line 5"));
        assert!(msg.contains("invalid literal"));
    }

    #[test]
    fn test_drat_error_binary() {
        let err = DratError::BinaryFormat { offset: 100 };
        let msg = format!("{}", err);
        assert!(msg.contains("byte 100"));
    }

    // ==================== File Parsing Tests ====================

    #[test]
    fn test_parse_file_empty() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("empty.drat");
        std::fs::write(&path, "").unwrap();

        let clauses = DratParser::parse_file(&path).unwrap();
        assert!(clauses.is_empty());
    }

    #[test]
    fn test_parse_file_text() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.drat");
        std::fs::write(&path, "1 2 0\n-3 -4 0\n").unwrap();

        let clauses = DratParser::parse_file(&path).unwrap();
        assert_eq!(clauses.len(), 2);
        assert_eq!(clauses[0].literals, vec![1, 2]);
        assert_eq!(clauses[1].literals, vec![-3, -4]);
    }

    #[test]
    fn test_parse_file_not_found() {
        let result = DratParser::parse_file(Path::new("/nonexistent/path/to/file.drat"));
        assert!(result.is_err());
        if let Err(DratError::Io(_)) = result {
            // Expected
        } else {
            panic!("Expected Io error");
        }
    }
}
