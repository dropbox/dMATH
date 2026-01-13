//! DRAT and LRAT proof generation for UNSAT certificates
//!
//! ## DRAT
//!
//! DRAT (Deletion Resolution Asymmetric Tautology) proofs provide a way to verify
//! that an UNSAT result is correct. Every learned clause and deletion is logged,
//! and can be verified by tools like drat-trim.
//!
//! ## LRAT
//!
//! LRAT (Linear Resolution Asymmetric Tautology) extends DRAT with clause IDs
//! and resolution hints. This enables linear-time proof checking (vs quadratic
//! for DRAT), making verification much faster for large proofs.
//!
//! ## Format
//!
//! **DRAT Text format:**
//! ```text
//! 1 2 -3 0        # add clause (1 OR 2 OR -3)
//! d 1 2 0         # delete clause (1 OR 2)
//! ```
//!
//! **DRAT Binary format:**
//! - 'a' byte (0x61) followed by literals then 0 for addition
//! - 'd' byte (0x64) followed by literals then 0 for deletion
//! - Literals encoded as LEB128-style variable-length integers
//!
//! **LRAT Text format:**
//! ```text
//! 4 1 2 0 1 2 3 0    # add clause 4: (1 OR 2) with hints 1,2,3
//! 5 d 1 2 0          # delete clauses 1 and 2 (latest ID is 5)
//! ```
//!
//! **LRAT Binary format:**
//! - 'a' byte followed by binary_id, binary_lits..., 0, binary_hints..., 0
//! - 'd' byte followed by binary_ids..., 0

use crate::literal::Literal;
use std::io::{self, Write};

/// DRAT proof writer for generating UNSAT certificates
pub struct DratWriter<W: Write> {
    writer: W,
    binary: bool,
    /// Count of clauses added
    added_count: u64,
    /// Count of clauses deleted
    deleted_count: u64,
}

impl<W: Write> DratWriter<W> {
    /// Create a new DRAT writer with text format
    pub fn new_text(writer: W) -> Self {
        DratWriter {
            writer,
            binary: false,
            added_count: 0,
            deleted_count: 0,
        }
    }

    /// Create a new DRAT writer with binary format
    pub fn new_binary(writer: W) -> Self {
        DratWriter {
            writer,
            binary: true,
            added_count: 0,
            deleted_count: 0,
        }
    }

    /// Log addition of a learned clause
    pub fn add(&mut self, clause: &[Literal]) -> io::Result<()> {
        self.added_count += 1;
        if self.binary {
            self.write_binary_clause(clause, false)
        } else {
            self.write_text_clause(clause, false)
        }
    }

    /// Log deletion of a clause
    pub fn delete(&mut self, clause: &[Literal]) -> io::Result<()> {
        self.deleted_count += 1;
        if self.binary {
            self.write_binary_clause(clause, true)
        } else {
            self.write_text_clause(clause, true)
        }
    }

    /// Write clause in text format
    fn write_text_clause(&mut self, clause: &[Literal], is_delete: bool) -> io::Result<()> {
        if is_delete {
            write!(self.writer, "d ")?;
        }
        for lit in clause {
            write!(self.writer, "{} ", lit.to_dimacs())?;
        }
        writeln!(self.writer, "0")
    }

    /// Write clause in binary format
    fn write_binary_clause(&mut self, clause: &[Literal], is_delete: bool) -> io::Result<()> {
        // Write marker byte
        self.writer
            .write_all(&[if is_delete { b'd' } else { b'a' }])?;

        // Write each literal in binary encoding
        for lit in clause {
            self.write_binary_lit(*lit)?;
        }

        // Write terminating 0
        self.writer.write_all(&[0])
    }

    /// Write a literal in binary (variable-length) encoding
    ///
    /// Binary literal encoding: positive lit v -> 2*v, negative lit v -> 2*v+1
    /// Then encoded as variable-length integer
    fn write_binary_lit(&mut self, lit: Literal) -> io::Result<()> {
        // DIMACS-style encoding: var+1 for 1-indexed, then sign
        let var = lit.variable().0 + 1; // 1-indexed
        let encoded = if lit.is_positive() {
            2 * var
        } else {
            2 * var + 1
        };

        // Variable-length encoding (similar to LEB128)
        let mut val = encoded;
        while val > 127 {
            self.writer.write_all(&[(val as u8 & 0x7f) | 0x80])?;
            val >>= 7;
        }
        self.writer.write_all(&[val as u8])
    }

    /// Get the number of clauses added
    pub fn added_count(&self) -> u64 {
        self.added_count
    }

    /// Get the number of clauses deleted
    pub fn deleted_count(&self) -> u64 {
        self.deleted_count
    }

    /// Flush the writer
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    /// Get the inner writer back
    pub fn into_inner(self) -> W {
        self.writer
    }
}

/// Extension trait for literals to convert to DIMACS format
trait ToDimacs {
    /// Convert to DIMACS format (1-indexed, negative for negated)
    fn to_dimacs(&self) -> i32;
}

impl ToDimacs for Literal {
    fn to_dimacs(&self) -> i32 {
        let var = self.variable().0 as i32 + 1; // 1-indexed
        if self.is_positive() {
            var
        } else {
            -var
        }
    }
}

/// LRAT proof writer for generating UNSAT certificates with clause IDs and hints
///
/// LRAT proofs include clause IDs and resolution hints, enabling linear-time
/// proof checking. Each added clause includes:
/// - A unique clause ID
/// - The clause literals
/// - Hint IDs (clause IDs used to derive this clause)
pub struct LratWriter<W: Write> {
    writer: W,
    binary: bool,
    /// Next clause ID to assign
    next_id: u64,
    /// ID of the most recently added clause (for deletion batching)
    latest_id: u64,
    /// Pending deletions (batched for efficiency)
    pending_deletions: Vec<u64>,
    /// Count of clauses added
    added_count: u64,
    /// Count of clauses deleted
    deleted_count: u64,
}

impl<W: Write> LratWriter<W> {
    /// Create a new LRAT writer with text format
    ///
    /// `num_original_clauses` is the number of clauses in the original formula.
    /// The first learned clause will get ID `num_original_clauses + 1`.
    pub fn new_text(writer: W, num_original_clauses: u64) -> Self {
        LratWriter {
            writer,
            binary: false,
            next_id: num_original_clauses + 1,
            latest_id: num_original_clauses,
            pending_deletions: Vec::new(),
            added_count: 0,
            deleted_count: 0,
        }
    }

    /// Create a new LRAT writer with binary format
    ///
    /// `num_original_clauses` is the number of clauses in the original formula.
    pub fn new_binary(writer: W, num_original_clauses: u64) -> Self {
        LratWriter {
            writer,
            binary: true,
            next_id: num_original_clauses + 1,
            latest_id: num_original_clauses,
            pending_deletions: Vec::new(),
            added_count: 0,
            deleted_count: 0,
        }
    }

    /// Flush any pending deletions to the output
    fn flush_deletions(&mut self) -> io::Result<()> {
        if self.pending_deletions.is_empty() {
            return Ok(());
        }

        if self.binary {
            self.writer.write_all(b"d")?;
            // Take the pending deletions to avoid borrow conflict
            let deletions = std::mem::take(&mut self.pending_deletions);
            for id in &deletions {
                self.write_binary_id(*id)?;
            }
            self.writer.write_all(&[0])?;
        } else {
            // Text format: "latest_id d id1 id2 ... 0"
            write!(self.writer, "{} d ", self.latest_id)?;
            for &id in &self.pending_deletions {
                write!(self.writer, "{} ", id)?;
            }
            writeln!(self.writer, "0")?;
            self.pending_deletions.clear();
        }

        Ok(())
    }

    /// Log addition of a learned clause with resolution hints
    ///
    /// Returns the assigned clause ID.
    ///
    /// # Arguments
    /// * `clause` - The literals in the learned clause
    /// * `hints` - Clause IDs used to derive this clause (resolution chain)
    pub fn add(&mut self, clause: &[Literal], hints: &[u64]) -> io::Result<u64> {
        // Flush any pending deletions first
        self.flush_deletions()?;

        let id = self.next_id;
        self.next_id += 1;
        self.latest_id = id;
        self.added_count += 1;

        if self.binary {
            self.write_binary_add(id, clause, hints)
        } else {
            self.write_text_add(id, clause, hints)
        }?;

        Ok(id)
    }

    /// Write addition in text format: "id lit1 lit2 ... 0 hint1 hint2 ... 0"
    fn write_text_add(&mut self, id: u64, clause: &[Literal], hints: &[u64]) -> io::Result<()> {
        write!(self.writer, "{} ", id)?;
        for lit in clause {
            write!(self.writer, "{} ", lit.to_dimacs())?;
        }
        write!(self.writer, "0 ")?;
        for &hint in hints {
            write!(self.writer, "{} ", hint)?;
        }
        writeln!(self.writer, "0")
    }

    /// Write addition in binary format
    fn write_binary_add(&mut self, id: u64, clause: &[Literal], hints: &[u64]) -> io::Result<()> {
        self.writer.write_all(b"a")?;
        self.write_binary_id(id)?;
        for lit in clause {
            self.write_binary_lit(*lit)?;
        }
        self.writer.write_all(&[0])?;
        for &hint in hints {
            self.write_binary_id(hint)?;
        }
        self.writer.write_all(&[0])
    }

    /// Log deletion of a clause by ID
    ///
    /// Deletions are batched for efficiency and flushed on the next add.
    pub fn delete(&mut self, clause_id: u64) -> io::Result<()> {
        self.deleted_count += 1;
        self.pending_deletions.push(clause_id);
        Ok(())
    }

    /// Write a literal in binary encoding (same as DRAT)
    fn write_binary_lit(&mut self, lit: Literal) -> io::Result<()> {
        let var = lit.variable().0 + 1; // 1-indexed
        let encoded = if lit.is_positive() {
            2 * var
        } else {
            2 * var + 1
        };

        let mut val = encoded;
        while val > 127 {
            self.writer.write_all(&[(val as u8 & 0x7f) | 0x80])?;
            val >>= 7;
        }
        self.writer.write_all(&[val as u8])
    }

    /// Write a clause ID in binary encoding (variable-length)
    fn write_binary_id(&mut self, id: u64) -> io::Result<()> {
        // IDs are always positive, so we encode directly (not 2*id like literals)
        let mut val = id;
        while val > 127 {
            self.writer.write_all(&[(val as u8 & 0x7f) | 0x80])?;
            val >>= 7;
        }
        self.writer.write_all(&[val as u8])
    }

    /// Get the next clause ID that will be assigned
    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    /// Get the number of original clauses
    pub fn num_original_clauses(&self) -> u64 {
        // First learned clause ID is num_original + 1, so subtract 1 from initial next_id
        // But we track next_id which increments, so this is latest_id at start
        // Actually we can compute from next_id - added_count - 1
        self.next_id - self.added_count - 1
    }

    /// Get the number of clauses added
    pub fn added_count(&self) -> u64 {
        self.added_count
    }

    /// Get the number of clauses deleted
    pub fn deleted_count(&self) -> u64 {
        self.deleted_count
    }

    /// Flush the writer (including any pending deletions)
    pub fn flush(&mut self) -> io::Result<()> {
        self.flush_deletions()?;
        self.writer.flush()
    }

    /// Finalize the proof by flushing pending deletions and returning the inner writer
    pub fn into_inner(mut self) -> io::Result<W> {
        self.flush_deletions()?;
        Ok(self.writer)
    }
}

/// Unified proof output that can be either DRAT or LRAT format
///
/// This enum allows the solver to write proofs in either format while maintaining
/// a single proof_writer field. LRAT proofs include clause IDs and resolution hints
/// for linear-time verification.
pub enum ProofOutput<W: Write> {
    /// DRAT proof format (no hints, clause-based deletions)
    Drat(DratWriter<W>),
    /// LRAT proof format (with hints, ID-based deletions)
    Lrat(LratWriter<W>),
}

impl<W: Write> ProofOutput<W> {
    /// Create a new DRAT text proof output
    pub fn drat_text(writer: W) -> Self {
        ProofOutput::Drat(DratWriter::new_text(writer))
    }

    /// Create a new DRAT binary proof output
    pub fn drat_binary(writer: W) -> Self {
        ProofOutput::Drat(DratWriter::new_binary(writer))
    }

    /// Create a new LRAT text proof output
    pub fn lrat_text(writer: W, num_original_clauses: u64) -> Self {
        ProofOutput::Lrat(LratWriter::new_text(writer, num_original_clauses))
    }

    /// Create a new LRAT binary proof output
    pub fn lrat_binary(writer: W, num_original_clauses: u64) -> Self {
        ProofOutput::Lrat(LratWriter::new_binary(writer, num_original_clauses))
    }

    /// Check if this is an LRAT proof
    pub fn is_lrat(&self) -> bool {
        matches!(self, ProofOutput::Lrat(_))
    }

    /// Add a learned clause to the proof
    ///
    /// For DRAT, hints are ignored. For LRAT, hints are the clause IDs used
    /// to derive this clause. Returns the clause ID (for LRAT) or 0 (for DRAT).
    pub fn add(&mut self, clause: &[Literal], hints: &[u64]) -> io::Result<u64> {
        match self {
            ProofOutput::Drat(w) => {
                w.add(clause)?;
                Ok(0)
            }
            ProofOutput::Lrat(w) => w.add(clause, hints),
        }
    }

    /// Delete a clause from the proof
    ///
    /// For DRAT, uses the clause literals. For LRAT, uses the clause ID.
    pub fn delete(&mut self, clause: &[Literal], clause_id: u64) -> io::Result<()> {
        match self {
            ProofOutput::Drat(w) => w.delete(clause),
            ProofOutput::Lrat(w) => w.delete(clause_id),
        }
    }

    /// Flush the proof writer
    pub fn flush(&mut self) -> io::Result<()> {
        match self {
            ProofOutput::Drat(w) => w.flush(),
            ProofOutput::Lrat(w) => w.flush(),
        }
    }

    /// Get the number of clauses added
    pub fn added_count(&self) -> u64 {
        match self {
            ProofOutput::Drat(w) => w.added_count(),
            ProofOutput::Lrat(w) => w.added_count(),
        }
    }

    /// Get the number of clauses deleted
    pub fn deleted_count(&self) -> u64 {
        match self {
            ProofOutput::Drat(w) => w.deleted_count(),
            ProofOutput::Lrat(w) => w.deleted_count(),
        }
    }

    /// Get the inner writer back, consuming the ProofOutput
    pub fn into_inner(self) -> W {
        match self {
            ProofOutput::Drat(w) => w.into_inner(),
            ProofOutput::Lrat(w) => w.into_inner().expect("Failed to flush LRAT writer"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Variable;

    #[test]
    fn test_text_add_clause() {
        let mut buf = Vec::new();
        let mut writer = DratWriter::new_text(&mut buf);

        let clause = vec![
            Literal::positive(Variable(0)),
            Literal::negative(Variable(1)),
            Literal::positive(Variable(2)),
        ];
        writer.add(&clause).unwrap();

        let output = String::from_utf8(buf).unwrap();
        assert_eq!(output, "1 -2 3 0\n");
    }

    #[test]
    fn test_text_delete_clause() {
        let mut buf = Vec::new();
        let mut writer = DratWriter::new_text(&mut buf);

        let clause = vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ];
        writer.delete(&clause).unwrap();

        let output = String::from_utf8(buf).unwrap();
        assert_eq!(output, "d 1 2 0\n");
    }

    #[test]
    fn test_binary_add_clause() {
        let mut buf = Vec::new();
        let mut writer = DratWriter::new_binary(&mut buf);

        // Single variable clause: literal +x0 -> var 1 (1-indexed), encoded as 2*1=2
        let clause = vec![Literal::positive(Variable(0))];
        writer.add(&clause).unwrap();

        // Expected: 'a' (0x61), then 2 (the encoded literal), then 0
        assert_eq!(buf, vec![0x61, 2, 0]);
    }

    #[test]
    fn test_binary_delete_clause() {
        let mut buf = Vec::new();
        let mut writer = DratWriter::new_binary(&mut buf);

        // Literal -x0 -> var 1 (1-indexed), encoded as 2*1+1=3
        let clause = vec![Literal::negative(Variable(0))];
        writer.delete(&clause).unwrap();

        // Expected: 'd' (0x64), then 3, then 0
        assert_eq!(buf, vec![0x64, 3, 0]);
    }

    #[test]
    fn test_counts() {
        let mut buf = Vec::new();
        let mut writer = DratWriter::new_text(&mut buf);

        let clause1 = vec![Literal::positive(Variable(0))];
        let clause2 = vec![Literal::positive(Variable(1))];

        writer.add(&clause1).unwrap();
        writer.add(&clause2).unwrap();
        writer.delete(&clause1).unwrap();

        assert_eq!(writer.added_count(), 2);
        assert_eq!(writer.deleted_count(), 1);
    }

    #[test]
    fn test_empty_clause() {
        let mut buf = Vec::new();
        let mut writer = DratWriter::new_text(&mut buf);

        // Empty clause (used to indicate final conflict)
        writer.add(&[]).unwrap();

        let output = String::from_utf8(buf).unwrap();
        assert_eq!(output, "0\n");
    }

    // LRAT tests

    #[test]
    fn test_lrat_text_add_clause() {
        let mut buf = Vec::new();
        let mut writer = LratWriter::new_text(&mut buf, 3); // 3 original clauses

        let clause = vec![
            Literal::positive(Variable(0)),
            Literal::negative(Variable(1)),
        ];
        let hints = vec![1, 2, 3]; // derived from clauses 1, 2, 3
        let id = writer.add(&clause, &hints).unwrap();

        assert_eq!(id, 4); // First learned clause gets ID 4
        let output = String::from_utf8(buf).unwrap();
        assert_eq!(output, "4 1 -2 0 1 2 3 0\n");
    }

    #[test]
    fn test_lrat_text_multiple_adds() {
        let mut buf = Vec::new();
        let mut writer = LratWriter::new_text(&mut buf, 2); // 2 original clauses

        let clause1 = vec![Literal::positive(Variable(0))];
        let clause2 = vec![Literal::negative(Variable(1))];

        let id1 = writer.add(&clause1, &[1, 2]).unwrap();
        let id2 = writer.add(&clause2, &[1, 3]).unwrap();

        assert_eq!(id1, 3);
        assert_eq!(id2, 4);

        let output = String::from_utf8(buf).unwrap();
        assert_eq!(output, "3 1 0 1 2 0\n4 -2 0 1 3 0\n");
    }

    #[test]
    fn test_lrat_text_delete() {
        let mut buf = Vec::new();
        let mut writer = LratWriter::new_text(&mut buf, 2);

        // Add a clause first
        let clause = vec![Literal::positive(Variable(0))];
        writer.add(&clause, &[1, 2]).unwrap();

        // Delete clause 1
        writer.delete(1).unwrap();

        // Deletions are batched, flush on next add or flush
        let clause2 = vec![Literal::negative(Variable(0))];
        writer.add(&clause2, &[2]).unwrap();

        let output = String::from_utf8(buf).unwrap();
        // First add, then deletion batch, then second add
        assert_eq!(output, "3 1 0 1 2 0\n3 d 1 0\n4 -1 0 2 0\n");
    }

    #[test]
    fn test_lrat_binary_add_clause() {
        let mut buf = Vec::new();
        let mut writer = LratWriter::new_binary(&mut buf, 2);

        // Add clause with literal +x0 (encoded as 2) and hints [1, 2]
        let clause = vec![Literal::positive(Variable(0))];
        let id = writer.add(&clause, &[1, 2]).unwrap();

        assert_eq!(id, 3);
        // Expected: 'a' (0x61), id=3, lit=2, 0, hint=1, hint=2, 0
        assert_eq!(buf, vec![0x61, 3, 2, 0, 1, 2, 0]);
    }

    #[test]
    fn test_lrat_binary_delete() {
        let mut buf = Vec::new();
        let mut writer = LratWriter::new_binary(&mut buf, 2);

        // Add a clause
        let clause = vec![Literal::positive(Variable(0))];
        writer.add(&clause, &[1]).unwrap();

        // Delete clauses 1 and 2
        writer.delete(1).unwrap();
        writer.delete(2).unwrap();

        // Flush to write deletions
        writer.flush().unwrap();

        // Expected: add (3 1 0 1 0), then delete (d 1 2 0)
        // 'a'=0x61, id=3, lit=2, 0, hint=1, 0
        // 'd'=0x64, id=1, id=2, 0
        assert_eq!(buf, vec![0x61, 3, 2, 0, 1, 0, 0x64, 1, 2, 0]);
    }

    #[test]
    fn test_lrat_empty_hints() {
        let mut buf = Vec::new();
        let mut writer = LratWriter::new_text(&mut buf, 1);

        // Empty clause with no hints (final conflict)
        let id = writer.add(&[], &[]).unwrap();

        assert_eq!(id, 2);
        let output = String::from_utf8(buf).unwrap();
        assert_eq!(output, "2 0 0\n");
    }

    #[test]
    fn test_lrat_counts() {
        let mut buf = Vec::new();
        let mut writer = LratWriter::new_text(&mut buf, 5);

        let clause = vec![Literal::positive(Variable(0))];
        writer.add(&clause, &[1]).unwrap();
        writer.add(&clause, &[2]).unwrap();
        writer.delete(1).unwrap();
        writer.delete(2).unwrap();

        assert_eq!(writer.added_count(), 2);
        assert_eq!(writer.deleted_count(), 2);
        assert_eq!(writer.next_id(), 8); // 5 original + 2 added = next is 8
    }
}
