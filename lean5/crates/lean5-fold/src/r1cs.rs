//! R1CS (Rank-1 Constraint System) representation
//!
//! Standard R1CS represents a computation as:
//!   Az ∘ Bz = Cz
//!
//! where A, B, C are sparse matrices, z is the concatenation of
//! (1, public inputs, private witness), and ∘ is element-wise product.

use crate::{error::FoldError, Scalar};
use ark_ff::Zero;

/// A sparse matrix in CSR (Compressed Sparse Row) format
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseMatrix {
    /// Number of rows
    pub num_rows: usize,
    /// Number of columns
    pub num_cols: usize,
    /// Row pointers (length = num_rows + 1)
    pub row_ptr: Vec<usize>,
    /// Column indices for non-zero entries
    pub col_idx: Vec<usize>,
    /// Values of non-zero entries
    pub values: Vec<Scalar>,
}

impl SparseMatrix {
    /// Create a new empty sparse matrix
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            num_rows,
            num_cols,
            row_ptr: vec![0; num_rows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create a sparse matrix from triplets (row, col, value)
    pub fn from_triplets(
        num_rows: usize,
        num_cols: usize,
        triplets: &[(usize, usize, Scalar)],
    ) -> Self {
        // Sort triplets by row, then by column
        let mut sorted = triplets.to_vec();
        sorted.sort_by_key(|(r, c, _)| (*r, *c));

        let mut row_ptr = vec![0usize; num_rows + 1];
        let mut col_idx = Vec::with_capacity(triplets.len());
        let mut values = Vec::with_capacity(triplets.len());

        for (r, c, v) in sorted {
            row_ptr[r + 1] += 1;
            col_idx.push(c);
            values.push(v);
        }

        // Convert counts to cumulative offsets
        for i in 0..num_rows {
            row_ptr[i + 1] += row_ptr[i];
        }

        Self {
            num_rows,
            num_cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Multiply matrix by vector: result = M * v
    pub fn mul_vec(&self, v: &[Scalar]) -> Result<Vec<Scalar>, FoldError> {
        if v.len() != self.num_cols {
            return Err(FoldError::DimensionMismatch(format!(
                "matrix has {} cols, vector has {} elements",
                self.num_cols,
                v.len()
            )));
        }

        let mut result = vec![Scalar::zero(); self.num_rows];

        for (row, result_val) in result.iter_mut().enumerate() {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for idx in start..end {
                *result_val += self.values[idx] * v[self.col_idx[idx]];
            }
        }

        Ok(result)
    }

    /// Get the number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// Shape of an R1CS instance (defines the constraint structure)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1CSShape {
    /// Number of constraints
    pub num_constraints: usize,
    /// Number of public inputs (not counting the constant 1)
    pub num_io: usize,
    /// Number of private witness elements
    pub num_vars: usize,
    /// A matrix
    pub a: SparseMatrix,
    /// B matrix
    pub b: SparseMatrix,
    /// C matrix
    pub c: SparseMatrix,
}

impl R1CSShape {
    /// Total number of variables in z = (1, io, W)
    pub fn num_z(&self) -> usize {
        1 + self.num_io + self.num_vars
    }

    /// Create a new R1CS shape from matrices
    pub fn new(
        num_constraints: usize,
        num_io: usize,
        num_vars: usize,
        a: SparseMatrix,
        b: SparseMatrix,
        c: SparseMatrix,
    ) -> Result<Self, FoldError> {
        let num_z = 1 + num_io + num_vars;

        // Validate matrix dimensions
        if a.num_rows != num_constraints || a.num_cols != num_z {
            return Err(FoldError::DimensionMismatch(
                "A matrix dimensions mismatch".to_string(),
            ));
        }
        if b.num_rows != num_constraints || b.num_cols != num_z {
            return Err(FoldError::DimensionMismatch(
                "B matrix dimensions mismatch".to_string(),
            ));
        }
        if c.num_rows != num_constraints || c.num_cols != num_z {
            return Err(FoldError::DimensionMismatch(
                "C matrix dimensions mismatch".to_string(),
            ));
        }

        Ok(Self {
            num_constraints,
            num_io,
            num_vars,
            a,
            b,
            c,
        })
    }

    /// Check if a witness satisfies the R1CS relation
    pub fn is_satisfied(
        &self,
        instance: &R1CSInstance,
        witness: &R1CSWitness,
    ) -> Result<bool, FoldError> {
        let z = self.build_z(instance, witness);

        let az = self.a.mul_vec(&z)?;
        let bz = self.b.mul_vec(&z)?;
        let cz = self.c.mul_vec(&z)?;

        // Check Az ∘ Bz = Cz for each constraint
        for i in 0..self.num_constraints {
            if az[i] * bz[i] != cz[i] {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Build the z vector from instance and witness
    pub fn build_z(&self, instance: &R1CSInstance, witness: &R1CSWitness) -> Vec<Scalar> {
        let mut z = Vec::with_capacity(self.num_z());
        z.push(Scalar::from(1u64)); // constant
        z.extend_from_slice(&instance.x); // public inputs
        z.extend_from_slice(&witness.w); // private witness
        z
    }
}

/// Public instance for R1CS (public inputs)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1CSInstance {
    /// Public inputs
    pub x: Vec<Scalar>,
}

impl R1CSInstance {
    /// Create a new instance with given public inputs
    pub fn new(x: Vec<Scalar>) -> Self {
        Self { x }
    }
}

/// Private witness for R1CS
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1CSWitness {
    /// Private witness elements
    pub w: Vec<Scalar>,
}

impl R1CSWitness {
    /// Create a new witness
    pub fn new(w: Vec<Scalar>) -> Self {
        Self { w }
    }
}

/// Linear combination: a sum of (index, coefficient) pairs
pub type LinearCombination = Vec<(usize, Scalar)>;

/// An R1CS constraint: A * B = C where each is a linear combination
pub type R1CSConstraint = (LinearCombination, LinearCombination, LinearCombination);

/// Builder for constructing R1CS constraints
#[derive(Debug)]
pub struct R1CSBuilder {
    constraints: Vec<R1CSConstraint>,
    num_io: usize,
    num_vars: usize,
    next_var: usize,
}

impl R1CSBuilder {
    /// Create a new R1CS builder
    pub fn new(num_io: usize) -> Self {
        Self {
            constraints: Vec::new(),
            num_io,
            num_vars: 0,
            next_var: 0,
        }
    }

    /// Allocate a new private variable, returns its index in the witness
    pub fn alloc_var(&mut self) -> usize {
        let idx = self.next_var;
        self.next_var += 1;
        self.num_vars = self.num_vars.max(self.next_var);
        idx
    }

    /// Add a constraint: (sum a_i * z_i) * (sum b_i * z_i) = (sum c_i * z_i)
    ///
    /// Indices are into the full z vector: 0 = constant 1, 1..=num_io = public inputs,
    /// num_io+1.. = private witness
    pub fn add_constraint(
        &mut self,
        a: Vec<(usize, Scalar)>,
        b: Vec<(usize, Scalar)>,
        c: Vec<(usize, Scalar)>,
    ) {
        self.constraints.push((a, b, c));
    }

    /// Index of the constant 1 in z
    pub fn const_idx(&self) -> usize {
        0
    }

    /// Index of public input i (0-indexed) in z
    pub fn io_idx(&self, i: usize) -> usize {
        1 + i
    }

    /// Index of private witness variable i (0-indexed) in z
    pub fn var_idx(&self, i: usize) -> usize {
        1 + self.num_io + i
    }

    /// Get the current number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Build the R1CS shape from constraints
    pub fn build(self) -> Result<R1CSShape, FoldError> {
        let num_constraints = self.constraints.len();
        let num_z = 1 + self.num_io + self.num_vars;

        let mut a_triplets = Vec::new();
        let mut b_triplets = Vec::new();
        let mut c_triplets = Vec::new();

        for (row, (a, b, c)) in self.constraints.into_iter().enumerate() {
            for (col, val) in a {
                a_triplets.push((row, col, val));
            }
            for (col, val) in b {
                b_triplets.push((row, col, val));
            }
            for (col, val) in c {
                c_triplets.push((row, col, val));
            }
        }

        let a = SparseMatrix::from_triplets(num_constraints, num_z, &a_triplets);
        let b = SparseMatrix::from_triplets(num_constraints, num_z, &b_triplets);
        let c = SparseMatrix::from_triplets(num_constraints, num_z, &c_triplets);

        R1CSShape::new(num_constraints, self.num_io, self.num_vars, a, b, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matrix_mul() {
        // Create a 2x3 matrix:
        // [1 2 0]
        // [0 3 4]
        let triplets = vec![
            (0, 0, Scalar::from(1u64)),
            (0, 1, Scalar::from(2u64)),
            (1, 1, Scalar::from(3u64)),
            (1, 2, Scalar::from(4u64)),
        ];
        let m = SparseMatrix::from_triplets(2, 3, &triplets);

        // Multiply by [1, 2, 3]
        let v = vec![Scalar::from(1u64), Scalar::from(2u64), Scalar::from(3u64)];
        let result = m.mul_vec(&v).unwrap();

        // Expected: [1*1 + 2*2 + 0*3, 0*1 + 3*2 + 4*3] = [5, 18]
        assert_eq!(result[0], Scalar::from(5u64));
        assert_eq!(result[1], Scalar::from(18u64));
    }

    #[test]
    fn test_r1cs_builder() {
        // Simple constraint: x * y = z where x=3, y=4, z=12
        let mut builder = R1CSBuilder::new(0); // no public inputs

        let x = builder.alloc_var();
        let y = builder.alloc_var();
        let z = builder.alloc_var();

        // Constraint: x * y = z
        builder.add_constraint(
            vec![(builder.var_idx(x), Scalar::from(1u64))],
            vec![(builder.var_idx(y), Scalar::from(1u64))],
            vec![(builder.var_idx(z), Scalar::from(1u64))],
        );

        let shape = builder.build().unwrap();

        // Test with valid witness: x=3, y=4, z=12
        let instance = R1CSInstance::new(vec![]);
        let witness = R1CSWitness::new(vec![
            Scalar::from(3u64),
            Scalar::from(4u64),
            Scalar::from(12u64),
        ]);

        assert!(shape.is_satisfied(&instance, &witness).unwrap());

        // Test with invalid witness: x=3, y=4, z=11 (wrong!)
        let bad_witness = R1CSWitness::new(vec![
            Scalar::from(3u64),
            Scalar::from(4u64),
            Scalar::from(11u64),
        ]);

        assert!(!shape.is_satisfied(&instance, &bad_witness).unwrap());
    }

    #[test]
    fn test_r1cs_with_public_inputs() {
        // Constraint: public_x * w = public_y
        // Public inputs: x=5, y=15
        // Private witness: w=3
        let mut builder = R1CSBuilder::new(2); // 2 public inputs

        let w = builder.alloc_var();

        // Constraint: io[0] * w = io[1]
        builder.add_constraint(
            vec![(builder.io_idx(0), Scalar::from(1u64))],  // A: x
            vec![(builder.var_idx(w), Scalar::from(1u64))], // B: w
            vec![(builder.io_idx(1), Scalar::from(1u64))],  // C: y
        );

        let shape = builder.build().unwrap();

        let instance = R1CSInstance::new(vec![Scalar::from(5u64), Scalar::from(15u64)]);
        let witness = R1CSWitness::new(vec![Scalar::from(3u64)]);

        assert!(shape.is_satisfied(&instance, &witness).unwrap());
    }
}
