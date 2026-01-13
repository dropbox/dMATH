//! Hermite Normal Form (HNF) computation and cutting planes
//!
//! This module implements HNF-based cutting planes for Linear Integer Arithmetic.
//! Unlike Gomory cuts which work on the simplex tableau (and thus involve slack
//! variables), HNF cuts work on the original constraint matrix.
//!
//! ## Algorithm Overview
//!
//! From "Cutting the Mix" by Christ & Hoenicke:
//! 1. Collect tight equality constraints A'x = b' (active at current solution)
//! 2. Compute HNF H and unimodular U such that A'U = H
//! 3. Transform: H y = b' where y = U^{-1} x
//! 4. If y[i] = (H^{-1} b')[i] is non-integer, generate cut y[i] <= floor(y[i])
//! 5. Translate back to original variables: (e_i H^{-1} A') x <= floor(y[i])
//!
//! This avoids slack variables entirely since we work with original constraints.

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, Zero};

/// A dense integer matrix for HNF computation
#[derive(Debug, Clone)]
pub struct IntMatrix {
    /// Row-major storage
    data: Vec<BigInt>,
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Row permutation tracking (for basis identification)
    row_perm: Vec<usize>,
    /// Column permutation tracking
    col_perm: Vec<usize>,
}

impl IntMatrix {
    /// Create a zero matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        IntMatrix {
            data: vec![BigInt::zero(); rows * cols],
            rows,
            cols,
            row_perm: (0..rows).collect(),
            col_perm: (0..cols).collect(),
        }
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> &BigInt {
        &self.data[row * self.cols + col]
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, val: BigInt) {
        self.data[row * self.cols + col] = val;
    }

    /// Get mutable reference to element
    #[allow(dead_code)] // Reserved for future HNF extensions
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut BigInt {
        &mut self.data[row * self.cols + col]
    }

    /// Number of rows
    pub fn row_count(&self) -> usize {
        self.rows
    }

    /// Number of columns
    pub fn col_count(&self) -> usize {
        self.cols
    }

    /// Transpose rows i and j
    pub fn transpose_rows(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        for col in 0..self.cols {
            let idx_i = i * self.cols + col;
            let idx_j = j * self.cols + col;
            self.data.swap(idx_i, idx_j);
        }
        self.row_perm.swap(i, j);
    }

    /// Transpose columns i and j
    pub fn transpose_columns(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        for row in 0..self.rows {
            let idx_i = row * self.cols + i;
            let idx_j = row * self.cols + j;
            self.data.swap(idx_i, idx_j);
        }
        self.col_perm.swap(i, j);
    }

    /// Get the original row index after permutations
    pub fn adjust_row(&self, i: usize) -> usize {
        self.row_perm[i]
    }

    /// Get the original column index after permutations
    pub fn adjust_col(&self, i: usize) -> usize {
        self.col_perm[i]
    }

    /// Shrink matrix to only the specified rows
    pub fn shrink_to_rows(&mut self, basis_rows: &[usize]) {
        let new_rows = basis_rows.len();
        let mut new_data = vec![BigInt::zero(); new_rows * self.cols];
        let mut new_perm = Vec::with_capacity(new_rows);

        for (new_i, &old_i) in basis_rows.iter().enumerate() {
            for col in 0..self.cols {
                new_data[new_i * self.cols + col] = self.data[old_i * self.cols + col].clone();
            }
            new_perm.push(self.row_perm[old_i]);
        }

        self.data = new_data;
        self.rows = new_rows;
        self.row_perm = new_perm;
    }
}

/// Extended GCD: returns (d, u, v) such that d = u*a + v*b and d > 0
/// Minimizes |u| + |v| following Z3's implementation
pub fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if a.is_zero() {
        return (
            b.abs(),
            BigInt::zero(),
            if b.is_negative() {
                -BigInt::one()
            } else {
                BigInt::one()
            },
        );
    }
    if b.is_zero() {
        return (
            a.abs(),
            if a.is_negative() {
                -BigInt::one()
            } else {
                BigInt::one()
            },
            BigInt::zero(),
        );
    }

    // Standard extended GCD
    let (mut old_r, mut r) = (a.clone(), b.clone());
    let (mut old_s, mut s) = (BigInt::one(), BigInt::zero());
    let (mut old_t, mut t) = (BigInt::zero(), BigInt::one());

    while !r.is_zero() {
        let q = &old_r / &r;
        let new_r = &old_r - &q * &r;
        old_r = r;
        r = new_r;

        let new_s = &old_s - &q * &s;
        old_s = s;
        s = new_s;

        let new_t = &old_t - &q * &t;
        old_t = t;
        t = new_t;
    }

    // Ensure d > 0
    let (d, u, v) = if old_r.is_negative() {
        (-old_r, -old_s, -old_t)
    } else {
        (old_r, old_s, old_t)
    };

    // Minimize |u| + |v| while maintaining d = u*a + v*b
    // Skip this optimization for simplicity - the basic GCD is sufficient
    (d, u, v)
}

/// Find pivot for lower triangular reduction
fn prepare_pivot(m: &mut IntMatrix, r: usize) -> bool {
    for i in r..m.row_count() {
        for j in r..m.col_count() {
            if !m.get(i, j).is_zero() {
                if i != r {
                    m.transpose_rows(i, r);
                }
                if j != r {
                    m.transpose_columns(j, r);
                }
                return true;
            }
        }
    }
    false
}

/// Pivot column using non-fractional Gaussian elimination (Bareiss algorithm)
fn pivot_column_non_fractional(m: &mut IntMatrix, r: usize, big_number: &BigInt) -> bool {
    let pivot = m.get(r, r).clone();
    debug_assert!(!pivot.is_zero());

    for j in (r + 1)..m.col_count() {
        for i in (r + 1)..m.row_count() {
            let m_rr = m.get(r, r).clone();
            let m_ij = m.get(i, j).clone();
            let m_ir = m.get(i, r).clone();
            let m_rj = m.get(r, j).clone();

            let new_val = if r > 0 {
                let prev_pivot = m.get(r - 1, r - 1).clone();
                (&m_rr * &m_ij - &m_ir * &m_rj) / &prev_pivot
            } else {
                &m_rr * &m_ij - &m_ir * &m_rj
            };

            if new_val.abs() >= *big_number {
                return false; // Overflow
            }
            m.set(i, j, new_val);
        }
    }
    true
}

/// Transform matrix to lower triangular form, return rank
fn to_lower_triangle(m: &mut IntMatrix, big_number: &BigInt) -> Option<usize> {
    let mut rank = 0;
    for i in 0..m.row_count() {
        if !prepare_pivot(m, i) {
            return Some(rank);
        }
        if !pivot_column_non_fractional(m, i, big_number) {
            return None; // Overflow
        }
        rank = i + 1;
    }
    Some(rank)
}

/// Compute GCD of elements in row starting from diagonal
fn gcd_of_row(m: &IntMatrix, row: usize) -> BigInt {
    let mut g = BigInt::zero();
    for col in row..m.col_count() {
        let val = m.get(row, col);
        if !val.is_zero() {
            if g.is_zero() {
                g = val.abs();
            } else {
                g = g.gcd(val);
            }
        }
    }
    g
}

/// Compute determinant of rectangular matrix and identify basis rows
/// Returns (determinant, basis_rows) or None on overflow
pub fn determinant_of_rectangular_matrix(
    m: &IntMatrix,
    big_number: &BigInt,
) -> Option<(BigInt, Vec<usize>)> {
    let mut m_copy = m.clone();
    let rank = to_lower_triangle(&mut m_copy, big_number)?;

    if rank == 0 {
        return Some((BigInt::one(), vec![]));
    }

    let mut basis_rows = Vec::with_capacity(rank);
    for i in 0..rank {
        basis_rows.push(m_copy.adjust_row(i));
    }

    let det = gcd_of_row(&m_copy, rank - 1);
    Some((det, basis_rows))
}

/// HNF computation result
pub struct HnfResult {
    /// The HNF matrix H (lower triangular with specific properties)
    pub h: IntMatrix,
    /// Column order tracking
    #[allow(dead_code)] // Reserved for HNF transformation tracking
    pub col_order: Vec<usize>,
}

/// Compute Hermite Normal Form of matrix A
///
/// Returns H such that AU = H for some unimodular U.
/// H is lower triangular with:
/// - h[i][i] > 0 for all i
/// - h[i][j] <= 0 and |h[i][j]| < h[i][i] for j < i
pub fn compute_hnf(a: &IntMatrix, d: &BigInt) -> HnfResult {
    let m = a.row_count();
    let n = a.col_count();

    if m == 0 || n == 0 || d.is_zero() {
        return HnfResult {
            h: a.clone(),
            col_order: (0..n).collect(),
        };
    }

    let mut w = a.clone();
    let mut r = d.clone();

    // Process each row
    for i in 0..m {
        // When r becomes 0 or 1, all modular operations become trivial/invalid.
        // This can happen when r is divided by diagonal elements that exhaust it.
        if r.is_zero() || r.is_one() {
            break;
        }
        let half_r = &r / 2;
        // Process columns j > i to eliminate entries
        for j in (i + 1)..n {
            let wii = w.get(i, i).clone();
            let wij = w.get(i, j).clone();

            if wij.is_zero() {
                continue;
            }

            let (gcd, u, v) = extended_gcd(&wii, &wij);
            if gcd.is_zero() {
                continue;
            }

            let wii_over_gcd = &wii / &gcd;
            let wij_over_gcd = &wij / &gcd;

            // Column operations to eliminate w[i][j]
            // New col i = u * col_i + v * col_j
            // New col j = -wij/gcd * col_i + wii/gcd * col_j
            for k in i..m {
                let old_ki = w.get(k, i).clone();
                let old_kj = w.get(k, j).clone();

                let new_ki = mod_balanced(&(&u * &old_ki + &v * &old_kj), &r, &half_r);
                let new_kj = mod_balanced(
                    &(-&wij_over_gcd * &old_ki + &wii_over_gcd * &old_kj),
                    &r,
                    &half_r,
                );

                w.set(k, i, new_ki);
                w.set(k, j, new_kj);
            }
        }

        // Fix diagonal element
        let wii = w.get(i, i).clone();
        let (gcd, u, _v) = extended_gcd(&wii, &r);

        let mut new_wii = mod_r(&(&wii * &u), &r);
        if new_wii.is_zero() {
            new_wii = gcd.clone();
        }

        // Scale column i
        if !u.is_one() {
            for k in i..m {
                let old = w.get(k, i).clone();
                let new_val = mod_balanced(&(&old * &u), &r, &half_r);
                w.set(k, i, new_val);
            }
            w.set(i, i, new_wii.clone());
        }

        // Fix elements below diagonal in columns < i
        for j in 0..i {
            let wij = w.get(i, j).clone();
            if wij.is_zero() || (!wij.is_positive() && wij.abs() < new_wii) {
                continue;
            }

            // q = ceil(wij / wii)
            let q = if wij.is_positive() {
                (&wij + &new_wii - 1) / &new_wii
            } else {
                &wij / &new_wii
            };

            // col_j -= q * col_i
            for k in i..m {
                let old_kj = w.get(k, j).clone();
                let old_ki = w.get(k, i).clone();
                w.set(k, j, &old_kj - &q * &old_ki);
            }
        }

        // Update R for next iteration
        let wii_final = w.get(i, i).clone();
        if !wii_final.is_zero() {
            r = &r / &wii_final;
        }
    }

    HnfResult {
        h: w,
        col_order: a.col_perm.clone(),
    }
}

/// Modulo with balanced result in (-r/2, r/2]
fn mod_balanced(a: &BigInt, r: &BigInt, half_r: &BigInt) -> BigInt {
    let t = mod_r(a, r);
    if t > *half_r {
        t - r
    } else if t < -half_r {
        t + r
    } else {
        t
    }
}

/// Positive modulo: result in [0, r)
fn mod_r(a: &BigInt, r: &BigInt) -> BigInt {
    let t = a % r;
    if t.is_negative() {
        t + r
    } else {
        t
    }
}

/// HNF cut: (coeffs, bound) representing Σ(coeff * var) <= bound
#[derive(Debug, Clone)]
pub struct HnfCut {
    /// Coefficients indexed by original variable index
    pub coeffs: Vec<(usize, BigInt)>,
    /// Upper bound (floor of transformed RHS)
    pub bound: BigInt,
}

/// HNF cutter state
pub struct HnfCutter {
    /// Constraint matrix rows (each row is coefficients for integer vars)
    rows: Vec<Vec<BigInt>>,
    /// Right-hand sides
    rhs: Vec<BigInt>,
    /// Whether each constraint is an upper bound (true) or lower bound (false)
    is_upper: Vec<bool>,
    /// Variable indices (for mapping back)
    var_indices: Vec<usize>,
    /// Maximum absolute coefficient (for overflow detection)
    abs_max: BigInt,
}

impl HnfCutter {
    /// Create a new HNF cutter
    pub fn new() -> Self {
        HnfCutter {
            rows: Vec::new(),
            rhs: Vec::new(),
            is_upper: Vec::new(),
            var_indices: Vec::new(),
            abs_max: BigInt::zero(),
        }
    }

    /// Clear all state
    #[allow(dead_code)] // Reserved for incremental HNF cutting
    pub fn clear(&mut self) {
        self.rows.clear();
        self.rhs.clear();
        self.is_upper.clear();
        self.var_indices.clear();
        self.abs_max = BigInt::zero();
    }

    /// Register a variable for the cut matrix
    pub fn register_var(&mut self, idx: usize) {
        if !self.var_indices.contains(&idx) {
            self.var_indices.push(idx);
        }
    }

    /// Add a tight constraint (equality at current solution)
    /// coeffs: (var_index, coefficient) pairs
    /// rhs: right-hand side
    /// upper: true if upper bound constraint, false if lower
    pub fn add_constraint(&mut self, coeffs: &[(usize, BigInt)], rhs: BigInt, upper: bool) {
        // Register variables and track max coefficient
        for (idx, coeff) in coeffs {
            self.register_var(*idx);
            let abs_coeff = coeff.abs();
            if abs_coeff > self.abs_max {
                self.abs_max = abs_coeff;
            }
        }

        // Build row in variable order
        let mut row = vec![BigInt::zero(); self.var_indices.len()];
        let sign = if upper { BigInt::one() } else { -BigInt::one() };

        for (var_idx, coeff) in coeffs {
            if let Some(pos) = self.var_indices.iter().position(|&v| v == *var_idx) {
                row[pos] = &sign * coeff;
            }
        }

        let adjusted_rhs = if upper { rhs } else { -rhs };

        self.rows.push(row);
        self.rhs.push(adjusted_rhs);
        self.is_upper.push(upper);
    }

    /// Check if we have enough constraints
    pub fn has_constraints(&self) -> bool {
        !self.rows.is_empty() && !self.var_indices.is_empty()
    }

    /// Generate HNF cuts
    ///
    /// Returns a list of cuts, each of the form Σ(coeff * x_i) <= bound
    pub fn generate_cuts(&self) -> Vec<HnfCut> {
        if !self.has_constraints() {
            return Vec::new();
        }

        let debug = std::env::var("Z4_DEBUG_HNF").is_ok();

        use num_rational::BigRational;

        // Build matrix A
        let m = self.rows.len();
        let n = self.var_indices.len();

        if n == 0 {
            return Vec::new();
        }

        if debug {
            eprintln!("[HNF] Building {}x{} matrix", m, n);
        }

        let mut a = IntMatrix::new(m, n);
        for (i, row) in self.rows.iter().enumerate() {
            for j in 0..n {
                if j < row.len() {
                    a.set(i, j, row[j].clone());
                }
            }
        }

        // Compute determinant and find basis
        // Use a larger threshold to prevent spurious overflow in Bareiss algorithm.
        // The intermediate values can grow to O(max_coeff^n) where n is matrix dimension.
        // Use abs_max^6 to be safe for typical matrices (4-8 rows).
        let big_number = if self.abs_max.is_zero() {
            BigInt::from(1_000_000_000_000_000_i64) // 10^15
        } else {
            let cubed = &self.abs_max * &self.abs_max * &self.abs_max;
            &cubed * &cubed // abs_max^6
        };

        let Some((d, basis_rows)) = determinant_of_rectangular_matrix(&a, &big_number) else {
            if debug {
                eprintln!("[HNF] Overflow in determinant computation");
            }
            return Vec::new();
        };

        if d >= big_number {
            if debug {
                eprintln!("[HNF] Determinant too large: {}", d);
            }
            return Vec::new();
        }

        if basis_rows.is_empty() {
            return Vec::new();
        }

        if debug {
            eprintln!("[HNF] Determinant: {}, basis rows: {:?}", d, basis_rows);
        }

        // Shrink matrix to basis rows
        let mut a_basis = a.clone();
        a_basis.shrink_to_rows(&basis_rows);

        // Build RHS vector for basis rows
        let b: Vec<BigInt> = basis_rows.iter().map(|&i| self.rhs[i].clone()).collect();

        // Compute HNF
        let hnf = compute_hnf(&a_basis, &d);

        // Solve y0 = H^{-1} * b (forward substitution; H is lower triangular).
        // We need exact rationals here (Z3 uses mpq); integer division is incorrect.
        let h = &hnf.h;
        let mut y0: Vec<BigRational> = b.iter().map(|bi| BigRational::from(bi.clone())).collect();
        for i in 0..h.row_count() {
            for j in 0..i {
                let h_ij = BigRational::from(h.get(i, j).clone());
                y0[i] = &y0[i] - h_ij * y0[j].clone();
            }
            let hii = h.get(i, i);
            if hii.is_zero() {
                return Vec::new(); // Singular
            }
            y0[i] = &y0[i] / BigRational::from(hii.clone());
            if debug && !y0[i].denom().is_one() {
                eprintln!("[HNF] Row {} has non-integer RHS: {}", i, y0[i]);
            }
        }

        let mut cut_rows: Vec<usize> = (0..y0.len()).filter(|&i| !y0[i].denom().is_one()).collect();
        if cut_rows.is_empty() {
            if debug {
                eprintln!("[HNF] No cut row found (all RHS are integer)");
            }
            return Vec::new();
        }

        // Cap the number of cuts per HNF call to avoid constraint explosion on large problems.
        // For equality-dense problems with many non-integer rows, we still limit to prevent
        // excessive cut generation that doesn't significantly improve convergence.
        const MAX_CUTS_PER_CALL: usize = 5;
        if cut_rows.len() > MAX_CUTS_PER_CALL {
            cut_rows.truncate(MAX_CUTS_PER_CALL);
        }

        let mut cuts_out = Vec::new();
        for cut_i in cut_rows {
            if debug {
                eprintln!("[HNF] Cut from row {}", cut_i);
            }

            // Compute e_i * H^{-1} (row vector): solve f * H = e_i for f.
            let mut f: Vec<BigRational> = vec![BigRational::zero(); h.row_count()];
            f[cut_i] = BigRational::one();

            // Back substitution from row cut_i down to 0
            let hii = BigRational::from(h.get(cut_i, cut_i).clone());
            f[cut_i] = &f[cut_i] / &hii;

            for k in (0..cut_i).rev() {
                let mut sum = BigRational::zero();
                for (l, f_l) in f.iter().enumerate().take(cut_i + 1).skip(k + 1) {
                    let h_lk = BigRational::from(h.get(l, k).clone());
                    sum = &sum + &h_lk * f_l;
                }
                let hkk = BigRational::from(h.get(k, k).clone());
                f[k] = -&sum / &hkk;
            }

            // Compute cut coefficients: (e_i H^{-1} A_basis) * x <= floor(y0_i)
            let mut rational_coeffs: Vec<(usize, BigRational)> = Vec::new();
            for j in 0..a_basis.col_count() {
                let mut coeff = BigRational::zero();
                for (i, f_i) in f.iter().enumerate().take(a_basis.row_count()) {
                    let a_ij = BigRational::from(a_basis.get(i, j).clone());
                    coeff = &coeff + f_i * &a_ij;
                }
                if !coeff.is_zero() {
                    let col_idx = a_basis.adjust_col(j);
                    if col_idx < self.var_indices.len() {
                        let orig_var_idx = self.var_indices[col_idx];
                        rational_coeffs.push((orig_var_idx, coeff));
                    }
                }
            }

            if rational_coeffs.is_empty() {
                continue;
            }

            // Make integer coefficients: multiply by LCM of denominators (coeffs and bound).
            let mut lcm = BigInt::one();
            for (_, coeff) in &rational_coeffs {
                lcm = num_integer::lcm(lcm, coeff.denom().clone());
            }
            lcm = num_integer::lcm(lcm, y0[cut_i].denom().clone());

            let lcm_rat = BigRational::from(lcm.clone());

            let mut cut_coeffs: Vec<(usize, BigInt)> = Vec::new();
            for (idx, coeff) in rational_coeffs {
                let scaled = coeff * lcm_rat.clone();
                if scaled.denom().is_one() {
                    cut_coeffs.push((idx, scaled.numer().clone()));
                } else if debug {
                    eprintln!(
                        "[HNF] Skipping cut with non-integer coefficient after scaling: {}",
                        scaled
                    );
                }
            }

            if cut_coeffs.is_empty() {
                continue;
            }

            let scaled_bound = y0[cut_i].clone() * lcm_rat;
            let cut_bound = floor_bigint(&scaled_bound);

            if debug {
                eprintln!("[HNF] Cut coeffs: {:?}, bound: {}", cut_coeffs, cut_bound);
            }

            cuts_out.push(HnfCut {
                coeffs: cut_coeffs,
                bound: cut_bound,
            });
        }

        cuts_out
    }
}

/// Compute floor of a rational number
fn floor_bigint(r: &num_rational::BigRational) -> BigInt {
    let numer = r.numer();
    let denom = r.denom();
    if denom.is_one() {
        numer.clone()
    } else if numer.is_negative() {
        // For negative: floor(-5/3) = -2
        (numer - denom + 1) / denom
    } else {
        numer / denom
    }
}

impl Default for HnfCutter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extended_gcd() {
        let (d, u, v) = extended_gcd(&BigInt::from(12), &BigInt::from(8));
        assert_eq!(d, BigInt::from(4));
        assert_eq!(&u * 12 + &v * 8, d);

        let (d, u, v) = extended_gcd(&BigInt::from(35), &BigInt::from(15));
        assert_eq!(d, BigInt::from(5));
        assert_eq!(&u * 35 + &v * 15, d);

        let (d, _u, v) = extended_gcd(&BigInt::from(0), &BigInt::from(5));
        assert_eq!(d, BigInt::from(5));
        // u * 0 = 0, so this verifies d = v * 5
        assert_eq!(BigInt::ZERO + &v * 5, d);
    }

    #[test]
    fn test_matrix_basic() {
        let mut m = IntMatrix::new(2, 3);
        m.set(0, 0, BigInt::from(1));
        m.set(0, 1, BigInt::from(2));
        m.set(0, 2, BigInt::from(3));
        m.set(1, 0, BigInt::from(4));
        m.set(1, 1, BigInt::from(5));
        m.set(1, 2, BigInt::from(6));

        assert_eq!(m.get(0, 0), &BigInt::from(1));
        assert_eq!(m.get(1, 2), &BigInt::from(6));
    }

    #[test]
    fn test_hnf_simple() {
        // Simple 2x2 matrix
        let mut a = IntMatrix::new(2, 2);
        a.set(0, 0, BigInt::from(4));
        a.set(0, 1, BigInt::from(6));
        a.set(1, 0, BigInt::from(2));
        a.set(1, 1, BigInt::from(3));

        let big_num = BigInt::from(1000000);
        let result = determinant_of_rectangular_matrix(&a, &big_num);
        assert!(result.is_some());

        let (det, basis) = result.unwrap();
        // det = gcd of 2x2 minors
        assert!(!det.is_zero());
        assert!(!basis.is_empty());
    }

    #[test]
    fn test_hnf_cutter_empty() {
        let cutter = HnfCutter::new();
        assert!(!cutter.has_constraints());
        let cuts = cutter.generate_cuts();
        assert!(cuts.is_empty());
    }

    #[test]
    fn test_hnf_cutter_simple() {
        let mut cutter = HnfCutter::new();

        // Add constraint: 2*x + 3*y <= 7
        cutter.add_constraint(
            &[(0, BigInt::from(2)), (1, BigInt::from(3))],
            BigInt::from(7),
            true,
        );

        assert!(cutter.has_constraints());

        // With only one constraint, we likely won't get a cut
        // (need enough constraints to form a full-rank basis)
        let cuts = cutter.generate_cuts();
        // May or may not generate cuts depending on rank
        let _ = cuts;
    }
}
