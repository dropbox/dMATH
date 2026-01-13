//! Walk-based local search for phase initialization
//!
//! Implements CaDiCaL-style ProbSAT random walk for finding good initial phases.
//! This helps reduce heuristic variance and improves average-case performance.
//!
//! ## Walk (ProbSAT)
//! Random walk local search that:
//! 1. Starts from current phase assignment
//! 2. Counts broken (unsatisfied) clauses
//! 3. Picks random broken clause, scores literals by break-count
//! 4. Flips literal with probability proportional to score
//! 5. Saves phases at minimum broken count
//!
//! ## O(1) Amortized Complexity
//! Uses occurrence lists and satisfaction counts for efficient updates:
//! - Occurrence list: for each literal, list of clause indices containing it
//! - Satisfaction count: for each clause, number of satisfied literals
//! - Break-value: count clauses where literal is sole satisfier
//! - Flip update: only modify affected clauses (via occurrence lists)

use crate::clause_db::ClauseDB;
use crate::literal::Literal;

/// CB values for ProbSAT scoring based on average clause size.
/// From Adrian Balint's thesis on ProbSAT.
/// Higher clause sizes need higher CB (more selective scoring).
const CB_VALUES: [(f64, f64); 6] = [
    (0.0, 2.00),
    (3.0, 2.50),
    (4.0, 2.85),
    (5.0, 3.70),
    (6.0, 5.10),
    (7.0, 7.40),
];

/// Interpolate CB value for a given average clause size
fn fit_cb_value(size: f64) -> f64 {
    let mut i = 0;
    while i + 2 < CB_VALUES.len() && (CB_VALUES[i].0 > size || CB_VALUES[i + 1].0 < size) {
        i += 1;
    }
    let (x1, y1) = CB_VALUES[i];
    let (x2, y2) = CB_VALUES[i + 1];
    let dx = x2 - x1;
    let dy = y2 - y1;
    dy * (size - x1) / dx + y1
}

/// Simple linear congruential generator for random numbers.
/// Based on CaDiCaL's Random class.
struct Random {
    state: u64,
}

impl Random {
    fn new(seed: u64) -> Self {
        Random {
            state: seed.wrapping_mul(6364136223846793005).wrapping_add(1),
        }
    }

    /// Generate random u64
    pub fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Generate random integer in [0, max)
    pub fn pick(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next() % max as u64) as usize
    }

    /// Generate random double in [0, 1)
    pub fn generate_double(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }
}

/// Statistics from walk phase
#[derive(Debug, Default, Clone)]
pub struct WalkStats {
    /// Number of walk rounds executed
    pub walk_count: u64,
    /// Number of flips executed
    pub flips: u64,
    /// Total broken clauses encountered
    pub broken_total: u64,
    /// Best (minimum) broken count found
    pub best_minimum: u64,
}

/// Convert literal to occurrence list index
/// Positive literal for var v: 2*v
/// Negative literal for var v: 2*v + 1
#[inline]
fn lit_to_occurs_idx(lit: Literal) -> usize {
    let var = lit.variable().index();
    if lit.is_positive() {
        2 * var
    } else {
        2 * var + 1
    }
}

/// Walker state for local search with O(1) amortized operations
struct Walker {
    /// Random number generator
    random: Random,
    /// Current assignment (true = positive, false = negative)
    values: Vec<bool>,
    /// For each literal index, list of clause indices containing it
    occurs: Vec<Vec<usize>>,
    /// For each clause, count of satisfied literals
    sat_count: Vec<u32>,
    /// Indices of currently broken clauses (sat_count == 0)
    broken: Vec<usize>,
    /// Position of each clause in broken list (-1 if not broken)
    broken_pos: Vec<i32>,
    /// Score table (precomputed scores for break counts)
    score_table: Vec<f64>,
    /// Smallest score (for break counts beyond table size)
    epsilon: f64,
    /// Best phases found (minimum broken count)
    best_phases: Vec<bool>,
    /// Minimum broken count seen
    minimum: usize,
    /// Tick counter (for effort limiting)
    ticks: u64,
    /// Tick limit
    limit: u64,
}

impl Walker {
    /// Create new walker for n variables with given tick limit
    fn new(num_vars: usize, num_clauses: usize, seed: u64, limit: u64) -> Self {
        Walker {
            random: Random::new(seed),
            values: vec![false; num_vars],
            occurs: vec![Vec::new(); 2 * num_vars],
            sat_count: vec![0; num_clauses],
            broken: Vec::new(),
            broken_pos: vec![-1; num_clauses],
            score_table: Vec::new(),
            epsilon: 0.0,
            best_phases: vec![false; num_vars],
            minimum: usize::MAX,
            ticks: 0,
            limit,
        }
    }

    /// Populate score table based on average clause size
    fn populate_table(&mut self, avg_size: f64, walk_count: u64) {
        // Alternate between size-based CB and default CB=2.0
        let use_size_based = (walk_count & 1) == 0;
        let cb = if use_size_based {
            fit_cb_value(avg_size)
        } else {
            2.0
        };
        let base = 1.0 / cb;

        self.score_table.clear();
        // score[i] = base^i, so score[0] = 1.0, score[1] = base, etc.
        // Higher break count = lower score (exponential decay)
        let mut score = 1.0;
        while score > 1e-300 {
            self.score_table.push(score);
            score *= base;
        }
        self.epsilon = score.max(1e-300);
    }

    /// Get score for a given break count
    #[inline]
    fn score(&self, break_count: usize) -> f64 {
        if break_count < self.score_table.len() {
            self.score_table[break_count]
        } else {
            self.epsilon
        }
    }

    /// Add clause to broken list
    #[inline]
    fn add_broken(&mut self, clause_idx: usize) {
        if self.broken_pos[clause_idx] < 0 {
            self.broken_pos[clause_idx] = self.broken.len() as i32;
            self.broken.push(clause_idx);
        }
    }

    /// Compute break-value for flipping a variable: how many currently satisfied
    /// clauses would become unsatisfied if we flip this variable.
    /// O(degree) - only scans clauses containing the variable
    fn compute_break_value(&self, var_idx: usize) -> usize {
        let current_val = self.values[var_idx];
        // The literal that is currently TRUE and would become FALSE
        let true_lit_idx = if current_val {
            2 * var_idx
        } else {
            2 * var_idx + 1
        };

        let mut break_count = 0;

        // For each clause containing the currently-true literal
        for &clause_idx in &self.occurs[true_lit_idx] {
            // If this clause has exactly 1 satisfier, flipping would break it
            if self.sat_count[clause_idx] == 1 {
                break_count += 1;
            }
        }

        break_count
    }

    /// Flip a variable and incrementally update broken list
    /// O(degree) - only updates affected clauses
    fn flip(&mut self, var_idx: usize) {
        let old_val = self.values[var_idx];
        self.values[var_idx] = !old_val;

        // The literal that was TRUE and is now FALSE
        let was_true_idx = if old_val {
            2 * var_idx
        } else {
            2 * var_idx + 1
        };
        // The literal that was FALSE and is now TRUE
        let now_true_idx = if old_val {
            2 * var_idx + 1
        } else {
            2 * var_idx
        };

        // Process clauses containing the now-FALSE literal (lose a satisfier)
        // Use index iteration to avoid borrow conflicts
        let was_true_len = self.occurs[was_true_idx].len();
        for i in 0..was_true_len {
            let clause_idx = self.occurs[was_true_idx][i];
            self.sat_count[clause_idx] -= 1;
            if self.sat_count[clause_idx] == 0 {
                // Inline add_broken to avoid method call borrow issues
                if self.broken_pos[clause_idx] < 0 {
                    self.broken_pos[clause_idx] = self.broken.len() as i32;
                    self.broken.push(clause_idx);
                }
            }
            self.ticks += 1;
        }

        // Process clauses containing the now-TRUE literal (gain a satisfier)
        let now_true_len = self.occurs[now_true_idx].len();
        for i in 0..now_true_len {
            let clause_idx = self.occurs[now_true_idx][i];
            if self.sat_count[clause_idx] == 0 {
                // Inline remove_broken to avoid method call borrow issues
                let pos = self.broken_pos[clause_idx];
                if pos >= 0 {
                    let pos = pos as usize;
                    let last = self.broken.len() - 1;
                    if pos != last {
                        let last_clause = self.broken[last];
                        self.broken[pos] = last_clause;
                        self.broken_pos[last_clause] = pos as i32;
                    }
                    self.broken.pop();
                    self.broken_pos[clause_idx] = -1;
                }
            }
            self.sat_count[clause_idx] += 1;
            self.ticks += 1;
        }
    }
}

/// Performs ProbSAT random walk to find good phases.
///
/// The walk starts from current phases, randomly picks unsatisfied clauses,
/// scores literals by break-count, and flips to minimize unsatisfied clauses.
/// Best phases (at minimum broken count) are saved for CDCL.
///
/// Uses O(1) amortized complexity via occurrence lists and satisfaction counts.
pub fn walk(
    clause_db: &ClauseDB,
    num_vars: usize,
    phases: &mut [Option<bool>],
    stats: &mut WalkStats,
    seed: u64,
    tick_limit: u64,
) -> bool {
    stats.walk_count += 1;

    // Count original clauses for threshold check
    let mut num_original = 0;
    for i in 0..clause_db.len() {
        let header = clause_db.header(i);
        if !header.is_empty() && !header.is_learned() {
            num_original += 1;
        }
    }

    // Skip walk for small instances
    // With O(1) amortized operations, walk is efficient for medium instances too
    // Threshold: walk helps when phase initialization matters (200+ vars, 800+ clauses)
    if num_vars < 200 || num_original < 800 {
        return false;
    }

    let mut walker = Walker::new(
        num_vars,
        clause_db.len(),
        seed.wrapping_add(stats.walk_count),
        tick_limit,
    );

    // Initialize values from saved phases
    for (phase, (val, best)) in phases
        .iter()
        .zip(walker.values.iter_mut().zip(walker.best_phases.iter_mut()))
        .take(num_vars)
    {
        *val = phase.unwrap_or(true);
        *best = *val;
    }

    // Build occurrence lists and compute initial satisfaction counts
    let mut total_lits: usize = 0;
    let mut num_clauses: usize = 0;

    for clause_idx in 0..clause_db.len() {
        let header = clause_db.header(clause_idx);
        if header.is_empty() || header.is_learned() {
            continue;
        }

        let lits = clause_db.literals(clause_idx);
        total_lits += lits.len();
        num_clauses += 1;

        let mut sat_count = 0u32;
        for &lit in lits {
            // Add to occurrence list
            let occurs_idx = lit_to_occurs_idx(lit);
            if occurs_idx < walker.occurs.len() {
                walker.occurs[occurs_idx].push(clause_idx);
            }

            // Count satisfied literals
            let var_idx = lit.variable().index();
            if var_idx < walker.values.len() {
                let val = walker.values[var_idx];
                let lit_sat = (lit.is_positive() && val) || (!lit.is_positive() && !val);
                if lit_sat {
                    sat_count += 1;
                }
            }
        }

        walker.sat_count[clause_idx] = sat_count;
        if sat_count == 0 {
            walker.add_broken(clause_idx);
        }

        walker.ticks += lits.len() as u64;
    }

    // Compute average clause size for CB parameter
    let avg_size = if num_clauses > 0 {
        total_lits as f64 / num_clauses as f64
    } else {
        3.0
    };
    walker.populate_table(avg_size, stats.walk_count);

    // Save initial minimum
    walker.minimum = walker.broken.len();
    stats.best_minimum = walker.minimum as u64;

    // Main walk loop
    while !walker.broken.is_empty() && walker.ticks < walker.limit {
        walker.ticks += 1;
        stats.flips += 1;
        stats.broken_total += walker.broken.len() as u64;

        // Pick random broken clause
        let broken_idx = walker.random.pick(walker.broken.len());
        let clause_idx = walker.broken[broken_idx];

        // Score literals in the broken clause
        let lits = clause_db.literals(clause_idx);

        let mut scores: Vec<(usize, f64)> = Vec::with_capacity(lits.len());
        let mut sum = 0.0;

        for &lit in lits {
            let var_idx = lit.variable().index();
            if var_idx >= walker.values.len() {
                continue;
            }
            // All literals in broken clause are false, so we want to flip one to true
            // Break value is how many satisfied clauses would become broken
            let break_val = walker.compute_break_value(var_idx);
            let score = walker.score(break_val);
            scores.push((var_idx, score));
            sum += score;

            // Count ticks for break-value computation
            let true_lit_idx = if walker.values[var_idx] {
                2 * var_idx
            } else {
                2 * var_idx + 1
            };
            if true_lit_idx < walker.occurs.len() {
                walker.ticks += walker.occurs[true_lit_idx].len() as u64;
            }
        }

        if sum == 0.0 || scores.is_empty() {
            continue;
        }

        // Sample literal proportional to score
        let limit = sum * walker.random.generate_double();
        let mut cumsum = 0.0;
        let mut flip_var = scores[0].0;

        for (var_idx, score) in &scores {
            cumsum += score;
            if cumsum > limit {
                flip_var = *var_idx;
                break;
            }
        }

        // Flip the chosen variable (O(degree) incremental update)
        walker.flip(flip_var);

        // Check for new minimum
        if walker.broken.len() < walker.minimum {
            walker.minimum = walker.broken.len();
            stats.best_minimum = walker.minimum as u64;

            // Save best phases
            for var_idx in 0..num_vars {
                walker.best_phases[var_idx] = walker.values[var_idx];
            }

            // If we found a satisfying assignment, we're done
            if walker.minimum == 0 {
                break;
            }
        }
    }

    // Copy best phases back
    for (phase, &best) in phases
        .iter_mut()
        .zip(walker.best_phases.iter())
        .take(num_vars)
    {
        *phase = Some(best);
    }

    // Return true if we found a satisfying assignment
    walker.minimum == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random() {
        let mut rng = Random::new(42);
        let a = rng.next();
        let b = rng.next();
        assert_ne!(a, b);

        // Test bounded random
        for _ in 0..100 {
            let x = rng.pick(10);
            assert!(x < 10);
        }
    }

    #[test]
    fn test_fit_cb_value() {
        // Test interpolation
        let cb3 = fit_cb_value(3.0);
        assert!((cb3 - 2.50).abs() < 0.01);

        let cb5 = fit_cb_value(5.0);
        assert!((cb5 - 3.70).abs() < 0.01);

        // Test interpolation between points
        let cb4 = fit_cb_value(4.0);
        assert!(cb4 > 2.50 && cb4 < 3.70);
    }

    #[test]
    fn test_walker_score_table() {
        let mut walker = Walker::new(10, 100, 42, 1000);
        walker.populate_table(3.0, 0);

        // Score should decrease with break count
        let s0 = walker.score(0);
        let s1 = walker.score(1);
        let s2 = walker.score(2);
        assert!(s0 > s1);
        assert!(s1 > s2);
    }

    #[test]
    fn test_broken_list_operations() {
        let mut walker = Walker::new(10, 100, 42, 1000);

        // Add some broken clauses
        walker.add_broken(5);
        walker.add_broken(10);
        walker.add_broken(3);

        assert_eq!(walker.broken.len(), 3);
        assert!(walker.broken.contains(&5));
        assert!(walker.broken.contains(&10));
        assert!(walker.broken.contains(&3));

        // Check positions are tracked
        assert!(walker.broken_pos[5] >= 0);
        assert!(walker.broken_pos[10] >= 0);
        assert!(walker.broken_pos[3] >= 0);

        // Adding same clause again should not duplicate
        walker.add_broken(5);
        assert_eq!(walker.broken.len(), 3);
    }
}
