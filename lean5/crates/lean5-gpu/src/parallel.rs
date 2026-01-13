//! CPU-parallel batch operations using Rayon
//!
//! While GPU acceleration is available for WHNF reduction, type checking
//! is better suited to CPU parallelism due to:
//! 1. Complex control flow (recursion, pattern matching)
//! 2. Environment lookups
//! 3. Dynamic recursion depths
//!
//! This module provides Rayon-based parallel implementations that achieve
//! good speedups on multi-core machines.

use lean5_kernel::{Environment, Expr, TypeChecker, TypeError};
use rayon::prelude::*;

/// Configuration for parallel batch operations
#[derive(Clone, Debug)]
pub struct ParallelConfig {
    /// Minimum batch size to use parallelism (smaller batches run sequentially)
    pub min_parallel_batch: usize,
    /// Number of threads (None = use Rayon default based on CPU cores)
    pub num_threads: Option<usize>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_parallel_batch: 4,
            num_threads: None,
        }
    }
}

impl ParallelConfig {
    /// Create config optimized for latency (lower threshold)
    pub fn low_latency() -> Self {
        Self {
            min_parallel_batch: 2,
            num_threads: None,
        }
    }

    /// Create config optimized for throughput (higher threshold)
    pub fn high_throughput() -> Self {
        Self {
            min_parallel_batch: 16,
            num_threads: None,
        }
    }
}

/// Parallel batch type checker
///
/// Provides CPU-parallel implementations of batch operations.
/// Use this for checking multiple independent expressions/declarations.
pub struct ParallelBatch<'env> {
    env: &'env Environment,
    config: ParallelConfig,
}

impl<'env> ParallelBatch<'env> {
    /// Create a new parallel batch processor
    pub fn new(env: &'env Environment) -> Self {
        Self {
            env,
            config: ParallelConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(env: &'env Environment, config: ParallelConfig) -> Self {
        Self { env, config }
    }

    /// Batch type inference
    ///
    /// Infers types for multiple expressions in parallel.
    /// Each expression is independent and can be processed concurrently.
    pub fn batch_infer_type(&self, exprs: &[Expr]) -> Vec<Result<Expr, TypeError>> {
        if exprs.len() < self.config.min_parallel_batch {
            // Sequential for small batches
            exprs
                .iter()
                .map(|e| {
                    let mut tc = TypeChecker::new(self.env);
                    tc.infer_type(e)
                })
                .collect()
        } else {
            // Parallel for larger batches
            exprs
                .par_iter()
                .map(|e| {
                    let mut tc = TypeChecker::new(self.env);
                    tc.infer_type(e)
                })
                .collect()
        }
    }

    /// Batch type checking
    ///
    /// Checks multiple (expression, expected_type) pairs in parallel.
    pub fn batch_check_type(&self, pairs: &[(Expr, Expr)]) -> Vec<Result<(), TypeError>> {
        if pairs.len() < self.config.min_parallel_batch {
            pairs
                .iter()
                .map(|(e, ty)| {
                    let mut tc = TypeChecker::new(self.env);
                    tc.check_type(e, ty)
                })
                .collect()
        } else {
            pairs
                .par_iter()
                .map(|(e, ty)| {
                    let mut tc = TypeChecker::new(self.env);
                    tc.check_type(e, ty)
                })
                .collect()
        }
    }

    /// Batch WHNF reduction
    ///
    /// Reduces multiple expressions to weak head normal form in parallel.
    pub fn batch_whnf(&self, exprs: &[Expr]) -> Vec<Expr> {
        if exprs.len() < self.config.min_parallel_batch {
            let tc = TypeChecker::new(self.env);
            exprs.iter().map(|e| tc.whnf(e)).collect()
        } else {
            exprs
                .par_iter()
                .map(|e| {
                    let tc = TypeChecker::new(self.env);
                    tc.whnf(e)
                })
                .collect()
        }
    }

    /// Batch definitional equality checking
    ///
    /// Checks multiple pairs for definitional equality in parallel.
    pub fn batch_is_def_eq(&self, pairs: &[(Expr, Expr)]) -> Vec<bool> {
        if pairs.len() < self.config.min_parallel_batch {
            let tc = TypeChecker::new(self.env);
            pairs.iter().map(|(a, b)| tc.is_def_eq(a, b)).collect()
        } else {
            pairs
                .par_iter()
                .map(|(a, b)| {
                    let tc = TypeChecker::new(self.env);
                    tc.is_def_eq(a, b)
                })
                .collect()
        }
    }
}

/// Convenience function for parallel type inference
pub fn parallel_infer_type(env: &Environment, exprs: &[Expr]) -> Vec<Result<Expr, TypeError>> {
    ParallelBatch::new(env).batch_infer_type(exprs)
}

/// Convenience function for parallel WHNF
pub fn parallel_whnf(env: &Environment, exprs: &[Expr]) -> Vec<Expr> {
    ParallelBatch::new(env).batch_whnf(exprs)
}

/// Convenience function for parallel definitional equality
pub fn parallel_is_def_eq(env: &Environment, pairs: &[(Expr, Expr)]) -> Vec<bool> {
    ParallelBatch::new(env).batch_is_def_eq(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lean5_kernel::{BinderInfo, Level};

    /// Helper to build Level with n successors
    fn succ_n(n: usize) -> Level {
        let mut level = Level::Zero;
        for _ in 0..n {
            level = Level::succ(level);
        }
        level
    }

    #[test]
    fn test_parallel_infer_type_sorts() {
        let env = Environment::new();
        let exprs: Vec<Expr> = (0..100).map(|i| Expr::Sort(succ_n(i))).collect();

        let results = parallel_infer_type(&env, &exprs);

        assert_eq!(results.len(), 100);
        for (i, result) in results.iter().enumerate() {
            let ty = result.as_ref().expect("Should succeed");
            // Type of Sort(n) is Sort(n+1)
            assert_eq!(*ty, Expr::Sort(succ_n(i + 1)));
        }
    }

    #[test]
    fn test_parallel_infer_type_lambdas() {
        let env = Environment::new();

        // Create 50 identity functions at different type levels
        let exprs: Vec<Expr> = (0..50)
            .map(|i| {
                let level = succ_n(i);
                Expr::lam(BinderInfo::Default, Expr::Sort(level), Expr::bvar(0))
            })
            .collect();

        let results = parallel_infer_type(&env, &exprs);

        assert_eq!(results.len(), 50);
        for result in &results {
            assert!(result.is_ok(), "Type inference should succeed");
            let ty = result.as_ref().unwrap();
            // Should be Pi type
            assert!(matches!(ty, Expr::Pi(_, _, _)));
        }
    }

    #[test]
    fn test_parallel_whnf() {
        let env = Environment::new();

        // Create 100 beta redexes
        let exprs: Vec<Expr> = (0..100)
            .map(|_| {
                let id = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::bvar(0));
                Expr::app(id, Expr::prop())
            })
            .collect();

        let results = parallel_whnf(&env, &exprs);

        assert_eq!(results.len(), 100);
        for result in &results {
            // (λx.x) Prop should reduce to Prop
            assert!(result.is_prop());
        }
    }

    #[test]
    fn test_parallel_is_def_eq() {
        let env = Environment::new();

        // Create pairs that should be equal
        let pairs: Vec<(Expr, Expr)> = (0..50)
            .map(|i| {
                let level = succ_n(i);
                (Expr::Sort(level.clone()), Expr::Sort(level))
            })
            .collect();

        let results = parallel_is_def_eq(&env, &pairs);

        assert_eq!(results.len(), 50);
        for result in &results {
            assert!(*result, "Same levels should be def eq");
        }
    }

    #[test]
    fn test_parallel_is_def_eq_with_reduction() {
        let env = Environment::new();

        // (λx.x) Prop =def= Prop
        let pairs: Vec<(Expr, Expr)> = (0..50)
            .map(|_| {
                let id = Expr::lam(BinderInfo::Default, Expr::type_(), Expr::bvar(0));
                let app = Expr::app(id, Expr::prop());
                (app, Expr::prop())
            })
            .collect();

        let results = parallel_is_def_eq(&env, &pairs);

        assert_eq!(results.len(), 50);
        for result in &results {
            assert!(*result, "Beta-reduced expression should be def eq");
        }
    }

    #[test]
    fn test_parallel_check_type() {
        let env = Environment::new();

        // Create (Prop, Type) pairs - Prop : Type should succeed
        let pairs: Vec<(Expr, Expr)> = (0..50).map(|_| (Expr::prop(), Expr::type_())).collect();

        let batch = ParallelBatch::new(&env);
        let results = batch.batch_check_type(&pairs);

        assert_eq!(results.len(), 50);
        for result in &results {
            assert!(result.is_ok(), "Prop : Type should succeed");
        }
    }

    #[test]
    fn test_sequential_for_small_batch() {
        let env = Environment::new();
        let config = ParallelConfig {
            min_parallel_batch: 100, // Force sequential
            num_threads: None,
        };

        let batch = ParallelBatch::with_config(&env, config);
        let exprs: Vec<Expr> = (0..10).map(|_| Expr::prop()).collect();

        // Should run sequentially without panicking
        let results = batch.batch_infer_type(&exprs);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_config_presets() {
        let low = ParallelConfig::low_latency();
        let high = ParallelConfig::high_throughput();

        assert!(low.min_parallel_batch < high.min_parallel_batch);
    }
}
