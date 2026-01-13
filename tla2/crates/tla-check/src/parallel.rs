//! Parallel Model Checker - Multi-threaded BFS state exploration
//!
//! This module implements parallel state space exploration using:
//! - DashMap for lock-free concurrent state storage
//! - Work-stealing deques for load balancing
//! - Thread pool for worker management
//!
//! # Algorithm
//!
//! The parallel checker uses a work-stealing approach:
//! 1. Main thread generates initial states and distributes them
//! 2. Worker threads explore states using local deques
//! 3. Workers steal from each other when their local queue is empty
//! 4. First worker to find a violation signals all others to stop
//! 5. Results are aggregated at the end
//!
//! # Performance
//!
//! For large state spaces, the parallel checker can provide near-linear speedup
//! with the number of cores. Overhead is minimal due to:
//! - Lock-free seen set (DashMap)
//! - Work stealing minimizes idle time
//! - Early termination on violations

use crate::arena::BulkStateStorage;
use crate::check::{
    CheckError, CheckResult, CheckStats, LimitType, Progress, ProgressCallback, Trace,
};
use crate::compiled_guard::{compile_guard, compile_guard_for_filter, CompiledGuard};
use crate::config::Config;
use crate::constants::bind_constants_from_config;
use crate::coverage::detect_actions;
use crate::enumerate::{
    enumerate_constraints_to_bulk, enumerate_states_from_constraint_branches, enumerate_successors,
    extract_conjunction_remainder, extract_init_constraints, find_unconstrained_vars,
    print_enum_profile_stats, LocalScope,
};
use crate::eval::{eval, Env, EvalCtx};
use crate::intern::{get_interner, HandleState, ValueInterner};
use crate::state::{ArrayState, Fingerprint, State};
use crate::storage::{CapacityStatus, FingerprintSet, ShardedFingerprintSet};
use crate::var_index::VarRegistry;
use crate::Value;
use crossbeam_channel::{bounded, Sender};
use crossbeam_deque::{Injector, Stealer, Worker};
use dashmap::DashMap;
use rustc_hash::{FxHashSet, FxHasher};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};
use tla_core::ast::{Module, OperatorDef, Unit};

/// FxHasher-based BuildHasher for faster hashing of Fingerprint keys.
/// Since Fingerprint is already a 64-bit hash, FxHasher is much faster than SipHash.
type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// FxHasher-based DashMap for concurrent fingerprint storage.
type FxDashMap<K, V> = DashMap<K, V, FxBuildHasher>;

/// Capacity status encoded as u8 for atomic operations.
/// 0 = Normal, 1 = Warning, 2 = Critical
const CAPACITY_NORMAL: u8 = 0;
const CAPACITY_WARNING: u8 = 1;
const CAPACITY_CRITICAL: u8 = 2;

/// Convert CapacityStatus to u8 for atomic operations.
fn capacity_status_to_u8(status: &CapacityStatus) -> u8 {
    match status {
        CapacityStatus::Normal => CAPACITY_NORMAL,
        CapacityStatus::Warning { .. } => CAPACITY_WARNING,
        CapacityStatus::Critical { .. } => CAPACITY_CRITICAL,
    }
}

/// Check capacity status and emit warning if status has changed.
///
/// Returns the new status value to store in the atomic.
fn check_and_warn_capacity(seen_fps: &dyn FingerprintSet, last_status: &AtomicU8) {
    let status = seen_fps.capacity_status();
    let status_u8 = capacity_status_to_u8(&status);
    let prev_status = last_status.load(Ordering::Relaxed);

    // Only warn if status has changed
    if status_u8 == prev_status {
        return;
    }

    match status {
        CapacityStatus::Normal => {
            // Status improved back to normal - no warning needed
        }
        CapacityStatus::Warning {
            count,
            capacity,
            usage,
        } => {
            eprintln!(
                "Warning: Fingerprint storage at {:.1}% capacity ({} / {} states). \
                 Consider increasing --mmap-fingerprints capacity if state space is larger.",
                usage * 100.0,
                count,
                capacity
            );
        }
        CapacityStatus::Critical {
            count,
            capacity,
            usage,
        } => {
            eprintln!(
                "CRITICAL: Fingerprint storage at {:.1}% capacity ({} / {} states). \
                 Insert failures imminent! Increase --mmap-fingerprints capacity.",
                usage * 100.0,
                count,
                capacity
            );
        }
    }

    last_status.store(status_u8, Ordering::Relaxed);
}

/// Check if HandleState mode is enabled via environment variable.
///
/// When enabled (TLA2_USE_HANDLE_STATE=1), the parallel checker uses
/// HandleState instead of ArrayState in work queues, eliminating Arc
/// atomic contention during state cloning.
fn use_handle_state() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_USE_HANDLE_STATE").is_ok())
}

/// Result from a worker thread
enum WorkerResult {
    /// Found an invariant violation
    Violation {
        invariant: String,
        state_fp: Fingerprint,
        stats: WorkerStats,
    },
    /// Found a deadlock
    Deadlock {
        state_fp: Fingerprint,
        stats: WorkerStats,
    },
    /// Worker completed successfully (no violations)
    Done(WorkerStats),
    /// Worker encountered an error
    Error(CheckError, WorkerStats),
    /// Exploration limit was reached
    LimitReached {
        limit_type: LimitType,
        stats: WorkerStats,
    },
}

/// Statistics from a single worker
#[derive(Debug, Clone, Default)]
struct WorkerStats {
    states_explored: usize,
    transitions: usize,
    /// Profiling: number of work steals from injector/other workers
    steals: usize,
    /// Profiling: number of pushes to global injector
    injector_pushes: usize,
    /// Profiling: duplicate fingerprint encounters
    dedup_hits: usize,
    /// Profiling: empty poll retries (no work found)
    empty_polls: usize,
    /// Profiling: local cache hits (duplicates found in per-worker cache)
    local_cache_hits: usize,
    /// Timing: nanoseconds spent in enumeration
    enum_ns: u64,
    /// Timing: nanoseconds spent in fingerprint contains check
    contains_ns: u64,
    /// Timing: nanoseconds spent in fingerprint insert
    insert_ns: u64,
    /// Timing: nanoseconds spent in invariant checking
    invariant_ns: u64,
    /// Timing: nanoseconds spent materializing states
    materialize_ns: u64,
}

/// Size of per-worker local duplicate cache.
/// Larger cache = higher hit rate but more memory per worker.
/// 64K entries = ~512KB per worker, fits in L2/L3 cache.
/// Testing shows larger caches improve hit rate but not performance
/// (global seen set lookups are already fast).
const LOCAL_CACHE_SIZE: usize = 64 * 1024;

/// Check if detailed timing profiling is enabled (TLA2_TIMING=1).
fn timing_enabled() -> bool {
    static TIMING: OnceLock<bool> = OnceLock::new();
    *TIMING.get_or_init(|| std::env::var("TLA2_TIMING").is_ok())
}

/// Batch size for work_remaining counter updates.
/// Updating work_remaining atomically every state causes cache line bouncing.
/// Batching locally and flushing periodically reduces contention.
const WORK_BATCH_SIZE: usize = 256;

/// Parallel model checker using work-stealing
pub struct ParallelChecker {
    /// Number of worker threads
    num_workers: usize,
    /// Global seen set - concurrent hash map (used when `store_full_states` is true)
    /// Uses FxHasher for faster hashing since Fingerprint is already a 64-bit hash.
    /// Stores ArrayState (compact Box<[Value]>) instead of State (OrdMap) for memory efficiency.
    seen: Arc<FxDashMap<Fingerprint, ArrayState>>,
    /// Seen fingerprints (used when `store_full_states` is false)
    /// Uses FingerprintSet trait which supports both DashSet and MmapFingerprintSet.
    seen_fps: Arc<dyn FingerprintSet>,
    /// Parent pointers for trace reconstruction.
    /// Uses FxHasher for faster hashing since Fingerprint is already a 64-bit hash.
    parents: Arc<FxDashMap<Fingerprint, Fingerprint>>,
    /// Variable registry for ArrayState <-> State conversion
    var_registry: Arc<VarRegistry>,
    /// Whether to store full states for trace reconstruction
    store_full_states: bool,
    /// Flag to signal early termination
    stop_flag: Arc<AtomicBool>,
    /// Number of states remaining to explore (for termination detection)
    ///
    /// This counts the number of queued + in-progress states. It is incremented
    /// when a new state is discovered and enqueued, and decremented when a
    /// worker finishes processing a state.
    work_remaining: Arc<AtomicUsize>,
    /// Maximum queue depth seen
    max_queue_depth: Arc<AtomicUsize>,
    /// Maximum BFS depth seen
    max_depth: Arc<AtomicUsize>,
    /// Total transitions counter (atomic for thread safety)
    total_transitions: Arc<AtomicUsize>,
    /// Variable names
    vars: Vec<Arc<str>>,
    /// Operator definitions
    op_defs: HashMap<String, OperatorDef>,
    /// Configuration
    config: Config,
    /// Whether to check for deadlocks
    check_deadlock: bool,
    /// Module for worker thread cloning
    module: Arc<Module>,
    /// Extended modules for worker thread cloning
    extended_modules: Arc<Vec<Module>>,
    /// Maximum states to explore (None = unlimited)
    max_states_limit: Option<usize>,
    /// Maximum BFS depth (None = unlimited)
    max_depth_limit: Option<usize>,
    /// Progress callback (called periodically during checking)
    progress_callback: Option<Arc<ProgressCallback>>,
    /// How often to report progress (in milliseconds)
    progress_interval_ms: u64,
}

impl ParallelChecker {
    /// Create a new parallel model checker
    ///
    /// # Arguments
    /// * `module` - The TLA+ module to check
    /// * `config` - Model checking configuration
    /// * `num_workers` - Number of worker threads (0 = use number of CPUs)
    pub fn new(module: &Module, config: &Config, num_workers: usize) -> Self {
        Self::new_with_extends(module, &[], config, num_workers)
    }

    /// Create a new parallel model checker with extended modules
    ///
    /// The `extended_modules` should be modules that `module` extends (via EXTENDS).
    /// Their operator definitions will be loaded first, then the main module's
    /// definitions (which may override them).
    ///
    /// # Arguments
    /// * `module` - The TLA+ module to check
    /// * `extended_modules` - Modules that the main module extends
    /// * `config` - Model checking configuration
    /// * `num_workers` - Number of worker threads (0 = use number of CPUs)
    pub fn new_with_extends(
        module: &Module,
        extended_modules: &[&Module],
        config: &Config,
        num_workers: usize,
    ) -> Self {
        let num_workers = if num_workers == 0 {
            thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            num_workers
        };

        // Create evaluation context and load modules (for instance detection)
        let mut ctx = EvalCtx::new();
        // Avoid loading operators from modules that are only referenced via INSTANCE
        // (e.g., `C == INSTANCE M WITH ...` or `INSTANCE M`). These must not pollute
        // the main namespace.
        use tla_core::ast::Expr;
        let mut named_instance_modules: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Helper to scan a module for INSTANCE declarations
        fn collect_instance_modules(
            module: &Module,
            named_instance_modules: &mut std::collections::HashSet<String>,
        ) {
            for unit in &module.units {
                // Standalone INSTANCE declarations
                if let Unit::Instance(inst) = &unit.node {
                    named_instance_modules.insert(inst.module.node.clone());
                }
                // Named instances: TD == INSTANCE SyncTerminationDetection
                if let Unit::Operator(def) = &unit.node {
                    if let Expr::InstanceExpr(module_name, _subs) = &def.body.node {
                        named_instance_modules.insert(module_name.clone());
                    }
                }
            }
        }

        // Scan both extended modules and main module for INSTANCE declarations
        for ext_mod in extended_modules {
            collect_instance_modules(ext_mod, &mut named_instance_modules);
        }
        collect_instance_modules(module, &mut named_instance_modules);

        let mut module_by_name: std::collections::HashMap<&str, &Module> =
            std::collections::HashMap::new();
        for ext_mod in extended_modules {
            module_by_name.insert(ext_mod.name.node.as_str(), *ext_mod);
        }

        let mut extends_closure: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut stack: Vec<&str> = module.extends.iter().map(|s| s.node.as_str()).collect();
        while let Some(name) = stack.pop() {
            if !extends_closure.insert(name) {
                continue;
            }
            if let Some(m) = module_by_name.get(name) {
                for ext in &m.extends {
                    stack.push(ext.node.as_str());
                }
            }
        }

        // Load extended modules first (in order), skipping named-INSTANCE-only modules.
        for ext_mod in extended_modules {
            let is_named_instance_only = named_instance_modules.contains(&ext_mod.name.node)
                && !extends_closure.contains(ext_mod.name.node.as_str());
            if is_named_instance_only {
                continue;
            }
            ctx.load_module(ext_mod);
        }
        // Load main module last (can override)
        ctx.load_module(module);

        // Collect instance module names to skip their variables
        let instance_module_names: Vec<String> = ctx
            .instances()
            .values()
            .map(|info| info.module_name.clone())
            .collect();
        let instance_module_name_set: std::collections::HashSet<&str> =
            instance_module_names.iter().map(|s| s.as_str()).collect();

        // Extract variable names and operator definitions from all modules
        let mut vars: Vec<Arc<str>> = Vec::new();
        let mut op_defs: HashMap<String, OperatorDef> = HashMap::new();

        // First from extended modules
        for ext_mod in extended_modules {
            // Skip INSTANCE'd modules for variable collection AND operator collection
            // INSTANCE'd modules should only contribute operators via instance prefix (e.g., C!Init)
            let is_instanced = instance_module_name_set.contains(ext_mod.name.node.as_str());
            if is_instanced {
                continue;
            }
            for unit in &ext_mod.units {
                match &unit.node {
                    Unit::Variable(var_names) => {
                        for var in var_names {
                            if !vars.iter().any(|v| v.as_ref() == var.node.as_str()) {
                                vars.push(Arc::from(var.node.as_str()));
                            }
                        }
                    }
                    Unit::Operator(def) => {
                        op_defs.insert(def.name.node.clone(), def.clone());
                    }
                    _ => {}
                }
            }
        }

        // Then from main module (may shadow)
        for unit in &module.units {
            match &unit.node {
                Unit::Variable(var_names) => {
                    for var in var_names {
                        if !vars.iter().any(|v| v.as_ref() == var.node.as_str()) {
                            vars.push(Arc::from(var.node.as_str()));
                        }
                    }
                }
                Unit::Operator(def) => {
                    op_defs.insert(def.name.node.clone(), def.clone());
                }
                _ => {}
            }
        }

        // Sort variables to ensure fingerprints are consistent
        vars.sort();

        // Create VarRegistry for ArrayState <-> State conversion
        let var_registry = Arc::new(VarRegistry::from_names(vars.iter().cloned()));

        // Clone extended modules for worker threads
        let extended_modules_owned: Vec<Module> =
            extended_modules.iter().map(|m| (*m).clone()).collect();

        // Use 256 shards (2^8) for fingerprint set - reduces lock contention in no-trace mode
        // With 256 shards and 8 workers, collision probability is ~3.1%
        // Testing showed more shards (1024) didn't improve performance
        let shard_bits = 8; // 256 shards for ShardedFingerprintSet

        ParallelChecker {
            num_workers,
            seen: Arc::new(DashMap::with_hasher(FxBuildHasher::default())),
            seen_fps: Arc::new(ShardedFingerprintSet::new(shard_bits)),
            parents: Arc::new(DashMap::with_hasher(FxBuildHasher::default())),
            var_registry,
            store_full_states: true, // Default: store full states for trace reconstruction
            stop_flag: Arc::new(AtomicBool::new(false)),
            work_remaining: Arc::new(AtomicUsize::new(0)),
            max_queue_depth: Arc::new(AtomicUsize::new(0)),
            max_depth: Arc::new(AtomicUsize::new(0)),
            total_transitions: Arc::new(AtomicUsize::new(0)),
            vars,
            op_defs,
            config: config.clone(),
            check_deadlock: config.check_deadlock,
            module: Arc::new(module.clone()),
            extended_modules: Arc::new(extended_modules_owned),
            max_states_limit: None,
            max_depth_limit: None,
            progress_callback: None,
            progress_interval_ms: 1000, // Default: report every second
        }
    }

    /// Enable or disable deadlock checking
    pub fn set_deadlock_check(&mut self, check: bool) {
        self.check_deadlock = check;
    }

    /// Enable or disable storing full states for trace reconstruction.
    ///
    /// When `store` is false (no-trace mode), only fingerprints are stored and
    /// counterexample traces will be unavailable.
    pub fn set_store_states(&mut self, store: bool) {
        self.store_full_states = store;
    }

    /// Set whether to auto-create a temp trace file (no-op for parallel checker).
    ///
    /// ParallelChecker doesn't support disk-based trace files. This method
    /// is provided for API compatibility with ModelChecker.
    pub fn set_auto_create_trace_file(&mut self, _auto_create: bool) {
        // No-op: ParallelChecker doesn't support trace files
    }

    /// Set the fingerprint storage backend.
    ///
    /// This allows using memory-mapped storage for large state spaces that
    /// exceed available RAM. Must be called before `check()`.
    ///
    /// Only used when `store_full_states` is false (no-trace mode).
    /// When `store_full_states` is true, full states are stored in a DashMap
    /// regardless of this setting.
    pub fn set_fingerprint_storage(&mut self, storage: Arc<dyn FingerprintSet>) {
        self.seen_fps = storage;
    }

    /// Register an inline NEXT expression from a ResolvedSpec.
    ///
    /// When the SPECIFICATION formula contains an inline NEXT expression like
    /// `Init /\ [][\E n \in Node: Next(n)]_vars`, the `resolved.next_node` contains
    /// the CST node for the expression. This method lowers it to an AST and creates
    /// a synthetic operator definition.
    ///
    /// Call this after creating the checker if `resolved.next_node` is Some.
    pub fn register_inline_next(
        &mut self,
        resolved: &crate::check::ResolvedSpec,
    ) -> Result<(), crate::check::CheckError> {
        let Some(ref node) = resolved.next_node else {
            return Ok(()); // No inline NEXT, nothing to do
        };

        // Lower the CST node to an AST expression
        let expr = tla_core::lower_single_expr(tla_core::FileId(0), node).ok_or_else(|| {
            crate::check::CheckError::SpecificationError(format!(
                "Failed to lower inline NEXT expression: {}",
                node.text()
            ))
        })?;

        // Create a synthetic operator definition
        let op_def = tla_core::ast::OperatorDef {
            name: tla_core::Spanned::dummy(crate::check::INLINE_NEXT_NAME.to_string()),
            params: vec![],
            body: tla_core::Spanned::dummy(expr),
            local: false,
        };

        // Register the operator in our definitions
        self.op_defs
            .insert(crate::check::INLINE_NEXT_NAME.to_string(), op_def);

        Ok(())
    }

    /// Get the number of states found (works in both modes)
    fn states_count(&self) -> usize {
        if self.store_full_states {
            self.seen.len()
        } else {
            self.seen_fps.len()
        }
    }

    /// Set maximum number of states to explore
    ///
    /// When this limit is reached, model checking stops with `CheckResult::LimitReached`.
    /// This is useful for unbounded specifications that would otherwise run indefinitely.
    pub fn set_max_states(&mut self, limit: usize) {
        self.max_states_limit = Some(limit);
    }

    /// Set maximum BFS depth to explore
    ///
    /// When this limit is reached, model checking stops with `CheckResult::LimitReached`.
    /// Depth 0 = initial states, depth 1 = first successors, etc.
    pub fn set_max_depth(&mut self, limit: usize) {
        self.max_depth_limit = Some(limit);
    }

    /// Set a progress callback to receive periodic updates during model checking
    ///
    /// The callback is called approximately every `interval_ms` milliseconds (default: 1000ms).
    /// This is useful for long-running model checks to show progress to users.
    pub fn set_progress_callback(&mut self, callback: ProgressCallback) {
        self.progress_callback = Some(Arc::new(callback));
    }

    /// Set how often progress is reported (in milliseconds)
    ///
    /// Default is 1000ms. Setting to 0 disables progress reporting.
    pub fn set_progress_interval_ms(&mut self, interval_ms: u64) {
        self.progress_interval_ms = interval_ms;
    }

    /// Run the parallel model checker
    pub fn check(&self) -> CheckResult {
        // Validate config
        let init_name = match &self.config.init {
            Some(name) => name.clone(),
            None => {
                return CheckResult::Error {
                    error: CheckError::MissingInit,
                    stats: CheckStats::default(),
                }
            }
        };

        let next_name = match &self.config.next {
            Some(name) => name.clone(),
            None => {
                return CheckResult::Error {
                    error: CheckError::MissingNext,
                    stats: CheckStats::default(),
                }
            }
        };

        // Detect actions in Next relation for coverage statistics / reporting.
        let detected_actions: Vec<String> = self
            .op_defs
            .get(&next_name)
            .map(|next_def| {
                detect_actions(next_def)
                    .into_iter()
                    .map(|a| a.name)
                    .collect()
            })
            .unwrap_or_default();

        if self.vars.is_empty() {
            return CheckResult::Error {
                error: CheckError::NoVariables,
                stats: CheckStats::default(),
            };
        }

        // Create evaluation context for initial state generation
        let mut ctx = EvalCtx::new();
        // Load extended modules first (in order)
        for ext_mod in self.extended_modules.iter() {
            ctx.load_module(ext_mod);
        }
        // Load main module last (can override)
        ctx.load_module(&self.module);

        // For each registered instance, load the instanced module's operators
        let instance_module_names: Vec<String> = ctx
            .instances()
            .values()
            .map(|info| info.module_name.clone())
            .collect();
        for module_name in &instance_module_names {
            for ext_mod in self.extended_modules.iter() {
                if ext_mod.name.node == *module_name {
                    ctx.load_instance_module(module_name.clone(), ext_mod);
                    break;
                }
            }
        }

        // Register state variables in VarRegistry for efficient array-based state handling
        ctx.register_vars(self.vars.iter().cloned());

        // Bind constants from config before generating initial states
        if let Err(e) = bind_constants_from_config(&mut ctx, &self.config) {
            return CheckResult::Error {
                error: CheckError::EvalError(e),
                stats: CheckStats::default(),
            };
        }

        // Validate invariants exist
        for inv_name in &self.config.invariants {
            if !ctx.has_op(inv_name) {
                return CheckResult::Error {
                    error: CheckError::MissingInvariant(inv_name.clone()),
                    stats: CheckStats::default(),
                };
            }
        }

        // Pre-compile invariants for efficient evaluation in workers
        // This avoids AST traversal overhead per state (significant savings for large state spaces)
        let registry = ctx.var_registry().clone();
        let empty_local_scope = LocalScope::new();
        let compiled_invariants: Arc<Vec<(String, CompiledGuard)>> = Arc::new(
            self.config
                .invariants
                .iter()
                .filter_map(|inv_name| {
                    ctx.get_op(inv_name).map(|def| {
                        let compiled =
                            compile_guard(&ctx, &def.body, &registry, &empty_local_scope);
                        (inv_name.clone(), compiled)
                    })
                })
                .collect(),
        );

        // Get the VarRegistry for efficient array-based state handling.
        let var_registry = Arc::new(ctx.var_registry().clone());

        enum InitialStates {
            Bulk(BulkStateStorage),
            Vec(Vec<State>),
        }

        // Try streaming initial-state enumeration to BulkStateStorage to avoid large Vec<State>
        // OrdMap allocations (e.g., MCBakery ISpec with 655K initial states).
        // This works for both trace and no-trace modes - the seeding loop handles trace storage.
        let initial_states: InitialStates =
            match self.generate_initial_states_to_bulk(&mut ctx, &init_name) {
                Ok(Some(storage)) => InitialStates::Bulk(storage),
                Ok(None) => match self.generate_initial_states(&ctx, &init_name) {
                    Ok(states) => InitialStates::Vec(states),
                    Err(e) => {
                        return CheckResult::Error {
                            error: e,
                            stats: CheckStats::default(),
                        }
                    }
                },
                Err(e) => {
                    return CheckResult::Error {
                        error: e,
                        stats: CheckStats::default(),
                    }
                }
            };

        // Channel for worker results
        let (result_tx, result_rx) = bounded::<WorkerResult>(self.num_workers);

        // Spawn worker threads - branch based on HandleState mode
        let mut handles = Vec::new();

        // Initial states count (after constraints/invariant checks and dedup).
        let mut num_initial: usize = 0;

        if use_handle_state() {
            // HandleState mode: use HandleState in queues for zero-Arc-atomic cloning
            let interner = get_interner();

            // Create the global injector queue for work stealing (HandleState)
            let injector: Arc<Injector<(HandleState, usize)>> = Arc::new(Injector::new());

            // Create workers and their stealers (HandleState)
            let mut workers: Vec<Worker<(HandleState, usize)>> = Vec::new();
            let mut stealers: Vec<Stealer<(HandleState, usize)>> = Vec::new();

            for _ in 0..self.num_workers {
                let worker = Worker::new_fifo();
                stealers.push(worker.stealer());
                workers.push(worker);
            }

            let stealers = Arc::new(stealers);

            // Seed initial states into local queues (avoids injecting huge frontiers into Injector).
            // Reuse a single ArrayState scratch buffer for constraint/invariant checking.
            let mut scratch = ArrayState::new(var_registry.len());

            match initial_states {
                InitialStates::Bulk(storage) => {
                    let total = storage.len() as u32;
                    for idx in 0..total {
                        scratch.overwrite_from_slice(storage.get_state(idx));

                        // State constraints (CONSTRAINT directive)
                        if !self.config.constraints.is_empty() {
                            let prev = ctx.bind_state_array(scratch.values());
                            let ok = self.config.constraints.iter().all(|constraint_name| {
                                matches!(ctx.eval_op(constraint_name), Ok(Value::Bool(true)))
                            });
                            ctx.restore_state_env(prev);
                            if !ok {
                                continue;
                            }
                        }

                        // Initial invariants check (single-threaded)
                        if !self.config.invariants.is_empty() {
                            let prev = ctx.bind_state_array(scratch.values());
                            let violated = if !compiled_invariants.is_empty() {
                                compiled_invariants
                                    .iter()
                                    .find(|(_, guard)| {
                                        match guard.eval_with_array(&mut ctx, &scratch) {
                                            Ok(true) => false,
                                            Ok(false) => true,
                                            Err(_) => true,
                                        }
                                    })
                                    .map(|(name, _)| name.clone())
                            } else {
                                self.config
                                    .invariants
                                    .iter()
                                    .find(|inv_name| match ctx.eval_op(inv_name) {
                                        Ok(Value::Bool(true)) => false,
                                        Ok(Value::Bool(false)) => true,
                                        _ => true,
                                    })
                                    .cloned()
                            };
                            ctx.restore_state_env(prev);

                            if let Some(invariant) = violated {
                                let state = State::from_indexed(scratch.values(), &var_registry);
                                let trace = Trace::from_states(vec![state]);
                                return CheckResult::InvariantViolation {
                                    invariant,
                                    trace,
                                    stats: CheckStats {
                                        initial_states: num_initial,
                                        ..Default::default()
                                    },
                                };
                            }
                        }

                        let handle_state =
                            HandleState::from_values(scratch.values(), &var_registry, interner);
                        let fp = handle_state.fingerprint();

                        let is_new = if self.store_full_states {
                            // Full-state mode: store ArrayState for trace reconstruction
                            let arr_state = ArrayState::from_values(scratch.values().to_vec());
                            self.seen.insert(fp, arr_state).is_none()
                        } else {
                            self.seen_fps.insert(fp)
                        };

                        if !is_new {
                            continue;
                        }

                        let worker_idx = num_initial % self.num_workers;
                        workers[worker_idx].push((handle_state, 0));
                        num_initial += 1;
                    }
                }
                InitialStates::Vec(states) => {
                    for state in states {
                        // Fast array-based constraint/invariant checks using ArrayState scratch.
                        // State vars are in sorted order, matching VarRegistry order.
                        let values: Vec<Value> = state.vars().map(|(_, v)| v.clone()).collect();
                        scratch.overwrite_from_slice(&values);

                        if !self.config.constraints.is_empty() {
                            let prev = ctx.bind_state_array(scratch.values());
                            let ok = self.config.constraints.iter().all(|constraint_name| {
                                matches!(ctx.eval_op(constraint_name), Ok(Value::Bool(true)))
                            });
                            ctx.restore_state_env(prev);
                            if !ok {
                                continue;
                            }
                        }

                        if !self.config.invariants.is_empty() {
                            let prev = ctx.bind_state_array(scratch.values());
                            let violated = if !compiled_invariants.is_empty() {
                                compiled_invariants
                                    .iter()
                                    .find(|(_, guard)| {
                                        match guard.eval_with_array(&mut ctx, &scratch) {
                                            Ok(true) => false,
                                            Ok(false) => true,
                                            Err(_) => true,
                                        }
                                    })
                                    .map(|(name, _)| name.clone())
                            } else {
                                self.config
                                    .invariants
                                    .iter()
                                    .find(|inv_name| match ctx.eval_op(inv_name) {
                                        Ok(Value::Bool(true)) => false,
                                        Ok(Value::Bool(false)) => true,
                                        _ => true,
                                    })
                                    .cloned()
                            };
                            ctx.restore_state_env(prev);

                            if let Some(invariant) = violated {
                                let trace = Trace::from_states(vec![state]);
                                return CheckResult::InvariantViolation {
                                    invariant,
                                    trace,
                                    stats: CheckStats {
                                        initial_states: num_initial,
                                        ..Default::default()
                                    },
                                };
                            }
                        }

                        let handle_state =
                            HandleState::from_values(&values, &var_registry, interner);
                        let fp = handle_state.fingerprint();

                        let is_new = if self.store_full_states {
                            // Convert State to ArrayState for memory efficiency
                            let arr_state = ArrayState::from_values(values);
                            self.seen.insert(fp, arr_state).is_none()
                        } else {
                            self.seen_fps.insert(fp)
                        };

                        if !is_new {
                            continue;
                        }

                        let worker_idx = num_initial % self.num_workers;
                        workers[worker_idx].push((handle_state, 0));
                        num_initial += 1;
                    }
                }
            }

            self.work_remaining.store(num_initial, Ordering::Release);

            for (worker_id, local_queue) in workers.into_iter().enumerate() {
                let seen = Arc::clone(&self.seen);
                let seen_fps = Arc::clone(&self.seen_fps);
                let parents = Arc::clone(&self.parents);
                let stop_flag = Arc::clone(&self.stop_flag);
                let work_remaining = Arc::clone(&self.work_remaining);
                let max_queue = Arc::clone(&self.max_queue_depth);
                let max_depth_atomic = Arc::clone(&self.max_depth);
                let total_transitions = Arc::clone(&self.total_transitions);
                let injector = Arc::clone(&injector);
                let stealers = Arc::clone(&stealers);
                let result_tx = result_tx.clone();
                let module = Arc::clone(&self.module);
                let extended_modules = Arc::clone(&self.extended_modules);
                let vars = self.vars.clone();
                let op_defs = self.op_defs.clone();
                let config = self.config.clone();
                let check_deadlock = self.check_deadlock;
                let next_name = next_name.clone();
                let num_workers = self.num_workers;
                let max_states_limit = self.max_states_limit;
                let max_depth_limit = self.max_depth_limit;
                let store_full_states = self.store_full_states;
                let registry = Arc::clone(&var_registry);
                let compiled_invs = Arc::clone(&compiled_invariants);

                // Use a large stack (16MB) for worker threads to handle deeply nested
                // TLA+ expressions. stacker will grow the stack dynamically if needed,
                // but starting with a larger stack avoids frequent growths.
                let handle = thread::Builder::new()
                    .stack_size(16 * 1024 * 1024) // 16MB initial stack
                    .spawn(move || {
                        run_worker_handle(
                            worker_id,
                            local_queue,
                            injector,
                            stealers,
                            seen,
                            seen_fps,
                            parents,
                            store_full_states,
                            stop_flag,
                            work_remaining,
                            max_queue,
                            max_depth_atomic,
                            total_transitions,
                            result_tx,
                            module,
                            extended_modules,
                            vars,
                            op_defs,
                            config,
                            check_deadlock,
                            next_name,
                            num_workers,
                            max_states_limit,
                            max_depth_limit,
                            registry,
                            compiled_invs,
                            interner,
                        );
                    })
                    .expect("Failed to spawn worker thread");
                handles.push(handle);
            }
        } else {
            // Standard mode: use ArrayState in queues
            // Create the global injector queue for work stealing
            // Use (ArrayState, depth) to avoid depths DashMap lookups in the hot path
            // Depth is embedded in the work item to reduce DashMap contention
            let injector: Arc<Injector<(ArrayState, usize)>> = Arc::new(Injector::new());

            // Create workers and their stealers
            // Use (ArrayState, depth) for efficient depth tracking without DashMap lookup
            let mut workers: Vec<Worker<(ArrayState, usize)>> = Vec::new();
            let mut stealers: Vec<Stealer<(ArrayState, usize)>> = Vec::new();

            for _ in 0..self.num_workers {
                let worker = Worker::new_fifo();
                stealers.push(worker.stealer());
                workers.push(worker);
            }

            let stealers = Arc::new(stealers);

            // Seed initial states into local queues (avoids injecting huge frontiers into Injector).
            match initial_states {
                InitialStates::Bulk(storage) => {
                    let total = storage.len() as u32;
                    for idx in 0..total {
                        let mut arr_state =
                            ArrayState::from_values(storage.get_state(idx).to_vec());

                        // State constraints (CONSTRAINT directive)
                        if !self.config.constraints.is_empty() {
                            let prev = ctx.bind_state_array(arr_state.values());
                            let ok = self.config.constraints.iter().all(|constraint_name| {
                                matches!(ctx.eval_op(constraint_name), Ok(Value::Bool(true)))
                            });
                            ctx.restore_state_env(prev);
                            if !ok {
                                continue;
                            }
                        }

                        // Initial invariants check (single-threaded)
                        if !self.config.invariants.is_empty() {
                            let prev = ctx.bind_state_array(arr_state.values());
                            let violated = if !compiled_invariants.is_empty() {
                                compiled_invariants
                                    .iter()
                                    .find(|(_, guard)| {
                                        match guard.eval_with_array(&mut ctx, &arr_state) {
                                            Ok(true) => false,
                                            Ok(false) => true,
                                            Err(_) => true,
                                        }
                                    })
                                    .map(|(name, _)| name.clone())
                            } else {
                                self.config
                                    .invariants
                                    .iter()
                                    .find(|inv_name| match ctx.eval_op(inv_name) {
                                        Ok(Value::Bool(true)) => false,
                                        Ok(Value::Bool(false)) => true,
                                        _ => true,
                                    })
                                    .cloned()
                            };
                            ctx.restore_state_env(prev);

                            if let Some(invariant) = violated {
                                let state = arr_state.to_state(&var_registry);
                                let trace = Trace::from_states(vec![state]);
                                return CheckResult::InvariantViolation {
                                    invariant,
                                    trace,
                                    stats: CheckStats {
                                        initial_states: num_initial,
                                        ..Default::default()
                                    },
                                };
                            }
                        }

                        let fp = arr_state.fingerprint(&var_registry);
                        let is_new = if self.store_full_states {
                            // Store a clone; we need arr_state for work queue
                            self.seen.insert(fp, arr_state.clone()).is_none()
                        } else {
                            self.seen_fps.insert(fp)
                        };
                        if !is_new {
                            continue;
                        }

                        let worker_idx = num_initial % self.num_workers;
                        workers[worker_idx].push((arr_state, 0));
                        num_initial += 1;
                    }
                }
                InitialStates::Vec(states) => {
                    for state in states {
                        let mut arr_state = ArrayState::from_state(&state, &var_registry);

                        // State constraints (CONSTRAINT directive)
                        if !self.config.constraints.is_empty() {
                            let prev = ctx.bind_state_array(arr_state.values());
                            let ok = self.config.constraints.iter().all(|constraint_name| {
                                matches!(ctx.eval_op(constraint_name), Ok(Value::Bool(true)))
                            });
                            ctx.restore_state_env(prev);
                            if !ok {
                                continue;
                            }
                        }

                        // Initial invariants check (single-threaded)
                        if !self.config.invariants.is_empty() {
                            let prev = ctx.bind_state_array(arr_state.values());
                            let violated = if !compiled_invariants.is_empty() {
                                compiled_invariants
                                    .iter()
                                    .find(|(_, guard)| {
                                        match guard.eval_with_array(&mut ctx, &arr_state) {
                                            Ok(true) => false,
                                            Ok(false) => true,
                                            Err(_) => true,
                                        }
                                    })
                                    .map(|(name, _)| name.clone())
                            } else {
                                self.config
                                    .invariants
                                    .iter()
                                    .find(|inv_name| match ctx.eval_op(inv_name) {
                                        Ok(Value::Bool(true)) => false,
                                        Ok(Value::Bool(false)) => true,
                                        _ => true,
                                    })
                                    .cloned()
                            };
                            ctx.restore_state_env(prev);

                            if let Some(invariant) = violated {
                                let trace = Trace::from_states(vec![state]);
                                return CheckResult::InvariantViolation {
                                    invariant,
                                    trace,
                                    stats: CheckStats {
                                        initial_states: num_initial,
                                        ..Default::default()
                                    },
                                };
                            }
                        }

                        let fp = arr_state.fingerprint(&var_registry);
                        let is_new = if self.store_full_states {
                            // Store a clone; we need arr_state for work queue
                            self.seen.insert(fp, arr_state.clone()).is_none()
                        } else {
                            self.seen_fps.insert(fp)
                        };
                        if !is_new {
                            continue;
                        }

                        let worker_idx = num_initial % self.num_workers;
                        workers[worker_idx].push((arr_state, 0));
                        num_initial += 1;
                    }
                }
            }

            self.work_remaining.store(num_initial, Ordering::Release);

            for (worker_id, local_queue) in workers.into_iter().enumerate() {
                let seen = Arc::clone(&self.seen);
                let seen_fps = Arc::clone(&self.seen_fps);
                let parents = Arc::clone(&self.parents);
                let stop_flag = Arc::clone(&self.stop_flag);
                let work_remaining = Arc::clone(&self.work_remaining);
                let max_queue = Arc::clone(&self.max_queue_depth);
                let max_depth_atomic = Arc::clone(&self.max_depth);
                let total_transitions = Arc::clone(&self.total_transitions);
                let injector = Arc::clone(&injector);
                let stealers = Arc::clone(&stealers);
                let result_tx = result_tx.clone();
                let module = Arc::clone(&self.module);
                let extended_modules = Arc::clone(&self.extended_modules);
                let vars = self.vars.clone();
                let op_defs = self.op_defs.clone();
                let config = self.config.clone();
                let check_deadlock = self.check_deadlock;
                let next_name = next_name.clone();
                let num_workers = self.num_workers;
                let max_states_limit = self.max_states_limit;
                let max_depth_limit = self.max_depth_limit;
                let store_full_states = self.store_full_states;
                let registry = Arc::clone(&var_registry);
                let compiled_invs = Arc::clone(&compiled_invariants);

                // Use a large stack (16MB) for worker threads to handle deeply nested
                // TLA+ expressions. stacker will grow the stack dynamically if needed,
                // but starting with a larger stack avoids frequent growths.
                let handle = thread::Builder::new()
                    .stack_size(16 * 1024 * 1024) // 16MB initial stack
                    .spawn(move || {
                        run_worker(
                            worker_id,
                            local_queue,
                            injector,
                            stealers,
                            seen,
                            seen_fps,
                            parents,
                            store_full_states,
                            stop_flag,
                            work_remaining,
                            max_queue,
                            max_depth_atomic,
                            total_transitions,
                            result_tx,
                            module,
                            extended_modules,
                            vars,
                            op_defs,
                            config,
                            check_deadlock,
                            next_name,
                            num_workers,
                            max_states_limit,
                            max_depth_limit,
                            registry,
                            compiled_invs,
                        );
                    })
                    .expect("Failed to spawn worker thread");
                handles.push(handle);
            }
        }

        // Drop our sender so result_rx will close when all workers are done
        drop(result_tx);

        // Spawn progress reporting thread if callback is set
        // Also handles capacity warnings for fingerprint storage
        let progress_handle = if let Some(ref callback) = self.progress_callback {
            if self.progress_interval_ms > 0 {
                let callback = Arc::clone(callback);
                let stop_flag = Arc::clone(&self.stop_flag);
                let seen = Arc::clone(&self.seen);
                let seen_fps = Arc::clone(&self.seen_fps);
                let max_depth = Arc::clone(&self.max_depth);
                let max_queue_depth = Arc::clone(&self.max_queue_depth);
                let total_transitions = Arc::clone(&self.total_transitions);
                let interval = Duration::from_millis(self.progress_interval_ms);
                let start_time = Instant::now();
                let store_full_states = self.store_full_states;
                // Track capacity status for warning suppression
                let last_capacity_status = Arc::new(AtomicU8::new(CAPACITY_NORMAL));

                Some(thread::spawn(move || {
                    while !stop_flag.load(Ordering::Relaxed) {
                        thread::sleep(interval);
                        if stop_flag.load(Ordering::Relaxed) {
                            break;
                        }
                        let states_found = if store_full_states {
                            seen.len()
                        } else {
                            seen_fps.len()
                        };
                        let elapsed_secs = start_time.elapsed().as_secs_f64();
                        let states_per_sec = if elapsed_secs > 0.0 {
                            states_found as f64 / elapsed_secs
                        } else {
                            0.0
                        };
                        let progress = Progress {
                            states_found,
                            current_depth: max_depth.load(Ordering::Relaxed),
                            queue_size: max_queue_depth.load(Ordering::Relaxed),
                            transitions: total_transitions.load(Ordering::Relaxed),
                            elapsed_secs,
                            states_per_sec,
                        };
                        callback(&progress);

                        // Check capacity warnings at progress intervals
                        check_and_warn_capacity(seen_fps.as_ref(), &last_capacity_status);
                    }
                }))
            } else {
                None
            }
        } else {
            None
        };

        // Collect results from workers
        let mut total_stats = WorkerStats::default();
        let mut first_violation: Option<(String, Fingerprint)> = None;
        let mut first_deadlock: Option<Fingerprint> = None;
        let mut first_error: Option<CheckError> = None;
        let mut first_limit: Option<LimitType> = None;

        // Aggregate all worker stats including profiling counters
        fn aggregate_stats(total: &mut WorkerStats, stats: &WorkerStats) {
            total.states_explored += stats.states_explored;
            total.transitions += stats.transitions;
            total.steals += stats.steals;
            total.injector_pushes += stats.injector_pushes;
            total.dedup_hits += stats.dedup_hits;
            total.empty_polls += stats.empty_polls;
            total.local_cache_hits += stats.local_cache_hits;
            // Timing stats
            total.enum_ns += stats.enum_ns;
            total.contains_ns += stats.contains_ns;
            total.insert_ns += stats.insert_ns;
            total.invariant_ns += stats.invariant_ns;
            total.materialize_ns += stats.materialize_ns;
        }

        for result in result_rx {
            match result {
                WorkerResult::Violation {
                    invariant,
                    state_fp,
                    stats,
                } => {
                    if first_violation.is_none() {
                        first_violation = Some((invariant, state_fp));
                    }
                    aggregate_stats(&mut total_stats, &stats);
                }
                WorkerResult::Deadlock { state_fp, stats } => {
                    if first_deadlock.is_none() && first_violation.is_none() {
                        first_deadlock = Some(state_fp);
                    }
                    aggregate_stats(&mut total_stats, &stats);
                }
                WorkerResult::Done(stats) => {
                    aggregate_stats(&mut total_stats, &stats);
                }
                WorkerResult::Error(e, stats) => {
                    if first_error.is_none()
                        && first_violation.is_none()
                        && first_deadlock.is_none()
                    {
                        first_error = Some(e);
                    }
                    aggregate_stats(&mut total_stats, &stats);
                }
                WorkerResult::LimitReached { limit_type, stats } => {
                    if first_limit.is_none()
                        && first_error.is_none()
                        && first_violation.is_none()
                        && first_deadlock.is_none()
                    {
                        first_limit = Some(limit_type);
                    }
                    aggregate_stats(&mut total_stats, &stats);
                }
            }
        }

        // Output profiling stats when TLA_PARALLEL_PROFILING=1
        if std::env::var("TLA_PARALLEL_PROFILING")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            eprintln!("=== Parallel Profiling Stats ===");
            eprintln!("  Workers: {}", self.num_workers);
            eprintln!("  Steals: {}", total_stats.steals);
            eprintln!("  Injector pushes: {}", total_stats.injector_pushes);
            eprintln!("  Dedup hits: {}", total_stats.dedup_hits);
            eprintln!("  Local cache hits: {}", total_stats.local_cache_hits);
            eprintln!("  Empty polls: {}", total_stats.empty_polls);
            eprintln!("  States explored: {}", total_stats.states_explored);
            eprintln!("  Transitions: {}", total_stats.transitions);
            if total_stats.transitions > 0 {
                let dedup_rate =
                    total_stats.dedup_hits as f64 / total_stats.transitions as f64 * 100.0;
                eprintln!("  Dedup rate: {:.1}%", dedup_rate);
                let cache_hit_rate =
                    total_stats.local_cache_hits as f64 / total_stats.transitions as f64 * 100.0;
                eprintln!("  Local cache hit rate: {:.1}%", cache_hit_rate);
            }
            eprintln!("================================");
        }

        // Output detailed timing when TLA2_TIMING=1
        if timing_enabled() {
            let total_ns = total_stats.enum_ns
                + total_stats.contains_ns
                + total_stats.insert_ns
                + total_stats.invariant_ns
                + total_stats.materialize_ns;
            eprintln!("=== Timing Breakdown (all workers) ===");
            eprintln!(
                "  Enumeration:    {:>8.2}ms ({:>5.1}%)",
                total_stats.enum_ns as f64 / 1_000_000.0,
                if total_ns > 0 {
                    total_stats.enum_ns as f64 / total_ns as f64 * 100.0
                } else {
                    0.0
                }
            );
            eprintln!(
                "  Contains check: {:>8.2}ms ({:>5.1}%)",
                total_stats.contains_ns as f64 / 1_000_000.0,
                if total_ns > 0 {
                    total_stats.contains_ns as f64 / total_ns as f64 * 100.0
                } else {
                    0.0
                }
            );
            eprintln!(
                "  Insert:         {:>8.2}ms ({:>5.1}%)",
                total_stats.insert_ns as f64 / 1_000_000.0,
                if total_ns > 0 {
                    total_stats.insert_ns as f64 / total_ns as f64 * 100.0
                } else {
                    0.0
                }
            );
            eprintln!(
                "  Invariant:      {:>8.2}ms ({:>5.1}%)",
                total_stats.invariant_ns as f64 / 1_000_000.0,
                if total_ns > 0 {
                    total_stats.invariant_ns as f64 / total_ns as f64 * 100.0
                } else {
                    0.0
                }
            );
            eprintln!(
                "  Materialize:    {:>8.2}ms ({:>5.1}%)",
                total_stats.materialize_ns as f64 / 1_000_000.0,
                if total_ns > 0 {
                    total_stats.materialize_ns as f64 / total_ns as f64 * 100.0
                } else {
                    0.0
                }
            );
            eprintln!("  Total measured: {:>8.2}ms", total_ns as f64 / 1_000_000.0);
            eprintln!("======================================");
        }

        // Output detailed enumeration profiling when TLA2_PROFILE_ENUM_DETAIL=1
        print_enum_profile_stats();

        // Wait for all workers to finish
        for handle in handles {
            let _ = handle.join();
        }

        // Stop and wait for progress thread
        if let Some(handle) = progress_handle {
            self.stop_flag.store(true, Ordering::SeqCst);
            let _ = handle.join();
        }

        let final_stats = CheckStats {
            states_found: self.states_count(),
            initial_states: num_initial,
            max_queue_depth: self.max_queue_depth.load(Ordering::SeqCst),
            transitions: total_stats.transitions,
            max_depth: self.max_depth.load(Ordering::SeqCst),
            detected_actions,
            coverage: None,
        };

        // Return the first violation/deadlock/limit/error found
        if let Some((invariant, state_fp)) = first_violation {
            let trace = self.reconstruct_trace(state_fp);
            return CheckResult::InvariantViolation {
                invariant,
                trace,
                stats: final_stats,
            };
        }

        if let Some(state_fp) = first_deadlock {
            let trace = self.reconstruct_trace(state_fp);
            return CheckResult::Deadlock {
                trace,
                stats: final_stats,
            };
        }

        if let Some(limit_type) = first_limit {
            return CheckResult::LimitReached {
                limit_type,
                stats: final_stats,
            };
        }

        if let Some(error) = first_error {
            return CheckResult::Error {
                error,
                stats: final_stats,
            };
        }

        // Check for fingerprint storage errors before returning success
        if self.seen_fps.has_errors() {
            let dropped = self.seen_fps.dropped_count();
            return CheckResult::Error {
                error: CheckError::FingerprintStorageOverflow { dropped },
                stats: final_stats,
            };
        }

        CheckResult::Success(final_stats)
    }

    /// Generate initial states directly to BulkStateStorage when possible.
    ///
    /// This is a memory-efficient alternative to `generate_initial_states` for workloads with
    /// very large initial-state frontiers (e.g., MCBakery ISpec with 655K initial states).
    ///
    /// Returns Ok(None) when streaming enumeration isn't possible so the caller can fall back
    /// to the Vec<State> path.
    fn generate_initial_states_to_bulk(
        &self,
        ctx: &mut EvalCtx,
        init_name: &str,
    ) -> Result<Option<BulkStateStorage>, CheckError> {
        let def = self.op_defs.get(init_name).ok_or(CheckError::MissingInit)?;
        let init_body = def.body.clone();

        // Try direct constraint extraction from Init predicate first.
        if let Some(branches) = extract_init_constraints(ctx, &init_body, &self.vars) {
            let unconstrained = find_unconstrained_vars(&self.vars, &branches);
            if unconstrained.is_empty() {
                let vars_len = ctx.var_registry().len();
                let mut storage = BulkStateStorage::new(vars_len, 1000);

                let count = enumerate_constraints_to_bulk(
                    ctx,
                    &self.vars,
                    &branches,
                    &mut storage,
                    |_values, _ctx| Ok(true),
                );

                return match count {
                    Some(_) => Ok(Some(storage)),
                    None => Ok(None),
                };
            }
        }

        // Fallback: enumerate from a bounded type predicate, then filter by the remainder
        // of Init (to avoid re-evaluating the candidate).
        let mut candidates: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for name in ["TypeOK", "TypeOk"] {
            if name != init_name && seen.insert(name) {
                candidates.push(name.to_string());
            }
        }
        for inv in &self.config.invariants {
            let inv_name = inv.as_str();
            if inv_name != init_name && seen.insert(inv_name) {
                candidates.push(inv.clone());
            }
        }

        for cand_name in candidates {
            let Some(cand_def) = self.op_defs.get(&cand_name) else {
                continue;
            };
            let Some(branches) = extract_init_constraints(ctx, &cand_def.body, &self.vars) else {
                continue;
            };

            let unconstrained = find_unconstrained_vars(&self.vars, &branches);
            if !unconstrained.is_empty() {
                continue;
            }

            let filter_expr = extract_conjunction_remainder(ctx, &init_body, &cand_name)
                .unwrap_or_else(|| init_body.clone());

            let filter_is_trivial = matches!(filter_expr.node, tla_core::ast::Expr::Bool(true));

            let compiled_filter = if filter_is_trivial {
                CompiledGuard::True
            } else {
                compile_guard_for_filter(ctx, &filter_expr, ctx.var_registry(), &LocalScope::new())
            };

            let vars_len = ctx.var_registry().len();
            let mut storage = BulkStateStorage::new(vars_len, 1000);

            let count = enumerate_constraints_to_bulk(
                ctx,
                &self.vars,
                &branches,
                &mut storage,
                |values, ctx| compiled_filter.eval_with_values(ctx, values),
            );

            if count.is_some() {
                return Ok(Some(storage));
            }
        }

        Ok(None)
    }

    /// Generate initial states
    ///
    /// First attempts direct constraint extraction from the Init predicate.
    /// If that fails (unsupported expressions or missing per-variable constraints),
    /// falls back to enumerating states from a type constraint (usually TypeOK)
    /// and filtering by evaluating the full Init predicate.
    fn generate_initial_states(
        &self,
        ctx: &EvalCtx,
        init_name: &str,
    ) -> Result<Vec<State>, CheckError> {
        let def = self.op_defs.get(init_name).ok_or(CheckError::MissingInit)?;
        let init_body = def.body.clone();

        // Try to extract constraints directly from the Init predicate
        let direct_hint = if let Some(branches) =
            extract_init_constraints(ctx, &init_body, &self.vars)
        {
            let unconstrained = find_unconstrained_vars(&self.vars, &branches);
            if unconstrained.is_empty() {
                return match enumerate_states_from_constraint_branches(
                    Some(ctx),
                    &self.vars,
                    &branches,
                ) {
                    Some(states) => Ok(states),
                    None => Err(CheckError::InitCannotEnumerate(
                        "failed to enumerate states from constraints".to_string(),
                    )),
                };
            }
            format!(
                "variable(s) {} have no constraints",
                unconstrained.join(", ")
            )
        } else {
            "Init predicate contains unsupported expressions (only equality, set membership, conjunction, disjunction, and TRUE/FALSE are supported)".to_string()
        };

        // Fallback: enumerate from a bounded type predicate, then filter by the full Init.
        let mut candidates: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();

        // Common type predicate names
        for name in ["TypeOK", "TypeOk"] {
            if name != init_name && seen.insert(name) {
                candidates.push(name.to_string());
            }
        }
        // Also consider configured invariants (often includes TypeOK)
        for inv in &self.config.invariants {
            let inv_name = inv.as_str();
            if inv_name != init_name && seen.insert(inv_name) {
                candidates.push(inv.clone());
            }
        }

        for cand_name in candidates {
            let Some(cand_def) = self.op_defs.get(&cand_name) else {
                continue;
            };
            let cand_body = cand_def.body.clone();

            let Some(branches) = extract_init_constraints(ctx, &cand_body, &self.vars) else {
                continue;
            };

            let unconstrained = find_unconstrained_vars(&self.vars, &branches);
            if !unconstrained.is_empty() {
                continue;
            }

            let Some(base_states) =
                enumerate_states_from_constraint_branches(Some(ctx), &self.vars, &branches)
            else {
                continue;
            };

            // Filter base states by the original Init predicate
            let mut filtered: Vec<State> = Vec::new();
            for state in base_states {
                let mut eval_ctx = ctx.clone();
                for (name, value) in state.vars() {
                    eval_ctx.bind_mut(Arc::clone(name), value.clone());
                }
                let keep = match eval(&eval_ctx, &init_body) {
                    Ok(Value::Bool(b)) => b,
                    Ok(_) => {
                        return Err(CheckError::InitNotBoolean);
                    }
                    Err(e) => {
                        return Err(CheckError::EvalError(e));
                    }
                };

                if keep {
                    filtered.push(state);
                }
            }

            return Ok(filtered);
        }

        Err(CheckError::InitCannotEnumerate(direct_hint))
    }

    /// Reconstruct trace from initial state to given state
    fn reconstruct_trace(&self, end_fp: Fingerprint) -> Trace {
        if !self.store_full_states {
            return Trace::new();
        }

        let mut fps = Vec::new();
        let mut current = end_fp;

        // Walk back through parents
        while let Some(entry) = self.parents.get(&current) {
            fps.push(current);
            current = *entry;
        }
        fps.push(current); // Add initial state

        // Reverse to get initial -> end order
        fps.reverse();

        // Convert fingerprints to ArrayStates, then to States on demand
        let states: Vec<State> = fps
            .iter()
            .filter_map(|fp| {
                self.seen
                    .get(fp)
                    .map(|entry| entry.to_state(&self.var_registry))
            })
            .collect();

        Trace::from_states(states)
    }
}

/// Adaptive backoff thresholds for empty poll handling.
/// After this many consecutive empty polls, start sleeping instead of yielding.
const BACKOFF_YIELD_THRESHOLD: usize = 3;
/// After this many consecutive empty polls, use longer sleep.
const BACKOFF_LONG_THRESHOLD: usize = 10;

/// Worker thread function
#[allow(clippy::too_many_arguments)]
fn run_worker(
    _worker_id: usize,
    local_queue: Worker<(ArrayState, usize)>,
    injector: Arc<Injector<(ArrayState, usize)>>,
    stealers: Arc<Vec<Stealer<(ArrayState, usize)>>>,
    seen: Arc<FxDashMap<Fingerprint, ArrayState>>,
    seen_fps: Arc<dyn FingerprintSet>,
    parents: Arc<FxDashMap<Fingerprint, Fingerprint>>,
    store_full_states: bool,
    stop_flag: Arc<AtomicBool>,
    work_remaining: Arc<AtomicUsize>,
    max_queue: Arc<AtomicUsize>,
    max_depth_atomic: Arc<AtomicUsize>,
    total_transitions: Arc<AtomicUsize>,
    result_tx: Sender<WorkerResult>,
    module: Arc<Module>,
    extended_modules: Arc<Vec<Module>>,
    vars: Vec<Arc<str>>,
    op_defs: HashMap<String, OperatorDef>,
    config: Config,
    check_deadlock: bool,
    next_name: String,
    num_workers: usize,
    max_states_limit: Option<usize>,
    max_depth_limit: Option<usize>,
    var_registry: Arc<VarRegistry>,
    compiled_invariants: Arc<Vec<(String, CompiledGuard)>>,
) {
    let mut stats = WorkerStats::default();
    let mut ctx = EvalCtx::new();

    let mut local_max_queue: usize = 0;

    // Batched work counter to reduce atomic contention.
    // Positive = new states queued, negative = states completed.
    // Flushed to global work_remaining when |delta| >= WORK_BATCH_SIZE.
    let mut local_work_delta: isize = 0;

    // Adaptive backoff state - tracks consecutive empty polls for exponential backoff
    let mut consecutive_empty: usize = 0;

    // Helper to flush local work delta to global counter.
    // Returns true if global work_remaining is now zero (potential termination).
    #[inline]
    fn flush_work_delta(delta: &mut isize, work_remaining: &AtomicUsize) -> bool {
        if *delta == 0 {
            return work_remaining.load(Ordering::Acquire) == 0;
        }
        if *delta > 0 {
            work_remaining.fetch_add(*delta as usize, Ordering::Release);
        } else {
            // Saturating sub to avoid underflow
            let sub_amount = (-*delta) as usize;
            let prev = work_remaining.fetch_sub(sub_amount, Ordering::Release);
            // If we would underflow, add back (shouldn't happen in practice)
            if prev < sub_amount {
                work_remaining.fetch_add(sub_amount - prev, Ordering::Release);
            }
        }
        *delta = 0;
        work_remaining.load(Ordering::Acquire) == 0
    }

    // Per-worker local duplicate cache - avoids hitting shared FP set for hot duplicates
    // Uses FxHashSet for O(1) lookup with minimal hash overhead (FP is already a hash)
    let mut local_cache: FxHashSet<Fingerprint> = FxHashSet::default();
    local_cache.reserve(LOCAL_CACHE_SIZE);

    // Load extended modules first (in order)
    for ext_mod in extended_modules.iter() {
        ctx.load_module(ext_mod);
    }
    // Load main module last (can override)
    ctx.load_module(&module);

    // For each registered instance, load the instanced module's operators
    let instance_module_names: Vec<String> = ctx
        .instances()
        .values()
        .map(|info| info.module_name.clone())
        .collect();
    for module_name in &instance_module_names {
        for ext_mod in extended_modules.iter() {
            if ext_mod.name.node == *module_name {
                ctx.load_instance_module(module_name.clone(), ext_mod);
                break;
            }
        }
    }

    // Register state variables in VarRegistry for efficient array-based state handling
    // This must match the order in var_registry parameter for correct indexing
    ctx.register_vars(vars.iter().cloned());

    // Bind constants from config
    if let Err(e) = bind_constants_from_config(&mut ctx, &config) {
        stop_flag.store(true, Ordering::SeqCst);
        let _ = result_tx.send(WorkerResult::Error(CheckError::EvalError(e), stats));
        return;
    }

    loop {
        // Check for early termination
        if stop_flag.load(Ordering::Relaxed) {
            let _ = result_tx.send(WorkerResult::Done(stats));
            return;
        }

        // Check state limit
        if let Some(max_states) = max_states_limit {
            let states_found = if store_full_states {
                seen.len()
            } else {
                seen_fps.len()
            };
            if states_found >= max_states {
                stop_flag.store(true, Ordering::SeqCst);
                let _ = result_tx.send(WorkerResult::LimitReached {
                    limit_type: LimitType::States,
                    stats,
                });
                return;
            }
        }

        // Try to get work: local queue first, then steal
        // Queue contains (ArrayState, depth) tuples for efficient depth tracking
        let work_item = if let Some(item) = local_queue.pop() {
            Some(item)
        } else {
            // Try to steal from global injector
            let from_injector =
                std::iter::repeat_with(|| injector.steal_batch_and_pop(&local_queue))
                    .find(|s| !s.is_retry())
                    .and_then(|s| s.success());
            if from_injector.is_some() {
                stats.steals += 1;
                from_injector
            } else {
                // Try to steal from other workers
                let from_worker = stealers
                    .iter()
                    .map(|s| s.steal())
                    .find(|s| !s.is_retry())
                    .and_then(|s| s.success());
                if from_worker.is_some() {
                    stats.steals += 1;
                }
                from_worker
            }
        };

        let (mut current_array, current_depth) = match work_item {
            Some(item) => {
                // Reset backoff counter when work is found
                consecutive_empty = 0;
                item
            }
            None => {
                stats.empty_polls += 1;
                consecutive_empty += 1;

                // No work visible to this worker. Flush local delta and check termination.
                // If there are no outstanding states (queued or in-progress), exploration is complete.
                if flush_work_delta(&mut local_work_delta, &work_remaining) {
                    let _ = result_tx.send(WorkerResult::Done(stats));
                    return;
                }

                // Adaptive backoff: avoid spinning when work is scarce
                // This dramatically reduces CPU waste vs TLC's blocking queue approach
                if consecutive_empty < BACKOFF_YIELD_THRESHOLD {
                    // First few empty polls: quick retry via yield
                    thread::yield_now();
                } else if consecutive_empty < BACKOFF_LONG_THRESHOLD {
                    // Moderate backoff: short sleep (10μs)
                    thread::sleep(Duration::from_micros(10));
                } else {
                    // Extended backoff: longer sleep (100μs)
                    // Work distribution has stabilized; avoid CPU waste
                    thread::sleep(Duration::from_micros(100));
                }
                continue;
            }
        };

        let fp = current_array.fingerprint(&var_registry);
        stats.states_explored += 1;

        // Depth is now embedded in work item - no DashMap lookup needed!

        // Check depth limit - skip generating successors if at max depth
        if let Some(max_depth) = max_depth_limit {
            if current_depth >= max_depth {
                // At depth limit - don't generate successors, signal limit reached
                stop_flag.store(true, Ordering::SeqCst);
                let _ = result_tx.send(WorkerResult::LimitReached {
                    limit_type: LimitType::Depth,
                    stats,
                });
                return;
            }
        }

        // Update max queue depth
        let queue_size = local_queue.len();
        if queue_size > local_max_queue {
            local_max_queue = queue_size;
            let _ = max_queue.fetch_max(queue_size, Ordering::Relaxed);
        }

        // Generate successors using AST-based enumeration
        {
            let state = current_array.to_state(&var_registry);

            // Bind state to env for State-based enumeration
            let saved = ctx.save_scope();
            for (name, value) in state.vars() {
                ctx.bind_mut(Arc::clone(name), value.clone());
            }

            let def = match op_defs.get(&next_name) {
                Some(d) => d,
                None => {
                    ctx.restore_scope(saved);
                    stop_flag.store(true, Ordering::SeqCst);
                    let _ = result_tx.send(WorkerResult::Error(CheckError::MissingNext, stats));
                    return;
                }
            };

            let successors = match enumerate_successors(&mut ctx, def, &state, &vars) {
                Ok(s) => s,
                Err(e) => {
                    ctx.restore_scope(saved);
                    stop_flag.store(true, Ordering::SeqCst);
                    let _ = result_tx.send(WorkerResult::Error(CheckError::EvalError(e), stats));
                    return;
                }
            };

            ctx.restore_scope(saved);

            // Deadlock check
            if check_deadlock && successors.is_empty() {
                stop_flag.store(true, Ordering::SeqCst);
                let _ = result_tx.send(WorkerResult::Deadlock {
                    state_fp: fp,
                    stats,
                });
                return;
            }

            stats.transitions += successors.len();
            total_transitions.fetch_add(successors.len(), Ordering::Relaxed);

            let succ_depth = current_depth + 1;

            // Process State-based successors (original path)
            for succ in successors {
                let succ_fp = succ.fingerprint();

                // Check state constraints first
                if !config.constraints.is_empty() {
                    let saved = ctx.save_scope();
                    for (name, value) in succ.vars() {
                        ctx.bind_mut(Arc::clone(name), value.clone());
                    }
                    let satisfies_constraints = config.constraints.iter().all(|constraint_name| {
                        matches!(ctx.eval_op(constraint_name), Ok(Value::Bool(true)))
                    });
                    ctx.restore_scope(saved);
                    if !satisfies_constraints {
                        continue;
                    }
                }

                // Check action constraints
                if !config.action_constraints.is_empty() {
                    let saved = ctx.save_scope();
                    let saved_next = ctx.next_state.clone();
                    for (name, value) in state.vars() {
                        ctx.bind_mut(Arc::clone(name), value.clone());
                    }
                    let mut next_env = Env::new();
                    for (name, value) in succ.vars() {
                        next_env.insert(Arc::clone(name), value.clone());
                    }
                    ctx.next_state = Some(std::sync::Arc::new(next_env));
                    let satisfies_action_constraints =
                        config.action_constraints.iter().all(|constraint_name| {
                            matches!(ctx.eval_op(constraint_name), Ok(Value::Bool(true)))
                        });
                    ctx.next_state = saved_next;
                    ctx.restore_scope(saved);
                    if !satisfies_action_constraints {
                        continue;
                    }
                }

                // Convert State to ArrayState early for storage and queuing
                let succ_arr = ArrayState::from_state(&succ, &var_registry);

                // Check if state is new
                let is_new = if store_full_states {
                    seen.insert(succ_fp, succ_arr.clone()).is_none()
                } else {
                    seen_fps.insert(succ_fp)
                };

                if is_new {
                    // Batch work counter update to reduce atomic contention
                    local_work_delta += 1;
                    if local_work_delta >= WORK_BATCH_SIZE as isize {
                        flush_work_delta(&mut local_work_delta, &work_remaining);
                    }
                    if store_full_states {
                        parents.insert(succ_fp, fp);
                    }
                    // Depth embedded in queue item - only update max depth atomic
                    let _ = max_depth_atomic.fetch_max(succ_depth, Ordering::Relaxed);

                    if !config.invariants.is_empty() {
                        let prev = ctx.bind_state_array(succ_arr.values());
                        let violated = if !compiled_invariants.is_empty() {
                            compiled_invariants
                                .iter()
                                .find(|(_, guard)| match guard.eval_with_array(&mut ctx, &succ_arr)
                                {
                                    Ok(true) => false,
                                    Ok(false) => true,
                                    Err(_) => true,
                                })
                                .map(|(name, _)| name.clone())
                        } else {
                            config
                                .invariants
                                .iter()
                                .find(|inv_name| match ctx.eval_op(inv_name) {
                                    Ok(Value::Bool(true)) => false,
                                    Ok(Value::Bool(false)) => true,
                                    _ => true,
                                })
                                .cloned()
                        };
                        ctx.restore_state_env(prev);

                        if let Some(invariant) = violated {
                            stop_flag.store(true, Ordering::SeqCst);
                            let _ = result_tx.send(WorkerResult::Violation {
                                invariant,
                                state_fp: succ_fp,
                                stats,
                            });
                            return;
                        }
                    }

                    // Push to global injector periodically for work stealing, keep rest local
                    if stats.states_explored % num_workers == 0 {
                        injector.push((succ_arr, succ_depth));
                    } else {
                        local_queue.push((succ_arr, succ_depth));
                    }
                }
            }

            // Batch work counter update for completing this state
            local_work_delta -= 1;
            if local_work_delta <= -(WORK_BATCH_SIZE as isize) {
                flush_work_delta(&mut local_work_delta, &work_remaining);
            }
        }
    }
}

/// Worker thread function using HandleState for reduced Arc contention.
///
/// This is an optimized version of `run_worker` that uses `HandleState` instead
/// of `ArrayState` in the work queue. HandleState cloning is a pure memcpy with
/// no atomic operations, eliminating the Arc contention that limits parallel scaling.
///
/// # Performance
///
/// In the standard `run_worker`:
/// - State cloning requires Arc atomic increments for each Value with shared data
/// - With 8 threads and millions of states, this causes cache line bouncing
///
/// With `run_worker_handle`:
/// - State cloning is memcpy of `Box<[ValueHandle]>` (8 bytes per variable)
/// - Zero atomic operations during cloning
/// - Values are looked up from interner only when needed (lazy materialization)
#[allow(clippy::too_many_arguments)]
fn run_worker_handle(
    _worker_id: usize,
    local_queue: Worker<(HandleState, usize)>,
    injector: Arc<Injector<(HandleState, usize)>>,
    stealers: Arc<Vec<Stealer<(HandleState, usize)>>>,
    seen: Arc<FxDashMap<Fingerprint, ArrayState>>,
    seen_fps: Arc<dyn FingerprintSet>,
    parents: Arc<FxDashMap<Fingerprint, Fingerprint>>,
    store_full_states: bool,
    stop_flag: Arc<AtomicBool>,
    work_remaining: Arc<AtomicUsize>,
    max_queue: Arc<AtomicUsize>,
    max_depth_atomic: Arc<AtomicUsize>,
    total_transitions: Arc<AtomicUsize>,
    result_tx: Sender<WorkerResult>,
    module: Arc<Module>,
    extended_modules: Arc<Vec<Module>>,
    vars: Vec<Arc<str>>,
    op_defs: HashMap<String, OperatorDef>,
    config: Config,
    check_deadlock: bool,
    next_name: String,
    num_workers: usize,
    max_states_limit: Option<usize>,
    max_depth_limit: Option<usize>,
    var_registry: Arc<VarRegistry>,
    compiled_invariants: Arc<Vec<(String, CompiledGuard)>>,
    interner: &'static ValueInterner,
) {
    let mut stats = WorkerStats::default();
    let mut ctx = EvalCtx::new();

    let mut local_max_queue: usize = 0;

    // Batched work counter to reduce atomic contention.
    let mut local_work_delta: isize = 0;

    // Adaptive backoff state
    let mut consecutive_empty: usize = 0;

    // Helper to flush local work delta to global counter.
    #[inline]
    fn flush_work_delta(delta: &mut isize, work_remaining: &AtomicUsize) -> bool {
        if *delta == 0 {
            return work_remaining.load(Ordering::Acquire) == 0;
        }
        if *delta > 0 {
            work_remaining.fetch_add(*delta as usize, Ordering::Release);
        } else {
            let sub_amount = (-*delta) as usize;
            let prev = work_remaining.fetch_sub(sub_amount, Ordering::Release);
            if prev < sub_amount {
                work_remaining.fetch_add(sub_amount - prev, Ordering::Release);
            }
        }
        *delta = 0;
        work_remaining.load(Ordering::Acquire) == 0
    }

    // Per-worker local duplicate cache
    let mut local_cache: FxHashSet<Fingerprint> = FxHashSet::default();
    local_cache.reserve(LOCAL_CACHE_SIZE);

    // Load modules
    for ext_mod in extended_modules.iter() {
        ctx.load_module(ext_mod);
    }
    ctx.load_module(&module);

    // Load instanced modules
    let instance_module_names: Vec<String> = ctx
        .instances()
        .values()
        .map(|info| info.module_name.clone())
        .collect();
    for module_name in &instance_module_names {
        for ext_mod in extended_modules.iter() {
            if ext_mod.name.node == *module_name {
                ctx.load_instance_module(module_name.clone(), ext_mod);
                break;
            }
        }
    }

    ctx.register_vars(vars.iter().cloned());

    if let Err(e) = bind_constants_from_config(&mut ctx, &config) {
        stop_flag.store(true, Ordering::SeqCst);
        let _ = result_tx.send(WorkerResult::Error(CheckError::EvalError(e), stats));
        return;
    }

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            let _ = result_tx.send(WorkerResult::Done(stats));
            return;
        }

        if let Some(max_states) = max_states_limit {
            let states_found = if store_full_states {
                seen.len()
            } else {
                seen_fps.len()
            };
            if states_found >= max_states {
                stop_flag.store(true, Ordering::SeqCst);
                let _ = result_tx.send(WorkerResult::LimitReached {
                    limit_type: LimitType::States,
                    stats,
                });
                return;
            }
        }

        // Try to get work from queues
        let work_item = if let Some(item) = local_queue.pop() {
            Some(item)
        } else {
            let from_injector =
                std::iter::repeat_with(|| injector.steal_batch_and_pop(&local_queue))
                    .find(|s| !s.is_retry())
                    .and_then(|s| s.success());
            if from_injector.is_some() {
                stats.steals += 1;
                from_injector
            } else {
                let from_worker = stealers
                    .iter()
                    .map(|s| s.steal())
                    .find(|s| !s.is_retry())
                    .and_then(|s| s.success());
                if from_worker.is_some() {
                    stats.steals += 1;
                }
                from_worker
            }
        };

        let (current_handle, current_depth) = match work_item {
            Some(item) => {
                consecutive_empty = 0;
                item
            }
            None => {
                stats.empty_polls += 1;
                consecutive_empty += 1;

                if flush_work_delta(&mut local_work_delta, &work_remaining) {
                    let _ = result_tx.send(WorkerResult::Done(stats));
                    return;
                }

                if consecutive_empty < BACKOFF_YIELD_THRESHOLD {
                    thread::yield_now();
                } else if consecutive_empty < BACKOFF_LONG_THRESHOLD {
                    thread::sleep(Duration::from_micros(10));
                } else {
                    thread::sleep(Duration::from_micros(100));
                }
                continue;
            }
        };

        let fp = current_handle.fingerprint();
        stats.states_explored += 1;

        if let Some(max_depth) = max_depth_limit {
            if current_depth >= max_depth {
                stop_flag.store(true, Ordering::SeqCst);
                let _ = result_tx.send(WorkerResult::LimitReached {
                    limit_type: LimitType::Depth,
                    stats,
                });
                return;
            }
        }

        let queue_size = local_queue.len();
        if queue_size > local_max_queue {
            local_max_queue = queue_size;
            let _ = max_queue.fetch_max(queue_size, Ordering::Relaxed);
        }

        // Materialize HandleState to ArrayState for enumeration
        // This is the lazy materialization point - values are looked up only when needed
        let current_values = current_handle.materialize(interner);
        let current_array = ArrayState::from_values(current_values);

        // Generate successors using AST-based enumeration
        {
            let state = current_array.to_state(&var_registry);

            let saved = ctx.save_scope();
            for (name, value) in state.vars() {
                ctx.bind_mut(Arc::clone(name), value.clone());
            }

            let def = match op_defs.get(&next_name) {
                Some(d) => d,
                None => {
                    ctx.restore_scope(saved);
                    stop_flag.store(true, Ordering::SeqCst);
                    let _ = result_tx.send(WorkerResult::Error(CheckError::MissingNext, stats));
                    return;
                }
            };

            let successors = match enumerate_successors(&mut ctx, def, &state, &vars) {
                Ok(s) => s,
                Err(e) => {
                    ctx.restore_scope(saved);
                    stop_flag.store(true, Ordering::SeqCst);
                    let _ = result_tx.send(WorkerResult::Error(CheckError::EvalError(e), stats));
                    return;
                }
            };

            ctx.restore_scope(saved);

            if check_deadlock && successors.is_empty() {
                stop_flag.store(true, Ordering::SeqCst);
                let _ = result_tx.send(WorkerResult::Deadlock {
                    state_fp: fp,
                    stats,
                });
                return;
            }

            stats.transitions += successors.len();
            total_transitions.fetch_add(successors.len(), Ordering::Relaxed);

            let succ_depth = current_depth + 1;

            // Process State-based successors (fallback path)
            for succ in successors {
                let succ_fp = succ.fingerprint();

                if !config.constraints.is_empty() {
                    let saved = ctx.save_scope();
                    for (name, value) in succ.vars() {
                        ctx.bind_mut(Arc::clone(name), value.clone());
                    }
                    let satisfies_constraints = config.constraints.iter().all(|constraint_name| {
                        matches!(ctx.eval_op(constraint_name), Ok(Value::Bool(true)))
                    });
                    ctx.restore_scope(saved);
                    if !satisfies_constraints {
                        continue;
                    }
                }

                if !config.action_constraints.is_empty() {
                    let saved = ctx.save_scope();
                    let saved_next = ctx.next_state.clone();
                    for (name, value) in state.vars() {
                        ctx.bind_mut(Arc::clone(name), value.clone());
                    }
                    let mut next_env = Env::new();
                    for (name, value) in succ.vars() {
                        next_env.insert(Arc::clone(name), value.clone());
                    }
                    ctx.next_state = Some(std::sync::Arc::new(next_env));
                    let satisfies_action_constraints =
                        config.action_constraints.iter().all(|constraint_name| {
                            matches!(ctx.eval_op(constraint_name), Ok(Value::Bool(true)))
                        });
                    ctx.next_state = saved_next;
                    ctx.restore_scope(saved);
                    if !satisfies_action_constraints {
                        continue;
                    }
                }

                // Convert State to ArrayState early for storage and queuing
                let succ_arr = ArrayState::from_state(&succ, &var_registry);

                let is_new = if store_full_states {
                    seen.insert(succ_fp, succ_arr.clone()).is_none()
                } else {
                    seen_fps.insert(succ_fp)
                };

                if is_new {
                    local_work_delta += 1;
                    if local_work_delta >= WORK_BATCH_SIZE as isize {
                        flush_work_delta(&mut local_work_delta, &work_remaining);
                    }
                    if store_full_states {
                        parents.insert(succ_fp, fp);
                    }
                    let _ = max_depth_atomic.fetch_max(succ_depth, Ordering::Relaxed);

                    if !config.invariants.is_empty() {
                        let prev = ctx.bind_state_array(succ_arr.values());
                        let violated = if !compiled_invariants.is_empty() {
                            compiled_invariants
                                .iter()
                                .find(|(_, guard)| match guard.eval_with_array(&mut ctx, &succ_arr)
                                {
                                    Ok(true) => false,
                                    Ok(false) => true,
                                    Err(_) => true,
                                })
                                .map(|(name, _)| name.clone())
                        } else {
                            config
                                .invariants
                                .iter()
                                .find(|inv_name| match ctx.eval_op(inv_name) {
                                    Ok(Value::Bool(true)) => false,
                                    Ok(Value::Bool(false)) => true,
                                    _ => true,
                                })
                                .cloned()
                        };
                        ctx.restore_state_env(prev);

                        if let Some(invariant) = violated {
                            stop_flag.store(true, Ordering::SeqCst);
                            let _ = result_tx.send(WorkerResult::Violation {
                                invariant,
                                state_fp: succ_fp,
                                stats,
                            });
                            return;
                        }
                    }

                    // Convert ArrayState to HandleState for queuing (key optimization!)
                    let succ_handle =
                        HandleState::from_values(succ_arr.values(), &var_registry, interner);
                    if stats.states_explored % num_workers == 0 {
                        injector.push((succ_handle, succ_depth));
                    } else {
                        local_queue.push((succ_handle, succ_depth));
                    }
                }
            }

            local_work_delta -= 1;
            if local_work_delta <= -(WORK_BATCH_SIZE as isize) {
                flush_work_delta(&mut local_work_delta, &work_remaining);
            }
        }
    }
}

/// Run parallel model checking on a module
///
/// # Arguments
/// * `module` - The TLA+ module to check
/// * `config` - Model checking configuration
/// * `num_workers` - Number of worker threads (0 = auto-detect based on CPU count)
///
/// # Returns
/// The result of model checking (success, violation, deadlock, or error)
pub fn check_module_parallel(module: &Module, config: &Config, num_workers: usize) -> CheckResult {
    let checker = ParallelChecker::new(module, config, num_workers);
    checker.check()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    fn parse_module(src: &str) -> Module {
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        lower_result.module.unwrap()
    }

    #[test]
    fn test_parallel_simple_counter() {
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
InRange == x >= 0 /\ x <= 2
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["InRange".to_string()],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 3);
                assert_eq!(stats.initial_states, 1);
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_no_trace_mode_success() {
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
InRange == x >= 0 /\ x <= 2
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["InRange".to_string()],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);
        checker.set_store_states(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 3);
                assert_eq!(stats.initial_states, 1);
            }
            other => panic!("Expected success, got: {:?}", other),
        }

        // Verify internal state: seen_fps should be populated, seen should be empty
        assert!(!checker.store_full_states);
        assert!(
            checker.seen.is_empty(),
            "seen map should be empty in no-trace mode"
        );
        assert_eq!(checker.seen_fps.len(), 3, "seen_fps should have 3 entries");
    }

    #[test]
    fn test_parallel_invariant_violation() {
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
SmallValue == x < 3
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["SmallValue".to_string()],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::InvariantViolation {
                invariant,
                trace,
                stats,
            } => {
                assert_eq!(invariant, "SmallValue");
                assert_eq!(trace.len(), 4); // x=0, x=1, x=2, x=3
                assert!(stats.states_found >= 3);
            }
            other => panic!("Expected invariant violation, got: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_no_trace_mode_violation_empty_trace() {
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
SmallValue == x < 3
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["SmallValue".to_string()],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);
        checker.set_store_states(false);

        let result = checker.check();

        match result {
            CheckResult::InvariantViolation {
                invariant,
                trace,
                stats,
            } => {
                assert_eq!(invariant, "SmallValue");
                assert!(trace.is_empty(), "Trace should be empty in no-trace mode");
                assert_eq!(stats.states_found, 4);
            }
            other => panic!("Expected invariant violation, got: {:?}", other),
        }

        assert!(!checker.store_full_states);
        assert!(
            checker.seen.is_empty(),
            "seen map should be empty in no-trace mode"
        );
        assert_eq!(checker.seen_fps.len(), 4, "seen_fps should have 4 entries");
    }

    #[test]
    fn test_parallel_multiple_initial_states() {
        let src = r#"
---- MODULE Multi ----
VARIABLE x
Init == x \in {0, 1}
Next == x < 3 /\ x' = x + 1
InRange == x >= 0 /\ x <= 3
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["InRange".to_string()],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.initial_states, 2);
                assert_eq!(stats.states_found, 4); // x=0,1,2,3
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_two_variables() {
        let src = r#"
---- MODULE TwoVars ----
VARIABLE x, y
Init == x = 0 /\ y = 5
Next == x' = x + 1 /\ UNCHANGED y
Bounded == x < 2
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Bounded".to_string()],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::InvariantViolation {
                invariant,
                trace,
                stats: _,
            } => {
                assert_eq!(invariant, "Bounded");
                assert!(trace.len() >= 3);

                // Verify y stayed unchanged
                for state in &trace.states {
                    let y_val = state.vars().find(|(n, _)| n.as_ref() == "y");
                    assert!(y_val.is_some());
                    assert_eq!(y_val.unwrap().1, &Value::int(5));
                }
            }
            other => panic!("Expected invariant violation, got: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_vs_sequential_consistency() {
        // Verify that parallel and sequential checkers produce the same result
        let src = r#"
---- MODULE Consistency ----
VARIABLE x, y
Init == x \in {0, 1} /\ y \in {0, 1}
Next == x' \in {0, 1} /\ y' \in {0, 1}
Valid == x + y < 3
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Valid".to_string()],
            ..Default::default()
        };

        // Run sequential checker
        let mut seq_checker = crate::check::ModelChecker::new(&module, &config);
        seq_checker.set_deadlock_check(false);
        let seq_result = seq_checker.check();

        // Run parallel checker
        let mut par_checker = ParallelChecker::new(&module, &config, 4);
        par_checker.set_deadlock_check(false);
        let par_result = par_checker.check();

        // Both should succeed with same state count
        match (seq_result, par_result) {
            (CheckResult::Success(seq_stats), CheckResult::Success(par_stats)) => {
                assert_eq!(seq_stats.states_found, par_stats.states_found);
                assert_eq!(seq_stats.initial_states, par_stats.initial_states);
            }
            (seq, par) => panic!("Results differ: seq={:?}, par={:?}", seq, par),
        }
    }

    #[test]
    fn test_parallel_does_not_drop_work_items() {
        // Regression test: parallel exploration must not silently miss states.
        //
        // This is a self-contained "Majority-like" spec with a known, moderate-sized
        // state space (2733 states). We compare sequential vs parallel results.
        let src = r#"
---- MODULE ParMajority ----
CONSTANTS A, B, C

Value == {A, B, C}
BoundedSeq(S) == UNION { [1 .. n -> S] : n \in 0 .. 5 }

VARIABLES seq, i, cand, cnt

Init ==
    /\ seq \in BoundedSeq(Value)
    /\ i = 1
    /\ cand \in Value
    /\ cnt = 0

Next ==
    /\ i <= Len(seq)
    /\ i' = i + 1
    /\ seq' = seq
    /\ \/ /\ cnt = 0
          /\ cand' = seq[i]
          /\ cnt' = 1
       \/ /\ cnt # 0 /\ cand = seq[i]
          /\ cand' = cand
          /\ cnt' = cnt + 1
       \/ /\ cnt # 0 /\ cand # seq[i]
          /\ cand' = cand
          /\ cnt' = cnt - 1
====
"#;
        let module = parse_module(src);

        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        config.constants.insert(
            "A".to_string(),
            crate::config::ConstantValue::Value("A".to_string()),
        );
        config.constants.insert(
            "B".to_string(),
            crate::config::ConstantValue::Value("B".to_string()),
        );
        config.constants.insert(
            "C".to_string(),
            crate::config::ConstantValue::Value("C".to_string()),
        );

        // Run sequential checker
        let mut seq_checker = crate::check::ModelChecker::new(&module, &config);
        let seq_result = seq_checker.check();

        // Run parallel checker
        let par_checker = ParallelChecker::new(&module, &config, 8);
        let par_result = par_checker.check();

        match (seq_result, par_result) {
            (CheckResult::Success(seq_stats), CheckResult::Success(par_stats)) => {
                assert_eq!(seq_stats.initial_states, 1092);
                assert_eq!(seq_stats.states_found, 2733);
                assert_eq!(seq_stats.transitions, 2367);

                assert_eq!(par_stats.initial_states, 1092);
                assert_eq!(par_stats.states_found, 2733);
                assert_eq!(par_stats.transitions, 2367);
            }
            (seq, par) => panic!("Results differ: seq={:?}, par={:?}", seq, par),
        }
    }

    #[test]
    fn test_parallel_single_worker() {
        // Test with single worker (should behave like sequential)
        let src = r#"
---- MODULE Single ----
VARIABLE x
Init == x = 0
Next == x < 3 /\ x' = x + 1
InRange == x >= 0 /\ x <= 3
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["InRange".to_string()],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 1);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 4);
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_missing_init() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
Next == x' = x + 1
====
"#;
        let module = parse_module(src);

        let config = Config {
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let checker = ParallelChecker::new(&module, &config, 2);
        let result = checker.check();

        assert!(matches!(
            result,
            CheckResult::Error {
                error: CheckError::MissingInit,
                ..
            }
        ));
    }

    #[test]
    fn test_parallel_missing_next() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            ..Default::default()
        };

        let checker = ParallelChecker::new(&module, &config, 2);
        let result = checker.check();

        assert!(matches!(
            result,
            CheckResult::Error {
                error: CheckError::MissingNext,
                ..
            }
        ));
    }

    // ============================
    // Exploration limit tests
    // ============================

    #[test]
    fn test_parallel_max_states_limit() {
        // Unbounded counter that would run forever without limits
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);
        checker.set_max_states(5);

        let result = checker.check();

        match result {
            CheckResult::LimitReached { limit_type, stats } => {
                assert_eq!(limit_type, LimitType::States);
                // Should have found at least 5 states (might have a few more due to parallelism)
                assert!(stats.states_found >= 5);
            }
            other => panic!("Expected LimitReached(States), got: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_max_depth_limit() {
        // Unbounded counter that would run forever without limits
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);
        checker.set_max_depth(3);

        let result = checker.check();

        match result {
            CheckResult::LimitReached { limit_type, stats } => {
                assert_eq!(limit_type, LimitType::Depth);
                // Should have explored depth 0, 1, 2, 3 (up to 4 states)
                assert!(stats.states_found >= 3);
                assert!(stats.max_depth <= 3);
            }
            other => panic!("Expected LimitReached(Depth), got: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_invariant_found_before_limit() {
        // Counter with invariant that will be violated before hitting limit
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
SmallValue == x < 3
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["SmallValue".to_string()],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);
        checker.set_max_states(100); // High limit that won't be reached

        let result = checker.check();

        match result {
            CheckResult::InvariantViolation {
                invariant, stats, ..
            } => {
                assert_eq!(invariant, "SmallValue");
                // Should find violation at x=3 before hitting 100 state limit
                assert!(stats.states_found < 100);
            }
            other => panic!("Expected InvariantViolation, got: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_success_within_limits() {
        // Bounded counter that terminates naturally
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x < 3 /\ x' = x + 1
InRange == x >= 0 /\ x <= 3
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["InRange".to_string()],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);
        checker.set_max_states(100);
        checker.set_max_depth(100);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // Should complete naturally with 4 states
                assert_eq!(stats.states_found, 4);
            }
            other => panic!("Expected Success, got: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_depth_tracking() {
        // Bounded counter to verify depth tracking
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x < 5 /\ x' = x + 1
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 6); // x=0,1,2,3,4,5
                assert_eq!(stats.max_depth, 5); // 0->1->2->3->4->5
            }
            other => panic!("Expected Success, got: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_limits_single_worker() {
        // Test limits with single worker for deterministic behavior
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let mut checker = ParallelChecker::new(&module, &config, 1);
        checker.set_deadlock_check(false);
        checker.set_max_states(10);

        let result = checker.check();

        match result {
            CheckResult::LimitReached { limit_type, stats } => {
                assert_eq!(limit_type, LimitType::States);
                assert_eq!(stats.states_found, 10);
            }
            other => panic!("Expected LimitReached(States), got: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_with_constants() {
        // Test that parallel checker correctly binds constants from config
        let src = r#"
---- MODULE WithConstants ----
CONSTANT N
VARIABLE x
Init == x = N
Next == x < N + 3 /\ x' = x + 1
Bounded == x < N + 5
====
"#;
        let module = parse_module(src);

        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Bounded".to_string()],
            ..Default::default()
        };
        config.constants.insert(
            "N".to_string(),
            crate::config::ConstantValue::Value("5".to_string()),
        );

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // N=5, starts at x=5, Next enabled while x < 8
                // States: x=5, x=6, x=7, x=8 (then x < 8 is false, deadlock disabled)
                assert_eq!(stats.states_found, 4);
                assert_eq!(stats.initial_states, 1);
            }
            other => panic!("Expected success with constants, got: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_with_model_value_set() {
        // Test that parallel checker correctly binds model value set constants
        let src = r#"
---- MODULE WithModelValues ----
CONSTANT Procs
VARIABLE current
Init == current \in Procs
Next == current' \in Procs
AlwaysProc == current \in Procs
====
"#;
        let module = parse_module(src);

        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["AlwaysProc".to_string()],
            ..Default::default()
        };
        config.constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
        );

        let mut checker = ParallelChecker::new(&module, &config, 2);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // Should find 2 states: current=p1, current=p2
                assert_eq!(stats.states_found, 2);
                assert_eq!(stats.initial_states, 2);
            }
            other => panic!("Expected success with model values, got: {:?}", other),
        }
    }

    // ========== Capacity status helper function tests ==========

    #[test]
    fn test_capacity_status_to_u8() {
        assert_eq!(
            capacity_status_to_u8(&CapacityStatus::Normal),
            CAPACITY_NORMAL
        );
        assert_eq!(
            capacity_status_to_u8(&CapacityStatus::Warning {
                count: 100,
                capacity: 150,
                usage: 0.66,
            }),
            CAPACITY_WARNING
        );
        assert_eq!(
            capacity_status_to_u8(&CapacityStatus::Critical {
                count: 100,
                capacity: 120,
                usage: 0.83,
            }),
            CAPACITY_CRITICAL
        );
    }

    #[test]
    fn test_check_and_warn_capacity_no_change() {
        // When status doesn't change, no warning should be emitted
        use dashmap::DashSet;

        let set: Arc<dyn FingerprintSet> = Arc::new(DashSet::<Fingerprint>::new());
        let last_status = AtomicU8::new(CAPACITY_NORMAL);

        // DashSet always returns Normal status
        check_and_warn_capacity(set.as_ref(), &last_status);

        // Status should remain Normal
        assert_eq!(last_status.load(Ordering::Relaxed), CAPACITY_NORMAL);
    }

    #[test]
    fn test_parallel_mmap_fingerprint_storage_integration() {
        // Test that parallel checker can use mmap storage and reports errors on overflow
        use crate::storage::MmapFingerprintSet;

        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        // Create mmap storage with very small capacity to trigger overflow
        let mmap_set = MmapFingerprintSet::new(10, None).expect("Failed to create mmap storage");
        let storage: Arc<dyn FingerprintSet> = Arc::new(mmap_set);

        let mut checker = ParallelChecker::new(&module, &config, 1);
        checker.set_deadlock_check(false);
        checker.set_store_states(false);
        checker.set_fingerprint_storage(storage);
        checker.set_max_states(50); // Would exceed capacity of 10

        let result = checker.check();

        // Should fail due to fingerprint storage overflow
        match result {
            CheckResult::Error {
                error: CheckError::FingerprintStorageOverflow { dropped },
                ..
            } => {
                assert!(dropped > 0, "Should have dropped some fingerprints");
            }
            CheckResult::LimitReached { .. } => {
                // Might hit limit first - also acceptable
            }
            other => panic!("Expected overflow error or limit, got: {:?}", other),
        }
    }

    // ==========================================
    // Stress tests for high concurrency (#42)
    // ==========================================

    #[test]
    fn test_parallel_stress_many_workers() {
        // Stress test with 8, 12, and 16 workers on a moderate state space
        // Verifies that more workers still produce correct results
        let src = r#"
---- MODULE StressWorkers ----
VARIABLE x, y, z

Init == x \in 0..2 /\ y \in 0..2 /\ z \in 0..2
Next == /\ x' \in 0..2
        /\ y' \in 0..2
        /\ z' \in 0..2
        /\ x' + y' + z' <= 6

Valid == x + y + z <= 6
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Valid".to_string()],
            ..Default::default()
        };

        // Get baseline from sequential checker
        let mut seq_checker = crate::check::ModelChecker::new(&module, &config);
        seq_checker.set_deadlock_check(false);
        let seq_result = seq_checker.check();

        let expected_states = match seq_result {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Sequential check failed: {:?}", other),
        };

        // Test with increasing worker counts
        for workers in [8, 12, 16] {
            let mut checker = ParallelChecker::new(&module, &config, workers);
            checker.set_deadlock_check(false);
            let result = checker.check();

            match result {
                CheckResult::Success(stats) => {
                    assert_eq!(
                        stats.states_found, expected_states,
                        "Worker count {} produced wrong state count: {} vs expected {}",
                        workers, stats.states_found, expected_states
                    );
                }
                other => panic!("Parallel check with {} workers failed: {:?}", workers, other),
            }
        }
    }

    #[test]
    fn test_parallel_stress_high_contention() {
        // High-contention scenario: many initial states, all workers
        // competing for work simultaneously
        //
        // This creates 4*4*4*4 = 256 initial states, causing high contention
        // as workers race to claim and process them.
        let src = r#"
---- MODULE HighContention ----
VARIABLE a, b, c, d

Init == a \in 1..4 /\ b \in 1..4 /\ c \in 1..4 /\ d \in 1..4
Next == /\ a' = (a % 4) + 1
        /\ b' = (b % 4) + 1
        /\ c' = (c % 4) + 1
        /\ d' = (d % 4) + 1

TypeOK == /\ a \in 1..4
          /\ b \in 1..4
          /\ c \in 1..4
          /\ d \in 1..4
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        // Get baseline from sequential checker
        let mut seq_checker = crate::check::ModelChecker::new(&module, &config);
        seq_checker.set_deadlock_check(false);
        let seq_result = seq_checker.check();

        let (expected_states, expected_initial) = match seq_result {
            CheckResult::Success(stats) => (stats.states_found, stats.initial_states),
            other => panic!("Sequential check failed: {:?}", other),
        };

        // Verify we have many initial states (high contention)
        assert_eq!(expected_initial, 256, "Should have 256 initial states");

        // Run with 8 workers under high contention
        let mut checker = ParallelChecker::new(&module, &config, 8);
        checker.set_deadlock_check(false);
        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(
                    stats.initial_states, expected_initial,
                    "Initial state count mismatch under contention"
                );
                assert_eq!(
                    stats.states_found, expected_states,
                    "State count mismatch under high contention"
                );
            }
            other => panic!("High contention test failed: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_determinism_across_worker_counts() {
        // Determinism verification: same spec should produce identical
        // results regardless of worker count
        //
        // This tests a spec with non-trivial state space to verify
        // determinism across 1, 2, 4, 8, and 12 workers.
        let src = r#"
---- MODULE Determinism ----
EXTENDS Naturals

VARIABLE pc, x, y

Init == pc = "start" /\ x = 0 /\ y = 0

Next == \/ /\ pc = "start"
           /\ pc' = "inc_x"
           /\ x' = x
           /\ y' = y
        \/ /\ pc = "inc_x"
           /\ x < 3
           /\ x' = x + 1
           /\ pc' = "inc_x"
           /\ y' = y
        \/ /\ pc = "inc_x"
           /\ x >= 3
           /\ pc' = "inc_y"
           /\ x' = x
           /\ y' = y
        \/ /\ pc = "inc_y"
           /\ y < 3
           /\ y' = y + 1
           /\ pc' = "inc_y"
           /\ x' = x
        \/ /\ pc = "inc_y"
           /\ y >= 3
           /\ pc' = "done"
           /\ x' = x
           /\ y' = y
        \/ /\ pc = "done"
           /\ UNCHANGED <<pc, x, y>>

TypeOK == /\ pc \in {"start", "inc_x", "inc_y", "done"}
          /\ x \in 0..3
          /\ y \in 0..3
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        // Collect results from different worker counts
        let worker_counts = [1, 2, 4, 8, 12];
        let mut results: Vec<(usize, usize, usize, usize)> = Vec::new();

        for workers in worker_counts {
            let mut checker = ParallelChecker::new(&module, &config, workers);
            checker.set_deadlock_check(false);
            let result = checker.check();

            match result {
                CheckResult::Success(stats) => {
                    results.push((
                        workers,
                        stats.states_found,
                        stats.initial_states,
                        stats.transitions,
                    ));
                }
                other => panic!(
                    "Determinism test failed with {} workers: {:?}",
                    workers, other
                ),
            }
        }

        // Verify all results match
        let (_, first_states, first_initial, first_transitions) = results[0];
        for (workers, states, initial, transitions) in &results[1..] {
            assert_eq!(
                *states, first_states,
                "State count differs between 1 worker and {} workers: {} vs {}",
                workers, first_states, states
            );
            assert_eq!(
                *initial, first_initial,
                "Initial state count differs between 1 worker and {} workers",
                workers
            );
            assert_eq!(
                *transitions, first_transitions,
                "Transition count differs between 1 worker and {} workers",
                workers
            );
        }
    }

    #[test]
    fn test_parallel_stress_large_fanout() {
        // Large fanout: each state has many successors, creating work
        // stealing opportunities
        //
        // From each state, there are up to 3 successors, reduced from original to speed up test
        let src = r#"
---- MODULE LargeFanout ----
VARIABLE a, b

Init == a = 0 /\ b = 0

Next == /\ a' \in {a, (a + 1) % 3}
        /\ b' \in {b, (b + 1) % 3}
        /\ (a' /= a \/ b' /= b)

TypeOK == a \in 0..2 /\ b \in 0..2
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        // Get baseline from sequential checker
        let mut seq_checker = crate::check::ModelChecker::new(&module, &config);
        seq_checker.set_deadlock_check(false);
        let seq_result = seq_checker.check();

        let expected_states = match seq_result {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Sequential check failed: {:?}", other),
        };

        // Run with 8 workers to exercise work stealing
        let mut checker = ParallelChecker::new(&module, &config, 8);
        checker.set_deadlock_check(false);
        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(
                    stats.states_found, expected_states,
                    "Large fanout test produced wrong state count"
                );
            }
            other => panic!("Large fanout test failed: {:?}", other),
        }
    }

    #[test]
    fn test_parallel_stress_repeated_runs() {
        // Run the same spec multiple times to check for race conditions
        // that might cause non-deterministic failures
        let src = r#"
---- MODULE RepeatedRuns ----
VARIABLE x

Init == x \in 1..5
Next == x' = (x % 5) + 1

TypeOK == x \in 1..5
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        // Run 10 times and verify consistent results
        let mut state_counts: Vec<usize> = Vec::new();
        for _ in 0..10 {
            let mut checker = ParallelChecker::new(&module, &config, 8);
            checker.set_deadlock_check(false);
            let result = checker.check();

            match result {
                CheckResult::Success(stats) => {
                    state_counts.push(stats.states_found);
                }
                other => panic!("Repeated run failed: {:?}", other),
            }
        }

        // All runs should produce the same state count
        let first = state_counts[0];
        for (i, count) in state_counts.iter().enumerate() {
            assert_eq!(
                *count, first,
                "Run {} produced {} states, expected {} (non-deterministic behavior detected)",
                i, count, first
            );
        }
    }
}
