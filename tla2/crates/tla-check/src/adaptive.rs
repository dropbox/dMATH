//! Adaptive Model Checker - Automatically selects sequential or parallel execution
//!
//! This module implements an adaptive approach to model checking that:
//! 1. Runs a pilot phase to sample the spec's characteristics
//! 2. Estimates the total state space size based on branching factor
//! 3. Automatically selects sequential or parallel execution based on heuristics
//!
//! # Heuristics
//!
//! Based on benchmarking (see reports/main/benchmark-report-2025-12-26-21-33.md):
//! - Sequential is 3-4x faster for specs with <1000 states (no sync overhead)
//! - Parallel with 2-4 workers provides 2-3x speedup for specs with >1000 states
//! - Diminishing returns beyond 4 workers for most specs
//!
//! The adaptive checker uses initial state count and sampled branching factor
//! to estimate spec size and select the optimal strategy.

use crate::check::{CheckResult, CheckStats, ModelChecker, ProgressCallback};
use crate::config::Config;
use crate::constants::bind_constants_from_config;
use crate::enumerate::{
    debug_enum, enumerate_states_from_constraint_branches, enumerate_successors,
    extract_init_constraints, find_unconstrained_vars,
};
use crate::eval::{eval, EvalCtx};
use crate::parallel::ParallelChecker;
use crate::spec_formula::FairnessConstraint;
use crate::state::State;
use crate::storage::FingerprintSet;
use crate::value::Value;
use crate::CheckError;
use std::collections::HashMap;
use std::sync::Arc;
use tla_core::ast::{Module, OperatorDef, Unit};

/// Thresholds for adaptive parallelism decisions
const PARALLEL_THRESHOLD: usize = 20_000; // Use parallel if estimated states > this
const MEDIUM_SPEC_THRESHOLD: usize = 200_000; // Use 2 workers if estimated < this
const LARGE_SPEC_THRESHOLD: usize = 1_000_000; // Use 4 workers if estimated < this
const PILOT_SAMPLE_SIZE: usize = 50; // Number of states to sample in pilot phase

/// Strategy selected by adaptive checker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Use sequential checker (no parallelism overhead)
    Sequential,
    /// Use parallel checker with specified worker count
    Parallel(usize),
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::Sequential => write!(f, "sequential"),
            Strategy::Parallel(n) => write!(f, "parallel ({} workers)", n),
        }
    }
}

/// Result of pilot phase analysis
#[derive(Debug, Clone)]
pub struct PilotAnalysis {
    /// Number of initial states
    pub initial_states: usize,
    /// Average branching factor (successors per state)
    pub avg_branching_factor: f64,
    /// Estimated total state space size
    pub estimated_states: usize,
    /// Selected execution strategy
    pub strategy: Strategy,
    /// Number of states sampled during pilot
    pub states_sampled: usize,
}

/// Adaptive model checker
pub struct AdaptiveChecker {
    /// The TLA+ module being checked
    module: Arc<Module>,
    /// Extended/instanced modules (kept separately for instance ops lookup)
    extended_modules: Vec<Arc<Module>>,
    /// Model checking configuration
    config: Config,
    /// Variable names
    vars: Vec<Arc<str>>,
    /// Operator definitions
    op_defs: HashMap<String, OperatorDef>,
    /// Whether to check for deadlocks
    check_deadlock: bool,
    /// Maximum states limit
    max_states: Option<usize>,
    /// Maximum depth limit
    max_depth: Option<usize>,
    /// Progress callback
    progress_callback: Option<ProgressCallback>,
    /// Number of available CPU cores
    available_cores: usize,
    /// Fairness constraints from SPECIFICATION formula (for liveness checking)
    fairness: Vec<FairnessConstraint>,
    /// Whether to collect per-action coverage statistics (forces sequential strategy)
    collect_coverage: bool,
    /// Whether to store full states for trace reconstruction
    store_full_states: bool,
    /// Whether to auto-create temp trace file (for ModelChecker in fingerprint-only mode)
    auto_create_trace_file: bool,
    /// Optional fingerprint storage for no-trace mode (memory-mapped for large state spaces)
    fingerprint_storage: Option<Arc<dyn FingerprintSet>>,
}

impl AdaptiveChecker {
    /// Create a new adaptive model checker
    pub fn new(module: &Module, config: &Config) -> Self {
        Self::new_with_extends(module, &[], config)
    }

    /// Create a new adaptive model checker with extended modules
    ///
    /// The `extended_modules` should be modules that `module` extends (via EXTENDS).
    /// Their operator definitions will be loaded first, then the main module's
    /// definitions (which may override them).
    pub fn new_with_extends(
        module: &Module,
        extended_modules: &[&Module],
        config: &Config,
    ) -> Self {
        // Extract variable names and operator definitions from all modules
        let mut vars: Vec<Arc<str>> = Vec::new();
        let mut op_defs: HashMap<String, OperatorDef> = HashMap::new();

        // First, determine which modules are INSTANCE'd (vs EXTENDS'd)
        // We should NOT collect variables from INSTANCE'd modules - their variables
        // are mapped to the main module's symbols through implicit substitution.
        // IMPORTANT: We must also scan extended modules for their INSTANCE declarations,
        // because if Main EXTENDS Voting and Voting has `C == INSTANCE Consensus`,
        // then Consensus should NOT contribute operators to Main's namespace.
        use tla_core::ast::Expr;
        let mut instance_module_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Helper to scan a module for NAMED INSTANCE declarations
        // NOTE: Standalone INSTANCE (Unit::Instance) imports operators UNQUALIFIED,
        // so we should NOT skip those modules. Only skip modules from named instances
        // like `M == INSTANCE Module` which are accessed via prefix (M!Op).
        fn collect_instance_modules(
            module: &Module,
            instance_module_names: &mut std::collections::HashSet<String>,
        ) {
            for unit in &module.units {
                // Named instances ONLY: TD == INSTANCE SyncTerminationDetection
                // These modules should only be accessible via instance prefix (TD!Op)
                if let Unit::Operator(def) = &unit.node {
                    if let Expr::InstanceExpr(module_name, _subs) = &def.body.node {
                        instance_module_names.insert(module_name.clone());
                    }
                }
                // NOTE: Standalone INSTANCE (Unit::Instance) is NOT added here
                // because it imports operators unqualified into the current namespace
            }
        }

        // Scan main module for INSTANCE declarations
        collect_instance_modules(module, &mut instance_module_names);

        // Also scan extended modules for INSTANCE declarations
        // This ensures that if Main EXTENDS Voting and Voting has `C == INSTANCE Consensus`,
        // Consensus is properly excluded from the main namespace
        for ext_mod in extended_modules {
            collect_instance_modules(ext_mod, &mut instance_module_names);
        }

        // First from extended modules
        // Skip INSTANCE'd modules - they should only contribute via instance prefix (e.g., C!Init)
        for ext_mod in extended_modules {
            let is_instanced = instance_module_names.contains(&ext_mod.name.node);
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
                        if debug_enum() {
                            eprintln!(
                                "Loaded operator (ext): {} with body span {:?}",
                                def.name.node, def.body.span
                            );
                        }
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
                    if debug_enum() {
                        eprintln!(
                            "Loaded operator: {} with body span {:?}",
                            def.name.node, def.body.span
                        );
                    }
                    op_defs.insert(def.name.node.clone(), def.clone());
                }
                _ => {}
            }
        }

        let available_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Store extended modules along with main module for later use
        let mut combined_module = module.clone();
        // Merge definitions from extended modules into the main module
        // This ensures run_pilot and other methods see all definitions
        for ext_mod in extended_modules {
            let is_instanced = instance_module_names.contains(&ext_mod.name.node);
            // Skip INSTANCE'd modules entirely - they should only contribute
            // operators via instance prefix (e.g., C!Init), not to the main namespace
            if is_instanced {
                continue;
            }
            for unit in &ext_mod.units {
                match &unit.node {
                    Unit::Operator(def) => {
                        // Add operator if not already present
                        if !combined_module.units.iter().any(|u| {
                            matches!(&u.node, Unit::Operator(d) if d.name.node == def.name.node)
                        }) {
                            combined_module.units.push(tla_core::Spanned {
                                node: Unit::Operator(def.clone()),
                                span: unit.span,
                            });
                        }
                    }
                    Unit::Variable(var_names) => {
                        // Add variables if not already present
                        let new_vars: Vec<_> = var_names
                            .iter()
                            .filter(|v| {
                                !combined_module.units.iter().any(|u| {
                                    matches!(&u.node, Unit::Variable(vs) if vs.iter().any(|vn| vn.node == v.node))
                                })
                            })
                            .cloned()
                            .collect();
                        if !new_vars.is_empty() {
                            combined_module.units.push(tla_core::Spanned {
                                node: Unit::Variable(new_vars),
                                span: unit.span,
                            });
                        }
                    }
                    Unit::Constant(consts) => {
                        // Add constants if not already present
                        let new_consts: Vec<_> = consts
                            .iter()
                            .filter(|c| {
                                !combined_module.units.iter().any(|u| {
                                    matches!(&u.node, Unit::Constant(cs) if cs.iter().any(|cn| cn.name.node == c.name.node))
                                })
                            })
                            .cloned()
                            .collect();
                        if !new_consts.is_empty() {
                            combined_module.units.push(tla_core::Spanned {
                                node: Unit::Constant(new_consts),
                                span: unit.span,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        // Store extended modules for instance ops lookup
        let extended_modules: Vec<Arc<Module>> = extended_modules
            .iter()
            .map(|m| Arc::new((*m).clone()))
            .collect();

        AdaptiveChecker {
            module: Arc::new(combined_module),
            extended_modules,
            config: config.clone(),
            vars,
            op_defs,
            check_deadlock: config.check_deadlock,
            max_states: None,
            max_depth: None,
            progress_callback: None,
            available_cores,
            fairness: Vec::new(),
            collect_coverage: false,
            store_full_states: false, // Default: fingerprint-only for 42x memory reduction (#88)
            auto_create_trace_file: true, // Auto-create temp trace file for reconstruction
            fingerprint_storage: None,
        }
    }

    /// Enable or disable deadlock checking
    pub fn set_deadlock_check(&mut self, check: bool) {
        self.check_deadlock = check;
    }

    /// Enable or disable storing full states for trace reconstruction.
    ///
    /// When `store` is false (no-trace mode), counterexample traces will be unavailable.
    /// This can significantly reduce memory usage for large state spaces.
    pub fn set_store_states(&mut self, store: bool) {
        self.store_full_states = store;
    }

    /// Set whether to auto-create a temp trace file for fingerprint-only mode.
    ///
    /// When true (default): Creates a temporary trace file automatically if
    /// `store_full_states` is false and no explicit trace file is set.
    ///
    /// When false (--no-trace mode): No trace file is created, traces are
    /// completely unavailable for maximum memory efficiency.
    pub fn set_auto_create_trace_file(&mut self, auto_create: bool) {
        self.auto_create_trace_file = auto_create;
    }

    /// Set the fingerprint storage backend.
    ///
    /// This allows using memory-mapped storage for large state spaces that
    /// exceed available RAM. Must be called before `check()`.
    ///
    /// Only used when `store_full_states` is false (no-trace mode).
    /// When `store_full_states` is true, full states are stored regardless of this setting.
    pub fn set_fingerprint_storage(&mut self, storage: Arc<dyn FingerprintSet>) {
        self.fingerprint_storage = Some(storage);
    }

    /// Set fairness constraints from SPECIFICATION formula
    ///
    /// These constraints are passed to the sequential ModelChecker for liveness checking.
    pub fn set_fairness(&mut self, fairness: Vec<FairnessConstraint>) {
        self.fairness = fairness;
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

    /// Enable or disable per-action coverage statistics collection.
    ///
    /// Coverage collection is implemented in the sequential `ModelChecker`, so enabling this
    /// forces the adaptive strategy to choose `Strategy::Sequential`.
    pub fn set_collect_coverage(&mut self, collect: bool) {
        self.collect_coverage = collect;
    }

    /// Set maximum number of states to explore
    pub fn set_max_states(&mut self, limit: usize) {
        self.max_states = Some(limit);
    }

    /// Set maximum BFS depth to explore
    pub fn set_max_depth(&mut self, limit: usize) {
        self.max_depth = Some(limit);
    }

    /// Set a progress callback
    pub fn set_progress_callback(&mut self, callback: ProgressCallback) {
        self.progress_callback = Some(callback);
    }

    /// Run pilot phase to analyze spec characteristics
    fn run_pilot(&mut self) -> Result<PilotAnalysis, CheckError> {
        // Toolbox-generated "constant-expression evaluation" models may contain only
        // ASSUME statements (sometimes with Print/PrintT side effects), but provide no
        // INIT/NEXT or SPECIFICATION and declare no state variables. Treat these as
        // successful assume-only checks and force sequential execution.
        if self.config.init.is_none()
            && self.config.next.is_none()
            && self.config.specification.is_none()
            && self.vars.is_empty()
            && self.config.invariants.is_empty()
            && self.config.properties.is_empty()
            && self
                .module
                .units
                .iter()
                .any(|u| matches!(u.node, Unit::Assume(_)))
        {
            return Ok(PilotAnalysis {
                initial_states: 0,
                avg_branching_factor: 0.0,
                estimated_states: 0,
                strategy: Strategy::Sequential,
                states_sampled: 0,
            });
        }

        let init_name = self.config.init.clone().ok_or(CheckError::MissingInit)?;
        let next_name = self.config.next.clone().ok_or(CheckError::MissingNext)?;

        if self.vars.is_empty() {
            return Err(CheckError::NoVariables);
        }

        // Create evaluation context
        let mut ctx = EvalCtx::new();
        ctx.load_module(&self.module);

        // For named instances, load the instanced module's operators into instance_ops
        let instance_module_names: Vec<String> = ctx
            .instances()
            .values()
            .map(|info| info.module_name.clone())
            .collect();
        for module_name in &instance_module_names {
            // Find the module in extended_modules by name
            for ext_mod in &self.extended_modules {
                if ext_mod.name.node == *module_name {
                    ctx.load_instance_module(module_name.clone(), ext_mod);
                    break;
                }
            }
        }

        // Variables from INSTANCE'd modules are skipped during var collection, so no extra
        // filtering is needed here. Filtering by substitution "from" names is unsafe because
        // it can remove real state variables that happen to share a name with an INSTANCE formal.
        let active_vars: Vec<Arc<str>> = self.vars.clone();

        // Register variables in sorted order for consistent fingerprinting
        // VarRegistry stores names in registration order, which must match OrdMap iteration
        let mut sorted_vars = active_vars.to_vec();
        sorted_vars.sort();
        ctx.register_vars(sorted_vars.iter().cloned());

        // Bind constants from config
        bind_constants_from_config(&mut ctx, &self.config).map_err(CheckError::EvalError)?;

        // Generate initial states using filtered vars list
        let initial_states = self.generate_initial_states(&mut ctx, &init_name, &active_vars)?;
        let num_initial = initial_states.len();

        if num_initial == 0 {
            return Err(CheckError::InitCannotEnumerate(
                "Init predicate has no solutions".to_string(),
            ));
        }

        // Sample branching factor by exploring a few states
        let mut total_successors = 0usize;
        let mut states_sampled = 0usize;

        let next_def = self
            .op_defs
            .get(&next_name)
            .ok_or(CheckError::MissingNext)?;

        for state in initial_states.iter().take(PILOT_SAMPLE_SIZE) {
            // Bind current state
            let saved = ctx.save_scope();
            for (name, value) in state.vars() {
                ctx.bind_mut(name.to_string(), value.clone());
            }

            // Generate successors
            match enumerate_successors(&mut ctx, next_def, state, &self.vars) {
                Ok(successors) => {
                    total_successors += successors.len();
                    states_sampled += 1;
                }
                Err(_) => {
                    // Skip states that error during enumeration
                }
            }

            ctx.restore_scope(saved);

            if states_sampled >= PILOT_SAMPLE_SIZE {
                break;
            }
        }

        // Calculate average branching factor
        let avg_branching_factor = if states_sampled > 0 {
            total_successors as f64 / states_sampled as f64
        } else {
            1.0
        };

        // Estimate total states using geometric series approximation
        // For a BFS with branching factor b, after d levels: 1 + b + b^2 + ... + b^d
        // We estimate based on initial states and branching factor
        let estimated_states = estimate_state_space(num_initial, avg_branching_factor);

        // Select strategy based on estimate
        let strategy = select_strategy(estimated_states, self.available_cores);

        Ok(PilotAnalysis {
            initial_states: num_initial,
            avg_branching_factor,
            estimated_states,
            strategy,
            states_sampled,
        })
    }

    /// Generate initial states
    ///
    /// First attempts direct constraint extraction from the Init predicate.
    /// If that fails (unsupported expressions or missing per-variable constraints),
    /// falls back to enumerating states from a type constraint (usually TypeOK)
    /// and filtering by evaluating the full Init predicate.
    fn generate_initial_states(
        &mut self,
        ctx: &mut EvalCtx,
        init_name: &str,
        vars: &[Arc<str>],
    ) -> Result<Vec<State>, CheckError> {
        let def = self.op_defs.get(init_name).ok_or(CheckError::MissingInit)?;
        let init_body = def.body.clone();

        // Try to extract constraints directly from the Init predicate
        let direct_hint = if let Some(branches) = extract_init_constraints(ctx, &init_body, vars) {
            let unconstrained = find_unconstrained_vars(vars, &branches);
            if unconstrained.is_empty() {
                return match enumerate_states_from_constraint_branches(Some(ctx), vars, &branches) {
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

            let Some(branches) = extract_init_constraints(ctx, &cand_body, vars) else {
                continue;
            };

            let unconstrained = find_unconstrained_vars(vars, &branches);
            if !unconstrained.is_empty() {
                continue;
            }

            let Some(base_states) =
                enumerate_states_from_constraint_branches(Some(ctx), vars, &branches)
            else {
                continue;
            };

            // Filter base states by the original Init predicate
            let mut filtered: Vec<State> = Vec::new();
            for state in base_states {
                let saved = ctx.save_scope();
                for (name, value) in state.vars() {
                    ctx.bind_mut(Arc::clone(name), value.clone());
                }
                let keep = match eval(ctx, &init_body) {
                    Ok(Value::Bool(b)) => b,
                    Ok(_) => {
                        ctx.restore_scope(saved);
                        return Err(CheckError::InitNotBoolean);
                    }
                    Err(e) => {
                        ctx.restore_scope(saved);
                        return Err(CheckError::EvalError(e));
                    }
                };
                ctx.restore_scope(saved);

                if keep {
                    filtered.push(state);
                }
            }

            return Ok(filtered);
        }

        Err(CheckError::InitCannotEnumerate(direct_hint))
    }

    /// Run model checking with adaptive strategy selection
    pub fn check(&mut self) -> (CheckResult, Option<PilotAnalysis>) {
        // Run pilot phase
        let mut analysis = match self.run_pilot() {
            Ok(a) => a,
            Err(e) => {
                return (
                    CheckResult::Error {
                        error: e,
                        stats: CheckStats::default(),
                    },
                    None,
                )
            }
        };

        // Force sequential mode when liveness properties are defined
        // (ParallelChecker doesn't support liveness checking)
        let has_liveness = !self.config.properties.is_empty();
        if has_liveness && analysis.strategy != Strategy::Sequential {
            analysis.strategy = Strategy::Sequential;
        }

        // Coverage collection is implemented in ModelChecker (sequential) only.
        if self.collect_coverage && analysis.strategy != Strategy::Sequential {
            analysis.strategy = Strategy::Sequential;
        }

        // Execute with selected strategy
        // Convert extended_modules Arc<Module> to &Module for new_with_extends
        let ext_refs: Vec<&Module> = self.extended_modules.iter().map(|m| m.as_ref()).collect();
        let result = match analysis.strategy {
            Strategy::Sequential => {
                let mut checker =
                    ModelChecker::new_with_extends(&self.module, &ext_refs, &self.config);
                checker.set_deadlock_check(self.check_deadlock);
                checker.set_collect_coverage(self.collect_coverage);
                checker.set_store_states(self.store_full_states);
                checker.set_auto_create_trace_file(self.auto_create_trace_file);
                // Pass fingerprint storage for no-trace mode
                if let Some(ref storage) = self.fingerprint_storage {
                    checker.set_fingerprint_storage(Arc::clone(storage));
                }
                // Pass fairness constraints for liveness checking
                if !self.fairness.is_empty() {
                    checker.set_fairness(self.fairness.clone());
                }
                if let Some(limit) = self.max_states {
                    checker.set_max_states(limit);
                }
                if let Some(limit) = self.max_depth {
                    checker.set_max_depth(limit);
                }
                if let Some(callback) = self.progress_callback.take() {
                    checker.set_progress_callback(callback);
                }
                checker.check()
            }
            Strategy::Parallel(workers) => {
                let mut checker = ParallelChecker::new_with_extends(
                    &self.module,
                    &ext_refs,
                    &self.config,
                    workers,
                );
                checker.set_deadlock_check(self.check_deadlock);
                checker.set_store_states(self.store_full_states);
                checker.set_auto_create_trace_file(self.auto_create_trace_file);
                // Pass fingerprint storage for no-trace mode
                if let Some(ref storage) = self.fingerprint_storage {
                    checker.set_fingerprint_storage(Arc::clone(storage));
                }
                if let Some(limit) = self.max_states {
                    checker.set_max_states(limit);
                }
                if let Some(limit) = self.max_depth {
                    checker.set_max_depth(limit);
                }
                if let Some(callback) = self.progress_callback.take() {
                    checker.set_progress_callback(callback);
                }
                checker.check()
            }
        };

        (result, Some(analysis))
    }
}

/// Estimate total state space size based on initial states and branching factor
fn estimate_state_space(initial_states: usize, branching_factor: f64) -> usize {
    // Simple heuristic: if branching factor is high, expect larger state space
    // We use a rough estimate based on typical spec patterns:
    // - b < 1.5: likely bounded, estimate = initial * 10
    // - b < 3.0: moderate growth, estimate = initial * 100
    // - b < 10.0: high growth, estimate = initial * 1000
    // - b >= 10.0: very high branching, estimate = initial * 200000
    //
    // Rationale: Some real-world specs (e.g., bosco) have extremely high branching
    // factors but also deep state spaces; a 1000x multiplier badly underestimates
    // the explored states and causes auto mode to incorrectly choose sequential.

    let multiplier = if branching_factor < 1.5 {
        10.0
    } else if branching_factor < 3.0 {
        100.0
    } else if branching_factor < 10.0 {
        1000.0
    } else {
        200000.0
    };

    (initial_states as f64 * multiplier) as usize
}

/// Select execution strategy based on estimated state space
fn select_strategy(estimated_states: usize, available_cores: usize) -> Strategy {
    if estimated_states < PARALLEL_THRESHOLD {
        Strategy::Sequential
    } else if estimated_states < MEDIUM_SPEC_THRESHOLD {
        Strategy::Parallel(2.min(available_cores))
    } else if estimated_states < LARGE_SPEC_THRESHOLD {
        Strategy::Parallel(4.min(available_cores))
    } else {
        Strategy::Parallel(available_cores)
    }
}

/// Run adaptive model checking on a module
///
/// This is the recommended entry point for model checking. It automatically
/// selects the best execution strategy (sequential vs parallel) based on
/// the spec's characteristics.
///
/// # Returns
/// A tuple of (CheckResult, `Option<PilotAnalysis>`) where the analysis
/// contains details about the selected strategy.
pub fn check_module_adaptive(
    module: &Module,
    config: &Config,
) -> (CheckResult, Option<PilotAnalysis>) {
    let mut checker = AdaptiveChecker::new(module, config);
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
    fn test_adaptive_small_spec_uses_sequential() {
        // Small spec should use sequential
        let src = r#"
---- MODULE SmallCounter ----
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

        let mut checker = AdaptiveChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let (result, analysis) = checker.check();

        assert!(matches!(result, CheckResult::Success(_)));

        let analysis = analysis.unwrap();
        assert_eq!(analysis.initial_states, 1);
        // Small spec should choose sequential
        assert!(
            matches!(analysis.strategy, Strategy::Sequential),
            "Expected Sequential for small spec, got {:?}",
            analysis.strategy
        );
    }

    #[test]
    fn test_adaptive_larger_spec_uses_parallel() {
        // Spec with many initial states should use parallel
        let src = r#"
---- MODULE LargerSpec ----
VARIABLE x, y, z
Init == x \in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
     /\ y \in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
     /\ z \in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
Next == x' \in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
     /\ y' \in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
     /\ z' \in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
Valid == TRUE
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Valid".to_string()],
            ..Default::default()
        };

        let mut checker = AdaptiveChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_max_states(100); // Limit to avoid long test

        let (_result, analysis) = checker.check();

        let analysis = analysis.unwrap();
        // 10 * 10 * 10 = 1000 initial states
        assert_eq!(analysis.initial_states, 1000);
        // Large initial state count with high branching should use parallel
        assert!(
            matches!(analysis.strategy, Strategy::Parallel(_)),
            "Expected Parallel for large spec, got {:?}",
            analysis.strategy
        );
    }

    #[test]
    fn test_adaptive_no_trace_propagates_to_parallel_checker() {
        // Ensure store_full_states=false is passed through when strategy selects Parallel.
        let src = r#"
---- MODULE NoTraceParallel ----
VARIABLE x
Init == x \in 0 .. 999
Next == x' \in 0 .. 1000
Safe == x < 1000
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safe".to_string()],
            ..Default::default()
        };

        let mut checker = AdaptiveChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_store_states(false);
        checker.set_max_depth(1);

        let (result, analysis) = checker.check();

        let analysis = analysis.unwrap();
        assert!(
            matches!(analysis.strategy, Strategy::Parallel(_)),
            "Expected Parallel for no-trace propagation test, got {:?}",
            analysis.strategy
        );

        match result {
            CheckResult::InvariantViolation { trace, .. } => {
                assert!(trace.is_empty(), "Trace should be empty in no-trace mode");
            }
            other => panic!("Expected invariant violation, got: {:?}", other),
        }
    }

    #[test]
    fn test_pilot_analysis_branching_factor() {
        // Test branching factor calculation
        let src = r#"
---- MODULE Branching ----
VARIABLE x
Init == x = 0
Next == x' \in {x + 1, x + 2, x + 3}
Valid == TRUE
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Valid".to_string()],
            ..Default::default()
        };

        let mut checker = AdaptiveChecker::new(&module, &config);
        let analysis = checker.run_pilot().unwrap();

        // Each state has 3 successors
        assert!(
            analysis.avg_branching_factor >= 2.5,
            "Expected branching factor >= 2.5, got {}",
            analysis.avg_branching_factor
        );
    }

    #[test]
    fn test_estimate_state_space() {
        // Low branching factor
        let estimate = estimate_state_space(10, 1.0);
        assert_eq!(estimate, 100); // 10 * 10

        // Medium branching factor
        let estimate = estimate_state_space(10, 2.0);
        assert_eq!(estimate, 1000); // 10 * 100

        // High branching factor
        let estimate = estimate_state_space(10, 5.0);
        assert_eq!(estimate, 10000); // 10 * 1000
    }

    #[test]
    fn test_select_strategy() {
        // Small specs get sequential
        assert_eq!(select_strategy(500, 8), Strategy::Sequential);

        // Medium specs get 2 workers
        assert_eq!(select_strategy(30000, 8), Strategy::Parallel(2));

        // Large specs get 4 workers
        assert_eq!(select_strategy(300000, 8), Strategy::Parallel(4));

        // Very large specs get all cores
        assert_eq!(select_strategy(1500000, 8), Strategy::Parallel(8));

        // Respect available cores limit
        assert_eq!(select_strategy(1500000, 2), Strategy::Parallel(2));
    }

    #[test]
    fn test_adaptive_consistency_with_direct_checkers() {
        // Adaptive should produce same results as direct checkers
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

        // Run sequential
        let mut seq_checker = ModelChecker::new(&module, &config);
        seq_checker.set_deadlock_check(false);
        let seq_result = seq_checker.check();

        // Run adaptive
        let mut adaptive_checker = AdaptiveChecker::new(&module, &config);
        adaptive_checker.set_deadlock_check(false);
        let (adaptive_result, _) = adaptive_checker.check();

        // Both should succeed with same state count
        match (seq_result, adaptive_result) {
            (CheckResult::Success(seq_stats), CheckResult::Success(adaptive_stats)) => {
                assert_eq!(seq_stats.states_found, adaptive_stats.states_found);
                assert_eq!(seq_stats.initial_states, adaptive_stats.initial_states);
            }
            (seq, adaptive) => panic!("Results differ: seq={:?}, adaptive={:?}", seq, adaptive),
        }
    }

    #[test]
    fn test_adaptive_typeok_fallback() {
        // Test that adaptive mode falls back to TypeOK when Init uses unsupported expressions.
        // This pattern occurs in specs where Init = Inv = TypeOK /\ IInv.
        let src = r#"
---- MODULE TypeOKFallback ----
VARIABLES x, y

TypeOK == x \in {0, 1, 2} /\ y \in {0, 1}

(* IInv has a nested IF expression that cannot be directly extracted *)
IInv == IF x = 0 THEN y = 0 ELSE TRUE

(* Init uses conjunction with unsupported expression *)
Inv == TypeOK /\ IInv

Init == Inv

Next == x' \in {0, 1, 2} /\ y' \in {0, 1}
====
"#;
        let module = parse_module(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        let mut checker = AdaptiveChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        // This should succeed by falling back to TypeOK enumeration + filtering
        let (result, analysis) = checker.check();

        assert!(
            matches!(result, CheckResult::Success(_)),
            "Expected success with TypeOK fallback, got {:?}",
            result
        );

        let analysis = analysis.unwrap();
        // TypeOK gives 3*2=6 states, but IInv filters when x=0 to require y=0
        // When x=0: only y=0 is valid (1 state)
        // When x=1,2: y can be 0 or 1 (4 states)
        // Total: 5 initial states
        assert_eq!(
            analysis.initial_states, 5,
            "Expected 5 initial states after filtering by IInv"
        );
    }

    #[test]
    fn test_adaptive_allows_assume_only_models_without_init_next() {
        let src = r#"
---- MODULE AssumeOnly ----
ASSUME TRUE
====
"#;
        let module = parse_module(src);
        let config = Config::default();

        let mut checker = AdaptiveChecker::new(&module, &config);
        let (result, analysis) = checker.check();

        assert!(matches!(result, CheckResult::Success(_)));
        assert!(analysis.is_some());
        assert!(matches!(analysis.unwrap().strategy, Strategy::Sequential));
    }
}
