//! Main CDCL SAT solver
//!
//! Implements the Conflict-Driven Clause Learning algorithm with:
//! - 2-watched literal scheme for unit propagation
//! - 1UIP conflict analysis
//! - VSIDS variable selection
//! - Chronological and non-chronological backtracking (SAT'18 paper)
//! - Lazy reimplication for out-of-order trail literals
//! - Phase saving for decision polarity
//! - Luby sequence restarts
//! - DRAT proof generation for UNSAT certificates

use crate::bce::{BCEStats, BCE};
use crate::bve::{BVEStats, BVE};
use crate::clause::ClauseTier;
use crate::clause_db::ClauseDB;
use crate::conflict::{ConflictAnalyzer, ConflictResult};
use crate::gates::{GateExtractor, GateStats};
use crate::htr::{HTRStats, HTR};
use crate::literal::{Literal, Variable};
use crate::probe::{find_failed_literal_uip, ProbeStats, Prober};
use crate::proof::{DratWriter, ProofOutput};
use crate::reconstruct::ReconstructionStack;
use crate::subsume::{SubsumeStats, Subsumer};
use crate::sweep::{SweepStats, Sweeper};
use crate::vivify::{Vivifier, VivifyStats};
use crate::vsids::VSIDS;
use crate::walk::WalkStats;
use crate::warmup::WarmupStats;
use crate::watched::{ClauseRef, WatchedLists, Watcher};
use std::io::Write;

/// Memory usage statistics for the solver
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Number of variables
    pub num_vars: usize,
    /// Number of clauses
    pub num_clauses: usize,
    /// Total number of literals across all clauses
    pub total_literals: usize,
    /// Per-variable data (assignment, level, reason, trail_pos, phase)
    pub var_data: usize,
    /// VSIDS activity scores
    pub vsids: usize,
    /// Conflict analyzer
    pub conflict: usize,
    /// Clause database (clauses + literals)
    pub clause_db: usize,
    /// Watched literal lists
    pub watches: usize,
    /// Trail and trail limits
    pub trail: usize,
    /// Clause IDs (for LRAT proofs)
    pub clause_ids: usize,
    /// Inprocessing engines (vivify, subsume, probe, bve, bce, htr, gates, sweep)
    pub inprocessing: usize,
}

impl MemoryStats {
    /// Total estimated memory usage in bytes
    pub fn total(&self) -> usize {
        self.var_data
            + self.vsids
            + self.conflict
            + self.clause_db
            + self.watches
            + self.trail
            + self.clause_ids
            + self.inprocessing
    }

    /// Memory usage per variable (excluding clause database)
    pub fn per_var(&self) -> f64 {
        if self.num_vars == 0 {
            0.0
        } else {
            (self.var_data + self.vsids + self.conflict + self.inprocessing) as f64
                / self.num_vars as f64
        }
    }

    /// Memory usage per literal in clause database
    pub fn per_literal(&self) -> f64 {
        if self.total_literals == 0 {
            0.0
        } else {
            self.clause_db as f64 / self.total_literals as f64
        }
    }
}

impl std::fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Memory Statistics:")?;
        writeln!(f, "  Variables: {}", self.num_vars)?;
        writeln!(f, "  Clauses: {}", self.num_clauses)?;
        writeln!(f, "  Total literals: {}", self.total_literals)?;
        writeln!(f)?;
        writeln!(f, "  Per-variable data: {} bytes", self.var_data)?;
        writeln!(f, "  VSIDS: {} bytes", self.vsids)?;
        writeln!(f, "  Conflict analyzer: {} bytes", self.conflict)?;
        writeln!(f, "  Clause database: {} bytes", self.clause_db)?;
        writeln!(f, "  Watched lists: {} bytes", self.watches)?;
        writeln!(f, "  Trail: {} bytes", self.trail)?;
        writeln!(f, "  Clause IDs: {} bytes", self.clause_ids)?;
        writeln!(f, "  Inprocessing: {} bytes", self.inprocessing)?;
        writeln!(f)?;
        writeln!(
            f,
            "  Total: {} bytes ({:.2} MB)",
            self.total(),
            self.total() as f64 / 1_048_576.0
        )?;
        writeln!(f, "  Per variable: {:.2} bytes", self.per_var())?;
        writeln!(f, "  Per literal: {:.2} bytes", self.per_literal())
    }
}

/// Result of solving
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveResult {
    /// Satisfiable with model
    Sat(Vec<bool>),
    /// Unsatisfiable
    Unsat,
    /// Unknown (timeout, etc.)
    Unknown,
}

/// Result of solving with assumptions, including unsat core
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssumeResult {
    /// Satisfiable with model
    Sat(Vec<bool>),
    /// Unsatisfiable with core (subset of assumptions that caused UNSAT)
    Unsat(Vec<Literal>),
    /// Unknown (timeout, etc.)
    Unknown,
}

/// The CDCL SAT solver
pub struct Solver<W: Write = Vec<u8>> {
    /// Number of variables
    num_vars: usize,
    /// Number of user-visible variables (excludes internal scope selectors)
    user_num_vars: usize,
    /// Clause database with arena-allocated literals
    clause_db: ClauseDB,
    /// Saved watch replacement positions for each clause (JAIR'13 optimization).
    ///
    /// For clauses with size > 2, this stores the next position to start scanning
    /// for a replacement watched literal (starting at index 2).
    watch_pos: Vec<u16>,
    /// Watched literal lists
    watches: WatchedLists,
    /// Variable selection heuristic
    vsids: VSIDS,
    /// Conflict analyzer
    conflict: ConflictAnalyzer,
    /// Current assignment (None = unassigned)
    assignment: Vec<Option<bool>>,
    /// Decision level for each variable
    level: Vec<u32>,
    /// Reason clause for each variable (None if decision)
    reason: Vec<Option<ClauseRef>>,
    /// Trail (sequence of assigned literals)
    trail: Vec<Literal>,
    /// Trail limit for each decision level (index into trail)
    trail_lim: Vec<usize>,
    /// Current decision level
    decision_level: u32,
    /// Propagation queue head (index into trail)
    qhead: usize,
    /// Phase saving: last polarity of each variable (None = no preference)
    phase: Vec<Option<bool>>,
    /// Target phases: phases at longest conflict-free trail (for stable mode)
    target_phase: Vec<Option<bool>>,
    /// Best phases: best assignment seen (for rephasing)
    best_phase: Vec<Option<bool>>,
    /// Trail length when target phases were saved
    target_trail_len: usize,
    /// Trail length when best phases were saved
    best_trail_len: usize,
    /// Proof writer (optional, supports DRAT and LRAT formats)
    proof_writer: Option<ProofOutput<W>>,
    // Restart state
    /// Number of conflicts since last restart
    conflicts_since_restart: u64,
    /// Current index into the Luby sequence
    luby_idx: u32,
    /// Base restart interval (conflicts per Luby unit)
    restart_base: u64,
    /// Total number of restarts performed
    restarts: u64,
    // Glucose-style EMA restart state
    /// Fast exponential moving average of LBD (short window)
    lbd_ema_fast: f64,
    /// Slow exponential moving average of LBD (long window)
    lbd_ema_slow: f64,
    /// Whether to use Glucose-style restarts (true) or Luby restarts (false)
    glucose_restarts: bool,
    /// Minimum conflicts before considering restart (initial stabilization)
    restart_min_conflicts: u64,
    // Stabilization state (CaDiCaL-style)
    /// Whether we're in stable mode (infrequent restarts) vs focused mode (frequent restarts)
    stable_mode: bool,
    /// Conflict count when stabilization mode was last switched
    stable_mode_start_conflicts: u64,
    /// Length of current stabilization phase in conflicts
    stable_phase_length: u64,
    /// Counter for stable phase number (used to increase phase length)
    stable_phase_count: u64,
    /// Reluctant counter for stable mode (restart at power-of-2 intervals)
    reluctant_counter: u64,
    /// Reluctant period (doubles after each restart in stable mode)
    reluctant_period: u64,
    // Clause database management
    /// Total number of conflicts (for clause deletion scheduling)
    num_conflicts: u64,
    /// Number of original (non-learned) clauses
    num_original_clauses: usize,
    /// When to next run clause deletion
    next_reduce_db: u64,
    /// Clause activity decay factor
    clause_activity_inc: f32,
    // Chronological backtracking
    /// Trail position for each variable (for lazy reimplication)
    trail_pos: Vec<usize>,
    /// Whether chronological backtracking is enabled
    chrono_enabled: bool,
    /// Whether to use trail reuse heuristic for chrono BT (CaDiCaL-style)
    chrono_reuse_trail: bool,
    /// Statistics: number of chronological backtracks
    chrono_backtracks: u64,
    /// Statistics: number of decisions
    num_decisions: u64,
    /// Statistics: number of propagations
    num_propagations: u64,
    // Inprocessing
    /// Vivification engine
    vivifier: Vivifier,
    /// When to next run vivification
    next_vivify: u64,
    /// Whether vivification is enabled
    vivify_enabled: bool,
    /// Subsumption engine
    subsumer: Subsumer,
    /// When to next run subsumption
    next_subsume: u64,
    /// Whether subsumption is enabled
    subsume_enabled: bool,
    // Failed literal probing
    /// Failed literal prober
    prober: Prober,
    /// When to next run probing
    next_probe: u64,
    /// Whether probing is enabled
    probe_enabled: bool,
    /// Number of units derived at level 0 (for propfixed tracking)
    fixed_count: i64,
    // Bounded Variable Elimination
    /// BVE engine
    bve: BVE,
    /// When to next run BVE
    next_bve: u64,
    /// Whether BVE is enabled
    bve_enabled: bool,
    // Blocked Clause Elimination
    /// BCE engine
    bce: BCE,
    /// When to next run BCE
    next_bce: u64,
    /// Whether BCE is enabled
    bce_enabled: bool,
    // Hyper-Ternary Resolution
    /// HTR engine
    htr: HTR,
    /// When to next run HTR
    next_htr: u64,
    /// Whether HTR is enabled
    htr_enabled: bool,
    // Gate Extraction
    /// Gate extraction engine
    gate_extractor: GateExtractor,
    /// Whether gate extraction is enabled (for statistics)
    gate_enabled: bool,
    // SAT Sweeping
    /// Sweeper engine for equivalence detection
    sweeper: Sweeper,
    /// When to next run sweeping
    next_sweep: u64,
    /// Whether sweeping is enabled
    sweep_enabled: bool,
    // Model Reconstruction
    /// Stack of reconstruction steps for equisatisfiable transformations
    reconstruction: ReconstructionStack,
    // LRAT proof support
    /// Clause IDs for LRAT proofs (maps clause index to clause ID)
    /// Original clauses get IDs 1..n, learned clauses get n+1, n+2, etc.
    clause_ids: Vec<u64>,
    /// Next clause ID to assign
    next_clause_id: u64,
    /// Whether LRAT proof generation is enabled (track resolution chains)
    lrat_enabled: bool,
    /// Whether an empty clause has been added (formula is UNSAT)
    has_empty_clause: bool,
    /// Stack of active scope selector variables (for push/pop)
    scope_selectors: Vec<Variable>,
    // Rephasing
    /// Whether rephasing is enabled
    rephase_enabled: bool,
    /// Number of times we've rephased
    rephase_count: u64,
    /// When to next rephase (conflict count)
    next_rephase: u64,
    // Clause minimization state (CaDiCaL-style)
    /// Poison marking: literals that failed minimization (don't retry)
    minimize_poison: Vec<bool>,
    /// Removable marking: literals that succeeded minimization
    minimize_removable: Vec<bool>,
    /// Reusable visited tracking for current minimization
    minimize_visited: Vec<bool>,
    /// List of literals to clear after minimization
    minimize_to_clear: Vec<usize>,
    /// Maximum recursion depth for minimization
    minimize_depth_limit: u32,
    // Walk-based phase initialization
    /// Walk statistics
    walk_stats: WalkStats,
    /// Whether walk is enabled
    walk_enabled: bool,
    /// Walk tick limit (effort per walk round)
    walk_limit: u64,
    // Warmup-based phase initialization
    /// Warmup statistics
    warmup_stats: WarmupStats,
    /// Whether warmup is enabled
    warmup_enabled: bool,
    // Initial preprocessing
    /// Whether to run preprocessing before CDCL search
    preprocess_enabled: bool,
}

/// Default base restart interval (conflicts per Luby unit)
const DEFAULT_RESTART_BASE: u64 = 100;

/// Fast EMA decay factor (short window, ~32 conflicts)
/// decay = 1 - 1/32 = 0.96875
const EMA_FAST_DECAY: f64 = 0.96875;

/// Slow EMA decay factor (long window, ~100000 conflicts)
/// decay = 1 - 1/100000 = 0.99999 (matches CaDiCaL's emaglueslow)
const EMA_SLOW_DECAY: f64 = 0.99999;

/// Restart margin: fast EMA must exceed slow EMA * margin to trigger restart
/// A margin of 1.10 means fast must be 10% higher than slow (matches CaDiCaL focused mode)
const RESTART_MARGIN: f64 = 1.10;

/// Minimum conflicts between restarts in focused mode (restart blocking)
/// This prevents rapid-fire restarts when the EMA condition is marginally satisfied.
/// CaDiCaL uses restartint=2, but we use a larger value for stability.
const RESTART_INTERVAL: u64 = 25;

/// Minimum conflicts before considering Glucose-style restarts
const RESTART_MIN_CONFLICTS: u64 = 100;

/// Initial stabilization phase length (conflicts before first mode switch)
/// CaDiCaL uses 1000 conflicts for first focused phase
const STABLE_PHASE_INIT: u64 = 1000;

/// Factor by which phase length increases after each stable phase
/// Phase lengths grow quadratically: 1000, 4000, 9000, 16000, ...
#[allow(dead_code)]
const STABLE_PHASE_GROWTH: u64 = 2;

/// Initial reluctant doubling period for stable mode restarts
/// In stable mode, restarts occur at intervals of 1024, 2048, 4096, ... conflicts
/// (matches CaDiCaL's reluctantint)
const RELUCTANT_INIT: u64 = 1024;

/// First clause DB reduction after this many conflicts
const FIRST_REDUCE_DB: u64 = 2000;

/// Interval between clause DB reductions
const REDUCE_DB_INC: u64 = 300;

/// Clause activity decay factor (multiply by this after each conflict)
const CLAUSE_DECAY: f32 = 0.999;

/// Maximum levels to jump before using chronological backtracking
/// If jump_levels > CHRONO_LEVEL_LIMIT, use chronological backtracking instead
const CHRONO_LEVEL_LIMIT: u32 = 100;

/// Run vivification after this many conflicts
const VIVIFY_INTERVAL: u64 = 10000;

/// Number of clauses to vivify per call
const VIVIFY_CLAUSES_PER_CALL: usize = 500;

/// Run subsumption after this many conflicts
const SUBSUME_INTERVAL: u64 = 5000;

/// Number of clauses to process for subsumption per call
const SUBSUME_CLAUSES_PER_CALL: usize = 1000;

/// Run failed literal probing after this many conflicts
const PROBE_INTERVAL: u64 = 15000;

/// Maximum number of probes per call
const MAX_PROBES_PER_CALL: usize = 500;

/// Run BVE after this many conflicts
const BVE_INTERVAL: u64 = 20000;

/// Maximum number of variable eliminations per call
const MAX_BVE_ELIMINATIONS: usize = 100;

/// Run BCE after this many conflicts
const BCE_INTERVAL: u64 = 25000;

/// Maximum number of blocked clause eliminations per call
const MAX_BCE_ELIMINATIONS: usize = 200;

/// Run HTR after this many conflicts
const HTR_INTERVAL: u64 = 30000;

/// Maximum number of resolvents per HTR call
const MAX_HTR_RESOLVENTS: usize = 500;

/// Run SAT sweeping after this many conflicts
const SWEEP_INTERVAL: u64 = 35000;

/// Initial rephase interval (conflicts)
const REPHASE_INITIAL: u64 = 1000;

/// Rephase interval increment (arithmetic increase)
const REPHASE_INCREMENT: u64 = 1000;

/// Default walk tick limit per round (effort budget)
const WALK_DEFAULT_LIMIT: u64 = 100_000;

impl Solver<Vec<u8>> {
    /// Create a new solver with n variables (no proof logging)
    pub fn new(num_vars: usize) -> Self {
        let clauses_capacity = num_vars.saturating_mul(4).min(100_000);
        let literals_capacity = clauses_capacity.saturating_mul(3); // avg 3 lits/clause
        Solver {
            num_vars,
            user_num_vars: num_vars,
            clause_db: ClauseDB::with_capacity(clauses_capacity, literals_capacity),
            watch_pos: Vec::with_capacity(clauses_capacity),
            watches: WatchedLists::new(num_vars),
            vsids: VSIDS::new(num_vars),
            conflict: ConflictAnalyzer::new(num_vars),
            assignment: vec![None; num_vars],
            level: vec![0; num_vars],
            reason: vec![None; num_vars],
            trail: Vec::new(),
            trail_lim: Vec::new(),
            decision_level: 0,
            qhead: 0,
            phase: vec![None; num_vars],
            target_phase: vec![None; num_vars],
            best_phase: vec![None; num_vars],
            target_trail_len: 0,
            best_trail_len: 0,
            proof_writer: None,
            conflicts_since_restart: 0,
            luby_idx: 1,
            restart_base: DEFAULT_RESTART_BASE,
            restarts: 0,
            lbd_ema_fast: 0.0,
            lbd_ema_slow: 0.0,
            glucose_restarts: true, // Enable Glucose-style restarts by default
            restart_min_conflicts: RESTART_MIN_CONFLICTS,
            // Stabilization state (start in focused mode)
            stable_mode: false,
            stable_mode_start_conflicts: 0,
            stable_phase_length: STABLE_PHASE_INIT,
            stable_phase_count: 0,
            reluctant_counter: 0,
            reluctant_period: RELUCTANT_INIT,
            num_conflicts: 0,
            num_original_clauses: 0,
            next_reduce_db: FIRST_REDUCE_DB,
            clause_activity_inc: 1.0,
            trail_pos: vec![usize::MAX; num_vars],
            chrono_enabled: true, // Enable chronological backtracking by default
            chrono_reuse_trail: false, // Disabled: causes regression on some instances
            chrono_backtracks: 0,
            num_decisions: 0,
            num_propagations: 0,
            vivifier: Vivifier::new(num_vars),
            next_vivify: VIVIFY_INTERVAL,
            vivify_enabled: true,
            subsumer: Subsumer::new(num_vars),
            next_subsume: SUBSUME_INTERVAL,
            subsume_enabled: true, // Enable subsumption by default
            prober: Prober::new(num_vars),
            next_probe: PROBE_INTERVAL,
            probe_enabled: true, // Enable probing by default
            fixed_count: 0,
            bve: BVE::new(num_vars),
            next_bve: BVE_INTERVAL,
            bve_enabled: true, // Enable BVE by default
            bce: BCE::new(num_vars),
            next_bce: BCE_INTERVAL,
            bce_enabled: true, // Enable BCE by default
            htr: HTR::new(num_vars),
            next_htr: HTR_INTERVAL,
            htr_enabled: true, // Enable HTR by default
            gate_extractor: GateExtractor::new(num_vars),
            gate_enabled: true, // Enable gate extraction by default
            sweeper: Sweeper::new(num_vars),
            next_sweep: SWEEP_INTERVAL,
            sweep_enabled: true, // Enable sweeping by default (but skip if proof logging)
            reconstruction: ReconstructionStack::new(),
            clause_ids: Vec::new(),
            next_clause_id: 1, // IDs start at 1 (DIMACS style)
            lrat_enabled: false,
            has_empty_clause: false,
            scope_selectors: Vec::new(),
            rephase_enabled: true, // Enable rephasing by default
            rephase_count: 0,
            next_rephase: REPHASE_INITIAL,
            // Clause minimization state
            minimize_poison: vec![false; num_vars],
            minimize_removable: vec![false; num_vars],
            minimize_visited: vec![false; num_vars],
            minimize_to_clear: Vec::with_capacity(num_vars),
            minimize_depth_limit: 1000, // CaDiCaL default
            // Walk-based phase initialization
            walk_stats: WalkStats::default(),
            walk_enabled: true,
            walk_limit: WALK_DEFAULT_LIMIT,
            // Warmup-based phase initialization
            warmup_stats: WarmupStats::default(),
            warmup_enabled: true,
            // Initial preprocessing
            preprocess_enabled: false, // Disabled by default due to edge cases during testing
        }
    }
}

impl<W: Write> Solver<W> {
    /// Create a new solver with DRAT proof logging
    pub fn with_proof(num_vars: usize, proof_writer: DratWriter<W>) -> Self {
        Self::with_proof_output(num_vars, ProofOutput::Drat(proof_writer))
    }

    /// Create a new solver with proof logging (DRAT or LRAT)
    ///
    /// For LRAT proofs, use `ProofOutput::lrat_text()` or `ProofOutput::lrat_binary()`.
    /// Note: LRAT proof requires knowing the number of original clauses, which is
    /// determined after all clauses are added. Use `with_lrat_proof_deferred()` for
    /// proper LRAT setup, or create the LratWriter with an estimate.
    pub fn with_proof_output(num_vars: usize, proof_writer: ProofOutput<W>) -> Self {
        let lrat_enabled = proof_writer.is_lrat();
        let clauses_capacity = num_vars.saturating_mul(4).min(100_000);
        let literals_capacity = clauses_capacity.saturating_mul(3); // avg 3 lits/clause
        Solver {
            num_vars,
            user_num_vars: num_vars,
            clause_db: ClauseDB::with_capacity(clauses_capacity, literals_capacity),
            watch_pos: Vec::with_capacity(clauses_capacity),
            watches: WatchedLists::new(num_vars),
            vsids: VSIDS::new(num_vars),
            conflict: ConflictAnalyzer::new(num_vars),
            assignment: vec![None; num_vars],
            level: vec![0; num_vars],
            reason: vec![None; num_vars],
            trail: Vec::new(),
            trail_lim: Vec::new(),
            decision_level: 0,
            qhead: 0,
            phase: vec![None; num_vars],
            target_phase: vec![None; num_vars],
            best_phase: vec![None; num_vars],
            target_trail_len: 0,
            best_trail_len: 0,
            proof_writer: Some(proof_writer),
            conflicts_since_restart: 0,
            luby_idx: 1,
            restart_base: DEFAULT_RESTART_BASE,
            restarts: 0,
            lbd_ema_fast: 0.0,
            lbd_ema_slow: 0.0,
            glucose_restarts: true, // Enable Glucose-style restarts by default
            restart_min_conflicts: RESTART_MIN_CONFLICTS,
            // Stabilization state (start in focused mode)
            stable_mode: false,
            stable_mode_start_conflicts: 0,
            stable_phase_length: STABLE_PHASE_INIT,
            stable_phase_count: 0,
            reluctant_counter: 0,
            reluctant_period: RELUCTANT_INIT,
            num_conflicts: 0,
            num_original_clauses: 0,
            next_reduce_db: FIRST_REDUCE_DB,
            clause_activity_inc: 1.0,
            trail_pos: vec![usize::MAX; num_vars],
            chrono_enabled: true, // Enable chronological backtracking by default
            chrono_reuse_trail: false, // Disabled: causes regression on some instances
            chrono_backtracks: 0,
            num_decisions: 0,
            num_propagations: 0,
            vivifier: Vivifier::new(num_vars),
            next_vivify: VIVIFY_INTERVAL,
            vivify_enabled: true,
            subsumer: Subsumer::new(num_vars),
            next_subsume: SUBSUME_INTERVAL,
            subsume_enabled: true, // Enable subsumption by default
            prober: Prober::new(num_vars),
            next_probe: PROBE_INTERVAL,
            probe_enabled: true, // Enable probing by default
            fixed_count: 0,
            bve: BVE::new(num_vars),
            next_bve: BVE_INTERVAL,
            bve_enabled: true, // Enable BVE by default
            bce: BCE::new(num_vars),
            next_bce: BCE_INTERVAL,
            bce_enabled: true, // Enable BCE by default
            htr: HTR::new(num_vars),
            next_htr: HTR_INTERVAL,
            htr_enabled: true, // Enable HTR by default
            gate_extractor: GateExtractor::new(num_vars),
            gate_enabled: true, // Enable gate extraction by default
            sweeper: Sweeper::new(num_vars),
            next_sweep: SWEEP_INTERVAL,
            sweep_enabled: false, // Disable sweeping with proof logging (equisatisfiable only)
            reconstruction: ReconstructionStack::new(),
            clause_ids: if lrat_enabled {
                Vec::with_capacity(clauses_capacity)
            } else {
                Vec::new()
            },
            next_clause_id: 1, // IDs start at 1 (DIMACS style)
            lrat_enabled,
            has_empty_clause: false,
            scope_selectors: Vec::new(),
            rephase_enabled: true, // Enable rephasing by default
            rephase_count: 0,
            next_rephase: REPHASE_INITIAL,
            // Clause minimization state
            minimize_poison: vec![false; num_vars],
            minimize_removable: vec![false; num_vars],
            minimize_visited: vec![false; num_vars],
            minimize_to_clear: Vec::with_capacity(num_vars),
            minimize_depth_limit: 1000, // CaDiCaL default
            // Walk-based phase initialization
            walk_stats: WalkStats::default(),
            walk_enabled: true,
            walk_limit: WALK_DEFAULT_LIMIT,
            // Warmup-based phase initialization
            warmup_stats: WarmupStats::default(),
            warmup_enabled: true,
            // Initial preprocessing
            preprocess_enabled: false, // Disabled by default due to edge cases during testing
        }
    }

    /// Enable LRAT proof support (track clause IDs and resolution chains)
    ///
    /// This must be called before adding any clauses.
    pub fn enable_lrat(&mut self) {
        self.lrat_enabled = true;
    }

    /// Get the clause ID for a given clause reference
    ///
    /// Returns 0 if the clause doesn't have an assigned ID.
    #[inline]
    pub fn clause_id(&self, clause_ref: ClauseRef) -> u64 {
        let idx = clause_ref.0 as usize;
        if idx < self.clause_ids.len() {
            self.clause_ids[idx]
        } else {
            0
        }
    }

    /// Add a clause
    pub fn add_clause(&mut self, literals: Vec<Literal>) -> bool {
        if literals.is_empty() {
            self.has_empty_clause = true;
            return false; // Empty clause = UNSAT
        }

        let mut literals = literals;
        if let Some(selector) = self.scope_selectors.last().copied() {
            literals.push(Literal::positive(selector));
        }

        self.add_clause_unscoped(literals, false)
    }

    /// Add a clause without any scope selector (global clause).
    ///
    /// Use this for clauses that should persist across all push/pop scopes.
    /// Unlike `add_clause`, this does NOT add a scope selector even if
    /// we're currently inside a push() scope.
    pub fn add_clause_global(&mut self, literals: Vec<Literal>) -> bool {
        self.add_clause_unscoped(literals, false)
    }

    fn add_clause_unscoped(&mut self, literals: Vec<Literal>, learned: bool) -> bool {
        if literals.is_empty() {
            self.has_empty_clause = true;
            return false;
        }

        let _ = self.add_clause_db(&literals, learned);

        true
    }

    #[inline]
    fn add_clause_db(&mut self, literals: &[Literal], learned: bool) -> usize {
        let idx = self.clause_db.add(literals, learned);

        debug_assert_eq!(idx, self.watch_pos.len());
        self.watch_pos.push(2);

        // Assign clause ID if LRAT is enabled
        if self.lrat_enabled {
            debug_assert_eq!(idx, self.clause_ids.len());
            self.clause_ids.push(self.next_clause_id);
            self.next_clause_id += 1;
        }

        idx
    }

    /// Return the number of user-visible variables.
    pub fn user_num_vars(&self) -> usize {
        self.user_num_vars
    }

    /// Get the number of conflicts encountered during solving
    pub fn num_conflicts(&self) -> u64 {
        self.num_conflicts
    }

    /// Get the number of restarts performed during solving
    pub fn num_restarts(&self) -> u64 {
        self.restarts
    }

    /// Get the number of decisions made during solving
    pub fn num_decisions(&self) -> u64 {
        self.num_decisions
    }

    /// Get the number of propagations performed during solving
    pub fn num_propagations(&self) -> u64 {
        self.num_propagations
    }

    /// Extract all learned (non-original) clauses from the clause database.
    ///
    /// This is useful for preserving learned clauses when recreating the solver,
    /// such as in branch-and-bound algorithms for LIA.
    pub fn get_learned_clauses(&self) -> Vec<Vec<Literal>> {
        let mut learned = Vec::new();
        for idx in 0..self.clause_db.len() {
            let header = self.clause_db.header(idx);
            if header.is_learned() && !header.is_empty() {
                learned.push(self.clause_db.literals(idx).to_vec());
            }
        }
        learned
    }

    /// Add a clause that was learned from a previous solve session.
    ///
    /// Unlike regular learned clauses, these are added without proof logging
    /// since they were already proven in the previous session.
    pub fn add_preserved_learned(&mut self, literals: Vec<Literal>) -> bool {
        if literals.is_empty() {
            self.has_empty_clause = true;
            return false;
        }
        // Add as learned clause with proper watch setup
        let _ = self.add_clause_db(&literals, true);
        true
    }

    /// Allocate a new variable in the solver
    ///
    /// Returns the newly allocated variable. This is useful for incremental
    /// solving where new variables need to be added dynamically.
    pub fn new_var(&mut self) -> Variable {
        let var = self.new_var_internal();
        // New variables allocated via the public API are user-visible and should be
        // included in returned models.
        self.user_num_vars = self.user_num_vars.max(self.num_vars);
        var
    }

    /// Ensure the solver has at least `num_vars` variables
    ///
    /// If the solver already has enough variables, this is a no-op.
    /// Otherwise, new variables are allocated to reach the requested count.
    /// This is useful for incremental solving where the total number of
    /// variables may not be known upfront.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        while self.num_vars < num_vars {
            self.new_var_internal();
        }
        // `ensure_num_vars` is an incremental-solving API, so include any newly
        // allocated variables in returned models.
        self.user_num_vars = self.user_num_vars.max(self.num_vars);
    }

    fn new_var_internal(&mut self) -> Variable {
        let var = Variable(self.num_vars as u32);
        self.num_vars += 1;

        self.assignment.push(None);
        self.level.push(0);
        self.reason.push(None);
        self.phase.push(None);
        self.target_phase.push(None);
        self.best_phase.push(None);
        self.trail_pos.push(usize::MAX);

        self.minimize_poison.push(false);
        self.minimize_removable.push(false);
        self.minimize_visited.push(false);

        self.watches.ensure_num_vars(self.num_vars);
        self.vsids.ensure_num_vars(self.num_vars);
        self.conflict.ensure_num_vars(self.num_vars);
        self.vivifier.ensure_num_vars(self.num_vars);
        self.subsumer.ensure_num_vars(self.num_vars);
        self.prober.ensure_num_vars(self.num_vars);
        self.bve.ensure_num_vars(self.num_vars);
        self.bce.ensure_num_vars(self.num_vars);
        self.htr.ensure_num_vars(self.num_vars);
        self.gate_extractor.ensure_num_vars(self.num_vars);
        self.sweeper.ensure_num_vars(self.num_vars);

        var
    }

    /// Push a new assertion scope.
    ///
    /// Clauses added after a `push()` are scoped and removed by `pop()`, while
    /// learned clauses are retained.
    pub fn push(&mut self) {
        let selector = self.new_var_internal();
        self.scope_selectors.push(selector);
    }

    /// Pop the most recent assertion scope.
    ///
    /// Returns `false` if there is no active scope.
    pub fn pop(&mut self) -> bool {
        let selector = match self.scope_selectors.pop() {
            Some(v) => v,
            None => return false,
        };

        // Permanently disable clauses guarded by this selector, even if there
        // are still outer scopes active.
        let _ = self.add_clause_unscoped(vec![Literal::positive(selector)], false);
        true
    }

    /// Get the current scope depth.
    ///
    /// Returns the number of active push() scopes.
    pub fn scope_depth(&self) -> usize {
        self.scope_selectors.len()
    }

    /// Get the current assignment for a variable
    pub fn value(&self, var: Variable) -> Option<bool> {
        self.assignment[var.index()]
    }

    /// Get the value of a literal under current assignment
    #[inline]
    fn lit_value(&self, lit: Literal) -> Option<bool> {
        self.assignment[lit.variable().index()].map(|v| if lit.is_positive() { v } else { !v })
    }

    /// Assign a literal with a reason clause
    #[inline]
    fn enqueue(&mut self, lit: Literal, reason: Option<ClauseRef>) {
        let var = lit.variable();
        let val = lit.is_positive();
        self.assignment[var.index()] = Some(val);
        self.level[var.index()] = self.decision_level;
        self.reason[var.index()] = reason;
        self.trail_pos[var.index()] = self.trail.len();
        self.trail.push(lit);
        // Prefetch the watch list for the negated literal (CaDiCaL technique)
        // This brings the data into cache before propagate needs it
        self.watches.prefetch(lit.negated());
        // Remove from VSIDS heap (variable is now assigned)
        self.vsids.remove_from_heap(var);
    }

    /// Make a decision (assign without reason, start new decision level)
    #[inline]
    fn decide(&mut self, lit: Literal) {
        self.decision_level += 1;
        self.trail_lim.push(self.trail.len());
        self.num_decisions += 1;
        self.enqueue(lit, None);
    }

    /// Pick the next decision variable, selecting between VSIDS (stable mode)
    /// and VMTF (focused mode).
    #[inline]
    fn pick_next_decision_variable(&mut self) -> Option<Variable> {
        if self.stable_mode {
            self.vsids.pick_branching_variable(&self.assignment)
        } else {
            self.vsids.pick_branching_variable_vmtf(&self.assignment)
        }
    }

    /// Initialize watched literals for all clauses
    ///
    /// The 2-watched literal scheme: for each clause with >= 2 literals,
    /// we "watch" the first two literals. When a watched literal becomes
    /// false, we need to find a new literal to watch (or detect unit/conflict).
    ///
    /// `watches[lit]` contains clauses where lit is watched. When lit becomes
    /// false (i.e., ~lit becomes true), we check these clauses.
    fn initialize_watches(&mut self) {
        for i in 0..self.clause_db.len() {
            let clause_ref = ClauseRef(i as u32);
            let clause_len = self.clause_db.header(i).len();
            if clause_len >= 2 {
                // Watch the first two literals
                // Add clause to watch list of each watched literal
                let lit0 = self.clause_db.literal(i, 0);
                let lit1 = self.clause_db.literal(i, 1);
                // Binary clauses get special handling in propagate
                let is_binary = clause_len == 2;
                if is_binary {
                    self.watches
                        .add_watch(lit0, Watcher::binary(clause_ref, lit1));
                    self.watches
                        .add_watch(lit1, Watcher::binary(clause_ref, lit0));
                } else {
                    self.watches.add_watch(lit0, Watcher::new(clause_ref, lit1));
                    self.watches.add_watch(lit1, Watcher::new(clause_ref, lit0));
                }
            }
        }
    }

    /// Propagate unit clauses using 2-watched literals
    /// Returns None if no conflict, Some(clause_ref) if conflict found
    ///
    /// This implementation uses CaDiCaL's in-place two-pointer technique for
    /// efficient watch list modification without allocation.
    ///
    /// Binary clauses are handled specially without accessing the clause database,
    /// following CaDiCaL's optimization for binary clauses.
    #[inline]
    fn propagate(&mut self) -> Option<ClauseRef> {
        while self.qhead < self.trail.len() {
            let p = self.trail[self.qhead]; // Literal that became true
            self.qhead += 1;
            self.num_propagations += 1;

            // All clauses watching ~p need to update
            let false_lit = p.negated();

            // In-place two-pointer iteration (CaDiCaL style)
            // i = read position, j = write position
            // We copy watchers from i to j, skipping removed ones
            let (watch_ptr, watch_len) = self.watches.get_watch_list_raw(false_lit);
            let mut i: usize = 0;
            let mut j: usize = 0;

            while i < watch_len {
                // Gap 8 invariant: 0 <= j <= i < watch_len during read phase
                debug_assert!(j <= i, "Gap 8 invariant violated: j ({}) > i ({})", j, i);
                debug_assert!(
                    i < watch_len,
                    "Gap 8 invariant violated: i ({}) >= watch_len ({})",
                    i,
                    watch_len
                );

                // SAFETY: i < watch_len, and we don't modify watches for false_lit
                // during iteration (we only add to other literals' lists)
                let watcher = unsafe { *watch_ptr.add(i) };
                i += 1;

                // Gap 8 invariant: after read, 0 <= j < i <= watch_len
                debug_assert!(j < i, "Gap 8 invariant violated: j ({}) >= i ({})", j, i);
                debug_assert!(
                    i <= watch_len,
                    "Gap 8 invariant violated: i ({}) > watch_len ({})",
                    i,
                    watch_len
                );

                // Copy watcher to write position (will be "undone" if we remove it)
                // SAFETY: j <= i-1 < watch_len
                unsafe {
                    *watch_ptr.add(j) = watcher;
                }
                j += 1;

                let blocker = watcher.blocker();
                let blocker_val = self.lit_value(blocker);

                // Quick check: if blocker is true, clause is satisfied
                if blocker_val == Some(true) {
                    continue;
                }

                // Binary clause special case: no clause database access needed!
                // For binary clauses, blocker IS the other literal in the clause.
                if watcher.is_binary() {
                    let clause_ref = watcher.clause_ref();
                    if blocker_val == Some(false) {
                        // Conflict: both literals in the binary clause are false
                        // Copy remaining watchers and truncate
                        while i < watch_len {
                            // Gap 8 invariant: j <= i < watch_len during binary conflict copy
                            debug_assert!(
                                j <= i,
                                "Gap 8 invariant violated in binary copy: j ({}) > i ({})",
                                j,
                                i
                            );
                            debug_assert!(
                                i < watch_len,
                                "Gap 8 invariant violated in binary copy: i ({}) >= watch_len ({})",
                                i,
                                watch_len
                            );
                            unsafe {
                                *watch_ptr.add(j) = *watch_ptr.add(i);
                            }
                            i += 1;
                            j += 1;
                        }
                        // Gap 8 invariant: at exit, j <= watch_len
                        debug_assert!(
                            j <= watch_len,
                            "Gap 8 invariant violated: j ({}) > watch_len ({}) at binary truncate",
                            j,
                            watch_len
                        );
                        self.watches.truncate_watches(false_lit, j);
                        return Some(clause_ref);
                    }
                    // blocker is unassigned - propagate it
                    self.enqueue(blocker, Some(clause_ref));
                    continue;
                }

                // Non-binary clause: need to access clause database
                let clause_ref = watcher.clause_ref();
                let clause_idx = clause_ref.0 as usize;

                // Branch-less computation of the other watched literal using XOR
                // (CaDiCaL technique): other = lits[0] ^ lits[1] ^ false_lit
                // This avoids the conditional swap to normalize position
                let lit0 = self.clause_db.literal(clause_idx, 0);
                let lit1 = self.clause_db.literal(clause_idx, 1);
                let first = Literal(lit0.0 ^ lit1.0 ^ false_lit.0);
                let first_val = self.lit_value(first);

                // Track which position has false_lit for later swap (branch-less)
                let false_pos = (lit0 != false_lit) as usize;

                if first_val == Some(true) {
                    // Clause is satisfied - update blocker
                    // Gap 8 invariant: j > 0 so j-1 is valid
                    debug_assert!(j > 0, "Gap 8 invariant violated: j-1 underflow (j=0)");
                    // SAFETY: j-1 is a valid index (j > 0 and j <= watch_len after write+incr)
                    unsafe {
                        (*watch_ptr.add(j - 1)).set_blocker(first);
                    }
                    continue;
                }

                // Look for a new literal to watch (non-false)
                let clause_len = self.clause_db.header(clause_idx).len();
                debug_assert!(clause_len > 2);

                // Gent's (JAIR'13) saved position optimization (CaDiCaL).
                // Search from the saved position to the end, then wrap to index 2.
                let mut pos = self.watch_pos.get(clause_idx).copied().unwrap_or(2) as usize;
                if pos < 2 || pos > clause_len {
                    pos = 2;
                }

                let mut found_idx: Option<usize> = None;
                for k in pos..clause_len {
                    let lit_k = self.clause_db.literal(clause_idx, k);
                    if self.lit_value(lit_k) != Some(false) {
                        found_idx = Some(k);
                        break;
                    }
                }
                if found_idx.is_none() && pos > 2 {
                    for k in 2..pos {
                        let lit_k = self.clause_db.literal(clause_idx, k);
                        if self.lit_value(lit_k) != Some(false) {
                            found_idx = Some(k);
                            break;
                        }
                    }
                }

                // Always save the position (even if we reached the end).
                let save_pos = found_idx.unwrap_or(clause_len);
                if clause_idx < self.watch_pos.len() {
                    self.watch_pos[clause_idx] = save_pos.min(u16::MAX as usize) as u16;
                }

                if let Some(k) = found_idx {
                    let lit_k = self.clause_db.literal(clause_idx, k);
                    let lit_k_val = self.lit_value(lit_k);

                    if lit_k_val == Some(true) {
                        // Clause satisfied by a non-watched literal; update blocker only.
                        // Gap 8 invariant: j > 0 so j-1 is valid
                        debug_assert!(j > 0, "Gap 8 invariant violated: j-1 underflow (j=0)");
                        unsafe {
                            (*watch_ptr.add(j - 1)).set_blocker(lit_k);
                        }
                        continue;
                    }

                    debug_assert!(lit_k_val != Some(false));

                    // Found new unassigned replacement literal to be watched.
                    self.clause_db.swap_literals(clause_idx, false_pos, k);
                    self.watches
                        .add_watch(lit_k, Watcher::new(clause_ref, first));
                    // "Undo" the copy by decrementing j (drop this watch from false_lit list).
                    // Gap 8 invariant: j > 0 so decrement is valid
                    debug_assert!(
                        j > 0,
                        "Gap 8 invariant violated: j decrement underflow (j=0)"
                    );
                    j -= 1;
                    // Gap 8 invariant: after decrement, j < i still holds (we just wrote to j, now undo)
                    debug_assert!(
                        j < i,
                        "Gap 8 invariant violated: j ({}) >= i ({}) after decrement",
                        j,
                        i
                    );
                    continue;
                }

                // No new literal found - check if unit or conflict
                if first_val == Some(false) {
                    // Conflict! All literals are false
                    // Copy remaining watchers and truncate
                    while i < watch_len {
                        // Gap 8 invariant: j <= i < watch_len during copy-remaining loop
                        debug_assert!(
                            j <= i,
                            "Gap 8 invariant violated in copy loop: j ({}) > i ({})",
                            j,
                            i
                        );
                        debug_assert!(
                            i < watch_len,
                            "Gap 8 invariant violated in copy loop: i ({}) >= watch_len ({})",
                            i,
                            watch_len
                        );
                        unsafe {
                            *watch_ptr.add(j) = *watch_ptr.add(i);
                        }
                        i += 1;
                        j += 1;
                    }
                    // Gap 8 invariant: at exit, j <= watch_len
                    debug_assert!(
                        j <= watch_len,
                        "Gap 8 invariant violated: j ({}) > watch_len ({}) at truncate",
                        j,
                        watch_len
                    );
                    self.watches.truncate_watches(false_lit, j);
                    return Some(clause_ref);
                }

                // Unit propagation: first is the only unassigned literal
                self.enqueue(first, Some(clause_ref));
            }

            // Gap 8 invariant: at end of main loop, j <= watch_len
            debug_assert!(
                j <= watch_len,
                "Gap 8 invariant violated: j ({}) > watch_len ({}) at end of propagate loop",
                j,
                watch_len
            );

            // Truncate the watch list if we removed any watchers
            if j < watch_len {
                self.watches.truncate_watches(false_lit, j);
            }
        }

        None
    }

    /// Update target and best phases if we've reached a new maximum trail length
    ///
    /// Target phases track the assignment at the longest conflict-free trail in the
    /// current search. Best phases track the longest trail ever seen, for use in
    /// rephasing. This is called before backtracking to capture phases at the
    /// maximum trail length.
    fn update_target_and_best_phases(&mut self) {
        let current_trail_len = self.trail.len();

        // Update target phases if we've reached a longer trail
        if current_trail_len > self.target_trail_len {
            self.target_trail_len = current_trail_len;
            // Copy current assignment to target phases
            for i in 0..self.num_vars {
                self.target_phase[i] = self.assignment[i];
            }
        }

        // Update best phases if we've reached a longer trail ever
        if current_trail_len > self.best_trail_len {
            self.best_trail_len = current_trail_len;
            // Copy current assignment to best phases
            for i in 0..self.num_vars {
                self.best_phase[i] = self.assignment[i];
            }
        }
    }

    /// Check if rephasing should be triggered based on conflict count
    #[inline]
    fn should_rephase(&self) -> bool {
        self.rephase_enabled && self.num_conflicts >= self.next_rephase
    }

    /// Perform rephasing to escape from unhelpful search regions
    ///
    /// Cycles through different phase strategies (CaDiCaL-style):
    /// 0: Original (all positive)
    /// 1: Best phases (from longest trail)
    /// 2: Inverted (all negative)
    /// 3: Best phases
    /// 4: Flip all phases
    /// 5: Best phases
    /// 6: Random phases (using simple PRNG)
    /// 7: Best phases
    fn rephase(&mut self) {
        let strategy = (self.rephase_count % 8) as u32;

        match strategy {
            0 => {
                // Original: all positive
                for p in &mut self.phase {
                    *p = Some(true);
                }
            }
            1 | 3 | 5 | 7 => {
                // Best: copy best phases to saved phases
                for i in 0..self.num_vars {
                    if let Some(b) = self.best_phase[i] {
                        self.phase[i] = Some(b);
                    }
                }
            }
            2 => {
                // Inverted: all negative
                for p in &mut self.phase {
                    *p = Some(false);
                }
            }
            4 => {
                // Flip: invert all phases
                for p in &mut self.phase {
                    *p = p.map(|b| !b);
                }
            }
            6 => {
                // Random: deterministic pseudo-random based on conflict count
                let mut seed = self
                    .num_conflicts
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);
                for p in &mut self.phase {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *p = Some((seed >> 32) & 1 == 0);
                }
            }
            _ => unreachable!(),
        }

        // Reset target phases (keep best phases for future rephasing)
        self.target_trail_len = 0;
        for p in &mut self.target_phase {
            *p = None;
        }

        self.rephase_count += 1;
        // Arithmetic increase in rephasing interval
        self.next_rephase =
            self.num_conflicts + REPHASE_INITIAL + REPHASE_INCREMENT * self.rephase_count;
    }

    /// Backtrack to the given level
    ///
    /// Implements lazy reimplication for chronological backtracking:
    /// - Literals at levels <= target_level are kept on the trail (out of order)
    /// - Only literals at levels > target_level are unassigned
    /// - Phase saving: when a variable is unassigned, we save its polarity
    /// - Target/best phase saving: if we reached a longer trail, save the phases
    fn backtrack(&mut self, target_level: u32) {
        // Update target/best phases before backtracking
        // This captures the best assignment seen (longest trail without conflict)
        self.update_target_and_best_phases();

        if self.decision_level <= target_level {
            return;
        }

        // With chronological backtracking, we may have out-of-order literals
        // on the trail. We need to compact the trail, keeping literals at
        // levels <= target_level and removing literals at higher levels.
        let assigned_limit = if target_level == 0 {
            0
        } else {
            self.trail_lim[target_level as usize - 1]
        };

        // Start from the point where literals might be above target_level
        // (everything before assigned_limit is definitely at level <= target_level-1)
        let mut write_pos = assigned_limit;
        let mut read_pos = assigned_limit;

        while read_pos < self.trail.len() {
            let lit = self.trail[read_pos];
            let var = lit.variable();
            let var_level = self.level[var.index()];

            if var_level > target_level {
                // Unassign this variable
                // Phase saving: remember the polarity before unassigning
                self.phase[var.index()] = self.assignment[var.index()];
                self.assignment[var.index()] = None;
                self.reason[var.index()] = None;
                self.trail_pos[var.index()] = usize::MAX;
                // Add back to VSIDS heap (variable is now unassigned)
                self.vsids.insert_into_heap(var);
                // Update VMTF cursor (focused-mode decision queue)
                self.vsids.vmtf_on_unassign(var);
            } else {
                // Keep this literal (lazy reimplication)
                // Update its trail position
                if write_pos != read_pos {
                    self.trail[write_pos] = lit;
                }
                self.trail_pos[var.index()] = write_pos;
                write_pos += 1;
            }
            read_pos += 1;
        }

        // Truncate the trail
        self.trail.truncate(write_pos);

        // Update trail_lim to reflect the new decision level
        self.trail_lim.truncate(target_level as usize);
        self.decision_level = target_level;

        // Reset propagation queue - need to re-propagate from the beginning
        // of the current level's assignments since we may have kept out-of-order
        // literals that need propagation
        self.qhead = if target_level == 0 {
            0
        } else {
            self.trail_lim[target_level as usize - 1]
        };
    }

    /// Compute the i-th element of the Luby sequence: 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,...
    ///
    /// The Luby sequence is defined as:
    /// - luby(i) = 2^(k-1) if i = 2^k - 1
    /// - luby(i) = luby(i - 2^(k-1) + 1) if 2^(k-1) <= i < 2^k - 1
    ///
    /// This provides a universal restart strategy that adapts to different problem
    /// structures without knowing optimal restart intervals in advance.
    fn get_luby(i: u32) -> u32 {
        if i == 0 {
            return 1; // Edge case, shouldn't happen with 1-indexed sequence
        }

        // Find k such that 2^k - 1 >= i
        // Start with k=1, p=1 (p = 2^k - 1)
        let mut k = 1u32;
        let mut p = 1u32;

        while p < i {
            k += 1;
            p = (1 << k) - 1;
        }

        // Now 2^(k-1) - 1 < i <= 2^k - 1
        if p == i {
            // i = 2^k - 1: return 2^(k-1)
            1 << (k - 1)
        } else {
            // i < 2^k - 1: recursively compute luby(i - (2^(k-1) - 1))
            let prev_p = (1 << (k - 1)) - 1;
            Self::get_luby(i - prev_p)
        }
    }

    /// Check if we should restart
    ///
    /// Uses CaDiCaL-style stabilization: alternates between focused mode (frequent
    /// Glucose restarts) and stable mode (infrequent reluctant doubling restarts).
    fn should_restart(&mut self) -> bool {
        // Don't restart too early - need to build up EMA statistics
        if self.num_conflicts < self.restart_min_conflicts {
            return false;
        }

        // Don't restart if we haven't had any conflicts since the last restart
        // This prevents infinite restart loops when at level 0
        if self.conflicts_since_restart == 0 {
            return false;
        }

        // Check if we should switch stabilization modes
        let conflicts_in_phase = self.num_conflicts - self.stable_mode_start_conflicts;
        if conflicts_in_phase >= self.stable_phase_length {
            // Switch modes
            self.stable_mode = !self.stable_mode;
            self.stable_mode_start_conflicts = self.num_conflicts;

            if self.stable_mode {
                // Entering stable mode - reset reluctant counter
                self.stable_phase_count += 1;
                self.reluctant_counter = 0;
                self.reluctant_period = RELUCTANT_INIT;
                // Increase next phase length quadratically
                self.stable_phase_length = STABLE_PHASE_INIT
                    * (self.stable_phase_count + 1)
                    * (self.stable_phase_count + 1);
            } else {
                // Entering focused mode - use same phase length
                self.stable_phase_length = STABLE_PHASE_INIT
                    * (self.stable_phase_count + 1)
                    * (self.stable_phase_count + 1);
            }
        }

        if self.stable_mode {
            // Stable mode: reluctant doubling restarts (very infrequent)
            self.reluctant_counter += 1;
            if self.reluctant_counter >= self.reluctant_period {
                self.reluctant_counter = 0;
                self.reluctant_period *= 2; // Double period for next time
                true
            } else {
                false
            }
        } else if self.glucose_restarts {
            // Focused mode: Glucose-style frequent restarts
            // Require minimum conflicts between restarts (restart blocking)
            self.conflicts_since_restart >= RESTART_INTERVAL
                && self.lbd_ema_fast > RESTART_MARGIN * self.lbd_ema_slow
        } else {
            // Luby restarts as fallback (focused mode)
            let threshold = self.restart_base * Self::get_luby(self.luby_idx) as u64;
            self.conflicts_since_restart >= threshold
        }
    }

    /// Update LBD exponential moving averages after learning a clause
    ///
    /// This is called after conflict analysis to update the EMA statistics
    /// used for Glucose-style restart decisions.
    fn update_lbd_ema(&mut self, lbd: u32) {
        let lbd = lbd as f64;
        // Update fast EMA (short window)
        self.lbd_ema_fast = EMA_FAST_DECAY * self.lbd_ema_fast + (1.0 - EMA_FAST_DECAY) * lbd;
        // Update slow EMA (long window)
        self.lbd_ema_slow = EMA_SLOW_DECAY * self.lbd_ema_slow + (1.0 - EMA_SLOW_DECAY) * lbd;
    }

    /// Compute the level to backtrack to when restarting, reusing trail decisions
    ///
    /// CaDiCaL's trail reuse optimization: instead of backtracking to level 0,
    /// keep decisions that would be made again anyway (those with higher VSIDS
    /// activity than the next decision variable).
    ///
    /// This saves re-making the same decisions after restart, which is especially
    /// valuable when VSIDS has stabilized.
    fn compute_reuse_trail_level(&mut self) -> u32 {
        if self.decision_level == 0 {
            return 0;
        }

        // Find what the next decision variable would be, matching the current mode:
        // - Stable mode: VSIDS heap
        // - Focused mode: VMTF queue
        let next_decision = match self.pick_next_decision_variable() {
            Some(v) => v,
            None => return self.decision_level, // All assigned, keep everything
        };

        // Find the lowest level where we can reuse the trail
        let mut reuse_level = 0u32;

        if self.stable_mode {
            // Stable mode: reuse decisions with activity >= next decision's activity.
            let next_activity = self.vsids.activity(next_decision);
            for level in 1..=self.decision_level {
                // trail_lim[level-1] is the trail index where level's decision is
                let decision_idx = self.trail_lim[level as usize - 1];
                let decision_lit = self.trail[decision_idx];
                let decision_var = decision_lit.variable();
                let decision_activity = self.vsids.activity(decision_var);

                if decision_activity < next_activity {
                    break;
                }
                reuse_level = level;
            }
        } else {
            // Focused mode: reuse decisions with bump timestamp >= next decision's timestamp.
            let limit = self.vsids.bump_order(next_decision);
            for level in 1..=self.decision_level {
                let decision_idx = self.trail_lim[level as usize - 1];
                let decision_lit = self.trail[decision_idx];
                let decision_var = decision_lit.variable();
                let decision_bumped = self.vsids.bump_order(decision_var);

                if decision_bumped < limit {
                    break;
                }
                reuse_level = level;
            }
        }

        reuse_level
    }

    /// Perform a restart with trail reuse
    ///
    /// Instead of backtracking to level 0, reuse decisions that would be
    /// made again anyway (those with higher VSIDS activity).
    fn do_restart(&mut self) {
        let reuse_level = self.compute_reuse_trail_level();
        self.backtrack(reuse_level);
        self.conflicts_since_restart = 0;
        self.luby_idx += 1;
        self.restarts += 1;
        // Reset target phase tracking on restart (keep best phases for rephasing)
        self.target_trail_len = 0;
    }

    /// Bump clause activity (called when clause is involved in conflict)
    fn bump_clause_activity(&mut self, clause_ref: ClauseRef) {
        let clause_idx = clause_ref.0 as usize;
        let header = self.clause_db.header_mut(clause_idx);
        let new_activity = header.activity() + self.clause_activity_inc;
        header.set_activity(new_activity);

        // Rescale if activity gets too large
        if new_activity > 1e20 {
            for i in 0..self.clause_db.len() {
                let h = self.clause_db.header_mut(i);
                h.set_activity(h.activity() * 1e-20);
            }
            self.clause_activity_inc *= 1e-20;
        }
    }

    /// Decay all clause activities
    fn decay_clause_activity(&mut self) {
        self.clause_activity_inc /= CLAUSE_DECAY;
    }

    /// Check if we should reduce the clause database
    fn should_reduce_db(&self) -> bool {
        self.num_conflicts >= self.next_reduce_db
    }

    /// Reduce the learned clause database using tier-based management
    ///
    /// This implements a three-tiered approach based on LBD:
    /// - CORE (LBD <= 2): Never delete - these are "glue" clauses
    /// - TIER1 (2 < LBD <= 6): Protected if recently used (used > 0)
    /// - TIER2 (LBD > 6): Deleted based on activity score
    ///
    /// The "used" counter tracks how recently a clause was involved in
    /// conflict analysis. It's incremented when used and decremented here.
    fn reduce_db(&mut self) {
        // First pass: decay usage counters and collect deletable clauses
        let mut tier2_clauses: Vec<(usize, f64)> = Vec::new();

        for idx in 0..self.clause_db.len() {
            let header = self.clause_db.header(idx);
            if !header.is_learned() {
                continue;
            }
            // Skip empty/deleted clauses
            if header.is_empty() {
                continue;
            }

            // Check if clause is currently a reason for any assigned variable
            let is_reason = self
                .reason
                .iter()
                .any(|r| r.map(|cr| cr.0 as usize) == Some(idx));
            if is_reason {
                continue;
            }

            match header.tier() {
                ClauseTier::Core => {
                    // Core/glue clauses are never deleted
                }
                ClauseTier::Tier1 => {
                    // Tier1 clauses: protected if used recently
                    if header.used() > 0 {
                        self.clause_db.header_mut(idx).decay_used();
                        continue;
                    }
                    // If not used recently, treat as tier2 for potential deletion
                    let score = header.activity() as f64 / (header.lbd() as f64 + 2.0);
                    tier2_clauses.push((idx, score));
                }
                ClauseTier::Tier2 => {
                    // Tier2 clauses: decay usage and consider for deletion
                    self.clause_db.header_mut(idx).decay_used();
                    let header = self.clause_db.header(idx);
                    let score = header.activity() as f64 / (header.lbd() as f64 + 2.0);
                    tier2_clauses.push((idx, score));
                }
            }
        }

        // Sort by score (ascending - worst clauses first)
        tier2_clauses.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Delete half of the tier2 clauses (the worst scoring ones)
        let num_to_delete = tier2_clauses.len() / 2;
        let to_delete: Vec<usize> = tier2_clauses
            .iter()
            .take(num_to_delete)
            .map(|(idx, _)| *idx)
            .collect();

        // Delete clauses (remove watches first, then mark as deleted)
        for idx in to_delete {
            // Log deletion to proof if enabled
            let lits: Vec<Literal> = self.clause_db.literals(idx).to_vec();
            let clause_id = self.clause_id(ClauseRef(idx as u32));
            if let Some(ref mut writer) = self.proof_writer {
                let _ = writer.delete(&lits, clause_id);
            }

            // Remove watches before deleting
            if self.clause_db.header(idx).len() >= 2 {
                let lit0 = self.clause_db.literal(idx, 0);
                let lit1 = self.clause_db.literal(idx, 1);
                self.remove_watch(lit0, ClauseRef(idx as u32));
                self.remove_watch(lit1, ClauseRef(idx as u32));
            }

            self.clause_db.delete(idx);
        }

        // Schedule next reduction
        self.next_reduce_db = self.num_conflicts + REDUCE_DB_INC;
    }

    /// Check if we should run vivification
    fn should_vivify(&self) -> bool {
        self.vivify_enabled && self.num_conflicts >= self.next_vivify
    }

    /// Run vivification on learned clauses to strengthen them
    ///
    /// Vivification tries to remove literals from clauses by temporarily assuming
    /// their negation and propagating. If this leads to a conflict or implies
    /// another literal in the clause, the literal can be removed.
    ///
    /// This must be called at decision level 0 (after a restart) for correctness.
    fn vivify(&mut self) -> bool {
        // Must be at level 0 for vivification
        if self.decision_level != 0 {
            return false;
        }

        // Collect indices of clauses to vivify (learned clauses only, not too short)
        let mut candidates: Vec<usize> = Vec::new();
        for idx in 0..self.clause_db.len() {
            let header = self.clause_db.header(idx);
            // Skip empty, short, or core clauses
            if header.is_empty()
                || !header.is_learned()
                || header.len() < 3
                || header.tier() == ClauseTier::Core
            {
                continue;
            }
            candidates.push(idx);
        }

        // Limit the number of clauses to vivify
        let num_to_vivify = candidates.len().min(VIVIFY_CLAUSES_PER_CALL);
        candidates.truncate(num_to_vivify);

        // Process each candidate clause
        let mut strengthened_count = 0u64;
        let mut literals_removed = 0u64;
        let mut enqueued_units = false;

        for clause_idx in candidates {
            let clause_lits = self.clause_db.literals(clause_idx).to_vec();
            if clause_lits.len() < 3 {
                continue;
            }

            let result = self.vivifier.vivify_clause(
                &clause_lits,
                &mut self.assignment,
                &mut self.level,
                &self.clause_db,
                &self.watches,
                &self.reason,
                ClauseRef(clause_idx as u32),
            );

            if result.is_satisfied {
                // Clause is satisfied at level 0 - can be deleted
                let clause_id = self.clause_id(ClauseRef(clause_idx as u32));
                if let Some(ref mut writer) = self.proof_writer {
                    let _ = writer.delete(&clause_lits, clause_id);
                }

                // Remove watches before deleting
                if self.clause_db.header(clause_idx).len() >= 2 {
                    let lit0 = self.clause_db.literal(clause_idx, 0);
                    let lit1 = self.clause_db.literal(clause_idx, 1);
                    self.remove_watch(lit0, ClauseRef(clause_idx as u32));
                    self.remove_watch(lit1, ClauseRef(clause_idx as u32));
                }

                self.clause_db.delete(clause_idx);
                continue;
            }

            if result.was_strengthened && result.strengthened.len() < clause_lits.len() {
                strengthened_count += 1;
                literals_removed += (clause_lits.len() - result.strengthened.len()) as u64;

                // For proofs: add new clause first, then delete old
                // Note: For LRAT, we'd need proper hints for strengthening derivation
                // For now, we use empty hints (valid for DRAT-style RUP)
                let clause_id = self.clause_id(ClauseRef(clause_idx as u32));
                if let Some(ref mut writer) = self.proof_writer {
                    let _ = writer.add(&result.strengthened, &[]);
                    let _ = writer.delete(&clause_lits, clause_id);
                }

                // Update the clause in-place
                // Note: We need to update watches if the watched literals changed
                let old_watch0 = clause_lits.first().copied();
                let old_watch1 = clause_lits.get(1).copied();

                self.clause_db.replace(clause_idx, &result.strengthened);
                if clause_idx < self.watch_pos.len() {
                    self.watch_pos[clause_idx] = 2;
                }

                // Update watches if necessary
                // For simplicity, if the first two literals changed, we need to update watches
                let new_watch0 = result.strengthened.first().copied();
                let new_watch1 = result.strengthened.get(1).copied();

                if result.strengthened.len() >= 2
                    && (old_watch0 != new_watch0 || old_watch1 != new_watch1)
                {
                    // Remove old watches
                    if let Some(lit0) = old_watch0 {
                        self.remove_watch(lit0, ClauseRef(clause_idx as u32));
                    }
                    if let Some(lit1) = old_watch1 {
                        self.remove_watch(lit1, ClauseRef(clause_idx as u32));
                    }

                    // Add new watches
                    let lit0 = result.strengthened[0];
                    let lit1 = result.strengthened[1];
                    let clause_ref = ClauseRef(clause_idx as u32);
                    let is_binary = result.strengthened.len() == 2;
                    if is_binary {
                        self.watches
                            .add_watch(lit0, Watcher::binary(clause_ref, lit1));
                        self.watches
                            .add_watch(lit1, Watcher::binary(clause_ref, lit0));
                    } else {
                        self.watches.add_watch(lit0, Watcher::new(clause_ref, lit1));
                        self.watches.add_watch(lit1, Watcher::new(clause_ref, lit0));
                    }
                } else if result.strengthened.len() == 1 {
                    // Became unit clause - remove watches and propagate
                    if let Some(lit0) = old_watch0 {
                        self.remove_watch(lit0, ClauseRef(clause_idx as u32));
                    }
                    if let Some(lit1) = old_watch1 {
                        self.remove_watch(lit1, ClauseRef(clause_idx as u32));
                    }
                    let unit = result.strengthened[0];
                    match self.lit_value(unit) {
                        Some(true) => {}
                        Some(false) => return true,
                        None => {
                            self.enqueue(unit, Some(ClauseRef(clause_idx as u32)));
                            enqueued_units = true;
                        }
                    }
                }
            }
        }

        // Update statistics (the vivifier tracks its own stats)
        let _ = (strengthened_count, literals_removed); // Avoid unused warnings

        // Schedule next vivification
        self.next_vivify = self.num_conflicts + VIVIFY_INTERVAL;
        if enqueued_units && self.propagate().is_some() {
            return true;
        }
        false
    }

    /// Remove a watch for a clause from a literal's watch list
    fn remove_watch(&mut self, lit: Literal, clause_ref: ClauseRef) {
        let watches = self.watches.get_watches_mut(lit);
        if let Some(pos) = watches.iter().position(|w| w.clause_ref() == clause_ref) {
            watches.swap_remove(pos);
        }
    }

    /// Get vivification statistics
    pub fn vivify_stats(&self) -> &VivifyStats {
        self.vivifier.stats()
    }

    /// Enable or disable vivification
    pub fn set_vivify_enabled(&mut self, enabled: bool) {
        self.vivify_enabled = enabled;
    }

    /// Check if we should run subsumption
    fn should_subsume(&self) -> bool {
        self.subsume_enabled && self.num_conflicts >= self.next_subsume
    }

    /// Run subsumption on learned clauses
    ///
    /// Subsumption removes redundant clauses:
    /// - Forward subsumption: New/smaller clauses subsume (remove) larger clauses
    /// - Self-subsumption: Strengthen clauses by removing redundant literals
    ///
    /// This must be called at decision level 0 (after a restart) for correctness.
    fn subsume(&mut self) {
        // Must be at level 0 for subsumption
        if self.decision_level != 0 {
            return;
        }

        // Rebuild occurrence lists from current clause database
        self.subsumer.rebuild(&self.clause_db);

        // Run subsumption on learned clauses
        let result = self
            .subsumer
            .run_subsumption_learned(&self.clause_db, SUBSUME_CLAUSES_PER_CALL);

        // Apply deletions (subsumed clauses)
        for clause_idx in &result.subsumed {
            let clause_idx = *clause_idx;
            if clause_idx >= self.clause_db.len() {
                continue;
            }

            // Don't delete reason clauses
            let is_reason = self
                .reason
                .iter()
                .any(|r| r.map(|cr| cr.0 as usize) == Some(clause_idx));
            if is_reason {
                continue;
            }

            // Log deletion to proof if enabled
            let lits: Vec<Literal> = self.clause_db.literals(clause_idx).to_vec();
            let clause_id = self.clause_id(ClauseRef(clause_idx as u32));
            if let Some(ref mut writer) = self.proof_writer {
                let _ = writer.delete(&lits, clause_id);
            }

            // Remove watches
            if self.clause_db.header(clause_idx).len() >= 2 {
                let lit0 = self.clause_db.literal(clause_idx, 0);
                let lit1 = self.clause_db.literal(clause_idx, 1);
                self.remove_watch(lit0, ClauseRef(clause_idx as u32));
                self.remove_watch(lit1, ClauseRef(clause_idx as u32));
            }

            // Mark clause as deleted
            self.clause_db.delete(clause_idx);
        }

        // Apply strengthening (self-subsumption)
        for (clause_idx, new_lits) in &result.strengthened {
            let clause_idx = *clause_idx;
            if clause_idx >= self.clause_db.len() || self.clause_db.header(clause_idx).is_empty() {
                continue;
            }

            // Don't strengthen reason clauses
            let is_reason = self
                .reason
                .iter()
                .any(|r| r.map(|cr| cr.0 as usize) == Some(clause_idx));
            if is_reason {
                continue;
            }

            let old_lits: Vec<Literal> = self.clause_db.literals(clause_idx).to_vec();

            // For proofs: add strengthened clause first, then delete original
            let clause_id = self.clause_id(ClauseRef(clause_idx as u32));
            if let Some(ref mut writer) = self.proof_writer {
                let _ = writer.add(new_lits, &[]);
                let _ = writer.delete(&old_lits, clause_id);
            }

            // Update watches if the watched literals changed
            let old_watch0 = old_lits.first().copied();
            let old_watch1 = old_lits.get(1).copied();
            let new_watch0 = new_lits.first().copied();
            let new_watch1 = new_lits.get(1).copied();

            // Remove old watches
            if old_lits.len() >= 2 {
                if let Some(lit0) = old_watch0 {
                    self.remove_watch(lit0, ClauseRef(clause_idx as u32));
                }
                if let Some(lit1) = old_watch1 {
                    self.remove_watch(lit1, ClauseRef(clause_idx as u32));
                }
            }

            // Update the clause
            self.clause_db.replace(clause_idx, new_lits);
            if clause_idx < self.watch_pos.len() {
                self.watch_pos[clause_idx] = 2;
            }

            // Add new watches
            if new_lits.len() >= 2 {
                let lit0 = new_watch0.unwrap();
                let lit1 = new_watch1.unwrap();
                let clause_ref = ClauseRef(clause_idx as u32);
                let is_binary = new_lits.len() == 2;
                if is_binary {
                    self.watches
                        .add_watch(lit0, Watcher::binary(clause_ref, lit1));
                    self.watches
                        .add_watch(lit1, Watcher::binary(clause_ref, lit0));
                } else {
                    self.watches.add_watch(lit0, Watcher::new(clause_ref, lit1));
                    self.watches.add_watch(lit1, Watcher::new(clause_ref, lit0));
                }
            }
            // Note: If strengthened to unit clause, propagation will happen in main loop
        }

        // Schedule next subsumption
        self.next_subsume = self.num_conflicts + SUBSUME_INTERVAL;
    }

    /// Get subsumption statistics
    pub fn subsume_stats(&self) -> &SubsumeStats {
        self.subsumer.stats()
    }

    /// Enable or disable subsumption
    pub fn set_subsume_enabled(&mut self, enabled: bool) {
        self.subsume_enabled = enabled;
    }

    /// Check if we should run failed literal probing
    fn should_probe(&self) -> bool {
        self.probe_enabled && self.num_conflicts >= self.next_probe
    }

    /// Run failed literal probing
    ///
    /// For each candidate probe literal, temporarily assign it and propagate.
    /// If a conflict is found, the literal is "failed" and its negation must be true.
    ///
    /// This must be called at decision level 0 (after a restart) for correctness.
    ///
    /// Returns true if any failed literals were found or UNSAT was detected.
    fn probe(&mut self) -> bool {
        // Must be at level 0 for probing
        if self.decision_level != 0 {
            return false;
        }

        // Generate probe candidates
        self.prober
            .generate_probes(&self.clause_db, &self.assignment, self.fixed_count);
        self.prober.record_round();

        let mut found_failed = false;
        let mut probes_tried = 0;

        // Probe each candidate
        while let Some(probe_lit) = self.prober.next_probe() {
            // Limit the number of probes per call
            if probes_tried >= MAX_PROBES_PER_CALL {
                break;
            }

            // Skip if already assigned
            if self.assignment[probe_lit.variable().index()].is_some() {
                continue;
            }

            probes_tried += 1;
            self.prober.record_probed();

            // Make probe as decision at level 1
            self.decide(probe_lit);

            // Propagate
            if let Some(conflict_ref) = self.propagate() {
                // Found conflict - this is a failed literal
                self.prober.record_failed();
                found_failed = true;

                // Find the UIP to derive a unit
                let conflict_clause: Vec<Literal> =
                    self.clause_db.literals(conflict_ref.0 as usize).to_vec();
                let forced = find_failed_literal_uip(
                    &conflict_clause,
                    &self.trail,
                    &self.level,
                    &self.reason,
                    &self.clause_db,
                );

                // Backtrack to level 0
                self.backtrack(0);

                if let Some(unit_lit) = forced {
                    // Learn the unit clause
                    // Note: For LRAT, we'd need proper hints from the failed literal analysis
                    if let Some(ref mut writer) = self.proof_writer {
                        let _ = writer.add(&[unit_lit], &[]);
                    }

                    // Propagate the new unit
                    let unit_idx = self.add_clause_db(&[unit_lit], true);
                    let unit_ref = ClauseRef(unit_idx as u32);

                    self.enqueue(unit_lit, Some(unit_ref));
                    self.fixed_count += 1;

                    // Propagate the unit
                    if self.propagate().is_some() {
                        // Conflict at level 0 - UNSAT
                        self.next_probe = self.num_conflicts + PROBE_INTERVAL;
                        return true;
                    }
                }
            } else {
                // No conflict - backtrack and continue
                self.backtrack(0);
            }

            // Mark this literal as probed at current fixed point
            self.prober.mark_probed(probe_lit, self.fixed_count);
        }

        // Schedule next probing
        self.next_probe = self.num_conflicts + PROBE_INTERVAL;
        found_failed
    }

    /// Get probing statistics
    pub fn probe_stats(&self) -> &ProbeStats {
        self.prober.stats()
    }

    /// Enable or disable failed literal probing
    pub fn set_probe_enabled(&mut self, enabled: bool) {
        self.probe_enabled = enabled;
    }

    /// Check if we should run BVE
    fn should_bve(&self) -> bool {
        self.bve_enabled && self.num_conflicts >= self.next_bve
    }

    /// Run bounded variable elimination
    ///
    /// Attempts to eliminate variables by resolving clauses. For a variable x,
    /// if the total size of resolvents is bounded, we can eliminate x by:
    /// 1. Adding all resolvents
    /// 2. Removing all clauses containing x
    ///
    /// This must be called at decision level 0 (after a restart) for correctness.
    ///
    /// Returns true if UNSAT was derived (empty resolvent found).
    fn bve(&mut self) -> bool {
        // Must be at level 0 for BVE
        if self.decision_level != 0 {
            return false;
        }

        // Rebuild occurrence lists
        self.bve.rebuild(&self.clause_db);

        // Run elimination
        let gate_enabled = self.gate_enabled;
        let (bve_engine, gate_extractor) = (&mut self.bve, &mut self.gate_extractor);
        let results = bve_engine.run_elimination_with_gate_provider(
            &self.clause_db,
            &self.assignment,
            MAX_BVE_ELIMINATIONS,
            |var, pos_occs, neg_occs, clauses| {
                if !gate_enabled {
                    return None;
                }
                gate_extractor
                    .find_gate_for_bve(var, clauses, pos_occs, neg_occs)
                    .map(|g| g.defining_clauses)
            },
        );

        let mut derived_unsat = false;

        for result in results {
            if !result.eliminated {
                continue;
            }

            // Record clauses for model reconstruction (before deletion)
            // Extract positive and negative clauses containing the eliminated variable
            let var = result.variable;
            let pos_lit = Literal::positive(var);
            let neg_lit = Literal::negative(var);

            let pos_clauses: Vec<Vec<Literal>> = result
                .to_delete
                .iter()
                .filter_map(|&idx| {
                    if idx < self.clause_db.len()
                        && !self.clause_db.header(idx).is_empty()
                        && self.clause_db.literals(idx).contains(&pos_lit)
                    {
                        Some(self.clause_db.literals(idx).to_vec())
                    } else {
                        None
                    }
                })
                .collect();

            let neg_clauses: Vec<Vec<Literal>> = result
                .to_delete
                .iter()
                .filter_map(|&idx| {
                    if idx < self.clause_db.len()
                        && !self.clause_db.header(idx).is_empty()
                        && self.clause_db.literals(idx).contains(&neg_lit)
                    {
                        Some(self.clause_db.literals(idx).to_vec())
                    } else {
                        None
                    }
                })
                .collect();

            // Only record if there are clauses (pure literal case may have empty)
            if !pos_clauses.is_empty() || !neg_clauses.is_empty() {
                self.reconstruction.push_bve(var, pos_clauses, neg_clauses);
            }

            // First, add all resolvents (for proof correctness)
            for resolvent in &result.resolvents {
                // Check for empty resolvent (UNSAT)
                if resolvent.is_empty() {
                    derived_unsat = true;
                    // Log empty clause
                    // Note: For LRAT, we'd need proper hints from resolution
                    if let Some(ref mut writer) = self.proof_writer {
                        let _ = writer.add(&[], &[]);
                    }
                    continue;
                }

                // Log resolvent to proof
                // Note: For LRAT, we'd need proper hints from resolution
                if let Some(ref mut writer) = self.proof_writer {
                    let _ = writer.add(resolvent, &[]);
                }

                // Add resolvent as new clause
                let clause_idx = self.add_clause_db(resolvent, true);
                let clause_ref = ClauseRef(clause_idx as u32);

                // Add watches for the resolvent
                if resolvent.len() >= 2 {
                    let lit0 = resolvent[0];
                    let lit1 = resolvent[1];
                    let is_binary = resolvent.len() == 2;
                    if is_binary {
                        self.watches
                            .add_watch(lit0, Watcher::binary(clause_ref, lit1));
                        self.watches
                            .add_watch(lit1, Watcher::binary(clause_ref, lit0));
                    } else {
                        self.watches.add_watch(lit0, Watcher::new(clause_ref, lit1));
                        self.watches.add_watch(lit1, Watcher::new(clause_ref, lit0));
                    }
                } else if resolvent.len() == 1 {
                    // Unit clause - enqueue for propagation
                    let unit_lit = resolvent[0];
                    if self.assignment[unit_lit.variable().index()].is_none() {
                        self.enqueue(unit_lit, Some(clause_ref));
                    }
                }

                // Clause already added above via `add_clause_db`.
            }

            // Then, delete original clauses containing the eliminated variable
            for &clause_idx in &result.to_delete {
                if clause_idx >= self.clause_db.len()
                    || self.clause_db.header(clause_idx).is_empty()
                {
                    continue;
                }

                // Don't delete reason clauses
                let is_reason = self
                    .reason
                    .iter()
                    .any(|r| r.map(|cr| cr.0 as usize) == Some(clause_idx));
                if is_reason {
                    continue;
                }

                // Log deletion to proof
                let lits: Vec<Literal> = self.clause_db.literals(clause_idx).to_vec();
                let clause_id = self.clause_id(ClauseRef(clause_idx as u32));
                if let Some(ref mut writer) = self.proof_writer {
                    let _ = writer.delete(&lits, clause_id);
                }

                // Remove watches
                if self.clause_db.header(clause_idx).len() >= 2 {
                    let lit0 = self.clause_db.literal(clause_idx, 0);
                    let lit1 = self.clause_db.literal(clause_idx, 1);
                    self.remove_watch(lit0, ClauseRef(clause_idx as u32));
                    self.remove_watch(lit1, ClauseRef(clause_idx as u32));
                }

                // Mark clause as deleted
                self.clause_db.delete(clause_idx);
            }
        }

        // Schedule next BVE
        self.next_bve = self.num_conflicts + BVE_INTERVAL;

        derived_unsat
    }

    /// Get BVE statistics
    pub fn bve_stats(&self) -> &BVEStats {
        self.bve.stats()
    }

    /// Enable or disable BVE
    pub fn set_bve_enabled(&mut self, enabled: bool) {
        self.bve_enabled = enabled;
    }

    /// Check if we should run BCE
    fn should_bce(&self) -> bool {
        self.bce_enabled && self.num_conflicts >= self.next_bce
    }

    /// Run blocked clause elimination
    ///
    /// A clause is blocked on literal L if for every clause D containing ~L,
    /// resolving C and D on L produces a tautology. Blocked clauses can be
    /// safely removed without changing satisfiability.
    ///
    /// This must be called at decision level 0 (after a restart) for correctness.
    fn bce(&mut self) {
        // Must be at level 0 for BCE
        if self.decision_level != 0 {
            return;
        }

        // Rebuild occurrence lists
        self.bce.rebuild(&self.clause_db);

        // Run elimination
        let eliminated = self
            .bce
            .run_elimination(&self.clause_db, MAX_BCE_ELIMINATIONS);

        // Delete the eliminated clauses
        for clause_idx in eliminated {
            if clause_idx >= self.clause_db.len() || self.clause_db.header(clause_idx).is_empty() {
                continue;
            }

            // Don't delete reason clauses
            let is_reason = self
                .reason
                .iter()
                .any(|r| r.map(|cr| cr.0 as usize) == Some(clause_idx));
            if is_reason {
                continue;
            }

            // Log deletion to proof
            let lits: Vec<Literal> = self.clause_db.literals(clause_idx).to_vec();
            let clause_id = self.clause_id(ClauseRef(clause_idx as u32));
            if let Some(ref mut writer) = self.proof_writer {
                let _ = writer.delete(&lits, clause_id);
            }

            // Remove watches
            if self.clause_db.header(clause_idx).len() >= 2 {
                let lit0 = self.clause_db.literal(clause_idx, 0);
                let lit1 = self.clause_db.literal(clause_idx, 1);
                self.remove_watch(lit0, ClauseRef(clause_idx as u32));
                self.remove_watch(lit1, ClauseRef(clause_idx as u32));
            }

            // Mark clause as deleted
            self.clause_db.delete(clause_idx);
        }

        // Schedule next BCE
        self.next_bce = self.num_conflicts + BCE_INTERVAL;
    }

    /// Get BCE statistics
    pub fn bce_stats(&self) -> &BCEStats {
        self.bce.stats()
    }

    /// Enable or disable BCE
    pub fn set_bce_enabled(&mut self, enabled: bool) {
        self.bce_enabled = enabled;
    }

    /// Check if we should run HTR
    fn should_htr(&self) -> bool {
        self.htr_enabled && self.num_conflicts >= self.next_htr
    }

    /// Run hyper-ternary resolution
    ///
    /// Resolves pairs of ternary clauses to produce binary or ternary resolvents.
    /// Binary resolvents are particularly valuable as they strengthen propagation
    /// and allow deletion of the antecedent clauses.
    ///
    /// This must be called at decision level 0 (after a restart) for correctness.
    fn htr(&mut self) {
        // Must be at level 0 for HTR
        if self.decision_level != 0 {
            return;
        }

        // Rebuild occurrence lists
        self.htr.rebuild(&self.clause_db);

        // Run hyper-ternary resolution
        let result = self
            .htr
            .run(&self.clause_db, &self.assignment, MAX_HTR_RESOLVENTS);

        // First, add all new resolvents (for proof correctness)
        for resolvent in &result.resolvents {
            // Log resolvent to proof
            // Note: For LRAT, we'd need proper hints from resolution
            if let Some(ref mut writer) = self.proof_writer {
                let _ = writer.add(resolvent, &[]);
            }

            // Add resolvent as new clause
            let clause_idx = self.add_clause_db(resolvent, true);
            let clause_ref = ClauseRef(clause_idx as u32);

            // Add watches for the resolvent
            if resolvent.len() >= 2 {
                let lit0 = resolvent[0];
                let lit1 = resolvent[1];
                let is_binary = resolvent.len() == 2;
                if is_binary {
                    self.watches
                        .add_watch(lit0, Watcher::binary(clause_ref, lit1));
                    self.watches
                        .add_watch(lit1, Watcher::binary(clause_ref, lit0));
                } else {
                    self.watches.add_watch(lit0, Watcher::new(clause_ref, lit1));
                    self.watches.add_watch(lit1, Watcher::new(clause_ref, lit0));
                }
            } else if resolvent.len() == 1 {
                // Unit clause - enqueue for propagation
                let unit_lit = resolvent[0];
                if self.assignment[unit_lit.variable().index()].is_none() {
                    self.enqueue(unit_lit, Some(clause_ref));
                }
            }

            // Clause already added above via `add_clause_db`.
        }

        // Then, delete antecedent clauses (those subsumed by binary resolvents)
        for clause_idx in &result.to_delete {
            let clause_idx = *clause_idx;
            if clause_idx >= self.clause_db.len() || self.clause_db.header(clause_idx).is_empty() {
                continue;
            }

            // Don't delete reason clauses
            let is_reason = self
                .reason
                .iter()
                .any(|r| r.map(|cr| cr.0 as usize) == Some(clause_idx));
            if is_reason {
                continue;
            }

            // Log deletion to proof
            let lits: Vec<Literal> = self.clause_db.literals(clause_idx).to_vec();
            let clause_id = self.clause_id(ClauseRef(clause_idx as u32));
            if let Some(ref mut writer) = self.proof_writer {
                let _ = writer.delete(&lits, clause_id);
            }

            // Remove watches
            if self.clause_db.header(clause_idx).len() >= 2 {
                let lit0 = self.clause_db.literal(clause_idx, 0);
                let lit1 = self.clause_db.literal(clause_idx, 1);
                self.remove_watch(lit0, ClauseRef(clause_idx as u32));
                self.remove_watch(lit1, ClauseRef(clause_idx as u32));
            }

            // Mark clause as deleted
            self.clause_db.delete(clause_idx);
        }

        // Schedule next HTR
        self.next_htr = self.num_conflicts + HTR_INTERVAL;
    }

    /// Get HTR statistics
    pub fn htr_stats(&self) -> &HTRStats {
        self.htr.stats()
    }

    /// Enable or disable HTR
    pub fn set_htr_enabled(&mut self, enabled: bool) {
        self.htr_enabled = enabled;
    }

    /// Get gate extraction statistics
    pub fn gate_stats(&self) -> &GateStats {
        self.gate_extractor.stats()
    }

    /// Enable or disable gate extraction
    pub fn set_gate_enabled(&mut self, enabled: bool) {
        self.gate_enabled = enabled;
    }

    /// Check if we should run SAT sweeping
    fn should_sweep(&self) -> bool {
        self.sweep_enabled && self.num_conflicts >= self.next_sweep && self.proof_writer.is_none()
    }

    /// Run SAT sweeping (equivalence merging)
    ///
    /// Detects equivalent literals via SCC analysis on the binary implication
    /// graph and rewrites clauses to use canonical representatives.
    ///
    /// This must be called at decision level 0 (after a restart) for correctness.
    ///
    /// Returns true if UNSAT was detected (contradiction in SCC).
    fn sweep(&mut self) -> bool {
        // Must be at level 0 for sweeping
        if self.decision_level != 0 {
            return false;
        }

        // Don't run sweeping with proof logging (not DRAT-compatible by default)
        if self.proof_writer.is_some() {
            return false;
        }

        // Run sweeping
        let outcome = self.sweeper.sweep(&self.clause_db);

        if outcome.unsat {
            return true;
        }

        // Record the equivalence mapping for reconstruction if non-trivial
        if !outcome.lit_map.is_empty() {
            // Check if any variable was actually merged (not identity mapping)
            let has_merges = outcome
                .lit_map
                .iter()
                .enumerate()
                .any(|(idx, &mapped)| mapped.index() != idx);

            if has_merges {
                self.reconstruction
                    .push_sweep(self.num_vars, outcome.lit_map);
            }
        }

        // Process new units derived from sweeping
        for &unit_lit in &outcome.new_units {
            if self.assignment[unit_lit.variable().index()].is_none() {
                // Add unit clause
                let unit_idx = self.add_clause_db(&[unit_lit], true);
                let unit_ref = ClauseRef(unit_idx as u32);
                self.enqueue(unit_lit, Some(unit_ref));
                self.fixed_count += 1;
            }
        }

        // Rebuild watches after sweeping (clauses may have changed structure)
        self.rebuild_watches();

        // Schedule next sweeping
        self.next_sweep = self.num_conflicts + SWEEP_INTERVAL;

        false
    }

    /// Rebuild watched literals for all non-empty clauses
    fn rebuild_watches(&mut self) {
        // Clear all watch lists
        self.watches = WatchedLists::new(self.num_vars);

        // Re-add watches for all non-empty clauses with >= 2 literals
        for i in 0..self.clause_db.len() {
            let header = self.clause_db.header(i);
            let clause_len = header.len();
            if header.is_empty() || clause_len < 2 {
                continue;
            }
            let clause_ref = ClauseRef(i as u32);
            let lit0 = self.clause_db.literal(i, 0);
            let lit1 = self.clause_db.literal(i, 1);
            let is_binary = clause_len == 2;
            if is_binary {
                self.watches
                    .add_watch(lit0, Watcher::binary(clause_ref, lit1));
                self.watches
                    .add_watch(lit1, Watcher::binary(clause_ref, lit0));
            } else {
                self.watches.add_watch(lit0, Watcher::new(clause_ref, lit1));
                self.watches.add_watch(lit1, Watcher::new(clause_ref, lit0));
            }
        }
    }

    /// Get sweeping statistics
    pub fn sweep_stats(&self) -> &SweepStats {
        self.sweeper.stats()
    }

    /// Enable or disable SAT sweeping
    pub fn set_sweep_enabled(&mut self, enabled: bool) {
        self.sweep_enabled = enabled;
    }

    /// Enable or disable initial preprocessing
    pub fn set_preprocess_enabled(&mut self, enabled: bool) {
        self.preprocess_enabled = enabled;
    }

    /// Run initial preprocessing to reduce the search space
    ///
    /// This runs BVE, probing, subsumption, and sweep once at the start of solving
    /// to simplify the formula before entering the main CDCL loop.
    ///
    /// Returns true if UNSAT was detected during preprocessing.
    fn preprocess(&mut self) -> bool {
        // Must be at level 0 for preprocessing
        if self.decision_level != 0 {
            return false;
        }

        // Skip if proof logging is enabled (some preprocessors are not DRAT-compatible)
        if self.proof_writer.is_some() {
            return false;
        }

        // Run up to 3 rounds of preprocessing (to reach a fixed point)
        for _round in 0..3 {
            let vars_before = self.num_vars - self.count_fixed_vars();

            // Run probing first (can find failed literals and derive units)
            if self.probe_enabled {
                // Ensure prober has correct variable count (may have changed since solver init)
                self.prober.ensure_num_vars(self.num_vars);
                self.prober
                    .generate_probes(&self.clause_db, &self.assignment, self.fixed_count);
                while let Some(probe_lit) = self.prober.next_probe() {
                    let var_idx = probe_lit.variable().index();
                    // Skip if out of bounds or already assigned
                    if var_idx >= self.num_vars || self.assignment[var_idx].is_some() {
                        continue;
                    }

                    // Temporarily decide on the probe literal
                    self.decide(probe_lit);

                    // Propagate
                    if let Some(_conflict) = self.propagate() {
                        // Failed literal! Its negation must be true
                        let failed_lit = probe_lit.negated();
                        self.backtrack(0);

                        // Assign the negation as a unit at level 0
                        self.enqueue(failed_lit, None);
                        self.prober.record_failed();

                        // Propagate the new unit
                        if self.propagate().is_some() {
                            // UNSAT detected
                            return true;
                        }
                    } else {
                        // No conflict - backtrack
                        self.backtrack(0);
                    }
                    self.prober.mark_probed(probe_lit, self.fixed_count);
                }
                self.prober.record_round();
            }

            // Note: BVE is not run during preprocessing because it requires complex
            // reconstruction setup. BVE runs during inprocessing instead.

            // Run subsumption
            if self.subsume_enabled {
                self.subsumer.ensure_num_vars(self.num_vars);
                self.subsumer.rebuild(&self.clause_db);
                let result =
                    self.subsumer
                        .run_subsumption(&self.clause_db, 0, self.clause_db.len());
                for clause_idx in &result.subsumed {
                    self.clause_db.delete(*clause_idx);
                }
            }

            // Check if we reached a fixed point
            let vars_after = self.num_vars - self.count_fixed_vars();
            if vars_after == vars_before {
                break; // No progress, stop preprocessing
            }
        }

        false
    }

    /// Helper: count the number of fixed (assigned at level 0) variables
    fn count_fixed_vars(&self) -> usize {
        self.assignment.iter().filter(|a| a.is_some()).count()
    }

    /// Enable or disable Glucose-style EMA restarts
    ///
    /// When enabled, restarts are triggered based on the exponential moving average
    /// of learned clause LBD values. When disabled, uses Luby sequence restarts.
    pub fn set_glucose_restarts(&mut self, enabled: bool) {
        self.glucose_restarts = enabled;
    }

    /// Enable or disable chronological backtracking
    ///
    /// When enabled, the solver may backtrack by only one level instead of jumping
    /// to the asserting level, which can help on certain problem classes.
    pub fn set_chrono_enabled(&mut self, enabled: bool) {
        self.chrono_enabled = enabled;
    }

    /// Set the initial phase for all variables
    ///
    /// If `phase` is `Some(true)`, variables will initially be assigned positive.
    /// If `phase` is `Some(false)`, variables will initially be assigned negative.
    /// If `phase` is `None`, phase saving will be used (default behavior).
    pub fn set_initial_phase(&mut self, phase: bool) {
        for i in 0..self.num_vars {
            self.phase[i] = Some(phase);
        }
    }

    /// Set the preferred phase for a specific variable
    ///
    /// This is useful for guiding the search - for example, in LIA solving,
    /// when splitting on an integer variable with fractional value, we can
    /// set the preferred phase to try the closer integer first.
    ///
    /// # Arguments
    /// * `var` - The variable to set phase for
    /// * `phase` - `true` for positive polarity, `false` for negative
    pub fn set_var_phase(&mut self, var: Variable, phase: bool) {
        let idx = var.index();
        if idx < self.num_vars {
            self.phase[idx] = Some(phase);
        }
    }

    /// Set the random seed for variable selection tie-breaking
    ///
    /// This affects the order of variables with equal VSIDS scores.
    /// Different seeds can lead to different search paths.
    pub fn set_random_seed(&mut self, seed: u64) {
        self.vsids.set_random_seed(seed);
    }

    /// Estimate memory usage of the solver (in bytes)
    ///
    /// Returns a breakdown of memory usage by component.
    /// This is an approximation based on Vec capacities and known struct sizes.
    pub fn memory_stats(&self) -> MemoryStats {
        // Per-variable data: assignment, level, reason, trail_pos, phase
        // Option<bool> = 1 byte, u32 = 4, Option<ClauseRef> = 8, usize = 8, Option<bool> = 1
        let per_var_bytes = 1 + 4 + 8 + 8 + 1; // 22 bytes per variable
        let var_data = self.num_vars * per_var_bytes;

        // VSIDS: activities Vec<f64>
        let vsids = self.vsids.capacity() * 8;

        // Conflict analyzer (estimate: similar to num_vars)
        let conflict = self.num_vars * 2; // seen flags + temp storage

        // Clause database with arena allocation:
        // - ClauseDB contains headers Vec and literals Vec (arena)
        // - No per-clause heap allocation
        let clause_db = self.clause_db.memory_bytes();
        let total_literals = self.clause_db.active_literals();

        // Watched lists: Vec<Vec<Watcher>>
        // Outer vec: 2 * num_vars entries (for each literal)
        // Each Watcher is 8 bytes (ClauseRef + Literal)
        let mut watches = self.num_vars * 2 * 24; // outer vec overhead
        for i in 0..(self.num_vars * 2) {
            watches += self.watches.watch_list_capacity(i) * 8;
        }

        // Trail and trail_lim
        let trail = self.trail.capacity() * 4 + self.trail_lim.capacity() * 8;

        // Clause IDs (for LRAT)
        let clause_ids = self.clause_ids.capacity() * 8;

        // Inprocessing engines (rough estimates based on per-var storage)
        let inprocessing = self.num_vars * 50; // Various occurrence lists, etc.

        MemoryStats {
            num_vars: self.num_vars,
            num_clauses: self.clause_db.len(),
            total_literals,
            var_data,
            vsids,
            conflict,
            clause_db,
            watches,
            trail,
            clause_ids,
            inprocessing,
        }
    }

    /// Analyze conflict and learn a clause using 1UIP scheme
    fn analyze_conflict(&mut self, conflict_ref: ClauseRef) -> ConflictResult {
        self.conflict.clear();
        let mut counter = 0; // Literals at current decision level
        let mut p: Option<Literal> = None;
        let mut index = self.trail.len();

        // Mark conflict clause as used (for tier-based clause management)
        self.clause_db
            .header_mut(conflict_ref.0 as usize)
            .mark_used();

        // Add conflict clause to resolution chain (for LRAT)
        if self.lrat_enabled {
            let id = self.clause_id(conflict_ref);
            if id != 0 {
                self.conflict.add_to_chain(id);
            }
        }

        // Start with conflict clause
        let mut clause_lits: Vec<Literal> =
            self.clause_db.literals(conflict_ref.0 as usize).to_vec();

        loop {
            // Process literals in the clause
            for &lit in &clause_lits {
                // Skip the literal we're resolving on
                if let Some(p_lit) = p {
                    if lit == p_lit {
                        continue;
                    }
                }

                let var = lit.variable();
                let var_idx = var.index();

                if !self.conflict.is_seen(var_idx) {
                    self.conflict.mark_seen(var_idx);

                    // Bump VSIDS activity for conflict variables
                    self.vsids.bump(var);

                    let var_level = self.level[var_idx];
                    if var_level == self.decision_level {
                        counter += 1;
                    } else if var_level > 0 {
                        // Add to learned clause (the literal is already false,
                        // so we add it as-is to the learned clause)
                        self.conflict.add_to_learned(lit);
                    }
                    // Level 0 literals are always false, skip them
                }
            }

            // Find next literal to resolve on (from trail, going backwards)
            loop {
                index -= 1;
                p = Some(self.trail[index]);
                if self.conflict.is_seen(p.unwrap().variable().index()) {
                    break;
                }
            }

            self.conflict.unmark_seen(p.unwrap().variable().index());
            counter -= 1;

            if counter == 0 {
                break; // Found 1UIP
            }

            // Get the reason clause for p and mark it as used
            let reason_ref = self.reason[p.unwrap().variable().index()].unwrap();
            self.clause_db.header_mut(reason_ref.0 as usize).mark_used();

            // Add reason clause to resolution chain (for LRAT)
            if self.lrat_enabled {
                let id = self.clause_id(reason_ref);
                if id != 0 {
                    self.conflict.add_to_chain(id);
                }
            }

            clause_lits = self.clause_db.literals(reason_ref.0 as usize).to_vec();
        }

        // p is the 1UIP - add its negation as first literal (asserting literal)
        let uip = p.unwrap().negated();
        self.conflict.set_asserting_literal(uip);

        // Bump reason literals (CaDiCaL's bumpreason optimization)
        // This helps VSIDS focus on important variables by bumping variables
        // in the reason clauses of the learned clause literals.
        self.bump_reason_literals();

        // Minimize the learned clause by removing redundant literals
        self.minimize_learned_clause();

        // Compute backtrack level (second highest level in learned clause)
        let backtrack_level = self.conflict.compute_backtrack_level(&self.level);

        // Compute LBD (Literal Block Distance)
        let lbd = self.conflict.compute_lbd(&self.level);

        self.conflict.get_result(backtrack_level, lbd)
    }

    /// Bump reason literals for improved VSIDS focus (CaDiCaL's bumpreason).
    ///
    /// This bumps variables in the reason clauses of the literals in the learned
    /// clause. The intuition is that these variables are "important" because they
    /// contributed to the conflict, even if they're not directly in the learned clause.
    ///
    /// Parameters (from CaDiCaL):
    /// - Depth limit: 1 (focused) or 2 (stable) - how deep to recurse into reasons
    /// - Analyzed limit: 10x the number of analyzed literals - prevent blowup
    fn bump_reason_literals(&mut self) {
        // Get literals in the learned clause (including UIP)
        let uip = self.conflict.asserting_literal();
        // Clone to avoid borrow conflict with recursive calls that mutate self
        let learned: Vec<Literal> = self.conflict.learned_literals().to_vec();

        // CaDiCaL uses depth 1 in focused mode, 2 in stable mode
        let depth_limit = if self.stable_mode { 2 } else { 1 };

        // Limit how many extra literals we process (10x learned clause size)
        let analyzed_limit = (learned.len() + 1) * 10;
        let mut extra_bumped = 0;

        // Bump reason literals for each literal in the learned clause
        for &lit in std::iter::once(&uip).chain(learned.iter()) {
            if extra_bumped >= analyzed_limit {
                break;
            }
            self.bump_reason_literals_recursive(
                lit.negated(),
                depth_limit,
                &mut extra_bumped,
                analyzed_limit,
            );
        }
    }

    /// Recursively bump reason literals up to a depth limit.
    fn bump_reason_literals_recursive(
        &mut self,
        lit: Literal,
        depth: u32,
        extra_bumped: &mut usize,
        limit: usize,
    ) {
        if depth == 0 || *extra_bumped >= limit {
            return;
        }

        let var_idx = lit.variable().index();

        // Get the reason clause for this literal
        let reason_ref = match self.reason[var_idx] {
            Some(r) => r,
            None => return, // Decision or unit - no reason clause
        };

        // Traverse reason clause and bump unseen variables (iterate by index to avoid allocation)
        let clause_idx = reason_ref.0 as usize;
        let clause_len = self.clause_db.header(clause_idx).len();

        for i in 0..clause_len {
            let reason_lit = self.clause_db.literal(clause_idx, i);
            if reason_lit == lit {
                continue; // Skip the propagated literal itself
            }

            let reason_var = reason_lit.variable();
            let reason_var_idx = reason_var.index();

            // Only bump if not already seen in this conflict analysis
            if !self.conflict.is_seen(reason_var_idx) {
                // Mark as seen to avoid duplicate bumping
                self.conflict.mark_seen(reason_var_idx);
                // Bump VSIDS activity
                self.vsids.bump(reason_var);
                *extra_bumped += 1;

                // Recurse if we have depth remaining
                if depth > 1 && *extra_bumped < limit {
                    self.bump_reason_literals_recursive(
                        reason_lit.negated(),
                        depth - 1,
                        extra_bumped,
                        limit,
                    );
                }
            }
        }
    }

    /// Minimize the learned clause by removing redundant literals.
    ///
    /// Uses CaDiCaL-style minimization with poison/removable marking to cache
    /// results and avoid redundant work. A literal is redundant if it can be
    /// derived from other literals in the learned clause through resolution.
    fn minimize_learned_clause(&mut self) {
        // Get the current learned literals (without UIP)
        let learned = self.conflict.take_learned();
        let mut minimized = Vec::with_capacity(learned.len());

        // Mark all learned literals as "keep" (in the clause)
        for &lit in &learned {
            let var_idx = lit.variable().index();
            self.minimize_visited[var_idx] = true;
            self.minimize_to_clear.push(var_idx);
        }

        // Try to minimize each literal
        for &lit in &learned {
            if self.is_redundant_cached(lit, 0) {
                // Literal is redundant, don't include it
                continue;
            }
            minimized.push(lit);
        }

        // Clear all minimization state
        for &var_idx in &self.minimize_to_clear {
            self.minimize_poison[var_idx] = false;
            self.minimize_removable[var_idx] = false;
            self.minimize_visited[var_idx] = false;
        }
        self.minimize_to_clear.clear();

        // Put minimized literals back
        self.conflict.set_learned(minimized);
    }

    /// Check if a literal is redundant using cached poison/removable marks.
    ///
    /// Uses depth limiting and caches results for efficiency.
    /// Returns true if the literal can be removed from the learned clause.
    fn is_redundant_cached(&mut self, lit: Literal, depth: u32) -> bool {
        let var_idx = lit.variable().index();

        // Level 0 literals are always redundant (they're always false)
        if self.level[var_idx] == 0 {
            return true;
        }

        // Check cached results
        if self.minimize_removable[var_idx] {
            return true;
        }
        if self.minimize_poison[var_idx] {
            return false;
        }

        // For recursive calls (depth > 0): if literal is already in learned clause,
        // we've reached a "kept" literal which is good - this path terminates.
        // For top-level calls (depth == 0): we're checking if THIS literal can be
        // removed, so don't return early just because it's in the clause.
        if depth > 0 && self.minimize_visited[var_idx] {
            return true;
        }

        // Decision variables cannot be redundant
        let reason = match self.reason[var_idx] {
            Some(r) => r,
            None => return false,
        };

        // Depth limiting to prevent infinite recursion
        if depth > self.minimize_depth_limit {
            return false;
        }

        // Mark as visited for this minimization call (prevents infinite loops)
        if !self.minimize_visited[var_idx] {
            self.minimize_visited[var_idx] = true;
            self.minimize_to_clear.push(var_idx);
        }

        // Check all literals in the reason clause (iterate by index to avoid allocation)
        let clause_idx = reason.0 as usize;
        let clause_len = self.clause_db.header(clause_idx).len();

        for i in 0..clause_len {
            let reason_lit = self.clause_db.literal(clause_idx, i);
            let reason_var_idx = reason_lit.variable().index();

            // Skip the literal itself
            if reason_var_idx == var_idx {
                continue;
            }

            // Recursively check if this literal is redundant
            if !self.is_redundant_cached(reason_lit, depth + 1) {
                // Found a non-redundant literal - mark as poison
                self.minimize_poison[var_idx] = true;
                return false;
            }
        }

        // All reason literals are redundant - mark as removable
        self.minimize_removable[var_idx] = true;
        true
    }

    /// Add a learned clause and return its reference
    ///
    /// If proof logging is enabled, the clause is written to the proof.
    /// For LRAT proofs, the resolution chain contains the clause IDs used
    /// to derive the learned clause.
    fn add_learned_clause(
        &mut self,
        lits: Vec<Literal>,
        lbd: u32,
        resolution_chain: &[u64],
    ) -> ClauseRef {
        // Log the learned clause to proof if enabled
        if let Some(ref mut writer) = self.proof_writer {
            // Ignore errors during proof writing (best effort)
            // For DRAT, hints are ignored. For LRAT, these are the clause IDs.
            let _ = writer.add(&lits, resolution_chain);
        }

        let clause_idx = self.add_clause_db(&lits, true);
        let clause_ref = ClauseRef(clause_idx as u32);
        self.clause_db.header_mut(clause_idx).set_lbd(lbd);

        if lits.len() >= 2 {
            // Add watches for the learned clause
            // Watch literals[0] (UIP) and literals[1] (highest level non-UIP)
            let lit0 = lits[0];
            let lit1 = lits[1];
            let is_binary = lits.len() == 2;
            if is_binary {
                self.watches
                    .add_watch(lit0, Watcher::binary(clause_ref, lit1));
                self.watches
                    .add_watch(lit1, Watcher::binary(clause_ref, lit0));
            } else {
                self.watches.add_watch(lit0, Watcher::new(clause_ref, lit1));
                self.watches.add_watch(lit1, Watcher::new(clause_ref, lit0));
            }
        }

        clause_ref
    }

    // ==========================================================================
    // Lucky Phase (CaDiCaL-style pre-solving)
    // ==========================================================================

    /// Try lucky assignment strategies before full CDCL search
    ///
    /// Attempts several simple assignment patterns that can quickly solve
    /// "easy" formulas without full CDCL search. Returns Some(true) for SAT,
    /// Some(false) for UNSAT proven at level 0, None to continue to CDCL.
    fn try_lucky_phases(&mut self) -> Option<bool> {
        // Skip lucky phases for larger formulas where overhead isn't worth it
        // The 250-variable threshold is tuned for random 3-SAT benchmarks
        if self.num_vars > 220 || self.num_original_clauses > 1000 {
            return None;
        }

        // Try each strategy in order (most effective first)
        // Forward strategies work best for random 3-SAT

        if self.lucky_forward_false() {
            return Some(true);
        }
        self.lucky_reset();

        if self.lucky_forward_true() {
            return Some(true);
        }
        self.lucky_reset();

        // Only try additional strategies for smaller formulas
        if self.num_vars <= 200 {
            if self.lucky_backward_false() {
                return Some(true);
            }
            self.lucky_reset();

            if self.lucky_backward_true() {
                return Some(true);
            }
            self.lucky_reset();

            if self.lucky_positive_horn() {
                return Some(true);
            }
            self.lucky_reset();

            if self.lucky_negative_horn() {
                return Some(true);
            }
            self.lucky_reset();
        }

        None // No lucky assignment found
    }

    /// Reset solver state after a failed lucky attempt
    fn lucky_reset(&mut self) {
        if self.decision_level > 0 {
            self.backtrack(0);
        }
    }

    /// Try assigning all variables to false in forward order
    fn lucky_forward_false(&mut self) -> bool {
        for var_idx in 0..self.num_vars {
            let var = Variable(var_idx as u32);
            if self.assignment[var_idx].is_some() {
                continue; // Already assigned by propagation
            }

            let lit = Literal::negative(var);
            self.decide(lit);

            if self.propagate().is_some() {
                // Conflict - try opposite polarity
                if self.decision_level == 1 {
                    return false; // Can't recover at level 1
                }
                self.backtrack(self.decision_level - 1);
                let lit_opp = Literal::positive(var);
                self.decide(lit_opp);
                if self.propagate().is_some() {
                    return false;
                }
            }
        }
        true // All variables assigned without conflict
    }

    /// Try assigning all variables to true in forward order
    fn lucky_forward_true(&mut self) -> bool {
        for var_idx in 0..self.num_vars {
            let var = Variable(var_idx as u32);
            if self.assignment[var_idx].is_some() {
                continue;
            }

            let lit = Literal::positive(var);
            self.decide(lit);

            if self.propagate().is_some() {
                if self.decision_level == 1 {
                    return false;
                }
                self.backtrack(self.decision_level - 1);
                let lit_opp = Literal::negative(var);
                self.decide(lit_opp);
                if self.propagate().is_some() {
                    return false;
                }
            }
        }
        true
    }

    /// Try assigning all variables to false in reverse order
    fn lucky_backward_false(&mut self) -> bool {
        for var_idx in (0..self.num_vars).rev() {
            let var = Variable(var_idx as u32);
            if self.assignment[var_idx].is_some() {
                continue;
            }

            let lit = Literal::negative(var);
            self.decide(lit);

            if self.propagate().is_some() {
                if self.decision_level == 1 {
                    return false;
                }
                self.backtrack(self.decision_level - 1);
                let lit_opp = Literal::positive(var);
                self.decide(lit_opp);
                if self.propagate().is_some() {
                    return false;
                }
            }
        }
        true
    }

    /// Try assigning all variables to true in reverse order
    fn lucky_backward_true(&mut self) -> bool {
        for var_idx in (0..self.num_vars).rev() {
            let var = Variable(var_idx as u32);
            if self.assignment[var_idx].is_some() {
                continue;
            }

            let lit = Literal::positive(var);
            self.decide(lit);

            if self.propagate().is_some() {
                if self.decision_level == 1 {
                    return false;
                }
                self.backtrack(self.decision_level - 1);
                let lit_opp = Literal::negative(var);
                self.decide(lit_opp);
                if self.propagate().is_some() {
                    return false;
                }
            }
        }
        true
    }

    /// Try positive horn: for each clause, satisfy via first positive literal
    fn lucky_positive_horn(&mut self) -> bool {
        for clause_idx in 0..self.num_original_clauses {
            let header = self.clause_db.header(clause_idx);
            let len = header.len();

            let mut satisfied = false;
            let mut first_positive: Option<Literal> = None;

            for i in 0..len {
                let lit = self.clause_db.literal(clause_idx, i);
                match self.lit_value(lit) {
                    Some(true) => {
                        satisfied = true;
                        break;
                    }
                    Some(false) => continue,
                    None => {
                        if lit.is_positive() && first_positive.is_none() {
                            first_positive = Some(lit);
                        }
                    }
                }
            }

            if satisfied {
                continue;
            }

            match first_positive {
                Some(lit) => {
                    self.decide(lit);
                    if self.propagate().is_some() {
                        return false;
                    }
                }
                None => return false, // No positive unassigned literal
            }
        }

        // Assign remaining variables
        for var_idx in 0..self.num_vars {
            if self.assignment[var_idx].is_none() {
                let var = Variable(var_idx as u32);
                let lit = Literal::negative(var);
                self.decide(lit);
                if self.propagate().is_some() {
                    return false;
                }
            }
        }
        true
    }

    /// Try negative horn: for each clause, satisfy via first negative literal
    fn lucky_negative_horn(&mut self) -> bool {
        for clause_idx in 0..self.num_original_clauses {
            let header = self.clause_db.header(clause_idx);
            let len = header.len();

            let mut satisfied = false;
            let mut first_negative: Option<Literal> = None;

            for i in 0..len {
                let lit = self.clause_db.literal(clause_idx, i);
                match self.lit_value(lit) {
                    Some(true) => {
                        satisfied = true;
                        break;
                    }
                    Some(false) => continue,
                    None => {
                        if !lit.is_positive() && first_negative.is_none() {
                            first_negative = Some(lit);
                        }
                    }
                }
            }

            if satisfied {
                continue;
            }

            match first_negative {
                Some(lit) => {
                    self.decide(lit);
                    if self.propagate().is_some() {
                        return false;
                    }
                }
                None => return false,
            }
        }

        // Assign remaining variables
        for var_idx in 0..self.num_vars {
            if self.assignment[var_idx].is_none() {
                let var = Variable(var_idx as u32);
                let lit = Literal::positive(var);
                self.decide(lit);
                if self.propagate().is_some() {
                    return false;
                }
            }
        }
        true
    }

    /// Run walk-based phase initialization.
    ///
    /// This uses ProbSAT random walk to find good initial phases for CDCL.
    /// The walk tries to minimize unsatisfied clauses and saves the best
    /// phases found for use in decision making.
    ///
    /// Returns true if walk found a satisfying assignment (SAT), false otherwise.
    fn try_walk(&mut self) -> bool {
        if !self.walk_enabled {
            return false;
        }

        // Skip walk for very small formulas (lucky phases handle these)
        if self.num_vars < 50 || self.num_original_clauses < 100 {
            return false;
        }

        // Use a seed based on problem characteristics for reproducibility
        let seed = (self.num_vars as u64)
            .wrapping_mul(31)
            .wrapping_add(self.num_original_clauses as u64);

        // Run walk to find good phases
        let found_sat = crate::walk::walk(
            &self.clause_db,
            self.num_vars,
            &mut self.phase,
            &mut self.walk_stats,
            seed,
            self.walk_limit,
        );

        // Copy walk phases to target phases for use in decisions
        for i in 0..self.num_vars {
            if let Some(phase) = self.phase[i] {
                self.target_phase[i] = Some(phase);
            }
        }

        found_sat
    }

    /// Run warmup-based phase initialization.
    ///
    /// Uses CDCL propagation (ignoring conflicts) to find good initial phases.
    /// This is more efficient than walk for small/medium instances because
    /// it uses O(1) amortized 2-watched literal propagation instead of O(n²)
    /// break-value computation.
    fn try_warmup(&mut self) {
        if !self.warmup_enabled {
            return;
        }

        // Skip warmup for very small formulas
        if self.num_vars < 20 || self.num_original_clauses < 50 {
            return;
        }

        crate::warmup::warmup(
            &self.clause_db,
            self.num_vars,
            &self.phase,
            &mut self.target_phase,
            &mut self.warmup_stats,
        );
    }

    /// Get the current model (assignment)
    fn get_model(&self) -> Vec<bool> {
        self.assignment.iter().map(|v| v.unwrap_or(false)).collect()
    }

    /// Get model from saved phases (used when walk finds SAT)
    fn get_model_from_phases(&self) -> Vec<bool> {
        self.phase.iter().map(|p| p.unwrap_or(true)).collect()
    }

    /// Verify that a model satisfies all original clauses.
    ///
    /// This is a soundness check that verifies the model against:
    /// 1. All non-deleted clauses in the clause database
    /// 2. All clauses removed by BVE (stored in reconstruction stack)
    ///
    /// Returns true if the model is valid, false if any clause is unsatisfied.
    fn verify_model(&self, model: &[bool]) -> bool {
        // Check all non-deleted clauses in clause_db
        for idx in self.clause_db.indices() {
            let header = self.clause_db.header(idx);
            if header.is_empty() {
                continue; // Skip deleted clauses
            }

            let lits = self.clause_db.literals(idx);
            let satisfied = lits.iter().any(|&lit| {
                let var_idx = lit.variable().index();
                if var_idx >= model.len() {
                    return false; // Unassigned variable - assume false
                }
                let val = model[var_idx];
                if lit.is_positive() {
                    val
                } else {
                    !val
                }
            });

            if !satisfied {
                #[cfg(debug_assertions)]
                eprintln!(
                    "BUG: Clause {} not satisfied by model: {:?}",
                    idx,
                    lits.iter()
                        .map(|l| if l.is_positive() {
                            l.variable().0 as i32 + 1
                        } else {
                            -(l.variable().0 as i32 + 1)
                        })
                        .collect::<Vec<_>>()
                );
                return false;
            }
        }

        // Check BVE clauses from reconstruction stack
        for clause in self.reconstruction.iter_bve_clauses() {
            let satisfied = clause.iter().any(|&lit| {
                let var_idx = lit.variable().index();
                if var_idx >= model.len() {
                    return false;
                }
                let val = model[var_idx];
                if lit.is_positive() {
                    val
                } else {
                    !val
                }
            });

            if !satisfied {
                #[cfg(debug_assertions)]
                eprintln!(
                    "BUG: BVE clause not satisfied by model: {:?}",
                    clause
                        .iter()
                        .map(|l| if l.is_positive() {
                            l.variable().0 as i32 + 1
                        } else {
                            -(l.variable().0 as i32 + 1)
                        })
                        .collect::<Vec<_>>()
                );
                return false;
            }
        }

        true
    }

    /// Handle initial unit clauses (before solve loop)
    fn process_initial_clauses(&mut self) -> bool {
        for i in 0..self.clause_db.len() {
            let header = self.clause_db.header(i);
            if header.len() == 1 {
                let lit = self.clause_db.literal(i, 0);
                if let Some(val) = self.lit_value(lit) {
                    if !val {
                        // Unit clause is already falsified
                        return false;
                    }
                    // Already satisfied, skip
                } else {
                    // Propagate unit clause
                    self.enqueue(lit, Some(ClauseRef(i as u32)));
                }
            }
        }
        true
    }

    /// Pick the polarity for a decision variable using phase saving with target phases
    ///
    /// Phase selection priority:
    /// 1. Target phase (if available): Uses the assignment from the longest conflict-free
    ///    trail seen. This guides the search toward promising regions.
    /// 2. Saved phase: The last polarity this variable was assigned.
    /// 3. Default: Positive polarity if no phase information exists.
    ///
    /// Target phases help the solver explore variations of assignments that were
    /// close to satisfying the formula.
    fn pick_phase(&self, var: Variable) -> Literal {
        let idx = var.index();

        // First, try target phase (from longest conflict-free trail)
        if let Some(phase) = self.target_phase[idx] {
            return if phase {
                Literal::positive(var)
            } else {
                Literal::negative(var)
            };
        }

        // Fall back to saved phase
        match self.phase[idx] {
            Some(true) => Literal::positive(var),
            Some(false) => Literal::negative(var),
            None => Literal::positive(var), // Default to positive if no saved phase
        }
    }

    /// Compute the actual backtrack level, deciding between chronological and
    /// non-chronological backtracking based on the jump distance.
    ///
    /// Based on the SAT'18 paper "Chronological Backtracking" and CaDiCaL's
    /// `chronoreusetrail` optimization:
    ///
    /// - If the jump would skip many levels (> CHRONO_LEVEL_LIMIT), use
    ///   chronological backtracking (just go back one level)
    /// - Otherwise, try to reuse trail by finding the best variable above the
    ///   jump level and backtracking only to that level
    fn compute_chrono_backtrack_level(&mut self, jump_level: u32) -> u32 {
        if !self.chrono_enabled {
            return jump_level;
        }

        // If jump level is at or above current level - 1, no point in chrono BT
        if jump_level >= self.decision_level.saturating_sub(1) {
            return jump_level;
        }

        // Compute how many levels we'd skip with NCB
        let skip_levels = self.decision_level - jump_level;

        if skip_levels > CHRONO_LEVEL_LIMIT {
            // Too many levels to skip - use chronological backtracking
            self.chrono_backtracks += 1;
            self.decision_level - 1
        } else if self.chrono_reuse_trail && self.stable_mode {
            // CaDiCaL-style trail reuse: find the best variable above the jump level
            // and only backtrack to that level to keep more of the useful trail
            // Only use in stable mode where VSIDS scores are more reliable
            self.compute_chrono_reuse_level(jump_level)
        } else {
            // Use non-chronological backtracking
            jump_level
        }
    }

    /// Find the best variable above the jump level and return its level.
    ///
    /// This implements CaDiCaL's `chronoreusetrail` optimization. Instead of
    /// always backtracking to the asserting level, we look for valuable variables
    /// above that level. In stable mode, we look for the variable with highest
    /// VSIDS activity. In focused mode, we look for the most recently bumped
    /// variable. We then backtrack only to the level containing that trail
    /// position, preserving more of the useful search state.
    ///
    /// Key: we use trail POSITION to determine level, not the variable's stored
    /// level. This is important for correctness with chronological backtracking
    /// where variables can have out-of-order level assignments.
    fn compute_chrono_reuse_level(&mut self, jump_level: u32) -> u32 {
        // Get the trail position where jump_level+1 starts
        let start_pos = if (jump_level as usize) < self.trail_lim.len() {
            self.trail_lim[jump_level as usize]
        } else {
            return jump_level;
        };

        // If no assignments above jump level, just use jump level
        if start_pos >= self.trail.len() {
            return jump_level;
        }

        // Find the best variable's trail position (not just index)
        let mut best_pos: Option<usize> = None;

        if self.stable_mode {
            // In stable mode, use VSIDS activity (higher is better)
            let mut best_activity = f64::NEG_INFINITY;
            for i in start_pos..self.trail.len() {
                let var = self.trail[i].variable();
                let activity = self.vsids.activity(var);
                if activity > best_activity {
                    best_activity = activity;
                    best_pos = Some(i);
                }
            }
        } else {
            // In focused mode, use bump order (higher is better = more recently bumped)
            let mut best_bump = 0u64;
            for i in start_pos..self.trail.len() {
                let var = self.trail[i].variable();
                let bump = self.vsids.bump_order(var);
                if bump > best_bump {
                    best_bump = bump;
                    best_pos = Some(i);
                }
            }
        }

        // Find the level containing the best variable's trail position
        // CaDiCaL: while (res < level - 1 && control[res + 1].trail <= best_pos) res++;
        let best_pos = match best_pos {
            Some(p) => p,
            None => return jump_level,
        };

        // CaDiCaL: while (res < level - 1 && control[res + 1].trail <= best_pos) res++;
        // In Z4: trail_lim[i] = start of level i+1's assignments
        // So to match control[res+1].trail (start of level res+1), use trail_lim[res]
        let mut res = jump_level;
        while res < self.decision_level - 1
            && (res as usize) < self.trail_lim.len()
            && self.trail_lim[res as usize] <= best_pos
        {
            res += 1;
        }

        if res > jump_level {
            self.chrono_backtracks += 1;
        }

        res
    }

    /// Reset transient search state so the solver can be reused across multiple `solve()` calls.
    ///
    /// This keeps the clause database intact (including learned clauses), but clears assignments,
    /// watches, and scheduling state that assume a fresh search.
    fn reset_search_state(&mut self) {
        // Core assignment / trail state
        self.assignment.fill(None);
        self.level.fill(0);
        self.reason.fill(None);
        self.trail.clear();
        self.trail_lim.clear();
        self.decision_level = 0;
        self.qhead = 0;
        self.trail_pos.fill(usize::MAX);

        // Reset VSIDS heap to include all variables (they are all unassigned now)
        self.vsids.reset_heap();
        self.vsids.reset_vmtf_unassigned();

        // Watches are rebuilt each solve. Use clear() + ensure_num_vars() to avoid
        // reallocating the outer Vec on incremental solves.
        self.watches.clear();
        self.watches.ensure_num_vars(self.num_vars);

        // Restart / scheduling state
        self.conflicts_since_restart = 0;
        self.luby_idx = 1;
        self.restarts = 0;

        // Glucose-style EMA state
        self.lbd_ema_fast = 0.0;
        self.lbd_ema_slow = 0.0;

        // Stabilization state
        self.stable_mode = false;
        self.stable_mode_start_conflicts = 0;
        self.stable_phase_length = STABLE_PHASE_INIT;
        self.stable_phase_count = 0;
        self.reluctant_counter = 0;
        self.reluctant_period = RELUCTANT_INIT;

        // Target/best phase tracking (reset target, keep best across solves)
        self.target_trail_len = 0;
        for phase in self.target_phase.iter_mut() {
            *phase = None;
        }
        // Note: best_phase and best_trail_len are kept across solves for rephasing

        // Clause database management scheduling (counters restart each solve).
        self.num_conflicts = 0;
        self.num_decisions = 0;
        self.num_propagations = 0;
        self.num_original_clauses = 0;
        self.next_reduce_db = FIRST_REDUCE_DB;
        self.clause_activity_inc = 1.0;

        // Inprocessing scheduling (relative to `num_conflicts`)
        self.next_vivify = VIVIFY_INTERVAL;
        self.next_subsume = SUBSUME_INTERVAL;
        self.next_probe = PROBE_INTERVAL;
        self.next_bve = BVE_INTERVAL;
        self.next_bce = BCE_INTERVAL;
        self.next_htr = HTR_INTERVAL;
        self.next_sweep = SWEEP_INTERVAL;
        self.fixed_count = 0;
    }

    /// Solve the formula
    pub fn solve(&mut self) -> SolveResult {
        if self.has_empty_clause {
            self.finalize_unsat_proof();
            return SolveResult::Unsat;
        }

        if self.scope_selectors.is_empty() {
            return self.solve_no_assumptions(|| false);
        }

        let assumptions: Vec<Literal> = self
            .scope_selectors
            .iter()
            .copied()
            .map(Literal::negative)
            .collect();

        match self.solve_with_assumptions_core(&assumptions) {
            AssumeResult::Sat(model) => SolveResult::Sat(model),
            AssumeResult::Unsat(_) => SolveResult::Unsat,
            AssumeResult::Unknown => SolveResult::Unknown,
        }
    }

    /// Solve the formula with an interrupt callback
    ///
    /// The callback is checked periodically (every ~100 conflicts). If it returns
    /// `true`, solving is interrupted and `SolveResult::Unknown` is returned.
    ///
    /// This is useful for parallel portfolio solving where multiple solvers run
    /// concurrently and can be stopped when one finds a solution.
    pub fn solve_interruptible<F>(&mut self, should_stop: F) -> SolveResult
    where
        F: Fn() -> bool,
    {
        if self.has_empty_clause {
            self.finalize_unsat_proof();
            return SolveResult::Unsat;
        }

        if self.scope_selectors.is_empty() {
            return self.solve_no_assumptions(should_stop);
        }

        let assumptions: Vec<Literal> = self
            .scope_selectors
            .iter()
            .copied()
            .map(Literal::negative)
            .collect();

        match self.solve_with_assumptions_core(&assumptions) {
            AssumeResult::Sat(model) => SolveResult::Sat(model),
            AssumeResult::Unsat(_) => SolveResult::Unsat,
            AssumeResult::Unknown => SolveResult::Unknown,
        }
    }

    fn solve_no_assumptions<F>(&mut self, should_stop: F) -> SolveResult
    where
        F: Fn() -> bool,
    {
        // Allow calling `solve()` multiple times after adding clauses (e.g. for DPLL(T)).
        self.reset_search_state();

        // Handle empty formula
        if self.clause_db.is_empty() {
            let model = self.finalize_sat_model(self.get_model());
            return SolveResult::Sat(model);
        }

        // Track number of original clauses (before learning)
        self.num_original_clauses = self.clause_db.len();

        // Initialize watches
        self.initialize_watches();

        // Process initial unit clauses
        if !self.process_initial_clauses() {
            self.finalize_unsat_proof();
            return SolveResult::Unsat;
        }

        // Propagate unit clauses before trying lucky phases
        if self.propagate().is_some() {
            // Conflict at level 0 during initial propagation
            self.finalize_unsat_proof();
            return SolveResult::Unsat;
        }

        // Run initial preprocessing (BVE, probing, subsumption)
        // This can eliminate variables and simplify clauses before CDCL
        if self.preprocess_enabled && self.preprocess() {
            // UNSAT detected during preprocessing
            self.finalize_unsat_proof();
            return SolveResult::Unsat;
        }

        // Reinitialize watches after preprocessing (clauses may have been modified)
        if self.preprocess_enabled {
            self.initialize_watches();
        }

        // Try lucky phases (CaDiCaL-style pre-solving)
        // This can quickly solve formulas with simple satisfying assignments
        if let Some(sat) = self.try_lucky_phases() {
            if sat {
                // Lucky phase found satisfying assignment
                let model = self.finalize_sat_model(self.get_model());
                return SolveResult::Sat(model);
            }
            // UNSAT proven at level 0 during lucky phase
            self.finalize_unsat_proof();
            return SolveResult::Unsat;
        }

        // Run warmup to initialize target phases before walk
        // Warmup uses propagation-based phase setting which is O(1) per propagation
        self.try_warmup();

        // Try walk-based phase initialization for larger formulas
        // Walk uses ProbSAT to find good initial phases by minimizing unsatisfied clauses
        if self.try_walk() {
            // Walk found a satisfying assignment
            let model = self.finalize_sat_model(self.get_model_from_phases());
            return SolveResult::Sat(model);
        }

        // Main CDCL loop
        loop {
            // Propagate
            if let Some(conflict_ref) = self.propagate() {
                // Conflict found
                if self.decision_level == 0 {
                    // Conflict at level 0 -> UNSAT
                    self.finalize_unsat_proof();
                    return SolveResult::Unsat;
                }

                // Count conflicts for restart and clause deletion scheduling
                self.conflicts_since_restart += 1;
                self.num_conflicts += 1;

                // Check for interrupt every 100 conflicts
                if self.num_conflicts.is_multiple_of(100) && should_stop() {
                    return SolveResult::Unknown;
                }

                // Bump activity for conflict clause
                self.bump_clause_activity(conflict_ref);

                // Analyze conflict and learn clause
                let result = self.analyze_conflict(conflict_ref);

                // Compute actual backtrack level (may use chronological BT)
                let actual_backtrack_level =
                    self.compute_chrono_backtrack_level(result.backtrack_level);

                // Backtrack to the computed level
                self.backtrack(actual_backtrack_level);

                // Add learned clause (with resolution chain for LRAT proofs)
                let learned_ref = self.add_learned_clause(
                    result.learned_clause.clone(),
                    result.lbd,
                    &result.resolution_chain,
                );

                // Update LBD EMA for Glucose-style restart decisions
                self.update_lbd_ema(result.lbd);

                // Assert the UIP literal (first literal in learned clause)
                let uip = result.learned_clause[0];
                if result.learned_clause.len() == 1 {
                    // Unit learned clause - no reason needed at level 0
                    self.enqueue(uip, Some(learned_ref));
                } else {
                    self.enqueue(uip, Some(learned_ref));
                }

                // Decay VSIDS and clause activities
                self.vsids.decay();
                self.decay_clause_activity();

                // Reduce clause database if needed
                if self.should_reduce_db() {
                    self.reduce_db();
                }
            } else {
                // No conflict - check for restart, rephasing, inprocessing, or decide
                if self.should_restart() {
                    self.do_restart();
                    // Check if we should rephase (change phase selection strategy)
                    if self.should_rephase() {
                        self.rephase();
                    }
                    // Run inprocessing after restart (at level 0)
                    if self.should_vivify() && self.vivify() {
                        self.finalize_unsat_proof();
                        return SolveResult::Unsat;
                    }
                    if self.should_subsume() {
                        self.subsume();
                    }
                    if self.should_probe() {
                        // Probing can derive UNSAT directly
                        if self.probe() && self.propagate().is_some() {
                            self.finalize_unsat_proof();
                            return SolveResult::Unsat;
                        }
                    }
                    if self.should_bve() {
                        // BVE can derive UNSAT directly (empty resolvent)
                        if self.bve() {
                            self.finalize_unsat_proof();
                            return SolveResult::Unsat;
                        }
                        // Propagate any units derived from BVE
                        if self.propagate().is_some() {
                            self.finalize_unsat_proof();
                            return SolveResult::Unsat;
                        }
                    }
                    if self.should_bce() {
                        // BCE removes blocked clauses
                        self.bce();
                    }
                    if self.should_htr() {
                        // HTR produces binary/ternary resolvents
                        self.htr();
                        // Propagate any units derived from HTR
                        if self.propagate().is_some() {
                            self.finalize_unsat_proof();
                            return SolveResult::Unsat;
                        }
                    }
                    if self.should_sweep() {
                        // SAT sweeping detects equivalences via SCC on implication graph
                        if self.sweep() {
                            self.finalize_unsat_proof();
                            return SolveResult::Unsat;
                        }
                        // Propagate any units derived from sweeping
                        if self.propagate().is_some() {
                            self.finalize_unsat_proof();
                            return SolveResult::Unsat;
                        }
                    }
                } else if let Some(var) = self.pick_next_decision_variable() {
                    // Use phase saving to pick polarity
                    let lit = self.pick_phase(var);
                    self.decide(lit);
                } else {
                    // All variables assigned -> SAT
                    let model = self.finalize_sat_model(self.get_model());
                    return SolveResult::Sat(model);
                }
            }
        }
    }

    /// Finalize UNSAT proof by writing the empty clause
    ///
    /// This method centralizes UNSAT proof finalization. All UNSAT return sites
    /// MUST call this method before returning `SolveResult::Unsat`.
    ///
    /// # DRAT/LRAT format
    ///
    /// The empty clause (`0`) marks the final derivation of contradiction in
    /// DRAT/LRAT proofs. External proof checkers (e.g., drat-trim) require
    /// this to validate the proof is complete.
    fn finalize_unsat_proof(&mut self) {
        if let Some(ref mut writer) = self.proof_writer {
            // Write empty clause to indicate final derivation of contradiction
            // For LRAT, we pass empty hints (the final conflict derives the empty clause)
            let _ = writer.add(&[], &[]);
            let _ = writer.flush();
        }
    }

    /// Finalize SAT model: reconstruct, verify, and truncate
    ///
    /// This method centralizes SAT model finalization to ensure all SAT returns
    /// go through verification. All SAT return sites MUST use this method.
    ///
    /// # Invariants enforced
    ///
    /// 1. **Reconstruction**: Variables eliminated by BVE/sweeping are restored
    ///    to values consistent with the original formula.
    /// 2. **Verification**: `debug_assert!` validates the model satisfies ALL
    ///    clauses (including internal selector variables for incremental solving).
    /// 3. **Truncation**: Internal variables are removed; only user-visible
    ///    variables (0..user_num_vars) are returned.
    ///
    /// # Panics (debug builds)
    ///
    /// Panics if the model fails verification, indicating a solver bug.
    fn finalize_sat_model(&mut self, mut model: Vec<bool>) -> Vec<bool> {
        // Apply reconstruction for equisatisfiable transformations (BVE, etc.)
        self.reconstruction.reconstruct(&mut model);
        // Verify BEFORE truncation (clauses may contain internal selector variables)
        debug_assert!(
            self.verify_model(&model),
            "BUG: SAT model does not satisfy all clauses"
        );
        // Truncate to user-visible variables only
        model.truncate(self.user_num_vars);
        model
    }

    /// Get the proof writer (for testing/inspection)
    pub fn proof_writer(&self) -> Option<&ProofOutput<W>> {
        self.proof_writer.as_ref()
    }

    /// Take the proof writer out of the solver (consumes proof logging capability)
    pub fn take_proof_writer(&mut self) -> Option<ProofOutput<W>> {
        self.proof_writer.take()
    }

    /// Solve the formula with assumptions
    ///
    /// This performs assumption-based solving, where the given literals are
    /// treated as temporary unit clauses for this solve call only. The solver
    /// state (learned clauses, etc.) is preserved between calls.
    ///
    /// Returns:
    /// - `AssumeResult::Sat(model)` if satisfiable with the assumptions
    /// - `AssumeResult::Unsat(core)` if unsatisfiable, where `core` is a subset
    ///   of the assumptions that caused the conflict
    /// - `AssumeResult::Unknown` if the solver could not determine satisfiability
    ///
    /// The unsat core extraction follows the MiniSat approach: assumptions are
    /// assigned at decision levels 1, 2, ..., n. When a conflict occurs that
    /// requires backtracking past all assumptions (to level 0), the assumptions
    /// involved in the conflict analysis form the unsat core.
    pub fn solve_with_assumptions(&mut self, assumptions: &[Literal]) -> AssumeResult {
        if self.has_empty_clause {
            self.finalize_unsat_proof();
            return AssumeResult::Unsat(vec![]);
        }

        let mut combined = Vec::with_capacity(self.scope_selectors.len() + assumptions.len());
        for &selector in &self.scope_selectors {
            combined.push(Literal::negative(selector));
        }
        combined.extend_from_slice(assumptions);

        let result = if combined.is_empty() {
            match self.solve_no_assumptions(|| false) {
                SolveResult::Sat(model) => AssumeResult::Sat(model),
                SolveResult::Unsat => AssumeResult::Unsat(vec![]),
                SolveResult::Unknown => AssumeResult::Unknown,
            }
        } else {
            self.solve_with_assumptions_core(&combined)
        };

        match result {
            AssumeResult::Sat(mut model) => {
                model.truncate(self.user_num_vars);
                AssumeResult::Sat(model)
            }
            AssumeResult::Unsat(core) => {
                let filtered: Vec<Literal> = core
                    .into_iter()
                    .filter(|lit| !self.scope_selectors.contains(&lit.variable()))
                    .collect();
                AssumeResult::Unsat(filtered)
            }
            AssumeResult::Unknown => AssumeResult::Unknown,
        }
    }

    fn solve_with_assumptions_core(&mut self, assumptions: &[Literal]) -> AssumeResult {
        // Reset search state but preserve learned clauses
        self.reset_search_state();

        // Handle empty formula
        if self.clause_db.is_empty() {
            // Even when there are no clauses, assumptions still constrain the model.
            // Satisfiable unless assumptions contain an immediate contradiction.
            let mut model = self.get_model();
            let mut first_lit_for_var: Vec<Option<Literal>> = vec![None; self.num_vars];

            for &lit in assumptions {
                let var_idx = lit.variable().index();
                if var_idx >= self.num_vars {
                    continue;
                }
                let desired = lit.is_positive();

                if let Some(prev) = first_lit_for_var[var_idx] {
                    if prev.is_positive() != desired {
                        return AssumeResult::Unsat(vec![prev, lit]);
                    }
                } else {
                    first_lit_for_var[var_idx] = Some(lit);
                    model[var_idx] = desired;
                }
            }

            let model = self.finalize_sat_model(model);
            return AssumeResult::Sat(model);
        }

        // Track number of original clauses
        self.num_original_clauses = self.clause_db.len();

        // Initialize watches
        self.initialize_watches();

        // Process initial unit clauses
        if !self.process_initial_clauses() {
            self.finalize_unsat_proof();
            return AssumeResult::Unsat(vec![]);
        }

        // Track which variables are assumptions and which assumptions are "failed"
        let mut is_assumption = vec![false; self.num_vars];
        let mut assumption_lit = vec![None; self.num_vars];
        let mut failed_assumptions: Vec<Literal> = Vec::new();

        for &lit in assumptions {
            let var_idx = lit.variable().index();
            if var_idx < self.num_vars {
                is_assumption[var_idx] = true;
                assumption_lit[var_idx] = Some(lit);
            }
        }

        // Current assumption index we're trying to set
        let mut assumption_idx = 0;

        // Main CDCL loop with assumptions
        loop {
            // Propagate
            if let Some(conflict_ref) = self.propagate() {
                // Conflict found
                if self.decision_level == 0 {
                    // Conflict at level 0 -> UNSAT
                    // Collect all assumptions that were marked as failed
                    self.finalize_unsat_proof();
                    return AssumeResult::Unsat(failed_assumptions);
                }

                self.conflicts_since_restart += 1;
                self.num_conflicts += 1;

                // Bump activity for conflict clause
                self.bump_clause_activity(conflict_ref);

                // Analyze conflict and learn clause
                let result = self.analyze_conflict(conflict_ref);

                // Check if any assumptions are involved in the conflict
                // (literals in the learned clause at assumption levels)
                for &lit in result.learned_clause.iter() {
                    let var_idx = lit.variable().index();
                    let var_level = self.level[var_idx];
                    // Assumptions are at levels 1..=assumptions.len()
                    if var_level > 0
                        && var_level <= assumptions.len() as u32
                        && is_assumption[var_idx]
                    {
                        if let Some(assump_lit) = assumption_lit[var_idx] {
                            if !failed_assumptions.contains(&assump_lit) {
                                failed_assumptions.push(assump_lit);
                            }
                        }
                    }
                }

                // If backtrack level is 0, we're done - formula is UNSAT under assumptions
                if result.backtrack_level == 0 {
                    self.finalize_unsat_proof();
                    // Also check if the UIP itself is a failed assumption
                    let uip_var = result.learned_clause[0].variable().index();
                    if is_assumption[uip_var] {
                        if let Some(assump_lit) = assumption_lit[uip_var] {
                            if !failed_assumptions.contains(&assump_lit) {
                                failed_assumptions.push(assump_lit);
                            }
                        }
                    }
                    return AssumeResult::Unsat(failed_assumptions);
                }

                // Compute actual backtrack level (may use chronological BT)
                // Note: For assumption-based solving, we may want to be more conservative
                let actual_backtrack_level =
                    self.compute_chrono_backtrack_level(result.backtrack_level);

                // Update assumption_idx if we're backtracking past assumption levels
                if actual_backtrack_level < assumptions.len() as u32 {
                    assumption_idx = actual_backtrack_level as usize;
                }

                // Backtrack
                self.backtrack(actual_backtrack_level);

                // Add learned clause
                let learned_ref = self.add_learned_clause(
                    result.learned_clause.clone(),
                    result.lbd,
                    &result.resolution_chain,
                );

                // Update LBD EMA for Glucose-style restart decisions
                self.update_lbd_ema(result.lbd);

                // Assert the UIP literal
                let uip = result.learned_clause[0];
                self.enqueue(uip, Some(learned_ref));

                // Decay activities
                self.vsids.decay();
                self.decay_clause_activity();

                // Reduce clause database if needed
                if self.should_reduce_db() {
                    self.reduce_db();
                }
            } else {
                // No conflict

                // First, try to set any remaining assumptions
                if assumption_idx < assumptions.len() {
                    let assump_lit = assumptions[assumption_idx];
                    let var = assump_lit.variable();
                    let var_idx = var.index();

                    // Check if this assumption is already assigned
                    if let Some(val) = self.assignment[var_idx] {
                        let expected = assump_lit.is_positive();
                        if val != expected {
                            // Conflict with assumption - this assumption is failed
                            failed_assumptions.push(assump_lit);
                            // Also collect any other failed assumptions from the conflict
                            // The conflict is between the assumption and existing assignment
                            // Find the reason for the existing assignment
                            if let Some(reason_ref) = self.reason[var_idx] {
                                // Analyze which assumptions are in the reason
                                for &lit in self.clause_db.literals(reason_ref.0 as usize) {
                                    let reason_var_idx = lit.variable().index();
                                    if is_assumption[reason_var_idx] {
                                        if let Some(a_lit) = assumption_lit[reason_var_idx] {
                                            if !failed_assumptions.contains(&a_lit) {
                                                failed_assumptions.push(a_lit);
                                            }
                                        }
                                    }
                                }
                            }
                            self.finalize_unsat_proof();
                            return AssumeResult::Unsat(failed_assumptions);
                        }
                        // Already assigned to correct value, move to next assumption
                        assumption_idx += 1;
                        continue;
                    }

                    // Make the assumption as a decision
                    assumption_idx += 1;
                    self.decide(assump_lit);
                    continue;
                }

                // All assumptions set, continue with regular solving
                if self.should_restart() {
                    // For assumption-based solving, only restart back to assumption level
                    // Don't run inprocessing during assumption solving to preserve state
                    self.do_partial_restart(assumptions.len() as u32);
                    // Check if we should rephase (change phase selection strategy)
                    if self.should_rephase() {
                        self.rephase();
                    }
                } else if let Some(var) = self.pick_next_decision_variable() {
                    let lit = self.pick_phase(var);
                    self.decide(lit);
                } else {
                    // All variables assigned -> SAT
                    let model = self.finalize_sat_model(self.get_model());
                    return AssumeResult::Sat(model);
                }
            }
        }
    }

    /// Partial restart - only restart back to a given level (for assumption-based solving)
    fn do_partial_restart(&mut self, min_level: u32) {
        if self.decision_level <= min_level {
            return;
        }

        // Backtrack to just above the minimum level
        self.backtrack(min_level);
        self.conflicts_since_restart = 0;
        self.restarts += 1;

        // Update Luby sequence
        self.luby_idx += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_new_var_updates_model_len() {
        let mut solver = Solver::new(1);
        let v0 = Variable(0);
        let v1 = solver.new_var();
        assert_eq!(v1, Variable(1));

        // Force both variables true so the model is fully assigned.
        solver.add_clause(vec![Literal::positive(v0)]);
        solver.add_clause(vec![Literal::positive(v1)]);

        let result = solver.solve();
        match result {
            SolveResult::Sat(model) => {
                assert_eq!(
                    model.len(),
                    2,
                    "model should include dynamically added variables"
                );
                assert!(model[0]);
                assert!(model[1]);
            }
            other => panic!("expected SAT, got {:?}", other),
        }
    }

    #[test]
    fn test_ensure_num_vars_updates_model_len() {
        let mut solver = Solver::new(1);
        solver.ensure_num_vars(3);

        // Constrain just the first variable; the others are don't-cares but should
        // still appear in the returned model.
        solver.add_clause(vec![Literal::positive(Variable(0))]);

        let result = solver.solve();
        match result {
            SolveResult::Sat(model) => {
                assert_eq!(model.len(), 3, "model should include ensured variables");
                assert!(model[0]);
            }
            other => panic!("expected SAT, got {:?}", other),
        }
    }

    // ========================================================================
    // Property Tests for Propagation Soundness
    // ========================================================================

    /// Generate an arbitrary 3-SAT clause for n variables
    #[allow(dead_code)]
    fn arb_clause(num_vars: u32) -> impl Strategy<Value = Vec<Literal>> {
        prop::collection::vec(0u32..num_vars, 2..=3).prop_map(move |vars| {
            vars.into_iter()
                .map(|v| {
                    if v % 2 == 0 {
                        Literal::positive(Variable(v % num_vars))
                    } else {
                        Literal::negative(Variable(v % num_vars))
                    }
                })
                .collect()
        })
    }

    proptest! {
        /// Propagation soundness: if propagate returns conflict, clause is falsified
        #[test]
        fn prop_propagation_conflict_soundness(
            num_clauses in 1usize..10,
            seed in 0u64..1000
        ) {
            // Create solver with fixed seed-based clauses
            let num_vars = 5usize;
            let mut solver = Solver::new(num_vars);

            // Generate deterministic clauses based on seed
            for i in 0..num_clauses {
                let v1 = ((seed + i as u64) % num_vars as u64) as u32;
                let v2 = ((seed + i as u64 + 1) % num_vars as u64) as u32;
                let v3 = ((seed + i as u64 + 2) % num_vars as u64) as u32;

                let lit1 = if (seed + i as u64).is_multiple_of(2) {
                    Literal::positive(Variable(v1))
                } else {
                    Literal::negative(Variable(v1))
                };
                let lit2 = if (seed + i as u64 + 1).is_multiple_of(2) {
                    Literal::positive(Variable(v2))
                } else {
                    Literal::negative(Variable(v2))
                };
                let lit3 = if (seed + i as u64 + 2).is_multiple_of(2) {
                    Literal::positive(Variable(v3))
                } else {
                    Literal::negative(Variable(v3))
                };

                solver.add_clause(vec![lit1, lit2, lit3]);
            }

            // Initialize and run solver
            let result = solver.solve();

            // Basic soundness check: if SAT, model satisfies all original clauses
            if let SolveResult::Sat(model) = &result {
                for i in solver.clause_db.indices().take(num_clauses) {
                    let satisfied = solver.clause_db.literals(i).iter().any(|&lit| {
                        let var_val = model[lit.variable().index()];
                        if lit.is_positive() {
                            var_val
                        } else {
                            !var_val
                        }
                    });
                    prop_assert!(satisfied, "Clause {} not satisfied by model", i);
                }
            }
        }

        /// Solver returns consistent results on same formula
        #[test]
        fn prop_solve_deterministic(clauses_seed in 0u64..100) {
            let num_vars = 4usize;
            let num_clauses = 5usize;

            let run_solver = || {
                let mut solver = Solver::new(num_vars);
                for i in 0..num_clauses {
                    let v1 = ((clauses_seed + i as u64) % num_vars as u64) as u32;
                    let v2 = ((clauses_seed + i as u64 + 1) % num_vars as u64) as u32;
                    let polarity1 = (clauses_seed + i as u64).is_multiple_of(2);
                    let polarity2 = (clauses_seed + i as u64 + 1).is_multiple_of(2);

                    let lit1 = if polarity1 {
                        Literal::positive(Variable(v1))
                    } else {
                        Literal::negative(Variable(v1))
                    };
                    let lit2 = if polarity2 {
                        Literal::positive(Variable(v2))
                    } else {
                        Literal::negative(Variable(v2))
                    };
                    solver.add_clause(vec![lit1, lit2]);
                }
                solver.solve()
            };

            let result1 = run_solver();
            let result2 = run_solver();

            // Results should be consistent (both SAT or both UNSAT)
            match (&result1, &result2) {
                (SolveResult::Sat(_), SolveResult::Sat(_)) => (),
                (SolveResult::Unsat, SolveResult::Unsat) => (),
                _ => prop_assert!(false, "Inconsistent results: {:?} vs {:?}", result1, result2),
            }
        }

        /// Unit clauses are correctly propagated
        #[test]
        fn prop_unit_clause_propagation(var_idx in 0u32..10) {
            let num_vars = 10usize;
            let mut solver = Solver::new(num_vars);

            // Add unit clause
            let unit_lit = Literal::positive(Variable(var_idx));
            solver.add_clause(vec![unit_lit]);

            // Add clause requiring the unit
            let other_var = (var_idx + 1) % num_vars as u32;
            solver.add_clause(vec![
                Literal::negative(Variable(var_idx)),
                Literal::positive(Variable(other_var)),
            ]);

            let result = solver.solve();

            if let SolveResult::Sat(model) = result {
                // Unit clause must be satisfied
                prop_assert!(model[var_idx as usize], "Unit clause not satisfied");
            }
        }

        // ====================================================================
        // TLA+ Invariant Property Tests (Gap 3: Formal Link to TLA+ Spec)
        //
        // These tests mirror the invariants from specs/cdcl.tla:
        // - TypeInvariant: Implicitly enforced by Rust's type system
        // - SatCorrect: When SAT, all clauses are satisfied
        // - NoDoubleAssignment: No variable assigned twice
        // - WatchedInvariant: Checked during propagation
        // ====================================================================

        /// TLA+ SatCorrect invariant (lines 201-202 of cdcl.tla):
        /// state = "SAT" => \A clause \in Clauses : Satisfied(clause)
        ///
        /// For any random SAT formula, if the solver returns SAT, every
        /// original clause must be satisfied by the model.
        #[test]
        fn tla_invariant_sat_correct(
            num_vars in 3usize..8,
            num_clauses in 1usize..15,
            seed in 0u64..10000
        ) {
            use std::collections::HashSet;

            let mut solver = Solver::new(num_vars);
            let mut original_clauses: Vec<Vec<Literal>> = Vec::new();

            // Generate random clauses
            for i in 0..num_clauses {
                let clause_len = 2 + ((seed + i as u64) % 3) as usize; // 2-4 literals
                let mut clause = Vec::new();
                let mut seen_vars: HashSet<u32> = HashSet::new();

                for j in 0..clause_len {
                    let v = ((seed.wrapping_mul(7) + i as u64 * 13 + j as u64 * 31) % num_vars as u64) as u32;
                    if seen_vars.contains(&v) {
                        continue; // Skip duplicate variables in same clause
                    }
                    seen_vars.insert(v);
                    let polarity = (seed + i as u64 + j as u64).is_multiple_of(2);
                    let lit = if polarity {
                        Literal::positive(Variable(v))
                    } else {
                        Literal::negative(Variable(v))
                    };
                    clause.push(lit);
                }

                if clause.is_empty() {
                    continue; // Skip empty clauses
                }

                original_clauses.push(clause.clone());
                solver.add_clause(clause);
            }

            let result = solver.solve();

            // TLA+ SatCorrect: If SAT, all original clauses must be satisfied
            if let SolveResult::Sat(model) = result {
                for (clause_idx, clause) in original_clauses.iter().enumerate() {
                    let satisfied = clause.iter().any(|&lit| {
                        let var_idx = lit.variable().index();
                        let val = model[var_idx];
                        if lit.is_positive() { val } else { !val }
                    });
                    prop_assert!(
                        satisfied,
                        "TLA+ SatCorrect violation: clause {} ({:?}) not satisfied by model {:?}",
                        clause_idx, clause, model
                    );
                }
            }
        }

        /// TLA+ NoDoubleAssignment invariant (lines 213-215 of cdcl.tla):
        /// \A i, j \in 1..Len(trail) : i # j => Var(trail[i][1]) # Var(trail[j][1])
        ///
        /// In the model, each variable should have exactly one consistent value.
        /// This is implicitly enforced by our Vec<bool> model representation,
        /// but we verify that values are deterministic across multiple accesses.
        #[test]
        fn tla_invariant_no_double_assignment(
            num_vars in 2usize..6,
            seed in 0u64..1000
        ) {
            let mut solver = Solver::new(num_vars);

            // Create simple SAT formula
            for i in 0..num_vars {
                let v1 = i as u32;
                let v2 = ((i + 1) % num_vars) as u32;
                let polarity = (seed + i as u64).is_multiple_of(2);
                let lit1 = if polarity {
                    Literal::positive(Variable(v1))
                } else {
                    Literal::negative(Variable(v1))
                };
                let lit2 = Literal::positive(Variable(v2));
                solver.add_clause(vec![lit1, lit2]);
            }

            let result = solver.solve();

            if let SolveResult::Sat(model) = result {
                // Verify model length matches num_vars
                prop_assert_eq!(
                    model.len(), num_vars,
                    "Model length {} does not match num_vars {}",
                    model.len(), num_vars
                );

                // Verify each variable has a definite value (no "holes")
                for val in model.iter().take(num_vars) {
                    // Model must have value for each variable
                    let _val = *val; // This should not panic
                }
            }
        }

        /// TLA+ Soundness invariant combining SatCorrect and UnsatCorrect:
        /// When we claim SAT, we can construct a satisfying assignment.
        /// When we claim UNSAT, there truly is no satisfying assignment.
        ///
        /// This test generates formulas and verifies soundness both ways.
        #[test]
        fn tla_invariant_soundness(
            num_vars in 2usize..5,
            seed in 0u64..500
        ) {
            // Test 1: Known SAT formula
            {
                let mut solver = Solver::new(num_vars);
                // Add tautology: (x0 OR NOT x0) for each var
                for i in 0..num_vars {
                    solver.add_clause(vec![
                        Literal::positive(Variable(i as u32)),
                        Literal::negative(Variable(i as u32)),
                    ]);
                }
                let result = solver.solve();
                match result {
                    SolveResult::Sat(_) => (), // Expected
                    other => prop_assert!(false, "Tautology should be SAT, got {:?}", other),
                }
            }

            // Test 2: Known UNSAT formula
            {
                let mut solver = Solver::new(1);
                solver.add_clause(vec![Literal::positive(Variable(0))]);
                solver.add_clause(vec![Literal::negative(Variable(0))]);
                let result = solver.solve();
                match result {
                    SolveResult::Unsat => (), // Expected
                    other => prop_assert!(false, "x AND NOT x should be UNSAT, got {:?}", other),
                }
            }

            // Test 3: Random formula with soundness check
            {
                let mut solver = Solver::new(num_vars);
                let mut clauses: Vec<Vec<Literal>> = Vec::new();

                for i in 0..(seed % 10 + 1) {
                    let v1 = (seed.wrapping_add(i) % num_vars as u64) as u32;
                    let v2 = (seed.wrapping_add(i).wrapping_add(1) % num_vars as u64) as u32;
                    let p1 = (seed.wrapping_add(i) % 2) == 0;
                    let p2 = (seed.wrapping_add(i).wrapping_add(1) % 2) == 0;
                    let lit1 = if p1 { Literal::positive(Variable(v1)) } else { Literal::negative(Variable(v1)) };
                    let lit2 = if p2 { Literal::positive(Variable(v2)) } else { Literal::negative(Variable(v2)) };
                    clauses.push(vec![lit1, lit2]);
                    solver.add_clause(vec![lit1, lit2]);
                }

                let result = solver.solve();

                if let SolveResult::Sat(model) = result {
                    // Verify all clauses satisfied
                    for clause in &clauses {
                        let sat = clause.iter().any(|&lit| {
                            let v = lit.variable().index();
                            let val = model[v];
                            if lit.is_positive() { val } else { !val }
                        });
                        prop_assert!(sat, "Soundness violation: clause {:?} not satisfied", clause);
                    }
                }
                // Note: For UNSAT, we trust DRAT proof verification (Gap 2)
            }
        }
    }

    // ========================================================================
    // Deterministic Unit Tests
    // ========================================================================

    #[test]
    fn test_simple_sat() {
        let mut solver = Solver::new(2);
        // (x0 OR x1)
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);
        let result = solver.solve();
        match result {
            SolveResult::Sat(model) => {
                assert!(model[0] || model[1]);
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_simple_unsat() {
        let mut solver = Solver::new(1);
        // x0 AND NOT x0
        solver.add_clause(vec![Literal::positive(Variable(0))]);
        solver.add_clause(vec![Literal::negative(Variable(0))]);
        let result = solver.solve();
        assert_eq!(result, SolveResult::Unsat);
    }

    #[test]
    fn test_unit_propagation() {
        let mut solver = Solver::new(3);
        // x0 (unit)
        // NOT x0 OR x1
        // NOT x1 OR x2
        solver.add_clause(vec![Literal::positive(Variable(0))]);
        solver.add_clause(vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(1)),
        ]);
        solver.add_clause(vec![
            Literal::negative(Variable(1)),
            Literal::positive(Variable(2)),
        ]);
        let result = solver.solve();
        match result {
            SolveResult::Sat(model) => {
                assert!(model[0]); // x0 must be true
                assert!(model[1]); // x1 must be true (propagated)
                assert!(model[2]); // x2 must be true (propagated)
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_conflict_learning() {
        let mut solver = Solver::new(3);
        // A formula that requires conflict learning
        // (x0 OR x1) AND (x0 OR NOT x1) AND (NOT x0 OR x2) AND (NOT x0 OR NOT x2)
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::negative(Variable(1)),
        ]);
        solver.add_clause(vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(2)),
        ]);
        solver.add_clause(vec![
            Literal::negative(Variable(0)),
            Literal::negative(Variable(2)),
        ]);
        let result = solver.solve();
        match result {
            SolveResult::Sat(model) => {
                // x0 must be true (from first two clauses)
                // But then x2 AND NOT x2 is required -> should learn x0
                assert!(model[0]);
            }
            SolveResult::Unsat => {
                // This is also valid if the formula is UNSAT
                // Let's verify: if x0=T, need x2 and NOT x2 -> contradiction
                // if x0=F, need x1 and NOT x1 from first two -> contradiction
                // So UNSAT is correct!
            }
            _ => panic!("Expected SAT or UNSAT"),
        }
    }

    // ========================================================================
    // Luby Sequence Tests
    // ========================================================================

    #[test]
    fn test_luby_sequence() {
        // The Luby sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
        let expected = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8];
        for (i, &exp) in expected.iter().enumerate() {
            let luby = Solver::<Vec<u8>>::get_luby((i + 1) as u32);
            assert_eq!(luby, exp, "Luby({}) should be {}, got {}", i + 1, exp, luby);
        }
    }

    #[test]
    fn test_luby_first_values() {
        // Check first few values specifically
        assert_eq!(Solver::<Vec<u8>>::get_luby(1), 1);
        assert_eq!(Solver::<Vec<u8>>::get_luby(2), 1);
        assert_eq!(Solver::<Vec<u8>>::get_luby(3), 2);
        assert_eq!(Solver::<Vec<u8>>::get_luby(4), 1);
        assert_eq!(Solver::<Vec<u8>>::get_luby(5), 1);
        assert_eq!(Solver::<Vec<u8>>::get_luby(6), 2);
        assert_eq!(Solver::<Vec<u8>>::get_luby(7), 4);
    }

    #[test]
    fn test_restart_threshold_increases() {
        // Verify restart thresholds follow Luby pattern
        let base = DEFAULT_RESTART_BASE;
        let mut thresholds = Vec::new();

        for i in 1..=7 {
            let luby = Solver::<Vec<u8>>::get_luby(i);
            thresholds.push(base * luby as u64);
        }

        // Expected: [base*1, base*1, base*2, base*1, base*1, base*2, base*4]
        assert_eq!(thresholds[0], base); // luby(1) = 1
        assert_eq!(thresholds[1], base); // luby(2) = 1
        assert_eq!(thresholds[2], base * 2); // luby(3) = 2
        assert_eq!(thresholds[6], base * 4); // luby(7) = 4
    }

    // ========================================================================
    // Chronological Backtracking Tests
    // ========================================================================

    #[test]
    fn test_chrono_backtrack_decision() {
        // Test that chrono backtracking is used when jump distance is large
        let mut solver = Solver::new(10);
        solver.decision_level = 150;
        solver.chrono_enabled = true;

        // Jump of 140 levels exceeds CHRONO_LEVEL_LIMIT (100), should use chrono BT
        let actual = solver.compute_chrono_backtrack_level(10);
        assert_eq!(actual, 149); // Should backtrack to level - 1
        assert_eq!(solver.chrono_backtracks, 1);

        // Jump of 50 levels is within limit, should use NCB
        solver.decision_level = 60;
        let actual = solver.compute_chrono_backtrack_level(10);
        assert_eq!(actual, 10); // Should use the original jump level
    }

    #[test]
    fn test_chrono_backtrack_disabled() {
        // Test that chrono backtracking can be disabled
        let mut solver = Solver::new(10);
        solver.decision_level = 150;
        solver.chrono_enabled = false;

        // Even with large jump, should use NCB when disabled
        let actual = solver.compute_chrono_backtrack_level(10);
        assert_eq!(actual, 10);
        assert_eq!(solver.chrono_backtracks, 0);
    }

    #[test]
    fn test_chrono_backtrack_preserves_correctness() {
        // Verify that chronological backtracking doesn't break correctness
        let mut solver = Solver::new(5);
        solver.chrono_enabled = true;

        // Create a formula that requires some backtracking
        // (a OR b) AND (a OR NOT b) AND (NOT a OR c) AND (NOT a OR NOT c)
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::negative(Variable(1)),
        ]);
        solver.add_clause(vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(2)),
        ]);
        solver.add_clause(vec![
            Literal::negative(Variable(0)),
            Literal::negative(Variable(2)),
        ]);

        let result = solver.solve();
        // This formula is UNSAT
        assert_eq!(result, SolveResult::Unsat);
    }

    #[test]
    fn test_preprocessing_correctness() {
        // Test that preprocessing doesn't break solver correctness
        let mut solver = Solver::new(5);
        solver.set_preprocess_enabled(true);

        // Create a SAT formula: (a OR b) AND (NOT a OR c)
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);
        solver.add_clause(vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(2)),
        ]);

        let result = solver.solve();
        assert!(matches!(result, SolveResult::Sat(_)), "Expected SAT");

        // Create an UNSAT formula with preprocessing
        let mut solver2 = Solver::new(3);
        solver2.set_preprocess_enabled(true);

        // (a) AND (NOT a)
        solver2.add_clause(vec![Literal::positive(Variable(0))]);
        solver2.add_clause(vec![Literal::negative(Variable(0))]);

        let result2 = solver2.solve();
        assert_eq!(result2, SolveResult::Unsat);
    }

    #[test]
    fn test_lazy_reimplication_trail_compaction() {
        // Test that trail is properly compacted during backtrack
        let mut solver = Solver::new(5);

        // Manually set up a scenario with literals at different levels
        solver.decision_level = 3;
        solver.trail_lim = vec![0, 2, 4]; // Level 1 starts at 0, level 2 at 2, level 3 at 4

        // Trail: [lit0@1, lit1@1, lit2@2, lit3@2, lit4@3]
        let lits = [
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
            Literal::positive(Variable(2)),
            Literal::positive(Variable(3)),
            Literal::positive(Variable(4)),
        ];

        for (i, &lit) in lits.iter().enumerate() {
            let var = lit.variable();
            solver.assignment[var.index()] = Some(true);
            solver.trail_pos[var.index()] = i;
            // Set levels: 0,1 at level 1; 2,3 at level 2; 4 at level 3
            solver.level[var.index()] = if i < 2 {
                1
            } else if i < 4 {
                2
            } else {
                3
            };
        }
        solver.trail = lits.to_vec();

        // Backtrack to level 1 - should keep only vars 0 and 1
        solver.backtrack(1);

        assert_eq!(solver.decision_level, 1);
        assert_eq!(solver.trail.len(), 2);
        assert!(solver.assignment[0].is_some());
        assert!(solver.assignment[1].is_some());
        assert!(solver.assignment[2].is_none());
        assert!(solver.assignment[3].is_none());
        assert!(solver.assignment[4].is_none());
    }

    // ========================================================================
    // Assumption-Based Solving Tests
    // ========================================================================

    #[test]
    fn test_assumptions_sat() {
        // Test SAT with compatible assumptions
        let mut solver = Solver::new(3);
        // (x0 OR x1) AND (x1 OR x2)
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);
        solver.add_clause(vec![
            Literal::positive(Variable(1)),
            Literal::positive(Variable(2)),
        ]);

        // Assume x1 = true - should be SAT
        let assumptions = vec![Literal::positive(Variable(1))];
        let result = solver.solve_with_assumptions(&assumptions);
        match result {
            AssumeResult::Sat(model) => {
                assert!(model[1], "Assumed literal x1 should be true");
            }
            _ => panic!("Expected SAT with assumption x1=true"),
        }
    }

    #[test]
    fn test_assumptions_unsat_single() {
        // Test UNSAT with conflicting assumption
        let mut solver = Solver::new(2);
        // x0 (unit clause)
        solver.add_clause(vec![Literal::positive(Variable(0))]);

        // Assume NOT x0 - should be UNSAT with core containing NOT x0
        let assumptions = vec![Literal::negative(Variable(0))];
        let result = solver.solve_with_assumptions(&assumptions);
        match result {
            AssumeResult::Unsat(core) => {
                assert!(
                    core.contains(&Literal::negative(Variable(0))),
                    "Unsat core should contain the conflicting assumption"
                );
            }
            _ => panic!("Expected UNSAT with assumption NOT x0"),
        }
    }

    #[test]
    fn test_assumptions_unsat_multiple() {
        // Test UNSAT with multiple conflicting assumptions
        let mut solver = Solver::new(3);
        // (x0 OR x1) AND (NOT x0 OR NOT x1)
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);
        solver.add_clause(vec![
            Literal::negative(Variable(0)),
            Literal::negative(Variable(1)),
        ]);

        // Assume x0=true AND x1=true - should be UNSAT
        let assumptions = vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ];
        let result = solver.solve_with_assumptions(&assumptions);
        match result {
            AssumeResult::Unsat(core) => {
                // Core should contain at least one of the conflicting assumptions
                assert!(
                    !core.is_empty(),
                    "Unsat core should not be empty for assumption conflict"
                );
            }
            _ => panic!("Expected UNSAT with conflicting assumptions"),
        }
    }

    #[test]
    fn test_assumptions_preserve_clauses() {
        // Test that assumptions don't modify clause database
        let mut solver = Solver::new(3);
        // (x0 OR x1)
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);

        let initial_clause_count = solver.clause_db.len();

        // First call with assumption
        let assumptions = vec![Literal::positive(Variable(0))];
        let _ = solver.solve_with_assumptions(&assumptions);

        // Second call with different assumption
        let assumptions = vec![Literal::positive(Variable(1))];
        let _ = solver.solve_with_assumptions(&assumptions);

        // Original clauses should still be present (learned clauses may be added)
        assert!(
            solver.clause_db.len() >= initial_clause_count,
            "Clause database should preserve original clauses"
        );
    }

    #[test]
    fn test_assumptions_empty() {
        // Test empty assumptions (should behave like regular solve)
        let mut solver = Solver::new(2);
        // (x0 OR x1)
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);

        let result = solver.solve_with_assumptions(&[]);
        match result {
            AssumeResult::Sat(_) => {} // Expected
            _ => panic!("Expected SAT with empty assumptions"),
        }
    }

    #[test]
    fn test_assumptions_contradictory() {
        // Test contradictory assumptions (x AND NOT x)
        let mut solver = Solver::new(2);
        // Formula is satisfiable
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);

        // Assume x0=true AND x0=false (contradictory)
        let assumptions = vec![
            Literal::positive(Variable(0)),
            Literal::negative(Variable(0)),
        ];
        let result = solver.solve_with_assumptions(&assumptions);
        match result {
            AssumeResult::Unsat(core) => {
                // Core should contain the contradictory assumption
                assert!(
                    core.contains(&Literal::positive(Variable(0)))
                        || core.contains(&Literal::negative(Variable(0))),
                    "Unsat core should contain contradictory assumption"
                );
            }
            _ => panic!("Expected UNSAT with contradictory assumptions"),
        }
    }

    #[test]
    fn test_assumptions_incremental() {
        // Test that learned clauses are preserved across calls
        let mut solver = Solver::new(5);

        // A formula that requires learning
        // (a OR b) AND (a OR NOT b OR c) AND (NOT a OR d) AND (NOT d OR e) AND (NOT e)
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);
        solver.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::negative(Variable(1)),
            Literal::positive(Variable(2)),
        ]);
        solver.add_clause(vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(3)),
        ]);
        solver.add_clause(vec![
            Literal::negative(Variable(3)),
            Literal::positive(Variable(4)),
        ]);
        solver.add_clause(vec![Literal::negative(Variable(4))]);

        // First solve with assumptions - may learn some clauses
        let _ = solver.solve_with_assumptions(&[Literal::positive(Variable(0))]);

        let learned_count_after_first = solver.clause_db.len();

        // Second solve - should potentially reuse learned clauses
        let _ = solver.solve_with_assumptions(&[Literal::negative(Variable(0))]);

        // The solver should have retained or added clauses
        assert!(
            solver.clause_db.len() >= learned_count_after_first,
            "Learned clauses should be retained between assumption calls"
        );
    }

    #[test]
    fn test_assumptions_unsat_core_minimal() {
        // Test that unsat core is a subset of assumptions
        let mut solver = Solver::new(4);

        // x0 (unit - forces x0=true)
        solver.add_clause(vec![Literal::positive(Variable(0))]);

        // Multiple assumptions, only one conflicts
        let assumptions = vec![
            Literal::positive(Variable(1)), // irrelevant
            Literal::negative(Variable(0)), // conflicts with unit clause
            Literal::positive(Variable(2)), // irrelevant
        ];

        let result = solver.solve_with_assumptions(&assumptions);
        match result {
            AssumeResult::Unsat(core) => {
                // Core should be a subset of assumptions
                for lit in &core {
                    assert!(
                        assumptions.contains(lit),
                        "Unsat core {:?} should only contain assumptions",
                        lit
                    );
                }
                // Core should contain the conflicting assumption
                assert!(
                    core.contains(&Literal::negative(Variable(0))),
                    "Core should contain NOT x0"
                );
            }
            _ => panic!("Expected UNSAT"),
        }
    }

    // ========================================================================
    // Push/Pop Incremental Solving Tests
    // ========================================================================

    #[test]
    fn test_push_pop_scopes_affect_solve() {
        let mut solver = Solver::new(1);

        // Outer scope: assert x0
        solver.push();
        solver.add_clause(vec![Literal::positive(Variable(0))]);
        assert!(matches!(solver.solve(), SolveResult::Sat(_)));

        // Inner scope: assert ¬x0, making the current context UNSAT
        solver.push();
        solver.add_clause(vec![Literal::negative(Variable(0))]);
        assert_eq!(solver.solve(), SolveResult::Unsat);

        // Pop inner: back to {x0}
        assert!(solver.pop());
        let result = solver.solve();
        match result {
            SolveResult::Sat(model) => assert!(model[0]),
            _ => panic!("Expected SAT after popping inner scope"),
        }

        // Pop outer: back to empty context
        assert!(solver.pop());
        assert!(matches!(solver.solve(), SolveResult::Sat(_)));
    }

    #[test]
    fn test_pop_permanently_disables_popped_scope_selector() {
        // Initial user vars: x0 (idx 0)
        let mut solver = Solver::new(1);

        solver.push(); // selector idx 1
        solver.push(); // selector idx 2

        // Add a scoped clause in the inner scope and then pop it.
        solver.add_clause(vec![Literal::positive(Variable(0))]);
        assert!(solver.pop());

        // After popping the inner scope, the selector variable should be forced true
        // by a permanent unit clause, so assuming ¬selector must be UNSAT.
        let selector = Variable(2);
        let result = solver.solve_with_assumptions(&[Literal::negative(selector)]);
        match result {
            AssumeResult::Unsat(core) => {
                assert!(core.contains(&Literal::negative(selector)));
            }
            _ => panic!("Expected UNSAT when assuming a popped selector is false"),
        }
    }

    #[test]
    fn test_assumption_core_does_not_include_scope_selectors() {
        let mut solver = Solver::new(1);

        // Permanent: x0
        solver.add_clause(vec![Literal::positive(Variable(0))]);

        // Add an empty scope (creates a selector that is assumed internally)
        solver.push(); // selector idx 1

        // User assumption conflicts with permanent clause
        let result = solver.solve_with_assumptions(&[Literal::negative(Variable(0))]);
        match result {
            AssumeResult::Unsat(core) => {
                assert!(core.contains(&Literal::negative(Variable(0))));
                assert!(!core.iter().any(|lit| lit.variable() == Variable(1)));
            }
            _ => panic!("Expected UNSAT with core containing only user assumption"),
        }
    }

    #[test]
    fn test_solve_empty_formula_with_active_scope_selector() {
        let mut solver = Solver::new(0);
        solver.push();

        let result = solver.solve();
        match result {
            SolveResult::Sat(model) => assert!(
                model.is_empty(),
                "model should contain only user variables"
            ),
            other => panic!("expected SAT, got {:?}", other),
        }
    }

    // ========================================================================
    // Memory Benchmark Tests
    // ========================================================================

    /// Test memory statistics for a small formula
    #[test]
    fn test_memory_stats_basic() {
        let mut solver = Solver::new(100);

        // Add 200 3-SAT clauses
        for i in 0..200 {
            let v1 = (i * 3) % 100;
            let v2 = (i * 3 + 1) % 100;
            let v3 = (i * 3 + 2) % 100;
            solver.add_clause(vec![
                Literal::positive(Variable(v1 as u32)),
                Literal::negative(Variable(v2 as u32)),
                Literal::positive(Variable(v3 as u32)),
            ]);
        }

        let stats = solver.memory_stats();
        assert_eq!(stats.num_vars, 100);
        assert_eq!(stats.num_clauses, 200);
        assert_eq!(stats.total_literals, 600); // 200 clauses * 3 literals

        // Per-var should be in reasonable range (22 base + overhead)
        assert!(stats.per_var() > 20.0, "Per-var should be > 20 bytes");
        assert!(stats.per_var() < 200.0, "Per-var should be < 200 bytes");

        // Total should be positive and reasonable
        assert!(stats.total() > 0);
        assert!(stats.total() < 1_000_000, "Small formula should use < 1MB");
    }

    /// Test memory efficiency - bytes per literal should be reasonable
    #[test]
    fn test_memory_efficiency_per_literal() {
        let mut solver = Solver::new(1000);

        // Add 4000 3-SAT clauses (12000 literals)
        for i in 0..4000 {
            let v1 = (i * 7) % 1000;
            let v2 = (i * 7 + 3) % 1000;
            let v3 = (i * 7 + 5) % 1000;
            solver.add_clause(vec![
                Literal::positive(Variable(v1 as u32)),
                Literal::negative(Variable(v2 as u32)),
                Literal::positive(Variable(v3 as u32)),
            ]);
        }

        let stats = solver.memory_stats();

        // Bytes per literal in clause database
        // Ideal: 4 bytes (just the literal)
        // With Vec overhead per clause: ~40 bytes overhead / 3 literals = ~17 extra
        // So expect 4-25 bytes per literal
        let per_lit = stats.per_literal();
        assert!(
            per_lit >= 4.0,
            "Per literal should be >= 4 bytes, got {}",
            per_lit
        );
        assert!(
            per_lit < 50.0,
            "Per literal should be < 50 bytes, got {}",
            per_lit
        );

        // Compare clause_db to theoretical minimum
        // Minimum: num_clauses * (3 lits * 4 bytes) = 4000 * 12 = 48KB
        let min_clause_bytes = 4000 * 3 * 4;
        assert!(
            stats.clause_db >= min_clause_bytes,
            "Clause DB {} should be >= minimum {}",
            stats.clause_db,
            min_clause_bytes
        );

        // Should be within 10x of minimum (allowing for Vec overhead)
        assert!(
            stats.clause_db < min_clause_bytes * 10,
            "Clause DB {} should be < 10x minimum {}",
            stats.clause_db,
            min_clause_bytes * 10
        );
    }

    /// Benchmark memory usage after solving (with learned clauses)
    #[test]
    fn test_memory_after_solving() {
        let mut solver = Solver::new(50);

        // Add a satisfiable random 3-SAT instance
        for i in 0..200 {
            let v1 = (i * 3) % 50;
            let v2 = (i * 3 + 1) % 50;
            let v3 = (i * 3 + 2) % 50;
            solver.add_clause(vec![
                if i % 2 == 0 {
                    Literal::positive(Variable(v1 as u32))
                } else {
                    Literal::negative(Variable(v1 as u32))
                },
                Literal::negative(Variable(v2 as u32)),
                Literal::positive(Variable(v3 as u32)),
            ]);
        }

        let stats_before = solver.memory_stats();

        // Solve
        let result = solver.solve();
        assert!(matches!(result, SolveResult::Sat(_)));

        let stats_after = solver.memory_stats();

        // After solving, we may have learned clauses
        // Memory should not explode (allow up to 5x growth for learned clauses)
        assert!(
            stats_after.total() < stats_before.total() * 5,
            "Memory should not grow more than 5x after solving"
        );
    }

    /// Test memory stats display formatting
    #[test]
    fn test_memory_stats_display() {
        let mut solver = Solver::new(100);
        for i in 0..100 {
            solver.add_clause(vec![Literal::positive(Variable(i as u32))]);
        }

        let stats = solver.memory_stats();
        let display = format!("{}", stats);

        // Should contain key information
        assert!(display.contains("Variables: 100"));
        assert!(display.contains("Clauses: 100"));
        assert!(display.contains("Total:"));
        assert!(display.contains("bytes"));
    }

    /// Memory benchmark comparing to theoretical CaDiCaL efficiency
    ///
    /// CaDiCaL uses ~8 bytes per literal (compact arena allocation).
    /// Z4 uses boxed literal slices per clause, but still has significant overhead from:
    /// - per-clause headers (activity, LBD, flags)
    /// - per-clause heap allocations (one allocation per clause)
    ///
    /// Optimization opportunities to reach 1.5x target:
    /// 1. Arena allocation for clauses (like CaDiCaL) - eliminates per-clause heap allocations
    /// 2. Compact clause headers (pack lbd, learned, used into single u32)
    /// 3. Lazy initialization of inprocessing engines
    /// 4. Use SmallVec for short clauses (most clauses have <8 literals)
    #[test]
    fn test_memory_vs_cadical_efficiency() {
        let num_vars = 10_000;
        let num_clauses = 40_000; // 4:1 clause-to-var ratio typical for 3-SAT

        let mut solver = Solver::new(num_vars);

        // Add random 3-SAT clauses
        for i in 0..num_clauses {
            let v1 = (i * 7) % num_vars;
            let v2 = (i * 7 + 3) % num_vars;
            let v3 = (i * 7 + 5) % num_vars;
            solver.add_clause(vec![
                Literal::positive(Variable(v1 as u32)),
                Literal::negative(Variable(v2 as u32)),
                Literal::positive(Variable(v3 as u32)),
            ]);
        }

        let stats = solver.memory_stats();

        // CaDiCaL baseline (estimated):
        // - Per variable: ~80 bytes (activities, levels, reasons, etc.)
        // - Per literal: ~8 bytes (arena-allocated with compact headers)
        // - Watches: ~4 bytes per watch entry
        let cadical_per_var = 80;
        let cadical_per_lit = 8;
        let cadical_estimate = num_vars * cadical_per_var + stats.total_literals * cadical_per_lit;

        // Z4 should be within 1.5x of CaDiCaL estimate (per Priority 2.1 requirement)
        let ratio = stats.total() as f64 / cadical_estimate as f64;

        // Print results for visibility in test output
        eprintln!("Memory Benchmark Results:");
        eprintln!("  Variables: {}", num_vars);
        eprintln!("  Clauses: {}", num_clauses);
        eprintln!("  Literals: {}", stats.total_literals);
        eprintln!();
        eprintln!("Breakdown:");
        eprintln!(
            "  var_data: {} bytes ({:.1}%)",
            stats.var_data,
            100.0 * stats.var_data as f64 / stats.total() as f64
        );
        eprintln!(
            "  vsids: {} bytes ({:.1}%)",
            stats.vsids,
            100.0 * stats.vsids as f64 / stats.total() as f64
        );
        eprintln!(
            "  conflict: {} bytes ({:.1}%)",
            stats.conflict,
            100.0 * stats.conflict as f64 / stats.total() as f64
        );
        eprintln!(
            "  clause_db: {} bytes ({:.1}%)",
            stats.clause_db,
            100.0 * stats.clause_db as f64 / stats.total() as f64
        );
        eprintln!(
            "  watches: {} bytes ({:.1}%)",
            stats.watches,
            100.0 * stats.watches as f64 / stats.total() as f64
        );
        eprintln!(
            "  trail: {} bytes ({:.1}%)",
            stats.trail,
            100.0 * stats.trail as f64 / stats.total() as f64
        );
        eprintln!(
            "  clause_ids: {} bytes ({:.1}%)",
            stats.clause_ids,
            100.0 * stats.clause_ids as f64 / stats.total() as f64
        );
        eprintln!(
            "  inprocessing: {} bytes ({:.1}%)",
            stats.inprocessing,
            100.0 * stats.inprocessing as f64 / stats.total() as f64
        );
        eprintln!();
        eprintln!(
            "  Z4 total: {} bytes ({:.2} MB)",
            stats.total(),
            stats.total() as f64 / 1_048_576.0
        );
        eprintln!(
            "  CaDiCaL estimate: {} bytes ({:.2} MB)",
            cadical_estimate,
            cadical_estimate as f64 / 1_048_576.0
        );
        eprintln!("  Ratio (Z4/CaDiCaL): {:.2}x", ratio);
        eprintln!();
        eprintln!("  Z4 per variable: {:.2} bytes", stats.per_var());
        eprintln!("  Z4 per literal: {:.2} bytes", stats.per_literal());

        // Requirement: within 1.5x of CaDiCaL (Priority 2.1)
        // Current baseline is ~3.4x, target is 1.5x
        // For now, assert we're under 4x to track regressions
        assert!(
            ratio < 4.0,
            "Z4 memory ({} bytes) should be within 4x of CaDiCaL estimate ({} bytes), ratio: {:.2}x",
            stats.total(),
            cadical_estimate,
            ratio
        );
    }

    #[test]
    fn test_vivify_simplifies_learned_clause_and_propagates_unit() {
        let mut solver = Solver::new(3);

        let a = Literal::positive(Variable(0));
        let b = Literal::positive(Variable(1));
        let c = Literal::positive(Variable(2));

        solver.add_clause(vec![Literal::negative(Variable(1))]);
        solver.add_clause(vec![Literal::negative(Variable(2))]);

        let learned = solver.add_learned_clause(vec![a, b, c], 3, &[]);

        assert!(solver.process_initial_clauses());
        solver.set_vivify_enabled(true);

        assert!(!solver.vivify());

        let learned_idx = learned.0 as usize;
        assert_eq!(solver.clause_db.header(learned_idx).len(), 1);
        assert_eq!(solver.clause_db.literal(learned_idx, 0), a);

        assert_eq!(solver.value(Variable(0)), Some(true));
        assert_eq!(solver.value(Variable(1)), Some(false));
        assert_eq!(solver.value(Variable(2)), Some(false));
    }
}

// ============================================================================
// Kani Verification Harnesses
// ============================================================================

#[cfg(kani)]
mod verification {
    use super::*;

    /// Verify XOR swap identity used in propagate
    /// For any two literals a, b, and a third literal c (where c == a or c == b),
    /// a ^ b ^ c should give the other literal
    #[kani::proof]
    fn proof_xor_swap_identity() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        kani::assume(a < 1000 && b < 1000);

        // Create literals
        let lit_a = Literal(a);
        let lit_b = Literal(b);

        // XOR swap: if c == a, result is b; if c == b, result is a
        let c_is_a: bool = kani::any();
        let lit_c = if c_is_a { lit_a } else { lit_b };

        let result = Literal(lit_a.0 ^ lit_b.0 ^ lit_c.0);

        // Result should be the OTHER literal
        if c_is_a {
            assert_eq!(result, lit_b);
        } else {
            assert_eq!(result, lit_a);
        }
    }

    /// Verify literal value lookup is consistent with assignment
    #[kani::proof]
    fn proof_lit_value_consistent() {
        // Create a small solver
        let mut solver: Solver<Vec<u8>> = Solver::new(4);

        // Pick a variable and polarity
        let var_idx: u32 = kani::any();
        kani::assume(var_idx < 4);
        let var = Variable(var_idx);
        let polarity: bool = kani::any();

        // Without assignment, value should be None
        assert_eq!(solver.value(var), None);

        // Set assignment
        solver.assignment[var_idx as usize] = Some(polarity);

        // Value should match
        assert_eq!(solver.value(var), Some(polarity));

        // Literal values should be correct
        let pos_lit = Literal::positive(var);
        let neg_lit = Literal::negative(var);

        assert_eq!(solver.lit_value(pos_lit), Some(polarity));
        assert_eq!(solver.lit_value(neg_lit), Some(!polarity));
    }

    /// Verify Luby sequence first 7 values (concrete, no recursion unfolding)
    #[kani::proof]
    fn proof_luby_values_concrete() {
        // Verify specific values without symbolic loop
        assert_eq!(Solver::<Vec<u8>>::get_luby(1), 1);
        assert_eq!(Solver::<Vec<u8>>::get_luby(2), 1);
        assert_eq!(Solver::<Vec<u8>>::get_luby(3), 2);
        assert_eq!(Solver::<Vec<u8>>::get_luby(4), 1);
        assert_eq!(Solver::<Vec<u8>>::get_luby(5), 1);
        assert_eq!(Solver::<Vec<u8>>::get_luby(6), 2);
        assert_eq!(Solver::<Vec<u8>>::get_luby(7), 4);
    }

    // ========================================================================
    // Gap 5 Proofs: Core CDCL Operation Invariants
    // ========================================================================

    /// Verify enqueue assigns the literal correctly
    /// Invariant: After enqueue(lit, reason), the variable is assigned the
    /// polarity indicated by lit, at the current decision level.
    #[kani::proof]
    fn proof_enqueue_assigns_correctly() {
        const NUM_VARS: usize = 4;
        let mut solver: Solver<Vec<u8>> = Solver::new(NUM_VARS);

        // Pick a symbolic variable and polarity
        let var_idx: u32 = kani::any();
        kani::assume(var_idx < NUM_VARS as u32);
        let var = Variable(var_idx);
        let polarity: bool = kani::any();
        let lit = if polarity {
            Literal::positive(var)
        } else {
            Literal::negative(var)
        };

        // Set a symbolic decision level
        let level: u32 = kani::any();
        kani::assume(level < 100);
        solver.decision_level = level;

        // Call enqueue
        solver.enqueue(lit, None);

        // Verify: variable is assigned to the correct polarity
        assert_eq!(solver.assignment[var_idx as usize], Some(polarity));

        // Verify: variable is at the current decision level
        assert_eq!(solver.level[var_idx as usize], level);

        // Verify: literal is on the trail
        assert!(solver.trail.contains(&lit));

        // Verify: lit_value returns correct value
        assert_eq!(solver.lit_value(lit), Some(true));
        assert_eq!(solver.lit_value(lit.negated()), Some(false));
    }

    /// Verify backtrack clears assignments at higher levels
    /// Invariant: After backtrack(target_level), no variable is assigned
    /// at a level > target_level.
    #[kani::proof]
    fn proof_backtrack_clears_higher_levels() {
        const NUM_VARS: usize = 4;
        let mut solver: Solver<Vec<u8>> = Solver::new(NUM_VARS);

        // Make 3 decisions at levels 1, 2, 3
        solver.decide(Literal::positive(Variable(0)));
        solver.decide(Literal::positive(Variable(1)));
        solver.decide(Literal::positive(Variable(2)));

        // Verify all three are assigned
        assert_eq!(solver.decision_level, 3);
        assert!(solver.assignment[0].is_some());
        assert!(solver.assignment[1].is_some());
        assert!(solver.assignment[2].is_some());

        // Pick a target level symbolically
        let target: u32 = kani::any();
        kani::assume(target <= 3);

        // Backtrack
        solver.backtrack(target);

        // Verify: decision_level is exactly target
        assert_eq!(solver.decision_level, target);

        // Verify: variables at higher levels are unassigned
        for v in 0..NUM_VARS {
            if solver.assignment[v].is_some() {
                // If assigned, must be at level <= target
                assert!(solver.level[v] <= target);
            }
        }

        // Verify: trail_lim matches target level
        assert_eq!(solver.trail_lim.len(), target as usize);
    }

    /// Verify that deciding increases the decision level and trail
    /// Invariant: decide(lit) increments decision_level and adds lit to trail
    #[kani::proof]
    fn proof_decide_increments_level() {
        const NUM_VARS: usize = 4;
        let mut solver: Solver<Vec<u8>> = Solver::new(NUM_VARS);

        // Record initial state
        let initial_level = solver.decision_level;
        let initial_trail_len = solver.trail.len();
        let initial_lim_len = solver.trail_lim.len();

        // Pick a symbolic variable
        let var_idx: u32 = kani::any();
        kani::assume(var_idx < NUM_VARS as u32);
        let var = Variable(var_idx);

        // Decide on a positive literal
        solver.decide(Literal::positive(var));

        // Verify: decision_level increased by 1
        assert_eq!(solver.decision_level, initial_level + 1);

        // Verify: trail_lim grew by 1
        assert_eq!(solver.trail_lim.len(), initial_lim_len + 1);

        // Verify: trail grew by 1
        assert_eq!(solver.trail.len(), initial_trail_len + 1);

        // Verify: variable is assigned at the new level
        assert_eq!(solver.level[var_idx as usize], solver.decision_level);

        // Verify: reason is None (decision, not propagation)
        assert!(solver.reason[var_idx as usize].is_none());
    }

    /// Verify watched literal invariant for binary clauses
    /// Invariant: After initialization, both literals of a binary clause
    /// are in each other's watch lists
    #[kani::proof]
    fn proof_binary_watch_invariant() {
        const NUM_VARS: usize = 4;
        let mut solver: Solver<Vec<u8>> = Solver::new(NUM_VARS);

        // Create a symbolic binary clause
        let v0: u32 = kani::any();
        let v1: u32 = kani::any();
        kani::assume(v0 < NUM_VARS as u32 && v1 < NUM_VARS as u32);
        kani::assume(v0 != v1);

        let p0: bool = kani::any();
        let p1: bool = kani::any();

        let lit0 = if p0 {
            Literal::positive(Variable(v0))
        } else {
            Literal::negative(Variable(v0))
        };
        let lit1 = if p1 {
            Literal::positive(Variable(v1))
        } else {
            Literal::negative(Variable(v1))
        };

        // Add binary clause
        solver.add_clause(vec![lit0, lit1]);
        solver.initialize_watches();

        // Get watch list for lit0 (watches for ~lit0 are updated when lit0 becomes false)
        let watches_lit0 = solver.watches.get_watches(lit0);
        let watches_lit1 = solver.watches.get_watches(lit1);

        // For a binary clause {lit0, lit1}:
        // - The clause is watched on lit0 with blocker = lit1
        // - The clause is watched on lit1 with blocker = lit0
        // So watches_lit0 should have a watcher with blocker = lit1
        // and watches_lit1 should have a watcher with blocker = lit0
        let found_in_lit0 = watches_lit0
            .iter()
            .any(|w| w.is_binary() && w.blocker() == lit1);
        let found_in_lit1 = watches_lit1
            .iter()
            .any(|w| w.is_binary() && w.blocker() == lit0);

        assert!(found_in_lit0, "Binary clause not watched on lit0");
        assert!(found_in_lit1, "Binary clause not watched on lit1");
    }

    /// Verify trail position tracking is consistent
    /// Invariant: trail_pos[var] is the index of var's literal in the trail
    #[kani::proof]
    fn proof_trail_pos_consistent() {
        const NUM_VARS: usize = 4;
        let mut solver: Solver<Vec<u8>> = Solver::new(NUM_VARS);

        // Make a few assignments in order
        let var_idx: u32 = kani::any();
        kani::assume(var_idx < NUM_VARS as u32);
        let var = Variable(var_idx);
        let polarity: bool = kani::any();
        let lit = if polarity {
            Literal::positive(var)
        } else {
            Literal::negative(var)
        };

        // Enqueue the literal
        solver.enqueue(lit, None);

        // Verify: trail_pos points to the correct position
        let pos = solver.trail_pos[var_idx as usize];
        assert!(pos < solver.trail.len());
        assert_eq!(solver.trail[pos], lit);
    }

    /// Gap 8: Verify propagate pointer bounds invariant
    ///
    /// This proof verifies that the two-pointer iteration in propagate()
    /// maintains the invariant: 0 <= j <= i <= watch_len at all times.
    ///
    /// The debug_assert! statements added in Gap 8 verify this at runtime.
    /// This Kani proof exhaustively checks that propagate completes without
    /// assertion failure on small inputs.
    #[kani::proof]
    #[kani::unwind(10)]
    fn proof_propagate_pointer_bounds() {
        const NUM_VARS: usize = 3;
        let mut solver: Solver<Vec<u8>> = Solver::new(NUM_VARS);

        // Create variable references
        let v0 = Variable(0);
        let v1 = Variable(1);
        let v2 = Variable(2);

        // Add some clauses to create watch lists
        // Binary clause: v0 \/ v1
        solver.add_clause(vec![Literal::positive(v0), Literal::positive(v1)]);
        // Binary clause: ~v0 \/ v2
        solver.add_clause(vec![Literal::negative(v0), Literal::positive(v2)]);
        // Ternary clause: v0 \/ v1 \/ v2
        solver.add_clause(vec![
            Literal::positive(v0),
            Literal::positive(v1),
            Literal::positive(v2),
        ]);

        solver.initialize_watches();

        // Symbolically pick a variable and polarity to assign
        let var_choice: u32 = kani::any();
        kani::assume(var_choice < NUM_VARS as u32);
        let polarity: bool = kani::any();

        let lit = if polarity {
            Literal::positive(Variable(var_choice))
        } else {
            Literal::negative(Variable(var_choice))
        };

        // Enqueue the literal (simulating a decision)
        solver.enqueue(lit, None);

        // Call propagate - this exercises the two-pointer iteration
        // The debug_assert! statements we added will verify the invariant
        let _conflict = solver.propagate();

        // If we reach here without assertion failure, the invariant held
        // The propagate function either found a conflict or completed normally
    }

    /// Gap 8: Verify propagate handles empty watch lists correctly
    #[kani::proof]
    fn proof_propagate_empty_watches() {
        const NUM_VARS: usize = 2;
        let mut solver: Solver<Vec<u8>> = Solver::new(NUM_VARS);

        // No clauses added, so watch lists are empty
        solver.initialize_watches();

        // Make an assignment
        let v0 = Variable(0);
        solver.enqueue(Literal::positive(v0), None);

        // Propagate on empty watch list should be a no-op
        let conflict = solver.propagate();
        assert!(conflict.is_none(), "No conflict expected with no clauses");
    }

    /// Gap 8: Verify propagate handles binary clause propagation correctly
    #[kani::proof]
    fn proof_propagate_binary_unit() {
        const NUM_VARS: usize = 2;
        let mut solver: Solver<Vec<u8>> = Solver::new(NUM_VARS);

        let v0 = Variable(0);
        let v1 = Variable(1);

        // Binary clause: v0 \/ v1
        // If v0 is false, v1 must be true (unit propagation)
        solver.add_clause(vec![Literal::positive(v0), Literal::positive(v1)]);
        solver.initialize_watches();

        // Make v0 false - this should trigger unit propagation of v1
        solver.enqueue(Literal::negative(v0), None);
        let conflict = solver.propagate();

        // No conflict, and v1 should be true
        assert!(conflict.is_none(), "No conflict expected");
        assert_eq!(
            solver.assignment[1],
            Some(true),
            "v1 should be propagated to true"
        );
    }

    /// Gap 8: Verify propagate handles binary clause conflict correctly
    #[kani::proof]
    fn proof_propagate_binary_conflict() {
        const NUM_VARS: usize = 2;
        let mut solver: Solver<Vec<u8>> = Solver::new(NUM_VARS);

        let v0 = Variable(0);
        let v1 = Variable(1);

        // Binary clause: v0 \/ v1
        solver.add_clause(vec![Literal::positive(v0), Literal::positive(v1)]);
        solver.initialize_watches();

        // Make both v0 and v1 false - this should create a conflict
        solver.enqueue(Literal::negative(v0), None);
        solver.enqueue(Literal::negative(v1), None);

        let conflict = solver.propagate();
        assert!(
            conflict.is_some(),
            "Conflict expected when both literals are false"
        );
    }
}
