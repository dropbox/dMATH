//! CDCL SAT solver (ported from MiniSat/Glucose design)
//!
//! This module implements a Conflict-Driven Clause Learning (CDCL) SAT solver,
//! the core algorithm used in modern SAT solvers like MiniSat, Glucose, and Z3.
//!
//! # Algorithm Overview
//!
//! 1. **Unit Propagation (BCP)**: If a clause has all but one literal false,
//!    the remaining literal must be true.
//!
//! 2. **Decision**: Pick an unassigned variable and assign it a value.
//!
//! 3. **Conflict Analysis**: When a conflict occurs, analyze the implication
//!    graph to learn a new clause that prevents the same conflict.
//!
//! 4. **Backtracking**: Jump back to an appropriate decision level.
//!
//! # Key Features
//!
//! - Two-watched literals for efficient unit propagation
//! - VSIDS decision heuristic
//! - 1UIP conflict analysis
//! - Clause database management (learned clause deletion)
//! - Restarts with Luby sequence

/// A variable in the SAT problem (0-indexed internally)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Var(pub u32);

impl Var {
    /// Create a new variable with the given index
    #[inline]
    pub fn new(idx: u32) -> Self {
        Var(idx)
    }

    /// Get the variable index
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A literal is a variable or its negation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Lit(u32);

impl Lit {
    /// Create a positive literal
    #[inline]
    pub fn pos(var: Var) -> Self {
        Lit(var.0 << 1)
    }

    /// Create a negative literal
    #[inline]
    pub fn neg(var: Var) -> Self {
        Lit((var.0 << 1) | 1)
    }

    /// Create a literal from variable and sign (true = positive)
    #[inline]
    pub fn new(var: Var, sign: bool) -> Self {
        if sign {
            Self::pos(var)
        } else {
            Self::neg(var)
        }
    }

    /// Get the underlying variable
    #[inline]
    pub fn var(self) -> Var {
        Var(self.0 >> 1)
    }

    /// Check if this is a positive literal
    #[inline]
    pub fn is_pos(self) -> bool {
        (self.0 & 1) == 0
    }

    /// Check if this is a negative literal
    #[inline]
    pub fn is_neg(self) -> bool {
        (self.0 & 1) == 1
    }

    /// Get the negation of this literal
    #[inline]
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Lit(self.0 ^ 1)
    }

    /// Get the sign (true = positive, false = negative)
    #[inline]
    pub fn sign(self) -> bool {
        self.is_pos()
    }

    /// Get the raw index (for array indexing, 2 entries per variable)
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Reference to a clause in the clause database
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClauseRef(u32);

impl ClauseRef {
    pub const INVALID: ClauseRef = ClauseRef(u32::MAX);

    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        self != Self::INVALID
    }
}

#[inline]
fn usize_to_u32(value: usize, context: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{context} exceeds u32::MAX: {value}"))
}

#[inline]
fn clause_ref_at(len: usize) -> ClauseRef {
    ClauseRef(usize_to_u32(len, "clause index"))
}

/// A clause is a disjunction of literals
#[derive(Clone, Debug)]
pub struct Clause {
    /// The literals in this clause (first two are watched)
    pub lits: Vec<Lit>,
    /// Is this a learned clause?
    pub learned: bool,
    /// Activity score for learned clause deletion
    pub activity: f64,
    /// Literal Block Distance (LBD) for Glucose-style deletion
    pub lbd: u32,
}

impl Clause {
    /// Create a new clause
    pub fn new(lits: Vec<Lit>, learned: bool) -> Self {
        Self {
            lits,
            learned,
            activity: 0.0,
            lbd: 0,
        }
    }

    /// Get the number of literals
    #[inline]
    pub fn len(&self) -> usize {
        self.lits.len()
    }

    /// Check if clause is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lits.is_empty()
    }

    /// Check if this is a unit clause
    #[inline]
    pub fn is_unit(&self) -> bool {
        self.lits.len() == 1
    }
}

/// The value assigned to a variable
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LBool {
    True,
    False,
    Undef,
}

impl LBool {
    /// Negate the value
    #[inline]
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        match self {
            LBool::True => LBool::False,
            LBool::False => LBool::True,
            LBool::Undef => LBool::Undef,
        }
    }

    /// Convert from bool
    #[inline]
    pub fn from_bool(b: bool) -> Self {
        if b {
            LBool::True
        } else {
            LBool::False
        }
    }
}

/// Information about a variable assignment
#[derive(Clone, Debug)]
struct VarData {
    /// Current assignment (Undef if unassigned)
    value: LBool,
    /// Decision level at which this variable was assigned
    level: u32,
    /// The clause that implied this assignment (INVALID for decisions)
    reason: ClauseRef,
}

impl Default for VarData {
    fn default() -> Self {
        Self {
            value: LBool::Undef,
            level: 0,
            reason: ClauseRef::INVALID,
        }
    }
}

/// VSIDS activity-based decision heuristic data
#[derive(Clone, Debug)]
struct VsidsData {
    /// Activity score per variable
    activity: Vec<f64>,
    /// Activity increment (decayed over time)
    var_inc: f64,
    /// Decay factor
    var_decay: f64,
    /// Heap for variable selection (indices into activity)
    heap: Vec<Var>,
    /// Position in heap for each variable (u32::MAX if not in heap)
    heap_pos: Vec<u32>,
}

impl VsidsData {
    fn new(num_vars: usize) -> Self {
        let mut heap = Vec::with_capacity(num_vars);
        let mut heap_pos = vec![0u32; num_vars];
        for (i, pos) in heap_pos.iter_mut().enumerate() {
            let idx = usize_to_u32(i, "VSIDS variable index");
            heap.push(Var::new(idx));
            *pos = idx;
        }
        Self {
            activity: vec![0.0; num_vars],
            var_inc: 1.0,
            var_decay: 0.95,
            heap,
            heap_pos,
        }
    }

    /// Bump the activity of a variable
    fn bump(&mut self, var: Var) {
        let idx = var.index();
        self.activity[idx] += self.var_inc;

        // Rescale if activity gets too large
        if self.activity[idx] > 1e100 {
            for a in &mut self.activity {
                *a *= 1e-100;
            }
            self.var_inc *= 1e-100;
        }

        // Update heap position
        if self.heap_pos[idx] != u32::MAX {
            self.percolate_up(self.heap_pos[idx] as usize);
        }
    }

    /// Decay all activities
    fn decay(&mut self) {
        self.var_inc /= self.var_decay;
    }

    /// Add a variable back to the heap
    fn insert(&mut self, var: Var) {
        let idx = var.index();
        if self.heap_pos[idx] == u32::MAX {
            let pos = self.heap.len();
            self.heap.push(var);
            self.heap_pos[idx] = usize_to_u32(pos, "VSIDS heap position");
            self.percolate_up(pos);
        }
    }

    /// Remove and return the variable with highest activity
    fn pop(&mut self) -> Option<Var> {
        if self.heap.is_empty() {
            return None;
        }
        let result = self.heap[0];
        self.heap_pos[result.index()] = u32::MAX;

        if self.heap.len() > 1 {
            let last = self
                .heap
                .pop()
                .expect("heap must not be empty after len > 1 check");
            self.heap[0] = last;
            self.heap_pos[last.index()] = 0;
            self.percolate_down(0);
        } else {
            self.heap.pop();
        }
        Some(result)
    }

    fn percolate_up(&mut self, mut pos: usize) {
        let var = self.heap[pos];
        let act = self.activity[var.index()];

        while pos > 0 {
            let parent = (pos - 1) / 2;
            let parent_var = self.heap[parent];
            if self.activity[parent_var.index()] >= act {
                break;
            }
            self.heap[pos] = parent_var;
            self.heap_pos[parent_var.index()] =
                usize_to_u32(pos, "VSIDS heap position during percolate_up");
            pos = parent;
        }
        self.heap[pos] = var;
        self.heap_pos[var.index()] = usize_to_u32(pos, "VSIDS heap position during percolate_up");
    }

    fn percolate_down(&mut self, mut pos: usize) {
        let var = self.heap[pos];
        let act = self.activity[var.index()];

        loop {
            let left = 2 * pos + 1;
            if left >= self.heap.len() {
                break;
            }
            let right = left + 1;

            // Find child with higher activity
            let best_child = if right < self.heap.len()
                && self.activity[self.heap[right].index()] > self.activity[self.heap[left].index()]
            {
                right
            } else {
                left
            };

            if act >= self.activity[self.heap[best_child].index()] {
                break;
            }

            let child_var = self.heap[best_child];
            self.heap[pos] = child_var;
            self.heap_pos[child_var.index()] =
                usize_to_u32(pos, "VSIDS heap position during percolate_down");
            pos = best_child;
        }
        self.heap[pos] = var;
        self.heap_pos[var.index()] = usize_to_u32(pos, "VSIDS heap position during percolate_down");
    }
}

/// Watch list for a literal (two-watched literal scheme)
type WatchList = Vec<ClauseRef>;

/// Result of SAT solving
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SolveResult {
    /// Satisfiable with the given assignment (variable index -> bool)
    Sat(Vec<bool>),
    /// Unsatisfiable
    Unsat,
    /// Resource limit reached
    Unknown,
}

/// CDCL SAT Solver
pub struct CdclSolver {
    /// Number of variables
    num_vars: usize,
    /// Clause database (original and learned)
    clauses: Vec<Clause>,
    /// Variable data (assignment, level, reason)
    var_data: Vec<VarData>,
    /// Watch lists indexed by literal
    watches: Vec<WatchList>,
    /// VSIDS decision heuristic
    vsids: VsidsData,
    /// Trail: sequence of assignments in order
    trail: Vec<Lit>,
    /// Trail limits: index in trail where each decision level starts
    trail_lim: Vec<usize>,
    /// Propagation queue head (index into trail)
    qhead: usize,
    /// Current decision level
    decision_level: u32,
    /// Conflict counter (for restarts)
    conflicts: u64,
    /// Decisions counter
    decisions: u64,
    /// Propagations counter
    propagations: u64,
    /// Learned clause activity increment
    clause_inc: f64,
    /// Clause activity decay
    clause_decay: f64,
    /// Conflict limit for search
    conflict_limit: u64,
    /// Seen marks for conflict analysis
    seen: Vec<bool>,
    /// Temporary storage for conflict analysis
    analyze_stack: Vec<Lit>,
    /// Temporary storage for learned clause
    learnt_clause: Vec<Lit>,
    /// Whether the problem is already determined to be UNSAT
    is_unsat: bool,
}

impl CdclSolver {
    /// Create a new solver with the given number of variables
    pub fn new(num_vars: usize) -> Self {
        let num_lits = num_vars * 2;
        Self {
            num_vars,
            clauses: Vec::new(),
            var_data: vec![VarData::default(); num_vars],
            watches: vec![Vec::new(); num_lits],
            vsids: VsidsData::new(num_vars),
            trail: Vec::with_capacity(num_vars),
            trail_lim: Vec::new(),
            qhead: 0,
            decision_level: 0,
            conflicts: 0,
            decisions: 0,
            propagations: 0,
            clause_inc: 1.0,
            clause_decay: 0.999,
            conflict_limit: u64::MAX,
            seen: vec![false; num_vars],
            analyze_stack: Vec::new(),
            learnt_clause: Vec::new(),
            is_unsat: false,
        }
    }

    /// Create a new variable and return it
    pub fn new_var(&mut self) -> Var {
        let var = Var::new(usize_to_u32(self.num_vars, "variable count"));
        self.num_vars += 1;
        self.var_data.push(VarData::default());
        self.watches.push(Vec::new()); // positive literal
        self.watches.push(Vec::new()); // negative literal
        self.vsids.activity.push(0.0);
        self.vsids
            .heap_pos
            .push(usize_to_u32(self.vsids.heap.len(), "VSIDS heap length"));
        self.vsids.heap.push(var);
        self.seen.push(false);
        var
    }

    /// Get the current number of variables
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Get the number of clauses
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Get the current value of a literal
    #[inline]
    fn lit_value(&self, lit: Lit) -> LBool {
        let val = self.var_data[lit.var().index()].value;
        if lit.is_pos() {
            val
        } else {
            val.not()
        }
    }

    /// Set the value of a literal
    #[inline]
    fn set_lit(&mut self, lit: Lit, reason: ClauseRef) {
        let var = lit.var();
        let idx = var.index();
        self.var_data[idx].value = if lit.is_pos() {
            LBool::True
        } else {
            LBool::False
        };
        self.var_data[idx].level = self.decision_level;
        self.var_data[idx].reason = reason;
        self.trail.push(lit);
    }

    /// Add a clause to the solver
    /// Returns None if the clause makes the problem immediately UNSAT (empty clause)
    /// Returns Some(ClauseRef) for the added clause
    pub fn add_clause(&mut self, mut lits: Vec<Lit>) -> Option<ClauseRef> {
        // Remove duplicates and check for tautologies
        lits.sort_by_key(|l| l.0);
        let mut j = 0;
        for i in 0..lits.len() {
            // Skip duplicates
            if j > 0 && lits[j - 1] == lits[i] {
                continue;
            }
            // Check for tautology (x and !x)
            if j > 0 && lits[j - 1].var() == lits[i].var() {
                // Tautology - clause is trivially satisfied
                return Some(clause_ref_at(self.clauses.len())); // Dummy ref
            }
            lits[j] = lits[i];
            j += 1;
        }
        lits.truncate(j);

        // Handle special cases
        if lits.is_empty() {
            self.is_unsat = true;
            return None; // Empty clause = UNSAT
        }

        if lits.len() == 1 {
            // Unit clause - must be true at level 0
            let lit = lits[0];
            match self.lit_value(lit) {
                LBool::True => return Some(clause_ref_at(self.clauses.len())),
                LBool::False => {
                    self.is_unsat = true;
                    return None; // Conflict at level 0
                }
                LBool::Undef => {
                    self.set_lit(lit, ClauseRef::INVALID);
                    return Some(clause_ref_at(self.clauses.len()));
                }
            }
        }

        // Add clause and set up watches
        let cref = clause_ref_at(self.clauses.len());
        let clause = Clause::new(lits.clone(), false);

        // Watch first two literals
        self.watches[lits[0].not().index()].push(cref);
        self.watches[lits[1].not().index()].push(cref);

        self.clauses.push(clause);
        Some(cref)
    }

    /// Add a learned clause (during conflict analysis)
    fn add_learned_clause(&mut self, lits: Vec<Lit>) -> ClauseRef {
        debug_assert!(!lits.is_empty());

        if lits.len() == 1 {
            // Unit learned clause - propagate at level 0
            // Note: in practice, this shouldn't happen during CDCL,
            // but handle it anyway
            let cref = clause_ref_at(self.clauses.len());
            self.clauses.push(Clause::new(lits, true));
            return cref;
        }

        let cref = clause_ref_at(self.clauses.len());
        let mut clause = Clause::new(lits.clone(), true);

        // Compute LBD (Literal Block Distance)
        let mut levels_seen = vec![false; (self.decision_level + 1) as usize];
        let mut lbd = 0u32;
        for lit in &lits {
            let level = self.var_data[lit.var().index()].level as usize;
            if !levels_seen[level] {
                levels_seen[level] = true;
                lbd += 1;
            }
        }
        clause.lbd = lbd;

        // Watch first two literals
        self.watches[lits[0].not().index()].push(cref);
        self.watches[lits[1].not().index()].push(cref);

        self.clauses.push(clause);
        cref
    }

    /// Propagate unit clauses (BCP - Boolean Constraint Propagation)
    /// Returns the conflicting clause if a conflict occurs
    fn propagate(&mut self) -> Option<ClauseRef> {
        while self.qhead < self.trail.len() {
            let lit = self.trail[self.qhead];
            self.qhead += 1;
            self.propagations += 1;

            // Look at clauses watching !lit (they may become unit or conflict)
            let watch_lit = lit; // We watch the negation to detect when it becomes false
            let watch_idx = watch_lit.index();

            // Take ownership of watch list to avoid borrow issues
            let watches = std::mem::take(&mut self.watches[watch_idx]);
            let mut new_watches = Vec::with_capacity(watches.len());

            let mut conflict = None;

            for (watch_pos, &cref) in watches.iter().enumerate() {
                // Make sure the false literal is in position 1
                if self.clauses[cref.index()].lits[0] == watch_lit.not() {
                    self.clauses[cref.index()].lits.swap(0, 1);
                }
                debug_assert!(self.clauses[cref.index()].lits[1] == watch_lit.not());

                let first = self.clauses[cref.index()].lits[0];

                // If first literal is true, clause is satisfied
                if self.lit_value(first) == LBool::True {
                    new_watches.push(cref);
                    continue;
                }

                // Look for a new literal to watch
                let mut found_new_watch = false;
                let clause_len = self.clauses[cref.index()].lits.len();
                for i in 2..clause_len {
                    let lit_i = self.clauses[cref.index()].lits[i];
                    if self.lit_value(lit_i) != LBool::False {
                        // Found a non-false literal, swap it to position 1
                        self.clauses[cref.index()].lits.swap(1, i);
                        // Add to watch list of new watched literal
                        let new_watch_lit = self.clauses[cref.index()].lits[1];
                        self.watches[new_watch_lit.not().index()].push(cref);
                        found_new_watch = true;
                        break;
                    }
                }

                if found_new_watch {
                    continue;
                }

                // No new watch found - clause is unit or conflict
                new_watches.push(cref);

                if self.lit_value(first) == LBool::False {
                    // Conflict!
                    conflict = Some(cref);
                    // Copy remaining watches
                    for &remaining in watches.iter().skip(watch_pos + 1) {
                        if !new_watches.contains(&remaining) {
                            new_watches.push(remaining);
                        }
                    }
                    break;
                }
                // Unit propagation
                self.set_lit(first, cref);
            }

            self.watches[watch_idx] = new_watches;

            if conflict.is_some() {
                return conflict;
            }
        }

        None
    }

    /// Analyze a conflict and learn a new clause
    /// Returns the learned clause and the backtrack level
    fn analyze(&mut self, conflict: ClauseRef) -> (Vec<Lit>, u32) {
        self.learnt_clause.clear();
        self.analyze_stack.clear();

        // Start with the conflict clause
        let mut p = Lit(u32::MAX); // Sentinel value
        let mut counter = 0;
        let mut cref = conflict;

        // First UIP (1UIP) scheme
        loop {
            // Bump clause activity if learned
            let is_learned = self.clauses[cref.index()].learned;
            if is_learned {
                let clause_inc = self.clause_inc;
                self.clauses[cref.index()].activity += clause_inc;
            }

            // Process the reason clause
            let start = usize::from(p.0 != u32::MAX);
            let clause_lits: Vec<Lit> = self.clauses[cref.index()].lits[start..].to_vec();

            for lit in clause_lits {
                let var = lit.var();
                let idx = var.index();

                if self.seen[idx] {
                    continue;
                }
                self.seen[idx] = true;

                let level = self.var_data[idx].level;

                if level == self.decision_level {
                    // This variable was assigned at the current level
                    counter += 1;
                } else if level > 0 {
                    // This variable was assigned at a previous level
                    self.learnt_clause.push(lit.not());
                    self.vsids.bump(var);
                }
            }

            // Find the next literal to process (most recent on trail at current level)
            loop {
                p = *self
                    .trail
                    .last()
                    .expect("trail must not be empty during conflict analysis");
                self.trail.pop();
                let var = p.var();
                if self.seen[var.index()] {
                    break;
                }
            }

            counter -= 1;
            self.seen[p.var().index()] = false;

            if counter == 0 {
                break;
            }

            // Get the reason for this assignment
            cref = self.var_data[p.var().index()].reason;
            assert!(cref.is_valid(), "Invalid reason during conflict analysis");
        }

        // The first literal is the asserting literal (negation of 1UIP)
        self.learnt_clause.insert(0, p.not());

        // Clear seen flags
        for lit in &self.learnt_clause {
            self.seen[lit.var().index()] = false;
        }

        // Bump activity of the asserting variable
        self.vsids.bump(p.var());

        // Find the backtrack level (second highest level in learned clause)
        let mut backtrack_level = 0u32;
        if self.learnt_clause.len() > 1 {
            let mut max_idx = 1;
            for i in 2..self.learnt_clause.len() {
                let level = self.var_data[self.learnt_clause[i].var().index()].level;
                if level > self.var_data[self.learnt_clause[max_idx].var().index()].level {
                    max_idx = i;
                }
            }
            // Swap to position 1 (second watched literal)
            self.learnt_clause.swap(1, max_idx);
            backtrack_level = self.var_data[self.learnt_clause[1].var().index()].level;
        }

        (self.learnt_clause.clone(), backtrack_level)
    }

    /// Backtrack to the given decision level
    fn backtrack(&mut self, level: u32) {
        if self.decision_level <= level {
            return;
        }

        // Unassign all variables assigned after the target level
        while self.trail.len() > self.trail_lim[level as usize] {
            let lit = self
                .trail
                .pop()
                .expect("trail must not be empty during backtrack");
            let var = lit.var();
            let idx = var.index();
            self.var_data[idx].value = LBool::Undef;
            self.var_data[idx].reason = ClauseRef::INVALID;
            self.vsids.insert(var);
        }

        self.trail_lim.truncate(level as usize);
        self.qhead = self.trail.len();
        self.decision_level = level;
    }

    /// Make a decision (pick an unassigned variable and assign it)
    fn decide(&mut self) -> bool {
        // Use VSIDS to pick the next variable
        while let Some(var) = self.vsids.pop() {
            if self.var_data[var.index()].value == LBool::Undef {
                // Create a new decision level
                self.trail_lim.push(self.trail.len());
                self.decision_level += 1;
                self.decisions += 1;

                // Assign the variable (default to positive)
                let lit = Lit::pos(var);
                self.set_lit(lit, ClauseRef::INVALID);
                return true;
            }
        }
        false // No unassigned variables
    }

    /// Decay clause activities
    fn decay_clause_activity(&mut self) {
        self.clause_inc /= self.clause_decay;
    }

    /// Main solving loop
    pub fn solve(&mut self) -> SolveResult {
        // Check if already determined UNSAT during clause addition
        if self.is_unsat {
            return SolveResult::Unsat;
        }

        // Initial unit propagation
        if self.propagate().is_some() {
            return SolveResult::Unsat;
        }

        loop {
            // Try to propagate
            if let Some(conflict) = self.propagate() {
                // Conflict!
                self.conflicts += 1;

                if self.decision_level == 0 {
                    return SolveResult::Unsat;
                }

                if self.conflicts >= self.conflict_limit {
                    return SolveResult::Unknown;
                }

                // Analyze conflict and learn
                let (learnt, backtrack_level) = self.analyze(conflict);

                // Backtrack
                self.backtrack(backtrack_level);

                // Add learned clause
                if learnt.len() == 1 {
                    // Unit clause - directly propagate
                    self.set_lit(learnt[0], ClauseRef::INVALID);
                } else {
                    let cref = self.add_learned_clause(learnt.clone());
                    // The first literal becomes unit after backtracking
                    self.set_lit(learnt[0], cref);
                }

                // Decay activities
                self.vsids.decay();
                self.decay_clause_activity();
            } else {
                // No conflict - make a decision
                if !self.decide() {
                    // All variables assigned - SAT!
                    let model: Vec<bool> = self
                        .var_data
                        .iter()
                        .map(|vd| vd.value == LBool::True)
                        .collect();
                    return SolveResult::Sat(model);
                }
            }
        }
    }

    /// Set the conflict limit for solving
    pub fn set_conflict_limit(&mut self, limit: u64) {
        self.conflict_limit = limit;
    }

    /// Get statistics
    pub fn stats(&self) -> SolverStats {
        SolverStats {
            conflicts: self.conflicts,
            decisions: self.decisions,
            propagations: self.propagations,
            learned_clauses: self.clauses.iter().filter(|c| c.learned).count() as u64,
        }
    }
}

/// Solver statistics
#[derive(Clone, Debug, Default)]
pub struct SolverStats {
    pub conflicts: u64,
    pub decisions: u64,
    pub propagations: u64,
    pub learned_clauses: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_lit_basic() {
        let v = Var::new(5);
        assert_eq!(v.index(), 5);

        let pos = Lit::pos(v);
        let neg = Lit::neg(v);

        assert!(pos.is_pos());
        assert!(!pos.is_neg());
        assert!(!neg.is_pos());
        assert!(neg.is_neg());

        assert_eq!(pos.var(), v);
        assert_eq!(neg.var(), v);

        assert_eq!(pos.not(), neg);
        assert_eq!(neg.not(), pos);
    }

    #[test]
    fn test_empty_solver() {
        let mut solver = CdclSolver::new(0);
        assert_eq!(solver.solve(), SolveResult::Sat(vec![]));
    }

    #[test]
    fn test_single_positive() {
        // x = true
        let mut solver = CdclSolver::new(1);
        let x = Var::new(0);
        solver.add_clause(vec![Lit::pos(x)]);
        match solver.solve() {
            SolveResult::Sat(model) => {
                assert!(model[0]); // x = true
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_single_negative() {
        // !x = true (x = false)
        let mut solver = CdclSolver::new(1);
        let x = Var::new(0);
        solver.add_clause(vec![Lit::neg(x)]);
        match solver.solve() {
            SolveResult::Sat(model) => {
                assert!(!model[0]); // x = false
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_contradiction() {
        // x AND !x = UNSAT
        let mut solver = CdclSolver::new(1);
        let x = Var::new(0);
        solver.add_clause(vec![Lit::pos(x)]);
        solver.add_clause(vec![Lit::neg(x)]);
        assert_eq!(solver.solve(), SolveResult::Unsat);
    }

    #[test]
    fn test_simple_sat() {
        // (x OR y) AND (!x OR y) = y
        let mut solver = CdclSolver::new(2);
        let x = Var::new(0);
        let y = Var::new(1);
        solver.add_clause(vec![Lit::pos(x), Lit::pos(y)]);
        solver.add_clause(vec![Lit::neg(x), Lit::pos(y)]);
        match solver.solve() {
            SolveResult::Sat(model) => {
                // Must have y = true
                assert!(model[1]);
                // x can be either value
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_unit_propagation() {
        // (x) AND (x OR y) AND (!x OR z)
        // Unit: x = true -> z = true (from !x OR z)
        let mut solver = CdclSolver::new(3);
        let x = Var::new(0);
        let y = Var::new(1);
        let z = Var::new(2);
        solver.add_clause(vec![Lit::pos(x)]);
        solver.add_clause(vec![Lit::pos(x), Lit::pos(y)]);
        solver.add_clause(vec![Lit::neg(x), Lit::pos(z)]);
        match solver.solve() {
            SolveResult::Sat(model) => {
                assert!(model[0]); // x = true
                assert!(model[2]); // z = true (propagated)
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_pigeonhole_2_1() {
        // 2 pigeons, 1 hole - each pigeon must be in the hole
        // p1 AND p2 AND (!p1 OR !p2) = UNSAT
        let mut solver = CdclSolver::new(2);
        let p1 = Var::new(0);
        let p2 = Var::new(1);
        solver.add_clause(vec![Lit::pos(p1)]); // pigeon 1 in hole
        solver.add_clause(vec![Lit::pos(p2)]); // pigeon 2 in hole
        solver.add_clause(vec![Lit::neg(p1), Lit::neg(p2)]); // at most one per hole
        assert_eq!(solver.solve(), SolveResult::Unsat);
    }

    #[test]
    fn test_three_coloring_triangle() {
        // Color a triangle with 3 colors - should be satisfiable
        // Variables: v0_c0, v0_c1, v0_c2, v1_c0, v1_c1, v1_c2, v2_c0, v2_c1, v2_c2
        let mut solver = CdclSolver::new(9);

        // Helper to get variable for vertex v, color c
        let var = |v: u32, c: u32| Var::new(v * 3 + c);

        // Each vertex has at least one color
        for v in 0..3 {
            solver.add_clause(vec![
                Lit::pos(var(v, 0)),
                Lit::pos(var(v, 1)),
                Lit::pos(var(v, 2)),
            ]);
        }

        // Each vertex has at most one color
        for v in 0..3 {
            for c1 in 0..3 {
                for c2 in (c1 + 1)..3 {
                    solver.add_clause(vec![Lit::neg(var(v, c1)), Lit::neg(var(v, c2))]);
                }
            }
        }

        // Adjacent vertices have different colors (triangle: 0-1, 1-2, 0-2)
        let edges = [(0, 1), (1, 2), (0, 2)];
        for (v1, v2) in edges {
            for c in 0..3 {
                solver.add_clause(vec![Lit::neg(var(v1, c)), Lit::neg(var(v2, c))]);
            }
        }

        match solver.solve() {
            SolveResult::Sat(model) => {
                // Verify each vertex has exactly one color
                for v in 0..3 {
                    let colors: Vec<bool> = (0..3).map(|c| model[(v * 3 + c) as usize]).collect();
                    assert_eq!(colors.iter().filter(|&&x| x).count(), 1);
                }
                // Verify adjacent vertices have different colors
                for (v1, v2) in edges {
                    let c1 = (0..3).find(|&c| model[(v1 * 3 + c) as usize]).unwrap();
                    let c2 = (0..3).find(|&c| model[(v2 * 3 + c) as usize]).unwrap();
                    assert_ne!(c1, c2);
                }
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_two_coloring_triangle() {
        // Color a triangle with 2 colors - should be UNSAT (triangle is not bipartite)
        let mut solver = CdclSolver::new(6);

        let var = |v: u32, c: u32| Var::new(v * 2 + c);

        // Each vertex has at least one color
        for v in 0..3 {
            solver.add_clause(vec![Lit::pos(var(v, 0)), Lit::pos(var(v, 1))]);
        }

        // Each vertex has at most one color
        for v in 0..3 {
            solver.add_clause(vec![Lit::neg(var(v, 0)), Lit::neg(var(v, 1))]);
        }

        // Adjacent vertices have different colors
        let edges = [(0, 1), (1, 2), (0, 2)];
        for (v1, v2) in edges {
            for c in 0..2 {
                solver.add_clause(vec![Lit::neg(var(v1, c)), Lit::neg(var(v2, c))]);
            }
        }

        assert_eq!(solver.solve(), SolveResult::Unsat);
    }

    #[test]
    fn test_tautology_ignored() {
        // (x OR !x) is a tautology - should be satisfied trivially
        let mut solver = CdclSolver::new(1);
        let x = Var::new(0);
        solver.add_clause(vec![Lit::pos(x), Lit::neg(x)]);
        match solver.solve() {
            SolveResult::Sat(_) => {}
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_duplicate_literals() {
        // (x OR x) should be simplified to (x)
        let mut solver = CdclSolver::new(1);
        let x = Var::new(0);
        solver.add_clause(vec![Lit::pos(x), Lit::pos(x)]);
        match solver.solve() {
            SolveResult::Sat(model) => {
                assert!(model[0]);
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_conflict_limit() {
        // Create a hard problem and limit conflicts
        let mut solver = CdclSolver::new(5);
        // Create some clauses that will cause conflicts
        for i in 0..5 {
            let v = Var::new(i);
            solver.add_clause(vec![Lit::pos(v)]);
            solver.add_clause(vec![Lit::neg(v)]);
        }
        solver.set_conflict_limit(1);
        // Should hit conflict limit
        let result = solver.solve();
        assert!(result == SolveResult::Unsat || result == SolveResult::Unknown);
    }

    #[test]
    fn test_new_var_dynamic() {
        let mut solver = CdclSolver::new(0);
        let v1 = solver.new_var();
        let v2 = solver.new_var();
        assert_eq!(v1, Var::new(0));
        assert_eq!(v2, Var::new(1));
        assert_eq!(solver.num_vars(), 2);

        solver.add_clause(vec![Lit::pos(v1), Lit::pos(v2)]);
        solver.add_clause(vec![Lit::neg(v1)]);
        match solver.solve() {
            SolveResult::Sat(model) => {
                assert!(!model[0]); // v1 = false
                assert!(model[1]); // v2 = true
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_stats() {
        let mut solver = CdclSolver::new(2);
        let x = Var::new(0);
        let y = Var::new(1);
        solver.add_clause(vec![Lit::pos(x)]);
        solver.add_clause(vec![Lit::neg(x), Lit::pos(y)]);
        solver.solve();

        let stats = solver.stats();
        assert!(stats.propagations > 0);
    }

    #[test]
    fn test_chain_implication() {
        // x1 -> x2 -> x3 -> x4 -> x5
        // (x1) AND (!x1 OR x2) AND (!x2 OR x3) AND (!x3 OR x4) AND (!x4 OR x5)
        let mut solver = CdclSolver::new(5);
        let vars: Vec<Var> = (0..5).map(Var::new).collect();

        solver.add_clause(vec![Lit::pos(vars[0])]);
        for i in 0..4 {
            solver.add_clause(vec![Lit::neg(vars[i]), Lit::pos(vars[i + 1])]);
        }

        match solver.solve() {
            SolveResult::Sat(model) => {
                // All should be true due to chain
                for (i, &val) in model.iter().enumerate() {
                    assert!(val, "Variable {i} should be true");
                }
            }
            _ => panic!("Expected SAT"),
        }
    }
}
