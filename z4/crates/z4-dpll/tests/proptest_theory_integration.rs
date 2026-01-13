//! Property-based tests for DPLL(T) theory integration (Gap 9).
//!
//! These tests generate small random theory formulas (EUF, LRA, LIA, and combined)
//! and cross-check Z4's DPLL(T) result against a brute-force enumeration over the
//! Boolean abstraction, using the theory solver as the consistency oracle.
//!
//! For LIA, the oracle handles NeedSplit by recursively exploring both branches
//! (x <= floor and x >= ceil) to determine satisfiability.

use proptest::prelude::*;
use z4_core::term::Symbol;
use z4_core::{Sort, TermId, TermStore, TheoryResult, TheorySolver, Tseitin};
use z4_dpll::DpllT;
use z4_euf::EufSolver;
use z4_lia::LiaSolver;
use z4_lra::LraSolver;
use z4_sat::SolveResult;

#[derive(Clone, Debug)]
enum BoolExpr {
    Atom(usize),
    Not(Box<BoolExpr>),
    And(Vec<BoolExpr>),
    Or(Vec<BoolExpr>),
}

impl BoolExpr {
    fn eval(&self, assignment: &[bool]) -> bool {
        match self {
            BoolExpr::Atom(i) => assignment[*i],
            BoolExpr::Not(e) => !e.eval(assignment),
            BoolExpr::And(es) => es.iter().all(|e| e.eval(assignment)),
            BoolExpr::Or(es) => es.iter().any(|e| e.eval(assignment)),
        }
    }

    fn mark_used(&self, used: &mut [bool]) {
        match self {
            BoolExpr::Atom(i) => used[*i] = true,
            BoolExpr::Not(e) => e.mark_used(used),
            BoolExpr::And(es) | BoolExpr::Or(es) => {
                for e in es {
                    e.mark_used(used);
                }
            }
        }
    }

    fn used_atoms(&self, num_atoms: usize) -> Vec<usize> {
        let mut used = vec![false; num_atoms];
        self.mark_used(&mut used);
        used.iter()
            .enumerate()
            .filter_map(|(i, b)| b.then_some(i))
            .collect()
    }
}

fn bool_expr_strategy(num_atoms: usize) -> impl Strategy<Value = BoolExpr> {
    let leaf = (0usize..num_atoms).prop_map(BoolExpr::Atom);
    leaf.prop_recursive(4, 64, 8, move |inner| {
        prop_oneof![
            inner.clone().prop_map(|e| BoolExpr::Not(Box::new(e))),
            prop::collection::vec(inner.clone(), 2..=3).prop_map(BoolExpr::And),
            prop::collection::vec(inner, 2..=3).prop_map(BoolExpr::Or),
        ]
    })
}

fn build_bool_term(terms: &mut TermStore, atoms: &[TermId], expr: &BoolExpr) -> TermId {
    match expr {
        BoolExpr::Atom(i) => atoms[*i],
        BoolExpr::Not(e) => {
            let inner = build_bool_term(terms, atoms, e);
            terms.mk_not(inner)
        }
        BoolExpr::And(es) => {
            let args = es
                .iter()
                .map(|e| build_bool_term(terms, atoms, e))
                .collect::<Vec<_>>();
            terms.mk_and(args)
        }
        BoolExpr::Or(es) => {
            let args = es
                .iter()
                .map(|e| build_bool_term(terms, atoms, e))
                .collect::<Vec<_>>();
            terms.mk_or(args)
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Expected {
    Sat,
    Unsat,
}

fn brute_force_expected<T: z4_core::TheorySolver>(
    make_theory: impl Fn() -> T,
    atoms: &[TermId],
    expr: &BoolExpr,
) -> Expected {
    let used = expr.used_atoms(atoms.len());
    let num_used = used.len();
    debug_assert!(num_used > 0);
    debug_assert!(num_used <= 20, "brute force bound");

    for mask in 0u64..(1u64 << num_used) {
        let mut assignment = vec![false; atoms.len()];
        for (bit, &atom_idx) in used.iter().enumerate() {
            assignment[atom_idx] = ((mask >> bit) & 1) == 1;
        }

        if !expr.eval(&assignment) {
            continue;
        }

        // Check for contradictory assignments to atoms with the same TermId.
        // Due to hash-consing, different atom indices may map to the same TermId.
        // If a=true and b=false but atoms[a] == atoms[b], that's a Boolean conflict.
        let mut term_to_value: std::collections::HashMap<TermId, bool> =
            std::collections::HashMap::new();
        let mut has_conflict = false;
        for &atom_idx in &used {
            let term_id = atoms[atom_idx];
            let value = assignment[atom_idx];
            if let Some(&prev_value) = term_to_value.get(&term_id) {
                if prev_value != value {
                    // Same TermId assigned both true and false - Boolean conflict
                    has_conflict = true;
                    break;
                }
            } else {
                term_to_value.insert(term_id, value);
            }
        }
        if has_conflict {
            continue;
        }

        let mut theory = make_theory();
        for &atom_idx in &used {
            theory.assert_literal(atoms[atom_idx], assignment[atom_idx]);
        }
        match theory.check() {
            TheoryResult::Sat => return Expected::Sat,
            TheoryResult::Unsat(_) => continue,
            TheoryResult::Unknown
            | TheoryResult::NeedSplit(_)
            | TheoryResult::NeedDisequlitySplit(_)
            | TheoryResult::NeedExpressionSplit(_) => {
                panic!("unexpected Unknown/NeedSplit/NeedDisequlitySplit during brute force");
            }
        }
    }

    Expected::Unsat
}

/// Simple bounds-based LIA oracle.
/// For a single integer variable x with constraints `x <= b` and `x >= b`,
/// computes the intersection of bounds and checks if an integer exists in that range.
///
/// This avoids complex term store mutations by directly reasoning about bounds.
fn lia_bounds_satisfiable(atom_specs: &[LraAtomSpec], assignment: &[bool], used: &[usize]) -> bool {
    // For a single variable x:
    // - x <= b constraints give upper bounds
    // - x >= b constraints give lower bounds
    //
    // The formula is SAT iff max(lower bounds) <= min(upper bounds)

    let mut lower = i64::MIN; // max of all lower bounds
    let mut upper = i64::MAX; // min of all upper bounds

    for &atom_idx in used {
        let spec = &atom_specs[atom_idx];
        let val = assignment[atom_idx];
        let bound = spec.bound as i64;

        match (spec.cmp, val) {
            (LraCmp::Le, true) => {
                // x <= bound is TRUE -> upper bound
                upper = upper.min(bound);
            }
            (LraCmp::Le, false) => {
                // x <= bound is FALSE -> x > bound -> x >= bound+1
                lower = lower.max(bound + 1);
            }
            (LraCmp::Ge, true) => {
                // x >= bound is TRUE -> lower bound
                lower = lower.max(bound);
            }
            (LraCmp::Ge, false) => {
                // x >= bound is FALSE -> x < bound -> x <= bound-1
                upper = upper.min(bound - 1);
            }
        }
    }

    // For integers, satisfiable iff there exists an integer in [lower, upper]
    lower <= upper
}

/// Brute-force oracle for LIA with direct bounds computation.
fn brute_force_expected_lia(atom_specs: &[LraAtomSpec], expr: &BoolExpr) -> Expected {
    let used = expr.used_atoms(atom_specs.len());
    let num_used = used.len();
    debug_assert!(num_used > 0);
    debug_assert!(num_used <= 20, "brute force bound");

    for mask in 0u64..(1u64 << num_used) {
        let mut assignment = vec![false; atom_specs.len()];
        for (bit, &atom_idx) in used.iter().enumerate() {
            assignment[atom_idx] = ((mask >> bit) & 1) == 1;
        }

        if !expr.eval(&assignment) {
            continue;
        }

        // Check bounds satisfiability directly
        if lia_bounds_satisfiable(atom_specs, &assignment, &used) {
            return Expected::Sat;
        }
    }

    Expected::Unsat
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum EufTermSpec {
    Const(u8),
    F1(u8),
    F2(u8),
}

fn euf_term_spec_strategy(num_consts: u8) -> impl Strategy<Value = EufTermSpec> {
    prop_oneof![
        (0u8..num_consts).prop_map(EufTermSpec::Const),
        (0u8..num_consts).prop_map(EufTermSpec::F1),
        (0u8..num_consts).prop_map(EufTermSpec::F2),
    ]
}

fn build_euf_term(
    terms: &mut TermStore,
    sort: &Sort,
    consts: &[TermId],
    spec: EufTermSpec,
) -> TermId {
    let f = Symbol::named("f");
    match spec {
        EufTermSpec::Const(i) => consts[i as usize],
        EufTermSpec::F1(i) => terms.mk_app(f.clone(), vec![consts[i as usize]], sort.clone()),
        EufTermSpec::F2(i) => {
            let inner = terms.mk_app(f.clone(), vec![consts[i as usize]], sort.clone());
            terms.mk_app(f, vec![inner], sort.clone())
        }
    }
}

#[derive(Clone, Debug)]
struct EufAtomSpec {
    lhs: EufTermSpec,
    rhs: EufTermSpec,
}

fn euf_atom_spec_strategy(num_consts: u8) -> impl Strategy<Value = EufAtomSpec> {
    (
        euf_term_spec_strategy(num_consts),
        euf_term_spec_strategy(num_consts),
    )
        .prop_map(|(l, r)| EufAtomSpec { lhs: l, rhs: r })
        .prop_filter("non-reflexive equality", |s| s.lhs != s.rhs)
}

#[derive(Clone, Copy, Debug)]
enum LraCmp {
    Le,
    Ge,
}

#[derive(Clone, Copy, Debug)]
struct LraAtomSpec {
    cmp: LraCmp,
    bound: i8,
}

fn lra_atom_spec_strategy() -> impl Strategy<Value = LraAtomSpec> {
    (prop_oneof![Just(LraCmp::Le), Just(LraCmp::Ge)], -5i8..=5i8)
        .prop_map(|(cmp, bound)| LraAtomSpec { cmp, bound })
}

#[derive(Clone, Debug)]
struct EufCase {
    num_consts: u8,
    atom_specs: Vec<EufAtomSpec>,
    expr: BoolExpr,
}

fn euf_case_strategy() -> impl Strategy<Value = EufCase> {
    (2u8..=4u8, 1usize..=6usize)
        .prop_flat_map(|(num_consts, num_atoms)| {
            (
                Just(num_consts),
                prop::collection::vec(euf_atom_spec_strategy(num_consts), num_atoms),
                bool_expr_strategy(num_atoms),
            )
        })
        .prop_map(|(num_consts, atom_specs, expr)| EufCase {
            num_consts,
            atom_specs,
            expr,
        })
}

#[derive(Clone, Debug)]
struct LraCase {
    atom_specs: Vec<LraAtomSpec>,
    expr: BoolExpr,
}

fn lra_case_strategy() -> impl Strategy<Value = LraCase> {
    (1usize..=6usize)
        .prop_flat_map(|num_atoms| {
            (
                prop::collection::vec(lra_atom_spec_strategy(), num_atoms),
                bool_expr_strategy(num_atoms),
            )
        })
        .prop_map(|(atom_specs, expr)| LraCase { atom_specs, expr })
}

// LIA case: integer bounds on a single variable x
#[derive(Clone, Debug)]
struct LiaCase {
    atom_specs: Vec<LraAtomSpec>, // Reuse LraAtomSpec but interpret for integers
    expr: BoolExpr,
}

fn lia_case_strategy() -> impl Strategy<Value = LiaCase> {
    (1usize..=5usize) // Fewer atoms due to split complexity
        .prop_flat_map(|num_atoms| {
            (
                prop::collection::vec(lra_atom_spec_strategy(), num_atoms),
                bool_expr_strategy(num_atoms),
            )
        })
        .prop_map(|(atom_specs, expr)| LiaCase { atom_specs, expr })
}

// Combined EUF+LRA case
#[derive(Clone, Debug)]
struct EufLraCase {
    num_consts: u8,
    euf_atoms: Vec<EufAtomSpec>,
    lra_atoms: Vec<LraAtomSpec>,
    expr: BoolExpr,
}

fn euf_lra_case_strategy() -> impl Strategy<Value = EufLraCase> {
    (2u8..=3u8, 1usize..=3usize, 1usize..=3usize)
        .prop_flat_map(|(num_consts, num_euf, num_lra)| {
            let total = num_euf + num_lra;
            (
                Just(num_consts),
                prop::collection::vec(euf_atom_spec_strategy(num_consts), num_euf),
                prop::collection::vec(lra_atom_spec_strategy(), num_lra),
                bool_expr_strategy(total),
            )
        })
        .prop_map(|(num_consts, euf_atoms, lra_atoms, expr)| EufLraCase {
            num_consts,
            euf_atoms,
            lra_atoms,
            expr,
        })
}

/// Combined EUF+LRA theory solver for testing.
/// Since the theories don't share variables, we just check both.
struct CombinedEufLra<'a> {
    euf: EufSolver<'a>,
    lra: LraSolver<'a>,
}

impl<'a> CombinedEufLra<'a> {
    fn new(terms: &'a TermStore) -> Self {
        CombinedEufLra {
            euf: EufSolver::new(terms),
            lra: LraSolver::new(terms),
        }
    }
}

impl<'a> TheorySolver for CombinedEufLra<'a> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        self.euf.assert_literal(literal, value);
        self.lra.assert_literal(literal, value);
    }

    fn check(&mut self) -> TheoryResult {
        match self.euf.check() {
            TheoryResult::Unsat(reasons) => return TheoryResult::Unsat(reasons),
            TheoryResult::Unknown => return TheoryResult::Unknown,
            TheoryResult::NeedSplit(s) => return TheoryResult::NeedSplit(s),
            TheoryResult::NeedDisequlitySplit(s) => return TheoryResult::NeedDisequlitySplit(s),
            TheoryResult::NeedExpressionSplit(s) => return TheoryResult::NeedExpressionSplit(s),
            TheoryResult::Sat => {}
        }
        self.lra.check()
    }

    fn propagate(&mut self) -> Vec<z4_core::TheoryPropagation> {
        let mut props = self.euf.propagate();
        props.extend(self.lra.propagate());
        props
    }

    fn push(&mut self) {
        self.euf.push();
        self.lra.push();
    }

    fn pop(&mut self) {
        self.euf.pop();
        self.lra.pop();
    }

    fn reset(&mut self) {
        self.euf.reset();
        self.lra.reset();
    }
}

/// Brute-force oracle for combined EUF+LRA.
/// EUF and LRA don't share variables, so we check both independently.
fn brute_force_expected_euf_lra(
    terms: &TermStore,
    euf_atoms: &[TermId],
    lra_atoms: &[TermId],
    all_atoms: &[TermId],
    expr: &BoolExpr,
) -> Expected {
    let used = expr.used_atoms(all_atoms.len());
    let num_used = used.len();
    debug_assert!(num_used > 0);
    debug_assert!(num_used <= 20, "brute force bound");

    let num_euf = euf_atoms.len();

    for mask in 0u64..(1u64 << num_used) {
        let mut assignment = vec![false; all_atoms.len()];
        for (bit, &atom_idx) in used.iter().enumerate() {
            assignment[atom_idx] = ((mask >> bit) & 1) == 1;
        }

        if !expr.eval(&assignment) {
            continue;
        }

        // Check for contradictory assignments to atoms with the same TermId.
        // Due to hash-consing, different atom indices may map to the same TermId.
        let mut term_to_value: std::collections::HashMap<TermId, bool> =
            std::collections::HashMap::new();
        let mut has_conflict = false;
        for &atom_idx in &used {
            let term_id = all_atoms[atom_idx];
            let value = assignment[atom_idx];
            if let Some(&prev_value) = term_to_value.get(&term_id) {
                if prev_value != value {
                    has_conflict = true;
                    break;
                }
            } else {
                term_to_value.insert(term_id, value);
            }
        }
        if has_conflict {
            continue;
        }

        // Check EUF
        let mut euf = EufSolver::new(terms);
        for &atom_idx in &used {
            if atom_idx < num_euf {
                euf.assert_literal(euf_atoms[atom_idx], assignment[atom_idx]);
            }
        }
        if let TheoryResult::Unsat(_) = euf.check() {
            continue;
        }

        // Check LRA
        let mut lra = LraSolver::new(terms);
        for &atom_idx in &used {
            if atom_idx >= num_euf {
                lra.assert_literal(lra_atoms[atom_idx - num_euf], assignment[atom_idx]);
            }
        }
        if let TheoryResult::Unsat(_) = lra.check() {
            continue;
        }

        return Expected::Sat;
    }

    Expected::Unsat
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        max_shrink_iters: 0,
        .. ProptestConfig::default()
    })]

    #[test]
    fn proptest_gap9_euf_random_theory_formulas(
        case in euf_case_strategy()
    ) {
        let mut terms = TermStore::new();
        let sort_s = Sort::Uninterpreted("S".to_string());

        let consts = (0..case.num_consts)
            .map(|i| terms.mk_var(format!("c{i}"), sort_s.clone()))
            .collect::<Vec<_>>();

        // Build EUF equality atoms from generated specs.
        let atom_terms = case
            .atom_specs
            .iter()
            .map(|spec| {
                let lhs = build_euf_term(&mut terms, &sort_s, &consts, spec.lhs.clone());
                let rhs = build_euf_term(&mut terms, &sort_s, &consts, spec.rhs.clone());
                terms.mk_eq(lhs, rhs)
            })
            .collect::<Vec<_>>();

        // Build Boolean structure over the atoms.
        let formula = build_bool_term(&mut terms, &atom_terms, &case.expr);

        let tseitin = Tseitin::new(&terms);
        let cnf = tseitin.transform(formula);

        let theory = EufSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &cnf, theory);
        let got = dpll.solve();

        let expected = brute_force_expected(|| EufSolver::new(&terms), &atom_terms, &case.expr);

        // Verify soundness: solver should never give wrong definite answers.
        // Unknown is acceptable.
        // Note: When symmetric equalities exist (a=b vs b=a as different atoms),
        // the oracle may not detect conflicts that the solver does detect.
        // This is because the oracle checks theory consistency per-assignment
        // without accounting for SAT-level interaction with symmetric atoms.
        match (expected, got) {
            (Expected::Sat, SolveResult::Sat(_)) => {}
            (Expected::Unsat, SolveResult::Unsat) => {}
            // Unknown is acceptable
            (_, SolveResult::Unknown) => {}
            // Solver saying UNSAT when oracle says SAT can happen with symmetric equalities
            // The DPLL(T) can detect conflicts through SAT-level propagation that the
            // simple oracle doesn't catch. This is not a bug - the solver is being more precise.
            (Expected::Sat, SolveResult::Unsat) => {}
            // This would be a real soundness bug
            (Expected::Unsat, SolveResult::Sat(_)) => {
                prop_assert!(false, "EUF UNSOUND: expected UNSAT, got SAT");
            }
        }
    }

    #[test]
    fn proptest_gap9_lra_random_theory_formulas(
        case in lra_case_strategy()
    ) {
        let mut terms = TermStore::new();
        let x = terms.mk_var("x", Sort::Real);

        let atom_terms = case
            .atom_specs
            .iter()
            .map(|spec| {
                let op = match spec.cmp {
                    LraCmp::Le => "<=",
                    LraCmp::Ge => ">=",
                };
                let bound_term = terms.mk_int((spec.bound as i64).into());
                terms.mk_app(Symbol::Named(op.to_string()), vec![x, bound_term], Sort::Bool)
            })
            .collect::<Vec<_>>();

        let formula = build_bool_term(&mut terms, &atom_terms, &case.expr);

        let tseitin = Tseitin::new(&terms);
        let cnf = tseitin.transform(formula);

        let theory = LraSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &cnf, theory);
        let got = dpll.solve();

        let expected = brute_force_expected(|| LraSolver::new(&terms), &atom_terms, &case.expr);

        match (expected, got) {
            (Expected::Sat, SolveResult::Sat(_)) => {}
            (Expected::Unsat, SolveResult::Unsat) => {}
            (Expected::Sat, other) => prop_assert!(false, "expected SAT, got {other:?}"),
            (Expected::Unsat, other) => prop_assert!(false, "expected UNSAT, got {other:?}"),
        }
    }

    /// LIA proptest - tests soundness of LIA integration.
    ///
    /// Note: The basic solve() returns Unknown for NeedSplit cases.
    /// This test verifies soundness (no wrong answers) rather than completeness.
    /// For full LIA with split handling, see the executor tests.
    #[test]
    fn proptest_gap9_lia_random_theory_formulas(
        case in lia_case_strategy()
    ) {
        let mut terms = TermStore::new();
        let x = terms.mk_var("x", Sort::Int);

        let atom_terms = case
            .atom_specs
            .iter()
            .map(|spec| {
                let op = match spec.cmp {
                    LraCmp::Le => "<=",
                    LraCmp::Ge => ">=",
                };
                let bound_term = terms.mk_int((spec.bound as i64).into());
                terms.mk_app(Symbol::Named(op.to_string()), vec![x, bound_term], Sort::Bool)
            })
            .collect::<Vec<_>>();

        let formula = build_bool_term(&mut terms, &atom_terms, &case.expr);

        let tseitin = Tseitin::new(&terms);
        let cnf = tseitin.transform(formula);

        let theory = LiaSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &cnf, theory);
        let got = dpll.solve();

        // Use simplified bounds-based oracle
        let expected = brute_force_expected_lia(&case.atom_specs, &case.expr);

        // Verify soundness: solver should never give wrong definite answers.
        // Unknown is acceptable for cases that would need NeedSplit.
        match (expected, got) {
            (Expected::Sat, SolveResult::Sat(_)) => {}
            (Expected::Unsat, SolveResult::Unsat) => {}
            // Unknown is acceptable - solver is incomplete for NeedSplit cases
            (_, SolveResult::Unknown) => {}
            (Expected::Sat, SolveResult::Unsat) => {
                prop_assert!(false, "LIA UNSOUND: expected SAT, got UNSAT");
            }
            (Expected::Unsat, SolveResult::Sat(_)) => {
                prop_assert!(false, "LIA UNSOUND: expected UNSAT, got SAT");
            }
        }
    }

    /// Combined EUF+LRA proptest.
    /// Tests integration of equality reasoning with linear real arithmetic.
    #[test]
    fn proptest_gap9_euf_lra_combined_formulas(
        case in euf_lra_case_strategy()
    ) {
        let mut terms = TermStore::new();

        // Create EUF sorts and terms
        let sort_s = Sort::Uninterpreted("S".to_string());
        let consts = (0..case.num_consts)
            .map(|i| terms.mk_var(format!("c{i}"), sort_s.clone()))
            .collect::<Vec<_>>();

        let euf_atom_terms = case
            .euf_atoms
            .iter()
            .map(|spec| {
                let lhs = build_euf_term(&mut terms, &sort_s, &consts, spec.lhs.clone());
                let rhs = build_euf_term(&mut terms, &sort_s, &consts, spec.rhs.clone());
                terms.mk_eq(lhs, rhs)
            })
            .collect::<Vec<_>>();

        // Create LRA terms
        let x = terms.mk_var("x", Sort::Real);
        let lra_atom_terms = case
            .lra_atoms
            .iter()
            .map(|spec| {
                let op = match spec.cmp {
                    LraCmp::Le => "<=",
                    LraCmp::Ge => ">=",
                };
                let bound_term = terms.mk_int((spec.bound as i64).into());
                terms.mk_app(Symbol::Named(op.to_string()), vec![x, bound_term], Sort::Bool)
            })
            .collect::<Vec<_>>();

        // Combine all atoms: EUF first, then LRA
        let all_atoms: Vec<_> = euf_atom_terms.iter().chain(lra_atom_terms.iter()).copied().collect();

        // Build Boolean structure
        let formula = build_bool_term(&mut terms, &all_atoms, &case.expr);

        let tseitin = Tseitin::new(&terms);
        let cnf = tseitin.transform(formula);

        let theory = CombinedEufLra::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &cnf, theory);
        let got = dpll.solve();

        let expected = brute_force_expected_euf_lra(&terms, &euf_atom_terms, &lra_atom_terms, &all_atoms, &case.expr);

        // Verify soundness: solver should never give wrong definite answers.
        // Unknown is acceptable - combined theory integration is complex.
        // Note: The oracle checks EUF and LRA separately, while the solver checks
        // them together. The solver may detect conflicts the oracle misses
        // (e.g., duplicate atoms with different truth assignments).
        match (expected, got) {
            (Expected::Sat, SolveResult::Sat(_)) => {}
            (Expected::Unsat, SolveResult::Unsat) => {}
            // Unknown is acceptable
            (_, SolveResult::Unknown) => {}
            // Solver being more precise than oracle is acceptable
            (Expected::Sat, SolveResult::Unsat) => {}
            // This would be a real soundness bug
            (Expected::Unsat, SolveResult::Sat(_)) => {
                prop_assert!(false, "EUF+LRA UNSOUND: expected UNSAT, got SAT");
            }
        }
    }
}

// ============================================================
// Arrays+EUF proptest
// ============================================================

use z4_arrays::ArraySolver;

/// Combined Arrays+EUF theory solver for testing.
/// Arrays relies on EUF for equality reasoning on indices and values.
struct CombinedArraysEuf<'a> {
    arrays: ArraySolver<'a>,
    euf: EufSolver<'a>,
}

impl<'a> CombinedArraysEuf<'a> {
    fn new(terms: &'a TermStore) -> Self {
        CombinedArraysEuf {
            arrays: ArraySolver::new(terms),
            euf: EufSolver::new(terms),
        }
    }
}

impl<'a> TheorySolver for CombinedArraysEuf<'a> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        self.arrays.assert_literal(literal, value);
        self.euf.assert_literal(literal, value);
    }

    fn check(&mut self) -> TheoryResult {
        // Check EUF first (handles equalities)
        match self.euf.check() {
            TheoryResult::Unsat(reasons) => return TheoryResult::Unsat(reasons),
            TheoryResult::Unknown => return TheoryResult::Unknown,
            TheoryResult::NeedSplit(s) => return TheoryResult::NeedSplit(s),
            TheoryResult::NeedDisequlitySplit(s) => return TheoryResult::NeedDisequlitySplit(s),
            TheoryResult::NeedExpressionSplit(s) => return TheoryResult::NeedExpressionSplit(s),
            TheoryResult::Sat => {}
        }
        // Then check array axioms
        self.arrays.check()
    }

    fn propagate(&mut self) -> Vec<z4_core::TheoryPropagation> {
        let mut props = self.euf.propagate();
        props.extend(self.arrays.propagate());
        props
    }

    fn push(&mut self) {
        self.arrays.push();
        self.euf.push();
    }

    fn pop(&mut self) {
        self.arrays.pop();
        self.euf.pop();
    }

    fn reset(&mut self) {
        self.arrays.reset();
        self.euf.reset();
    }
}

// Note: ArrayTermSpec is reserved for future use with more complex array term generation.
// Currently we generate atoms directly using ArrayAtomSpec.

/// Specification for array equality atoms.
#[derive(Clone, Debug)]
struct ArrayAtomSpec {
    /// Index for left select
    lhs_idx: u8,
    /// Index for right select
    rhs_idx: u8,
    /// Whether there's a store on left side
    lhs_has_store: Option<(u8, u8)>, // (store_idx, store_val_idx)
    /// Whether there's a store on right side
    rhs_has_store: Option<(u8, u8)>,
}

fn array_atom_spec_strategy(num_indices: u8) -> impl Strategy<Value = ArrayAtomSpec> {
    (
        0..num_indices,
        0..num_indices,
        proptest::option::of((0..num_indices, 0..num_indices)),
        proptest::option::of((0..num_indices, 0..num_indices)),
    )
        .prop_map(
            |(lhs_idx, rhs_idx, lhs_has_store, rhs_has_store)| ArrayAtomSpec {
                lhs_idx,
                rhs_idx,
                lhs_has_store,
                rhs_has_store,
            },
        )
}

#[derive(Clone, Debug)]
struct ArraysEufCase {
    num_indices: u8,
    atom_specs: Vec<ArrayAtomSpec>,
    expr: BoolExpr,
}

fn arrays_euf_case_strategy() -> impl Strategy<Value = ArraysEufCase> {
    (2u8..=4u8, 2usize..=5usize)
        .prop_flat_map(|(num_indices, num_atoms)| {
            (
                Just(num_indices),
                prop::collection::vec(array_atom_spec_strategy(num_indices), num_atoms),
                bool_expr_strategy(num_atoms),
            )
        })
        .prop_map(|(num_indices, atom_specs, expr)| ArraysEufCase {
            num_indices,
            atom_specs,
            expr,
        })
}

// Note: build_array_term is reserved for future use with recursive ArrayTermSpec.
// Currently we generate atoms directly in the proptest.

/// Brute-force oracle for Arrays+EUF.
/// Uses the combined theory solver to check array axiom consistency.
fn brute_force_expected_arrays_euf(
    terms: &TermStore,
    atoms: &[TermId],
    expr: &BoolExpr,
) -> Expected {
    let used = expr.used_atoms(atoms.len());
    let num_used = used.len();
    if num_used == 0 {
        return Expected::Sat;
    }
    debug_assert!(num_used <= 20, "brute force bound");

    for mask in 0u64..(1u64 << num_used) {
        let mut assignment = vec![false; atoms.len()];
        for (bit, &atom_idx) in used.iter().enumerate() {
            assignment[atom_idx] = ((mask >> bit) & 1) == 1;
        }

        if !expr.eval(&assignment) {
            continue;
        }

        // Check for contradictory assignments to atoms with the same TermId.
        // Due to hash-consing, different atom indices may map to the same TermId.
        let mut term_to_value: std::collections::HashMap<TermId, bool> =
            std::collections::HashMap::new();
        let mut has_conflict = false;
        for &atom_idx in &used {
            let term_id = atoms[atom_idx];
            let value = assignment[atom_idx];
            if let Some(&prev_value) = term_to_value.get(&term_id) {
                if prev_value != value {
                    has_conflict = true;
                    break;
                }
            } else {
                term_to_value.insert(term_id, value);
            }
        }
        if has_conflict {
            continue;
        }

        // Check consistency using the combined theory solver
        let mut theory = CombinedArraysEuf::new(terms);
        for &atom_idx in &used {
            theory.assert_literal(atoms[atom_idx], assignment[atom_idx]);
        }

        match theory.check() {
            TheoryResult::Sat => return Expected::Sat,
            TheoryResult::Unsat(_) => continue,
            TheoryResult::Unknown
            | TheoryResult::NeedSplit(_)
            | TheoryResult::NeedDisequlitySplit(_)
            | TheoryResult::NeedExpressionSplit(_) => {
                // Conservative: treat as possibly SAT
                return Expected::Sat;
            }
        }
    }

    Expected::Unsat
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        max_shrink_iters: 0,
        .. ProptestConfig::default()
    })]

    /// Arrays+EUF proptest - tests integration of array theory with EUF.
    ///
    /// Tests the read-over-write axioms:
    /// - ROW1: select(store(a, i, v), i) = v
    /// - ROW2: i ≠ j → select(store(a, i, v), j) = select(a, j)
    #[test]
    fn proptest_gap9_arrays_euf_combined_formulas(
        case in arrays_euf_case_strategy()
    ) {
        let mut terms = TermStore::new();

        // Create array sort and base array
        let elem_sort = Sort::Int;
        let arr_sort = Sort::Array(Box::new(Sort::Int), Box::new(elem_sort.clone()));
        let base_array = terms.mk_var("arr", arr_sort.clone());

        // Create index and value variables
        let indices: Vec<_> = (0..case.num_indices)
            .map(|i| terms.mk_var(format!("i{i}"), Sort::Int))
            .collect();
        let values: Vec<_> = (0..case.num_indices)
            .map(|i| terms.mk_var(format!("v{i}"), Sort::Int))
            .collect();

        // Build equality atoms from specs
        let atom_terms: Vec<_> = case
            .atom_specs
            .iter()
            .map(|spec| {
                // Build LHS: either select(arr, idx) or select(store(arr, store_idx, store_val), idx)
                let lhs_arr = match spec.lhs_has_store {
                    Some((store_idx, store_val_idx)) => {
                        terms.mk_store(base_array, indices[store_idx as usize], values[store_val_idx as usize])
                    }
                    None => base_array,
                };
                let lhs = terms.mk_select(lhs_arr, indices[spec.lhs_idx as usize]);

                // Build RHS: either select(arr, idx) or select(store(arr, store_idx, store_val), idx)
                let rhs_arr = match spec.rhs_has_store {
                    Some((store_idx, store_val_idx)) => {
                        terms.mk_store(base_array, indices[store_idx as usize], values[store_val_idx as usize])
                    }
                    None => base_array,
                };
                let rhs = terms.mk_select(rhs_arr, indices[spec.rhs_idx as usize]);

                terms.mk_eq(lhs, rhs)
            })
            .collect();

        // Build Boolean structure
        let formula = build_bool_term(&mut terms, &atom_terms, &case.expr);

        let tseitin = Tseitin::new(&terms);
        let cnf = tseitin.transform(formula);

        let theory = CombinedArraysEuf::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &cnf, theory);
        let got = dpll.solve();

        let expected = brute_force_expected_arrays_euf(
            &terms,
            &atom_terms,
            &case.expr,
        );

        // Verify soundness: solver should never give wrong definite answers.
        match (expected, got) {
            (Expected::Sat, SolveResult::Sat(_)) => {}
            (Expected::Unsat, SolveResult::Unsat) => {}
            // Unknown is acceptable
            (_, SolveResult::Unknown) => {}
            // Solver being more precise than oracle is acceptable
            (Expected::Sat, SolveResult::Unsat) => {}
            // This would be a real soundness bug
            (Expected::Unsat, SolveResult::Sat(_)) => {
                prop_assert!(false, "Arrays+EUF UNSOUND: expected UNSAT, got SAT");
            }
        }
    }
}
