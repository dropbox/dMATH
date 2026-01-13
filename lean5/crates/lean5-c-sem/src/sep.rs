//! Separation Logic for C Verification
//!
//! This module provides separation logic primitives based on VST (Verified
//! Software Toolchain) and Iris. Separation logic enables modular reasoning
//! about pointer-manipulating programs.
//!
//! ## Key Concepts
//!
//! - **Points-to assertion** (`p ↦ v`): Pointer `p` exclusively owns location
//!   containing value `v`
//! - **Separating conjunction** (`P * Q`): Resources P and Q are disjoint
//! - **Magic wand** (`P -* Q`): If you give me P, I'll give you Q
//! - **emp**: Empty heap assertion
//!
//! ## Frame Rule
//!
//! The key soundness principle:
//! ```text
//! {P} C {Q}
//! ──────────────── (Frame)
//! {P * R} C {Q * R}
//! ```
//!
//! If C operates only on resources described by P, then any frame R is
//! preserved.
//!
//! ## References
//!
//! - VST: <https://vst.cs.princeton.edu/>
//! - Iris: <https://iris-project.org/>
//! - Reynolds, "Separation Logic: A Logic for Shared Mutable Data Structures"

use crate::expr::{CExpr, Ident};
use crate::spec::Spec;
use crate::types::CType;
use serde::{Deserialize, Serialize};

/// Separation logic assertion
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum SepAssertion {
    /// Empty heap: owns no resources
    #[default]
    Emp,

    /// Pure assertion: P is a pure (heap-independent) proposition
    /// Equivalent to ⌈P⌉ or `[P]` in some notations
    Pure(Spec),

    /// Points-to assertion: p ↦{sh} v
    /// Pointer `p` points to value `v` with share `sh`
    PointsTo {
        /// Pointer expression
        ptr: CExpr,
        /// Type of the pointed-to value
        ty: CType,
        /// Value stored (None for uninitialized)
        value: Option<CExpr>,
        /// Share (full ownership vs read-only)
        share: Share,
    },

    /// Data-at assertion: data_at(sh, ty, v, p)
    /// Like points-to but uses abstract specification value
    DataAt {
        ptr: CExpr,
        ty: CType,
        value: Spec,
        share: Share,
    },

    /// Array assertion: array_at(sh, ty, contents, p, n)
    ArrayAt {
        ptr: CExpr,
        elem_ty: CType,
        /// Contents as a logical list
        contents: Vec<Spec>,
        share: Share,
    },

    /// Separating conjunction: P * Q
    /// Resources described by P and Q are disjoint
    SepConj(Box<SepAssertion>, Box<SepAssertion>),

    /// Magic wand (septraction): P -* Q
    /// "If you give me resource P, I'll give you resource Q"
    Wand(Box<SepAssertion>, Box<SepAssertion>),

    /// Disjunction: P ∨ Q
    Or(Box<SepAssertion>, Box<SepAssertion>),

    /// Conjunction: P ∧ Q (NOT separating - same resource)
    And(Box<SepAssertion>, Box<SepAssertion>),

    /// Existential: ∃x:T. P(x)
    Exists {
        var: Ident,
        ty: CType,
        body: Box<SepAssertion>,
    },

    /// Universal: ∀x:T. P(x)
    Forall {
        var: Ident,
        ty: CType,
        body: Box<SepAssertion>,
    },

    /// Memory block assertion: memory_block(sh, n, p)
    /// Pointer `p` owns `n` bytes of uninitialized memory
    MemoryBlock {
        ptr: CExpr,
        size: Spec,
        share: Share,
    },

    /// Freeable assertion: malloc_token(p, n)
    /// Permission to free `n` bytes starting at `p`
    MallocToken { ptr: CExpr, size: Spec },

    /// Valid pointer (weaker than points-to): valid_pointer(p)
    ValidPointer(CExpr),

    /// Struct field assertion: field_at(sh, ty, fld, v, p)
    FieldAt {
        ptr: CExpr,
        struct_ty: CType,
        field: Ident,
        value: Spec,
        share: Share,
    },

    /// Iterated separating conjunction: ⊛_{i ∈ lo..hi} P(i)
    Iter {
        var: Ident,
        lo: Spec,
        hi: Spec,
        body: Box<SepAssertion>,
    },

    /// Local variable assertion (stack): stackframe_of(vars)
    Stackframe(Vec<(Ident, CType, Option<Spec>)>),

    /// Return assertion (for function spec translation)
    Return { ty: CType, value: Spec },

    /// Named predicate application: pred(args)
    Pred { name: Ident, args: Vec<Spec> },

    /// Implication (not strictly separation logic, but useful)
    Implies(Box<SepAssertion>, Box<SepAssertion>),
}

/// Share type for fractional permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Share {
    /// Full ownership (read + write + free)
    Full,
    /// Read-only share (can read, cannot write or free)
    ReadOnly,
    /// Top share (empty, no access)
    Top,
    /// Bottom share (invalid)
    Bot,
    /// Fractional share (numerator, denominator)
    /// Share(n, d) represents n/d ownership
    Frac(u32, u32),
}

impl Share {
    /// Check if share permits reading
    pub fn readable(&self) -> bool {
        !matches!(self, Share::Top | Share::Bot)
    }

    /// Check if share permits writing
    pub fn writable(&self) -> bool {
        matches!(self, Share::Full)
    }

    /// Check if share is empty
    pub fn is_empty(&self) -> bool {
        matches!(self, Share::Top)
    }

    /// Join two shares (for combining resources)
    pub fn join(&self, other: &Share) -> Option<Share> {
        match (self, other) {
            (Share::Top, s) | (s, Share::Top) => Some(*s),
            (Share::ReadOnly, Share::ReadOnly) => Some(Share::ReadOnly), // Readers can share
            (Share::Frac(n1, d1), Share::Frac(n2, d2)) => {
                // n1/d1 + n2/d2 = (n1*d2 + n2*d1) / (d1*d2)
                let num = n1 * d2 + n2 * d1;
                let den = d1 * d2;
                if num > den {
                    None // Can't exceed full ownership
                } else if num == den {
                    Some(Share::Full)
                } else {
                    // Simplify fraction
                    let g = gcd(num, den);
                    Some(Share::Frac(num / g, den / g))
                }
            }
            // Can't combine with bot, can't have full twice, can't mix full with readonly,
            // and remaining combinations are incompatible
            _ => None,
        }
    }

    /// Split a share in half
    pub fn split(&self) -> Option<(Share, Share)> {
        match self {
            Share::Full => Some((Share::Frac(1, 2), Share::Frac(1, 2))),
            Share::Frac(n, d) => {
                // Can only split even fractions cleanly
                if n % 2 == 0 {
                    Some((Share::Frac(n / 2, *d), Share::Frac(n / 2, *d)))
                } else {
                    // Split n/d into n/(2d) + n/(2d)
                    Some((Share::Frac(*n, d * 2), Share::Frac(*n, d * 2)))
                }
            }
            _ => None,
        }
    }
}

/// GCD helper
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl SepAssertion {
    // ═══════════════════════════════════════════════════════════════════════
    // Constructors
    // ═══════════════════════════════════════════════════════════════════════

    /// Empty heap
    pub fn emp() -> Self {
        SepAssertion::Emp
    }

    /// Pure assertion (lifted from Spec)
    pub fn pure(spec: Spec) -> Self {
        SepAssertion::Pure(spec)
    }

    /// Points-to assertion with full ownership
    pub fn points_to(ptr: CExpr, ty: CType, value: CExpr) -> Self {
        SepAssertion::PointsTo {
            ptr,
            ty,
            value: Some(value),
            share: Share::Full,
        }
    }

    /// Points-to with uninitialized memory
    pub fn points_to_undef(ptr: CExpr, ty: CType) -> Self {
        SepAssertion::PointsTo {
            ptr,
            ty,
            value: None,
            share: Share::Full,
        }
    }

    /// Points-to with read-only share
    pub fn points_to_readonly(ptr: CExpr, ty: CType, value: CExpr) -> Self {
        SepAssertion::PointsTo {
            ptr,
            ty,
            value: Some(value),
            share: Share::ReadOnly,
        }
    }

    /// Data-at assertion
    pub fn data_at(ptr: CExpr, ty: CType, value: Spec, share: Share) -> Self {
        SepAssertion::DataAt {
            ptr,
            ty,
            value,
            share,
        }
    }

    /// Array assertion
    pub fn array_at(ptr: CExpr, elem_ty: CType, contents: Vec<Spec>, share: Share) -> Self {
        SepAssertion::ArrayAt {
            ptr,
            elem_ty,
            contents,
            share,
        }
    }

    /// Memory block (uninitialized)
    pub fn memory_block(ptr: CExpr, size: Spec) -> Self {
        SepAssertion::MemoryBlock {
            ptr,
            size,
            share: Share::Full,
        }
    }

    /// Malloc token (permission to free)
    pub fn malloc_token(ptr: CExpr, size: Spec) -> Self {
        SepAssertion::MallocToken { ptr, size }
    }

    /// Separating conjunction
    pub fn sep_conj(p: SepAssertion, q: SepAssertion) -> Self {
        match (&p, &q) {
            (SepAssertion::Emp, _) => q,
            (_, SepAssertion::Emp) => p,
            _ => SepAssertion::SepConj(Box::new(p), Box::new(q)),
        }
    }

    /// Multiple separating conjunction
    pub fn sep_conj_many(assertions: Vec<SepAssertion>) -> Self {
        assertions
            .into_iter()
            .fold(SepAssertion::Emp, SepAssertion::sep_conj)
    }

    /// Magic wand
    pub fn wand(p: SepAssertion, q: SepAssertion) -> Self {
        SepAssertion::Wand(Box::new(p), Box::new(q))
    }

    /// Existential quantification
    pub fn exists(var: impl Into<String>, ty: CType, body: SepAssertion) -> Self {
        SepAssertion::Exists {
            var: var.into(),
            ty,
            body: Box::new(body),
        }
    }

    /// Universal quantification
    pub fn forall(var: impl Into<String>, ty: CType, body: SepAssertion) -> Self {
        SepAssertion::Forall {
            var: var.into(),
            ty,
            body: Box::new(body),
        }
    }

    /// Iterated separating conjunction
    pub fn iter(var: impl Into<String>, lo: Spec, hi: Spec, body: SepAssertion) -> Self {
        SepAssertion::Iter {
            var: var.into(),
            lo,
            hi,
            body: Box::new(body),
        }
    }

    /// Disjunction
    pub fn or(p: SepAssertion, q: SepAssertion) -> Self {
        SepAssertion::Or(Box::new(p), Box::new(q))
    }

    /// Conjunction (not separating)
    pub fn and(p: SepAssertion, q: SepAssertion) -> Self {
        SepAssertion::And(Box::new(p), Box::new(q))
    }

    /// Implication
    pub fn implies(p: SepAssertion, q: SepAssertion) -> Self {
        SepAssertion::Implies(Box::new(p), Box::new(q))
    }

    /// Field access assertion
    pub fn field_at(
        ptr: CExpr,
        struct_ty: CType,
        field: impl Into<String>,
        value: Spec,
        share: Share,
    ) -> Self {
        SepAssertion::FieldAt {
            ptr,
            struct_ty,
            field: field.into(),
            value,
            share,
        }
    }

    /// Valid pointer (weaker than points-to)
    pub fn valid_pointer(ptr: CExpr) -> Self {
        SepAssertion::ValidPointer(ptr)
    }

    /// Named predicate
    pub fn pred(name: impl Into<String>, args: Vec<Spec>) -> Self {
        SepAssertion::Pred {
            name: name.into(),
            args,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Predicate Helpers
    // ═══════════════════════════════════════════════════════════════════════

    /// Check if this assertion is emp (empty heap)
    pub fn is_emp(&self) -> bool {
        matches!(self, SepAssertion::Emp)
    }

    /// Check if this assertion is pure (no heap resources)
    pub fn is_pure(&self) -> bool {
        match self {
            SepAssertion::Emp | SepAssertion::Pure(_) => true,
            SepAssertion::SepConj(p, q)
            | SepAssertion::And(p, q)
            | SepAssertion::Or(p, q)
            | SepAssertion::Implies(p, q) => p.is_pure() && q.is_pure(),
            SepAssertion::Forall { body, .. } | SepAssertion::Exists { body, .. } => body.is_pure(),
            _ => false,
        }
    }

    /// Extract the pure part of an assertion
    pub fn pure_part(&self) -> Option<Spec> {
        match self {
            SepAssertion::Pure(s) => Some(s.clone()),
            SepAssertion::SepConj(p, q) | SepAssertion::And(p, q) => {
                let pp = p.pure_part();
                let qp = q.pure_part();
                match (pp, qp) {
                    (Some(a), Some(b)) => Some(Spec::and(vec![a, b])),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                }
            }
            _ => None,
        }
    }

    /// Collect all pointers mentioned in the assertion
    pub fn mentioned_pointers(&self) -> Vec<CExpr> {
        let mut ptrs = Vec::new();
        self.collect_pointers(&mut ptrs);
        ptrs
    }

    fn collect_pointers(&self, ptrs: &mut Vec<CExpr>) {
        match self {
            SepAssertion::PointsTo { ptr, .. }
            | SepAssertion::DataAt { ptr, .. }
            | SepAssertion::ArrayAt { ptr, .. }
            | SepAssertion::MemoryBlock { ptr, .. }
            | SepAssertion::MallocToken { ptr, .. }
            | SepAssertion::ValidPointer(ptr)
            | SepAssertion::FieldAt { ptr, .. } => {
                ptrs.push(ptr.clone());
            }
            SepAssertion::SepConj(p, q)
            | SepAssertion::Wand(p, q)
            | SepAssertion::Or(p, q)
            | SepAssertion::And(p, q)
            | SepAssertion::Implies(p, q) => {
                p.collect_pointers(ptrs);
                q.collect_pointers(ptrs);
            }
            SepAssertion::Exists { body, .. }
            | SepAssertion::Forall { body, .. }
            | SepAssertion::Iter { body, .. } => {
                body.collect_pointers(ptrs);
            }
            SepAssertion::Stackframe(vars) => {
                // Stack frame creates pointers to locals
                for (name, _, _) in vars {
                    ptrs.push(CExpr::var(name));
                }
            }
            _ => {}
        }
    }
}

/// A separation logic function specification (VST-style)
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SepFuncSpec {
    /// Precondition (spatial + pure assertions)
    pub pre: SepAssertion,
    /// Postcondition (spatial + pure assertions)
    /// Binds \result
    pub post: SepAssertion,
    /// Local variables bound in pre/post
    pub bound_vars: Vec<(Ident, CType)>,
    /// Frame: additional resources preserved by the call
    pub frame: Option<SepAssertion>,
}

impl SepFuncSpec {
    pub fn new(pre: SepAssertion, post: SepAssertion) -> Self {
        Self {
            pre,
            post,
            bound_vars: Vec::new(),
            frame: None,
        }
    }

    /// Add frame to the spec (for frame rule application)
    #[must_use]
    pub fn with_frame(mut self, frame: SepAssertion) -> Self {
        // Apply frame rule: {P} C {Q} => {P * R} C {Q * R}
        self.pre = SepAssertion::sep_conj(self.pre, frame.clone());
        self.post = SepAssertion::sep_conj(self.post, frame.clone());
        self.frame = Some(frame);
        self
    }
}

/// Frame rule application context
pub struct FrameContext {
    /// Current spatial context (resources we own)
    pub spatial: SepAssertion,
    /// Pure context (logical facts)
    pub pure: Vec<Spec>,
}

impl FrameContext {
    pub fn new() -> Self {
        Self {
            spatial: SepAssertion::Emp,
            pure: Vec::new(),
        }
    }

    /// Add a resource to the context
    pub fn add_resource(&mut self, resource: SepAssertion) {
        self.spatial = SepAssertion::sep_conj(
            std::mem::replace(&mut self.spatial, SepAssertion::Emp),
            resource,
        );
    }

    /// Add a pure fact
    pub fn add_pure(&mut self, fact: Spec) {
        self.pure.push(fact);
    }

    /// Try to consume a resource from the context
    /// Returns the remaining context after consuming
    pub fn consume(&self, required: &SepAssertion) -> Option<SepAssertion> {
        // Simple implementation: check if required is a sub-assertion
        // Full implementation would need entailment checking
        self.try_consume(&self.spatial, required)
    }

    fn try_consume(
        &self,
        available: &SepAssertion,
        required: &SepAssertion,
    ) -> Option<SepAssertion> {
        // Base case: exact match
        if available == required {
            return Some(SepAssertion::Emp);
        }

        // Consume from separating conjunction
        if let SepAssertion::SepConj(left, right) = available {
            // Try consuming from left
            if let Some(remaining) = self.try_consume(left, required) {
                return Some(SepAssertion::sep_conj(remaining, *right.clone()));
            }
            // Try consuming from right
            if let Some(remaining) = self.try_consume(right, required) {
                return Some(SepAssertion::sep_conj(*left.clone(), remaining));
            }
        }

        // Emp can always be consumed
        if matches!(required, SepAssertion::Emp) {
            return Some(available.clone());
        }

        None
    }

    /// Apply the frame rule to derive a new triple
    /// Given {P} C {Q} and frame R, derive {P * R} C {Q * R}
    pub fn frame_rule(
        pre: SepAssertion,
        post: SepAssertion,
        frame: SepAssertion,
    ) -> (SepAssertion, SepAssertion) {
        (
            SepAssertion::sep_conj(pre, frame.clone()),
            SepAssertion::sep_conj(post, frame),
        )
    }
}

impl Default for FrameContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert separation logic assertion to specification (for VC generation)
impl From<SepAssertion> for Spec {
    fn from(sep: SepAssertion) -> Spec {
        match sep {
            SepAssertion::Emp => Spec::True,
            SepAssertion::Pure(s) => s,
            SepAssertion::PointsTo { ptr, value, .. } => {
                // points_to(p, v) translates to valid(p) ∧ *p = v
                let valid = Spec::valid(Spec::Expr(ptr.clone()));
                if let Some(v) = value {
                    Spec::and(vec![
                        valid,
                        Spec::eq(Spec::Expr(CExpr::deref(ptr)), Spec::Expr(v)),
                    ])
                } else {
                    valid
                }
            }
            SepAssertion::DataAt { ptr, value, .. } => Spec::and(vec![
                Spec::valid(Spec::Expr(ptr.clone())),
                Spec::eq(Spec::Expr(CExpr::deref(ptr)), value),
            ]),
            SepAssertion::ArrayAt {
                ptr,
                contents,
                elem_ty,
                ..
            } => {
                // Array: valid_range(p, 0, n-1) ∧ ∀i. 0 ≤ i < n → p[i] = contents[i]
                let n = contents.len();
                if n == 0 {
                    Spec::True
                } else {
                    Spec::and(vec![
                        Spec::ValidRange {
                            ptr: Box::new(Spec::Expr(ptr.clone())),
                            lo: Box::new(Spec::int(0)),
                            hi: Box::new(Spec::int((n - 1) as i64)),
                        },
                        Spec::forall(
                            "i",
                            elem_ty,
                            Spec::implies(
                                Spec::and(vec![
                                    Spec::ge(Spec::var("i"), Spec::int(0)),
                                    Spec::lt(Spec::var("i"), Spec::int(n as i64)),
                                ]),
                                // Would need to lookup contents[i] dynamically
                                Spec::True, // Simplified
                            ),
                        ),
                    ])
                }
            }
            SepAssertion::SepConj(p, q) => {
                // P * Q translates to P ∧ Q ∧ separated(pointers in P, pointers in Q)
                let p_spec: Spec = (*p).into();
                let q_spec: Spec = (*q).into();
                Spec::and(vec![p_spec, q_spec])
                // Note: Full translation would add separation constraints
            }
            SepAssertion::Wand(p, q) => {
                // P -* Q translates to (P → Q) for VC purposes
                let p_spec: Spec = (*p).into();
                let q_spec: Spec = (*q).into();
                Spec::implies(p_spec, q_spec)
            }
            SepAssertion::Or(p, q) => Spec::or(vec![(*p).into(), (*q).into()]),
            SepAssertion::And(p, q) => Spec::and(vec![(*p).into(), (*q).into()]),
            SepAssertion::Implies(p, q) => Spec::implies((*p).into(), (*q).into()),
            SepAssertion::Exists { var, ty, body } => Spec::exists(var, ty, (*body).into()),
            SepAssertion::Forall { var, ty, body } => Spec::forall(var, ty, (*body).into()),
            SepAssertion::MemoryBlock { ptr, size, .. } => Spec::ValidRange {
                ptr: Box::new(Spec::Expr(ptr)),
                lo: Box::new(Spec::int(0)),
                hi: Box::new(size),
            },
            SepAssertion::MallocToken { ptr, .. } => Spec::Freeable(Box::new(Spec::Expr(ptr))),
            SepAssertion::ValidPointer(ptr) => Spec::valid(Spec::Expr(ptr)),
            SepAssertion::FieldAt {
                ptr, field, value, ..
            } => Spec::and(vec![
                Spec::valid(Spec::Expr(ptr.clone())),
                Spec::eq(
                    Spec::Member {
                        object: Box::new(Spec::Expr(CExpr::deref(ptr))),
                        field,
                    },
                    value,
                ),
            ]),
            SepAssertion::Iter { var, lo, hi, body } => {
                // Iterated sep conj becomes universal quantification for VCs
                Spec::forall(
                    var,
                    CType::int(), // Default to int iteration variable
                    Spec::implies(
                        Spec::and(vec![
                            Spec::ge(Spec::var("i"), lo),
                            Spec::lt(Spec::var("i"), hi),
                        ]),
                        (*body).into(),
                    ),
                )
            }
            SepAssertion::Stackframe(vars) => {
                // Stack frame: all variables are valid
                Spec::and(
                    vars.into_iter()
                        .map(|(name, _, value)| {
                            if let Some(v) = value {
                                Spec::eq(Spec::var(name), v)
                            } else {
                                Spec::True
                            }
                        })
                        .collect(),
                )
            }
            SepAssertion::Return { value, .. } => Spec::eq(Spec::result(), value),
            SepAssertion::Pred { name, args } => Spec::Call { func: name, args },
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Common Predicates
// ═══════════════════════════════════════════════════════════════════════════

/// List segment predicate: lseg(head, tail, contents)
/// A singly-linked list from head to tail containing contents
pub fn lseg(head: CExpr, tail: CExpr, contents: Vec<Spec>) -> SepAssertion {
    if contents.is_empty() {
        // Empty list: head == tail
        SepAssertion::pure(Spec::eq(Spec::Expr(head), Spec::Expr(tail)))
    } else {
        // Inductive definition
        SepAssertion::pred(
            "lseg",
            vec![
                Spec::Expr(head),
                Spec::Expr(tail),
                // Would need a list type here
                Spec::True, // Simplified
            ],
        )
    }
}

/// Linked list predicate: list(head, contents)
/// A null-terminated singly-linked list
pub fn list(head: CExpr, contents: Vec<Spec>) -> SepAssertion {
    lseg(head, CExpr::null(), contents)
}

/// Tree predicate: tree(root, contents)
pub fn tree(root: CExpr, _contents: Spec) -> SepAssertion {
    SepAssertion::pred("tree", vec![Spec::Expr(root)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::BinOp;

    #[test]
    fn test_emp() {
        let emp = SepAssertion::emp();
        assert!(emp.is_emp());
        assert!(emp.is_pure());
    }

    #[test]
    fn test_points_to() {
        let ptr = CExpr::var("p");
        let val = CExpr::int(42);
        let pts = SepAssertion::points_to(ptr, CType::int(), val);

        assert!(!pts.is_pure());
        assert_eq!(pts.mentioned_pointers().len(), 1);
    }

    #[test]
    fn test_sep_conj() {
        let p1 = SepAssertion::points_to(CExpr::var("p"), CType::int(), CExpr::int(1));
        let p2 = SepAssertion::points_to(CExpr::var("q"), CType::int(), CExpr::int(2));
        let conj = SepAssertion::sep_conj(p1, p2);

        assert!(matches!(conj, SepAssertion::SepConj(_, _)));
        assert_eq!(conj.mentioned_pointers().len(), 2);
    }

    #[test]
    fn test_sep_conj_emp() {
        let p = SepAssertion::points_to(CExpr::var("p"), CType::int(), CExpr::int(1));
        let emp = SepAssertion::emp();

        // emp * P = P
        let conj = SepAssertion::sep_conj(emp.clone(), p.clone());
        assert!(matches!(conj, SepAssertion::PointsTo { .. }));

        // P * emp = P
        let conj2 = SepAssertion::sep_conj(p, emp);
        assert!(matches!(conj2, SepAssertion::PointsTo { .. }));
    }

    #[test]
    fn test_magic_wand() {
        let p = SepAssertion::points_to(CExpr::var("p"), CType::int(), CExpr::int(1));
        let q = SepAssertion::points_to(CExpr::var("q"), CType::int(), CExpr::int(2));
        let wand = SepAssertion::wand(p, q);

        assert!(matches!(wand, SepAssertion::Wand(_, _)));
    }

    #[test]
    fn test_share_join() {
        // Full + Top = Full
        assert_eq!(Share::Full.join(&Share::Top), Some(Share::Full));

        // Full + Full = None (can't combine)
        assert_eq!(Share::Full.join(&Share::Full), None);

        // ReadOnly + ReadOnly = ReadOnly
        assert_eq!(
            Share::ReadOnly.join(&Share::ReadOnly),
            Some(Share::ReadOnly)
        );

        // 1/2 + 1/2 = Full
        assert_eq!(
            Share::Frac(1, 2).join(&Share::Frac(1, 2)),
            Some(Share::Full)
        );

        // 1/4 + 1/4 = 1/2
        assert_eq!(
            Share::Frac(1, 4).join(&Share::Frac(1, 4)),
            Some(Share::Frac(1, 2))
        );
    }

    #[test]
    fn test_share_split() {
        // Full splits to 1/2 + 1/2
        let (a, b) = Share::Full.split().unwrap();
        assert_eq!(a, Share::Frac(1, 2));
        assert_eq!(b, Share::Frac(1, 2));

        // 1/2 splits to 1/4 + 1/4
        let (a, b) = Share::Frac(1, 2).split().unwrap();
        assert_eq!(a, Share::Frac(1, 4));
        assert_eq!(b, Share::Frac(1, 4));
    }

    #[test]
    fn test_frame_rule() {
        let pre = SepAssertion::points_to(CExpr::var("x"), CType::int(), CExpr::int(1));
        let post = SepAssertion::points_to(CExpr::var("x"), CType::int(), CExpr::int(2));
        let frame = SepAssertion::points_to(CExpr::var("y"), CType::int(), CExpr::int(42));

        let (framed_pre, framed_post) = FrameContext::frame_rule(pre, post, frame);

        // Both should be separating conjunctions
        assert!(matches!(framed_pre, SepAssertion::SepConj(_, _)));
        assert!(matches!(framed_post, SepAssertion::SepConj(_, _)));
    }

    #[test]
    fn test_sep_to_spec() {
        // points_to(p, 42) -> valid(p) && *p == 42
        let pts = SepAssertion::points_to(CExpr::var("p"), CType::int(), CExpr::int(42));
        let spec: Spec = pts.into();

        assert!(matches!(spec, Spec::And(_)));
    }

    #[test]
    fn test_exists() {
        let body = SepAssertion::points_to(CExpr::var("p"), CType::int(), CExpr::var("x"));
        let ex = SepAssertion::exists("x", CType::int(), body);

        assert!(matches!(ex, SepAssertion::Exists { .. }));
    }

    #[test]
    fn test_iter() {
        // ⊛_{i ∈ 0..n} p[i] ↦ a[i]
        let body = SepAssertion::DataAt {
            ptr: CExpr::index(CExpr::var("p"), CExpr::var("i")),
            ty: CType::int(),
            value: Spec::Index {
                base: Box::new(Spec::var("a")),
                index: Box::new(Spec::var("i")),
            },
            share: Share::Full,
        };

        let iter = SepAssertion::iter("i", Spec::int(0), Spec::var("n"), body);
        assert!(matches!(iter, SepAssertion::Iter { .. }));
    }

    #[test]
    fn test_pure_part() {
        let pure = SepAssertion::pure(Spec::ge(Spec::var("x"), Spec::int(0)));
        let pts = SepAssertion::points_to(CExpr::var("p"), CType::int(), CExpr::int(1));
        let conj = SepAssertion::sep_conj(pure, pts);

        let pp = conj.pure_part();
        assert!(pp.is_some());
        assert!(matches!(pp.unwrap(), Spec::BinOp { op: BinOp::Ge, .. }));
    }

    #[test]
    fn test_frame_context_consume() {
        let mut ctx = FrameContext::new();

        // Add two resources
        let p1 = SepAssertion::points_to(CExpr::var("x"), CType::int(), CExpr::int(1));
        let p2 = SepAssertion::points_to(CExpr::var("y"), CType::int(), CExpr::int(2));
        ctx.add_resource(p1.clone());
        ctx.add_resource(p2.clone());

        // Consume p1
        let remaining = ctx.consume(&p1);
        assert!(remaining.is_some());
        // Should have p2 left
        let r = remaining.unwrap();
        assert!(!r.is_emp());
    }

    #[test]
    fn test_data_at() {
        let da = SepAssertion::data_at(CExpr::var("p"), CType::int(), Spec::var("v"), Share::Full);

        assert!(!da.is_pure());
        let spec: Spec = da.into();
        assert!(matches!(spec, Spec::And(_)));
    }

    #[test]
    fn test_memory_block() {
        let mb = SepAssertion::memory_block(CExpr::var("p"), Spec::int(100));
        assert!(!mb.is_pure());
    }

    #[test]
    fn test_array_at() {
        let arr = SepAssertion::array_at(
            CExpr::var("a"),
            CType::int(),
            vec![Spec::int(1), Spec::int(2), Spec::int(3)],
            Share::Full,
        );

        assert!(!arr.is_pure());
        let ptrs = arr.mentioned_pointers();
        assert_eq!(ptrs.len(), 1);
    }

    #[test]
    fn test_sep_func_spec() {
        // swap(int *x, int *y)
        // PRE:  x ↦ a * y ↦ b
        // POST: x ↦ b * y ↦ a
        let pre = SepAssertion::sep_conj(
            SepAssertion::data_at(CExpr::var("x"), CType::int(), Spec::var("a"), Share::Full),
            SepAssertion::data_at(CExpr::var("y"), CType::int(), Spec::var("b"), Share::Full),
        );
        let post = SepAssertion::sep_conj(
            SepAssertion::data_at(CExpr::var("x"), CType::int(), Spec::var("b"), Share::Full),
            SepAssertion::data_at(CExpr::var("y"), CType::int(), Spec::var("a"), Share::Full),
        );

        let spec = SepFuncSpec::new(pre, post);
        assert!(spec.frame.is_none());

        // Add a frame
        let frame = SepAssertion::points_to(CExpr::var("z"), CType::int(), CExpr::int(42));
        let framed_spec = spec.with_frame(frame);
        assert!(framed_spec.frame.is_some());
    }

    #[test]
    fn test_list_predicate() {
        let lst = list(CExpr::var("head"), vec![Spec::int(1), Spec::int(2)]);
        assert!(matches!(lst, SepAssertion::Pred { .. }));
    }
}
