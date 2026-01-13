//! Universe levels
//!
//! Universe levels form a well-founded partial order used to stratify types
//! and avoid Russell's paradox.
//!
//! Key properties:
//! - `imax(l1, l2) = 0` if `l2 = 0`, otherwise `max(l1, l2)`
//! - This is used for Prop-elimination: `(x : Prop) → T` should have level `imax(0, level(T))`
//!   which is `level(T)` if `T` is a type, but `0` if `T` is also Prop.

use crate::name::Name;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Universe level
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Level {
    /// Zero (the lowest level)
    Zero,
    /// Successor: l + 1
    Succ(Arc<Level>),
    /// Maximum: max(l1, l2)
    Max(Arc<Level>, Arc<Level>),
    /// Impredicative maximum: imax(l1, l2) = 0 if l2 = 0, else max(l1, l2)
    IMax(Arc<Level>, Arc<Level>),
    /// Universe parameter (polymorphism)
    Param(Name),
}

impl Level {
    /// Create zero level
    pub fn zero() -> Self {
        Level::Zero
    }

    /// Create successor level, simplifying if possible
    pub fn succ(l: Level) -> Self {
        Level::Succ(Arc::new(l))
    }

    /// Create max level, simplifying if possible
    pub fn max(l1: Level, l2: Level) -> Self {
        // Simplifications:
        // max(l, l) = l
        // max(0, l) = l
        // max(l, 0) = l
        if l1 == l2 {
            return l1;
        }
        if l1.is_zero() {
            return l2;
        }
        if l2.is_zero() {
            return l1;
        }
        // Check if one is definitely >= the other
        if Level::is_geq(&l1, &l2) {
            return l1;
        }
        if Level::is_geq(&l2, &l1) {
            return l2;
        }
        Level::Max(Arc::new(l1), Arc::new(l2))
    }

    /// Create imax level, simplifying if possible
    ///
    /// imax(l1, l2) = 0 if l2 = 0, else max(l1, l2)
    pub fn imax(l1: Level, l2: Level) -> Self {
        // imax(l, 0) = 0
        if l2.is_zero() {
            return Level::Zero;
        }
        // imax(l, succ(l')) = max(l, succ(l')) since succ(l') > 0
        if matches!(l2, Level::Succ(_)) {
            return Level::max(l1, l2);
        }
        // imax(0, l) = l (if l != 0, which we handled above)
        if l1.is_zero() {
            return l2;
        }
        // imax(l, l) = l
        if l1 == l2 {
            return l1;
        }
        Level::IMax(Arc::new(l1), Arc::new(l2))
    }

    /// Create parameter level
    pub fn param(name: Name) -> Self {
        Level::Param(name)
    }

    /// Check if this is definitely zero
    pub fn is_zero(&self) -> bool {
        match self {
            Level::Zero => true,
            Level::Succ(_) | Level::Param(_) => false, // Succ always > 0; Params might be 0 at runtime
            Level::Max(l1, l2) => l1.is_zero() && l2.is_zero(),
            Level::IMax(_, l2) => l2.is_zero(), // imax(_, 0) = 0
        }
    }

    /// Check if this is definitely nonzero (i.e., definitely > 0)
    pub fn is_nonzero(&self) -> bool {
        match self {
            Level::Zero | Level::Param(_) => false, // Zero is 0; Params might be 0
            Level::Succ(_) => true,                 // succ(l) > 0 for all l
            Level::Max(l1, l2) => l1.is_nonzero() || l2.is_nonzero(),
            Level::IMax(_, l2) => l2.is_nonzero(), // If l2 > 0, imax reduces to max
        }
    }

    /// Get the base level and offset (number of Succ applications)
    /// e.g., succ(succ(u)) => (u, 2)
    pub fn get_offset(&self) -> (&Level, u32) {
        match self {
            Level::Succ(inner) => {
                let (base, offset) = inner.get_offset();
                (base, offset + 1)
            }
            _ => (self, 0),
        }
    }

    /// Add an offset to a level
    #[must_use]
    pub fn add_offset(&self, n: u32) -> Level {
        if n == 0 {
            self.clone()
        } else {
            Level::succ(self.add_offset(n - 1))
        }
    }

    /// Normalize the level to a canonical form
    #[must_use]
    pub fn normalize(&self) -> Level {
        match self {
            Level::Zero => Level::Zero,
            Level::Succ(l) => Level::succ(l.normalize()),
            Level::Max(l1, l2) => Level::max(l1.normalize(), l2.normalize()),
            Level::IMax(l1, l2) => {
                let l2_norm = l2.normalize();
                // imax(l1, 0) = 0
                if l2_norm.is_zero() {
                    return Level::Zero;
                }
                // imax(l1, succ(l')) = max(l1, succ(l'))
                if l2_norm.is_nonzero() {
                    return Level::max(l1.normalize(), l2_norm);
                }
                Level::imax(l1.normalize(), l2_norm)
            }
            Level::Param(_) => self.clone(),
        }
    }

    /// Check if l1 ≥ l2 (l1 is greater than or equal to l2)
    ///
    /// This is a conservative approximation - returns true only if definitely ≥
    pub fn is_geq(l1: &Level, l2: &Level) -> bool {
        // Same level
        if l1 == l2 {
            return true;
        }

        // Zero is the minimum
        if l2.is_zero() {
            return true;
        }

        // Get offsets
        let (base1, offset1) = l1.get_offset();
        let (base2, offset2) = l2.get_offset();

        // If same base, compare offsets
        if base1 == base2 {
            return offset1 >= offset2;
        }

        // Key insight: all universe params are >= 0, and Zero is the minimum.
        // So if l2 = succ^k(Zero) (a concrete level), then any l1 = succ^k(X) where
        // k1 >= k2 is >= l2, since X >= 0.
        // For example: succ(Param(u)) >= succ(Zero) because Param(u) >= Zero.
        if base2.is_zero() && offset1 >= offset2 {
            return true;
        }

        // l1 = succ^k(l1') and k > 0 and l1' >= l2 implies l1 >= l2
        if offset1 > 0 {
            let l1_inner = l1.as_inner();
            if Level::is_geq(l1_inner, l2) {
                return true;
            }
        }

        // max(a, b) >= l if a >= l or b >= l
        if let Level::Max(a, b) = l1 {
            if Level::is_geq(a, l2) || Level::is_geq(b, l2) {
                return true;
            }
        }

        // l >= max(a, b) if l >= a and l >= b
        if let Level::Max(a, b) = l2 {
            if Level::is_geq(l1, a) && Level::is_geq(l1, b) {
                return true;
            }
        }

        // imax(a, b) reduces to max(a, b) if b is nonzero
        if let Level::IMax(a, b) = l1 {
            if b.is_nonzero() {
                return Level::is_geq(&Level::max(a.as_ref().clone(), b.as_ref().clone()), l2);
            }
        }
        if let Level::IMax(a, b) = l2 {
            if b.is_nonzero() {
                return Level::is_geq(l1, &Level::max(a.as_ref().clone(), b.as_ref().clone()));
            }
        }

        false
    }

    /// Check if l1 ≤ l2
    pub fn leq(l1: &Level, l2: &Level) -> bool {
        Level::is_geq(l2, l1)
    }

    /// Check if two levels are definitionally equal
    pub fn is_def_eq(l1: &Level, l2: &Level) -> bool {
        l1.normalize() == l2.normalize()
    }

    /// Get the inner level if this is a Succ
    fn as_inner(&self) -> &Level {
        match self {
            Level::Succ(inner) => inner.as_ref(),
            _ => self,
        }
    }

    /// Substitute universe parameters
    #[must_use]
    pub fn substitute(&self, subst: &[(Name, Level)]) -> Level {
        match self {
            Level::Zero => Level::Zero,
            Level::Succ(l) => Level::succ(l.substitute(subst)),
            Level::Max(l1, l2) => Level::max(l1.substitute(subst), l2.substitute(subst)),
            Level::IMax(l1, l2) => Level::imax(l1.substitute(subst), l2.substitute(subst)),
            Level::Param(name) => {
                for (n, level) in subst {
                    if n == name {
                        return level.clone();
                    }
                }
                self.clone()
            }
        }
    }

    /// Check if this level contains any parameters
    pub fn has_params(&self) -> bool {
        match self {
            Level::Zero => false,
            Level::Succ(l) => l.has_params(),
            Level::Max(l1, l2) | Level::IMax(l1, l2) => l1.has_params() || l2.has_params(),
            Level::Param(_) => true,
        }
    }

    /// Collect all parameter names in this level
    pub fn collect_params(&self, params: &mut Vec<Name>) {
        match self {
            Level::Zero => {}
            Level::Succ(l) => l.collect_params(params),
            Level::Max(l1, l2) | Level::IMax(l1, l2) => {
                l1.collect_params(params);
                l2.collect_params(params);
            }
            Level::Param(name) => {
                if !params.contains(name) {
                    params.push(name.clone());
                }
            }
        }
    }
}

impl std::fmt::Display for Level {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Level::Zero => write!(f, "0"),
            Level::Succ(l) => {
                // Count successive Succs for prettier output
                let mut count = 1u64;
                let mut inner = l.as_ref();
                while let Level::Succ(next) = inner {
                    count += 1;
                    inner = next.as_ref();
                }
                if inner.is_zero() {
                    write!(f, "{count}")
                } else {
                    write!(f, "{inner} + {count}")
                }
            }
            Level::Max(l1, l2) => write!(f, "max({l1}, {l2})"),
            Level::IMax(l1, l2) => write!(f, "imax({l1}, {l2})"),
            Level::Param(name) => write!(f, "{name}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_zero() {
        assert!(Level::zero().is_zero());
        assert!(!Level::succ(Level::zero()).is_zero());
        assert!(Level::max(Level::zero(), Level::zero()).is_zero());
        assert!(Level::imax(Level::succ(Level::zero()), Level::zero()).is_zero());
        // imax(u, 0) = 0, so this simplifies directly
        let level = Level::imax(Level::param(Name::from_string("u")), Level::zero());
        assert!(level.is_zero());
    }

    #[test]
    fn test_is_nonzero() {
        assert!(!Level::zero().is_nonzero());
        assert!(Level::succ(Level::zero()).is_nonzero());
        // max(0, succ(0)) = succ(0), which is nonzero
        let m = Level::max(Level::zero(), Level::succ(Level::zero()));
        assert!(m.is_nonzero());
    }

    #[test]
    fn test_max_simplification() {
        // max(l, l) = l
        let u = Level::param(Name::from_string("u"));
        let m = Level::max(u.clone(), u.clone());
        assert_eq!(m, u);

        // max(0, l) = l
        let m = Level::max(Level::zero(), u.clone());
        assert_eq!(m, u);

        // max(l, 0) = l
        let m = Level::max(u.clone(), Level::zero());
        assert_eq!(m, u);
    }

    #[test]
    fn test_imax_simplification() {
        let u = Level::param(Name::from_string("u"));

        // imax(u, 0) = 0
        let i = Level::imax(u.clone(), Level::zero());
        assert!(i.is_zero());

        // imax(u, succ(0)) = max(u, succ(0)) since succ(0) > 0
        let one = Level::succ(Level::zero());
        let i = Level::imax(u.clone(), one.clone());
        // Should be max(u, 1), not imax
        match i {
            Level::Max(_, _) => {} // Good - reduced to max
            Level::IMax(_, _) => panic!("Should have reduced to Max"),
            other => {
                // Might simplify further depending on implementation
                assert!(!matches!(other, Level::IMax(_, _)));
            }
        }
    }

    #[test]
    fn test_is_geq() {
        // l >= 0 for all l
        assert!(Level::is_geq(
            &Level::param(Name::from_string("u")),
            &Level::zero()
        ));
        assert!(Level::is_geq(&Level::zero(), &Level::zero()));

        // succ(l) >= l
        let u = Level::param(Name::from_string("u"));
        assert!(Level::is_geq(&Level::succ(u.clone()), &u));

        // succ(succ(0)) >= succ(0)
        let one = Level::succ(Level::zero());
        let two = Level::succ(one.clone());
        assert!(Level::is_geq(&two, &one));
        assert!(Level::is_geq(&two, &Level::zero()));
    }

    #[test]
    fn test_normalize() {
        // imax(u, 0) normalizes to 0
        let u = Level::param(Name::from_string("u"));
        let i = Level::IMax(Arc::new(u.clone()), Arc::new(Level::zero()));
        // After simplification in imax(), this is already Zero
        // But if we construct it manually:
        let normalized = i.normalize();
        assert!(normalized.is_zero());

        // max(0, u) normalizes to u
        let m = Level::Max(Arc::new(Level::zero()), Arc::new(u.clone()));
        let normalized = m.normalize();
        assert_eq!(normalized, u);
    }

    #[test]
    fn test_substitute() {
        let u = Level::param(Name::from_string("u"));
        let v = Level::param(Name::from_string("v"));
        let two = Level::succ(Level::succ(Level::zero()));

        // Substitute u -> 2
        let subst = vec![(Name::from_string("u"), two.clone())];
        let result = u.substitute(&subst);
        assert_eq!(result, two);

        // v should be unchanged
        let result = v.substitute(&subst);
        assert_eq!(result, v);

        // max(u, v) with u -> 2 should give max(2, v)
        let max_uv = Level::max(u.clone(), v.clone());
        let result = max_uv.substitute(&subst);
        // Should be max(2, v) - check structure
        assert!(result.has_params()); // Still has v
    }

    #[test]
    fn test_collect_params() {
        let u = Level::param(Name::from_string("u"));
        let v = Level::param(Name::from_string("v"));
        let level = Level::max(u, Level::imax(v, Level::succ(Level::zero())));

        let mut params = Vec::new();
        level.collect_params(&mut params);

        assert!(params.contains(&Name::from_string("u")));
        assert!(params.contains(&Name::from_string("v")));
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_get_offset() {
        let u = Level::param(Name::from_string("u"));
        let (base, offset) = u.get_offset();
        assert_eq!(offset, 0);
        assert_eq!(base, &u);

        let succ_u = Level::succ(u.clone());
        let (base, offset) = succ_u.get_offset();
        assert_eq!(offset, 1);
        assert_eq!(base, &u);

        let succ_succ_u = Level::succ(succ_u);
        let (base, offset) = succ_succ_u.get_offset();
        assert_eq!(offset, 2);
        assert_eq!(base, &u);
    }

    #[test]
    fn test_add_offset() {
        let u = Level::param(Name::from_string("u"));
        let result = u.add_offset(3);
        let (base, offset) = result.get_offset();
        assert_eq!(offset, 3);
        assert_eq!(base, &u);
    }

    // =========================================================================
    // Mutation Testing Kill Tests
    // =========================================================================

    #[test]
    fn test_is_zero_logic() {
        // Kill mutant: is_zero max case && to ||

        // max(0, 0) IS zero
        let max_00 = Level::Max(Arc::new(Level::Zero), Arc::new(Level::Zero));
        assert!(max_00.is_zero());

        // max(1, 0) is NOT zero (one side is nonzero)
        let max_10 = Level::Max(Arc::new(Level::succ(Level::zero())), Arc::new(Level::Zero));
        assert!(!max_10.is_zero());

        // max(0, 1) is NOT zero (one side is nonzero)
        let max_01 = Level::Max(Arc::new(Level::Zero), Arc::new(Level::succ(Level::zero())));
        assert!(!max_01.is_zero());

        // max(1, 1) is NOT zero
        let one = Level::succ(Level::zero());
        let max_11 = Level::Max(Arc::new(one.clone()), Arc::new(one));
        assert!(!max_11.is_zero());
    }

    #[test]
    fn test_is_geq_comparison_operators() {
        // Kill mutants: > vs < vs >= comparisons in is_geq

        // Test offset comparison: l1 >= l2 when offsets differ
        let u = Level::param(Name::from_string("u"));
        let u1 = Level::succ(u.clone()); // u + 1
        let u2 = Level::succ(u1.clone()); // u + 2

        // u + 2 >= u + 1 (offset 2 >= offset 1)
        assert!(Level::is_geq(&u2, &u1));

        // u + 1 NOT >= u + 2 (offset 1 < offset 2)
        assert!(!Level::is_geq(&u1, &u2));

        // u + 2 >= u (offset 2 >= offset 0)
        assert!(Level::is_geq(&u2, &u));

        // u NOT >= u + 1 (offset 0 < offset 1)
        assert!(!Level::is_geq(&u, &u1));

        // Test > vs >= : offset1 > 0 check
        // succ(u) where succ(u) >= v should check if u >= v
        let v = Level::param(Name::from_string("v"));
        let succ_v = Level::succ(v.clone());

        // succ(v) >= v  (offset > 0, then v >= v is true)
        assert!(Level::is_geq(&succ_v, &v));
    }

    #[test]
    fn test_is_geq_max_logic() {
        // Kill mutant: is_geq max cases && to ||

        let u = Level::param(Name::from_string("u"));
        let _v = Level::param(Name::from_string("v")); // unused but kept for clarity
        let one = Level::succ(Level::zero());
        let two = Level::succ(one.clone());

        // max(a, b) >= l if a >= l OR b >= l (||)
        // max(u, 2) >= 1 should be true because 2 >= 1
        let max_u2 = Level::max(u.clone(), two.clone());
        assert!(Level::is_geq(&max_u2, &one));

        // l >= max(a, b) if l >= a AND l >= b (&&)
        // 2 >= max(0, 1) should be true because 2 >= 0 AND 2 >= 1
        let max_01 = Level::max(Level::zero(), one.clone());
        assert!(Level::is_geq(&two, &max_01));

        // 1 >= max(0, 2) should be FALSE because 1 >= 0 but NOT 1 >= 2
        let max_02 = Level::max(Level::zero(), two.clone());
        assert!(!Level::is_geq(&one, &max_02));

        // 0 >= max(1, 0) should be FALSE because NOT 0 >= 1
        let max_10 = Level::max(one.clone(), Level::zero());
        assert!(!Level::is_geq(&Level::zero(), &max_10));
    }

    #[test]
    fn test_leq_uses_is_geq() {
        // Kill mutant: leq can return true/false always

        let one = Level::succ(Level::zero());
        let two = Level::succ(one.clone());

        // 1 <= 2
        assert!(Level::leq(&one, &two));

        // NOT 2 <= 1
        assert!(!Level::leq(&two, &one));

        // 0 <= anything
        assert!(Level::leq(&Level::zero(), &one));
        assert!(Level::leq(&Level::zero(), &Level::zero()));

        // NOT 1 <= 0
        assert!(!Level::leq(&one, &Level::zero()));
    }

    #[test]
    fn test_has_params_predicate() {
        // Kill mutant: has_params can return true always

        // Zero has no params
        assert!(!Level::zero().has_params());

        // succ(0) has no params
        assert!(!Level::succ(Level::zero()).has_params());

        // Param has params
        let u = Level::param(Name::from_string("u"));
        assert!(u.has_params());

        // succ(u) has params
        assert!(Level::succ(u.clone()).has_params());

        // max(0, u) has params
        assert!(Level::max(Level::zero(), u.clone()).has_params());

        // max(0, 0) has no params
        assert!(!Level::max(Level::zero(), Level::zero()).has_params());

        // max(1, 2) has no params
        let one = Level::succ(Level::zero());
        let two = Level::succ(one.clone());
        assert!(!Level::max(one, two).has_params());
    }

    #[test]
    fn test_display_count_increment() {
        // Kill mutant: Display += with *= in count increment

        // Test display output for various succ levels
        let zero = Level::zero();
        let one = Level::succ(zero.clone());
        let two = Level::succ(one.clone());
        let three = Level::succ(two.clone());

        assert_eq!(format!("{zero}"), "0");
        assert_eq!(format!("{one}"), "1");
        assert_eq!(format!("{two}"), "2");
        assert_eq!(format!("{three}"), "3");

        // Test with parameter base
        let u = Level::param(Name::from_string("u"));
        let u1 = Level::succ(u.clone());
        let u2 = Level::succ(u1.clone());

        assert_eq!(format!("{u}"), "u");
        assert_eq!(format!("{u1}"), "u + 1");
        assert_eq!(format!("{u2}"), "u + 2");
    }

    // =========================================================================
    // Additional Mutation Kill Tests - is_geq specific
    // =========================================================================

    #[test]
    fn test_is_geq_offset_positive_check() {
        // Kill mutant at line 183: replace > with < in `offset1 > 0`
        // The check `offset1 > 0` is used to recursively check if l1' >= l2
        // where l1 = succ^k(l1') and k > 0
        //
        // With `<`: offset1 < 0 is NEVER true for u32, so the check never fires
        // This affects cases where succ^k(l1') >= l2 because l1' >= l2

        // Case: succ(u) >= u
        // offset1 = 1 > 0, so check if u >= u (true)
        // With < mutant: 1 < 0 is false, skip the check, but same base so still true
        // We need a case where bases differ
        let u = Level::param(Name::from_string("u"));
        let v = Level::param(Name::from_string("v"));

        // succ(u) >= u: bases same (u), offset 1 >= 0, true
        assert!(Level::is_geq(&Level::succ(u.clone()), &u));

        // succ(max(u, v)) >= max(u, v)
        // l1 = succ(max(u, v)), l2 = max(u, v)
        // bases differ (Succ vs Max), but offset1 = 1 > 0
        // So check if max(u, v) >= max(u, v) (true)
        // Result: true
        let max_uv = Level::max(u.clone(), v.clone());
        let succ_max = Level::succ(max_uv.clone());
        assert!(
            Level::is_geq(&succ_max, &max_uv),
            "succ(max(u,v)) >= max(u,v) should be true via offset > 0 check"
        );

        // Now test a case where the offset > 0 check is essential:
        // succ(u) >= v where u and v are different params
        // bases differ (u vs v), offset1 = 1 > 0
        // Check if u >= v (false - can't compare different params)
        // Without offset check, we'd return false immediately
        // With offset check, we also get false (u >= v is false)
        // So this doesn't distinguish the mutation...

        // Better test: succ^2(0) >= succ(0) = 2 >= 1
        // l1 = 2, l2 = 1
        // get_offset(2) = (0, 2), get_offset(1) = (0, 1)
        // bases same (Zero), offset 2 >= 1, true
        let one = Level::succ(Level::zero());
        let two = Level::succ(one.clone());
        assert!(Level::is_geq(&two, &one));

        // Test: succ(u + 1) >= u
        // = (u + 2) >= u
        // bases same, offset 2 >= 0, true
        let u_plus_1 = Level::succ(u.clone());
        let u_plus_2 = Level::succ(u_plus_1.clone());
        assert!(Level::is_geq(&u_plus_2, &u));

        // Key test: succ(max(u, 0)) >= u
        // l1 = succ(max(u, 0))
        // l2 = u
        // get_offset(l1) = (max(u, 0), 1)
        // get_offset(l2) = (u, 0)
        // bases differ (max vs param), so we can't just compare offsets
        // offset1 = 1 > 0, so check if max(u, 0) >= u
        // max(u, 0) >= u: max case, u >= u (true) or 0 >= u (false)
        // So true via the first arm
        // Result: true
        // With < mutant: 1 < 0 is false, skip offset check
        // Then max check for l1: l1 is Succ not Max, skip
        // Then max check for l2: l2 is Param not Max, skip
        // Return false (wrong!)
        let max_u0 = Level::max(u.clone(), Level::zero());
        let succ_max_u0 = Level::succ(max_u0);
        assert!(
            Level::is_geq(&succ_max_u0, &u),
            "succ(max(u, 0)) >= u should be true via offset > 0 recursive check"
        );

        // CRITICAL TEST: succ(max(u, v)) >= u
        // This is the key case that distinguishes the > vs < mutation
        // l1 = succ(max(u, v)), l2 = u
        // get_offset(l1) = (max(u, v), 1)
        // get_offset(l2) = (u, 0)
        // bases differ: max(u, v) != u structurally
        // With > 0: offset1=1 > 0, check is_geq(max(u, v), u)
        //   max check fires: u >= u || v >= u = true || false = true
        // With < 0: offset1=1 < 0 is false, skip
        //   max check for l1: l1 is Succ not Max, skip
        //   max check for l2: l2 is Param not Max, skip
        //   Return false (WRONG!)
        let succ_max_uv = Level::succ(max_uv.clone());
        assert!(
            Level::is_geq(&succ_max_uv, &u),
            "succ(max(u, v)) >= u should be true: bases differ but inner max(u,v) >= u"
        );
    }

    #[test]
    fn test_is_geq_max_requires_both_and() {
        // Kill mutant at line 199: replace && with || in
        // `Level::is_geq(l1, a) && Level::is_geq(l1, b)`
        //
        // This checks if l >= max(a, b) which requires l >= a AND l >= b
        // With ||: it would only require l >= a OR l >= b

        let one = Level::succ(Level::zero());
        let two = Level::succ(one.clone());
        let three = Level::succ(two.clone());

        // 2 >= max(1, 3)?
        // With &&: 2 >= 1 (true) AND 2 >= 3 (false) = false
        // With ||: 2 >= 1 (true) OR 2 >= 3 (false) = true
        let max_1_3 = Level::max(one.clone(), three.clone());
        assert!(
            !Level::is_geq(&two, &max_1_3),
            "2 >= max(1, 3) should be FALSE because 2 >= 3 is false"
        );

        // 3 >= max(1, 2)?
        // With &&: 3 >= 1 (true) AND 3 >= 2 (true) = true
        let max_1_2 = Level::max(one.clone(), two.clone());
        assert!(
            Level::is_geq(&three, &max_1_2),
            "3 >= max(1, 2) should be TRUE"
        );

        // 1 >= max(0, 2)?
        // With &&: 1 >= 0 (true) AND 1 >= 2 (false) = false
        // With ||: 1 >= 0 (true) OR 1 >= 2 (false) = true
        let max_0_2 = Level::max(Level::zero(), two.clone());
        assert!(
            !Level::is_geq(&one, &max_0_2),
            "1 >= max(0, 2) should be FALSE"
        );

        // 0 >= max(0, 1)?
        // With &&: 0 >= 0 (true) AND 0 >= 1 (false) = false
        let max_0_1 = Level::max(Level::zero(), one.clone());
        assert!(
            !Level::is_geq(&Level::zero(), &max_0_1),
            "0 >= max(0, 1) should be FALSE"
        );

        // Test with params: u >= max(u, v) where u and v are different
        let u = Level::param(Name::from_string("u"));
        let v = Level::param(Name::from_string("v"));
        let max_uv = Level::max(u.clone(), v.clone());

        // u >= max(u, v)?
        // With &&: u >= u (true) AND u >= v (false, can't compare) = false
        // Wait, is_geq for incomparable params... let's check
        // Actually, Level::is_geq returns false for incomparable params
        // So: u >= u (true) AND u >= v (false) = false
        // With ||: u >= u (true) OR u >= v (false) = true
        assert!(
            !Level::is_geq(&u, &max_uv),
            "u >= max(u, v) should be FALSE when u and v are independent params"
        );
    }

    // =========================================================================
    // lean4lean Theorem Coverage Tests - Phase V5 Completion
    // These tests verify the remaining 4 lean4lean theorems about universe levels
    // Reference: https://github.com/digama0/lean4lean Theory/VLevel.lean
    // =========================================================================

    #[test]
    fn test_equiv_congr_left() {
        // lean4lean theorem equiv_congr_left:
        //   {a b c : VLevel} (h : a ≈ b) : a ≈ c ↔ b ≈ c
        //
        // In Lean5: Level::is_def_eq uses normalization for equivalence.
        // If a ≈ b (a.normalize() == b.normalize()), then:
        //   a ≈ c ↔ b ≈ c
        // Because both reduce to comparing the same normal form with c.

        let u = Level::param(Name::from_string("u"));
        let v = Level::param(Name::from_string("v"));

        // Test case 1: a = max(u, 0), b = u
        // max(u, 0) normalizes to u, so a ≈ b
        let a = Level::max(u.clone(), Level::zero());
        let b = u.clone();
        let c = Level::succ(u.clone()); // u + 1

        // Verify a ≈ b
        assert!(Level::is_def_eq(&a, &b), "max(u, 0) ≈ u should hold");

        // Now: a ≈ c ↔ b ≈ c
        // a ≈ c: max(u, 0) ≈ u+1? No (u ≠ u+1)
        // b ≈ c: u ≈ u+1? No
        let a_eq_c = Level::is_def_eq(&a, &c);
        let b_eq_c = Level::is_def_eq(&b, &c);
        assert_eq!(a_eq_c, b_eq_c, "equiv_congr_left: a ≈ c ↔ b ≈ c when a ≈ b");

        // Test case 2: a = imax(u, succ(v)), b = max(u, succ(v))
        // imax(u, succ(v)) = max(u, succ(v)) because succ(v) is nonzero
        let a2 = Level::imax(u.clone(), Level::succ(v.clone()));
        let b2 = Level::max(u.clone(), Level::succ(v.clone()));
        let c2 = v.clone();

        // Verify a2 ≈ b2
        assert!(
            Level::is_def_eq(&a2, &b2),
            "imax(u, succ(v)) ≈ max(u, succ(v))"
        );

        // equiv_congr_left: a2 ≈ c2 ↔ b2 ≈ c2
        let a2_eq_c2 = Level::is_def_eq(&a2, &c2);
        let b2_eq_c2 = Level::is_def_eq(&b2, &c2);
        assert_eq!(a2_eq_c2, b2_eq_c2, "equiv_congr_left holds for imax/max");

        // Test case 3: Positive case where all are equal
        let a3 = Level::max(Level::zero(), Level::zero());
        let b3 = Level::zero();
        let c3 = Level::imax(Level::succ(u.clone()), Level::zero()); // = 0

        assert!(Level::is_def_eq(&a3, &b3), "max(0, 0) ≈ 0");
        assert!(Level::is_def_eq(&c3, &Level::zero()), "imax(_, 0) ≈ 0");

        let a3_eq_c3 = Level::is_def_eq(&a3, &c3);
        let b3_eq_c3 = Level::is_def_eq(&b3, &c3);
        assert_eq!(a3_eq_c3, b3_eq_c3, "equiv_congr_left: both should be true");
        assert!(a3_eq_c3, "All should be equivalent to zero");
    }

    #[test]
    fn test_equiv_congr_right() {
        // lean4lean theorem equiv_congr_right:
        //   {a b c : VLevel} (h : a ≈ b) : c ≈ a ↔ c ≈ b
        //
        // By symmetry of ≈, this is equivalent to equiv_congr_left.

        let u = Level::param(Name::from_string("u"));

        // Test: a = max(0, u), b = u, c = some level
        let a = Level::max(Level::zero(), u.clone()); // = u
        let b = u.clone();
        let c = Level::succ(Level::succ(Level::zero())); // = 2

        // Verify a ≈ b
        assert!(Level::is_def_eq(&a, &b), "max(0, u) ≈ u");

        // equiv_congr_right: c ≈ a ↔ c ≈ b
        let c_eq_a = Level::is_def_eq(&c, &a);
        let c_eq_b = Level::is_def_eq(&c, &b);
        assert_eq!(
            c_eq_a, c_eq_b,
            "equiv_congr_right: c ≈ a ↔ c ≈ b when a ≈ b"
        );

        // Test case 2: When equivalences hold
        let a2 = Level::imax(Level::zero(), Level::zero()); // = 0
        let b2 = Level::zero();
        let c2 = Level::max(Level::zero(), Level::zero()); // = 0

        assert!(Level::is_def_eq(&a2, &b2), "imax(0, 0) ≈ 0");

        let c2_eq_a2 = Level::is_def_eq(&c2, &a2);
        let c2_eq_b2 = Level::is_def_eq(&c2, &b2);
        assert_eq!(c2_eq_a2, c2_eq_b2, "equiv_congr_right: both should be true");
        assert!(c2_eq_a2, "All zeros are equivalent");
    }

    #[test]
    fn test_inst_id() {
        // lean4lean theorem inst_id:
        //   {l : VLevel} (h : l.WF u) : l.inst (params u) = l
        //
        // If you substitute each parameter with itself, you get back the same level.
        // In Lean5 terms: l.substitute([(u, Param(u)), (v, Param(v)), ...]) = l
        //
        // This is the identity substitution property.

        let u = Level::param(Name::from_string("u"));
        let v = Level::param(Name::from_string("v"));
        let w = Level::param(Name::from_string("w"));

        // Identity substitution: map each param to itself
        let id_subst = vec![
            (Name::from_string("u"), u.clone()),
            (Name::from_string("v"), v.clone()),
            (Name::from_string("w"), w.clone()),
        ];

        // Test 1: Simple param
        let result = u.substitute(&id_subst);
        assert_eq!(result, u, "inst_id: u[u/u] = u");

        // Test 2: Succ of param
        let succ_u = Level::succ(u.clone());
        let result = succ_u.substitute(&id_subst);
        assert_eq!(result, succ_u, "inst_id: (u+1)[id] = u+1");

        // Test 3: Max of params
        let max_uv = Level::max(u.clone(), v.clone());
        let result = max_uv.substitute(&id_subst);
        assert_eq!(result, max_uv, "inst_id: max(u,v)[id] = max(u,v)");

        // Test 4: IMax of params
        let imax_uv = Level::imax(u.clone(), v.clone());
        let result = imax_uv.substitute(&id_subst);
        assert_eq!(result, imax_uv, "inst_id: imax(u,v)[id] = imax(u,v)");

        // Test 5: Complex nested level
        let complex = Level::max(
            Level::succ(Level::succ(u.clone())),
            Level::imax(v.clone(), Level::max(w.clone(), Level::zero())),
        );
        let result = complex.substitute(&id_subst);
        assert_eq!(result, complex, "inst_id: complex[id] = complex");

        // Test 6: Level with no params (should be unchanged)
        let concrete = Level::succ(Level::succ(Level::zero())); // 2
        let result = concrete.substitute(&id_subst);
        assert_eq!(result, concrete, "inst_id: 2[id] = 2");

        // Test 7: Empty substitution should also preserve
        let empty_subst: Vec<(Name, Level)> = vec![];
        let result = max_uv.substitute(&empty_subst);
        assert_eq!(
            result, max_uv,
            "inst_id: max(u,v)[] = max(u,v) (param not in subst)"
        );
    }

    #[test]
    fn test_inst_map_id() {
        // lean4lean theorem inst_map_id:
        //   (h : ls.length = n) : (params n).map (inst ls) = ls
        //
        // If you have a list of levels ls = [l0, l1, l2, ...]
        // and you create params = [Param(0), Param(1), Param(2), ...]
        // then substituting params with ls gives back ls.
        //
        // In Lean5: for a list of levels ls and corresponding param names,
        // if subst = [(p0, l0), (p1, l1), ...], then
        // [Param(p0), Param(p1), ...].map(|p| p.substitute(subst)) = ls

        let l0 = Level::succ(Level::zero()); // 1
        let l1 = Level::succ(Level::succ(Level::zero())); // 2
        let l2 = Level::param(Name::from_string("x")); // x

        let ls = vec![l0.clone(), l1.clone(), l2.clone()];

        // Create param names and corresponding params
        let p0 = Name::from_string("p0");
        let p1 = Name::from_string("p1");
        let p2 = Name::from_string("p2");

        let params = [
            Level::param(p0.clone()),
            Level::param(p1.clone()),
            Level::param(p2.clone()),
        ];

        // Create substitution: p0 -> l0, p1 -> l1, p2 -> l2
        let subst = vec![
            (p0.clone(), l0.clone()),
            (p1.clone(), l1.clone()),
            (p2.clone(), l2.clone()),
        ];

        // Map substitute over params
        let result: Vec<Level> = params.iter().map(|p| p.substitute(&subst)).collect();

        // Should get back ls
        assert_eq!(result, ls, "inst_map_id: params.map(inst ls) = ls");

        // Test with single element
        let ls_single = [Level::succ(Level::succ(Level::succ(Level::zero())))]; // [3]
        let p_single = Name::from_string("p_single");
        let params_single = [Level::param(p_single.clone())];
        let subst_single = vec![(p_single, ls_single[0].clone())];

        let result_single: Vec<Level> = params_single
            .iter()
            .map(|p| p.substitute(&subst_single))
            .collect();
        assert_eq!(
            result_single.as_slice(),
            &ls_single,
            "inst_map_id: single element case"
        );

        // Test with empty list
        let ls_empty: Vec<Level> = vec![];
        let params_empty: Vec<Level> = vec![];
        let subst_empty: Vec<(Name, Level)> = vec![];

        let result_empty: Vec<Level> = params_empty
            .iter()
            .map(|p| p.substitute(&subst_empty))
            .collect();
        assert_eq!(result_empty, ls_empty, "inst_map_id: empty list case");
    }

    /// Test that universe level comparison correctly handles param >= concrete level cases.
    /// All universe params are >= 0, so succ(Param(u)) >= succ(Zero) because Param(u) >= Zero.
    #[test]
    fn test_is_geq_param_vs_concrete() {
        let u = Level::param(Name::from_string("u"));
        let one = Level::succ(Level::zero());
        let succ_u = Level::succ(u.clone());

        // succ(u) >= succ(0) because u >= 0 for all universe params
        assert!(
            Level::is_geq(&succ_u, &one),
            "succ(u) >= succ(0) should be true since u >= 0"
        );

        // Therefore max(succ(0), succ(u)) should simplify to succ(u)
        let max_level = Level::max(one.clone(), succ_u.clone());
        assert_eq!(
            max_level, succ_u,
            "max(1, u+1) should simplify to u+1 since u+1 >= 1"
        );

        // Test max(1, max(1, u+1)) = u+1
        let nested = Level::max(one.clone(), Level::max(one.clone(), succ_u.clone()));
        assert_eq!(
            nested, succ_u,
            "nested max with concrete levels should simplify"
        );

        // Test succ^2(u) >= succ^2(0)
        let two = Level::succ(one.clone());
        let succ_succ_u = Level::succ(succ_u.clone());
        assert!(
            Level::is_geq(&succ_succ_u, &two),
            "succ(succ(u)) >= succ(succ(0)) should be true"
        );
    }
}
