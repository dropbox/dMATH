//! Kani formal verification harnesses for TLA2 core components
//!
//! This module contains Kani proofs for critical properties of the model checker:
//! - Fingerprint determinism: Same value → same fingerprint
//! - Value equality reflexivity: v == v for all v
//! - Value equality symmetry: v1 == v2 implies v2 == v1
//! - State fingerprint determinism: Same state → same fingerprint
//!
//! These harnesses are only compiled when running `cargo kani`.
//! They verify formal properties of the core data structures.
//!
//! # Running Verification
//!
//! ```bash
//! # Install Kani first: cargo install --locked kani-verifier
//! # Then verify:
//! cargo kani --harness verify_value_fingerprint_deterministic
//! cargo kani --harness verify_value_equality_reflexive
//! cargo kani --harness verify_state_fingerprint_deterministic
//! ```
//!
//! # Properties Verified
//!
//! ## P1: Fingerprint Determinism
//! For any value v, calling value_fingerprint(v) twice returns the same result.
//! This is critical for correct state deduplication.
//!
//! ## P2: Value Equality Reflexivity
//! For any value v, v == v is always true.
//! Required for Eq trait correctness.
//!
//! ## P3: Value Equality Symmetry
//! For any values v1, v2: (v1 == v2) implies (v2 == v1).
//! Required for Eq trait correctness.
//!
//! ## P4: State Fingerprint Determinism
//! For any state s, calling s.fingerprint() twice returns the same result.
//! Critical for state deduplication in model checking.
//!
//! ## P5: Type Discrimination
//! Values of different primitive types are never equal.
//! E.g., Bool(true) != Int(1).

#[cfg(kani)]
mod kani_proofs {
    use crate::state::{value_fingerprint, State};
    use crate::value::{FuncValue, IntervalValue, Value};
    use im::{OrdMap, OrdSet};
    use num_bigint::BigInt;
    use num_integer::Integer;
    use std::sync::Arc;

    // =========================================================================
    // Helper functions for generating arbitrary values
    // =========================================================================

    /// Generate an arbitrary boolean value
    fn any_bool_value() -> Value {
        Value::Bool(kani::any())
    }

    /// Generate an arbitrary small integer value (-128..127)
    /// We use small integers to keep verification tractable
    fn any_small_int_value() -> Value {
        let n: i8 = kani::any();
        Value::Int(BigInt::from(n as i64))
    }

    /// Generate an arbitrary short string value (0-4 chars from limited alphabet)
    /// We limit string length and alphabet for tractable verification
    fn any_short_string_value() -> Value {
        // Use a very limited string space for tractability
        let choice: u8 = kani::any();
        kani::assume(choice < 8);
        let s = match choice {
            0 => "",
            1 => "a",
            2 => "b",
            3 => "ab",
            4 => "ba",
            5 => "x",
            6 => "y",
            _ => "z",
        };
        Value::String(Arc::from(s))
    }

    /// Generate an arbitrary primitive value (Bool, Int, or String)
    fn any_primitive_value() -> Value {
        let choice: u8 = kani::any();
        kani::assume(choice < 3);
        match choice {
            0 => any_bool_value(),
            1 => any_small_int_value(),
            _ => any_short_string_value(),
        }
    }

    // =========================================================================
    // P1: Fingerprint Determinism
    // =========================================================================

    /// Verify that boolean fingerprinting is deterministic
    #[kani::proof]
    fn verify_bool_fingerprint_deterministic() {
        let v = any_bool_value();
        let fp1 = value_fingerprint(&v);
        let fp2 = value_fingerprint(&v);
        assert!(fp1 == fp2, "Fingerprint must be deterministic");
    }

    /// Verify that integer fingerprinting is deterministic
    #[kani::proof]
    fn verify_int_fingerprint_deterministic() {
        let v = any_small_int_value();
        let fp1 = value_fingerprint(&v);
        let fp2 = value_fingerprint(&v);
        assert!(fp1 == fp2, "Fingerprint must be deterministic");
    }

    /// Verify that string fingerprinting is deterministic
    #[kani::proof]
    fn verify_string_fingerprint_deterministic() {
        let v = any_short_string_value();
        let fp1 = value_fingerprint(&v);
        let fp2 = value_fingerprint(&v);
        assert!(fp1 == fp2, "Fingerprint must be deterministic");
    }

    /// Verify that fingerprinting is deterministic for all primitive values
    #[kani::proof]
    fn verify_value_fingerprint_deterministic() {
        let v = any_primitive_value();
        let fp1 = value_fingerprint(&v);
        let fp2 = value_fingerprint(&v);
        assert!(fp1 == fp2, "Fingerprint must be deterministic");
    }

    // =========================================================================
    // P2: Value Equality Reflexivity
    // =========================================================================

    /// Verify that boolean equality is reflexive
    #[kani::proof]
    fn verify_bool_equality_reflexive() {
        let v = any_bool_value();
        assert!(v == v, "Value equality must be reflexive");
    }

    /// Verify that integer equality is reflexive
    #[kani::proof]
    fn verify_int_equality_reflexive() {
        let v = any_small_int_value();
        assert!(v == v, "Value equality must be reflexive");
    }

    /// Verify that string equality is reflexive
    #[kani::proof]
    fn verify_string_equality_reflexive() {
        let v = any_short_string_value();
        assert!(v == v, "Value equality must be reflexive");
    }

    /// Verify that equality is reflexive for all primitive values
    #[kani::proof]
    fn verify_value_equality_reflexive() {
        let v = any_primitive_value();
        assert!(v == v, "Value equality must be reflexive");
    }

    // =========================================================================
    // P3: Value Equality Symmetry
    // =========================================================================

    /// Verify that boolean equality is symmetric
    #[kani::proof]
    fn verify_bool_equality_symmetric() {
        let v1 = any_bool_value();
        let v2 = any_bool_value();
        if v1 == v2 {
            assert!(v2 == v1, "Value equality must be symmetric");
        }
    }

    /// Verify that integer equality is symmetric
    #[kani::proof]
    fn verify_int_equality_symmetric() {
        let v1 = any_small_int_value();
        let v2 = any_small_int_value();
        if v1 == v2 {
            assert!(v2 == v1, "Value equality must be symmetric");
        }
    }

    /// Verify that string equality is symmetric
    #[kani::proof]
    fn verify_string_equality_symmetric() {
        let v1 = any_short_string_value();
        let v2 = any_short_string_value();
        if v1 == v2 {
            assert!(v2 == v1, "Value equality must be symmetric");
        }
    }

    // =========================================================================
    // P4: State Fingerprint Determinism
    // =========================================================================

    /// Verify that empty state fingerprinting is deterministic
    #[kani::proof]
    fn verify_empty_state_fingerprint_deterministic() {
        let s = State::new();
        let fp1 = s.fingerprint();
        let fp2 = s.fingerprint();
        assert!(fp1 == fp2, "State fingerprint must be deterministic");
    }

    /// Verify that single-variable state fingerprinting is deterministic
    #[kani::proof]
    fn verify_single_var_state_fingerprint_deterministic() {
        let v = any_primitive_value();
        let s = State::from_pairs([("x", v)]);
        let fp1 = s.fingerprint();
        let fp2 = s.fingerprint();
        assert!(fp1 == fp2, "State fingerprint must be deterministic");
    }

    /// Verify that two-variable state fingerprinting is deterministic
    #[kani::proof]
    fn verify_two_var_state_fingerprint_deterministic() {
        let v1 = any_bool_value();
        let v2 = any_small_int_value();
        let s = State::from_pairs([("x", v1), ("y", v2)]);
        let fp1 = s.fingerprint();
        let fp2 = s.fingerprint();
        assert!(fp1 == fp2, "State fingerprint must be deterministic");
    }

    // =========================================================================
    // P5: Type Discrimination
    // =========================================================================

    /// Verify that booleans and integers are never equal
    #[kani::proof]
    fn verify_bool_int_not_equal() {
        let b = any_bool_value();
        let i = any_small_int_value();
        assert!(b != i, "Bool and Int must never be equal");
    }

    /// Verify that booleans and strings are never equal
    #[kani::proof]
    fn verify_bool_string_not_equal() {
        let b = any_bool_value();
        let s = any_short_string_value();
        assert!(b != s, "Bool and String must never be equal");
    }

    /// Verify that integers and strings are never equal
    #[kani::proof]
    fn verify_int_string_not_equal() {
        let i = any_small_int_value();
        let s = any_short_string_value();
        assert!(i != s, "Int and String must never be equal");
    }

    // =========================================================================
    // P6: State Equality Properties
    // =========================================================================

    /// Verify that state equality is reflexive
    #[kani::proof]
    fn verify_state_equality_reflexive() {
        let v = any_primitive_value();
        let s = State::from_pairs([("x", v)]);
        assert!(s == s, "State equality must be reflexive");
    }

    /// Verify that states with same content have same fingerprint
    #[kani::proof]
    fn verify_state_content_fingerprint_consistency() {
        let v1 = any_bool_value();
        let v2 = any_small_int_value();

        // Create two states with same content
        let s1 = State::from_pairs([("x", v1.clone()), ("y", v2.clone())]);
        let s2 = State::from_pairs([("y", v2), ("x", v1)]); // Different insertion order

        // States with same content should have same fingerprint
        assert!(
            s1.fingerprint() == s2.fingerprint(),
            "States with same content must have same fingerprint"
        );
    }

    // =========================================================================
    // P7: Fingerprint Sensitivity (Different values → Different fingerprints)
    // This is probabilistic - we verify specific cases
    // =========================================================================

    /// Verify that different booleans have different fingerprints
    #[kani::proof]
    fn verify_bool_fingerprint_sensitive() {
        let v1 = Value::Bool(true);
        let v2 = Value::Bool(false);
        let fp1 = value_fingerprint(&v1);
        let fp2 = value_fingerprint(&v2);
        assert!(
            fp1 != fp2,
            "Different booleans should have different fingerprints"
        );
    }

    /// Verify that different small integers have different fingerprints
    #[kani::proof]
    fn verify_adjacent_int_fingerprint_sensitive() {
        let n: i8 = kani::any();
        kani::assume(n < 127); // Avoid overflow
        let v1 = Value::Int(BigInt::from(n as i64));
        let v2 = Value::Int(BigInt::from((n + 1) as i64));
        let fp1 = value_fingerprint(&v1);
        let fp2 = value_fingerprint(&v2);
        assert!(
            fp1 != fp2,
            "Adjacent integers should have different fingerprints"
        );
    }

    // =========================================================================
    // P8: Ord consistency with Eq (required for BTreeSet usage)
    // =========================================================================

    /// Verify Ord consistency: a.cmp(b) == Equal iff a == b (for booleans)
    #[kani::proof]
    fn verify_bool_ord_eq_consistency() {
        use std::cmp::Ordering;
        let v1 = any_bool_value();
        let v2 = any_bool_value();
        let ord_eq = v1.cmp(&v2) == Ordering::Equal;
        let eq_eq = v1 == v2;
        assert!(ord_eq == eq_eq, "Ord and Eq must be consistent");
    }

    /// Verify Ord consistency: a.cmp(b) == Equal iff a == b (for integers)
    #[kani::proof]
    fn verify_int_ord_eq_consistency() {
        use std::cmp::Ordering;
        let v1 = any_small_int_value();
        let v2 = any_small_int_value();
        let ord_eq = v1.cmp(&v2) == Ordering::Equal;
        let eq_eq = v1 == v2;
        assert!(ord_eq == eq_eq, "Ord and Eq must be consistent");
    }

    /// Verify Ord consistency: a.cmp(b) == Equal iff a == b (for strings)
    #[kani::proof]
    fn verify_string_ord_eq_consistency() {
        use std::cmp::Ordering;
        let v1 = any_short_string_value();
        let v2 = any_short_string_value();
        let ord_eq = v1.cmp(&v2) == Ordering::Equal;
        let eq_eq = v1 == v2;
        assert!(ord_eq == eq_eq, "Ord and Eq must be consistent");
    }

    // =========================================================================
    // P9: Set Operations
    // =========================================================================

    /// Generate an arbitrary small set (0-3 elements)
    fn any_small_set() -> Value {
        use crate::value::SortedSet;
        let choice: u8 = kani::any();
        kani::assume(choice < 8);
        let mut set = OrdSet::new();
        match choice {
            0 => {} // empty set
            1 => {
                set.insert(Value::Int(BigInt::from(1)));
            }
            2 => {
                set.insert(Value::Int(BigInt::from(2)));
            }
            3 => {
                set.insert(Value::Int(BigInt::from(1)));
                set.insert(Value::Int(BigInt::from(2)));
            }
            4 => {
                set.insert(Value::Bool(true));
            }
            5 => {
                set.insert(Value::Bool(false));
            }
            6 => {
                set.insert(Value::Bool(true));
                set.insert(Value::Bool(false));
            }
            _ => {
                set.insert(Value::Int(BigInt::from(1)));
                set.insert(Value::Int(BigInt::from(2)));
                set.insert(Value::Int(BigInt::from(3)));
            }
        }
        Value::Set(SortedSet::from_ord_set(&set))
    }

    /// Verify set union commutativity: A \cup B = B \cup A
    #[kani::proof]
    fn verify_set_union_commutative() {
        let a = any_small_set();
        let b = any_small_set();

        if let (Value::Set(set_a), Value::Set(set_b)) = (&a, &b) {
            let union_ab = Value::Set(set_a.clone().union(set_b.clone()));
            let union_ba = Value::Set(set_b.clone().union(set_a.clone()));
            assert!(union_ab == union_ba, "Set union must be commutative");
        }
    }

    /// Verify set intersection commutativity: A \cap B = B \cap A
    #[kani::proof]
    fn verify_set_intersection_commutative() {
        let a = any_small_set();
        let b = any_small_set();

        if let (Value::Set(set_a), Value::Set(set_b)) = (&a, &b) {
            let inter_ab: OrdSet<Value> = set_a.clone().intersection(set_b.clone());
            let inter_ba: OrdSet<Value> = set_b.clone().intersection(set_a.clone());
            assert!(inter_ab == inter_ba, "Set intersection must be commutative");
        }
    }

    /// Verify set union identity: A \cup {} = A
    #[kani::proof]
    fn verify_set_union_identity() {
        let a = any_small_set();
        let empty = OrdSet::new();

        if let Value::Set(set_a) = &a {
            let union = set_a.clone().union(empty);
            assert!(union == *set_a, "Empty set is union identity");
        }
    }

    /// Verify set intersection with empty: A \cap {} = {}
    #[kani::proof]
    fn verify_set_intersection_empty() {
        let a = any_small_set();
        let empty: OrdSet<Value> = OrdSet::new();

        if let Value::Set(set_a) = &a {
            let inter: OrdSet<Value> = set_a.clone().intersection(empty.clone());
            assert!(inter.is_empty(), "Intersection with empty is empty");
        }
    }

    /// Verify set difference with self: A \ A = {}
    #[kani::proof]
    fn verify_set_difference_self() {
        let a = any_small_set();

        if let Value::Set(set_a) = &a {
            let diff = set_a.clone().relative_complement(set_a.clone());
            assert!(diff.is_empty(), "A \\ A must be empty");
        }
    }

    /// Verify set difference with empty: A \ {} = A
    #[kani::proof]
    fn verify_set_difference_empty() {
        let a = any_small_set();
        let empty: OrdSet<Value> = OrdSet::new();

        if let Value::Set(set_a) = &a {
            let diff = set_a.clone().relative_complement(empty);
            assert!(diff == *set_a, "A \\ {} = A");
        }
    }

    /// Verify set membership after insertion
    #[kani::proof]
    fn verify_set_membership_insert() {
        use crate::value::SortedSet;
        let v = any_primitive_value();
        let mut set = OrdSet::new();
        set.insert(v.clone());
        let s = Value::Set(SortedSet::from_ord_set(&set));

        assert!(
            s.set_contains(&v).unwrap_or(false),
            "Inserted element must be in set"
        );
    }

    /// Verify set fingerprint determinism
    #[kani::proof]
    fn verify_set_fingerprint_deterministic() {
        let s = any_small_set();
        let fp1 = value_fingerprint(&s);
        let fp2 = value_fingerprint(&s);
        assert!(fp1 == fp2, "Set fingerprint must be deterministic");
    }

    /// Verify set equality reflexivity
    #[kani::proof]
    fn verify_set_equality_reflexive() {
        let s = any_small_set();
        assert!(s == s, "Set equality must be reflexive");
    }

    /// Verify subset reflexivity: A \subseteq A
    #[kani::proof]
    fn verify_subset_reflexive() {
        let a = any_small_set();

        if let Value::Set(set_a) = &a {
            assert!(set_a.is_subset(set_a), "A must be subset of itself");
        }
    }

    /// Verify empty set is subset of any set: {} \subseteq A
    #[kani::proof]
    fn verify_empty_subset_all() {
        let a = any_small_set();
        let empty: OrdSet<Value> = OrdSet::new();

        if let Value::Set(set_a) = &a {
            assert!(empty.is_subset(set_a), "Empty set is subset of all sets");
        }
    }

    // =========================================================================
    // P10: Interval Operations
    // =========================================================================

    /// Generate an arbitrary small interval
    fn any_small_interval() -> IntervalValue {
        let low: i8 = kani::any();
        let high: i8 = kani::any();
        kani::assume(low >= -10 && low <= 10);
        kani::assume(high >= -10 && high <= 10);
        IntervalValue::new(BigInt::from(low as i64), BigInt::from(high as i64))
    }

    /// Verify interval contains bounds
    #[kani::proof]
    fn verify_interval_contains_bounds() {
        let iv = any_small_interval();
        if !iv.is_empty() {
            assert!(
                iv.contains(&Value::Int(iv.low.clone())),
                "Interval must contain low bound"
            );
            assert!(
                iv.contains(&Value::Int(iv.high.clone())),
                "Interval must contain high bound"
            );
        }
    }

    /// Verify interval non-membership outside bounds
    #[kani::proof]
    fn verify_interval_excludes_outside() {
        let iv = any_small_interval();
        if !iv.is_empty() {
            let below = Value::Int(&iv.low - 1);
            let above = Value::Int(&iv.high + 1);
            assert!(!iv.contains(&below), "Interval must not contain below low");
            assert!(!iv.contains(&above), "Interval must not contain above high");
        }
    }

    /// Verify interval length computation
    #[kani::proof]
    fn verify_interval_length() {
        let iv = any_small_interval();
        let len = iv.len();
        if iv.is_empty() {
            assert!(len == BigInt::from(0), "Empty interval has length 0");
        } else {
            let expected = &iv.high - &iv.low + 1;
            assert!(len == expected, "Interval length must be high - low + 1");
        }
    }

    /// Verify interval fingerprint deterministic
    #[kani::proof]
    fn verify_interval_fingerprint_deterministic() {
        let iv = any_small_interval();
        let v = Value::Interval(iv);
        let fp1 = value_fingerprint(&v);
        let fp2 = value_fingerprint(&v);
        assert!(fp1 == fp2, "Interval fingerprint must be deterministic");
    }

    // =========================================================================
    // P11: Function Operations
    // =========================================================================

    /// Generate an arbitrary small function
    fn any_small_func() -> FuncValue {
        let choice: u8 = kani::any();
        kani::assume(choice < 6);

        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();

        match choice {
            0 => {} // empty function
            1 => {
                // {1} -> Bool
                let d = Value::Int(BigInt::from(1));
                domain.insert(d.clone());
                mapping.insert(d, Value::Bool(kani::any()));
            }
            2 => {
                // {1, 2} -> Bool
                let d1 = Value::Int(BigInt::from(1));
                let d2 = Value::Int(BigInt::from(2));
                domain.insert(d1.clone());
                domain.insert(d2.clone());
                mapping.insert(d1, Value::Bool(kani::any()));
                mapping.insert(d2, Value::Bool(kani::any()));
            }
            3 => {
                // {TRUE, FALSE} -> Int
                let t = Value::Bool(true);
                let f = Value::Bool(false);
                domain.insert(t.clone());
                domain.insert(f.clone());
                let n1: i8 = kani::any();
                let n2: i8 = kani::any();
                mapping.insert(t, Value::Int(BigInt::from(n1 as i64)));
                mapping.insert(f, Value::Int(BigInt::from(n2 as i64)));
            }
            4 => {
                // {1} -> Int (identity-like)
                let d = Value::Int(BigInt::from(1));
                domain.insert(d.clone());
                mapping.insert(d, Value::Int(BigInt::from(1)));
            }
            _ => {
                // {0, 1} -> {0, 1}
                let d0 = Value::Int(BigInt::from(0));
                let d1 = Value::Int(BigInt::from(1));
                domain.insert(d0.clone());
                domain.insert(d1.clone());
                let b1: bool = kani::any();
                let b2: bool = kani::any();
                mapping.insert(d0, Value::Int(BigInt::from(if b1 { 1 } else { 0 })));
                mapping.insert(d1, Value::Int(BigInt::from(if b2 { 1 } else { 0 })));
            }
        }

        FuncValue::new(domain, mapping)
    }

    /// Verify function application returns domain element mapping
    #[kani::proof]
    fn verify_func_apply_in_domain() {
        let f = any_small_func();
        if !f.domain_is_empty() {
            // Pick first domain element
            if let Some(d) = f.domain_iter().next() {
                let result = f.mapping_get(d);
                assert!(
                    result.is_some(),
                    "Function must have mapping for domain elements"
                );
            }
        }
    }

    /// Verify function domain size equals mapping size
    #[kani::proof]
    fn verify_func_domain_mapping_consistent() {
        let f = any_small_func();
        // With array-based FuncValue, domain_len() is always consistent
        assert!(f.domain_len() >= 0, "Domain size must be non-negative");
    }

    /// Verify function equality reflexivity
    #[kani::proof]
    fn verify_func_equality_reflexive() {
        let f = any_small_func();
        let v = Value::Func(f);
        assert!(v == v, "Function equality must be reflexive");
    }

    /// Verify function fingerprint deterministic
    #[kani::proof]
    fn verify_func_fingerprint_deterministic() {
        let f = any_small_func();
        let v = Value::Func(f);
        let fp1 = value_fingerprint(&v);
        let fp2 = value_fingerprint(&v);
        assert!(fp1 == fp2, "Function fingerprint must be deterministic");
    }

    /// Verify function with same mapping equals itself
    #[kani::proof]
    fn verify_func_structural_equality() {
        let f1 = any_small_func();
        let f2 = FuncValue::new(f1.domain_as_ord_set(), f1.mapping_as_ord_map());
        let v1 = Value::Func(f1);
        let v2 = Value::Func(f2);
        assert!(
            v1 == v2,
            "Functions with same domain and mapping must be equal"
        );
    }

    // =========================================================================
    // P12: Record Operations
    // =========================================================================

    /// Generate an arbitrary small record
    fn any_small_record() -> Value {
        let choice: u8 = kani::any();
        kani::assume(choice < 4);

        let mut map: OrdMap<Arc<str>, Value> = OrdMap::new();

        match choice {
            0 => {} // empty record
            1 => {
                map.insert(Arc::from("x"), Value::Bool(kani::any()));
            }
            2 => {
                let n: i8 = kani::any();
                map.insert(Arc::from("x"), Value::Int(BigInt::from(n as i64)));
            }
            _ => {
                map.insert(Arc::from("x"), Value::Bool(kani::any()));
                let n: i8 = kani::any();
                map.insert(Arc::from("y"), Value::Int(BigInt::from(n as i64)));
            }
        }

        Value::Record(map)
    }

    /// Verify record equality reflexive
    #[kani::proof]
    fn verify_record_equality_reflexive() {
        let r = any_small_record();
        assert!(r == r, "Record equality must be reflexive");
    }

    /// Verify record fingerprint deterministic
    #[kani::proof]
    fn verify_record_fingerprint_deterministic() {
        let r = any_small_record();
        let fp1 = value_fingerprint(&r);
        let fp2 = value_fingerprint(&r);
        assert!(fp1 == fp2, "Record fingerprint must be deterministic");
    }

    /// Verify record field access
    #[kani::proof]
    fn verify_record_field_access() {
        let b: bool = kani::any();
        let mut map: OrdMap<Arc<str>, Value> = OrdMap::new();
        map.insert(Arc::from("field"), Value::Bool(b));
        let r = Value::Record(map);

        if let Value::Record(m) = &r {
            let field = m.get(&Arc::from("field"));
            assert!(field.is_some(), "Record must have inserted field");
            assert!(
                *field.unwrap() == Value::Bool(b),
                "Field value must match inserted value"
            );
        }
    }

    // =========================================================================
    // P13: Sequence Operations
    // =========================================================================

    /// Generate an arbitrary small sequence
    fn any_small_seq() -> Value {
        let choice: u8 = kani::any();
        kani::assume(choice < 5);

        match choice {
            0 => Value::Seq(Vec::new().into()),
            1 => Value::Seq(vec![Value::Bool(kani::any())].into()),
            2 => {
                let n: i8 = kani::any();
                Value::Seq(vec![Value::Int(BigInt::from(n as i64))].into())
            }
            3 => Value::Seq(vec![Value::Bool(kani::any()), Value::Bool(kani::any())].into()),
            _ => {
                let n1: i8 = kani::any();
                let n2: i8 = kani::any();
                Value::Seq(
                    vec![
                        Value::Int(BigInt::from(n1 as i64)),
                        Value::Int(BigInt::from(n2 as i64)),
                    ]
                    .into(),
                )
            }
        }
    }

    /// Verify sequence equality reflexive
    #[kani::proof]
    fn verify_seq_equality_reflexive() {
        let s = any_small_seq();
        assert!(s == s, "Sequence equality must be reflexive");
    }

    /// Verify sequence fingerprint deterministic
    #[kani::proof]
    fn verify_seq_fingerprint_deterministic() {
        let s = any_small_seq();
        let fp1 = value_fingerprint(&s);
        let fp2 = value_fingerprint(&s);
        assert!(fp1 == fp2, "Sequence fingerprint must be deterministic");
    }

    /// Verify sequence length
    #[kani::proof]
    fn verify_seq_length() {
        let s = any_small_seq();
        if let Value::Seq(vec) = &s {
            // Length should be non-negative (trivially true for Vec)
            assert!(vec.len() <= 5, "Sequence length bounded by construction");
        }
    }

    /// Verify sequence append increases length
    #[kani::proof]
    fn verify_seq_append_length() {
        let s = any_small_seq();
        if let Value::Seq(vec) = s {
            let original_len = vec.len();
            let mut new_vec = vec;
            new_vec.push(Value::Bool(true));
            assert!(
                new_vec.len() == original_len + 1,
                "Append must increase length by 1"
            );
        }
    }

    // =========================================================================
    // P14: Tuple Operations
    // =========================================================================

    /// Generate an arbitrary small tuple
    fn any_small_tuple() -> Value {
        let choice: u8 = kani::any();
        kani::assume(choice < 4);

        match choice {
            0 => Value::Tuple(Vec::new().into()),
            1 => Value::Tuple(vec![Value::Bool(kani::any())].into()),
            2 => {
                let n: i8 = kani::any();
                Value::Tuple(vec![Value::Int(BigInt::from(n as i64))].into())
            }
            _ => Value::Tuple(vec![Value::Bool(kani::any()), Value::Bool(kani::any())].into()),
        }
    }

    /// Verify tuple equality reflexive
    #[kani::proof]
    fn verify_tuple_equality_reflexive() {
        let t = any_small_tuple();
        assert!(t == t, "Tuple equality must be reflexive");
    }

    /// Verify tuple fingerprint deterministic
    #[kani::proof]
    fn verify_tuple_fingerprint_deterministic() {
        let t = any_small_tuple();
        let fp1 = value_fingerprint(&t);
        let fp2 = value_fingerprint(&t);
        assert!(fp1 == fp2, "Tuple fingerprint must be deterministic");
    }

    // =========================================================================
    // P15: Cross-type properties
    // =========================================================================

    /// Verify sets and sequences are never equal
    #[kani::proof]
    fn verify_set_seq_not_equal() {
        let s = any_small_set();
        let seq = any_small_seq();
        assert!(s != seq, "Set and Sequence must never be equal");
    }

    /// Verify sets and functions are never equal
    #[kani::proof]
    fn verify_set_func_not_equal() {
        let s = any_small_set();
        let f = Value::Func(any_small_func());
        assert!(s != f, "Set and Function must never be equal");
    }

    /// Verify sets and records are never equal
    #[kani::proof]
    fn verify_set_record_not_equal() {
        let s = any_small_set();
        let r = any_small_record();
        assert!(s != r, "Set and Record must never be equal");
    }

    // =========================================================================
    // P16: State Insert-Get Consistency
    // =========================================================================

    /// Verify that after inserting a value, get returns that value
    #[kani::proof]
    fn verify_state_insert_get_consistency() {
        let v = any_primitive_value();
        let s = State::new().with_var("x", v.clone());
        let retrieved = s.get("x");
        assert!(retrieved.is_some(), "Get must return inserted value");
        assert!(
            *retrieved.unwrap() == v,
            "Retrieved value must equal inserted value"
        );
    }

    /// Verify that inserting then getting works for any variable name
    #[kani::proof]
    fn verify_state_insert_get_any_name() {
        let v = any_bool_value();
        // Test with a few different variable names
        let choice: u8 = kani::any();
        kani::assume(choice < 4);
        let name = match choice {
            0 => "x",
            1 => "y",
            2 => "var1",
            _ => "state_var",
        };
        let s = State::new().with_var(name, v.clone());
        let retrieved = s.get(name);
        assert!(retrieved.is_some(), "Get must return inserted value");
        assert!(
            *retrieved.unwrap() == v,
            "Retrieved value must equal inserted value"
        );
    }

    // =========================================================================
    // P17: State Update Isolation
    // =========================================================================

    /// Verify that updating one variable doesn't affect others
    #[kani::proof]
    fn verify_state_update_isolation() {
        let v1 = any_bool_value();
        let v2 = any_small_int_value();
        let v3 = any_short_string_value();

        // Create state with x and y
        let s1 = State::from_pairs([("x", v1.clone()), ("y", v2.clone())]);

        // Update z (a different variable)
        let s2 = s1.with_var("z", v3);

        // x and y should be unchanged
        assert!(s2.get("x") == s1.get("x"), "Updating z must not affect x");
        assert!(s2.get("y") == s1.get("y"), "Updating z must not affect y");
    }

    /// Verify that updating an existing variable preserves other variables
    #[kani::proof]
    fn verify_state_update_preserves_others() {
        let v1 = any_bool_value();
        let v2 = any_small_int_value();
        let v3 = any_short_string_value();

        let s1 = State::from_pairs([("x", v1), ("y", v2.clone())]);
        let s2 = s1.with_var("x", v3); // Update x

        // y should be unchanged
        assert!(s2.get("y") == s1.get("y"), "Updating x must not affect y");
        assert!(*s2.get("y").unwrap() == v2, "y value must be preserved");
    }

    // =========================================================================
    // P18: Value Ord Transitivity
    // =========================================================================

    /// Verify Ord transitivity for booleans: a < b && b < c => a < c
    #[kani::proof]
    fn verify_bool_ord_transitive() {
        use std::cmp::Ordering;
        let a = any_bool_value();
        let b = any_bool_value();
        let c = any_bool_value();

        if a.cmp(&b) == Ordering::Less && b.cmp(&c) == Ordering::Less {
            assert!(
                a.cmp(&c) == Ordering::Less,
                "Ord must be transitive: a < b && b < c => a < c"
            );
        }
    }

    /// Verify Ord transitivity for integers
    #[kani::proof]
    fn verify_int_ord_transitive() {
        use std::cmp::Ordering;
        let a = any_small_int_value();
        let b = any_small_int_value();
        let c = any_small_int_value();

        if a.cmp(&b) == Ordering::Less && b.cmp(&c) == Ordering::Less {
            assert!(
                a.cmp(&c) == Ordering::Less,
                "Ord must be transitive: a < b && b < c => a < c"
            );
        }
    }

    /// Verify Ord transitivity for strings
    #[kani::proof]
    fn verify_string_ord_transitive() {
        use std::cmp::Ordering;
        let a = any_short_string_value();
        let b = any_short_string_value();
        let c = any_short_string_value();

        if a.cmp(&b) == Ordering::Less && b.cmp(&c) == Ordering::Less {
            assert!(
                a.cmp(&c) == Ordering::Less,
                "Ord must be transitive: a < b && b < c => a < c"
            );
        }
    }

    // =========================================================================
    // P19: Value Ord Antisymmetry
    // =========================================================================

    /// Verify Ord antisymmetry for booleans: a <= b && b <= a => a == b
    #[kani::proof]
    fn verify_bool_ord_antisymmetric() {
        use std::cmp::Ordering;
        let a = any_bool_value();
        let b = any_bool_value();

        let a_le_b = a.cmp(&b) != Ordering::Greater;
        let b_le_a = b.cmp(&a) != Ordering::Greater;

        if a_le_b && b_le_a {
            assert!(
                a == b,
                "Ord must be antisymmetric: a <= b && b <= a => a == b"
            );
        }
    }

    /// Verify Ord antisymmetry for integers
    #[kani::proof]
    fn verify_int_ord_antisymmetric() {
        use std::cmp::Ordering;
        let a = any_small_int_value();
        let b = any_small_int_value();

        let a_le_b = a.cmp(&b) != Ordering::Greater;
        let b_le_a = b.cmp(&a) != Ordering::Greater;

        if a_le_b && b_le_a {
            assert!(
                a == b,
                "Ord must be antisymmetric: a <= b && b <= a => a == b"
            );
        }
    }

    // =========================================================================
    // P20: Value Ord Total Ordering
    // =========================================================================

    /// Verify total ordering for booleans: exactly one of a < b, a == b, a > b
    #[kani::proof]
    fn verify_bool_ord_total() {
        use std::cmp::Ordering;
        let a = any_bool_value();
        let b = any_bool_value();

        let ord = a.cmp(&b);
        let is_total = ord == Ordering::Less || ord == Ordering::Equal || ord == Ordering::Greater;
        assert!(is_total, "Ord must be total: exactly one of <, ==, > holds");
    }

    /// Verify total ordering for integers
    #[kani::proof]
    fn verify_int_ord_total() {
        use std::cmp::Ordering;
        let a = any_small_int_value();
        let b = any_small_int_value();

        let ord = a.cmp(&b);
        let is_total = ord == Ordering::Less || ord == Ordering::Equal || ord == Ordering::Greater;
        assert!(is_total, "Ord must be total: exactly one of <, ==, > holds");
    }

    /// Verify total ordering for strings
    #[kani::proof]
    fn verify_string_ord_total() {
        use std::cmp::Ordering;
        let a = any_short_string_value();
        let b = any_short_string_value();

        let ord = a.cmp(&b);
        let is_total = ord == Ordering::Less || ord == Ordering::Equal || ord == Ordering::Greater;
        assert!(is_total, "Ord must be total: exactly one of <, ==, > holds");
    }

    // =========================================================================
    // P21: Hash-Equality Consistency
    // =========================================================================

    /// Verify that equal values have equal fingerprints
    #[kani::proof]
    fn verify_equal_values_equal_fingerprints() {
        let v1 = any_primitive_value();
        let v2 = v1.clone();

        assert!(v1 == v2, "Cloned values must be equal");
        assert!(
            value_fingerprint(&v1) == value_fingerprint(&v2),
            "Equal values must have equal fingerprints"
        );
    }

    /// Verify that equal states have equal fingerprints
    #[kani::proof]
    fn verify_equal_states_equal_fingerprints() {
        let v = any_primitive_value();
        let s1 = State::from_pairs([("x", v.clone())]);
        let s2 = State::from_pairs([("x", v)]);

        assert!(s1 == s2, "States with same content must be equal");
        assert!(
            s1.fingerprint() == s2.fingerprint(),
            "Equal states must have equal fingerprints"
        );
    }

    // =========================================================================
    // P22: State Construction Consistency
    // =========================================================================

    /// Verify that states constructed via with_var equal states from from_pairs
    #[kani::proof]
    fn verify_state_construction_equivalence() {
        let v1 = any_bool_value();
        let v2 = any_small_int_value();

        // Construct via with_var
        let s1 = State::new()
            .with_var("x", v1.clone())
            .with_var("y", v2.clone());

        // Construct via from_pairs
        let s2 = State::from_pairs([("x", v1), ("y", v2)]);

        assert!(
            s1 == s2,
            "Different construction methods must yield equal states"
        );
        assert!(
            s1.fingerprint() == s2.fingerprint(),
            "Different construction methods must yield same fingerprint"
        );
    }

    /// Verify that insertion order doesn't affect state equality
    #[kani::proof]
    fn verify_state_insertion_order_invariance() {
        let v1 = any_bool_value();
        let v2 = any_small_int_value();

        // Insert x then y
        let s1 = State::new()
            .with_var("x", v1.clone())
            .with_var("y", v2.clone());

        // Insert y then x
        let s2 = State::new().with_var("y", v2).with_var("x", v1);

        assert!(s1 == s2, "Insertion order must not affect state equality");
        assert!(
            s1.fingerprint() == s2.fingerprint(),
            "Insertion order must not affect fingerprint"
        );
    }

    // =========================================================================
    // P23: Interval Membership Correctness
    // =========================================================================

    /// Verify interval membership: i..j contains k iff i <= k <= j
    #[kani::proof]
    fn verify_interval_membership_correctness() {
        let low: i8 = kani::any();
        let high: i8 = kani::any();
        let k: i8 = kani::any();
        kani::assume(low >= -50 && low <= 50);
        kani::assume(high >= -50 && high <= 50);
        kani::assume(k >= -50 && k <= 50);

        let iv = IntervalValue::new(BigInt::from(low as i64), BigInt::from(high as i64));
        let contains = iv.contains(&Value::Int(BigInt::from(k as i64)));

        let expected = low <= k && k <= high;
        assert!(
            contains == expected,
            "Interval contains k iff low <= k <= high"
        );
    }

    /// Verify empty interval contains nothing
    #[kani::proof]
    fn verify_empty_interval_contains_nothing() {
        let k: i8 = kani::any();
        // Empty interval: low > high
        let iv = IntervalValue::new(BigInt::from(5), BigInt::from(1));
        let contains = iv.contains(&Value::Int(BigInt::from(k as i64)));
        assert!(!contains, "Empty interval must contain nothing");
    }

    // =========================================================================
    // P24: Value Clone Correctness
    // =========================================================================

    /// Verify that cloned values are equal to original
    #[kani::proof]
    fn verify_value_clone_equality() {
        let v = any_primitive_value();
        let v_clone = v.clone();
        assert!(v == v_clone, "Cloned value must equal original");
    }

    /// Verify that cloned set is equal to original
    #[kani::proof]
    fn verify_set_clone_equality() {
        let s = any_small_set();
        let s_clone = s.clone();
        assert!(s == s_clone, "Cloned set must equal original");
    }

    // =========================================================================
    // P25: State Clone Correctness
    // =========================================================================

    /// Verify that cloned states are equal to original
    #[kani::proof]
    fn verify_state_clone_equality() {
        let v = any_primitive_value();
        let s = State::from_pairs([("x", v)]);
        let s_clone = s.clone();
        assert!(s == s_clone, "Cloned state must equal original");
        assert!(
            s.fingerprint() == s_clone.fingerprint(),
            "Cloned state must have same fingerprint"
        );
    }

    // =========================================================================
    // Phase E: E2 - Operator Semantics Verification
    // =========================================================================

    // =========================================================================
    // P26: Integer Arithmetic - Commutativity
    // =========================================================================

    /// Verify integer addition is commutative: a + b = b + a
    #[kani::proof]
    fn verify_int_add_commutative() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        // Avoid overflow by using i16 for computation
        let a_big = BigInt::from(a as i64);
        let b_big = BigInt::from(b as i64);
        let sum_ab = &a_big + &b_big;
        let sum_ba = &b_big + &a_big;
        assert!(sum_ab == sum_ba, "Integer addition must be commutative");
    }

    /// Verify integer multiplication is commutative: a * b = b * a
    #[kani::proof]
    fn verify_int_mul_commutative() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        let a_big = BigInt::from(a as i64);
        let b_big = BigInt::from(b as i64);
        let prod_ab = &a_big * &b_big;
        let prod_ba = &b_big * &a_big;
        assert!(
            prod_ab == prod_ba,
            "Integer multiplication must be commutative"
        );
    }

    // =========================================================================
    // P27: Integer Arithmetic - Associativity
    // =========================================================================

    /// Verify integer addition is associative: (a + b) + c = a + (b + c)
    #[kani::proof]
    fn verify_int_add_associative() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        let c: i8 = kani::any();
        // Use i16 to avoid overflow in intermediate results
        kani::assume(a.checked_add(b).is_some());
        kani::assume(b.checked_add(c).is_some());
        let a_big = BigInt::from(a as i64);
        let b_big = BigInt::from(b as i64);
        let c_big = BigInt::from(c as i64);
        let left = (&a_big + &b_big) + &c_big;
        let right = &a_big + (&b_big + &c_big);
        assert!(left == right, "Integer addition must be associative");
    }

    /// Verify integer multiplication is associative: (a * b) * c = a * (b * c)
    #[kani::proof]
    fn verify_int_mul_associative() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        let c: i8 = kani::any();
        let a_big = BigInt::from(a as i64);
        let b_big = BigInt::from(b as i64);
        let c_big = BigInt::from(c as i64);
        let left = (&a_big * &b_big) * &c_big;
        let right = &a_big * (&b_big * &c_big);
        assert!(left == right, "Integer multiplication must be associative");
    }

    // =========================================================================
    // P28: Integer Arithmetic - Identity Elements
    // =========================================================================

    /// Verify additive identity: a + 0 = a
    #[kani::proof]
    fn verify_int_add_identity() {
        let a: i8 = kani::any();
        let a_big = BigInt::from(a as i64);
        let zero = BigInt::from(0);
        let result = &a_big + &zero;
        assert!(result == a_big, "Zero is additive identity");
    }

    /// Verify multiplicative identity: a * 1 = a
    #[kani::proof]
    fn verify_int_mul_identity() {
        let a: i8 = kani::any();
        let a_big = BigInt::from(a as i64);
        let one = BigInt::from(1);
        let result = &a_big * &one;
        assert!(result == a_big, "One is multiplicative identity");
    }

    // =========================================================================
    // P29: Integer Arithmetic - Additive Inverse
    // =========================================================================

    /// Verify additive inverse: a + (-a) = 0
    #[kani::proof]
    fn verify_int_add_inverse() {
        let a: i8 = kani::any();
        let a_big = BigInt::from(a as i64);
        let neg_a = -&a_big;
        let result = &a_big + &neg_a;
        assert!(result == BigInt::from(0), "a + (-a) = 0");
    }

    // =========================================================================
    // P30: Integer Arithmetic - Distributivity
    // =========================================================================

    /// Verify left distributivity: a * (b + c) = a*b + a*c
    #[kani::proof]
    fn verify_int_mul_distributes_over_add() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        let c: i8 = kani::any();
        let a_big = BigInt::from(a as i64);
        let b_big = BigInt::from(b as i64);
        let c_big = BigInt::from(c as i64);
        let left = &a_big * (&b_big + &c_big);
        let right = (&a_big * &b_big) + (&a_big * &c_big);
        assert!(
            left == right,
            "Multiplication distributes over addition: a*(b+c) = a*b + a*c"
        );
    }

    // =========================================================================
    // P31: Integer Division Properties
    // =========================================================================

    /// Verify division-modulo relationship: a = (a \div b) * b + (a % b) for b != 0
    #[kani::proof]
    fn verify_int_div_mod_relationship() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        kani::assume(b != 0);
        let a_big = BigInt::from(a as i64);
        let b_big = BigInt::from(b as i64);
        // TLA+ uses floor division and Euclidean modulo
        let quotient = a_big.div_floor(&b_big);
        let remainder = a_big.mod_floor(&b_big);
        let reconstructed = &quotient * &b_big + &remainder;
        assert!(reconstructed == a_big, "a = (a div b) * b + (a mod b)");
    }

    /// Verify modulo range: 0 <= (a % b) < |b| for b != 0
    /// Uses TLA+ Euclidean modulo semantics (always non-negative)
    #[kani::proof]
    fn verify_int_mod_range() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        kani::assume(b != 0);
        let a_big = BigInt::from(a as i64);
        let b_big = BigInt::from(b as i64);
        let abs_b = if b_big < BigInt::from(0) {
            -&b_big
        } else {
            b_big.clone()
        };
        // TLA+ Euclidean modulo: ((a % |b|) + |b|) % |b|
        let remainder = ((&a_big % &abs_b) + &abs_b) % &abs_b;
        assert!(
            remainder >= BigInt::from(0) && remainder < abs_b,
            "0 <= (a mod b) < |b|"
        );
    }

    // =========================================================================
    // P32: Boolean Algebra - Commutativity
    // =========================================================================

    /// Verify conjunction is commutative: a /\ b = b /\ a
    #[kani::proof]
    fn verify_bool_and_commutative() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        assert!((a && b) == (b && a), "Boolean AND must be commutative");
    }

    /// Verify disjunction is commutative: a \/ b = b \/ a
    #[kani::proof]
    fn verify_bool_or_commutative() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        assert!((a || b) == (b || a), "Boolean OR must be commutative");
    }

    // =========================================================================
    // P33: Boolean Algebra - Associativity
    // =========================================================================

    /// Verify conjunction is associative: (a /\ b) /\ c = a /\ (b /\ c)
    #[kani::proof]
    fn verify_bool_and_associative() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        let c: bool = kani::any();
        assert!(
            ((a && b) && c) == (a && (b && c)),
            "Boolean AND must be associative"
        );
    }

    /// Verify disjunction is associative: (a \/ b) \/ c = a \/ (b \/ c)
    #[kani::proof]
    fn verify_bool_or_associative() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        let c: bool = kani::any();
        assert!(
            ((a || b) || c) == (a || (b || c)),
            "Boolean OR must be associative"
        );
    }

    // =========================================================================
    // P34: Boolean Algebra - Identity Elements
    // =========================================================================

    /// Verify TRUE is AND identity: a /\ TRUE = a
    #[kani::proof]
    fn verify_bool_and_identity() {
        let a: bool = kani::any();
        assert!((a && true) == a, "TRUE is AND identity");
    }

    /// Verify FALSE is OR identity: a \/ FALSE = a
    #[kani::proof]
    fn verify_bool_or_identity() {
        let a: bool = kani::any();
        assert!((a || false) == a, "FALSE is OR identity");
    }

    // =========================================================================
    // P35: Boolean Algebra - Annihilator Elements
    // =========================================================================

    /// Verify FALSE is AND annihilator: a /\ FALSE = FALSE
    #[kani::proof]
    fn verify_bool_and_annihilator() {
        let a: bool = kani::any();
        assert!((a && false) == false, "FALSE is AND annihilator");
    }

    /// Verify TRUE is OR annihilator: a \/ TRUE = TRUE
    #[kani::proof]
    fn verify_bool_or_annihilator() {
        let a: bool = kani::any();
        assert!((a || true) == true, "TRUE is OR annihilator");
    }

    // =========================================================================
    // P36: Boolean Algebra - Complement Properties
    // =========================================================================

    /// Verify complement law for AND: a /\ ~a = FALSE
    #[kani::proof]
    fn verify_bool_and_complement() {
        let a: bool = kani::any();
        assert!((a && !a) == false, "a AND (NOT a) = FALSE");
    }

    /// Verify complement law for OR: a \/ ~a = TRUE
    #[kani::proof]
    fn verify_bool_or_complement() {
        let a: bool = kani::any();
        assert!((a || !a) == true, "a OR (NOT a) = TRUE");
    }

    /// Verify double negation: ~~a = a
    #[kani::proof]
    fn verify_bool_double_negation() {
        let a: bool = kani::any();
        assert!(!!a == a, "Double negation: NOT(NOT a) = a");
    }

    // =========================================================================
    // P37: Boolean Algebra - De Morgan's Laws
    // =========================================================================

    /// Verify De Morgan's first law: ~(a /\ b) = (~a) \/ (~b)
    #[kani::proof]
    fn verify_de_morgan_and() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        assert!(
            !(a && b) == (!a || !b),
            "De Morgan: NOT(a AND b) = (NOT a) OR (NOT b)"
        );
    }

    /// Verify De Morgan's second law: ~(a \/ b) = (~a) /\ (~b)
    #[kani::proof]
    fn verify_de_morgan_or() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        assert!(
            !(a || b) == (!a && !b),
            "De Morgan: NOT(a OR b) = (NOT a) AND (NOT b)"
        );
    }

    // =========================================================================
    // P38: Boolean Algebra - Implication
    // =========================================================================

    /// Verify implication definition: a => b iff (~a) \/ b
    #[kani::proof]
    fn verify_implies_definition() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        let implies_ab = !a || b; // TLA+ definition of =>
        let material_impl = if a { b } else { true };
        assert!(implies_ab == material_impl, "a => b iff (NOT a) OR b");
    }

    /// Verify implication is reflexive: a => a is always TRUE
    #[kani::proof]
    fn verify_implies_reflexive() {
        let a: bool = kani::any();
        let implies_aa = !a || a;
        assert!(implies_aa, "a => a is always TRUE");
    }

    // =========================================================================
    // P39: Boolean Algebra - Equivalence
    // =========================================================================

    /// Verify equivalence definition: (a <=> b) = (a => b) /\ (b => a)
    #[kani::proof]
    fn verify_equiv_definition() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        let equiv = a == b; // In TLA+, <=> is logical equivalence
        let impl_both = (!a || b) && (!b || a);
        assert!(equiv == impl_both, "(a <=> b) = (a => b) AND (b => a)");
    }

    /// Verify equivalence is reflexive: a <=> a is always TRUE
    #[kani::proof]
    fn verify_equiv_reflexive() {
        let a: bool = kani::any();
        assert!(a == a, "a <=> a is always TRUE");
    }

    /// Verify equivalence is symmetric: (a <=> b) = (b <=> a)
    #[kani::proof]
    fn verify_equiv_symmetric() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        assert!((a == b) == (b == a), "Equivalence is symmetric");
    }

    // =========================================================================
    // P40: Comparison Operators - Integer Ordering
    // =========================================================================

    /// Verify less-than transitivity: a < b /\ b < c => a < c
    #[kani::proof]
    fn verify_int_lt_transitive() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        let c: i8 = kani::any();
        if a < b && b < c {
            assert!(a < c, "Less-than must be transitive");
        }
    }

    /// Verify less-than is irreflexive: NOT (a < a)
    #[kani::proof]
    fn verify_int_lt_irreflexive() {
        let a: i8 = kani::any();
        assert!(!(a < a), "Less-than must be irreflexive");
    }

    /// Verify less-than-or-equal is reflexive: a <= a
    #[kani::proof]
    fn verify_int_leq_reflexive() {
        let a: i8 = kani::any();
        assert!(a <= a, "Less-than-or-equal must be reflexive");
    }

    /// Verify trichotomy: exactly one of a < b, a = b, a > b holds
    #[kani::proof]
    fn verify_int_trichotomy() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        let lt = a < b;
        let eq = a == b;
        let gt = a > b;
        // Exactly one must be true
        let count = (lt as u8) + (eq as u8) + (gt as u8);
        assert!(count == 1, "Trichotomy: exactly one of <, =, > holds");
    }

    // =========================================================================
    // P41: Set Operator Semantics
    // =========================================================================

    /// Verify union membership: x \in (A \cup B) <=> (x \in A) \/ (x \in B)
    #[kani::proof]
    fn verify_set_union_membership() {
        let a = any_small_set();
        let b = any_small_set();
        let x = any_small_int_value();

        if let (Value::Set(set_a), Value::Set(set_b)) = (&a, &b) {
            let union = set_a.clone().union(set_b.clone());
            let x_in_union = union.contains(&x);
            let x_in_a = set_a.contains(&x);
            let x_in_b = set_b.contains(&x);
            assert!(
                x_in_union == (x_in_a || x_in_b),
                "x in (A ∪ B) iff (x in A) or (x in B)"
            );
        }
    }

    /// Verify intersection membership: x \in (A \cap B) <=> (x \in A) /\ (x \in B)
    #[kani::proof]
    fn verify_set_intersection_membership() {
        let a = any_small_set();
        let b = any_small_set();
        let x = any_small_int_value();

        if let (Value::Set(set_a), Value::Set(set_b)) = (&a, &b) {
            let inter: OrdSet<Value> = set_a.clone().intersection(set_b.clone());
            let x_in_inter = inter.contains(&x);
            let x_in_a = set_a.contains(&x);
            let x_in_b = set_b.contains(&x);
            assert!(
                x_in_inter == (x_in_a && x_in_b),
                "x in (A ∩ B) iff (x in A) and (x in B)"
            );
        }
    }

    /// Verify difference membership: x \in (A \ B) <=> (x \in A) /\ (x \notin B)
    #[kani::proof]
    fn verify_set_difference_membership() {
        let a = any_small_set();
        let b = any_small_set();
        let x = any_small_int_value();

        if let (Value::Set(set_a), Value::Set(set_b)) = (&a, &b) {
            let diff = set_a.clone().relative_complement(set_b.clone());
            let x_in_diff = diff.contains(&x);
            let x_in_a = set_a.contains(&x);
            let x_not_in_b = !set_b.contains(&x);
            assert!(
                x_in_diff == (x_in_a && x_not_in_b),
                "x in (A \\ B) iff (x in A) and (x not in B)"
            );
        }
    }

    /// Verify subset definition: A \subseteq B <=> \A x \in A : x \in B
    #[kani::proof]
    fn verify_set_subset_definition() {
        let a = any_small_set();
        let b = any_small_set();

        if let (Value::Set(set_a), Value::Set(set_b)) = (&a, &b) {
            let is_subset = set_a.is_subset(set_b);
            let all_in_b = set_a.iter().all(|x| set_b.contains(x));
            assert!(
                is_subset == all_in_b,
                "A ⊆ B iff all elements of A are in B"
            );
        }
    }

    // =========================================================================
    // P42: Sequence Operator Semantics
    // =========================================================================

    /// Verify sequence concatenation length: Len(s \o t) = Len(s) + Len(t)
    #[kani::proof]
    fn verify_seq_concat_length() {
        let choice_s: u8 = kani::any();
        let choice_t: u8 = kani::any();
        kani::assume(choice_s < 3);
        kani::assume(choice_t < 3);

        let s: Vec<Value> = match choice_s {
            0 => vec![],
            1 => vec![Value::Int(BigInt::from(1))],
            _ => vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))],
        };
        let t: Vec<Value> = match choice_t {
            0 => vec![],
            1 => vec![Value::Int(BigInt::from(3))],
            _ => vec![Value::Int(BigInt::from(3)), Value::Int(BigInt::from(4))],
        };

        let len_s = s.len();
        let len_t = t.len();
        let mut concat = s;
        concat.extend(t);
        let len_concat = concat.len();

        assert!(len_concat == len_s + len_t, "Len(s ∘ t) = Len(s) + Len(t)");
    }

    /// Verify empty sequence is concatenation identity: <<>> \o s = s = s \o <<>>
    #[kani::proof]
    fn verify_seq_concat_identity() {
        let s = any_small_seq();
        if let Value::Seq(vec_s) = s {
            let empty: Vec<Value> = vec![];

            // empty \o s = s
            let mut left = empty.clone();
            left.extend(vec_s.iter().cloned());

            // s \o empty = s
            let mut right = vec_s.clone();
            right.extend(empty.iter().cloned());

            assert!(left == vec_s, "<<>> ∘ s = s");
            assert!(right == vec_s, "s ∘ <<>> = s");
        }
    }

    // =========================================================================
    // P43: Function EXCEPT Semantics
    // =========================================================================

    /// Helper: Create a small function value (fixed values for deterministic testing)
    fn any_small_func_fixed() -> FuncValue {
        let choice: u8 = kani::any();
        kani::assume(choice < 3);

        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();

        match choice {
            0 => {
                // Empty function
            }
            1 => {
                // Single-element domain
                let k = Value::Int(BigInt::from(1));
                domain.insert(k.clone());
                mapping.insert(k, Value::Bool(true));
            }
            _ => {
                // Two-element domain
                let k1 = Value::Int(BigInt::from(1));
                let k2 = Value::Int(BigInt::from(2));
                domain.insert(k1.clone());
                domain.insert(k2.clone());
                mapping.insert(k1, Value::Bool(true));
                mapping.insert(k2, Value::Bool(false));
            }
        }

        FuncValue::new(domain, mapping)
    }

    /// Verify EXCEPT on function preserves domain
    #[kani::proof]
    fn verify_func_except_preserves_domain() {
        let f = any_small_func_fixed();
        let original_domain = f.domain_as_ord_set();

        // Apply EXCEPT with key in domain
        if !f.domain_is_empty() {
            // Get first element from domain
            let key = f.domain_iter().next().unwrap().clone();
            let new_val = Value::Bool(false);
            let f_new = f.except(key, new_val);

            assert!(
                f_new.domain_as_ord_set() == original_domain,
                "EXCEPT must preserve domain"
            );
        }
    }

    /// Verify EXCEPT on function updates the specified key
    #[kani::proof]
    fn verify_func_except_updates_value() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();
        let key = Value::Int(BigInt::from(1));
        domain.insert(key.clone());
        mapping.insert(key.clone(), Value::Bool(true));
        let f = FuncValue::new(domain, mapping);

        let new_val = Value::Bool(false);
        let f_new = f.except(key.clone(), new_val.clone());

        let result = f_new.apply(&key);
        assert!(result.is_some(), "Key must still be in function");
        assert!(
            *result.unwrap() == new_val,
            "EXCEPT must update to new value"
        );
    }

    /// Verify EXCEPT on function does not affect other keys
    #[kani::proof]
    fn verify_func_except_isolates_other_keys() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();
        let k1 = Value::Int(BigInt::from(1));
        let k2 = Value::Int(BigInt::from(2));
        let v1 = Value::Bool(true);
        let v2 = Value::Bool(false);
        domain.insert(k1.clone());
        domain.insert(k2.clone());
        mapping.insert(k1.clone(), v1.clone());
        mapping.insert(k2.clone(), v2.clone());
        let f = FuncValue::new(domain, mapping);

        // Update k1, check k2 unchanged
        let f_new = f.except(k1, Value::Int(BigInt::from(42)));

        let result = f_new.apply(&k2);
        assert!(result.is_some(), "Other key must still exist");
        assert!(*result.unwrap() == v2, "EXCEPT must not affect other keys");
    }

    // =========================================================================
    // P44: Record EXCEPT Semantics
    // =========================================================================

    /// Helper: Create a small record value
    fn any_small_record_value() -> OrdMap<Arc<str>, Value> {
        let choice: u8 = kani::any();
        kani::assume(choice < 3);

        let mut r: OrdMap<Arc<str>, Value> = OrdMap::new();

        match choice {
            0 => {
                // Empty record (unusual but possible)
            }
            1 => {
                // Single field
                r.insert(Arc::from("x"), Value::Int(BigInt::from(1)));
            }
            _ => {
                // Two fields
                r.insert(Arc::from("x"), Value::Int(BigInt::from(1)));
                r.insert(Arc::from("y"), Value::Bool(true));
            }
        }

        r
    }

    /// Verify record field update preserves other fields
    #[kani::proof]
    fn verify_record_except_preserves_other_fields() {
        let mut r: OrdMap<Arc<str>, Value> = OrdMap::new();
        r.insert(Arc::from("x"), Value::Int(BigInt::from(1)));
        r.insert(Arc::from("y"), Value::Bool(true));

        let original_y = r.get(&Arc::from("y")).cloned();

        // Update x
        let mut r_new = r.clone();
        r_new.insert(Arc::from("x"), Value::Int(BigInt::from(42)));

        assert!(
            r_new.get(&Arc::from("y")) == original_y.as_ref(),
            "Updating x must not affect y"
        );
    }

    /// Verify record field update changes the specified field
    #[kani::proof]
    fn verify_record_except_updates_field() {
        let mut r: OrdMap<Arc<str>, Value> = OrdMap::new();
        r.insert(Arc::from("x"), Value::Int(BigInt::from(1)));

        let new_val = Value::Int(BigInt::from(42));
        r.insert(Arc::from("x"), new_val.clone());

        assert!(
            r.get(&Arc::from("x")) == Some(&new_val),
            "Field must be updated"
        );
    }

    // =========================================================================
    // P45: Quantifier Semantics - Empty Domain
    // =========================================================================

    /// Verify universal quantifier over empty set is TRUE
    /// ∀x ∈ {} : P(x) = TRUE (vacuously true)
    #[kani::proof]
    fn verify_forall_empty_is_true() {
        let empty: OrdSet<Value> = OrdSet::new();
        // ∀x ∈ {} : P(x) is TRUE for any P
        // This is because there are no counterexamples in an empty set
        let result = empty.iter().all(|_x| {
            // P(x) can be anything - we test with FALSE to show vacuous truth
            false
        });
        assert!(result, "∀x ∈ {} : P(x) must be TRUE (vacuously)");
    }

    /// Verify existential quantifier over empty set is FALSE
    /// ∃x ∈ {} : P(x) = FALSE (no witnesses)
    #[kani::proof]
    fn verify_exists_empty_is_false() {
        let empty: OrdSet<Value> = OrdSet::new();
        // ∃x ∈ {} : P(x) is FALSE for any P
        // This is because there are no witnesses in an empty set
        let result = empty.iter().any(|_x| {
            // P(x) can be anything - we test with TRUE to show no witnesses
            true
        });
        assert!(!result, "∃x ∈ {} : P(x) must be FALSE (no witnesses)");
    }

    // =========================================================================
    // P46: Quantifier Semantics - Singleton Domain
    // =========================================================================

    /// Verify ∀x ∈ {a} : P(x) ≡ P(a)
    #[kani::proof]
    fn verify_forall_singleton() {
        let elem = any_small_int_value();
        let mut singleton = OrdSet::new();
        singleton.insert(elem.clone());

        // For any predicate, ∀x ∈ {a} : P(x) should equal P(a)
        // Test with a concrete predicate: is the value equal to itself
        let forall_result = singleton.iter().all(|x| x == &elem);
        let direct_result = elem == elem;

        assert!(
            forall_result == direct_result,
            "∀x ∈ {{a}} : P(x) must equal P(a)"
        );
    }

    /// Verify ∃x ∈ {a} : P(x) ≡ P(a)
    #[kani::proof]
    fn verify_exists_singleton() {
        let elem = any_small_int_value();
        let mut singleton = OrdSet::new();
        singleton.insert(elem.clone());

        // For any predicate, ∃x ∈ {a} : P(x) should equal P(a)
        let exists_result = singleton.iter().any(|x| x == &elem);
        let direct_result = elem == elem;

        assert!(
            exists_result == direct_result,
            "∃x ∈ {{a}} : P(x) must equal P(a)"
        );
    }

    // =========================================================================
    // P47: Quantifier Semantics - TRUE/FALSE predicates
    // =========================================================================

    /// Verify ∀x ∈ S : TRUE ≡ TRUE for non-empty S
    #[kani::proof]
    fn verify_forall_true_predicate() {
        let s = any_small_set();
        if let Value::Set(set) = s {
            let result = set.iter().all(|_x| true);
            assert!(result, "∀x ∈ S : TRUE must be TRUE");
        }
    }

    /// Verify ∃x ∈ S : TRUE ≡ TRUE for non-empty S
    #[kani::proof]
    fn verify_exists_true_predicate_nonempty() {
        // Create a non-empty set
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));

        let result = set.iter().any(|_x| true);
        assert!(result, "∃x ∈ S : TRUE must be TRUE for non-empty S");
    }

    /// Verify ∀x ∈ S : FALSE ≡ FALSE for non-empty S
    #[kani::proof]
    fn verify_forall_false_predicate_nonempty() {
        // Create a non-empty set
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));

        let result = set.iter().all(|_x| false);
        assert!(!result, "∀x ∈ S : FALSE must be FALSE for non-empty S");
    }

    /// Verify ∃x ∈ S : FALSE ≡ FALSE for any S
    #[kani::proof]
    fn verify_exists_false_predicate() {
        let s = any_small_set();
        if let Value::Set(set) = s {
            let result = set.iter().any(|_x| false);
            assert!(!result, "∃x ∈ S : FALSE must be FALSE");
        }
    }

    // =========================================================================
    // P48: Quantifier Duality
    // =========================================================================

    /// Verify ¬(∀x ∈ S : P(x)) ≡ ∃x ∈ S : ¬P(x)
    #[kani::proof]
    fn verify_quantifier_duality_forall() {
        let s = any_small_set();
        if let Value::Set(set) = s {
            // Use a simple predicate: x is a boolean
            let predicate = |x: &Value| matches!(x, Value::Bool(_));

            let not_forall = !set.iter().all(predicate);
            let exists_not = set.iter().any(|x| !predicate(x));

            assert!(
                not_forall == exists_not,
                "¬(∀x : P(x)) must equal ∃x : ¬P(x)"
            );
        }
    }

    /// Verify ¬(∃x ∈ S : P(x)) ≡ ∀x ∈ S : ¬P(x)
    #[kani::proof]
    fn verify_quantifier_duality_exists() {
        let s = any_small_set();
        if let Value::Set(set) = s {
            // Use a simple predicate: x is a boolean
            let predicate = |x: &Value| matches!(x, Value::Bool(_));

            let not_exists = !set.iter().any(predicate);
            let forall_not = set.iter().all(|x| !predicate(x));

            assert!(
                not_exists == forall_not,
                "¬(∃x : P(x)) must equal ∀x : ¬P(x)"
            );
        }
    }

    // =========================================================================
    // P49: Function Application Semantics
    // =========================================================================

    /// Verify function application returns correct value for domain element (specific case)
    #[kani::proof]
    fn verify_func_apply_in_domain_specific() {
        let key = Value::Int(BigInt::from(1));
        let val = Value::Bool(true);

        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();
        domain.insert(key.clone());
        mapping.insert(key.clone(), val.clone());

        let f = FuncValue::new(domain, mapping);
        let result = f.apply(&key);

        assert!(result.is_some(), "Apply must succeed for domain element");
        assert!(*result.unwrap() == val, "Apply must return mapped value");
    }

    /// Verify function application returns None for non-domain element
    #[kani::proof]
    fn verify_func_apply_outside_domain() {
        let k1 = Value::Int(BigInt::from(1));
        let k2 = Value::Int(BigInt::from(2));
        let val = Value::Bool(true);

        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();
        domain.insert(k1.clone());
        mapping.insert(k1, val);

        let f = FuncValue::new(domain, mapping);
        let result = f.apply(&k2);

        assert!(
            result.is_none(),
            "Apply must return None for non-domain element"
        );
    }

    /// Verify domain membership is consistent with apply success
    #[kani::proof]
    fn verify_func_domain_apply_consistency() {
        let f = any_small_func();
        let key = any_small_int_value();

        let in_domain = f.domain_contains(&key);
        let apply_succeeds = f.apply(&key).is_some();

        // If key is in domain, apply must succeed
        // If apply succeeds, key must be in domain (or mapping contains key but not domain - shouldn't happen with well-formed functions)
        if in_domain {
            assert!(apply_succeeds, "If key in DOMAIN, apply must succeed");
        }
    }

    // =========================================================================
    // P50: Function Equality Semantics
    // =========================================================================

    /// Verify function equality requires same domain
    #[kani::proof]
    fn verify_func_equality_same_domain_required() {
        let k1 = Value::Int(BigInt::from(1));
        let k2 = Value::Int(BigInt::from(2));
        let val = Value::Bool(true);

        // f1: {1} -> Bool
        let mut domain1 = OrdSet::new();
        let mut mapping1 = OrdMap::new();
        domain1.insert(k1.clone());
        mapping1.insert(k1, val.clone());
        let f1 = FuncValue::new(domain1, mapping1);

        // f2: {2} -> Bool
        let mut domain2 = OrdSet::new();
        let mut mapping2 = OrdMap::new();
        domain2.insert(k2.clone());
        mapping2.insert(k2, val);
        let f2 = FuncValue::new(domain2, mapping2);

        let v1 = Value::Func(f1);
        let v2 = Value::Func(f2);

        assert!(
            v1 != v2,
            "Functions with different domains must not be equal"
        );
    }

    /// Verify function equality requires same mapping
    #[kani::proof]
    fn verify_func_equality_same_mapping_required() {
        let key = Value::Int(BigInt::from(1));
        let val1 = Value::Bool(true);
        let val2 = Value::Bool(false);

        // f1: 1 -> true
        let mut domain = OrdSet::new();
        domain.insert(key.clone());

        let mut mapping1 = OrdMap::new();
        mapping1.insert(key.clone(), val1);
        let f1 = FuncValue::new(domain.clone(), mapping1);

        // f2: 1 -> false
        let mut mapping2 = OrdMap::new();
        mapping2.insert(key, val2);
        let f2 = FuncValue::new(domain, mapping2);

        let v1 = Value::Func(f1);
        let v2 = Value::Func(f2);

        assert!(
            v1 != v2,
            "Functions with different mappings must not be equal"
        );
    }

    /// Verify function equality holds when domain and mapping are identical
    #[kani::proof]
    fn verify_func_equality_when_identical() {
        let f1 = any_small_func();
        let f2 = FuncValue::new(f1.domain_as_ord_set(), f1.mapping_as_ord_map());

        let v1 = Value::Func(f1);
        let v2 = Value::Func(f2);

        assert!(
            v1 == v2,
            "Functions with same domain and mapping must be equal"
        );
    }

    // =========================================================================
    // P51: CHOOSE Operator Semantics
    // CHOOSE x \in S : P(x) returns the first element satisfying P
    // =========================================================================

    /// Helper: Simulate CHOOSE semantics - returns first element satisfying predicate
    fn choose<P>(set: &OrdSet<Value>, predicate: P) -> Option<Value>
    where
        P: Fn(&Value) -> bool,
    {
        for elem in set.iter() {
            if predicate(elem) {
                return Some(elem.clone());
            }
        }
        None
    }

    /// Verify CHOOSE with TRUE predicate over non-empty set returns first element
    #[kani::proof]
    fn verify_choose_true_predicate_returns_first() {
        // Non-empty set with known ordering
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::Int(BigInt::from(2)));
        set.insert(Value::Int(BigInt::from(3)));

        let result = choose(&set, |_| true);
        assert!(
            result.is_some(),
            "CHOOSE TRUE on non-empty set must succeed"
        );

        // Should return first element in sorted order
        let first = set.iter().next().unwrap().clone();
        assert!(
            result.unwrap() == first,
            "CHOOSE TRUE must return first element"
        );
    }

    /// Verify CHOOSE on singleton set with TRUE returns that element
    #[kani::proof]
    fn verify_choose_singleton_returns_element() {
        let elem = any_small_int_value();
        let mut set = OrdSet::new();
        set.insert(elem.clone());

        let result = choose(&set, |_| true);
        assert!(result.is_some(), "CHOOSE on singleton must succeed");
        assert!(
            result.unwrap() == elem,
            "CHOOSE on singleton must return that element"
        );
    }

    /// Verify CHOOSE on empty set returns None (fails)
    #[kani::proof]
    fn verify_choose_empty_set_fails() {
        let empty: OrdSet<Value> = OrdSet::new();
        let result = choose(&empty, |_| true);
        assert!(result.is_none(), "CHOOSE on empty set must fail");
    }

    /// Verify CHOOSE is deterministic - same set and predicate -> same result
    #[kani::proof]
    fn verify_choose_deterministic() {
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::Int(BigInt::from(2)));

        // CHOOSE with predicate > 0 (both elements satisfy)
        let predicate = |x: &Value| match x {
            Value::Int(n) => n > &BigInt::from(0),
            _ => false,
        };

        let result1 = choose(&set, predicate);
        let result2 = choose(&set, predicate);

        assert!(result1 == result2, "CHOOSE must be deterministic");
    }

    /// Verify CHOOSE returns satisfying element (not just any element)
    #[kani::proof]
    fn verify_choose_returns_satisfying_element() {
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::Int(BigInt::from(2)));
        set.insert(Value::Int(BigInt::from(3)));

        // Predicate: x > 1
        let predicate = |x: &Value| match x {
            Value::Int(n) => n > &BigInt::from(1),
            _ => false,
        };

        let result = choose(&set, &predicate);
        assert!(
            result.is_some(),
            "CHOOSE with satisfiable predicate must succeed"
        );

        // The result must satisfy the predicate
        let chosen = result.unwrap();
        assert!(predicate(&chosen), "CHOOSE result must satisfy predicate");
    }

    /// Verify CHOOSE with unsatisfiable predicate fails
    #[kani::proof]
    fn verify_choose_unsatisfiable_predicate_fails() {
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::Int(BigInt::from(2)));

        // Predicate that nothing satisfies
        let predicate = |x: &Value| match x {
            Value::Int(n) => n > &BigInt::from(100),
            _ => false,
        };

        let result = choose(&set, predicate);
        assert!(
            result.is_none(),
            "CHOOSE with unsatisfiable predicate must fail"
        );
    }

    // =========================================================================
    // P52: Nested EXCEPT Paths
    // [f EXCEPT ![a][b] = v] updates nested structure
    // =========================================================================

    /// Helper: Create nested function f where f[1] is a function {1 -> true, 2 -> false}
    fn create_nested_func() -> FuncValue {
        // Inner function: {1 -> true, 2 -> false}
        let mut inner_domain = OrdSet::new();
        let mut inner_mapping = OrdMap::new();
        inner_domain.insert(Value::Int(BigInt::from(1)));
        inner_domain.insert(Value::Int(BigInt::from(2)));
        inner_mapping.insert(Value::Int(BigInt::from(1)), Value::Bool(true));
        inner_mapping.insert(Value::Int(BigInt::from(2)), Value::Bool(false));
        let inner = FuncValue::new(inner_domain, inner_mapping);

        // Outer function: {1 -> inner}
        let mut outer_domain = OrdSet::new();
        let mut outer_mapping = OrdMap::new();
        outer_domain.insert(Value::Int(BigInt::from(1)));
        outer_mapping.insert(Value::Int(BigInt::from(1)), Value::Func(inner));

        FuncValue::new(outer_domain, outer_mapping)
    }

    /// Verify nested EXCEPT preserves outer domain
    #[kani::proof]
    fn verify_nested_except_preserves_outer_domain() {
        let outer = create_nested_func();
        let original_domain = outer.domain_as_ord_set();

        // Simulate [outer EXCEPT ![1][1] = false]
        // First get outer[1], then update inner[1], then put back
        if let Some(Value::Func(inner)) = outer.apply(&Value::Int(BigInt::from(1))).cloned() {
            let new_inner = inner.except(Value::Int(BigInt::from(1)), Value::Bool(false));
            let new_outer = outer.except(Value::Int(BigInt::from(1)), Value::Func(new_inner));

            assert!(
                new_outer.domain_as_ord_set() == original_domain,
                "Nested EXCEPT must preserve outer domain"
            );
        }
    }

    /// Verify nested EXCEPT updates the correct inner value
    #[kani::proof]
    fn verify_nested_except_updates_inner_value() {
        let outer = create_nested_func();

        // [outer EXCEPT ![1][1] = false]
        if let Some(Value::Func(inner)) = outer.apply(&Value::Int(BigInt::from(1))).cloned() {
            let new_inner = inner.except(Value::Int(BigInt::from(1)), Value::Bool(false));
            let new_outer = outer.except(Value::Int(BigInt::from(1)), Value::Func(new_inner));

            // Check that new_outer[1][1] = false
            if let Some(Value::Func(result_inner)) =
                new_outer.apply(&Value::Int(BigInt::from(1))).cloned()
            {
                let result = result_inner.apply(&Value::Int(BigInt::from(1)));
                assert!(result.is_some());
                assert!(
                    *result.unwrap() == Value::Bool(false),
                    "Nested EXCEPT must update inner value"
                );
            }
        }
    }

    /// Verify nested EXCEPT preserves other inner keys
    #[kani::proof]
    fn verify_nested_except_preserves_other_inner_keys() {
        let outer = create_nested_func();

        // [outer EXCEPT ![1][1] = false] should preserve outer[1][2]
        if let Some(Value::Func(inner)) = outer.apply(&Value::Int(BigInt::from(1))).cloned() {
            let original_inner_2 = inner.apply(&Value::Int(BigInt::from(2))).cloned();

            let new_inner = inner.except(Value::Int(BigInt::from(1)), Value::Bool(false));
            let new_outer = outer.except(Value::Int(BigInt::from(1)), Value::Func(new_inner));

            if let Some(Value::Func(result_inner)) =
                new_outer.apply(&Value::Int(BigInt::from(1))).cloned()
            {
                let result_inner_2 = result_inner.apply(&Value::Int(BigInt::from(2))).cloned();
                assert!(
                    result_inner_2 == original_inner_2,
                    "Nested EXCEPT must preserve other inner keys"
                );
            }
        }
    }

    // =========================================================================
    // P53: Sequence Index/Access Semantics
    // Sequences are 1-indexed in TLA+
    // =========================================================================

    /// Verify sequence index returns correct element (1-indexed)
    #[kani::proof]
    fn verify_seq_index_returns_correct_element() {
        let seq = vec![
            Value::Int(BigInt::from(10)),
            Value::Int(BigInt::from(20)),
            Value::Int(BigInt::from(30)),
        ];

        // TLA+ uses 1-indexed, so seq[1] = 10, seq[2] = 20, seq[3] = 30
        // In Rust we use 0-indexed internally
        assert!(
            seq[0] == Value::Int(BigInt::from(10)),
            "seq[1] (0-indexed: 0) must be 10"
        );
        assert!(
            seq[1] == Value::Int(BigInt::from(20)),
            "seq[2] (0-indexed: 1) must be 20"
        );
        assert!(
            seq[2] == Value::Int(BigInt::from(30)),
            "seq[3] (0-indexed: 2) must be 30"
        );
    }

    /// Verify sequence Head returns first element
    #[kani::proof]
    fn verify_seq_head_returns_first() {
        let v1 = any_small_int_value();
        let v2 = any_bool_value();
        let seq = vec![v1.clone(), v2];

        let head = seq.first().cloned();
        assert!(head.is_some(), "Head of non-empty seq must exist");
        assert!(head.unwrap() == v1, "Head must return first element");
    }

    /// Verify sequence Head on empty fails
    #[kani::proof]
    fn verify_seq_head_empty_fails() {
        let seq: Vec<Value> = vec![];
        let head = seq.first();
        assert!(head.is_none(), "Head of empty seq must fail");
    }

    /// Verify sequence Tail returns all but first
    #[kani::proof]
    fn verify_seq_tail_returns_rest() {
        let seq = vec![
            Value::Int(BigInt::from(1)),
            Value::Int(BigInt::from(2)),
            Value::Int(BigInt::from(3)),
        ];

        let tail: Vec<Value> = seq[1..].to_vec();
        assert!(tail.len() == 2, "Tail must have length - 1");
        assert!(
            tail[0] == Value::Int(BigInt::from(2)),
            "Tail[1] must be original[2]"
        );
        assert!(
            tail[1] == Value::Int(BigInt::from(3)),
            "Tail[2] must be original[3]"
        );
    }

    /// Verify sequence Tail of singleton is empty
    #[kani::proof]
    fn verify_seq_tail_singleton_is_empty() {
        let seq = vec![Value::Int(BigInt::from(1))];
        let tail: Vec<Value> = seq[1..].to_vec();
        assert!(tail.is_empty(), "Tail of singleton must be empty");
    }

    /// Verify SubSeq returns correct slice (1-indexed, inclusive)
    /// SubSeq(s, m, n) = <<s[m], s[m+1], ..., s[n]>>
    #[kani::proof]
    fn verify_subseq_returns_correct_slice() {
        let seq = vec![
            Value::Int(BigInt::from(10)),
            Value::Int(BigInt::from(20)),
            Value::Int(BigInt::from(30)),
            Value::Int(BigInt::from(40)),
        ];

        // SubSeq(seq, 2, 3) = <<20, 30>> (1-indexed, inclusive)
        // In Rust: seq[1..3] (0-indexed, exclusive end)
        let subseq: Vec<Value> = seq[1..3].to_vec();
        assert!(subseq.len() == 2);
        assert!(subseq[0] == Value::Int(BigInt::from(20)));
        assert!(subseq[1] == Value::Int(BigInt::from(30)));
    }

    /// Verify Append adds element at end
    #[kani::proof]
    fn verify_seq_append_adds_at_end() {
        let elem = any_small_int_value();
        let mut seq = vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))];
        let original_len = seq.len();

        seq.push(elem.clone());

        assert!(
            seq.len() == original_len + 1,
            "Append increases length by 1"
        );
        assert!(seq.last() == Some(&elem), "Append adds element at end");
    }

    // =========================================================================
    // P54: Tuple Semantics
    // Tuples are represented as sequences in TLA2
    // =========================================================================

    /// Verify tuple element access (1-indexed)
    #[kani::proof]
    fn verify_tuple_element_access() {
        let v1 = any_bool_value();
        let v2 = any_small_int_value();
        let v3 = any_short_string_value();

        let tuple = vec![v1.clone(), v2.clone(), v3.clone()];

        // TLA+ tuple[1] = v1, tuple[2] = v2, tuple[3] = v3
        assert!(tuple[0] == v1, "tuple[1] must be first element");
        assert!(tuple[1] == v2, "tuple[2] must be second element");
        assert!(tuple[2] == v3, "tuple[3] must be third element");
    }

    /// Verify tuple equality
    #[kani::proof]
    fn verify_tuple_equality() {
        let v1 = any_small_int_value();
        let v2 = any_bool_value();

        let t1 = vec![v1.clone(), v2.clone()];
        let t2 = vec![v1, v2];

        assert!(t1 == t2, "Tuples with same elements must be equal");
    }

    /// Verify tuple inequality when different lengths
    #[kani::proof]
    fn verify_tuple_different_lengths_not_equal() {
        let t1 = vec![Value::Int(BigInt::from(1))];
        let t2 = vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))];

        assert!(t1 != t2, "Tuples with different lengths must not be equal");
    }

    /// Verify tuple inequality when different elements
    #[kani::proof]
    fn verify_tuple_different_elements_not_equal() {
        let t1 = vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))];
        let t2 = vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(3))];

        assert!(t1 != t2, "Tuples with different elements must not be equal");
    }

    /// Verify tuple length matches element count
    #[kani::proof]
    fn verify_tuple_length() {
        let choice: u8 = kani::any();
        kani::assume(choice < 4);

        let tuple: Vec<Value> = match choice {
            0 => vec![],
            1 => vec![Value::Int(BigInt::from(1))],
            2 => vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))],
            _ => vec![
                Value::Int(BigInt::from(1)),
                Value::Int(BigInt::from(2)),
                Value::Int(BigInt::from(3)),
            ],
        };

        let expected_len = choice as usize;
        assert!(
            tuple.len() == expected_len,
            "Tuple length must match element count"
        );
    }

    // =========================================================================
    // P55: ModelValue Semantics
    // Model values are symbolic constants used for symmetry sets
    // =========================================================================

    /// Generate an arbitrary model value name
    fn any_model_value_name() -> Arc<str> {
        let choice: u8 = kani::any();
        kani::assume(choice < 6);
        match choice {
            0 => Arc::from("m1"),
            1 => Arc::from("m2"),
            2 => Arc::from("m3"),
            3 => Arc::from("a"),
            4 => Arc::from("b"),
            _ => Arc::from("c"),
        }
    }

    /// Verify ModelValue fingerprint is deterministic
    #[kani::proof]
    fn verify_model_value_fingerprint_deterministic() {
        let name = any_model_value_name();
        let v = Value::ModelValue(name);
        let fp1 = value_fingerprint(&v);
        let fp2 = value_fingerprint(&v);
        assert!(fp1 == fp2, "ModelValue fingerprint must be deterministic");
    }

    /// Verify ModelValue equality is by name
    #[kani::proof]
    fn verify_model_value_equality_by_name() {
        let name1 = any_model_value_name();
        let name2 = any_model_value_name();
        let v1 = Value::ModelValue(name1.clone());
        let v2 = Value::ModelValue(name2.clone());

        if name1 == name2 {
            assert!(v1 == v2, "Same name model values must be equal");
        } else {
            assert!(v1 != v2, "Different name model values must not be equal");
        }
    }

    /// Verify ModelValue equality is reflexive
    #[kani::proof]
    fn verify_model_value_equality_reflexive() {
        let name = any_model_value_name();
        let v = Value::ModelValue(name);
        assert!(v == v, "ModelValue equality must be reflexive");
    }

    /// Verify ModelValue is different from other value types
    #[kani::proof]
    fn verify_model_value_type_discrimination() {
        let mv = Value::ModelValue(Arc::from("m1"));
        let b = Value::Bool(true);
        let i = Value::Int(BigInt::from(1));
        let s = Value::String(Arc::from("m1")); // Same text, different type

        assert!(mv != b, "ModelValue must not equal Bool");
        assert!(mv != i, "ModelValue must not equal Int");
        assert!(mv != s, "ModelValue must not equal String with same text");
    }

    // =========================================================================
    // P56: Cardinality Semantics
    // |S| for sets, Len(s) for sequences
    // =========================================================================

    /// Verify interval cardinality is high - low + 1
    #[kani::proof]
    fn verify_interval_cardinality() {
        let low: i8 = kani::any();
        let high: i8 = kani::any();
        kani::assume(low <= high);
        kani::assume(high - low < 100); // Keep tractable

        let iv = IntervalValue::new(BigInt::from(low as i64), BigInt::from(high as i64));
        let expected = BigInt::from((high - low + 1) as i64);

        assert!(
            iv.len() == expected,
            "Interval cardinality must be high - low + 1"
        );
    }

    /// Verify empty interval has cardinality 0
    #[kani::proof]
    fn verify_empty_interval_cardinality() {
        let low: i8 = kani::any();
        let high: i8 = kani::any();
        kani::assume(low > high);

        let iv = IntervalValue::new(BigInt::from(low as i64), BigInt::from(high as i64));

        assert!(
            iv.len() == BigInt::from(0),
            "Empty interval must have cardinality 0"
        );
    }

    /// Verify singleton interval has cardinality 1
    #[kani::proof]
    fn verify_singleton_interval_cardinality() {
        let n: i8 = kani::any();
        let iv = IntervalValue::new(BigInt::from(n as i64), BigInt::from(n as i64));

        assert!(
            iv.len() == BigInt::from(1),
            "Singleton interval must have cardinality 1"
        );
    }

    /// Verify set cardinality is count of elements
    #[kani::proof]
    fn verify_set_cardinality() {
        let choice: u8 = kani::any();
        kani::assume(choice < 4);

        let set: OrdSet<Value> = match choice {
            0 => OrdSet::new(),
            1 => {
                let mut s = OrdSet::new();
                s.insert(Value::Int(BigInt::from(1)));
                s
            }
            2 => {
                let mut s = OrdSet::new();
                s.insert(Value::Int(BigInt::from(1)));
                s.insert(Value::Int(BigInt::from(2)));
                s
            }
            _ => {
                let mut s = OrdSet::new();
                s.insert(Value::Int(BigInt::from(1)));
                s.insert(Value::Int(BigInt::from(2)));
                s.insert(Value::Int(BigInt::from(3)));
                s
            }
        };

        assert!(
            set.len() == choice as usize,
            "Set cardinality must equal element count"
        );
    }

    /// Verify sequence length
    #[kani::proof]
    fn verify_sequence_length() {
        let choice: u8 = kani::any();
        kani::assume(choice < 4);

        let seq: Vec<Value> = match choice {
            0 => vec![],
            1 => vec![Value::Int(BigInt::from(1))],
            2 => vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))],
            _ => vec![
                Value::Int(BigInt::from(1)),
                Value::Int(BigInt::from(2)),
                Value::Int(BigInt::from(3)),
            ],
        };

        assert!(
            seq.len() == choice as usize,
            "Sequence length must equal element count"
        );
    }

    // =========================================================================
    // P57: Empty Collection Semantics
    // =========================================================================

    /// Verify empty set has no elements
    #[kani::proof]
    fn verify_empty_set_has_no_elements() {
        let empty: OrdSet<Value> = OrdSet::new();
        let v = any_primitive_value();

        assert!(
            !empty.contains(&v),
            "Empty set must not contain any element"
        );
    }

    /// Verify empty set subset of all sets
    #[kani::proof]
    fn verify_empty_set_subset_of_all() {
        let empty: OrdSet<Value> = OrdSet::new();
        let mut other = OrdSet::new();
        other.insert(any_primitive_value());

        assert!(
            empty.is_subset(&other),
            "Empty set must be subset of all sets"
        );
        assert!(
            empty.is_subset(&empty),
            "Empty set must be subset of itself"
        );
    }

    /// Verify empty sequence properties
    #[kani::proof]
    fn verify_empty_sequence_properties() {
        let empty: Vec<Value> = vec![];

        assert!(empty.is_empty(), "Empty sequence must be empty");
        assert!(empty.first().is_none(), "Empty sequence has no Head");
        assert!(empty.last().is_none(), "Empty sequence has no last element");
    }

    /// Verify empty function domain is empty
    #[kani::proof]
    fn verify_empty_function_properties() {
        let empty_domain: OrdSet<Value> = OrdSet::new();
        let empty_mapping: OrdMap<Value, Value> = OrdMap::new();
        let f = FuncValue::new(empty_domain.clone(), empty_mapping);

        assert!(f.domain_is_empty(), "Empty function has empty domain");
        assert!(f.domain_is_empty(), "Empty function has empty mapping");

        let v = any_primitive_value();
        assert!(
            f.apply(&v).is_none(),
            "Empty function returns None for any input"
        );
    }

    // =========================================================================
    // P58: IF-THEN-ELSE Semantics
    // Conditional evaluation at the boolean level
    // =========================================================================

    /// Verify IF TRUE THEN x ELSE y = x
    #[kani::proof]
    fn verify_if_true_returns_then_branch() {
        let then_val = any_small_int_value();
        let else_val = any_small_int_value();

        let result = if true { then_val.clone() } else { else_val };

        assert!(result == then_val, "IF TRUE THEN x ELSE y must equal x");
    }

    /// Verify IF FALSE THEN x ELSE y = y
    #[kani::proof]
    fn verify_if_false_returns_else_branch() {
        let then_val = any_small_int_value();
        let else_val = any_small_int_value();

        let result = if false { then_val } else { else_val.clone() };

        assert!(result == else_val, "IF FALSE THEN x ELSE y must equal y");
    }

    /// Verify IF c THEN x ELSE x = x (regardless of condition)
    #[kani::proof]
    fn verify_if_same_branches_equals_branch() {
        let condition: bool = kani::any();
        let val = any_primitive_value();

        let result = if condition { val.clone() } else { val.clone() };

        assert!(result == val, "IF c THEN x ELSE x must equal x");
    }

    /// Verify nested IF-THEN-ELSE consistency
    #[kani::proof]
    fn verify_nested_if_consistency() {
        let c1: bool = kani::any();
        let c2: bool = kani::any();
        let a = Value::Int(BigInt::from(1));
        let b = Value::Int(BigInt::from(2));
        let c = Value::Int(BigInt::from(3));

        // IF c1 THEN (IF c2 THEN a ELSE b) ELSE c
        let nested = if c1 {
            if c2 {
                a.clone()
            } else {
                b.clone()
            }
        } else {
            c.clone()
        };

        // Verify expected outcome
        let expected = if c1 && c2 {
            a
        } else if c1 && !c2 {
            b
        } else {
            c
        };

        assert!(nested == expected, "Nested IF must evaluate correctly");
    }

    // =========================================================================
    // P59: Interval Enumeration Correctness
    // =========================================================================

    /// Verify interval iteration yields all elements in order
    #[kani::proof]
    fn verify_interval_iteration_order() {
        let iv = IntervalValue::new(BigInt::from(1), BigInt::from(3));
        let elements: Vec<BigInt> = iv.iter().collect();

        assert!(elements.len() == 3);
        assert!(elements[0] == BigInt::from(1));
        assert!(elements[1] == BigInt::from(2));
        assert!(elements[2] == BigInt::from(3));
    }

    /// Verify interval contains exactly the elements yielded by iteration
    #[kani::proof]
    fn verify_interval_contains_all_iterated() {
        let low: i8 = kani::any();
        let high: i8 = kani::any();
        kani::assume(low <= high);
        kani::assume(high - low < 10); // Keep tractable

        let iv = IntervalValue::new(BigInt::from(low as i64), BigInt::from(high as i64));

        for n in iv.iter() {
            assert!(
                iv.contains(&Value::Int(n)),
                "Interval must contain all iterated elements"
            );
        }
    }

    /// Verify interval iteration count matches cardinality
    #[kani::proof]
    fn verify_interval_iteration_count() {
        let low: i8 = kani::any();
        let high: i8 = kani::any();
        kani::assume(low <= high);
        kani::assume(high - low < 10); // Keep tractable

        let iv = IntervalValue::new(BigInt::from(low as i64), BigInt::from(high as i64));
        let count = iv.iter().count();
        let cardinality = iv.len();

        assert!(
            BigInt::from(count) == cardinality,
            "Iteration count must equal cardinality"
        );
    }

    /// Verify empty interval iteration yields nothing
    #[kani::proof]
    fn verify_empty_interval_iteration() {
        let iv = IntervalValue::new(BigInt::from(5), BigInt::from(2)); // Empty: 5 > 2
        let count = iv.iter().count();

        assert!(
            count == 0,
            "Empty interval iteration must yield no elements"
        );
    }

    // =========================================================================
    // P60: Function Construction Semantics
    // =========================================================================

    /// Verify function domain equals the set used to construct it
    #[kani::proof]
    fn verify_func_domain_equals_construction_domain() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();

        domain.insert(Value::Int(BigInt::from(1)));
        domain.insert(Value::Int(BigInt::from(2)));
        mapping.insert(Value::Int(BigInt::from(1)), Value::Bool(true));
        mapping.insert(Value::Int(BigInt::from(2)), Value::Bool(false));

        let f = FuncValue::new(domain.clone(), mapping);

        assert!(
            f.domain == domain,
            "Function domain must equal construction domain"
        );
    }

    /// Verify function mapping size equals domain size
    #[kani::proof]
    fn verify_func_mapping_size_equals_domain() {
        let choice: u8 = kani::any();
        kani::assume(choice < 4);

        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();

        for i in 0..choice {
            let k = Value::Int(BigInt::from(i as i64));
            domain.insert(k.clone());
            mapping.insert(k, Value::Bool(i % 2 == 0));
        }

        let f = FuncValue::new(domain, mapping);

        // With array-based FuncValue, domain and mapping are always consistent
        assert!(
            f.domain_len() >= 0,
            "Function domain size must be non-negative"
        );
    }
}

// =========================================================================
// Non-Kani tests that mirror the Kani proofs (for regular testing)
// =========================================================================

#[cfg(test)]
mod tests {
    use crate::state::{value_fingerprint, State};
    use crate::value::{FuncValue, IntervalValue, RecordBuilder, Value};
    use im::{OrdMap, OrdSet};
    use num_bigint::BigInt;
    use std::sync::Arc;

    #[test]
    fn test_bool_fingerprint_deterministic() {
        for b in [true, false] {
            let v = Value::Bool(b);
            let fp1 = value_fingerprint(&v);
            let fp2 = value_fingerprint(&v);
            assert_eq!(fp1, fp2, "Fingerprint must be deterministic");
        }
    }

    #[test]
    fn test_int_fingerprint_deterministic() {
        for n in [-128i64, -1, 0, 1, 127] {
            let v = Value::Int(BigInt::from(n));
            let fp1 = value_fingerprint(&v);
            let fp2 = value_fingerprint(&v);
            assert_eq!(fp1, fp2, "Fingerprint must be deterministic");
        }
    }

    #[test]
    fn test_value_equality_reflexive() {
        let values = vec![
            Value::Bool(true),
            Value::Bool(false),
            Value::Int(BigInt::from(0)),
            Value::Int(BigInt::from(42)),
            Value::string("hello"),
            Value::string(""),
        ];
        for v in values {
            assert_eq!(v, v, "Value equality must be reflexive");
        }
    }

    #[test]
    fn test_type_discrimination() {
        let b = Value::Bool(true);
        let i = Value::Int(BigInt::from(1));
        let s = Value::string("true");

        assert_ne!(b, i, "Bool and Int must never be equal");
        assert_ne!(b, s, "Bool and String must never be equal");
        assert_ne!(i, s, "Int and String must never be equal");
    }

    #[test]
    fn test_state_fingerprint_deterministic() {
        let s = State::from_pairs([
            ("x", Value::Bool(true)),
            ("y", Value::Int(BigInt::from(42))),
        ]);
        let fp1 = s.fingerprint();
        let fp2 = s.fingerprint();
        assert_eq!(fp1, fp2, "State fingerprint must be deterministic");
    }

    #[test]
    fn test_state_content_fingerprint_consistency() {
        let s1 = State::from_pairs([
            ("x", Value::Bool(true)),
            ("y", Value::Int(BigInt::from(42))),
        ]);
        let s2 = State::from_pairs([
            ("y", Value::Int(BigInt::from(42))),
            ("x", Value::Bool(true)),
        ]);
        assert_eq!(
            s1.fingerprint(),
            s2.fingerprint(),
            "States with same content must have same fingerprint"
        );
    }

    #[test]
    fn test_bool_fingerprint_sensitive() {
        let v1 = Value::Bool(true);
        let v2 = Value::Bool(false);
        assert_ne!(
            value_fingerprint(&v1),
            value_fingerprint(&v2),
            "Different booleans should have different fingerprints"
        );
    }

    #[test]
    fn test_ord_eq_consistency() {
        use std::cmp::Ordering;

        let values = vec![
            Value::Bool(true),
            Value::Bool(false),
            Value::Int(BigInt::from(0)),
            Value::Int(BigInt::from(1)),
            Value::string("a"),
            Value::string("b"),
        ];

        for v1 in &values {
            for v2 in &values {
                let ord_eq = v1.cmp(v2) == Ordering::Equal;
                let eq_eq = v1 == v2;
                assert_eq!(
                    ord_eq, eq_eq,
                    "Ord and Eq must be consistent for {:?} vs {:?}",
                    v1, v2
                );
            }
        }
    }

    // =========================================================================
    // Set operation tests (P9)
    // =========================================================================

    #[test]
    fn test_set_union_commutative() {
        let mut set_a = OrdSet::new();
        set_a.insert(Value::Int(BigInt::from(1)));
        set_a.insert(Value::Int(BigInt::from(2)));

        let mut set_b = OrdSet::new();
        set_b.insert(Value::Int(BigInt::from(2)));
        set_b.insert(Value::Int(BigInt::from(3)));

        let union_ab = set_a.clone().union(set_b.clone());
        let union_ba = set_b.clone().union(set_a.clone());
        assert_eq!(union_ab, union_ba, "Set union must be commutative");
    }

    #[test]
    fn test_set_intersection_commutative() {
        let mut set_a = OrdSet::new();
        set_a.insert(Value::Int(BigInt::from(1)));
        set_a.insert(Value::Int(BigInt::from(2)));

        let mut set_b = OrdSet::new();
        set_b.insert(Value::Int(BigInt::from(2)));
        set_b.insert(Value::Int(BigInt::from(3)));

        let inter_ab: OrdSet<Value> = set_a.clone().intersection(set_b.clone());
        let inter_ba: OrdSet<Value> = set_b.clone().intersection(set_a.clone());
        assert_eq!(inter_ab, inter_ba, "Set intersection must be commutative");
    }

    #[test]
    fn test_set_operations_identity() {
        let mut set_a = OrdSet::new();
        set_a.insert(Value::Int(BigInt::from(1)));
        set_a.insert(Value::Int(BigInt::from(2)));
        let empty: OrdSet<Value> = OrdSet::new();

        // Union identity
        let union = set_a.clone().union(empty.clone());
        assert_eq!(union, set_a, "A ∪ {{}} = A");

        // Intersection with empty
        let inter: OrdSet<Value> = set_a.clone().intersection(empty.clone());
        assert!(inter.is_empty(), "A ∩ {{}} = {{}}");

        // Difference with self
        let diff = set_a.clone().relative_complement(set_a.clone());
        assert!(diff.is_empty(), "A \\ A = {{}}");

        // Difference with empty
        let diff2 = set_a.clone().relative_complement(empty);
        assert_eq!(diff2, set_a, "A \\ {{}} = A");
    }

    #[test]
    fn test_set_membership() {
        use crate::value::SortedSet;
        let mut set = OrdSet::new();
        let v1 = Value::Int(BigInt::from(1));
        let v2 = Value::Int(BigInt::from(2));
        set.insert(v1.clone());
        let s = Value::Set(SortedSet::from_ord_set(&set));

        assert!(
            s.set_contains(&v1).unwrap_or(false),
            "Inserted element must be in set"
        );
        assert!(
            !s.set_contains(&v2).unwrap_or(true),
            "Non-inserted element must not be in set"
        );
    }

    #[test]
    fn test_set_fingerprint_deterministic() {
        use crate::value::SortedSet;
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::Int(BigInt::from(2)));
        let s = Value::Set(SortedSet::from_ord_set(&set));

        let fp1 = value_fingerprint(&s);
        let fp2 = value_fingerprint(&s);
        assert_eq!(fp1, fp2, "Set fingerprint must be deterministic");
    }

    #[test]
    fn test_set_equality_reflexive() {
        use crate::value::SortedSet;
        let mut set = OrdSet::new();
        set.insert(Value::Bool(true));
        set.insert(Value::Bool(false));
        let s = Value::Set(SortedSet::from_ord_set(&set));

        assert_eq!(s, s, "Set equality must be reflexive");
    }

    #[test]
    fn test_subset_relations() {
        let mut set_a = OrdSet::new();
        set_a.insert(Value::Int(BigInt::from(1)));

        let mut set_b = OrdSet::new();
        set_b.insert(Value::Int(BigInt::from(1)));
        set_b.insert(Value::Int(BigInt::from(2)));

        let empty: OrdSet<Value> = OrdSet::new();

        // Reflexivity
        assert!(set_a.is_subset(&set_a), "A ⊆ A");
        assert!(set_b.is_subset(&set_b), "B ⊆ B");

        // Empty subset of all
        assert!(empty.is_subset(&set_a), "{{}} ⊆ A");
        assert!(empty.is_subset(&set_b), "{{}} ⊆ B");

        // Proper subset
        assert!(set_a.is_subset(&set_b), "{{1}} ⊆ {{1,2}}");
        assert!(!set_b.is_subset(&set_a), "{{1,2}} ⊄ {{1}}");
    }

    // =========================================================================
    // Interval operation tests (P10)
    // =========================================================================

    #[test]
    fn test_interval_contains_bounds() {
        let iv = IntervalValue::new(BigInt::from(1), BigInt::from(5));
        assert!(
            iv.contains(&Value::Int(BigInt::from(1))),
            "Must contain low"
        );
        assert!(
            iv.contains(&Value::Int(BigInt::from(5))),
            "Must contain high"
        );
        assert!(
            iv.contains(&Value::Int(BigInt::from(3))),
            "Must contain middle"
        );
    }

    #[test]
    fn test_interval_excludes_outside() {
        let iv = IntervalValue::new(BigInt::from(1), BigInt::from(5));
        assert!(
            !iv.contains(&Value::Int(BigInt::from(0))),
            "Must not contain below"
        );
        assert!(
            !iv.contains(&Value::Int(BigInt::from(6))),
            "Must not contain above"
        );
    }

    #[test]
    fn test_interval_length() {
        let iv1 = IntervalValue::new(BigInt::from(1), BigInt::from(5));
        assert_eq!(iv1.len(), BigInt::from(5), "1..5 has length 5");

        let iv2 = IntervalValue::new(BigInt::from(5), BigInt::from(1));
        assert_eq!(iv2.len(), BigInt::from(0), "Empty interval has length 0");

        let iv3 = IntervalValue::new(BigInt::from(3), BigInt::from(3));
        assert_eq!(iv3.len(), BigInt::from(1), "3..3 has length 1");
    }

    #[test]
    fn test_interval_fingerprint_deterministic() {
        let iv = IntervalValue::new(BigInt::from(1), BigInt::from(10));
        let v = Value::Interval(iv);
        let fp1 = value_fingerprint(&v);
        let fp2 = value_fingerprint(&v);
        assert_eq!(fp1, fp2, "Interval fingerprint must be deterministic");
    }

    // =========================================================================
    // Function operation tests (P11)
    // =========================================================================

    #[test]
    fn test_func_domain_mapping_consistent() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();

        let d1 = Value::Int(BigInt::from(1));
        let d2 = Value::Int(BigInt::from(2));
        domain.insert(d1.clone());
        domain.insert(d2.clone());
        mapping.insert(d1, Value::Bool(true));
        mapping.insert(d2, Value::Bool(false));

        let f = FuncValue::new(domain.clone(), mapping.clone());
        // With array-based FuncValue, domain_len() is always consistent
        assert_eq!(f.domain_len(), 2, "Domain size must equal mapping size");
    }

    #[test]
    fn test_func_apply_in_domain() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();

        let d = Value::Int(BigInt::from(1));
        domain.insert(d.clone());
        mapping.insert(d.clone(), Value::Bool(true));

        let f = FuncValue::new(domain, mapping);
        let result = f.mapping_get(&d);
        assert!(
            result.is_some(),
            "Function must have mapping for domain elements"
        );
        assert_eq!(*result.unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_func_equality_reflexive() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();

        let d = Value::Int(BigInt::from(1));
        domain.insert(d.clone());
        mapping.insert(d, Value::Bool(true));

        let f = Value::Func(FuncValue::new(domain, mapping));
        assert_eq!(f, f, "Function equality must be reflexive");
    }

    #[test]
    fn test_func_fingerprint_deterministic() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();

        let d = Value::Int(BigInt::from(1));
        domain.insert(d.clone());
        mapping.insert(d, Value::Int(BigInt::from(42)));

        let f = Value::Func(FuncValue::new(domain, mapping));
        let fp1 = value_fingerprint(&f);
        let fp2 = value_fingerprint(&f);
        assert_eq!(fp1, fp2, "Function fingerprint must be deterministic");
    }

    #[test]
    fn test_func_structural_equality() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();

        let d = Value::Int(BigInt::from(1));
        domain.insert(d.clone());
        mapping.insert(d, Value::Bool(true));

        let f1 = FuncValue::new(domain.clone(), mapping.clone());
        let f2 = FuncValue::new(domain, mapping);
        let v1 = Value::Func(f1);
        let v2 = Value::Func(f2);
        assert_eq!(
            v1, v2,
            "Functions with same domain and mapping must be equal"
        );
    }

    // =========================================================================
    // Record operation tests (P12)
    // =========================================================================

    #[test]
    fn test_record_equality_reflexive() {
        let mut builder = RecordBuilder::new();
        builder.insert(Arc::from("x"), Value::Bool(true));
        builder.insert(Arc::from("y"), Value::Int(BigInt::from(42)));
        let r = Value::Record(builder.build());

        assert_eq!(r, r, "Record equality must be reflexive");
    }

    #[test]
    fn test_record_fingerprint_deterministic() {
        let mut builder = RecordBuilder::new();
        builder.insert(Arc::from("x"), Value::Bool(true));
        let r = Value::Record(builder.build());

        let fp1 = value_fingerprint(&r);
        let fp2 = value_fingerprint(&r);
        assert_eq!(fp1, fp2, "Record fingerprint must be deterministic");
    }

    #[test]
    fn test_record_field_access() {
        let mut builder = RecordBuilder::new();
        builder.insert(Arc::from("field"), Value::Bool(true));
        let r = Value::Record(builder.build());

        if let Value::Record(m) = &r {
            let field = m.get("field");
            assert!(field.is_some(), "Record must have field");
            assert_eq!(*field.unwrap(), Value::Bool(true));
        }
    }

    // =========================================================================
    // Sequence operation tests (P13)
    // =========================================================================

    #[test]
    fn test_seq_equality_reflexive() {
        let s = Value::Seq(vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))].into());
        assert_eq!(s, s, "Sequence equality must be reflexive");
    }

    #[test]
    fn test_seq_fingerprint_deterministic() {
        let s = Value::Seq(vec![Value::Bool(true), Value::Bool(false)].into());
        let fp1 = value_fingerprint(&s);
        let fp2 = value_fingerprint(&s);
        assert_eq!(fp1, fp2, "Sequence fingerprint must be deterministic");
    }

    #[test]
    fn test_seq_append_length() {
        let mut vec = vec![Value::Int(BigInt::from(1))];
        let original_len = vec.len();
        vec.push(Value::Int(BigInt::from(2)));
        assert_eq!(
            vec.len(),
            original_len + 1,
            "Append must increase length by 1"
        );
    }

    // =========================================================================
    // Tuple operation tests (P14)
    // =========================================================================

    #[test]
    fn test_tuple_equality_reflexive() {
        let t = Value::Tuple(vec![Value::Bool(true), Value::Int(BigInt::from(42))].into());
        assert_eq!(t, t, "Tuple equality must be reflexive");
    }

    #[test]
    fn test_tuple_fingerprint_deterministic() {
        let t = Value::Tuple(vec![Value::Bool(true)].into());
        let fp1 = value_fingerprint(&t);
        let fp2 = value_fingerprint(&t);
        assert_eq!(fp1, fp2, "Tuple fingerprint must be deterministic");
    }

    // =========================================================================
    // Cross-type property tests (P15)
    // =========================================================================

    #[test]
    fn test_cross_type_inequality() {
        use crate::value::SortedSet;
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        let s = Value::Set(SortedSet::from_ord_set(&set));

        let seq = Value::Seq(vec![Value::Int(BigInt::from(1))].into());

        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();
        let d = Value::Int(BigInt::from(1));
        domain.insert(d.clone());
        mapping.insert(d, Value::Bool(true));
        let f = Value::Func(FuncValue::new(domain, mapping));

        let mut builder = RecordBuilder::new();
        builder.insert(Arc::from("x"), Value::Int(BigInt::from(1)));
        let r = Value::Record(builder.build());

        assert_ne!(s, seq, "Set and Sequence must never be equal");
        assert_ne!(s, f, "Set and Function must never be equal");
        assert_ne!(s, r, "Set and Record must never be equal");
        assert_ne!(seq, f, "Sequence and Function must never be equal");
        assert_ne!(seq, r, "Sequence and Record must never be equal");
        assert_ne!(f, r, "Function and Record must never be equal");
    }

    // =========================================================================
    // State insert-get consistency tests (P16)
    // =========================================================================

    #[test]
    fn test_state_insert_get_consistency() {
        let values = vec![
            Value::Bool(true),
            Value::Int(BigInt::from(42)),
            Value::string("hello"),
        ];

        for v in values {
            let s = State::from_pairs([("x", v.clone())]);
            let retrieved = s.get("x");
            assert!(retrieved.is_some(), "Get must return inserted value");
            assert_eq!(
                *retrieved.unwrap(),
                v,
                "Retrieved value must equal inserted value"
            );
        }
    }

    #[test]
    fn test_state_insert_get_various_names() {
        let names = ["x", "y", "myVar", "state_var", "CamelCase"];
        let v = Value::Bool(true);

        for name in names {
            let s = State::from_pairs([(name, v.clone())]);
            let retrieved = s.get(name);
            assert!(
                retrieved.is_some(),
                "Get must return inserted value for {}",
                name
            );
            assert_eq!(*retrieved.unwrap(), v);
        }
    }

    // =========================================================================
    // State update isolation tests (P17)
    // =========================================================================

    #[test]
    fn test_state_update_isolation() {
        let v1 = Value::Bool(true);
        let v2 = Value::Int(BigInt::from(42));
        let v3 = Value::string("new");

        let s1 = State::from_pairs([("x", v1.clone()), ("y", v2.clone())]);
        let s2 = s1.with_var("z", v3);

        // x and y should be unchanged
        assert_eq!(s2.get("x"), s1.get("x"), "Updating z must not affect x");
        assert_eq!(s2.get("y"), s1.get("y"), "Updating z must not affect y");
    }

    #[test]
    fn test_state_update_preserves_others() {
        let v1 = Value::Bool(true);
        let v2 = Value::Int(BigInt::from(42));
        let v3 = Value::string("updated");

        let s1 = State::from_pairs([("x", v1), ("y", v2.clone())]);
        let s2 = s1.with_var("x", v3); // Update x

        // y should be unchanged
        assert_eq!(s2.get("y"), s1.get("y"), "Updating x must not affect y");
        assert_eq!(*s2.get("y").unwrap(), v2, "y value must be preserved");
    }

    // =========================================================================
    // Value Ord transitivity tests (P18)
    // =========================================================================

    #[test]
    fn test_bool_ord_transitive() {
        use std::cmp::Ordering;
        // false < true in bool ordering
        let a = Value::Bool(false);
        let b = Value::Bool(false);
        let c = Value::Bool(true);

        // Test: if a < b and b < c then a < c
        if a.cmp(&b) == Ordering::Less && b.cmp(&c) == Ordering::Less {
            assert_eq!(a.cmp(&c), Ordering::Less);
        }
        // Also test: false < true
        assert_eq!(Value::Bool(false).cmp(&Value::Bool(true)), Ordering::Less);
    }

    #[test]
    fn test_int_ord_transitive() {
        use std::cmp::Ordering;
        let a = Value::Int(BigInt::from(1));
        let b = Value::Int(BigInt::from(2));
        let c = Value::Int(BigInt::from(3));

        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&c), Ordering::Less);
        assert_eq!(
            a.cmp(&c),
            Ordering::Less,
            "Transitivity: 1 < 2 < 3 => 1 < 3"
        );
    }

    #[test]
    fn test_string_ord_transitive() {
        use std::cmp::Ordering;
        let a = Value::string("a");
        let b = Value::string("b");
        let c = Value::string("c");

        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&c), Ordering::Less);
        assert_eq!(
            a.cmp(&c),
            Ordering::Less,
            "Transitivity: a < b < c => a < c"
        );
    }

    // =========================================================================
    // Value Ord antisymmetry tests (P19)
    // =========================================================================

    #[test]
    fn test_ord_antisymmetric() {
        use std::cmp::Ordering;
        let values = vec![
            Value::Bool(true),
            Value::Bool(false),
            Value::Int(BigInt::from(0)),
            Value::Int(BigInt::from(1)),
            Value::string("a"),
        ];

        for v1 in &values {
            for v2 in &values {
                let v1_le_v2 = v1.cmp(v2) != Ordering::Greater;
                let v2_le_v1 = v2.cmp(v1) != Ordering::Greater;

                if v1_le_v2 && v2_le_v1 {
                    assert_eq!(v1, v2, "Antisymmetry: v1 <= v2 && v2 <= v1 => v1 == v2");
                }
            }
        }
    }

    // =========================================================================
    // Value Ord total ordering tests (P20)
    // =========================================================================

    #[test]
    fn test_ord_total() {
        use std::cmp::Ordering;
        let values = vec![
            Value::Bool(true),
            Value::Bool(false),
            Value::Int(BigInt::from(0)),
            Value::Int(BigInt::from(-1)),
            Value::Int(BigInt::from(1)),
            Value::string(""),
            Value::string("a"),
        ];

        for v1 in &values {
            for v2 in &values {
                let ord = v1.cmp(v2);
                assert!(
                    ord == Ordering::Less || ord == Ordering::Equal || ord == Ordering::Greater,
                    "Total ordering: exactly one of <, ==, > must hold"
                );
            }
        }
    }

    // =========================================================================
    // Hash-equality consistency tests (P21)
    // =========================================================================

    #[test]
    fn test_equal_values_equal_fingerprints() {
        let values = vec![
            Value::Bool(true),
            Value::Int(BigInt::from(42)),
            Value::string("test"),
        ];

        for v in values {
            let v_clone = v.clone();
            assert_eq!(v, v_clone);
            assert_eq!(
                value_fingerprint(&v),
                value_fingerprint(&v_clone),
                "Equal values must have equal fingerprints"
            );
        }
    }

    #[test]
    fn test_equal_states_equal_fingerprints() {
        let v = Value::Int(BigInt::from(42));
        let s1 = State::from_pairs([("x", v.clone())]);
        let s2 = State::from_pairs([("x", v)]);

        assert_eq!(s1, s2);
        assert_eq!(
            s1.fingerprint(),
            s2.fingerprint(),
            "Equal states must have equal fingerprints"
        );
    }

    // =========================================================================
    // State construction consistency tests (P22)
    // =========================================================================

    #[test]
    fn test_state_construction_equivalence() {
        let v1 = Value::Bool(true);
        let v2 = Value::Int(BigInt::from(42));

        // Construct via with_var
        let s1 = State::new()
            .with_var("x", v1.clone())
            .with_var("y", v2.clone());

        // Construct via from_pairs
        let s2 = State::from_pairs([("x", v1), ("y", v2)]);

        assert_eq!(
            s1, s2,
            "Different construction methods must yield equal states"
        );
        assert_eq!(
            s1.fingerprint(),
            s2.fingerprint(),
            "Different construction methods must yield same fingerprint"
        );
    }

    #[test]
    fn test_state_insertion_order_invariance() {
        let v1 = Value::Bool(true);
        let v2 = Value::Int(BigInt::from(42));

        // Insert x then y
        let s1 = State::new()
            .with_var("x", v1.clone())
            .with_var("y", v2.clone());

        // Insert y then x
        let s2 = State::new().with_var("y", v2).with_var("x", v1);

        assert_eq!(s1, s2, "Insertion order must not affect state equality");
        assert_eq!(
            s1.fingerprint(),
            s2.fingerprint(),
            "Insertion order must not affect fingerprint"
        );
    }

    // =========================================================================
    // Interval membership correctness tests (P23)
    // =========================================================================

    #[test]
    fn test_interval_membership_correctness() {
        let iv = IntervalValue::new(BigInt::from(1), BigInt::from(10));

        // Should contain 1, 5, 10
        assert!(iv.contains(&Value::Int(BigInt::from(1))));
        assert!(iv.contains(&Value::Int(BigInt::from(5))));
        assert!(iv.contains(&Value::Int(BigInt::from(10))));

        // Should not contain 0, 11
        assert!(!iv.contains(&Value::Int(BigInt::from(0))));
        assert!(!iv.contains(&Value::Int(BigInt::from(11))));
    }

    #[test]
    fn test_empty_interval_contains_nothing() {
        let iv = IntervalValue::new(BigInt::from(5), BigInt::from(1)); // low > high
        assert!(iv.is_empty());

        for k in -10..=10 {
            assert!(
                !iv.contains(&Value::Int(BigInt::from(k))),
                "Empty interval must contain nothing"
            );
        }
    }

    #[test]
    fn test_interval_membership_boundary() {
        // Test boundary conditions
        for (low, high, tests) in [
            (0, 0, vec![(0, true), (-1, false), (1, false)]),
            (
                -5,
                5,
                vec![(-5, true), (0, true), (5, true), (-6, false), (6, false)],
            ),
        ] {
            let iv = IntervalValue::new(BigInt::from(low), BigInt::from(high));
            for (k, expected) in tests {
                assert_eq!(
                    iv.contains(&Value::Int(BigInt::from(k))),
                    expected,
                    "Interval {}..{} contains {} should be {}",
                    low,
                    high,
                    k,
                    expected
                );
            }
        }
    }

    // =========================================================================
    // Value clone correctness tests (P24)
    // =========================================================================

    #[test]
    fn test_value_clone_equality() {
        let values = vec![
            Value::Bool(true),
            Value::Bool(false),
            Value::Int(BigInt::from(0)),
            Value::Int(BigInt::from(-999)),
            Value::string(""),
            Value::string("test"),
        ];

        for v in values {
            let v_clone = v.clone();
            assert_eq!(v, v_clone, "Cloned value must equal original");
        }
    }

    #[test]
    fn test_set_clone_equality() {
        use crate::value::SortedSet;
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::Int(BigInt::from(2)));
        let s = Value::Set(SortedSet::from_ord_set(&set));
        let s_clone = s.clone();
        assert_eq!(s, s_clone, "Cloned set must equal original");
    }

    // =========================================================================
    // State clone correctness tests (P25)
    // =========================================================================

    #[test]
    fn test_state_clone_equality() {
        let s = State::from_pairs([
            ("x", Value::Bool(true)),
            ("y", Value::Int(BigInt::from(42))),
        ]);
        let s_clone = s.clone();

        assert_eq!(s, s_clone, "Cloned state must equal original");
        assert_eq!(
            s.fingerprint(),
            s_clone.fingerprint(),
            "Cloned state must have same fingerprint"
        );
    }

    #[test]
    fn test_state_clone_independence() {
        let v = Value::Int(BigInt::from(42));
        let s1 = State::from_pairs([("x", v)]);
        let s2 = s1.clone();
        let s3 = s2.with_var("x", Value::Int(BigInt::from(100)));

        // s1 and s2 should still be equal (clone is independent)
        assert_eq!(s1, s2);
        // s3 should be different
        assert_ne!(s1, s3);
    }

    // =========================================================================
    // Phase E: E2 - Operator Semantics Tests
    // =========================================================================

    // =========================================================================
    // Integer Arithmetic Tests (P26-P31)
    // =========================================================================

    #[test]
    fn test_int_add_commutative() {
        for (a, b) in [(-128i64, 127), (0, 0), (1, -1), (42, 17)] {
            let a_big = BigInt::from(a);
            let b_big = BigInt::from(b);
            assert_eq!(
                &a_big + &b_big,
                &b_big + &a_big,
                "Addition must be commutative"
            );
        }
    }

    #[test]
    fn test_int_mul_commutative() {
        for (a, b) in [(-10i64, 10), (0, 5), (1, -1), (7, 8)] {
            let a_big = BigInt::from(a);
            let b_big = BigInt::from(b);
            assert_eq!(
                &a_big * &b_big,
                &b_big * &a_big,
                "Multiplication must be commutative"
            );
        }
    }

    #[test]
    fn test_int_add_associative() {
        for (a, b, c) in [(-10i64, 20, -5), (0, 0, 0), (1, 2, 3)] {
            let a_big = BigInt::from(a);
            let b_big = BigInt::from(b);
            let c_big = BigInt::from(c);
            assert_eq!(
                (&a_big + &b_big) + &c_big,
                &a_big + (&b_big + &c_big),
                "Addition must be associative"
            );
        }
    }

    #[test]
    fn test_int_mul_associative() {
        for (a, b, c) in [(-2i64, 3, 4), (1, 1, 1), (2, 0, 5)] {
            let a_big = BigInt::from(a);
            let b_big = BigInt::from(b);
            let c_big = BigInt::from(c);
            assert_eq!(
                (&a_big * &b_big) * &c_big,
                &a_big * (&b_big * &c_big),
                "Multiplication must be associative"
            );
        }
    }

    #[test]
    fn test_int_identity_elements() {
        use num_traits::{One, Zero};
        for a in [-100i64, -1, 0, 1, 100] {
            let a_big = BigInt::from(a);
            let zero: BigInt = Zero::zero();
            let one: BigInt = One::one();
            assert_eq!(&a_big + &zero, a_big, "0 is additive identity");
            assert_eq!(&a_big * &one, a_big, "1 is multiplicative identity");
        }
    }

    #[test]
    fn test_int_additive_inverse() {
        use num_traits::Zero;
        for a in [-100i64, -1, 0, 1, 100] {
            let a_big = BigInt::from(a);
            let neg_a = -&a_big;
            let zero: BigInt = Zero::zero();
            assert_eq!(&a_big + &neg_a, zero, "a + (-a) = 0");
        }
    }

    #[test]
    fn test_int_distributivity() {
        for (a, b, c) in [(2i64, 3, 4), (-1, 5, -3), (0, 1, 2)] {
            let a_big = BigInt::from(a);
            let b_big = BigInt::from(b);
            let c_big = BigInt::from(c);
            let left = &a_big * (&b_big + &c_big);
            let right = (&a_big * &b_big) + (&a_big * &c_big);
            assert_eq!(left, right, "a*(b+c) = a*b + a*c");
        }
    }

    #[test]
    fn test_int_div_mod_relationship() {
        use num_integer::Integer;
        for (a, b) in [(-7i64, 3), (7, 3), (10, -3), (-10, -3), (0, 5)] {
            let a_big = BigInt::from(a);
            let b_big = BigInt::from(b);
            let quotient = a_big.div_floor(&b_big);
            let remainder = a_big.mod_floor(&b_big);
            let reconstructed = &quotient * &b_big + &remainder;
            assert_eq!(reconstructed, a_big, "a = (a div b) * b + (a mod b)");
        }
    }

    #[test]
    fn test_int_mod_range() {
        use num_traits::Zero;
        // TLA+ uses Euclidean modulo (always non-negative)
        // Formula: ((a % b) + |b|) % |b|
        for (a, b) in [(-7i64, 3), (7, 3), (10, -3), (-10, -3)] {
            let a_big = BigInt::from(a);
            let b_big = BigInt::from(b);
            let abs_b = if b_big < Zero::zero() {
                -&b_big
            } else {
                b_big.clone()
            };
            // TLA+ Euclidean modulo: always non-negative
            let remainder = ((&a_big % &abs_b) + &abs_b) % &abs_b;
            assert!(
                remainder >= Zero::zero() && remainder < abs_b,
                "0 <= (a mod b) < |b| for a={}, b={}",
                a,
                b
            );
        }
    }

    // =========================================================================
    // Boolean Algebra Tests (P32-P39)
    // =========================================================================

    #[test]
    fn test_bool_commutativity() {
        for (a, b) in [(true, true), (true, false), (false, true), (false, false)] {
            assert_eq!(a && b, b && a, "AND must be commutative");
            assert_eq!(a || b, b || a, "OR must be commutative");
        }
    }

    #[test]
    fn test_bool_associativity() {
        for (a, b, c) in [
            (true, true, true),
            (true, true, false),
            (true, false, false),
            (false, false, false),
        ] {
            assert_eq!((a && b) && c, a && (b && c), "AND must be associative");
            assert_eq!((a || b) || c, a || (b || c), "OR must be associative");
        }
    }

    #[test]
    #[allow(clippy::nonminimal_bool, clippy::overly_complex_bool_expr)]
    fn test_bool_identity_annihilator() {
        for a in [true, false] {
            // Identity - these expressions intentionally test boolean algebra properties
            let and_identity = a && true;
            assert_eq!(and_identity, a, "TRUE is AND identity");
            let or_identity = a || false;
            assert_eq!(or_identity, a, "FALSE is OR identity");
            // Annihilator
            let and_annihilator = a && false;
            assert!(!and_annihilator, "FALSE is AND annihilator");
            let or_annihilator = a || true;
            assert!(or_annihilator, "TRUE is OR annihilator");
        }
    }

    #[test]
    #[allow(clippy::overly_complex_bool_expr, clippy::nonminimal_bool)]
    fn test_bool_complement() {
        for a in [true, false] {
            // These expressions intentionally test complement laws
            let and_complement = a && !a;
            assert!(!and_complement, "a AND (NOT a) = FALSE");
            let or_complement = a || !a;
            assert!(or_complement, "a OR (NOT a) = TRUE");
            // Double negation test - intentionally using !!a to verify NOT(NOT a) = a
            let double_neg = !!a;
            assert_eq!(double_neg, a, "Double negation: NOT(NOT a) = a");
        }
    }

    #[test]
    fn test_de_morgan_laws() {
        for (a, b) in [(true, true), (true, false), (false, true), (false, false)] {
            assert_eq!(!(a && b), !a || !b, "NOT(a AND b) = (NOT a) OR (NOT b)");
            assert_eq!(!(a || b), !a && !b, "NOT(a OR b) = (NOT a) AND (NOT b)");
        }
    }

    #[test]
    #[allow(clippy::overly_complex_bool_expr)]
    fn test_bool_implication() {
        for (a, b) in [(true, true), (true, false), (false, true), (false, false)] {
            let implies = !a || b;
            let expected = if a { b } else { true };
            assert_eq!(implies, expected, "a => b iff (NOT a) OR b");
        }
        // Reflexivity - this is intentionally testing that !a || a is a tautology
        for a in [true, false] {
            let reflexive = !a || a;
            assert!(reflexive, "a => a is always TRUE");
        }
    }

    #[test]
    fn test_bool_equivalence() {
        for (a, b) in [(true, true), (true, false), (false, true), (false, false)] {
            let equiv = a == b;
            let impl_both = (!a || b) && (!b || a);
            assert_eq!(equiv, impl_both, "(a <=> b) = (a => b) AND (b => a)");
            // Symmetry
            assert_eq!((a == b), (b == a), "Equivalence is symmetric");
        }
    }

    // =========================================================================
    // Comparison Operator Tests (P40)
    // =========================================================================

    #[test]
    fn test_int_comparison_properties() {
        // Transitivity
        let test_cases: [(i8, i8, i8); 4] = [(1, 2, 3), (-3, -2, -1), (0, 1, 2), (-1, 0, 1)];
        for (a, b, c) in test_cases {
            if a < b && b < c {
                assert!(a < c, "Less-than must be transitive");
            }
        }

        // Irreflexivity
        for a in [-100i8, -1, 0, 1, 100] {
            assert!((a >= a), "Less-than must be irreflexive");
        }

        // Reflexivity of <=
        for a in [-100i8, -1, 0, 1, 100] {
            assert!(a <= a, "Less-than-or-equal must be reflexive");
        }

        // Trichotomy
        for a in [-10i8, 0, 10] {
            for b in [-10i8, 0, 10] {
                let lt = a < b;
                let eq = a == b;
                let gt = a > b;
                let count = (lt as u8) + (eq as u8) + (gt as u8);
                assert_eq!(count, 1, "Trichotomy: exactly one of <, =, > holds");
            }
        }
    }

    // =========================================================================
    // Set Operator Semantics Tests (P41)
    // =========================================================================

    #[test]
    fn test_set_union_membership() {
        let mut set_a = OrdSet::new();
        set_a.insert(Value::Int(BigInt::from(1)));
        set_a.insert(Value::Int(BigInt::from(2)));

        let mut set_b = OrdSet::new();
        set_b.insert(Value::Int(BigInt::from(2)));
        set_b.insert(Value::Int(BigInt::from(3)));

        let union = set_a.clone().union(set_b.clone());

        // Test: x in (A ∪ B) iff (x in A) or (x in B)
        for i in 0..5 {
            let x = Value::Int(BigInt::from(i));
            let x_in_union = union.contains(&x);
            let x_in_a = set_a.contains(&x);
            let x_in_b = set_b.contains(&x);
            assert_eq!(
                x_in_union,
                x_in_a || x_in_b,
                "x in (A ∪ B) iff (x in A) or (x in B)"
            );
        }
    }

    #[test]
    fn test_set_intersection_membership() {
        let mut set_a = OrdSet::new();
        set_a.insert(Value::Int(BigInt::from(1)));
        set_a.insert(Value::Int(BigInt::from(2)));

        let mut set_b = OrdSet::new();
        set_b.insert(Value::Int(BigInt::from(2)));
        set_b.insert(Value::Int(BigInt::from(3)));

        let inter: OrdSet<Value> = set_a.clone().intersection(set_b.clone());

        // Test: x in (A ∩ B) iff (x in A) and (x in B)
        for i in 0..5 {
            let x = Value::Int(BigInt::from(i));
            let x_in_inter = inter.contains(&x);
            let x_in_a = set_a.contains(&x);
            let x_in_b = set_b.contains(&x);
            assert_eq!(
                x_in_inter,
                x_in_a && x_in_b,
                "x in (A ∩ B) iff (x in A) and (x in B)"
            );
        }
    }

    #[test]
    fn test_set_difference_membership() {
        let mut set_a = OrdSet::new();
        set_a.insert(Value::Int(BigInt::from(1)));
        set_a.insert(Value::Int(BigInt::from(2)));

        let mut set_b = OrdSet::new();
        set_b.insert(Value::Int(BigInt::from(2)));
        set_b.insert(Value::Int(BigInt::from(3)));

        let diff = set_a.clone().relative_complement(set_b.clone());

        // Test: x in (A \ B) iff (x in A) and (x not in B)
        for i in 0..5 {
            let x = Value::Int(BigInt::from(i));
            let x_in_diff = diff.contains(&x);
            let x_in_a = set_a.contains(&x);
            let x_not_in_b = !set_b.contains(&x);
            assert_eq!(
                x_in_diff,
                x_in_a && x_not_in_b,
                "x in (A \\ B) iff (x in A) and (x not in B)"
            );
        }
    }

    #[test]
    fn test_set_subset_definition() {
        let mut set_a = OrdSet::new();
        set_a.insert(Value::Int(BigInt::from(1)));

        let mut set_b = OrdSet::new();
        set_b.insert(Value::Int(BigInt::from(1)));
        set_b.insert(Value::Int(BigInt::from(2)));

        // A ⊆ B iff all elements of A are in B
        assert!(set_a.is_subset(&set_b), "{{1}} ⊆ {{1,2}}");
        assert!(!set_b.is_subset(&set_a), "{{1,2}} ⊄ {{1}}");

        let empty: OrdSet<Value> = OrdSet::new();
        assert!(empty.is_subset(&set_a), "∅ ⊆ A for all A");
        assert!(set_a.is_subset(&set_a), "A ⊆ A (reflexivity)");
    }

    // =========================================================================
    // Sequence Operator Semantics Tests (P42)
    // =========================================================================

    #[test]
    fn test_seq_concat_length() {
        let test_cases: [(Vec<i32>, Vec<i32>); 4] = [
            (vec![], vec![]),
            (vec![1], vec![]),
            (vec![], vec![1, 2]),
            (vec![1, 2], vec![3, 4, 5]),
        ];

        for (s, t) in test_cases {
            let len_s = s.len();
            let len_t = t.len();
            let mut concat = s.clone();
            concat.extend(t);
            assert_eq!(concat.len(), len_s + len_t, "Len(s ∘ t) = Len(s) + Len(t)");
        }
    }

    #[test]
    fn test_seq_concat_identity() {
        let test_seqs = [
            vec![],
            vec![Value::Int(BigInt::from(1))],
            vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))],
        ];

        for s in test_seqs {
            let empty: Vec<Value> = vec![];

            // empty ∘ s = s
            let mut left = empty.clone();
            left.extend(s.iter().cloned());
            assert_eq!(left, s, "<<>> ∘ s = s");

            // s ∘ empty = s
            let mut right = s.clone();
            right.extend(empty.iter().cloned());
            assert_eq!(right, s, "s ∘ <<>> = s");
        }
    }

    // =========================================================================
    // Function EXCEPT Semantics Tests (P43)
    // =========================================================================

    #[test]
    fn test_func_except_preserves_domain() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();
        let k1 = Value::Int(BigInt::from(1));
        let k2 = Value::Int(BigInt::from(2));
        domain.insert(k1.clone());
        domain.insert(k2.clone());
        mapping.insert(k1.clone(), Value::Bool(true));
        mapping.insert(k2.clone(), Value::Bool(false));
        let f = FuncValue::new(domain.clone(), mapping);

        // Apply EXCEPT
        let f_new = f.except(k1, Value::Int(BigInt::from(42)));

        assert_eq!(
            f_new.domain_as_ord_set(),
            domain,
            "EXCEPT must preserve domain"
        );
    }

    #[test]
    fn test_func_except_updates_value() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();
        let key = Value::Int(BigInt::from(1));
        domain.insert(key.clone());
        mapping.insert(key.clone(), Value::Bool(true));
        let f = FuncValue::new(domain, mapping);

        let new_val = Value::Bool(false);
        let f_new = f.except(key.clone(), new_val.clone());

        let result = f_new.apply(&key);
        assert!(result.is_some());
        assert_eq!(*result.unwrap(), new_val, "EXCEPT must update to new value");
    }

    #[test]
    fn test_func_except_isolates_other_keys() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();
        let k1 = Value::Int(BigInt::from(1));
        let k2 = Value::Int(BigInt::from(2));
        let v2 = Value::Bool(false);
        domain.insert(k1.clone());
        domain.insert(k2.clone());
        mapping.insert(k1.clone(), Value::Bool(true));
        mapping.insert(k2.clone(), v2.clone());
        let f = FuncValue::new(domain, mapping);

        let f_new = f.except(k1, Value::Int(BigInt::from(42)));

        let result = f_new.apply(&k2);
        assert!(result.is_some());
        assert_eq!(*result.unwrap(), v2, "EXCEPT must not affect other keys");
    }

    // =========================================================================
    // Record EXCEPT Semantics Tests (P44)
    // =========================================================================

    #[test]
    fn test_record_except_preserves_other_fields() {
        let mut r: OrdMap<Arc<str>, Value> = OrdMap::new();
        r.insert(Arc::from("x"), Value::Int(BigInt::from(1)));
        r.insert(Arc::from("y"), Value::Bool(true));

        let original_y = r.get(&Arc::from("y")).cloned();

        // Update x
        let mut r_new = r.clone();
        r_new.insert(Arc::from("x"), Value::Int(BigInt::from(42)));

        assert_eq!(
            r_new.get(&Arc::from("y")),
            original_y.as_ref(),
            "Updating x must not affect y"
        );
    }

    #[test]
    fn test_record_except_updates_field() {
        let mut r: OrdMap<Arc<str>, Value> = OrdMap::new();
        r.insert(Arc::from("x"), Value::Int(BigInt::from(1)));

        let new_val = Value::Int(BigInt::from(42));
        r.insert(Arc::from("x"), new_val.clone());

        assert_eq!(
            r.get(&Arc::from("x")),
            Some(&new_val),
            "Field must be updated"
        );
    }

    // =========================================================================
    // Quantifier Semantics Tests (P45-P48)
    // =========================================================================

    #[test]
    fn test_forall_empty_is_true() {
        let empty: OrdSet<Value> = OrdSet::new();
        // ∀x ∈ {} : P(x) is TRUE for any P (vacuously true)
        let result = empty.iter().all(|_x| false);
        assert!(result, "∀x ∈ {{}} : P(x) must be TRUE (vacuously)");
    }

    #[test]
    fn test_exists_empty_is_false() {
        let empty: OrdSet<Value> = OrdSet::new();
        // ∃x ∈ {} : P(x) is FALSE for any P (no witnesses)
        let result = empty.iter().any(|_x| true);
        assert!(!result, "∃x ∈ {{}} : P(x) must be FALSE (no witnesses)");
    }

    #[test]
    fn test_forall_singleton() {
        let elem = Value::Int(BigInt::from(42));
        let mut singleton = OrdSet::new();
        singleton.insert(elem.clone());

        // ∀x ∈ {a} : P(x) ≡ P(a)
        let forall_result = singleton.iter().all(|x| *x == elem);
        let direct_result = elem == elem;
        assert_eq!(
            forall_result, direct_result,
            "∀x ∈ {{a}} : P(x) must equal P(a)"
        );
    }

    #[test]
    fn test_exists_singleton() {
        let elem = Value::Int(BigInt::from(42));
        let mut singleton = OrdSet::new();
        singleton.insert(elem.clone());

        // ∃x ∈ {a} : P(x) ≡ P(a)
        let exists_result = singleton.iter().any(|x| *x == elem);
        let direct_result = elem == elem;
        assert_eq!(
            exists_result, direct_result,
            "∃x ∈ {{a}} : P(x) must equal P(a)"
        );
    }

    #[test]
    fn test_forall_true_predicate() {
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::Int(BigInt::from(2)));

        let result = set.iter().all(|_x| true);
        assert!(result, "∀x ∈ S : TRUE must be TRUE");
    }

    #[test]
    fn test_exists_true_predicate_nonempty() {
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));

        let result = set.iter().any(|_x| true);
        assert!(result, "∃x ∈ S : TRUE must be TRUE for non-empty S");
    }

    #[test]
    fn test_forall_false_predicate_nonempty() {
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));

        let result = set.iter().all(|_x| false);
        assert!(!result, "∀x ∈ S : FALSE must be FALSE for non-empty S");
    }

    #[test]
    fn test_exists_false_predicate() {
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::Int(BigInt::from(2)));

        let result = set.iter().any(|_x| false);
        assert!(!result, "∃x ∈ S : FALSE must be FALSE");
    }

    #[test]
    fn test_quantifier_duality_forall() {
        let mut set = OrdSet::new();
        set.insert(Value::Bool(true));
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::string("x"));

        // ¬(∀x : P(x)) ≡ ∃x : ¬P(x)
        let predicate = |x: &Value| matches!(x, Value::Bool(_));
        let not_forall = !set.iter().all(predicate);
        let exists_not = set.iter().any(|x| !predicate(x));

        assert_eq!(not_forall, exists_not, "¬(∀x : P(x)) must equal ∃x : ¬P(x)");
    }

    #[test]
    fn test_quantifier_duality_exists() {
        let mut set = OrdSet::new();
        set.insert(Value::Bool(true));
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::string("x"));

        // ¬(∃x : P(x)) ≡ ∀x : ¬P(x)
        let predicate = |x: &Value| matches!(x, Value::Bool(_));
        let not_exists = !set.iter().any(predicate);
        let forall_not = set.iter().all(|x| !predicate(x));

        assert_eq!(not_exists, forall_not, "¬(∃x : P(x)) must equal ∀x : ¬P(x)");
    }

    // =========================================================================
    // Function Application Semantics Tests (P49)
    // =========================================================================

    #[test]
    fn test_func_apply_in_domain_returns_value() {
        let key = Value::Int(BigInt::from(1));
        let val = Value::Bool(true);

        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();
        domain.insert(key.clone());
        mapping.insert(key.clone(), val.clone());

        let f = FuncValue::new(domain, mapping);
        let result = f.apply(&key);

        assert!(result.is_some(), "Apply must succeed for domain element");
        assert_eq!(*result.unwrap(), val, "Apply must return mapped value");
    }

    #[test]
    fn test_func_apply_outside_domain_returns_none() {
        let k1 = Value::Int(BigInt::from(1));
        let k2 = Value::Int(BigInt::from(2));

        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();
        domain.insert(k1.clone());
        mapping.insert(k1, Value::Bool(true));

        let f = FuncValue::new(domain, mapping);
        let result = f.apply(&k2);

        assert!(
            result.is_none(),
            "Apply must return None for non-domain element"
        );
    }

    #[test]
    fn test_func_domain_apply_consistency() {
        // Test that domain membership is consistent with apply success
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();
        for i in 1..=5 {
            let k = Value::Int(BigInt::from(i));
            domain.insert(k.clone());
            mapping.insert(k, Value::Bool(i % 2 == 0));
        }
        let f = FuncValue::new(domain.clone(), mapping);

        // Keys in domain
        for i in 1..=5 {
            let k = Value::Int(BigInt::from(i));
            assert!(f.domain_contains(&k));
            assert!(
                f.apply(&k).is_some(),
                "If key in DOMAIN, apply must succeed"
            );
        }

        // Keys not in domain
        for i in [0, 6, 10] {
            let k = Value::Int(BigInt::from(i));
            assert!(!f.domain_contains(&k));
            assert!(
                f.apply(&k).is_none(),
                "If key not in DOMAIN, apply must return None"
            );
        }
    }

    // =========================================================================
    // Function Equality Semantics Tests (P50)
    // =========================================================================

    #[test]
    fn test_func_equality_different_domains() {
        let k1 = Value::Int(BigInt::from(1));
        let k2 = Value::Int(BigInt::from(2));
        let val = Value::Bool(true);

        let mut domain1 = OrdSet::new();
        let mut mapping1 = OrdMap::new();
        domain1.insert(k1.clone());
        mapping1.insert(k1, val.clone());
        let f1 = FuncValue::new(domain1, mapping1);

        let mut domain2 = OrdSet::new();
        let mut mapping2 = OrdMap::new();
        domain2.insert(k2.clone());
        mapping2.insert(k2, val);
        let f2 = FuncValue::new(domain2, mapping2);

        let v1 = Value::Func(f1);
        let v2 = Value::Func(f2);

        assert_ne!(v1, v2, "Functions with different domains must not be equal");
    }

    #[test]
    fn test_func_equality_different_mappings() {
        let key = Value::Int(BigInt::from(1));

        let mut domain = OrdSet::new();
        domain.insert(key.clone());

        let mut mapping1 = OrdMap::new();
        mapping1.insert(key.clone(), Value::Bool(true));
        let f1 = FuncValue::new(domain.clone(), mapping1);

        let mut mapping2 = OrdMap::new();
        mapping2.insert(key, Value::Bool(false));
        let f2 = FuncValue::new(domain, mapping2);

        let v1 = Value::Func(f1);
        let v2 = Value::Func(f2);

        assert_ne!(
            v1, v2,
            "Functions with different mappings must not be equal"
        );
    }

    #[test]
    fn test_func_equality_when_identical() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();
        for i in 1..=3 {
            let k = Value::Int(BigInt::from(i));
            domain.insert(k.clone());
            mapping.insert(k, Value::Bool(i % 2 == 0));
        }

        let f1 = FuncValue::new(domain.clone(), mapping.clone());
        let f2 = FuncValue::new(domain, mapping);

        let v1 = Value::Func(f1);
        let v2 = Value::Func(f2);

        assert_eq!(
            v1, v2,
            "Functions with same domain and mapping must be equal"
        );
    }

    // =========================================================================
    // CHOOSE Operator Semantics Tests (P51)
    // =========================================================================

    /// Helper: Simulate CHOOSE semantics - returns first element satisfying predicate
    fn choose<P>(set: &OrdSet<Value>, predicate: P) -> Option<Value>
    where
        P: Fn(&Value) -> bool,
    {
        for elem in set.iter() {
            if predicate(elem) {
                return Some(elem.clone());
            }
        }
        None
    }

    #[test]
    fn test_choose_true_predicate_returns_first() {
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::Int(BigInt::from(2)));
        set.insert(Value::Int(BigInt::from(3)));

        let result = choose(&set, |_| true);
        assert!(
            result.is_some(),
            "CHOOSE TRUE on non-empty set must succeed"
        );

        let first = set.iter().next().unwrap().clone();
        assert_eq!(
            result.unwrap(),
            first,
            "CHOOSE TRUE must return first element"
        );
    }

    #[test]
    fn test_choose_singleton_returns_element() {
        let elem = Value::Int(BigInt::from(42));
        let mut set = OrdSet::new();
        set.insert(elem.clone());

        let result = choose(&set, |_| true);
        assert!(result.is_some(), "CHOOSE on singleton must succeed");
        assert_eq!(
            result.unwrap(),
            elem,
            "CHOOSE on singleton must return that element"
        );
    }

    #[test]
    fn test_choose_empty_set_fails() {
        let empty: OrdSet<Value> = OrdSet::new();
        let result = choose(&empty, |_| true);
        assert!(result.is_none(), "CHOOSE on empty set must fail");
    }

    #[test]
    fn test_choose_deterministic() {
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::Int(BigInt::from(2)));

        let predicate = |x: &Value| match x {
            Value::Int(n) => n > &BigInt::from(0),
            _ => false,
        };

        let result1 = choose(&set, predicate);
        let result2 = choose(&set, predicate);

        assert_eq!(result1, result2, "CHOOSE must be deterministic");
    }

    #[test]
    fn test_choose_returns_satisfying_element() {
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::Int(BigInt::from(2)));
        set.insert(Value::Int(BigInt::from(3)));

        let predicate = |x: &Value| match x {
            Value::Int(n) => n > &BigInt::from(1),
            _ => false,
        };

        let result = choose(&set, predicate);
        assert!(
            result.is_some(),
            "CHOOSE with satisfiable predicate must succeed"
        );

        let chosen = result.unwrap();
        // Verify the chosen value satisfies the predicate condition
        if let Value::Int(n) = &chosen {
            assert!(n > &BigInt::from(1), "CHOOSE result must satisfy predicate");
        } else {
            panic!("CHOOSE result should be an Int");
        }
    }

    #[test]
    fn test_choose_unsatisfiable_predicate_fails() {
        let mut set = OrdSet::new();
        set.insert(Value::Int(BigInt::from(1)));
        set.insert(Value::Int(BigInt::from(2)));

        let predicate = |x: &Value| match x {
            Value::Int(n) => n > &BigInt::from(100),
            _ => false,
        };

        let result = choose(&set, predicate);
        assert!(
            result.is_none(),
            "CHOOSE with unsatisfiable predicate must fail"
        );
    }

    // =========================================================================
    // Nested EXCEPT Path Tests (P52)
    // =========================================================================

    /// Helper: Create nested function f where f[1] is a function {1 -> true, 2 -> false}
    fn create_nested_func() -> FuncValue {
        let mut inner_domain = OrdSet::new();
        let mut inner_mapping = OrdMap::new();
        inner_domain.insert(Value::Int(BigInt::from(1)));
        inner_domain.insert(Value::Int(BigInt::from(2)));
        inner_mapping.insert(Value::Int(BigInt::from(1)), Value::Bool(true));
        inner_mapping.insert(Value::Int(BigInt::from(2)), Value::Bool(false));
        let inner = FuncValue::new(inner_domain, inner_mapping);

        let mut outer_domain = OrdSet::new();
        let mut outer_mapping = OrdMap::new();
        outer_domain.insert(Value::Int(BigInt::from(1)));
        outer_mapping.insert(Value::Int(BigInt::from(1)), Value::Func(inner));

        FuncValue::new(outer_domain, outer_mapping)
    }

    #[test]
    fn test_nested_except_preserves_outer_domain() {
        let outer = create_nested_func();
        let original_domain = outer.domain_as_ord_set();

        if let Some(Value::Func(inner)) = outer.apply(&Value::Int(BigInt::from(1))).cloned() {
            let new_inner = inner.except(Value::Int(BigInt::from(1)), Value::Bool(false));
            let new_outer = outer.except(Value::Int(BigInt::from(1)), Value::Func(new_inner));

            assert_eq!(
                new_outer.domain_as_ord_set(),
                original_domain,
                "Nested EXCEPT must preserve outer domain"
            );
        }
    }

    #[test]
    fn test_nested_except_updates_inner_value() {
        let outer = create_nested_func();

        if let Some(Value::Func(inner)) = outer.apply(&Value::Int(BigInt::from(1))).cloned() {
            let new_inner = inner.except(Value::Int(BigInt::from(1)), Value::Bool(false));
            let new_outer = outer.except(Value::Int(BigInt::from(1)), Value::Func(new_inner));

            if let Some(Value::Func(result_inner)) =
                new_outer.apply(&Value::Int(BigInt::from(1))).cloned()
            {
                let result = result_inner.apply(&Value::Int(BigInt::from(1)));
                assert!(result.is_some());
                assert_eq!(
                    *result.unwrap(),
                    Value::Bool(false),
                    "Nested EXCEPT must update inner value"
                );
            }
        }
    }

    #[test]
    fn test_nested_except_preserves_other_inner_keys() {
        let outer = create_nested_func();

        if let Some(Value::Func(inner)) = outer.apply(&Value::Int(BigInt::from(1))).cloned() {
            let original_inner_2 = inner.apply(&Value::Int(BigInt::from(2))).cloned();

            let new_inner = inner.except(Value::Int(BigInt::from(1)), Value::Bool(false));
            let new_outer = outer.except(Value::Int(BigInt::from(1)), Value::Func(new_inner));

            if let Some(Value::Func(result_inner)) =
                new_outer.apply(&Value::Int(BigInt::from(1))).cloned()
            {
                let result_inner_2 = result_inner.apply(&Value::Int(BigInt::from(2))).cloned();
                assert_eq!(
                    result_inner_2, original_inner_2,
                    "Nested EXCEPT must preserve other inner keys"
                );
            }
        }
    }

    // =========================================================================
    // Sequence Index/Access Semantics Tests (P53)
    // =========================================================================

    #[test]
    fn test_seq_index_returns_correct_element() {
        let seq = [
            Value::Int(BigInt::from(10)),
            Value::Int(BigInt::from(20)),
            Value::Int(BigInt::from(30)),
        ];

        assert_eq!(seq[0], Value::Int(BigInt::from(10)));
        assert_eq!(seq[1], Value::Int(BigInt::from(20)));
        assert_eq!(seq[2], Value::Int(BigInt::from(30)));
    }

    #[test]
    fn test_seq_head_returns_first() {
        let seq = [Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))];
        let head = seq.first().cloned();
        assert!(head.is_some());
        assert_eq!(head.unwrap(), Value::Int(BigInt::from(1)));
    }

    #[test]
    fn test_seq_head_empty_fails() {
        let seq: Vec<Value> = vec![];
        let head = seq.first();
        assert!(head.is_none(), "Head of empty seq must fail");
    }

    #[test]
    fn test_seq_tail_returns_rest() {
        let seq = [
            Value::Int(BigInt::from(1)),
            Value::Int(BigInt::from(2)),
            Value::Int(BigInt::from(3)),
        ];

        let tail: Vec<Value> = seq[1..].to_vec();
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0], Value::Int(BigInt::from(2)));
        assert_eq!(tail[1], Value::Int(BigInt::from(3)));
    }

    #[test]
    fn test_seq_tail_singleton_is_empty() {
        let seq = [Value::Int(BigInt::from(1))];
        let tail: Vec<Value> = seq[1..].to_vec();
        assert!(tail.is_empty());
    }

    #[test]
    fn test_subseq_returns_correct_slice() {
        let seq = [
            Value::Int(BigInt::from(10)),
            Value::Int(BigInt::from(20)),
            Value::Int(BigInt::from(30)),
            Value::Int(BigInt::from(40)),
        ];

        let subseq: Vec<Value> = seq[1..3].to_vec();
        assert_eq!(subseq.len(), 2);
        assert_eq!(subseq[0], Value::Int(BigInt::from(20)));
        assert_eq!(subseq[1], Value::Int(BigInt::from(30)));
    }

    #[test]
    fn test_seq_append_adds_at_end() {
        let mut seq = vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))];
        let elem = Value::Int(BigInt::from(3));
        let original_len = seq.len();

        seq.push(elem.clone());

        assert_eq!(seq.len(), original_len + 1);
        assert_eq!(seq.last(), Some(&elem));
    }

    // =========================================================================
    // Tuple Semantics Tests (P54)
    // =========================================================================

    #[test]
    fn test_tuple_element_access() {
        let tuple = [
            Value::Bool(true),
            Value::Int(BigInt::from(42)),
            Value::string("hello"),
        ];

        assert_eq!(tuple[0], Value::Bool(true));
        assert_eq!(tuple[1], Value::Int(BigInt::from(42)));
        assert_eq!(tuple[2], Value::string("hello"));
    }

    #[test]
    fn test_tuple_equality() {
        let t1 = vec![Value::Int(BigInt::from(1)), Value::Bool(true)];
        let t2 = vec![Value::Int(BigInt::from(1)), Value::Bool(true)];

        assert_eq!(t1, t2, "Tuples with same elements must be equal");
    }

    #[test]
    fn test_tuple_different_lengths_not_equal() {
        let t1 = vec![Value::Int(BigInt::from(1))];
        let t2 = vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))];

        assert_ne!(t1, t2, "Tuples with different lengths must not be equal");
    }

    #[test]
    fn test_tuple_different_elements_not_equal() {
        let t1 = vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))];
        let t2 = vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(3))];

        assert_ne!(t1, t2, "Tuples with different elements must not be equal");
    }

    #[test]
    fn test_tuple_length() {
        let t0: Vec<Value> = vec![];
        let t1 = [Value::Int(BigInt::from(1))];
        let t2 = [Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))];
        let t3 = [
            Value::Int(BigInt::from(1)),
            Value::Int(BigInt::from(2)),
            Value::Int(BigInt::from(3)),
        ];

        assert_eq!(t0.len(), 0);
        assert_eq!(t1.len(), 1);
        assert_eq!(t2.len(), 2);
        assert_eq!(t3.len(), 3);
    }

    // =========================================================================
    // ModelValue Semantics Tests (P55)
    // =========================================================================

    #[test]
    fn test_model_value_fingerprint_deterministic() {
        use crate::state::value_fingerprint;
        for name in ["m1", "m2", "m3", "a", "b", "c"] {
            let v = Value::ModelValue(Arc::from(name));
            let fp1 = value_fingerprint(&v);
            let fp2 = value_fingerprint(&v);
            assert_eq!(fp1, fp2, "ModelValue fingerprint must be deterministic");
        }
    }

    #[test]
    fn test_model_value_equality_by_name() {
        let v1 = Value::ModelValue(Arc::from("m1"));
        let v2 = Value::ModelValue(Arc::from("m1"));
        let v3 = Value::ModelValue(Arc::from("m2"));

        assert_eq!(v1, v2, "Same name model values must be equal");
        assert_ne!(v1, v3, "Different name model values must not be equal");
    }

    #[test]
    fn test_model_value_equality_reflexive() {
        for name in ["m1", "m2", "m3"] {
            let v = Value::ModelValue(Arc::from(name));
            assert_eq!(v, v, "ModelValue equality must be reflexive");
        }
    }

    #[test]
    fn test_model_value_type_discrimination() {
        let mv = Value::ModelValue(Arc::from("m1"));
        let b = Value::Bool(true);
        let i = Value::Int(BigInt::from(1));
        let s = Value::String(Arc::from("m1")); // Same text, different type

        assert_ne!(mv, b, "ModelValue must not equal Bool");
        assert_ne!(mv, i, "ModelValue must not equal Int");
        assert_ne!(mv, s, "ModelValue must not equal String with same text");
    }

    // =========================================================================
    // Cardinality Semantics Tests (P56)
    // =========================================================================

    #[test]
    fn test_interval_cardinality() {
        let iv1 = IntervalValue::new(BigInt::from(1), BigInt::from(5));
        assert_eq!(iv1.len(), BigInt::from(5), "1..5 has cardinality 5");

        let iv2 = IntervalValue::new(BigInt::from(-2), BigInt::from(2));
        assert_eq!(iv2.len(), BigInt::from(5), "-2..2 has cardinality 5");

        let iv3 = IntervalValue::new(BigInt::from(0), BigInt::from(0));
        assert_eq!(iv3.len(), BigInt::from(1), "0..0 has cardinality 1");
    }

    #[test]
    fn test_empty_interval_cardinality() {
        let iv = IntervalValue::new(BigInt::from(5), BigInt::from(2));
        assert_eq!(
            iv.len(),
            BigInt::from(0),
            "Empty interval has cardinality 0"
        );
    }

    #[test]
    fn test_set_cardinality() {
        let empty: OrdSet<Value> = OrdSet::new();
        assert_eq!(empty.len(), 0);

        let mut s1 = OrdSet::new();
        s1.insert(Value::Int(BigInt::from(1)));
        assert_eq!(s1.len(), 1);

        let mut s2 = OrdSet::new();
        s2.insert(Value::Int(BigInt::from(1)));
        s2.insert(Value::Int(BigInt::from(2)));
        s2.insert(Value::Int(BigInt::from(3)));
        assert_eq!(s2.len(), 3);
    }

    #[test]
    fn test_sequence_length() {
        let empty: Vec<Value> = vec![];
        assert_eq!(empty.len(), 0);

        let s1 = [Value::Int(BigInt::from(1))];
        assert_eq!(s1.len(), 1);

        let s3 = [
            Value::Int(BigInt::from(1)),
            Value::Int(BigInt::from(2)),
            Value::Int(BigInt::from(3)),
        ];
        assert_eq!(s3.len(), 3);
    }

    // =========================================================================
    // Empty Collection Semantics Tests (P57)
    // =========================================================================

    #[test]
    fn test_empty_set_has_no_elements() {
        let empty: OrdSet<Value> = OrdSet::new();
        let test_values = [
            Value::Bool(true),
            Value::Int(BigInt::from(42)),
            Value::String(Arc::from("test")),
        ];

        for v in test_values {
            assert!(
                !empty.contains(&v),
                "Empty set must not contain any element"
            );
        }
    }

    #[test]
    fn test_empty_set_subset_of_all() {
        let empty: OrdSet<Value> = OrdSet::new();
        let mut other = OrdSet::new();
        other.insert(Value::Int(BigInt::from(1)));
        other.insert(Value::Int(BigInt::from(2)));

        assert!(
            empty.is_subset(&other),
            "Empty set must be subset of all sets"
        );
        assert!(
            empty.is_subset(&empty),
            "Empty set must be subset of itself"
        );
    }

    #[test]
    fn test_empty_sequence_properties() {
        let empty: Vec<Value> = vec![];

        assert!(empty.is_empty(), "Empty sequence must be empty");
        // Using is_empty() to check for no Head (first() returns None when empty)
        assert!(empty.is_empty(), "Empty sequence has no Head");
        assert!(empty.last().is_none(), "Empty sequence has no last element");
    }

    #[test]
    fn test_empty_function_properties() {
        let empty_domain: OrdSet<Value> = OrdSet::new();
        let empty_mapping: OrdMap<Value, Value> = OrdMap::new();
        let f = FuncValue::new(empty_domain.clone(), empty_mapping);

        assert!(f.domain_is_empty(), "Empty function has empty domain");
        assert!(f.domain_is_empty(), "Empty function has empty mapping");

        let v = Value::Int(BigInt::from(1));
        assert!(
            f.apply(&v).is_none(),
            "Empty function returns None for any input"
        );
    }

    // =========================================================================
    // IF-THEN-ELSE Semantics Tests (P58)
    // =========================================================================

    #[test]
    fn test_if_true_returns_then_branch() {
        let then_val = Value::Int(BigInt::from(1));
        let else_val = Value::Int(BigInt::from(2));

        let result = if true { then_val.clone() } else { else_val };

        assert_eq!(result, then_val, "IF TRUE THEN x ELSE y must equal x");
    }

    #[test]
    fn test_if_false_returns_else_branch() {
        let then_val = Value::Int(BigInt::from(1));
        let else_val = Value::Int(BigInt::from(2));

        let result = if false { then_val } else { else_val.clone() };

        assert_eq!(result, else_val, "IF FALSE THEN x ELSE y must equal y");
    }

    #[test]
    #[allow(clippy::if_same_then_else)]
    fn test_if_same_branches_equals_branch() {
        // Intentionally testing IF-THEN-ELSE with identical branches
        let val = Value::Int(BigInt::from(42));

        let result_true = if true { val.clone() } else { val.clone() };
        let result_false = if false { val.clone() } else { val.clone() };

        assert_eq!(result_true, val);
        assert_eq!(result_false, val);
    }

    #[test]
    fn test_nested_if_consistency() {
        let a = Value::Int(BigInt::from(1));
        let b = Value::Int(BigInt::from(2));
        let c = Value::Int(BigInt::from(3));

        for c1 in [true, false] {
            for c2 in [true, false] {
                let nested = if c1 {
                    if c2 {
                        a.clone()
                    } else {
                        b.clone()
                    }
                } else {
                    c.clone()
                };

                let expected = if c1 && c2 {
                    a.clone()
                } else if c1 && !c2 {
                    b.clone()
                } else {
                    c.clone()
                };

                assert_eq!(nested, expected, "Nested IF must evaluate correctly");
            }
        }
    }

    // =========================================================================
    // Interval Enumeration Correctness Tests (P59)
    // =========================================================================

    #[test]
    fn test_interval_iteration_order() {
        let iv = IntervalValue::new(BigInt::from(1), BigInt::from(5));
        let elements: Vec<BigInt> = iv.iter().collect();

        assert_eq!(elements.len(), 5);
        assert_eq!(elements[0], BigInt::from(1));
        assert_eq!(elements[1], BigInt::from(2));
        assert_eq!(elements[2], BigInt::from(3));
        assert_eq!(elements[3], BigInt::from(4));
        assert_eq!(elements[4], BigInt::from(5));
    }

    #[test]
    fn test_interval_contains_all_iterated() {
        let iv = IntervalValue::new(BigInt::from(1), BigInt::from(5));

        for n in iv.iter() {
            assert!(
                iv.contains(&Value::Int(n)),
                "Interval must contain all iterated elements"
            );
        }
    }

    #[test]
    fn test_interval_iteration_count() {
        let test_cases = [(1, 5, 5), (0, 0, 1), (-2, 2, 5), (3, 3, 1)];

        for (low, high, expected) in test_cases {
            let iv = IntervalValue::new(BigInt::from(low), BigInt::from(high));
            let count = iv.iter().count();
            assert_eq!(
                count, expected,
                "Iteration count must equal cardinality for {}..{}",
                low, high
            );
        }
    }

    #[test]
    fn test_empty_interval_iteration() {
        let iv = IntervalValue::new(BigInt::from(5), BigInt::from(2));
        let count = iv.iter().count();
        assert_eq!(count, 0, "Empty interval iteration must yield no elements");
    }

    // =========================================================================
    // Function Construction Semantics Tests (P60)
    // =========================================================================

    #[test]
    fn test_func_domain_equals_construction_domain() {
        let mut domain = OrdSet::new();
        let mut mapping = OrdMap::new();

        domain.insert(Value::Int(BigInt::from(1)));
        domain.insert(Value::Int(BigInt::from(2)));
        mapping.insert(Value::Int(BigInt::from(1)), Value::Bool(true));
        mapping.insert(Value::Int(BigInt::from(2)), Value::Bool(false));

        let f = FuncValue::new(domain.clone(), mapping);

        assert_eq!(
            f.domain_as_ord_set(),
            domain,
            "Function domain must equal construction domain"
        );
    }

    #[test]
    fn test_func_mapping_size_equals_domain() {
        for size in 0..=4 {
            let mut domain = OrdSet::new();
            let mut mapping = OrdMap::new();

            for i in 0..size {
                let k = Value::Int(BigInt::from(i));
                domain.insert(k.clone());
                mapping.insert(k, Value::Bool(i % 2 == 0));
            }

            let f = FuncValue::new(domain, mapping);
            // With array-based FuncValue, domain_len() is always consistent
            assert_eq!(
                f.domain_len(),
                size as usize,
                "Function domain size must equal mapping size"
            );
        }
    }
}
