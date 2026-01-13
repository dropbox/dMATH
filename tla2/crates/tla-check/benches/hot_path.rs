//! Microbenchmarks for hot path components
//!
//! These benchmarks measure isolated operations in the model checking hot path:
//! - State fingerprinting (value_fingerprint, ArrayState::fingerprint)
//! - Value hashing for different value types
//! - Set operations (union, intersection, contains)
//! - Function application (apply, except)
//!
//! Run with: cargo bench -p tla-check --bench hot_path

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_bigint::BigInt;
use std::sync::Arc;
use tla_check::state::{
    value_fingerprint, value_fingerprint_ahash, value_fingerprint_xxh3, ArrayState,
};
use tla_check::value::{FuncValue, SeqValue, SortedSet};
use tla_check::var_index::VarRegistry;
use tla_check::Value;

// ============================================================================
// Test Data Generators
// ============================================================================

/// Create a small integer value
fn small_int(n: i64) -> Value {
    Value::SmallInt(n)
}

/// Create a big integer value
fn big_int(n: i64) -> Value {
    Value::Int(BigInt::from(n))
}

/// Create a string value
fn string_val(s: &str) -> Value {
    Value::String(Arc::from(s))
}

/// Create a set of integers {0, 1, ..., n-1}
fn int_set(n: usize) -> Value {
    let values: Vec<Value> = (0..n as i64).map(Value::SmallInt).collect();
    Value::Set(SortedSet::from_sorted_vec(values))
}

/// Create a SortedSet of integers {0, 1, ..., n-1}
fn sorted_int_set(n: usize) -> SortedSet {
    let values: Vec<Value> = (0..n as i64).map(Value::SmallInt).collect();
    SortedSet::from_sorted_vec(values)
}

/// Create a function from integers to integers: [i \in 0..n-1 |-> i * 2]
fn int_func(n: usize) -> FuncValue {
    let entries: Vec<(Value, Value)> = (0..n as i64)
        .map(|i| (Value::SmallInt(i), Value::SmallInt(i * 2)))
        .collect();
    FuncValue::from_sorted_entries(entries)
}

/// Create a VarRegistry with n variables
fn make_registry(n: usize) -> VarRegistry {
    let names: Vec<Arc<str>> = (0..n).map(|i| Arc::from(format!("var{}", i))).collect();
    VarRegistry::from_names(names)
}

/// Create an ArrayState with n integer variables
fn make_array_state(n: usize) -> ArrayState {
    let values: Vec<Value> = (0..n as i64).map(Value::SmallInt).collect();
    ArrayState::from_values(values)
}

/// Create an ArrayState simulating LamportMutex-style state (functions over processes)
fn make_mutex_state(num_procs: usize) -> ArrayState {
    // pc[p]: process program counters (function from process id to string)
    let pc_entries: Vec<(Value, Value)> = (0..num_procs)
        .map(|i| (Value::SmallInt(i as i64), string_val("idle")))
        .collect();
    let pc = Value::Func(FuncValue::from_sorted_entries(pc_entries));

    // num[p]: bakery numbers (function from process id to int)
    let num_entries: Vec<(Value, Value)> = (0..num_procs)
        .map(|i| (Value::SmallInt(i as i64), Value::SmallInt(i as i64 + 1)))
        .collect();
    let num = Value::Func(FuncValue::from_sorted_entries(num_entries));

    // flag[p]: flags (function from process id to bool)
    let flag_entries: Vec<(Value, Value)> = (0..num_procs)
        .map(|i| (Value::SmallInt(i as i64), Value::Bool(i % 2 == 0)))
        .collect();
    let flag = Value::Func(FuncValue::from_sorted_entries(flag_entries));

    ArrayState::from_values(vec![pc, num, flag])
}

// ============================================================================
// Value Fingerprinting Benchmarks
// ============================================================================

fn bench_value_fingerprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("fingerprint/value");

    // SmallInt fingerprinting
    group.bench_function("smallint", |b| {
        let v = small_int(12345);
        b.iter(|| black_box(value_fingerprint(black_box(&v))))
    });

    // BigInt fingerprinting
    group.bench_function("bigint_small", |b| {
        let v = big_int(12345);
        b.iter(|| black_box(value_fingerprint(black_box(&v))))
    });

    group.bench_function("bigint_large", |b| {
        let v = Value::Int(BigInt::from(10).pow(50));
        b.iter(|| black_box(value_fingerprint(black_box(&v))))
    });

    // String fingerprinting
    group.bench_function("string_short", |b| {
        let v = string_val("idle");
        b.iter(|| black_box(value_fingerprint(black_box(&v))))
    });

    group.bench_function("string_medium", |b| {
        let v = string_val("waiting_for_critical_section");
        b.iter(|| black_box(value_fingerprint(black_box(&v))))
    });

    // Bool fingerprinting
    group.bench_function("bool", |b| {
        let v = Value::Bool(true);
        b.iter(|| black_box(value_fingerprint(black_box(&v))))
    });

    // Set fingerprinting (various sizes)
    for size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("set", size), &size, |b, &size| {
            let v = int_set(size);
            b.iter(|| black_box(value_fingerprint(black_box(&v))))
        });
    }

    // Function fingerprinting (various sizes)
    for size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("func", size), &size, |b, &size| {
            let func = int_func(size);
            let v = Value::Func(func);
            b.iter(|| {
                // Clear cache to measure actual fingerprinting work
                let v_clone = v.clone();
                black_box(value_fingerprint(black_box(&v_clone)))
            })
        });
    }

    // Function fingerprinting with cache (measures cache lookup overhead)
    group.bench_function("func_cached", |b| {
        let func = int_func(100);
        let v = Value::Func(func);
        // Warm the cache
        let _ = value_fingerprint(&v);
        b.iter(|| black_box(value_fingerprint(black_box(&v))))
    });

    group.finish();
}

// ============================================================================
// FNV vs XXH3 Comparison Benchmarks
// ============================================================================

fn bench_fnv_vs_xxh3(c: &mut Criterion) {
    let mut group = c.benchmark_group("fingerprint/hash_comparison");

    // SmallInt comparison
    {
        let v = small_int(12345);
        group.bench_function("smallint/fnv", |b| {
            b.iter(|| black_box(value_fingerprint(black_box(&v))))
        });
        group.bench_function("smallint/xxh3", |b| {
            b.iter(|| black_box(value_fingerprint_xxh3(black_box(&v))))
        });
        group.bench_function("smallint/ahash", |b| {
            b.iter(|| black_box(value_fingerprint_ahash(black_box(&v))))
        });
    }

    // String comparison
    {
        let v = string_val("waiting_for_critical_section");
        group.bench_function("string_medium/fnv", |b| {
            b.iter(|| black_box(value_fingerprint(black_box(&v))))
        });
        group.bench_function("string_medium/xxh3", |b| {
            b.iter(|| black_box(value_fingerprint_xxh3(black_box(&v))))
        });
        group.bench_function("string_medium/ahash", |b| {
            b.iter(|| black_box(value_fingerprint_ahash(black_box(&v))))
        });
    }

    // Set of 100 integers
    {
        let v = int_set(100);
        group.bench_function("set_100/fnv", |b| {
            b.iter(|| black_box(value_fingerprint(black_box(&v))))
        });
        group.bench_function("set_100/xxh3", |b| {
            b.iter(|| black_box(value_fingerprint_xxh3(black_box(&v))))
        });
        group.bench_function("set_100/ahash", |b| {
            b.iter(|| black_box(value_fingerprint_ahash(black_box(&v))))
        });
    }

    // Set of 1000 integers
    {
        let v = int_set(1000);
        group.bench_function("set_1000/fnv", |b| {
            b.iter(|| black_box(value_fingerprint(black_box(&v))))
        });
        group.bench_function("set_1000/xxh3", |b| {
            b.iter(|| black_box(value_fingerprint_xxh3(black_box(&v))))
        });
        group.bench_function("set_1000/ahash", |b| {
            b.iter(|| black_box(value_fingerprint_ahash(black_box(&v))))
        });
    }

    // Function of 100 entries (without cache)
    {
        let func = int_func(100);
        group.bench_function("func_100/fnv", |b| {
            let v = Value::Func(func.clone());
            b.iter(|| {
                let v_clone = v.clone();
                black_box(value_fingerprint(black_box(&v_clone)))
            })
        });
        let func = int_func(100);
        group.bench_function("func_100/xxh3", |b| {
            let v = Value::Func(func.clone());
            b.iter(|| {
                let v_clone = v.clone();
                black_box(value_fingerprint_xxh3(black_box(&v_clone)))
            })
        });
        let func = int_func(100);
        group.bench_function("func_100/ahash", |b| {
            let v = Value::Func(func.clone());
            b.iter(|| {
                let v_clone = v.clone();
                black_box(value_fingerprint_ahash(black_box(&v_clone)))
            })
        });
    }

    // Function of 1000 entries (without cache)
    {
        let func = int_func(1000);
        group.bench_function("func_1000/fnv", |b| {
            let v = Value::Func(func.clone());
            b.iter(|| {
                let v_clone = v.clone();
                black_box(value_fingerprint(black_box(&v_clone)))
            })
        });
        let func = int_func(1000);
        group.bench_function("func_1000/xxh3", |b| {
            let v = Value::Func(func.clone());
            b.iter(|| {
                let v_clone = v.clone();
                black_box(value_fingerprint_xxh3(black_box(&v_clone)))
            })
        });
        let func = int_func(1000);
        group.bench_function("func_1000/ahash", |b| {
            let v = Value::Func(func.clone());
            b.iter(|| {
                let v_clone = v.clone();
                black_box(value_fingerprint_ahash(black_box(&v_clone)))
            })
        });
    }

    group.finish();
}

// ============================================================================
// ArrayState Fingerprinting Benchmarks
// ============================================================================

fn bench_array_state_fingerprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("fingerprint/array_state");

    // Small state (3 variables, simple values)
    group.bench_function("small_3vars", |b| {
        let registry = make_registry(3);
        b.iter(|| {
            let mut state = make_array_state(3);
            black_box(state.fingerprint(black_box(&registry)))
        })
    });

    // Medium state (10 variables)
    group.bench_function("medium_10vars", |b| {
        let registry = make_registry(10);
        b.iter(|| {
            let mut state = make_array_state(10);
            black_box(state.fingerprint(black_box(&registry)))
        })
    });

    // Mutex-style state (3 function variables, 4 processes)
    group.bench_function("mutex_4proc", |b| {
        let registry =
            VarRegistry::from_names(vec![Arc::from("pc"), Arc::from("num"), Arc::from("flag")]);
        b.iter(|| {
            let mut state = make_mutex_state(4);
            black_box(state.fingerprint(black_box(&registry)))
        })
    });

    // Mutex-style state (3 function variables, 8 processes)
    group.bench_function("mutex_8proc", |b| {
        let registry =
            VarRegistry::from_names(vec![Arc::from("pc"), Arc::from("num"), Arc::from("flag")]);
        b.iter(|| {
            let mut state = make_mutex_state(8);
            black_box(state.fingerprint(black_box(&registry)))
        })
    });

    // Incremental fingerprint update (simulates changing one variable)
    group.bench_function("incremental_update", |b| {
        let registry =
            VarRegistry::from_names(vec![Arc::from("pc"), Arc::from("num"), Arc::from("flag")]);
        let mut state = make_mutex_state(4);
        state.ensure_fp_cache_with_value_fps(&registry);

        b.iter(|| {
            let mut state_clone = state.clone();
            let new_value = string_val("critical");
            let new_fp = value_fingerprint(&new_value);
            state_clone.set_with_fp(
                tla_check::var_index::VarIndex(0),
                new_value,
                new_fp,
                &registry,
            );
            black_box(state_clone.cached_fingerprint())
        })
    });

    group.finish();
}

// ============================================================================
// Set Operations Benchmarks
// ============================================================================

fn bench_set_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_ops");

    // Contains (membership test)
    for size in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("contains", size), &size, |b, &size| {
            let set = sorted_int_set(size);
            let target = small_int((size / 2) as i64);
            b.iter(|| black_box(set.contains(black_box(&target))))
        });
    }

    // Union
    for size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(size as u64 * 2));
        group.bench_with_input(BenchmarkId::new("union", size), &size, |b, &size| {
            let set1 = sorted_int_set(size);
            // Second set: {size/2, size/2+1, ..., 3*size/2-1} (50% overlap)
            let set2: SortedSet = {
                let start = (size / 2) as i64;
                let values: Vec<Value> =
                    (start..start + size as i64).map(Value::SmallInt).collect();
                SortedSet::from_sorted_vec(values)
            };
            b.iter(|| black_box(set1.union(black_box(&set2))))
        });
    }

    // Intersection
    for size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(size as u64 * 2));
        group.bench_with_input(BenchmarkId::new("intersection", size), &size, |b, &size| {
            let set1 = sorted_int_set(size);
            let set2: SortedSet = {
                let start = (size / 2) as i64;
                let values: Vec<Value> =
                    (start..start + size as i64).map(Value::SmallInt).collect();
                SortedSet::from_sorted_vec(values)
            };
            b.iter(|| black_box(set1.intersection(black_box(&set2))))
        });
    }

    // Difference
    for size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(size as u64 * 2));
        group.bench_with_input(BenchmarkId::new("difference", size), &size, |b, &size| {
            let set1 = sorted_int_set(size);
            let set2: SortedSet = {
                let start = (size / 2) as i64;
                let values: Vec<Value> =
                    (start..start + size as i64).map(Value::SmallInt).collect();
                SortedSet::from_sorted_vec(values)
            };
            b.iter(|| black_box(set1.difference(black_box(&set2))))
        });
    }

    // Insert (adding one element)
    for size in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("insert", size), &size, |b, &size| {
            let set = sorted_int_set(size);
            let new_elem = small_int(size as i64 + 1); // Element not in set
            b.iter(|| black_box(set.insert(black_box(new_elem.clone()))))
        });
    }

    // is_subset
    for size in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("is_subset", size), &size, |b, &size| {
            let subset: SortedSet = {
                let values: Vec<Value> = (0..size as i64 / 2).map(Value::SmallInt).collect();
                SortedSet::from_sorted_vec(values)
            };
            let superset = sorted_int_set(size);
            b.iter(|| black_box(subset.is_subset(black_box(&superset))))
        });
    }

    group.finish();
}

// ============================================================================
// Function Application Benchmarks
// ============================================================================

fn bench_func_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("func_ops");

    // Apply (lookup)
    for size in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("apply", size), &size, |b, &size| {
            let func = int_func(size);
            let key = small_int((size / 2) as i64);
            b.iter(|| black_box(func.apply(black_box(&key))))
        });
    }

    // Apply with miss (key not in domain)
    group.bench_function("apply_miss", |b| {
        let func = int_func(100);
        let key = small_int(1000); // Not in domain
        b.iter(|| black_box(func.apply(black_box(&key))))
    });

    // Except (update one entry)
    for size in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("except", size), &size, |b, &size| {
            let func = int_func(size);
            let key = small_int((size / 2) as i64);
            let new_val = small_int(999);
            b.iter(|| black_box(func.except(black_box(key.clone()), black_box(new_val.clone()))))
        });
    }

    // Except with same value (no-op optimization)
    group.bench_function("except_same_value", |b| {
        let func = int_func(100);
        let key = small_int(50);
        let same_val = small_int(100); // Same as func[50] = 50*2 = 100
        b.iter(|| black_box(func.except(black_box(key.clone()), black_box(same_val.clone()))))
    });

    // Domain contains
    for size in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::new("domain_contains", size),
            &size,
            |b, &size| {
                let func = int_func(size);
                let key = small_int((size / 2) as i64);
                b.iter(|| black_box(func.domain_contains(black_box(&key))))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Value Comparison Benchmarks (used in set operations)
// ============================================================================

fn bench_value_cmp(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_cmp");

    // SmallInt comparison
    group.bench_function("smallint", |b| {
        let a = small_int(12345);
        let b_val = small_int(12346);
        b.iter(|| black_box(a.cmp(black_box(&b_val))))
    });

    // String comparison
    group.bench_function("string", |b| {
        let a = string_val("idle");
        let b_val = string_val("waiting");
        b.iter(|| black_box(a.cmp(black_box(&b_val))))
    });

    // Set comparison (same size, different elements)
    group.bench_function("set_100", |b| {
        let a = int_set(100);
        let b_val: Value = {
            let values: Vec<Value> = (1..101).map(Value::SmallInt).collect();
            Value::Set(SortedSet::from_sorted_vec(values))
        };
        b.iter(|| black_box(a.cmp(black_box(&b_val))))
    });

    // Function comparison
    group.bench_function("func_100", |b| {
        let a = Value::Func(int_func(100));
        let b_val = Value::Func(int_func(100));
        b.iter(|| black_box(a.cmp(black_box(&b_val))))
    });

    group.finish();
}

// ============================================================================
// Value Clone Benchmarks
// ============================================================================

fn bench_value_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_clone");

    // SmallInt (trivial copy)
    group.bench_function("smallint", |b| {
        let v = small_int(12345);
        b.iter(|| black_box(v.clone()))
    });

    // String (Arc increment)
    group.bench_function("string", |b| {
        let v = string_val("waiting_for_critical_section");
        b.iter(|| black_box(v.clone()))
    });

    // Set (Arc increment)
    group.bench_function("set_100", |b| {
        let v = int_set(100);
        b.iter(|| black_box(v.clone()))
    });

    // Function (Arc increment)
    group.bench_function("func_100", |b| {
        let v = Value::Func(int_func(100));
        b.iter(|| black_box(v.clone()))
    });

    group.finish();
}

// ============================================================================
// Sequence Operation Benchmarks
// ============================================================================

/// Create a sequence of integers <<1, 2, ..., n>>
fn int_seq(n: usize) -> Value {
    let values: Vec<Value> = (1..=n as i64).map(Value::SmallInt).collect();
    Value::Seq(values.into())
}

/// Simulate Tail operation: seq[1..].to_vec().into()
fn seq_tail(seq: &SeqValue) -> Value {
    let slice = seq.as_slice();
    if slice.is_empty() {
        Value::Seq(Vec::new().into())
    } else {
        Value::Seq(slice[1..].to_vec().into())
    }
}

/// Simulate Append operation: clone all elements and push new
fn seq_append(seq: &SeqValue, elem: Value) -> Value {
    let mut new_seq: Vec<Value> = seq.iter().cloned().collect();
    new_seq.push(elem);
    Value::Seq(new_seq.into())
}

fn bench_seq_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("seq_ops");

    // Head (read first element - should be O(1))
    for size in [5, 10, 50] {
        group.bench_with_input(BenchmarkId::new("head", size), &size, |b, &size| {
            let seq = int_seq(size);
            let seq_value = match &seq {
                Value::Seq(s) => s,
                _ => panic!("Expected Seq"),
            };
            b.iter(|| black_box(seq_value.as_slice().first().cloned()))
        });
    }

    // Tail (creates new sequence without first element - O(n) clone)
    for size in [5, 10, 50] {
        group.bench_with_input(BenchmarkId::new("tail", size), &size, |b, &size| {
            let seq = int_seq(size);
            let arc_seq = match &seq {
                Value::Seq(s) => s.clone(),
                _ => panic!("Expected Seq"),
            };
            b.iter(|| black_box(seq_tail(&arc_seq)))
        });
    }

    // Append (creates new sequence with element added - O(n) clone)
    for size in [5, 10, 50] {
        group.bench_with_input(BenchmarkId::new("append", size), &size, |b, &size| {
            let seq = int_seq(size);
            let arc_seq = match &seq {
                Value::Seq(s) => s.clone(),
                _ => panic!("Expected Seq"),
            };
            let new_elem = small_int(999);
            b.iter(|| black_box(seq_append(&arc_seq, new_elem.clone())))
        });
    }

    // Compare with IntFunc EXCEPT (should be O(1) with COW)
    for size in [5, 10, 50] {
        group.bench_with_input(
            BenchmarkId::new("intfunc_except", size),
            &size,
            |b, &size| {
                use tla_check::value::IntIntervalFunc;
                let values: Vec<Value> = (1..=size as i64).map(Value::SmallInt).collect();
                let func = IntIntervalFunc::new(1, size as i64, values);
                let key = small_int(size as i64 / 2);
                let new_val = small_int(999);
                b.iter(|| black_box(func.clone().except(&key, new_val.clone())))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_value_fingerprint,
    bench_fnv_vs_xxh3,
    bench_array_state_fingerprint,
    bench_set_operations,
    bench_func_operations,
    bench_value_cmp,
    bench_value_clone,
    bench_seq_operations,
);
criterion_main!(benches);
