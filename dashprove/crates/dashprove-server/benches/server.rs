//! Performance benchmarks for DashProve Server
//!
//! Run with: cargo bench -p dashprove-server
//!
//! These benchmarks measure the performance of server operations:
//! - ProofCache operations (hash, get, put, eviction, stats)
//! - AuthConfig builder patterns
//! - RateLimiter rate limit checks (async)
//! - AuthState key management operations (async)

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use dashprove_backends::BackendId;
use dashprove_server::auth::{AuthConfig, AuthState, RateLimiter};
use dashprove_server::cache::ProofCache;
use dashprove_usl::ast::{Expr, Invariant, Property, Theorem, Type};
use tokio::runtime::Runtime;

const CACHE_TTL_SECS: u64 = 3600;
const CACHE_MAX_ENTRIES: usize = 10_000;
const ADMIN_KEY: &str = "admin-key-123456789";
const USER_KEY: &str = "user-key-1234567890";

// Helper to create test properties of varying complexity
fn make_simple_property(name: &str) -> Property {
    Property::Invariant(Invariant {
        name: name.to_string(),
        body: Expr::Bool(true),
    })
}

fn make_medium_property(name: &str) -> Property {
    Property::Theorem(Theorem {
        name: name.to_string(),
        body: Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("Bool".to_string())),
            body: Box::new(Expr::Or(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Not(Box::new(Expr::Var("x".to_string())))),
            )),
        },
    })
}

fn make_complex_property(name: &str) -> Property {
    Property::Theorem(Theorem {
        name: name.to_string(),
        body: Expr::ForAll {
            var: "p".to_string(),
            ty: Some(Type::Named("Bool".to_string())),
            body: Box::new(Expr::ForAll {
                var: "q".to_string(),
                ty: Some(Type::Named("Bool".to_string())),
                body: Box::new(Expr::Implies(
                    Box::new(Expr::And(
                        Box::new(Expr::Var("p".to_string())),
                        Box::new(Expr::Implies(
                            Box::new(Expr::Var("p".to_string())),
                            Box::new(Expr::Var("q".to_string())),
                        )),
                    )),
                    Box::new(Expr::Var("q".to_string())),
                )),
            }),
        },
    })
}

fn populate_cache(cache: &mut ProofCache, count: usize, prefix: &str, backend: BackendId) {
    for i in 0..count {
        cache.put(
            format!("{prefix}{i}"),
            i as u64,
            true,
            backend,
            "code".to_string(),
            None,
        );
    }
}

fn cache_with_entries(count: usize, max_entries: usize) -> ProofCache {
    let mut cache = ProofCache::with_config(CACHE_TTL_SECS, max_entries);
    populate_cache(&mut cache, count, "prop_", BackendId::Lean4);
    cache
}

fn bench_proof_cache_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_cache/hash");

    let cases = [
        ("simple", make_simple_property("simple")),
        ("medium", make_medium_property("medium")),
        ("complex", make_complex_property("complex")),
    ];

    for (name, property) in cases {
        group.bench_function(name, |b| {
            b.iter(|| ProofCache::hash_property(black_box(&property)))
        });
    }

    group.finish();
}

fn bench_proof_cache_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_cache/get");

    // Create cache with entries
    let mut cache = ProofCache::with_config(CACHE_TTL_SECS, CACHE_MAX_ENTRIES);
    let property = make_medium_property("test");
    let hash = ProofCache::hash_property(&property);

    // Add test entry
    cache.put(
        "test".to_string(),
        hash,
        true,
        BackendId::Lean4,
        "-- lean code".to_string(),
        None,
    );

    // Add many other entries for realistic cache state
    populate_cache(&mut cache, 100, "prop_", BackendId::TlaPlus);

    group.bench_function("hit", |b| {
        b.iter(|| cache.get(black_box("test"), black_box(hash)))
    });

    group.bench_function("miss_wrong_name", |b| {
        b.iter(|| cache.get(black_box("nonexistent"), black_box(hash)))
    });

    group.bench_function("miss_wrong_hash", |b| {
        b.iter(|| cache.get(black_box("test"), black_box(hash + 1)))
    });

    group.finish();
}

fn bench_proof_cache_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_cache/put");

    group.bench_function("new_entry", |b| {
        let mut cache = ProofCache::with_config(CACHE_TTL_SECS, CACHE_MAX_ENTRIES);
        let mut i = 0u64;
        b.iter(|| {
            cache.put(
                format!("prop_{i}"),
                i,
                true,
                BackendId::Lean4,
                black_box("code".to_string()),
                None,
            );
            i += 1;
        })
    });

    group.bench_function("update_entry", |b| {
        let mut cache = ProofCache::with_config(CACHE_TTL_SECS, CACHE_MAX_ENTRIES);
        cache.put(
            "test".to_string(),
            42,
            true,
            BackendId::Lean4,
            "code".to_string(),
            None,
        );
        let mut version = 0u64;
        b.iter(|| {
            cache.put(
                "test".to_string(),
                version,
                true,
                BackendId::Lean4,
                black_box("code".to_string()),
                None,
            );
            version += 1;
        })
    });

    group.finish();
}

fn bench_proof_cache_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_cache/eviction");

    // Benchmark put when cache is at capacity (triggers eviction)
    group.bench_function("put_at_capacity", |b| {
        let mut cache = cache_with_entries(100, 100);
        let mut i = 100u64;
        b.iter(|| {
            cache.put(
                format!("overflow_{i}"),
                i,
                true,
                BackendId::Lean4,
                black_box("code".to_string()),
                None,
            );
            i += 1;
        })
    });

    group.finish();
}

fn bench_proof_cache_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_cache/stats");

    let empty_cache = ProofCache::new();
    group.bench_function("empty", |b| b.iter(|| empty_cache.stats()));

    let caches = [
        ("10_entries", cache_with_entries(10, CACHE_MAX_ENTRIES)),
        ("100_entries", cache_with_entries(100, CACHE_MAX_ENTRIES)),
        ("1000_entries", cache_with_entries(1000, CACHE_MAX_ENTRIES)),
    ];

    for (name, cache) in caches {
        group.bench_function(name, |b| b.iter(|| cache.stats()));
    }

    group.finish();
}

fn bench_proof_cache_invalidate(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_cache/invalidate");

    for affected in [1usize, 10, 50] {
        group.bench_with_input(
            BenchmarkId::new("affected", affected),
            &affected,
            |b, &n| {
                let affected: Vec<String> = (0..n).map(|i| format!("prop_{i}")).collect();
                b.iter_batched(
                    || (cache_with_entries(100, CACHE_MAX_ENTRIES), affected.clone()),
                    |(mut cache, affected)| cache.invalidate_affected(black_box(&affected)),
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

fn bench_auth_config_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("auth_config/builder");

    group.bench_function("default", |b| b.iter(AuthConfig::default));

    group.bench_function("with_single_key", |b| {
        b.iter(|| AuthConfig::default().with_api_key("test-key-12345", "Test User"))
    });

    group.bench_function("with_multiple_keys", |b| {
        b.iter(|| {
            AuthConfig::default()
                .with_api_key("key1-12345678", "User 1")
                .with_api_key("key2-12345678", "User 2")
                .with_api_key("key3-12345678", "User 3")
                .with_admin_key("admin-12345678", "Admin")
        })
    });

    group.bench_function("with_rate_limits", |b| {
        b.iter(|| {
            AuthConfig::default()
                .with_api_key_rate_limit("key1", "User 1", 100)
                .with_api_key_rate_limit("key2", "User 2", 200)
                .with_admin_key_rate_limit("admin", "Admin", 1000)
                .with_anonymous_rate_limit(10)
        })
    });

    group.finish();
}

fn bench_rate_limiter(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("rate_limiter");

    // Benchmark rate limiter check (under limit)
    group.bench_function("check_under_limit", |b| {
        let limiter = RateLimiter::new();
        let mut i = 0u64;
        b.iter(|| {
            rt.block_on(async {
                // Use different keys to avoid hitting the limit
                let key = format!("key_{}", i % 1000);
                i += 1;
                limiter.check(black_box(&key), black_box(100)).await
            })
        })
    });

    // Benchmark rate limiter check (at limit)
    group.bench_function("check_at_limit", |b| {
        let limiter = RateLimiter::new();
        // Pre-exhaust the limit
        rt.block_on(async {
            for _ in 0..100 {
                let _ = limiter.check("exhausted_key", 100).await;
            }
        });
        b.iter(|| {
            rt.block_on(async {
                limiter
                    .check(black_box("exhausted_key"), black_box(100))
                    .await
            })
        })
    });

    // Benchmark current_count lookup
    group.bench_function("current_count", |b| {
        let limiter = RateLimiter::new();
        rt.block_on(async {
            for _ in 0..50 {
                let _ = limiter.check("tracked_key", 100).await;
            }
        });
        b.iter(|| rt.block_on(async { limiter.current_count(black_box("tracked_key")).await }))
    });

    // Benchmark cleanup
    group.bench_function("cleanup", |b| {
        let limiter = RateLimiter::new();
        // Add some buckets
        rt.block_on(async {
            for i in 0..100 {
                let _ = limiter.check(&format!("key_{i}"), 100).await;
            }
        });
        b.iter(|| rt.block_on(async { limiter.cleanup().await }))
    });

    group.finish();
}

fn bench_auth_state(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("auth_state");

    // Create auth state with some keys
    let config = AuthConfig::default()
        .with_api_key(USER_KEY, "User")
        .with_admin_key(ADMIN_KEY, "Admin");
    let base_state = AuthState::new(config);

    // Benchmark is_admin check
    group.bench_function("is_admin/admin_key", |b| {
        let state = base_state.clone();
        b.iter(|| rt.block_on(async { state.is_admin(black_box(ADMIN_KEY)).await }))
    });

    group.bench_function("is_admin/user_key", |b| {
        let state = base_state.clone();
        b.iter(|| rt.block_on(async { state.is_admin(black_box(USER_KEY)).await }))
    });

    group.bench_function("is_admin/nonexistent", |b| {
        let state = base_state.clone();
        b.iter(|| rt.block_on(async { state.is_admin(black_box("nonexistent-key")).await }))
    });

    // Benchmark has_key check
    group.bench_function("has_key/exists", |b| {
        let state = base_state.clone();
        b.iter(|| rt.block_on(async { state.has_key(black_box(USER_KEY)).await }))
    });

    group.bench_function("has_key/not_exists", |b| {
        let state = base_state.clone();
        b.iter(|| rt.block_on(async { state.has_key(black_box("nonexistent-key")).await }))
    });

    // Benchmark get_key_info
    group.bench_function("get_key_info/exists", |b| {
        let state = base_state.clone();
        b.iter(|| rt.block_on(async { state.get_key_info(black_box(USER_KEY)).await }))
    });

    // Benchmark list_keys
    group.bench_function("list_keys/2_keys", |b| {
        let state = base_state.clone();
        b.iter(|| rt.block_on(async { state.list_keys().await }))
    });

    // Benchmark with more keys
    let mut many_keys_config = AuthConfig::default();
    for i in 0..50 {
        many_keys_config =
            many_keys_config.with_api_key(&format!("key-{i:03}-12345678"), &format!("User {i}"));
    }

    group.bench_function("list_keys/50_keys", |b| {
        let state = AuthState::new(many_keys_config.clone());
        b.iter(|| rt.block_on(async { state.list_keys().await }))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_proof_cache_hash,
    bench_proof_cache_get,
    bench_proof_cache_put,
    bench_proof_cache_eviction,
    bench_proof_cache_stats,
    bench_proof_cache_invalidate,
    bench_auth_config_builder,
    bench_rate_limiter,
    bench_auth_state,
);

criterion_main!(benches);
