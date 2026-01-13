//! Performance benchmarks for DashProve LSP
//!
//! Run with: cargo bench -p dashprove-lsp
//!
//! These benchmarks measure the performance of LSP operations:
//! - Document creation and analysis (parse + typecheck)
//! - Position conversion (LSP position <-> byte offset)
//! - Word finding at position
//! - Reference finding (find all identifier occurrences)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dashprove_lsp::{Document, DocumentStore};
use tower_lsp::lsp_types::Url;

// Test specifications of varying complexity
const SIMPLE_SPEC: &str = "theorem test { forall x: Bool . x or not x }";

const MEDIUM_SPEC: &str = r#"
type Counter = {
    value: Int,
    max: Int
}

theorem excluded_middle {
    forall x: Bool . x or not x
}

theorem implication {
    forall p: Bool, q: Bool . (p and (p implies q)) implies q
}

invariant positive_counter {
    forall c: Counter . c.value >= 0
}

invariant bounded_counter {
    forall c: Counter . c.value <= c.max
}
"#;

const LARGE_SPEC: &str = r#"
type Account = {
    id: Int,
    balance: Int,
    owner: String,
    active: Bool
}

type Transaction = {
    from_account: Int,
    to_account: Int,
    amount: Int,
    timestamp: Int
}

type LedgerEntry = {
    transaction: Transaction,
    resulting_balance: Int
}

theorem excluded_middle {
    forall x: Bool . x or not x
}

theorem implication {
    forall p: Bool, q: Bool . (p and (p implies q)) implies q
}

theorem de_morgan {
    forall a: Bool, b: Bool . not (a and b) == (not a or not b)
}

theorem contraposition {
    forall p: Bool, q: Bool . (p implies q) == (not q implies not p)
}

theorem double_negation {
    forall x: Bool . not (not x) == x
}

invariant positive_balance {
    forall a: Account . a.active implies a.balance >= 0
}

invariant positive_amount {
    forall t: Transaction . t.amount > 0
}

invariant valid_accounts {
    forall t: Transaction . t.from_account != t.to_account
}

invariant sequential_timestamps {
    forall t1: Transaction, t2: Transaction .
        t1.timestamp < t2.timestamp implies t1.timestamp + 1 <= t2.timestamp
}

invariant balanced_ledger {
    forall e: LedgerEntry . e.resulting_balance >= 0
}

contract transfer_valid {
    requires: forall t: Transaction . t.amount > 0
    ensures: forall t: Transaction . t.from_account != t.to_account
}
"#;

fn test_uri() -> Url {
    Url::parse("file:///test.usl").unwrap()
}

fn bench_document_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("document_creation");

    // Document creation includes parsing and type checking
    group.bench_function("simple", |b| {
        b.iter(|| Document::new(test_uri(), 1, black_box(SIMPLE_SPEC.to_string())))
    });

    group.bench_function("medium", |b| {
        b.iter(|| Document::new(test_uri(), 1, black_box(MEDIUM_SPEC.to_string())))
    });

    group.bench_function("large", |b| {
        b.iter(|| Document::new(test_uri(), 1, black_box(LARGE_SPEC.to_string())))
    });

    group.finish();
}

fn bench_position_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("position_conversion");

    // Create documents
    let simple_doc = Document::new(test_uri(), 1, SIMPLE_SPEC.to_string());
    let medium_doc = Document::new(test_uri(), 1, MEDIUM_SPEC.to_string());
    let large_doc = Document::new(test_uri(), 1, LARGE_SPEC.to_string());

    // Benchmark position_to_offset
    group.bench_with_input(
        BenchmarkId::new("position_to_offset", "simple/start"),
        &simple_doc,
        |b, doc| b.iter(|| doc.position_to_offset(black_box(0), black_box(0))),
    );

    group.bench_with_input(
        BenchmarkId::new("position_to_offset", "simple/middle"),
        &simple_doc,
        |b, doc| b.iter(|| doc.position_to_offset(black_box(0), black_box(20))),
    );

    group.bench_with_input(
        BenchmarkId::new("position_to_offset", "medium/line5"),
        &medium_doc,
        |b, doc| b.iter(|| doc.position_to_offset(black_box(5), black_box(10))),
    );

    group.bench_with_input(
        BenchmarkId::new("position_to_offset", "large/line10"),
        &large_doc,
        |b, doc| b.iter(|| doc.position_to_offset(black_box(10), black_box(15))),
    );

    group.bench_with_input(
        BenchmarkId::new("position_to_offset", "large/line50"),
        &large_doc,
        |b, doc| b.iter(|| doc.position_to_offset(black_box(50), black_box(10))),
    );

    // Benchmark offset_to_position
    group.bench_with_input(
        BenchmarkId::new("offset_to_position", "simple/start"),
        &simple_doc,
        |b, doc| b.iter(|| doc.offset_to_position(black_box(0))),
    );

    group.bench_with_input(
        BenchmarkId::new("offset_to_position", "simple/middle"),
        &simple_doc,
        |b, doc| b.iter(|| doc.offset_to_position(black_box(20))),
    );

    group.bench_with_input(
        BenchmarkId::new("offset_to_position", "large/middle"),
        &large_doc,
        |b, doc| b.iter(|| doc.offset_to_position(black_box(500))),
    );

    group.bench_with_input(
        BenchmarkId::new("offset_to_position", "large/end"),
        &large_doc,
        |b, doc| b.iter(|| doc.offset_to_position(black_box(1500))),
    );

    group.finish();
}

fn bench_word_at_position(c: &mut Criterion) {
    let mut group = c.benchmark_group("word_at_position");

    let medium_doc = Document::new(test_uri(), 1, MEDIUM_SPEC.to_string());
    let large_doc = Document::new(test_uri(), 1, LARGE_SPEC.to_string());

    // Find "theorem" keyword
    group.bench_with_input(
        BenchmarkId::new("word_at", "medium/keyword"),
        &medium_doc,
        |b, doc| b.iter(|| doc.word_at_position(black_box(6), black_box(3))),
    );

    // Find identifier in large doc
    group.bench_with_input(
        BenchmarkId::new("word_at", "large/identifier"),
        &large_doc,
        |b, doc| b.iter(|| doc.word_at_position(black_box(15), black_box(10))),
    );

    // Find at various positions
    group.bench_with_input(
        BenchmarkId::new("word_at", "large/line30"),
        &large_doc,
        |b, doc| b.iter(|| doc.word_at_position(black_box(30), black_box(8))),
    );

    group.finish();
}

fn bench_find_references(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_references");

    let medium_doc = Document::new(test_uri(), 1, MEDIUM_SPEC.to_string());
    let large_doc = Document::new(test_uri(), 1, LARGE_SPEC.to_string());

    // Find single identifier range
    group.bench_with_input(
        BenchmarkId::new("find_identifier_range", "medium/Counter"),
        &medium_doc,
        |b, doc| b.iter(|| doc.find_identifier_range(black_box("Counter"))),
    );

    group.bench_with_input(
        BenchmarkId::new("find_identifier_range", "large/Account"),
        &large_doc,
        |b, doc| b.iter(|| doc.find_identifier_range(black_box("Account"))),
    );

    group.bench_with_input(
        BenchmarkId::new("find_identifier_range", "large/Transaction"),
        &large_doc,
        |b, doc| b.iter(|| doc.find_identifier_range(black_box("Transaction"))),
    );

    // Find all references (multiple occurrences)
    group.bench_with_input(
        BenchmarkId::new("find_all_references", "medium/Counter"),
        &medium_doc,
        |b, doc| b.iter(|| doc.find_all_references(black_box("Counter"))),
    );

    group.bench_with_input(
        BenchmarkId::new("find_all_references", "large/Account"),
        &large_doc,
        |b, doc| b.iter(|| doc.find_all_references(black_box("Account"))),
    );

    group.bench_with_input(
        BenchmarkId::new("find_all_references", "large/Transaction"),
        &large_doc,
        |b, doc| b.iter(|| doc.find_all_references(black_box("Transaction"))),
    );

    // Find common identifier that appears many times
    group.bench_with_input(
        BenchmarkId::new("find_all_references", "large/forall"),
        &large_doc,
        |b, doc| b.iter(|| doc.find_all_references(black_box("forall"))),
    );

    group.finish();
}

fn bench_document_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("document_store");

    // Benchmark document store operations
    group.bench_function("open_close_cycle", |b| {
        let store = DocumentStore::new();
        let uri = test_uri();
        b.iter(|| {
            store.open(uri.clone(), 1, black_box(MEDIUM_SPEC.to_string()));
            store.close(&uri);
        })
    });

    group.bench_function("update_document", |b| {
        let store = DocumentStore::new();
        let uri = test_uri();
        store.open(uri.clone(), 1, MEDIUM_SPEC.to_string());
        let mut version = 2;
        b.iter(|| {
            store.update(&uri, version, black_box(MEDIUM_SPEC.to_string()));
            version += 1;
        })
    });

    group.bench_function("with_document_read", |b| {
        let store = DocumentStore::new();
        let uri = test_uri();
        store.open(uri.clone(), 1, MEDIUM_SPEC.to_string());
        b.iter(|| store.with_document(black_box(&uri), |doc| doc.version))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_document_creation,
    bench_position_conversion,
    bench_word_at_position,
    bench_find_references,
    bench_document_store,
);

criterion_main!(benches);
