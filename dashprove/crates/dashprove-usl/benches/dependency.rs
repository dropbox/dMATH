//! Performance benchmarks for USL dependency analysis
//!
//! Run with: cargo bench -p dashprove-usl
//!
//! These benchmarks measure the performance of dependency graph operations:
//! - DependencyGraph construction from spec
//! - Affected property computation
//! - Transitive closure computation
//! - SpecDiff computation

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dashprove_usl::{parse, DependencyGraph, SpecDiff};

// Small specification (3 properties, 1 type)
const SMALL_SPEC: &str = r#"
type Node = { id: Int }

theorem node_id_positive {
    forall n: Node . n.id >= 0
}

theorem simple_bool {
    forall x: Bool . x or not x
}

invariant node_exists {
    forall n: Node . n.id == n.id
}
"#;

// Medium specification (10 properties, 5 types, property dependencies)
const MEDIUM_SPEC: &str = r#"
type Node = { id: Int, label: String }
type Edge = { from: Int, to: Int, weight: Int }
type Graph = { nodes: Set<Node>, edges: Set<Edge> }
type Path = { steps: List<Node> }
type State = { current: Node, visited: Set<Node> }

theorem graph_has_nodes {
    forall g: Graph . g.nodes.len() >= 0
}

theorem edge_weight_positive {
    forall e: Edge . e.weight >= 0
}

theorem path_non_empty {
    forall p: Path . p.steps.len() >= 0
}

theorem state_current_in_graph {
    forall s: State, g: Graph . s.current in g.nodes
}

invariant positive_ids {
    forall n: Node . n.id >= 0
}

invariant valid_edges {
    forall e: Edge . e.from >= 0 and e.to >= 0
}

theorem lemma_nodes {
    forall g: Graph . graph_has_nodes implies g.nodes.len() >= 0
}

theorem lemma_edges {
    forall g: Graph . graph_has_nodes implies g.edges.len() >= 0
}

theorem derived_theorem {
    forall g: Graph . lemma_nodes and lemma_edges
}

theorem final_theorem {
    forall g: Graph . derived_theorem implies true
}
"#;

// Large specification (20 properties, 8 types, deep dependency chains)
const LARGE_SPEC: &str = r#"
type Node = { id: Int, label: String, weight: Int }
type Edge = { from: Int, to: Int, weight: Int, label: String }
type Graph = { nodes: Set<Node>, edges: Set<Edge> }
type Path = { steps: List<Node>, total_weight: Int }
type State = { current: Node, visited: Set<Node>, distance: Int }
type Queue = { items: List<Node>, size: Int }
type Result = { found: Bool, path: Path }
type Config = { max_depth: Int, timeout: Int }

theorem t1 { forall x: Int . x >= 0 implies x * x >= 0 }
theorem t2 { forall n: Node . n.id >= 0 }
theorem t3 { forall e: Edge . e.weight >= 0 }
theorem t4 { forall g: Graph . g.nodes.len() >= 0 }
theorem t5 { forall p: Path . p.total_weight >= 0 }

theorem t6 { forall x: Int . t1 implies x >= 0 }
theorem t7 { forall n: Node . t2 implies n.weight >= 0 }
theorem t8 { forall e: Edge . t3 implies e.from >= 0 }
theorem t9 { forall g: Graph . t4 implies g.edges.len() >= 0 }
theorem t10 { forall p: Path . t5 implies p.steps.len() >= 0 }

theorem t11 { t6 and t7 }
theorem t12 { t8 and t9 }
theorem t13 { t10 and t11 }
theorem t14 { t12 and t13 }
theorem t15 { t14 implies true }

invariant i1 { forall n: Node . n.id >= 0 }
invariant i2 { forall e: Edge . e.from >= 0 and e.to >= 0 }
invariant i3 { forall s: State . s.distance >= 0 }
invariant i4 { forall q: Queue . q.size >= 0 }
invariant i5 { forall c: Config . c.max_depth > 0 }
"#;

// Modified version of medium spec for diff benchmarks
const MEDIUM_SPEC_MODIFIED: &str = r#"
type Node = { id: Int, label: String, active: Bool }
type Edge = { from: Int, to: Int, weight: Int }
type Graph = { nodes: Set<Node>, edges: Set<Edge> }
type Path = { steps: List<Node> }
type State = { current: Node, visited: Set<Node> }
type NewType = { value: Int }

theorem graph_has_nodes {
    forall g: Graph . g.nodes.len() > 0
}

theorem edge_weight_positive {
    forall e: Edge . e.weight >= 0
}

theorem path_non_empty {
    forall p: Path . p.steps.len() >= 0
}

theorem state_current_in_graph {
    forall s: State, g: Graph . s.current in g.nodes
}

invariant positive_ids {
    forall n: Node . n.id >= 0
}

invariant valid_edges {
    forall e: Edge . e.from >= 0 and e.to >= 0
}

theorem lemma_nodes {
    forall g: Graph . graph_has_nodes implies g.nodes.len() >= 0
}

theorem lemma_edges {
    forall g: Graph . graph_has_nodes implies g.edges.len() >= 0
}

theorem new_theorem {
    forall x: Int . x == x
}
"#;

fn bench_dependency_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dependency_graph_construction");

    // Pre-parse specs
    let small_spec = parse(SMALL_SPEC).unwrap();
    let medium_spec = parse(MEDIUM_SPEC).unwrap();
    let large_spec = parse(LARGE_SPEC).unwrap();

    group.bench_with_input(
        BenchmarkId::new("from_spec", "small (3 props)"),
        &small_spec,
        |b, spec| b.iter(|| DependencyGraph::from_spec(black_box(spec))),
    );

    group.bench_with_input(
        BenchmarkId::new("from_spec", "medium (10 props)"),
        &medium_spec,
        |b, spec| b.iter(|| DependencyGraph::from_spec(black_box(spec))),
    );

    group.bench_with_input(
        BenchmarkId::new("from_spec", "large (20 props)"),
        &large_spec,
        |b, spec| b.iter(|| DependencyGraph::from_spec(black_box(spec))),
    );

    group.finish();
}

fn bench_affected_properties(c: &mut Criterion) {
    let mut group = c.benchmark_group("affected_properties");

    // Build graphs
    let small_spec = parse(SMALL_SPEC).unwrap();
    let medium_spec = parse(MEDIUM_SPEC).unwrap();
    let large_spec = parse(LARGE_SPEC).unwrap();

    let small_graph = DependencyGraph::from_spec(&small_spec);
    let medium_graph = DependencyGraph::from_spec(&medium_spec);
    let large_graph = DependencyGraph::from_spec(&large_spec);

    // Benchmark properties_affected_by with single type
    group.bench_with_input(
        BenchmarkId::new("by_type/single", "small"),
        &(&small_graph, vec!["Node".to_string()]),
        |b, (graph, types)| b.iter(|| graph.properties_affected_by(black_box(types))),
    );

    group.bench_with_input(
        BenchmarkId::new("by_type/single", "medium"),
        &(&medium_graph, vec!["Graph".to_string()]),
        |b, (graph, types)| b.iter(|| graph.properties_affected_by(black_box(types))),
    );

    group.bench_with_input(
        BenchmarkId::new("by_type/single", "large"),
        &(&large_graph, vec!["Node".to_string()]),
        |b, (graph, types)| b.iter(|| graph.properties_affected_by(black_box(types))),
    );

    // Benchmark with multiple types
    group.bench_with_input(
        BenchmarkId::new("by_type/multiple", "medium"),
        &(
            &medium_graph,
            vec!["Node".to_string(), "Edge".to_string(), "Graph".to_string()],
        ),
        |b, (graph, types)| b.iter(|| graph.properties_affected_by(black_box(types))),
    );

    group.bench_with_input(
        BenchmarkId::new("by_type/multiple", "large"),
        &(
            &large_graph,
            vec![
                "Node".to_string(),
                "Edge".to_string(),
                "Graph".to_string(),
                "Path".to_string(),
            ],
        ),
        |b, (graph, types)| b.iter(|| graph.properties_affected_by(black_box(types))),
    );

    group.finish();
}

fn bench_compute_affected(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_affected");

    let medium_spec = parse(MEDIUM_SPEC).unwrap();
    let large_spec = parse(LARGE_SPEC).unwrap();

    let medium_graph = DependencyGraph::from_spec(&medium_spec);
    let large_graph = DependencyGraph::from_spec(&large_spec);

    // Benchmark compute_affected (includes transitive closure)
    group.bench_with_input(
        BenchmarkId::new("with_transitive", "medium/type_change"),
        &(
            &medium_graph,
            vec!["Graph".to_string()],
            Vec::<String>::new(),
            Vec::<String>::new(),
        ),
        |b, (graph, types, funcs, props)| {
            b.iter(|| graph.compute_affected(black_box(types), black_box(funcs), black_box(props)))
        },
    );

    group.bench_with_input(
        BenchmarkId::new("with_transitive", "medium/prop_change"),
        &(
            &medium_graph,
            Vec::<String>::new(),
            Vec::<String>::new(),
            vec!["graph_has_nodes".to_string()],
        ),
        |b, (graph, types, funcs, props)| {
            b.iter(|| graph.compute_affected(black_box(types), black_box(funcs), black_box(props)))
        },
    );

    group.bench_with_input(
        BenchmarkId::new("with_transitive", "large/type_change"),
        &(
            &large_graph,
            vec!["Node".to_string()],
            Vec::<String>::new(),
            Vec::<String>::new(),
        ),
        |b, (graph, types, funcs, props)| {
            b.iter(|| graph.compute_affected(black_box(types), black_box(funcs), black_box(props)))
        },
    );

    // Trigger deep transitive closure chain
    group.bench_with_input(
        BenchmarkId::new("with_transitive", "large/deep_chain"),
        &(
            &large_graph,
            Vec::<String>::new(),
            Vec::<String>::new(),
            vec!["t1".to_string()],
        ),
        |b, (graph, types, funcs, props)| {
            b.iter(|| graph.compute_affected(black_box(types), black_box(funcs), black_box(props)))
        },
    );

    group.finish();
}

fn bench_get_dependencies(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_dependencies");

    let medium_spec = parse(MEDIUM_SPEC).unwrap();
    let large_spec = parse(LARGE_SPEC).unwrap();

    let medium_graph = DependencyGraph::from_spec(&medium_spec);
    let large_graph = DependencyGraph::from_spec(&large_spec);

    group.bench_with_input(
        BenchmarkId::new("single_property", "medium"),
        &(&medium_graph, "derived_theorem"),
        |b, (graph, prop)| b.iter(|| graph.get_dependencies(black_box(prop))),
    );

    group.bench_with_input(
        BenchmarkId::new("single_property", "large"),
        &(&large_graph, "t14"),
        |b, (graph, prop)| b.iter(|| graph.get_dependencies(black_box(prop))),
    );

    group.bench_with_input(
        BenchmarkId::new("get_dependents", "medium"),
        &(&medium_graph, "graph_has_nodes"),
        |b, (graph, prop)| b.iter(|| graph.get_dependents(black_box(prop))),
    );

    group.bench_with_input(
        BenchmarkId::new("get_dependents", "large"),
        &(&large_graph, "t1"),
        |b, (graph, prop)| b.iter(|| graph.get_dependents(black_box(prop))),
    );

    group.finish();
}

fn bench_spec_diff(c: &mut Criterion) {
    let mut group = c.benchmark_group("spec_diff");

    let base_spec = parse(MEDIUM_SPEC).unwrap();
    let modified_spec = parse(MEDIUM_SPEC_MODIFIED).unwrap();

    // Self-diff (no changes)
    group.bench_with_input(
        BenchmarkId::new("identical", "medium"),
        &(&base_spec, &base_spec),
        |b, (base, current)| b.iter(|| SpecDiff::diff(black_box(base), black_box(current))),
    );

    // Diff with modifications
    group.bench_with_input(
        BenchmarkId::new("with_changes", "medium"),
        &(&base_spec, &modified_spec),
        |b, (base, current)| b.iter(|| SpecDiff::diff(black_box(base), black_box(current))),
    );

    // all_changed after diff
    let diff = SpecDiff::diff(&base_spec, &modified_spec);
    group.bench_with_input(
        BenchmarkId::new("all_changed", "medium"),
        &diff,
        |b, diff| b.iter(|| black_box(diff).all_changed()),
    );

    group.finish();
}

fn bench_all_properties(c: &mut Criterion) {
    let mut group = c.benchmark_group("enumeration");

    let medium_spec = parse(MEDIUM_SPEC).unwrap();
    let large_spec = parse(LARGE_SPEC).unwrap();

    let medium_graph = DependencyGraph::from_spec(&medium_spec);
    let large_graph = DependencyGraph::from_spec(&large_spec);

    group.bench_with_input(
        BenchmarkId::new("all_properties", "medium"),
        &medium_graph,
        |b, graph| b.iter(|| black_box(graph).all_properties()),
    );

    group.bench_with_input(
        BenchmarkId::new("all_properties", "large"),
        &large_graph,
        |b, graph| b.iter(|| black_box(graph).all_properties()),
    );

    group.bench_with_input(
        BenchmarkId::new("all_referenced_types", "medium"),
        &medium_graph,
        |b, graph| b.iter(|| black_box(graph).all_referenced_types()),
    );

    group.bench_with_input(
        BenchmarkId::new("all_referenced_types", "large"),
        &large_graph,
        |b, graph| b.iter(|| black_box(graph).all_referenced_types()),
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_dependency_graph_construction,
    bench_affected_properties,
    bench_compute_affected,
    bench_get_dependencies,
    bench_spec_diff,
    bench_all_properties,
);

criterion_main!(benches);
