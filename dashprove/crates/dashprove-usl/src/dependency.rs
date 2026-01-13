//! Dependency analysis for USL specifications
//!
//! This module provides tools for analyzing dependencies between types and
//! properties in a USL specification. This enables incremental verification
//! by determining which properties need re-verification when parts of the
//! specification change.
//!
//! # Architecture
//!
//! The dependency graph tracks:
//! - Which types are referenced by each property
//! - Which functions/predicates are referenced by each property
//! - Which properties reference other properties (for lemma dependencies)
//!
//! # Example
//!
//! ```rust
//! use dashprove_usl::{parse, typecheck, DependencyGraph};
//!
//! let spec = r#"
//!     type Graph = { nodes: Set<Node>, edges: Set<Edge> }
//!     type Node = { id: Int }
//!
//!     theorem graph_has_nodes {
//!         forall g: Graph . g.nodes.len() >= 0
//!     }
//! "#;
//!
//! let parsed = parse(spec).unwrap();
//! let typed = typecheck(parsed).unwrap();
//! let deps = DependencyGraph::from_spec(&typed.spec);
//!
//! // The theorem depends on Graph and Node types
//! let affected = deps.properties_affected_by(&["Graph".to_string()]);
//! assert!(affected.contains(&"graph_has_nodes".to_string()));
//! ```

use crate::ast::{
    Bisimulation, Contract, Expr, Invariant, Param, PlatformApi, Probabilistic, Property,
    Refinement, Security, SemanticProperty, Spec, Temporal, TemporalExpr, Theorem, Type, TypeDef,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A dependency graph for a USL specification
///
/// Tracks which definitions (types, functions, properties) each property depends on.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DependencyGraph {
    /// Map from property name to its dependencies
    property_deps: HashMap<String, PropertyDependencies>,
    /// Map from type name to properties that reference it
    type_to_properties: HashMap<String, HashSet<String>>,
    /// Map from function name to properties that use it
    function_to_properties: HashMap<String, HashSet<String>>,
    /// All type names defined in the spec
    defined_types: HashSet<String>,
    /// All property names defined in the spec
    defined_properties: HashSet<String>,
}

/// Dependencies for a single property
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PropertyDependencies {
    /// Types referenced by this property
    pub types: HashSet<String>,
    /// Functions/predicates used by this property
    pub functions: HashSet<String>,
    /// Other properties referenced (for lemma applications)
    pub properties: HashSet<String>,
    /// Variables bound in this property
    pub bound_vars: HashSet<String>,
}

impl DependencyGraph {
    /// Create a new empty dependency graph
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a dependency graph from a specification
    #[must_use]
    pub fn from_spec(spec: &Spec) -> Self {
        let mut graph = Self::new();

        // First pass: record all defined type names
        for type_def in &spec.types {
            graph.defined_types.insert(type_def.name.clone());
        }

        // First pass: record all defined property names
        for property in &spec.properties {
            graph.defined_properties.insert(property.name());
        }

        // Second pass: analyze each property with knowledge of all defined names
        for property in &spec.properties {
            let name = property.name();
            let deps = graph.analyze_property(property);

            // Update reverse mappings
            for type_name in &deps.types {
                graph
                    .type_to_properties
                    .entry(type_name.clone())
                    .or_default()
                    .insert(name.clone());
            }

            for func_name in &deps.functions {
                graph
                    .function_to_properties
                    .entry(func_name.clone())
                    .or_default()
                    .insert(name.clone());
            }

            graph.property_deps.insert(name, deps);
        }

        graph
    }

    /// Get dependencies for a specific property
    #[must_use]
    pub fn get_dependencies(&self, property_name: &str) -> Option<&PropertyDependencies> {
        self.property_deps.get(property_name)
    }

    /// Get all properties that depend on given types
    ///
    /// Returns set of property names that reference any of the given type names.
    #[must_use]
    pub fn properties_affected_by(&self, changed_types: &[String]) -> HashSet<String> {
        let mut affected = HashSet::new();

        for type_name in changed_types {
            if let Some(props) = self.type_to_properties.get(type_name) {
                affected.extend(props.iter().cloned());
            }
        }

        affected
    }

    /// Get all properties that use given functions
    #[must_use]
    pub fn properties_using_functions(&self, functions: &[String]) -> HashSet<String> {
        let mut affected = HashSet::new();

        for func_name in functions {
            if let Some(props) = self.function_to_properties.get(func_name) {
                affected.extend(props.iter().cloned());
            }
        }

        affected
    }

    /// Determine affected properties given a set of changes
    ///
    /// This combines type changes, function changes, and direct property changes
    /// to produce a complete set of properties that need re-verification.
    /// Includes transitive closure: if property A depends on property B, and B is affected,
    /// then A is also affected.
    #[must_use]
    pub fn compute_affected(
        &self,
        changed_types: &[String],
        changed_functions: &[String],
        changed_properties: &[String],
    ) -> HashSet<String> {
        let mut affected = HashSet::new();

        // Properties affected by type changes
        affected.extend(self.properties_affected_by(changed_types));

        // Properties affected by function changes
        affected.extend(self.properties_using_functions(changed_functions));

        // Directly changed properties
        affected.extend(changed_properties.iter().cloned());

        // Transitive closure: if property A depends on property B, and B is affected, A is affected
        self.compute_transitive_closure(&mut affected);

        affected
    }

    /// Compute the transitive closure of affected properties
    ///
    /// If property A references property B (e.g., uses it as a lemma), and B is in the
    /// affected set, then A should also be added to the affected set. This repeats until
    /// no new properties are added.
    fn compute_transitive_closure(&self, affected: &mut HashSet<String>) {
        loop {
            let mut newly_affected = HashSet::new();

            // For each property, check if any of its property dependencies are affected
            for (prop_name, deps) in &self.property_deps {
                if affected.contains(prop_name) {
                    continue; // Already affected
                }

                // Check if any property this one depends on is affected
                for dep_prop in &deps.properties {
                    if affected.contains(dep_prop) {
                        newly_affected.insert(prop_name.clone());
                        break;
                    }
                }
            }

            if newly_affected.is_empty() {
                break; // Fixed point reached
            }

            affected.extend(newly_affected);
        }
    }

    /// Get direct property dependencies for a property
    ///
    /// Returns the set of property names that the given property directly references.
    #[must_use]
    pub fn get_property_dependencies(&self, property_name: &str) -> Option<&HashSet<String>> {
        self.property_deps
            .get(property_name)
            .map(|deps| &deps.properties)
    }

    /// Get all properties that directly depend on the given property
    ///
    /// Returns the set of property names that reference the given property.
    #[must_use]
    pub fn get_dependents(&self, property_name: &str) -> HashSet<String> {
        let mut dependents = HashSet::new();
        for (name, deps) in &self.property_deps {
            if deps.properties.contains(property_name) {
                dependents.insert(name.clone());
            }
        }
        dependents
    }

    /// Get all property names in the graph
    #[must_use]
    pub fn all_properties(&self) -> Vec<String> {
        self.property_deps.keys().cloned().collect()
    }

    /// Get all type names that are used by properties
    #[must_use]
    pub fn all_referenced_types(&self) -> Vec<String> {
        self.type_to_properties.keys().cloned().collect()
    }

    /// Check if a property depends on a specific type
    #[must_use]
    pub fn property_depends_on_type(&self, property: &str, type_name: &str) -> bool {
        self.property_deps
            .get(property)
            .is_some_and(|deps| deps.types.contains(type_name))
    }

    // ========== Internal analysis methods ==========

    fn analyze_property(&self, property: &Property) -> PropertyDependencies {
        let mut deps = PropertyDependencies::default();

        match property {
            Property::Theorem(t) => self.analyze_theorem(t, &mut deps),
            Property::Temporal(t) => self.analyze_temporal(t, &mut deps),
            Property::Contract(c) => self.analyze_contract(c, &mut deps),
            Property::Invariant(i) => self.analyze_invariant(i, &mut deps),
            Property::Refinement(r) => self.analyze_refinement(r, &mut deps),
            Property::Probabilistic(p) => self.analyze_probabilistic(p, &mut deps),
            Property::Security(s) => self.analyze_security(s, &mut deps),
            Property::Semantic(s) => self.analyze_semantic(s, &mut deps),
            Property::PlatformApi(p) => self.analyze_platform_api(p, &mut deps),
            Property::Bisimulation(b) => self.analyze_bisimulation(b, &mut deps),
            Property::Version(v) => self.analyze_version_spec(v, &mut deps),
            Property::Capability(c) => self.analyze_capability_spec(c, &mut deps),
            Property::DistributedInvariant(d) => self.analyze_expr(&d.body, &mut deps),
            Property::DistributedTemporal(d) => self.analyze_temporal_expr(&d.body, &mut deps),
            Property::Composed(c) => self.analyze_composed_theorem(c, &mut deps),
            Property::ImprovementProposal(i) => self.analyze_improvement_proposal(i, &mut deps),
            Property::VerificationGate(v) => self.analyze_verification_gate(v, &mut deps),
            Property::Rollback(r) => self.analyze_rollback_spec(r, &mut deps),
        }

        deps
    }

    fn analyze_theorem(&self, theorem: &Theorem, deps: &mut PropertyDependencies) {
        self.analyze_expr(&theorem.body, deps);
    }

    fn analyze_composed_theorem(
        &self,
        composed: &crate::ast::ComposedTheorem,
        deps: &mut PropertyDependencies,
    ) {
        // Composed theorems depend on other properties
        for used_prop in &composed.uses {
            deps.properties.insert(used_prop.clone());
        }
        // Also analyze the body expression
        self.analyze_expr(&composed.body, deps);
    }

    fn analyze_improvement_proposal(
        &self,
        proposal: &crate::ast::ImprovementProposal,
        deps: &mut PropertyDependencies,
    ) {
        // Analyze target expression
        self.analyze_expr(&proposal.target, deps);
        // Analyze improves expressions
        for expr in &proposal.improves {
            self.analyze_expr(expr, deps);
        }
        // Analyze preserves expressions
        for expr in &proposal.preserves {
            self.analyze_expr(expr, deps);
        }
        // Analyze requires expressions
        for expr in &proposal.requires {
            self.analyze_expr(expr, deps);
        }
    }

    fn analyze_verification_gate(
        &self,
        gate: &crate::ast::VerificationGate,
        deps: &mut PropertyDependencies,
    ) {
        // Analyze input parameter types
        for param in &gate.inputs {
            self.analyze_type(&param.ty, deps);
        }
        // Analyze check conditions
        for check in &gate.checks {
            self.analyze_expr(&check.condition, deps);
        }
        // Analyze on_pass and on_fail expressions
        self.analyze_expr(&gate.on_pass, deps);
        self.analyze_expr(&gate.on_fail, deps);
    }

    fn analyze_rollback_spec(
        &self,
        rollback: &crate::ast::RollbackSpec,
        deps: &mut PropertyDependencies,
    ) {
        // Analyze state parameter types
        for param in &rollback.state {
            self.analyze_type(&param.ty, deps);
        }
        // Analyze invariants
        for expr in &rollback.invariants {
            self.analyze_expr(expr, deps);
        }
        // Analyze trigger
        self.analyze_expr(&rollback.trigger, deps);
        // Analyze action assignments and ensure
        for (_, expr) in &rollback.action.assignments {
            self.analyze_expr(expr, deps);
        }
        if let Some(ensure) = &rollback.action.ensure {
            self.analyze_expr(ensure, deps);
        }
        // Analyze guarantees
        for expr in &rollback.guarantees {
            self.analyze_expr(expr, deps);
        }
    }

    fn analyze_temporal(&self, temporal: &Temporal, deps: &mut PropertyDependencies) {
        self.analyze_temporal_expr(&temporal.body, deps);
    }

    fn analyze_contract(&self, contract: &Contract, deps: &mut PropertyDependencies) {
        // Contract type path references types
        if let Some(type_name) = contract.type_path.first() {
            deps.types.insert(type_name.clone());
        }

        // Analyze parameters
        for param in &contract.params {
            self.analyze_param(param, deps);
        }

        // Analyze return type
        if let Some(ret_ty) = &contract.return_type {
            self.analyze_type(ret_ty, deps);
        }

        // Analyze requires/ensures clauses
        for expr in &contract.requires {
            self.analyze_expr(expr, deps);
        }
        for expr in &contract.ensures {
            self.analyze_expr(expr, deps);
        }
        for expr in &contract.ensures_err {
            self.analyze_expr(expr, deps);
        }
    }

    fn analyze_invariant(&self, invariant: &Invariant, deps: &mut PropertyDependencies) {
        self.analyze_expr(&invariant.body, deps);
    }

    fn analyze_refinement(&self, refinement: &Refinement, deps: &mut PropertyDependencies) {
        // The refines clause references another spec/type
        deps.types.insert(refinement.refines.clone());

        self.analyze_expr(&refinement.abstraction, deps);
        self.analyze_expr(&refinement.simulation, deps);
    }

    fn analyze_probabilistic(&self, prob: &Probabilistic, deps: &mut PropertyDependencies) {
        self.analyze_expr(&prob.condition, deps);
    }

    fn analyze_security(&self, security: &Security, deps: &mut PropertyDependencies) {
        self.analyze_expr(&security.body, deps);
    }

    fn analyze_semantic(&self, semantic: &SemanticProperty, deps: &mut PropertyDependencies) {
        self.analyze_expr(&semantic.body, deps);
    }

    fn analyze_platform_api(&self, api: &PlatformApi, deps: &mut PropertyDependencies) {
        for state in &api.states {
            // Analyze invariants
            for inv in &state.invariants {
                self.analyze_expr(inv, deps);
            }
            // Analyze transitions
            for transition in &state.transitions {
                for req in &transition.requires {
                    self.analyze_expr(req, deps);
                }
                for ens in &transition.ensures {
                    self.analyze_expr(ens, deps);
                }
            }
        }
    }

    fn analyze_bisimulation(&self, bisim: &Bisimulation, deps: &mut PropertyDependencies) {
        // Analyze property expression if present
        if let Some(ref prop) = bisim.property {
            // The var_type might reference a type
            self.analyze_type(&prop.var_type, deps);
            // Analyze the oracle and subject expressions
            self.analyze_expr(&prop.oracle_expr, deps);
            self.analyze_expr(&prop.subject_expr, deps);
        }
    }

    fn analyze_version_spec(
        &self,
        version: &crate::ast::VersionSpec,
        deps: &mut PropertyDependencies,
    ) {
        // Analyze capability expressions
        for cap in &version.capabilities {
            self.analyze_expr(&cap.expr, deps);
        }
        // Analyze preserves expressions
        for pres in &version.preserves {
            self.analyze_expr(&pres.property, deps);
        }
    }

    fn analyze_capability_spec(
        &self,
        capability: &crate::ast::CapabilitySpec,
        deps: &mut PropertyDependencies,
    ) {
        // Analyze ability signatures
        for ability in &capability.abilities {
            for param in &ability.params {
                self.analyze_type(&param.ty, deps);
            }
            if let Some(ref ret) = ability.return_type {
                self.analyze_type(ret, deps);
            }
        }
        // Analyze requires expressions
        for req in &capability.requires {
            self.analyze_expr(req, deps);
        }
    }

    fn analyze_temporal_expr(&self, expr: &TemporalExpr, deps: &mut PropertyDependencies) {
        match expr {
            TemporalExpr::Always(inner) | TemporalExpr::Eventually(inner) => {
                self.analyze_temporal_expr(inner, deps);
            }
            TemporalExpr::LeadsTo(lhs, rhs) => {
                self.analyze_temporal_expr(lhs, deps);
                self.analyze_temporal_expr(rhs, deps);
            }
            TemporalExpr::Atom(expr) => {
                self.analyze_expr(expr, deps);
            }
        }
    }

    fn analyze_expr(&self, expr: &Expr, deps: &mut PropertyDependencies) {
        match expr {
            Expr::Var(name) => {
                // If it's a defined type name, add as type dependency
                if self.defined_types.contains(name) {
                    deps.types.insert(name.clone());
                }
                // If it's a defined property name, add as property dependency
                // (e.g., referencing a theorem by name without arguments)
                if self.defined_properties.contains(name) {
                    deps.properties.insert(name.clone());
                }
            }
            Expr::Int(_) | Expr::Float(_) | Expr::String(_) | Expr::Bool(_) => {}
            Expr::ForAll { var, ty, body } | Expr::Exists { var, ty, body } => {
                deps.bound_vars.insert(var.clone());
                if let Some(t) = ty {
                    self.analyze_type(t, deps);
                }
                self.analyze_expr(body, deps);
            }
            Expr::ForAllIn {
                var,
                collection,
                body,
            }
            | Expr::ExistsIn {
                var,
                collection,
                body,
            } => {
                deps.bound_vars.insert(var.clone());
                self.analyze_expr(collection, deps);
                self.analyze_expr(body, deps);
            }
            Expr::Implies(lhs, rhs)
            | Expr::And(lhs, rhs)
            | Expr::Or(lhs, rhs)
            | Expr::Compare(lhs, _, rhs)
            | Expr::Binary(lhs, _, rhs) => {
                self.analyze_expr(lhs, deps);
                self.analyze_expr(rhs, deps);
            }
            Expr::Not(inner) | Expr::Neg(inner) => {
                self.analyze_expr(inner, deps);
            }
            Expr::App(func_name, args) => {
                // Check if this is a reference to another property (lemma application)
                if self.defined_properties.contains(func_name) {
                    deps.properties.insert(func_name.clone());
                } else {
                    deps.functions.insert(func_name.clone());
                }
                for arg in args {
                    self.analyze_expr(arg, deps);
                }
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                deps.functions.insert(method.clone());
                self.analyze_expr(receiver, deps);
                for arg in args {
                    self.analyze_expr(arg, deps);
                }
            }
            Expr::FieldAccess(object, _field) => {
                self.analyze_expr(object, deps);
            }
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    fn analyze_type(&self, ty: &Type, deps: &mut PropertyDependencies) {
        match ty {
            Type::Named(name) => {
                deps.types.insert(name.clone());
            }
            Type::Set(inner) | Type::List(inner) | Type::Result(inner) => {
                self.analyze_type(inner, deps);
            }
            Type::Map(key, val) | Type::Relation(key, val) | Type::Function(key, val) => {
                self.analyze_type(key, deps);
                self.analyze_type(val, deps);
            }
            Type::Graph(n, e) => {
                self.analyze_type(n, deps);
                self.analyze_type(e, deps);
            }
            Type::Path(n) => {
                self.analyze_type(n, deps);
            }
            Type::Unit => {}
        }
    }

    fn analyze_param(&self, param: &Param, deps: &mut PropertyDependencies) {
        deps.bound_vars.insert(param.name.clone());
        self.analyze_type(&param.ty, deps);
    }
}

/// Compare two specifications to find what changed
#[derive(Debug, Clone, Default)]
pub struct SpecDiff {
    /// Types that were added
    pub added_types: Vec<String>,
    /// Types that were removed
    pub removed_types: Vec<String>,
    /// Types that were modified
    pub modified_types: Vec<String>,
    /// Properties that were added
    pub added_properties: Vec<String>,
    /// Properties that were removed
    pub removed_properties: Vec<String>,
    /// Properties that were modified
    pub modified_properties: Vec<String>,
}

impl SpecDiff {
    /// Compute the difference between two specifications
    #[must_use]
    pub fn diff(base: &Spec, current: &Spec) -> Self {
        let mut diff = Self::default();

        // Compare types
        let base_types: HashMap<_, _> = base.types.iter().map(|t| (&t.name, t)).collect();
        let current_types: HashMap<_, _> = current.types.iter().map(|t| (&t.name, t)).collect();

        for name in current_types.keys() {
            if !base_types.contains_key(name) {
                diff.added_types.push((*name).clone());
            }
        }

        for (name, base_def) in &base_types {
            match current_types.get(name) {
                None => diff.removed_types.push((*name).clone()),
                Some(current_def) => {
                    if !Self::types_equal(base_def, current_def) {
                        diff.modified_types.push((*name).clone());
                    }
                }
            }
        }

        // Compare properties
        let base_props: HashMap<_, _> = base.properties.iter().map(|p| (p.name(), p)).collect();
        let current_props: HashMap<_, _> =
            current.properties.iter().map(|p| (p.name(), p)).collect();

        for name in current_props.keys() {
            if !base_props.contains_key(name) {
                diff.added_properties.push(name.clone());
            }
        }

        for (name, base_prop) in &base_props {
            match current_props.get(name) {
                None => diff.removed_properties.push(name.clone()),
                Some(current_prop) => {
                    if *base_prop != *current_prop {
                        diff.modified_properties.push(name.clone());
                    }
                }
            }
        }

        diff
    }

    /// Check if two type definitions are equal
    fn types_equal(a: &TypeDef, b: &TypeDef) -> bool {
        a == b
    }

    /// Get all changed items (types and properties)
    #[must_use]
    pub fn all_changed(&self) -> Vec<String> {
        let mut changed = Vec::new();
        changed.extend(self.added_types.iter().cloned());
        changed.extend(self.removed_types.iter().cloned());
        changed.extend(self.modified_types.iter().cloned());
        changed.extend(self.added_properties.iter().cloned());
        changed.extend(self.removed_properties.iter().cloned());
        changed.extend(self.modified_properties.iter().cloned());
        changed
    }

    /// Check if there are any changes
    #[must_use]
    pub fn has_changes(&self) -> bool {
        !self.added_types.is_empty()
            || !self.removed_types.is_empty()
            || !self.modified_types.is_empty()
            || !self.added_properties.is_empty()
            || !self.removed_properties.is_empty()
            || !self.modified_properties.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, typecheck};

    fn build_graph(input: &str) -> DependencyGraph {
        let spec = parse(input).expect("parse failed");
        let typed = typecheck(spec).expect("typecheck failed");
        DependencyGraph::from_spec(&typed.spec)
    }

    #[test]
    fn test_theorem_depends_on_type() {
        let graph = build_graph(
            r#"
            type Node = { id: Int }

            theorem node_has_id {
                forall n: Node . n.id >= 0
            }
        "#,
        );

        let deps = graph.get_dependencies("node_has_id").unwrap();
        assert!(deps.types.contains("Node"));
    }

    #[test]
    fn test_affected_by_type_change() {
        let graph = build_graph(
            r#"
            type Graph = { nodes: Set<Node> }
            type Node = { id: Int }

            theorem graph_nodes_exist {
                forall g: Graph . g.nodes.len() >= 0
            }

            theorem unrelated {
                forall x: Int . x == x
            }
        "#,
        );

        let affected = graph.properties_affected_by(&["Graph".to_string()]);
        assert!(affected.contains("graph_nodes_exist"));
        assert!(!affected.contains("unrelated"));
    }

    #[test]
    fn test_function_dependency() {
        let graph = build_graph(
            r#"
            theorem uses_func {
                forall x: Int . is_positive(x) implies x > 0
            }
        "#,
        );

        let deps = graph.get_dependencies("uses_func").unwrap();
        assert!(deps.functions.contains("is_positive"));
    }

    #[test]
    fn test_contract_type_dependency() {
        let graph = build_graph(
            r#"
            type Counter = { value: Int }

            contract Counter::increment(self: Counter, amount: Int) -> Result<()> {
                requires { amount > 0 }
                ensures { self'.value == self.value + amount }
            }
        "#,
        );

        let deps = graph.get_dependencies("Counter::increment").unwrap();
        assert!(deps.types.contains("Counter"));
        assert!(deps.types.contains("Int"));
    }

    #[test]
    fn test_invariant_dependency() {
        let graph = build_graph(
            r#"
            type State = { count: Int }

            invariant count_positive {
                forall s: State . s.count >= 0
            }
        "#,
        );

        let deps = graph.get_dependencies("count_positive").unwrap();
        assert!(deps.types.contains("State"));
    }

    #[test]
    fn test_temporal_dependency() {
        let graph = build_graph(
            r#"
            type Process = { running: Bool }

            temporal eventually_stops {
                eventually(not running)
            }
        "#,
        );

        let _deps = graph.get_dependencies("eventually_stops").unwrap();
        // Note: 'running' here is a variable, not a type
        // The temporal property parses 'running' as a predicate reference
    }

    #[test]
    fn test_spec_diff_added_type() {
        let base = parse("theorem t { true }").unwrap();
        let current = parse(
            r#"
            type NewType = { x: Int }
            theorem t { true }
        "#,
        )
        .unwrap();

        let diff = SpecDiff::diff(&base, &current);
        assert!(diff.added_types.contains(&"NewType".to_string()));
        assert!(diff.removed_types.is_empty());
        assert!(diff.modified_types.is_empty());
    }

    #[test]
    fn test_spec_diff_removed_property() {
        let base = parse(
            r#"
            theorem t1 { true }
            theorem t2 { false }
        "#,
        )
        .unwrap();
        let current = parse("theorem t1 { true }").unwrap();

        let diff = SpecDiff::diff(&base, &current);
        assert!(diff.removed_properties.contains(&"t2".to_string()));
        assert!(diff.added_properties.is_empty());
    }

    #[test]
    fn test_spec_diff_modified_property() {
        let base = parse("theorem t { true }").unwrap();
        let current = parse("theorem t { false }").unwrap();

        let diff = SpecDiff::diff(&base, &current);
        assert!(diff.modified_properties.contains(&"t".to_string()));
    }

    #[test]
    fn test_compute_affected_comprehensive() {
        let graph = build_graph(
            r#"
            type A = { x: Int }
            type B = { y: Int }

            theorem uses_a { forall a: A . a.x >= 0 }
            theorem uses_b { forall b: B . b.y >= 0 }
            theorem uses_both { forall a: A, b: B . a.x + b.y >= 0 }
            theorem independent { forall x: Int . x == x }
        "#,
        );

        let affected = graph.compute_affected(&["A".to_string()], &[], &[]);

        assert!(affected.contains("uses_a"));
        assert!(affected.contains("uses_both"));
        assert!(!affected.contains("uses_b"));
        assert!(!affected.contains("independent"));
    }

    #[test]
    fn test_nested_type_dependency() {
        let graph = build_graph(
            r#"
            type Inner = { val: Int }
            type Outer = { items: Set<Inner> }

            theorem outer_prop {
                forall o: Outer . o.items.len() >= 0
            }
        "#,
        );

        let deps = graph.get_dependencies("outer_prop").unwrap();
        assert!(deps.types.contains("Outer"));
        // Note: Inner is referenced through Outer's definition,
        // but our simple analysis only captures direct type references in the property body.
        // For full transitive dependencies, we'd need to analyze type definitions too.
    }

    #[test]
    fn test_property_references_another_property() {
        let graph = build_graph(
            r#"
            theorem base_lemma {
                forall x: Int . x == x
            }

            theorem uses_lemma {
                forall y: Int . base_lemma(y) implies y == y
            }
        "#,
        );

        let deps = graph.get_dependencies("uses_lemma").unwrap();
        assert!(deps.properties.contains("base_lemma"));
        assert!(!deps.functions.contains("base_lemma")); // Should be in properties, not functions
    }

    #[test]
    fn test_transitive_closure_simple() {
        let graph = build_graph(
            r#"
            theorem lemma_a {
                forall x: Int . x >= 0
            }

            theorem lemma_b {
                forall y: Int . lemma_a(y) implies y >= 0
            }

            theorem theorem_c {
                forall z: Int . lemma_b(z) implies z >= 0
            }
        "#,
        );

        // When lemma_a changes, both lemma_b and theorem_c should be affected
        let affected = graph.compute_affected(&[], &[], &["lemma_a".to_string()]);

        assert!(affected.contains("lemma_a")); // Directly changed
        assert!(affected.contains("lemma_b")); // Depends on lemma_a
        assert!(affected.contains("theorem_c")); // Transitively depends on lemma_a via lemma_b
    }

    #[test]
    fn test_transitive_closure_chain() {
        let graph = build_graph(
            r#"
            theorem t1 { true }
            theorem t2 { t1 implies true }
            theorem t3 { t2 implies true }
            theorem t4 { t3 implies true }
            theorem independent { false implies false }
        "#,
        );

        // When t1 changes, t2, t3, and t4 should all be affected (transitive chain)
        let affected = graph.compute_affected(&[], &[], &["t1".to_string()]);

        assert!(affected.contains("t1"));
        assert!(affected.contains("t2"));
        assert!(affected.contains("t3"));
        assert!(affected.contains("t4"));
        assert!(!affected.contains("independent")); // Does not depend on the chain
    }

    #[test]
    fn test_transitive_closure_diamond() {
        // Diamond dependency pattern:
        //     t1
        //    /  \
        //   t2  t3
        //    \  /
        //     t4
        let graph = build_graph(
            r#"
            theorem t1 { true }
            theorem t2 { t1 implies true }
            theorem t3 { t1 implies true }
            theorem t4 { t2 and t3 }
        "#,
        );

        // When t1 changes, all should be affected
        let affected = graph.compute_affected(&[], &[], &["t1".to_string()]);

        assert!(affected.contains("t1"));
        assert!(affected.contains("t2"));
        assert!(affected.contains("t3"));
        assert!(affected.contains("t4"));
    }

    #[test]
    fn test_get_property_dependencies() {
        let graph = build_graph(
            r#"
            theorem base { true }
            theorem derived { base implies true }
        "#,
        );

        let deps = graph.get_property_dependencies("derived").unwrap();
        assert!(deps.contains("base"));
        assert_eq!(deps.len(), 1);

        let base_deps = graph.get_property_dependencies("base").unwrap();
        assert!(base_deps.is_empty());
    }

    #[test]
    fn test_get_dependents() {
        let graph = build_graph(
            r#"
            theorem base { true }
            theorem derived1 { base implies true }
            theorem derived2 { base implies false }
            theorem unrelated { false }
        "#,
        );

        let dependents = graph.get_dependents("base");
        assert!(dependents.contains("derived1"));
        assert!(dependents.contains("derived2"));
        assert!(!dependents.contains("unrelated"));
        assert_eq!(dependents.len(), 2);
    }

    #[test]
    fn test_transitive_closure_with_type_change() {
        let graph = build_graph(
            r#"
            type Counter = { value: Int }

            theorem counter_non_negative {
                forall c: Counter . c.value >= 0
            }

            theorem derived_from_counter {
                forall x: Int . counter_non_negative(x) implies x >= 0
            }
        "#,
        );

        // When Counter type changes:
        // - counter_non_negative is affected (uses Counter)
        // - derived_from_counter is affected transitively (depends on counter_non_negative)
        let affected = graph.compute_affected(&["Counter".to_string()], &[], &[]);

        assert!(affected.contains("counter_non_negative"));
        assert!(affected.contains("derived_from_counter"));
    }

    #[test]
    fn test_no_self_dependency() {
        let graph = build_graph(
            r#"
            theorem recursive_looking {
                forall x: Int . x == x
            }
        "#,
        );

        let deps = graph.get_dependencies("recursive_looking").unwrap();
        // Property should not depend on itself
        assert!(!deps.properties.contains("recursive_looking"));
    }

    #[test]
    fn test_spec_diff_has_changes_added_type() {
        // Tests has_changes detects added_types
        let mut diff = SpecDiff::default();
        assert!(!diff.has_changes());
        diff.added_types.push("NewType".to_string());
        assert!(diff.has_changes());
    }

    #[test]
    fn test_spec_diff_has_changes_removed_type() {
        // Tests has_changes detects removed_types
        let mut diff = SpecDiff::default();
        diff.removed_types.push("OldType".to_string());
        assert!(diff.has_changes());
    }

    #[test]
    fn test_spec_diff_has_changes_modified_type() {
        // Tests has_changes detects modified_types
        let mut diff = SpecDiff::default();
        diff.modified_types.push("ChangedType".to_string());
        assert!(diff.has_changes());
    }

    #[test]
    fn test_spec_diff_has_changes_added_property() {
        // Tests has_changes detects added_properties
        let mut diff = SpecDiff::default();
        diff.added_properties.push("new_prop".to_string());
        assert!(diff.has_changes());
    }

    #[test]
    fn test_spec_diff_has_changes_removed_property() {
        // Tests has_changes detects removed_properties
        let mut diff = SpecDiff::default();
        diff.removed_properties.push("old_prop".to_string());
        assert!(diff.has_changes());
    }

    #[test]
    fn test_spec_diff_has_changes_modified_property() {
        // Tests has_changes detects modified_properties
        let mut diff = SpecDiff::default();
        diff.modified_properties.push("changed_prop".to_string());
        assert!(diff.has_changes());
    }

    #[test]
    fn test_spec_diff_all_changed() {
        // Tests all_changed collects from all fields
        let mut diff = SpecDiff::default();
        diff.added_types.push("at".to_string());
        diff.removed_types.push("rt".to_string());
        diff.modified_types.push("mt".to_string());
        diff.added_properties.push("ap".to_string());
        diff.removed_properties.push("rp".to_string());
        diff.modified_properties.push("mp".to_string());

        let changed = diff.all_changed();
        assert_eq!(changed.len(), 6);
        assert!(changed.contains(&"at".to_string()));
        assert!(changed.contains(&"rt".to_string()));
        assert!(changed.contains(&"mt".to_string()));
        assert!(changed.contains(&"ap".to_string()));
        assert!(changed.contains(&"rp".to_string()));
        assert!(changed.contains(&"mp".to_string()));
    }

    #[test]
    fn test_spec_diff_types_equal() {
        let spec1 = parse("type Foo = { x: Int }").expect("parse");
        let spec2 = parse("type Foo = { x: Int }").expect("parse");
        let spec3 = parse("type Foo = { y: Bool }").expect("parse");

        assert!(SpecDiff::types_equal(&spec1.types[0], &spec2.types[0]));
        assert!(!SpecDiff::types_equal(&spec1.types[0], &spec3.types[0]));
    }

    #[test]
    fn test_all_properties() {
        let graph = build_graph(
            r#"
            theorem prop1 { true }
            invariant prop2 { x > 0 }
        "#,
        );

        let props = graph.all_properties();
        assert!(props.contains(&"prop1".to_string()));
        assert!(props.contains(&"prop2".to_string()));
        assert_eq!(props.len(), 2);
    }

    #[test]
    fn test_all_referenced_types() {
        // Type is only in all_referenced_types if a property references it
        let graph = build_graph(
            r#"
            type MyType = { field: Int }
            theorem prop_with_type { forall x: MyType . x.field > 0 }
        "#,
        );

        let types = graph.all_referenced_types();
        // MyType should be in referenced types because prop_with_type uses it
        assert!(types.contains(&"MyType".to_string()));
    }

    #[test]
    fn test_properties_using_functions_empty() {
        // Test that properties_using_functions returns empty for no matching functions
        let graph = build_graph(
            r#"
            theorem simple { true }
        "#,
        );

        let fns = vec!["nonexistent".to_string()];
        let using_fns = graph.properties_using_functions(&fns);
        assert!(using_fns.is_empty());
    }

    #[test]
    fn test_property_depends_on_type() {
        let graph = build_graph(
            r#"
            type Counter = { value: Int }
            theorem counter_prop { forall c: Counter . c.value >= 0 }
        "#,
        );

        assert!(graph.property_depends_on_type("counter_prop", "Counter"));
        assert!(!graph.property_depends_on_type("counter_prop", "NonExistent"));
    }

    // ============== Mutation-killing tests for has_changes || -> && mutations ==============
    // These tests verify that has_changes returns true when EXACTLY ONE field is non-empty.
    // If || is mutated to &&, these tests would fail because && requires ALL fields non-empty.

    #[test]
    fn test_has_changes_only_added_types() {
        // Kill mutation: line 579 || -> && (added_types only)
        let mut diff = SpecDiff::default();
        diff.added_types.push("T".to_string());
        // All other fields empty - if || became &&, this would return false
        assert!(diff.has_changes());
    }

    #[test]
    fn test_has_changes_only_removed_types() {
        // Kill mutation: line 580 || -> && (removed_types only)
        let mut diff = SpecDiff::default();
        diff.removed_types.push("T".to_string());
        assert!(diff.has_changes());
    }

    #[test]
    fn test_has_changes_only_modified_types() {
        // Kill mutation: line 581 || -> && (modified_types only)
        let mut diff = SpecDiff::default();
        diff.modified_types.push("T".to_string());
        assert!(diff.has_changes());
    }

    #[test]
    fn test_has_changes_only_added_properties() {
        // Kill mutation: line 582 || -> && (added_properties only)
        let mut diff = SpecDiff::default();
        diff.added_properties.push("p".to_string());
        assert!(diff.has_changes());
    }

    #[test]
    fn test_has_changes_only_removed_properties() {
        // Kill mutation: line 583 || -> && (removed_properties only)
        let mut diff = SpecDiff::default();
        diff.removed_properties.push("p".to_string());
        assert!(diff.has_changes());
    }

    #[test]
    fn test_has_changes_only_modified_properties() {
        // Kill mutation: line 584 || -> && (modified_properties only)
        let mut diff = SpecDiff::default();
        diff.modified_properties.push("p".to_string());
        assert!(diff.has_changes());
    }

    // ============== Mutation-killing tests for has_changes delete ! mutations ==============
    // These test that empty() is negated: !empty() means "has items"
    // If ! is deleted, empty() alone would be truthy when there are NO items

    #[test]
    fn test_has_changes_false_when_all_empty() {
        // Kill mutation: delete ! on any field - if ! deleted, empty fields would return true
        let diff = SpecDiff::default();
        assert!(!diff.has_changes()); // Must be false when all empty
    }

    // ============== Mutation-killing tests for types_equal ==============

    #[test]
    fn test_types_equal_same_type_returns_true() {
        // Kill mutation: replace types_equal -> bool with false
        let spec = parse("type Foo = { x: Int }").expect("parse");
        assert!(SpecDiff::types_equal(&spec.types[0], &spec.types[0]));
    }

    #[test]
    fn test_types_equal_different_types_returns_false() {
        // Kill mutation: replace types_equal -> bool with true
        // Kill mutation: replace == with !=
        let spec1 = parse("type Foo = { x: Int }").expect("parse");
        let spec2 = parse("type Bar = { y: Bool }").expect("parse");
        assert!(!SpecDiff::types_equal(&spec1.types[0], &spec2.types[0]));
    }

    // ============== Mutation-killing tests for all_changed ==============

    #[test]
    fn test_all_changed_returns_correct_count() {
        // Kill mutation: replace all_changed -> Vec<String> with vec![]
        let mut diff = SpecDiff::default();
        diff.added_types.push("a".to_string());
        let changed = diff.all_changed();
        assert!(!changed.is_empty()); // vec![] would fail this
        assert_eq!(changed.len(), 1);
    }

    #[test]
    fn test_all_changed_returns_correct_values() {
        // Kill mutation: replace with vec![String::new()] or vec!["xyzzy".into()]
        let mut diff = SpecDiff::default();
        diff.added_types.push("MyActualType".to_string());
        let changed = diff.all_changed();
        assert!(changed.contains(&"MyActualType".to_string()));
        assert!(!changed.contains(&"xyzzy".to_string())); // xyzzy mutation would fail
        assert!(!changed.contains(&String::new())); // empty string mutation would fail
    }

    // ============== Mutation-killing tests for properties_using_functions ==============

    #[test]
    fn test_properties_using_functions_returns_matching() {
        // Kill mutation: replace -> HashSet<String> with HashSet::new()
        let graph = build_graph(
            r#"
            theorem uses_func { my_function(x) implies true }
        "#,
        );
        let result = graph.properties_using_functions(&["my_function".to_string()]);
        assert!(!result.is_empty()); // HashSet::new() would fail
        assert!(result.contains("uses_func"));
    }

    #[test]
    fn test_properties_using_functions_no_xyzzy() {
        // Kill mutation: replace with HashSet::from_iter(["xyzzy".into()])
        let graph = build_graph(
            r#"
            theorem uses_func { my_function(x) implies true }
        "#,
        );
        let result = graph.properties_using_functions(&["my_function".to_string()]);
        assert!(!result.contains("xyzzy")); // xyzzy mutation would fail
    }

    // ============== Mutation-killing tests for all_properties ==============

    #[test]
    fn test_all_properties_returns_correct_values() {
        // Kill mutation: replace -> Vec<String> with vec![]
        let graph = build_graph("theorem my_theorem { true }");
        let props = graph.all_properties();
        assert!(!props.is_empty()); // vec![] would fail
        assert!(props.contains(&"my_theorem".to_string()));
    }

    #[test]
    fn test_all_properties_no_xyzzy() {
        // Kill mutation: replace with vec!["xyzzy".into()]
        let graph = build_graph("theorem actual_name { true }");
        let props = graph.all_properties();
        assert!(!props.contains(&"xyzzy".to_string()));
        assert!(props.contains(&"actual_name".to_string()));
    }

    // ============== Mutation-killing tests for all_referenced_types ==============

    #[test]
    fn test_all_referenced_types_returns_correct_values() {
        // Kill mutation: replace -> Vec<String> with vec![]
        let graph = build_graph(
            r#"
            type MyType = { x: Int }
            theorem uses_type { forall t: MyType . t.x > 0 }
        "#,
        );
        let types = graph.all_referenced_types();
        assert!(!types.is_empty()); // vec![] would fail
        assert!(types.contains(&"MyType".to_string()));
    }

    #[test]
    fn test_all_referenced_types_no_xyzzy() {
        // Kill mutation: replace with vec!["xyzzy".into()]
        let graph = build_graph(
            r#"
            type ActualType = { x: Int }
            theorem uses_type { forall t: ActualType . t.x > 0 }
        "#,
        );
        let types = graph.all_referenced_types();
        assert!(!types.contains(&"xyzzy".to_string()));
    }

    // ============== Mutation-killing tests for property_depends_on_type ==============

    #[test]
    fn test_property_depends_on_type_true_case() {
        // Kill mutation: replace -> bool with false
        let graph = build_graph(
            r#"
            type MyType = { x: Int }
            theorem depends { forall t: MyType . t.x > 0 }
        "#,
        );
        assert!(graph.property_depends_on_type("depends", "MyType"));
    }

    #[test]
    fn test_property_depends_on_type_false_case() {
        // Kill mutation: replace -> bool with true
        let graph = build_graph(
            r#"
            type MyType = { x: Int }
            theorem no_depend { true }
        "#,
        );
        assert!(!graph.property_depends_on_type("no_depend", "MyType"));
    }

    // ============== Mutation-killing tests for analyze_* methods ==============
    // These methods populate the PropertyDependencies struct.
    // If they're replaced with (), the deps struct would be empty.

    #[test]
    fn test_analyze_temporal_populates_deps() {
        // Kill mutation: replace analyze_temporal with ()
        let graph = build_graph(
            r#"
            type State = { running: Bool }
            temporal eventually_done {
                eventually(forall s: State . not s.running)
            }
        "#,
        );
        let deps = graph.get_dependencies("eventually_done").unwrap();
        // If analyze_temporal is (), deps.types would be empty
        assert!(deps.types.contains("State"));
    }

    #[test]
    fn test_analyze_refinement_populates_deps() {
        // Kill mutation: replace analyze_refinement with ()
        let graph = build_graph(
            r#"
            type Abstract = { x: Int }
            type Concrete = { x: Int, y: Int }

            refinement concrete_refines_abstract refines Abstract {
                abstraction { forall c: Concrete . c.x >= 0 }
                simulation { true }
            }
        "#,
        );
        let deps = graph.get_dependencies("concrete_refines_abstract").unwrap();
        // If analyze_refinement is (), deps.types would be empty
        assert!(deps.types.contains("Abstract"));
        assert!(deps.types.contains("Concrete"));
    }

    #[test]
    fn test_analyze_probabilistic_populates_deps() {
        // Kill mutation: replace analyze_probabilistic with ()
        let graph = build_graph(
            r#"
            type CoinState = { heads: Bool }
            probabilistic fair_coin {
                probability(forall c: CoinState . c.heads) >= 0.5
            }
        "#,
        );
        let deps = graph.get_dependencies("fair_coin").unwrap();
        // If analyze_probabilistic is (), deps.types would be empty
        assert!(deps.types.contains("CoinState"));
    }

    #[test]
    fn test_analyze_security_populates_deps() {
        // Kill mutation: replace analyze_security with ()
        let graph = build_graph(
            r#"
            type Secret = { data: Int }
            security secret_confidential {
                forall s: Secret . s.data > 0
            }
        "#,
        );
        let deps = graph.get_dependencies("secret_confidential").unwrap();
        // If analyze_security is (), deps.types would be empty
        assert!(deps.types.contains("Secret"));
    }

    #[test]
    fn test_analyze_semantic_populates_deps() {
        // Kill mutation: replace analyze_semantic with ()
        let graph = build_graph(
            r#"
            type Model = { state: Int }
            semantic_property model_property {
                forall m: Model . m.state >= 0
            }
        "#,
        );
        let deps = graph.get_dependencies("model_property").unwrap();
        // If analyze_semantic is (), deps.types would be empty
        assert!(deps.types.contains("Model"));
    }

    #[test]
    fn test_analyze_platform_api_populates_deps() {
        // Kill mutation: replace analyze_platform_api with ()
        let graph = build_graph(
            r#"
            type ApiState = { connected: Bool }
            platform_api my_api {
                state Ready {
                    invariant { forall s: ApiState . s.connected }
                }
            }
        "#,
        );
        let deps = graph.get_dependencies("my_api").unwrap();
        // If analyze_platform_api is (), deps.types would be empty
        assert!(deps.types.contains("ApiState"));
    }

    #[test]
    fn test_analyze_bisimulation_populates_deps() {
        // Kill mutation: replace analyze_bisimulation with ()
        let graph = build_graph(
            r#"
            type TestInput = { data: Int }
            bisimulation process_equiv {
                oracle: "./impl1"
                subject: "./impl2"
                equivalent on { output }
                forall input: TestInput . traces_equivalent(oracle(input), subject(input))
            }
        "#,
        );
        let deps = graph.get_dependencies("process_equiv").unwrap();
        // If analyze_bisimulation is (), deps.types would be empty
        assert!(deps.types.contains("TestInput"));
    }

    #[test]
    fn test_analyze_temporal_expr_populates_deps() {
        // Kill mutation: replace analyze_temporal_expr with ()
        let graph = build_graph(
            r#"
            type Counter = { value: Int }
            temporal counter_always_positive {
                always(forall c: Counter . c.value >= 0)
            }
        "#,
        );
        let deps = graph.get_dependencies("counter_always_positive").unwrap();
        // If analyze_temporal_expr is (), the nested type wouldn't be found
        assert!(deps.types.contains("Counter"));
    }

    // ============== Test for diff delete ! mutation (line 526) ==============

    #[test]
    fn test_spec_diff_detects_unchanged_type() {
        // Kill mutation: delete ! in SpecDiff::diff (line 526)
        // If ! is deleted, base_types.contains_key(name) without ! would add types
        // that ARE in base to added_types (wrong!)
        let spec1 = parse("type Foo = { x: Int }").expect("parse");
        let spec2 = parse("type Foo = { x: Int }").expect("parse");

        let diff = SpecDiff::diff(&spec1, &spec2);
        // Foo exists in both - should NOT be in added_types
        assert!(diff.added_types.is_empty());
        assert!(diff.removed_types.is_empty());
        assert!(diff.modified_types.is_empty());
    }

    #[test]
    fn test_spec_diff_detects_new_type_not_in_base() {
        // Complementary test: type in current but not in base IS added
        let base = parse("theorem t { true }").expect("parse");
        let current = parse(
            r#"
            type NewType = { x: Int }
            theorem t { true }
        "#,
        )
        .expect("parse");

        let diff = SpecDiff::diff(&base, &current);
        assert!(diff.added_types.contains(&"NewType".to_string()));
    }
}
