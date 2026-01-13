//! TLA+ backend compiler
//!
//! Compiles USL specifications to TLA+ modules for temporal properties and invariants.

use crate::ast::{
    BinaryOp, CapabilitySpec, ComparisonOp, Expr, FairnessConstraint, FairnessKind,
    ImprovementProposal, Invariant, Property, Refinement, RollbackSpec, Temporal, TemporalExpr,
    Type, VerificationGate, VersionSpec,
};
use crate::typecheck::TypedSpec;

use super::CompiledSpec;

/// TLA+ compiler
pub struct TlaPlusCompiler {
    module_name: String,
}

impl TlaPlusCompiler {
    /// Create a new TLA+ compiler with the given module name
    #[must_use]
    pub fn new(module_name: &str) -> Self {
        Self {
            module_name: module_name.to_string(),
        }
    }

    /// Compile an expression to TLA+ syntax
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn compile_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Var(name) => {
                // Handle primed variables (self' -> state')
                if name.ends_with('\'') {
                    format!("{}'", &name[..name.len() - 1])
                } else {
                    name.clone()
                }
            }
            Expr::Int(n) => n.to_string(),
            Expr::Float(f) => f.to_string(),
            Expr::String(s) => format!("\"{s}\""),
            Expr::Bool(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),

            Expr::ForAll { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map(|t| format!(" \\in {}", self.compile_type(t)))
                    .unwrap_or_default();
                format!("\\A {}{}: {}", var, ty_str, self.compile_expr(body))
            }
            Expr::Exists { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map(|t| format!(" \\in {}", self.compile_type(t)))
                    .unwrap_or_default();
                format!("\\E {}{}: {}", var, ty_str, self.compile_expr(body))
            }
            Expr::ForAllIn {
                var,
                collection,
                body,
            } => {
                format!(
                    "\\A {} \\in {}: {}",
                    var,
                    self.compile_expr(collection),
                    self.compile_expr(body)
                )
            }
            Expr::ExistsIn {
                var,
                collection,
                body,
            } => {
                format!(
                    "\\E {} \\in {}: {}",
                    var,
                    self.compile_expr(collection),
                    self.compile_expr(body)
                )
            }

            Expr::Implies(lhs, rhs) => {
                format!(
                    "({}) => ({})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::And(lhs, rhs) => {
                format!(
                    "({}) /\\ ({})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Or(lhs, rhs) => {
                format!(
                    "({}) \\/ ({})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Not(e) => format!("~({})", self.compile_expr(e)),

            Expr::Compare(lhs, op, rhs) => {
                let op_str = match op {
                    ComparisonOp::Eq => "=",
                    ComparisonOp::Ne => "/=",
                    ComparisonOp::Lt => "<",
                    ComparisonOp::Le => "<=",
                    ComparisonOp::Gt => ">",
                    ComparisonOp::Ge => ">=",
                };
                format!(
                    "({}) {} ({})",
                    self.compile_expr(lhs),
                    op_str,
                    self.compile_expr(rhs)
                )
            }
            Expr::Binary(lhs, op, rhs) => {
                let op_str = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "\\div",
                    BinaryOp::Mod => "%",
                };
                format!(
                    "({}) {} ({})",
                    self.compile_expr(lhs),
                    op_str,
                    self.compile_expr(rhs)
                )
            }
            Expr::Neg(e) => format!("-({})", self.compile_expr(e)),

            Expr::App(name, args) => {
                // Handle graph predicates with proper TLA+ translations
                self.compile_graph_function(name, args).unwrap_or_else(|| {
                    // Default: pass through as function call
                    if args.is_empty() {
                        name.clone()
                    } else {
                        let args_str: Vec<String> =
                            args.iter().map(|a| self.compile_expr(a)).collect();
                        format!("{}({})", name, args_str.join(", "))
                    }
                })
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                let recv_str = self.compile_expr(receiver);
                if args.is_empty() {
                    format!("{recv_str}_{method}")
                } else {
                    let args_str: Vec<String> = args.iter().map(|a| self.compile_expr(a)).collect();
                    format!("{recv_str}_{method}({})", args_str.join(", "))
                }
            }
            Expr::FieldAccess(obj, field) => {
                // TLA+ uses subscript notation: state.field -> state[field]
                // or record syntax: state.field -> state.field
                format!("{}.{}", self.compile_expr(obj), field)
            }
        }
    }

    /// Compile graph-specific functions to TLA+ expressions
    ///
    /// Returns Some(tla_expr) for recognized graph functions, None otherwise.
    /// These functions are defined in Phase 17.3 for DashFlow execution graph verification.
    fn compile_graph_function(&self, name: &str, args: &[Expr]) -> Option<String> {
        let args_str: Vec<String> = args.iter().map(|a| self.compile_expr(a)).collect();

        match name {
            // Graph predicates
            "is_acyclic" | "is_dag" => {
                // is_acyclic(g) => IsAcyclic(g.nodes, g.edges)
                // Checks that there's no cycle in the graph
                if args.len() == 1 {
                    let g = &args_str[0];
                    Some(format!(
                        "~\\E path \\in Seq({g}.nodes): Len(path) > 1 /\\ path[1] = path[Len(path)] /\\ \\A i \\in 1..(Len(path)-1): <<path[i], path[i+1]>> \\in {g}.edges"
                    ))
                } else {
                    None
                }
            }

            "is_connected" => {
                // is_connected(g) => all nodes reachable from any other
                if args.len() == 1 {
                    let g = &args_str[0];
                    Some(format!(
                        "\\A n1, n2 \\in {g}.nodes: Reachable(n1, n2, {g}.edges) \\/ Reachable(n2, n1, {g}.edges)"
                    ))
                } else {
                    None
                }
            }

            "has_path" | "reachable" => {
                // has_path(g, from, to) => Reachable(from, to, g.edges)
                if args.len() == 3 {
                    Some(format!(
                        "Reachable({}, {}, {}.edges)",
                        args_str[1], args_str[2], args_str[0]
                    ))
                } else if args.len() == 2 {
                    // reachable(from, to) with implicit edges
                    Some(format!(
                        "Reachable({}, {}, edges)",
                        args_str[0], args_str[1]
                    ))
                } else {
                    None
                }
            }

            "in_graph" => {
                // in_graph(node, g) => node \in g.nodes
                if args.len() == 2 {
                    Some(format!("{} \\in {}.nodes", args_str[0], args_str[1]))
                } else {
                    None
                }
            }

            "edge_exists" => {
                // edge_exists(g, from, to) => <<from, to>> \in g.edges
                if args.len() == 3 {
                    Some(format!(
                        "<<{}, {}>> \\in {}.edges",
                        args_str[1], args_str[2], args_str[0]
                    ))
                } else {
                    None
                }
            }

            // DashFlow modification predicates
            "preserves_completed" => {
                // preserves_completed(g, g') => all completed nodes in g are completed in g'
                if args.len() == 2 {
                    Some(format!(
                        "\\A n \\in {}.nodes: completed(n) => n \\in {}.nodes /\\ completed(n')",
                        args_str[0], args_str[1]
                    ))
                } else {
                    None
                }
            }

            "valid_modification" => {
                // valid_modification(m, g, g') => modification is valid
                if args.len() == 3 {
                    Some(format!(
                        "ValidModification({}, {}, {})",
                        args_str[0], args_str[1], args_str[2]
                    ))
                } else {
                    None
                }
            }

            "preserves_dag" => {
                // preserves_dag(g, g') => if g is DAG, g' is also DAG
                if args.len() == 2 {
                    let g = &args_str[0];
                    let g_prime = &args_str[1];
                    Some(format!(
                        "IsAcyclic({g}.nodes, {g}.edges) => IsAcyclic({g_prime}.nodes, {g_prime}.edges)"
                    ))
                } else {
                    None
                }
            }

            "is_ready" => {
                // is_ready(node, g) => node's dependencies are satisfied
                if args.len() == 2 {
                    Some(format!(
                        "\\A dep \\in predecessors({}, {}): completed(dep)",
                        args_str[0], args_str[1]
                    ))
                } else {
                    None
                }
            }

            "all_deps_completed" => {
                // all_deps_completed(node, g) => all predecessors are completed
                if args.len() == 2 {
                    Some(format!(
                        "\\A dep \\in predecessors({}, {}): completed(dep)",
                        args_str[0], args_str[1]
                    ))
                } else {
                    None
                }
            }

            // Graph accessor functions
            "nodes" => {
                // nodes(g) => g.nodes
                if args.len() == 1 {
                    Some(format!("{}.nodes", args_str[0]))
                } else {
                    None
                }
            }

            "edges" => {
                // edges(g) => g.edges
                if args.len() == 1 {
                    Some(format!("{}.edges", args_str[0]))
                } else {
                    None
                }
            }

            "successors" => {
                // successors(g, node) => {n \in g.nodes: <<node, n>> \in g.edges}
                if args.len() == 2 {
                    Some(format!(
                        "{{n \\in {}.nodes: <<{}, n>> \\in {}.edges}}",
                        args_str[0], args_str[1], args_str[0]
                    ))
                } else {
                    None
                }
            }

            "predecessors" => {
                // predecessors(g, node) => {n \in g.nodes: <<n, node>> \in g.edges}
                if args.len() == 2 {
                    Some(format!(
                        "{{n \\in {}.nodes: <<n, {}>> \\in {}.edges}}",
                        args_str[0], args_str[1], args_str[0]
                    ))
                } else {
                    None
                }
            }

            "node_count" => {
                // node_count(g) => Cardinality(g.nodes)
                if args.len() == 1 {
                    Some(format!("Cardinality({}.nodes)", args_str[0]))
                } else {
                    None
                }
            }

            "edge_count" => {
                // edge_count(g) => Cardinality(g.edges)
                if args.len() == 1 {
                    Some(format!("Cardinality({}.edges)", args_str[0]))
                } else {
                    None
                }
            }

            "in_degree" => {
                // in_degree(g, node) => Cardinality({e \in g.edges: e[2] = node})
                if args.len() == 2 {
                    Some(format!(
                        "Cardinality({{e \\in {}.edges: e[2] = {}}})",
                        args_str[0], args_str[1]
                    ))
                } else {
                    None
                }
            }

            "out_degree" => {
                // out_degree(g, node) => Cardinality({e \in g.edges: e[1] = node})
                if args.len() == 2 {
                    Some(format!(
                        "Cardinality({{e \\in {}.edges: e[1] = {}}})",
                        args_str[0], args_str[1]
                    ))
                } else {
                    None
                }
            }

            // Node status predicates (simple pass-through)
            "completed" | "pending" | "running" | "failed" => {
                // These typically check a status field
                if args.len() == 1 {
                    Some(format!("{}.status = \"{}\"", args_str[0], name))
                } else {
                    None
                }
            }

            _ => None, // Not a graph function, let default handling take over
        }
    }

    /// Compile a temporal expression to TLA+ syntax
    #[must_use]
    pub fn compile_temporal_expr(&self, expr: &TemporalExpr) -> String {
        match expr {
            TemporalExpr::Always(inner) => {
                format!("[]({})", self.compile_temporal_expr(inner))
            }
            TemporalExpr::Eventually(inner) => {
                format!("<>({})", self.compile_temporal_expr(inner))
            }
            TemporalExpr::LeadsTo(from, to) => {
                format!(
                    "({}) ~> ({})",
                    self.compile_temporal_expr(from),
                    self.compile_temporal_expr(to)
                )
            }
            TemporalExpr::Atom(e) => self.compile_expr(e),
        }
    }

    /// Compile a type to TLA+ syntax
    #[must_use]
    pub fn compile_type(&self, ty: &Type) -> String {
        match ty {
            Type::Named(name) => name.clone(),
            Type::Set(inner) => format!("SUBSET {}", self.compile_type(inner)),
            Type::List(inner) => format!("Seq({})", self.compile_type(inner)),
            Type::Map(k, v) => {
                format!("[{} -> {}]", self.compile_type(k), self.compile_type(v))
            }
            Type::Relation(a, b) => {
                format!(
                    "SUBSET ({} \\X {})",
                    self.compile_type(a),
                    self.compile_type(b)
                )
            }
            Type::Function(a, b) => {
                format!("[{} -> {}]", self.compile_type(a), self.compile_type(b))
            }
            Type::Result(_) => "Result".to_string(),
            Type::Unit => "{}".to_string(),
            Type::Graph(n, e) => {
                format!(
                    "[nodes: SUBSET {}, edges: SUBSET ({} \\X {})]",
                    self.compile_type(n),
                    self.compile_type(n),
                    self.compile_type(e)
                )
            }
            Type::Path(n) => format!("Seq({})", self.compile_type(n)),
        }
    }

    /// Compile an invariant to TLA+ definition
    #[must_use]
    pub fn compile_invariant(&self, inv: &Invariant) -> String {
        format!("{} ==\n    {}", inv.name, self.compile_expr(&inv.body))
    }

    /// Compile a fairness constraint to TLA+ syntax
    ///
    /// Generates `WF_vars(Action)` for weak fairness or `SF_vars(Action)` for strong fairness.
    #[must_use]
    pub fn compile_fairness(&self, fairness: &FairnessConstraint) -> String {
        let prefix = match fairness.kind {
            FairnessKind::Weak => "WF",
            FairnessKind::Strong => "SF",
        };
        let vars = fairness.vars.as_deref().unwrap_or("vars");
        format!("{}_{vars}({})", prefix, fairness.action)
    }

    /// Compile a temporal property to TLA+ definition
    ///
    /// For liveness properties with fairness constraints, this generates a property
    /// that conjoins the fairness assumptions with the temporal formula.
    #[must_use]
    pub fn compile_temporal(&self, temp: &Temporal) -> String {
        let body = self.compile_temporal_expr(&temp.body);

        if temp.fairness.is_empty() {
            // No fairness constraints - simple definition
            format!("{} ==\n    {}", temp.name, body)
        } else {
            // With fairness: Fairness => Property
            // TLC checks: Spec /\ WF_vars(Next) => <>Property
            // We encode this as: Property conjoined with fairness in the spec
            let fairness_formulas: Vec<String> = temp
                .fairness
                .iter()
                .map(|f| self.compile_fairness(f))
                .collect();

            // Generate both the raw property and a fairness-augmented version
            let property_def = format!("{}_body ==\n    {}", temp.name, body);

            // Generate the full property with fairness
            // This is the formula: (Fairness1 /\ Fairness2 /\ ...) => Property
            // However, in TLA+, we typically add fairness to the spec, not the property
            // So we'll generate a comment and the raw property
            let fairness_comment = format!(
                "(* Fairness assumptions for {}: {} *)",
                temp.name,
                fairness_formulas.join(" /\\ ")
            );

            // The main property definition
            let main_def = format!("{} ==\n    {}", temp.name, body);

            format!("{}\n{}\n{}", fairness_comment, property_def, main_def)
        }
    }

    /// Extract fairness constraints from a temporal property
    #[must_use]
    pub fn extract_fairness(&self, temp: &Temporal) -> Vec<String> {
        temp.fairness
            .iter()
            .map(|f| self.compile_fairness(f))
            .collect()
    }

    /// Compile a refinement to TLA+ refinement mapping
    ///
    /// In TLA+, refinement is expressed by:
    /// 1. INSTANCE of the abstract spec with substitutions
    /// 2. Refinement theorems that relate implementation to spec
    ///
    /// Example output:
    /// ```tla
    /// (* Refinement mapping: StreamPoolImpl refines StreamPoolSpec *)
    /// SpecVars == spec.streams
    /// ImplVars == impl.m_streams
    ///
    /// (* Variable mappings *)
    /// spec_streams == impl.m_streams
    /// spec_bindings == impl.thread_local_slots
    ///
    /// (* Instantiate abstract spec with refinement mapping *)
    /// AbsSpec == INSTANCE StreamPoolSpec WITH streams <- spec_streams, bindings <- spec_bindings
    ///
    /// (* Refinement theorems *)
    /// RefinementMapping == AbsSpec!Spec
    /// ```
    #[must_use]
    pub fn compile_refinement(&self, ref_: &Refinement) -> String {
        let mut lines = Vec::new();

        lines.push(format!(
            "(* Refinement mapping: {} refines {} *)",
            ref_.name, ref_.refines
        ));
        lines.push(String::new());

        // Generate variable mapping definitions
        if !ref_.mappings.is_empty() {
            lines.push("(* Variable mappings *)".to_string());
            for (i, mapping) in ref_.mappings.iter().enumerate() {
                let spec_var = self.compile_expr(&mapping.spec_var);
                let impl_var = self.compile_expr(&mapping.impl_var);
                // Extract just the variable name from spec side for substitution
                let spec_name = match &mapping.spec_var {
                    Expr::FieldAccess(_, field) => field.clone(),
                    Expr::Var(name) => name.clone(),
                    _ => format!("mapping_{i}"),
                };
                lines.push(format!(
                    "{}_refinement_{} == {}  (* {} <- {} *)",
                    ref_.name, spec_name, impl_var, spec_var, impl_var
                ));
            }
            lines.push(String::new());
        }

        // Generate refinement invariants
        if !ref_.invariants.is_empty() {
            lines.push("(* Refinement invariants *)".to_string());
            for (i, inv) in ref_.invariants.iter().enumerate() {
                lines.push(format!(
                    "{}_Invariant_{} ==\n    {}",
                    ref_.name,
                    i,
                    self.compile_expr(inv)
                ));
            }
            lines.push(String::new());
        }

        // Generate INSTANCE statement with substitutions
        lines.push("(* Instantiate abstract spec with refinement mapping *)".to_string());
        let substitutions: Vec<String> = ref_
            .mappings
            .iter()
            .enumerate()
            .map(|(i, mapping)| {
                let spec_name = match &mapping.spec_var {
                    Expr::FieldAccess(_, field) => field.clone(),
                    Expr::Var(name) => name.clone(),
                    _ => format!("mapping_{i}"),
                };
                format!("{} <- {}_refinement_{}", spec_name, ref_.name, spec_name)
            })
            .collect();

        if substitutions.is_empty() {
            lines.push(format!(
                "{}_AbsSpec == INSTANCE {}",
                ref_.name, ref_.refines
            ));
        } else {
            lines.push(format!(
                "{}_AbsSpec == INSTANCE {} WITH {}",
                ref_.name,
                ref_.refines,
                substitutions.join(", ")
            ));
        }
        lines.push(String::new());

        // Generate abstraction theorem
        lines.push("(* Abstraction function *)".to_string());
        lines.push(format!(
            "{}_Abstraction ==\n    {}",
            ref_.name,
            self.compile_expr(&ref_.abstraction)
        ));
        lines.push(String::new());

        // Generate simulation theorem
        lines.push("(* Simulation relation *)".to_string());
        lines.push(format!(
            "{}_Simulation ==\n    {}",
            ref_.name,
            self.compile_expr(&ref_.simulation)
        ));
        lines.push(String::new());

        // Generate action mappings
        if !ref_.actions.is_empty() {
            lines.push("(* Action correspondence *)".to_string());
            for action in &ref_.actions {
                let impl_action = action.impl_action.join("_");
                if let Some(guard) = &action.guard {
                    lines.push(format!(
                        "{}_Action_{} ==\n    ({}) => ({}_AbsSpec!{} = {})",
                        ref_.name,
                        action.name,
                        self.compile_expr(guard),
                        ref_.name,
                        action.spec_action,
                        impl_action
                    ));
                } else {
                    lines.push(format!(
                        "{}_Action_{} ==\n    {}_AbsSpec!{} = {}",
                        ref_.name, action.name, ref_.name, action.spec_action, impl_action
                    ));
                }
            }
            lines.push(String::new());
        }

        // Generate the complete refinement theorem
        let mut refinement_conjuncts = vec![format!("{}_Simulation", ref_.name)];
        for action in &ref_.actions {
            refinement_conjuncts.push(format!("{}_Action_{}", ref_.name, action.name));
        }
        for (i, _) in ref_.invariants.iter().enumerate() {
            refinement_conjuncts.push(format!("{}_Invariant_{}", ref_.name, i));
        }

        lines.push("(* Complete refinement theorem *)".to_string());
        lines.push(format!(
            "{}_RefinementTheorem ==\n    {}",
            ref_.name,
            refinement_conjuncts.join(" /\\ ")
        ));

        lines.join("\n")
    }

    /// Compile a version specification to TLA+ definitions
    ///
    /// A version spec `version V2 improves V1 { capability {...}; preserves {...} }` compiles to:
    /// - Operator definitions for each capability clause (model-checkable properties)
    /// - Operator definitions for each preserves clause
    /// - A combined version improvement theorem
    ///
    /// TLA+ is ideal for version specs because TLC can model-check the improvement properties.
    #[must_use]
    pub fn compile_version_spec(&self, version: &VersionSpec) -> String {
        let mut lines = Vec::new();

        lines.push(format!(
            "(* Version Improvement: {} improves {} *)",
            version.name, version.improves
        ));
        lines.push(format!(
            "(* This specification proves {} is at least as capable as {} *)",
            version.name, version.improves
        ));
        lines.push(String::new());

        // Generate operators for capability clauses
        // In TLA+, these become properties that TLC can check
        for (i, cap) in version.capabilities.iter().enumerate() {
            let body_str = self.compile_expr(&cap.expr);
            lines.push(format!(
                "(* Capability improvement: {} over {} *)",
                version.name, version.improves
            ));
            lines.push(format!(
                "{}_Capability_{} ==\n    {}",
                version.name,
                i + 1,
                body_str
            ));
            lines.push(String::new());
        }

        // Generate operators for preserves clauses
        // These ensure backward compatibility
        for (i, pres) in version.preserves.iter().enumerate() {
            let body_str = self.compile_expr(&pres.property);
            lines.push(format!(
                "(* Preserved property from {} *)",
                version.improves
            ));
            lines.push(format!(
                "{}_Preserves_{} ==\n    {}",
                version.name,
                i + 1,
                body_str
            ));
            lines.push(String::new());
        }

        // Generate a combined version improvement operator
        if !version.capabilities.is_empty() || !version.preserves.is_empty() {
            let mut conjuncts = Vec::new();

            for i in 0..version.capabilities.len() {
                conjuncts.push(format!("{}_Capability_{}", version.name, i + 1));
            }
            for i in 0..version.preserves.len() {
                conjuncts.push(format!("{}_Preserves_{}", version.name, i + 1));
            }

            lines.push(format!(
                "(* Combined version improvement: {} improves {} *)",
                version.name, version.improves
            ));
            lines.push(format!(
                "{}_ImprovesOver_{} ==\n    {}",
                version.name,
                version.improves,
                conjuncts.join(" /\\ ")
            ));
        }

        lines.join("\n")
    }

    /// Compile a capability specification to TLA+ definitions
    ///
    /// A capability spec `capability Name { can f(...) -> T; requires { P } }` compiles to:
    /// - A record type representing the capability
    /// - Operators for each ability
    /// - Constraint operators for each requirement
    ///
    /// TLA+ uses records to model capabilities and operators for the abilities.
    #[must_use]
    pub fn compile_capability_spec(&self, capability: &CapabilitySpec) -> String {
        let mut lines = Vec::new();

        lines.push(format!(
            "(* Capability Specification: {} *)",
            capability.name
        ));
        lines.push(String::new());

        // Generate CONSTANTS declaration for the capability record type
        lines.push(format!("CONSTANTS {}", capability.name));
        lines.push(String::new());

        // Generate operators for each ability
        // In TLA+, abilities are modeled as operators/functions
        for ability in &capability.abilities {
            let return_type = ability
                .return_type
                .as_ref()
                .map(|t| self.compile_type(t))
                .unwrap_or_else(|| "{}".to_string());

            let params: Vec<String> = ability
                .params
                .iter()
                .map(|p| format!("{} \\in {}", p.name, self.compile_type(&p.ty)))
                .collect();

            lines.push(format!("(* Ability: {} *)", ability.name));
            if params.is_empty() {
                // Ability with no parameters - constant operator
                lines.push(format!(
                    "{}_{} == (* returns {} *)\n    TRUE  (* TODO: Define ability *)",
                    capability.name, ability.name, return_type
                ));
            } else {
                // Ability with parameters - operator taking arguments
                let param_names: Vec<String> =
                    ability.params.iter().map(|p| p.name.clone()).collect();
                lines.push(format!(
                    "{}_{}({}) == (* {} => {} *)\n    TRUE  (* TODO: Define ability *)",
                    capability.name,
                    ability.name,
                    param_names.join(", "),
                    params.join(", "),
                    return_type
                ));
            }
            lines.push(String::new());
        }

        // Generate operators for requirements
        for (i, req) in capability.requires.iter().enumerate() {
            let body_str = self.compile_expr(req);
            lines.push(format!(
                "(* Requirement {} for {} *)",
                i + 1,
                capability.name
            ));
            lines.push(format!(
                "{}_Requirement_{} ==\n    {}",
                capability.name,
                i + 1,
                body_str
            ));
            lines.push(String::new());
        }

        // Generate a combined requirements operator
        if !capability.requires.is_empty() {
            let req_conjuncts: Vec<String> = (0..capability.requires.len())
                .map(|i| format!("{}_Requirement_{}", capability.name, i + 1))
                .collect();

            lines.push(format!("(* All requirements for {} *)", capability.name));
            lines.push(format!(
                "{}_Requirements ==\n    {}",
                capability.name,
                req_conjuncts.join(" /\\ ")
            ));
        }

        lines.join("\n")
    }

    /// Compile a distributed invariant to TLA+
    ///
    /// Distributed invariants specify properties that must hold across multiple agents.
    /// In TLA+, these become multi-process invariants, typically with quantifiers over agents.
    ///
    /// Example USL:
    /// ```text
    /// distributed invariant proof_consensus {
    ///     forall d1 d2: Dasher, prop: Property .
    ///         (d1.proves(prop) and d2.proves(prop)) implies (d1.result == d2.result)
    /// }
    /// ```
    ///
    /// Compiles to:
    /// ```tla+
    /// (* Distributed Invariant: proof_consensus *)
    /// ProofConsensus ==
    ///     \A d1, d2 \in Dasher, prop \in Property :
    ///         (d1.proves[prop] /\ d2.proves[prop]) => (d1.result = d2.result)
    /// ```
    #[must_use]
    pub fn compile_distributed_invariant(
        &self,
        dist_inv: &crate::ast::DistributedInvariant,
    ) -> String {
        let mut lines = Vec::new();

        lines.push(format!("(* Distributed Invariant: {} *)", dist_inv.name));
        lines.push("(* Multi-agent invariant for distributed coordination *)".to_string());
        lines.push(String::new());

        let body_str = self.compile_expr(&dist_inv.body);
        lines.push(format!("{} ==", dist_inv.name));
        lines.push(format!("    {}", body_str));

        lines.join("\n")
    }

    /// Compile a distributed temporal property to TLA+
    ///
    /// Distributed temporal properties specify temporal formulas about agent coordination.
    /// In TLA+, these become temporal formulas with multi-process fairness constraints.
    ///
    /// Example USL:
    /// ```text
    /// distributed temporal version_convergence {
    ///     eventually(forall d1 d2: Dasher . d1.version == d2.version)
    /// }
    /// ```
    ///
    /// Compiles to:
    /// ```tla+
    /// (* Distributed Temporal: version_convergence *)
    /// VersionConvergence ==
    ///     <> (\A d1, d2 \in Dasher : d1.version = d2.version)
    /// ```
    #[must_use]
    pub fn compile_distributed_temporal(
        &self,
        dist_temp: &crate::ast::DistributedTemporal,
    ) -> String {
        let mut lines = Vec::new();

        lines.push(format!("(* Distributed Temporal: {} *)", dist_temp.name));
        lines.push("(* Multi-agent temporal property for coordination *)".to_string());
        lines.push(String::new());

        // Add fairness constraints as comments and in the formula
        for fc in &dist_temp.fairness {
            let fairness_op = match fc.kind {
                crate::ast::FairnessKind::Weak => "WF",
                crate::ast::FairnessKind::Strong => "SF",
            };
            let vars = fc.vars.as_deref().unwrap_or("vars");
            lines.push(format!(
                "(* Fairness: {}_{} ({})*)",
                fairness_op, vars, fc.action
            ));
        }

        let body_str = self.compile_temporal_expr(&dist_temp.body);

        // Build the temporal formula including fairness
        if dist_temp.fairness.is_empty() {
            lines.push(format!("{} ==", dist_temp.name));
            lines.push(format!("    {}", body_str));
        } else {
            // With fairness: Spec => Property where Spec includes fairness
            let fairness_clauses: Vec<String> = dist_temp
                .fairness
                .iter()
                .map(|fc| {
                    let fairness_op = match fc.kind {
                        crate::ast::FairnessKind::Weak => "WF",
                        crate::ast::FairnessKind::Strong => "SF",
                    };
                    let vars = fc.vars.as_deref().unwrap_or("vars");
                    format!("{}_{}({})", fairness_op, vars, fc.action)
                })
                .collect();
            let fairness_conj = fairness_clauses.join(" /\\ ");
            lines.push(format!("{} ==", dist_temp.name));
            lines.push(format!("    ({}) => ({})", fairness_conj, body_str));
        }

        lines.join("\n")
    }

    /// Compile an improvement proposal to TLA+ definitions
    ///
    /// An improvement proposal specifies what must be improved and what must be preserved.
    /// In TLA+, this compiles to operators for the proposal validation.
    ///
    /// Example USL:
    /// ```text
    /// improvement_proposal CodeOptimization {
    ///     target { Dasher.verify_rust_code }
    ///     improves { execution_speed >= 1.1 * baseline }
    ///     preserves { soundness, completeness }
    ///     requires { valid_rust_syntax(new_code) }
    /// }
    /// ```
    ///
    /// Compiles to:
    /// ```tlaplus
    /// (* Improvement Proposal: CodeOptimization *)
    /// CodeOptimization_Target == Dasher.verify_rust_code
    /// CodeOptimization_Improves_1 == execution_speed >= 1.1 * baseline
    /// CodeOptimization_Preserves_1 == soundness
    /// CodeOptimization_Requires_1 == valid_rust_syntax(new_code)
    /// CodeOptimization_Valid == CodeOptimization_Improves_1 /\ ... /\ CodeOptimization_Requires_1
    /// ```
    #[must_use]
    pub fn compile_improvement_proposal(&self, proposal: &ImprovementProposal) -> String {
        let mut lines = Vec::new();

        lines.push(format!("(* Improvement Proposal: {} *)", proposal.name));
        lines.push(
            "(* This proposal specifies conditions for a valid self-improvement *)".to_string(),
        );
        lines.push(String::new());

        // Target definition
        lines.push(format!(
            "{}_Target ==\n    {}",
            proposal.name,
            self.compile_expr(&proposal.target)
        ));
        lines.push(String::new());

        // Improvement clauses
        for (i, improves) in proposal.improves.iter().enumerate() {
            lines.push(format!(
                "(* Improvement clause {}: must be strictly better *)",
                i + 1
            ));
            lines.push(format!(
                "{}_Improves_{} ==\n    {}",
                proposal.name,
                i + 1,
                self.compile_expr(improves)
            ));
            lines.push(String::new());
        }

        // Preservation clauses
        for (i, preserves) in proposal.preserves.iter().enumerate() {
            lines.push(format!(
                "(* Preservation clause {}: must be at least as good *)",
                i + 1
            ));
            lines.push(format!(
                "{}_Preserves_{} ==\n    {}",
                proposal.name,
                i + 1,
                self.compile_expr(preserves)
            ));
            lines.push(String::new());
        }

        // Precondition clauses
        for (i, requires) in proposal.requires.iter().enumerate() {
            lines.push(format!(
                "(* Precondition {}: must hold before improvement *)",
                i + 1
            ));
            lines.push(format!(
                "{}_Requires_{} ==\n    {}",
                proposal.name,
                i + 1,
                self.compile_expr(requires)
            ));
            lines.push(String::new());
        }

        // Combined validity operator
        let mut conjuncts = Vec::new();
        for i in 0..proposal.improves.len() {
            conjuncts.push(format!("{}_Improves_{}", proposal.name, i + 1));
        }
        for i in 0..proposal.preserves.len() {
            conjuncts.push(format!("{}_Preserves_{}", proposal.name, i + 1));
        }
        for i in 0..proposal.requires.len() {
            conjuncts.push(format!("{}_Requires_{}", proposal.name, i + 1));
        }

        if !conjuncts.is_empty() {
            lines.push("(* Combined validity: all conditions must hold *)".to_string());
            lines.push(format!(
                "{}_Valid ==\n    {}",
                proposal.name,
                conjuncts.join(" /\\ ")
            ));
        }

        lines.join("\n")
    }

    /// Compile a verification gate to TLA+ definitions
    ///
    /// A verification gate specifies mandatory checks before accepting changes.
    /// In TLA+, this compiles to a state machine with guard conditions.
    ///
    /// Example USL:
    /// ```text
    /// verification_gate SelfModificationGate {
    ///     inputs { current: DasherVersion, proposed: Improvement }
    ///     checks {
    ///         check soundness_preserved { verify_soundness(current, proposed) }
    ///         check capability_preserved { verify_capabilities(current, proposed) }
    ///     }
    ///     on_pass { result = apply(proposed) }
    ///     on_fail { result = reject(proposed, errors) }
    /// }
    /// ```
    #[must_use]
    pub fn compile_verification_gate(&self, gate: &VerificationGate) -> String {
        let mut lines = Vec::new();

        lines.push(format!("(* Verification Gate: {} *)", gate.name));
        lines.push(
            "(* This gate enforces mandatory verification before self-modification *)".to_string(),
        );
        lines.push(String::new());

        // Input variables
        lines.push(format!("VARIABLES {}", gate.name.to_lowercase()));
        lines.push(String::new());

        // Input type constraints
        if !gate.inputs.is_empty() {
            let input_names: Vec<String> = gate.inputs.iter().map(|p| p.name.clone()).collect();
            let input_types: Vec<String> = gate
                .inputs
                .iter()
                .map(|p| format!("{} \\in {}", p.name, self.compile_type(&p.ty)))
                .collect();
            lines.push(format!("(* Inputs: {} *)", input_names.join(", ")));
            lines.push(format!(
                "{}_InputTypes ==\n    {}",
                gate.name,
                input_types.join(" /\\ ")
            ));
            lines.push(String::new());
        }

        // Individual checks
        for check in &gate.checks {
            lines.push(format!("(* Verification check: {} *)", check.name));
            lines.push(format!(
                "{}_Check_{} ==\n    {}",
                gate.name,
                check.name,
                self.compile_expr(&check.condition)
            ));
            lines.push(String::new());
        }

        // Combined check (all checks must pass)
        let check_names: Vec<String> = gate
            .checks
            .iter()
            .map(|c| format!("{}_Check_{}", gate.name, c.name))
            .collect();
        if !check_names.is_empty() {
            lines.push("(* All checks must pass *)".to_string());
            lines.push(format!(
                "{}_AllChecksPass ==\n    {}",
                gate.name,
                check_names.join(" /\\ ")
            ));
            lines.push(String::new());
        }

        // On pass action
        lines.push("(* Action when all checks pass *)".to_string());
        lines.push(format!(
            "{}_OnPass ==\n    {}_AllChecksPass /\\ {}",
            gate.name,
            gate.name,
            self.compile_expr(&gate.on_pass)
        ));
        lines.push(String::new());

        // On fail action
        lines.push("(* Action when any check fails *)".to_string());
        lines.push(format!(
            "{}_OnFail ==\n    ~{}_AllChecksPass /\\ {}",
            gate.name,
            gate.name,
            self.compile_expr(&gate.on_fail)
        ));
        lines.push(String::new());

        // Next action (either pass or fail)
        lines.push("(* Gate transition action *)".to_string());
        lines.push(format!(
            "{}_Next ==\n    {}_OnPass \\/ {}_OnFail",
            gate.name, gate.name, gate.name
        ));

        lines.join("\n")
    }

    /// Compile a rollback specification to TLA+ definitions
    ///
    /// A rollback spec defines safe state recovery after failed improvements.
    /// In TLA+, this compiles to actions with invariant preservation.
    ///
    /// Example USL:
    /// ```text
    /// rollback_spec SafeRollback {
    ///     state { current: DasherVersion, history: List<DasherVersion> }
    ///     invariant { |history| > 0 }
    ///     trigger { verification_failed or runtime_error }
    ///     action { current = history.last(); ensure { verified(current) } }
    ///     guarantee { always(verified(current)) }
    /// }
    /// ```
    #[must_use]
    pub fn compile_rollback_spec(&self, rollback: &RollbackSpec) -> String {
        let mut lines = Vec::new();

        lines.push(format!("(* Rollback Specification: {} *)", rollback.name));
        lines.push(
            "(* This specification ensures safe recovery from failed improvements *)".to_string(),
        );
        lines.push(String::new());

        // State variables
        if !rollback.state.is_empty() {
            let state_vars: Vec<String> = rollback.state.iter().map(|p| p.name.clone()).collect();
            lines.push(format!("VARIABLES {}", state_vars.join(", ")));
            lines.push(String::new());

            // State type constraints
            let state_types: Vec<String> = rollback
                .state
                .iter()
                .map(|p| format!("{} \\in {}", p.name, self.compile_type(&p.ty)))
                .collect();
            lines.push("(* State type constraints *)".to_string());
            lines.push(format!(
                "{}_StateTypes ==\n    {}",
                rollback.name,
                state_types.join(" /\\ ")
            ));
            lines.push(String::new());
        }

        // Invariants
        for (i, inv) in rollback.invariants.iter().enumerate() {
            lines.push(format!(
                "(* Invariant {}: must hold before and after rollback *)",
                i + 1
            ));
            lines.push(format!(
                "{}_Invariant_{} ==\n    {}",
                rollback.name,
                i + 1,
                self.compile_expr(inv)
            ));
            lines.push(String::new());
        }

        // Combined invariant
        if !rollback.invariants.is_empty() {
            let inv_names: Vec<String> = (0..rollback.invariants.len())
                .map(|i| format!("{}_Invariant_{}", rollback.name, i + 1))
                .collect();
            lines.push("(* All invariants *)".to_string());
            lines.push(format!(
                "{}_AllInvariants ==\n    {}",
                rollback.name,
                inv_names.join(" /\\ ")
            ));
            lines.push(String::new());
        }

        // Trigger condition
        lines.push("(* Rollback trigger condition *)".to_string());
        lines.push(format!(
            "{}_Trigger ==\n    {}",
            rollback.name,
            self.compile_expr(&rollback.trigger)
        ));
        lines.push(String::new());

        // Rollback action
        lines.push("(* Rollback action *)".to_string());
        let mut action_parts = Vec::new();
        for (var, expr) in &rollback.action.assignments {
            action_parts.push(format!("{}' = {}", var, self.compile_expr(expr)));
        }
        if let Some(ensure) = &rollback.action.ensure {
            action_parts.push(self.compile_expr(ensure));
        }
        lines.push(format!(
            "{}_Action ==\n    {}_Trigger /\\ {}",
            rollback.name,
            rollback.name,
            if action_parts.is_empty() {
                "TRUE".to_string()
            } else {
                action_parts.join(" /\\ ")
            }
        ));
        lines.push(String::new());

        // Guarantees
        for (i, guarantee) in rollback.guarantees.iter().enumerate() {
            lines.push(format!(
                "(* Guarantee {}: must hold after rollback completes *)",
                i + 1
            ));
            lines.push(format!(
                "{}_Guarantee_{} ==\n    {}",
                rollback.name,
                i + 1,
                self.compile_expr(guarantee)
            ));
            lines.push(String::new());
        }

        // Combined specification
        let mut spec_parts = Vec::new();
        if !rollback.invariants.is_empty() {
            spec_parts.push(format!("{}_AllInvariants", rollback.name));
        }
        for i in 0..rollback.guarantees.len() {
            spec_parts.push(format!("{}_Guarantee_{}", rollback.name, i + 1));
        }

        if !spec_parts.is_empty() {
            lines.push("(* Complete rollback specification *)".to_string());
            lines.push(format!(
                "{}_Spec ==\n    {}_Action => ({})",
                rollback.name,
                rollback.name,
                spec_parts.join(" /\\ ")
            ));
        }

        lines.join("\n")
    }

    /// Compile a composed theorem to TLA+
    ///
    /// Composed theorems declare dependencies on other properties via `uses`.
    /// In TLA+, these are compiled as THEOREM declarations with ASSUME clauses.
    ///
    /// Example USL:
    /// ```text
    /// composed theorem modular_safety {
    ///     uses { acyclic_theorem, connectivity_invariant }
    ///     acyclic_theorem and connectivity_invariant implies safe_execution
    /// }
    /// ```
    ///
    /// Compiles to:
    /// ```tlaplus
    /// (* Composed Theorem: modular_safety *)
    /// (* Uses: acyclic_theorem, connectivity_invariant *)
    /// THEOREM modular_safety ==
    ///     ASSUME acyclic_theorem, connectivity_invariant
    ///     PROVE acyclic_theorem /\ connectivity_invariant => safe_execution
    /// ```
    #[must_use]
    pub fn compile_composed_theorem(&self, composed: &crate::ast::ComposedTheorem) -> String {
        let mut lines = Vec::new();

        lines.push(format!("(* Composed Theorem: {} *)", composed.name));
        lines.push(format!("(* Uses: {} *)", composed.uses.join(", ")));
        lines.push(String::new());

        let body_str = self.compile_expr(&composed.body);

        if composed.uses.is_empty() {
            // No assumptions, just a simple theorem
            lines.push(format!("THEOREM {} ==", composed.name));
            lines.push(format!("    {}", body_str));
        } else {
            // With assumptions from uses
            lines.push(format!("THEOREM {} ==", composed.name));
            lines.push(format!("    ASSUME {}", composed.uses.join(", ")));
            lines.push(format!("    PROVE {}", body_str));
        }

        lines.join("\n")
    }

    /// Generate complete TLA+ module from spec
    #[must_use]
    pub fn compile_module(&self, typed_spec: &TypedSpec) -> CompiledSpec {
        let mut sections = Vec::new();

        // Module header
        sections.push(format!("---- MODULE {} ----", self.module_name));
        sections.push("EXTENDS Naturals, Sequences, FiniteSets, TLC".to_string());
        sections.push(String::new());

        // Compile type definitions as constants/operators
        for type_def in &typed_spec.spec.types {
            sections.push(format!("(* Type: {} *)", type_def.name));
            // Types in TLA+ are often represented as CONSTANTS or sets
            sections.push(format!("CONSTANTS {}", type_def.name));
        }
        if !typed_spec.spec.types.is_empty() {
            sections.push(String::new());
        }

        // Compile properties
        for property in &typed_spec.spec.properties {
            match property {
                Property::Invariant(inv) => {
                    sections.push(format!("(* Invariant: {} *)", inv.name));
                    sections.push(self.compile_invariant(inv));
                    sections.push(String::new());
                }
                Property::Temporal(temp) => {
                    sections.push(format!("(* Temporal Property: {} *)", temp.name));
                    sections.push(self.compile_temporal(temp));
                    sections.push(String::new());
                }
                Property::Theorem(thm) => {
                    // Theorems in TLA+ become operator definitions or assertions
                    sections.push(format!("(* Theorem: {} *)", thm.name));
                    sections.push(format!(
                        "{} ==\n    {}",
                        thm.name,
                        self.compile_expr(&thm.body)
                    ));
                    sections.push(String::new());
                }
                Property::Security(security) => {
                    sections.push(format!("(* Security Property: {} *)", security.name));
                    sections.push(format!(
                        "{} ==\n    {}",
                        security.name,
                        self.compile_expr(&security.body)
                    ));
                    sections.push(String::new());
                }
                Property::Refinement(ref_) => {
                    sections.push(self.compile_refinement(ref_));
                    sections.push(String::new());
                }
                Property::Version(version) => {
                    sections.push(self.compile_version_spec(version));
                    sections.push(String::new());
                }
                Property::Capability(capability) => {
                    sections.push(self.compile_capability_spec(capability));
                    sections.push(String::new());
                }
                Property::DistributedInvariant(dist_inv) => {
                    sections.push(self.compile_distributed_invariant(dist_inv));
                    sections.push(String::new());
                }
                Property::DistributedTemporal(dist_temp) => {
                    sections.push(self.compile_distributed_temporal(dist_temp));
                    sections.push(String::new());
                }
                Property::Composed(composed) => {
                    sections.push(self.compile_composed_theorem(composed));
                    sections.push(String::new());
                }
                Property::ImprovementProposal(proposal) => {
                    sections.push(self.compile_improvement_proposal(proposal));
                    sections.push(String::new());
                }
                Property::VerificationGate(gate) => {
                    sections.push(self.compile_verification_gate(gate));
                    sections.push(String::new());
                }
                Property::Rollback(rollback) => {
                    sections.push(self.compile_rollback_spec(rollback));
                    sections.push(String::new());
                }
                _ => {
                    // Contracts, probabilistic - not supported in TLA+
                }
            }
        }

        // Module footer
        sections.push("====".to_string());

        CompiledSpec {
            backend: "TLA+".to_string(),
            code: sections.join("\n"),
            module_name: Some(self.module_name.clone()),
            imports: vec![
                "Naturals".to_string(),
                "Sequences".to_string(),
                "FiniteSets".to_string(),
            ],
        }
    }
}

/// Compile to TLA+ module
#[must_use]
pub fn compile_to_tlaplus(spec: &TypedSpec) -> CompiledSpec {
    let module_name = "USLSpec".to_string();
    let compiler = TlaPlusCompiler::new(&module_name);
    compiler.compile_module(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        ActionMapping, CapabilityAbility, CapabilityClause, FairnessKind, Field, Param,
        PreservesClause, Security, Spec, Temporal, TemporalExpr, Theorem, TypeDef, VariableMapping,
    };

    fn make_typed_spec(properties: Vec<Property>) -> TypedSpec {
        TypedSpec {
            spec: Spec {
                types: vec![],
                properties,
            },
            type_info: std::collections::HashMap::new(),
        }
    }

    fn make_typed_spec_with_types(properties: Vec<Property>, types: Vec<TypeDef>) -> TypedSpec {
        TypedSpec {
            spec: Spec { types, properties },
            type_info: std::collections::HashMap::new(),
        }
    }

    // ============ compile_type tests ============

    #[test]
    fn test_compile_type_named() {
        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_type(&Type::Named("Int".to_string()));
        assert_eq!(result, "Int");
    }

    #[test]
    fn test_compile_type_set() {
        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_type(&Type::Set(Box::new(Type::Named("Nat".to_string()))));
        assert_eq!(result, "SUBSET Nat");
    }

    #[test]
    fn test_compile_type_list() {
        let compiler = TlaPlusCompiler::new("Test");
        let result =
            compiler.compile_type(&Type::List(Box::new(Type::Named("String".to_string()))));
        assert_eq!(result, "Seq(String)");
    }

    #[test]
    fn test_compile_type_map() {
        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_type(&Type::Map(
            Box::new(Type::Named("Key".to_string())),
            Box::new(Type::Named("Value".to_string())),
        ));
        assert_eq!(result, "[Key -> Value]");
    }

    #[test]
    fn test_compile_type_relation() {
        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_type(&Type::Relation(
            Box::new(Type::Named("A".to_string())),
            Box::new(Type::Named("B".to_string())),
        ));
        assert_eq!(result, "SUBSET (A \\X B)");
    }

    #[test]
    fn test_compile_type_function() {
        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_type(&Type::Function(
            Box::new(Type::Named("X".to_string())),
            Box::new(Type::Named("Y".to_string())),
        ));
        assert_eq!(result, "[X -> Y]");
    }

    #[test]
    fn test_compile_type_result() {
        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_type(&Type::Result(Box::new(Type::Named("T".to_string()))));
        assert_eq!(result, "Result");
    }

    #[test]
    fn test_compile_type_unit() {
        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_type(&Type::Unit);
        assert_eq!(result, "{}");
    }

    // ============ compile_fairness tests ============

    #[test]
    fn test_compile_fairness_weak() {
        let compiler = TlaPlusCompiler::new("Test");
        let fairness = FairnessConstraint {
            kind: FairnessKind::Weak,
            action: "Next".to_string(),
            vars: None,
        };
        let result = compiler.compile_fairness(&fairness);
        assert_eq!(result, "WF_vars(Next)");
    }

    #[test]
    fn test_compile_fairness_strong() {
        let compiler = TlaPlusCompiler::new("Test");
        let fairness = FairnessConstraint {
            kind: FairnessKind::Strong,
            action: "Acquire".to_string(),
            vars: None,
        };
        let result = compiler.compile_fairness(&fairness);
        assert_eq!(result, "SF_vars(Acquire)");
    }

    #[test]
    fn test_compile_fairness_with_vars() {
        let compiler = TlaPlusCompiler::new("Test");
        let fairness = FairnessConstraint {
            kind: FairnessKind::Weak,
            action: "DoAction".to_string(),
            vars: Some("<<x, y>>".to_string()),
        };
        let result = compiler.compile_fairness(&fairness);
        assert_eq!(result, "WF_<<x, y>>(DoAction)");
    }

    // ============ extract_fairness tests ============

    #[test]
    fn test_extract_fairness_single() {
        let compiler = TlaPlusCompiler::new("Test");
        let temporal = Temporal {
            name: "Liveness".to_string(),
            body: TemporalExpr::Eventually(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
            fairness: vec![FairnessConstraint {
                kind: FairnessKind::Weak,
                action: "Next".to_string(),
                vars: None,
            }],
        };
        let result = compiler.extract_fairness(&temporal);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "WF_vars(Next)");
    }

    #[test]
    fn test_extract_fairness_multiple() {
        let compiler = TlaPlusCompiler::new("Test");
        let temporal = Temporal {
            name: "Liveness".to_string(),
            body: TemporalExpr::Eventually(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
            fairness: vec![
                FairnessConstraint {
                    kind: FairnessKind::Weak,
                    action: "Action1".to_string(),
                    vars: None,
                },
                FairnessConstraint {
                    kind: FairnessKind::Strong,
                    action: "Action2".to_string(),
                    vars: Some("state".to_string()),
                },
            ],
        };
        let result = compiler.extract_fairness(&temporal);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "WF_vars(Action1)");
        assert_eq!(result[1], "SF_state(Action2)");
    }

    #[test]
    fn test_extract_fairness_empty() {
        let compiler = TlaPlusCompiler::new("Test");
        let temporal = Temporal {
            name: "Simple".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
            fairness: vec![],
        };
        let result = compiler.extract_fairness(&temporal);
        assert!(result.is_empty());
    }

    // ============ compile_refinement tests ============

    #[test]
    fn test_compile_refinement_basic() {
        let compiler = TlaPlusCompiler::new("Test");
        let refinement = Refinement {
            name: "ImplRefinesSpec".to_string(),
            refines: "AbstractSpec".to_string(),
            mappings: vec![],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        };
        let result = compiler.compile_refinement(&refinement);
        assert!(result.contains("ImplRefinesSpec refines AbstractSpec"));
        assert!(result.contains("ImplRefinesSpec_AbsSpec == INSTANCE AbstractSpec"));
        assert!(result.contains("ImplRefinesSpec_Abstraction"));
        assert!(result.contains("ImplRefinesSpec_Simulation"));
    }

    #[test]
    fn test_compile_refinement_with_mappings_field_access() {
        let compiler = TlaPlusCompiler::new("Test");
        let refinement = Refinement {
            name: "Refine".to_string(),
            refines: "Spec".to_string(),
            mappings: vec![VariableMapping {
                spec_var: Expr::FieldAccess(
                    Box::new(Expr::Var("spec".to_string())),
                    "streams".to_string(),
                ),
                impl_var: Expr::FieldAccess(
                    Box::new(Expr::Var("impl".to_string())),
                    "m_streams".to_string(),
                ),
            }],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        };
        let result = compiler.compile_refinement(&refinement);
        assert!(result.contains("Variable mappings"));
        assert!(result.contains("Refine_refinement_streams == impl.m_streams"));
        assert!(result.contains("streams <- Refine_refinement_streams"));
    }

    #[test]
    fn test_compile_refinement_with_mappings_var() {
        let compiler = TlaPlusCompiler::new("Test");
        let refinement = Refinement {
            name: "R".to_string(),
            refines: "S".to_string(),
            mappings: vec![VariableMapping {
                spec_var: Expr::Var("specVar".to_string()),
                impl_var: Expr::Var("implVar".to_string()),
            }],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        };
        let result = compiler.compile_refinement(&refinement);
        assert!(result.contains("R_refinement_specVar == implVar"));
        assert!(result.contains("specVar <- R_refinement_specVar"));
    }

    #[test]
    fn test_compile_refinement_with_invariants() {
        let compiler = TlaPlusCompiler::new("Test");
        let refinement = Refinement {
            name: "RefWithInv".to_string(),
            refines: "BaseSpec".to_string(),
            mappings: vec![],
            invariants: vec![Expr::Compare(
                Box::new(Expr::Var("x".to_string())),
                ComparisonOp::Ge,
                Box::new(Expr::Int(0)),
            )],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        };
        let result = compiler.compile_refinement(&refinement);
        assert!(result.contains("Refinement invariants"));
        assert!(result.contains("RefWithInv_Invariant_0"));
        assert!(result.contains("(x) >= (0)"));
    }

    #[test]
    fn test_compile_refinement_with_actions() {
        let compiler = TlaPlusCompiler::new("Test");
        let refinement = Refinement {
            name: "ActionRef".to_string(),
            refines: "AbstractSpec".to_string(),
            mappings: vec![],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![ActionMapping {
                name: "DoStep".to_string(),
                spec_action: "Step".to_string(),
                impl_action: vec!["Impl".to_string(), "step".to_string()],
                guard: None,
            }],
        };
        let result = compiler.compile_refinement(&refinement);
        assert!(result.contains("Action correspondence"));
        assert!(result.contains("ActionRef_Action_DoStep"));
        assert!(result.contains("ActionRef_AbsSpec!Step = Impl_step"));
    }

    #[test]
    fn test_compile_refinement_with_action_guard() {
        let compiler = TlaPlusCompiler::new("Test");
        let refinement = Refinement {
            name: "Guarded".to_string(),
            refines: "Spec".to_string(),
            mappings: vec![],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![ActionMapping {
                name: "ConditionalAction".to_string(),
                spec_action: "AbstractAction".to_string(),
                impl_action: vec!["impl_action".to_string()],
                guard: Some(Expr::Var("enabled".to_string())),
            }],
        };
        let result = compiler.compile_refinement(&refinement);
        assert!(result.contains("(enabled) => (Guarded_AbsSpec!AbstractAction = impl_action)"));
    }

    #[test]
    fn test_compile_refinement_mapping_fallback() {
        // Test the fallback case when mapping.spec_var is neither FieldAccess nor Var
        let compiler = TlaPlusCompiler::new("Test");
        let refinement = Refinement {
            name: "R".to_string(),
            refines: "S".to_string(),
            mappings: vec![VariableMapping {
                spec_var: Expr::Int(42), // Neither FieldAccess nor Var
                impl_var: Expr::Int(100),
            }],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        };
        let result = compiler.compile_refinement(&refinement);
        // Should use fallback naming like "mapping_0"
        assert!(result.contains("R_refinement_mapping_0 == 100"));
        assert!(result.contains("mapping_0 <- R_refinement_mapping_0"));
    }

    // ============ compile_module tests ============

    #[test]
    fn test_compile_module_with_security() {
        let spec = make_typed_spec(vec![Property::Security(Security {
            name: "NoOverflow".to_string(),
            body: Expr::Compare(
                Box::new(Expr::Var("x".to_string())),
                ComparisonOp::Le,
                Box::new(Expr::Int(100)),
            ),
        })]);
        let compiler = TlaPlusCompiler::new("SecurityTest");
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("Security Property: NoOverflow"));
        assert!(result.code.contains("NoOverflow =="));
        assert!(result.code.contains("(x) <= (100)"));
    }

    #[test]
    fn test_compile_module_with_refinement() {
        let spec = make_typed_spec(vec![Property::Refinement(Refinement {
            name: "ImplRef".to_string(),
            refines: "AbstractModule".to_string(),
            mappings: vec![],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        })]);
        let compiler = TlaPlusCompiler::new("RefTest");
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("ImplRef refines AbstractModule"));
        assert!(result
            .code
            .contains("ImplRef_AbsSpec == INSTANCE AbstractModule"));
    }

    #[test]
    fn test_compile_module_with_types() {
        let spec = make_typed_spec_with_types(
            vec![],
            vec![TypeDef {
                name: "StreamId".to_string(),
                fields: vec![Field {
                    name: "id".to_string(),
                    ty: Type::Named("Nat".to_string()),
                }],
            }],
        );
        let compiler = TlaPlusCompiler::new("TypedModule");
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("(* Type: StreamId *)"));
        assert!(result.code.contains("CONSTANTS StreamId"));
    }

    #[test]
    fn test_compile_module_without_types() {
        let spec = make_typed_spec(vec![Property::Theorem(Theorem {
            name: "SimpleTheorem".to_string(),
            body: Expr::Bool(true),
        })]);
        let compiler = TlaPlusCompiler::new("NoTypes");
        let result = compiler.compile_module(&spec);
        // Should not have the extra newline after types section
        assert!(result.code.contains("---- MODULE NoTypes ----"));
        assert!(result.code.contains("SimpleTheorem =="));
        // Verify no CONSTANTS line since no types
        assert!(!result.code.contains("CONSTANTS"));
    }

    #[test]
    fn test_compile_module_header_and_footer() {
        let spec = make_typed_spec(vec![]);
        let compiler = TlaPlusCompiler::new("TestModule");
        let result = compiler.compile_module(&spec);
        assert!(result.code.starts_with("---- MODULE TestModule ----"));
        assert!(result
            .code
            .contains("EXTENDS Naturals, Sequences, FiniteSets, TLC"));
        assert!(result.code.ends_with("===="));
        assert_eq!(result.backend, "TLA+");
        assert_eq!(result.module_name, Some("TestModule".to_string()));
    }

    #[test]
    fn test_compile_module_types_add_blank_line_after() {
        // Types should be followed by a blank line (empty section separator)
        let spec = make_typed_spec_with_types(
            vec![Property::Invariant(Invariant {
                name: "Inv".to_string(),
                body: Expr::Bool(true),
            })],
            vec![TypeDef {
                name: "MyType".to_string(),
                fields: vec![],
            }],
        );
        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_module(&spec);
        // After CONSTANTS line, there should be a blank line before the invariant
        // The pattern is: "CONSTANTS MyType\n\n(* Invariant"
        assert!(result.code.contains("CONSTANTS MyType\n\n(* Invariant"));
    }

    #[test]
    fn test_compile_module_no_types_no_extra_blank() {
        // Without types, there should NOT be extra blank lines after the EXTENDS line
        let spec = make_typed_spec(vec![Property::Invariant(Invariant {
            name: "Inv".to_string(),
            body: Expr::Bool(true),
        })]);
        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_module(&spec);
        // Without types, EXTENDS is directly followed by blank then invariant
        // Should NOT have "CONSTANTS" at all
        assert!(!result.code.contains("CONSTANTS"));
        // Should have EXTENDS line followed by blank then invariant
        assert!(result
            .code
            .contains("EXTENDS Naturals, Sequences, FiniteSets, TLC\n\n(* Invariant"));
    }

    // ============ compile_temporal with fairness tests ============

    #[test]
    fn test_compile_temporal_with_fairness() {
        let compiler = TlaPlusCompiler::new("Test");
        let temporal = Temporal {
            name: "Liveness".to_string(),
            body: TemporalExpr::Eventually(Box::new(TemporalExpr::Atom(Expr::Var(
                "done".to_string(),
            )))),
            fairness: vec![FairnessConstraint {
                kind: FairnessKind::Weak,
                action: "Next".to_string(),
                vars: None,
            }],
        };
        let result = compiler.compile_temporal(&temporal);
        assert!(result.contains("Fairness assumptions for Liveness"));
        assert!(result.contains("WF_vars(Next)"));
        assert!(result.contains("Liveness_body =="));
        assert!(result.contains("Liveness =="));
    }

    #[test]
    fn test_compile_temporal_without_fairness() {
        let compiler = TlaPlusCompiler::new("Test");
        let temporal = Temporal {
            name: "Safety".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Var("safe".to_string())))),
            fairness: vec![],
        };
        let result = compiler.compile_temporal(&temporal);
        assert_eq!(result, "Safety ==\n    [](safe)");
        // Should NOT contain fairness comment
        assert!(!result.contains("Fairness assumptions"));
    }

    // ============ compile_version_spec tests ============

    #[test]
    fn test_compile_version_spec_basic() {
        let compiler = TlaPlusCompiler::new("Test");
        let version = VersionSpec {
            name: "DasherV2".to_string(),
            improves: "DasherV1".to_string(),
            capabilities: vec![CapabilityClause {
                expr: Expr::Compare(
                    Box::new(Expr::FieldAccess(
                        Box::new(Expr::Var("V2".to_string())),
                        "speed".to_string(),
                    )),
                    ComparisonOp::Ge,
                    Box::new(Expr::FieldAccess(
                        Box::new(Expr::Var("V1".to_string())),
                        "speed".to_string(),
                    )),
                ),
            }],
            preserves: vec![PreservesClause {
                property: Expr::FieldAccess(
                    Box::new(Expr::Var("V1".to_string())),
                    "soundness".to_string(),
                ),
            }],
        };
        let result = compiler.compile_version_spec(&version);
        assert!(result.contains("Version Improvement: DasherV2 improves DasherV1"));
        assert!(result.contains("DasherV2_Capability_1 =="));
        assert!(result.contains("(V2.speed) >= (V1.speed)"));
        assert!(result.contains("DasherV2_Preserves_1 =="));
        assert!(result.contains("V1.soundness"));
        assert!(result.contains("DasherV2_ImprovesOver_DasherV1 =="));
    }

    #[test]
    fn test_compile_version_spec_multiple_capabilities() {
        let compiler = TlaPlusCompiler::new("Test");
        let version = VersionSpec {
            name: "V2".to_string(),
            improves: "V1".to_string(),
            capabilities: vec![
                CapabilityClause {
                    expr: Expr::Compare(
                        Box::new(Expr::Var("speed".to_string())),
                        ComparisonOp::Ge,
                        Box::new(Expr::Int(100)),
                    ),
                },
                CapabilityClause {
                    expr: Expr::Compare(
                        Box::new(Expr::Var("accuracy".to_string())),
                        ComparisonOp::Ge,
                        Box::new(Expr::Int(90)),
                    ),
                },
            ],
            preserves: vec![],
        };
        let result = compiler.compile_version_spec(&version);
        assert!(result.contains("V2_Capability_1 =="));
        assert!(result.contains("V2_Capability_2 =="));
        assert!(result.contains("(speed) >= (100)"));
        assert!(result.contains("(accuracy) >= (90)"));
        assert!(result.contains("V2_Capability_1 /\\ V2_Capability_2"));
    }

    #[test]
    fn test_compile_version_spec_empty() {
        let compiler = TlaPlusCompiler::new("Test");
        let version = VersionSpec {
            name: "V2".to_string(),
            improves: "V1".to_string(),
            capabilities: vec![],
            preserves: vec![],
        };
        let result = compiler.compile_version_spec(&version);
        assert!(result.contains("Version Improvement: V2 improves V1"));
        // No combined theorem for empty version
        assert!(!result.contains("V2_ImprovesOver_V1"));
    }

    // ============ compile_capability_spec tests ============

    #[test]
    fn test_compile_capability_spec_basic() {
        let compiler = TlaPlusCompiler::new("Test");
        let capability = CapabilitySpec {
            name: "DasherCapability".to_string(),
            abilities: vec![CapabilityAbility {
                name: "verify_code".to_string(),
                params: vec![Param {
                    name: "code".to_string(),
                    ty: Type::Named("Code".to_string()),
                }],
                return_type: Some(Type::Named("Result".to_string())),
            }],
            requires: vec![Expr::Compare(
                Box::new(Expr::Var("x".to_string())),
                ComparisonOp::Gt,
                Box::new(Expr::Int(0)),
            )],
        };
        let result = compiler.compile_capability_spec(&capability);
        assert!(result.contains("Capability Specification: DasherCapability"));
        assert!(result.contains("CONSTANTS DasherCapability"));
        assert!(result.contains("DasherCapability_verify_code(code) =="));
        assert!(result.contains("code \\in Code"));
        assert!(result.contains("Result"));
        assert!(result.contains("DasherCapability_Requirement_1 =="));
        assert!(result.contains("(x) > (0)"));
        assert!(result.contains("DasherCapability_Requirements =="));
    }

    #[test]
    fn test_compile_capability_spec_multiple_abilities() {
        let compiler = TlaPlusCompiler::new("Test");
        let capability = CapabilitySpec {
            name: "Cap".to_string(),
            abilities: vec![
                CapabilityAbility {
                    name: "ability1".to_string(),
                    params: vec![Param {
                        name: "x".to_string(),
                        ty: Type::Named("Int".to_string()),
                    }],
                    return_type: Some(Type::Named("Bool".to_string())),
                },
                CapabilityAbility {
                    name: "ability2".to_string(),
                    params: vec![
                        Param {
                            name: "a".to_string(),
                            ty: Type::Named("A".to_string()),
                        },
                        Param {
                            name: "b".to_string(),
                            ty: Type::Named("B".to_string()),
                        },
                    ],
                    return_type: Some(Type::Named("C".to_string())),
                },
            ],
            requires: vec![],
        };
        let result = compiler.compile_capability_spec(&capability);
        assert!(result.contains("Cap_ability1(x) =="));
        assert!(result.contains("Cap_ability2(a, b) =="));
        assert!(result.contains("a \\in A, b \\in B"));
    }

    #[test]
    fn test_compile_capability_spec_no_params() {
        let compiler = TlaPlusCompiler::new("Test");
        let capability = CapabilitySpec {
            name: "Simple".to_string(),
            abilities: vec![CapabilityAbility {
                name: "get_status".to_string(),
                params: vec![],
                return_type: Some(Type::Named("Status".to_string())),
            }],
            requires: vec![],
        };
        let result = compiler.compile_capability_spec(&capability);
        assert!(result.contains("Simple_get_status == (* returns Status *)"));
        // Should not have parameter list
        assert!(!result.contains("Simple_get_status("));
    }

    #[test]
    fn test_compile_capability_spec_no_return() {
        let compiler = TlaPlusCompiler::new("Test");
        let capability = CapabilitySpec {
            name: "Action".to_string(),
            abilities: vec![CapabilityAbility {
                name: "do_action".to_string(),
                params: vec![Param {
                    name: "input".to_string(),
                    ty: Type::Named("Input".to_string()),
                }],
                return_type: None, // No return type
            }],
            requires: vec![],
        };
        let result = compiler.compile_capability_spec(&capability);
        assert!(result.contains("Action_do_action(input) =="));
        // Should use {} (Unit) for no return type
        assert!(result.contains("=> {}"));
    }

    #[test]
    fn test_compile_capability_spec_multiple_requirements() {
        let compiler = TlaPlusCompiler::new("Test");
        let capability = CapabilitySpec {
            name: "Constrained".to_string(),
            abilities: vec![],
            requires: vec![
                Expr::Compare(
                    Box::new(Expr::Var("x".to_string())),
                    ComparisonOp::Ge,
                    Box::new(Expr::Int(0)),
                ),
                Expr::Compare(
                    Box::new(Expr::Var("y".to_string())),
                    ComparisonOp::Lt,
                    Box::new(Expr::Int(100)),
                ),
            ],
        };
        let result = compiler.compile_capability_spec(&capability);
        assert!(result.contains("Constrained_Requirement_1 =="));
        assert!(result.contains("Constrained_Requirement_2 =="));
        assert!(result.contains("Constrained_Requirements =="));
        assert!(result.contains("Constrained_Requirement_1 /\\ Constrained_Requirement_2"));
    }

    // ============ compile_module with version/capability tests ============

    #[test]
    fn test_compile_module_with_version() {
        let version = VersionSpec {
            name: "V2".to_string(),
            improves: "V1".to_string(),
            capabilities: vec![CapabilityClause {
                expr: Expr::Bool(true),
            }],
            preserves: vec![],
        };
        let spec = make_typed_spec(vec![Property::Version(version)]);
        let compiler = TlaPlusCompiler::new("VersionTest");
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("---- MODULE VersionTest ----"));
        assert!(result.code.contains("Version Improvement: V2 improves V1"));
        assert!(result.code.contains("V2_Capability_1 =="));
        assert!(result.code.ends_with("===="));
    }

    #[test]
    fn test_compile_module_with_capability() {
        let capability = CapabilitySpec {
            name: "MyCap".to_string(),
            abilities: vec![CapabilityAbility {
                name: "foo".to_string(),
                params: vec![],
                return_type: Some(Type::Named("Bool".to_string())),
            }],
            requires: vec![],
        };
        let spec = make_typed_spec(vec![Property::Capability(capability)]);
        let compiler = TlaPlusCompiler::new("CapTest");
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("---- MODULE CapTest ----"));
        assert!(result.code.contains("Capability Specification: MyCap"));
        assert!(result.code.contains("CONSTANTS MyCap"));
        assert!(result.code.contains("MyCap_foo =="));
        assert!(result.code.ends_with("===="));
    }

    #[test]
    fn test_compile_module_with_version_and_invariant() {
        let version = VersionSpec {
            name: "V2".to_string(),
            improves: "V1".to_string(),
            capabilities: vec![],
            preserves: vec![PreservesClause {
                property: Expr::Var("safety".to_string()),
            }],
        };
        let spec = make_typed_spec(vec![
            Property::Invariant(Invariant {
                name: "SafetyInv".to_string(),
                body: Expr::Bool(true),
            }),
            Property::Version(version),
        ]);
        let compiler = TlaPlusCompiler::new("Combined");
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("Invariant: SafetyInv"));
        assert!(result.code.contains("Version Improvement: V2 improves V1"));
        assert!(result.code.contains("V2_Preserves_1 =="));
    }

    // =========================================================================
    // Kani proofs for TLA+ compiler correctness
    // =========================================================================

    /// Prove that compile_expr never produces empty output for integer literals.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_compile_expr_int_nonempty() {
        let compiler = TlaPlusCompiler::new("Test");
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_compile_expr_bool_nonempty() {
        let compiler = TlaPlusCompiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Also verify correct TLA+ syntax
        assert!(result == "TRUE" || result == "FALSE");
    }

    /// Prove that compile_type always produces valid TLA+ type syntax.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_compile_type_named_nonempty() {
        let compiler = TlaPlusCompiler::new("Test");
        // Test with a bounded-length type name
        let ty = Type::Named("T".to_string());
        let result = compiler.compile_type(&ty);
        assert!(!result.is_empty());
    }

    /// Prove that TLA+ module header is always well-formed.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_module_header_structure() {
        let compiler = TlaPlusCompiler::new("M");
        let spec = make_typed_spec(vec![]);
        let result = compiler.compile_module(&spec);
        // Module must start with header
        assert!(result.code.starts_with("---- MODULE M ----"));
        // Module must end with footer
        assert!(result.code.ends_with("===="));
        // Backend must be TLA+
        assert_eq!(result.backend, "TLA+");
    }

    /// Prove that temporal always operator produces correct TLA+ syntax.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_compile_temporal_always_syntax() {
        let compiler = TlaPlusCompiler::new("Test");
        let inner = TemporalExpr::Atom(Expr::Bool(true));
        let result = compiler.compile_temporal_expr(&TemporalExpr::Always(Box::new(inner)));
        // TLA+ always is []
        assert!(result.starts_with("[]"));
    }

    /// Prove that temporal eventually operator produces correct TLA+ syntax.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_compile_temporal_eventually_syntax() {
        let compiler = TlaPlusCompiler::new("Test");
        let inner = TemporalExpr::Atom(Expr::Bool(true));
        let result = compiler.compile_temporal_expr(&TemporalExpr::Eventually(Box::new(inner)));
        // TLA+ eventually is <>
        assert!(result.starts_with("<>"));
    }

    /// Prove that weak fairness generates WF_ prefix.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_weak_fairness_syntax() {
        let compiler = TlaPlusCompiler::new("Test");
        let constraint = FairnessConstraint {
            kind: FairnessKind::Weak,
            action: "A".to_string(),
            vars: None,
        };
        let result = compiler.compile_fairness(&constraint);
        assert!(result.starts_with("WF_"));
    }

    /// Prove that strong fairness generates SF_ prefix.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_strong_fairness_syntax() {
        let compiler = TlaPlusCompiler::new("Test");
        let constraint = FairnessConstraint {
            kind: FairnessKind::Strong,
            action: "A".to_string(),
            vars: None,
        };
        let result = compiler.compile_fairness(&constraint);
        assert!(result.starts_with("SF_"));
    }

    /// Prove that comparison operators are compiled to valid TLA+ operators.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_comparison_op_valid() {
        let compiler = TlaPlusCompiler::new("Test");
        let ops = [
            ComparisonOp::Eq,
            ComparisonOp::Ne,
            ComparisonOp::Lt,
            ComparisonOp::Le,
            ComparisonOp::Gt,
            ComparisonOp::Ge,
        ];
        let idx: usize = kani::any();
        kani::assume(idx < ops.len());
        let op = ops[idx];

        let expr = Expr::Compare(
            Box::new(Expr::Var("x".to_string())),
            op,
            Box::new(Expr::Var("y".to_string())),
        );
        let result = compiler.compile_expr(&expr);
        // Result must contain at least one comparison operator
        assert!(
            result.contains('=')
                || result.contains('#')
                || result.contains('<')
                || result.contains('>')
        );
    }

    // ========================================================================
    // Phase 17.3: Graph predicate compilation tests
    // ========================================================================

    #[test]
    fn test_compile_is_acyclic() {
        let compiler = TlaPlusCompiler::new("Test");
        let expr = Expr::App("is_acyclic".to_string(), vec![Expr::Var("g".to_string())]);
        let result = compiler.compile_expr(&expr);
        // Should compile to TLA+ acyclicity check
        assert!(result.contains("g.nodes"));
        assert!(result.contains("g.edges"));
        assert!(result.contains("~\\E"));
    }

    #[test]
    fn test_compile_is_dag() {
        let compiler = TlaPlusCompiler::new("Test");
        let expr = Expr::App("is_dag".to_string(), vec![Expr::Var("graph".to_string())]);
        let result = compiler.compile_expr(&expr);
        // is_dag is an alias for is_acyclic
        assert!(result.contains("graph.nodes"));
        assert!(result.contains("graph.edges"));
    }

    #[test]
    fn test_compile_has_path() {
        let compiler = TlaPlusCompiler::new("Test");
        let expr = Expr::App(
            "has_path".to_string(),
            vec![
                Expr::Var("g".to_string()),
                Expr::Var("n1".to_string()),
                Expr::Var("n2".to_string()),
            ],
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("Reachable"));
        assert!(result.contains("n1"));
        assert!(result.contains("n2"));
        assert!(result.contains("g.edges"));
    }

    #[test]
    fn test_compile_in_graph() {
        let compiler = TlaPlusCompiler::new("Test");
        let expr = Expr::App(
            "in_graph".to_string(),
            vec![Expr::Var("node".to_string()), Expr::Var("g".to_string())],
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("node"));
        assert!(result.contains("\\in"));
        assert!(result.contains("g.nodes"));
    }

    #[test]
    fn test_compile_edge_exists() {
        let compiler = TlaPlusCompiler::new("Test");
        let expr = Expr::App(
            "edge_exists".to_string(),
            vec![
                Expr::Var("g".to_string()),
                Expr::Var("from".to_string()),
                Expr::Var("to".to_string()),
            ],
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("<<from, to>>"));
        assert!(result.contains("\\in"));
        assert!(result.contains("g.edges"));
    }

    #[test]
    fn test_compile_successors() {
        let compiler = TlaPlusCompiler::new("Test");
        let expr = Expr::App(
            "successors".to_string(),
            vec![Expr::Var("g".to_string()), Expr::Var("n".to_string())],
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("g.nodes"));
        assert!(result.contains("<<n, n>>"));
    }

    #[test]
    fn test_compile_predecessors() {
        let compiler = TlaPlusCompiler::new("Test");
        let expr = Expr::App(
            "predecessors".to_string(),
            vec![Expr::Var("g".to_string()), Expr::Var("n".to_string())],
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("g.nodes"));
        assert!(result.contains("<<n, n>>"));
    }

    #[test]
    fn test_compile_node_count() {
        let compiler = TlaPlusCompiler::new("Test");
        let expr = Expr::App("node_count".to_string(), vec![Expr::Var("g".to_string())]);
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("Cardinality"));
        assert!(result.contains("g.nodes"));
    }

    #[test]
    fn test_compile_preserves_completed() {
        let compiler = TlaPlusCompiler::new("Test");
        let expr = Expr::App(
            "preserves_completed".to_string(),
            vec![Expr::Var("g".to_string()), Expr::Var("g_prime".to_string())],
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("\\A n"));
        assert!(result.contains("g.nodes"));
        assert!(result.contains("completed"));
    }

    #[test]
    fn test_compile_node_status() {
        let compiler = TlaPlusCompiler::new("Test");
        let expr = Expr::App("completed".to_string(), vec![Expr::Var("node".to_string())]);
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("node.status"));
        assert!(result.contains("completed"));
    }

    // =========================================================================
    // Distributed property compilation tests
    // =========================================================================

    #[test]
    fn test_compile_distributed_invariant_basic() {
        use crate::ast::DistributedInvariant;

        let dist_inv = DistributedInvariant {
            name: "proof_consensus".to_string(),
            body: Expr::ForAll {
                var: "d1".to_string(),
                ty: Some(Type::Named("Dasher".to_string())),
                body: Box::new(Expr::ForAll {
                    var: "d2".to_string(),
                    ty: Some(Type::Named("Dasher".to_string())),
                    body: Box::new(Expr::Compare(
                        Box::new(Expr::FieldAccess(
                            Box::new(Expr::Var("d1".to_string())),
                            "result".to_string(),
                        )),
                        ComparisonOp::Eq,
                        Box::new(Expr::FieldAccess(
                            Box::new(Expr::Var("d2".to_string())),
                            "result".to_string(),
                        )),
                    )),
                }),
            },
        };

        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_distributed_invariant(&dist_inv);

        assert!(result.contains("Distributed Invariant: proof_consensus"));
        assert!(result.contains("Multi-agent invariant"));
        assert!(result.contains("proof_consensus =="));
        assert!(result.contains("\\A d1 \\in Dasher"));
        assert!(result.contains("\\A d2 \\in Dasher"));
    }

    #[test]
    fn test_compile_distributed_invariant_simple() {
        use crate::ast::DistributedInvariant;

        let dist_inv = DistributedInvariant {
            name: "no_split_brain".to_string(),
            body: Expr::App(
                "can_communicate".to_string(),
                vec![Expr::Var("d1".to_string()), Expr::Var("d2".to_string())],
            ),
        };

        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_distributed_invariant(&dist_inv);

        assert!(result.contains("Distributed Invariant: no_split_brain"));
        assert!(result.contains("no_split_brain =="));
        assert!(result.contains("can_communicate(d1, d2)"));
    }

    #[test]
    fn test_compile_distributed_temporal_basic() {
        use crate::ast::DistributedTemporal;

        let dist_temp = DistributedTemporal {
            name: "version_convergence".to_string(),
            body: TemporalExpr::Eventually(Box::new(TemporalExpr::Atom(Expr::ForAll {
                var: "d1".to_string(),
                ty: Some(Type::Named("Dasher".to_string())),
                body: Box::new(Expr::ForAll {
                    var: "d2".to_string(),
                    ty: Some(Type::Named("Dasher".to_string())),
                    body: Box::new(Expr::Compare(
                        Box::new(Expr::FieldAccess(
                            Box::new(Expr::Var("d1".to_string())),
                            "version".to_string(),
                        )),
                        ComparisonOp::Eq,
                        Box::new(Expr::FieldAccess(
                            Box::new(Expr::Var("d2".to_string())),
                            "version".to_string(),
                        )),
                    )),
                }),
            }))),
            fairness: vec![],
        };

        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_distributed_temporal(&dist_temp);

        assert!(result.contains("Distributed Temporal: version_convergence"));
        assert!(result.contains("Multi-agent temporal property"));
        assert!(result.contains("version_convergence =="));
        assert!(result.contains("<>")); // eventually in TLA+
    }

    #[test]
    fn test_compile_distributed_temporal_with_fairness() {
        use crate::ast::{DistributedTemporal, FairnessConstraint, FairnessKind};

        let dist_temp = DistributedTemporal {
            name: "eventual_agreement".to_string(),
            body: TemporalExpr::Eventually(Box::new(TemporalExpr::Atom(Expr::Var(
                "agreed".to_string(),
            )))),
            fairness: vec![FairnessConstraint {
                kind: FairnessKind::Weak,
                action: "Next".to_string(),
                vars: None,
            }],
        };

        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_distributed_temporal(&dist_temp);

        assert!(result.contains("Distributed Temporal: eventual_agreement"));
        assert!(result.contains("Fairness: WF_vars (Next)"));
        assert!(result.contains("eventual_agreement =="));
        assert!(result.contains("WF_vars(Next)"));
    }

    #[test]
    fn test_compile_distributed_temporal_always() {
        use crate::ast::DistributedTemporal;

        let dist_temp = DistributedTemporal {
            name: "global_safety".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Var("safe".to_string())))),
            fairness: vec![],
        };

        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_distributed_temporal(&dist_temp);

        assert!(result.contains("global_safety =="));
        assert!(result.contains("[]")); // always in TLA+
    }

    #[test]
    fn test_compile_module_with_distributed_invariant() {
        use crate::ast::DistributedInvariant;

        let dist_inv = DistributedInvariant {
            name: "consensus".to_string(),
            body: Expr::Bool(true),
        };
        let spec = make_typed_spec(vec![Property::DistributedInvariant(dist_inv)]);

        let compiler = TlaPlusCompiler::new("DistTest");
        let result = compiler.compile_module(&spec);

        assert!(result.code.contains("---- MODULE DistTest ----"));
        assert!(result.code.contains("Distributed Invariant: consensus"));
        assert!(result.code.contains("consensus =="));
    }

    #[test]
    fn test_compile_module_with_distributed_temporal() {
        use crate::ast::DistributedTemporal;

        let dist_temp = DistributedTemporal {
            name: "eventual_sync".to_string(),
            body: TemporalExpr::Eventually(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
            fairness: vec![],
        };
        let spec = make_typed_spec(vec![Property::DistributedTemporal(dist_temp)]);

        let compiler = TlaPlusCompiler::new("DistTest");
        let result = compiler.compile_module(&spec);

        assert!(result.code.contains("---- MODULE DistTest ----"));
        assert!(result.code.contains("Distributed Temporal: eventual_sync"));
        assert!(result.code.contains("eventual_sync =="));
    }

    // =========================================================================
    // Phase 17.6: Self-improvement constructs compilation tests
    // =========================================================================

    #[test]
    fn test_compile_improvement_proposal_basic() {
        use crate::ast::ImprovementProposal;

        let proposal = ImprovementProposal {
            name: "CodeOptimization".to_string(),
            target: Expr::FieldAccess(
                Box::new(Expr::Var("Dasher".to_string())),
                "verify_rust_code".to_string(),
            ),
            improves: vec![Expr::Compare(
                Box::new(Expr::Var("execution_speed".to_string())),
                ComparisonOp::Ge,
                Box::new(Expr::Binary(
                    Box::new(Expr::Float(1.1)),
                    crate::ast::BinaryOp::Mul,
                    Box::new(Expr::Var("baseline".to_string())),
                )),
            )],
            preserves: vec![Expr::Var("soundness".to_string())],
            requires: vec![Expr::App(
                "valid_rust_syntax".to_string(),
                vec![Expr::Var("new_code".to_string())],
            )],
        };

        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_improvement_proposal(&proposal);

        assert!(result.contains("Improvement Proposal: CodeOptimization"));
        assert!(result.contains("CodeOptimization_Target =="));
        assert!(result.contains("Dasher.verify_rust_code"));
        assert!(result.contains("CodeOptimization_Improves_1 =="));
        assert!(result.contains("execution_speed"));
        assert!(result.contains("CodeOptimization_Preserves_1 =="));
        assert!(result.contains("soundness"));
        assert!(result.contains("CodeOptimization_Requires_1 =="));
        assert!(result.contains("valid_rust_syntax"));
        assert!(result.contains("CodeOptimization_Valid =="));
    }

    #[test]
    fn test_compile_improvement_proposal_empty() {
        use crate::ast::ImprovementProposal;

        let proposal = ImprovementProposal {
            name: "NoChange".to_string(),
            target: Expr::Var("target".to_string()),
            improves: vec![],
            preserves: vec![],
            requires: vec![],
        };

        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_improvement_proposal(&proposal);

        assert!(result.contains("Improvement Proposal: NoChange"));
        assert!(result.contains("NoChange_Target =="));
        // Should not contain combined validity since no clauses
        assert!(!result.contains("NoChange_Valid =="));
    }

    #[test]
    fn test_compile_verification_gate_basic() {
        use crate::ast::{GateCheck, VerificationGate};

        let gate = VerificationGate {
            name: "SelfModGate".to_string(),
            inputs: vec![
                Param {
                    name: "current".to_string(),
                    ty: Type::Named("DasherVersion".to_string()),
                },
                Param {
                    name: "proposed".to_string(),
                    ty: Type::Named("Improvement".to_string()),
                },
            ],
            checks: vec![
                GateCheck {
                    name: "soundness".to_string(),
                    condition: Expr::App(
                        "verify_soundness".to_string(),
                        vec![
                            Expr::Var("current".to_string()),
                            Expr::Var("proposed".to_string()),
                        ],
                    ),
                },
                GateCheck {
                    name: "capability".to_string(),
                    condition: Expr::App(
                        "verify_capabilities".to_string(),
                        vec![
                            Expr::Var("current".to_string()),
                            Expr::Var("proposed".to_string()),
                        ],
                    ),
                },
            ],
            on_pass: Expr::App("apply".to_string(), vec![Expr::Var("proposed".to_string())]),
            on_fail: Expr::App(
                "reject".to_string(),
                vec![Expr::Var("proposed".to_string())],
            ),
        };

        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_verification_gate(&gate);

        assert!(result.contains("Verification Gate: SelfModGate"));
        assert!(result.contains("VARIABLES selfmodgate"));
        assert!(result.contains("SelfModGate_InputTypes =="));
        assert!(result.contains("current \\in DasherVersion"));
        assert!(result.contains("SelfModGate_Check_soundness =="));
        assert!(result.contains("SelfModGate_Check_capability =="));
        assert!(result.contains("SelfModGate_AllChecksPass =="));
        assert!(result.contains("SelfModGate_OnPass =="));
        assert!(result.contains("SelfModGate_OnFail =="));
        assert!(result.contains("SelfModGate_Next =="));
    }

    #[test]
    fn test_compile_verification_gate_no_inputs() {
        use crate::ast::{GateCheck, VerificationGate};

        let gate = VerificationGate {
            name: "SimpleGate".to_string(),
            inputs: vec![],
            checks: vec![GateCheck {
                name: "check1".to_string(),
                condition: Expr::Bool(true),
            }],
            on_pass: Expr::Var("success".to_string()),
            on_fail: Expr::Var("failure".to_string()),
        };

        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_verification_gate(&gate);

        assert!(result.contains("Verification Gate: SimpleGate"));
        assert!(!result.contains("SimpleGate_InputTypes =="));
        assert!(result.contains("SimpleGate_Check_check1 =="));
    }

    #[test]
    fn test_compile_rollback_spec_basic() {
        use crate::ast::{RollbackAction, RollbackSpec};

        let rollback = RollbackSpec {
            name: "SafeRollback".to_string(),
            state: vec![
                Param {
                    name: "current".to_string(),
                    ty: Type::Named("DasherVersion".to_string()),
                },
                Param {
                    name: "history".to_string(),
                    ty: Type::List(Box::new(Type::Named("DasherVersion".to_string()))),
                },
            ],
            invariants: vec![Expr::Compare(
                Box::new(Expr::App(
                    "len".to_string(),
                    vec![Expr::Var("history".to_string())],
                )),
                ComparisonOp::Gt,
                Box::new(Expr::Int(0)),
            )],
            trigger: Expr::Or(
                Box::new(Expr::Var("verification_failed".to_string())),
                Box::new(Expr::Var("runtime_error".to_string())),
            ),
            action: RollbackAction {
                assignments: vec![(
                    "current".to_string(),
                    Expr::App("last".to_string(), vec![Expr::Var("history".to_string())]),
                )],
                ensure: Some(Expr::App(
                    "verified".to_string(),
                    vec![Expr::Var("current".to_string())],
                )),
            },
            guarantees: vec![Expr::App(
                "always".to_string(),
                vec![Expr::App(
                    "verified".to_string(),
                    vec![Expr::Var("current".to_string())],
                )],
            )],
        };

        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_rollback_spec(&rollback);

        assert!(result.contains("Rollback Specification: SafeRollback"));
        assert!(result.contains("VARIABLES current, history"));
        assert!(result.contains("SafeRollback_StateTypes =="));
        assert!(result.contains("SafeRollback_Invariant_1 =="));
        assert!(result.contains("SafeRollback_AllInvariants =="));
        assert!(result.contains("SafeRollback_Trigger =="));
        assert!(result.contains("verification_failed"));
        assert!(result.contains("runtime_error"));
        assert!(result.contains("SafeRollback_Action =="));
        assert!(result.contains("current' = last(history)"));
        assert!(result.contains("SafeRollback_Guarantee_1 =="));
        assert!(result.contains("SafeRollback_Spec =="));
    }

    #[test]
    fn test_compile_rollback_spec_minimal() {
        use crate::ast::{RollbackAction, RollbackSpec};

        let rollback = RollbackSpec {
            name: "MinimalRollback".to_string(),
            state: vec![],
            invariants: vec![],
            trigger: Expr::Var("error".to_string()),
            action: RollbackAction {
                assignments: vec![],
                ensure: None,
            },
            guarantees: vec![],
        };

        let compiler = TlaPlusCompiler::new("Test");
        let result = compiler.compile_rollback_spec(&rollback);

        assert!(result.contains("Rollback Specification: MinimalRollback"));
        assert!(result.contains("MinimalRollback_Trigger =="));
        assert!(result.contains("MinimalRollback_Action =="));
        // Should not have spec since no invariants or guarantees
        assert!(!result.contains("MinimalRollback_Spec =="));
    }

    #[test]
    fn test_compile_module_with_improvement_proposal() {
        use crate::ast::ImprovementProposal;

        let proposal = ImprovementProposal {
            name: "TestProposal".to_string(),
            target: Expr::Var("target".to_string()),
            improves: vec![Expr::Bool(true)],
            preserves: vec![],
            requires: vec![],
        };
        let spec = make_typed_spec(vec![Property::ImprovementProposal(proposal)]);

        let compiler = TlaPlusCompiler::new("ImprovementTest");
        let result = compiler.compile_module(&spec);

        assert!(result.code.contains("---- MODULE ImprovementTest ----"));
        assert!(result.code.contains("Improvement Proposal: TestProposal"));
        assert!(result.code.contains("TestProposal_Target =="));
    }

    #[test]
    fn test_compile_module_with_verification_gate() {
        use crate::ast::{GateCheck, VerificationGate};

        let gate = VerificationGate {
            name: "TestGate".to_string(),
            inputs: vec![],
            checks: vec![GateCheck {
                name: "check".to_string(),
                condition: Expr::Bool(true),
            }],
            on_pass: Expr::Var("pass".to_string()),
            on_fail: Expr::Var("fail".to_string()),
        };
        let spec = make_typed_spec(vec![Property::VerificationGate(gate)]);

        let compiler = TlaPlusCompiler::new("GateTest");
        let result = compiler.compile_module(&spec);

        assert!(result.code.contains("---- MODULE GateTest ----"));
        assert!(result.code.contains("Verification Gate: TestGate"));
    }

    #[test]
    fn test_compile_module_with_rollback() {
        use crate::ast::{RollbackAction, RollbackSpec};

        let rollback = RollbackSpec {
            name: "TestRollback".to_string(),
            state: vec![],
            invariants: vec![],
            trigger: Expr::Var("trigger".to_string()),
            action: RollbackAction {
                assignments: vec![],
                ensure: None,
            },
            guarantees: vec![],
        };
        let spec = make_typed_spec(vec![Property::Rollback(rollback)]);

        let compiler = TlaPlusCompiler::new("RollbackTest");
        let result = compiler.compile_module(&spec);

        assert!(result.code.contains("---- MODULE RollbackTest ----"));
        assert!(result.code.contains("Rollback Specification: TestRollback"));
    }
}
