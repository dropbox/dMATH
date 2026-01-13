//! Abstract Syntax Tree for USL
//!
//! This module defines the AST types for the Unified Specification Language.
//! See docs/DESIGN.md for the full grammar specification.

use serde::{Deserialize, Serialize};

/// A complete USL specification
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Spec {
    /// Type definitions in this specification
    pub types: Vec<TypeDef>,
    /// Properties to verify
    pub properties: Vec<Property>,
}

impl Spec {
    /// Returns unique property kinds in this specification.
    ///
    /// This is useful for determining which property types are present in a spec,
    /// enabling domain-aware backend selection and weighted consensus.
    ///
    /// # Example
    ///
    /// ```
    /// use dashprove_usl::ast::{Spec, Property, Theorem, Invariant, Expr};
    ///
    /// let spec = Spec {
    ///     types: vec![],
    ///     properties: vec![
    ///         Property::Theorem(Theorem {
    ///             name: "thm1".to_string(),
    ///             body: Expr::Bool(true),
    ///         }),
    ///         Property::Invariant(Invariant {
    ///             name: "inv1".to_string(),
    ///             body: Expr::Bool(true),
    ///         }),
    ///     ],
    /// };
    ///
    /// let kinds = spec.property_kinds();
    /// assert!(kinds.contains(&"theorem"));
    /// assert!(kinds.contains(&"invariant"));
    /// ```
    #[must_use]
    pub fn property_kinds(&self) -> Vec<&'static str> {
        let mut kinds: Vec<&'static str> =
            self.properties.iter().map(|p| p.property_kind()).collect();
        kinds.sort_unstable();
        kinds.dedup();
        kinds
    }

    /// Returns a map of property names to their kinds.
    ///
    /// This is useful for tracking which property type each named property belongs to,
    /// enabling property-type-specific result merging in domain-weighted consensus.
    #[must_use]
    pub fn property_kind_map(&self) -> std::collections::HashMap<String, &'static str> {
        self.properties
            .iter()
            .map(|p| (p.name(), p.property_kind()))
            .collect()
    }
}

/// Type definition: `type Name = { field: Type, ... }`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TypeDef {
    /// Name of the type
    pub name: String,
    /// Fields in this type
    pub fields: Vec<Field>,
}

/// Field in a type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Field {
    /// Field name
    pub name: String,
    /// Field type
    pub ty: Type,
}

/// Type expression
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Type {
    /// Named type: `Graph`, `State`, etc.
    Named(String),
    /// Set type: `Set<T>`
    Set(Box<Type>),
    /// List type: `List<T>`
    List(Box<Type>),
    /// Map type: `Map<K, V>`
    Map(Box<Type>, Box<Type>),
    /// Relation type: `Relation<A, B>`
    Relation(Box<Type>, Box<Type>),
    /// Function type: `A -> B`
    Function(Box<Type>, Box<Type>),
    /// Result type: `Result<T>`
    Result(Box<Type>),
    /// Graph type: `Graph<N, E>` for DashFlow execution graphs
    /// N is the node type, E is the edge type
    Graph(Box<Type>, Box<Type>),
    /// Path type: `Path<N>` for paths in a graph
    /// N is the node type
    Path(Box<Type>),
    /// Unit type (for functions returning nothing)
    Unit,
}

/// Property to verify
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Property {
    /// Mathematical theorem (LEAN 4, Coq)
    Theorem(Theorem),
    /// Temporal logic property (TLA+)
    Temporal(Temporal),
    /// Pre/post contract (Kani)
    Contract(Contract),
    /// State invariant (LEAN, Alloy)
    Invariant(Invariant),
    /// Refinement relation
    Refinement(Refinement),
    /// Probabilistic property
    Probabilistic(Probabilistic),
    /// Security property (confidentiality, integrity)
    Security(Security),
    /// Semantic/fuzzy property (embedding-based checks)
    Semantic(SemanticProperty),
    /// Platform API constraint (external API state machines)
    PlatformApi(PlatformApi),
    /// Bisimulation (behavioral equivalence checking)
    Bisimulation(Bisimulation),
    /// Version improvement specification (for recursive self-improvement)
    Version(VersionSpec),
    /// Capability specification (what a system can do)
    Capability(CapabilitySpec),
    /// Distributed invariant (multi-agent coordination)
    DistributedInvariant(DistributedInvariant),
    /// Distributed temporal (multi-agent temporal properties)
    DistributedTemporal(DistributedTemporal),
    /// Composed theorem (combines multiple existing properties)
    Composed(ComposedTheorem),
    /// Improvement proposal specification (Phase 17.6.1)
    ImprovementProposal(ImprovementProposal),
    /// Verification gate specification (Phase 17.6.2)
    VerificationGate(VerificationGate),
    /// Rollback specification (Phase 17.6.3)
    Rollback(RollbackSpec),
}

impl Property {
    /// Returns the name of this property.
    ///
    /// For most property types, this returns the declared name.
    /// For contracts, this returns the type path joined with `::` (e.g., `Type::method`).
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::Theorem(t) => t.name.clone(),
            Self::Temporal(t) => t.name.clone(),
            Self::Contract(c) => c.type_path.join("::"),
            Self::Invariant(i) => i.name.clone(),
            Self::Refinement(r) => r.name.clone(),
            Self::Probabilistic(p) => p.name.clone(),
            Self::Security(s) => s.name.clone(),
            Self::Semantic(s) => s.name.clone(),
            Self::PlatformApi(p) => p.name.clone(),
            Self::Bisimulation(b) => b.name.clone(),
            Self::Version(v) => v.name.clone(),
            Self::Capability(c) => c.name.clone(),
            Self::DistributedInvariant(d) => d.name.clone(),
            Self::DistributedTemporal(d) => d.name.clone(),
            Self::Composed(c) => c.name.clone(),
            Self::ImprovementProposal(i) => i.name.clone(),
            Self::VerificationGate(v) => v.name.clone(),
            Self::Rollback(r) => r.name.clone(),
        }
    }

    /// Returns the property kind as a lowercase string.
    ///
    /// This string is compatible with `property_type_from_string()` in dashprove-learning,
    /// enabling automatic property type inference for domain-weighted consensus.
    ///
    /// # Returns
    ///
    /// A lowercase string identifying the property kind:
    /// - `"theorem"` for mathematical theorems
    /// - `"temporal"` for temporal logic properties
    /// - `"contract"` for pre/post contracts
    /// - `"invariant"` for state invariants
    /// - `"refinement"` for refinement relations
    /// - `"probabilistic"` for probabilistic properties
    /// - `"security"` for security/authentication properties
    /// - `"semantic"` for embedding-based semantic checks
    /// - `"platform_api"` for platform API constraints
    /// - `"refinement"` for bisimulation (behavioral equivalence)
    /// - `"theorem"` for version/capability specs (theorem-like verification)
    /// - `"distributed_invariant"` for multi-agent invariants
    /// - `"distributed_temporal"` for multi-agent temporal properties
    #[must_use]
    pub fn property_kind(&self) -> &'static str {
        match self {
            Self::Theorem(_) => "theorem",
            Self::Temporal(_) => "temporal",
            Self::Contract(_) => "contract",
            Self::Invariant(_) => "invariant",
            Self::Refinement(_) => "refinement",
            Self::Probabilistic(_) => "probabilistic",
            Self::Security(_) => "security",
            Self::Semantic(_) => "semantic",
            Self::PlatformApi(_) => "platform_api",
            Self::Bisimulation(_) => "refinement", // behavioral equivalence is refinement-like
            Self::Version(_) => "theorem",         // version specs verify theorem-like properties
            Self::Capability(_) => "theorem", // capability specs verify theorem-like properties
            Self::DistributedInvariant(_) => "distributed_invariant", // multi-agent invariants
            Self::DistributedTemporal(_) => "distributed_temporal", // multi-agent temporal
            Self::Composed(_) => "composed",  // composed theorems
            Self::ImprovementProposal(_) => "improvement_proposal", // self-improvement proposals
            Self::VerificationGate(_) => "verification_gate", // immutable verification checkpoints
            Self::Rollback(_) => "rollback",  // safe rollback specifications
        }
    }
}

/// Mathematical theorem (compiles to LEAN, Coq)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Theorem {
    /// Theorem name
    pub name: String,
    /// Theorem body expression
    pub body: Expr,
}

/// Temporal property (compiles to TLA+)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Temporal {
    /// Property name
    pub name: String,
    /// Temporal formula
    pub body: TemporalExpr,
    /// Fairness constraints for liveness checking
    /// Without fairness, TLC may find "stuttering" counterexamples
    pub fairness: Vec<FairnessConstraint>,
}

/// Contract with pre/post conditions (compiles to Kani)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Contract {
    /// Type path (e.g., [`Type`, `method`])
    pub type_path: Vec<String>,
    /// Function parameters
    pub params: Vec<Param>,
    /// Return type (optional)
    pub return_type: Option<Type>,
    /// Preconditions (requires clauses)
    pub requires: Vec<Expr>,
    /// Postconditions for success (ensures clauses)
    pub ensures: Vec<Expr>,
    /// Postconditions for error case
    pub ensures_err: Vec<Expr>,
    /// Frame conditions - memory locations the function may modify (assigns clause)
    /// Each expression represents a memory location that may be modified.
    /// An empty list means "may modify anything" (no frame restriction).
    /// `assigns \nothing` is represented as a single Expr::Var("nothing").
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assigns: Vec<Expr>,
    /// Memory locations that may be allocated by this function (allocates clause)
    /// Used for heap verification. `allocates \nothing` means no dynamic allocation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allocates: Vec<Expr>,
    /// Memory locations that may be freed by this function (frees clause)
    /// Used for heap verification. `frees \nothing` means no deallocation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frees: Vec<Expr>,
    /// Termination condition (terminates clause)
    /// The condition under which the function is guaranteed to terminate.
    /// Default is `\true` (always terminates). `terminates \false` means non-terminating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminates: Option<Expr>,
    /// Variant expression for termination proofs (decreases clause)
    /// An expression that decreases at each recursive call or loop iteration.
    /// Used to prove termination by well-founded induction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decreases: Option<Expr>,
    /// Named specification cases (ACSL behaviors)
    /// Each behavior has a name and its own assumes/requires/ensures clauses.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub behaviors: Vec<Behavior>,
    /// Whether behaviors are declared complete (cover all cases)
    /// `complete behaviors` in ACSL means the disjunction of all behavior assumptions is true.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub complete_behaviors: bool,
    /// Whether behaviors are declared disjoint (mutually exclusive)
    /// `disjoint behaviors` in ACSL means at most one behavior's assumptions can hold at once.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub disjoint_behaviors: bool,
}

/// Named specification case (ACSL behavior)
/// A behavior groups specifications that apply under certain assumptions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Behavior {
    /// Behavior name (e.g., "success", "error", "null_input")
    pub name: String,
    /// Assumptions under which this behavior applies (assumes clauses)
    /// These are conditions on inputs that determine when this behavior is relevant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assumes: Vec<Expr>,
    /// Preconditions specific to this behavior (requires clauses)
    /// Additional requirements beyond the function's base requires.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<Expr>,
    /// Postconditions when this behavior applies (ensures clauses)
    /// Guarantees that hold when the behavior's assumptions are met.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ensures: Vec<Expr>,
    /// Frame conditions for this behavior (assigns clauses)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assigns: Vec<Expr>,
}

/// Function parameter
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Param {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub ty: Type,
}

/// Invariant (compiles to LEAN or Alloy)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Invariant {
    /// Invariant name
    pub name: String,
    /// Invariant body expression
    pub body: Expr,
}

/// Refinement proof - proves implementation refines abstract specification
///
/// Refinement mapping is a formal technique for proving that a concrete implementation
/// correctly implements an abstract specification. The key components are:
///
/// 1. **Variable mappings**: Define correspondence between spec and impl state
/// 2. **Abstraction function**: Maps concrete states to abstract states
/// 3. **Simulation relation**: Proves each concrete action simulates an abstract action
/// 4. **Invariants**: Properties that must hold at the refinement level
/// 5. **Action mappings**: Explicit correspondence between spec and impl actions
///
/// Example USL:
/// ```text
/// refinement MPSStreamPoolImpl refines MPSStreamPoolSpec {
///     mapping {
///         spec.streams <-> impl.m_streams
///         spec.bindings <-> impl.thread_local_slots
///     }
///
///     invariant { |impl.m_streams| == MAX_STREAMS }
///
///     abstraction {
///         to_abstract(impl) == spec
///     }
///
///     simulation {
///         forall impl: Impl, action: Action .
///             step(to_abstract(impl), action) == to_abstract(step(impl, action))
///     }
///
///     action acquire_stream {
///         spec: AcquireStream
///         impl: MPSStreamPool::acquireStream()
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Refinement {
    /// Refinement name
    pub name: String,
    /// Name of the spec being refined
    pub refines: String,
    /// Variable mappings between spec and implementation
    pub mappings: Vec<VariableMapping>,
    /// Invariants that must hold at the refinement level
    pub invariants: Vec<Expr>,
    /// Abstraction function expression (maps impl state to spec state)
    pub abstraction: Expr,
    /// Simulation relation expression (proves action correspondence)
    pub simulation: Expr,
    /// Action mappings (explicit spec-impl action correspondence)
    pub actions: Vec<ActionMapping>,
}

/// Variable mapping between spec and implementation
///
/// Maps a specification-level variable to its implementation counterpart.
/// This is used to establish state correspondence for refinement proofs.
///
/// Example: `spec.streams <-> impl.m_streams`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VariableMapping {
    /// Specification-side expression (e.g., "spec.streams")
    pub spec_var: Expr,
    /// Implementation-side expression (e.g., "impl.m_streams")
    pub impl_var: Expr,
}

/// Action mapping between spec and implementation
///
/// Maps an abstract specification action to its concrete implementation.
/// This establishes the correspondence needed for simulation proofs.
///
/// Example:
/// ```text
/// action acquire_stream {
///     spec: AcquireStream
///     impl: MPSStreamPool::acquireStream()
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionMapping {
    /// Name of this action mapping
    pub name: String,
    /// Specification-side action name (e.g., "AcquireStream")
    pub spec_action: String,
    /// Implementation-side action path (e.g., ["MPSStreamPool", "acquireStream"])
    pub impl_action: Vec<String>,
    /// Optional guard condition (when this action mapping applies)
    pub guard: Option<Expr>,
}

/// Probabilistic property (compiles to Storm/PRISM)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Probabilistic {
    /// Property name
    pub name: String,
    /// Probability condition expression
    pub condition: Expr,
    /// Comparison operator for bound
    pub comparison: ComparisonOp,
    /// Probability bound (0.0 to 1.0)
    pub bound: f64,
}

/// Security property (compiles to Tamarin/ProVerif)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Security {
    /// Property name
    pub name: String,
    /// Security property body
    pub body: Expr,
}

/// Semantic property (embedding-based or fuzzy matching)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticProperty {
    /// Property name
    pub name: String,
    /// Semantic predicate expression
    pub body: Expr,
}

/// Platform API constraint definition
///
/// Models external API contracts as state machines for verification.
/// These constraints cannot be verified by traditional formal tools because
/// they represent external API behavior (Metal, CUDA, Vulkan, POSIX, etc.).
///
/// Example USL:
/// ```text
/// platform_api Metal {
///     state MTLCommandBuffer {
///         enum Status { Created, Encoding, Committed, Completed }
///
///         transition commit() {
///             requires { status in { Created, Encoding } }
///             ensures { status == Committed }
///         }
///
///         transition addCompletedHandler(block: Block) {
///             requires { status in { Created, Encoding } }  // CRITICAL: must be before commit
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlatformApi {
    /// Platform name (e.g., "Metal", "CUDA", "Vulkan", "POSIX")
    pub name: String,
    /// API object state machines
    pub states: Vec<ApiState>,
}

/// State machine definition for an API object
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiState {
    /// State type name (e.g., "MTLCommandBuffer", "CUstream")
    pub name: String,
    /// State enum definition (possible states this object can be in)
    pub status_enum: Option<StateEnum>,
    /// Allowed transitions (methods that change state)
    pub transitions: Vec<ApiTransition>,
    /// Invariants that must always hold
    pub invariants: Vec<Expr>,
}

/// Enum definition for API object states
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateEnum {
    /// Enum name (e.g., "Status")
    pub name: String,
    /// Possible values
    pub variants: Vec<String>,
}

/// A state transition (method call that may change state)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiTransition {
    /// Method name (e.g., "commit", "addCompletedHandler")
    pub name: String,
    /// Parameters
    pub params: Vec<Param>,
    /// Preconditions (state requirements before the call)
    pub requires: Vec<Expr>,
    /// Postconditions (state guarantees after the call)
    pub ensures: Vec<Expr>,
}

/// Bisimulation property for behavioral equivalence checking
///
/// Specifies that two implementations (oracle and subject) should behave
/// equivalently according to specified criteria.
///
/// Example USL:
/// ```text
/// bisimulation ClaudeCodeParity {
///     oracle: "./claude-code"
///     subject: "./claude-code-rs"
///     equivalent on { api_requests, tool_calls, output }
///     tolerance { timing: 0.1, semantic: 0.95 }
///     forall input: TestInput . traces_equivalent(oracle(input), subject(input))
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bisimulation {
    /// Property name
    pub name: String,
    /// Path to oracle (reference) implementation
    pub oracle: String,
    /// Path to subject (test) implementation
    pub subject: String,
    /// Aspects that must be equivalent (e.g., api_requests, tool_calls, output)
    pub equivalent_on: Vec<String>,
    /// Tolerance values for various aspects
    pub tolerance: Option<BisimulationTolerance>,
    /// Optional property expression (forall input . traces_equivalent(...))
    pub property: Option<BisimulationPropertyExpr>,
}

/// Tolerance configuration for bisimulation checking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct BisimulationTolerance {
    /// Timing tolerance (e.g., 0.1 for 10% variance)
    pub timing: Option<f64>,
    /// Semantic similarity threshold (e.g., 0.95 for 95% similarity)
    pub semantic: Option<f64>,
    /// Additional tolerance fields as key-value pairs
    pub other: Vec<(String, f64)>,
}

/// Property expression for bisimulation (forall input . traces_equivalent(...))
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BisimulationPropertyExpr {
    /// Bound variable name (e.g., "input")
    pub var_name: String,
    /// Type of the bound variable
    pub var_type: Type,
    /// Oracle expression
    pub oracle_expr: Expr,
    /// Subject expression
    pub subject_expr: Expr,
}

/// Version improvement specification for recursive self-improvement
///
/// Specifies that a new version improves upon a previous version while
/// preserving critical properties. Used by Dasher to verify self-modifications.
///
/// Example USL:
/// ```text
/// version DasherV2 improves DasherV1 {
///     capability { V2.verification_speed >= V1.verification_speed }
///     capability { V2.proof_success_rate >= V1.proof_success_rate }
///     preserves { V1.soundness }
///     preserves { V1.termination }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VersionSpec {
    /// Name of the new version (e.g., "DasherV2")
    pub name: String,
    /// Name of the base version being improved (e.g., "DasherV1")
    pub improves: String,
    /// Capability improvement clauses - properties that must be >= the base version
    pub capabilities: Vec<CapabilityClause>,
    /// Preserves clauses - properties that must remain unchanged from base
    pub preserves: Vec<PreservesClause>,
}

/// A capability clause in a version specification
///
/// Specifies that a capability of the new version must be at least as good
/// as the base version.
///
/// Example: `capability { V2.speed >= V1.speed }`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CapabilityClause {
    /// The capability expression (typically a comparison)
    pub expr: Expr,
}

/// A preserves clause in a version specification
///
/// Specifies that a property from the base version must be preserved.
///
/// Example: `preserves { V1.soundness }`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreservesClause {
    /// The property that must be preserved
    pub property: Expr,
}

/// Capability specification - what a system CAN do
///
/// Defines the abilities of a system, used for proving capability
/// preservation across version upgrades.
///
/// Example USL:
/// ```text
/// capability DasherCapability {
///     can verify_rust_code(code: RustCode) -> VerificationResult
///     can verify_usl_spec(spec: UslSpec) -> VerificationResult
///     can improve_self(improvement: Improvement) -> DasherVersion
///
///     requires { soundness_preserved }
///     requires { termination_guaranteed }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CapabilitySpec {
    /// Name of the capability specification
    pub name: String,
    /// List of abilities (things the system can do)
    pub abilities: Vec<CapabilityAbility>,
    /// Required constraints for the capability
    pub requires: Vec<Expr>,
}

/// An ability declaration within a capability specification
///
/// Example: `can verify_rust_code(code: RustCode) -> VerificationResult`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CapabilityAbility {
    /// Name of the ability
    pub name: String,
    /// Parameters for the ability
    pub params: Vec<Param>,
    /// Return type of the ability
    pub return_type: Option<Type>,
}

/// Distributed invariant - property that must hold across multiple agents
///
/// Used for specifying properties in distributed systems where multiple
/// instances (e.g., Dashers) must coordinate and agree.
///
/// Example USL:
/// ```text
/// distributed invariant proof_consensus {
///     forall d1 d2: Dasher, prop: Property .
///         (d1.proves(prop) and d2.proves(prop)) implies (d1.result == d2.result)
/// }
/// ```
///
/// Compiles to multi-process TLA+ specifications where each agent is modeled
/// as a separate process with shared state for coordination.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DistributedInvariant {
    /// Invariant name
    pub name: String,
    /// Invariant body expression (typically contains agent quantifiers)
    pub body: Expr,
}

/// Distributed temporal - temporal property about agent coordination
///
/// Used for verifying eventual consistency, consensus, and coordination
/// properties in distributed systems.
///
/// Example USL:
/// ```text
/// distributed temporal version_convergence {
///     eventually(forall d1 d2: Dasher . d1.version == d2.version)
/// }
/// ```
///
/// Compiles to TLA+ temporal formulas with multi-process semantics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DistributedTemporal {
    /// Property name
    pub name: String,
    /// Temporal formula (typically contains agent quantifiers)
    pub body: TemporalExpr,
    /// Fairness constraints for liveness checking
    pub fairness: Vec<FairnessConstraint>,
}

/// Composed theorem - combines multiple existing properties
///
/// Allows modular proof construction by explicitly declaring dependencies
/// on other theorems, lemmas, or invariants. The body expression can
/// reference the dependency names to use their results.
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
/// - Lean4: theorem with `have` bindings for dependencies
/// - TLA+: THEOREM with ASSUME clauses for dependencies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComposedTheorem {
    /// Theorem name
    pub name: String,
    /// Dependencies on other properties (by name)
    pub uses: Vec<String>,
    /// Theorem body expression (can reference dependencies as variables)
    pub body: Expr,
}

/// Improvement proposal specification (Phase 17.6.1)
///
/// Specifies what a proposed improvement to the system must satisfy before
/// it can be considered for application. Used for recursive self-improvement.
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
/// - Lean4: structure with obligations as propositions
/// - TLA+: operators for proposal validation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImprovementProposal {
    /// Name of the improvement proposal
    pub name: String,
    /// What is being improved (function, module, capability)
    pub target: Expr,
    /// Properties that must be strictly improved
    pub improves: Vec<Expr>,
    /// Properties that must be preserved (at least as good)
    pub preserves: Vec<Expr>,
    /// Preconditions the proposal must satisfy
    pub requires: Vec<Expr>,
}

/// Verification gate specification (Phase 17.6.2)
///
/// Specifies mandatory verification checks that must pass before any
/// improvement can be applied. This represents the immutable, hardcoded
/// verification checkpoint that cannot be bypassed.
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
///
/// Compiles to:
/// - Lean4: inductive type with proof obligations
/// - TLA+: state machine with guard conditions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VerificationGate {
    /// Name of the verification gate
    pub name: String,
    /// Input parameters to the gate
    pub inputs: Vec<Param>,
    /// Named verification checks that must all pass
    pub checks: Vec<GateCheck>,
    /// Action when all checks pass
    pub on_pass: Expr,
    /// Action when any check fails
    pub on_fail: Expr,
}

/// A named verification check within a verification gate
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GateCheck {
    /// Name of the check (for error reporting)
    pub name: String,
    /// The verification condition
    pub condition: Expr,
}

/// Rollback specification (Phase 17.6.3)
///
/// Specifies how to safely rollback a failed improvement attempt.
/// Ensures that the system always returns to a verified state.
///
/// Example USL:
/// ```text
/// rollback_spec SafeRollback {
///     state { current: DasherVersion, history: List<DasherVersion> }
///     invariant { |history| > 0 }
///     trigger { verification_failed or runtime_error }
///     action {
///         current = history.last();
///         ensure { verified(current) }
///     }
///     guarantee { always(verified(current)) }
/// }
/// ```
///
/// Compiles to:
/// - Lean4: structure with invariant proofs
/// - TLA+: action with fairness constraints
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RollbackSpec {
    /// Name of the rollback specification
    pub name: String,
    /// State variables tracked for rollback
    pub state: Vec<Param>,
    /// Invariants that must hold before and after rollback
    pub invariants: Vec<Expr>,
    /// Conditions that trigger rollback
    pub trigger: Expr,
    /// Rollback action (assignments and ensure clause)
    pub action: RollbackAction,
    /// Properties guaranteed after rollback
    pub guarantees: Vec<Expr>,
}

/// The action taken during rollback
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RollbackAction {
    /// State variable assignments (var = expr)
    pub assignments: Vec<(String, Expr)>,
    /// Optional ensure clause that must hold after assignments
    pub ensure: Option<Expr>,
}

/// Comparison operators
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComparisonOp {
    /// Equal (==)
    Eq,
    /// Not equal (!=)
    Ne,
    /// Less than (<)
    Lt,
    /// Less than or equal (<=)
    Le,
    /// Greater than (>)
    Gt,
    /// Greater than or equal (>=)
    Ge,
}

impl std::fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eq => write!(f, "=="),
            Self::Ne => write!(f, "!="),
            Self::Lt => write!(f, "<"),
            Self::Le => write!(f, "<="),
            Self::Gt => write!(f, ">"),
            Self::Ge => write!(f, ">="),
        }
    }
}

/// Binary operators for arithmetic expressions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition (+)
    Add,
    /// Subtraction (-)
    Sub,
    /// Multiplication (*)
    Mul,
    /// Division (/)
    Div,
    /// Modulo (%)
    Mod,
}

/// Expression
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Expr {
    /// Variable reference: `x`, `self`, `self'`
    Var(String),
    /// Integer literal
    Int(i64),
    /// Float literal
    Float(f64),
    /// String literal
    String(String),
    /// Boolean literal
    Bool(bool),
    /// Universal quantifier: `forall x: T . body`
    ForAll {
        /// Bound variable name
        var: String,
        /// Optional type annotation
        ty: Option<Type>,
        /// Quantifier body
        body: Box<Expr>,
    },
    /// Existential quantifier: `exists x: T . body`
    Exists {
        /// Bound variable name
        var: String,
        /// Optional type annotation
        ty: Option<Type>,
        /// Quantifier body
        body: Box<Expr>,
    },
    /// In-set binding: `forall x in set . body`
    ForAllIn {
        /// Bound variable name
        var: String,
        /// Collection to iterate over
        collection: Box<Expr>,
        /// Quantifier body
        body: Box<Expr>,
    },
    /// In-set existential: `exists x in set . body`
    ExistsIn {
        /// Bound variable name
        var: String,
        /// Collection to iterate over
        collection: Box<Expr>,
        /// Quantifier body
        body: Box<Expr>,
    },
    /// Implication: `a implies b`
    Implies(Box<Expr>, Box<Expr>),
    /// Conjunction: `a and b`
    And(Box<Expr>, Box<Expr>),
    /// Disjunction: `a or b`
    Or(Box<Expr>, Box<Expr>),
    /// Negation: `not a`
    Not(Box<Expr>),
    /// Comparison: `a == b`, `a < b`, etc.
    Compare(Box<Expr>, ComparisonOp, Box<Expr>),
    /// Binary arithmetic: `a + b`, `a * b`, etc.
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    /// Unary minus: `-a`
    Neg(Box<Expr>),
    /// Function application: `f(a, b, ...)`
    App(String, Vec<Expr>),
    /// Method call: `obj.method(args)`
    MethodCall {
        /// Object receiving the method call
        receiver: Box<Expr>,
        /// Method name
        method: String,
        /// Method arguments
        args: Vec<Expr>,
    },
    /// Field access: `obj.field`
    FieldAccess(Box<Expr>, String),
}

impl Expr {
    /// Check if this expression contains any quantifiers (forall, exists).
    #[must_use]
    pub fn contains_quantifier(&self) -> bool {
        match self {
            Self::ForAll { .. }
            | Self::Exists { .. }
            | Self::ForAllIn { .. }
            | Self::ExistsIn { .. } => true,
            Self::Implies(l, r) | Self::And(l, r) | Self::Or(l, r) => {
                l.contains_quantifier() || r.contains_quantifier()
            }
            Self::Compare(l, _, r) | Self::Binary(l, _, r) => {
                l.contains_quantifier() || r.contains_quantifier()
            }
            Self::Not(e) | Self::Neg(e) => e.contains_quantifier(),
            Self::App(_, args) => args.iter().any(Self::contains_quantifier),
            Self::MethodCall { receiver, args, .. } => {
                receiver.contains_quantifier() || args.iter().any(Self::contains_quantifier)
            }
            Self::FieldAccess(obj, _) => obj.contains_quantifier(),
            Self::Var(_) | Self::Int(_) | Self::Float(_) | Self::String(_) | Self::Bool(_) => false,
        }
    }
}

/// Temporal expression
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TemporalExpr {
    /// Always/globally: `always(P)`
    Always(Box<TemporalExpr>),
    /// Eventually/finally: `eventually(P)`
    Eventually(Box<TemporalExpr>),
    /// Leads-to: `P ~> Q`
    LeadsTo(Box<TemporalExpr>, Box<TemporalExpr>),
    /// Atomic expression (lift from Expr)
    Atom(Expr),
}

/// Fairness kind for liveness properties
///
/// TLC supports two types of fairness:
/// - **Weak Fairness (WF)**: If an action is continuously enabled, it must eventually be taken.
///   "If you can always do it, you eventually must."
/// - **Strong Fairness (SF)**: If an action is repeatedly enabled, it must eventually be taken.
///   "If you can repeatedly do it, you eventually must."
///
/// Strong fairness is stricter - it handles cases where an action may be temporarily disabled.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FairnessKind {
    /// Weak fairness: `WF_vars(Action)` - if continuously enabled, must occur
    Weak,
    /// Strong fairness: `SF_vars(Action)` - if infinitely often enabled, must occur
    Strong,
}

impl std::fmt::Display for FairnessKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Weak => write!(f, "weak"),
            Self::Strong => write!(f, "strong"),
        }
    }
}

/// Fairness constraint specification
///
/// Fairness constraints are essential for verifying liveness properties in TLA+.
/// Without them, TLC will find "stuttering" counterexamples where enabled actions
/// are never taken.
///
/// Example USL:
/// ```text
/// temporal eventually_served {
///     fair weak Next    // Adds WF_vars(Next) to the spec
///     forall req in requests .
///         eventually(served(req))
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FairnessConstraint {
    /// The kind of fairness (weak or strong)
    pub kind: FairnessKind,
    /// The action name that must be fair (e.g., "Next", "Acquire", "Release")
    pub action: String,
    /// Optional state variables to scope the fairness (for `WF_vars(Action)`)
    /// If None, defaults to all state variables
    pub vars: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_op_display() {
        // Tests ComparisonOp Display impl - catches mutation that returns Ok(Default::default())
        assert_eq!(format!("{}", ComparisonOp::Eq), "==");
        assert_eq!(format!("{}", ComparisonOp::Ne), "!=");
        assert_eq!(format!("{}", ComparisonOp::Lt), "<");
        assert_eq!(format!("{}", ComparisonOp::Le), "<=");
        assert_eq!(format!("{}", ComparisonOp::Gt), ">");
        assert_eq!(format!("{}", ComparisonOp::Ge), ">=");
    }

    #[test]
    fn test_fairness_kind_display() {
        // Tests FairnessKind Display impl - catches mutation that returns Ok(Default::default())
        assert_eq!(format!("{}", FairnessKind::Weak), "weak");
        assert_eq!(format!("{}", FairnessKind::Strong), "strong");
    }

    #[test]
    fn test_contains_quantifier_simple() {
        // Simple expressions without quantifiers
        assert!(!Expr::Bool(true).contains_quantifier());
        assert!(!Expr::Int(42).contains_quantifier());
        assert!(!Expr::Var("x".to_string()).contains_quantifier());
    }

    #[test]
    fn test_contains_quantifier_forall() {
        let forall = Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("Bool".to_string())),
            body: Box::new(Expr::Var("x".to_string())),
        };
        assert!(forall.contains_quantifier());
    }

    #[test]
    fn test_contains_quantifier_exists() {
        let exists = Expr::Exists {
            var: "x".to_string(),
            ty: Some(Type::Named("Int".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert!(exists.contains_quantifier());
    }

    #[test]
    fn test_contains_quantifier_nested() {
        // forall inside implies
        let inner = Expr::ForAll {
            var: "x".to_string(),
            ty: None,
            body: Box::new(Expr::Bool(true)),
        };
        let outer = Expr::Implies(Box::new(Expr::Bool(false)), Box::new(inner));
        assert!(outer.contains_quantifier());
    }

    #[test]
    fn test_contains_quantifier_no_quantifier_nested() {
        // No quantifiers even in nested expression
        let nested = Expr::And(
            Box::new(Expr::Bool(true)),
            Box::new(Expr::Or(
                Box::new(Expr::Bool(false)),
                Box::new(Expr::Var("x".to_string())),
            )),
        );
        assert!(!nested.contains_quantifier());
    }

    #[test]
    fn test_contains_quantifier_in_app() {
        let quantified_arg = Expr::ForAll {
            var: "x".to_string(),
            ty: None,
            body: Box::new(Expr::Bool(true)),
        };
        let app = Expr::App("f".to_string(), vec![quantified_arg]);
        assert!(app.contains_quantifier());
    }

    #[test]
    fn test_contains_quantifier_in_method_call() {
        let quantified_receiver = Expr::Exists {
            var: "x".to_string(),
            ty: None,
            body: Box::new(Expr::Bool(true)),
        };
        let call = Expr::MethodCall {
            receiver: Box::new(quantified_receiver),
            method: "test".to_string(),
            args: vec![],
        };
        assert!(call.contains_quantifier());
    }

    #[test]
    fn test_contains_quantifier_in_compare_rhs_only() {
        // Tests || vs && mutation in Compare branch (line 598)
        // Quantifier only on RHS should still return true
        let quantified = Expr::ForAll {
            var: "x".to_string(),
            ty: None,
            body: Box::new(Expr::Bool(true)),
        };
        let compare = Expr::Compare(
            Box::new(Expr::Int(0)),
            ComparisonOp::Lt,
            Box::new(quantified),
        );
        assert!(compare.contains_quantifier());
    }

    #[test]
    fn test_contains_quantifier_in_binary_rhs_only() {
        // Tests || vs && mutation in Binary branch (same line pattern)
        // Quantifier only on RHS should still return true
        let quantified = Expr::Exists {
            var: "y".to_string(),
            ty: None,
            body: Box::new(Expr::Int(42)),
        };
        let binary = Expr::Binary(Box::new(Expr::Int(1)), BinaryOp::Add, Box::new(quantified));
        assert!(binary.contains_quantifier());
    }

    // Property kind tests
    #[test]
    fn test_property_kind_theorem() {
        let prop = Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });
        assert_eq!(prop.property_kind(), "theorem");
    }

    #[test]
    fn test_property_kind_temporal() {
        let prop = Property::Temporal(Temporal {
            name: "test".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
            fairness: vec![],
        });
        assert_eq!(prop.property_kind(), "temporal");
    }

    #[test]
    fn test_property_kind_contract() {
        let prop = Property::Contract(Contract {
            type_path: vec!["Type".to_string(), "method".to_string()],
            params: vec![],
            return_type: None,
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        });
        assert_eq!(prop.property_kind(), "contract");
    }

    #[test]
    fn test_property_kind_invariant() {
        let prop = Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });
        assert_eq!(prop.property_kind(), "invariant");
    }

    #[test]
    fn test_property_kind_refinement() {
        let prop = Property::Refinement(Refinement {
            name: "test".to_string(),
            refines: "spec".to_string(),
            mappings: vec![],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        });
        assert_eq!(prop.property_kind(), "refinement");
    }

    #[test]
    fn test_property_kind_security() {
        let prop = Property::Security(Security {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });
        assert_eq!(prop.property_kind(), "security");
    }

    #[test]
    fn test_property_kind_platform_api() {
        let prop = Property::PlatformApi(PlatformApi {
            name: "test".to_string(),
            states: vec![],
        });
        assert_eq!(prop.property_kind(), "platform_api");
    }

    #[test]
    fn test_property_kind_bisimulation_is_refinement() {
        // Bisimulation is treated as refinement-like
        let prop = Property::Bisimulation(Bisimulation {
            name: "test".to_string(),
            oracle: "oracle".to_string(),
            subject: "subject".to_string(),
            equivalent_on: vec![],
            tolerance: None,
            property: None,
        });
        assert_eq!(prop.property_kind(), "refinement");
    }

    #[test]
    fn test_spec_property_kinds_empty() {
        let spec = Spec::default();
        assert!(spec.property_kinds().is_empty());
    }

    #[test]
    fn test_spec_property_kinds_single() {
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Theorem(Theorem {
                name: "test".to_string(),
                body: Expr::Bool(true),
            })],
        };
        let kinds = spec.property_kinds();
        assert_eq!(kinds.len(), 1);
        assert_eq!(kinds[0], "theorem");
    }

    #[test]
    fn test_spec_property_kinds_multiple_unique() {
        let spec = Spec {
            types: vec![],
            properties: vec![
                Property::Theorem(Theorem {
                    name: "thm1".to_string(),
                    body: Expr::Bool(true),
                }),
                Property::Temporal(Temporal {
                    name: "temp1".to_string(),
                    body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
                    fairness: vec![],
                }),
                Property::Invariant(Invariant {
                    name: "inv".to_string(),
                    body: Expr::Bool(true),
                }),
            ],
        };
        let kinds = spec.property_kinds();
        assert_eq!(kinds.len(), 3);
        // Should be sorted
        assert!(kinds.contains(&"invariant"));
        assert!(kinds.contains(&"temporal"));
        assert!(kinds.contains(&"theorem"));
    }

    #[test]
    fn test_spec_property_kinds_deduplicates() {
        let spec = Spec {
            types: vec![],
            properties: vec![
                Property::Theorem(Theorem {
                    name: "thm1".to_string(),
                    body: Expr::Bool(true),
                }),
                Property::Theorem(Theorem {
                    name: "thm2".to_string(),
                    body: Expr::Bool(false),
                }),
                Property::Theorem(Theorem {
                    name: "thm3".to_string(),
                    body: Expr::Int(42),
                }),
            ],
        };
        let kinds = spec.property_kinds();
        assert_eq!(kinds.len(), 1);
        assert_eq!(kinds[0], "theorem");
    }

    #[test]
    fn test_spec_property_kind_map() {
        let spec = Spec {
            types: vec![],
            properties: vec![
                Property::Theorem(Theorem {
                    name: "my_theorem".to_string(),
                    body: Expr::Bool(true),
                }),
                Property::Invariant(Invariant {
                    name: "my_invariant".to_string(),
                    body: Expr::Bool(true),
                }),
            ],
        };
        let map = spec.property_kind_map();
        assert_eq!(map.get("my_theorem"), Some(&"theorem"));
        assert_eq!(map.get("my_invariant"), Some(&"invariant"));
    }
}
