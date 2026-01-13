//! Phase 18: AI-driven proof search agent
//!
//! This module implements an iterative proof search agent that:
//! - Combines LLM tactic prediction with structural suggestions
//! - Updates an internal policy using reward signals from validation
//! - Propagates hints and failures across backends for reuse
//!
//! The agent is backend-agnostic: it focuses on generating proof attempts and
//! producing actionable hints, leaving backend execution to callers.

use crate::suggest::suggest_tactics;
use crate::tactic_llm::{LlmTacticPredictor, TacticPredictionRequest};
use crate::{
    validation::validate_synthesized_proof, Confidence, ProofSynthesizer, SuggestionSource,
    SynthesisError, SynthesisRequest, SynthesisResult, TacticSuggestion,
};
use dashprove_backends::traits::{BackendId, BackendResult, VerificationStatus};
use dashprove_learning::ProofLearningSystem;
use dashprove_usl::ast::Property;
use std::collections::HashMap;

/// Induction mode selection for decomposition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InductionMode {
    /// Simple induction: base case + P(n) => P(n+1)
    #[default]
    Simple,
    /// Strong induction: (∀k<n. P(k)) => P(n)
    Strong,
    /// Well-founded: uses decreasing measure function
    WellFounded,
}

/// Configuration for the proof search agent
#[derive(Debug, Clone)]
pub struct ProofSearchConfig {
    /// Maximum search iterations
    pub max_iterations: u32,
    /// Max LLM synthesis attempts per iteration
    pub max_attempts_per_iteration: u32,
    /// Minimum validation score considered successful
    pub validation_threshold: f64,
    /// Reward applied when validation passes threshold
    pub success_reward: f64,
    /// Penalty applied when validation fails
    pub failure_penalty: f64,
    /// Fraction of tactics to explore even if low-weight
    pub exploration_rate: f64,
    /// Maximum number of tactic-derived hints to pass to synthesis
    pub max_hints: usize,
    /// Enable hierarchical decomposition for complex proofs
    pub enable_decomposition: bool,
    /// Maximum depth for hierarchical decomposition
    pub max_decomposition_depth: u32,
    /// Threshold for considering a property "complex" enough to decompose
    pub decomposition_complexity_threshold: f64,
    /// Induction mode to use for numeric/recursive types
    pub induction_mode: InductionMode,
}

impl Default for ProofSearchConfig {
    fn default() -> Self {
        Self {
            max_iterations: 4,
            max_attempts_per_iteration: 2,
            validation_threshold: 0.75,
            success_reward: 1.0,
            failure_penalty: 0.6,
            exploration_rate: 0.2,
            max_hints: 5,
            enable_decomposition: true,
            max_decomposition_depth: 3,
            decomposition_complexity_threshold: 0.7,
            induction_mode: InductionMode::default(),
        }
    }
}

/// Represents a decomposed sub-goal in hierarchical proof search
#[derive(Debug, Clone)]
pub struct SubGoal {
    /// Identifier for the sub-goal
    pub id: String,
    /// The sub-property to prove
    pub property: Property,
    /// Parent sub-goal id (None for root)
    pub parent: Option<String>,
    /// Depth in the decomposition tree
    pub depth: u32,
    /// Whether this sub-goal has been proven
    pub proven: bool,
    /// Dependencies on other sub-goals
    pub dependencies: Vec<String>,
}

/// Result of hierarchical decomposition
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// Sub-goals extracted from the property
    pub sub_goals: Vec<SubGoal>,
    /// Decomposition strategy used
    pub strategy: DecompositionStrategy,
    /// Estimated complexity reduction
    pub complexity_reduction: f64,
}

/// Strategy used for decomposition
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompositionStrategy {
    /// Split conjunction into separate goals
    ConjunctionSplit,
    /// Split implications into antecedent + consequent
    ImplicationChain,
    /// Use simple structural induction (base + step)
    Induction,
    /// Use strong induction (∀k<n. P(k)) => P(n))
    StrongInduction,
    /// Use well-founded induction with decreasing measure
    WellFoundedInduction,
    /// Split by cases
    CaseSplit,
    /// Disjunction elimination (prove goal assuming each disjunct)
    DisjunctionElimination,
    /// No decomposition needed
    None,
}

/// Request for proof search
#[derive(Debug, Clone)]
pub struct ProofSearchRequest<'a> {
    /// Property to prove
    pub property: &'a Property,
    /// Target backend
    pub backend: BackendId,
    /// Optional proof context (hypotheses, local lemmas)
    pub context: Option<String>,
    /// Backends that should receive propagated hints
    pub propagate_to: Vec<BackendId>,
    /// Additional user hints to seed search
    pub additional_hints: Vec<String>,
    /// Preferred tactics to prioritize first
    pub preferred_tactics: Vec<String>,
    /// Feedback from previous backend runs (for cross-backend hints)
    pub feedback: Vec<BackendResult>,
    /// Learning system for structural suggestions (read-only)
    pub learning: Option<&'a ProofLearningSystem>,
}

impl<'a> ProofSearchRequest<'a> {
    /// Create a request with sane defaults
    pub fn new(property: &'a Property, backend: BackendId) -> Self {
        Self {
            property,
            backend,
            context: None,
            propagate_to: Vec::new(),
            additional_hints: Vec::new(),
            preferred_tactics: Vec::new(),
            feedback: Vec::new(),
            learning: None,
        }
    }
}

/// Result of a proof search iteration
#[derive(Debug, Clone)]
pub struct ProofSearchStep {
    /// Iteration number (0-based)
    pub iteration: u32,
    /// Tactics prioritized this iteration
    pub tactics: Vec<String>,
    /// Hints supplied to the synthesizer
    pub hints: Vec<String>,
    /// Synthesized proof attempt
    pub synthesis: SynthesisResult,
    /// Validation outcome
    pub validation: crate::validation::ProofValidation,
    /// Reward assigned to the tactic policy
    pub reward: f64,
}

/// Hint for transferring knowledge across backends
#[derive(Debug, Clone, PartialEq)]
pub struct CrossBackendHint {
    /// Backend that produced the hint
    pub source_backend: BackendId,
    /// Backend that should consume the hint
    pub target_backend: BackendId,
    /// Natural-language or tactic hint
    pub hint: String,
    /// Confidence (0.0 - 1.0)
    pub confidence: f64,
}

/// Snapshot of the tactic policy after search
#[derive(Debug, Clone, PartialEq)]
pub struct TacticPolicySnapshot {
    /// Final weight assigned to each tactic
    pub weights: Vec<(String, f64)>,
}

/// Overall proof search result
#[derive(Debug, Clone)]
pub struct ProofSearchResult {
    /// Best proof attempt discovered (if any)
    pub best_proof: Option<SynthesisResult>,
    /// All iterations executed
    pub steps: Vec<ProofSearchStep>,
    /// Generated cross-backend hints
    pub propagated_hints: Vec<CrossBackendHint>,
    /// Final tactic policy weights
    pub policy: TacticPolicySnapshot,
}

/// Result of a single sub-goal proof attempt
#[derive(Debug, Clone)]
pub struct SubGoalResult {
    /// Identifier of the sub-goal
    pub sub_goal_id: String,
    /// The proof search result for this sub-goal
    pub result: ProofSearchResult,
    /// Whether this sub-goal was successfully proven
    pub proven: bool,
    /// Nested sub-goal results (if this sub-goal was further decomposed)
    pub nested_results: Vec<SubGoalResult>,
}

/// Result of hierarchical proof search with decomposition
#[derive(Debug, Clone)]
pub struct HierarchicalSearchResult {
    /// The main proof search result (for the original property)
    pub main_result: ProofSearchResult,
    /// Results for each sub-goal (if decomposition was used)
    pub sub_goal_results: Vec<SubGoalResult>,
    /// The decomposition strategy used
    pub decomposition_strategy: DecompositionStrategy,
    /// Total number of search iterations across all goals
    pub total_iterations: u32,
}

impl HierarchicalSearchResult {
    /// Check if all sub-goals and the main goal are proven
    pub fn all_proven(&self) -> bool {
        let main_proven = self
            .main_result
            .best_proof
            .as_ref()
            .map(|p| p.confidence >= 0.5)
            .unwrap_or(false);

        let sub_goals_proven = self.sub_goal_results.iter().all(|r| r.all_proven());

        main_proven && sub_goals_proven
    }

    /// Get the total number of proven sub-goals (including nested)
    pub fn proven_count(&self) -> usize {
        self.sub_goal_results
            .iter()
            .map(|r| r.proven_count())
            .sum::<usize>()
    }

    /// Get all hints from proven sub-goals
    pub fn collect_proven_hints(&self) -> Vec<String> {
        self.sub_goal_results
            .iter()
            .flat_map(|r| r.collect_proven_hints())
            .collect()
    }
}

impl SubGoalResult {
    /// Check if this sub-goal and all nested sub-goals are proven
    pub fn all_proven(&self) -> bool {
        self.proven && self.nested_results.iter().all(|r| r.all_proven())
    }

    /// Get count of proven goals (including nested)
    pub fn proven_count(&self) -> usize {
        let self_count = if self.proven { 1 } else { 0 };
        let nested_count: usize = self.nested_results.iter().map(|r| r.proven_count()).sum();
        self_count + nested_count
    }

    /// Collect hints from this sub-goal and all nested proven sub-goals
    pub fn collect_proven_hints(&self) -> Vec<String> {
        let mut hints = Vec::new();

        if self.proven {
            if let Some(proof) = &self.result.best_proof {
                hints.push(format!(
                    "Sub-goal '{}' proven with tactics: {}",
                    self.sub_goal_id,
                    proof.tactics_used.join(", ")
                ));
            }
        }

        for nested in &self.nested_results {
            hints.extend(nested.collect_proven_hints());
        }

        hints
    }
}

/// Proof search agent orchestrating LLM synthesis, tactic prediction, and reward updates
pub struct ProofSearchAgent {
    synthesizer: ProofSynthesizer,
    tactic_predictor: Option<LlmTacticPredictor>,
    policy: TacticPolicy,
    config: ProofSearchConfig,
    /// Template registry for structure-to-instance matching (Phase 18.6 SITA)
    template_registry: Option<TemplateRegistry>,
}

impl ProofSearchAgent {
    /// Create a new agent with a synthesizer
    pub fn new(synthesizer: ProofSynthesizer) -> Self {
        Self {
            synthesizer,
            tactic_predictor: None,
            policy: TacticPolicy::default(),
            config: ProofSearchConfig::default(),
            template_registry: None,
        }
    }

    /// Attach an LLM tactic predictor
    pub fn with_tactic_predictor(mut self, predictor: LlmTacticPredictor) -> Self {
        self.tactic_predictor = Some(predictor);
        self
    }

    /// Override the default configuration
    pub fn with_config(mut self, config: ProofSearchConfig) -> Self {
        self.policy.exploration_rate = config.exploration_rate;
        self.config = config;
        self
    }

    /// Attach a template registry for structure-to-instance matching (SITA)
    ///
    /// When enabled, the agent will try to match the property against known
    /// proof templates and use template-derived hints and tactics.
    pub fn with_template_registry(mut self, registry: TemplateRegistry) -> Self {
        self.template_registry = Some(registry);
        self
    }

    /// Enable built-in templates for common proof patterns
    pub fn with_builtin_templates(mut self) -> Self {
        self.template_registry = Some(TemplateRegistry::with_builtins());
        self
    }

    /// Get template-derived hints for a property (SITA integration)
    fn template_hints(&self, property: &Property) -> Vec<String> {
        let Some(registry) = &self.template_registry else {
            return Vec::new();
        };

        let matches = registry.find_matches(property);
        let mut hints = Vec::new();

        // Use top 3 template matches
        for m in matches.iter().take(3) {
            if let Some(template) = registry.get(&m.template_id) {
                // Add template name as a hint
                hints.push(format!(
                    "Property matches '{}' template pattern (confidence: {:.2})",
                    template.name, m.confidence
                ));

                // Add template-specific hints
                hints.extend(template.hints.iter().cloned());

                // Add recommended tactics
                for tactic in &template.tactics {
                    hints.push(format!("Template suggests tactic: {}", tactic));
                }
            }
        }

        hints
    }

    /// Run iterative proof search
    pub async fn search(
        &mut self,
        request: ProofSearchRequest<'_>,
    ) -> Result<ProofSearchResult, SynthesisError> {
        let mut steps = Vec::new();
        let mut best: Option<(SynthesisResult, crate::validation::ProofValidation)> = None;
        let feedback_hints = self.feedback_hints(&request);
        let template_hints = self.template_hints(request.property);

        for iteration in 0..self.config.max_iterations {
            // Collect tactic candidates
            let tactic_candidates = self
                .collect_tactics(&request, request.learning)
                .await
                .unwrap_or_default();
            let ranked = self
                .policy
                .rank(&tactic_candidates, &request.preferred_tactics);

            // Build hints: template hints + feedback + tactic names + user-provided hints
            let mut hints = Vec::new();
            hints.extend(template_hints.clone()); // SITA template-derived hints first
            hints.extend(request.additional_hints.clone());
            hints.extend(feedback_hints.iter().map(|h| h.hint.clone()));
            for tactic in ranked.iter().take(self.config.max_hints) {
                hints.push(format!("try tactic `{}`", tactic.tactic));
            }

            // Issue synthesis request
            let mut synth_request =
                SynthesisRequest::new(request.property, request.backend).with_hints(hints.clone());
            synth_request.max_attempts = self.config.max_attempts_per_iteration;

            let synthesis = self.synthesizer.synthesize(&synth_request).await?;
            let validation = validate_synthesized_proof(&synthesis, request.backend);

            let reward =
                if validation.score >= self.config.validation_threshold && validation.is_valid {
                    self.config.success_reward * validation.score
                } else {
                    -self.config.failure_penalty * (1.0 - validation.score)
                };

            let tactic_names: Vec<String> = ranked.iter().map(|t| t.tactic.clone()).collect();
            self.policy.apply_reward(&tactic_names, reward);

            if best
                .as_ref()
                .map(|(_, v)| validation.score > v.score)
                .unwrap_or(true)
            {
                best = Some((synthesis.clone(), validation.clone()));
            }

            steps.push(ProofSearchStep {
                iteration,
                tactics: tactic_names,
                hints,
                synthesis,
                validation,
                reward,
            });

            if let Some((_, best_validation)) = best.as_ref() {
                if best_validation.is_valid
                    && best_validation.score >= self.config.validation_threshold
                {
                    break;
                }
            }
        }

        let propagated_hints = self.derive_propagated_hints(
            &request,
            &feedback_hints,
            best.as_ref().map(|(proof, _)| proof),
        );

        Ok(ProofSearchResult {
            best_proof: best.map(|(proof, _)| proof),
            steps,
            propagated_hints,
            policy: self.policy.snapshot(),
        })
    }

    /// Gather tactic suggestions from LLM + structure + learning
    async fn collect_tactics(
        &self,
        request: &ProofSearchRequest<'_>,
        learning: Option<&ProofLearningSystem>,
    ) -> Result<Vec<TacticSuggestion>, SynthesisError> {
        let mut suggestions = Vec::new();

        // LLM tactic prediction
        if let Some(predictor) = &self.tactic_predictor {
            let tactic_request = TacticPredictionRequest {
                property: request.property,
                backend: request.backend,
                context: request.context.clone(),
                previous_tactics: request.preferred_tactics.clone(),
                num_suggestions: self.config.max_hints.max(3),
            };

            if let Ok(result) = predictor.predict_tactics(&tactic_request).await {
                suggestions.extend(result.suggestions);
            }
        }

        // Structural suggestions
        suggestions.extend(suggest_tactics(
            request.property,
            &request.backend,
            learning,
        ));

        // Manual preferences come first
        for tactic in &request.preferred_tactics {
            if suggestions.iter().any(|s| &s.tactic == tactic) {
                continue;
            }
            suggestions.insert(
                0,
                TacticSuggestion {
                    tactic: tactic.clone(),
                    confidence: Confidence::High,
                    source: SuggestionSource::Heuristic,
                    rationale: "User-preferred tactic".to_string(),
                },
            );
        }

        // Deduplicate by tactic name (keep highest confidence)
        let mut best = HashMap::new();
        for suggestion in suggestions {
            best.entry(suggestion.tactic.clone())
                .and_modify(|existing: &mut TacticSuggestion| {
                    if suggestion.confidence.to_score() > existing.confidence.to_score() {
                        *existing = suggestion.clone();
                    }
                })
                .or_insert(suggestion);
        }

        Ok(best.into_values().collect())
    }

    /// Convert backend feedback into reusable hints
    fn feedback_hints(&self, request: &ProofSearchRequest<'_>) -> Vec<CrossBackendHint> {
        let mut hints = Vec::new();
        for fb in &request.feedback {
            let confidence = match fb.status {
                VerificationStatus::Proven => 0.9,
                VerificationStatus::Partial { .. } => 0.55,
                VerificationStatus::Unknown { .. } => 0.35,
                VerificationStatus::Disproven => 0.65,
            };

            let hint_text = self.describe_feedback(fb);
            if hint_text.is_empty() {
                continue;
            }

            for target in &request.propagate_to {
                if *target == fb.backend {
                    continue;
                }
                hints.push(CrossBackendHint {
                    source_backend: fb.backend,
                    target_backend: *target,
                    hint: hint_text.clone(),
                    confidence,
                });
            }
        }

        hints
    }

    fn describe_feedback(&self, feedback: &BackendResult) -> String {
        match feedback.status.clone() {
            VerificationStatus::Proven => {
                if let Some(proof) = &feedback.proof {
                    format!(
                        "{:?} succeeded; reuse proof outline: {}",
                        feedback.backend,
                        proof.lines().next().unwrap_or_default()
                    )
                } else {
                    format!(
                        "{:?} proved the property; reuse tactic ordering",
                        feedback.backend
                    )
                }
            }
            VerificationStatus::Disproven => {
                if let Some(cx) = &feedback.counterexample {
                    format!(
                        "{:?} found counterexample: {}",
                        feedback.backend,
                        cx.summary()
                    )
                } else if let Some(diag) = feedback.diagnostics.first() {
                    format!("{:?} failed: {}", feedback.backend, diag)
                } else {
                    format!("{:?} produced a failing run", feedback.backend)
                }
            }
            VerificationStatus::Unknown { reason } => format!(
                "{:?} inconclusive: {reason}. Try simplifying obligations.",
                feedback.backend
            ),
            VerificationStatus::Partial {
                verified_percentage,
            } => format!(
                "{:?} partially verified ({verified_percentage:.1}%). Focus remaining cases.",
                feedback.backend
            ),
        }
    }

    /// Run hierarchical proof search with decomposition
    ///
    /// This method recursively decomposes complex properties into sub-goals and
    /// proves each independently. Sub-goal proofs are composed back into a
    /// complete proof for the original property.
    ///
    /// # Arguments
    /// * `request` - The proof search request
    ///
    /// # Returns
    /// A `HierarchicalSearchResult` containing the main proof and all sub-goal proofs
    pub fn search_with_decomposition<'a>(
        &'a mut self,
        request: ProofSearchRequest<'a>,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<HierarchicalSearchResult, SynthesisError>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(self.search_with_decomposition_internal(request, 0))
    }

    /// Internal recursive implementation of hierarchical search
    fn search_with_decomposition_internal<'a>(
        &'a mut self,
        request: ProofSearchRequest<'a>,
        depth: u32,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<HierarchicalSearchResult, SynthesisError>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            // Check if decomposition is enabled and we haven't exceeded depth
            if !self.config.enable_decomposition || depth >= self.config.max_decomposition_depth {
                // Just run normal search
                let result = self.search(request).await?;
                return Ok(HierarchicalSearchResult {
                    main_result: result,
                    sub_goal_results: Vec::new(),
                    decomposition_strategy: DecompositionStrategy::None,
                    total_iterations: 0,
                });
            }

            // Attempt decomposition
            let decomposition =
                decompose_property_with_mode(request.property, depth, self.config.induction_mode);

            // If no decomposition possible or not complex enough, run normal search
            if decomposition.sub_goals.is_empty()
                || decomposition.complexity_reduction
                    < self.config.decomposition_complexity_threshold
            {
                let result = self.search(request).await?;
                return Ok(HierarchicalSearchResult {
                    main_result: result,
                    sub_goal_results: Vec::new(),
                    decomposition_strategy: DecompositionStrategy::None,
                    total_iterations: 0,
                });
            }

            // Process sub-goals
            let mut sub_goal_results = Vec::new();
            let mut all_hints = request.additional_hints.clone();
            let mut total_iterations = 0;

            // Sort sub-goals by dependencies (process independent ones first)
            let mut sorted_goals: Vec<_> = decomposition.sub_goals.iter().collect();
            sorted_goals.sort_by(|a, b| a.dependencies.len().cmp(&b.dependencies.len()));

            for sub_goal in sorted_goals {
                // Check if dependencies are satisfied
                let deps_satisfied = sub_goal.dependencies.iter().all(|dep_id| {
                    sub_goal_results
                        .iter()
                        .any(|r: &SubGoalResult| r.sub_goal_id == *dep_id && r.proven)
                });

                if !deps_satisfied {
                    // Skip for now - would need more sophisticated dependency resolution
                    continue;
                }

                // Build hints from previously proven sub-goals
                let sub_goal_hints: Vec<String> = sub_goal_results
                    .iter()
                    .filter(|r: &&SubGoalResult| r.proven)
                    .map(|r| {
                        format!(
                            "Sub-goal '{}' proven using tactics: {:?}",
                            r.sub_goal_id,
                            r.result
                                .best_proof
                                .as_ref()
                                .map(|p| p.tactics_used.clone())
                                .unwrap_or_default()
                        )
                    })
                    .collect();

                all_hints.extend(sub_goal_hints);

                // Add decomposition-specific hint
                let strategy_hint = match decomposition.strategy {
                    DecompositionStrategy::ConjunctionSplit => {
                        format!("Proving conjunction component: {}", sub_goal.id)
                    }
                    DecompositionStrategy::DisjunctionElimination => {
                        if sub_goal.id.contains("left") {
                            "Case analysis: assuming left disjunct holds".to_string()
                        } else {
                            "Case analysis: assuming right disjunct holds".to_string()
                        }
                    }
                    DecompositionStrategy::ImplicationChain => {
                        if sub_goal.id.contains("antecedent") {
                            "Establishing antecedent for implication chain".to_string()
                        } else {
                            "Proving consequent assuming antecedent holds".to_string()
                        }
                    }
                    DecompositionStrategy::CaseSplit => {
                        if sub_goal.id.contains("true") {
                            "Case: boolean variable is true".to_string()
                        } else {
                            "Case: boolean variable is false".to_string()
                        }
                    }
                    DecompositionStrategy::Induction => {
                        if sub_goal.id.contains("base") || sub_goal.id.contains("nil") {
                            "Base case of induction".to_string()
                        } else {
                            "Inductive step: assume P(n), prove P(n+1)".to_string()
                        }
                    }
                    DecompositionStrategy::StrongInduction => {
                        "Strong induction: assume P(k) for all k < n, prove P(n)".to_string()
                    }
                    DecompositionStrategy::WellFoundedInduction => {
                        if sub_goal.id.contains("order") || sub_goal.id.contains("measure") {
                            "Well-founded induction: prove ordering is well-founded".to_string()
                        } else {
                            "Well-founded step: assume P(y) for all y < x, prove P(x)".to_string()
                        }
                    }
                    DecompositionStrategy::None => String::new(),
                };

                if !strategy_hint.is_empty() {
                    all_hints.push(strategy_hint);
                }

                // Create sub-goal request
                let sub_request = ProofSearchRequest {
                    property: &sub_goal.property,
                    backend: request.backend,
                    context: request.context.clone(),
                    propagate_to: request.propagate_to.clone(),
                    additional_hints: all_hints.clone(),
                    preferred_tactics: request.preferred_tactics.clone(),
                    feedback: request.feedback.clone(),
                    learning: request.learning,
                };

                // Recursively search for sub-goal proof
                let sub_result = self
                    .search_with_decomposition_internal(sub_request, depth + 1)
                    .await?;

                let proven = sub_result
                    .main_result
                    .best_proof
                    .as_ref()
                    .map(|p| p.confidence >= 0.5)
                    .unwrap_or(false);

                total_iterations += sub_result.main_result.steps.len() as u32;
                total_iterations += sub_result.total_iterations;

                sub_goal_results.push(SubGoalResult {
                    sub_goal_id: sub_goal.id.clone(),
                    result: sub_result.main_result,
                    proven,
                    nested_results: sub_result.sub_goal_results,
                });
            }

            // Compose final hints from all proven sub-goals
            let composition_hints: Vec<String> = sub_goal_results
                .iter()
                .filter(|r| r.proven)
                .flat_map(|r| {
                    let mut hints = vec![format!(
                        "Sub-goal '{}' is proven; use its proof structure",
                        r.sub_goal_id
                    )];
                    if let Some(proof) = &r.result.best_proof {
                        hints.push(format!(
                            "Tactics from {}: {}",
                            r.sub_goal_id,
                            proof.tactics_used.join(", ")
                        ));
                    }
                    hints
                })
                .collect();

            // Build final request with composition hints
            let final_hints: Vec<String> = request
                .additional_hints
                .iter()
                .chain(composition_hints.iter())
                .cloned()
                .collect();

            let final_request = ProofSearchRequest {
                property: request.property,
                backend: request.backend,
                context: request.context.clone(),
                propagate_to: request.propagate_to.clone(),
                additional_hints: final_hints,
                preferred_tactics: request.preferred_tactics.clone(),
                feedback: request.feedback.clone(),
                learning: request.learning,
            };

            // Run final search with composed hints
            let main_result = self.search(final_request).await?;
            total_iterations += main_result.steps.len() as u32;

            Ok(HierarchicalSearchResult {
                main_result,
                sub_goal_results,
                decomposition_strategy: decomposition.strategy,
                total_iterations,
            })
        }) // Close Box::pin(async move { ... })
    }

    /// Build propagated hints that future backends can consume
    fn derive_propagated_hints(
        &self,
        request: &ProofSearchRequest<'_>,
        feedback_hints: &[CrossBackendHint],
        best: Option<&SynthesisResult>,
    ) -> Vec<CrossBackendHint> {
        let mut hints = Vec::new();

        // Include feedback-derived hints
        hints.extend_from_slice(feedback_hints);

        // Propagate best proof outline to other backends
        if let Some(best) = best {
            for target in &request.propagate_to {
                if *target == request.backend {
                    continue;
                }
                hints.push(CrossBackendHint {
                    source_backend: request.backend,
                    target_backend: *target,
                    hint: format!(
                        "Proof attempt from {:?} with tactics [{}]",
                        request.backend,
                        best.tactics_used.join(", ")
                    ),
                    confidence: best.confidence,
                });
            }
        }

        // De-duplicate identical hints
        let mut seen = HashMap::new();
        hints.retain(|h| {
            seen.insert((h.source_backend, h.target_backend, h.hint.clone()), ())
                .is_none()
        });

        hints
    }
}

/// Decompose a property into sub-goals using structural analysis
///
/// # Arguments
/// * `property` - The property to decompose
/// * `depth` - Current decomposition depth
///
/// Uses simple induction mode by default. For strong/well-founded induction,
/// use `decompose_property_with_mode`.
pub fn decompose_property(property: &Property, depth: u32) -> DecompositionResult {
    decompose_property_with_mode(property, depth, InductionMode::Simple)
}

/// Decompose a property into sub-goals with a specific induction mode
///
/// # Arguments
/// * `property` - The property to decompose
/// * `depth` - Current decomposition depth
/// * `induction_mode` - Which induction strategy to use for numeric/recursive types
pub fn decompose_property_with_mode(
    property: &Property,
    depth: u32,
    induction_mode: InductionMode,
) -> DecompositionResult {
    use dashprove_usl::ast::{Expr, Theorem};

    // Helper to estimate expression complexity
    fn expr_complexity(expr: &Expr) -> f64 {
        match expr {
            Expr::Bool(_) | Expr::Int(_) | Expr::Float(_) | Expr::String(_) | Expr::Var(_) => 1.0,
            Expr::Not(inner) | Expr::Neg(inner) => 1.0 + expr_complexity(inner),
            Expr::And(l, r) | Expr::Or(l, r) | Expr::Implies(l, r) => {
                1.5 + expr_complexity(l) + expr_complexity(r)
            }
            Expr::Compare(l, _, r) | Expr::Binary(l, _, r) => {
                1.0 + expr_complexity(l) + expr_complexity(r)
            }
            Expr::ForAll { body, .. }
            | Expr::Exists { body, .. }
            | Expr::ForAllIn { body, .. }
            | Expr::ExistsIn { body, .. } => 2.0 + expr_complexity(body),
            Expr::App(_, args) => 1.0 + args.iter().map(expr_complexity).sum::<f64>(),
            Expr::MethodCall { receiver, args, .. } => {
                1.5 + expr_complexity(receiver) + args.iter().map(expr_complexity).sum::<f64>()
            }
            Expr::FieldAccess(obj, _) => 1.0 + expr_complexity(obj),
        }
    }

    // Extract the body expression from the property
    let (name, body) = match property {
        Property::Theorem(t) => (t.name.clone(), t.body.clone()),
        Property::Invariant(i) => (i.name.clone(), i.body.clone()),
        _ => {
            return DecompositionResult {
                sub_goals: vec![],
                strategy: DecompositionStrategy::None,
                complexity_reduction: 1.0,
            }
        }
    };

    let complexity = expr_complexity(&body);
    let mut sub_goals = Vec::new();
    let mut strategy = DecompositionStrategy::None;

    // Try conjunction split: `A && B` -> prove A, prove B
    if let Expr::And(left, right) = &body {
        let left_complexity = expr_complexity(left);
        let right_complexity = expr_complexity(right);

        if left_complexity > 2.0 || right_complexity > 2.0 {
            strategy = DecompositionStrategy::ConjunctionSplit;

            sub_goals.push(SubGoal {
                id: format!("{}_left", name),
                property: Property::Theorem(Theorem {
                    name: format!("{}_left", name),
                    body: *left.clone(),
                }),
                parent: Some(name.clone()),
                depth: depth + 1,
                proven: false,
                dependencies: vec![],
            });

            sub_goals.push(SubGoal {
                id: format!("{}_right", name),
                property: Property::Theorem(Theorem {
                    name: format!("{}_right", name),
                    body: *right.clone(),
                }),
                parent: Some(name.clone()),
                depth: depth + 1,
                proven: false,
                dependencies: vec![format!("{}_left", name)],
            });
        }
    }

    // Try disjunction elimination: `A || B` in hypothesis context
    // When we need to prove a goal G given A || B, prove G from A and G from B
    if let Expr::Or(left, right) = &body {
        if sub_goals.is_empty() {
            let left_complexity = expr_complexity(left);
            let right_complexity = expr_complexity(right);

            if left_complexity > 1.5 || right_complexity > 1.5 {
                strategy = DecompositionStrategy::DisjunctionElimination;

                // Sub-goal 1: prove A implies the goal (we treat A as hypothesis)
                sub_goals.push(SubGoal {
                    id: format!("{}_case_left", name),
                    property: Property::Theorem(Theorem {
                        name: format!("{}_case_left", name),
                        body: *left.clone(),
                    }),
                    parent: Some(name.clone()),
                    depth: depth + 1,
                    proven: false,
                    dependencies: vec![],
                });

                // Sub-goal 2: prove B implies the goal
                sub_goals.push(SubGoal {
                    id: format!("{}_case_right", name),
                    property: Property::Theorem(Theorem {
                        name: format!("{}_case_right", name),
                        body: *right.clone(),
                    }),
                    parent: Some(name.clone()),
                    depth: depth + 1,
                    proven: false,
                    dependencies: vec![],
                });
            }
        }
    }

    // Try implication chain: `A => B => C` -> prove A, then B assuming A, then C assuming A,B
    if let Expr::Implies(antecedent, consequent) = &body {
        if sub_goals.is_empty() && expr_complexity(consequent) > 3.0 {
            strategy = DecompositionStrategy::ImplicationChain;

            // First sub-goal: prove we can establish the antecedent
            sub_goals.push(SubGoal {
                id: format!("{}_antecedent", name),
                property: Property::Theorem(Theorem {
                    name: format!("{}_antecedent", name),
                    body: *antecedent.clone(),
                }),
                parent: Some(name.clone()),
                depth: depth + 1,
                proven: false,
                dependencies: vec![],
            });

            // Second sub-goal: prove the consequent assuming the antecedent
            sub_goals.push(SubGoal {
                id: format!("{}_consequent", name),
                property: Property::Theorem(Theorem {
                    name: format!("{}_consequent", name),
                    body: *consequent.clone(),
                }),
                parent: Some(name.clone()),
                depth: depth + 1,
                proven: false,
                dependencies: vec![format!("{}_antecedent", name)],
            });
        }
    }

    // Try case split for universal quantification over booleans
    if let Expr::ForAll {
        var,
        ty,
        body: inner_body,
    } = &body
    {
        if sub_goals.is_empty() {
            if let Some(dashprove_usl::ast::Type::Named(type_name)) = ty {
                if type_name == "Bool" {
                    strategy = DecompositionStrategy::CaseSplit;

                    // For case split on Bool, we substitute directly in the body
                    // by creating new subgoals with hints for the specific case
                    sub_goals.push(SubGoal {
                        id: format!("{}_true", name),
                        property: Property::Theorem(Theorem {
                            name: format!("{}_true", name),
                            // We keep the body and add hint that var = true
                            body: *inner_body.clone(),
                        }),
                        parent: Some(name.clone()),
                        depth: depth + 1,
                        proven: false,
                        dependencies: vec![],
                    });

                    sub_goals.push(SubGoal {
                        id: format!("{}_false", name),
                        property: Property::Theorem(Theorem {
                            name: format!("{}_false", name),
                            // We keep the body and add hint that var = false
                            body: *inner_body.clone(),
                        }),
                        parent: Some(name.clone()),
                        depth: depth + 1,
                        proven: false,
                        dependencies: vec![],
                    });

                    // Mark the var for substitution guidance
                    // (In a real implementation, we'd do proper substitution)
                    let _ = var; // Acknowledge unused var for now
                }
                // Try induction for universal quantification over Nat/Int
                else if type_name == "Nat" || type_name == "Int" || type_name == "ℕ" {
                    match induction_mode {
                        InductionMode::Simple => {
                            strategy = DecompositionStrategy::Induction;

                            // Base case: prove the property for 0
                            sub_goals.push(SubGoal {
                                id: format!("{}_base", name),
                                property: Property::Theorem(Theorem {
                                    name: format!("{}_base", name),
                                    body: *inner_body.clone(),
                                }),
                                parent: Some(name.clone()),
                                depth: depth + 1,
                                proven: false,
                                dependencies: vec![],
                            });

                            // Inductive step: prove P(n) => P(n+1)
                            sub_goals.push(SubGoal {
                                id: format!("{}_inductive", name),
                                property: Property::Theorem(Theorem {
                                    name: format!("{}_inductive", name),
                                    body: Expr::Implies(
                                        inner_body.clone(),
                                        inner_body.clone(), // P(n) => P(n+1) - simplified
                                    ),
                                }),
                                parent: Some(name.clone()),
                                depth: depth + 1,
                                proven: false,
                                dependencies: vec![format!("{}_base", name)],
                            });
                        }
                        InductionMode::Strong => {
                            strategy = DecompositionStrategy::StrongInduction;

                            // Strong induction: (∀k<n. P(k)) => P(n)
                            // Single sub-goal with stronger hypothesis
                            sub_goals.push(SubGoal {
                                id: format!("{}_strong_ih", name),
                                property: Property::Theorem(Theorem {
                                    name: format!("{}_strong_ih", name),
                                    // The body represents: given P holds for all k < n, prove P(n)
                                    // For the hint system, we encode this as the implication
                                    body: Expr::ForAll {
                                        var: format!("{}_k", var),
                                        ty: ty.clone(),
                                        body: Box::new(Expr::Implies(
                                            // Assumption: P(k) for all k < n (represented as inner_body)
                                            inner_body.clone(),
                                            // Goal: P(n) (represented as inner_body)
                                            inner_body.clone(),
                                        )),
                                    },
                                }),
                                parent: Some(name.clone()),
                                depth: depth + 1,
                                proven: false,
                                dependencies: vec![],
                            });
                        }
                        InductionMode::WellFounded => {
                            strategy = DecompositionStrategy::WellFoundedInduction;

                            // Well-founded induction: needs a decreasing measure
                            // We create a sub-goal that assumes the property holds for
                            // all elements smaller under the well-founded ordering
                            sub_goals.push(SubGoal {
                                id: format!("{}_wf_step", name),
                                property: Property::Theorem(Theorem {
                                    name: format!("{}_wf_step", name),
                                    // For well-founded: assume P(y) for all y < x, prove P(x)
                                    body: Expr::ForAll {
                                        var: format!("{}_smaller", var),
                                        ty: ty.clone(),
                                        body: Box::new(Expr::Implies(
                                            // Hypothesis: P holds for all smaller elements
                                            inner_body.clone(),
                                            // Goal: P(x)
                                            inner_body.clone(),
                                        )),
                                    },
                                }),
                                parent: Some(name.clone()),
                                depth: depth + 1,
                                proven: false,
                                dependencies: vec![],
                            });

                            // Well-foundedness proof obligation
                            sub_goals.push(SubGoal {
                                id: format!("{}_wf_order", name),
                                property: Property::Theorem(Theorem {
                                    name: format!("{}_wf_order", name),
                                    // Placeholder for well-foundedness of the ordering
                                    body: Expr::Bool(true), // Will be filled with actual WF proof
                                }),
                                parent: Some(name.clone()),
                                depth: depth + 1,
                                proven: false,
                                dependencies: vec![],
                            });
                        }
                    }

                    let _ = var; // Variable used in hints
                }
            }
        }
    }

    // Try induction for recursive type patterns (List, Tree, etc.)
    if let Expr::ForAll {
        var,
        ty,
        body: inner_body,
    } = &body
    {
        if sub_goals.is_empty() {
            if let Some(dashprove_usl::ast::Type::Named(type_name)) = ty {
                // Check for common recursive types
                if type_name == "List"
                    || type_name.starts_with("List<")
                    || type_name == "Tree"
                    || type_name.starts_with("Tree<")
                    || type_name.ends_with("List")
                {
                    match induction_mode {
                        InductionMode::Simple => {
                            strategy = DecompositionStrategy::Induction;

                            // Base case: empty list / leaf
                            sub_goals.push(SubGoal {
                                id: format!("{}_nil", name),
                                property: Property::Theorem(Theorem {
                                    name: format!("{}_nil", name),
                                    body: *inner_body.clone(),
                                }),
                                parent: Some(name.clone()),
                                depth: depth + 1,
                                proven: false,
                                dependencies: vec![],
                            });

                            // Inductive case: cons / node
                            sub_goals.push(SubGoal {
                                id: format!("{}_cons", name),
                                property: Property::Theorem(Theorem {
                                    name: format!("{}_cons", name),
                                    body: Expr::Implies(inner_body.clone(), inner_body.clone()),
                                }),
                                parent: Some(name.clone()),
                                depth: depth + 1,
                                proven: false,
                                dependencies: vec![format!("{}_nil", name)],
                            });
                        }
                        InductionMode::Strong => {
                            strategy = DecompositionStrategy::StrongInduction;

                            // Strong structural induction: P holds for all sublists
                            sub_goals.push(SubGoal {
                                id: format!("{}_strong_struct", name),
                                property: Property::Theorem(Theorem {
                                    name: format!("{}_strong_struct", name),
                                    body: Expr::ForAll {
                                        var: format!("{}_sub", var),
                                        ty: ty.clone(),
                                        body: Box::new(Expr::Implies(
                                            inner_body.clone(),
                                            inner_body.clone(),
                                        )),
                                    },
                                }),
                                parent: Some(name.clone()),
                                depth: depth + 1,
                                proven: false,
                                dependencies: vec![],
                            });
                        }
                        InductionMode::WellFounded => {
                            strategy = DecompositionStrategy::WellFoundedInduction;

                            // Well-founded on structure size
                            sub_goals.push(SubGoal {
                                id: format!("{}_wf_size", name),
                                property: Property::Theorem(Theorem {
                                    name: format!("{}_wf_size", name),
                                    body: Expr::ForAll {
                                        var: format!("{}_smaller", var),
                                        ty: ty.clone(),
                                        body: Box::new(Expr::Implies(
                                            inner_body.clone(),
                                            inner_body.clone(),
                                        )),
                                    },
                                }),
                                parent: Some(name.clone()),
                                depth: depth + 1,
                                proven: false,
                                dependencies: vec![],
                            });

                            sub_goals.push(SubGoal {
                                id: format!("{}_wf_measure", name),
                                property: Property::Theorem(Theorem {
                                    name: format!("{}_wf_measure", name),
                                    body: Expr::Bool(true), // size/length is well-founded
                                }),
                                parent: Some(name.clone()),
                                depth: depth + 1,
                                proven: false,
                                dependencies: vec![],
                            });
                        }
                    }

                    let _ = var;
                }
            }
        }
    }

    let sub_goal_complexity: f64 = sub_goals
        .iter()
        .map(|sg| {
            if let Property::Theorem(t) = &sg.property {
                expr_complexity(&t.body)
            } else {
                complexity
            }
        })
        .sum();

    let complexity_reduction = if sub_goals.is_empty() {
        1.0
    } else {
        (complexity - sub_goal_complexity.max(1.0)).max(0.0) / complexity
    };

    DecompositionResult {
        sub_goals,
        strategy,
        complexity_reduction,
    }
}

// ============================================================================
// SITA: Structure-to-Instance Theorem Application (Phase 18.6)
// ============================================================================

/// Template parameter kind - what type of value can be substituted
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateParamKind {
    /// A type parameter (e.g., `T` in `forall x: T`)
    Type,
    /// A function symbol (e.g., `op` in `op(x, y)`)
    Function { arity: usize },
    /// A predicate (e.g., `P` in `P(x)`)
    Predicate { arity: usize },
    /// A term/expression parameter
    Term,
    /// A constant value
    Constant,
}

/// A single template parameter with constraints
#[derive(Debug, Clone)]
pub struct TemplateParam {
    /// Parameter name (e.g., "T", "op", "P")
    pub name: String,
    /// Kind of parameter
    pub kind: TemplateParamKind,
    /// Human-readable description
    pub description: String,
    /// Optional constraint expression (e.g., "monoid", "commutative")
    pub constraints: Vec<String>,
}

/// Category of proof template for matching purposes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemplateCategory {
    /// Algebraic properties (associativity, commutativity, identity, etc.)
    Algebraic,
    /// Order/comparison properties (reflexivity, transitivity, antisymmetry)
    Order,
    /// Collection properties (membership, containment, cardinality)
    Collection,
    /// Induction templates (natural, structural, well-founded)
    Induction,
    /// Equivalence and isomorphism
    Equivalence,
    /// Function properties (injectivity, surjectivity, monotonicity)
    Function,
    /// Recursive structure properties
    Recursive,
    /// Custom/user-defined
    Custom,
}

/// A reusable proof template that can be instantiated for specific properties
///
/// Templates capture common proof patterns like:
/// - Associativity: `forall a b c. op(op(a, b), c) = op(a, op(b, c))`
/// - Commutativity: `forall a b. op(a, b) = op(b, a)`
/// - Induction: Base case + inductive step
#[derive(Debug, Clone)]
pub struct ProofTemplate {
    /// Unique identifier for this template
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Category for quick filtering
    pub category: TemplateCategory,
    /// Template parameters that can be instantiated
    pub params: Vec<TemplateParam>,
    /// The property pattern (with parameter placeholders)
    pub pattern: Property,
    /// Proof sketch or tactics to use
    pub tactics: Vec<String>,
    /// Hints for the synthesis system
    pub hints: Vec<String>,
    /// Prerequisites - other theorems that should be proven first
    pub prerequisites: Vec<String>,
    /// Success rate from previous applications (0.0 - 1.0)
    pub success_rate: f64,
    /// Number of times this template has been used
    pub usage_count: u64,
}

impl ProofTemplate {
    /// Create a new proof template
    pub fn new(id: impl Into<String>, name: impl Into<String>, category: TemplateCategory) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            category,
            params: Vec::new(),
            pattern: Property::Theorem(dashprove_usl::ast::Theorem {
                name: "template".to_string(),
                body: dashprove_usl::ast::Expr::Bool(true),
            }),
            tactics: Vec::new(),
            hints: Vec::new(),
            prerequisites: Vec::new(),
            success_rate: 0.5, // Start with neutral expectation
            usage_count: 0,
        }
    }

    /// Add a parameter to the template
    pub fn with_param(mut self, param: TemplateParam) -> Self {
        self.params.push(param);
        self
    }

    /// Set the pattern property
    pub fn with_pattern(mut self, pattern: Property) -> Self {
        self.pattern = pattern;
        self
    }

    /// Add recommended tactics
    pub fn with_tactics(mut self, tactics: Vec<String>) -> Self {
        self.tactics = tactics;
        self
    }

    /// Add hints for synthesis
    pub fn with_hints(mut self, hints: Vec<String>) -> Self {
        self.hints = hints;
        self
    }

    /// Add prerequisites
    pub fn with_prerequisites(mut self, prereqs: Vec<String>) -> Self {
        self.prerequisites = prereqs;
        self
    }
}

/// Result of matching a template against a property
#[derive(Debug, Clone)]
pub struct TemplateMatch {
    /// The matched template
    pub template_id: String,
    /// Parameter bindings discovered during matching
    pub bindings: HashMap<String, TemplateBinding>,
    /// Confidence of the match (0.0 - 1.0)
    pub confidence: f64,
    /// Why this template was matched
    pub rationale: String,
}

/// A binding from template parameter to concrete value
#[derive(Debug, Clone)]
pub enum TemplateBinding {
    /// Type binding (e.g., T -> Int)
    Type(String),
    /// Function binding (e.g., op -> addition)
    Function(String),
    /// Predicate binding (e.g., P -> is_even)
    Predicate(String),
    /// Term binding (e.g., e -> 0 for identity element)
    Term(dashprove_usl::ast::Expr),
    /// Constant binding
    Constant(String),
}

/// Registry of proof templates for matching and instantiation
#[derive(Debug, Default)]
pub struct TemplateRegistry {
    /// All registered templates
    templates: HashMap<String, ProofTemplate>,
    /// Index by category for fast lookup
    by_category: HashMap<TemplateCategory, Vec<String>>,
    /// Index by keyword for search
    by_keyword: HashMap<String, Vec<String>>,
}

impl TemplateRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with built-in templates
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register_builtins();
        registry
    }

    /// Register a new template
    pub fn register(&mut self, template: ProofTemplate) {
        let id = template.id.clone();
        let category = template.category;

        // Extract keywords from name and description
        let keywords: Vec<String> = template
            .name
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        self.templates.insert(id.clone(), template);
        self.by_category
            .entry(category)
            .or_default()
            .push(id.clone());

        for keyword in keywords {
            self.by_keyword.entry(keyword).or_default().push(id.clone());
        }
    }

    /// Get a template by ID
    pub fn get(&self, id: &str) -> Option<&ProofTemplate> {
        self.templates.get(id)
    }

    /// Get mutable reference to update statistics
    pub fn get_mut(&mut self, id: &str) -> Option<&mut ProofTemplate> {
        self.templates.get_mut(id)
    }

    /// Find templates matching a property
    pub fn find_matches(&self, property: &Property) -> Vec<TemplateMatch> {
        let mut matches = Vec::new();

        for template in self.templates.values() {
            if let Some(m) = self.try_match(template, property) {
                matches.push(m);
            }
        }

        // Sort by confidence (highest first)
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }

    /// Find templates by category
    pub fn find_by_category(&self, category: TemplateCategory) -> Vec<&ProofTemplate> {
        self.by_category
            .get(&category)
            .map(|ids| ids.iter().filter_map(|id| self.templates.get(id)).collect())
            .unwrap_or_default()
    }

    /// Search templates by keyword
    pub fn search(&self, keyword: &str) -> Vec<&ProofTemplate> {
        let kw = keyword.to_lowercase();
        self.by_keyword
            .get(&kw)
            .map(|ids| ids.iter().filter_map(|id| self.templates.get(id)).collect())
            .unwrap_or_default()
    }

    /// Try to match a template against a property
    fn try_match(&self, template: &ProofTemplate, property: &Property) -> Option<TemplateMatch> {
        #![allow(unused_imports)]
        use dashprove_usl::ast::Expr;

        let (prop_name, prop_body) = match property {
            Property::Theorem(t) => (&t.name, &t.body),
            Property::Invariant(i) => (&i.name, &i.body),
            _ => return None,
        };

        let (_, pattern_body) = match &template.pattern {
            Property::Theorem(t) => (&t.name, &t.body),
            Property::Invariant(i) => (&i.name, &i.body),
            _ => return None,
        };

        // Try structural pattern matching
        let mut bindings = HashMap::new();
        let confidence = self.match_expr(pattern_body, prop_body, &template.params, &mut bindings);

        if confidence > 0.3 {
            // Also check for keyword matches in the property name
            let name_boost = template
                .name
                .to_lowercase()
                .split_whitespace()
                .filter(|w| prop_name.to_lowercase().contains(w))
                .count() as f64
                * 0.1;

            Some(TemplateMatch {
                template_id: template.id.clone(),
                bindings,
                confidence: (confidence + name_boost).min(1.0),
                rationale: format!(
                    "Structural match with {} template ({})",
                    template.name, template.category as u8
                ),
            })
        } else {
            None
        }
    }

    /// Match expressions recursively, discovering parameter bindings
    #[allow(clippy::only_used_in_recursion)]
    fn match_expr(
        &self,
        pattern: &dashprove_usl::ast::Expr,
        target: &dashprove_usl::ast::Expr,
        params: &[TemplateParam],
        bindings: &mut HashMap<String, TemplateBinding>,
    ) -> f64 {
        use dashprove_usl::ast::Expr;

        // Check if pattern is a parameter placeholder
        if let Expr::Var(name) = pattern {
            if let Some(param) = params.iter().find(|p| &p.name == name) {
                // This is a template parameter - bind it
                let binding = match param.kind {
                    TemplateParamKind::Type => TemplateBinding::Type(format!("{:?}", target)),
                    TemplateParamKind::Function { .. } => {
                        TemplateBinding::Function(format!("{:?}", target))
                    }
                    TemplateParamKind::Predicate { .. } => {
                        TemplateBinding::Predicate(format!("{:?}", target))
                    }
                    TemplateParamKind::Term => TemplateBinding::Term(target.clone()),
                    TemplateParamKind::Constant => {
                        TemplateBinding::Constant(format!("{:?}", target))
                    }
                };
                bindings.insert(name.clone(), binding);
                return 1.0;
            }
        }

        match (pattern, target) {
            // Exact matches
            (Expr::Bool(a), Expr::Bool(b)) if a == b => 1.0,
            (Expr::Int(a), Expr::Int(b)) if a == b => 1.0,
            (Expr::Float(a), Expr::Float(b)) if (a - b).abs() < f64::EPSILON => 1.0,
            (Expr::String(a), Expr::String(b)) if a == b => 1.0,
            (Expr::Var(a), Expr::Var(b)) if a == b => 1.0,

            // Structural matches
            (Expr::Not(p), Expr::Not(t)) => self.match_expr(p, t, params, bindings),
            (Expr::Neg(p), Expr::Neg(t)) => self.match_expr(p, t, params, bindings),

            (Expr::And(p1, p2), Expr::And(t1, t2))
            | (Expr::Or(p1, p2), Expr::Or(t1, t2))
            | (Expr::Implies(p1, p2), Expr::Implies(t1, t2)) => {
                let m1 = self.match_expr(p1, t1, params, bindings);
                let m2 = self.match_expr(p2, t2, params, bindings);
                (m1 + m2) / 2.0
            }

            (Expr::Compare(p1, op1, p2), Expr::Compare(t1, op2, t2)) if op1 == op2 => {
                let m1 = self.match_expr(p1, t1, params, bindings);
                let m2 = self.match_expr(p2, t2, params, bindings);
                (m1 + m2) / 2.0
            }

            (Expr::Binary(p1, op1, p2), Expr::Binary(t1, op2, t2)) if op1 == op2 => {
                let m1 = self.match_expr(p1, t1, params, bindings);
                let m2 = self.match_expr(p2, t2, params, bindings);
                (m1 + m2) / 2.0
            }

            (
                Expr::ForAll {
                    var: _,
                    ty: ty1,
                    body: p,
                },
                Expr::ForAll {
                    var: _,
                    ty: ty2,
                    body: t,
                },
            ) => {
                // Match types if both present
                let type_match = match (ty1, ty2) {
                    (Some(t1), Some(t2)) => {
                        if t1 == t2 {
                            1.0
                        } else {
                            0.5
                        }
                    }
                    _ => 0.5,
                };
                let body_match = self.match_expr(p, t, params, bindings);
                (type_match + body_match) / 2.0
            }

            (
                Expr::Exists {
                    var: _,
                    ty: ty1,
                    body: p,
                },
                Expr::Exists {
                    var: _,
                    ty: ty2,
                    body: t,
                },
            ) => {
                let type_match = match (ty1, ty2) {
                    (Some(t1), Some(t2)) => {
                        if t1 == t2 {
                            1.0
                        } else {
                            0.5
                        }
                    }
                    _ => 0.5,
                };
                let body_match = self.match_expr(p, t, params, bindings);
                (type_match + body_match) / 2.0
            }

            (Expr::App(f1, args1), Expr::App(f2, args2)) if args1.len() == args2.len() => {
                // Check if f1 is a parameter
                let func_match = if params.iter().any(|p| &p.name == f1) {
                    bindings.insert(f1.clone(), TemplateBinding::Function(f2.clone()));
                    1.0
                } else if f1 == f2 {
                    1.0
                } else {
                    0.0
                };

                let args_match: f64 = args1
                    .iter()
                    .zip(args2.iter())
                    .map(|(p, t)| self.match_expr(p, t, params, bindings))
                    .sum::<f64>()
                    / args1.len().max(1) as f64;

                (func_match + args_match) / 2.0
            }

            _ => 0.0,
        }
    }

    /// Register built-in templates for common proof patterns
    fn register_builtins(&mut self) {
        use dashprove_usl::ast::{ComparisonOp, Expr, Theorem, Type};

        // Associativity template
        // forall a b c. op(op(a, b), c) = op(a, op(b, c))
        let assoc_template =
            ProofTemplate::new("assoc", "Associativity", TemplateCategory::Algebraic)
                .with_param(TemplateParam {
                    name: "T".to_string(),
                    kind: TemplateParamKind::Type,
                    description: "Element type".to_string(),
                    constraints: vec![],
                })
                .with_param(TemplateParam {
                    name: "op".to_string(),
                    kind: TemplateParamKind::Function { arity: 2 },
                    description: "Binary operation".to_string(),
                    constraints: vec![],
                })
                .with_pattern(Property::Theorem(Theorem {
                    name: "associativity".to_string(),
                    body: Expr::ForAll {
                        var: "a".to_string(),
                        ty: Some(Type::Named("T".to_string())),
                        body: Box::new(Expr::ForAll {
                            var: "b".to_string(),
                            ty: Some(Type::Named("T".to_string())),
                            body: Box::new(Expr::ForAll {
                                var: "c".to_string(),
                                ty: Some(Type::Named("T".to_string())),
                                body: Box::new(Expr::Compare(
                                    Box::new(Expr::App(
                                        "op".to_string(),
                                        vec![
                                            Expr::App(
                                                "op".to_string(),
                                                vec![
                                                    Expr::Var("a".to_string()),
                                                    Expr::Var("b".to_string()),
                                                ],
                                            ),
                                            Expr::Var("c".to_string()),
                                        ],
                                    )),
                                    ComparisonOp::Eq,
                                    Box::new(Expr::App(
                                        "op".to_string(),
                                        vec![
                                            Expr::Var("a".to_string()),
                                            Expr::App(
                                                "op".to_string(),
                                                vec![
                                                    Expr::Var("b".to_string()),
                                                    Expr::Var("c".to_string()),
                                                ],
                                            ),
                                        ],
                                    )),
                                )),
                            }),
                        }),
                    },
                }))
                .with_tactics(vec![
                    "unfold op".to_string(),
                    "ring".to_string(),
                    "simp [assoc_law]".to_string(),
                ])
                .with_hints(vec![
                    "Use associativity law".to_string(),
                    "May require unfolding definitions".to_string(),
                ]);
        self.register(assoc_template);

        // Commutativity template
        // forall a b. op(a, b) = op(b, a)
        let commut_template =
            ProofTemplate::new("commut", "Commutativity", TemplateCategory::Algebraic)
                .with_param(TemplateParam {
                    name: "T".to_string(),
                    kind: TemplateParamKind::Type,
                    description: "Element type".to_string(),
                    constraints: vec![],
                })
                .with_param(TemplateParam {
                    name: "op".to_string(),
                    kind: TemplateParamKind::Function { arity: 2 },
                    description: "Binary operation".to_string(),
                    constraints: vec![],
                })
                .with_pattern(Property::Theorem(Theorem {
                    name: "commutativity".to_string(),
                    body: Expr::ForAll {
                        var: "a".to_string(),
                        ty: Some(Type::Named("T".to_string())),
                        body: Box::new(Expr::ForAll {
                            var: "b".to_string(),
                            ty: Some(Type::Named("T".to_string())),
                            body: Box::new(Expr::Compare(
                                Box::new(Expr::App(
                                    "op".to_string(),
                                    vec![Expr::Var("a".to_string()), Expr::Var("b".to_string())],
                                )),
                                ComparisonOp::Eq,
                                Box::new(Expr::App(
                                    "op".to_string(),
                                    vec![Expr::Var("b".to_string()), Expr::Var("a".to_string())],
                                )),
                            )),
                        }),
                    },
                }))
                .with_tactics(vec![
                    "ring".to_string(),
                    "comm".to_string(),
                    "simp [comm_law]".to_string(),
                ])
                .with_hints(vec!["Use commutativity law".to_string()]);
        self.register(commut_template);

        // Identity element template
        // forall a. op(a, e) = a and op(e, a) = a
        let identity_template =
            ProofTemplate::new("identity", "Identity Element", TemplateCategory::Algebraic)
                .with_param(TemplateParam {
                    name: "T".to_string(),
                    kind: TemplateParamKind::Type,
                    description: "Element type".to_string(),
                    constraints: vec![],
                })
                .with_param(TemplateParam {
                    name: "op".to_string(),
                    kind: TemplateParamKind::Function { arity: 2 },
                    description: "Binary operation".to_string(),
                    constraints: vec![],
                })
                .with_param(TemplateParam {
                    name: "e".to_string(),
                    kind: TemplateParamKind::Term,
                    description: "Identity element".to_string(),
                    constraints: vec![],
                })
                .with_pattern(Property::Theorem(Theorem {
                    name: "identity".to_string(),
                    body: Expr::ForAll {
                        var: "a".to_string(),
                        ty: Some(Type::Named("T".to_string())),
                        body: Box::new(Expr::And(
                            Box::new(Expr::Compare(
                                Box::new(Expr::App(
                                    "op".to_string(),
                                    vec![Expr::Var("a".to_string()), Expr::Var("e".to_string())],
                                )),
                                ComparisonOp::Eq,
                                Box::new(Expr::Var("a".to_string())),
                            )),
                            Box::new(Expr::Compare(
                                Box::new(Expr::App(
                                    "op".to_string(),
                                    vec![Expr::Var("e".to_string()), Expr::Var("a".to_string())],
                                )),
                                ComparisonOp::Eq,
                                Box::new(Expr::Var("a".to_string())),
                            )),
                        )),
                    },
                }))
                .with_tactics(vec![
                    "simp [id_left, id_right]".to_string(),
                    "rfl".to_string(),
                ])
                .with_hints(vec![
                    "Apply identity laws".to_string(),
                    "May need case split on left/right identity".to_string(),
                ]);
        self.register(identity_template);

        // Reflexivity template
        // forall a. R(a, a)
        let refl_template = ProofTemplate::new("refl", "Reflexivity", TemplateCategory::Order)
            .with_param(TemplateParam {
                name: "T".to_string(),
                kind: TemplateParamKind::Type,
                description: "Element type".to_string(),
                constraints: vec![],
            })
            .with_param(TemplateParam {
                name: "R".to_string(),
                kind: TemplateParamKind::Predicate { arity: 2 },
                description: "Binary relation".to_string(),
                constraints: vec![],
            })
            .with_pattern(Property::Theorem(Theorem {
                name: "reflexivity".to_string(),
                body: Expr::ForAll {
                    var: "a".to_string(),
                    ty: Some(Type::Named("T".to_string())),
                    body: Box::new(Expr::App(
                        "R".to_string(),
                        vec![Expr::Var("a".to_string()), Expr::Var("a".to_string())],
                    )),
                },
            }))
            .with_tactics(vec!["rfl".to_string(), "intro; rfl".to_string()])
            .with_hints(vec!["Use reflexivity of the relation".to_string()]);
        self.register(refl_template);

        // Transitivity template
        // forall a b c. R(a, b) and R(b, c) => R(a, c)
        let trans_template = ProofTemplate::new("trans", "Transitivity", TemplateCategory::Order)
            .with_param(TemplateParam {
                name: "T".to_string(),
                kind: TemplateParamKind::Type,
                description: "Element type".to_string(),
                constraints: vec![],
            })
            .with_param(TemplateParam {
                name: "R".to_string(),
                kind: TemplateParamKind::Predicate { arity: 2 },
                description: "Binary relation".to_string(),
                constraints: vec![],
            })
            .with_pattern(Property::Theorem(Theorem {
                name: "transitivity".to_string(),
                body: Expr::ForAll {
                    var: "a".to_string(),
                    ty: Some(Type::Named("T".to_string())),
                    body: Box::new(Expr::ForAll {
                        var: "b".to_string(),
                        ty: Some(Type::Named("T".to_string())),
                        body: Box::new(Expr::ForAll {
                            var: "c".to_string(),
                            ty: Some(Type::Named("T".to_string())),
                            body: Box::new(Expr::Implies(
                                Box::new(Expr::And(
                                    Box::new(Expr::App(
                                        "R".to_string(),
                                        vec![
                                            Expr::Var("a".to_string()),
                                            Expr::Var("b".to_string()),
                                        ],
                                    )),
                                    Box::new(Expr::App(
                                        "R".to_string(),
                                        vec![
                                            Expr::Var("b".to_string()),
                                            Expr::Var("c".to_string()),
                                        ],
                                    )),
                                )),
                                Box::new(Expr::App(
                                    "R".to_string(),
                                    vec![Expr::Var("a".to_string()), Expr::Var("c".to_string())],
                                )),
                            )),
                        }),
                    }),
                },
            }))
            .with_tactics(vec![
                "intro h1 h2; trans b".to_string(),
                "apply trans_of".to_string(),
            ])
            .with_hints(vec![
                "Use transitivity of the relation".to_string(),
                "May need intermediate witness".to_string(),
            ]);
        self.register(trans_template);

        // Symmetry template
        // forall a b. R(a, b) => R(b, a)
        let symm_template = ProofTemplate::new("symm", "Symmetry", TemplateCategory::Order)
            .with_param(TemplateParam {
                name: "T".to_string(),
                kind: TemplateParamKind::Type,
                description: "Element type".to_string(),
                constraints: vec![],
            })
            .with_param(TemplateParam {
                name: "R".to_string(),
                kind: TemplateParamKind::Predicate { arity: 2 },
                description: "Binary relation".to_string(),
                constraints: vec![],
            })
            .with_pattern(Property::Theorem(Theorem {
                name: "symmetry".to_string(),
                body: Expr::ForAll {
                    var: "a".to_string(),
                    ty: Some(Type::Named("T".to_string())),
                    body: Box::new(Expr::ForAll {
                        var: "b".to_string(),
                        ty: Some(Type::Named("T".to_string())),
                        body: Box::new(Expr::Implies(
                            Box::new(Expr::App(
                                "R".to_string(),
                                vec![Expr::Var("a".to_string()), Expr::Var("b".to_string())],
                            )),
                            Box::new(Expr::App(
                                "R".to_string(),
                                vec![Expr::Var("b".to_string()), Expr::Var("a".to_string())],
                            )),
                        )),
                    }),
                },
            }))
            .with_tactics(vec!["intro h; symm; exact h".to_string()])
            .with_hints(vec!["Use symmetry of the relation".to_string()]);
        self.register(symm_template);

        // Natural number induction template
        // (P(0) and (forall n. P(n) => P(n+1))) => forall n. P(n)
        let nat_ind_template = ProofTemplate::new(
            "nat_ind",
            "Natural Number Induction",
            TemplateCategory::Induction,
        )
        .with_param(TemplateParam {
            name: "P".to_string(),
            kind: TemplateParamKind::Predicate { arity: 1 },
            description: "Property to prove".to_string(),
            constraints: vec![],
        })
        .with_pattern(Property::Theorem(Theorem {
            name: "nat_induction".to_string(),
            body: Expr::ForAll {
                var: "n".to_string(),
                ty: Some(Type::Named("Nat".to_string())),
                body: Box::new(Expr::App("P".to_string(), vec![Expr::Var("n".to_string())])),
            },
        }))
        .with_tactics(vec![
            "induction n".to_string(),
            "· -- base case".to_string(),
            "· -- inductive case".to_string(),
        ])
        .with_hints(vec![
            "Use induction on natural numbers".to_string(),
            "Base case: prove P(0)".to_string(),
            "Inductive step: assume P(n), prove P(n+1)".to_string(),
        ]);
        self.register(nat_ind_template);

        // List induction template
        let list_ind_template =
            ProofTemplate::new("list_ind", "List Induction", TemplateCategory::Induction)
                .with_param(TemplateParam {
                    name: "T".to_string(),
                    kind: TemplateParamKind::Type,
                    description: "Element type".to_string(),
                    constraints: vec![],
                })
                .with_param(TemplateParam {
                    name: "P".to_string(),
                    kind: TemplateParamKind::Predicate { arity: 1 },
                    description: "Property to prove".to_string(),
                    constraints: vec![],
                })
                .with_pattern(Property::Theorem(Theorem {
                    name: "list_induction".to_string(),
                    body: Expr::ForAll {
                        var: "xs".to_string(),
                        ty: Some(Type::Named("List".to_string())),
                        body: Box::new(Expr::App(
                            "P".to_string(),
                            vec![Expr::Var("xs".to_string())],
                        )),
                    },
                }))
                .with_tactics(vec![
                    "induction xs".to_string(),
                    "· -- nil case".to_string(),
                    "· -- cons case".to_string(),
                ])
                .with_hints(vec![
                    "Use structural induction on lists".to_string(),
                    "Base case: prove P(nil)".to_string(),
                    "Inductive step: assume P(xs), prove P(x::xs)".to_string(),
                ]);
        self.register(list_ind_template);

        // Distributivity template
        // forall a b c. op1(a, op2(b, c)) = op2(op1(a, b), op1(a, c))
        let distrib_template =
            ProofTemplate::new("distrib", "Distributivity", TemplateCategory::Algebraic)
                .with_param(TemplateParam {
                    name: "T".to_string(),
                    kind: TemplateParamKind::Type,
                    description: "Element type".to_string(),
                    constraints: vec![],
                })
                .with_param(TemplateParam {
                    name: "op1".to_string(),
                    kind: TemplateParamKind::Function { arity: 2 },
                    description: "Distributing operation".to_string(),
                    constraints: vec![],
                })
                .with_param(TemplateParam {
                    name: "op2".to_string(),
                    kind: TemplateParamKind::Function { arity: 2 },
                    description: "Operation being distributed over".to_string(),
                    constraints: vec![],
                })
                .with_pattern(Property::Theorem(Theorem {
                    name: "distributivity".to_string(),
                    body: Expr::ForAll {
                        var: "a".to_string(),
                        ty: Some(Type::Named("T".to_string())),
                        body: Box::new(Expr::ForAll {
                            var: "b".to_string(),
                            ty: Some(Type::Named("T".to_string())),
                            body: Box::new(Expr::ForAll {
                                var: "c".to_string(),
                                ty: Some(Type::Named("T".to_string())),
                                body: Box::new(Expr::Compare(
                                    Box::new(Expr::App(
                                        "op1".to_string(),
                                        vec![
                                            Expr::Var("a".to_string()),
                                            Expr::App(
                                                "op2".to_string(),
                                                vec![
                                                    Expr::Var("b".to_string()),
                                                    Expr::Var("c".to_string()),
                                                ],
                                            ),
                                        ],
                                    )),
                                    ComparisonOp::Eq,
                                    Box::new(Expr::App(
                                        "op2".to_string(),
                                        vec![
                                            Expr::App(
                                                "op1".to_string(),
                                                vec![
                                                    Expr::Var("a".to_string()),
                                                    Expr::Var("b".to_string()),
                                                ],
                                            ),
                                            Expr::App(
                                                "op1".to_string(),
                                                vec![
                                                    Expr::Var("a".to_string()),
                                                    Expr::Var("c".to_string()),
                                                ],
                                            ),
                                        ],
                                    )),
                                )),
                            }),
                        }),
                    },
                }))
                .with_tactics(vec!["ring".to_string(), "simp [distrib_law]".to_string()])
                .with_hints(vec!["Apply distributivity law".to_string()]);
        self.register(distrib_template);
    }

    /// Count total templates registered
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// List all template IDs
    pub fn template_ids(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }
}

/// Instantiate a template with specific bindings to produce a property
pub fn instantiate_template(
    template: &ProofTemplate,
    bindings: &HashMap<String, TemplateBinding>,
) -> Result<Property, String> {
    use dashprove_usl::ast::Theorem;

    let (name, body) = match &template.pattern {
        Property::Theorem(t) => (&t.name, &t.body),
        Property::Invariant(i) => (&i.name, &i.body),
        _ => return Err("Unsupported property type for instantiation".to_string()),
    };

    // Substitute parameters in the body
    let instantiated_body = substitute_expr(body, bindings)?;

    Ok(Property::Theorem(Theorem {
        name: name.clone(),
        body: instantiated_body,
    }))
}

/// Substitute bindings in an expression
fn substitute_expr(
    expr: &dashprove_usl::ast::Expr,
    bindings: &HashMap<String, TemplateBinding>,
) -> Result<dashprove_usl::ast::Expr, String> {
    use dashprove_usl::ast::Expr;

    match expr {
        Expr::Var(name) => {
            if let Some(binding) = bindings.get(name) {
                match binding {
                    TemplateBinding::Term(e) => Ok(e.clone()),
                    TemplateBinding::Constant(c) => Ok(Expr::Var(c.clone())),
                    _ => Ok(expr.clone()),
                }
            } else {
                Ok(expr.clone())
            }
        }
        Expr::Bool(_) | Expr::Int(_) | Expr::Float(_) | Expr::String(_) => Ok(expr.clone()),
        Expr::Not(inner) => Ok(Expr::Not(Box::new(substitute_expr(inner, bindings)?))),
        Expr::Neg(inner) => Ok(Expr::Neg(Box::new(substitute_expr(inner, bindings)?))),
        Expr::And(l, r) => Ok(Expr::And(
            Box::new(substitute_expr(l, bindings)?),
            Box::new(substitute_expr(r, bindings)?),
        )),
        Expr::Or(l, r) => Ok(Expr::Or(
            Box::new(substitute_expr(l, bindings)?),
            Box::new(substitute_expr(r, bindings)?),
        )),
        Expr::Implies(l, r) => Ok(Expr::Implies(
            Box::new(substitute_expr(l, bindings)?),
            Box::new(substitute_expr(r, bindings)?),
        )),
        Expr::Compare(l, op, r) => Ok(Expr::Compare(
            Box::new(substitute_expr(l, bindings)?),
            *op,
            Box::new(substitute_expr(r, bindings)?),
        )),
        Expr::Binary(l, op, r) => Ok(Expr::Binary(
            Box::new(substitute_expr(l, bindings)?),
            *op,
            Box::new(substitute_expr(r, bindings)?),
        )),
        Expr::App(func, args) => {
            // Check if function name should be substituted
            let new_func = if let Some(TemplateBinding::Function(f)) = bindings.get(func) {
                f.clone()
            } else {
                func.clone()
            };
            let new_args: Result<Vec<_>, _> =
                args.iter().map(|a| substitute_expr(a, bindings)).collect();
            Ok(Expr::App(new_func, new_args?))
        }
        Expr::ForAll { var, ty, body } => {
            // Substitute type if it's a parameter
            let new_ty = ty.as_ref().map(|t| substitute_type(t, bindings));
            Ok(Expr::ForAll {
                var: var.clone(),
                ty: new_ty,
                body: Box::new(substitute_expr(body, bindings)?),
            })
        }
        Expr::Exists { var, ty, body } => {
            let new_ty = ty.as_ref().map(|t| substitute_type(t, bindings));
            Ok(Expr::Exists {
                var: var.clone(),
                ty: new_ty,
                body: Box::new(substitute_expr(body, bindings)?),
            })
        }
        Expr::ForAllIn {
            var,
            collection,
            body,
        } => Ok(Expr::ForAllIn {
            var: var.clone(),
            collection: Box::new(substitute_expr(collection, bindings)?),
            body: Box::new(substitute_expr(body, bindings)?),
        }),
        Expr::ExistsIn {
            var,
            collection,
            body,
        } => Ok(Expr::ExistsIn {
            var: var.clone(),
            collection: Box::new(substitute_expr(collection, bindings)?),
            body: Box::new(substitute_expr(body, bindings)?),
        }),
        Expr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let new_args: Result<Vec<_>, _> =
                args.iter().map(|a| substitute_expr(a, bindings)).collect();
            Ok(Expr::MethodCall {
                receiver: Box::new(substitute_expr(receiver, bindings)?),
                method: method.clone(),
                args: new_args?,
            })
        }
        Expr::FieldAccess(obj, field) => Ok(Expr::FieldAccess(
            Box::new(substitute_expr(obj, bindings)?),
            field.clone(),
        )),
    }
}

/// Substitute bindings in a type
fn substitute_type(
    ty: &dashprove_usl::ast::Type,
    bindings: &HashMap<String, TemplateBinding>,
) -> dashprove_usl::ast::Type {
    use dashprove_usl::ast::Type;

    match ty {
        Type::Named(name) => {
            if let Some(TemplateBinding::Type(new_name)) = bindings.get(name) {
                Type::Named(new_name.clone())
            } else {
                ty.clone()
            }
        }
        Type::Set(inner) => Type::Set(Box::new(substitute_type(inner, bindings))),
        Type::List(inner) => Type::List(Box::new(substitute_type(inner, bindings))),
        Type::Map(k, v) => Type::Map(
            Box::new(substitute_type(k, bindings)),
            Box::new(substitute_type(v, bindings)),
        ),
        Type::Relation(a, b) => Type::Relation(
            Box::new(substitute_type(a, bindings)),
            Box::new(substitute_type(b, bindings)),
        ),
        Type::Function(param, ret) => Type::Function(
            Box::new(substitute_type(param, bindings)),
            Box::new(substitute_type(ret, bindings)),
        ),
        Type::Result(inner) => Type::Result(Box::new(substitute_type(inner, bindings))),
        Type::Graph(n, e) => Type::Graph(
            Box::new(substitute_type(n, bindings)),
            Box::new(substitute_type(e, bindings)),
        ),
        Type::Path(inner) => Type::Path(Box::new(substitute_type(inner, bindings))),
        Type::Unit => Type::Unit,
    }
}

// ============================================================================
// End of SITA Templates
// ============================================================================

/// Lightweight policy over tactic weights with reward updates
#[derive(Debug, Default, Clone)]
struct TacticPolicy {
    weights: HashMap<String, f64>,
    exploration_rate: f64,
}

impl TacticPolicy {
    fn rank(
        &mut self,
        tactics: &[TacticSuggestion],
        preferred: &[String],
    ) -> Vec<TacticSuggestion> {
        let mut ranked: Vec<_> = tactics
            .iter()
            .cloned()
            .map(|mut t| {
                let weight = *self.weights.get(&t.tactic).unwrap_or(&0.5);
                t.confidence =
                    Confidence::from_score((t.confidence.to_score() * 0.6) + (weight * 0.4));
                t
            })
            .collect();

        // Deterministic exploration: rotate a slice of lowest-weight tactics to front
        let explore_count = ((ranked.len() as f64) * self.exploration_rate).ceil() as usize;
        ranked.sort_by(|a, b| {
            b.confidence
                .to_score()
                .partial_cmp(&a.confidence.to_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if explore_count > 0 && explore_count < ranked.len() {
            let tail = ranked.split_off(ranked.len() - explore_count);
            ranked.splice(0..0, tail);
        }

        // Ensure preferred tactics stay near the front
        for (idx, tactic) in preferred.iter().enumerate() {
            if let Some(pos) = ranked.iter().position(|t| &t.tactic == tactic) {
                let tactic = ranked.remove(pos);
                ranked.insert(idx.min(ranked.len()), tactic);
            }
        }

        ranked
    }

    fn apply_reward(&mut self, tactics: &[String], reward: f64) {
        for tactic in tactics {
            let entry = self.weights.entry(tactic.clone()).or_insert(0.5);
            *entry = (*entry + reward).clamp(0.0, 1.5);
        }
    }

    fn snapshot(&self) -> TacticPolicySnapshot {
        let mut weights: Vec<_> = self.weights.iter().map(|(k, v)| (k.clone(), *v)).collect();
        weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        TacticPolicySnapshot { weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmClient, LlmError, LlmMessage, LlmResponse};
    use async_trait::async_trait;
    use dashprove_backends::counterexample::StructuredCounterexample;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn simple_property() -> Property {
        Property::Theorem(dashprove_usl::ast::Theorem {
            name: "trivial".to_string(),
            body: dashprove_usl::ast::Expr::Bool(true),
        })
    }

    #[test]
    fn policy_updates_weights_with_reward() {
        let mut policy = TacticPolicy::default();
        let suggestions = vec![
            TacticSuggestion {
                tactic: "simp".to_string(),
                confidence: Confidence::Low,
                source: SuggestionSource::Compiler,
                rationale: "baseline".to_string(),
            },
            TacticSuggestion {
                tactic: "intro".to_string(),
                confidence: Confidence::Low,
                source: SuggestionSource::Compiler,
                rationale: "baseline".to_string(),
            },
        ];

        let ranked = policy.rank(&suggestions, &[]);
        assert_eq!(ranked.len(), 2);

        policy.apply_reward(&["simp".to_string()], 0.8);
        let reranked = policy.rank(&suggestions, &[]);
        assert_eq!(reranked.first().unwrap().tactic, "simp");
        let simp_weight = policy.weights.get("simp").expect("simp weight");
        assert!(*simp_weight > 0.5);
        assert!(!policy.weights.contains_key("intro"));
    }

    struct SequenceLlm {
        responses: Vec<String>,
        index: AtomicUsize,
    }

    impl SequenceLlm {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses,
                index: AtomicUsize::new(0),
            }
        }

        fn next_response(&self) -> String {
            let idx = self.index.fetch_add(1, Ordering::SeqCst);
            self.responses
                .get(idx)
                .cloned()
                .unwrap_or_else(|| "fallback".to_string())
        }
    }

    #[async_trait]
    impl LlmClient for SequenceLlm {
        async fn complete(&self, _prompt: &str) -> Result<LlmResponse, LlmError> {
            Ok(LlmResponse {
                content: self.next_response(),
                model: "test".to_string(),
                input_tokens: None,
                output_tokens: None,
                stop_reason: None,
            })
        }

        async fn complete_messages(
            &self,
            _messages: &[LlmMessage],
        ) -> Result<LlmResponse, LlmError> {
            self.complete("").await
        }

        fn is_configured(&self) -> bool {
            true
        }

        fn model_id(&self) -> &str {
            "test"
        }
    }

    #[tokio::test]
    async fn proof_search_iterates_until_valid_attempt() {
        let llm = SequenceLlm::new(vec![
            "```lean\ntheorem trivial : True := by\n  sorry\n```".to_string(), // synthesis #1
            "```lean\ntheorem trivial : True := by\n  trivial\n```".to_string(), // synthesis #2
        ]);

        let synthesizer = ProofSynthesizer::new(Box::new(llm));
        let predictor = LlmTacticPredictor::new(Box::new(SequenceLlm::new(vec![
            "1. simp - simplify\n2. intro - introduce hypothesis".to_string(),
            "1. trivial - solve\n2. rfl - reflexivity".to_string(),
        ])));

        let mut agent = ProofSearchAgent::new(synthesizer)
            .with_tactic_predictor(predictor)
            .with_config(ProofSearchConfig {
                max_iterations: 3,
                max_attempts_per_iteration: 1,
                validation_threshold: 0.5,
                success_reward: 1.0,
                failure_penalty: 0.4,
                exploration_rate: 0.0,
                max_hints: 3,
                enable_decomposition: false,
                max_decomposition_depth: 2,
                decomposition_complexity_threshold: 0.7,
                induction_mode: InductionMode::Simple,
            });

        let property = simple_property();
        let request = ProofSearchRequest::new(&property, BackendId::Lean4);
        let result = agent.search(request).await.expect("search should succeed");

        assert_eq!(result.steps.len(), 2);
        assert!(result.best_proof.is_some());
        let best = result.best_proof.unwrap();
        assert!(
            !best.proof.contains("sorry"),
            "best proof should not contain sorry"
        );
        assert!(!result.policy.weights.is_empty());
    }

    #[test]
    fn decompose_conjunction_splits_into_subgoals() {
        use dashprove_usl::ast::{Expr, Theorem};

        let property = Property::Theorem(Theorem {
            name: "conjunction".to_string(),
            body: Expr::And(
                Box::new(Expr::ForAll {
                    var: "x".to_string(),
                    ty: Some(dashprove_usl::ast::Type::Named("Nat".to_string())),
                    body: Box::new(Expr::Bool(true)),
                }),
                Box::new(Expr::ForAll {
                    var: "y".to_string(),
                    ty: Some(dashprove_usl::ast::Type::Named("Nat".to_string())),
                    body: Box::new(Expr::Bool(true)),
                }),
            ),
        });

        let result = decompose_property(&property, 0);
        assert_eq!(result.strategy, DecompositionStrategy::ConjunctionSplit);
        assert_eq!(result.sub_goals.len(), 2);
        assert_eq!(result.sub_goals[0].id, "conjunction_left");
        assert_eq!(result.sub_goals[1].id, "conjunction_right");
        assert!(result.sub_goals[1]
            .dependencies
            .contains(&"conjunction_left".to_string()));
    }

    #[test]
    fn decompose_implication_creates_chain() {
        use dashprove_usl::ast::{Expr, Theorem};

        let property = Property::Theorem(Theorem {
            name: "impl_chain".to_string(),
            body: Expr::Implies(
                Box::new(Expr::Var("P".to_string())),
                Box::new(Expr::ForAll {
                    var: "x".to_string(),
                    ty: Some(dashprove_usl::ast::Type::Named("Nat".to_string())),
                    body: Box::new(Expr::And(
                        Box::new(Expr::Var("Q".to_string())),
                        Box::new(Expr::Var("R".to_string())),
                    )),
                }),
            ),
        });

        let result = decompose_property(&property, 0);
        assert_eq!(result.strategy, DecompositionStrategy::ImplicationChain);
        assert_eq!(result.sub_goals.len(), 2);
        assert_eq!(result.sub_goals[0].id, "impl_chain_antecedent");
        assert_eq!(result.sub_goals[1].id, "impl_chain_consequent");
    }

    #[test]
    fn decompose_forall_bool_creates_case_split() {
        use dashprove_usl::ast::{Expr, Theorem};

        let property = Property::Theorem(Theorem {
            name: "bool_cases".to_string(),
            body: Expr::ForAll {
                var: "b".to_string(),
                ty: Some(dashprove_usl::ast::Type::Named("Bool".to_string())),
                body: Box::new(Expr::Or(
                    Box::new(Expr::Var("b".to_string())),
                    Box::new(Expr::Not(Box::new(Expr::Var("b".to_string())))),
                )),
            },
        });

        let result = decompose_property(&property, 0);
        assert_eq!(result.strategy, DecompositionStrategy::CaseSplit);
        assert_eq!(result.sub_goals.len(), 2);
        assert_eq!(result.sub_goals[0].id, "bool_cases_true");
        assert_eq!(result.sub_goals[1].id, "bool_cases_false");
    }

    #[test]
    fn decompose_simple_property_returns_none() {
        let property = simple_property();
        let result = decompose_property(&property, 0);
        assert_eq!(result.strategy, DecompositionStrategy::None);
        assert!(result.sub_goals.is_empty());
    }

    #[test]
    fn decompose_forall_nat_creates_induction() {
        use dashprove_usl::ast::{ComparisonOp, Expr, Theorem};

        let property = Property::Theorem(Theorem {
            name: "nat_induction".to_string(),
            body: Expr::ForAll {
                var: "n".to_string(),
                ty: Some(dashprove_usl::ast::Type::Named("Nat".to_string())),
                body: Box::new(Expr::Compare(
                    Box::new(Expr::Var("n".to_string())),
                    ComparisonOp::Ge,
                    Box::new(Expr::Int(0)),
                )),
            },
        });

        let result = decompose_property(&property, 0);
        assert_eq!(result.strategy, DecompositionStrategy::Induction);
        assert_eq!(result.sub_goals.len(), 2);
        assert_eq!(result.sub_goals[0].id, "nat_induction_base");
        assert_eq!(result.sub_goals[1].id, "nat_induction_inductive");
        // Inductive case depends on base case
        assert!(result.sub_goals[1]
            .dependencies
            .contains(&"nat_induction_base".to_string()));
    }

    #[test]
    fn decompose_forall_list_creates_induction() {
        use dashprove_usl::ast::{Expr, Theorem};

        let property = Property::Theorem(Theorem {
            name: "list_induction".to_string(),
            body: Expr::ForAll {
                var: "xs".to_string(),
                ty: Some(dashprove_usl::ast::Type::Named("List<Int>".to_string())),
                body: Box::new(Expr::Bool(true)),
            },
        });

        let result = decompose_property(&property, 0);
        assert_eq!(result.strategy, DecompositionStrategy::Induction);
        assert_eq!(result.sub_goals.len(), 2);
        assert_eq!(result.sub_goals[0].id, "list_induction_nil");
        assert_eq!(result.sub_goals[1].id, "list_induction_cons");
    }

    #[test]
    fn cross_backend_hints_include_counterexamples() {
        let mut ce = StructuredCounterexample::new();
        ce.witness.insert(
            "x".to_string(),
            dashprove_backends::counterexample::CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );

        let feedback = BackendResult {
            backend: BackendId::Kani,
            status: VerificationStatus::Disproven,
            proof: None,
            counterexample: Some(ce),
            diagnostics: vec!["overflow in loop".to_string()],
            time_taken: std::time::Duration::from_secs(1),
        };

        let property = simple_property();
        let request = ProofSearchRequest {
            property: &property,
            backend: BackendId::Lean4,
            context: None,
            propagate_to: vec![BackendId::Coq],
            additional_hints: Vec::new(),
            preferred_tactics: Vec::new(),
            feedback: vec![feedback.clone()],
            learning: None,
        };

        let agent =
            ProofSearchAgent::new(ProofSynthesizer::new(Box::new(SequenceLlm::new(vec![]))));

        let hints = agent.feedback_hints(&request);
        assert_eq!(hints.len(), 1);
        let hint = &hints[0];
        assert_eq!(hint.source_backend, BackendId::Kani);
        assert_eq!(hint.target_backend, BackendId::Coq);
        assert!(hint.hint.contains("counterexample"));
        assert!(hint.confidence > 0.5);
    }

    #[tokio::test]
    async fn hierarchical_search_decomposes_conjunction() {
        use dashprove_usl::ast::{Expr, Theorem};

        // Create an LLM that returns valid proofs
        let llm = SequenceLlm::new(vec![
            "```lean\ntheorem left : True := by trivial\n```".to_string(),
            "```lean\ntheorem right : True := by trivial\n```".to_string(),
            "```lean\ntheorem conj : True ∧ True := by exact ⟨trivial, trivial⟩\n```".to_string(),
        ]);

        let synthesizer = ProofSynthesizer::new(Box::new(llm));
        let mut agent = ProofSearchAgent::new(synthesizer).with_config(ProofSearchConfig {
            max_iterations: 2,
            max_attempts_per_iteration: 1,
            validation_threshold: 0.3,
            success_reward: 1.0,
            failure_penalty: 0.4,
            exploration_rate: 0.0,
            max_hints: 3,
            enable_decomposition: true,
            max_decomposition_depth: 2,
            decomposition_complexity_threshold: 0.0, // Decompose everything
            induction_mode: InductionMode::Simple,
        });

        // Create a conjunction property
        let property = Property::Theorem(Theorem {
            name: "conjunction_proof".to_string(),
            body: Expr::And(
                Box::new(Expr::ForAll {
                    var: "x".to_string(),
                    ty: Some(dashprove_usl::ast::Type::Named("Nat".to_string())),
                    body: Box::new(Expr::Bool(true)),
                }),
                Box::new(Expr::ForAll {
                    var: "y".to_string(),
                    ty: Some(dashprove_usl::ast::Type::Named("Nat".to_string())),
                    body: Box::new(Expr::Bool(true)),
                }),
            ),
        });

        let request = ProofSearchRequest::new(&property, BackendId::Lean4);
        let result = agent
            .search_with_decomposition(request)
            .await
            .expect("search should succeed");

        // Should have used conjunction split
        assert_eq!(
            result.decomposition_strategy,
            DecompositionStrategy::ConjunctionSplit
        );

        // Should have sub-goal results
        assert!(!result.sub_goal_results.is_empty());

        // Should have main result
        assert!(!result.main_result.steps.is_empty());
    }

    #[tokio::test]
    async fn hierarchical_search_respects_depth_limit() {
        use dashprove_usl::ast::{Expr, Theorem};

        let llm = SequenceLlm::new(vec![
            "```lean\ntheorem nested : True := by trivial\n```".to_string()
        ]);

        let synthesizer = ProofSynthesizer::new(Box::new(llm));
        let mut agent = ProofSearchAgent::new(synthesizer).with_config(ProofSearchConfig {
            max_iterations: 1,
            max_attempts_per_iteration: 1,
            validation_threshold: 0.3,
            success_reward: 1.0,
            failure_penalty: 0.4,
            exploration_rate: 0.0,
            max_hints: 3,
            enable_decomposition: true,
            max_decomposition_depth: 0, // No decomposition allowed
            decomposition_complexity_threshold: 0.0,
            induction_mode: InductionMode::Simple,
        });

        // Create a nested conjunction that would normally decompose
        let property = Property::Theorem(Theorem {
            name: "nested".to_string(),
            body: Expr::And(
                Box::new(Expr::And(
                    Box::new(Expr::Bool(true)),
                    Box::new(Expr::Bool(true)),
                )),
                Box::new(Expr::Bool(true)),
            ),
        });

        let request = ProofSearchRequest::new(&property, BackendId::Lean4);
        let result = agent
            .search_with_decomposition(request)
            .await
            .expect("search should succeed");

        // Should NOT have decomposed due to depth limit
        assert_eq!(result.decomposition_strategy, DecompositionStrategy::None);
        assert!(result.sub_goal_results.is_empty());
    }

    #[tokio::test]
    async fn hierarchical_search_disabled_runs_normal_search() {
        let llm = SequenceLlm::new(vec![
            "```lean\ntheorem trivial : True := by trivial\n```".to_string()
        ]);

        let synthesizer = ProofSynthesizer::new(Box::new(llm));
        let mut agent = ProofSearchAgent::new(synthesizer).with_config(ProofSearchConfig {
            max_iterations: 2,
            max_attempts_per_iteration: 1,
            validation_threshold: 0.3,
            success_reward: 1.0,
            failure_penalty: 0.4,
            exploration_rate: 0.0,
            max_hints: 3,
            enable_decomposition: false, // Disabled
            max_decomposition_depth: 3,
            decomposition_complexity_threshold: 0.0,
            induction_mode: InductionMode::Simple,
        });

        let property = simple_property();
        let request = ProofSearchRequest::new(&property, BackendId::Lean4);
        let result = agent
            .search_with_decomposition(request)
            .await
            .expect("search should succeed");

        // Should have no decomposition
        assert_eq!(result.decomposition_strategy, DecompositionStrategy::None);
        assert!(result.sub_goal_results.is_empty());
        // But should still have main result
        assert!(!result.main_result.steps.is_empty());
    }

    #[test]
    fn sub_goal_result_proven_count_works() {
        let empty_result = ProofSearchResult {
            best_proof: None,
            steps: vec![],
            propagated_hints: vec![],
            policy: TacticPolicySnapshot { weights: vec![] },
        };

        let sub_goal = SubGoalResult {
            sub_goal_id: "test".to_string(),
            result: empty_result.clone(),
            proven: true,
            nested_results: vec![
                SubGoalResult {
                    sub_goal_id: "nested1".to_string(),
                    result: empty_result.clone(),
                    proven: true,
                    nested_results: vec![],
                },
                SubGoalResult {
                    sub_goal_id: "nested2".to_string(),
                    result: empty_result.clone(),
                    proven: false,
                    nested_results: vec![],
                },
            ],
        };

        assert_eq!(sub_goal.proven_count(), 2); // self + nested1
        assert!(!sub_goal.all_proven()); // nested2 not proven
    }

    #[test]
    fn hierarchical_result_collect_hints() {
        let empty_result = ProofSearchResult {
            best_proof: Some(SynthesisResult {
                proof: "trivial".to_string(),
                tactics_used: vec!["simp".to_string(), "exact".to_string()],
                confidence: 0.8,
                attempts: 1,
                reasoning: None,
            }),
            steps: vec![],
            propagated_hints: vec![],
            policy: TacticPolicySnapshot { weights: vec![] },
        };

        let hier_result = HierarchicalSearchResult {
            main_result: empty_result.clone(),
            sub_goal_results: vec![SubGoalResult {
                sub_goal_id: "subgoal_a".to_string(),
                result: empty_result.clone(),
                proven: true,
                nested_results: vec![],
            }],
            decomposition_strategy: DecompositionStrategy::ConjunctionSplit,
            total_iterations: 3,
        };

        let hints = hier_result.collect_proven_hints();
        assert_eq!(hints.len(), 1);
        assert!(hints[0].contains("subgoal_a"));
        assert!(hints[0].contains("simp"));
    }

    #[test]
    fn decompose_forall_nat_with_strong_induction() {
        use dashprove_usl::ast::{ComparisonOp, Expr, Theorem};

        let property = Property::Theorem(Theorem {
            name: "strong_nat".to_string(),
            body: Expr::ForAll {
                var: "n".to_string(),
                ty: Some(dashprove_usl::ast::Type::Named("Nat".to_string())),
                body: Box::new(Expr::Compare(
                    Box::new(Expr::Var("n".to_string())),
                    ComparisonOp::Ge,
                    Box::new(Expr::Int(0)),
                )),
            },
        });

        let result = decompose_property_with_mode(&property, 0, InductionMode::Strong);
        assert_eq!(result.strategy, DecompositionStrategy::StrongInduction);
        assert_eq!(result.sub_goals.len(), 1);
        assert_eq!(result.sub_goals[0].id, "strong_nat_strong_ih");
        // Strong induction has no base case dependency - single sub-goal
        assert!(result.sub_goals[0].dependencies.is_empty());
    }

    #[test]
    fn decompose_forall_nat_with_well_founded() {
        use dashprove_usl::ast::{ComparisonOp, Expr, Theorem};

        let property = Property::Theorem(Theorem {
            name: "wf_nat".to_string(),
            body: Expr::ForAll {
                var: "n".to_string(),
                ty: Some(dashprove_usl::ast::Type::Named("Nat".to_string())),
                body: Box::new(Expr::Compare(
                    Box::new(Expr::Var("n".to_string())),
                    ComparisonOp::Ge,
                    Box::new(Expr::Int(0)),
                )),
            },
        });

        let result = decompose_property_with_mode(&property, 0, InductionMode::WellFounded);
        assert_eq!(result.strategy, DecompositionStrategy::WellFoundedInduction);
        assert_eq!(result.sub_goals.len(), 2);
        // First sub-goal is the inductive step
        assert_eq!(result.sub_goals[0].id, "wf_nat_wf_step");
        // Second sub-goal is the well-foundedness obligation
        assert_eq!(result.sub_goals[1].id, "wf_nat_wf_order");
    }

    #[test]
    fn decompose_disjunction_creates_case_analysis() {
        use dashprove_usl::ast::{Expr, Theorem};

        let property = Property::Theorem(Theorem {
            name: "disj_elim".to_string(),
            body: Expr::Or(
                Box::new(Expr::ForAll {
                    var: "x".to_string(),
                    ty: Some(dashprove_usl::ast::Type::Named("Nat".to_string())),
                    body: Box::new(Expr::Bool(true)),
                }),
                Box::new(Expr::ForAll {
                    var: "y".to_string(),
                    ty: Some(dashprove_usl::ast::Type::Named("Nat".to_string())),
                    body: Box::new(Expr::Bool(true)),
                }),
            ),
        });

        let result = decompose_property(&property, 0);
        assert_eq!(
            result.strategy,
            DecompositionStrategy::DisjunctionElimination
        );
        assert_eq!(result.sub_goals.len(), 2);
        assert_eq!(result.sub_goals[0].id, "disj_elim_case_left");
        assert_eq!(result.sub_goals[1].id, "disj_elim_case_right");
        // Both cases are independent
        assert!(result.sub_goals[0].dependencies.is_empty());
        assert!(result.sub_goals[1].dependencies.is_empty());
    }

    #[test]
    fn decompose_list_with_strong_induction() {
        use dashprove_usl::ast::{Expr, Theorem};

        let property = Property::Theorem(Theorem {
            name: "strong_list".to_string(),
            body: Expr::ForAll {
                var: "xs".to_string(),
                ty: Some(dashprove_usl::ast::Type::Named("List<Int>".to_string())),
                body: Box::new(Expr::Bool(true)),
            },
        });

        let result = decompose_property_with_mode(&property, 0, InductionMode::Strong);
        assert_eq!(result.strategy, DecompositionStrategy::StrongInduction);
        assert_eq!(result.sub_goals.len(), 1);
        assert_eq!(result.sub_goals[0].id, "strong_list_strong_struct");
    }

    #[test]
    fn decompose_list_with_well_founded_induction() {
        use dashprove_usl::ast::{Expr, Theorem};

        let property = Property::Theorem(Theorem {
            name: "wf_list".to_string(),
            body: Expr::ForAll {
                var: "xs".to_string(),
                ty: Some(dashprove_usl::ast::Type::Named("List<Int>".to_string())),
                body: Box::new(Expr::Bool(true)),
            },
        });

        let result = decompose_property_with_mode(&property, 0, InductionMode::WellFounded);
        assert_eq!(result.strategy, DecompositionStrategy::WellFoundedInduction);
        assert_eq!(result.sub_goals.len(), 2);
        assert_eq!(result.sub_goals[0].id, "wf_list_wf_size");
        assert_eq!(result.sub_goals[1].id, "wf_list_wf_measure");
    }

    #[test]
    fn induction_mode_default_is_simple() {
        assert_eq!(InductionMode::default(), InductionMode::Simple);
    }

    // ========================================================================
    // SITA Template Tests
    // ========================================================================

    #[test]
    fn template_registry_with_builtins_has_templates() {
        let registry = TemplateRegistry::with_builtins();
        assert!(!registry.is_empty());
        // Check that key templates are registered
        assert!(registry.get("assoc").is_some());
        assert!(registry.get("commut").is_some());
        assert!(registry.get("identity").is_some());
        assert!(registry.get("refl").is_some());
        assert!(registry.get("trans").is_some());
        assert!(registry.get("symm").is_some());
        assert!(registry.get("nat_ind").is_some());
        assert!(registry.get("list_ind").is_some());
        assert!(registry.get("distrib").is_some());
        assert_eq!(registry.len(), 9);
    }

    #[test]
    fn template_registry_find_by_category() {
        let registry = TemplateRegistry::with_builtins();

        let algebraic = registry.find_by_category(TemplateCategory::Algebraic);
        assert!(algebraic.len() >= 3); // assoc, commut, identity, distrib

        let order = registry.find_by_category(TemplateCategory::Order);
        assert!(order.len() >= 3); // refl, trans, symm

        let induction = registry.find_by_category(TemplateCategory::Induction);
        assert!(induction.len() >= 2); // nat_ind, list_ind
    }

    #[test]
    fn template_registry_search_by_keyword() {
        let registry = TemplateRegistry::with_builtins();

        let assoc_results = registry.search("associativity");
        assert_eq!(assoc_results.len(), 1);
        assert_eq!(assoc_results[0].id, "assoc");

        let ind_results = registry.search("induction");
        assert!(ind_results.len() >= 2); // nat_ind, list_ind
    }

    #[test]
    fn template_has_correct_parameters() {
        let registry = TemplateRegistry::with_builtins();

        let assoc = registry.get("assoc").unwrap();
        assert_eq!(assoc.params.len(), 2);
        assert_eq!(assoc.params[0].name, "T");
        assert_eq!(assoc.params[0].kind, TemplateParamKind::Type);
        assert_eq!(assoc.params[1].name, "op");
        assert!(matches!(
            assoc.params[1].kind,
            TemplateParamKind::Function { arity: 2 }
        ));

        let identity = registry.get("identity").unwrap();
        assert_eq!(identity.params.len(), 3);
        assert_eq!(identity.params[2].name, "e");
        assert_eq!(identity.params[2].kind, TemplateParamKind::Term);
    }

    #[test]
    fn template_matching_finds_reflexivity_pattern() {
        use dashprove_usl::ast::{Expr, Theorem, Type};

        let registry = TemplateRegistry::with_builtins();

        // Create a property that matches reflexivity: forall x: Int. less_eq(x, x)
        let property = Property::Theorem(Theorem {
            name: "int_reflexive".to_string(),
            body: Expr::ForAll {
                var: "x".to_string(),
                ty: Some(Type::Named("Int".to_string())),
                body: Box::new(Expr::App(
                    "less_eq".to_string(),
                    vec![Expr::Var("x".to_string()), Expr::Var("x".to_string())],
                )),
            },
        });

        let matches = registry.find_matches(&property);
        assert!(!matches.is_empty());
        // Reflexivity template should match with reasonable confidence
        let refl_match = matches.iter().find(|m| m.template_id == "refl");
        assert!(refl_match.is_some());
    }

    #[test]
    fn template_instantiation_substitutes_function() {
        #![allow(unused_imports)]
        use dashprove_usl::ast::{Expr, Theorem, Type};

        let registry = TemplateRegistry::with_builtins();
        let commut = registry.get("commut").unwrap();

        // Create bindings for commutativity of addition
        let mut bindings = HashMap::new();
        bindings.insert("T".to_string(), TemplateBinding::Type("Int".to_string()));
        bindings.insert(
            "op".to_string(),
            TemplateBinding::Function("add".to_string()),
        );

        let result = instantiate_template(commut, &bindings);
        assert!(result.is_ok());

        let instantiated = result.unwrap();
        if let Property::Theorem(t) = instantiated {
            // The body should now reference "add" instead of "op"
            let body_str = format!("{:?}", t.body);
            assert!(body_str.contains("add"));
        } else {
            panic!("Expected Theorem property");
        }
    }

    #[test]
    fn template_categories_are_distinct() {
        // Ensure all categories are different values
        let cats = [
            TemplateCategory::Algebraic,
            TemplateCategory::Order,
            TemplateCategory::Collection,
            TemplateCategory::Induction,
            TemplateCategory::Equivalence,
            TemplateCategory::Function,
            TemplateCategory::Recursive,
            TemplateCategory::Custom,
        ];

        for (i, cat1) in cats.iter().enumerate() {
            for (j, cat2) in cats.iter().enumerate() {
                if i != j {
                    assert_ne!(cat1, cat2);
                }
            }
        }
    }

    #[test]
    fn template_builder_pattern_works() {
        let template = ProofTemplate::new("test", "Test Template", TemplateCategory::Custom)
            .with_param(TemplateParam {
                name: "X".to_string(),
                kind: TemplateParamKind::Type,
                description: "Test type".to_string(),
                constraints: vec!["bounded".to_string()],
            })
            .with_tactics(vec!["auto".to_string(), "simp".to_string()])
            .with_hints(vec!["Try automation".to_string()])
            .with_prerequisites(vec!["lemma1".to_string()]);

        assert_eq!(template.id, "test");
        assert_eq!(template.name, "Test Template");
        assert_eq!(template.category, TemplateCategory::Custom);
        assert_eq!(template.params.len(), 1);
        assert_eq!(template.params[0].constraints.len(), 1);
        assert_eq!(template.tactics.len(), 2);
        assert_eq!(template.hints.len(), 1);
        assert_eq!(template.prerequisites.len(), 1);
    }

    #[test]
    fn template_registry_is_initially_empty() {
        let registry = TemplateRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn template_registry_register_and_get() {
        let mut registry = TemplateRegistry::new();

        let template = ProofTemplate::new("custom", "Custom Template", TemplateCategory::Custom);
        registry.register(template);

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);

        let retrieved = registry.get("custom");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Custom Template");
    }

    #[test]
    fn template_registry_list_ids() {
        let registry = TemplateRegistry::with_builtins();
        let ids = registry.template_ids();
        assert_eq!(ids.len(), 9);
        assert!(ids.contains(&"assoc"));
        assert!(ids.contains(&"commut"));
    }

    #[test]
    fn template_match_confidence_scales() {
        use dashprove_usl::ast::{Expr, Theorem};

        let registry = TemplateRegistry::with_builtins();

        // A property that doesn't match any template well
        let unrelated = Property::Theorem(Theorem {
            name: "unrelated".to_string(),
            body: Expr::Bool(true), // Too simple, won't match patterns
        });

        let matches = registry.find_matches(&unrelated);
        // Should have no high-confidence matches for a trivial property
        let high_confidence: Vec<_> = matches.iter().filter(|m| m.confidence > 0.8).collect();
        assert!(high_confidence.is_empty());
    }

    #[test]
    fn template_binding_variants() {
        use dashprove_usl::ast::Expr;

        // Test that all binding variants can be created
        let type_binding = TemplateBinding::Type("Int".to_string());
        let func_binding = TemplateBinding::Function("add".to_string());
        let pred_binding = TemplateBinding::Predicate("is_positive".to_string());
        let term_binding = TemplateBinding::Term(Expr::Int(42));
        let const_binding = TemplateBinding::Constant("zero".to_string());

        // Just verify they can be created and formatted
        assert!(format!("{:?}", type_binding).contains("Int"));
        assert!(format!("{:?}", func_binding).contains("add"));
        assert!(format!("{:?}", pred_binding).contains("is_positive"));
        assert!(format!("{:?}", term_binding).contains("42"));
        assert!(format!("{:?}", const_binding).contains("zero"));
    }
}
