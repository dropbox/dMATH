//! Lazy SMT verifier using CEGAR (Counterexample-Guided Abstraction Refinement).
//!
//! This module implements lazy ReLU splitting, where:
//! 1. Start with triangle relaxation (LP-feasible, fast but incomplete)
//! 2. When SAT, verify the solution against true ReLU semantics
//! 3. If any neuron violates y = max(0, x), add exact constraints for those neurons
//! 4. Re-solve and repeat until UNSAT (verified) or valid counterexample found

use crate::encoder::SmtResult;
use crate::model_parser::parse_model_to_map;
use crate::{Result, SmtError};
use gamma_core::{Bound, InformativeCounterexample, VerificationResult};
use std::collections::{HashMap, HashSet};
use z4_dpll::Executor;
use z4_frontend::{parse, Command};

/// Configuration for lazy SMT verification.
#[derive(Debug, Clone)]
pub struct LazyVerifierConfig {
    /// Maximum refinement iterations before giving up.
    pub max_iterations: usize,
    /// Big-M constant for exact ReLU encoding.
    pub big_m: f64,
    /// Tolerance for checking ReLU violations.
    pub relu_tolerance: f64,
    /// Timeout in milliseconds for each SMT query.
    pub timeout_ms: Option<u64>,
}

impl Default for LazyVerifierConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            big_m: 1e6,
            relu_tolerance: 1e-6,
            timeout_ms: Some(60_000),
        }
    }
}

/// Lazy SMT verifier using CEGAR for ReLU refinement.
///
/// Instead of encoding all ReLU constraints upfront, this verifier:
/// 1. Uses triangle relaxation initially (fast, LP-feasible)
/// 2. Checks candidate counterexamples against true ReLU semantics
/// 3. Refines only the violated ReLU neurons
///
/// This can be orders of magnitude faster for networks with many ReLU neurons.
#[derive(Debug)]
pub struct LazyVerifier {
    config: LazyVerifierConfig,
}

impl Default for LazyVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a ReLU neuron for lazy encoding.
#[derive(Debug, Clone)]
pub struct ReluNeuron {
    /// Layer index (0-based, hidden layers only).
    pub layer_idx: usize,
    /// Neuron index within the layer.
    pub neuron_idx: usize,
    /// Input variable name (pre-ReLU).
    pub input_var: String,
    /// Output variable name (post-ReLU).
    pub output_var: String,
    /// Pre-computed bounds on the input.
    pub bounds: Bound,
}

/// State for CEGAR refinement loop.
struct CegarState {
    /// Input variable names.
    input_vars: Vec<String>,
    /// Output variable names.
    output_vars: Vec<String>,
    /// All ReLU neurons with their metadata.
    relu_neurons: Vec<ReluNeuron>,
    /// Mapping for fast (layer_idx, neuron_idx) -> relu_neurons index.
    relu_index: HashMap<(usize, usize), usize>,
    /// Set of refined neuron indices (layer_idx, neuron_idx).
    refined: HashSet<(usize, usize)>,
    /// Current iteration.
    iteration: usize,
}

struct SmtSession {
    executor: Executor,
}

impl SmtSession {
    fn from_base_formula(base_formula: &str) -> Result<Self> {
        let commands = parse(base_formula)
            .map_err(|e| SmtError::SolverError(format!("parse error: {}", e)))?;

        let mut executor = Executor::new();
        for cmd in &commands {
            if matches!(cmd, Command::CheckSat | Command::GetModel) {
                return Err(SmtError::EncodingError(
                    "base formula must not include (check-sat) or (get-model)".to_string(),
                ));
            }
            if let Err(e) = executor.execute(cmd) {
                return Err(SmtError::SolverError(format!(
                    "command execution failed: {}",
                    e
                )));
            }
        }

        Ok(Self { executor })
    }

    fn check_sat_and_get_model(&mut self) -> Result<(SmtResult, Option<String>)> {
        let output = self
            .executor
            .execute(&Command::CheckSat)
            .map_err(|e| SmtError::SolverError(format!("check-sat failed: {}", e)))?;

        let Some(output) = output else {
            return Ok((SmtResult::Unknown, None));
        };

        match output.as_str() {
            "sat" => {
                let model = self.executor.execute(&Command::GetModel).ok().flatten();
                Ok((
                    SmtResult::Sat {
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                    },
                    model,
                ))
            }
            "unsat" => Ok((SmtResult::Unsat { proof: None }, None)),
            _ => Ok((SmtResult::Unknown, None)),
        }
    }

    fn execute_snippet(&mut self, snippet: &str) -> Result<()> {
        let commands =
            parse(snippet).map_err(|e| SmtError::SolverError(format!("parse error: {}", e)))?;
        for cmd in &commands {
            if let Err(e) = self.executor.execute(cmd) {
                return Err(SmtError::SolverError(format!(
                    "command execution failed: {}",
                    e
                )));
            }
        }
        Ok(())
    }
}

impl LazyVerifier {
    /// Create a new lazy verifier with default configuration.
    pub fn new() -> Self {
        Self {
            config: LazyVerifierConfig::default(),
        }
    }

    /// Create a verifier with custom configuration.
    pub fn with_config(config: LazyVerifierConfig) -> Self {
        Self { config }
    }

    /// Verify a feedforward network using lazy ReLU refinement.
    ///
    /// # Arguments
    /// * `weights` - Weight matrices for each layer (row-major)
    /// * `biases` - Bias vectors for each layer
    /// * `layer_dims` - Dimensions: [input, hidden1, ..., output]
    /// * `input_bounds` - Input region bounds
    /// * `output_bounds` - Required output bounds (property)
    /// * `intermediate_bounds` - Pre-computed bounds on hidden neurons
    pub fn verify_feedforward(
        &self,
        weights: &[Vec<f64>],
        biases: &[Vec<f64>],
        layer_dims: &[usize],
        input_bounds: &[Bound],
        output_bounds: &[Bound],
        intermediate_bounds: &[Vec<Bound>],
    ) -> Result<VerificationResult> {
        // Initialize CEGAR state with relaxation encoding
        let (base_formula, state) = self.encode_with_relaxation(
            weights,
            biases,
            layer_dims,
            input_bounds,
            output_bounds,
            intermediate_bounds,
        )?;

        let total_relus = state.relu_neurons.len();
        let uncertain_relus = state
            .relu_neurons
            .iter()
            .filter(|n| n.bounds.lower < 0.0 && n.bounds.upper > 0.0)
            .count();

        tracing::info!(
            "Lazy verification: {} total ReLUs, {} uncertain (will be refined on-demand)",
            total_relus,
            uncertain_relus
        );

        // Run CEGAR loop
        self.cegar_loop(base_formula, state, output_bounds)
    }

    /// Encode network with triangle relaxation for initial abstraction.
    fn encode_with_relaxation(
        &self,
        weights: &[Vec<f64>],
        biases: &[Vec<f64>],
        layer_dims: &[usize],
        input_bounds: &[Bound],
        output_bounds: &[Bound],
        intermediate_bounds: &[Vec<Bound>],
    ) -> Result<(String, CegarState)> {
        let num_layers = weights.len();
        if num_layers == 0 {
            return Err(SmtError::EncodingError("empty network".to_string()));
        }

        let input_dim = layer_dims[0];

        // We'll build the formula manually to track variable names
        let mut declarations = Vec::new();
        let mut constraints = Vec::new();
        let mut var_counter = 0;
        let mut relu_neurons = Vec::new();
        let mut relu_index = HashMap::new();

        // Helper to generate unique variable names
        let mut fresh_var = |prefix: &str| {
            let name = format!("{}_{}", prefix, var_counter);
            var_counter += 1;
            name
        };

        // Create input variables
        let mut input_vars = Vec::with_capacity(input_dim);
        for _ in 0..input_dim {
            let var = fresh_var("x");
            declarations.push(format!("(declare-const {} Real)", var));
            input_vars.push(var);
        }

        // Encode input bounds
        for (var, bound) in input_vars.iter().zip(input_bounds.iter()) {
            constraints.push(format!("(assert (>= {} {}))", var, bound.lower));
            constraints.push(format!("(assert (<= {} {}))", var, bound.upper));
        }

        // Process each layer
        let mut current_vars = input_vars.clone();
        for layer_idx in 0..num_layers {
            let in_dim = current_vars.len();
            let out_dim = layer_dims[layer_idx + 1];

            // Linear transformation
            let mut linear_out = Vec::with_capacity(out_dim);
            for i in 0..out_dim {
                let out_var = fresh_var("lin");
                declarations.push(format!("(declare-const {} Real)", out_var));

                // Build linear combination
                let mut terms = Vec::new();
                for (j, x_var) in current_vars.iter().enumerate() {
                    let w = weights[layer_idx][i * in_dim + j];
                    if w.abs() > 1e-15 {
                        terms.push(format!("(* {} {})", w, x_var));
                    }
                }
                let b = biases[layer_idx][i];
                if b.abs() > 1e-15 {
                    terms.push(format!("{}", b));
                }

                let rhs = if terms.is_empty() {
                    "0.0".to_string()
                } else if terms.len() == 1 {
                    terms[0].clone()
                } else {
                    format!("(+ {})", terms.join(" "))
                };

                constraints.push(format!("(assert (= {} {}))", out_var, rhs));
                linear_out.push(out_var);
            }

            // Apply ReLU to all layers except the last
            if layer_idx < num_layers - 1 {
                if layer_idx >= intermediate_bounds.len() {
                    return Err(SmtError::EncodingError(format!(
                        "missing intermediate bounds for layer {}",
                        layer_idx
                    )));
                }

                let bounds_slice = &intermediate_bounds[layer_idx];
                let mut relu_out = Vec::with_capacity(linear_out.len());

                for (neuron_idx, (x_var, bound)) in
                    linear_out.iter().zip(bounds_slice.iter()).enumerate()
                {
                    let y_var = fresh_var("relu");
                    declarations.push(format!("(declare-const {} Real)", y_var));

                    let l = bound.lower as f64;
                    let u = bound.upper as f64;

                    if l >= 0.0 {
                        // Definitely active: y = x
                        constraints.push(format!("(assert (= {} {}))", y_var, x_var));
                    } else if u <= 0.0 {
                        // Definitely inactive: y = 0
                        constraints.push(format!("(assert (= {} 0.0))", y_var));
                    } else {
                        // Uncertain: use triangle relaxation
                        // y >= 0
                        constraints.push(format!("(assert (>= {} 0.0))", y_var));
                        // y >= x
                        constraints.push(format!("(assert (>= {} {}))", y_var, x_var));
                        // y <= u*(x-l)/(u-l)
                        let slope = u / (u - l);
                        let intercept = -u * l / (u - l);
                        constraints.push(format!(
                            "(assert (<= {} (+ (* {} {}) {})))",
                            y_var, slope, x_var, intercept
                        ));

                        // Track this neuron for potential refinement
                        let idx = relu_neurons.len();
                        relu_neurons.push(ReluNeuron {
                            layer_idx,
                            neuron_idx,
                            input_var: x_var.clone(),
                            output_var: y_var.clone(),
                            bounds: *bound,
                        });
                        relu_index.insert((layer_idx, neuron_idx), idx);
                    }

                    relu_out.push(y_var);
                }

                current_vars = relu_out;
            } else {
                current_vars = linear_out;
            }
        }

        let output_vars = current_vars.clone();

        // Encode property negation
        let mut disjuncts = Vec::new();
        for (var, bound) in output_vars.iter().zip(output_bounds.iter()) {
            disjuncts.push(format!("(< {} {})", var, bound.lower));
            disjuncts.push(format!("(> {} {})", var, bound.upper));
        }
        if disjuncts.len() == 1 {
            constraints.push(format!("(assert {})", disjuncts[0]));
        } else {
            constraints.push(format!("(assert (or {}))", disjuncts.join(" ")));
        }

        // Build base formula (no query commands; those are executed incrementally).
        let mut formula = String::new();
        formula.push_str("(set-logic QF_LRA)\n");
        for decl in &declarations {
            formula.push_str(decl);
            formula.push('\n');
        }
        for constr in &constraints {
            formula.push_str(constr);
            formula.push('\n');
        }

        let state = CegarState {
            input_vars,
            output_vars,
            relu_neurons,
            relu_index,
            refined: HashSet::new(),
            iteration: 0,
        };

        Ok((formula, state))
    }

    /// Run the CEGAR refinement loop.
    fn cegar_loop(
        &self,
        base_formula: String,
        mut state: CegarState,
        output_bounds: &[Bound],
    ) -> Result<VerificationResult> {
        let mut session = SmtSession::from_base_formula(&base_formula)?;

        loop {
            state.iteration += 1;
            if state.iteration > self.config.max_iterations {
                tracing::warn!(
                    "Lazy verification reached max iterations ({})",
                    self.config.max_iterations
                );
                return Ok(VerificationResult::Timeout {
                    partial_bounds: Some(output_bounds.to_vec()),
                });
            }

            tracing::debug!(
                "CEGAR iteration {}: {} refined / {} total",
                state.iteration,
                state.refined.len(),
                state.relu_neurons.len()
            );

            // Solve current constraints
            let (result, model_str) = session.check_sat_and_get_model()?;

            match result {
                SmtResult::Unsat { proof: proof_str } => {
                    // Abstraction is UNSAT -> property holds
                    tracing::info!(
                        "Lazy verification: UNSAT after {} iterations, {} refinements",
                        state.iteration,
                        state.refined.len()
                    );
                    // Convert proof string to VerificationProof if available
                    let proof =
                        proof_str.map(|s| Box::new(gamma_core::VerificationProof::alethe(s)));
                    return Ok(VerificationResult::Verified {
                        output_bounds: output_bounds.to_vec(),
                        proof,
                    });
                }
                SmtResult::Sat { .. } => {
                    let Some(model_str) = model_str else {
                        return Ok(VerificationResult::Unknown {
                            bounds: output_bounds.to_vec(),
                            reason: format!(
                                "SMT solver returned sat but no model at iteration {}",
                                state.iteration
                            ),
                        });
                    };

                    let Some(model) = parse_model_to_map(&model_str) else {
                        return Ok(VerificationResult::Unknown {
                            bounds: output_bounds.to_vec(),
                            reason: format!(
                                "failed to parse SMT model at iteration {}",
                                state.iteration
                            ),
                        });
                    };

                    let inputs = state
                        .input_vars
                        .iter()
                        .map(|name| model.get(name).copied())
                        .collect::<Option<Vec<_>>>()
                        .unwrap_or_default();
                    let outputs = state
                        .output_vars
                        .iter()
                        .map(|name| model.get(name).copied())
                        .collect::<Option<Vec<_>>>()
                        .unwrap_or_default();

                    // Check if this is a true counterexample
                    let violations = self.find_relu_violations(&state, &model);

                    if violations.is_empty() {
                        // Valid counterexample found
                        tracing::info!(
                            "Lazy verification: valid counterexample after {} iterations",
                            state.iteration
                        );
                        let input_f32: Vec<f32> = inputs.iter().map(|&x| x as f32).collect();
                        let output_f32: Vec<f32> = outputs.iter().map(|&x| x as f32).collect();
                        let details = InformativeCounterexample::new(
                            input_f32.clone(),
                            output_f32.clone(),
                            Some(output_bounds),
                        );
                        return Ok(VerificationResult::Violated {
                            counterexample: input_f32,
                            output: output_f32,
                            details: Some(Box::new(details)),
                        });
                    }

                    // Refine violated neurons
                    tracing::debug!("Found {} ReLU violations, refining", violations.len());

                    self.refine_session(&mut session, &state, &violations)?;

                    for (layer_idx, neuron_idx) in &violations {
                        state.refined.insert((*layer_idx, *neuron_idx));
                    }
                }
                SmtResult::Unknown | SmtResult::Timeout => {
                    return Ok(VerificationResult::Unknown {
                        bounds: output_bounds.to_vec(),
                        reason: format!(
                            "SMT solver returned {:?} after {} iterations",
                            result, state.iteration
                        ),
                    });
                }
            }
        }
    }

    /// Find ReLU neurons that violate y = max(0, x) in the current model.
    fn find_relu_violations(
        &self,
        state: &CegarState,
        model: &HashMap<String, f64>,
    ) -> Vec<(usize, usize)> {
        let mut violations = Vec::new();
        let tol = self.config.relu_tolerance;

        for neuron in &state.relu_neurons {
            // Skip already refined neurons
            if state
                .refined
                .contains(&(neuron.layer_idx, neuron.neuron_idx))
            {
                continue;
            }

            let Some(&x) = model.get(&neuron.input_var) else {
                continue;
            };
            let Some(&y) = model.get(&neuron.output_var) else {
                continue;
            };

            let expected_y = x.max(0.0);
            if (y - expected_y).abs() > tol {
                violations.push((neuron.layer_idx, neuron.neuron_idx));
            }
        }

        violations
    }

    fn refine_session(
        &self,
        session: &mut SmtSession,
        state: &CegarState,
        violations: &[(usize, usize)],
    ) -> Result<()> {
        for &(layer_idx, neuron_idx) in violations {
            let Some(&idx) = state.relu_index.get(&(layer_idx, neuron_idx)) else {
                continue;
            };
            let neuron = &state.relu_neurons[idx];

            let x = &neuron.input_var;
            let y = &neuron.output_var;

            session.execute_snippet(&format!(
                "(assert (or (and (<= {} 0.0) (= {} 0.0)) (and (>= {} 0.0) (= {} {}))))\n",
                x, y, x, y, x
            ))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== LazyVerifierConfig tests ====================

    #[test]
    fn test_lazy_verifier_config_default() {
        let config = LazyVerifierConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.big_m, 1e6);
        assert_eq!(config.relu_tolerance, 1e-6);
        assert_eq!(config.timeout_ms, Some(60_000));
    }

    #[test]
    fn test_lazy_verifier_config_clone() {
        let config = LazyVerifierConfig {
            max_iterations: 50,
            big_m: 1e5,
            relu_tolerance: 1e-8,
            timeout_ms: Some(30_000),
        };
        let cloned = config.clone();
        assert_eq!(cloned.max_iterations, 50);
        assert_eq!(cloned.big_m, 1e5);
        assert_eq!(cloned.relu_tolerance, 1e-8);
        assert_eq!(cloned.timeout_ms, Some(30_000));
    }

    #[test]
    fn test_lazy_verifier_config_debug() {
        let config = LazyVerifierConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("LazyVerifierConfig"));
        assert!(debug_str.contains("max_iterations"));
    }

    // ==================== LazyVerifier construction tests ====================

    #[test]
    fn test_lazy_verifier_new() {
        let verifier = LazyVerifier::new();
        assert_eq!(verifier.config.max_iterations, 100);
        assert_eq!(verifier.config.big_m, 1e6);
    }

    #[test]
    fn test_lazy_verifier_default() {
        let verifier = LazyVerifier::default();
        assert_eq!(verifier.config.max_iterations, 100);
    }

    #[test]
    fn test_lazy_verifier_with_config() {
        let config = LazyVerifierConfig {
            max_iterations: 25,
            big_m: 500.0,
            relu_tolerance: 1e-4,
            timeout_ms: None,
        };
        let verifier = LazyVerifier::with_config(config);
        assert_eq!(verifier.config.max_iterations, 25);
        assert_eq!(verifier.config.big_m, 500.0);
        assert_eq!(verifier.config.relu_tolerance, 1e-4);
        assert!(verifier.config.timeout_ms.is_none());
    }

    #[test]
    fn test_lazy_verifier_debug() {
        let verifier = LazyVerifier::new();
        let debug_str = format!("{:?}", verifier);
        assert!(debug_str.contains("LazyVerifier"));
    }

    // ==================== ReluNeuron tests ====================

    #[test]
    fn test_relu_neuron_fields() {
        let neuron = ReluNeuron {
            layer_idx: 2,
            neuron_idx: 5,
            input_var: "x_pre_3".to_string(),
            output_var: "x_post_3".to_string(),
            bounds: Bound::new(-1.0, 2.0),
        };
        assert_eq!(neuron.layer_idx, 2);
        assert_eq!(neuron.neuron_idx, 5);
        assert_eq!(neuron.input_var, "x_pre_3");
        assert_eq!(neuron.output_var, "x_post_3");
        assert_eq!(neuron.bounds.lower, -1.0);
        assert_eq!(neuron.bounds.upper, 2.0);
    }

    #[test]
    fn test_relu_neuron_clone() {
        let neuron = ReluNeuron {
            layer_idx: 1,
            neuron_idx: 3,
            input_var: "in".to_string(),
            output_var: "out".to_string(),
            bounds: Bound::new(-0.5, 0.5),
        };
        let cloned = neuron.clone();
        assert_eq!(cloned.layer_idx, neuron.layer_idx);
        assert_eq!(cloned.neuron_idx, neuron.neuron_idx);
        assert_eq!(cloned.input_var, neuron.input_var);
        assert_eq!(cloned.output_var, neuron.output_var);
    }

    #[test]
    fn test_relu_neuron_debug() {
        let neuron = ReluNeuron {
            layer_idx: 0,
            neuron_idx: 0,
            input_var: "x".to_string(),
            output_var: "y".to_string(),
            bounds: Bound::new(0.0, 1.0),
        };
        let debug_str = format!("{:?}", neuron);
        assert!(debug_str.contains("ReluNeuron"));
        assert!(debug_str.contains("layer_idx"));
    }

    // ==================== SmtSession tests ====================

    #[test]
    fn test_smt_session_from_valid_formula() {
        let formula = "(set-logic QF_LRA)\n(declare-const x Real)\n(assert (> x 0))\n";
        let session = SmtSession::from_base_formula(formula);
        assert!(session.is_ok());
    }

    #[test]
    fn test_smt_session_rejects_check_sat_in_base() {
        let formula = "(set-logic QF_LRA)\n(declare-const x Real)\n(check-sat)\n";
        let result = SmtSession::from_base_formula(formula);
        assert!(result.is_err());
        match result {
            Err(e) => assert!(e.to_string().contains("check-sat")),
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    fn test_smt_session_rejects_get_model_in_base() {
        let formula = "(set-logic QF_LRA)\n(declare-const x Real)\n(get-model)\n";
        let result = SmtSession::from_base_formula(formula);
        assert!(result.is_err());
        match result {
            Err(e) => assert!(e.to_string().contains("get-model")),
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    fn test_smt_session_parse_error() {
        let formula = "(set-logic QF_LRA\n(declare-const x Real)"; // missing )
        let result = SmtSession::from_base_formula(formula);
        assert!(result.is_err());
        match result {
            Err(e) => assert!(e.to_string().contains("parse error")),
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    fn test_smt_session_check_sat_unsat() {
        let formula =
            "(set-logic QF_LRA)\n(declare-const x Real)\n(assert (> x 0))\n(assert (< x 0))\n";
        let mut session = SmtSession::from_base_formula(formula).unwrap();
        let (result, model) = session.check_sat_and_get_model().unwrap();
        assert!(matches!(result, SmtResult::Unsat { .. }));
        assert!(model.is_none());
    }

    #[test]
    fn test_smt_session_check_sat_sat() {
        let formula =
            "(set-logic QF_LRA)\n(declare-const x Real)\n(assert (> x 0))\n(assert (< x 10))\n";
        let mut session = SmtSession::from_base_formula(formula).unwrap();
        let (result, model) = session.check_sat_and_get_model().unwrap();
        assert!(matches!(result, SmtResult::Sat { .. }));
        assert!(model.is_some());
    }

    #[test]
    fn test_smt_session_execute_snippet() {
        let formula = "(set-logic QF_LRA)\n(declare-const x Real)\n";
        let mut session = SmtSession::from_base_formula(formula).unwrap();
        let result = session.execute_snippet("(assert (> x 5))");
        assert!(result.is_ok());
    }

    #[test]
    fn test_smt_session_execute_snippet_parse_error() {
        let formula = "(set-logic QF_LRA)\n(declare-const x Real)\n";
        let mut session = SmtSession::from_base_formula(formula).unwrap();
        let result = session.execute_snippet("(assert (> x 5)"); // missing )
        assert!(result.is_err());
    }

    // ==================== encode_with_relaxation tests ====================

    #[test]
    fn test_encode_with_relaxation_empty_network() {
        let verifier = LazyVerifier::new();
        let result = verifier.verify_feedforward(
            &[],  // empty weights
            &[],  // empty biases
            &[1], // just input dim
            &[Bound::new(0.0, 1.0)],
            &[Bound::new(0.0, 1.0)],
            &[],
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty network"));
    }

    #[test]
    fn test_encode_with_relaxation_missing_intermediate_bounds() {
        let verifier = LazyVerifier::new();
        // 2-layer network with hidden layer but no intermediate bounds
        let weights = vec![vec![1.0], vec![1.0]];
        let biases = vec![vec![0.0], vec![0.0]];
        let layer_dims = vec![1, 1, 1];

        let result = verifier.verify_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &[Bound::new(0.0, 1.0)],
            &[Bound::new(0.0, 1.0)],
            &[], // missing intermediate bounds for layer 0
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("missing intermediate bounds"));
    }

    #[test]
    fn test_encode_with_relaxation_definitely_active_relu() {
        // ReLU where input bounds are >= 0 (definitely active)
        let verifier = LazyVerifier::new();
        let weights = vec![vec![1.0], vec![1.0]];
        let biases = vec![vec![0.0], vec![0.0]];
        let layer_dims = vec![1, 1, 1];

        let input_bounds = vec![Bound::new(1.0, 2.0)]; // always positive
        let output_bounds = vec![Bound::new(0.5, 2.5)];
        let intermediate_bounds = vec![vec![Bound::new(1.0, 2.0)]]; // >= 0, definitely active

        let result = verifier
            .verify_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
            )
            .unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_encode_with_relaxation_definitely_inactive_relu() {
        // ReLU where input bounds are <= 0 (definitely inactive)
        let verifier = LazyVerifier::new();
        let weights = vec![vec![1.0], vec![1.0]];
        let biases = vec![vec![-5.0], vec![0.0]]; // bias shifts input to negative
        let layer_dims = vec![1, 1, 1];

        let input_bounds = vec![Bound::new(0.0, 1.0)];
        // After linear: -5 to -4, definitely negative
        // After ReLU: always 0
        let output_bounds = vec![Bound::new(-0.5, 0.5)];
        let intermediate_bounds = vec![vec![Bound::new(-5.0, -4.0)]]; // <= 0, definitely inactive

        let result = verifier
            .verify_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
            )
            .unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    // ==================== find_relu_violations tests ====================

    #[test]
    fn test_find_relu_violations_no_violations() {
        let verifier = LazyVerifier::new();
        let state = CegarState {
            input_vars: vec!["x_0".to_string()],
            output_vars: vec!["y_0".to_string()],
            relu_neurons: vec![ReluNeuron {
                layer_idx: 0,
                neuron_idx: 0,
                input_var: "pre_0".to_string(),
                output_var: "post_0".to_string(),
                bounds: Bound::new(-1.0, 1.0),
            }],
            relu_index: [(0, 0)].into_iter().map(|k| (k, 0)).collect(),
            refined: HashSet::new(),
            iteration: 1,
        };

        // Model where ReLU is satisfied: x = 0.5, y = 0.5 (max(0, 0.5) = 0.5)
        let mut model = HashMap::new();
        model.insert("pre_0".to_string(), 0.5);
        model.insert("post_0".to_string(), 0.5);

        let violations = verifier.find_relu_violations(&state, &model);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_find_relu_violations_with_violation() {
        let verifier = LazyVerifier::new();
        let state = CegarState {
            input_vars: vec!["x_0".to_string()],
            output_vars: vec!["y_0".to_string()],
            relu_neurons: vec![ReluNeuron {
                layer_idx: 0,
                neuron_idx: 0,
                input_var: "pre_0".to_string(),
                output_var: "post_0".to_string(),
                bounds: Bound::new(-1.0, 1.0),
            }],
            relu_index: [(0, 0)].into_iter().map(|k| (k, 0)).collect(),
            refined: HashSet::new(),
            iteration: 1,
        };

        // Model where ReLU is violated: x = -0.5 but y = 0.3 (should be 0)
        let mut model = HashMap::new();
        model.insert("pre_0".to_string(), -0.5);
        model.insert("post_0".to_string(), 0.3); // violation: max(0, -0.5) = 0 != 0.3

        let violations = verifier.find_relu_violations(&state, &model);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0], (0, 0));
    }

    #[test]
    fn test_find_relu_violations_skips_refined() {
        let verifier = LazyVerifier::new();
        let mut refined = HashSet::new();
        refined.insert((0, 0)); // Mark as already refined

        let state = CegarState {
            input_vars: vec!["x_0".to_string()],
            output_vars: vec!["y_0".to_string()],
            relu_neurons: vec![ReluNeuron {
                layer_idx: 0,
                neuron_idx: 0,
                input_var: "pre_0".to_string(),
                output_var: "post_0".to_string(),
                bounds: Bound::new(-1.0, 1.0),
            }],
            relu_index: [(0, 0)].into_iter().map(|k| (k, 0)).collect(),
            refined,
            iteration: 2,
        };

        // Model with violation - but should be skipped because already refined
        let mut model = HashMap::new();
        model.insert("pre_0".to_string(), -0.5);
        model.insert("post_0".to_string(), 0.3);

        let violations = verifier.find_relu_violations(&state, &model);
        assert!(violations.is_empty()); // Skipped because refined
    }

    #[test]
    fn test_find_relu_violations_missing_vars_in_model() {
        let verifier = LazyVerifier::new();
        let state = CegarState {
            input_vars: vec!["x_0".to_string()],
            output_vars: vec!["y_0".to_string()],
            relu_neurons: vec![ReluNeuron {
                layer_idx: 0,
                neuron_idx: 0,
                input_var: "pre_0".to_string(),
                output_var: "post_0".to_string(),
                bounds: Bound::new(-1.0, 1.0),
            }],
            relu_index: [(0, 0)].into_iter().map(|k| (k, 0)).collect(),
            refined: HashSet::new(),
            iteration: 1,
        };

        // Model missing variables - should skip gracefully
        let model = HashMap::new();

        let violations = verifier.find_relu_violations(&state, &model);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_find_relu_violations_within_tolerance() {
        let config = LazyVerifierConfig {
            relu_tolerance: 0.1, // Large tolerance
            ..Default::default()
        };
        let verifier = LazyVerifier::with_config(config);

        let state = CegarState {
            input_vars: vec!["x_0".to_string()],
            output_vars: vec!["y_0".to_string()],
            relu_neurons: vec![ReluNeuron {
                layer_idx: 0,
                neuron_idx: 0,
                input_var: "pre_0".to_string(),
                output_var: "post_0".to_string(),
                bounds: Bound::new(-1.0, 1.0),
            }],
            relu_index: [(0, 0)].into_iter().map(|k| (k, 0)).collect(),
            refined: HashSet::new(),
            iteration: 1,
        };

        // Model with small violation within tolerance
        let mut model = HashMap::new();
        model.insert("pre_0".to_string(), 0.5);
        model.insert("post_0".to_string(), 0.55); // 0.05 error, within 0.1 tolerance

        let violations = verifier.find_relu_violations(&state, &model);
        assert!(violations.is_empty());
    }

    // ==================== max_iterations tests ====================

    #[test]
    fn test_lazy_verifier_max_iterations_timeout() {
        let config = LazyVerifierConfig {
            max_iterations: 1, // Very low limit
            ..Default::default()
        };
        let verifier = LazyVerifier::with_config(config);

        // Network that would need refinement
        let weights = vec![vec![1.0], vec![1.0]];
        let biases = vec![vec![0.0], vec![0.0]];
        let layer_dims = vec![1, 1, 1];

        let input_bounds = vec![Bound::new(-1.0, 1.0)];
        let output_bounds = vec![Bound::new(-0.5, 1.5)];
        let intermediate_bounds = vec![vec![Bound::new(-1.0, 1.0)]];

        let result = verifier
            .verify_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
            )
            .unwrap();

        // Should either verify, find counterexample, or timeout - not error
        assert!(
            matches!(
                result,
                VerificationResult::Verified { .. }
                    | VerificationResult::Violated { .. }
                    | VerificationResult::Timeout { .. }
            ),
            "Unexpected result: {:?}",
            result
        );
    }

    // ==================== Integration tests (existing + new) ====================

    #[test]
    fn test_lazy_verifier_simple() {
        let verifier = LazyVerifier::new();

        // Simple network: y = ReLU(x + 1)
        // Input: x in [-2, 2]
        // After linear: z = x + 1, z in [-1, 3]
        // After ReLU: y in [0, 3]
        let weights = vec![vec![1.0], vec![1.0]];
        let biases = vec![vec![1.0], vec![0.0]];
        let layer_dims = vec![1, 1, 1];

        let input_bounds = vec![Bound::new(-2.0, 2.0)];
        let output_bounds = vec![Bound::new(-0.5, 3.5)];
        let intermediate_bounds = vec![vec![Bound::new(-1.0, 3.0)]];

        let result = verifier
            .verify_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
            )
            .unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_lazy_verifier_violated() {
        let verifier = LazyVerifier::new();

        // Same network but tighter property
        let weights = vec![vec![1.0], vec![1.0]];
        let biases = vec![vec![1.0], vec![0.0]];
        let layer_dims = vec![1, 1, 1];

        let input_bounds = vec![Bound::new(-2.0, 2.0)];
        // Property: y in [1, 2] - violated when x in [-2, -1] gives y = 0
        let output_bounds = vec![Bound::new(1.0, 2.0)];
        let intermediate_bounds = vec![vec![Bound::new(-1.0, 3.0)]];

        let result = verifier
            .verify_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
            )
            .unwrap();

        // Should find a counterexample
        assert!(
            matches!(result, VerificationResult::Violated { .. }),
            "Expected Violated, got {:?}",
            result
        );
    }

    #[test]
    fn test_lazy_verifier_uncertain_relu() {
        let verifier = LazyVerifier::new();

        // Network with uncertain ReLU that requires refinement
        // y = ReLU(2*x - 1)
        // Input: x in [0, 1]
        // After linear: z = 2*x - 1, z in [-1, 1]
        // After ReLU: y = max(0, 2*x - 1), y in [0, 1]
        let weights = vec![vec![2.0], vec![1.0]];
        let biases = vec![vec![-1.0], vec![0.0]];
        let layer_dims = vec![1, 1, 1];

        let input_bounds = vec![Bound::new(0.0, 1.0)];
        let output_bounds = vec![Bound::new(-0.5, 1.5)];
        let intermediate_bounds = vec![vec![Bound::new(-1.0, 1.0)]];

        let result = verifier
            .verify_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
            )
            .unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_lazy_verifier_linear_only_network() {
        // Single layer network with no ReLU
        let verifier = LazyVerifier::new();
        let weights = vec![vec![2.0, 3.0]]; // 1 output, 2 inputs
        let biases = vec![vec![1.0]];
        let layer_dims = vec![2, 1];

        let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
        // y = 2*x1 + 3*x2 + 1 in [1, 6]
        let output_bounds = vec![Bound::new(0.5, 6.5)];
        let intermediate_bounds: Vec<Vec<Bound>> = vec![];

        let result = verifier
            .verify_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
            )
            .unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_lazy_verifier_multi_input_uncertain_relu() {
        let verifier = LazyVerifier::new();

        // 2-input network with uncertain ReLU
        let weights = vec![
            vec![1.0, -1.0], // hidden: h = x1 - x2
            vec![1.0],       // output: y = relu(h)
        ];
        let biases = vec![vec![0.0], vec![0.0]];
        let layer_dims = vec![2, 1, 1];

        let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
        // h = x1 - x2 in [-1, 1], uncertain
        // y = relu(h) in [0, 1]
        let output_bounds = vec![Bound::new(-0.5, 1.5)];
        let intermediate_bounds = vec![vec![Bound::new(-1.0, 1.0)]];

        let result = verifier
            .verify_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
            )
            .unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_lazy_verifier_zero_weights_handled() {
        // Network with zero weights (should be skipped in encoding)
        let verifier = LazyVerifier::new();
        let weights = vec![vec![0.0, 1.0], vec![1.0]];
        let biases = vec![vec![0.5], vec![0.0]];
        let layer_dims = vec![2, 1, 1];

        let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
        // h = 0*x1 + 1*x2 + 0.5 = x2 + 0.5 in [0.5, 1.5]
        // y = relu(h) = h (all positive)
        let output_bounds = vec![Bound::new(0.0, 2.0)];
        let intermediate_bounds = vec![vec![Bound::new(0.5, 1.5)]];

        let result = verifier
            .verify_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
            )
            .unwrap();

        assert!(matches!(result, VerificationResult::Verified { .. }));
    }

    #[test]
    fn test_lazy_verifier_counterexample_has_values() {
        let verifier = LazyVerifier::new();

        let weights = vec![vec![1.0], vec![1.0]];
        let biases = vec![vec![0.0], vec![0.0]];
        let layer_dims = vec![1, 1, 1];

        let input_bounds = vec![Bound::new(-1.0, 1.0)];
        // Property that is violated: y must be > 0.5 always
        // But when x <= 0, y = relu(x) = 0 < 0.5
        let output_bounds = vec![Bound::new(0.5, 2.0)];
        let intermediate_bounds = vec![vec![Bound::new(-1.0, 1.0)]];

        let result = verifier
            .verify_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
            )
            .unwrap();

        match result {
            VerificationResult::Violated {
                counterexample,
                output,
                ..
            } => {
                assert_eq!(counterexample.len(), 1);
                assert_eq!(output.len(), 1);
                // The counterexample input should be in bounds
                assert!(counterexample[0] >= -1.0 && counterexample[0] <= 1.0);
            }
            _ => panic!("Expected Violated, got {:?}", result),
        }
    }
}
