//! SMT-based neural network verifier using Z4.
//!
//! Provides complete verification capability by encoding networks as SMT
//! formulas and solving with Z4.

use crate::encoder::{NetworkEncoder, SmtResult};
use crate::model_parser::parse_model;
use crate::{Result, SmtError};
use gamma_core::{Bound, InformativeCounterexample, VerificationResult};
use z4_dpll::Executor;
use z4_frontend::{parse, Command};
use z4_proof::export_alethe;

/// Configuration for the SMT verifier.
#[derive(Debug, Clone)]
pub struct SmtVerifierConfig {
    /// Use Big-M encoding for complete verification (slower but exact).
    /// If false, uses triangle relaxation (faster but incomplete).
    pub use_bigm: bool,
    /// Big-M constant for ReLU encoding.
    pub big_m: f64,
    /// Timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Produce UNSAT proof certificates.
    /// When enabled, verified results include an Alethe-format proof.
    pub produce_proofs: bool,
}

impl Default for SmtVerifierConfig {
    fn default() -> Self {
        Self {
            use_bigm: false,
            big_m: 1e6,
            timeout_ms: Some(60_000), // 60 seconds
            produce_proofs: false,
        }
    }
}

/// SMT-based neural network verifier.
///
/// Uses Z4 SMT solver for complete verification of neural network properties.
#[derive(Debug)]
pub struct SmtVerifier {
    config: SmtVerifierConfig,
}

impl Default for SmtVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl SmtVerifier {
    /// Create a new verifier with default configuration.
    pub fn new() -> Self {
        Self {
            config: SmtVerifierConfig::default(),
        }
    }

    /// Create a verifier with custom configuration.
    pub fn with_config(config: SmtVerifierConfig) -> Self {
        Self { config }
    }

    /// Solve an SMT formula and return the result.
    pub fn solve(&self, formula: &str) -> Result<SmtResult> {
        self.solve_with_vars(formula, &[], &[])
    }

    /// Solve an SMT formula and extract counterexample values for named variables.
    ///
    /// # Arguments
    /// * `formula` - The SMT-LIB formula to solve
    /// * `input_vars` - Variable names for inputs (to extract from model)
    /// * `output_vars` - Variable names for outputs (to extract from model)
    pub fn solve_with_vars(
        &self,
        formula: &str,
        input_vars: &[String],
        output_vars: &[String],
    ) -> Result<SmtResult> {
        tracing::debug!("Solving SMT formula ({} bytes)", formula.len());

        // Parse the SMT-LIB formula
        let commands =
            parse(formula).map_err(|e| SmtError::SolverError(format!("parse error: {}", e)))?;

        // Create Z4 executor
        let mut executor = Executor::new();

        // Enable proof production if configured
        if self.config.produce_proofs {
            executor.set_produce_proofs(true);
            tracing::debug!("Proof production enabled");
        }

        // Execute commands and capture check-sat result
        let mut result = SmtResult::Unknown;
        let mut model_str: Option<String> = None;

        for cmd in &commands {
            match cmd {
                Command::CheckSat => match executor.execute(cmd) {
                    Ok(Some(output)) => {
                        tracing::debug!("check-sat result: {}", output);
                        if output == "sat" {
                            result = SmtResult::Sat {
                                inputs: Vec::new(),
                                outputs: Vec::new(),
                            };
                        } else if output == "unsat" {
                            // Extract proof if enabled
                            let proof = if self.config.produce_proofs {
                                if let Some(proof_data) = executor.get_last_proof() {
                                    let proof_str = export_alethe(proof_data, executor.terms());
                                    tracing::debug!("Generated proof ({} bytes)", proof_str.len());
                                    Some(proof_str)
                                } else {
                                    tracing::warn!(
                                        "Proof production enabled but no proof generated"
                                    );
                                    None
                                }
                            } else {
                                None
                            };
                            result = SmtResult::Unsat { proof };
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        tracing::warn!("Z4 error on check-sat: {}", e);
                        return Err(SmtError::SolverError(format!("check-sat failed: {}", e)));
                    }
                },
                Command::GetModel => {
                    // Try to get the model if SAT
                    if matches!(result, SmtResult::Sat { .. }) {
                        match executor.execute(cmd) {
                            Ok(Some(s)) => {
                                tracing::debug!("Model: {}", s);
                                model_str = Some(s);
                            }
                            Ok(None) => {}
                            Err(e) => {
                                tracing::debug!("get-model failed (expected if unsat): {}", e);
                            }
                        }
                    }
                }
                _ => {
                    // Execute other commands (set-logic, declare-const, assert, etc.)
                    if let Err(e) = executor.execute(cmd) {
                        return Err(SmtError::SolverError(format!(
                            "command execution failed: {}",
                            e
                        )));
                    }
                }
            }
        }

        // Parse counterexample values from model if available
        if let (SmtResult::Sat { .. }, Some(ref model)) = (&result, &model_str) {
            let inputs = if !input_vars.is_empty() {
                parse_model(model, input_vars).unwrap_or_else(|| {
                    tracing::warn!("Failed to parse input values from model");
                    Vec::new()
                })
            } else {
                Vec::new()
            };

            let outputs = if !output_vars.is_empty() {
                parse_model(model, output_vars).unwrap_or_else(|| {
                    tracing::warn!("Failed to parse output values from model");
                    Vec::new()
                })
            } else {
                Vec::new()
            };

            result = SmtResult::Sat { inputs, outputs };
        }

        Ok(result)
    }

    /// Verify a simple feedforward network.
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
        // Encode the network
        let mut encoder = NetworkEncoder::new().with_big_m(self.config.big_m);

        let network = encoder.encode_feedforward(
            weights,
            biases,
            layer_dims,
            input_bounds,
            output_bounds,
            intermediate_bounds,
            self.config.use_bigm,
        )?;

        tracing::info!(
            "Encoded network: {} inputs, {} outputs, {} total variables",
            network.num_inputs,
            network.num_outputs,
            network.total_vars
        );

        // Solve the SMT formula with variable names for counterexample extraction
        let smt_result =
            self.solve_with_vars(&network.formula, &network.input_vars, &network.output_vars)?;

        // Convert SMT result to verification result
        match smt_result {
            SmtResult::Unsat { proof: proof_str } => {
                // Property negation is UNSAT -> property holds
                // Convert proof string to VerificationProof if available
                let proof = proof_str.map(|s| Box::new(gamma_core::VerificationProof::alethe(s)));
                Ok(VerificationResult::Verified {
                    output_bounds: output_bounds.to_vec(),
                    proof,
                })
            }
            SmtResult::Sat { inputs, outputs } => {
                // Property negation is SAT -> counterexample found
                let input_f32: Vec<f32> = inputs.iter().map(|&x| x as f32).collect();
                let output_f32: Vec<f32> = outputs.iter().map(|&x| x as f32).collect();

                // Create informative counterexample with violation details
                let details = InformativeCounterexample::new(
                    input_f32.clone(),
                    output_f32.clone(),
                    Some(output_bounds),
                );

                Ok(VerificationResult::Violated {
                    counterexample: input_f32,
                    output: output_f32,
                    details: Some(Box::new(details)),
                })
            }
            SmtResult::Unknown => Ok(VerificationResult::Unknown {
                bounds: output_bounds.to_vec(),
                reason: "SMT solver returned unknown".to_string(),
            }),
            SmtResult::Timeout => Ok(VerificationResult::Timeout {
                partial_bounds: Some(output_bounds.to_vec()),
            }),
        }
    }

    /// Check if a property is satisfiable given SMT constraints.
    ///
    /// Returns true if the formula is satisfiable (counterexample exists).
    pub fn is_sat(&self, formula: &str) -> Result<bool> {
        match self.solve(formula)? {
            SmtResult::Sat { .. } => Ok(true),
            SmtResult::Unsat { .. } => Ok(false),
            SmtResult::Unknown | SmtResult::Timeout => Err(SmtError::SolverError(
                "solver returned unknown/timeout".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_lra_sat() {
        let verifier = SmtVerifier::new();

        let formula = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (declare-const y Real)
            (assert (> x 0))
            (assert (< y 10))
            (assert (= y (+ x 1)))
            (check-sat)
        "#;

        let result = verifier.solve(formula).unwrap();
        assert!(matches!(result, SmtResult::Sat { .. }));
    }

    #[test]
    fn test_simple_lra_unsat() {
        let verifier = SmtVerifier::new();

        let formula = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (assert (> x 0))
            (assert (< x 0))
            (check-sat)
        "#;

        let result = verifier.solve(formula).unwrap();
        assert!(matches!(result, SmtResult::Unsat { .. }));
    }

    #[test]
    fn test_unsat_proof_generation() {
        let config = SmtVerifierConfig {
            produce_proofs: true,
            ..Default::default()
        };
        let verifier = SmtVerifier::with_config(config);

        // Simple UNSAT formula: x > 0 AND x < 0
        let formula = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (assert (> x 0))
            (assert (< x 0))
            (check-sat)
        "#;

        let result = verifier.solve(formula).unwrap();
        match result {
            SmtResult::Unsat { proof } => {
                assert!(
                    proof.is_some(),
                    "Proof should be generated when produce_proofs=true"
                );
                let proof_str = proof.unwrap();
                assert!(!proof_str.is_empty(), "Proof should not be empty");
                // Alethe proofs typically contain "(step" or "(assume"
                // At minimum they should be parseable text
                println!(
                    "Generated proof ({} bytes):\n{}",
                    proof_str.len(),
                    &proof_str[..proof_str.len().min(500)]
                );
            }
            _ => panic!("Expected Unsat result"),
        }
    }

    #[test]
    fn test_verification_with_proof() {
        // Test that verified results include proofs when enabled
        let config = SmtVerifierConfig {
            produce_proofs: true,
            ..Default::default()
        };
        let verifier = SmtVerifier::with_config(config);

        // Linear network: y = x + 1
        // Input: x in [0, 1] -> y in [1, 2]
        // Property: y in [0.5, 2.5] (satisfied)
        let weights = vec![vec![1.0]];
        let biases = vec![vec![1.0]];
        let layer_dims = vec![1, 1];

        let input_bounds = vec![Bound::new(0.0, 1.0)];
        let output_bounds = vec![Bound::new(0.5, 2.5)];
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

        match result {
            VerificationResult::Verified { proof, .. } => {
                assert!(
                    proof.is_some(),
                    "Proof should be generated for verified result"
                );
                let proof = proof.unwrap();
                assert_eq!(proof.format, gamma_core::ProofFormat::Alethe);
                println!("Verification proof ({} bytes)", proof.data.len());
            }
            _ => panic!("Expected Verified result"),
        }
    }

    #[test]
    fn test_linear_network_verification() {
        let verifier = SmtVerifier::new();

        // Simple linear network: y = x + 1
        // Input: x in [0, 1]
        // Property: y in [0.5, 2.5]
        let weights = vec![vec![1.0]]; // 1x1 identity
        let biases = vec![vec![1.0]]; // add 1
        let layer_dims = vec![1, 1];

        let input_bounds = vec![Bound::new(0.0, 1.0)];
        let output_bounds = vec![Bound::new(0.5, 2.5)];
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

        assert!(result.is_verified());
    }

    #[test]
    fn test_linear_network_violated() {
        let verifier = SmtVerifier::new();

        // Same network but tighter bounds that should be violated
        let weights = vec![vec![1.0]];
        let biases = vec![vec![1.0]];
        let layer_dims = vec![1, 1];

        let input_bounds = vec![Bound::new(0.0, 1.0)];
        // Property: y in [1.5, 1.6] - this is violated since y can be 1.0 or 2.0
        let output_bounds = vec![Bound::new(1.5, 1.6)];
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

        // Should find a counterexample with actual values
        match result {
            VerificationResult::Violated {
                counterexample,
                output,
                details,
            } => {
                // Should have informative details
                assert!(details.is_some(), "Should have counterexample details");
                let details = details.unwrap();
                assert!(
                    details.violated_constraint.is_some(),
                    "Should detect violated constraint"
                );
                // Counterexample should have input values
                assert_eq!(counterexample.len(), 1, "Should have 1 input");
                assert_eq!(output.len(), 1, "Should have 1 output");

                // For y = x + 1 with x in [0, 1], y in [1, 2]
                // Property y in [1.5, 1.6] violated when y < 1.5 or y > 1.6
                // Valid counterexample: x such that y < 1.5 (x < 0.5) or y > 1.6 (x > 0.6)
                let x = counterexample[0] as f64;
                let y = output[0] as f64;

                // Check x is within input bounds
                assert!(
                    (0.0..=1.0).contains(&x),
                    "Counterexample input x={} should be in [0,1]",
                    x
                );

                // Check y = x + 1 (approximately)
                let expected_y = x + 1.0;
                assert!(
                    (y - expected_y).abs() < 1e-6,
                    "Output y={} should equal x+1={}",
                    y,
                    expected_y
                );

                // Check y violates the property (either < 1.5 or > 1.6)
                assert!(
                    !(1.5..=1.6).contains(&y),
                    "Output y={} should violate bounds [1.5, 1.6]",
                    y
                );
            }
            _ => panic!("Expected Violated result, got {:?}", result),
        }
    }

    #[test]
    fn test_counterexample_extraction() {
        let verifier = SmtVerifier::new();

        // Network: y = 2x + 1
        // Input: x in [0, 5]
        // Output y is in [1, 11]
        // Property: y in [5, 7] - violated when x < 2 (y < 5) or x > 3 (y > 7)
        let weights = vec![vec![2.0]];
        let biases = vec![vec![1.0]];
        let layer_dims = vec![1, 1];

        let input_bounds = vec![Bound::new(0.0, 5.0)];
        let output_bounds = vec![Bound::new(5.0, 7.0)];
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

        match result {
            VerificationResult::Violated {
                counterexample,
                output,
                details,
            } => {
                // Verify informative details
                assert!(details.is_some(), "Should have counterexample details");
                let details = details.unwrap();
                assert!(
                    details.violated_constraint.is_some(),
                    "Should detect violated constraint"
                );
                println!("Violation explanation: {}", details.explanation);

                let x = counterexample[0] as f64;
                let y = output[0] as f64;

                // Verify the counterexample satisfies y = 2x + 1
                let expected_y = 2.0 * x + 1.0;
                assert!(
                    (y - expected_y).abs() < 1e-6,
                    "Output y={} should equal 2*x+1={} for x={}",
                    y,
                    expected_y,
                    x
                );

                // Verify the counterexample violates the property
                assert!(
                    !(5.0..=7.0).contains(&y),
                    "Output y={} should be outside [5, 7]",
                    y
                );
            }
            _ => panic!("Expected Violated result, got {:?}", result),
        }
    }

    // ========== SmtVerifierConfig tests ==========

    #[test]
    fn test_smt_verifier_config_default() {
        let config = SmtVerifierConfig::default();
        assert!(!config.use_bigm, "Default should not use big-M encoding");
        assert_eq!(config.big_m, 1e6, "Default big_m should be 1e6");
        assert_eq!(
            config.timeout_ms,
            Some(60_000),
            "Default timeout should be 60s"
        );
        assert!(!config.produce_proofs, "Default should not produce proofs");
    }

    #[test]
    fn test_smt_verifier_config_clone() {
        let config = SmtVerifierConfig {
            use_bigm: true,
            big_m: 1e3,
            timeout_ms: Some(30_000),
            produce_proofs: true,
        };
        let cloned = config.clone();
        assert_eq!(config.use_bigm, cloned.use_bigm);
        assert_eq!(config.big_m, cloned.big_m);
        assert_eq!(config.timeout_ms, cloned.timeout_ms);
        assert_eq!(config.produce_proofs, cloned.produce_proofs);
    }

    #[test]
    fn test_smt_verifier_config_debug() {
        let config = SmtVerifierConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("SmtVerifierConfig"));
        assert!(debug_str.contains("use_bigm"));
        assert!(debug_str.contains("big_m"));
        assert!(debug_str.contains("timeout_ms"));
        assert!(debug_str.contains("produce_proofs"));
    }

    // ========== SmtVerifier construction tests ==========

    #[test]
    fn test_smt_verifier_new() {
        let verifier = SmtVerifier::new();
        // Verify it uses default config
        assert!(!verifier.config.use_bigm);
        assert_eq!(verifier.config.timeout_ms, Some(60_000));
    }

    #[test]
    fn test_smt_verifier_default() {
        let verifier = SmtVerifier::default();
        // Should be equivalent to new()
        assert!(!verifier.config.use_bigm);
        assert_eq!(verifier.config.timeout_ms, Some(60_000));
        assert!(!verifier.config.produce_proofs);
    }

    #[test]
    fn test_smt_verifier_with_config() {
        let config = SmtVerifierConfig {
            use_bigm: true,
            big_m: 5000.0,
            timeout_ms: Some(10_000),
            produce_proofs: true,
        };
        let verifier = SmtVerifier::with_config(config);
        assert!(verifier.config.use_bigm);
        assert_eq!(verifier.config.big_m, 5000.0);
        assert_eq!(verifier.config.timeout_ms, Some(10_000));
        assert!(verifier.config.produce_proofs);
    }

    #[test]
    fn test_smt_verifier_debug() {
        let verifier = SmtVerifier::new();
        let debug_str = format!("{:?}", verifier);
        assert!(debug_str.contains("SmtVerifier"));
        assert!(debug_str.contains("config"));
    }

    // ========== is_sat() tests ==========

    #[test]
    fn test_is_sat_returns_true_for_sat() {
        let verifier = SmtVerifier::new();
        let formula = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (assert (> x 0))
            (assert (< x 10))
            (check-sat)
        "#;
        let result = verifier.is_sat(formula).unwrap();
        assert!(result, "Should return true for SAT formula");
    }

    #[test]
    fn test_is_sat_returns_false_for_unsat() {
        let verifier = SmtVerifier::new();
        let formula = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (assert (> x 5))
            (assert (< x 3))
            (check-sat)
        "#;
        let result = verifier.is_sat(formula).unwrap();
        assert!(!result, "Should return false for UNSAT formula");
    }

    // ========== Error handling tests ==========

    #[test]
    fn test_solve_parse_error() {
        let verifier = SmtVerifier::new();
        let formula = "(set-logic QF_LRA)(invalid syntax here";
        let result = verifier.solve(formula);
        assert!(result.is_err(), "Should return error for invalid formula");
        let err = result.unwrap_err();
        let err_str = format!("{}", err);
        assert!(
            err_str.contains("parse error"),
            "Error should mention parse error: {}",
            err_str
        );
    }

    #[test]
    fn test_solve_with_empty_formula() {
        let verifier = SmtVerifier::new();
        // Empty formula has no check-sat, should return Unknown
        let formula = "(set-logic QF_LRA)";
        let result = verifier.solve(formula).unwrap();
        assert!(
            matches!(result, SmtResult::Unknown),
            "Empty formula without check-sat should return Unknown"
        );
    }

    // ========== solve_with_vars tests ==========

    #[test]
    fn test_solve_with_vars_extracts_values() {
        let verifier = SmtVerifier::new();
        let formula = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (declare-const y Real)
            (assert (= x 3.5))
            (assert (= y 7.0))
            (check-sat)
            (get-model)
        "#;

        let input_vars = vec!["x".to_string()];
        let output_vars = vec!["y".to_string()];
        let result = verifier
            .solve_with_vars(formula, &input_vars, &output_vars)
            .unwrap();

        match result {
            SmtResult::Sat { inputs, outputs } => {
                assert_eq!(inputs.len(), 1, "Should have 1 input");
                assert_eq!(outputs.len(), 1, "Should have 1 output");
                assert!(
                    (inputs[0] - 3.5).abs() < 1e-6,
                    "Input x should be 3.5, got {}",
                    inputs[0]
                );
                assert!(
                    (outputs[0] - 7.0).abs() < 1e-6,
                    "Output y should be 7.0, got {}",
                    outputs[0]
                );
            }
            _ => panic!("Expected Sat result with values"),
        }
    }

    #[test]
    fn test_solve_with_empty_vars() {
        let verifier = SmtVerifier::new();
        let formula = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (assert (= x 1.0))
            (check-sat)
        "#;

        let result = verifier.solve_with_vars(formula, &[], &[]).unwrap();
        match result {
            SmtResult::Sat { inputs, outputs } => {
                assert!(inputs.is_empty(), "Inputs should be empty");
                assert!(outputs.is_empty(), "Outputs should be empty");
            }
            _ => panic!("Expected Sat result"),
        }
    }

    // ========== Multi-input/output network tests ==========

    #[test]
    fn test_multi_input_network() {
        let verifier = SmtVerifier::new();

        // 2-input, 1-output network: y = x1 + 2*x2
        // Weights: [1.0, 2.0] (row-major for 1x2)
        let weights = vec![vec![1.0, 2.0]];
        let biases = vec![vec![0.0]];
        let layer_dims = vec![2, 1];

        // Inputs: x1 in [0, 1], x2 in [0, 1]
        // Output: y in [0, 3] (when x1=1, x2=1, y=3)
        let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
        let output_bounds = vec![Bound::new(-0.5, 3.5)]; // Should hold
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

        assert!(
            result.is_verified(),
            "Multi-input network should verify: {:?}",
            result
        );
    }

    #[test]
    fn test_multi_output_network() {
        let verifier = SmtVerifier::new();

        // 1-input, 2-output network: y1 = x, y2 = 2*x
        // Weights: [[1.0], [2.0]] flattened row-major = [1.0, 2.0]
        let weights = vec![vec![1.0, 2.0]];
        let biases = vec![vec![0.0, 0.0]];
        let layer_dims = vec![1, 2];

        // Input: x in [0, 1]
        // Output: y1 in [0, 1], y2 in [0, 2]
        let input_bounds = vec![Bound::new(0.0, 1.0)];
        let output_bounds = vec![Bound::new(-0.5, 1.5), Bound::new(-0.5, 2.5)];
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

        assert!(
            result.is_verified(),
            "Multi-output network should verify: {:?}",
            result
        );
    }

    // ========== Big-M encoding tests ==========

    #[test]
    fn test_bigm_encoding_linear_network() {
        let config = SmtVerifierConfig {
            use_bigm: true,
            big_m: 1e4,
            ..Default::default()
        };
        let verifier = SmtVerifier::with_config(config);

        // Linear network (no ReLU) - should work same as triangle relaxation
        let weights = vec![vec![1.0]];
        let biases = vec![vec![1.0]];
        let layer_dims = vec![1, 1];

        let input_bounds = vec![Bound::new(0.0, 1.0)];
        let output_bounds = vec![Bound::new(0.5, 2.5)];
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

        assert!(
            result.is_verified(),
            "Big-M encoding should verify linear network"
        );
    }

    // ========== Edge case tests ==========

    #[test]
    fn test_zero_width_input_bounds() {
        let verifier = SmtVerifier::new();

        // Input fixed at exactly 0.5: x in [0.5, 0.5]
        // y = x + 1 = 1.5
        let weights = vec![vec![1.0]];
        let biases = vec![vec![1.0]];
        let layer_dims = vec![1, 1];

        let input_bounds = vec![Bound::new(0.5, 0.5)]; // Point input
        let output_bounds = vec![Bound::new(1.4, 1.6)]; // Should contain 1.5
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

        assert!(
            result.is_verified(),
            "Point input should verify: {:?}",
            result
        );
    }

    #[test]
    fn test_negative_input_bounds() {
        let verifier = SmtVerifier::new();

        // Input in [-2, -1], y = x + 5, so y in [3, 4]
        let weights = vec![vec![1.0]];
        let biases = vec![vec![5.0]];
        let layer_dims = vec![1, 1];

        let input_bounds = vec![Bound::new(-2.0, -1.0)];
        let output_bounds = vec![Bound::new(2.5, 4.5)]; // Contains [3, 4]
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

        assert!(
            result.is_verified(),
            "Negative input bounds should verify: {:?}",
            result
        );
    }

    #[test]
    fn test_large_weight_network() {
        let verifier = SmtVerifier::new();

        // y = 1000*x with x in [0, 0.001], y in [0, 1]
        let weights = vec![vec![1000.0]];
        let biases = vec![vec![0.0]];
        let layer_dims = vec![1, 1];

        let input_bounds = vec![Bound::new(0.0, 0.001)];
        let output_bounds = vec![Bound::new(-0.1, 1.1)]; // Contains [0, 1]
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

        assert!(
            result.is_verified(),
            "Large weight network should verify: {:?}",
            result
        );
    }

    #[test]
    fn test_small_weight_network() {
        let verifier = SmtVerifier::new();

        // y = 0.001*x with x in [0, 1000], y in [0, 1]
        let weights = vec![vec![0.001]];
        let biases = vec![vec![0.0]];
        let layer_dims = vec![1, 1];

        let input_bounds = vec![Bound::new(0.0, 1000.0)];
        let output_bounds = vec![Bound::new(-0.1, 1.1)]; // Contains [0, 1]
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

        assert!(
            result.is_verified(),
            "Small weight network should verify: {:?}",
            result
        );
    }
}
