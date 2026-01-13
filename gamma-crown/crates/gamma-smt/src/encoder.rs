//! Neural network to SMT formula encoder.
//!
//! Encodes feedforward neural networks with ReLU activations into
//! SMT-LIB formulas in the QF_LRA theory.

use crate::{Result, SmtError};
use gamma_core::Bound;

/// SMT encoding of a neural network.
///
/// Represents a network as a set of SMT-LIB constraints that can be
/// solved by Z4 or other SMT solvers.
#[derive(Debug, Clone)]
pub struct SmtNetwork {
    /// SMT-LIB formula as a string.
    pub formula: String,
    /// Number of input variables.
    pub num_inputs: usize,
    /// Number of output variables.
    pub num_outputs: usize,
    /// Input variable names (for parsing counterexamples).
    pub input_vars: Vec<String>,
    /// Output variable names (for parsing counterexamples).
    pub output_vars: Vec<String>,
    /// Total number of variables in the encoding.
    pub total_vars: usize,
}

/// Result of SMT solving.
#[derive(Debug, Clone)]
pub enum SmtResult {
    /// Satisfiable: counterexample found.
    Sat {
        /// Counterexample input values.
        inputs: Vec<f64>,
        /// Output values at counterexample.
        outputs: Vec<f64>,
    },
    /// Unsatisfiable: property verified.
    Unsat {
        /// Optional UNSAT proof in Alethe format.
        proof: Option<String>,
    },
    /// Unknown: solver could not determine.
    Unknown,
    /// Solver timed out.
    Timeout,
}

/// Encoder for converting neural networks to SMT formulas.
///
/// Currently supports:
/// - Linear layers (Dense/Fully connected)
/// - ReLU activations (using Big-M encoding)
///
/// The encoding uses QF_LRA (Quantifier-Free Linear Real Arithmetic).
#[derive(Debug)]
pub struct NetworkEncoder {
    /// Big-M constant for ReLU encoding.
    pub big_m: f64,
    /// Variable counter for unique naming.
    var_counter: usize,
    /// Accumulated constraints.
    constraints: Vec<String>,
    /// Variable declarations.
    declarations: Vec<String>,
}

impl Default for NetworkEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkEncoder {
    /// Create a new encoder with default settings.
    pub fn new() -> Self {
        Self {
            big_m: 1e6,
            var_counter: 0,
            constraints: Vec::new(),
            declarations: Vec::new(),
        }
    }

    /// Set the Big-M constant for ReLU encoding.
    pub fn with_big_m(mut self, big_m: f64) -> Self {
        self.big_m = big_m;
        self
    }

    /// Reset the encoder state for a new network.
    pub fn reset(&mut self) {
        self.var_counter = 0;
        self.constraints.clear();
        self.declarations.clear();
    }

    /// Generate a unique variable name.
    fn fresh_var(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.var_counter);
        self.var_counter += 1;
        name
    }

    /// Declare a Real variable.
    fn declare_real(&mut self, name: &str) {
        self.declarations
            .push(format!("(declare-const {} Real)", name));
    }

    /// Add a constraint.
    fn add_constraint(&mut self, constraint: &str) {
        self.constraints.push(format!("(assert {})", constraint));
    }

    /// Encode input bounds.
    fn encode_input_bounds(&mut self, input_vars: &[String], bounds: &[Bound]) -> Result<()> {
        if input_vars.len() != bounds.len() {
            return Err(SmtError::InvalidBounds(format!(
                "expected {} bounds, got {}",
                input_vars.len(),
                bounds.len()
            )));
        }

        for (var, bound) in input_vars.iter().zip(bounds.iter()) {
            // Lower bound: var >= lower
            self.add_constraint(&format!("(>= {} {})", var, bound.lower));
            // Upper bound: var <= upper
            self.add_constraint(&format!("(<= {} {})", var, bound.upper));
        }

        Ok(())
    }

    /// Encode a linear layer: y = W*x + b.
    ///
    /// # Arguments
    /// * `input_vars` - Names of input variables
    /// * `weights` - Weight matrix (output_dim x input_dim, row-major)
    /// * `bias` - Bias vector (output_dim)
    ///
    /// # Returns
    /// Names of output variables
    pub fn encode_linear(
        &mut self,
        input_vars: &[String],
        weights: &[f64],
        bias: &[f64],
        output_dim: usize,
    ) -> Result<Vec<String>> {
        let input_dim = input_vars.len();

        if weights.len() != output_dim * input_dim {
            return Err(SmtError::EncodingError(format!(
                "weight matrix size mismatch: expected {}x{}, got {}",
                output_dim,
                input_dim,
                weights.len()
            )));
        }

        if bias.len() != output_dim {
            return Err(SmtError::EncodingError(format!(
                "bias size mismatch: expected {}, got {}",
                output_dim,
                bias.len()
            )));
        }

        let mut output_vars = Vec::with_capacity(output_dim);

        for i in 0..output_dim {
            let out_var = self.fresh_var("lin");
            self.declare_real(&out_var);

            // Build the linear combination: sum_j(W[i,j] * x[j]) + b[i]
            let mut terms = Vec::new();
            for (j, x_var) in input_vars.iter().enumerate() {
                let w = weights[i * input_dim + j];
                if w.abs() > 1e-15 {
                    terms.push(format!("(* {} {})", w, x_var));
                }
            }

            let b = bias[i];
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

            self.add_constraint(&format!("(= {} {})", out_var, rhs));
            output_vars.push(out_var);
        }

        Ok(output_vars)
    }

    /// Encode ReLU activation using triangle relaxation (sound but incomplete).
    ///
    /// For each neuron:
    /// - If lower >= 0: y = x (active)
    /// - If upper <= 0: y = 0 (inactive)
    /// - Otherwise: y >= 0, y >= x, y <= upper*(x-lower)/(upper-lower)
    ///
    /// This is the standard linear relaxation from DeepPoly/CROWN.
    pub fn encode_relu_relaxation(
        &mut self,
        input_vars: &[String],
        bounds: &[Bound],
    ) -> Result<Vec<String>> {
        if input_vars.len() != bounds.len() {
            return Err(SmtError::EncodingError(format!(
                "ReLU bounds mismatch: {} vars, {} bounds",
                input_vars.len(),
                bounds.len()
            )));
        }

        let mut output_vars = Vec::with_capacity(input_vars.len());

        for (x_var, bound) in input_vars.iter().zip(bounds.iter()) {
            let y_var = self.fresh_var("relu");
            self.declare_real(&y_var);

            let l = bound.lower as f64;
            let u = bound.upper as f64;

            if l >= 0.0 {
                // Active: y = x
                self.add_constraint(&format!("(= {} {})", y_var, x_var));
            } else if u <= 0.0 {
                // Inactive: y = 0
                self.add_constraint(&format!("(= {} 0.0)", y_var));
            } else {
                // Uncertain: triangle relaxation
                // y >= 0
                self.add_constraint(&format!("(>= {} 0.0)", y_var));
                // y >= x
                self.add_constraint(&format!("(>= {} {})", y_var, x_var));
                // y <= u*(x-l)/(u-l)
                // Rewritten: y <= (u/(u-l))*x - (u*l/(u-l))
                let slope = u / (u - l);
                let intercept = -u * l / (u - l);
                self.add_constraint(&format!(
                    "(<= {} (+ (* {} {}) {}))",
                    y_var, slope, x_var, intercept
                ));
            }

            output_vars.push(y_var);
        }

        Ok(output_vars)
    }

    /// Encode ReLU activation using Big-M encoding (complete but harder).
    ///
    /// Uses integer indicator variable p:
    /// - y >= 0
    /// - y >= x
    /// - y <= x + M*(1-p)
    /// - y <= M*p
    ///
    /// Where p=1 means active (y=x), p=0 means inactive (y=0).
    /// Requires QF_UFLIA or similar logic that supports integer constraints.
    pub fn encode_relu_bigm(
        &mut self,
        input_vars: &[String],
        bounds: &[Bound],
    ) -> Result<Vec<String>> {
        if input_vars.len() != bounds.len() {
            return Err(SmtError::EncodingError(format!(
                "ReLU bounds mismatch: {} vars, {} bounds",
                input_vars.len(),
                bounds.len()
            )));
        }

        let mut output_vars = Vec::with_capacity(input_vars.len());
        let m = self.big_m;

        for (x_var, bound) in input_vars.iter().zip(bounds.iter()) {
            let y_var = self.fresh_var("relu");
            let p_var = self.fresh_var("phase");
            self.declare_real(&y_var);
            self.declarations
                .push(format!("(declare-const {} Int)", p_var));

            let l = bound.lower as f64;
            let u = bound.upper as f64;

            if l >= 0.0 {
                // Definitely active
                self.add_constraint(&format!("(= {} {})", y_var, x_var));
                self.add_constraint(&format!("(= {} 1)", p_var));
            } else if u <= 0.0 {
                // Definitely inactive
                self.add_constraint(&format!("(= {} 0.0)", y_var));
                self.add_constraint(&format!("(= {} 0)", p_var));
            } else {
                // Uncertain: Big-M encoding
                // Phase is binary
                self.add_constraint(&format!("(>= {} 0)", p_var));
                self.add_constraint(&format!("(<= {} 1)", p_var));

                // y >= 0
                self.add_constraint(&format!("(>= {} 0.0)", y_var));

                // y >= x (implied by y = x when active, and y >= 0 when inactive)
                self.add_constraint(&format!("(>= {} {})", y_var, x_var));

                // y <= x + M*(1-p)
                // When p=1: y <= x (active)
                // When p=0: y <= x + M (effectively no constraint when M large)
                self.add_constraint(&format!(
                    "(<= {} (+ {} (* {} (- 1 {}))))",
                    y_var, x_var, m, p_var
                ));

                // y <= M*p
                // When p=1: y <= M (no effective constraint)
                // When p=0: y <= 0, combined with y >= 0 gives y = 0
                self.add_constraint(&format!("(<= {} (* {} {}))", y_var, m, p_var));
            }

            output_vars.push(y_var);
        }

        Ok(output_vars)
    }

    /// Encode output property negation.
    ///
    /// To verify that outputs are within bounds, we encode the negation:
    /// EXISTS output: (output < lower_bound OR output > upper_bound)
    ///
    /// If this is UNSAT, the property holds. If SAT, we have a counterexample.
    pub fn encode_output_property_negation(
        &mut self,
        output_vars: &[String],
        bounds: &[Bound],
    ) -> Result<()> {
        if output_vars.len() != bounds.len() {
            return Err(SmtError::EncodingError(format!(
                "Output bounds mismatch: {} vars, {} bounds",
                output_vars.len(),
                bounds.len()
            )));
        }

        // Negation of "all outputs within bounds":
        // At least one output is outside its bounds
        let mut disjuncts = Vec::new();

        for (var, bound) in output_vars.iter().zip(bounds.iter()) {
            // var < lower OR var > upper
            disjuncts.push(format!("(< {} {})", var, bound.lower));
            disjuncts.push(format!("(> {} {})", var, bound.upper));
        }

        if disjuncts.len() == 1 {
            self.add_constraint(&disjuncts[0]);
        } else {
            self.add_constraint(&format!("(or {})", disjuncts.join(" ")));
        }

        Ok(())
    }

    /// Build the final SMT-LIB formula.
    pub fn build(
        &self,
        input_vars: &[String],
        output_vars: &[String],
        use_lia: bool,
    ) -> SmtNetwork {
        let logic = if use_lia { "QF_UFLIA" } else { "QF_LRA" };

        let mut formula = String::new();
        formula.push_str(&format!("(set-logic {})\n", logic));

        for decl in &self.declarations {
            formula.push_str(decl);
            formula.push('\n');
        }

        for constraint in &self.constraints {
            formula.push_str(constraint);
            formula.push('\n');
        }

        formula.push_str("(check-sat)\n");
        formula.push_str("(get-model)\n");

        SmtNetwork {
            formula,
            num_inputs: input_vars.len(),
            num_outputs: output_vars.len(),
            input_vars: input_vars.to_vec(),
            output_vars: output_vars.to_vec(),
            total_vars: self.var_counter,
        }
    }

    /// Encode a simple feedforward network.
    ///
    /// # Arguments
    /// * `weights` - List of weight matrices (one per layer), each row-major
    /// * `biases` - List of bias vectors (one per layer)
    /// * `layer_dims` - Dimensions: [input_dim, hidden1, hidden2, ..., output_dim]
    /// * `input_bounds` - Bounds on inputs
    /// * `output_bounds` - Required output bounds (for property encoding)
    /// * `intermediate_bounds` - Pre-computed bounds on intermediate neurons (for ReLU)
    #[allow(clippy::too_many_arguments)]
    pub fn encode_feedforward(
        &mut self,
        weights: &[Vec<f64>],
        biases: &[Vec<f64>],
        layer_dims: &[usize],
        input_bounds: &[Bound],
        output_bounds: &[Bound],
        intermediate_bounds: &[Vec<Bound>],
        use_bigm: bool,
    ) -> Result<SmtNetwork> {
        self.reset();

        let num_layers = weights.len();
        if num_layers == 0 {
            return Err(SmtError::EncodingError("empty network".to_string()));
        }

        if layer_dims.len() != num_layers + 1 {
            return Err(SmtError::EncodingError(format!(
                "layer_dims length {} doesn't match {} layers + 1",
                layer_dims.len(),
                num_layers
            )));
        }

        let input_dim = layer_dims[0];
        let _output_dim = layer_dims[num_layers];

        // Create input variables
        let mut input_vars = Vec::with_capacity(input_dim);
        for i in 0..input_dim {
            let var = format!("x_{}", i);
            self.declare_real(&var);
            input_vars.push(var);
        }

        // Encode input bounds
        self.encode_input_bounds(&input_vars, input_bounds)?;

        // Process each layer
        let mut current_vars = input_vars.clone();
        for layer_idx in 0..num_layers {
            let out_dim = layer_dims[layer_idx + 1];

            // Linear transformation
            current_vars = self.encode_linear(
                &current_vars,
                &weights[layer_idx],
                &biases[layer_idx],
                out_dim,
            )?;

            // Apply ReLU to all layers except the last
            if layer_idx < num_layers - 1 {
                if layer_idx >= intermediate_bounds.len() {
                    return Err(SmtError::EncodingError(format!(
                        "missing intermediate bounds for layer {}",
                        layer_idx
                    )));
                }

                let bounds = &intermediate_bounds[layer_idx];
                current_vars = if use_bigm {
                    self.encode_relu_bigm(&current_vars, bounds)?
                } else {
                    self.encode_relu_relaxation(&current_vars, bounds)?
                };
            }
        }

        // Record output variables
        let output_vars = current_vars.clone();

        // Encode property negation
        self.encode_output_property_negation(&output_vars, output_bounds)?;

        Ok(self.build(&input_vars, &output_vars, use_bigm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_encoding() {
        let mut encoder = NetworkEncoder::new();

        // Simple 2x2 network: y = [[1, 2], [3, 4]] * x + [0.1, 0.2]
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.1, 0.2];

        encoder.declare_real("x_0");
        encoder.declare_real("x_1");
        let input_vars = vec!["x_0".to_string(), "x_1".to_string()];

        let output_vars = encoder
            .encode_linear(&input_vars, &weights, &bias, 2)
            .unwrap();

        assert_eq!(output_vars.len(), 2);
        assert!(encoder.constraints.len() >= 2);
    }

    #[test]
    fn test_relu_relaxation() {
        let mut encoder = NetworkEncoder::new();

        encoder.declare_real("x_0");
        encoder.declare_real("x_1");
        encoder.declare_real("x_2");
        let input_vars = vec!["x_0".to_string(), "x_1".to_string(), "x_2".to_string()];

        // Three cases: definitely active, definitely inactive, uncertain
        let bounds = vec![
            Bound::new(1.0, 2.0),   // active
            Bound::new(-2.0, -1.0), // inactive
            Bound::new(-1.0, 1.0),  // uncertain
        ];

        let output_vars = encoder
            .encode_relu_relaxation(&input_vars, &bounds)
            .unwrap();

        assert_eq!(output_vars.len(), 3);
        // Should have constraints for each case
        assert!(!encoder.constraints.is_empty());
    }

    #[test]
    fn test_simple_network_encoding() {
        let mut encoder = NetworkEncoder::new();

        // Simple 2 -> 2 -> 1 network
        let weights = vec![
            vec![1.0, -1.0, 0.5, 0.5], // 2x2 -> first hidden
            vec![1.0, 1.0],            // 2 -> 1 output
        ];
        let biases = vec![vec![0.0, 0.0], vec![0.0]];
        let layer_dims = vec![2, 2, 1];

        let input_bounds = vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)];
        let output_bounds = vec![Bound::new(-10.0, 10.0)];

        // Pre-computed intermediate bounds (after first linear, before ReLU)
        let intermediate_bounds = vec![vec![Bound::new(-2.0, 2.0), Bound::new(-1.0, 1.0)]];

        let network = encoder
            .encode_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
                false,
            )
            .unwrap();

        assert_eq!(network.num_inputs, 2);
        assert_eq!(network.num_outputs, 1);
        assert!(network.formula.contains("set-logic"));
        assert!(network.formula.contains("check-sat"));
    }

    // ============ New tests added for better coverage ============

    #[test]
    fn test_network_encoder_default() {
        let encoder = NetworkEncoder::default();
        assert_eq!(encoder.big_m, 1e6);
        assert_eq!(encoder.var_counter, 0);
        assert!(encoder.constraints.is_empty());
        assert!(encoder.declarations.is_empty());
    }

    #[test]
    fn test_network_encoder_with_big_m() {
        let encoder = NetworkEncoder::new().with_big_m(1000.0);
        assert_eq!(encoder.big_m, 1000.0);
    }

    #[test]
    fn test_network_encoder_reset() {
        let mut encoder = NetworkEncoder::new();
        encoder.declare_real("test_var");
        encoder.add_constraint("(> test_var 0)");
        let _ = encoder.fresh_var("x");

        assert!(!encoder.declarations.is_empty());
        assert!(!encoder.constraints.is_empty());
        assert!(encoder.var_counter > 0);

        encoder.reset();

        assert!(encoder.declarations.is_empty());
        assert!(encoder.constraints.is_empty());
        assert_eq!(encoder.var_counter, 0);
    }

    #[test]
    fn test_fresh_var_unique_names() {
        let mut encoder = NetworkEncoder::new();
        let v1 = encoder.fresh_var("x");
        let v2 = encoder.fresh_var("x");
        let v3 = encoder.fresh_var("y");

        assert_eq!(v1, "x_0");
        assert_eq!(v2, "x_1");
        assert_eq!(v3, "y_2");
        assert_ne!(v1, v2);
        assert_ne!(v2, v3);
    }

    #[test]
    fn test_smt_network_fields() {
        let network = SmtNetwork {
            formula: "(set-logic QF_LRA)(check-sat)".to_string(),
            num_inputs: 3,
            num_outputs: 2,
            input_vars: vec!["x_0".to_string(), "x_1".to_string(), "x_2".to_string()],
            output_vars: vec!["y_0".to_string(), "y_1".to_string()],
            total_vars: 10,
        };

        assert_eq!(network.num_inputs, 3);
        assert_eq!(network.num_outputs, 2);
        assert_eq!(network.input_vars.len(), 3);
        assert_eq!(network.output_vars.len(), 2);
        assert_eq!(network.total_vars, 10);
        assert!(network.formula.contains("check-sat"));
    }

    #[test]
    fn test_smt_result_sat_variant() {
        let result = SmtResult::Sat {
            inputs: vec![1.0, 2.0],
            outputs: vec![3.0],
        };
        match result {
            SmtResult::Sat { inputs, outputs } => {
                assert_eq!(inputs, vec![1.0, 2.0]);
                assert_eq!(outputs, vec![3.0]);
            }
            _ => panic!("Expected Sat variant"),
        }
    }

    #[test]
    fn test_smt_result_unsat_variant() {
        let result = SmtResult::Unsat {
            proof: Some("(proof ...)".to_string()),
        };
        match result {
            SmtResult::Unsat { proof } => {
                assert!(proof.is_some());
                assert!(proof.unwrap().contains("proof"));
            }
            _ => panic!("Expected Unsat variant"),
        }
    }

    #[test]
    fn test_smt_result_unknown_and_timeout() {
        let unknown = SmtResult::Unknown;
        let timeout = SmtResult::Timeout;

        match unknown {
            SmtResult::Unknown => (),
            _ => panic!("Expected Unknown variant"),
        }
        match timeout {
            SmtResult::Timeout => (),
            _ => panic!("Expected Timeout variant"),
        }
    }

    #[test]
    fn test_encode_input_bounds_mismatch() {
        let mut encoder = NetworkEncoder::new();
        let input_vars = vec!["x_0".to_string(), "x_1".to_string()];
        let bounds = vec![Bound::new(0.0, 1.0)]; // Only 1 bound for 2 vars

        let result = encoder.encode_input_bounds(&input_vars, &bounds);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SmtError::InvalidBounds(_)));
    }

    #[test]
    fn test_encode_input_bounds_correct() {
        let mut encoder = NetworkEncoder::new();
        encoder.declare_real("x_0");
        encoder.declare_real("x_1");
        let input_vars = vec!["x_0".to_string(), "x_1".to_string()];
        let bounds = vec![Bound::new(-1.0, 1.0), Bound::new(0.0, 2.0)];

        let result = encoder.encode_input_bounds(&input_vars, &bounds);
        assert!(result.is_ok());
        // 2 vars * 2 constraints each (lower and upper) = 4 constraints
        assert_eq!(encoder.constraints.len(), 4);
    }

    #[test]
    fn test_encode_linear_weight_mismatch() {
        let mut encoder = NetworkEncoder::new();
        let input_vars = vec!["x_0".to_string(), "x_1".to_string()];
        let weights = vec![1.0, 2.0, 3.0]; // Wrong size
        let bias = vec![0.0, 0.0];

        let result = encoder.encode_linear(&input_vars, &weights, &bias, 2);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SmtError::EncodingError(_)));
    }

    #[test]
    fn test_encode_linear_bias_mismatch() {
        let mut encoder = NetworkEncoder::new();
        let input_vars = vec!["x_0".to_string(), "x_1".to_string()];
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.0]; // Wrong size

        let result = encoder.encode_linear(&input_vars, &weights, &bias, 2);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SmtError::EncodingError(_)));
    }

    #[test]
    fn test_encode_linear_zero_weights_skipped() {
        let mut encoder = NetworkEncoder::new();
        encoder.declare_real("x_0");
        let input_vars = vec!["x_0".to_string()];
        // Weight is effectively zero (< 1e-15)
        let weights = vec![1e-20];
        let bias = vec![5.0];

        let output_vars = encoder
            .encode_linear(&input_vars, &weights, &bias, 1)
            .unwrap();
        assert_eq!(output_vars.len(), 1);
        // Should have constraint: lin_0 = 5.0 (no weight term)
        let constraint = &encoder.constraints[0];
        assert!(constraint.contains("5"));
    }

    #[test]
    fn test_encode_relu_bigm_all_cases() {
        let mut encoder = NetworkEncoder::new().with_big_m(100.0);

        encoder.declare_real("x_0");
        encoder.declare_real("x_1");
        encoder.declare_real("x_2");
        let input_vars = vec!["x_0".to_string(), "x_1".to_string(), "x_2".to_string()];

        // Three cases: active, inactive, uncertain
        let bounds = vec![
            Bound::new(1.0, 2.0),   // active: l >= 0
            Bound::new(-2.0, -1.0), // inactive: u <= 0
            Bound::new(-1.0, 1.0),  // uncertain: l < 0 < u
        ];

        let output_vars = encoder.encode_relu_bigm(&input_vars, &bounds).unwrap();
        assert_eq!(output_vars.len(), 3);

        // Should have declarations for both Real (relu) and Int (phase) variables
        let int_decls: Vec<_> = encoder
            .declarations
            .iter()
            .filter(|d| d.contains("Int"))
            .collect();
        // Uncertain case adds 1 Int variable, active/inactive also add 1 each
        // Actually: active has p=1, inactive has p=0, uncertain has binary phase
        assert_eq!(int_decls.len(), 3);
    }

    #[test]
    fn test_encode_relu_bigm_bounds_mismatch() {
        let mut encoder = NetworkEncoder::new();
        let input_vars = vec!["x_0".to_string()];
        let bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)]; // 2 bounds for 1 var

        let result = encoder.encode_relu_bigm(&input_vars, &bounds);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_relu_relaxation_bounds_mismatch() {
        let mut encoder = NetworkEncoder::new();
        let input_vars = vec!["x_0".to_string()];
        let bounds = vec![]; // No bounds

        let result = encoder.encode_relu_relaxation(&input_vars, &bounds);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_output_property_negation() {
        let mut encoder = NetworkEncoder::new();
        encoder.declare_real("y_0");
        encoder.declare_real("y_1");
        let output_vars = vec!["y_0".to_string(), "y_1".to_string()];
        let bounds = vec![Bound::new(0.0, 1.0), Bound::new(-1.0, 2.0)];

        encoder
            .encode_output_property_negation(&output_vars, &bounds)
            .unwrap();

        // Should have one constraint with disjunction
        assert_eq!(encoder.constraints.len(), 1);
        let constraint = &encoder.constraints[0];
        assert!(constraint.contains("or"));
        // Should have 4 disjuncts: y_0 < 0, y_0 > 1, y_1 < -1, y_1 > 2
        assert!(constraint.contains("< y_0"));
        assert!(constraint.contains("> y_0"));
        assert!(constraint.contains("< y_1"));
        assert!(constraint.contains("> y_1"));
    }

    #[test]
    fn test_encode_output_property_single_output() {
        let mut encoder = NetworkEncoder::new();
        encoder.declare_real("y_0");
        let output_vars = vec!["y_0".to_string()];
        let bounds = vec![Bound::new(0.0, 1.0)];

        encoder
            .encode_output_property_negation(&output_vars, &bounds)
            .unwrap();

        // With 2 disjuncts, should still use "or"
        let constraint = &encoder.constraints[0];
        assert!(constraint.contains("or"));
    }

    #[test]
    fn test_encode_output_property_bounds_mismatch() {
        let mut encoder = NetworkEncoder::new();
        let output_vars = vec!["y_0".to_string()];
        let bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)]; // 2 bounds for 1 var

        let result = encoder.encode_output_property_negation(&output_vars, &bounds);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_qf_lra_logic() {
        let mut encoder = NetworkEncoder::new();
        encoder.declare_real("x_0");
        let input_vars = vec!["x_0".to_string()];
        let output_vars = vec!["y_0".to_string()];

        let network = encoder.build(&input_vars, &output_vars, false);

        assert!(network.formula.contains("set-logic QF_LRA"));
        assert!(!network.formula.contains("QF_UFLIA"));
    }

    #[test]
    fn test_build_qf_uflia_logic() {
        let mut encoder = NetworkEncoder::new();
        encoder.declare_real("x_0");
        let input_vars = vec!["x_0".to_string()];
        let output_vars = vec!["y_0".to_string()];

        let network = encoder.build(&input_vars, &output_vars, true);

        assert!(network.formula.contains("set-logic QF_UFLIA"));
        assert!(!network.formula.contains("QF_LRA\n"));
    }

    #[test]
    fn test_build_formula_structure() {
        let mut encoder = NetworkEncoder::new();
        encoder.declare_real("x_0");
        encoder.add_constraint("(> x_0 0)");
        let input_vars = vec!["x_0".to_string()];
        let output_vars = vec!["y_0".to_string()];

        let network = encoder.build(&input_vars, &output_vars, false);

        // Check formula structure
        assert!(network.formula.contains("declare-const x_0 Real"));
        assert!(network.formula.contains("assert (> x_0 0)"));
        assert!(network.formula.contains("check-sat"));
        assert!(network.formula.contains("get-model"));
    }

    #[test]
    fn test_encode_feedforward_empty_network() {
        let mut encoder = NetworkEncoder::new();
        let weights: Vec<Vec<f64>> = vec![];
        let biases: Vec<Vec<f64>> = vec![];
        let layer_dims = vec![1];
        let input_bounds = vec![Bound::new(0.0, 1.0)];
        let output_bounds = vec![Bound::new(0.0, 1.0)];
        let intermediate_bounds: Vec<Vec<Bound>> = vec![];

        let result = encoder.encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &output_bounds,
            &intermediate_bounds,
            false,
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SmtError::EncodingError(_)));
    }

    #[test]
    fn test_encode_feedforward_layer_dims_mismatch() {
        let mut encoder = NetworkEncoder::new();
        let weights = vec![vec![1.0]];
        let biases = vec![vec![0.0]];
        let layer_dims = vec![1]; // Should be [1, 1] for 1 layer
        let input_bounds = vec![Bound::new(0.0, 1.0)];
        let output_bounds = vec![Bound::new(0.0, 1.0)];
        let intermediate_bounds: Vec<Vec<Bound>> = vec![];

        let result = encoder.encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &output_bounds,
            &intermediate_bounds,
            false,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_encode_feedforward_missing_intermediate_bounds() {
        let mut encoder = NetworkEncoder::new();
        // 1 -> 1 -> 1 network (2 layers, 1 ReLU in between)
        let weights = vec![vec![1.0], vec![1.0]];
        let biases = vec![vec![0.0], vec![0.0]];
        let layer_dims = vec![1, 1, 1];
        let input_bounds = vec![Bound::new(-1.0, 1.0)];
        let output_bounds = vec![Bound::new(0.0, 1.0)];
        let intermediate_bounds: Vec<Vec<Bound>> = vec![]; // Missing!

        let result = encoder.encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &output_bounds,
            &intermediate_bounds,
            false,
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SmtError::EncodingError(_)));
    }

    #[test]
    fn test_encode_feedforward_with_bigm() {
        let mut encoder = NetworkEncoder::new().with_big_m(1000.0);

        // Simple 1 -> 1 -> 1 network with ReLU
        let weights = vec![vec![1.0], vec![1.0]];
        let biases = vec![vec![0.5], vec![0.0]];
        let layer_dims = vec![1, 1, 1];
        let input_bounds = vec![Bound::new(-1.0, 1.0)];
        let output_bounds = vec![Bound::new(0.0, 2.0)];
        let intermediate_bounds = vec![vec![Bound::new(-0.5, 1.5)]];

        let network = encoder
            .encode_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
                true, // use_bigm = true
            )
            .unwrap();

        // Should use QF_UFLIA logic for Big-M (integer variables)
        assert!(network.formula.contains("QF_UFLIA"));
        // Should have integer declarations for phase variables
        assert!(network.formula.contains("Int"));
    }

    #[test]
    fn test_encode_feedforward_single_layer_no_relu() {
        let mut encoder = NetworkEncoder::new();

        // Single layer network (just linear, no ReLU)
        let weights = vec![vec![2.0, 3.0]]; // 2 inputs -> 1 output
        let biases = vec![vec![1.0]];
        let layer_dims = vec![2, 1];
        let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
        let output_bounds = vec![Bound::new(0.0, 10.0)];
        let intermediate_bounds: Vec<Vec<Bound>> = vec![]; // No intermediate for single layer

        let network = encoder
            .encode_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &output_bounds,
                &intermediate_bounds,
                false,
            )
            .unwrap();

        assert_eq!(network.num_inputs, 2);
        assert_eq!(network.num_outputs, 1);
        assert_eq!(network.input_vars, vec!["x_0", "x_1"]);
    }

    #[test]
    fn test_smt_network_clone() {
        let network = SmtNetwork {
            formula: "test".to_string(),
            num_inputs: 1,
            num_outputs: 1,
            input_vars: vec!["x".to_string()],
            output_vars: vec!["y".to_string()],
            total_vars: 5,
        };

        let cloned = network.clone();
        assert_eq!(cloned.formula, network.formula);
        assert_eq!(cloned.num_inputs, network.num_inputs);
        assert_eq!(cloned.input_vars, network.input_vars);
    }

    #[test]
    fn test_smt_result_clone() {
        let sat = SmtResult::Sat {
            inputs: vec![1.0],
            outputs: vec![2.0],
        };
        let sat_clone = sat.clone();
        match sat_clone {
            SmtResult::Sat { inputs, outputs } => {
                assert_eq!(inputs, vec![1.0]);
                assert_eq!(outputs, vec![2.0]);
            }
            _ => panic!("Clone failed"),
        }

        let unsat = SmtResult::Unsat { proof: None };
        let unsat_clone = unsat.clone();
        match unsat_clone {
            SmtResult::Unsat { proof } => assert!(proof.is_none()),
            _ => panic!("Clone failed"),
        }
    }
}
