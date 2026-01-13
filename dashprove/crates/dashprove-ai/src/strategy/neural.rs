//! Neural network primitives for strategy prediction
//!
//! This module provides the building blocks for the neural network used
//! in strategy prediction: dense layers and activation functions.

use serde::{Deserialize, Serialize};

/// Simple neural network layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    /// Weight matrix (input_size x output_size)
    pub weights: Vec<Vec<f64>>,
    /// Bias vector (output_size)
    pub biases: Vec<f64>,
}

impl DenseLayer {
    /// Create a new dense layer with random initialization
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Xavier initialization
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..input_size)
            .map(|i| {
                (0..output_size)
                    .map(|j| {
                        // Deterministic pseudo-random for reproducibility
                        let seed = ((i * 17 + j * 31 + 7) % 997) as f64;
                        ((seed.sin() * 10000.0).fract() - 0.5) * 2.0 * scale
                    })
                    .collect()
            })
            .collect();

        let biases = vec![0.0; output_size];

        DenseLayer { weights, biases }
    }

    /// Forward pass through the layer
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let output_size = self.biases.len();
        let mut output = vec![0.0; output_size];

        for (i, &x) in input.iter().enumerate() {
            if i < self.weights.len() {
                for (j, out_val) in output.iter_mut().enumerate() {
                    if j < self.weights[i].len() {
                        *out_val += x * self.weights[i][j];
                    }
                }
            }
        }

        for (out_val, bias) in output.iter_mut().zip(self.biases.iter()) {
            *out_val += bias;
        }

        output
    }

    /// Forward pass with ReLU activation
    pub fn forward_relu(&self, input: &[f64]) -> Vec<f64> {
        self.forward(input)
            .into_iter()
            .map(|x| x.max(0.0))
            .collect()
    }

    /// Forward pass with softmax activation
    pub fn forward_softmax(&self, input: &[f64]) -> Vec<f64> {
        let output = self.forward(input);
        softmax(&output)
    }

    /// Backward pass through the layer
    ///
    /// Given the gradient of the loss with respect to the output,
    /// computes the gradient with respect to the input and updates weights.
    ///
    /// # Arguments
    /// * `input` - The input that was used in forward pass
    /// * `output_grad` - Gradient of loss w.r.t. layer output
    /// * `learning_rate` - Learning rate for weight updates
    /// * `weight_decay` - L2 regularization coefficient (0.0 for no regularization)
    ///
    /// # Returns
    /// Gradient of loss w.r.t. layer input (for backpropagating to previous layer)
    pub fn backward(
        &mut self,
        input: &[f64],
        output_grad: &[f64],
        learning_rate: f64,
        weight_decay: f64,
    ) -> Vec<f64> {
        let input_size = self.weights.len();

        // Compute gradient w.r.t. input (for previous layer)
        let mut input_grad = vec![0.0; input_size];
        for (i, grad_i) in input_grad.iter_mut().enumerate() {
            for (j, &grad_out) in output_grad.iter().enumerate() {
                if j < self.weights[i].len() {
                    *grad_i += grad_out * self.weights[i][j];
                }
            }
        }

        // Update weights and biases with optional L2 regularization
        for (i, &x) in input.iter().enumerate() {
            if i < self.weights.len() {
                for (j, &grad) in output_grad.iter().enumerate() {
                    if j < self.weights[i].len() {
                        // Gradient descent with weight decay (L2 regularization)
                        let weight_grad = grad * x + weight_decay * self.weights[i][j];
                        self.weights[i][j] -= learning_rate * weight_grad;
                    }
                }
            }
        }

        // Update biases (no weight decay on biases)
        for (bias, &grad) in self.biases.iter_mut().zip(output_grad.iter()) {
            *bias -= learning_rate * grad;
        }

        input_grad
    }
}

/// Softmax function for probability distribution
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x: Vec<f64> = x.iter().map(|&xi| (xi - max_x).exp()).collect();
    let sum: f64 = exp_x.iter().sum();
    exp_x.into_iter().map(|xi| xi / sum).collect()
}

/// ReLU derivative: 1 if x > 0, else 0
/// Applied element-wise to gradient, masking where pre-activation was <= 0
pub fn relu_derivative(pre_activation: &[f64], grad: &[f64]) -> Vec<f64> {
    pre_activation
        .iter()
        .zip(grad.iter())
        .map(|(&pre, &g)| if pre > 0.0 { g } else { 0.0 })
        .collect()
}
