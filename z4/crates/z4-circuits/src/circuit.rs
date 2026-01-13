//! Boolean circuit representation.
//!
//! A Boolean circuit is a directed acyclic graph (DAG) where:
//! - Source nodes are inputs (variables)
//! - Internal nodes are gates (AND, OR, NOT, etc.)
//! - One or more nodes are designated as outputs

use std::fmt;

/// Index of a gate in the circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GateId(pub usize);

impl fmt::Display for GateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "g{}", self.0)
    }
}

/// A Boolean gate in the circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gate {
    /// Input variable (index 0..num_inputs-1)
    Input(usize),
    /// Constant true
    True,
    /// Constant false
    False,
    /// Logical NOT
    Not(GateId),
    /// Binary AND
    And(GateId, GateId),
    /// Binary OR
    Or(GateId, GateId),
    /// Binary XOR
    Xor(GateId, GateId),
    /// N-ary AND (unbounded fan-in, for AC0)
    NaryAnd(Vec<GateId>),
    /// N-ary OR (unbounded fan-in, for AC0)
    NaryOr(Vec<GateId>),
    /// Threshold gate: output 1 iff at least `threshold` inputs are 1 (for TC0)
    Threshold {
        inputs: Vec<GateId>,
        threshold: usize,
    },
    /// Mod gate: output 1 iff (sum of inputs) mod modulus == 0 (for ACC0)
    Mod { inputs: Vec<GateId>, modulus: usize },
}

impl Gate {
    /// Returns the input GateIds this gate depends on.
    pub fn inputs(&self) -> Vec<GateId> {
        match self {
            Gate::Input(_) | Gate::True | Gate::False => vec![],
            Gate::Not(a) => vec![*a],
            Gate::And(a, b) | Gate::Or(a, b) | Gate::Xor(a, b) => vec![*a, *b],
            Gate::NaryAnd(inputs) | Gate::NaryOr(inputs) => inputs.clone(),
            Gate::Threshold { inputs, .. } | Gate::Mod { inputs, .. } => inputs.clone(),
        }
    }

    /// Returns true if this is an input gate.
    pub fn is_input(&self) -> bool {
        matches!(self, Gate::Input(_))
    }

    /// Returns true if this is a constant gate.
    pub fn is_constant(&self) -> bool {
        matches!(self, Gate::True | Gate::False)
    }

    /// Returns the fan-in (number of inputs) of this gate.
    pub fn fan_in(&self) -> usize {
        match self {
            Gate::Input(_) | Gate::True | Gate::False => 0,
            Gate::Not(_) => 1,
            Gate::And(_, _) | Gate::Or(_, _) | Gate::Xor(_, _) => 2,
            Gate::NaryAnd(inputs) | Gate::NaryOr(inputs) => inputs.len(),
            Gate::Threshold { inputs, .. } | Gate::Mod { inputs, .. } => inputs.len(),
        }
    }

    /// Evaluate the gate given input values.
    pub fn evaluate(&self, values: &[bool]) -> bool {
        match self {
            Gate::Input(i) => values[*i],
            Gate::True => true,
            Gate::False => false,
            Gate::Not(a) => !values[a.0],
            Gate::And(a, b) => values[a.0] && values[b.0],
            Gate::Or(a, b) => values[a.0] || values[b.0],
            Gate::Xor(a, b) => values[a.0] ^ values[b.0],
            Gate::NaryAnd(inputs) => inputs.iter().all(|g| values[g.0]),
            Gate::NaryOr(inputs) => inputs.iter().any(|g| values[g.0]),
            Gate::Threshold { inputs, threshold } => {
                let count: usize = inputs.iter().filter(|g| values[g.0]).count();
                count >= *threshold
            }
            Gate::Mod { inputs, modulus } => {
                let sum: usize = inputs.iter().filter(|g| values[g.0]).count();
                sum.is_multiple_of(*modulus)
            }
        }
    }
}

/// A Boolean circuit represented as a DAG of gates.
#[derive(Debug, Clone)]
pub struct Circuit {
    /// Number of input variables
    num_inputs: usize,
    /// All gates in the circuit (including inputs)
    gates: Vec<Gate>,
    /// Output gate IDs
    outputs: Vec<GateId>,
}

impl Circuit {
    /// Create a new circuit with the given number of inputs.
    ///
    /// Input gates are automatically created as gates 0..num_inputs-1.
    pub fn new(num_inputs: usize) -> Self {
        let mut gates = Vec::with_capacity(num_inputs);
        for i in 0..num_inputs {
            gates.push(Gate::Input(i));
        }
        Circuit {
            num_inputs,
            gates,
            outputs: vec![],
        }
    }

    /// Get the number of input variables.
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Get the total number of gates (including inputs).
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    /// Get the gate ID for input variable i.
    pub fn input(&self, i: usize) -> GateId {
        assert!(i < self.num_inputs, "Input index out of bounds");
        GateId(i)
    }

    /// Add a gate to the circuit and return its ID.
    pub fn add_gate(&mut self, gate: Gate) -> GateId {
        // Validate that all referenced gates exist
        for input in gate.inputs() {
            assert!(
                input.0 < self.gates.len(),
                "Gate references non-existent gate {}",
                input
            );
        }
        let id = GateId(self.gates.len());
        self.gates.push(gate);
        id
    }

    /// Set the output of the circuit (single output).
    pub fn set_output(&mut self, gate: GateId) {
        assert!(gate.0 < self.gates.len(), "Output gate does not exist");
        self.outputs = vec![gate];
    }

    /// Set multiple outputs for the circuit.
    pub fn set_outputs(&mut self, gates: Vec<GateId>) {
        for gate in &gates {
            assert!(gate.0 < self.gates.len(), "Output gate does not exist");
        }
        self.outputs = gates;
    }

    /// Get the output gate IDs.
    pub fn outputs(&self) -> &[GateId] {
        &self.outputs
    }

    /// Get a gate by its ID.
    pub fn gate(&self, id: GateId) -> &Gate {
        &self.gates[id.0]
    }

    /// Get all gates in the circuit.
    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    /// Evaluate the circuit on the given input assignment.
    /// Returns the values of all output gates.
    pub fn evaluate(&self, inputs: &[bool]) -> Vec<bool> {
        assert_eq!(
            inputs.len(),
            self.num_inputs,
            "Wrong number of inputs: expected {}, got {}",
            self.num_inputs,
            inputs.len()
        );

        // Evaluate all gates in topological order (they're already ordered)
        let mut values = vec![false; self.gates.len()];
        for (i, gate) in self.gates.iter().enumerate() {
            values[i] = gate.evaluate(&values);
            // For input gates, override with actual input
            if let Gate::Input(idx) = gate {
                values[i] = inputs[*idx];
            }
        }

        // Return output values
        self.outputs.iter().map(|g| values[g.0]).collect()
    }

    /// Evaluate the circuit for a single output (assumes single output circuit).
    pub fn evaluate_single(&self, inputs: &[bool]) -> bool {
        let outputs = self.evaluate(inputs);
        assert_eq!(outputs.len(), 1, "Circuit has multiple outputs");
        outputs[0]
    }

    /// Create a circuit that computes the constant function.
    pub fn constant(num_inputs: usize, value: bool) -> Self {
        let mut circuit = Circuit::new(num_inputs);
        let gate = if value { Gate::True } else { Gate::False };
        let id = circuit.add_gate(gate);
        circuit.set_output(id);
        circuit
    }

    /// Create a circuit that returns the value of input i.
    pub fn identity(num_inputs: usize, input_idx: usize) -> Self {
        assert!(input_idx < num_inputs);
        let mut circuit = Circuit::new(num_inputs);
        circuit.set_output(circuit.input(input_idx));
        circuit
    }

    /// Create a circuit that computes NOT of input i.
    pub fn not_input(num_inputs: usize, input_idx: usize) -> Self {
        assert!(input_idx < num_inputs);
        let mut circuit = Circuit::new(num_inputs);
        let inp = circuit.input(input_idx);
        let not = circuit.add_gate(Gate::Not(inp));
        circuit.set_output(not);
        circuit
    }

    /// Create a circuit that computes AND of two inputs.
    pub fn and2(num_inputs: usize, a: usize, b: usize) -> Self {
        assert!(a < num_inputs && b < num_inputs);
        let mut circuit = Circuit::new(num_inputs);
        let ga = circuit.input(a);
        let gb = circuit.input(b);
        let and = circuit.add_gate(Gate::And(ga, gb));
        circuit.set_output(and);
        circuit
    }

    /// Create a circuit that computes OR of two inputs.
    pub fn or2(num_inputs: usize, a: usize, b: usize) -> Self {
        assert!(a < num_inputs && b < num_inputs);
        let mut circuit = Circuit::new(num_inputs);
        let ga = circuit.input(a);
        let gb = circuit.input(b);
        let or = circuit.add_gate(Gate::Or(ga, gb));
        circuit.set_output(or);
        circuit
    }

    /// Create a circuit that computes XOR of two inputs.
    pub fn xor2(num_inputs: usize, a: usize, b: usize) -> Self {
        assert!(a < num_inputs && b < num_inputs);
        let mut circuit = Circuit::new(num_inputs);
        let ga = circuit.input(a);
        let gb = circuit.input(b);
        let xor = circuit.add_gate(Gate::Xor(ga, gb));
        circuit.set_output(xor);
        circuit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_circuit() {
        let c = Circuit::constant(2, true);
        assert!(c.evaluate_single(&[false, false]));
        assert!(c.evaluate_single(&[true, true]));

        let c = Circuit::constant(2, false);
        assert!(!c.evaluate_single(&[false, false]));
        assert!(!c.evaluate_single(&[true, true]));
    }

    #[test]
    fn test_identity_circuit() {
        let c = Circuit::identity(2, 0);
        assert!(!c.evaluate_single(&[false, true]));
        assert!(c.evaluate_single(&[true, false]));
    }

    #[test]
    fn test_not_circuit() {
        let c = Circuit::not_input(2, 0);
        assert!(c.evaluate_single(&[false, true]));
        assert!(!c.evaluate_single(&[true, false]));
    }

    #[test]
    fn test_and_circuit() {
        let c = Circuit::and2(2, 0, 1);
        assert!(!c.evaluate_single(&[false, false]));
        assert!(!c.evaluate_single(&[true, false]));
        assert!(!c.evaluate_single(&[false, true]));
        assert!(c.evaluate_single(&[true, true]));
    }

    #[test]
    fn test_or_circuit() {
        let c = Circuit::or2(2, 0, 1);
        assert!(!c.evaluate_single(&[false, false]));
        assert!(c.evaluate_single(&[true, false]));
        assert!(c.evaluate_single(&[false, true]));
        assert!(c.evaluate_single(&[true, true]));
    }

    #[test]
    fn test_xor_circuit() {
        let c = Circuit::xor2(2, 0, 1);
        assert!(!c.evaluate_single(&[false, false]));
        assert!(c.evaluate_single(&[true, false]));
        assert!(c.evaluate_single(&[false, true]));
        assert!(!c.evaluate_single(&[true, true]));
    }

    #[test]
    fn test_complex_circuit() {
        // Build (x AND y) OR (NOT x AND NOT y) = XNOR
        let mut c = Circuit::new(2);
        let x = c.input(0);
        let y = c.input(1);
        let not_x = c.add_gate(Gate::Not(x));
        let not_y = c.add_gate(Gate::Not(y));
        let x_and_y = c.add_gate(Gate::And(x, y));
        let not_x_and_not_y = c.add_gate(Gate::And(not_x, not_y));
        let xnor = c.add_gate(Gate::Or(x_and_y, not_x_and_not_y));
        c.set_output(xnor);

        assert!(c.evaluate_single(&[false, false])); // XNOR(0,0) = 1
        assert!(!c.evaluate_single(&[true, false])); // XNOR(1,0) = 0
        assert!(!c.evaluate_single(&[false, true])); // XNOR(0,1) = 0
        assert!(c.evaluate_single(&[true, true])); // XNOR(1,1) = 1
    }

    #[test]
    fn test_threshold_gate() {
        // Majority function: at least 2 of 3 inputs must be true
        let mut c = Circuit::new(3);
        let inputs = vec![c.input(0), c.input(1), c.input(2)];
        let maj = c.add_gate(Gate::Threshold {
            inputs,
            threshold: 2,
        });
        c.set_output(maj);

        assert!(!c.evaluate_single(&[false, false, false])); // 0
        assert!(!c.evaluate_single(&[true, false, false])); // 1
        assert!(!c.evaluate_single(&[false, true, false])); // 1
        assert!(c.evaluate_single(&[true, true, false])); // 2
        assert!(!c.evaluate_single(&[false, false, true])); // 1
        assert!(c.evaluate_single(&[true, false, true])); // 2
        assert!(c.evaluate_single(&[false, true, true])); // 2
        assert!(c.evaluate_single(&[true, true, true])); // 3
    }

    #[test]
    fn test_mod_gate() {
        // MOD3: true iff number of 1s is divisible by 3
        let mut c = Circuit::new(3);
        let inputs = vec![c.input(0), c.input(1), c.input(2)];
        let mod3 = c.add_gate(Gate::Mod { inputs, modulus: 3 });
        c.set_output(mod3);

        assert!(c.evaluate_single(&[false, false, false])); // 0 mod 3 = 0
        assert!(!c.evaluate_single(&[true, false, false])); // 1 mod 3 = 1
        assert!(!c.evaluate_single(&[true, true, false])); // 2 mod 3 = 2
        assert!(c.evaluate_single(&[true, true, true])); // 3 mod 3 = 0
    }

    #[test]
    fn test_nary_and() {
        let mut c = Circuit::new(3);
        let inputs = vec![c.input(0), c.input(1), c.input(2)];
        let and3 = c.add_gate(Gate::NaryAnd(inputs));
        c.set_output(and3);

        assert!(!c.evaluate_single(&[false, false, false]));
        assert!(!c.evaluate_single(&[true, true, false]));
        assert!(c.evaluate_single(&[true, true, true]));
    }

    #[test]
    fn test_nary_or() {
        let mut c = Circuit::new(3);
        let inputs = vec![c.input(0), c.input(1), c.input(2)];
        let or3 = c.add_gate(Gate::NaryOr(inputs));
        c.set_output(or3);

        assert!(!c.evaluate_single(&[false, false, false]));
        assert!(c.evaluate_single(&[true, false, false]));
        assert!(c.evaluate_single(&[true, true, true]));
    }
}
