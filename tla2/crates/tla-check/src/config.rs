//! TLC Configuration file (.cfg) parser
//!
//! TLC uses configuration files to specify:
//! - INIT: The initial state predicate
//! - NEXT: The next-state relation
//! - INVARIANT: Safety properties to check
//! - PROPERTY: Temporal properties to check
//! - CONSTANT: Constant assignments and model values
//! - SYMMETRY: Symmetry sets for state reduction
//! - CONSTRAINT: State constraints (limits search space)
//! - ACTION_CONSTRAINT: Action constraints
//! - VIEW: Expression to use for state fingerprinting (reduces state space)
//!
//! # Example .cfg file
//!
//! ```text
//! INIT Init
//! NEXT Next
//! INVARIANT TypeOK
//! INVARIANT Safety
//! PROPERTY Liveness
//! CONSTANT N = 3
//! CONSTANT Procs = {p1, p2, p3}
//! CONSTANT Procs <- [ model value ]
//! ```

use std::collections::HashMap;
use std::fmt;

/// A parsed TLC configuration
#[derive(Debug, Clone)]
pub struct Config {
    /// Initial state predicate name
    pub init: Option<String>,
    /// Next-state relation name
    pub next: Option<String>,
    /// Invariants to check (safety properties)
    pub invariants: Vec<String>,
    /// Temporal properties to check
    pub properties: Vec<String>,
    /// Constant assignments (name -> expression string)
    pub constants: HashMap<String, ConstantValue>,
    /// Symmetry set name
    pub symmetry: Option<String>,
    /// State constraints
    pub constraints: Vec<String>,
    /// Action constraints
    pub action_constraints: Vec<String>,
    /// Specification (replaces INIT/NEXT with a temporal formula)
    pub specification: Option<String>,
    /// Whether to check for deadlock (default: true)
    pub check_deadlock: bool,
    /// Whether CHECK_DEADLOCK was explicitly set in the config file.
    /// Used with stuttering semantics: if spec uses `[A]_v` (stuttering allowed)
    /// and CHECK_DEADLOCK was not explicitly set, deadlock checking is disabled.
    pub check_deadlock_explicit: bool,
    /// VIEW expression for fingerprinting (optional state abstraction)
    ///
    /// When set, the fingerprint of a state is computed by evaluating this
    /// expression rather than using all state variables. This can dramatically
    /// reduce state space when some variables don't affect correctness.
    pub view: Option<String>,
    /// Terminal states specification (states that should not be considered deadlocks)
    ///
    /// When set, states matching these conditions are considered intentional
    /// termination points, not deadlocks. This is useful for specs like SAT
    /// solvers where reaching "SAT" or "UNSAT" is a successful termination.
    pub terminal: Option<TerminalSpec>,
}

/// Specification for terminal states (states that are intentionally final)
///
/// Terminal states are states where the spec should not proceed further,
/// but which should NOT be reported as deadlocks.
#[derive(Debug, Clone)]
pub enum TerminalSpec {
    /// Terminal states defined by variable predicates: `TERMINAL state = "SAT"`
    /// Each entry is (variable_name, expected_value_string)
    Predicates(Vec<(String, String)>),
    /// Terminal states defined by a TLA+ operator: `TERMINAL IsTerminal`
    Operator(String),
}

impl Default for Config {
    fn default() -> Self {
        Self {
            init: None,
            next: None,
            invariants: Vec::new(),
            properties: Vec::new(),
            constants: HashMap::new(),
            symmetry: None,
            constraints: Vec::new(),
            action_constraints: Vec::new(),
            specification: None,
            check_deadlock: true, // TLC default is to check for deadlock
            check_deadlock_explicit: false, // Not explicitly set until parsed from config
            view: None,
            terminal: None,
        }
    }
}

/// A constant value in the configuration
#[derive(Debug, Clone, PartialEq)]
pub enum ConstantValue {
    /// Direct value assignment: CONSTANT N = 3
    Value(String),
    /// Model value: CONSTANT Procs <- [ model value ]
    ModelValue,
    /// Set of model values: CONSTANT Procs <- {p1, p2, p3}
    ModelValueSet(Vec<String>),
    /// Replacement operator: CONSTANT Op <- OtherOp
    Replacement(String),
}

impl fmt::Display for ConstantValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstantValue::Value(v) => write!(f, "{}", v),
            ConstantValue::ModelValue => write!(f, "[ model value ]"),
            ConstantValue::ModelValueSet(vs) => {
                write!(f, "{{{}}}", vs.join(", "))
            }
            ConstantValue::Replacement(r) => write!(f, "<- {}", r),
        }
    }
}

/// Configuration parse error
#[derive(Debug, Clone)]
pub struct ConfigError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ConfigError {}

/// Block directive context for multi-line parsing
#[derive(Debug, Clone, Copy, PartialEq)]
enum BlockContext {
    None,
    Init,
    Next,
    Constants,
    Invariants,
    Properties,
    Specification,
    Symmetry,
    Constraints,
    ActionConstraints,
    Alias,
    CheckDeadlock,
    View,
    Terminal,
}

impl Config {
    /// Create a new empty configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a configuration file from a string
    ///
    /// Supports multi-line block syntax for CONSTANTS:
    /// ```text
    /// CONSTANTS
    ///     Name = Value
    ///     Name2 = Value2
    /// ```
    pub fn parse(input: &str) -> Result<Config, Vec<ConfigError>> {
        let mut config = Config::new();
        let mut errors = Vec::new();
        let mut block_context = BlockContext::None;
        let mut pending_init_line: Option<usize> = None;
        let mut pending_next_line: Option<usize> = None;
        let mut check_deadlock_explicit = false;

        for (line_num, raw_line) in input.lines().enumerate() {
            let line_num = line_num + 1; // 1-indexed

            // Strip inline comments (everything after \*)
            let line_without_comment = if let Some(pos) = raw_line.find("\\*") {
                &raw_line[..pos]
            } else {
                raw_line
            };

            // Strip TLA+ block comments (* ... *)
            let line_without_block_comment = strip_block_comments(line_without_comment);
            let line = line_without_block_comment.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with("(*") {
                continue;
            }

            // Check if line starts with a known directive keyword (this ends any block context)
            let is_directive = line.starts_with("INIT")
                || line.starts_with("NEXT")
                || line.starts_with("INVARIANT")
                || line.starts_with("PROPERT")
                || line.starts_with("CONSTANT")
                || line.starts_with("SYMMETRY")
                || line.starts_with("CONSTRAINT")
                || line.starts_with("ACTION_CONSTRAINT")
                || line.starts_with("SPECIFICATION")
                || line.starts_with("CHECK_DEADLOCK")
                || line.starts_with("ALIAS")
                || line.starts_with("VIEW")
                || line.starts_with("TERMINAL");

            // If we were in a block context that requires a value and we hit a new directive,
            // report a missing value error for the original directive line.
            if is_directive {
                if block_context == BlockContext::Init && config.init.is_none() {
                    errors.push(ConfigError {
                        line: pending_init_line.unwrap_or(line_num),
                        message: "INIT requires a predicate name".to_string(),
                    });
                    pending_init_line = None;
                }
                if block_context == BlockContext::Next && config.next.is_none() {
                    errors.push(ConfigError {
                        line: pending_next_line.unwrap_or(line_num),
                        message: "NEXT requires a relation name".to_string(),
                    });
                    pending_next_line = None;
                }
            }

            // Handle non-directive lines within a block context (TLC Toolbox format - no indentation required)
            if !is_directive && block_context != BlockContext::None {
                match block_context {
                    BlockContext::Init => {
                        if !line.is_empty() && config.init.is_none() {
                            config.init = Some(line.to_string());
                            pending_init_line = None;
                        }
                        continue;
                    }
                    BlockContext::Next => {
                        if !line.is_empty() && config.next.is_none() {
                            config.next = Some(line.to_string());
                            pending_next_line = None;
                        }
                        continue;
                    }
                    BlockContext::Constants => {
                        if let Err(e) = parse_constant_assignment(&mut config, line, line_num) {
                            errors.push(e);
                        }
                        continue;
                    }
                    BlockContext::Invariants => {
                        // Each line may have multiple invariant names (space-separated)
                        for name in line.split_whitespace() {
                            config.invariants.push(name.to_string());
                        }
                        continue;
                    }
                    BlockContext::Properties => {
                        // Each line may have multiple property names (space-separated)
                        for name in line.split_whitespace() {
                            config.properties.push(name.to_string());
                        }
                        continue;
                    }
                    BlockContext::Specification => {
                        // SPECIFICATION followed by name
                        if !line.is_empty() && config.specification.is_none() {
                            config.specification = Some(line.to_string());
                        }
                        continue;
                    }
                    BlockContext::Symmetry => {
                        // SYMMETRY followed by name
                        if !line.is_empty() && config.symmetry.is_none() {
                            config.symmetry = Some(line.to_string());
                        }
                        continue;
                    }
                    BlockContext::Constraints => {
                        // Each line may have multiple constraint names (space-separated)
                        for name in line.split_whitespace() {
                            config.constraints.push(name.to_string());
                        }
                        continue;
                    }
                    BlockContext::ActionConstraints => {
                        // Each line may have multiple action constraint names (space-separated)
                        for name in line.split_whitespace() {
                            config.action_constraints.push(name.to_string());
                        }
                        continue;
                    }
                    BlockContext::Alias => {
                        // Alias lines are ignored (used for TLC state output formatting)
                        continue;
                    }
                    BlockContext::CheckDeadlock => {
                        // Handle TRUE/FALSE for CHECK_DEADLOCK block
                        let value = line.to_uppercase();
                        if value == "FALSE" || value == "OFF" || value == "NO" {
                            config.check_deadlock = false;
                        } else if value == "TRUE" || value == "ON" || value == "YES" {
                            config.check_deadlock = true;
                        }
                        continue;
                    }
                    BlockContext::View => {
                        // VIEW followed by expression name
                        if !line.is_empty() && config.view.is_none() {
                            config.view = Some(line.to_string());
                        }
                        continue;
                    }
                    BlockContext::Terminal => {
                        // TERMINAL can be followed by:
                        // - Operator name: "IsTerminal" (single identifier)
                        // - Variable predicate: "state = \"SAT\""
                        if let Some((var, val)) = line.split_once('=') {
                            let var = var.trim().to_string();
                            let val = val.trim().to_string();
                            // Add predicate to existing predicates or create new
                            match &mut config.terminal {
                                Some(TerminalSpec::Predicates(preds)) => {
                                    preds.push((var, val));
                                }
                                _ => {
                                    config.terminal =
                                        Some(TerminalSpec::Predicates(vec![(var, val)]));
                                }
                            }
                        } else if is_valid_cfg_identifier(line) && config.terminal.is_none() {
                            // Single identifier = operator reference
                            config.terminal = Some(TerminalSpec::Operator(line.to_string()));
                        }
                        continue;
                    }
                    BlockContext::None => unreachable!(),
                }
            }

            // Directive line resets block context
            block_context = BlockContext::None;

            // Parse directive
            if let Some(rest) = line.strip_prefix("INIT") {
                let name = rest.trim().to_string();
                if name.is_empty() {
                    // Multi-line block: INIT on its own, value on following indented line
                    block_context = BlockContext::Init;
                    pending_init_line = Some(line_num);
                } else {
                    config.init = Some(name);
                    pending_init_line = None;
                }
            } else if let Some(rest) = line.strip_prefix("NEXT") {
                let name = rest.trim().to_string();
                if name.is_empty() {
                    // Multi-line block: NEXT on its own, value on following indented line
                    block_context = BlockContext::Next;
                    pending_next_line = Some(line_num);
                } else {
                    config.next = Some(name);
                    pending_next_line = None;
                }
            } else if let Some(rest) = line.strip_prefix("SPECIFICATION") {
                let name = rest.trim().to_string();
                if name.is_empty() {
                    // Multi-line block: SPECIFICATION on its own, value on following indented line
                    block_context = BlockContext::Specification;
                } else {
                    config.specification = Some(name);
                }
            } else if let Some(rest) = line
                .strip_prefix("INVARIANTS")
                .or_else(|| line.strip_prefix("INVARIANT"))
            {
                let rest = rest.trim();
                if rest.is_empty() {
                    // Multi-line block: INVARIANTS on its own, values on following indented lines
                    block_context = BlockContext::Invariants;
                } else {
                    // Can be multiple names separated by commas or spaces
                    for part in rest.split(&[',', ' '][..]) {
                        let name = part.trim().to_string();
                        if !name.is_empty() {
                            config.invariants.push(name);
                        }
                    }
                }
            } else if let Some(rest) = line
                .strip_prefix("PROPERTIES")
                .or_else(|| line.strip_prefix("PROPERTY"))
            {
                let rest = rest.trim();
                if rest.is_empty() {
                    // Multi-line block: PROPERTIES on its own, values on following indented lines
                    block_context = BlockContext::Properties;
                } else {
                    // Can be multiple names separated by commas or spaces
                    for part in rest.split(&[',', ' '][..]) {
                        let name = part.trim().to_string();
                        if !name.is_empty() {
                            config.properties.push(name);
                        }
                    }
                }
            } else if let Some(rest) = line
                .strip_prefix("CONSTANTS")
                .or_else(|| line.strip_prefix("CONSTANT"))
            {
                let rest = rest.trim();
                if rest.is_empty() {
                    // Multi-line block: CONSTANTS on its own, values on following indented lines
                    block_context = BlockContext::Constants;
                } else {
                    // Single-line format: CONSTANT Name = Value
                    // Also set block_context so subsequent indented lines are parsed as constants
                    if let Err(e) = parse_constant_assignment(&mut config, rest, line_num) {
                        errors.push(e);
                    }
                    block_context = BlockContext::Constants;
                }
            } else if let Some(rest) = line.strip_prefix("SYMMETRY") {
                let name = rest.trim().to_string();
                if name.is_empty() {
                    // Multi-line block: SYMMETRY on its own, name on following indented line
                    block_context = BlockContext::Symmetry;
                } else {
                    config.symmetry = Some(name);
                }
            } else if let Some(rest) = line
                .strip_prefix("CONSTRAINTS")
                .or_else(|| line.strip_prefix("CONSTRAINT"))
            {
                let rest = rest.trim();
                if rest.is_empty() {
                    // Multi-line block: CONSTRAINTS on its own, names on following indented lines
                    block_context = BlockContext::Constraints;
                } else {
                    for name in rest.split(',') {
                        let name = name.trim().to_string();
                        if !name.is_empty() {
                            config.constraints.push(name);
                        }
                    }
                }
            } else if let Some(rest) = line
                .strip_prefix("ACTION_CONSTRAINTS")
                .or_else(|| line.strip_prefix("ACTION_CONSTRAINT"))
            {
                let rest = rest.trim();
                if rest.is_empty() {
                    // Multi-line block: ACTION_CONSTRAINTS on its own, names on following indented lines
                    block_context = BlockContext::ActionConstraints;
                } else {
                    for name in rest.split(',') {
                        let name = name.trim().to_string();
                        if !name.is_empty() {
                            config.action_constraints.push(name);
                        }
                    }
                }
            } else if line.starts_with("ALIAS") {
                // ALIAS is used by TLC to rename variables in state output.
                // We ignore it since it doesn't affect model checking semantics.
                // Can be single-line (ALIAS Alias) or multi-line block.
                block_context = BlockContext::Alias;
            } else if let Some(rest) = line.strip_prefix("CHECK_DEADLOCK") {
                // CHECK_DEADLOCK can be followed by TRUE/FALSE or on separate line
                check_deadlock_explicit = true;
                let value = rest.trim().to_uppercase();
                if value == "FALSE" || value == "OFF" || value == "NO" {
                    config.check_deadlock = false;
                } else if value == "TRUE" || value == "ON" || value == "YES" {
                    config.check_deadlock = true;
                } else if value.is_empty() {
                    // Multi-line format: CHECK_DEADLOCK followed by indented TRUE/FALSE
                    block_context = BlockContext::CheckDeadlock;
                }
            } else if let Some(rest) = line.strip_prefix("VIEW") {
                // VIEW specifies an expression to use for fingerprinting instead of the full state
                let name = rest.trim().to_string();
                if name.is_empty() {
                    // Multi-line format: VIEW followed by expression name on next line
                    block_context = BlockContext::View;
                } else {
                    config.view = Some(name);
                }
            } else if let Some(rest) = line.strip_prefix("TERMINAL") {
                // TERMINAL specifies states that should not be considered deadlocks
                // Syntax options:
                //   TERMINAL IsTerminal    (operator reference)
                //   TERMINAL state = "SAT" (variable predicate, single-line)
                //   TERMINAL               (multi-line block)
                //       state = "SAT"
                //       state = "UNSAT"
                let rest = rest.trim();
                if rest.is_empty() {
                    // Multi-line block: TERMINAL on its own
                    block_context = BlockContext::Terminal;
                } else if let Some((var, val)) = rest.split_once('=') {
                    // Single-line predicate: TERMINAL state = "SAT"
                    let var = var.trim().to_string();
                    let val = val.trim().to_string();
                    config.terminal = Some(TerminalSpec::Predicates(vec![(var, val)]));
                    block_context = BlockContext::Terminal; // Allow more predicates on following lines
                } else if is_valid_cfg_identifier(rest) {
                    // Operator reference: TERMINAL IsTerminal
                    config.terminal = Some(TerminalSpec::Operator(rest.to_string()));
                } else {
                    errors.push(ConfigError {
                        line: line_num,
                        message: format!("Invalid TERMINAL syntax: {}", rest),
                    });
                }
            } else {
                errors.push(ConfigError {
                    line: line_num,
                    message: format!("Unknown directive: {}", line),
                });
            }
        }

        // If the file ends while still waiting for a block value, report it.
        if config.init.is_none() {
            if let Some(line) = pending_init_line {
                errors.push(ConfigError {
                    line,
                    message: "INIT requires a predicate name".to_string(),
                });
            }
        }
        if config.next.is_none() {
            if let Some(line) = pending_next_line {
                errors.push(ConfigError {
                    line,
                    message: "NEXT requires a relation name".to_string(),
                });
            }
        }

        // TLC does not report deadlock as an error while checking temporal properties
        // (PROPERTY/PROPERTIES) unless explicitly enabled.
        if !check_deadlock_explicit && !config.properties.is_empty() {
            config.check_deadlock = false;
        }

        // Store whether CHECK_DEADLOCK was explicitly set in the config.
        // This is used for stuttering semantics: when a spec uses [A]_v form (allowing stuttering),
        // deadlock checking is disabled unless explicitly requested via CHECK_DEADLOCK TRUE.
        config.check_deadlock_explicit = check_deadlock_explicit;

        if errors.is_empty() {
            Ok(config)
        } else {
            Err(errors)
        }
    }

    /// Write the configuration to TLC .cfg format
    pub fn to_cfg_string(&self) -> String {
        let mut out = String::new();

        if let Some(ref spec) = self.specification {
            out.push_str(&format!("SPECIFICATION {}\n", spec));
        }

        if let Some(ref init) = self.init {
            out.push_str(&format!("INIT {}\n", init));
        }

        if let Some(ref next) = self.next {
            out.push_str(&format!("NEXT {}\n", next));
        }

        for inv in &self.invariants {
            out.push_str(&format!("INVARIANT {}\n", inv));
        }

        for prop in &self.properties {
            out.push_str(&format!("PROPERTY {}\n", prop));
        }

        for (name, value) in &self.constants {
            match value {
                ConstantValue::Value(v) => {
                    out.push_str(&format!("CONSTANT {} = {}\n", name, v));
                }
                ConstantValue::ModelValue => {
                    out.push_str(&format!("CONSTANT {} <- [ model value ]\n", name));
                }
                ConstantValue::ModelValueSet(vs) => {
                    out.push_str(&format!("CONSTANT {} <- {{{}}}\n", name, vs.join(", ")));
                }
                ConstantValue::Replacement(r) => {
                    out.push_str(&format!("CONSTANT {} <- {}\n", name, r));
                }
            }
        }

        if let Some(ref sym) = self.symmetry {
            out.push_str(&format!("SYMMETRY {}\n", sym));
        }

        for c in &self.constraints {
            out.push_str(&format!("CONSTRAINT {}\n", c));
        }

        for ac in &self.action_constraints {
            out.push_str(&format!("ACTION_CONSTRAINT {}\n", ac));
        }

        if let Some(ref view) = self.view {
            out.push_str(&format!("VIEW {}\n", view));
        }

        out
    }
}

/// Strip TLA+ block comments (* ... *) from a line.
/// Handles multiple comments and nested comments are NOT supported (matches TLC behavior).
fn strip_block_comments(line: &str) -> String {
    let mut result = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut in_comment = false;

    while let Some(c) = chars.next() {
        if in_comment {
            // Look for closing *)
            if c == '*' && chars.peek() == Some(&')') {
                chars.next(); // consume ')'
                in_comment = false;
            }
            // Skip character while in comment
        } else {
            // Look for opening (*
            if c == '(' && chars.peek() == Some(&'*') {
                chars.next(); // consume '*'
                in_comment = true;
            } else {
                result.push(c);
            }
        }
    }

    result
}

/// Parse a constant assignment line
///
/// Formats:
/// - `Name = Value` - direct value assignment
/// - `Name <- [ model value ]` - model value
/// - `Name <- {a, b, c}` - set of model values
/// - `Name <- OtherOp` - replacement operator
/// - `Name <-[Module] OtherOp` - module-scoped replacement (parsed but scope ignored for now)
/// - `a1=a1  a2=a2  a3=a3` - multiple space-separated model values (TLC Toolbox format)
fn parse_constant_assignment(
    config: &mut Config,
    line: &str,
    line_num: usize,
) -> Result<(), ConfigError> {
    // If line contains `<-`, it's a single replacement-style assignment
    if let Some((name, replacement)) = line.split_once("<-") {
        let name = name.trim().to_string();
        let replacement = replacement.trim();

        // Check for module-scoped substitution: `Name <-[Module] Value`
        // This syntax is used by TLC for INSTANCE substitutions
        // For now, we extract the actual replacement value and ignore the module scope
        // (the global substitution usually handles this case)
        let replacement = if replacement.starts_with('[')
            && !replacement.starts_with("[ model value")
            && !replacement.starts_with("[model value")
        {
            // Find the closing bracket and extract the value after it
            if let Some(close_bracket) = replacement.find(']') {
                let after_bracket = replacement[close_bracket + 1..].trim();
                if !after_bracket.is_empty() {
                    // This is module-scoped: `[Module] Value`
                    // Extract just the Value part
                    after_bracket
                } else {
                    // This is a function/record expression, not module-scoped
                    replacement
                }
            } else {
                replacement
            }
        } else {
            replacement
        };

        if replacement == "[ model value ]" || replacement == "[model value]" {
            config.constants.insert(name, ConstantValue::ModelValue);
        } else if replacement.starts_with('{') && replacement.ends_with('}') {
            // Set of model values
            let inner = &replacement[1..replacement.len() - 1];
            let values: Vec<String> = inner
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            config
                .constants
                .insert(name, ConstantValue::ModelValueSet(values));
        } else {
            config
                .constants
                .insert(name, ConstantValue::Replacement(replacement.to_string()));
        }
        return Ok(());
    }

    // Check if this is multiple space-separated assignments like "a1=a1  a2=a2  a3=a3"
    // These are model value assignments without spaces around the equals sign
    let parts: Vec<&str> = line.split_whitespace().collect();
    let all_simple_assignments = parts.iter().all(|part| {
        // Each part should be of the form "name=value" without spaces
        if let Some((n, v)) = part.split_once('=') {
            is_valid_cfg_identifier(n.trim()) && !v.is_empty()
        } else {
            false
        }
    });

    if all_simple_assignments && !parts.is_empty() {
        // Parse each simple assignment
        for part in parts {
            if let Some((name, value)) = part.split_once('=') {
                let name = name.trim().to_string();
                let value = value.trim().to_string();
                config.constants.insert(name, ConstantValue::Value(value));
            }
        }
        return Ok(());
    }

    // Single assignment with equals sign and potentially spaces around it
    if let Some((name, value)) = line.split_once('=') {
        let name = name.trim().to_string();
        let value = value.trim().to_string();
        config.constants.insert(name, ConstantValue::Value(value));
        Ok(())
    } else {
        Err(ConfigError {
            line: line_num,
            message: format!("Invalid CONSTANT syntax: {}", line),
        })
    }
}

/// Check if string is a valid TLA+ identifier for config files
fn is_valid_cfg_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !first.is_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_config() {
        let input = r#"
INIT Init
NEXT Next
INVARIANT TypeOK
INVARIANT Safety
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.init, Some("Init".to_string()));
        assert_eq!(config.next, Some("Next".to_string()));
        assert_eq!(config.invariants, vec!["TypeOK", "Safety"]);
    }

    #[test]
    fn test_parse_init_next_block_syntax() {
        let input = r#"
INIT
    Init
NEXT
    Next
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.init, Some("Init".to_string()));
        assert_eq!(config.next, Some("Next".to_string()));
    }

    #[test]
    fn test_parse_with_constants() {
        let input = r#"
INIT Init
NEXT Next
CONSTANT N = 3
CONSTANT Procs <- {p1, p2, p3}
CONSTANT Server <- [ model value ]
"#;
        let config = Config::parse(input).unwrap();
        assert!(matches!(
            config.constants.get("N"),
            Some(ConstantValue::Value(v)) if v == "3"
        ));
        assert!(matches!(
            config.constants.get("Procs"),
            Some(ConstantValue::ModelValueSet(vs)) if vs == &vec!["p1", "p2", "p3"]
        ));
        assert!(matches!(
            config.constants.get("Server"),
            Some(ConstantValue::ModelValue)
        ));
    }

    #[test]
    fn test_parse_multiple_invariants_on_one_line() {
        let input = "INVARIANT TypeOK, Safety, Liveness\n";
        let config = Config::parse(input).unwrap();
        assert_eq!(config.invariants, vec!["TypeOK", "Safety", "Liveness"]);
    }

    #[test]
    fn test_parse_specification() {
        let input = "SPECIFICATION Spec\n";
        let config = Config::parse(input).unwrap();
        assert_eq!(config.specification, Some("Spec".to_string()));
    }

    #[test]
    fn test_parse_with_comments() {
        let input = r#"
\* This is a comment
INIT Init
(* Another comment *)
NEXT Next
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.init, Some("Init".to_string()));
        assert_eq!(config.next, Some("Next".to_string()));
    }

    #[test]
    fn test_roundtrip() {
        let mut config = Config::new();
        config.init = Some("Init".to_string());
        config.next = Some("Next".to_string());
        config.invariants = vec!["TypeOK".to_string()];
        config
            .constants
            .insert("N".to_string(), ConstantValue::Value("3".to_string()));

        let output = config.to_cfg_string();
        let parsed = Config::parse(&output).unwrap();

        assert_eq!(parsed.init, config.init);
        assert_eq!(parsed.next, config.next);
        assert_eq!(parsed.invariants, config.invariants);
    }

    #[test]
    fn test_parse_error() {
        let input = "INIT\n"; // Missing name
        let result = Config::parse(input);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("requires"));
    }

    #[test]
    fn test_parse_constraints() {
        let input = r#"
CONSTRAINT Bound
ACTION_CONSTRAINT NoStutter
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.constraints, vec!["Bound"]);
        assert_eq!(config.action_constraints, vec!["NoStutter"]);
    }

    #[test]
    fn test_parse_multiline_constants() {
        // This is the format used by CigaretteSmokers.cfg and many others
        let input = r#"
CONSTANTS
  Ingredients = {matches, paper, tobacco}
  Offers = {{matches, paper}, {matches, tobacco}, {paper, tobacco}}
INVARIANTS TypeOK AtMostOne
SPECIFICATION Spec
"#;
        let config = Config::parse(input).unwrap();
        assert!(matches!(
            config.constants.get("Ingredients"),
            Some(ConstantValue::Value(v)) if v == "{matches, paper, tobacco}"
        ));
        assert!(matches!(
            config.constants.get("Offers"),
            Some(ConstantValue::Value(v)) if v == "{{matches, paper}, {matches, tobacco}, {paper, tobacco}}"
        ));
        assert_eq!(config.invariants, vec!["TypeOK", "AtMostOne"]);
        assert_eq!(config.specification, Some("Spec".to_string()));
    }

    #[test]
    fn test_parse_multiline_constants_with_tabs() {
        let input = "CONSTANTS\n\tN = 3\n\tM = 5\nINIT Init\n";
        let config = Config::parse(input).unwrap();
        assert!(matches!(
            config.constants.get("N"),
            Some(ConstantValue::Value(v)) if v == "3"
        ));
        assert!(matches!(
            config.constants.get("M"),
            Some(ConstantValue::Value(v)) if v == "5"
        ));
        assert_eq!(config.init, Some("Init".to_string()));
    }

    #[test]
    fn test_parse_model_value_same_name() {
        // NoVal = NoVal pattern (model value with same name)
        let input = r#"
CONSTANTS
    NoVal = NoVal
"#;
        let config = Config::parse(input).unwrap();
        // This is parsed as a value assignment, but TLC treats it as model value
        assert!(matches!(
            config.constants.get("NoVal"),
            Some(ConstantValue::Value(v)) if v == "NoVal"
        ));
    }

    #[test]
    fn test_parse_multiline_invariants() {
        // SlidingPuzzles.cfg format
        let input = r#"
INIT Init
NEXT Next
INVARIANTS
  TypeOK
  KlotskiGoal
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.init, Some("Init".to_string()));
        assert_eq!(config.next, Some("Next".to_string()));
        assert_eq!(config.invariants, vec!["TypeOK", "KlotskiGoal"]);
    }

    #[test]
    fn test_parse_multiline_properties() {
        let input = r#"
SPECIFICATION Spec
PROPERTIES
  Liveness
  Fairness
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.specification, Some("Spec".to_string()));
        assert_eq!(config.properties, vec!["Liveness", "Fairness"]);
    }

    #[test]
    fn test_parse_check_deadlock_false() {
        let input = "CHECK_DEADLOCK FALSE\n";
        let config = Config::parse(input).unwrap();
        assert!(!config.check_deadlock);
    }

    #[test]
    fn test_parse_check_deadlock_true() {
        let input = "CHECK_DEADLOCK TRUE\n";
        let config = Config::parse(input).unwrap();
        assert!(config.check_deadlock);
    }

    #[test]
    fn test_properties_disable_deadlock_by_default() {
        let input = r#"
SPECIFICATION Spec
PROPERTY Liveness
"#;
        let config = Config::parse(input).unwrap();
        assert!(!config.check_deadlock);
    }

    #[test]
    fn test_properties_do_not_override_explicit_check_deadlock() {
        let input = r#"
SPECIFICATION Spec
PROPERTY Liveness
CHECK_DEADLOCK TRUE
"#;
        let config = Config::parse(input).unwrap();
        assert!(config.check_deadlock);
    }

    #[test]
    fn test_default_check_deadlock() {
        let config = Config::new();
        assert!(config.check_deadlock); // Default is true
    }

    #[test]
    fn test_parse_check_deadlock_multiline_false() {
        // EWD998.cfg format
        let input = "CHECK_DEADLOCK\n    FALSE\n";
        let config = Config::parse(input).unwrap();
        assert!(!config.check_deadlock);
    }

    #[test]
    fn test_parse_check_deadlock_multiline_true() {
        let input = "CHECK_DEADLOCK\n    TRUE\n";
        let config = Config::parse(input).unwrap();
        assert!(config.check_deadlock);
    }

    #[test]
    fn test_parse_multiline_specification() {
        // CoffeeCan100Beans.cfg format
        let input = r#"
CONSTANTS
    MaxBeanCount = 100

SPECIFICATION
    Spec

PROPERTY
    EventuallyTerminates
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.specification, Some("Spec".to_string()));
        assert_eq!(config.properties, vec!["EventuallyTerminates"]);
        assert!(matches!(
            config.constants.get("MaxBeanCount"),
            Some(ConstantValue::Value(v)) if v == "100"
        ));
    }

    #[test]
    fn test_parse_multiline_symmetry() {
        // MCKVsnap.cfg format
        let input = r#"
CONSTANTS
    Key = {k1, k2}
    TxId = {t1, t2, t3}

SYMMETRY
    TxIdSymmetric

SPECIFICATION
    Spec
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.symmetry, Some("TxIdSymmetric".to_string()));
        assert_eq!(config.specification, Some("Spec".to_string()));
        assert!(matches!(
            config.constants.get("Key"),
            Some(ConstantValue::Value(v)) if v == "{k1, k2}"
        ));
    }

    #[test]
    fn test_parse_multiline_constraints() {
        let input = r#"
CONSTRAINTS
    Bound1
    Bound2

ACTION_CONSTRAINTS
    NoStutter
    FairAction
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.constraints, vec!["Bound1", "Bound2"]);
        assert_eq!(config.action_constraints, vec!["NoStutter", "FairAction"]);
    }

    #[test]
    fn test_parse_space_separated_invariants() {
        // SimpleAllocator.cfg format - multiple invariants on single line
        let input = r#"
SPECIFICATION Spec
INVARIANTS
  TypeInvariant ResourceMutex
PROPERTIES
  Liveness Fairness
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.invariants, vec!["TypeInvariant", "ResourceMutex"]);
        assert_eq!(config.properties, vec!["Liveness", "Fairness"]);
    }

    #[test]
    fn test_parse_view() {
        let input = "VIEW ViewExpr\n";
        let config = Config::parse(input).unwrap();
        assert_eq!(config.view, Some("ViewExpr".to_string()));
    }

    #[test]
    fn test_parse_view_multiline() {
        let input = r#"
VIEW
    ViewExpr
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.view, Some("ViewExpr".to_string()));
    }

    #[test]
    fn test_parse_view_with_other_directives() {
        let input = r#"
INIT Init
NEXT Next
VIEW StateView
INVARIANT TypeOK
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.init, Some("Init".to_string()));
        assert_eq!(config.next, Some("Next".to_string()));
        assert_eq!(config.view, Some("StateView".to_string()));
        assert_eq!(config.invariants, vec!["TypeOK"]);
    }

    #[test]
    fn test_default_view_none() {
        let config = Config::new();
        assert!(config.view.is_none());
    }

    #[test]
    fn test_to_cfg_string_with_view() {
        let mut config = Config::new();
        config.init = Some("Init".to_string());
        config.view = Some("ViewExpr".to_string());

        let output = config.to_cfg_string();
        assert!(output.contains("VIEW ViewExpr"));
    }

    // ==================== TERMINAL directive tests ====================

    #[test]
    fn test_parse_terminal_operator() {
        let input = "TERMINAL IsTerminal\n";
        let config = Config::parse(input).unwrap();
        assert!(matches!(
            config.terminal,
            Some(TerminalSpec::Operator(ref op)) if op == "IsTerminal"
        ));
    }

    #[test]
    fn test_parse_terminal_single_predicate() {
        let input = "TERMINAL state = \"SAT\"\n";
        let config = Config::parse(input).unwrap();
        match config.terminal {
            Some(TerminalSpec::Predicates(preds)) => {
                assert_eq!(preds.len(), 1);
                assert_eq!(preds[0].0, "state");
                assert_eq!(preds[0].1, "\"SAT\"");
            }
            _ => panic!("Expected predicates"),
        }
    }

    #[test]
    fn test_parse_terminal_multiline_predicates() {
        let input = r#"
TERMINAL
    state = "SAT"
    state = "UNSAT"
"#;
        let config = Config::parse(input).unwrap();
        match config.terminal {
            Some(TerminalSpec::Predicates(preds)) => {
                assert_eq!(preds.len(), 2);
                assert_eq!(preds[0].0, "state");
                assert_eq!(preds[0].1, "\"SAT\"");
                assert_eq!(preds[1].0, "state");
                assert_eq!(preds[1].1, "\"UNSAT\"");
            }
            _ => panic!("Expected predicates"),
        }
    }

    #[test]
    fn test_parse_terminal_multiline_operator() {
        let input = r#"
TERMINAL
    IsTerminal
"#;
        let config = Config::parse(input).unwrap();
        assert!(matches!(
            config.terminal,
            Some(TerminalSpec::Operator(ref op)) if op == "IsTerminal"
        ));
    }

    #[test]
    fn test_parse_terminal_with_other_directives() {
        let input = r#"
INIT Init
NEXT Next
TERMINAL state = "SAT"
INVARIANT TypeOK
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.init, Some("Init".to_string()));
        assert_eq!(config.next, Some("Next".to_string()));
        assert!(matches!(config.terminal, Some(TerminalSpec::Predicates(_))));
        assert_eq!(config.invariants, vec!["TypeOK"]);
    }

    #[test]
    fn test_default_terminal_none() {
        let config = Config::new();
        assert!(config.terminal.is_none());
    }

    #[test]
    fn test_parse_inline_comments() {
        // MCVoting.cfg style - inline comments after values
        let input = r#"
SPECIFICATION Spec         \* MCSpec
INVARIANT Inv              \* MCInv
PROPERTY ConsensusSpecBar
CHECK_DEADLOCK FALSE
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(config.specification, Some("Spec".to_string()));
        assert_eq!(config.invariants, vec!["Inv"]);
        assert_eq!(config.properties, vec!["ConsensusSpecBar"]);
        assert!(!config.check_deadlock);
    }

    #[test]
    fn test_parse_space_separated_model_values() {
        // MCVoting.cfg style - multiple model values on one line
        let input = r#"
CONSTANTS
  a1=a1  a2=a2  a3=a3  v1=v1  v2=v2
  Acceptor <- MCAcceptor
SPECIFICATION Spec
"#;
        let config = Config::parse(input).unwrap();
        assert_eq!(
            config.constants.get("a1"),
            Some(&ConstantValue::Value("a1".to_string()))
        );
        assert_eq!(
            config.constants.get("a2"),
            Some(&ConstantValue::Value("a2".to_string()))
        );
        assert_eq!(
            config.constants.get("v1"),
            Some(&ConstantValue::Value("v1".to_string()))
        );
        assert_eq!(
            config.constants.get("Acceptor"),
            Some(&ConstantValue::Replacement("MCAcceptor".to_string()))
        );
        assert_eq!(config.specification, Some("Spec".to_string()));
    }
}
