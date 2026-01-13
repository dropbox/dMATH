//! MIRI test harness generation

use crate::error::MiriResult;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for harness generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessConfig {
    /// Generate harness for unsafe blocks
    pub check_unsafe: bool,
    /// Generate harness for raw pointer operations
    pub check_raw_pointers: bool,
    /// Generate harness for slice operations
    pub check_slices: bool,
    /// Generate harness for transmute operations
    pub check_transmute: bool,
    /// Maximum recursion depth for generated tests
    pub max_recursion: usize,
    /// Include exhaustive boundary tests
    pub boundary_tests: bool,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            check_unsafe: true,
            check_raw_pointers: true,
            check_slices: true,
            check_transmute: true,
            max_recursion: 3,
            boundary_tests: true,
        }
    }
}

/// A generated MIRI test harness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiriHarness {
    /// Name of the harness
    pub name: String,
    /// Target function or code being tested
    pub target: String,
    /// Generated test code
    pub code: String,
    /// Description of what the harness tests
    pub description: String,
    /// Test inputs used
    pub inputs: Vec<HarnessInput>,
}

/// An input value for a harness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessInput {
    /// Input name
    pub name: String,
    /// Input type
    pub input_type: String,
    /// Example values to test
    pub values: Vec<String>,
}

/// Generator for MIRI test harnesses
#[derive(Debug)]
pub struct HarnessGenerator {
    config: HarnessConfig,
}

impl HarnessGenerator {
    /// Create a new harness generator
    pub fn new(config: HarnessConfig) -> Self {
        Self { config }
    }

    /// Create a generator with default config
    pub fn default_config() -> Self {
        Self::new(HarnessConfig::default())
    }

    /// Generate a harness for a specific function
    pub fn generate_function_harness(
        &self,
        function_name: &str,
        function_signature: &str,
        module_path: Option<&str>,
    ) -> MiriResult<MiriHarness> {
        let module_prefix = module_path.map(|m| format!("{}::", m)).unwrap_or_default();

        // Parse function signature to determine parameters
        let params = parse_function_params(function_signature)?;
        let inputs = generate_test_inputs(&params, &self.config);

        let test_body = generate_test_body(function_name, &module_prefix, &params, &inputs);

        let code = format!(
            r#"#[test]
fn miri_test_{name}() {{
{body}
}}"#,
            name = sanitize_name(function_name),
            body = test_body,
        );

        Ok(MiriHarness {
            name: format!("miri_test_{}", sanitize_name(function_name)),
            target: format!("{}{}", module_prefix, function_name),
            code,
            description: format!("MIRI test harness for {}", function_name),
            inputs,
        })
    }

    /// Generate harnesses for unsafe blocks in a file
    pub fn generate_unsafe_harnesses(&self, source_code: &str) -> MiriResult<Vec<MiriHarness>> {
        if !self.config.check_unsafe {
            return Ok(Vec::new());
        }

        let mut harnesses = Vec::new();

        // Find unsafe blocks and generate harnesses for them
        let unsafe_blocks = find_unsafe_blocks(source_code);

        for (index, block) in unsafe_blocks.into_iter().enumerate() {
            let harness = self.generate_unsafe_block_harness(index, &block)?;
            harnesses.push(harness);
        }

        Ok(harnesses)
    }

    /// Generate a harness for an unsafe block
    fn generate_unsafe_block_harness(
        &self,
        index: usize,
        block_info: &UnsafeBlockInfo,
    ) -> MiriResult<MiriHarness> {
        let code = format!(
            r#"#[test]
fn miri_test_unsafe_block_{index}() {{
    // Testing unsafe block at line {line}
    // Context: {context}
    unsafe {{
        // Re-execute the unsafe operations under MIRI
        {operations}
    }}
}}"#,
            index = index,
            line = block_info.line_number,
            context = block_info.context,
            operations = block_info.simplified_code,
        );

        Ok(MiriHarness {
            name: format!("miri_test_unsafe_block_{}", index),
            target: format!("unsafe block at line {}", block_info.line_number),
            code,
            description: format!(
                "MIRI test for unsafe block at line {}",
                block_info.line_number
            ),
            inputs: Vec::new(),
        })
    }

    /// Generate a complete test file from multiple harnesses
    pub fn generate_test_file(&self, harnesses: &[MiriHarness], crate_name: &str) -> String {
        let mut code = String::new();

        // Header
        code.push_str("//! Auto-generated MIRI test harnesses\n");
        code.push_str("//! Generated by dashprove-miri\n\n");

        // Import the crate being tested
        code.push_str(&format!("use {}::*;\n\n", crate_name));

        // Generate each harness
        for harness in harnesses {
            code.push_str(&harness.code);
            code.push_str("\n\n");
        }

        code
    }

    /// Generate a harness for raw pointer operations
    pub fn generate_ptr_harness(
        &self,
        ptr_type: &str,
        operations: &[&str],
    ) -> MiriResult<MiriHarness> {
        let ops_code: Vec<String> = operations.iter().map(|op| format!("    {};", op)).collect();

        let code = format!(
            r#"#[test]
fn miri_test_ptr_ops() {{
    let data: {ptr_type} = Default::default();
    let ptr = &data as *const _ as *mut {ptr_type};
    unsafe {{
{operations}
    }}
}}"#,
            ptr_type = ptr_type,
            operations = ops_code.join("\n"),
        );

        Ok(MiriHarness {
            name: "miri_test_ptr_ops".to_string(),
            target: format!("pointer operations on {}", ptr_type),
            code,
            description: "MIRI test for raw pointer operations".to_string(),
            inputs: Vec::new(),
        })
    }

    /// Generate a harness from a code template
    pub fn generate_from_template(&self, template: &str, substitutions: &[(&str, &str)]) -> String {
        let mut result = template.to_string();
        for (key, value) in substitutions {
            result = result.replace(key, value);
        }
        result
    }
}

/// Information about an unsafe block
#[derive(Debug)]
struct UnsafeBlockInfo {
    line_number: usize,
    context: String,
    simplified_code: String,
}

/// Find unsafe blocks in source code
fn find_unsafe_blocks(source: &str) -> Vec<UnsafeBlockInfo> {
    let mut blocks = Vec::new();

    for (line_num, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("unsafe") && (trimmed.contains('{') || trimmed.ends_with("fn")) {
            // Extract context (surrounding code)
            let context = source
                .lines()
                .skip(line_num.saturating_sub(2))
                .take(5)
                .collect::<Vec<_>>()
                .join("\n");

            blocks.push(UnsafeBlockInfo {
                line_number: line_num + 1,
                context: context.chars().take(100).collect(),
                simplified_code: "// TODO: Extract and simplify unsafe operations".to_string(),
            });
        }
    }

    blocks
}

/// Parsed function parameter
#[derive(Debug)]
struct FunctionParam {
    name: String,
    param_type: String,
}

/// Parse function parameters from signature
fn parse_function_params(signature: &str) -> MiriResult<Vec<FunctionParam>> {
    let mut params = Vec::new();

    // Simple regex-based parsing for fn name(params) -> ret
    if let Some(start) = signature.find('(') {
        if let Some(end) = signature.find(')') {
            let params_str = &signature[start + 1..end];

            for param in params_str.split(',') {
                let param = param.trim();
                if param.is_empty() {
                    continue;
                }

                // Handle &self, &mut self, self
                if param.contains("self") {
                    continue;
                }

                // Split on : to get name and type
                if let Some(colon_pos) = param.find(':') {
                    let name = param[..colon_pos].trim().to_string();
                    let param_type = param[colon_pos + 1..].trim().to_string();
                    params.push(FunctionParam { name, param_type });
                }
            }
        }
    }

    Ok(params)
}

/// Generate test inputs for parameters
fn generate_test_inputs(params: &[FunctionParam], config: &HarnessConfig) -> Vec<HarnessInput> {
    params
        .iter()
        .map(|p| HarnessInput {
            name: p.name.clone(),
            input_type: p.param_type.clone(),
            values: generate_values_for_type(&p.param_type, config),
        })
        .collect()
}

/// Generate test values for a type
fn generate_values_for_type(type_name: &str, config: &HarnessConfig) -> Vec<String> {
    let trimmed = type_name.trim();
    let mut values = Vec::new();

    match trimmed {
        "i8" | "i16" | "i32" | "i64" | "isize" => {
            values.push("0".to_string());
            values.push("-1".to_string());
            values.push("1".to_string());
            if config.boundary_tests {
                values.push(format!("{}::MIN", trimmed));
                values.push(format!("{}::MAX", trimmed));
            }
        }
        "u8" | "u16" | "u32" | "u64" | "usize" => {
            values.push("0".to_string());
            values.push("1".to_string());
            if config.boundary_tests {
                values.push(format!("{}::MAX", trimmed));
            }
        }
        "bool" => {
            values.push("true".to_string());
            values.push("false".to_string());
        }
        "f32" | "f64" => {
            values.push("0.0".to_string());
            values.push("1.0".to_string());
            values.push("-1.0".to_string());
            if config.boundary_tests {
                values.push(format!("{}::NAN", trimmed));
                values.push(format!("{}::INFINITY", trimmed));
            }
        }
        s if s.starts_with("&str") || s == "String" => {
            values.push(r#""""#.to_string());
            values.push(r#""test""#.to_string());
        }
        _ => {
            values.push("Default::default()".to_string());
        }
    }

    values
}

/// Generate test body code
fn generate_test_body(
    function_name: &str,
    module_prefix: &str,
    params: &[FunctionParam],
    inputs: &[HarnessInput],
) -> String {
    let mut body = String::new();

    if inputs.is_empty() {
        // No parameters, just call the function
        body.push_str(&format!(
            "    let _ = {}{}();\n",
            module_prefix, function_name
        ));
    } else {
        // Generate nested loops for each parameter
        body.push_str(&generate_nested_test_loops(
            function_name,
            module_prefix,
            params,
            inputs,
            0,
            &[],
        ));
    }

    body
}

/// Generate nested loops for combinatorial testing
fn generate_nested_test_loops(
    function_name: &str,
    module_prefix: &str,
    _params: &[FunctionParam],
    inputs: &[HarnessInput],
    depth: usize,
    current_bindings: &[String],
) -> String {
    if depth >= inputs.len() {
        // All parameters bound, make the call
        let args: Vec<&str> = current_bindings.iter().map(|s| s.as_str()).collect();
        return format!(
            "{}let _ = {}{}({});\n",
            "    ".repeat(depth + 1),
            module_prefix,
            function_name,
            args.join(", ")
        );
    }

    let input = &inputs[depth];
    let var_name = format!("{}_{}", input.name, depth);
    let indent = "    ".repeat(depth + 1);

    let mut code = String::new();
    code.push_str(&format!("{}for {} in [", indent, var_name));
    code.push_str(&input.values.join(", "));
    code.push_str("] {\n");

    let mut new_bindings = current_bindings.to_vec();
    new_bindings.push(var_name);

    code.push_str(&generate_nested_test_loops(
        function_name,
        module_prefix,
        _params,
        inputs,
        depth + 1,
        &new_bindings,
    ));

    code.push_str(&format!("{}}}\n", indent));
    code
}

/// Sanitize a name for use as a Rust identifier
fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

/// Write harness to a file
pub fn write_harness_file(harness: &MiriHarness, output_path: &Path) -> MiriResult<()> {
    std::fs::write(output_path, &harness.code)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_function_params() {
        let sig = "fn add(a: i32, b: i32) -> i32";
        let params = parse_function_params(sig).unwrap();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name, "a");
        assert_eq!(params[0].param_type, "i32");
    }

    #[test]
    fn test_parse_function_params_with_self() {
        let sig = "fn method(&self, x: usize)";
        let params = parse_function_params(sig).unwrap();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, "x");
    }

    #[test]
    fn test_generate_values_for_integers() {
        let config = HarnessConfig::default();
        let values = generate_values_for_type("i32", &config);
        assert!(values.contains(&"0".to_string()));
        assert!(values.contains(&"-1".to_string()));
        assert!(values.contains(&"i32::MIN".to_string()));
    }

    #[test]
    fn test_generate_values_for_bool() {
        let config = HarnessConfig::default();
        let values = generate_values_for_type("bool", &config);
        assert_eq!(values.len(), 2);
        assert!(values.contains(&"true".to_string()));
        assert!(values.contains(&"false".to_string()));
    }

    #[test]
    fn test_sanitize_name() {
        assert_eq!(sanitize_name("foo::bar"), "foo__bar");
        assert_eq!(sanitize_name("test-fn"), "test_fn");
        assert_eq!(sanitize_name("simple"), "simple");
    }

    #[test]
    fn test_generate_function_harness() {
        let generator = HarnessGenerator::default_config();
        let harness = generator
            .generate_function_harness("add", "fn add(a: i32, b: i32) -> i32", None)
            .unwrap();

        assert!(harness.code.contains("fn miri_test_add"));
        assert!(harness.code.contains("#[test]"));
        assert_eq!(harness.inputs.len(), 2);
    }

    #[test]
    fn test_generate_test_file() {
        let generator = HarnessGenerator::default_config();
        let harnesses = vec![
            MiriHarness {
                name: "test1".to_string(),
                target: "fn1".to_string(),
                code: "#[test]\nfn test1() {}".to_string(),
                description: "Test 1".to_string(),
                inputs: Vec::new(),
            },
            MiriHarness {
                name: "test2".to_string(),
                target: "fn2".to_string(),
                code: "#[test]\nfn test2() {}".to_string(),
                description: "Test 2".to_string(),
                inputs: Vec::new(),
            },
        ];

        let file = generator.generate_test_file(&harnesses, "mylib");
        assert!(file.contains("use mylib::*"));
        assert!(file.contains("fn test1()"));
        assert!(file.contains("fn test2()"));
    }

    #[test]
    fn test_generate_unsafe_harnesses_disabled() {
        // Test the !self.config.check_unsafe branch (line 115)
        let generator = HarnessGenerator::new(HarnessConfig {
            check_unsafe: false,
            ..Default::default()
        });
        let harnesses = generator
            .generate_unsafe_harnesses("unsafe { let x = 5; }")
            .unwrap();
        assert!(
            harnesses.is_empty(),
            "Should return empty when check_unsafe is false"
        );
    }

    #[test]
    fn test_generate_unsafe_harnesses_enabled() {
        let generator = HarnessGenerator::new(HarnessConfig {
            check_unsafe: true,
            ..Default::default()
        });
        let source = "fn foo() {\n    unsafe {\n        let x = 5;\n    }\n}";
        let harnesses = generator.generate_unsafe_harnesses(source).unwrap();
        assert!(
            !harnesses.is_empty(),
            "Should generate harnesses when check_unsafe is true"
        );
    }

    #[test]
    fn test_find_unsafe_blocks_line_number() {
        // Test line_num + 1 calculation (line 250)
        let source = "fn foo() {}\nunsafe {\n    let x = 5;\n}\n";
        let blocks = find_unsafe_blocks(source);
        assert_eq!(blocks.len(), 1);
        // Line numbers are 1-indexed: unsafe block is on line 2
        assert_eq!(blocks[0].line_number, 2);
    }

    #[test]
    fn test_find_unsafe_blocks_both_conditions() {
        // Test the && condition: starts_with("unsafe") && (contains('{') || ends_with("fn"))
        let source1 = "unsafe { let x = 5; }"; // unsafe with {
        let blocks1 = find_unsafe_blocks(source1);
        assert!(!blocks1.is_empty(), "Should find unsafe with brace");

        let source2 = "unsafe fn foo() {}"; // unsafe fn
        let blocks2 = find_unsafe_blocks(source2);
        assert!(!blocks2.is_empty(), "Should find unsafe fn");

        let source3 = "let unsafe_var = 5;"; // false positive check - should NOT match
        let blocks3 = find_unsafe_blocks(source3);
        assert!(
            blocks3.is_empty(),
            "Should not match variable named unsafe_var"
        );
    }

    #[test]
    fn test_find_unsafe_blocks_returns_vec() {
        // Test that find_unsafe_blocks returns non-empty Vec (line 236)
        let source = "unsafe { a(); }\nunsafe { b(); }";
        let blocks = find_unsafe_blocks(source);
        assert!(!blocks.is_empty(), "Should return non-empty vec");
        assert_eq!(blocks.len(), 2, "Should find both unsafe blocks");
    }

    #[test]
    fn test_generate_values_for_unsigned_integers() {
        // Test the "u8" | "u16" | "u32" | "u64" | "usize" match arm (line 327)
        let config = HarnessConfig::default();

        let values_u8 = generate_values_for_type("u8", &config);
        assert!(values_u8.contains(&"0".to_string()));
        assert!(values_u8.contains(&"1".to_string()));
        assert!(values_u8.contains(&"u8::MAX".to_string()));

        let values_usize = generate_values_for_type("usize", &config);
        assert!(values_usize.contains(&"0".to_string()));
        assert!(values_usize.contains(&"usize::MAX".to_string()));
    }

    #[test]
    fn test_generate_values_for_floats() {
        // Test the "f32" | "f64" match arm (line 338)
        let config = HarnessConfig::default();

        let values_f32 = generate_values_for_type("f32", &config);
        assert!(values_f32.contains(&"0.0".to_string()));
        assert!(values_f32.contains(&"1.0".to_string()));
        assert!(values_f32.contains(&"-1.0".to_string()));
        assert!(values_f32.contains(&"f32::NAN".to_string()));
        assert!(values_f32.contains(&"f32::INFINITY".to_string()));

        let values_f64 = generate_values_for_type("f64", &config);
        assert!(values_f64.contains(&"f64::NAN".to_string()));
    }

    #[test]
    fn test_generate_values_for_strings() {
        // Test the str/String match (line 347): starts_with("&str") || == "String"
        let config = HarnessConfig::default();

        let values_str = generate_values_for_type("&str", &config);
        assert!(values_str.len() >= 2, "Should have test values for &str");
        assert!(values_str.iter().any(|v| v.contains("\"\"")));

        let values_string = generate_values_for_type("String", &config);
        assert!(
            values_string.len() >= 2,
            "Should have test values for String"
        );
    }

    #[test]
    fn test_generate_test_body_empty_inputs() {
        // Test generate_test_body with no inputs (line 366)
        let body = generate_test_body("my_fn", "mod::", &[], &[]);
        assert!(!body.is_empty(), "Should generate non-empty body");
        assert!(
            body.contains("mod::my_fn()"),
            "Should contain function call"
        );
    }

    #[test]
    fn test_generate_test_body_with_inputs() {
        // Test generate_test_body with inputs
        let params = vec![FunctionParam {
            name: "x".to_string(),
            param_type: "i32".to_string(),
        }];
        let inputs = vec![HarnessInput {
            name: "x".to_string(),
            input_type: "i32".to_string(),
            values: vec!["0".to_string(), "1".to_string()],
        }];
        let body = generate_test_body("my_fn", "", &params, &inputs);
        assert!(!body.is_empty());
        assert!(body.contains("for "), "Should have loop");
    }

    #[test]
    fn test_generate_nested_test_loops_base_case() {
        // Test generate_nested_test_loops when depth >= inputs.len() (line 398)
        let result = generate_nested_test_loops("fn_name", "prefix::", &[], &[], 0, &[]);
        assert!(!result.is_empty());
        assert!(result.contains("prefix::fn_name()"));
    }

    #[test]
    fn test_generate_nested_test_loops_with_depth() {
        let inputs = vec![
            HarnessInput {
                name: "a".to_string(),
                input_type: "i32".to_string(),
                values: vec!["1".to_string(), "2".to_string()],
            },
            HarnessInput {
                name: "b".to_string(),
                input_type: "i32".to_string(),
                values: vec!["3".to_string()],
            },
        ];
        let result = generate_nested_test_loops("test", "", &[], &inputs, 0, &[]);
        assert!(result.contains("for a_0"));
        assert!(result.contains("for b_1"));
    }
}
