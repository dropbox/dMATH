//! Utility functions for TLA+ output processing

use regex::Regex;

/// Extract error message from TLC output
pub fn extract_error(output: &str) -> String {
    for line in output.lines() {
        if line.contains("Error:") || line.contains("error:") {
            return line.to_string();
        }
    }
    "Unknown error".to_string()
}

/// Extract diagnostics from TLC output
pub fn extract_diagnostics(output: &str) -> Vec<String> {
    let mut diagnostics = Vec::new();

    // Extract statistics
    let stats_re = Regex::new(r"(\d+) states generated.*?(\d+) distinct states").ok();
    if let Some(re) = stats_re {
        if let Some(caps) = re.captures(output) {
            diagnostics.push(format!(
                "States: {} generated, {} distinct",
                caps.get(1).map(|m| m.as_str()).unwrap_or("?"),
                caps.get(2).map(|m| m.as_str()).unwrap_or("?")
            ));
        }
    }

    // Extract time
    let time_re = Regex::new(r"Finished in (\d+)ms").ok();
    if let Some(re) = time_re {
        if let Some(caps) = re.captures(output) {
            diagnostics.push(format!(
                "TLC execution time: {}ms",
                caps.get(1).map(|m| m.as_str()).unwrap_or("?")
            ));
        }
    }

    diagnostics
}

/// Extract coverage percentage from TLC output
pub fn extract_coverage(output: &str) -> Option<f64> {
    // Look for coverage info in TLC output
    let re = Regex::new(r"Coverage:\s*(\d+(?:\.\d+)?)%").ok()?;
    let caps = re.captures(output)?;
    caps.get(1)?.as_str().parse().ok()
}
