//! LangSmith installation detection

use std::path::PathBuf;
use std::process::Command;

/// Check if LangSmith (langsmith Python package) is available
pub fn detect_langsmith() -> Option<PathBuf> {
    // Try to find Python with langsmith
    let python_cmds = ["python3", "python"];

    for python in python_cmds {
        if let Ok(output) = Command::new(python)
            .args(["-c", "import langsmith; print('ok')"])
            .output()
        {
            if output.status.success() {
                if let Ok(path_output) = Command::new(python)
                    .args(["-c", "import sys; print(sys.executable)"])
                    .output()
                {
                    if path_output.status.success() {
                        let path = String::from_utf8_lossy(&path_output.stdout)
                            .trim()
                            .to_string();
                        return Some(PathBuf::from(path));
                    }
                }
            }
        }
    }

    None
}

/// Get the version of LangSmith if available
pub fn get_langsmith_version() -> Option<String> {
    let python_cmds = ["python3", "python"];

    for python in python_cmds {
        if let Ok(output) = Command::new(python)
            .args(["-c", "import langsmith; print(langsmith.__version__)"])
            .output()
        {
            if output.status.success() {
                return Some(String::from_utf8_lossy(&output.stdout).trim().to_string());
            }
        }
    }

    None
}
