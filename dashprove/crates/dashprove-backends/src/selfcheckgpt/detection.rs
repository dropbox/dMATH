//! SelfCheckGPT installation detection

use std::path::PathBuf;
use std::process::Command;

/// Check if SelfCheckGPT is available
pub fn detect_selfcheckgpt() -> Option<PathBuf> {
    let python_cmds = ["python3", "python"];

    for python in python_cmds {
        if let Ok(output) = Command::new(python)
            .args(["-c", "import selfcheckgpt; print('ok')"])
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

/// Get the version of SelfCheckGPT if available
pub fn get_selfcheckgpt_version() -> Option<String> {
    let python_cmds = ["python3", "python"];

    for python in python_cmds {
        if let Ok(output) = Command::new(python)
            .args(["-c", "import selfcheckgpt; print(selfcheckgpt.__version__)"])
            .output()
        {
            if output.status.success() {
                return Some(String::from_utf8_lossy(&output.stdout).trim().to_string());
            }
        }
    }

    None
}
