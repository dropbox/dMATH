//! Lake configuration parsing
//!
//! Parses lakefile.lean to extract package configuration.

use crate::error::{LakeError, LakeResult};
use crate::manifest::{LakeManifest, ManifestPackage};
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Lake configuration loaded from lakefile.lean
#[derive(Debug, Clone, Default)]
pub struct LakeConfig {
    /// Package configuration
    pub package: PackageConfig,
    /// Lean libraries
    pub libs: Vec<LeanLib>,
    /// Lean executables
    pub exes: Vec<LeanExe>,
    /// Lean tests
    pub tests: Vec<LeanTest>,
    /// Lake scripts
    pub scripts: Vec<LakeScript>,
    /// Default targets
    pub default_targets: Vec<String>,
}

/// Package configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PackageConfig {
    /// Package name
    pub name: String,
    /// Package version (optional)
    pub version: Option<String>,
    /// Package description (optional)
    pub description: Option<String>,
    /// Package dependencies
    pub dependencies: Vec<Dependency>,
    /// Source directory (defaults to ".")
    pub src_dir: Option<PathBuf>,
    /// Build directory (defaults to ".lake/build")
    pub build_dir: Option<PathBuf>,
    /// Lean version constraint
    pub lean_version: Option<String>,
    /// Extra compiler options
    pub more_lean_args: Vec<String>,
    /// Extra linker options
    pub more_link_args: Vec<String>,
}

/// A package dependency
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Dependency {
    /// Dependency name
    pub name: String,
    /// Git repository URL
    pub git: Option<String>,
    /// Git revision (commit, branch, or tag)
    pub rev: Option<String>,
    /// Path to local dependency
    pub path: Option<PathBuf>,
    /// Version constraint
    pub version: Option<String>,
}

/// Lean library configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LeanLib {
    /// Library name
    pub name: String,
    /// Root modules (e.g., #[`MyLib, `MyLib.Extra])
    pub roots: Vec<String>,
    /// Glob patterns for source files
    pub globs: Vec<String>,
    /// Extra Lean arguments
    pub more_lean_args: Vec<String>,
    /// Whether this is the default target
    pub default_facets: Vec<String>,
    /// Source directory override
    pub src_dir: Option<PathBuf>,
    /// Pre-compilation hooks
    pub pre_compile_hooks: Vec<String>,
}

/// Lean executable configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LeanExe {
    /// Executable name
    pub name: String,
    /// Root module
    pub root: String,
    /// Extra Lean arguments
    pub more_lean_args: Vec<String>,
    /// Extra linker arguments
    pub more_link_args: Vec<String>,
    /// Source directory override
    pub src_dir: Option<PathBuf>,
    /// Support backends
    pub supported_backends: Vec<String>,
}

/// Lean test configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LeanTest {
    /// Test name
    pub name: String,
    /// Root module
    pub root: String,
    /// Extra Lean arguments
    pub more_lean_args: Vec<String>,
    /// Source directory override
    pub src_dir: Option<PathBuf>,
}

/// Lake script configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LakeScript {
    /// Script name
    pub name: String,
    /// Script body (Lean code to execute)
    pub body: String,
    /// Script documentation
    pub doc: Option<String>,
}

impl LakeConfig {
    /// Load configuration from a lakefile.lean file
    pub fn load(lakefile_path: &Path) -> LakeResult<Self> {
        if !lakefile_path.exists() {
            return Err(LakeError::LakefileNotFound(lakefile_path.to_path_buf()));
        }

        let content = std::fs::read_to_string(lakefile_path)?;
        Self::parse(&content)
    }

    /// Alias for load - load configuration from a lakefile.lean file
    pub fn from_file(lakefile_path: &Path) -> LakeResult<Self> {
        Self::load(lakefile_path)
    }

    /// Parse lakefile.lean content
    pub fn parse(content: &str) -> LakeResult<Self> {
        let mut config = LakeConfig::default();

        // Parse Lean surface syntax and extract DSL constructs
        // This is a simplified parser - a full implementation would use lean5-parser

        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;
        // Track if the next declaration should be marked as default target
        let mut next_is_default = false;

        while i < lines.len() {
            let line = lines[i].trim();

            // Skip comments and empty lines
            if line.is_empty()
                || line.starts_with("--")
                || line.starts_with("import")
                || line.starts_with("open ")
            {
                i += 1;
                continue;
            }

            if let Some(dep) = Self::parse_require_line(line) {
                config.package.dependencies.push(dep);
                i += 1;
                continue;
            }

            // Parse package declaration
            if line.starts_with("package ") {
                let rest = line
                    .strip_prefix("package ")
                    .expect("prefix checked by starts_with")
                    .trim();
                // Handle "package name where" or "package name"
                let name = if let Some(idx) = rest.find(" where") {
                    &rest[..idx]
                } else {
                    rest.trim()
                };
                config.package.name = name.to_string();

                // Parse package block if present
                if rest.contains("where") {
                    i += 1;
                    while i < lines.len() {
                        let block_line = lines[i].trim();
                        if block_line.is_empty() || block_line.starts_with("--") {
                            i += 1;
                            continue;
                        }
                        // Check for next top-level declaration
                        if !block_line.starts_with("--")
                            && (block_line.starts_with("@[")
                                || block_line.starts_with("lean_lib")
                                || block_line.starts_with("lean_exe")
                                || block_line.starts_with("lean_test")
                                || block_line.starts_with("script")
                                || block_line.starts_with("package"))
                        {
                            break;
                        }
                        // Parse package fields
                        Self::parse_package_field(&mut config.package, block_line);
                        i += 1;
                    }
                    continue;
                }
                i += 1;
                continue;
            }

            // Parse @[default_target] attribute - affects next declaration
            if line.starts_with("@[default_target]") {
                next_is_default = true;
                i += 1;
                continue;
            }

            // Parse lean_lib declaration
            if line.starts_with("lean_lib ") {
                let rest = line
                    .strip_prefix("lean_lib ")
                    .expect("prefix checked by starts_with")
                    .trim();
                let name = Self::parse_decl_name(rest);

                let mut lib = LeanLib {
                    name: name.to_string(),
                    ..Default::default()
                };

                // Parse lib block if present
                if rest.contains("where") {
                    i += 1;
                    while i < lines.len() {
                        let block_line = lines[i].trim();
                        if block_line.is_empty() || block_line.starts_with("--") {
                            i += 1;
                            continue;
                        }
                        // Check for next top-level declaration
                        if !block_line.starts_with("--")
                            && (block_line.starts_with("@[")
                                || block_line.starts_with("lean_lib")
                                || block_line.starts_with("lean_exe")
                                || block_line.starts_with("lean_test")
                                || block_line.starts_with("script")
                                || block_line.starts_with("package"))
                        {
                            break;
                        }
                        Self::parse_lib_field(&mut lib, block_line);
                        i += 1;
                    }
                } else {
                    i += 1;
                }

                if next_is_default {
                    config.default_targets.push(lib.name.clone());
                    next_is_default = false;
                }
                config.libs.push(lib);
                continue;
            }

            // Parse lean_exe declaration
            if line.starts_with("lean_exe ") {
                let rest = line
                    .strip_prefix("lean_exe ")
                    .expect("prefix checked by starts_with")
                    .trim();
                let name = Self::parse_decl_name(rest);

                let mut exe = LeanExe {
                    name: name.to_string(),
                    ..Default::default()
                };

                // Parse exe block if present
                if rest.contains("where") {
                    i += 1;
                    while i < lines.len() {
                        let block_line = lines[i].trim();
                        if block_line.is_empty() || block_line.starts_with("--") {
                            i += 1;
                            continue;
                        }
                        // Check for next top-level declaration
                        if !block_line.starts_with("--")
                            && (block_line.starts_with("@[")
                                || block_line.starts_with("lean_lib")
                                || block_line.starts_with("lean_exe")
                                || block_line.starts_with("lean_test")
                                || block_line.starts_with("script")
                                || block_line.starts_with("package"))
                        {
                            break;
                        }
                        Self::parse_exe_field(&mut exe, block_line);
                        i += 1;
                    }
                } else {
                    i += 1;
                }

                if next_is_default {
                    config.default_targets.push(exe.name.clone());
                    next_is_default = false;
                }
                config.exes.push(exe);
                continue;
            }

            // Parse lean_test declaration
            if line.starts_with("lean_test ") {
                let rest = line
                    .strip_prefix("lean_test ")
                    .expect("prefix checked by starts_with")
                    .trim();
                let name = if let Some(idx) = rest.find(" where") {
                    &rest[..idx]
                } else {
                    rest.trim()
                };

                let mut test = LeanTest {
                    name: name.to_string(),
                    ..Default::default()
                };

                // Parse test block if present
                if rest.contains("where") {
                    i += 1;
                    while i < lines.len() {
                        let block_line = lines[i].trim();
                        if block_line.is_empty() || block_line.starts_with("--") {
                            i += 1;
                            continue;
                        }
                        // Check for next top-level declaration
                        if !block_line.starts_with("--")
                            && (block_line.starts_with("@[")
                                || block_line.starts_with("lean_lib")
                                || block_line.starts_with("lean_exe")
                                || block_line.starts_with("lean_test")
                                || block_line.starts_with("package"))
                        {
                            break;
                        }
                        Self::parse_test_field(&mut test, block_line);
                        i += 1;
                    }
                } else {
                    i += 1;
                }

                config.tests.push(test);
                continue;
            }

            // Parse script declaration
            if line.starts_with("script ") {
                let rest = line
                    .strip_prefix("script ")
                    .expect("prefix checked by starts_with")
                    .trim();
                let name = if let Some(idx) = rest.find(" where") {
                    &rest[..idx]
                } else if let Some(idx) = rest.find(" :=") {
                    &rest[..idx]
                } else {
                    rest.trim()
                };

                let mut script = LakeScript {
                    name: name.to_string(),
                    ..Default::default()
                };

                // Parse script block if present
                if rest.contains("where") || rest.contains(":=") {
                    i += 1;
                    let mut body_lines = Vec::new();
                    while i < lines.len() {
                        let block_line = lines[i].trim();
                        if block_line.is_empty() {
                            i += 1;
                            continue;
                        }
                        // Check for next top-level declaration
                        if !block_line.starts_with("--")
                            && (block_line.starts_with("@[")
                                || block_line.starts_with("lean_lib")
                                || block_line.starts_with("lean_exe")
                                || block_line.starts_with("lean_test")
                                || block_line.starts_with("script")
                                || block_line.starts_with("package"))
                        {
                            break;
                        }
                        // Collect script body
                        if block_line.starts_with("doc :=") {
                            script.doc = Some(
                                block_line
                                    .strip_prefix("doc :=")
                                    .expect("prefix checked by starts_with")
                                    .trim()
                                    .trim_matches('"')
                                    .to_string(),
                            );
                        } else if !block_line.starts_with("--") {
                            body_lines.push(block_line.to_string());
                        }
                        i += 1;
                    }
                    script.body = body_lines.join("\n");
                } else {
                    i += 1;
                }

                config.scripts.push(script);
                continue;
            }

            i += 1;
        }

        // Validate configuration
        if config.package.name.is_empty() {
            return Err(LakeError::MissingField {
                field: "name".to_string(),
                context: "package".to_string(),
            });
        }

        Ok(config)
    }

    /// Parse a field in the package block
    fn parse_package_field(package: &mut PackageConfig, line: &str) {
        if let Some(rest) = line.strip_prefix("version :=") {
            package.version = Some(rest.trim().trim_matches('"').to_string());
        } else if let Some(rest) = line.strip_prefix("description :=") {
            package.description = Some(rest.trim().trim_matches('"').to_string());
        } else if let Some(rest) = line.strip_prefix("srcDir :=") {
            package.src_dir = Some(PathBuf::from(rest.trim().trim_matches('"')));
        } else if let Some(rest) = line.strip_prefix("buildDir :=") {
            package.build_dir = Some(PathBuf::from(rest.trim().trim_matches('"')));
        } else if let Some(rest) = line.strip_prefix("leanVersion :=") {
            package.lean_version = Some(rest.trim().trim_matches('"').to_string());
        } else if let Some(rest) = line.strip_prefix("dependencies :=") {
            let deps = Self::parse_dependency_array(rest.trim());
            package.dependencies.extend(deps);
        } else if let Some(rest) = line.strip_prefix("moreLeanArgs :=") {
            package.more_lean_args = Self::parse_string_array(rest.trim());
        } else if let Some(rest) = line.strip_prefix("moreLinkArgs :=") {
            package.more_link_args = Self::parse_string_array(rest.trim());
        }
    }

    /// Parse a field in a lean_lib block
    fn parse_lib_field(lib: &mut LeanLib, line: &str) {
        if let Some(rest) = line.strip_prefix("roots :=") {
            // Parse #[`Module1, `Module2] syntax
            lib.roots = Self::parse_name_array(rest.trim());
        } else if let Some(rest) = line.strip_prefix("globs :=") {
            lib.globs = Self::parse_string_array(rest.trim());
        } else if let Some(rest) = line.strip_prefix("srcDir :=") {
            lib.src_dir = Some(PathBuf::from(rest.trim().trim_matches('"')));
        }
    }

    /// Parse a field in a lean_exe block
    fn parse_exe_field(exe: &mut LeanExe, line: &str) {
        if let Some(rest) = line.strip_prefix("root :=") {
            exe.root = rest.trim().trim_start_matches('`').to_string();
        } else if let Some(rest) = line.strip_prefix("srcDir :=") {
            exe.src_dir = Some(PathBuf::from(rest.trim().trim_matches('"')));
        }
    }

    /// Parse a field in a lean_test block
    fn parse_test_field(test: &mut LeanTest, line: &str) {
        if let Some(rest) = line.strip_prefix("root :=") {
            test.root = rest.trim().trim_start_matches('`').to_string();
        } else if let Some(rest) = line.strip_prefix("srcDir :=") {
            test.src_dir = Some(PathBuf::from(rest.trim().trim_matches('"')));
        }
    }

    /// Parse a Lean name array like #[`Name1, `Name2]
    fn parse_name_array(s: &str) -> Vec<String> {
        let s = s.trim();
        if !s.starts_with("#[") || !s.ends_with(']') {
            // Single value
            return vec![s.trim_start_matches('`').to_string()];
        }
        let inner = &s[2..s.len() - 1];
        inner
            .split(',')
            .map(|part| part.trim().trim_start_matches('`').to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Parse a string array like #["str1", "str2"]
    fn parse_string_array(s: &str) -> Vec<String> {
        let s = s.trim();
        if !s.starts_with("#[") || !s.ends_with(']') {
            return vec![s.trim_matches('"').to_string()];
        }
        let inner = &s[2..s.len() - 1];
        inner
            .split(',')
            .map(|part| part.trim().trim_matches('"').to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Get the source directory for the package
    #[must_use]
    pub fn src_dir(&self) -> PathBuf {
        self.package
            .src_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// Get the build directory for the package
    #[must_use]
    pub fn build_dir(&self) -> PathBuf {
        self.package
            .build_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(".lake/build"))
    }

    /// Validate that declared dependencies are satisfied by the manifest
    pub fn validate_manifest(&self, manifest: &LakeManifest) -> LakeResult<()> {
        self.package.validate_manifest(manifest)
    }
}

impl PackageConfig {
    /// Create a minimal package configuration
    #[must_use]
    pub fn minimal(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Validate that all declared dependencies are satisfied by the manifest
    pub fn validate_manifest(&self, manifest: &LakeManifest) -> LakeResult<()> {
        for dep in &self.dependencies {
            let pkg =
                manifest
                    .get_package(&dep.name)
                    .ok_or_else(|| LakeError::DependencyNotFound {
                        name: dep.name.clone(),
                    })?;
            dep.matches_manifest(pkg)?;
        }
        Ok(())
    }
}

impl LeanLib {
    /// Create a minimal library configuration
    #[must_use]
    pub fn minimal(name: &str) -> Self {
        Self {
            name: name.to_string(),
            roots: vec![name.to_string()],
            ..Default::default()
        }
    }

    /// Get the root modules for this library
    #[must_use]
    pub fn root_modules(&self) -> Vec<String> {
        if self.roots.is_empty() {
            vec![self.name.clone()]
        } else {
            self.roots.clone()
        }
    }
}

impl LeanExe {
    /// Create a minimal executable configuration
    #[must_use]
    pub fn minimal(name: &str, root: &str) -> Self {
        Self {
            name: name.to_string(),
            root: root.to_string(),
            ..Default::default()
        }
    }
}

impl LeanTest {
    /// Create a minimal test configuration
    #[must_use]
    pub fn minimal(name: &str, root: &str) -> Self {
        Self {
            name: name.to_string(),
            root: root.to_string(),
            ..Default::default()
        }
    }
}

impl Dependency {
    /// Check if this dependency is satisfied by the manifest entry
    pub fn matches_manifest(&self, pkg: &ManifestPackage) -> LakeResult<()> {
        match pkg {
            ManifestPackage::Git(g) => {
                if let Some(path) = &self.path {
                    return Err(LakeError::DependencyMismatch {
                        name: self.name.clone(),
                        reason: format!(
                            "expected path dependency at {}, found git",
                            path.display()
                        ),
                    });
                }

                if let Some(dep_url) = &self.git {
                    if Self::normalize_git_url(dep_url) != Self::normalize_git_url(&g.url) {
                        return Err(LakeError::DependencyMismatch {
                            name: self.name.clone(),
                            reason: format!(
                                "git URL mismatch (expected {}, found {})",
                                dep_url, g.url
                            ),
                        });
                    }
                }

                if let Some(req_rev) = &self.rev {
                    if !(g.rev.starts_with(req_rev) || req_rev.starts_with(&g.rev)) {
                        return Err(LakeError::DependencyMismatch {
                            name: self.name.clone(),
                            reason: format!(
                                "revision mismatch (required {}, manifest has {})",
                                req_rev, g.rev
                            ),
                        });
                    }
                }

                if let Some(version_req) = &self.version {
                    let req = VersionReq::parse(version_req).map_err(|e| {
                        LakeError::InvalidConfig(format!(
                            "invalid version requirement for {}: {}",
                            self.name, e
                        ))
                    })?;

                    let version =
                        Version::parse(&g.rev).map_err(|_| LakeError::DependencyMismatch {
                            name: self.name.clone(),
                            reason: format!(
                                "manifest revision '{}' is not a semantic version",
                                g.rev
                            ),
                        })?;

                    if !req.matches(&version) {
                        return Err(LakeError::DependencyMismatch {
                            name: self.name.clone(),
                            reason: format!(
                                "version constraint {version_req} not satisfied by {version}"
                            ),
                        });
                    }
                }

                Ok(())
            }
            ManifestPackage::Path(p) => {
                if self.git.is_some() || self.rev.is_some() {
                    return Err(LakeError::DependencyMismatch {
                        name: self.name.clone(),
                        reason: "expected git dependency, manifest has path".to_string(),
                    });
                }
                if let Some(expected) = &self.path {
                    if Path::new(&p.path) != expected.as_path() {
                        return Err(LakeError::DependencyMismatch {
                            name: self.name.clone(),
                            reason: format!(
                                "path mismatch (expected {}, found {})",
                                expected.display(),
                                p.path
                            ),
                        });
                    }
                }
                if self.version.is_some() {
                    return Err(LakeError::DependencyMismatch {
                        name: self.name.clone(),
                        reason: "version constraints are not supported for path dependencies"
                            .to_string(),
                    });
                }
                Ok(())
            }
        }
    }

    fn normalize_git_url(url: &str) -> String {
        url.trim_end_matches(".git")
            .trim_end_matches('/')
            .to_string()
    }
}

/// Tokenize whitespace-separated tokens, preserving quoted strings
#[allow(dead_code)] // May be useful for future parsing needs
fn tokenize_quoted(input: &str) -> Vec<String> {
    let mut tokens = vec![];
    let mut current = String::new();
    let mut in_quote = false;

    for c in input.chars() {
        match c {
            '"' => {
                in_quote = !in_quote;
            }
            ch if ch.is_whitespace() && !in_quote => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(c),
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// Tokenize a require line, treating "/" and "@" as separate tokens
fn tokenize_require(input: &str) -> Vec<String> {
    let mut tokens = vec![];
    let mut current = String::new();
    let mut in_quote = false;

    for c in input.chars() {
        match c {
            '"' => {
                in_quote = !in_quote;
                current.push(c);
            }
            '/' if !in_quote => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push("/".to_string());
            }
            '@' if !in_quote => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push("@".to_string());
            }
            ch if ch.is_whitespace() && !in_quote => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(c),
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

impl LakeConfig {
    fn parse_require_line(line: &str) -> Option<Dependency> {
        if !line.starts_with("require ") {
            return None;
        }

        let rest = line.strip_prefix("require ")?.trim();
        // Stop at "with" clause or line comment
        let rest = rest.split(" with ").next()?.trim();
        let rest = rest.split(" -- ").next()?.trim();
        let tokens = tokenize_require(rest);
        if tokens.is_empty() {
            return None;
        }

        let mut iter = tokens.into_iter().peekable();
        let mut dep = Dependency::default();

        // Parse the name: either "owner" / "repo" format or just name
        let first = iter.next()?;

        // Check if it's the new "owner" / "repo" format
        if iter.peek().map(String::as_str) == Some("/") {
            // New format: "owner" / "repo"
            let owner = first.trim_matches('"');
            iter.next(); // consume "/"
            let repo_str = iter.next()?;
            let repo = repo_str.trim_matches('"');
            dep.name = repo.to_string();
            // Construct github URL from owner/repo
            dep.git = Some(format!("https://github.com/{owner}/{repo}"));
        } else {
            // Old format: just name
            dep.name = first;
        }

        while let Some(tok) = iter.next() {
            match tok.as_str() {
                "from" => {
                    if let Some(source) = iter.next() {
                        match source.as_str() {
                            "git" => {
                                dep.git = iter.next().map(|s| s.trim_matches('"').to_string());
                            }
                            "path" => {
                                dep.path = iter.next().map(|s| PathBuf::from(s.trim_matches('"')));
                            }
                            _ => {}
                        }
                    }
                }
                "@" => {
                    // New format: @ git "branch" or old format @ "version"
                    if let Some(val) = iter.next() {
                        if val == "git" {
                            // @ git "branch" format (new Mathlib style)
                            if let Some(branch) = iter.next() {
                                dep.rev = Some(branch.trim_matches('"').to_string());
                            }
                        } else {
                            // @ "version" or @ "rev" format
                            let cleaned = val.trim_matches('"');
                            if VersionReq::parse(cleaned).is_ok() {
                                dep.version = Some(cleaned.to_string());
                            } else {
                                dep.rev = Some(cleaned.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Some(dep)
    }

    /// Extract name from lean_lib/lean_exe declaration, handling guillemet names
    fn parse_decl_name(rest: &str) -> &str {
        let rest = rest.trim();
        // Handle guillemet names: «name» -> name
        // « is U+00AB (2 bytes in UTF-8), » is U+00BB (2 bytes in UTF-8)
        if rest.starts_with('«') {
            if let Some(end_char_idx) = rest.find('»') {
                // Get byte index after the opening «
                let start_byte = '«'.len_utf8();
                return &rest[start_byte..end_char_idx];
            }
        }
        // Regular name extraction
        if let Some(idx) = rest.find(" where") {
            &rest[..idx]
        } else {
            rest.split_whitespace()
                .next()
                .unwrap_or_else(|| rest.trim())
        }
    }

    fn parse_dependency_array(spec: &str) -> Vec<Dependency> {
        // Very small parser for #[{ name := "...", git := "...", rev := "..." }, ...]
        // Primarily for tests and simple lakefile usage
        let mut deps = vec![];
        let trimmed = spec.trim();
        if !trimmed.starts_with("#[") || !trimmed.ends_with(']') {
            return deps;
        }

        let inner = &trimmed[2..trimmed.len() - 1];
        for entry in inner.split("},") {
            let mut dep = Dependency::default();
            for field in entry.split(',') {
                let kv: Vec<_> = field.split(":=").map(str::trim).collect();
                if kv.len() != 2 {
                    continue;
                }
                match kv[0] {
                    "name" => dep.name = kv[1].trim_matches('"').to_string(),
                    "git" => dep.git = Some(kv[1].trim_matches('"').to_string()),
                    "rev" => dep.rev = Some(kv[1].trim_matches('"').to_string()),
                    "path" => dep.path = Some(PathBuf::from(kv[1].trim_matches('"'))),
                    "version" => dep.version = Some(kv[1].trim_matches('"').to_string()),
                    _ => {}
                }
            }
            if !dep.name.is_empty() {
                deps.push(dep);
            }
        }

        deps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::GitPackage;

    #[test]
    fn test_parse_minimal_lakefile() {
        let content = r"
import Lake
open Lake DSL

package test
lean_lib Test
";
        let config = LakeConfig::parse(content).unwrap();
        assert_eq!(config.package.name, "test");
        assert_eq!(config.libs.len(), 1);
        assert_eq!(config.libs[0].name, "Test");
    }

    #[test]
    fn test_parse_lakefile_with_where() {
        let content = r#"
import Lake
open Lake DSL

package myproject where
  version := "0.1.0"

@[default_target]
lean_lib MyLib where
  roots := #[`MyLib]

lean_exe myexe where
  root := `Main
"#;
        let config = LakeConfig::parse(content).unwrap();
        assert_eq!(config.package.name, "myproject");
        assert_eq!(config.package.version, Some("0.1.0".to_string()));
        assert_eq!(config.libs.len(), 1);
        assert_eq!(config.libs[0].name, "MyLib");
        assert_eq!(config.libs[0].roots, vec!["MyLib"]);
        assert_eq!(config.exes.len(), 1);
        assert_eq!(config.exes[0].name, "myexe");
        assert_eq!(config.exes[0].root, "Main");
        assert_eq!(config.default_targets, vec!["MyLib"]);
    }

    #[test]
    fn test_parse_name_array() {
        assert_eq!(
            LakeConfig::parse_name_array("#[`A, `B, `C]"),
            vec!["A", "B", "C"]
        );
        assert_eq!(LakeConfig::parse_name_array("`Single"), vec!["Single"]);
    }

    #[test]
    fn test_package_config_minimal() {
        let pkg = PackageConfig::minimal("test");
        assert_eq!(pkg.name, "test");
        assert!(pkg.version.is_none());
    }

    #[test]
    fn test_parse_more_lean_args() {
        let content = r#"
import Lake
open Lake DSL

package test where
  moreLeanArgs := #["-DautoImplicit=false", "-Dpp.all"]
  moreLinkArgs := #["-L/usr/local/lib", "-lmylib"]

lean_lib Test
"#;
        let config = LakeConfig::parse(content).unwrap();
        assert_eq!(config.package.name, "test");
        assert_eq!(
            config.package.more_lean_args,
            vec!["-DautoImplicit=false", "-Dpp.all"]
        );
        assert_eq!(
            config.package.more_link_args,
            vec!["-L/usr/local/lib", "-lmylib"]
        );
    }

    #[test]
    fn test_lean_lib_root_modules() {
        let lib = LeanLib::minimal("MyLib");
        assert_eq!(lib.root_modules(), vec!["MyLib"]);

        let lib_with_roots = LeanLib {
            name: "MyLib".to_string(),
            roots: vec!["MyLib.Core".to_string(), "MyLib.Utils".to_string()],
            ..Default::default()
        };
        assert_eq!(
            lib_with_roots.root_modules(),
            vec!["MyLib.Core", "MyLib.Utils"]
        );
    }

    #[test]
    fn test_parse_require_git_dependency() {
        let content = r#"
import Lake
open Lake DSL

require std from git "https://example.com/std4" @ "1.2.3"

package demo
lean_lib Demo
"#;
        let config = LakeConfig::parse(content).unwrap();
        assert_eq!(config.package.dependencies.len(), 1);
        let dep = &config.package.dependencies[0];
        assert_eq!(dep.name, "std");
        assert_eq!(dep.git.as_deref(), Some("https://example.com/std4"));
        assert_eq!(dep.version.as_deref(), Some("1.2.3"));
    }

    #[test]
    fn test_validate_manifest_semver_constraint() {
        let dep = Dependency {
            name: "std".to_string(),
            git: Some("https://example.com/std4".to_string()),
            rev: None,
            path: None,
            version: Some("^1.2".to_string()),
        };

        let manifest = LakeManifest {
            version: 7,
            packages_dir: ".lake/packages".to_string(),
            packages: vec![ManifestPackage::Git(GitPackage {
                name: "std".to_string(),
                url: "https://example.com/std4".to_string(),
                rev: "1.2.5".to_string(),
                input_rev: None,
                subdir: None,
            })],
        };

        PackageConfig {
            name: "demo".to_string(),
            dependencies: vec![dep],
            ..Default::default()
        }
        .validate_manifest(&manifest)
        .unwrap();
    }

    #[test]
    fn test_validate_manifest_mismatch() {
        let dep = Dependency {
            name: "std".to_string(),
            git: Some("https://example.com/std4".to_string()),
            rev: Some("main".to_string()),
            path: None,
            version: None,
        };

        let manifest = LakeManifest {
            version: 7,
            packages_dir: ".lake/packages".to_string(),
            packages: vec![ManifestPackage::Git(GitPackage {
                name: "std".to_string(),
                url: "https://example.com/std4".to_string(),
                rev: "dev-branch".to_string(),
                input_rev: None,
                subdir: None,
            })],
        };

        let result = PackageConfig {
            name: "demo".to_string(),
            dependencies: vec![dep],
            ..Default::default()
        }
        .validate_manifest(&manifest);

        assert!(result.is_err());
    }

    #[test]
    fn test_parse_lakefile_with_tests() {
        let content = r"
import Lake
open Lake DSL

package myproject

lean_lib MyLib

lean_test mytest where
  root := `Test.Main

lean_test integration where
  root := `Test.Integration
";
        let config = LakeConfig::parse(content).unwrap();
        assert_eq!(config.package.name, "myproject");
        assert_eq!(config.libs.len(), 1);
        assert_eq!(config.tests.len(), 2);
        assert_eq!(config.tests[0].name, "mytest");
        assert_eq!(config.tests[0].root, "Test.Main");
        assert_eq!(config.tests[1].name, "integration");
        assert_eq!(config.tests[1].root, "Test.Integration");
    }

    #[test]
    fn test_lean_test_minimal() {
        let test = LeanTest::minimal("unit", "Test.Unit");
        assert_eq!(test.name, "unit");
        assert_eq!(test.root, "Test.Unit");
    }

    #[test]
    fn test_parse_lakefile_with_scripts() {
        let content = r#"
import Lake
open Lake DSL

package myproject

lean_lib MyLib

script hello where
  doc := "Prints hello"
  IO.println "Hello, World!"

script build_docs :=
  do
    IO.println "Building docs..."
    pure 0
"#;
        let config = LakeConfig::parse(content).unwrap();
        assert_eq!(config.package.name, "myproject");
        assert_eq!(config.scripts.len(), 2);
        assert_eq!(config.scripts[0].name, "hello");
        assert_eq!(config.scripts[0].doc, Some("Prints hello".to_string()));
        assert!(config.scripts[0].body.contains("IO.println"));
        assert_eq!(config.scripts[1].name, "build_docs");
        assert!(config.scripts[1].body.contains("Building docs"));
    }

    #[test]
    fn test_lake_script_default() {
        let script = LakeScript::default();
        assert!(script.name.is_empty());
        assert!(script.body.is_empty());
        assert!(script.doc.is_none());
    }

    #[test]
    fn test_parse_mathlib_style_lakefile() {
        // Test parsing a Mathlib-style lakefile with modern require syntax
        let content = r#"
import Lake

open Lake DSL

/-!
## Dependencies
-/

require "leanprover-community" / "batteries" @ git "main"
require "leanprover-community" / "Qq" @ git "master"
require "leanprover-community" / "aesop" @ git "master"

package mathlib where
  testDriver := "MathlibTest"

/-!
## Libraries
-/

@[default_target]
lean_lib Mathlib where
  leanOptions := mathlibLeanOptions

lean_lib Cache
lean_lib LongestPole

lean_lib MathlibTest where
  globs := #[.submodules `MathlibTest]

/-- Documentation library -/
lean_lib docs where
  roots := #[`docs]

/-!
## Executables
-/

/-- Adds labels to PRs -/
lean_exe autolabel where
  srcDir := "scripts"

lean_exe cache where
  root := `Cache.Main

lean_exe «check-yaml» where
  srcDir := "scripts"
  supportInterpreter := true

lean_exe mk_all where
  srcDir := "scripts"
  supportInterpreter := true
  weakLinkArgs := #["-lLake"]
"#;
        let config = LakeConfig::parse(content).unwrap();

        // Check package
        assert_eq!(config.package.name, "mathlib");

        // Check dependencies - should parse "owner" / "repo" format
        assert!(
            config.package.dependencies.len() >= 3,
            "Expected at least 3 dependencies, got {}",
            config.package.dependencies.len()
        );

        // Check libraries
        assert!(
            config.libs.len() >= 5,
            "Expected at least 5 libraries, got {}",
            config.libs.len()
        );
        let lib_names: Vec<_> = config.libs.iter().map(|l| l.name.as_str()).collect();
        assert!(lib_names.contains(&"Mathlib"));
        assert!(lib_names.contains(&"Cache"));
        assert!(lib_names.contains(&"docs"));

        // Check executables - should handle guillemet names
        assert!(
            config.exes.len() >= 4,
            "Expected at least 4 executables, got {}",
            config.exes.len()
        );
        let exe_names: Vec<_> = config.exes.iter().map(|e| e.name.as_str()).collect();
        assert!(exe_names.contains(&"autolabel"));
        assert!(exe_names.contains(&"cache"));
        assert!(exe_names.contains(&"mk_all"));

        // Check default target
        assert!(config.default_targets.contains(&"Mathlib".to_string()));
    }

    #[test]
    fn test_parse_require_new_syntax() {
        // Test the new "owner" / "repo" @ git "branch" syntax
        let content = r#"
import Lake
open Lake DSL

require "leanprover-community" / "batteries" @ git "main"
require "leanprover-community" / "Qq" @ git "v1.0.0"

package test
lean_lib Test
"#;
        let config = LakeConfig::parse(content).unwrap();
        assert_eq!(config.package.dependencies.len(), 2);

        let batteries = &config.package.dependencies[0];
        assert_eq!(batteries.name, "batteries");
        assert!(batteries.git.is_some());
        assert_eq!(batteries.rev.as_deref(), Some("main"));

        let qq = &config.package.dependencies[1];
        assert_eq!(qq.name, "Qq");
        assert!(qq.git.is_some());
    }

    #[test]
    fn test_parse_guillemet_name() {
        // Test parsing names with guillemets (French quotes) «name»
        let content = r"
import Lake
open Lake DSL

package test

lean_exe «check-yaml» where
  root := `CheckYaml
";
        let config = LakeConfig::parse(content).unwrap();
        assert_eq!(config.exes.len(), 1);
        assert_eq!(config.exes[0].name, "check-yaml");
    }
}
