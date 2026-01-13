//! Lake workspace management
//!
//! A workspace represents a Lake project with its configuration,
//! manifest, and build state.

use crate::config::{LakeConfig, LeanLib};
use crate::error::{LakeError, LakeResult};
use crate::manifest::LakeManifest;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A Lake workspace
#[derive(Debug)]
pub struct Workspace {
    /// Root directory of the workspace
    root: PathBuf,
    /// Lake configuration from lakefile.lean
    config: LakeConfig,
    /// Lake manifest from lake-manifest.json (if present)
    manifest: Option<LakeManifest>,
    /// Resolved module paths
    module_paths: HashMap<String, PathBuf>,
}

impl Workspace {
    /// Load a workspace from a directory
    pub fn load(root: &Path) -> LakeResult<Self> {
        let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());

        // Load lakefile.lean
        let lakefile_path = root.join("lakefile.lean");
        let config = LakeConfig::load(&lakefile_path)?;

        // Load manifest if present
        let manifest_path = root.join("lake-manifest.json");
        let manifest = if manifest_path.exists() {
            Some(LakeManifest::load(&manifest_path)?)
        } else {
            None
        };

        let mut ws = Self {
            root,
            config,
            manifest,
            module_paths: HashMap::new(),
        };

        // Index modules
        ws.index_modules();

        Ok(ws)
    }

    /// Create a new workspace with minimal configuration
    #[must_use]
    pub fn new(root: &Path, package_name: &str) -> Self {
        Self {
            root: root.to_path_buf(),
            config: LakeConfig {
                package: crate::config::PackageConfig::minimal(package_name),
                libs: vec![],
                exes: vec![],
                tests: vec![],
                scripts: vec![],
                default_targets: vec![],
            },
            manifest: None,
            module_paths: HashMap::new(),
        }
    }

    /// Create a workspace from a pre-parsed configuration
    #[must_use]
    pub fn from_config(root: &Path, config: LakeConfig) -> Self {
        let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());

        // Load manifest if present
        let manifest_path = root.join("lake-manifest.json");
        let manifest = if manifest_path.exists() {
            LakeManifest::load(&manifest_path).ok()
        } else {
            None
        };

        let mut ws = Self {
            root,
            config,
            manifest,
            module_paths: HashMap::new(),
        };

        // Index modules (no errors for new projects with no source files)
        ws.index_modules();

        ws
    }

    /// Get the workspace root directory
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Get the lake configuration
    #[must_use]
    pub fn config(&self) -> &LakeConfig {
        &self.config
    }

    /// Get the manifest (if present)
    #[must_use]
    pub fn manifest(&self) -> Option<&LakeManifest> {
        self.manifest.as_ref()
    }

    /// Get the source directory
    #[must_use]
    pub fn src_dir(&self) -> PathBuf {
        self.root.join(self.config.src_dir())
    }

    /// Get the build directory
    #[must_use]
    pub fn build_dir(&self) -> PathBuf {
        self.root.join(self.config.build_dir())
    }

    /// Get the lib directory (for .olean files)
    #[must_use]
    pub fn lib_dir(&self) -> PathBuf {
        self.build_dir().join("lib")
    }

    /// Get the packages directory
    #[must_use]
    pub fn packages_dir(&self) -> PathBuf {
        self.manifest.as_ref().map_or_else(
            || self.root.join(".lake/packages"),
            |m| self.root.join(&m.packages_dir),
        )
    }

    /// Index all module files in the workspace
    fn index_modules(&mut self) {
        self.module_paths.clear();

        let src_dir = self.src_dir();
        if !src_dir.exists() {
            return;
        }

        // Walk the source directory
        for entry in walkdir::WalkDir::new(&src_dir)
            .follow_links(true)
            .into_iter()
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "lean") {
                // Convert path to module name
                if let Ok(rel_path) = path.strip_prefix(&src_dir) {
                    let module_name = Self::path_to_module_name(rel_path);
                    self.module_paths.insert(module_name, path.to_path_buf());
                }
            }
        }
    }

    /// Convert a relative path to a module name
    fn path_to_module_name(path: &Path) -> String {
        let parts: Vec<_> = path
            .with_extension("")
            .components()
            .filter_map(|c| c.as_os_str().to_str())
            .map(String::from)
            .collect();

        parts.join(".")
    }

    /// Convert a module name to a relative path
    fn module_name_to_path(module_name: &str) -> PathBuf {
        let parts: Vec<&str> = module_name.split('.').collect();
        let mut path = PathBuf::new();
        for part in parts {
            path.push(part);
        }
        path.with_extension("lean")
    }

    /// Find a module's source file
    #[must_use]
    pub fn find_module(&self, module_name: &str) -> Option<PathBuf> {
        // Check indexed modules first
        if let Some(path) = self.module_paths.get(module_name) {
            return Some(path.clone());
        }

        // Try to construct path
        let rel_path = Self::module_name_to_path(module_name);
        let src_path = self.src_dir().join(&rel_path);
        if src_path.exists() {
            return Some(src_path);
        }

        // Check packages
        for pkg_dir in self.package_dirs() {
            let pkg_src = pkg_dir.join(&rel_path);
            if pkg_src.exists() {
                return Some(pkg_src);
            }
        }

        None
    }

    /// Get the .olean file path for a module
    #[must_use]
    pub fn olean_path(&self, module_name: &str) -> PathBuf {
        let rel_path = Self::module_name_to_path(module_name).with_extension("olean");
        self.lib_dir().join(rel_path)
    }

    /// Get the .ilean file path for a module
    #[must_use]
    pub fn ilean_path(&self, module_name: &str) -> PathBuf {
        let rel_path = Self::module_name_to_path(module_name).with_extension("ilean");
        self.lib_dir().join(rel_path)
    }

    /// Get all package directories
    #[must_use]
    pub fn package_dirs(&self) -> Vec<PathBuf> {
        let mut dirs = vec![];

        if let Some(manifest) = &self.manifest {
            let pkg_dir = self.packages_dir();
            for pkg in &manifest.packages {
                dirs.push(pkg_dir.join(pkg.name()));
            }
        }

        dirs
    }

    /// Get all modules in a library
    pub fn lib_modules(&self, lib_name: &str) -> Vec<String> {
        let lib = self.config.libs.iter().find(|l| l.name == lib_name);
        let roots = lib.map(LeanLib::root_modules).unwrap_or_default();

        let mut modules = vec![];
        for root in roots {
            // Add the root module
            modules.push(root.clone());

            // Add all submodules
            for name in self.module_paths.keys() {
                if name.starts_with(&root) && name != &root {
                    modules.push(name.clone());
                }
            }
        }

        modules
    }

    /// Get all modules that need to be built
    #[must_use]
    pub fn all_modules(&self) -> Vec<String> {
        self.module_paths.keys().cloned().collect()
    }

    /// Check if a module needs rebuilding
    #[must_use]
    pub fn needs_rebuild(&self, module_name: &str) -> bool {
        let Some(src) = self.find_module(module_name) else {
            return false;
        };

        let olean = self.olean_path(module_name);

        // Rebuild if .olean doesn't exist
        if !olean.exists() {
            return true;
        }

        // Rebuild if source is newer
        let src_time = std::fs::metadata(&src).and_then(|m| m.modified()).ok();
        let olean_time = std::fs::metadata(&olean).and_then(|m| m.modified()).ok();

        match (src_time, olean_time) {
            (Some(src_t), Some(olean_t)) => src_t > olean_t,
            _ => true,
        }
    }

    /// Create workspace directories
    pub fn create_dirs(&self) -> LakeResult<()> {
        std::fs::create_dir_all(self.build_dir())?;
        std::fs::create_dir_all(self.lib_dir())?;
        Ok(())
    }

    /// Validate that declared dependencies are satisfied by the manifest
    pub fn validate_dependencies(&self) -> LakeResult<()> {
        if self.config.package.dependencies.is_empty() {
            return Ok(());
        }

        if let Some(manifest) = &self.manifest {
            self.config.validate_manifest(manifest)
        } else {
            Err(LakeError::ManifestMissingForDependencies)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_path_to_module_name() {
        assert_eq!(
            Workspace::path_to_module_name(Path::new("MyLib/Core.lean")),
            "MyLib.Core"
        );
        assert_eq!(
            Workspace::path_to_module_name(Path::new("Main.lean")),
            "Main"
        );
        assert_eq!(
            Workspace::path_to_module_name(Path::new("A/B/C.lean")),
            "A.B.C"
        );
    }

    #[test]
    fn test_module_name_to_path() {
        assert_eq!(
            Workspace::module_name_to_path("MyLib.Core"),
            PathBuf::from("MyLib/Core.lean")
        );
        assert_eq!(
            Workspace::module_name_to_path("Main"),
            PathBuf::from("Main.lean")
        );
    }

    #[test]
    fn test_workspace_new() {
        let tmp = TempDir::new().unwrap();
        let ws = Workspace::new(tmp.path(), "test");
        assert_eq!(ws.config().package.name, "test");
    }

    #[test]
    fn test_workspace_load() {
        let tmp = TempDir::new().unwrap();

        // Create lakefile.lean
        let lakefile = tmp.path().join("lakefile.lean");
        fs::write(&lakefile, "package test\nlean_lib Test").unwrap();

        // Create a source file
        let src = tmp.path().join("Test.lean");
        fs::write(&src, "-- Test module").unwrap();

        let ws = Workspace::load(tmp.path()).unwrap();
        assert_eq!(ws.config().package.name, "test");
        assert!(ws.find_module("Test").is_some());
    }

    #[test]
    fn test_workspace_olean_path() {
        let tmp = TempDir::new().unwrap();
        let ws = Workspace::new(tmp.path(), "test");

        let olean = ws.olean_path("MyLib.Core");
        assert!(olean.to_string_lossy().contains("MyLib"));
        assert!(olean.to_string_lossy().ends_with(".olean"));
    }
}
