//! Dependency fetching
//!
//! Fetches git dependencies for Lake projects.

use crate::error::{LakeError, LakeResult};
use crate::manifest::{GitPackage, LakeManifest, ManifestPackage};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Fetch manager for Lake dependencies
pub struct FetchManager {
    /// Root directory of the workspace
    root: PathBuf,
    /// Packages directory
    packages_dir: PathBuf,
}

impl FetchManager {
    /// Create a new fetch manager
    #[must_use]
    pub fn new(root: &Path, packages_dir: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            packages_dir: packages_dir.to_path_buf(),
        }
    }

    /// Fetch all dependencies listed in the manifest
    pub fn fetch_all(&self, manifest: &LakeManifest) -> LakeResult<Vec<String>> {
        let mut fetched = vec![];

        for pkg in &manifest.packages {
            match pkg {
                ManifestPackage::Git(git_pkg) => {
                    self.fetch_git_package(git_pkg)?;
                    fetched.push(git_pkg.name.clone());
                }
                ManifestPackage::Path(path_pkg) => {
                    // Path packages don't need fetching, but validate they exist
                    let pkg_path = self.root.join(&path_pkg.path);
                    if !pkg_path.exists() {
                        return Err(LakeError::PackageNotFound {
                            name: path_pkg.name.clone(),
                            path: pkg_path,
                        });
                    }
                }
            }
        }

        Ok(fetched)
    }

    /// Fetch a single git package
    pub fn fetch_git_package(&self, pkg: &GitPackage) -> LakeResult<PathBuf> {
        let pkg_dir = self.packages_dir.join(&pkg.name);

        // Check if already fetched at correct revision
        if pkg_dir.exists() {
            let rev = &pkg.rev;
            if rev.is_empty() {
                // No specific revision requested, assume current is fine
                return Ok(pkg_dir);
            }
            let current_rev = self.get_git_rev(&pkg_dir)?;
            if current_rev.starts_with(rev) || rev.starts_with(&current_rev) {
                // Already at correct revision
                return Ok(pkg_dir);
            }
        }

        // Create packages directory if needed
        std::fs::create_dir_all(&self.packages_dir)?;

        if pkg_dir.exists() {
            // Update existing clone
            self.update_git_package(&pkg_dir, pkg)?;
        } else {
            // Clone new package
            self.clone_git_package(&pkg_dir, pkg)?;
        }

        Ok(pkg_dir)
    }

    /// Clone a git package
    fn clone_git_package(&self, target_dir: &Path, pkg: &GitPackage) -> LakeResult<()> {
        let rev = &pkg.rev;

        // Try shallow clone with branch/tag first
        if !rev.is_empty() {
            let output = Command::new("git")
                .arg("clone")
                .arg("--depth")
                .arg("1")
                .arg("--branch")
                .arg(rev)
                .arg(&pkg.url)
                .arg(target_dir)
                .output()
                .map_err(|e| LakeError::GitError {
                    operation: "clone".to_string(),
                    message: e.to_string(),
                })?;

            if output.status.success() {
                return Ok(());
            }

            // Remove partial clone before full clone
            if target_dir.exists() {
                std::fs::remove_dir_all(target_dir)?;
            }

            // Shallow clone failed, try full clone with checkout
            return self.clone_with_checkout(target_dir, pkg);
        }

        // No specific revision, just clone
        let output = Command::new("git")
            .arg("clone")
            .arg("--depth")
            .arg("1")
            .arg(&pkg.url)
            .arg(target_dir)
            .output()
            .map_err(|e| LakeError::GitError {
                operation: "clone".to_string(),
                message: e.to_string(),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(LakeError::GitError {
                operation: "clone".to_string(),
                message: stderr.to_string(),
            });
        }

        Ok(())
    }

    /// Clone with full history and checkout specific commit
    fn clone_with_checkout(&self, target_dir: &Path, pkg: &GitPackage) -> LakeResult<()> {
        // Remove partial clone if it exists
        if target_dir.exists() {
            std::fs::remove_dir_all(target_dir)?;
        }

        // Clone without depth
        let output = Command::new("git")
            .arg("clone")
            .arg(&pkg.url)
            .arg(target_dir)
            .output()
            .map_err(|e| LakeError::GitError {
                operation: "clone".to_string(),
                message: e.to_string(),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(LakeError::GitError {
                operation: "clone".to_string(),
                message: stderr.to_string(),
            });
        }

        // Checkout specific revision
        let rev = &pkg.rev;
        if !rev.is_empty() {
            let output = Command::new("git")
                .arg("-C")
                .arg(target_dir)
                .arg("checkout")
                .arg(rev)
                .output()
                .map_err(|e| LakeError::GitError {
                    operation: "checkout".to_string(),
                    message: e.to_string(),
                })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(LakeError::GitError {
                    operation: "checkout".to_string(),
                    message: stderr.to_string(),
                });
            }
        }

        Ok(())
    }

    /// Update an existing git package
    fn update_git_package(&self, pkg_dir: &Path, pkg: &GitPackage) -> LakeResult<()> {
        // Fetch latest
        let output = Command::new("git")
            .arg("-C")
            .arg(pkg_dir)
            .arg("fetch")
            .arg("--all")
            .output()
            .map_err(|e| LakeError::GitError {
                operation: "fetch".to_string(),
                message: e.to_string(),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(LakeError::GitError {
                operation: "fetch".to_string(),
                message: stderr.to_string(),
            });
        }

        // Checkout specific revision if specified
        let rev = &pkg.rev;
        if !rev.is_empty() {
            let output = Command::new("git")
                .arg("-C")
                .arg(pkg_dir)
                .arg("checkout")
                .arg(rev)
                .output()
                .map_err(|e| LakeError::GitError {
                    operation: "checkout".to_string(),
                    message: e.to_string(),
                })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(LakeError::GitError {
                    operation: "checkout".to_string(),
                    message: stderr.to_string(),
                });
            }
        }

        Ok(())
    }

    /// Get current git revision for a directory
    fn get_git_rev(&self, dir: &Path) -> LakeResult<String> {
        let output = Command::new("git")
            .arg("-C")
            .arg(dir)
            .arg("rev-parse")
            .arg("HEAD")
            .output()
            .map_err(|e| LakeError::GitError {
                operation: "rev-parse".to_string(),
                message: e.to_string(),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(LakeError::GitError {
                operation: "rev-parse".to_string(),
                message: stderr.to_string(),
            });
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Check if git is available
    #[must_use]
    pub fn git_available() -> bool {
        Command::new("git")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Update a git package to the latest revision from remote
    /// Returns the new commit SHA
    pub fn update_to_latest(&self, pkg: &GitPackage) -> LakeResult<String> {
        let pkg_dir = self.packages_dir.join(&pkg.name);

        if !pkg_dir.exists() {
            // Package not fetched yet, fetch it first
            self.fetch_git_package(pkg)?;
        }

        // Determine the branch to update
        let branch = pkg.input_rev.as_deref().unwrap_or("main");

        // Fetch from remote
        let output = Command::new("git")
            .arg("-C")
            .arg(&pkg_dir)
            .arg("fetch")
            .arg("origin")
            .arg(branch)
            .output()
            .map_err(|e| LakeError::GitError {
                operation: "fetch".to_string(),
                message: e.to_string(),
            })?;

        if !output.status.success() {
            // Try fetching all if specific branch fails
            Command::new("git")
                .arg("-C")
                .arg(&pkg_dir)
                .arg("fetch")
                .arg("--all")
                .output()
                .map_err(|e| LakeError::GitError {
                    operation: "fetch".to_string(),
                    message: e.to_string(),
                })?;
        }

        // Reset to origin/branch
        let target = format!("origin/{branch}");
        let output = Command::new("git")
            .arg("-C")
            .arg(&pkg_dir)
            .arg("reset")
            .arg("--hard")
            .arg(&target)
            .output()
            .map_err(|e| LakeError::GitError {
                operation: "reset".to_string(),
                message: e.to_string(),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(LakeError::GitError {
                operation: "reset".to_string(),
                message: stderr.to_string(),
            });
        }

        // Get the new revision
        self.get_git_rev(&pkg_dir)
    }

    /// Update all git packages and return the updated manifest
    pub fn update_all(
        &self,
        manifest: &LakeManifest,
    ) -> LakeResult<(LakeManifest, Vec<UpdateResult>)> {
        let mut updated_manifest = manifest.clone();
        let mut results = vec![];

        for (idx, pkg) in manifest.packages.iter().enumerate() {
            match pkg {
                ManifestPackage::Git(git_pkg) => {
                    let old_rev = git_pkg.rev.clone();
                    match self.update_to_latest(git_pkg) {
                        Ok(new_rev) => {
                            if new_rev != old_rev {
                                // Update the manifest entry
                                if let Some(ManifestPackage::Git(ref mut mp)) =
                                    updated_manifest.packages.get_mut(idx)
                                {
                                    mp.rev = new_rev.clone();
                                }
                                results.push(UpdateResult {
                                    name: git_pkg.name.clone(),
                                    old_rev,
                                    new_rev,
                                    status: UpdateStatus::Updated,
                                });
                            } else {
                                results.push(UpdateResult {
                                    name: git_pkg.name.clone(),
                                    old_rev: old_rev.clone(),
                                    new_rev: old_rev,
                                    status: UpdateStatus::UpToDate,
                                });
                            }
                        }
                        Err(e) => {
                            results.push(UpdateResult {
                                name: git_pkg.name.clone(),
                                old_rev: old_rev.clone(),
                                new_rev: old_rev,
                                status: UpdateStatus::Error(e.to_string()),
                            });
                        }
                    }
                }
                ManifestPackage::Path(path_pkg) => {
                    // Path packages don't get updated
                    results.push(UpdateResult {
                        name: path_pkg.name.clone(),
                        old_rev: String::new(),
                        new_rev: String::new(),
                        status: UpdateStatus::Skipped,
                    });
                }
            }
        }

        Ok((updated_manifest, results))
    }
}

/// Result of updating a single package
#[derive(Debug)]
pub struct UpdateResult {
    /// Package name
    pub name: String,
    /// Previous revision
    pub old_rev: String,
    /// New revision
    pub new_rev: String,
    /// Update status
    pub status: UpdateStatus,
}

/// Status of an update operation
#[derive(Debug)]
pub enum UpdateStatus {
    /// Package was updated to a new revision
    Updated,
    /// Package was already at the latest revision
    UpToDate,
    /// Package was skipped (path package)
    Skipped,
    /// Error occurred during update
    Error(String),
}

/// Result of resolving dependencies
#[derive(Debug)]
pub struct ResolveResult {
    /// Packages that were resolved
    pub resolved: Vec<ResolvedPackage>,
    /// Errors encountered during resolution
    pub errors: Vec<(String, String)>,
}

/// A resolved package with concrete revision
#[derive(Debug, Clone)]
pub struct ResolvedPackage {
    /// Package name
    pub name: String,
    /// Git URL (if git dependency)
    pub url: Option<String>,
    /// Resolved commit SHA
    pub rev: String,
    /// Original input revision (branch/tag name)
    pub input_rev: Option<String>,
    /// Path (if path dependency)
    pub path: Option<String>,
}

impl FetchManager {
    /// Resolve dependencies declared in the lakefile to concrete revisions.
    /// This clones/fetches each git dependency to determine its commit SHA.
    pub fn resolve_dependencies(
        &self,
        dependencies: &[crate::config::Dependency],
    ) -> LakeResult<ResolveResult> {
        let mut resolved = vec![];
        let mut errors = vec![];

        for dep in dependencies {
            match self.resolve_single_dependency(dep) {
                Ok(pkg) => resolved.push(pkg),
                Err(e) => errors.push((dep.name.clone(), e.to_string())),
            }
        }

        Ok(ResolveResult { resolved, errors })
    }

    /// Resolve a single dependency to a concrete revision
    fn resolve_single_dependency(
        &self,
        dep: &crate::config::Dependency,
    ) -> LakeResult<ResolvedPackage> {
        // Handle path dependencies
        if let Some(path) = &dep.path {
            let full_path = self.root.join(path);
            if !full_path.exists() {
                return Err(LakeError::PackageNotFound {
                    name: dep.name.clone(),
                    path: full_path,
                });
            }
            return Ok(ResolvedPackage {
                name: dep.name.clone(),
                url: None,
                rev: String::new(),
                input_rev: None,
                path: Some(path.to_string_lossy().to_string()),
            });
        }

        // Handle git dependencies
        let url = dep.git.as_ref().ok_or_else(|| {
            LakeError::InvalidConfig(format!(
                "dependency '{}' has neither git nor path specified",
                dep.name
            ))
        })?;

        // Create a temporary GitPackage to fetch
        let input_rev = dep.rev.clone().or_else(|| dep.version.clone());
        let temp_pkg = GitPackage::new(&dep.name, url, input_rev.as_deref().unwrap_or("main"));

        // Fetch the package
        let pkg_dir = self.fetch_git_package(&temp_pkg)?;

        // Get the actual commit SHA
        let rev = self.get_git_rev(&pkg_dir)?;

        Ok(ResolvedPackage {
            name: dep.name.clone(),
            url: Some(url.clone()),
            rev,
            input_rev,
            path: None,
        })
    }

    /// Resolve dependencies and generate a manifest
    pub fn resolve_to_manifest(
        &self,
        dependencies: &[crate::config::Dependency],
    ) -> LakeResult<(LakeManifest, ResolveResult)> {
        let result = self.resolve_dependencies(dependencies)?;

        let mut manifest = LakeManifest::empty();
        for pkg in &result.resolved {
            if let Some(url) = &pkg.url {
                manifest.upsert_package(ManifestPackage::Git(GitPackage {
                    name: pkg.name.clone(),
                    url: url.clone(),
                    rev: pkg.rev.clone(),
                    input_rev: pkg.input_rev.clone(),
                    subdir: None,
                }));
            } else if let Some(path) = &pkg.path {
                manifest.upsert_package(ManifestPackage::Path(crate::manifest::PathPackage::new(
                    &pkg.name, path,
                )));
            }
        }

        Ok((manifest, result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Dependency;
    use std::path::PathBuf;

    #[test]
    fn test_git_available() {
        // Git should be available in test environment
        assert!(FetchManager::git_available());
    }

    #[test]
    fn test_fetch_manager_new() {
        let root = Path::new("/tmp/test_project");
        let packages = Path::new("/tmp/test_project/.lake/packages");
        let fm = FetchManager::new(root, packages);
        assert_eq!(fm.root, root);
        assert_eq!(fm.packages_dir, packages);
    }

    #[test]
    fn test_resolve_result_empty() {
        let result = ResolveResult {
            resolved: vec![],
            errors: vec![],
        };
        assert!(result.resolved.is_empty());
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_resolved_package_git() {
        let pkg = ResolvedPackage {
            name: "std".to_string(),
            url: Some("https://github.com/leanprover/std4".to_string()),
            rev: "abc123def456".to_string(),
            input_rev: Some("main".to_string()),
            path: None,
        };
        assert_eq!(pkg.name, "std");
        assert!(pkg.url.is_some());
        assert!(pkg.path.is_none());
    }

    #[test]
    fn test_resolved_package_path() {
        let pkg = ResolvedPackage {
            name: "local".to_string(),
            url: None,
            rev: String::new(),
            input_rev: None,
            path: Some("../local-pkg".to_string()),
        };
        assert_eq!(pkg.name, "local");
        assert!(pkg.url.is_none());
        assert!(pkg.path.is_some());
    }

    #[test]
    fn test_resolve_path_dependency_not_found() {
        use tempfile::tempdir;

        let temp = tempdir().unwrap();
        let root = temp.path();
        let packages_dir = root.join(".lake/packages");

        let fm = FetchManager::new(root, &packages_dir);

        let dep = Dependency {
            name: "missing".to_string(),
            git: None,
            rev: None,
            path: Some(PathBuf::from("nonexistent")),
            version: None,
        };

        let result = fm.resolve_dependencies(&[dep]).unwrap();
        assert!(result.resolved.is_empty());
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].1.contains("not found"));
    }

    #[test]
    fn test_resolve_dependency_no_source() {
        use tempfile::tempdir;

        let temp = tempdir().unwrap();
        let root = temp.path();
        let packages_dir = root.join(".lake/packages");

        let fm = FetchManager::new(root, &packages_dir);

        let dep = Dependency {
            name: "invalid".to_string(),
            git: None,
            rev: None,
            path: None,
            version: None,
        };

        let result = fm.resolve_dependencies(&[dep]).unwrap();
        assert!(result.resolved.is_empty());
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].1.contains("neither git nor path"));
    }
}
