//! Lake manifest parsing (lake-manifest.json)

use crate::error::{LakeError, LakeResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Lake manifest loaded from lake-manifest.json
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LakeManifest {
    /// Manifest version
    pub version: u32,
    /// Packages directory (relative to project root)
    #[serde(rename = "packagesDir")]
    pub packages_dir: String,
    /// List of packages
    pub packages: Vec<ManifestPackage>,
}

/// A package entry in the manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ManifestPackage {
    /// Git-based package
    Git(GitPackage),
    /// Path-based package
    Path(PathPackage),
}

/// Git-based package in manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitPackage {
    /// Package name
    pub name: String,
    /// Git URL
    pub url: String,
    /// Git revision (commit SHA, tag, or branch)
    pub rev: String,
    /// Input revision (user-specified, e.g., "main")
    #[serde(rename = "inputRev")]
    pub input_rev: Option<String>,
    /// Subdirectory containing the package
    pub subdir: Option<String>,
}

/// Path-based package in manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathPackage {
    /// Package name
    pub name: String,
    /// Path to the package
    pub path: String,
}

impl LakeManifest {
    /// Load manifest from a lake-manifest.json file
    pub fn load(manifest_path: &Path) -> LakeResult<Self> {
        let content = std::fs::read_to_string(manifest_path)?;
        Self::parse(&content)
    }

    /// Parse lake-manifest.json content
    pub fn parse(content: &str) -> LakeResult<Self> {
        serde_json::from_str(content).map_err(|e| LakeError::ManifestParse(e.to_string()))
    }

    /// Save manifest to a file
    pub fn save(&self, manifest_path: &Path) -> LakeResult<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(manifest_path, content)?;
        Ok(())
    }

    /// Get a package by name
    #[must_use]
    pub fn get_package(&self, name: &str) -> Option<&ManifestPackage> {
        self.packages.iter().find(|p| p.name() == name)
    }

    /// Add or update a package
    pub fn upsert_package(&mut self, package: ManifestPackage) {
        let name = package.name().to_string();
        if let Some(existing) = self.packages.iter_mut().find(|p| p.name() == name) {
            *existing = package;
        } else {
            self.packages.push(package);
        }
    }

    /// Create an empty manifest
    #[must_use]
    pub fn empty() -> Self {
        Self {
            version: 7, // Current Lake manifest version
            packages_dir: ".lake/packages".to_string(),
            packages: Vec::new(),
        }
    }
}

impl ManifestPackage {
    /// Get the package name
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            ManifestPackage::Git(g) => &g.name,
            ManifestPackage::Path(p) => &p.name,
        }
    }

    /// Check if this is a git package
    #[must_use]
    pub fn is_git(&self) -> bool {
        matches!(self, ManifestPackage::Git(_))
    }

    /// Check if this is a path package
    #[must_use]
    pub fn is_path(&self) -> bool {
        matches!(self, ManifestPackage::Path(_))
    }

    /// Get as git package
    #[must_use]
    pub fn as_git(&self) -> Option<&GitPackage> {
        match self {
            ManifestPackage::Git(g) => Some(g),
            _ => None,
        }
    }

    /// Get as path package
    #[must_use]
    pub fn as_path(&self) -> Option<&PathPackage> {
        match self {
            ManifestPackage::Path(p) => Some(p),
            _ => None,
        }
    }
}

impl GitPackage {
    /// Create a new git package
    #[must_use]
    pub fn new(name: &str, url: &str, rev: &str) -> Self {
        Self {
            name: name.to_string(),
            url: url.to_string(),
            rev: rev.to_string(),
            input_rev: None,
            subdir: None,
        }
    }
}

impl PathPackage {
    /// Create a new path package
    #[must_use]
    pub fn new(name: &str, path: &str) -> Self {
        Self {
            name: name.to_string(),
            path: path.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty_manifest() {
        let content = r#"{
            "version": 7,
            "packagesDir": ".lake/packages",
            "packages": []
        }"#;
        let manifest = LakeManifest::parse(content).unwrap();
        assert_eq!(manifest.version, 7);
        assert_eq!(manifest.packages_dir, ".lake/packages");
        assert!(manifest.packages.is_empty());
    }

    #[test]
    fn test_parse_manifest_with_git_package() {
        let content = r#"{
            "version": 7,
            "packagesDir": ".lake/packages",
            "packages": [
                {
                    "name": "std",
                    "url": "https://github.com/leanprover/std4",
                    "rev": "abc123"
                }
            ]
        }"#;
        let manifest = LakeManifest::parse(content).unwrap();
        assert_eq!(manifest.packages.len(), 1);
        let pkg = &manifest.packages[0];
        assert_eq!(pkg.name(), "std");
        assert!(pkg.is_git());
        let git = pkg.as_git().unwrap();
        assert_eq!(git.url, "https://github.com/leanprover/std4");
        assert_eq!(git.rev, "abc123");
    }

    #[test]
    fn test_parse_manifest_with_path_package() {
        let content = r#"{
            "version": 7,
            "packagesDir": ".lake/packages",
            "packages": [
                {
                    "name": "local",
                    "path": "../local-pkg"
                }
            ]
        }"#;
        let manifest = LakeManifest::parse(content).unwrap();
        assert_eq!(manifest.packages.len(), 1);
        let pkg = &manifest.packages[0];
        assert_eq!(pkg.name(), "local");
        assert!(pkg.is_path());
        let path = pkg.as_path().unwrap();
        assert_eq!(path.path, "../local-pkg");
    }

    #[test]
    fn test_manifest_get_package() {
        let mut manifest = LakeManifest::empty();
        manifest.packages.push(ManifestPackage::Git(GitPackage::new(
            "test",
            "https://example.com/test",
            "main",
        )));

        assert!(manifest.get_package("test").is_some());
        assert!(manifest.get_package("nonexistent").is_none());
    }

    #[test]
    fn test_manifest_upsert_package() {
        let mut manifest = LakeManifest::empty();

        // Add new package
        manifest.upsert_package(ManifestPackage::Git(GitPackage::new(
            "test",
            "https://example.com/test",
            "v1",
        )));
        assert_eq!(manifest.packages.len(), 1);

        // Update existing package
        manifest.upsert_package(ManifestPackage::Git(GitPackage::new(
            "test",
            "https://example.com/test",
            "v2",
        )));
        assert_eq!(manifest.packages.len(), 1);
        let git = manifest.get_package("test").unwrap().as_git().unwrap();
        assert_eq!(git.rev, "v2");
    }

    #[test]
    fn test_manifest_roundtrip() {
        let manifest = LakeManifest {
            version: 7,
            packages_dir: ".lake/packages".to_string(),
            packages: vec![ManifestPackage::Git(GitPackage::new(
                "std",
                "https://github.com/leanprover/std4",
                "abc123",
            ))],
        };

        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: LakeManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.version, manifest.version);
        assert_eq!(parsed.packages.len(), 1);
    }
}
