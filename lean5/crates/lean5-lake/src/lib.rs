//! Lake build system for Lean5
//!
//! This crate implements Lake, Lean 4's package manager and build system.
//! It provides:
//!
//! - lakefile.lean parsing
//! - lake-manifest.json parsing
//! - Incremental compilation
//! - Dependency management
//! - Parallel builds

pub mod build;
pub mod config;
pub mod error;
pub mod fetch;
pub mod manifest;
pub mod workspace;

pub use build::{BuildContext, BuildOptions, BuildResult};
pub use config::{LakeConfig, LakeScript, LeanExe, LeanLib, LeanTest, PackageConfig};
pub use error::{LakeError, LakeResult};
pub use fetch::{FetchManager, ResolveResult, ResolvedPackage, UpdateResult, UpdateStatus};
pub use manifest::{GitPackage, LakeManifest, ManifestPackage, PathPackage};
pub use workspace::Workspace;
