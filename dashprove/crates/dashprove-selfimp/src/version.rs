//! Version types and capability tracking

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Unique identifier for a version
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VersionId(pub String);

impl VersionId {
    /// Create a new version ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a version ID from content hash
    pub fn from_content_hash(content: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(content);
        let hash = hasher.finalize();
        Self(hex::encode(&hash[..16])) // Use first 16 bytes for shorter ID
    }

    /// Get the underlying string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for VersionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A system version with its capabilities and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version {
    /// Unique identifier for this version
    pub id: VersionId,

    /// Human-readable version string (e.g., "1.0.0", "2025.12.27-abc123")
    pub version_string: String,

    /// The capabilities of this version
    pub capabilities: CapabilitySet,

    /// Metadata about this version
    pub metadata: VersionMetadata,

    /// Hash of the version's content (for integrity verification)
    pub content_hash: String,

    /// ID of the previous version (None for genesis)
    pub previous_version: Option<VersionId>,
}

impl Version {
    /// Create a new version
    pub fn new(
        version_string: impl Into<String>,
        capabilities: CapabilitySet,
        content: &[u8],
    ) -> Self {
        let id = VersionId::from_content_hash(content);
        let mut hasher = Sha256::new();
        hasher.update(content);
        let content_hash = hex::encode(hasher.finalize());

        Self {
            id,
            version_string: version_string.into(),
            capabilities,
            metadata: VersionMetadata::default(),
            content_hash,
            previous_version: None,
        }
    }

    /// Create a genesis version (first version in history)
    pub fn genesis(version_string: impl Into<String>, content: &[u8]) -> Self {
        let mut version = Self::new(version_string, CapabilitySet::default(), content);
        version.metadata.is_genesis = true;
        version
    }

    /// Create a new version derived from a previous version
    pub fn derived_from(
        previous: &Version,
        version_string: impl Into<String>,
        capabilities: CapabilitySet,
        content: &[u8],
    ) -> Self {
        let mut version = Self::new(version_string, capabilities, content);
        version.previous_version = Some(previous.id.clone());
        version
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: VersionMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Check if capabilities improved or stayed same compared to another version
    pub fn capabilities_at_least(&self, other: &Version) -> bool {
        self.capabilities.at_least(&other.capabilities)
    }

    /// Get capability improvements over another version
    pub fn capability_improvements(&self, other: &Version) -> Vec<CapabilityChange> {
        self.capabilities.changes_from(&other.capabilities)
    }
}

/// Metadata about a version
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VersionMetadata {
    /// When the version was created
    pub created_at: Option<DateTime<Utc>>,

    /// Description of what changed
    pub description: Option<String>,

    /// Author of this version
    pub author: Option<String>,

    /// Is this the genesis (first) version?
    pub is_genesis: bool,

    /// Tags/labels for this version
    pub tags: Vec<String>,

    /// Custom key-value metadata
    pub custom: HashMap<String, String>,
}

impl VersionMetadata {
    /// Create metadata with a description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Create metadata with an author
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Create metadata with creation time
    pub fn with_created_at(mut self, time: DateTime<Utc>) -> Self {
        self.created_at = Some(time);
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

/// A single capability with a measurable value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    /// Name of the capability
    pub name: String,

    /// Current value (must be comparable)
    pub value: CapabilityValue,

    /// Description of what this capability measures
    pub description: Option<String>,

    /// Unit of measurement (if applicable)
    pub unit: Option<String>,
}

impl Capability {
    /// Create a boolean capability
    pub fn boolean(name: impl Into<String>, value: bool) -> Self {
        Self {
            name: name.into(),
            value: CapabilityValue::Boolean(value),
            description: None,
            unit: None,
        }
    }

    /// Create a numeric capability
    pub fn numeric(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value: CapabilityValue::Numeric(value),
            description: None,
            unit: None,
        }
    }

    /// Create a count capability (non-negative integer)
    pub fn count(name: impl Into<String>, value: u64) -> Self {
        Self {
            name: name.into(),
            value: CapabilityValue::Count(value),
            description: None,
            unit: None,
        }
    }

    /// Create a version capability
    pub fn version(name: impl Into<String>, major: u32, minor: u32, patch: u32) -> Self {
        Self {
            name: name.into(),
            value: CapabilityValue::Version(major, minor, patch),
            description: None,
            unit: None,
        }
    }

    /// Add a description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a unit
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Check if this capability is at least as good as another
    pub fn at_least(&self, other: &Capability) -> bool {
        self.value.at_least(&other.value)
    }
}

/// Value types for capabilities (all must be orderable)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CapabilityValue {
    /// Boolean capability (false < true)
    Boolean(bool),

    /// Numeric capability (higher is better)
    Numeric(f64),

    /// Count capability (higher is better)
    Count(u64),

    /// Semantic version (newer is better)
    Version(u32, u32, u32),
}

impl CapabilityValue {
    /// Check if this value is at least as good as another
    pub fn at_least(&self, other: &CapabilityValue) -> bool {
        match (self, other) {
            (CapabilityValue::Boolean(a), CapabilityValue::Boolean(b)) => *a >= *b,
            (CapabilityValue::Numeric(a), CapabilityValue::Numeric(b)) => *a >= *b,
            (CapabilityValue::Count(a), CapabilityValue::Count(b)) => *a >= *b,
            (CapabilityValue::Version(a1, a2, a3), CapabilityValue::Version(b1, b2, b3)) => {
                (*a1, *a2, *a3) >= (*b1, *b2, *b3)
            }
            // Different types are incomparable
            _ => false,
        }
    }

    /// Check if this value is strictly better than another
    pub fn better_than(&self, other: &CapabilityValue) -> bool {
        match (self, other) {
            (CapabilityValue::Boolean(a), CapabilityValue::Boolean(b)) => *a && !*b,
            (CapabilityValue::Numeric(a), CapabilityValue::Numeric(b)) => *a > *b,
            (CapabilityValue::Count(a), CapabilityValue::Count(b)) => *a > *b,
            (CapabilityValue::Version(a1, a2, a3), CapabilityValue::Version(b1, b2, b3)) => {
                (*a1, *a2, *a3) > (*b1, *b2, *b3)
            }
            _ => false,
        }
    }
}

impl std::fmt::Display for CapabilityValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CapabilityValue::Boolean(v) => write!(f, "{}", v),
            CapabilityValue::Numeric(v) => write!(f, "{:.4}", v),
            CapabilityValue::Count(v) => write!(f, "{}", v),
            CapabilityValue::Version(major, minor, patch) => {
                write!(f, "{}.{}.{}", major, minor, patch)
            }
        }
    }
}

/// A set of capabilities for a version
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilitySet {
    /// All capabilities
    pub capabilities: HashMap<String, Capability>,
}

impl CapabilitySet {
    /// Create an empty capability set
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a capability
    pub fn add(&mut self, cap: Capability) {
        self.capabilities.insert(cap.name.clone(), cap);
    }

    /// Add a capability (builder pattern)
    pub fn with(mut self, cap: Capability) -> Self {
        self.add(cap);
        self
    }

    /// Get a capability by name
    pub fn get(&self, name: &str) -> Option<&Capability> {
        self.capabilities.get(name)
    }

    /// Check if all capabilities are at least as good as another set
    pub fn at_least(&self, other: &CapabilitySet) -> bool {
        for (name, other_cap) in &other.capabilities {
            match self.capabilities.get(name) {
                Some(self_cap) => {
                    if !self_cap.at_least(other_cap) {
                        return false;
                    }
                }
                None => return false, // Missing capability is regression
            }
        }
        true
    }

    /// Get the changes from another capability set
    pub fn changes_from(&self, other: &CapabilitySet) -> Vec<CapabilityChange> {
        let mut changes = Vec::new();

        // Check existing capabilities
        for (name, self_cap) in &self.capabilities {
            match other.capabilities.get(name) {
                Some(other_cap) => {
                    if self_cap.value.better_than(&other_cap.value) {
                        changes.push(CapabilityChange::Improved {
                            name: name.clone(),
                            old_value: other_cap.value.clone(),
                            new_value: self_cap.value.clone(),
                        });
                    } else if other_cap.value.better_than(&self_cap.value) {
                        changes.push(CapabilityChange::Regressed {
                            name: name.clone(),
                            old_value: other_cap.value.clone(),
                            new_value: self_cap.value.clone(),
                        });
                    }
                }
                None => {
                    changes.push(CapabilityChange::Added {
                        name: name.clone(),
                        value: self_cap.value.clone(),
                    });
                }
            }
        }

        // Check for removed capabilities
        for (name, other_cap) in &other.capabilities {
            if !self.capabilities.contains_key(name) {
                changes.push(CapabilityChange::Removed {
                    name: name.clone(),
                    value: other_cap.value.clone(),
                });
            }
        }

        changes
    }
}

/// A change in capability between versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CapabilityChange {
    /// Capability improved
    Improved {
        name: String,
        old_value: CapabilityValue,
        new_value: CapabilityValue,
    },
    /// Capability regressed (BAD)
    Regressed {
        name: String,
        old_value: CapabilityValue,
        new_value: CapabilityValue,
    },
    /// New capability added
    Added {
        name: String,
        value: CapabilityValue,
    },
    /// Capability removed (BAD)
    Removed {
        name: String,
        value: CapabilityValue,
    },
}

impl CapabilityChange {
    /// Check if this change is a regression
    pub fn is_regression(&self) -> bool {
        matches!(
            self,
            CapabilityChange::Regressed { .. } | CapabilityChange::Removed { .. }
        )
    }

    /// Check if this change is an improvement
    pub fn is_improvement(&self) -> bool {
        matches!(
            self,
            CapabilityChange::Improved { .. } | CapabilityChange::Added { .. }
        )
    }
}
