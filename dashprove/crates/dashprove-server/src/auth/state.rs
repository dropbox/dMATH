//! Authentication state management

use super::config::{ApiKeyInfo, AuthConfig};
use super::rate_limiter::RateLimiter;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Persisted API keys for disk storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct PersistedKeys {
    pub keys: HashMap<String, ApiKeyInfo>,
}

/// Shared authentication state
#[derive(Clone)]
pub struct AuthState {
    /// Authentication configuration
    pub config: Arc<RwLock<AuthConfig>>,
    /// Rate limiter for API requests
    pub rate_limiter: Arc<RateLimiter>,
    /// Path to persist API keys (if set)
    persist_path: Option<PathBuf>,
}

impl AuthState {
    /// Create a new auth state with the given configuration
    pub fn new(config: AuthConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            rate_limiter: Arc::new(RateLimiter::new()),
            persist_path: None,
        }
    }

    /// Create auth state with persistence to a file
    pub fn with_persistence(config: AuthConfig, path: impl AsRef<Path>) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            rate_limiter: Arc::new(RateLimiter::new()),
            persist_path: Some(path.as_ref().to_path_buf()),
        }
    }

    /// Load API keys from a persistence file, merging with existing config
    pub fn load_from_file(mut config: AuthConfig, path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        if path.exists() {
            match std::fs::read_to_string(path) {
                Ok(contents) => match serde_json::from_str::<PersistedKeys>(&contents) {
                    Ok(persisted) => {
                        info!(
                            keys = persisted.keys.len(),
                            "Loaded API keys from persistence"
                        );
                        // Merge persisted keys with config (persisted keys take precedence for existing keys)
                        for (key, info) in persisted.keys {
                            config.api_keys.insert(key, info);
                        }
                    }
                    Err(e) => {
                        warn!(?e, "Failed to parse API keys file, starting fresh");
                    }
                },
                Err(e) => {
                    warn!(?e, "Failed to read API keys file, starting fresh");
                }
            }
        } else {
            info!(
                ?path,
                "API keys file does not exist, will create on first save"
            );
        }

        Self {
            config: Arc::new(RwLock::new(config)),
            rate_limiter: Arc::new(RateLimiter::new()),
            persist_path: Some(path.to_path_buf()),
        }
    }

    /// Create a disabled auth state (no auth required, generous rate limits)
    pub fn disabled() -> Self {
        Self::new(AuthConfig::disabled())
    }

    /// Save current API keys to persistence file (if configured)
    pub(super) async fn persist(&self) -> Result<(), std::io::Error> {
        if let Some(path) = &self.persist_path {
            let config = self.config.read().await;
            let persisted = PersistedKeys {
                keys: config.api_keys.clone(),
            };
            let contents = serde_json::to_string_pretty(&persisted)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            // Write atomically via temp file
            let temp_path = path.with_extension("tmp");
            std::fs::write(&temp_path, contents)?;
            std::fs::rename(&temp_path, path)?;
            info!(?path, keys = config.api_keys.len(), "Persisted API keys");
        }
        Ok(())
    }

    /// Add an API key at runtime (persists to disk if configured)
    pub async fn add_key(&self, key: &str, name: &str, rate_limit: Option<u32>, is_admin: bool) {
        {
            let mut config = self.config.write().await;
            let info = ApiKeyInfo {
                name: name.to_string(),
                rate_limit: rate_limit.unwrap_or(100),
                is_admin,
            };
            config.api_keys.insert(key.to_string(), info);
        }
        // Persist after releasing the lock
        if let Err(e) = self.persist().await {
            warn!(?e, "Failed to persist API keys after add");
        }
    }

    /// Remove an API key at runtime (persists to disk if configured)
    pub async fn remove_key(&self, key: &str) -> bool {
        let removed = {
            let mut config = self.config.write().await;
            config.api_keys.remove(key).is_some()
        };
        // Persist after releasing the lock
        if removed {
            if let Err(e) = self.persist().await {
                warn!(?e, "Failed to persist API keys after remove");
            }
        }
        removed
    }

    /// List all API key names (without revealing actual keys)
    pub async fn list_keys(&self) -> Vec<KeyInfo> {
        let config = self.config.read().await;
        config
            .api_keys
            .iter()
            .map(|(key, info)| KeyInfo {
                // Show only first 8 chars of key for identification
                key_prefix: if key.len() > 8 {
                    format!("{}...", &key[..8])
                } else {
                    key.clone()
                },
                name: info.name.clone(),
                rate_limit: info.rate_limit,
                is_admin: info.is_admin,
            })
            .collect()
    }

    /// Check if a key has admin privileges
    pub async fn is_admin(&self, key: &str) -> bool {
        let config = self.config.read().await;
        config
            .api_keys
            .get(key)
            .map(|info| info.is_admin)
            .unwrap_or(false)
    }

    /// Check if a key exists
    pub async fn has_key(&self, key: &str) -> bool {
        let config = self.config.read().await;
        config.api_keys.contains_key(key)
    }

    /// Update properties of an existing API key (persists to disk if configured)
    /// Returns true if the key was found and updated, false if key not found
    pub async fn update_key(
        &self,
        key: &str,
        rate_limit: Option<u32>,
        is_admin: Option<bool>,
    ) -> bool {
        let updated = {
            let mut config = self.config.write().await;
            if let Some(info) = config.api_keys.get_mut(key) {
                if let Some(limit) = rate_limit {
                    info.rate_limit = limit;
                }
                if let Some(admin) = is_admin {
                    info.is_admin = admin;
                }
                true
            } else {
                false
            }
        };
        // Persist after releasing the lock
        if updated {
            if let Err(e) = self.persist().await {
                warn!(?e, "Failed to persist API keys after update");
            }
        }
        updated
    }

    /// Get information about a specific key (returns None if key not found)
    pub async fn get_key_info(&self, key: &str) -> Option<KeyInfo> {
        let config = self.config.read().await;
        config.api_keys.get(key).map(|info| KeyInfo {
            key_prefix: if key.len() > 8 {
                format!("{}...", &key[..8])
            } else {
                key.to_string()
            },
            name: info.name.clone(),
            rate_limit: info.rate_limit,
            is_admin: info.is_admin,
        })
    }
}

/// Information about a key (safe to expose)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyInfo {
    /// Prefix of the key for identification
    pub key_prefix: String,
    /// Human-readable name
    pub name: String,
    /// Rate limit (requests per minute)
    pub rate_limit: u32,
    /// Whether this key has admin privileges
    pub is_admin: bool,
}

/// Authenticated user information stored in request extensions
#[derive(Debug, Clone)]
pub struct AuthenticatedUser {
    /// The API key used for authentication (if any)
    pub api_key: Option<String>,
    /// Whether this user has admin privileges
    pub is_admin: bool,
}
