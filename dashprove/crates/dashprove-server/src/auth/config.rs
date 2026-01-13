//! Authentication configuration types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify AuthConfig::default creates valid config with expected values
    #[kani::proof]
    fn verify_auth_config_default() {
        let config = AuthConfig::default();
        kani::assert(!config.required, "default config should not require auth");
        kani::assert(
            config.api_keys.is_empty(),
            "default config should have no api keys",
        );
        kani::assert(
            config.anonymous_rate_limit == 10,
            "default anonymous rate limit should be 10",
        );
    }

    /// Verify AuthConfig::disabled creates same config as default
    #[kani::proof]
    fn verify_auth_config_disabled() {
        let config = AuthConfig::disabled();
        kani::assert(!config.required, "disabled config should not require auth");
        kani::assert(
            config.api_keys.is_empty(),
            "disabled config should have no api keys",
        );
    }

    /// Verify AuthConfig::required sets required flag to true
    #[kani::proof]
    fn verify_auth_config_required() {
        let config = AuthConfig::required();
        kani::assert(config.required, "required config should require auth");
        kani::assert(
            config.api_keys.is_empty(),
            "required config starts with no api keys",
        );
    }

    /// Verify with_api_key adds key with correct default rate limit
    #[kani::proof]
    fn verify_with_api_key_adds_key() {
        let config = AuthConfig::default().with_api_key("test_key", "Test User");
        kani::assert(
            config.api_keys.len() == 1,
            "should have exactly one api key",
        );
        let info = config.api_keys.get("test_key");
        kani::assert(info.is_some(), "key should exist");
        if let Some(info) = info {
            kani::assert(info.rate_limit == 100, "default rate limit should be 100");
            kani::assert(!info.is_admin, "non-admin key by default");
        }
    }

    /// Verify with_api_key_rate_limit sets custom rate limit
    #[kani::proof]
    fn verify_with_api_key_rate_limit() {
        let limit: u32 = kani::any();
        kani::assume(limit > 0 && limit < 10000); // Reasonable bounds

        let config = AuthConfig::default().with_api_key_rate_limit("key", "User", limit);
        let info = config.api_keys.get("key");
        kani::assert(info.is_some(), "key should exist");
        if let Some(info) = info {
            kani::assert(
                info.rate_limit == limit,
                "rate limit should match specified value",
            );
        }
    }

    /// Verify with_admin_key creates admin key
    #[kani::proof]
    fn verify_with_admin_key() {
        let config = AuthConfig::default().with_admin_key("admin_key", "Admin");
        let info = config.api_keys.get("admin_key");
        kani::assert(info.is_some(), "admin key should exist");
        if let Some(info) = info {
            kani::assert(info.is_admin, "should be marked as admin");
            kani::assert(
                info.rate_limit == 100,
                "default admin rate limit should be 100",
            );
        }
    }

    /// Verify with_admin_key_rate_limit sets admin status and custom rate limit
    #[kani::proof]
    fn verify_with_admin_key_rate_limit() {
        let limit: u32 = kani::any();
        kani::assume(limit > 0 && limit < 10000);

        let config = AuthConfig::default().with_admin_key_rate_limit("key", "Admin", limit);
        let info = config.api_keys.get("key");
        kani::assert(info.is_some(), "admin key should exist");
        if let Some(info) = info {
            kani::assert(info.is_admin, "should be marked as admin");
            kani::assert(
                info.rate_limit == limit,
                "rate limit should match specified value",
            );
        }
    }

    /// Verify with_anonymous_rate_limit sets the limit correctly
    #[kani::proof]
    fn verify_with_anonymous_rate_limit() {
        let limit: u32 = kani::any();
        kani::assume(limit > 0);

        let config = AuthConfig::default().with_anonymous_rate_limit(limit);
        kani::assert(
            config.anonymous_rate_limit == limit,
            "anonymous rate limit should match specified value",
        );
    }

    /// Verify multiple keys can be added via chaining
    #[kani::proof]
    fn verify_multiple_keys_chained() {
        let config = AuthConfig::default()
            .with_api_key("key1", "User1")
            .with_api_key("key2", "User2");

        kani::assert(config.api_keys.len() == 2, "should have two api keys");
        kani::assert(
            config.api_keys.contains_key("key1"),
            "first key should exist",
        );
        kani::assert(
            config.api_keys.contains_key("key2"),
            "second key should exist",
        );
    }

    /// Verify adding same key twice overwrites the first
    #[kani::proof]
    fn verify_key_overwrite() {
        let config = AuthConfig::default()
            .with_api_key_rate_limit("key", "User1", 50)
            .with_api_key_rate_limit("key", "User2", 200);

        kani::assert(config.api_keys.len() == 1, "should still have one key");
        let info = config.api_keys.get("key");
        if let Some(info) = info {
            kani::assert(
                info.rate_limit == 200,
                "rate limit should be from second insert",
            );
        }
    }

    /// Verify ApiKeyInfo stores name correctly
    #[kani::proof]
    fn verify_api_key_info_stores_data() {
        let info = ApiKeyInfo {
            name: String::from("Test"),
            rate_limit: 100,
            is_admin: false,
        };

        kani::assert(info.rate_limit == 100, "rate_limit preserved");
        kani::assert(!info.is_admin, "is_admin preserved");
    }

    /// Verify required config preserves anonymous rate limit from default
    #[kani::proof]
    fn verify_required_preserves_anonymous_limit() {
        let config = AuthConfig::required();
        kani::assert(
            config.anonymous_rate_limit == 10,
            "required should preserve default anonymous rate limit",
        );
    }
}

/// Configuration for authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Whether authentication is required (if false, auth is optional)
    pub required: bool,
    /// Valid API keys mapped to their metadata
    pub api_keys: HashMap<String, ApiKeyInfo>,
    /// Rate limit for anonymous requests (requests per minute)
    pub anonymous_rate_limit: u32,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            required: false,
            api_keys: HashMap::new(),
            anonymous_rate_limit: 10, // 10 requests per minute for anonymous
        }
    }
}

impl AuthConfig {
    /// Create new auth config with authentication disabled
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Create auth config with required authentication
    pub fn required() -> Self {
        Self {
            required: true,
            ..Default::default()
        }
    }

    /// Add an API key with default rate limit
    pub fn with_api_key(mut self, key: &str, name: &str) -> Self {
        self.api_keys.insert(
            key.to_string(),
            ApiKeyInfo {
                name: name.to_string(),
                rate_limit: 100, // 100 requests per minute default
                is_admin: false,
            },
        );
        self
    }

    /// Add an API key with custom rate limit
    pub fn with_api_key_rate_limit(mut self, key: &str, name: &str, rate_limit: u32) -> Self {
        self.api_keys.insert(
            key.to_string(),
            ApiKeyInfo {
                name: name.to_string(),
                rate_limit,
                is_admin: false,
            },
        );
        self
    }

    /// Add an admin API key with default rate limit
    pub fn with_admin_key(mut self, key: &str, name: &str) -> Self {
        self.api_keys.insert(
            key.to_string(),
            ApiKeyInfo {
                name: name.to_string(),
                rate_limit: 100,
                is_admin: true,
            },
        );
        self
    }

    /// Add an admin API key with custom rate limit
    pub fn with_admin_key_rate_limit(mut self, key: &str, name: &str, rate_limit: u32) -> Self {
        self.api_keys.insert(
            key.to_string(),
            ApiKeyInfo {
                name: name.to_string(),
                rate_limit,
                is_admin: true,
            },
        );
        self
    }

    /// Set the anonymous rate limit
    pub fn with_anonymous_rate_limit(mut self, limit: u32) -> Self {
        self.anonymous_rate_limit = limit;
        self
    }
}

/// Information about an API key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyInfo {
    /// Human-readable name for the key
    pub name: String,
    /// Rate limit (requests per minute)
    pub rate_limit: u32,
    /// Whether this key has admin privileges
    #[serde(default)]
    pub is_admin: bool,
}
