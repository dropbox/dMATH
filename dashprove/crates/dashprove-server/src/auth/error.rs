//! Authentication error types

use serde::{Deserialize, Serialize};

/// Error response for auth/rate limit failures
#[derive(Debug, Serialize, Deserialize)]
pub struct AuthError {
    /// Error message
    pub error: String,
    /// Error code for programmatic handling
    pub code: String,
    /// Seconds until retry is allowed (for rate limiting)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<u64>,
}
