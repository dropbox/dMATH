//! Authentication and rate limiting middleware for DashProve API
//!
//! Provides:
//! - API key authentication via `X-API-Key` header or `Authorization: Bearer` header
//! - Configurable rate limiting per API key
//! - Anonymous rate limiting (stricter)
//! - Persistent API key storage

mod config;
mod error;
mod middleware;
mod rate_limiter;
mod state;

#[cfg(test)]
mod tests;

pub use config::{ApiKeyInfo, AuthConfig};
pub use error::AuthError;
pub use middleware::{admin_middleware, auth_middleware};
pub use rate_limiter::RateLimiter;
pub use state::{AuthState, AuthenticatedUser, KeyInfo};
