// Crate-level lint configuration for pedantic clippy
#![allow(clippy::must_use_candidate)] // Server handlers don't need must_use
#![allow(clippy::missing_const_for_fn)] // const fn optimization is minor
#![allow(clippy::use_self)] // Self vs TypeName - style preference
#![allow(clippy::unused_self)] // Some methods keep self for API compatibility
#![allow(clippy::doc_markdown)] // Missing backticks - low priority
#![allow(clippy::missing_errors_doc)] // Server errors are obvious
#![allow(clippy::uninlined_format_args)] // Named args are clearer
#![allow(clippy::map_unwrap_or)] // Style preference
#![allow(clippy::similar_names)] // e.g., req/resp, state/states
#![allow(clippy::redundant_closure_for_method_calls)] // Minor style issue
#![allow(clippy::option_if_let_else)] // Style preference for match vs map_or
#![allow(clippy::needless_pass_by_value)] // Ownership semantics may be intentional
#![allow(clippy::too_many_lines)] // Server handlers may be long
#![allow(clippy::wildcard_imports)] // Re-exports use wildcard pattern
#![allow(clippy::trivially_copy_pass_by_ref)] // &BackendId is API consistency
#![allow(clippy::needless_raw_string_hashes)] // Raw strings for templates
#![allow(clippy::match_same_arms)] // Sometimes clarity > deduplication
#![allow(clippy::cast_precision_loss)] // usize to f64 is intentional
#![allow(clippy::single_match_else)] // match is clearer for some patterns
#![allow(clippy::format_push_string)] // Common pattern in output
#![allow(clippy::module_name_repetitions)] // routes::RoutesState is clear
#![allow(clippy::significant_drop_tightening)] // Lock scopes are intentional
#![allow(clippy::let_underscore_must_use)] // Some results intentionally ignored
#![allow(clippy::or_fun_call)] // Style preference, ok_or pattern is clear
#![allow(clippy::unused_async)] // async required for trait consistency
#![allow(clippy::missing_panics_doc)] // Panics are implementation details
#![allow(clippy::cast_lossless)] // Explicit casts are clearer
#![allow(clippy::cast_possible_truncation)] // Bounds checked at runtime
#![allow(clippy::items_after_statements)] // Sometimes clearer to define helpers near use
#![allow(clippy::return_self_not_must_use)] // Builder methods don't need must_use

//! DashProve REST API Server Library
//!
//! Provides REST API routes and handlers for the DashProve verification platform.
//! Supports both synchronous REST endpoints and WebSocket streaming for
//! long-running verifications.
//!
//! Features:
//! - REST API endpoints for verification, corpus search, tactic suggestions
//! - WebSocket streaming for long-running verifications
//! - API key authentication via `X-API-Key` or `Authorization: Bearer` headers
//! - Configurable rate limiting per API key
//! - Admin endpoints for API key management
//!
//! See docs/DESIGN.md for API specification.

pub mod admin;
pub mod auth;
pub mod cache;
pub mod metrics;
pub mod middleware;
pub mod routes;
pub mod ws;
