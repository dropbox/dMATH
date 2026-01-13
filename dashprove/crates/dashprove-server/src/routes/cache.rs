//! Cache route handlers for DashProve server
//!
//! Endpoints:
//! - GET /cache/stats - Get proof cache statistics
//! - POST /cache/save - Save cache to disk
//! - POST /cache/load - Load cache from disk
//! - DELETE /cache/clear - Clear all cache entries

use axum::{extract::State, http::StatusCode, Json};
use std::sync::Arc;

use super::types::{CacheOperationResponse, CacheStatsResponse, ErrorResponse};
use super::AppState;
use crate::cache::ProofCache;

/// GET /cache/stats - Get proof cache statistics
///
/// Returns statistics about the proof result cache including entry counts,
/// TTL settings, and capacity information.
pub async fn cache_stats(State(state): State<Arc<AppState>>) -> Json<CacheStatsResponse> {
    let cache = state.proof_cache.read().await;
    let stats = cache.stats();

    Json(CacheStatsResponse {
        total_entries: stats.total_entries,
        valid_entries: stats.valid_entries,
        expired_entries: stats.expired_entries,
        max_entries: stats.max_entries,
        default_ttl_secs: stats.default_ttl_secs,
    })
}

/// POST /cache/save - Save the cache to disk
///
/// Saves all valid (non-expired) cache entries to the configured cache file.
/// Returns an error if no cache path is configured.
pub async fn cache_save(
    State(state): State<Arc<AppState>>,
) -> Result<Json<CacheOperationResponse>, (StatusCode, Json<ErrorResponse>)> {
    let cache_path = state.cache_path.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Cache persistence not configured".to_string(),
                details: Some("Server was not started with a cache path".to_string()),
            }),
        )
    })?;

    let cache = state.proof_cache.read().await;
    match cache.save_to_file(cache_path) {
        Ok(entries) => Ok(Json(CacheOperationResponse {
            success: true,
            entries,
            message: Some(format!(
                "Saved {} entries to {}",
                entries,
                cache_path.display()
            )),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to save cache".to_string(),
                details: Some(e.to_string()),
            }),
        )),
    }
}

/// POST /cache/load - Load the cache from disk
///
/// Loads cache entries from the configured cache file, replacing current entries.
/// Only valid (non-expired) entries are loaded. Returns an error if no cache path
/// is configured or the file doesn't exist.
pub async fn cache_load(
    State(state): State<Arc<AppState>>,
) -> Result<Json<CacheOperationResponse>, (StatusCode, Json<ErrorResponse>)> {
    let cache_path = state.cache_path.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Cache persistence not configured".to_string(),
                details: Some("Server was not started with a cache path".to_string()),
            }),
        )
    })?;

    if !cache_path.exists() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Cache file not found".to_string(),
                details: Some(format!("File does not exist: {}", cache_path.display())),
            }),
        ));
    }

    match ProofCache::load_from_file(cache_path) {
        Ok(new_cache) => {
            let entries = new_cache.len();
            let mut cache = state.proof_cache.write().await;
            *cache = new_cache;
            Ok(Json(CacheOperationResponse {
                success: true,
                entries,
                message: Some(format!(
                    "Loaded {} entries from {}",
                    entries,
                    cache_path.display()
                )),
            }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to load cache".to_string(),
                details: Some(e.to_string()),
            }),
        )),
    }
}

/// DELETE /cache/clear - Clear all cache entries
///
/// Removes all entries from the in-memory cache. Does not affect the disk cache.
pub async fn cache_clear(State(state): State<Arc<AppState>>) -> Json<CacheOperationResponse> {
    let mut cache = state.proof_cache.write().await;
    let entries = cache.len();
    cache.clear();

    Json(CacheOperationResponse {
        success: true,
        entries,
        message: Some(format!("Cleared {} entries from cache", entries)),
    })
}
