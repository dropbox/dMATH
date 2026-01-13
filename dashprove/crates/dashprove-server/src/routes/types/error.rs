use serde::Serialize;

// ============ Error Types ============

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
    /// Additional details (optional)
    pub details: Option<String>,
}
