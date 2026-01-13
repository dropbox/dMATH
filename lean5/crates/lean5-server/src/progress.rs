//! Progress reporting utilities for streaming transports

use crate::rpc::RequestId;
use serde_json::Value;
use tokio::sync::mpsc;

/// Progress update payload emitted by handlers
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    /// Request ID associated with the progress message
    pub request_id: RequestId,
    /// Human-readable progress message
    pub message: String,
    /// Optional percentage (0-100)
    pub percentage: Option<u8>,
    /// Optional structured details
    pub details: Option<Value>,
}

/// Sender used by handlers to emit progress updates
#[derive(Clone, Debug)]
pub struct ProgressSender {
    tx: mpsc::Sender<ProgressUpdate>,
    request_id: RequestId,
}

impl ProgressSender {
    /// Create a new progress sender for the given request
    #[must_use]
    pub fn new(request_id: RequestId, tx: mpsc::Sender<ProgressUpdate>) -> Self {
        Self { tx, request_id }
    }

    /// Send a progress update
    pub async fn notify(
        &self,
        message: impl Into<String>,
        percentage: Option<u8>,
        details: Option<Value>,
    ) {
        let update = ProgressUpdate {
            request_id: self.request_id.clone(),
            message: message.into(),
            percentage: percentage.map(|p| p.min(100)),
            details,
        };

        let _ = self.tx.send(update).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_update_fields() {
        let update = ProgressUpdate {
            request_id: RequestId::Number(42),
            message: "Processing...".to_string(),
            percentage: Some(50),
            details: Some(serde_json::json!({"step": 3})),
        };

        assert_eq!(update.request_id, RequestId::Number(42));
        assert_eq!(update.message, "Processing...");
        assert_eq!(update.percentage, Some(50));
        assert!(update.details.is_some());
    }

    #[test]
    fn test_progress_update_no_percentage() {
        let update = ProgressUpdate {
            request_id: RequestId::String("req-123".into()),
            message: "Starting".to_string(),
            percentage: None,
            details: None,
        };

        assert_eq!(update.request_id, RequestId::String("req-123".into()));
        assert_eq!(update.message, "Starting");
        assert!(update.percentage.is_none());
        assert!(update.details.is_none());
    }

    #[test]
    fn test_progress_update_clone() {
        let update = ProgressUpdate {
            request_id: RequestId::Number(1),
            message: "Test".to_string(),
            percentage: Some(75),
            details: None,
        };

        let cloned = update.clone();
        assert_eq!(cloned.request_id, update.request_id);
        assert_eq!(cloned.message, update.message);
        assert_eq!(cloned.percentage, update.percentage);
    }

    #[test]
    fn test_progress_update_debug() {
        let update = ProgressUpdate {
            request_id: RequestId::Null,
            message: "Debug test".to_string(),
            percentage: Some(100),
            details: None,
        };

        let debug_str = format!("{update:?}");
        assert!(debug_str.contains("ProgressUpdate"));
        assert!(debug_str.contains("Debug test"));
    }

    #[tokio::test]
    async fn test_progress_sender_new() {
        let (tx, _rx) = mpsc::channel::<ProgressUpdate>(8);
        let sender = ProgressSender::new(RequestId::Number(42), tx);

        // ProgressSender should be clonable
        let _cloned = sender.clone();
    }

    #[tokio::test]
    async fn test_progress_sender_notify_basic() {
        let (tx, mut rx) = mpsc::channel::<ProgressUpdate>(8);
        let sender = ProgressSender::new(RequestId::Number(1), tx);

        sender.notify("Testing", Some(50), None).await;

        let received = rx.recv().await.expect("Should receive update");
        assert_eq!(received.request_id, RequestId::Number(1));
        assert_eq!(received.message, "Testing");
        assert_eq!(received.percentage, Some(50));
        assert!(received.details.is_none());
    }

    #[tokio::test]
    async fn test_progress_sender_notify_with_details() {
        let (tx, mut rx) = mpsc::channel::<ProgressUpdate>(8);
        let sender = ProgressSender::new(RequestId::String("test".into()), tx);

        let details = serde_json::json!({"items_processed": 10, "total_items": 100});
        sender
            .notify("Processing batch", Some(10), Some(details))
            .await;

        let received = rx.recv().await.expect("Should receive update");
        assert_eq!(received.request_id, RequestId::String("test".into()));
        assert_eq!(received.message, "Processing batch");
        assert_eq!(received.percentage, Some(10));
        assert!(received.details.is_some());
        let details = received.details.unwrap();
        assert_eq!(details["items_processed"], 10);
        assert_eq!(details["total_items"], 100);
    }

    #[tokio::test]
    async fn test_progress_sender_percentage_clamped_to_100() {
        let (tx, mut rx) = mpsc::channel::<ProgressUpdate>(8);
        let sender = ProgressSender::new(RequestId::Number(1), tx);

        // Send percentage > 100, should be clamped to 100
        sender.notify("Done", Some(150), None).await;

        let received = rx.recv().await.expect("Should receive update");
        assert_eq!(received.percentage, Some(100));
    }

    #[tokio::test]
    async fn test_progress_sender_multiple_updates() {
        let (tx, mut rx) = mpsc::channel::<ProgressUpdate>(8);
        let sender = ProgressSender::new(RequestId::Number(1), tx);

        sender.notify("Step 1", Some(25), None).await;
        sender.notify("Step 2", Some(50), None).await;
        sender.notify("Step 3", Some(75), None).await;
        sender.notify("Done", Some(100), None).await;

        let mut messages = Vec::new();
        while let Ok(update) = rx.try_recv() {
            messages.push(update.message);
        }

        assert_eq!(messages, vec!["Step 1", "Step 2", "Step 3", "Done"]);
    }

    #[tokio::test]
    async fn test_progress_sender_handles_closed_channel() {
        let (tx, rx) = mpsc::channel::<ProgressUpdate>(1);
        let sender = ProgressSender::new(RequestId::Number(1), tx);

        // Drop receiver to close channel
        drop(rx);

        // notify should not panic when channel is closed (it ignores the send error)
        sender.notify("After close", Some(50), None).await;
        // Test passes if no panic occurs
    }

    #[tokio::test]
    async fn test_progress_sender_debug() {
        let (tx, _rx) = mpsc::channel::<ProgressUpdate>(8);
        let sender = ProgressSender::new(RequestId::Number(42), tx);

        let debug_str = format!("{sender:?}");
        assert!(debug_str.contains("ProgressSender"));
    }
}
