# DashFlow Integration Protocol

**Version:** 1.0
**Date:** 2025-12-26
**Status:** Specification

---

## Overview

This document specifies the integration protocol between DashProve (verification platform) and DashFlow (AI orchestration system). The integration enables ML-based backend selection, continuous learning from verification results, and intelligent proof strategy optimization.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DashFlow (AI Operating System)                       │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    ML Pipeline                                        │   │
│  │                                                                       │   │
│  │  1. Receive verification feedback from DashProve                     │   │
│  │  2. Extract features from properties and results                     │   │
│  │  3. Train backend selection model                                    │   │
│  │  4. Serve predictions via inference endpoint                         │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Strategy Model                                     │   │
│  │                                                                       │   │
│  │  Input:  PropertyFeatures + CodeContext                              │   │
│  │  Output: BackendPrediction { backend_id, confidence, alternatives }  │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
└──────────────────────────────┼───────────────────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │ HTTP/gRPC API    │                  │
            │ (Predict/Feedback│                  │
            ▼                  ▼                  │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DashProve (Verification Platform)                    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Backend Selector                                   │   │
│  │                                                                       │   │
│  │  if strategy == MlBased {                                            │   │
│  │      prediction = dashflow.predict_backend(features);                │   │
│  │      if prediction.confidence >= min_confidence {                    │   │
│  │          return prediction.backend;                                  │   │
│  │      }                                                               │   │
│  │  }                                                                   │   │
│  │  return rule_based_fallback(property);                               │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Verification Engine                                │   │
│  │                                                                       │   │
│  │  result = backend.verify(property);                                  │   │
│  │  dashflow.record_feedback(property, backend, result);                │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Types

### PropertyFeatures

Features extracted from USL properties for ML prediction. Defined in `dashprove-learning::similarity::PropertyFeatures`.

```rust
pub struct PropertyFeatures {
    /// Property type: "theorem", "invariant", "temporal", "contract", "refinement"
    pub property_type: String,

    /// Maximum nesting depth of expressions
    pub depth: usize,

    /// Number of quantifiers (forall, exists)
    pub quantifier_depth: usize,

    /// Number of implications
    pub implication_count: usize,

    /// Number of arithmetic operations
    pub arithmetic_ops: usize,

    /// Number of function calls
    pub function_calls: usize,

    /// Number of variables
    pub variable_count: usize,

    /// Uses temporal operators (always, eventually, until, leads_to)
    pub has_temporal: bool,

    /// Type names referenced
    pub type_refs: Vec<String>,

    /// Keywords for text-based search
    pub keywords: Vec<String>,
}
```

### CodeContext (Optional)

Additional context when verifying code contracts.

```rust
pub struct CodeContext {
    /// Target programming language
    pub language: String,

    /// Lines of code in target
    pub lines_of_code: usize,

    /// Cyclomatic complexity
    pub cyclomatic_complexity: usize,

    /// Has unsafe blocks (Rust)
    pub has_unsafe: bool,

    /// Has concurrency primitives
    pub has_concurrency: bool,

    /// Has heap allocation
    pub has_heap: bool,
}
```

### BackendPrediction

ML model output. Defined in `dashprove-ai::strategy::predictions::BackendPrediction`.

```rust
pub struct BackendPrediction {
    /// Predicted best backend
    pub backend: BackendId,

    /// Confidence (0.0-1.0)
    pub confidence: f64,

    /// Alternative backends with their probabilities
    pub alternatives: Vec<(BackendId, f64)>,
}
```

### VerificationFeedback

Feedback sent to DashFlow after each verification.

```rust
pub struct VerificationFeedback {
    /// Property features for the verified property
    pub features: PropertyFeatures,

    /// Optional code context
    pub code_context: Option<CodeContext>,

    /// Backend used for verification
    pub backend: BackendId,

    /// Verification outcome
    pub status: VerificationStatus,

    /// Time taken for verification (seconds)
    pub time_seconds: f64,

    /// Proof size (if successful)
    pub proof_size: Option<usize>,

    /// Error message (if failed)
    pub error_message: Option<String>,

    /// Tactics used (if any)
    pub tactics: Vec<String>,

    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

pub enum VerificationStatus {
    Proven,
    Refuted,
    Unknown,
    Timeout,
    Error,
}
```

---

## API Endpoints

### Prediction API

DashProve calls this endpoint to get backend predictions.

#### Request

```http
POST /api/v1/predict
Content-Type: application/json

{
    "features": {
        "property_type": "theorem",
        "depth": 4,
        "quantifier_depth": 2,
        "implication_count": 1,
        "arithmetic_ops": 3,
        "function_calls": 2,
        "variable_count": 5,
        "has_temporal": false,
        "type_refs": ["List", "Int"],
        "keywords": ["append", "length", "induction"]
    },
    "code_context": null
}
```

#### Response

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
    "backend": "Lean4",
    "confidence": 0.87,
    "alternatives": [
        ["Coq", 0.09],
        ["Isabelle", 0.03],
        ["Dafny", 0.01]
    ]
}
```

### Feedback API

DashProve calls this endpoint after each verification.

#### Request

```http
POST /api/v1/feedback
Content-Type: application/json

{
    "features": {
        "property_type": "theorem",
        "depth": 4,
        "quantifier_depth": 2,
        "implication_count": 1,
        "arithmetic_ops": 3,
        "function_calls": 2,
        "variable_count": 5,
        "has_temporal": false,
        "type_refs": ["List", "Int"],
        "keywords": ["append", "length", "induction"]
    },
    "code_context": null,
    "backend": "Lean4",
    "status": "Proven",
    "time_seconds": 2.34,
    "proof_size": 156,
    "error_message": null,
    "tactics": ["induction", "simp", "rfl"],
    "timestamp": "2025-12-26T12:00:00Z"
}
```

#### Response

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
    "recorded": true,
    "feedback_id": "fb_abc123"
}
```

### Batch Feedback API

For bulk upload of historical verification results.

#### Request

```http
POST /api/v1/feedback/batch
Content-Type: application/json

{
    "feedbacks": [
        { /* VerificationFeedback */ },
        { /* VerificationFeedback */ },
        ...
    ]
}
```

#### Response

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
    "recorded_count": 150,
    "failed_count": 0
}
```

### Model Status API

Check the current model status and statistics.

#### Request

```http
GET /api/v1/model/status
```

#### Response

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
    "model_version": "1.2.3",
    "training_samples": 15000,
    "last_trained": "2025-12-25T00:00:00Z",
    "accuracy": 0.82,
    "backends_covered": 15,
    "feature_importance": {
        "property_type": 0.35,
        "has_temporal": 0.22,
        "quantifier_depth": 0.18,
        "depth": 0.12,
        "arithmetic_ops": 0.08,
        "other": 0.05
    }
}
```

---

## DashProve Client Configuration

### Using ML-Based Selection

```rust
use dashprove::{DashProve, DashProveConfig, SelectionStrategy};

// Configure with DashFlow integration
let config = DashProveConfig::default()
    .with_strategy(SelectionStrategy::MlBased {
        min_confidence: 0.7
    })
    .with_dashflow_url("http://localhost:8080")
    .with_learning();

let dashprove = DashProve::new(config)?;

// Verify - automatically uses ML prediction and records feedback
let result = dashprove.verify_spec("spec.usl").await?;
```

### Selection Strategy Enum

```rust
pub enum SelectionStrategy {
    /// Use the first compatible backend
    Single,

    /// Try all compatible backends
    All,

    /// Use N backends for redundancy
    Redundant { min_backends: usize },

    /// ML-based selection via DashFlow
    MlBased { min_confidence: f64 },

    /// Knowledge-enhanced selection (RAG + rules)
    KnowledgeEnhanced { min_confidence: f64 },
}
```

---

## Training Pipeline

### Feature Engineering

DashFlow extracts the following features for model training:

1. **Property Type** (categorical): theorem, invariant, temporal, contract, refinement
2. **Structural Complexity** (numerical): depth, quantifier_depth, implication_count
3. **Operation Types** (numerical): arithmetic_ops, function_calls, variable_count
4. **Temporal Flag** (boolean): has_temporal
5. **Type References** (bag of words): type_refs encoded as TF-IDF
6. **Keywords** (bag of words): keywords encoded as TF-IDF

### Model Architecture

```
Input Features → Dense(128) → ReLU → Dropout(0.3) →
Dense(64) → ReLU → Dropout(0.2) →
Dense(num_backends) → Softmax → Backend Probabilities
```

### Training Loop

```python
# Pseudo-code for training
for epoch in range(num_epochs):
    for batch in feedback_dataloader:
        features = encode_features(batch.features)
        labels = encode_backend(batch.backend)
        weights = compute_sample_weights(batch.status, batch.time_seconds)

        predictions = model(features)
        loss = weighted_cross_entropy(predictions, labels, weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Sample Weighting

- **Successful verifications**: weight = 1.0
- **Failed verifications**: weight = 0.5 (negative signal)
- **Timeouts**: weight = 0.3 (weak negative signal)
- **Errors**: weight = 0.0 (ignore, likely configuration issue)

### Incremental Training

Model is retrained incrementally as new feedback arrives:
- **Batch size**: 1000 samples
- **Training trigger**: Every 1000 new feedbacks or daily
- **Warm start**: Initialize from previous model weights

---

## Metrics and Monitoring

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Prediction Accuracy | % of predictions that lead to successful verification | >80% |
| Confidence Calibration | Correlation between confidence and actual success rate | >0.9 |
| Latency (p50) | Prediction latency | <10ms |
| Latency (p99) | Prediction latency | <50ms |
| Fallback Rate | % of predictions below confidence threshold | <20% |

### Monitoring Dashboards

1. **Prediction Performance**: Accuracy over time, per backend
2. **Feature Drift**: Distribution changes in incoming features
3. **Backend Usage**: Which backends are being recommended
4. **Feedback Volume**: Rate of incoming verification results

---

## Error Handling

### Prediction Failures

If DashFlow is unavailable or returns an error, DashProve falls back to rule-based selection:

```rust
async fn select_backend(&self, property: &Property) -> BackendId {
    if let SelectionStrategy::MlBased { min_confidence } = self.strategy {
        match self.dashflow_client.predict(&features).await {
            Ok(prediction) if prediction.confidence >= min_confidence => {
                return prediction.backend;
            }
            Ok(_) => {
                // Below confidence threshold, fall back
            }
            Err(e) => {
                warn!("DashFlow prediction failed: {}, using fallback", e);
            }
        }
    }
    self.rule_based_select(property)
}
```

### Feedback Failures

Feedback is queued locally if DashFlow is unavailable:

```rust
async fn record_feedback(&self, feedback: VerificationFeedback) {
    match self.dashflow_client.send_feedback(&feedback).await {
        Ok(_) => {}
        Err(e) => {
            warn!("Failed to send feedback: {}, queueing locally", e);
            self.feedback_queue.push(feedback);
        }
    }
}

// Background task flushes queue when DashFlow is available
async fn flush_feedback_queue(&self) {
    while let Some(feedback) = self.feedback_queue.pop() {
        if self.dashflow_client.send_feedback(&feedback).await.is_err() {
            self.feedback_queue.push_front(feedback);
            break;
        }
    }
}
```

---

## Security Considerations

### Authentication

All API calls between DashProve and DashFlow use bearer token authentication:

```http
Authorization: Bearer <jwt_token>
```

DashProve mints HS256 JWTs for each request when `DashFlowMlConfig::with_jwt_secret` is configured (defaults: `iss=dashprove`, `aud=dashflow`, `sub=dashprove-client`, TTL=300s). API keys remain supported as a fallback for legacy setups.

### Rate Limiting

- **Prediction**: 1000 requests/minute per client
- **Feedback**: 10000 requests/minute per client
- **Batch Feedback**: 10 requests/minute per client

### Data Privacy

- No source code is transmitted in feedback
- Only structural features and verification outcomes are shared
- Keywords are hashed before transmission if privacy mode is enabled

---

## Implementation Status

### DashProve Side (Implemented)

- [x] `PropertyFeatures` extraction
- [x] `BackendPrediction` type
- [x] `SelectionStrategy::MlBased` enum variant
- [x] Fallback to rule-based selection
- [x] Proof corpus storage (local)
- [x] DashFlow HTTP client (`DashFlowMlClient` in `crates/dashprove/src/remote.rs`)
- [x] Feedback queue with persistence (`FeedbackQueue` in `crates/dashprove/src/remote.rs`)
- [x] JWT authentication (HS256 short-lived tokens minted by DashProve)

### DashFlow Side (To Be Implemented)

- [ ] `/api/v1/predict` endpoint
- [ ] `/api/v1/feedback` endpoint
- [ ] `/api/v1/feedback/batch` endpoint
- [ ] `/api/v1/model/status` endpoint
- [ ] Training pipeline
- [ ] Model serving infrastructure

---

## References

1. DashProve Design Document: `docs/DESIGN.md`
2. Backend Selection Strategy: `crates/dashprove-ai/src/strategy/`
3. Property Features: `crates/dashprove-learning/src/similarity.rs`
4. Backend Predictions: `crates/dashprove-ai/src/strategy/predictions.rs`
5. Research: `reports/main/verification_tools_research_2025-12-25.md`
