# DashProve Integration Examples

**Version:** 1.0
**Date:** 2025-12-26

This document provides practical integration examples for embedding DashProve into external systems such as CI/CD pipelines, AI agents, and development workflows.

---

## Table of Contents

1. [Library Integration (Rust)](#library-integration-rust)
2. [CI/CD Integration](#cicd-integration)
3. [AI Agent Integration](#ai-agent-integration)
4. [DashFlow Integration](#dashflow-integration)
5. [IDE/Editor Integration](#ideeditor-integration)
6. [Monitoring Integration](#monitoring-integration)

---

## Library Integration (Rust)

### Basic Verification

```rust
use dashprove::{DashProve, DashProveConfig, BackendId};

async fn verify_specification() -> Result<(), Box<dyn std::error::Error>> {
    // Create client with default configuration
    let mut client = DashProve::new(DashProveConfig::default());

    // Verify a USL specification
    let result = client.verify(r#"
        theorem excluded_middle {
            forall x: Bool . x or not x
        }
    "#).await?;

    if result.is_proven() {
        println!("Specification verified successfully!");
        println!("Confidence: {:.2}%", result.confidence * 100.0);
    } else {
        println!("Verification failed");
        for suggestion in &result.suggestions {
            println!("Suggestion: {}", suggestion);
        }
    }

    Ok(())
}
```

### Backend Selection

```rust
use dashprove::{DashProve, DashProveConfig, BackendId};

async fn verify_with_specific_backends() -> Result<(), Box<dyn std::error::Error>> {
    // Configure specific backends
    let config = DashProveConfig::default()
        .with_backend(BackendId::Lean)
        .with_backend(BackendId::TlaPlus);

    let mut client = DashProve::new(config);

    // Verify temporal property with TLA+
    let result = client.verify(r#"
        temporal liveness {
            always(eventually(progress))
        }
    "#).await?;

    // Check which backend succeeded
    for prop_result in &result.property_results {
        println!("{}: {:?} (via {:?})",
            prop_result.name,
            prop_result.status,
            prop_result.backends_used);
    }

    Ok(())
}
```

### ML-Based Backend Selection

```rust
use dashprove::{DashProve, DashProveConfig};

async fn verify_with_ml() -> Result<(), Box<dyn std::error::Error>> {
    // Enable ML-based backend selection with learning
    let config = DashProveConfig::all_backends()
        .with_learning();

    let mut client = DashProve::new(config);

    // The ML model predicts the best backend based on property features
    let result = client.verify(r#"
        contract Buffer::push(self: Buffer, item: Int) -> Buffer {
            requires { self.len < self.capacity }
            ensures { self'.len == self.len + 1 }
        }
    "#).await?;

    Ok(())
}
```

### Rust Code Verification

```rust
use dashprove::{DashProve, DashProveConfig};

async fn verify_rust_code() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = DashProve::new(DashProveConfig::default());

    let rust_code = r#"
        pub fn safe_add(a: u32, b: u32) -> Option<u32> {
            a.checked_add(b)
        }
    "#;

    let spec = r#"
        contract safe_add(a: Int, b: Int) -> Option<Int> {
            requires { a >= 0 and b >= 0 }
            ensures { result.is_some() implies result.unwrap() == a + b }
        }
    "#;

    let result = client.verify_code(rust_code, spec).await?;

    if result.is_proven() {
        println!("Rust code satisfies the contract!");
    }

    Ok(())
}
```

### Runtime Monitor Generation

```rust
use dashprove::{RuntimeMonitor, MonitorConfig};
use dashprove_usl::{parse, typecheck};

fn generate_monitors() -> Result<(), Box<dyn std::error::Error>> {
    let spec = parse(r#"
        invariant positive_balance {
            forall a: Account . a.balance >= 0
        }

        invariant total_conserved {
            sum(accounts, |a| a.balance) == initial_total
        }
    "#)?;

    let typed = typecheck(spec)?;

    // Generate Rust monitor with assertions
    let config = MonitorConfig::default()
        .with_assertions()
        .with_logging();

    let monitor = RuntimeMonitor::from_spec(&typed, &config);

    // Write to file
    std::fs::write("monitors.rs", &monitor.code)?;

    // Example generated code structure:
    // pub struct PositiveBalanceMonitor { ... }
    // impl PositiveBalanceMonitor {
    //     pub fn check(&self, accounts: &[Account]) -> bool { ... }
    // }

    Ok(())
}
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/verify.yml
name: Formal Verification

on:
  push:
    branches: [main]
    paths:
      - 'specs/**'
      - 'src/**'
  pull_request:
    paths:
      - 'specs/**'
      - 'src/**'

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Install DashProve
        run: cargo install dashprove-cli

      - name: Install Verification Backends
        run: |
          # Install Kani for Rust verification
          cargo install --locked kani-verifier
          kani setup

          # Install Z3 for SMT solving
          sudo apt-get install -y z3

      - name: Check Backend Health
        run: dashprove check-tools

      - name: Verify Specifications
        run: |
          for spec in specs/*.usl; do
            echo "Verifying $spec..."
            dashprove verify "$spec" --backends kani,z3 --timeout 300
          done

      - name: Verify Rust Contracts
        run: |
          dashprove verify-code src/lib.rs --spec specs/contracts.usl --timeout 600

      - name: Upload Results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: verification-results
          path: ~/.dashprove/results/
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - verify

variables:
  DASHPROVE_DATA_DIR: ${CI_PROJECT_DIR}/.dashprove

verify-specs:
  stage: verify
  image: rust:latest
  before_script:
    - cargo install dashprove-cli
    - cargo install --locked kani-verifier && kani setup
    - apt-get update && apt-get install -y z3
  script:
    - dashprove verify specs/ --backends kani --timeout 300
    - dashprove verify-code src/lib.rs --spec specs/contracts.usl
  artifacts:
    paths:
      - .dashprove/results/
    when: always
  rules:
    - changes:
        - specs/**/*
        - src/**/*.rs
```

### Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any

    environment {
        DASHPROVE_DATA_DIR = "${WORKSPACE}/.dashprove"
    }

    stages {
        stage('Setup') {
            steps {
                sh 'cargo install dashprove-cli'
                sh 'cargo install --locked kani-verifier && kani setup'
            }
        }

        stage('Verify') {
            parallel {
                stage('Spec Verification') {
                    steps {
                        sh 'dashprove verify specs/ --backends lean,z3 --timeout 300'
                    }
                }
                stage('Contract Verification') {
                    steps {
                        sh 'dashprove verify-code src/lib.rs --spec specs/contracts.usl'
                    }
                }
            }
        }

        stage('Report') {
            steps {
                sh 'dashprove corpus stats --format json > verification-report.json'
                archiveArtifacts 'verification-report.json'
            }
        }
    }

    post {
        failure {
            emailext subject: 'Verification Failed',
                     body: 'Formal verification failed. Check the build logs.',
                     to: '${DEFAULT_RECIPIENTS}'
        }
    }
}
```

### Incremental Verification

```bash
#!/bin/bash
# verify-incremental.sh - Only verify changed files

# Get changed files since last successful verification
LAST_VERIFIED_COMMIT=$(cat .dashprove/last_verified_commit 2>/dev/null || echo "HEAD~1")

# Find changed USL specs
CHANGED_SPECS=$(git diff --name-only "$LAST_VERIFIED_COMMIT" HEAD -- '*.usl')

if [ -z "$CHANGED_SPECS" ]; then
    echo "No specification changes detected"
    exit 0
fi

echo "Verifying changed specifications:"
echo "$CHANGED_SPECS"

# Verify each changed spec
for spec in $CHANGED_SPECS; do
    dashprove verify "$spec" --timeout 120
    if [ $? -ne 0 ]; then
        echo "Verification failed for $spec"
        exit 1
    fi
done

# Record successful verification
git rev-parse HEAD > .dashprove/last_verified_commit
echo "All verifications passed"
```

---

## AI Agent Integration

### Dasher/AI Coding Agent

```rust
use dashprove::{DashProve, DashProveConfig, VerificationResult};
use serde::{Deserialize, Serialize};

/// Integration for AI coding agents like Dasher
pub struct AgentVerifier {
    client: DashProve,
}

#[derive(Serialize, Deserialize)]
pub struct AgentVerificationRequest {
    /// Code to verify
    pub code: String,
    /// USL specification (optional - can be inferred)
    pub spec: Option<String>,
    /// Property types to check
    pub property_types: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct AgentVerificationResponse {
    pub verified: bool,
    pub confidence: f64,
    pub issues: Vec<String>,
    pub suggestions: Vec<String>,
    pub inferred_spec: Option<String>,
}

impl AgentVerifier {
    pub fn new() -> Self {
        let config = DashProveConfig::all_backends()
            .with_learning();
        Self {
            client: DashProve::new(config),
        }
    }

    /// Verify code before committing
    pub async fn verify_code_change(
        &mut self,
        request: AgentVerificationRequest,
    ) -> Result<AgentVerificationResponse, Box<dyn std::error::Error>> {
        // If no spec provided, infer from code
        let spec = match request.spec {
            Some(s) => s,
            None => self.infer_spec(&request.code).await?,
        };

        let result = self.client.verify_code(&request.code, &spec).await?;

        Ok(AgentVerificationResponse {
            verified: result.is_proven(),
            confidence: result.confidence,
            issues: result.property_results
                .iter()
                .filter(|p| !p.status.is_proven())
                .map(|p| format!("{}: {:?}", p.name, p.status))
                .collect(),
            suggestions: result.suggestions.clone(),
            inferred_spec: Some(spec),
        })
    }

    /// Infer USL spec from code using AI
    async fn infer_spec(&self, code: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Use dashprove-ai spec inference
        use dashprove::ai::spec_inference::infer_contracts;
        let contracts = infer_contracts(code)?;
        Ok(contracts)
    }
}
```

### Agent Workflow Example

```rust
/// Example: AI agent integrating verification into code generation
async fn agent_code_generation_workflow() {
    let mut verifier = AgentVerifier::new();

    // Step 1: Agent generates code
    let generated_code = r#"
        pub fn transfer(from: &mut Account, to: &mut Account, amount: u64) {
            from.balance -= amount;
            to.balance += amount;
        }
    "#;

    // Step 2: Verify the generated code
    let result = verifier.verify_code_change(AgentVerificationRequest {
        code: generated_code.to_string(),
        spec: None, // Let DashProve infer the spec
        property_types: vec!["memory_safety".to_string(), "panic_freedom".to_string()],
    }).await.unwrap();

    if !result.verified {
        // Step 3: Agent fixes issues based on feedback
        println!("Issues found: {:?}", result.issues);
        println!("Suggestions: {:?}", result.suggestions);

        // Agent should regenerate code addressing the issues
    }
}
```

---

## DashFlow Integration

### Feature Extraction for ML Prediction

```rust
use dashprove::learning::similarity::PropertyFeatures;
use dashprove_usl::{parse, typecheck};

/// Extract features for DashFlow ML backend selection
pub fn extract_features_for_dashflow(spec_source: &str) -> Result<PropertyFeatures, Box<dyn std::error::Error>> {
    let spec = parse(spec_source)?;
    let typed = typecheck(spec)?;

    // Extract features from the typed specification
    let features = PropertyFeatures::from_typed_spec(&typed);

    Ok(features)
}
```

### Feedback Loop

```rust
use dashprove::{BackendId, VerificationResult};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct VerificationFeedback {
    pub property_features: PropertyFeatures,
    pub backend_id: String,
    pub success: bool,
    pub duration_ms: u64,
    pub error_category: Option<String>,
}

/// Send verification results to DashFlow for model training
pub async fn send_feedback_to_dashflow(
    features: PropertyFeatures,
    backend: BackendId,
    result: &VerificationResult,
    duration: std::time::Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let feedback = VerificationFeedback {
        property_features: features,
        backend_id: backend.to_string(),
        success: result.is_proven(),
        duration_ms: duration.as_millis() as u64,
        error_category: result.error_category(),
    };

    // HTTP POST to DashFlow feedback endpoint
    let client = reqwest::Client::new();
    client.post("https://dashflow.example.com/api/v1/verification/feedback")
        .json(&feedback)
        .send()
        .await?;

    Ok(())
}
```

### Full DashFlow Integration Example

```rust
use dashprove::{DashProve, DashProveConfig, BackendId};
use std::time::Instant;

/// Complete DashFlow-integrated verification workflow
pub async fn dashflow_integrated_verify(
    spec_source: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Extract features
    let features = extract_features_for_dashflow(spec_source)?;

    // 2. Request backend prediction from DashFlow
    let predicted_backend = request_backend_prediction(&features).await?;

    // 3. Run verification with predicted backend
    let config = DashProveConfig::default()
        .with_backend(predicted_backend.clone());
    let mut client = DashProve::new(config);

    let start = Instant::now();
    let result = client.verify(spec_source).await?;
    let duration = start.elapsed();

    // 4. Send feedback to DashFlow
    send_feedback_to_dashflow(features, predicted_backend, &result, duration).await?;

    // 5. If ML prediction failed, try fallback backends
    if !result.is_proven() && predicted_backend != BackendId::Z3 {
        let mut fallback_client = DashProve::new(
            DashProveConfig::default().with_backend(BackendId::Z3)
        );
        let _ = fallback_client.verify(spec_source).await?;
    }

    Ok(())
}

async fn request_backend_prediction(
    features: &PropertyFeatures,
) -> Result<BackendId, Box<dyn std::error::Error>> {
    // Call DashFlow prediction API
    let client = reqwest::Client::new();
    let response: BackendPrediction = client
        .post("https://dashflow.example.com/api/v1/verification/predict")
        .json(features)
        .send()
        .await?
        .json()
        .await?;

    Ok(response.backend_id.parse()?)
}

#[derive(Deserialize)]
struct BackendPrediction {
    backend_id: String,
    confidence: f64,
}
```

---

## IDE/Editor Integration

### VS Code Extension Backend

```rust
use dashprove::{DashProve, DashProveConfig};
use tower_lsp::{LspService, Server};
use tower_lsp::lsp_types::*;

/// LSP server for VS Code extension
pub async fn run_lsp_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| {
        DashProveLspBackend::new(client)
    });

    Server::new(stdin, stdout, socket).serve(service).await;
}

struct DashProveLspBackend {
    client: tower_lsp::Client,
    verifier: DashProve,
}

impl DashProveLspBackend {
    fn new(client: tower_lsp::Client) -> Self {
        Self {
            client,
            verifier: DashProve::new(DashProveConfig::default()),
        }
    }

    /// Verify document on save
    async fn verify_document(&mut self, uri: &Url, content: &str) {
        let result = self.verifier.verify(content).await;

        let diagnostics: Vec<Diagnostic> = match result {
            Ok(r) => r.property_results
                .iter()
                .filter(|p| !p.status.is_proven())
                .map(|p| Diagnostic {
                    range: Range::default(), // TODO: map to actual location
                    severity: Some(DiagnosticSeverity::ERROR),
                    message: format!("{}: verification failed", p.name),
                    ..Default::default()
                })
                .collect(),
            Err(e) => vec![Diagnostic {
                range: Range::default(),
                severity: Some(DiagnosticSeverity::ERROR),
                message: e.to_string(),
                ..Default::default()
            }],
        };

        self.client.publish_diagnostics(uri.clone(), diagnostics, None).await;
    }
}
```

### Command-Line Editor Integration (vim/neovim)

```vim
" ~/.config/nvim/ftplugin/usl.vim

" Verify current file
nnoremap <leader>v :!dashprove verify %<CR>

" Verify and show suggestions
nnoremap <leader>vs :!dashprove verify % --suggest<CR>

" Export to Lean
nnoremap <leader>el :!dashprove export % --target lean -o %:r.lean<CR>

" Async verification with quickfix
function! DashProveVerify()
    let l:file = expand('%')
    let l:cmd = 'dashprove verify ' . l:file . ' --format json'
    call jobstart(l:cmd, {
        \ 'on_stdout': function('s:OnVerifyComplete'),
        \ 'stdout_buffered': v:true
    \ })
endfunction

function! s:OnVerifyComplete(job_id, data, event)
    " Parse JSON and populate quickfix
    let l:results = json_decode(join(a:data, ''))
    let l:qf_items = []
    for prop in l:results.property_results
        if !prop.verified
            call add(l:qf_items, {
                \ 'filename': expand('%'),
                \ 'text': prop.name . ': ' . prop.status
            \ })
        endif
    endfor
    call setqflist(l:qf_items)
    copen
endfunction

command! DashProve call DashProveVerify()
```

---

## Monitoring Integration

### Prometheus Metrics

```rust
use dashprove::{DashProve, DashProveConfig, VerificationResult};
use prometheus::{Counter, Histogram, Registry};
use std::time::Instant;

pub struct MonitoredVerifier {
    client: DashProve,

    // Prometheus metrics
    verifications_total: Counter,
    verifications_success: Counter,
    verification_duration: Histogram,
}

impl MonitoredVerifier {
    pub fn new(registry: &Registry) -> Self {
        let verifications_total = Counter::new(
            "dashprove_verifications_total",
            "Total verification attempts"
        ).unwrap();

        let verifications_success = Counter::new(
            "dashprove_verifications_success",
            "Successful verifications"
        ).unwrap();

        let verification_duration = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "dashprove_verification_duration_seconds",
                "Verification duration in seconds"
            ).buckets(vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0])
        ).unwrap();

        registry.register(Box::new(verifications_total.clone())).unwrap();
        registry.register(Box::new(verifications_success.clone())).unwrap();
        registry.register(Box::new(verification_duration.clone())).unwrap();

        Self {
            client: DashProve::new(DashProveConfig::default()),
            verifications_total,
            verifications_success,
            verification_duration,
        }
    }

    pub async fn verify(&mut self, spec: &str) -> Result<VerificationResult, Box<dyn std::error::Error>> {
        self.verifications_total.inc();

        let start = Instant::now();
        let result = self.client.verify(spec).await?;
        let duration = start.elapsed();

        self.verification_duration.observe(duration.as_secs_f64());

        if result.is_proven() {
            self.verifications_success.inc();
        }

        Ok(result)
    }
}
```

### OpenTelemetry Tracing

```rust
use dashprove::{DashProve, DashProveConfig};
use opentelemetry::trace::{Tracer, TracerProvider};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::prelude::*;

pub fn setup_tracing() {
    let tracer = opentelemetry_jaeger::new_agent_pipeline()
        .with_service_name("dashprove")
        .install_simple()
        .unwrap();

    let telemetry = OpenTelemetryLayer::new(tracer);

    tracing_subscriber::registry()
        .with(telemetry)
        .init();
}

#[tracing::instrument(skip(client))]
pub async fn traced_verify(
    client: &mut DashProve,
    spec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("Starting verification");

    let result = client.verify(spec).await?;

    tracing::info!(
        verified = result.is_proven(),
        confidence = result.confidence,
        "Verification complete"
    );

    Ok(())
}
```

### Webhook Notifications

```rust
use dashprove::{DashProve, DashProveConfig, VerificationResult};
use serde::Serialize;

#[derive(Serialize)]
struct WebhookPayload {
    event: String,
    spec_name: String,
    verified: bool,
    confidence: f64,
    timestamp: String,
    details: String,
}

pub async fn verify_with_webhook(
    spec_source: &str,
    spec_name: &str,
    webhook_url: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut client = DashProve::new(DashProveConfig::default());
    let result = client.verify(spec_source).await?;

    let payload = WebhookPayload {
        event: "verification_complete".to_string(),
        spec_name: spec_name.to_string(),
        verified: result.is_proven(),
        confidence: result.confidence,
        timestamp: chrono::Utc::now().to_rfc3339(),
        details: format!("{:?}", result.property_results),
    };

    // Send webhook
    reqwest::Client::new()
        .post(webhook_url)
        .json(&payload)
        .send()
        .await?;

    Ok(())
}
```

---

## Error Handling Best Practices

```rust
use dashprove::{DashProve, DashProveConfig, DashProveError};

pub async fn robust_verification(spec: &str) -> Result<bool, String> {
    let mut client = DashProve::new(DashProveConfig::all_backends());

    match client.verify(spec).await {
        Ok(result) => {
            if result.is_proven() {
                Ok(true)
            } else {
                // Log suggestions for failed verification
                for suggestion in &result.suggestions {
                    tracing::info!("Suggestion: {}", suggestion);
                }
                Ok(false)
            }
        }
        Err(DashProveError::ParseError(e)) => {
            Err(format!("Specification syntax error: {}", e))
        }
        Err(DashProveError::TypeCheckError(e)) => {
            Err(format!("Type error in specification: {}", e))
        }
        Err(DashProveError::BackendError(backend, e)) => {
            tracing::warn!("Backend {} failed: {}", backend, e);
            // Retry with different backend
            let config = DashProveConfig::default()
                .with_backend(dashprove::BackendId::Z3);
            let mut retry_client = DashProve::new(config);
            match retry_client.verify(spec).await {
                Ok(result) => Ok(result.is_proven()),
                Err(e) => Err(format!("All backends failed: {}", e)),
            }
        }
        Err(DashProveError::Timeout) => {
            Err("Verification timed out - try simpler properties or increase timeout".to_string())
        }
        Err(e) => Err(format!("Unexpected error: {}", e)),
    }
}
```

---

## See Also

- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [API_REFERENCE.md](API_REFERENCE.md) - Full API documentation
- [DASHFLOW_INTEGRATION.md](DASHFLOW_INTEGRATION.md) - DashFlow integration protocol
- [DESIGN.md](DESIGN.md) - Architecture and internals
