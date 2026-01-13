# Plan: Rigorous Benchmark Framework

**Goal:** A single, clean pytest system that runs ALL benchmarks from ALL competitors with consistent, reproducible, comparable results.

**Status:** PLANNING
**Created:** 2026-01-02

---

## Critical Questions (Skeptical Analysis)

Before building, we must answer:

### Reproducibility
- [ ] Are we using the exact same ONNX files as VNN-COMP?
- [ ] Are we using the exact same VNNLIB properties?
- [ ] Are our timeouts consistent with VNN-COMP rules (category-specific)?
- [ ] Do we handle floating point determinism?
- [ ] Can someone else run this and get identical results?

### Correctness
- [ ] How do we know our "verified" results are actually correct?
- [ ] How do we know our "falsified" results have valid counterexamples?
- [ ] Are we validating counterexamples against the actual network?
- [ ] What is our false positive/negative rate?

### Comparability
- [ ] Do we have published results from ALL competitors?
- [ ] Are we using the same scoring formula as VNN-COMP?
- [ ] Are competitor results on the same hardware? (need normalization)
- [ ] Are we comparing apples to apples?

### Completeness
- [ ] Do we have ALL instances from ALL years?
- [ ] Are any instances missing or corrupted?
- [ ] Do we support all required input formats (ONNX, VNNLIB)?

---

## Current State Assessment

### What We Have
```
benchmarks/
├── conftest.py           # Basic infrastructure, needs expansion
├── test_acasxu.py        # ACAS-Xu tests (working)
├── test_vnncomp.py       # Multi-year tests (partial)
├── requirements.txt      # Minimal
├── download_benchmarks.sh # Downloads data
└── vnncomp{2021,2023,2024,2025}/  # Data (gitignored)
```

### What's Missing
1. **Instance-level tracking** - Currently aggregate only
2. **Published baselines** - No competitor results to compare against
3. **VNN-COMP scoring** - Don't implement their exact formula
4. **Memory measurement** - Not tracked
5. **Regression detection** - No historical comparison
6. **Counterexample validation** - Not verified
7. **Deterministic runs** - Not guaranteed
8. **Multi-format reporting** - Only basic output

---

## Proposed Architecture

### Directory Structure

```
benchmarks/
├── pytest.ini                      # pytest configuration
├── conftest.py                     # Shared fixtures, CLI options
├── requirements.txt                # All dependencies
├── README.md                       # How to run benchmarks
│
├── data/                           # Benchmark data (gitignored, ~15GB)
│   ├── vnncomp2021/
│   │   ├── acasxu/
│   │   ├── mnist_fc/
│   │   ├── cifar10_resnet/
│   │   └── ...
│   ├── vnncomp2023/
│   ├── vnncomp2024/
│   └── vnncomp2025/
│
├── baselines/                      # Published competitor results (committed)
│   ├── README.md                   # Sources and methodology
│   ├── vnncomp2021/
│   │   ├── alpha_beta_crown.json   # Their official results
│   │   ├── eran.json
│   │   ├── nnenum.json
│   │   ├── marabou.json
│   │   └── oval.json
│   ├── vnncomp2023/
│   ├── vnncomp2024/
│   └── vnncomp2025/
│
├── results/                        # Our results (gitignored)
│   ├── latest/                     # Most recent run
│   │   ├── summary.json
│   │   ├── acasxu.json
│   │   ├── mnist.json
│   │   └── ...
│   ├── history/                    # Historical runs
│   │   └── 2026-01-02_abc123/
│   └── comparisons/                # vs competitor reports
│
├── tests/                          # Test modules (one per benchmark category)
│   ├── __init__.py
│   ├── test_acasxu.py              # ACAS-Xu (45 networks × 10 properties)
│   ├── test_mnist_fc.py            # MNIST fully-connected
│   ├── test_cifar_resnet.py        # CIFAR-10 ResNet
│   ├── test_vit.py                 # Vision Transformer
│   ├── test_nn4sys.py              # Systems benchmarks
│   ├── test_vggnet.py              # VGGNet
│   ├── test_yolo.py                # YOLO
│   ├── test_soundness.py           # VNN-COMP 2025 soundness benchmark
│   └── test_all_instances.py       # Parametrized over ALL instances
│
├── lib/                            # Benchmark library
│   ├── __init__.py
│   ├── runner.py                   # γ-CROWN execution wrapper
│   ├── validator.py                # Counterexample validation
│   ├── scoring.py                  # VNN-COMP scoring formula
│   ├── comparison.py               # Compare against baselines
│   ├── reporting.py                # Generate reports (JSON, HTML, MD, CSV)
│   ├── instances.py                # Load instances from CSV
│   ├── memory.py                   # Memory profiling
│   └── regression.py               # Detect regressions vs history
│
└── scripts/
    ├── download_all.sh             # Download all benchmark data
    ├── download_baselines.sh       # Fetch published competitor results
    ├── run_full_suite.sh           # Run everything with standard settings
    ├── compare_to_sota.sh          # Generate comparison report
    └── validate_installation.py    # Check everything is set up correctly
```

### Core Data Structures

```python
# lib/types.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict
from datetime import datetime

class Status(Enum):
    VERIFIED = "verified"       # Property holds (proven)
    FALSIFIED = "falsified"     # Property violated (counterexample found)
    UNKNOWN = "unknown"         # Could not determine
    TIMEOUT = "timeout"         # Exceeded time limit
    MEMOUT = "memout"           # Exceeded memory limit
    ERROR = "error"             # Crashed or invalid

@dataclass
class Counterexample:
    """A concrete input that violates the property."""
    input: List[float]
    output: List[float]
    violation_amount: float     # How much property is violated
    validated: bool             # Did we verify this against the network?

@dataclass
class InstanceResult:
    """Result for a single (network, property) instance."""
    # Identity
    benchmark: str              # e.g., "acasxu"
    year: int                   # e.g., 2021
    network: str                # e.g., "ACASXU_run2a_1_1_batch_2000.onnx"
    property: str               # e.g., "prop_1.vnnlib"
    instance_id: str            # Unique identifier

    # Result
    status: Status

    # Timing
    time_seconds: float
    timeout_seconds: float      # What timeout was used

    # Memory
    peak_memory_mb: float

    # Details
    counterexample: Optional[Counterexample] = None
    bounds: Optional[Dict] = None
    error_message: Optional[str] = None

    # Metadata
    method: str = "beta-crown"  # Which method was used
    git_commit: str = ""        # For reproducibility
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BenchmarkResult:
    """Aggregate result for a benchmark category."""
    benchmark: str
    year: int

    # Counts
    total: int
    verified: int
    falsified: int
    unknown: int
    timeout: int
    memout: int
    error: int

    # Rates
    @property
    def verified_rate(self) -> float:
        return self.verified / self.total * 100 if self.total > 0 else 0

    @property
    def resolved_rate(self) -> float:
        return (self.verified + self.falsified) / self.total * 100 if self.total > 0 else 0

    # Timing
    total_time: float
    avg_time: float

    # Memory
    peak_memory_mb: float
    avg_memory_mb: float

    # VNN-COMP score
    vnncomp_score: float

    # Instance details
    instances: List[InstanceResult] = field(default_factory=list)

@dataclass
class ComparisonResult:
    """Comparison between γ-CROWN and a competitor."""
    competitor: str
    benchmark: str
    year: int

    # Our results
    gamma_verified: int
    gamma_time: float
    gamma_memory: float

    # Their results
    competitor_verified: int
    competitor_time: float
    competitor_memory: Optional[float]

    # Deltas
    verified_delta: int         # positive = we're better
    time_speedup: float         # >1 = we're faster
    memory_ratio: Optional[float]  # <1 = we use less

    # Per-instance comparison
    we_win: List[str]           # Instances only we solved
    they_win: List[str]         # Instances only they solved
    both_solved: List[str]      # Both solved
    neither_solved: List[str]   # Neither solved
```

### VNN-COMP Scoring Implementation

```python
# lib/scoring.py

def vnncomp_score(results: List[InstanceResult], category_timeout: float) -> float:
    """
    Calculate VNN-COMP score for a set of instances.

    Scoring formula (VNN-COMP 2024):
    - Verified/Falsified within timeout: 1 point
    - Unknown/Timeout: 0 points
    - Incorrect result: -10 points (CRITICAL)

    Tiebreaker: Total time for solved instances
    """
    score = 0
    total_time = 0

    for r in results:
        if r.status in [Status.VERIFIED, Status.FALSIFIED]:
            if r.time_seconds <= category_timeout:
                score += 1
                total_time += r.time_seconds
            else:
                # Solved but over timeout - no points
                pass
        elif r.status == Status.ERROR:
            # Potential incorrect result - investigate
            score -= 10  # Penalty for crashes

    return score, total_time

def category_timeout(benchmark: str, year: int) -> float:
    """Get VNN-COMP timeout for a benchmark category."""
    # VNN-COMP 2024 timeouts (from their scripts)
    TIMEOUTS = {
        ("acasxu", 2021): 180,
        ("acasxu", 2023): 116,
        ("acasxu", 2024): 116,
        ("mnist_fc", 2021): 60,
        ("cifar10_resnet", 2021): 400,
        ("vit", 2023): 300,
        ("vit", 2024): 300,
        # ... etc
    }
    return TIMEOUTS.get((benchmark, year), 300)  # Default 5 min
```

### CLI Interface

```python
# conftest.py additions

def pytest_addoption(parser):
    # Execution options
    parser.addoption("--method", default="auto",
        choices=["ibp", "crown", "alpha", "beta", "gcp", "auto"],
        help="Verification method to use")
    parser.addoption("--timeout", type=int, default=None,
        help="Override timeout (uses VNN-COMP default if not set)")
    parser.addoption("--memory-limit", type=int, default=None,
        help="Memory limit in MB")

    # Benchmark selection
    parser.addoption("--year", type=int, default=None,
        help="VNN-COMP year (2021, 2023, 2024, 2025)")
    parser.addoption("--benchmark", default=None,
        help="Specific benchmark category")
    parser.addoption("--instance", default=None,
        help="Specific instance ID")

    # Comparison options
    parser.addoption("--compare-to", default=None,
        help="Compare against competitor (alpha-beta-crown, eran, etc.)")
    parser.addoption("--baseline-file", default=None,
        help="Path to baseline results JSON")

    # Output options
    parser.addoption("--save-results", default=None,
        help="Save results to file")
    parser.addoption("--report-format", default="json",
        choices=["json", "csv", "html", "markdown"],
        help="Report output format")

    # Scoring options
    parser.addoption("--vnncomp-scoring", action="store_true",
        help="Use VNN-COMP scoring formula")

    # Regression options
    parser.addoption("--compare-history", default=None,
        help="Compare against historical run")
    parser.addoption("--fail-on-regression", action="store_true",
        help="Fail if any instance regressed")
```

### Example Usage

```bash
# Run all benchmarks with default settings
pytest benchmarks/tests/ -v

# Run specific benchmark
pytest benchmarks/tests/test_acasxu.py -v

# Run with specific method and timeout
pytest benchmarks/tests/test_acasxu.py --method=beta --timeout=60

# Run single instance
pytest benchmarks/tests/test_acasxu.py --instance="1_1_prop_1"

# Compare against α,β-CROWN
pytest benchmarks/tests/test_acasxu.py --compare-to=alpha-beta-crown

# Generate VNN-COMP style report
pytest benchmarks/tests/ --vnncomp-scoring --report-format=html --save-results=report.html

# Check for regressions
pytest benchmarks/tests/ --compare-history=results/history/2026-01-01 --fail-on-regression

# Run only 2025 benchmarks
pytest benchmarks/tests/ --year=2025

# Full suite with comparison and reporting
./scripts/run_full_suite.sh --compare-all --report
```

---

## Implementation Phases

### Phase 1: Foundation (5 commits)

**Goal:** Clean up existing infrastructure, establish conventions.

- [ ] Reorganize directory structure as specified above
- [ ] Create `lib/types.py` with data structures
- [ ] Refactor `conftest.py` with full CLI options
- [ ] Create `lib/runner.py` wrapper for γ-CROWN execution
- [ ] Add `scripts/validate_installation.py`

**Deliverable:** `pytest benchmarks/tests/test_acasxu.py -v` works with new structure

### Phase 2: Instance-Level Tracking (5 commits)

**Goal:** Track every instance individually, not just aggregates.

- [ ] Create `lib/instances.py` to parse VNN-COMP instances.csv files
- [ ] Modify tests to be parametrized over all instances
- [ ] Store per-instance results in JSON
- [ ] Add instance ID to all results

**Deliverable:** Can see results for each of 186 ACAS-Xu instances individually

### Phase 3: Competitor Baselines (5 commits)

**Goal:** Get published results from ALL competitors.

- [ ] Research and document sources for competitor results
- [ ] Create `baselines/` directory with published results
- [ ] Create `lib/comparison.py` to compare against baselines
- [ ] Add `--compare-to` CLI option

**Sources needed:**
- α,β-CROWN: VNN-COMP submissions + their papers
- ERAN: VNN-COMP submissions
- nnenum: VNN-COMP submissions
- Marabou: VNN-COMP submissions
- OVAL: VNN-COMP submissions
- PyRAT: VNN-COMP 2023/2024 submissions

**Deliverable:** `pytest --compare-to=alpha-beta-crown` shows side-by-side results

### Phase 4: VNN-COMP Scoring (3 commits)

**Goal:** Implement exact VNN-COMP scoring formula.

- [ ] Create `lib/scoring.py` with official formula
- [ ] Get per-category timeouts from VNN-COMP scripts
- [ ] Add `--vnncomp-scoring` option
- [ ] Verify scoring matches published results

**Deliverable:** Our score calculation matches VNN-COMP published scores

### Phase 5: Memory Measurement (3 commits)

**Goal:** Track peak memory usage.

- [ ] Create `lib/memory.py` with memory profiling
- [ ] Add memory tracking to runner
- [ ] Store peak_memory_mb in results
- [ ] Add memory comparison in reports

**Deliverable:** Every result includes accurate memory measurement

### Phase 6: Counterexample Validation (3 commits)

**Goal:** Verify that counterexamples are actually valid.

- [ ] Create `lib/validator.py`
- [ ] For each falsified result, run counterexample through network
- [ ] Verify property is actually violated
- [ ] Flag any invalid counterexamples

**Deliverable:** 100% confidence in falsified results

### Phase 7: Reporting (5 commits)

**Goal:** Generate beautiful, useful reports.

- [ ] Create `lib/reporting.py`
- [ ] JSON output (machine-readable)
- [ ] CSV output (spreadsheet-friendly)
- [ ] HTML output (visual, interactive)
- [ ] Markdown output (documentation-friendly)

**Deliverable:** `--report-format=html` generates publication-quality report

### Phase 8: Regression Detection (3 commits)

**Goal:** Automatically detect if we got worse.

- [ ] Create `lib/regression.py`
- [ ] Store historical results in `results/history/`
- [ ] Compare current run against history
- [ ] `--fail-on-regression` fails CI if any instance regressed

**Deliverable:** CI catches regressions automatically

### Phase 9: All Benchmarks (10 commits)

**Goal:** Tests for every VNN-COMP benchmark category.

Create test files for:
- [ ] `test_acasxu.py` (refactor existing)
- [ ] `test_mnist_fc.py`
- [ ] `test_cifar_resnet.py`
- [ ] `test_cifar100.py`
- [ ] `test_vit.py`
- [ ] `test_vggnet.py`
- [ ] `test_nn4sys.py`
- [ ] `test_yolo.py`
- [ ] `test_soundness.py` (VNN-COMP 2025)
- [ ] `test_tinyimagenet.py`

**Deliverable:** Every VNN-COMP 2021-2025 benchmark has a test file

### Phase 10: Documentation & Polish (3 commits)

**Goal:** Make it easy for others to use.

- [ ] Comprehensive `benchmarks/README.md`
- [ ] Document how to add new benchmarks
- [ ] Document how to add new competitors
- [ ] Example output in docs

**Deliverable:** New contributor can run benchmarks in 5 minutes

### Phase 10.5: Model Format Support (10-16 commits)

**Goal:** Load and verify models in any format, not just ONNX.

**Why:** Most HuggingFace models are in SafeTensors format. Requiring ONNX export is:
- Error-prone (complex models fail to export)
- Time-consuming (manual export for each model)
- Lossy (some ops don't translate well)

#### Current Format Support

| Format | Load Weights | Build Graph | Verify |
|--------|--------------|-------------|--------|
| ONNX | ✅ | ✅ | ✅ |
| SafeTensors | ✅ | ❌ | ❌ |
| GGUF | ✅ | ❌ | ❌ |
| PyTorch | ✅ | ❌ | ❌ |

#### Implementation Phases

**Phase A: Complete ONNX Op Coverage (2-3 commits)**
- Add `Neg` op (blocks CosyVoice3)
- Add `Where` op (conditional selection)
- Add `Abs` op (absolute value)
- Add `NonZero` op (detection models)

**Phase B: HuggingFace Config-Based Loading (5-8 commits)**
- Parse `config.json` to determine architecture
- Build graph programmatically for known architectures
- Load weights from SafeTensors/GGUF/PyTorch
- Combine graph + weights

Supported architectures:
1. `EfficientNetForImageClassification`
2. `RTDetrForObjectDetection`
3. `Idefics3ForConditionalGeneration`
4. `LlamaForCausalLM`
5. `WhisperForConditionalGeneration`

**Phase C: PyTorch Graph Import (3-5 commits)**
- Parse TorchScript models directly
- Handle custom architectures
- Fall back to ONNX export if needed

#### Success Criteria

- [ ] `gamma verify model.safetensors --config config.json` works
- [ ] All Docling models load without manual ONNX export
- [ ] GGUF models (TinyLLaMA, Gemma) load and verify
- [ ] CosyVoice3 loads without warnings

#### Deliverable

Any HuggingFace model with `config.json` can be verified directly.

---

### Phase 11: LLM & Target Model Benchmarks (8 commits)

**Goal:** Demonstrate γ-CROWN's unique capability: billion-parameter scale verification.

This is our **differentiator**. No other verifier can handle these models.

#### Available Models (in `models/`)

**Speech & Audio Models:**
| Model | Parameters | Format | Size | Status |
|-------|------------|--------|------|--------|
| Whisper-tiny | 39M | ONNX/safetensors | 32MB | Primary target |
| Whisper-small | 244M | safetensors | ~500MB | Scale test |
| Whisper-medium | 769M | safetensors | ~1.5GB | Scale test |
| Whisper-large-v3 | 1.55B | ONNX/safetensors | ~2.5GB | Ultimate target |
| CosyVoice2 | ~100M | ONNX | ~200MB | Speech model |

**LLM Models:**
| Model | Parameters | Format | Size | Status |
|-------|------------|--------|------|--------|
| TinyLLaMA-1.1B | 1.1B | GGUF (Q4) | ~600MB | LLM verification |
| Gemma-2B | 2B | GGUF (Q4) | ~1.4GB | Large LLM |

**Docling Document Understanding Models (from [docling-project](https://huggingface.co/docling-project)):**
| Model | Parameters | Architecture | Purpose | Status |
|-------|------------|--------------|---------|--------|
| GraniteDocling-258M | 258M | VLM (SigLIP + Granite LLM) | Document conversion | Primary target |
| SmolDocling-256M | 256M | Image-Text-to-Text | Document conversion | Scale test |
| docling-layout-egret-xlarge | 62.7M | Layout detection | Page layout analysis | Layout verification |
| docling-layout-egret-large | 31.2M | Layout detection | Page layout analysis | Layout verification |
| DocumentFigureClassifier | 4.07M | Classifier | Figure classification | Small model baseline |
| CodeFormulaV2 | 300M | Code/formula model | Formula recognition | Code verification |
| MolGrapher | ~50M | Graph model | Molecular structure | Specialized |

**Why Docling Models Matter:**
- Used in production document processing
- Same verification challenges as Whisper (transformers, attention, etc.)
- Smaller models (4M-300M) provide stepping stones to Whisper (1.5B)
- VLM architecture combines vision + language (harder than single modality)

#### Verification Tasks for LLMs/Transformers

Unlike image classifiers (robustness to perturbations), LLMs need different verification tasks:

1. **Token Embedding Robustness**: Given token embeddings, verify output bounds under perturbation
2. **Attention Pattern Stability**: Verify attention doesn't collapse under noise
3. **Layer-wise Bounds**: Track how bounds propagate through transformer layers
4. **Output Logit Bounds**: Bound the output distribution range

#### Verification Tasks for Docling Models

Document understanding models have specific verification needs:

1. **Layout Detection Robustness**: Verify bounding box predictions under image perturbations
2. **OCR/Text Extraction Bounds**: Verify text extraction is stable under noise
3. **VLM Equivalence**: Verify Metal/GPU port produces bounded deviation from reference
4. **Figure Classification Stability**: Verify classifier outputs under input variations
5. **Formula Recognition Bounds**: Verify LaTeX output bounds for equation images

#### Implementation Tasks

- [ ] Create `benchmarks/tests/test_whisper.py`
  - Whisper-tiny encoder verification
  - Whisper encoder output bounds given input perturbations
  - Scale tests up to whisper-large-v3

- [ ] Create `benchmarks/tests/test_llm.py`
  - TinyLLaMA embedding → output bounds
  - Gemma-2B single forward pass bounds

- [ ] Create `benchmarks/tests/test_docling.py`
  - DocumentFigureClassifier (smallest, good baseline)
  - docling-layout-egret-large (layout detection)
  - GraniteDocling-258M (VLM, harder)
  - SmolDocling-256M (VLM scale test)
  - Memory scaling analysis

- [ ] Create `benchmarks/tests/test_speech.py`
  - CosyVoice2 speech tokenizer bounds
  - End-to-end speech verification

- [ ] Create VNNLIB properties for transformers
  - `tests/models/whisper_robustness.vnnlib`
  - `tests/models/llm_embedding_bounds.vnnlib`

- [ ] Add model download script
  - `scripts/download_llm_models.sh`
  - HuggingFace integration for safetensors
  - GGUF to ONNX conversion (if needed)

- [ ] Create baseline comparison for scale
  - What can α,β-CROWN handle? (typically <1M params)
  - What can ERAN handle?
  - Demonstrate 1000x scale advantage

- [ ] Memory/time scaling analysis
  - Plot verification time vs model size
  - Plot memory vs model size
  - Identify scaling bottlenecks

- [ ] Integration tests
  - CI-friendly subset (whisper-tiny only)
  - Full suite for nightly runs

#### Target Metrics for LLM Benchmarks

| Model | Target Bound Width | Target Time | Target Memory |
|-------|-------------------|-------------|---------------|
| Whisper-tiny | <1.0 | <60s | <4GB |
| Whisper-small | <2.0 | <5min | <8GB |
| Whisper-medium | <5.0 | <15min | <16GB |
| Whisper-large-v3 | <10.0 | <30min | <32GB |
| TinyLLaMA-1.1B | <100.0 | <20min | <16GB |
| Gemma-2B | <500.0 | <1hr | <32GB |

Current status (from PLAN-path-to-sota.md):
- Whisper-tiny bound width: ~2-4 (need <1.0)
- Qwen-0.6B bound width: ~444,000 (need <1,000)

#### Success Criteria

- [ ] Whisper-tiny verification completes in <60s with bounds <1.0
- [ ] At least one model >1B parameters verified
- [ ] Memory scaling is sublinear (streaming propagation works)
- [ ] Published comparison showing 1000x scale advantage over competitors

**Deliverable:** Benchmark proving γ-CROWN is the ONLY verifier for billion-parameter models

---

## Total Effort Estimate

| Phase | Commits | Description |
|-------|---------|-------------|
| 1 | 5 | Foundation |
| 2 | 5 | Instance tracking |
| 3 | 5 | Competitor baselines |
| 4 | 3 | VNN-COMP scoring |
| 5 | 3 | Memory measurement |
| 6 | 3 | Counterexample validation |
| 7 | 5 | Reporting |
| 8 | 3 | Regression detection |
| 9 | 10 | All VNN-COMP benchmarks |
| 10 | 3 | Documentation |
| 10.5 | 13 | Model format support (SafeTensors, GGUF, PyTorch) |
| 11 | 8 | LLM & Target model benchmarks |
| **Total** | **66** | **~13 hours AI time** |

---

## Success Criteria

### Functional Requirements

- [ ] Single command runs ALL benchmarks: `pytest benchmarks/tests/`
- [ ] Every VNN-COMP 2021-2025 benchmark has tests
- [ ] Results exactly reproducible (same inputs → same outputs)
- [ ] Comparison against ALL major competitors
- [ ] VNN-COMP scoring matches official results
- [ ] Memory tracked for every instance
- [ ] Counterexamples validated
- [ ] Historical regression detection

### Non-Functional Requirements

- [ ] Full suite completes in <4 hours
- [ ] Clear, readable output
- [ ] Machine-parseable JSON results
- [ ] Publication-quality HTML reports
- [ ] CI-friendly (exit codes, --fail-on-regression)

### Quality Gates

Before merging, verify:
1. [ ] ACAS-Xu 2021 results match our published 100% resolution
2. [ ] VNN-COMP score calculation matches published 2024 results
3. [ ] Counterexample validation passes 100% (no false falsifications)
4. [ ] Memory measurements are reasonable (<10% variance across runs)
5. [ ] All competitor baselines have documented sources

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Competitor results unavailable | Can't compare | Use VNN-COMP published PDFs, contact authors |
| Scoring formula unclear | Wrong scores | Study VNN-COMP scoring scripts exactly |
| Memory measurement inaccurate | Misleading comparison | Use multiple measurement methods, average |
| Benchmarks too slow | CI timeout | Use subset for CI, full suite nightly |
| Counterexample validation fails | False confidence | Investigate every failure, never skip |

---

## Open Questions

1. **Hardware normalization**: Competitors ran on different hardware. How do we normalize time comparisons fairly?

2. **Incomplete baselines**: What if we can't get results for some competitor/benchmark combinations?

3. **Versioning**: Which version of competitor tools? VNN-COMP submission or latest?

4. **Instance variations**: Some benchmarks have random instances. How do we ensure same instances?

5. **Floating point**: Are results deterministic across machines? Need to test.

---

## Appendix: VNN-COMP Benchmark Categories

### VNN-COMP 2021
| Category | Networks | Properties | Instances | Notes |
|----------|----------|------------|-----------|-------|
| acasxu | 45 | 10 | 186 | Collision avoidance |
| mnist_fc | 3 | ~100 | ~100 | Fully connected |
| cifar10_resnet | 2 | ~100 | ~100 | ResNets |
| oval21 | 6 | ~600 | ~600 | Small networks |
| nn4sys | 4 | ~100 | ~100 | Systems |

### VNN-COMP 2023
| Category | Networks | Properties | Instances | Notes |
|----------|----------|------------|-----------|-------|
| acasxu | 45 | 10 | 186 | Same as 2021 |
| vit | 2 | ~50 | ~50 | Vision Transformer |
| vggnet16 | 1 | ~100 | ~100 | Large CNN |
| nn4sys | 5 | ~100 | ~100 | Systems |
| cgan | 2 | ~50 | ~50 | Generative |

### VNN-COMP 2024
| Category | Networks | Properties | Instances | Notes |
|----------|----------|------------|-----------|-------|
| acasxu_2023 | 45 | 10 | 186 | Same as before |
| vit_2023 | 2 | ~50 | ~50 | Vision Transformer |
| cifar100 | 3 | ~100 | ~100 | 100-class |
| tinyimagenet | 2 | ~50 | ~50 | ImageNet subset |
| cora | 2 | ~50 | ~50 | Graph NN |

### VNN-COMP 2025
| Category | Networks | Properties | Instances | Notes |
|----------|----------|------------|-----------|-------|
| acasxu_2023 | 45 | 10 | 186 | Same as before |
| soundnessbench | ? | ? | ? | **Tests verifier correctness** |
| malbeware | ? | ? | ? | Malware detection |
| sat_relu | ? | ? | ? | SAT-based |
| cersyve | ? | ? | ? | New |

---

## References

- VNN-COMP 2024 rules: https://github.com/ChristopherBrix/vnncomp2024_results
- α,β-CROWN: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- VNN-COMP benchmarks: https://github.com/VNN-COMP/vnncomp2025_benchmarks
- Scoring scripts: VNN-COMP repos under SCORING/

---

## WORKER DIRECTIVE

After this plan is approved:

1. Start with Phase 1 (Foundation)
2. Each phase should be a separate PR/commit series
3. Test after each phase before moving on
4. Document any deviations from this plan

**The goal is a benchmark framework that definitively proves γ-CROWN is #1.**
