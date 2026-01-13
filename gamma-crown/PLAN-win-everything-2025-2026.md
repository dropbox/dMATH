# PLAN: Win Everything 2025-2026

**Strategy:** Leverage γ-CROWN's verification engine across multiple competition domains.

---

## Competition Portfolio

### Tier 1: Primary Focus (Direct γ-CROWN Application)

| Competition | Deadline | Focus | γ-CROWN Role |
|-------------|----------|-------|--------------|
| **VNN-COMP 2025** | ~July 2025 | Neural network verification | Core application |
| **VNN-COMP 2026** | ~July 2026 | Neural network verification | Win outright |
| **SAT Competition 2025** | May 7, 2025 | Boolean satisfiability | z4 solver entry |

### Tier 2: High Synergy (Verification + Optimization)

| Competition | Deadline | Focus | γ-CROWN Role |
|-------------|----------|-------|--------------|
| **PACE 2025** | June 13, 2025 | Dominating/Hitting Set | z4 + verified bounds |
| **ROADEF/EURO Challenge** | TBA (EURO 2025) | Industrial optimization | Verified ML surrogates |
| **INFORMS TSL Challenge** | April 1, 2025 | Transportation/Logistics | Verified routing/forecasting |

### Tier 3: Research Recognition

| Prize | Deadline | Focus | Submission |
|-------|----------|-------|------------|
| **Wagner Prize** | May 15, 2025 | OR Practice | Verified ML case study |
| **Dantzig Dissertation Award** | June 30, 2026 | OR Dissertation | If applicable |
| **INFORMS Impact Prize** | April 30, 2025 | Impactful OR work | γ-CROWN impact story |

---

## Detailed Competition Strategy

### 1. VNN-COMP 2025/2026 (Primary)

**Status:** Current focus per PLAN-gamma-crown-v3.md

**Goal:**
- 2025: Top 3 finish, establish credibility
- 2026: Win outright

**Key advantages:**
- Sound-first design (no false UNSATs)
- z4 DPLL(T) integration for hard instances
- GPU acceleration for throughput

### 2. SAT Competition 2025

**Deadline:** May 7, 2025

**Entry:** z4 solver (we own it)

**Tracks:**
- Main Track: General SAT instances
- Parallel Track: Multi-core solving

**Preparation:**
- [ ] Benchmark z4 on SAT Competition 2024 instances
- [ ] Profile and optimize hot paths
- [ ] Add competition-specific tuning (restart policies, VSIDS decay)

**Synergy:** SAT solving improvements directly benefit γ-CROWN's DPLL(T) backend.

### 3. PACE 2025 (Parameterized Algorithms)

**Deadline:** June 13, 2025

**Problems:**
- Dominating Set (Exact, Heuristic, Lite)
- Hitting Set (Exact, Heuristic, Lite)

**Approach:**
```
Encode problem as SAT/SMT → Solve with z4
Use verified bounds to prune search space
Hybrid: heuristic for upper bound, exact for optimality proof
```

**Preparation:**
- [ ] Implement Dominating Set → SAT reduction
- [ ] Implement Hitting Set → SAT reduction
- [ ] Test on PACE 2024 instances
- [ ] Tune for competition scoring

### 4. INFORMS TSL Data-Driven Research Challenge

**Deadline:** April 1, 2025 (SOON)

**Topic:** Data-driven transportation/logistics research

**Submission idea:** "Verified Neural Networks for Robust Logistics Optimization"
- Train NN for travel time / demand prediction
- Compute verified bounds
- Show robust routing/inventory decisions
- Compare vs conformal prediction baselines

**Preparation:**
- [ ] Identify suitable public dataset (NYC taxi, Chicago rideshare)
- [ ] Train small forecasting network
- [ ] Demonstrate verified bounds integration
- [ ] Write 10-page research paper

### 5. ROADEF/EURO Challenge (TBA at EURO 2025)

**Announcement:** July 6-9, 2025 at EURO Leeds

**Historical topics:**
- 2022: 3D truck loading (Renault)
- 2020: Maintenance scheduling (RTE)
- 2018: Glass cutting (Saint-Gobain)

**Preparation:**
- Monitor EURO 2025 for announcement
- Typical structure: industrial partner provides problem + data
- Approach: ML surrogate + verified bounds for robustness

### 6. Wagner Prize (OR Practice Excellence)

**Deadline:** May 15, 2025

**Prize:** $5,000

**Requirement:** Quality OR analysis with verifiable success in practice

**Submission idea:** Case study of verified ML in operations
- Partner with company using ML for operations
- Show verification catches edge cases
- Quantify business value of guaranteed bounds

**Status:** Requires industry partnership or compelling case study

---

## Timeline

```
2025
├── April 1: INFORMS TSL Challenge deadline
├── April 30: INFORMS Impact Prize deadline
├── May 7: SAT Competition 2025 solver submission
├── May 15: Wagner Prize deadline
├── June 13: PACE 2025 submission
├── July 6-9: EURO 2025 (ROADEF/EURO challenge announced)
├── July: VNN-COMP 2025
└── Nov 24: UPS Smith Prize deadline

2026
├── Jan 10: AtCoder Heuristic 059
├── June 30: Dantzig Dissertation Award
├── July: VNN-COMP 2026 (WIN)
└── July 31: INFORMS Case Competition
```

---

## Resource Allocation

### Phase 0: Now → April 2025

**Focus:** Fix γ-CROWN bugs, prepare TSL Challenge submission

| Task | Priority | Effort |
|------|----------|--------|
| Fix ViT shape mismatch | P0 | Phase 0 |
| TSL Challenge paper draft | P1 | 2 weeks |
| z4 SAT Competition benchmark | P2 | 1 week |

### Phase 1: April → June 2025

**Focus:** SAT Competition, PACE 2025

| Task | Priority | Effort |
|------|----------|--------|
| z4 optimization for SAT Comp | P1 | Ongoing |
| PACE problem encodings | P1 | 2 weeks |
| γ-CROWN Phase 1-2 (typed core, DPLL(T)) | P0 | Per v3 plan |

### Phase 2: July → December 2025

**Focus:** VNN-COMP 2025, ROADEF/EURO prep

| Task | Priority | Effort |
|------|----------|--------|
| VNN-COMP 2025 competition | P0 | Major |
| ROADEF/EURO challenge (if announced) | P1 | TBD |
| γ-CROWN Phase 3-4 (proofs, GPU) | P0 | Per v3 plan |

### Phase 3: 2026

**Focus:** Win VNN-COMP 2026, research publications

| Task | Priority | Effort |
|------|----------|--------|
| VNN-COMP 2026 | P0 | Major |
| Management Science / OR paper | P1 | Ongoing |
| INFORMS Annual presentation | P2 | Abstract |

---

## Cross-Competition Synergies

```
z4 Solver
├── SAT Competition 2025 (direct entry)
├── PACE 2025 (SAT encoding backend)
└── γ-CROWN (DPLL(T) theory solver)

γ-CROWN Verification
├── VNN-COMP 2025/2026 (direct entry)
├── INFORMS TSL Challenge (verified forecasting)
├── ROADEF/EURO (verified surrogates)
└── Wagner Prize (OR practice case)

Verified ML for OR
├── Research paper (Management Science)
├── INFORMS Annual talk
└── Industry partnerships
```

---

## Success Metrics

### 2025 Goals

| Competition | Target |
|-------------|--------|
| VNN-COMP 2025 | Top 3 |
| SAT Competition 2025 | Top 10 |
| PACE 2025 | Top 5 in one track |
| INFORMS TSL | Honorable mention |

### 2026 Goals

| Competition | Target |
|-------------|--------|
| VNN-COMP 2026 | **WIN** |
| SAT Competition 2026 | Top 5 |
| Management Science paper | Accept |

---

## Competitions NOT Targeting (Low Synergy)

| Competition | Reason |
|-------------|--------|
| Kaggle ML competitions | Generic ML, not verification |
| KDD Cup | Usually NLP/recommender focused |
| ImageNet challenges | Classification accuracy, not verification |
| Robotics challenges | Hardware dependent |

---

## Immediate Actions

### This Week
1. [ ] Fix ViT shape mismatch (worker directive active)
2. [ ] Draft INFORMS TSL abstract (April 1 deadline)

### This Month
1. [ ] Benchmark z4 on SAT Competition 2024 instances
2. [ ] Prototype verified demand forecasting demo
3. [ ] Submit INFORMS TSL Challenge entry

### Q1 2025
1. [ ] Complete γ-CROWN Phase 1 (typed core)
2. [ ] Submit z4 to SAT Competition
3. [ ] Implement PACE problem encodings
