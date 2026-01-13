# Research Direction: Neural Network Verification for Operations Research

**Target Venue:** INFORMS Annual Meeting, Management Science, Operations Research, M&SOM

**Core Thesis:** Formal verification of neural networks enables *provably robust* decision-making in operations, replacing heuristic uncertainty quantification with mathematical guarantees.

---

## The Problem: ML in Operations is Fragile

Operations research increasingly uses ML for:
- Demand forecasting
- Lead time prediction
- Pricing optimization
- Resource allocation
- Routing and scheduling

**The fragility problem:**
```
1. Train ML model f(x) on historical data
2. Use f(x) in optimization: min cost s.t. f(x) ≤ capacity
3. Deploy to production
4. ML model encounters distribution shift
5. f(x) underestimates demand → stockout
6. Or f(x) overestimates → excess inventory
```

**Current mitigations (all heuristic):**
- Add safety stock buffer (how much?)
- Ensemble models (which one to trust?)
- Conformal prediction intervals (only probabilistic)
- Robust optimization with uncertainty sets (how to construct?)

**What's missing:** Mathematical guarantees that hold for ALL inputs in a specified region.

---

## The Solution: Verified ML for Operations

Neural network verification computes **sound bounds** on model outputs:

```
Given: Neural network f, input region X = {x : l ≤ x ≤ u}
Compute: [L, U] such that ∀x ∈ X: L ≤ f(x) ≤ U
Guarantee: Mathematical proof, not statistical estimate
```

**Application to robust operations:**

```
Traditional:     min c'x s.t. A*x ≤ b, f(x) ≤ d
Verified robust: min c'x s.t. A*x ≤ b, U(f, X) ≤ d

where U(f, X) = verified upper bound of f over input region X
```

If the verified robust problem is feasible, the solution is **guaranteed feasible** for any realization within X.

---

## Research Directions

### 1. Verified Demand Forecasting for Inventory Optimization

**Setting:** Retailer uses neural network to forecast demand, sets inventory levels.

**Current approach:**
```python
demand_forecast = neural_net(features)
safety_stock = z_alpha * historical_std  # heuristic
order_quantity = demand_forecast + safety_stock
```

**Verified approach:**
```python
demand_lower, demand_upper = verify_bounds(neural_net, feature_region)
# Guaranteed: actual demand ∈ [demand_lower, demand_upper] if features ∈ region
order_quantity = demand_upper  # Never stockout within verified region
```

**Research questions:**
1. How tight are verified bounds vs conformal intervals?
2. Cost of conservatism: verified bounds vs probabilistic bounds?
3. How to construct meaningful feature regions (perturbation sets)?

**Contribution:** First provably robust inventory policy using NN verification.

---

### 2. Safe Reinforcement Learning for Warehouse Operations

**Setting:** RL agent controls warehouse robots, conveyors, picking sequences.

**Safety requirements:**
- Robots never collide
- Throughput never drops below threshold
- Energy consumption within limits

**Current approach:** Hope the trained policy is safe, add heuristic constraints.

**Verified approach:**
```
Policy: π(state) → action
Safety property: ∀ states in S_safe, π(state) ∈ A_safe

Verification: Prove the property holds for all states in region
If verified: Deploy with mathematical safety guarantee
If counterexample: Retrain or add constraint
```

**Research questions:**
1. How to specify safety regions for warehouse states?
2. Scalability: can we verify policies for realistic warehouse sizes?
3. Verified policy vs constrained MDP: performance comparison?

**Contribution:** First formally verified RL policies for logistics.

---

### 3. Algorithmic Pricing with Fairness Guarantees

**Setting:** ML model sets prices based on customer features.

**Legal/ethical requirements:**
- No price discrimination by protected class
- Prices within regulatory bounds
- Consistent pricing (similar customers → similar prices)

**Verification formulation:**
```
Property: ∀ customers x, x' where x and x' differ only in protected attribute:
          |price(x) - price(x')| ≤ ε

Verification: Prove this holds for all customers in specified region
Result: Mathematical certificate of non-discrimination
```

**Research questions:**
1. How to formalize fairness as verifiable properties?
2. Trade-off: fairness constraints vs revenue optimization?
3. Can verification replace fairness audits?

**Contribution:** Provably fair pricing algorithms.

---

### 4. Robust Supply Chain Network Design

**Setting:** Design supply chain network using ML-predicted disruption probabilities.

**Problem:** ML predictions are point estimates; network design is long-term.

**Current approach:**
```
min network_cost
s.t. expected_service_level(ML_predictions) ≥ target
```

**Verified approach:**
```
For each supplier/route, compute verified bounds on:
- Lead time: [L_min, L_max]
- Disruption probability: [p_min, p_max]
- Capacity: [C_min, C_max]

min network_cost
s.t. worst_case_service_level(verified_bounds) ≥ target
```

**Research questions:**
1. How to propagate verified bounds through network optimization?
2. Computational tractability of verified robust network design?
3. Value of verification vs scenario-based robust optimization?

**Contribution:** Supply chain design with ML + formal guarantees.

---

### 5. Verified Surrogate Models for Simulation Optimization

**Setting:** Expensive simulation (factory, hospital, logistics) approximated by neural network surrogate.

**Problem:** Surrogate model is fast but may be inaccurate. Optimization may find "optimal" point that is actually infeasible in true simulation.

**Verified approach:**
```
1. Train surrogate f_NN to approximate simulation f_sim
2. Compute verification bounds: f_NN(x) ∈ [L(x), U(x)]
3. Calibrate: ensure f_sim(x) ∈ [L(x) - δ, U(x) + δ] on test set
4. Optimize: min c(x) s.t. U(x) + δ ≤ constraint

Guarantee: If surrogate + bounds are valid, solution is feasible for true simulation
```

**Research questions:**
1. How to calibrate verified bounds to true simulation?
2. Tightness vs coverage trade-off?
3. When is verified surrogate better than Gaussian process?

**Contribution:** Surrogate-based optimization with formal feasibility guarantees.

---

## Technical Approach

### Verification Methods for OR Applications

| Method | Speed | Tightness | Best For |
|--------|-------|-----------|----------|
| **IBP** | Very fast | Loose | Quick feasibility check |
| **CROWN** | Fast | Medium | Most OR applications |
| **α-CROWN** | Medium | Tight | When tightness matters |
| **MILP** | Slow | Exact | Small networks, certificates |

### Integration with Optimization Solvers

```
Verified NN Bounds + Optimization Solver
├── Linear bounds → LP/MILP as linear constraints
├── Convex relaxation → Second-order cone programming
└── Exact encoding → MILP with big-M (expensive)
```

**Key insight:** CROWN produces linear bounds, which integrate directly into LP/MILP solvers used in OR.

### Computational Considerations

| Network Size | Verification Time | OR Applicability |
|--------------|-------------------|------------------|
| Small (< 1K params) | < 1 sec | Real-time decisions |
| Medium (1K-100K) | 1-60 sec | Batch optimization |
| Large (100K-1M) | Minutes | Strategic planning |
| Very large (> 1M) | Research frontier | Future work |

Most OR applications use small-medium networks → verification is practical.

---

## Comparison: Verification vs Existing Robust Methods

| Approach | Guarantee | Assumption | Computation |
|----------|-----------|------------|-------------|
| **Point forecast + buffer** | None | Heuristic | O(1) |
| **Conformal prediction** | Probabilistic | Exchangeability | O(n) |
| **Bayesian NN** | Probabilistic | Prior correct | O(expensive) |
| **Robust optimization** | Worst-case | Uncertainty set known | O(optimization) |
| **NN Verification** | Mathematical | Input region specified | O(network size) |

**Verification advantage:** No distributional assumptions. If input is in region, bound holds.

**Verification limitation:** Must specify input region. Garbage in, garbage out.

---

## Empirical Validation Plan

### Dataset 1: Retail Demand Forecasting
- Data: UCI Online Retail, Kaggle M5
- Model: LSTM/Transformer for demand
- Baseline: Conformal prediction intervals
- Metric: Coverage, interval width, inventory cost

### Dataset 2: Routing with Travel Time Prediction
- Data: NYC taxi, Chicago ride-share
- Model: GNN for travel time
- Baseline: Robust shortest path with ellipsoidal uncertainty
- Metric: On-time arrival, path cost

### Dataset 3: Healthcare Scheduling
- Data: Hospital patient flow (synthetic or partnership)
- Model: NN for patient arrival/service time
- Baseline: Robust appointment scheduling
- Metric: Wait time, overtime, patient throughput

---

## Paper Outline (Management Science / Operations Research)

**Title:** "Provably Robust Operations with Verified Neural Networks"

**Abstract:**
- ML increasingly used in operations decisions
- Current uncertainty quantification is probabilistic, not guaranteed
- We propose using formal NN verification for provable robustness
- Demonstrate on [demand forecasting / routing / scheduling]
- Results: [X]% tighter than robust optimization, guaranteed feasible

**Sections:**
1. Introduction: ML in operations + fragility problem
2. Background: NN verification (accessible to OR audience)
3. Framework: Verified ML for robust optimization
4. Application: [Primary domain]
5. Computational Study
6. Managerial Insights
7. Conclusion

**Contribution positioning:**
- First paper to apply formal verification to operations decisions
- Bridge between ML verification (CS) and robust optimization (OR)
- Practical algorithms with theoretical guarantees

---

## INFORMS Submission Timeline

| Conference | Deadline | Format |
|------------|----------|--------|
| **INFORMS Annual 2026** | ~May 2026 | Abstract (talk) |
| **MSOM 2026** | ~Feb 2026 | Full paper |
| **Management Science** | Rolling | Full paper |
| **Operations Research** | Rolling | Full paper |
| **M&SOM** | Rolling | Full paper |

**Strategy:**
1. Submit abstract to INFORMS Annual for visibility
2. Develop full paper for journal submission
3. Target Management Science (highest impact) or Operations Research

---

## Why This Matters

**For OR community:**
- New tool for robust decision-making
- Bridges ML and optimization formally
- Enables ML adoption with guarantees

**For verification community:**
- New application domain with practical impact
- Different network architectures (time series, GNN)
- Real-world deployment requirements

**For practice:**
- ML with guarantees → safer deployment
- Reduces reliance on heuristic buffers
- Regulatory compliance (auditable guarantees)

---

## Next Steps

1. **Prototype:** Verified demand forecasting on public dataset
2. **Baseline comparison:** vs conformal prediction, robust optimization
3. **Write-up:** Target INFORMS Annual abstract (May deadline)
4. **Full paper:** Target Management Science or Operations Research

---

## Connection to γ-CROWN

γ-CROWN provides the verification engine. OR applications require:
- Fast bounds for operational decisions (IBP/CROWN)
- Integration with optimization solvers (linear bounds → LP)
- Scalability to production networks (GPU acceleration)

The VNN-COMP work builds the core engine. OR applications are a high-impact use case that motivates the engineering investment.
