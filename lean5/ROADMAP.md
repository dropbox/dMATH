# Roadmap: lean5

> Pure Rust Theorem Prover

## Current Focus

Tactic framework completion (currently 70%) and Mathlib compatibility.

## Active Issues

| # | Priority | Title |
|---|----------|-------|
| #13 | P2 | [RESEARCHER] Create system diagrams |
| #12 | P2 | TLAPS backend requirements for TLA2 |
| #11 | P2 | [MANAGER] Roadmap Audit |
| #10 | P2 | Geometry benchmarking + certificate plan |
| #9 | P2 | Axiom takeaways: PutnamBench + tactic mining |
| #8 | P2 | PutnamBench Target + Axiom Tactic Mining |

## Implementation Status

| Component | Lines | Tests | Status |
|-----------|-------|-------|--------|
| Kernel | 169K | 2,554 | Complete |
| Parser | 8.8K | 228 | 97% compatible |
| Elaborator | 44.5K | 831 | Complete |
| Automation | 15.8K | 223 | Complete |
| Server | 7.7K | 138 | Complete |
| C Verification | 20K | 296 | Complete |
| Rust Semantics | 5.7K | 90 | Complete |
| Self-Verification | 4.4K | 31 | Complete |
| .olean Import | 17K | 263 | Complete |
| Tactic Framework | - | - | 70% (120 tactics) |

## Phases

| Phase | Status |
|-------|--------|
| 1-8: Core | Complete |
| 9: .olean import | Complete |
| 10: Tactics | 70% |
| 11-14: Macros, Lake, LSP | Planned |
| 15: Full Mathlib | Planned |

## Performance

| Operation | Latency |
|-----------|---------|
| `infer_type` | 20-100ns |
| `is_def_eq` | 1.4-28ns |
| `whnf` | 16-117ns |
| JSON-RPC check | 485ns |
| Batch throughput | 1M ops/sec |
