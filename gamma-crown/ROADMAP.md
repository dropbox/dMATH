# Roadmap: gamma-crown

> Neural Network Verification via Bound Propagation

## Current Focus

Transformer verification and GPU acceleration.

## Active Issues

| # | Priority | Title |
|---|----------|-------|
| #28 | P2 | [RESEARCHER] Create system diagrams |
| #27 | P2 | Native compilation for bound propagation via tRust |
| #26 | P2 | Zipformer ASR Neural Network Verification |
| #23 | P2 | Transformer verification roadmap + papers |
| #22 | P2 | Transformer Verification + gamma-guide Crate |
| #21 | P3 | TLA+ DPLL(T) loop state machine spec |

## Verification Phases

1. **Foundation** - Complete
   - Core bound propagation (CROWN, α-CROWN)
   - ONNX model loading
   - Basic tensor operations

2. **GPU Acceleration** - In Progress
   - Metal/wgpu backend
   - 1000x+ speedup target

3. **Transformer Support** - Active
   - Attention mechanism verification
   - Whisper model support
   - Foundation model diffing

4. **Production** - Planned
   - Python bindings
   - CLI tooling
   - Integration with z4

## Dependencies

- **z4** - SMT solving for verification queries
- **model_mlx_migration** - MLX model support
