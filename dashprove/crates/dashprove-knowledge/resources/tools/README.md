# DashProve Tool Knowledge Base

This directory contains comprehensive knowledge entries for all 150+ verification tools.

## Directory Structure

```
tools/
├── rust/
│   ├── formal/          # Kani, Verus, Creusot, Prusti, Flux, MIRAI, Rudra, Miri
│   ├── sanitizers/      # ASAN, MSAN, TSAN, LSAN
│   ├── concurrency/     # Loom, Shuttle, CDSChecker, GenMC
│   ├── fuzzing/         # LibFuzzer, AFL, Honggfuzz, Bolero
│   ├── pbt/             # Proptest, QuickCheck
│   └── static/          # Clippy, SemverChecks, Audit, Deny, etc.
├── theorem_provers/
│   ├── dependent/       # Lean4, Coq, Agda, Idris, F*, ATS
│   ├── classical/       # Isabelle, ACL2, HOL4, HOL Light, PVS, Mizar
│   └── specialized/     # Metamath, etc.
├── smt_sat/
│   ├── smt/            # Z3, CVC5, Yices, Boolector, MathSAT, etc.
│   └── sat/            # MiniSat, Glucose, CaDiCaL, Kissat, etc.
├── model_checkers/
│   ├── temporal/       # TLA+, Apalache, SPIN, NuSMV, UPPAAL
│   └── software/       # CBMC, ESBMC, CPAchecker, SeaHorn, JPF
├── program_verification/
│   ├── c_cpp/          # Frama-C, VCC, VeriFast
│   ├── java/           # KeY, OpenJML
│   └── other/          # SPARK, Why3, Stainless, LiquidHaskell
├── neural_network/
│   ├── verifiers/      # Marabou, CROWN, ERAN, NNV, VeriNet, Venus
│   └── adversarial/    # ART, Foolbox, CleverHans, TextAttack
├── ai_ml/
│   ├── optimization/   # ONNX, TensorRT, TVM, OpenVINO
│   ├── compression/    # NNCF, Brevitas, AIMET
│   ├── quality/        # GreatExpectations, Deepchecks, Evidently
│   ├── fairness/       # Fairlearn, AIF360, Aequitas
│   └── interpretability/ # SHAP, LIME, Captum
├── llm/
│   ├── guardrails/     # GuardrailsAI, NeMo, Guidance
│   ├── evaluation/     # Promptfoo, TruLens, Ragas, DeepEval
│   └── hallucination/  # SelfCheckGPT, FactScore
├── security/
│   ├── protocol/       # Tamarin, ProVerif, Verifpal
│   └── crypto/         # EasyCrypt, CryptoVerif, Jasmin
├── distributed/        # P, Ivy, mCRL2, CADP
├── hardware/           # Yosys, SymbiYosys
└── symbolic_exec/      # KLEE, Angr, Manticore
```

## Entry Format

Each tool has a JSON file with this schema:

```json
{
  "id": "tool_id",
  "name": "Display Name",
  "category": "category_name",
  "subcategory": "subcategory_name",

  "description": "Brief description of the tool",
  "long_description": "Detailed description...",

  "capabilities": ["cap1", "cap2"],
  "property_types": ["theorem", "contract", "temporal"],
  "input_languages": ["rust", "c", "java"],
  "output_formats": ["proof", "counterexample", "trace"],

  "installation": {
    "methods": [
      {"type": "cargo", "command": "cargo install tool"},
      {"type": "brew", "command": "brew install tool"},
      {"type": "pip", "command": "pip install tool"},
      {"type": "source", "url": "https://github.com/..."}
    ],
    "dependencies": ["z3", "llvm"],
    "platforms": ["linux", "macos", "windows"]
  },

  "documentation": {
    "official": "https://...",
    "tutorial": "https://...",
    "api_reference": "https://...",
    "examples": "https://..."
  },

  "tactics": [
    {
      "name": "tactic_name",
      "description": "What it does",
      "syntax": "tactic arg1 arg2",
      "when_to_use": "Use when...",
      "examples": ["example1", "example2"]
    }
  ],

  "error_patterns": [
    {
      "pattern": "regex or string pattern",
      "meaning": "What this error means",
      "common_causes": ["cause1", "cause2"],
      "fixes": ["fix1", "fix2"]
    }
  ],

  "integration": {
    "dashprove_backend": true,
    "usl_property_types": ["theorem", "contract"],
    "cli_command": "dashprove verify --backend tool"
  },

  "performance": {
    "typical_runtime": "seconds to minutes",
    "scalability": "Handles up to N lines of code",
    "memory_usage": "typical RAM needed"
  },

  "comparisons": {
    "similar_tools": ["tool1", "tool2"],
    "advantages": ["adv1", "adv2"],
    "disadvantages": ["dis1", "dis2"]
  },

  "metadata": {
    "version": "1.0.0",
    "last_updated": "2025-12-20",
    "maintainer": "organization",
    "license": "MIT/Apache/etc"
  }
}
```

## Building the Knowledge Base

Run: `dashprove knowledge build --tools`

This will:
1. Validate all JSON entries
2. Build embeddings for semantic search
3. Create indexes for keyword search
4. Generate relationship graphs
5. Export to RAG-ready format

## Querying

```rust
// Via dashprove-knowledge crate
let kb = KnowledgeBase::load()?;

// Find tools for a capability
let tools = kb.find_by_capability("memory_safety");

// Find similar tools
let similar = kb.find_similar("kani");

// Get error fix suggestions
let fixes = kb.suggest_fix("unwinding assertion loop 0");

// Get tactic recommendations
let tactics = kb.recommend_tactics(&property, &context);
```
