# Polynomial Commitments and Folding Schemes for Lean5

**Author:** Worker #565
**Date:** 2026-01-06
**Status:** Draft
**Issue:** #1 (from leader)

## Overview

This document describes the design for adding polynomial commitment schemes (KZG/IPA) and Nova-style folding to lean5 for proof compression and incremental verification.

## Background

### Current Proof Certificate System

Lean5's `lean5-kernel/src/cert/mod.rs` implements proof certificates that:
1. Witness type-correctness derivations (ProofCert enum)
2. Support structural compression via structure-sharing
3. Support byte-level compression via LZ4/Zstd
4. Enable batch verification

**Key structures:**
- `ProofCert`: Tree-structured certificate mirroring expression structure
- `CompressedCert`: Hash-consed representation with `CertIdx` references
- `CertArchive`: Serialization format with streaming support

### Problem Statement

Proof certificates can be large for complex proofs. Polynomial commitments enable:
1. **Smaller proof files**: Commit to polynomial representation instead of full structure
2. **Batch verification**: Verify multiple proofs with single pairing check
3. **Incremental updates**: Update proofs without full reconstruction

---

## Architecture

### New Crates

```
crates/
├── lean5-commit/     # Polynomial commitments (KZG/IPA)
│   ├── src/
│   │   ├── lib.rs
│   │   ├── kzg.rs         # Kate-Zaverucha-Goldberg commitments
│   │   ├── ipa.rs         # Inner Product Argument commitments
│   │   ├── encoding.rs    # ProofCert → polynomial encoding
│   │   ├── committed.rs   # CommittedProof type
│   │   └── verify.rs      # Commitment verification
│   └── Cargo.toml
│
└── lean5-fold/       # Nova-style folding schemes
    ├── src/
    │   ├── lib.rs
    │   ├── r1cs.rs        # Relaxed R1CS representation
    │   ├── folding.rs     # Folding operation
    │   ├── ivc.rs         # Incrementally Verifiable Computation
    │   └── transcript.rs  # Fiat-Shamir transcript
    └── Cargo.toml
```

### Dependency Choice: arkworks vs. custom

**Decision: Use arkworks ecosystem**

Rationale:
1. Production-tested cryptographic implementations
2. Modular design (ark-ec, ark-ff, ark-poly)
3. Multiple commitment scheme implementations
4. Active maintenance

Dependencies:
```toml
# lean5-commit/Cargo.toml
[dependencies]
ark-ff = "0.4"
ark-ec = "0.4"
ark-poly = "0.4"
ark-poly-commit = "0.4"
ark-bls12-381 = "0.4"  # For KZG
ark-serialize = "0.4"
ark-std = "0.4"

lean5-kernel = { path = "../lean5-kernel" }
```

---

## Part 1: Polynomial Commitments (lean5-commit)

### ProofCert → Polynomial Encoding

The key challenge is encoding a tree-structured `ProofCert` as polynomial evaluations.

**Approach: Merkle-tree commitment with polynomial witnesses**

```rust
/// A polynomial encoding of a proof certificate
pub struct EncodedProof {
    /// Root commitment to the proof tree
    root: Commitment,
    /// Polynomial representing the flattened certificate
    poly: DensePolynomial<Fr>,
    /// Evaluation proofs for verification points
    opening_proofs: Vec<OpeningProof>,
}

/// Encode a ProofCert as a polynomial
pub fn encode_cert(cert: &ProofCert) -> DensePolynomial<Fr> {
    // 1. Flatten certificate to sequence of field elements
    let elements = flatten_cert(cert);

    // 2. Interpolate polynomial through points (i, elements[i])
    let domain = GeneralEvaluationDomain::new(elements.len()).unwrap();
    DensePolynomial::from_coefficients_vec(domain.ifft(&elements))
}
```

**Flattening strategy:**

```rust
/// Convert cert node to field elements
fn flatten_node(cert: &ProofCert, out: &mut Vec<Fr>) {
    match cert {
        ProofCert::Sort { level } => {
            out.push(Fr::from(0u64)); // tag: Sort
            out.push(encode_level(level));
        }
        ProofCert::App { fn_cert, fn_type, arg_cert, result_type } => {
            out.push(Fr::from(1u64)); // tag: App
            flatten_node(fn_cert, out);
            flatten_expr(fn_type, out);
            flatten_node(arg_cert, out);
            flatten_expr(result_type, out);
        }
        // ... other variants
    }
}
```

### KZG Implementation

```rust
/// KZG polynomial commitment scheme for proof certificates
pub struct KzgProofCommitment {
    /// Structured reference string (trusted setup)
    srs: KzgSrs,
}

impl KzgProofCommitment {
    /// Generate commitment to a proof certificate
    pub fn commit(&self, cert: &ProofCert) -> Result<CommittedProof, CommitError> {
        let poly = encode_cert(cert);
        let commitment = self.srs.commit(&poly)?;

        Ok(CommittedProof {
            commitment,
            degree: poly.degree(),
        })
    }

    /// Verify a committed proof at random challenge points
    pub fn verify(
        &self,
        committed: &CommittedProof,
        challenge: Fr,
        claimed_eval: Fr,
        proof: &OpeningProof,
    ) -> Result<bool, VerifyError> {
        self.srs.verify(
            &committed.commitment,
            challenge,
            claimed_eval,
            proof,
        )
    }
}
```

### IPA Implementation (Transparent Setup)

For applications requiring no trusted setup:

```rust
/// Inner Product Argument commitment (no trusted setup)
pub struct IpaProofCommitment {
    /// Public parameters (generated via hash)
    pp: IpaParams,
}

impl IpaProofCommitment {
    /// Create with transparent setup
    pub fn new(max_degree: usize) -> Self {
        let pp = IpaParams::setup(max_degree);
        Self { pp }
    }

    pub fn commit(&self, cert: &ProofCert) -> Result<CommittedProof, CommitError> {
        let poly = encode_cert(cert);
        let commitment = self.pp.commit(&poly)?;

        Ok(CommittedProof {
            commitment,
            degree: poly.degree(),
        })
    }
}
```

### Public API

```rust
// crates/lean5-commit/src/lib.rs

pub mod encoding;
pub mod kzg;
pub mod ipa;
pub mod verify;

/// Commitment scheme trait
pub trait ProofCommitmentScheme {
    type Commitment;
    type OpeningProof;
    type Error;

    /// Commit to a proof certificate
    fn commit(&self, cert: &ProofCert) -> Result<Self::Commitment, Self::Error>;

    /// Open commitment at a point
    fn open(&self, cert: &ProofCert, point: Fr)
        -> Result<(Fr, Self::OpeningProof), Self::Error>;

    /// Verify opening proof
    fn verify(
        &self,
        commitment: &Self::Commitment,
        point: Fr,
        value: Fr,
        proof: &Self::OpeningProof,
    ) -> Result<bool, Self::Error>;

    /// Batch verify multiple commitments
    fn batch_verify(
        &self,
        items: &[(Self::Commitment, Fr, Fr, Self::OpeningProof)],
    ) -> Result<bool, Self::Error>;
}
```

---

## Part 2: Folding Schemes (lean5-fold)

### Overview

Nova-style folding enables composing proofs incrementally without proof size blowup.

**Key insight:** Instead of proving `F(x₁) ∧ F(x₂)` separately, "fold" instances:
- Two instances `(u₁, w₁)` and `(u₂, w₂)` fold into one `(u, w)`
- Verifier work is constant per fold step
- Final verification is single instance check

### Relaxed R1CS

Nova uses "relaxed R1CS" to enable folding:

```rust
/// Standard R1CS: Az ∘ Bz = Cz
/// Relaxed R1CS: Az ∘ Bz = u·Cz + E
pub struct RelaxedR1CS {
    /// Constraint matrices
    a: SparseMatrix<Fr>,
    b: SparseMatrix<Fr>,
    c: SparseMatrix<Fr>,

    /// Relaxation scalar (u=1 for standard R1CS)
    u: Fr,

    /// Error vector
    e: Vec<Fr>,
}
```

### ProofCert → R1CS Encoding

Convert proof verification into R1CS constraints:

```rust
/// Encode proof certificate verification as R1CS
pub fn cert_to_r1cs(cert: &ProofCert, env: &Environment) -> R1CSInstance {
    let mut r1cs = R1CSBuilder::new();

    // Each certificate node generates verification constraints
    encode_cert_constraints(&mut r1cs, cert, env);

    r1cs.finalize()
}

fn encode_cert_constraints(
    r1cs: &mut R1CSBuilder,
    cert: &ProofCert,
    env: &Environment,
) {
    match cert {
        ProofCert::App { fn_cert, fn_type, arg_cert, result_type } => {
            // Constraint 1: fn_cert verifies to Pi type
            let fn_type_var = r1cs.alloc_expr(fn_type);
            encode_cert_constraints(r1cs, fn_cert, env);
            r1cs.constrain_is_pi(fn_type_var);

            // Constraint 2: arg_cert matches domain type
            encode_cert_constraints(r1cs, arg_cert, env);
            // ... additional constraints
        }
        // ... other cert types
    }
}
```

### Folding Operation

```rust
/// Fold two relaxed R1CS instances
pub fn fold(
    inst1: &RelaxedR1CSInstance,
    inst2: &RelaxedR1CSInstance,
    r: Fr,  // Random challenge
) -> RelaxedR1CSInstance {
    RelaxedR1CSInstance {
        // Linearly combine commitments
        commit_w: inst1.commit_w + inst2.commit_w * r,
        commit_e: inst1.commit_e + inst2.commit_e * r,

        // Combine scalars
        u: inst1.u + inst2.u * r,

        // Public inputs combine similarly
        x: inst1.x.iter()
            .zip(&inst2.x)
            .map(|(a, b)| *a + *b * r)
            .collect(),
    }
}
```

### IVC (Incrementally Verifiable Computation)

```rust
/// IVC proof for sequence of certificate verifications
pub struct IvcProof {
    /// Current folded instance
    running_instance: RelaxedR1CSInstance,

    /// Commitment to running witness
    running_witness_commitment: Commitment,

    /// Step counter
    step: u64,
}

impl IvcProof {
    /// Extend IVC proof with new certificate
    pub fn extend(
        &mut self,
        new_cert: &ProofCert,
        env: &Environment,
    ) -> Result<(), IvcError> {
        // 1. Generate R1CS instance for new cert
        let new_instance = cert_to_r1cs(new_cert, env);

        // 2. Generate Fiat-Shamir challenge
        let r = self.transcript.squeeze();

        // 3. Fold new instance into running instance
        self.running_instance = fold(&self.running_instance, &new_instance, r);
        self.step += 1;

        Ok(())
    }

    /// Verify IVC proof
    pub fn verify(&self, env: &Environment) -> Result<bool, VerifyError> {
        // Verify final folded instance satisfies relaxed R1CS
        verify_relaxed_r1cs(&self.running_instance)
    }
}
```

### Public API

```rust
// crates/lean5-fold/src/lib.rs

pub mod r1cs;
pub mod folding;
pub mod ivc;
pub mod transcript;

/// Create IVC proof starting from initial certificate
pub fn start_ivc(cert: &ProofCert, env: &Environment) -> Result<IvcProof, IvcError>;

/// Extend IVC proof with additional certificate
pub fn extend_ivc(
    proof: &mut IvcProof,
    cert: &ProofCert,
    env: &Environment,
) -> Result<(), IvcError>;

/// Verify IVC proof
pub fn verify_ivc(proof: &IvcProof, env: &Environment) -> Result<bool, IvcError>;

/// Compress IVC proof to succinct form (using Spartan)
pub fn compress_ivc(proof: &IvcProof) -> Result<CompressedIvcProof, CompressError>;
```

---

## Integration with Existing Kernel

### New exports in lean5-kernel

```rust
// Add to lean5-kernel/src/lib.rs

#[cfg(feature = "commitments")]
pub use lean5_commit::{
    KzgProofCommitment,
    IpaProofCommitment,
    ProofCommitmentScheme,
    CommittedProof,
};

#[cfg(feature = "folding")]
pub use lean5_fold::{
    IvcProof,
    start_ivc,
    extend_ivc,
    verify_ivc,
};
```

### CLI Integration

```bash
# Commit a proof
lean5 commit-proof input.cert -o output.committed

# Verify committed proof
lean5 verify-committed output.committed

# Start IVC accumulator
lean5 ivc-start first.cert -o accumulator.ivc

# Extend IVC with additional proofs
lean5 ivc-extend accumulator.ivc second.cert

# Verify IVC proof
lean5 ivc-verify accumulator.ivc
```

---

## Implementation Plan

### Phase 1: lean5-commit foundation (15-20 commits)
1. Create crate skeleton with arkworks dependencies
2. Implement ProofCert → field element encoding
3. Implement KZG commitment using ark-poly-commit
4. Implement IPA commitment (transparent setup)
5. Add batch verification
6. Integration tests with real certificates

### Phase 2: lean5-fold foundation (20-25 commits)
1. Create crate skeleton
2. Implement R1CS builder for constraint generation
3. Implement relaxed R1CS structure
4. Implement folding operation
5. Implement Fiat-Shamir transcript
6. Implement IVC prover/verifier
7. Integration tests

### Phase 3: Integration and optimization (10-15 commits)
1. CLI commands
2. Feature flags in lean5-kernel
3. Performance benchmarks
4. Documentation

---

## Security Considerations

1. **KZG trusted setup**: Powers-of-tau ceremony required. Consider using existing ceremony results (Zcash, Filecoin).

2. **IPA transparency**: No trusted setup but larger proofs and slower verification.

3. **Fiat-Shamir security**: Use transcript abstraction with domain separation.

4. **Commitment binding**: Polynomial commitments are computationally binding under hardness assumptions.

---

## Open Questions

1. **Encoding efficiency**: What's the optimal polynomial encoding for ProofCert trees?

2. **Constraint complexity**: How many R1CS constraints per certificate node?

3. **zksolve integration**: Should we reuse primitives from dropbox/zksolve?

4. **Proof composition**: How to handle proofs that depend on other proofs?

---

## References

1. arkworks poly-commit: https://github.com/arkworks-rs/poly-commit
2. Nova paper: https://eprint.iacr.org/2021/370
3. Microsoft Nova implementation: https://github.com/microsoft/Nova
4. KZG original paper: Kate, Zaverucha, Goldberg 2010
