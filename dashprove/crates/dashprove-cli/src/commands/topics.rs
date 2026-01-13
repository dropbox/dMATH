//! Topics help command implementation
//!
//! Provides detailed explanations of DashProve concepts.

/// Run the topics help command
pub fn run_topics(topic: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    match topic {
        None => print_topics_overview(),
        Some("usl") => print_usl_topic(),
        Some("backends") => print_backends_topic(),
        Some("counterexamples") => print_counterexamples_topic(),
        Some("learning") => print_learning_topic(),
        Some("properties") => print_properties_topic(),
        Some(unknown) => {
            eprintln!("Unknown topic: {}", unknown);
            eprintln!();
            print_topics_overview();
            return Err(format!("Unknown topic: {}. See available topics above.", unknown).into());
        }
    }
    Ok(())
}

fn print_topics_overview() {
    println!(
        r#"DashProve Help Topics
=====================

Available topics:

  usl             Unified Specification Language syntax and semantics
  backends        Available verification backends and their capabilities
  counterexamples Understanding and analyzing counterexamples
  learning        The proof learning system and corpus
  properties      Types of properties (theorems, invariants, contracts, etc.)

Use 'dashprove topics <topic>' to learn more about a specific topic.

Quick Start
-----------
1. Write a specification in USL format (see 'dashprove topics usl')
2. Run 'dashprove verify spec.usl' to verify your specification
3. If verification fails, use 'dashprove explain counterexample.json' to understand why
4. Use 'dashprove verify --learn' to build a corpus for future suggestions
"#
    );
}

fn print_usl_topic() {
    println!(
        r#"Unified Specification Language (USL)
=====================================

USL is DashProve's specification language that compiles to multiple verification
backends (LEAN 4, TLA+, Kani, Alloy).

Basic Syntax
------------
A USL specification contains properties to verify:

  // Theorem: a named logical property
  theorem my_theorem {{
    forall x: Int. x + 0 = x
  }}

  // Invariant: a property that must hold in all states
  invariant balance_non_negative {{
    balance >= 0
  }}

  // Contract: pre/post conditions for functions
  contract MyModule::transfer {{
    requires amount > 0
    requires balance >= amount
    ensures balance' = balance - amount
  }}

Types
-----
  Int           - Integers
  Bool          - Booleans (true, false)
  String        - Text strings
  Set<T>        - Sets of type T
  Seq<T>        - Sequences (lists) of type T
  Map<K, V>     - Maps from K to V

Expressions
-----------
  forall x: T. P    - Universal quantification
  exists x: T. P    - Existential quantification
  P && Q            - Logical AND
  P || Q            - Logical OR
  !P                - Logical NOT
  P => Q            - Implication
  P <=> Q           - Bi-implication (iff)
  x' = e            - Next-state value (temporal)

Temporal Operators (for TLA+ backend)
--------------------------------------
  []P               - Always P (globally)
  <>P               - Eventually P
  P ~> Q            - P leads to Q

File Extension
--------------
USL files use the .usl extension.

Example File
------------
  // counter.usl
  state {{
    counter: Int
  }}

  init {{
    counter = 0
  }}

  action increment {{
    counter' = counter + 1
  }}

  invariant counter_positive {{
    counter >= 0
  }}

  theorem always_positive {{
    []counter >= 0
  }}
"#
    );
}

fn print_backends_topic() {
    println!(
        r#"Verification Backends
=====================

DashProve supports multiple formal verification backends. Each backend has
different strengths and is suited to different types of properties.

Available Backends
------------------

LEAN 4 (lean)
  Type: Interactive theorem prover
  Best for: Mathematical theorems, type-theoretic proofs
  Strengths: Very expressive, proof automation with tactics
  Command: dashprove verify spec.usl --backends lean

TLA+ (tla+)
  Type: Model checker
  Best for: Distributed systems, concurrent algorithms, state machines
  Strengths: Exhaustive state space exploration, temporal properties
  Requirements: TLC model checker installed
  Command: dashprove verify spec.usl --backends tla+

Kani (kani)
  Type: Rust verification tool
  Best for: Rust code verification, memory safety, panic freedom
  Strengths: Verifies actual Rust code, integrates with cargo
  Requirements: Kani installed via cargo
  Command: dashprove verify spec.usl --backends kani

Alloy (alloy)
  Type: Relational model finder
  Best for: Data models, constraints, finding counterexamples
  Strengths: Fast counterexample generation, visual output
  Requirements: Alloy analyzer installed
  Command: dashprove verify spec.usl --backends alloy

Checking Backend Availability
-----------------------------
Run 'dashprove backends' to see which backends are available on your system.

Using Multiple Backends
-----------------------
You can verify against multiple backends for higher confidence:

  dashprove verify spec.usl --backends lean,tla+

The results from all backends are merged, and confidence increases when
multiple backends agree.

Skipping Health Checks
----------------------
Use --skip-health-check for faster startup if you know backends are available:

  dashprove verify spec.usl --skip-health-check
"#
    );
}

fn print_counterexamples_topic() {
    println!(
        r#"Understanding Counterexamples
=============================

When verification fails (property is DISPROVEN), the backend produces a
counterexample showing how the property can be violated.

What is a Counterexample?
-------------------------
A counterexample is a concrete execution trace or assignment that demonstrates
a property violation. It includes:

  - Variable bindings: Specific values that cause the violation
  - Execution trace: Sequence of states leading to the violation
  - Violated property: Which property failed and why

Analyzing Counterexamples
-------------------------

1. Explain the counterexample:
   dashprove explain counterexample.json

2. Visualize the trace:
   dashprove visualize counterexample.json --format html -o trace.html

3. Analyze patterns:
   dashprove analyze counterexample.json suggest

4. Compress repeating states:
   dashprove analyze counterexample.json compress

5. Find actor interleavings (concurrency):
   dashprove analyze counterexample.json interleavings

6. Minimize the trace:
   dashprove analyze counterexample.json minimize

Comparing Counterexamples
-------------------------
Compare two counterexamples to understand differences:
  dashprove analyze cx1.json diff cx2.json

Clustering Counterexamples
--------------------------
When you have multiple counterexamples, cluster them to find patterns:
  dashprove cluster cx1.json cx2.json cx3.json

This helps identify systematic issues vs. one-off bugs.

Counterexample Formats
----------------------
- JSON: Structured format with metadata (recommended)
- Plain text: Raw backend output (requires --backend flag to explain)
"#
    );
}

fn print_learning_topic() {
    println!(
        r#"Proof Learning System
=====================

DashProve includes a learning system that improves over time by recording
verification results and using them to make better suggestions.

How It Works
------------
1. Verify with --learn: Results are recorded to the proof corpus
2. Future verifications get suggestions based on similar past proofs
3. The corpus grows and suggestions improve over time

Recording Results
-----------------
Add --learn to record verification results:
  dashprove verify spec.usl --learn

Results are stored in ~/.dashprove by default. Use --data-dir to customize:
  dashprove verify spec.usl --learn --data-dir /path/to/corpus

Getting Suggestions
-------------------
Use --suggest to get tactic suggestions before verification:
  dashprove verify spec.usl --suggest

Suggestions come from:
  - Compiler analysis of your specification
  - Similar proofs in the corpus
  - Previously successful tactics

Corpus Operations
-----------------

View corpus statistics:
  dashprove corpus stats

Search for similar proofs:
  dashprove corpus search my_spec.usl

View corpus history:
  dashprove corpus history

Compare time periods:
  dashprove corpus compare --baseline-from 2024-01-01 --baseline-to 2024-01-31 \
    --compare-from 2024-02-01 --compare-to 2024-02-28

Text Search
-----------
Search the corpus by concept:
  dashprove search "mutex safety"
  dashprove search "termination proof"

Counterexample Corpus
---------------------
Counterexamples are also recorded and can be searched:
  dashprove corpus cx-search counterexample.json
  dashprove corpus cx-add counterexample.json --property my_invariant
"#
    );
}

fn print_properties_topic() {
    println!(
        r#"Property Types
==============

DashProve supports several types of formal properties, each suited to
different verification goals.

Theorem
-------
A named logical property to prove.

  theorem addition_commutative {{
    forall x: Int. forall y: Int. x + y = y + x
  }}

Best for: Mathematical properties, logical relationships

Invariant
---------
A property that must hold in every reachable state.

  invariant balance_non_negative {{
    balance >= 0
  }}

Best for: Safety properties, data integrity, state constraints

Contract
--------
Pre/post conditions for functions or methods.

  contract Account::withdraw {{
    requires amount > 0
    requires balance >= amount
    ensures balance' = balance - amount
    ensures result = Ok(())
  }}

Best for: Function specifications, API contracts

Temporal
--------
Properties about sequences of states over time.

  temporal eventually_terminates {{
    <>(state = Terminated)
  }}

  temporal mutex_safety {{
    [](!(thread1_in_critical && thread2_in_critical))
  }}

Best for: Liveness, progress, concurrent systems (TLA+ backend)

Refinement
----------
One specification refines (correctly implements) another.

  refinement impl_refines_spec {{
    Implementation refines Specification
  }}

Best for: Proving implementations match specifications

Probabilistic
-------------
Properties with probability bounds.

  probabilistic high_availability {{
    Pr[available] >= 0.999
  }}

Best for: Reliability, availability, randomized algorithms

Security
--------
Security-specific properties.

  security no_information_leak {{
    noninterference(public_inputs, secret_data)
  }}

Best for: Information flow, confidentiality, integrity
"#
    );
}
