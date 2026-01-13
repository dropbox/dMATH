# tla-wire: Implementation Wiring Verification

## Overview

`tla-wire` verifies that application components are properly connected. It addresses a common problem with AI-generated code: components exist but are not wired together.

The tool operates at two levels:
1. **Automated checks** - Universal structural analysis requiring no annotations
2. **Hot spot analysis** - User-annotated states with path reachability proofs

## Design Principles

1. **No magic extraction** - We don't attempt to extract full state machines from arbitrary code
2. **User declares intent** - Annotations mark states the user cares about
3. **Tractable problems** - Check specific paths, not enumerate all states
4. **Actionable output** - Report what's broken and where
5. **Incremental adoption** - Automated checks work with zero annotations

---

## Part 1: Automated Easy Wins

These checks require **no annotations** and catch common wiring failures.

### 1.1 Entry Point to Effect

Every application should do *something* observable.

```
Entry Point → ... → Effect (stdout, network, file, UI)
```

**Check**: Does `main()` (or equivalent) have a path to any effect?

**Catches**:
```rust
fn main() {
    let app = App::new();  // Created but never run
    // No println!, no network, no file write, nothing
}
```

**Output**:
```
[CRITICAL] main() produces no observable effects
  Entry point at src/main.rs:1 has no path to any effect
  The application will appear to do nothing when run
```

### 1.2 Dead Code Detection

Functions defined but unreachable from any entry point.

**Check**: For each function, is there a path from any entry point?

**Catches**:
```rust
fn main() { foo(); }
fn foo() { println!("used"); }
fn bar() { println!("never called"); }  // Dead code
```

**Output**:
```
[WARNING] 3 functions (127 lines) unreachable from entry points
  src/handlers.rs:45  handle_chat()      - 34 lines
  src/handlers.rs:89  handle_complete()  - 56 lines
  src/utils.rs:12     format_response()  - 37 lines
```

### 1.3 Framework Pattern Checks

Detect common framework wiring mistakes without annotations.

#### Clap (CLI)
| Pattern | Check |
|---------|-------|
| `#[derive(Parser)]` exists | `.parse()` must be called |
| `#[derive(Subcommand)]` exists | `match` on command must exist |
| Command handler functions | Must be reachable from match arms |

#### Tokio (Async)
| Pattern | Check |
|---------|-------|
| `async fn` exists | `#[tokio::main]` or `Runtime::block_on` must exist |
| `tokio::spawn()` called | Must be inside async context |
| Async function called | Must be `.await`ed or spawned |

#### Axum/Web (HTTP)
| Pattern | Check |
|---------|-------|
| `Router::new()` exists | Must be passed to `serve()` |
| Handler functions defined | Must be registered with `.route()` |
| `TcpListener::bind()` called | Must be `.await`ed |

### 1.4 Structural Issues

| Check | What it catches |
|-------|-----------------|
| Unhandled Results | `fallible_fn();` without `?` or match |
| Missing await | `async_fn();` without `.await` |
| Unused return values | `compute_value();` result discarded |
| Dangling references | Call to undefined function |

---

## Part 2: Annotated Hot Spot Analysis

For deeper analysis, users annotate states they care about.

### 2.1 Annotations

```rust
/// Mark the application entry point
#[wire::start]
fn main() { }

/// Mark a state/feature the user cares about
#[wire::state("login")]
fn show_login() { }

#[wire::state("dashboard")]
fn show_dashboard() { }

#[wire::state("checkout")]
fn begin_checkout() { }

/// Mark that one state must be able to reach another
#[wire::must_reach("payment_success")]
#[wire::state("checkout")]
fn begin_checkout() { }

/// Mark that this state requires a condition
#[wire::requires("authenticated")]
#[wire::state("dashboard")]
fn show_dashboard() { }

/// Mark that error states must have recovery paths
#[wire::recoverable]
#[wire::state("payment_error")]
fn show_payment_error() { }
```

### 2.2 Path Analysis

Given annotated states, prove reachability between them.

**Input**: Set of annotated states S = {start, login, dashboard, checkout, ...}

**Output**: For each pair (A, B) in S × S:
- Does a path exist from A to B?
- If yes, what is the path?
- What conditions must hold for the path?
- What issues exist along the path?

### 2.3 Output Format

```
$ tla wire ./my-app

══════════════════════════════════════════════════════════════
ANNOTATED STATES
══════════════════════════════════════════════════════════════

  [start]     main                 src/main.rs:1
  [login]     show_login           src/auth.rs:45
  [dashboard] show_dashboard       src/views.rs:23
  [checkout]  begin_checkout       src/shop.rs:12
  [payment]   payment_success      src/shop.rs:89

══════════════════════════════════════════════════════════════
PATH ANALYSIS
══════════════════════════════════════════════════════════════

[start] → [login]
  ✓ REACHABLE
  Path: main → App::new → router.add("/login", show_login)

[start] → [dashboard]
  ✓ REACHABLE (conditional)
  Path: main → App::new → check_session → show_dashboard
  Requires: session.is_authenticated() = true

[login] → [dashboard]
  ✓ REACHABLE
  Path: show_login → handle_submit → authenticate → redirect("/dashboard")
        → router.dispatch → show_dashboard

[dashboard] → [checkout]
  ✗ UNREACHABLE

  Analysis:
    show_dashboard calls: render_widgets, fetch_data, update_ui
    None of these have paths to begin_checkout

  Nearest connection found:
    show_dashboard → cart_widget → [DEAD END]
    begin_checkout exists but is orphaned

  Suggestion:
    cart_widget should call begin_checkout on "checkout" button click

[checkout] → [payment]
  ⚠ REACHABLE (with issues)
  Path: begin_checkout → validate → process_payment → payment_success

  Issues on path:
    [ERROR] src/shop.rs:67 - process_payment returns Result<_, PaymentError>
            but error case not handled; PaymentError leads to dead end
    [WARN]  src/shop.rs:45 - validate() can return early on invalid cart
            but no path back to checkout from validation error

══════════════════════════════════════════════════════════════
REACHABILITY MATRIX
══════════════════════════════════════════════════════════════

From \ To       login  dashboard  checkout  payment
─────────────────────────────────────────────────────
start             ✓       ✓(c)       ✗         ✗
login             -         ✓        ✗         ✗
dashboard         ✓         -        ✗         ✗
checkout          ✗         ✗        -         ⚠

Legend: ✓ = reachable, ✓(c) = conditional, ⚠ = issues, ✗ = unreachable

══════════════════════════════════════════════════════════════
SUMMARY
══════════════════════════════════════════════════════════════

States: 5 annotated
Paths checked: 12
  Reachable: 4
  Conditional: 1
  Issues: 1
  Unreachable: 6

CRITICAL: [checkout] and [payment] not reachable from [start]
          Core features are not wired to the application entry point
```

### 2.4 Path Anomaly Detection: What's "Suspicious"?

When a path exists between two annotated states, we analyze what's *on* that path and flag anything unusual. This is where tla-wire provides real value beyond simple reachability.

#### 2.4.1 Error Handling Anomalies

**Unhandled Error Branches**

```rust
#[wire::state("checkout")]
fn checkout() {
    let result = validate_cart();  // Returns Result<Cart, ValidationError>
    process(result.unwrap());      // SUSPICIOUS: .unwrap() hides error path
}
```

Warning: `validate_cart()` can fail but error path leads nowhere. What happens when validation fails?

**Error Paths to Dead Ends**

```rust
fn process_payment() -> Result<Receipt, PaymentError> {
    let response = gateway.charge()?;  // ? propagates error
    Ok(response.receipt)
}

#[wire::state("checkout")]
fn checkout() {
    match process_payment() {
        Ok(receipt) => show_success(receipt),
        Err(e) => log::error!("{}", e),  // SUSPICIOUS: logs error but then what?
                                          // No recovery, no retry, no user feedback
    }
}
```

Warning: Error path exists but terminates without reaching any annotated state. User stuck after payment failure.

**Swallowed Errors**

```rust
fn load_config() -> Config {
    fs::read_to_string("config.toml")
        .ok()                           // SUSPICIOUS: converts error to None
        .and_then(|s| toml::parse(&s).ok())  // SUSPICIOUS: silently drops parse errors
        .unwrap_or_default()            // Falls back silently
}
```

Warning: Multiple error conditions silently swallowed. Failures will be invisible.

#### 2.4.2 Async/Concurrency Anomalies

**Missing Await**

```rust
#[wire::state("send_email")]
async fn send_email(to: &str) {
    let future = client.send(to);  // SUSPICIOUS: Future created but not awaited
    log::info!("Email sent");      // This runs before email actually sends!
}
```

Error: Async operation not awaited. `send_email` completes before email is sent.

**Fire and Forget Without Spawn**

```rust
async fn background_job() { /* ... */ }

fn handle_request() {
    background_job();  // SUSPICIOUS: async fn called without .await or spawn
                       // This does NOTHING - the future is dropped immediately
}
```

Error: Async function called synchronously. The operation never executes.

**Spawn Without Error Handling**

```rust
fn start_worker() {
    tokio::spawn(async {
        do_work().await?;  // SUSPICIOUS: ? in spawned task
                           // Error goes nowhere - task just dies silently
    });
}
```

Warning: Spawned task can fail but errors are not observable. Failures will be silent.

#### 2.4.3 Control Flow Anomalies

**Conditional-Only Paths**

```rust
#[wire::state("admin_panel")]
fn show_admin() { /* ... */ }

fn route(user: &User) {
    if user.is_admin {
        show_admin();  // Only reachable if is_admin
    }
}
```

Warning (Info): Path to `admin_panel` requires `user.is_admin == true`. Is this intentional access control or missing feature?

**Early Returns That Skip States**

```rust
#[wire::state("checkout")]
fn checkout(cart: &Cart) -> Result<()> {
    if cart.is_empty() {
        return Ok(());  // SUSPICIOUS: returns early without reaching payment
    }
    validate(cart)?;
    process_payment(cart)?;
    Ok(())
}
```

Warning: Early return skips path to payment. Empty cart silently succeeds without user feedback.

**Unreachable After Condition**

```rust
fn process(status: Status) {
    match status {
        Status::Success => handle_success(),
        Status::Pending => handle_pending(),
        // No Status::Failed arm!
    }
    // SUSPICIOUS: if Status::Failed exists, this is incomplete match
}
```

Error: Match is non-exhaustive. Some status values have no handler.

#### 2.4.4 Resource/State Anomalies

**Resource Acquired But Not Released**

```rust
fn process_file() {
    let file = File::open("data.txt")?;
    let lock = file.lock()?;  // Lock acquired
    process(&file)?;
    // SUSPICIOUS: lock not explicitly released before error return
    // (Rust's Drop helps, but pattern may indicate logic error)
}
```

Info: Lock acquired but function has early return paths. Verify lock semantics.

**State Modified Without Persistence**

```rust
fn update_user(user: &mut User) {
    user.name = "New Name".to_string();
    user.email = "new@example.com".to_string();
    // SUSPICIOUS: state modified but no save/persist call
    // Did we forget to call user.save()?
}
```

Warning: Object modified but no persistence operation on path. Changes may be lost.

**Partial State Updates**

```rust
fn transfer_funds(from: &mut Account, to: &mut Account, amount: u64) -> Result<()> {
    from.balance -= amount;  // Debit happens
    validate_transfer()?;    // SUSPICIOUS: validation AFTER debit
    to.balance += amount;    // Credit happens
    Ok(())
}
```

Warning: State modification before validation. If validation fails, `from` is debited but `to` is not credited.

#### 2.4.5 Path Structure Anomalies

**Unusually Long Paths**

```
[start] → [checkout]: 15 hops
  main → init → setup → config → load → parse → validate →
  transform → prepare → route → match → dispatch → handle →
  verify → checkout
```

Info: Path has 15 intermediate steps. This may indicate:
- Over-abstraction
- Missing direct route
- Accidental complexity

**Circular Paths Without Exit**

```
[error] → [retry] → [error] → [retry] → ...
```

Warning: Circular path detected with no exit condition visible. Potential infinite loop on repeated failures.

**Multiple Divergent Paths**

```
[start] → [payment] has 4 distinct paths:
  Path 1: start → quick_buy → payment (3 hops)
  Path 2: start → cart → checkout → review → payment (5 hops)
  Path 3: start → wishlist → cart → checkout → review → payment (6 hops)
  Path 4: start → browse → product → cart → checkout → review → payment (7 hops)
```

Info: Multiple paths to same state. Verify all paths set up required preconditions.

#### 2.4.6 Summary: Anomaly Categories

| Category | What We Check | Why It Matters |
|----------|---------------|----------------|
| **Error Handling** | Unhandled Results, swallowed errors, dead-end error paths | Users get stuck, failures are invisible |
| **Async** | Missing await, fire-and-forget, spawn without error handling | Operations don't execute or fail silently |
| **Control Flow** | Early returns, incomplete matches, conditional-only access | Features unreachable or inconsistent |
| **Resources** | Locks, transactions, partial updates | Data corruption, deadlocks |
| **Structure** | Long paths, cycles, divergent routes | Complexity, maintenance burden |

#### 2.4.7 Severity Levels

| Severity | Meaning | Examples |
|----------|---------|----------|
| **Error** | Path exists but is definitely broken | Missing await, unhandled error leads to dead end |
| **Warning** | Path exists but likely has problems | Swallowed errors, conditional-only access |
| **Info** | Path exists, noteworthy observation | Long path, multiple routes, resource patterns |

### 2.5 Must-Reach Constraints

```rust
#[wire::must_reach("payment_success", "payment_error")]
#[wire::state("checkout")]
fn begin_checkout() { }
```

**Check**: From `checkout`, both `payment_success` AND `payment_error` must be reachable.

**Output if violated**:
```
[ERROR] must_reach constraint violated
  State [checkout] must reach [payment_error] but no path exists

  Payment errors will have nowhere to go.
  Users will be stuck if payment fails.
```

### 2.6 Recoverable Constraint

```rust
#[wire::recoverable]
#[wire::state("error")]
fn show_error() { }
```

**Check**: From `error`, there must be a path to some non-error state.

**Output if violated**:
```
[ERROR] State [error] marked recoverable but has no exit path
  show_error() at src/ui.rs:123 has no outgoing transitions
  Users who reach this state will be stuck
```

---

## Part 3: Implementation Architecture

### 3.1 Core Types

```rust
/// User-annotated state
pub struct AnnotatedState {
    pub name: String,
    pub kind: StateKind,
    pub node_id: NodeId,
    pub location: Location,
    pub constraints: Vec<Constraint>,
}

pub enum StateKind {
    Start,
    State,
}

pub enum Constraint {
    MustReach(Vec<String>),
    Requires(String),
    Recoverable,
}

/// Result of analyzing a path between two states
pub struct PathAnalysis {
    pub from: String,
    pub to: String,
    pub status: PathStatus,
    pub path: Option<Vec<PathSegment>>,
    pub conditions: Vec<PathCondition>,
    pub issues: Vec<PathIssue>,
}

pub enum PathStatus {
    Reachable,
    Conditional,
    ReachableWithIssues,
    Unreachable,
}

pub struct PathSegment {
    pub node_id: NodeId,
    pub name: String,
    pub location: Location,
    pub edge_kind: EdgeKind,  // Call, Await, Spawn, Route, etc.
}

pub struct PathCondition {
    pub condition: String,
    pub location: Location,
}

pub enum PathIssue {
    UnhandledError {
        location: Location,
        error_type: String,
        leads_to: Option<NodeId>,  // Where does error path go?
    },
    MissingAwait {
        location: Location,
        async_fn: String,
    },
    DeadEnd {
        location: Location,
        state: String,
    },
    NonRecoverableError {
        location: Location,
        error_state: String,
    },
    ConditionalAccess {
        location: Location,
        condition: String,
    },
    UnusedReturn {
        location: Location,
        return_type: String,
    },
}
```

### 3.2 Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. EXTRACTION                                               │
│    ├─ Parse source files (tree-sitter)                     │
│    ├─ Extract functions, calls, effects                    │
│    ├─ Extract annotations (#[wire::*])                     │
│    └─ Build WiringGraph                                    │
├─────────────────────────────────────────────────────────────┤
│ 2. AUTOMATED CHECKS (no annotations needed)                │
│    ├─ Entry → Effect reachability                          │
│    ├─ Dead code detection                                  │
│    ├─ Framework pattern validation                         │
│    └─ Structural issue detection                           │
├─────────────────────────────────────────────────────────────┤
│ 3. PATH ANALYSIS (requires annotations)                    │
│    ├─ For each pair of annotated states:                   │
│    │   ├─ BFS/DFS to find path                            │
│    │   ├─ Extract conditions along path                    │
│    │   └─ Detect issues along path                        │
│    ├─ Build reachability matrix                           │
│    └─ Check constraints (must_reach, recoverable)         │
├─────────────────────────────────────────────────────────────┤
│ 4. REPORTING                                               │
│    ├─ Format terminal output                               │
│    ├─ Generate JSON/SARIF for CI                          │
│    └─ Exit code based on severity                         │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Module Structure

```
crates/tla-wire/
├── src/
│   ├── lib.rs              # Public API
│   ├── graph.rs            # WiringGraph, Node, Edge types
│   ├── extract/
│   │   ├── mod.rs          # LanguageAdapter trait
│   │   ├── rust.rs         # Rust extraction
│   │   ├── annotations.rs  # #[wire::*] parsing
│   │   └── ...
│   ├── analyze/
│   │   ├── mod.rs          # Main analyzer
│   │   ├── automated.rs    # Automated checks (Part 1)
│   │   ├── paths.rs        # Path analysis (Part 2)
│   │   └── issues.rs       # Issue detection along paths
│   ├── patterns/
│   │   ├── mod.rs          # FrameworkPattern trait
│   │   ├── clap.rs
│   │   ├── tokio.rs
│   │   └── axum.rs
│   └── report/
│       ├── mod.rs          # WiringReport types
│       ├── terminal.rs     # Terminal formatting
│       ├── json.rs         # JSON output
│       └── sarif.rs        # SARIF for CI
├── wire-macros/            # Proc macros for #[wire::*]
│   ├── src/
│   │   └── lib.rs
│   └── Cargo.toml
└── DESIGN.md               # This file
```

### 3.4 CLI Integration

```bash
# Basic check (automated only)
tla wire ./my-project

# With annotations
tla wire ./my-project --check-paths

# Output formats
tla wire ./my-project --format json
tla wire ./my-project --format sarif

# CI mode (exit code reflects issues)
tla wire ./my-project --ci --min-score 80

# Verbose path output
tla wire ./my-project --show-paths --show-conditions

# Check specific states
tla wire ./my-project --from start --to checkout
```

---

## Part 4: Relationship to TLA+

### 4.1 Wiring as Temporal Properties

Annotated state analysis maps to TLA+ properties:

| Annotation | TLA+ Property |
|------------|---------------|
| `#[wire::start]` → `#[wire::state("X")]` reachable | `<>(state = X)` |
| `#[wire::must_reach("Y")]` on state X | `(state = X) ~> (state = Y)` |
| `#[wire::recoverable]` on error state | `(state = Error) ~> (state /= Error)` |

### 4.2 Integration with tla-check

Future: Generate TLA+ specs from annotations, use tla-check for verification.

```rust
// Annotations in code
#[wire::start]
fn main() { }

#[wire::state("checkout")]
fn checkout() { }

#[wire::state("payment")]
fn payment() { }
```

Generated TLA+ spec:
```tla
---- MODULE AppWiring ----
VARIABLE state

States == {"start", "checkout", "payment"}

Init == state = "start"

\* Extracted from code analysis
Next ==
    \/ state = "start" /\ state' = "checkout"
    \/ state = "checkout" /\ state' = "payment"
    \/ ...

\* From annotations
StartReachesCheckout == <>(state = "checkout")
CheckoutReachesPayment == (state = "checkout") ~> (state = "payment")
====
```

---

## Part 5: Example Walkthrough

### Input: Broken CLI App

```rust
// src/main.rs
#[wire::start]
fn main() {
    let args = Args::parse();  // Good: clap called

    match args.command {
        Command::Chat(chat_args) => {
            // Oops: forgot to call handle_chat!
        }
        Command::Config(config_args) => {
            handle_config(config_args);
        }
    }
}

// src/chat.rs
#[wire::state("chat")]
async fn handle_chat(args: ChatArgs) -> Result<()> {
    let response = call_api(&args.message).await?;
    render_response(response);  // Missing .await? No, render is sync
    Ok(())
}

#[wire::state("api_call")]
async fn call_api(message: &str) -> Result<Response> {
    client.post(API_URL).send().await?.json().await
}

#[wire::state("output")]
fn render_response(response: Response) {
    println!("{}", response.text);
}

// src/config.rs
#[wire::state("config")]
fn handle_config(args: ConfigArgs) -> Result<()> {
    // ... config handling
    Ok(())
}
```

### Output

```
$ tla wire ./my-cli

══════════════════════════════════════════════════════════════
AUTOMATED CHECKS
══════════════════════════════════════════════════════════════

[PASS] Entry point produces effects
  main → handle_config → ... → println!

[WARN] Dead code detected
  src/chat.rs:12  handle_chat     - 15 lines
  src/chat.rs:28  call_api        - 8 lines
  src/chat.rs:35  render_response - 5 lines

  These functions are not reachable from main()

[WARN] Async function handle_chat defined but:
  - No #[tokio::main] on main()
  - handle_chat is never called anyway

══════════════════════════════════════════════════════════════
ANNOTATED STATES
══════════════════════════════════════════════════════════════

  [start]    main              src/main.rs:1
  [chat]     handle_chat       src/chat.rs:12
  [api_call] call_api          src/chat.rs:28
  [output]   render_response   src/chat.rs:35
  [config]   handle_config     src/config.rs:8

══════════════════════════════════════════════════════════════
PATH ANALYSIS
══════════════════════════════════════════════════════════════

[start] → [chat]
  ✗ UNREACHABLE

  main() matches on Command::Chat but the match arm is empty.
  handle_chat() is never called.

  Location: src/main.rs:8
    Command::Chat(chat_args) => {
        // Empty! Should call handle_chat(chat_args)
    }

[start] → [config]
  ✓ REACHABLE
  Path: main → match Command::Config → handle_config

[chat] → [api_call]
  ✓ REACHABLE (if chat were reachable)
  Path: handle_chat → call_api

[chat] → [output]
  ✓ REACHABLE (if chat were reachable)
  Path: handle_chat → call_api → render_response

[start] → [api_call]
  ✗ UNREACHABLE (transitive: chat unreachable)

[start] → [output]
  ✗ UNREACHABLE (transitive: chat unreachable)

══════════════════════════════════════════════════════════════
REACHABILITY MATRIX
══════════════════════════════════════════════════════════════

From \ To     chat  api_call  output  config
────────────────────────────────────────────
start          ✗       ✗        ✗       ✓
chat           -       ✓        ✓       ✗
api_call       ✗       -        ✓       ✗
output         ✗       ✗        -       ✗
config         ✗       ✗        ✗       -

══════════════════════════════════════════════════════════════
SUMMARY
══════════════════════════════════════════════════════════════

Automated Checks:
  Passed: 1
  Warnings: 2
  Errors: 0

Path Analysis:
  Reachable from start: 1/4 states (config only)

CRITICAL: Core feature [chat] is not wired
  The chat functionality exists but is never called.
  Fix: Add handle_chat(chat_args) call in main.rs:8
```

---

## Part 6: Implementation Roadmap

### Phase 1: Foundation (Current)
- [x] WiringGraph, Node, Edge types
- [x] Rust extraction skeleton
- [x] Framework pattern stubs
- [ ] Complete Rust call extraction
- [ ] Annotation parsing (#[wire::*])

### Phase 2: Automated Checks
- [ ] Entry → Effect analysis
- [ ] Dead code detection
- [ ] Complete framework patterns (clap, tokio, axum)
- [ ] Structural issue detection (unhandled Result, missing await)

### Phase 3: Path Analysis
- [ ] BFS path finding between annotated states
- [ ] Condition extraction along paths
- [ ] Issue detection along paths
- [ ] Reachability matrix generation

### Phase 4: Constraints
- [ ] must_reach validation
- [ ] recoverable validation
- [ ] requires condition tracking

### Phase 5: Reporting & CLI
- [ ] Terminal output formatting
- [ ] JSON output
- [ ] SARIF output for CI
- [ ] tla CLI integration (`tla wire` command)

### Phase 6: Polish
- [ ] Error message quality
- [ ] Performance optimization
- [ ] Documentation
- [ ] Test fixtures (intentionally broken apps)

---

## Part 7: Analysis Tiers and Honest Limitations

tla-wire has two analysis tiers. With tRust, we either have syntax-only (standalone) or full compiler access. There's no middle ground.

### Tier 1: Syntax-Only (Standalone Tool)

**Purpose**: Quick checks, non-tRust code, bootstrapping, prototyping

**What we have**: Tree-sitter AST (syntax trees, no semantic info)

**What works**:
| Feature | Status | How |
|---------|--------|-----|
| Find `#[wire::*]` annotations | ✓ | Attribute parsing |
| Direct function calls `foo()` | ✓ | Call expression in AST |
| Entry point detection | ✓ | `fn main()` pattern |
| Effect detection | ✓ | `println!`, `write!`, known macros |
| Dead code (unreachable functions) | ✓ | Graph reachability |
| Framework patterns | ✓ | String matching heuristics |

**What doesn't work**:
| Feature | Status | Why |
|---------|--------|-----|
| "Returns Result" detection | ✗ | Need compiler |
| Trait method resolution | ✗ | Need compiler |
| Cross-file imports | ✗ | Need compiler |
| Closure/callback tracing | ✗ | Need compiler |
| Macro expansion | ✗ | Need compiler |
| Condition extraction | ✗ | Need compiler |

**Output honesty**:
```
[start] → [checkout]
  ✓ REACHABLE (via direct calls)

  ⚠ Analysis limitations on this path:
    - Line 45: trait method call (handler.process()) - cannot resolve impl
    - Line 67: closure passed to on_click() - cannot trace into closure

  Heuristic issues (may be false positives):
    - Line 52: .unwrap() call - potential unhandled error
    - Line 58: call to async-named function without .await
```

### Tier 2: Full Analysis (tRust Compiler Integration)

**What we have**: MIR access, borrow checker info, control flow graph with data dependencies

**What is Data Flow Analysis?**

Data flow tracks how values move through a program and what constraints apply:

```rust
fn checkout(user: User, cart: Cart) -> Result<Receipt, Error> {
    // Data flow tracks:
    // - user comes from parameter
    // - is_verified is derived from user.verified
    // - the if-branch constrains is_verified = true for inner code

    let is_verified = user.verified;        // (1) Definition

    if !is_verified {                       // (2) Branch condition
        return Err(Error::Unverified);      // (3) Early exit: is_verified = false
    }

    // (4) Here we KNOW: is_verified = true (path condition)

    let total = cart.total();               // (5) Definition from cart
    if total > user.credit_limit {          // (6) Branch condition
        return Err(Error::OverLimit);       // (7) Early exit
    }

    // (8) Here we KNOW: is_verified = true AND total <= credit_limit

    let receipt = charge(user, cart)?;      // (9) Can fail, propagates

    // (10) Here we KNOW: all above AND charge succeeded

    Ok(receipt)
}
```

**Data flow tells us**:
- At (4): `is_verified = true` (path condition from branch)
- At (8): `is_verified = true AND total <= credit_limit`
- At (10): All conditions plus `charge()` returned `Ok`
- Error exits: exact conditions for each `Err` return

**Unlocks**:
| Feature | What It Enables |
|---------|-----------------|
| **Path conditions** | "To reach [payment], user.verified must be true" |
| **Error path tracing** | "Err(Unverified) leads to... nowhere (dead end)" |
| **Value provenance** | "receipt comes from charge(), which depends on user and cart" |
| **Partial update detection** | "from.balance modified before validation - bug if validation fails" |
| **Taint tracking** | "user_input flows to sql_query without sanitization" |
| **Resource tracking** | "lock acquired but not released on error path" |

**Data Flow Analyses**:

| Analysis | Question Answered |
|----------|-------------------|
| **Reaching Definitions** | "Where did this value come from?" |
| **Live Variables** | "Is this variable used later?" |
| **Available Expressions** | "Was this already computed?" |
| **Path Conditions** | "What must be true to reach here?" |
| **Def-Use Chains** | "Where is this definition used?" |
| **Taint Propagation** | "Does untrusted data reach this sink?" |

### tRust Integration Architecture

With tRust (our Rust fork), we integrate directly into the compiler:

```
┌─────────────────────────────────────────────────────────────┐
│                    tRust Compiler                           │
├─────────────────────────────────────────────────────────────┤
│  Source                                                     │
│    ↓                                                        │
│  Parsing (AST)                                              │
│    ↓                                                        │
│  HIR (High-level IR) ← #[wire::*] annotations extracted    │
│    ↓                                                        │
│  Type Checking ← Types available                           │
│    ↓                                                        │
│  MIR (Mid-level IR) ← Control flow + data flow available   │
│    ↓                                                        │
│  ┌─────────────────────────────────────────┐               │
│  │         tla-wire Analysis Pass          │               │
│  │  ┌─────────────────────────────────┐    │               │
│  │  │ 1. Extract annotated states     │    │               │
│  │  │ 2. Build wiring graph from MIR  │    │               │
│  │  │ 3. Path analysis with types     │    │               │
│  │  │ 4. Data flow for conditions     │    │               │
│  │  │ 5. Anomaly detection            │    │               │
│  │  └─────────────────────────────────┘    │               │
│  │              ↓                          │               │
│  │    Wiring Report (errors, warnings)     │               │
│  └─────────────────────────────────────────┘               │
│    ↓                                                        │
│  LLVM IR                                                    │
│    ↓                                                        │
│  Binary                                                     │
└─────────────────────────────────────────────────────────────┘
```

**MIR gives us**:
- Control Flow Graph (CFG) with basic blocks
- Explicit drops and moves (borrow checker info)
- Simplified, desugared code (no macros, no syntactic sugar)
- Type information on every expression

**Example MIR-level analysis**:
```
fn checkout(_1: User, _2: Cart) -> Result<Receipt, Error> {
    let _3: bool;                  // is_verified
    let _4: u64;                   // total

    bb0: {
        _3 = (_1.0: bool);         // is_verified = user.verified
        switchInt(_3) -> [false: bb1, otherwise: bb2];
    }

    bb1: {                         // !is_verified branch
        return Err(Unverified);    // Dead end - goes to caller
    }

    bb2: {                         // is_verified = true
        _4 = Cart::total(_2);      // total = cart.total()
        // ... continues
    }
}
```

From this we extract:
- `bb2` has path condition `_3 = true` (i.e., `user.verified = true`)
- `bb1` is an error exit with condition `_3 = false`

### Capability Summary by Tier

| Capability | Tier 1 (Syntax) | Tier 2 (tRust) |
|------------|-----------------|----------------|
| Direct call reachability | ✓ | ✓ |
| Annotation extraction | ✓ | ✓ |
| Dead code detection | ✓ | ✓ |
| Framework patterns | Heuristic | ✓ |
| Result/Option handling | Heuristic | ✓ |
| Trait resolution | ✗ | ✓ |
| Cross-file paths | ✗ | ✓ |
| Async/await correctness | Heuristic | ✓ |
| Macro expansion | ✗ | ✓ |
| Path conditions | ✗ | ✓ |
| Error path destinations | ✗ | ✓ |
| Value provenance | ✗ | ✓ |
| Partial update detection | ✗ | ✓ |
| Resource leak detection | ✗ | ✓ |
| Taint tracking | ✗ | ✓ |

### Implementation Strategy

**Phase A**: Tier 1 (Now)
- Standalone tool using tree-sitter
- Honest about limitations in output
- Useful for catching obvious wiring bugs
- Works on any Rust code (not just tRust)

**Phase B**: Tier 2 (tRust compiler pass)
- Hook into tRust after MIR generation
- Full access to types, traits, control flow, data flow
- Complete analysis with no limitations
- This is the real tool

---

## Summary

`tla-wire` provides wiring verification at two tiers:

1. **Tier 1 - Syntax** (standalone): Direct calls, annotations, heuristics. Works anywhere, honest about limitations.

2. **Tier 2 - tRust** (compiler pass): Full analysis. Types, traits, MIR, data flow, path conditions, everything.

With tRust, there's no middle ground - we either have syntax-only or full compiler access.

The design is:
- **Honest**: Tier 1 explicitly states what it cannot do
- **Complete**: Tier 2 with tRust has no limitations
- **Tractable**: User-annotated hot spots, not full state machine extraction
- **Actionable**: Reports show exactly what's broken and why
- **Ambitious**: Full data flow analysis answers "what conditions lead to this state?"
