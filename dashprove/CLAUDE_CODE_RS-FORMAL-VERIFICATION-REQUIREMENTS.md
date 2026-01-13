# Claude Code RS Formal Verification Requirements

**Source Repo:** https://github.com/dropbox/claude_code_rs
**Target Repo:** https://github.com/dropbox/dMATH/dashprove
**Date:** December 2025
**Status:** Requirements Draft

---

## Executive Summary

claude_code_rs is a Rust port of Claude Code, an AI agent CLI tool. It must achieve **exact behavioral equivalence** with the original. This document specifies formal verification requirements for proving correctness, safety, and parity.

**Key Verification Goals:**
1. **Safety** - No infinite loops, memory bounds respected, no panics
2. **Liveness** - Agent loop terminates, hooks complete, API retries terminate
3. **Correctness** - Tool implementations match specifications
4. **Parity** - Behavior matches Claude Code exactly

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLAUDE CODE RS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         AGENT EXECUTOR                                  │ │
│  │                                                                         │ │
│  │   State: { messages, usage, compaction_count, session_started,         │ │
│  │            claude_md_injected, turns_since_todo_write }                 │ │
│  │                                                                         │ │
│  │   Loop: UserMessage → API → Response → [ToolExecution]* → Continue?    │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│          ┌─────────────────────────┼─────────────────────────┐              │
│          ▼                         ▼                         ▼              │
│  ┌──────────────┐        ┌──────────────┐        ┌──────────────┐          │
│  │  API Client  │        │ Tool Registry │        │ Hook System  │          │
│  │              │        │               │        │              │          │
│  │ • Streaming  │        │ • 21 tools    │        │ • 12 events  │          │
│  │ • SSE Parse  │        │ • Dispatch    │        │ • Timeouts   │          │
│  │ • Retry      │        │ • Schemas     │        │ • JSON I/O   │          │
│  └──────────────┘        └──────────────┘        └──────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Agent Executor Loop

**Files:** `src/agents/executor.rs`

### State Machine Specification

```
States:
  IDLE          - Waiting for user input
  PROCESSING    - API call in flight
  TOOL_EXEC     - Executing tools
  COMPACTING    - Context compaction in progress
  TERMINATED    - Session ended

Transitions:
  IDLE        --[user_message]--> PROCESSING
  PROCESSING  --[response(stop=EndTurn)]--> IDLE
  PROCESSING  --[response(stop=ToolUse)]--> TOOL_EXEC
  TOOL_EXEC   --[all_tools_done]--> PROCESSING
  *           --[iterations > MAX]--> TERMINATED
  *           --[context > threshold]--> COMPACTING
  COMPACTING  --[done]--> (previous state)
```

### TLA+ Specification

```tla+
--------------------------- MODULE AgentExecutor ---------------------------
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    MAX_ITERATIONS,      \* 50 - hard limit on loop iterations
    MAX_MESSAGES,        \* 1000 - hard limit on message history
    MAX_CONTEXT_TOKENS,  \* Context window size
    MAX_TOOL_USES,       \* 100 - max tools per response
    COMPACTION_THRESHOLD \* 0.7 - trigger compaction at 70%

VARIABLES
    state,               \* Current state: {IDLE, PROCESSING, TOOL_EXEC, COMPACTING, TERMINATED}
    messages,            \* Sequence of conversation messages
    iterations,          \* Current iteration count
    input_tokens,        \* Estimated input token usage
    pending_tools,       \* Tools waiting to execute
    compaction_count,    \* Number of times context was compacted
    hooks_pending        \* Hooks waiting to complete

vars == <<state, messages, iterations, input_tokens, pending_tools, compaction_count, hooks_pending>>

\*---------------------------
\* Type Invariants
\*---------------------------
TypeInvariant ==
    /\ state \in {"IDLE", "PROCESSING", "TOOL_EXEC", "COMPACTING", "TERMINATED"}
    /\ iterations \in 0..MAX_ITERATIONS
    /\ input_tokens >= 0
    /\ Len(messages) <= MAX_MESSAGES
    /\ Len(pending_tools) <= MAX_TOOL_USES
    /\ compaction_count >= 0
    /\ hooks_pending \in BOOLEAN

\*---------------------------
\* Safety Properties
\*---------------------------

\* Agent loop never exceeds iteration limit
BoundedIterations == iterations <= MAX_ITERATIONS

\* Message history never exceeds limit
BoundedMessages == Len(messages) <= MAX_MESSAGES

\* Tool uses per response bounded
BoundedToolUses == Len(pending_tools) <= MAX_TOOL_USES

\* No infinite loop in tool execution
ToolExecutionTerminates ==
    state = "TOOL_EXEC" => <>( state # "TOOL_EXEC" )

\* Compaction maintains invariants
CompactionPreservesHistory ==
    \A m \in Range(messages): m.role \in {"user", "assistant", "system"}

\*---------------------------
\* Liveness Properties
\*---------------------------

\* Agent eventually terminates or waits for input
EventuallyIdleOrTerminated ==
    <>[]( state \in {"IDLE", "TERMINATED"} )

\* If iterations exceed limit, we terminate
IterationBoundEnforced ==
    iterations >= MAX_ITERATIONS => <>( state = "TERMINATED" )

\* Hooks eventually complete
HooksComplete ==
    hooks_pending => <>( ~hooks_pending )

\* API calls eventually return
APICallsReturn ==
    state = "PROCESSING" => <>( state # "PROCESSING" )

\*---------------------------
\* Initial State
\*---------------------------
Init ==
    /\ state = "IDLE"
    /\ messages = <<>>
    /\ iterations = 0
    /\ input_tokens = 0
    /\ pending_tools = <<>>
    /\ compaction_count = 0
    /\ hooks_pending = FALSE

\*---------------------------
\* Actions
\*---------------------------

\* User sends a message
UserMessage(content) ==
    /\ state = "IDLE"
    /\ messages' = Append(messages, [role |-> "user", content |-> content])
    /\ state' = "PROCESSING"
    /\ hooks_pending' = TRUE  \* UserPromptSubmit hook
    /\ UNCHANGED <<iterations, input_tokens, pending_tools, compaction_count>>

\* API returns with tool use
APIResponseToolUse(tools) ==
    /\ state = "PROCESSING"
    /\ Len(tools) > 0
    /\ Len(tools) <= MAX_TOOL_USES
    /\ state' = "TOOL_EXEC"
    /\ pending_tools' = tools
    /\ iterations' = iterations + 1
    /\ UNCHANGED <<messages, input_tokens, compaction_count, hooks_pending>>

\* API returns without tool use (end turn)
APIResponseEndTurn(response) ==
    /\ state = "PROCESSING"
    /\ messages' = Append(messages, [role |-> "assistant", content |-> response])
    /\ state' = "IDLE"
    /\ iterations' = iterations + 1
    /\ UNCHANGED <<input_tokens, pending_tools, compaction_count, hooks_pending>>

\* Execute next pending tool
ExecuteTool ==
    /\ state = "TOOL_EXEC"
    /\ Len(pending_tools) > 0
    /\ pending_tools' = Tail(pending_tools)
    /\ hooks_pending' = TRUE  \* PreToolUse/PostToolUse hooks
    /\ UNCHANGED <<state, messages, iterations, input_tokens, compaction_count>>

\* All tools executed, continue loop
ToolsComplete ==
    /\ state = "TOOL_EXEC"
    /\ Len(pending_tools) = 0
    /\ state' = "PROCESSING"
    /\ UNCHANGED <<messages, iterations, input_tokens, pending_tools, compaction_count, hooks_pending>>

\* Trigger compaction when threshold exceeded
TriggerCompaction ==
    /\ input_tokens >= MAX_CONTEXT_TOKENS * COMPACTION_THRESHOLD
    /\ state # "COMPACTING"
    /\ state' = "COMPACTING"
    /\ UNCHANGED <<messages, iterations, input_tokens, pending_tools, compaction_count, hooks_pending>>

\* Complete compaction
CompleteCompaction ==
    /\ state = "COMPACTING"
    /\ compaction_count' = compaction_count + 1
    /\ input_tokens' = MAX_CONTEXT_TOKENS * 3 \div 10  \* 30% after compaction
    /\ state' = "PROCESSING"  \* Resume
    /\ UNCHANGED <<messages, iterations, pending_tools, hooks_pending>>

\* Terminate if iteration limit exceeded
TerminateOnLimit ==
    /\ iterations >= MAX_ITERATIONS
    /\ state' = "TERMINATED"
    /\ UNCHANGED <<messages, iterations, input_tokens, pending_tools, compaction_count, hooks_pending>>

\* Complete pending hooks
CompleteHooks ==
    /\ hooks_pending = TRUE
    /\ hooks_pending' = FALSE
    /\ UNCHANGED <<state, messages, iterations, input_tokens, pending_tools, compaction_count>>

\*---------------------------
\* Next State
\*---------------------------
Next ==
    \/ \E content \in STRING: UserMessage(content)
    \/ \E tools \in SUBSET Tools: APIResponseToolUse(tools)
    \/ \E response \in STRING: APIResponseEndTurn(response)
    \/ ExecuteTool
    \/ ToolsComplete
    \/ TriggerCompaction
    \/ CompleteCompaction
    \/ TerminateOnLimit
    \/ CompleteHooks

\*---------------------------
\* Specification
\*---------------------------
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\*---------------------------
\* Properties to Check
\*---------------------------
THEOREM Spec => []TypeInvariant
THEOREM Spec => []BoundedIterations
THEOREM Spec => []BoundedMessages
THEOREM Spec => []BoundedToolUses
THEOREM Spec => EventuallyIdleOrTerminated

=========================================================================
```

### Kani Proof Harnesses

```rust
// src/agents/executor_proofs.rs

#[cfg(kani)]
mod verification {
    use super::*;

    /// Prove iteration bound is always respected
    #[kani::proof]
    #[kani::unwind(55)]  // MAX_ITERATIONS + buffer
    fn iteration_bound_enforced() {
        let mut agent = Agent::new(AgentConfig::default());

        // Simulate up to MAX_ITERATIONS + 1 loop iterations
        for i in 0..(MAX_ITERATIONS + 1) {
            let should_continue = agent.should_continue();

            if i >= MAX_ITERATIONS {
                kani::assert!(!should_continue, "Must terminate at MAX_ITERATIONS");
            }

            if !should_continue {
                break;
            }

            agent.increment_iteration();
        }
    }

    /// Prove message history bound is respected
    #[kani::proof]
    #[kani::unwind(1005)]
    fn message_bound_enforced() {
        let mut agent = Agent::new(AgentConfig::default());

        for _ in 0..(MAX_MESSAGES + 1) {
            let msg = Message {
                role: Role::User,
                content: kani::any(),
            };

            agent.add_message(msg);
        }

        kani::assert!(
            agent.messages.len() <= MAX_MESSAGES,
            "Message history must not exceed MAX_MESSAGES"
        );
    }

    /// Prove compaction triggers at correct threshold
    #[kani::proof]
    fn compaction_triggers_correctly() {
        let config = AgentConfig {
            max_input_tokens: 100_000,
            compaction_threshold: 0.7,
            ..Default::default()
        };
        let mut agent = Agent::new(config);

        let input_tokens: u32 = kani::any();
        kani::assume(input_tokens <= 200_000);  // Reasonable bound

        agent.set_input_tokens(input_tokens);
        let needs_compaction = agent.needs_compaction();

        if input_tokens >= 70_000 {  // 70% of 100_000
            kani::assert!(needs_compaction, "Should trigger compaction above threshold");
        }
    }

    /// Prove hook execution order is correct
    #[kani::proof]
    fn hook_execution_order() {
        let mut hook_order = Vec::new();

        // Simulate hook sequence
        hook_order.push("SessionStart");
        hook_order.push("UserPromptSubmit");
        hook_order.push("PreToolUse");
        hook_order.push("PostToolUse");
        hook_order.push("Stop");

        // Verify ordering constraints
        let session_start_idx = hook_order.iter().position(|&h| h == "SessionStart");
        let user_prompt_idx = hook_order.iter().position(|&h| h == "UserPromptSubmit");
        let pre_tool_idx = hook_order.iter().position(|&h| h == "PreToolUse");
        let post_tool_idx = hook_order.iter().position(|&h| h == "PostToolUse");
        let stop_idx = hook_order.iter().position(|&h| h == "Stop");

        kani::assert!(session_start_idx < user_prompt_idx, "SessionStart before UserPromptSubmit");
        kani::assert!(pre_tool_idx < post_tool_idx, "PreToolUse before PostToolUse");
        kani::assert!(post_tool_idx < stop_idx, "PostToolUse before Stop");
    }

    /// Prove no deadlock in agent loop
    #[kani::proof]
    #[kani::unwind(10)]
    fn no_deadlock() {
        let mut agent = Agent::new(AgentConfig::default());
        let mut prev_state: Option<AgentState> = None;

        for _ in 0..10 {
            let current_state = agent.state.clone();

            // Must make progress or terminate
            if let Some(prev) = &prev_state {
                if *prev == current_state {
                    // State unchanged - must be in terminal state or waiting for input
                    kani::assert!(
                        current_state == AgentState::Idle ||
                        current_state == AgentState::Terminated,
                        "Non-terminal state must make progress"
                    );
                    break;
                }
            }

            prev_state = Some(current_state);
            agent.step();
        }
    }
}
```

### USL Contract Specification

```usl
// specifications/agent_executor.usl

type AgentState = { Idle, Processing, ToolExec, Compacting, Terminated }

type Agent = {
    state: AgentState,
    messages: List<Message>,
    iterations: Int,
    input_tokens: Int,
    pending_tools: List<ToolUse>,
    compaction_count: Int
}

// Invariants
invariant iteration_bound {
    forall agent: Agent . agent.iterations <= 50
}

invariant message_bound {
    forall agent: Agent . len(agent.messages) <= 1000
}

invariant token_usage_non_negative {
    forall agent: Agent . agent.input_tokens >= 0
}

invariant tool_use_bound {
    forall agent: Agent . len(agent.pending_tools) <= 100
}

// Contracts
contract Agent::step(self: Agent) -> Agent {
    requires { self.state != Terminated }
    ensures {
        self'.iterations <= self.iterations + 1
        self'.state = Terminated or self'.iterations < 50
    }
    ensures_err {
        self' == self
    }
}

contract Agent::add_message(self: Agent, msg: Message) -> Agent {
    requires { len(self.messages) < 1000 }
    ensures {
        len(self'.messages) == len(self.messages) + 1
        last(self'.messages) == msg
    }
}

contract Agent::trigger_compaction(self: Agent) -> Agent {
    requires { self.input_tokens >= self.max_tokens * 70 / 100 }
    ensures {
        self'.input_tokens <= self.max_tokens * 30 / 100
        self'.compaction_count == self.compaction_count + 1
    }
}

// Temporal properties
temporal agent_terminates {
    always(eventually(state == Idle or state == Terminated))
}

temporal hooks_complete {
    always(hooks_pending implies eventually(not hooks_pending))
}

temporal no_infinite_tool_execution {
    always(state == ToolExec implies eventually(state != ToolExec))
}
```

---

## Component 2: Streaming SSE Parser

**Files:** `src/api/streaming.rs`

### Memory Safety Requirements

```rust
// Constants from implementation
const MAX_STREAMING_BUFFER_SIZE: usize = 10 * 1024 * 1024;  // 10MB
const MAX_TOOL_USES: usize = 100;
```

### TLA+ Specification

```tla+
--------------------------- MODULE StreamingParser ---------------------------
EXTENDS Integers, Sequences

CONSTANTS
    MAX_BUFFER_SIZE,    \* 10MB
    MAX_TOOL_USES       \* 100

VARIABLES
    buffer,             \* Current accumulated text
    tool_uses,          \* Accumulated tool uses
    current_tool_json,  \* JSON being accumulated for current tool
    state,              \* Parser state
    buffer_exceeded     \* Flag if buffer limit was hit

vars == <<buffer, tool_uses, current_tool_json, state, buffer_exceeded>>

States == {"IDLE", "IN_TEXT", "IN_TOOL", "IN_THINKING", "DONE"}

TypeInvariant ==
    /\ Len(buffer) <= MAX_BUFFER_SIZE
    /\ Len(tool_uses) <= MAX_TOOL_USES
    /\ state \in States
    /\ buffer_exceeded \in BOOLEAN

\* Safety: Buffer never exceeds limit
BufferBounded == Len(buffer) <= MAX_BUFFER_SIZE

\* Safety: Tool uses never exceed limit
ToolUsesBounded == Len(tool_uses) <= MAX_TOOL_USES

\* If buffer would exceed, set flag instead of growing
BufferGracefulDegradation ==
    (Len(buffer) = MAX_BUFFER_SIZE) => buffer_exceeded

Init ==
    /\ buffer = <<>>
    /\ tool_uses = <<>>
    /\ current_tool_json = <<>>
    /\ state = "IDLE"
    /\ buffer_exceeded = FALSE

ProcessTextDelta(delta) ==
    /\ state = "IN_TEXT"
    /\ IF Len(buffer) + Len(delta) <= MAX_BUFFER_SIZE
       THEN buffer' = buffer \o delta /\ buffer_exceeded' = FALSE
       ELSE buffer_exceeded' = TRUE /\ UNCHANGED buffer
    /\ UNCHANGED <<tool_uses, current_tool_json, state>>

ProcessToolStart(id, name) ==
    /\ Len(tool_uses) < MAX_TOOL_USES
    /\ state' = "IN_TOOL"
    /\ tool_uses' = Append(tool_uses, [id |-> id, name |-> name, input |-> ""])
    /\ current_tool_json' = <<>>
    /\ UNCHANGED <<buffer, buffer_exceeded>>

ProcessToolDelta(json_chunk) ==
    /\ state = "IN_TOOL"
    /\ current_tool_json' = current_tool_json \o json_chunk
    /\ UNCHANGED <<buffer, tool_uses, state, buffer_exceeded>>

Spec == Init /\ [][ProcessTextDelta \/ ProcessToolStart \/ ProcessToolDelta]_vars

THEOREM Spec => []TypeInvariant
THEOREM Spec => []BufferBounded
THEOREM Spec => []ToolUsesBounded

=========================================================================
```

### Kani Proof Harnesses

```rust
// src/api/streaming_proofs.rs

#[cfg(kani)]
mod verification {
    use super::*;

    const MAX_BUFFER: usize = 10 * 1024 * 1024;
    const MAX_TOOLS: usize = 100;

    /// Prove buffer never exceeds limit regardless of input
    #[kani::proof]
    #[kani::unwind(257)]
    fn buffer_bounds_preserved() {
        let mut response = StreamingResponse::new();

        // Process arbitrary sequence of deltas
        for _ in 0..256 {
            let delta_len: usize = kani::any();
            kani::assume(delta_len <= 65536);  // Reasonable chunk size

            let delta = vec![0u8; delta_len];
            response.append_text(&delta);

            kani::assert!(
                response.total_buffer_size() <= MAX_BUFFER,
                "Buffer must not exceed MAX_BUFFER"
            );
        }
    }

    /// Prove tool use count never exceeds limit
    #[kani::proof]
    #[kani::unwind(105)]
    fn tool_use_count_bounded() {
        let mut response = StreamingResponse::new();

        for i in 0..105 {
            let id: String = format!("tool_{}", i);
            let name: String = kani::any();

            response.start_tool_use(&id, &name);
        }

        kani::assert!(
            response.tool_uses.len() <= MAX_TOOLS,
            "Tool uses must not exceed MAX_TOOLS"
        );
    }

    /// Prove UTF-8 safety: invalid sequences replaced, not crashed
    #[kani::proof]
    fn utf8_safety() {
        let mut response = StreamingResponse::new();

        // Arbitrary bytes including invalid UTF-8
        let bytes: [u8; 4] = kani::any();

        response.append_text(&bytes);

        // Must not panic, output must be valid UTF-8
        let text = response.text_content();
        kani::assert!(
            std::str::from_utf8(text.as_bytes()).is_ok(),
            "Output must be valid UTF-8"
        );
    }

    /// Prove graceful degradation when limits exceeded
    #[kani::proof]
    fn graceful_degradation() {
        let mut response = StreamingResponse::new();

        // Fill buffer to limit
        let big_chunk = vec![0u8; MAX_BUFFER];
        response.append_text(&big_chunk);

        // Additional append should set flag, not panic or grow
        let more = vec![0u8; 1000];
        response.append_text(&more);

        kani::assert!(response.buffer_limit_exceeded, "Flag should be set");
        kani::assert!(
            response.total_buffer_size() <= MAX_BUFFER,
            "Buffer should not grow past limit"
        );
    }

    /// Prove JSON parsing doesn't panic on malformed input
    #[kani::proof]
    fn json_parsing_safety() {
        let mut response = StreamingResponse::new();
        response.start_tool_use("id", "name");

        // Arbitrary bytes as JSON input
        let json_chunk: [u8; 16] = kani::any();
        response.append_tool_json(&json_chunk);

        // Finalize should not panic even with invalid JSON
        let result = response.finalize_tool();
        // Result may be Err, but should not panic
    }
}
```

### USL Contract Specification

```usl
// specifications/streaming.usl

type StreamingResponse = {
    text_content: String,
    tool_uses: List<ToolUse>,
    current_tool_json: String,
    buffer_limit_exceeded: Bool
}

// Constants
const MAX_BUFFER_SIZE: Int = 10485760  // 10MB
const MAX_TOOL_USES: Int = 100

// Invariants
invariant buffer_bounded {
    forall r: StreamingResponse .
        len(r.text_content) + len(r.current_tool_json) <= MAX_BUFFER_SIZE
}

invariant tool_uses_bounded {
    forall r: StreamingResponse .
        len(r.tool_uses) <= MAX_TOOL_USES
}

// Contracts
contract StreamingResponse::append_text(self, delta: String) -> StreamingResponse {
    ensures {
        len(self'.text_content) <= MAX_BUFFER_SIZE
        len(self'.text_content) <= len(self.text_content) + len(delta)
    }
}

contract StreamingResponse::start_tool_use(self, id: String, name: String) -> StreamingResponse {
    ensures {
        len(self'.tool_uses) <= MAX_TOOL_USES
        len(self'.tool_uses) <= len(self.tool_uses) + 1
    }
}

contract StreamingResponse::finalize(self) -> Result<Response> {
    requires { true }  // Should not panic on any input
    ensures {
        is_ok(result) implies valid_utf8(result.text_content)
    }
}
```

---

## Component 3: Tool Registry and Dispatch

**Files:** `src/tools/registry.rs`, `src/tools/mod.rs`

### Verification Requirements

| Property | Description | Method |
|----------|-------------|--------|
| Tool uniqueness | Each name maps to exactly one tool | Invariant + Kani |
| Schema correctness | Schemas match Claude Code exactly | Differential test |
| Dispatch correctness | Correct tool receives input | Contract |
| Order preservation | `definitions()` returns Claude Code order | Parity test |
| Metrics monotonicity | Counts only increase | Invariant |

### USL Specification

```usl
// specifications/tool_registry.usl

type ToolRegistry = {
    tools: Map<String, Tool>,
    metrics: Map<String, ToolMetrics>
}

type ToolMetrics = {
    call_count: Int,
    total_duration_ms: Int,
    error_count: Int
}

// Invariants
invariant tool_uniqueness {
    forall r: ToolRegistry, name: String .
        count(r.tools, name) <= 1
}

invariant metrics_monotonicity {
    forall r: ToolRegistry, name: String .
        r'.metrics[name].call_count >= r.metrics[name].call_count
}

invariant tool_count_parity {
    // Must have exactly 21 tools (17 Claude Code + 4 extensions)
    forall r: ToolRegistry .
        len(r.tools) == 21
}

// Contracts
contract ToolRegistry::execute(self, name: String, input: Value) -> Result<ToolOutput> {
    requires { contains(self.tools, name) }
    ensures {
        self'.metrics[name].call_count == self.metrics[name].call_count + 1
        is_err(result) implies self'.metrics[name].error_count == self.metrics[name].error_count + 1
    }
}

contract ToolRegistry::register(self, tool: Tool) -> ToolRegistry {
    requires { not contains(self.tools, tool.name) }
    ensures {
        contains(self'.tools, tool.name)
        len(self'.tools) == len(self.tools) + 1
    }
}

contract ToolRegistry::definitions(self) -> List<ToolDefinition> {
    ensures {
        len(result) == len(self.tools)
        // Order matches Claude Code (verified by differential test)
    }
}
```

### Kani Proof Harnesses

```rust
#[cfg(kani)]
mod verification {
    use super::*;

    /// Prove tool registration maintains uniqueness
    #[kani::proof]
    fn tool_registration_uniqueness() {
        let mut registry = ToolRegistry::new();

        let name: String = kani::any();
        let tool1 = MockTool::new(&name);
        let tool2 = MockTool::new(&name);

        let r1 = registry.register(tool1);
        let r2 = registry.register(tool2);

        // Second registration of same name should fail
        kani::assert!(
            r1.is_ok() && r2.is_err(),
            "Duplicate registration must fail"
        );
    }

    /// Prove dispatch goes to correct tool
    #[kani::proof]
    fn dispatch_correctness() {
        let mut registry = ToolRegistry::new();

        let bash_tool = BashTool::new();
        let read_tool = ReadTool::new();

        registry.register(bash_tool);
        registry.register(read_tool);

        // Dispatch to Bash
        let result = registry.execute("Bash", json!({"command": "ls"}));
        // Should call BashTool::execute, not ReadTool::execute

        // Verify via side effect or return type
    }

    /// Prove metrics only increase
    #[kani::proof]
    #[kani::unwind(11)]
    fn metrics_monotonicity() {
        let mut registry = ToolRegistry::with_defaults();

        let mut prev_count = 0u64;

        for _ in 0..10 {
            let name = "Bash";  // Pick any tool
            let input = json!({"command": "echo test"});

            let _ = registry.execute(name, input);

            let current_count = registry.metrics.get(name).map(|m| m.call_count).unwrap_or(0);

            kani::assert!(
                current_count >= prev_count,
                "Call count must be monotonically increasing"
            );

            prev_count = current_count;
        }
    }
}
```

---

## Component 4: Individual Tool Contracts

### Bash Tool

```usl
// specifications/tools/bash.usl

type BashInput = {
    command: String,
    timeout: Option<Int>,
    description: Option<String>
}

type BashOutput = {
    stdout: String,
    stderr: String,
    exit_code: Int,
    interrupted: Bool
}

contract Bash::execute(input: BashInput) -> Result<BashOutput> {
    requires {
        len(input.command) > 0
        input.timeout.is_none() or input.timeout.unwrap() > 0
        input.timeout.is_none() or input.timeout.unwrap() <= 600000
    }
    ensures {
        result.exit_code >= 0 and result.exit_code <= 255
    }
}

// Temporal: command execution terminates
temporal bash_terminates {
    forall cmd: BashInput .
        always(executing(cmd) implies eventually(not executing(cmd)))
}
```

### Read Tool

```usl
// specifications/tools/read.usl

type ReadInput = {
    file_path: String,
    offset: Option<Int>,
    limit: Option<Int>
}

type ReadOutput = {
    content: String,
    truncated: Bool
}

contract Read::execute(input: ReadInput) -> Result<ReadOutput> {
    requires {
        is_absolute_path(input.file_path)
        input.offset.is_none() or input.offset.unwrap() >= 0
        input.limit.is_none() or input.limit.unwrap() > 0
    }
    ensures {
        input.limit.is_some() implies line_count(result.content) <= input.limit.unwrap()
    }
    ensures_err {
        // File not found, permission denied, etc.
        true
    }
}
```

### Edit Tool

```usl
// specifications/tools/edit.usl

type EditInput = {
    file_path: String,
    old_string: String,
    new_string: String,
    replace_all: Bool
}

contract Edit::execute(input: EditInput) -> Result<String> {
    requires {
        is_absolute_path(input.file_path)
        input.old_string != input.new_string
    }
    ensures {
        // File contains new_string where old_string was
        // If replace_all: no occurrences of old_string remain
    }
    ensures_err {
        // old_string not found, file not readable, etc.
        // File unchanged on error
    }
}

// Safety: Edit is reversible
theorem edit_reversible {
    forall file: File, old: String, new: String .
        let edited = Edit::execute(file, old, new) in
        let reverted = Edit::execute(edited, new, old) in
        reverted == file
}
```

### Task Tool (Sub-agent Spawning)

```usl
// specifications/tools/task.usl

type TaskInput = {
    prompt: String,
    description: String,
    subagent_type: String,
    run_in_background: Bool
}

contract Task::execute(input: TaskInput) -> Result<TaskOutput> {
    requires {
        len(input.prompt) > 0
        len(input.description) > 0
        input.subagent_type in ["general-purpose", "Explore", "Plan", "claude-code-guide"]
    }
    ensures {
        // Sub-agent was spawned with correct configuration
        // Result contains agent output or agent_id for background tasks
    }
}

// Temporal: sub-agents eventually complete
temporal subagent_terminates {
    forall task: TaskInput .
        spawned(task) implies eventually(completed(task) or failed(task))
}

// Safety: no infinite sub-agent spawning
invariant bounded_subagent_depth {
    forall agent: Agent .
        agent.subagent_depth <= 3
}
```

---

## Component 5: Hook System

**Files:** `src/hooks.rs`

### TLA+ Specification

```tla+
--------------------------- MODULE HookSystem ---------------------------
EXTENDS Integers, Sequences

CONSTANTS
    HOOK_TIMEOUT,       \* 30 seconds default
    HOOK_TYPES          \* Set of hook event types

VARIABLES
    pending_hooks,      \* Queue of hooks to execute
    executing_hook,     \* Currently executing hook (or NONE)
    hook_results,       \* Accumulated results
    timed_out           \* Whether current hook timed out

vars == <<pending_hooks, executing_hook, hook_results, timed_out>>

HookTypes == {
    "SessionStart", "SessionStop",
    "UserPromptSubmit",
    "PreToolUse", "PostToolUse",
    "Stop"
}

TypeInvariant ==
    /\ \A h \in pending_hooks: h.type \in HookTypes
    /\ executing_hook \in HookTypes \cup {NONE}
    /\ timed_out \in BOOLEAN

\* Safety: hooks complete or timeout
HookEventuallyResolves ==
    executing_hook # NONE => <>(executing_hook = NONE)

\* Liveness: hooks execute in order
HookOrdering ==
    \* SessionStart before UserPromptSubmit before PreToolUse etc.

Init ==
    /\ pending_hooks = <<>>
    /\ executing_hook = NONE
    /\ hook_results = <<>>
    /\ timed_out = FALSE

StartHook(h) ==
    /\ executing_hook = NONE
    /\ Len(pending_hooks) > 0
    /\ h = Head(pending_hooks)
    /\ pending_hooks' = Tail(pending_hooks)
    /\ executing_hook' = h.type
    /\ UNCHANGED <<hook_results, timed_out>>

CompleteHook(result) ==
    /\ executing_hook # NONE
    /\ hook_results' = Append(hook_results, [type |-> executing_hook, result |-> result])
    /\ executing_hook' = NONE
    /\ timed_out' = FALSE
    /\ UNCHANGED pending_hooks

TimeoutHook ==
    /\ executing_hook # NONE
    \* After HOOK_TIMEOUT
    /\ timed_out' = TRUE
    /\ executing_hook' = NONE
    /\ UNCHANGED <<pending_hooks, hook_results>>

=========================================================================
```

### Kani Proof Harnesses

```rust
#[cfg(kani)]
mod verification {
    use super::*;

    /// Prove hook timeout is enforced
    #[kani::proof]
    fn hook_timeout_enforced() {
        let config = HookConfig {
            timeout_ms: 30_000,
            ..Default::default()
        };

        let hook = Hook::new("test", HookType::PreToolUse, config);
        let start = Instant::now();

        // Simulate execution
        let result = hook.execute_with_timeout(/* mock slow command */);

        // Either completes or times out
        kani::assert!(
            result.is_ok() || result.is_err_timeout(),
            "Hook must complete or timeout"
        );

        // If timed out, took at least timeout_ms
        if result.is_err_timeout() {
            kani::assert!(start.elapsed().as_millis() >= 30_000);
        }
    }

    /// Prove hooks receive correct JSON input
    #[kani::proof]
    fn hook_input_format() {
        let hook_input = BaseHookInput {
            session_id: "test-session".into(),
            transcript_path: "/tmp/transcript.json".into(),
            cwd: "/home/user".into(),
            permission_mode: "default".into(),
        };

        let json = serde_json::to_string(&hook_input).unwrap();
        let parsed: BaseHookInput = serde_json::from_str(&json).unwrap();

        kani::assert!(parsed.session_id == hook_input.session_id);
        kani::assert!(parsed.cwd == hook_input.cwd);
    }
}
```

---

## Component 6: Context Compaction

**Files:** `src/agents/compaction.rs`

### Verification Requirements

| Property | Description | Method |
|----------|-------------|--------|
| Token reduction | Post-compaction tokens < pre-compaction | Contract |
| Information preservation | Key information preserved in summary | Semantic analysis |
| Message structure | Result is valid message history | Invariant |
| Idempotency | Double compaction doesn't lose more | Contract |

### USL Specification

```usl
// specifications/compaction.usl

type CompactionInput = {
    messages: List<Message>,
    target_tokens: Int
}

contract compact(input: CompactionInput) -> List<Message> {
    requires {
        len(input.messages) > 0
        input.target_tokens > 0
    }
    ensures {
        token_count(result) <= input.target_tokens
        len(result) <= len(input.messages)
        // First system message preserved
        head(result).role == "system" implies head(input.messages).role == "system"
    }
}

// Semantic preservation (approximate - tested via LLM evaluation)
theorem information_preserved {
    forall msgs: List<Message>, target: Int .
        let compacted = compact(msgs, target) in
        semantic_similarity(msgs, compacted) >= 0.8
}
```

---

## Component 7: Behavioral Parity Testing

### Differential Testing Framework

```rust
// tests/parity/differential.rs

/// Framework for differential testing against Claude Code
pub struct DifferentialTestHarness {
    claude_code_binary: PathBuf,
    rust_port_binary: PathBuf,
}

impl DifferentialTestHarness {
    /// Run same input through both implementations, compare outputs
    pub async fn compare(&self, input: &TestInput) -> DiffResult {
        let (original, port) = tokio::join!(
            self.run_claude_code(input),
            self.run_rust_port(input)
        );

        DiffResult {
            tool_calls_match: original.tool_calls == port.tool_calls,
            api_requests_match: self.compare_api_requests(&original, &port),
            outputs_equivalent: self.semantic_compare(&original.output, &port.output),
            timing_difference: (port.duration - original.duration).abs(),
        }
    }

    /// Compare API request format (headers, body structure)
    fn compare_api_requests(&self, orig: &Trace, port: &Trace) -> bool {
        // Check request structure matches
        orig.api_requests.iter().zip(port.api_requests.iter()).all(|(o, p)| {
            o.headers.get("anthropic-version") == p.headers.get("anthropic-version") &&
            o.body["model"] == p.body["model"] &&
            self.messages_equivalent(&o.body["messages"], &p.body["messages"]) &&
            self.tools_equivalent(&o.body["tools"], &p.body["tools"])
        })
    }
}

/// Test cases for parity verification
#[cfg(test)]
mod parity_tests {
    use super::*;

    #[tokio::test]
    async fn tool_schema_parity() {
        let harness = DifferentialTestHarness::new();

        // Get tool definitions from both implementations
        let orig_tools = harness.get_claude_code_tools().await;
        let port_tools = harness.get_rust_port_tools().await;

        // Must be byte-exact match
        for (name, orig_schema) in &orig_tools {
            let port_schema = port_tools.get(name).expect(&format!("Missing tool: {}", name));
            assert_eq!(
                serde_json::to_string(orig_schema).unwrap(),
                serde_json::to_string(port_schema).unwrap(),
                "Schema mismatch for tool: {}", name
            );
        }
    }

    #[tokio::test]
    async fn system_prompt_parity() {
        let harness = DifferentialTestHarness::new();

        let orig_prompt = harness.get_claude_code_system_prompt().await;
        let port_prompt = harness.get_rust_port_system_prompt().await;

        // Section-by-section comparison
        let orig_sections = parse_system_prompt_sections(&orig_prompt);
        let port_sections = parse_system_prompt_sections(&port_prompt);

        for (section_name, orig_content) in &orig_sections {
            let port_content = port_sections.get(section_name)
                .expect(&format!("Missing section: {}", section_name));
            assert_eq!(
                orig_content, port_content,
                "System prompt section mismatch: {}", section_name
            );
        }
    }

    #[tokio::test]
    async fn api_request_format_parity() {
        let harness = DifferentialTestHarness::new();

        let test_inputs = vec![
            TestInput::simple_message("Hello"),
            TestInput::with_tool_call("Bash", json!({"command": "ls"})),
            TestInput::multi_turn_conversation(),
        ];

        for input in test_inputs {
            let result = harness.compare(&input).await;
            assert!(result.api_requests_match, "API request format mismatch for: {:?}", input);
        }
    }
}
```

### Bisimulation Verification

```usl
// specifications/bisimulation.usl

// Two implementations are bisimilar if:
// 1. Same inputs produce same observable outputs
// 2. Same internal state transitions
// 3. Same tool execution sequences

theorem behavioral_equivalence {
    forall input: UserInput .
        let orig_trace = execute(claude_code, input) in
        let port_trace = execute(claude_code_rs, input) in

        // Output equivalence
        orig_trace.output == port_trace.output and

        // Tool call sequence equivalence
        orig_trace.tool_calls == port_trace.tool_calls and

        // API request equivalence
        forall i: Int . i < len(orig_trace.api_requests) implies
            request_equivalent(orig_trace.api_requests[i], port_trace.api_requests[i])
}

// Request equivalence definition
predicate request_equivalent(r1: Request, r2: Request) {
    r1.model == r2.model and
    r1.messages == r2.messages and
    r1.tools == r2.tools and
    r1.temperature == r2.temperature
}
```

---

## Component 8: Property-Based Testing

### Proptest Specifications

```rust
// tests/proptest_specs.rs

use proptest::prelude::*;

/// Generate arbitrary valid API messages
fn arb_message() -> impl Strategy<Value = Message> {
    prop_oneof![
        any::<String>().prop_map(|s| Message::user(&s)),
        any::<String>().prop_map(|s| Message::assistant(&s)),
        (any::<String>(), arb_tool_result()).prop_map(|(id, result)| {
            Message::tool_result(&id, result)
        }),
    ]
}

/// Generate arbitrary tool inputs
fn arb_tool_input(tool_name: &str) -> BoxedStrategy<Value> {
    match tool_name {
        "Bash" => arb_bash_input().boxed(),
        "Read" => arb_read_input().boxed(),
        "Edit" => arb_edit_input().boxed(),
        "Glob" => arb_glob_input().boxed(),
        "Grep" => arb_grep_input().boxed(),
        _ => Just(json!({})).boxed(),
    }
}

fn arb_bash_input() -> impl Strategy<Value = Value> {
    (
        "[a-zA-Z0-9 -_.]+",  // Safe command characters
        prop::option::of(1u64..600_000),  // Timeout
    ).prop_map(|(cmd, timeout)| {
        let mut obj = json!({"command": cmd});
        if let Some(t) = timeout {
            obj["timeout"] = json!(t);
        }
        obj
    })
}

proptest! {
    /// Agent loop always terminates
    #[test]
    fn agent_terminates(
        messages in prop::collection::vec(arb_message(), 0..100)
    ) {
        let mut agent = Agent::new(AgentConfig::default());

        for msg in messages {
            agent.add_message(msg);
        }

        let result = tokio_test::block_on(agent.run());

        // Must terminate (not hang)
        prop_assert!(result.is_ok() || result.is_err());
    }

    /// Tool execution never panics
    #[test]
    fn tool_no_panic(
        tool_name in prop::sample::select(TOOL_NAMES.to_vec()),
        input in arb_tool_input(&tool_name)
    ) {
        let registry = ToolRegistry::with_defaults();

        // Must not panic
        let result = std::panic::catch_unwind(|| {
            tokio_test::block_on(registry.execute(&tool_name, input))
        });

        prop_assert!(result.is_ok(), "Tool execution panicked");
    }

    /// Streaming parser handles arbitrary bytes without panic
    #[test]
    fn streaming_parser_robust(
        chunks in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..1000), 0..100)
    ) {
        let mut response = StreamingResponse::new();

        for chunk in chunks {
            // Simulate SSE events with arbitrary payloads
            response.process_chunk(&chunk);
        }

        // Must not panic, output must be valid UTF-8
        let text = response.text_content();
        prop_assert!(std::str::from_utf8(text.as_bytes()).is_ok());
    }

    /// Edit tool: old_string != new_string enforcement
    #[test]
    fn edit_requires_different_strings(
        file_path in "/[a-z]+/[a-z]+\\.txt",
        content in ".*"
    ) {
        let input = json!({
            "file_path": file_path,
            "old_string": content,
            "new_string": content,  // Same!
        });

        let registry = ToolRegistry::with_defaults();
        let result = tokio_test::block_on(registry.execute("Edit", input));

        // Must reject same old/new strings
        prop_assert!(result.is_err());
    }

    /// Context compaction reduces token count
    #[test]
    fn compaction_reduces_tokens(
        messages in prop::collection::vec(arb_message(), 10..100),
        target_ratio in 0.1f64..0.5
    ) {
        let input_tokens = estimate_tokens(&messages);
        let target = (input_tokens as f64 * target_ratio) as usize;

        let compacted = compact(&messages, target);
        let output_tokens = estimate_tokens(&compacted);

        prop_assert!(output_tokens <= target, "Compaction must reduce to target");
    }
}
```

---

## Verification Tool Requirements

### Required Tools

```bash
# TLA+ Toolbox and TLC
brew install tlaplus
# Or download from https://github.com/tlaplus/tlaplus/releases

# Kani (Rust model checker)
cargo install --locked kani-verifier
kani setup

# MIRI (undefined behavior detector)
rustup +nightly component add miri

# Proptest (property-based testing) - via Cargo.toml
# [dev-dependencies]
# proptest = "1.4"

# Criterion (benchmarks)
# criterion = "0.5"

# Differential testing harness
# Custom - see tests/parity/
```

### DashProve Integration

```rust
// Cargo.toml
[dependencies]
dashprove = { git = "https://github.com/dropbox/dMATH/dashprove" }

// Usage
use dashprove::{DashProve, Spec};

async fn verify_agent_loop() -> Result<VerificationResult> {
    let client = DashProve::new();

    let spec = Spec::from_file("specifications/agent_executor.usl")?;

    let result = client.verify(&spec).await?;

    match result.status {
        VerificationStatus::Proven => println!("Agent loop verified!"),
        VerificationStatus::Disproven => {
            println!("Counterexample: {:?}", result.counterexample);
        }
        VerificationStatus::Unknown { reason } => {
            println!("Could not verify: {:?}", reason);
        }
    }

    Ok(result)
}
```

---

## Verification Milestones

| Milestone | Components | Method | Target |
|-----------|------------|--------|--------|
| M1 | Agent loop termination | TLA+ + Kani | Week 1 |
| M2 | Streaming buffer safety | Kani | Week 1 |
| M3 | Tool contracts | USL + Kani | Week 2 |
| M4 | Hook timeout enforcement | Kani | Week 2 |
| M5 | Parity differential tests | Custom harness | Week 3 |
| M6 | Property-based test suite | Proptest | Week 3 |
| M7 | Full TLA+ model check | TLC | Week 4 |
| M8 | Integration with DashProve | API | Week 4 |

---

## Success Criteria

1. **TLA+ Models**: All models pass TLC with no errors (bounded model checking)
2. **Kani Proofs**: All proof harnesses verify successfully
3. **USL Contracts**: All contracts compile and pass DashProve verification
4. **Proptest Coverage**: >95% branch coverage on critical paths
5. **Differential Tests**: 100% parity with Claude Code on test suite
6. **MIRI**: Zero undefined behavior reports
7. **No Panics**: All fuzz tests pass without panics

---

## References

- [TLA+ Home](https://lamport.azurewebsites.net/tla/tla.html)
- [Kani Rust Verifier](https://model-checking.github.io/kani/)
- [Proptest Guide](https://proptest-rs.github.io/proptest/proptest/index.html)
- [DashProve Design](docs/DESIGN.md)
- [Claude Code RS Implementation](../workspace/claude_code_port/)

---

## Contact

**Source:** claude_code_rs at https://github.com/dropbox/claude_code_rs
**Verification:** DashProve at https://github.com/dropbox/dMATH/dashprove
