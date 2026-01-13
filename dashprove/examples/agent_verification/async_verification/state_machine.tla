---------------------------- MODULE AgentStateMachine ----------------------------
(* TLA+ specification for an AI agent state machine
   Used for async verification and trace checking *)

EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    MaxIterations,      \* Maximum allowed iterations (e.g., 10)
    MaxMessages,        \* Maximum messages in history
    MaxToolCalls        \* Maximum concurrent tool calls

VARIABLES
    state,              \* Current agent state
    iteration,          \* Current iteration count
    messages,           \* Message history
    pendingTools,       \* Tool calls awaiting completion
    completedTools      \* Completed tool calls

vars == <<state, iteration, messages, pendingTools, completedTools>>

(* State type definitions *)
States == {"Idle", "Processing", "ToolExecution", "Waiting", "Done", "Error"}

TypeInvariant ==
    /\ state \in States
    /\ iteration \in 0..MaxIterations
    /\ messages \in Seq(STRING)
    /\ pendingTools \in SUBSET STRING
    /\ completedTools \in SUBSET STRING

(* Initial state *)
Init ==
    /\ state = "Idle"
    /\ iteration = 0
    /\ messages = <<>>
    /\ pendingTools = {}
    /\ completedTools = {}

(* Receive a user message *)
ReceiveMessage ==
    /\ state = "Idle"
    /\ Len(messages) < MaxMessages
    /\ state' = "Processing"
    /\ iteration' = iteration + 1
    /\ messages' = Append(messages, "user_message")
    /\ UNCHANGED <<pendingTools, completedTools>>

(* Process message and decide on action *)
ProcessMessage ==
    /\ state = "Processing"
    /\ iteration < MaxIterations
    /\ \/ /\ state' = "ToolExecution"  \* Need to call a tool
          /\ UNCHANGED <<iteration, messages, pendingTools, completedTools>>
       \/ /\ state' = "Done"           \* No tool needed, complete
          /\ UNCHANGED <<iteration, messages, pendingTools, completedTools>>

(* Initiate a tool call *)
InitiateToolCall ==
    /\ state = "ToolExecution"
    /\ Cardinality(pendingTools) < MaxToolCalls
    /\ \E tool \in {"read_file", "write_file", "bash", "search"} :
        /\ tool \notin pendingTools
        /\ tool \notin completedTools
        /\ pendingTools' = pendingTools \union {tool}
    /\ state' = "Waiting"
    /\ UNCHANGED <<iteration, messages, completedTools>>

(* Tool call completes *)
CompleteToolCall ==
    /\ state = "Waiting"
    /\ pendingTools /= {}
    /\ \E tool \in pendingTools :
        /\ completedTools' = completedTools \union {tool}
        /\ pendingTools' = pendingTools \ {tool}
    /\ state' = IF pendingTools' = {} THEN "Processing" ELSE "Waiting"
    /\ iteration' = iteration + 1
    /\ UNCHANGED messages

(* Complete the agent loop *)
Complete ==
    /\ state = "Processing"
    /\ pendingTools = {}
    /\ state' = "Done"
    /\ UNCHANGED <<iteration, messages, pendingTools, completedTools>>

(* Error handling *)
HandleError ==
    /\ state \in {"Processing", "Waiting", "ToolExecution"}
    /\ state' = "Error"
    /\ UNCHANGED <<iteration, messages, pendingTools, completedTools>>

(* Next state relation *)
Next ==
    \/ ReceiveMessage
    \/ ProcessMessage
    \/ InitiateToolCall
    \/ CompleteToolCall
    \/ Complete
    \/ HandleError

(* Fairness - tools eventually complete *)
Fairness ==
    /\ WF_vars(CompleteToolCall)
    /\ WF_vars(Complete)

(* Specification *)
Spec == Init /\ [][Next]_vars /\ Fairness

(* Safety invariants *)
BoundedIterations == iteration <= MaxIterations

NoOrphanedTools == state = "Done" => pendingTools = {}

ValidTransitions ==
    /\ state = "Idle" => state' \in {"Processing", "Idle"}
    /\ state = "Done" => state' = "Done"
    /\ state = "Error" => state' = "Error"

(* Liveness properties *)
EventuallyTerminates == <>(state \in {"Done", "Error"})

ToolsEventuallyComplete ==
    \A tool \in pendingTools : <>(tool \in completedTools)

(* Model checking configuration *)
ASSUME MaxIterations \in Nat /\ MaxIterations > 0
ASSUME MaxMessages \in Nat /\ MaxMessages > 0
ASSUME MaxToolCalls \in Nat /\ MaxToolCalls > 0

=============================================================================
