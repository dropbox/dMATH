---------------------------- MODULE AgentModel ----------------------------
(* Simplified agent model for model-based test generation
   This spec is used to generate test cases covering all states and transitions *)

EXTENDS Integers, Sequences

CONSTANTS
    MaxIterations   \* Set to 3 for test generation

VARIABLES
    state,          \* Agent state
    iteration       \* Iteration counter

vars == <<state, iteration>>

States == {"Idle", "Processing", "ToolExecution", "Done", "Error"}

(* Type invariant *)
TypeOK ==
    /\ state \in States
    /\ iteration \in 0..MaxIterations

(* Initial state *)
Init ==
    /\ state = "Idle"
    /\ iteration = 0

(* State transitions *)

StartProcessing ==
    /\ state = "Idle"
    /\ iteration < MaxIterations
    /\ state' = "Processing"
    /\ iteration' = iteration + 1

NeedTool ==
    /\ state = "Processing"
    /\ state' = "ToolExecution"
    /\ UNCHANGED iteration

ToolComplete ==
    /\ state = "ToolExecution"
    /\ state' = "Processing"
    /\ iteration' = iteration + 1

Finish ==
    /\ state = "Processing"
    /\ state' = "Done"
    /\ UNCHANGED iteration

HitError ==
    /\ state \in {"Processing", "ToolExecution"}
    /\ state' = "Error"
    /\ UNCHANGED iteration

Reset ==
    /\ state \in {"Done", "Error"}
    /\ state' = "Idle"
    /\ iteration' = 0

(* Next state *)
Next ==
    \/ StartProcessing
    \/ NeedTool
    \/ ToolComplete
    \/ Finish
    \/ HitError
    \/ Reset

(* Specification *)
Spec == Init /\ [][Next]_vars

(* Invariants *)
BoundedIterations == iteration <= MaxIterations
ValidStateTransitions ==
    state = "Done" => state' \in {"Done", "Idle"}

(* Liveness *)
EventuallyDone == <>(state = "Done")

=============================================================================
\* Test generation hints for dashprove-mbt:
\*
\* State coverage: Generate tests reaching each state
\*   - Idle (initial)
\*   - Processing (after StartProcessing)
\*   - ToolExecution (after NeedTool)
\*   - Done (after Finish)
\*   - Error (after HitError)
\*
\* Transition coverage: Generate tests executing each transition
\*   - StartProcessing: Idle -> Processing
\*   - NeedTool: Processing -> ToolExecution
\*   - ToolComplete: ToolExecution -> Processing
\*   - Finish: Processing -> Done
\*   - HitError: Processing/ToolExecution -> Error
\*   - Reset: Done/Error -> Idle
\*
\* Boundary coverage: Test at iteration boundaries
\*   - iteration = 0 (initial)
\*   - iteration = MaxIterations - 1 (near limit)
\*   - iteration = MaxIterations (at limit)
