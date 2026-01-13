---- MODULE GraphExecution ----
(***************************************************************************)
(* TLA+ Specification for DashFlow Graph Execution                         *)
(*                                                                         *)
(* This spec models the execution of a directed graph of nodes where:      *)
(* - Nodes can only execute after all predecessors complete                *)
(* - Parallel branches execute concurrently                                *)
(* - The system eventually terminates (no deadlock)                        *)
(*                                                                         *)
(* Status: DRAFT - Example for future implementation                       *)
(***************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Nodes, Edges, EntryNode, ExitNodes

VARIABLES nodeState, edgeTraversed, currentNode

vars == <<nodeState, edgeTraversed, currentNode>>

(***************************************************************************)
(* Helper Functions                                                        *)
(***************************************************************************)

Predecessors(n) == {m \in Nodes : <<m, n>> \in Edges}
Successors(n) == {m \in Nodes : <<n, m>> \in Edges}

(***************************************************************************)
(* Type Invariant                                                          *)
(***************************************************************************)

TypeInvariant ==
    /\ nodeState \in [Nodes -> {"pending", "active", "completed", "error"}]
    /\ edgeTraversed \subseteq Edges
    /\ currentNode \subseteq Nodes

(***************************************************************************)
(* Safety Invariant: A node can only execute if all predecessors completed *)
(***************************************************************************)

SafetyInvariant ==
    \A n \in Nodes:
        nodeState[n] = "active" =>
            \A pred \in Predecessors(n): nodeState[pred] = "completed"

(***************************************************************************)
(* No Orphan Execution: Only reachable nodes can execute                   *)
(***************************************************************************)

NoOrphanExecution ==
    \A n \in Nodes:
        nodeState[n] \in {"active", "completed"} =>
            n = EntryNode \/ \E pred \in Predecessors(n): nodeState[pred] = "completed"

(***************************************************************************)
(* Initial State                                                           *)
(***************************************************************************)

Init ==
    /\ nodeState = [n \in Nodes |-> IF n = EntryNode THEN "active" ELSE "pending"]
    /\ edgeTraversed = {}
    /\ currentNode = {EntryNode}

(***************************************************************************)
(* Node Completion: A node completes and enables successors               *)
(***************************************************************************)

NodeComplete(n) ==
    /\ nodeState[n] = "active"
    /\ nodeState' = [nodeState EXCEPT ![n] = "completed"]
    /\ edgeTraversed' = edgeTraversed \cup {<<n, s>> : s \in Successors(n)}
    /\ currentNode' = (currentNode \ {n}) \cup
        {s \in Successors(n) : \A pred \in Predecessors(s) \ {n}: nodeState[pred] = "completed"}

(***************************************************************************)
(* Node Error: A node fails                                                *)
(***************************************************************************)

NodeError(n) ==
    /\ nodeState[n] = "active"
    /\ nodeState' = [nodeState EXCEPT ![n] = "error"]
    /\ UNCHANGED <<edgeTraversed, currentNode>>

(***************************************************************************)
(* Enable Successor: Activate a node whose predecessors are all completed  *)
(***************************************************************************)

EnableSuccessor(n) ==
    /\ nodeState[n] = "pending"
    /\ \A pred \in Predecessors(n): nodeState[pred] = "completed"
    /\ nodeState' = [nodeState EXCEPT ![n] = "active"]
    /\ currentNode' = currentNode \cup {n}
    /\ UNCHANGED edgeTraversed

(***************************************************************************)
(* Next State Relation                                                     *)
(***************************************************************************)

Next ==
    \/ \E n \in Nodes: NodeComplete(n)
    \/ \E n \in Nodes: NodeError(n)
    \/ \E n \in Nodes: EnableSuccessor(n)

(***************************************************************************)
(* Liveness: Eventually all nodes complete or error (no deadlock)          *)
(***************************************************************************)

LivenessProperty ==
    <>(\A n \in Nodes: nodeState[n] \in {"completed", "error"})

(***************************************************************************)
(* Fairness: If a node can complete, it eventually will                    *)
(***************************************************************************)

Fairness ==
    /\ \A n \in Nodes: WF_vars(NodeComplete(n))
    /\ \A n \in Nodes: WF_vars(EnableSuccessor(n))

(***************************************************************************)
(* Specification                                                           *)
(***************************************************************************)

Spec == Init /\ [][Next]_vars /\ Fairness

(***************************************************************************)
(* Properties to Check                                                     *)
(***************************************************************************)

THEOREM Spec => []TypeInvariant
THEOREM Spec => []SafetyInvariant
THEOREM Spec => []NoOrphanExecution
THEOREM Spec => LivenessProperty

====
