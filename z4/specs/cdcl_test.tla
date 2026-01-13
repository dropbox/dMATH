---------------------------- MODULE cdcl_test ----------------------------
\* Test instance of CDCL with a specific UNSAT formula
\* Formula: (v1 OR v2) AND (NOT v1) AND (NOT v2)

EXTENDS Integers, Sequences, FiniteSets, TLC

\* Variables are model values (atoms)
CONSTANTS v1, v2

Variables == {v1, v2}

\* A literal is <<var, sign>> where sign is "pos" or "neg"
PosLit(v) == <<v, "pos">>
NegLit(v) == <<v, "neg">>

\* All possible literals
Literals == {<<v, s>> : v \in Variables, s \in {"pos", "neg"}}

\* Get the variable of a literal
Var(lit) == lit[1]

\* Get the sign of a literal
IsPositive(lit) == lit[2] = "pos"

\* The UNSAT formula: (v1 OR v2) AND (NOT v1) AND (NOT v2)
Clauses == {
    {PosLit(v1), PosLit(v2)},   \* v1 OR v2
    {NegLit(v1)},               \* NOT v1
    {NegLit(v2)}                \* NOT v2
}

\* Possible values for assignment
Values == {"TRUE", "FALSE", "UNDEF"}

VARIABLES
    assignment,
    trail,
    state,
    decisionLevel

vars == <<assignment, trail, state, decisionLevel>>

\* Get the value of a literal under current assignment
LitValue(lit) ==
    LET v == Var(lit)
        val == assignment[v]
    IN IF val = "UNDEF" THEN "UNDEF"
       ELSE IF IsPositive(lit) THEN val
            ELSE IF val = "TRUE" THEN "FALSE" ELSE "TRUE"

\* A clause is satisfied if at least one literal is TRUE
Satisfied(clause) == \E lit \in clause : LitValue(lit) = "TRUE"

\* A clause is falsified if all literals are FALSE
Falsified(clause) == \A lit \in clause : LitValue(lit) = "FALSE"

\* A clause is unit if exactly one literal is UNDEF and rest are FALSE
IsUnit(clause) ==
    /\ ~Satisfied(clause)
    /\ Cardinality({lit \in clause : LitValue(lit) = "UNDEF"}) = 1

\* Get the unit literal from a unit clause
UnitLit(clause) == CHOOSE lit \in clause : LitValue(lit) = "UNDEF"

Init ==
    /\ assignment = [v \in Variables |-> "UNDEF"]
    /\ trail = <<>>
    /\ state = "PROPAGATING"
    /\ decisionLevel = 0

\* Unit propagation
Propagate ==
    /\ state = "PROPAGATING"
    /\ \E clause \in Clauses :
        /\ IsUnit(clause)
        /\ LET lit == UnitLit(clause)
               v == Var(lit)
               val == IF IsPositive(lit) THEN "TRUE" ELSE "FALSE"
           IN /\ assignment' = [assignment EXCEPT ![v] = val]
              /\ trail' = Append(trail, lit)
              /\ UNCHANGED <<state, decisionLevel>>

\* Detect conflict
DetectConflict ==
    /\ state = "PROPAGATING"
    /\ \E clause \in Clauses : Falsified(clause)
    /\ state' = "CONFLICTING"
    /\ UNCHANGED <<assignment, trail, decisionLevel>>

\* No propagation, no conflict - ready to decide
ReadyToDecide ==
    /\ state = "PROPAGATING"
    /\ ~(\E clause \in Clauses : IsUnit(clause))
    /\ ~(\E clause \in Clauses : Falsified(clause))
    /\ state' = "DECIDING"
    /\ UNCHANGED <<assignment, trail, decisionLevel>>

\* Make a decision
Decide ==
    /\ state = "DECIDING"
    /\ \E v \in Variables :
        /\ assignment[v] = "UNDEF"
        /\ decisionLevel' = decisionLevel + 1
        /\ assignment' = [assignment EXCEPT ![v] = "TRUE"]
        /\ trail' = Append(trail, PosLit(v))
        /\ state' = "PROPAGATING"

\* All assigned - SAT
DeclareSat ==
    /\ state = "DECIDING"
    /\ \A v \in Variables : assignment[v] # "UNDEF"
    /\ state' = "SAT"
    /\ UNCHANGED <<assignment, trail, decisionLevel>>

\* Conflict at level 0 - UNSAT
DeclareUnsat ==
    /\ state = "CONFLICTING"
    /\ decisionLevel = 0
    /\ state' = "UNSAT"
    /\ UNCHANGED <<assignment, trail, decisionLevel>>

\* Backtrack (simplified - no learning for this test)
Backtrack ==
    /\ state = "CONFLICTING"
    /\ decisionLevel > 0
    /\ decisionLevel' = 0
    /\ assignment' = [v \in Variables |-> "UNDEF"]
    /\ trail' = <<>>
    /\ state' = "PROPAGATING"

Next ==
    \/ Propagate
    \/ DetectConflict
    \/ ReadyToDecide
    \/ Decide
    \/ DeclareSat
    \/ DeclareUnsat
    \/ Backtrack

Spec == Init /\ [][Next]_vars

\* Invariants
TypeInvariant ==
    /\ assignment \in [Variables -> Values]
    /\ state \in {"PROPAGATING", "DECIDING", "CONFLICTING", "SAT", "UNSAT"}

\* Soundness: If SAT, assignment satisfies all clauses
SatCorrect == state = "SAT" => \A clause \in Clauses : Satisfied(clause)

\* This formula is UNSAT - we should never reach SAT
NeverSat == state # "SAT"

=============================================================================
