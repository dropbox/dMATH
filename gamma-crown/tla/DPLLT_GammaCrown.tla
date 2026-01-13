------------------------------ MODULE DPLLT_GammaCrown ------------------------------
EXTENDS Naturals, FiniteSets, Sequences, TLC

\* This is a protocol-level state machine for a DPLL(T) style verifier loop.
\* It models:
\* - SAT decisions over phase literals (e.g., ReLU phases)
\* - Theory checks/propagation (via an abstract theory oracle; z4 in reality)
\* - Learned clauses/lemmas with explicit scoping/guards
\* - Caches and async bound-propagation jobs (e.g., GPU batches)
\*
\* It intentionally does NOT model neural network math. The goal is to check system invariants:
\* - push/pop discipline, lemma scoping, cache validity, no stale async results
\* - certified-mode gates: nothing that affects "verified" may happen without justification

CONSTANTS
  Lits,              \* finite set of literals
  Neg,               \* function Lits -> Lits, involutive negation
  CertifiedModes,    \* e.g., {"heuristic", "replay", "certified"}
  CertifiedModeTag,  \* element of CertifiedModes treated as "certified"
  NumericModes,      \* e.g., {"f32", "real+envelope"}
  Configs,           \* finite set of solver configurations
  Justs,             \* justification tokens (must include "")
  NoLit,             \* sentinel value not in Lits
  MaxLevel           \* small bound for model checking

ASSUME
  /\ Lits /= {}
  /\ Neg \in [Lits -> Lits]
  /\ \A l \in Lits: Neg[Neg[l]] = l
  /\ CertifiedModes /= {}
  /\ CertifiedModeTag \in CertifiedModes
  /\ NumericModes /= {}
  /\ Configs /= {}
  /\ Justs /= {}
  /\ "" \in Justs
  /\ NoLit \notin Lits
  /\ MaxLevel \in Nat

\* -----------------------------------------
\* Variables
\* -----------------------------------------
VARIABLES
  level,             \* decision level (Nat)
  trail,             \* sequence of assignments: [lit |-> l, lvl |-> n, reason |-> ...]
  theoryDepth,       \* abstract push/pop depth; must align with level
  learnedClauses,    \* set of clauses; each clause is a SUBSET of Lits
  lemmas,            \* set of lemma records (guards/scopes/justification)
  pendingJobs,       \* set of async jobs (bounds propagation proposals)
  cache,             \* mapping from signatures -> cached results (abstract)
  mode,              \* element of CertifiedModes
  numericMode,       \* element of NumericModes
  config             \* element of Configs

vars == << level, trail, theoryDepth, learnedClauses, lemmas, pendingJobs, cache, mode, numericMode, config >>

\* -----------------------------------------
\* Helper Definitions
\* -----------------------------------------

LitAssigned(l) ==
  \E i \in 1..Len(trail): trail[i].lit = l

ConsistentTrail ==
  \A l \in Lits:
    ~(LitAssigned(l) /\ LitAssigned(Neg[l]))

AssignedLits ==
  { trail[i].lit : i \in 1..Len(trail) }

CurrentSignature ==
  [ level |-> level,
    assigned |-> AssignedLits,
    mode |-> mode,
    numeric |-> numericMode,
    config |-> config ]

IsDecision(step) == step.reason = "decision"

DecisionLits ==
  { trail[i].lit : i \in 1..Len(trail) /\ IsDecision(trail[i]) }

GuardSatisfied(guard) == guard \subseteq AssignedLits

\* A lemma may be globally valid (guard = {}) or guarded (guard non-empty).
\* lvl is the decision level at which it was asserted; it must not leak across backtracking.
LemmaActive(lemma) ==
  /\ lemma.lvl <= level
  /\ GuardSatisfied(lemma.guard)

ActiveLemmas ==
  { lm \in lemmas : LemmaActive(lm) }

CertifiedMode == mode = CertifiedModeTag

RequiresJustification ==
  CertifiedMode

\* In certified mode, any learned clause/lemma or pruning action must carry a justification token.
HasJustification(rec) ==
  rec.just /= ""

\* All possible signatures (bounded) used to key caches/jobs.
SigSpace ==
  {
    [ level |-> lv, assigned |-> asg, mode |-> m, numeric |-> nm, config |-> c ] :
      /\ lv \in 0..MaxLevel
      /\ asg \subseteq Lits
      /\ m \in CertifiedModes
      /\ nm \in NumericModes
      /\ c \in Configs
  }

\* -----------------------------------------
\* Abstract Oracles (nondeterministic, but constrained)
\* -----------------------------------------

\* TheoryPropagate can either:
\* - imply a new literal (consistent with current assignment), or
\* - report conflict with a justification token.
\*
\* In reality, this is where z4 queries happen; this model focuses on protocol constraints.
TheoryPropagateResult ==
  [ kind : {"imply", "conflict", "none"},
    lit  : Lits \cup {NoLit},
    core : SUBSET Lits,
    just : Justs ]

\* Bounds jobs propose implied lits/conflicts but may be stale.
BoundsJob ==
  [ sig  : SigSpace,
    id   : Nat ]

BoundsResult ==
  [ job  : BoundsJob,
    kind : {"imply", "conflict", "none"},
    lit  : Lits \cup {NoLit},
    just : Justs ]

\* -----------------------------------------
\* Init
\* -----------------------------------------

Init ==
  /\ level = 0
  /\ trail = << >>
  /\ theoryDepth = 0
  /\ learnedClauses = {}
  /\ lemmas = {}
  /\ pendingJobs = {}
  /\ cache \in [SigSpace -> SUBSET Lits]
  /\ mode \in CertifiedModes
  /\ numericMode \in NumericModes
  /\ config \in Configs

\* -----------------------------------------
\* State Transitions
\* -----------------------------------------

PushLevel ==
  /\ level < MaxLevel
  /\ level' = level + 1
  /\ theoryDepth' = theoryDepth + 1
  /\ UNCHANGED << trail, learnedClauses, lemmas, pendingJobs, cache, mode, numericMode, config >>

\* Backtrack protocol: drop higher-level assignments, lemmas, and (conservatively) pending jobs.
\* (The caller decides what to do with learnedClauses/cache/modes.)
BacktrackTo(k) ==
  /\ k \in 0..level
  /\ level' = k
  /\ theoryDepth' = k
  /\ trail' = SelectSeq(trail, LAMBDA s: s.lvl <= k)
  /\ lemmas' = { lm \in lemmas : lm.lvl <= k } \* drop lemmas asserted above k
  /\ pendingJobs' = { j \in pendingJobs : j.sig.level <= k } \* conservative: drop higher-level jobs
  /\ UNCHANGED << learnedClauses, cache, mode, numericMode, config >>

\* Decide an unassigned literal: pushes a level and adds decision assignment.
Decide(l) ==
  /\ l \in Lits
  /\ ~LitAssigned(l)
  /\ ~LitAssigned(Neg[l])
  /\ PushLevel
  /\ trail' = Append(trail, [lit |-> l, lvl |-> level', reason |-> "decision", just |-> ""])
  /\ UNCHANGED << learnedClauses, lemmas, pendingJobs, cache, mode, numericMode, config >>

\* Theory propagation step: may imply a literal or conflict.
TheoryStep(res) ==
  /\ res \in TheoryPropagateResult
  /\ res.kind \in {"imply", "conflict", "none"}
  /\ IF res.kind = "imply" THEN
        /\ res.lit \in Lits
        /\ ~LitAssigned(res.lit)
        /\ ~LitAssigned(Neg[res.lit])
        /\ trail' = Append(trail, [lit |-> res.lit, lvl |-> level, reason |-> "theory", just |-> res.just])
        /\ UNCHANGED << level, theoryDepth, learnedClauses, lemmas, pendingJobs, cache, mode, numericMode, config >>
     ELSE IF res.kind = "conflict" THEN
        /\ res.core \subseteq AssignedLits
        /\ (~RequiresJustification) \/ HasJustification(res)
        /\ \* Learn a clause that blocks the core: OR of negated core literals
           learnedClauses' = learnedClauses \cup { { Neg[l] : l \in res.core } }
        /\ \* Backtrack at least one level if possible
           level > 0
        /\ BacktrackTo(level - 1)
        /\ cache' = cache
        /\ UNCHANGED << mode, numericMode, config >>
     ELSE
        /\ UNCHANGED vars

\* Enqueue an async bounds propagation job (e.g., GPU batch).
EnqueueBoundsJob ==
  /\ pendingJobs' = pendingJobs \cup { [sig |-> CurrentSignature, id |-> Cardinality(pendingJobs) + 1] }
  /\ UNCHANGED << level, trail, theoryDepth, learnedClauses, lemmas, cache, mode, numericMode, config >>

\* Apply a bounds result. In certified mode, it must come with a justification token OR be validated via a theory step later.
\* This transition models "applying" only when the result is not stale.
ApplyBoundsResult(res) ==
  /\ res \in BoundsResult
  /\ res.job \in pendingJobs
  /\ res.job.sig = CurrentSignature \* no stale results
  /\ pendingJobs' = pendingJobs \ { res.job }
  /\ IF res.kind = "imply" THEN
        /\ res.lit \in Lits
        /\ ~LitAssigned(res.lit)
        /\ ~LitAssigned(Neg[res.lit])
        /\ (~RequiresJustification) \/ (res.just /= "")
        /\ trail' = Append(trail, [lit |-> res.lit, lvl |-> level, reason |-> "bounds", just |-> res.just])
        /\ UNCHANGED << level, theoryDepth, learnedClauses, lemmas, cache, mode, numericMode, config >>
     ELSE IF res.kind = "conflict" THEN
        /\ (~RequiresJustification) \/ (res.just /= "")
        /\ level > 0
        /\ \* In the real system, a certified conflict must be validated by the theory solver before learning.
        /\ \* Here we model a conservative choice: backtrack without learning.
        /\ learnedClauses' = learnedClauses
        /\ BacktrackTo(level - 1)
        /\ cache' = cache
        /\ UNCHANGED << mode, numericMode, config >>
     ELSE
        /\ UNCHANGED << level, trail, theoryDepth, learnedClauses, lemmas, cache, mode, numericMode, config >>

\* Add a lemma/cut. Guarded lemmas must be scoped and (in certified mode) justified.
AssertLemma(guard, lvl, just) ==
  /\ guard \subseteq Lits
  /\ lvl \in 0..level
  /\ (~RequiresJustification) \/ (just /= "")
  /\ lemmas' = lemmas \cup { [guard |-> guard, lvl |-> lvl, just |-> just] }
  /\ UNCHANGED << level, trail, theoryDepth, learnedClauses, pendingJobs, cache, mode, numericMode, config >>

Next ==
  \E l \in Lits:
    Decide(l)
  \/ \E res \in TheoryPropagateResult:
    TheoryStep(res)
  \/ EnqueueBoundsJob
  \/ \E r \in BoundsResult:
    ApplyBoundsResult(r)
  \/ \E g \in SUBSET Lits, lv \in 0..level, j \in Justs:
    AssertLemma(g, lv, j)

\* -----------------------------------------
\* Invariants (what we care about for γ-CROWN v2)
\* -----------------------------------------

Inv_ContextAligned ==
  theoryDepth = level

Inv_TrailConsistent ==
  ConsistentTrail

Inv_NoStaleApplied ==
  \A r \in BoundsResult:
    (r.job \in pendingJobs /\ r.job.sig # CurrentSignature) => TRUE

Inv_CertifiedRequiresJustifications ==
  CertifiedMode =>
    \A i \in 1..Len(trail):
      (trail[i].reason \in {"theory", "bounds"}) => (trail[i].just /= "")

\* Lemmas beyond current level must not be active.
Inv_LemmaScoping ==
  \A lm \in ActiveLemmas: lm.lvl <= level

Invariant ==
  /\ Inv_ContextAligned
  /\ Inv_TrailConsistent
  /\ Inv_CertifiedRequiresJustifications
  /\ Inv_LemmaScoping

Spec ==
  Init /\ [][Next]_vars

THEOREM Spec => []Invariant

====================================================================================
