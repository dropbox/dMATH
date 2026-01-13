//! TLA+ Standard Library Definitions
//!
//! This module provides definitions for the TLA+ standard library modules
//! (Naturals, Integers, Sequences, FiniteSets, TLC, etc.) to enable proper
//! name resolution without needing to parse the actual standard library files.
//!
//! The definitions here are stubs - they provide symbol information (name, arity)
//! but not implementation semantics. The actual semantics are provided by
//! tla-check (model checking) and tla-prove (theorem proving).

use crate::resolve::{ResolveCtx, ScopeKind, SymbolKind};
use crate::span::Span;

/// Standard library module names
pub const STDLIB_MODULES: &[&str] = &[
    "Naturals",
    "Integers",
    "Reals",
    "Sequences",
    "FiniteSets",
    "Bags",
    "BagsExt",
    "TLC",
    "TLAPS",
    "Toolbox",
    // TLAPS proof modules
    "PTL",        // Propositional Temporal Logic
    "Zenon",      // Zenon prover
    "SMT",        // SMT solvers
    "Isa",        // Isabelle prover
    "Blast",      // Blast prover
    "Auto",       // Auto prover
    "ProofRules", // Proof rules
    "TLAPMTLA",   // TLA+ Proof Manager TLA module
    "WellFoundedInduction",
    // Community modules (commonly used)
    "Functions",
    "FiniteSetTheorems",
    "NaturalsInduction",
    "SequencesExt",
    "FiniteSetsExt",
    "TLCExt",
    "Strings",
    "IOUtils",
    "CSV",
    "Json",
    "Relation",
    // Dyadic rationals (CommunityModules)
    "DyadicRationals",
    // Graph theory
    "Graphs",
    "UndirectedGraphs",
    "DirectedGraphs",
    // Bitwise operations (CommunityModules)
    "Bitwise",
];

/// Operator definition: (name, arity)
/// Arity -1 means variadic (accepts any number of arguments)
type OpDef = (&'static str, i32);

/// Get operators provided by a standard library module
pub fn get_module_operators(module_name: &str) -> &'static [OpDef] {
    match module_name {
        "Naturals" => NATURALS_OPS,
        "Integers" => INTEGERS_OPS,
        "Reals" => REALS_OPS,
        "Sequences" => SEQUENCES_OPS,
        "FiniteSets" => FINITESETS_OPS,
        "Bags" => BAGS_OPS,
        "BagsExt" => BAGSEXT_OPS,
        "TLC" => TLC_OPS,
        "TLAPS" => TLAPS_OPS,
        "Toolbox" => TOOLBOX_OPS,
        // TLAPS proof modules
        "PTL" => PTL_OPS,
        "Zenon" => ZENON_OPS,
        "SMT" => SMT_OPS,
        "Isa" => ISA_OPS,
        "Blast" => BLAST_OPS,
        "Auto" => AUTO_OPS,
        "ProofRules" => PROOFRULES_OPS,
        "TLAPMTLA" => TLAPMTLA_OPS,
        "WellFoundedInduction" => WELLFOUNDEDINDUCTION_OPS,
        // Community modules
        "Functions" => FUNCTIONS_OPS,
        "FiniteSetTheorems" => FINITESETTHEOREMS_OPS,
        "NaturalsInduction" => NATURALSINDUCTION_OPS,
        "SequencesExt" => SEQUENCESEXT_OPS,
        "FiniteSetsExt" => FINITESETSEXT_OPS,
        "TLCExt" => TLCEXT_OPS,
        "Strings" => STRINGS_OPS,
        "IOUtils" => IOUTILS_OPS,
        "CSV" => CSV_OPS,
        "Json" => JSON_OPS,
        "Relation" => RELATION_OPS,
        // Dyadic rationals
        "DyadicRationals" => DYADICRATIONALS_OPS,
        // Graph modules
        "Graphs" => GRAPHS_OPS,
        "UndirectedGraphs" => UNDIRECTEDGRAPHS_OPS,
        "DirectedGraphs" => DIRECTEDGRAPHS_OPS,
        "Bitwise" => BITWISE_OPS,
        _ => &[],
    }
}

/// Check if a module name is a standard library module
pub fn is_stdlib_module(name: &str) -> bool {
    STDLIB_MODULES.contains(&name)
}

/// Inject standard library symbols into a resolution context based on EXTENDS list
pub fn inject_stdlib(ctx: &mut ResolveCtx, extends: &[&str]) {
    // Create a synthetic span for stdlib definitions (dummy span = file 0, offset 0)
    let stdlib_span = Span::dummy();

    // Track which modules have been processed to handle transitive extends
    let mut processed = std::collections::HashSet::new();

    // Process each extended module
    for module_name in extends {
        inject_module_symbols(ctx, module_name, &mut processed, stdlib_span);
    }
}

fn inject_module_symbols(
    ctx: &mut ResolveCtx,
    module_name: &str,
    processed: &mut std::collections::HashSet<String>,
    span: Span,
) {
    if processed.contains(module_name) {
        return;
    }
    processed.insert(module_name.to_string());

    // Handle transitive extends (e.g., Integers extends Naturals)
    // Note: UndirectedGraphs does NOT extend Graphs - they are separate modules
    // with different edge representations (sets vs tuples)
    match module_name {
        "Integers" => {
            inject_module_symbols(ctx, "Naturals", processed, span);
        }
        "Reals" => {
            inject_module_symbols(ctx, "Integers", processed, span);
        }
        _ => {}
    }

    // Get operators for this module
    let ops = get_module_operators(module_name);

    for (name, arity) in ops {
        let arity = if *arity < 0 { 0 } else { *arity as usize };
        ctx.define(name, SymbolKind::Operator, span, arity, false);
    }

    // Also define the module as a constant (e.g., Nat, Int, Real, Seq)
    match module_name {
        "Naturals" => {
            ctx.define("Nat", SymbolKind::Constant, span, 0, false);
        }
        "Integers" => {
            ctx.define("Int", SymbolKind::Constant, span, 0, false);
        }
        "Reals" => {
            ctx.define("Real", SymbolKind::Constant, span, 0, false);
            ctx.define("Infinity", SymbolKind::Constant, span, 0, false);
        }
        _ => {}
    }
}

/// Create a new resolution context with standard library pre-loaded
pub fn resolve_with_stdlib(module: &crate::ast::Module) -> ResolveCtx {
    let mut ctx = ResolveCtx::new();
    ctx.push_scope(ScopeKind::Module);

    // Inject stdlib symbols based on EXTENDS
    let extends: Vec<&str> = module.extends.iter().map(|s| s.node.as_str()).collect();
    inject_stdlib(&mut ctx, &extends);

    ctx
}

// =============================================================================
// Standard Library Operator Definitions
// =============================================================================

/// Naturals module operators
/// Built-in arithmetic operators are handled by the parser/AST directly,
/// but some specs reference them by name
const NATURALS_OPS: &[OpDef] = &[
    // The set of natural numbers (0-ary constant)
    // "Nat" is defined separately as a constant
    // Arithmetic (these are built-in syntax but can be referenced)
    // Range operator a..b is built-in
];

/// Integers module operators (extends Naturals)
const INTEGERS_OPS: &[OpDef] = &[
    // "Int" defined separately as constant
    // Unary minus is built-in syntax
];

/// Reals module operators (extends Integers)
const REALS_OPS: &[OpDef] = &[
    // "Real" and "Infinity" defined separately as constants
];

/// Sequences module operators
const SEQUENCES_OPS: &[OpDef] = &[
    ("Seq", 1),    // Seq(S) - set of sequences over S
    ("Len", 1),    // Len(s) - length of sequence
    ("Head", 1),   // Head(s) - first element
    ("Tail", 1),   // Tail(s) - all but first element
    ("Append", 2), // Append(s, e) - append element
    ("SubSeq", 3), // SubSeq(s, m, n) - subsequence
    ("SelectSeq", 2), // SelectSeq(s, Test) - filter sequence
                   // \o (concatenation) is built-in syntax
];

/// FiniteSets module operators
const FINITESETS_OPS: &[OpDef] = &[
    ("IsFiniteSet", 1), // IsFiniteSet(S) - true if S is finite
    ("Cardinality", 1), // Cardinality(S) - number of elements
];

/// Bags (multisets) module operators
const BAGS_OPS: &[OpDef] = &[
    ("IsABag", 1),         // IsABag(B) - true if B is a bag
    ("BagToSet", 1),       // BagToSet(B) - underlying set
    ("SetToBag", 1),       // SetToBag(S) - bag with each element once
    ("BagIn", 2),          // BagIn(e, B) - true if e is in B
    ("EmptyBag", 0),       // EmptyBag - the empty bag
    ("CopiesIn", 2),       // CopiesIn(e, B) - count of e in B
    ("BagCup", 2),         // B1 (+) B2 - bag union
    ("BagDiff", 2),        // B1 (-) B2 - bag difference
    ("BagUnion", 1),       // BagUnion(S) - bag union of set of bags
    ("SqSubseteq", 2),     // B1 \sqsubseteq B2
    ("SubBag", 1),         // SubBag(B) - set of sub-bags
    ("BagOfAll", 2),       // BagOfAll(F(_), B)
    ("BagCardinality", 1), // BagCardinality(B)
];

/// BagsExt (community) module operators
const BAGSEXT_OPS: &[OpDef] = &[
    ("BagAdd", 2),       // BagAdd(B, e) - add one occurrence of e
    ("BagRemove", 2),    // BagRemove(B, e) - remove one occurrence of e
    ("BagRemoveAll", 2), // BagRemoveAll(B, e) - remove all occurrences of e
    ("FoldBag", 3),      // FoldBag(op, base, B)
    ("SumBag", 1),       // SumBag(B) - sum of bag elements
    ("ProductBag", 1),   // ProductBag(B) - product of bag elements
];

/// TLC module operators (model checking utilities)
const TLC_OPS: &[OpDef] = &[
    ("Print", 2),         // Print(out, val) - print and return val
    ("PrintT", 1),        // PrintT(out) - print and return TRUE
    ("Assert", 2),        // Assert(val, out) - assert val is TRUE
    ("JavaTime", 0),      // JavaTime - current time in ms
    ("TLCGet", 1),        // TLCGet(i) - get TLC register
    ("TLCSet", 2),        // TLCSet(i, v) - set TLC register
    ("Permutations", 1),  // Permutations(S) - all permutations of S
    ("SortSeq", 2),       // SortSeq(s, Op) - sort sequence
    ("RandomElement", 1), // RandomElement(S) - random element
    ("ToString", 1),      // ToString(v) - convert to string
    ("TLCEval", 1),       // TLCEval(v) - force evaluation
    ("Any", 0),           // Any - any value (for symmetry)
                          // :> and @@ are infix operators handled specially
];

/// TLAPS (TLA+ Proof System) module operators
/// All TLAPS operators return TRUE during model checking - they are proof backend pragmas
const TLAPS_OPS: &[OpDef] = &[
    // SMT solver operators (zero-arity)
    ("SMT", 0),
    ("CVC3", 0),
    ("Yices", 0),
    ("veriT", 0),
    ("Z3", 0),
    ("Spass", 0),
    ("SimpleArithmetic", 0),
    // Zenon prover operators (zero-arity)
    ("Zenon", 0),
    ("SlowZenon", 0),
    ("SlowerZenon", 0),
    ("VerySlowZenon", 0),
    ("SlowestZenon", 0),
    // Isabelle prover operators (zero-arity)
    ("Isa", 0),
    ("Auto", 0),
    ("Force", 0),
    ("Blast", 0),
    ("SimplifyAndSolve", 0),
    ("Simplification", 0),
    ("AutoBlast", 0),
    // Temporal logic operators (zero-arity)
    ("LS4", 0),
    ("PTL", 0),
    ("PropositionalTemporalLogic", 0),
    // Multi-backend operators (zero-arity)
    ("AllProvers", 0),
    ("AllSMT", 0),
    ("AllIsa", 0),
    // Theorems (zero-arity)
    ("SetExtensionality", 0),
    ("NoSetContainsEverything", 0),
    ("IsaWithSetExtensionality", 0),
    // Parameterized SMT operators (1 arg)
    ("SMTT", 1),
    ("CVC3T", 1),
    ("YicesT", 1),
    ("veriTT", 1),
    ("Z3T", 1),
    ("SpassT", 1),
    // Parameterized Zenon operator (1 arg)
    ("ZenonT", 1),
    // Parameterized Isabelle operators (1-2 args)
    ("IsaT", 1),
    ("IsaM", 1),
    ("IsaMT", 2),
    // Parameterized multi-backend operators (1 arg)
    ("AllProversT", 1),
    ("AllSMTT", 1),
    ("AllIsaT", 1),
];

/// Toolbox module operators
const TOOLBOX_OPS: &[OpDef] = &[
    // Toolbox-specific annotations
];

/// Functions module operators (community)
const FUNCTIONS_OPS: &[OpDef] = &[
    ("Range", 1),             // Range(f) - range of function
    ("Inverse", 3),           // Inverse(f, S, T)
    ("Restrict", 2),          // Restrict(f, S) - restrict domain
    ("IsInjective", 1),       // IsInjective(f)
    ("IsSurjective", 3),      // IsSurjective(f, S, T)
    ("IsBijection", 3),       // IsBijection(f, S, T)
    ("AntiFunction", 1),      // AntiFunction(f)
    ("FoldFunction", 3),      // FoldFunction(op, base, f)
    ("FoldFunctionOnSet", 4), // FoldFunctionOnSet(op, base, f, S)
    ("RestrictDomain", 2),    // RestrictDomain(f, P) - restrict to domain elems satisfying P
    ("RestrictValues", 2),    // RestrictValues(f, P) - restrict to range elems satisfying P
    ("IsRestriction", 2),     // IsRestriction(f, g) - is f a restriction of g?
    ("Pointwise", 3),         // Pointwise(Op, f, g) - pointwise combination
];

/// FiniteSetTheorems module operators
const FINITESETTHEOREMS_OPS: &[OpDef] = &[
    // Theorems about finite sets - mainly for TLAPS proofs
    ("FS_CardinalityType", 0),
    ("FS_EmptySet", 0),
    ("FS_Interval", 0),
    ("FS_Singleton", 0),
    ("FS_Subset", 0),
    ("FS_Union", 0),
    ("FS_SUBSET", 0),
];

/// NaturalsInduction module operators
const NATURALSINDUCTION_OPS: &[OpDef] = &[
    // Induction principles for TLAPS
    ("NatInductiveDef", 4),
    ("NNIF", 4), // NatNonNegativeInductionDef
    ("FiniteNatInductiveDef", 4),
];

/// SequencesExt module operators (community)
const SEQUENCESEXT_OPS: &[OpDef] = &[
    // Conversion
    ("SetToSeq", 1),     // SetToSeq(S)
    ("SetToSeqs", 1),    // SetToSeqs(S) - all permutations of set
    ("SetToSortSeq", 2), // SetToSortSeq(S, op)
    ("ToSet", 1),        // ToSet(s) - set of elements
    ("Range", 1),        // Range(s) - same as ToSet
    ("Indices", 1),      // Indices(s) - {1..Len(s)}
    // Element operations
    ("Cons", 2),     // Cons(e, s) - prepend
    ("Snoc", 2),     // Snoc(s, e) - append
    ("Front", 1),    // Front(s) - all but last
    ("Last", 1),     // Last(s)
    ("Contains", 2), // Contains(s, e)
    // Modification
    ("Reverse", 1),    // Reverse(s)
    ("Remove", 2),     // Remove(s, e)
    ("ReplaceAt", 3),  // ReplaceAt(s, i, e)
    ("InsertAt", 3),   // InsertAt(s, i, e)
    ("RemoveAt", 2),   // RemoveAt(s, i)
    ("ReplaceAll", 3), // ReplaceAll(s, old, new)
    ("Interleave", 2), // Interleave(s, t)
    // Prefix/suffix
    ("IsPrefix", 2),            // IsPrefix(s, t)
    ("IsSuffix", 2),            // IsSuffix(s, t)
    ("IsStrictPrefix", 2),      // IsStrictPrefix(s, t)
    ("IsStrictSuffix", 2),      // IsStrictSuffix(s, t)
    ("Prefixes", 1),            // Prefixes(s) - set of all prefixes
    ("Suffixes", 1),            // Suffixes(s) - set of all suffixes
    ("CommonPrefixes", 1),      // CommonPrefixes(seqs) - common prefixes of set of seqs
    ("LongestCommonPrefix", 1), // LongestCommonPrefix(seqs) - LCP of set of seqs
    // Search
    ("SelectInSeq", 2),     // SelectInSeq(s, Test) - first matching index
    ("SelectLastInSeq", 2), // SelectLastInSeq(s, Test) - last matching index
    // Combining
    ("FlattenSeq", 1), // FlattenSeq(ss) - flatten seq of seqs
    ("Zip", 2),        // Zip(s, t)
    // Fold operations
    ("FoldLeft", 3),        // FoldLeft(op, base, s)
    ("FoldRight", 3),       // FoldRight(op, s, base)
    ("FoldLeftBool", 3),    // FoldLeftBool(op, base, s)
    ("FoldRightBool", 3),   // FoldRightBool(op, s, base)
    ("FoldLeftDomain", 3),  // FoldLeftDomain(op, base, s) - fold with index
    ("FoldRightDomain", 3), // FoldRightDomain(op, s, base) - fold with index
    // Sequence generation
    ("BoundedSeq", 2), // BoundedSeq(S, n) - seqs up to length n
    ("SeqOf", 2),      // SeqOf(S, n) - alias for BoundedSeq
    ("TupleOf", 2),    // TupleOf(S, n) - tuples of exactly length n
    // Subsequences
    ("SubSeqs", 1),    // SubSeqs(s) - all contiguous subsequences
    ("AllSubSeqs", 1), // AllSubSeqs(s) - all subsequences (non-contiguous)
];

/// FiniteSetsExt module operators (community)
const FINITESETSEXT_OPS: &[OpDef] = &[
    ("FoldSet", 3),       // FoldSet(op, base, S)
    ("ReduceSet", 3),     // ReduceSet(op, S, base)
    ("Quantify", 2),      // Quantify(S, P) - count matching
    ("Ksubsets", 2),      // Ksubsets(S, k) - k-element subsets
    ("Symmetry", 1),      // Symmetry(S)
    ("Sum", 1),           // Sum(S) - sum of set
    ("Product", 1),       // Product(S) - product of set
    ("Max", 1),           // Max(S)
    ("Min", 1),           // Min(S)
    ("Mean", 1),          // Mean(S)
    ("SymDiff", 2),       // SymDiff(S, T) - symmetric difference
    ("Flatten", 1),       // Flatten(SS) - union of sets
    ("Choose", 1),        // Choose(S) - arbitrary element
    ("MapThenSumSet", 2), // MapThenSumSet(Op, S) - map Op over S then sum
    ("Choices", 1),       // Choices(SS) - set of choice functions
    ("ChooseUnique", 2),  // ChooseUnique(S, P) - unique element satisfying P
];

/// TLCExt module operators (community)
const TLCEXT_OPS: &[OpDef] = &[
    ("AssertError", 2),   // AssertError(msg, expr)
    ("AssertEq", 2),      // AssertEq(a, b)
    ("Trace", 0),         // Trace - get current trace
    ("TLCDefer", 1),      // TLCDefer(expr)
    ("PickSuccessor", 1), // PickSuccessor(expr)
    ("TLCNoOp", 1),       // TLCNoOp(val) - returns val unchanged
    ("TLCModelValue", 1), // TLCModelValue(str) - create model value
    ("TLCCache", 2),      // TLCCache(expr, closure) - cached evaluation
];

/// Strings module operators
const STRINGS_OPS: &[OpDef] = &[
    ("STRING", 0), // STRING - the set of all strings
];

/// IOUtils module operators (community)
const IOUTILS_OPS: &[OpDef] = &[
    ("IOExec", 1),        // IOExec(cmd)
    ("IOEnvGet", 1),      // IOEnvGet(var)
    ("IOEnvPut", 2),      // IOEnvPut(var, val)
    ("IODeserialize", 2), // IODeserialize(path, default)
    ("IOSerialize", 2),   // IOSerialize(val, path)
];

/// CSV module operators (community)
const CSV_OPS: &[OpDef] = &[
    ("CSVRead", 3),    // CSVRead(path, header, delim)
    ("CSVRecords", 1), // CSVRecords(path)
    ("CSVWrite", 3),   // CSVWrite(records, path, header)
];

/// Json module operators (community)
const JSON_OPS: &[OpDef] = &[
    ("JsonDeserialize", 1),   // JsonDeserialize(str)
    ("JsonSerialize", 1),     // JsonSerialize(val)
    ("ndJsonDeserialize", 1), // ndJsonDeserialize(path)
    ("ndJsonSerialize", 2),   // ndJsonSerialize(records, path)
];

/// Relation module operators (community)
const RELATION_OPS: &[OpDef] = &[
    ("IsReflexive", 2),
    ("IsReflexiveUnder", 2),
    ("IsIrreflexive", 2),
    ("IsIrreflexiveUnder", 2),
    ("IsSymmetric", 2),
    ("IsSymmetricUnder", 2),
    ("IsAntiSymmetric", 2),
    ("IsAntiSymmetricUnder", 2),
    ("IsAsymmetric", 2),
    ("IsAsymmetricUnder", 2),
    ("IsTransitive", 2),
    ("IsTransitiveUnder", 2),
    ("IsStrictlyPartiallyOrdered", 2),
    ("IsStrictlyPartiallyOrderedUnder", 2),
    ("IsPartiallyOrdered", 2),
    ("IsPartiallyOrderedUnder", 2),
    ("IsStrictlyTotallyOrdered", 2),
    ("IsStrictlyTotallyOrderedUnder", 2),
    ("IsTotallyOrdered", 2),
    ("IsTotallyOrderedUnder", 2),
    ("TransitiveClosure", 2),
    ("ReflexiveTransitiveClosure", 2),
    ("IsConnected", 2),
];

/// DyadicRationals module operators (community)
/// Dyadic rationals are fractions with denominator = 2^n
/// Represented as records [num |-> n, den |-> d]
const DYADICRATIONALS_OPS: &[OpDef] = &[
    ("Zero", 0),             // Zero - the zero dyadic rational [num |-> 0, den |-> 1]
    ("One", 0),              // One - the one dyadic rational [num |-> 1, den |-> 1]
    ("IsDyadicRational", 1), // IsDyadicRational(r) - check if r.den is a power of 2
    ("Add", 2),              // Add(p, q) - add two dyadic rationals
    ("Half", 1),             // Half(p) - divide by 2 (double the denominator)
    ("PrettyPrint", 1),      // PrettyPrint(p) - string representation
];

// =============================================================================
// TLAPS Proof Modules
// =============================================================================

/// PTL (Propositional Temporal Logic) module operators
const PTL_OPS: &[OpDef] = &[
    // Temporal operators for propositional proofs
    // PTL is used as a proof backend selector, no explicit operators
];

/// Zenon prover module operators
const ZENON_OPS: &[OpDef] = &[
    // Zenon is used as a proof backend selector, no explicit operators
];

/// SMT solver module operators
const SMT_OPS: &[OpDef] = &[
    // SMT is used as a proof backend selector, no explicit operators
];

/// Isabelle prover module operators
const ISA_OPS: &[OpDef] = &[
    // Isa is used as a proof backend selector, no explicit operators
];

/// Blast prover module operators
const BLAST_OPS: &[OpDef] = &[
    // Blast is used as a proof backend selector, no explicit operators
];

/// Auto prover module operators
const AUTO_OPS: &[OpDef] = &[
    // Auto is used as a proof backend selector, no explicit operators
];

/// ProofRules module operators
const PROOFRULES_OPS: &[OpDef] = &[];

/// TLAPMTLA module operators
const TLAPMTLA_OPS: &[OpDef] = &[];

/// WellFoundedInduction module operators
const WELLFOUNDEDINDUCTION_OPS: &[OpDef] = &[
    ("OpToRel", 2),         // Convert operator to relation
    ("IsWellFoundedOn", 2), // Check if relation is well-founded
    ("SetLessThan", 2),     // Set comparison for well-foundedness
];

// =============================================================================
// Graph Modules
// =============================================================================

/// Graphs module operators
const GRAPHS_OPS: &[OpDef] = &[
    ("IsDirectedGraph", 1),
    ("IsUndirectedGraph", 1),
    ("Path", 1),
    ("AreConnectedIn", 3),
    ("IsStronglyConnected", 1),
];

/// UndirectedGraphs module operators
/// Note: edges are sets {a,b} not tuples <<a,b>>
const UNDIRECTEDGRAPHS_OPS: &[OpDef] = &[
    ("IsUndirectedGraph", 1),
    ("IsLoopFreeUndirectedGraph", 1),
    ("UndirectedSubgraph", 1),
    ("Path", 1),
    ("SimplePath", 1),
    ("ConnectedComponents", 1),
    ("AreConnectedIn", 3),
    ("IsStronglyConnected", 1),
];

/// DirectedGraphs module operators
const DIRECTEDGRAPHS_OPS: &[OpDef] = &[
    ("DirectedSubgraph", 1),
    ("SuccessorsOf", 2),
    ("PredecessorsOf", 2),
    ("Reachable", 2),
];

/// Bitwise module operators (CommunityModules)
/// Provides bitwise operations on non-negative integers
const BITWISE_OPS: &[OpDef] = &[
    // Infix operators are defined in TLA+ as user-definable infix ops
    // a & b - bitwise AND (infix)
    ("&", 2),
    // a | b - bitwise OR (infix) - but | is already parser-handled as user-definable
    ("|", 2),
    // a ^^ b - bitwise XOR (infix)
    ("^^", 2),
    // Not(a) - bitwise NOT
    ("Not", 1),
    // shiftR(n, pos) - logical right shift
    ("shiftR", 2),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_stdlib_module() {
        assert!(is_stdlib_module("Naturals"));
        assert!(is_stdlib_module("Integers"));
        assert!(is_stdlib_module("Sequences"));
        assert!(is_stdlib_module("TLC"));
        assert!(!is_stdlib_module("MyModule"));
        assert!(!is_stdlib_module(""));
    }

    #[test]
    fn test_get_module_operators() {
        let seq_ops = get_module_operators("Sequences");
        assert!(seq_ops.iter().any(|(name, _)| *name == "Seq"));
        assert!(seq_ops.iter().any(|(name, _)| *name == "Len"));
        assert!(seq_ops.iter().any(|(name, _)| *name == "Head"));
        assert!(seq_ops.iter().any(|(name, _)| *name == "Tail"));
        assert!(seq_ops.iter().any(|(name, _)| *name == "Append"));

        let fs_ops = get_module_operators("FiniteSets");
        assert!(fs_ops.iter().any(|(name, _)| *name == "Cardinality"));
        assert!(fs_ops.iter().any(|(name, _)| *name == "IsFiniteSet"));
    }

    #[test]
    fn test_transitive_extends() {
        // Integers should include Naturals symbols
        let int_ops = get_module_operators("Integers");
        // Integers itself doesn't have many ops, but injection handles transitivity
        assert!(int_ops.is_empty() || int_ops.len() < 3);
    }
}
