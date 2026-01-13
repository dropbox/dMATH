//! Documentation and info helpers for USL language elements.
//!
//! Provides hover documentation for keywords, builtin types, and verification backends.

/// USL keywords
pub const KEYWORDS: &[&str] = &[
    "theorem",
    "temporal",
    "contract",
    "invariant",
    "refinement",
    "probabilistic",
    "security",
    "type",
    "forall",
    "exists",
    "implies",
    "and",
    "or",
    "not",
    "always",
    "eventually",
    "requires",
    "ensures",
    "ensures_err",
    "abstraction",
    "simulation",
    "probability",
    "true",
    "false",
    "refines",
    "in",
];

/// Builtin types
pub const BUILTIN_TYPES: &[&str] = &[
    "Bool", "Int", "Float", "String", "Unit", "Set", "List", "Map", "Relation", "Result",
];

/// Backend names for hover and completion
pub const BACKEND_NAMES: &[&str] = &[
    "lean4",
    "tlaplus",
    "kani",
    "alloy",
    "isabelle",
    "coq",
    "dafny",
    "marabou",
    "alphabetacrown",
    "eran",
    "storm",
    "prism",
    "tamarin",
    "proverif",
    "verifpal",
    "verus",
    "creusot",
    "prusti",
    "z3",
    "cvc5",
];

/// Get documentation for a keyword.
pub fn keyword_info(kw: &str) -> Option<&'static str> {
    match kw {
        "theorem" => Some(
            "**theorem** - Mathematical property to prove\n\n\
            Compiles to LEAN 4, Coq, Isabelle.\n\n\
            ```usl\ntheorem name {\n    forall x: Type . property\n}\n```",
        ),
        "temporal" => Some(
            "**temporal** - Temporal logic property\n\n\
            Compiles to TLA+.\n\n\
            ```usl\ntemporal name {\n    always(eventually(predicate))\n}\n```",
        ),
        "contract" => Some(
            "**contract** - Pre/post conditions for functions\n\n\
            Compiles to Kani.\n\n\
            ```usl\ncontract Type::method(self, param: Type) -> RetType {\n    requires { precondition }\n    ensures { postcondition }\n}\n```",
        ),
        "invariant" => Some(
            "**invariant** - State invariant that must always hold\n\n\
            Compiles to LEAN 4, Alloy.\n\n\
            ```usl\ninvariant name {\n    forall x: Type . property\n}\n```",
        ),
        "refinement" => Some(
            "**refinement** - Proves implementation refines specification\n\n\
            Compiles to LEAN 4.\n\n\
            ```usl\nrefinement name refines spec {\n    abstraction { ... }\n    simulation { ... }\n}\n```",
        ),
        "forall" => Some(
            "**forall** - Universal quantifier\n\n\
            `forall x: Type . P(x)` - P holds for all x of type Type",
        ),
        "exists" => Some(
            "**exists** - Existential quantifier\n\n\
            `exists x: Type . P(x)` - There exists an x of type Type where P holds",
        ),
        "implies" => Some(
            "**implies** - Logical implication\n\n\
            `A implies B` - If A is true, then B must be true",
        ),
        "always" => Some(
            "**always** - Temporal always/globally operator\n\n\
            `always(P)` - P holds in all states",
        ),
        "eventually" => Some(
            "**eventually** - Temporal eventually/finally operator\n\n\
            `eventually(P)` - P will hold in some future state",
        ),
        "requires" => Some(
            "**requires** - Precondition clause in contracts\n\n\
            Specifies conditions that must be true before function execution.",
        ),
        "ensures" => Some(
            "**ensures** - Postcondition clause in contracts\n\n\
            Specifies conditions that must be true after successful function execution.",
        ),
        "ensures_err" => Some(
            "**ensures_err** - Error postcondition in contracts\n\n\
            Specifies conditions that must hold when function returns an error.",
        ),
        "and" => Some("**and** - Logical conjunction\n\n`A and B` - Both A and B must be true"),
        "or" => Some(
            "**or** - Logical disjunction\n\n`A or B` - At least one of A or B must be true",
        ),
        "not" => Some("**not** - Logical negation\n\n`not A` - A must be false"),
        "true" => Some("**true** - Boolean literal true"),
        "false" => Some("**false** - Boolean literal false"),
        "type" => Some(
            "**type** - Type definition\n\n\
            ```usl\ntype Name = { field1: Type1, field2: Type2 }\n```",
        ),
        _ => None,
    }
}

/// Get documentation for a builtin type.
pub fn builtin_type_info(ty: &str) -> Option<&'static str> {
    match ty {
        "Bool" => Some("**Bool** - Boolean type\n\nValues: `true`, `false`"),
        "Int" => Some("**Int** - Integer type\n\nArbitrary precision integers"),
        "Float" => Some("**Float** - Floating point type\n\n64-bit IEEE 754 floating point"),
        "String" => Some("**String** - String type\n\nUTF-8 encoded text"),
        "Unit" => Some("**Unit** - Unit type\n\nEmpty type with single value ()"),
        "Set" => Some("**Set<T>** - Set type\n\nUnordered collection of unique elements of type T"),
        "List" => Some("**List<T>** - List type\n\nOrdered sequence of elements of type T"),
        "Map" => Some(
            "**Map<K, V>** - Map type\n\nKey-value mapping from keys of type K to values of type V",
        ),
        "Relation" => Some(
            "**Relation<A, B>** - Relation type\n\nBinary relation between elements of type A and B",
        ),
        "Result" => Some("**Result<T>** - Result type\n\nSuccess with value of type T, or error"),
        _ => None,
    }
}

/// Get documentation for a verification backend.
pub fn backend_info(name: &str) -> Option<&'static str> {
    match name.to_lowercase().as_str() {
        "lean4" | "lean" => Some(
            "**Lean 4** - Interactive theorem prover\n\n\
            Best for: Mathematical proofs, functional correctness\n\n\
            Features:\n\
            - Tactic-based proofs\n\
            - Mathlib library for mathematics\n\
            - Metaprogramming support\n\n\
            Use with: `theorem`, `invariant`, `refinement`",
        ),
        "tlaplus" | "tla+" | "tla" => Some(
            "**TLA+** - Temporal Logic of Actions\n\n\
            Best for: Distributed systems, concurrent algorithms\n\n\
            Features:\n\
            - State machine modeling\n\
            - Temporal logic specifications\n\
            - TLC model checker\n\n\
            Use with: `temporal`, `invariant`",
        ),
        "kani" => Some(
            "**Kani** - Rust verification via bounded model checking\n\n\
            Best for: Rust code memory safety, panics\n\n\
            Features:\n\
            - Symbolic execution\n\
            - Contract verification with `#[kani::requires]`, `#[kani::ensures]`\n\
            - Automatic harness generation\n\n\
            Use with: `contract`",
        ),
        "alloy" => Some(
            "**Alloy** - Relational modeling language\n\n\
            Best for: Design exploration, constraint analysis\n\n\
            Features:\n\
            - SAT-based analysis\n\
            - Instance finding\n\
            - Bounded verification\n\n\
            Use with: `invariant`",
        ),
        "coq" => Some(
            "**Coq** - Proof assistant\n\n\
            Best for: Strong formal guarantees, program extraction\n\n\
            Features:\n\
            - Dependent types\n\
            - Extraction to OCaml/Haskell\n\
            - Large standard library\n\n\
            Use with: `theorem`",
        ),
        "isabelle" => Some(
            "**Isabelle/HOL** - Proof assistant\n\n\
            Best for: Mathematical proofs, AFP library use\n\n\
            Features:\n\
            - Higher-order logic\n\
            - Archive of Formal Proofs\n\
            - Sledgehammer automation\n\n\
            Use with: `theorem`",
        ),
        "dafny" => Some(
            "**Dafny** - Auto-active verification language\n\n\
            Best for: Program verification with automation\n\n\
            Features:\n\
            - Automatic verification\n\
            - Built-in specifications\n\
            - Code generation\n\n\
            Use with: `contract`, `invariant`",
        ),
        "verus" => Some(
            "**Verus** - High-performance Rust verification\n\n\
            Best for: Systems Rust code verification\n\n\
            Features:\n\
            - SMT-based verification\n\
            - Linear types support\n\
            - Low-level memory reasoning\n\n\
            Use with: `contract`",
        ),
        "creusot" => Some(
            "**Creusot** - Rust verification via Why3\n\n\
            Best for: Rust functional correctness\n\n\
            Features:\n\
            - Why3 backend\n\
            - Prophecy-based approach\n\
            - Interior mutability support\n\n\
            Use with: `contract`",
        ),
        "prusti" => Some(
            "**Prusti** - Rust verification via Viper\n\n\
            Best for: Rust memory safety with separation logic\n\n\
            Features:\n\
            - Viper intermediate language\n\
            - Automatic inference\n\
            - Pledge system\n\n\
            Use with: `contract`",
        ),
        "z3" => Some(
            "**Z3** - SMT solver\n\n\
            Best for: Satisfiability, constraint solving\n\n\
            Features:\n\
            - Multiple theories (arithmetic, arrays, bitvectors)\n\
            - Optimization\n\
            - Proof generation\n\n\
            Use with: `theorem` (simple properties)",
        ),
        "cvc5" => Some(
            "**CVC5** - SMT solver\n\n\
            Best for: Theory-heavy SMT problems\n\n\
            Features:\n\
            - Strong string theory\n\
            - Finite model finding\n\
            - Proof production\n\n\
            Use with: `theorem` (simple properties)",
        ),
        "marabou" => Some(
            "**Marabou** - Neural network verifier\n\n\
            Best for: DNN robustness verification\n\n\
            Features:\n\
            - ReLU network support\n\
            - Local robustness\n\
            - Reachability analysis\n\n\
            Use with: Neural network properties",
        ),
        "alphabetacrown" | "crown" => Some(
            "**α,β-CROWN** - Neural network verifier\n\n\
            Best for: Scalable neural network verification\n\n\
            Features:\n\
            - Linear bound propagation\n\
            - Branch and bound\n\
            - GPU acceleration\n\n\
            Use with: Neural network properties",
        ),
        "eran" => Some(
            "**ERAN** - Neural network analyzer\n\n\
            Best for: Abstract interpretation for DNNs\n\n\
            Features:\n\
            - Multiple abstract domains\n\
            - Certified defenses\n\
            - Adversarial robustness\n\n\
            Use with: Neural network properties",
        ),
        "storm" => Some(
            "**Storm** - Probabilistic model checker\n\n\
            Best for: Markov chains, MDPs\n\n\
            Features:\n\
            - PCTL/CSL model checking\n\
            - Parametric analysis\n\
            - Counterexample generation\n\n\
            Use with: `probabilistic`",
        ),
        "prism" => Some(
            "**PRISM** - Probabilistic model checker\n\n\
            Best for: Stochastic systems\n\n\
            Features:\n\
            - DTMCs, CTMCs, MDPs\n\
            - Reward properties\n\
            - Statistical model checking\n\n\
            Use with: `probabilistic`",
        ),
        "tamarin" => Some(
            "**Tamarin** - Security protocol verifier\n\n\
            Best for: Cryptographic protocol analysis\n\n\
            Features:\n\
            - Symbolic Dolev-Yao model\n\
            - Equational theories\n\
            - Observational equivalence\n\n\
            Use with: `security`",
        ),
        "proverif" => Some(
            "**ProVerif** - Security protocol verifier\n\n\
            Best for: Automated protocol verification\n\n\
            Features:\n\
            - Applied pi calculus\n\
            - Unbounded sessions\n\
            - Secrecy and authentication\n\n\
            Use with: `security`",
        ),
        "verifpal" => Some(
            "**Verifpal** - Security protocol verifier\n\n\
            Best for: Accessible protocol analysis\n\n\
            Features:\n\
            - Simple modeling language\n\
            - Active/passive attackers\n\
            - Freshness analysis\n\n\
            Use with: `security`",
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_info_all_keywords() {
        // Test all keyword match arms have distinct content
        let theorem = keyword_info("theorem").unwrap();
        let temporal = keyword_info("temporal").unwrap();
        let contract = keyword_info("contract").unwrap();
        let invariant = keyword_info("invariant").unwrap();
        let refinement = keyword_info("refinement").unwrap();
        let forall = keyword_info("forall").unwrap();
        let exists = keyword_info("exists").unwrap();
        let implies = keyword_info("implies").unwrap();
        let always = keyword_info("always").unwrap();
        let eventually = keyword_info("eventually").unwrap();
        let requires = keyword_info("requires").unwrap();
        let ensures = keyword_info("ensures").unwrap();
        let ensures_err = keyword_info("ensures_err").unwrap();
        let and = keyword_info("and").unwrap();
        let or = keyword_info("or").unwrap();
        let not = keyword_info("not").unwrap();
        let true_kw = keyword_info("true").unwrap();
        let false_kw = keyword_info("false").unwrap();
        let type_kw = keyword_info("type").unwrap();

        // Verify each has specific expected content
        assert!(theorem.contains("theorem"));
        assert!(temporal.contains("temporal"));
        assert!(contract.contains("contract"));
        assert!(invariant.contains("invariant"));
        assert!(refinement.contains("refinement"));
        assert!(forall.contains("forall"));
        assert!(exists.contains("exists"));
        assert!(implies.contains("implies"));
        assert!(always.contains("always"));
        assert!(eventually.contains("eventually"));
        assert!(requires.contains("requires"));
        assert!(ensures.contains("ensures"));
        assert!(ensures_err.contains("ensures_err"));
        assert!(and.contains("and"));
        assert!(or.contains("or"));
        assert!(not.contains("not"));
        assert!(true_kw.contains("true"));
        assert!(false_kw.contains("false"));
        assert!(type_kw.contains("type"));

        // Verify unknown returns None
        assert!(keyword_info("unknown_keyword").is_none());
    }

    #[test]
    fn test_keyword_info_distinct_content() {
        // Mutation test: ensure each arm returns distinct content
        let temporal = keyword_info("temporal").unwrap();
        let contract = keyword_info("contract").unwrap();
        let invariant = keyword_info("invariant").unwrap();
        let refinement = keyword_info("refinement").unwrap();

        // Each should have its keyword in the content
        assert!(temporal.contains("TLA+"));
        assert!(contract.contains("Pre/post"));
        assert!(invariant.contains("State invariant"));
        assert!(refinement.contains("implementation refines"));
    }

    #[test]
    fn test_keyword_info_operators() {
        // Mutation test: ensure operator keywords have distinct content
        let exists = keyword_info("exists").unwrap();
        let implies = keyword_info("implies").unwrap();
        let always = keyword_info("always").unwrap();
        let eventually = keyword_info("eventually").unwrap();

        assert!(exists.contains("Existential"));
        assert!(implies.contains("implication"));
        assert!(always.contains("all states"));
        assert!(eventually.contains("future state"));
    }

    #[test]
    fn test_keyword_info_contract_clauses() {
        // Mutation test: ensure contract clause keywords are distinct
        let requires = keyword_info("requires").unwrap();
        let ensures = keyword_info("ensures").unwrap();
        let ensures_err = keyword_info("ensures_err").unwrap();

        assert!(requires.contains("Precondition"));
        assert!(ensures.contains("Postcondition"));
        assert!(ensures_err.contains("Error postcondition"));
    }

    #[test]
    fn test_keyword_info_logical_ops() {
        // Mutation test: ensure logical operators have distinct content
        let and = keyword_info("and").unwrap();
        let or = keyword_info("or").unwrap();
        let not = keyword_info("not").unwrap();
        let true_kw = keyword_info("true").unwrap();
        let false_kw = keyword_info("false").unwrap();
        let type_kw = keyword_info("type").unwrap();

        assert!(and.contains("conjunction"));
        assert!(or.contains("disjunction"));
        assert!(not.contains("negation"));
        assert!(true_kw.contains("Boolean literal true"));
        assert!(false_kw.contains("Boolean literal false"));
        assert!(type_kw.contains("Type definition"));
    }

    #[test]
    fn test_builtin_type_info_all_types() {
        // Test all builtin type match arms have distinct content
        let bool_info = builtin_type_info("Bool").unwrap();
        let int_info = builtin_type_info("Int").unwrap();
        let float_info = builtin_type_info("Float").unwrap();
        let string_info = builtin_type_info("String").unwrap();
        let unit_info = builtin_type_info("Unit").unwrap();
        let set_info = builtin_type_info("Set").unwrap();
        let list_info = builtin_type_info("List").unwrap();
        let map_info = builtin_type_info("Map").unwrap();
        let relation_info = builtin_type_info("Relation").unwrap();
        let result_info = builtin_type_info("Result").unwrap();

        // Verify each has specific expected content
        assert!(bool_info.contains("Bool"));
        assert!(int_info.contains("Int"));
        assert!(float_info.contains("Float"));
        assert!(string_info.contains("String"));
        assert!(unit_info.contains("Unit"));
        assert!(set_info.contains("Set"));
        assert!(list_info.contains("List"));
        assert!(map_info.contains("Map"));
        assert!(relation_info.contains("Relation"));
        assert!(result_info.contains("Result"));

        // Verify unknown returns None
        assert!(builtin_type_info("CustomType").is_none());
    }

    #[test]
    fn test_builtin_type_info_distinct_descriptions() {
        // Mutation test: ensure each type has distinct description
        let int_info = builtin_type_info("Int").unwrap();
        let float_info = builtin_type_info("Float").unwrap();
        let string_info = builtin_type_info("String").unwrap();
        let unit_info = builtin_type_info("Unit").unwrap();
        let list_info = builtin_type_info("List").unwrap();
        let map_info = builtin_type_info("Map").unwrap();
        let relation_info = builtin_type_info("Relation").unwrap();
        let result_info = builtin_type_info("Result").unwrap();

        assert!(int_info.contains("Integer") || int_info.contains("integer"));
        assert!(float_info.contains("floating point") || float_info.contains("Floating"));
        assert!(string_info.contains("UTF-8"));
        assert!(unit_info.contains("Empty type"));
        assert!(list_info.contains("Ordered sequence"));
        assert!(map_info.contains("Key-value"));
        assert!(relation_info.contains("Binary relation"));
        assert!(result_info.contains("Success") || result_info.contains("error"));
    }

    #[test]
    fn test_backend_info_all_backends() {
        // Test all backend match arms have distinct content
        let lean4 = backend_info("lean4").unwrap();
        let tlaplus = backend_info("tlaplus").unwrap();
        let kani = backend_info("kani").unwrap();
        let alloy = backend_info("alloy").unwrap();
        let coq = backend_info("coq").unwrap();
        let isabelle = backend_info("isabelle").unwrap();
        let dafny = backend_info("dafny").unwrap();
        let verus = backend_info("verus").unwrap();
        let creusot = backend_info("creusot").unwrap();
        let prusti = backend_info("prusti").unwrap();
        let z3 = backend_info("z3").unwrap();
        let cvc5 = backend_info("cvc5").unwrap();
        let marabou = backend_info("marabou").unwrap();
        let crown = backend_info("alphabetacrown").unwrap();
        let eran = backend_info("eran").unwrap();
        let storm = backend_info("storm").unwrap();
        let prism = backend_info("prism").unwrap();
        let tamarin = backend_info("tamarin").unwrap();
        let proverif = backend_info("proverif").unwrap();
        let verifpal = backend_info("verifpal").unwrap();

        // Verify each has expected content
        assert!(lean4.contains("Lean 4"));
        assert!(tlaplus.contains("TLA+"));
        assert!(kani.contains("Kani"));
        assert!(alloy.contains("Alloy"));
        assert!(coq.contains("Coq"));
        assert!(isabelle.contains("Isabelle"));
        assert!(dafny.contains("Dafny"));
        assert!(verus.contains("Verus"));
        assert!(creusot.contains("Creusot"));
        assert!(prusti.contains("Prusti"));
        assert!(z3.contains("Z3"));
        assert!(cvc5.contains("CVC5"));
        assert!(marabou.contains("Marabou"));
        assert!(crown.contains("CROWN"));
        assert!(eran.contains("ERAN"));
        assert!(storm.contains("Storm"));
        assert!(prism.contains("PRISM"));
        assert!(tamarin.contains("Tamarin"));
        assert!(proverif.contains("ProVerif"));
        assert!(verifpal.contains("Verifpal"));

        // Test unknown
        assert!(backend_info("unknown_backend").is_none());
    }

    #[test]
    fn test_backend_info_distinct_features() {
        // Mutation test: ensure backends have distinct feature descriptions
        let dafny = backend_info("dafny").unwrap();
        let verus = backend_info("verus").unwrap();
        let creusot = backend_info("creusot").unwrap();
        let prusti = backend_info("prusti").unwrap();

        assert!(dafny.contains("Auto-active"));
        assert!(verus.contains("High-performance"));
        assert!(creusot.contains("Why3"));
        assert!(prusti.contains("Viper"));
    }

    #[test]
    fn test_backend_info_neural_network_verifiers() {
        // Mutation test: ensure neural network verifiers are distinct
        let marabou = backend_info("marabou").unwrap();
        let eran = backend_info("eran").unwrap();

        assert!(marabou.contains("ReLU"));
        assert!(eran.contains("Abstract interpretation"));
    }

    #[test]
    fn test_backend_info_probabilistic() {
        // Mutation test: ensure probabilistic checkers are distinct
        let storm = backend_info("storm").unwrap();
        let prism = backend_info("prism").unwrap();

        assert!(storm.contains("Markov"));
        assert!(prism.contains("Stochastic") || prism.contains("DTMCs"));
    }

    #[test]
    fn test_backend_info_security_protocol() {
        // Mutation test: ensure security protocol verifiers are distinct
        let tamarin = backend_info("tamarin").unwrap();
        let proverif = backend_info("proverif").unwrap();
        let verifpal = backend_info("verifpal").unwrap();

        assert!(tamarin.contains("Dolev-Yao"));
        assert!(proverif.contains("pi calculus") || proverif.contains("Applied pi"));
        assert!(verifpal.contains("Accessible"));
    }

    #[test]
    fn test_backend_info_case_insensitive() {
        // Test case-insensitivity
        assert!(backend_info("Lean4").is_some());
        assert!(backend_info("LEAN4").is_some());
        assert!(backend_info("lean").is_some());
        assert!(backend_info("tla+").is_some());
        assert!(backend_info("crown").is_some());
    }

    #[test]
    fn test_backend_names_constant() {
        // Verify BACKEND_NAMES contains expected backends
        assert!(BACKEND_NAMES.contains(&"lean4"));
        assert!(BACKEND_NAMES.contains(&"kani"));
        assert!(BACKEND_NAMES.contains(&"tlaplus"));
        assert!(BACKEND_NAMES.contains(&"z3"));
        // Should have 20 backends
        assert_eq!(BACKEND_NAMES.len(), 20);
    }

    #[test]
    fn test_keywords_constant() {
        assert!(KEYWORDS.contains(&"theorem"));
        assert!(KEYWORDS.contains(&"contract"));
        assert!(KEYWORDS.contains(&"forall"));
        assert!(KEYWORDS.contains(&"refines"));
    }

    #[test]
    fn test_builtin_types_constant() {
        assert!(BUILTIN_TYPES.contains(&"Bool"));
        assert!(BUILTIN_TYPES.contains(&"Int"));
        assert!(BUILTIN_TYPES.contains(&"Set"));
        assert!(BUILTIN_TYPES.contains(&"Result"));
    }
}
