[ The Rocq Prover ][1]

* [Introduction and Contents][2]

Specification language

* [Core language][3]
  
  * [Basic notions and conventions][4]
    
    * [Syntax and lexical conventions][5]
      
      * [Syntax conventions][6]
      * [Lexical conventions][7]
    * [Essential vocabulary][8]
    * [Settings][9]
      
      * [Attributes][10]
      * [Flags, Options and Tables][11]
  * [Sorts][12]
  * [Functions and assumptions][13]
    
    * [Binders][14]
    * [Functions (fun) and function types (forall)][15]
    * [Function application][16]
    * [Assumptions][17]
  * [Definitions][18]
    
    * [Let-in definitions][19]
    * [Type cast][20]
    * [Top-level definitions][21]
    * [Assertions and proofs][22]
  * [Conversion rules][23]
    
    * [α-conversion][24]
    * [β-reduction][25]
    * [δ-reduction][26]
    * [ι-reduction][27]
    * [ζ-reduction][28]
    * [η-expansion][29]
    * [Examples][30]
    * [Proof Irrelevance][31]
    * [Convertibility][32]
  * [Typing rules][33]
    
    * [The terms][34]
    * [Typing rules][35]
    * [Subtyping rules][36]
    * [The Calculus of Inductive Constructions with impredicative Set][37]
  * [Variants and the `match` construct][38]
    
    * [Variants][39]
      
      * [Private (matching) inductive types][40]
    * [Definition by cases: match][41]
  * [Record types][42]
    
    * [Defining record types][43]
    * [Constructing records][44]
    * [Accessing fields (projections)][45]
    * [Settings for printing records][46]
    * [Primitive Projections][47]
      
      * [Reduction][48]
      * [Compatibility Constants for Projections][49]
  * [Inductive types and recursive functions][50]
    
    * [Inductive types][51]
      
      * [Simple inductive types][52]
      * [Simple indexed inductive types][53]
      * [Parameterized inductive types][54]
      * [Mutually defined inductive types][55]
    * [Recursive functions: fix][56]
    * [Top-level recursive functions][57]
    * [Theory of inductive definitions][58]
      
      * [Types of inductive objects][59]
      * [Well-formed inductive definitions][60]
      * [Destructors][61]
      * [Fixpoint definitions][62]
  * [Coinductive types and corecursive functions][63]
    
    * [Coinductive types][64]
      
      * [Caveat][65]
    * [Co-recursive functions: cofix][66]
    * [Top-level definitions of corecursive functions][67]
  * [Sections][68]
    
    * [Using sections][69]
    * [Summary of locality attributes in a section][70]
    * [Typing rules used at the end of a section][71]
  * [The Module System][72]
    
    * [Modules and module types][73]
    * [Using modules][74]
      
      * [Examples][75]
    * [Qualified names][76]
    * [Controlling the scope of commands with locality attributes][77]
    * [Summary of locality attributes in a module][78]
    * [Typing Modules][79]
  * [Primitive objects][80]
    
    * [Primitive Integers][81]
    * [Primitive Floats][82]
    * [Primitive Arrays][83]
    * [Primitive (Byte-Based) Strings][84]
  * [Polymorphic Universes][85]
    
    * [General Presentation][86]
    * [Polymorphic, Monomorphic][87]
    * [Cumulative, NonCumulative][88]
      
      * [Specifying cumulativity][89]
      * [Cumulativity Weak Constraints][90]
    * [Global and local universes][91]
    * [Conversion and unification][92]
    * [Minimization][93]
    * [Explicit Universes][94]
    * [Printing universes][95]
      
      * [Polymorphic definitions][96]
    * [Sort polymorphism][97]
    * [Explicit Sorts][98]
    * [Universe polymorphism and sections][99]
  * [SProp (proof irrelevant propositions)][100]
    
    * [Basic constructs][101]
    * [Encodings for strict propositions][102]
    * [Definitional UIP][103]
      
      * [Non Termination with UIP][104]
    * [Debugging \(\SProp\) issues][105]
  * [User-defined rewrite rules][106]
    
    * [Symbols][107]
    * [Rewrite rules][108]
    * [Pattern syntax][109]
    * [Higher-order pattern holes][110]
    * [Universe polymorphic rules][111]
    * [Rewrite rules, type preservation, confluence and termination][112]
    * [Compatibility with the eta laws][113]
    * [Level of support][114]
* [Language extensions][115]
  
  * [Command level processing][116]
    
    * [Lexing][117]
    * [Parsing][118]
    * [Synterp][119]
    * [Interp][120]
  * [Term level processing][121]
  * [Existential variables][122]
    
    * [Inferable subterms][123]
    * [e* tactics that can create existential variables][124]
    * [Automatic resolution of existential variables][125]
    * [Explicit display of existential instances for pretty-printing][126]
    * [Solving existential variables using tactics][127]
  * [Implicit arguments][128]
    
    * [The different kinds of implicit arguments][129]
      
      * [Implicit arguments inferable from the knowledge of other arguments of a
        function][130]
      * [Implicit arguments inferable by resolution][131]
    * [Maximal and non-maximal insertion of implicit arguments][132]
      
      * [Trailing Implicit Arguments][133]
    * [Casual use of implicit arguments][134]
    * [Declaration of implicit arguments][135]
      
      * [Implicit Argument Binders][136]
      * [Mode for automatic declaration of implicit arguments][137]
      * [Controlling strict implicit arguments][138]
      * [Controlling contextual implicit arguments][139]
      * [Controlling reversible-pattern implicit arguments][140]
      * [Controlling the insertion of implicit arguments not followed by
        explicit arguments][141]
      * [Combining manual declaration and automatic declaration][142]
    * [Explicit applications][143]
    * [Displaying implicit arguments][144]
    * [Displaying implicit arguments when pretty-printing][145]
    * [Interaction with subtyping][146]
    * [Deactivation of implicit arguments for parsing][147]
    * [Implicit types of variables][148]
    * [Implicit generalization][149]
  * [Extended pattern matching][150]
    
    * [Variants and extensions of `match`][151]
      
      * [Multiple and nested pattern matching][152]
      * [Pattern-matching on boolean values: the if expression][153]
      * [Irrefutable patterns: the destructuring let variants][154]
      * [Controlling pretty-printing of match expressions][155]
      * [Conventions about unused pattern-matching variables][156]
    * [Patterns][157]
    * [Multiple patterns][158]
    * [Aliasing subpatterns][159]
    * [Nested patterns][160]
    * [Disjunctive patterns][161]
    * [About patterns of parametric types][162]
      
      * [Parameters in patterns][163]
    * [Implicit arguments in patterns][164]
    * [Matching objects of dependent types][165]
    * [Understanding dependencies in patterns][166]
    * [When the elimination predicate must be provided][167]
      
      * [Dependent pattern matching][168]
      * [Multiple dependent pattern matching][169]
      * [Patterns in `in`][170]
    * [Using pattern matching to write proofs][171]
    * [Pattern-matching on inductive objects involving local definitions][172]
    * [Pattern-matching and coercions][173]
    * [When does the expansion strategy fail?][174]
  * [Syntax extensions and notation scopes][175]
    
    * [Notations][176]
      
      * [Basic notations][177]
      * [Precedences and associativity][178]
      * [Complex notations][179]
      * [Simple factorization rules][180]
      * [Use of notations for printing][181]
      * [The Infix command][182]
      * [Reserving notations][183]
      * [Simultaneous definition of terms and notations][184]
      * [Enabling and disabling notations][185]
      * [Displaying information about notations][186]
      * [Locating notations][187]
      * [Inheritance of the properties of arguments of constants bound to a
        notation][188]
      * [Notations and binders][189]
      * [Notations with recursive patterns][190]
      * [Notations with recursive patterns involving binders][191]
      * [Predefined entries][192]
      * [Custom entries][193]
      * [Syntax][194]
    * [Notation scopes][195]
      
      * [Global interpretation rules for notations][196]
      * [Local interpretation rules for notations][197]
      * [The `type_scope` notation scope][198]
      * [The `function_scope` notation scope][199]
      * [Notation scopes used in the standard library of Rocq][200]
      * [Displaying information about scopes][201]
    * [Abbreviations][202]
    * [Numbers and strings][203]
      
      * [Number notations][204]
      * [String notations][205]
    * [Tactic Notations][206]
  * [Setting properties of a function's arguments][207]
    
    * [Manual declaration of implicit arguments][208]
    * [Automatic declaration of implicit arguments][209]
    * [Renaming implicit arguments][210]
    * [Binding arguments to scopes][211]
    * [Effects of `Arguments` on unfolding][212]
    * [Bidirectionality hints][213]
  * [Implicit Coercions][214]
    
    * [General Presentation][215]
    * [Coercion Classes][216]
    * [Coercions][217]
    * [Reversible Coercions][218]
    * [Identity Coercions][219]
    * [Inheritance Graph][220]
    * [Coercion Classes][221]
    * [Displaying Available Coercions][222]
    * [Activating the Printing of Coercions][223]
    * [Classes as Records][224]
    * [Coercions and Sections][225]
    * [Coercions and Modules][226]
    * [Examples][227]
  * [Typeclasses][228]
    
    * [Typeclass and instance declarations][229]
    * [Binding typeclasses][230]
    * [Parameterized instances][231]
    * [Sections and contexts][232]
    * [Building hierarchies][233]
      
      * [Superclasses][234]
      * [Substructures][235]
    * [Command summary][236]
      
      * [Typeclasses Transparent, Typeclasses Opaque][237]
      * [Settings][238]
      * [Typeclasses eauto][239]
  * [Canonical Structures][240]
    
    * [Declaration of canonical structures][241]
    * [Notation overloading][242]
      
      * [Derived Canonical Structures][243]
    * [Hierarchy of structures][244]
      
      * [Compact declaration of Canonical Structures][245]
  * [Program][246]
    
    * [Elaborating programs][247]
      
      * [Syntactic control over equalities][248]
      * [Program Definition][249]
      * [Program Fixpoint][250]
      * [Program Lemma][251]
    * [Solving obligations][252]
    * [Frequently Asked Questions][253]
  * [Commands][254]
    
    * [Displaying][255]
    * [Query commands][256]
    * [Requests to the environment][257]
    * [Printing flags][258]
    * [Loading files][259]
    * [Compiled files][260]
    * [Load paths][261]
    * [Extra Dependencies][262]
    * [Backtracking][263]
    * [Quitting and debugging][264]
    * [Controlling display][265]
    * [Printing constructions in full][266]
    * [Controlling Typing Flags][267]
    * [Internal registration commands][268]
      
      * [Exposing constants to OCaml libraries][269]
      * [Inlining hints for the fast reduction machines][270]
      * [Registering primitive operations][271]

Proofs

* [Basic proof writing][272]
  
  * [Proof mode][273]
    
    * [Proof State][274]
    * [Entering and exiting proof mode][275]
      
      * [Proof using options][276]
      * [Name a set of section hypotheses for `Proof using`][277]
    * [Proof modes][278]
    * [Managing goals][279]
      
      * [Focusing goals][280]
      * [Shelving goals][281]
      * [Reordering goals][282]
    * [Proving a subgoal as a separate lemma: abstract][283]
    * [Requesting information][284]
    * [Showing differences between proof steps][285]
      
      * [How to enable diffs][286]
      * [How diffs are calculated][287]
      * ["Show Proof" differences][288]
    * [Delaying solving unification constraints][289]
    * [Proof maintenance][290]
    * [Controlling proof mode][291]
    * [Controlling memory usage][292]
  * [Tactics][293]
    
    * [Common elements of tactics][294]
      
      * [Reserved keywords][295]
      * [Invocation of tactics][296]
      * [Bindings][297]
      * [Intro patterns][298]
      * [Occurrence clauses][299]
      * [Automatic clearing of hypotheses][300]
    * [Applying theorems][301]
    * [Managing the local context][302]
    * [Controlling the proof flow][303]
    * [Classical tactics][304]
    * [Performance-oriented tactic variants][305]
  * [Reasoning with equalities][306]
    
    * [Tactics for simple equalities][307]
    * [Rewriting with Leibniz and setoid equality][308]
    * [Rewriting with definitional equality][309]
    * [Applying conversion rules][310]
      
      * [Fast reduction tactics: vm_compute and native_compute][311]
      * [Computing in a term: eval and Eval][312]
    * [Controlling reduction strategies and the conversion algorithm][313]
  * [Reasoning with inductive types][314]
    
    * [Applying constructors][315]
    * [Case analysis][316]
    * [Induction][317]
    * [Equality of inductive types][318]
      
      * [Helper tactics][319]
    * [Generation of induction principles with `Scheme`][320]
      
      * [Automatic declaration of schemes][321]
      * [Combined Scheme][322]
    * [Generation of inversion principles with `Derive` `Inversion`][323]
    * [Examples of `dependent destruction` / `dependent induction`][324]
      
      * [A larger example][325]
  * [The SSReflect proof language][326]
    
    * [Introduction][327]
      
      * [Acknowledgments][328]
    * [Usage][329]
      
      * [Getting started][330]
      * [Compatibility issues][331]
    * [Gallina extensions][332]
      
      * [Pattern assignment][333]
      * [Pattern conditional][334]
      * [Parametric polymorphism][335]
      * [Anonymous arguments][336]
      * [Wildcards][337]
      * [Definitions][338]
      * [Abbreviations][339]
      * [Basic localization][340]
    * [Basic tactics][341]
      
      * [Bookkeeping][342]
      * [The defective tactics][343]
      * [Discharge][344]
      * [Introduction in the context][345]
      * [Generation of equations][346]
      * [Type families][347]
    * [Control flow][348]
      
      * [Indentation and bullets][349]
      * [Terminators][350]
      * [Selectors][351]
      * [Iteration][352]
      * [Localization][353]
      * [Structure][354]
    * [Rewriting][355]
      
      * [An extended rewrite tactic][356]
      * [Remarks and examples][357]
      * [Rewriting under binders][358]
      * [Locking, unlocking][359]
      * [Congruence][360]
    * [Contextual patterns][361]
      
      * [Syntax][362]
      * [Matching contextual patterns][363]
      * [Examples][364]
      * [Patterns for recurrent contexts][365]
    * [Views and reflection][366]
      
      * [Interpreting eliminations][367]
      * [Interpreting assumptions][368]
      * [Interpreting goals][369]
      * [Boolean reflection][370]
      * [The reflect predicate][371]
      * [General mechanism for interpreting goals and assumptions][372]
      * [Interpreting equivalences][373]
      * [Declaring new Hint Views][374]
      * [Multiple views][375]
      * [Additional view shortcuts][376]
    * [Synopsis and Index][377]
      
      * [Parameters][378]
      * [Items and switches][379]
      * [Tactics][380]
      * [Tacticals][381]
      * [Commands][382]
      * [Settings][383]
* [Automatic solvers and programmable tactics][384]
  
  * [Solvers for logic and equality][385]
  * [Micromega: solvers for arithmetic goals over ordered rings][386]
    
    * [Short description of the tactics][387]
    * [*Positivstellensatz* refutations][388]
    * [`lra`: a decision procedure for linear real and rational arithmetic][389]
    * [`lia`: a tactic for linear integer arithmetic][390]
      
      * [High level view of `lia`][391]
      * [Cutting plane proofs][392]
      * [Case split][393]
    * [`nra`: a proof procedure for non-linear arithmetic][394]
    * [`nia`: a proof procedure for non-linear integer arithmetic][395]
    * [`psatz`: a proof procedure for non-linear arithmetic][396]
    * [`zify`: pre-processing of arithmetic goals][397]
  * [ring and field: solvers for polynomial and rational equations][398]
    
    * [What does this tactic do?][399]
    * [The variables map][400]
    * [Is it automatic?][401]
    * [Concrete usage][402]
    * [Adding a ring structure][403]
    * [How does it work?][404]
    * [Dealing with fields][405]
    * [Adding a new field structure][406]
    * [History of ring][407]
    * [Discussion][408]
  * [Nsatz: a solver for equalities in integral domains][409]
    
    * [More about `nsatz`][410]
  * [Programmable proof search][411]
    
    * [Tactics][412]
    * [Hint databases][413]
      
      * [Creating hint databases][414]
      * [Hint databases defined in the Rocq standard library][415]
    * [Creating Hints][416]
      
      * [Hint locality][417]
    * [Setting implicit automation tactics][418]
  * [Generalized rewriting][419]
    
    * [Introduction to generalized rewriting][420]
      
      * [Relations and morphisms][421]
      * [Adding new relations and morphisms][422]
      * [Rewriting and nonreflexive relations][423]
      * [Rewriting and nonsymmetric relations][424]
      * [Rewriting in ambiguous setoid contexts][425]
      * [Rewriting with `Type` valued relations][426]
    * [Declaring rewrite relations][427]
    * [Commands and tactics][428]
      
      * [First class setoids and morphisms][429]
      * [Tactics enabled on user provided relations][430]
      * [Printing relations and morphisms][431]
    * [Understanding and fixing failed resolutions][432]
      
      * [Deprecated syntax and backward incompatibilities][433]
    * [Extensions][434]
      
      * [Rewriting under binders][435]
      * [Subrelations][436]
      * [Constant unfolding during rewriting][437]
      * [Constant unfolding during `Proper`-instance search][438]
    * [Strategies for rewriting][439]
      
      * [Usage][440]
      * [Definitions][441]
* [Creating new tactics][442]
  
  * [Ltac][443]
    
    * [Defects][444]
    * [Syntax][445]
    * [Values][446]
      
      * [Syntactic values][447]
      * [Substitution][448]
      * [Local definitions: let][449]
      * [Function construction and application][450]
      * [Tactics in terms][451]
    * [Goal selectors][452]
    * [Processing multiple goals][453]
    * [Branching and backtracking][454]
    * [Control flow][455]
      
      * [Sequence: ;][456]
      * [Do loop][457]
      * [Repeat loop][458]
      * [Catching errors: try][459]
      * [Conditional branching: tryif][460]
    * [Alternatives][461]
      
      * [Branching with backtracking: +][462]
      * [Local application of tactics: [> ... ]][463]
      * [First tactic to succeed][464]
      * [Solving][465]
      * [First tactic to make progress: ||][466]
      * [Detecting progress][467]
    * [Success and failure][468]
      
      * [Checking for success: assert_succeeds][469]
      * [Checking for failure: assert_fails][470]
      * [Failing][471]
      * [Soft cut: once][472]
      * [Checking for a single success: exactly_once][473]
    * [Manipulating values][474]
      
      * [Pattern matching on terms: match][475]
      * [Pattern matching on goals and hypotheses: match goal][476]
      * [Filling a term context][477]
      * [Generating fresh hypothesis names][478]
      * [Computing in a term: eval][479]
      * [Getting the type of a term][480]
      * [Manipulating untyped terms: type_term][481]
      * [Counting goals: numgoals][482]
      * [Testing boolean expressions: guard][483]
      * [Checking properties of terms][484]
    * [Timing][485]
      
      * [Timeout][486]
      * [Timing a tactic][487]
      * [Timing a tactic that evaluates to a term: time_constr][488]
    * [Print/identity tactic: idtac][489]
    * [Tactic toplevel definitions][490]
      
      * [Defining `L`tac symbols][491]
      * [Printing `L`tac tactics][492]
    * [Examples of using `L`tac][493]
      
      * [Proof that the natural numbers have at least two elements][494]
      * [Proving that a list is a permutation of a second list][495]
      * [Deciding intuitionistic propositional logic][496]
      * [Deciding type isomorphisms][497]
    * [Debugging `L`tac tactics][498]
      
      * [Backtraces][499]
      * [Tracing execution][500]
      * [Interactive debugger][501]
      * [Profiling `L`tac tactics][502]
      * [Run-time optimization tactic][503]
  * [Ltac2][504]
    
    * [General design][505]
    * [ML component][506]
      
      * [Overview][507]
      * [Type Syntax][508]
      * [Type declarations][509]
      * [APIs][510]
      * [Term Syntax][511]
      * [Ltac2 Definitions][512]
      * [Printing Ltac2 tactics][513]
      * [Reduction][514]
      * [Typing][515]
      * [Effects][516]
    * [Meta-programming][517]
      
      * [Overview][518]
      * [Quotations][519]
      * [Term Antiquotations][520]
      * [Match over terms][521]
      * [Match over goals][522]
      * [Match on values][523]
    * [Notations][524]
      
      * [Abbreviations][525]
      * [Defining tactics][526]
      * [Syntactic classes][527]
    * [Evaluation][528]
    * [Debug][529]
    * [Profiling][530]
    * [Compatibility layer with Ltac1][531]
      
      * [Ltac1 from Ltac2][532]
      * [Ltac2 from Ltac1][533]
      * [Switching between Ltac languages][534]
    * [Transition from Ltac1][535]
      
      * [Syntax changes][536]
      * [Tactic delay][537]
      * [Variable binding][538]
      * [Exception catching][539]

Using the Rocq Prover

* [Libraries and plugins][540]
  
  * [The Coq libraries][541]
    
    * [The prelude][542]
      
      * [Notations][543]
      * [Logic][544]
      * [Datatypes][545]
      * [Specification][546]
      * [Basic Arithmetic][547]
      * [Well-founded recursion][548]
      * [Tactics][549]
    * [Opam repository][550]
  * [Program extraction][551]
    
    * [Generating ML Code][552]
    * [Extraction Options][553]
      
      * [Setting the target language][554]
      * [Inlining and optimizations][555]
      * [Extra elimination of useless arguments][556]
      * [Accessing opaque proofs][557]
      * [Realizing axioms][558]
      * [Realizing inductive types][559]
      * [Generating FFI Code][560]
      * [Avoiding conflicts with existing filenames][561]
      * [Additional settings][562]
    * [Differences between Rocq and ML type systems][563]
    * [Some examples][564]
      
      * [A detailed example: Euclidean division][565]
      * [Extraction's horror museum][566]
      * [Users' Contributions][567]
  * [Program derivation][568]
  * [Functional induction][569]
    
    * [Advanced recursive functions][570]
    * [Tactics][571]
    * [Generation of induction principles with `Functional` `Scheme`][572]
    * [Flags][573]
  * [Writing Rocq libraries and plugins][574]
    
    * [Deprecating library objects, tactics or library files][575]
    * [Triggering warning for library objects or library files][576]
* [Command-line and graphical tools][577]
  
  * [Building Rocq Projects][578]
    
    * [Rocq configuration basics][579]
      
      * [Installing the Rocq Prover and Rocq packages with opam][580]
      * [Setup for working on your own projects][581]
      * [Building a project with _CoqProject (overview)][582]
      * [Logical paths and the load path][583]
      * [Modifying multiple interdependent projects at the same time][584]
      * [Installed and uninstalled packages][585]
      * [Upgrading to a new version of Rocq][586]
    * [Building a Rocq project with rocq makefile (details)][587]
      
      * [Comments][588]
    * [Building a Rocq project with Dune][589]
    * [rocq dep: Computing Module dependencies][590]
    * [Split compilation of native computation files][591]
    * [Using Rocq as a library][592]
    * [Embedded Rocq phrases inside LaTeX documents][593]
    * [Man pages][594]
  * [The Rocq Prover commands][595]
    
    * [Interactive use (rocq repl)][596]
    * [Batch compilation (rocq compile)][597]
    * [System configuration][598]
    * [Customization at launch time][599]
    * [Command parameters][600]
      
      * [`coqrc` start up script][601]
      * [Environment variables][602]
      * [Command line options][603]
    * [Profiling][604]
    * [Compiled interfaces (produced using `-vos`)][605]
    * [Compiled libraries checker (rocqchk)][606]
  * [Documenting Rocq files with rocq doc][607]
    
    * [Principles][608]
      
      * [Rocq material inside documentation.][609]
      * [Pretty-printing.][610]
      * [Sections][611]
      * [Lists.][612]
      * [Rules.][613]
      * [Emphasis.][614]
      * [Escaping to LaTeX and HTML.][615]
      * [Verbatim][616]
      * [Hyperlinks][617]
      * [Hiding / Showing parts of the source][618]
    * [Usage][619]
      
      * [Command line options][620]
    * [The rocq doc LaTeX style file][621]
  * [RocqIDE][622]
    
    * [Managing files and buffers, basic editing][623]
    * [Running Coq scripts][624]
    * [Asynchronous mode][625]
    * [Commands and templates][626]
    * [Queries][627]
    * [Compilation][628]
    * [Customizations][629]
      
      * [Preferences][630]
      * [Key bindings][631]
    * [Using Unicode symbols][632]
      
      * [Displaying Unicode symbols][633]
      * [Bindings for input of Unicode symbols][634]
      * [Adding custom bindings][635]
      * [Character encoding for saved files][636]
    * [Debugger][637]
      
      * [Breakpoints][638]
      * [Call Stack and Variables][639]
      * [Supported use cases][640]
  * [Asynchronous and Parallel Proof Processing][641]
    
    * [Proof annotations][642]
      
      * [Automatic suggestion of proof annotations][643]
    * [Proof blocks and error resilience][644]
      
      * [Caveats][645]
    * [Interactive mode][646]
    * [Limiting the number of parallel workers][647]
      
      * [Caveats][648]

Appendix

* [History and recent changes][649]
  
  * [Early history of Coq][650]
    
    * [Historical roots][651]
    * [Versions 1 to 5][652]
      
      * [Version 1][653]
      * [Version 2][654]
      * [Version 3][655]
      * [Version 4][656]
      * [Version 5][657]
    * [Versions 6][658]
      
      * [Version 6.1][659]
      * [Version 6.2][660]
      * [Version 6.3][661]
    * [Versions 7][662]
      
      * [Summary of changes][663]
      * [Details of changes in 7.0 and 7.1][664]
      * [Details of changes in 7.2][665]
      * [Details of changes in 7.3][666]
      * [Details of changes in 7.4][667]
  * [Recent changes][668]
    
    * [Version 9.1][669]
      
      * [Summary of changes][670]
      * [Changes in 9.1.0][671]
    * [Version 9.0][672]
      
      * [Summary of changes][673]
      * [Porting to The Rocq Prover][674]
      * [Renaming Advice][675]
      * [The Rocq Prover Website][676]
      * [Changes in 9.0.0][677]
    * [Version 8.20][678]
      
      * [Summary of changes][679]
      * [Changes in 8.20.0][680]
      * [Changes in 8.20.1][681]
    * [Version 8.19][682]
      
      * [Summary of changes][683]
      * [Changes in 8.19.0][684]
      * [Changes in 8.19.1][685]
      * [Changes in 8.19.2][686]
    * [Version 8.18][687]
      
      * [Summary of changes][688]
      * [Changes in 8.18.0][689]
    * [Version 8.17][690]
      
      * [Summary of changes][691]
      * [Changes in 8.17.0][692]
      * [Changes in 8.17.1][693]
    * [Version 8.16][694]
      
      * [Summary of changes][695]
      * [Changes in 8.16.0][696]
      * [Changes in 8.16.1][697]
    * [Version 8.15][698]
      
      * [Summary of changes][699]
      * [Changes in 8.15.0][700]
      * [Changes in 8.15.1][701]
      * [Changes in 8.15.2][702]
    * [Version 8.14][703]
      
      * [Summary of changes][704]
      * [Changes in 8.14.0][705]
      * [Changes in 8.14.1][706]
    * [Version 8.13][707]
      
      * [Summary of changes][708]
      * [Changes in 8.13+beta1][709]
      * [Changes in 8.13.0][710]
      * [Changes in 8.13.1][711]
      * [Changes in 8.13.2][712]
    * [Version 8.12][713]
      
      * [Summary of changes][714]
      * [Changes in 8.12+beta1][715]
      * [Changes in 8.12.0][716]
      * [Changes in 8.12.1][717]
      * [Changes in 8.12.2][718]
    * [Version 8.11][719]
      
      * [Summary of changes][720]
      * [Changes in 8.11+beta1][721]
      * [Changes in 8.11.0][722]
      * [Changes in 8.11.1][723]
      * [Changes in 8.11.2][724]
    * [Version 8.10][725]
      
      * [Summary of changes][726]
      * [Other changes in 8.10+beta1][727]
      * [Changes in 8.10+beta2][728]
      * [Changes in 8.10+beta3][729]
      * [Changes in 8.10.0][730]
      * [Changes in 8.10.1][731]
      * [Changes in 8.10.2][732]
    * [Version 8.9][733]
      
      * [Summary of changes][734]
      * [Details of changes in 8.9+beta1][735]
      * [Changes in 8.8.0][736]
      * [Changes in 8.8.1][737]
    * [Version 8.8][738]
      
      * [Summary of changes][739]
      * [Details of changes in 8.8+beta1][740]
      * [Details of changes in 8.8.0][741]
      * [Details of changes in 8.8.1][742]
      * [Details of changes in 8.8.2][743]
    * [Version 8.7][744]
      
      * [Summary of changes][745]
      * [Potential compatibility issues][746]
      * [Details of changes in 8.7+beta1][747]
      * [Details of changes in 8.7+beta2][748]
      * [Details of changes in 8.7.0][749]
      * [Details of changes in 8.7.1][750]
      * [Details of changes in 8.7.2][751]
    * [Version 8.6][752]
      
      * [Summary of changes][753]
      * [Potential sources of incompatibilities][754]
      * [Details of changes in 8.6beta1][755]
      * [Details of changes in 8.6][756]
      * [Details of changes in 8.6.1][757]
    * [Version 8.5][758]
      
      * [Summary of changes][759]
      * [Potential sources of incompatibilities][760]
      * [Details of changes in 8.5beta1][761]
      * [Details of changes in 8.5beta2][762]
      * [Details of changes in 8.5beta3][763]
      * [Details of changes in 8.5][764]
      * [Details of changes in 8.5pl1][765]
      * [Details of changes in 8.5pl2][766]
      * [Details of changes in 8.5pl3][767]
    * [Version 8.4][768]
      
      * [Summary of changes][769]
      * [Potential sources of incompatibilities][770]
      * [Details of changes in 8.4beta][771]
      * [Details of changes in 8.4beta2][772]
      * [Details of changes in 8.4][773]
    * [Version 8.3][774]
      
      * [Summary of changes][775]
      * [Details of changes][776]
    * [Version 8.2][777]
      
      * [Summary of changes][778]
      * [Details of changes][779]
    * [Version 8.1][780]
      
      * [Summary of changes][781]
      * [Details of changes in 8.1beta][782]
      * [Details of changes in 8.1gamma][783]
      * [Details of changes in 8.1][784]
    * [Version 8.0][785]
      
      * [Summary of changes][786]
      * [Details of changes in 8.0beta old syntax][787]
      * [Details of changes in 8.0beta new syntax][788]
      * [Details of changes in 8.0][789]
* [Indexes][790]
  
  * [Glossary index][791]
  * [Command index][792]
  * [Tactic index][793]
  * [Attribute index][794]
  * [Flags, options and tables index][795]
  * [Errors and warnings index][796]
  * [Index][797]
* [Bibliography][798]
** [The Rocq Prover][799]

* Introduction and Contents
* [ Edit on GitHub][800]
\[\begin{split}\newcommand{\as}{\kw{as}} \newcommand{\case}{\kw{case}}
\newcommand{\cons}{\textsf{cons}} \newcommand{\consf}{\textsf{consf}}
\newcommand{\emptyf}{\textsf{emptyf}} \newcommand{\End}{\kw{End}}
\newcommand{\kwend}{\kw{end}} \newcommand{\even}{\textsf{even}}
\newcommand{\evenO}{\textsf{even}_\textsf{O}}
\newcommand{\evenS}{\textsf{even}_\textsf{S}} \newcommand{\Fix}{\kw{Fix}}
\newcommand{\fix}{\kw{fix}} \newcommand{\for}{\textsf{for}}
\newcommand{\forest}{\textsf{forest}} \newcommand{\Functor}{\kw{Functor}}
\newcommand{\In}{\kw{in}}
\newcommand{\ind}[3]{\kw{Ind}~[#1]\left(#2\mathrm{~:=~}#3\right)}
\newcommand{\Indp}[4]{\kw{Ind}_{#4}[#1](#2:=#3)}
\newcommand{\Indpstr}[5]{\kw{Ind}_{#4}[#1](#2:=#3)/{#5}}
\newcommand{\injective}{\kw{injective}} \newcommand{\kw}[1]{\textsf{#1}}
\newcommand{\length}{\textsf{length}}
\newcommand{\letin}[3]{\kw{let}~#1:=#2~\kw{in}~#3}
\newcommand{\List}{\textsf{list}} \newcommand{\lra}{\longrightarrow}
\newcommand{\Match}{\kw{match}}
\newcommand{\Mod}[3]{{\kw{Mod}}({#1}:{#2}\,\zeroone{:={#3}})}
\newcommand{\ModImp}[3]{{\kw{Mod}}({#1}:{#2}:={#3})}
\newcommand{\ModA}[2]{{\kw{ModA}}({#1}=={#2})}
\newcommand{\ModS}[2]{{\kw{Mod}}({#1}:{#2})}
\newcommand{\ModType}[2]{{\kw{ModType}}({#1}:={#2})} \newcommand{\mto}{.\;}
\newcommand{\nat}{\textsf{nat}} \newcommand{\Nil}{\textsf{nil}}
\newcommand{\nilhl}{\textsf{nil\_hl}} \newcommand{\nO}{\textsf{O}}
\newcommand{\node}{\textsf{node}} \newcommand{\nS}{\textsf{S}}
\newcommand{\odd}{\textsf{odd}} \newcommand{\oddS}{\textsf{odd}_\textsf{S}}
\newcommand{\ovl}[1]{\overline{#1}} \newcommand{\Pair}{\textsf{pair}}
\newcommand{\plus}{\mathsf{plus}} \newcommand{\SProp}{\textsf{SProp}}
\newcommand{\Prop}{\textsf{Prop}} \newcommand{\return}{\kw{return}}
\newcommand{\Set}{\textsf{Set}} \newcommand{\Sort}{\mathcal{S}}
\newcommand{\Str}{\textsf{Stream}} \newcommand{\Struct}{\kw{Struct}}
\newcommand{\subst}[3]{#1\{#2/#3\}} \newcommand{\tl}{\textsf{tl}}
\newcommand{\tree}{\textsf{tree}} \newcommand{\trii}{\triangleright_\iota}
\newcommand{\Type}{\textsf{Type}} \newcommand{\WEV}[3]{\mbox{$#1[] \vdash #2
\lra #3$}} \newcommand{\WEVT}[3]{\mbox{$#1[] \vdash #2 \lra$}\\ \mbox{$ #3$}}
\newcommand{\WF}[2]{{\mathcal{W\!F}}(#1)[#2]} \newcommand{\WFE}[1]{\WF{E}{#1}}
\newcommand{\WFT}[2]{#1[] \vdash {\mathcal{W\!F}}(#2)}
\newcommand{\WFTWOLINES}[2]{{\mathcal{W\!F}}\begin{array}{l}(#1)\\\mbox{}[{#2}]\
end{array}} \newcommand{\with}{\kw{with}} \newcommand{\WS}[3]{#1[] \vdash #2 <:
#3} \newcommand{\WSE}[2]{\WS{E}{#1}{#2}} \newcommand{\WT}[4]{#1[#2] \vdash #3 :
#4} \newcommand{\WTE}[3]{\WT{E}{#1}{#2}{#3}}
\newcommand{\WTEG}[2]{\WTE{\Gamma}{#1}{#2}}
\newcommand{\WTM}[3]{\WT{#1}{}{#2}{#3}} \newcommand{\zeroone}[1]{[{#1}]}
\end{split}\]

# Introduction and Contents[][801]

This is the reference manual of the Rocq Prover. Rocq is a proof assistant or
interactive theorem prover. It lets you formalize mathematical concepts and then
helps you interactively generate machine-checked proofs of theorems. Machine
checking gives users much more confidence that the proofs are correct compared
to human-generated and -checked proofs. Rocq has been used in a number of
flagship verification projects, including the [CompCert verified C
compiler][802], and has served to verify the proof of the [four color
theorem][803] (among many other mathematical formalizations).

Users generate proofs by entering a series of tactics that constitute steps in
the proof. There are many built-in tactics, some of which are elementary, while
others implement complex decision procedures (such as [`lia`][804], a decision
procedure for linear integer arithmetic). [Ltac][805] and its planned
replacement, [Ltac2][806], provide languages to define new tactics by combining
existing tactics with looping and conditional constructs. These permit
automation of large parts of proofs and sometimes entire proofs. Furthermore,
users can add novel tactics or functionality by creating Rocq plugins using
OCaml.

The Rocq kernel, a small part of the Rocq Prover, does the final verification
that the tactic-generated proof is valid. Usually the tactic-generated proof is
indeed correct, but delegating proof verification to the kernel means that even
if a tactic is buggy, it won't be able to introduce an incorrect proof into the
system.

Finally, Rocq also supports extraction of verified programs to programming
languages such as OCaml and Haskell. This provides a way of executing Rocq code
efficiently and can be used to create verified software libraries.

To learn Rocq, beginners are advised to first start with a tutorial / book.
Several such tutorials / books are listed at
[https://rocq-prover.org/docs][807].

This manual is organized in three main parts, plus an appendix:

* **The first part presents the specification language of the Rocq Prover**,
  that allows to define programs and state mathematical theorems. [Core
  language][808] presents the language that the kernel of Rocq understands.
  [Language extensions][809] presents the richer language, with notations,
  implicits, etc. that a user can use and which is translated down to the
  language of the kernel by means of an "elaboration process".
* **The second part presents proof mode**, the central feature of the Rocq
  Prover. [Basic proof writing][810] introduces this interactive mode and the
  available proof languages. [Automatic solvers and programmable tactics][811]
  presents some more advanced tactics, while [Creating new tactics][812] is
  about the languages that allow a user to combine tactics together and develop
  new ones.
* **The third part shows how to use the Rocq Prover in practice.** [Libraries
  and plugins][813] presents some of the essential reusable blocks from the
  ecosystem and some particularly important extensions such as the program
  extraction mechanism. [Command-line and graphical tools][814] documents
  important tools that a user needs to build a Rocq project.
* In the appendix, [History and recent changes][815] presents the history of
  Rocq and changes in recent releases. This is an important reference if you
  upgrade the version of Rocq that you use. The various [indexes][816] are very
  useful to **quickly browse the manual and find what you are looking for.**
  They are often the main entry point to the manual.

The full table of contents is presented below:

## Contents[][817]

* [Introduction and Contents][818]

Specification language

* [Core language][819]
  
  * [Basic notions and conventions][820]
    
    * [Syntax and lexical conventions][821]
      
      * [Syntax conventions][822]
      * [Lexical conventions][823]
    * [Essential vocabulary][824]
    * [Settings][825]
      
      * [Attributes][826]
        
        * [Generic attributes][827]
        * [Document-level attributes][828]
      * [Flags, Options and Tables][829]
        
        * [Locality attributes supported by `Set` and `Unset`][830]
  * [Sorts][831]
  * [Functions and assumptions][832]
    
    * [Binders][833]
    * [Functions (fun) and function types (forall)][834]
    * [Function application][835]
    * [Assumptions][836]
  * [Definitions][837]
    
    * [Let-in definitions][838]
    * [Type cast][839]
    * [Top-level definitions][840]
    * [Assertions and proofs][841]
  * [Conversion rules][842]
    
    * [α-conversion][843]
    * [β-reduction][844]
    * [δ-reduction][845]
    * [ι-reduction][846]
    * [ζ-reduction][847]
    * [η-expansion][848]
    * [Examples][849]
    * [Proof Irrelevance][850]
    * [Convertibility][851]
  * [Typing rules][852]
    
    * [The terms][853]
    * [Typing rules][854]
    * [Subtyping rules][855]
    * [The Calculus of Inductive Constructions with impredicative Set][856]
  * [Variants and the `match` construct][857]
    
    * [Variants][858]
      
      * [Private (matching) inductive types][859]
    * [Definition by cases: match][860]
  * [Record types][861]
    
    * [Defining record types][862]
    * [Constructing records][863]
    * [Accessing fields (projections)][864]
    * [Settings for printing records][865]
    * [Primitive Projections][866]
      
      * [Reduction][867]
      * [Compatibility Constants for Projections][868]
  * [Inductive types and recursive functions][869]
    
    * [Inductive types][870]
      
      * [Simple inductive types][871]
        
        * [Automatic Prop lowering][872]
      * [Simple indexed inductive types][873]
      * [Parameterized inductive types][874]
      * [Mutually defined inductive types][875]
    * [Recursive functions: fix][876]
    * [Top-level recursive functions][877]
    * [Theory of inductive definitions][878]
      
      * [Types of inductive objects][879]
      * [Well-formed inductive definitions][880]
        
        * [Arity of a given sort][881]
        * [Arity][882]
        * [Type of constructor][883]
        * [Positivity Condition][884]
        * [Strict positivity][885]
        * [Nested Positivity][886]
        * [Correctness rules][887]
        * [Template polymorphism][888]
      * [Destructors][889]
        
        * [The match ... with ... end construction][890]
      * [Fixpoint definitions][891]
        
        * [Typing rule][892]
        * [Reduction rule][893]
  * [Coinductive types and corecursive functions][894]
    
    * [Coinductive types][895]
      
      * [Caveat][896]
    * [Co-recursive functions: cofix][897]
    * [Top-level definitions of corecursive functions][898]
  * [Sections][899]
    
    * [Using sections][900]
    * [Summary of locality attributes in a section][901]
    * [Typing rules used at the end of a section][902]
  * [The Module System][903]
    
    * [Modules and module types][904]
    * [Using modules][905]
      
      * [Examples][906]
    * [Qualified names][907]
    * [Controlling the scope of commands with locality attributes][908]
    * [Summary of locality attributes in a module][909]
    * [Typing Modules][910]
  * [Primitive objects][911]
    
    * [Primitive Integers][912]
    * [Primitive Floats][913]
    * [Primitive Arrays][914]
    * [Primitive (Byte-Based) Strings][915]
  * [Polymorphic Universes][916]
    
    * [General Presentation][917]
    * [Polymorphic, Monomorphic][918]
    * [Cumulative, NonCumulative][919]
      
      * [Specifying cumulativity][920]
      * [Cumulativity Weak Constraints][921]
    * [Global and local universes][922]
    * [Conversion and unification][923]
    * [Minimization][924]
    * [Explicit Universes][925]
    * [Printing universes][926]
      
      * [Polymorphic definitions][927]
    * [Sort polymorphism][928]
    * [Explicit Sorts][929]
    * [Universe polymorphism and sections][930]
  * [SProp (proof irrelevant propositions)][931]
    
    * [Basic constructs][932]
    * [Encodings for strict propositions][933]
    * [Definitional UIP][934]
      
      * [Non Termination with UIP][935]
    * [Debugging \(\SProp\) issues][936]
  * [User-defined rewrite rules][937]
    
    * [Symbols][938]
    * [Rewrite rules][939]
    * [Pattern syntax][940]
    * [Higher-order pattern holes][941]
    * [Universe polymorphic rules][942]
    * [Rewrite rules, type preservation, confluence and termination][943]
    * [Compatibility with the eta laws][944]
    * [Level of support][945]
* [Language extensions][946]
  
  * [Command level processing][947]
    
    * [Lexing][948]
    * [Parsing][949]
    * [Synterp][950]
    * [Interp][951]
  * [Term level processing][952]
  * [Existential variables][953]
    
    * [Inferable subterms][954]
    * [e* tactics that can create existential variables][955]
    * [Automatic resolution of existential variables][956]
    * [Explicit display of existential instances for pretty-printing][957]
    * [Solving existential variables using tactics][958]
  * [Implicit arguments][959]
    
    * [The different kinds of implicit arguments][960]
      
      * [Implicit arguments inferable from the knowledge of other arguments of a
        function][961]
      * [Implicit arguments inferable by resolution][962]
    * [Maximal and non-maximal insertion of implicit arguments][963]
      
      * [Trailing Implicit Arguments][964]
    * [Casual use of implicit arguments][965]
    * [Declaration of implicit arguments][966]
      
      * [Implicit Argument Binders][967]
      * [Mode for automatic declaration of implicit arguments][968]
      * [Controlling strict implicit arguments][969]
      * [Controlling contextual implicit arguments][970]
      * [Controlling reversible-pattern implicit arguments][971]
      * [Controlling the insertion of implicit arguments not followed by
        explicit arguments][972]
      * [Combining manual declaration and automatic declaration][973]
    * [Explicit applications][974]
    * [Displaying implicit arguments][975]
    * [Displaying implicit arguments when pretty-printing][976]
    * [Interaction with subtyping][977]
    * [Deactivation of implicit arguments for parsing][978]
    * [Implicit types of variables][979]
    * [Implicit generalization][980]
  * [Extended pattern matching][981]
    
    * [Variants and extensions of `match`][982]
      
      * [Multiple and nested pattern matching][983]
      * [Pattern-matching on boolean values: the if expression][984]
      * [Irrefutable patterns: the destructuring let variants][985]
        
        * [First destructuring let syntax][986]
        * [Second destructuring let syntax][987]
      * [Controlling pretty-printing of match expressions][988]
        
        * [Printing nested patterns][989]
        * [Factorization of clauses with same right-hand side][990]
        * [Use of a default clause][991]
        * [Printing of wildcard patterns][992]
        * [Printing of the elimination predicate][993]
        * [Printing of hidden subterms][994]
        * [Printing matching on irrefutable patterns][995]
        * [Printing matching on booleans][996]
      * [Conventions about unused pattern-matching variables][997]
    * [Patterns][998]
    * [Multiple patterns][999]
    * [Aliasing subpatterns][1000]
    * [Nested patterns][1001]
    * [Disjunctive patterns][1002]
    * [About patterns of parametric types][1003]
      
      * [Parameters in patterns][1004]
    * [Implicit arguments in patterns][1005]
    * [Matching objects of dependent types][1006]
    * [Understanding dependencies in patterns][1007]
    * [When the elimination predicate must be provided][1008]
      
      * [Dependent pattern matching][1009]
      * [Multiple dependent pattern matching][1010]
      * [Patterns in `in`][1011]
    * [Using pattern matching to write proofs][1012]
    * [Pattern-matching on inductive objects involving local definitions][1013]
    * [Pattern-matching and coercions][1014]
    * [When does the expansion strategy fail?][1015]
  * [Syntax extensions and notation scopes][1016]
    
    * [Notations][1017]
      
      * [Basic notations][1018]
      * [Precedences and associativity][1019]
      * [Complex notations][1020]
      * [Simple factorization rules][1021]
      * [Use of notations for printing][1022]
      * [The Infix command][1023]
      * [Reserving notations][1024]
      * [Simultaneous definition of terms and notations][1025]
      * [Enabling and disabling notations][1026]
      * [Displaying information about notations][1027]
      * [Locating notations][1028]
      * [Inheritance of the properties of arguments of constants bound to a
        notation][1029]
      * [Notations and binders][1030]
        
        * [Binders bound in the notation and parsed as identifiers][1031]
        * [Binders bound in the notation and parsed as patterns][1032]
        * [Binders bound in the notation and parsed as terms][1033]
        * [Binders bound in the notation and parsed as general binders][1034]
        * [Binders not bound in the notation][1035]
        * [Notations with expressions used both as binder and term][1036]
      * [Notations with recursive patterns][1037]
      * [Notations with recursive patterns involving binders][1038]
      * [Predefined entries][1039]
      * [Custom entries][1040]
      * [Syntax][1041]
    * [Notation scopes][1042]
      
      * [Global interpretation rules for notations][1043]
      * [Local interpretation rules for notations][1044]
        
        * [Opening a notation scope locally][1045]
        * [Binding types or coercion classes to notation scopes][1046]
      * [The `type_scope` notation scope][1047]
      * [The `function_scope` notation scope][1048]
      * [Notation scopes used in the standard library of Rocq][1049]
      * [Displaying information about scopes][1050]
    * [Abbreviations][1051]
    * [Numbers and strings][1052]
      
      * [Number notations][1053]
      * [String notations][1054]
    * [Tactic Notations][1055]
  * [Setting properties of a function's arguments][1056]
    
    * [Manual declaration of implicit arguments][1057]
    * [Automatic declaration of implicit arguments][1058]
    * [Renaming implicit arguments][1059]
    * [Binding arguments to scopes][1060]
    * [Effects of `Arguments` on unfolding][1061]
    * [Bidirectionality hints][1062]
  * [Implicit Coercions][1063]
    
    * [General Presentation][1064]
    * [Coercion Classes][1065]
    * [Coercions][1066]
    * [Reversible Coercions][1067]
    * [Identity Coercions][1068]
    * [Inheritance Graph][1069]
    * [Coercion Classes][1070]
    * [Displaying Available Coercions][1071]
    * [Activating the Printing of Coercions][1072]
    * [Classes as Records][1073]
    * [Coercions and Sections][1074]
    * [Coercions and Modules][1075]
    * [Examples][1076]
  * [Typeclasses][1077]
    
    * [Typeclass and instance declarations][1078]
    * [Binding typeclasses][1079]
    * [Parameterized instances][1080]
    * [Sections and contexts][1081]
    * [Building hierarchies][1082]
      
      * [Superclasses][1083]
      * [Substructures][1084]
    * [Command summary][1085]
      
      * [Typeclasses Transparent, Typeclasses Opaque][1086]
      * [Settings][1087]
      * [Typeclasses eauto][1088]
  * [Canonical Structures][1089]
    
    * [Declaration of canonical structures][1090]
    * [Notation overloading][1091]
      
      * [Derived Canonical Structures][1092]
    * [Hierarchy of structures][1093]
      
      * [Compact declaration of Canonical Structures][1094]
  * [Program][1095]
    
    * [Elaborating programs][1096]
      
      * [Syntactic control over equalities][1097]
      * [Program Definition][1098]
      * [Program Fixpoint][1099]
      * [Program Lemma][1100]
    * [Solving obligations][1101]
    * [Frequently Asked Questions][1102]
  * [Commands][1103]
    
    * [Displaying][1104]
    * [Query commands][1105]
    * [Requests to the environment][1106]
    * [Printing flags][1107]
    * [Loading files][1108]
    * [Compiled files][1109]
    * [Load paths][1110]
    * [Extra Dependencies][1111]
    * [Backtracking][1112]
    * [Quitting and debugging][1113]
    * [Controlling display][1114]
    * [Printing constructions in full][1115]
    * [Controlling Typing Flags][1116]
    * [Internal registration commands][1117]
      
      * [Exposing constants to OCaml libraries][1118]
      * [Inlining hints for the fast reduction machines][1119]
      * [Registering primitive operations][1120]

Proofs

* [Basic proof writing][1121]
  
  * [Proof mode][1122]
    
    * [Proof State][1123]
    * [Entering and exiting proof mode][1124]
      
      * [Proof using options][1125]
      * [Name a set of section hypotheses for `Proof using`][1126]
    * [Proof modes][1127]
    * [Managing goals][1128]
      
      * [Focusing goals][1129]
        
        * [Curly braces][1130]
        * [Bullets][1131]
        * [Other focusing commands][1132]
      * [Shelving goals][1133]
      * [Reordering goals][1134]
    * [Proving a subgoal as a separate lemma: abstract][1135]
    * [Requesting information][1136]
    * [Showing differences between proof steps][1137]
      
      * [How to enable diffs][1138]
      * [How diffs are calculated][1139]
      * ["Show Proof" differences][1140]
    * [Delaying solving unification constraints][1141]
    * [Proof maintenance][1142]
    * [Controlling proof mode][1143]
    * [Controlling memory usage][1144]
  * [Tactics][1145]
    
    * [Common elements of tactics][1146]
      
      * [Reserved keywords][1147]
      * [Invocation of tactics][1148]
      * [Bindings][1149]
      * [Intro patterns][1150]
      * [Occurrence clauses][1151]
      * [Automatic clearing of hypotheses][1152]
    * [Applying theorems][1153]
    * [Managing the local context][1154]
    * [Controlling the proof flow][1155]
    * [Classical tactics][1156]
    * [Performance-oriented tactic variants][1157]
  * [Reasoning with equalities][1158]
    
    * [Tactics for simple equalities][1159]
    * [Rewriting with Leibniz and setoid equality][1160]
    * [Rewriting with definitional equality][1161]
    * [Applying conversion rules][1162]
      
      * [Fast reduction tactics: vm_compute and native_compute][1163]
      * [Computing in a term: eval and Eval][1164]
    * [Controlling reduction strategies and the conversion algorithm][1165]
  * [Reasoning with inductive types][1166]
    
    * [Applying constructors][1167]
    * [Case analysis][1168]
    * [Induction][1169]
    * [Equality of inductive types][1170]
      
      * [Helper tactics][1171]
    * [Generation of induction principles with `Scheme`][1172]
      
      * [Automatic declaration of schemes][1173]
      * [Combined Scheme][1174]
    * [Generation of inversion principles with `Derive` `Inversion`][1175]
    * [Examples of `dependent destruction` / `dependent induction`][1176]
      
      * [A larger example][1177]
  * [The SSReflect proof language][1178]
    
    * [Introduction][1179]
      
      * [Acknowledgments][1180]
    * [Usage][1181]
      
      * [Getting started][1182]
      * [Compatibility issues][1183]
    * [Gallina extensions][1184]
      
      * [Pattern assignment][1185]
      * [Pattern conditional][1186]
      * [Parametric polymorphism][1187]
      * [Anonymous arguments][1188]
      * [Wildcards][1189]
      * [Definitions][1190]
      * [Abbreviations][1191]
        
        * [Matching][1192]
        * [Occurrence selection][1193]
      * [Basic localization][1194]
    * [Basic tactics][1195]
      
      * [Bookkeeping][1196]
      * [The defective tactics][1197]
        
        * [The move tactic.][1198]
        * [The case tactic][1199]
        * [The elim tactic][1200]
        * [The apply tactic][1201]
      * [Discharge][1202]
        
        * [Clear rules][1203]
        * [Matching for apply and exact][1204]
        * [The abstract tactic][1205]
      * [Introduction in the context][1206]
        
        * [Simplification items][1207]
        * [Views][1208]
        * [Intro patterns][1209]
        * [Clear switch][1210]
        * [Branching and destructuring][1211]
        * [Block introduction][1212]
      * [Generation of equations][1213]
      * [Type families][1214]
    * [Control flow][1215]
      
      * [Indentation and bullets][1216]
      * [Terminators][1217]
      * [Selectors][1218]
      * [Iteration][1219]
      * [Localization][1220]
      * [Structure][1221]
        
        * [The have tactic.][1222]
        * [Generating let in context entries with have][1223]
        * [The have tactic and typeclass resolution][1224]
        * [Variants: the suff and wlog tactics][1225]
          
          * [Advanced generalization][1226]
    * [Rewriting][1227]
      
      * [An extended rewrite tactic][1228]
      * [Remarks and examples][1229]
        
        * [Rewrite redex selection][1230]
        * [Chained rewrite steps][1231]
        * [Explicit redex switches are matched first][1232]
        * [Occurrence switches and redex switches][1233]
        * [Occurrence selection and repetition][1234]
        * [Multi-rule rewriting][1235]
        * [Wildcards vs abstractions][1236]
        * [When SSReflect rewrite fails on standard Rocq licit rewrite][1237]
        * [Existential metavariables and rewriting][1238]
      * [Rewriting under binders][1239]
        
        * [The under tactic][1240]
        * [Interactive mode][1241]
          
          * [The over tactic][1242]
        * [One-liner mode][1243]
      * [Locking, unlocking][1244]
      * [Congruence][1245]
    * [Contextual patterns][1246]
      
      * [Syntax][1247]
      * [Matching contextual patterns][1248]
      * [Examples][1249]
        
        * [Contextual pattern in set and the : tactical][1250]
        * [Contextual patterns in rewrite][1251]
      * [Patterns for recurrent contexts][1252]
    * [Views and reflection][1253]
      
      * [Interpreting eliminations][1254]
      * [Interpreting assumptions][1255]
        
        * [Specializing assumptions][1256]
      * [Interpreting goals][1257]
      * [Boolean reflection][1258]
      * [The reflect predicate][1259]
      * [General mechanism for interpreting goals and assumptions][1260]
        
        * [Specializing assumptions][1261]
        * [Interpreting assumptions][1262]
        * [Interpreting goals][1263]
      * [Interpreting equivalences][1264]
      * [Declaring new Hint Views][1265]
      * [Multiple views][1266]
      * [Additional view shortcuts][1267]
    * [Synopsis and Index][1268]
      
      * [Parameters][1269]
      * [Items and switches][1270]
      * [Tactics][1271]
      * [Tacticals][1272]
      * [Commands][1273]
      * [Settings][1274]
* [Automatic solvers and programmable tactics][1275]
  
  * [Solvers for logic and equality][1276]
  * [Micromega: solvers for arithmetic goals over ordered rings][1277]
    
    * [Short description of the tactics][1278]
    * [*Positivstellensatz* refutations][1279]
    * [`lra`: a decision procedure for linear real and rational
      arithmetic][1280]
    * [`lia`: a tactic for linear integer arithmetic][1281]
      
      * [High level view of `lia`][1282]
      * [Cutting plane proofs][1283]
      * [Case split][1284]
    * [`nra`: a proof procedure for non-linear arithmetic][1285]
    * [`nia`: a proof procedure for non-linear integer arithmetic][1286]
    * [`psatz`: a proof procedure for non-linear arithmetic][1287]
    * [`zify`: pre-processing of arithmetic goals][1288]
  * [ring and field: solvers for polynomial and rational equations][1289]
    
    * [What does this tactic do?][1290]
    * [The variables map][1291]
    * [Is it automatic?][1292]
    * [Concrete usage][1293]
    * [Adding a ring structure][1294]
    * [How does it work?][1295]
    * [Dealing with fields][1296]
    * [Adding a new field structure][1297]
    * [History of ring][1298]
    * [Discussion][1299]
  * [Nsatz: a solver for equalities in integral domains][1300]
    
    * [More about `nsatz`][1301]
  * [Programmable proof search][1302]
    
    * [Tactics][1303]
    * [Hint databases][1304]
      
      * [Creating hint databases][1305]
      * [Hint databases defined in the Rocq standard library][1306]
    * [Creating Hints][1307]
      
      * [Hint locality][1308]
    * [Setting implicit automation tactics][1309]
  * [Generalized rewriting][1310]
    
    * [Introduction to generalized rewriting][1311]
      
      * [Relations and morphisms][1312]
      * [Adding new relations and morphisms][1313]
      * [Rewriting and nonreflexive relations][1314]
      * [Rewriting and nonsymmetric relations][1315]
      * [Rewriting in ambiguous setoid contexts][1316]
      * [Rewriting with `Type` valued relations][1317]
    * [Declaring rewrite relations][1318]
    * [Commands and tactics][1319]
      
      * [First class setoids and morphisms][1320]
      * [Tactics enabled on user provided relations][1321]
      * [Printing relations and morphisms][1322]
    * [Understanding and fixing failed resolutions][1323]
      
      * [Deprecated syntax and backward incompatibilities][1324]
    * [Extensions][1325]
      
      * [Rewriting under binders][1326]
      * [Subrelations][1327]
      * [Constant unfolding during rewriting][1328]
      * [Constant unfolding during `Proper`-instance search][1329]
    * [Strategies for rewriting][1330]
      
      * [Usage][1331]
      * [Definitions][1332]
* [Creating new tactics][1333]
  
  * [Ltac][1334]
    
    * [Defects][1335]
    * [Syntax][1336]
    * [Values][1337]
      
      * [Syntactic values][1338]
      * [Substitution][1339]
      * [Local definitions: let][1340]
      * [Function construction and application][1341]
      * [Tactics in terms][1342]
    * [Goal selectors][1343]
    * [Processing multiple goals][1344]
    * [Branching and backtracking][1345]
    * [Control flow][1346]
      
      * [Sequence: ;][1347]
      * [Do loop][1348]
      * [Repeat loop][1349]
      * [Catching errors: try][1350]
      * [Conditional branching: tryif][1351]
    * [Alternatives][1352]
      
      * [Branching with backtracking: +][1353]
      * [Local application of tactics: [> ... ]][1354]
      * [First tactic to succeed][1355]
      * [Solving][1356]
      * [First tactic to make progress: ||][1357]
      * [Detecting progress][1358]
    * [Success and failure][1359]
      
      * [Checking for success: assert_succeeds][1360]
      * [Checking for failure: assert_fails][1361]
      * [Failing][1362]
      * [Soft cut: once][1363]
      * [Checking for a single success: exactly_once][1364]
    * [Manipulating values][1365]
      
      * [Pattern matching on terms: match][1366]
      * [Pattern matching on goals and hypotheses: match goal][1367]
      * [Filling a term context][1368]
      * [Generating fresh hypothesis names][1369]
      * [Computing in a term: eval][1370]
      * [Getting the type of a term][1371]
      * [Manipulating untyped terms: type_term][1372]
      * [Counting goals: numgoals][1373]
      * [Testing boolean expressions: guard][1374]
      * [Checking properties of terms][1375]
    * [Timing][1376]
      
      * [Timeout][1377]
      * [Timing a tactic][1378]
      * [Timing a tactic that evaluates to a term: time_constr][1379]
    * [Print/identity tactic: idtac][1380]
    * [Tactic toplevel definitions][1381]
      
      * [Defining `L`tac symbols][1382]
      * [Printing `L`tac tactics][1383]
    * [Examples of using `L`tac][1384]
      
      * [Proof that the natural numbers have at least two elements][1385]
      * [Proving that a list is a permutation of a second list][1386]
      * [Deciding intuitionistic propositional logic][1387]
      * [Deciding type isomorphisms][1388]
    * [Debugging `L`tac tactics][1389]
      
      * [Backtraces][1390]
      * [Tracing execution][1391]
      * [Interactive debugger][1392]
      * [Profiling `L`tac tactics][1393]
      * [Run-time optimization tactic][1394]
  * [Ltac2][1395]
    
    * [General design][1396]
    * [ML component][1397]
      
      * [Overview][1398]
      * [Type Syntax][1399]
      * [Type declarations][1400]
      * [APIs][1401]
      * [Term Syntax][1402]
      * [Ltac2 Definitions][1403]
      * [Printing Ltac2 tactics][1404]
      * [Reduction][1405]
      * [Typing][1406]
      * [Effects][1407]
        
        * [Standard IO][1408]
        * [Fatal errors][1409]
        * [Backtracking][1410]
        * [Goals][1411]
    * [Meta-programming][1412]
      
      * [Overview][1413]
      * [Quotations][1414]
        
        * [Built-in quotations][1415]
        * [Strict vs. non-strict mode][1416]
      * [Term Antiquotations][1417]
        
        * [Syntax][1418]
        * [Semantics][1419]
          
          * [Static semantics][1420]
          * [Dynamic semantics][1421]
      * [Match over terms][1422]
      * [Match over goals][1423]
      * [Match on values][1424]
    * [Notations][1425]
      
      * [Abbreviations][1426]
      * [Defining tactics][1427]
      * [Syntactic classes][1428]
    * [Evaluation][1429]
    * [Debug][1430]
    * [Profiling][1431]
    * [Compatibility layer with Ltac1][1432]
      
      * [Ltac1 from Ltac2][1433]
        
        * [Simple API][1434]
        * [Low-level API][1435]
      * [Ltac2 from Ltac1][1436]
      * [Switching between Ltac languages][1437]
    * [Transition from Ltac1][1438]
      
      * [Syntax changes][1439]
      * [Tactic delay][1440]
      * [Variable binding][1441]
        
        * [In Ltac expressions][1442]
        * [In quotations][1443]
      * [Exception catching][1444]

Using the Rocq Prover

* [Libraries and plugins][1445]
  
  * [The Coq libraries][1446]
    
    * [The prelude][1447]
      
      * [Notations][1448]
      * [Logic][1449]
        
        * [Propositional Connectives][1450]
        * [Quantifiers][1451]
        * [Equality][1452]
        * [Lemmas][1453]
      * [Datatypes][1454]
        
        * [Programming][1455]
      * [Specification][1456]
      * [Basic Arithmetic][1457]
      * [Well-founded recursion][1458]
      * [Tactics][1459]
    * [Opam repository][1460]
  * [Program extraction][1461]
    
    * [Generating ML Code][1462]
    * [Extraction Options][1463]
      
      * [Setting the target language][1464]
      * [Inlining and optimizations][1465]
      * [Extra elimination of useless arguments][1466]
      * [Accessing opaque proofs][1467]
      * [Realizing axioms][1468]
      * [Realizing inductive types][1469]
      * [Generating FFI Code][1470]
      * [Avoiding conflicts with existing filenames][1471]
      * [Additional settings][1472]
    * [Differences between Rocq and ML type systems][1473]
    * [Some examples][1474]
      
      * [A detailed example: Euclidean division][1475]
      * [Extraction's horror museum][1476]
      * [Users' Contributions][1477]
  * [Program derivation][1478]
  * [Functional induction][1479]
    
    * [Advanced recursive functions][1480]
    * [Tactics][1481]
    * [Generation of induction principles with `Functional` `Scheme`][1482]
    * [Flags][1483]
  * [Writing Rocq libraries and plugins][1484]
    
    * [Deprecating library objects, tactics or library files][1485]
    * [Triggering warning for library objects or library files][1486]
* [Command-line and graphical tools][1487]
  
  * [Building Rocq Projects][1488]
    
    * [Rocq configuration basics][1489]
      
      * [Installing the Rocq Prover and Rocq packages with opam][1490]
      * [Setup for working on your own projects][1491]
      * [Building a project with _CoqProject (overview)][1492]
      * [Logical paths and the load path][1493]
      * [Modifying multiple interdependent projects at the same time][1494]
      * [Installed and uninstalled packages][1495]
      * [Upgrading to a new version of Rocq][1496]
    * [Building a Rocq project with rocq makefile (details)][1497]
      
      * [Comments][1498]
        
        * [Quoting arguments to rocq c][1499]
        * [Forbidden filenames][1500]
        * [Warning: No common logical root][1501]
        * [CoqMakefile.local][1502]
        * [CoqMakefile.local-late][1503]
        * [Timing targets and performance testing][1504]
        * [Building a subset of the targets with `-j`][1505]
        * [Precompiling for `native_compute`][1506]
        * [The grammar of _CoqProject][1507]
    * [Building a Rocq project with Dune][1508]
    * [rocq dep: Computing Module dependencies][1509]
    * [Split compilation of native computation files][1510]
    * [Using Rocq as a library][1511]
    * [Embedded Rocq phrases inside LaTeX documents][1512]
    * [Man pages][1513]
  * [The Rocq Prover commands][1514]
    
    * [Interactive use (rocq repl)][1515]
    * [Batch compilation (rocq compile)][1516]
    * [System configuration][1517]
    * [Customization at launch time][1518]
    * [Command parameters][1519]
      
      * [`coqrc` start up script][1520]
      * [Environment variables][1521]
      * [Command line options][1522]
    * [Profiling][1523]
    * [Compiled interfaces (produced using `-vos`)][1524]
    * [Compiled libraries checker (rocqchk)][1525]
  * [Documenting Rocq files with rocq doc][1526]
    
    * [Principles][1527]
      
      * [Rocq material inside documentation.][1528]
      * [Pretty-printing.][1529]
      * [Sections][1530]
      * [Lists.][1531]
      * [Rules.][1532]
      * [Emphasis.][1533]
      * [Escaping to LaTeX and HTML.][1534]
      * [Verbatim][1535]
      * [Hyperlinks][1536]
      * [Hiding / Showing parts of the source][1537]
    * [Usage][1538]
      
      * [Command line options][1539]
    * [The rocq doc LaTeX style file][1540]
  * [RocqIDE][1541]
    
    * [Managing files and buffers, basic editing][1542]
    * [Running Coq scripts][1543]
    * [Asynchronous mode][1544]
    * [Commands and templates][1545]
    * [Queries][1546]
    * [Compilation][1547]
    * [Customizations][1548]
      
      * [Preferences][1549]
      * [Key bindings][1550]
    * [Using Unicode symbols][1551]
      
      * [Displaying Unicode symbols][1552]
      * [Bindings for input of Unicode symbols][1553]
      * [Adding custom bindings][1554]
      * [Character encoding for saved files][1555]
    * [Debugger][1556]
      
      * [Breakpoints][1557]
      * [Call Stack and Variables][1558]
      * [Supported use cases][1559]
  * [Asynchronous and Parallel Proof Processing][1560]
    
    * [Proof annotations][1561]
      
      * [Automatic suggestion of proof annotations][1562]
    * [Proof blocks and error resilience][1563]
      
      * [Caveats][1564]
    * [Interactive mode][1565]
    * [Limiting the number of parallel workers][1566]
      
      * [Caveats][1567]

Appendix

* [History and recent changes][1568]
  
  * [Early history of Coq][1569]
    
    * [Historical roots][1570]
    * [Versions 1 to 5][1571]
      
      * [Version 1][1572]
      * [Version 2][1573]
      * [Version 3][1574]
      * [Version 4][1575]
      * [Version 5][1576]
    * [Versions 6][1577]
      
      * [Version 6.1][1578]
      * [Version 6.2][1579]
      * [Version 6.3][1580]
    * [Versions 7][1581]
      
      * [Summary of changes][1582]
      * [Details of changes in 7.0 and 7.1][1583]
        
        * [Main novelties][1584]
        * [Details of changes][1585]
          
          * [Language: new "let-in" construction][1586]
          * [Language: long names][1587]
          * [Language: miscellaneous][1588]
          * [Language: Cases][1589]
          * [Reduction][1590]
          * [New tactics][1591]
          * [Changes in existing tactics][1592]
          * [Efficiency][1593]
          * [Concrete syntax of constructions][1594]
          * [Parsing and grammar extension][1595]
          * [New commands][1596]
          * [Changes in existing commands][1597]
          * [Tools][1598]
          * [Extraction][1599]
          * [Standard library][1600]
          * [New user contributions][1601]
      * [Details of changes in 7.2][1602]
      * [Details of changes in 7.3][1603]
        
        * [Changes in 7.3.1][1604]
      * [Details of changes in 7.4][1605]
  * [Recent changes][1606]
    
    * [Version 9.1][1607]
      
      * [Summary of changes][1608]
      * [Changes in 9.1.0][1609]
        
        * [Kernel][1610]
        * [Specification language, type inference][1611]
        * [Notations][1612]
        * [Tactics][1613]
        * [Ltac language][1614]
        * [Ltac2 language][1615]
        * [SSReflect][1616]
        * [Commands and options][1617]
        * [Command-line tools][1618]
        * [RocqIDE][1619]
        * [Corelib][1620]
        * [Infrastructure and dependencies][1621]
        * [Extraction][1622]
        * [Miscellaneous][1623]
    * [Version 9.0][1624]
      
      * [Summary of changes][1625]
      * [Porting to The Rocq Prover][1626]
      * [Renaming Advice][1627]
      * [The Rocq Prover Website][1628]
      * [Changes in 9.0.0][1629]
        
        * [Kernel][1630]
        * [Specification language, type inference][1631]
        * [Notations][1632]
        * [Tactics][1633]
        * [Ltac2 language][1634]
        * [SSReflect][1635]
        * [Commands and options][1636]
        * [Command-line tools][1637]
        * [RocqIDE][1638]
        * [Standard library][1639]
        * [Infrastructure and dependencies][1640]
        * [Miscellaneous][1641]
    * [Version 8.20][1642]
      
      * [Summary of changes][1643]
      * [Changes in 8.20.0][1644]
        
        * [Kernel][1645]
        * [Specification language, type inference][1646]
        * [Notations][1647]
        * [Tactics][1648]
        * [Ltac language][1649]
        * [Ltac2 language][1650]
        * [SSReflect][1651]
        * [Commands and options][1652]
        * [Command-line tools][1653]
        * [CoqIDE][1654]
        * [Standard library][1655]
        * [Infrastructure and dependencies][1656]
        * [Extraction][1657]
      * [Changes in 8.20.1][1658]
        
        * [Kernel][1659]
        * [Notations][1660]
        * [Tactics][1661]
    * [Version 8.19][1662]
      
      * [Summary of changes][1663]
      * [Changes in 8.19.0][1664]
        
        * [Kernel][1665]
        * [Specification language, type inference][1666]
        * [Notations][1667]
        * [Tactics][1668]
        * [Ltac language][1669]
        * [Ltac2 language][1670]
        * [Commands and options][1671]
        * [Command-line tools][1672]
        * [Standard library][1673]
        * [Extraction][1674]
      * [Changes in 8.19.1][1675]
        
        * [Kernel][1676]
        * [Notations][1677]
        * [Tactics][1678]
        * [Ltac2 language][1679]
        * [Infrastructure and dependencies][1680]
      * [Changes in 8.19.2][1681]
        
        * [Specification language, type inference][1682]
        * [Notations][1683]
        * [Tactics][1684]
        * [Ltac2 language][1685]
        * [Commands and options][1686]
        * [CoqIDE][1687]
        * [Infrastructure and dependencies][1688]
    * [Version 8.18][1689]
      
      * [Summary of changes][1690]
      * [Changes in 8.18.0][1691]
        
        * [Kernel][1692]
        * [Specification language, type inference][1693]
        * [Notations][1694]
        * [Tactics][1695]
        * [Ltac2 language][1696]
        * [Commands and options][1697]
        * [Command-line tools][1698]
        * [CoqIDE][1699]
        * [Standard library][1700]
        * [Infrastructure and dependencies][1701]
        * [Extraction][1702]
    * [Version 8.17][1703]
      
      * [Summary of changes][1704]
      * [Changes in 8.17.0][1705]
        
        * [Kernel][1706]
        * [Specification language, type inference][1707]
        * [Notations][1708]
        * [Tactics][1709]
        * [Ltac language][1710]
        * [Ltac2 language][1711]
        * [SSReflect][1712]
        * [Commands and options][1713]
        * [Command-line tools][1714]
        * [Standard library][1715]
        * [Infrastructure and dependencies][1716]
        * [Miscellaneous][1717]
      * [Changes in 8.17.1][1718]
    * [Version 8.16][1719]
      
      * [Summary of changes][1720]
      * [Changes in 8.16.0][1721]
        
        * [Kernel][1722]
        * [Specification language, type inference][1723]
        * [Notations][1724]
        * [Tactics][1725]
        * [Tactic language][1726]
        * [SSReflect][1727]
        * [Commands and options][1728]
        * [Command-line tools][1729]
        * [CoqIDE][1730]
        * [Standard library][1731]
        * [Infrastructure and dependencies][1732]
        * [Extraction][1733]
      * [Changes in 8.16.1][1734]
        
        * [Kernel][1735]
        * [Commands and options][1736]
        * [CoqIDE][1737]
    * [Version 8.15][1738]
      
      * [Summary of changes][1739]
      * [Changes in 8.15.0][1740]
        
        * [Kernel][1741]
        * [Specification language, type inference][1742]
        * [Notations][1743]
        * [Tactics][1744]
        * [Tactic language][1745]
        * [SSReflect][1746]
        * [Commands and options][1747]
        * [Command-line tools][1748]
        * [CoqIDE][1749]
        * [Standard library][1750]
        * [Infrastructure and dependencies][1751]
        * [Extraction][1752]
      * [Changes in 8.15.1][1753]
        
        * [Kernel][1754]
        * [Notations][1755]
        * [Tactics][1756]
        * [Command-line tools][1757]
        * [CoqIDE][1758]
        * [Miscellaneous][1759]
      * [Changes in 8.15.2][1760]
        
        * [Tactics][1761]
        * [CoqIDE][1762]
        * [Standard library][1763]
    * [Version 8.14][1764]
      
      * [Summary of changes][1765]
      * [Changes in 8.14.0][1766]
        
        * [Kernel][1767]
        * [Specification language, type inference][1768]
        * [Notations][1769]
        * [Tactics][1770]
        * [Tactic language][1771]
        * [SSReflect][1772]
        * [Commands and options][1773]
        * [Command-line tools][1774]
        * [Native Compilation][1775]
        * [CoqIDE][1776]
        * [Standard library][1777]
        * [Infrastructure and dependencies][1778]
        * [Miscellaneous][1779]
      * [Changes in 8.14.1][1780]
        
        * [Kernel][1781]
        * [Specification language, type inference][1782]
        * [Tactics][1783]
        * [Commands and options][1784]
    * [Version 8.13][1785]
      
      * [Summary of changes][1786]
      * [Changes in 8.13+beta1][1787]
        
        * [Kernel][1788]
        * [Specification language, type inference][1789]
        * [Notations][1790]
        * [Tactics][1791]
        * [Tactic language][1792]
        * [SSReflect][1793]
        * [Commands and options][1794]
        * [Tools][1795]
        * [CoqIDE][1796]
        * [Standard library][1797]
        * [Infrastructure and dependencies][1798]
      * [Changes in 8.13.0][1799]
        
        * [Commands and options][1800]
      * [Changes in 8.13.1][1801]
        
        * [Kernel][1802]
        * [CoqIDE][1803]
      * [Changes in 8.13.2][1804]
        
        * [Kernel][1805]
        * [Tactic language][1806]
    * [Version 8.12][1807]
      
      * [Summary of changes][1808]
      * [Changes in 8.12+beta1][1809]
        
        * [Kernel][1810]
        * [Specification language, type inference][1811]
        * [Notations][1812]
        * [Tactics][1813]
        * [Tactic language][1814]
        * [SSReflect][1815]
        * [Flags, options and attributes][1816]
        * [Commands][1817]
        * [Tools][1818]
        * [CoqIDE][1819]
        * [Standard library][1820]
        * [Reals library][1821]
        * [Extraction][1822]
        * [Reference manual][1823]
        * [Infrastructure and dependencies][1824]
      * [Changes in 8.12.0][1825]
      * [Changes in 8.12.1][1826]
      * [Changes in 8.12.2][1827]
    * [Version 8.11][1828]
      
      * [Summary of changes][1829]
      * [Changes in 8.11+beta1][1830]
      * [Changes in 8.11.0][1831]
      * [Changes in 8.11.1][1832]
      * [Changes in 8.11.2][1833]
    * [Version 8.10][1834]
      
      * [Summary of changes][1835]
      * [Other changes in 8.10+beta1][1836]
      * [Changes in 8.10+beta2][1837]
      * [Changes in 8.10+beta3][1838]
      * [Changes in 8.10.0][1839]
      * [Changes in 8.10.1][1840]
      * [Changes in 8.10.2][1841]
    * [Version 8.9][1842]
      
      * [Summary of changes][1843]
      * [Details of changes in 8.9+beta1][1844]
      * [Changes in 8.8.0][1845]
      * [Changes in 8.8.1][1846]
    * [Version 8.8][1847]
      
      * [Summary of changes][1848]
      * [Details of changes in 8.8+beta1][1849]
      * [Details of changes in 8.8.0][1850]
      * [Details of changes in 8.8.1][1851]
      * [Details of changes in 8.8.2][1852]
    * [Version 8.7][1853]
      
      * [Summary of changes][1854]
      * [Potential compatibility issues][1855]
      * [Details of changes in 8.7+beta1][1856]
      * [Details of changes in 8.7+beta2][1857]
      * [Details of changes in 8.7.0][1858]
      * [Details of changes in 8.7.1][1859]
      * [Details of changes in 8.7.2][1860]
    * [Version 8.6][1861]
      
      * [Summary of changes][1862]
      * [Potential sources of incompatibilities][1863]
      * [Details of changes in 8.6beta1][1864]
      * [Details of changes in 8.6][1865]
      * [Details of changes in 8.6.1][1866]
    * [Version 8.5][1867]
      
      * [Summary of changes][1868]
      * [Potential sources of incompatibilities][1869]
      * [Details of changes in 8.5beta1][1870]
      * [Details of changes in 8.5beta2][1871]
      * [Details of changes in 8.5beta3][1872]
      * [Details of changes in 8.5][1873]
      * [Details of changes in 8.5pl1][1874]
      * [Details of changes in 8.5pl2][1875]
      * [Details of changes in 8.5pl3][1876]
    * [Version 8.4][1877]
      
      * [Summary of changes][1878]
      * [Potential sources of incompatibilities][1879]
      * [Details of changes in 8.4beta][1880]
      * [Details of changes in 8.4beta2][1881]
      * [Details of changes in 8.4][1882]
    * [Version 8.3][1883]
      
      * [Summary of changes][1884]
      * [Details of changes][1885]
    * [Version 8.2][1886]
      
      * [Summary of changes][1887]
      * [Details of changes][1888]
    * [Version 8.1][1889]
      
      * [Summary of changes][1890]
      * [Details of changes in 8.1beta][1891]
      * [Details of changes in 8.1gamma][1892]
      * [Details of changes in 8.1][1893]
    * [Version 8.0][1894]
      
      * [Summary of changes][1895]
      * [Details of changes in 8.0beta old syntax][1896]
      * [Details of changes in 8.0beta new syntax][1897]
      * [Details of changes in 8.0][1898]
* [Indexes][1899]
  
  * [Glossary index][1900]
  * [Command index][1901]
  * [Tactic index][1902]
  * [Attribute index][1903]
  * [Flags, options and tables index][1904]
  * [Errors and warnings index][1905]
  * [Index][1906]
* [Bibliography][1907]

Note

**License**

This material (the Rocq Reference Manual) may be distributed only subject to the
terms and conditions set forth in the Open Publication License, v1.0 or later
(the latest version is presently available at
[http://www.opencontent.org/openpub][1908]). Options A and B are not elected.

[Next ][1909]

© Copyright 1999-2025, Inria, CNRS and contributors.

Built with [Sphinx][1910] using a [theme][1911] provided by [Read the
Docs][1912].
Other versions v: 9.1.0

*Versions*
  [dev][1913]
  [stable][1914]
  [9.1][1915]
  [9.0][1916]
  [8.20][1917]
  [8.19][1918]
  [8.18][1919]
  [8.17][1920]
  [8.16][1921]
  [8.15][1922]
  [8.14][1923]
  [8.13][1924]
  [8.12][1925]
  [8.11][1926]
  [8.10][1927]
  [8.9][1928]
  [8.8][1929]
  [8.7][1930]
  [8.6][1931]
  [8.5][1932]
  [8.4][1933]
  [8.3][1934]
  [8.2][1935]
  [8.1][1936]
  [8.0][1937]

*Downloads*
  [PDF][1938]

[1]: #
[2]: #
[3]: language/core/index.html
[4]: language/core/basic.html
[5]: language/core/basic.html#syntax-and-lexical-conventions
[6]: language/core/basic.html#syntax-conventions
[7]: language/core/basic.html#lexical-conventions
[8]: language/core/basic.html#essential-vocabulary
[9]: language/core/basic.html#settings
[10]: language/core/basic.html#attributes
[11]: language/core/basic.html#flags-options-and-tables
[12]: language/core/sorts.html
[13]: language/core/assumptions.html
[14]: language/core/assumptions.html#binders
[15]: language/core/assumptions.html#functions-fun-and-function-types-forall
[16]: language/core/assumptions.html#function-application
[17]: language/core/assumptions.html#assumptions
[18]: language/core/definitions.html
[19]: language/core/definitions.html#let-in-definitions
[20]: language/core/definitions.html#type-cast
[21]: language/core/definitions.html#top-level-definitions
[22]: language/core/definitions.html#assertions-and-proofs
[23]: language/core/conversion.html
[24]: language/core/conversion.html#conversion
[25]: language/core/conversion.html#reduction
[26]: language/core/conversion.html#delta-reduction-sect
[27]: language/core/conversion.html#iota-reduction-sect
[28]: language/core/conversion.html#zeta-reduction-sect
[29]: language/core/conversion.html#expansion
[30]: language/core/conversion.html#examples
[31]: language/core/conversion.html#proof-irrelevance
[32]: language/core/conversion.html#convertibility
[33]: language/cic.html
[34]: language/cic.html#the-terms
[35]: language/cic.html#id4
[36]: language/cic.html#subtyping-rules
[37]: language/cic.html#the-calculus-of-inductive-constructions-with-impredicati
ve-set
[38]: language/core/variants.html
[39]: language/core/variants.html#id1
[40]: language/core/variants.html#private-matching-inductive-types
[41]: language/core/variants.html#definition-by-cases-match
[42]: language/core/records.html
[43]: language/core/records.html#defining-record-types
[44]: language/core/records.html#constructing-records
[45]: language/core/records.html#accessing-fields-projections
[46]: language/core/records.html#settings-for-printing-records
[47]: language/core/records.html#primitive-projections
[48]: language/core/records.html#reduction
[49]: language/core/records.html#compatibility-constants-for-projections
[50]: language/core/inductive.html
[51]: language/core/inductive.html#inductive-types
[52]: language/core/inductive.html#simple-inductive-types
[53]: language/core/inductive.html#simple-indexed-inductive-types
[54]: language/core/inductive.html#parameterized-inductive-types
[55]: language/core/inductive.html#mutually-defined-inductive-types
[56]: language/core/inductive.html#recursive-functions-fix
[57]: language/core/inductive.html#top-level-recursive-functions
[58]: language/core/inductive.html#theory-of-inductive-definitions
[59]: language/core/inductive.html#types-of-inductive-objects
[60]: language/core/inductive.html#well-formed-inductive-definitions
[61]: language/core/inductive.html#destructors
[62]: language/core/inductive.html#fixpoint-definitions
[63]: language/core/coinductive.html
[64]: language/core/coinductive.html#coinductive-types
[65]: language/core/coinductive.html#caveat
[66]: language/core/coinductive.html#co-recursive-functions-cofix
[67]: language/core/coinductive.html#top-level-definitions-of-corecursive-functi
ons
[68]: language/core/sections.html
[69]: language/core/sections.html#using-sections
[70]: language/core/sections.html#summary-of-locality-attributes-in-a-section
[71]: language/core/sections.html#typing-rules-used-at-the-end-of-a-section
[72]: language/core/modules.html
[73]: language/core/modules.html#modules-and-module-types
[74]: language/core/modules.html#using-modules
[75]: language/core/modules.html#examples
[76]: language/core/modules.html#qualified-names
[77]: language/core/modules.html#controlling-the-scope-of-commands-with-locality
-attributes
[78]: language/core/modules.html#summary-of-locality-attributes-in-a-module
[79]: language/core/modules.html#typing-modules
[80]: language/core/primitive.html
[81]: language/core/primitive.html#primitive-integers
[82]: language/core/primitive.html#primitive-floats
[83]: language/core/primitive.html#primitive-arrays
[84]: language/core/primitive.html#primitive-byte-based-strings
[85]: addendum/universe-polymorphism.html
[86]: addendum/universe-polymorphism.html#general-presentation
[87]: addendum/universe-polymorphism.html#polymorphic-monomorphic
[88]: addendum/universe-polymorphism.html#cumulative-noncumulative
[89]: addendum/universe-polymorphism.html#specifying-cumulativity
[90]: addendum/universe-polymorphism.html#cumulativity-weak-constraints
[91]: addendum/universe-polymorphism.html#global-and-local-universes
[92]: addendum/universe-polymorphism.html#conversion-and-unification
[93]: addendum/universe-polymorphism.html#minimization
[94]: addendum/universe-polymorphism.html#explicit-universes
[95]: addendum/universe-polymorphism.html#printing-universes
[96]: addendum/universe-polymorphism.html#polymorphic-definitions
[97]: addendum/universe-polymorphism.html#sort-polymorphism
[98]: addendum/universe-polymorphism.html#explicit-sorts
[99]: addendum/universe-polymorphism.html#universe-polymorphism-and-sections
[100]: addendum/sprop.html
[101]: addendum/sprop.html#basic-constructs
[102]: addendum/sprop.html#encodings-for-strict-propositions
[103]: addendum/sprop.html#definitional-uip
[104]: addendum/sprop.html#non-termination-with-uip
[105]: addendum/sprop.html#debugging-sprop-issues
[106]: addendum/rewrite-rules.html
[107]: addendum/rewrite-rules.html#symbols
[108]: addendum/rewrite-rules.html#id1
[109]: addendum/rewrite-rules.html#pattern-syntax
[110]: addendum/rewrite-rules.html#higher-order-pattern-holes
[111]: addendum/rewrite-rules.html#universe-polymorphic-rules
[112]: addendum/rewrite-rules.html#rewrite-rules-type-preservation-confluence-an
d-termination
[113]: addendum/rewrite-rules.html#compatibility-with-the-eta-laws
[114]: addendum/rewrite-rules.html#level-of-support
[115]: language/extensions/index.html
[116]: language/extensions/compil-steps.html
[117]: language/extensions/compil-steps.html#lexing
[118]: language/extensions/compil-steps.html#parsing
[119]: language/extensions/compil-steps.html#synterp
[120]: language/extensions/compil-steps.html#interp
[121]: language/extensions/compil-steps.html#term-level-processing
[122]: language/extensions/evars.html
[123]: language/extensions/evars.html#inferable-subterms
[124]: language/extensions/evars.html#e-tactics-that-can-create-existential-vari
ables
[125]: language/extensions/evars.html#automatic-resolution-of-existential-variab
les
[126]: language/extensions/evars.html#explicit-display-of-existential-instances-
for-pretty-printing
[127]: language/extensions/evars.html#solving-existential-variables-using-tactic
s
[128]: language/extensions/implicit-arguments.html
[129]: language/extensions/implicit-arguments.html#the-different-kinds-of-implic
it-arguments
[130]: language/extensions/implicit-arguments.html#implicit-arguments-inferable-
from-the-knowledge-of-other-arguments-of-a-function
[131]: language/extensions/implicit-arguments.html#implicit-arguments-inferable-
by-resolution
[132]: language/extensions/implicit-arguments.html#maximal-and-non-maximal-inser
tion-of-implicit-arguments
[133]: language/extensions/implicit-arguments.html#trailing-implicit-arguments
[134]: language/extensions/implicit-arguments.html#casual-use-of-implicit-argume
nts
[135]: language/extensions/implicit-arguments.html#declaration-of-implicit-argum
ents
[136]: language/extensions/implicit-arguments.html#implicit-argument-binders
[137]: language/extensions/implicit-arguments.html#mode-for-automatic-declaratio
n-of-implicit-arguments
[138]: language/extensions/implicit-arguments.html#controlling-strict-implicit-a
rguments
[139]: language/extensions/implicit-arguments.html#controlling-contextual-implic
it-arguments
[140]: language/extensions/implicit-arguments.html#controlling-reversible-patter
n-implicit-arguments
[141]: language/extensions/implicit-arguments.html#controlling-the-insertion-of-
implicit-arguments-not-followed-by-explicit-arguments
[142]: language/extensions/implicit-arguments.html#combining-manual-declaration-
and-automatic-declaration
[143]: language/extensions/implicit-arguments.html#explicit-applications
[144]: language/extensions/implicit-arguments.html#displaying-implicit-arguments
[145]: language/extensions/implicit-arguments.html#displaying-implicit-arguments
-when-pretty-printing
[146]: language/extensions/implicit-arguments.html#interaction-with-subtyping
[147]: language/extensions/implicit-arguments.html#deactivation-of-implicit-argu
ments-for-parsing
[148]: language/extensions/implicit-arguments.html#implicit-types-of-variables
[149]: language/extensions/implicit-arguments.html#implicit-generalization
[150]: language/extensions/match.html
[151]: language/extensions/match.html#variants-and-extensions-of-match
[152]: language/extensions/match.html#multiple-and-nested-pattern-matching
[153]: language/extensions/match.html#pattern-matching-on-boolean-values-the-if-
expression
[154]: language/extensions/match.html#irrefutable-patterns-the-destructuring-let
-variants
[155]: language/extensions/match.html#controlling-pretty-printing-of-match-expre
ssions
[156]: language/extensions/match.html#conventions-about-unused-pattern-matching-
variables
[157]: language/extensions/match.html#patterns
[158]: language/extensions/match.html#multiple-patterns
[159]: language/extensions/match.html#aliasing-subpatterns
[160]: language/extensions/match.html#nested-patterns
[161]: language/extensions/match.html#disjunctive-patterns
[162]: language/extensions/match.html#about-patterns-of-parametric-types
[163]: language/extensions/match.html#parameters-in-patterns
[164]: language/extensions/match.html#implicit-arguments-in-patterns
[165]: language/extensions/match.html#matching-objects-of-dependent-types
[166]: language/extensions/match.html#understanding-dependencies-in-patterns
[167]: language/extensions/match.html#when-the-elimination-predicate-must-be-pro
vided
[168]: language/extensions/match.html#dependent-pattern-matching
[169]: language/extensions/match.html#multiple-dependent-pattern-matching
[170]: language/extensions/match.html#patterns-in-in
[171]: language/extensions/match.html#using-pattern-matching-to-write-proofs
[172]: language/extensions/match.html#pattern-matching-on-inductive-objects-invo
lving-local-definitions
[173]: language/extensions/match.html#pattern-matching-and-coercions
[174]: language/extensions/match.html#when-does-the-expansion-strategy-fail
[175]: user-extensions/syntax-extensions.html
[176]: user-extensions/syntax-extensions.html#notations
[177]: user-extensions/syntax-extensions.html#basic-notations
[178]: user-extensions/syntax-extensions.html#precedences-and-associativity
[179]: user-extensions/syntax-extensions.html#complex-notations
[180]: user-extensions/syntax-extensions.html#simple-factorization-rules
[181]: user-extensions/syntax-extensions.html#use-of-notations-for-printing
[182]: user-extensions/syntax-extensions.html#the-infix-command
[183]: user-extensions/syntax-extensions.html#reserving-notations
[184]: user-extensions/syntax-extensions.html#simultaneous-definition-of-terms-a
nd-notations
[185]: user-extensions/syntax-extensions.html#enabling-and-disabling-notations
[186]: user-extensions/syntax-extensions.html#displaying-information-about-notat
ions
[187]: user-extensions/syntax-extensions.html#locating-notations
[188]: user-extensions/syntax-extensions.html#inheritance-of-the-properties-of-a
rguments-of-constants-bound-to-a-notation
[189]: user-extensions/syntax-extensions.html#notations-and-binders
[190]: user-extensions/syntax-extensions.html#notations-with-recursive-patterns
[191]: user-extensions/syntax-extensions.html#notations-with-recursive-patterns-
involving-binders
[192]: user-extensions/syntax-extensions.html#predefined-entries
[193]: user-extensions/syntax-extensions.html#custom-entries
[194]: user-extensions/syntax-extensions.html#syntax
[195]: user-extensions/syntax-extensions.html#notation-scopes
[196]: user-extensions/syntax-extensions.html#global-interpretation-rules-for-no
tations
[197]: user-extensions/syntax-extensions.html#local-interpretation-rules-for-not
ations
[198]: user-extensions/syntax-extensions.html#the-type-scope-notation-scope
[199]: user-extensions/syntax-extensions.html#the-function-scope-notation-scope
[200]: user-extensions/syntax-extensions.html#notation-scopes-used-in-the-standa
rd-library-of-rocq
[201]: user-extensions/syntax-extensions.html#displaying-information-about-scope
s
[202]: user-extensions/syntax-extensions.html#abbreviations
[203]: user-extensions/syntax-extensions.html#numbers-and-strings
[204]: user-extensions/syntax-extensions.html#number-notations
[205]: user-extensions/syntax-extensions.html#string-notations
[206]: user-extensions/syntax-extensions.html#tactic-notations
[207]: language/extensions/arguments-command.html
[208]: language/extensions/arguments-command.html#manual-declaration-of-implicit
-arguments
[209]: language/extensions/arguments-command.html#automatic-declaration-of-impli
cit-arguments
[210]: language/extensions/arguments-command.html#renaming-implicit-arguments
[211]: language/extensions/arguments-command.html#binding-arguments-to-scopes
[212]: language/extensions/arguments-command.html#effects-of-arguments-on-unfold
ing
[213]: language/extensions/arguments-command.html#bidirectionality-hints
[214]: addendum/implicit-coercions.html
[215]: addendum/implicit-coercions.html#general-presentation
[216]: addendum/implicit-coercions.html#coercion-classes
[217]: addendum/implicit-coercions.html#id1
[218]: addendum/implicit-coercions.html#reversible-coercions
[219]: addendum/implicit-coercions.html#identity-coercions
[220]: addendum/implicit-coercions.html#inheritance-graph
[221]: addendum/implicit-coercions.html#id2
[222]: addendum/implicit-coercions.html#displaying-available-coercions
[223]: addendum/implicit-coercions.html#activating-the-printing-of-coercions
[224]: addendum/implicit-coercions.html#classes-as-records
[225]: addendum/implicit-coercions.html#coercions-and-sections
[226]: addendum/implicit-coercions.html#coercions-and-modules
[227]: addendum/implicit-coercions.html#examples
[228]: addendum/type-classes.html
[229]: addendum/type-classes.html#typeclass-and-instance-declarations
[230]: addendum/type-classes.html#binding-typeclasses
[231]: addendum/type-classes.html#parameterized-instances
[232]: addendum/type-classes.html#sections-and-contexts
[233]: addendum/type-classes.html#building-hierarchies
[234]: addendum/type-classes.html#superclasses
[235]: addendum/type-classes.html#substructures
[236]: addendum/type-classes.html#command-summary
[237]: addendum/type-classes.html#typeclasses-transparent-typeclasses-opaque
[238]: addendum/type-classes.html#settings
[239]: addendum/type-classes.html#typeclasses-eauto
[240]: language/extensions/canonical.html
[241]: language/extensions/canonical.html#declaration-of-canonical-structures
[242]: language/extensions/canonical.html#notation-overloading
[243]: language/extensions/canonical.html#derived-canonical-structures
[244]: language/extensions/canonical.html#hierarchy-of-structures
[245]: language/extensions/canonical.html#compact-declaration-of-canonical-struc
tures
[246]: addendum/program.html
[247]: addendum/program.html#elaborating-programs
[248]: addendum/program.html#syntactic-control-over-equalities
[249]: addendum/program.html#program-definition
[250]: addendum/program.html#program-fixpoint
[251]: addendum/program.html#program-lemma
[252]: addendum/program.html#solving-obligations
[253]: addendum/program.html#frequently-asked-questions
[254]: proof-engine/vernacular-commands.html
[255]: proof-engine/vernacular-commands.html#displaying
[256]: proof-engine/vernacular-commands.html#query-commands
[257]: proof-engine/vernacular-commands.html#requests-to-the-environment
[258]: proof-engine/vernacular-commands.html#printing-flags
[259]: proof-engine/vernacular-commands.html#loading-files
[260]: proof-engine/vernacular-commands.html#compiled-files
[261]: proof-engine/vernacular-commands.html#load-paths
[262]: proof-engine/vernacular-commands.html#extra-dependencies
[263]: proof-engine/vernacular-commands.html#backtracking
[264]: proof-engine/vernacular-commands.html#quitting-and-debugging
[265]: proof-engine/vernacular-commands.html#controlling-display
[266]: proof-engine/vernacular-commands.html#printing-constructions-in-full
[267]: proof-engine/vernacular-commands.html#controlling-typing-flags
[268]: proof-engine/vernacular-commands.html#internal-registration-commands
[269]: proof-engine/vernacular-commands.html#exposing-constants-to-ocaml-librari
es
[270]: proof-engine/vernacular-commands.html#inlining-hints-for-the-fast-reducti
on-machines
[271]: proof-engine/vernacular-commands.html#registering-primitive-operations
[272]: proofs/writing-proofs/index.html
[273]: proofs/writing-proofs/proof-mode.html
[274]: proofs/writing-proofs/proof-mode.html#proof-state
[275]: proofs/writing-proofs/proof-mode.html#entering-and-exiting-proof-mode
[276]: proofs/writing-proofs/proof-mode.html#proof-using-options
[277]: proofs/writing-proofs/proof-mode.html#name-a-set-of-section-hypotheses-fo
r-proof-using
[278]: proofs/writing-proofs/proof-mode.html#proof-modes
[279]: proofs/writing-proofs/proof-mode.html#managing-goals
[280]: proofs/writing-proofs/proof-mode.html#focusing-goals
[281]: proofs/writing-proofs/proof-mode.html#shelving-goals
[282]: proofs/writing-proofs/proof-mode.html#reordering-goals
[283]: proofs/writing-proofs/proof-mode.html#proving-a-subgoal-as-a-separate-lem
ma-abstract
[284]: proofs/writing-proofs/proof-mode.html#requesting-information
[285]: proofs/writing-proofs/proof-mode.html#showing-differences-between-proof-s
teps
[286]: proofs/writing-proofs/proof-mode.html#how-to-enable-diffs
[287]: proofs/writing-proofs/proof-mode.html#how-diffs-are-calculated
[288]: proofs/writing-proofs/proof-mode.html#show-proof-differences
[289]: proofs/writing-proofs/proof-mode.html#delaying-solving-unification-constr
aints
[290]: proofs/writing-proofs/proof-mode.html#proof-maintenance
[291]: proofs/writing-proofs/proof-mode.html#controlling-proof-mode
[292]: proofs/writing-proofs/proof-mode.html#controlling-memory-usage
[293]: proof-engine/tactics.html
[294]: proof-engine/tactics.html#common-elements-of-tactics
[295]: proof-engine/tactics.html#reserved-keywords
[296]: proof-engine/tactics.html#invocation-of-tactics
[297]: proof-engine/tactics.html#bindings
[298]: proof-engine/tactics.html#intro-patterns
[299]: proof-engine/tactics.html#occurrence-clauses
[300]: proof-engine/tactics.html#automatic-clearing-of-hypotheses
[301]: proof-engine/tactics.html#applying-theorems
[302]: proof-engine/tactics.html#managing-the-local-context
[303]: proof-engine/tactics.html#controlling-the-proof-flow
[304]: proof-engine/tactics.html#classical-tactics
[305]: proof-engine/tactics.html#performance-oriented-tactic-variants
[306]: proofs/writing-proofs/equality.html
[307]: proofs/writing-proofs/equality.html#tactics-for-simple-equalities
[308]: proofs/writing-proofs/equality.html#rewriting-with-leibniz-and-setoid-equ
ality
[309]: proofs/writing-proofs/equality.html#rewriting-with-definitional-equality
[310]: proofs/writing-proofs/equality.html#applying-conversion-rules
[311]: proofs/writing-proofs/equality.html#fast-reduction-tactics-vm-compute-and
-native-compute
[312]: proofs/writing-proofs/equality.html#computing-in-a-term-eval-and-eval
[313]: proofs/writing-proofs/equality.html#controlling-reduction-strategies-and-
the-conversion-algorithm
[314]: proofs/writing-proofs/reasoning-inductives.html
[315]: proofs/writing-proofs/reasoning-inductives.html#applying-constructors
[316]: proofs/writing-proofs/reasoning-inductives.html#case-analysis
[317]: proofs/writing-proofs/reasoning-inductives.html#induction
[318]: proofs/writing-proofs/reasoning-inductives.html#equality-of-inductive-typ
es
[319]: proofs/writing-proofs/reasoning-inductives.html#helper-tactics
[320]: proofs/writing-proofs/reasoning-inductives.html#generation-of-induction-p
rinciples-with-scheme
[321]: proofs/writing-proofs/reasoning-inductives.html#automatic-declaration-of-
schemes
[322]: proofs/writing-proofs/reasoning-inductives.html#combined-scheme
[323]: proofs/writing-proofs/reasoning-inductives.html#generation-of-inversion-p
rinciples-with-derive-inversion
[324]: proofs/writing-proofs/reasoning-inductives.html#examples-of-dependent-des
truction-dependent-induction
[325]: proofs/writing-proofs/reasoning-inductives.html#a-larger-example
[326]: proof-engine/ssreflect-proof-language.html
[327]: proof-engine/ssreflect-proof-language.html#introduction
[328]: proof-engine/ssreflect-proof-language.html#acknowledgments
[329]: proof-engine/ssreflect-proof-language.html#usage
[330]: proof-engine/ssreflect-proof-language.html#getting-started
[331]: proof-engine/ssreflect-proof-language.html#compatibility-issues
[332]: proof-engine/ssreflect-proof-language.html#gallina-extensions
[333]: proof-engine/ssreflect-proof-language.html#pattern-assignment
[334]: proof-engine/ssreflect-proof-language.html#pattern-conditional
[335]: proof-engine/ssreflect-proof-language.html#parametric-polymorphism
[336]: proof-engine/ssreflect-proof-language.html#anonymous-arguments
[337]: proof-engine/ssreflect-proof-language.html#wildcards
[338]: proof-engine/ssreflect-proof-language.html#definitions
[339]: proof-engine/ssreflect-proof-language.html#abbreviations
[340]: proof-engine/ssreflect-proof-language.html#basic-localization
[341]: proof-engine/ssreflect-proof-language.html#basic-tactics
[342]: proof-engine/ssreflect-proof-language.html#bookkeeping
[343]: proof-engine/ssreflect-proof-language.html#the-defective-tactics
[344]: proof-engine/ssreflect-proof-language.html#discharge
[345]: proof-engine/ssreflect-proof-language.html#introduction-in-the-context
[346]: proof-engine/ssreflect-proof-language.html#generation-of-equations
[347]: proof-engine/ssreflect-proof-language.html#type-families
[348]: proof-engine/ssreflect-proof-language.html#control-flow
[349]: proof-engine/ssreflect-proof-language.html#indentation-and-bullets
[350]: proof-engine/ssreflect-proof-language.html#terminators
[351]: proof-engine/ssreflect-proof-language.html#selectors
[352]: proof-engine/ssreflect-proof-language.html#iteration
[353]: proof-engine/ssreflect-proof-language.html#localization
[354]: proof-engine/ssreflect-proof-language.html#structure
[355]: proof-engine/ssreflect-proof-language.html#rewriting
[356]: proof-engine/ssreflect-proof-language.html#an-extended-rewrite-tactic
[357]: proof-engine/ssreflect-proof-language.html#remarks-and-examples
[358]: proof-engine/ssreflect-proof-language.html#rewriting-under-binders
[359]: proof-engine/ssreflect-proof-language.html#locking-unlocking
[360]: proof-engine/ssreflect-proof-language.html#congruence
[361]: proof-engine/ssreflect-proof-language.html#contextual-patterns
[362]: proof-engine/ssreflect-proof-language.html#syntax
[363]: proof-engine/ssreflect-proof-language.html#matching-contextual-patterns
[364]: proof-engine/ssreflect-proof-language.html#examples
[365]: proof-engine/ssreflect-proof-language.html#patterns-for-recurrent-context
s
[366]: proof-engine/ssreflect-proof-language.html#views-and-reflection
[367]: proof-engine/ssreflect-proof-language.html#interpreting-eliminations
[368]: proof-engine/ssreflect-proof-language.html#interpreting-assumptions
[369]: proof-engine/ssreflect-proof-language.html#interpreting-goals
[370]: proof-engine/ssreflect-proof-language.html#boolean-reflection
[371]: proof-engine/ssreflect-proof-language.html#the-reflect-predicate
[372]: proof-engine/ssreflect-proof-language.html#general-mechanism-for-interpre
ting-goals-and-assumptions
[373]: proof-engine/ssreflect-proof-language.html#interpreting-equivalences
[374]: proof-engine/ssreflect-proof-language.html#declaring-new-hint-views
[375]: proof-engine/ssreflect-proof-language.html#multiple-views
[376]: proof-engine/ssreflect-proof-language.html#additional-view-shortcuts
[377]: proof-engine/ssreflect-proof-language.html#synopsis-and-index
[378]: proof-engine/ssreflect-proof-language.html#parameters
[379]: proof-engine/ssreflect-proof-language.html#items-and-switches
[380]: proof-engine/ssreflect-proof-language.html#tactics
[381]: proof-engine/ssreflect-proof-language.html#tacticals
[382]: proof-engine/ssreflect-proof-language.html#commands
[383]: proof-engine/ssreflect-proof-language.html#settings
[384]: proofs/automatic-tactics/index.html
[385]: proofs/automatic-tactics/logic.html
[386]: addendum/micromega.html
[387]: addendum/micromega.html#short-description-of-the-tactics
[388]: addendum/micromega.html#positivstellensatz-refutations
[389]: addendum/micromega.html#lra-a-decision-procedure-for-linear-real-and-rati
onal-arithmetic
[390]: addendum/micromega.html#lia-a-tactic-for-linear-integer-arithmetic
[391]: addendum/micromega.html#high-level-view-of-lia
[392]: addendum/micromega.html#cutting-plane-proofs
[393]: addendum/micromega.html#case-split
[394]: addendum/micromega.html#nra-a-proof-procedure-for-non-linear-arithmetic
[395]: addendum/micromega.html#nia-a-proof-procedure-for-non-linear-integer-arit
hmetic
[396]: addendum/micromega.html#psatz-a-proof-procedure-for-non-linear-arithmetic
[397]: addendum/micromega.html#zify-pre-processing-of-arithmetic-goals
[398]: addendum/ring.html
[399]: addendum/ring.html#what-does-this-tactic-do
[400]: addendum/ring.html#the-variables-map
[401]: addendum/ring.html#is-it-automatic
[402]: addendum/ring.html#concrete-usage
[403]: addendum/ring.html#adding-a-ring-structure
[404]: addendum/ring.html#how-does-it-work
[405]: addendum/ring.html#dealing-with-fields
[406]: addendum/ring.html#adding-a-new-field-structure
[407]: addendum/ring.html#history-of-ring
[408]: addendum/ring.html#discussion
[409]: addendum/nsatz.html
[410]: addendum/nsatz.html#more-about-nsatz
[411]: proofs/automatic-tactics/auto.html
[412]: proofs/automatic-tactics/auto.html#tactics
[413]: proofs/automatic-tactics/auto.html#hint-databases
[414]: proofs/automatic-tactics/auto.html#creating-hint-databases
[415]: proofs/automatic-tactics/auto.html#hint-databases-defined-in-the-rocq-sta
ndard-library
[416]: proofs/automatic-tactics/auto.html#creating-hints
[417]: proofs/automatic-tactics/auto.html#hint-locality
[418]: proofs/automatic-tactics/auto.html#setting-implicit-automation-tactics
[419]: addendum/generalized-rewriting.html
[420]: addendum/generalized-rewriting.html#introduction-to-generalized-rewriting
[421]: addendum/generalized-rewriting.html#relations-and-morphisms
[422]: addendum/generalized-rewriting.html#adding-new-relations-and-morphisms
[423]: addendum/generalized-rewriting.html#rewriting-and-nonreflexive-relations
[424]: addendum/generalized-rewriting.html#rewriting-and-nonsymmetric-relations
[425]: addendum/generalized-rewriting.html#rewriting-in-ambiguous-setoid-context
s
[426]: addendum/generalized-rewriting.html#rewriting-with-type-valued-relations
[427]: addendum/generalized-rewriting.html#declaring-rewrite-relations
[428]: addendum/generalized-rewriting.html#commands-and-tactics
[429]: addendum/generalized-rewriting.html#first-class-setoids-and-morphisms
[430]: addendum/generalized-rewriting.html#tactics-enabled-on-user-provided-rela
tions
[431]: addendum/generalized-rewriting.html#printing-relations-and-morphisms
[432]: addendum/generalized-rewriting.html#understanding-and-fixing-failed-resol
utions
[433]: addendum/generalized-rewriting.html#deprecated-syntax-and-backward-incomp
atibilities
[434]: addendum/generalized-rewriting.html#extensions
[435]: addendum/generalized-rewriting.html#rewriting-under-binders
[436]: addendum/generalized-rewriting.html#subrelations
[437]: addendum/generalized-rewriting.html#constant-unfolding-during-rewriting
[438]: addendum/generalized-rewriting.html#constant-unfolding-during-proper-inst
ance-search
[439]: addendum/generalized-rewriting.html#strategies-for-rewriting
[440]: addendum/generalized-rewriting.html#usage
[441]: addendum/generalized-rewriting.html#definitions
[442]: proofs/creating-tactics/index.html
[443]: proof-engine/ltac.html
[444]: proof-engine/ltac.html#defects
[445]: proof-engine/ltac.html#syntax
[446]: proof-engine/ltac.html#values
[447]: proof-engine/ltac.html#syntactic-values
[448]: proof-engine/ltac.html#substitution
[449]: proof-engine/ltac.html#local-definitions-let
[450]: proof-engine/ltac.html#function-construction-and-application
[451]: proof-engine/ltac.html#tactics-in-terms
[452]: proof-engine/ltac.html#goal-selectors
[453]: proof-engine/ltac.html#processing-multiple-goals
[454]: proof-engine/ltac.html#branching-and-backtracking
[455]: proof-engine/ltac.html#control-flow
[456]: proof-engine/ltac.html#sequence
[457]: proof-engine/ltac.html#do-loop
[458]: proof-engine/ltac.html#repeat-loop
[459]: proof-engine/ltac.html#catching-errors-try
[460]: proof-engine/ltac.html#conditional-branching-tryif
[461]: proof-engine/ltac.html#alternatives
[462]: proof-engine/ltac.html#branching-with-backtracking
[463]: proof-engine/ltac.html#local-application-of-tactics
[464]: proof-engine/ltac.html#first-tactic-to-succeed
[465]: proof-engine/ltac.html#solving
[466]: proof-engine/ltac.html#first-tactic-to-make-progress
[467]: proof-engine/ltac.html#detecting-progress
[468]: proof-engine/ltac.html#success-and-failure
[469]: proof-engine/ltac.html#checking-for-success-assert-succeeds
[470]: proof-engine/ltac.html#checking-for-failure-assert-fails
[471]: proof-engine/ltac.html#failing
[472]: proof-engine/ltac.html#soft-cut-once
[473]: proof-engine/ltac.html#checking-for-a-single-success-exactly-once
[474]: proof-engine/ltac.html#manipulating-values
[475]: proof-engine/ltac.html#pattern-matching-on-terms-match
[476]: proof-engine/ltac.html#pattern-matching-on-goals-and-hypotheses-match-goa
l
[477]: proof-engine/ltac.html#filling-a-term-context
[478]: proof-engine/ltac.html#generating-fresh-hypothesis-names
[479]: proof-engine/ltac.html#computing-in-a-term-eval
[480]: proof-engine/ltac.html#getting-the-type-of-a-term
[481]: proof-engine/ltac.html#manipulating-untyped-terms-type-term
[482]: proof-engine/ltac.html#counting-goals-numgoals
[483]: proof-engine/ltac.html#testing-boolean-expressions-guard
[484]: proof-engine/ltac.html#checking-properties-of-terms
[485]: proof-engine/ltac.html#timing
[486]: proof-engine/ltac.html#timeout
[487]: proof-engine/ltac.html#timing-a-tactic
[488]: proof-engine/ltac.html#timing-a-tactic-that-evaluates-to-a-term-time-cons
tr
[489]: proof-engine/ltac.html#print-identity-tactic-idtac
[490]: proof-engine/ltac.html#tactic-toplevel-definitions
[491]: proof-engine/ltac.html#defining-ltac-symbols
[492]: proof-engine/ltac.html#printing-ltac-tactics
[493]: proof-engine/ltac.html#examples-of-using-ltac
[494]: proof-engine/ltac.html#proof-that-the-natural-numbers-have-at-least-two-e
lements
[495]: proof-engine/ltac.html#proving-that-a-list-is-a-permutation-of-a-second-l
ist
[496]: proof-engine/ltac.html#deciding-intuitionistic-propositional-logic
[497]: proof-engine/ltac.html#deciding-type-isomorphisms
[498]: proof-engine/ltac.html#debugging-ltac-tactics
[499]: proof-engine/ltac.html#backtraces
[500]: proof-engine/ltac.html#tracing-execution
[501]: proof-engine/ltac.html#interactive-debugger
[502]: proof-engine/ltac.html#profiling-ltac-tactics
[503]: proof-engine/ltac.html#run-time-optimization-tactic
[504]: proof-engine/ltac2.html
[505]: proof-engine/ltac2.html#general-design
[506]: proof-engine/ltac2.html#ml-component
[507]: proof-engine/ltac2.html#overview
[508]: proof-engine/ltac2.html#type-syntax
[509]: proof-engine/ltac2.html#type-declarations
[510]: proof-engine/ltac2.html#apis
[511]: proof-engine/ltac2.html#term-syntax
[512]: proof-engine/ltac2.html#ltac2-definitions
[513]: proof-engine/ltac2.html#printing-ltac2-tactics
[514]: proof-engine/ltac2.html#reduction
[515]: proof-engine/ltac2.html#typing
[516]: proof-engine/ltac2.html#effects
[517]: proof-engine/ltac2.html#meta-programming
[518]: proof-engine/ltac2.html#id3
[519]: proof-engine/ltac2.html#quotations
[520]: proof-engine/ltac2.html#term-antiquotations
[521]: proof-engine/ltac2.html#match-over-terms
[522]: proof-engine/ltac2.html#match-over-goals
[523]: proof-engine/ltac2.html#match-on-values
[524]: proof-engine/ltac2.html#notations
[525]: proof-engine/ltac2.html#abbreviations
[526]: proof-engine/ltac2.html#defining-tactics
[527]: proof-engine/ltac2.html#syntactic-classes
[528]: proof-engine/ltac2.html#evaluation
[529]: proof-engine/ltac2.html#debug
[530]: proof-engine/ltac2.html#profiling
[531]: proof-engine/ltac2.html#compatibility-layer-with-ltac1
[532]: proof-engine/ltac2.html#ltac1-from-ltac2
[533]: proof-engine/ltac2.html#ltac2-from-ltac1
[534]: proof-engine/ltac2.html#switching-between-ltac-languages
[535]: proof-engine/ltac2.html#transition-from-ltac1
[536]: proof-engine/ltac2.html#syntax-changes
[537]: proof-engine/ltac2.html#tactic-delay
[538]: proof-engine/ltac2.html#variable-binding
[539]: proof-engine/ltac2.html#exception-catching
[540]: using/libraries/index.html
[541]: language/coq-library.html
[542]: language/coq-library.html#the-prelude
[543]: language/coq-library.html#notations
[544]: language/coq-library.html#logic
[545]: language/coq-library.html#datatypes
[546]: language/coq-library.html#specification
[547]: language/coq-library.html#basic-arithmetic
[548]: language/coq-library.html#well-founded-recursion
[549]: language/coq-library.html#tactics
[550]: language/coq-library.html#opam-repository
[551]: addendum/extraction.html
[552]: addendum/extraction.html#generating-ml-code
[553]: addendum/extraction.html#extraction-options
[554]: addendum/extraction.html#setting-the-target-language
[555]: addendum/extraction.html#inlining-and-optimizations
[556]: addendum/extraction.html#extra-elimination-of-useless-arguments
[557]: addendum/extraction.html#accessing-opaque-proofs
[558]: addendum/extraction.html#realizing-axioms
[559]: addendum/extraction.html#realizing-inductive-types
[560]: addendum/extraction.html#generating-ffi-code
[561]: addendum/extraction.html#avoiding-conflicts-with-existing-filenames
[562]: addendum/extraction.html#additional-settings
[563]: addendum/extraction.html#differences-between-rocq-and-ml-type-systems
[564]: addendum/extraction.html#some-examples
[565]: addendum/extraction.html#a-detailed-example-euclidean-division
[566]: addendum/extraction.html#extraction-s-horror-museum
[567]: addendum/extraction.html#users-contributions
[568]: addendum/miscellaneous-extensions.html
[569]: using/libraries/funind.html
[570]: using/libraries/funind.html#advanced-recursive-functions
[571]: using/libraries/funind.html#tactics
[572]: using/libraries/funind.html#generation-of-induction-principles-with-funct
ional-scheme
[573]: using/libraries/funind.html#flags
[574]: using/libraries/writing.html
[575]: using/libraries/writing.html#deprecating-library-objects-tactics-or-libra
ry-files
[576]: using/libraries/writing.html#triggering-warning-for-library-objects-or-li
brary-files
[577]: using/tools/index.html
[578]: practical-tools/utilities.html
[579]: practical-tools/utilities.html#rocq-configuration-basics
[580]: practical-tools/utilities.html#installing-the-rocq-prover-and-rocq-packag
es-with-opam
[581]: practical-tools/utilities.html#setup-for-working-on-your-own-projects
[582]: practical-tools/utilities.html#building-a-project-with-coqproject-overvie
w
[583]: practical-tools/utilities.html#logical-paths-and-the-load-path
[584]: practical-tools/utilities.html#modifying-multiple-interdependent-projects
-at-the-same-time
[585]: practical-tools/utilities.html#installed-and-uninstalled-packages
[586]: practical-tools/utilities.html#upgrading-to-a-new-version-of-rocq
[587]: practical-tools/utilities.html#building-a-rocq-project-with-rocq-makefile
-details
[588]: practical-tools/utilities.html#comments
[589]: practical-tools/utilities.html#building-a-rocq-project-with-dune
[590]: practical-tools/utilities.html#rocq-dep-computing-module-dependencies
[591]: practical-tools/utilities.html#split-compilation-of-native-computation-fi
les
[592]: practical-tools/utilities.html#using-rocq-as-a-library
[593]: practical-tools/utilities.html#embedded-rocq-phrases-inside-latex-documen
ts
[594]: practical-tools/utilities.html#man-pages
[595]: practical-tools/coq-commands.html
[596]: practical-tools/coq-commands.html#interactive-use-rocq-repl
[597]: practical-tools/coq-commands.html#batch-compilation-rocq-compile
[598]: practical-tools/coq-commands.html#system-configuration
[599]: practical-tools/coq-commands.html#customization-at-launch-time
[600]: practical-tools/coq-commands.html#command-parameters
[601]: practical-tools/coq-commands.html#coqrc-start-up-script
[602]: practical-tools/coq-commands.html#environment-variables
[603]: practical-tools/coq-commands.html#command-line-options
[604]: practical-tools/coq-commands.html#profiling
[605]: practical-tools/coq-commands.html#compiled-interfaces-produced-using-vos
[606]: practical-tools/coq-commands.html#compiled-libraries-checker-rocqchk
[607]: using/tools/coqdoc.html
[608]: using/tools/coqdoc.html#principles
[609]: using/tools/coqdoc.html#rocq-material-inside-documentation
[610]: using/tools/coqdoc.html#pretty-printing
[611]: using/tools/coqdoc.html#sections
[612]: using/tools/coqdoc.html#lists
[613]: using/tools/coqdoc.html#rules
[614]: using/tools/coqdoc.html#emphasis
[615]: using/tools/coqdoc.html#escaping-to-latex-and-html
[616]: using/tools/coqdoc.html#verbatim
[617]: using/tools/coqdoc.html#hyperlinks
[618]: using/tools/coqdoc.html#hiding-showing-parts-of-the-source
[619]: using/tools/coqdoc.html#usage
[620]: using/tools/coqdoc.html#command-line-options
[621]: using/tools/coqdoc.html#the-rocq-doc-latex-style-file
[622]: practical-tools/coqide.html
[623]: practical-tools/coqide.html#managing-files-and-buffers-basic-editing
[624]: practical-tools/coqide.html#running-coq-scripts
[625]: practical-tools/coqide.html#asynchronous-mode
[626]: practical-tools/coqide.html#commands-and-templates
[627]: practical-tools/coqide.html#queries
[628]: practical-tools/coqide.html#compilation
[629]: practical-tools/coqide.html#customizations
[630]: practical-tools/coqide.html#preferences
[631]: practical-tools/coqide.html#key-bindings
[632]: practical-tools/coqide.html#using-unicode-symbols
[633]: practical-tools/coqide.html#displaying-unicode-symbols
[634]: practical-tools/coqide.html#bindings-for-input-of-unicode-symbols
[635]: practical-tools/coqide.html#adding-custom-bindings
[636]: practical-tools/coqide.html#character-encoding-for-saved-files
[637]: practical-tools/coqide.html#debugger
[638]: practical-tools/coqide.html#breakpoints
[639]: practical-tools/coqide.html#call-stack-and-variables
[640]: practical-tools/coqide.html#supported-use-cases
[641]: addendum/parallel-proof-processing.html
[642]: addendum/parallel-proof-processing.html#proof-annotations
[643]: addendum/parallel-proof-processing.html#automatic-suggestion-of-proof-ann
otations
[644]: addendum/parallel-proof-processing.html#proof-blocks-and-error-resilience
[645]: addendum/parallel-proof-processing.html#caveats
[646]: addendum/parallel-proof-processing.html#interactive-mode
[647]: addendum/parallel-proof-processing.html#limiting-the-number-of-parallel-w
orkers
[648]: addendum/parallel-proof-processing.html#id3
[649]: appendix/history-and-changes/index.html
[650]: history.html
[651]: history.html#historical-roots
[652]: history.html#versions-1-to-5
[653]: history.html#version-1
[654]: history.html#version-2
[655]: history.html#version-3
[656]: history.html#version-4
[657]: history.html#version-5
[658]: history.html#versions-6
[659]: history.html#version-6-1
[660]: history.html#version-6-2
[661]: history.html#version-6-3
[662]: history.html#versions-7
[663]: history.html#summary-of-changes
[664]: history.html#details-of-changes-in-7-0-and-7-1
[665]: history.html#details-of-changes-in-7-2
[666]: history.html#details-of-changes-in-7-3
[667]: history.html#details-of-changes-in-7-4
[668]: changes.html
[669]: changes.html#version-9-1
[670]: changes.html#summary-of-changes
[671]: changes.html#changes-in-9-1-0
[672]: changes.html#version-9-0
[673]: changes.html#id105
[674]: changes.html#porting-to-the-rocq-prover
[675]: changes.html#renaming-advice
[676]: changes.html#the-rocq-prover-website
[677]: changes.html#changes-in-9-0-0
[678]: changes.html#version-8-20
[679]: changes.html#id222
[680]: changes.html#changes-in-8-20-0
[681]: changes.html#changes-in-8-20-1
[682]: changes.html#version-8-19
[683]: changes.html#id435
[684]: changes.html#changes-in-8-19-0
[685]: changes.html#changes-in-8-19-1
[686]: changes.html#changes-in-8-19-2
[687]: changes.html#version-8-18
[688]: changes.html#id641
[689]: changes.html#changes-in-8-18-0
[690]: changes.html#version-8-17
[691]: changes.html#id783
[692]: changes.html#changes-in-8-17-0
[693]: changes.html#changes-in-8-17-1
[694]: changes.html#version-8-16
[695]: changes.html#id943
[696]: changes.html#changes-in-8-16-0
[697]: changes.html#changes-in-8-16-1
[698]: changes.html#version-8-15
[699]: changes.html#id1106
[700]: changes.html#changes-in-8-15-0
[701]: changes.html#changes-in-8-15-1
[702]: changes.html#changes-in-8-15-2
[703]: changes.html#version-8-14
[704]: changes.html#id1334
[705]: changes.html#changes-in-8-14-0
[706]: changes.html#changes-in-8-14-1
[707]: changes.html#version-8-13
[708]: changes.html#id1530
[709]: changes.html#changes-in-8-13-beta1
[710]: changes.html#changes-in-8-13-0
[711]: changes.html#changes-in-8-13-1
[712]: changes.html#changes-in-8-13-2
[713]: changes.html#version-8-12
[714]: changes.html#id1694
[715]: changes.html#changes-in-8-12-beta1
[716]: changes.html#changes-in-8-12-0
[717]: changes.html#changes-in-8-12-1
[718]: changes.html#changes-in-8-12-2
[719]: changes.html#version-8-11
[720]: changes.html#id2000
[721]: changes.html#changes-in-8-11-beta1
[722]: changes.html#changes-in-8-11-0
[723]: changes.html#changes-in-8-11-1
[724]: changes.html#changes-in-8-11-2
[725]: changes.html#version-8-10
[726]: changes.html#id2138
[727]: changes.html#other-changes-in-8-10-beta1
[728]: changes.html#changes-in-8-10-beta2
[729]: changes.html#changes-in-8-10-beta3
[730]: changes.html#changes-in-8-10-0
[731]: changes.html#changes-in-8-10-1
[732]: changes.html#changes-in-8-10-2
[733]: changes.html#version-8-9
[734]: changes.html#id2285
[735]: changes.html#details-of-changes-in-8-9-beta1
[736]: changes.html#changes-in-8-8-0
[737]: changes.html#changes-in-8-8-1
[738]: changes.html#version-8-8
[739]: changes.html#id2287
[740]: changes.html#details-of-changes-in-8-8-beta1
[741]: changes.html#details-of-changes-in-8-8-0
[742]: changes.html#details-of-changes-in-8-8-1
[743]: changes.html#details-of-changes-in-8-8-2
[744]: changes.html#version-8-7
[745]: changes.html#id2288
[746]: changes.html#potential-compatibility-issues
[747]: changes.html#details-of-changes-in-8-7-beta1
[748]: changes.html#details-of-changes-in-8-7-beta2
[749]: changes.html#details-of-changes-in-8-7-0
[750]: changes.html#details-of-changes-in-8-7-1
[751]: changes.html#details-of-changes-in-8-7-2
[752]: changes.html#version-8-6
[753]: changes.html#id2289
[754]: changes.html#potential-sources-of-incompatibilities
[755]: changes.html#details-of-changes-in-8-6beta1
[756]: changes.html#details-of-changes-in-8-6
[757]: changes.html#details-of-changes-in-8-6-1
[758]: changes.html#version-8-5
[759]: changes.html#id2290
[760]: changes.html#id2291
[761]: changes.html#details-of-changes-in-8-5beta1
[762]: changes.html#details-of-changes-in-8-5beta2
[763]: changes.html#details-of-changes-in-8-5beta3
[764]: changes.html#details-of-changes-in-8-5
[765]: changes.html#details-of-changes-in-8-5pl1
[766]: changes.html#details-of-changes-in-8-5pl2
[767]: changes.html#details-of-changes-in-8-5pl3
[768]: changes.html#version-8-4
[769]: changes.html#id2292
[770]: changes.html#id2294
[771]: changes.html#details-of-changes-in-8-4beta
[772]: changes.html#details-of-changes-in-8-4beta2
[773]: changes.html#details-of-changes-in-8-4
[774]: changes.html#version-8-3
[775]: changes.html#id2295
[776]: changes.html#details-of-changes
[777]: changes.html#version-8-2
[778]: changes.html#id2296
[779]: changes.html#id2297
[780]: changes.html#version-8-1
[781]: changes.html#id2298
[782]: changes.html#details-of-changes-in-8-1beta
[783]: changes.html#details-of-changes-in-8-1gamma
[784]: changes.html#details-of-changes-in-8-1
[785]: changes.html#version-8-0
[786]: changes.html#id2299
[787]: changes.html#details-of-changes-in-8-0beta-old-syntax
[788]: changes.html#details-of-changes-in-8-0beta-new-syntax
[789]: changes.html#details-of-changes-in-8-0
[790]: appendix/indexes/index.html
[791]: std-glossindex.html
[792]: coq-cmdindex.html
[793]: coq-tacindex.html
[794]: coq-attrindex.html
[795]: coq-optindex.html
[796]: coq-exnindex.html
[797]: genindex.html
[798]: zebibliography.html
[799]: #
[800]: https://github.com/coq/coq/blob/master/doc/sphinx/index.rst
[801]: #introduction-and-contents
[802]: http://compcert.inria.fr/
[803]: https://github.com/math-comp/fourcolor
[804]: addendum/micromega.html#coq:tacn.lia
[805]: proof-engine/ltac.html#ltac
[806]: proof-engine/ltac2.html#ltac2
[807]: https://rocq-prover.org/docs
[808]: language/core/index.html#core-language
[809]: language/extensions/index.html#extensions
[810]: proofs/writing-proofs/index.html#writing-proofs
[811]: proofs/automatic-tactics/index.html#automatic-tactics
[812]: proofs/creating-tactics/index.html#writing-tactics
[813]: using/libraries/index.html#libraries
[814]: using/tools/index.html#tools
[815]: appendix/history-and-changes/index.html#history-and-changes
[816]: appendix/indexes/index.html#indexes
[817]: #contents
[818]: #
[819]: language/core/index.html
[820]: language/core/basic.html
[821]: language/core/basic.html#syntax-and-lexical-conventions
[822]: language/core/basic.html#syntax-conventions
[823]: language/core/basic.html#lexical-conventions
[824]: language/core/basic.html#essential-vocabulary
[825]: language/core/basic.html#settings
[826]: language/core/basic.html#attributes
[827]: language/core/basic.html#generic-attributes
[828]: language/core/basic.html#document-level-attributes
[829]: language/core/basic.html#flags-options-and-tables
[830]: language/core/basic.html#locality-attributes-supported-by-set-and-unset
[831]: language/core/sorts.html
[832]: language/core/assumptions.html
[833]: language/core/assumptions.html#binders
[834]: language/core/assumptions.html#functions-fun-and-function-types-forall
[835]: language/core/assumptions.html#function-application
[836]: language/core/assumptions.html#assumptions
[837]: language/core/definitions.html
[838]: language/core/definitions.html#let-in-definitions
[839]: language/core/definitions.html#type-cast
[840]: language/core/definitions.html#top-level-definitions
[841]: language/core/definitions.html#assertions-and-proofs
[842]: language/core/conversion.html
[843]: language/core/conversion.html#conversion
[844]: language/core/conversion.html#reduction
[845]: language/core/conversion.html#delta-reduction-sect
[846]: language/core/conversion.html#iota-reduction-sect
[847]: language/core/conversion.html#zeta-reduction-sect
[848]: language/core/conversion.html#expansion
[849]: language/core/conversion.html#examples
[850]: language/core/conversion.html#proof-irrelevance
[851]: language/core/conversion.html#convertibility
[852]: language/cic.html
[853]: language/cic.html#the-terms
[854]: language/cic.html#id4
[855]: language/cic.html#subtyping-rules
[856]: language/cic.html#the-calculus-of-inductive-constructions-with-impredicat
ive-set
[857]: language/core/variants.html
[858]: language/core/variants.html#id1
[859]: language/core/variants.html#private-matching-inductive-types
[860]: language/core/variants.html#definition-by-cases-match
[861]: language/core/records.html
[862]: language/core/records.html#defining-record-types
[863]: language/core/records.html#constructing-records
[864]: language/core/records.html#accessing-fields-projections
[865]: language/core/records.html#settings-for-printing-records
[866]: language/core/records.html#primitive-projections
[867]: language/core/records.html#reduction
[868]: language/core/records.html#compatibility-constants-for-projections
[869]: language/core/inductive.html
[870]: language/core/inductive.html#inductive-types
[871]: language/core/inductive.html#simple-inductive-types
[872]: language/core/inductive.html#automatic-prop-lowering
[873]: language/core/inductive.html#simple-indexed-inductive-types
[874]: language/core/inductive.html#parameterized-inductive-types
[875]: language/core/inductive.html#mutually-defined-inductive-types
[876]: language/core/inductive.html#recursive-functions-fix
[877]: language/core/inductive.html#top-level-recursive-functions
[878]: language/core/inductive.html#theory-of-inductive-definitions
[879]: language/core/inductive.html#types-of-inductive-objects
[880]: language/core/inductive.html#well-formed-inductive-definitions
[881]: language/core/inductive.html#arity-of-a-given-sort
[882]: language/core/inductive.html#arity
[883]: language/core/inductive.html#type-of-constructor
[884]: language/core/inductive.html#positivity-condition
[885]: language/core/inductive.html#strict-positivity
[886]: language/core/inductive.html#nested-positivity
[887]: language/core/inductive.html#correctness-rules
[888]: language/core/inductive.html#template-polymorphism
[889]: language/core/inductive.html#destructors
[890]: language/core/inductive.html#the-match-with-end-construction
[891]: language/core/inductive.html#fixpoint-definitions
[892]: language/core/inductive.html#id10
[893]: language/core/inductive.html#reduction-rule
[894]: language/core/coinductive.html
[895]: language/core/coinductive.html#coinductive-types
[896]: language/core/coinductive.html#caveat
[897]: language/core/coinductive.html#co-recursive-functions-cofix
[898]: language/core/coinductive.html#top-level-definitions-of-corecursive-funct
ions
[899]: language/core/sections.html
[900]: language/core/sections.html#using-sections
[901]: language/core/sections.html#summary-of-locality-attributes-in-a-section
[902]: language/core/sections.html#typing-rules-used-at-the-end-of-a-section
[903]: language/core/modules.html
[904]: language/core/modules.html#modules-and-module-types
[905]: language/core/modules.html#using-modules
[906]: language/core/modules.html#examples
[907]: language/core/modules.html#qualified-names
[908]: language/core/modules.html#controlling-the-scope-of-commands-with-localit
y-attributes
[909]: language/core/modules.html#summary-of-locality-attributes-in-a-module
[910]: language/core/modules.html#typing-modules
[911]: language/core/primitive.html
[912]: language/core/primitive.html#primitive-integers
[913]: language/core/primitive.html#primitive-floats
[914]: language/core/primitive.html#primitive-arrays
[915]: language/core/primitive.html#primitive-byte-based-strings
[916]: addendum/universe-polymorphism.html
[917]: addendum/universe-polymorphism.html#general-presentation
[918]: addendum/universe-polymorphism.html#polymorphic-monomorphic
[919]: addendum/universe-polymorphism.html#cumulative-noncumulative
[920]: addendum/universe-polymorphism.html#specifying-cumulativity
[921]: addendum/universe-polymorphism.html#cumulativity-weak-constraints
[922]: addendum/universe-polymorphism.html#global-and-local-universes
[923]: addendum/universe-polymorphism.html#conversion-and-unification
[924]: addendum/universe-polymorphism.html#minimization
[925]: addendum/universe-polymorphism.html#explicit-universes
[926]: addendum/universe-polymorphism.html#printing-universes
[927]: addendum/universe-polymorphism.html#polymorphic-definitions
[928]: addendum/universe-polymorphism.html#sort-polymorphism
[929]: addendum/universe-polymorphism.html#explicit-sorts
[930]: addendum/universe-polymorphism.html#universe-polymorphism-and-sections
[931]: addendum/sprop.html
[932]: addendum/sprop.html#basic-constructs
[933]: addendum/sprop.html#encodings-for-strict-propositions
[934]: addendum/sprop.html#definitional-uip
[935]: addendum/sprop.html#non-termination-with-uip
[936]: addendum/sprop.html#debugging-sprop-issues
[937]: addendum/rewrite-rules.html
[938]: addendum/rewrite-rules.html#symbols
[939]: addendum/rewrite-rules.html#id1
[940]: addendum/rewrite-rules.html#pattern-syntax
[941]: addendum/rewrite-rules.html#higher-order-pattern-holes
[942]: addendum/rewrite-rules.html#universe-polymorphic-rules
[943]: addendum/rewrite-rules.html#rewrite-rules-type-preservation-confluence-an
d-termination
[944]: addendum/rewrite-rules.html#compatibility-with-the-eta-laws
[945]: addendum/rewrite-rules.html#level-of-support
[946]: language/extensions/index.html
[947]: language/extensions/compil-steps.html
[948]: language/extensions/compil-steps.html#lexing
[949]: language/extensions/compil-steps.html#parsing
[950]: language/extensions/compil-steps.html#synterp
[951]: language/extensions/compil-steps.html#interp
[952]: language/extensions/compil-steps.html#term-level-processing
[953]: language/extensions/evars.html
[954]: language/extensions/evars.html#inferable-subterms
[955]: language/extensions/evars.html#e-tactics-that-can-create-existential-vari
ables
[956]: language/extensions/evars.html#automatic-resolution-of-existential-variab
les
[957]: language/extensions/evars.html#explicit-display-of-existential-instances-
for-pretty-printing
[958]: language/extensions/evars.html#solving-existential-variables-using-tactic
s
[959]: language/extensions/implicit-arguments.html
[960]: language/extensions/implicit-arguments.html#the-different-kinds-of-implic
it-arguments
[961]: language/extensions/implicit-arguments.html#implicit-arguments-inferable-
from-the-knowledge-of-other-arguments-of-a-function
[962]: language/extensions/implicit-arguments.html#implicit-arguments-inferable-
by-resolution
[963]: language/extensions/implicit-arguments.html#maximal-and-non-maximal-inser
tion-of-implicit-arguments
[964]: language/extensions/implicit-arguments.html#trailing-implicit-arguments
[965]: language/extensions/implicit-arguments.html#casual-use-of-implicit-argume
nts
[966]: language/extensions/implicit-arguments.html#declaration-of-implicit-argum
ents
[967]: language/extensions/implicit-arguments.html#implicit-argument-binders
[968]: language/extensions/implicit-arguments.html#mode-for-automatic-declaratio
n-of-implicit-arguments
[969]: language/extensions/implicit-arguments.html#controlling-strict-implicit-a
rguments
[970]: language/extensions/implicit-arguments.html#controlling-contextual-implic
it-arguments
[971]: language/extensions/implicit-arguments.html#controlling-reversible-patter
n-implicit-arguments
[972]: language/extensions/implicit-arguments.html#controlling-the-insertion-of-
implicit-arguments-not-followed-by-explicit-arguments
[973]: language/extensions/implicit-arguments.html#combining-manual-declaration-
and-automatic-declaration
[974]: language/extensions/implicit-arguments.html#explicit-applications
[975]: language/extensions/implicit-arguments.html#displaying-implicit-arguments
[976]: language/extensions/implicit-arguments.html#displaying-implicit-arguments
-when-pretty-printing
[977]: language/extensions/implicit-arguments.html#interaction-with-subtyping
[978]: language/extensions/implicit-arguments.html#deactivation-of-implicit-argu
ments-for-parsing
[979]: language/extensions/implicit-arguments.html#implicit-types-of-variables
[980]: language/extensions/implicit-arguments.html#implicit-generalization
[981]: language/extensions/match.html
[982]: language/extensions/match.html#variants-and-extensions-of-match
[983]: language/extensions/match.html#multiple-and-nested-pattern-matching
[984]: language/extensions/match.html#pattern-matching-on-boolean-values-the-if-
expression
[985]: language/extensions/match.html#irrefutable-patterns-the-destructuring-let
-variants
[986]: language/extensions/match.html#first-destructuring-let-syntax
[987]: language/extensions/match.html#second-destructuring-let-syntax
[988]: language/extensions/match.html#controlling-pretty-printing-of-match-expre
ssions
[989]: language/extensions/match.html#printing-nested-patterns
[990]: language/extensions/match.html#factorization-of-clauses-with-same-right-h
and-side
[991]: language/extensions/match.html#use-of-a-default-clause
[992]: language/extensions/match.html#printing-of-wildcard-patterns
[993]: language/extensions/match.html#printing-of-the-elimination-predicate
[994]: language/extensions/match.html#printing-of-hidden-subterms
[995]: language/extensions/match.html#printing-matching-on-irrefutable-patterns
[996]: language/extensions/match.html#printing-matching-on-booleans
[997]: language/extensions/match.html#conventions-about-unused-pattern-matching-
variables
[998]: language/extensions/match.html#patterns
[999]: language/extensions/match.html#multiple-patterns
[1000]: language/extensions/match.html#aliasing-subpatterns
[1001]: language/extensions/match.html#nested-patterns
[1002]: language/extensions/match.html#disjunctive-patterns
[1003]: language/extensions/match.html#about-patterns-of-parametric-types
[1004]: language/extensions/match.html#parameters-in-patterns
[1005]: language/extensions/match.html#implicit-arguments-in-patterns
[1006]: language/extensions/match.html#matching-objects-of-dependent-types
[1007]: language/extensions/match.html#understanding-dependencies-in-patterns
[1008]: language/extensions/match.html#when-the-elimination-predicate-must-be-pr
ovided
[1009]: language/extensions/match.html#dependent-pattern-matching
[1010]: language/extensions/match.html#multiple-dependent-pattern-matching
[1011]: language/extensions/match.html#patterns-in-in
[1012]: language/extensions/match.html#using-pattern-matching-to-write-proofs
[1013]: language/extensions/match.html#pattern-matching-on-inductive-objects-inv
olving-local-definitions
[1014]: language/extensions/match.html#pattern-matching-and-coercions
[1015]: language/extensions/match.html#when-does-the-expansion-strategy-fail
[1016]: user-extensions/syntax-extensions.html
[1017]: user-extensions/syntax-extensions.html#notations
[1018]: user-extensions/syntax-extensions.html#basic-notations
[1019]: user-extensions/syntax-extensions.html#precedences-and-associativity
[1020]: user-extensions/syntax-extensions.html#complex-notations
[1021]: user-extensions/syntax-extensions.html#simple-factorization-rules
[1022]: user-extensions/syntax-extensions.html#use-of-notations-for-printing
[1023]: user-extensions/syntax-extensions.html#the-infix-command
[1024]: user-extensions/syntax-extensions.html#reserving-notations
[1025]: user-extensions/syntax-extensions.html#simultaneous-definition-of-terms-
and-notations
[1026]: user-extensions/syntax-extensions.html#enabling-and-disabling-notations
[1027]: user-extensions/syntax-extensions.html#displaying-information-about-nota
tions
[1028]: user-extensions/syntax-extensions.html#locating-notations
[1029]: user-extensions/syntax-extensions.html#inheritance-of-the-properties-of-
arguments-of-constants-bound-to-a-notation
[1030]: user-extensions/syntax-extensions.html#notations-and-binders
[1031]: user-extensions/syntax-extensions.html#binders-bound-in-the-notation-and
-parsed-as-identifiers
[1032]: user-extensions/syntax-extensions.html#binders-bound-in-the-notation-and
-parsed-as-patterns
[1033]: user-extensions/syntax-extensions.html#binders-bound-in-the-notation-and
-parsed-as-terms
[1034]: user-extensions/syntax-extensions.html#binders-bound-in-the-notation-and
-parsed-as-general-binders
[1035]: user-extensions/syntax-extensions.html#binders-not-bound-in-the-notation
[1036]: user-extensions/syntax-extensions.html#notations-with-expressions-used-b
oth-as-binder-and-term
[1037]: user-extensions/syntax-extensions.html#notations-with-recursive-patterns
[1038]: user-extensions/syntax-extensions.html#notations-with-recursive-patterns
-involving-binders
[1039]: user-extensions/syntax-extensions.html#predefined-entries
[1040]: user-extensions/syntax-extensions.html#custom-entries
[1041]: user-extensions/syntax-extensions.html#syntax
[1042]: user-extensions/syntax-extensions.html#notation-scopes
[1043]: user-extensions/syntax-extensions.html#global-interpretation-rules-for-n
otations
[1044]: user-extensions/syntax-extensions.html#local-interpretation-rules-for-no
tations
[1045]: user-extensions/syntax-extensions.html#opening-a-notation-scope-locally
[1046]: user-extensions/syntax-extensions.html#binding-types-or-coercion-classes
-to-notation-scopes
[1047]: user-extensions/syntax-extensions.html#the-type-scope-notation-scope
[1048]: user-extensions/syntax-extensions.html#the-function-scope-notation-scope
[1049]: user-extensions/syntax-extensions.html#notation-scopes-used-in-the-stand
ard-library-of-rocq
[1050]: user-extensions/syntax-extensions.html#displaying-information-about-scop
es
[1051]: user-extensions/syntax-extensions.html#abbreviations
[1052]: user-extensions/syntax-extensions.html#numbers-and-strings
[1053]: user-extensions/syntax-extensions.html#number-notations
[1054]: user-extensions/syntax-extensions.html#string-notations
[1055]: user-extensions/syntax-extensions.html#tactic-notations
[1056]: language/extensions/arguments-command.html
[1057]: language/extensions/arguments-command.html#manual-declaration-of-implici
t-arguments
[1058]: language/extensions/arguments-command.html#automatic-declaration-of-impl
icit-arguments
[1059]: language/extensions/arguments-command.html#renaming-implicit-arguments
[1060]: language/extensions/arguments-command.html#binding-arguments-to-scopes
[1061]: language/extensions/arguments-command.html#effects-of-arguments-on-unfol
ding
[1062]: language/extensions/arguments-command.html#bidirectionality-hints
[1063]: addendum/implicit-coercions.html
[1064]: addendum/implicit-coercions.html#general-presentation
[1065]: addendum/implicit-coercions.html#coercion-classes
[1066]: addendum/implicit-coercions.html#id1
[1067]: addendum/implicit-coercions.html#reversible-coercions
[1068]: addendum/implicit-coercions.html#identity-coercions
[1069]: addendum/implicit-coercions.html#inheritance-graph
[1070]: addendum/implicit-coercions.html#id2
[1071]: addendum/implicit-coercions.html#displaying-available-coercions
[1072]: addendum/implicit-coercions.html#activating-the-printing-of-coercions
[1073]: addendum/implicit-coercions.html#classes-as-records
[1074]: addendum/implicit-coercions.html#coercions-and-sections
[1075]: addendum/implicit-coercions.html#coercions-and-modules
[1076]: addendum/implicit-coercions.html#examples
[1077]: addendum/type-classes.html
[1078]: addendum/type-classes.html#typeclass-and-instance-declarations
[1079]: addendum/type-classes.html#binding-typeclasses
[1080]: addendum/type-classes.html#parameterized-instances
[1081]: addendum/type-classes.html#sections-and-contexts
[1082]: addendum/type-classes.html#building-hierarchies
[1083]: addendum/type-classes.html#superclasses
[1084]: addendum/type-classes.html#substructures
[1085]: addendum/type-classes.html#command-summary
[1086]: addendum/type-classes.html#typeclasses-transparent-typeclasses-opaque
[1087]: addendum/type-classes.html#settings
[1088]: addendum/type-classes.html#typeclasses-eauto
[1089]: language/extensions/canonical.html
[1090]: language/extensions/canonical.html#declaration-of-canonical-structures
[1091]: language/extensions/canonical.html#notation-overloading
[1092]: language/extensions/canonical.html#derived-canonical-structures
[1093]: language/extensions/canonical.html#hierarchy-of-structures
[1094]: language/extensions/canonical.html#compact-declaration-of-canonical-stru
ctures
[1095]: addendum/program.html
[1096]: addendum/program.html#elaborating-programs
[1097]: addendum/program.html#syntactic-control-over-equalities
[1098]: addendum/program.html#program-definition
[1099]: addendum/program.html#program-fixpoint
[1100]: addendum/program.html#program-lemma
[1101]: addendum/program.html#solving-obligations
[1102]: addendum/program.html#frequently-asked-questions
[1103]: proof-engine/vernacular-commands.html
[1104]: proof-engine/vernacular-commands.html#displaying
[1105]: proof-engine/vernacular-commands.html#query-commands
[1106]: proof-engine/vernacular-commands.html#requests-to-the-environment
[1107]: proof-engine/vernacular-commands.html#printing-flags
[1108]: proof-engine/vernacular-commands.html#loading-files
[1109]: proof-engine/vernacular-commands.html#compiled-files
[1110]: proof-engine/vernacular-commands.html#load-paths
[1111]: proof-engine/vernacular-commands.html#extra-dependencies
[1112]: proof-engine/vernacular-commands.html#backtracking
[1113]: proof-engine/vernacular-commands.html#quitting-and-debugging
[1114]: proof-engine/vernacular-commands.html#controlling-display
[1115]: proof-engine/vernacular-commands.html#printing-constructions-in-full
[1116]: proof-engine/vernacular-commands.html#controlling-typing-flags
[1117]: proof-engine/vernacular-commands.html#internal-registration-commands
[1118]: proof-engine/vernacular-commands.html#exposing-constants-to-ocaml-librar
ies
[1119]: proof-engine/vernacular-commands.html#inlining-hints-for-the-fast-reduct
ion-machines
[1120]: proof-engine/vernacular-commands.html#registering-primitive-operations
[1121]: proofs/writing-proofs/index.html
[1122]: proofs/writing-proofs/proof-mode.html
[1123]: proofs/writing-proofs/proof-mode.html#proof-state
[1124]: proofs/writing-proofs/proof-mode.html#entering-and-exiting-proof-mode
[1125]: proofs/writing-proofs/proof-mode.html#proof-using-options
[1126]: proofs/writing-proofs/proof-mode.html#name-a-set-of-section-hypotheses-f
or-proof-using
[1127]: proofs/writing-proofs/proof-mode.html#proof-modes
[1128]: proofs/writing-proofs/proof-mode.html#managing-goals
[1129]: proofs/writing-proofs/proof-mode.html#focusing-goals
[1130]: proofs/writing-proofs/proof-mode.html#curly-braces
[1131]: proofs/writing-proofs/proof-mode.html#bullets
[1132]: proofs/writing-proofs/proof-mode.html#other-focusing-commands
[1133]: proofs/writing-proofs/proof-mode.html#shelving-goals
[1134]: proofs/writing-proofs/proof-mode.html#reordering-goals
[1135]: proofs/writing-proofs/proof-mode.html#proving-a-subgoal-as-a-separate-le
mma-abstract
[1136]: proofs/writing-proofs/proof-mode.html#requesting-information
[1137]: proofs/writing-proofs/proof-mode.html#showing-differences-between-proof-
steps
[1138]: proofs/writing-proofs/proof-mode.html#how-to-enable-diffs
[1139]: proofs/writing-proofs/proof-mode.html#how-diffs-are-calculated
[1140]: proofs/writing-proofs/proof-mode.html#show-proof-differences
[1141]: proofs/writing-proofs/proof-mode.html#delaying-solving-unification-const
raints
[1142]: proofs/writing-proofs/proof-mode.html#proof-maintenance
[1143]: proofs/writing-proofs/proof-mode.html#controlling-proof-mode
[1144]: proofs/writing-proofs/proof-mode.html#controlling-memory-usage
[1145]: proof-engine/tactics.html
[1146]: proof-engine/tactics.html#common-elements-of-tactics
[1147]: proof-engine/tactics.html#reserved-keywords
[1148]: proof-engine/tactics.html#invocation-of-tactics
[1149]: proof-engine/tactics.html#bindings
[1150]: proof-engine/tactics.html#intro-patterns
[1151]: proof-engine/tactics.html#occurrence-clauses
[1152]: proof-engine/tactics.html#automatic-clearing-of-hypotheses
[1153]: proof-engine/tactics.html#applying-theorems
[1154]: proof-engine/tactics.html#managing-the-local-context
[1155]: proof-engine/tactics.html#controlling-the-proof-flow
[1156]: proof-engine/tactics.html#classical-tactics
[1157]: proof-engine/tactics.html#performance-oriented-tactic-variants
[1158]: proofs/writing-proofs/equality.html
[1159]: proofs/writing-proofs/equality.html#tactics-for-simple-equalities
[1160]: proofs/writing-proofs/equality.html#rewriting-with-leibniz-and-setoid-eq
uality
[1161]: proofs/writing-proofs/equality.html#rewriting-with-definitional-equality
[1162]: proofs/writing-proofs/equality.html#applying-conversion-rules
[1163]: proofs/writing-proofs/equality.html#fast-reduction-tactics-vm-compute-an
d-native-compute
[1164]: proofs/writing-proofs/equality.html#computing-in-a-term-eval-and-eval
[1165]: proofs/writing-proofs/equality.html#controlling-reduction-strategies-and
-the-conversion-algorithm
[1166]: proofs/writing-proofs/reasoning-inductives.html
[1167]: proofs/writing-proofs/reasoning-inductives.html#applying-constructors
[1168]: proofs/writing-proofs/reasoning-inductives.html#case-analysis
[1169]: proofs/writing-proofs/reasoning-inductives.html#induction
[1170]: proofs/writing-proofs/reasoning-inductives.html#equality-of-inductive-ty
pes
[1171]: proofs/writing-proofs/reasoning-inductives.html#helper-tactics
[1172]: proofs/writing-proofs/reasoning-inductives.html#generation-of-induction-
principles-with-scheme
[1173]: proofs/writing-proofs/reasoning-inductives.html#automatic-declaration-of
-schemes
[1174]: proofs/writing-proofs/reasoning-inductives.html#combined-scheme
[1175]: proofs/writing-proofs/reasoning-inductives.html#generation-of-inversion-
principles-with-derive-inversion
[1176]: proofs/writing-proofs/reasoning-inductives.html#examples-of-dependent-de
struction-dependent-induction
[1177]: proofs/writing-proofs/reasoning-inductives.html#a-larger-example
[1178]: proof-engine/ssreflect-proof-language.html
[1179]: proof-engine/ssreflect-proof-language.html#introduction
[1180]: proof-engine/ssreflect-proof-language.html#acknowledgments
[1181]: proof-engine/ssreflect-proof-language.html#usage
[1182]: proof-engine/ssreflect-proof-language.html#getting-started
[1183]: proof-engine/ssreflect-proof-language.html#compatibility-issues
[1184]: proof-engine/ssreflect-proof-language.html#gallina-extensions
[1185]: proof-engine/ssreflect-proof-language.html#pattern-assignment
[1186]: proof-engine/ssreflect-proof-language.html#pattern-conditional
[1187]: proof-engine/ssreflect-proof-language.html#parametric-polymorphism
[1188]: proof-engine/ssreflect-proof-language.html#anonymous-arguments
[1189]: proof-engine/ssreflect-proof-language.html#wildcards
[1190]: proof-engine/ssreflect-proof-language.html#definitions
[1191]: proof-engine/ssreflect-proof-language.html#abbreviations
[1192]: proof-engine/ssreflect-proof-language.html#matching
[1193]: proof-engine/ssreflect-proof-language.html#occurrence-selection
[1194]: proof-engine/ssreflect-proof-language.html#basic-localization
[1195]: proof-engine/ssreflect-proof-language.html#basic-tactics
[1196]: proof-engine/ssreflect-proof-language.html#bookkeeping
[1197]: proof-engine/ssreflect-proof-language.html#the-defective-tactics
[1198]: proof-engine/ssreflect-proof-language.html#the-move-tactic
[1199]: proof-engine/ssreflect-proof-language.html#the-case-tactic
[1200]: proof-engine/ssreflect-proof-language.html#the-elim-tactic
[1201]: proof-engine/ssreflect-proof-language.html#the-apply-tactic
[1202]: proof-engine/ssreflect-proof-language.html#discharge
[1203]: proof-engine/ssreflect-proof-language.html#clear-rules
[1204]: proof-engine/ssreflect-proof-language.html#matching-for-apply-and-exact
[1205]: proof-engine/ssreflect-proof-language.html#the-abstract-tactic
[1206]: proof-engine/ssreflect-proof-language.html#introduction-in-the-context
[1207]: proof-engine/ssreflect-proof-language.html#simplification-items
[1208]: proof-engine/ssreflect-proof-language.html#views
[1209]: proof-engine/ssreflect-proof-language.html#intro-patterns
[1210]: proof-engine/ssreflect-proof-language.html#clear-switch
[1211]: proof-engine/ssreflect-proof-language.html#branching-and-destructuring
[1212]: proof-engine/ssreflect-proof-language.html#block-introduction
[1213]: proof-engine/ssreflect-proof-language.html#generation-of-equations
[1214]: proof-engine/ssreflect-proof-language.html#type-families
[1215]: proof-engine/ssreflect-proof-language.html#control-flow
[1216]: proof-engine/ssreflect-proof-language.html#indentation-and-bullets
[1217]: proof-engine/ssreflect-proof-language.html#terminators
[1218]: proof-engine/ssreflect-proof-language.html#selectors
[1219]: proof-engine/ssreflect-proof-language.html#iteration
[1220]: proof-engine/ssreflect-proof-language.html#localization
[1221]: proof-engine/ssreflect-proof-language.html#structure
[1222]: proof-engine/ssreflect-proof-language.html#the-have-tactic
[1223]: proof-engine/ssreflect-proof-language.html#generating-let-in-context-ent
ries-with-have
[1224]: proof-engine/ssreflect-proof-language.html#the-have-tactic-and-typeclass
-resolution
[1225]: proof-engine/ssreflect-proof-language.html#variants-the-suff-and-wlog-ta
ctics
[1226]: proof-engine/ssreflect-proof-language.html#advanced-generalization
[1227]: proof-engine/ssreflect-proof-language.html#rewriting
[1228]: proof-engine/ssreflect-proof-language.html#an-extended-rewrite-tactic
[1229]: proof-engine/ssreflect-proof-language.html#remarks-and-examples
[1230]: proof-engine/ssreflect-proof-language.html#rewrite-redex-selection
[1231]: proof-engine/ssreflect-proof-language.html#chained-rewrite-steps
[1232]: proof-engine/ssreflect-proof-language.html#explicit-redex-switches-are-m
atched-first
[1233]: proof-engine/ssreflect-proof-language.html#occurrence-switches-and-redex
-switches
[1234]: proof-engine/ssreflect-proof-language.html#occurrence-selection-and-repe
tition
[1235]: proof-engine/ssreflect-proof-language.html#multi-rule-rewriting
[1236]: proof-engine/ssreflect-proof-language.html#wildcards-vs-abstractions
[1237]: proof-engine/ssreflect-proof-language.html#when-ssr-rewrite-fails-on-sta
ndard-rocq-licit-rewrite
[1238]: proof-engine/ssreflect-proof-language.html#existential-metavariables-and
-rewriting
[1239]: proof-engine/ssreflect-proof-language.html#rewriting-under-binders
[1240]: proof-engine/ssreflect-proof-language.html#the-under-tactic
[1241]: proof-engine/ssreflect-proof-language.html#interactive-mode
[1242]: proof-engine/ssreflect-proof-language.html#the-over-tactic
[1243]: proof-engine/ssreflect-proof-language.html#one-liner-mode
[1244]: proof-engine/ssreflect-proof-language.html#locking-unlocking
[1245]: proof-engine/ssreflect-proof-language.html#congruence
[1246]: proof-engine/ssreflect-proof-language.html#contextual-patterns
[1247]: proof-engine/ssreflect-proof-language.html#syntax
[1248]: proof-engine/ssreflect-proof-language.html#matching-contextual-patterns
[1249]: proof-engine/ssreflect-proof-language.html#examples
[1250]: proof-engine/ssreflect-proof-language.html#contextual-pattern-in-set-and
-the-tactical
[1251]: proof-engine/ssreflect-proof-language.html#contextual-patterns-in-rewrit
e
[1252]: proof-engine/ssreflect-proof-language.html#patterns-for-recurrent-contex
ts
[1253]: proof-engine/ssreflect-proof-language.html#views-and-reflection
[1254]: proof-engine/ssreflect-proof-language.html#interpreting-eliminations
[1255]: proof-engine/ssreflect-proof-language.html#interpreting-assumptions
[1256]: proof-engine/ssreflect-proof-language.html#specializing-assumptions
[1257]: proof-engine/ssreflect-proof-language.html#interpreting-goals
[1258]: proof-engine/ssreflect-proof-language.html#boolean-reflection
[1259]: proof-engine/ssreflect-proof-language.html#the-reflect-predicate
[1260]: proof-engine/ssreflect-proof-language.html#general-mechanism-for-interpr
eting-goals-and-assumptions
[1261]: proof-engine/ssreflect-proof-language.html#id15
[1262]: proof-engine/ssreflect-proof-language.html#id16
[1263]: proof-engine/ssreflect-proof-language.html#id17
[1264]: proof-engine/ssreflect-proof-language.html#interpreting-equivalences
[1265]: proof-engine/ssreflect-proof-language.html#declaring-new-hint-views
[1266]: proof-engine/ssreflect-proof-language.html#multiple-views
[1267]: proof-engine/ssreflect-proof-language.html#additional-view-shortcuts
[1268]: proof-engine/ssreflect-proof-language.html#synopsis-and-index
[1269]: proof-engine/ssreflect-proof-language.html#parameters
[1270]: proof-engine/ssreflect-proof-language.html#items-and-switches
[1271]: proof-engine/ssreflect-proof-language.html#tactics
[1272]: proof-engine/ssreflect-proof-language.html#tacticals
[1273]: proof-engine/ssreflect-proof-language.html#commands
[1274]: proof-engine/ssreflect-proof-language.html#settings
[1275]: proofs/automatic-tactics/index.html
[1276]: proofs/automatic-tactics/logic.html
[1277]: addendum/micromega.html
[1278]: addendum/micromega.html#short-description-of-the-tactics
[1279]: addendum/micromega.html#positivstellensatz-refutations
[1280]: addendum/micromega.html#lra-a-decision-procedure-for-linear-real-and-rat
ional-arithmetic
[1281]: addendum/micromega.html#lia-a-tactic-for-linear-integer-arithmetic
[1282]: addendum/micromega.html#high-level-view-of-lia
[1283]: addendum/micromega.html#cutting-plane-proofs
[1284]: addendum/micromega.html#case-split
[1285]: addendum/micromega.html#nra-a-proof-procedure-for-non-linear-arithmetic
[1286]: addendum/micromega.html#nia-a-proof-procedure-for-non-linear-integer-ari
thmetic
[1287]: addendum/micromega.html#psatz-a-proof-procedure-for-non-linear-arithmeti
c
[1288]: addendum/micromega.html#zify-pre-processing-of-arithmetic-goals
[1289]: addendum/ring.html
[1290]: addendum/ring.html#what-does-this-tactic-do
[1291]: addendum/ring.html#the-variables-map
[1292]: addendum/ring.html#is-it-automatic
[1293]: addendum/ring.html#concrete-usage
[1294]: addendum/ring.html#adding-a-ring-structure
[1295]: addendum/ring.html#how-does-it-work
[1296]: addendum/ring.html#dealing-with-fields
[1297]: addendum/ring.html#adding-a-new-field-structure
[1298]: addendum/ring.html#history-of-ring
[1299]: addendum/ring.html#discussion
[1300]: addendum/nsatz.html
[1301]: addendum/nsatz.html#more-about-nsatz
[1302]: proofs/automatic-tactics/auto.html
[1303]: proofs/automatic-tactics/auto.html#tactics
[1304]: proofs/automatic-tactics/auto.html#hint-databases
[1305]: proofs/automatic-tactics/auto.html#creating-hint-databases
[1306]: proofs/automatic-tactics/auto.html#hint-databases-defined-in-the-rocq-st
andard-library
[1307]: proofs/automatic-tactics/auto.html#creating-hints
[1308]: proofs/automatic-tactics/auto.html#hint-locality
[1309]: proofs/automatic-tactics/auto.html#setting-implicit-automation-tactics
[1310]: addendum/generalized-rewriting.html
[1311]: addendum/generalized-rewriting.html#introduction-to-generalized-rewritin
g
[1312]: addendum/generalized-rewriting.html#relations-and-morphisms
[1313]: addendum/generalized-rewriting.html#adding-new-relations-and-morphisms
[1314]: addendum/generalized-rewriting.html#rewriting-and-nonreflexive-relations
[1315]: addendum/generalized-rewriting.html#rewriting-and-nonsymmetric-relations
[1316]: addendum/generalized-rewriting.html#rewriting-in-ambiguous-setoid-contex
ts
[1317]: addendum/generalized-rewriting.html#rewriting-with-type-valued-relations
[1318]: addendum/generalized-rewriting.html#declaring-rewrite-relations
[1319]: addendum/generalized-rewriting.html#commands-and-tactics
[1320]: addendum/generalized-rewriting.html#first-class-setoids-and-morphisms
[1321]: addendum/generalized-rewriting.html#tactics-enabled-on-user-provided-rel
ations
[1322]: addendum/generalized-rewriting.html#printing-relations-and-morphisms
[1323]: addendum/generalized-rewriting.html#understanding-and-fixing-failed-reso
lutions
[1324]: addendum/generalized-rewriting.html#deprecated-syntax-and-backward-incom
patibilities
[1325]: addendum/generalized-rewriting.html#extensions
[1326]: addendum/generalized-rewriting.html#rewriting-under-binders
[1327]: addendum/generalized-rewriting.html#subrelations
[1328]: addendum/generalized-rewriting.html#constant-unfolding-during-rewriting
[1329]: addendum/generalized-rewriting.html#constant-unfolding-during-proper-ins
tance-search
[1330]: addendum/generalized-rewriting.html#strategies-for-rewriting
[1331]: addendum/generalized-rewriting.html#usage
[1332]: addendum/generalized-rewriting.html#definitions
[1333]: proofs/creating-tactics/index.html
[1334]: proof-engine/ltac.html
[1335]: proof-engine/ltac.html#defects
[1336]: proof-engine/ltac.html#syntax
[1337]: proof-engine/ltac.html#values
[1338]: proof-engine/ltac.html#syntactic-values
[1339]: proof-engine/ltac.html#substitution
[1340]: proof-engine/ltac.html#local-definitions-let
[1341]: proof-engine/ltac.html#function-construction-and-application
[1342]: proof-engine/ltac.html#tactics-in-terms
[1343]: proof-engine/ltac.html#goal-selectors
[1344]: proof-engine/ltac.html#processing-multiple-goals
[1345]: proof-engine/ltac.html#branching-and-backtracking
[1346]: proof-engine/ltac.html#control-flow
[1347]: proof-engine/ltac.html#sequence
[1348]: proof-engine/ltac.html#do-loop
[1349]: proof-engine/ltac.html#repeat-loop
[1350]: proof-engine/ltac.html#catching-errors-try
[1351]: proof-engine/ltac.html#conditional-branching-tryif
[1352]: proof-engine/ltac.html#alternatives
[1353]: proof-engine/ltac.html#branching-with-backtracking
[1354]: proof-engine/ltac.html#local-application-of-tactics
[1355]: proof-engine/ltac.html#first-tactic-to-succeed
[1356]: proof-engine/ltac.html#solving
[1357]: proof-engine/ltac.html#first-tactic-to-make-progress
[1358]: proof-engine/ltac.html#detecting-progress
[1359]: proof-engine/ltac.html#success-and-failure
[1360]: proof-engine/ltac.html#checking-for-success-assert-succeeds
[1361]: proof-engine/ltac.html#checking-for-failure-assert-fails
[1362]: proof-engine/ltac.html#failing
[1363]: proof-engine/ltac.html#soft-cut-once
[1364]: proof-engine/ltac.html#checking-for-a-single-success-exactly-once
[1365]: proof-engine/ltac.html#manipulating-values
[1366]: proof-engine/ltac.html#pattern-matching-on-terms-match
[1367]: proof-engine/ltac.html#pattern-matching-on-goals-and-hypotheses-match-go
al
[1368]: proof-engine/ltac.html#filling-a-term-context
[1369]: proof-engine/ltac.html#generating-fresh-hypothesis-names
[1370]: proof-engine/ltac.html#computing-in-a-term-eval
[1371]: proof-engine/ltac.html#getting-the-type-of-a-term
[1372]: proof-engine/ltac.html#manipulating-untyped-terms-type-term
[1373]: proof-engine/ltac.html#counting-goals-numgoals
[1374]: proof-engine/ltac.html#testing-boolean-expressions-guard
[1375]: proof-engine/ltac.html#checking-properties-of-terms
[1376]: proof-engine/ltac.html#timing
[1377]: proof-engine/ltac.html#timeout
[1378]: proof-engine/ltac.html#timing-a-tactic
[1379]: proof-engine/ltac.html#timing-a-tactic-that-evaluates-to-a-term-time-con
str
[1380]: proof-engine/ltac.html#print-identity-tactic-idtac
[1381]: proof-engine/ltac.html#tactic-toplevel-definitions
[1382]: proof-engine/ltac.html#defining-ltac-symbols
[1383]: proof-engine/ltac.html#printing-ltac-tactics
[1384]: proof-engine/ltac.html#examples-of-using-ltac
[1385]: proof-engine/ltac.html#proof-that-the-natural-numbers-have-at-least-two-
elements
[1386]: proof-engine/ltac.html#proving-that-a-list-is-a-permutation-of-a-second-
list
[1387]: proof-engine/ltac.html#deciding-intuitionistic-propositional-logic
[1388]: proof-engine/ltac.html#deciding-type-isomorphisms
[1389]: proof-engine/ltac.html#debugging-ltac-tactics
[1390]: proof-engine/ltac.html#backtraces
[1391]: proof-engine/ltac.html#tracing-execution
[1392]: proof-engine/ltac.html#interactive-debugger
[1393]: proof-engine/ltac.html#profiling-ltac-tactics
[1394]: proof-engine/ltac.html#run-time-optimization-tactic
[1395]: proof-engine/ltac2.html
[1396]: proof-engine/ltac2.html#general-design
[1397]: proof-engine/ltac2.html#ml-component
[1398]: proof-engine/ltac2.html#overview
[1399]: proof-engine/ltac2.html#type-syntax
[1400]: proof-engine/ltac2.html#type-declarations
[1401]: proof-engine/ltac2.html#apis
[1402]: proof-engine/ltac2.html#term-syntax
[1403]: proof-engine/ltac2.html#ltac2-definitions
[1404]: proof-engine/ltac2.html#printing-ltac2-tactics
[1405]: proof-engine/ltac2.html#reduction
[1406]: proof-engine/ltac2.html#typing
[1407]: proof-engine/ltac2.html#effects
[1408]: proof-engine/ltac2.html#standard-io
[1409]: proof-engine/ltac2.html#fatal-errors
[1410]: proof-engine/ltac2.html#backtracking
[1411]: proof-engine/ltac2.html#goals
[1412]: proof-engine/ltac2.html#meta-programming
[1413]: proof-engine/ltac2.html#id3
[1414]: proof-engine/ltac2.html#quotations
[1415]: proof-engine/ltac2.html#built-in-quotations
[1416]: proof-engine/ltac2.html#strict-vs-non-strict-mode
[1417]: proof-engine/ltac2.html#term-antiquotations
[1418]: proof-engine/ltac2.html#syntax
[1419]: proof-engine/ltac2.html#semantics
[1420]: proof-engine/ltac2.html#static-semantics
[1421]: proof-engine/ltac2.html#dynamic-semantics
[1422]: proof-engine/ltac2.html#match-over-terms
[1423]: proof-engine/ltac2.html#match-over-goals
[1424]: proof-engine/ltac2.html#match-on-values
[1425]: proof-engine/ltac2.html#notations
[1426]: proof-engine/ltac2.html#abbreviations
[1427]: proof-engine/ltac2.html#defining-tactics
[1428]: proof-engine/ltac2.html#syntactic-classes
[1429]: proof-engine/ltac2.html#evaluation
[1430]: proof-engine/ltac2.html#debug
[1431]: proof-engine/ltac2.html#profiling
[1432]: proof-engine/ltac2.html#compatibility-layer-with-ltac1
[1433]: proof-engine/ltac2.html#ltac1-from-ltac2
[1434]: proof-engine/ltac2.html#simple-api
[1435]: proof-engine/ltac2.html#low-level-api
[1436]: proof-engine/ltac2.html#ltac2-from-ltac1
[1437]: proof-engine/ltac2.html#switching-between-ltac-languages
[1438]: proof-engine/ltac2.html#transition-from-ltac1
[1439]: proof-engine/ltac2.html#syntax-changes
[1440]: proof-engine/ltac2.html#tactic-delay
[1441]: proof-engine/ltac2.html#variable-binding
[1442]: proof-engine/ltac2.html#in-ltac-expressions
[1443]: proof-engine/ltac2.html#in-quotations
[1444]: proof-engine/ltac2.html#exception-catching
[1445]: using/libraries/index.html
[1446]: language/coq-library.html
[1447]: language/coq-library.html#the-prelude
[1448]: language/coq-library.html#notations
[1449]: language/coq-library.html#logic
[1450]: language/coq-library.html#propositional-connectives
[1451]: language/coq-library.html#quantifiers
[1452]: language/coq-library.html#equality
[1453]: language/coq-library.html#lemmas
[1454]: language/coq-library.html#datatypes
[1455]: language/coq-library.html#programming
[1456]: language/coq-library.html#specification
[1457]: language/coq-library.html#basic-arithmetic
[1458]: language/coq-library.html#well-founded-recursion
[1459]: language/coq-library.html#tactics
[1460]: language/coq-library.html#opam-repository
[1461]: addendum/extraction.html
[1462]: addendum/extraction.html#generating-ml-code
[1463]: addendum/extraction.html#extraction-options
[1464]: addendum/extraction.html#setting-the-target-language
[1465]: addendum/extraction.html#inlining-and-optimizations
[1466]: addendum/extraction.html#extra-elimination-of-useless-arguments
[1467]: addendum/extraction.html#accessing-opaque-proofs
[1468]: addendum/extraction.html#realizing-axioms
[1469]: addendum/extraction.html#realizing-inductive-types
[1470]: addendum/extraction.html#generating-ffi-code
[1471]: addendum/extraction.html#avoiding-conflicts-with-existing-filenames
[1472]: addendum/extraction.html#additional-settings
[1473]: addendum/extraction.html#differences-between-rocq-and-ml-type-systems
[1474]: addendum/extraction.html#some-examples
[1475]: addendum/extraction.html#a-detailed-example-euclidean-division
[1476]: addendum/extraction.html#extraction-s-horror-museum
[1477]: addendum/extraction.html#users-contributions
[1478]: addendum/miscellaneous-extensions.html
[1479]: using/libraries/funind.html
[1480]: using/libraries/funind.html#advanced-recursive-functions
[1481]: using/libraries/funind.html#tactics
[1482]: using/libraries/funind.html#generation-of-induction-principles-with-func
tional-scheme
[1483]: using/libraries/funind.html#flags
[1484]: using/libraries/writing.html
[1485]: using/libraries/writing.html#deprecating-library-objects-tactics-or-libr
ary-files
[1486]: using/libraries/writing.html#triggering-warning-for-library-objects-or-l
ibrary-files
[1487]: using/tools/index.html
[1488]: practical-tools/utilities.html
[1489]: practical-tools/utilities.html#rocq-configuration-basics
[1490]: practical-tools/utilities.html#installing-the-rocq-prover-and-rocq-packa
ges-with-opam
[1491]: practical-tools/utilities.html#setup-for-working-on-your-own-projects
[1492]: practical-tools/utilities.html#building-a-project-with-coqproject-overvi
ew
[1493]: practical-tools/utilities.html#logical-paths-and-the-load-path
[1494]: practical-tools/utilities.html#modifying-multiple-interdependent-project
s-at-the-same-time
[1495]: practical-tools/utilities.html#installed-and-uninstalled-packages
[1496]: practical-tools/utilities.html#upgrading-to-a-new-version-of-rocq
[1497]: practical-tools/utilities.html#building-a-rocq-project-with-rocq-makefil
e-details
[1498]: practical-tools/utilities.html#comments
[1499]: practical-tools/utilities.html#quoting-arguments-to-rocq-c
[1500]: practical-tools/utilities.html#forbidden-filenames
[1501]: practical-tools/utilities.html#warning-no-common-logical-root
[1502]: practical-tools/utilities.html#coqmakefile-local
[1503]: practical-tools/utilities.html#coqmakefile-local-late
[1504]: practical-tools/utilities.html#timing-targets-and-performance-testing
[1505]: practical-tools/utilities.html#building-a-subset-of-the-targets-with-j
[1506]: practical-tools/utilities.html#precompiling-for-native-compute
[1507]: practical-tools/utilities.html#the-grammar-of-coqproject
[1508]: practical-tools/utilities.html#building-a-rocq-project-with-dune
[1509]: practical-tools/utilities.html#rocq-dep-computing-module-dependencies
[1510]: practical-tools/utilities.html#split-compilation-of-native-computation-f
iles
[1511]: practical-tools/utilities.html#using-rocq-as-a-library
[1512]: practical-tools/utilities.html#embedded-rocq-phrases-inside-latex-docume
nts
[1513]: practical-tools/utilities.html#man-pages
[1514]: practical-tools/coq-commands.html
[1515]: practical-tools/coq-commands.html#interactive-use-rocq-repl
[1516]: practical-tools/coq-commands.html#batch-compilation-rocq-compile
[1517]: practical-tools/coq-commands.html#system-configuration
[1518]: practical-tools/coq-commands.html#customization-at-launch-time
[1519]: practical-tools/coq-commands.html#command-parameters
[1520]: practical-tools/coq-commands.html#coqrc-start-up-script
[1521]: practical-tools/coq-commands.html#environment-variables
[1522]: practical-tools/coq-commands.html#command-line-options
[1523]: practical-tools/coq-commands.html#profiling
[1524]: practical-tools/coq-commands.html#compiled-interfaces-produced-using-vos
[1525]: practical-tools/coq-commands.html#compiled-libraries-checker-rocqchk
[1526]: using/tools/coqdoc.html
[1527]: using/tools/coqdoc.html#principles
[1528]: using/tools/coqdoc.html#rocq-material-inside-documentation
[1529]: using/tools/coqdoc.html#pretty-printing
[1530]: using/tools/coqdoc.html#sections
[1531]: using/tools/coqdoc.html#lists
[1532]: using/tools/coqdoc.html#rules
[1533]: using/tools/coqdoc.html#emphasis
[1534]: using/tools/coqdoc.html#escaping-to-latex-and-html
[1535]: using/tools/coqdoc.html#verbatim
[1536]: using/tools/coqdoc.html#hyperlinks
[1537]: using/tools/coqdoc.html#hiding-showing-parts-of-the-source
[1538]: using/tools/coqdoc.html#usage
[1539]: using/tools/coqdoc.html#command-line-options
[1540]: using/tools/coqdoc.html#the-rocq-doc-latex-style-file
[1541]: practical-tools/coqide.html
[1542]: practical-tools/coqide.html#managing-files-and-buffers-basic-editing
[1543]: practical-tools/coqide.html#running-coq-scripts
[1544]: practical-tools/coqide.html#asynchronous-mode
[1545]: practical-tools/coqide.html#commands-and-templates
[1546]: practical-tools/coqide.html#queries
[1547]: practical-tools/coqide.html#compilation
[1548]: practical-tools/coqide.html#customizations
[1549]: practical-tools/coqide.html#preferences
[1550]: practical-tools/coqide.html#key-bindings
[1551]: practical-tools/coqide.html#using-unicode-symbols
[1552]: practical-tools/coqide.html#displaying-unicode-symbols
[1553]: practical-tools/coqide.html#bindings-for-input-of-unicode-symbols
[1554]: practical-tools/coqide.html#adding-custom-bindings
[1555]: practical-tools/coqide.html#character-encoding-for-saved-files
[1556]: practical-tools/coqide.html#debugger
[1557]: practical-tools/coqide.html#breakpoints
[1558]: practical-tools/coqide.html#call-stack-and-variables
[1559]: practical-tools/coqide.html#supported-use-cases
[1560]: addendum/parallel-proof-processing.html
[1561]: addendum/parallel-proof-processing.html#proof-annotations
[1562]: addendum/parallel-proof-processing.html#automatic-suggestion-of-proof-an
notations
[1563]: addendum/parallel-proof-processing.html#proof-blocks-and-error-resilienc
e
[1564]: addendum/parallel-proof-processing.html#caveats
[1565]: addendum/parallel-proof-processing.html#interactive-mode
[1566]: addendum/parallel-proof-processing.html#limiting-the-number-of-parallel-
workers
[1567]: addendum/parallel-proof-processing.html#id3
[1568]: appendix/history-and-changes/index.html
[1569]: history.html
[1570]: history.html#historical-roots
[1571]: history.html#versions-1-to-5
[1572]: history.html#version-1
[1573]: history.html#version-2
[1574]: history.html#version-3
[1575]: history.html#version-4
[1576]: history.html#version-5
[1577]: history.html#versions-6
[1578]: history.html#version-6-1
[1579]: history.html#version-6-2
[1580]: history.html#version-6-3
[1581]: history.html#versions-7
[1582]: history.html#summary-of-changes
[1583]: history.html#details-of-changes-in-7-0-and-7-1
[1584]: history.html#main-novelties
[1585]: history.html#details-of-changes
[1586]: history.html#language-new-let-in-construction
[1587]: history.html#language-long-names
[1588]: history.html#language-miscellaneous
[1589]: history.html#language-cases
[1590]: history.html#reduction
[1591]: history.html#new-tactics
[1592]: history.html#changes-in-existing-tactics
[1593]: history.html#efficiency
[1594]: history.html#concrete-syntax-of-constructions
[1595]: history.html#parsing-and-grammar-extension
[1596]: history.html#new-commands
[1597]: history.html#changes-in-existing-commands
[1598]: history.html#tools
[1599]: history.html#extraction
[1600]: history.html#standard-library
[1601]: history.html#new-user-contributions
[1602]: history.html#details-of-changes-in-7-2
[1603]: history.html#details-of-changes-in-7-3
[1604]: history.html#changes-in-7-3-1
[1605]: history.html#details-of-changes-in-7-4
[1606]: changes.html
[1607]: changes.html#version-9-1
[1608]: changes.html#summary-of-changes
[1609]: changes.html#changes-in-9-1-0
[1610]: changes.html#id3
[1611]: changes.html#id16
[1612]: changes.html#id23
[1613]: changes.html#id26
[1614]: changes.html#id33
[1615]: changes.html#id38
[1616]: changes.html#id63
[1617]: changes.html#id68
[1618]: changes.html#id81
[1619]: changes.html#id86
[1620]: changes.html#id91
[1621]: changes.html#id93
[1622]: changes.html#id100
[1623]: changes.html#id102
[1624]: changes.html#version-9-0
[1625]: changes.html#id105
[1626]: changes.html#porting-to-the-rocq-prover
[1627]: changes.html#renaming-advice
[1628]: changes.html#the-rocq-prover-website
[1629]: changes.html#changes-in-9-0-0
[1630]: changes.html#id116
[1631]: changes.html#id123
[1632]: changes.html#id142
[1633]: changes.html#id145
[1634]: changes.html#id159
[1635]: changes.html#id170
[1636]: changes.html#id174
[1637]: changes.html#id196
[1638]: changes.html#id202
[1639]: changes.html#standard-library
[1640]: changes.html#id214
[1641]: changes.html#id219
[1642]: changes.html#version-8-20
[1643]: changes.html#id222
[1644]: changes.html#changes-in-8-20-0
[1645]: changes.html#id232
[1646]: changes.html#changes-spec-language
[1647]: changes.html#id260
[1648]: changes.html#id269
[1649]: changes.html#id296
[1650]: changes.html#id301
[1651]: changes.html#id326
[1652]: changes.html#id335
[1653]: changes.html#id366
[1654]: changes.html#coqide
[1655]: changes.html#renaming-stdlib
[1656]: changes.html#id401
[1657]: changes.html#id418
[1658]: changes.html#changes-in-8-20-1
[1659]: changes.html#id424
[1660]: changes.html#id427
[1661]: changes.html#id432
[1662]: changes.html#version-8-19
[1663]: changes.html#id435
[1664]: changes.html#changes-in-8-19-0
[1665]: changes.html#id447
[1666]: changes.html#id452
[1667]: changes.html#id464
[1668]: changes.html#id499
[1669]: changes.html#id534
[1670]: changes.html#ltac2
[1671]: changes.html#id556
[1672]: changes.html#id574
[1673]: changes.html#id582
[1674]: changes.html#id595
[1675]: changes.html#changes-in-8-19-1
[1676]: changes.html#id599
[1677]: changes.html#id604
[1678]: changes.html#id606
[1679]: changes.html#id609
[1680]: changes.html#id614
[1681]: changes.html#changes-in-8-19-2
[1682]: changes.html#id619
[1683]: changes.html#id622
[1684]: changes.html#id625
[1685]: changes.html#id628
[1686]: changes.html#id630
[1687]: changes.html#id634
[1688]: changes.html#id637
[1689]: changes.html#version-8-18
[1690]: changes.html#id641
[1691]: changes.html#changes-in-8-18-0
[1692]: changes.html#id643
[1693]: changes.html#id649
[1694]: changes.html#id662
[1695]: changes.html#id675
[1696]: changes.html#id695
[1697]: changes.html#id713
[1698]: changes.html#id744
[1699]: changes.html#id753
[1700]: changes.html#id756
[1701]: changes.html#id777
[1702]: changes.html#id780
[1703]: changes.html#version-8-17
[1704]: changes.html#id783
[1705]: changes.html#changes-in-8-17-0
[1706]: changes.html#id789
[1707]: changes.html#id794
[1708]: changes.html#id800
[1709]: changes.html#id808
[1710]: changes.html#id828
[1711]: changes.html#id831
[1712]: changes.html#id855
[1713]: changes.html#id857
[1714]: changes.html#id886
[1715]: changes.html#id892
[1716]: changes.html#id912
[1717]: changes.html#id932
[1718]: changes.html#changes-in-8-17-1
[1719]: changes.html#version-8-16
[1720]: changes.html#id943
[1721]: changes.html#changes-in-8-16-0
[1722]: changes.html#id950
[1723]: changes.html#id962
[1724]: changes.html#id974
[1725]: changes.html#id980
[1726]: changes.html#tactic-language
[1727]: changes.html#id1014
[1728]: changes.html#id1016
[1729]: changes.html#id1042
[1730]: changes.html#id1047
[1731]: changes.html#id1049
[1732]: changes.html#id1068
[1733]: changes.html#id1080
[1734]: changes.html#changes-in-8-16-1
[1735]: changes.html#id1091
[1736]: changes.html#id1100
[1737]: changes.html#id1103
[1738]: changes.html#version-8-15
[1739]: changes.html#id1106
[1740]: changes.html#changes-in-8-15-0
[1741]: changes.html#id1113
[1742]: changes.html#id1116
[1743]: changes.html#id1127
[1744]: changes.html#id1144
[1745]: changes.html#id1191
[1746]: changes.html#id1194
[1747]: changes.html#id1203
[1748]: changes.html#id1235
[1749]: changes.html#id1250
[1750]: changes.html#id1257
[1751]: changes.html#id1270
[1752]: changes.html#id1276
[1753]: changes.html#changes-in-8-15-1
[1754]: changes.html#id1281
[1755]: changes.html#id1286
[1756]: changes.html#id1289
[1757]: changes.html#id1298
[1758]: changes.html#id1300
[1759]: changes.html#id1309
[1760]: changes.html#changes-in-8-15-2
[1761]: changes.html#id1314
[1762]: changes.html#id1319
[1763]: changes.html#id1331
[1764]: changes.html#version-8-14
[1765]: changes.html#id1334
[1766]: changes.html#changes-in-8-14-0
[1767]: changes.html#id1341
[1768]: changes.html#id1348
[1769]: changes.html#id1359
[1770]: changes.html#id1376
[1771]: changes.html#id1413
[1772]: changes.html#id1433
[1773]: changes.html#id1436
[1774]: changes.html#id1459
[1775]: changes.html#native-compilation
[1776]: changes.html#id1474
[1777]: changes.html#id1478
[1778]: changes.html#id1505
[1779]: changes.html#id1511
[1780]: changes.html#changes-in-8-14-1
[1781]: changes.html#id1514
[1782]: changes.html#id1517
[1783]: changes.html#id1520
[1784]: changes.html#id1525
[1785]: changes.html#version-8-13
[1786]: changes.html#id1530
[1787]: changes.html#changes-in-8-13-beta1
[1788]: changes.html#id1535
[1789]: changes.html#id1545
[1790]: changes.html#id1574
[1791]: changes.html#id1607
[1792]: changes.html#id1628
[1793]: changes.html#id1634
[1794]: changes.html#id1638
[1795]: changes.html#tools
[1796]: changes.html#id1656
[1797]: changes.html#id1659
[1798]: changes.html#id1674
[1799]: changes.html#changes-in-8-13-0
[1800]: changes.html#id1680
[1801]: changes.html#changes-in-8-13-1
[1802]: changes.html#id1682
[1803]: changes.html#id1684
[1804]: changes.html#changes-in-8-13-2
[1805]: changes.html#id1686
[1806]: changes.html#id1691
[1807]: changes.html#version-8-12
[1808]: changes.html#id1694
[1809]: changes.html#changes-in-8-12-beta1
[1810]: changes.html#id1700
[1811]: changes.html#id1703
[1812]: changes.html#id1724
[1813]: changes.html#id1745
[1814]: changes.html#id1786
[1815]: changes.html#id1795
[1816]: changes.html#flags-options-and-attributes
[1817]: changes.html#id1809
[1818]: changes.html#id1827
[1819]: changes.html#id1862
[1820]: changes.html#id1865
[1821]: changes.html#reals-library
[1822]: changes.html#id1902
[1823]: changes.html#refman
[1824]: changes.html#id1940
[1825]: changes.html#changes-in-8-12-0
[1826]: changes.html#changes-in-8-12-1
[1827]: changes.html#changes-in-8-12-2
[1828]: changes.html#version-8-11
[1829]: changes.html#id2000
[1830]: changes.html#changes-in-8-11-beta1
[1831]: changes.html#changes-in-8-11-0
[1832]: changes.html#changes-in-8-11-1
[1833]: changes.html#changes-in-8-11-2
[1834]: changes.html#version-8-10
[1835]: changes.html#id2138
[1836]: changes.html#other-changes-in-8-10-beta1
[1837]: changes.html#changes-in-8-10-beta2
[1838]: changes.html#changes-in-8-10-beta3
[1839]: changes.html#changes-in-8-10-0
[1840]: changes.html#changes-in-8-10-1
[1841]: changes.html#changes-in-8-10-2
[1842]: changes.html#version-8-9
[1843]: changes.html#id2285
[1844]: changes.html#details-of-changes-in-8-9-beta1
[1845]: changes.html#changes-in-8-8-0
[1846]: changes.html#changes-in-8-8-1
[1847]: changes.html#version-8-8
[1848]: changes.html#id2287
[1849]: changes.html#details-of-changes-in-8-8-beta1
[1850]: changes.html#details-of-changes-in-8-8-0
[1851]: changes.html#details-of-changes-in-8-8-1
[1852]: changes.html#details-of-changes-in-8-8-2
[1853]: changes.html#version-8-7
[1854]: changes.html#id2288
[1855]: changes.html#potential-compatibility-issues
[1856]: changes.html#details-of-changes-in-8-7-beta1
[1857]: changes.html#details-of-changes-in-8-7-beta2
[1858]: changes.html#details-of-changes-in-8-7-0
[1859]: changes.html#details-of-changes-in-8-7-1
[1860]: changes.html#details-of-changes-in-8-7-2
[1861]: changes.html#version-8-6
[1862]: changes.html#id2289
[1863]: changes.html#potential-sources-of-incompatibilities
[1864]: changes.html#details-of-changes-in-8-6beta1
[1865]: changes.html#details-of-changes-in-8-6
[1866]: changes.html#details-of-changes-in-8-6-1
[1867]: changes.html#version-8-5
[1868]: changes.html#id2290
[1869]: changes.html#id2291
[1870]: changes.html#details-of-changes-in-8-5beta1
[1871]: changes.html#details-of-changes-in-8-5beta2
[1872]: changes.html#details-of-changes-in-8-5beta3
[1873]: changes.html#details-of-changes-in-8-5
[1874]: changes.html#details-of-changes-in-8-5pl1
[1875]: changes.html#details-of-changes-in-8-5pl2
[1876]: changes.html#details-of-changes-in-8-5pl3
[1877]: changes.html#version-8-4
[1878]: changes.html#id2292
[1879]: changes.html#id2294
[1880]: changes.html#details-of-changes-in-8-4beta
[1881]: changes.html#details-of-changes-in-8-4beta2
[1882]: changes.html#details-of-changes-in-8-4
[1883]: changes.html#version-8-3
[1884]: changes.html#id2295
[1885]: changes.html#details-of-changes
[1886]: changes.html#version-8-2
[1887]: changes.html#id2296
[1888]: changes.html#id2297
[1889]: changes.html#version-8-1
[1890]: changes.html#id2298
[1891]: changes.html#details-of-changes-in-8-1beta
[1892]: changes.html#details-of-changes-in-8-1gamma
[1893]: changes.html#details-of-changes-in-8-1
[1894]: changes.html#version-8-0
[1895]: changes.html#id2299
[1896]: changes.html#details-of-changes-in-8-0beta-old-syntax
[1897]: changes.html#details-of-changes-in-8-0beta-new-syntax
[1898]: changes.html#details-of-changes-in-8-0
[1899]: appendix/indexes/index.html
[1900]: std-glossindex.html
[1901]: coq-cmdindex.html
[1902]: coq-tacindex.html
[1903]: coq-attrindex.html
[1904]: coq-optindex.html
[1905]: coq-exnindex.html
[1906]: genindex.html
[1907]: zebibliography.html
[1908]: http://www.opencontent.org/openpub
[1909]: language/core/index.html
[1910]: https://www.sphinx-doc.org/
[1911]: https://github.com/readthedocs/sphinx_rtd_theme
[1912]: https://readthedocs.org
[1913]: https://rocq-prover.org/doc/master/refman/
[1914]: https://rocq-prover.org/refman/
[1915]: https://rocq-prover.org/doc/v9.1/refman/
[1916]: https://rocq-prover.org/doc/v9.0/refman/
[1917]: https://rocq-prover.org/doc/V8.20.1/refman/
[1918]: https://rocq-prover.org/doc/V8.19.2/refman/
[1919]: https://rocq-prover.org/doc/V8.18.0/refman/
[1920]: https://rocq-prover.org/doc/V8.17.1/refman/
[1921]: https://rocq-prover.org/doc/V8.16.1/refman/
[1922]: https://rocq-prover.org/doc/V8.15.2/refman/
[1923]: https://rocq-prover.org/doc/V8.14.1/refman/
[1924]: https://rocq-prover.org/doc/V8.13.2/refman/
[1925]: https://rocq-prover.org/doc/V8.12.2/refman/
[1926]: https://rocq-prover.org/doc/V8.11.2/refman/
[1927]: https://rocq-prover.org/doc/V8.10.2/refman/
[1928]: https://rocq-prover.org/doc/V8.9.1/refman/
[1929]: https://rocq-prover.org/doc/V8.8.2/refman/
[1930]: https://rocq-prover.org/doc/V8.7.2/refman/
[1931]: https://rocq-prover.org/doc/V8.6.1/refman/
[1932]: https://rocq-prover.org/doc/V8.5pl3/refman/
[1933]: https://rocq-prover.org/doc/V8.4pl6/refman/
[1934]: https://rocq-prover.org/doc/V8.3pl5/refman/
[1935]: https://rocq-prover.org/doc/V8.2pl3/refman/
[1936]: https://rocq-prover.org/doc/V8.1pl6/refman/
[1937]: https://rocq-prover.org/doc/V8.0/doc/
[1938]: https://github.com/coq/coq/releases/download/V9.1.0/rocq-9.1.0-reference
-manual.pdf
