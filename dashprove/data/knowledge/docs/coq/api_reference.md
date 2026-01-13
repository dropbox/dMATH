* Introduction and Contents
* [ Edit on GitHub][1]
\[\begin{split}\newcommand{\as}{\kw{as}} \newcommand{\case}{\kw{case}}
\newcommand{\cons}{\textsf{cons}} \newcommand{\consf}{\textsf{consf}}
\newcommand{\emptyf}{\textsf{emptyf}} \newcommand{\End}{\kw{End}} \newcommand{\kwend}{\kw{end}}
\newcommand{\even}{\textsf{even}} \newcommand{\evenO}{\textsf{even}_\textsf{O}}
\newcommand{\evenS}{\textsf{even}_\textsf{S}} \newcommand{\Fix}{\kw{Fix}}
\newcommand{\fix}{\kw{fix}} \newcommand{\for}{\textsf{for}} \newcommand{\forest}{\textsf{forest}}
\newcommand{\Functor}{\kw{Functor}} \newcommand{\In}{\kw{in}}
\newcommand{\ind}[3]{\kw{Ind}~[#1]\left(#2\mathrm{~:=~}#3\right)}
\newcommand{\Indp}[4]{\kw{Ind}_{#4}[#1](#2:=#3)}
\newcommand{\Indpstr}[5]{\kw{Ind}_{#4}[#1](#2:=#3)/{#5}} \newcommand{\injective}{\kw{injective}}
\newcommand{\kw}[1]{\textsf{#1}} \newcommand{\length}{\textsf{length}}
\newcommand{\letin}[3]{\kw{let}~#1:=#2~\kw{in}~#3} \newcommand{\List}{\textsf{list}}
\newcommand{\lra}{\longrightarrow} \newcommand{\Match}{\kw{match}}
\newcommand{\Mod}[3]{{\kw{Mod}}({#1}:{#2}\,\zeroone{:={#3}})}
\newcommand{\ModImp}[3]{{\kw{Mod}}({#1}:{#2}:={#3})} \newcommand{\ModA}[2]{{\kw{ModA}}({#1}=={#2})}
\newcommand{\ModS}[2]{{\kw{Mod}}({#1}:{#2})} \newcommand{\ModType}[2]{{\kw{ModType}}({#1}:={#2})}
\newcommand{\mto}{.\;} \newcommand{\nat}{\textsf{nat}} \newcommand{\Nil}{\textsf{nil}}
\newcommand{\nilhl}{\textsf{nil\_hl}} \newcommand{\nO}{\textsf{O}} \newcommand{\node}{\textsf{node}}
\newcommand{\nS}{\textsf{S}} \newcommand{\odd}{\textsf{odd}}
\newcommand{\oddS}{\textsf{odd}_\textsf{S}} \newcommand{\ovl}[1]{\overline{#1}}
\newcommand{\Pair}{\textsf{pair}} \newcommand{\plus}{\mathsf{plus}}
\newcommand{\SProp}{\textsf{SProp}} \newcommand{\Prop}{\textsf{Prop}}
\newcommand{\return}{\kw{return}} \newcommand{\Set}{\textsf{Set}} \newcommand{\Sort}{\mathcal{S}}
\newcommand{\Str}{\textsf{Stream}} \newcommand{\Struct}{\kw{Struct}}
\newcommand{\subst}[3]{#1\{#2/#3\}} \newcommand{\tl}{\textsf{tl}} \newcommand{\tree}{\textsf{tree}}
\newcommand{\trii}{\triangleright_\iota} \newcommand{\Type}{\textsf{Type}}
\newcommand{\WEV}[3]{\mbox{$#1[] \vdash #2 \lra #3$}} \newcommand{\WEVT}[3]{\mbox{$#1[] \vdash #2
\lra$}\\ \mbox{$ #3$}} \newcommand{\WF}[2]{{\mathcal{W\!F}}(#1)[#2]}
\newcommand{\WFE}[1]{\WF{E}{#1}} \newcommand{\WFT}[2]{#1[] \vdash {\mathcal{W\!F}}(#2)}
\newcommand{\WFTWOLINES}[2]{{\mathcal{W\!F}}\begin{array}{l}(#1)\\\mbox{}[{#2}]\end{array}}
\newcommand{\with}{\kw{with}} \newcommand{\WS}[3]{#1[] \vdash #2 <: #3}
\newcommand{\WSE}[2]{\WS{E}{#1}{#2}} \newcommand{\WT}[4]{#1[#2] \vdash #3 : #4}
\newcommand{\WTE}[3]{\WT{E}{#1}{#2}{#3}} \newcommand{\WTEG}[2]{\WTE{\Gamma}{#1}{#2}}
\newcommand{\WTM}[3]{\WT{#1}{}{#2}{#3}} \newcommand{\zeroone}[1]{[{#1}]} \end{split}\]

# Introduction and Contents[][2]

This is the reference manual of the Rocq Prover. Rocq is a proof assistant or interactive theorem
prover. It lets you formalize mathematical concepts and then helps you interactively generate
machine-checked proofs of theorems. Machine checking gives users much more confidence that the
proofs are correct compared to human-generated and -checked proofs. Rocq has been used in a number
of flagship verification projects, including the [CompCert verified C compiler][3], and has served
to verify the proof of the [four color theorem][4] (among many other mathematical formalizations).

Users generate proofs by entering a series of tactics that constitute steps in the proof. There are
many built-in tactics, some of which are elementary, while others implement complex decision
procedures (such as [`lia`][5], a decision procedure for linear integer arithmetic). [Ltac][6] and
its planned replacement, [Ltac2][7], provide languages to define new tactics by combining existing
tactics with looping and conditional constructs. These permit automation of large parts of proofs
and sometimes entire proofs. Furthermore, users can add novel tactics or functionality by creating
Rocq plugins using OCaml.

The Rocq kernel, a small part of the Rocq Prover, does the final verification that the
tactic-generated proof is valid. Usually the tactic-generated proof is indeed correct, but
delegating proof verification to the kernel means that even if a tactic is buggy, it won't be able
to introduce an incorrect proof into the system.

Finally, Rocq also supports extraction of verified programs to programming languages such as OCaml
and Haskell. This provides a way of executing Rocq code efficiently and can be used to create
verified software libraries.

To learn Rocq, beginners are advised to first start with a tutorial / book. Several such tutorials /
books are listed at [https://rocq-prover.org/docs][8].

This manual is organized in three main parts, plus an appendix:

* **The first part presents the specification language of the Rocq Prover**, that allows to define
  programs and state mathematical theorems. [Core language][9] presents the language that the kernel
  of Rocq understands. [Language extensions][10] presents the richer language, with notations,
  implicits, etc. that a user can use and which is translated down to the language of the kernel by
  means of an "elaboration process".
* **The second part presents proof mode**, the central feature of the Rocq Prover. [Basic proof
  writing][11] introduces this interactive mode and the available proof languages. [Automatic
  solvers and programmable tactics][12] presents some more advanced tactics, while [Creating new
  tactics][13] is about the languages that allow a user to combine tactics together and develop new
  ones.
* **The third part shows how to use the Rocq Prover in practice.** [Libraries and plugins][14]
  presents some of the essential reusable blocks from the ecosystem and some particularly important
  extensions such as the program extraction mechanism. [Command-line and graphical tools][15]
  documents important tools that a user needs to build a Rocq project.
* In the appendix, [History and recent changes][16] presents the history of Rocq and changes in
  recent releases. This is an important reference if you upgrade the version of Rocq that you use.
  The various [indexes][17] are very useful to **quickly browse the manual and find what you are
  looking for.** They are often the main entry point to the manual.

The full table of contents is presented below:

## Contents[][18]

* [Introduction and Contents][19]

Specification language

* [Core language][20]
  
  * [Basic notions and conventions][21]
    
    * [Syntax and lexical conventions][22]
      
      * [Syntax conventions][23]
      * [Lexical conventions][24]
    * [Essential vocabulary][25]
    * [Settings][26]
      
      * [Attributes][27]
        
        * [Generic attributes][28]
        * [Document-level attributes][29]
      * [Flags, Options and Tables][30]
        
        * [Locality attributes supported by `Set` and `Unset`][31]
  * [Sorts][32]
  * [Functions and assumptions][33]
    
    * [Binders][34]
    * [Functions (fun) and function types (forall)][35]
    * [Function application][36]
    * [Assumptions][37]
  * [Definitions][38]
    
    * [Let-in definitions][39]
    * [Type cast][40]
    * [Top-level definitions][41]
    * [Assertions and proofs][42]
  * [Conversion rules][43]
    
    * [α-conversion][44]
    * [β-reduction][45]
    * [δ-reduction][46]
    * [ι-reduction][47]
    * [ζ-reduction][48]
    * [η-expansion][49]
    * [Examples][50]
    * [Proof Irrelevance][51]
    * [Convertibility][52]
  * [Typing rules][53]
    
    * [The terms][54]
    * [Typing rules][55]
    * [Subtyping rules][56]
    * [The Calculus of Inductive Constructions with impredicative Set][57]
  * [Variants and the `match` construct][58]
    
    * [Variants][59]
      
      * [Private (matching) inductive types][60]
    * [Definition by cases: match][61]
  * [Record types][62]
    
    * [Defining record types][63]
    * [Constructing records][64]
    * [Accessing fields (projections)][65]
    * [Settings for printing records][66]
    * [Primitive Projections][67]
      
      * [Reduction][68]
      * [Compatibility Constants for Projections][69]
  * [Inductive types and recursive functions][70]
    
    * [Inductive types][71]
      
      * [Simple inductive types][72]
        
        * [Automatic Prop lowering][73]
      * [Simple indexed inductive types][74]
      * [Parameterized inductive types][75]
      * [Mutually defined inductive types][76]
    * [Recursive functions: fix][77]
    * [Top-level recursive functions][78]
    * [Theory of inductive definitions][79]
      
      * [Types of inductive objects][80]
      * [Well-formed inductive definitions][81]
        
        * [Arity of a given sort][82]
        * [Arity][83]
        * [Type of constructor][84]
        * [Positivity Condition][85]
        * [Strict positivity][86]
        * [Nested Positivity][87]
        * [Correctness rules][88]
        * [Template polymorphism][89]
      * [Destructors][90]
        
        * [The match ... with ... end construction][91]
      * [Fixpoint definitions][92]
        
        * [Typing rule][93]
        * [Reduction rule][94]
  * [Coinductive types and corecursive functions][95]
    
    * [Coinductive types][96]
      
      * [Caveat][97]
    * [Co-recursive functions: cofix][98]
    * [Top-level definitions of corecursive functions][99]
  * [Sections][100]
    
    * [Using sections][101]
    * [Summary of locality attributes in a section][102]
    * [Typing rules used at the end of a section][103]
  * [The Module System][104]
    
    * [Modules and module types][105]
    * [Using modules][106]
      
      * [Examples][107]
    * [Qualified names][108]
    * [Controlling the scope of commands with locality attributes][109]
    * [Summary of locality attributes in a module][110]
    * [Typing Modules][111]
  * [Primitive objects][112]
    
    * [Primitive Integers][113]
    * [Primitive Floats][114]
    * [Primitive Arrays][115]
    * [Primitive (Byte-Based) Strings][116]
  * [Polymorphic Universes][117]
    
    * [General Presentation][118]
    * [Polymorphic, Monomorphic][119]
    * [Cumulative, NonCumulative][120]
      
      * [Specifying cumulativity][121]
      * [Cumulativity Weak Constraints][122]
    * [Global and local universes][123]
    * [Conversion and unification][124]
    * [Minimization][125]
    * [Explicit Universes][126]
    * [Printing universes][127]
      
      * [Polymorphic definitions][128]
    * [Sort polymorphism][129]
    * [Explicit Sorts][130]
    * [Universe polymorphism and sections][131]
  * [SProp (proof irrelevant propositions)][132]
    
    * [Basic constructs][133]
    * [Encodings for strict propositions][134]
    * [Definitional UIP][135]
      
      * [Non Termination with UIP][136]
    * [Debugging \(\SProp\) issues][137]
  * [User-defined rewrite rules][138]
    
    * [Symbols][139]
    * [Rewrite rules][140]
    * [Pattern syntax][141]
    * [Higher-order pattern holes][142]
    * [Universe polymorphic rules][143]
    * [Rewrite rules, type preservation, confluence and termination][144]
    * [Compatibility with the eta laws][145]
    * [Level of support][146]
* [Language extensions][147]
  
  * [Command level processing][148]
    
    * [Lexing][149]
    * [Parsing][150]
    * [Synterp][151]
    * [Interp][152]
  * [Term level processing][153]
  * [Existential variables][154]
    
    * [Inferable subterms][155]
    * [e* tactics that can create existential variables][156]
    * [Automatic resolution of existential variables][157]
    * [Explicit display of existential instances for pretty-printing][158]
    * [Solving existential variables using tactics][159]
  * [Implicit arguments][160]
    
    * [The different kinds of implicit arguments][161]
      
      * [Implicit arguments inferable from the knowledge of other arguments of a function][162]
      * [Implicit arguments inferable by resolution][163]
    * [Maximal and non-maximal insertion of implicit arguments][164]
      
      * [Trailing Implicit Arguments][165]
    * [Casual use of implicit arguments][166]
    * [Declaration of implicit arguments][167]
      
      * [Implicit Argument Binders][168]
      * [Mode for automatic declaration of implicit arguments][169]
      * [Controlling strict implicit arguments][170]
      * [Controlling contextual implicit arguments][171]
      * [Controlling reversible-pattern implicit arguments][172]
      * [Controlling the insertion of implicit arguments not followed by explicit arguments][173]
      * [Combining manual declaration and automatic declaration][174]
    * [Explicit applications][175]
    * [Displaying implicit arguments][176]
    * [Displaying implicit arguments when pretty-printing][177]
    * [Interaction with subtyping][178]
    * [Deactivation of implicit arguments for parsing][179]
    * [Implicit types of variables][180]
    * [Implicit generalization][181]
  * [Extended pattern matching][182]
    
    * [Variants and extensions of `match`][183]
      
      * [Multiple and nested pattern matching][184]
      * [Pattern-matching on boolean values: the if expression][185]
      * [Irrefutable patterns: the destructuring let variants][186]
        
        * [First destructuring let syntax][187]
        * [Second destructuring let syntax][188]
      * [Controlling pretty-printing of match expressions][189]
        
        * [Printing nested patterns][190]
        * [Factorization of clauses with same right-hand side][191]
        * [Use of a default clause][192]
        * [Printing of wildcard patterns][193]
        * [Printing of the elimination predicate][194]
        * [Printing of hidden subterms][195]
        * [Printing matching on irrefutable patterns][196]
        * [Printing matching on booleans][197]
      * [Conventions about unused pattern-matching variables][198]
    * [Patterns][199]
    * [Multiple patterns][200]
    * [Aliasing subpatterns][201]
    * [Nested patterns][202]
    * [Disjunctive patterns][203]
    * [About patterns of parametric types][204]
      
      * [Parameters in patterns][205]
    * [Implicit arguments in patterns][206]
    * [Matching objects of dependent types][207]
    * [Understanding dependencies in patterns][208]
    * [When the elimination predicate must be provided][209]
      
      * [Dependent pattern matching][210]
      * [Multiple dependent pattern matching][211]
      * [Patterns in `in`][212]
    * [Using pattern matching to write proofs][213]
    * [Pattern-matching on inductive objects involving local definitions][214]
    * [Pattern-matching and coercions][215]
    * [When does the expansion strategy fail?][216]
  * [Syntax extensions and notation scopes][217]
    
    * [Notations][218]
      
      * [Basic notations][219]
      * [Precedences and associativity][220]
      * [Complex notations][221]
      * [Simple factorization rules][222]
      * [Use of notations for printing][223]
      * [The Infix command][224]
      * [Reserving notations][225]
      * [Simultaneous definition of terms and notations][226]
      * [Enabling and disabling notations][227]
      * [Displaying information about notations][228]
      * [Locating notations][229]
      * [Inheritance of the properties of arguments of constants bound to a notation][230]
      * [Notations and binders][231]
        
        * [Binders bound in the notation and parsed as identifiers][232]
        * [Binders bound in the notation and parsed as patterns][233]
        * [Binders bound in the notation and parsed as terms][234]
        * [Binders bound in the notation and parsed as general binders][235]
        * [Binders not bound in the notation][236]
        * [Notations with expressions used both as binder and term][237]
      * [Notations with recursive patterns][238]
      * [Notations with recursive patterns involving binders][239]
      * [Predefined entries][240]
      * [Custom entries][241]
      * [Syntax][242]
    * [Notation scopes][243]
      
      * [Global interpretation rules for notations][244]
      * [Local interpretation rules for notations][245]
        
        * [Opening a notation scope locally][246]
        * [Binding types or coercion classes to notation scopes][247]
      * [The `type_scope` notation scope][248]
      * [The `function_scope` notation scope][249]
      * [Notation scopes used in the standard library of Rocq][250]
      * [Displaying information about scopes][251]
    * [Abbreviations][252]
    * [Numbers and strings][253]
      
      * [Number notations][254]
      * [String notations][255]
    * [Tactic Notations][256]
  * [Setting properties of a function's arguments][257]
    
    * [Manual declaration of implicit arguments][258]
    * [Automatic declaration of implicit arguments][259]
    * [Renaming implicit arguments][260]
    * [Binding arguments to scopes][261]
    * [Effects of `Arguments` on unfolding][262]
    * [Bidirectionality hints][263]
  * [Implicit Coercions][264]
    
    * [General Presentation][265]
    * [Coercion Classes][266]
    * [Coercions][267]
    * [Reversible Coercions][268]
    * [Identity Coercions][269]
    * [Inheritance Graph][270]
    * [Coercion Classes][271]
    * [Displaying Available Coercions][272]
    * [Activating the Printing of Coercions][273]
    * [Classes as Records][274]
    * [Coercions and Sections][275]
    * [Coercions and Modules][276]
    * [Examples][277]
  * [Typeclasses][278]
    
    * [Typeclass and instance declarations][279]
    * [Binding typeclasses][280]
    * [Parameterized instances][281]
    * [Sections and contexts][282]
    * [Building hierarchies][283]
      
      * [Superclasses][284]
      * [Substructures][285]
    * [Command summary][286]
      
      * [Typeclasses Transparent, Typeclasses Opaque][287]
      * [Settings][288]
      * [Typeclasses eauto][289]
  * [Canonical Structures][290]
    
    * [Declaration of canonical structures][291]
    * [Notation overloading][292]
      
      * [Derived Canonical Structures][293]
    * [Hierarchy of structures][294]
      
      * [Compact declaration of Canonical Structures][295]
  * [Program][296]
    
    * [Elaborating programs][297]
      
      * [Syntactic control over equalities][298]
      * [Program Definition][299]
      * [Program Fixpoint][300]
      * [Program Lemma][301]
    * [Solving obligations][302]
    * [Frequently Asked Questions][303]
  * [Commands][304]
    
    * [Displaying][305]
    * [Query commands][306]
    * [Requests to the environment][307]
    * [Printing flags][308]
    * [Loading files][309]
    * [Compiled files][310]
    * [Load paths][311]
    * [Extra Dependencies][312]
    * [Backtracking][313]
    * [Quitting and debugging][314]
    * [Controlling display][315]
    * [Printing constructions in full][316]
    * [Controlling Typing Flags][317]
    * [Internal registration commands][318]
      
      * [Exposing constants to OCaml libraries][319]
      * [Inlining hints for the fast reduction machines][320]
      * [Registering primitive operations][321]

Proofs

* [Basic proof writing][322]
  
  * [Proof mode][323]
    
    * [Proof State][324]
    * [Entering and exiting proof mode][325]
      
      * [Proof using options][326]
      * [Name a set of section hypotheses for `Proof using`][327]
    * [Proof modes][328]
    * [Managing goals][329]
      
      * [Focusing goals][330]
        
        * [Curly braces][331]
        * [Bullets][332]
        * [Other focusing commands][333]
      * [Shelving goals][334]
      * [Reordering goals][335]
    * [Proving a subgoal as a separate lemma: abstract][336]
    * [Requesting information][337]
    * [Showing differences between proof steps][338]
      
      * [How to enable diffs][339]
      * [How diffs are calculated][340]
      * ["Show Proof" differences][341]
    * [Delaying solving unification constraints][342]
    * [Proof maintenance][343]
    * [Controlling proof mode][344]
    * [Controlling memory usage][345]
  * [Tactics][346]
    
    * [Common elements of tactics][347]
      
      * [Reserved keywords][348]
      * [Invocation of tactics][349]
      * [Bindings][350]
      * [Intro patterns][351]
      * [Occurrence clauses][352]
      * [Automatic clearing of hypotheses][353]
    * [Applying theorems][354]
    * [Managing the local context][355]
    * [Controlling the proof flow][356]
    * [Classical tactics][357]
    * [Performance-oriented tactic variants][358]
  * [Reasoning with equalities][359]
    
    * [Tactics for simple equalities][360]
    * [Rewriting with Leibniz and setoid equality][361]
    * [Rewriting with definitional equality][362]
    * [Applying conversion rules][363]
      
      * [Fast reduction tactics: vm_compute and native_compute][364]
      * [Computing in a term: eval and Eval][365]
    * [Controlling reduction strategies and the conversion algorithm][366]
  * [Reasoning with inductive types][367]
    
    * [Applying constructors][368]
    * [Case analysis][369]
    * [Induction][370]
    * [Equality of inductive types][371]
      
      * [Helper tactics][372]
    * [Generation of induction principles with `Scheme`][373]
      
      * [Automatic declaration of schemes][374]
      * [Combined Scheme][375]
    * [Generation of inversion principles with `Derive` `Inversion`][376]
    * [Examples of `dependent destruction` / `dependent induction`][377]
      
      * [A larger example][378]
  * [The SSReflect proof language][379]
    
    * [Introduction][380]
      
      * [Acknowledgments][381]
    * [Usage][382]
      
      * [Getting started][383]
      * [Compatibility issues][384]
    * [Gallina extensions][385]
      
      * [Pattern assignment][386]
      * [Pattern conditional][387]
      * [Parametric polymorphism][388]
      * [Anonymous arguments][389]
      * [Wildcards][390]
      * [Definitions][391]
      * [Abbreviations][392]
        
        * [Matching][393]
        * [Occurrence selection][394]
      * [Basic localization][395]
    * [Basic tactics][396]
      
      * [Bookkeeping][397]
      * [The defective tactics][398]
        
        * [The move tactic.][399]
        * [The case tactic][400]
        * [The elim tactic][401]
        * [The apply tactic][402]
      * [Discharge][403]
        
        * [Clear rules][404]
        * [Matching for apply and exact][405]
        * [The abstract tactic][406]
      * [Introduction in the context][407]
        
        * [Simplification items][408]
        * [Views][409]
        * [Intro patterns][410]
        * [Clear switch][411]
        * [Branching and destructuring][412]
        * [Block introduction][413]
      * [Generation of equations][414]
      * [Type families][415]
    * [Control flow][416]
      
      * [Indentation and bullets][417]
      * [Terminators][418]
      * [Selectors][419]
      * [Iteration][420]
      * [Localization][421]
      * [Structure][422]
        
        * [The have tactic.][423]
        * [Generating let in context entries with have][424]
        * [The have tactic and typeclass resolution][425]
        * [Variants: the suff and wlog tactics][426]
          
          * [Advanced generalization][427]
    * [Rewriting][428]
      
      * [An extended rewrite tactic][429]
      * [Remarks and examples][430]
        
        * [Rewrite redex selection][431]
        * [Chained rewrite steps][432]
        * [Explicit redex switches are matched first][433]
        * [Occurrence switches and redex switches][434]
        * [Occurrence selection and repetition][435]
        * [Multi-rule rewriting][436]
        * [Wildcards vs abstractions][437]
        * [When SSReflect rewrite fails on standard Rocq licit rewrite][438]
        * [Existential metavariables and rewriting][439]
      * [Rewriting under binders][440]
        
        * [The under tactic][441]
        * [Interactive mode][442]
          
          * [The over tactic][443]
        * [One-liner mode][444]
      * [Locking, unlocking][445]
      * [Congruence][446]
    * [Contextual patterns][447]
      
      * [Syntax][448]
      * [Matching contextual patterns][449]
      * [Examples][450]
        
        * [Contextual pattern in set and the : tactical][451]
        * [Contextual patterns in rewrite][452]
      * [Patterns for recurrent contexts][453]
    * [Views and reflection][454]
      
      * [Interpreting eliminations][455]
      * [Interpreting assumptions][456]
        
        * [Specializing assumptions][457]
      * [Interpreting goals][458]
      * [Boolean reflection][459]
      * [The reflect predicate][460]
      * [General mechanism for interpreting goals and assumptions][461]
        
        * [Specializing assumptions][462]
        * [Interpreting assumptions][463]
        * [Interpreting goals][464]
      * [Interpreting equivalences][465]
      * [Declaring new Hint Views][466]
      * [Multiple views][467]
      * [Additional view shortcuts][468]
    * [Synopsis and Index][469]
      
      * [Parameters][470]
      * [Items and switches][471]
      * [Tactics][472]
      * [Tacticals][473]
      * [Commands][474]
      * [Settings][475]
* [Automatic solvers and programmable tactics][476]
  
  * [Solvers for logic and equality][477]
  * [Micromega: solvers for arithmetic goals over ordered rings][478]
    
    * [Short description of the tactics][479]
    * [*Positivstellensatz* refutations][480]
    * [`lra`: a decision procedure for linear real and rational arithmetic][481]
    * [`lia`: a tactic for linear integer arithmetic][482]
      
      * [High level view of `lia`][483]
      * [Cutting plane proofs][484]
      * [Case split][485]
    * [`nra`: a proof procedure for non-linear arithmetic][486]
    * [`nia`: a proof procedure for non-linear integer arithmetic][487]
    * [`psatz`: a proof procedure for non-linear arithmetic][488]
    * [`zify`: pre-processing of arithmetic goals][489]
  * [ring and field: solvers for polynomial and rational equations][490]
    
    * [What does this tactic do?][491]
    * [The variables map][492]
    * [Is it automatic?][493]
    * [Concrete usage][494]
    * [Adding a ring structure][495]
    * [How does it work?][496]
    * [Dealing with fields][497]
    * [Adding a new field structure][498]
    * [History of ring][499]
    * [Discussion][500]
  * [Nsatz: a solver for equalities in integral domains][501]
    
    * [More about `nsatz`][502]
  * [Programmable proof search][503]
    
    * [Tactics][504]
    * [Hint databases][505]
      
      * [Creating hint databases][506]
      * [Hint databases defined in the Rocq standard library][507]
    * [Creating Hints][508]
      
      * [Hint locality][509]
    * [Setting implicit automation tactics][510]
  * [Generalized rewriting][511]
    
    * [Introduction to generalized rewriting][512]
      
      * [Relations and morphisms][513]
      * [Adding new relations and morphisms][514]
      * [Rewriting and nonreflexive relations][515]
      * [Rewriting and nonsymmetric relations][516]
      * [Rewriting in ambiguous setoid contexts][517]
      * [Rewriting with `Type` valued relations][518]
    * [Declaring rewrite relations][519]
    * [Commands and tactics][520]
      
      * [First class setoids and morphisms][521]
      * [Tactics enabled on user provided relations][522]
      * [Printing relations and morphisms][523]
    * [Understanding and fixing failed resolutions][524]
      
      * [Deprecated syntax and backward incompatibilities][525]
    * [Extensions][526]
      
      * [Rewriting under binders][527]
      * [Subrelations][528]
      * [Constant unfolding during rewriting][529]
      * [Constant unfolding during `Proper`-instance search][530]
    * [Strategies for rewriting][531]
      
      * [Usage][532]
      * [Definitions][533]
* [Creating new tactics][534]
  
  * [Ltac][535]
    
    * [Defects][536]
    * [Syntax][537]
    * [Values][538]
      
      * [Syntactic values][539]
      * [Substitution][540]
      * [Local definitions: let][541]
      * [Function construction and application][542]
      * [Tactics in terms][543]
    * [Goal selectors][544]
    * [Processing multiple goals][545]
    * [Branching and backtracking][546]
    * [Control flow][547]
      
      * [Sequence: ;][548]
      * [Do loop][549]
      * [Repeat loop][550]
      * [Catching errors: try][551]
      * [Conditional branching: tryif][552]
    * [Alternatives][553]
      
      * [Branching with backtracking: +][554]
      * [Local application of tactics: [> ... ]][555]
      * [First tactic to succeed][556]
      * [Solving][557]
      * [First tactic to make progress: ||][558]
      * [Detecting progress][559]
    * [Success and failure][560]
      
      * [Checking for success: assert_succeeds][561]
      * [Checking for failure: assert_fails][562]
      * [Failing][563]
      * [Soft cut: once][564]
      * [Checking for a single success: exactly_once][565]
    * [Manipulating values][566]
      
      * [Pattern matching on terms: match][567]
      * [Pattern matching on goals and hypotheses: match goal][568]
      * [Filling a term context][569]
      * [Generating fresh hypothesis names][570]
      * [Computing in a term: eval][571]
      * [Getting the type of a term][572]
      * [Manipulating untyped terms: type_term][573]
      * [Counting goals: numgoals][574]
      * [Testing boolean expressions: guard][575]
      * [Checking properties of terms][576]
    * [Timing][577]
      
      * [Timeout][578]
      * [Timing a tactic][579]
      * [Timing a tactic that evaluates to a term: time_constr][580]
    * [Print/identity tactic: idtac][581]
    * [Tactic toplevel definitions][582]
      
      * [Defining `L`tac symbols][583]
      * [Printing `L`tac tactics][584]
    * [Examples of using `L`tac][585]
      
      * [Proof that the natural numbers have at least two elements][586]
      * [Proving that a list is a permutation of a second list][587]
      * [Deciding intuitionistic propositional logic][588]
      * [Deciding type isomorphisms][589]
    * [Debugging `L`tac tactics][590]
      
      * [Backtraces][591]
      * [Tracing execution][592]
      * [Interactive debugger][593]
      * [Profiling `L`tac tactics][594]
      * [Run-time optimization tactic][595]
  * [Ltac2][596]
    
    * [General design][597]
    * [ML component][598]
      
      * [Overview][599]
      * [Type Syntax][600]
      * [Type declarations][601]
      * [APIs][602]
      * [Term Syntax][603]
      * [Ltac2 Definitions][604]
      * [Printing Ltac2 tactics][605]
      * [Reduction][606]
      * [Typing][607]
      * [Effects][608]
        
        * [Standard IO][609]
        * [Fatal errors][610]
        * [Backtracking][611]
        * [Goals][612]
    * [Meta-programming][613]
      
      * [Overview][614]
      * [Quotations][615]
        
        * [Built-in quotations][616]
        * [Strict vs. non-strict mode][617]
      * [Term Antiquotations][618]
        
        * [Syntax][619]
        * [Semantics][620]
          
          * [Static semantics][621]
          * [Dynamic semantics][622]
      * [Match over terms][623]
      * [Match over goals][624]
      * [Match on values][625]
    * [Notations][626]
      
      * [Abbreviations][627]
      * [Defining tactics][628]
      * [Syntactic classes][629]
    * [Evaluation][630]
    * [Debug][631]
    * [Profiling][632]
    * [Compatibility layer with Ltac1][633]
      
      * [Ltac1 from Ltac2][634]
        
        * [Simple API][635]
        * [Low-level API][636]
      * [Ltac2 from Ltac1][637]
      * [Switching between Ltac languages][638]
    * [Transition from Ltac1][639]
      
      * [Syntax changes][640]
      * [Tactic delay][641]
      * [Variable binding][642]
        
        * [In Ltac expressions][643]
        * [In quotations][644]
      * [Exception catching][645]

Using the Rocq Prover

* [Libraries and plugins][646]
  
  * [The Coq libraries][647]
    
    * [The prelude][648]
      
      * [Notations][649]
      * [Logic][650]
        
        * [Propositional Connectives][651]
        * [Quantifiers][652]
        * [Equality][653]
        * [Lemmas][654]
      * [Datatypes][655]
        
        * [Programming][656]
      * [Specification][657]
      * [Basic Arithmetic][658]
      * [Well-founded recursion][659]
      * [Tactics][660]
    * [Opam repository][661]
  * [Program extraction][662]
    
    * [Generating ML Code][663]
    * [Extraction Options][664]
      
      * [Setting the target language][665]
      * [Inlining and optimizations][666]
      * [Extra elimination of useless arguments][667]
      * [Accessing opaque proofs][668]
      * [Realizing axioms][669]
      * [Realizing inductive types][670]
      * [Generating FFI Code][671]
      * [Avoiding conflicts with existing filenames][672]
      * [Additional settings][673]
    * [Differences between Rocq and ML type systems][674]
    * [Some examples][675]
      
      * [A detailed example: Euclidean division][676]
      * [Extraction's horror museum][677]
      * [Users' Contributions][678]
  * [Program derivation][679]
  * [Functional induction][680]
    
    * [Advanced recursive functions][681]
    * [Tactics][682]
    * [Generation of induction principles with `Functional` `Scheme`][683]
    * [Flags][684]
  * [Writing Rocq libraries and plugins][685]
    
    * [Deprecating library objects, tactics or library files][686]
    * [Triggering warning for library objects or library files][687]
* [Command-line and graphical tools][688]
  
  * [Building Rocq Projects][689]
    
    * [Rocq configuration basics][690]
      
      * [Installing the Rocq Prover and Rocq packages with opam][691]
      * [Setup for working on your own projects][692]
      * [Building a project with _CoqProject (overview)][693]
      * [Logical paths and the load path][694]
      * [Modifying multiple interdependent projects at the same time][695]
      * [Installed and uninstalled packages][696]
      * [Upgrading to a new version of Rocq][697]
    * [Building a Rocq project with rocq makefile (details)][698]
      
      * [Comments][699]
        
        * [Quoting arguments to rocq c][700]
        * [Forbidden filenames][701]
        * [Warning: No common logical root][702]
        * [CoqMakefile.local][703]
        * [CoqMakefile.local-late][704]
        * [Timing targets and performance testing][705]
        * [Building a subset of the targets with `-j`][706]
        * [Precompiling for `native_compute`][707]
        * [The grammar of _CoqProject][708]
    * [Building a Rocq project with Dune][709]
    * [rocq dep: Computing Module dependencies][710]
    * [Split compilation of native computation files][711]
    * [Using Rocq as a library][712]
    * [Embedded Rocq phrases inside LaTeX documents][713]
    * [Man pages][714]
  * [The Rocq Prover commands][715]
    
    * [Interactive use (rocq repl)][716]
    * [Batch compilation (rocq compile)][717]
    * [System configuration][718]
    * [Customization at launch time][719]
    * [Command parameters][720]
      
      * [`coqrc` start up script][721]
      * [Environment variables][722]
      * [Command line options][723]
    * [Profiling][724]
    * [Compiled interfaces (produced using `-vos`)][725]
    * [Compiled libraries checker (rocqchk)][726]
  * [Documenting Rocq files with rocq doc][727]
    
    * [Principles][728]
      
      * [Rocq material inside documentation.][729]
      * [Pretty-printing.][730]
      * [Sections][731]
      * [Lists.][732]
      * [Rules.][733]
      * [Emphasis.][734]
      * [Escaping to LaTeX and HTML.][735]
      * [Verbatim][736]
      * [Hyperlinks][737]
      * [Hiding / Showing parts of the source][738]
    * [Usage][739]
      
      * [Command line options][740]
    * [The rocq doc LaTeX style file][741]
  * [RocqIDE][742]
    
    * [Managing files and buffers, basic editing][743]
    * [Running Coq scripts][744]
    * [Asynchronous mode][745]
    * [Commands and templates][746]
    * [Queries][747]
    * [Compilation][748]
    * [Customizations][749]
      
      * [Preferences][750]
      * [Key bindings][751]
    * [Using Unicode symbols][752]
      
      * [Displaying Unicode symbols][753]
      * [Bindings for input of Unicode symbols][754]
      * [Adding custom bindings][755]
      * [Character encoding for saved files][756]
    * [Debugger][757]
      
      * [Breakpoints][758]
      * [Call Stack and Variables][759]
      * [Supported use cases][760]
  * [Asynchronous and Parallel Proof Processing][761]
    
    * [Proof annotations][762]
      
      * [Automatic suggestion of proof annotations][763]
    * [Proof blocks and error resilience][764]
      
      * [Caveats][765]
    * [Interactive mode][766]
    * [Limiting the number of parallel workers][767]
      
      * [Caveats][768]

Appendix

* [History and recent changes][769]
  
  * [Early history of Coq][770]
    
    * [Historical roots][771]
    * [Versions 1 to 5][772]
      
      * [Version 1][773]
      * [Version 2][774]
      * [Version 3][775]
      * [Version 4][776]
      * [Version 5][777]
    * [Versions 6][778]
      
      * [Version 6.1][779]
      * [Version 6.2][780]
      * [Version 6.3][781]
    * [Versions 7][782]
      
      * [Summary of changes][783]
      * [Details of changes in 7.0 and 7.1][784]
        
        * [Main novelties][785]
        * [Details of changes][786]
          
          * [Language: new "let-in" construction][787]
          * [Language: long names][788]
          * [Language: miscellaneous][789]
          * [Language: Cases][790]
          * [Reduction][791]
          * [New tactics][792]
          * [Changes in existing tactics][793]
          * [Efficiency][794]
          * [Concrete syntax of constructions][795]
          * [Parsing and grammar extension][796]
          * [New commands][797]
          * [Changes in existing commands][798]
          * [Tools][799]
          * [Extraction][800]
          * [Standard library][801]
          * [New user contributions][802]
      * [Details of changes in 7.2][803]
      * [Details of changes in 7.3][804]
        
        * [Changes in 7.3.1][805]
      * [Details of changes in 7.4][806]
  * [Recent changes][807]
    
    * [Version 9.1][808]
      
      * [Summary of changes][809]
      * [Changes in 9.1.0][810]
        
        * [Kernel][811]
        * [Specification language, type inference][812]
        * [Notations][813]
        * [Tactics][814]
        * [Ltac language][815]
        * [Ltac2 language][816]
        * [SSReflect][817]
        * [Commands and options][818]
        * [Command-line tools][819]
        * [RocqIDE][820]
        * [Corelib][821]
        * [Infrastructure and dependencies][822]
        * [Extraction][823]
        * [Miscellaneous][824]
    * [Version 9.0][825]
      
      * [Summary of changes][826]
      * [Porting to The Rocq Prover][827]
      * [Renaming Advice][828]
      * [The Rocq Prover Website][829]
      * [Changes in 9.0.0][830]
        
        * [Kernel][831]
        * [Specification language, type inference][832]
        * [Notations][833]
        * [Tactics][834]
        * [Ltac2 language][835]
        * [SSReflect][836]
        * [Commands and options][837]
        * [Command-line tools][838]
        * [RocqIDE][839]
        * [Standard library][840]
        * [Infrastructure and dependencies][841]
        * [Miscellaneous][842]
    * [Version 8.20][843]
      
      * [Summary of changes][844]
      * [Changes in 8.20.0][845]
        
        * [Kernel][846]
        * [Specification language, type inference][847]
        * [Notations][848]
        * [Tactics][849]
        * [Ltac language][850]
        * [Ltac2 language][851]
        * [SSReflect][852]
        * [Commands and options][853]
        * [Command-line tools][854]
        * [CoqIDE][855]
        * [Standard library][856]
        * [Infrastructure and dependencies][857]
        * [Extraction][858]
      * [Changes in 8.20.1][859]
        
        * [Kernel][860]
        * [Notations][861]
        * [Tactics][862]
    * [Version 8.19][863]
      
      * [Summary of changes][864]
      * [Changes in 8.19.0][865]
        
        * [Kernel][866]
        * [Specification language, type inference][867]
        * [Notations][868]
        * [Tactics][869]
        * [Ltac language][870]
        * [Ltac2 language][871]
        * [Commands and options][872]
        * [Command-line tools][873]
        * [Standard library][874]
        * [Extraction][875]
      * [Changes in 8.19.1][876]
        
        * [Kernel][877]
        * [Notations][878]
        * [Tactics][879]
        * [Ltac2 language][880]
        * [Infrastructure and dependencies][881]
      * [Changes in 8.19.2][882]
        
        * [Specification language, type inference][883]
        * [Notations][884]
        * [Tactics][885]
        * [Ltac2 language][886]
        * [Commands and options][887]
        * [CoqIDE][888]
        * [Infrastructure and dependencies][889]
    * [Version 8.18][890]
      
      * [Summary of changes][891]
      * [Changes in 8.18.0][892]
        
        * [Kernel][893]
        * [Specification language, type inference][894]
        * [Notations][895]
        * [Tactics][896]
        * [Ltac2 language][897]
        * [Commands and options][898]
        * [Command-line tools][899]
        * [CoqIDE][900]
        * [Standard library][901]
        * [Infrastructure and dependencies][902]
        * [Extraction][903]
    * [Version 8.17][904]
      
      * [Summary of changes][905]
      * [Changes in 8.17.0][906]
        
        * [Kernel][907]
        * [Specification language, type inference][908]
        * [Notations][909]
        * [Tactics][910]
        * [Ltac language][911]
        * [Ltac2 language][912]
        * [SSReflect][913]
        * [Commands and options][914]
        * [Command-line tools][915]
        * [Standard library][916]
        * [Infrastructure and dependencies][917]
        * [Miscellaneous][918]
      * [Changes in 8.17.1][919]
    * [Version 8.16][920]
      
      * [Summary of changes][921]
      * [Changes in 8.16.0][922]
        
        * [Kernel][923]
        * [Specification language, type inference][924]
        * [Notations][925]
        * [Tactics][926]
        * [Tactic language][927]
        * [SSReflect][928]
        * [Commands and options][929]
        * [Command-line tools][930]
        * [CoqIDE][931]
        * [Standard library][932]
        * [Infrastructure and dependencies][933]
        * [Extraction][934]
      * [Changes in 8.16.1][935]
        
        * [Kernel][936]
        * [Commands and options][937]
        * [CoqIDE][938]
    * [Version 8.15][939]
      
      * [Summary of changes][940]
      * [Changes in 8.15.0][941]
        
        * [Kernel][942]
        * [Specification language, type inference][943]
        * [Notations][944]
        * [Tactics][945]
        * [Tactic language][946]
        * [SSReflect][947]
        * [Commands and options][948]
        * [Command-line tools][949]
        * [CoqIDE][950]
        * [Standard library][951]
        * [Infrastructure and dependencies][952]
        * [Extraction][953]
      * [Changes in 8.15.1][954]
        
        * [Kernel][955]
        * [Notations][956]
        * [Tactics][957]
        * [Command-line tools][958]
        * [CoqIDE][959]
        * [Miscellaneous][960]
      * [Changes in 8.15.2][961]
        
        * [Tactics][962]
        * [CoqIDE][963]
        * [Standard library][964]
    * [Version 8.14][965]
      
      * [Summary of changes][966]
      * [Changes in 8.14.0][967]
        
        * [Kernel][968]
        * [Specification language, type inference][969]
        * [Notations][970]
        * [Tactics][971]
        * [Tactic language][972]
        * [SSReflect][973]
        * [Commands and options][974]
        * [Command-line tools][975]
        * [Native Compilation][976]
        * [CoqIDE][977]
        * [Standard library][978]
        * [Infrastructure and dependencies][979]
        * [Miscellaneous][980]
      * [Changes in 8.14.1][981]
        
        * [Kernel][982]
        * [Specification language, type inference][983]
        * [Tactics][984]
        * [Commands and options][985]
    * [Version 8.13][986]
      
      * [Summary of changes][987]
      * [Changes in 8.13+beta1][988]
        
        * [Kernel][989]
        * [Specification language, type inference][990]
        * [Notations][991]
        * [Tactics][992]
        * [Tactic language][993]
        * [SSReflect][994]
        * [Commands and options][995]
        * [Tools][996]
        * [CoqIDE][997]
        * [Standard library][998]
        * [Infrastructure and dependencies][999]
      * [Changes in 8.13.0][1000]
        
        * [Commands and options][1001]
      * [Changes in 8.13.1][1002]
        
        * [Kernel][1003]
        * [CoqIDE][1004]
      * [Changes in 8.13.2][1005]
        
        * [Kernel][1006]
        * [Tactic language][1007]
    * [Version 8.12][1008]
      
      * [Summary of changes][1009]
      * [Changes in 8.12+beta1][1010]
        
        * [Kernel][1011]
        * [Specification language, type inference][1012]
        * [Notations][1013]
        * [Tactics][1014]
        * [Tactic language][1015]
        * [SSReflect][1016]
        * [Flags, options and attributes][1017]
        * [Commands][1018]
        * [Tools][1019]
        * [CoqIDE][1020]
        * [Standard library][1021]
        * [Reals library][1022]
        * [Extraction][1023]
        * [Reference manual][1024]
        * [Infrastructure and dependencies][1025]
      * [Changes in 8.12.0][1026]
      * [Changes in 8.12.1][1027]
      * [Changes in 8.12.2][1028]
    * [Version 8.11][1029]
      
      * [Summary of changes][1030]
      * [Changes in 8.11+beta1][1031]
      * [Changes in 8.11.0][1032]
      * [Changes in 8.11.1][1033]
      * [Changes in 8.11.2][1034]
    * [Version 8.10][1035]
      
      * [Summary of changes][1036]
      * [Other changes in 8.10+beta1][1037]
      * [Changes in 8.10+beta2][1038]
      * [Changes in 8.10+beta3][1039]
      * [Changes in 8.10.0][1040]
      * [Changes in 8.10.1][1041]
      * [Changes in 8.10.2][1042]
    * [Version 8.9][1043]
      
      * [Summary of changes][1044]
      * [Details of changes in 8.9+beta1][1045]
      * [Changes in 8.8.0][1046]
      * [Changes in 8.8.1][1047]
    * [Version 8.8][1048]
      
      * [Summary of changes][1049]
      * [Details of changes in 8.8+beta1][1050]
      * [Details of changes in 8.8.0][1051]
      * [Details of changes in 8.8.1][1052]
      * [Details of changes in 8.8.2][1053]
    * [Version 8.7][1054]
      
      * [Summary of changes][1055]
      * [Potential compatibility issues][1056]
      * [Details of changes in 8.7+beta1][1057]
      * [Details of changes in 8.7+beta2][1058]
      * [Details of changes in 8.7.0][1059]
      * [Details of changes in 8.7.1][1060]
      * [Details of changes in 8.7.2][1061]
    * [Version 8.6][1062]
      
      * [Summary of changes][1063]
      * [Potential sources of incompatibilities][1064]
      * [Details of changes in 8.6beta1][1065]
      * [Details of changes in 8.6][1066]
      * [Details of changes in 8.6.1][1067]
    * [Version 8.5][1068]
      
      * [Summary of changes][1069]
      * [Potential sources of incompatibilities][1070]
      * [Details of changes in 8.5beta1][1071]
      * [Details of changes in 8.5beta2][1072]
      * [Details of changes in 8.5beta3][1073]
      * [Details of changes in 8.5][1074]
      * [Details of changes in 8.5pl1][1075]
      * [Details of changes in 8.5pl2][1076]
      * [Details of changes in 8.5pl3][1077]
    * [Version 8.4][1078]
      
      * [Summary of changes][1079]
      * [Potential sources of incompatibilities][1080]
      * [Details of changes in 8.4beta][1081]
      * [Details of changes in 8.4beta2][1082]
      * [Details of changes in 8.4][1083]
    * [Version 8.3][1084]
      
      * [Summary of changes][1085]
      * [Details of changes][1086]
    * [Version 8.2][1087]
      
      * [Summary of changes][1088]
      * [Details of changes][1089]
    * [Version 8.1][1090]
      
      * [Summary of changes][1091]
      * [Details of changes in 8.1beta][1092]
      * [Details of changes in 8.1gamma][1093]
      * [Details of changes in 8.1][1094]
    * [Version 8.0][1095]
      
      * [Summary of changes][1096]
      * [Details of changes in 8.0beta old syntax][1097]
      * [Details of changes in 8.0beta new syntax][1098]
      * [Details of changes in 8.0][1099]
* [Indexes][1100]
  
  * [Glossary index][1101]
  * [Command index][1102]
  * [Tactic index][1103]
  * [Attribute index][1104]
  * [Flags, options and tables index][1105]
  * [Errors and warnings index][1106]
  * [Index][1107]
* [Bibliography][1108]

Note

**License**

This material (the Rocq Reference Manual) may be distributed only subject to the terms and
conditions set forth in the Open Publication License, v1.0 or later (the latest version is presently
available at [http://www.opencontent.org/openpub][1109]). Options A and B are not elected.

[Next ][1110]

© Copyright 1999-2025, Inria, CNRS and contributors.

Built with [Sphinx][1111] using a [theme][1112] provided by [Read the Docs][1113].

[1]: https://github.com/coq/coq/blob/master/doc/sphinx/index.rst
[2]: #introduction-and-contents
[3]: http://compcert.inria.fr/
[4]: https://github.com/math-comp/fourcolor
[5]: addendum/micromega.html#coq:tacn.lia
[6]: proof-engine/ltac.html#ltac
[7]: proof-engine/ltac2.html#ltac2
[8]: https://rocq-prover.org/docs
[9]: language/core/index.html#core-language
[10]: language/extensions/index.html#extensions
[11]: proofs/writing-proofs/index.html#writing-proofs
[12]: proofs/automatic-tactics/index.html#automatic-tactics
[13]: proofs/creating-tactics/index.html#writing-tactics
[14]: using/libraries/index.html#libraries
[15]: using/tools/index.html#tools
[16]: appendix/history-and-changes/index.html#history-and-changes
[17]: appendix/indexes/index.html#indexes
[18]: #contents
[19]: #
[20]: language/core/index.html
[21]: language/core/basic.html
[22]: language/core/basic.html#syntax-and-lexical-conventions
[23]: language/core/basic.html#syntax-conventions
[24]: language/core/basic.html#lexical-conventions
[25]: language/core/basic.html#essential-vocabulary
[26]: language/core/basic.html#settings
[27]: language/core/basic.html#attributes
[28]: language/core/basic.html#generic-attributes
[29]: language/core/basic.html#document-level-attributes
[30]: language/core/basic.html#flags-options-and-tables
[31]: language/core/basic.html#locality-attributes-supported-by-set-and-unset
[32]: language/core/sorts.html
[33]: language/core/assumptions.html
[34]: language/core/assumptions.html#binders
[35]: language/core/assumptions.html#functions-fun-and-function-types-forall
[36]: language/core/assumptions.html#function-application
[37]: language/core/assumptions.html#assumptions
[38]: language/core/definitions.html
[39]: language/core/definitions.html#let-in-definitions
[40]: language/core/definitions.html#type-cast
[41]: language/core/definitions.html#top-level-definitions
[42]: language/core/definitions.html#assertions-and-proofs
[43]: language/core/conversion.html
[44]: language/core/conversion.html#conversion
[45]: language/core/conversion.html#reduction
[46]: language/core/conversion.html#delta-reduction-sect
[47]: language/core/conversion.html#iota-reduction-sect
[48]: language/core/conversion.html#zeta-reduction-sect
[49]: language/core/conversion.html#expansion
[50]: language/core/conversion.html#examples
[51]: language/core/conversion.html#proof-irrelevance
[52]: language/core/conversion.html#convertibility
[53]: language/cic.html
[54]: language/cic.html#the-terms
[55]: language/cic.html#id4
[56]: language/cic.html#subtyping-rules
[57]: language/cic.html#the-calculus-of-inductive-constructions-with-impredicative-set
[58]: language/core/variants.html
[59]: language/core/variants.html#id1
[60]: language/core/variants.html#private-matching-inductive-types
[61]: language/core/variants.html#definition-by-cases-match
[62]: language/core/records.html
[63]: language/core/records.html#defining-record-types
[64]: language/core/records.html#constructing-records
[65]: language/core/records.html#accessing-fields-projections
[66]: language/core/records.html#settings-for-printing-records
[67]: language/core/records.html#primitive-projections
[68]: language/core/records.html#reduction
[69]: language/core/records.html#compatibility-constants-for-projections
[70]: language/core/inductive.html
[71]: language/core/inductive.html#inductive-types
[72]: language/core/inductive.html#simple-inductive-types
[73]: language/core/inductive.html#automatic-prop-lowering
[74]: language/core/inductive.html#simple-indexed-inductive-types
[75]: language/core/inductive.html#parameterized-inductive-types
[76]: language/core/inductive.html#mutually-defined-inductive-types
[77]: language/core/inductive.html#recursive-functions-fix
[78]: language/core/inductive.html#top-level-recursive-functions
[79]: language/core/inductive.html#theory-of-inductive-definitions
[80]: language/core/inductive.html#types-of-inductive-objects
[81]: language/core/inductive.html#well-formed-inductive-definitions
[82]: language/core/inductive.html#arity-of-a-given-sort
[83]: language/core/inductive.html#arity
[84]: language/core/inductive.html#type-of-constructor
[85]: language/core/inductive.html#positivity-condition
[86]: language/core/inductive.html#strict-positivity
[87]: language/core/inductive.html#nested-positivity
[88]: language/core/inductive.html#correctness-rules
[89]: language/core/inductive.html#template-polymorphism
[90]: language/core/inductive.html#destructors
[91]: language/core/inductive.html#the-match-with-end-construction
[92]: language/core/inductive.html#fixpoint-definitions
[93]: language/core/inductive.html#id10
[94]: language/core/inductive.html#reduction-rule
[95]: language/core/coinductive.html
[96]: language/core/coinductive.html#coinductive-types
[97]: language/core/coinductive.html#caveat
[98]: language/core/coinductive.html#co-recursive-functions-cofix
[99]: language/core/coinductive.html#top-level-definitions-of-corecursive-functions
[100]: language/core/sections.html
[101]: language/core/sections.html#using-sections
[102]: language/core/sections.html#summary-of-locality-attributes-in-a-section
[103]: language/core/sections.html#typing-rules-used-at-the-end-of-a-section
[104]: language/core/modules.html
[105]: language/core/modules.html#modules-and-module-types
[106]: language/core/modules.html#using-modules
[107]: language/core/modules.html#examples
[108]: language/core/modules.html#qualified-names
[109]: language/core/modules.html#controlling-the-scope-of-commands-with-locality-attributes
[110]: language/core/modules.html#summary-of-locality-attributes-in-a-module
[111]: language/core/modules.html#typing-modules
[112]: language/core/primitive.html
[113]: language/core/primitive.html#primitive-integers
[114]: language/core/primitive.html#primitive-floats
[115]: language/core/primitive.html#primitive-arrays
[116]: language/core/primitive.html#primitive-byte-based-strings
[117]: addendum/universe-polymorphism.html
[118]: addendum/universe-polymorphism.html#general-presentation
[119]: addendum/universe-polymorphism.html#polymorphic-monomorphic
[120]: addendum/universe-polymorphism.html#cumulative-noncumulative
[121]: addendum/universe-polymorphism.html#specifying-cumulativity
[122]: addendum/universe-polymorphism.html#cumulativity-weak-constraints
[123]: addendum/universe-polymorphism.html#global-and-local-universes
[124]: addendum/universe-polymorphism.html#conversion-and-unification
[125]: addendum/universe-polymorphism.html#minimization
[126]: addendum/universe-polymorphism.html#explicit-universes
[127]: addendum/universe-polymorphism.html#printing-universes
[128]: addendum/universe-polymorphism.html#polymorphic-definitions
[129]: addendum/universe-polymorphism.html#sort-polymorphism
[130]: addendum/universe-polymorphism.html#explicit-sorts
[131]: addendum/universe-polymorphism.html#universe-polymorphism-and-sections
[132]: addendum/sprop.html
[133]: addendum/sprop.html#basic-constructs
[134]: addendum/sprop.html#encodings-for-strict-propositions
[135]: addendum/sprop.html#definitional-uip
[136]: addendum/sprop.html#non-termination-with-uip
[137]: addendum/sprop.html#debugging-sprop-issues
[138]: addendum/rewrite-rules.html
[139]: addendum/rewrite-rules.html#symbols
[140]: addendum/rewrite-rules.html#id1
[141]: addendum/rewrite-rules.html#pattern-syntax
[142]: addendum/rewrite-rules.html#higher-order-pattern-holes
[143]: addendum/rewrite-rules.html#universe-polymorphic-rules
[144]: addendum/rewrite-rules.html#rewrite-rules-type-preservation-confluence-and-termination
[145]: addendum/rewrite-rules.html#compatibility-with-the-eta-laws
[146]: addendum/rewrite-rules.html#level-of-support
[147]: language/extensions/index.html
[148]: language/extensions/compil-steps.html
[149]: language/extensions/compil-steps.html#lexing
[150]: language/extensions/compil-steps.html#parsing
[151]: language/extensions/compil-steps.html#synterp
[152]: language/extensions/compil-steps.html#interp
[153]: language/extensions/compil-steps.html#term-level-processing
[154]: language/extensions/evars.html
[155]: language/extensions/evars.html#inferable-subterms
[156]: language/extensions/evars.html#e-tactics-that-can-create-existential-variables
[157]: language/extensions/evars.html#automatic-resolution-of-existential-variables
[158]: language/extensions/evars.html#explicit-display-of-existential-instances-for-pretty-printing
[159]: language/extensions/evars.html#solving-existential-variables-using-tactics
[160]: language/extensions/implicit-arguments.html
[161]: language/extensions/implicit-arguments.html#the-different-kinds-of-implicit-arguments
[162]: language/extensions/implicit-arguments.html#implicit-arguments-inferable-from-the-knowledge-o
f-other-arguments-of-a-function
[163]: language/extensions/implicit-arguments.html#implicit-arguments-inferable-by-resolution
[164]: language/extensions/implicit-arguments.html#maximal-and-non-maximal-insertion-of-implicit-arg
uments
[165]: language/extensions/implicit-arguments.html#trailing-implicit-arguments
[166]: language/extensions/implicit-arguments.html#casual-use-of-implicit-arguments
[167]: language/extensions/implicit-arguments.html#declaration-of-implicit-arguments
[168]: language/extensions/implicit-arguments.html#implicit-argument-binders
[169]: language/extensions/implicit-arguments.html#mode-for-automatic-declaration-of-implicit-argume
nts
[170]: language/extensions/implicit-arguments.html#controlling-strict-implicit-arguments
[171]: language/extensions/implicit-arguments.html#controlling-contextual-implicit-arguments
[172]: language/extensions/implicit-arguments.html#controlling-reversible-pattern-implicit-arguments
[173]: language/extensions/implicit-arguments.html#controlling-the-insertion-of-implicit-arguments-n
ot-followed-by-explicit-arguments
[174]: language/extensions/implicit-arguments.html#combining-manual-declaration-and-automatic-declar
ation
[175]: language/extensions/implicit-arguments.html#explicit-applications
[176]: language/extensions/implicit-arguments.html#displaying-implicit-arguments
[177]: language/extensions/implicit-arguments.html#displaying-implicit-arguments-when-pretty-printin
g
[178]: language/extensions/implicit-arguments.html#interaction-with-subtyping
[179]: language/extensions/implicit-arguments.html#deactivation-of-implicit-arguments-for-parsing
[180]: language/extensions/implicit-arguments.html#implicit-types-of-variables
[181]: language/extensions/implicit-arguments.html#implicit-generalization
[182]: language/extensions/match.html
[183]: language/extensions/match.html#variants-and-extensions-of-match
[184]: language/extensions/match.html#multiple-and-nested-pattern-matching
[185]: language/extensions/match.html#pattern-matching-on-boolean-values-the-if-expression
[186]: language/extensions/match.html#irrefutable-patterns-the-destructuring-let-variants
[187]: language/extensions/match.html#first-destructuring-let-syntax
[188]: language/extensions/match.html#second-destructuring-let-syntax
[189]: language/extensions/match.html#controlling-pretty-printing-of-match-expressions
[190]: language/extensions/match.html#printing-nested-patterns
[191]: language/extensions/match.html#factorization-of-clauses-with-same-right-hand-side
[192]: language/extensions/match.html#use-of-a-default-clause
[193]: language/extensions/match.html#printing-of-wildcard-patterns
[194]: language/extensions/match.html#printing-of-the-elimination-predicate
[195]: language/extensions/match.html#printing-of-hidden-subterms
[196]: language/extensions/match.html#printing-matching-on-irrefutable-patterns
[197]: language/extensions/match.html#printing-matching-on-booleans
[198]: language/extensions/match.html#conventions-about-unused-pattern-matching-variables
[199]: language/extensions/match.html#patterns
[200]: language/extensions/match.html#multiple-patterns
[201]: language/extensions/match.html#aliasing-subpatterns
[202]: language/extensions/match.html#nested-patterns
[203]: language/extensions/match.html#disjunctive-patterns
[204]: language/extensions/match.html#about-patterns-of-parametric-types
[205]: language/extensions/match.html#parameters-in-patterns
[206]: language/extensions/match.html#implicit-arguments-in-patterns
[207]: language/extensions/match.html#matching-objects-of-dependent-types
[208]: language/extensions/match.html#understanding-dependencies-in-patterns
[209]: language/extensions/match.html#when-the-elimination-predicate-must-be-provided
[210]: language/extensions/match.html#dependent-pattern-matching
[211]: language/extensions/match.html#multiple-dependent-pattern-matching
[212]: language/extensions/match.html#patterns-in-in
[213]: language/extensions/match.html#using-pattern-matching-to-write-proofs
[214]: language/extensions/match.html#pattern-matching-on-inductive-objects-involving-local-definiti
ons
[215]: language/extensions/match.html#pattern-matching-and-coercions
[216]: language/extensions/match.html#when-does-the-expansion-strategy-fail
[217]: user-extensions/syntax-extensions.html
[218]: user-extensions/syntax-extensions.html#notations
[219]: user-extensions/syntax-extensions.html#basic-notations
[220]: user-extensions/syntax-extensions.html#precedences-and-associativity
[221]: user-extensions/syntax-extensions.html#complex-notations
[222]: user-extensions/syntax-extensions.html#simple-factorization-rules
[223]: user-extensions/syntax-extensions.html#use-of-notations-for-printing
[224]: user-extensions/syntax-extensions.html#the-infix-command
[225]: user-extensions/syntax-extensions.html#reserving-notations
[226]: user-extensions/syntax-extensions.html#simultaneous-definition-of-terms-and-notations
[227]: user-extensions/syntax-extensions.html#enabling-and-disabling-notations
[228]: user-extensions/syntax-extensions.html#displaying-information-about-notations
[229]: user-extensions/syntax-extensions.html#locating-notations
[230]: user-extensions/syntax-extensions.html#inheritance-of-the-properties-of-arguments-of-constant
s-bound-to-a-notation
[231]: user-extensions/syntax-extensions.html#notations-and-binders
[232]: user-extensions/syntax-extensions.html#binders-bound-in-the-notation-and-parsed-as-identifier
s
[233]: user-extensions/syntax-extensions.html#binders-bound-in-the-notation-and-parsed-as-patterns
[234]: user-extensions/syntax-extensions.html#binders-bound-in-the-notation-and-parsed-as-terms
[235]: user-extensions/syntax-extensions.html#binders-bound-in-the-notation-and-parsed-as-general-bi
nders
[236]: user-extensions/syntax-extensions.html#binders-not-bound-in-the-notation
[237]: user-extensions/syntax-extensions.html#notations-with-expressions-used-both-as-binder-and-ter
m
[238]: user-extensions/syntax-extensions.html#notations-with-recursive-patterns
[239]: user-extensions/syntax-extensions.html#notations-with-recursive-patterns-involving-binders
[240]: user-extensions/syntax-extensions.html#predefined-entries
[241]: user-extensions/syntax-extensions.html#custom-entries
[242]: user-extensions/syntax-extensions.html#syntax
[243]: user-extensions/syntax-extensions.html#notation-scopes
[244]: user-extensions/syntax-extensions.html#global-interpretation-rules-for-notations
[245]: user-extensions/syntax-extensions.html#local-interpretation-rules-for-notations
[246]: user-extensions/syntax-extensions.html#opening-a-notation-scope-locally
[247]: user-extensions/syntax-extensions.html#binding-types-or-coercion-classes-to-notation-scopes
[248]: user-extensions/syntax-extensions.html#the-type-scope-notation-scope
[249]: user-extensions/syntax-extensions.html#the-function-scope-notation-scope
[250]: user-extensions/syntax-extensions.html#notation-scopes-used-in-the-standard-library-of-rocq
[251]: user-extensions/syntax-extensions.html#displaying-information-about-scopes
[252]: user-extensions/syntax-extensions.html#abbreviations
[253]: user-extensions/syntax-extensions.html#numbers-and-strings
[254]: user-extensions/syntax-extensions.html#number-notations
[255]: user-extensions/syntax-extensions.html#string-notations
[256]: user-extensions/syntax-extensions.html#tactic-notations
[257]: language/extensions/arguments-command.html
[258]: language/extensions/arguments-command.html#manual-declaration-of-implicit-arguments
[259]: language/extensions/arguments-command.html#automatic-declaration-of-implicit-arguments
[260]: language/extensions/arguments-command.html#renaming-implicit-arguments
[261]: language/extensions/arguments-command.html#binding-arguments-to-scopes
[262]: language/extensions/arguments-command.html#effects-of-arguments-on-unfolding
[263]: language/extensions/arguments-command.html#bidirectionality-hints
[264]: addendum/implicit-coercions.html
[265]: addendum/implicit-coercions.html#general-presentation
[266]: addendum/implicit-coercions.html#coercion-classes
[267]: addendum/implicit-coercions.html#id1
[268]: addendum/implicit-coercions.html#reversible-coercions
[269]: addendum/implicit-coercions.html#identity-coercions
[270]: addendum/implicit-coercions.html#inheritance-graph
[271]: addendum/implicit-coercions.html#id2
[272]: addendum/implicit-coercions.html#displaying-available-coercions
[273]: addendum/implicit-coercions.html#activating-the-printing-of-coercions
[274]: addendum/implicit-coercions.html#classes-as-records
[275]: addendum/implicit-coercions.html#coercions-and-sections
[276]: addendum/implicit-coercions.html#coercions-and-modules
[277]: addendum/implicit-coercions.html#examples
[278]: addendum/type-classes.html
[279]: addendum/type-classes.html#typeclass-and-instance-declarations
[280]: addendum/type-classes.html#binding-typeclasses
[281]: addendum/type-classes.html#parameterized-instances
[282]: addendum/type-classes.html#sections-and-contexts
[283]: addendum/type-classes.html#building-hierarchies
[284]: addendum/type-classes.html#superclasses
[285]: addendum/type-classes.html#substructures
[286]: addendum/type-classes.html#command-summary
[287]: addendum/type-classes.html#typeclasses-transparent-typeclasses-opaque
[288]: addendum/type-classes.html#settings
[289]: addendum/type-classes.html#typeclasses-eauto
[290]: language/extensions/canonical.html
[291]: language/extensions/canonical.html#declaration-of-canonical-structures
[292]: language/extensions/canonical.html#notation-overloading
[293]: language/extensions/canonical.html#derived-canonical-structures
[294]: language/extensions/canonical.html#hierarchy-of-structures
[295]: language/extensions/canonical.html#compact-declaration-of-canonical-structures
[296]: addendum/program.html
[297]: addendum/program.html#elaborating-programs
[298]: addendum/program.html#syntactic-control-over-equalities
[299]: addendum/program.html#program-definition
[300]: addendum/program.html#program-fixpoint
[301]: addendum/program.html#program-lemma
[302]: addendum/program.html#solving-obligations
[303]: addendum/program.html#frequently-asked-questions
[304]: proof-engine/vernacular-commands.html
[305]: proof-engine/vernacular-commands.html#displaying
[306]: proof-engine/vernacular-commands.html#query-commands
[307]: proof-engine/vernacular-commands.html#requests-to-the-environment
[308]: proof-engine/vernacular-commands.html#printing-flags
[309]: proof-engine/vernacular-commands.html#loading-files
[310]: proof-engine/vernacular-commands.html#compiled-files
[311]: proof-engine/vernacular-commands.html#load-paths
[312]: proof-engine/vernacular-commands.html#extra-dependencies
[313]: proof-engine/vernacular-commands.html#backtracking
[314]: proof-engine/vernacular-commands.html#quitting-and-debugging
[315]: proof-engine/vernacular-commands.html#controlling-display
[316]: proof-engine/vernacular-commands.html#printing-constructions-in-full
[317]: proof-engine/vernacular-commands.html#controlling-typing-flags
[318]: proof-engine/vernacular-commands.html#internal-registration-commands
[319]: proof-engine/vernacular-commands.html#exposing-constants-to-ocaml-libraries
[320]: proof-engine/vernacular-commands.html#inlining-hints-for-the-fast-reduction-machines
[321]: proof-engine/vernacular-commands.html#registering-primitive-operations
[322]: proofs/writing-proofs/index.html
[323]: proofs/writing-proofs/proof-mode.html
[324]: proofs/writing-proofs/proof-mode.html#proof-state
[325]: proofs/writing-proofs/proof-mode.html#entering-and-exiting-proof-mode
[326]: proofs/writing-proofs/proof-mode.html#proof-using-options
[327]: proofs/writing-proofs/proof-mode.html#name-a-set-of-section-hypotheses-for-proof-using
[328]: proofs/writing-proofs/proof-mode.html#proof-modes
[329]: proofs/writing-proofs/proof-mode.html#managing-goals
[330]: proofs/writing-proofs/proof-mode.html#focusing-goals
[331]: proofs/writing-proofs/proof-mode.html#curly-braces
[332]: proofs/writing-proofs/proof-mode.html#bullets
[333]: proofs/writing-proofs/proof-mode.html#other-focusing-commands
[334]: proofs/writing-proofs/proof-mode.html#shelving-goals
[335]: proofs/writing-proofs/proof-mode.html#reordering-goals
[336]: proofs/writing-proofs/proof-mode.html#proving-a-subgoal-as-a-separate-lemma-abstract
[337]: proofs/writing-proofs/proof-mode.html#requesting-information
[338]: proofs/writing-proofs/proof-mode.html#showing-differences-between-proof-steps
[339]: proofs/writing-proofs/proof-mode.html#how-to-enable-diffs
[340]: proofs/writing-proofs/proof-mode.html#how-diffs-are-calculated
[341]: proofs/writing-proofs/proof-mode.html#show-proof-differences
[342]: proofs/writing-proofs/proof-mode.html#delaying-solving-unification-constraints
[343]: proofs/writing-proofs/proof-mode.html#proof-maintenance
[344]: proofs/writing-proofs/proof-mode.html#controlling-proof-mode
[345]: proofs/writing-proofs/proof-mode.html#controlling-memory-usage
[346]: proof-engine/tactics.html
[347]: proof-engine/tactics.html#common-elements-of-tactics
[348]: proof-engine/tactics.html#reserved-keywords
[349]: proof-engine/tactics.html#invocation-of-tactics
[350]: proof-engine/tactics.html#bindings
[351]: proof-engine/tactics.html#intro-patterns
[352]: proof-engine/tactics.html#occurrence-clauses
[353]: proof-engine/tactics.html#automatic-clearing-of-hypotheses
[354]: proof-engine/tactics.html#applying-theorems
[355]: proof-engine/tactics.html#managing-the-local-context
[356]: proof-engine/tactics.html#controlling-the-proof-flow
[357]: proof-engine/tactics.html#classical-tactics
[358]: proof-engine/tactics.html#performance-oriented-tactic-variants
[359]: proofs/writing-proofs/equality.html
[360]: proofs/writing-proofs/equality.html#tactics-for-simple-equalities
[361]: proofs/writing-proofs/equality.html#rewriting-with-leibniz-and-setoid-equality
[362]: proofs/writing-proofs/equality.html#rewriting-with-definitional-equality
[363]: proofs/writing-proofs/equality.html#applying-conversion-rules
[364]: proofs/writing-proofs/equality.html#fast-reduction-tactics-vm-compute-and-native-compute
[365]: proofs/writing-proofs/equality.html#computing-in-a-term-eval-and-eval
[366]: proofs/writing-proofs/equality.html#controlling-reduction-strategies-and-the-conversion-algor
ithm
[367]: proofs/writing-proofs/reasoning-inductives.html
[368]: proofs/writing-proofs/reasoning-inductives.html#applying-constructors
[369]: proofs/writing-proofs/reasoning-inductives.html#case-analysis
[370]: proofs/writing-proofs/reasoning-inductives.html#induction
[371]: proofs/writing-proofs/reasoning-inductives.html#equality-of-inductive-types
[372]: proofs/writing-proofs/reasoning-inductives.html#helper-tactics
[373]: proofs/writing-proofs/reasoning-inductives.html#generation-of-induction-principles-with-schem
e
[374]: proofs/writing-proofs/reasoning-inductives.html#automatic-declaration-of-schemes
[375]: proofs/writing-proofs/reasoning-inductives.html#combined-scheme
[376]: proofs/writing-proofs/reasoning-inductives.html#generation-of-inversion-principles-with-deriv
e-inversion
[377]: proofs/writing-proofs/reasoning-inductives.html#examples-of-dependent-destruction-dependent-i
nduction
[378]: proofs/writing-proofs/reasoning-inductives.html#a-larger-example
[379]: proof-engine/ssreflect-proof-language.html
[380]: proof-engine/ssreflect-proof-language.html#introduction
[381]: proof-engine/ssreflect-proof-language.html#acknowledgments
[382]: proof-engine/ssreflect-proof-language.html#usage
[383]: proof-engine/ssreflect-proof-language.html#getting-started
[384]: proof-engine/ssreflect-proof-language.html#compatibility-issues
[385]: proof-engine/ssreflect-proof-language.html#gallina-extensions
[386]: proof-engine/ssreflect-proof-language.html#pattern-assignment
[387]: proof-engine/ssreflect-proof-language.html#pattern-conditional
[388]: proof-engine/ssreflect-proof-language.html#parametric-polymorphism
[389]: proof-engine/ssreflect-proof-language.html#anonymous-arguments
[390]: proof-engine/ssreflect-proof-language.html#wildcards
[391]: proof-engine/ssreflect-proof-language.html#definitions
[392]: proof-engine/ssreflect-proof-language.html#abbreviations
[393]: proof-engine/ssreflect-proof-language.html#matching
[394]: proof-engine/ssreflect-proof-language.html#occurrence-selection
[395]: proof-engine/ssreflect-proof-language.html#basic-localization
[396]: proof-engine/ssreflect-proof-language.html#basic-tactics
[397]: proof-engine/ssreflect-proof-language.html#bookkeeping
[398]: proof-engine/ssreflect-proof-language.html#the-defective-tactics
[399]: proof-engine/ssreflect-proof-language.html#the-move-tactic
[400]: proof-engine/ssreflect-proof-language.html#the-case-tactic
[401]: proof-engine/ssreflect-proof-language.html#the-elim-tactic
[402]: proof-engine/ssreflect-proof-language.html#the-apply-tactic
[403]: proof-engine/ssreflect-proof-language.html#discharge
[404]: proof-engine/ssreflect-proof-language.html#clear-rules
[405]: proof-engine/ssreflect-proof-language.html#matching-for-apply-and-exact
[406]: proof-engine/ssreflect-proof-language.html#the-abstract-tactic
[407]: proof-engine/ssreflect-proof-language.html#introduction-in-the-context
[408]: proof-engine/ssreflect-proof-language.html#simplification-items
[409]: proof-engine/ssreflect-proof-language.html#views
[410]: proof-engine/ssreflect-proof-language.html#intro-patterns
[411]: proof-engine/ssreflect-proof-language.html#clear-switch
[412]: proof-engine/ssreflect-proof-language.html#branching-and-destructuring
[413]: proof-engine/ssreflect-proof-language.html#block-introduction
[414]: proof-engine/ssreflect-proof-language.html#generation-of-equations
[415]: proof-engine/ssreflect-proof-language.html#type-families
[416]: proof-engine/ssreflect-proof-language.html#control-flow
[417]: proof-engine/ssreflect-proof-language.html#indentation-and-bullets
[418]: proof-engine/ssreflect-proof-language.html#terminators
[419]: proof-engine/ssreflect-proof-language.html#selectors
[420]: proof-engine/ssreflect-proof-language.html#iteration
[421]: proof-engine/ssreflect-proof-language.html#localization
[422]: proof-engine/ssreflect-proof-language.html#structure
[423]: proof-engine/ssreflect-proof-language.html#the-have-tactic
[424]: proof-engine/ssreflect-proof-language.html#generating-let-in-context-entries-with-have
[425]: proof-engine/ssreflect-proof-language.html#the-have-tactic-and-typeclass-resolution
[426]: proof-engine/ssreflect-proof-language.html#variants-the-suff-and-wlog-tactics
[427]: proof-engine/ssreflect-proof-language.html#advanced-generalization
[428]: proof-engine/ssreflect-proof-language.html#rewriting
[429]: proof-engine/ssreflect-proof-language.html#an-extended-rewrite-tactic
[430]: proof-engine/ssreflect-proof-language.html#remarks-and-examples
[431]: proof-engine/ssreflect-proof-language.html#rewrite-redex-selection
[432]: proof-engine/ssreflect-proof-language.html#chained-rewrite-steps
[433]: proof-engine/ssreflect-proof-language.html#explicit-redex-switches-are-matched-first
[434]: proof-engine/ssreflect-proof-language.html#occurrence-switches-and-redex-switches
[435]: proof-engine/ssreflect-proof-language.html#occurrence-selection-and-repetition
[436]: proof-engine/ssreflect-proof-language.html#multi-rule-rewriting
[437]: proof-engine/ssreflect-proof-language.html#wildcards-vs-abstractions
[438]: proof-engine/ssreflect-proof-language.html#when-ssr-rewrite-fails-on-standard-rocq-licit-rewr
ite
[439]: proof-engine/ssreflect-proof-language.html#existential-metavariables-and-rewriting
[440]: proof-engine/ssreflect-proof-language.html#rewriting-under-binders
[441]: proof-engine/ssreflect-proof-language.html#the-under-tactic
[442]: proof-engine/ssreflect-proof-language.html#interactive-mode
[443]: proof-engine/ssreflect-proof-language.html#the-over-tactic
[444]: proof-engine/ssreflect-proof-language.html#one-liner-mode
[445]: proof-engine/ssreflect-proof-language.html#locking-unlocking
[446]: proof-engine/ssreflect-proof-language.html#congruence
[447]: proof-engine/ssreflect-proof-language.html#contextual-patterns
[448]: proof-engine/ssreflect-proof-language.html#syntax
[449]: proof-engine/ssreflect-proof-language.html#matching-contextual-patterns
[450]: proof-engine/ssreflect-proof-language.html#examples
[451]: proof-engine/ssreflect-proof-language.html#contextual-pattern-in-set-and-the-tactical
[452]: proof-engine/ssreflect-proof-language.html#contextual-patterns-in-rewrite
[453]: proof-engine/ssreflect-proof-language.html#patterns-for-recurrent-contexts
[454]: proof-engine/ssreflect-proof-language.html#views-and-reflection
[455]: proof-engine/ssreflect-proof-language.html#interpreting-eliminations
[456]: proof-engine/ssreflect-proof-language.html#interpreting-assumptions
[457]: proof-engine/ssreflect-proof-language.html#specializing-assumptions
[458]: proof-engine/ssreflect-proof-language.html#interpreting-goals
[459]: proof-engine/ssreflect-proof-language.html#boolean-reflection
[460]: proof-engine/ssreflect-proof-language.html#the-reflect-predicate
[461]: proof-engine/ssreflect-proof-language.html#general-mechanism-for-interpreting-goals-and-assum
ptions
[462]: proof-engine/ssreflect-proof-language.html#id15
[463]: proof-engine/ssreflect-proof-language.html#id16
[464]: proof-engine/ssreflect-proof-language.html#id17
[465]: proof-engine/ssreflect-proof-language.html#interpreting-equivalences
[466]: proof-engine/ssreflect-proof-language.html#declaring-new-hint-views
[467]: proof-engine/ssreflect-proof-language.html#multiple-views
[468]: proof-engine/ssreflect-proof-language.html#additional-view-shortcuts
[469]: proof-engine/ssreflect-proof-language.html#synopsis-and-index
[470]: proof-engine/ssreflect-proof-language.html#parameters
[471]: proof-engine/ssreflect-proof-language.html#items-and-switches
[472]: proof-engine/ssreflect-proof-language.html#tactics
[473]: proof-engine/ssreflect-proof-language.html#tacticals
[474]: proof-engine/ssreflect-proof-language.html#commands
[475]: proof-engine/ssreflect-proof-language.html#settings
[476]: proofs/automatic-tactics/index.html
[477]: proofs/automatic-tactics/logic.html
[478]: addendum/micromega.html
[479]: addendum/micromega.html#short-description-of-the-tactics
[480]: addendum/micromega.html#positivstellensatz-refutations
[481]: addendum/micromega.html#lra-a-decision-procedure-for-linear-real-and-rational-arithmetic
[482]: addendum/micromega.html#lia-a-tactic-for-linear-integer-arithmetic
[483]: addendum/micromega.html#high-level-view-of-lia
[484]: addendum/micromega.html#cutting-plane-proofs
[485]: addendum/micromega.html#case-split
[486]: addendum/micromega.html#nra-a-proof-procedure-for-non-linear-arithmetic
[487]: addendum/micromega.html#nia-a-proof-procedure-for-non-linear-integer-arithmetic
[488]: addendum/micromega.html#psatz-a-proof-procedure-for-non-linear-arithmetic
[489]: addendum/micromega.html#zify-pre-processing-of-arithmetic-goals
[490]: addendum/ring.html
[491]: addendum/ring.html#what-does-this-tactic-do
[492]: addendum/ring.html#the-variables-map
[493]: addendum/ring.html#is-it-automatic
[494]: addendum/ring.html#concrete-usage
[495]: addendum/ring.html#adding-a-ring-structure
[496]: addendum/ring.html#how-does-it-work
[497]: addendum/ring.html#dealing-with-fields
[498]: addendum/ring.html#adding-a-new-field-structure
[499]: addendum/ring.html#history-of-ring
[500]: addendum/ring.html#discussion
[501]: addendum/nsatz.html
[502]: addendum/nsatz.html#more-about-nsatz
[503]: proofs/automatic-tactics/auto.html
[504]: proofs/automatic-tactics/auto.html#tactics
[505]: proofs/automatic-tactics/auto.html#hint-databases
[506]: proofs/automatic-tactics/auto.html#creating-hint-databases
[507]: proofs/automatic-tactics/auto.html#hint-databases-defined-in-the-rocq-standard-library
[508]: proofs/automatic-tactics/auto.html#creating-hints
[509]: proofs/automatic-tactics/auto.html#hint-locality
[510]: proofs/automatic-tactics/auto.html#setting-implicit-automation-tactics
[511]: addendum/generalized-rewriting.html
[512]: addendum/generalized-rewriting.html#introduction-to-generalized-rewriting
[513]: addendum/generalized-rewriting.html#relations-and-morphisms
[514]: addendum/generalized-rewriting.html#adding-new-relations-and-morphisms
[515]: addendum/generalized-rewriting.html#rewriting-and-nonreflexive-relations
[516]: addendum/generalized-rewriting.html#rewriting-and-nonsymmetric-relations
[517]: addendum/generalized-rewriting.html#rewriting-in-ambiguous-setoid-contexts
[518]: addendum/generalized-rewriting.html#rewriting-with-type-valued-relations
[519]: addendum/generalized-rewriting.html#declaring-rewrite-relations
[520]: addendum/generalized-rewriting.html#commands-and-tactics
[521]: addendum/generalized-rewriting.html#first-class-setoids-and-morphisms
[522]: addendum/generalized-rewriting.html#tactics-enabled-on-user-provided-relations
[523]: addendum/generalized-rewriting.html#printing-relations-and-morphisms
[524]: addendum/generalized-rewriting.html#understanding-and-fixing-failed-resolutions
[525]: addendum/generalized-rewriting.html#deprecated-syntax-and-backward-incompatibilities
[526]: addendum/generalized-rewriting.html#extensions
[527]: addendum/generalized-rewriting.html#rewriting-under-binders
[528]: addendum/generalized-rewriting.html#subrelations
[529]: addendum/generalized-rewriting.html#constant-unfolding-during-rewriting
[530]: addendum/generalized-rewriting.html#constant-unfolding-during-proper-instance-search
[531]: addendum/generalized-rewriting.html#strategies-for-rewriting
[532]: addendum/generalized-rewriting.html#usage
[533]: addendum/generalized-rewriting.html#definitions
[534]: proofs/creating-tactics/index.html
[535]: proof-engine/ltac.html
[536]: proof-engine/ltac.html#defects
[537]: proof-engine/ltac.html#syntax
[538]: proof-engine/ltac.html#values
[539]: proof-engine/ltac.html#syntactic-values
[540]: proof-engine/ltac.html#substitution
[541]: proof-engine/ltac.html#local-definitions-let
[542]: proof-engine/ltac.html#function-construction-and-application
[543]: proof-engine/ltac.html#tactics-in-terms
[544]: proof-engine/ltac.html#goal-selectors
[545]: proof-engine/ltac.html#processing-multiple-goals
[546]: proof-engine/ltac.html#branching-and-backtracking
[547]: proof-engine/ltac.html#control-flow
[548]: proof-engine/ltac.html#sequence
[549]: proof-engine/ltac.html#do-loop
[550]: proof-engine/ltac.html#repeat-loop
[551]: proof-engine/ltac.html#catching-errors-try
[552]: proof-engine/ltac.html#conditional-branching-tryif
[553]: proof-engine/ltac.html#alternatives
[554]: proof-engine/ltac.html#branching-with-backtracking
[555]: proof-engine/ltac.html#local-application-of-tactics
[556]: proof-engine/ltac.html#first-tactic-to-succeed
[557]: proof-engine/ltac.html#solving
[558]: proof-engine/ltac.html#first-tactic-to-make-progress
[559]: proof-engine/ltac.html#detecting-progress
[560]: proof-engine/ltac.html#success-and-failure
[561]: proof-engine/ltac.html#checking-for-success-assert-succeeds
[562]: proof-engine/ltac.html#checking-for-failure-assert-fails
[563]: proof-engine/ltac.html#failing
[564]: proof-engine/ltac.html#soft-cut-once
[565]: proof-engine/ltac.html#checking-for-a-single-success-exactly-once
[566]: proof-engine/ltac.html#manipulating-values
[567]: proof-engine/ltac.html#pattern-matching-on-terms-match
[568]: proof-engine/ltac.html#pattern-matching-on-goals-and-hypotheses-match-goal
[569]: proof-engine/ltac.html#filling-a-term-context
[570]: proof-engine/ltac.html#generating-fresh-hypothesis-names
[571]: proof-engine/ltac.html#computing-in-a-term-eval
[572]: proof-engine/ltac.html#getting-the-type-of-a-term
[573]: proof-engine/ltac.html#manipulating-untyped-terms-type-term
[574]: proof-engine/ltac.html#counting-goals-numgoals
[575]: proof-engine/ltac.html#testing-boolean-expressions-guard
[576]: proof-engine/ltac.html#checking-properties-of-terms
[577]: proof-engine/ltac.html#timing
[578]: proof-engine/ltac.html#timeout
[579]: proof-engine/ltac.html#timing-a-tactic
[580]: proof-engine/ltac.html#timing-a-tactic-that-evaluates-to-a-term-time-constr
[581]: proof-engine/ltac.html#print-identity-tactic-idtac
[582]: proof-engine/ltac.html#tactic-toplevel-definitions
[583]: proof-engine/ltac.html#defining-ltac-symbols
[584]: proof-engine/ltac.html#printing-ltac-tactics
[585]: proof-engine/ltac.html#examples-of-using-ltac
[586]: proof-engine/ltac.html#proof-that-the-natural-numbers-have-at-least-two-elements
[587]: proof-engine/ltac.html#proving-that-a-list-is-a-permutation-of-a-second-list
[588]: proof-engine/ltac.html#deciding-intuitionistic-propositional-logic
[589]: proof-engine/ltac.html#deciding-type-isomorphisms
[590]: proof-engine/ltac.html#debugging-ltac-tactics
[591]: proof-engine/ltac.html#backtraces
[592]: proof-engine/ltac.html#tracing-execution
[593]: proof-engine/ltac.html#interactive-debugger
[594]: proof-engine/ltac.html#profiling-ltac-tactics
[595]: proof-engine/ltac.html#run-time-optimization-tactic
[596]: proof-engine/ltac2.html
[597]: proof-engine/ltac2.html#general-design
[598]: proof-engine/ltac2.html#ml-component
[599]: proof-engine/ltac2.html#overview
[600]: proof-engine/ltac2.html#type-syntax
[601]: proof-engine/ltac2.html#type-declarations
[602]: proof-engine/ltac2.html#apis
[603]: proof-engine/ltac2.html#term-syntax
[604]: proof-engine/ltac2.html#ltac2-definitions
[605]: proof-engine/ltac2.html#printing-ltac2-tactics
[606]: proof-engine/ltac2.html#reduction
[607]: proof-engine/ltac2.html#typing
[608]: proof-engine/ltac2.html#effects
[609]: proof-engine/ltac2.html#standard-io
[610]: proof-engine/ltac2.html#fatal-errors
[611]: proof-engine/ltac2.html#backtracking
[612]: proof-engine/ltac2.html#goals
[613]: proof-engine/ltac2.html#meta-programming
[614]: proof-engine/ltac2.html#id3
[615]: proof-engine/ltac2.html#quotations
[616]: proof-engine/ltac2.html#built-in-quotations
[617]: proof-engine/ltac2.html#strict-vs-non-strict-mode
[618]: proof-engine/ltac2.html#term-antiquotations
[619]: proof-engine/ltac2.html#syntax
[620]: proof-engine/ltac2.html#semantics
[621]: proof-engine/ltac2.html#static-semantics
[622]: proof-engine/ltac2.html#dynamic-semantics
[623]: proof-engine/ltac2.html#match-over-terms
[624]: proof-engine/ltac2.html#match-over-goals
[625]: proof-engine/ltac2.html#match-on-values
[626]: proof-engine/ltac2.html#notations
[627]: proof-engine/ltac2.html#abbreviations
[628]: proof-engine/ltac2.html#defining-tactics
[629]: proof-engine/ltac2.html#syntactic-classes
[630]: proof-engine/ltac2.html#evaluation
[631]: proof-engine/ltac2.html#debug
[632]: proof-engine/ltac2.html#profiling
[633]: proof-engine/ltac2.html#compatibility-layer-with-ltac1
[634]: proof-engine/ltac2.html#ltac1-from-ltac2
[635]: proof-engine/ltac2.html#simple-api
[636]: proof-engine/ltac2.html#low-level-api
[637]: proof-engine/ltac2.html#ltac2-from-ltac1
[638]: proof-engine/ltac2.html#switching-between-ltac-languages
[639]: proof-engine/ltac2.html#transition-from-ltac1
[640]: proof-engine/ltac2.html#syntax-changes
[641]: proof-engine/ltac2.html#tactic-delay
[642]: proof-engine/ltac2.html#variable-binding
[643]: proof-engine/ltac2.html#in-ltac-expressions
[644]: proof-engine/ltac2.html#in-quotations
[645]: proof-engine/ltac2.html#exception-catching
[646]: using/libraries/index.html
[647]: language/coq-library.html
[648]: language/coq-library.html#the-prelude
[649]: language/coq-library.html#notations
[650]: language/coq-library.html#logic
[651]: language/coq-library.html#propositional-connectives
[652]: language/coq-library.html#quantifiers
[653]: language/coq-library.html#equality
[654]: language/coq-library.html#lemmas
[655]: language/coq-library.html#datatypes
[656]: language/coq-library.html#programming
[657]: language/coq-library.html#specification
[658]: language/coq-library.html#basic-arithmetic
[659]: language/coq-library.html#well-founded-recursion
[660]: language/coq-library.html#tactics
[661]: language/coq-library.html#opam-repository
[662]: addendum/extraction.html
[663]: addendum/extraction.html#generating-ml-code
[664]: addendum/extraction.html#extraction-options
[665]: addendum/extraction.html#setting-the-target-language
[666]: addendum/extraction.html#inlining-and-optimizations
[667]: addendum/extraction.html#extra-elimination-of-useless-arguments
[668]: addendum/extraction.html#accessing-opaque-proofs
[669]: addendum/extraction.html#realizing-axioms
[670]: addendum/extraction.html#realizing-inductive-types
[671]: addendum/extraction.html#generating-ffi-code
[672]: addendum/extraction.html#avoiding-conflicts-with-existing-filenames
[673]: addendum/extraction.html#additional-settings
[674]: addendum/extraction.html#differences-between-rocq-and-ml-type-systems
[675]: addendum/extraction.html#some-examples
[676]: addendum/extraction.html#a-detailed-example-euclidean-division
[677]: addendum/extraction.html#extraction-s-horror-museum
[678]: addendum/extraction.html#users-contributions
[679]: addendum/miscellaneous-extensions.html
[680]: using/libraries/funind.html
[681]: using/libraries/funind.html#advanced-recursive-functions
[682]: using/libraries/funind.html#tactics
[683]: using/libraries/funind.html#generation-of-induction-principles-with-functional-scheme
[684]: using/libraries/funind.html#flags
[685]: using/libraries/writing.html
[686]: using/libraries/writing.html#deprecating-library-objects-tactics-or-library-files
[687]: using/libraries/writing.html#triggering-warning-for-library-objects-or-library-files
[688]: using/tools/index.html
[689]: practical-tools/utilities.html
[690]: practical-tools/utilities.html#rocq-configuration-basics
[691]: practical-tools/utilities.html#installing-the-rocq-prover-and-rocq-packages-with-opam
[692]: practical-tools/utilities.html#setup-for-working-on-your-own-projects
[693]: practical-tools/utilities.html#building-a-project-with-coqproject-overview
[694]: practical-tools/utilities.html#logical-paths-and-the-load-path
[695]: practical-tools/utilities.html#modifying-multiple-interdependent-projects-at-the-same-time
[696]: practical-tools/utilities.html#installed-and-uninstalled-packages
[697]: practical-tools/utilities.html#upgrading-to-a-new-version-of-rocq
[698]: practical-tools/utilities.html#building-a-rocq-project-with-rocq-makefile-details
[699]: practical-tools/utilities.html#comments
[700]: practical-tools/utilities.html#quoting-arguments-to-rocq-c
[701]: practical-tools/utilities.html#forbidden-filenames
[702]: practical-tools/utilities.html#warning-no-common-logical-root
[703]: practical-tools/utilities.html#coqmakefile-local
[704]: practical-tools/utilities.html#coqmakefile-local-late
[705]: practical-tools/utilities.html#timing-targets-and-performance-testing
[706]: practical-tools/utilities.html#building-a-subset-of-the-targets-with-j
[707]: practical-tools/utilities.html#precompiling-for-native-compute
[708]: practical-tools/utilities.html#the-grammar-of-coqproject
[709]: practical-tools/utilities.html#building-a-rocq-project-with-dune
[710]: practical-tools/utilities.html#rocq-dep-computing-module-dependencies
[711]: practical-tools/utilities.html#split-compilation-of-native-computation-files
[712]: practical-tools/utilities.html#using-rocq-as-a-library
[713]: practical-tools/utilities.html#embedded-rocq-phrases-inside-latex-documents
[714]: practical-tools/utilities.html#man-pages
[715]: practical-tools/coq-commands.html
[716]: practical-tools/coq-commands.html#interactive-use-rocq-repl
[717]: practical-tools/coq-commands.html#batch-compilation-rocq-compile
[718]: practical-tools/coq-commands.html#system-configuration
[719]: practical-tools/coq-commands.html#customization-at-launch-time
[720]: practical-tools/coq-commands.html#command-parameters
[721]: practical-tools/coq-commands.html#coqrc-start-up-script
[722]: practical-tools/coq-commands.html#environment-variables
[723]: practical-tools/coq-commands.html#command-line-options
[724]: practical-tools/coq-commands.html#profiling
[725]: practical-tools/coq-commands.html#compiled-interfaces-produced-using-vos
[726]: practical-tools/coq-commands.html#compiled-libraries-checker-rocqchk
[727]: using/tools/coqdoc.html
[728]: using/tools/coqdoc.html#principles
[729]: using/tools/coqdoc.html#rocq-material-inside-documentation
[730]: using/tools/coqdoc.html#pretty-printing
[731]: using/tools/coqdoc.html#sections
[732]: using/tools/coqdoc.html#lists
[733]: using/tools/coqdoc.html#rules
[734]: using/tools/coqdoc.html#emphasis
[735]: using/tools/coqdoc.html#escaping-to-latex-and-html
[736]: using/tools/coqdoc.html#verbatim
[737]: using/tools/coqdoc.html#hyperlinks
[738]: using/tools/coqdoc.html#hiding-showing-parts-of-the-source
[739]: using/tools/coqdoc.html#usage
[740]: using/tools/coqdoc.html#command-line-options
[741]: using/tools/coqdoc.html#the-rocq-doc-latex-style-file
[742]: practical-tools/coqide.html
[743]: practical-tools/coqide.html#managing-files-and-buffers-basic-editing
[744]: practical-tools/coqide.html#running-coq-scripts
[745]: practical-tools/coqide.html#asynchronous-mode
[746]: practical-tools/coqide.html#commands-and-templates
[747]: practical-tools/coqide.html#queries
[748]: practical-tools/coqide.html#compilation
[749]: practical-tools/coqide.html#customizations
[750]: practical-tools/coqide.html#preferences
[751]: practical-tools/coqide.html#key-bindings
[752]: practical-tools/coqide.html#using-unicode-symbols
[753]: practical-tools/coqide.html#displaying-unicode-symbols
[754]: practical-tools/coqide.html#bindings-for-input-of-unicode-symbols
[755]: practical-tools/coqide.html#adding-custom-bindings
[756]: practical-tools/coqide.html#character-encoding-for-saved-files
[757]: practical-tools/coqide.html#debugger
[758]: practical-tools/coqide.html#breakpoints
[759]: practical-tools/coqide.html#call-stack-and-variables
[760]: practical-tools/coqide.html#supported-use-cases
[761]: addendum/parallel-proof-processing.html
[762]: addendum/parallel-proof-processing.html#proof-annotations
[763]: addendum/parallel-proof-processing.html#automatic-suggestion-of-proof-annotations
[764]: addendum/parallel-proof-processing.html#proof-blocks-and-error-resilience
[765]: addendum/parallel-proof-processing.html#caveats
[766]: addendum/parallel-proof-processing.html#interactive-mode
[767]: addendum/parallel-proof-processing.html#limiting-the-number-of-parallel-workers
[768]: addendum/parallel-proof-processing.html#id3
[769]: appendix/history-and-changes/index.html
[770]: history.html
[771]: history.html#historical-roots
[772]: history.html#versions-1-to-5
[773]: history.html#version-1
[774]: history.html#version-2
[775]: history.html#version-3
[776]: history.html#version-4
[777]: history.html#version-5
[778]: history.html#versions-6
[779]: history.html#version-6-1
[780]: history.html#version-6-2
[781]: history.html#version-6-3
[782]: history.html#versions-7
[783]: history.html#summary-of-changes
[784]: history.html#details-of-changes-in-7-0-and-7-1
[785]: history.html#main-novelties
[786]: history.html#details-of-changes
[787]: history.html#language-new-let-in-construction
[788]: history.html#language-long-names
[789]: history.html#language-miscellaneous
[790]: history.html#language-cases
[791]: history.html#reduction
[792]: history.html#new-tactics
[793]: history.html#changes-in-existing-tactics
[794]: history.html#efficiency
[795]: history.html#concrete-syntax-of-constructions
[796]: history.html#parsing-and-grammar-extension
[797]: history.html#new-commands
[798]: history.html#changes-in-existing-commands
[799]: history.html#tools
[800]: history.html#extraction
[801]: history.html#standard-library
[802]: history.html#new-user-contributions
[803]: history.html#details-of-changes-in-7-2
[804]: history.html#details-of-changes-in-7-3
[805]: history.html#changes-in-7-3-1
[806]: history.html#details-of-changes-in-7-4
[807]: changes.html
[808]: changes.html#version-9-1
[809]: changes.html#summary-of-changes
[810]: changes.html#changes-in-9-1-0
[811]: changes.html#id3
[812]: changes.html#id16
[813]: changes.html#id23
[814]: changes.html#id26
[815]: changes.html#id33
[816]: changes.html#id38
[817]: changes.html#id63
[818]: changes.html#id68
[819]: changes.html#id81
[820]: changes.html#id86
[821]: changes.html#id91
[822]: changes.html#id93
[823]: changes.html#id100
[824]: changes.html#id102
[825]: changes.html#version-9-0
[826]: changes.html#id105
[827]: changes.html#porting-to-the-rocq-prover
[828]: changes.html#renaming-advice
[829]: changes.html#the-rocq-prover-website
[830]: changes.html#changes-in-9-0-0
[831]: changes.html#id116
[832]: changes.html#id123
[833]: changes.html#id142
[834]: changes.html#id145
[835]: changes.html#id159
[836]: changes.html#id170
[837]: changes.html#id174
[838]: changes.html#id196
[839]: changes.html#id202
[840]: changes.html#standard-library
[841]: changes.html#id214
[842]: changes.html#id219
[843]: changes.html#version-8-20
[844]: changes.html#id222
[845]: changes.html#changes-in-8-20-0
[846]: changes.html#id232
[847]: changes.html#changes-spec-language
[848]: changes.html#id260
[849]: changes.html#id269
[850]: changes.html#id296
[851]: changes.html#id301
[852]: changes.html#id326
[853]: changes.html#id335
[854]: changes.html#id366
[855]: changes.html#coqide
[856]: changes.html#renaming-stdlib
[857]: changes.html#id401
[858]: changes.html#id418
[859]: changes.html#changes-in-8-20-1
[860]: changes.html#id424
[861]: changes.html#id427
[862]: changes.html#id432
[863]: changes.html#version-8-19
[864]: changes.html#id435
[865]: changes.html#changes-in-8-19-0
[866]: changes.html#id447
[867]: changes.html#id452
[868]: changes.html#id464
[869]: changes.html#id499
[870]: changes.html#id534
[871]: changes.html#ltac2
[872]: changes.html#id556
[873]: changes.html#id574
[874]: changes.html#id582
[875]: changes.html#id595
[876]: changes.html#changes-in-8-19-1
[877]: changes.html#id599
[878]: changes.html#id604
[879]: changes.html#id606
[880]: changes.html#id609
[881]: changes.html#id614
[882]: changes.html#changes-in-8-19-2
[883]: changes.html#id619
[884]: changes.html#id622
[885]: changes.html#id625
[886]: changes.html#id628
[887]: changes.html#id630
[888]: changes.html#id634
[889]: changes.html#id637
[890]: changes.html#version-8-18
[891]: changes.html#id641
[892]: changes.html#changes-in-8-18-0
[893]: changes.html#id643
[894]: changes.html#id649
[895]: changes.html#id662
[896]: changes.html#id675
[897]: changes.html#id695
[898]: changes.html#id713
[899]: changes.html#id744
[900]: changes.html#id753
[901]: changes.html#id756
[902]: changes.html#id777
[903]: changes.html#id780
[904]: changes.html#version-8-17
[905]: changes.html#id783
[906]: changes.html#changes-in-8-17-0
[907]: changes.html#id789
[908]: changes.html#id794
[909]: changes.html#id800
[910]: changes.html#id808
[911]: changes.html#id828
[912]: changes.html#id831
[913]: changes.html#id855
[914]: changes.html#id857
[915]: changes.html#id886
[916]: changes.html#id892
[917]: changes.html#id912
[918]: changes.html#id932
[919]: changes.html#changes-in-8-17-1
[920]: changes.html#version-8-16
[921]: changes.html#id943
[922]: changes.html#changes-in-8-16-0
[923]: changes.html#id950
[924]: changes.html#id962
[925]: changes.html#id974
[926]: changes.html#id980
[927]: changes.html#tactic-language
[928]: changes.html#id1014
[929]: changes.html#id1016
[930]: changes.html#id1042
[931]: changes.html#id1047
[932]: changes.html#id1049
[933]: changes.html#id1068
[934]: changes.html#id1080
[935]: changes.html#changes-in-8-16-1
[936]: changes.html#id1091
[937]: changes.html#id1100
[938]: changes.html#id1103
[939]: changes.html#version-8-15
[940]: changes.html#id1106
[941]: changes.html#changes-in-8-15-0
[942]: changes.html#id1113
[943]: changes.html#id1116
[944]: changes.html#id1127
[945]: changes.html#id1144
[946]: changes.html#id1191
[947]: changes.html#id1194
[948]: changes.html#id1203
[949]: changes.html#id1235
[950]: changes.html#id1250
[951]: changes.html#id1257
[952]: changes.html#id1270
[953]: changes.html#id1276
[954]: changes.html#changes-in-8-15-1
[955]: changes.html#id1281
[956]: changes.html#id1286
[957]: changes.html#id1289
[958]: changes.html#id1298
[959]: changes.html#id1300
[960]: changes.html#id1309
[961]: changes.html#changes-in-8-15-2
[962]: changes.html#id1314
[963]: changes.html#id1319
[964]: changes.html#id1331
[965]: changes.html#version-8-14
[966]: changes.html#id1334
[967]: changes.html#changes-in-8-14-0
[968]: changes.html#id1341
[969]: changes.html#id1348
[970]: changes.html#id1359
[971]: changes.html#id1376
[972]: changes.html#id1413
[973]: changes.html#id1433
[974]: changes.html#id1436
[975]: changes.html#id1459
[976]: changes.html#native-compilation
[977]: changes.html#id1474
[978]: changes.html#id1478
[979]: changes.html#id1505
[980]: changes.html#id1511
[981]: changes.html#changes-in-8-14-1
[982]: changes.html#id1514
[983]: changes.html#id1517
[984]: changes.html#id1520
[985]: changes.html#id1525
[986]: changes.html#version-8-13
[987]: changes.html#id1530
[988]: changes.html#changes-in-8-13-beta1
[989]: changes.html#id1535
[990]: changes.html#id1545
[991]: changes.html#id1574
[992]: changes.html#id1607
[993]: changes.html#id1628
[994]: changes.html#id1634
[995]: changes.html#id1638
[996]: changes.html#tools
[997]: changes.html#id1656
[998]: changes.html#id1659
[999]: changes.html#id1674
[1000]: changes.html#changes-in-8-13-0
[1001]: changes.html#id1680
[1002]: changes.html#changes-in-8-13-1
[1003]: changes.html#id1682
[1004]: changes.html#id1684
[1005]: changes.html#changes-in-8-13-2
[1006]: changes.html#id1686
[1007]: changes.html#id1691
[1008]: changes.html#version-8-12
[1009]: changes.html#id1694
[1010]: changes.html#changes-in-8-12-beta1
[1011]: changes.html#id1700
[1012]: changes.html#id1703
[1013]: changes.html#id1724
[1014]: changes.html#id1745
[1015]: changes.html#id1786
[1016]: changes.html#id1795
[1017]: changes.html#flags-options-and-attributes
[1018]: changes.html#id1809
[1019]: changes.html#id1827
[1020]: changes.html#id1862
[1021]: changes.html#id1865
[1022]: changes.html#reals-library
[1023]: changes.html#id1902
[1024]: changes.html#refman
[1025]: changes.html#id1940
[1026]: changes.html#changes-in-8-12-0
[1027]: changes.html#changes-in-8-12-1
[1028]: changes.html#changes-in-8-12-2
[1029]: changes.html#version-8-11
[1030]: changes.html#id2000
[1031]: changes.html#changes-in-8-11-beta1
[1032]: changes.html#changes-in-8-11-0
[1033]: changes.html#changes-in-8-11-1
[1034]: changes.html#changes-in-8-11-2
[1035]: changes.html#version-8-10
[1036]: changes.html#id2138
[1037]: changes.html#other-changes-in-8-10-beta1
[1038]: changes.html#changes-in-8-10-beta2
[1039]: changes.html#changes-in-8-10-beta3
[1040]: changes.html#changes-in-8-10-0
[1041]: changes.html#changes-in-8-10-1
[1042]: changes.html#changes-in-8-10-2
[1043]: changes.html#version-8-9
[1044]: changes.html#id2285
[1045]: changes.html#details-of-changes-in-8-9-beta1
[1046]: changes.html#changes-in-8-8-0
[1047]: changes.html#changes-in-8-8-1
[1048]: changes.html#version-8-8
[1049]: changes.html#id2287
[1050]: changes.html#details-of-changes-in-8-8-beta1
[1051]: changes.html#details-of-changes-in-8-8-0
[1052]: changes.html#details-of-changes-in-8-8-1
[1053]: changes.html#details-of-changes-in-8-8-2
[1054]: changes.html#version-8-7
[1055]: changes.html#id2288
[1056]: changes.html#potential-compatibility-issues
[1057]: changes.html#details-of-changes-in-8-7-beta1
[1058]: changes.html#details-of-changes-in-8-7-beta2
[1059]: changes.html#details-of-changes-in-8-7-0
[1060]: changes.html#details-of-changes-in-8-7-1
[1061]: changes.html#details-of-changes-in-8-7-2
[1062]: changes.html#version-8-6
[1063]: changes.html#id2289
[1064]: changes.html#potential-sources-of-incompatibilities
[1065]: changes.html#details-of-changes-in-8-6beta1
[1066]: changes.html#details-of-changes-in-8-6
[1067]: changes.html#details-of-changes-in-8-6-1
[1068]: changes.html#version-8-5
[1069]: changes.html#id2290
[1070]: changes.html#id2291
[1071]: changes.html#details-of-changes-in-8-5beta1
[1072]: changes.html#details-of-changes-in-8-5beta2
[1073]: changes.html#details-of-changes-in-8-5beta3
[1074]: changes.html#details-of-changes-in-8-5
[1075]: changes.html#details-of-changes-in-8-5pl1
[1076]: changes.html#details-of-changes-in-8-5pl2
[1077]: changes.html#details-of-changes-in-8-5pl3
[1078]: changes.html#version-8-4
[1079]: changes.html#id2292
[1080]: changes.html#id2294
[1081]: changes.html#details-of-changes-in-8-4beta
[1082]: changes.html#details-of-changes-in-8-4beta2
[1083]: changes.html#details-of-changes-in-8-4
[1084]: changes.html#version-8-3
[1085]: changes.html#id2295
[1086]: changes.html#details-of-changes
[1087]: changes.html#version-8-2
[1088]: changes.html#id2296
[1089]: changes.html#id2297
[1090]: changes.html#version-8-1
[1091]: changes.html#id2298
[1092]: changes.html#details-of-changes-in-8-1beta
[1093]: changes.html#details-of-changes-in-8-1gamma
[1094]: changes.html#details-of-changes-in-8-1
[1095]: changes.html#version-8-0
[1096]: changes.html#id2299
[1097]: changes.html#details-of-changes-in-8-0beta-old-syntax
[1098]: changes.html#details-of-changes-in-8-0beta-new-syntax
[1099]: changes.html#details-of-changes-in-8-0
[1100]: appendix/indexes/index.html
[1101]: std-glossindex.html
[1102]: coq-cmdindex.html
[1103]: coq-tacindex.html
[1104]: coq-attrindex.html
[1105]: coq-optindex.html
[1106]: coq-exnindex.html
[1107]: genindex.html
[1108]: zebibliography.html
[1109]: http://www.opencontent.org/openpub
[1110]: language/core/index.html
[1111]: https://www.sphinx-doc.org/
[1112]: https://github.com/readthedocs/sphinx_rtd_theme
[1113]: https://readthedocs.org
