* Language Reference

# Language Reference[][1]

* [Abstract definitions][2]
  
  * [Synopsis][3]
  * [Examples][4]
  * [Scope of abstraction][5]
  * [Abstract definitions with where-blocks][6]
* [Built-ins][7]
  
  * [Using the built-in types][8]
  * [The unit type][9]
  * [The Σ-type][10]
  * [Lists][11]
  * [Maybe][12]
  * [Booleans][13]
  * [Natural numbers][14]
  * [Machine words][15]
  * [Integers][16]
  * [Floats][17]
  * [Characters][18]
  * [Strings][19]
  * [Equality][20]
  * [Sorts][21]
  * [Universe levels][22]
  * [Sized types][23]
  * [Coinduction][24]
  * [IO][25]
  * [Literal overloading][26]
  * [Reflection][27]
  * [Rewriting][28]
  * [Static values][29]
  * [Strictness][30]
* [Coinduction][31]
  
  * [Coinductive Records][32]
  * [Old Coinduction][33]
* [Copatterns][34]
  
  * [Copatterns in function definitions][35]
  * [Mixing patterns and copatterns][36]
* [Core language][37]
  
  * [Grammar][38]
  * [Syntax overview][39]
  * [Lexer][40]
  * [Parser][41]
  * [Concrete Syntax][42]
  * [Nice Concrete Syntax][43]
  * [Abstract Syntax][44]
  * [Internal Syntax][45]
  * [Treeless Syntax][46]
* [Coverage Checking][47]
  
  * [Single match on a non-indexed datatype][48]
  * [Matching on multiple arguments][49]
  * [Copattern matching][50]
  * [Matching on indexed datatypes][51]
  * [General case][52]
* [Cubical][53]
  
  * [The interval and path types][54]
  * [Transport][55]
  * [Partial elements][56]
  * [Homogeneous composition][57]
  * [Glue types][58]
  * [Higher inductive types][59]
  * [Indexed inductive types][60]
  * [Variants][61]
  * [References][62]
  * [Appendix: Cubical Agda primitives][63]
* [Cubical compatible][64]
* [Cumulativity][65]
  
  * [Basics][66]
  * [Example usage: N-ary functions][67]
  * [Limitations][68]
  * [Constraint solving][69]
* [Data Types][70]
  
  * [Simple datatypes][71]
  * [Parametrized datatypes][72]
  * [Indexed datatypes][73]
  * [Strict positivity][74]
* [Flat Modality][75]
  
  * [Pattern Matching on `@♭`][76]
* [Foreign Function Interface][77]
  
  * [Compiler Pragmas][78]
  * [Haskell FFI][79]
  * [JavaScript FFI][80]
* [Function Definitions][81]
  
  * [Introduction][82]
  * [General form][83]
  * [Special patterns][84]
  * [Case trees][85]
* [Function Types][86]
  
  * [Notational conventions][87]
* [Generalization of Declared Variables][88]
  
  * [Overview][89]
  * [Nested generalization][90]
  * [Placement of generalized bindings][91]
  * [Instance and irrelevant variables][92]
  * [Importing and exporting variables][93]
  * [Interaction][94]
  * [Modalities][95]
* [Guarded Type Theory][96]
  
  * [References][97]
* [Implicit Arguments][98]
  
  * [Tactic arguments][99]
  * [Metavariables][100]
  * [Unification][101]
* [Instance Arguments][102]
  
  * [Usage][103]
  * [Overlap and backtracking][104]
  * [Instance resolution][105]
* [Irrelevance][106]
  
  * [Motivating example][107]
  * [Irrelevant function types][108]
  * [Irrelevant declarations][109]
  * [Irrelevant record fields][110]
  * [Dependent irrelevant function types][111]
  * [Irrelevant instance arguments][112]
* [Lambda Abstraction][113]
  
  * [Lambda expressions][114]
  * [Pattern lambda][115]
  * [Absurd lambda][116]
* [Local Definitions: let and where][117]
  
  * [let-expressions][118]
  * [where-blocks][119]
  * [Proving properties][120]
  * [More Examples (for Beginners)][121]
* [Lexical Structure][122]
  
  * [Tokens][123]
  * [Layout][124]
  * [Literate Agda][125]
* [Literal Overloading][126]
  
  * [Natural numbers][127]
  * [Negative numbers][128]
  * [Strings][129]
  * [Restrictions][130]
* [Lossy Unification][131]
  
  * [Heuristic][132]
  * [Example][133]
  * [Drawbacks][134]
* [Mixfix Operators][135]
  
  * [Precedence][136]
  * [Associativity][137]
  * [Ambiguity and Scope][138]
  * [Operators in telescopes][139]
* [Modalities][140]
  
  * [General modalities][141]
  * [Positional modality systems][142]
  * [Pure modality systems][143]
* [Module System][144]
  
  * [Basics][145]
  * [Private definitions][146]
  * [Name modifiers][147]
  * [Re-exporting names][148]
  * [Parameterised modules][149]
  * [Splitting a program over multiple files][150]
  * [Datatype modules and record modules][151]
  * [References][152]
* [Mutual Recursion][153]
  
  * [Interleaved mutual blocks][154]
  * [Forward declaration][155]
  * [Old-style `mutual` blocks][156]
* [Opaque definitions][157]
  
  * [Overview][158]
  * [Unfolding opaque definitions][159]
  * [What actually unfolds?][160]
  * [Unfolding in types][161]
  * [Bibliography][162]
* [Pattern Synonyms][163]
  
  * [Overloading][164]
  * [Refolding][165]
* [Polarity Annotations][166]
  
  * [The polarity modality][167]
  * [Positivity checking][168]
  * [References][169]
* [Positivity Checking][170]
  
  * [The `NO_POSITIVITY_CHECK` pragma][171]
  * [POLARITY pragmas][172]
* [Postulates][173]
  
  * [Postulated built-ins][174]
  * [Local uses of `postulate`][175]
* [Pragmas][176]
  
  * [Index of pragmas][177]
* [Prop][178]
  
  * [Usage][179]
  * [The predicative hierarchy of `Prop`][180]
  * [The propositional squash type][181]
  * [Limitations][182]
* [Record Types][183]
  
  * [Example: the Pair type constructor][184]
  * [Declaring, constructing and decomposing records][185]
  * [Record modules][186]
  * [Eta-expansion][187]
  * [Recursive records][188]
  * [Instance fields][189]
* [Reflection][190]
  
  * [Builtin types][191]
  * [Metaprogramming][192]
* [Rewriting][193]
  
  * [Rewrite rules by example][194]
  * [General shape of rewrite rules][195]
  * [Confluence checking][196]
  * [Advanced usage][197]
  * [Importing rewrite rules][198]
* [Run-time Irrelevance][199]
  
  * [Syntax][200]
  * [Rules][201]
  * [References][202]
* [Safe Agda][203]
* [Sized Types][204]
  
  * [Example for coinduction: finite languages][205]
  * [References][206]
* [Sort System][207]
  
  * [Introduction to universes][208]
  * [Agda’s sort system][209]
  * [Sort metavariables and unknown sorts][210]
* [Syntactic Sugar][211]
  
  * [Hidden argument puns][212]
  * [Do-notation][213]
  * [Idiom brackets][214]
* [Syntax Declarations][215]
* [Telescopes][216]
  
  * [Irrefutable Patterns in Binding Positions][217]
  * [Let Bindings in Telescopes][218]
* [Termination Checking][219]
  
  * [Primitive recursion][220]
  * [Structural recursion][221]
  * [With-functions][222]
  * [Pragmas and Options][223]
  * [References][224]
* [Two-Level Type Theory][225]
  
  * [Basics][226]
* [Universe Levels][227]
  
  * [Level arithmetic][228]
  * [Intrinsic level properties][229]
  * [`forall` notation][230]
  * [Expressions of sort `Setω`][231]
  * [Pragmas and options][232]
* [With-Abstraction][233]
  
  * [Usage][234]
  * [Technical details][235]
* [Without K][236]
  
  * [Restrictions on pattern matching][237]
  * [Restrictions on termination checking][238]
  * [Restrictions on universe levels][239]
[ Previous][240] [Next ][241]

© Copyright 2005–2025 remains with the authors..

Built with [Sphinx][242] using a [theme][243] provided by [Read the Docs][244].

[1]: #language-reference
[2]: abstract-definitions.html
[3]: abstract-definitions.html#synopsis
[4]: abstract-definitions.html#examples
[5]: abstract-definitions.html#scope-of-abstraction
[6]: abstract-definitions.html#abstract-definitions-with-where-blocks
[7]: built-ins.html
[8]: built-ins.html#using-the-built-in-types
[9]: built-ins.html#the-unit-type
[10]: built-ins.html#the-type
[11]: built-ins.html#lists
[12]: built-ins.html#maybe
[13]: built-ins.html#booleans
[14]: built-ins.html#natural-numbers
[15]: built-ins.html#machine-words
[16]: built-ins.html#integers
[17]: built-ins.html#floats
[18]: built-ins.html#characters
[19]: built-ins.html#strings
[20]: built-ins.html#equality
[21]: built-ins.html#sorts
[22]: built-ins.html#universe-levels
[23]: built-ins.html#sized-types
[24]: built-ins.html#coinduction
[25]: built-ins.html#io
[26]: built-ins.html#literal-overloading
[27]: built-ins.html#reflection
[28]: built-ins.html#rewriting
[29]: built-ins.html#static-values
[30]: built-ins.html#strictness
[31]: coinduction.html
[32]: coinduction.html#coinductive-records
[33]: coinduction.html#old-coinduction
[34]: copatterns.html
[35]: copatterns.html#copatterns-in-function-definitions
[36]: copatterns.html#mixing-patterns-and-copatterns
[37]: core-language.html
[38]: core-language.html#grammar
[39]: core-language.html#syntax-overview
[40]: core-language.html#lexer
[41]: core-language.html#parser
[42]: core-language.html#concrete-syntax
[43]: core-language.html#nice-concrete-syntax
[44]: core-language.html#abstract-syntax
[45]: core-language.html#internal-syntax
[46]: core-language.html#treeless-syntax
[47]: coverage-checking.html
[48]: coverage-checking.html#single-match-on-a-non-indexed-datatype
[49]: coverage-checking.html#matching-on-multiple-arguments
[50]: coverage-checking.html#copattern-matching
[51]: coverage-checking.html#matching-on-indexed-datatypes
[52]: coverage-checking.html#general-case
[53]: cubical.html
[54]: cubical.html#the-interval-and-path-types
[55]: cubical.html#transport
[56]: cubical.html#partial-elements
[57]: cubical.html#homogeneous-composition
[58]: cubical.html#glue-types
[59]: cubical.html#higher-inductive-types
[60]: cubical.html#indexed-inductive-types
[61]: cubical.html#variants
[62]: cubical.html#references
[63]: cubical.html#appendix-cubical-agda-primitives
[64]: cubical-compatible.html
[65]: cumulativity.html
[66]: cumulativity.html#basics
[67]: cumulativity.html#example-usage-n-ary-functions
[68]: cumulativity.html#limitations
[69]: cumulativity.html#constraint-solving
[70]: data-types.html
[71]: data-types.html#simple-datatypes
[72]: data-types.html#parametrized-datatypes
[73]: data-types.html#indexed-datatypes
[74]: data-types.html#strict-positivity
[75]: flat.html
[76]: flat.html#pattern-matching-on
[77]: foreign-function-interface.html
[78]: foreign-function-interface.html#compiler-pragmas
[79]: foreign-function-interface.html#haskell-ffi
[80]: foreign-function-interface.html#javascript-ffi
[81]: function-definitions.html
[82]: function-definitions.html#introduction
[83]: function-definitions.html#general-form
[84]: function-definitions.html#special-patterns
[85]: function-definitions.html#case-trees
[86]: function-types.html
[87]: function-types.html#notational-conventions
[88]: generalization-of-declared-variables.html
[89]: generalization-of-declared-variables.html#overview
[90]: generalization-of-declared-variables.html#nested-generalization
[91]: generalization-of-declared-variables.html#placement-of-generalized-bindings
[92]: generalization-of-declared-variables.html#instance-and-irrelevant-variables
[93]: generalization-of-declared-variables.html#importing-and-exporting-variables
[94]: generalization-of-declared-variables.html#interaction
[95]: generalization-of-declared-variables.html#modalities
[96]: guarded.html
[97]: guarded.html#references
[98]: implicit-arguments.html
[99]: implicit-arguments.html#tactic-arguments
[100]: implicit-arguments.html#metavariables
[101]: implicit-arguments.html#unification
[102]: instance-arguments.html
[103]: instance-arguments.html#usage
[104]: instance-arguments.html#overlap-and-backtracking
[105]: instance-arguments.html#instance-resolution
[106]: irrelevance.html
[107]: irrelevance.html#motivating-example
[108]: irrelevance.html#irrelevant-function-types
[109]: irrelevance.html#irrelevant-declarations
[110]: irrelevance.html#irrelevant-record-fields
[111]: irrelevance.html#dependent-irrelevant-function-types
[112]: irrelevance.html#irrelevant-instance-arguments
[113]: lambda-abstraction.html
[114]: lambda-abstraction.html#lambda-expressions
[115]: lambda-abstraction.html#pattern-lambda
[116]: lambda-abstraction.html#absurd-lambda
[117]: let-and-where.html
[118]: let-and-where.html#let-expressions
[119]: let-and-where.html#where-blocks
[120]: let-and-where.html#proving-properties
[121]: let-and-where.html#more-examples-for-beginners
[122]: lexical-structure.html
[123]: lexical-structure.html#tokens
[124]: lexical-structure.html#layout
[125]: lexical-structure.html#literate-agda
[126]: literal-overloading.html
[127]: literal-overloading.html#natural-numbers
[128]: literal-overloading.html#negative-numbers
[129]: literal-overloading.html#strings
[130]: literal-overloading.html#restrictions
[131]: lossy-unification.html
[132]: lossy-unification.html#heuristic
[133]: lossy-unification.html#example
[134]: lossy-unification.html#drawbacks
[135]: mixfix-operators.html
[136]: mixfix-operators.html#precedence
[137]: mixfix-operators.html#associativity
[138]: mixfix-operators.html#ambiguity-and-scope
[139]: mixfix-operators.html#operators-in-telescopes
[140]: modalities.html
[141]: modalities.html#general-modalities
[142]: modalities.html#positional-modality-systems
[143]: modalities.html#pure-modality-systems
[144]: module-system.html
[145]: module-system.html#basics
[146]: module-system.html#private-definitions
[147]: module-system.html#name-modifiers
[148]: module-system.html#re-exporting-names
[149]: module-system.html#parameterised-modules
[150]: module-system.html#splitting-a-program-over-multiple-files
[151]: module-system.html#datatype-modules-and-record-modules
[152]: module-system.html#references
[153]: mutual-recursion.html
[154]: mutual-recursion.html#interleaved-mutual-blocks
[155]: mutual-recursion.html#forward-declaration
[156]: mutual-recursion.html#old-style-mutual-blocks
[157]: opaque-definitions.html
[158]: opaque-definitions.html#overview
[159]: opaque-definitions.html#unfolding-opaque-definitions
[160]: opaque-definitions.html#what-actually-unfolds
[161]: opaque-definitions.html#unfolding-in-types
[162]: opaque-definitions.html#bibliography
[163]: pattern-synonyms.html
[164]: pattern-synonyms.html#overloading
[165]: pattern-synonyms.html#refolding
[166]: polarity.html
[167]: polarity.html#the-polarity-modality
[168]: polarity.html#positivity-checking
[169]: polarity.html#references
[170]: positivity-checking.html
[171]: positivity-checking.html#the-no-positivity-check-pragma
[172]: positivity-checking.html#polarity-pragmas
[173]: postulates.html
[174]: postulates.html#postulated-built-ins
[175]: postulates.html#local-uses-of-postulate
[176]: pragmas.html
[177]: pragmas.html#index-of-pragmas
[178]: prop.html
[179]: prop.html#usage
[180]: prop.html#the-predicative-hierarchy-of-prop
[181]: prop.html#the-propositional-squash-type
[182]: prop.html#limitations
[183]: record-types.html
[184]: record-types.html#example-the-pair-type-constructor
[185]: record-types.html#declaring-constructing-and-decomposing-records
[186]: record-types.html#record-modules
[187]: record-types.html#eta-expansion
[188]: record-types.html#recursive-records
[189]: record-types.html#instance-fields
[190]: reflection.html
[191]: reflection.html#builtin-types
[192]: reflection.html#metaprogramming
[193]: rewriting.html
[194]: rewriting.html#rewrite-rules-by-example
[195]: rewriting.html#general-shape-of-rewrite-rules
[196]: rewriting.html#confluence-checking
[197]: rewriting.html#advanced-usage
[198]: rewriting.html#importing-rewrite-rules
[199]: runtime-irrelevance.html
[200]: runtime-irrelevance.html#syntax
[201]: runtime-irrelevance.html#rules
[202]: runtime-irrelevance.html#references
[203]: safe-agda.html
[204]: sized-types.html
[205]: sized-types.html#example-for-coinduction-finite-languages
[206]: sized-types.html#references
[207]: sort-system.html
[208]: sort-system.html#introduction-to-universes
[209]: sort-system.html#agda-s-sort-system
[210]: sort-system.html#sort-metavariables-and-unknown-sorts
[211]: syntactic-sugar.html
[212]: syntactic-sugar.html#hidden-argument-puns
[213]: syntactic-sugar.html#do-notation
[214]: syntactic-sugar.html#idiom-brackets
[215]: syntax-declarations.html
[216]: telescopes.html
[217]: telescopes.html#irrefutable-patterns-in-binding-positions
[218]: telescopes.html#let-bindings-in-telescopes
[219]: termination-checking.html
[220]: termination-checking.html#primitive-recursion
[221]: termination-checking.html#structural-recursion
[222]: termination-checking.html#with-functions
[223]: termination-checking.html#pragmas-and-options
[224]: termination-checking.html#references
[225]: two-level.html
[226]: two-level.html#basics
[227]: universe-levels.html
[228]: universe-levels.html#level-arithmetic
[229]: universe-levels.html#intrinsic-level-properties
[230]: universe-levels.html#forall-notation
[231]: universe-levels.html#expressions-of-sort-set
[232]: universe-levels.html#pragmas-and-options
[233]: with-abstraction.html
[234]: with-abstraction.html#usage
[235]: with-abstraction.html#technical-details
[236]: without-k.html
[237]: without-k.html#restrictions-on-pattern-matching
[238]: without-k.html#restrictions-on-termination-checking
[239]: without-k.html#restrictions-on-universe-levels
[240]: ../getting-started/tutorial-list.html
[241]: abstract-definitions.html
[242]: https://www.sphinx-doc.org/
[243]: https://github.com/readthedocs/sphinx_rtd_theme
[244]: https://readthedocs.org
