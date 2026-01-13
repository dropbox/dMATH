* Stainless documentation
* [ View page source][1]

# Stainless documentation[¶][2]

Contents:

* [Introduction][3]
  
  * [Stainless and Scala][4]
  * [Software Verification][5]
  * [Program Termination][6]
* [Installing Stainless][7]
  
  * [General Requirement][8]
  * [Obtain From A Package Manager][9]
  * [Github Codespaces][10]
  * [Use Standalone Release][11]
  * [Usage Within An Existing Project][12]
  * [Running Code with Stainless dependencies][13]
  * [External Solver Binaries][14]
  * [Build from Source on Linux & macOS][15]
  * [Build from Source on Windows][16]
  * [Running Tests][17]
  * [Building Stainless Documentation][18]
  * [Using IDEs with –no-colors option. Emacs illustration][19]
* [Verifying and Compiling Examples][20]
  
  * [Verifying Examples][21]
  * [Compiling and Executing Examples][22]
* [Tutorial: Sorting][23]
  
  * [Warm-up: Max][24]
  * [Defining Lists and Their Properties][25]
  * [Insertion into Sorted List][26]
  * [Being Sorted is Not Enough][27]
  * [Using Size in Specification][28]
  * [Using Content in Specification][29]
* [Specifying Options][30]
  
  * [Choosing which Stainless feature to use][31]
  * [Additional top-level options][32]
  * [Additional Options (by component)][33]
  * [Configuration File][34]
  * [Library Files][35]
* [Verification conditions][36]
  
  * [Postconditions][37]
  * [Preconditions][38]
  * [Sharing bindings between specifications and function body][39]
  * [Loop invariants][40]
  * [Decrease annotation in loops][41]
  * [Array access safety][42]
  * [ADT invariants][43]
  * [Pattern matching exhaustiveness][44]
* [Specifying Algebraic Properties][45]
  
  * [Introduction][46]
  * [Typeclasses][47]
  * [Typeclass inheritance][48]
  * [Associated methods][49]
  * [Coherence][50]
  * [Under the hood][51]
* [Imperative and Other Effects][52]
  
  * [Imperative Code][53]
  * [While loops][54]
  * [Arrays][55]
  * [Mutable Objects][56]
  * [Aliasing][57]
  * [Annotations for Imperative Programming][58]
  * [Extern functions and abstract methods][59]
  * [Trait Variables][60]
  * [Return keyword][61]
  * [Exceptions][62]
* [Equivalence Checking][63]
  
  * [Example run][64]
* [Ghost Context][65]
  
  * [Introduction][66]
  * [Ghost annotation][67]
  * [Correctness check][68]
  * [Ghost expression elimination][69]
  * [Case study][70]
* [Working With Existing Code][71]
  
  * [A wrapper for TrieMap][72]
  * [Extern methods][73]
  * [Contracts][74]
  * [Purity annotations][75]
* [Pure Scala][76]
  
  * [Booleans][77]
  * [Algebraic Data Types][78]
  * [Generics][79]
  * [Methods][80]
  * [Type Definitions][81]
  * [Specifications][82]
  * [Expressions][83]
  * [Predefined Types][84]
* [Stainless Library][85]
  
  * [Annotations][86]
  * [List[T]][87]
  * [Set[T], Map[T]][88]
  * [PartialFunction[A, B]][89]
* [Generating C Code][90]
  
  * [Requirements][91]
  * [Export][92]
  * [Supported Features][93]
  * [Global State][94]
  * [Custom Conversion][95]
  * [API For Safe Low Level Programs][96]
* [Proving Theorems][97]
  
  * [A practical introduction to proofs][98]
  * [Techniques for proving non-trivial propositions][99]
  * [Techniques for proving non-trivial postconditions][100]
  * [A complex example: additivity of measures][101]
  * [Quick Recap][102]
* [Limitations of Verification][103]
  
  * [Out of Memory Errors][104]
* [Case Studies][105]
  
  * [Case Study #1: Proving invariants of actor systems][106]
* [Translation from Stainless to Coq][107]
  
  * [Requirements][108]
  * [Usage of the Coq option][109]
* [FAQ: (Frequently) Asked Questions][110]
  
  * [How does Stainless compare to other verification tools?][111]
  * [How does Stainless compare to fuzzing tools?][112]
  * [Does Stainless use SMT solvers?][113]
  * [What are the conditions required for Stainless to be applied to industry software?][114]
  * [Can I use Stainless with Java?][115]
  * [Can I use Stainless with Rust?][116]
  * [Proving properties of size][117]
  * [Compiling Stainless programs to bytecode][118]
* [References][119]
  
  * [Videos][120]
  * [Papers][121]
  * [Books][122]
* [Stainless’ Internals][123]

* [Search Page][124]
[Next ][125]

© Copyright 2009-2021 EPFL, Lausanne. Last updated on Sep 11, 2025.

[1]: _sources/index.rst.txt
[2]: #stainless-documentation
[3]: intro.html
[4]: intro.html#stainless-and-scala
[5]: intro.html#software-verification
[6]: intro.html#program-termination
[7]: installation.html
[8]: installation.html#general-requirement
[9]: installation.html#obtain-from-a-package-manager
[10]: installation.html#github-codespaces
[11]: installation.html#use-standalone-release
[12]: installation.html#usage-within-an-existing-project
[13]: installation.html#running-code-with-stainless-dependencies
[14]: installation.html#external-solver-binaries
[15]: installation.html#build-from-source-on-linux-macos
[16]: installation.html#build-from-source-on-windows
[17]: installation.html#running-tests
[18]: installation.html#building-stainless-documentation
[19]: installation.html#using-ides-with-no-colors-option-emacs-illustration
[20]: gettingstarted.html
[21]: gettingstarted.html#verifying-examples
[22]: gettingstarted.html#compiling-and-executing-examples
[23]: tutorial.html
[24]: tutorial.html#warm-up-max
[25]: tutorial.html#defining-lists-and-their-properties
[26]: tutorial.html#insertion-into-sorted-list
[27]: tutorial.html#being-sorted-is-not-enough
[28]: tutorial.html#using-size-in-specification
[29]: tutorial.html#using-content-in-specification
[30]: options.html
[31]: options.html#choosing-which-stainless-feature-to-use
[32]: options.html#additional-top-level-options
[33]: options.html#additional-options-by-component
[34]: options.html#configuration-file
[35]: options.html#library-files
[36]: verification.html
[37]: verification.html#postconditions
[38]: verification.html#preconditions
[39]: verification.html#sharing-bindings-between-specifications-and-function-body
[40]: verification.html#loop-invariants
[41]: verification.html#decrease-annotation-in-loops
[42]: verification.html#array-access-safety
[43]: verification.html#adt-invariants
[44]: verification.html#pattern-matching-exhaustiveness
[45]: laws.html
[46]: laws.html#introduction
[47]: laws.html#typeclasses
[48]: laws.html#typeclass-inheritance
[49]: laws.html#associated-methods
[50]: laws.html#coherence
[51]: laws.html#under-the-hood
[52]: imperative.html
[53]: imperative.html#imperative-code
[54]: imperative.html#while-loops
[55]: imperative.html#arrays
[56]: imperative.html#mutable-objects
[57]: imperative.html#aliasing
[58]: imperative.html#annotations-for-imperative-programming
[59]: imperative.html#extern-functions-and-abstract-methods
[60]: imperative.html#trait-variables
[61]: imperative.html#return-keyword
[62]: imperative.html#exceptions
[63]: equivalence.html
[64]: equivalence.html#example-run
[65]: ghost.html
[66]: ghost.html#introduction
[67]: ghost.html#ghost-annotation
[68]: ghost.html#correctness-check
[69]: ghost.html#ghost-expression-elimination
[70]: ghost.html#case-study
[71]: wrap.html
[72]: wrap.html#a-wrapper-for-triemap
[73]: wrap.html#extern-methods
[74]: wrap.html#contracts
[75]: wrap.html#purity-annotations
[76]: purescala.html
[77]: purescala.html#booleans
[78]: purescala.html#algebraic-data-types
[79]: purescala.html#generics
[80]: purescala.html#methods
[81]: purescala.html#type-definitions
[82]: purescala.html#specifications
[83]: purescala.html#expressions
[84]: purescala.html#predefined-types
[85]: library.html
[86]: library.html#annotations
[87]: library.html#list-t
[88]: library.html#set-t-map-t
[89]: library.html#partialfunction-a-b
[90]: genc.html
[91]: genc.html#requirements
[92]: genc.html#export
[93]: genc.html#supported-features
[94]: genc.html#global-state
[95]: genc.html#custom-conversion
[96]: genc.html#api-for-safe-low-level-programs
[97]: neon.html
[98]: neon.html#a-practical-introduction-to-proofs
[99]: neon.html#techniques-for-proving-non-trivial-propositions
[100]: neon.html#techniques-for-proving-non-trivial-postconditions
[101]: neon.html#a-complex-example-additivity-of-measures
[102]: neon.html#quick-recap
[103]: limitations.html
[104]: limitations.html#out-of-memory-errors
[105]: casestudies.html
[106]: casestudies.html#case-study-1-proving-invariants-of-actor-systems
[107]: coq.html
[108]: coq.html#requirements
[109]: coq.html#usage-of-the-coq-option
[110]: faq.html
[111]: faq.html#how-does-stainless-compare-to-other-verification-tools
[112]: faq.html#how-does-stainless-compare-to-fuzzing-tools
[113]: faq.html#does-stainless-use-smt-solvers
[114]: faq.html#what-are-the-conditions-required-for-stainless-to-be-applied-to-industry-software
[115]: faq.html#can-i-use-stainless-with-java
[116]: faq.html#can-i-use-stainless-with-rust
[117]: faq.html#proving-properties-of-size
[118]: faq.html#compiling-stainless-programs-to-bytecode
[119]: references.html
[120]: references.html#videos
[121]: references.html#papers
[122]: references.html#books
[123]: internals.html
[124]: search.html
[125]: intro.html
