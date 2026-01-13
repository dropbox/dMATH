# Introduction To SPARK[¶][1]

This tutorial is an interactive introduction to the SPARK programming language and its formal
verification tools. You will learn the difference between Ada and SPARK and how to use the various
analysis tools that come with SPARK.

This document was prepared by Claire Dross and Yannick Moy.

Note

The code examples in this course use an 80-column limit, which is a typical limit for Ada code. Note
that, on devices with a small screen size, some code examples might be difficult to read.

[ Download PDF ][2] [ Download EPUB ][3]

Contents:

* [Overview][4]
  
  * [What is it?][5]
  * [What do the tools do?][6]
  * [Key Tools][7]
  * [A trivial example][8]
  * [The Programming Language][9]
  * [Limitations][10]
    
    * [No side-effects in expressions][11]
    * [No aliasing of names][12]
  * [Designating SPARK Code][13]
  * [Code Examples / Pitfalls][14]
    
    * [Example #1][15]
    * [Example #2][16]
    * [Example #3][17]
    * [Example #4][18]
    * [Example #5][19]
    * [Example #6][20]
    * [Example #7][21]
    * [Example #8][22]
    * [Example #9][23]
    * [Example #10][24]
* [Flow Analysis][25]
  
  * [What does flow analysis do?][26]
  * [Errors Detected][27]
    
    * [Uninitialized Variables][28]
    * [Ineffective Statements][29]
    * [Incorrect Parameter Mode][30]
  * [Additional Verifications][31]
    
    * [Global Contracts][32]
    * [Depends Contracts][33]
  * [Shortcomings][34]
    
    * [Modularity][35]
    * [Composite Types][36]
    * [Value Dependency][37]
    * [Contract Computation][38]
  * [Code Examples / Pitfalls][39]
    
    * [Example #1][40]
    * [Example #2][41]
    * [Example #3][42]
    * [Example #4][43]
    * [Example #5][44]
    * [Example #6][45]
    * [Example #7][46]
    * [Example #8][47]
    * [Example #9][48]
    * [Example #10][49]
* [Proof of Program Integrity][50]
  
  * [Runtime Errors][51]
  * [Modularity][52]
    
    * [Exceptions][53]
  * [Contracts][54]
    
    * [Executable Semantics][55]
    * [Additional Assertions and Contracts][56]
  * [Debugging Failed Proof Attempts][57]
    
    * [Debugging Errors in Code or Specification][58]
    * [Debugging Cases where more Information is Required][59]
    * [Debugging Prover Limitations][60]
  * [Code Examples / Pitfalls][61]
    
    * [Example #1][62]
    * [Example #2][63]
    * [Example #3][64]
    * [Example #4][65]
    * [Example #5][66]
    * [Example #6][67]
    * [Example #7][68]
    * [Example #8][69]
    * [Example #9][70]
    * [Example #10][71]
* [State Abstraction][72]
  
  * [What's an Abstraction?][73]
  * [Why is Abstraction Useful?][74]
  * [Abstraction of a Package's State][75]
  * [Declaring a State Abstraction][76]
  * [Refining an Abstract State][77]
  * [Representing Private Variables][78]
  * [Additional State][79]
    
    * [Nested Packages][80]
    * [Constants that Depend on Variables][81]
  * [Subprogram Contracts][82]
    
    * [Global and Depends][83]
    * [Preconditions and Postconditions][84]
  * [Initialization of Local Variables][85]
  * [Code Examples / Pitfalls][86]
    
    * [Example #1][87]
    * [Example #2][88]
    * [Example #3][89]
    * [Example #4][90]
    * [Example #5][91]
    * [Example #6][92]
    * [Example #7][93]
    * [Example #8][94]
    * [Example #9][95]
    * [Example #10][96]
* [Proof of Functional Correctness][97]
  
  * [Beyond Program Integrity][98]
  * [Advanced Contracts][99]
    
    * [Ghost Code][100]
    * [Ghost Functions][101]
    * [Global Ghost Variables][102]
  * [Guide Proof][103]
    
    * [Local Ghost Variables][104]
    * [Ghost Procedures][105]
    * [Handling of Loops][106]
    * [Loop Invariants][107]
  * [Code Examples / Pitfalls][108]
    
    * [Example #1][109]
    * [Example #2][110]
    * [Example #3][111]
    * [Example #4][112]
    * [Example #5][113]
    * [Example #6][114]
    * [Example #7][115]
    * [Example #8][116]
    * [Example #9][117]
    * [Example #10][118]

[1]: #introduction-to-spark
[2]: /pdf_books/courses/intro-to-spark.pdf
[3]: /epub_books/courses/intro-to-spark.epub
[4]: chapters/01_Overview.html
[5]: chapters/01_Overview.html#what-is-it
[6]: chapters/01_Overview.html#what-do-the-tools-do
[7]: chapters/01_Overview.html#key-tools
[8]: chapters/01_Overview.html#a-trivial-example
[9]: chapters/01_Overview.html#the-programming-language
[10]: chapters/01_Overview.html#limitations
[11]: chapters/01_Overview.html#no-side-effects-in-expressions
[12]: chapters/01_Overview.html#no-aliasing-of-names
[13]: chapters/01_Overview.html#designating-spark-code
[14]: chapters/01_Overview.html#code-examples-pitfalls
[15]: chapters/01_Overview.html#example-1
[16]: chapters/01_Overview.html#example-2
[17]: chapters/01_Overview.html#example-3
[18]: chapters/01_Overview.html#example-4
[19]: chapters/01_Overview.html#example-5
[20]: chapters/01_Overview.html#example-6
[21]: chapters/01_Overview.html#example-7
[22]: chapters/01_Overview.html#example-8
[23]: chapters/01_Overview.html#example-9
[24]: chapters/01_Overview.html#example-10
[25]: chapters/02_Flow_Analysis.html
[26]: chapters/02_Flow_Analysis.html#what-does-flow-analysis-do
[27]: chapters/02_Flow_Analysis.html#errors-detected
[28]: chapters/02_Flow_Analysis.html#uninitialized-variables
[29]: chapters/02_Flow_Analysis.html#ineffective-statements
[30]: chapters/02_Flow_Analysis.html#incorrect-parameter-mode
[31]: chapters/02_Flow_Analysis.html#additional-verifications
[32]: chapters/02_Flow_Analysis.html#global-contracts
[33]: chapters/02_Flow_Analysis.html#depends-contracts
[34]: chapters/02_Flow_Analysis.html#shortcomings
[35]: chapters/02_Flow_Analysis.html#modularity
[36]: chapters/02_Flow_Analysis.html#composite-types
[37]: chapters/02_Flow_Analysis.html#value-dependency
[38]: chapters/02_Flow_Analysis.html#contract-computation
[39]: chapters/02_Flow_Analysis.html#code-examples-pitfalls
[40]: chapters/02_Flow_Analysis.html#example-1
[41]: chapters/02_Flow_Analysis.html#example-2
[42]: chapters/02_Flow_Analysis.html#example-3
[43]: chapters/02_Flow_Analysis.html#example-4
[44]: chapters/02_Flow_Analysis.html#example-5
[45]: chapters/02_Flow_Analysis.html#example-6
[46]: chapters/02_Flow_Analysis.html#example-7
[47]: chapters/02_Flow_Analysis.html#example-8
[48]: chapters/02_Flow_Analysis.html#example-9
[49]: chapters/02_Flow_Analysis.html#example-10
[50]: chapters/03_Proof_Of_Program_Integrity.html
[51]: chapters/03_Proof_Of_Program_Integrity.html#runtime-errors
[52]: chapters/03_Proof_Of_Program_Integrity.html#modularity
[53]: chapters/03_Proof_Of_Program_Integrity.html#exceptions
[54]: chapters/03_Proof_Of_Program_Integrity.html#contracts
[55]: chapters/03_Proof_Of_Program_Integrity.html#executable-semantics
[56]: chapters/03_Proof_Of_Program_Integrity.html#additional-assertions-and-contracts
[57]: chapters/03_Proof_Of_Program_Integrity.html#debugging-failed-proof-attempts
[58]: chapters/03_Proof_Of_Program_Integrity.html#debugging-errors-in-code-or-specification
[59]: chapters/03_Proof_Of_Program_Integrity.html#debugging-cases-where-more-information-is-required
[60]: chapters/03_Proof_Of_Program_Integrity.html#debugging-prover-limitations
[61]: chapters/03_Proof_Of_Program_Integrity.html#code-examples-pitfalls
[62]: chapters/03_Proof_Of_Program_Integrity.html#example-1
[63]: chapters/03_Proof_Of_Program_Integrity.html#example-2
[64]: chapters/03_Proof_Of_Program_Integrity.html#example-3
[65]: chapters/03_Proof_Of_Program_Integrity.html#example-4
[66]: chapters/03_Proof_Of_Program_Integrity.html#example-5
[67]: chapters/03_Proof_Of_Program_Integrity.html#example-6
[68]: chapters/03_Proof_Of_Program_Integrity.html#example-7
[69]: chapters/03_Proof_Of_Program_Integrity.html#example-8
[70]: chapters/03_Proof_Of_Program_Integrity.html#example-9
[71]: chapters/03_Proof_Of_Program_Integrity.html#example-10
[72]: chapters/04_State_Abstraction.html
[73]: chapters/04_State_Abstraction.html#what-s-an-abstraction
[74]: chapters/04_State_Abstraction.html#why-is-abstraction-useful
[75]: chapters/04_State_Abstraction.html#abstraction-of-a-package-s-state
[76]: chapters/04_State_Abstraction.html#declaring-a-state-abstraction
[77]: chapters/04_State_Abstraction.html#refining-an-abstract-state
[78]: chapters/04_State_Abstraction.html#representing-private-variables
[79]: chapters/04_State_Abstraction.html#additional-state
[80]: chapters/04_State_Abstraction.html#nested-packages
[81]: chapters/04_State_Abstraction.html#constants-that-depend-on-variables
[82]: chapters/04_State_Abstraction.html#subprogram-contracts
[83]: chapters/04_State_Abstraction.html#global-and-depends
[84]: chapters/04_State_Abstraction.html#preconditions-and-postconditions
[85]: chapters/04_State_Abstraction.html#initialization-of-local-variables
[86]: chapters/04_State_Abstraction.html#code-examples-pitfalls
[87]: chapters/04_State_Abstraction.html#example-1
[88]: chapters/04_State_Abstraction.html#example-2
[89]: chapters/04_State_Abstraction.html#example-3
[90]: chapters/04_State_Abstraction.html#example-4
[91]: chapters/04_State_Abstraction.html#example-5
[92]: chapters/04_State_Abstraction.html#example-6
[93]: chapters/04_State_Abstraction.html#example-7
[94]: chapters/04_State_Abstraction.html#example-8
[95]: chapters/04_State_Abstraction.html#example-9
[96]: chapters/04_State_Abstraction.html#example-10
[97]: chapters/05_Proof_Of_Functional_Correctness.html
[98]: chapters/05_Proof_Of_Functional_Correctness.html#beyond-program-integrity
[99]: chapters/05_Proof_Of_Functional_Correctness.html#advanced-contracts
[100]: chapters/05_Proof_Of_Functional_Correctness.html#ghost-code
[101]: chapters/05_Proof_Of_Functional_Correctness.html#ghost-functions
[102]: chapters/05_Proof_Of_Functional_Correctness.html#global-ghost-variables
[103]: chapters/05_Proof_Of_Functional_Correctness.html#guide-proof
[104]: chapters/05_Proof_Of_Functional_Correctness.html#local-ghost-variables
[105]: chapters/05_Proof_Of_Functional_Correctness.html#ghost-procedures
[106]: chapters/05_Proof_Of_Functional_Correctness.html#handling-of-loops
[107]: chapters/05_Proof_Of_Functional_Correctness.html#loop-invariants
[108]: chapters/05_Proof_Of_Functional_Correctness.html#code-examples-pitfalls
[109]: chapters/05_Proof_Of_Functional_Correctness.html#example-1
[110]: chapters/05_Proof_Of_Functional_Correctness.html#example-2
[111]: chapters/05_Proof_Of_Functional_Correctness.html#example-3
[112]: chapters/05_Proof_Of_Functional_Correctness.html#example-4
[113]: chapters/05_Proof_Of_Functional_Correctness.html#example-5
[114]: chapters/05_Proof_Of_Functional_Correctness.html#example-6
[115]: chapters/05_Proof_Of_Functional_Correctness.html#example-7
[116]: chapters/05_Proof_Of_Functional_Correctness.html#example-8
[117]: chapters/05_Proof_Of_Functional_Correctness.html#example-9
[118]: chapters/05_Proof_Of_Functional_Correctness.html#example-10
