[

# UPPAAL

][1]
** **

* [** Home][2]

* [1. GUI Reference][3]
  
  * [Menu Bar][4]
    
    * [File Menu][5]
    * [Edit Menu][6]
    * [View Menu][7]
    * [Tools Menu][8]
    * [Options Menu][9]
    * [Help Menu][10]
  * [Tool Bar][11]
  * [System Editor][12]
    
    * [Navigation Tree][13]
    * [Drawing][14]
    * [Declarations][15]
  * [Symbolic Simulator][16]
    
    * [Simulation Control][17]
    * [Variables Panel][18]
    * [Process Panel][19]
    * [Message Sequence Chart][20]
    * [Symbolic Traces][21]
  * [Concrete Simulator][22]
    
    * [Simulation Control][23]
    * [Variables Panel][24]
    * [Process Panel][25]
    * [Message Sequence Chart][26]
    * [Gantt Chart Panel][27]
  * [Verifier][28]
    
    * [Specifying Requirements][29]
    * [Verifying Requirements][30]
  * [Test Cases (Yggdrasil)][31]
    
    * [Generating Traces][32]
    * [Specifying Test Code][33]
    * [Tutorial][34]
      
      * [Basic Test Generation][35]
      * [Using Model Variables][36]
      * [Using Queries][37]
* [2. Language Reference][38]
  
  * [System Description][39]
    
    * [Declarations][40]
      
      * [Types][41]
      * [Functions][42]
      * [External Functions][43]
    * [Templates][44]
      
      * [Locations][45]
      * [Edges][46]
    * [Parameters][47]
    * [System Definition][48]
      
      * [Template Instantiation][49]
      * [Progress Measures][50]
      * [Gantt Chart][51]
    * [Priorities][52]
    * [Scope Rules][53]
    * [Semantics][54]
  * [Query Syntax][55]
    
    * [Symbolic Queries][56]
    * [Controller Synthesis][57]
    * [Statistical Queries][58]
    * [Learning Queries][59]
    * [Strategy Queries][60]
  * [Query Semantics][61]
    
    * [Symbolic Queries][62]
    * [Statistical Queries][63]
      
      * [Confidence Intervals][64]
  * [Expressions][65]
    
    * [Identifiers][66]
  * [Reserved Keywords][67]
* [3. Tools & API][68]
  
  * [UPPAAL][69]
  * [verifyta][70]
  * [Java API][71]
  * [Socketserver][72]
  * [Docker][73]
  * [File Formats][74]
  * [Latex][75]
* [4. Extensions][76]
  
  * [CORA][77]
  * [ECDAR][78]
  * [UPPAAL Tiga][79]
  * [UPPAAL TRON][80]

(c) 1995-2023 [UPPAAL][81]

[** navigation][82]

# UPPAAL Help

[UPPAAL][83] is a tool for modeling, validation and verification of real-time systems. It is
appropriate for systems that can be modeled as a collection of non-deterministic processes with
finite control structure and real-valued clocks (i.e. timed automata), communicating through
channels and (or) shared data structures. Typical application areas include real-time controllers,
communication protocols, and other systems in which timing aspects are critical.

The UPPAAL tool consists of three main parts:

* a graphical user interface (GUI),
* a verification server, and
* a command line tool.

The GUI is used for modelling, symbolic simulation, concrete simulation, and verification. For both
simulation and verification, the GUI uses the verification server. In simulation, the server is used
to compute successor states. See also the section on setting up a remote server. The command line
tool is a stand-alone verifier, appropriate for e.g. batch verifications.

More information can be found at the UPPAAL web site: [http://www.uppaal.org][84].

[1]: /
[2]: /
[3]: /gui-reference/
[4]: /gui-reference/menu-bar/
[5]: /gui-reference/menu-bar/file/
[6]: /gui-reference/menu-bar/edit/
[7]: /gui-reference/menu-bar/view/
[8]: /gui-reference/menu-bar/tools/
[9]: /gui-reference/menu-bar/options/
[10]: /gui-reference/menu-bar/help/
[11]: /gui-reference/toolbar/
[12]: /gui-reference/system-editor/
[13]: /gui-reference/system-editor/navigation-tree/
[14]: /gui-reference/system-editor/drawing/
[15]: /gui-reference/system-editor/declarations/
[16]: /gui-reference/symbolic-simulator/
[17]: /gui-reference/symbolic-simulator/simulation-control/
[18]: /gui-reference/symbolic-simulator/variables/
[19]: /gui-reference/symbolic-simulator/process/
[20]: /gui-reference/symbolic-simulator/sequence-charts/
[21]: /gui-reference/symbolic-simulator/symbolic-traces/
[22]: /gui-reference/concrete-simulator/
[23]: /gui-reference/concrete-simulator/simulation-control/
[24]: /gui-reference/concrete-simulator/variables/
[25]: /gui-reference/concrete-simulator/process/
[26]: /gui-reference/concrete-simulator/sequence-chart/
[27]: /gui-reference/concrete-simulator/gantt-chart/
[28]: /gui-reference/verifier/
[29]: /gui-reference/verifier/specifying/
[30]: /gui-reference/verifier/verifying/
[31]: /gui-reference/yggdrasil/
[32]: /gui-reference/yggdrasil/traces/
[33]: /gui-reference/yggdrasil/test-code/
[34]: /gui-reference/yggdrasil/tutorial/
[35]: /gui-reference/yggdrasil/tutorial/basic-test-generation/
[36]: /gui-reference/yggdrasil/tutorial/using-variables/
[37]: /gui-reference/yggdrasil/tutorial/using-queries/
[38]: /language-reference/
[39]: /language-reference/system-description/
[40]: /language-reference/system-description/declarations/
[41]: /language-reference/system-description/declarations/types/
[42]: /language-reference/system-description/declarations/functions/
[43]: /language-reference/system-description/declarations/external-functions/
[44]: /language-reference/system-description/templates/
[45]: /language-reference/system-description/templates/locations/
[46]: /language-reference/system-description/templates/edges/
[47]: /language-reference/system-description/parameters/
[48]: /language-reference/system-description/system-definition/
[49]: /language-reference/system-description/system-definition/template-instantiation/
[50]: /language-reference/system-description/system-definition/progress-measures/
[51]: /language-reference/system-description/system-definition/gantt-chart/
[52]: /language-reference/system-description/priorities/
[53]: /language-reference/system-description/scope-rules/
[54]: /language-reference/system-description/semantics/
[55]: /language-reference/query-syntax/
[56]: /language-reference/query-syntax/symbolic_queries/
[57]: /language-reference/query-syntax/controller_synthesis/
[58]: /language-reference/query-syntax/statistical_queries/
[59]: /language-reference/query-syntax/learning_queries/
[60]: /language-reference/query-syntax/strategy_queries/
[61]: /language-reference/query-semantics/
[62]: /language-reference/query-semantics/symb_queries/
[63]: /language-reference/query-semantics/smc_queries/
[64]: /language-reference/query-semantics/smc_queries/ci_estimation/
[65]: /language-reference/expressions/
[66]: /language-reference/expressions/identifiers/
[67]: /language-reference/reserved-keywords/
[68]: /toolsandapi/
[69]: /toolsandapi/uppaal/
[70]: /toolsandapi/verifyta/
[71]: /toolsandapi/javaapi/
[72]: /toolsandapi/socketserver/
[73]: /toolsandapi/docker/
[74]: /toolsandapi/file-formats/
[75]: /toolsandapi/latex/
[76]: /extensions/
[77]: /extensions/cora/
[78]: /extensions/ecdar/
[79]: /extensions/tiga/
[80]: /extensions/tron/
[81]: https://www.uppaal.org
[82]: #
[83]: http://www.uppaal.org
[84]: http://www.uppaal.org
