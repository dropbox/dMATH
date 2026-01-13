# Documentation Alloy 6

Alloy 6 is a self-contained executable, which includes the Pardinus/Kodkod model
finder and a variety of SAT solvers, as well as the standard Alloy library and a
collection of tutorial examples. The same jar file can be incorporated into
other applications to use Alloy as an API, and includes the source code.

You can download alloy from
https://github.com/AlloyTools/org.alloytools.alloy/releases/tag/v6.0.0

To execute, simply double-click on the jar file, or type `java -jar
org.alloytools.alloy.dist.jar` in a console. For MacOS users, there is a DMG
file.

Since Alloy 6, the tool can also perform temporal model-checking. at the time of
this writing, this relies on external tools [NuSMV][1] or [nuXmv][2] (preferred
from an efficiency point of view) that should be installed by the user and mad
available in the PATH.

──────────────────────────┬─────────────────────────────────────────────────────
[Formal Software Design   │outdated online book specifically dedicated to       
with Alloy 6][3]          │learning Alloy 6                                     
──────────────────────────┼─────────────────────────────────────────────────────
[FAQ][4]                  │Frequently asked questions about Alloy               
──────────────────────────┼─────────────────────────────────────────────────────
[reference][5]            │Language reference for Alloy 6                       
──────────────────────────┼─────────────────────────────────────────────────────
[grammar][6]              │Alloy Grammar in Java CUP                            
──────────────────────────┼─────────────────────────────────────────────────────
[lex][7]                  │Alloy Lexical tokens in flex                         
──────────────────────────┼─────────────────────────────────────────────────────
[quick guide][8]          │Overview of new features in Alloy 4                  
──────────────────────────┼─────────────────────────────────────────────────────
[comparisons][9]          │Comparisons to Z, B, VDM and OCL (PDF)               
──────────────────────────┼─────────────────────────────────────────────────────
[Alloy API][10]           │Documentation for Alloy API                          
──────────────────────────┼─────────────────────────────────────────────────────
[Alloy API Examples][11]  │examples of using [the compiler][12], [the ast][13], 
                          │[the evaluator][14]                                  
──────────────────────────┼─────────────────────────────────────────────────────
[online tutorial][15]     │A step-by-step walkthrough and tutorial for Alloy 4  
──────────────────────────┼─────────────────────────────────────────────────────
[tutorial slides][16]     │Slides for day-long tutorial by Rob Seater and Greg  
                          │Dennis                                               
──────────────────────────┼─────────────────────────────────────────────────────
[digital humanities       │A tutorial introducing Alloy for digital humanities  
tutorial][17]             │work, by C. M. Sperberg-McQueen                      
──────────────────────────┴─────────────────────────────────────────────────────

[1]: https://nusmv.fbk.eu/
[2]: https://nuxmv.fbk.eu/
[3]: https://haslab.github.io/formal-software-design/
[4]: /faq/faq.html
[5]: spec.html
[6]: https://github.com/AlloyTools/org.alloytools.alloy/blob/master/org.alloytoo
ls.alloy.core/parser/Alloy.cup
[7]: https://github.com/AlloyTools/org.alloytools.alloy/blob/master/org.alloytoo
ls.alloy.core/parser/Alloy.lex
[8]: quickguide
[9]: /download/comparisons.pdf
[10]: /documentation/alloy-api/index.html
[11]: documentation/alloy-api-examples.html
[12]: code/ExampleUsingTheCompiler.java.html
[13]: code/ExampleUsingTheAPI.java.html
[14]: code/EvaluatorExample.java.html
[15]: tutorials/online
[16]: tutorials/day-course
[17]: http://blackmesatech.com/2013/07/alloy/
