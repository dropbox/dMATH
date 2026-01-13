───────────────────────────────────
libTriton version 1.0 build 1599   
───────────────────────────────────
Loading...
Searching...
No Matches
Triton: A dynamic binary analysis library

### Table of Contents

* [Description][1]
* [Presentations and Publications][2]

# Description

Triton is a dynamic binary analysis library. It provides internal components like a dynamic symbolic
execution engine, a dynamic taint analysis engine, AST representation of the x86, x86-64, ARM32 and
AArch64 ISA semantic, an expressions synthesis engine, some SMT simplification passes, SMT solver
interface to Z3 and Bitwuzla and, the last but not least, Python bindings. Based on these
components, you are able to build your program analysis tools, automate reverse engineering, perform
software verification or just emulate code.





# Presentations and Publications

* Greybox Program Synthesis: A New Approach to Attack Dataflow Obfuscation
  Talk at: Blackhat USA, Las Vegas, Nevada, 2021. [[slide][3]]
  Auhtors: Robin David
  Abstract: *This talk presents the latest advances in program synthesis applied for deobfuscation.
  It aims at demystifying this analysis technique by showing how it can be put into action on
  obfuscation. Especially the implementation Qsynthesis released for this talk shows a complete
  end-to-end workflow to deobfuscate assembly instructions back in optimized (deobfuscated)
  instructions reassembled back in the binary.*
* From source code to crash test-case through software testing automation
  Talk at: C&ESAR, Rennes, France, 2021. [[paper][4]] [[slide][5]]
  Auhtors: Robin David, Jonathan Salwan, Justin Bourroux
  Abstract: *This paper present an approach automating the software testing process from a source
  code to the dynamic testing of the compiled program. More specifically, from a static analysis
  report indicating alerts on source lines it enables testing to cover these lines dynamically and
  opportunistically checking whether whether or not they can trigger a crash. The result is a test
  corpus allowing to cover alerts and to trigger them if they happen to be true positives. This
  paper discuss the methodology employed to track alerts down in the compiled binary, the testing
  engines selection process and the results obtained on a TCP/IP stack implementation for embedded
  and IoT systems.*
* QSynth: A Program Synthesis based Approach for Binary Code Deobfuscation
  Talk at: BAR, San Diego, California, 2020. [[paper][6]]
  Auhtors: Robin David, Luigi Coniglio, Mariano Ceccato
  Abstract: *We present a generic approach leveraging both DSE and program synthesis to successfully
  synthesize programs obfuscated with Mixed-Boolean-Arithmetic, Data-Encoding or Virtualization. The
  synthesis algorithm proposed is an offline enumerate synthesis primitive guided by top-down
  breath-first search. We shows its effectiveness against a state-of-the-art obfuscator and its
  scalability as it supersedes other similar approaches based on synthesis. We also show its
  effectiveness in presence of composite obfuscation (combination of various techniques). This
  ongoing work enlightens the effectiveness of synthesis to target certain kinds of obfuscations and
  opens the way to more robust algorithms and simplification strategies.*
* Sydr: Cutting Edge Dynamic Symbolic Execution
  Talk at: Ivannikov ISP RAS Open Conference, Moscow, Russia, 2020. [[paper][7]]
  Auhtors: A.Vishnyakov, A.Fedotov, D.Kuts, A.Novikov, D.Parygina, E.Kobrin, V.Logunova, P.Belecky,
  S.Kurmangaleev
  Abstract: *Dynamic symbolic execution (DSE) has enormous amount of applications in computer
  security (fuzzing, vulnerability discovery, reverse-engineering, etc.). We propose several
  performance and accuracy improvements for dynamic symbolic execution. Skipping non-symbolic
  instructions allows to build a path predicate 1.2–3.5 times faster. Symbolic engine simplifies
  formulas during symbolic execution. Path predicate slicing eliminates irrelevant conjuncts from
  solver queries. We handle each jump table (switch statement) as multiple branches and describe the
  method for symbolic execution of multi-threaded programs. The proposed solutions were implemented
  in Sydr tool. Sydr performs inversion of branches in path predicate. Sydr combines DynamoRIO
  dynamic binary instrumentation tool with Triton symbolic engine.*
* Symbolic Deobfuscation: From Virtualized Code Back to the Original
  Talk at: DIMVA, Paris-Saclay, France, 2018. [[paper][8]] [[slide][9]]
  Auhtors: Jonathan Salwan, Sébastien Bardin, Marie-Laure Potet
  Abstract: *Software protection has taken an important place during the last decade in order to
  protect legit software against reverse engineering or tampering. Virtualization is considered as
  one of the very best defenses against such attacks. We present a generic approach based on
  symbolic path exploration, taint and recompilation allowing to recover, from a virtualized code, a
  devirtualized code semantically identical to the original one and close in size. We define
  criteria and metrics to evaluate the relevance of the deobfuscated results in terms of correctness
  and precision. Finally we propose an open-source setup allowing to evaluate the proposed approach
  against several forms of virtualization.*
* Deobfuscation of VM based software protection
  Talk at: SSTIC, Rennes, France, 2017. [[french paper][10]] [[english slide][11]] [[french
  video][12]]
  Auhtors: Jonathan Salwan, Sébastien Bardin, Marie-Laure Potet
  Abstract: *In this presentation we describe an approach which consists to automatically analyze
  virtual machine based software protections and which recompiles a new version of the binary
  without such protections. This automated approach relies on a symbolic execution guide by a taint
  analysis and some concretization policies, then on a binary rewriting using LLVM transition.*
* How Triton can help to reverse virtual machine based software protections
  Talk at: CSAW SOS, NYC, New York, 2016. [[slide][13]]
  Auhtors: Jonathan Salwan, Romain Thomas
  Abstract: *The first part of the talk is going to be an introduction to the Triton framework to
  expose its components and to explain how they work together. Then, the second part will include
  demonstrations on how it's possible to reverse virtual machine based protections using taint
  analysis, symbolic execution, SMT simplifications and LLVM-IR optimizations.*
* Dynamic Binary Analysis and Obfuscated Codes
  Talk at: St'Hack, Bordeaux, France, 2016. [[slide][14]]
  Auhtors: Jonathan Salwan, Romain Thomas
  Abstract: *At this presentation we will talk about how a DBA (Dynamic Binary Analysis) may help a
  reverse engineer to reverse obfuscated code. We will first introduce some basic obfuscation
  techniques and then expose how it's possible to break some stuffs (using our open-source DBA
  framework - Triton) like detect opaque predicates, reconstruct CFG, find the original algorithm,
  isolate sensible data and many more... Then, we will conclude with a demo and few words about our
  future work.*
* How Triton may help to analyse obfuscated binaries
  Publication at: MISC magazine 82, 2015. [[french article][15]]
  Auhtors: Jonathan Salwan, Romain Thomas
  Abstract: *Binary obfuscation is used to protect software's intellectual property. There exist
  different kinds of obfucation but roughly, it transforms a binary structure into another binary
  structure by preserving the same semantic. The aim of obfuscation is to ensure that the original
  information is "drown" in useless information that will make reverse engineering harder. In this
  article we will show how we can analyse an ofbuscated program and break some obfuscations using
  the Triton framework.*
* Triton: A Concolic Execution Framework
  Talk at: SSTIC, Rennes, France, 2015. [[french paper][16]] [[detailed english slide][17]]
  Auhtors: Jonathan Salwan, Florent Saudel
  Abstract: *This talk is about the release of Triton, a concolic execution framework based on Pin.
  It provides components like a taint engine, a dynamic symbolic execution engine, a snapshot
  engine, translation of x64 instruction to SMT2, a Z3 interface to solve constraints and Python
  bindings. Based on these components, Triton offers the possibility to build tools for
  vulnerabilities research or reverse-engineering assistance.*
* Dynamic Behavior Analysis Using Binary Instrumentation
  Talk at: St'Hack, Bordeaux, France, 2015. [[slide][18]]
  Auhtors: Jonathan Salwan
  Abstract: *This talk can be considered like the part 2 of our talk at SecurityDay. In the previous
  part, we talked about how it was possible to cover a targeted function in memory using the DSE
  (Dynamic Symbolic Execution) approach. Cover a function (or its states) doesn't mean find all
  vulnerabilities, some vulnerability doesn't crashes the program. That's why we must implement
  specific analysis to find specific bugs. These analysis are based on the binary instrumentation
  and the runtime behavior analysis of the program. In this talk, we will see how it's possible to
  find these following kind of bugs : off-by-one, stack / heap overflow, use-after-free, format
  string and {write, read}-what-where.*
* Covering a function using a Dynamic Symbolic Execution approach
  Talk at: Security Day, Lille, France, 2015. [[slide][19]]
  Auhtors: Jonathan Salwan
  Abstract: *This talk is about binary analysis and instrumentation. We will see how it's possible
  to target a specific function, snapshot the context memory/registers before the function,
  translate the instrumentation into an intermediate representation,apply a taint analysis based on
  this IR, build/keep formulas for a Dynamic Symbolic Execution (DSE), generate a concrete value to
  go through a specific path, restore the context memory/register and generate another concrete
  value to go through another path then repeat this operation until the target function is covered.*
[ Generated by ][20][[doxygen]][21] 1.10.0

[1]: #description_sec
[2]: #publications_sec
[3]: https://github.com/JonathanSalwan/Triton/tree/master/publications/BHUSA2021-David-Greybox-Progr
am-Synthesis.pdf
[4]: https://github.com/JonathanSalwan/Triton/tree/master/publications/CESAR2021_robin-david-paper.p
df
[5]: https://github.com/JonathanSalwan/Triton/tree/master/publications/CESAR2021_robin-david-slide.p
df
[6]: https://github.com/JonathanSalwan/Triton/tree/master/publications/BAR2020-qsynth-robin-david.pd
f
[7]: https://github.com/JonathanSalwan/Triton/tree/master/publications/ISPRAS2020-sydr.pdf
[8]: https://github.com/JonathanSalwan/Triton/tree/master/publications/DIMVA2018-deobfuscation-salwa
n-bardin-potet.pdf
[9]: https://github.com/JonathanSalwan/Triton/tree/master/publications/DIMVA2018-slide-deobfuscation
-salwan-bardin-potet.pdf
[10]: https://github.com/JonathanSalwan/Triton/tree/master/publications/SSTIC2017-French-Article-des
obfuscation_binaire_reconstruction_de_fonctions_virtualisees-salwan_potet_bardin.pdf
[11]: https://github.com/JonathanSalwan/Triton/tree/master/publications/SSTIC2017_Deobfuscation_of_V
M_based_software_protection.pdf
[12]: https://static.sstic.org/videos2017/SSTIC_2017-06-07_P08.mp4
[13]: https://github.com/JonathanSalwan/Triton/tree/master/publications/CSAW2016-SOS-Virtual-Machine
-Deobfuscation-RThomas_JSalwan.pdf
[14]: https://github.com/JonathanSalwan/Triton/tree/master/publications/StHack2016_Dynamic_Binary_An
alysis_and_Obfuscated_Codes_RThomas_JSalwan.pdf
[15]: https://github.com/JonathanSalwan/Triton/tree/master/publications/MISC-82_French_Paper_How_Tri
ton_may_help_to_analyse_obfuscated_binaries_RThomas_JSalwan.pdf
[16]: https://github.com/JonathanSalwan/Triton/tree/master/publications/SSTIC2015_French_Paper_Trito
n_Framework_dexecution_Concolique_FSaudel_JSalwan.pdf
[17]: https://github.com/JonathanSalwan/Triton/tree/master/publications/SSTIC2015_English_slide_deta
iled_version_Triton_Concolic_Execution_FrameWork_FSaudel_JSalwan.pdf
[18]: https://github.com/JonathanSalwan/Triton/tree/master/publications/StHack2015_Dynamic_Behavior_
Analysis_using_Binary_Instrumentation_Jonathan_Salwan.pdf
[19]: https://github.com/JonathanSalwan/Triton/tree/master/publications/SecurityDay2015_dynamic_symb
olic_execution_Jonathan_Salwan.pdf
[20]: doxygen_crawl.html
[21]: https://www.doxygen.org/index.html
