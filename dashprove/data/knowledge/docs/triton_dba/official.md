### About

Triton is a dynamic binary analysis library. It provides internal components that allow you to build
your program analysis tools, automate reverse engineering, perform software verification or just
emulate code.

* Dynamic symbolic execution
* Dynamic taint analysis
* AST representation of the x86, x86-64, ARM32, AArch64 and RISC-V 32/64 ISA semantic
* Expressions synthesis
* SMT simplification passes
* Lifting to LLVM as well as Z3 and back
* SMT solver interface to Z3 and Bitwuzla
* C++ and Python API

### Open source

Linux, Windows and OS X compatible. You can directly compile source code from our Github repository.

[Download**][1]
**

#### Install

The Triton library is Linux, Windows and OS X compatible. The Install procedure is available on this
[page][2]

**

#### Python Examples

A potential way to getting started is to see our Python [examples][3] already bootstrapped

**

#### C++ API

The library is fully developed in C++ and the API is available [here][4]

**

#### Python API

The library provides Python bindings and the API is available [here][5]

## They already used Triton

### Tools and libraries


* [Ponce][6]: IDA 2016 plugin contest winner! Symbolic Execution just one-click away!
* [QSynthesis][7]: Greybox Synthesizer geared for deobfuscation of assembly instructions.
* [Pimp][8]: Triton based R2 plugin for concolic execution and total control.
* [Exrop][9]: Automatic ROPChain Generation.
* [TritonDSE][10]: TritonDSE is a Python library providing exploration capabilities to Triton and
  some refinement easing its usage.
* [Titan][11]: Titan is a VMProtect devirtualizer using Triton.


### Papers and conference


* Strong Optimistic Solving for Dynamic Symbolic Execution
  Talk at: Ivannikov Memorial Workshop, Kazan, Russia, 2022. [[paper][12]] [[slide][13]]
  Authors: Parygina D., Vishnyakov A., Fedotov A.
  Abstract: *Dynamic symbolic execution (DSE) is an effective method for automated program testing
  and bug detection. It is increasing the code coverage by the complex branches exploration during
  hybrid fuzzing. DSE tools invert the branches along some execution path and help fuzzer examine
  previously unavailable program parts. DSE often faces over and underconstraint problems. The first
  one leads to significant analysis complication while the second one causes inaccurate symbolic
  execution. We propose strong optimistic solving method that eliminates irrelevant path predicate
  constraints for target branch inversion. We eliminate such symbolic constraints that the target
  branch is not control dependent on. Moreover, we separately handle symbolic branches that have
  nested control transfer instructions that pass control beyond the parent branch scope, e.g.
  return, goto, break, etc. We implement the proposed method in our dynamic symbolic execution tool
  Sydr. We evaluate the strong optimistic strategy, the optimistic strategy that contains only the
  last constraint negation, and their combination. The results show that the strategies combination
  helps increase either the code coverage or the average number of correctly inverted branches per
  one minute. It is optimal to apply both strategies together in contrast with other configurations.
  *
* 
* Greybox Program Synthesis: A New Approach to Attack Dataflow Obfuscation
  Talk at: Blackhat USA, Las Vegas, Nevada, 2021. [[slide][14]]
  Authors: Robin David
  Abstract: *This talk presents the latest advances in program synthesis applied for deobfuscation.
  It aims at demystifying this analysis technique by showing how it can be put into action on
  obfuscation. Especially the implementation Qsynthesis released for this talk shows a complete
  end-to-end workflow to deobfuscate assembly instructions back in optimized (deobfuscated)
  instructions reassembled back in the binary.*
* 
* From source code to crash test-case through software testing automation
  Talk at: C&ESAR, Rennes, France, 2021. [[paper][15]] [[slide][16]]
  Authors: Robin David, Jonathan Salwan, Justin Bourroux
  Abstract: *This paper present an approach automating the software testing process from a source
  code to the dynamic testing of the compiled program. More specifically, from a static analysis
  report indicating alerts on source lines it enables testing to cover these lines dynamically and
  opportunistically checking whether whether or not they can trigger a crash. The result is a test
  corpus allowing to cover alerts and to trigger them if they happen to be true positives. This
  paper discuss the methodology employed to track alerts down in the compiled binary, the testing
  engines selection process and the results obtained on a TCP/IP stack implementation for embedded
  and IoT systems.*
* 
* Symbolic Security Predicates: Hunt Program Weaknesses
  Talk at: Ivannikov ISP RAS Open Conference, Moscow, Russia, 2021. [[paper][17]] [[slide][18]]
  Authors: A.Vishnyakov, V.Logunova, E.Kobrin, D.Kuts, D.Parygina, A.Fedotov
  Abstract: *Dynamic symbolic execution (DSE) is a powerful method for path exploration during
  hybrid fuzzing and automatic bug detection. We propose security predicates to effectively detect
  undefined behavior and memory access violation errors. Initially, we symbolically execute program
  on paths that don’t trigger any errors (hybrid fuzzing may explore these paths). Then we construct
  a symbolic security predicate to verify some error condition. Thus, we may change the program data
  flow to entail null pointer dereference, division by zero, out-of-bounds access, or integer
  overflow weaknesses. Unlike static analysis, dynamic symbolic execution does not only report
  errors but also generates new input data to reproduce them. Furthermore, we introduce function
  semantics modeling for common C/C++ standard library functions. We aim to model the control flow
  inside a function with a single symbolic formula. This assists bug detection, speeds up path
  exploration, and overcomes overconstraints in path predicate. We implement the proposed techniques
  in our dynamic symbolic execution tool Sydr. Thus, we utilize powerful methods from Sydr such as
  path predicate slicing that eliminates irrelevant constraints. We present Juliet Dynamic to
  measure dynamic bug detection tools accuracy. The testing system also verifies that generated
  inputs trigger sanitizers. We evaluate Sydr accuracy for 11 CWEs from Juliet test suite. Sydr
  shows 95.59% overall accuracy. We make Sydr evaluation artifacts publicly available to facilitate
  results reproducibility.*
* 
* Towards Symbolic Pointers Reasoning in Dynamic Symbolic Execution
  Talk at: Ivannikov Memorial Workshop, Nizhny Novgorod, Russia, 2021. [[paper][19]] [[slide][20]]
  Authors: Daniil Kuts
  Abstract: *Dynamic symbolic execution is a widely used technique for automated software testing,
  designed for execution paths exploration and program errors detection. A hybrid approach has
  recently become widespread, when the main goal of symbolic execution is helping fuzzer increase
  program coverage. The more branches symbolic executor can invert, the more useful it is for
  fuzzer. A program control flow often depends on memory values, which are obtained by computing
  address indexes from user input. However, most DSE tools don't support such dependencies, so they
  miss some desired program branches. We implement symbolic addresses reasoning on memory reads in
  our dynamic symbolic execution tool Sydr. Possible memory access regions are determined by either
  analyzing memory address symbolic expressions, or binary searching with SMT-solver. We propose an
  enhanced linearization technique to model memory accesses. Different memory modeling methods are
  compared on the set of programs. Our evaluation shows that symbolic addresses handling allows to
  discover new symbolic branches and increase the program coverage.*
* 
* QSynth: A Program Synthesis based Approach for Binary Code Deobfuscation
  Talk at: BAR, San Diego, California, 2020. [[paper][21]]
  Authors: Robin David, Luigi Coniglio, Mariano Ceccato
  Abstract: *We present a generic approach leveraging both DSE and program synthesis to successfully
  synthesize programs obfuscated with Mixed-Boolean-Arithmetic, Data-Encoding or Virtualization. The
  synthesis algorithm proposed is an offline enumerate synthesis primitive guided by top-down
  breath-first search. We shows its effectiveness against a state-of-the-art obfuscator and its
  scalability as it supersedes other similar approaches based on synthesis. We also show its
  effectiveness in presence of composite obfuscation (combination of various techniques). This
  ongoing work enlightens the effectiveness of synthesis to target certain kinds of obfuscations and
  opens the way to more robust algorithms and simplification strategies.*
* 
* Sydr: Cutting Edge Dynamic Symbolic Execution
  Talk at: Ivannikov ISP RAS Open Conference, Moscow, Russia, 2020. [[paper][22]]
  Authors: A.Vishnyakov, A.Fedotov, D.Kuts, A.Novikov, D.Parygina, E.Kobrin, V.Logunova, P.Belecky,
  S.Kurmangaleev
  Abstract: *Dynamic symbolic execution (DSE) has enormous amount of applications in computer
  security (fuzzing, vulnerability discovery, reverse-engineering, etc.). We propose several
  performance and accuracy improvements for dynamic symbolic execution. Skipping non-symbolic
  instructions allows to build a path predicate 1.2--3.5 times faster. Symbolic engine simplifies
  formulas during symbolic execution. Path predicate slicing eliminates irrelevant conjuncts from
  solver queries. We handle each jump table (switch statement) as multiple branches and describe the
  method for symbolic execution of multi-threaded programs. The proposed solutions were implemented
  in Sydr tool. Sydr performs inversion of branches in path predicate. Sydr combines DynamoRIO
  dynamic binary instrumentation tool with Triton symbolic engine.*
* 
* Symbolic Deobfuscation: From Virtualized Code Back to the Original
  Talk at: DIMVA, Paris-Saclay, France, 2018. [[paper][23]] [[slide][24]]
  Authors: Jonathan Salwan, Sébastien Bardin, Marie-Laure Potet
  Abstract: *Software protection has taken an important place during the last decade in order to
  protect legit software against reverse engineering or tampering. Virtualization is considered as
  one of the very best defenses against such attacks. We present a generic approach based on
  symbolic path exploration, taint and recompilation allowing to recover, from a virtualized code, a
  devirtualized code semantically identical to the original one and close in size. We define
  criteria and metrics to evaluate the relevance of the deobfuscated results in terms of correctness
  and precision. Finally we propose an open-source setup allowing to evaluate the proposed approach
  against several forms of virtualization.*
* 
* Deobfuscation of VM based software protection
  Talk at: SSTIC, Rennes, France, 2017. [[french paper][25]] [[english slide][26]] [[french
  video][27]]
  Authors: Jonathan Salwan, Sébastien Bardin, Marie-Laure Potet
  Abstract: *In this presentation we describe an approach which consists to automatically analyze
  virtual machine based software protections and which recompiles a new version of the binary
  without such protections. This automated approach relies on a symbolic execution guide by a taint
  analysis and some concretization policies, then on a binary rewriting using LLVM transition.*
* 
* How Triton can help to reverse virtual machine based software protections
  Talk at: CSAW SOS, NYC, New York, 2016. [[slide][28]]
  Authors: Jonathan Salwan, Romain Thomas
  Abstract: *The first part of the talk is going to be an introduction to the Triton framework to
  expose its components and to explain how they work together. Then, the second part will include
  demonstrations on how it's possible to reverse virtual machine based protections using taint
  analysis, symbolic execution, SMT simplifications and LLVM-IR optimizations.*
* 
* Dynamic Binary Analysis and Obfuscated Codes
  Talk at: St'Hack, Bordeaux, France, 2016. [[slide][29]]
  Authors: Jonathan Salwan, Romain Thomas
  Abstract: *At this presentation we will talk about how a DBA (Dynamic Binary Analysis) may help a
  reverse engineer to reverse obfuscated code. We will first introduce some basic obfuscation
  techniques and then expose how it's possible to break some stuffs (using our open-source DBA
  framework - Triton) like detect opaque predicates, reconstruct CFG, find the original algorithm,
  isolate sensible data and many more... Then, we will conclude with a demo and few words about our
  future work.*
* 
* How Triton may help to analyse obfuscated binaries
  Publication at: MISC magazine 82, 2015. [[french article][30]]
  Authors: Jonathan Salwan, Romain Thomas
  Abstract: *Binary obfuscation is used to protect software's intellectual property. There exist
  different kinds of obfucation but roughly, it transforms a binary structure into another binary
  structure by preserving the same semantic. The aim of obfuscation is to ensure that the original
  information is "drown" in useless information that will make reverse engineering harder. In this
  article we will show how we can analyse an ofbuscated program and break some obfuscations using
  the Triton framework.*
* 
* Triton: A Concolic Execution Framework
  Talk at: SSTIC, Rennes, France, 2015. [[french paper][31]] [[detailed english slide][32]]
  Authors: Jonathan Salwan, Florent Saudel
  Abstract: *This talk is about the release of Triton, a concolic execution framework based on Pin.
  It provides components like a taint engine, a dynamic symbolic execution engine, a snapshot
  engine, translation of x64 instruction to SMT2, a Z3 interface to solve constraints and Python
  bindings. Based on these components, Triton offers the possibility to build tools for
  vulnerabilities research or reverse-engineering assistance.*
* 
* Dynamic Behavior Analysis Using Binary Instrumentation
  Talk at: St'Hack, Bordeaux, France, 2015. [[slide][33]]
  Authors: Jonathan Salwan
  Abstract: *This talk can be considered like the part 2 of our talk at SecurityDay. In the previous
  part, we talked about how it was possible to cover a targeted function in memory using the DSE
  (Dynamic Symbolic Execution) approach. Cover a function (or its states) doesn't mean find all
  vulnerabilities, some vulnerability doesn't crashes the program. That's why we must implement
  specific analysis to find specific bugs. These analysis are based on the binary instrumentation
  and the runtime behavior analysis of the program. In this talk, we will see how it's possible to
  find these following kind of bugs : off-by-one, stack / heap overflow, use-after-free, format
  string and {write, read}-what-where.*
* 
* Covering a function using a Dynamic Symbolic Execution approach
  Talk at: Security Day, Lille, France, 2015. [[slide][34]]
  Authors: Jonathan Salwan
  Abstract: *This talk is about binary analysis and instrumentation. We will see how it's possible
  to target a specific function, snapshot the context memory/registers before the function,
  translate the instrumentation into an intermediate representation,apply a taint analysis based on
  this IR, build/keep formulas for a Dynamic Symbolic Execution (DSE), generate a concrete value to
  go through a specific path, restore the context memory/register and generate another concrete
  value to go through another path then repeat this operation until the target function is covered.*

[1]: https://github.com/jonathansalwan/Triton
[2]: https://github.com/jonathansalwan/Triton#install
[3]: https://github.com/JonathanSalwan/Triton/tree/master/src/examples/python
[4]: documentation/doxygen/annotated.html
[5]: documentation/doxygen/py_triton_page.html
[6]: https://github.com/illera88/Ponce
[7]: https://github.com/quarkslab/qsynthesis
[8]: https://github.com/kamou/pimp
[9]: https://github.com/d4em0n/exrop
[10]: https://github.com/quarkslab/tritondse
[11]: https://github.com/archercreat/titan
[12]: https://github.com/JonathanSalwan/Triton/blob/master/publications/IVMEM2022-strong-optimistic-
parygina.pdf
[13]: https://github.com/JonathanSalwan/Triton/blob/master/publications/IVMEM2022-slide-strong-optim
istic-parygina.pdf
[14]: https://github.com/JonathanSalwan/Triton/raw/master/publications/BHUSA2021-David-Greybox-Progr
am-Synthesis.pdf
[15]: https://github.com/JonathanSalwan/Triton/raw/master/publications/CESAR2021_robin-david-paper.p
df
[16]: https://github.com/JonathanSalwan/Triton/raw/master/publications/CESAR2021_robin-david-slide.p
df
[17]: https://github.com/JonathanSalwan/Triton/raw/master/publications/ISPOPEN2021-security-predicat
es-vishnyakov.pdf
[18]: https://github.com/JonathanSalwan/Triton/raw/master/publications/ISPOPEN2021-slide-security-pr
edicates-vishnyakov.pdf
[19]: https://github.com/JonathanSalwan/Triton/raw/master/publications/IVMEM2021-symbolic-pointers-k
uts.pdf
[20]: https://github.com/JonathanSalwan/Triton/raw/master/publications/IVMEM2021-slide-symbolic-poin
ters-kuts.pdf
[21]: https://github.com/JonathanSalwan/Triton/raw/master/publications/BAR2020-qsynth-robin-david.pd
f
[22]: https://github.com/JonathanSalwan/Triton/raw/master/publications/ISPRAS2020-sydr.pdf
[23]: https://github.com/JonathanSalwan/Triton/raw/master/publications/DIMVA2018-deobfuscation-salwa
n-bardin-potet.pdf
[24]: https://github.com/JonathanSalwan/Triton/raw/master/publications/DIMVA2018-slide-deobfuscation
-salwan-bardin-potet.pdf
[25]: https://github.com/JonathanSalwan/Triton/raw/master/publications/SSTIC2017-French-Article-deso
bfuscation_binaire_reconstruction_de_fonctions_virtualisees-salwan_potet_bardin.pdf
[26]: https://github.com/JonathanSalwan/Triton/raw/master/publications/SSTIC2017_Deobfuscation_of_VM
_based_software_protection.pdf
[27]: https://static.sstic.org/videos2017/SSTIC_2017-06-07_P08.mp4
[28]: https://github.com/JonathanSalwan/Triton/raw/master/publications/CSAW2016-SOS-Virtual-Machine-
Deobfuscation-RThomas_JSalwan.pdf
[29]: https://github.com/JonathanSalwan/Triton/raw/master/publications/StHack2016_Dynamic_Binary_Ana
lysis_and_Obfuscated_Codes_RThomas_JSalwan.pdf
[30]: https://github.com/JonathanSalwan/Triton/raw/master/publications/MISC-82_French_Paper_How_Trit
on_may_help_to_analyse_obfuscated_binaries_RThomas_JSalwan.pdf
[31]: https://github.com/JonathanSalwan/Triton/raw/master/publications/SSTIC2015_French_Paper_Triton
_Framework_dexecution_Concolique_FSaudel_JSalwan.pdf
[32]: https://github.com/JonathanSalwan/Triton/raw/master/publications/SSTIC2015_English_slide_detai
led_version_Triton_Concolic_Execution_FrameWork_FSaudel_JSalwan.pdf
[33]: https://github.com/JonathanSalwan/Triton/raw/master/publications/StHack2015_Dynamic_Behavior_A
nalysis_using_Binary_Instrumentation_Jonathan_Salwan.pdf
[34]: https://github.com/JonathanSalwan/Triton/raw/master/publications/SecurityDay2015_dynamic_symbo
lic_execution_Jonathan_Salwan.pdf
