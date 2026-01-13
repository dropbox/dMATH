Ultimate is a program analysis framework. Ultimate consists of several plugins that perform steps of
an program analysis, e.g., parsing source code, transforming programs from one representation to
another, or analyze programs. Toolchains of these plugins can perform complex tasks, e.g., verify
that a C program fulfills a given specification.

### Tools

[Automizer][1] Boogie C

Verification of safety properties based on an automata-theoretic approach to software verification.

[Read more][2] [Try online][3]
[Büchi Automizer][4] Boogie C

Termination analysis based on Büchi automata.

[Read more][5] [Try online][6]
[GemCutter][7] Boogie C

A verifier for concurrent programs based on commutativity – the observation that for certain
statements, the execution order does not matter.

[Read more][8] [Try online][9]
[Kojak][10] Boogie C

A software model checker.

[Read more][11] [Try online][12]
[Taipan][13] Boogie C

Verification of safety properties using trace abstraction and abstract interpretation on path
programs.

[Read more][14] [Try online][15]
[LTL Automizer][16] C

An LTL software model checker based on Büchi programs.

[Read more][17] [Try online][18]
[Lasso Ranker][19] Boogie C

Synthesis of ranking functions and nontermination arguments.

[Read more][20] [Try online][21]
[Automata Library][22] automata_script

Nested Word Automata, Büchi Nested Word Automata, Petri Net, Alternating Finite Automata, Tree
Automata.

[Read more][23] [Try online][24]
[Referee][25] Boogie C

Checking validity of given invariants.

[Read more][26] [Try online][27]
[Eliminator][28] Smt

Run SMT script.

[Read more][29] [Try online][30]

### [Awards][31]

ULTIMATE tools have won many prizes at competitions like the [International Competition on Software
Verification (SV-COMP)][32] or the [Termination Competition][33]. ULTIMATE Automizer has won the
*Overall* ranking of SV-COMP five times (2016, 2017, 2023, 2024 and 2025). Find more information
[here][34].

### Development

The Ultimate [source code and some documentation][35] is available at GitHub.
The core of Ultimate and many plugins are subject to the LGPLv3 license with a linking exception to
Eclipse RCP and Eclipse CDT.

### [Developers][36]

The current main developers of Ultimate are:

[Matthias Heizmann][37]
[Daniel Dietsch][38]
[Dominik Klumpp][39]
[Frank Schüssele][40]
[Manuel Bentele][41]
[Marcel Ebbinghaus][42]

Most Ultimate developers are students and researchers in the [software engineering group of Andreas
Podelski][43] at the [University of Freiburg][44]. Current and former developers of Ultimate can be
found [here][45].

### Development

The Ultimate [source code and some documentation][46] is available at GitHub.
The core of Ultimate and many plugins are subject to the LGPLv3 license with a linking exception to
Eclipse RCP and Eclipse CDT.

[1]: /automizer/
[2]: /automizer/
[3]: /webinterface/?tool=automizer
[4]: /buechi_automizer/
[5]: /buechi_automizer/
[6]: /webinterface/?tool=buechi_automizer
[7]: /gemcutter/
[8]: /gemcutter/
[9]: /webinterface/?tool=gemcutter
[10]: /kojak/
[11]: /kojak/
[12]: /webinterface/?tool=kojak
[13]: /taipan/
[14]: /taipan/
[15]: /webinterface/?tool=taipan
[16]: /ltl_automizer/
[17]: /ltl_automizer/
[18]: /webinterface/?tool=ltl_automizer
[19]: /lasso_ranker/
[20]: /lasso_ranker/
[21]: /webinterface/?tool=lasso_ranker
[22]: /automata_library/
[23]: /automata_library/
[24]: /webinterface/?tool=automata_library
[25]: /referee/
[26]: /referee/
[27]: /webinterface/?tool=referee
[28]: /eliminator/
[29]: /eliminator/
[30]: /webinterface/?tool=eliminator
[31]: /awards/
[32]: https://sv-comp.sosy-lab.org/
[33]: https://termination-portal.org/wiki/Termination_Portal
[34]: /awards/
[35]: https://github.com/ultimate-pa/ultimate
[36]: /developers/
[37]: https://swt.informatik.uni-freiburg.de/staff/heizmann
[38]: https://swt.informatik.uni-freiburg.de/staff/dietsch
[39]: https://dominik-klumpp.net/
[40]: https://swt.informatik.uni-freiburg.de/staff/schuessele
[41]: https://swt.informatik.uni-freiburg.de/staff/bentele
[42]: https://swt.informatik.uni-freiburg.de/staff/ebbinghaus
[43]: https://swt.informatik.uni-freiburg.de/
[44]: https://www.uni-freiburg.de/
[45]: /developers/
[46]: https://github.com/ultimate-pa/ultimate
