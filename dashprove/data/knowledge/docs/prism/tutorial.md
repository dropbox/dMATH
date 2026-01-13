[[www.prismmodelchecker.org]][1]
[[P]][2]

* [Home][3]
* •
* [About][4]
  
  * » [About PRISM][5]
  * » [People][6]
  * » [Sponsors][7]
  * » [Contact][8]
  * » [News][9]
* •
* [Downloads][10]
  
  * » [PRISM][11]
  * » [PRISM-games][12]
  * » [Benchmarks][13]
  * » [GitHub][14]
  * » [Other downloads][15]
* •
* [Documentation][16]
  
  * » [Installation][17]
  * » [Manual][18]
  * » [FAQ][19]
  * » [Tutorial][20]
  * » [Lectures][21]
* •
* [Manual][22]
* •
* [Publications][23]
  
  * » [Selected publications][24]
  * » [PRISM publications][25]
  * » [PRISM bibliography][26]
  * » [External publications][27]
  * » [Search][28]
* •
* [Case Studies][29]
* •
* [Support][30]
  
  * » [Installation FAQ][31]
  * » [PRISM FAQ][32]
  * » [Forum (Google)][33]
* •
* [Developers][34]
  
  * » [GitHub][35]
  * » [Developer resources][36]
  * » [Developer forum][37]
* •
* [PRISM-games][38]
  
  * » [Download][39]
  * » [Publications][40]

# PRISM Tutorial

This tutorial will introduce you to the PRISM tool using a selection of example models.

The tutorial comprises several parts. If you are new to the tool, we recommend that you start by
working through the first part (Knuth's die algorithm) to see the basics. After that, you should be
able to look at the remaining parts in any order, depending in which models or applications you are
interested in. If anything is unclear, the best place to look for answers is the [PRISM manual][41].

* [Knuth's die algorithm][42]: This uses a simple discrete-time Markov chain (DTMC) example - a
  randomised algorithm for modelling a 6-sided die with a fair coin due to Don Knuth. It introduces
  the basics of the PRISM modelling language and the PRISM tool.
  
* [Herman's self-stabilisation algorithm][43]: This uses another simple randomised algorithm: a
  self-stabilisation algorithm due to Herman. This is also modelled as a DTMC and introduces some
  additional features of the PRISM modelling and property languages.
  
* [Robot navigation][44]: This introduces the use of Markov decision processes (MDPs) for generating
  control policies based on temporal logic specifications, via a simple example of a robot
  navigating through an uncertain environment.
  
* [Autonomous drone][45]: This shows how to use interval Markov decision processes (IMDPs) to
  incorporate epistemic uncertainty into probabilistic model checking of MDPs, specifically
  regarding the precise value of transition probabilities. It uses an example of an autonomous
  drone.
  
* [Dynamic power management][46]: This introduces a multi-component system modelled as a
  continuous-time Markov chain (CTMC): a controller for a dynamic power management system.
  
* [Circadian clock][47]: This demonstrates the use of PRISM to study a biological system, modelled
  as a CTMC: a circadian clock.
  
* [EGL contract signing protocol][48]: This uses a case study from the field of computer security,
  modelled as a DTMC: the EGL contract signing protocol.
  
* [Dining philosophers problem][49]: This introduces the use of a MDP to verify a simple distributed
  randomised algorithm: the dining philosophers problem.

If you have questions, comments, or suggestions regarding this tutorial, please [contact][50] us.








Site hosted at the Department of Computer Science, University of Oxford

### [Documentation][51]

* [Installation][52]
* [Manual][53]
* [FAQ][54]
* [Tutorial][55]
* [Lectures][56]

[1]: /
[2]: /
[3]: /
[4]: /about.php
[5]: /about.php
[6]: /people.php
[7]: /sponsors.php
[8]: /contact.php
[9]: /news.php
[10]: /download.php
[11]: /download.php
[12]: /games/download.php
[13]: /benchmarks/
[14]: https://github.com/prismmodelchecker
[15]: /other-downloads.php
[16]: /doc/
[17]: /manual/InstallingPRISM
[18]: /manual/
[19]: /manual/FrequentlyAskedQuestions
[20]: /tutorial/
[21]: /lectures/
[22]: /manual/
[23]: /publ-lists.php
[24]: /publ-selected.php
[25]: /publications.php
[26]: /bib.php
[27]: /bib-ext.php
[28]: /publ-search.php
[29]: /casestudies/index.php
[30]: /support.php
[31]: /manual/InstallingPRISM/CommonProblemsAndQuestions
[32]: /manual/FrequentlyAskedQuestions/
[33]: http://groups.google.com/group/prismmodelchecker
[34]: https://github.com/prismmodelchecker/prism/wiki
[35]: https://github.com/prismmodelchecker
[36]: https://github.com/prismmodelchecker/prism/wiki
[37]: http://groups.google.com/group/prismmodelchecker-dev
[38]: /games/
[39]: /games/download.php
[40]: /games/publ.php
[41]: /manual/
[42]: die.php
[43]: herman.php
[44]: robot.php
[45]: drone.php
[46]: power.php
[47]: circadian.php
[48]: egl.php
[49]: phil.php
[50]: /contact.php
[51]: /doc/
[52]: /manual/InstallingPRISM/
[53]: /manual/
[54]: /manual/FrequentlyAskedQuestions/
[55]: /tutorial/
[56]: /lectures/
