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
[PRISM-games][41]

[PRISM-games][42] is an extension of PRISM for probabilistic model checking of *stochastic
multi-player games*.

See the [website][43] and read the [papers][44] for more information.

PRISM is a *probabilistic model checker*, a tool for formal modelling and analysis of systems that
exhibit random or probabilistic behaviour. It has been used to analyse systems from many different
[application domains][45], including communication and multimedia protocols, randomised distributed
algorithms, security protocols, biological systems and many others.

[Example PRISM graph]

PRISM can build and analyse many types of probabilistic models:

* discrete-time and continuous-time Markov chains (DTMCs and CTMCs)
* Markov decision processes (MDPs) and probabilistic automata (PAs)
* probabilistic timed automata (PTAs)
* partially observable MDPs and PTAs (POMDPs and POPTAs)
* interval Markov chains and MDPs (IDTMCs and IMDPs)

plus extensions of these models with costs and rewards.

Models are described using the [PRISM language][46], a simple, state-based language. PRISM provides
support for automated analysis of a wide range of quantitative properties of these models, e.g.
"what is the probability of a failure causing the system to shut down within 4 hours?", "what is the
worst-case probability of the protocol terminating in error, over all possible initial
configurations?", "what is the expected size of the message queue after 30 minutes?", or "what is
the worst-case expected time taken for the algorithm to terminate?". The [property specification
language][47] incorporates the temporal logics PCTL, CSL, LTL and PCTL*, as well as extensions for
quantitative specifications and costs/rewards.

PRISM incorporates state-of-the art *symbolic* data structures and algorithms, based on BDDs (Binary
Decision Diagrams) and MTBDDs (Multi-Terminal Binary Decision Diagrams) [[KNP04b][48], [Par02][49]].
It also includes a *discrete-event simulation* engine, providing support for
*approximate/statistical model checking*, and implementations of various different analysis
techniques, such as *quantitative abstraction refinement* and *symmetry reduction*.

PRISM is *free* and *open source*, [released][50] under the GNU General Public License ([GPL][51]).

To cite PRISM, please use the most recent tool paper, from CAV'11:

* [Marta Kwiatkowska][52], [Gethin Norman][53] and [David Parker][54]. [PRISM 4.0: Verification of
  Probabilistic Real-time Systems][55]. In *Proc. 23rd International Conference on Computer Aided
  Verification (CAV’11)*, volume 6806 of LNCS, pages 585-591, Springer, 2011 [[pdf][56]] [[bib][57]]
  [[cites][58]]
Site hosted at the Department of Computer Science, University of Oxford

### [Latest News][59]

September 2025: New paper surveying [applications of probabilistic model checking][60], a
contribution to the [Festschrift][61] for Christel Baier.
August 2025: PRISM 4.9 is now [available][62], including model import/export enhancements and more.
Further details [here][63].
April 2024: We are honoured that PRISM and its creators have won the 2024 [ETAPS Test-of-Time Tool
Award][64]! For more details, see [here][65].

[ [more news...][66] ]

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
[41]: /games/
[42]: /games/
[43]: /games/
[44]: /games/publ.php
[45]: /casestudies/
[46]: /manual/ThePRISMLanguage/Introduction
[47]: /manual/PropertySpecification/Introduction
[48]: /bibitem.php?key=KNP04b
[49]: /bibitem.php?key=Par02
[50]: download.php
[51]: http://www.gnu.org/licenses/gpl.html
[52]: http://www.cs.ox.ac.uk/people/marta.kwiatkowska/
[53]: http://www.dcs.gla.ac.uk/people/personal/gethin/
[54]: https://www.cs.ox.ac.uk/people/david.parker/home.html
[55]: /bibitem.php?key=KNP11
[56]: /papers/cav11.pdf
[57]: /bibtex/KNP11.bib
[58]: https://scholar.google.com/citations?view_op=view_citation&citation_for_view=nEKbXSMAAAAJ:aqlV
kmm33-oC
[59]: news.php
[60]: /bibitem.php?key=KNP25
[61]: https://link.springer.com/book/10.1007/978-3-031-97439-7
[62]: /download.php
[63]: /download.php
[64]: https://etaps.org/awards/test-of-time-tool/
[65]: /news/2024-04-etaps.php
[66]: /news.php
