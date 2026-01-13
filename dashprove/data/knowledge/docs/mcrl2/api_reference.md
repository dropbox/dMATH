* Tool documentation
* [ View page source][1]

# Tool documentation[][2]

Below the main tools provided in the toolset are given. The common tools are suitable for all main
tasks to be carried out with the toolset. The experimental tools are tools that are under
development and provide additional but still experimental functionality.

## List of the common tools[][3]

* [besinfo][4]
* [bespp][5]
* [bessolve][6]
* [diagraphica][7]
* [lps2lts][8]
* [lps2pbes][9]
* [lpsactionrename][10]
* [lpsbinary][11]
* [lpsbisim2pbes][12]
* [lpsconfcheck][13]
* [lpsconstelm][14]
* [lpsfununfold][15]
* [lpsinfo][16]
* [lpsinvelm][17]
* [lpsparelm][18]
* [lpsparunfold][19]
* [lpspp][20]
* [lpsreach][21]
* [lpsrewr][22]
* [lpssim][23]
* [lpsstategraph][24]
* [lpssumelm][25]
* [lpssuminst][26]
* [lpsuntime][27]
* [lpsxsim][28]
* [lts2lps][29]
* [lts2pbes][30]
* [ltscompare][31]
* [ltsconvert][32]
* [ltsgraph][33]
* [ltsinfo][34]
* [ltspbisim][35]
* [ltspcompare][36]
* [ltsview][37]
* [mcrl2-gui][38]
* [mcrl22lps][39]
* [mcrl2i][40]
* [mcrl2ide][41]
* [mcrl2xi][42]
* [pbes2bes][43]
* [pbes2bool][44]
* [pbes2booldeprecated][45]
* [pbesconstelm][46]
* [pbesinfo][47]
* [pbesinst][48]
* [pbesparelm][49]
* [pbespgsolve][50]
* [pbespp][51]
* [pbesrewr][52]
* [pbessolve][53]
* [pbessolvesymbolic][54]
* [pbesstategraph][55]
* [tracepp][56]
* [txt2bes][57]
* [txt2lps][58]
* [txt2pbes][59]

## List of the experimental tools[][60]

* [besconvert][61]
* [lpscleave][62]
* [lpscombine][63]
* [lpsrealelm][64]
* [lpssymbolicbisim][65]
* [pbesabsinthe][66]
* [pbesabstract][67]
* [pbeschain][68]
* [pbesfixpointsolve][69]
* [pbespareqelm][70]
* [pbespor][71]
* [pbessymbolicbisim][72]
* [symbolic_exploration][73]

In the source distribution there are more tools such as the deprecated and developer tools.

## File formats[][74]

This page lists all file formats supported by the mCRL2 toolset.

> ───────────┬─────────┬───────┬───────────────────────────────────────────────────────────
> File format│Extension│Type   │Description                                                
> ───────────┼─────────┼───────┼───────────────────────────────────────────────────────────
> mCRL2      │.mcrl2   │textual│[mCRL2 specification][75]                                  
> ───────────┼─────────┼───────┼───────────────────────────────────────────────────────────
> MCF        │.mcf     │textual│[µ-Calculus][76]                                           
> ───────────┼─────────┼───────┼───────────────────────────────────────────────────────────
> LPS        │.lps     │binary │[Linear Process Specifications][77]                        
> ───────────┼─────────┼───────┼───────────────────────────────────────────────────────────
> PBES       │.pbes    │binary │[Parameterised Boolean Equation Systems][78]               
> ───────────┼─────────┼───────┼───────────────────────────────────────────────────────────
> BES        │.bes     │binary │[Boolean Equation Systems][79]                             
> ───────────┼─────────┼───────┼───────────────────────────────────────────────────────────
> GM         │.gm      │textual│parity game in the PGSolver format                         
> ───────────┼─────────┼───────┼───────────────────────────────────────────────────────────
> LTS        │.lts     │binary │labelled transition system in the [mCRL2 LTS format][80]   
> ───────────┼─────────┼───────┼───────────────────────────────────────────────────────────
> AUT        │.aut     │textual│labelled transition system in the [The aut format][81]     
> ───────────┼─────────┼───────┼───────────────────────────────────────────────────────────
> FSM        │.fsm     │textual│labelled transition system in the [The FSM file format][82]
> ───────────┼─────────┼───────┼───────────────────────────────────────────────────────────
> trace      │.trc     │binary │trace for simulation                                       
> ───────────┼─────────┼───────┼───────────────────────────────────────────────────────────
> DOT        │.dot     │textual│[DOT file format][83] (subgraphs as nodes are not          
>            │         │       │supported)                                                 
> ───────────┴─────────┴───────┴───────────────────────────────────────────────────────────

* [Formats for Labelled Transition Systems][84]

## External tools[][85]

The tools given below are not part of the toolset, but are standalone tools that have mCRL2-related
functionalities.

* A VSCode [extension][86] similar to the mcrl2ide, allowing visualisation, simulation and
  verification.
* A tool for generating LaTeX from a mu-calculus formula in the mCRL2 syntax: [GitHub][87], [Web
  app][88].
[ Previous][89] [Next ][90]

© Copyright 2011-2025, Technische Universiteit Eindhoven.

Built with [Sphinx][91] using a [theme][92] provided by [Read the Docs][93].

[1]: ../../_sources/user_manual/tools/tools.rst.txt
[2]: #tool-documentation
[3]: #list-of-the-common-tools
[4]: release/besinfo.html
[5]: release/bespp.html
[6]: release/bessolve.html
[7]: release/diagraphica.html
[8]: release/lps2lts.html
[9]: release/lps2pbes.html
[10]: release/lpsactionrename.html
[11]: release/lpsbinary.html
[12]: release/lpsbisim2pbes.html
[13]: release/lpsconfcheck.html
[14]: release/lpsconstelm.html
[15]: release/lpsfununfold.html
[16]: release/lpsinfo.html
[17]: release/lpsinvelm.html
[18]: release/lpsparelm.html
[19]: release/lpsparunfold.html
[20]: release/lpspp.html
[21]: release/lpsreach.html
[22]: release/lpsrewr.html
[23]: release/lpssim.html
[24]: release/lpsstategraph.html
[25]: release/lpssumelm.html
[26]: release/lpssuminst.html
[27]: release/lpsuntime.html
[28]: release/lpsxsim.html
[29]: release/lts2lps.html
[30]: release/lts2pbes.html
[31]: release/ltscompare.html
[32]: release/ltsconvert.html
[33]: release/ltsgraph.html
[34]: release/ltsinfo.html
[35]: release/ltspbisim.html
[36]: release/ltspcompare.html
[37]: release/ltsview.html
[38]: release/mcrl2-gui.html
[39]: release/mcrl22lps.html
[40]: release/mcrl2i.html
[41]: release/mcrl2ide.html
[42]: release/mcrl2xi.html
[43]: release/pbes2bes.html
[44]: release/pbes2bool.html
[45]: release/pbes2booldeprecated.html
[46]: release/pbesconstelm.html
[47]: release/pbesinfo.html
[48]: release/pbesinst.html
[49]: release/pbesparelm.html
[50]: release/pbespgsolve.html
[51]: release/pbespp.html
[52]: release/pbesrewr.html
[53]: release/pbessolve.html
[54]: release/pbessolvesymbolic.html
[55]: release/pbesstategraph.html
[56]: release/tracepp.html
[57]: release/txt2bes.html
[58]: release/txt2lps.html
[59]: release/txt2pbes.html
[60]: #list-of-the-experimental-tools
[61]: experimental/besconvert.html
[62]: experimental/lpscleave.html
[63]: experimental/lpscombine.html
[64]: experimental/lpsrealelm.html
[65]: experimental/lpssymbolicbisim.html
[66]: experimental/pbesabsinthe.html
[67]: experimental/pbesabstract.html
[68]: experimental/pbeschain.html
[69]: experimental/pbesfixpointsolve.html
[70]: experimental/pbespareqelm.html
[71]: experimental/pbespor.html
[72]: experimental/pbessymbolicbisim.html
[73]: experimental/symbolic_exploration.html
[74]: #file-formats
[75]: ../language_reference/mcrl2.html#language-mcrl2
[76]: ../language_reference/mucalc.html#language-mu-calculus
[77]: ../language_reference/lps.html#language-lps
[78]: ../language_reference/pbes.html#language-pbes
[79]: ../language_reference/bes.html#language-bes
[80]: lts.html#language-mcrl2-lts
[81]: lts.html#language-aut-lts
[82]: lts.html#language-fsm-lts
[83]: http://www.graphviz.org/doc/info/lang.html
[84]: lts.html
[85]: #external-tools
[86]: https://marketplace.visualstudio.com/items/?itemName=CptWesley.mcrl2
[87]: https://github.com/TarVK/mCRL2-formatter
[88]: https://tarvk.github.io/mCRL2-formatter/demo/build/
[89]: ../language_reference/parsing_2025.html
[90]: release/besinfo.html
[91]: https://www.sphinx-doc.org/
[92]: https://github.com/readthedocs/sphinx_rtd_theme
[93]: https://readthedocs.org
