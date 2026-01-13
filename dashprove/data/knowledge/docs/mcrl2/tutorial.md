* mCRL2 tutorial
* [ View page source][1]

# mCRL2 tutorial[][2]

In this tutorial we give a basic introduction into the use of the mCRL2 toolset. In each of the
sections we present a number of new concepts, guided by an example, and some exercises to gain hands
on experience with the tools. Note that in this tutorial we mainly focus at the use of the tools,
and not on the theory that is underlying the tools. For the latter, we refer to [[GMRUW09]][3] as a
brief introduction of the main concepts, and to [[GM14a]][4] for an in-depth discussion.

Before starting this tutorial you should first [download][5] a copy of mCRL2. See also the [build
instructions][6].

Note

If you are using mCRL2 on Windows, then the compiling rewriter is unavailable, meaning that the flag
`-r jittyc` to any of the tools will fail.

In this tutorial, we assume that you will be using the tools from the command line. On Windows this
is the command prompt, on other platforms this is a terminal. Commands that should be entered at the
prompt are displayed as:

$ command

* [A Vending Machine][7]
  
  * [First variation][8]
  * [Second variation][9]
  * [Third variation][10]
* [Water cans][11]
* [Towers of Hanoi][12]
  
  * [Optimal strategy][13]
* [The Rope Bridge][14]
* [A Telephone Book][15]
* [Gossips][16]
* [Probabilistic processes][17]
* [Using mcrl2-gui][18]
  
  * [Obtaining a linear process specification][19]
  * [Generating a labelled transition system][20]
  * [*ltsgraph* and *ltsconvert*][21]
  * [*ltsview* and *diagraphica*][22]
  * [Simulating a linear process specification][23]
  * [Setting an external editor in mcrl2-gui][24]
    
    * [Concluding remarks][25]

## References[][26]

[[GM14a][27]]

J.F. Groote and M.R. Mousavi. [Modelling and Analysis of Communicating Systems][28]. The MIT Press.
2014.

[[GMRUW09][29]]

J.F. Groote, A.H.J. Mathijssen, M.A. Reniers, Y.S. Usenko and M.J. van Weerdenburg. Analysis of
distributed systems with mCRL2. In Process Algebra for Parallel and Distributed Processing. M.
Alexander and W. Gardner, eds. pp 99-128. Chapman & Hall, 2009.

[ Previous][30] [Next ][31]

© Copyright 2011-2025, Technische Universiteit Eindhoven.

Built with [Sphinx][32] using a [theme][33] provided by [Read the Docs][34].

[1]: ../../_sources/user_manual/tutorial/tutorial.rst.txt
[2]: #mcrl2-tutorial
[3]: #gmruw09
[4]: #gm14a
[5]: ../../home/download.html#download
[6]: ../../developer_manual/build_instructions/instructions.html#build-instructions
[7]: machine/index.html
[8]: machine/index.html#first-variation
[9]: machine/index.html#second-variation
[10]: machine/index.html#third-variation
[11]: watercans/index.html
[12]: hanoi/index.html
[13]: hanoi/index.html#optimal-strategy
[14]: ropebridge/index.html
[15]: phonebook/index.html
[16]: gossip/index.html
[17]: probability/index.html
[18]: mcrl2-gui/index.html
[19]: mcrl2-gui/index.html#obtaining-a-linear-process-specification
[20]: mcrl2-gui/index.html#generating-a-labelled-transition-system
[21]: mcrl2-gui/index.html#ltsgraph-and-ltsconvert
[22]: mcrl2-gui/index.html#ltsview-and-diagraphica
[23]: mcrl2-gui/index.html#simulating-a-linear-process-specification
[24]: mcrl2-gui/index.html#setting-an-external-editor-in-mcrl2-gui
[25]: mcrl2-gui/index.html#concluding-remarks
[26]: #references
[27]: #id2
[28]: https://mitpress.mit.edu/9780262027717/
[29]: #id1
[30]: ../../home/showcases/Stella_solar_car.html
[31]: machine/index.html
[32]: https://www.sphinx-doc.org/
[33]: https://github.com/readthedocs/sphinx_rtd_theme
[34]: https://readthedocs.org
