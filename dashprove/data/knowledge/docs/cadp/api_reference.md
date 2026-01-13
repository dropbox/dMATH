CADP On-Line Manual Pages

────────────────────────────────────────────────────────────────────────────────────────────────────
[[CADP logo]][1]                                                                                    
────────────────────────────────────────────────────────────────────────────────────────────────────
What is CADP?                                                                                       
────────────────────────────────────────────────────────────────────────────────────────────────────
[Home Page][2]                                                                                      
[Tools Overview][3]                                                                                 
[Current Status][4]                                                                                 
[Recent Changes][5]                                                                                 
────────────────────────────────────────────────────────────────────────────────────────────────────
Installation                                                                                        
────────────────────────────────────────────────────────────────────────────────────────────────────
[How to obtain CADP?][6]                                                                            
[Usage Statistics][7]                                                                               
────────────────────────────────────────────────────────────────────────────────────────────────────
Documentation                                                                                       
────────────────────────────────────────────────────────────────────────────────────────────────────
[FAQ][8]                                                                                            
[Tutorials][9]                                                                                      
[Publications][10]                                                                                  
[Manual Pages][11]                                                                                  
[Demo Examples][12]                                                                                 
[Wikipedia][13]                                                                                     
────────────────────────────────────────────────────────────────────────────────────────────────────
CADP Community                                                                                      
────────────────────────────────────────────────────────────────────────────────────────────────────
[Newsletters][14]                                                                                   
[Education & Training][15]                                                                          
[Case Studies][16]                                                                                  
[Research Tools][17]                                                                                
────────────────────────────────────────────────────────────────────────────────────────────────────
CADP Resources                                                                                      
────────────────────────────────────────────────────────────────────────────────────────────────────
[VLPN Benchmarks][18]                                                                               
[VLSAT Benchmarks][19]                                                                              
[VLTS Benchmarks][20]                                                                               
[Other Resources][21]                                                                               
────────────────────────────────────────────────────────────────────────────────────────────────────
About CADP                                                                                          
────────────────────────────────────────────────────────────────────────────────────────────────────
[Contributors][22]                                                                                  
────────────────────────────────────────────────────────────────────────────────────────────────────

This Page gives the latest versions of the manual pages of the tools available in the [CADP
toolbox][23].

Note: The manual pages below are the most recent versions of the CADP documentation, and refer to
the current or immediately forthcoming release of CADP. If an earlier version of CADP version is
installed at your site, the documentation may contain some differences.


───────────────────────────────
1. CADP Languages and Formats  
───────────────────────────────


* *[aut][24]*
    simple file format for labelled transition systems
  *[bcg][25]*
    Binary Coded Graphs: binary file format for labelled transition systems
  *[bes][26]*
    text file format for Boolean Equation Systems
  *[exp][27]*
    language for describing networks of communicating automata
  *fsp*
    process calculus defined by Jeff Magee and Jeff Kramer (Imperial College)
    *see the [fsp2lotos][28] and [fsp.open][29] manual pages*
  *[gcf][30]*
    Grid Configuration File format
  *hid*
    file format for hiding labels
    *see the [caesar_hide_1][31] manual page*
  *lnt*
    newest specification language supported by CADP
    adapted from E-LOTOS (ISO International Standard 15437:2001)
    *see the [tutorial page][32], section "Tutorials for LNT"*
  *[lotos][33]*
    oldest specification language supported by CADP
    (ISO International Standard 8807:1989)
    *see the [tutorial page][34], section "Tutorials for LOTOS"*
  *[mcl][35]*
    Model Checking Language (versions 3, 4, and 5)
  *[mcl3][36]*
    Model Checking Language version 3 (regular alternation-free mu-calculus)
  *[mcl4][37]*
    Model Checking Language version 4 (value-passing modal mu-calculus)
  *[mcl5][38]*
    Model Checking Language version 5 (probabilistic value-passing modal mu-calculus)
  *[nupn][39]*
    Nested-Unit Petri Nets
  *[pbg][40]*
    Partitioned BCG File format
  *[rbc][41]*
    textual file format for random BES (Boolean Equation Systems) configuration
  *ren*
    file format for renaming labels
    *see the [caesar_rename_1][42] manual page*
  *[seq][43]*
    CADP common format for execution sequences (i.e., traces)
  *[svl-lang][44]*
    script language for verification scenarios
  *[xtl-lang][45]*
    language for value-based temporal logic formulas

────────────────────────────
2. CADP Command-Line Tools  
────────────────────────────


* *[bcg_cmp][46]*
    equivalence comparison of normal, probabilistic, or stochastic labeled transitions systems (LTS)
    encoded in the BCG format
  *[bcg_draw][47]*
    display graphs encoded in the BCG format
  *[bcg_edit][48]*
    edit interactively the PostScript representation of BCG graphs
  *[bcg_graph][49]*
    generate various kinds of useful BCG graphs
  *[bcg_info][50]*
    display information about graphs encoded in the BCG format
  *[bcg_io][51]*
    convert graphs from and into the BCG format
  *[bcg_labels][52]*
    modify the labels of graphs encoded in the BCG format
  *[bcg_lib][53]*
    generate dynamic libraries for graphs encoded in the BCG format
  *[bcg_merge][54]*
    translation of a partitioned BCG graph into one single BCG graph
  *[bcg_min][55]*
    minimization or reduction of normal, probabilistic, or stochastic labeled transitions systems
    (LTS) encoded in the BCG format
  *[bcg_open][56]*
    OPEN/CAESAR connection for graphs encoded in the BCG format
  *[bcg_steady][57]*
    steady-state numerical analysis of (extended) continuous-time Markov chains encoded in the BCG
    format
  *[bcg_transient][58]*
    transient numerical analysis of (extended) continuous-time Markov chains encoded in the BCG
    format
  *[bes_solve][59]*
    resolution of boolean equation systems
  *[bisimulator][60]*
    on-the-fly equivalence/preorder checking
  *[caesar.adt][61]*
    translation of LOTOS abstract data types into C
  *[caesar.bdd][62]*
    structural and behavioural analysis of Nested-Unit Petri Nets
  *[caesar][63]*
    compilation & verification of LOTOS specifications
  *[caesar.indent][64]*
    LOTOS specifications pretty-printer
  *[contributor][65]*
    CADP contribution assistant
  *[cunctator][66]*
    on-the-fly steady-state simulation of continuous-time Markov chains
  *[declarator][67]*
    test an OPEN/CAESAR implementation
  *[determinator][68]*
    elimination of nondeterminism for stochastic systems
  *[distributor][69]*
    state space generation using distributed reachability analysis
  *[evaluator][70]*
    a family of on-the-fly model checkers
  *[evaluator3][71]*
    on-the-fly model checking of MCL v3 formulas
  *[evaluator4][72]*
    on-the-fly model checking of MCL v4 formulas
  *[evaluator5][73]*
    on-the-fly model checking of MCL v5 formulas
  *[executor][74]*
    random execution
  *[exhibitor][75]*
    search for execution sequences matching a given pattern
  *[exp.open][76]*
    OPEN/CAESAR connection for EXP networks of communicating automata
  *[fsp.open][77]*
    OPEN/CAESAR connection for the FSP language
  *[fsp2lotos][78]*
    FSP to LOTOS translator
  *[generator][79]*
    BCG graph generation using reachability analysis
  *[installator][80]*
    CADP installation assistant
  *[lnt.open][81]*
    OPEN/CAESAR connection for the LNT language
  *[lnt2lotos][82]*
    LNT to LOTOS translator
  *[lnt_merge][83]*
    merge a multi-module LNT model into a single-module LNT model
  *[lotos.open][84]*
    OPEN/CAESAR connection for the LOTOS language
  *[nupn_info][85]*
    query and transformation of Nested-Unit Petri Nets
  *[ocis][86]*
    Open/Caesar Interactive Simulator
  *[pbg_cp][87]*
    copy a partitioned BCG graph
  *[pbg_info][88]*
    display information about a partitioned BCG graph
  *[pbg_mv][89]*
    move a partitioned BCG graph
  *[pbg_rm][90]*
    remove a partitioned BCG graph
  *[predictor][91]*
    predict the feasability of reachability analysis
  *[projector][92]*
    semi-composition and generation of Labelled Transition Systems
  *[reductor][93]*
    BCG graph generation using reachability analysis combined with on-the-fly reduction
  *[scrutator][94]*
    pruning of Labelled Transition Systems
  *[seq.open][95]*
    OPEN/CAESAR connection for traces encoded in the SEQ format
  *[simulator][96]*
    interactive simulator with ASCII command-line interface
  *[svl][97]*
    compilation and execution of SVL scripts
  *[terminator][98]*
    deadlock detection
  *[tgv][99]*
    Test Generation from transitions systems using Verification techniques
  *[traian][100]*
    compilation of LNT specifications
  *[tst][101]*
    CADP installation and configuration auto-test facility
  *[xeuca][102]*
    graphical-user interface for the EUCALYPTUS tools
  *[xsimulator][103]*
    interactive simulator with X-windows interface
  *[xtl][104]*
    evaluation of value-based temporal logic formulas

────────────────────────────────────────────
3. CADP Application Programming Interfaces  
────────────────────────────────────────────

Note: The manual pages of the Open/Caesar Application Programming Interfaces are available as part
of the CADP software package.


* *[bcg_read][105]*
    a simple interface to read a BCG graph
  *[bcg_write][106]*
    a simple interface to produce a BCG graph
  *[caesar_area_1][107]*
    the ``area_1'' library of OPEN/CAESAR
  *[caesar_bitmap][108]*
    the ``bitmap'' library of OPEN/CAESAR
  *[caesar_cache_1][109]*
    the ``cache_1'' library of OPEN/CAESAR
  *[caesar_diagnostic_1][110]*
    the ``diagnostic_1'' library of OPEN/CAESAR
  *[caesar_edge][111]*
    the ``edge'' library of OPEN/CAESAR
  *[caesar_graph][112]*
    the ``graph'' library of OPEN/CAESAR
  *[caesar_hash][113]*
    the ``hash'' library of OPEN/CAESAR
  *[caesar_hide_1][114]*
    the ``hide_1'' library of OPEN/CAESAR
  *[caesar_mask_1][115]*
    the ``mask_1'' library of OPEN/CAESAR
  *[caesar_rename_1][116]*
    the ``rename_1'' library of OPEN/CAESAR
  *[caesar_solve_1][117]*
    the ``solve_1'' library of OPEN/CAESAR
  *[caesar_solve_2][118]*
    the ``solve_2'' library of OPEN/CAESAR
  *[caesar_stack_1][119]*
    the ``stack_1'' library of OPEN/CAESAR
  *[caesar_standard][120]*
    the ``standard'' library of OPEN/CAESAR
  *[caesar_table_1][121]*
    the ``table_1'' library of OPEN/CAESAR
  *[caesar_version][122]*
    the ``version'' library of OPEN/CAESAR

────────────────────────────────
4. Deprecated or Deleted Tools  
────────────────────────────────


* *[aldebaran][123]*
    minimization and comparison of labelled transitions systems
    *deprecated in 2008 - see HISTORY file entries #1299, #1742, and #1827*
  *caesar.aldebaran*
    combination of CAESAR and ALDEBARAN
    *deleted in 2002 - see HISTORY file entry #826*
  *caesar.open*
    use [lotos.open][124] instead - OPEN/CAESAR connection for the LOTOS language
    *deprecated in 2020 - see HISTORY file entry #2569*
  *des2aut*
    composition generation of a labelled transitions system
    *deleted in 2002 - see HISTORY file entry #783*
  *exp2fc2*
    conversion of .exp files to parallel .fc2 files
    *deleted in 2014 - see HISTORY file entry #1844*
  *fc2open*
    connection of FC2 format to OPEN/CAESAR
    *deleted in 2002 - see HISTORY file entry #819*
  *lpp *
    LNT pre-processor - auxiliary tool for LNT2LOTOS
    *deleted in 2024 - see HISTORY file entry #3070*
  *lnt_check*
    auxiliary script for LNT2LOTOS
    *deleted in 2024 - see HISTORY file entry #2986*
  *lnt_depend*
    auxiliary script for LNT2LOTOS
    *deleted in 2024 - see HISTORY file entry #3062*


[ Back to the CADP Home Page ][125]

[1]: /
[2]: /
[3]: /tools.html
[4]: /status.html
[5]: /changes.html
[6]: https://cadp.inria.fr/registration/
[7]: /usage/
[8]: /faq.html
[9]: /tutorial/
[10]: /publications/
[11]: /man/
[12]: /demos.html
[13]: http://en.wikipedia.org/wiki/CADP
[14]: /news/
[15]: /training/
[16]: /case-studies/
[17]: /software/
[18]: /resources/vlpn
[19]: /resources/vlsat
[20]: /resources/vlts
[21]: /resources/
[22]: /contributors.html
[23]: http://cadp.inria.fr
[24]: aut.html
[25]: bcg.html
[26]: bes.html
[27]: exp.html
[28]: fsp2lotos.html
[29]: fsp.open.html
[30]: gcf.html
[31]: caesar_hide_1.html
[32]: http://cadp.inria.fr/tutorial
[33]: lotos.html
[34]: http://cadp.inria.fr/tutorial
[35]: mcl.html
[36]: mcl3.html
[37]: mcl4.html
[38]: mcl5.html
[39]: nupn.html
[40]: pbg.html
[41]: rbc.html
[42]: caesar_rename_1.html
[43]: seq.html
[44]: svl-lang.html
[45]: xtl-lang.html
[46]: bcg_cmp.html
[47]: bcg_draw.html
[48]: bcg_edit.html
[49]: bcg_graph.html
[50]: bcg_info.html
[51]: bcg_io.html
[52]: bcg_labels.html
[53]: bcg_lib.html
[54]: bcg_merge.html
[55]: bcg_min.html
[56]: bcg_open.html
[57]: bcg_steady.html
[58]: bcg_transient.html
[59]: bes_solve.html
[60]: bisimulator.html
[61]: caesar.adt.html
[62]: caesar.bdd.html
[63]: caesar.html
[64]: caesar.indent.html
[65]: contributor.html
[66]: cunctator.html
[67]: declarator.html
[68]: determinator.html
[69]: distributor.html
[70]: evaluator.html
[71]: evaluator3.html
[72]: evaluator4.html
[73]: evaluator5.html
[74]: executor.html
[75]: exhibitor.html
[76]: exp.open.html
[77]: fsp.open.html
[78]: fsp2lotos.html
[79]: generator.html
[80]: installator.html
[81]: lnt.open.html
[82]: lnt2lotos.html
[83]: lnt_merge.html
[84]: lotos.open.html
[85]: nupn_info.html
[86]: ocis.html
[87]: pbg_cp.html
[88]: pbg_info.html
[89]: pbg_mv.html
[90]: pbg_rm.html
[91]: predictor.html
[92]: projector.html
[93]: reductor.html
[94]: scrutator.html
[95]: seq.open.html
[96]: simulator.html
[97]: svl.html
[98]: terminator.html
[99]: tgv.html
[100]: traian.html
[101]: tst.html
[102]: xeuca.html
[103]: xsimulator.html
[104]: xtl.html
[105]: bcg_read.html
[106]: bcg_write.html
[107]: caesar_area_1.html
[108]: caesar_bitmap.html
[109]: caesar_cache_1.html
[110]: caesar_diagnostic_1.html
[111]: caesar_edge.html
[112]: caesar_graph.html
[113]: caesar_hash.html
[114]: caesar_hide_1.html
[115]: caesar_mask_1.html
[116]: caesar_rename_1.html
[117]: caesar_solve_1.html
[118]: caesar_solve_2.html
[119]: caesar_stack_1.html
[120]: caesar_standard.html
[121]: caesar_table_1.html
[122]: caesar_version.html
[123]: aldebaran.html
[124]: lotos.open.html
[125]: http://cadp.inria.fr
