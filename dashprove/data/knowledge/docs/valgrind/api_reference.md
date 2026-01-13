─────────────────────────────────────────────────┬──────────────────────────────────────────────────
[[Valgrind Home] ][1]                            │[Information][2] [Source Code][3] Documentation   
                                                 │[Contact][4] [How to Help][5] [Gallery][6]        
                                                 │| [Table of Contents][7] | [Quick Start][8] |     
                                                 │[FAQ][9] | [User Manual][10] | [Download          
                                                 │Manual][11] | [Research Papers][12] | [Books][13] 
                                                 │|                                                 
─────────────────────────────────────────────────┴──────────────────────────────────────────────────
Valgrind User Manual                                                                                
                                                                                                    
───────────┬─────────┬─────────┬──────────────────────┬─────────────────────────────────────────────
[[Prev]][14│[[Up]][15│[[Up]][16│Valgrind Documentation│[[Next]][17                                  
]          │]        │]        │                      │]                                            
───────────┴─────────┴─────────┴──────────────────────┴─────────────────────────────────────────────
                                                                                                    
# Valgrind User Manual                                                                              
                                                                                                    
Release 3.26.0 24 Oct 2025                                                                          
                                                                                                    
Copyright © 2000-2025 [Valgrind Developers][18]                                                     
                                                                                                    
Email: [valgrind@valgrind.org][19]                                                                  
                                                                                                    
Table of Contents                                                                                   
                                                                                                    
*[1. Introduction][20]*                                                                             
  *[1.1. An Overview of Valgrind][21]*                                                              
  *[1.2. How to navigate this manual][22]*                                                          
*[2. Using and understanding the Valgrind core][23]*                                                
  *[2.1. What Valgrind does with your program][24]*                                                 
  *[2.2. Getting started][25]*                                                                      
  *[2.3. The Commentary][26]*                                                                       
  *[2.4. Reporting of errors][27]*                                                                  
  *[2.5. Suppressing errors][28]*                                                                   
  *[2.6. Debuginfod][29]*                                                                           
  *[2.7. Core Command-line Options][30]*                                                            
    *[2.7.1. Tool-selection Option][31]*                                                            
    *[2.7.2. Basic Options][32]*                                                                    
    *[2.7.3. Error-related Options][33]*                                                            
    *[2.7.4. malloc-related Options][34]*                                                           
    *[2.7.5. Uncommon Options][35]*                                                                 
    *[2.7.6. Debugging Options][36]*                                                                
    *[2.7.7. Setting Default Options][37]*                                                          
    *[2.7.8. Dynamically Changing Options][38]*                                                     
  *[2.8. Support for Threads][39]*                                                                  
    *[2.8.1. Scheduling and Multi-Thread Performance][40]*                                          
  *[2.9. Handling of Signals][41]*                                                                  
  *[2.10. Execution Trees][42]*                                                                     
  *[2.11. Building and Installing Valgrind][43]*                                                    
  *[2.12. If You Have Problems][44]*                                                                
  *[2.13. Limitations][45]*                                                                         
  *[2.14. An Example Run][46]*                                                                      
  *[2.15. Warning Messages You Might See][47]*                                                      
*[3. Using and understanding the Valgrind core: Advanced Topics][48]*                               
  *[3.1. The Client Request mechanism][49]*                                                         
  *[3.2. Debugging your program using Valgrind gdbserver and GDB][50]*                              
    *[3.2.1. Quick Start: debugging in 3 steps][51]*                                                
    *[3.2.2. Valgrind gdbserver overall organisation][52]*                                          
    *[3.2.3. Connecting GDB to a Valgrind gdbserver][53]*                                           
    *[3.2.4. Connecting to an Android gdbserver][54]*                                               
    *[3.2.5. Monitor command handling by the Valgrind gdbserver][55]*                               
    *[3.2.6. GDB front end commands for Valgrind gdbserver monitor commands][56]*                   
    *[3.2.7. Valgrind gdbserver thread information][57]*                                            
    *[3.2.8. Examining and modifying Valgrind shadow registers][58]*                                
    *[3.2.9. Limitations of the Valgrind gdbserver][59]*                                            
    *[3.2.10. vgdb command line options][60]*                                                       
    *[3.2.11. Valgrind monitor commands][61]*                                                       
  *[3.3. Function wrapping][62]*                                                                    
    *[3.3.1. A Simple Example][63]*                                                                 
    *[3.3.2. Wrapping Specifications][64]*                                                          
    *[3.3.3. Wrapping Semantics][65]*                                                               
    *[3.3.4. Debugging][66]*                                                                        
    *[3.3.5. Limitations - control flow][67]*                                                       
    *[3.3.6. Limitations - original function signatures][68]*                                       
    *[3.3.7. Examples][69]*                                                                         
*[4. Memcheck: a memory error detector][70]*                                                        
  *[4.1. Overview][71]*                                                                             
  *[4.2. Explanation of error messages from Memcheck][72]*                                          
    *[4.2.1. Illegal read / Illegal write errors][73]*                                              
    *[4.2.2. Use of uninitialised values][74]*                                                      
    *[4.2.3. Use of uninitialised or unaddressable values in system calls][75]*                     
    *[4.2.4. Illegal frees][76]*                                                                    
    *[4.2.5. When a heap block is freed with an inappropriate deallocation function][77]*           
    *[4.2.6. Overlapping source and destination blocks][78]*                                        
    *[4.2.7. Fishy argument values][79]*                                                            
    *[4.2.8. Realloc size zero][80]*                                                                
    *[4.2.9. Alignment Errors][81]*                                                                 
    *[4.2.10. Memory leak detection][82]*                                                           
  *[4.3. Memcheck Command-Line Options][83]*                                                        
  *[4.4. Writing suppression files][84]*                                                            
  *[4.5. Details of Memcheck's checking machinery][85]*                                             
    *[4.5.1. Valid-value (V) bits][86]*                                                             
    *[4.5.2. Valid-address (A) bits][87]*                                                           
    *[4.5.3. Putting it all together][88]*                                                          
  *[4.6. Memcheck Monitor Commands][89]*                                                            
  *[4.7. Client Requests][90]*                                                                      
  *[4.8. Memory Pools: describing and working with custom allocators][91]*                          
  *[4.9. Debugging MPI Parallel Programs with Valgrind][92]*                                        
    *[4.9.1. Building and installing the wrappers][93]*                                             
    *[4.9.2. Getting started][94]*                                                                  
    *[4.9.3. Controlling the wrapper library][95]*                                                  
    *[4.9.4. Functions][96]*                                                                        
    *[4.9.5. Types][97]*                                                                            
    *[4.9.6. Writing new wrappers][98]*                                                             
    *[4.9.7. What to expect when using the wrappers][99]*                                           
*[5. Cachegrind: a high-precision tracing profiler][100]*                                           
  *[5.1. Overview][101]*                                                                            
  *[5.2. Using Cachegrind and cg_annotate][102]*                                                    
    *[5.2.1. Running Cachegrind][103]*                                                              
    *[5.2.2. Output File][104]*                                                                     
    *[5.2.3. Running cg_annotate][105]*                                                             
    *[5.2.4. The Metadata Section][106]*                                                            
    *[5.2.5. Global, File, and Function-level Counts][107]*                                         
    *[5.2.6. Per-line Counts][108]*                                                                 
    *[5.2.7. Forking Programs][109]*                                                                
    *[5.2.8. cg_annotate Warnings][110]*                                                            
    *[5.2.9. Merging Cachegrind Output Files][111]*                                                 
    *[5.2.10. Differencing Cachegrind output files][112]*                                           
    *[5.2.11. Cache and Branch Simulation][113]*                                                    
  *[5.3. Cachegrind Command-line Options][114]*                                                     
  *[5.4. cg_annotate Command-line Options][115]*                                                    
  *[5.5. cg_merge Command-line Options][116]*                                                       
  *[5.6. cg_diff Command-line Options][117]*                                                        
  *[5.7. Cachegrind Client Requests][118]*                                                          
  *[5.8. Simulation Details][119]*                                                                  
    *[5.8.1. Cache Simulation Specifics][120]*                                                      
    *[5.8.2. Branch Simulation Specifics][121]*                                                     
    *[5.8.3. Accuracy][122]*                                                                        
  *[5.9. Implementation Details][123]*                                                              
    *[5.9.1. How Cachegrind Works][124]*                                                            
    *[5.9.2. Cachegrind Output File Format][125]*                                                   
*[6. Callgrind: a call-graph generating cache and branch prediction profiler][126]*                 
  *[6.1. Overview][127]*                                                                            
    *[6.1.1. Functionality][128]*                                                                   
    *[6.1.2. Basic Usage][129]*                                                                     
  *[6.2. Advanced Usage][130]*                                                                      
    *[6.2.1. Multiple profiling dumps from one program run][131]*                                   
    *[6.2.2. Limiting the range of collected events][132]*                                          
    *[6.2.3. Counting global bus events][133]*                                                      
    *[6.2.4. Avoiding cycles][134]*                                                                 
    *[6.2.5. Forking Programs][135]*                                                                
  *[6.3. Callgrind Command-line Options][136]*                                                      
    *[6.3.1. Dump creation options][137]*                                                           
    *[6.3.2. Activity options][138]*                                                                
    *[6.3.3. Data collection options][139]*                                                         
    *[6.3.4. Cost entity separation options][140]*                                                  
    *[6.3.5. Simulation options][141]*                                                              
    *[6.3.6. Cache simulation options][142]*                                                        
  *[6.4. Callgrind Monitor Commands][143]*                                                          
  *[6.5. Callgrind specific client requests][144]*                                                  
  *[6.6. callgrind_annotate Command-line Options][145]*                                             
  *[6.7. callgrind_control Command-line Options][146]*                                              
*[7. Helgrind: a thread error detector][147]*                                                       
  *[7.1. Overview][148]*                                                                            
  *[7.2. Detected errors: Misuses of the POSIX pthreads API][149]*                                  
  *[7.3. Detected errors: Inconsistent Lock Orderings][150]*                                        
  *[7.4. Detected errors: Data Races][151]*                                                         
    *[7.4.1. A Simple Data Race][152]*                                                              
    *[7.4.2. Helgrind's Race Detection Algorithm][153]*                                             
    *[7.4.3. Interpreting Race Error Messages][154]*                                                
  *[7.5. Hints and Tips for Effective Use of Helgrind][155]*                                        
  *[7.6. Helgrind Command-line Options][156]*                                                       
  *[7.7. Helgrind Monitor Commands][157]*                                                           
  *[7.8. Helgrind Client Requests][158]*                                                            
  *[7.9. A To-Do List for Helgrind][159]*                                                           
*[8. DRD: a thread error detector][160]*                                                            
  *[8.1. Overview][161]*                                                                            
    *[8.1.1. Multithreaded Programming Paradigms][162]*                                             
    *[8.1.2. POSIX Threads Programming Model][163]*                                                 
    *[8.1.3. Multithreaded Programming Problems][164]*                                              
    *[8.1.4. Data Race Detection][165]*                                                             
  *[8.2. Using DRD][166]*                                                                           
    *[8.2.1. DRD Command-line Options][167]*                                                        
    *[8.2.2. Detected Errors: Data Races][168]*                                                     
    *[8.2.3. Detected Errors: Lock Contention][169]*                                                
    *[8.2.4. Detected Errors: Misuse of the POSIX threads API][170]*                                
    *[8.2.5. Client Requests][171]*                                                                 
    *[8.2.6. Debugging C++11 Programs][172]*                                                        
    *[8.2.7. Debugging GNOME Programs][173]*                                                        
    *[8.2.8. Debugging Boost.Thread Programs][174]*                                                 
    *[8.2.9. Debugging OpenMP Programs][175]*                                                       
    *[8.2.10. DRD and Custom Memory Allocators][176]*                                               
    *[8.2.11. DRD Versus Memcheck][177]*                                                            
    *[8.2.12. Resource Requirements][178]*                                                          
    *[8.2.13. Hints and Tips for Effective Use of DRD][179]*                                        
  *[8.3. Using the POSIX Threads API Effectively][180]*                                             
    *[8.3.1. Mutex types][181]*                                                                     
    *[8.3.2. Condition variables][182]*                                                             
    *[8.3.3. pthread_cond_timedwait and timeouts][183]*                                             
  *[8.4. Limitations][184]*                                                                         
  *[8.5. Feedback][185]*                                                                            
*[9. Massif: a heap profiler][186]*                                                                 
  *[9.1. Overview][187]*                                                                            
  *[9.2. Using Massif and ms_print][188]*                                                           
    *[9.2.1. An Example Program][189]*                                                              
    *[9.2.2. Running Massif][190]*                                                                  
    *[9.2.3. Running ms_print][191]*                                                                
    *[9.2.4. The Output Preamble][192]*                                                             
    *[9.2.5. The Output Graph][193]*                                                                
    *[9.2.6. The Snapshot Details][194]*                                                            
    *[9.2.7. Forking Programs][195]*                                                                
    *[9.2.8. Measuring All Memory in a Process][196]*                                               
    *[9.2.9. Acting on Massif's Information][197]*                                                  
  *[9.3. Using massif-visualizer][198]*                                                             
  *[9.4. Massif Command-line Options][199]*                                                         
  *[9.5. Massif Monitor Commands][200]*                                                             
  *[9.6. Massif Client Requests][201]*                                                              
  *[9.7. ms_print Command-line Options][202]*                                                       
  *[9.8. Massif's Output File Format][203]*                                                         
*[10. DHAT: a dynamic heap analysis tool][204]*                                                     
  *[10.1. Overview][205]*                                                                           
  *[10.2. Using DHAT][206]*                                                                         
    *[10.2.1. Running DHAT][207]*                                                                   
    *[10.2.2. Output File][208]*                                                                    
  *[10.3. DHAT's Viewer][209]*                                                                      
    *[10.3.1. The Output Header][210]*                                                              
    *[10.3.2. The PP Tree][211]*                                                                    
    *[10.3.3. The Output Footer][212]*                                                              
    *[10.3.4. Sort Metrics][213]*                                                                   
  *[10.4. Treatment of realloc][214]*                                                               
  *[10.5. Copy profiling][215]*                                                                     
  *[10.6. Ad hoc profiling][216]*                                                                   
  *[10.7. DHAT Command-line Options][217]*                                                          
*[11. Lackey: an example tool][218]*                                                                
  *[11.1. Overview][219]*                                                                           
  *[11.2. Lackey Command-line Options][220]*                                                        
*[12. Nulgrind: the minimal Valgrind tool][221]*                                                    
  *[12.1. Overview][222]*                                                                           
*[13. BBV: an experimental basic block vector generation tool][223]*                                
  *[13.1. Overview][224]*                                                                           
  *[13.2. Using Basic Block Vectors to create SimPoints][225]*                                      
  *[13.3. BBV Command-line Options][226]*                                                           
  *[13.4. Basic Block Vector File Format][227]*                                                     
  *[13.5. Implementation][228]*                                                                     
  *[13.6. Threaded Executable Support][229]*                                                        
  *[13.7. Validation][230]*                                                                         
  *[13.8. Performance][231]*                                                                        
                                                                                                    
                                                                                                    
────────────────────────────────────────┬───────┬───────────────────────────────────────────────────
[<< The Valgrind Quick Start Guide][232]│[Up][23│[1. Introduction >>][234]                          
                                        │3]     │                                                   
────────────────────────────────────────┴───────┴───────────────────────────────────────────────────
[Home][235]                                                                                         
────────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                                    
[Bad, Bad Bug!]                                                                                     
                                                                                                    
Copyright © 2000-2025 [Valgrind™ Developers][236]                                                   
                                                                                                    
Hosting kindly provided by [sourceware.org][237]                                                    
[*Best Viewed With A(ny) Browser*][238]                                                             
                                                                                                    
────────────────────────────────────────────────────────────────────────────────────────────────────

[1]: /
[2]: /info/
[3]: /downloads/
[4]: /support/
[5]: /help/
[6]: /gallery/
[7]: /docs/manual/index.html
[8]: /docs/manual/QuickStart.html
[9]: /docs/manual/FAQ.html
[10]: /docs/manual/manual.html
[11]: /docs/download_docs.html
[12]: /docs/pubs.html
[13]: /docs/books.html
[14]: quick-start.html
[15]: index.html
[16]: index.html
[17]: manual-intro.html
[18]: http://www.valgrind.org/info/developers.html
[19]: mailto:valgrind@valgrind.org
[20]: manual-intro.html
[21]: manual-intro.html#manual-intro.overview
[22]: manual-intro.html#manual-intro.navigation
[23]: manual-core.html
[24]: manual-core.html#manual-core.whatdoes
[25]: manual-core.html#manual-core.started
[26]: manual-core.html#manual-core.comment
[27]: manual-core.html#manual-core.report
[28]: manual-core.html#manual-core.suppress
[29]: manual-core.html#manual-core.debuginfod
[30]: manual-core.html#manual-core.options
[31]: manual-core.html#manual-core.toolopts
[32]: manual-core.html#manual-core.basicopts
[33]: manual-core.html#manual-core.erropts
[34]: manual-core.html#manual-core.mallocopts
[35]: manual-core.html#manual-core.rareopts
[36]: manual-core.html#manual-core.debugopts
[37]: manual-core.html#manual-core.defopts
[38]: manual-core.html#manual-core.dynopts
[39]: manual-core.html#manual-core.pthreads
[40]: manual-core.html#manual-core.pthreads_perf_sched
[41]: manual-core.html#manual-core.signals
[42]: manual-core.html#manual-core.xtree
[43]: manual-core.html#manual-core.install
[44]: manual-core.html#manual-core.problems
[45]: manual-core.html#manual-core.limits
[46]: manual-core.html#manual-core.example
[47]: manual-core.html#manual-core.warnings
[48]: manual-core-adv.html
[49]: manual-core-adv.html#manual-core-adv.clientreq
[50]: manual-core-adv.html#manual-core-adv.gdbserver
[51]: manual-core-adv.html#manual-core-adv.gdbserver-simple
[52]: manual-core-adv.html#manual-core-adv.gdbserver-concept
[53]: manual-core-adv.html#manual-core-adv.gdbserver-gdb
[54]: manual-core-adv.html#manual-core-adv.gdbserver-gdb-android
[55]: manual-core-adv.html#manual-core-adv.gdbserver-commandhandling
[56]: manual-core-adv.html#manual-core-adv.gdbserver-gdbmonitorfrontend
[57]: manual-core-adv.html#manual-core-adv.gdbserver-threads
[58]: manual-core-adv.html#manual-core-adv.gdbserver-shadowregisters
[59]: manual-core-adv.html#manual-core-adv.gdbserver-limitations
[60]: manual-core-adv.html#manual-core-adv.vgdb
[61]: manual-core-adv.html#manual-core-adv.valgrind-monitor-commands
[62]: manual-core-adv.html#manual-core-adv.wrapping
[63]: manual-core-adv.html#manual-core-adv.wrapping.example
[64]: manual-core-adv.html#manual-core-adv.wrapping.specs
[65]: manual-core-adv.html#manual-core-adv.wrapping.semantics
[66]: manual-core-adv.html#manual-core-adv.wrapping.debugging
[67]: manual-core-adv.html#manual-core-adv.wrapping.limitations-cf
[68]: manual-core-adv.html#manual-core-adv.wrapping.limitations-sigs
[69]: manual-core-adv.html#manual-core-adv.wrapping.examples
[70]: mc-manual.html
[71]: mc-manual.html#mc-manual.overview
[72]: mc-manual.html#mc-manual.errormsgs
[73]: mc-manual.html#mc-manual.badrw
[74]: mc-manual.html#mc-manual.uninitvals
[75]: mc-manual.html#mc-manual.bad-syscall-args
[76]: mc-manual.html#mc-manual.badfrees
[77]: mc-manual.html#mc-manual.rudefn
[78]: mc-manual.html#mc-manual.overlap
[79]: mc-manual.html#mc-manual.fishyvalue
[80]: mc-manual.html#mc-manual.reallocsizezero
[81]: mc-manual.html#mc-manual.alignment
[82]: mc-manual.html#mc-manual.leaks
[83]: mc-manual.html#mc-manual.options
[84]: mc-manual.html#mc-manual.suppfiles
[85]: mc-manual.html#mc-manual.machine
[86]: mc-manual.html#mc-manual.value
[87]: mc-manual.html#mc-manual.vaddress
[88]: mc-manual.html#mc-manual.together
[89]: mc-manual.html#mc-manual.monitor-commands
[90]: mc-manual.html#mc-manual.clientreqs
[91]: mc-manual.html#mc-manual.mempools
[92]: mc-manual.html#mc-manual.mpiwrap
[93]: mc-manual.html#mc-manual.mpiwrap.build
[94]: mc-manual.html#mc-manual.mpiwrap.gettingstarted
[95]: mc-manual.html#mc-manual.mpiwrap.controlling
[96]: mc-manual.html#mc-manual.mpiwrap.limitations.functions
[97]: mc-manual.html#mc-manual.mpiwrap.limitations.types
[98]: mc-manual.html#mc-manual.mpiwrap.writingwrappers
[99]: mc-manual.html#mc-manual.mpiwrap.whattoexpect
[100]: cg-manual.html
[101]: cg-manual.html#cg-manual.overview
[102]: cg-manual.html#cg-manual.profile
[103]: cg-manual.html#cg-manual.running-cachegrind
[104]: cg-manual.html#cg-manual.outputfile
[105]: cg-manual.html#cg-manual.running-cg_annotate
[106]: cg-manual.html#cg-manual.the-metadata
[107]: cg-manual.html#cg-manual.the-global
[108]: cg-manual.html#cg-manual.line-by-line
[109]: cg-manual.html#cg-manual.forkingprograms
[110]: cg-manual.html#cg-manual.annopts.warnings
[111]: cg-manual.html#cg-manual.cg_merge
[112]: cg-manual.html#cg-manual.cg_diff
[113]: cg-manual.html#cg-manual.cache-branch-sim
[114]: cg-manual.html#cg-manual.cgopts
[115]: cg-manual.html#cg-manual.annopts
[116]: cg-manual.html#cg-manual.mergeopts
[117]: cg-manual.html#cg-manual.diffopts
[118]: cg-manual.html#cg-manual.clientrequests
[119]: cg-manual.html#cg-manual.sim-details
[120]: cg-manual.html#cache-sim
[121]: cg-manual.html#branch-sim
[122]: cg-manual.html#cg-manual.annopts.accuracy
[123]: cg-manual.html#cg-manual.impl-details
[124]: cg-manual.html#cg-manual.impl-details.how-cg-works
[125]: cg-manual.html#cg-manual.impl-details.file-format
[126]: cl-manual.html
[127]: cl-manual.html#cl-manual.use
[128]: cl-manual.html#cl-manual.functionality
[129]: cl-manual.html#cl-manual.basics
[130]: cl-manual.html#cl-manual.usage
[131]: cl-manual.html#cl-manual.dumps
[132]: cl-manual.html#cl-manual.limits
[133]: cl-manual.html#cl-manual.busevents
[134]: cl-manual.html#cl-manual.cycles
[135]: cl-manual.html#cl-manual.forkingprograms
[136]: cl-manual.html#cl-manual.options
[137]: cl-manual.html#cl-manual.options.creation
[138]: cl-manual.html#cl-manual.options.activity
[139]: cl-manual.html#cl-manual.options.collection
[140]: cl-manual.html#cl-manual.options.separation
[141]: cl-manual.html#cl-manual.options.simulation
[142]: cl-manual.html#cl-manual.options.cachesimulation
[143]: cl-manual.html#cl-manual.monitor-commands
[144]: cl-manual.html#cl-manual.clientrequests
[145]: cl-manual.html#cl-manual.callgrind_annotate-options
[146]: cl-manual.html#cl-manual.callgrind_control-options
[147]: hg-manual.html
[148]: hg-manual.html#hg-manual.overview
[149]: hg-manual.html#hg-manual.api-checks
[150]: hg-manual.html#hg-manual.lock-orders
[151]: hg-manual.html#hg-manual.data-races
[152]: hg-manual.html#hg-manual.data-races.example
[153]: hg-manual.html#hg-manual.data-races.algorithm
[154]: hg-manual.html#hg-manual.data-races.errmsgs
[155]: hg-manual.html#hg-manual.effective-use
[156]: hg-manual.html#hg-manual.options
[157]: hg-manual.html#hg-manual.monitor-commands
[158]: hg-manual.html#hg-manual.client-requests
[159]: hg-manual.html#hg-manual.todolist
[160]: drd-manual.html
[161]: drd-manual.html#drd-manual.overview
[162]: drd-manual.html#drd-manual.mt-progr-models
[163]: drd-manual.html#drd-manual.pthreads-model
[164]: drd-manual.html#drd-manual.mt-problems
[165]: drd-manual.html#drd-manual.data-race-detection
[166]: drd-manual.html#drd-manual.using-drd
[167]: drd-manual.html#drd-manual.options
[168]: drd-manual.html#drd-manual.data-races
[169]: drd-manual.html#drd-manual.lock-contention
[170]: drd-manual.html#drd-manual.api-checks
[171]: drd-manual.html#drd-manual.clientreqs
[172]: drd-manual.html#drd-manual.CXX11
[173]: drd-manual.html#drd-manual.gnome
[174]: drd-manual.html#drd-manual.boost.thread
[175]: drd-manual.html#drd-manual.openmp
[176]: drd-manual.html#drd-manual.cust-mem-alloc
[177]: drd-manual.html#drd-manual.drd-versus-memcheck
[178]: drd-manual.html#drd-manual.resource-requirements
[179]: drd-manual.html#drd-manual.effective-use
[180]: drd-manual.html#drd-manual.Pthreads
[181]: drd-manual.html#drd-manual.mutex-types
[182]: drd-manual.html#drd-manual.condvar
[183]: drd-manual.html#drd-manual.pctw
[184]: drd-manual.html#drd-manual.limitations
[185]: drd-manual.html#drd-manual.feedback
[186]: ms-manual.html
[187]: ms-manual.html#ms-manual.overview
[188]: ms-manual.html#ms-manual.using-print
[189]: ms-manual.html#ms-manual.anexample
[190]: ms-manual.html#ms-manual.running-massif
[191]: ms-manual.html#ms-manual.running-ms_print
[192]: ms-manual.html#ms-manual.theoutputpreamble
[193]: ms-manual.html#ms-manual.theoutputgraph
[194]: ms-manual.html#ms-manual.thesnapshotdetails
[195]: ms-manual.html#ms-manual.forkingprograms
[196]: ms-manual.html#ms-manual.not-measured
[197]: ms-manual.html#ms-manual.acting
[198]: ms-manual.html#ms-manual.using-visualizer
[199]: ms-manual.html#ms-manual.options
[200]: ms-manual.html#ms-manual.monitor-commands
[201]: ms-manual.html#ms-manual.clientreqs
[202]: ms-manual.html#ms-manual.ms_print-options
[203]: ms-manual.html#ms-manual.fileformat
[204]: dh-manual.html
[205]: dh-manual.html#dh-manual.overview
[206]: dh-manual.html#dh-manual.profile
[207]: dh-manual.html#dh-manual.running-DHAT
[208]: dh-manual.html#dh-manual.outputfile
[209]: dh-manual.html#dh-manual.viewer
[210]: dh-manual.html#dh-output-header
[211]: dh-manual.html#dh-ap-tree
[212]: dh-manual.html#dh-output-footer
[213]: dh-manual.html#dh-sort-metrics
[214]: dh-manual.html#dh-manual.realloc
[215]: dh-manual.html#dh-manual.copy-profiling
[216]: dh-manual.html#dh-manual.ad-hoc-profiling
[217]: dh-manual.html#dh-manual.options
[218]: lk-manual.html
[219]: lk-manual.html#lk-manual.overview
[220]: lk-manual.html#lk-manual.options
[221]: nl-manual.html
[222]: nl-manual.html#nl-manual.overview
[223]: bbv-manual.html
[224]: bbv-manual.html#bbv-manual.overview
[225]: bbv-manual.html#bbv-manual.quickstart
[226]: bbv-manual.html#bbv-manual.usage
[227]: bbv-manual.html#bbv-manual.fileformat
[228]: bbv-manual.html#bbv-manual.implementation
[229]: bbv-manual.html#bbv-manual.threadsupport
[230]: bbv-manual.html#bbv-manual.validation
[231]: bbv-manual.html#bbv-manual.performance
[232]: quick-start.html
[233]: index.html
[234]: manual-intro.html
[235]: index.html
[236]: /info/developers.html
[237]: https://sourceware.org/
[238]: http://www.anybrowser.org/campaign/
