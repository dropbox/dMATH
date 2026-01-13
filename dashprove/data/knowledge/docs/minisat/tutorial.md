[MiniSat logo]by Niklas Eén, Niklas Sörensson
[Main][1] [MiniSat][2] [MiniSat+][3] [SatELite][4] [Papers][5] [Authors][6] [Links][7]

# MiniSat

MiniSat started out 2003 as an effort to help people get into the SAT community by providing a
small, yet efficient, SAT solver with good documentation (through the following [paper][8]). The
first version was just above 600 lines, not counting comments and empty lines, while still
containing all the features of the current state-of-the-art solvers in 2003 (conflict-clause
recording, conflict-driven backjumping, VSIDS dynamic variable order, two-literal watch scheme), and
even extensions for incremental SAT and for non-clausal constraints over boolean variables.

In later versions, the code base has grown a bit to emcompass recent improvements, but is still
quite small and hopefully readable. In the SAT competition 2005, version 1.13 proved that MiniSat
still is state-of-the-art; at least for publically available solvers.

Below we provide a set of different versions of MiniSat to suit the needs of different applications.
We encourage you to submit bugfixes, extensions and suggestions for improvements, as well as basing
products on MiniSat. The solver is available under the MIT licence, a strictly freer licence than
the LGPL, basically allowing you to use the code as you like.


────────────────────────────────────────────────────────────────────────────────────────────────────
Source code... 
──┬─────────┬──┬
  │[minisat-│- │The first public release after a period of inactivity. Similar in performance to the
  │2.2.0.tar│  │version that won in [SAT-Race 2008][10] but with some clean-ups and minor feature   
  │.gz][9]  │  │additions. For more info, see the [release notes][11].                              
──┼─────────┼──┼────────────────────────────────────────────────────────────────────────────────────
  │[minisat2│- │This is the first release of MiniSat 2, featuring variable elimination style        
  │-070721.z│  │simplification natively. It is a cleaned up version of the winning entry of         
  │ip][12]  │  │[SAT-Race 2006][13] and is intended to subsume SatELite and SatELiteGTI.            
  │         │  │Documentation is scarce at the moment, but feel free to send questions to the       
  │         │  │mailinglist ([minisat@googlegroups.com][14]).                                       
──┼─────────┼──┼────────────────────────────────────────────────────────────────────────────────────
  │[MiniSat_│- │The cleaned-up verison of "v1.13" with the last minute hacks properly implemented   
  │v1.14_src│  │(e.g. the hack in "simplifyDB()"), and the worst code abuse removed. Names and      
  │.zip     │  │comments have also been improved. It performs just as good as the 1.13, but due to  
  │][15]    │  │the small changes, it may perform differently on individual problems.               
──┼─────────┼──┼────────────────────────────────────────────────────────────────────────────────────
  │[MiniSat-│- │Proof-logging version. This is a new feature, but there is a risk that it works. To 
  │p_v1.14.s│  │simplify the code a bit, the binary-clause trick in the non-proof-logging version   
  │rc.zip   │  │has been removed, making it a constant 10-20% slower. But if you need proof-logging,
  │][16]    │  │here it is.                                                                         
──┼─────────┼──┼────────────────────────────────────────────────────────────────────────────────────
  │[MiniSat-│- │If you despise C++, here is an experimental C version by Niklas Sörensson           
  │C_v1.14.1│  │                                                                                    
  │.src.zip │  │                                                                                    
  │][17]    │  │                                                                                    
──┼─────────┼──┼────────────────────────────────────────────────────────────────────────────────────
  │[MiniSat_│- │This version still supports non-clausal constraints. The new improved variable order
  │v1.12b_sr│  │from version 1.13 is backported to this "b"-version of 1.12.                        
  │c.zip    │  │                                                                                    
  │][18]    │  │                                                                                    
──┼─────────┼──┼────────────────────────────────────────────────────────────────────────────────────
  │[sat2005_│- │This archive contains the SAT 2005 competition version. It's hacked up and a bit    
  │submissio│  │ugly, but it is fast and it works.                                                  
  │n.zip    │  │                                                                                    
  │][19]    │  │                                                                                    
──┼─────────┼──┼────────────────────────────────────────────────────────────────────────────────────
  │[        │- │A patch for MiniSat v1.14 to compile under MS Visual Studio (based on submission by 
  │MiniSat_v│  │Jean Gressmann).                                                                    
  │1.14_VC-0│  │                                                                                    
  │80206.pat│  │                                                                                    
  │ch ][20] │  │                                                                                    
──┼─────────┼──┼────────────────────────────────────────────────────────────────────────────────────
  │[        │- │A patch to the "friends" declaration of "SolverTypes.h" to compile under GCC 4.1    
  │MiniSat_v│  │(and probably other compilers too) (submitted by Peter Hawkins).                    
  │1.14_gcc4│  │                                                                                    
  │1.patch  │  │                                                                                    
  │][21]    │  │                                                                                    
──┼─────────┼──┼────────────────────────────────────────────────────────────────────────────────────
  │[        │- │Link to Michal Moskal's C# version of MiniSat.                                      
  │MiniSat_v│  │                                                                                    
  │1.14 for │  │                                                                                    
  │C# ][22] │  │                                                                                    
──┼─────────┼──┼────────────────────────────────────────────────────────────────────────────────────
  │[        │- │Link to Flavio Lerda Home Page .                                                    
  │MiniSat_v│  │                                                                                    
  │1.14     │  │                                                                                    
  │wrapper  │  │                                                                                    
  │for OCAML│  │                                                                                    
  │][23]    │  │                                                                                    
──┴─────────┴──┴────────────────────────────────────────────────────────────────────────────────────
Pre-compiled   
binaries...    
──┬─────────┬──┬
  │[MiniSat_│- │Statically linked Linux binary.                                                     
  │v1.14_lin│  │                                                                                    
  │ux ][24] │  │                                                                                    
──┼─────────┼──┼────────────────────────────────────────────────────────────────────────────────────
  │[MiniSat_│- │Cygwin/Windows binary.                                                              
  │v1.14_cyg│  │                                                                                    
  │win ][25]│  │                                                                                    
──┴─────────┴──┴────────────────────────────────────────────────────────────────────────────────────

[1]: Main.html
[2]: MiniSat.html
[3]: MiniSat+.html
[4]: SatELite.html
[5]: Papers.html
[6]: Authors.html
[7]: Links.html
[8]: downloads/MiniSat.pdf
[9]: downloads/minisat-2.2.0.tar.gz
[10]: http://baldur.iti.uka.de/sat-race-2008/
[11]: downloads/ReleaseNotes-2.2.0.txt
[12]: downloads/minisat2-070721.zip
[13]: http://fmv.jku.at/sat-race-2006/
[14]: mailto:minisat@googlegroups.com
[15]: downloads/MiniSat_v1.14.2006-Aug-29.src.zip
[16]: downloads/MiniSat-p_v1.14.2006-Sep-07.src.zip
[17]: downloads/MiniSat-C_v1.14.1.src.zip
[18]: downloads/MiniSat_v1.12b_src.zip
[19]: downloads/sat2005_submission.zip
[20]: downloads/MiniSat_v1.14_VC-080206.patch
[21]: downloads/MiniSat_v1.14_gcc41.patch
[22]: http://nemerle.org/svn.fx7/trunk/minisatcs/
[23]:  http://www.cs.cmu.edu/~flerda/programming/
[24]: downloads/MiniSat_v1.14_linux
[25]: downloads/MiniSat_v1.14_cygwin
