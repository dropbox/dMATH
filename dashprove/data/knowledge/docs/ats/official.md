─────┬──────────────────────────────────────────────────────────────────────────────────────────────
[ATSl│The ATS Programming Language                                                                  
ogo] │Unleashing the Potentials of Types and Templates!                                             
─────│                                                                                              
Home(│──────────────────────────────────────────────────────────────────────────────────────────┬───
[old]│* Home                                                                                    │   
[1]) │* ───────────────────────────────                                                         │   
─────│  [What is ATS?][14]                                                                      │   
[Down│  ───────────────────────────────                                                         │   
loads│  [What is ATS good for?][15]                                                             │   
][2] │  ───────────────────────────────                                                         │   
─────│  [Suggestion on learning                                                                 │   
[Docu│  ATS][16]                                                                                │   
ments│  ───────────────────────────────                                                         │   
][3] │  [Acknowledgments][17]                                                                   │   
─────│  ───────────────────────────────                                                         │   
[Libr│* [Downloads][18]                                                                         │   
aries│* ────────────────────────────────────────────                                            │   
][4] │  [ATS packages][19]                                                                      │   
─────│  ────────────────────────────────────────────                                            │   
[Comm│  [Uninstallation for ATS][20]                                                            │   
unity│  ────────────────────────────────────────────                                            │   
][5] │  [Requirements for installation][21]                                                     │   
─────│  ────────────────────────────────────────────                                            │   
[Pape│  [Precompiled packages for installation][22]                                             │   
rs][6│  ────────────────────────────────────────────                                            │   
]    │  [Installation through source                                                            │   
─────│  compilation][23]                                                                        │   
[Exam│  ────────────────────────────────────────────                                            │   
ples]│  [Installation of ATS2-contrib][24]                                                      │   
[7]  │  ────────────────────────────────────────────                                            │   
─────│  [Installation of ATS2-include][25]                                                      │   
[Reso│  ────────────────────────────────────────────                                            │   
urces│  [Scripts for installing ATS-Postiats][26]                                               │   
][8] │  ────────────────────────────────────────────                                            │   
─────│* [Documents][27]                                                                         │   
[Impl│* ──────────────────────────────────────────────────                                      │   
ement│  [Introduction to Programming in ATS][28]                                                │   
s][9]│  ──────────────────────────────────────────────────                                      │   
─────│  [A Tutorial on Programming Features in ATS][29]                                         │   
[Mail│  ──────────────────────────────────────────────────                                      │   
ing-l│  [A Crash into Functional Programming via ATS][30]                                       │   
ist][│  ──────────────────────────────────────────────────                                      │   
10]  │  [Effective Programming in ATS through                                                   │   
─────│  Examples][31]                                                                           │   
[ats-│  ──────────────────────────────────────────────────                                      │   
lang-│* [Libraries][32]                                                                         │   
users│* [Community][33]                                                                         │   
][11]│* ────────────────────────────────────────                                                │   
─────│  [Wiki for ATS users][34]                                                                │   
[ats-│  ────────────────────────────────────────                                                │   
lang-│  [ATS news links at reddit][35]                                                          │   
devel│  ────────────────────────────────────────                                                │   
][12]│  [IRC channel for ATS users][36]                                                         │   
─────│  ────────────────────────────────────────                                                │   
[Try │  [StackOverflow tag for ATS][37]                                                         │   
ATS  │  ────────────────────────────────────────                                                │   
on-li│  [Q&A forum for ATS users][38]                                                           │   
ne][1│  ────────────────────────────────────────                                                │   
3]   │  [Discussion forum for ATS                                                               │   
     │  developers][39]                                                                         │   
     │  ────────────────────────────────────────                                                │   
     │  [Mailing-list for ATS users][40]                                                        │   
     │  ────────────────────────────────────────                                                │   
     │  [JATS-UG: Japan ATS User Group][41]                                                     │   
     │  ────────────────────────────────────────                                                │   
     │──────────────────────────────────────────────────────────────────────────────────────────┴───
     │                                                                                              
     │───────────────────────────────────────────────────────┬──────────────────────────────────────
     │# About ATS                                            │## Yes, ATS can!                      
     │                                                       │                                      
     │* [What is ATS?][42]                                   │What is new in the community? GO      
     │* [What is ATS good for?][43]                          │                                      
     │* [Suggestion on learning ATS][44]                     │Would you like to try ATS on-line? OK 
     │* [Acknowledgments][45]                                │                                      
     │                                                       │The core of ATS is a typed            
     │## What is ATS?                                        │call-by-value functional programming  
     │                                                       │language that is largely inspired by  
     │ATS is a statically typed programming language that    │ML. For instance, the following tiny  
     │unifies implementation with formal specification. It is│ATS program is written in a style of  
     │equipped with a highly expressive type system rooted in│functional programming:               
     │the framework *Applied Type System*, which gives the   │                                      
     │language its name. In particular, both dependent types │──────────────────────────────────────
     │and linear types are available in ATS.                 │// // Yes, you can edit // (* Say     
     │                                                       │Hello! once *) val () =               
     │The current implementation of ATS2 (ATS/Postiats) is   │print"Hello!\n" // (* Say Hello! 3    
     │written in ATS1 (ATS/Anairiats), consisting of more    │times *) val () =                     
     │than 180K lines of code. ATS can be as efficient as    │3*delay(print"Hello!") val () =       
     │C/C++ both time-wise and memory-wise and supports a    │print_newline((*void*)) //            
     │variety of programming paradigms that include:         │──────────────────────────────────────
     │                                                       │Try-it-yourself                       
     │* **Functional programming**. The core of ATS is a     │──────────────────────────────────────
     │  call-by-value functional language inspired by ML. The│                                      
     │  availability of linear types in ATS often makes      │ATS is both accurate and expressive in
     │  functional programs written in it run not only with  │its support for (static) typechecking.
     │  surprisingly high efficiency (when compared to C) but│The following code demonstrates the   
     │  also with surprisingly small (memory) footprint (when│ability of ATS in detecting           
     │  compared to C as well).                              │out-of-bounds subscripting at         
     │* **Imperative programming**. The novel approach to    │compile-time:                         
     │  imperative programming in ATS is firmly rooted in the│                                      
     │  paradigm of *programming with theorem-proving*. The  │──────────────────────────────────────
     │  type system of ATS allows many features considered   │// // Yes, you can edit // (* Build   
     │  dangerous in other languages (such as explicit       │a list of 3 *) val xs =               
     │  pointer arithmetic and explicit memory               │$list{int}(0, 1, 2) // val x0 =       
     │  allocation/deallocation) to be safely supported in   │xs[0] // legal val x1 = xs[1] //      
     │  ATS, making ATS well-suited for implementing         │legal val x2 = xs[2] // legal val x3  
     │  high-quality low-level systems.                      │= xs[3] // illegal //                 
     │* **Concurrent programming**. ATS can support          │──────────────────────────────────────
     │  multithreaded programming through safe use of        │Try-it-yourself                       
     │  pthreads. The availability of linear types for       │──────────────────────────────────────
     │  tracking and safely manipulating resources provides  │                                      
     │  an effective approach to constructing reliable       │ATS is highly effective and flexible  
     │  programs that can take great advantage of multicore  │in its support for a template-based   
     │  architectures.                                       │approach to code reuse. As an example,
     │* **Modular programming**. The module system of ATS is │the following code is likely to remind
     │  largely infuenced by that of Modula-3, which is both │someone of higher-order functions but 
     │  simple and general as well as effective in supporting│it is actually every bit of a         
     │  large scale programming.                             │first-order implementation in ATS:    
     │                                                       │                                      
     │In addition, ATS contains a subsystem ATS/LF that      │──────────────────────────────────────
     │supports a form of (interactive) theorem-proving, where│// // Yes, you can edit // extern     
     │proofs are constructed as total functions. With this   │fun{} f0 (): int extern fun{} f1      
     │subsystem, ATS is able to advocate a                   │(int): int extern fun{} repeat_f0f1   
     │*programmer-centric* approach to program verification  │(int): int // implement {}(*tmp*)     
     │that combines programming with theorem-proving in a    │repeat_f0f1(n) = if n = 0 then f0()   
     │syntactically intertwined manner. Furthermore, ATS/LF  │else f1(repeat_f0f1(n-1)) // end of   
     │can also serve as a logical framework (LF) for encoding│[if] // fun times ( m:int, n:int ) :  
     │various formal systems (such as logic systems and type │int = // m*n repeat_f0f1 (m) where {  
     │systems) together with proofs of their                 │implement f0<> () = 0 implement f1<>  
     │(meta-)properties.                                     │(x) = x + n } // fun power ( m:int,   
     │                                                       │n:int ) : int = // m^n repeat_f0f1    
     │## What is ATS good for?                               │(n) where { implement f0<> () = 1     
     │                                                       │implement f1<> (x) = m * x } // val   
     │* ATS can greatly enforce precision in practical       │() = println! ("5*5 = ", times(5,5))  
     │  programming.                                         │val () = println! ("5^2 = ",          
     │* ATS can greatly facilitate refinement-based software │power(5,2)) val () = println! ("2^10  
     │  development.                                         │= ", power(2,10)) val () = println!   
     │* ATS allows the programmer to write efficient         │("3^10 = ", power(3,10)) //           
     │  functional programs that directly manipulate native  │──────────────────────────────────────
     │  unboxed data representation.                         │Try-it-yourself                       
     │* ATS allows the programmer to reduce the memory       │──────────────────────────────────────
     │  footprint of a program by making use of linear types.│                                      
     │* ATS allows the programmer to enhance the safety (and │With a functional core of ML-style and
     │  efficiency) of a program by making use of            │certain ad-hoc support for overloading
     │  theorem-proving.                                     │(of function symbols), ATS can readily
     │* ATS allows the programmer to write safe low-level    │accommodate a typical combinator-based
     │  code that runs in OS kernels.                        │style of coding that is often         
     │* ATS can help teach type theory, demonstrating both   │considered a prominent signature of   
     │  convincingly and concretely the power and potential  │functional programming. The following 
     │  of types in constructing high-quality software.      │"one-liner" solution to the famous    
     │                                                       │Queen Puzzle should offer a glimpse of
     │## Suggestion on learning ATS                          │using combinators in ATS:             
     │                                                       │                                      
     │ATS is feature-rich (like C++). Prior knowledge of     │──────────────────────────────────────
     │functional programming based on ML and imperative      │// (* Solving the Queen Puzzle *) //  
     │programming based on C can be a big plus for learning  │#define N 8 // it can be changed      
     │ATS. In general, one should expect to encounter many   │#define NSOL 10 // it can be changed  
     │unfamiliar programming concepts and features in ATS and│// val () = (((fix qsolve(n: int):    
     │be prepared to spend a substantial amount of time on   │stream(list0(int)) => if(n >          
     │learning them. Hopefully, one will become a superbly   │0)then((qsolve(n-1)*list0_make_intra  
     │confident programmer at the end who can enjoy          │nge(0,N)).map(TYPE{list0(int)})(lam(  
     │implementing large and complex systems with minimal    │$tup(xs,x))=>cons0(x,xs))).filter()(  
     │need for debugging.                                    │lam(xs)=>let val-cons0(x0,xs) = xs    
     │                                                       │in xs.iforall()(lam(i,                
     │## Acknowledgments                                     │x)=>((x0)!=x)&&(abs(x0-x)!=i+1))      
     │                                                       │end)else(stream_make_sing(nil0())))(  
     │The development of ATS has been funded in part by      │N)).takeLte(NSOL)).iforeach()(lam(i,  
     │[National Science Foundation][46] (NSF) under the      │xs)=>(println!("Solution#", i+1,      
     │grants no. CCR-0081316/CCR-0224244, no.                │":"); xs.rforeach()(lam(x) =>         
     │CCR-0092703/0229480, no. CNS-0202067, no. CCF-0702665  │((N).foreach()(lam(i)=>(print_string  
     │and no CCF-1018601. As always, *any opinions, findings,│(ifval(i=x," Q", "                    
     │and conclusions or recommendations expressed here are  │."))));println!()));println!())) //   
     │those of the author(s) and do not necessarily reflect  │──────────────────────────────────────
     │the views of the NSF.*                                 │Try-it-yourself                       
     │                                                       │──────────────────────────────────────
     │                                                       │                                      
     │                                                       │Please find [on-line][47] the entirety
     │                                                       │of this example, which is meant to    
     │                                                       │showcase programming with combinators 
     │                                                       │in ATS.                               
     │───────────────────────────────────────────────────────┴──────────────────────────────────────
     │[thePageRFooterSep]                                                                           
     │                                                                                              
     │──────────────────────────────────────────────────────────────────────────────────────────────
     │This page is created by [Hongwei Xi][48]                                                      
     │with tools including ATS/weboxy, atscc2js and atscc2php.                                      
─────┴──────────────────────────────────────────────────────────────────────────────────────────────

[1]: http://ats-lang.sourceforge.net
[2]: Downloads.html
[3]: Documents.html
[4]: Libraries.html
[5]: Community.html
[6]: Papers.html
[7]: Examples.html
[8]: Resources.html
[9]: Implements.html
[10]: https://sourceforge.net/projects/ats-lang/lists/ats-lang-users
[11]: https://groups.google.com/forum/#!forum/ats-lang-users
[12]: https://groups.google.com/forum/#!forum/ats-lang-devel
[13]: #
[14]: #What_is_ATS
[15]: #What_is_ATS_good_for
[16]: #Suggestion_on_learning_ATS
[17]: #Acknowledgments
[18]: Downloads.html
[19]: Downloads.html#ATS_packages
[20]: Downloads.html#Uninstall_for_ATS
[21]: Downloads.html#Requirements_install
[22]: Downloads.html#Precompiledpack_install
[23]: Downloads.html#Install_source_compile
[24]: Downloads.html#Install_of_ATS2_contrib
[25]: Downloads.html#Install_of_ATS2_include
[26]: Downloads.html#Scripts_for_installing_ATS_Postiats
[27]: Documents.html
[28]: Documents.html#INT2PROGINATS
[29]: Documents.html#ATS2TUTORIAL0
[30]: Documents.html#ATS2FUNCRASH0
[31]: Documents.html#EFF2ATSPROGEX
[32]: Libraries.html
[33]: Community.html
[34]: Community.html#ATS_wikipage
[35]: Community.html#ATS_subreddit
[36]: Community.html#ATS_IRC_channel
[37]: Community.html#ATS_stackoverflow_tag
[38]: Community.html#ATS_QandA_forum
[39]: Community.html#ATS_devel_forum
[40]: Community.html#ATS_mailing_list
[41]: Community.html#JATS_user_group
[42]: #What_is_ATS
[43]: #What_is_ATS_good_for
[44]: #Suggestion_on_learning_ATS
[45]: #Acknowledgments
[46]: http://www.nsf.gov
[47]: https://github.com/githwxi/ATS-Postiats/blob/master/doc/EXAMPLE/MISC/queens_comb.dats
[48]: http://www.cs.bu.edu/~hwxi/
